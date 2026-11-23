# 5.4 EP：专家并行与 All-to-All（Expert Parallelism and All-to-All）

> MoE 把 FFN 复制成 $N_e$ 份、每个 token 只激活 $k$ 份（模型侧数学归 models 4.5）。总参数 $N_{\mathrm{tot}}$ 的显存账让 expert 分卡放置成为容量前提，而不是可选优化——这就是专家并行（expert parallelism, EP）。本章讲 EP 的训练侧系统层，七个主题依次展开。第一，dispatch/combine 两次 all-to-all 的语义与置换矩阵形式化。第二，通信账的命中分布细化：models 4.5 §7 已交付「每 token 每层 $2kd$ 个元素（bf16 合 $4kd$ bytes）、扇出 $\le M$ 节点」的上界，本章按均衡与偏斜两情形细化。第三，capacity factor 的三笔系统账：溢出、丢弃与 padding。第四，负载不均的系统形态（热点卡 = 尾延迟）与 aux-loss-free 偏置的系统读法。第五，EP×TP×DP 的组合账。第六，DeepSeek-V3 的 EP64 部署与两级转发内核。第七，Megatron-Core MoE 的文件级走读 | 前置：models 4.5（MoE 路由数学、容量与均衡——只 recap 不重复推导）、4.1（集合通信账）、5.3（DualPipe 对 all-to-all 的重叠） | 锚点：DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、GShard（[Lepikhin et al., 2020](../references.md)，arXiv:2006.16668，[B]）、Megatron-Core MoE（NVIDIA/Megatron-LM repo `megatron/core/transformer/moe/`，教学参照 + 官方实现——Megatron 三篇论文均不含 MoE/EP，MoE 支持无对应论文）

## 1. EP 的语义：expert 按卡放置，token 按路由流动

一个 671B 参数的 MoE 模型放不进一张卡，怎么办？EP 的规则只有一条： $e$ 张卡各持有 MoE 层的 $N_e / e$ 个 expert（含参数、梯度、优化器状态的全部份额），token 按 router 的 top- $k$ 决策跨卡流动。一层 MoE 的前向数据流：

```
EP 组内一次 MoE 层前向（e = 4 卡、N_e = 8 expert、每卡 2 个、top-2 示意）
card0      card1      card2      card3
  │          │          │          │
  ├─ permute：按 (token, expert) 对把隐状态复制成 k 份（dispatch 矩阵 P）
  ▼
all-to-all dispatch ──► 每卡收到「路由到本地 expert」的 token 副本
  │  卡 j 本地批量：expert FFN（batch = 命中数 R_j，各卡不等）
  ▼
all-to-all combine ◄── 输出按路由权重 w_ti 发回原 token 所在卡并归约（P 的加权转置）
```

通信原语为什么选 all-to-all，而不是 all-gather？all-gather 路线让每卡先收齐全组 $n$ 个 token、本地只算自己的 expert——发送量与路由无关（每卡收 $n \cdot d$ 元素），把入口带宽花在不会命中的 token 上。all-to-all 按路由精确投递，每 token 的发送量正比于 $k$ 而非组内 token 数。 $k \ll e$ 是 MoE 的常态（V3： $k = 8$ 对 $e = 64$），all-to-all 的账严格更优。Megatron-Core 的 `MoEAlltoAllTokenDispatcher` 与 `MoEAllGatherTokenDispatcher` 两族实现正对应这两个选择（§7）。

形式化只需两个对象。设 EP 组内一个 step 的 token 总数为 $n$（组内各卡本地 micro-batch 之和）。路由结果写成选择矩阵 $R \in \lbrace 0, 1 \rbrace^{n \times N_e}$： $R_{ti} = 1$ 当且仅当 token $t$ 的 top- $k$ 集合含 expert $i$，每行恰 $k$ 个 1（top- $k$ 路由与权重 $w_{ti}$ 的模型侧定义归 models 4.5 §2-3，此处只用结果）。把 $R$ 的非零元按行优先排成 $nk \times n$ 的 0/1 矩阵 $P$（第 $(t, i)$ 行只在第 $t$ 列为 1），dispatch 与 combine 就是一对互逆结构：

$$
X^{\prime} = P \thinspace X, \qquad Y = P^{\top} \thinspace (W \odot F(X^{\prime})) \qquad (1)
$$

**符号**： $X \in \mathbb{R}^{n \times d}$：EP 组的输入隐状态（ $d$ = hidden 维）； $P \in \lbrace 0,1 \rbrace^{nk \times n}$：dispatch 的复制-置换矩阵； $F$：逐行施加的 expert FFN（第 $(t,i)$ 行走 expert $i$ 的参数）； $W \in \mathbb{R}^{nk \times d}$：路由权重 $w_{ti}$ 的逐行广播； $Y \in \mathbb{R}^{n \times d}$：合并输出。式 (1) 的两行各说一件事。第一行：dispatch 是「复制加置换」，每个 (token, expert) 对一份副本。第二行：combine 是同一个置换的加权转置，发回原位并按 $w_{ti}$ 求和。这就是「dispatch 矩阵 = one-hot 路由的置换形式化」这句话的意思。系统层的对应物：all-to-all 是没有归约语义的集合通信——每对 rank 互换的数据内容各不相同（对照 4.1 的 all-reduce：同一份数据的逐元素求和）；combine 回程在通信上还是 all-to-all，归约（加权求和）发生在接收之后的本地。

通信量的上界由此直接读出： $X^{\prime}$ 有 $nk$ 行，dispatch 移动 $nkd$ 个元素，combine 同量——每 token 每层 $2kd$ 个元素，bf16 合 $4kd$ bytes。这正是 models 4.5 §7 交付的「每 token 每层 $2kd$ 个元素（bf16 合 $4kd$ bytes）、扇出 $\le M$ 节点」的上界（V3 实例： $2 \times 8 \times 7168 \times 2 \approx 229$ KB，乘 58 个 MoE 层）。本章不重复推导，只在 §2 把这个上界按命中分布细化。

为什么要接受这套跨卡机制？因为 MoE 的经济学是「FLOPs 按 $N_{\mathrm{act}}$ 付、容量按 $N_{\mathrm{tot}}$ 付」（models 4.5 §1），而 $N_{\mathrm{tot}}$ 的上限由显存固定——DeepSeek-V3 的 671B 总参数中 expert 占绝对大头，单卡放不下时按卡分片是唯一出路（V3：671B 总参 / 37B 激活，[DeepSeek-AI, 2024](../references.md)，[A]）。

> **提点**：EP 买的不是速度，是容量——它的全部系统问题都长在这条通信回路上。

本章口径：数字例统一用 DeepSeek-V3 的配置——61 层、256+1 expert、top-8、hidden 7168、EP64 跨 8 节点；换口径处随文声明。

## 2. 通信账细化：命中分布函数

式 (1) 的账按「每 (token, expert) 对一份副本」计——Megatron 式实现确实如此（permute 步先把 token 复制成 $k$ 份再进 all-to-all），此时 $2kd$ 上界就是精确值。再看按目标卡去重的实现：同一卡上的多个目标 expert 只发一份副本、卡内再分发，发送份数变成命中的不同卡数。设 $D_t$ 为 token $t$ 的 top- $k$ expert 命中的不同卡数，则

$$
D_t \le \min(k, e), \qquad \text{每 token 每层字节数} = 2 \thinspace D_t \thinspace d \times 2 \quad (\text{bf16}) \qquad (2)
$$

**均衡情形**。路由均衡、expert 均匀分布时， $k$ 个选择近似独立均匀落在 $e$ 张卡上。卡 $j$ 不被某 token 命中的概率为 $(1 - 1/e)^k$，故期望命中卡数

$$
\mathbb{E}[D] = e \left( 1 - \left(1 - \frac{1}{e}\right)^{k} \right) \qquad (3)
$$

$k \ll e$ 时 Taylor 展开得 $\mathbb{E}[D] \approx k - \binom{k}{2}/e$——几乎不打折。代入 V3 的数字： $e = 64$、 $k = 8$ 时 $\mathbb{E}[D] \approx 7.6$，卡级去重只省约 5%。去重的收益在更粗的粒度上：把「卡」换成「节点」。 $e$ 张卡组在 $e/8$ 个节点上（8 卡/节点）、均衡命中时， $k = 8$ 的期望命中节点数为 $8(1 - (7/8)^8) \approx 5.3$ 个——跨节点流量按节点去重可省约 1/3。如果路由规则直接把节点数压到 $M$（V3 的组限定分发： $M = 4$，选中组内均衡时再算组内命中约 $4(1 - (3/4)^8) \approx 3.6$ 个节点），IB 上的副本数就从逐 expert 的 $k = 8$ 份压到 $\le 4$ 份。**node-limited routing 的通信语义在此**：它不改每 token 的 expert 数，改的是副本的粒度与扇出（口径：一个 token 的副本发往的不同节点数；§6 展开其内核实现）。

**偏斜情形**。上界不保证时间。记卡 $j$ 一个 step 收到的 (token, expert) 副本数

$$
R_j = \sum_{t=1}^{n} \sum_{i=1}^{N_e} R_{ti} \cdot \mathbb{1}[\pi(i) = j] \qquad (4)
$$

**符号**： $\pi(i)$：expert $i$ 所在的卡。总量守恒： $\sum_j R_j = nk$；均衡时 $R_j \approx nk/e$。但 all-to-all 与随后的 expert FFN 都是同步点——一层的时间由最慢的卡决定： $T_{\mathrm{layer}} \propto \max_j R_j$。路由塌缩（models 4.5 §4 的病理学）把 $\max_j R_j$ 推向 $nk$ 的极端，系统读数是 $e$ 倍的层时间劣化。这就是「负载不均不是统计问题、是尾延迟问题」（models 4.5 §7 的口径）的公式化。代入一组数：组内 n = 65536 个 token、k = 2、e = 64 时，均衡值是 nk/e = 2048；若偏斜使某卡的 R_j 涨到 8192，其余 63 张卡都要等它——该层时长被拖到 4 倍。

## 3. Capacity factor 与溢出的系统后果

容量机制的模型侧定义已在 models 4.5 §2.3 交付：每 expert 每 batch 只处理 $C_e = \lceil \mathrm{CF} \cdot T_{\mathrm{batch}} / N_e \rceil$ 个 token（其式 (11)）， $\mathrm{CF} \ge 1$。本章只补系统侧的三笔账。

- **丢弃（drop）**：溢出 token 的该层输出取零，仅靠 residual 穿层（Switch/GShard 口径），被丢 token 的 router 梯度断供。系统读法：drop 不浪费算力，丢的是数据与梯度信号；质量影响随溢出率上升（models 4.5 §2.3 的扫描结论）。
- **padding**：定长 buffer 按 $C_e$ 分配，实际命中不足的槽位以零行补齐、照样过 GEMM。浪费率 $= 1 - \bar{R}/C_e$（ $\bar{R} = T_{\mathrm{batch}}/N_e$ 为均值负载）： $\mathrm{CF} = 1.25$ 且理想均衡时，约 20% 的 expert FFN FLOPs 打在 padding 上。偏斜时两种浪费同时发生：热门 expert 溢出（丢数据），冷门 expert padding（丢算力）。
- **容量的本质**：CF 把「token 命中的随机波动」兑换成「固定的算力过配」——CF 调大则溢出率降、padding 浪费升。GShard 的两个原始设计一并记录（[Lepikhin et al., 2020](../references.md)，arXiv:2006.16668，[B]）：expert 容量取 $O(N/E)$ 量级（ $N$ = batch token 数、 $E$ = expert 数）；双 expert 均满员时，该 token 退化为零向量经 residual 穿层；另以「第二 expert 按其门控权重 $g_2$ 的概率改派」做负载平滑（random routing）。

现代栈还有第三条路：不做容量截断的 dropless 分发，按实际命中数变长组批（grouped GEMM 按 expert 分段）。它消除 drop 与 padding，代价是变长 kernel 与负载监控的复杂度——kernel 层的实现归 gpu 维度，Megatron-Core 的 dispatcher 族（§7）提供多条实现路线。

> **坑**：把 capacity factor 当质量旋钮猛调会两头落空——压向 1 时溢出率抬升（丢数据与 router 梯度），调大则 padding 行的 FLOPs 浪费上升（丢算力）。它是负载波动的缓冲预分配，不是超参。

## 4. 负载不均衡的系统形态

**命题**（木桶）：一层 MoE 的墙上时间由最慢的卡决定，偏斜度直接乘进层时长。式 (4) 的 $\max_j R_j$ 在三个环节制造延迟，按出现顺序：dispatch 的接收端（热点卡的入口带宽与接收 buffer 先饱和）、expert FFN（热点卡的 batch 大、算得慢）、combine（回程再堵一次）。**证明**（要点）：MoE 层前后都是同步点，所以热点卡把 $\max_j R_j / \bar{R}$ 的倍数直接乘进整层的墙上时间，其余 $e - 1$ 张卡空等——偏斜的代价是乘性的，不是加性的。严格化（把系数拆成计算与通信两个）留作练习 4。

工程上有两族对策。模型侧是均衡机制本身：aux loss 把均衡写进优化目标，aux-loss-free 偏置把均衡移出优化目标（两者的数学与梯度分析归 models 4.5 §2.2 与 §3.2，不重复）。这里只记偏置方案的系统读法：V3 的偏置 $b_i$ 只参与 expert 的选择、不进权重、不在计算图里——在系统视角下，它是一个对队列长度的离散积分控制器（超载则扣、欠载则加），把「均衡」从 loss 面降级为控制回路。V3 的部署口径：更新步长 $\gamma = 10^{-3}$（符号步）作用于前 14.3T tokens，最后 500B tokens 关闭；另有 $\alpha = 10^{-4}$ 的 sequence-wise 均衡损失兜底单序列内的极端偏斜（[DeepSeek-AI, 2024](../references.md)，[A]，§4.2 超参节）。系统侧的第二族是复制热点：推理部署把高负载 expert 冗余部署多份（V3 推理框架按在线统计每约 10 分钟检测并调整、prefill 阶段设 32 个冗余 expert，论文 §3.4）——训练侧靠路由控制、推理侧靠容量复制，EPLB 的 serving 侧部署与算法归 inference 7.4（8.1 recap 待 W5），硬件 / 网络拓扑归 cluster 维度。训练侧的可见诊断信号与式 (4) 是同一偏斜的两端：router 的负载统计 $\max_i f_i$ 与分布熵（models 4.5 的病理诊断口径）是模型侧读数， $\max_j R_j$ 是它落到卡上的系统读数。

## 5. EP × TP × DP 的组合账

EP 与已有的并行维正交：TP 切 expert 内部的 matmul（5.1）、PP 切层（5.3）、DP/ZeRO 复制或分片其余状态（4.1/4.2），EP 沿 expert 维再切一刀。world size 的乘积约束扩为 $d \cdot t \cdot c \cdot p \cdot e$（spec §4.3 约定）。EP 组在实现上是 DP 域的子划分：expert 参数的梯度只在「持有同一 expert 的 DP 副本」之间归约，非 expert 参数在全 DP 域归约（Megatron 的 group 树初始化归 6.2 走读）。以 V3 的 2048 卡为例做算术：PP16 × EP64 占掉 1024，剩余因子 2 即 expert 参数的 DP 副本数（dense 参数则按 stage 内 128 卡的 DP 域走 ZeRO-1）。

带宽论证决定 EP 组的拓扑摆位。all-to-all 的域直径就是链路选择。EP 域收进节点内（NVLink，V3 集群 160 GB/s）则通信快，但 $e$ 受节点卡数封顶，每卡 expert 数 $N_e/e$ 上升、expert 显存份额回升；EP 域跨节点（IB，同集群 50 GB/s，恰好 1/3.2）则 $e$ 可以开大（V3 取 64、跨 8 节点），代价是 dispatch/combine 上了 IB。TP 与 EP 在 expert 内部基本互斥：细粒度 expert 的 $d_{\mathrm{ff}}^{(e)}$ 只有 2048（V3），单层 matmul 已经很窄，再切 TP 的收益小、每层两次 all-reduce 的通信照旧——V3 完全不用 TP 的论文口径是显存工程（5.1 章），窄 FFN 的分析是本书注记。叠加账一行：每 step 的 EP 通信 = 每 token 每 MoE 层全程 $\le 8kd$ bytes（前向 $4kd$ ：dispatch+combine 各 $kd$ 个元素 ×2 bytes；反向对称，§7）× token 数 × MoE 层数（按式 (2) 的命中细化修正），与 5.3 式 (11) 的 PP 边界流量、4.1 的 DP all-reduce 同层叠加——跨家族总账与决策图归 5.5。

## 6. DeepSeek-V3 的 EP 部署

V3 的 MoE 结构参数（论文 §2 与超参节）：61 层 Transformer，首 3 层为 dense FFN、其余 58 层为 MoE；每 MoE 层 256 个 routed expert + 1 个 shared expert，expert 中间维 $d_{\mathrm{ff}}^{(e)}$ = 2048， $d$ = 7168，top- $k$ = 8。部署（论文 §3.2 配置段）：16 路 PP × 64 路 EP（跨 8 节点）× ZeRO-1 DP，2048 张 H800；每层的 256 个 routed expert 均布 64 张卡（每卡 4 个）。

路由与硬件的协同设计是本章最干净的一例：256 个 expert 分 8 组、每组 32 个，而每台 8 卡节点恰好容纳 $4 \times 8 = 32$ 个 expert——**一组就是一节点**。组限定分发（每 token 先选 4 组，再在组内选 8 个 expert，模型侧公式归 models 4.5 §3.2）由此等价于「每 token 至多发往 $M = 4$ 个节点」（论文超参节原话）：IB 上的副本粒度从逐 expert 的 8 份压到逐节点的 $\le 4$ 份。配合 §2 式 (3) 的账，跨节点流量相比无限制路由（期望约 5.3 个节点）再降约 1/3（ $5.25 \to 3.6$ 个节点，降幅约 31%）。

通信内核按两级网络定制（论文 §3.2.2）。token 的隐状态先经 IB 发到目标节点上与发送卡同序号的 GPU，到节点后立刻经 NVLink 转发到目标 expert 所在的 GPU，两段带宽（50 与 160 GB/s）流水重叠；回程 combine 对称（NVLink 归拢、IB 回传并累加）。全部 dispatch/combine 内核与计算流重叠，且只占用 20 个 SM（H800 共 132 个；warp specialization、10 条通信通道、按负载动态调 warp 数——细节归 gpu/cluster 维度）。论文的伸缩性结论：该设计下每 token 平均每节点可选 3.2 个 expert，同通信成本最多可扩到 13 个（4 节点 × 3.2）——细粒度 expert 数量因此还留有上调空间；只要通信-计算比维持恒定，all-to-all 开销近零（"near-zero all-to-all communication overhead"，§3.2.1 口径）。

这个「通信-计算约 1:1」（论文 §3.2.1 的自述动机）可以当场复算（本书复算，口径假设随注）：每 token 每 MoE 层，IB 通信按节点去重取 $2M \cdot d \cdot 2$ bytes = $2 \times 4 \times 7168 \times 2 \approx 115$ KB（前向一程的 dispatch + combine），expert 计算取 routed 部分 $6 \cdot k \cdot 3 d \thinspace d_{\mathrm{ff}}^{(e)} = 6 \times 8 \times 3 \times 7168 \times 2048 \approx 2.1 \times 10^{9}$ FLOPs（前向加反向，6ND 口径）——通信-计算比 $\approx 5.4 \times 10^{-5}$ bytes/FLOP；H800 的机器平衡点为 IB 50 GB/s 对 BF16 dense 峰值 989 TFLOPS $\approx 5.1 \times 10^{-5}$ bytes/FLOP——比值 1.1，与论文的「约 1:1」相合（attention 与 shared expert 的计算未计入分母，计入后比值再降）。口径注记：上式分子只是前向一程的通信、分母却是全程（前向加反向）的计算，是非对称 pairing；通信按反向对称翻倍时比值约 2.2，「无重叠时 all-to-all 吃掉约一半算力」的读数相应变为「通信超过计算」——论文 1:1 的分子口径未披露，本书复算只支撑「同量级」结论。1:1 的工程含义：没有任何重叠手段时，all-to-all 的暴露时间（口径：没有被计算重叠盖住的通信时间）与计算同量级——这就是 DualPipe 把 all-to-all 列为一等重叠对象的原因（5.3 §6）。DeepSeek 后来把这套 dispatch/combine 内核开源为 DeepEP 库（deepseek-ai/DeepEP repo），kernel 实现归 gpu/cluster 维度，本书不展开（serving 侧 dispatch / combine 语义与账见 inference 3.3）。

## 7. Megatron-Core MoE 走读

Megatron-Core 的 MoE 支持在 `megatron/core/transformer/moe/`（NVIDIA/Megatron-LM repo，教学参照 + 官方实现；Megatron 三篇论文均不含 MoE/EP，此模块无对应论文——spec §4.1 口径；下列路径与类名核实于 2026-11-17，repo main 分支）。文件级分工与式 (1) 逐段对应：`router.py` 算 top- $k$ 打分与掩码（路由决策）；`token_dispatcher.py` 是分发器的家——`MoEAlltoAllTokenDispatcher` 实现本章的全链：permute（按 $(t, i)$ 对复制排序，式 (1) 的 $P$）→ all-to-all → 本地 expert 计算 → all-to-all → unpermute 加权归约（ $P$ 的加权转置）；同文件另有 `MoEAllGatherTokenDispatcher`（all-gather 换维的另一族实现）与面向 DeepEP 等后端的管理类；`experts.py` 承载按 expert 分段的 grouped GEMM；`moe_layer.py` 把 router、dispatcher、experts 装配成层。反向传播的对应结构顺手一记：combine 的反向就是 dispatch（梯度沿 $P$ 回流到各 expert 输出）、dispatch 的反向是 combine 的未加权版（梯度按 $P^{\top}$ 归拢回 token 原位）——式 (1) 的 vjp 结构（1.2）使两次 all-to-all 在反向再走一遍、方向相反，EP 的通信账因此前向反向对称。教学参照的读法：把式 (1) 与 dispatcher 的五个阶段对齐，生产实现与语义的差距全在工程（变长分批、与 EP 重叠的通信流、容量策略开关）。

教学骨架（教学用，非优化实现）把式 (1) 写成单进程可跑的语义参考：

```python
# 教学用，非优化实现：EP dispatch → expert FFN → combine 的语义（式 (1)）
import torch

def ep_moe_forward(x, topk_idx, topk_w, expert_fns, n_expert, e):
    # x: [n, d]（教学模拟把 EP 组的 token 放在一进程内）
    # topk_idx/topk_w: [n, k]；expert_fns: 长度 n_expert 的 FFN 列表
    # expert i 驻留卡 i * e // n_expert（均布）；e: EP 卡数
    n, d = x.shape
    k = topk_idx.shape[1]
    flat_e = topk_idx.reshape(-1)                  # [n*k]：(token, expert) 对
    flat_x = x.repeat_interleave(k, dim=0)         # dispatch 的复制：P·X
    y_flat = torch.zeros_like(flat_x)
    for i in range(n_expert):                      # all-to-all + 各卡本地批量 + 回程（语义级）
        mask = flat_e == i
        if mask.any():
            y_flat[mask] = expert_fns[i](flat_x[mask])
    y = (y_flat.reshape(n, k, d) * topk_w.unsqueeze(-1)).sum(dim=1)  # 加权转置归约
    return y
```

与生产实现的差异：真实栈的 dispatch/combine 走集合通信库的 all-to-all 或定制 kernel（DeepEP 一路），permute/unpermute 用索引而非物化大矩阵 $P$，且两程 all-to-all 与 expert 计算在流上重叠（5.3 §6 的 DualPipe 结构）；骨架只保留语义。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| dispatch/combine | 式 (1)： $X' = PX$、 $Y = P^{\top}(W \odot F(X'))$ | 置换矩阵与其加权转置；all-to-all 无归约语义、归约在本地 |
| 通信上界 | 每 token 每层 $2kd$ 个元素（bf16 合 $4kd$ bytes）；全程 $\le 8kd$ bytes | models 4.5 §7 已交付；V3 实例约 229 KB × 58 层 |
| 命中细化 | 式 (2)(3)：去重后 $2D_t d$ ；均衡下 $\mathbb{E}[D] = e(1-(1-1/e)^k)$ | 卡级去重收益小（7.6/8）；节点级去重 + 组限定（ $M$ = 4）是主战场 |
| 偏斜 | 式 (4)：层时间 $\propto \max_j R_j$ | 木桶乘性劣化；尾延迟问题 |
| capacity factor | $C_e = \lceil \mathrm{CF} \cdot T_{\mathrm{batch}}/N_e \rceil$（models 4.5 式 (11)） | drop = 丢数据与梯度；padding 浪费 $1 - \bar{R}/C_e$；dropless 归 gpu |
| 均衡的系统读法 | V3： $\gamma = 10^{-3}$ 符号步、末 500B 关闭；seq 级 $\alpha = 10^{-4}$ | 偏置 = 队列长度的积分控制器；推理侧冗余 expert 对照（§4） |
| 组合 | world = $d \cdot t \cdot c \cdot p \cdot e$；V3：2048 = PP16 × EP64 × DP2 | EP 域收节点内 vs 跨节点（NVLink 160 vs IB 50 GB/s）；总账归 5.5；练习 2 |
| V3 部署 | EP64 跨 8 节点、每卡 4 expert；组 = 节点；两级转发；20/132 SM；1:1 复算 | 路由规则为通信服务；DualPipe 重叠归 5.3；练习 2 |

## 练习

1.（推导 · [20]）式 (3) 的完整推导：(a) 由「卡 $j$ 不被命中」的概率出发完成期望命中卡数的推导；(b) 给出 $k \ll e$ 的二阶展开并说明卡级去重为何收益小；(c) 把粒度换成节点（8 卡/节点），对 $e = 64$、 $k = 8$ 算期望命中节点数，并对照组限定 $M = 4$（组内均衡）的 $4(1 - (3/4)^8)$，量化组限定分发省下的 IB 副本比例。校验：(c) 的两个期望值应与正文代入 V3 数字得到的两个读数一致。

2.（计算 · [20]）V3 的 EP 通信账：取每卡每 micro-batch $b \cdot T = 4 \times 4096$ 个 token、 $k = 8$、 $d$ = 7168、bf16、58 个 MoE 层。先写出两口径字节数之比的预测，再动手算。(a) 按逐 (token, expert) 对口径与按节点去重（ $D = 3.6$ ）两口径，算每卡每 micro-batch 的 dispatch+combine 字节数；(b) 对 50 GB/s 的 IB 有效带宽估两种口径的通信时间；(c) 按 §6 的复算口径给出该 micro-batch 的 expert 计算时间下界（BF16 峰值 989 TFLOPS），三种结果对照并说出重叠的必要性量级。校验：两口径之比应等于 k 与节点去重份数之比。

3.（计算 · [30]）容量账：某 MoE 层 $N_e = 64$、一个 step 组内 $T_{\mathrm{batch}} = 2^{20}$ 个 token、 $k = 2$，命中呈均衡分布加噪（各 expert 命中数 $\sim$ Poisson( $\bar{R}$ )，可用代码模拟）。(a) $\mathrm{CF} = 1.25$ 时每 expert 容量 $C_e$ 与期望溢出 token 占比；(b) 同一命中样本下 padding 行的 FLOPs 浪费率；(c) CF 调到 1.1 与 1.5 重算两数，画出「溢出率-浪费率」交换曲线的工作点。校验：溢出率应随 CF 单调降、浪费率随 CF 单调升。

4.（推导 · [30]）木桶的形式化：设层内各卡的计算与通信时间均正比于 $R_j$（系数 $t_e$ 与 $t_c$），证明该 MoE 层的墙上时间下界为 $\max_j R_j \cdot (t_e + t_c)$；定义偏斜度 $\sigma = \max_j R_j / \bar{R}$，说明为何偏斜的代价是乘性，并据此解释「均衡机制的工作点应使 $\sigma$ 贴近 1 而非最小化 aux loss 数值」这一工程口径（提示：结合 models 4.5 §2.2 的「全局最小不在均匀」诚实边界）。校验：偏斜度取 1 时，下界应退化为均衡时长。

## 参考

- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——EP64 × PP16 × ZeRO-1 部署与 2048 H800（§3.2）、61 层 / 256+1 expert / top-8 结构（§2）、组限定与 $M = 4$（超参节）、两级转发内核与 20 SM（§3.2.2）、通信-计算约 1:1（§3.2.1）、偏置控制器 $\gamma$ 与 sequence-wise 兜底（§4.2）、推理侧冗余 expert（§3.4）
- [Lepikhin et al., 2020](../references.md)：GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding，arXiv:2006.16668，[B]（key: gshard）——expert 容量 $O(N/E)$ 与溢出退化、random routing（ $g_2$ 概率改派）、all-to-all 换维原型（§2-3）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——NVIDIA/Megatron-LM repo `megatron/core/transformer/moe/`（教学参照 + 官方实现）：`token_dispatcher.py`（`MoEAlltoAllTokenDispatcher` 等）、`router.py`、`experts.py`、`moe_layer.py`；MoE 支持无对应论文；核实于 2026-11-17
