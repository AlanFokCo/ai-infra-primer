# 5.5 组合策略与总账（Combining Strategies: The Grand Ledger）

> 前四章把每种并行单独讲完了（到 5.4 为止）；本章把它们装进同一个集群：world size 固定时， $t, c, p, e, d$ 五个度数各取多少、谁放节点内、谁跨节点。论证分四步。先用 405B 实例算出 TP 与 PP 的通信强度差约一个数量级、且只有 TP 压在关键路径上。再给出 GPU 数乘积约束与 Megatron 的 group 树（rank 怎么分组）。然后对照 Llama-3 405B、DeepSeek-V3、Kimi K2、OLMo 2 四家的真实配置。最后把各维通信叠加成一次 step 的总账，落成一张吞吐-显存-通信三元权衡的决策流程图 | 前置：4.1（ring all-reduce 账）、4.2-4.4（ZeRO/FSDP 与统一账）、5.1-5.4（TP/SP/CP/PP/EP 各维推导）、models 4.5 §7（EP 每 token 通信上界） | 锚点：Megatron-LM `parallel_state.py`（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、Kimi K2（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]）、OLMo 2 + OLMo-core repo（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]）

## 1. 组合问题：五个度数，一个 world size

纯数据并行能走多远？4.4 的决策树给出的答案是：ZeRO-3 / FSDP 把三态压到 $16N/d$，度数越大每卡越省，没有地板——单看显存，DP 家族够到 1T。那为什么不全用 DP？因为还有两堵墙是 DP 拆不动的。第一，激活随每 GPU 的 token 数线性上涨（1.4 的逐算子账），DP 只切模型状态、不切激活。第二，度数扩大后每 GPU 的计算量反比下降，梯度通信量却不降（4.4 §4 的论证），扩到一定程度，通信必然顶到计算前面。

Part 5 的四个维度就是对着这两堵墙来的：TP 沿宽度切权重与激活（5.1）；SP/CP 沿序列维切激活（5.2）；PP 沿深度切层（5.3）；EP 沿 expert 维切 MoE 权重（5.4）。组合问题由此成形：world size 固定时，五个度数各取多少、各自落在网络拓扑的哪一层。

本章符号（全书约定）：并行度数 $t$（TP）、 $c$（CP）、 $p$（PP）、 $e$（EP）、 $d$（DP）； $b$ = micro-batch 条数， $s$ = 序列长， $h$ = hidden size， $L$ = 层数， $m$ = 每 optimizer step 的 micro-batch 数， $N$ = 参数量。另记 $\tau$ = 每 GPU 每 step 处理的 token 数（全局 tokens/batch 除以 $d$）， $L_p$ = 每个 PP stage 持有的层数（均匀划分下 $L/p$）——下文所有 per-step 公式都用这两个量写。本章口径：算账实例统一用 Llama-3 405B 的 8192 GPU 配置（Table 4 第一行），其余系统的数字只在真实配置对照一节出现；五个度数的一组正例就是该配置——TP 8、CP 1、PP 16、EP 未开、DP 64。

## 2. 拓扑-带宽匹配：谁进节点，谁出节点

为什么有的并行必须留在节点内、有的可以放出去？答案由两档带宽决定：节点内 900 GB/s、跨节点 50 GB/s 这个量级差。先把数字固定下来（均为公开规格或论文原文口径）：

| 链路 | 每 GPU 带宽 | 口径出处 |
| --- | --- | --- |
| H100 节点内 NVLink | 900 GB/s | 厂商公开规格（4.4 §4 同口径） |
| H800 节点内 NVLink | 160 GB/s | DSv3 论文 §3.2.2（[DeepSeek-AI, 2024](../references.md)，[A]）——「NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)」 |
| 跨节点 IB / RoCE | 400 Gb/s = 50 GB/s | Llama-3 herd（"400 Gbps interconnects between GPUs"，[Grattafiori et al., 2024](../references.md)，[A]，§3.3.1）；OLMo 2 Jupiter 每节点 8 张 400 Gbps IB 卡（[Team OLMo et al., 2025](../references.md)，[A]，§6.1）；Kimi K2 每节点 8×400 Gbps RoCE（[Kimi Team, 2025](../references.md)，[A]，§2.4.1） |

读这张表要注意两处单位换算。跨节点一栏的 400 Gb/s 是 bits，折成字节是 50 GB/s；H100 的 900 GB/s 是 NVLink 的聚合约数。换算完看倍数：节点内与跨节点在 H100 这代差约 18 倍，在 H800 代只差 3.2 倍。同一个放置规则在两代硬件上的紧迫程度因此不同，§4 的 DSv3 配置与此直接相关。

> **坑**：带宽的 bits 与 bytes 混读——把 400 Gb/s 当成 400 GB/s 会让通信预算虚增 8 倍，后面整章的账全错。本书统一折成字节（GB/s）再比较。

### 2.1 通信强度算例：Llama-3 405B 配置下 TP 与 PP 的对比

TP 和 PP 各要多少带宽？拿真实配置算一遍就清楚了。取 herd 论文 Table 4 第一行（[A]）：8192 GPU、 $t = 8$、 $c = 1$、 $p = 16$、 $d = 64$、tokens/batch = 16M；模型几何 $L = 126$、 $h = 16384$（同论文 Table 3，4.4 练习 1 已引）。该配置下 step 时间约 11.6 s，由 MFU 口径反推（4.4 §5 的数字链）。

路线只有一步叠加：把每层的通信结论乘上本 stage 的层数与每 GPU 的 token 数，再折成带宽。先看 TP。它的通信推导归 5.1，这里只把结果代进来：每层前向有 2 次 all-reduce，分别在 attention 输出投影之后、MLP 第二投影之后；每次的消息体是 $[b, s, h]$ 的 bf16 激活，折合每 token 每次 $2h$ bytes。反向对称，再计 2 次。合计每 token 每层 $8h$ bytes。一个 GPU 只持有本 stage 的 $L_p \approx 126/16 \approx 8$ 层，所以每 GPU 每 step 的 TP 消息体总量为

$$
C_{\mathrm{TP}} = 8h \thinspace L_p \thinspace \tau = 8 \times 16384 \times 8 \times 262144 \approx 2.75 \times 10^{11}\ \text{bytes} \qquad (1)
$$

**符号**： $\tau = 16\text{M} / 64 = 262144$ tokens——16M 为二进制口径 $2^{24}$，恰为 batch/DP 32 条 × seq 8192 × $d$ 64 之积； $L_p$ 取 8（126 层对 16 stage 是非均匀划分，取整近似）。

折成带宽看：消息体口径是 $2.75 \times 10^{11}$ bytes / 11.6 s ≈ 24 GB/s；按 ring all-reduce 的每 rank 实际搬运量再乘 $2(t-1)/t \approx 1.75$（4.1 式 (2)， $t = 8$），约 41 GB/s。这个数字放到两条链路上读。放 50 GB/s 的跨节点链路：独占八成以上，DP/PP/CP 的通信无处安放。放节点内 NVLink（900 GB/s）：只占约 5%。还有第二层问题：TP 的 all-reduce 逐层串行、夹在相邻算子之间，没有跨层 overlap 的余地，跨节点时每次还要多吃网络延迟。两层合起来的结论是：TP 必须留在节点内， $t$ 的上界就是节点内 GPU 数（8）。

再看 PP（推导归 5.3）。每个 stage 边界上，每个 micro-batch 前向传一份激活 $[b, s, h]$（bf16， $2bsh$ bytes），反向传一份梯度，同量；内部 stage 一进一出两个边界，首末 stage 减半。于是每 GPU 每 step：

$$
C_{\mathrm{PP}} = 8h \thinspace \tau = 8 \times 16384 \times 262144 \approx 3.4 \times 10^{10}\ \text{bytes} \qquad (2)
$$

折成带宽约 3.0 GB/s，只占 50 GB/s 跨节点链路的 6%。**命题**（PP 通信量与 p 无关）：式 (2) 里不含 $p$。**证明**（要点）：每个 stage 只与相邻两个 stage 通信；stage 数翻倍时，每个 GPU 仍然只接触自己的两个边界，所以通信量与切多深无关。对照式 (1)：TP 与 PP 的通信强度之比恰好是 $L_p$（本例约 8 倍），这还没算关键路径的差别——TP 的每字节都堵在前向反向的串行链上，PP 的字节可以被流水调度掩盖（5.3）。

DP 侧只登记量级。Llama-3 的 FSDP 用法是「切 optimizer 状态与梯度、前向后不再切参数」这一档——herd 论文 §3.3.2 原话是 "for model shards we do not reshard after forward computation"，即 ZeRO-2 同族。每 step 每 GPU 约 $4N$ bytes（4.4 §4 表）；405B 下约 $1.6 \times 10^{12}$ bytes。这个数字值得停一下：若全部平铺到 50 GB/s 的跨机链路，需要约 32 s，而实测 step ≈ 11.6 s（43% MFU）——账面对不上。论文没有披露其集合通信的拓扑组织方式；工程上的标准出路有两条，一是把 DP 集合通信做成分层、拓扑感知的形式（4.3 式 (1) 的 hybrid 口径是同族机制：跨机字节除以节点内度数），二是让梯度归约藏进反向（4.1 的 bucket overlap）。DP 放在最外层，靠的正是这个调度自由度：它的通信对延迟不敏感，又可以被层级化压缩。

### 2.2 放置规则

为什么 TP 必须进节点、PP 可以跨节点？上面算例的两个数——一个 41 GB/s、一个 3 GB/s——已经是答案的骨架。把四个维度按通信强度与延迟敏感性排序，逐条看：

- **TP 进节点内**。通信强度 $8h L_p \tau$ / step，且逐层串行压在关键路径上。 $t$ ≤ 节点内 GPU 数因此是硬约束——herd 的 $t = 8$ 恰等于每节点 8 卡。
- **PP 随便跨节点**。通信强度 $8h\tau$ / step，还能被调度掩盖；它的代价不在字节，在气泡（§5 表的结构成本列）。
- **DP/ZeRO 最外层**。字节量大，但可 bucket overlap、可分层压缩；度数由剩余 GPU 数决定（§3）。
- **EP 折中**。all-to-all 的通信量介于 TP 与 PP 之间（上界每 token 每 MoE 层 $2kh$ 元素，models 4.5 §7），但 expert 权重又必须摊开。两家给出两种取舍。DSv3 让 EP 跨 8 个节点（ $e = 64$），代价用 node-limited routing 收回：每 token 的跨节点扇出压到 ≤4 个节点（[DeepSeek-AI, 2024](../references.md)，[A]，§3.2.2）。K2 反过来把 $e$ 收到 16（2 个节点），理由是「最小可行 EP」能放松 expert 均衡约束、压低 all-to-all 开销（[Kimi Team, 2025](../references.md)，[A]，§2.4.2）。
- **CP 按需**。只在长序列阶段开启（herd 第三阶段才开 $c = 16$）；ring 还是 Ulysses，选型归 5.2。

## 3. GPU 数乘积公式与 group 树

world size 固定时，五个度数能随便取吗？不能——乘积约束把它们固定在 world size 上。先看 dense 路径（attention 与非 expert 层）：

$$
\mathrm{world} = t \cdot c \cdot p \cdot d \qquad (3)
$$

**符号**：四个因子就是 §1 的度数约定。MoE 层另有一张 expert 网格——expert 维是从 DP 域里再切出来的一刀。Megatron 的实现里这是两个独立的 rank 生成器：token 网格 $(t, c, p, d)$ 与 expert 网格 $(t', p, e, d_e)$，其中 $t'$ 是 expert 层的 TP 度数（默认与 $t$ 相同）， $d_e$ 是 expert 数据并行度数：

$$
\mathrm{world} = t' \cdot p \cdot e \cdot d_e, \qquad t \cdot c \cdot d = t' \cdot e \cdot d_e \qquad (4)
$$

第二式是两个网格共享同一个 world 的相容条件； $c = 1$ 且 $t' = t$ 时它退化为 $d = e \cdot d_e$——DP 域恰好是 EP × 剩余 DP。代码侧的情况（`megatron/core/parallel_state.py`，[A]，行级事实核实于 2026-11-17）：`initialize_model_parallel` 先算 `data_parallel_size = world_size // (tp × pp × cp)`，不能整除直接报错；再算 `expert_data_parallel_size = world_size // (etp × ep × pp)`；并断言两个生成器产出的 PP 组必须一致。式 (4) 的相容条件，在代码里就是这条断言。代入一组真实数字：2048 张卡、16 路 PP 的 MoE 配置下，dense 网格给出 2048 = 16 × 128，expert 网格给出 2048 = 16 × 64 × 2，两式相容。

度数定了，rank 的摆放还没定。同一个 $t = 8$，把 8 个连续 rank 编进一个 TP 组（同一节点内），和把 stride 为 512 的 8 个 rank 编进一组（跨 64 个节点），通信成本天差地别。Megatron 的摆放规则是混合进制分解——所谓混合进制分解，是指把 rank 号写成各维度坐标的变进制展开：给定从内到外的维度顺序 $\sigma = (\nu_1, \nu_2, \ldots, \nu_k)$ 与各维度数 $n_1, \ldots, n_k$，

$$
\mathrm{rank} = i_1 + i_2 \thinspace n_1 + i_3 \thinspace n_1 n_2 + \cdots + i_k \prod_{j \lt k} n_j, \qquad i_j \in [0, n_j) \qquad (5)
$$

**符号**： $i_j$ 是 rank 在第 $j$ 维上的坐标。维度 $\nu_j$ 的通信组，就是固定其余坐标、让 $i_j$ 取遍 $[0, n_j)$ 得到的 rank 集合。**命题**（内层组连续）：越靠内的维度 stride 越小， $\nu_1$ 的组由连续 rank 组成。**证明**（要点）：最内层维度的 stride 是空前缀积 1，第一个坐标取遍其范围即得连续整数区间；外层维度的 stride 是内层度数之积，至少为 2。完整形式化归练习 3。

代码实现是 `parallel_state.py` 的 `generate_masked_orthogonal_rank_groups` 与 `RankGenerator` 类；前者 docstring 的 $[2, 3, 4]$ 算例按 tp-dp-pp 顺序展开同一公式。一处读源码注记：该 docstring 的 prose 举例写的是 order 'tp-pp-dp'，但它的式 (1) 进制链与 $[2, 3, 4]$ 算例都是 tp-dp-pp——源码自身表述不一致，以公式与算例为准。`initialize_model_parallel` 的默认顺序是 `order="tp-cp-ep-dp-pp"`，TP 最内、PP 最外。

对照 herd 论文 Figure 5：它用的顺序是 [TP, CP, PP, DP]，DP 最外。两种顺序都把 TP 放在 stride 1 的位置；差别只在 DP 与 PP 谁更靠外，而两者的通信都能跨节点（§2），谁先谁后是次级决策。group 树的完整初始化走读（含 embedding 组、DP+CP 复合组）归 6.2。

用 16 张卡、 $t = c = p = d = 2$ 把这棵 group 树画全（按 herd Figure 5 的 [TP, CP, PP, DP] 顺序 ASCII 重绘，出处见参考；rank 编号与论文 Figure 5 的实例一致）：

```
rank 编号：rank = i_tp + 2·i_cp + 4·i_pp + 8·i_dp

rank:   0  1 | 2  3 | 4  5 | 6  7 || 8  9 |10 11 |12 13 |14 15
        └TP┘          ← TP 组 {0,1},{2,3},…         stride 1（同节点）
        └──CP──┘      ← CP 组 {0,2},{1,3},…         stride 2
        └──────PP──────┘ ← PP 组 {0,4},{1,5},…      stride 4
        └───────DP────────┘ ← DP 组 {0,8},{1,9},…   stride 8（跨节点）

herd Figure 5 原文实例：GPU0 与 GPU1 同 TP 组、与 GPU2 同 CP 组、
与 GPU4 同 PP 组、与 GPU8 同 DP 组
```

读法：每个 rank 同时属于每一层的一个组；整张树把 16 张卡切成 4 层正交分组。所有通信原语（all-reduce、all-gather、P2P）都只在组内发生，组间零通信。式 (3)-(5) 合起来回答两件事：world size 给定时度数能取什么；取定之后，每张卡进哪个组。

整除约束清单（配置新度数时逐项检查）：world 须被 $t \cdot c \cdot p$（dense 网格）与 $t' \cdot e \cdot p$（expert 网格）整除； $t$ 整除 attention head 数（GQA 下 $t$ 超过 KV head 数时走复制分支，代码形态见 6.1）； $p$ 整除层数，否则容忍非均匀划分（DSv3 的 61 层对 $p = 16$ 就是非均匀，5.3）； $e$ 整除 expert 数（DSv3 是 256/64 = 4、K2 是 384/16 = 24）； $c$ 整除序列长。

## 4. 真实配置走读对照

四家真实配置各开了哪几维？先看总表，再逐家说理由（「—」 = 该维未用 / 未披露）：

| 系统 | world（GPU） | $t$ | $c$ | $p$ | $e$ | DP 家族 | 集群与网络 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3 405B 阶段一 | 8192 H100 | 8 | 1 | 16 | — | $d$ = 64，FSDP（不前向后再分片） | RoCE 400 Gb/s/GPU，节点内 NVLink |
| Llama-3 405B 阶段二 | 16384 | 8 | 1 | 16 | — | $d$ = 128 | 同上 |
| Llama-3 405B 阶段三 | 16384 | 8 | 16 | 16 | — | $d$ = 8，seq 131072 | 同上 |
| DeepSeek-V3 | 2048 H800 | 1（不用 TP） | — | 16 | 64（跨 8 节点） | ZeRO-1 | IB 50 GB/s + 节点内 NVLink 160 GB/s |
| Kimi K2 | 节点数为 32 的倍数（每节点 8×H800） | 未提及 | — | 16（virtual stages） | 16 | ZeRO-1 | RoCE 8×400 Gb/s/节点 |
| OLMo 2（7B/13B/32B 发布配置） | ≤1024 H100 | 1 | 1 | 1 | 1 | HSDP | Jupiter：每节点 8×400 Gb/s IB |

**Llama-3 405B**（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，Table 4 与 §3.3）。三个阶段用同一组 $t = 8$、 $p = 16$：一个 model replica 占 $8 \times 16 = 128$ 卡，即 16 个节点，TP 组恰好就是一台 8 卡机（Grand Teton 平台，节点内 NVLink）。变化全在 $c$ 和 $d$ 上。阶段二把 $d$ 从 64 开到 128；为了保住 tokens/batch = 16M 不变，batch/DP 从 32 降到 16——这是 MFU 从 43% 掉到 41% 的归因（论文原文口径）。阶段三序列从 8192 拉到 131072： $c$ 开到 16（2 个节点）来顶住 KV 与激活， $d$ 收到 8，MFU 再到 38%——这一段的 CP 归因是本书分析（4.4 §5 同口径）。FSDP 用法的档位（切 optimizer 状态与梯度、不前向后再切参数）就是 4.4 §4 表里的 ZeRO-2 同族行，通信约 $4N$ / step。

**DeepSeek-V3**（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§3.1-3.2）。2048 H800，即 256 节点； $p = 16$、 $e = 64$（EP 域跨 8 个节点）、ZeRO-1 数据并行——论文只给这三个度数。按式 (3)(4) 可以反推：dense 路径 $d = 2048/16 = 128$，expert 网格 $d_e = 2048/(16 \times 64) = 2$，相容条件 $d = e \cdot d_e$ 成立（本书换算，论文未直接给 DP 度数）。两个结构性选择值得对照。其一，不用 TP。论文口径是显存工程：「without using costly tensor parallelism」，配套 MLA 低秩结构与 FP8 存储（见 1.3 与 5.1）。硬件侧另有一条注脚：H800 节点内 NVLink 只有 160 GB/s，只是 IB 的 3.2 倍——TP 留在节点内的带宽红利，比 H100 代的 18 倍薄得多（本书分析）。其二，EP 跨节点的压力由路由侧吸收：node-limited routing 把每 token 的目标节点数压到 ≤4、平均每节点 3.2 个 expert（§3.2.2）；all-to-all 的调度重叠归 DualPipe（5.3）与定制通信 kernel（5.4）。

**Kimi K2**（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]，§2.4.1-2.4.2）。模型是 1.04T 总参、32.6B 激活、384 expert、top-8、61 层的 MoE。并行配置： $p = 16$ 带 virtual stages（interleaved，5.3）、 $e = 16$、ZeRO-1 数据并行；论文未提及 TP 与 CP。model-parallel group 是 $16 \times 16 = 256$ 卡，即 32 节点。整套配置可以在任意「32 的倍数」个节点上复用——这是论文明说的设计目标：资源动态变化时不改并行方案。节点多，optimizer 状态按 DP 域摊开；节点少（如 32 节点），把部分 optimizer 状态 offload 到 CPU。显存账（论文口径）：BF16 参数加 FP32 梯度累加缓冲约 6 TB，摊在 256 卡上；再加分布式 optimizer 状态，每卡全部状态约 30 GB，其余显存留给激活（其 §2.4.3）。调度侧的两个选择与 5.3/5.4 呼应。第一，拒绝 DualPipe——它把参数与梯度显存翻倍，1T 规模下要再加并行度来补偿，太贵（论文口径）。第二，改用 interleaved 1F1B 并增加 warmup micro-batch 数，把 EP all-to-all 藏进计算；另把 weight-grad 计算从反向中解耦、与 PP 通信并行——这与 zero-bubble 的反向拆分同族（5.3）。

**OLMo 2**（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A] + OLMo-core repo [A]）。这是另一个极端：发布配置是纯 DP 家族。OLMo-core 的 7B 训练脚本（`src/scripts/train/OLMo2/OLMo2-7B.py`）只有一项并行配置：`dp_config = HSDP`（bf16 参数、fp32 归约、按 block 包装）；32B 脚本（`OLMo2-32B.py`）同样是 HSDP，普通 FSDP 留作注释掉的备选——TP/PP/CP/EP 均未启用（两个配置文件核实于 2026-11-17）。一处口径注记：OLMo 2 论文正文不出现 FSDP 字样，并行配置以 repo 发布配置为一手来源。这组配置的含义是：32B 以下、序列 ≤4096 的尺度上，4.4 的 DP 家族决策树已经够用——模型并行是 100B+ 或长上下文才被迫支付的成本。OLMo-core 的 `src/olmo_core/distributed/parallel/` 下另有 data / tensor / pipeline / context / expert 五套并行实现（目录核实于 2026-11-17）。栈里有全套原语，与配方实际用了哪个，是两件事。

## 5. 总通信账叠加表

一次 step 里，一张 GPU 总共要发多少字节？答案是把各维的通信叠到一次 optimizer step、一张 GPU 上相加。先声明三条口径。第一，字节数按消息体口径、bf16（2 bytes/元素）；ring/all-to-all 的每 rank 实际搬运量要再乘 4.1 式 (2) 的 $2(\nu-1)/\nu$ 类因子，结论方向不变。第二， $b$、 $s$ 取该 rank 的本地口径——开 CP 时 $s$ 已经是 $s/c$。第三，各维作用在不同张量上、在不同 group 内发生，总账按加法叠加：同一根链路上的各维字节数相加，除以有效带宽，与该配置的单步计算时间比大小——这就是 4.4 §4 的通信-时间换算。

| 维度 | 每 GPU 每 step 字节数（消息体口径） | 结构成本 | 推导 |
| --- | --- | --- | --- |
| TP（开 SP 同量） | $8h L_p \tau$（每层 4 次集合通信：前向 2 + 反向 2） | 逐层串行、压在关键路径、不可 overlap | 5.1 / 5.2 |
| CP（ring） | $8h_{kv} L_p (c-1) \tau$，GQA 按 KV head 数折算 $h_{kv}$ | 与 attention 计算重叠（环形轮转） | 5.2 |
| CP（Ulysses） | $16h L_p \dfrac{c-1}{c} \tau$（全程口径：前向 $8h$ = QKV 换维 $6h$ + 输出换回 $2h$，反向同量） | 同上 | 5.2 |
| PP | $8h \tau$（内部 stage；首末减半）；与 $p$ 无关 | 气泡占比 $(p-1)/(m+p-1)$（naive 与 1F1B），interleaved 再除 $v$ | 5.3 |
| EP | $\le 8kh \cdot L_{\mathrm{moe},p} \tau$（全程口径：前向 dispatch + combine 共 $2kh$ 元素/token/层，反向对称再一倍；bf16） | all-to-all 抖动、热点卡排队 | 5.4、models 4.5 §7 |
| DP / ZeRO | $\approx 4N$（DP / ZeRO-1 / ZeRO-2 档）； $\approx 6N$（ZeRO-3 / FSDP 全分片档）；hybrid 跨机部分 $2M/d_s$ | 可 bucket overlap、可分层；梯度归约项可 ÷ $m$、参数取回不摊薄 | 4.1-4.4 |

**符号**： $h_{kv}$ = KV 侧宽度（GQA 下 $h \times$ KV head 数 / head 数）； $L_{\mathrm{moe},p}$ = 本 stage 的 MoE 层数； $k$ = 每 token 激活的 expert 数； $v$ = interleaved 的 virtual stage 数； $d_s$ = hybrid 分片组大小；其余同 §1。

代入 Llama-3 405B 阶段一（数字链见 §2.1）：TP ≈ 41 GB/s（ring 口径）留在节点内，PP ≈ 3 GB/s 上跨节点，CP/EP 未开，DP 约 $4N$ bytes/step 靠分层与 overlap 消化。各维字节数与链路带宽的匹配关系，全部落回 §2 的放置规则。这张表还可以反过来用：任何一维想换档（比如 $t$ 从 8 加到 16），先把该行字节数按新度数重算，再除以它将要落脚的链路带宽，超线即否决。练习 2 会走一遍这个否决过程。

## 6. 吞吐-显存-通信三元权衡：决策流程

度数怎么选？把式 (3)(4) 的乘积约束、§2 的放置规则、§5 的总账合到一起，就是下面这张选择流程图（DP 家族内部的 ZeRO/FSDP 深度选择归 4.4 §6，这里不重复）：

```
输入：模型几何（N, L, h, head 数, expert 数, s_max）、world size、节点内/跨节点带宽
│
① t ← min(节点内 GPU 数, head 数的可行整除值)
│    校验：8h·L_p·τ/step 必须落在 NVLink 域内（§2.1 算例）
│    GQA：t > KV head 数走 KV 复制分支（5.1；代码形态 6.1）
│
② s_max 触发长上下文（KV/激活超限）？──是──► c > 1（ring / Ulysses 选型归 5.2）
│
③ MoE？──是──► e 整除 expert 数且每卡 expert 权重装得下
│    all-to-all 跨节点 ⇒ node-limited routing 或调度重叠（5.4；DSv3/K2 两种答案）
│
④ p = 使「三态分片 + 激活分片 ≤ 单卡容量」的最小深度（显存五项账归 4.4 式 (1)）
│    校验：气泡 (p−1)/(m+p−1) ≤ 容忍度，否则加 m 或开 interleaved（5.3）
│
⑤ d ← world ÷ (t·c·p)；MoE 时 d_e ← world ÷ (t'·e·p)，校验 d·c = e·d_e（式 (4)）
│    ZeRO/FSDP 深度按 4.4 §6 决策树
│
⑥ 回代总账：§5 表逐维字节 ÷ 链路带宽 < 单步计算时间（4.4 §4 口径）；
│    显存五项（4.4 式 (1)）≤ 标称容量并留 buffer 余量
└─ 任一校验失败：回到对应步骤换档，或判定 world size / 硬件不满足
```

这张流程的形状由两股力夹出来：显存把度数往大推（每开一维都在除显存），通信把度数往小拉（除式 (5) 的乘积约束外，每开一维都在关键路径或链路上多加一行字节）。§4 四家的真实配置全落在这个夹缝里：405B dense 全开四维，671B MoE 用 EP 换 TP，1T MoE 把 EP 收小保重叠，32B 以下纯 DP——没有一家为「先进」而多开一个维度。

## 7. 容错与弹性的并行含义

world size 翻倍，故障率跟着涨多少？组合度越高，单点故障的爆炸半径越大：TP/CP/PP 组内任何一张卡挂掉，整组停摆——组内是同步的串行依赖。world size 越大，单位时间内出故障的期望次数也越高。量级上可以参考 Llama-3 405B：54 天的训练快照记录了 466 次作业中断（47 次计划内维护、419 次意外），靠自动检出与快速恢复把有效训练时间占比做到 90% 以上（[Grattafiori et al., 2024](../references.md)，[A]，§3.3.4）。checkpoint 的并行度无关保存与弹性续训，归 7.4 讲。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 放置规则 | TP 进节点（强度 $8hL_p\tau$/step、压关键路径）；PP 跨节点（ $8h\tau$/step、与 $p$ 无关）；DP 最外层（可 overlap 可分层）；EP 折中；CP 按需 | 式 (1)(2)；§2 |
| 带宽事实 | H100：节点内 900 GB/s vs 跨节点 50 GB/s（18×）；H800：160 vs 50（3.2×） | §2 表（厂商规格 + 三家论文口径） |
| 乘积约束 | world $= t \cdot c \cdot p \cdot d = t' \cdot p \cdot e \cdot d_e$；相容条件 $tcd = t'ed_e$ | 式 (3)(4)；`parallel_state.py` |
| rank 映射 | 混合进制分解，内层维度 stride 小 ⇒ TP 组连续 rank；Megatron 默认 tp-cp-ep-dp-pp，herd 用 [TP,CP,PP,DP] | 式 (5)；Figure 5 |
| 真实配置 | Llama-3 405B：TP8×CP1/16×PP16×DP64/128/8；DSv3：PP16×EP64×ZeRO-1 不用 TP；K2：PP16(virtual)×EP16×ZeRO-1、每卡全状态约 30 GB；OLMo 2：纯 HSDP | §4 表（各家一手口径） |
| 总账 | §5 表：各维字节加法叠加、除链路带宽对计算时间校验 | §5 |
| 决策流程 | 显存推大度数、通信拉回；§6 流程图 | §6 |

## 练习

1.（计算 · [20]）world = 4096 H100（512 节点 × 8 卡），模型 $L = 80$、head 数 64：(a) 取 $t = 8$、 $p = 8$、 $c = 1$，求 $d$ 并按式 (3) 校验整除；(b) 同模型改 MoE、 $e = 32$、 $t' = 8$，按式 (4) 求 $d_e$ 并验证 $d = e \cdot d_e$（ $c = 1$）；(c) 若 naive/1F1B 气泡占比要求 ≤5%， $p = 8$ 时 $m$ 的下界是多少？校验：(a) 四个度数的乘积应恰为 4096；(b) 两个网格的度数乘积都应等于 4096。

2.（计算 · [20]）把 §2.1 的 TP 算例改到 $t = 16$（跨 2 节点）：重算 $C_{\mathrm{TP}}$ 与 ring 口径的每 rank 字节数，除以 50 GB/s 跨节点带宽得到通信时间，与 11.6 s 的单步计算时间对比；说明「TP 开到 16」在带宽与关键路径两个角度各自为什么不成立（假设 NVLink 域仍是 8 卡）。先按量级写下你的预测（通信时间会超过单步计算吗），再算。校验：消息体字节数与 t 无关（正文的 TP 通信公式不含 t）；ring 口径每 rank 字节数只涨约 7%——涨的不是字节，是它被压到跨节点链路上的事实。

3.（推导 · [30]）从式 (5) 出发，给定 order 为 tp-cp-ep-dp-pp、度数 $(t, c, e, d, p)$：写出任一 rank $r$ 的 DP 组内另一 rank $r'$ 的差 $r' - r$ 的形式（用坐标 $i_{\mathrm{dp}}$ 与 stride 表示）；证明同一 TP 组内的 rank 构成连续整数区间（即正文「内层组连续」命题的完整形式化）。再答：herd 的顺序 [TP, CP, PP, DP] 下，PP 组的 stride 是多少（用 $t, c$ 表示）？

4.（代码分析 · [30]）读 `megatron/core/parallel_state.py` 的 `generate_masked_orthogonal_rank_groups` 与 `RankGenerator.get_ranks`：以 world = 16、 $(t, c, d, p) = (2, 2, 2, 2)$、order "tp-cp-dp-pp" 为输入，手推 `get_ranks("dp")` 与 `get_ranks("pp")` 的输出列表，与 §3 的 16 卡图对照；再说明 `initialize_model_parallel` 里 `data_parallel_size` 的计算位置对应式 (3) 的哪一步。校验：任一维度的组数乘组大小都应等于 16——每个 rank 在每一维上恰好出现一次。

## 参考

- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——Table 4 三阶段度数与 MFU、Figure 5 的 4D 并行顺序与 16 卡实例、FSDP 用法口径（§3.3.2）、RoCE 400 Gbps（§3.3.1）、466 次中断与 >90% 有效训练时间（§3.3.4）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——2048 H800 集群（§3.1）、PP16 × EP64 × ZeRO-1（§3.2）、不用 TP 的显存口径、NVLink 160 GB/s vs IB 50 GB/s 与 node-limited routing（§3.2.2）
- [Kimi Team, 2025](../references.md)：Kimi K2: Open Agentic Intelligence，arXiv:2507.20534，[A]（key: kimi-k2）——PP16(virtual) × EP16 × ZeRO-1、32 节点倍数弹性、6 TB / 256 卡与每卡约 30 GB、拒绝 DualPipe 的口径、interleaved 1F1B 重叠 EP all-to-all（§2.4.1-2.4.2）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——Jupiter 集群 8×400 Gbps IB/节点（§6.1）；并行配置未在正文披露，以 repo 为准
- [AllenAI, OLMo-core repo](../references.md)（key: olmo-core，[A]）——`src/scripts/train/OLMo2/OLMo2-7B.py` 与 `OLMo2-32B.py`（HSDP 配置）、`src/olmo_core/distributed/parallel/`（五套并行实现）；核实于 2026-11-17
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——`megatron/core/parallel_state.py` 的 group 树实现（`initialize_model_parallel` / `RankGenerator` / `generate_masked_orthogonal_rank_groups`）；代码核实于 2026-11-17
- [Narayanan et al., 2021](../references.md)：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM，arXiv:2104.04473，[A]（key: megatron-cluster）——interleaved schedule（virtual stages，K2 所引机制）与 TP/PP 放置分析的先例
- [Zhao et al., 2023](../references.md)：PyTorch FSDP，arXiv:2304.11277，[A]（key: fsdp）——hybrid sharding 的跨机字节口径（4.3 式 (1)，本章 §2.1 转引）
- [Rajbhandari et al., 2020](../references.md)：ZeRO，arXiv:1910.02054，[A]（key: zero）——ZeRO-1 的 DP 族身份（DSv3/K2 配置的 DP 档）
