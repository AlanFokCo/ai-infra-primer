# 5.5 组合策略与总账（Combining Strategies: The Grand Ledger）

> 并行原语到 5.4 为止逐维就位，本章回答组合问题：world size 固定时 $t, c, p, e, d$ 五个度数怎么配、谁贴节点内、谁跨节点。先给拓扑-带宽匹配论证（405B 实例：TP 与 PP 的通信强度差约一个数量级、且只有 TP 压在关键路径上），再给 GPU 数乘积约束与 Megatron group 树的混合进制 rank 映射，然后走读 Llama-3 405B / DeepSeek-V3 / Kimi K2 / OLMo 2 四家真实配置，最后把各维通信叠加成一次 step 的总账并给出吞吐-显存-通信三元权衡的决策流程图 | 前置：4.1（ring all-reduce 账）、4.2-4.4（ZeRO/FSDP 与统一账）、5.1-5.4（TP/SP/CP/PP/EP 各维推导）、models 4.5 §7（EP 每 token 通信上界） | 锚点：Megatron-LM `parallel_state.py`（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、Kimi K2（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]）、OLMo 2 + OLMo-core repo（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]）

## 1. 组合问题：五个度数，一个 world size

4.4 的决策树在 DP 家族内部收敛：ZeRO-3 / FSDP 把三态压到 $16N/d$ 且没有地板，纯数据并行在显存侧够到 1T。真正迫使训练引入模型并行的是另外两堵墙：激活随每 GPU 的 token 数线性涨（1.4 的逐算子账，DP 切不动它），以及扩度后每 GPU 计算量反比下降、梯度通信量不降（4.4 §4 的通信显性化论证）。Part 5 的四个维度各砍这两堵墙的不同侧面：TP 沿宽度切权重与激活（5.1），SP/CP 沿序列维切激活（5.2），PP 沿深度切层（5.3），EP 沿 expert 维切 MoE 权重（5.4）。组合问题由此成形：world size 固定时，五个度数取多少、各自落在网络拓扑的哪一层。

先立符号（全书约定）：并行度数 $t$（TP）、 $c$（CP）、 $p$（PP）、 $e$（EP）、 $d$（DP）； $b$ = micro-batch 条数， $s$ = 序列长， $h$ = hidden size， $L$ = 层数， $m$ = 每 optimizer step 的 micro-batch 数， $N$ = 参数量。另记 $\tau$ = 每 GPU 每 step 处理的 token 数（全局 tokens/batch 除以 $d$）， $L_p$ = 每个 PP stage 持有的层数（均匀划分下 $L/p$）——这两个量把下文所有 per-step 公式收紧。

## 2. 拓扑-带宽匹配：谁进节点，谁出节点

放置规则的原料是两档带宽，先钉死数字（均为公开规格或论文原文口径）：

| 链路 | 每 GPU 带宽 | 口径出处 |
| --- | --- | --- |
| H100 节点内 NVLink | 900 GB/s | 厂商公开规格（4.4 §4 同口径） |
| H800 节点内 NVLink | 160 GB/s | DSv3 论文 §3.2.2（[DeepSeek-AI, 2024](../references.md)，[A]）——「NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)」 |
| 跨节点 IB / RoCE | 400 Gb/s = 50 GB/s | Llama-3 herd（"400 Gbps interconnects between GPUs"，[Grattafiori et al., 2024](../references.md)，[A]，§3.3.1）；OLMo 2 Jupiter 每节点 8 张 400 Gbps IB 卡（[Team OLMo et al., 2025](../references.md)，[A]，§6.1）；Kimi K2 每节点 8×400 Gbps RoCE（[Kimi Team, 2025](../references.md)，[A]，§2.4.1） |

注意两处单位换算：跨节点口径的 400 Gb/s 是 bits，折字节为 50 GB/s；H100 的 900 GB/s 是 NVLink 聚合约数。节点内与跨节点在 H100 代差约 18 倍，在 H800 代只差 3.2 倍——同一个放置规则在两代硬件上的紧迫程度不同，§4 的 DSv3 配置与此直接相关。

### 2.1 通信强度算例：Llama-3 405B 配置下 TP 与 PP 的对比

取 herd 论文 Table 4 第一行（[A]）：8192 GPU、 $t = 8$、 $c = 1$、 $p = 16$、 $d = 64$、tokens/batch = 16M；模型几何 $L = 126$、 $h = 16384$（同论文 Table 3，4.4 练习 1 已引）。step 时间由 MFU 口径反推约 11.6 s（4.4 §5 的数字链）。

TP 的通信（推导归 5.1，此处只叠加）：每层前向 2 次 all-reduce（attention 输出投影后、MLP 第二投影后各一次），消息体为 $[b, s, h]$ 的 bf16 激活，即每 token 每次 $2h$ bytes；反向对称再计 2 次。每 token 每层合计 $8h$ bytes。一个 GPU 只持有本 stage 的 $L_p \approx 126/16 \approx 8$ 层，每 GPU 每 step 的 TP 消息体总量为

$$
C_{\mathrm{TP}} = 8h \thinspace L_p \thinspace \tau = 8 \times 16384 \times 8 \times 262144 \approx 2.75 \times 10^{11}\ \text{bytes} \qquad (1)
$$

**符号**： $\tau = 16\text{M} / 64 = 262144$ tokens——16M 为二进制口径 $2^{24}$，恰为 batch/DP 32 条 × seq 8192 × $d$ 64 之积； $L_p$ 取 8（126 层对 16 stage 是非均匀划分，取整近似）。折成带宽： $2.75 \times 10^{11}$ bytes / 11.6 s ≈ 24 GB/s 消息体口径；按 ring all-reduce 的每 rank 实际搬运量再乘 $2(t-1)/t \approx 1.75$（4.1 式 (2)， $t = 8$）后约 41 GB/s。两个判读：(i) 带宽占比——41 GB/s 放在 50 GB/s 的跨节点链路上独占八成以上，DP/PP/CP 的通信将无处安放；放节点内 NVLink（900 GB/s）只占约 5%；(ii) 关键路径——TP 的 all-reduce 逐层串行、夹在相邻算子之间，没有跨层 overlap 的余地，跨节点时每次还要多吃网络延迟。两条合起来：TP 必须留在节点内， $t$ 的上界就是节点内 GPU 数（8）。

PP 的通信（推导归 5.3）：每个 stage 边界每 micro-batch 前向传一份激活 $[b, s, h]$（bf16， $2bsh$ bytes）、反向传一份梯度（同量）；内部 stage 一进一出两个边界，首末 stage 减半。每 GPU 每 step：

$$
C_{\mathrm{PP}} = 8h \thinspace \tau = 8 \times 16384 \times 262144 \approx 3.4 \times 10^{10}\ \text{bytes} \qquad (2)
$$

折带宽约 3.0 GB/s——50 GB/s 跨节点链路的 6%。且式 (2) 不含 $p$：stage 数翻倍，每个 GPU 仍只接触自己的两个边界，PP 通信量与切多深无关。对照式 (1)：TP/PP 的通信强度比恰好是 $L_p$（本例约 8 倍）再乘关键路径属性——TP 的每字节都堵在前向反向的串行链上，PP 的字节可以被流水调度掩盖（5.3）。

DP 侧登记一句量级：Llama-3 的 FSDP 用法是「切 optimizer 状态与梯度、前向后不再切参数」档（herd 论文 §3.3.2 原话："for model shards we do not reshard after forward computation"，即 ZeRO-2 同族），每 step 每 GPU 约 $4N$ bytes（4.4 §4 表）——405B 下约 $1.6 \times 10^{12}$ bytes，若全部平铺到 50 GB/s 跨机链路需约 32 s，与实测 43% MFU（step ≈ 11.6 s）不自洽。论文未披露其集合通信的拓扑组织方式；工程上的标准出路是把 DP 集合通信做成分层/拓扑感知形式（4.3 式 (1) 的 hybrid 口径是同族机制：跨机字节除以节点内度数），并让梯度归约藏进反向（4.1 的 bucket overlap）。DP 放最外层的原因正是这个调度自由度：它的通信既不对延迟敏感、又可被层级化压缩。

### 2.2 放置规则

四个维度的放置按通信强度与延迟敏感性排序：

- **TP 进节点内**：通信强度 $8h L_p \tau$ / step、逐层串行在关键路径上。 $t$ ≤ 节点内 GPU 数是硬约束（herd 的 $t = 8$ 恰等于每节点 8 卡）。
- **PP 随便跨节点**：通信强度 $8h\tau$ / step 且可调度掩盖；代价不是字节而是气泡（§5 表的结构成本列）。
- **DP/ZeRO 最外层**：字节量大但可 bucket overlap、可分层压缩；度数由剩余 GPU 数决定（§3）。
- **EP 折中**：all-to-all 通信量介于 TP 与 PP 之间（上界每 token 每 MoE 层 $2kh$ 元素，models 4.5 §7），又必须把 expert 权重摊开。两家的取舍对照：DSv3 让 EP 跨 8 个节点（ $e = 64$）并用 node-limited routing 把每 token 的跨节点扇出压到 ≤4 节点（[DeepSeek-AI, 2024](../references.md)，[A]，§3.2.2）；K2 把 $e$ 收到 16（2 个节点），理由是「最小可行 EP」放松 expert 均衡约束并压 all-to-all 开销（[Kimi Team, 2025](../references.md)，[A]，§2.4.2）。
- **CP 按需**：只在长序列阶段开启（herd 第三阶段才开 $c = 16$），ring 还是 Ulysses 的选型归 5.2。

## 3. GPU 数乘积公式与 group 树

五个度数不是自由变量，乘积约束把它们钉在 world size 上。dense 路径（attention 与非 expert 层）：

$$
\mathrm{world} = t \cdot c \cdot p \cdot d \qquad (3)
$$

**符号**：四因子即 §1 的度数约定。MoE 层另有一张 expert 网格：expert 维从 DP 域里再切一刀，Megatron 的实现是两个独立的 rank 生成器——token 网格 $(t, c, p, d)$ 与 expert 网格 $(t', p, e, d_e)$， $t'$ 为 expert 层的 TP 度数（默认与 $t$ 相同）、 $d_e$ 为 expert 数据并行度数：

$$
\mathrm{world} = t' \cdot p \cdot e \cdot d_e, \qquad t \cdot c \cdot d = t' \cdot e \cdot d_e \qquad (4)
$$

第二式是两网格共享同一 world 的相容条件； $c = 1$ 且 $t' = t$ 时退化为 $d = e \cdot d_e$——DP 域恰好是 EP × 剩余 DP。代码侧（`megatron/core/parallel_state.py`，[A]，行级事实核实于 2026-11-17）：`initialize_model_parallel` 先算 `data_parallel_size = world_size // (tp × pp × cp)`（不能整除直接报错），`expert_data_parallel_size = world_size // (etp × ep × pp)`，并断言两个生成器产出的 PP 组必须一致——式 (4) 的相容条件在代码里就是这条断言。

度数之外是 rank 的摆放：同一个 $t = 8$，把 8 个连续 rank 编进一个 TP 组（同节点）还是把 stride 为 512 的 8 个 rank 编进一组（跨 64 个节点），通信成本天差地别。Megatron 的摆放规则是混合进制分解：给定从内到外的维度顺序 $\sigma = (\nu_1, \nu_2, \ldots, \nu_k)$ 与各色度数 $n_1, \ldots, n_k$，

$$
\mathrm{rank} = i_1 + i_2 \thinspace n_1 + i_3 \thinspace n_1 n_2 + \cdots + i_k \prod_{j \lt k} n_j, \qquad i_j \in [0, n_j) \qquad (5)
$$

**符号**： $i_j$ 为 rank 在第 $j$ 维上的坐标。维度 $\nu_j$ 的通信组 = 固定其余坐标、让 $i_j$ 取遍 $[0, n_j)$ 的 rank 集合。性质：越靠内的维度 stride 越小， $\nu_1$ 的组由连续 rank 组成。实现即 `parallel_state.py` 的 `generate_masked_orthogonal_rank_groups` 与 `RankGenerator` 类——前者 docstring 的 $[2, 3, 4]$ 算例按 tp-dp-pp 顺序展开同一公式（读源码注记：该 docstring 的 prose 举例写 order 'tp-pp-dp'，而其式 (1) 的进制链与 $[2, 3, 4]$ 算例均为 tp-dp-pp，源码自身表述不一致，以公式与算例为准）；`initialize_model_parallel` 的默认顺序为 `order="tp-cp-ep-dp-pp"`——TP 最内、PP 最外。herd 论文 Figure 5 的顺序是 [TP, CP, PP, DP]（DP 最外）：两者都把 TP 放在 stride 1 的位置，差别只在 DP 与 PP 谁更靠外——两者的通信都 tolerate 跨节点（§2），谁先谁后是次级决策。group 树的完整初始化走读（含 embedding 组、DP+CP 复合组）归 6.2。

以 16 张卡、 $t = c = p = d = 2$ 为例画全这棵 group 树（按 herd Figure 5 的 [TP, CP, PP, DP] 顺序，ASCII 重绘，出处见参考；该例与论文 Figure 5 的编号一致）：

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

读法：每个 rank 属于每层的一个组，整张树把 16 张卡切成 4 层正交分组；通信原语（all-reduce、all-gather、P2P）全部在组内发生，组间零通信。式 (3)-(5) 合起来回答「world size 给定时度数能取什么」与「取定后每张卡进哪个组」。

整除约束清单（配置时的逐项检查）：world 须被 $t \cdot c \cdot p$（dense）与 $t' \cdot e \cdot p$（expert）整除； $t$ 整除 attention head 数（GQA 下 $t$ 超过 KV head 数时走复制分支，代码形态见 6.1）； $p$ 整除层数或容忍非均匀划分（DSv3 的 61 层对 $p = 16$ 即非均匀，5.3）； $e$ 整除 expert 数（DSv3 256/64 = 4、K2 384/16 = 24）； $c$ 整除序列长。

## 4. 真实配置走读对照

四家的配置与出处（逐项见下文分节；「—」 = 该维未用 / 未披露）：

| 系统 | world（GPU） | $t$ | $c$ | $p$ | $e$ | DP 家族 | 集群与网络 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3 405B 阶段一 | 8192 H100 | 8 | 1 | 16 | — | $d$ = 64，FSDP（不前向后再分片） | RoCE 400 Gb/s/GPU，节点内 NVLink |
| Llama-3 405B 阶段二 | 16384 | 8 | 1 | 16 | — | $d$ = 128 | 同上 |
| Llama-3 405B 阶段三 | 16384 | 8 | 16 | 16 | — | $d$ = 8，seq 131072 | 同上 |
| DeepSeek-V3 | 2048 H800 | 1（不用 TP） | — | 16 | 64（跨 8 节点） | ZeRO-1 | IB 50 GB/s + 节点内 NVLink 160 GB/s |
| Kimi K2 | 节点数为 32 的倍数（每节点 8×H800） | 未提及 | — | 16（virtual stages） | 16 | ZeRO-1 | RoCE 8×400 Gb/s/节点 |
| OLMo 2（7B/13B/32B 发布配置） | ≤1024 H100 | 1 | 1 | 1 | 1 | HSDP | Jupiter：每节点 8×400 Gb/s IB |

**Llama-3 405B**（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，Table 4 与 §3.3）。三阶段同一组 $t = 8$、 $p = 16$：一个 model replica 占 $8 \times 16 = 128$ 卡 = 16 个节点，TP 组恰为一台 8 卡机（Grand Teton 平台，节点内 NVLink）。变化全在 $c$ 与 $d$ 上：阶段二 $d$ 从 64 开到 128，为保 tokens/batch = 16M 不变，batch/DP 从 32 降到 16——MFU 43% → 41% 的归因（论文原文口径）；阶段三序列从 8192 拉到 131072， $c$ 开到 16（2 个节点）顶住 KV 与激活， $d$ 收到 8——MFU 再到 38%，该段的 CP 归因是本书分析（4.4 §5 同口径）。FSDP 用法的档位（切 optimizer 状态与梯度、不前向后再切参数）即 4.4 §4 表中 ZeRO-2 同族行，通信约 $4N$ / step。

**DeepSeek-V3**（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§3.1-3.2）。2048 H800 = 256 节点； $p = 16$、 $e = 64$（EP 域跨 8 个节点）、ZeRO-1 数据并行；论文只给这三个度数，按式 (3)(4) 反推：dense 路径 $d = 2048/16 = 128$，expert 网格 $d_e = 2048/(16 \times 64) = 2$，相容条件 $d = e \cdot d_e$ 成立（本书换算，论文未直接给 DP 度数）。两个结构性选择值得对照。其一，不用 TP——论文口径是显存工程（「without using costly tensor parallelism」，配套 MLA 低秩结构与 FP8 存储，见 1.3 与 5.1）；硬件侧另有一条注脚：H800 的节点内 NVLink 只有 160 GB/s（对 IB 的 3.2 倍），TP 留在节点内的带宽红利比 H100 代的 18 倍薄得多（本书分析）。其二，EP 跨节点用路由侧手段兜底：node-limited routing 把每 token 的目标节点数压到 ≤4、平均每节点 3.2 个 expert（§3.2.2），all-to-all 的调度重叠归 DualPipe（5.3）与定制通信 kernel（5.4）。

**Kimi K2**（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]，§2.4.1-2.4.2）。1.04T 总参、32.6B 激活、384 expert、top-8、61 层的 MoE。并行配置： $p = 16$ 带 virtual stages（interleaved，5.3）、 $e = 16$、ZeRO-1 数据并行，论文未提及 TP 与 CP。model-parallel group 为 $16 \times 16 = 256$ 卡 = 32 节点，整套配置可在任意「32 的倍数」个节点上复用——这是论文明说的设计目标（动态资源下不改并行方案）：节点多时 optimizer 状态按 DP 域摊开，节点少（如 32 节点）时把部分 optimizer 状态 offload 到 CPU。显存账（论文口径）：BF16 参数 + FP32 梯度累加缓冲约 6 TB 摊在 256 卡上，加分布式 optimizer 状态后每卡全部状态约 30 GB，其余显存留给激活（其 §2.4.3）。调度侧两个选择与 5.3/5.4 呼应：拒绝 DualPipe，理由是它把参数与梯度显存翻倍、1T 规模上加并行度补偿太贵（论文口径）；改用 interleaved 1F1B 并增加 warmup micro-batch 数把 EP all-to-all 藏进计算，另把 weight-grad 计算从反向中解耦、与 PP 通信并行——与 zero-bubble 的反向拆分同族（5.3）。

**OLMo 2**（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A] + OLMo-core repo [A]）。对照极：发布配置是纯 DP 家族。OLMo-core 的 7B 训练脚本（`src/scripts/train/OLMo2/OLMo2-7B.py`）只有一项并行配置：`dp_config = HSDP`（bf16 参数、fp32 归约、按 block 包装）；32B 脚本（`OLMo2-32B.py`）同为 HSDP（普通 FSDP 留作注释掉的备选）——TP/PP/CP/EP 均未启用（两个配置文件核实于 2026-11-17）。OLMo 2 论文正文不出现 FSDP 字样，并行配置以 repo 发布配置为一手来源（口径注记）。含义：32B 以下、序列 ≤4096 的尺度上，4.4 的 DP 家族决策树已够用——模型并行是 100B+ 或长上下文才被迫支付的成本。OLMo-core 的 `src/olmo_core/distributed/parallel/` 下另有 data / tensor / pipeline / context / expert 五套并行实现（目录核实于 2026-11-17）：栈里有全套原语与配方实际用了哪个，是两件事。

## 5. 总通信账叠加表

把各维的通信叠到一次 optimizer step、一张 GPU 上。口径声明先行：(i) 字节数为消息体口径、bf16（2 bytes/元素），ring/all-to-all 的每 rank 实际搬运量按 4.1 式 (2) 的 $2(\nu-1)/\nu$ 类因子另乘，结论方向不变；(ii) $b$、 $s$ 取该 rank 的本地口径（开 CP 时 $s$ 已是 $s/c$）；(iii) 各维作用于不同张量、在不同 group 内发生，总账按加法叠加——同一根链路上的各维字节数相加后除以有效带宽，与该配置的单步计算时间比大小，即 4.4 §4 的通信-时间换算。

| 维度 | 每 GPU 每 step 字节数（消息体口径） | 结构成本 | 推导 |
| --- | --- | --- | --- |
| TP（开 SP 同量） | $8h L_p \tau$（每层 4 次集合通信：前向 2 + 反向 2） | 逐层串行、压在关键路径、不可 overlap | 5.1 / 5.2 |
| CP（ring） | $8h_{kv} L_p (c-1) \tau$，GQA 按 KV head 数折算 $h_{kv}$ | 与 attention 计算重叠（环形轮转） | 5.2 |
| CP（Ulysses） | $16h L_p \dfrac{c-1}{c} \tau$（全程口径：前向 $8h$ = QKV 换维 $6h$ + 输出换回 $2h$，反向同量） | 同上 | 5.2 |
| PP | $8h \tau$（内部 stage；首末减半）；与 $p$ 无关 | 气泡占比 $(p-1)/(m+p-1)$（naive 与 1F1B），interleaved 再除 $v$ | 5.3 |
| EP | $\le 8kh \cdot L_{\mathrm{moe},p} \tau$（全程口径：前向 dispatch + combine 共 $2kh$ 元素/token/层，反向对称再一倍；bf16） | all-to-all 抖动、热点卡排队 | 5.4、models 4.5 §7 |
| DP / ZeRO | $\approx 4N$（DP / ZeRO-1 / ZeRO-2 档）； $\approx 6N$（ZeRO-3 / FSDP 全分片档）；hybrid 跨机部分 $2M/d_s$ | 可 bucket overlap、可分层；梯度归约项可 ÷ $m$、参数取回不摊薄 | 4.1-4.4 |

**符号**： $h_{kv}$ = KV 侧宽度（GQA 下 $h \times$ KV head 数 / head 数）； $L_{\mathrm{moe},p}$ = 本 stage 的 MoE 层数； $k$ = 每 token 激活的 expert 数； $v$ = interleaved 的 virtual stage 数； $d_s$ = hybrid 分片组大小；其余同 §1。

代入 Llama-3 405B 阶段一（数字链见 §2.1）：TP ≈ 41 GB/s（ring 口径）留在节点内、PP ≈ 3 GB/s 上跨节点、CP/EP 未开、DP 约 $4N$ bytes/step 靠分层与 overlap 消化——各维字节数 × 链路带宽的匹配关系全部落回 §2 的放置规则。这张表同时给出反向校验：任何一个维度换档（如 $t$ 从 8 加到 16），先把该行字节数按新度数重算、再除以其将要落脚的链路带宽，超线即否决——练习 2 走一遍这个否决过程。

## 6. 吞吐-显存-通信三元权衡：决策流程

把式 (3)(4) 的乘积约束、§2 的放置规则与 §5 的总账合成一个选择流程（ASCII 图；DP 家族内部的 ZeRO/FSDP 深度选择归 4.4 §6，此处不重复）：

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

流程的形状由两股力夹出：显存把度数往大推（每个维度都在除显存），通信把度数往小拉（除式 (5) 的乘积约束外，每开一维都在关键路径或链路上加一行字节）。四家的真实配置（§4）全部落在这个夹缝里：405B dense 全开四维、671B MoE 用 EP 换 TP、1T MoE 把 EP 收小保重叠、32B 以下纯 DP——没有一家为「先进」而多开一个维度。

## 7. 容错与弹性的并行含义

组合度越高，单点故障的爆炸半径越大：TP/CP/PP 组内任何一张卡挂掉，整组停摆（组内是同步的串行依赖）；world size 越大，单位时间内出故障的期望次数越高。量级感：Llama-3 405B 训练的 54 天快照记录了 466 次作业中断（47 次计划内维护、419 次意外），靠自动检出与快速恢复把有效训练时间占比做到 90% 以上（[Grattafiori et al., 2024](../references.md)，[A]，§3.3.4）。checkpoint 的并行度无关保存与弹性续训的账归 7.4。

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

1.（计算）world = 4096 H100（512 节点 × 8 卡），模型 $L = 80$、head 数 64：(a) 取 $t = 8$、 $p = 8$、 $c = 1$，求 $d$ 并按式 (3) 校验整除；(b) 同模型改 MoE、 $e = 32$、 $t' = 8$，按式 (4) 求 $d_e$ 并验证 $d = e \cdot d_e$（ $c = 1$）；(c) 若 naive/1F1B 气泡占比要求 ≤5%， $p = 8$ 时 $m$ 的下界是多少？

2.（计算）把 §2.1 的 TP 算例改到 $t = 16$（跨 2 节点）：重算 $C_{\mathrm{TP}}$ 与 ring 口径的每 rank 字节数，除以 50 GB/s 跨节点带宽得到通信时间，与 11.6 s 的单步计算时间对比；说明「TP 开到 16」在带宽与关键路径两个角度各自为什么不成立（假设 NVLink 域仍是 8 卡）。

3.（推导）从式 (5) 出发，给定 order 为 tp-cp-ep-dp-pp、度数 $(t, c, e, d, p)$：写出任一 rank $r$ 的 DP 组内另一 rank $r'$ 的差 $r' - r$ 的形式（用坐标 $i_{\mathrm{dp}}$ 与 stride 表示）；证明同一 TP 组内的 rank 构成连续整数区间。再答：herd 的顺序 [TP, CP, PP, DP] 下，PP 组的 stride 是多少（用 $t, c$ 表示）？

4.（代码分析）读 `megatron/core/parallel_state.py` 的 `generate_masked_orthogonal_rank_groups` 与 `RankGenerator.get_ranks`：以 world = 16、 $(t, c, d, p) = (2, 2, 2, 2)$、order "tp-cp-dp-pp" 为输入，手推 `get_ranks("dp")` 与 `get_ranks("pp")` 的输出列表，与 §3 的 16 卡图对照；再说明 `initialize_model_parallel` 里 `data_parallel_size` 的计算位置对应式 (3) 的哪一步。

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
