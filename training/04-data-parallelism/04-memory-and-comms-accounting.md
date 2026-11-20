# 4.4 显存与通信的统一账（The Unified Memory and Communication Ledger）

> 本章把 Part 4 的账合成一张总账。先给单卡显存的五项分解公式：参数分片、梯度分片、optimizer 分片、激活、通信 buffer。然后填真实数字——7B / 70B / 405B / 1T 四个规模配上 DP / ZeRO / FSDP 各档，在 H100 80 GiB / H200 141 GiB / B200 192 GiB 三档卡上逐格判定装不装得下（含一个算到临界值的例子）。接着算每 step 的通信总量并换算成时间。最后是 MFU 的定义与三家口径对照（Llama-3 直报、DeepSeek-V3 由 GPU-hours 反推、OLMo 2 不报），以及 DP/ZeRO/FSDP 家族内部的策略决策树 | 前置：4.1（DP all-reduce 通信账）、4.2（ZeRO 三阶切分推导）、4.3（FSDP 峰值公式与策略映射）、1.4（激活账与 checkpointing）；models 1.2 式 (5)(8)（ $6ND$ 与 $16N$ 分解） | 锚点：[Grattafiori et al., 2024]（Llama-3 herd，[A]）、[DeepSeek-AI, 2024]（[A]）、[Team OLMo et al., 2025]（[A]）、[Rajbhandari et al., 2020]（ZeRO，[A]）、[Zhao et al., 2023]（FSDP，[A]）

## 1. 总账：单卡显存的五项分解

单卡峰值显存究竟花在哪几处？Part 4 的前三章各算了账的一个切片：4.1 给出「全量复制 + all-reduce」的基线，4.2 把三态逐阶切分，4.3 给出 unshard 暂态。把它们合起来，单卡峰值显存是五项之和。本章口径：显存数字统一按 GiB（2^30 bytes）读，模型规模取 7B / 70B / 405B / 1T 四档，精度取 bf16 参数与梯度加 fp32 master 与矩估计；例外处随文声明。五项之和为：

$$
M_{\mathrm{gpu}} = \underbrace{M_{\mathrm{param}} + M_{\mathrm{grad}} + M_{\mathrm{opt}}}_{\text{三态，切分深度的函数}} + \underbrace{M_{\mathrm{act}}(\kappa)}_{\text{checkpointing 策略的函数}} + M_{\mathrm{buf}} \qquad (1)
$$

**符号**： $N$：参数量； $d$：DP 度数（本书全书约定）； $\kappa$：重计算策略标记（不重算 / selective / full，1.4 的口径）； $M_{\mathrm{buf}}$：通信与框架 buffer（all-gather 输出缓冲、NCCL workspace、allocator 碎片）。三态各项按策略取值（bf16 参数与梯度、fp32 master 与 fp32 矩估计的参考配置，即 models 1.2 式 (8) 的 $16N$ 分解；矩估计降精度时按 $(4+2c)N$ 口径换系数，系数定义见 models 1.2 式 (8)，BF16 矩估计实例见 1.3 §4）：

式 (1) 的五项按时间结构分两类。所谓常驻项，是指在整个 step 内不变的占用（三态的分片部分）。所谓暂态项，是指只在 step 的某个区间存在、用完即释放的占用（FSDP 的 unshard 全量参数与梯度、激活、buffer）；峰值出现在反向经过第一个通信组的时刻——4.3 §3 的时序图就是这个峰值的来源。 $M_{\mathrm{buf}}$ 不进公式，理由不是它小，而是它由实现决定：DDP 的梯度 bucket（默认按 25 MiB 分桶，4.1 走读）、FSDP 的 all-gather 输出缓冲（已计入暂态项）、NCCL 每个 communicator 的固定 workspace、allocator 圆整与碎片都在其中。工程上按 GiB 级安全余量对待；临界判定时它决定成败（§3 的算例）。代入一个数：7B 模型在 DDP 基线下三态合计约 104 GiB，单张 H100 已经装不下——分片存在的理由就在这里，逐项字节数在下一节展开。

| 策略 | $M_{\mathrm{param}}$ | $M_{\mathrm{grad}}$ | $M_{\mathrm{opt}}$ | 三态合计 | 暂态 |
| --- | --- | --- | --- | --- | --- |
| DP / DDP | $2N$ | $2N$ | $12N$ | $16N$ | 无 |
| ZeRO-1 | $2N$ | $2N$ | $12N/d$ | $4N + 12N/d$ | 无 |
| ZeRO-2 / FSDP `reshard_after_forward=False` | $2N$ | $2N/d$ | $12N/d$ | $2N + 14N/d$ | 更新后参数 all-gather 的缓冲一份 |
| ZeRO-3 / FSDP `reshard_after_forward=True` | $2N/d$ | $2N/d$ | $12N/d$ | $16N/d$ | $2P_u(2+n_{\mathrm{pf}})$（4.3 式 (2)） |

表的读法有两点。第一，ZeRO-1 与 ZeRO-2 不动参数项： $2N$ 全量参数始终驻留。只有 ZeRO-3 / FSDP 全分片档把参数也除以 $d$，代价是引入一个与 unit 粒度成正比的暂态项。第二，激活项由 1.4 全额供给：全存时每层 $sbh(34 + 5as/h)$ bytes（Megatron 逐算子账，[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]），full checkpoint 压到每层 $2sbh$，代价是一次额外前向（step 时间 +30–40%，1.4 §3 的实测区间）。本章所有表格只算三态与暂态；激活放不放得下，用总账逐项代入判定——第 3 节给一个算到临界值的完整例子。

## 2. 三态静态账：规模 × 策略 × 度数

把 $N$ 取 7B / 70B / 405B / 1T 四档， $d$ 取 8（单节点）/ 64 / 512 三档，代入上表。三态合计的单卡字节数如下（统一换算 GiB， $2^{30}$ bytes； $16N$ 一列即 DP 基线）：

| 模型 | $16N$（DP 基线） | ZeRO-1 @ $d$=8 | ZeRO-2 @ $d$=8 | ZeRO-3 @ $d$=8 | ZeRO-3 @ $d$=64 | ZeRO-3 @ $d$=512 |
| --- | --- | --- | --- | --- | --- | --- |
| 7B | 104.3 | 35.9 | 24.4 | 13.0 | 1.6 | 0.2 |
| 70B | 1043.1 | 358.6 | 244.5 | 130.4 | 16.3 | 2.0 |
| 405B | 6035.0 | 2074.5 | 1414.4 | 754.4 | 94.3 | 11.8 |
| 1T | 14901.2 | 5122.3 | 3492.5 | 1862.6 | 232.8 | 29.1 |

这张表有三个结构性读数。第一，ZeRO-1/2 的账渐近地由未分片的部分主导，而且饱和来得很早：ZeRO-1 的 $4N + 12N/d$ 在 $d$=8 时是 $5.5N$， $d \to \infty$ 的极限是 $4N$—— $d$ 从 8 加到 512，7B 一列只从 35.9 GiB 降到 26.2 GiB；ZeRO-2 同理，从 $3.75N$ 到极限 $2N$。参数项 $2N$（以及 ZeRO-1 的梯度项 $2N$）是地板；想再降，必须动参数，也就是上 ZeRO-3。第二，ZeRO-3 行随 $d$ 线性下降，没有地板——这就是「显存不够就加卡」在公式层面的形态，代价在通信侧（§4）。第三，1T 那一格可以对照 ZeRO 论文的原句：万亿参数模型在 Adam 类优化器、16-bit 精度下约需 16 TB 状态存储，1024 卡 ZeRO-3 之后每卡约 16 GB（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]，§1）。 $16N$ 的论文原始出处，与上表末行（ $d$=1024 时 14.6 GiB）同口径。

这张表还有一层时间维度的读法。它是静态账，假设一步之内只有分片部分驻留。实际上 ZeRO-3 / FSDP 全分片档的参数以「分片常驻 + 全量暂态」交替存在，峰值账必须把暂态加回来（4.3 式 (2)）。逐层包装时，暂态相当于一份全量参数（bf16）的约 $(2 + n_{\mathrm{pf}})/L$；层数越深，占比越小——深模型反而让全分片的近似更干净。

## 3. 装得下判定：三张卡与临界算例

GPU 侧先定口径：H100 SXM 标称 80 GB HBM3，H200 141 GB HBM3e，B200 192 GB HBM3e（厂商公开规格；标称容量按 GiB = $2^{30}$ bytes 读，与上表同单位）。判定规则：三态 + 暂态 + 激活预算 $\le$ 标称容量，并为 buffer 与碎片留余量。只看三态静态账的初判如下：

> **坑**：把标称的 80 GB 当十进制 GB（10^9 bytes）读，会凭空多出约 7% 的余量——临界判定直接翻转。本章统一按 GiB 读。

| 模型 | 策略 @ $d$ | 三态（GiB） | H100 (80) | H200 (141) | B200 (192) |
| --- | --- | --- | --- | --- | --- |
| 7B | DP（任意 $d$） | 104.3 | ✗ | ✓ | ✓ |
| 7B | ZeRO-1 @ 8 | 35.9 | ✓ | ✓ | ✓ |
| 7B | ZeRO-2 @ 8 | 24.4 | ✓ | ✓ | ✓ |
| 7B | ZeRO-3 @ 8 | 13.0 + 暂态 | ✓ | ✓ | ✓ |
| 70B | ZeRO-1 @ 8 | 358.6 | ✗ | ✗ | ✗ |
| 70B | ZeRO-2 @ 8 | 244.5 | ✗ | ✗ | ✗ |
| 70B | ZeRO-3 @ 8 | 130.4 + 暂态 | ✗ | 临界 | ✓ |
| 70B | ZeRO-3 @ 64 | 16.3 + 暂态 | ✓ | ✓ | ✓ |
| 405B | ZeRO-3 @ 64 | 94.3 + 暂态 | ✗ | ✓ | ✓ |
| 405B | ZeRO-3 @ 512 | 11.8 + 暂态 | ✓ | ✓ | ✓ |
| 1T | ZeRO-3 @ 64 | 232.8 + 暂态 | ✗ | ✗ | ✗ |
| 1T | ZeRO-3 @ 512 | 29.1 + 暂态 | ✓ | ✓ | ✓ |
| 1T | ZeRO-3 @ 1024 | 14.6 + 暂态 | ✓ | ✓ | ✓ |

「临界」这一格要算到激活才算数，现在把它算完。70B 取 80 层、hidden 8192 的几何（Llama-2 70B 量级，[Touvron et al., 2023](../references.md)，arXiv:2307.09288，[A]）；配置为 $d$=8、逐层包装（ $P_u = N/L$ ）、 $n_{\mathrm{pf}} = 1$、micro-batch 一条 4096 序列、激活 full checkpoint：

- 暂态： $2P_u(2 + n_{\mathrm{pf}}) = 6N/L = 5.25 \times 10^{9}$ bytes ≈ 4.9 GiB（4.3 式 (2)）；
- 激活：full checkpoint 每层只存层输入， $2sbhL = 2 \times 4096 \times 8192 \times 80$ bytes ≈ 5.0 GiB；
- 合计： $130.4 + 4.9 + 5.0 = 140.3$ GiB，对标称 141 GiB 只剩约 0.7 GiB 余量——buffer、NCCL workspace 与 allocator 碎片没有任何裕度。

结论：账上成立，工程上按放不下处理。两条出路： $d$ 加倍（70B @ 16：三态 65.2 GiB，余量回到 70+ GiB）；或把分片组收进节点内做 hybrid（4.3 §1，通信量不变，跨机流量除以分片组大小）。

同一张表的下端也值得走一遍——这次咬人的是激活项，不是三态项。7B 在 H100 上用 ZeRO-2 @ 8：三态 24.4 GiB，余量约 55 GiB。激活取 Llama-3.1-8B 几何（ $h$=4096、 $a$=32、 $s$=4096、 $L$=32）。1.4 的逐算子账是每 token 每层约 $7.9 \times 10^{5}$ bytes；全存一条 4096 序列即约 97 GiB，接近余量的两倍——全存直接出局。selective recompute 把每层激活从系数 $34 + 5as/h = 194$ 压到 34（1.4 的口径）：一条序列降到约 17 GiB，装得下，step 时间只加 2–7%。full checkpoint 压到约 1 GiB，代价是 30–40% 时间。同一个模型、同一张卡，激活策略的三档把「余量怎么用」从不可能拉到宽裕——这就是式 (1) 里 $\kappa$ 的实质。

这张表还说明一件常被忽略的事：DP 家族内部就足够覆盖到 1T。覆盖不了的场景，瓶颈不是「装不下」，而是「通信划不来」和「激活随 $BT$ 涨」。后两者的解法是模型并行——TP 沿宽度切、PP 沿深度切，各自对式 (1) 的参数项与激活项再除一个度数，归 P5：TP/PP 的激活切分公式见 5.1 与 5.3（P5 预告），跨家族组合的总账与最终决策图统一在 5.5，本章不重复造。

## 4. 每 step 通信总量账

通信侧有一笔对偶账：每个 optimizer step、每个 GPU 各走多少字节。梯度按 bf16 计，即每参数 2 bytes。表中各行数字的出处是 4.1 的 ring all-reduce 推导与 4.2 的逐阶论证（[Rajbhandari et al., 2020](../references.md)，[A]，§7.2）：

| 策略 | 每 step 通信量（bytes） | 构成 |
| --- | --- | --- |
| DP / DDP | $\approx 4N$ | 梯度 all-reduce $2 \cdot \frac{d-1}{d} \cdot 2N$ |
| ZeRO-1 | $\approx 4N$ | 梯度 reduce-scatter $2N$ + 更新后参数 all-gather $2N$（与 DP 的 all-reduce 等量；与 ZeRO-2 的差别只在梯度是否持久分片，不在通信） |
| ZeRO-2 / FSDP `reshard=False` | $\approx 4N$ | 梯度 reduce-scatter $2N$ + 更新后参数 all-gather $2N$ |
| ZeRO-3 / FSDP `reshard=True` | $\approx 6N$ | 前向参数 all-gather $2N$ + 反向再 all-gather $2N$ + 梯度 reduce-scatter $2N$（ $3\Psi$ 对 $2\Psi$，即 ×1.5） |
| hybrid / HSDP（分片组 $d_s$ 收进节点） | 跨机部分 $\approx 2M/d_s$（ $M$ = 梯度字节数 = $2N$） | 4.3 式 (1)：all-gather 与 reduce-scatter 留节点内，跨机只剩复制组 all-reduce |

两行修正口径。其一，梯度累积摊薄的只是梯度归约项：micro-batch 内不通信，累加边界一次归约（1.5 与 4.3 §5 的 `set_requires_gradient_sync`）。DP 档的 $4N$ 可整体除以 $m$——梯度本就全量驻留，没有额外代价。分片档只有梯度 reduce-scatter 那一项除以 $m$；参数取回（ZeRO-3/FSDP 的前向 + 反向 all-gather，每 micro-batch 共 $4N$）不摊薄。免通信变体还要另付一份未分片全量梯度缓冲（ $2N$ bytes，FSDP 论文 §3.3.4 的两变体口径）。其二，`reduce_dtype=fp32` 时梯度路径字节数翻倍（4.3 §5）：ZeRO-3 档从 $6N$ 涨到 $2N + 2N + 4N = 8N$。通信换归约精度，是一个显式可算的旋钮。

字节数要换成时间才有意义，换算是带宽除法。以 7B、全局 batch 400 万 token 为例（梯度通信约 $28 \times 10^{9}$ bytes ≈ 26.1 GiB）：

- $d$=8 单节点内：NVLink 域约 900 GB/s（H100 节点内公开规格），通信下界约 31 ms。该 $d$ 下每 GPU 分到 50 万 token，每步计算 $6N \times 5 \times 10^{5} \approx 2.1 \times 10^{16}$ FLOPs，按 400 TFLOPS 有效吞吐（口径：实测可持续的单卡 FLOPS，不是峰值）约 52 s。通信只占计算的 0.06%，overlap 毫无压力。
- $d$=512（64 节点）：每 GPU 约 7800 token，计算时间缩到约 0.82 s。通信量不变（ $28 \times 10^{9}$ bytes）。跨节点 IB 按每 GPU 400 Gb/s = 50 GB/s 计——OLMo 2 的 Jupiter 集群就是每 8 卡节点配 8 张 400 Gb/s 网卡（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§6.1）——通信约 0.56 s。藏不住的话，占 step 的 68%。

**命题**（通信显性化）：全局 batch 固定时，DP 家族的每 GPU 通信量与 $d$ 无关，每 GPU 计算量随 $d$ 反比下降；所以存在一个度数，越过它通信时间就顶到计算前面。**证明**（要点）：通信量公式不含该度数（上表）；计算量等于全局 token 数乘每 token FLOPs、再除以 $d$；两式相除即见交点存在。Llama-3 的实测记录正是这个机制：16K GPU 配置把 DP 组从 64 扩到 128，为保全局 tokens/batch 不变，每 DP 组的 batch 从 32 降到 16，MFU 从 43% 掉到 41%（[Grattafiori et al., 2024](../references.md)，[A]，§3.3.2）。ZeRO-3 的 ×1.5 与 hybrid 的 $1/d_s$ 两档，就是这条曲线上的两个修正手柄。

通信账还有一条只在大尺度出现的修正：集合通信的有效带宽随参与 rank 数下降。原因是 straggler 与网络干涉——全局尺度的 all-reduce 达不到小组的效率。FSDP 论文用它论证 hybrid sharding 的另一重收益：复制组的 all-reduce 只在小世界内运行，实测好过全局尺度（[Zhao et al., 2023](../references.md)，[A]，§3.2.2）。前面的换算用的是名义带宽；大度数下的实际通信时间要把这个折扣乘回去。它不改变字节账，只改变字节 ÷ 带宽的时间账。

同一组数字换 ZeRO-3 档再走一遍，就能看清 ×1.5 落在哪。7B、 $d$=512 时通信量涨到约 $42 \times 10^{9}$ bytes，50 GB/s 下要 0.84 s——已经超过 0.82 s 的计算时间，通信与计算完全重叠也藏不下，这一档在该规模下不成立。收成分片组 $d_s$=8 的 hybrid：跨机只剩复制组 all-reduce，约 $2M/d_s = 3.5 \times 10^{9}$ bytes ≈ 70 ms，落回可藏区间。决策树的形状由此而来：显存把策略推向深分片，通信把同一策略推回来，可行解在两股力的交集里。

## 5. MFU：定义与三家口径

MFU 是怎么算的，为什么各家的数字不能直接比？因为它的分子分母各是一个约定。MFU（model FLOPs utilization）定义为

$$
\mathrm{MFU} = \frac{\text{有效吞吐（FLOP/s）}}{\text{硬件峰值（FLOP/s）}} = \frac{6N \cdot \text{每 GPU 每秒 token 数}}{P_{\mathrm{peak}}} \qquad (2)
$$

**符号**：分子按模型 FLOPs 口径计：每 token $6N$（dense；MoE 换 $N_{\mathrm{act}}$，models 1.2 式 (5)），不含重计算多付的前向。这里有一个直接的后果：full checkpoint 下硬件实际执行约 $8N$/token，而 MFU 分子仍记 $6N$——开重算时 MFU 系统性低估硬件占用，差额恰为那 1/3。分母取所用精度的 dense 峰值；sparse 峰值是营销口径的 2 倍，不作分母。反例：拿 sparse 峰值当分母，同一训练的 MFU 会凭空放大一倍——dense 之外没有合法分母。该口径源自 Chowdhery et al. 的用法，Llama-3 herd 论文标注沿用（[Grattafiori et al., 2024](../references.md)，[A]，§3.3.2）。

**Llama-3：直报 BF16 MFU**。herd 论文 Table 4 给出 405B 预训练三个阶段的逐配置数字（[A]）：

| GPUs | TP | CP | PP | DP | seq len | batch/DP | tokens/batch | TFLOPs/GPU | BF16 MFU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

表中两个口径可以当场自验。其一，隐含峰值： $430 / 0.43 \approx 1000$ TFLOPS。H100 SXM 的 BF16 dense 峰值公开规格为 989 TFLOPS，论文的分母就是它； $400/0.41$ 与 $380/0.38$ 同阶，差异是舍入。其二，分子口径： $6 \times 405 \times 10^{9}$ FLOPs/token、16M tokens/batch（取 Table 4 的精确基数 32 seq × 8192 × 64 = $2^{24}$ ）、8192 GPU、430 TFLOPS/GPU，反推 step 时间约 11.6 s，数字链闭合。三行的相对下降：43→41 的归因出自论文（§3.3.2：DP 组从 64 开到 128，每组 batch 从 32 降到 16）；41→38 发生在长上下文阶段（CP 从 1 开到 16，序列从 8K 到 128K）——后一段的 CP 归因是本书分析，论文未逐句解释。

**DeepSeek-V3：不报 MFU，报 GPU-hours，可反推**。论文给出的数字（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）：预训练 14.8T tokens 用 2.664M H800 GPU-hours，折合每万亿 token 180K GPU-hours（Table 1 与 §3 口径）。反推链分两步。第一步，每万亿 token 的模型 FLOPs 为 $6 N_{\mathrm{act}} \times 10^{12}$（ $N_{\mathrm{act}} = 37 \times 10^{9}$ ）。第二步，摊到 $1.8 \times 10^{5} \times 3600$ GPU-秒，得每 GPU 有效吞吐约 $3.43 \times 10^{14}$ FLOPS——同一条推导在 models 1.2 §9 已走一遍。然后选分母：按 H800 的 BF16 dense 峰值 989 TFLOPS（与 H100 同算力，仅互联带宽不同，厂商公开规格）得约 35%；但 V3 的 GEMM 实际跑在自研 FP8 框架下（1.3 §6），若按 FP8 dense 峰值 1979 TFLOPS 计，则约 17%。同一个训练，MFU 随分母口径差一倍。分子侧还有三个系统性偏差： $6N_{\mathrm{act}}$ 不含 attention 的二次项、不含 MTP 模块的前向反向、不含任何重算——三者都让真实硬件占用高于式 (2) 的读数。这些口径注记是本书分析，论文本身未给 MFU 数字。

**OLMo 2：MFU 与 GPU-hours 都不报**。论文给的是另一侧的透明度：完整训练 config、代码与日志全开放（[Team OLMo et al., 2025](../references.md)，[A]）；总算力按 $6ND$ 近似列入评测表（Table 6：7B 为 $1.8 \times 10^{23}$ FLOPs、13B 为 $4.6 \times 10^{23}$）；吞吐只以实例形式出现（其 Figure 15 的每设备 tokens/s 曲线，用于展示垃圾回收对训练吞吐的影响）。也就是说，三家的「效率披露」是三个口径：Llama-3 直接报 MFU 与逐配置 TFLOPs/GPU；DeepSeek-V3 报总 GPU-hours，让读者自行反推；OLMo 2 报总 FLOPs 与原始日志。

MFU 到 100% 之间的差额由什么构成？三家各给出一块拼图。Llama-3 把单步效率与墙上时间分开：MFU 管前者，另有 effective training time 管后者——54 天快照内 466 次作业中断（47 次计划内维护、419 次意外），全程做到 90% 以上的有效训练时间占比（[Grattafiori et al., 2024](../references.md)，[A]，§3.3.4）。集群级真实吞吐还要在 MFU 上再乘这一层。DeepSeek-V3 的缺口在芯片内：通信内核占用 SM（H800 共 132 个 SM，划走其中 20 个）；通信以 CUDA core 执行，直接后果是 Tensor Core 在通信期间闲置（[DeepSeek-AI, 2024](../references.md)，[A]，§3.5.1）。DualPipe 把通信藏进计算气泡（5.3）是软件侧的一半，这 20 个 SM 是硬件侧的现况。把 MFU 当「离天花板多近」的读数时，这些结构性扣除项比算法本身更能解释差距。

横向比较任何两家的 MFU 之前，先核对四个口径开关。第一，分子取 $N$ 还是 $N_{\mathrm{act}}$——MoE 模型两者差一个数量级以上。第二，分子含不含重算 FLOPs——full checkpoint 差 4/3。第三，分母取 dense 还是 sparse 峰值——差 2 倍。第四，分母按哪个精度档——BF16 与 FP8 再差 2 倍。此外，MFU 的时间平均窗口（逐阶段报还是全程平均、含不含 warmup 与评估间隙）各家不一。四个开关全部对齐之前，「A 家 MFU 比 B 家高 5 个点」不构成命题。

> **坑**：引用别家 MFU 而不核对分母的精度档——同一训练按两档峰值算出的 MFU 差一倍。规避：先过一遍上面的四个口径开关。

## 6. DP / ZeRO / FSDP 家族决策树

给定模型、卡型与集群，分片选哪一档？把式 (1) 与 §4 的账落成选择流程（仅限 DP/ZeRO/FSDP 家族；与 TP/PP/EP 的组合决策归 5.5）：

```
三态 + 激活 + buffer 装进单卡？（式 (1) 逐项代入）
│
├─ 16N 装得下 ──► DDP（实现最简；bucket overlap 归 4.1）
│      └─ 想削 optimizer 态占用 / 省一点 ──► ZeRO-1（通信量与 DDP 完全相同，白拿 12N→12N/d）
│
├─ 16N 装不下、2N + 14N/d 装得下 ──► ZeRO-2 / FSDP reshard_after_forward=False
│      （通信仍 ≈4N；Llama-3 405B 的 FSDP 用法即此档）
│
├─ 还不行 ──► ZeRO-3 / FSDP reshard_after_forward=True（16N/d，无地板）
│      ├─ 通信 ×1.5 顶不住跨机带宽 ──► hybrid / HSDP：分片组收进节点，
│      │      跨机流量 ÷ d_s（4.3 式 (1)）；或 reshard_after_forward=int 折中
│      ├─ 峰值仍超 ──► 先动激活：selective → full checkpoint（1.4 决策表；
│      │      代价 2–7% / 30–40% step 时间），再动 micro-batch + 累积（1.5）
│      └─ 加卡也不可行 ──► CPU offload（4.2 的 ZeRO-Offload 账；
│             FSDP 侧 offload_policy，带宽下界账见 4.3 §3）
│
└─ 判定线之外：激活或单步延迟成为瓶颈 ──► 引入 TP/PP（P5；组合总账归 5.5）
```

两条总则收束全章。其一，同样装得下时，取分片更浅的。ZeRO-1/2 的通信与 DDP 相同；ZeRO-3 用 ×1.5 通信换 $16N/d$ 的无地板下降。分片深度是纯的显存-通信权衡，不存在「越深越先进」。其二，度数选择被两条曲线夹住：显存侧 $16N/d$ 随 $d$ 下降；通信侧每 GPU 计算量随 $d$ 下降、通信量不降（§4）。可行 $d$ 的下界由显存决定，上界由通信顶到计算的时刻决定——决策树的所有分支都在这两个界之间取值。

树上每个节点的判定都只有一行账。显存侧：把式 (1) 的五项在候选配置上逐项相加，与目标卡的标称容量比大小。通信侧：把 §4 的字节数除以有效带宽，与该配置的单步计算时间比大小。两次比较都通过的配置才可行——§3 的临界算例与 §4 的 512 卡反例，各是一次否决的示范。真实集群上还有第三组约束没有进树：拓扑（分片组能否收进节点）、故障率（大 $d$ 下中断频率随规模涨，7.4 的 MTBF 账）、全局 batch 下限（每 DP 组 batch 太小时，通信与收敛两端同时恶化——2.3 的 critical batch 与 §4 的通信账在此合流）。这些约束不改变账的形式，只是给可行区间再加几个下界。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 五项总账 | 式 (1) | 三态（切分深度的函数）+ 激活（ $\kappa$ 的函数）+ buffer |
| 三态静态表（第 2 节） | DP $16N$；Z1 $4N{+}12N/d$；Z2 $2N{+}14N/d$；Z3 $16N/d$ | Z1/Z2 的地板是 $2N$；Z3 无地板；1T @ 1024 = 14.6 GiB（ZeRO 论文 16 TB / 1024 口径） |
| 装得下判定 | §3 表（H100 80 / H200 141 / B200 192 GiB） | 70B Z3@8 对 H200 是临界算例：140.3 对 141，工程上按放不下处理 |
| 每 step 通信（第 4 节） | DP/Z1/Z2 ≈ $4N$；Z3/FSDP ≈ $6N$；hybrid 跨机 $2M/d_s$ | 通信量与 $d$ 无关、计算量随 $d$ 反比降 ⇒ 扩度必撞通信；累积 ÷ $m$；fp32 归约梯度项翻倍 |
| MFU | 式 (2)： $6N \cdot$ tok/s ÷ dense 峰值 | Llama-3 直报 38–43%（隐含峰值 ≈989–1000 TFLOPS）；DeepSeek-V3 反推 ≈35%（BF16 峰值口径）/ ≈17%（FP8 口径），论文未报；OLMo 2 不报；比较前先核对四个口径开关 |
| 决策树 | §6 | 同能装下取浅分片；可行 $d$ 区间 = 显存下界 × 通信上界；跨家族组合归 5.5 |

## 练习

1.（计算 · [20]）按 §3 的方法补全 405B 的临界分析：取 126 层、hidden 16384 的几何（Llama-3 405B，herd 论文 Table 3）， $d$=64、逐层包装、 $n_{\mathrm{pf}} = 1$、micro-batch 4096×1、full checkpoint，算三态 + 暂态 + 激活的合计（GiB），判定 H100 80 GiB 与 H200 141 GiB 上的余量，并给出 $M_{\mathrm{buf}}$ 可占的上界。校验：三态项与第 2 节表中 405B 行、d=64 格一致。
2.（计算 · [20]）通信显性化点：7B、全局 batch 400 万 token、有效吞吐 400 TFLOPS、跨机带宽 50 GB/s，求 DP 通信恰好等于计算时间的度数 $d^{\ast}$（通信按 $4N$、与 $d$ 无关）；再把策略换成 ZeRO-3 重算 $d^{\ast}$，解释差异。校验：把求得的度数代回，两档的通信时间与计算时间应分别相等。
3.（推导 · [30]）从 4.1 的 ring all-reduce 量与 4.2 的取回时序出发，重推 §4 表中 ZeRO-3 行的 $6N$（说明三次通信各自的内容与字节系数），并证明「ZeRO-1/2 与 DP 通信量相同」在 reduce-scatter + all-gather 分解下逐步成立。校验：三项之和与第 4 节表的 ZeRO-3 行一致。
4.（计算 · [30]）MFU 口径换算：DeepSeek-V3 的每万亿 token 180K H800 GPU-hours，(a) 用 $6N_{\mathrm{act}}$ 反推每 GPU 有效 FLOPS 并对 989 TFLOPS 求 MFU；(b) 若该训练的 checkpointing 策略相当于多付 1/3 前向，硬件口径的实际占用（硬件 FLOPs ÷ 峰值）是多少？(c) 同一份数据若被某第三方按 FP8 峰值 1979 TFLOPS 引用，MFU 变成多少——说明三个数字为何都「对」。校验：(a) 与第 5 节正文反推出的约 35% 一致；(c) 恰为 (a) 的一半。

## 参考

- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——Table 4 三阶段 MFU / TFLOPs/GPU / 并行度数、43%→41% 的 batch-per-DP 机制解释、FSDP 不前向后再分片的配置口径（§3.3.2）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——14.8T tokens / 2.664M H800 GPU-hours / 每万亿 token 180K GPU-hours（Table 1）； $N_{\mathrm{act}} = 37 \times 10^{9}$ ；2048 H800 集群（§3.1）；MFU 未报，反推口径为本章 §5 注记
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——不报 MFU 的对照口径：6ND 总 FLOPs 表（Table 6）、Jupiter 集群每节点 8×400 Gb/s IB（§6.1）、吞吐实例（Figure 15）
- [Rajbhandari et al., 2020](../references.md)：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，arXiv:1910.02054，[A]（key: zero）—— $2\Psi$ / $3\Psi$ 通信量（§7.2）、1T ≈ 16 TB / 1024 卡 ≈ 16 GB 的原始口径（§1）
- [Zhao et al., 2023](../references.md)：PyTorch FSDP，arXiv:2304.11277，[A]（key: fsdp）——hybrid 跨机流量公式（§3.2.2，本章 4.3 式 (1) 转引）
- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——激活逐算子账 $sbh(34 + 5as/h)$（本章经 1.4 引用其结论）
- [Touvron et al., 2023](../references.md)：Llama 2，arXiv:2307.09288，[A]（key: llama2）——70B 几何（80 层、hidden 8192）出处
