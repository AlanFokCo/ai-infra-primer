# 8.1 集合原语与消息量账（Collective Primitives and Message-Volume Accounting）

> 八个集合原语各搬多少字节——all-reduce 每 rank $2(n-1)/n \cdot M$ 已在 training 4.1 交付，all-gather / reduce-scatter / all-to-all 的账怎么立，每 rank 与每 peer 为什么差 $(n-1)$ 倍？同一个 ncclAllReduce，组在 NVLink 域内走 370.8 GB/s、跨机走 50 GB/s——原语自己挑链路吗？小消息延迟主导、大消息带宽主导，临界 $M_c$ 怎么解，训练梯度与 serving decode 为什么各落一侧？本章 recap training 4.1 §2.2 ring 消息量 + α-β 模型（ $d \to n$ 翻译，归纳证明不重推）与 training P5 / inference 8.1 并行账，交付四块 GPU 侧增量：八原语语义全表与每 rank 消息量账（命题 1 a2a 每 rank/每 peer）、原语组合与 a2a vs all-reduce 流量模式、latency/bandwidth regime 临界（命题 2 $M_c = n\alpha/\beta$）、原语→硬件映射与 NCCL API 接口 | 前置：training 4.1 §2.2（ring 消息量 + α-β，recap 源）、7.1 §6（NVLS）、7.2（有效带宽 370.8 GB/s、ring 映射、节点内 vs 机间 9×）、7.3（peer 映射是域内 transport 底座）；inference 8.1（serving 并行账 recap 源）；下游 8.2（ring 带宽最优性下界证明）、8.3（tree 延迟与 ring/tree 交叉）、8.4（NCCL 模块走读）、9.2（GDR）、10.2（分层 ring 与 fabric）、10.3（a2a 拥塞） | 锚点：NCCL docs（[NVIDIA, NCCL docs](../references.md)，[C] 官方规格披露·一手权威，2.32.3 原语语义）、NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd，nccl.h.in API + transport.cc）；recap 源 training 4.1 / inference 8.1（纯文字指引）——代码锚 2026-11-25 fresh tarball 核实

## 1. 三个问题与本章口径

all-to-all 每 rank 到底发 $(n-1)/n \cdot M$ 还是 $M/n$——两个数差 $(n-1)$ 倍，哪个是「每 rank」、哪个是「每 peer」，自环那份算不算？原语语义表里 all-reduce 的账 training 4.1 已经推过，本章为什么不重推归纳、增量又是什么？小消息延迟主导、大消息带宽主导，这条分界的临界消息尺寸怎么从 α-β 模型解出来，训练梯度与 serving decode 为什么各落一侧？本章回答三个问题：八个原语各自的语义与每 rank 消息量怎么立账？原语语义与硬件链路是什么关系（谁挑链路）？regime 临界 $M_c$ 怎么解、两侧各该怎么优化？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（ $n = 8$，NVLink4，几何 recap 7.1/7.2）；有效带宽取 **NCCL 模型 370.8 GB/s 单向**（7.2 式 (2)，18 × 20.6，raw 25 与模型 20.6 两层账分离）； $\alpha = 2$ μs 为**教学假设口径**（7.2 §3，NVLink P2P 延迟量级假设，非实测）；跨机对比行 400 Gb/s NIC = 50 GB/s 单向（7.2 §7）。贯穿实例 $M = 1$ GB = $10^{9}$ B；通信精度 bf16（梯度/激活口径沿 recap 源）。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，与 models/inference $N$ = 参数量、training $d$ = DP 度严格区分；**recap training 4.1 时 $d \to n$ 翻译**，附口径注）； $M$ = 被归约/传输的全量数据字节（各原语所指见 §3 口径注，分片一律 $M/n$）； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数（recap training 4.1 式 (3)）； $M_c$ = regime 临界消息尺寸（本章局部符号，§5）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用（recap 数值以文字「参数量 7B」「batch 60」给出，不引入裸 $N$ 与 $B$）。

**recap 边界声明**（强制，spec §4.2）：training 4.1 §2.2 式 (2) ring 每 rank 消息量 $2(d-1)/d \cdot M$ 与式 (3) α-β 模型**已交付（含 scatter-reduce 不变式归纳证明），本章只 recap + $d \to n$ 翻译，不重推归纳**；本章增量 = 八原语语义全表 + 每原语消息量账 + 原语→硬件映射 + regime 临界。ring 带宽最优性下界证明归 8.2，ring/tree 延迟交叉归 8.3，NCCL 环构建与 cost model 代码走读归 8.4，a2a 拥塞归 10.3，GDR 归 9.2。**代码 scope 声明**：本章 = 机制片段级（API 签名锚 nccl.h.in + transport 表锚 transport.cc；模块全量走读归 8.4）。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. Recap：ring 消息量与并行账

ring all-reduce 每 rank 发多少、耗时多少？training 4.1 §2.2 已把这两笔账推到数，本章 recap 并翻译符号，不重推归纳。

**ring 每 rank 消息量**（recap training 4.1 式 (2)， $d \to n$ 翻译）：ring 把向量切 $n$ 块、每块 $M/n$，分 scatter-reduce 与 all-gather 两阶段各走 $n-1$ 步，每步每 rank 发 $M/n$；合计每 rank 发送（接收同量）：

$$
C_{\mathrm{rank}} = 2 \thinspace \frac{n-1}{n} \cdot M \hspace{5mu} \approx \hspace{5mu} 2M \qquad (1)
$$

**符号**： $C_{\mathrm{rank}}$ = 一次 all-reduce 每 rank 发送字节数； $M$ = 被归约向量总字节； $n$ = rank 数（training 原式用 $d$ = DP 度，此处译 $n$，口径注：gpu 维度 $n$ 专归 rank 数）。系数 $2(n-1)/n$ 从 $n=2$ 的 1 单调逼近 2，**带宽账与 $n$ 基本无关**——ring 撑到万卡的核心性质（对照星型基线 root 链路 $2(n-1)M$）。**归纳证明已交付 training 4.1 §2.2（scatter-reduce 不变式：第 $s$ 步每块部分和累加 $s+1$ 份贡献），本章不重推**；两阶段各 $(n-1)/n \cdot M$ 的账在 §3 独立登记为 reduce-scatter / all-gather 原语账。代入 $n=8$、 $M=1$ GB： $2 \times 7/8 \times 1 = 1.75$ GB。

**α-β 时间模型**（recap training 4.1 式 (3)， $d \to n$ 翻译）：

$$
T_{\mathrm{ring}} = 2(n-1)\thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \beta \qquad (2)
$$

**符号**： $\alpha$ = 每跳消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $`\beta = 1/\mathrm{BW}_{\mathrm{eff}}`$ = 每字节传输时间。延迟项 $2(n-1)\alpha$ 与 $M$ 无关、随 $n$ 线性涨；带宽项 $2(n-1)/n \cdot M\beta$ 随 $M$ 线性、与 $n$ 基本无关。代入 $n=8$、 $M=1$ GB、 $\beta = 1/(370.8 \times 10^9)$ s/B、 $\alpha = 2$ μs：带宽项 ≈4.72 ms、延迟项 0.028 ms、合计 ≈4.75 ms（与 7.2 §3 逐数一致，带宽主导）。regime 临界 §5 解。

**并行账 recap**（training P5 / inference 8.1，纯文字指引不重推）——各并行维度用哪个原语、消息多大、落哪个 regime，是本章原语表的用例来源：

| 并行维度 | 用的原语 | 消息量级（recap） | regime 与放置 | 指引 |
| --- | --- | --- | --- | --- |
| DP 梯度同步 | all-reduce | bf16 梯度 = 2 × 参数量，7B ≈ 13 GiB | 带宽主导；跨机分层 ring | training 4.1 §2.2 |
| ZeRO/FSDP | reduce-scatter（梯度）+ all-gather（参数） | 各 ≈ 参数量字节，合计与 DP 同构 | 带宽主导 | training 4.2/4.3 |
| TP 层内归约 | all-reduce（serving 每层 2 次前向） | decode 单次 ≈0.61 MB（batch 60） | 延迟主导；留 NVLink 域内 | training 5.1 / inference 8.1 §2 |
| SP/CP | AR = RS + AG 恒等；Ulysses 用 a2a 换维 | 随序列/batch | 视消息尺寸 | training 5.2 §2 |
| EP dispatch/combine | all-to-all（无归约语义） | decode 逐 peer KB 级 | 延迟主导；域内或跨机 Wide-EP | training 5.4 / inference 8.1 §4 |
| PP 层间 | send/recv（点对点，非集合） | 每 stage 边界一份激活 | 延迟；跨机 stage 边界 | training 5.3 / inference 8.1 §3 |

这张表只登记「谁用什么原语、消息多大」，各并行维度的完整推导（TP f/g 共轭、PP 气泡、EP 木桶、ZeRO 通信不变）归训练/推理维度，本章 recap + 指引。两条结构结论提前登记：其一，**AR = RS + AG**（training 5.2 §2 SP 恒等，§4 展开）；其二，**serving 无 backward 把训练侧带宽主导翻成延迟主导**（inference 8.1 §1/§2 recap，§5 用 $M_c$ 给定量读数）。

## 3. 集合原语语义全表与消息量账

八个原语各是什么语义、每 rank 搬多少字节？先立口径，再给全表，最后分「均衡四原语」与「root 四原语」两类读。

**定义**（集合原语）：所谓集合原语（collective primitive），是指通信组内 $n$ 个 rank 共同参与、按固定语义完成数据搬运/归约的通信操作——每个原语由「输入 layout → 输出 layout」唯一确定，与实现算法（ring/tree/…）无关。语义是接口，算法是实现；同一语义可由不同算法达界（8.2/8.3）。

**$M$ 口径注**（全表统一，勿混）： $M$ 恒指「那份完整数据」的总字节，分片恒为 $M/n$；但各原语的 $M$ 所指不同——all-reduce / reduce-scatter / reduce / broadcast 取每 rank（或 root）持有的全量向量；all-gather / gather 取聚合后的全量（每 rank 贡献 $M/n$）；all-to-all / scatter 取每 rank（或 root）持有的输入总量（切 $n$ 份、每份 $M/n$）。此口径与 training 4.1（ $M$ = 被归约向量总字节）及 NCCL API count 约定（§7）一致。

| 原语 | 语义：输入 → 输出 | 每 rank 发送（口径注明） | 典型用例（recap） |
| --- | --- | --- | --- |
| **all-reduce** | 各持 $M$ 字节向量 → 各持逐元素归约结果 $M$ 字节 | $2(n-1)/n \cdot M$（ring，带宽最优 → 8.2） | DP 梯度、TP 层内归约 |
| **all-gather** | 各持 $M/n$ 分片 → 各持全量 $M$ 字节 | $(n-1)/n \cdot M$（ring， $n-1$ 步 × $M/n$） | ZeRO/FSDP 参数取回、SP |
| **reduce-scatter** | 各持 $M$ 字节向量 → rank $r$ 持分片 $r$ 的归约结果 $M/n$ | $(n-1)/n \cdot M$（ring， $n-1$ 步 × $M/n$） | ZeRO/FSDP 梯度归约、SP |
| **all-to-all** | 各持 $M$ 字节（切 $n$ 份）→ rank $r$ 收到各 rank 的第 $r$ 份 | 总发送 $(n-1)/n \cdot M$；**每 peer** $M/n$（直连，不含自环） | EP dispatch/combine、Ulysses |
| **broadcast** | root 持 $M$ 字节 → 全员持 $M$ 字节 | root 至少 $M$；chain 口径每转发者 $M$（tree 分摊 → 8.3） | 参数/初始化广播 |
| **reduce** | 各持 $M$ 字节向量 → root 持归约结果 $M$ 字节 | 非 root $M$（tree/chain）；naive star root 收 $(n-1)M$ | 集中归约（少用） |
| **gather** | 各持 $M/n$ 分片 → root 持全量 $M$ 字节 | 非 root $M/n$（直连）；root 收 $(n-1)/n \cdot M$ | 集中收集 |
| **scatter** | root 持 $M$ 字节 → rank $r$ 持分片 $M/n$ | root $(n-1)/n \cdot M$；各收 $M/n$ | 集中分发 |

全数字代入（ $n=8$、 $M=1$ GB）：all-gather / reduce-scatter / all-to-all 每 rank 发送均 $7/8 = 0.875$ GB；all-to-all 每 peer $1/8 = 0.125$ GB；all-reduce 每 rank $1.75$ GB；gather root 收 0.875 GB、scatter root 发 0.875 GB；naive star reduce root 收 $(n-1)M = 7$ GB。all-gather / reduce-scatter 的 $(n-1)/n \cdot M$ 即 training 4.1 §2.2 两阶段各自的账（ $n-1$ 步、每步每 rank 发 $M/n$，计数即得；不变式归纳已交付该处，不重推），此处独立登记为原语账。

**均衡四原语 vs root 四原语**。all-reduce / all-gather / reduce-scatter / all-to-all 是「均衡四原语」：每 rank 发送系数 $(n-1)/n$（all-reduce 为 $2(n-1)/n$）随 $n$ 趋于常数 1（或 2），**与 $n$ 解耦、无 root 热点、收发对称**——这是它们撑到万卡的结构原因（recap training 4.1 §2.2「带宽账与 $d$ 基本无关」）。broadcast / reduce / gather / scatter 是「root 四原语」：root 链路是潜在热点，但分两种情形。gather/scatter 的 root 负载**有界**——分片尺寸 $M/n$，root 收/发 $(n-1)/n \cdot M \to M$，不随 $n$ 线性涨。broadcast/reduce 的 root 负载**依赖实现**：naive star reduce 每 rank 把全量 $M$ 发给 root，root 收 $(n-1)M$ 随 $n$ 线性涨（training 4.1 §2.1 星型基线：7B 梯度、 $n=512$ 时 root 每步收发各约 6.5 TiB）；binomial tree reduce root 有 $\log_2 n$ 个孩子、各发部分和 $M$，root 收 $M \log_2 n$（ $n=8$ 即 $3M$；教学计数口径，树结构细节归 8.3）；chain broadcast root 只发 $M$ 但延迟 $n-1$ 跳。tree 如何分摊 root 负载、延迟项怎么算，归 8.3。

**命题 1**（all-to-all 每 rank / 每 peer 消息量账）。设每 rank 持 $M$ 字节输入，均分为 $n$ 份、第 $j$ 份发往 rank $j$（含自身）。则每 rank 总发送与每 peer 发送各为：

$$
C_{\mathrm{rank}}^{\mathrm{a2a}} = \frac{n-1}{n} \cdot M, \qquad C_{\mathrm{peer}} = \frac{M}{n} \qquad (3)
$$

**证明**（路线：切分 → 目的地计数 → 自环排除，三步）：其一， $M$ 字节均分 $n$ 份，每份 $M/n$。其二，第 $j$ 份目的地是 rank $j$，故 rank $r$ 向每个 peer $j \ne r$ 各发一份 $M/n$，共 $n-1$ 份，总发送 $(n-1) \cdot M/n$。其三，自身那一份（ $j = r$）留本地、不经链路发送，不计入发送量。∎ 正例： $n=8$、 $M=1$ GB，每 rank 发 0.875 GB、每 peer 0.125 GB。边界反例（自环）：若把自身份计入发送，得 $n \cdot M/n = M$，**多计 $M/n$**—— $n=8$、 $M=1$ GB 时多计 125 MB。「每 rank」与「每 peer」两个口径差 $(n-1)$ 倍：每 rank $(n-1)/n \cdot M$ 是对 $n-1$ 个 peer 的发送总和，每 peer $M/n$ 是单份尺寸，两者都不含自环。

> **坑**：a2a 每 rank / 每 peer 口径混用。触发条件：把 a2a 每 rank 发送写成 $M/n$（拿每 peer 当每 rank），或把自环计入写成 $M$；现象：消息量账差 $(n-1)$ 倍或多计 $M/n$，带宽/延迟账整笔错；规避：每 rank = $(n-1)$ × 每 peer，自环份留本地不发；对账 ncclAlltoAll 的 count 约定（count = 每 peer 元素数、sendbuff = $n \times$ count，§7）。

## 4. 原语组合与流量模式

all-reduce 为什么能拆成 reduce-scatter + all-gather？a2a 和 all-reduce 的流量模式差在哪？

**组合恒等**（recap training 4.1 §2.2 / training 5.2 §2，不重推）：all-reduce = reduce-scatter + all-gather。语义追踪：reduce-scatter 让 rank $r$ 拿到分片 $r$ 的全归约和（每 rank 持一个全和分片），all-gather 把各分片的全和分发到全员——复合即「各持全量归约结果」的 all-reduce 语义。消息量恒等： $(n-1)/n \cdot M + (n-1)/n \cdot M = 2(n-1)/n \cdot M$，与式 (1) 逐项一致。语义组合的归纳证明已交付 training 4.1 §2.2，此处只登记恒等与原语级读法。ZeRO 读法（training 4.2 recap）：梯度 reduce-scatter + 参数 all-gather，两段分别落在「梯度归约」与「参数同步」两个事件上，每 rank 通信量与 DP 基线 all-reduce 严格同构——通信预算一文未涨，换优化器状态显存 $\div n$。

**a2a vs all-reduce 流量模式**。两者通信图不同（ $n=4$ 示意）：

```
ring all-reduce（每 rank 只与 2 邻居通信）：     all-to-all（完全图，每 rank 向 n-1 peer）：
      r0 ──► r1                                     r0 ──► r1, r2, r3
      ▲        │                                    r1 ──► r0, r2, r3
      │        ▼                                    r2 ──► r0, r1, r3
      r3 ◄── r2                                     r3 ──► r0, r1, r2
   2(n-1) 步转发，邻居链路持续均匀负载             单跳直达，每 peer M/n，全 bisection 需求
```

ring all-reduce 每 rank 只与 prev/next 两个邻居通信，走 $2(n-1)$ 步转发，邻居链路承载持续均匀负载；all-to-all 是完全图，每 rank 同时向 $n-1$ 个 peer 各发 $M/n$，单跳直达、无多步转发。NVSwitch 非阻塞对任意排列线速（recap 7.2 §4）：均衡 a2a 是一个排列（每 rank 收发均 $(n-1)/n \cdot M$），域内总注入不变，switch 任意对可满带宽。a2a 的压力在**全 bisection 需求与突发性**，以及偏斜路由（热 expert）造成的 incast（多对一同时涌入）——拥塞账归 10.3。语义差一条：all-reduce 有归约语义（数据被求和改变），a2a 是纯数据搬运（无归约；EP combine 的加权求和在接收后本地做，training 5.4 recap）。

## 5. α-β regime：延迟主导 vs 带宽主导

小消息延迟主导、大消息带宽主导，分界在哪？从式 (2) 的两项解临界。

**命题 2**（regime 临界消息尺寸）。令式 (2) 延迟项等于带宽项，解得临界：

$$
M_c = \frac{n \thinspace \alpha}{\beta} \qquad (4)
$$

当 $M \lt M_c$ 延迟项主导（latency-bound）， $M \gt M_c$ 带宽项主导（bandwidth-bound）；带宽项与延迟项之比恰为 $M/M_c$。

**证明**（路线：令两项相等 → 解 $M$ → 比值，三步）：其一，令 $2(n-1)\alpha = 2(n-1)/n \cdot M\beta$。其二，两边约去 $2(n-1)$ 得 $\alpha = M\beta/n$，即 $M_c = n\alpha/\beta$。其三，带宽项/延迟项 $= [2(n-1)/n \cdot M\beta] / [2(n-1)\alpha] = M\beta/(n\alpha) = M/M_c$，随 $M$ 线性。∎

全数字代入（ $n=8$、 $\alpha = 2$ μs 教学假设、 $\beta = 1/(370.8 \times 10^9)$ s/B）： $M_c = 8 \times 2 \times 10^{-6} \times 370.8 \times 10^9 = 5.93 \times 10^6$ B ≈ **5.93 MB**。 $M = 1$ GB 时比值 $M/M_c \approx 168.6$，带宽项主导（与 §2 实例 4.72 ms vs 0.028 ms 一致）。

**跨维度闭合**（两个 recap 源各落 $M_c$ 一侧）：训练侧 7B bf16 梯度 $M \approx 13$ GiB ≈ 13.96 GB ≈ $2353 \times M_c$，牢固带宽主导（training 4.1 §2.2 结论 recap，注意力全在有效带宽）；serving 侧 TP decode 单次归约 $M = 2 \times 60 \times 5120 = 614400$ B ≈ 0.61 MB = $0.104 \times M_c$（inference 8.1 式 (1) recap，batch 60、 $d_{\mathrm{model}} = 5120$、bf16），延迟主导。**同一个 $n=8$ NVLink 域，训练梯度与 serving decode 各落 $M_c$ 两侧**——这就是「训练看带宽、serving 看延迟」的定量读数，不是经验习惯。DDP 桶 25 MiB（training 4.1 §3.1 recap）大于 $M_c$，落带宽 regime；serving decode 0.61 MB 小于 $M_c$，落延迟 regime。

后果（优化方向随 regime 分）：大消息 regime 注意力在有效带宽——ring 带宽最优（8.2 下界证明）、NVLS 把每 rank 发送降到 $M$（7.1 §6）；小消息 regime 注意力在延迟项——压跳数：tree $O(\log n)$ 延迟（8.3）、NVLS 约 2 跳（7.1 §6）、custom all-reduce kernel（inference 8.1 §2 recap）。**口径注**： $M_c$ 是 ring 算法**内部**延迟项与带宽项的临界，与 8.3 的 ring/tree **算法间**交叉临界不同——后者来自实测有效带宽差 $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$，理想 α-β 模型两者带宽项相同、无交叉（training 4.1 已交付结论 recap），两个临界勿混。

a2a 的时间模型另有一笔（inference 8.1 §4 recap，共享出口口径）： $T_{\mathrm{a2a}} \approx \alpha_{\mathrm{a2a}} + (n-1)/n \cdot M\beta$， $\alpha_{\mathrm{a2a}}$ 为单次 a2a 固定延迟（launch + 组同步 + 单跳直达，非 ring 的 $2(n-1)$ 跳）；逐 peer 交换在 fabric 上并行，聚合时间不串行累加 $n-1$ 份。代入 $n=8$、 $M=1$ GB：带宽项 $(7/8) \times 10^9 / (370.8 \times 10^9) \approx 2.36$ ms。a2a 单跳直达、延迟项小，但逐 peer 消息小（每 peer $M/n$），小消息下每条消息的固定开销占比高——EP decode 逐 peer KB 级延迟主导（inference 8.1 §4 recap）正是此读数。

## 6. 原语 → 硬件映射：谁挑链路

同一个 ncclAllReduce，组在域内走 NVLink、跨机走 RDMA——原语自己挑链路吗？不挑。**原语是语义，链路是放置**：通信组（communicator）的 rank 摆在哪些 GPU 上，决定走哪条物理路径；原语语义不变，transport 随放置变。NCCL 的 transport 表（`src/transport.cc` L15-20，[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1）只有四项 `ncclTransports[] = {p2p, shm, net, collNet}`，NVLS 独立路径不在此表：

| 通信范围 | 物理路径 | transport | 带宽口径 |
| --- | --- | --- | --- |
| NVLink 域内 | GPU → NVSwitch → GPU（PATH_NVL） | **p2p**（peer 映射直访，7.3） | 370.8 GB/s 模型单向（7.2 式 (2)） |
| 同机 PCIe（无 NVLink） | 同 root complex / PCIe switch（PATH_PHB 等） | **p2p**（PCIe P2P，三道门 7.3 §5）或 **shm**（P2P 不可行时 host 共享内存 proxy 拷贝） | ≤64 GB/s 单向再扣折扣 |
| 跨节点 | NIC → fabric → NIC | **net**（IB RDMA / socket；GDR 9.2） | NIC 线速 50 GB/s（400G） |
| 网内归约 | NVSwitch SHARP / IB SHARP | NVLS 独立路径 / **collNet** | 7.1 §6 / 10.2 |

p2p transport 的机制底座正是 7.3 的 peer 映射（p2pMap，p2p.cc L358）；shm 用于同机 PCIe 无 P2P 时经 host 共享内存中转（shm.cc，proxy 线程拷贝）；net 用于跨机（net.cc sendProxyProgress L1327，proxy 线程网络进度状态机）。路径类型 PATH_NVL / PATH_PHB / PATH_PXN 由拓扑检测判定（paths.cc，8.4 走读）。

**放置表**（原语 × 并行 × 放置 × 硬件依据，recap §2 并行账）：

| 原语 / 并行 | 消息 regime | 放置 | 硬件依据 |
| --- | --- | --- | --- |
| TP 层内 all-reduce | 小消息延迟主导（serving decode） | NVLink 域内（节点内） | 9× 带宽差 + 延迟项敏感（7.2 §7） |
| DP 梯度 all-reduce | 大消息带宽主导 | 分层 ring：节点内 NVLink + 机间 RDMA | 压跨域字节（10.2） |
| EP all-to-all | decode 逐 peer KB 级延迟主导 | 域内受限或跨机 RDMA（Wide-EP） | inference 8.1 §4 recap |
| ZeRO all-gather / reduce-scatter | 大消息带宽主导 | 同 DP 组 | training 4.2/4.3 |
| PP send/recv | 点对点（非集合） | 跨机 stage 边界 | inference 8.1 §3 recap |

放置规则的算术依据是节点内 vs 机间的数量级差（recap 7.2 §7）：NVLink 单向 450 / NIC 50 = **9×**（B200 900/50 = 18×）。高频集合（TP 逐层归约）尽量留 NVLink 域内——域内每跳 370.8 GB/s 有效，跨机每跳只有 NIC 线速量级；DP/PP 跨机流量按 rail 与分层 ring 组织（10.2），把跨域字节压到最小。跨机 net transport 的 NIC→HBM 直路是 GDR（9.2），P2P 只管域内段（7.3 §7）；NVLS 把归约搬进 NVSwitch、collNet 把归约搬进 IB 交换机（SHARP），是两条网内归约 bypass（7.1 §6 / 10.2）。

## 7. NCCL API 接口

原语语义落到代码是哪些函数？count 参数怎么对应 §3 的 $M$？NCCL 公共 API 在 `src/nccl.h.in`（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd，行号 fresh tarball 核实）：

| 原语 | API | nccl.h.in | count 约定 |
| --- | --- | --- | --- |
| all-reduce | ncclAllReduce | L602 | count = 每 rank 全量向量元素数 |
| reduce-scatter | ncclReduceScatter | L618 | recvcount = 分片元素数（sendbuff = $n \times$ recvcount） |
| all-gather | ncclAllGather | L635 | sendcount = 分片元素数（recvbuff = $n \times$ sendcount） |
| all-to-all | ncclAlltoAll | L648 | count = **每 peer** 元素数（sendbuff = $n \times$ count） |
| broadcast | ncclBroadcast | L589 | count = 全量元素数 + root |
| reduce | ncclReduce | L561 | count = 全量元素数 + root |
| gather | ncclGather | L663 | count = 每 rank 发送分片元素数 + root |
| scatter | ncclScatter | L678 | count = 每 rank 接收分片元素数 + root |
| 点对点 | ncclSend / ncclRecv | L739 / L758 | count = 消息元素数 + peer |

**拼写注**（版本事实）：all-to-all 的 API 名是 **ncclAlltoAll**（小写 t），不是 ncclAllToAll——2.32.7-1 nccl.h.in L648 逐字。count 约定与 §3 的 $M$ 口径逐项对应：ncclAllReduce 的 count × bytes = $M$（每 rank 全量）；ncclAllGather 的 sendcount × bytes = $M/n$（分片）；ncclAlltoAll 的 count × bytes = $M/n$（每 peer），sendbuff 总量 $n \times$ count × bytes = $M$——命题 1 的每 peer 口径在 API 层逐字兑现。

三条接口语义（nccl.h.in doc comment 逐字）。其一，**communicator**：集合操作在 communicator 内调用，ncclCommInitRank 建立 $n$ rank 的通信组（init.cc L3071，8.4 走读）；「Collective communication operations must be called separately for each communicator」。其二，**stream 语义**：「They return when operations have been enqueued on the CUDA stream」——API 返回 = 入队完成，不是传输完成；完成靠 stream 同步（3.3 异步执行口径）。其三，**in-place 规则**：all-reduce 在 sendbuff == recvbuff 时原地；all-gather 在 sendbuff == recvbuff + rank × sendcount 时原地；reduce-scatter 在 recvbuff == sendbuff + rank × recvcount 时原地。

**group 语义**：ncclGroupStart（L873）/ ncclGroupEnd（L883）把多个调用融成一组。两个用途（doc comment 逐字）：单线程管理多 GPU 时把不同 rank 的调用 group 起来（集合调用可能做跨 CPU 同步）；以及「If multiple ncclSend and ncclRecv operations need to progress concurrently to complete, they must be fused within a ncclGroupStart/ncclGroupEnd section」——多个 send/recv 要并发推进必须 group，a2a 也可用 grouped send/recv 拼成。注意 ncclGroupEnd 对集合通信只保证入队、不保证完成（doc 逐字「only guarantees that the operations are enqueued on the streams, not that the operation is effectively done」）。

算法选择（ring/tree/NVLS/PAT）、协议（LL/LL128/Simple）、cost model（ncclTuningGetTime 的 α-β 代码形态）与环构建的模块全量走读归 8.4；本章只交付接口与语义。NCCL docs 的原语语义与 NCCL_ALGO/NCCL_PROTO 环境变量口径见 [NVIDIA, NCCL docs](../references.md)（[C]，2.32.3，滞后 repo 2.32.7 注）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| ring 每 rank $2(n-1)/n \cdot M \approx 2M$，带宽账与 $n$ 解耦；α-β 延迟项 $2(n-1)\alpha$ + 带宽项 $2(n-1)/n \cdot M\beta$（recap training 4.1 式 (2)(3)， $d \to n$，归纳不重推） | 式 (1)(2) | §2、练习 2 |
| 八原语语义全表； $M$ = 全量数据字节、分片 $M/n$；均衡四原语（AR/AG/RS/a2a）每 rank 与 $n$ 解耦无 root 热点，root 四原语（broadcast/reduce/gather/scatter）root 负载分有界（gather/scatter ≈M）与依赖实现（broadcast/reduce） | §3 全表 | §3、练习 1 |
| a2a 每 rank 总发送 $(n-1)/n \cdot M$、每 peer $M/n$，自环份留本地不发；含自环多计 $M/n$（ $n=8$、 $M=1$ GB 时多计 125 MB） | 式 (3)，命题 1 | §3、练习 3 |
| AR = RS + AG，量恒等 $(n-1)/n + (n-1)/n = 2(n-1)/n$（recap training 4.1/5.2）；ZeRO 梯度 RS + 参数 AG 与 DP 同构 | §4 组合 | §4、练习 4 |
| a2a 完全图单跳直达、全 bisection 需求 + 偏斜 incast（→10.3）；ring 邻居链路持续负载；a2a 无归约语义（combine 加权求和本地） | §4 流量图 | §4、10.3 |
| regime 临界 $M_c = n\alpha/\beta$（ $n=8$ → 5.93 MB）；带宽/延迟比 = $M/M_c$；训练 13 GiB ≈2353× 带宽主导 vs serving 0.61 MB = 0.104× 延迟主导 | 式 (4)，命题 2 | §5、练习 2/3 |
| $M_c$（ring 内部两项临界）≠ 8.3 ring/tree 算法间交叉（后者来自实测 $\beta_{\mathrm{eff}}$ 差，理想模型无交叉） | §5 口径注 | §5、8.3 |
| 原语是语义、链路是放置；transport 表 p2p/shm/net/collNet（NVLS 独立）；TP 留域内、DP 分层 ring、EP 域内或 Wide-EP；9× 带宽差是放置依据 | transport.cc L15-20；7.2 §7 | §6、练习 1 |
| NCCL API：ncclAllReduce L602 / ReduceScatter L618 / AllGather L635 / **AlltoAll L648（小写 t）** / Broadcast L589 / Reduce L561 / Gather L663 / Scatter L678 / Send L739 / Recv L758；count 对应 $M$ 或 $M/n$；返回 = 入队；group 语义 | nccl.h.in 版本锚 2.32.7-1 | §7、8.4 |

## 练习

1.（计算 · [10]）原语消息量代入（§3 表， $n=8$、 $M=1$ GB）：(a) all-gather / reduce-scatter / all-to-all 每 rank 发送各多少 GB；(b) all-to-all 每 peer 多少 GB；(c) all-reduce 每 rank 多少 GB；(d) gather root 收多少 GB。校验：(a) 均 0.875 GB（ $7/8$）；(b) 0.125 GB（ $1/8$）；(c) 1.75 GB（ $2 \times 7/8$）；(d) 0.875 GB，与 §3 全表逐数一致。
2.（计算 · [20]）regime 判定（式 (2)(4)， $n=8$、 $\alpha=2$ μs、 $\beta=1/370.8$ GB/s，教学假设口径）：(a) $M_c$ 多少 MB；(b) 训练 7B bf16 梯度（≈13 GiB）与 serving decode 单次归约（0.61 MB）各约 $M_c$ 的几倍、各落哪个 regime；(c) $M=1$ GB 时 ring 带宽项 / 延迟项 / 合计各多少 ms？先预测 (b) 两负载的 regime 再算。校验：(a) ≈5.93 MB；(b) ≈2353× 带宽主导 / 0.104× 延迟主导；(c) 4.72 / 0.028 / ≈4.75 ms，与 7.2 §3 逐数一致（容差 1%）。
3.（推导 · [30]）证明命题 2（式 (4)）并做命题 1 自环反例：(a) 令式 (2) 延迟项 = 带宽项，解出 $M_c = n\alpha/\beta$；(b) 证明带宽项/延迟项 = $M/M_c$；(c) a2a 计数：为什么自环份不计发送，含自环会多计多少（ $n=8$、 $M=1$ GB）；(d) 说明 $M_c$ 与 8.3 ring/tree 交叉临界的区别。校验：(a) $n\alpha/\beta$；(b) $M/M_c$；(c) 多计 $M/n = 0.125$ GB；(d) $M_c$ 是 ring 内部两项临界，8.3 临界来自实测 $\beta_{\mathrm{eff}}$ 差（理想模型无交叉）。
4.（代码 · [30]）教学用，非优化实现：纯 Python 模拟 $n=4$ 的原语消息量账——(a) 逐步执行 ring all-gather（ $n-1$ 步、每步每 rank 发 $M/n$），断言每 rank 累计发送 = $(n-1)/n \cdot M$；(b) 模拟 a2a 每 peer 发送矩阵，断言每 rank 总发送 = $(n-1)/n \cdot M$、每 peer = $M/n$、自环（对角）发送 = 0；(c) 断言 AR = RS + AG 量恒等 $2(n-1)/n \cdot M$。校验：断言与 §3 表一致； $n=4$ 系数 0.75（AG/RS/a2a）/ 1.5（AR）；(b) 对角发送为 0（命题 1 自环排除）。

## 参考

- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e，master 2026-11-24）fresh tarball 核实——src/nccl.h.in：ncclAllReduce L602 / ncclReduceScatter L618 / ncclAllGather L635 / ncclAlltoAll L648（小写 t）/ ncclBroadcast L589 / ncclReduce L561 / ncclGather L663 / ncclScatter L678 / ncclSend L739 / ncclRecv L758 / ncclGroupStart L873 / ncclGroupEnd L883，count 约定 + in-place 规则 + stream 入队语义 + group 语义 doc comment 逐字；src/transport.cc L15-20 ncclTransports[] = {p2p, shm, net, collNet}（NVLS 独立）；src/transport/p2p.cc p2pMap L358 / net.cc sendProxyProgress L1327；src/init.cc ncclCommInitRank L3071（模块全量走读归 8.4）（§6/§7）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2026-11-25，版本 2.32.3（滞后 repo 2.32.7 注）——集合原语语义、NCCL_ALGO（Ring/Tree/NVLS/NVLSTree/PAT）/ NCCL_PROTO（LL/LL128/Simple）环境变量口径（§7，算法选择走读归 8.4）
- training 4.1 / 4.2 / 4.3 / 5.1 / 5.2 / 5.3 / 5.4（纯文字指引，不挂链接）：ring 消息量式 (2) + α-β 式 (3) + scatter-reduce 不变式归纳证明（4.1 §2.2，recap 不重推）、星型基线 root 链路 $2(d-1)M$（4.1 §2.1）、DDP 桶 25 MiB（4.1 §3.1）、ZeRO 梯度 RS + 参数 AG 通信不变（4.2）、FSDP（4.3）、TP f/g 共轭（5.1）、SP 恒等 AR=RS+AG（5.2 §2）、PP 气泡（5.3）、EP a2a 无归约语义 + 木桶（5.4）——训练账 recap 不重推（§2/§3/§4/§5）
- inference 8.1（纯文字指引，不挂链接）：serving 无 backward 翻 regime（§1）、TP decode 消息 $2B d_{\mathrm{model}}$ 式 (1) + 延迟主导（§2）、PP 层间 send/recv（§3）、EP a2a 共享出口时间模型 + 逐 peer KB 级（§4）——serving 账 recap 不重推（§2/§5）
