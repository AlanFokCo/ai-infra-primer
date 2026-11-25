# 8.2 Ring 算法（Ring Algorithms）

> ring all-reduce 每 rank 发 $2(n-1)/n \cdot M$ 字节——这个数还能再压吗？本章先 recap training 4.1 §2.2 式 (2)(3)（消息量 + α-β 模型， $d \to n$ 翻译，归纳证明已交付不重推），再交付四块 GPU 侧增量：**带宽最优性下界证明**（命题 1 总流量 $\ge 2(n-1)M$，chunk 归约树 + 广播树两类不相交传输；推论负载均衡每 rank $\ge 2(n-1)/n \cdot M$，ring 达界；单阶段界 $(n-1)/n \cdot M$ 单独归属、勿与总量混 2×；弱界 $S_r \ge M$ 不足闭合）、ring 构造（逻辑环 → 物理拓扑，NVLink 域内 ring recap 7.2）、latency-bandwidth 账（chunking/pipelining 摊薄延迟项 + 多 ring 并行带宽聚合）、ring vs 星型 / recursive halving 对照与万卡可扩展论证 + NVLink-aware 分层 ring | 前置：training 4.1 §2.2（ring 消息量 + α-β，recap 源）、8.1（原语语义与 regime 临界 $M_c$）、7.2（有效带宽 370.8 GB/s、ring 映射 NVLink 域、节点内 vs 机间 9×）、7.1 §6（NVLS 网内归约对照）；下游 8.3（tree 延迟与 ring/tree 交叉 $M^{\ast}$）、8.4（NCCL 环构建与 cost model 模块走读）、10.2（分层 ring 与 fabric 工程） | 锚点：Patarasuk-Yuan 2009（[Patarasuk et al., 2009](../references.md)，[B] 老经典无 arXiv，JPDC 69(2):117-124，ring 带宽最优出处支撑）、Rabenseifner 2004（[Rabenseifner, 2004](../references.md)，[B] 老经典无 arXiv，LNCS 3036，recursive halving 对照）、NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd，rings.cc ncclBuildRings + tuning_general.cc ncclTuningGetTime）——经典 Crossref 2026-11-25 再核、代码 fresh tarball 核实

## 1. 三个问题与本章口径

ring all-reduce 每 rank 发送 $2(n-1)/n \cdot M$ 字节（training 4.1 式 (2)），这个系数能不能再压？换句话说，有没有任何算法能让每 rank 发得更少、却仍把全和送到每个 rank？单阶段（reduce-scatter 或 all-gather）的 $(n-1)/n \cdot M$ 和 all-reduce 总量的 $2(n-1)/n \cdot M$ 差一个 2 倍因子，下界证明里怎么不把这两者混起来？大消息下 ring 的带宽项与 $n$ 无关，这条性质怎么撑到万卡、延迟项又为什么在万卡下成为新瓶颈？本章回答三个问题：all-reduce 的总流量与每 rank 发送的紧下界是什么、ring 为什么达界（带宽最优）？逻辑环怎么落到物理拓扑、延迟项怎么摊薄、多 ring 怎么聚合带宽？大消息可扩展性的边界在哪、分层 ring 怎么把跨域字节压到最小？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（ $n = 8$，NVLink4 + 4 颗第三代 NVSwitch，几何 recap 7.1/7.2）；有效带宽取 **NCCL 模型 370.8 GB/s 单向**（7.2 式 (2)，18 × 20.6，raw 25 与模型 20.6 两层账分离）； $\alpha = 2$ μs 为**教学假设口径**（7.2 §3，NVLink P2P 延迟量级假设，非实测）；跨机对比行 400 Gb/s NIC = 50 GB/s 单向（7.2 §7）。贯穿实例 $M = 1$ GB = $10^{9}$ B；通信精度 bf16（梯度口径沿 recap 源）。万卡论证段换 $n = 10000$ 只为展示系数趋 2 的极限，除该段外不再换。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，与 models/inference $N$ = 参数量、training $d$ = DP 度严格区分；**recap training 4.1 时 $d \to n$ 翻译**，附口径注）； $M$ = 被归约向量总字节（分片恒 $M/n$）； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数（recap training 4.1 式 (3)）； $S_r$ = rank $r$ 的发送字节数、 $C_{\mathrm{rank}}$ = 每 rank 发送量、 $k$ = 并行 ring（channel）数（本章局部符号）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 $T_{\mathrm{ring}}$ / $T_{\mathrm{comm}}$ 下标。

**recap 边界声明**（强制，spec §4.2）：training 4.1 §2.2 式 (2) ring 每 rank 消息量 $2(d-1)/d \cdot M$ 与式 (3) α-β 模型**已交付（含 scatter-reduce 不变式归纳证明），本章只 recap + $d \to n$ 翻译，不重推归纳**；本章增量 = 带宽最优性下界证明（spec §6.3 强制重推导）+ chunking/pipelining 摊薄 + 多 ring 并行 + NVLink-aware 分层 ring。ring/tree 延迟交叉临界 $M^{\ast}$ 归 8.3，NCCL 环构建与 cost model 模块全量走读归 8.4，机间 fabric 工程归 10.2，GDR 归 9.2。**代码 scope 声明**：本章 = 机制片段级（环构建锚 rings.cc ncclBuildRings + α-β 代码形态锚 tuning_general.cc ncclTuningGetTime；模块全量走读归 8.4）。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. Recap：ring 消息量与时间模型

ring all-reduce 每 rank 发多少、耗时多少？training 4.1 §2.2 已把这两笔账推到数，本章 recap 并翻译符号，不重推归纳。

**ring 每 rank 消息量**（recap training 4.1 式 (2)， $d \to n$ 翻译）：ring 把向量切 $n$ 块、每块 $M/n$，分 scatter-reduce 与 all-gather 两阶段各走 $n-1$ 步，每步每 rank 发 $M/n$；合计每 rank 发送（接收同量）：

$$
C_{\mathrm{rank}} = 2 \thinspace \frac{n-1}{n} \cdot M \hspace{5mu} \approx \hspace{5mu} 2M \qquad (1)
$$

**符号**： $C_{\mathrm{rank}}$ = 一次 all-reduce 每 rank 发送字节数； $M$ = 被归约向量总字节； $n$ = rank 数（training 原式用 $d$ = DP 度，此处译 $n$，口径注：gpu 维度 $n$ 专归 rank 数）。系数 $2(n-1)/n$ 从 $n=2$ 的 1 单调逼近 2。**归纳证明已交付 training 4.1 §2.2（scatter-reduce 不变式：第 $s$ 步每块部分和累加 $s+1$ 份贡献），本章不重推**；两阶段各 $(n-1)/n \cdot M$ 的账在 8.1 §3 已独立登记为 reduce-scatter / all-gather 原语账。代入 $n=8$、 $M=1$ GB： $2 \times 7/8 \times 1 = 1.75$ GB。这个 1.75 GB 是不是最少？§3 证它是紧下界。

**α-β 时间模型**（recap training 4.1 式 (3)， $d \to n$ 翻译）：

$$
T_{\mathrm{ring}} = 2(n-1)\thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \beta \qquad (2)
$$

**符号**： $\alpha$ = 每跳消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $`\beta = 1/\mathrm{BW}_{\mathrm{eff}}`$ = 每字节传输时间。延迟项 $2(n-1)\alpha$ 与 $M$ 无关、随 $n$ 线性涨；带宽项 $2(n-1)/n \cdot M\beta$ 随 $M$ 线性、与 $n$ 基本无关。代入 $n=8$、 $M=1$ GB、 $\beta = 1/(370.8 \times 10^9)$ s/B、 $\alpha = 2$ μs：带宽项 ≈4.72 ms、延迟项 0.028 ms、合计 ≈4.75 ms（与 7.2 §3 / 8.1 §2 逐数一致，带宽主导，比值约 169）。regime 临界 $M_c = n\alpha/\beta$ 已在 8.1 §5 解出（ $n=8$ → 5.93 MB），本章不重推。

ring 的两阶段结构如下（逻辑环，箭头 = 发送方向；归纳不变式 recap training 4.1 §2.2，不重推）：

```
逻辑环（n=4 示意，每 rank 只与 prev/next 两邻居通信）：
        r0 ──► r1
        ▲        │
        │        ▼
        r3 ◄── r2
scatter-reduce（n-1 步）：每步每 rank 发 M/n 部分和，沿环累加
        终态：每 rank 恰持一个全和块（r1 持 c0 全和 …）
all-gather（n-1 步）：每 rank 把自己的全和块沿环转发
        终态：每 rank 集齐 n 个全和块 = 完整归约向量
每 rank 发送 = (n-1)·M/n + (n-1)·M/n = 2(n-1)/n·M（式 (1)）
```

## 3. 带宽最优性下界证明

式 (1) 的 $2(n-1)/n \cdot M$ 是 ring 的发送量，但它是**所有** all-reduce 算法的下界吗？本节证它是——先证总流量下界（命题 1），再在负载均衡下推到每 rank（推论），最后说明单阶段界与弱界。这是本章的核心增量（spec §6.3 强制重推导），出处支撑 Patarasuk-Yuan 2009（[Patarasuk et al., 2009](../references.md)，[B]，ring 带宽最优经典），证明按本书命题-证明骨架独立重推。

**证明路线**（先给路线再下符号）：把向量切 $n$ 个 chunk，对每个 chunk 区分两类传输——归约树边（携带**未齐全**的部分和）与广播树边（携带**已齐全**的全和）；关键一步是这两类传输集合**不相交**（一次传输要么携带部分和、要么携带全和，不能同时），于是每 chunk 的流量下界是两类之和 $2(n-1)M/n$，对 $n$ 个 chunk 求和得总流量 $\ge 2(n-1)M$。下面逐 chunk 展开。

**命题 1**（all-reduce 总流量下界）。设 $n$ 个 rank 各持 $M$ 字节向量，all-reduce 结束后每 rank 持逐元素归约结果。记 $S_r$ 为 rank $r$ 的发送字节数；每次传输恰有一个发送方，故总流量（所有链路传输字节之和，每跳计一次）等于 $\sum_r S_r$。任何正确算法满足：

$$
\sum_{r=1}^{n} S_r \ge 2(n-1)\thinspace M \qquad (3)
$$

**证明**（路线：切 chunk → 每 chunk 归约树界 + 广播树界 → 不相交相加 → 对 chunk 求和，四步）。

其一，**切 chunk**。把 $M$ 字节向量均分为 $n$ 个 chunk，每 chunk $M/n$ 字节。all-reduce 逐元素归约，故每个 chunk 的全和必须独立地送到每个 rank。下面固定一个 chunk $c$，证它的流量 $\ge 2(n-1)M/n$。

其二，**归约树界**（携带未齐部分和的传输）。设 rank $r_c$ 是**第一个**持齐 chunk $c$ 全和的 rank。在 $r_c$ 持齐之前，其余 $n-1$ 个 rank 各自的贡献（各 $M/n$ 字节）都必须传到 $r_c$（否则 $r_c$ 算不出全和）。每个非 $r_c$ 的 rank $r$ 的贡献必须**离开** $r$ 至少一次；取「首次携带 $r$ 的贡献离开 $r$ 的那次传输」 $t_r$。这 $n-1$ 次传输 $\lbrace t_r \rbrace$ 源 rank 互异（ $t_r$ 的源是 $r$），故是 $n-1$ 次**不同**的传输；每次携带一个 chunk $c$ 的部分和，对不可压缩的任意数据其尺寸 $\ge M/n$ 字节。所以归约树类传输的总字节 $\ge (n-1) \cdot M/n$。

其三，**广播树界**（携带已齐全和的传输）。 $r_c$ 持齐 chunk $c$ 全和后，其余 $n-1$ 个 rank 每个都必须拿到这份全和（ $M/n$ 字节），即每个都须有 $\ge M/n$ 字节的全和信息流入。所以广播树类传输的总字节 $\ge (n-1) \cdot M/n$。

其四，**不相交相加**。一次传输要么发生在「chunk $c$ 全和尚不存在」时（携带部分和，归约树类），要么发生在「全和已存在」后（携带全和，广播树类），二者互斥——故两类传输集合不相交，字节数可直接相加：chunk $c$ 流量 $\ge (n-1)M/n + (n-1)M/n = 2(n-1)M/n$。对 $n$ 个 chunk 求和：总流量 $\ge n \cdot 2(n-1)M/n = 2(n-1)M$。∎

全数字代入（ $n=8$、 $M=1$ GB）：总流量下界 $2 \times 7 \times 1 = 14$ GB；每 chunk 下界 $2 \times 7/8 = 1.75$ GB，8 个 chunk 合计 14 GB。注意这是**全网所有链路**的流量总和，不是单 rank 的账。

**最小例走读**（ $n=4$，固定 chunk $c_0$，看两类传输怎么数）。设 rank 0 是首个持齐 $c_0$ 全和的 rank（ $r_c = 0$）。归约树侧：rank 1、2、3 各自的 $c_0$ 贡献（各 $M/4$ 字节）都要传到 rank 0。哪怕算法先把 rank 1 与 rank 2 的部分和在 rank 2 合并、再转发给 rank 0，rank 1 的贡献仍须**首次离开 rank 1** 一次（ $t_1$， $M/4$ 字节）、rank 2 的贡献须首次离开 rank 2 一次（ $t_2$， $M/4$）、rank 3 同理（ $t_3$， $M/4$）——三次传输源互异，合计 $\ge 3M/4$。合并只是让某次传输携带了多份贡献，并不能让任何一份贡献「不离开自己的源 rank」，所以 $n-1$ 次首次离开一次都省不掉。广播树侧：rank 0 持齐后，rank 1、2、3 各须拿到 $c_0$ 全和（各 $M/4$），合计 $\ge 3M/4$。两类相加 $\ge 6M/4 = 2(4-1)M/4$，对 4 个 chunk 求和 $\ge 6M = 2(4-1)M$，与式 (3) 逐数闭合。这个例子说明：下界不依赖具体环排列或合并时机，只依赖「每份贡献必须离开源、每个 rank 必须收齐全和」这两条信息流约束。

**口径注**（下界的适用前提）。其一，**不可压缩数据**：证明假设梯度数据任意、不可压缩，故每份贡献 $M/n$ 字节的信息必须实打实传 $M/n$ 字节；若数据可压缩（如稀疏梯度），实际流量可低于此界，但那是改变了「每份贡献 $M/n$ 字节」的前提，不推翻本界对稠密梯度的紧性。其二，**最坏情形下界**：命题 1 是对所有正确算法的流量下界（存在性意义——任何算法都至少这么多），ring 达界说明它紧。其三，**每跳计一次**：总流量按发送方计（ $\sum_r S_r$），中继转发被重复计入（rank 1 发给 rank 2、rank 2 再发给 rank 0，两跳都计），这正是 ring 转发流量的来源，也是弱界 $S_r \ge M$ 漏掉的那一笔。

**推论**（负载均衡下每 rank 发送下界，ring 达界）。设完成时间为 $T_{\mathrm{comm}}$，每 rank 发送 $S_r$。在 α-β 模型下，每 rank 的发送在其出口链路上串行，故 $T_{\mathrm{comm}} \ge \max_r S_r \cdot \beta$；又 $\sum_r S_r \ge 2(n-1)M$（命题 1），于是 $\max_r S_r \ge \frac{1}{n}\sum_r S_r \ge 2(n-1)/n \cdot M$。要使完成时间最小就要让 $\max_r S_r$ 最小，即把总流量**均摊**到 $n$ 个 rank（负载均衡），此时每 rank 发送 $\ge 2(n-1)/n \cdot M$：

$$
S_r \ge 2 \thinspace \frac{n-1}{n} \cdot M \qquad (\text{负载均衡，ring 达界}) \qquad (4)
$$

ring 每 rank 恰发 $2(n-1)/n \cdot M$（式 (1)），且每步每 rank 发送量相同（天然负载均衡），带宽项 $2(n-1)/n \cdot M\beta$ 恰好等于下界——**ring 带宽最优**。代入 $n=8$、 $M=1$ GB：每 rank 下界 1.75 GB，ring 实发 1.75 GB，达界。∎

**单阶段界单独归属**（勿与总量混 2×）。命题 1 的归约树界单独就是 **reduce-scatter** 的总流量下界 $(n-1)M$，广播树界单独就是 **all-gather** 的总流量下界 $(n-1)M$。负载均衡下每 rank：

$$
S_r^{\mathrm{单阶段}} \ge \frac{n-1}{n} \cdot M \qquad (\text{reduce-scatter 或 all-gather，ring 逐阶段达界}) \qquad (5)
$$

**口径注**（硬规则）： $(n-1)/n \cdot M$ 是**单阶段**（reduce-scatter **或** all-gather）每 rank 紧下界；all-reduce = 两阶段复合，每 rank 总下界是它的 **2 倍** $2(n-1)/n \cdot M$。代入 $n=8$、 $M=1$ GB：单阶段 0.875 GB，all-reduce 总量 1.75 GB。ring 的 scatter-reduce 阶段发 0.875 GB、all-gather 阶段再发 0.875 GB，逐阶段各达单阶段界，合计达总界。**把单阶段界当 all-reduce 总量会少算一半**——这是下界证明最易踩的 2× 坑。

> **坑**：单阶段界与 all-reduce 总量混 2×。触发条件：把 reduce-scatter 的 $(n-1)/n \cdot M$ 直接当 all-reduce 每 rank 下界，或把总流量下界写成 $(n-1)M$；现象：下界少算一半，「ring 达界」变成「ring 超界 2×」的假象；规避：单阶段 $(n-1)/n \cdot M$（式 (5)）与 all-reduce 总量 $2(n-1)/n \cdot M$（式 (4)）分开标注，总流量 $2(n-1)M$（式 (3)）是单阶段 $(n-1)M$ 的 2 倍，对账时先问「这是单阶段还是两阶段复合」。

**弱界 $S_r \ge M$ 不足闭合**。一个直觉的信息论弱界是「每 rank 交出自有份、收齐全体和」：每 rank 须发出自己的贡献（约 $M$）、收回全和（约 $M$），故 $S_r \ge M$。这个界**不足以**证 ring 最优——它只数了「自有份离开」这一项，漏了 ring 里每 rank 还要**转发其他 rank 的 chunk**（约再一个 $M$ 的中继流量）。代入 $n=8$、 $M=1$ GB：弱界只给 $S_r \ge 1$ GB，而 ring 实发 1.75 GB、紧下界也是 1.75 GB——弱界差 1.75 倍，无法区分 ring 与一个只发 1 GB 的（不存在的）算法。命题 1 的归约树 + 广播树两类不相交论证恰好把「归约流入」与「广播流出」两笔都数到，才闭合到 $2(n-1)/n \cdot M$。这就是为什么下界证明必须切 chunk 分类传输，而不能停在「交出自有份」的直觉。

**与均衡四原语的结构联系**（recap 8.1 §3）。命题 1 的论证只用了「每份贡献必须离开源 rank、每个 rank 必须收齐全和」两条信息流约束，没有用环的具体形状——所以它适用于任何 all-reduce 实现，而 ring 只是恰好达界的一个构造。同理，reduce-scatter 与 all-gather 各自的单阶段界 $(n-1)/n \cdot M$ 也来自同样的切 chunk 论证（只取归约树或广播树一类）。这解释了 8.1 「均衡四原语」（all-reduce / all-gather / reduce-scatter / all-to-all）为什么每 rank 发送系数与 $n$ 解耦、无 root 热点：它们的语义本身就要求信息在 rank 间均匀流动，要在完成时间意义下达每 rank 界就必须负载均衡，而负载均衡正是把总流量均摊、让每 rank 账与 $n$ 解耦的结构原因。星型之所以不合格，不是语义错，而是它把流量集中到 root、违反了达界所需的负载均衡。

> **提点**：下界证明的可复用骨架是「切 chunk → 按信息状态分类传输 → 证各类不相交 → 分别计数相加」。这套骨架不只用于 all-reduce：任何「数据必须从一组源汇到一组目的地」的集合操作，都可以按「中间态」与「终态」把传输分类，再证不相交得到流量下界。掌握这个骨架，比记住 $2(n-1)M$ 这个具体数字更重要。

两类传输不相交的结构如下（固定 chunk $c$， $r_c$ 为首个持齐全和的 rank）：

```
归约树（全和尚不存在，边携带部分和）        广播树（全和已存在，边携带全和）
      r_a ─┐                                    r_c ─┬──► r_a
           ├──► … ──► r_c                          ├──► r_b
      r_b ─┘   (首个持齐全和)                       └──► … ──► r_{n-1}
  n-1 个贡献各首次离开源 rank 一次            n-1 个 rank 各须流入 ≥ M/n 全和
  归约树流量 ≥ (n-1)·M/n                     广播树流量 ≥ (n-1)·M/n
        └──────── 两类不相交，相加 ────────┘
        每 chunk ≥ 2(n-1)·M/n  →  n chunk 求和 ≥ 2(n-1)M（式 (3)）
```

## 4. ring 构造：逻辑环落到物理拓扑

逻辑环 0 → 1 → … → 7 → 0 的每条边，在 HGX H100 上走的是哪条物理路径？recap 7.2 §4：每跳是「源 GPU → NVSwitch crossbar → 目标 GPU」的一条 switch 路径，不是直连链路（HGX 8-GPU 的 18 links 全汇 NVSwitch）。NVSwitch 非阻塞（7.1 §3 blog 逐字）保证**任意环排列**线速接通——ring 的邻居关系是逻辑的，物理上任意两 GPU 都可经交换矩阵一跳直达，所以环怎么排都不损失带宽。

```
逻辑环（NCCL 搜索得到）          物理路径（HGX H100，每跳经 NVSwitch）
  0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0     GPU_i ─► NVSwitch ─► GPU_{i+1}
  每 rank 只与 prev/next 通信            每跳单向 raw 450 / 模型有效 370.8 GB/s
  环排列任意（非阻塞 crossbar）          每 rank 发送 2(n-1)/n·M（式 (1)）
```

NCCL 把搜索得到的 prev/next 邻接表落成环数组的代码是 `ncclBuildRings`（src/graph/rings.cc L29，[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd，fresh tarball 核实）：从本 rank 出发沿 next 指针走 $n$ 步生成环，并做两道校验——「ring does not loop back to start」（走 $n$ 步须回到起点，L51）与全 rank 覆盖校验（每个 rank 恰出现一次）。这两道校验保证逻辑环是一个覆盖全 rank 的单一循环，正是 §3 负载均衡达界的结构前提（每 rank 在环上地位对称）。环搜索本身（ncclTopoSearchRec / ncclTopoCompute，按路径类型与带宽约束）归 8.4 模块走读。

**直连拓扑的边界**（非 NVSwitch 情形）：若 GPU 间是直连全互连而非 switch（如 4-GPU HGX 无 NVSwitch，7.2 §2），ring 的每条边必须落在一条**真实直连链路**上，环排列受物理接线约束；每对带宽按直连均分 $`\mathrm{BW}_{\mathrm{GPU,bidir}}/(n-1)`$（7.2 命题 1）。此时 ring 仍带宽最优（每 rank 发送量不变），但每跳可用带宽是直连均分值而非 switch 满带宽——拓扑改变的是 $\beta$（有效带宽），不是 §3 的流量下界。这条区分很重要：**下界是拓扑无关的流量账，达界所需的带宽由拓扑供给**。

## 5. latency-bandwidth 账：摊薄延迟与多 ring 并行

式 (2) 的延迟项 $2(n-1)\alpha$ 随 $n$ 线性涨，大消息下怎么摊薄、多 ring 怎么聚合带宽？本节交付两个增量：延迟项的摊薄机制与多 ring 并行的带宽聚合。

**延迟项摊薄**（amortization）。延迟项 $2(n-1)\alpha$ 与 $M$ 无关，是固定的流水线填充/排空成本；每字节摊到的延迟为 $2(n-1)\alpha / M$，随 $M$ 增大趋于 0。代入 $n=8$、 $\alpha=2$ μs、 $M=1$ GB：每字节延迟 $28 \times 10^{-6} / 10^{9} = 2.8 \times 10^{-14}$ s/B，而每字节带宽时间 $\beta = 2.70 \times 10^{-12}$ s/B——延迟摊薄到带宽项的约 1/96，大消息牢固带宽主导（8.1 §5 regime 结论 recap）。**chunking/pipelining 的作用**是把消息切成 chunk 让各链路同时忙：ring 每步 $n$ 个 rank 并行发送，流水线填满后所有邻居链路持续满载，带宽项才能实现在 §3 的下界上；若不切块、整条消息串行等每跳完成，链路利用率塌陷，带宽项远高于下界。所以 chunking 不减少字节数（式 (1) 不变），它保证下界带宽**可达**。

流水线的时间结构如下（scatter-reduce 阶段， $n=4$，每格一步、每步每 rank 发 $M/n$）：

```
步 →     1      2      3      （n-1 = 3 步）
r0:    发c0   发c3   发c2     每步发一块，链路不空等
r1:    发c1   发c0   发c3
r2:    发c2   发c1   发c0
r3:    发c3   发c2   发c1
       └─ 填充期 ─┘└─ 稳态：n 条邻居链路同时满载 ─┘
延迟项 = 流水线填充/排空 2(n-1)α（与 M 无关）
带宽项 = 稳态传输 2(n-1)/n·Mβ（chunking 保证链路满载才达界）
```

关键观察：流水线填满后， $n$ 条邻居链路**同时**在传不同 chunk，没有链路空等——这就是带宽项能实现在下界上的机制。延迟项 $2(n-1)\alpha$ 是填充与排空的固定成本，不随 chunk 数增加而叠加（更多 chunk 只是让每步更小、稳态更长，填充成本不变）。所以大消息下延迟项被摊薄、带宽项主导；小消息下填充成本占比高，落延迟主导 regime（8.1 §5 $M_c$ 临界 recap）。

**多 ring 并行带宽聚合**（增量）。把消息 $M$ 拆到 $k$ 个并行 ring（NCCL 称 channel）上，每个 ring 处理 $M/k$，各自走式 (2)； $k$ 个 ring 同时跑，带宽项除以 $k$、延迟项不变（并行不叠加）：

$$
T_{\mathrm{ring}}(k) = 2(n-1)\thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \frac{\beta}{k} \qquad (6)
$$

**符号**： $k$ = 并行 ring/channel 数； $\beta/k$ = 聚合带宽 $`k \cdot \mathrm{BW}_{\mathrm{eff}}`$ 的每字节时间。代入 $n=8$、 $M=1$ GB、 $\alpha=2$ μs： $k=1$ 带宽项 4.72 ms； $k=4$ 带宽项 1.18 ms（延迟项仍 0.028 ms）； $k=8$ 带宽项 0.59 ms。NCCL 的 channel 上限 MAXCHANNELS = 64（src/include/device.h L80，版本锚 2.32.7-1），多 channel 把域内聚合带宽吃满。口径注：式 (6) 假设 $k$ 个 ring 的带宽可线性聚合（NVSwitch 非阻塞、各 channel 走不同 link 子集），实际受每 GPU 注入带宽上限约束—— $k$ 增大到注入带宽饱和后不再增益（7.2 §3 注入带宽口径）。多 channel 的图构建与 kernel 调度归 8.4。

## 6. ring vs 星型 / recursive halving

ring 带宽最优，那星型和 recursive halving 差在哪？recap training 4.1 §2.1 / 练习 3，本节只做对照登记，不重推。

**星型归约**（不合格基线，recap training 4.1 §2.1）：所有 rank 把全量 $M$ 发给 root，root 归约后广播回去。root 链路一步承载 $2(n-1)M$，随 $n$ 线性膨胀。代入 $n=8$、 $M=1$ GB：root 收发各 7 GB（对比 ring 每 rank 1.75 GB）；training 4.1 的标尺例 7B bf16 梯度、 $n=512$ 时 root 每步收发各约 6.5 TiB。星型违反 §3 的负载均衡前提（root 热点），完成时间被 root 链路卡死——任何可用方案必须让单 rank 账与 $n$ 解耦。

**recursive halving/doubling**（Rabenseifner 2004，[Rabenseifner, 2004](../references.md)，[B]；recap training 4.1 练习 3）：每步与距离 $2^j$ 的 peer 交换半块，消息步数从 ring 的 $2(n-1)$ 压到 $2\lceil \log_2 n \rceil$，每 rank 字节数仍约 $2M$ 量级（带宽项与 ring 同量级）。代入 $n=8$：recursive halving/doubling 走 $2 \times 3 = 6$ 步 vs ring 14 步——**赢的是延迟项，不是带宽项**。大消息下两者带宽项相同，树形/蝶形只在延迟项占优；这条结论的定量展开与 ring/tree 交叉临界 $M^{\ast}$ 归 8.3。换句话说，ring 与 recursive halving/doubling 在带宽账上是同一档（都达 §3 的下界量级），真正的分野在延迟项随 $n$ 的增长方式——一个线性、一个对数，这正是下一章要解的交叉。

| 算法 | 消息步数 | 每 rank 发送 | 延迟项 | 带宽项 | 出处 |
| --- | --- | --- | --- | --- | --- |
| 星型 | 2（root 收 + 发） | root $2(n-1)M$、其余 $M$ | 小 | root 热点随 $n$ 线性 | training 4.1 §2.1 |
| ring | $2(n-1)$ | $2(n-1)/n \cdot M$（达界） | $2(n-1)\alpha$ | $2(n-1)/n \cdot M\beta$（最优） | 式 (1)(2)、命题 1 |
| recursive halving/doubling | $2\lceil \log_2 n \rceil$ | $\approx 2M$ 量级 | $2\lceil \log_2 n \rceil \alpha$ | 与 ring 同量级 | training 4.1 练习 3、8.3 |

## 7. 万卡可扩展与 NVLink-aware 分层 ring

大消息下 ring 的带宽项为什么与 $n$ 无关、万卡下延迟项又为什么成为新瓶颈？式 (1) 系数 $2(n-1)/n$ 随 $n$ 单调趋 2：代入 $n=10000$，系数 $2 \times 9999/10000 = 1.9998$，与 $n=8$ 的 1.75 同量级、与极限 2 仅差 0.01%。所以**带宽项 $2(n-1)/n \cdot M\beta \to 2M\beta$ 与 $n$ 解耦**——这是 ring 撑到万卡的核心性质（recap training 4.1 §2.2「带宽账与 $d$ 基本无关」）。但延迟项 $2(n-1)\alpha$ 随 $n$ 线性涨：代入 $n=10000$、 $\alpha=2$ μs，延迟项 $2 \times 9999 \times 2 = 39996$ μs ≈ 40 ms，已远超 $n=8$ 的 0.028 ms。**万卡下小消息的延迟项成为瓶颈**——这正是 8.3 tree 算法（延迟 $O(\log n)$）与分层 ring 的动机。

**NVLink-aware 分层 ring**（hierarchical / 2D ring）：把 ring 拆成节点内 NVLink 环 + 机间 RDMA 环两层。NCCL 2.4 blog 逐字描述 2D ring「performs a reduce-scatter operation inside the node, then multiple all-reduce operations between nodes, then an all-gather operation inside the node again」（[NVIDIA, NCCL 2.4 blog](../references.md)，[C] 开放技术博客）：节点内先 reduce-scatter（每 rank 持 $M/n_{\mathrm{node}}$ 分片），机间只对分片做 all-reduce（跨域字节压到 $1/n_{\mathrm{node}}$），节点内再 all-gather 复原。算术依据是节点内 vs 机间的数量级差（recap 7.2 §7）：NVLink 单向 450 / NIC 50 = 9×，把跨域流量压到最小就是把慢链路的使用压到最小。代入 8-GPU 节点：机间 all-reduce 的消息从 $M$ 降到 $M/8$，跨域字节省 8×。分层 ring 的 fabric 工程（rail 对齐、跨机环排列、拥塞）归 10.2，本章只交付「分层压跨域字节」的算术动机。

**分层为什么省时间**（两层账，教学推导口径）。设 $n_{\mathrm{node}} = 8$ 个节点、每节点 8 GPU（共 $n = 64$ rank），节点内有效带宽 $\beta_{\mathrm{intra}}^{-1} = 370.8$ GB/s、机间 $\beta_{\mathrm{inter}}^{-1} = 50$ GB/s（7.2 §7）。扁平 ring 把 64 rank 排一个大环，每跳都可能跨机，带宽项被慢链路（50 GB/s）卡住，延迟项 $2 \times 63 \times \alpha$ 随 64 线性。分层 ring 三段：节点内 reduce-scatter（8 rank 环，走 370.8 GB/s，消息 $M$）→ 机间 all-reduce（8 个节点代表，走 50 GB/s，但消息已压到 $M/8$）→ 节点内 all-gather（8 rank 环，走 370.8 GB/s，消息 $M/8$ 复原）。机间段虽慢，字节只有 $M/8$，慢链路的使用被压到最小；节点内段虽快，但只在域内走。这就是「把慢链路的使用压到最小」的定量含义——不是消除跨域流量，而是让跨域流量只承载分片。延迟侧，机间环只有 8 个节点代表（ $2 \times 7 \times \alpha$）而非 64 rank（ $2 \times 63 \times \alpha$），延迟项也大幅下降。分层的精确时间账与 rail 对齐、拥塞交互归 10.2，本章只立「压跨域字节 + 压跨域跳数」两条动机。NVLS（NVSwitch 网内归约，7.1 §6）是另一条 bypass：每 rank 发送从 $2(n-1)/n \cdot M$ 降到 $M$， $n=8$ 时带宽项 4.72 → 2.70 ms，与 ring 的流量下界不矛盾——NVLS 把归约搬进交换芯片、改变了「每跳必须 GPU 转发」的前提（命题 1 的归约树/广播树论证以 GPU 间转发为口径），算法走读归 8.4。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| ring 每 rank $2(n-1)/n \cdot M \approx 2M$、α-β 延迟项 $2(n-1)\alpha$ + 带宽项 $2(n-1)/n \cdot M\beta$（recap training 4.1 式 (2)(3)， $d \to n$，归纳不重推） | 式 (1)(2) | §2、练习 1 |
| 总流量下界 $\ge 2(n-1)M$：chunk 归约树（未齐部分和）+ 广播树（已齐全和）两类不相交传输，每 chunk $\ge 2(n-1)M/n$ 求和 | 式 (3)，命题 1 | §3、练习 3 |
| 负载均衡每 rank $\ge 2(n-1)/n \cdot M$，ring 达界带宽最优；单阶段界 $(n-1)/n \cdot M$ 单独归属，勿与总量混 2× | 式 (4)(5)，推论 | §3、练习 3 |
| 弱界 $S_r \ge M$ 漏转发流量、差 1.75×（ $n=8$）不足闭合；必须切 chunk 分类传输 | §3 弱界段 | §3、练习 3 |
| 逻辑环每跳 = GPU→NVSwitch→GPU（recap 7.2）；ncclBuildRings 两道校验（loopback + 全覆盖）；下界拓扑无关、达界带宽由拓扑供给 | rings.cc L29/L51 | §4、8.4 |
| 延迟项摊薄 $2(n-1)\alpha/M \to 0$；chunking 保下界带宽可达；多 ring $T_{\mathrm{ring}}(k)$ 带宽项 $/k$（MAXCHANNELS 64） | 式 (6) | §5、练习 2 |
| 星型 root $2(n-1)M$ 热点不合格；recursive halving/doubling $2\lceil \log_2 n \rceil$ 步赢延迟不赢带宽（recap training 4.1 练习 3） | §6 对照表 | §6、8.3 |
| 带宽项与 $n$ 解耦（ $n=10000$ 系数 1.9998）撑万卡；延迟项随 $n$ 线性（万卡 ≈40 ms）→ tree/分层；分层 ring 压跨域字节 8×、NVLS 每 rank 发送 $M$ | §7 可扩展段 | §7、8.3/10.2、7.1 §6 |

## 练习

1.（计算 · [10]）ring 消息量与时间代入（式 (1)(2)， $n=8$、 $M=1$ GB、 $\beta=1/370.8$ GB/s、 $\alpha=2$ μs 教学假设）：(a) 每 rank 发送多少 GB；(b) 带宽项、延迟项、合计各多少 ms；(c) 单阶段（reduce-scatter）每 rank 发送多少 GB。校验：(a) 1.75 GB；(b) 4.72 / 0.028 / ≈4.75 ms，与 7.2 §3 逐数一致（容差 1%）；(c) 0.875 GB（单阶段是总量之半，勿混）。
2.（计算 · [20]）多 ring 并行与延迟摊薄（式 (6)）： $n=8$、 $M=1$ GB、 $\alpha=2$ μs。(a) $k=4$ 个 channel 时带宽项多少 ms、延迟项变不变；(b) 每字节摊到的延迟 $2(n-1)\alpha/M$ 多少 s/B、是 $\beta$ 的几分之一；(c) $n=10000$ 时延迟项多少 ms、系数 $2(n-1)/n$ 多少？先预测 (a) 带宽项相对 $k=1$ 的倍数再算。校验：(a) 1.18 ms、延迟项仍 0.028 ms；(b) $2.8 \times 10^{-14}$ s/B、约 1/96；(c) ≈40 ms、1.9998，与 §5/§7 逐数一致。
3.（推导 · [30]）证明命题 1（式 (3) 总流量下界）并辨析单阶段/总量：(a) 对固定 chunk 写出归约树界（ $n-1$ 个贡献各首次离开源 rank、源互异、每次 $\ge M/n$）；(b) 写出广播树界（ $n-1$ 个 rank 各须 $\ge M/n$ 全和流入）；(c) 论证两类传输不相交并相加得每 chunk $\ge 2(n-1)M/n$、求和得 $2(n-1)M$；(d) 说明单阶段界 $(n-1)/n \cdot M$ 与 all-reduce 总量 $2(n-1)/n \cdot M$ 的 2× 关系，以及弱界 $S_r \ge M$ 为什么不足。校验：(c) $n=8$、 $M=1$ GB 总流量 14 GB；(d) 单阶段 0.875 GB、总量 1.75 GB、弱界 1 GB 差 1.75×，与 §3 逐数一致。
4.（代码 · [30]）教学用，非优化实现：纯 Python 模拟 $n=4$ ring all-reduce 并验证下界达界——(a) 逐步执行 scatter-reduce 与 all-gather（每步每 rank 发 $M/n$），断言结束后每 rank 持全和向量；(b) 断言每 rank 累计发送恰 $2(n-1)/n \cdot M$、全网总流量恰 $2(n-1)M$；(c) 断言每步每 rank 发送量相同（负载均衡），从而每 rank 达式 (4) 下界。校验：断言 (b) $n=4$ 每 rank 系数 $2(n-1)/n = 1.5$、总流量 $2(n-1)M = 6M$；(c) 每步发送量恒 $M/n$，与命题 1 推论一致。

## 参考

- [Patarasuk et al., 2009](../references.md)：Bandwidth optimal all-reduce algorithms for clusters of workstations（key: patarasuk-yuan），[B] 老经典无 arXiv，JPDC 69(2):117-124，2009-02，DOI 10.1016/j.jpdc.2008.09.002，Crossref 2026-11-25 再核——ring all-reduce 带宽最优经典出处支撑；本章下界证明按 spec 命题-证明骨架独立重推，经典只作出处（§3）
- [Rabenseifner, 2004](../references.md)：Optimization of Collective Reduction Operations（key: rabenseifner），[B] 老经典无 arXiv，Springer LNCS 3036 ICCS 2004 pp 1-9，DOI 10.1007/978-3-540-24685-5_1，Crossref 2026-11-25 再核——recursive halving/doubling 对照（§6，延迟交叉展开归 8.3）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e，master 2026-11-24）fresh tarball 核实——src/graph/rings.cc ncclBuildRings L29（prev/next → 环数组 + loopback 校验 L51 + 全 rank 覆盖校验）；src/tuning/tuning_general.cc ncclTuningGetTime L58（time = lat×latCount + nBytes/(1000×bw)，α-β 代码形态，8.4 走读）；src/include/device.h MAXCHANNELS 64 L80（多 channel 上限）（§4/§5）
- [NVIDIA, NCCL 2.4 blog](../references.md)：Massively Scale Deep Learning Training with NCCL 2.4（key: nccl-tree-blog），[C] 开放技术博客、非闭源，2026-11-25——2D hierarchical ring 逐字「reduce-scatter inside the node, then multiple all-reduce between nodes, then all-gather inside the node again」（§7 分层 ring 动机；double binary tree 与 ring/tree 交叉归 8.3）
- training 4.1（纯文字指引，不挂链接）：ring 消息量式 (2) + α-β 式 (3) + scatter-reduce 不变式归纳证明（§2.2，recap 不重推）、星型基线 root 链路 $2(d-1)M$（§2.1）、recursive halving/doubling 练习 3（§6 对照）——训练账 recap 不重推（§2/§6）
- 7.1 / 7.2 / 8.1（纯文字指引，不挂链接）：NVLink 域几何与非阻塞 crossbar（7.1 §3）、NVLS 网内归约每 rank 发送 $M$（7.1 §6）、有效带宽 370.8 GB/s 与 ring 映射（7.2 §3/§4）、节点内 vs 机间 9×（7.2 §7）、原语语义与 regime 临界 $M_c$（8.1 §3/§5）——硬件账 recap 不重推（§2/§4/§5/§7）
