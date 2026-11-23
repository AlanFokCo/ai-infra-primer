# 5.2 SP / CP：序列维的切分（Sequence and Context Parallelism）

> 本章处理 5.1 留下的两个「切不动」。第一个是 TP 区段之间的复制区段：RMSNorm / dropout / 残差的激活在每卡冗余一份。它的答案是 sequence parallelism（SP）——把复制区段沿序列维切片，代价是把 TP 的 f/g 换成 all-gather / reduce-scatter 对；本章证明这次替换通信量恒等。第二个是 attention 的跨位置耦合：沿序列切分时，key 维收缩无处可切。它的答案是 context parallelism（CP），本章讲三件机制：Ring Attention 的 K/V 分块轮转加 online softmax 逐块累积（正确性归 models 4.1 的 online softmax 引理）；causal 负载均衡的两种修法（striped 取余条带、zigzag 首末配对）的推导；Ulysses 的 all-to-all 换维。最后给 SP / CP / TP 的通信账总表与适用边界 | 前置：5.1（f/g 算子、TP 通信账、复制区段冗余）、4.1（all-reduce = reduce-scatter + all-gather 的 ring 分解）、1.4（激活逐算子账）；models 4.1（online softmax 引理——本章只复用结论，纯文字指引）、models 2.3（attention 公式） | 锚点：Megatron SP（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]）、Ring Attention（[Liu et al., 2023](../references.md)，arXiv:2310.01889，[A]）、Striped Attention（[Brandon et al., 2023](../references.md)，arXiv:2311.09431，[A]）、DeepSpeed Ulysses（[Jacobs et al., 2023](../references.md)，arXiv:2309.14509，[A]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，[A]，CP16 @ 128K 配置）；实现对照 NVIDIA/Megatron-LM repo `mappings.py` 的 SP 算子族（核实于 2026-11-16）

## 1. 序列维上还剩什么

先看 5.1 §9 留下的账：TP 之后，每层激活里还有系数为 10 的一份冗余——两个 norm 的输入、dropout 掩码、残差流，完整 $[b, s, h]$ 在每卡各存一份。这些算子有一个共同点：逐位置（position-wise）。RMSNorm 对每个 token 的行内向量独立归一化（1.2 §5），dropout 逐元素，残差逐元素。逐位置算子对序列维没有任何耦合：第 $i$ 个位置的输出只依赖第 $i$ 个位置的输入。所以它们不仅可以沿宽度复制着算，也可以沿**序列**切开算——每张卡只持 $[b, s/t, h]$ 的一段，逐位置算子本地完成。所谓 sequence parallelism（SP），就是指这种把逐位置区段的激活沿序列维切给 TP 组各卡的做法。这就是 SP 的全部动机。

真正切不动的是 attention。softmax 的归一化沿 key 位置维收缩（models 2.3 式 (4)），第 $i$ 个 query 要看全部 $j \le i$ 的 K/V；沿序列切开后，任何一段的 attention 都需要其他段的 K/V。这条耦合把「沿序列切」分成两个子问题：逐位置区段的切分（SP，§2）与 attention 本体的切分（CP，§3-5）。至此并行维度的拼图也齐了：batch 维归 DP（P4），宽度维归 TP（5.1），深度维归 PP（5.3）；序列维是激活账里最后一根可切的轴，而它恰是长上下文时代涨得最快的一根。记号沿用 5.1： $t$ = TP 度数、 $s$ = 序列长、 $b$ = 每卡 micro-batch 序列数、 $h$ = hidden；本章新增 $c$ = CP 度数（spec §4.3 约定）。本章口径：通信账以 bf16 激活消息为基准单位（消息量定义见上章）；数字实例集中在练习 3 的合成几何（题面注明），正文其余数字均是被引论文口径的转述。

## 2. SP：把 all-reduce 拆成 all-gather + reduce-scatter【关键推导】

先立本节的问题：把复制区段沿序列切给各卡，进 TP 区段前要聚齐、出来要散回去——这两次搬运会不会白付通信？答案是不会，本节给出证明。**机制**。SP 区段（norm、dropout、残差）的激活沿序列维切，每卡持 $[b, s/t, h]$。进入 TP 区段（QKV 列切）前，各卡需要完整 $[b, s, h]$，要把分片聚齐——沿 $s$ 做 all-gather。出 TP 区段（行切 + 归约）后，结果要落回序列分片形态——归约并沿 $s$ 散片。所以 5.1 的 f/g 换成一对 SP 变体：

$$
g: \quad \text{前向 all-gather（沿 } s \text{）}, \quad \text{反向 reduce-scatter}; \qquad \bar{g}: \quad \text{前向 reduce-scatter}, \quad \text{反向 all-gather（沿 } s \text{）} \qquad (1)
$$

**符号**： $g$ 占据 5.1 中 f 的位置（SP→TP 入口）， $\bar{g}$ 占据 5.1 中 g 的位置（TP→SP 出口）；归约均为求和语义（5.1 §3 注记）。命名用论文原记号（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，§4.2.2 / Figure 5）——即论文的 g / ḡ。这里有一处必须点破的同名易混：此 $g$ 非 5.1 的 g。5.1 的 g 前向是 all-reduce（TP 区段出口），SP 的 g 前向是 all-gather（SP→TP 入口）；两个算子占据不同位置，后文一律按式 (1) 的口径使用。前向函数与 5.1 逐点恒等，原因各半：all-gather 只是把复制在各卡的同一份张量显式拼出（复制区段的各卡内容本就相同，gather 不改变数值）；reduce-scatter 求和后散片（TP-only 出口算子——5.1 的前向 all-reduce g——的全量输出在复制区段本来也要逐位置独立存在，散片只是不再复制）。反向由同一对算子的 backward 自动闭合，与 5.1 §3 的论证逐项同构。实现对照（NVIDIA/Megatron-LM repo `megatron/core/tensor_parallel/mappings.py`，核实于 2026-11-16）： `_GatherFromSequenceParallelRegion`（forward `_gather_along_first_dim` / backward `_reduce_scatter_along_first_dim`）即 $g$， `_ReduceScatterToSequenceParallelRegion`（forward reduce-scatter / backward all-gather）即 $\bar{g}$。

**命题**（SP 通信恒等）：TP+SP 与 TP-only 每层每方向的每卡字节数严格相等。**证明路线**：把每次 all-reduce 按 ring 分解拆成两半，逐块逐方向对账——关键一步是上章已证的「all-reduce = reduce-scatter + all-gather」。下面逐块列账。令 $M = 2bsh$ bytes 为完整激活的消息量（5.1 式 (9)）。4.1 §2.2 已经证明，ring 口径下 all-reduce = reduce-scatter + all-gather：一次 all-reduce 每卡收发 $2(t-1)/t \cdot M$；一次 all-gather 或 reduce-scatter 每卡各 $(t-1)/t \cdot M$。对照两种形态的一个块（attention 或 MLP）：

- **TP-only**：前向 1 次 all-reduce（g），每卡 $2(t-1)/t \cdot M$；反向 1 次（f），同量。
- **TP+SP**：前向 1 次 all-gather（ $g$ ）+ 1 次 reduce-scatter（ $\bar{g}$ ），每卡 $2 \times (t-1)/t \cdot M$；反向对称（ $\bar{g}$ 的 backward all-gather + $g$ 的 backward reduce-scatter），同量。

反向的通信序列也对称展开一次： $\bar{g}$ 的 backward all-gather 把行切输出的梯度从序列分片聚回全量；行切反向（5.1 式 (4)）免通信；列切反向的输入梯度部分和再由 $g$ 的 backward reduce-scatter 归约，并直接散回序列分片形态。每一步都是 5.1 §3 链式论证在 SP 算子上的重演，梯度与「TP-only 反向后再沿 $s$ 切走」逐位一致（浮点归约顺序除外）。

逐字节相等——SP 没有引入任何新通信，它只是把每次 all-reduce 沿环拆成两半、分别摆到 TP 计算区段的两个边界上：

```
             TP-only（5.1）                    TP + SP
 复制区段    x 全量复制 [b,s,h]                 x_shard [b,s/t,h]（norm/dropout/残差沿 s 切）
 入口        f：恒等（无通信）          ──►      g：all-gather  (t−1)/t·M
 TP 区段     QKV 列切 ─► attn/MLP ─► 行切        QKV 列切 ─► attn/MLP ─► 行切
 出口        g：all-reduce 2(t−1)/t·M   ──►      ḡ：reduce-scatter (t−1)/t·M
 每方向每块  2(t−1)/t·M                          2(t−1)/t·M   —— 恒等
```

**换来的是什么**？复制区段的全部激活从 $[b, s, h]$ 缩到 $[b, s/t, h]$。落到 1.4 的逐算子账上，就是系数 10 的那一份被除以 $t$：TP-only 的 $sbh(10 + 24/t + 5as/(ht))$ 变为 TP+SP 的 $sbh(34 + 5as/h)/t$（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]，Table 2 口径）——定长项与 $s^{2}$ 项全部除 $t$。同一论文还有实测：selective 重算叠加 SP 后，重算开销（口径：step 时间占比）从 4% 降到 530B / 1T 规模的 2%（1.4 §3 已引其 Figure 8 口径）。代价写在机制里：SP 把「复制区段各卡内容一致」从隐式性质变成了显式通信——all-gather 的出口处，各卡必须拿到逐位一致的全量张量。5.1 §3 的 RNG 同步注记（TP 组内共享种子）在这里是同一份纪律。

一句边界：SP 与 TP 共用同一个进程组、同一个度数（沿 $s$ 切 $t$ 份）——它不是新的并行维度，而是 TP 在序列维上的激活回收。真正的新维度在下一节。

## 3. CP 与 Ring Attention：让 attention 也沿序列切

长序列会把 attention 推成主账：scores 类中间量随 $s^{2}$ 涨（1.4 式 (1) 的 $5as/h$ 项），而 SP 帮不了 attention——softmax 的 key 维收缩把每段 query 绑在全部 key 上。所谓 context parallelism（CP），是指把整层输入沿序列维切给一个独立进程组（度数 $c$ ）、attention 靠轮转 K/V 看完全部 key 的做法。具体地：开一个独立的 CP 组，把整层输入沿 $s$ 切给 $c$ 张卡。逐位置算子照旧本地（与 SP 同构）；attention 靠**轮转 K/V** 让每段 query 看完全部 key。这就是 Ring Attention（[Liu et al., 2023](../references.md)，arXiv:2310.01889，[A]）。

**机制**。序列切成 $c$ 个连续块；rank $r$ 常驻 query 块 $Q_r$（ $[b, s/c, \ldots]$ ）与初始的 $K_r, V_r$。各卡排成环，每步把当前持有的 K/V 块发给下一卡、从上一卡收一块； $c - 1$ 步轮转后，每段 query 见过全部 $c$ 个 K/V 块。每步内做块间 attention 的**部分计算**，用 online softmax 状态累积：每行 query 维护三元组 $(m, \ell, O)$（当前最大值、当前指数和、当前未归一化输出），收到新块后按合并规则更新—— $m^{\mathrm{new}} = \max(m, m_{\mathrm{blk}})$，输出与指数和各乘相应的重缩放因子再加权相加。**命题**（轮转精确性）：这条合并规则的结合律、以及「逐块合并 = 全量 softmax」的精确等价，已在 models 4.1 §5.2 完整推出（online softmax 引理）；此处只登记用法：**轮转是精确的**。 $c$ 步后每行的 $(m, \ell, O)$ 与一次性全量 attention 逐点相同（浮点求和顺序除外），没有任何近似项。数值稳定的 max 减法也一并继承：各块的 $m_{\mathrm{blk}}$ 只影响缩放因子，不进指数的自变量。

轮转图（ $c = 4$ ，✓ = 该步对该 rank 有贡献的块；causal 下打 ✗ 的块整块被 mask）：

```
       round 0        round 1        round 2        round 3
rank0  Q0 × K0V0 ✓    Q0 × K3V3 ✗    Q0 × K2V2 ✗    Q0 × K1V1 ✗
rank1  Q1 × K1V1 ✓    Q1 × K0V0 ✓    Q1 × K3V3 ✗    Q1 × K2V2 ✗
rank2  Q2 × K2V2 ✓    Q2 × K1V1 ✓    Q2 × K0V0 ✓    Q2 × K3V3 ✗
rank3  Q3 × K3V3 ✓    Q3 × K2V2 ✓    Q3 × K1V1 ✓    Q3 × K0V0 ✓
       每步：发当前 K/V 块给下一 rank、收上一 rank 的块（与当步计算重叠）
       每 rank 内：(m, ℓ, O) 逐块合并；c 步后 = 全量 causal softmax·V
```

**通信账**。每步每卡发出一个 K 块加一个 V 块，各 $b \cdot (s/c) \cdot a_{kv} \cdot d_{\mathrm{head}}$ 元素； $c - 1$ 步合计每卡发送

$$
C_{\mathrm{ring}} = 4 \thinspace b \thinspace s \thinspace a_{kv} \thinspace d_{\mathrm{head}} \cdot \frac{c-1}{c} \hspace{5mu} \approx \hspace{5mu} 4 \thinspace b \thinspace s \thinspace a_{kv} \thinspace d_{\mathrm{head}} \quad \text{bytes / 层 / 前向} \qquad (2)
$$

**符号**：系数 4 = 2 个张量（K、V）× 2 bytes（bf16）； $a_{kv}$：KV head 数。两个结构性质值得展开。其一，通信量与 $c$ 几乎无关（系数饱和到 1），而且是**点对点流水**而非阻塞集合通信：第 $i$ 步的收发可以与第 $i$ 步的块计算重叠。每步块计算量正比于 $(s/c)^{2}$、传输量正比于 $s/c$——块越大算术强度越高，长序列恰是 overlap 最友好的区间。其二，与 5.1 式 (10) 对照：MHA（ $a_{kv} = a$ ）下式 (2) 为 $4bsh(c-1)/c$，是 TP 每层前向通信（ $8bsh(t-1)/t$ ）的一半量级。换句话说，CP 用与 TP 同阶的通信换掉了 $s^{2}$ 项的显存。代入练习 3 的合成几何（hidden 8192、MHA、序列长 131072、单序列、度数 16）算一遍：每卡每层前向约 4 GB——符号账立刻有了尺寸感。反向补一句： $\mathrm{d}K, \mathrm{d}V$ 块需沿环反向再轮转一圈（各块的反向贡献来自全部 query 块）， $\mathrm{d}Q$ 本地累积，通信量对称翻倍。

**显存账与「近无限」的读法**。CP 每卡的 attention 常驻只有本段 Q 与当前轮到的 K/V 块，相关激活从 $O(s)$ 降到 $O(s/c)$ 加一块工作缓冲。scores 类 $s^{2}$ 中间量在 FlashAttention 内核下本就不物化（models 4.1）；CP 进一步把单块的算子内尺寸压到 $(s/c)^{2}$。论文标题里的 "near-infinite context" 说的就是这条线性：显存约束下，可训练的上下文长度随 $c$ 线性放大——每加一卡，序列就能再长一份 $s/c$。代价是式 (2) 的同阶通信与 ring 步进的同步开销。

> **提点**：工程落地——每个「Q 块 × 收到的 K/V 块」的部分 attention 由支持分块 K/V 摄入的 FlashAttention 族 kernel 完成；online softmax 状态的合并（重缩放与累加）在 kernel 边界显式进行。块调度与 SRAM 驻留的 kernel 细节归 gpu 维度；训练栈侧的 CP 组配置走读归 6.2。

**归属边界**：ring 轮转与 online softmax 累积的机制出自 Ring Attention 原文；下一节的负载均衡修正（striped / zigzag）不在其中（[Liu et al., 2023](../references.md) 不含此项——出处勿张冠李戴）。指针一句：本章只管「长序列怎么切」；上下文窗口本身的训练期扩展（staged 拉长与 mid-training 配方一侧）归 7.5。

## 4. causal 负载均衡：倾斜推导与 striped / zigzag 两种修法【关键推导】

上节轮转图的 ✗ 列就是问题本身：causal mask 下 rank $r$ 的 query 块只需要 K/V 块 $0, \ldots, r$，连续切分时各 rank 的有效计算量随块位次线性倾斜。把账算出来。设每个「query 块 × key 块」对的计算量为单位 $u$（同形块对 FLOPs 相同），连续切分下 rank $r$ 的总工作量是它可以看到的块对数：

$$
w_r = (r + 1) \thinspace u, \qquad \bar{w} = \frac{1}{c} \sum_{r=0}^{c-1} w_r = \frac{c+1}{2} \thinspace u, \qquad \frac{\max_r w_r}{\bar{w}} = \frac{c \thinspace u}{(c+1) \thinspace u / 2} = \frac{2c}{c+1} \qquad (3)
$$

**符号**： $w_r$：rank $r$ 在一个层 attention 前向中的块对工作量； $\bar{w}$：跨 rank 均值。最大/均值比 $2c/(c+1)$ 随 $c$ 单调趋于 2。ring 的步进由最重的 rank 决定（每步所有 rank 同步收发），所以近一半算力在空转。 $c = 16$ 时比值为 $32/17 \approx 1.88$——这不是长尾噪声，是近 2 倍的结构性倾斜。

修法有两条路线，归属必须分清——Ring Attention 原文（2310.01889）不含任何负载均衡修正，以下两者都是后续工作。

**striped：按位置对 $c$ 取余的等距条带**（Striped Attention，[Brandon et al., 2023](../references.md)，arXiv:2311.09431，[A]）。做法是把序列位置（或细分块）按「索引对设备数取余」分给各卡：rank $r$ 持位置类 $\lbrace r, \thinspace r + c, \thinspace r + 2c, \thinspace \ldots \rbrace$，即等距条带、步长 $c$ 跨步取样（原文口径 "evenly-spaced stripes based on their residues modulo the device count"；Figure 1 的条带排列 $\lbrace 0, 4, 8, 12 \rbrace$ / $\lbrace 1, 5, 9, 13 \rbrace$ / $\ldots$ 即 $c = 4$ 实例）。效果上，causal 三角被条带均匀采样，每卡分到的可见块对数随之近似均摊。注意论文的均衡口径是**近似**——"approximately half of the query/key interactions on each device will be inhibited"、"almost entirely eliminates the workload imbalance"——不存在精确恒等式。机制好处是：不动 ring 的收发节奏，无需显式配对设计；均衡精度随条带密度提高。

**zigzag： $2c$ 个半块的首末配对**（Megatron-Core CP 的实现口径，教学参照）。把序列切成 $2c$ 个 $s/(2c)$ 小块，rank $r$ 持块 $r$ 与块 $2c - 1 - r$。设小块对工作量单位为 $u^{\prime}$（块长减半、块对 FLOPs 变四分之一， $u^{\prime} = u/4$ ），其工作量

$$
w_r^{\mathrm{zigzag}} = (r + 1) \thinspace u^{\prime} + (2c - r) \thinspace u^{\prime} = (2c + 1) \thinspace u^{\prime} \qquad (4)
$$

结果与 $r$ 无关，倾斜比精确等于 1。直觉一行：早块算得少、晚块算得多，首末互补配对后，任何 $r$ 的两块之和恒等于 $2c + 1$。总账核对： $\sum_r w_r = c \thinspace (2c + 1) \thinspace u^{\prime}$，恰为 $c$ 个 rank 各分一份。参数化的要点在「 $2c$ 个半块」而不是「每 rank 持两个 $c$ 等分整块」——后者的驻留显存与式 (2) 的收发量都要翻倍。半块口径下，每 rank 驻留序列总量仍为 $s/c$，式 (2) 的通信账一字不改；奇数 $c$ 也不存在「中间块无人配对」的问题—— $2c$ 恒为偶数，配对 $(r, \thinspace 2c-1-r)$ 总成立。ring 的重叠结构逐字保留。

> **坑**：causal 下「整块跳过 ✗」看似白省 FLOPs，触发后果是 ring 的同步收发节奏垮掉——各 rank 轮转步数不一、收发错位，重叠结构随之失效。所以两种修法的共同点是都不动 ring 机制本身，只改序列位置到 rank 的静态映射。

## 5. Ulysses：all-to-all 换维

DeepSpeed Ulysses（[Jacobs et al., 2023](../references.md)，arXiv:2309.14509，[A]）走另一条路：不轮转 K/V，而是在 attention 前后各做一次 all-to-all，把「沿序列切」换成「沿 head 切」，算完再换回。具体看形状。进入 attention 前，各卡持 $[b, s/c, a \cdot d_{\mathrm{head}}]$（序列分片、全 head）；all-to-all 之后，各卡持 $[b, s, (a/c) \cdot d_{\mathrm{head}}]$——完整序列、 $a/c$ 个 head。attention 在本地全序列上精确计算（依据同 5.1 §5 的 head 独立性）；出 attention 再做一次 all-to-all，换回 $[b, s/c, h]$，供行切投影与逐位置区段使用。

通信账：一次 all-to-all 各卡把本地张量的 $(c-1)/c$ 份发给其余 $c - 1$ 卡。Q、K、V 三个张量合算，每层前向每卡发送

$$
C_{\mathrm{ulysses}} = \underbrace{6 \thinspace b \thinspace s \thinspace a \thinspace d_{\mathrm{head}}}_{\text{QKV 换维}} \cdot \frac{c-1}{c^{2}} \hspace{5mu} + \hspace{5mu} \underbrace{2 \thinspace b \thinspace s \thinspace a \thinspace d_{\mathrm{head}}}_{\text{attention 输出换回}} \cdot \frac{c-1}{c^{2}} \hspace{5mu} = \hspace{5mu} 8 \thinspace b \thinspace s \thinspace h \cdot \frac{c-1}{c^{2}} \hspace{5mu} \approx \hspace{5mu} \frac{8bsh}{c} \quad \text{bytes} \qquad (5)
$$

**符号**：末步用 MHA 口径 $h = a \cdot d_{\mathrm{head}}$。与式 (2) 对照，结构差异全在 $c$ 的指数上：Ring 每卡发送量饱和到常数 $4bsh$，Ulysses 随 $1/c$ **下降**。代价藏在消息粒度里：all-to-all 的单对消息是 $\Theta(bsh / c^{2})$， $c$ 翻倍单消息缩四倍，很快跌进延迟主导的区间（4.1 式 (3) 的 $\alpha$ 项）；而且 $c$ 不能超过 head 数 $a$——head 是换维的原子，GQA 下 KV 侧收紧到 $a_{kv}$（同 5.1 §8 的复制问题）。Ring 没有 head 约束：单步消息 $\Theta(bsh / c)$ 只随 $1/c$ 缩，块内计算 $\Theta((s/c)^{2})$ 保证长 $s$ 下始终压得住传输。所以两者的边界互补：Ulysses 适合 head 多、序列中长的 regime（消息还没缩到延迟地板，度数不必超过 head 数）；Ring 服务超长上下文—— $c$ 可以开大到序列本身的极限，KV cache 形态与 head 结构都不挡路。沿用上面那组合成几何：Ulysses 每卡每层前向约 500 MB，约为 Ring 的 1/8。

GQA 注记：换维按 head 对齐时，共享同一组 KV 的 $g = a / a_{kv}$ 个 Q head 必须与该 KV head 落在同一 rank。也就是说，Ulysses 的切分原子是 KV head 族而非单 head： $c \le a_{kv}$ 时族内 Q head 随同搬迁； $c \gt a_{kv}$ 则只能复制 KV（与 5.1 §8 同一条边界），或退回 Ring。

## 6. 通信账总表与适用边界

把 5.1 与本章的每层前向每卡发送账摆到一张表（bf16、MHA 口径 $h = a \cdot d_{\mathrm{head}}$、消息量 $M = 2bsh$；反向各项对称翻倍）：

| 方案 | 每层前向每卡发送 | 集合形态 | 度数约束 | 换到的东西 |
| --- | --- | --- | --- | --- |
| TP（5.1） | $8bsh(t-1)/t \approx 8bsh$ | 2× all-reduce（关键路径上） | $t \le 8$，节点内 NVLink | 三态 $16N/t$ + 块内激活 $/t$ |
| SP（§2） | 与 TP 恒等（ $2 \times (t-1)/t \cdot M$ 拆半重摆） | 2× all-gather + 2× reduce-scatter | 与 TP 同组同度数 | 复制区段激活 $/t$（系数 10 那份） |
| Ring CP（§3） | $4bsh(c-1)/c \approx 4bsh$ | $c-1$ 步点对点流水（可重叠） | 无 head 约束；causal 需负载均衡修正（§4 的 striped / zigzag） | 超长 $s$ 的 attention 可切， $s^{2}$ 项显存 $/c$ |
| Ulysses（§5） | $8bsh(c-1)/c^{2} \approx 8bsh/c$ | 2× all-to-all | $c \le a$（GQA 下 $a_{kv}$ ）；消息 $\propto 1/c^{2}$ 有延迟地板 | 同上，通信随 $1/c$ 降 |

适用边界三行收束。其一，TP 留在节点内：通信-计算比随 $t$ 线性涨（5.1 式 (11)）， $t \le 8$ 是 NVLink 与 IB 量级差的落点。其二，CP 服务超长上下文：Llama-3 405B 的长上下文阶段把 CP 从 1 开到 16、序列从 8K 拉到 131072，配置 TP8 × CP16 × PP16 × DP8（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，Table 4；4.4 §5 已立该表）。在 16K 卡上，CP 是独立的乘性维度，与 TP 组正交。其三，Ulysses 占 head 多的中长序列区间：通信量随 $1/c$ 下降的优势在 head 数允许的度数内兑现，超过后让位给 Ring。三者在一次 step 内的叠加总账（与 DP / PP 的字节相加）归 5.5。

组合注记一句：SP 与 CP 同时开启时，逐位置区段的序列分片是复合的——TP 组沿 $s$ 切一次、CP 组沿 $s$ 再切一次，每卡持 $[b, s/(t \cdot c), h]$；attention 区段的 ring 轮转在 CP 组内进行，与 TP 的 f/g 边界正交。这些进程组的嵌套初始化（TP / CP / DP / PP group 树的 rank 映射）归 6.2 的 `parallel_state.py` 走读。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| SP 机制 | 复制区段沿 $s$ 切 $t$ 份；f/g 换成 $g$（AG 前向 / RS 反向）与 $\bar{g}$（RS 前向 / AG 反向），论文原记号 | 式 (1)；`mappings.py` 的 `_GatherFromSequenceParallelRegion` / `_ReduceScatterToSequenceParallelRegion` |
| SP 通信恒等 | all-reduce = RS + AG 的 ring 分解 ⇒ TP+SP 与 TP-only 每层每方向字节数严格相等；换来冗余激活 $/t$（ $sbh(34+5as/h)/t$ ） | §2 推导；4.1 式 (2)；arXiv:2205.05198 Table 2 |
| Ring CP | K/V 分块沿环轮转 $c-1$ 步，每卡常驻一段 Q；online softmax 逐块合并精确（models 4.1 引理）；每卡每层前向 $\approx 4bsh$ 点对点、可重叠 | 式 (2)；arXiv:2310.01889 |
| causal 均衡 | 连续切分倾斜比 $2c/(c+1) \to 2$；striped = 按位置对 $c$ 取余的等距条带（近似均衡，论文口径）；zigzag = $2c$ 半块首末配对，每 rank 恒为 $(2c+1)u^{\prime}$、显存与通信账不变；两者均非 Ring 原文内容 | 式 (3)(4)；arXiv:2311.09431 + Megatron-Core CP 实现 |
| Ulysses | all-to-all 换维（attention 时按 head 并、其余按序列并）；每卡每层前向 $\approx 8bsh/c$； $c \le a$、单消息 $\propto 1/c^{2}$ 有延迟地板 | 式 (5)；arXiv:2309.14509 |
| 边界 | TP ≤ 8 节点内；CP 服务超长上下文（Llama-3 CP16 @ 128K）；Ulysses 适合 head 多的中长序列 | §6 表；herd Table 4 |

## 练习

1.（推导 · [20]）SP 通信恒等的形式化：按 4.1 式 (2) 的 ring 口径，写出 all-gather 与 reduce-scatter 的每卡收发字节数（ $(t-1)/t \cdot M$ ），逐项对照 TP-only 的 f/g 对与 TP+SP 的 $g$ / $\bar{g}$ 对（式 (1) 的论文原记号）在一个 transformer 层前向 + 反向的通信序列，证明两者每卡总字节数严格相等；再指出两者的**消息步数**差多少，说明小消息 regime（ $s$ 短、 $t$ 大）下 SP 为什么会略慢。本题即本节通信恒等命题的证明闭环。校验：把度数设为 2 时，两种形态的每卡字节数应逐项相等（ring 系数在端点的自检）。
2.（推导 · [30]）负载均衡两方案：(a) 补全式 (3) 的求和与比值；(b) 对 zigzag 参数化（ $2c$ 个 $s/(2c)$ 小块、rank $r$ 持 $\lbrace r, \thinspace 2c-1-r \rbrace$ ）证明式 (4) 的工作量恒等为 $(2c+1) \thinspace u^{\prime}$，验证 $c$ 为奇数时配对依然完好，并核对每 rank 驻留序列长度（ $s/c$ ）与式 (2) 通信账均不变；(c) striped 的均衡是近似口径：说明为什么每 rank 的可见块对数随条带变细（每 rank 条带数增加）收敛到均值（提示：causal 三角被步长 $c$ 的条带均匀采样），并比较两方案对 ring 收发与 mask 实现粒度的影响（块级配对 vs 位置级取余）。
3.（计算 · [30]）Ring vs Ulysses： $h = 8192$、 MHA $a = 64$、 $d_{\mathrm{head}} = 128$、 $s = 131072$、 $b = 1$：(a) 按式 (2)(5) 分别算 $c \in \lbrace 8, 16, 32 \rbrace$ 时每层前向每卡发送字节数；(b) 跨节点 IB 每卡 50 GB/s 下换算时间，并与单步块计算时间（按每卡有效 400 TFLOPS 折算 attention 块对 FLOPs）对照，判断各度数下谁能被重叠藏住；(c) Ulysses 在 $c = 32$ 时的单对消息字节数是多少，对照 4.1 式 (3) 的 $\alpha$ 项说明延迟地板何时接管。先预测哪一列随度数下降、哪一列饱和，再算。校验：Ring 一列在度数翻倍时几乎不变，Ulysses 一列近似减半。

## 参考

- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——SP 机制与 f/g 的 AG/RS 替换、通信量恒等口径、激活账 Table 2（TP+SP 为 $sbh(34+5as/h)/t$ ）、selective + SP 的重算开销实测（Figure 8）
- [Liu et al., 2023](../references.md)：Ring Attention with Blockwise Transformers for Near-Infinite Context，arXiv:2310.01889，[A]（key: ring-attention）——K/V 分块轮转 + online softmax 逐块累积的机制出处；causal 负载均衡不在本文（勿张冠李戴）
- [Brandon et al., 2023](../references.md)：Striped Attention: Faster Ring Attention for Causal Transformers，arXiv:2311.09431，[A]（key: striped-attention）——按位置对设备数取余的等距条带切分（residues modulo device count）与近似负载均衡口径（§4 式 (3) 问题的 striped 修法）；官方 repo：https://github.com/exists-forall/striped_attention（论文首页脚注出处，2026-11-17 核实）
- [Jacobs et al., 2023](../references.md)：DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models，arXiv:2309.14509，[A]（key: ulysses）——all-to-all 换维机制与长序列训练的通信设计
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——f/g 算子与 TP 基线（5.1）；SP 算子实现对照 `megatron/core/tensor_parallel/mappings.py`（核实于 2026-11-16）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——TP8 × CP16 × PP16 × DP8 @ 131072 的长上下文配置（Table 4）
