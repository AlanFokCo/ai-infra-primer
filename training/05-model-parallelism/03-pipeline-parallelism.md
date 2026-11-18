# 5.3 PP：流水线气泡与 Schedule（Pipeline Parallelism: Bubbles and Schedules）

> 流水线并行（pipeline parallelism, PP）把模型按层切成 $p$ 段、放到 $p$ 组设备上，micro-batch 在段之间流动。什么时候需要它：DP/ZeRO 的决策树走到「判定线之外」（4.4 §6）的时候。本章回答一个问题——每个 stage 在每个时刻，该算哪个 micro-batch 的前向还是反向。五个答案依次展开：GPipe 的全前向再全反向（气泡占比 $(p-1)/(m+p-1)$ 的推导与 $m \gg p$ 极限）；1F1B（气泡不变，峰值激活从 $m$ 份降到 $\le p$ 份）；interleaved 1F1B（用 $v$ 倍通信换气泡再除 $v$）；Zero Bubble（把反向拆成 input-grad 与 weight-grad 两路，拿后者填气泡）；DeepSeek-V3 DualPipe（双向流水线，计算与通信互相重叠）。最后落到三张账：激活内存、stage 均衡、Megatron `schedules.py` 的实现走读 | 前置：1.5（micro-batch 与累加语义）、1.2（matmul 反向的两个 matmul）、1.4（激活账与重计算）、4.1（通信的时间模型）；models 1.2（ $6ND$ 口径） | 锚点：Megatron-LM repo `megatron/core/pipeline_parallel/schedules.py`（[A]，1F1B 与 interleaved 的官方实现）、GPipe（[Huang et al., 2019](../references.md)，arXiv:1811.06965，[B]）、Megatron 集群论文（[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]）、Zero Bubble（[Qi et al., 2024](../references.md)，arXiv:2401.10241，[A]）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）

## 1. PP 的切分语义：按层分 stage，激活点对点传递

显存装不下整个模型时，先从哪里切？PP 的切法只有一句话：把 $L$ 层按连续段切成 $p$ 个 stage，stage $i$ 的设备组只持有第 $i$ 段的参数、梯度与优化器状态。三族切分各切不同的东西：DP/ZeRO 切数据与状态的存放，模型本体在逻辑上保持完整（4.1/4.2）；TP 切每一层张量的内部（5.1）；PP 切层与层之间的边界。三者的显存效应正交——PP 先把三态与激活都除以 $p$，因为每个 stage 只存本段；每段内部还可以再叠 DP/ZeRO/TP。组合的整除约束与总账归 5.5，本章只管 PP 这一维。

stage 之间的通信是点对点（P2P）send/recv，不是集合通信。前向时，stage $i$ 收上一段发来的输入激活，算完发给下一段；反向时，收后一段传来的梯度，算完发回。每个 micro-batch 在每条边界上前向传一次激活、反向传一次梯度，各 $b \cdot T \cdot h$ 个元素（[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]，§3.2 的口径），bf16 下各 $2bTh$ bytes：

**符号**： $b$：micro-batch 序列数； $T$：序列长； $h$：hidden 维。把这个数和 DP 对照着看：DP 每 step 每 rank 的 all-reduce 约 $4N$ bytes（4.4 §4），随参数量涨；PP 的边界通信与参数量 $N$ 无关，只随激活体积走。这就是「PP 适合跨节点、TP 留在节点内」这条组合规则的账面根源（5.5）。

调度问题的形状由依赖结构决定。记 $F(i, j)$ 为 stage $i$ 上 micro-batch $j$ 的前向、 $B(i, j)$ 为其反向，则

$$
F(i, j) \leftarrow F(i-1, j), \qquad B(i, j) \leftarrow B(i+1, j) \qquad (1)
$$

**符号**： $\leftarrow$ 表示「依赖、必须晚于」。换句话说：一条 micro-batch 的前向沿 stage 正序推进，反向沿 stage 逆序回传。只跑一个 micro-batch 会发生什么？式 (1) 让任意时刻至多一个 stage 在工作， $p$ 台设备的利用率只有 $1/p$——这是 naive 模型并行的下场（GPipe 论文 Figure 2b 画的正是这个结构，[Huang et al., 2019](../references.md)，arXiv:1811.06965，[B]）。

解法与 1.5 是同一个动作：把 batch 拆成 $m$ 个 micro-batch（梯度累加保证数学等价，1.5 式 (3)），让不同 stage 同时流不同的 micro-batch。**schedule（调度）要回答的问题只有一个：每个 stage 在每个时刻，算哪个 micro-batch 的前向还是反向**。本章余下部分给出四种显存/通信约束下的四个答案。先统一记法： $p$ 是 stage 数（PP 度数，spec §4.3 约定）， $m$ 是 micro-batch 数；单 micro-batch 在单 stage 上的前向耗时记为 $t_f$、反向耗时记为 $t_b$。所有推导先假设各 stage 时长均匀（均衡假设），§8 处理不均衡。一个数量关系后面会反复用到： $t_b \approx 2 t_f$，它是 1.1「反向 FLOPs ≈ 2× 前向」的直接推论。Megatron 论文明确其气泡推导不依赖这个系数（arXiv:2104.04473，Figure 3 注），下面的公式也都不含这个比值。

## 2. GPipe：全前向再全反向，与气泡公式

最简单的调度长什么样？GPipe 的规则（[Huang et al., 2019](../references.md)，arXiv:1811.06965，[B]，§2.2）：每个 stage 先把 $m$ 个 micro-batch 的前向全部跑完，再把 $m$ 个反向全部跑完。各 micro-batch 的梯度在本地累加，全部反向结束后做一次参数更新（pipeline flush）。flush 保证严格的同步语义：无论 $p$ 是多少，参数轨迹与单机顺序跑 $m$ 个 micro-batch 的累加逐步相同（1.5 式 (3)）。GPipe 原文把「梯度更新与切分数无关」列为设计目标。

把 $p = 4$ 、 $m = 4$ 的时序画出来（F/B 等宽格仅为依赖结构示意，与 GPipe 原文 Figure 2c 同画法；按时长展开见本节末的账与 §3 末的模拟器）：

```
时间 →   1   2   3   4   5   6   7   8   9   10  11  12  13  14
stage1   F1  F2  F3  F4  ·   ·   ·   ·   ·   ·   B1  B2  B3  B4
stage2   ·   F1  F2  F3  F4  ·   ·   ·   ·   B1  B2  B3  B4  ·
stage3   ·   ·   F1  F2  F3  F4  ·   ·   B1  B2  B3  B4  ·   ·
stage4   ·   ·   ·   F1  F2  F3  F4  B1  B2  B3  B4  ·   ·   ·
         |← p−1 格前向气泡        |← p−1 格反向气泡
flush：t = 14 之后统一参数更新（同步语义）
```

**气泡推导（本章关键推导之一）**。只需追踪两个时刻。第一步，末段的前向。micro-batch $j$ 的前向到达 stage $p$ 之前要走 $p$ 段，而 stage $p$ 自己的 $m$ 个前向是连续的，所以 stage $p$ 的最后一个前向结束于

$$
t_{\mathrm{F,end}} = (p - 1)\thinspace t_f + m \thinspace t_f \qquad (2)
$$

**符号**： $(p-1)\thinspace t_f$ 是首 micro-batch 走完前向链的时间（每段一跳 $t_f$），之后每段每 micro-batch 再续 $t_f$。第二步，反向链。stage $p$ 的反向紧跟其前向连续排——梯度的源头在它这里，没有等待——micro-batch $j$ 的反向在 stage $p$ 结束于 $t_{\mathrm{F,end}} + j \thinspace t_b$。反向沿 stage 逆序每段再加一跳 $t_b$，所以 micro-batch $j$ 在 stage 1 的反向结束于 $t_{\mathrm{F,end}} + (j + p - 1)\thinspace t_b$。取 $j = m$，得整个迭代的跨度（makespan）：

$$
T = (m + p - 1)\thinspace (t_f + t_b) \qquad (3)
$$

每个 stage 的有效工作时间是 $m \thinspace (t_f + t_b)$。总跨度减去它，就是每个 stage 的空闲——**流水线气泡（pipeline bubble）**：

$$
t_{\mathrm{pb}} = T - m \thinspace (t_f + t_b) = (p - 1)\thinspace (t_f + t_b) \qquad (4)
$$

气泡占多大比例？文献里有两个归一化口径，引用时先看分母。占总迭代时间的比例（GPipe 原文口径，其记号 $K = p$、 $M = m$）：

$$
\text{气泡占比} = \frac{t_{\mathrm{pb}}}{T} = \frac{p - 1}{m + p - 1} \qquad (5)
$$

对理想计算时间的比例（Megatron 论文的口径，arXiv:2104.04473 §2.2.1）：

$$
\frac{t_{\mathrm{pb}}}{t_{\mathrm{id}}} = \frac{p - 1}{m}, \qquad t_{\mathrm{id}} = m \thinspace (t_f + t_b) \qquad (6)
$$

两个口径只差一件事：分母含不含气泡本身。式 (5) = 式 (6) ÷（1 + 式 (6)）。 $m \gg p$ 时两式都趋于 $(p-1)/m \to 0$。这是式 (5) 的全部信息：**消气泡靠的是 micro-batch 数对 stage 数的过采样**。GPipe 的经验规则是 $m \ge 4p$ 时气泡「几乎可忽略」（原文 §2.3 口径，并注明部分原因是重计算的反向可以提前调度、不必等浅层梯度）。代入式 (5) 看这个「可忽略」有多大： $m = 4p$ 时占比 $(p-1)/(5p-1)$，大 $p$ 极限恰为 20%——「可忽略」是工程判断，账面上是约 1/5 的空闲。

代价在显存。GPipe 把全部反向排在全部前向之后，所以每个 stage 必须为**全部 $m$ 个** in-flight micro-batch 保存供反向使用的激活：每 stage 峰值激活 $= m$ 份「单 micro-batch 单 stage 激活」。气泡要小（ $m$ 大）、显存要小（ $m$ 小），两者在 GPipe 上正面相撞。GPipe 的配套答案是 re-materialization（重计算，机制归 1.4）：只保存各 micro-batch 的 stage 边界输入，反向时重算段内激活。峰值由此降到 $m$ 份边界激活（每份 $2bTh$ bytes）加一段重算工作区——原文 §2.3 的口径是峰值激活从 $O(N_{\mathrm{seq}} L)$ 降到 $O(N_{\mathrm{seq}} + (L/K)(N_{\mathrm{seq}}/M))$（其记号 $N_{\mathrm{seq}}$ = mini-batch 大小、 $L$ = 层数、 $K$ = 段数、 $M$ = micro-batch 数）。重计算压低了显存的量级，但 $m$ 份边界激活这个乘数仍在。1F1B 要动的就是它。

## 3. 1F1B：气泡不动，峰值激活从 $m$ 份降到 $\le p$ 份

1F1B 的观察只有一条：反向不必等全部前向。看式 (1)： $B(i, j)$ 依赖的只是 $B(i+1, j)$ 与本 stage 自己的 $F(i, j)$。后一段把 micro-batch $j$ 的梯度送回来，本段就能立刻开始它的反向；反向一完成，这份 micro-batch 的激活就能释放。早反向就是早释放。这个调度即 PipeDream-Flush / 1F1B（one-forward-one-backward；[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]，§2.2.1）。它分三相：

- **warmup（预热）**：stage $i$（从 1 计）先连续跑 $p - i$ 个前向，不发反向；
- **steady（稳态）**：1 个前向 + 1 个反向严格交替（1F1B 的名字由此而来）；
- **cooldown（排空）**：不再发新前向，把剩余 in-flight micro-batch 的反向逐个跑完。

```
时间 →  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22
stage1  F1  F2  F3  F4  ·   ·   ·   B1  F5  B2  F6  B3  F7  B4  F8  B5  ·   B6  ·   B7  ·   B8
stage2  ·   F1  F2  F3  ·   ·   B1  F4  B2  F5  B3  F6  B4  F7  B5  F8  B6  ·   B7  ·   B8  ·
stage3  ·   ·   F1  F2  ·   B1  F3  B2  F4  B3  F5  B4  F6  B5  F7  B6  F8  B7  ·   B8  ·   ·
stage4  ·   ·   ·   F1  B1  F2  B2  F3  B3  F4  B4  F5  B5  F6  B6  F7  B7  F8  B8  ·   ·   ·
（p = 4，m = 8，每格时长按 t_f = t_b 画。注意 B 的级联逐段错开一格（B1 依次在
 stage4/3/2/1 的第 5/6/7/8 格）——式 (1) 的依赖在格粒度下逐条满足：本图与本节末
 模拟器在 t_f = t_b = 1 下的输出逐格一致，T = 22 = (m+p−1)(t_f+t_b)，每 stage
 空闲 6 格 = (p−1)(t_f+t_b)。首段在首个 B1 到达前的三格等待即气泡的一部分。
 Megatron 论文 Figure 4 按 t_b = 2 t_f 的块宽绘制，本图是它的等宽格重绘——
 块宽改变只伸缩绝对时长，气泡占比（式 (5)）与 in-flight 峰值（式 (7)）不变。）
```

**峰值激活论证（本章关键推导之二）**。把 stage $i$ 在时刻 $t$ 的 in-flight 数定义为「前向已完成、反向尚未完成的 micro-batch 数」——这正是必须为反向保存激活的份数。逐相看这个数怎么变：

1. warmup 结束时，stage $i$ 的 in-flight 恰为 $p - i$（发了 $p - i$ 个 F、零个 B）；
2. 稳态中每个「F 后接 B」对让 in-flight 先 +1 再 −1，净变化为零；峰值出现在稳态的 F 刚完成、B 未完成的瞬间： $p - i + 1$；
3. cooldown 只减不增。

还有一个问题要分开说：B 什么时候到达，warmup 起什么作用。沿首条反向链算：前向 $p$ 跳（ $F(1,1)$ 到 $F(p,1)$ ）加反向 $p$ 跳，stage 1 的首个反向 $B(1,1)$ 要到 $p \thinspace t_f + (p-1)\thinspace t_b$ 才能开始；而 stage 1 此时已排完 $p$ 个前向（warmup 的 $p - 1$ 个加稳态第 1 个），所以等待恰为 $(p-1)\thinspace t_b$。这段等待不是调度失误，它是气泡的一部分——式 (4) 的 $(p-1)(t_f+t_b)$ 就住在各段的这类等待里。warmup 数 $p - i$ 的真正作用是**给显存封顶**：它把 stage $i$ 的 in-flight 固定在 $p - i$，稳态交替后至多 $+1$。等待只加空闲、不加在飞份数，所以这个上界与梯度到达的早晚无关。把 warmup 开大可以填掉等待，代价是峰值同步上涨——warmup $= m$ 的极端就是 GPipe。于是

$$
\text{stage } i \text{ 的峰值 in-flight} \le p - i + 1 \le p \qquad (7)
$$

**符号**：峰值在 stage 1 取到 $p$。对照 GPipe 的每 stage $m$ 份：1F1B 把峰值激活从 $m$ 份压到 $\le p$ 份（Megatron 论文原话："activations to be stashed for $p$ or fewer microbatches, compared to $m$ microbatches for the GPipe schedule"）。当 $m \gg p$ ，这是数量级的差别——也正是 $m \gg p$ 这个气泡条件（§2）在显存上第一次变得可行。

**气泡不变**。逐 stage 的空闲总量与 GPipe 相同： $t_{\mathrm{pb}} = (p-1)(t_f + t_b)$，式 (3) 的 makespan 不变（Megatron 论文原话："the time spent in the bubble is the same for this new schedule"）。论证走关键路径。makespan 由 stage 1 的最后一个反向 $B(1, m)$ 的完成时刻决定，它被两条链夹住：micro-batch $m$ 的全程链（stage 1 串行发 $m$ 个前向 $\ge m t_f$，再过正向 $p - 1$ 跳与反向 $p$ 跳），与 stage 1 自己的工作量（ $m(t_f + t_b)$ 的活，加首份反向到达前的等待）。1F1B 没有改变任何一条依赖边，只是把同一团空闲挪了位置：GPipe 里 stage 1 的空闲是中段一整块；1F1B 里 stage 1 的空闲仍在中段（等梯度回来），stage $p$ 的空闲则变成开局 $(p-1)\thinspace t_f$ 与末尾 $(p-1)\thinspace t_b$ 两段。§9 的 Megatron 代码与下面的模拟器，各自从实现与数值两头确认这笔账。

下面的离散步进模拟器（教学用，非优化实现）把式 (3)(4)(7) 变成可断言的事实：按式 (1) 的依赖与各 stage 的任务队列（GPipe = 全 F 后全 B；1F1B = warmup $p-i$ 个 F 后 F/B 交替）做事件模拟：

```python
# 教学用，非优化实现：GPipe 与 1F1B 的离散步进模拟
# 断言：T = (m + p - 1) * (tf + tb)（两调度相同）；峰值 in-flight：GPipe = m，1F1B <= p
def simulate(p, m, tf, tb, schedule):
    f_end, b_end = {}, {}          # F(i,j)/B(i,j) 的完成时刻
    free = [0] * (p + 1)           # 各 stage 的空闲起点
    done_f = [0] * (p + 1); done_b = [0] * (p + 1)
    peak = [0] * (p + 1)           # 各 stage 的峰值 in-flight
    queues = [None]
    for i in range(1, p + 1):
        if schedule == "gpipe":
            q = [("F", j) for j in range(1, m + 1)] + [("B", j) for j in range(1, m + 1)]
        else:  # 1f1b：warmup (p - i) 个 F，随后 F/B 交替，尾部排空剩余 B
            warmup = min(p - i, m)
            q = [("F", j) for j in range(1, warmup + 1)]
            nb = 0
            for j in range(warmup + 1, m + 1):
                q.append(("F", j)); nb += 1; q.append(("B", nb))
            while nb < m:
                nb += 1; q.append(("B", nb))
        queues.append(q)
    remaining = sum(len(q) for q in queues if q)
    while remaining:
        best = None
        for i in range(1, p + 1):
            if not queues[i]:
                continue
            kind, j = queues[i][0]
            if kind == "F":
                dep = f_end.get((i - 1, j)) if i > 1 else 0
            elif i == p:
                dep = f_end[(i, j)]            # 末 stage：B 紧跟自己的 F
            else:
                dep = b_end.get((i + 1, j))    # 式 (1)：B(i,j) 依赖 B(i+1,j)
            if dep is None:
                continue                       # 依赖未就绪
            start = max(free[i], dep)
            if best is None or start < best[0]:
                best = (start, i, kind, j)
        start, i, kind, j = best
        free[i] = start + (tf if kind == "F" else tb)
        if kind == "F":
            f_end[(i, j)] = free[i]; done_f[i] += 1
        else:
            b_end[(i, j)] = free[i]; done_b[i] += 1
        peak[i] = max(peak[i], done_f[i] - done_b[i])   # in-flight = 已前向 − 已反向
        queues[i].pop(0); remaining -= 1
    return max(free), peak

# 实测（已运行通过）：p=4, m=8, tf=1, tb=2 → T = 33 = (8+3)·3，两调度相同；
# 峰值 in-flight：GPipe 各 stage [8, 8, 8, 8]，1F1B 各 stage [4, 3, 2, 1]
```

两个读数把 §2-3 合拢：气泡相同（ $T$ 逐位相等）、峰值激活从 $m$ = 8 份降到 $\le p$ = 4 份且逐 stage 递减（stage $i$ 恰为 $p - i + 1$，式 (7) 取等）。练习 3 用它扫 $(p, m, t_f, t_b)$ 的网格。

## 4. Interleaved 1F1B：气泡再除 $v$，通信乘 $v$

气泡公式 (6) 里还有一个可动的量：单个 micro-batch 在单个 stage 上的耗时。Interleaved 1F1B（[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]，§2.2.2）的做法是给每台设备分配 $v$ 个**不相邻**的层段（virtual stage / model chunk），而不是一段连续层：

```
设备 → chunk 映射（p = 4，v = 2，共 8 个 chunk；原 16 层每 chunk 2 层）
device1: chunk1（层 1–2）  + chunk5（层 9–10）
device2: chunk2（层 3–4）  + chunk6（层 11–12）
device3: chunk3（层 5–6）  + chunk7（层 13–14）
device4: chunk4（层 7–8）  + chunk8（层 15–16）
一个 micro-batch 的前向路径：d1.c1 → d2.c2 → d3.c3 → d4.c4 → d1.c5 → d2.c6 → d3.c7 → d4.c8
```

每 chunk 的计算量是原来的 $1/v$，单 micro-batch 单 chunk 的前向/反向耗时随之变成 $t_f / v$ 、 $t_b / v$。流水线的「级数」从 $p$ 变成 $p \thinspace v$（每条 micro-batch 要走 $p \thinspace v$ 个 chunk），但每级的耗时同步除以 $v$。把式 (4) 按 chunk 粒度重写：气泡仍由首条梯度回程的 $p - 1$ 个 stage 级差构成，但每一跳只剩 chunk 粒度的时长（原文 Figure 4：interleaved 时间线的 flush 更早发生）。得

$$
t_{\mathrm{pb}}^{\mathrm{int}} = \frac{(p - 1)\thinspace (t_f + t_b)}{v}, \qquad
\text{气泡占比} = \frac{1}{v} \cdot \frac{p - 1}{m} \qquad (8)
$$

**符号**： $v$：每台设备的 model chunk 数（virtual pipeline size）。直观读法：同一批空闲被切成了 $v$ 倍细的颗粒，填充率不变，总时长缩短——等价于把气泡除了 $v$。

代价写在通信侧。原来相邻设备对之间只有 1 条边界，现在有 $v$ 条：device1↔device2 之间要传 chunk1→chunk2 与 chunk5→chunk6 两处的激活与梯度。每 micro-batch 的 P2P 消息数与字节数同步乘 $v$（原文原话："the amount of communication also increases by $v$"）。另有一条约束： $m$ 必须是 $p$ 的整数倍（原文 §2.2.2）。Megatron 实现里 interleaved 的 warmup 计数为 `2·(p − rank − 1) + (v − 1)·group_size`（§9）。

多出来的 $v$ 倍通信还能再削一层。当 TP 度为 $t$ 时，同一份边界激活会从 $t$ 个 TP rank 各发一遍——跨节点 IB 上是 $t$ 份冗余流量。论文的 **scatter/gather 优化**针对这一点：发送端把张量切成 $t$ 块，每个 TP rank 只经自己的 IB 卡发一块；接收端在节点内用 NVLink all-gather 拼回全量。跨节点流量由此除以 $t$（机制见原文 §4.1 / Figure 9；实测对通信密集配置提速最高 11%，§5.7）。实测规律一并记录：interleaved 相对 1F1B 的吞吐优势随 batch 增大而收窄——大 batch 下默认调度的气泡本就被摊薄，而 interleaved 的通信量随 batch 线性涨；不开 scatter/gather 时，大 batch 下默认调度反超（原文 Figure 12 及注）。论文的旗舰数字：1T 参数模型在 3072 张 A100 上训到 502 PFLOP/s 聚合吞吐、单卡 52% 峰值（论文摘要与 §5）。

## 5. Zero Bubble：把反向拆成 B 与 W 两路

1F1B 与 interleaved 都在「F 和 B 两个原子」的粒度上排程。Zero Bubble（[Qi et al., 2024](../references.md)，arXiv:2401.10241，[A]）把切分推进到原子内部。一层的反向在数学上本来就是两个独立的 vjp——1.2 的 matmul 反向逐字如此： $\partial L/\partial X = G W^{\top}$ 与 $\partial L/\partial W = X^{\top} G$ 两个同形 matmul：

$$
B:\quad g_x = \left(\frac{\partial f}{\partial x}\right)^{\top} g_y \qquad\qquad
W:\quad g_W = \left(\frac{\partial f}{\partial W}\right)^{\top} g_y \qquad (9)
$$

**符号**： $f(x, W)$：某层（或某 stage 内一原子块）的前向映射； $g_y$：上游传回的输出梯度； $g_x$：input-grad，要沿式 (1) 继续传回前一个 stage； $g_W$：weight-grad，写进本 stage 的梯度缓冲即终点。论文记号 $T_F$ / $T_B$ / $T_W$ 为单 micro-batch 单 stage 上 F/B/W 的耗时。前向的每个 matmul 在反向对应两个同 FLOPs 的 matmul，分属 B 与 W，所以

$$
T_B + T_W = 2 \thinspace T_F, \qquad T_W < T_F < T_B \qquad (10)
$$

（论文 Table 1 的逐层账；手工调度的简化假设取 $T_F = T_B = T_W$，与 GPipe Figure 2c 的等时长画法同源。）

**依赖论证（本章关键推导之三）**：式 (1) 的链式依赖只穿过 B—— $B(i-1, j)$ 需要 $B(i, j)$ 产出的 $g_x$。 $W(i, j)$ 的输入（ $g_y$ 与本段激活）在同 stage 的 $B(i, j)$ 完成时就全部就位，不再跨 stage。所以 **W 可以排在其对应 B 之后的任意空档里**，它是调度器手里的自由填充物。经典实现把 B 与 W 打包成一个 backward()：这在 DP 下无害（weight-grad 的通信可以与下一层反向重叠，4.1 §3），在 PP 下却人为加长了链上的串行段——stage $i-1$ 要等 $W(i)$ 也算完，才拿得到 $g_x$。拆开之后，链上只传 B 的时间，W 被挪去填气泡。

内存账同步改写。记 $M_B$ / $M_W$ 为单 micro-batch 单 stage 上为 B / W 保存的激活量。F 完成时净增 $M_B$（为整个反向存的激活）；B 完成时释放不再需要的部分，但要把 W 的输入（ $g_y$ 与该层输入）留住，净变化 $M_W - M_B$；W 完成后释放 $M_W$。论文 Table 1 的逐层账给出 $M_W < M_B$，因为 W 只需要自己那两个 matmul 的输入。峰值落在哪、有多大，由调度决定：

| 调度 | 气泡时间 | 峰值激活（全机最大值） |
| --- | --- | --- |
| 1F1B | $(p-1)\thinspace(T_F + T_B + T_W)$ | $p \thinspace M_B$ |
| ZB-H1 | $(p-1)\thinspace(T_F + T_B - T_W)$ | $p \thinspace M_B$ |
| ZB-H2 | $(p-1)\thinspace(T_F + T_B - 2\thinspace T_W)$ | $(2p-1)\thinspace M_B$ |

（这是 Zero Bubble 论文 Table 2 的转写；1F1B 一行即式 (4) 换记号—— $T_B + T_W$ 合起来是过去的「全反向 $t_b$」。）两个手工调度的构造：

- **ZB-H1（省显存档）**：骨架跟随 1F1B，按 warmup 数调整各 stage 首个 W 的启动点，使所有 worker 的 in-flight 数一致。效果是每个 stage 的 B 都比 1F1B 更早启动（不再拖着 W），尾部气泡被推迟启动的 W 填掉。worker $i$ 的峰值激活为 $(p - i + 1)\thinspace M_B + (i - 1)\thinspace M_W$；由 $M_W < M_B$ 知最大在 worker 1： $p \thinspace M_B$，与 1F1B 严格相同。等时长假设 $T_F = T_B = T_W$ 下，气泡从 1F1B 的 $3(p-1)\thinspace T_F$ 降到 $(p-1)\thinspace T_F$——**恰好三分之一**（论文原文口径："the bubble size is reduced to a third of 1F1B's size"）。
- **ZB-H2（真零气泡档）**：允许显存超 1F1B 且 $m$ 足够大时，warmup 期多排 F，把「首个 B 之前」的空档填满；尾部把各 stage 的 W 重排，调度轮廓从梯形变成平行四边形（各 stage 首尾对齐），气泡归零：等时长假设下 $(p-1)(T_F + T_B - 2T_W) = 0$。worker $i$ 的峰值激活 $(2p - 2i + 1)\thinspace M_B + (2i - 2)\thinspace M_W$，最大 $(2p - 1)\thinspace M_B$——接近 1F1B 的两倍。这就是零气泡的显存价格。

```
反向拆分的依赖图（一层内；出处：Zero Bubble 论文 Figure 1 的 ASCII 重绘）
  g_y ──┬──► B：g_x = (∂f/∂x)ᵀ g_y ──► 传回前一 stage（链式依赖，不可推迟）
        └──► W：g_W = (∂f/∂W)ᵀ g_y ──► 本地梯度缓冲（同 stage 内自由排序）

ZB-H2 的轮廓（每行一个 stage，▓ = F/B 占用，W = weight-grad 填充；Figure 3 的示意重绘）
1F1B（梯形）            ZB-H2（平行四边形）
stage1 ▓▓▓▓▓▓▓▓▓▓····     stage1 ▓▓▓▓▓▓▓▓▓▓WWWW
stage2 ·▓▓▓▓▓▓▓▓▓▓··      stage2 ·▓▓▓▓▓▓▓▓▓▓WWWW
stage3 ··▓▓▓▓▓▓▓▓▓▓·      stage3 ··▓▓▓▓▓▓▓▓▓▓WWWW
stage4 ···▓▓▓▓▓▓▓▓▓▓      stage4 ···▓▓▓▓▓▓▓▓▓WWWW
```

ZB-H2 还有一个同步语义问题要解决：PP 实践中 optimizer step 前有跨 stage 的同步（全局梯度范数裁剪的 all-reduce——2.4 的机制；混合精度的 inf/NaN 全局检查——1.3），它会打破平行四边形。论文 §4 的旁路方案是 **post-validation**。optimizer step 前只做沿流水线的逐段部分归约（stage 间接力传递），各 stage 按部分状态先更新（发现 NaN 则跳过）；完全归约的全局状态在**下一轮迭代的 warmup 期**从末段回传到首段，各 stage 收到后校验上一步是否合法，需要修正时 rollback 重做一次更新。同步语义保持——任何时刻全体等价于同一参数版本上的同步更新——流水线内的硬同步点消失。工程前提是该校验的失败率足够低（裁剪触发率低、数值异常罕见）。

手工调度之外，论文还给了一套自动调度（§3）：以实测的 $T_F, T_B, T_W, T_{\mathrm{comm}}$ 与显存上限 $M_{\mathrm{limit}}$ 为输入做启发式搜索，必要时接 ILP 精修。启发式的规则四条：warmup 内在显存上限内尽量多排 F；稳态按 1F-1B-1W；空档 $\ge T_W$ 就插 W；撞显存上限时也插 W 回收内存。两个实例：**ZB-1p**（上限 $p \thinspace M_B$，与 1F1B 同峰值）、**ZB-2p**（上限 $2p \thinspace M_B$——原文实测在 $T_F \approx T_B$ 、通信较小的条件下， $2p \thinspace M_B$ 是达到近零气泡的经验阈值，Table 5 多数设置下气泡率 < 1%）。§6 还有 **ZB-V**：把 $2p$ 个 chunk 按 V 形（首→尾→首）分给各 worker，每个 micro-batch 的前向与反向从同一 worker 出发，首 worker 无需等末段回传即可早开 B——在 $T_F = T_B = T_W$ 假设下以 $p \thinspace M_B$ 的显存做到零气泡（原文 §6）。实测口径：相对 1F1B 吞吐提升至多 23%（同显存上限）、放松显存约束后到 31%（论文摘要）；实验平台为至多 32 张 A100-80G（4 节点 RoCE）。作者还用固定种子验证了 ZB-1p/ZB-2p 与 1F1B 的 loss 曲线逐位一致（bit-to-bit identical，§5.1）——同步语义未动。

与 DeepSeek-V3 论文 Table 2 的记号换算一并立此存照（§6 要用）：V3 表中的 $B$ 是**全反向**（= 本章 $T_B + T_W$），其 ZB1P 行气泡 $(p-1)(F + B - 2W)$ 展开为 $(p-1)(T_F + T_B + T_W - 2T_W) = (p-1)(T_F + T_B - T_W)$——正是 ZB-H1/ZB-1p 一行的公式。两表在各自记号下一致。

## 6. DeepSeek-V3 DualPipe：双向流水线与计算通信重叠

前面所有调度只跟气泡作战。DeepSeek-V3 的约束不同：跨节点 EP 的 all-to-all 把通信-计算时间比推到约 1:1（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§3.2.1 原话），于是「填气泡」之外出现了第二个目标——**把通信藏进计算**。DualPipe 的结构有两件：

1. **双向流水线（bidirectional）**：micro-batch 从流水线两端同时对灌（论文 Figure 5 的例子：8 个 PP rank、20 个 micro-batch 分两个方向，反向方向的调度与正向对称）。每个 PP rank 因此持有两段模型（两个方向各一段），参数显存 ×2。论文的注解：大 EP 配置下专家参数的大头已被 EP 切走（5.4），这份 ×2 不显著。
2. **chunk 内的计算通信重叠**：每个 chunk 拆成四段——attention、all-to-all dispatch、MLP、all-to-all combine——外加 PP 通信段；反向 chunk 的 attention 与 MLP 再按 Zero Bubble 的方式拆成 backward-for-input 与 backward-for-weights。一对方向相反的前向/反向 chunk 并排执行，手工调配通信与计算占用的 SM 比例，使 all-to-all 与 PP 通信都被对侧的计算盖住（论文 Figure 4 的色块图：前向、backward-for-input、backward-for-weights、PP 通信、barrier 五色）。

```
DualPipe 示意（出处：论文 Figure 4/5 的 ASCII 重绘）
方向 →：mb 1..10 从 rank 1 → 8      方向 ←：mb 11..20 从 rank 8 → 1（与正向对称）
rank i 持两段模型（正向段 + 反向段），参数 ×2
chunk 内：attention ─ all-to-all dispatch ─ MLP ─ all-to-all combine（+ PP 通信）
成对的正/反向 chunk 并排：一路算 attention/MLP 时，另一路在跑 all-to-all 与 P2P
```

论文 Table 2 的三方对照（原表转写； $F$ = 前向 chunk 时长、 $B$ = 全反向 chunk 时长、 $W$ = weight-grad 时长、F&B = 一对互相重叠的正/反 chunk 的合起来时长）：

| 方法 | 气泡 | 参数 | 激活 |
| --- | --- | --- | --- |
| 1F1B | $(p-1)\thinspace(F + B)$ | 1× | $p$ |
| ZB1P | $(p-1)\thinspace(F + B - 2W)$ | 1× | $p$ |
| DualPipe | $(p/2 - 1)\thinspace(\mathrm{F\&B} + B - 3W)$ | 2× | $p + 1$ |

**符号**：激活列为单 micro-batch 单 stage 激活份数的系数。读法：双向对灌把「等首条梯度回来」的级差从 $p - 1$ 砍到 $p/2 - 1$（两端各走半程）；峰值激活 $p + 1$ 份，相对 1F1B 只多 $1/p$（论文原话："only increasing the peak activation memory by $\frac{1}{PP}$ times"）。两条工程注记（均原文口径）。其一，DualPipe 只要求 stage 数与 micro-batch 数能被 2 整除——对照 Chimera 要求 micro-batch 数被 stage 数整除，约束更松。其二，气泡与激活都不随 $m$ 增长——「大 batch 稀释气泡」的说法在这里失效，因为气泡本就不靠 $m$ 摊薄。

V3 的训练配置把本章与 5.4 连成一套：PP16 × EP64（跨 8 节点）+ ZeRO-1 DP，2048 张 H800（论文 §3.2 配置段）。DualPipe 要藏的主要对象就是 EP 的 all-to-all（通信内核的细节——20/132 个 SM 专责通信、IB/NVLink 两段转发——归 5.4 §6 展开）。另有一个与 stage 均衡相关的部署决定：V3 把最浅若干层（含 embedding）与最深若干层（含 output head）放在**同一个 PP rank** 上，由此 embedding 与 output head 的参数与梯度可以和 MTP（Multi-Token Prediction，模型侧数学归 models 4.5 §6）模块物理共享，再省一份显存（论文 §3.2.3 口径）。

## 7. 激活内存总账：五种调度一张表

把 §2-6 的峰值激活放到同一张表上。记 $a$ = 单 micro-batch 单 stage 为反向保存的激活字节数。重计算策略 $\kappa$ 决定 $a$ 本身的大小（全存 / selective / full checkpoint 的逐算子账归 1.4），与调度正交叠加。ZB 的 $M_B$ 即 $a$ 在「B 所需激活」口径下的对应物， $M_W < M_B$ 是拆分后多留的小份：

| 调度 | 峰值激活（全机最大） | 参数副本 | 气泡（等时长 $T_F = T_B = T_W$ 口径） | 额外约束 |
| --- | --- | --- | --- | --- |
| GPipe | $m \thinspace a$（重计算时 $m$ 份边界输入） | 1× | $(p-1)/(m+p-1)$，式 (5) | — |
| 1F1B | $\le p \thinspace a$（stage $i$： $p - i + 1$ 份，式 (7)） | 1× | 同 GPipe | — |
| interleaved 1F1B | $\le p \thinspace a$ 量级（chunk 粒度摊派） | 1× | 再除 $v$，式 (8) | $m$ 为 $p$ 的整数倍；通信 × $v$ |
| ZB-H1 / ZB-1p | $p \thinspace M_B$ | 1× | 1F1B 的 1/3 | W 需可独立排程 |
| ZB-H2 / ZB-2p | $(2p-1)\thinspace M_B$ / $\le 2p \thinspace M_B$ | 1× | $\approx 0$ | $m$ 足够大；optimizer 同步需 post-validation |
| DualPipe | $(p + 1)\thinspace a$ | 2× | $(p/2-1)\thinspace(\mathrm{F\&B} + B - 3W)$ | $p$ 、 $m$ 为偶数 |

读表三条。其一，整条光谱是「显存 ↔ 气泡」的交换：从 GPipe 的 $m$ 份到 ZB-H2 的近 $2p$ 份，气泡从 $(p-1)/m$ 压到零；1F1B 与 ZB-H1 停在同一峰值 $p$ 上，气泡差 3 倍——同显存下调度还能挤出的空间，就是 ZB 论文那 23%。其二， $M_W$ 这一小份是 ZB 系列的隐藏成本：B/W 拆分要求把每个 in-flight micro-batch 的 $g_y$ 与层输入留到 W 执行，worker $i$ 的账 $(p-i+1)M_B + (i-1)M_W$ 里第二项就是它。其三，PP 下 $m$ 不是自由旋钮：气泡公式要 $m \gg p$，显存侧 1F1B 的峰值已封顶在 $p$ 而不限 $m$，两头夹出工作区间。还要注意 PP 不省激活的「份数」，省的是「每份的层数」：每 stage 只存 $L/p$ 层的激活。这正是 4.4 预告的 PP 激活切分公式。

## 8. Stage 均衡与 P2P 通信账

§2-5 的全部公式都长在「各 stage 等时」的假设上。不均衡的后果很直接：流水线吞吐由最慢 stage 决定（makespan 的所有链都过它），慢 stage 多出的每一秒同步放大所有 stage 的气泡。两个非 Transformer 层的负载必须进账：首段的 embedding 查表，末段的 output head 与 loss。logit 这一层 matmul 的前向 FLOPs 为 $2bThV$、反向 $4bThV$（ $V$ = 词表；Megatron 论文附录「Floating-Point Operations」的口径）， $V$ = 128256 量级时它不是零头。三家的处理各记一笔。GPipe 让分区算法直接最小化各 cell 估计成本的方差（arXiv:1811.06965 §2.2）。Zero Bubble 的实验配置让首、末 stage 各少分一层 Transformer 层，补偿 embedding 与 loss 的额外计算（arXiv:2401.10241 §5.1 原话）。DeepSeek-V3 把 embedding 与 output head 放同一 PP rank（§6）——动机主要是与 MTP 共享显存，客观上也把两个「轻计算重参数」的端点并到了一处。Megatron 集群论文则明确只考虑均匀切分，非对称结构的分区问题留给相关文献（arXiv:2104.04473 §2.2）——「把 61 层分到 16 个 stage 时每段几层」这类精确划分，V3 论文未披露，本书不推测。

通信侧把 §1 的边界账算完。每 micro-batch 每条边界：前向一次激活 + 反向一次梯度，各 $bTh$ 个元素（bf16 各 $2bTh$ bytes）。一个 step 内 $m$ 个 micro-batch、 $p - 1$ 条边界，全链路合计

$$
C_{\mathrm{PP}} = 4 \thinspace (p-1) \thinspace m \thinspace b \thinspace T \thinspace h \qquad \text{bytes} \qquad (11)
$$

**符号**：系数 4 = 2（前向 + 反向）× 2（bf16 字节数）。摊到单卡：每个中间 stage 的设备每 micro-batch 发两次（前向发下一段、反向发上一段），每 step 发送 $4mbTh$ bytes。代数字： $b = 1$、 $T = 4096$、 $h = 7168$、 $m = 32$ 时，单卡每 step 发送 $4 \times 32 \times 4096 \times 7168 \approx 3.8 \times 10^{9}$ bytes ≈ 3.8 GB。两个对照给出量级感。其一，同规模 DP 的梯度 all-reduce 随 $N$ 涨，7B 即约 26 GiB（4.1 §2.2 的账）——PP 通信小一个量级以上，且与 $N$ 无关。其二，跨节点 IB 的有效带宽量级是每 GPU 50 GB/s（V3 集群口径），这笔通信约 76 ms，还能与计算重叠（P2P 的收发可以挂在独立流上）。interleaved 把式 (11) 乘 $v$（§4）——它买的是气泡，付的就是这里。这个数量级关系是 5.5「PP 跨节点」规则的全部依据。

## 9. Megatron 实现走读：`pipeline_parallel/schedules.py`

Megatron-LM 的官方实现把 §2-4 落在 `megatron/core/pipeline_parallel/schedules.py`（NVIDIA/Megatron-LM repo，[A]；下列函数名与逻辑核实于 2026-11-17，repo main 分支）。入口 `get_forward_backward_func` 按配置三选一：`forward_backward_no_pipelining`（ $p = 1$ ）、`forward_backward_pipelining_without_interleaving`（1F1B）、`forward_backward_pipelining_with_interleaving`（interleaved 1F1B）。

1F1B 函数里的 warmup 计数与 §3 逐字对应：`num_warmup_microbatches = total_stages - current_stage - 1`（0 起始 rank；钳到 $m$），随后 `num_microbatches_remaining` 个「F/B 交替」稳态步，尾部排空剩余 B——三相对应的正是 §3 的图。反向经 `backward_step` 调 `custom_backward`：输入梯度与参数梯度在 autograd 引擎内一次算出、未拆分，ZB 式 B/W 分离不在 Megatron 主线里。P2P 收发由 `P2PCommunicator` 封装。注意一个细节：非交错档对 `overlap_p2p_comm` 直接抛错（收发必须串行于计算流），P2P 重叠是交错档的专属优化。flush 的落点在所有 micro-batch 跑完之后：由 `finalize_model_grads_func` 收尾（梯度 finalize 与同步），同步语义的 flush 边界由此实现。interleaved 档的 warmup 公式在 `get_pp_rank_microbatches`：`(p − rank − 1) * 2 + (num_model_chunks − 1) * microbatch_group_size_per_vp_stage`——代码注释把第二项写为全体 worker 统一执行的组数，第一项随 stage 深度线性（靠前 stage 预热更多、末段几乎立即进 1F1B）；MoE 场景开 `overlap_moe_expert_parallel_comm` 时再 +1（保证 1F1B 步内前向与反向相互独立，该 flag 的注释原话）。

Zero Bubble 的官方实现是论文作者基于 Megatron-LM 的 fork（sail-sg/zero-bubble-pipeline-parallelism，论文摘要给出的地址；[A]），B/W 拆分落在自定义的反向图调度上。V3 技术报告只交付了 DualPipe 的算法描述与通信内核（5.4 §6 的 DeepEP 一句），调度器实现于 2025 年 2 月单独开源（deepseek-ai/DualPipe repo）。教学对照：§3 的模拟器与 Megatron 代码的关系是「依赖结构一致、工程细节全略」——真实实现要处理变长序列、多模态 loss 缩放（`_get_moe_loss_scale` / `_get_mtp_loss_scale`）、首末 stage 的 embedding 激活缓冲清理（`clear_embedding_activation_buffer`）等分支，均不改变式 (1)-(8) 的账。

## 总结

| 调度 | 气泡（占总时间 / 对理想时间） | 峰值激活 | 代价与约束 | 出处 |
| --- | --- | --- | --- | --- |
| GPipe | 式 (5)： $(p-1)/(m+p-1)$ | $m \thinspace a$ | 显存随 $m$ 涨；flush 同步语义 | [Huang et al., 2019]，[B] |
| 1F1B | 同 GPipe（式 (4) 不变） | $\le p \thinspace a$（式 (7)） | warmup $p - i$ 个 F | [Narayanan et al., 2021]，[A] |
| interleaved 1F1B | 再除 $v$（式 (8)） | $\le p \thinspace a$ 量级 | 通信 × $v$； $m$ 为 $p$ 整数倍 | 同上，§2.2.2 |
| ZB-H1 / ZB-1p | $(p-1)(T_F{+}T_B{-}T_W)$，等时长下 = 1F1B 的 1/3 | $p \thinspace M_B$（不涨） | B/W 拆分； $M_W$ 暂存 | [Qi et al., 2024]，[A] |
| ZB-H2 / ZB-2p | 等时长下 0；实测 < 1% | $(2p-1)\thinspace M_B$ / 上限 $2p \thinspace M_B$ | post-validation 旁路 optimizer 同步； $m$ 需够大 | 同上 |
| ZB-V | 0（等时长） | $p \thinspace M_B$ | V 形双 chunk；等时长假设立场 | 同上，§6 |
| DualPipe | $(p/2 - 1)(\mathrm{F\&B} + B - 3W)$ | $(p+1)\thinspace a$ | 参数 ×2； $p$ 、 $m$ 偶数；计算通信重叠 | [DeepSeek-AI, 2024]，[A] |
| 边界通信 | 式 (11)：每 step 全链 $4(p-1)mbTh$ bytes | — | 与 $N$ 无关 ⇒ PP 跨节点 | §8 |

## 练习

1.（推导）从式 (3) 出发完成三步：(a) 补全 stage $p$ 前向结束时刻 $(p-1)t_f + m t_f$ 的归纳（对 $j$ 归纳）；(b) 证明式 (5) 与式 (6) 的换算关系「式 (5) = 式 (6) / (1 + 式 (6))」；(c) 取 $p = 16$、 $m \in \lbrace 16, 32, 64, 128 \rbrace$ 算两口径的气泡占比各一排，并说明 GPipe「 $m \ge 4p$ 可忽略」在这两排数字里各自读到多少。

2.（推导）1F1B 的峰值激活上界：把 §3 的论证形式化——设 stage $i$ 的 warmup 数为 $w_i$，稳态严格 F/B 交替，证明 (a) 任意时刻 in-flight $\le w_i + 1$；(b) 首条反向到达 stage 1 前的等待恰为 $(p-1)\thinspace t_b$（正文给出的到达时刻减去本段已排的前向时长）；(c) 说明为什么 $w_i$ 开大可以填掉这段等待但峰值同步上涨（warmup $= m$ 即 GPipe），以及 Megatron 代码 `total_stages - current_stage - 1` 对应正文哪个量。

3.（代码）跑通 §3 的模拟器，完成三个实验：(a) 扫 $(t_f, t_b) \in \lbrace 1 \rbrace \times \lbrace 1, 2, 4 \rbrace$ 与 $(p, m) \in \lbrace 4, 8 \rbrace \times \lbrace 8, 16, 64 \rbrace$，验证式 (3) 与「GPipe 峰值 = $m$、1F1B 峰值 = $\min(p, m)$」全部成立；(b) 把 1F1B 的 warmup 强行减 1（ $w_i = p - i - 1$），观察 $T$ 是否变差、峰值是否变小，解释结果；(c) 在 GPipe 队列里把反向按 micro-batch 逆序排（B(m) 先做）， $T$ 变不变？为什么。

4.（推导 + 计算）ZB 的显存-气泡交换：(a) 由式 (9) 的依赖论证说明「 $W(i, j)$ 只需排在 $B(i, j)$ 之后」为什么是 B/W 可拆的充分条件（对照 $g_x$ 的链式位置）；(b) 在 $T_F = T_B = T_W$ 假设下，用总结表的三行公式验证 ZB-H1 气泡恰为 1F1B 的 1/3、ZB-H2 为 0；(c) 取 $p = 16$、 $M_W = 0.5 \thinspace M_B$，按 ZB 论文的 worker 公式算 ZB-H1 与 ZB-H2 各 worker 的峰值激活（以 $M_B$ 为单位），验证全机最大值分别为 $16 \thinspace M_B$ 与 $31 \thinspace M_B$ 并指出各自落在哪个 worker。

## 参考

- [Huang et al., 2019](../references.md)：GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism，arXiv:1811.06965，[B]（key: gpipe）——全前向再全反向调度与同步语义（§2.2）、气泡 $O((K-1)/(M+K-1))$ 与 $M \ge 4K$ 经验规则（§2.3）、re-materialization 显存口径
- [Narayanan et al., 2021](../references.md)：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM，arXiv:2104.04473，[A]（key: megatron-cluster）——气泡两口径（§2.2.1）、1F1B 与峰值激活 $\le p$（§2.2.1）、interleaved 调度与式 (8)（§2.2.2）、scatter/gather 优化（机制 §4.1 / Figure 9，11% 数字 §5.7）、1T @ 3072 A100 @ 502 PFLOP/s（摘要/§5）
- [Qi et al., 2024](../references.md)：Zero Bubble Pipeline Parallelism，arXiv:2401.10241，[A]（key: zero-bubble）——B/W 拆分（§1-2）、ZB-H1/ZB-H2 与气泡-显存公式（Table 2）、post-validation（§4）、自动调度 ZB-1p/ZB-2p（§3）与 ZB-V（§6）、23%/31% 提速与 bit-identical 验证（§5）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——DualPipe 机制与论文 Table 2 三行对照（§3.2.1）、通信-计算约 1:1 的动机、PP16 × EP64 × ZeRO-1 配置（§3.2）、embedding/head 同 rank 部署（§3.2.3）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——NVIDIA/Megatron-LM repo `megatron/core/pipeline_parallel/schedules.py`：`get_forward_backward_func` / `forward_backward_pipelining_without_interleaving` / `forward_backward_pipelining_with_interleaving` / `get_pp_rank_microbatches`（warmup 公式）/ `P2PCommunicator`；核实于 2026-11-17
