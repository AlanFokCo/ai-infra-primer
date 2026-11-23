# 4.4 投机解码原理与账（Speculative Decoding: Principles and Accounting）

> 8B 模型在 H100 上每生成一个 token 要约 5 ms，其中真正的浮点计算只占 0.017 ms——其余时间都在搬权重。能不能让这 5 ms 产出多个 token，且输出分布一个字不改？本章从 draft-verify 范式推出接受概率 $\min(1, p/q)$ 的分布保持证明、每轮期望产出 $(1-\alpha^{\gamma+1})/(1-\alpha)$ 与无量纲加速比定理，再给 tree verification 的正确性命题，最后算 batch 下固定长度 verify 的浪费账 | 前置：models 2.4（KV cache 与前缀不可变命题）、inference 1.2（decode FLOPs 式 (2)、强度式 (4)、ridge）、1.4（采样与 logits processor）、2.1（块管理与 COW）、4.1（continuous batching） | 锚点：Leviathan 投机解码 [Leviathan et al., 2022] [B]（arXiv:2211.17192，Google 无官方开源）、Chen 投机采样 [Chen et al., 2023] [B]（arXiv:2302.01318）、SpecInfer [Miao et al., 2023] [A]（arXiv:2305.09781，官方代码在 flexflow/flexflow-train 的 inference 分支）

## 1. draft-verify 范式：用闲置算力换串行步数

decode 为什么慢？不是因为算得多，而是因为算得少、搬得多。recap 1.2 的账：单序列第 $T$ 步 decode 的 FLOPs 为 $f(T) = 2N_{\mathrm{act}} + 4LTd_{\mathrm{model}}$（1.2 式 (2)），而每步必须把全量权重 $W$ 字节与该序列的全量 KV cache $\kappa L T$ 字节从 HBM 读一遍。代入章脊口径（8B dense bf16， $W = 1.61 \times 10^{10}$ 字节， $\kappa L$ = 128 KiB/token，H100 SXM 带宽 3.35 TB/s、BF16 峰值 989.5 TFLOPS，[NVIDIA datasheet](../references.md)，[C] 官方规格披露）： $T$ = 4096、 $B$ = 1 时步时带宽下界 $(W + \kappa L T)/\mathrm{BW} \approx 4.97$ ms，而这一步的浮点计算 $f(4096) \approx 1.71 \times 10^{10}$ FLOPs 折合峰值时间仅 0.017 ms——计算只占步时下界的 0.35%，GPU 的算力几乎整步闲置（1.2 §3 同账）。

**本章数值口径**（全章例题复用）：target 模型 $M_p$ = 8B dense bf16（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128，几何同 1.2 / 2.1），硬件 H100 SXM；draft 模型 $M_q$ = 0.8B dense bf16（教学口径，与 target 同架构同 tokenizer）；上下文 $T$ = 4096、单请求 $B$ = 1 为延迟账基准；接受率 $\alpha$ = 0.8、draft 长度 $\gamma$ = 4 为账脊。§6 换短上下文大 batch 口径（ $T$ = 256、 $B$ = 128），原因在该节声明：长上下文下 KV 读主导分母，verify 的算力开销被带宽下界掩盖，须换口径才能暴露 compute-bound 的浪费。

投机解码（speculative decoding）的出发点正是这笔闲置算力。所谓 draft-verify 范式，是指：先用一个便宜的 draft 模型 $M_q$ 串行猜出 $\gamma$ 个 token，再用 target 模型 $M_p$ 一次前向并行验证这 $\gamma$ 个猜测（外加一个 bonus 位），接受其中正确的前缀、在第一个错误处重采样。一轮的串行步数从「 $\gamma+1$ 次 $M_p$ 」压成「 $\gamma$ 次 $M_q$ + 1 次 $M_p$ 」。关键的结构事实是：verify 一次前向打 $\gamma+1$ 个 token，是 prefill 形态（query 长度 $\gamma+1 > 1$），其算力开销随 token 数线性涨、而权重仍只读一遍——在 memory-bound 区，多打几个 token 几乎不增加步时。代入章脊：verify 5 个 token 的计算 $5 f(4096) \approx 8.55 \times 10^{10}$ FLOPs 折合 0.087 ms，仍只占 4.97 ms 步时下界的 1.7%。所以「一次 verify 5 个 token」与「一次 decode 1 个 token」的墙钟几乎相同，产出却可能多 4 个。这就是全部加速的来源：把闲置算力换成被接受的 draft token。

```
一轮 draft-verify（γ=4）：
  draft（M_q 串行，每步≈c·T）：
    x1 ~ q(·|prefix) → x2 ~ q(·|prefix,x1) → x3 → x4
  verify（M_p 一次前向，prefill 形态，≈T）：
    输入 [prefix, x1, x2, x3, x4]
    输出 p1(·|prefix), p2(·|prefix,x1), ..., p5(·|prefix,x1..x4)
  accept：
    逐位比 p_i 与 q_i → 接受前 n 个 → 第 n+1 位从 p（或残差）采样
    产出 n 个 draft token + 1 个 target token（n≤γ）
```

两个模型的角色不对称。 $M_p$ 是权威：输出分布最终必须严格等于 $M_p$ 的分布（§2 证明）， $M_q$ 只是提议者，猜错不污染输出、只浪费算力。所谓接受率（acceptance rate），是指一个 draft token 被 verify 接受的概率，记 $\alpha$（精确定义见 §2 命题 2）； $\gamma$ 专归 draft 长度（spec §4.3）。draft 模型可以很小——Leviathan 的实验口径是 $M_q$ 比 $M_p$ 小约两个数量级时综合最优（[Leviathan et al., 2022](../references.md)，§3.6），因为要在「 $\alpha$ 高（猜得准）」与「 $c$ 小（猜得快）」之间权衡， $c$ 是 draft 与 verify 的单步成本比（§4 定义）。Chen 的投机采样（[Chen et al., 2023](../references.md)，arXiv:2302.01318，[B]）与 Leviathan 独立同期，给出同一套 modified rejection sampling 公式与分布保持结论（基准为 Chinchilla 70B）；本章公式以 Leviathan 的编号体系为主线，Chen 作并列源头，不展开历史。

> **提点**：draft-verify 与「beam search 多候选」是两件事。beam search 保留多条候选序列、最终选一条，改变了输出分布（它是搜索算法）；投机解码的 verify 是拒绝采样，输出分布严格等于 target 单步采样（§2 命题 1），多候选只是提高接受率的提议手段，不是搜索。

## 2. 接受规则与分布保持证明

verify 凭什么接受一个 draft token？设某位置上 target 分布为 $p(x)$、draft 分布为 $q(x)$（都是经过 1.4 的 logits processor 链与温度 / top-p 截断后的最终采样分布，recap 一句：采样分布的模型侧定义与截断再归一化归 1.4）。draft 先采出 $x \sim q$，verify 以概率 $\min(1, p(x)/q(x))$ 接受它。直觉：若 target 给 $x$ 的概率不低于 draft（ $p(x) \ge q(x)$ ），则无条件接受；若 target 给得更少，则按比例 $p(x)/q(x)$ 打折接受——draft 过度提议的部分被拒掉。

被拒之后怎么办？朴素拒绝采样会「重试」：再从 $q$ 采、再比，直到接受。但这浪费 target 前向（每轮只产一个 token），且重试次数无界。投机解码改用残差修正：拒绝后不重试，而是直接从残差分布（residual distribution）采一个 token：

$$
\tilde{p}(x) = \frac{\max(0, p(x) - q(x))}{Z}, \qquad Z = \sum_{x} \max(0, p(x) - q(x)) \qquad (1)
$$

**符号**： $p(x)$ = target 在该位置的采样分布； $q(x)$ = draft 分布； $\tilde{p}(x)$ = 残差分布（拒绝后的重采样分布）； $Z$ = 归一化常数； $\max(0, \cdot)$ = 逐元素取正部。式 (1) 把 target 分布中「draft 已经覆盖够」的部分削掉，只留下 draft 提议不足的部分再归一。代入一个 3-token 词表的最小实例（教学口径）： $p = (0.5, 0.3, 0.2)$、 $q = (0.4, 0.4, 0.2)$，逐元素 $\min(p,q) = (0.4, 0.3, 0.2)$、正部 $\max(0, p-q) = (0.1, 0, 0)$， $Z = 0.1$，故 $\tilde{p} = (1, 0, 0)$——拒绝后必采第一个 token。

这套「接受 + 残差修正」为什么恰好还原 $p$？下面是本章第一个核心命题。

**命题 1**（分布保持）。对任意两个分布 $p, q$（同一词表、 $q$ 的支撑覆盖 $p$ 的支撑），单步 draft-verify 的输出分布严格等于 $p$：draft 采 $x \sim q$，以 $\min(1, p(x)/q(x))$ 接受；拒绝则从式 (1) 的 $\tilde{p}$ 采。无论 $q$ 选得多差，输出分布都是 $p$。

**证明**（路线：把输出分布拆成「经 draft 接受」与「经残差重采样」两条互斥路径，分别算概率再相加；关键一步是恒等式 $\min(a,b) + \max(0, a-b) = a$）。先算归一化常数 $Z$。因为 $\max(0, p-q) = p - \min(p,q)$，所以

$$
Z = \sum_{x} (p(x) - \min(p(x), q(x))) = 1 - \sum_{x} \min(p(x), q(x)) \qquad (2)
$$

记 $A = \sum_{x} \min(p(x), q(x))$（这正是接受概率，命题 2 会证明），则 $Z = 1 - A$，恰为拒绝概率。再算输出落在 token $x$ 的总概率，分两条路径：

$$
\begin{aligned}
P(\mathrm{out} = x) &= \underbrace{q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)}_{\text{accept}} + \underbrace{(1 - A) \cdot \tilde{p}(x)}_{\text{residual}} \\
&= \min(q(x), p(x)) + (1-A) \cdot \frac{\max(0, p(x)-q(x))}{1-A} \\
&= \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)
\end{aligned}
$$

第一行第一项：draft 采出 $x$ 的概率是 $q(x)$，接受概率是 $\min(1, p(x)/q(x))$，乘积 $q(x) \min(1, p(x)/q(x)) = \min(q(x), p(x))$。第二项：拒绝概率 $1-A$ 乘残差分布 $\tilde{p}(x)$，归一化常数 $Z = 1-A$ 恰好对消。最后一步用恒等式 $\min(a,b) + \max(0,a-b) = a$（分 $a \ge b$ 与 $a < b$ 两种情形直接验证）。故 $P(\mathrm{out}=x) = p(x)$ 对每个 $x$ 成立。证毕。

边界与口径。其一，命题对任意 $q$ 成立—— $q$ 越接近 $p$，接受率 $A$ 越高、残差路径越少用，但输出分布始终是 $p$； $q$ 与 $p$ 支撑不相交时 $A=0$，每个 token 都走残差路径，输出仍是 $p$（此时投机解码退化为「每轮只产一个 token」，无加速但无损）。其二，「严格等于」是数学口径；实际硬件上 $p/q$ 的浮点除法与归一化有舍入误差，Chen 的表述是「在 hardware numerics 内」保持目标分布（[Chen et al., 2023](../references.md)）。其三， $p$ 与 $q$ 必须是同一套采样配置下的最终分布——若 draft 用 temperature 0.7、target 用 1.0，或一方过了 top-p 另一方没过，命题仍成立（它只要求两个合法分布），但接受率会因分布错位而下降；工程上 draft 与 target 通常共用采样配置以最大化 $\alpha$。

> **坑**：把接受概率用在 logits 而非概率上会破坏分布保持。触发条件：实现里直接比 $p_{\mathrm{logit}}/q_{\mathrm{logit}}$ 或对未归一化 logits 套 $\min(1, \cdot)$；现象：输出分布偏离 target，且偏差随温度变化；规避：接受规则必须作用在 softmax（及 logits processor）之后的归一化概率上（1.4 的采样链），残差修正同样在概率空间做。

接受率本身有一个干净的闭式，它把「猜得准不准」量化成两个分布的距离。

**命题 2**（接受率 = 1 − 总变差距离）。单位置接受概率 $\beta = \sum_{x} \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q)$，其中 $\mathrm{TV}(p,q) = \frac{1}{2}\sum_{x} \lvert p(x) - q(x) \rvert$ 是总变差距离（total variation distance）。

**证明**（路线：先按接受规则对 $x \sim q$ 求期望得 $\beta = \sum \min(p,q)$，再用 $\min$ 与绝对值的恒等式把它写成 $1 - \mathrm{TV}$）。接受概率是对 draft 采样求期望：

$$
\beta = \sum_{x} q(x) \min\left(1, \frac{p(x)}{q(x)}\right) = \sum_{x} \min(p(x), q(x)) \qquad (3)
$$

再用恒等式 $\min(a,b) = \frac{a+b-\lvert a-b \rvert}{2}$：

$$
\sum_{x} \min(p,q) = \sum_{x} \frac{p(x)+q(x)-\lvert p(x)-q(x) \rvert}{2} = \frac{1+1}{2} - \frac{1}{2}\sum_{x}\lvert p(x)-q(x) \rvert = 1 - \mathrm{TV}(p,q)
$$

证毕。这正是 Leviathan 的 Def 3.2 / Lemma 3.3 / Thm 3.5（[Leviathan et al., 2022](../references.md)，§3.2，原文记 $D_{LK}(p,q) = \sum_x \lvert p(x)-q(x) \rvert / 2$，即总变差）。代入上方 3-token 实例： $\mathrm{TV} = \frac{1}{2}(0.1+0.1+0) = 0.1$，故 $\beta = 0.9$——与直接算 $\sum \min = 0.4+0.3+0.2 = 0.9$ 一致。两个极限钉住直觉： $p=q$ 时 $\mathrm{TV}=0$、 $\beta=1$（全接受）；支撑不相交时 $\mathrm{TV}=1$、 $\beta=0$（全拒绝）。所以提升接受率等价于缩小 draft 与 target 的总变差距离——这是 §7 draft 分类学里「feature 级 draft 比 token 级更准」的账面依据（4.5 EAGLE）。

i.i.d. 假设与 $\alpha$。单位置的 $\beta$ 依赖前缀（不同位置、不同请求的 $p, q$ 不同）。Leviathan 的简化假设是各位置的 $\beta$ 独立同分布，记 $\alpha = E[\beta]$（[Leviathan et al., 2022](../references.md)，§3.1）。本章 §3–§4 的闭式都在此假设下成立； $\alpha$ 是常量参数，不是随机变量，勿写 $E[\alpha]$（spec §4.3）。真实 serving 里 $\beta$ 跨请求 / 跨位置有方差，这个方差正是 §6 batch 浪费账与 4.5 动态 draft 的动机。

## 3. 每轮期望产出：capped geometric

一轮 draft-verify 平均产出几个 token？答案是一个截断几何级数（capped geometric）。路线：先写出产出 token 数 $N_{\mathrm{tok}}$ 的分布（逐位接受是独立 Bernoulli），再用尾和公式 $E[N] = \sum_{k \ge 1} P(N \ge k)$ 求期望——关键一步是「产出至少 $k$ 个 token」等价于「前 $k-1$ 个 draft 全被接受」。

设各位置接受概率均为 $\alpha$（i.i.d. 假设）。一轮产出 $N_{\mathrm{tok}}$ 个 token，含被接受的 draft token 与最后一个 target token（拒绝位的重采样，或全接受时第 $\gamma+1$ 位的 bonus）。事件 $\lbrace N_{\mathrm{tok}} \ge k \rbrace$（ $1 \le k \le \gamma+1$）等价于前 $k-1$ 个 draft token 全部被接受，概率 $\alpha^{k-1}$。于是 $N_{\mathrm{tok}}$ 的分布为

$$
P(N_{\mathrm{tok}} = k) = \alpha^{k-1}(1-\alpha) \quad (1 \le k \le \gamma), \qquad P(N_{\mathrm{tok}} = \gamma+1) = \alpha^{\gamma} \qquad (4)
$$

**符号**： $N_{\mathrm{tok}}$ = 每轮产出 token 数（含 bonus / 重采样位）； $\alpha$ = 接受率； $\gamma$ = draft 长度。前 $\gamma$ 项是「第 $k$ 位首次拒绝」（前 $k-1$ 接受、第 $k$ 拒绝，概率 $\alpha^{k-1}(1-\alpha)$），末项是「 $\gamma$ 个 draft 全接受、再产一个 bonus」（概率 $\alpha^{\gamma}$）。这就是成功概率 $1-\alpha$、上限 $\gamma+1$ 的截断几何变量（Leviathan §3.1）。用尾和公式：

$$
E[N_{\mathrm{tok}}] = \sum_{k=1}^{\gamma+1} P(N_{\mathrm{tok}} \ge k) = \sum_{k=1}^{\gamma+1} \alpha^{k-1} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} \qquad (5)
$$

最后一步是首项 1、公比 $\alpha$、共 $\gamma+1$ 项的等比数列求和。代入章脊 $\alpha$ = 0.8、 $\gamma$ = 4： $E[N_{\mathrm{tok}}] = (1 - 0.8^{5})/0.2 = (1-0.32768)/0.2 = 3.3616 \approx 3.36$ token/轮。两个极限： $\alpha \to 1$ 时 $E[N_{\mathrm{tok}}] \to \gamma+1 = 5$（全接受，产出上限）； $\alpha \to 0$ 时 $E[N_{\mathrm{tok}}] \to 1$（全拒绝，退化为普通 decode）。

**口径注**（spec §5 硬口径，与 DSpark 对齐）：式 (5) 的「期望产出长度」 $E[N_{\mathrm{tok}}]$ 含拒绝位重采样 / 全接受 bonus 的第 $\gamma+1$ 个 token，**不是**「被接受的 draft token 数」。后者记 $K = N_{\mathrm{tok}} - 1$，期望为

$$
E[K] = E[N_{\mathrm{tok}}] - 1 = \frac{\alpha - \alpha^{\gamma+1}}{1 - \alpha} \qquad (6)
$$

代入章脊： $E[K] = 3.3616 - 1 = 2.3616 \approx 2.36$ 个 draft token 被接受。两个口径差恒为 1（那个 target token）。DeepInfer DSpark 增益表里的「accepted length」（3.29 vs 3.20，§6）是 $E[K]$ 口径（被接受的 draft 数），「verified length」（5.00 → 3.2–4.4）是实际送去 verify 的 token 数；读生产数字时先对齐口径，勿把 accepted 当 $E[N_{\mathrm{tok}}]$（7.2 正式走读）。

> **坑**：把 $E[N_{\mathrm{tok}}]$ 写成「期望接受长度」再与 DSpark 的 accepted 直接对比会差一个 bonus token。触发条件：跨章 / 跨系统对比接受长度；现象：3.36 对 3.29 看似接近实则口径不同（前者含 bonus、后者不含）；规避：声明口径——本书 $E[N_{\mathrm{tok}}]$ 含 bonus（式 (5)）， $E[K]$ 不含（式 (6)），DSpark accepted = $E[K]$。

## 4. 加速比账：无量纲成本比与定理

产出多了，墙钟省了多少？这要把 draft 的成本算进去。先定义成本比——Leviathan Def 3.7 逐字（[Leviathan et al., 2022](../references.md)，§3.3）：「Let $c$, the cost coefficient, be the ratio between the time for a single run of $M_q$ and the time for a single run of $M_p$.」即

$$
c = \frac{c_d}{c_v} \qquad (7)
$$

**符号**： $c_d$ = draft 模型 $M_q$ 单步绝对成本（时间）； $c_v$ = target 模型 $M_p$ 单步绝对成本（时间）； $c$ = 成本比，**无量纲**。 $c$ 不是参数量比，是时间比——它依赖硬件与软件实现（Leviathan 原文： unlike $\alpha$ which is an intrinsic property of the models and the task, $c$ depends on the hardware configuration and software implementation details）。在 memory-bound 区，单步时间约等于权重字节 ÷ 带宽，故 $c \approx N_q / N_p$（参数量比）是一个可用的估算：章脊 draft 0.8B / target 8B 给出 $c \approx 0.1$。Leviathan 实验中 $M_q$ 小两个数量级， $c$ 总小于 0.05（§3.3）。

**命题 3**（加速比定理，Leviathan Thm 3.8 + Cor 3.9）。在「 $\gamma+1$ 个 token 的并行 verify 不增加墙钟」的前提下（足够算力，§1 已数值验证），一轮 draft-verify 的期望墙钟改善因子为

$$
F = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)} \qquad (8)
$$

且**存在** draft 长度 $\gamma$ 使 $F > 1$ 当且仅当 $\alpha > c$（存在量词，Leviathan Cor 3.9 原文「there exists $\gamma$」）； $\gamma = 1$ 时 $F = (1+\alpha)/(1+c)$。代入章脊 $\alpha$ = 0.8、 $c$ = 0.1、 $\gamma$ = 4： $F = 3.3616/(4 \times 0.1 + 1) = 2.40$，每 token 墙钟降到基线的 $1/2.40 \approx 0.42$（42%）。

**证明**（路线：设 target 单步成本 $T = c_v$，算每轮总成本与期望产出，相除得每 token 期望成本，再与基线 $T$ 相比； iff 部分把 $F>1$ 拆成逐项和）。每轮成本 = $\gamma$ 次 draft + 1 次 verify = $T c \gamma + T = T(\gamma c + 1)$；期望产出由式 (5) 为 $(1-\alpha^{\gamma+1})/(1-\alpha)$。故每 token 期望成本为 $T(\gamma c+1)(1-\alpha)/(1-\alpha^{\gamma+1})$，基线（普通 decode）每 token 成本为 $T$，两者相除即式 (8)。这是 Leviathan Thm 3.8 的原证（设 $M_p$ 单步成本 $T$，每轮 $Tc\gamma + T$，除以 Eq 1）。

iff 部分： $F > 1$ 等价于 $1 - \alpha^{\gamma+1} > (1-\alpha)(\gamma c + 1)$。左边除以 $1-\alpha$ 是等比和 $\sum_{k=0}^{\gamma} \alpha^{k}$，右边展开 $(1-\alpha)(\gamma c+1)/(1-\alpha) = \gamma c + 1$，移项得

$$
F > 1 \iff \sum_{k=0}^{\gamma} \alpha^{k} - 1 - \gamma c > 0 \iff \sum_{k=1}^{\gamma} (\alpha^{k} - c) > 0 \qquad (9)
$$

若 $\alpha > c$，则 $k=1$ 项 $\alpha - c > 0$，取 $\gamma = 1$ 即满足，故存在 $\gamma$ 使 $F>1$；若 $\alpha \le c$，则每一项 $\alpha^{k} - c \le \alpha - c \le 0$（因 $\alpha \le 1$、 $k \ge 1$ 时 $\alpha^{k} \le \alpha$），和对任意 $\gamma$ 非正， $F \le 1$。故存在 $\gamma$ 使 $F>1$ 当且仅当 $\alpha > c$。 $\gamma=1$ 代入式 (8)： $F = (1-\alpha^{2})/((1-\alpha)(c+1)) = (1+\alpha)/(1+c)$，即 Cor 3.9 的下界。证毕。

边界辨析（量词不可省）： $\alpha > c$ 是**存在**某个 $\gamma$ 加速的条件，不是对每个固定 $\gamma$ 都加速。固定 $\gamma$ 的加速判据是式 (9) 的 $\sum_{k=1}^{\gamma}(\alpha^{k}-c) > 0$，未必等价于 $\alpha > c$。反例： $\alpha$ = 0.8、 $c$ = 0.7（满足 $\alpha > c$）、 $\gamma$ = 4 时 $F = (1-0.8^{5})/(0.2 \times (4 \times 0.7 + 1)) = 0.67232/0.76 = 0.885 < 1$——draft 太贵（ $c$ = 0.7）把 $\gamma$ = 4 的成本拉过产出；但 $\gamma$ = 1 仍给 $F = 1.8/1.7 = 1.06 > 1$，存在性成立。所以「 $\alpha > c$ 」读作「至少 $\gamma$ = 1 加速、故存在加速的 $\gamma$ 」，最优 $\gamma$ 另由 §4 数值搜索定。

式 (8) 是无量纲形式（ $c$ 无量纲， $F$ 是纯倍数）。若用绝对成本 $c_d, c_v$ 写，必须乘 $c_v$ 归一：

$$
F = \frac{(1 - \alpha^{\gamma+1}) \thinspace c_v}{(1-\alpha)(\gamma c_d + c_v)} \qquad (10)
$$

**符号**： $c_d, c_v$ 见式 (7)。式 (10) 与式 (8) 恒等（分子分母同除 $c_v$）。

> **坑**：绝对成本式漏乘 $c_v$ 会把量纲搞错（spec 四路审查 A-S2-1）。触发条件：写成 $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c_d + c_v))$ 而不乘 $c_v$；现象：该式量纲是 1/时间 = tokens/s，是吞吐而非改善因子，与基线相比时单位对不上；规避：加速比一律用无量纲 $c = c_d/c_v$ 的式 (8)，或显式乘 $c_v$ 的式 (10)。

$\gamma$ = 4 的 $F$ = 2.40 已算（命题 3 后）。 $\gamma$ 不是越大越好——式 (8) 分子随 $\gamma$ 趋于饱和（ $\alpha^{\gamma+1} \to 0$），分母却线性涨（ $\gamma c + 1$），故存在最优 $\gamma$。Leviathan §3.5：最优 $\gamma$ 是使式 (8) 最大的整数，数值搜索即得（原文 Figure 3）。章脊口径下逐 $\gamma$ 算：

| $\gamma$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $F$（ $\alpha$ =0.8, $c$ =0.1） | 1.64 | 2.03 | 2.27 | 2.40 | 2.46 | 2.47 | 2.45 | 2.40 |

最优 $\gamma$ = 6， $F \approx 2.47$； $\gamma$ = 4 的 2.40 已接近最优。成本比 $c$ 越大，最优 $\gamma$ 越小、峰值越低： $c$ = 0.3（draft 2.4B）时最优 $\gamma$ = 3、 $F \approx 1.55$；接受率 $\alpha$ = 0.5、 $c$ = 0.1 时最优 $\gamma$ = 2、 $F \approx 1.46$。这解释了 §1 的「小两个数量级」经验： $c$ 小才能容忍大 $\gamma$、吃满几何级数的产出。Leviathan 在 T5-XXL（11B）上实测 out-of-the-box 2–3× 延迟改善、输出不变（[Leviathan et al., 2022](../references.md)，摘要 / §4，batch 1、TPU-v4）——与式 (8) 的量级一致。

前提的边界。式 (8) 假设 verify $\gamma+1$ 个 token 不增墙钟，这只在算力有富余时成立（§1：单请求下 verify 计算占步时 1.7%）。大 batch / 短上下文下 verify 计算会超过带宽下界，前提失效， $F$ 须按真实 verify 成本重算——这是 §6 的账。Leviathan §3.4 另给一个算力侧定理（Thm 3.11）：总运算量相对基线的增加因子为 $(1-\alpha)(\gamma \hat{c} + \gamma + 1)/(1-\alpha^{\gamma+1})$（ $\hat{c}$ 是每 token 运算量比，Def 3.10）； $\alpha$ 越低浪费越大。而权重与 KV 的访存量反而下降——每轮只读一遍，按式 (5) 的因子缩减。所以投机解码是「用算力换访存」：memory-bound 区净赚，compute-bound 区可能净亏。

> **提点**： $c \approx 0$ 的极限值得单独记。若 draft 几乎免费（n-gram 查表、检索），式 (8) 退化为 $F = (1-\alpha^{\gamma+1})/(1-\alpha)$，上界 $1/(1-\alpha)$（ $\gamma$ 大时趋近）。Leviathan §3.6 的 n-gram 实验：T5-XXL 11B + 平凡 bigram，EnDe 翻译任务 $\alpha \approx 0.2$、 $\gamma$ = 3，实测改善因子 1.25×——代入 $(1-0.2^{4})/0.8 = 0.9984/0.8 = 1.248$，与实测吻合。零成本 draft 即使接受率低也有正收益，这是 §7 n-gram / REST 路线的账面依据。

## 5. tree verification：一次前向验证多候选

单链 draft 只猜一条序列，第一个错误之后的全部作废。能不能一次猜多条、共享前缀计算？SpecInfer（[Miao et al., 2023](../references.md)，arXiv:2305.09781，[A]，ASPLOS'24）的答案是 token tree + tree attention。所谓 token tree，是指把多个候选 token 序列组织成的树：每个节点代表一个候选 token，从根到该节点的路径就是它代表的候选序列（原文：nodes each represent a candidate token sequence）。构造有两种（§3）：expansion-based（利用单个 draft 模型内部的多样性，对同一前缀展开多个 top-k 候选）与 merge-based（合并多个 draft 模型的候选）。

verify 整棵树只需一次 $M_p$ 前向，关键是 tree attention mask：让每个节点只 attend 它在树上的祖先（加自身），而非序列里所有更早的 token。SpecInfer Def 4.1 逐字：「TreeAttention(u) = Attention(S_u)」，其中 $S_u$ 是节点 $u$ 代表的序列（根到 $u$ 的路径）。实现上用 topology-aware causal mask 把整棵树的 attention 融进单个 kernel（§4.2），原文论断：「computes the exact same attention output as incremental decoding」。下面把这个论断写成命题并证明。

**命题 4**（tree mask 等价性）。设 token tree 的 attention mask 满足「节点 $j$ 可见节点 $k$ 当且仅当 $k$ 是 $j$ 的祖先或 $k=j$」，节点的位置编码取其在路径上的深度（与独立序列一致，RoPE 的相对位置由此确定），且共享前缀（根路径）的 KV 已正确缓存。则对任意节点 $u$，tree attention 在 $u$ 处的输出等于把路径序列 $S_u$ 单独做标准 causal attention 的输出。

**证明**（路线：对层数归纳；关键一步是「节点的可见集恰为其路径前缀」，于是该节点的 Q/K/V 与单独跑 $S_u$ 时逐层相同）。记 $S_u = (t_0, t_1, \ldots, t_m)$ 为根到 $u$ 的路径。基例（第 1 层）：节点 $u$ 的 attention 可见集 = 祖先 + 自身 = $\lbrace t_0, \ldots, t_m \rbrace$，恰为 $S_u$ 的全部位置；causal mask 下 $S_u$ 单独前向时 $u$ 的可见集也是 $\lbrace t_0, \ldots, t_m \rbrace$。两者可见集相同、输入 embedding 相同、位置编码同取路径深度（RoPE 的 Q/K 相对位置一致），故第 1 层输出相同。归纳步：设第 $\ell$ 层各节点输出与单独跑各自路径时相同，则第 $\ell+1$ 层的 K/V 由第 $\ell$ 层输出线性投影而来，逐节点相同； $u$ 在第 $\ell+1$ 层的可见集仍是祖先 + 自身，attention 加权求和的项与单独跑 $S_u$ 时一一对应且相等，故输出相同。FFN 是逐位置算子，不跨节点混合，保持等价。终止：逐层成立，末层输出（经 LM head）即该节点的路径条件分布 $p(\cdot \mid S_u)$。证毕。

命题 4 是 tree verification 正确性的全部依据：mask 把「树」还原成「每条路径各自独立验证」，于是 §2 的接受规则可以逐路径套用。kernel 侧，tree mask 作为 custom mask 传给 attention 算子——FlashInfer 的 `flashinfer/prefill.py` `maybe_packed_custom_mask`（packed custom mask，[Ye et al., 2025](../references.md)，arXiv:2501.01005，[A]，2026-11-23 repo 核实）即此接口；split-KV / cascade 等 kernel 算法归 3.1，本章只钉「mask 语义 → 路径等价」这一层。

```
token tree（expansion，top-2 展开两层）与可见集：
        root(已验证前缀)
        /        \
      a1          b1        ← 第 1 层两个候选
     /  \        /  \
   a2   a3     b2   b3      ← 第 2 层各展开 2 个
  路径：root-a1-a2 / root-a1-a3 / root-b1-b2 / root-b1-b3
  tree mask 可见集：
    a2 → {root, a1, a2}      （不含 b1/b2/b3）
    b3 → {root, b1, b3}      （不含 a1/a2/a3）
  一次前向打 7 个节点（root+2+4），等价于 4 条路径各自 causal 验证
```

多候选的接受：greedy 与 stochastic 两条路。greedy（VerifyGreedy，Algorithm 2）：从根出发，沿「子节点 token = target argmax」的边下行，走到没有匹配子节点为止，再补一个 target 输出 token——即接受最长匹配前缀。stochastic 用 MSS（multi-step speculative sampling，多步投机采样）：逐节点对多个子候选做拒绝采样，保证输出分布仍等于 target（原文：guarantees equivalence）。其残差更新规则是 §2 命题 1 的多候选推广。

**命题 5**（MSS 残差保持）。设某节点处 target 分布为 $p$，有 $k$ 个子候选，第 $s$ 个候选 token 按 $x_s \sim q_s$ 独立采样（ $q_s$ 是该候选的 draft 分布——与命题 1「draft 采 $x \sim q$」同一采样假设，缺它则结论不成立，见下方边界辨析）。MSS 依次考察子候选：以 $\min(1, p(x_s)/q_s(x_s))$ 接受第 $s$ 个；拒绝则把 target 分布更新为 $\mathrm{norm}(\max(0, p - q_s))$ 再考察下一个；全部拒绝后从最终残差采样。则输出分布严格等于 $p$。原文 Algorithm 2 用随机序 `s ~ rand(H)` 选子候选；本命题对任意固定顺序亦成立（证明不依赖顺序）。

**证明**（路线：对子候选数 $k$ 归纳；基例 $k=1$ 即命题 1；归纳步用「第一个候选的接受 + 拒绝后残差」分解，关键一步是残差归一化常数恰等于第一个候选的拒绝概率）。基例 $k=1$：单个候选的接受 + 残差重采样正是命题 1，输出 $p$。归纳步：设 $k-1$ 个候选时结论成立。对 $k$ 个候选，第一个候选 $x_1$ 被接受的概率贡献为 $\min(q_1(x), p(x))$（同命题 1 推导）；拒绝概率为 $1 - \sum_x \min(p, q_1) = Z_1$，拒绝后分布更新为 $p_1 = \mathrm{norm}(\max(0, p - q_1))$。由归纳假设，剩余 $k-1$ 个候选在 $p_1$ 上输出分布 $p_1$。故总输出 $= \min(p, q_1) + Z_1 \cdot p_1 = \min(p,q_1) + \max(0, p-q_1) = p$。归纳步的完整代数留作练习 3。证毕（归纳步见练习 3）。

边界辨析（采样假设不可省）：证明把接受贡献写成 $\min(p, q_1)$（整个分布的逐元素 $\min$），这隐式用了 $x_1 \sim q_1$ 采样。若子候选是确定性 top-k token（greedy expansion 即如此）而非按 $q_s$ 采样，接受质量是点质量 $\min(1, p(x_s)/q_s(x_s))$ 而非 $\min(p, q_s)$，残差 $\max(0, p-q_s)$ 补不全被漏掉的 token。反例： $p=(0.5,0.3,0.2)$、 $q=(0.4,0.4,0.2)$（词表 $\lbrace a,b,c\rbrace$）、确定性候选 $\lbrace a,b\rbrace$： $a$ 的 $p(a)/q(a)=1.25>1$ 恒被接受， $c$ 既不在候选也不在残差 $\max(0,p-q)=(0.1,0,0)$，故 $P(c)=0 \ne 0.2$，分布破坏。所以 MSS 的分布保持针对随机解码（stochastic）；greedy 验证走 VerifyGreedy 另一条路（上文），不套此命题。

数值实例：tree 比单链多覆盖多少？SpecInfer Table 1 给了一组实测（[Miao et al., 2023](../references.md)，§3，LLaMA-7B target + LLaMA-68M draft，greedy，Alpaca 数据集）：top-k 候选对 target 下一 token 的覆盖率随 $k$ 为 68%（ $k$ =1）、77%（ $k$ =2）、81%（ $k$ =3）、84%（ $k$ =4）、85%（ $k$ =5）。 $k$ 从 1 到 2 覆盖率涨 9 个百分点，从 4 到 5 只涨 1 个——边际递减。所以 tree 的分支因子不必大：top-2 / top-3 已吃掉大部分收益，再宽只增 verify 节点数（算力）而接受率几乎不涨。这笔账与 §4 最优 $\gamma$ 同构：产出饱和、成本线性，存在最优规模。

tree 的 KV cache 含义。tree 前向会为所有节点（含将被拒绝的分支）写 KV；只有被接受路径的 KV 应保留，拒绝分支的 KV 须丢弃 / 不提交。块管理上这对应「 speculative 分配 + 按接受结果回滚」，与 2.1 的 COW / 块表语义衔接（接受路径继承共享前缀块，拒绝分支的独占块归池）；引擎侧的具体回滚实现归 4.5 / 5.4。SpecInfer 原文用 DFS 遍历更新共享 KV cache（§4.2）以避免逐序列各存一份——共享前缀的 KV 只存一份，正是命题 4「祖先 KV 相同」的工程兑现。

## 6. batch 效应：接受率方差与固定长度 verify 的浪费

$\alpha$ 是常量吗？单请求分析里假设 i.i.d.，真实 batch 里不是。DeepInfer 官方 doc 逐字（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实，L303）：「Because the acceptance probability of a draft token varies across requests and positions, fixed-length verification wastes verification compute on low-probability tokens.」接受概率跨请求、跨位置变化：代码补全的 draft 接受率高（局部模式强），开放对话低；同一请求里靠前的位置高、靠后的位置低（误差累积）。给所有请求统一 $\gamma$，后果是高 $\alpha$ 请求「猜少了」（本可接受更多）、低 $\alpha$ 请求「验多了」（算力浪费在低概率 token 上）。

单请求下这笔浪费看不见——§1 已算 verify 计算只占步时 1.7%，浪费的是闲置算力，不花钱。大 batch 下不一样。换口径声明：本节取 $T$ = 256、 $B$ = 128（短上下文大 batch），原因是长上下文下 KV 读主导分母、verify 计算被带宽下界掩盖，须短上下文才能暴露 compute-bound。此时每步 verify 打 $B(\gamma+1)$ 个 token，计算量随 batch 线性涨，而权重只读一遍、KV 读 $B \kappa L T$。代入章脊几何（ $\gamma$ = 4）：verify token 数 $128 \times 5 = 640$，计算 $640 \times f(256) \approx 640 \times 1.50 \times 10^{10} = 9.6 \times 10^{12}$ FLOPs，折合峰值时间 9.71 ms；带宽下界 $(W + B\kappa L T)/\mathrm{BW} = (1.61 \times 10^{10} + 128 \times 3.36 \times 10^{7})/(3.35 \times 10^{12}) \approx 6.09$ ms。计算 9.71 ms > 带宽 6.09 ms——verify 进入 compute-bound，步时由计算决定，每个被浪费的 verify token 都真实拉长墙钟。

浪费有多大？设该 batch 平均 $\alpha$ = 0.3（低接受率 workload），式 (5) 给 $E[N_{\mathrm{tok}}] = (1-0.3^{5})/0.7 = 1.4251$，有效产出 $128 \times 1.4251 \approx 182$ token，而 verify 打了 640 个——浪费比例 $1 - 182/640 = 0.715$（71.5%）。有效 token 自身的计算只需 $182 \times 1.50 \times 10^{10}/(989.5 \times 10^{12}) \approx 2.77$ ms，却被 640 token 的 verify 拖到 9.71 ms。用 Leviathan Thm 3.11 的运算量因子核对： $(1-\alpha)(\gamma\hat{c}+\gamma+1)/(1-\alpha^{\gamma+1})$，取 $\hat{c} = 0.1$ 得 $0.7 \times (0.4+5)/0.99757 \approx 3.79$——总运算量是基线的 3.79 倍，与「640 verify / 182 有效 ≈ 3.5」同量级（差异来自 draft 运算与 bonus 口径）。

变长 verify 正是冲这笔浪费来的。DSpark 的做法（官方 doc L303，详展归 7.2）：用 Confidence Head 预测每个 draft token 的接受率，按期望接受率排序，选最大化有效吞吐的前缀子集送去 verify，而非固定 $\gamma$。生产 draft 一次出 5 个 token，变长 verify 把每步实际 verified 数从 5.00 降到 3.2–4.4（约 −30%），平均 accepted 几乎不变（3.29 vs 3.20， $E[K]$ 口径）；EP64/DP64、约 60 并发/rank 的生产操作点上折合 +4–7% 吞吐（官方披露增益表：60 并发 +5.4%、72 并发 +6.7%）。机制对应本节账：砍掉低接受率尾部的 verify token，compute-bound 区直接省墙钟，而 accepted 几乎不损（被砍的本就大概率被拒）。本节只立动机与账，DSpark 的 Confidence Head 结构、排序与吞吐最优子集选择归 7.2 正式走读（回链 4.4）；模型侧 draft 结构归 models 维度（V4.1 待补）。

> **提点**：接受率方差把「 $\gamma$ 选多少」从单参数变成逐请求决策。固定 $\gamma$ 是 i.i.d. 假设下的最优（§4 数值搜索）；方差大时，逐请求按预测接受率分配 verify 预算才是最优——这是 4.5 EAGLE-2 动态 tree 与 7.2 DSpark 变长 verify 的共同动机，两者都在「把 verify 算力花在边际收益最高的 token 上」。

## 7. draft 分类学：六条路线

draft 模型从哪来？按「额外训练 / 额外内存 / 成本比 $c$ 」三轴分六条路线，机制细节与引擎实现归 4.5，本节只立分类与账面特征。

| 路线 | draft 来源 | 额外训练 | 额外内存 | $c$ 量级 | 代表（等级） |
| --- | --- | --- | --- | --- | --- |
| 独立小模型 | 同架构小 Transformer | 无（用现成 checkpoint） | 一份小模型权重 | $\approx N_q/N_p$，可至 0.01 | Leviathan / Chen [B] |
| self-draft（layer skip） | 同一模型跳过中间层 | 无 | 无 | $\approx$ 保留层占比，偏大 | Draft & Verify [A] |
| Medusa heads | 原模型顶上加多个预测头 | 训 heads（或全模型） | heads 参数 | 小（heads 轻） | Medusa [A] |
| EAGLE feature | 单层网络预测 target feature | 训 draft 网络 | 一份轻量 draft | 小（ $\approx 1/L$ ） | EAGLE-1/2/3 [A] |
| retrieval（REST） | datastore 检索候选续写 | 无 | datastore（可在 CPU/盘） | $\approx 0$（检索延迟） | REST [A] |
| n-gram | prompt / 语料 n-gram 查表 | 无 | 查表结构 | $\approx 0$ | Leviathan §3.6 [B] |

三轴读法。额外训练：独立小模型 / self-draft / REST / n-gram 免训练（plug-and-play），Medusa / EAGLE 须训 draft 部件——训练成本换接受率。额外内存：独立小模型要常驻一份权重（与 target 争显存），self-draft / n-gram 几乎零额外，REST 的 datastore 可放 CPU / 盘（不占 HBM），EAGLE / Medusa 的 draft 部件轻。成本比 $c$ ：检索 / 查表类 $c \approx 0$（§4 提点：即使 $\alpha$ 低也有正收益），独立小模型 $c \approx N_q/N_p$，self-draft 的 $c$ 偏大（跳过一半层也只省一半计算，4.5 算例 $F$ 仅 1.01），EAGLE 单层 draft $c \approx 1/L$ 最小。

两条路线的机制预览（细节 4.5）。REST（[He et al., 2023](../references.md)，arXiv:2311.08252，[A]，NAACL 2024）：用当前上下文的最长后缀去 datastore 检索匹配文档，把检索到的续写构造成 Trie、剪掉低频分支，候选送 tree attention verify——免训练、可加速任意现成模型（repo README 口径）。它与 n-gram 的差别在候选来源：n-gram 只查 prompt / 近期上下文，REST 查外部 datastore，覆盖面更广。EAGLE（[Li et al., 2024](../references.md)，arXiv:2401.15077，[A]）：在 target 的 second-to-top-layer feature 上做自回归，比 token 级 draft 更准（feature 更平滑、不确定性更低），4.5 §1 展开。self-speculative 的 MTP 路线（DeepSeek V3 / V4.1 的 nextn 模块作 draft）也归 4.5，模型侧 MTP 目标 recap models 4.5 + 指引。

分类学的账面终点是 §4 的 $F$ ：给定 workload 的 $\alpha$ 与硬件的 $c$，式 (8) 决定哪条路线净赚。免训练路线（REST / n-gram / self-draft）部署门槛低但 $\alpha$ 或 $c$ 受限；训练路线（EAGLE / Medusa） $\alpha$ 高、 $c$ 小，加速上限高但要付训练与 draft 内存。没有普适最优——这是 4.5 引擎对照与 9.2 选型的输入。

## 总结

| 论断 | 回指 |
| --- | --- |
| decode memory-bound：8B@H100 单步带宽下界 4.97 ms、计算仅 0.017 ms；verify 5 token 计算占步时 1.7% → 闲置算力可换产出 | §1、1.2 式 (2)(4) |
| 接受概率 $\min(1, p/q)$ + 残差 $\mathrm{norm}(\max(0,p-q))$ → 输出分布严格 = $p$（任意 $q$） | 命题 1、式 (1)(2) |
| 接受率 $\beta = \sum \min(p,q) = 1 - \mathrm{TV}(p,q)$； $p=q$ 全接受、支撑不交全拒绝 | 命题 2、式 (3) |
| 每轮期望产出 $E[N_{\mathrm{tok}}] = (1-\alpha^{\gamma+1})/(1-\alpha)$（含 bonus）；被接受 draft 数 $E[K] = E[N_{\mathrm{tok}}]-1$（DSpark accepted 口径） | 式 (5)(6)、§3 口径注、练习 1 |
| 加速比 $F = (1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$， $c = c_d/c_v$ 无量纲；**存在** $\gamma$ 使 $F>1$ 当且仅当 $\alpha>c$（固定 $\gamma$ 判据见式 (9)）； $\gamma=1$ 给 $(1+\alpha)/(1+c)$ | 命题 3、式 (8)(9)(10)、练习 1 |
| 绝对成本式须乘 $c_v$ 归一，否则量纲为 tokens/s 非 factor | 式 (10)、§4 坑注 |
| tree mask「可见集 = 祖先 + 自身」→ 各路径 attention 等价独立 causal 验证；MSS 多候选残差保持分布 | 命题 4、命题 5、§5、练习 3 |
| batch 下 $\alpha$ 跨请求/位置方差 → 固定 $\gamma$ 浪费 verify 算力；compute-bound 区 71.5% 浪费 → DSpark 变长 verify（+4–7%） | §6、7.2 回链 |
| draft 六路线按额外训练/内存/ $c$ 分； $c\approx 0$ 路线即使 $\alpha$ 低也正收益 | §7 表、§4 提点 |

## 练习

1.（计算 · [10]）章脊口径 $\alpha$ = 0.8、 $\gamma$ = 4、 $c$ = 0.1：(a) 用式 (5) 算 $E[N_{\mathrm{tok}}]$、式 (6) 算 $E[K]$；(b) 用式 (8) 算 $F$，并求每 token 墙钟占基线的百分比。校验：(a) $E[N_{\mathrm{tok}}] - E[K] = 1$（bonus 位）；(b) $\gamma$ = 1 特例应给 $F = (1+\alpha)/(1+c) \approx 1.64$，与式 (8) 代入一致。
2.（推导 · [20]）从头推导命题 2：(a) 由接受规则对 $x \sim q$ 求期望证 $\beta = \sum_x \min(p,q)$；(b) 用 $\min(a,b) = (a+b-\lvert a-b \rvert)/2$ 证 $\beta = 1 - \mathrm{TV}(p,q)$；(c) 代入 $p=(0.5,0.3,0.2)$、 $q=(0.4,0.4,0.2)$ 算 $\beta$ 与 $\mathrm{TV}$。校验： $p=q$ 时 $\beta=1$、支撑不相交时 $\beta=0$（命题 2 边界）；(c) 的 $\beta$ 应等于直接求和 $\sum \min(p,q)$。
3.（证明 · [30]）补全命题 5（MSS 残差保持）的归纳步：设 $k-1$ 个子候选时输出分布等于当前 target 分布，证 $k$ 个时仍成立。要求：(a) 写出第一个候选的接受概率贡献 $\min(p, q_1)$ 与拒绝概率 $Z_1$；(b) 证残差 $p_1 = \mathrm{norm}(\max(0, p-q_1))$ 的归一化常数恰为 $Z_1$；(c) 用归纳假设合成总输出并化简到 $p$。校验： $k=1$ 时你的证明应退化为命题 1；关键恒等式 $\min(a,b)+\max(0,a-b)=a$ 与命题 1 同源。
4.（计算 · [40]）batch verify 浪费账（§6 口径 $T$ = 256、 $B$ = 128、 $\gamma$ = 4、 $\alpha$ = 0.3、 $\hat{c}$ = 0.1）：(a) 先预测 verify 是 compute-bound 还是 memory-bound 并给理由，再算 verify 计算时间与带宽下界判定；(b) 算有效产出 token 数、浪费比例，并用 Leviathan Thm 3.11 算总运算量增加因子；(c) 若变长 verify 把每步 verified 数从 5 降到 3.58（DSpark 60 并发档），估算 compute-bound 步时的变化，并说明为何 accepted 几乎不损。校验：(a) 与 §6 的 9.71 ms / 6.09 ms 一致；(b) 浪费比例 ≈ 71.5%；(c) 方向与官方 doc L303 增益表（+4–7%）一致，accepted 口径见 §3 口径注。

## 参考

- [Leviathan et al., 2022](../references.md)：Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192，[B]（key: leviathan，Google 无官方开源）——接受规则 §2.3、Eq 1 与 capped geometric §3.1、Def 3.2 / Lemma 3.3 / Thm 3.5（ $\beta = 1-D_{LK}$ ）§3.2、Def 3.7 / Thm 3.8 / Cor 3.9 §3.3、Thm 3.11（运算量因子）§3.4、最优 $\gamma$ §3.5、近似模型与 n-gram §3.6、T5-XXL 2–3× §4
- [Chen et al., 2023](../references.md)：Accelerating Large Language Model Decoding with Speculative Sampling，arXiv:2302.01318，[B]（key: chen，DeepMind 无官方开源）——modified rejection sampling 保目标分布（hardware numerics 内）、Chinchilla 70B 基准；与 Leviathan 独立同期，公式源头并列
- [Miao et al., 2023](../references.md)：SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification，arXiv:2305.09781，[A]（key: specinfer，ASPLOS'24）——token tree（expansion / merge）§3、Tree Attention Def 4.1 §4.1、topology-aware causal mask §4.2、VerifyGreedy / VerifyStochastic（MSS）Algorithm 2 §4.3、Table 1 覆盖率；官方代码 flexflow/flexflow-train 的 inference 分支（spec_infer.cc / incr_decoding.cc）+ AE goliaro/specinfer-ae（原 flexflow/FlexFlow 已 301 更名，master 无 spec 代码，2026-11-23 核实）
- [He et al., 2023](../references.md)：REST: Retrieval-Based Speculative Decoding，arXiv:2311.08252，[A]（key: rest，NAACL 2024，FasterDecoding/REST）——datastore 检索 draft、免训练 plug-and-play（§7）
- [Li et al., 2024](../references.md)：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty，arXiv:2401.15077，[A]（key: eagle，SafeAILab/EAGLE）——feature 级 draft（§7 预览，4.5 详展）
- [Cai et al., 2024](../references.md)：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads，arXiv:2401.10774，[A]（key: medusa，FasterDecoding/Medusa）——多 heads + tree attention（§7 分类，4.5 详展）
- [Zhang et al., 2023](../references.md)：Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding，arXiv:2309.08168，[A]（key: draft-verify，ACL 2024，dilab-zju/self-speculative-decoding）——self-draft layer skip（§7 分类，4.5 详展）
- [Ye et al., 2025](../references.md)：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving，arXiv:2501.01005，[A]（key: flashinfer）——tree mask 接口 prefill.py maybe_packed_custom_mask（§5，kernel 算法归 3.1）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露）——3.35 TB/s、989.5 TFLOPS BF16 dense（§1 / §6 带宽与算力下界）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：DeepInfer 官方部署 QA，教学参照 + 官方披露（key: deepinfer-doc，2026-11-23 核实）——L303 接受率方差与变长 verify、DSpark 增益表（§6，详展归 7.2）
- 跨维度指引（纯文字）：models 2.4（KV cache 与前缀不可变命题，tree KV 回滚的依据）、models 4.5（MTP 目标，4.5 recap）；inference 1.2（decode FLOPs 式 (2)、强度式 (4)、ridge）、1.4（采样分布与 logits processor）、2.1（块管理 / COW，speculative KV 回滚）、3.1（tree mask kernel 算法）、4.1（continuous batching）、4.3（chunked prefill，verify 的 prefill 形态调度）、4.5（draft 路线系统实现）、5.4 / 6.x（引擎走读）、7.2（DSpark 变长 verify 正式走读，回链本章 §6）、9.2（引擎选型）
