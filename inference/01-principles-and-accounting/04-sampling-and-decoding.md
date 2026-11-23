# 1.4 采样与解码策略（Sampling and Decoding Strategies）

> 模型吐出的是一行 logits，用户要的是一段合法、可复现、可流式读取的文本——中间这层转换有哪些必须算清的数学与必须踩对的工程？本章交付截断采样的再归一化、logits processor 链、Gumbel-max 等价、结构化输出的 mask 原理、batch invariance 与流式 detokenization | 前置：1.1（请求生命周期、流式返回）、1.3（decode 步时与 batch 口径）、models 1.1（自回归分解与祖先采样）、models 2.3（softmax 与 max 减法）；KV cache COW 见 2.1 | 锚点：top-p [A]（arXiv:1904.09751）、min-p [A]（arXiv:2407.01082）、XGrammar [A]（arXiv:2411.15100）、Outlines [A]（arXiv:2307.09702）、guidance（教学参照）、batch-invariance 博客 [C]（开放技术博客、非闭源）、vLLM / SGLang / DeepInfer sampling 实现

## 1. 从 logits 到 token：采样分布 recap 与本章口径

一行 logits 怎么变成一个 token？模型侧的答案 models 1.1 已交付：自回归分解 $p(x) = \prod_t p(x_t \mid x_{\lt t})$ ，每步对词表做一个 categorical 分布、祖先采样（ancestral sampling）逐 token 抽取——本章只 recap 这一句，分布的模型侧定义归 models 1.1，不重推。serving 侧的问题是另一组：这个 categorical 分布要不要截断、logits 要先过哪些 processor、浮点怎么算才不溢出、同样的输入为什么换个 batch 就吐出不同的字、字节流怎么增量地还原成文本。这些是本章的增量。

**本章口径**（R2-2 章脊锁定）：词表大小记 $V$ ，token $i$ 的 logit 记 $z_i$ 、概率记 $p_i$ ；温度记 $\tau$（**非序列长 $T$**，亦与 1.3 TPOT SLO 上限 $\tau$ 不同章不同义）； $B$ = batch 内序列数。截断参数：top-k 数记 $k_{\mathrm{top}}$（**与 MoE top-k 的 $k$ 区分**，spec §4.3 的 $k$ 专归 MoE）、top-p 阈值记 $p_{\mathrm{th}}$ 、min-p 基准比例记 $\eta$ 。penalty 系数：repetition 记 $r$（ $r > 1$）、frequency / presence 记 $\alpha_{\mathrm{freq}}$ / $\alpha_{\mathrm{pres}}$（**与投机解码接受率 $\alpha$（4.4）无关**，下标区分）、token $i$ 已出现次数记 $c_i$ 。beam 宽度记 $w$ 。全章数值例用一个 5-token 玩具分布 $p = [0.4, 0.25, 0.15, 0.12, 0.08]$（和为 1）与 logits $z = [2, 1, 0]$，换口径处一句说明。

最朴素的解码是 greedy：每步取 logit 最大的 token（等价于 $\tau \to 0$ 的采样极限）。greedy 给定 logits 是确定的，但「确定」不等于「可复现」——logits 本身随 batch 组成漂移（§6）。带温度的采样把 logits 缩放后做 softmax：

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{V} \exp(z_j / \tau)} \qquad (1)
$$

**符号**： $\tau > 0$ = 温度； $V$ = 词表大小。 $\tau < 1$ 锐化分布（放大 logit 差）、 $\tau > 1$ 平坦化、 $\tau \to 0$ 退化为 greedy、 $\tau \to \infty$ 趋于均匀。代入 $z = [2, 1, 0]$： $\tau = 1$ 给 $p = [0.665, 0.245, 0.090]$； $\tau = 0.5$（logit 变 $[4, 2, 0]$）给 $[0.867, 0.117, 0.016]$——温度减半，头部 token 概率从 0.665 抬到 0.867。温度是多样性与连贯性的旋钮，但它只改分布形状，不解决「长尾 token 偶发乱码」——那要截断。

## 2. 截断采样：top-k / top-p / min-p 与再归一化

为什么不能直接从完整 softmax 采样？因为长尾里藏着低概率的坏 token，偶发抽中就是一段乱码或重复。截断采样把分布限制在一个「核」上再归一化。三种截断的差别只在核怎么选。

**top-k**：保留 logit 最大的 $k_{\mathrm{top}}$ 个 token，其余置零。核大小固定，与分布形状无关——分布很尖时仍强行保留 $k_{\mathrm{top}}$ 个（含坏 token），分布很平时又砍掉太多合理选项。这是 top-k 的结构性缺陷。

**top-p（nucleus sampling）**：核大小随分布自适应。把概率降序排列 $p_{(1)} \ge p_{(2)} \ge \cdots$ ，累积和 $C_j = \sum_{i \le j} p_{(i)}$ ，取累积质量首次达到 $p_{\mathrm{th}}$ 的最小前缀（[Holtzman et al., 2019](../references.md)，arXiv:1904.09751，[A]）：

$$
S_p = \lbrace (1), \ldots, (j^{\ast}) \rbrace, \qquad j^{\ast} = \min \lbrace j : C_j \ge p_{\mathrm{th}} \rbrace \qquad (2)
$$

**符号**： $p_{(i)}$ = 降序第 $i$ 个概率； $C_j$ = 前 $j$ 个累积质量； $j^{\ast}$ = 核的末位秩； $S_p$ = nucleus。口径注：含跨阈 token（ $C_{j^{\ast}} \ge p_{\mathrm{th}}$ 的那个本身入核），这是 Holtzman 原文「最小集合使累积概率 $\ge p$」的约定。

**命题 1**（nucleus 最小性）： $S_p$ 是降序前缀中累积质量 $\ge p_{\mathrm{th}}$ 的**最小**集合（按基数）。**证明**： $C_j$ 随 $j$ 单调增（概率非负），故 $\lbrace j : C_j \ge p_{\mathrm{th}} \rbrace$ 是一个上集，其最小元 $j^{\ast}$ 存在且唯一；任何基数更小的前缀 $j < j^{\ast}$ 都有 $C_j < p_{\mathrm{th}}$ ，不满足质量约束。正例： $p = [0.4, 0.25, 0.15, 0.12, 0.08]$、 $p_{\mathrm{th}} = 0.8$，累积 $C = [0.4, 0.65, 0.80, 0.92, 1.0]$， $j^{\ast} = 3$（ $C_3 = 0.80$ 首次达阈），核 $S_p = \lbrace 1, 2, 3 \rbrace$。反例（边界）：若误取 $j = 2$， $C_2 = 0.65 < 0.8$ 不达标——最小性把「恰好够」与「不够」分开。

截断后必须再归一化，否则概率和小于 1、不是合法分布。通用截断再归一化（对任意核 $S$）：

$$
p'_i = \begin{cases} p_i / \sum_{j \in S} p_j & i \in S \\ 0 & i \notin S \end{cases} \qquad (3)
$$

**符号**： $S$ = 截断核（top-k / top-p / min-p 选出）；分母是核内概率质量。代入 top-p 例：核质量 $0.4 + 0.25 + 0.15 = 0.8$，再归一化得 $p' = [0.5, 0.3125, 0.1875, 0, 0]$（和为 1）。注意核内相对比例不变（ $0.4 : 0.25 : 0.15 = 0.5 : 0.3125 : 0.1875$），截断只砍尾、不改头部形状。

**min-p**：阈值随分布峰值自适应。保留 $p_i \ge \eta \cdot p_{\max}$ 的 token（ $p_{\max} = \max_j p_j$，[Nguyen et al., 2024](../references.md)，arXiv:2407.01082，[A]）：

$$
S_{\min} = \lbrace i : p_i \ge \eta \cdot p_{\max} \rbrace \qquad (4)
$$

**符号**： $\eta \in (0, 1)$ = 基准比例（典型 0.05–0.1）； $p_{\max}$ = 峰值概率。与 top-p 的机制差异：top-p 按**累积质量**切（固定总概率预算），min-p 按**相对峰值**切（固定与最大概率的比值）——分布越尖（ $p_{\max}$ 大），min-p 阈值越高、核越小；分布越平，阈值越低、核越大。代入同一玩具分布、 $\eta = 0.3$：阈值 $0.3 \times 0.4 = 0.12$，保留 $p_i \ge 0.12$ 即 $\lbrace 0.4, 0.25, 0.15, 0.12 \rbrace$（4 个，砍掉 0.08），核质量 0.92，再归一化 $[0.435, 0.272, 0.163, 0.130, 0]$。对照 top-p（ $p_{\mathrm{th}} = 0.8$）核只有 3 个：同一分布上 min-p 更宽松，因为它按相对峰值而非累积质量切。两者都接式 (3) 再归一化。

## 3. logits processor 链与数值稳定

截断之前，logits 还要过一条 processor 链。顺序为什么重要？因为有的 processor 是乘性、有的是加性，彼此不可交换。所谓 logits processor，是指在采样前对 logits 逐 token 施加的一个变换（penalty、ban、mask），多个 processor 按固定顺序串成链。

**penalty 三式**（HF transformers / OpenAI API 的通用约定，教学口径；引擎实现见 [vLLM repo](../references.md) `vllm/v1/worker/gpu/sample/sampler.py` L42（`v1/sample/` 只剩 ops/，2026-11-23 main 核实；走读见 5.4 §7）、[SGLang repo](../references.md) `srt/layers/sampler.py`）。repetition penalty（乘性，对已出现 token）：

$$
z_i \leftarrow \begin{cases} z_i / r & c_i > 0 \text{ 且 } z_i > 0 \\ z_i \cdot r & c_i > 0 \text{ 且 } z_i \le 0 \\ z_i & c_i = 0 \end{cases} \qquad (5)
$$

frequency / presence penalty（加性）：

$$
z_i \leftarrow z_i - \alpha_{\mathrm{freq}} \cdot c_i - \alpha_{\mathrm{pres}} \cdot \mathbf{1}[c_i > 0] \qquad (6)
$$

**符号**： $r > 1$ = repetition 因子； $c_i$ = token $i$ 在已生成序列中的出现次数； $\alpha_{\mathrm{freq}}$ / $\alpha_{\mathrm{pres}}$ = frequency / presence 系数； $\mathbf{1}[\cdot]$ = 指示函数。frequency 按出现次数线性压（越常用越压），presence 只按「是否出现过」二值压（出现一次即压）。ban tokens 是 mask 的特例：对禁位 $z_i \leftarrow -\infty$，softmax 后概率精确为 0（models 2.4 §1 的 $-\infty$ 极限语义）。

**命题 2**（penalty 不可交换）：乘性 repetition（式 (5)）与加性 frequency（式 (6)）一般不可交换，链顺序改变输出。**证明**（构造反例）：取 $z = [2, -1]$、 $c = [1, 0]$、 $r = 2$、 $\alpha_{\mathrm{freq}} = 1$。先 repetition 后 frequency：token 0 正 logit 除 2 得 1，再减 $1 \times 1$ 得 0；token 1 未出现不动，得 $[0, -1]$。先 frequency 后 repetition：token 0 减 1 得 1，再除 2 得 0.5；得 $[0.5, -1]$。两序结果不同（ $0 \ne 0.5$），故不可交换。机制：乘性缩放与加性平移不 commute（ $(z/r) - \alpha_{\mathrm{freq}} \ne (z - \alpha_{\mathrm{freq}})/r$）。所以引擎必须固定 processor 顺序——典型链为 penalty → temperature → 截断 mask → 再归一化 → 采样，顺序是语义的一部分，不是实现细节。

数值稳定是另一条暗线。直接算 $\exp(z_i)$ 在 $z_i$ 大时溢出（fp32 上 $\exp(89)$ 即超最大可表值）。对策是 max 减法（models 2.3 已交付 softmax 的 max 减法，本章接 log 域）。

**命题 3**（softmax 平移不变）：对任意常数 $c$， $\mathrm{softmax}(z + c \mathbf{1}) = \mathrm{softmax}(z)$。**证明留作练习 2**（要点：分子分母同乘 $\exp(c)$ 约去）。由此得稳定的 log_softmax：

$$
\log p_i = z_i - m - \log \sum_{j} \exp(z_j - m), \qquad m = \max_j z_j \qquad (7)
$$

**符号**： $m$ = logits 最大值；减去 $m$ 后指数参数 $\le 0$， $\exp$ 落在 $(0, 1]$ 不溢出。代入 $z = [1000, 1001, 999]$：直接 $\exp(1000)$ 溢出；减 $m = 1001$ 得 $[-1, 0, -2]$， $\exp$ 为 $[0.368, 1, 0.135]$，softmax $[0.245, 0.665, 0.090]$——与不减 max 的数学值一致，但数值可算。Gumbel-max 直接在 logits 上加噪声取 argmax，绕开 softmax 的指数运算，是另一条数值路径。

**Gumbel-max trick**：把 categorical 采样改写成「加噪声取最大」。设 $g_i$ 独立同分布 $\mathrm{Gumbel}(0,1)$（CDF $F(x) = \exp(-\exp(-x))$，密度 $f(x) = \exp(-x - \exp(-x))$），采样规则为

$$
i^{\ast} = \mathrm{argmax}_i \left( \frac{z_i}{\tau} + g_i \right), \qquad g_i \sim \mathrm{Gumbel}(0,1) \qquad (8)
$$

**命题 4**（Gumbel-max 等价）：式 (8) 的 $i^{\ast}$ 服从 $\mathrm{Categorical}(\mathrm{softmax}(z/\tau))$，即与式 (1) 的祖先采样同分布。**证明**（路线：4 步——条件化 $g_i$ 用独立性把「 $i$ 胜出」写成 CDF 连乘 → 合并指数引出归一化常数 $A$ → 换元 $u = e^{-t}$ 把积分化成指数积分 → 得 $1/A$ 即 softmax；关键一步是换元后积分恰好是 $1/A$）。下面取 $\tau = 1$ 证（一般 $\tau$ 把 $z$ 换 $z/\tau$ 同理）。 $i$ 胜出当且仅当对所有 $j \ne i$ 有 $z_i + g_i > z_j + g_j$，即 $g_j < t + z_i - z_j$（条件化 $g_i = t$）。由独立性，概率为 CDF 连乘，再对 $t$ 积分：

$$
\begin{aligned}
\Pr(i^{\ast} = i) &= \int_{-\infty}^{\infty} f(t) \prod_{j \ne i} F(t + z_i - z_j) \thinspace dt \\
&= \int_{-\infty}^{\infty} e^{-t} \exp\big(-e^{-t} A\big) \thinspace dt, \qquad A = e^{-z_i} \sum_j e^{z_j} \\
&= \int_{0}^{\infty} e^{-A u} \thinspace du = \frac{1}{A} = \frac{e^{z_i}}{\sum_j e^{z_j}}
\end{aligned}
$$

第二行合并： $f(t) \prod_{j \ne i} F = \exp(-t - e^{-t}) \cdot \exp(-e^{-t-z_i} \sum_{j \ne i} e^{z_j}) = e^{-t} \exp(-e^{-t} A)$，其中 $A = 1 + e^{-z_i} \sum_{j \ne i} e^{z_j} = e^{-z_i} \sum_j e^{z_j}$。第三行换元 $u = e^{-t}$（ $du = -e^{-t} dt$， $t : -\infty \to \infty$ 对应 $u : \infty \to 0$），积分化为 $\int_0^\infty e^{-Au} du = 1/A$。末式正是式 (1) 的 $p_i$（ $\tau = 1$），证毕。数值一例： $z = [2, 1, 0]$、 $\tau = 1$，抽一组 Gumbel 噪声 $g = [0.5, -1.2, 0.3]$，则 $z + g = [2.5, -0.2, 0.3]$，argmax 取 token 0；换一组噪声 argmax 可能翻到 token 1，抽样 $10^{5}$ 次的经验频率收敛到 $[0.665, 0.245, 0.090]$（练习 4 验证）。工程含义：Gumbel-max 把采样变成逐 token 独立的「加噪声 + argmax」，无需先算全词表 softmax，且 argmax 对 logits 的公共平移不变（与式 (7) 同源）——这是它在 fused sampling kernel 里被采用的数学依据（kernel 归 3.3）。

processor 链的 shape 走读（ $B$ 条序列并行，每行独立）：

```
 logits z [B, V]
   │  repetition / frequency / presence penalty（式 (5)(6)，逐 token 改 logit）
   ▼
 z' [B, V]
   │  temperature：z' / τ（式 (1) 的缩放）
   ▼
 z'/τ [B, V]
   │  top-k / top-p / min-p / grammar mask：非法位 → −∞（式 (2)(4)、§5）
   ▼
 masked [B, V]
   │  log_softmax（式 (7)）+ 截断再归一化（式 (3)）
   ▼
 p [B, V]（每行和为 1，被截位为 0）
   │  Gumbel-max（式 (8)）/ categorical 采样
   ▼
 token id [B]
```

## 4. beam search 与 KV cache 耦合

采样保分布，beam search 求最优序列——两者目标不同。beam search 每步保留累计对数概率最高的 $w$ 条候选（beam 宽度 $w$），逐步扩展再剪枝。原始累计分 $\sum_t \log p_t$ 随长度单调减（每乘一个 $< 1$ 的概率），长序列天然吃亏——不做长度归一化，beam search 系统性偏好短输出。长度归一化（GNMT 口径）：

$$
s(y) = \frac{1}{|y|^{\alpha_{\mathrm{len}}}} \sum_{t=1}^{|y|} \log p(y_t \mid y_{\lt t}) \qquad (9)
$$

**符号**： $|y|$ = 序列长度； $\alpha_{\mathrm{len}} \in [0, 1]$ = 长度惩罚指数（下标区分投机解码接受率 $\alpha$； $\alpha_{\mathrm{len}} = 0$ 不归一、 $\alpha_{\mathrm{len}} = 1$ 完全按均分）。代入两条候选：A 长 2、累计 $\log p = -1.5$；B 长 10、累计 $\log p = -6.0$。原始分 A（ $-1.5$）> B（ $-6.0$），选 A（短）； $\alpha_{\mathrm{len}} = 1$ 归一后 A $= -0.75$、B $= -0.6$，B > A，选 B（每 token 质量更高）——归一化把长度偏置去掉。early stop：当已凑齐 $w$ 条完整假设（遇 EOS）时停止扩展，因为再扩展只会让未完整候选的归一化分更难看。

beam search 与 KV cache 的耦合是 serving 侧的真问题： $w$ 条 beam 共享公共前缀的 KV，分叉时若整份复制 KV 则显存翻 $w$ 倍。解法是块共享 + 写时复制（COW）——公共前缀块只存一份，某条 beam 要在共享块内写新 token 时才复制该块（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]，§4.4 原文「copying one block of data」）。COW 的块管理语义与正确性证明归 2.1，本章只立耦合事实：beam / parallel sampling 的 KV 共享是 COW 机制的直接消费方。现代 chat serving 以采样为主（多样性需求），beam search 多见于翻译、结构化抽取等求最优解的任务；投机解码是另一条加速路线（保分布的前提下提速，归 4.4），与 beam 的「求最优」目标不同。

## 5. 结构化输出原理：grammar → token mask

怎么保证模型一定吐出合法 JSON？答案不在 prompt 里，在采样前的 mask 里。所谓结构化输出（structured output），是指用形式语法约束生成，使输出必属于指定语言。机制统一为三步：把约束（regex / JSON schema / CFG）编译成一个自动机；每个 decode 步，自动机当前状态 $s$ 决定合法下一 token 集合 $A(s) \subseteq \lbrace 1, \ldots, V \rbrace$；对 $i \notin A(s)$ 置 $z_i = -\infty$（mask），再从受限分布采样。

**命题 5**（约束保持）：若每步都 mask 到 $A(s)$ 且按自动机转移 $s \leftarrow \delta(s, \text{token})$，则生成的字符串被自动机接受。**证明**（对步数归纳，三段）：基例——第 0 步处于起始状态 $s_0$，前缀为空，合法。保持——设第 $t$ 步后前缀合法、状态为 $s_t$；第 $t+1$ 步 mask 保证只抽 $A(s_t)$ 中的 token，转移 $\delta(s_t, \text{token})$ 有定义且到达合法状态 $s_{t+1}$，前缀仍合法。终止——生成结束（EOS）时处于接受状态，整串被接受。故逐步 mask 是输出合法性的充分保证。反例（边界）：若某步漏 mask 抽到 $A(s)$ 外的 token，转移无定义、自动机进入死状态，后续无论怎么生成都无法被接受——合法性是逐步不变式，一步破则全破。

三个系统的差别在自动机类型与 mask 计算时机（引擎集成归 5.4 / 6.1，本章只讲原理）：

| 系统 | 约束对象 | 自动机 | mask 计算 | 证据 |
| --- | --- | --- | --- | --- |
| Outlines | regex / JSON schema | FSM（字符级有限状态机） | 预计算「FSM 状态 → 允许 token 集」index，运行时 O(1) 查表 | [Willard & Louf, 2023](../references.md)，arXiv:2307.09702，[A] |
| XGrammar | 上下文无关语法（CFG） | PDA（下推自动机） | 自适应 mask 扩展：上下文无关 token 预计算 mask + 上下文相关 token 运行时检查 | [Dong et al., 2024](../references.md)，arXiv:2411.15100，[A] |
| guidance | 程序化 token 级约束 | 状态化程序 | 运行时按程序状态施加约束 | [guidance repo](../references.md)，教学参照 |

机制对照的关键：Outlines 把 regex 编译成字符级 FSM，再**离线**把每个 FSM 状态映射到「从该状态出发能合法 emit 的 token 集合」建成 index——运行时每步只做一次 O(1) 查表，mask 开销与词表大小无关（[Willard & Louf, 2023](../references.md)，[A]）。XGrammar 处理更强的 CFG（JSON schema 的嵌套递归是上下文无关而非正则），用 PDA；其自适应 mask 扩展把词表分成上下文无关 token（mask 可预计算）与上下文相关 token（须运行时按 PDA 栈状态检查，占比小），预计算 + 少量运行时检查结合（[Dong et al., 2024](../references.md)，[A]）。表达力与开销的权衡：FSM 只能表正则语言但 mask 全预计算、最便宜；PDA 能表上下文无关语言但需运行时栈检查；guidance 以程序表达约束、灵活但运行时逻辑最重。生产部署中 grammar 编译可异步化、结构化输出专用机形态归 9.4 展开（回链 1.4），本章不展开引擎实现。

## 6. 确定性采样与 batch invariance

greedy 解码为什么换个 batch 就吐出不同的字？因为 logits 本身不是 batch 不变的。根源是浮点加法不结合。

**命题 6**（浮点非结合）：浮点加法一般不满足结合律， $(a + b) + c \ne a + (b + c)$。**证明**（fp32 反例）：取 $a = 2^{24}$、 $b = 1$、 $c = -2^{24}$。fp32 有效位 24 位， $2^{24} + 1$ 不可表、舍入回 $2^{24}$，故 $(a+b)+c = 2^{24} - 2^{24} = 0$；而 $b + c = 1 - 2^{24} = -16777215$ 可表， $a + (b+c) = 2^{24} - 16777215 = 1$。 $0 \ne 1$，证毕。机制：大数吃小数——对齐阶码时小数的有效位被移出窗口（与 training 1.3 混合精度的更新消失界同源）。

把这条接到 serving：matmul / attention / RMSNorm 都含 reduction（沿某维求和）。reduction 的分区与顺序随 batch 组成变化——batch 大小不同，split-K 的切分、线程块的归约树形状就不同，求和顺序随之不同，由非结合律 logits 末位比特不同。两个 logit 接近时，末位漂移足以翻转 argmax，greedy 也吐出不同 token。Thinking Machines 博客（[He, 2025](../references.md)，[C] 开放技术博客、非闭源，2025-09-10，URL 已核实）把浮点非结合称为 "the original sin"，并指出「concurrency + floating point」假说（atomicAdd 累加顺序随并发完成顺序不定）**只是部分解释**——batch 组成改变 reduction 分区/顺序同样改变结果，无需并发非确定性。解法是 batch-invariant kernels：固定 reduction order，使输出与 batch 组成无关（RMSNorm / matmul / attention 三类）。

引擎的 batch-invariant 模式入口对照（kernel 层固定 reduction order 的实现——split-K 固定分区、禁 atomicAdd 非确定累加——归 3.4，本章只列模式入口）：

| 引擎 | 模式入口 | 路径（2026-11-23 tarball 核实） |
| --- | --- | --- |
| vLLM | determinism 模块 | `vllm/model_executor/determinism/batch_invariant.py` + `batch_invariant_configs.py`（无 `batch_invariant_ops` 目录） |
| SGLang | batch_invariant_ops | `srt/batch_invariant_ops/batch_invariant_ops.py` |
| DeepInfer | 环境变量 | `DEEPINFER_BATCH_INVARIANT`（`deepinfer/model_executor/models/v4_modules/attn.py` 消费） |

> **坑**：把「设了随机种子」当成「输出可复现」——种子只固定 Gumbel/categorical 的随机数流，固定不了 logits 的 reduction 漂移；batch 组成一变，greedy 输出照样变。规避：要 bitwise 可复现须开 batch-invariant 模式（上表），且固定 batch 组成；只设种子不够。

确定性有代价：固定 reduction order 往往放弃最优 split-K 并行度，kernel 吞吐下降（3.4 量化）。所以 batch invariance 是按需开关——调试、评测、需要可复现的生产场景开，纯吞吐场景关。这是「确定性 vs 性能」的又一个操作点选择，与 1.3 的 SLO 操作点同构。

## 7. 流式输出工程：增量 detokenization

token 流怎么变成用户屏幕上逐字浮现的文本？难点在 token 边界与字符边界不对齐。所谓增量 detokenization（incremental detokenization），是指每收到一个新 token 就增量地还原并输出文本，而非等全部生成完再一次性解码。两个必须处理的边界问题：

其一，**UTF-8 跨 chunk**。一个 Unicode 码点占 1–4 字节，BPE token 的字节边界可能把一个多字节字符劈成两半（前一个 token 含字符的前 2 字节、后一个 token 含第 3 字节）。逐 token 独立解码会在半个字符处产出替换符（U+FFFD）或乱码。对策：缓冲尾部不完整字节序列，凑齐一个完整码点再 emit——detokenizer 须持有一个「待定字节」状态，跨 token 维护。

其二，**stop 序列与 special token 的增量检测**。流式输出要随时判断是否命中停止串（stop sequence）或特殊 token（如工具调用标记），命中即截断。朴素做法是每来一个 chunk 就把已生成的整段输出重扫一遍找停止串——输出长 $n$、停止串长 $m$，每 chunk 重扫 $O(n m)$，全程 $O(n^2 m)$，长输出下平方级恶化。对策是增量 KMP：把停止串编译成 KMP 自动机，维护「当前部分匹配长度」状态 $j$ 跨 chunk 携带，每个新字符只做 $O(1)$ 摊还的状态转移，全程 $O(n)$。状态机在 chunk 边界不重置——这正是「增量」的含义：匹配进度是流式状态的一部分，不随 chunk 切分丢失。

DeepSeek Recipe（[deepseek-recipe repo](../references.md)，教学参照，Rust 实现）把这套工程落地：多协议（Chat Completions / Responses / Messages）统一对话格式、special-token 注入防护、分段并行 BPE、流式状态机 + 增量 KMP。DeepInfer doc 官方披露（[DeepInfer doc](../references.md)，教学参照 + 官方披露口径，2026-11-23 核实，L35）："DeepSeek Recipe handles this encoding and decoding efficiently and accurately; by segmenting on special tokens and encoding the segments in parallel, it prevents special-token injection while reducing tokenization latency for long inputs"。special-token 注入防护是安全要件：用户输入里若含模型的特殊 token 字面量，须在 tokenize 阶段拦截，防止伪造控制标记劫持生成。Recipe 的详细走读（状态机结构、增量 KMP 实现、分段并行 BPE）归 7.4，本章只立流式 detokenization 的原理与两个边界问题。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 温度缩放 logits 后 softmax；τ→0 退化为 greedy、τ→∞ 趋均匀 | 式 (1)；z=[2,1,0] 算例 | §1、练习 1 |
| top-p 核 = 累积质量首达 p_th 的最小降序前缀（含跨阈 token），最小性可证 | 式 (2)；命题 1 | §2、练习 1 |
| 截断后须再归一化（核内相对比例不变）；min-p 按相对峰值切、top-p 按累积质量切 | 式 (3)(4)；玩具分布算例 | §2、练习 1 |
| repetition（乘性）与 frequency（加性）penalty 不可交换，链顺序是语义 | 式 (5)(6)；命题 2 反例 | §3、练习 3 |
| softmax 平移不变 → log_softmax 减 max 防溢出；Gumbel-max 与祖先采样同分布 | 式 (7)(8)；命题 3 / 4 + 证明 | §3、练习 2 / 4 |
| beam search 须长度归一化去短偏置；KV 共享靠 COW（归 2.1） | 式 (9)；COW 耦合 | §4、2.1 |
| 结构化输出 = 自动机状态 → token mask，逐 step mask 是合法性不变式 | 命题 5（约束保持）；三系统对照 | §5、5.4 / 6.1 / 9.4 |
| 浮点非结合 → reduction order 随 batch 变 → logits 漂移 → greedy 也不可复现；batch-invariant kernel 固定 order | 命题 6（非结合）；引擎模式表 | §6、3.4 |
| 增量 detokenization 须缓冲 UTF-8 跨 chunk 字节 + 增量 KMP 检测 stop（O(n) 而非 O(n²m)） | §7；DeepSeek Recipe | §7、7.4 |

## 练习

1.（计算 · [10]）对 $p = [0.4, 0.25, 0.15, 0.12, 0.08]$：(a) 用式 (2) 求 $p_{\mathrm{th}} = 0.8$ 的 nucleus 并用式 (3) 再归一化；(b) 用式 (4) 求 $\eta = 0.3$ 的 min-p 核并再归一化；(c) 说明两核大小为何不同。校验：(a) 核 $\lbrace 1,2,3 \rbrace$、 $p' = [0.5, 0.3125, 0.1875, 0, 0]$；(b) 核 4 个、 $p' \approx [0.435, 0.272, 0.163, 0.130, 0]$，与 §2 一致。
2.（推导 · [20]）完成 §3 命题 3（softmax 平移不变）留作的证明：证 $\mathrm{softmax}(z + c\mathbf{1}) = \mathrm{softmax}(z)$，并由此导出式 (7) 的 log_softmax 稳定形式；再证 $\tau \to 0$ 时式 (1) 退化为 greedy（argmax）。校验：平移不变代入 $z = [1000, 1001, 999]$、 $c = -1001$ 后 softmax 不变（ $[0.245, 0.665, 0.090]$）； $\tau \to 0$ 极限概率集中在最大 logit 位。
3.（计算 · [30]）走完整 processor 链：输入 $z = [2, 1, 0]$（对应 3-token 词表）、 $c = [1, 0, 0]$、 $r = 2$、 $\alpha_{\mathrm{freq}} = 1$、 $\alpha_{\mathrm{pres}} = 0$、 $\tau = 1$、top-p $p_{\mathrm{th}} = 0.7$。按 penalty（式 (5) 后式 (6)）→ temperature → top-p 截断 → 再归一化顺序，逐步写出每步 logits / 概率，并验证最终概率和为 1。再交换 repetition 与 frequency 顺序重算，对照 §3 命题 2 说明结果差异。校验：最终概率和为 1（容差 $10^{-6}$）；两序 logits 不同（与 §3 反例同机制）。
4.（代码 · [30]）教学用：实现 Gumbel-max 采样（式 (8)， $\tau = 1$），对 $z = [2, 1, 0]$ 采样 $10^{5}$ 次统计经验频率，与式 (1) 的 softmax 概率对照；再实现增量 KMP 停止串检测，对一段分 chunk 到达的文本流验证其与朴素全量重扫的截断位置一致。校验：经验频率与 softmax 概率 $[0.665, 0.245, 0.090]$ 之差 $< 0.01$（ $10^{5}$ 样本）；增量 KMP 与朴素重扫截断位置逐字符相等。

## 参考

- [Holtzman et al., 2019](../references.md)：The Curious Case of Neural Text Degeneration，arXiv:1904.09751，[A]（key: top-p）——top-p / nucleus sampling 定义（最小集合累积质量 $\ge p$）、截断再归一化（§2）
- [Nguyen et al., 2024](../references.md)：Turning Up the Heat: Min-p Sampling，arXiv:2407.01082，[A]（key: min-p）——min-p 相对峰值阈值（§2）
- [Willard & Louf, 2023](../references.md)：Efficient Guided Generation for Large Language Models（Outlines），arXiv:2307.09702，[A]（key: outlines）——regex → FSM、预计算状态 → token 集 index、O(1) mask 查表（§5）
- [Dong et al., 2024](../references.md)：XGrammar，arXiv:2411.15100，[A]（key: xgrammar）——CFG → PDA、自适应 token mask 扩展（上下文无关预计算 + 上下文相关运行时）（§5）
- [guidance repo](../references.md)：guidance-ai/guidance，教学参照（key: guidance）——程序化 token 级约束（§5）
- [He, 2025](../references.md)：Defeating Nondeterminism in LLM Inference（Thinking Machines 博客，2025-09-10），[C] 开放技术博客、非闭源（key: batch-invariance）——浮点非结合 "the original sin"、batch 组成改变 reduction order、batch-invariant RMSNorm/matmul/attention（§6）
- [Kwon et al., 2023](../references.md)：PagedAttention / vLLM，arXiv:2309.06180，[A]（key: vllm）——beam / parallel sampling 的 KV 块共享与 COW（§4.4 原文，§4）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——sampling 实现 `vllm/v1/worker/gpu/sample/sampler.py` L42（`v1/sample/` 只剩 ops/，2026-11-23 main 核实）、batch-invariant `model_executor/determinism/{batch_invariant,batch_invariant_configs}.py`（§3 / §6）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——sampling `srt/layers/sampler.py`、batch-invariant `srt/batch_invariant_ops/batch_invariant_ops.py`（§3 / §6）
- [DeepInfer repo](../references.md)：deepseek-ai/DeepInfer，教学参照（key: deepinfer-repo）——`DEEPINFER_BATCH_INVARIANT` 环境变量（`deepinfer/model_executor/models/v4_modules/attn.py` 消费）（§6）
- [DeepInfer doc](../references.md)：doc/inference_deployment_qa.md，教学参照 + 官方披露（key: deepinfer-doc）——DeepSeek Recipe encoding/decoding 与 special-token 注入防护（L35，§7）
- [deepseek-recipe repo](../references.md)：deepseek-ai/deepseek-recipe，教学参照（key: deepseek-recipe）——Rust 多协议、流式状态机 + 增量 KMP、分段并行 BPE、special-token 注入防护（§7）
