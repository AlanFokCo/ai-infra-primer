# 10.2 语音识别（Automatic Speech Recognition）

> 本章回答一件事：输入是 100 fps 的声学帧序列、输出是约 10 fps 量级的文本标签、两者对齐事先未知时， $P(y \mid x)$ 怎么定义、训练、解码。三条结构路线各给一个完整答案——CTC 把对齐枚举为路径集合并精确边缘化（本章给 forward-backward 全推导与可复算数例）、RNN-T 把输出依赖塞进同一格点、AED 用 cross-attention 隐式对齐换最强输出先验；随后走读 Whisper 的 AED 形态与多任务前缀、流式 chunk attention 的 lookahead 账、2024–2026 开放系统（SenseVoice / FireRedASR / Qwen3-ASR），以及 LLM-based ASR 与 8.1 connector 的同构 | 前置：1.1（NLL 与 teacher forcing）、2.3（attention）、3.2 §7（encoder-decoder 与 cross-attention，式 (11)–(13)）、8.1（encoder → connector → LLM 组件链，式 (2)）、10.1（log-mel 特征与帧率账） | 锚点：openai/whisper [A]、espnet/espnet [A]（CTC/RNN-T 标准第三方实现）、QwenAudio/SenseVoice [A]；CTC 原文（Graves et al., ICML 2006，venue+DOI）[B]、RNN-T [B]、Whisper [A]、SenseVoice [A]、FireRedASR [A]、Qwen3-ASR [A]

## 1. 形式化：长度不匹配是 ASR 的第一约束

声学特征序列 $x = (x_1, \ldots, x_T)$： $T$ 帧 log-mel，按 10.1 的特征账（16 kHz、25 ms 窗、10 ms hop）帧率为 100 fps。标签序列 $y = (y_1, \ldots, y_U)$：字符或子词级文本，按正常语速折算约 10 fps 量级（教学账口径）。ASR 要建模的条件分布写作

$$
P(y \mid x), \qquad U \ll T \qquad (1)
$$

**符号**： $x_t \in \mathbb{R}^{80}$：第 $t$ 帧 log-mel 向量（80 通道取 Whisper 档，§6.1）； $y_u \in \mathcal{V}$：第 $u$ 个标签， $\mathcal{V}$ 为标签词表； $T$、 $U$：两条序列的长度，训练数据只给 $(x, y)$ 对、不给帧与标签的对应关系。

困难不在「条件」而在「对齐」：1 秒语音约 100 帧对约 10 个标签，帧到标签的映射是未知的多对一单调映射。监督信号里缺失的这个中间变量——对齐（alignment）——决定了一个 ASR 模型的结构长什么样。本章的三条路线正是处理缺失对齐的三种方式：

1. **CTC**：把对齐显式建模为路径 $\pi$，对全部合法路径精确求和（边缘化），用动态规划（dynamic programming，DP）在 $O(TU)$ 内完成求和（§2）。
2. **RNN-T**：同样显式边缘化，但把「已输出哪些标签」作为状态放进发射概率，解除 CTC 的帧间条件独立（§4）。
3. **AED**（attention-based encoder-decoder）：不显式建模对齐，让 cross-attention 学出软对齐，输出端用自回归分解 $P(y \mid x) = \prod_u P(y_u \mid y_{\lt u}, x)$——3.2 §7 的 enc-dec 条件生成原样套用（§6）。

还有一笔所有路线都逃不掉的账： $T / U \approx 10$ 的速率差意味着九成声学帧「没有对应标签」——它们要么被显式吸收（CTC/RNN-T 的 blank 帧），要么被 attention 软聚合（AED），要么在前端被下采样（LLM-based ASR 的 projector，§9）。对照 3.2 §7 的读法：ASR 把「输入固定、输出变化」推到极致——输入连续、输出离散、长度对齐事先未知——enc-dec 仍是标准答案之一，但 CTC/RNN-T 给出了不堆 decoder 栈的另外两种解，且这两种解天然流式（§7）。

## 2. CTC：对全部对齐路径做精确边缘化

连接时序分类（connectionist temporal classification，CTC；[Graves et al., 2006](../references.md)，ICML 2006, pp. 369–376，DOI 10.1145/1143844.1143891，**无 arXiv**，[B]——无官方代码，ESPnet 等为第三方标准实现）的路线是：不放弃逐帧输出的廉价结构，改为把「哪个帧对应哪个标签」当作潜变量，对全部合法对齐精确边缘化。本节给出完整推导链：假设（§2.1）→ 路径集合（§2.2）→ 状态行走规则（§2.3）→ 前向/后向递推（§2.4/§2.5）→ 行为分析（§2.6）。

### 2.1 帧级发射与条件独立假设

encoder（双向 RNN 或 Transformer 皆可，结构任选）把 $x$ 编成帧级隐状态，每帧一个 softmax，输出定义在 $\mathcal{V} \cup \lbrace\varnothing\rbrace$ 上：

$$
P_t(k) = P(\pi_t = k \mid x) = \mathrm{softmax}(z_t)_k, \qquad k \in \mathcal{V} \cup \lbrace\varnothing\rbrace \qquad (2)
$$

**符号**： $\varnothing$：blank 符（语义见 §2.6）； $z_t \in \mathbb{R}^{|\mathcal{V}|+1}$：第 $t$ 帧的 logit； $\pi_t$：第 $t$ 帧实际发射的符号。

CTC 的核心假设是**条件独立**：给定整条声学序列，各帧的发射相互独立，

$$
P(\pi \mid x) = \prod_{t=1}^{T} P_t(\pi_t) \qquad (3)
$$

**符号**： $\pi \in (\mathcal{V} \cup \lbrace\varnothing\rbrace)^T$：长度为 $T$ 的对齐路径（alignment path）。

注意独立的是「发射」而非「表征」—— $P_t$ 由双向 encoder 计算、可以看全 $x$；被砍掉的是标签之间的依赖通道：给定 $x$，第 5 帧输出什么与第 9 帧输出什么无关。这个假设换来 §2.4 的可分解动态规划，代价在 §2.6 结账。

### 2.2 坍缩映射与路径集合

路径 $\pi$ 与转写 $y$ 之间是多对一映射。定义坍缩映射（collapse map） $\mathcal{B}$：先把相邻重复合并为一个，再删去全部 blank：

$$
\mathcal{B}(a\ a\ \varnothing\ b\ b\ b) = ab, \qquad \mathcal{B}(a\ \varnothing\ a\ b) = aab \qquad (4)
$$

第二例是 blank 的边界语义：被 blank 隔开的两个 $a$ 不合并——blank 是「让同一标签连续出现两次」得以表示的唯一机制（§2.6 展开）。合法路径集合是反像

$$
\mathcal{B}^{-1}(y) = \lbrace \pi : \mathcal{B}(\pi) = y \rbrace \qquad (5)
$$

CTC 把转写概率定义为全部合法路径的概率之和（边缘化，marginalization）：

$$
P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P_t(\pi_t) \qquad (6)
$$

**符号**：求和遍历所有坍缩到 $y$ 的路径；每条路径的概率由式 (3) 给出。

直接枚举不可行：即使只数「无相邻重复」的路径，也相当于从 $T$ 帧中挑出承载 $U$ 个标签的子集，量级 $\binom{T}{U}$； $T = 3000$（30 秒窗）、 $U = 100$ 时是天文数字，再计入重复帧与 blank 的摆放只会更多。式 (6) 的求和之所以能在 $O(TU)$ 内精确完成，靠的是路径集合的马尔可夫结构——合法路径之间的差异只由「相邻两帧占据的位置」决定，这正是动态规划的用武之地，也是 §2.3–2.4 的推导目标。

### 2.3 扩展标签序列与状态行走规则

直接对「已输出标签数」做 DP 有一个麻烦：连续重复标签（如 $y = aa$）要求路径中两个 $a$ 之间至少隔一个 blank，转移规则必须区分「重复当前标签」与「前进到新标签」。Graves 2006 的处理是把 blank 预先插进标签序列，得到扩展标签序列（extended label sequence）

$$
y' = (\varnothing, y_1, \varnothing, y_2, \varnothing, \ldots, \varnothing, y_U, \varnothing), \qquad S = 2U + 1 \qquad (7)
$$

**符号**： $`y'_s`$：扩展序列第 $s$ 位——奇数位恒为 $\varnothing$，偶数位依次为 $y_1, \ldots, y_U$； $S$：扩展序列长度。

每条合法路径 $\pi \in \mathcal{B}^{-1}(y)$ 一一对应于 $y'$ 位置网格上的一条单调行走：第 $t$ 帧占据某个位置 $s_t$（即 $`\pi_t = y'_{s_t}`$），且 $s_{t+1} - s_t \in \lbrace0, 1, 2\rbrace$。三种步进的语义：

- **停留**（ $s \to s$）：再发一次当前符号。停在 blank 位无害；停在标签位则发出重复帧，被 $\mathcal{B}$ 合并——「一个标签占多个连续帧」正是这样实现的。
- **走一格**（ $s \to s+1$）：前进到下一位置，永远合法。
- **走两格**（ $s \to s+2$）：跳过 $`y'_{s+1}`$，**仅当 $`y'_{s+2} \ne y'_s`$ 时合法**。

两格步进的合法性条件是整套 DP 里唯一需要论证的规则，分两种情形：

(i) $s$ 为奇（ $`y'_s = \varnothing`$）：此时 $`y'_{s+2} = \varnothing = y'_s`$，条件不满足、步进被禁——这是对的： $s$ 与 $s+2$ 同为 blank 位，跳过的是标签位 $`y'_{s+1}`$，若放行，该标签在路径中永无发出的机会，坍缩结果不再是 $y$。

(ii) $s$ 为偶（ $`y'_s`$ 为标签）：跳过的是 blank 位 $`y'_{s+1}`$。若 $`y'_{s+2} = y'_s`$（重复标签，如 $aa$ 中的两个 $a$），跳格让两个相同标签在路径中相邻、被 $\mathcal{B}$ 合并为一个——路径坍缩结果不再是 $y$，故禁止；此时必须经由中间 blank 位（先发 $\varnothing$ 阻断合并）。若 $`y'_{s+2} \ne y'_s`$，两标签本来就不合并，跳过 blank 位只是少发一个可省的 $\varnothing$，合法。

行走的起点与终点各有一格自由度：路径可以以若干 blank 开头（从 $s = 1$ 出发），也可以直接以首个标签开头（从 $s = 2$ 出发）；同理终于 $s = S$（收尾 blank）或 $s = S - 1$（末标签）。这两组端点条件将在式 (9) 与式 (11) 落实为初始化与终止求和。

```
CTC 状态网格与三类步进（y = ab → y' = (∅, a, ∅, b, ∅)，S = 5；横轴 t = 1…T，纵轴 s）

s=5 ∅ ·─·─·─·─·─·─●   ← 终点候选 s = S（收尾 blank）
s=4 b ·─·─·─●══●─·─●   ← 终点候选 s = S−1（末标签）
s=3 ∅ ·─·─●═╝  ↗  ║·        ═：停留 s→s（重复帧，经合并被吸收）
s=2 a ·─●═╝  ↗   ║  ·      ↗：走一格 s→s+1（恒合法）
s=1 ∅ ●═╝  ↗     ║  ·      ║：走两格 s→s+2（仅当 y'_{s+2} ≠ y'_s 开放）
       t=1 t=2  t=3 …  t=T

节点 (t,s) 权重 = P_t(y'_s)；路径概率 = 途经节点权重连乘；
t=1 时起点 s ∈ {1, 2}，t = T 时终点 s ∈ {S−1, S}（式 (9)(11)）
```

由此还得到一个可检查的长度约束：每帧至多走两格，配合起点 $s = 2$ 与终点 $s = S - 1$ 的自由度（式 (9)(11)），无重复标签时 $T \ge U$ 即可走通（ $2T \ge 2U$）；紧约束来自重复标签—— $y$ 中每处连续重复都强制消耗至少一帧 blank（重复标签间必须走经 blank 位），故 CTC 可表示 $y$ 的紧约束是 $T \ge U + |\lbrace i : y_i = y_{i-1}\rbrace|$（无重复时退化为 $T \ge U$）。标签比帧还多的输入，CTC 给不出非零概率（RNN-T 无此约束，§4.3）。

### 2.4 前向变量 $\alpha$：定义、递推与终止

定义前向变量（forward variable）为「第 $t$ 帧占据位置 $s$、且路径前缀坍缩到对应标签前缀」的全部前缀概率之和：

$$
\alpha_t(s) = \sum_{\pi_{1:t}:\ \pi_t = y'_s,\ \mathcal{B}(\pi_{1:t}) = y_{1:u(s)}} \prod_{t'=1}^{t} P_{t'}(\pi_{t'}) \qquad (8)
$$

**符号**： $u(s)$：位置 $s$ 已覆盖的标签数—— $s$ 为偶时 $u(s) = s/2$， $s$ 为奇时 $u(s) = (s-1)/2$；求和遍历所有满足两个条件的 length-$`t`$ 前缀。

**初始化**（ $t = 1$，落实起点自由度）：

$$
\alpha_1(1) = P_1(\varnothing), \qquad \alpha_1(2) = P_1(y_1), \qquad \alpha_1(s) = 0 \ \ (s > 2) \qquad (9)
$$

**递推**（ $t \ge 2$）：到达 $(t, s)$ 的上一步只能来自 $s$（停留）、 $s-1$（走一格）、 $s-2$（走两格，附 §2.3 的合法性条件）——三类来源互斥且穷尽，故直接求和：

$$
\alpha_t(s) = P_t(y'_s) \begin{cases} \alpha_{t-1}(s) + \alpha_{t-1}(s-1), & y'_s = \varnothing \ \text{or}\ y'_{s-2} = y'_s \\ \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2), & \text{otherwise} \end{cases} \qquad (10)
$$

第一行对应 §2.3 的两类禁跳情形： $s$ 为奇时 $`y'_s = \varnothing`$（跳入将丢标签）； $s$ 为偶且 $`y'_{s-2} = y'_s`$ 时为重复标签（跳入将触发非法合并）。注意这里的条件用「到点」 $s$ 写出，与 §2.3 用「出点」 $s-2$ 写出等价： $s$ 偶、 $`y'_{s-2} \ne y'_s`$ 当且仅当 $`y'_s \ne \varnothing`$ 且 $`y'_{s-2} \ne y'_s`$。越界项按 0 计： $\alpha_{t-1}(0)$、 $\alpha_{t-1}(-1)$ 恒为 0（与式 (13) 的越界约定对称）。

**终止**（落实终点自由度）：

$$
P(y \mid x) = \alpha_T(S) + \alpha_T(S - 1) \qquad (11)
$$

复杂度：状态网格 $T \times S$、每格 $O(1)$，合计 $O(TU)$ 时间；只存相邻两行时滚动空间 $O(S)$。另有可达性剪枝： $t$ 帧最多走到 $s = 2t$（每帧至多两格），距终点差 $S - s$ 格至少需要 $\lceil (S-s)/2 \rceil$ 帧——实现只算带内格子。

**可复算数例**： $y = ab$、 $T = 3$，帧概率表

| $t$ | $P_t(a)$ | $P_t(b)$ | $P_t(\varnothing)$ |
| --- | --- | --- | --- |
| 1 | 0.5 | 0.2 | 0.3 |
| 2 | 0.4 | 0.4 | 0.2 |
| 3 | 0.1 | 0.6 | 0.3 |

先枚举： $\mathcal{B}^{-1}(ab)$ 恰含 5 条路径—— $(a,a,b)$、 $(a,b,b)$ 概率各 $0.5 \times 0.4 \times 0.6 = 0.12$， $(a,b,\varnothing)$、 $(a,\varnothing,b)$ 各 $0.06$， $(\varnothing,a,b)$ 为 $0.072$——合计 $0.432$。再跑 DP： $\alpha_1 = (0.3, 0.5, 0, 0, 0)$； $t = 2$ 时 $\alpha_2(2) = 0.4 \times (0.5 + 0.3) = 0.32$（停留 + 走一格）， $\alpha_2(3) = 0.2 \times 0.5 = 0.10$（blank 位禁跳，只收 $s = 2$ 来路）， $\alpha_2(4) = 0.4 \times (0 + 0 + 0.5) = 0.20$（ $`y'_4 = b \ne a = y'_2`$，跳格开放）； $t = 3$ 时 $\alpha_3(4) = 0.6 \times (0.20 + 0.10 + 0.32) = 0.372$， $\alpha_3(5) = 0.3 \times 0.20 = 0.06$。式 (11) 给出 $0.372 + 0.06 = 0.432$——与枚举逐项相符。换任意帧表都可同样复算，这是检验 CTC 实现正确性的最小手段。

### 2.5 后向变量 $\beta$、占有后验与 logit 梯度

后向变量（backward variable） $\beta_t(s)$ 定义为：已知第 $t$ 帧占据位置 $s$，全部「合法后缀」的概率之和。记合法后缀集合为 $`\mathcal{S}_t(s) = \lbrace \pi_{t+1:T} :`$ 存在以 $`y'_s`$ 结尾的合法前缀 $\pi_{1:t}$，使 $\mathcal{B}(\pi_{1:T}) = y \rbrace$，则

$$
\beta_t(s) = \sum_{\pi_{t+1:T} \in \mathcal{S}_t(s)} \prod_{t'=t+1}^{T} P_{t'}(\pi_{t'}) \qquad (12)
$$

**符号**：与前向的定义对偶—— $\alpha$ 含 $`P_t(y'_s)`$、 $\beta$ 不含，后缀从 $t+1$ 计起； $`\mathcal{S}_t(s)`$ 的合法性由 §2.3 的步进规则判定。

终止： $\beta_T(S) = \beta_T(S-1) = 1$（末位置合法、后缀为空），其余 $\beta_T(s) = 0$。递推与前向镜像，把「到达 $s$ 的来源」换成「从 $s$ 出发的去向」，合法性条件不变：

$$
\beta_t(s) = P_{t+1}(y'_s)\thinspace\beta_{t+1}(s) + P_{t+1}(y'_{s+1})\thinspace\beta_{t+1}(s+1) + \delta_s\thinspace P_{t+1}(y'_{s+2})\thinspace\beta_{t+1}(s+2) \qquad (13)
$$

**符号**： $\delta_s \in \lbrace0, 1\rbrace$：两格步进开关， $\delta_s = 1$ 当且仅当 $`y'_{s+2} \ne y'_s`$；越界项（ $s+1, s+2 > S$）按 0 计。

由定义直接乘开： $\alpha_t(s)$ 乘 $\beta_t(s)$ 恰好穷尽所有经过格点 $(t, s)$ 的完整路径（前缀 × 后缀， $`P_t(y'_s)`$ 恰好在 $\alpha$ 侧出现一次），得本章的恒等式校验点：

$$
\alpha_t(s)\thinspace\beta_t(s) = \sum_{\pi \in \mathcal{B}^{-1}(y):\ \pi_t = y'_s} P(\pi \mid x), \qquad P(y \mid x) = \sum_{s=1}^{S} \alpha_t(s)\thinspace\beta_t(s) \quad (\forall t) \qquad (14)
$$

右半式对任意 $t$ 成立—— $t = T$ 时退化回式 (11)。归一化即得**占有后验**（occupancy posterior） $\gamma_t(s) = \alpha_t(s)\thinspace\beta_t(s) / P(y \mid x)$：「第 $t$ 帧对齐到位置 $s$」的后验概率，CTC 学到的隐式对齐全部信息都在这张 $(t, s)$ 网格上。

训练目标是 $-\log P(y \mid x)$。对 logit 求导（练习 2 给完整推导）：先对 $P_t(k)$ 求， $\partial P(y \mid x) / \partial P_t(k)$ 是「经过 $(t, k)$ 的全部路径概率（去掉 $P_t(k)$ 因子）之和」，用式 (14) 改写后过 softmax 链式法则合并同类项，得

$$
\frac{\partial\thinspace(-\log P(y \mid x))}{\partial z_t(k)} = P_t(k) - \sum_{s:\ y'_s = k} \frac{\alpha_t(s)\thinspace\beta_t(s)}{P(y \mid x)} \qquad (15)
$$

**符号**：右端第二项：标签 $k$ 在第 $t$ 帧的占有后验（对 $y'$ 中所有等于 $k$ 的位置求和）。

式 (15) 是「softmax 预测 − 后验占有」的逐帧交叉熵形式：CTC 训练就是在给每帧分配一份软标签，软标签由当前对齐后验给出——EM 的味道由此而来（E 步 = forward-backward 算占有，M 步 = 帧级 CE）。工程注记：式 (8)(12) 都是 $O(T)$ 个概率的连乘， $T$ 上千时直接下溢，实现一律走 log 域（log-sum-exp）或逐帧重标定，ESPnet 等标准实现皆然（[espnet-repo](../references.md)，[A]）。

### 2.6 blank 语义、peaky 行为与弱 LM 的代价

blank $\varnothing$ 的三个语义角色全部由 §2.2–2.3 的数学唯一决定：

1. **吸收多余帧**： $T / U \approx 10$，九成帧必须「什么都不输出」——blank 是它们的去处。
2. **边界分隔**：被 $\varnothing$ 隔开的重复标签不合并（式 (4) 第二例）；没有 blank， $aa$ 这类串在 CTC 框架内无法表示——§2.3 的禁跳规则正是为此存在。
3. **本帧不作标签承诺**：blank 不是静音、也不是词间空格——静音若出现在词内，多半被模型标成某标签的延续帧；blank 只是对齐簿记的中性吸收态。

**peaky 行为**：训练收敛后的 CTC 帧级输出呈「尖峰 + blank 海洋」形态——标签概率集中在少数窄帧段，其余帧几乎全部分配给 $\varnothing$。机制上这是式 (15) 的自增强动力学：训练初期占有后验弥散，正梯度把概率质量摊给多条可行路径；一旦某条对齐取得多数（帧内 softmax 的竞争使这种对称性破缺几乎必然发生），式 (15) 的正项就集中压向该对齐的帧-标签分配，其余路径的占有进一步萎缩——rich-get-richer，直到后验接近 one-hot。两个直接后果：(i) 解码便宜，逐帧贪心 argmax 再过 $\mathcal{B}$ 往往已接近最优，工程上再叠 prefix beam search 与外部 LM；(ii) 校准差——模型只对「自己选中的那条对齐」有把握，跨对齐的不确定性被边缘化吸收掉了。

**弱 LM**：条件独立假设（式 (3)）砍掉了标签-标签通道—— $P(y \mid x)$ 对候选转写的排序只来自声学证据，语法约束与同音消歧在模型内部无从表达。外部 LM 以 shallow fusion 回接：解码时按 $\log P_{\mathrm{CTC}}(y \mid x) + \lambda \log P_{\mathrm{LM}}(y)$ 联合打分。这条外接回路正是 §4 的 RNN-T 与 §6 的 AED 各自内置掉的部件——读懂 CTC 的数学，也就读懂了后续结构为什么都要把输出依赖夺回来。

## 3. CTC + attention hybrid：一份编码器、两份监督

ESPnet 的标准配方在同一 encoder 上并挂两个输出头，联合目标为

$$
L = \lambda\thinspace L_{\mathrm{CTC}} + (1 - \lambda)\thinspace L_{\mathrm{att}} \qquad (16)
$$

**符号**： $L_{\mathrm{CTC}} = -\log P_{\mathrm{CTC}}(y \mid x)$（式 (6)）； $L_{\mathrm{att}}$：AED 分支的 teacher-forcing NLL（3.2 §7）； $\lambda \in [0, 1]$：混合权重， $\lambda = 1$ 退化为纯 CTC、 $\lambda = 0$ 为纯 attention，具体取值逐 recipe 调（[espnet-repo](../references.md)，[A]）。

两个分支的互补是机制级的：CTC 项在 encoder 输出上直接监督单调对齐，给 attention 分支提供 AED 自身没有的对齐先验——纯 AED 的 cross-attention 无单调约束，长输入或噪声下会出现跳读、复读、漏读（对齐漂移），hybrid 用 CTC 把 encoder 表征钉在单调坐标系里；attention 项则把 CTC 缺失的输出依赖补回来——decoder 看得见 $y_{\lt u}$，语言先验由此进入梯度。解码侧同样有联合形式：按 $\lambda \log P_{\mathrm{CTC}} + (1 - \lambda) \log P_{\mathrm{att}}$ 给候选假设打分（ESPnet joint decoding）。流式场景里 CTC 分支还兼职触发与端点检测的角色（§7 展开）。

## 4. RNN-T：把输出依赖放进对齐本身

### 4.1 三件套：transcription / prediction / joint

RNN-T（RNN-Transducer，[Graves, 2012](../references.md)，arXiv:1211.3711，[B]——2012 年论文无官方代码，ESPnet 等为第三方标准实现）由三个网络组成：

- **transcription network**（声学端）： $`f_t = \mathrm{enc}(x)_t`$，与 CTC 的 encoder 同位。
- **prediction network**（标签端）： $g_u = g(y_1, \ldots, y_u)$——消费完第 $u$ 个标签后的状态（ $g_0$ 为空历史的初始状态），只看标签历史、不看声学，一个小型 LM。格点 $(t, u)$ 已发出 $u$ 个标签（对照下方 $\alpha(t, u)$ 的定义），故 $P(k \mid t, u)$ 作为第 $u+1$ 次发射的分布，条件于 $g_u$ 即条件于 $y_1..\thinspace y_u$；式 (18) 中从 $(t, u-1)$ 发出的标签迁移预测 $y_u$，用的是 $g_{u-1} = g(y_1..\thinspace y_{u-1})$，索引自洽。
- **joint network**：两者相加融合后过 softmax：

$$
h(t, u) = \tanh\left( W^{f} f_t + W^{g} g_u + b \right), \qquad P(k \mid t, u) = \mathrm{softmax}\left( W^{o} h(t, u) \right)_k \qquad (17)
$$

**符号**： $k \in \mathcal{V} \cup \lbrace\varnothing\rbrace$； $(t, u)$：帧 $t$、已输出 $u$ 个标签的格点； $W^{f}, W^{g}, W^{o}$：joint 的投影与输出矩阵。

口径注记：式 (17) 取现代标准形（tanh 联合，ESPnet 默认）；2012 原文的联合是逐类直接相加 $\exp(f_t^k + g_u^k)$ 归一，无隐层。

对照式 (2)：发射概率现在依赖 $(t, u)$ 两个坐标—— $u$ 携带标签历史。RNN-T 原文对自身定位的表述是「extends CTC by defining a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output dependencies」——两个扩展点分别对应 §4.2 的格点结构（每帧可发多标签，故输出长度不受 $T$ 限制）与 prediction network（输出-输出依赖）。

### 4.2 二维 lattice 与前向递推

对齐空间从 CTC 的一维行走升级为二维格点（lattice）：点 $(t, u)$， $t \in \lbrace1, \ldots, T\rbrace$、 $u \in \lbrace0, \ldots, U\rbrace$。在每个格点，路径二选一：

- **blank 迁移**： $(t, u) \to (t+1, u)$，消费一帧、不发标签，概率 $P(\varnothing \mid t, u)$——格点上右移。
- **标签迁移**： $(t, u) \to (t, u+1)$，发标签 $y_{u+1}$、留在原帧，概率 $P(y_{u+1} \mid t, u)$——格点上移。

```
RNN-T 二维格点（T 帧 × U 个标签；路径 = (1,0) → (T,U) 的右移/上移交错）

u=U  ·──────●───∅──▶ ●  ← α(T,U)，终点须以 ∅ 迁移收尾（式 (19)）
             ▲
u=2  ·──●─∅─▶●          ▲：标签迁移，发 y_{u+1}，留在原帧（同一帧可连续上移多发）
         ▲  ▲
u=1  ·─●─∅─▶●           ∅：blank 迁移，消费一帧（右移）
      ▲  ▲  ▲
u=0  ●──∅─▶●──∅──▶ ●
     t=1  t=2  t=3  …  T

节点 (t,u) 的发射分布 = P(k|t,u)（式 (17)）；路径概率 = 途经迁移概率连乘
```

前向变量 $\alpha(t, u)$ 的定义引原文（§2.4）：「the probability of outputting $y_{[1:u]}$ during $f_{[1:t]}$」——在 $f_{[1:t]}$ 期间输出 $y_{[1:u]}$ 的概率。进入 $(t, u)$ 的路径只有两类：从 $(t-1, u)$ 经 blank 迁移来，或从 $(t, u-1)$ 经标签迁移来；两类互斥且穷尽，得递推

$$
\alpha(t, u) = \alpha(t-1, u)\thinspace P(\varnothing \mid t-1, u) + \alpha(t, u-1)\thinspace P(y_u \mid t, u-1) \qquad (18)
$$

**初始化**： $\alpha(1, 0) = 1$；首行 $\alpha(t, 0)$ 只能沿 blank 链累积，首列 $\alpha(1, u)$ 只能沿标签链累积。**终止**：路径最后必须离开最后一帧，

$$
P(y \mid x) = \alpha(T, U)\thinspace P(\varnothing \mid T, U) \qquad (19)
$$

式 (18)(19) 的推导链与 CTC 同构但更简单：没有坍缩合并语义（blank 只是格点上的右移），没有重复标签特例（同一帧可连续上移发任意标签）——复杂性从「转移规则」搬家到了「发射概率依赖 $u$」。后向变量 $\beta(t, u)$ 从终点反向镜像递推， $\alpha \cdot \beta$ 给出格点后验——与 §2.5 同构，留作练习 3。

### 4.3 与 CTC 的结构性差异

两家并排放，差异逐条落实：

1. **输出依赖**：CTC 的 $P_t$ 不依赖标签历史（式 (3)）；RNN-T 的 $P(k \mid t, u)$ 经 $g_u$ 依赖全部 $y_{\lt u+1}$。边缘化仍然是精确 DP——依赖被格点状态 $(t, u)$ 马尔可夫化，不需要枚举路径。
2. **长度约束**：标签迁移不消费帧，一帧可发任意多标签——RNN-T 对任意长度的输出序列都有定义（原文「output sequences of all lengths」），CTC 的 $T \ge U + r$ 约束消失。
3. **对齐语义**：CTC 路径有合并与 blank 边界规则（§2.2–2.3）；RNN-T 路径是右移/上移的任意交错，无合并语义，对齐与路径一一对应、更直白。
4. **成本账**：CTC 每帧一次 softmax，共 $O(T)$ 次 $|\mathcal{V}|+1$ 维；RNN-T 每个格点一次 joint softmax，共 $O(TU)$ 次，logit 张量本身是 $[T, U, |\mathcal{V}|+1]$—— $W^{f} f_t$ 与 $W^{g} g_u$ 虽可分别预计算，逐格点的加和与非线性省不掉； $T, U$ 数千、词表数千时，这块张量的显存是 transducer 训练的著名税。

若把 prediction network 拿掉（ $g_u$ 为常数）并限制每帧至多一发标签，RNN-T 退化为接近 CTC 的条件独立形式，差异只剩重复标签的合并语义——CTC 可以看作「发射无状态」的 transducer 特例（练习 3(c)）。

## 5. CTC / RNN-T / AED 对照

| 维度 | CTC（§2） | RNN-T（§4） | AED（3.2 §7；§6） |
| --- | --- | --- | --- |
| 对齐 | 显式路径集合，精确边缘化 | 显式二维格点，精确边缘化 | 隐式 soft alignment（cross-attention 权重），无单调约束 |
| 输出依赖 | 无（条件独立，式 (3)） | prediction network（局部 LM） | 自回归 decoder（完整 LM） |
| 长度约束 | $T \ge U + r$（ $r$ = 重复处数） | 无（一帧可发多标签） | 无 |
| 训练目标 | $-\log \sum_{\pi} P(\pi \mid x)$，一维 DP | $-\log P(y \mid x)$，二维格点 DP（式 (18)(19)） | teacher-forcing NLL |
| 训练期发射成本 | $O(T)$ 次帧级 softmax | $O(TU)$ 次 joint softmax | $U$ 步 decoder 前向 + $O(UT)$ cross-attn |
| 流式 | 天然（逐帧发射，encoder 因果化即可） | 天然（同上） | 需 chunk 改造（§7） |
| LM 回路 | 外部 shallow fusion | 内置 prediction net（实践中仍常外挂） | decoder 自身即 LM |
| 精度定性 | 单解码偏弱，靠外挂 LM 补齐 | 介于两者之间 | 上限最高，但依赖数据规模（§6.3） |
| 代表锚点 | Graves 2006 [B]；ESPnet [A] | Graves 2012 [B]；ESPnet [A] | Whisper [A]；FireRedASR-AED [A] |

一句话读表：CTC 用「无输出依赖」换可训练性与流式，RNN-T 用「 $O(TU)$ joint」换回输出依赖并保住流式，AED 用「非单调隐式对齐 + 自回归解码」换最强语言先验——§3 的 hybrid 与 §9 的 LLM-based ASR 是这三个端点之间的两次折中。

## 6. Whisper 走读：AED 路线的弱监督极限

Whisper（[Radford et al., 2022](../references.md)，arXiv:2212.04356，[A]；官方 repo [openai/whisper](../references.md)，[A]，whisper/model.py 文件级实测）是 AED 路线「结构不变、数据换鲁棒」的标本：骨架是未改动的 encoder-decoder Transformer（3.2 §7 的式 (11)–(13) 原样成立，3.2 已用它做 enc-dec 形态的活例），全部新意在前端约定、多任务前缀与数据规模。

### 6.1 前端与 30 秒窗 padding 账

前端是 10.1 特征账的直接实例：全部音频重采样到 16 kHz，提取 80 通道 log-magnitude mel（25 ms 窗、10 ms hop → 100 fps），再按 30 秒定长切段：

$$
30\ \mathrm{s} \times 100\ \mathrm{fps} = 3000\ \text{mel frames} \hspace{5mu}\to\hspace{5mu} \text{conv stem (stride 2)} \hspace{5mu}\to\hspace{5mu} 1500\ \text{encoder positions} \qquad (20)
$$

large-v3 档的 HF config 实测： $d_{\mathrm{model}} = 1280$、decoder 32 层、20 头（openai/whisper-large-v3 config.json，2026-11-13 实测）。定长窗的账两笔。**收益**：batch 形状恒定——3000 帧进、1500 位置出，padding/packing 的工程复杂度归零，68 万小时弱监督数据可以任意乱序成批。**代价**： $L$ 秒的短音频零填充到 30 秒，encoder 计算量中 $(30 - L)/30$ 的比例打在填充帧上，decoder 的 cross-attention K/V 恒为 1500 行同理；长于 30 秒的音频按段处理、解码侧拼接，长录音在切缝处的重复与漏字是这一路线的已知工程问题。

### 6.2 special tokens：多任务前缀控制

Whisper 的多任务训练格式用一组 special tokens「充当任务指定符或分类目标」（原文语）。机制是 decoder 前缀条件化：

$$
P(y \mid x, c) = \prod_{u} P\left( y_u \mid y_{\lt u},\ c,\ H^{\mathrm{enc}}(x) \right) \qquad (21)
$$

**符号**： $c$：前缀 token 序列——start-of-transcript、语言 token、任务 token（转写/翻译）、时间戳开关等； $H^{\mathrm{enc}}(x)$：encoder 输出，cross-attention 的 K/V 来源（3.2 §7 式 (12)(13)，K/V 一次计算全程复用）。

与 T5 式文本任务前缀同一原理（3.2），差别在两类 token 的双重身份：语言 token 本身是监督目标——模型必须从声学证据里猜出语种，其输出概率因此可读作语言识别分类器；no-speech token 同理充当 VAD 信号，官方解码逻辑用 $P(\text{no-speech}) > 0.6$ 且平均 logprob $< -1$ 联合判跳（原文阈值）。「任务控制」与「分类头」被压进同一套词表，这是 AED 形态特有的便宜：decoder 词表本来就是模型的全部输出接口，不需要外接任何判别头。

### 6.3 680k 小时弱监督：鲁棒性的来源

论文的规模主张一句话：「scaled to 680,000 hours of multilingual and multitask supervision」（数据清洗管线与配方属 training 维度，此处只取结论）。机制层面的鲁棒性论证有三层：(i) **覆盖换不变性**——架构上没有任何抗噪/抗口音设计，鲁棒性来自训练分布对录音条件、说话人、语种的覆盖广度；(ii) **多任务共享表征**——转写/翻译/语种识别共用一个 encoder（§6.2 的前缀只在 decoder 侧注入），任务间互相正则；(iii) **零样本协议倒逼**——不做逐数据集微调，评估即分布外泛化，数据规模是被这个协议「逼」出来的。结构侧的对照也值得记一笔：Whisper 证明 vanilla AED 在足够数据下不需要 CTC 的对齐监督也能学到稳定对齐；§3 的 hybrid 存在的意义因此主要在数据不充分的中小规模场景。

## 7. 流式识别：chunk attention 与 lookahead 账

CTC/RNN-T 的发射是逐帧/逐格点的，encoder 一旦因果化（只看左侧上下文）就天然流式：发射延迟 = 一帧 + encoder 右上下文。AED 有两处非流式：双向 encoder 看全句、cross-attention 覆盖全部 1500 个位置。流式改造的标准件是 chunk attention：帧按块长 $C$ 分组，第 $t$ 帧的 query 只可见键 $j \le \lceil t / C \rceil\thinspace C + L$（ $L$ 为 lookahead 帧数），写成 2.4 的加性 mask 口径（ $M_{tj} = 0$ 为可见、 $-\infty$ 为屏蔽，加在 attention logit 上）：

$$
M_{tj} = \begin{cases} 0, & j \le \lceil t / C \rceil\thinspace C + L \\ -\infty, & \text{otherwise} \end{cases} \qquad (22)
$$

延迟账（教学估算，hop = 10 ms）：块尾帧（ $t = cC$）的 encoder 输出只等 $L$ 帧右上下文，算法延迟下界 $\approx L \times 10\ \mathrm{ms}$；块首帧还要等本块填满，上界包络 $\approx (L + C) \times 10\ \mathrm{ms}$——逐帧延迟落在 $[L,\ L + C] \times 10\ \mathrm{ms}$ 区间内； $L$ 每加一帧换 10 ms 延迟，精度-延迟在这条直线上选点。Whisper 形态的 30 秒定长窗与流式不兼容——窗本身就是 30 秒的右上下文——故流式系统要么回到 CTC/RNN-T 主干，要么用 §3 的 hybrid：CTC 分支出增量结果、attention 分支周期性修正，是工业界常见组合（通用配方级陈述，不绑定具体系统数字）。

## 8. 2024–2026 开放系统三例

**SenseVoice**（[An et al., 2024](../references.md)，arXiv:2407.04051，[A]；repo [QwenAudio/SenseVoice](../references.md)，[A]——FunAudioLLM/SenseVoice 已 301 重定向至此）：非自回归（non-autoregressive）端到端架构，单模型同时输出多语转写、情感识别与音频事件检测；论文声明推理速度超 Whisper-small 5 倍以上、超 Whisper-large 15 倍以上。机制注记：非 AR 解码把「 $U$ 步串行 decoder」从延迟方程里整个删掉，识别延迟与转写长度解耦——与 §6 的 AED 形态恰好构成自回归税的正反例；其多任务侧与 Whisper 的前缀路线不同：非 AR 架构下没有逐 token 解码器，情感与事件标签和转写并列输出，而非进前缀。

**FireRedASR**（[Xu et al., 2025](../references.md)，arXiv:2501.14350，[A]；repo FireRedTeam/FireRedASR，[A]）：工业级中文 ASR 开源双型号——FireRedASR-AED 是本章 §5 右列的完整实例（attention-based encoder-decoder），论文同时定位其「serve as an effective speech representation module in LLM-based speech models」；FireRedASR-LLM 即 encoder + adapter + LLM，是 §9 结构的直接产品化。一家的两个型号恰好覆盖本章 AED 与 LLM-based 两个端点。

**Qwen3-ASR**（[Shi et al., 2026](../references.md)，arXiv:2601.21337，[A]）：开源家族含 Qwen3-ASR-1.7B / 0.6B 与 Qwen3-ForcedAligner-0.6B（非自回归强制对齐模型——把本章 §2 当作潜变量的「对齐」重新变成显式产品输出），覆盖 52 种语言与方言；结构基于 AuT 音频 encoder + Qwen3 LM（Qwen3-Omni 基座），即 §9 的 LLM-based 形态。

## 9. LLM-based ASR：与 8.1 connector 的同构

LLM-based ASR 的组件链——audio encoder → projector → LLM——与 8.1 的 vision encoder → connector → LLM 逐槽同构（8.1 式 (2)）：encoder 出帧级特征 $f(x) \in \mathbb{R}^{T' \times d_a}$，projector 做宽度转换 $d_a \to d_{\mathrm{llm}}$ 并通常附带速率转换（ $k$ 帧拼接为一个软 token），产出 $v \in \mathbb{R}^{M \times d_{\mathrm{llm}}}$（ $M \le T'$）拼进 LLM 输入嵌入序列，识别变成条件文本生成：

$$
P(y \mid x) = \prod_{u=1}^{U} P\left( y_u \mid y_{\lt u},\ v_{1:M} \right) \qquad (23)
$$

**符号**： $v_{1:M}$：audio token 段，占据 LLM 输入序列的一段连续位置，与文本 prompt 混排。

8.1 的两条读法原样搬运。其一，**connector 学空间对齐而非内容补全**：声学内容必须在 encoder 里备好，projector 只做坐标变换——FireRedASR 把 AED 款直接当作 LLM 款的表征模块，正是这个赌注的产品形态；§6.3 的 68 万小时则是把内容备齐的数据形态。其二，**长度不匹配以软形式回归**：§1 的 100 fps vs 10 fps 问题在 CTC 里由 blank 吸收、在 AED 里由 attention 聚合，在这里由 projector 的帧堆叠硬压缩——压缩比 $k$ 选多少，就是新一轮「token 预算 vs 信息损失」的权衡（10.4 对离散 token vs 连续特征进 LLM 有完整对照）。收益端：LLM 提供本章四种形态里最强的输出先验——标点、断句、领域术语、语码转换全部交给语言模型的文本先验，CTC 需要外挂 shallow fusion 才能得到的东西在这里是内置默认值。代价端：LLM 的自回归解码把 §8 SenseVoice 行里被删掉的延迟税又加了回来，且对齐完全隐式——Qwen3-ForcedAligner 作为独立产品存在，恰好说明 LLM-based 形态不免费给出对齐。这条链在 10.4 会扩成完整的 speech LLM（理解 + 生成 + 全双工），本章只取「识别」这一读法。

## 总结

| 对象 | 公式 / 结论 | 位置 |
| --- | --- | --- |
| ASR 形式化 | $P(y \mid x)$， $U \ll T$（100 fps vs ~10 fps，教学账） | 式 (1) |
| 条件独立假设 | $P(\pi \mid x) = \prod_t P_t(\pi_t)$ | 式 (3) |
| 坍缩映射 | $\mathcal{B}$：合并相邻重复、删 blank； $\varnothing$ 阻断合并 | 式 (4) |
| CTC 边缘化 | $P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi \mid x)$；枚举 $\sim \binom{T}{U}$ 不可行 | 式 (6) |
| 扩展序列与步进 | $y'$ 长 $2U+1$；两格步进 $\Leftrightarrow$ $`y'_{s+2} \ne y'_s`$；长度约束 $T \ge U + r$ | 式 (7)，§2.3 |
| 前向 / 后向递推 | $\alpha$：式 (8)–(11)； $\beta$：式 (12)(13)；恒等式 $\sum_s \alpha_t(s)\beta_t(s) = P(y \mid x)$（ $\forall t$） | 式 (14) |
| logit 梯度 | $P_t(k) - \gamma_t(k)$：softmax 预测 − 占有后验 | 式 (15) |
| peaky / 弱 LM | 式 (15) 自增强 → 尖峰 + blank 海洋；条件独立 → shallow fusion 外挂 | §2.6 |
| CTC+att hybrid | $L = \lambda L_{\mathrm{CTC}} + (1 - \lambda) L_{\mathrm{att}}$（ESPnet） | 式 (16) |
| RNN-T | $P(k \mid t, u)$ 式 (17)；格点递推式 (18)(19)； $O(TU)$ joint 税 | §4 |
| 三家对照 | 对齐 / 输出依赖 / 长度 / 成本 / 流式 / LM 回路 / 精度定性 | §5 表 |
| Whisper | 3000 帧 → 1500 位置（式 (20)）；前缀控制（式 (21)）；680k hrs 弱监督 | §6 |
| 流式 | chunk attention（式 (22)，加性 mask）；逐帧延迟 $\approx L \times 10$ ms（块尾下界）至 $(L + C) \times 10$ ms（块首上界），教学估算 | §7 |
| LLM-based ASR | encoder → projector → LLM（式 (23)），同构 8.1 式 (2) | §9 |

## 练习

1.（推导）后向递推与恒等式。从定义式 (12) 出发：(a) 证明递推式 (13)（提示：按「 $t+1$ 帧占据哪个位置」分三类，合法性条件沿用 §2.3）；(b) 证明 $\sum_{s=1}^{S} \alpha_t(s)\thinspace\beta_t(s) = P(y \mid x)$ 对任意 $t$ 成立（提示：把经过 $(t, s)$ 的完整路径按前缀 × 后缀分解求和，再对 $s$ 求和穷尽全部路径）；(c) 用 §2.4 数例的帧表算出全部 $\beta_2(s)$，在 $t = 2$ 处数值验证恒等式。

2.（推导）logit 梯度。证明式 (15)：先证 $`\partial P(y \mid x) / \partial P_t(k) = \sum_{s: y'_s = k} \alpha_t(s)\thinspace\beta_t(s) / P_t(k)`$（经过 $(t, k)$ 的路径概率去掉 $P_t(k)$ 因子，用式 (14)），再对 logit $z_t(k)$ 用 softmax 求导恒等式链式展开、合并两项。结论：CTC 训练 = 以占有后验为软目标的逐帧交叉熵。

3.（推导 + 计算）RNN-T 同构练习。(a) 写出 $\beta(t, u)$ 的定义与反向递推，使 $\alpha(t, u)\thinspace\beta(t, u)$ 给出格点后验（提示：终态含式 (19) 的收尾 blank）；(b) $T = 500$、 $U = 50$、 $|\mathcal{V}| = 4000$ 时，数 RNN-T 的 joint softmax 次数与 logit 张量元素数，对照 CTC 的帧级 softmax 次数；(c) 论证「prediction network 输出为常数 + 每帧至多一发标签」时 RNN-T 退化为 CTC 形态，并指出此时仍残留的一处语义差异。

4.（计算）窗口与流式账。(a) 按式 (20)：12 秒语音进 Whisper 的 encoder 计算浪费比例是多少？(b) 若 hop 改为 20 ms（50 fps），30 秒窗的 encoder 位置数与 $T/U$ 比率怎么变？(c) 某流式 AED 取 $C = 16$、 $L = 8$，按 §7 的账分别给出块尾下界与块首上界（ms）。

## 参考

- [Graves et al., 2006](../references.md)：Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks，ICML 2006, pp. 369–376，DOI 10.1145/1143844.1143891（无 arXiv），[B]（key: ctc）——式 (3)(4)(6)(8)–(14) 的原始出处
- [Graves, 2012](../references.md)：Sequence Transduction with Recurrent Neural Networks，arXiv:1211.3711，[B]（key: rnn-t）——式 (17)–(19)、lattice 结构与 forward 变量定义（§2.4 原文定义）；无官方代码，ESPnet 为第三方标准实现
- [Radford et al., 2022](../references.md)：Robust Speech Recognition via Large-Scale Weak Supervision，arXiv:2212.04356，[A]（key: whisper）——式 (20)(21)，680k hrs、16 kHz/80ch log-mel/25 ms/10 ms、30 s 段、special tokens、no-speech 0.6 与 logprob −1 阈值
- [An et al., 2024](../references.md)：FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs，arXiv:2407.04051，[A]（key: sensevoice）——非自回归端到端、5×/15× 速度声明
- [Xu et al., 2025](../references.md)：FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration，arXiv:2501.14350，[A]（key: fireredasr）——AED/LLM 双型号
- [Shi et al., 2026](../references.md)：Qwen3-ASR Technical Report，arXiv:2601.21337，[A]（key: qwen3-asr）——1.7B/0.6B、ForcedAligner-0.6B、52 语言方言、AuT + Qwen3 LM
- [openai/whisper](../references.md)：官方 repo（whisper/model.py 文件级实测），[A]（key: whisper-repo）——conv stem stride 2、VAD 解码逻辑
- [espnet/espnet](../references.md)：CTC/RNN-T 第三方标准实现与 hybrid 配方 $L = \lambda L_{\mathrm{CTC}} + (1-\lambda) L_{\mathrm{att}}$ 出处，[A]（key: espnet-repo）
- [QwenAudio/SenseVoice](../references.md)：官方 repo（FunAudioLLM/SenseVoice 301 重定向链），[A]（key: sensevoice-repo）
