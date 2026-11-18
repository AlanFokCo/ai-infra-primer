# 2.5 Muon 与谱方法（Muon and Spectral Methods）

> 本章把 Muon 拆成三层推导，每层完整给出：谱范数约束下的最速下降，其解是正交化方向 $\mathrm{msign}(G) = U V^{\top}$；Newton-Schulz 五次迭代如何不算 SVD 而逼近它，含系数 $(3.4445, -4.7750, 2.0315)$ 的出处与调参准则；Moonlight 把 Muon 推到 16B 规模的完整更新式（momentum → 正交化 → $0.2 \cdot \sqrt{\max(A, B)}$ 尺度匹配 + weight decay，逐式回原文）。后半章回答四个工程问题：哪些参数给 Muon、哪些留给 AdamW，约 $2\times$ 计算效率 claim 的精确口径，Kimi K2 的 MuonClip 怎么工作，谱假设何时失效 | 前置：2.2（AdamW 更新形式与超参）、2.4 §6（稳定原语 placement 总图）、models 2.3（attention 记号）、models 1.2（FLOPs 账口径） | 锚点：KellerJordan/Muon repo `muon.py`（教学参照+官方实现）、Muon 原始博客（教学参照）、Moonlight [A]（arXiv:2502.16982 + MoonshotAI/Moonlight repo `examples/toy_train.py`）、Kimi K2 [A]（arXiv:2507.20534）

## 1. 矩阵参数与向量参数的几何分野

一个 decoder-only Transformer 的参数清单按形状分成两类。第一类是矩阵：attention 的 Q/K/V/O 投影、FFN 的 up/gate/down、MoE 的 expert 矩阵。它们在前向里充当**算子**： $y = Wx$，把输入空间的向量映到输出空间。第二类是向量与标量：RMSNorm 的 gain、bias、按行走查表的 embedding。它们不充当空间之间的映射。这个二分有几何上的理由：对算子而言，「参数动了一步，模型功能变了多少」的自然度量是诱导范数。

设权重 $W \in \mathbb{R}^{n \times m}$ 收到扰动 $\Delta$。该层输出的扰动满足

$$
\lVert (W + \Delta) x - W x \rVert_2 = \lVert \Delta x \rVert_2 \le \lVert \Delta \rVert_2 \thinspace \lVert x \rVert_2 \qquad (1)
$$

**符号**： $\lVert \Delta \rVert_2 = \max_i \sigma_i(\Delta)$：谱范数（spectral norm），即最大奇异值； $x \in \mathbb{R}^{m}$：该层输入。

式 (1) 的界是紧的：取 $x$ 为 $\Delta$ 最大奇异值对应的右奇异向量即取等。所以 $\lVert \Delta \rVert_2$ 精确等于「单位输入下该层功能的最坏扰动」。对照之下，Frobenius 范数 $\lVert \Delta \rVert_F = \sqrt{\sum_{ij} \Delta_{ij}^2}$ 只量逐元素能量，与功能扰动之间隔着条件数。隐藏层的激活空间（局部）近似欧氏——这一观察由 Moonlight 论文 §2.1 归于 Cesista 的讨论（[Liu et al., 2025](../references.md)，arXiv:2502.16982，[A]）。于是「控制每层功能每步的最坏变化」翻译成数学就是：**对矩阵参数的更新步加谱范数约束**。向量参数没有算子结构，谱框架对它们无定义。embedding 与 lm_head 虽是 2D 张量，但一个是查表、一个是读出，实证上也必须排除在 Muon 之外（§5）。本章的谱方法只覆盖隐藏层矩阵参数这一类对象。

## 2. 谱范数约束下的最速下降 = msign（完整推导）

把「在谱范数预算 $\eta$ 内让 loss 的一阶近似下降最多」写成约束优化问题。当前梯度 $G \in \mathbb{R}^{n \times m}$（满秩情形先证，秩亏情形留练习 1）。求更新方向：

$$
\min_{\Delta \in \mathbb{R}^{n \times m}} \hspace{5mu} \langle G, \Delta \rangle_F \qquad \text{s.t.} \qquad \lVert \Delta \rVert_2 \le \eta \qquad (2)
$$

**符号**： $\langle G, \Delta \rangle_F = \mathrm{tr}(G^{\top} \Delta)$：Frobenius 内积，即 loss 一阶近似的改变量； $\eta > 0$：谱范数预算（每步功能扰动上界）。

**第一步：SVD 换元把约束与目标同时对角化。** 取 $G$ 的完整 SVD： $G = U \Sigma V^{\top}$， $U \in \mathbb{R}^{n \times n}$、 $V \in \mathbb{R}^{m \times m}$ 正交， $\Sigma \in \mathbb{R}^{n \times m}$ 对角（奇异值 $\sigma_1 \ge \cdots \ge \sigma_r \gt 0$， $r$ 为秩）。令 $D = U^{\top} \Delta V$。正交变换不改变谱范数： $\lVert D \rVert_2 = \lVert \Delta \rVert_2$。目标改写（迹的循环不变性）：

$$
\langle G, \Delta \rangle_F = \mathrm{tr}\left( V \Sigma^{\top} U^{\top} \Delta \right) = \mathrm{tr}\left( \Sigma^{\top} U^{\top} \Delta V \right) = \mathrm{tr}\left( \Sigma^{\top} D \right) = \sum_{i=1}^{\min(n,m)} \sigma_i \thinspace D_{ii} \qquad (3)
$$

**符号**： $D_{ii}$： $D$ 的对角元。注意式 (3) 右端只含 $D$ 的对角元——非对角元不进目标。

**第二步：对角元被谱范数压住。** 对任意矩阵 $D$， $|D_{ii}| = |e_i^{\top} D \thinspace e_i| \le \lVert D \rVert_2 \le \eta$（谱范数定义： $\lVert D \rVert_2 = \sup_{\lVert u \rVert = \lVert v \rVert = 1} u^{\top} D v$）。

**第三步：下界与取等。** 由第一步与第二步：

$$
\langle G, \Delta \rangle_F = \sum_i \sigma_i D_{ii} \ge - \sum_i \sigma_i \thinspace |D_{ii}| \ge - \eta \sum_{i=1}^{r} \sigma_i = - \eta \thinspace \lVert G \rVert_{\ast} \qquad (4)
$$

**符号**： $\lVert G \rVert_{\ast} = \sum_i \sigma_i$：核范数（nuclear norm，奇异值之和）——谱范数的对偶范数；式 (4) 正是「对偶范数配对」的实例：谱约束下的最大下降率等于梯度的对偶范数。

取 $D = -\eta \thinspace I$（ $n = m$ 满秩情形）可使链中两个不等号同时取等： $|D_{ii}| = \eta$ 且 $D_{ii} \lt 0$。换回原文基：

$$
\Delta^{\ast} = -\eta \thinspace U V^{\top} \qquad (5)
$$

唯一性：取等要求 $D_{ii} = -\eta$ 对所有 $i$ 成立，而 $|D_{ii}| = \eta = \lVert D \rVert_2$ 意味着第 $i$ 列 $D e_i$ 的模长达到 $\eta$、全部分量压在第 $i$ 个上，故非对角元全零， $D = -\eta \thinspace I$ 唯一。

**msign 的三个等价形式。** 定义矩阵符号函数（matrix sign function） $\mathrm{msign}(G) = U \thinspace \mathrm{sign}(\Sigma) \thinspace V^{\top}$：满秩时 $\mathrm{sign}(\Sigma) = I$，于是

$$
\mathrm{msign}(G) = U V^{\top} = \left( G G^{\top} \right)^{-1/2} G = G \left( G^{\top} G \right)^{-1/2} \qquad (6)
$$

第二、三式的验证只有一行： $G G^{\top} = U \Sigma^2 U^{\top}$，故 $(G G^{\top})^{-1/2} = U \Sigma^{-1} U^{\top}$，右乘 $G = U \Sigma V^{\top}$ 得 $U V^{\top}$；另一式对称。式 (6) 同时给出几何读法： $U V^{\top}$ 是**离 $G$ 最近的半正交矩阵**—— $\min_{O : O^{\top} O = I} \lVert O - G \rVert_F$ 的解（极分解的标准结论；Muon 博客以此定义 Ortho(G)，[Jordan, 2024](../references.md)，教学参照）。

把同一问题在 Frobenius 约束下重做一遍作对照： $\min \langle G, \Delta \rangle_F$ s.t. $\lVert \Delta \rVert_F \le \eta$，由 Cauchy-Schwarz 得 $\Delta^{\ast} = -\eta \thinspace G / \lVert G \rVert_F$——普通梯度下降。两种几何的答案并排：

| 约束 | 最速下降方向 | 每步下降量 | 对应优化器 |
| --- | --- | --- | --- |
| $\lVert \Delta \rVert_F \le \eta$ | $G / \lVert G \rVert_F$（保留方向与相对幅度） | $\eta \lVert G \rVert_F$ | SGD |
| $\lVert \Delta \rVert_2 \le \eta$ | $U V^{\top}$（全部奇异值压平到 1） | $\eta \lVert G \rVert_{\ast}$ | Muon 的骨架 |

第二行就是 Muon 的全部动机：谱约束把更新的每个奇异方向拉到同一幅度， $G$ 里幅度悬殊的方向（大奇异值主导、小奇异值近零）被等权对待。Muon 博客的实证观察是：SGD-momentum 与 Adam 产生的更新矩阵条件数很高，更新被少数主方向垄断；正交化「放大了那些幅度小但对学习重要的稀有方向」（原文自标注为 speculation，[Jordan, 2024](../references.md)，教学参照）。Moonlight 的 SVD 熵分析给了旁证：1.2T token checkpoint 上，超过 90% 的权重矩阵在 Muon 下的奇异值分布熵高于 AdamW（[Liu et al., 2025](../references.md)，[A]，附录 F）。把优化器统一看作「某种范数约束下的最速下降」的框架由 Bernstein & Newhouse 提出；Muon 博客与 Moonlight §2.1 均以此定位 Muon（Adam 对应逐坐标自适应的范数族，Muon 精确计算时对应谱范数）。该框架文献未收录进本书 references，此处仅作谱系注记。

## 3. Newton-Schulz 迭代：不算 SVD 的 msign

式 (6) 给出两条计算路径：显式 SVD（对每步每个矩阵太贵），或矩阵逆平方根。Newton-Schulz（NS）迭代是数值线性代数求逆平方根与正交化的经典不动点迭代（谱系 Kovarik 1970、Björck & Bowie 1971、Higham 教材，经 [Jordan, 2024](../references.md) 转述）。Muon 用的是它的五次多项式形式。输入 $G$，先归一化、再迭代：

$$
X_0 = \frac{G}{\lVert G \rVert_F}, \qquad X_{k+1} = a \thinspace X_k + b \left( X_k X_k^{\top} \right) X_k + c \left( X_k X_k^{\top} \right)^2 X_k \qquad (7)
$$

**符号**： $a, b, c$：多项式系数；归一化用 Frobenius 范数而非谱范数—— $\lVert G \rVert_F \ge \lVert G \rVert_2$，除以它保证 $X_0$ 的全部奇异值 $\le 1$（Frobenius 范数是免费可算的，谱范数本身就要迭代求）。

**关键结构：NS 迭代只改奇异值、不动奇异向量。** 设 $X_k = U \thinspace \mathrm{diag}(\sigma^{(k)}) \thinspace V^{\top}$，则 $X_k X_k^{\top} = U \thinspace \mathrm{diag}\left( (\sigma^{(k)})^2 \right) U^{\top}$，进而 $(X_k X_k^{\top}) X_k = U \thinspace \mathrm{diag}\left( (\sigma^{(k)})^3 \right) V^{\top}$、 $(X_k X_k^{\top})^2 X_k = U \thinspace \mathrm{diag}\left( (\sigma^{(k)})^5 \right) V^{\top}$。代回式 (7)：

$$
X_{k+1} = U \thinspace \mathrm{diag}\left( f\left( \sigma^{(k)} \right) \right) V^{\top}, \qquad f(\sigma) = a \thinspace \sigma + b \thinspace \sigma^3 + c \thinspace \sigma^5 \qquad (8)
$$

**符号**： $f$：逐奇异值作用的标量五次多项式。整个矩阵迭代退化为对每个奇异值独立跑标量迭代 $\sigma^{(k+1)} = f(\sigma^{(k)})$——分析 NS 只需要分析一维 map $f$。若 $f$ 把 $(0, 1]$ 映向 1，则 $X_k \to U V^{\top} = \mathrm{msign}(G)$，正交化完成。

**系数的故事（出处与调参准则）**。先看经典取值 $(a, b, c) = (2, -1.5, 0.5)$： $f(1) = 1$ 是精确不动点且 $f'(1) = 0$（超吸引，练习 2），从任意 $\sigma_0 \in (0, 1]$ 收敛到 1（逐步递增、但有轻微过冲—— $f$ 在 $\sigma \approx 0.89$ 处峰值约 1.002，随后回落到 1；本书数值复核）。问题是小奇异值收敛慢： $\sigma_0 = 0.001$ 时要约 12 步（11 步约 0.997、12 步约 1.0000；见下表）。Muon 博客的调参准则（[Jordan, 2024](../references.md)，教学参照）针对这一点： $f'(0) = a$ 控制小奇异值的早期收缩速率，要让 $a$ 尽量大。约束是长程行为不跑偏——博客的原文口径是「经验上 $\varepsilon$ 高达 0.3 也不伤 loss 曲线」，故把目标放宽为长程值落在 $[0.7, 1.3]$。用 ad-hoc 梯度法调出

$$
(a, b, c) = (3.4445, \thinspace -4.7750, \thinspace 2.0315) \qquad (9)
$$

Moonlight 沿用同一组系数并注明出处（[Liu et al., 2025](../references.md)，[A]，§2.1：为使小初始奇异值更快收敛）；Muon 官方实现 `muon.py` 的 `zeropower_via_newtonschulz5` 原样内置（[Jordan, KellerJordan/Muon](../references.md)，教学参照+官方实现）。

**数值复核（本书对式 (8)(9) 的独立验证）**。三个事实：

| $\sigma_0$ | 1 步 | 3 步 | 5 步 | 经典系数 5 步 |
| --- | --- | --- | --- | --- |
| 0.001 | 0.0034 | 0.0409 | 0.4705 | 0.0320 |
| 0.010 | 0.0344 | 0.4001 | 0.6989 | 0.3120 |
| 0.100 | 0.3397 | 0.7071 | 0.7121 | 1.0005 |
| 0.500 | 1.1889 | 0.8244 | 0.7654 | 1.0000 |
| 1.000 | 0.7010 | 0.7207 | 0.6964 | 1.0000 |

- $f(1) = 0.701 \neq 1$：**1 不是这组系数的不动点**；继续迭代也不收敛到点，而是落入有界 4-环 $\lbrace 0.68, 1.13, 0.75, 1.05 \rbrace$（近似值）。这与官方实现的 docstring 一致：「产出不是 $U V^{\top}$ 而是 $U S' V^{\top}$， $`S'_{ii} \sim \mathrm{Uniform}(0.5, 1.5)`$」——近似是刻意的，用精确性换小奇异值的抬升速度。
- 5 步把 $\gtrsim 10^{-3}$ 的输入压进约 $[0.47, 1.20]$（细网格复算： $[10^{-3}, 1]$ 的像为 $[0.473, 1.202]$，最大值 1.202 出现在 $\sigma_0 \approx 0.004$ 附近）；但小端的放大倍率是有限的： $f$ 在 0 附近斜率为 $a$，5 步约 $a^5 \approx 485$ 倍，即归一化后小于约 $10^{-3}$ 的奇异方向 5 步内抬不进 $[0.5, 1.5]$ 邻域。NS₅ 正交化的有效带宽是「归一化奇异值 $\gtrsim 10^{-3}$」的方向族，更深的近零方向保留其小幅度——这不是缺陷而是事实边界（§9 还会回来）。
- 迭代对输入有发散区：本书数值复核显示发散阈值约为 1.264（ $\sigma_0 = 1.25$ 仍有界， $1.27$ 起发散）。所以式 (7) 首行的归一化不是精度技巧而是**正确性前提**——谱范数必须先压进收敛域。

一个自然的追问：4-环残余误差会不会跨训练步累积？不会。NS 在每个训练步被独立调用——输入是当步的动量矩阵 $\mu M_t + G_t$（式 (10)），首步归一化把上一步的残余完全洗掉。每步的正交化误差是「当步有界」而非「跨步复利」：步 $t$ 的更新方向与精确 msign 的差异被限制在奇异值 $[0.5, 1.5]$ 的形变内，方向基 $U$、 $V$ 本身不受标量动力学影响（式 (8) 保持奇异向量严格不变）。跨步漂移的唯一来源是优化器状态 $M_t$ 的正常递推，与 NS 无关。

**为什么 bf16 下 5 步就够。** 分两层回答：实证口径与机制理由。实证口径：Muon 博客报告 NS 迭代在 bf16 下稳定可跑，而同样能做正交化的 coupled Newton 迭代「必须至少 fp32 才避免数值不稳」，故选 NS（[Jordan, 2024](../references.md)）；Moonlight 报告 $N = 10$ 比 $N = 5$ 正交化更准，但不带来更好性能，故定 $N = 5$（[Liu et al., 2025](../references.md)，[A]，§2.2），官方实现默认 `steps=5`。机制层面是本书分析，三条：

- (a) 目标是区间不是点：容许 $[0.7, 1.3]$ 的残余误差，远大于 bf16 的相对精度 $2^{-8}$（约 0.4%）。
- (b) 迭代自校正：轨道向有界环收缩，单步的舍入误差会被后续迭代吸收而非复利放大。
- (c) 操作数有界：归一化后 $\lVert X_k \rVert_2 \lesssim 1.3$，中间量 $A = X_k X_k^{\top}$ 的元素量级受控，bf16 的动态范围绰绰有余。

三条合起来：低保真迭代加高容忍目标，换来精度与步数双省。

**数据流与开销**（设矩阵形状 $n \times m$、 $n \ge m$；实现先做转置使 $X$ 为宽矩阵，Gram 矩阵取小边——官方实现 `if G.size(-2) > G.size(-1): X = X.mT`）：

```
G [n, m]
 │  X₀ = G / ‖G‖_F（谱范数压进 [0, 1]）
 ▼
X_k [m, n] ──► A = X_k X_kᵀ [m, m]          2 m² n FLOPs
                 │   B = b·A + c·(A A)       2 m³ FLOPs
                 ▼
          X_{k+1} = a·X_k + B X_k            2 m² n FLOPs
                 │   × T = 5 步，全程 bf16
                 ▼
          O = X₅ [n, m]（转置回原向）≈ U S′ Vᵀ，S′ ∈ [0.5, 1.5] 邻域
```

每步 $4 m^2 n + 2 m^3 \le 6 n m^2$ FLOPs， $T$ 步合计 $\le 6 T n m^2$。对照该层一次前向加反向的 $6 n m B$ FLOPs（ $B$ 为该步流经此层的 token 数，models 1.2 的 $2mnk$ 口径），Muon 的相对开销 $\le T \thinspace m / B$。代入两组真实数字（[Jordan, 2024](../references.md)）：NanoGPT 速度赛（ $m = 768$、 $B = 524288$）得 $5 \times 768 / 524288 \approx 0.0073$，即约 0.7%；Llama-3 405B 规模（ $m = 16384$、 $B = 1.6 \times 10^{7}$）得约 0.5%。结论：NS 的 FLOPs 开销在大小两个尺度上都低于 1%。Muon 的成本不在算力，在工程——分布式，见 §6 末。

## 4. Muon 更新规则完整形式（Moonlight 口径，逐式回原文）

Muon 的完整一步分四段：momentum、NS 正交化、尺度匹配、weight decay。以下逐段给出 Moonlight 论文的原始公式（[Liu et al., 2025](../references.md)，arXiv:2502.16982，[A]，§2.1-2.2）与实现对照。

**第一段：momentum。** 论文式 (1) + 脚注 1 的 Nesterov 变体：

$$
M_t = \mu \thinspace M_{t-1} + G_t, \qquad O_t = \mathrm{NewtonSchulz}\left( \mu \thinspace M_t + G_t \right), \qquad M_0 = 0 \qquad (10)
$$

**符号**： $`G_t = \nabla \mathcal{L}_t (W_{t-1})`$：当前梯度； $\mu = 0.95$：momentum 系数（Moonlight 与 Muon 博客同值）； $O_t$：正交化后的更新方向。论文印刷版式 (1) 中 NS 的输入是 $M_t$；脚注 1 注明实践中沿用 Jordan et al. 的 Nesterov 形式、改喂 $\mu M_t + G_t$——式 (10) 按脚注口径写出。博客口径为「Nesterov 在所有测试情形都略好，已成公开实现默认」。

实现对照注记：官方 repo 用 EMA 参数化。 `momentum.lerp_(grad, 1-beta)` 给出 $`\tilde{M}_t = \beta \tilde{M}_{t-1} + (1-\beta) G_t`$，Nesterov 更新方向为 $`\beta \tilde{M}_t + (1-\beta) G_t`$。它与式 (10) 的输入只差一个正标量因子 $(1-\beta)$： $`G_t + \mu M_t = (1-\mu)^{-1} \left( \mu \tilde{M}_t + (1-\mu) G_t \right)`$（由 $`M_t = \tilde{M}_t / (1-\mu)`$ 立得，本书核对）。NS 首步会把输入归一化到谱范数 1，正标量因子被精确消去。两种参数化因此数学上严格等价；本书行文与公式按论文口径，即用式 (10)。

**第二段：NS 正交化**——即 §3 的式 (7)(9)，不再重复。

**第三段：尺度匹配（Moonlight 的核心增量）**。问题先以引理形式出现（论文 Lemma 1 + 附录 A）：

**引理 1（Muon 更新的 RMS）**：满秩矩阵参数形状 $[A, B]$，其 Muon 更新 $O = U V^{\top}$ 的理论 RMS 为 $\sqrt{1 / \max(A, B)}$。

证明（补全原文附录 A 跳过的交叉项）：设 $O = U_r V_r^{\top}$， $U_r \in \mathbb{R}^{n \times r}$、 $V_r \in \mathbb{R}^{m \times r}$ 列正交（ $n = \max(A, B) \ge m \ge r$），则

$$
\mathrm{RMS}(O)^2 = \frac{1}{mn} \sum_{i, j} O_{ij}^2 = \frac{1}{mn} \sum_{i, j} \sum_{k, k'} U_{ik} U_{ik'} V_{jk} V_{jk'} = \frac{1}{mn} \sum_{k, k'} \delta_{k k'} \thinspace \delta_{k k'} = \frac{r}{mn} \qquad (11)
$$

**符号**： $O_{ij} = \sum_k U_{ik} V_{jk}$；交叉项 $k \neq k'$ 对 $i$ 求和时因 $U$ 列正交 $\sum_i U_{ik} U_{ik'} = \delta_{k k'}$ 而消失， $V$ 同理——这是原文排版省略的一步。满秩 $r = m$ 时 $\mathrm{RMS}(O) = \sqrt{1/n} = \sqrt{1/\max(A, B)}$。证毕。

引理 1 的后果（论文原文两条）：宽矩阵（如 $[H, 4H]$ 的 MLP）更新幅度被多压 $\sqrt{4} = 2$ 倍，模型容量受限；把 GQA / MLA 的每个 KV head 当独立小矩阵时 $\max(A, B)$ 过小、更新过大，训练不稳。修复方法：把每个矩阵的更新按 $\sqrt{\max(A, B)}$ 放大，让 RMS 回到与形状无关的常数。固定到多少？对齐 AdamW。Adam/AdamW 的理论更新 RMS 约为 1——机制读法：无噪声极限下 $\hat{m}/\sqrt{\hat{v}} = \mathrm{sign}(g)$，逐坐标模长 1； $\beta_1 \lt \beta_2$ 与 $\epsilon \gt 0$ 把实际值压到 1 以下（论文脚注口径）——经验观察落在 $0.2\text{–}0.4$ 区间。附录 A 的扫描实验 $\lbrace 0.05, 0.1, 0.2, 0.4, 0.8 \rbrace$ 显示 0.2 与 0.4 明显优于其余设置且彼此接近，于是取 0.2。最终更新式（论文式 (4)）：

$$
W_t = W_{t-1} - \eta_t \left( 0.2 \cdot O_t \cdot \sqrt{\max(A, B)} + \lambda \thinspace W_{t-1} \right) \qquad (12)
$$

**符号**： $\eta_t$：学习率（schedule 归 2.3）； $\lambda$：weight decay 系数（Moonlight 训练取 0.1）； $0.2 \cdot \sqrt{\max(A, B)}$：尺度匹配因子—— $\sqrt{\max}$ 抵消引理 1 的形状依赖，0.2 对齐 AdamW 经验 RMS。

三个实现注记。

- 其一，Moonlight 官方示例代码（MoonshotAI/Moonlight repo `examples/toy_train.py` 的 `adjust_lr_for_muon`）把该因子实现为「逐形状调整 lr」： `adjusted_lr = lr * 0.2 * sqrt(max(A, B))`，与式 (12) 同义。
- 其二，代码把式 (12) 拆成两行执行，逐项恒等：`p.mul_(1 - lr * wd)` 落实 $-\eta_t \thinspace \lambda W_{t-1}$ 项；`p.add_(u, alpha=-adjusted_lr)` 配合上行落实 $-\eta_t \cdot 0.2 \sqrt{\max(A, B)} \cdot O_t$ 项。尺度因子只属于正交化项，wd 项的 $\eta_t$ 就是未调整的 `lr`，与式 (12) 的括号分配一一对应。
- 其三，Muon 原始 repo 的缩放是 $\sqrt{\max(1, A/B)}$ 而非 $\sqrt{\max(A, B)}$。论文脚注 3 说明：所有矩阵第二维相同时，两者只差全局常数。Moonlight 的版本对不同形状的矩阵给出形状无关的 RMS，是严格更强的形式。

**第四段：weight decay 的由来**。Muon 原始设计不含 wd。Moonlight 放大规模后观察到：权重与层输出的 RMS 持续增长、超出 bf16 的高精度区间，性能增益随之缩水（[Liu et al., 2025](../references.md)，[A]，§2.2）。对照实验：800M 参数、100B token（约 5 倍 compute-optimal 数据量）上，加 wd 的 Muon 在过训练区段 validation loss 低于裸 Muon 与 AdamW。论文式 (3) 据此先把 AdamW 式解耦 wd 并入（ $W_t = W_{t-1} - \eta_t (O_t + \lambda W_{t-1})$ ），式 (4)（即本章式 (12)）再叠尺度匹配因子。同节还有一个稳定细节：对 RMSNorm 的 gamma 施加 wd，对防止层输出 RMS 失控「至关重要」（论文附录 D 口径）。

**超参复用主张**。式 (12) 的回报写在论文 §3.2：RMS 匹配之后，Muon 直接复用为 AdamW baseline 网格搜索出的 lr 与 wd，不再单独调参。摘要的原话是「work out-of-the-box on large-scale training without the need of hyper-parameter tuning」。这句主张的精确含义是「复用同任务 AdamW 调好的超参」，不是「超参无关紧要」——baseline 本身的网格搜索仍是做过的（附录 B）。剩余两个自有超参： $\mu = 0.95$（调不出一致收益，沿用博客默认）与 $N = 5$（ §3 已述）。

## 5. 与 AdamW 的分工

Moonlight 论文 §2.2 一句话划定边界：「Muon 为矩阵参数设计；实践中以 AdamW 处理非矩阵参数，如 RMSNorm、LM head 与 embedding。」两边官方实现把这条边界落成同一张参数二分表（Muon repo `muon.py` docstring；Moonlight `toy_train.py`： `{0,1}` 维或判定为 embed/lm_head 的参数自动走 AdamW 分支，AdamW 分支 betas $(0.9, 0.95)$、 $\epsilon = 10^{-8}$）：

```
参数全集
 ├── 隐藏层矩阵（attention Q/K/V/O 按 head 分块、FFN 三矩阵、MoE expert）
 │     └── Muon：M_t = μM_{t-1} + G_t → NS₅(μM_t + G_t) → ×0.2·√max(A,B) → +λW 解耦 decay
 └── 其余（embedding、lm_head、RMSNorm gain、bias、scalar、router 等）
       └── AdamW：m̂/√v̂ 逐坐标自适应 + 解耦 wd（2.2 全推导）
```

三条实证注记（均出自 [Jordan, 2024](../references.md)，教学参照，标注为经验结论而非定理）：

- (a) embedding 与输出头虽是 2D，也必须用 AdamW。embedding 侧有 modular-norm 理论的支持（博客引 Large et al.）；输出头侧，博客明言「似乎不遵循该理论，纯由实证驱动」。
- (b) Q、K、V 各自单独过 Muon 优于拼成单个 QKV 矩阵（fused 实现要拆开）。
- (c) Nesterov momentum 为默认。

机制读法（本书分析）：embedding 的反向是 scatter-add（1.2），单步只触少数行，梯度矩阵天然行稀疏。把这种矩阵正交化，会把「哪些行该动」的稀疏结构摊成稠密等幅更新，与查表语义相斥。lm_head 与 embedding 共享词表维的稀疏统计，同理。这条解释与 §1 的几何分野一致：谱框架的前提是「参数充当空间之间的算子」，查表与读出不满足。

顺手记一笔内存账：Muon 的优化器态只有 1 份 momentum buffer，AdamW 是 $m$、 $v$ 两份。矩阵参数走 Muon 后，优化器态从每参数 2 份降到 1 份（博客口径：内存需求与 SGD-momentum 相同）。在 models 1.2 式 (8) 的优化器态账里，这是一项实打实的削减。

## 6. 效率账：约 2× claim 的精确口径

Moonlight 摘要的核心 claim：「Muon achieves $\sim 2\times$ computational efficiency compared to AdamW with compute optimal training」。正文的精确版本（[Liu et al., 2025](../references.md)，[A]，§3.2）：在 compute-optimal 设定下，Muon 只需约 **52% 的训练 FLOPs** 即达到 AdamW 同等性能（ $1/0.52 \approx 1.92 \approx 2$）。这句 claim 的适用条件必须逐条列出：

- 实验对象：Llama 架构（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]——架构出处）的 dense 模型系列，非 MoE；
- baseline 强度：AdamW 超参经网格搜索（附录 B 的流程：compute-optimal 的 $N\text{–}D$ 对先定，再搜 lr 与 batch size，迭代 2-3 轮收窄）——「AdamW 没调好」这一常见反驳在此设定下不成立；
- Muon 侧超参：RMS 匹配后直接复用 AdamW 搜索结果（§4），即收益不来自额外的调参预算；
- 口径：scaling law 拟合曲线（论文图 3 / 表 3）的 FLOPs-loss 前沿比较，非单点。

规模化证据。Moonlight 本体是 deepseek-v3-small 架构的 MoE： $2.24 \times 10^{9}$ 激活 / $15.29 \times 10^{9}$ 总参数（含 embedding 口径 3B/16B），5.7T token 全程以 Muon 训练（式 (12)），在同等 token 量下超过同架构公开模型，推进了性能-FLOPs 的 Pareto 前沿（论文表 4/5、图 1b）。Kimi K2 把同一优化器（加 MuonClip，§7）推到 $1.04 \times 10^{12}$ 总参数 / $32 \times 10^{9}$ 激活、15.5T token（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]）。两个数量级的跨度上都有公开 checkpoint 背书。

成本侧两笔账。其一，NS 迭代本身：§3 已算， $\le T m / B$，两个尺度均低于 1% FLOPs。其二，分布式。Muon 需要**完整**梯度矩阵做正交化，与 ZeRO-1 的逐元素分片语义冲突。Moonlight §2.3 的 Distributed Muon 沿用 ZeRO-1 的 DP 分片优化器态，新增两项操作：一次 DP gather 集合通信（把分片梯度聚成完整梯度矩阵），加一次全矩阵 NS 计算（算出完整更新后只保留本地分片对应块）。实现内存最优且数学性质不变。通信账的展开归 Part 4-5 的并行框架，此处只登记结论：分布式化不改变式 (10)-(12) 的数学。

## 7. MuonClip：Kimi K2 的权重端稳定原语

这个问题最早出现在 Moonlight 的训练日志里（[Liu et al., 2025](../references.md)，[A]，附录 D）：训练 loss 与 grad norm 全程平稳，但**最大 attention logit**（global batch 内逐元素最大值）在初期训练阶段在特定层持续上升、突破 100；AdamW 在该指标上表现更健康；超 100 的大 logit 比例维持在约 $10^{-4}$，且随训练推进自行回落。Kimi K2 的中规模实验把问题放大到失稳阈值： $9 \times 10^{9}$ 激活 / $53 \times 10^{9}$ 总的 MoE 用裸 Muon 训练时，max attention logit 迅速突破 1000——该量级「通常伴随 loss spike 乃至偶发发散」（[Kimi Team, 2025](../references.md)，[A]，§2.1）。

已有手段为什么不够（K2 原文口径）：soft-capping 直接裁 logit，但 query-key 点积在裁剪前已经长大，治标；QK-Norm 不适用于 MLA——推理时 Key 矩阵不完整物化（MLA 的低秩结构，models 5.5 有模型侧展开）。K2 的方案是把干预点从 logit 移到**权重**：post-update 的 QK-Clip。

定义逐 head 的 max logit 监控量与缩放因子：

$$
S_{\max}^{h} = \frac{1}{\sqrt{d}} \max_{X \in \mathcal{B}} \max_{i, j} \thinspace Q_i^h K_j^{h \top}, \qquad \gamma_h = \min\left( 1, \hspace{5mu} \tau / S_{\max}^{h} \right) \qquad (13)
$$

**符号**： $d$：head 维（K2 论文记号）； $\mathcal{B}$：当前 batch； $i, j$：token 位置； $\tau$：目标阈值（K2 取 100）。

当 $S_{\max}^h > \tau$ 时，对该 head 的投影权重做等比收缩：

$$
W_q^h \leftarrow \gamma_h^{\alpha} \thinspace W_q^h, \qquad W_k^h \leftarrow \gamma_h^{1 - \alpha} \thinspace W_k^h \qquad (14)
$$

**符号**： $\alpha \in [0, 1]$：平衡参数，典型取 $0.5$（query 与 key 均摊）。

效果一行可验：logit $\propto Q^h K^{h \top} = X W_q^h (W_k^h)^{\top} X^{\top}$，两侧分别缩 $\gamma_h^{\alpha}$ 与 $\gamma_h^{1-\alpha}$ 后 logit 整体缩 $\gamma_h^{\alpha} \cdot \gamma_h^{1-\alpha} = \gamma_h$——下一行前向的最大 logit 即被压回 $\tau$（近似：缩放后其他 head 与后续层不变）。MLA 下的分量规则（K2 原文）：head 私有的 $q^C$、 $k^C$ 各乘 $\sqrt{\gamma_h}$；head 私有 rotary 分量 $q^R$ 乘 $\gamma_h$；head 间共享的 $k^R$ 不动（避免跨 head 串扰）。

训练动力学侧的三点分析（本书增量）。第一，QK-Clip 是 **post-update** 操作：它在 optimizer step 完成之后、下一步前向之前作用于权重，不进计算图、不改当前步的前向/反向。梯度统计（ $m$、 $v$、momentum buffer）完全不受影响——这与 2.4 §4 的梯度端裁剪形成截面差异（回接 2.4 §6 总图：QK-clip 在「更新之后、参数之上」那一格）。第二，它本质是一个 per-head 的控制回路：监控量 $S_{\max}^h$，执行器为权重缩放，目标是约束权重增长的**速率**而非方向。 $\gamma_h$ 逐 head 计算、仅少数 head 被触发（K2 原文：实践中只有少数 head 的 logit 爆炸，per-head 版本把干预最小化）。第三，与 soft-cap 的分工：soft-cap 改前向非线性、每步都生效；QK-clip 只在超阈时生效，作用在权重漂移的慢时间尺度上。两者不互斥，但 MLA 架构下只有后者可用。

结果口径：K2 以 MuonClip（ = Muon + wd + RMS 匹配 + QK-Clip，K2 论文 Algorithm 1 的组合体）、 $\tau = 100$、WSD schedule（2.3 的梯形族）预训练 15.5T token，**零 loss spike**；max logit 初期被封顶在 100，约 30% 训练步后自行衰减回稳定区间、 $\tau$ 全程无需调整（[Kimi Team, 2025](../references.md)，[A]，§2.1 与图 2-3）。模型侧的 MLA 公式与 QK-clip 的结构细节纯文字指引 models 5.5。

## 8. 与 µP 的对照：两条免调参路线

Moonlight 的主张（§4 末）——加 wd、逐形状 RMS 匹配之后 Muon 开箱即用、复用 AdamW 超参——与 µP（maximal update parametrization）解决的是同一个工程痛点：大模型的超参不能直接在小模型上扫。两条路线的哲学对照：

- **µP 路线**：从理论上推出初始化方差与 LR 随宽度的缩放律，使最优超参对宽度不变——小代理模型上扫参、按缩放律直迁大模型（µTransfer）。完整缩放结论表与迁移账归 7.2。
- **RMS 匹配路线**（Moonlight）：不改参数化，把每个参数的**更新幅度**固定到一个与形状无关的绝对工作点（RMS = 0.2，对齐 AdamW 经验区间），于是「AdamW 上调好的 lr/wd」直接可搬。

纪律注记（spec 四路审查销账项）：Moonlight 全文不含 transfer / µP 内容，其主张是 RMS 匹配后的复用，**不要**把它写成 µTransfer 的实践案例；两路线的实证对照（各自在不同规模上的迁移精度）是开放问题，7.2 收配方总账时再并陈。

## 9. 失效模式与开放问题

Moonlight 官方示例实现（MoonshotAI/Moonlight `examples/toy_train.py` 的 `Muon` 类 docstring）自带两条醒目的警告：「unlikely to work well for training with small batch size」「may not work well for finetuning pretrained models, but we haven't tested this」（[Liu et al., 2025](../references.md)，[A] 配套代码）——规模化实践者自己标注的适用范围，是失效边界讨论的诚实起点。逐条列出，注明证据等级：

1. **非矩阵参数不适用**：谱框架的前提（§1）不满足——embedding / lm_head / norm / scalar 一律 AdamW（§5，实证口径）。
2. **小 batch 与微调未验证**：即本节开头的两条 docstring 警告（Moonlight 示例实现）——本书锚点内的全部规模化证据都在预训练场景。
3. **正交化的双刃性**：把奇异值压平放大稀有方向——若小奇异方向承载的是噪声而非信号，Muon 放大噪声。「稀有方向重要」在源头处即为 speculation（博客原词）；SVD 熵证据（§2 末）是相关而非因果。
4. **NS₅ 的有效带宽**：§3 数值复核——归一化后 $\lesssim 10^{-3}$ 的奇异方向 5 步抬不进 $[0.5, 1.5]$；数值近秩亏的梯度矩阵只有部分谱被真正正交化。迭代的发散区（ $\sigma_0 \gtrsim 1.26$）使首步归一化成为硬前提。
5. **MuonClip 是控制不是病因**：QK-clip 压制 logit 爆炸的表型，「为什么 Muon 下 logit 更易漂移」的机制解释在本书锚点内没有定论（K2 只报告「Muon 下更频繁、AdamW 更健康」的现象学）。
6. **后训练与 RL 阶段**：K2 明言 post-training 全程用 Muon——「We employ the Muon optimizer in our post-training and recommend its use for fine-tuning with K2」（[Kimi Team, 2025](../references.md)，[A]，§3.1），RL 目标同样以 Muon 优化（「As in SFT, we employ the Muon optimizer to minimize this objective」，§3.2.3）；Moonlight 侧另有「Muon 预训练 checkpoint 配 Muon 微调性能最好」的结论（K2 §3.1 转引）。即预训练之外已有 1T 级公开实证；仍开放的口径收窄为：这些是单一配方的工作记录，缺与 AdamW 的对照消融和独立复现。

开放问题的谱系位置：Muon 博客原文的三个 open questions（更大规模、分布式 NS、预训练之外的泛化）中，前两个已被 Moonlight（16B / 5.7T token / Distributed Muon）与 K2（1T / 15.5T token）回答；第三个由 K2 的 post-training 实践部分回答（SFT 与 RL 均在 1T 级跑通，见第 6 条），但证据形态是单方工作记录而非对照实验，强度低于预训练侧的 scaling law 证据。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 谱几何动机 | $\lVert \Delta x \rVert \le \lVert \Delta \rVert_2 \lVert x \rVert$，式 (1) | 谱范数 = 单位输入的最坏功能扰动；只适用算子型矩阵参数 |
| 最速下降 | $\Delta^{\ast} = -\eta \thinspace U V^{\top}$，式 (2)-(5)；下降量 $\eta \lVert G \rVert_{\ast}$ | 对偶范数配对：谱 ↔ 核；满秩时解唯一 |
| msign 三形式 | $U V^{\top} = (G G^{\top})^{-1/2} G$，式 (6) | 即离 $G$ 最近的半正交矩阵 |
| NS 迭代 | 式 (7)(9)： $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ 逐奇异值作用，式 (8) | 系数 $(3.4445, -4.7750, 2.0315)$； $f(1) = 0.701$；长程 4-环 $[0.68, 1.13]$；5 步把 $\gtrsim 10^{-3}$ 的输入映进约 $[0.47, 1.20]$； $\sigma_0 \gtrsim 1.26$ 发散（本书数值复核） |
| NS 开销 | 每步 $\le 6 n m^2$ FLOPs；总开销 $\le T m / B$ | NanoGPT 0.7%、405B 0.5%（博客口径）；bf16 可跑 |
| Muon 完整式 | 式 (10)(12)：momentum $\mu = 0.95$ → NS₅ → $\times 0.2 \sqrt{\max(A,B)}$ → $+\lambda W$ | 引理 1： $\mathrm{RMS} = \sqrt{1/\max(A,B)}$；AdamW 经验 RMS $0.2\text{–}0.4$；wd $\lambda = 0.1$；复用 AdamW 超参 |
| 分工 | §5 参数二分 | 隐藏矩阵 → Muon；embedding/lm_head/norm/scalar → AdamW；优化器态 2 份降 1 份 |
| 效率 claim | compute-optimal 下约 52% FLOPs 追平 AdamW（ $\approx 2\times$） | 条件：dense Llama 架构系列、grid-searched baseline、复用超参 |
| MuonClip | 式 (13)(14)： $\gamma_h = \min(1, \tau / S_{\max}^h)$， $\tau = 100$ | post-update 权重缩放、不进计算图；K2 15.5T token 零 spike；MLA 分量规则 |
| µP 对照 | RMS 匹配 vs 缩放律迁移 | Moonlight 无 µP 内容；配方对照归 7.2 |

## 练习

1.（推导）秩亏情形的式 (2)：设 $G$ 的秩 $r \lt \min(n, m)$。证明解不唯一，解集为 $\lbrace \Delta = U D V^{\top} : D_{ii} = -\eta \thinspace (i \le r), \thinspace \lVert D \rVert_2 \le \eta \rbrace$；并证明 $\Delta = -\eta \thinspace U_r V_r^{\top}$（即 $\mathrm{msign}$ 的推广：零奇异值的符号取 0）是其中 Frobenius 范数最小的解。
2.（推导）完成 §3 的两处验证：(a) 若 $X_k = U \thinspace \mathrm{diag}(\sigma) \thinspace V^{\top}$ 则式 (7) 给出 $X_{k+1} = U \thinspace \mathrm{diag}(f(\sigma)) \thinspace V^{\top}$；(b) 经典系数 $(2, -1.5, 0.5)$ 满足 $f(1) = 1$ 且 $f'(1) = 0$（超吸引不动点），并求其二阶接触阶——由此解释为何它从小奇异值出发收敛慢（提示： $f(\sigma)/\sigma$ 在 0 附近的值）。
3.（计算）NS 开销账：对 $d = 4096$、 $d_{\mathrm{ff}} = 14336$ 的 FFN down 投影矩阵（ $[4096, 14336]$）， $T = 5$、该层每步流经 $B = 4 \times 10^{6}$ token。按 §3 的 $6 T n m^2$ 与 $6 n m B$ 估算 Muon 对该层的相对 FLOPs 开销，与 $T m / B$ 的捷径口径对照；再答：batch 缩到 $B = 4 \times 10^{4}$（小 batch 微调）时开销变成多少，这与 §9 第 2 条的官方警告有何呼应？
4.（代码）(a) 用 20 行 Python 实现标量 map $f$（式 (8)(9)），复现 §3 表格的 5 步值与长程 4-环；(b) 用 PyTorch 实现 NS₅（对照 `zeropower_via_newtonschulz5`，教学用，非优化实现），对随机高斯矩阵 $G \in \mathbb{R}^{256 \times 128}$ 验证 $X_5$ 的奇异值落在 $[0.5, 1.5]$ 邻域的比例，并与 SVD 的 $U V^{\top}$ 比较方向差异 $\lVert X_5 - U V^{\top} \rVert_F / \lVert U V^{\top} \rVert_F$。

## 参考

- [Jordan, 2024](../references.md)：Muon: An optimizer for hidden layers（博客），教学参照（key: muon-blog）——Muon 定义、NS 系数调参准则、bf16 与 5 步口径、开销公式 $T m / B$、分工与 QKV 拆分的实证注记
- [Jordan, KellerJordan/Muon repo](../references.md)：教学参照+官方实现（key: muon-repo）——`muon.py`：`zeropower_via_newtonschulz5`（式 (7)(9) 的实现）、`muon_update`（EMA/Nesterov 与 $\sqrt{\max(1, A/B)}$ 缩放）、`Muon` 类 docstring（隐藏层限定与分工口径）
- [Liu et al., 2025](../references.md)：Muon is Scalable for LLM Training（Moonlight），arXiv:2502.16982，[A]（key: moonlight）——式 (10)(12)、引理 1 与附录 A 证明、RMS 扫描、wd 动机、 $N=5$ 与 $\mu=0.95$、约 2× 效率口径、Distributed Muon、附录 D 的 max attention logit 记录；官方实现 MoonshotAI/Moonlight `examples/toy_train.py`（`adjust_lr_for_muon` 的逐形状 lr 实现、`Muon` 类 docstring 的小 batch 与微调警告）
- [Kimi Team, 2025](../references.md)：Kimi K2: Open Agentic Intelligence，arXiv:2507.20534，[A]（key: kimi-k2）——MuonClip（Algorithm 1）、QK-Clip 式 (13)(14) 与 MLA 分量规则、 $\tau = 100$、15.5T token 零 spike
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——Moonlight  scaling law 实验的架构出处
