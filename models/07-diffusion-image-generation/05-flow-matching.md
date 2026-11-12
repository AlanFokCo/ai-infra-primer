# 7.5 Flow Matching 与直线生成路径（Flow Matching and Rectified Flow）

> 本章把生成建模从「逐步去噪的马尔可夫链」改写为「沿速度场积分一条 ODE」：CNF 的 pushforward 密度与精确似然的 trace 公式及其成本账、FM 目标 $L_{FM}$、conditional FM 定理完整推导（条件速度场回归与边际速度场回归梯度恒等——期望交换是关键步）、Gaussian 条件路径的速度场闭式、rectified flow 线性路径与目标速度 $\varepsilon - x_0$、与 DDPM $\varepsilon$-prediction 的同族映射、logit-normal $t$ 采样与分辨率相关 timestep shift（SD3）、diffusion↔FM 统一表 | 前置：7.1（家族坐标与约定）、7.2（DDPM 全链推导、$L_{simple}$、$\varepsilon$/$x_0$/$v$ 参数化）、7.3（score 与 probability-flow ODE） | 锚点：Flow Matching [A]（arXiv:2210.02747）、rectified flow [A]（arXiv:2209.03003）、SD3 [A]（arXiv:2403.03206）；facebookresearch/flow_matching [A]、Stability-AI/sd3.5 [A]

## 1. CNF：把生成分布写成一条 ODE 的 pushforward

7.2 的 DDPM 把生成拆成 $T = 1000$ 步离散转移，本章换一个连续视角：样本沿一条常微分方程（ODE）的轨迹从噪声端运动到数据端。设 $d$ 维空间上有一族随时间变化的速度场（velocity field）$u_t(x) : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$，由它定义的 ODE 为

$$
\frac{d}{dt}\, \psi_t(x) = u_t\left( \psi_t(x) \right), \qquad \psi_0(x) = x \qquad (1)
$$

**符号**：$\psi_t(x)$：从位置 $x$ 在时刻 $0$ 出发、沿速度场流动到时刻 $t$ 的位置，称为 flow map；$u_t(x)$：时刻 $t$、位置 $x$ 处的瞬时速度；$t \in [0, 1]$，本书约定 $t = 0$ 为数据端、$t = 1$ 为噪声端（与 7.2 及 SD3 一致）。

给定初值分布 $p_0$，flow map 把它的质量整体搬运到 $t$ 时刻——每个点的概率质量随轨迹走，这就是 pushforward（推前）：

$$
p_t = [\psi_t]_{*}\, p_0 \qquad (2)
$$

**符号**：$[\psi_t]_{*}$：pushforward 算子，即随机变量 $x \sim p_0$ 经映射 $\psi_t$ 变换后的分布；$p_t$：$t$ 时刻的边际密度。

生成建模的目标由此改写为：找一个速度场 $u_t$，使 $p_0 = p_{\mathrm{data}}$（数据端）而 $p_1 = \mathcal{N}(0, I)$（纯噪声端）。采样时从 $x_1 \sim \mathcal{N}(0, I)$ 出发沿式 (1) 反向积分（$t: 1 \to 0$）即得数据样本。用速度场参数化的这类生成模型称为 continuous normalizing flow（CNF，连续归一化流）：

```
t: 0（数据端）────────────────► 1（噪声端）     采样方向相反：t: 1 → 0
p_0 = p_data                                  p_1 = N(0, I)

  x_0 ──ψ_t──► x_t ──► x_1 = ε      每个点沿 dx/dt = u_t(x) 运动
  ●──●──●      轨迹互不相交           密度连续变形：
   ●──●──●     （ODE 解唯一）         p_0 → p_t → p_1（pushforward）
    ●──●──●                         边界条件只钉两端
```

ODE 的解唯一性给出一条几何约束：过同一点 $(x, t)$ 只有一条轨迹——**轨迹互不相交**。于是整条流把噪声端与数据端之间的点一一配成对，生成是一个确定性映射（固定 $x_1$ 即固定输出），插值、反演这类操作免费可得（对照 7.4 的 DDIM $\sigma = 0$ 同款性质）。这也标出了 FM 与随机采样式扩散的边界：ancestral sampling 的随机性对应 SDE 的扩散项（7.3），FM/ODE 视角天然是确定性的那一侧。

与 7.1 §5 的离散 flow（Glow 式双射堆叠）对照：离散版把变量替换写成有限个可逆层的复合，每层要为 $\log|\det|$ 付出架构约束的代价；CNF 把它换成连续极限——层数趋于无穷、每层扰动趋于无穷小，变量替换公式相应地变成沿轨迹的一个积分（§2 的 trace 公式）。换来的架构自由度是实质性的：$u_t(x)$ 可以是任意神经网络（U-Net、DiT），不再需要可逆层设计。

密度 $p_t$ 与速度场 $u_t$ 之间的桥梁是连续性方程（continuity equation，质量守恒的局部形式）：

$$
\frac{\partial}{\partial t} p_t(x) + \nabla \cdot \left( p_t(x)\, u_t(x) \right) = 0 \qquad (3)
$$

**符号**：$\nabla \cdot$：散度，$\nabla \cdot F = \sum_{i=1}^{d} \partial F_i / \partial x_i$；$p_t u_t$：概率质量的通量。式 (3) 读法：某点密度的变化率 = 净流入量的负值——质量不生不灭，只随速度场搬运。式 (3) 是本章一切推导的基础设施：给定 $u_t$ 求 $p_t$ 是解这个偏微分方程，给定 $p_t$ 反推合法的 $u_t$（§3）也要过它。

## 2. 精确似然：trace 公式与成本账

CNF 的卖点之一是精确似然可算。先看离散版的对照物再取连续极限。有限步变量替换（7.1 式 (4)）给出 $p_t(\psi_t(x)) = p_0(x) \, |\det J_t(x)|^{-1}$，其中 $J_t = \partial \psi_t / \partial x$ 是 flow map 的 Jacobian。两侧取对数并对 $t$ 求导，需要 $J_t$ 的演化方程。对式 (1) 关于初值 $x$ 求导（交换 $\partial/\partial t$ 与 $\partial/\partial x$）：

$$
\frac{d}{dt} J_t(x) = \frac{\partial u_t}{\partial x}\bigg|_{\psi_t(x)} J_t(x) \qquad (4)
$$

**符号**：$\partial u_t / \partial x$：速度场在当前位置的 $d \times d$ Jacobian；式 (4) 是 $J_t$ 的线性 ODE，初值 $J_0 = I$。

再用 Jacobi 恒等式（矩阵行列式对参数的导数公式；练习 1 推导）：对可逆矩阵值函数 $J(t)$，$\frac{d}{dt} \log |\det J| = \operatorname{tr}\left( J^{-1} \frac{dJ}{dt} \right)$。代入式 (4)（$J^{-1}$ 与 $\partial u/\partial x \cdot J$ 的迹按循环性合并）：

$$
\frac{d}{dt} \log \left| \det J_t(x) \right| = \operatorname{tr}\left( \frac{\partial u_t}{\partial x}\bigg|_{\psi_t(x)} \right) = \nabla \cdot u_t\left( \psi_t(x) \right) \qquad (5)
$$

**符号**：$\operatorname{tr}$：矩阵的迹；末一等号用「Jacobian 的迹 = 散度」。式 (5) 称 instantaneous change of variables（瞬时变量替换）：对数体积元沿轨迹的变化率恰是速度场的散度——局部膨胀则密度稀释，收缩则浓缩。把它沿轨迹从 $0$ 积到 $1$（注意 $\log p_t(\psi_t(x)) = \log p_0(x) - \log|\det J_t(x)|$ 的符号），得精确似然：

$$
\log p_0(x) = \log p_1\left( \psi_1(x) \right) + \int_{0}^{1} \nabla \cdot u_t\left( \psi_t(x) \right) dt \qquad (6)
$$

**符号**：$\psi_1(x)$：把数据点 $x$ 沿 ODE 推到噪声端的终点；$\log p_1$：标准高斯密度，闭式。

式 (6) 为什么贵？两笔账，都是维度税：

- **散度本身**。$\nabla \cdot u_t = \sum_{i=1}^{d} \partial u_{t,i} / \partial x_i$ 含 $d$ 个偏导，对每个 $(x, t)$ 求值需要 $d$ 次 Jacobian-向量积（每次约一次反向传播的开销）。图像 latent $d$ 在 $10^4$ 量级，精确散度即每点上万次反传。
- **沿轨迹积分**。式 (6) 的积分要用 ODE solver 数值走完整条 $[0, 1]$ 轨迹，每个积分步都要重新算一次散度——总开销 = 积分步数 × 每步散度开销。

工程上的标准近似是 Hutchinson 迹估计：对任意方阵 $A$，$\operatorname{tr}(A) = \mathbb{E}_{z \sim \mathcal{N}(0, I)}\left[ z^{\top} A z \right]$（由 $\mathbb{E}[z z^{\top}] = I$ 即得）。每个探针 $z$ 只要一次向量-Jacobian 积，$K$ 个探针把方差降到 $1/K$ 量级——但引入的随机性使对数似然变成有噪声的估计，且每步仍要常数次额外反传。结论是：CNF 的精确似然「可算但贵」，按最大似然直接训练 CNF（目标含式 (6)，每步优化都要付这笔账）在大 $d$ 上不切实际。7.3 会把同一公式用在 diffusion 的 probability-flow ODE 上做离线 NLL 评估（评估一次可以忍，训练每步都付不能忍）；而 §3 的 flow matching 正是「绕开式 (6)、不碰散度」的训练路线。

## 3. FM 目标：回归一个「写出来、算不出」的速度场

Flow matching（FM，[Lipman et al., 2022](../references.md)，arXiv:2210.02747，[A]）的目标函数朴素到一句话：拿网络 $v_\theta(x, t)$ 去回归真实速度场 $u_t(x)$：

$$
L_{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t} \left\| v_\theta(x, t) - u_t(x) \right\|^2 \qquad (7)
$$

**符号**：$t$ 先按某分布（均匀或 §8 的 logit-normal）采样；$x$ 按 $t$ 时刻的真实边际 $p_t$ 采样；$u_t(x)$：生成 $p_t$ 这族密度的真实速度场（§1）。

> **约定映射框（Lipman 原文 ↔ 本书，本章只出这一次）**：FM 原论文约定 **$x_1$ = 数据、$x_0$ = 噪声**，路径从 $t = 0$（噪声）演化到 $t = 1$（数据），与本书相反。映射关系为
> $$x_0^{\text{Lipman}} = \varepsilon\ (\text{noise}),\qquad x_1^{\text{Lipman}} = x_0\ (\text{data}),\qquad t^{\text{Lipman}} = 1 - t$$
> 速度场随之反号：$u^{\text{Lipman}}(x, t_L) = -u_{1 - t_L}(x)$（因 $dx/dt_L = -dx/dt$）。本书此后一律用 $x_0$ = 数据、$t = 1$ = 噪声端重述，不再混写。

式 (7) 的困难在于 $u_t(x)$ 不可解。真实边际是数据分布沿路径的混合

$$
p_t(x) = \int p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)\, dx_0 \qquad (8)
$$

**符号**：$p_t(x \mid x_0)$：从单个数据点 $x_0$ 出发的条件路径（人为设计、闭式，§5）；$p_{\mathrm{data}}$ 未知，只有样本。

$p_t$ 本身是无闭式的混合密度，而连续性方程式 (3) 里的 $u_t$ 又依赖 $p_t$ 的时空结构——直接解出 $u_t$ 无从谈起。直观地看，$x$ 处在 $t$ 时刻可能来自许多不同的 $x_0$，每个来源推着不同的条件速度，真实的 $u_t(x)$ 是这些来源的某种平均。

一个能手算的最小例子把这层「平均」坐实。取两点数据集 $p_{\mathrm{data}} = \frac{1}{2}\delta(x_0 - a) + \frac{1}{2}\delta(x_0 + a)$，线性路径（§6 预告）$x_t = (1-t) x_0 + t\varepsilon$。$t$ 时刻的边际是两个高斯的等权混合 $\frac{1}{2}\mathcal{N}((1-t)a, t^2) + \frac{1}{2}\mathcal{N}(-(1-t)a, t^2)$；在 $x = 0$ 处两个来源完全对称，后验各占一半、条件速度 $\varepsilon \mp a$ 平均后互相抵消，$u_t(0) = \mathbb{E}[\varepsilon] = 0$——轨迹在此处必须减速让路。靠近 $x = (1-t)a$ 处后验压倒性偏向 $+a$ 来源，$u_t(x) \approx \mathbb{E}[\varepsilon \mid x] - a$ 恢复单源速度。边际场处处是后验加权平均，而 weights 随 $x$ 连续变化——这正是「直线条件路径」叠加出「弯曲边际轨迹」的机制（§10 的曲率账由此而来），也是 $u_t$ 必须依赖 $p_{\mathrm{data}}$ 全部信息的直接证据：它含后验 $p(x_0 \mid x_t)$，后验含数据分布。

式 (7) 因此是一个「目标函数写得出来、逐点目标值算不出来」的回归——和 7.2 推导前 $L_{vlb}$ 的处境同构，解法也同构：找一个期望相同（梯度相同）但可采样的替代目标。这就是 conditional FM 定理。

## 4. conditional FM 定理（完整推导）

构造如下对象。对每个 $x_0$ 取定一族**条件概率路径** $p_t(x \mid x_0)$ 与一个**条件速度场** $u_t(x \mid x_0)$，要求：

- （C1）对每个 $x_0$，条件对满足连续性方程：$\partial_t p_t(x \mid x_0) + \nabla \cdot \left( p_t(x \mid x_0)\, u_t(x \mid x_0) \right) = 0$；
- （C2）端点正确：$p_0(x \mid x_0)$ 集中于 $x_0$（数据端退化），$p_1(x \mid x_0) = \mathcal{N}(0, I)$ 且不依赖 $x_0$（噪声端统一）。

条件速度场是设计出来的、闭式可算（§5 给出 Gaussian 族的显式公式）。**conditional FM 目标**直接回归它：

$$
L_{CFM}(\theta) = \mathbb{E}_{t,\; x_0 \sim p_{\mathrm{data}},\; x \sim p_t(\cdot \mid x_0)} \left\| v_\theta(x, t) - u_t(x \mid x_0) \right\|^2 \qquad (9)
$$

**符号**：期望可采样：抽数据点 $x_0$、按闭式条件路径采 $x$、算条件速度——三项全可行。

**定理（conditional FM，Lipman et al.）**：$\nabla_\theta L_{FM}(\theta) = \nabla_\theta L_{CFM}(\theta)$——用条件速度场回归与用不可解的边际速度场回归，梯度完全相同。

证明分三步。

**第一步：定义边际速度场并验证它生成 $p_t$。** 定义

$$
u_t(x) \;=\; \int u_t(x \mid x_0)\, \frac{p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)}{p_t(x)}\, dx_0 \;=\; \mathbb{E}\left[ u_t(x \mid x_0) \mid x_t = x \right] \qquad (10)
$$

**符号**：被积权重 $p_t(x \mid x_0) p_{\mathrm{data}}(x_0) / p_t(x)$ 即后验 $p(x_0 \mid x_t = x)$（Bayes，分母是式 (8) 的边际）——边际速度场 = 条件速度场对「可能来源 $x_0$」的后验平均，印证 §3 的直觉。

验证 $(p_t, u_t)$ 满足式 (3)：对式 (8) 求 $\partial_t$，代入 (C1) 再整理：

$$
\begin{aligned}
\partial_t p_t(x) &= \int \partial_t p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)\, dx_0 = -\int \nabla \cdot \left( p_t(x \mid x_0)\, u_t(x \mid x_0) \right) p_{\mathrm{data}}(x_0)\, dx_0 \\
&= -\nabla \cdot \left( \int u_t(x \mid x_0)\, p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)\, dx_0 \right) = -\nabla \cdot \left( p_t(x)\, u_t(x) \right)
\end{aligned} \qquad (11)
$$

关键操作只有两个：散度对 $x$ 作用、对积分变量 $x_0$ 是线性的可换序；末步认回式 (10) 的定义（乘以 $p_t(x)$ 的形式）。端点由 (C2) 保证。所以式 (10) 确实是「生成式 (8) 这族边际」的合法速度场。

**第二步：展开两个损失，逐块对齐。** 两个平方式各自展开为三块（$\|a - b\|^2 = \|a\|^2 - 2\langle a, b \rangle + \|b\|^2$）：

$$
L_{FM} = \mathbb{E}_{t, x \sim p_t}\|v_\theta\|^2 - 2\, \mathbb{E}_{t, x \sim p_t}\langle v_\theta, u_t(x) \rangle + \mathbb{E}_{t, x \sim p_t}\|u_t(x)\|^2 \qquad (12)
$$

$$
L_{CFM} = \mathbb{E}_{t, x_0, x}\|v_\theta\|^2 - 2\, \mathbb{E}_{t, x_0, x}\langle v_\theta, u_t(x \mid x_0) \rangle + \mathbb{E}_{t, x_0, x}\|u_t(x \mid x_0)\|^2 \qquad (13)
$$

第一块相等：$v_\theta$ 不含 $x_0$，对联合 $(x_0, x)$ 的期望把 $x_0$ 积掉就回到边际期望，$\mathbb{E}_{t, x \sim p_t}\|v_\theta\|^2 = \mathbb{E}_{t, x_0, x}\|v_\theta\|^2$。第三块都不含 $\theta$（目标速度不依赖网络），求梯度时消失。

**第三步（关键步）：交叉项的期望交换。** 要证 $\mathbb{E}_{x \sim p_t}\langle v_\theta(x, t), u_t(x) \rangle = \mathbb{E}_{x_0, x \sim p_t(\cdot \mid x_0)}\langle v_\theta(x, t), u_t(x \mid x_0) \rangle$。把式 (10) 乘回 $p_t(x)$ 的形式代入：

$$
\begin{aligned}
\mathbb{E}_{x \sim p_t}\langle v_\theta(x, t), u_t(x) \rangle &= \int \langle v_\theta(x, t),\, p_t(x)\, u_t(x) \rangle\, dx \\
&= \int \left\langle v_\theta(x, t),\, \int u_t(x \mid x_0)\, p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)\, dx_0 \right\rangle dx \\
&= \int \int \langle v_\theta(x, t),\, u_t(x \mid x_0) \rangle\, p_t(x \mid x_0)\, p_{\mathrm{data}}(x_0)\, dx_0\, dx \\
&= \mathbb{E}_{x_0 \sim p_{\mathrm{data}},\; x \sim p_t(\cdot \mid x_0)}\langle v_\theta(x, t), u_t(x \mid x_0) \rangle
\end{aligned} \qquad (14)
$$

第一行把期望写成积分；第二行代入式 (10) 的定义（$p_t(x)$ 与分母相消——这是全证明的枢纽：内积对 $u_t$ 是线性的，后验平均可以提到内积外面变成对 $(x_0, x)$ 的联合采样）；第三行 Fubini 换序；末行认回联合期望。三块对齐完毕，$L_{FM} - L_{CFM}$ 与 $\theta$ 无关，梯度恒等成立。$\square$

两点注记，防误读：

- **定理管梯度、不管损失值**。$L_{CFM}$ 比 $L_{FM}$ 多一个非负的差距项 $\mathbb{E}\,\|u_t(x \mid x_0) - u_t(x)\|^2$（条件速度的方差，练习 2），所以它不再是速度的绝对拟合误差，不能用来横向比较两条路径的「学习难度」数值。
- **最优解一致**。回归的逐点最优是条件期望：式 (9) 对固定 $(x, t)$ 的最小解为 $v^{*}(x, t) = \mathbb{E}[u_t(x \mid x_0) \mid x_t = x]$，由式 (10) 这正是 $u_t(x)$。容量无限时 CFM 学到的就是真边际场——与 7.2 里 $L_{simple}$ 的最优解 $\varepsilon^{*}(x_t, t) = \mathbb{E}[\varepsilon \mid x_t]$（从而复原真实后验均值与 score）是同一个「MSE 回归出条件期望」的机制。

实现侧的对应读法：每个训练样本只采一组 $(t, x_0, \varepsilon)$、算一个 $\|v_\theta - u_t(x \mid x_0)\|^2$，这是式 (13) 的单样本 Monte Carlo 估计——定理保证梯度无偏，方差全部来自条件速度的散布（练习 2 的分解项），batch 平均即消。另注意定理对条件路径的唯一要求是 (C1)(C2)，Gaussian 只是闭式最方便的取法；任何能给出「可采样路径 + 可算条件速度」的设计（包括从配对数据出发的非高斯耦合）都落在同一框架内——这是 FM 被称为「路径设计框架」而非「某个具体模型」的原因。

## 5. Gaussian 条件路径：$u_t$ 的闭式

§4 把问题归约为「设计一对满足 (C1)(C2) 的条件路径与条件速度场」。取高斯族：

$$
p_t(x \mid x_0) = \mathcal{N}\left( \alpha_t\, x_0,\; \sigma_t^2\, I \right) \qquad (15)
$$

**符号**：$\alpha_t$：信号系数，光滑、单调递减，$\alpha_0 = 1$、$\alpha_1 = 0$；$\sigma_t$：噪声水平，光滑、单调递增，$\sigma_0 = 0$、$\sigma_1 = 1$——两端点恰实现 (C2)。注意本章 $\alpha_t$ 是 FM 语境的路径系数，勿与 7.2 的 $\alpha_t = 1 - \beta_t$ 混淆；DDPM 路径对应 $\alpha_t = \sqrt{\bar\alpha_t}$、$\sigma_t = \sqrt{1 - \bar\alpha_t}$（§7）。

条件路径的采样式为

$$
x_t = \alpha_t\, x_0 + \sigma_t\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) \qquad (16)
$$

**符号**：$\varepsilon$：条件路径内的标准高斯样本。固定 $(x_0, \varepsilon)$，式 (16) 给出一条从 $x_0$ 出发的轨迹。

对 $t$ 求导得该轨迹的速度，再把 $\varepsilon$ 用当前位置 $x$ 反解（式 (16) 移项，$\varepsilon = (x - \alpha_t x_0)/\sigma_t$）代回：

$$
u_t(x \mid x_0) = \dot\alpha_t\, x_0 + \dot\sigma_t\, \varepsilon = \frac{\dot\sigma_t}{\sigma_t}\, x + \left( \dot\alpha_t - \frac{\dot\sigma_t}{\sigma_t}\, \alpha_t \right) x_0 \qquad (17)
$$

**符号**：$\dot\alpha_t, \dot\sigma_t$：对 $t$ 的导数。式 (17) 即 Gaussian 条件路径的速度场闭式——是 $x$ 与 $x_0$ 的仿射函数，无积分、无 score，完全可算。

验证 (C1)：式 (17) 作为 $x$ 的 ODE（$x_0$ 固定），右端对 $x$ 仿射，其 flow map 仍是仿射映射（仿射 ODE 的解对初值仿射）；仿射映射推高斯仍为高斯。直接核对端点之间任一 $t' \to t$ 的推送：解出 $x_t = \frac{\sigma_t}{\sigma_{t'}} x_{t'} + \left( \alpha_t - \frac{\sigma_t}{\sigma_{t'}} \alpha_{t'} \right) x_0$，把 $x_{t'} \sim \mathcal{N}(\alpha_{t'} x_0, \sigma_{t'}^2 I)$ 代入，均值 $\alpha_t x_0$、方差 $\sigma_t^2 I$，恰为式 (15)——(C1) 成立。一个细节：式 (17) 含 $1/\sigma_t$，端点 $\sigma_0 = 0$ 处奇异；实现中 $t$ 采样避开 $0$ 或截断到 $\sigma_{\min}$，无伤大雅（端点邻域的测度为零）。

顺带把 §4 的最优解写成 score 形式备用。边际场是条件场的后验平均（式 (10)），而 $u_t(x \mid x_0)$ 经式 (17) 只通过 $x_0$ 依赖条件，故

$$
u_t(x) = \frac{\dot\sigma_t}{\sigma_t}\, x + \left( \dot\alpha_t - \frac{\dot\sigma_t}{\sigma_t}\, \alpha_t \right) \mathbb{E}\left[ x_0 \mid x_t = x \right] \qquad (18)
$$

Tweedie 公式给出 $\mathbb{E}[x_0 \mid x_t = x] = \left( x + \sigma_t^2\, \nabla \log p_t(x) \right) / \alpha_t$（对式 (15)(8) 的边际求梯度、交换积分与梯度即得；与 7.3 的 score 记号 $s_t(x) = \nabla \log p_t(x)$ 一致）。代入式 (18)，$x$ 的系数化简为 $\dot\alpha_t / \alpha_t$，得

$$
u_t(x) = \frac{\dot\alpha_t}{\alpha_t}\, x + \sigma_t \left( \frac{\dot\alpha_t\, \sigma_t}{\alpha_t} - \dot\sigma_t \right) s_t(x) \qquad (19)
$$

**符号**：右端 = 线性漂移项 + score 修正项。式 (19) 说明 FM 的速度场与 7.3 的 probability-flow ODE 是同一对象的两个坐标——PF-ODE 的漂移正是「schedule 决定的线性项 + score 项」的形状。这条对应会在 §9 的统一表里正式落座。

## 6. rectified flow：线性路径与目标速度 $\varepsilon - x_0$

把 Gaussian 路径取到最简——线性插值（rectified flow，[Liu et al., 2022](../references.md)，arXiv:2209.03003，[A]；SD3 采用的正是这条路径）：

$$
\alpha_t = 1 - t, \qquad \sigma_t = t \qquad \Longrightarrow \qquad x_t = (1 - t)\, x_0 + t\, \varepsilon \qquad (20)
$$

代入式 (17)（$\dot\alpha_t = -1$、$\dot\sigma_t = 1$，$x_0$ 系数为 $-1 - (1-t)/t = -1/t$）：

$$
u_t(x \mid x_0) = \frac{x - x_0}{t} = \varepsilon - x_0 \qquad (21)
$$

末一等号用式 (20) 移项。**目标速度是常数 $\varepsilon - x_0$**：给定两端点，条件轨迹是直线、速度不随 $t$ 变化。CFM 目标式 (9) 落成

$$
L_{RF}(\theta) = \mathbb{E}_{t,\; x_0,\; \varepsilon} \left\| v_\theta\left( (1-t)\, x_0 + t\, \varepsilon,\; t \right) - (\varepsilon - x_0) \right\|^2 \qquad (22)
$$

**符号**：与 7.2 的 $L_{simple}$（式 (26)）逐项同构——抽 $t$、闭式合成 $x_t$、回归一个 $(x_0, \varepsilon)$ 的线性函数；差别只在回归目标是 $\varepsilon - x_0$ 而非 $\varepsilon$。

采样是纯 ODE 积分，方向 $t: 1 \to 0$。Euler 离散（步长 $h = 1/K$）：

$$
x^{(k+1)} = x^{(k)} - h\, v_\theta(x^{(k)}, t_k), \qquad t_k = 1 - k h,\quad k = 0, \ldots, K-1,\quad x^{(0)} = \varepsilon \sim \mathcal{N}(0, I) \qquad (23)
$$

**符号**：$x^{(k)}$：第 $k$ 步的状态（上标为步数序号，与时间下标区分——初值 $x^{(0)} = \varepsilon$ 即噪声端 $x_1$，终态 $x^{(K)}$ 落在数据端 $t = 0$）；每步一次网络评估，共 $K$ 次 NFE。确定性采样——给定 $x_1 = \varepsilon$，输出唯一；更高阶求解器（Heun 等，7.4）同骨架换更新式。

教学用参考实现（非优化实现；对照锚点：FM 官方库 [facebookresearch/flow_matching](../references.md) 的 `flow_matching/path/` 提供条件路径、`loss/` 提供 CFM 损失、`solver/` 提供 ODE 积分器，[A]）：

```python
import torch

def rf_loss(v_net, x0, t_sampler):
    """教学用，非优化实现。x0: [B, C, H, W]；t_sampler: 均匀或 logit-normal（§8）。"""
    B = x0.shape[0]
    t = t_sampler(B).view(B, 1, 1, 1)              # t ∈ (0, 1)
    eps = torch.randn_like(x0)                     # ε ~ N(0, I)
    x_t = (1 - t) * x0 + t * eps                   # 线性路径，式 (20)
    target = eps - x0                              # 目标速度，式 (21)
    return torch.nn.functional.mse_loss(v_net(x_t, t), target)  # L_RF，式 (22)

def rf_sample(v_net, shape, K):
    """教学用，非优化实现。Euler 积分 t: 1 → 0，共 K 次 NFE。"""
    x = torch.randn(shape)                         # 初值 x^(0) = ε（时间 t = 1 噪声端）
    h = 1.0 / K
    for k in range(K):
        t = 1 - k * h
        x = x - h * v_net(x, t)                    # 式 (23)
    return x
```

一句防误读：式 (21) 的「直线、恒速」是**条件轨迹**的性质（固定端点对 $(x_0, \varepsilon)$）。不同端点对的直线互相交叉，边际场 $u_t(x) = \mathbb{E}[\varepsilon - x_0 \mid x_t = x]$ 在交叉处取后验平均——学到的 ODE 轨迹仍然弯曲，只是比 VP 路径弯得少（§10 的账）。进一步拉直靠 reflow（用当前模型自产 $(x_0, \hat\varepsilon)$ 配对再训一轮），属 7.4 蒸馏语境。

```
条件轨迹（固定端点对：直线、恒速 ε−x₀）          边际 ODE 轨迹（后验平均：弯曲）

x₀ ────────────────► ε                           x₀ ──╮
  x₀' ──────────╱                                ╲  x₀' ──╮
              ╱ 交叉处后验混合                      ╲        ╲  绕开交叉区
        x₀'' ─╱                                     ╲ x₀'' ──╯
        t: 0 → 1                                     t: 0 → 1（采样反向 1 → 0）

训练回归的目标 = 左图各线的恒定速度                 学到的 v_θ 生成的轨迹 = 右图
```

左右两图的缝就是 §3 两点例子里「后验权重随 $x$ 变化」的几何形态：交叉越多（数据多峰、维度高），边际轨迹越弯。这也给出 reflow 的机制读法——用模型自己的 ODE 端点重新配对后，新端点对的连线天然贴近已有轨迹，交叉减少，下一轮训练出的边际场更直。

## 7. 与 DDPM $\varepsilon$-prediction 的同族映射

DDPM 的前向闭式（7.2 式 (5)(6)）$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$ 本身就是式 (15)(16) 的特例：$\alpha_t = \sqrt{\bar\alpha_t}$、$\sigma_t = \sqrt{1 - \bar\alpha_t}$（且满足 VP 约束 $\alpha_t^2 + \sigma_t^2 = 1$）。所以 diffusion 与 FM 不是两类模型，而是**同一 Gaussian 条件路径族上两种不同的 schedule 与参数化选择**。两层的换算分别如下。

**schedule 层：SNR 匹配。** 一条 Gaussian 路径的全部语义由信噪比 $\mathrm{SNR}_t = \alpha_t^2 / \sigma_t^2$ 决定（7.2 式 (9) 的连续版）。线性路径在时刻 $t$ 的 SNR 为 $(1-t)^2 / t^2$；令 DDPM 在时刻 $t'$ 的 SNR 与之相等，$\bar\alpha_{t'} / (1 - \bar\alpha_{t'}) = (1-t)^2 / t^2$，解出

$$
\bar\alpha_{t'} = \frac{(1-t)^2}{(1-t)^2 + t^2} \qquad (24)
$$

**符号**：一一对应 $t \leftrightarrow t'$；$t = 0 \leftrightarrow \bar\alpha = 1$（数据端）、$t = 1 \leftrightarrow \bar\alpha = 0$（噪声端）。SNR 相等的两个时刻上，两条路径的边际分布 $p_t$ 逐点相同——「DDPM 的 $t'$ 时刻」与「rectified flow 的 $t$ 时刻」是同一张加噪图，只是钟走得不一样。

**参数化层：目标互转。** 在 SNR 匹配的时刻，DDPM 回归 $\varepsilon$、RF 回归 $\varepsilon - x_0$，给定 $x_t$ 后二者线性互表。由式 (20) 把 $x_t$ 写成 $x_t = x_0 + t(\varepsilon - x_0)$，直接读出

$$
\hat x_0 = x_t - t\, v_\theta, \qquad \hat\varepsilon = x_t + (1 - t)\, v_\theta, \qquad v_\theta = \frac{\varepsilon_\theta - x_t}{1 - t} \qquad (25)
$$

**符号**：前两式由 $v = \varepsilon - x_0$ 与 $x_t = x_0 + tv$ 联立解出；第三式把 $\varepsilon_\theta$ 代入 $v = (\varepsilon - x_t)/(1-t)$（由 $\varepsilon - x_0 = \varepsilon - (x_t - t\varepsilon)/(1-t) = (\varepsilon - x_t)/(1-t)$）得到。前两式（$v \to (\hat x_0, \hat\varepsilon)$ 方向）只含加乘、无除法；第三式（$\varepsilon \to v$ 方向）含 $1/(1-t)$，退化点在**噪声端** $t \to 1$——该方向只用于参数化之间的离线换算，不进训练与采样的计算图，且 $t$ 采样在端点邻域本就截断（§5 的 $\sigma_{\min}$ 注记），数值行为干净。

**v-prediction 的位置。** 7.2 式 (27) 的 v-prediction（[Salimans & Ho, 2022](../references.md)，arXiv:2202.00512，[A]）$v = \sqrt{\bar\alpha_t}\, \varepsilon - \sqrt{1 - \bar\alpha_t}\, x_0$ 在 VP 路径上恰是条件速度场的缩放版。把 VP 路径的条件速度式 (17) 用 7.2 式 (28) 的反解 $x_0 = \sqrt{\bar\alpha}\, x_t - \sqrt{1-\bar\alpha}\, v$、$\varepsilon = \sqrt{1-\bar\alpha}\, x_t + \sqrt{\bar\alpha}\, v$ 代入：

$$
u_t(x \mid x_0) = \dot\alpha_t\, x_0 + \dot\sigma_t\, \varepsilon = \underbrace{(\alpha_t \dot\alpha_t + \sigma_t \dot\sigma_t)}_{=\, \frac{1}{2}\frac{d}{dt}(\alpha_t^2 + \sigma_t^2) = 0\ (\text{VP})}\, x + (\alpha_t \dot\sigma_t - \sigma_t \dot\alpha_t)\, v = \omega_t\, v \qquad (26)
$$

**符号**：$\omega_t = \alpha_t \dot\sigma_t - \sigma_t \dot\alpha_t > 0$（由 $\alpha$ 减、$\sigma$ 增立得）。VP 约束消掉了 $x$ 项，剩下速度场 = v-prediction 目标乘一个正的 schedule 标量。也就是说：**v-prediction = VP 路径上的 FM（差一个已知的逐 $t$ 权重）**；DDPM $\varepsilon$-prediction、x₀-prediction、v-prediction、rectified flow 速度回归，全部是式 (15) 同族路径上的坐标选择。SD3 选择「线性路径 + 速度回归」、DDPM 选择「VP 路径 + $\varepsilon$ 回归」，差别落在 §8 的 $t$ 加权与 §10 的轨迹几何上，而不在模型类。

## 8. $t$ 的采样分布与分辨率相关的 timestep shift（SD3）

式 (7)(22) 里 $t$ 的采样分布是一个自由旋钮：它不改最优解（逐点回归的目标不变），只改各噪声水平分到的训练样本数——是一个纯加权设计。SD3（[Esser et al., 2024](../references.md)，arXiv:2403.03206，[A]）的两个工程选择都落在这个旋钮上。

**logit-normal 采样。** 取

$$
t = \operatorname{sigmoid}(u), \qquad u \sim \mathcal{N}(m, s^2) \qquad (27)
$$

**符号**：$\operatorname{sigmoid}(u) = 1/(1 + e^{-u})$；$m, s$ 控制钟形中心与宽度；SD3 主模型取 $m = 0$、$s = 1$。变量替换得 $t$ 的密度：

$$
\pi_{ln}(t) = \frac{1}{s\sqrt{2\pi}}\; \frac{1}{t(1-t)}\; \exp\left( -\frac{\left( \operatorname{logit}(t) - m \right)^2}{2 s^2} \right), \qquad \operatorname{logit}(t) = \log \frac{t}{1-t} \qquad (28)
$$

**符号**：$1/(t(1-t))$ 是 sigmoid 导数的倒数（Jacobi 因子），两端趋于零处密度归零——质量集中在中间段。

为什么中间段重要，用 log-SNR 坐标看最清楚。线性路径的 $\log \mathrm{SNR}(t) = 2\log(1-t) - 2\log t = -2\,\operatorname{logit}(t)$，故式 (27) 等价于

$$
\log \mathrm{SNR} \sim \mathcal{N}\left( -2m,\; 4s^2 \right) \qquad (29)
$$

**符号**：logit-normal 的 $t$ 采样 = 在 log-SNR 轴上放一个高斯。而均匀采 $t$ 时 $\operatorname{logit}(t)$ 的密度 $\propto t(1-t)$、质量被摊到 $|logit|$ 很大的两端——即 SNR 极高（近纯数据，回归残差在绝对尺度上微小）与 SNR 极低（近纯噪声，边际场近似只指向数据均值、任务退化）的区间。图像的语义结构（构图、物体、文字）是在 $\mathrm{SNR} \approx 1$ 的中段被决定的（对照 7.2 §2.3 的频谱读法：交叉频率 $f_c$ 扫过主要能量带的过程发生在中段），把样本预算压在中段 = 把网络容量压在感知上最敏感的噪声水平。SD3 论文对均匀、logit-normal、mode 加权等方案做了对照实验，主模型采用 logit-normal$(0, 1)$（[Esser et al., 2024](../references.md)，[A]）。

**分辨率相关的 timestep shift。** 分辨率升高后，同样的 $t$ 变得「更不禁噪」——像素数增加而自然图像高度冗余，固定噪声水平下可恢复的信息总量随像素数上升，去噪任务相对变易。SD3 的应对是把采样时刻向噪声端平移：

$$
t' = \frac{s\, t}{1 + (s - 1)\, t} \qquad (30)
$$

**符号**：$s > 0$：shift 系数；$s = 1$ 恒等，$s > 1$ 把 $(0, 1)$ 内的 $t$ 推向 1（噪声端），端点 $0, 1$ 不动。SD3 论文对 $1024^2$ 分辨率采用 $s = 3$（[Esser et al., 2024](../references.md)，[A]；推理侧实现见 [Stability-AI/sd3.5](../references.md)，[A]）。

式 (30) 的形状不是任意拍的：算一下赔率（odds），$1 - t' = (1 - t)/(1 + (s-1)t)$，故

$$
\frac{t'}{1 - t'} = s \cdot \frac{t}{1 - t} \qquad \Longleftrightarrow \qquad \operatorname{logit}(t') = \operatorname{logit}(t) + \log s \qquad (31)
$$

**符号**：shift = logit 空间的常数平移。结合式 (29)（线性路径上 $\log \mathrm{SNR} = -2\,\operatorname{logit}(t)$），shift 在 log-SNR 轴上是 $\log \mathrm{SNR} \to \log \mathrm{SNR} - 2\log s$ 的整体下移——把整条训练/采样的噪声档位曲线向「更噪」平移一个由分辨率决定的常量。这与式 (27) 的 logit-normal 是同坐标系的操作：两者都在 logit/log-SNR 轴上是线性运算，复合顺序不影响解读（先采 $t$ 再平移到 $t'$ 送进训练/采样）。

## 9. diffusion↔FM 统一表

§5-§7 的换算落成一张表。两列是同一 Gaussian 条件路径族式 (15) 的两种取点；任意一行都可经 §7 的映射互相推出。

| 对象 | DDPM（VP 扩散，7.2） | rectified flow / FM（本章） |
| --- | --- | --- |
| 条件路径 $p_t(x \mid x_0)$ | $\mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I)$ | $\mathcal{N}((1-t)\, x_0,\ t^2 I)$ |
| $\mathrm{SNR}_t = \alpha_t^2/\sigma_t^2$ | $\bar\alpha_t / (1 - \bar\alpha_t)$ | $(1-t)^2 / t^2$ |
| schedule 匹配 | $\bar\alpha = (1-t)^2 / ((1-t)^2 + t^2)$，式 (24) | 同左（同一映射） |
| 回归目标 | $\varepsilon$（$\varepsilon$-prediction，$L_{simple}$） | $u = \varepsilon - x_0$（速度，$L_{RF}$） |
| 目标互转 | — | $\hat x_0 = x_t - t v_\theta$、$\hat\varepsilon = x_t + (1-t) v_\theta$，式 (25) |
| 边际场结构 | 式 (19)：$\frac{\dot\alpha}{\alpha} x + \sigma(\frac{\dot\alpha\sigma}{\alpha} - \dot\sigma)\, s_t(x)$（= PF-ODE，7.3） | 同式 (19)，linear 特例 $u_t = -\frac{1}{1-t}(x + t\, s_t(x))$ |
| 采样 |  ancestral（随机）或 DDIM/ODE solver（7.4） | 纯 ODE 积分 $t: 1 \to 0$，式 (23) |
| 精确似然 | PF-ODE + trace 公式（7.3） | 同一 trace 公式，式 (6) |
| $t$ 采样分布 | 均匀（$L_{simple}$ 口径） | logit-normal$(0,1)$ + shift，式 (27)(30)（SD3） |
| 训练是否需积分 | 否（闭式 $x_t$ 直达） | 否（simulation-free，同左） |

读表要点：「扩散 vs FM」之争在模型类层面是空的——统一在式 (15)；真实的差异集中在三处可操作的设计上：schedule 形状（VP vs 线性）、回归目标的坐标（$\varepsilon$ vs 速度）、$t$ 的加权（均匀 vs logit-normal + shift）。SD3 = 线性路径 + 速度回归 + logit-normal/shift + MMDiT 主干（7.6），正是把这张表右列取全的系统实现。

统一表管不到的两件事也登记一下，防过度外推：(i) **逐 $t$ 的损失加权**是与参数化正交的又一个自由度——7.2 式 (24)(25) 的 SNR 权重、本章 §8 的 $t$ 采样密度、以及回归目标坐标三者共同决定有效权重，换坐标不改最优解但改各噪声水平的梯度量级分配；(ii) **guidance 不在此表**：CFG 作用在 score/$\varepsilon$ 层面（7.4 推导），经式 (19)(25) 的换算可原样搬进速度场坐标，SD3 系统侧的做法见 7.7。

## 10. FM 的工程优势：直轨迹 → 少步求解 → 蒸馏友好

把式 (23) 的 Euler 步与精确解比较。ODE $\dot x = v(x, t)$ 的单步局部截断误差为

$$
x(t - h) - \left( x(t) - h\, v(x(t), t) \right) = \frac{h^2}{2}\, \ddot x(\xi) = \frac{h^2}{2}\left( \partial_t v + (v \cdot \nabla)\, v \right)(\xi) \qquad (32)
$$

**符号**：$\ddot x$：轨迹加速度的物质导数（沿轨迹对 $t$ 的全导数）；$\xi$：区间内某点。误差由轨迹弯曲程度控制——直线轨迹（$\ddot x = 0$）上 Euler 精确，弯得越厉害同样 $h$ 误差越大。这是「少步 solver」的全部几何门槛：NFE 下界 ≈ 轨迹总长 / 单步可容弦长，而可容弦长随曲率收紧。

rectified flow 在这条账上的优势来自目标结构：条件目标 $\varepsilon - x_0$ 对每个端点对是常向量（式 (21)），网络学到的边际场是它们的后验平均——弯曲只来自端点对的交叉混合，而不像 VP 路径那样额外叠一层 schedule 自身的几何（VP 路径的轨迹在数据端附近明显收束弯曲；EDM 对这类 schedule 几何的系统梳理见 7.3）。rectified flow 原论文的实验结论是同一匹配设置下线性路径可用更少步数达到同等样本质量（[Liu et al., 2022](../references.md)，[A]），且每轮 reflow（用当前模型的 ODE 轨迹端点重新配对训练）进一步拉直轨迹——蒸馏语境归 7.4。

蒸馏友好是直轨迹的直接推论：7.4 的 consistency / DMD 族要让学生网络匹配教师 ODE 的轨迹或端点，教师轨迹越接近直线，少步学生的逼近误差越小（式 (32) 的同一曲率账，学生不过是把 $h$ 取得更大的 solver）。加上确定性采样（给定 $x_1$ 输出唯一，无 ancestral 噪声项），训练-蒸馏-推理全链路共享一条 ODE，这是 FM 框架在系统工程上的完整闭环。

确定性还附带一条操作面收益：ODE 采样是一个可逆映射（沿同一速度场把 $t$ 从 $0$ 积回 $1$ 即反演），真实图像可以精确映回噪声端再编辑重采——这与 DDIM $\sigma = 0$ 的反演（7.4）是同一性质在两个参数化下的呈现；ancestral 采样没有这条性质（随机项使轨迹不可逆推）。

最后把训练成本对齐记账：FM 训练是 simulation-free 的——每样本一次闭式合成 $x_t$（式 (20)）+ 一次前向，与 DDPM 的 $L_{simple}$ 逐项同价；§2 的 trace 公式只在需要精确似然评估时才出场。CNF 的表达能力（任意网络的 ODE 路径）留下，CNF 的训练账单（每步散度估计）扔掉——这是 FM 相对它自己数学前身的位置。

## 总结

公式清单（编号为章内序号）：

| 对象 | 公式 | 要点 |
| --- | --- | --- |
| CNF 定义 | $\dot\psi_t = u_t(\psi_t)$，式 (1)；$p_t = [\psi_t]_{*} p_0$，式 (2) | 生成 = ODE 的 pushforward，$t = 1$ 噪声端 |
| 连续性方程 | $\partial_t p_t + \nabla \cdot (p_t u_t) = 0$，式 (3) | 密度 ↔ 速度场的桥梁 |
| trace 公式 | $\log p_0(x) = \log p_1(\psi_1(x)) + \int_0^1 \nabla \cdot u_t\, dt$，式 (6) | 精确似然；散度 × 积分步数 = 维度税 |
| FM 目标 | $L_{FM} = \mathbb{E}\|v_\theta - u_t\|^2$，式 (7) | 边际场不可解（混合后验平均，式 (10)） |
| conditional FM 定理 | $\nabla L_{FM} = \nabla L_{CFM}$，式 (11)(14) | 关键步：内积线性 → 期望交换 |
| Gaussian 路径闭式 | $u_t(x \mid x_0) = \frac{\dot\sigma}{\sigma} x + (\dot\alpha - \frac{\dot\sigma}{\sigma}\alpha) x_0$，式 (17) | 仿射、无 score |
| 速度场 = 漂移 + score | 式 (19) | 与 PF-ODE（7.3）同对象 |
| 线性路径 | $x_t = (1-t) x_0 + t\varepsilon$，式 (20)；目标 $\varepsilon - x_0$，式 (21) | 条件轨迹直线、恒速 |
| RF 训练/采样 | $L_{RF}$ 式 (22)；Euler $x \leftarrow x - h v_\theta$，$t: 1 \to 0$，式 (23) | 确定性、$K$ 次 NFE |
| DDPM↔FM 映射 | $\bar\alpha = (1-t)^2/((1-t)^2 + t^2)$，式 (24)；$\hat x_0 = x_t - t v$、$\hat\varepsilon = x_t + (1-t) v$，式 (25) | 同族两坐标；v-prediction = VP 路径 FM（式 (26)） |
| logit-normal | $t = \operatorname{sigmoid}(u)$，$u \sim \mathcal{N}(0, 1)$，式 (27)(28) | = log-SNR 上的高斯，式 (29)；压中段 |
| timestep shift | $t' = st / (1 + (s-1)t)$，式 (30) | 分辨率升 → 向噪声端移；SD3 $1024^2$ 取 $s = 3$ |
| 少步的几何账 | 局部误差 $= \frac{h^2}{2}(\partial_t v + (v\cdot\nabla)v)$，式 (32) | 直轨迹 → 大步长 → 蒸馏友好 |

## 练习

1.（推导）补全式 (5) 的矩阵引理：对可逆 $J(t)$，证明 $\frac{d}{dt} \log |\det J| = \operatorname{tr}(J^{-1} \frac{dJ}{dt})$（提示：对 $\det J = \exp(\operatorname{tr} \log J)$ 求导，或用 $\det(I + \delta M) = 1 + \delta\, \operatorname{tr} M + O(\delta^2)$ 做一阶展开）；再由式 (4) 推出式 (5)，并说明为什么式 (6) 的积分符号是「加散度」而非「减散度」（注意 $\log p_t = \log p_0 - \log|\det J_t|$ 的符号链）。
2.（推导）证明 CFM 与 FM 的损失差恰为条件速度的方差：$L_{CFM}(\theta) - L_{FM}(\theta) = \mathbb{E}_{t, x \sim p_t}\left[ \mathbb{E}\left[ \|u_t(x \mid x_0) - u_t(x)\|^2 \mid x_t = x \right] \right]$（提示：对 $\|v - u_c\|^2$ 在条件期望下用「均值 + 偏差」分解 $u_c = u_t(x) + (u_c - u_t(x))$，交叉项归零）。由此说明：(a) 差距与 $\theta$ 无关（定理的另一证法）；(b) 为什么 $L_{CFM}$ 的数值不能跨路径比较。
3.（计算）logit-normal 与 shift 的网格账：(a) 由式 (28) 证明 $m = 0, s = 1$ 时 $\pi_{ln}(t)$ 关于 $t = 1/2$ 对称，并算 $\pi_{ln}(0.5)$、$\pi_{ln}(0.1)$、$\pi_{ln}(0.9)$ 的相对比例，与均匀分布对比中段权重；(b) 验证式 (29)：由 $\log \mathrm{SNR} = -2\,\operatorname{logit}(t)$ 推出其分布参数；(c) 取 $s = 3$，计算 $t' = st/(1 + (s-1)t)$ 在 $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ 的值，画（或描述）平移前后 $t$ 网格的偏移方向。
4.（代码）教学用，非优化实现：二维 two-moons 数据上的 rectified flow。(a) 用 §6 的 `rf_loss` 训练一个小型 MLP（$t$ 经 sinusoidal embedding 拼接输入），$t$ 均匀采样；(b) 用 `rf_sample` 分别取 $K \in \{1, 2, 8, 64\}$ 采样，与数据散点对比——观察 $K = 1$ 的样本落在何处（提示：直线外推的端点偏差 = 边际轨迹的弯曲量）；(c) 把 $t$ 采样换成 logit-normal$(0, 1)$（式 (27)），固定 $K = 8$ 比较样本质量；(d) 选做：实现一轮 reflow——用 (a) 的模型从 $\varepsilon$ 生成 $\hat x_0$，以 $(\hat x_0, \varepsilon)$ 为新端点对重训，再比较 $K = 1$。

## 参考

- [Lipman et al., 2022](../references.md)：Flow Matching for Generative Modeling，arXiv:2210.02747，[A]（key: flow-matching）——式 (7)(9)(10)(11)(14)：FM 目标、conditional FM 定理与边际速度场构造
- [Liu et al., 2022](../references.md)：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow，arXiv:2209.03003，[A]（key: rectified-flow）——式 (20)(21)(22)：线性路径、目标速度 $\varepsilon - x_0$；reflow 与少步行为（蒸馏侧见 7.4）
- [Esser et al., 2024](../references.md)：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis，arXiv:2403.03206，[A]（key: sd3）——式 (27)(28)(30)：logit-normal $t$ 采样与分辨率相关 timestep shift；系统侧 recap 见 7.6/7.7
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——同族映射的 $\varepsilon$-prediction 一侧（7.2 完整推导）
- [Salimans & Ho, 2022](../references.md)：Progressive Distillation for Fast Sampling of Diffusion Models，arXiv:2202.00512，[A]（key: progressive-distillation）——v-prediction 原始出处（式 (26) 的桥）
- [Song et al., 2020](../references.md)：Score-Based Generative Modeling through Stochastic Differential Equations，arXiv:2011.13456，[A]（key: score-sde）——probability-flow ODE 与精确似然（式 (19) 的对应物，7.3）
- [Kingma & Dhariwal, 2018](../references.md)：Glow: Generative Flow with Invertible 1x1 Convolutions，arXiv:1807.03039，[A]（key: glow）——离散 change of variables 对照（7.1）
- [facebookresearch/flow_matching](../references.md)：FM 官方库（`flow_matching/path/`、`loss/`、`solver/`），[A]（key: flow-matching-repo）——§6 代码对照
- [Stability-AI/sd3.5](../references.md)：SD3.5 官方推理代码，[A]（key: sd3.5-repo）——shift 与采样侧实现对照







