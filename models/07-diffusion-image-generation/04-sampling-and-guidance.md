# 7.4 采样器与 Guidance（Samplers and Guidance）

> 本章回答两笔账：一是 7.2 留下的 1000 次 NFE 怎么砍——DDIM 的非马尔可夫前向族推导、DDIM σ=0 = PF-ODE 指数积分的精确证明、DPM-Solver 的高阶展开 outline；二是采样预算固定时质量怎么换——classifier guidance 的 Bayes 展开、CFG 的完整推导（隐式分类器与 log-likelihood tilt）、失效机制（score norm 膨胀）与缓解（dynamic thresholding / rescaled CFG），最后是把「步数」与「引导强度」双双蒸进单模型的蒸馏族（progressive distillation / consistency / LCM / reflow / DMD）| 前置：7.2（ε-参数化、$L_{simple}$、后验 $\tilde\beta_t$、ancestral 采样）、7.3（PF-ODE 式 (20)(21)、score↔ε 换算式 (9)、Tweedie 式 (10)、EDM Heun）、7.5（rectified flow 线性路径——reflow 一节） | 锚点：DDIM [A]、classifier guidance [A]、CFG [A]、consistency models [A]、LCM [A]、DMD/DMD2 [A]、DPM-Solver [A]、progressive distillation [A]、rectified flow [A]、common-diffusion [A]、Imagen [B]；ermongroup/ddim [A]、openai/consistency_models [A]、tianweiy/DMD2 [A]

## 1. 两条战线与一个共同接口

7.2 §4.4 的采样算法先把账摆上桌面：

```
DDPM ancestral 采样（7.2 §4.4，T = 1000 步串行，随机）
─────────────────────────────────────────────────────────
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    ε_θ = 网络前向(x_t, t)                        # 1 次 NFE
    x̂_0 = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t            # Tweedie，7.3 式 (10)
    z ~ N(0, I)（t = 1 时 z = 0）
    x_{t-1} = (1/√α_t)(x_t − β_t/√(1−ᾱ_t)·ε_θ) + σ_t·z
return x_0
```

**符号**：NFE（network function evaluations）：采样全程的网络前向次数——本章统一计账单位（约定页口径）；$\sigma_t$：7.2 §5.2 的两个固定选择之一。

1000 次串行前向就是原生 DDPM 的全部生成成本，而且它是串行的——不能并行、不能摊销。本章的全部内容是把这 1000 压到 100、10、4、1 的四条技术路线，外加一条不降成本反而翻倍（每步 2 次 NFE）却决定工业质量的轴线：

| 路线 | 动什么 | 不动什么 | 代表 | 节 |
| --- | --- | --- | --- | --- |
| 换离散化 | 采样算法 | 模型与动力学 | DDIM 跳步 | §2 |
| 换求解器 | PF-ODE 的数值积分阶数 | 模型 | DPM-Solver | §3 |
| 加引导 | 每步的有效 score | 步数 | classifier guidance / CFG | §4 |
| 换模型 | 学生网络直接学长跳 | 推理时的教师 | 蒸馏族 | §5 |

**符号**：「长跳」：学生一次前向走教师多步的路程。

四条路线共享同一个接口：一切算法都只通过网络评估 $\varepsilon_\theta(x_t, t)$（或条件版 $\varepsilon_\theta(x_t, t, c)$）接触模型。这个接口事实是整章的组织原则——求解器把它当黑盒向量场（§2-§3），guidance 在接口外侧做向量场的线性组合（§4），蒸馏把接口的组合结果反向蒸进一个新接口（§5）。下游系统（7.6 的 latent 化、7.7 的 T2I 系统）不换这个接口，只换 $x$ 住在哪个空间。

一个贯穿全章的判别准则先记下来：**凡是只依赖边缘分布 $q(x_t \mid x_0)$ 的训练目标，都对前向链的联合结构不敏感**。$L_{simple}$（7.2 式 (26)）的期望只经过 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon$ 这一个边缘采样式，从没碰过转移核 $q(x_t \mid x_{t-1})$。这个观察是 DDIM 的立足点（§2），也是蒸馏可行的一半原因（§5）。

## 2. DDIM：同一批边缘分布的一族前向过程

### 2.1 关键观察：训练目标钉不住联合分布

DDIM（[Song et al., 2021](../references.md)，arXiv:2010.02502，[A]）的出发点是一个计数论证。$L_{simple}$ 对每个 $t$ 分别回归，只见到边缘 $q(x_t \mid x_0)$；$T$ 个边缘分布完全不能决定 $T$ 步链的联合分布 $q(x_{1:T} \mid x_0)$——联合的自由度远大于边缘。DDPM 选的马尔可夫链（7.2 式 (1)）只是满足边缘约束的无穷多联合分布之一。**换一个联合族，只要边缘不变，训练好的 $\varepsilon_\theta$ 就照样有效**——而它诱导出的反向采样过程可以完全不同，包括确定性的。

把这个观察落成构造：不再先给前向转移再推边缘，而是反过来——先钦定全部边缘（7.2 式 (5)），再设计一族以 $(x_t, x_0)$ 为条件的「反向转移」，让整个联合分布自洽。反向转移取高斯、均值对 $x_t$ 与 $x_0$ 线性、方差自由：

$$
q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left( \sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\cdot \frac{x_t - \sqrt{\bar\alpha_t}\, x_0}{\sqrt{1 - \bar\alpha_t}},\ \sigma_t^2 I \right) \qquad (1)
$$

**符号**：$\sigma_t \ge 0$：每步的自由方差参数（一族前向过程用一个序列 $\{\sigma_t\}$ 索引）；均值第一项是「数据锚点」，第二项是「$x_t$ 中超出 $\sqrt{\bar\alpha_t} x_0$ 的部分按缩小后的噪声系数回传」——系数 $\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}$ 的构造马上由归纳证明强制出来。$\sigma_t^2 \le 1 - \bar\alpha_{t-1}$ 保证根号内非负。

联合分布定义为 $q_\sigma(x_{1:T} \mid x_0) = q_\sigma(x_T \mid x_0) \prod_{t=2}^{T} q_\sigma(x_{t-1} \mid x_t, x_0)$，其中 $q_\sigma(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T) I)$。注意生成方向读这个因子分解：给定 $x_0$ 先采 $x_T$，再依次「反向」地采 $x_{T-1}, ..., x_1$——前向族的写法用了反向的条件方向，这正是「非马尔可夫」（相对加噪方向）的来源：$x_{t-1}$ 的生成同时依赖 $x_t$ 与 $x_0$，不是只依赖 $x_{t}$ 的马尔可夫链。

### 2.2 边缘匹配引理（归纳推导）

要验证这个族合法，只需证明它对一切 $t$ 复现 DDPM 的边缘。用采样口径（7.2 §2.2 的同一约定）：设 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon$（归纳假设，$t = T$ 为基），按式 (1) 采 $x_{t-1}$：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\cdot \varepsilon + \sigma_t\, z, \qquad z \sim \mathcal{N}(0, I) \qquad (2)
$$

**符号**：关键一步是代入归纳假设后，$(x_t - \sqrt{\bar\alpha_t} x_0)/\sqrt{1-\bar\alpha_t} = \varepsilon$——式 (1) 均值里那个回传项恰好抽出 $x_t$ 携带的全部噪声。

式 (2) 右端是「确定性项 + 两个独立零均值高斯的加权和」，由独立高斯和引理（7.2 式 (4)）合成单一高斯，方差为 $(1 - \bar\alpha_{t-1} - \sigma_t^2) + \sigma_t^2 = 1 - \bar\alpha_{t-1}$。于是

$$
q_\sigma(x_{t-1} \mid x_0) = \mathcal{N}\left( \sqrt{\bar\alpha_{t-1}}\, x_0,\ (1 - \bar\alpha_{t-1})\, I \right) \qquad (3)
$$

归纳成立，对所有 $t$ 边缘一致。$\square$ 式 (1) 均值系数的构造由此被迫：$\varepsilon$ 的系数平方加 $\sigma_t^2$ 必须等于 $1 - \bar\alpha_{t-1}$，自由度只剩 $\sigma_t$ 一个——**$\{\sigma_t\}$ 是保持边缘不变的全部自由度**。

这个族的物理读法：$\sigma_t$ 是反向一步里「重新掷骰子」的强度。$\varepsilon$（$x_t$ 里已有的噪声）以系数 $\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}$ 被保留，缺口 $\sigma_t^2$ 由新鲜噪声补上。$\sigma_t = 0$ 时旧噪声全保留、不加新噪声——确定性；$\sigma_t$ 取满时旧噪声按后验比例打折、新噪声补足——回到 DDPM。

### 2.3 两个端点：$\sigma_t = \sqrt{\tilde\beta_t}$ 收回 DDPM，$\sigma_t = 0$ 得 DDIM

**端点一**：取

$$
\sigma_t^2 = \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t \qquad (4)
$$

**符号**：$\tilde\beta_t$：7.2 式 (14) 的后验方差。

代入验证均值系数的回传项：$1 - \bar\alpha_{t-1} - \sigma_t^2 = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\left( (1-\bar\alpha_t) - \beta_t \right) = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\, \alpha_t (1 - \bar\alpha_{t-1})$（用 $1-\bar\alpha_t - \beta_t = \alpha_t - \bar\alpha_t = \alpha_t(1-\bar\alpha_{t-1})$）。于是 $\varepsilon$ 系数为 $\sqrt{\alpha_t}(1-\bar\alpha_{t-1})/\sqrt{1-\bar\alpha_t}$，与 $x_0$ 系数 $\sqrt{\bar\alpha_{t-1}}$ 合起来正是 7.2 式 (15) 的后验均值 $\tilde\mu_t$——式 (1) 在此取值下就是后验 $q(x_{t-1} \mid x_t, x_0)$，前向族退回马尔可夫链，生成过程退回 DDPM ancestral（方差选择 $\tilde\beta_t$ 档）。

**端点二**：$\sigma_t = 0$，全确定。把推理时不可得的 $x_0$ 换成网络的预测 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\, \varepsilon_\theta(x_t, t))/\sqrt{\bar\alpha_t}$（Tweedie，7.3 式 (10)），式 (1) 给出 **DDIM 更新**：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\; \hat x_0 + \sqrt{1 - \bar\alpha_{t-1}}\; \varepsilon_\theta(x_t, t) \qquad (5)
$$

**符号**：$\hat x_0 = \hat x_0(x_t, t)$：当前步的去噪估计。读法：「先 extrapolate 到数据端，再按 $t-1$ 的噪声预算重新加回预测噪声」——每一步都过一遍 $x_0$ 空间这个中转站。

式 (5) 的结构值得拆一眼：右端两项是「信号方向」与「噪声方向」在以 $\hat x_0$ 为新原点的坐标系里的重新配比，配比只由 $\bar\alpha_{t-1}$ 决定。轨迹完全确定——同一个 $x_T$ 永远生成同一张图。这带来两个工程性质：(i) **可逆与插值**：噪声种子 ↔ 图像是近似双射，可以做 latent 插值与图像反演（把真实图像正向推回噪声端）；(ii) **误差全是截断**：没有随机项帮忙平均掉模型误差，步数与误差的账是纯数值分析的账，§3 的求解器军备竞赛由此而来。

一般取 $\sigma_t = \eta\, \sqrt{\tilde\beta_t}$（$\eta \in [0, 1]$）在两个端点之间滑动，DDIM 原文记 $\eta$；$\eta$ 是随机性滑杆，7.3 §7 末尾的「注噪量可调的逆时扩散族」是它的连续对应。

**跳步子序列**。DDIM 与 ancestral 的决定性差异在：式 (5) 只用 $(\bar\alpha_t, \bar\alpha_{t-1})$ 两个端点量，从不依赖「$t-1$ 是 $t$ 的相邻训练步」。于是可以任取子序列 $\{\tau_1 < \tau_2 < \cdots < \tau_S\} \subset \{1, ..., T\}$（$S \ll T$），在子序列格点上直接跳：

$$
x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\; \hat x_0 + \sqrt{1 - \bar\alpha_{\tau_{i-1}}}\; \varepsilon_\theta(x_{\tau_i}, \tau_i) \qquad (6)
$$

**符号**：$\hat x_0$ 由 $(x_{\tau_i}, \tau_i)$ 处的预测给出；一跳跨过任意多个训练步。

合法性来自 §2.1 的观察：网络是在全部 $t$ 的边缘上分别训练的，任何格点上的 $\varepsilon_\theta$ 都是那个噪声水平下的合法 score（经 7.3 式 (9) 换算）——跳步不是近似调用一个「本来只在相邻步有效」的算子，而是同一场在稀疏格点上的采样。1000 步训练出的模型直接以 $S = 50$ 或 $S = 100$ 采样，NFE 同比例下降，无需任何重训（[Song et al., 2021](../references.md)，[A]；官方实现 [ermongroup/ddim](../references.md)，[A]）。代价是每跳跨距大、式 (5) 的离散误差按跨距累积——误差账的定量形态在 §2.4 与 §3。

### 2.4 DDIM $\sigma = 0$ = PF-ODE 的一阶离散化（精确证明）

「DDIM 确定性更新是 probability-flow ODE 的一阶离散」常被当口号引用，这里用指数积分把它证成一个恒等式，并顺手得到 §3 全部高阶求解器的母体。从 7.3 的 VP 型 PF-ODE（7.3 式 (21)）出发：

$$
\frac{dx}{dt} = -\frac{1}{2}\, \beta(t) \left[ x - \frac{\varepsilon_\theta(x, t)}{\sqrt{1 - \bar\alpha(t)}} \right] \qquad (7)
$$

**符号**：$t \in [0, 1]$ 连续时间（7.3 口径），采样沿 $t: 1 \to 0$；$\bar\alpha(t) = e^{-\int_0^t \beta(s) ds}$（7.3 式 (14)）。

换坐标。取**半对数信噪比**

$$
\lambda(t) = \frac{1}{2} \log \frac{\bar\alpha(t)}{1 - \bar\alpha(t)} = \frac{1}{2} \log \mathrm{SNR}(t) \qquad (8)
$$

**符号**：$\lambda$：随 $t$ 单调递减（$t = 0$ 数据端 $\lambda \to +\infty$，$t = 1$ 噪声端 $\lambda \to -\infty$）；采样方向即 $\lambda$ 增大的方向。

由 $\frac{d\bar\alpha}{dt} = -\beta \bar\alpha$ 得 $\frac{d\lambda}{dt} = \frac{1}{2}\left( \frac{-\beta\bar\alpha}{\bar\alpha} - \frac{\beta\bar\alpha}{1-\bar\alpha} \right)$，合并得 $-\frac{\beta}{2(1-\bar\alpha)}$，与式 (7) 链式相除：

$$
\frac{dx}{d\lambda} = \frac{dx/dt}{d\lambda/dt} = (1 - \bar\alpha(\lambda))\, x - \sqrt{1 - \bar\alpha(\lambda)}\; \varepsilon_\theta(x, \lambda) \qquad (9)
$$

**符号**：$\bar\alpha(\lambda) = e^{2\lambda}/(1 + e^{2\lambda})$：式 (8) 的反解。

式 (9) 是**半线性**（semilinear）常微分方程：线性项 $(1-\bar\alpha(\lambda)) x$ 的系数不含 $x$，非线性项只有网络输出 $\varepsilon_\theta$。线性部分可闭式积分——对线性项用积分因子，需要

$$
\int (1 - \bar\alpha(\lambda))\, d\lambda = \int \frac{d\lambda}{1 + e^{2\lambda}} = -\frac{1}{2} \log\left( 1 + e^{-2\lambda} \right) = \log \sqrt{\bar\alpha(\lambda)}
$$

末步用 $1 + e^{-2\lambda} = 1/\bar\alpha(\lambda)$：故积分因子的指数部分 $= \log \sqrt{\bar\alpha}$，相差常数。于是式 (9) 在区间 $[\lambda_t, \lambda_s]$（$\lambda_s > \lambda_t$，向数据端走一步）上的**精确解**为

$$
x_s = \frac{\sqrt{\bar\alpha_s}}{\sqrt{\bar\alpha_t}}\; x_t - \sqrt{\bar\alpha_s} \int_{\lambda_t}^{\lambda_s} e^{-\lambda}\, \varepsilon_\theta(x_\lambda, \lambda)\, d\lambda \qquad (10)
$$

**符号**：$x_\lambda$：ODE 轨迹在坐标 $\lambda$ 处的真值；第一项是线性传播子（只缩放、不修方向），第二项是网络场沿轨迹的指数加权积分——全部逼近误差只可能来自这一项。

数值方法的选择只剩一个：怎么逼近积分里的 $\varepsilon_\theta$。**最粗的逼近——左端点冻结** $\varepsilon_\theta(x_\lambda, \lambda) \approx \varepsilon_\theta(x_t, t)$（常数外提）：

$$
\int_{\lambda_t}^{\lambda_s} e^{-\lambda} d\lambda = e^{-\lambda_t} - e^{-\lambda_s} = \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}} - \frac{\sqrt{1-\bar\alpha_s}}{\sqrt{\bar\alpha_s}}
$$

代回式 (10)：

$$
x_s = \frac{\sqrt{\bar\alpha_s}}{\sqrt{\bar\alpha_t}} \left( x_t - \sqrt{1-\bar\alpha_t}\, \varepsilon_\theta(x_t, t) \right) + \sqrt{1-\bar\alpha_s}\, \varepsilon_\theta(x_t, t) = \sqrt{\bar\alpha_s}\; \hat x_0 + \sqrt{1-\bar\alpha_s}\; \varepsilon_\theta(x_t, t) \qquad (11)
$$

**符号**：末步把括号认出 $\sqrt{\bar\alpha_t}\, \hat x_0$。

式 (11) 与 DDIM 更新式 (6) **逐项相同**——不是「到一阶一致」，而是恒等：DDIM $\sigma = 0$ 更新 = PF-ODE 经指数积分、网络场取 0 阶（左端点常数）逼近的步进。线性传播子是精确的，误差整块来自「一步之内把 $\varepsilon_\theta$ 当常数」，阶为 $O\big( (\lambda_s - \lambda_t)^2 \big)$ 局部 / 全局一阶。这就是「DDIM = PF-ODE 一阶离散化」的精确含义，也把 DDIM 直接钉死在数值分析框架里：**要降误差，不必换模型，只需对网络场做更高阶的展开**——§3 的全部内容。

顺带一个统一视角的注记：7.3 §7 已证 DDPM ancestral = 逆时 SDE 的一阶随机离散；本节证 DDIM = PF-ODE 的一阶确定离散。同一个 $\varepsilon_\theta$、同一族边缘、两种动力学的各一阶离散——7.3 的统一表在本章落到了可执行的算法上。

## 3. 高阶求解器：DPM-Solver 的指数积分展开

### 3.1 从冻结到 Taylor：阶数怎么买到

式 (10) 把问题规约为对 $\varepsilon_\theta(x_\lambda, \lambda)$ 在 $\lambda \in [\lambda_t, \lambda_s]$ 上做多好的插值。自然升级是把 0 阶换成 $\lambda_t$ 处的 Taylor 展开（DPM-Solver，[Lu et al., 2022](../references.md)，arXiv:2206.00927，[A]）：

$$
\varepsilon_\theta(x_\lambda, \lambda) = \sum_{k=0}^{K-1} \frac{(\lambda - \lambda_t)^k}{k!}\, \varepsilon_\theta^{(k)}(x_t, \lambda_t) + O\left( (\lambda - \lambda_t)^K \right) \qquad (12)
$$

**符号**：$\varepsilon_\theta^{(k)} = \frac{d^k \varepsilon_\theta(x_\lambda, \lambda)}{d\lambda^k}$：网络场沿 ODE 轨迹的全导数（含轨迹自身对 $\lambda$ 的依赖，不是网络对输入的偏导）。

代入式 (10)，积分号下只剩 $e^{-\lambda} (\lambda - \lambda_t)^k$ 型矩，全部闭式（反复分部积分）：$\int e^{-\lambda} (\lambda-\lambda_t)^k d\lambda = -e^{-\lambda} \sum_{j=0}^{k} \frac{k!}{j!} (\lambda-\lambda_t)^j$。于是 $K$ 阶更新式为

$$
x_s = \frac{\sqrt{\bar\alpha_s}}{\sqrt{\bar\alpha_t}}\; x_t - \sqrt{\bar\alpha_s} \sum_{k=0}^{K-1} \varepsilon_\theta^{(k)}(x_t, \lambda_t)\; \frac{1}{k!} \int_{\lambda_t}^{\lambda_s} e^{-\lambda} (\lambda - \lambda_t)^k\, d\lambda + O\left( (\lambda_s - \lambda_t)^{K+1} \right) \qquad (13)
$$

**符号**：求和是「$k$ 阶导数 $\times$ 闭式标量矩」的线性组合——高阶信息全部进入系数，不打乱更新结构。

剩下的问题是怎么拿到 $\varepsilon_\theta^{(k)}$。两条路，DPM-Solver 都用了：**(i) 多步法（multistep）**：前若干步的网络评估值缓存下来，用有限差分组合出各阶导数的估计——历史评估免费复用，不增加 NFE，这是实际部署的主力形式（每步仍 1 次网络前向）；**(ii) 单步多阶段法**：在当前步内补采中间点（类似 Runge-Kutta 的级），每步 2-3 次 NFE 换更高的阶。$K = 1$ 时式 (13) 退化为式 (11)——DDIM 是 DPM-Solver 的第 1 阶；$K = 2, 3$ 时每步误差降到 $O(\Delta\lambda^3)$、$O(\Delta\lambda^4)$ 量级。

结论性的账（出处即论文标题与摘要口径）：在 $K = 2$ 或 $3$、配合格点在 $\lambda$ 坐标下均匀或按经验规则放置时，CIFAR-10 / ImageNet 规模上 **10 次上下 NFE 即可逼近数百步 DDPM ancestral 的样本质量**，同预算下显著优于 1 阶 DDIM 跳步（[Lu et al., 2022](../references.md)，[A]，具体 FID 数字以原表为准——口径随数据集与评估协议而变，本章不跨表转录）。对照 7.3 §8 的 EDM Heun（预测-校正、每步 2 NFE、二阶）：Heun 是把同样的二阶精度买在通用 ODE 框架里，DPM-Solver 则利用扩散 ODE 的半线性结构把线性项精确积分掉、只对网络场展开——同样阶数下后者的有效精度更高，因为误差只作用于网络项而非整个向量场。

### 3.2 步数-质量曲线的形状与读法

把 NFE 当横轴、样本质量（FID 等）当纵轴，各方法的曲线形状有机制上的解释：

```
质量 ↑
 │ DDPM ancestral（随机）：步数降到 ~100 以下后误差急升
 │   （随机项在低步数下不再被平均掉，反成主误差源）
 │ DDIM η=1→0：η 越小低步数段越好——确定性把误差预算
 │   全部留给可控的截断项
 │ DPM-Solver K=1  ─╮
 │ DPM-Solver K=2 ──┤ 同一网络，阶数平移整条曲线
 │ DPM-Solver K=3  ─╯
 │ 蒸馏学生（§5）：在 1-4 步的极左端单独开一条曲线
 └───────────────────────────────────────→ NFE
        1    4    10    50    100    1000
```

**符号**：示意形状，非实测描点；各曲线交点的定量位置见各论文原表。

三个读法要点：(i) **随机性在低步数是负债**：ancestral 的每步注噪设计假设了「多步平均」，步数一砍这个假设先崩——所以低 NFE 区间清一色是确定性或近确定性方法（DDIM η 调小、solver、蒸馏）；(ii) **阶数的边际收益递减**：1→2 阶的收益远大于 2→3 阶，且高阶对网络场的光滑性更敏感（$\varepsilon_\theta^{(k)}$ 的有限差分估计在极少步数下本身有噪声），10-20 NFE 是这类方法的甜区；(iii) **再往下是另一道题**：4 步以下，任何「对同一条 ODE 做更细积分」的思路都救不回来——轨迹本身的曲率成了墙，必须换轨迹（拉直，§5.4）或换模型（蒸馏，§5.1-§5.5）。这条分界就是 §3 与 §5 的分工。

## 4. Guidance：把条件压进 score

### 4.1 classifier guidance：Bayes 展开

条件生成的目标是采 $p(x \mid y)$（$y$：类别或文本）。对它取 score，用 Bayes 展开——$p(x \mid y) = p(y \mid x) p(x) / p(y)$，取对数再对 $x$ 求梯度，$\log p(y)$ 不含 $x$ 直接消失：

$$
\nabla_x \log p(x \mid y) = \nabla_x \log p(x) + \nabla_x \log p(y \mid x) \qquad (14)
$$

**符号**：第一项：无条件 score，已有（$s_\theta$）；第二项：分类器对数似然对**输入图像**的梯度——注意不是对分类器参数的梯度。

式 (14) 给出一条不改动扩散模型的条件化路线：额外训一个**耐噪分类器** $p_\phi(y \mid x_t, t)$（在加噪样本上训练，$t$ 作输入），采样时把两项相加。带强度 $w$ 的版本 $\tilde s = s_\theta + w\, \nabla_x \log p_\phi(y \mid x_t)$ 经 score↔ε 换算（7.3 式 (9)，$s = -\varepsilon/\sqrt{1-\bar\alpha_t}$）写成 ε 口径——两边同乘 $-\sqrt{1-\bar\alpha_t}$：

$$
\tilde\varepsilon = \varepsilon_\theta(x_t, t) - w\, \sqrt{1 - \bar\alpha_t}\; \nabla_{x_t} \log p_\phi(y \mid x_t) \qquad (15)
$$

**符号**：负号是换算的自然产物（score 与 ε 反向）；$\sqrt{1-\bar\alpha_t}$：score 与 ε 之间的尺度桥。

用 $\tilde\varepsilon$ 替换 ancestral 或 DDIM 更新里的 $\varepsilon_\theta$ 即得引导采样。这条路线是 Dhariwal & Nichol 的 ADM/ADM-G 的技术核心（[Dhariwal & Nichol, 2021](../references.md)，arXiv:2105.05233，[A]；注意该文标题是 *Diffusion Models Beat GANs on Image Synthesis*，classifier guidance 是其实现这一目标的技术贡献——标题与贡献勿张冠李戴）：论文在类别条件 ImageNet 上以足够大的 $w$ 把类别条件生成质量推过同期 GAN。$w > 1$ 才有效这件事本身是一条信息——真实的 $p(x \mid y)$（$w = 1$）不够好，采样器需要的是**比真后验更尖**的分布，§4.3 给它的分布语义。

成本与隐患记三笔：(i) 要多训一个吃噪声图的分类器，且它的梯度每步进采样环——pipeline 多一个模型；(ii) 分类器梯度不保证指向数据流形：它学的是「什么特征让 $y$ 好分」，不是「什么图自然」，大 $w$ 下样本会被推向分类器决策边界的偏好方向——一种温和的 adversarial 效应；(iii) 分类器信号随 $t$ 衰减（噪声大时 $x_t$ 几乎不可分），引导力在高噪声端最弱，而那里恰恰是构图定型处。

### 4.2 CFG：隐式分类器的完整推导

classifier-free guidance（CFG，[Ho & Salimans, 2022](../references.md)，arXiv:2207.12598，[A]）把显式分类器消掉：不从外面训 $p_\phi(y \mid x)$，而是从扩散模型自身**反解**一个隐式分类器。对同一 $y$ 用 Bayes 的另一面：

$$
\nabla_{x} \log p(y \mid x) = \nabla_{x} \log p(x \mid y) - \nabla_{x} \log p(x) \qquad (16)
$$

**符号**：右端：条件 score 减无条件 score——一个「只含生成模型」的分类器梯度。

扩散模型恰好两样都能给：条件版 $\varepsilon_\theta(x_t, t, c)$（$c$ = 文本/类别条件）与无条件版 $\varepsilon_\theta(x_t, t, \varnothing)$。实现上不是两个模型：训练同一个网络，以一定概率把条件输入置空（condition dropout），一个权重集同时学到两个 score——这是 CFG 的工程前提。代入 score↔ε 换算：

$$
\nabla_{x_t} \log p(y \mid x_t) = -\frac{\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing)}{\sqrt{1 - \bar\alpha_t}} \qquad (17)
$$

**符号**：$\varnothing$：空条件（置空的条件 embedding）。

把式 (17) 插回 classifier guidance 的 ε 形式（式 (15)），$\sqrt{1-\bar\alpha_t}$ 与负号双双相消：

$$
\tilde\varepsilon = \varepsilon_\theta(x_t, t, \varnothing) + w\, \big( \varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing) \big) \qquad (18)
$$

**符号**：$w$：引导强度（guidance scale）；$\tilde\varepsilon$：替换采样更新里的 $\varepsilon_\theta$。

式 (18) 是当前 T2I 采样环的标准组件。三个端点读法：$w = 0$ 退回无条件采样；**$w = 1$ 退回纯条件采样**（$\tilde\varepsilon = \varepsilon_\theta(x_t, t, c)$）——注意 classifier guidance 里 $w = 1$ 对应真后验，CFG 继承了同一标定；$w > 1$ 沿「条件方向」外推。工业典型区间 $w \in [2, 15]$（经验注记，约定页口径），远超 1——条件采样在实践中从来不是采真后验，而是采一个刻意加尖的分布。

等价的差分写法把机制看得更清：$\tilde\varepsilon = (1 - w)\, \varepsilon_\theta(\varnothing) + w\, \varepsilon_\theta(c)$——两个 score 的仿射组合，系数和为 1 但一正一负（$w > 1$ 时无条件项系数为负），不是凸组合。成本账：每步两次网络前向（条件与无条件各一），可拼成一个 batch 维翻倍的批——**NFE 计 2**，墙钟时间通常远小于 2 倍（GPU 并行吃掉一部分）。

### 4.3 $w$ 的数学含义：log-likelihood tilt 与失效机制

$w$ 到底把采样分布改成了什么？把 CFG 的 score 反推回分布：式 (18) 对应的有效 score 是 $\nabla \log p(x) + w\, \nabla \log p(y \mid x)$（经式 (16)(17) 还原），它正是分布

$$
q_w(x) \propto p(x)\; p(y \mid x)^{w} \qquad (19)
$$

**符号**：$q_w$：tilt 后的目标分布；$\propto$ 隐藏归一化常数。

的 score。所以 $w$ 的精确语义是**对分类器对数似然的指数倾斜（log-likelihood tilt）**：在无条件分布上把 $p(y \mid x)$ 高的区域密度乘上 $p(y \mid x)^{w-1}$ 的放大因子。$w = 1$ 复原 Bayes 后验 $p(x \mid y)$；$w > 1$ 是「比后验更自信」的分布——等价于把分类器项降到温度 $1/w$。text-to-image 里 $p(y \mid x)$ 极不平滑（大多数图完全不匹配 prompt），tilt 把质量集中到高度匹配的尖锐区域——这正是 $w$ 买 fidelity/对齐的机制，也是它付多样性账的同一个机制。

失效模式的机制拆解（过饱和与多样性下降不是玄学，是范数账）。记 $\Delta = \varepsilon_\theta(c) - \varepsilon_\theta(\varnothing)$，则 $\tilde\varepsilon = \varepsilon_\theta(\varnothing) + w \Delta$，其范数

$$
\| \tilde\varepsilon \|^2 = \| \varepsilon_\theta(\varnothing) \|^2 + 2w\, \langle \varepsilon_\theta(\varnothing), \Delta \rangle + w^2 \| \Delta \|^2 \qquad (20)
$$

**符号**：右端随 $w$ 二次增长，主导增长项是 $w^2 \|\Delta\|^2$——**score norm 随 $w$ 近似线性膨胀**。

后果分两路。(i) **过饱和/出界**：$\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\, \tilde\varepsilon)/\sqrt{\bar\alpha_t}$ 被膨胀的 $\tilde\varepsilon$ 推出数据范围 $[-1, 1]$，像素挤到边界，画面出现大面积饱和色块与过锐对比。(ii) **分布失配累积**：网络只在 $q(x_t \mid x_0)$ 的样本上训练过；膨胀的 $\tilde\varepsilon$ 产生的 $x_{t-1}$ 落在训练分布之外，下一步的 $\varepsilon_\theta$ 在未见输入上评估，误差沿链复合——步数越少（跳步越大）复合越快，这就是为什么大 $w$ 必须与低 NFE 一起调。(iii) **多样性塌缩**：式 (19) 的 tilt 按构造压低 $p(y \mid x)$ 非峰值区的密度——同一 prompt 的多次采样彼此趋同。

### 4.4 缓解：thresholding 与 rescale

两招都作用在每步的 $\hat x_0$ 或 $\tilde\varepsilon$ 上，把样本拉回数据流形附近而不动 $w$ 的对齐收益。

**static thresholding**（基线）：$\hat x_0 \leftarrow \mathrm{clip}(\hat x_0, -1, 1)$。问题：clip 是逐像素截断，大 $w$ 下大量像素同时贴边，画面被拉向灰平——把出界变成了另一种失真。

**dynamic thresholding**（Imagen，[Saharia et al., 2022](../references.md)，arXiv:2205.11487，[B]——Google 论文无官方代码，公式引论文）：每步先看 $\hat x_0$ 自身的幅度分布，取分位数 $s = \mathrm{percentile}(|\hat x_0|, p)$（$p$ 取高档位），仅当 $s > 1$ 时整体压缩：

$$
\hat x_0 \leftarrow \frac{\mathrm{clip}(\hat x_0, -s, s)}{s} \qquad (21)
$$

**符号**：分位数自适应于当前样本当前步——绝大多数步 $s \approx 1$ 不动手，只有真正出界的步被按比例压回 $[-1, 1]$，且无贴边堆积（除以 $s$ 保持相对对比度）。

**rescaled CFG**（[Hang et al., 2023](../references.md)，arXiv:2305.08891，[A]）：把引导后的预测的整体标准差拉回与纯条件预测一致，再与原引导输出混合：

$$
\tilde\varepsilon_{\mathrm{rs}} = \tilde\varepsilon \cdot \frac{\mathrm{std}\left( \varepsilon_\theta(x_t, t, c) \right)}{\mathrm{std}\left( \tilde\varepsilon \right)}, \qquad \tilde\varepsilon' = \phi\, \tilde\varepsilon_{\mathrm{rs}} + (1 - \phi)\, \tilde\varepsilon \qquad (22)
$$

**符号**：$\mathrm{std}$：对该步输出张量取的标量标准差；$\phi \in [0, 1]$：rescale 强度（$\phi = 0$ 不动，$\phi = 1$ 全 rescale）。

两招的共同点：都只修**尺度**不修方向——$w$ 买来的「条件方向」$\Delta$ 原样保留，膨胀的只是被除回去的范数。这是「保对齐、降失真」能用如此便宜的手术实现的原因。

### 4.5 guidance 蒸馏：把 $w$ 蒸进单模型

CFG 的 2 次 NFE/步在工业部署里是硬伤。解法思路直白：把式 (18) 的双前向输出当作教师信号，训一个把 $w$ 当输入的单网络：

$$
L_{\mathrm{gd}}(\psi) = \mathbb{E}_{t, x_0, \varepsilon, c, w} \left\| \varepsilon_\psi(x_t, t, c, w) - \left[ \varepsilon_\theta(x_t, t, \varnothing) + w\, \big( \varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \varnothing) \big) \right] \right\|^2 \qquad (23)
$$

**符号**：$\varepsilon_\theta$：冻结的条件教师；$w$ 在每样本上从一个区间采样，作为学生的额外条件输入——推理时 $w$ 仍是自由旋钮，但每步只付 1 次前向。

这能把 CFG 的边际成本降回无条件模型水平，是当前 T2I 部署的默认做法；它同时是 §5 各蒸馏路线的标准组件——LCM 的 latent 一致性蒸馏里，教师侧就是带 CFG 的增广 PF-ODE，$w$ 同样作为学生输入蒸入（[Luo et al., 2023](../references.md)，arXiv:2310.04378，[A]）。注意式 (23) 的教师输出按定义在数据流形的「tilt 后」位置上——学生直接学习 tilt 分布的 score，§4.3 的范数膨胀随之内化，thresholding/rescale 在蒸馏后仍然常用。

## 5. 蒸馏族：把步数压进权重

§3 的求解器在 10 NFE 以下撞上轨迹曲率的墙。蒸馏换思路：不再把教师当向量场去更细地积分，而是训一个学生直接输出教师的**多步合成结果**——长跳的逼近误差被一次性训练掉，而不是在推理时逐步累积。

### 5.1 progressive distillation：T → T/2 的迭代减半

progressive distillation（[Salimans & Ho, 2022](../references.md)，arXiv:2202.00512，[A]；v-prediction 的原始出处，见 7.2 §5.1）的基本单元是**步数减半**：学生 $\varepsilon_\psi$ 用一次前向走教师 $\varepsilon_\theta$ 的两步 DDIM（σ=0）。

目标构造：给定 $x_t$（教师时间网格上的偶数点），教师走两步 $t \to t-1 \to t-2$ 得 $x''_{t-2}$；学生要从 $x_t$ 一步到 $t-2$。学生的一步 DDIM（式 (6)，$t \to t-2$）由它的 $\hat x_0^\psi$ 唯一决定，于是「学生该输出什么」可以精确反解——令学生落点等于 $x''_{t-2}$，解出目标数据预测：

$$
\tilde x_0 = \frac{x''_{t-2}\, \sqrt{1-\bar\alpha_t} - \sqrt{1-\bar\alpha_{t-2}}\; x_t}{\sqrt{\bar\alpha_{t-2}}\, \sqrt{1-\bar\alpha_t} - \sqrt{\bar\alpha_t}\, \sqrt{1-\bar\alpha_{t-2}}} \qquad (24)
$$

**符号**：推导：把 DDIM 一步式 $x_s = \sqrt{\bar\alpha_s} \hat x_0 + \sqrt{1-\bar\alpha_s}\, (x_t - \sqrt{\bar\alpha_t} \hat x_0)/\sqrt{1-\bar\alpha_t}$ 对 $\hat x_0$ 解出即得；分母为正（$s < t$ 时 $\bar\alpha_s > \bar\alpha_t$）。论文实现用 v-prediction 参数化（式 (25) 的等价换算，7.2 §5.1）——原因正是 7.2 §5.1 的账：$\hat x_0$ 换算含 $1/\sqrt{\bar\alpha_t}$ 除法，蒸馏要跨全部 $t$ 回归目标，v 的无除法换算数值更稳。

损失即学生对 $\tilde x_0$ 的回归（等价地对 $\tilde v$ 回归）。一轮训练完成 $T \to T/2$；然后把学生拷贝为新教师，重复减半：

$$
T \to T/2 \to T/4 \to \cdots \to T / 2^{k} \qquad (25)
$$

**符号**：$k$：减半轮数，每轮一次完整蒸馏训练；从 $T = 1024$ 到 4 步需 $k = 8$ 轮。

机制读法：每一轮学生只需学「教师的二步合成」——目标始终来自同一个高质量教师链，误差不在轮次间爆炸（论文报告 CIFAR-10 上一直减半到 4 步、样本质量接近全步教师，再往下开始掉；具体 FID 见原表）。代价同样清楚：**每轮一次完整训练**，$k$ 轮的累计训练成本远超单次教师训练；且学生绑死一个特定教师与 schedule。这是「蒸馏换 NFE」路线的总账本原型——后面三个家族都在压缩这笔训练账或把目标换得更直接。

### 5.2 consistency models：ODE 轨迹的自一致性

一致性模型（consistency model，[Song et al., 2023](../references.md)，arXiv:2303.01469，[A]；官方实现 [openai/consistency_models](../references.md)，[A]，CT/CD loss 走读锚点 `cm/losses.py`、`cm/karras_diffusion.py`）把蒸馏目标从「匹配教师步进」换成 PF-ODE 的一条几何性质：**同一条 ODE 轨迹上的所有点，应当映射到同一个终点**。定义一致性函数 $f(x_t, t)$ 为「轨迹上点 $(x_t, t)$ 沿 PF-ODE 积到 $t = 0$ 的端点」，则它满足**自一致性**：

$$
f(x_t, t) = f(x_{t'}, t') \qquad \forall\, t, t' \in [0, 1] \qquad (26)
$$

**符号**：等式的成立条件是 $t$ 与 $t'$ 两处取自同一条轨迹；轨迹：PF-ODE（7.3 式 (20)）从噪声端出发的解曲线。

加上边界条件 $f(x_0, 0) = x_0$（数据点是自己轨迹的终点），$f$ 就被唯一定义。网络参数化把边界条件做成恒等式而非软约束（EDM 式外壳，7.3 §8 同族）：

$$
f_\theta(x, t) = c_{\mathrm{skip}}(t)\, x + c_{\mathrm{out}}(t)\, F_\theta(x, t), \qquad c_{\mathrm{skip}}(0) = 1,\ c_{\mathrm{out}}(0) = 0 \qquad (27)
$$

**符号**：$c_{\mathrm{skip}}, c_{\mathrm{out}}$：可微系数函数；$t \to 0$ 时 $f_\theta \to x$ 自动成立，网络 $F_\theta$ 不用学「复制输入」。

训练 = 对式 (26) 的回归，两条路线：

- **CD（consistency distillation，要教师）**：用冻结教师沿 PF-ODE 走一个求解器步——从 $x_{t_{n+1}}$ 到 $\hat x^\Phi_{t_n}$（$\Phi$：Euler/Heun 单步，教师 score 驱动）。这两点近似在同一轨迹上，于是要求学生的两处输出一致：

$$
L_{\mathrm{CD}}(\psi) = \mathbb{E}\left[ \lambda(t_n)\, d\left( f_\psi(x_{t_{n+1}}, t_{n+1}),\; f_{\psi^{-}}\left( \hat x^\Phi_{t_n}, t_n \right) \right) \right] \qquad (28)
$$

**符号**：$\psi^{-}$：学生参数的 EMA（指数滑动平均，6.3 术语）版本——目标端用 stop-gradient 的慢拷贝，防塌缩；$d$：距离度量（论文用 L2 与 LPIPS 两档）；$\lambda(t_n)$：步加权。

- **CT（consistency training，免教师）**：训练时本来就持有 $(x_0, \varepsilon)$ 对，教师那一步的「沿 ODE 走一格」可用已知噪声直接构造估计（把教师 score 换成该样本的已知条件方向），不需要任何预训练扩散模型。论文证明该替代的梯度在步长极限下与 CD 一致——免教师的代价是起步质量受限于估计偏差，CD 一般质量更高。

采样：1 步即 $x_0 = f_\psi(x_1, 1)$；要更好就**中途重启**：积到某个 $t' > 0$ 的 $f$ 输出重新加噪到 $t'$ 再映射一次——2 步、4 步档位免费获得，步数-质量权衡在推理时滑动，不需要重训。与 progressive distillation 的本质区别：PD 的学生只承诺「在这一对格点间等于教师」，CM 承诺「在整条轨迹上自洽」——后者一次性把全部时间格织进一个性质里，跳步的自由度（任意 $t' \to 0$）由此而来。

### 5.3 LCM：latent 空间 + CFG 的同步蒸馏

LCM（latent consistency model，[Luo et al., 2023](../references.md)，arXiv:2310.04378，[A]）把 CD 搬到 latent 扩散（7.6）的空间：$x$ 换成 VAE latent $z$，教师换成带 CFG 的增广 PF-ODE——§4.5 的 guidance 蒸馏在此与一致性蒸馏合流：学生 $f_\psi(z_t, t, c, w)$ 以 $w$ 为输入，回归目标是教师「CFG 双前向 + 求解器步」的合成输出。一次蒸馏同时蒸掉两样东西：步数（1000 → 2-4）与引导的双倍 NFE。标题口径即「高分辨率 few-step」：Stable Diffusion 规模的文生图在 2-4 步内出图。工程含义：一次蒸馏同时去掉「步数」与「引导」两份串行成本——这是实时 T2I 部署把一致性模型接在基础扩散模型下游的直接机制原因。

### 5.4 reflow：先把轨迹拉直，再谈少步

7.5 §10 的曲率账在这里兑现：rectified flow（[Liu et al., 2022](../references.md)，arXiv:2209.03003，[A]）的线性路径 $x_t = (1-t) x_0 + t\varepsilon$、目标速度 $\varepsilon - x_0$ 是**条件**直线；不同端点对的直线相交，学到的边际场在交点处平均——ODE 轨迹仍弯。**reflow** 用模型自身修这个耦合：用当前模型的确定性 ODE 生成配对 $(\hat x_0, \varepsilon)$（从 $\varepsilon$ 积到数据端），再以这批配对重新训练——新一轮的条件路径连接的是「真实会被映射到一起」的端点对，交叉减少，轨迹变直；迭代执行即 $k$-rectified flow：

$$
\varepsilon \sim \mathcal{N}(0, I), \qquad \hat x_0 = \mathrm{ODEFlow}^{(k)}(\varepsilon); \qquad L_{\mathrm{reflow}} = \mathbb{E}_{t, \varepsilon} \left\| v_\theta^{(k+1)}\left( (1-t)\, \hat x_0 + t\, \varepsilon,\ t \right) - \left( \varepsilon - \hat x_0 \right) \right\|^2 \qquad (29)
$$

**符号**：$\mathrm{ODEFlow}^{(k)}$：第 $k$ 轮模型的噪声→数据确定性映射（1 步或多步积出）；$L_{\mathrm{reflow}}$：第 $k+1$ 轮的训练目标——用第 $k$ 轮自产的配对 $(\hat x_0, \varepsilon)$ 按 7.5 的线性路径（$x_t = (1-t) x_0 + t\varepsilon$、目标速度 $\varepsilon - x_0$）重训。

在蒸馏语境里 reflow 的位置是**前置工序**：轨迹越直，§5.2 的自一致性约束越容易满足（$f$ 逼近的映射越接近线性插值端点），1-2 步学生的逼近误差越小。它与 CM/DMD 不互斥——先 reflow 教师、再蒸馏学生是常见组合。

### 5.5 DMD：分布匹配梯度（推导）

DMD（[Yin et al., 2023](../references.md)，arXiv:2311.18828，[A]）再进一步：不逐点匹配轨迹，直接匹配**分布**。学生是一步生成器 $G_\psi(z)$（$z \sim \mathcal{N}(0, I)$，$x = G_\psi(z)$），目标是最小化生成分布 $p_{\mathrm{fake}} = p_\psi$ 与教师分布 $p_{\mathrm{real}}$ 的 KL：

$$
\min_\psi\ \mathrm{KL}\left( p_\psi\, \| \, p_{\mathrm{real}} \right) = \min_\psi\ \left( -H(p_\psi) - \mathbb{E}_{x \sim p_\psi} \left[ \log p_{\mathrm{real}}(x) \right] \right) \qquad (30)
$$

**符号**：$H$：微分熵；$p_{\mathrm{real}}$：教师模型的生成分布（其 score 由冻结教师网络提供，见下）。

对 $\psi$ 求梯度，两项分别处理。**交叉项**用重参数化 $x = G_\psi(z)$ 直接链式：

$$
\nabla_\psi\, \mathbb{E}_{z} \left[ \log p_{\mathrm{real}}\left( G_\psi(z) \right) \right] = \mathbb{E}_{z} \left[ s_{\mathrm{real}}(x)\, \frac{\partial x}{\partial \psi} \right], \qquad x = G_\psi(z) \qquad (31)
$$

**符号**：$s_{\mathrm{real}} = \nabla_x \log p_{\mathrm{real}}$：真实（教师）score；$\partial x / \partial \psi$：生成器输出的参数 Jacobian——一次反向传播的事。

**熵项**：$\nabla_\psi\, \mathbb{E}_{z}[\log p_\psi(G_\psi(z))]$。注意这个期望对固定底分布 $z$ 取、即等于 $\mathbb{E}_{p_\psi}[\log p_\psi]$，其梯度拆两件：测度项 $\mathbb{E}_{p_\psi}[\partial_\psi \log p_\psi] = \partial_\psi \int p_\psi\, dx = 0$（密度归一的恒等推论），剩下样本点处的链式项 $\mathbb{E}_{z}[ s_{\mathrm{fake}}(x)\, \partial x / \partial \psi ]$，其中 $s_{\mathrm{fake}} = \nabla_x \log p_\psi$ 是**当前生成分布的 score**。合并得**分布匹配梯度**：

$$
\nabla_\psi\, \mathrm{KL}\left( p_\psi \| p_{\mathrm{real}} \right) = \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ \left( s_{\mathrm{fake}}(x) - s_{\mathrm{real}}(x) \right) \frac{\partial x}{\partial \psi} \right], \qquad x = G_\psi(z) \qquad (32)
$$

**符号**：逐样本的力 = （fake score − real score）经生成器 Jacobian 回传。

机制读法：在 $x$ 处 $s_{\mathrm{fake}} > s_{\mathrm{real}}$（fake 密度坡度更陡）意味着该处 fake 质量过剩，梯度把生成器输出往反方向推——质量从过密区迁往欠密区，直到两 score 处处相等（即 $p_\psi = p_{\mathrm{real}}$，KL 的唯一不动点）。

两个 score 的来源不对称，这是 DMD 的结构要点：**$s_{\mathrm{real}}$ 免费**——冻结的预训练教师经 score↔ε 换算（7.3 式 (9)）直接给出（把 fake 样本加噪到某 $t$、读教师 $\varepsilon_\theta$ 即得该噪声水平下的 real score）；**$s_{\mathrm{fake}}$ 要在线养**——$p_\psi$ 随训练时刻在变，没有现成 score，于是并行训一个小 score 网络，用标准 DSM（7.3 §3）在生成器当前输出上回归——生成器每更新一步，fake 分布变一点，fake score 网络跟着追。主循环是两者的交替优化。论文另配一项回归损失（在教师预生成的 $(z, x_0)$ ODE 配对上做 L2）作初始化与稳定器。成果口径：1 步生成、质量逼近多步教师（具体 FID 见原表）。

**DMD2 注记**（[Yin et al., 2024](../references.md)，arXiv:2405.14867，[A]；官方实现 [tianweiy/DMD2](../references.md)，[A]）：改进两点——(i) 去掉对教师生成数据配对的回归依赖（免教师数据）；(ii) **在分布匹配梯度之外加 GAN 分支**（判别器分真伪、生成器梯度与之叠加）补偿大步数压缩下的细节损失。防张冠李戴：**GAN 分量属 DMD2，不属 DMD**（与约定页口径一致）。

### 5.6 few-step 现状表

各路线的定位对照（数值口径随数据集/分辨率/评估协议而异——按 W4 研究纪律，FID 数字不跨表转录、一律回各论文原表；2026-11-12 口径）：

| 方法 | 机制 | 免重训 | 需教师 | 最低实用 NFE | 关键出处 |
| --- | --- | --- | --- | --- | --- |
| DDIM 跳步（η=0） | 非马尔可夫族 + 跳格 | 是 | — | ~50（再低质量明显下降） | [Song et al., 2021](../references.md)（ddim） |
| DPM-Solver（K=2/3） | 指数积分 + 高阶展开 | 是 | — | ~10（论文标题口径） | [Lu et al., 2022](../references.md)（dpm-solver） |
| progressive distillation | T→T/2 逐轮减半 | 否（k 轮训练） | 是 | 4（论文报告至 4 步质量近教师） | [Salimans & Ho, 2022](../references.md)（progressive-distillation） |
| consistency（CD/CT） | ODE 轨迹自一致 | 否（1 轮） | CD 需 / CT 免 | 1-2 | [Song et al., 2023](../references.md)（consistency-models） |
| LCM | latent 一致性 + CFG 蒸馏 | 否 | 是（带 CFG 教师） | 2-4（标题口径 few-step 高分辨率） | [Luo et al., 2023](../references.md)（lcm） |
| DMD | 分布匹配梯度 + 在线 fake score | 否 | 是（冻结 score + 配对初始化） | 1 | [Yin et al., 2023](../references.md)（dmd） |
| DMD2 | DMD 免教师数据 + GAN 分支 | 否 | 是（冻结 score） | 1 | [Yin et al., 2024](../references.md)（dmd2） |
| reflow（rectified flow） | 轨迹拉直（前置工序） | 否（可多轮） | 自蒸馏 | few-step（与上面组合使用） | [Liu et al., 2022](../references.md)（rectified-flow） |

读表要点：(i) 「免重训」一列是部署决策的第一分叉——求解器路线当天可换，蒸馏路线要付训练账；(ii) 1 步档目前由分布匹配路线（DMD 族）占据，自一致性路线的主力在 2-4 步，求解器在 ~10 步，跳步在 ~50 步——步数每下一档，训练账上一个量级；(iii) guidance 与该表正交：CFG 使所有行的 NFE ×2，guidance 蒸馏（§4.5）再把这个因子收回来，LCM 是两者合流的现存实例。

## 总结

本章公式清单（编号为章内序号）：

| 对象 | 公式 | 要点 |
| --- | --- | --- |
| DDIM 前向族 | $q_\sigma(x_{t-1} \mid x_t, x_0)$，式 (1) | 边缘不变的全部自由度 = $\{\sigma_t\}$；边缘匹配引理式 (2)(3) |
| DDIM 端点 | $\sigma_t = \sqrt{\tilde\beta_t}$ 退回 DDPM；$\sigma_t = 0$ 得式 (5) | $\eta$ 滑杆连接两端 |
| DDIM 更新 | $x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1-\bar\alpha_{t-1}} \varepsilon_\theta$，式 (5)；跳步版式 (6) | 确定性、可逆、跳步免重训 |
| DDIM = PF-ODE 一阶 | $\lambda = \frac{1}{2}\log\mathrm{SNR}$ 坐标式 (8)(9)；指数积分解式 (10)；冻结场收回 DDIM 式 (11) | 精确恒等，非渐近 |
| DPM-Solver | Taylor 展开式 (12) + 闭式矩更新式 (13) | 多步法复用历史评估，零附加 NFE |
| classifier guidance | $\tilde\varepsilon = \varepsilon_\theta - w\sqrt{1-\bar\alpha_t} \nabla \log p_\phi(y \mid x_t)$，式 (15) | Bayes 展开式 (14)；需耐噪分类器 |
| CFG | $\tilde\varepsilon = \varepsilon_\theta(\varnothing) + w(\varepsilon_\theta(c) - \varepsilon_\theta(\varnothing))$，式 (18) | 隐式分类器式 (16)(17)；$w=1$ 退化纯条件 |
| $w$ 的语义 | $q_w(x) \propto p(x) p(y \mid x)^w$，式 (19) | log-likelihood tilt；范数膨胀式 (20) |
| 缓解 | dynamic thresholding 式 (21)；rescaled CFG 式 (22) | 修尺度不修方向 |
| guidance 蒸馏 | 目标式 (23) | 每步 2 NFE → 1；工业默认 |
| progressive distillation | 反解目标式 (24) + 减半迭代式 (25) | $k$ 轮训练账 |
| consistency | 自一致性式 (26)；参数化式 (27)；CD 损失式 (28) | CT 免教师；任意 $t' \to 0$ 跳 |
| DMD | KL 梯度式 (32)（两项拆分式 (30)(31)） | real score 冻结教师、fake score 在线训练；GAN 分量属 DMD2 |
| reflow | 自产配对重训式 (29) | 拉直轨迹，蒸馏前置工序 |

## 练习

1.（推导）$\eta$ 滑杆的合法性。证明：取 $\sigma_t = \eta \sqrt{\tilde\beta_t}$（$\eta \in [0, 1]$ 任意），式 (1) 的前向族仍对所有 $t$ 复现边缘 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$；并写出该取值下生成更新中 $\varepsilon_\theta$ 项的系数，验证 $\eta = 1$ 退回 DDPM ancestral（$\tilde\beta_t$ 档）、$\eta = 0$ 退回式 (5)。提示：只须重跑 §2.2 的方差求和。
2.（推导）CFG 的 tilt 恒等。(a) 验证式 (18) 对应的有效 score 恰为 $\nabla_x \log q_w(x)$（式 (19) 的分布），归一化常数不进 score；(b) 证明 $w = 1$ 时 $q_w(x) = p(x \mid y)$；(c) 设 $p(y \mid x) \propto e^{-E(x, y)}$（能量形式），说明 $w$ 相当于该能量的逆温度，并由此解释「大 $w$ = 多样性换对齐」的统计力学读法。
3.（计算）二阶矩的闭式。(a) 计算 $\int_{\lambda_t}^{\lambda_s} e^{-\lambda} (\lambda - \lambda_t)\, d\lambda$ 的闭式（分部积分）；(b) 代入式 (13) 写出 $K = 2$ 的完整更新式，全部系数用 $\bar\alpha_t, \bar\alpha_s$ 表达；(c) 设格点在 $\lambda$ 坐标均匀、$\Delta\lambda = \lambda_s - \lambda_t$，验证 $K = 2$ 的局部截断误差为 $O(\Delta\lambda^3)$。
4.（代码）教学用，非优化实现：复用 7.2 练习 4 的二维 swiss-roll 迷你 DDPM。(a) 实现 DDIM 更新式 (5) 与 $\eta$ 滑杆（式 (1) 的生成版）；(b) 在 NFE ∈ {1000, 100, 20} 三档下对比 ancestral、DDIM $\eta = 1$、DDIM $\eta = 0$、DDIM 跳步（式 (6)，50 格点）的样本-数据协方差匹配误差，验证「低 NFE 下确定性占优」的曲线形状（§3.2）；(c) 加二分类条件（左月/右月），实现 CFG 式 (18)，扫 $w \in \{1, 3, 7, 15\}$，画出类内样本方差随 $w$ 的曲线——对应式 (20) 的范数膨胀与多样性塌缩。

## 参考

- [Song et al., 2021](../references.md)：Denoising Diffusion Implicit Models，arXiv:2010.02502，[A]（key: ddim）——式 (1)(2)(3)(4)(5)(6)：非马尔可夫前向族、σ 端点、跳步；$\eta$ 滑杆
- [ermongroup/ddim](../references.md)：DDIM 官方实现，[A]（key: ddim-repo）
- [Lu et al., 2022](../references.md)：DPM-Solver，arXiv:2206.00927，[A]（key: dpm-solver）——式 (12)(13)：半线性结构 + 指数积分 + Taylor 展开；~10 NFE 口径（标题）
- [Dhariwal & Nichol, 2021](../references.md)：Diffusion Models Beat GANs on Image Synthesis，arXiv:2105.05233，[A]（key: classifier-guidance）——式 (14)(15)：classifier guidance 的 Bayes 展开与 ε 形式
- [Ho & Salimans, 2022](../references.md)：Classifier-Free Diffusion Guidance，arXiv:2207.12598，[A]（key: cfg）——式 (16)(17)(18)(19)：隐式分类器推导、tilt 语义
- [Hang et al., 2023](../references.md)：Common Diffusion Noise Schedules and Sample Steps are Flawed，arXiv:2305.08891，[A]（key: common-diffusion）——rescaled CFG 式 (22)：std 比值与 $\phi$ 混合
- [Saharia et al., 2022](../references.md)：Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding，arXiv:2205.11487，[B]（key: imagen）——dynamic thresholding 式 (21)
- [Salimans & Ho, 2022](../references.md)：Progressive Distillation for Fast Sampling of Diffusion Models，arXiv:2202.00512，[A]（key: progressive-distillation）——式 (24)(25)：减半迭代与目标反解；v-prediction 原始出处（经 7.2 §5.1）
- [Song et al., 2023](../references.md)：Consistency Models，arXiv:2303.01469，[A]（key: consistency-models）——式 (26)(27)(28)：自一致性、边界参数化、CD/CT
- [openai/consistency_models](../references.md)：consistency models 官方实现（`cm/losses.py`、`cm/karras_diffusion.py`），[A]（key: consistency-models-repo）
- [Luo et al., 2023](../references.md)：Latent Consistency Models，arXiv:2310.04378，[A]（key: lcm）——latent 空间一致性 + CFG 蒸馏合流（§4.5、§5.3）
- [Yin et al., 2023](../references.md)：One-step Diffusion with Distribution Matching Distillation，arXiv:2311.18828，[A]（key: dmd）——式 (30)(31)(32)：分布匹配梯度
- [Yin et al., 2024](../references.md)：Improved Distribution Matching Distillation for Fast Image Synthesis，arXiv:2405.14867，[A]（key: dmd2）——免教师数据 + GAN 分支（GAN 分量属本文不属 DMD）
- [tianweiy/DMD2](../references.md)：DMD/DMD2 官方实现，[A]（key: dmd2-repo）
- [Liu et al., 2022](../references.md)：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow，arXiv:2209.03003，[A]（key: rectified-flow）——式 (29)：reflow（路径数学见 7.5）
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——ancestral 采样与后验（经 7.2 交叉引用）
- [Song et al., 2021](../references.md)：Score-Based Generative Modeling through SDEs，arXiv:2011.13456，[A]（key: score-sde）——PF-ODE 母体（经 7.3 交叉引用，式 (7) 的出处）
