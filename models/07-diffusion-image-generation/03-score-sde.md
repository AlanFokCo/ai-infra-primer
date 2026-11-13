# 7.3 Score-Based 视角与 SDE/ODE（The Score-Based View: SDE and Probability-Flow ODE）

> 本章把 7.2 的离散加噪链连续化成随机微分方程（SDE）：先证明 denoising score matching（DSM）与真 score 回归同梯度（Vincent 定理），给出「不知道数据分布也能学 score」的机制；再由 Anderson 逆时定理与 Fokker-Planck 方程推出共享同一边缘密度族的三条采样动力学——逆时 SDE、probability-flow ODE（PF-ODE）与它们的离散化——把 DDPM / DDIM / ODE solver 收进同一张表 | 前置：7.2（前向闭式解、 $\varepsilon$-参数化、 $L_{simple}$、SNR 权重；本章式号独立编号）、6.3（ELBO）、1.1（KL） | 锚点：score-SDE [A]、EDM [A]、Vincent 2011 [A]；yang-song/score_sde_pytorch [A]

## 1. score：密度景观的坡度，也是采样的接口

分数函数（score function）定义为对数密度对数据本身的梯度：

$$
s(x) = \nabla_x \log p(x) \qquad (1)
$$

**符号**： $x \in \mathbb{R}^d$：数据向量（图像拉直，CIFAR-10 规模 $d = 3072$）； $s(x) \in \mathbb{R}^d$：与 $x$ 同 shape 的向量场。

两个直接性质决定它的工程地位。**(i) 归一化常数被消掉**。任何能量式模型 $p(x) = \tilde p(x) / Z$ 的 score 是 $s(x) = \nabla_x \log \tilde p(x)$——配分函数 $Z$ 是常数，梯度下消失。密度只知道「形状」不知道「归一化」的模型，score 照样可算。**(ii) 几何上它是密度景观的坡度**： $s(x)$ 指向 $\log p$ 上升最快的方向，模长是坡的陡度。数据流形附近 score 大致指向流形（把离流形的点拉回去），低密度区之间 score 把样本推向最近的 mode。

score 还是采样的接口：给定任意可评估的向量场 $s(x)$，Langevin 动力学 $x \leftarrow x + \eta\thinspace s(x) + \sqrt{2\eta}\thinspace z$（ $z \sim \mathcal{N}(0, I)$， $\eta$ 为步长）在 $s = \nabla \log p$ 时以 $p$ 为平稳分布。也就是说，**只要拿到 score，采样问题就解了**——不需要会算 $p(x)$ 本身。本章的全部工作可以概括成一句话：7.2 训出来的 $\varepsilon_\theta$ 换一层皮就是 score（§3），而 score 喂给的动力学不止 Langevin 一种——SDE 框架给出完整菜单（§4-§7）。

先登记一个后面反复用的记号口径：本章起时间连续化， $t \in [0, 1]$，**$t = 0$ 数据端、 $t = 1$ 噪声端**；前向（加噪） $t: 0 \to 1$，采样 $t: 1 \to 0$。7.2 离散下标 $t \in \lbrace1, ..., T\rbrace$ 与连续坐标的对应是 $t_{\mathrm{cont}} = t / T$； $\bar\alpha_t$（离散）的连续对应写作 $\bar\alpha(t)$。前向加噪的边缘分布记

$$
q_t(x_t) = \int q(x_t \mid x_0)\thinspace p_{\mathrm{data}}(x_0)\thinspace dx_0
$$

**符号**： $q(x_t \mid x_0)$：人为给定的高斯加噪核（7.2 §2.2 的闭式解）； $q_t$：数据分布被加噪涂抹后的第 $t$ 时刻边缘密度——未知（依赖 $p_{\mathrm{data}}$），但样本可采（采 $x_0$ 再加噪）。

## 2. score matching：目标很直白，梯度拿不到

学 score 的自然目标是最小二乘回归——分数匹配（score matching）：

$$
J_{\mathrm{SM}}(\theta) = \frac{1}{2}\thinspace \mathbb{E}_{x \sim p_{\mathrm{data}}} \left\Vert s_\theta(x) - \nabla_x \log p_{\mathrm{data}}(x) \right\Vert^2 \qquad (2)
$$

**符号**： $s_\theta: \mathbb{R}^d \to \mathbb{R}^d$：待训的 score 网络；期望对数据分布取。

问题立刻暴露：回归靶 $\nabla \log p_{\mathrm{data}}$ 恰恰是不知道的东西——知道它就不用学了。式 (2) 的梯度算不出来，目标形同虚设。出路有两条。

**出路一：分部积分消掉未知量**（隐式 score matching）。展开式 (2) 的平方，交叉项里对 $\log p_{\mathrm{data}}$ 的依赖可以被积掉：

$$
\mathbb{E}_{p}\left\langle s_\theta,\ \nabla \log p \right\rangle = \int \left\langle s_\theta,\ \nabla p \right\rangle dx = - \int p\thinspace \nabla \cdot s_\theta\thinspace dx = -\mathbb{E}_p\left[ \nabla \cdot s_\theta \right]
$$

第一步把 $\nabla \log p = \nabla p / p$ 代回；第二步对每一维分部积分，边界项因密度衰减消失。于是

$$
J_{\mathrm{SM}}(\theta) = \mathbb{E}_{p_{\mathrm{data}}}\left[ \frac{1}{2} \left\Vert s_\theta(x) \right\Vert^2 + \nabla \cdot s_\theta(x) \right] + C \qquad (3)
$$

**符号**： $\nabla \cdot s_\theta = \mathrm{tr}\left( \partial s_\theta / \partial x \right)$：score 场的散度，即 $d \times d$ Jacobian 的迹； $C = \frac{1}{2} \mathbb{E} \Vert\nabla \log p_{\mathrm{data}}\Vert^2$：与 $\theta$ 无关的常数。

式 (3) 右边只含样本与网络，确实可算——但代价写在散度里：trace 要对 $d$ 个输入维各做一次反向传播，**每样本每步训练开销是 $d$ 次 backward**（ $d = 3072$ 时即三千余倍于一次前向的代价量级）。这条路在数学上成立、在高维图像上不可行。

**出路二：换一个已知的回归靶**。不去回归边缘 score $\nabla \log q_t$，改回归**条件 score** $\nabla_{x_t} \log q(x_t \mid x_0)$——加噪核是人为给定的高斯，它对 $x_t$ 的梯度有闭式。这一换之所以合法，是下一节的定理：两个目标的梯度完全相同。

在进入推导之前，先回答一个更根本的问题：为什么不干脆对**干净数据**做 score matching（式 (3) 的散度版虽然贵，终究可算），而要引入整条噪声阶梯 $t \in [0, 1]$？三个机制性的原因：

1. **支集问题**。自然图像被认为集中在 $d$ 维空间的低维流形附近（7.1 的 mode 结构讨论）， $p_{\mathrm{data}}$ 相对环境空间的 Lebesgue 测度没有密度—— $\nabla \log p_{\mathrm{data}}$ 在流形外根本没有定义，而采样轨迹恰恰要经过流形外。高斯卷积把任何分布涂抹成全支集： $t > 0$ 时 $q_t$ 处处为正，score 处处有定义。
2. **低密度区估计失效**。即使密度处处存在，两个 mode 之间的低密度谷底鲜有样本，score 估计在那里方差极大；而 Langevin 采样要穿越的正是这些谷底——训练信号最弱的地方恰是动力学最关键的路段。大 $t$ 的噪声把 mode 的尾部抹开、谷底填平，小 $t$ 恢复细节：噪声阶梯就是一条退火路径。
3. **单一网络摊销全部尺度**。让每个 $t$ 各训一个网络的代价不可接受； $s_\theta(x, t)$ 以 $t$ 为条件（注入方式同 7.2 §5.3 的 U-Net 契约），一个网络覆盖整条阶梯——这正是 7.2 里「 $t$ 均匀采样 + $L_{simple}$」训练循环的 score 读法。

所以本章的训练对象从来不是「一个 score」，而是**一族噪声条件下的 score** $\lbrace s_\theta(x, t)\rbrace_{t \in [0,1]}$；DSM 定理（下一节）对其中每个 $t$ 分别成立。

## 3. Denoising score matching：Vincent 等价性定理（完整推导）

对固定时刻 $t$，真 score 回归目标是

$$
J_t(\theta) = \frac{1}{2}\thinspace \mathbb{E}_{q_t(x_t)} \left\Vert s_\theta(x_t, t) - \nabla_{x_t} \log q_t(x_t) \right\Vert^2 \qquad (4)
$$

**符号**： $q_t$：未知边缘； $s_\theta(x_t, t)$：时刻 $t$ 的 score 网络。

去噪分数匹配（denoising score matching, DSM）把回归靶换成条件 score：

$$
J_{\mathrm{DSM},t}(\theta) = \frac{1}{2}\thinspace \mathbb{E}_{p_{\mathrm{data}}(x_0)\thinspace q(x_t \mid x_0)} \left\Vert s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0) \right\Vert^2 \qquad (5)
$$

**符号**：期望对联合分布取：先采数据 $x_0$，再经高斯核采 $x_t$——每一步都可执行。

**Vincent 定理**（[Vincent, 2011](../references.md)，DOI 10.1162/NECO_a_00142，[A]；原文对一般加噪核证明，此处按本书记号重述）：两个目标只差一个与 $\theta$ 无关的常数，

$$
J_t(\theta) = J_{\mathrm{DSM},t}(\theta) + C_t \qquad (6)
$$

因而 $\nabla_\theta J_t = \nabla_\theta J_{\mathrm{DSM},t}$：**对条件 score 回归，等价于对真（边缘）score 回归**。证明的关键是一条恒等式加一次期望换序，逐步展开。

**第一步：条件 score 恒等式**——边缘 score 是条件 score 的后验均值：

$$
\nabla_{x_t} \log q_t(x_t) = \mathbb{E}_{q(x_0 \mid x_t)}\left[ \nabla_{x_t} \log q(x_t \mid x_0) \right] \qquad (7)
$$

推导：对 $q_t(x_t) = \int q(x_t \mid x_0) p_{\mathrm{data}}(x_0) dx_0$ 两端取梯度，积分号与梯度交换（高斯核光滑衰减，交换合法）：

$$
\nabla_{x_t} q_t(x_t) = \int \nabla_{x_t} q(x_t \mid x_0)\thinspace p_{\mathrm{data}}(x_0)\thinspace dx_0 = \int q(x_t \mid x_0)\thinspace \nabla_{x_t} \log q(x_t \mid x_0)\thinspace p_{\mathrm{data}}(x_0)\thinspace dx_0
$$

第二步用了 $\nabla q = q\thinspace \nabla \log q$。两端除以 $q_t(x_t)$，并认出 Bayes 后验 $q(x_0 \mid x_t) = q(x_t \mid x_0) p_{\mathrm{data}}(x_0) / q_t(x_t)$，即得式 (7)。直觉： $x_t$ 处密度的真实坡度，等于「 $x_t$ 可能来自的各个 $x_0$」各自的指向箭头按后验加权平均。

**第二步：两目标的交叉项相等**。把式 (4)(5) 的平方都展开： $\frac{1}{2}\Vert s_\theta\Vert^2$ 项两式相同（式 (5) 的该项对 $x_0$ 求边缘即式 (4) 的该项）。差别只在交叉项与常数项。式 (4) 的交叉项代入式 (7)：

$$
\begin{aligned}
\mathbb{E}_{q_t(x_t)} \left\langle s_\theta(x_t, t),\ \nabla_{x_t} \log q_t(x_t) \right\rangle
&= \int q_t(x_t) \left\langle s_\theta(x_t, t),\ \int q(x_0 \mid x_t)\thinspace \nabla_{x_t} \log q(x_t \mid x_0)\thinspace dx_0 \right\rangle dx_t \\
&= \int \negthinspace\negthinspace \int q_t(x_t)\thinspace q(x_0 \mid x_t) \left\langle s_\theta(x_t, t),\ \nabla_{x_t} \log q(x_t \mid x_0) \right\rangle dx_0\thinspace dx_t \\
&= \mathbb{E}_{p_{\mathrm{data}}(x_0)\thinspace q(x_t \mid x_0)} \left\langle s_\theta(x_t, t),\ \nabla_{x_t} \log q(x_t \mid x_0) \right\rangle
\end{aligned}
$$

第二步把 $s_\theta$（不依赖 $x_0$）提出内层积分——这一步是「 $s_\theta$ 只需要是 $x_t$ 的函数」的根本来源；第三步用 $q_t(x_t) q(x_0 \mid x_t) = p_{\mathrm{data}}(x_0) q(x_t \mid x_0)$ 把测度换回联合分布。结果恰是式 (5) 的交叉项。

**第三步：收尾**。两目标相减，平方项与交叉项全部抵消，只剩

$$
C_t = \frac{1}{2}\thinspace \mathbb{E}_{q_t} \left\Vert \nabla \log q_t \right\Vert^2 - \frac{1}{2}\thinspace \mathbb{E}_{p_{\mathrm{data}}\thinspace q} \left\Vert \nabla \log q(x_t \mid x_0) \right\Vert^2
$$

两项都不含 $\theta$（第一项是未知但固定的 Fisher 信息量，第二项对高斯核闭式可算）。式 (6) 成立。 $\square$

定理的两个直接推论值得单独点出。**(i) 最优解一致**： $J_{\mathrm{DSM},t}$ 的最小值点同样是 $s_\theta = \nabla \log q_t$（回归目标为条件期望——最小二乘回归的最优解是对输入的条件均值，此处即式 (7) 右端）；DSM 不是近似目标，是同一目标的随机实现，每个样本的回归靶是「给定 $x_0$ 时确定的那个箭头」， $x_0$ 的随机性在期望里平均成真 score。**(ii) 样本可采**：式 (5) 的期望只需采 $(x_0, x_t)$ 对——训练循环与 7.2 §4.4 完全同构，每个样本一次前向一次回归，没有式 (3) 的 $d$ 次 backward。

把高斯核代入，回归靶显形。由 7.2 §2.2 的闭式解， $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\thinspace x_0, (1 - \bar\alpha_t) I)$，对 $x_t$ 取梯度：

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\thinspace x_0}{1 - \bar\alpha_t} = -\frac{\varepsilon}{\sqrt{1 - \bar\alpha_t}} \qquad (8)
$$

**符号**：第二步代入采样式 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon$（7.2 §2.2 的采样式）； $\varepsilon$：该样本实际注入的噪声。

式 (8) 是 DSM 的关键福利：**回归靶是注入噪声的线性函数，训练时免费可得**。结合最优解条件 $s_\theta(x_t, t) = \mathbb{E}\left[ -\varepsilon / \sqrt{1 - \bar\alpha_t} \mid x_t \right]$ 与 7.2 的 $\varepsilon_\theta$（回归 $\mathbb{E}[\varepsilon \mid x_t]$ 的同一个条件期望），读出两种参数化的精确换算：

$$
s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}} \qquad \Longleftrightarrow \qquad \varepsilon_\theta(x_t, t) = -\sqrt{1 - \bar\alpha_t}\hspace{5mu} s_\theta(x_t, t) \qquad (9)
$$

**符号**：换算逐点成立（不只是最优解处），可作为两个网络输出层的代数变换。

score 视角还顺手给出 7.2「 $x_0$-prediction」的概率语义。把式 (7) 的恒等式作用在高斯核上： $`(1 - \bar\alpha_t)\thinspace \nabla \log q_t(x_t) = \mathbb{E}_{q(x_0 \mid x_t)}[-x_t + \sqrt{\bar\alpha_t}\thinspace x_0]`$，整理得 Tweedie 公式：

$$
\mathbb{E}\left[ x_0 \mid x_t \right] = \frac{x_t + (1 - \bar\alpha_t)\thinspace s_\theta(x_t, t)}{\sqrt{\bar\alpha_t}} \qquad (10)
$$

**符号**：左端：加噪样本的后验均值——均方误差最优的「去噪图」。

即 7.2 的 $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon_\theta) / \sqrt{\bar\alpha_t}$ 不是工程 trick，而是后验均值的解析式：score 与 $x_t$ 的线性组合恰好抽出数据的条件期望。7.4 的全部确定性采样器都把式 (10) 当作每步的「lookahead 锚点」。

最后记一笔权重账，衔接 7.2 §4.3。把式 (9) 代入式 (5)，回归靶换成 $\varepsilon$ 后：

$$
J_{\mathrm{DSM},t}(\theta) = \frac{1}{2(1 - \bar\alpha_t)}\thinspace \mathbb{E}_{x_0, \varepsilon} \left\Vert \varepsilon - \varepsilon_\theta(x_t, t) \right\Vert^2 + C_t' \qquad (11)
$$

**符号**： $C_t'$：与 $\theta$ 无关项；权重 $1 / (1 - \bar\alpha_t)$ 在低噪声端（ $\bar\alpha_t \to 1$）发散。

三种目标于是排成一行： $L_{simple}$ 权重恒 1（7.2 §4.3）；裸 DSM（对 $t$ 均匀、不另加权）权重 $1/(1-\bar\alpha_t)$，死命压小噪声端；严格 ELBO 权重 $\approx \frac{1}{2} \Delta_t \log \mathrm{SNR}$（7.2 §4.3 的推导），压大噪声端。**损失权重从来不是定理强制的，是设计自由度**——score-SDE 论文取连续权重 $\lambda(t) = g(t)^2$（ $g$ 为扩散系数，§4）使 DSM 目标与似然的变分界对齐（[Song et al., 2021](../references.md)，arXiv:2011.13456，[A]）；EDM 把这条自由度系统化（§8）。

## 4. 连续化：从加噪链到 SDE

### 4.1 VP-SDE：DDPM 链的扩散极限

7.2 的前向链 $x_i = \sqrt{1 - \beta_i}\thinspace x_{i-1} + \sqrt{\beta_i}\thinspace z_i$ 是离散时间动力；把步长压向零，它收敛到一个 Itô 随机微分方程（stochastic differential equation, SDE）。取网格 $t_i = i/T$、 $\Delta t = 1/T$，定义连续噪声率 $\beta(t)$ 使 $\beta_i = \beta(t_i)\thinspace \Delta t$（即 $\beta(t) = T \beta_i$，离散值是连续率的「每步剂量」）。链的一步改写为

$$
x(t + \Delta t) - x(t) = \left( \sqrt{1 - \beta(t)\thinspace \Delta t} - 1 \right) x(t) + \sqrt{\beta(t)\thinspace \Delta t}\hspace{5mu} z \qquad (12)
$$

**符号**： $z \sim \mathcal{N}(0, I)$； $x(t)$： $x_i$ 的连续插值记号。

两项分别取极限。**漂移项**：Taylor 展开 $\sqrt{1 - u} = 1 - u/2 + O(u^2)$，得 $-\frac{1}{2} \beta(t)\thinspace x(t)\thinspace \Delta t + O(\Delta t^2)$——一阶项正比于 $\Delta t$。**噪声项**： $\sqrt{\beta(t) \Delta t}\thinspace z$ 的协方差是 $\beta(t) \Delta t \cdot I$，恰为维纳过程（Wiener process）增量的统计： $\Delta W \sim \mathcal{N}(0, \Delta t\thinspace I)$。 $\Delta t \to 0$ 时得 **VP-SDE**（variance preserving SDE；[Song et al., 2021](../references.md)，[A]，VE/VP 分类的出处）：

$$
dx = -\frac{1}{2}\thinspace \beta(t)\thinspace x\thinspace dt + \sqrt{\beta(t)}\hspace{5mu} dW \qquad (13)
$$

**符号**： $dW$：维纳过程微元；漂移 $f(x, t) = -\frac{1}{2}\beta(t) x$：把状态向原点收缩的线性弹簧；扩散系数 $g(t) = \sqrt{\beta(t)}$：各向同性注噪强度。方向 $t: 0 \to 1$（加噪）。

连续化没有偷换模型——验证边缘分布一致。线性 SDE 可解： $x(t) = x_0\thinspace e^{-\frac{1}{2} \int_0^t \beta(s) ds} + \int_0^t \sqrt{\beta(s)}\thinspace e^{-\frac{1}{2} \int_s^t \beta} dW_s$，第二项是零均值高斯，方差由 Itô 等距算出：

$$
\mathrm{Var} = \int_0^t \beta(s)\thinspace e^{-\int_s^t \beta(u) du} ds = e^{-\int_0^t \beta} \left( e^{\int_0^t \beta} - 1 \right) = 1 - e^{-\int_0^t \beta(s) ds}
$$

于是

$$
q(x_t \mid x_0) = \mathcal{N}\left( e^{-\frac{1}{2} \int_0^t \beta(s) ds} x_0,\ \left( 1 - e^{-\int_0^t \beta(s) ds} \right) I \right), \qquad \bar\alpha(t) = e^{-\int_0^t \beta(s) ds} \qquad (14)
$$

**符号**： $\bar\alpha(t)$：累计信号保留率的连续版本；离散 $\bar\alpha_t = \prod (1 - \beta_s)$ 在 $\beta_i \to 0$ 下由 $1 - u \approx e^{-u}$ 收敛到它。

与 7.2 §2.2 的闭式解逐项对照：同一个高斯、同一个 $\bar\alpha$ 语义。VP 之名同时落地： $t$ 全过程中 $\mathrm{Var}(x_t) \approx 1$（单位方差数据下），与 7.2 §2.1 的离散论证互为镜像。schedule 的连续读法也随之而来：linear schedule（7.2 §2.3）对应 $\beta(t)$ 从 $0.1$ 到 $20$ 线性增长（ $\beta(t) = T\beta_i$ 的换算），cosine schedule 对应一条 S 形 $\bar\alpha(t)$——7.2 §2.3 的 SNR 表原样适用。

### 4.2 VE-SDE：只注噪、不收缩的另一族

VP 不是唯一的连续化选择。另一条加噪链 $x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\thinspace z_i$（均值不衰减、方差逐步累加）的极限是 **VE-SDE**（variance exploding SDE）：

$$
dx = \sqrt{\frac{d \sigma^2(t)}{dt}}\hspace{5mu} dW \qquad (15)
$$

**符号**： $\sigma(t)$：单调递增的噪声尺度序列的连续化；漂移 $f = 0$，扩散系数 $g(t) = \sqrt{d\sigma^2/dt}$。

直接积分得边缘 $q(x_t \mid x_0) = \mathcal{N}(x_0, \sigma^2(t) I)$：信号原封不动，噪声方差随 $t$ 膨胀到掩盖一切——VE 之名由此。两族的分工在端点行为：VP 把信号指数压向零、端点趋于标准高斯（先验匹配干净， $p(x_1) = \mathcal{N}(0, I)$ 无失配）；VE 保留全部信号均值、用大噪声淹没，端点方差远大于 1。score-SDE 框架把二者收进同一套方程——区别只在 $(f, g)$ 的选取，本章 §5-§7 的所有结论对任意 $(f, g)$ 成立。

### 4.3 Fokker-Planck：密度如何随时间流（结论级）

SDE 描述单条轨迹，训练与似然关心的是密度。福克-普朗克方程（Fokker-Planck equation）给出两者的桥：对 $dx = f(x, t) dt + g(t) dW$（ $g$ 只依赖 $t$），边缘密度 $p_t$ 满足偏微分方程

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left( f\thinspace p_t \right) + \frac{1}{2}\thinspace g(t)^2\thinspace \Delta p_t \qquad (16)
$$

**符号**： $\nabla \cdot (f p_t) = \sum_i \partial_i (f_i p_t)$：漂移对密度的散度——连续性方程项，密度被速度场 $f$ 搬运； $\Delta p_t = \nabla \cdot \nabla p_t$：拉普拉斯算子——扩散项，把密度从高处往低处抹平，速率 $\frac{1}{2} g^2$。

本章不给 PDE 推导（结论级，按 spec 口径），只用它的读法：式 (16) 右端两项正好对应「确定性的输运」与「随机性的摊平」。这个二分是 §6 的关键——PF-ODE 的全部内容就是把第二项吸收进第一项，让一个**确定性**速度场复现同样的密度演化。

## 5. 逆时 SDE：Anderson 定理

生成是时间的反演。Anderson 定理（结论级陈述，本书引自 [Song et al., 2021](../references.md)，arXiv:2011.13456，[A] 的转述）：若前向过程由 $dx = f\thinspace dt + g\thinspace dW$（ $t: 0 \to 1$）生成，则把轨迹时间倒转（ $t: 1 \to 0$）得到的过程仍是 Itô 扩散，满足**逆时 SDE**（reverse-time SDE）：

$$
dx = \left[ f(x, t) - g(t)^2\thinspace \nabla_x \log p_t(x) \right] dt + g(t)\hspace{5mu} d\bar W \qquad (17)
$$

**符号**： $d\bar W$：逆时间维纳过程； $dt < 0$（积分方向 $1 \to 0$）； $p_t$：前向过程的时刻 $t$ 边缘密度。

逐项读：漂移里的 $f$ 原样保留（弹簧还在）；多出的 $-g^2 \nabla \log p_t$ 是反演的全部代价——**逆着时间走，必须知道密度景观的坡度**，用它抵消扩散的摊平并把轨迹拉回高密度区；扩散项系数不变，逆向仍每步注噪（随机性不消失，只是被 score 项对冲）。对 VP-SDE 代入 $f = -\frac{1}{2}\beta x$、 $g^2 = \beta$：

$$
dx = -\frac{1}{2} \beta(t) \left[ x + 2\thinspace \nabla_x \log p_t(x) \right] dt + \sqrt{\beta(t)}\hspace{5mu} d\bar W
$$

定理里唯一的未知量是 $\nabla \log p_t$——而 §3 的 DSM 恰好交付它（ $s_\theta$）。训练与采样在此合龙：**整个框架只需要一个网络 $s_\theta(x, t)$，其余全是闭式动力学**。式 (17) 用 $s_\theta$ 替换真 score 即得可执行的采样 SDE；§7 将看到它的 Euler 离散化就是 7.2 的 ancestral 采样。

score 项的系数 $g^2$（而不是别的什么倍数）可以从无穷小 Bayes 读出直觉：前向一小步 $x \to x + f\Delta t + g\sqrt{\Delta t}\thinspace z$ 是高斯转移，其逆转移按 Bayes 正比于「前向转移概率 $\times$ 密度比 $p_t(x) / p_{t + \Delta t}(x + \Delta x)$」；密度比的 $\log$ 对 $\Delta x$ 展开，一阶项正是 $-g^2 \nabla \log p_t\thinspace \Delta t$——逆时漂移比「单纯倒放 $-f$」多出的一份，就是在补偿前向扩散造成的密度涂抹。 $\Delta t \to 0$ 时高阶项消失，剩下式 (17) 的精确形式。这也解释了为什么扩散项系数原样保留：随机性本身是各向同性的，时间反演不改变它的强度，只改变漂移该往哪儿拉。

## 6. Probability-flow ODE：同一条密度流的确定性版本

### 6.1 推导：把扩散项折进漂移

逆时 SDE 每步注噪，轨迹随机。能不能有一条**确定性**动力学，让它的轨迹密度在每个 $t$ 都与前向 SDE 的边缘 $p_t$ 一致？可以，而且构造是强制的。任何确定性流 $dx/dt = v(x, t)$ 输运密度的方式由连续性方程给出：

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left( v\thinspace p_t \right) \qquad (18)
$$

**符号**： $v$：待求的速度场。

目标：选 $v$ 使式 (18) 与 Fokker-Planck 式 (16) 的右端相等（同一个 $p_t$ 演化）。关键代数恒等式：

$$
\Delta p_t = \nabla \cdot \left( \nabla p_t \right) = \nabla \cdot \left( p_t\thinspace \nabla \log p_t \right) \qquad (19)
$$

**符号**：第二步用 $\nabla p = p\thinspace \nabla \log p$ 把梯度改写成「密度 $\times$ score」。

代回式 (16) 右端并合并散度：

$$
-\nabla \cdot (f p_t) + \frac{1}{2} g^2\thinspace \nabla \cdot \left( p_t \nabla \log p_t \right) = -\nabla \cdot \left( \left[ f - \frac{1}{2} g^2\thinspace \nabla \log p_t \right] p_t \right)
$$

与式 (18) 对照，读出速度场——**probability-flow ODE**（PF-ODE；[Song et al., 2021](../references.md)，[A]）：

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2}\thinspace g(t)^2\thinspace \nabla_x \log p_t(x) \qquad (20)
$$

**符号**： $t$ 正向 $0 \to 1$ 时该 ODE 复现前向密度涂抹；采样沿 $t: 1 \to 0$ 积分。

推导的实质就一步：式 (19) 把「抹平密度」翻译成「沿 $-\nabla \log p$ 方向搬运密度」——扩散对密度的净效果等价于一个指向低密度的漂移；把这个等效漂移从随机项里拿出来并入确定性速度场，随机项清零。两个注记防误读：(i) $v$ 不唯一——任何满足 $\nabla \cdot (w\thinspace p_t) = 0$ 的无散修正场都能加上去，式 (20) 是把扩散系数取零的最简选择（也正是逆时 SDE 式 (17) 的「去噪极限」：比较两式，PF-ODE = 逆时 SDE 抽掉 $d\bar W$、并把 score 项减半）；(ii) $v$ 依赖 $p_t$ 本身——这不是循环论证，score 已由 $s_\theta$ 提供，代入即得可积分的 ODE。

对 VP-SDE 展开（ $f = -\frac{1}{2}\beta x$， $g^2 = \beta$），并用换算式 (9) 写成 $\varepsilon_\theta$ 口径：

$$
\frac{dx}{dt} = -\frac{1}{2} \beta(t) \left[ x + s_\theta(x, t) \right] = -\frac{1}{2} \beta(t) \left[ x - \frac{\varepsilon_\theta(x, t)}{\sqrt{1 - \bar\alpha(t)}} \right] \qquad (21)
$$

**符号**：积分方向 $t: 1 \to 0$； $\bar\alpha(t)$：式 (14)。

这个式子是 7.4 半章的母体：DDIM 的 $\sigma = 0$ 更新是它的离散化（7.4 证明），DPM-Solver 是利用其半线性结构（线性项 $-\frac{1}{2}\beta x$ 闭式积分、只对方括号内网络项做展开）的高阶求解器。

### 6.2 确定性采样：噪声到数据是一个映射

PF-ODE 没有噪声项， $x(1) \mapsto x(0)$ 是确定性双射（ $v$ 光滑时流可逆）。三个直接后果：

1. **latent 语义**：每个噪声种子唯一对应一张图，种子空间即 latent 空间——球面插值、轨迹反演（把真实图片沿 $t: 0 \to 1$ 积分回噪声再编辑）都由双射性免费支持。
2. **误差预算全在离散化**：随机采样器每步注噪、靠多步平均掉误差；ODE 采样器的误差只有数值积分的截断，阶数与步数是可计算的账——这是 7.4「采样器军备竞赛」发生在 PF-ODE 上的原因。
3. **似然可算**：确定性双射 + 连续 change of variables，见下节。

还有一个定位注记：PF-ODE 使扩散模型在数学上成为一种连续归一化流（continuous normalizing flow）——7.1 对照表里 Flow 与 Diffusion 两行的「似然可解性」一列由此合并。两家剩下的差别在训练侧：离散 flow 直接以似然为目标，训练每步都要付 Jacobian 行列式的账（7.1 §5 的对数行列式项，架构必须专门设计使其可算）；扩散把似然从训练里整个移出——训练只做回归（便宜、架构自由），似然作为 PF-ODE 的事后副产品按需评估（贵、但只在需要密度时付一次）。

### 6.3 精确似然：连续 change of variables

沿 ODE 轨迹，对数密度的变化率由速度场的散度给出（instantaneous change of variables）：

$$
\frac{d \log p_t(x(t))}{dt} = -\nabla \cdot v(x(t), t) = -\mathrm{tr}\left( \frac{\partial v}{\partial x} \right) \qquad (22)
$$

**符号**： $v$：式 (20) 的速度场； $\partial v / \partial x$： $d \times d$ Jacobian。

从 $t = 1$ 积分到 $t = 0$（等价写法：对 $0 \to 1$ 取积分）：

$$
\log p_0(x(0)) = \log p_1(x(1)) + \int_0^1 \nabla \cdot v(x(t), t)\thinspace dt \qquad (23)
$$

**符号**： $p_1 \approx \mathcal{N}(0, I)$ 已知；积分沿采样轨迹反向重走一遍。

7.1 的对照表里 diffusion 的「似然不可解」至此被补上缺口：**PF-ODE 给出精确似然**，代价写在式 (22) 的 trace 里——精确求迹要对 $d$ 个输入维各做一次向量-Jacobian 积（ $d$ 次 backward 量级/每步）；score-SDE 论文的实现用 Hutchinson 随机估计器 $`\mathrm{tr}(J) = \mathbb{E}_u[u^\top J u]`$（ $u$ 取 Rademacher 或高斯随机向量，一次 backward 出无偏估计）把每步压到一次额外 backward，再把整条轨迹的散度积分交给黑盒 ODE solver 的容差控制（[Song et al., 2021](../references.md)，[A]；走读锚点 [yang-song/score_sde_pytorch](../references.md)，[A]）。该论文以此在 CIFAR-10 上报告了与 flow 家族同口径的 bits/dim 似然——扩散模型从此同时握住 7.1 表里的两列：高质量采样与精确密度。训练侧对应物是 §3 末尾的账：想让 DSM 目标与这个似然对齐，损失权重取 $\lambda(t) = g(t)^2$。

## 7. 统一视角：一个网络，一族采样器

把 §4-§6 装进一张图。三条动力学共享同一边缘密度族 $\lbrace p_t\rbrace_{t \in [0,1]}$，共享同一个学习对象 $s_\theta$（或经式 (9) 换算的 $\varepsilon_\theta$）：

```
                 前向 SDE（固定，无参数，加噪 t: 0 → 1）
   x_0 ──────────────────────────────────────────────────► x_1
 p_data          dx = f(x,t)dt + g(t)dW                   ≈ N(0,I)
                  密度演化：Fokker-Planck（式 16）
   ▲                                                            │
   │      反向动力学（采样 t: 1 → 0），唯一未知量 s_θ ≈ ∇log p_t  │
   │                                                            │
   │   逆时 SDE（式 17）：dx = [f − g²s_θ]dt + g dW̄    （随机）  │
   │     └─ Euler–Maruyama 离散 ≈ DDPM ancestral（7.2，见下）    │
   │   PF-ODE（式 20）：dx/dt = f − ½g²s_θ             （确定）  │
   │     ├─ Euler 离散 = DDIM σ=0（7.4 证明）                    │
   │     ├─ 高阶 solver：Heun / DPM-Solver（7.4）                │
   │     └─ 精确似然：散度积分（式 23）                          │
   └────────────────────────────────────────────────────────────┘
```

「DDPM ancestral ≈ 逆时 SDE 的 Euler–Maruyama 离散」可以当场验证，不止口号。对逆时 SDE 沿 $t \to t - \Delta t$ 做一步 Euler–Maruyama（ $\Delta t = 1/T$， $dt = -\Delta t$， $d\bar W \to \sqrt{\Delta t}\thinspace z$）：

$$
x(t - \Delta t) = x(t) + \frac{1}{2} \beta(t) \left[ x(t) + 2 s_\theta \right] \Delta t + \sqrt{\beta(t) \Delta t}\hspace{5mu} z \qquad (24)
$$

代入离散对应 $\beta(t) \Delta t = \beta_i$、 $\bar\alpha$ 关系与式 (9)，整理：

$$
x_{i-1} = x_i + \frac{\beta_i}{2} x_i + \beta_i\thinspace s_\theta + \sqrt{\beta_i}\thinspace z = \frac{1}{\sqrt{\alpha_i}} \left( x_i + \beta_i\thinspace s_\theta \right) + \sqrt{\beta_i}\thinspace z + O\left( \beta_i^2 \right) \qquad (25)
$$

**符号**：第二步用 $1/\sqrt{\alpha_i} = 1/\sqrt{1 - \beta_i} = 1 + \beta_i/2 + O(\beta_i^2)$，并把 $\beta_i^2 s_\theta$ 量级项收入余项。

而 7.2 的 ancestral 更新（§3.1 与 §3.3 的两式合并、并经本章式 (9) 换写成 score 口径）恰为 $x_{i-1} = \frac{1}{\sqrt{\alpha_i}} (x_i + \beta_i s_\theta) + \sigma_i z$：取 $\sigma_i = \sqrt{\beta_i}$ 时与式 (25) 到一阶完全一致——**7.2 的采样算法是逆时 SDE 的一阶随机离散**，离散与连续的两个方差选择（ $\beta_i$ vs $\tilde\beta_i$，7.2 §5.2）之差也在 $O(\beta_i^2)$ 余项里。统一表：

| 动力学 | 方程 | 离散化 → 采样器 | 随机性 | 精确似然 |
| --- | --- | --- | --- | --- |
| 逆时 SDE（式 (17)） | $dx = [f - g^2 s_\theta] dt + g\thinspace d\bar W$ | Euler–Maruyama → DDPM ancestral（式 (25)） | 每步注噪 | 否 |
| PF-ODE（式 (20)） | $dx/dt = f - \frac{1}{2} g^2 s_\theta$ | Euler → DDIM（ $\sigma = 0$），7.4 | 无 | 是（式 (23)） |
| PF-ODE | 同上 | 高阶 solver → Heun（§8）/ DPM-Solver，7.4 | 无 | 是 |

更深一层：逆时 SDE 也不是唯一的随机方案。score-SDE 指出一族「每步注噪量可调」的逆时扩散共享同一 $p_t$（式 (17) 是注噪取满的一端，PF-ODE 是注噪为零的另一端）；DDIM 的 $\sigma$ 参数正是这条滑杆的离散对应——中间档位的质量-步数权衡留到 7.4。

## 8. 工程化收口：EDM 的设计空间

方程层面统一之后，剩下的自由度摆在哪里？EDM（[Karras et al., 2022](../references.md)，arXiv:2206.00364，[A]）给出系统答案：把噪声尺度提升为一等坐标，设计与训练解耦。EDM 记号下样本写成 $x = x_0 + \sigma \varepsilon$（信号系数恒 1、噪声尺度 $\sigma$ 显式），本章的两族 SDE 都是它的参数化特例：

$$
\mathrm{VP:}\quad \sigma(t) = \sqrt{\frac{1 - \bar\alpha(t)}{\bar\alpha(t)}} = \mathrm{SNR}(t)^{-1/2} \qquad (26)
$$

VE 情形下 $\sigma(t)$ 就是式 (15) 的 $\sigma(t)$ 本身，无需换算。

**符号**： $\sigma(t)$：EDM 噪声尺度； $\mathrm{SNR}(t)$：7.2 §2.3 定义的信噪比的连续版。

在此坐标上，EDM 把设计空间拆成互不纠缠的三轴：

1. **网络预处理（preconditioning）**：不让裸网络 $F_\theta$ 直接当去噪器，而是包一层代数外壳 $D_\theta(x; \sigma) = c_{\mathrm{skip}}(\sigma)\thinspace x + c_{\mathrm{out}}(\sigma)\thinspace F_\theta\negthinspace\left( c_{\mathrm{in}}(\sigma)\thinspace x;\ c_{\mathrm{noise}}(\sigma) \right)$。系数由「 $F_\theta$ 的输入与回归目标都保持单位方差」这一要求反解出来： $c_{\mathrm{skip}} = \sigma_d^2 / (\sigma^2 + \sigma_d^2)$、 $c_{\mathrm{out}} = \sigma \sigma_d / \sqrt{\sigma^2 + \sigma_d^2}$、 $c_{\mathrm{in}} = 1/\sqrt{\sigma^2 + \sigma_d^2}$、 $c_{\mathrm{noise}} = \ln(\sigma) / 4$（ $\sigma_d$：数据标准差，EDM 取 $0.5$；时间条件的嵌入改成噪声尺度的对数）。端点读法： $\sigma \to 0$ 时 $D_\theta \to x$（恒等旁路接管，网络不用学「复制输入」）； $\sigma \to \infty$ 时 $D_\theta \to c_{\mathrm{out}} F_\theta$（网络全权）。7.2 §5.1 的三种预测参数化是这个外壳的特例位置—— $\varepsilon$/$`x_0`$/$`v`$ 之争在 EDM 坐标下变成「 $c_{\mathrm{skip}}$ 与 $c_{\mathrm{out}}$ 怎么随 $\sigma$ 分配」一个连续问题。
2. **损失权重 $\lambda(\sigma)$ 与 $\sigma$ 采样分布**：§3 式 (11) 已证明权重是自由度；EDM 选 $\lambda(\sigma) = 1/c_{\mathrm{out}}(\sigma)^2$ 使有效回归误差跨 $\sigma$ 均匀，并配 log-normal 形的 $\sigma$ 采样密度——「哪些噪声水平值得更多训练样本」成为显式旋钮（7.5 的 logit-normal $t$ 采样是同一思想在 FM 坐标的版本）。
3. **采样器**：确定性 PF-ODE + Heun 二阶方法（预测-校正梯形步，每步 2 次 NFE、局部截断误差比 Euler 低一阶）作默认——同样的网络，求解器阶数是纯粹的质量/NFE 旋钮，留给 7.4 对账。

EDM 的结论是机制性的：schedule、参数化、权重、求解器在方程层面互相正交，调其中任何一个不需要重证其余——这是 SDE/ODE 统一框架付给工程的直接红利。

## 总结

本章公式清单（编号为章内序号）：

| 对象 | 公式 | 要点 |
| --- | --- | --- |
| score | $s(x) = \nabla_x \log p(x)$，式 (1) | 消掉归一化常数；Langevin 采样的接口 |
| score matching | 式 (2)；分部积分版式 (3) | 靶未知 → 散度版每样本 $d$ 次 backward |
| DSM | 式 (5)；Vincent 等价式 (6) | 条件 score 回归 ≡ 边缘 score 回归（同梯度） |
| 关键恒等式 | $`\nabla \log q_t = \mathbb{E}_{q(x_0 \mid x_t)}[\nabla \log q(x_t \mid x_0)]`$，式 (7) | 边缘 score = 条件 score 的后验均值 |
| 高斯核靶 | $\nabla \log q(x_t \mid x_0) = -\varepsilon/\sqrt{1-\bar\alpha_t}$，式 (8) | 回归靶 = 注入噪声，训练免费 |
| score↔ $\varepsilon$ | $s_\theta = -\varepsilon_\theta / \sqrt{1-\bar\alpha_t}$，式 (9) | 逐点代数换算 |
| Tweedie | $\mathbb{E}[x_0 \mid x_t] = (x_t + (1-\bar\alpha_t) s_\theta)/\sqrt{\bar\alpha_t}$，式 (10) | $x_0$-prediction = 后验均值 |
| VP-SDE | $dx = -\frac{1}{2}\beta x\thinspace dt + \sqrt{\beta}\thinspace dW$，式 (13)； $\bar\alpha(t) = e^{-\int_0^t \beta}$，式 (14) | DDPM 链的扩散极限；边缘一致 |
| VE-SDE | $dx = \sqrt{d\sigma^2/dt}\thinspace dW$，式 (15) | $f = 0$；方差膨胀 |
| Fokker-Planck | $\partial_t p_t = -\nabla\cdot(f p_t) + \frac{1}{2} g^2 \Delta p_t$，式 (16) | 结论级；输运 + 摊平二分 |
| 逆时 SDE | $dx = [f - g^2 \nabla\log p_t] dt + g\thinspace d\bar W$，式 (17) | Anderson 定理；唯一未知量 = score |
| PF-ODE | $dx/dt = f - \frac{1}{2} g^2 \nabla \log p_t$，式 (20)；VP 版式 (21) | 同密度流的确定性版本（关键恒等式式 (19)） |
| 精确似然 | $d\log p_t/dt = -\nabla\cdot v$，式 (22)(23) | trace 成本 + Hutchinson 估计 |
| 统一 | DDPM ancestral = 逆时 SDE 一阶离散，式 (24)(25) | DDIM/高阶 solver = PF-ODE 离散，7.4 |
| EDM | $\sigma(t) = \mathrm{SNR}^{-1/2}$（VP），式 (26)；preconditioning 三系数 | 设计空间三轴解耦 |

## 练习

1.（推导）补全 Vincent 定理的两处细节：(a) 严格写出式 (7) 推导中「梯度与积分交换」所需的条件，并验证高斯核满足；(b) 证明式 (6) 的常数 $C_t$ 对高斯核等于 $`\frac{1}{2}\mathbb{E}_{q_t}\Vert\nabla\log q_t\Vert^2 - \frac{d}{2(1-\bar\alpha_t)}`$，从而确认 $J_t \le J_{\mathrm{DSM},t}$ 恒成立（条件期望收缩 $L^2$ 模长： $\mathbb{E}\Vert\mathbb{E}[c \mid x_t]\Vert^2 \le \mathbb{E}\Vert c\Vert^2$）。
2.（推导）一维闭式全链。设 $p_{\mathrm{data}} = \mathcal{N}(0, 1)$，VE-SDE 取 $\sigma^2(t) = t$。(a) 写出 $q_t$、真 score $s(x, t)$ 与 PF-ODE $dx/dt$；(b) 从 $t = 1$ 到 $t = 0$ 解出 ODE 轨迹 $x(t)$，验证 $x(1) \sim q_1$ 时 $x(0) \sim p_{\mathrm{data}}$；(c) 用式 (23) 核算该轨迹上 $\log p_0$ 与 $\log q_1$ 的差，与散度积分逐项对上。
3.（计算）linear schedule（7.2 §2.3， $T = 1000$， $\beta_1 = 10^{-4}$、 $\beta_T = 0.02$）下，连续化 $\beta(t) = T \beta_{\lceil tT \rceil}$。(a) 用式 (14) 的积分近似计算 $\bar\alpha(0.5)$ 与 $\bar\alpha(1.0)$，与 7.2 §2.3 表的离散值（ $\bar\alpha_{500} = 0.0786$、 $\bar\alpha_{1000} = 4.04 \times 10^{-5}$）比较相对误差；(b) 由式 (26) 算 $t = 0.5$ 处的 EDM $\sigma$，并给出该 $\sigma$ 下 preconditioning 三个系数的数值（ $\sigma_d = 0.5$）。
4.（代码）教学用，非优化实现：二维 two-moons 数据上的 DSM + 双采样器对照。(a) 用式 (5)(8) 训练 score MLP（ $t$ 经 sinusoidal embedding 注入，VP 核 $\beta(t)$ 线性）；(b) 同一网络分别跑逆时 SDE 的 Euler–Maruyama（式 (24)）与 PF-ODE 的 Euler（式 (21)），在 100 / 1000 NFE 两档下对比样本散布；(c) 选 5 个数据点沿 PF-ODE 正向积分回 $t = 1$，用式 (23)（二维可精确求迹）报告 $\log p_0$，并与直方图密度估计对照。

## 参考

- [Song et al., 2021](../references.md)：Score-Based Generative Modeling through Stochastic Differential Equations，arXiv:2011.13456，[A]（key: score-sde）——式 (13)(15)(16)(17)(20)(22)(23)，VE/VP 分类、Anderson 逆时公式、 $\lambda(t) = g(t)^2$ 权重、Hutchinson 似然实现
- [Karras et al., 2022](../references.md)：Elucidating the Design Space of Diffusion-Based Generative Models，arXiv:2206.00364，[A]（key: edm）——式 (26) 的 $\sigma$ 坐标、preconditioning 系数、损失权重与 Heun 采样器
- [Vincent, 2011](../references.md)：A Connection Between Score Matching and Denoising Autoencoders，Neural Computation 23(7):1661–1674，DOI 10.1162/NECO_a_00142（无 arXiv），[A]（key: vincent-2011）——式 (6) 等价性定理原始出处
- [yang-song/score_sde_pytorch](../references.md)：score-SDE 官方 PyTorch 实现，[A]（key: score-sde-repo）——VP/VE-SDE、PF-ODE 与似然评估的走读锚点
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）—— $\varepsilon$-参数化与 ancestral 采样（本章经 7.2 交叉引用，式 (9)(25) 的换算对象）
