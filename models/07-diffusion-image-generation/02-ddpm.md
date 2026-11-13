# 7.2 DDPM 完整推导（Denoising Diffusion Probabilistic Models: The Complete Derivation）

> 本章把 DDPM 从一条高斯加噪链推到底：前向闭式解（高斯卷积递推）、noise schedule 的 linear/cosine 对比与频谱读法、反向过程的参数化、后验 $q(x_{t-1} \mid x_t, x_0)$ 的 Bayes 配方推导、ELBO 三项分解、 $\varepsilon$-参数化、 $L_{simple}$ 与严格 ELBO 的 SNR weighting 差异、 $\varepsilon$/$x_0$/$v$ 三种预测参数化换算、 $\sigma_t$ 的两个选择 | 前置：1.1（KL 散度与 CE 分解式 (6)）、6.3（ELBO 双路径推导式 (3)(4)、reparameterization、Gaussian KL 闭式式 (6)(7)）、7.1（生成模型家族定位） | 锚点：DDPM [A]、iDDPM [A]、progressive distillation [A]（v-prediction 原始出处）；hojonathanho/diffusion [A]、lucidrains/denoising-diffusion-pytorch（教学参照）

## 1. 问题设定：一条固定的加噪链 + 一个学习的逆过程

DDPM（[Ho et al., 2020](../references.md)，arXiv:2006.11239，[A]）把图像生成建模为一个潜变量模型。数据 $x_0 \sim p_{\mathrm{data}}$ 是 $d$ 维向量（论文主实验 CIFAR-10 为 $32 \times 32 \times 3$、 $d = 3072$，像素从 $\lbrace0, ..., 255\rbrace$ 线性缩放到 $[-1, 1]$；另有 LSUN $256 \times 256$ 实验）。潜变量是与 $x_0$ 同维的一串噪声图 $x_1, ..., x_T$， $T = 1000$。全书扩散约定（W4 约定页）：**$x_0$ = 数据端、 $t = T$ = 纯噪声端**；本章沿用 DDPM 原文的离散下标 $t \in \lbrace1, ..., T\rbrace$，与 7.3 起的连续记号 $t \in [0, 1]$ 的对应是 $t_{\mathrm{cont}} = t/T$（ $t=1$ 连续记号 = 噪声端），方向词统一：前向 = 加噪（ $t$ 增）、反向 = 去噪生成（ $t$ 减）。

模型分布定义为

$$
p_\theta(x_0) = \int p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t) \thinspace dx_{1:T}
$$

**符号**： $p(x_T) = \mathcal{N}(0, I)$：固定的标准高斯端点； $p_\theta(x_{t-1} \mid x_t)$：学习的反向单步转移；积分遍历全部中间噪声图。

与 6.3 的 VAE 对照，DDPM 的结构性差异有三点，每一点都决定了后面的数学长什么样：

1. **latent 与数据同维**。 $x_t$ 不是压缩表示，是加了噪的图像本身——没有「编码器把 $x$ 压成 $z$」这回事。
2. **后验是固定的，不学**。 $q(x_{1:T} \mid x_0)$ 是一条人为给定的高斯链（§2），不含参数。6.3 里 ELBO 的 gap 是近似后验误差（6.3 式 (4)），这里的 gap 来自反向转移的模型误差——摊销网络从「编码器」换成了「去噪器」。
3. **层级深度 $T = 1000$**。VAE 是一层 latent，DDPM 是 1000 层；能训得动，全靠固定后验的闭式性质（§2.2）把 1000 层摊成一次并行采样。

链式结构总览（本章全部公式都在这条链上展开）：

```
前向 q（固定，无参数）：  q(x_t | x_{t-1}) = N(√α_t · x_{t-1}, β_t · I)
反向 p_θ（学习）：        p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t² · I)

  x_0 ──q──► x_1 ──q──► x_2 ──q──► ··· ──q──► x_{T-1} ──q──► x_T ≈ N(0, I)
  ◄──p_θ──   ◄──p_θ──   ◄──p_θ──          ◄──p_θ──    ◄──p_θ──
数据端 t=0                                                  噪声端 t=T（连续记号 t=1）

跳步直达（§2.2 推导目标）：q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t) · I)
训练时的 t 不靠逐层滚动，任意 t 一步采出——训练并行、采样串行
```

**符号**： $\alpha_t, \beta_t, \bar\alpha_t$ 在 §2.1 定义； $\mu_\theta, \sigma_t$ 在 §3 定义。

训练侧一次前向就能拿到任意 $t$ 的监督信号（类似 1.1 teacher forcing 的并行性），生成侧则必须串行走完 $T$ 步、每步一次网络评估——这个「训练并行、采样串行」的不对称是 7.4 整章（采样器与蒸馏）存在的根本原因，先记账：DDPM 原生采样 = 1000 次 NFE（network function evaluations）。

把 DDPM 与 6.3 的 VAE 逐项摆在一起，后面的推导每一步行了什么棋都有对应物：

| 对象 | VAE（6.3） | DDPM（本章） |
| --- | --- | --- |
| latent | $z$，低维 | $x_1, ..., x_T$，与 $x_0$ 同维， $T = 1000$ 层 |
| 后验 $q$ | 学习的摊销 encoder $q_\phi(z \mid x)$ | 固定高斯链 $q(x_{1:T} \mid x_0)$，无参数 |
| ELBO 的 gap | 后验近似误差 | 反向转移 $p_\theta$ 的模型误差 |
| 可算性支点 | Gaussian KL 闭式（6.3 式 (6)） | 前向闭式解（本章式 (5)）+ 后验配方（式 (15)） |
| 采样 | 一次 decoder 前向 | $T$ 步串行，每步一次网络评估 |

固定后验不是偷工减料：它换来两条解析性质——任意 $t$ 的边缘闭式（§2.2）与以 $x_0$ 为条件的反向闭式（§3.2），前者让训练可并行，后者给反向均值提供了精确的回归靶。学习的全部自由度被集中到 $\mu_\theta$ 一个函数上，这是 DDPM 数学干净的根源。

## 2. 前向过程与闭式解

### 2.1 单步转移与方差保持（VP）设计

前向过程是固定的高斯马尔可夫链：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left( \sqrt{1 - \beta_t}\thinspace x_{t-1},\ \beta_t I \right) \qquad (1)
$$

**符号**： $\beta_t \in (0, 1)$：第 $t$ 步的噪声强度，人为给定的递增序列（schedule，§2.3）； $I$： $d$ 维单位阵——噪声各向同性、逐像素独立。

定义两个缩写，本章反复使用：

$$
\alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s \qquad (2)
$$

**符号**： $\alpha_t$：单步信号保留率； $\bar\alpha_t$：从 $x_0$ 到 $x_t$ 的累计信号保留率，单调递减。

式 (1) 的均值系数为什么取 $\sqrt{1 - \beta_t}$ 而不是更直觉的 $1$（即 $x_t = x_{t-1} + \sqrt{\beta_t}\thinspace \varepsilon$）？算一笔方差账。设 $x_{t-1}$ 的逐维方差为 $v$，则式 (1) 给出

$$
\mathrm{Var}(x_t) = (1 - \beta_t)\thinspace v + \beta_t
$$

**符号**： $\mathrm{Var}$：逐维方差（各维对称，只看一维）。

$v = 1$ 是不动点： $\mathrm{Var}(x_t) = 1 - \beta_t + \beta_t = 1$。数据归一化到近似单位方差后，整条链上 $x_t$ 的边际方差恒为 1——这就是 variance preserving（VP）设计。若均值系数取 1，方差每步加 $\beta_t$、 $T$ 步后膨胀到 $1 + \sum \beta_t$，激活尺度随 $t$ 漂移，网络要在 1000 个不同输入尺度间分摊容量；VP 把尺度问题从链上消掉，让「 $t$ 的信息」只通过信噪比一个通道流动（§2.3 的 SNR 由此成为唯一需要追踪的量）。

### 2.2 闭式解：高斯卷积递推（逐步展开）

训练需要 $q(x_t \mid x_0)$：直接从 $x_0$ 跳到第 $t$ 步的边缘分布。它是 $t$ 个高斯转移的复合，逐层卷积；用采样式展开最干净。先把式 (1) 重参数化（6.3 §2.5 的同一技巧）：

$$
x_t = \sqrt{\alpha_t}\thinspace x_{t-1} + \sqrt{\beta_t}\thinspace z_t, \qquad z_t \sim \mathcal{N}(0, I) \qquad (3)
$$

**符号**： $z_t$：第 $t$ 步注入的标准高斯，各步独立。等式两边对 $x_t \mid x_{t-1}$ 同分布：右端均值 $\sqrt{\alpha_t} x_{t-1}$、方差 $\beta_t I$，即式 (1)。

所需的概率论工具只有一条（**独立高斯和引理**）： $u \sim \mathcal{N}(\mu_1, \sigma_1^2 I)$、 $v \sim \mathcal{N}(\mu_2, \sigma_2^2 I)$ 独立，则

$$
u + v \sim \mathcal{N}\left( \mu_1 + \mu_2,\ (\sigma_1^2 + \sigma_2^2)\thinspace I \right) \qquad (4)
$$

（独立高斯的和仍高斯——由矩母函数相乘即得；方差相加是独立性的直接推论。）

先看两步展开，确认模式。把 $x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}} z_{t-1}$ 代入式 (3)：

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\thinspace x_{t-2} + \underbrace{\sqrt{\alpha_t \beta_{t-1}}\thinspace z_{t-1} + \sqrt{\beta_t}\thinspace z_t}_{z_{\mathrm{mix}}}
$$

**符号**： $z_{\mathrm{mix}}$：本步要合成的对象——两个独立高斯噪声的加权和； $z_{t-1}, z_t$ 独立。

两个噪声项是独立零均值高斯的缩放，由引理式 (4)，其和仍是零均值高斯、方差为缩放系数平方之和：

$$
\alpha_t \beta_{t-1} + \beta_t = \alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}
$$

第一步展开 $\beta_{t-1} = 1 - \alpha_{t-1}$ 与 $\beta_t = 1 - \alpha_t$，第二步消项。于是合成噪声可写成单一样本 $\sqrt{1 - \alpha_t \alpha_{t-1}}\thinspace \bar z$（ $\bar z \sim \mathcal{N}(0, I)$），得

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\thinspace x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\thinspace \bar z
$$

模式已经出现：跳两步后，信号系数 = 两步 $\alpha$ 之积的平方根，噪声方差 = 1 减去信号系数平方。对一般步数用归纳法。**归纳基**（ $s = 1$）：式 (3) 即 $x_t = \sqrt{\bar\alpha_t^{(1)}} x_{t-1} + \sqrt{1 - \bar\alpha_t^{(1)}} z_t$，其中 $\bar\alpha_t^{(1)} = \alpha_t$。**归纳假设**：对某 $s \ge 1$，

$$
x_t = \sqrt{\bar\alpha_t^{(s)}}\thinspace x_{t-s} + \sqrt{1 - \bar\alpha_t^{(s)}}\thinspace \bar z, \qquad \bar\alpha_t^{(s)} = \prod_{i=0}^{s-1} \alpha_{t-i}
$$

**符号**： $\bar\alpha_t^{(s)}$：从 $x_{t-s}$ 到 $x_t$ 的累计信号保留率； $\bar z$：合成标准高斯。

**归纳步骤**：把 $x_{t-s} = \sqrt{\alpha_{t-s}}\thinspace x_{t-s-1} + \sqrt{\beta_{t-s}}\thinspace z_{t-s}$ 代入假设式：

$$
x_t = \sqrt{\bar\alpha_t^{(s)} \alpha_{t-s}}\thinspace x_{t-s-1} + \sqrt{\bar\alpha_t^{(s)} \beta_{t-s}}\thinspace z_{t-s} + \sqrt{1 - \bar\alpha_t^{(s)}}\thinspace \bar z
$$

对两个噪声项再用引理式 (4)，合成方差为

$$
\bar\alpha_t^{(s)} \beta_{t-s} + \left(1 - \bar\alpha_t^{(s)}\right) = \bar\alpha_t^{(s)} \left(1 - \alpha_{t-s}\right) + 1 - \bar\alpha_t^{(s)} = 1 - \bar\alpha_t^{(s)} \alpha_{t-s} = 1 - \bar\alpha_t^{(s+1)}
$$

归纳成立。取 $s = t$（并记合成噪声为 $\varepsilon$），得**前向闭式解**与其采样式：

$$
q(x_t \mid x_0) = \mathcal{N}\left( \sqrt{\bar\alpha_t}\thinspace x_0,\ (1 - \bar\alpha_t)\thinspace I \right) \qquad (5)
$$

$$
x_t = \sqrt{\bar\alpha_t}\thinspace x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) \qquad (6)
$$

**符号**： $\varepsilon$：注入 $x_t$ 的全部累计噪声的单一代表样本。式 (5) 是密度口径，式 (6) 是采样口径，二者等价。

同一递推在密度层面的读法： $q(x_t \mid x_0) = \int q(x_t \mid x_{t-1})\thinspace q(x_{t-1} \mid x_0)\thinspace dx_{t-1}$ 是两个高斯密度的卷积，引理式 (4) 正是「高斯卷积仍是高斯、方差相加」这条密度事实的采样口径。递推每走一步，信号系数乘一个 $\sqrt{\alpha}$、噪声方差按「1 减累计保留率」归位——采样路径展开与密度卷积是同一枚硬币的两面，本书推导一律走采样口径（写起来没有积分号），需要密度语义时回引本段。

式 (6) 是 DDPM 训练可行的支点：给定 $x_0$，任意 $t$ 的 $x_t$ 一次采样直达，不需要真的滚动 $t$ 步链。训练时对每个样本独立随机抽 $t$（§4.4），1000 层的层级模型被摊平成「一次采样 + 一次回归」。同时式 (6) 也给出了噪声的精确语义： $\varepsilon$ 是 $x_t$ 中全部随机性的充分统计量——后面 $\varepsilon$-参数化（§3.3）让网络回归的正是它。

### 2.3 noise schedule：linear vs cosine 与频谱读法

噪声调度（noise schedule）是序列 $\lbrace\beta_t\rbrace$ 的选取，等价地由 $\lbrace\bar\alpha_t\rbrace$ 刻画。两个出处明确的实例：

**linear schedule**（DDPM 原文）： $\beta_t$ 从 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$ 线性递增， $T = 1000$：

$$
\beta_t = \beta_1 + \frac{t - 1}{T - 1}\left( \beta_T - \beta_1 \right), \qquad \beta_1 = 10^{-4},\ \beta_T = 0.02 \qquad (7)
$$

**cosine schedule**（iDDPM，[Nichol & Dhariwal, 2021](../references.md)，arXiv:2102.09672，[A]）：直接定义 $\bar\alpha_t$ 为余弦形，再回推 $\beta_t$：

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right), \qquad \beta_t = \min\left( 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}},\ 0.999 \right) \qquad (8)
$$

**符号**： $s = 0.008$：小偏移量，防止 $t = 0$ 附近 $\beta_t$ 过小导致的数值问题； $\min$ 截断： $f(T) = \cos^2(\pi/2) = 0$ 使 $\bar\alpha_T \to 0$，末端 $\beta_t \to 1$，截断在 $0.999$ 防奇异。

比较两条 schedule 的正确口径是**信噪比**：由式 (6)， $x_t$ 中信号功率 $\propto \bar\alpha_t$、噪声功率 $= 1 - \bar\alpha_t$，定义

$$
\mathrm{SNR}_t = \frac{\bar\alpha_t}{1 - \bar\alpha_t} \qquad (9)
$$

**符号**： $`\mathrm{SNR}_t`$：第 $t$ 步的信噪功率比；下文用分贝 $`\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10} \mathrm{SNR}_t`$。

按式 (2)(7)(8) 计算（数值由公式直接算出，schedule 公式出处如上）：

| $t$ | $\bar\alpha_t$（linear） | $`\mathrm{SNR}_{\mathrm{dB}}`$（linear） | $\bar\alpha_t$（cosine） | $`\mathrm{SNR}_{\mathrm{dB}}`$（cosine） |
| --- | --- | --- | --- | --- |
| 1 | 0.9999 | 40.0 | 0.99996 | 43.8 |
| 100 | 0.8970 | 9.4 | 0.9721 | 15.4 |
| 200 | 0.6590 | 2.9 | 0.8987 | 9.5 |
| 500 | 0.0786 | −10.7 | 0.4938 | −0.1 |
| 700 | 0.00697 | −21.5 | 0.2031 | −5.9 |
| 800 | 0.00153 | −28.1 | 0.0940 | −9.8 |
| 900 | 2.75e−4 | −35.6 | 0.0241 | −16.1 |
| 1000 | 4.04e−5 | −43.9 | $\to 0$ | $\to -\infty$ |

两条曲线的形状差异在 log-SNR 坐标下看得最清楚。linear-β： $\beta_t$ 小， $\log \bar\alpha_t = \sum_{s \le t} \log(1 - \beta_s) \approx -\sum_{s \le t} \beta_s$，等差数列求和出 $t$ 的二次函数——这只是 $\log \bar\alpha$ 的形状； $\log \mathrm{SNR} = \log \bar\alpha_t - \log(1 - \bar\alpha_t)$ 还要叠上 $-\log(1 - \bar\alpha_t)$ 项， $t$ 小时 $1 - \bar\alpha_t$ 从 0 快速升起、该项陡峭下跌，合成的 log-SNR 两端陡、中段相对缓。cosine：取 $s \to 0$ 的极限看结构， $\bar\alpha_t = \cos^2(\pi t / 2T)$、 $1 - \bar\alpha_t = \sin^2(\pi t / 2T)$，故

$$
\mathrm{SNR}_t = \frac{\cos^2\left( \pi t / 2T \right)}{\sin^2\left( \pi t / 2T \right)}, \qquad \log \mathrm{SNR}_t = 2 \log \cos\left( \frac{\pi t}{2T} \right) - 2 \log \sin\left( \frac{\pi t}{2T} \right)
$$

对 $u = \pi t / 2T$ 求导： $\frac{d}{du} \log \mathrm{SNR} = -2 \frac{\sin u}{\cos u} - 2 \frac{\cos u}{\sin u} = -\frac{2}{\sin u \cos u} = -\frac{4}{\sin(2u)}$——由 $\sin(2u) \le 1$ 知斜率绝对值 $\ge 4$，最小值 $4$ 在中点 $u = \pi/4$ 处取得，中段近似匀速；两端斜率发散，是整条曲线最陡的部分。带 $s = 0.008$ 时 $`\mathrm{SNR}_t = \cos^2(u_t) / (\sin^2(u_t) - \sin^2(u_0))`$（由 $\cos^2 a - \cos^2 b = \sin^2 b - \sin^2 a$ 化简）， $\sin^2(u_0) \approx 1.5 \times 10^{-4}$ 只在 $t$ 极小时偏离上式形状。一句话对比：linear-β 的 log-SNR 前半程掉得更急（ $t = 500$ 已 −10.7 dB，同期 cosine 仍在 0 dB 附近）、尾部却留着 $\bar\alpha_T > 0$ 的残余以有限斜率收场；cosine 中段匀速、两端陡，尾部斜率发散恰好把 $\bar\alpha$ 在 $t = T$ 压到 0。同样的 1000 步，cosine 的 $\Delta_t \log \mathrm{SNR}$ 分布更均匀——联系式 (25)（ $w_t \approx \frac{1}{2} \Delta_t \log \mathrm{SNR}$），cosine 的严格 ELBO 权重本身就更平。

**频谱读法**。式 (6) 在傅里叶域逐频率成立（傅里叶变换是正交线性变换，保持各向同性高斯的独立性，噪声谱不混合）： $\hat x_t(f) = \sqrt{\bar\alpha_t}\thinspace \hat x_0(f) + \sqrt{1 - \bar\alpha_t}\thinspace \hat\varepsilon(f)$，其中 $\hat\varepsilon(f)$ 各频率等功率（白噪声 = 平谱）。自然图像的功率谱随频率快速衰减、能量集中在低频（经验事实），于是对每个 $t$ 存在交叉频率 $f_c(t)$： $\bar\alpha_t \cdot P_{\mathrm{img}}(f_c) = (1 - \bar\alpha_t) \cdot P_{\varepsilon}$， $f > f_c$ 的频段信噪比小于 1，信息已被淹没。 $t$ 增大 $\Rightarrow$ $f_c$ 单调下移——**加噪是从高频到低频的逐层淹没，去噪反过来：先恢复低频构图与色块，后恢复高频纹理**。这也解释了生成中途图像「先成形、后出细节」的普遍观察。

schedule 的好坏由此可以量化。linear 的 log-SNR 两端陡、中段缓， $t = 500$（半程）时 SNR 已掉到 −10.7 dB；且尾部在 $\bar\alpha_T = 4.04 \times 10^{-5} > 0$ 处收场——末端 $\sqrt{\bar\alpha_T} = 0.0064$，已低于 8-bit 图像在 $[-1, 1]$ 标度下的一个量化台阶 $1/127.5 \approx 0.0078$：从 $t = 979$ 起到 $T$ 的 22 步残余信号都低于量化台阶，**对 32×32 图像几乎没有可恢复信息，链的长度被浪费**。cosine 把信息破坏摊匀：半程时 SNR 仍在 0 dB 附近，且 $\bar\alpha_t$ 恰在 $t = T$ 归零（低于量化台阶的只有从 $t = 995$ 起的 6 步），1000 步步步有活干。iDDPM 的动机正是此，且作用的分辨率区间与本节算例精确对齐：论文实测 linear schedule 在高分辨率图像上表现良好，但在 64×64 与 32×32 上 sub-optimal——小分辨率下 linear 的末端过于嘈杂，最后约四分之一的 latent 已几乎是纯噪声（原文 Figure 3 口径），与上面的 32×32 量化账互为印证。cosine 的设计意图（原文）： $\bar\alpha_t$ 在中段近似线性下降、在 $t = 0$ 与 $t = T$ 两端变化平缓，防止噪声水平在端点附近突变；其论文实验中 cosine schedule 改善 log-likelihood（[Nichol & Dhariwal, 2021](../references.md)，[A]），随后成为标准选项之一。

另外记一个后面要用的数字：linear schedule 下 $\bar\alpha_T = 4.04 \times 10^{-5}$，即 $q(x_T \mid x_0) \approx \mathcal{N}(0, I)$ 到很高精度——这给了模型端点 $p(x_T) = \mathcal{N}(0, I)$ 的合法性，§4.1 的 $L_T$ 项将因此近似为零。

## 3. 反向过程与后验推导

### 3.1 反向过程的参数化

生成 = 从 $x_T \sim \mathcal{N}(0, I)$ 出发逐步走反向链。DDPM 把每步参数化为对角高斯：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left( \mu_\theta(x_t, t),\ \sigma_t^2 I \right) \qquad (10)
$$

**符号**： $\mu_\theta$：网络给出的反向均值； $\sigma_t^2$：反向方差，DDPM 取固定值（ $\beta_t$ 或 $\tilde\beta_t$，§5.2），iDDPM 让它可学。

高斯假设的合法性：当 $\beta_t$ 足够小，前向单步只注入微量噪声，真实的反向条件分布 $q(x_{t-1} \mid x_t)$（对数据分布取边际）与它的前向镜像一样接近高斯——DDPM 原文以此论证式 (10) 的函数形式在 $\beta_t \to 0$ 极限下无损；严格的连续时间版本（reverse-time SDE）在 7.3 给出。直觉上， $\beta_t$ 大时反向一步的「可能来源」是数据流形上大片的 $x_0$ 混合，真实反向条件是多峰的，单高斯装不下； $\beta_t$ 小把来源收窄到 $x_t$ 的局部邻域，混合退化为单峰。这个论证同时说明了为什么 $T$ 要取到 1000 量级： $T$ 越大、每步 $\beta_t$ 越小，式 (10) 的函数形式误差越小。注意式 (10) 的条件里**没有 $x_0$**：生成时 $x_0$ 正是待求之物，网络只能从 $(x_t, t)$ 推断它——这是反向与下面后验的本质区别。

### 3.2 后验 $q(x_{t-1} \mid x_t, x_0)$：Bayes + 配方（完整推导）

训练目标（§4）需要的不是 $q(x_{t-1} \mid x_t)$，而是**以 $x_0$ 为条件的后验（posterior）** $q(x_{t-1} \mid x_t, x_0)$——它闭式可算，将充当反向均值 $\mu_\theta$ 的回归靶。对联合分布 $q(x_{t-1}, x_t \mid x_0)$ 用 Bayes：

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)\thinspace q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \qquad (11)
$$

**符号**：分子第一因子：前向转移（条件里多了 $x_0$）；第二因子：式 (5) 的边缘（ $t-1$ 版）；分母：式 (5)。

两个简化立刻到位：(i) 前向链是马尔可夫的， $x_t$ 只依赖 $x_{t-1}$，故 $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$，即式 (1)；(ii) 分母 $q(x_t \mid x_0)$ 不含 $x_{t-1}$，对「 $x_{t-1}$ 的分布」而言只是归一化常数。于是逐分量（各维独立，先看一维， $d$ 维由逐维相乘回收）：

$$
q(x_{t-1} \mid x_t, x_0) \propto \exp\left( -\frac{(x_t - \sqrt{\alpha_t}\thinspace x_{t-1})^2}{2\beta_t} - \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\thinspace x_0)^2}{2(1 - \bar\alpha_{t-1})} \right) \qquad (12)
$$

**符号**： $\propto$：相差一个不依赖 $x_{t-1}$ 的因子。

两个高斯密度之积对 $x_{t-1}$ 仍是高斯（配方即得）。把指数按 $x_{t-1}$ 的二次、一次项展开收集：

$$
-\frac{1}{2}\left[ \underbrace{\left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}} \right)}_{1 / \tilde\beta_t} x_{t-1}^2 - 2 \underbrace{\left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}} x_0 \right)}_{\tilde\mu_t / \tilde\beta_t} x_{t-1} + C(x_t, x_0) \right] \qquad (13)
$$

**符号**： $C(x_t, x_0)$：所有不含 $x_{t-1}$ 的项，归入归一化常数；两处下花括号是高斯配方的标准形态——二次项系数 = 精度（方差倒数），一次项系数 = 精度 $\times$ 均值； $\tilde\beta_t, \tilde\mu_t$ 即下面两步要注实的方差与均值。

**方差**：精度取倒数，

$$
\tilde\beta_t = \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}} \right)^{-1} = \frac{\beta_t (1 - \bar\alpha_{t-1})}{\alpha_t (1 - \bar\alpha_{t-1}) + \beta_t} = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\thinspace \beta_t \qquad (14)
$$

关键化简在分母： $\alpha_t (1 - \bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar\alpha_{t-1} + \beta_t = \alpha_t - \bar\alpha_t + (1 - \alpha_t) = 1 - \bar\alpha_t$（第二步用 $\bar\alpha_t = \alpha_t \bar\alpha_{t-1}$，第三步用 $\beta_t = 1 - \alpha_t$）。

**均值**：一次项系数乘方差，

$$
\begin{aligned}
\tilde\mu_t(x_t, x_0) &= \tilde\beta_t \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}} x_0 \right) \\
&= \frac{\sqrt{\alpha_t}\thinspace (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\thinspace x_t + \frac{\sqrt{\bar\alpha_{t-1}}\thinspace \beta_t}{1 - \bar\alpha_t}\thinspace x_0
\end{aligned} \qquad (15)
$$

第二步把式 (14) 乘进去逐项约简。合起来：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left( \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I \right)
$$

两个健全性检查。**(i) $t = 1$**： $\bar\alpha_0 = 1$（空积）， $\tilde\beta_1 = 0$，后验退化为 $x_0$ 上的点质量——条件里给了 $x_0$、 $x_1$ 又由 $x_0$ 加噪而来， $x_0$ 当然确定，一致。**(ii) 插值结构**： $\tilde\mu_t$ 是 $x_t$ 与 $x_0$ 的正系数线性组合； $t$ 小时 $x_0$ 系数主导（噪声少、后验贴近数据）， $t$ 大时 $x_t$ 系数主导。两系数之和一般不为 1（它不是凸组合，是高斯乘积的精度加权）。

### 3.3 $\varepsilon$-参数化：把后验均值里的 $x_0$ 换掉

式 (15) 的 $\tilde\mu_t$ 依赖 $x_0$，网络拿不到。但训练时 $x_t$ 由式 (6) 采样而来， $x_0$ 与 $\varepsilon$ 互为线性表达：

$$
x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left( x_t - \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon \right) \qquad (16)
$$

**符号**： $\varepsilon$：式 (6) 中采出的那个噪声样本。

代入式 (15)。 $x_t$ 的系数变为（第二项里 $\sqrt{\bar\alpha_{t-1}} / \sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$）：

$$
\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} + \frac{\beta_t}{(1 - \bar\alpha_t)\sqrt{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t}{1 - \bar\alpha_t} = \frac{1}{\sqrt{\alpha_t}}
$$

末步再次用到式 (14) 的同一化简 $\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$。 $\varepsilon$ 的系数：

$$
-\frac{\sqrt{\bar\alpha_{t-1}}\thinspace \beta_t\thinspace \sqrt{1 - \bar\alpha_t}}{(1 - \bar\alpha_t)\thinspace \sqrt{\bar\alpha_t}} = -\frac{\beta_t}{\sqrt{\alpha_t}\thinspace \sqrt{1 - \bar\alpha_t}}
$$

合得后验均值的 $\varepsilon$ 形式：

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\thinspace \varepsilon \right) \qquad (17)
$$

**符号**：右端除 $\varepsilon$ 外全是已知量（ $x_t$ 是反向步的输入， $\alpha_t, \beta_t, \bar\alpha_t$ 是 schedule 常数）。

网络的活儿由此明确：学一个 $\varepsilon_\theta(x_t, t)$ 逼近 $\varepsilon$，再按式 (17) 的形状组装均值：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\thinspace \varepsilon_\theta(x_t, t) \right) \qquad (18)
$$

**符号**： $\varepsilon_\theta(x_t, t)$：去噪网络，输入噪声图与时间步，输出噪声预测，shape 与 $x_t$ 相同（ $[B, C, H, W] \to [B, C, H, W]$）；网络结构 = U-Net，见 §5.3 预告与 7.6 详拆。

注意式 (18) 不是新假设，而是式 (17) 的参数化搬运—— $\mu_\theta$ 与 $\tilde\mu_t$ 的差完全被 $\varepsilon_\theta$ 与 $\varepsilon$ 的差决定：

$$
\tilde\mu_t - \mu_\theta(x_t, t) = \frac{\beta_t}{\sqrt{\alpha_t}\thinspace \sqrt{1 - \bar\alpha_t}}\left( \varepsilon - \varepsilon_\theta(x_t, t) \right) \qquad (19)
$$

这个恒等式是 §4.2 把 KL 项化成 $\varepsilon$ 回归的枢纽。

式 (18) 的更新结构值得多看一眼： $\mu_\theta$ 是「 $x_t$ 本身（缩放 $1/\sqrt{\alpha_t}$）减去沿噪声预测方向的一步修正」。网络从不直接输出 $x_{t-1}$，而是输出「 $x_t$ 里哪部分是噪声」——输入与输出的主干由恒等通路承担，网络只学残差。这与 2.1 残差连接的设计同源：恒等主干保住信号，学习集中在修正项上。反向一步「当前值减修正项」的形状也是 7.4 全部 ODE 求解器（DDIM、DPM-Solver）离散更新的离散原型。

## 4. ELBO 分解与 $L_{simple}$

### 4.1 ELBO 三项分解（完整推导）

训练目标是 $\log p_\theta(x_0)$ 的变分下界。DDPM 是层级 latent 模型，ELBO 工具与 6.3 完全相同（6.3 式 (3)(4)）：以固定后验 $q(x_{1:T} \mid x_0)$ 为变分分布，

$$
-\log p_\theta(x_0) \le \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[ \log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})} \right] =: L_{\mathrm{vlb}} \qquad (20)
$$

**符号**： $L_{\mathrm{vlb}}$：变分上界（要最小化的对象）；gap = $\mathrm{KL}(q(x_{1:T} \mid x_0) \Vert p_\theta(x_{1:T} \mid x_0))$，此处 $q$ 固定、gap 全部由模型侧承担。

代入两条链的因子分解。前向链对 $q(x_{1:T} \mid x_0)$ 做链式法则，每个因子本来就以 $x_0$ 为条件，再由马尔可夫性砍掉更早的依赖： $q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}, x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$（严格相等，不含近似）；反向： $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$。于是

$$
L_{\mathrm{vlb}} = \mathbb{E}_q\left[ -\log p(x_T) - \sum_{t=1}^{T} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1}, x_0)} \right]
$$

上式里 $q$ 侧仍是前向单步转移—— $p_\theta$ 要逐个去比的对象。直接比没有闭式（ $p_\theta$ 高斯、 $q$ 前向也高斯但条件不对齐：一个给 $x_t$、一个给 $x_{t-1}$）。关键一步是把 $q$ 的因子改写成以 $x_t$ 为条件的后验，让比值的分子分母「条件对齐」。

目标是把 $q(x_t \mid x_{t-1}, x_0)$ 换成 §3.2 的后验。对 $t \ge 2$ 用 Bayes（对联合 $q(x_{t-1}, x_t \mid x_0)$）：

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\thinspace q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

代入求和并拆分：

$$
\sum_{t=2}^{T} \log q(x_t \mid x_{t-1}, x_0) = \sum_{t=2}^{T} \log q(x_{t-1} \mid x_t, x_0) + \sum_{t=2}^{T} \left[ \log q(x_t \mid x_0) - \log q(x_{t-1} \mid x_0) \right]
$$

第二个和逐项相消（伸缩）： $= \log q(x_T \mid x_0) - \log q(x_1 \mid x_0)$。而 $t = 1$ 项单独是 $\log q(x_1 \mid x_0)$，恰好与之抵消。伸缩与抵消不是技巧性巧合：它们把「逐层前向比」重排成「端点 + 逐层后验比 + 首层重建」三件可分别闭式处理的对象—— $q(x_T \mid x_0)$ 与 $p(x_T)$ 都比的是端点， $q(x_{t-1} \mid x_t, x_0)$ 与 $p_\theta(x_{t-1} \mid x_t)$ 条件对齐（都给 $x_t$）， $p_\theta(x_0 \mid x_1)$ 单独成项。把所有项归位：

$$
L_{\mathrm{vlb}} = \mathbb{E}_q\left[ \log \frac{q(x_T \mid x_0)}{p(x_T)} + \sum_{t=2}^{T} \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} - \log p_\theta(x_0 \mid x_1) \right]
$$

最后一步把 $T$ 层期望压回单层：第 $t$ 项的 $\log(q(x_{t-1} \mid x_t, x_0) / p_\theta(x_{t-1} \mid x_t))$ 只依赖 $(x_{t-1}, x_t, x_0)$，故对 $q(x_{1:T} \mid x_0)$ 的期望可按塔性质先对「除 $(x_{t-1}, x_t)$ 外的所有中间变量」积分——联合里只剩 $q(x_{t-1}, x_t \mid x_0) = q(x_{t-1} \mid x_t, x_0)\thinspace q(x_t \mid x_0)$。再对内层 $x_{t-1}$ 的期望认出 KL 的定义：

$$
\mathbb{E}_{q(x_t \mid x_0)} \mathbb{E}_{q(x_{t-1} \mid x_t, x_0)}\left[ \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} \right] = \mathbb{E}_{q(x_t \mid x_0)}\left[ \mathrm{KL}\left( q(x_{t-1} \mid x_t, x_0)\thinspace \Vert \thinspace p_\theta(x_{t-1} \mid x_t) \right) \right]
$$

**符号**：外层期望：对 $t$ 时刻的加噪样本；内层：在给定 $(x_t, x_0)$ 下后验与模型反向的 KL。

同理， $L_T$ 项只依赖 $(x_T, x_0)$、期望收成 $\mathrm{KL}(q(x_T \mid x_0) \Vert p(x_T))$； $L_0$ 项只依赖 $(x_0, x_1)$。得**三项分解**：

$$
L_{\mathrm{vlb}} = \underbrace{\mathrm{KL}\left( q(x_T \mid x_0)\thinspace \Vert \thinspace p(x_T) \right)}_{L_T} + \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(x_t \mid x_0)}\left[ \mathrm{KL}\left( q(x_{t-1} \mid x_t, x_0)\thinspace \Vert \thinspace p_\theta(x_{t-1} \mid x_t) \right) \right]}_{L_{t-1}} + \underbrace{\mathbb{E}_{q(x_1 \mid x_0)}\left[ -\log p_\theta(x_0 \mid x_1) \right]}_{L_0} \qquad (21)
$$

**符号**： $L_T$：端点匹配项； $L_{t-1}$（ $t = 2, ..., T$）：逐时去噪匹配项； $L_0$：重建项。

三项的角色与可算性各不相同：

- **$L_T$：常数，扔掉**。 $q$ 与 $p(x_T)$ 都不含 $\theta$。且由 §2.3 的账， $\bar\alpha_T = 4.04 \times 10^{-5}$，直接代入高斯 KL（式 (22)）得 $L_T \approx \frac{\bar\alpha_T}{2} \Vert x_0\Vert^2$；对单位方差归一化数据 $\mathbb{E}\Vert x_0\Vert^2 = d$，CIFAR 规模下 $L_T \approx 3072 \times 4.04 \times 10^{-5} / 2 \approx 0.06$ nats/图——端点失配在数值上就是零。
- **$L_{t-1}$：主战场**。两个高斯之间的 KL，闭式（§4.2）。
- **$L_0$：离散化高斯似然**。图像像素本质是 $\lbrace0, ..., 255\rbrace$ 的离散值，DDPM 的做法：把像素映到 $[-1, 1]$（即 $(\text{pixel} - 127.5)/127.5$），定义

$$
p_\theta(x_0 \mid x_1) = \prod_{i=1}^{d} \int_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)} \mathcal{N}\left( x;\ \mu_\theta^i(x_1, 1),\ \sigma_1^2 \right) dx
$$

**符号**： $\delta_{+}(x) = x + 1/255$（ $x = 1$ 时取 $+\infty$）、 $\delta_{-}(x) = x - 1/255$（ $x = -1$ 时取 $-\infty$）：以真值为中心、半宽 $1/255$、全宽 $2/255 = 1/127.5$ 的量化 bin——相邻像素级的间距恰好是 $1/127.5$，bin 无缝平铺。积分把连续密度折成离散概率。 $t = 1$ 步噪声极小（ $\beta_1 = 10^{-4}$）， $\mu_\theta(x_1, 1) \approx x_1$，此项近似「 $x_1$ 落回正确 bin」的对数概率，实现上直接可算，不参与下面 $\varepsilon$ 化简。

### 4.2 高斯 KL 闭式：从 $L_{t-1}$ 到 $\varepsilon$ 回归

$d$ 维各向同性高斯之间的 KL（1 维推导见 6.3 式 (6)，逐维相加 + 均值项推广）：

$$
\mathrm{KL}\left( \mathcal{N}(\mu_1, \sigma_1^2 I)\thinspace \Vert \thinspace \mathcal{N}(\mu_2, \sigma_2^2 I) \right) = \frac{d}{2}\left( \frac{\sigma_1^2}{\sigma_2^2} - \log \frac{\sigma_1^2}{\sigma_2^2} - 1 \right) + \frac{\Vert \mu_1 - \mu_2 \Vert^2}{2\sigma_2^2} \qquad (22)
$$

**符号**： $d$：维度；右端第一项只含方差，第二项是加权均方误差。

代入 $L_{t-1}$： $q$ 侧是 $(\tilde\mu_t, \tilde\beta_t I)$， $p_\theta$ 侧是 $(\mu_\theta(x_t, t), \sigma_t^2 I)$。方差 $\sigma_t^2$ 固定（DDPM 口径，§5.2）时，式 (22) 第一项不含 $\theta$，只剩

$$
L_{t-1} = \frac{1}{2\sigma_t^2}\thinspace \mathbb{E}_{q(x_t \mid x_0)} \left\Vert \tilde\mu_t - \mu_\theta(x_t, t) \right\Vert^2 + C \qquad (23)
$$

**符号**： $C$：与 $\theta$ 无关的常数（方差项之和）。

代入式 (19) 的均值差，平方后系数出模：

$$
L_{t-1} = \underbrace{\frac{\beta_t^2}{2\sigma_t^2\thinspace \alpha_t\thinspace (1 - \bar\alpha_t)}}_{w_t}\thinspace \mathbb{E}_{x_0, \varepsilon} \left\Vert \varepsilon - \varepsilon_\theta\left( \sqrt{\bar\alpha_t}\thinspace x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon,\ t \right) \right\Vert^2 + C \qquad (24)
$$

**符号**： $w_t$：第 $t$ 步的标量权重；期望内的 $x_t$ 已按式 (6) 展开，期望对 $(x_0, \varepsilon)$ 取。

式 (24) 就是严格 ELBO 的逐时项：**带 $t$ 依赖权重 $w_t$ 的噪声回归**。到这一步，训练目标已经从「1000 层 latent 的变分上界」变成了「对随机 $t$、随机噪声量的去噪回归加权和」。

### 4.3 SNR weighting 差异推导： $L_{simple}$ 丢掉了什么

把 $w_t$ 用 SNR（式 (9)）改写。取 DDPM 的方差选择 $\sigma_t^2 = \beta_t$（§5.2）， $w_t = \beta_t / (2\alpha_t(1 - \bar\alpha_t))$。分子分母同乘 $\bar\alpha_{t-1}$ 并注意 $\beta_t = 1 - \alpha_t = (\bar\alpha_{t-1} - \bar\alpha_t)/\bar\alpha_{t-1}$：

$$
\frac{\beta_t}{1 - \bar\alpha_t} = \frac{\bar\alpha_{t-1} - \bar\alpha_t}{\bar\alpha_{t-1}(1 - \bar\alpha_t)} = 1 - \frac{\bar\alpha_t (1 - \bar\alpha_{t-1})}{\bar\alpha_{t-1}(1 - \bar\alpha_t)} = 1 - \frac{\mathrm{SNR}_t}{\mathrm{SNR}_{t-1}}
$$

中间步展开分子验证： $\bar\alpha_{t-1}(1 - \bar\alpha_t) - \bar\alpha_t(1 - \bar\alpha_{t-1}) = \bar\alpha_{t-1} - \bar\alpha_t$。所以

$$
w_t = \frac{1}{2\alpha_t}\left( 1 - \frac{\mathrm{SNR}_t}{\mathrm{SNR}_{t-1}} \right) \approx \frac{1}{2}\thinspace \Delta_t \log \mathrm{SNR} \qquad (25)
$$

**符号**： $`\Delta_t \log \mathrm{SNR} = \log \mathrm{SNR}_{t-1} - \log \mathrm{SNR}_t`$；近似号用 $\beta_t$ 小时 $1/(2\alpha_t) \approx 1/2$ 及 $1 - r \approx -\log r$。

式 (25) 给出严格 ELBO 的加权画像：**每步的权重 = 该步造成的 SNR 相对下降量**——信息销毁最剧烈的步占最大权重。代入 linear schedule 的端点数值： $w_1 = 1/(2\alpha_1) \approx 0.500$（ $t=1$ 处 $1 - \bar\alpha_1 = \beta_1$， $\beta_1$ 约去）； $w_T = \beta_T / (2\alpha_T(1 - \bar\alpha_T)) = 0.02/(2 \times 0.98 \times 0.99996) \approx 0.0102$。首末相差约 49 倍——**严格 ELBO 把近 50 倍的权重压在小 $t$ 端**，那些步的噪声量 $\beta_1 = 10^{-4}$ 级别，去噪任务接近「把像素微调回量化 bin」。

DDPM 的简化（论文消融后采用）：**把 $w_t$ 整个扔掉**：

$$
L_{simple}(\theta) = \mathbb{E}_{t \sim \mathrm{Uniform}\lbrace1, T\rbrace,\ x_0,\ \varepsilon} \left\Vert \varepsilon - \varepsilon_\theta\left( \sqrt{\bar\alpha_t}\thinspace x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon,\ t \right) \right\Vert^2 \qquad (26)
$$

**符号**： $t$ 均匀采样使式 (24) 的求和变成期望；每个样本一次回归。

差异的机制读法：相对严格 ELBO， $L_{simple}$ 把小 $t$（低噪声、接近重建）的项**相对**降权、把大 $t$（高噪声、语义重建）的项相对升权——容量从「像素级微调」挪到「从半噪声里恢复结构」，这正是扩散模型生成质量的主战场；DDPM 论文的消融中 $L_{simple}$ 的样本质量（FID）优于带权版本（[Ho et al., 2020](../references.md)，[A]）。两个边界说明，防误读：(i) $L_{simple}$ 不是把小 $t$ 项删掉—— $t$ 均匀采样下它们照样进 batch，只是不再乘 49 倍的权重；(ii) 丢掉 $w_t$ 的代价是目标不再是似然下界，要算 NLL 得回到式 (21) 全量评估——「训练目标」与「评价指标」从此分家，7.3 的精确似然路线会把指标侧补回来。iDDPM 折中：均值仍用 $L_{simple}$ 训，另加小权重的 $L_{\mathrm{vlb}}$ 项供方差分支学习（hybrid objective，§5.2）。weighting 作为设计空间的系统梳理（ $\sigma$-参数化与损失权重的解耦）属 EDM 口径，放 7.3。

### 4.4 训练与采样算法

把 §2-§4 装盒（对应 DDPM 原文 Algorithm 1/2）：

```
训练（ε-参数化 + L_simple）                     采样（ancestral，T 步串行）
─────────────────────────────────          ─────────────────────────────────
repeat                                       x_T ~ N(0, I)
  x_0 ~ p_data（像素缩放到 [-1,1]）            for t = T, T-1, ..., 1:
  t   ~ Uniform{1, ..., T}                       z ~ N(0, I)（t = 1 时 z = 0）
  ε   ~ N(0, I)                                  x_{t-1} = (1/√α_t)(x_t
  x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε       (式 6)            − β_t/√(1−ᾱ_t)·ε_θ(x_t, t))
  对 ‖ε − ε_θ(x_t, t)‖² 做一步梯度下降                   + σ_t·z   (式 10)(18)
until 收敛                                   return x_0
```

教学用参考实现（非优化实现；对照锚点：官方 TF 实现在 [hojonathanho/diffusion](../references.md) 的 `diffusion_tf/` 目录，[A]；PyTorch 教学版见 [lucidrains/denoising-diffusion-pytorch](../references.md)，教学参照）：

```python
import torch

def training_loss(eps_net, x0, abar, T=1000):
    """教学用，非优化实现。x0: [B, C, H, W]，已缩放到 [-1, 1]；abar: 长度 T 的 ᾱ_t 查表。"""
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,), device=x0.device)      # t ~ Uniform{1..T}
    eps = torch.randn_like(x0)                               # ε ~ N(0, I)
    a = abar[t - 1].view(B, 1, 1, 1)                         # 查 ᾱ_t，广播到像素维
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps               # 采样式，式 (6)
    return torch.nn.functional.mse_loss(eps, eps_net(x_t, t))  # L_simple，式 (26)
```

三行核心（抽 $t$、采样式合成 $x_t$、MSE 回归 $\varepsilon$）就是式 (26) 的直译——DDPM 的训练循环简单到不像是 1000 层模型的目标函数，全部复杂度都已被 §2.2 的闭式解和 §4 的分解吸收。采样侧每步一次网络评估， $T = 1000$ 次 NFE 的账记在 §1，7.4 的全部采样器与蒸馏工作都在砍这笔账。

## 5. 预测参数化换算与 $\sigma_t$ 选择

### 5.1 $\varepsilon$ / $x_0$ / $v$ 三种预测参数化的等价换算

式 (6) 里 $(x_0, \varepsilon)$ 线性互表，网络回归哪个都行；第三个选择是两者的旋转组合。**三种参数化**（约定页口径）：

- **$x_0$-prediction**：网络 $\hat x_{0,\theta}(x_t, t)$ 直接回归数据。与 $\varepsilon$-prediction 经式 (16) 互转： $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon_\theta) / \sqrt{\bar\alpha_t}$。
- **$\varepsilon$-prediction**：本书默认，式 (18)(26)。
- **$v$-prediction**（[Salimans & Ho, 2022](../references.md)，arXiv:2202.00512，[A]，v-prediction 的原始出处；SD 2.x 系采用）：

$$
v \equiv \sqrt{\bar\alpha_t}\thinspace \varepsilon - \sqrt{1 - \bar\alpha_t}\thinspace x_0 \qquad (27)
$$

**符号**： $v$：信号-噪声正交基 $(x_0, \varepsilon)$ 旋转后的目标； $v_\theta(x_t, t)$：回归 $v$ 的网络。

给定 $v_\theta$ 反解 $(\hat x_0, \hat\varepsilon)$：

$$
\hat x_0 = \sqrt{\bar\alpha_t}\thinspace x_t - \sqrt{1 - \bar\alpha_t}\thinspace v_\theta, \qquad \hat\varepsilon = \sqrt{1 - \bar\alpha_t}\thinspace x_t + \sqrt{\bar\alpha_t}\thinspace v_\theta \qquad (28)
$$

验证第一式（第二式同理）： $\sqrt{\bar\alpha_t} x_t - \sqrt{1 - \bar\alpha_t}\thinspace v = \sqrt{\bar\alpha_t}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon) - \sqrt{1 - \bar\alpha_t}(\sqrt{\bar\alpha_t}\thinspace \varepsilon - \sqrt{1 - \bar\alpha_t}\thinspace x_0) = \bar\alpha_t x_0 + (1 - \bar\alpha_t) x_0 = x_0$（ $\varepsilon$ 两项相消）。式 (28) 的系数全部落在 $[0, 1]$——对照 $x_0$-prediction 的换算式含 $1/\sqrt{\bar\alpha_t}$，在 $t \to T$ 端（ $\bar\alpha_T = 4 \times 10^{-5}$）把网络误差放大约 157 倍；v-prediction 的换算是无除法的，这是它在蒸馏场景（7.4）数值稳定的原因。

同一网络误差经三种 loss 计的相对权重（设 $e = \varepsilon_\theta - \varepsilon$，换算均按式 (16)(27)(28) 一致进行）：由 $x_0$ 误差与 $\varepsilon$ 误差的关系 $\hat x_0 - x_0 = \sqrt{1 - \bar\alpha_t}/\sqrt{\bar\alpha_t}\thinspace e$ 得 $`\Vert x_0\text{-loss}\Vert = \Vert e\Vert^2 / \mathrm{SNR}_t`$；由式 (27) 与式 (16) 联立得 $\hat v - v = e/\sqrt{\bar\alpha_t}$，即 $\Vert v\text{-loss}\Vert = \Vert e\Vert^2 / \bar\alpha_t$。利用 $`1/\bar\alpha_t = 1 + (1 - \bar\alpha_t)/\bar\alpha_t = 1 + 1/\mathrm{SNR}_t`$，得到一个精确的加法结构：

$$
\Vert v\text{-loss}\Vert = \Vert e\Vert^2 + \frac{\Vert e\Vert^2}{\mathrm{SNR}_t} = \Vert\varepsilon\text{-loss}\Vert + \Vert x_0\text{-loss}\Vert
$$

即 **v-loss 恰好是 $\varepsilon$-loss 与 $x_0$-loss 之和**（一致换算、同一误差下）。汇总：

| 直接回归目标 | 同一误差 $e$ 的 loss（相对 $\lVert e \rVert^2$） | 两端行为 |
| --- | --- | --- |
| $x_0$ | $`\lVert e \rVert^2 / \mathrm{SNR}_t`$ | $t \to 0$ 端趋于 0（低压）； $t \to T$ 端放大 $1/\mathrm{SNR}$ 倍 |
| $\varepsilon$（ $L_{simple}$） | $\lVert e \rVert^2$ | 两端均匀 |
| $v$ | $`\lVert e \rVert^2 / \bar\alpha_t = \lVert e \rVert^2 (1 + 1/\mathrm{SNR}_t)`$ | $t \to 0$ 端 $\approx \varepsilon$-loss； $t \to T$ 端 $\approx x_0$-loss——各取两者在该端较平的一侧 |

**术语防混**：iDDPM 的 hybrid objective 里也写作 $v$ 的量是**方差对数插值系数**（式 (30)），均值仍走 $\varepsilon$-参数化，与式 (27) 的 v-prediction 同名不同物，引用文献时勿混（约定页已锁此口径：v-prediction 归 Salimans & Ho）。

### 5.2 $\sigma_t$ 的选择： $\beta_t$ vs $\tilde\beta_t$ vs 学习

式 (10) 的方差不进均值回归（式 (23) 的 $C$），但影响采样轨迹的随机性。两个固定选择都在 DDPM 原文：

$$
\sigma_t^2 = \beta_t \qquad \text{or} \qquad \sigma_t^2 = \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\thinspace \beta_t \qquad (29)
$$

**符号**： $\beta_t$：前向单步方差； $\tilde\beta_t$：式 (14) 的后验方差。

读法： $\tilde\beta_t$ 是「 $x_0$ 已知」时反向一步的真实不确定性， $\beta_t$ 略大（ $\tilde\beta_t < \beta_t$，因 $(1 - \bar\alpha_{t-1}) < (1 - \bar\alpha_t)$）——多出的部分是「 $x_0$ 未知」的补偿。DDPM 消融中两者样本质量相近。端点行为不同： $\tilde\beta_1 = 0$（ $t=1$ 后验本来就退化，见 §3.2 检查 (i)）， $\beta_1 > 0$ 则最后一步仍注入噪声；实现上 $t = 1$ 一律取 $z = 0$。iDDPM 让方差可学，参数化为两个端点的对数插值：

$$
\Sigma_\theta(x_t, t) = \exp\left( v \log \beta_t + (1 - v) \log \tilde\beta_t \right) \qquad (30)
$$

**符号**： $v$：网络输出的插值系数（**不是 v-prediction**，见 §5.1 防混注记）。可学方差使 $L_{\mathrm{vlb}}$ 全部项都参与优化，iDDPM 以此把 NLL 逼近下界、同时保留 $L_{simple}$ 的样本质量（[Nichol & Dhariwal, 2021](../references.md)，[A]）。

### 5.3 $\varepsilon_\theta$ 的网络：U-Net 契约（预告 7.6）

本章只需要 $\varepsilon_\theta$ 的 I/O 契约，结构详拆连同 DiT 路线放 7.6。DDPM 用 U-Net：输入 $x_t$ 与输出 $\varepsilon_\theta$ 同 shape（ $[B, C, H, W] \to [B, C, H, W]$），时间 $t$ 经 sinusoidal embedding（同 2.5 的 sinusoidal 编码）注入每个 ResBlock：

```
ε_θ 的 I/O 契约（DDPM U-Net；结构细节 → 7.6）

x_t: [B, 3, 32, 32] ──┐
                      ├─► U-Net ──► ε_θ(x_t, t): [B, 3, 32, 32]（与输入同 shape）
t:   [B] ──sinusoidal─┘
         embedding         down: 32²→16²→8²→4²（ResBlock × N，每块注入 t embedding）
                           mid:  ResBlock + self-attention（低分辨率处）
                           up:   4²→8²→16²→32²（encoder skip 拼接）
```

两个设计要点先记账：(i) 同 shape 契约使 U-Net 的「下采样-上采样 + skip」结构成为自然选择——细节走 skip 直达，语义在 bottleneck 处理；(ii) 条件只有标量 $t$，注入用 scale-shift（7.6 的 adaLN 是它的推广）。文本等条件进入同一骨架的机制（cross-attention / concat / in-context）是 7.6 的主题。

## 总结

本章公式清单（编号为章内序号）：

| 对象 | 公式 | 出处 |
| --- | --- | --- |
| 前向单步 | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$，式 (1) | 定义；VP 方差账 §2.1 |
| 前向闭式 | $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1{-}\bar\alpha_t) I)$，式 (5) | 高斯卷积递推 §2.2 |
| 采样式 | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1{-}\bar\alpha_t}\thinspace \varepsilon$，式 (6) | 训练并行的支点 |
| schedule | linear 式 (7)（ $\beta_1{=}10^{-4}, \beta_T{=}0.02$）；cosine 式 (8)（ $s{=}0.008$，clip 0.999） | DDPM / iDDPM |
| SNR | $`\mathrm{SNR}_t = \bar\alpha_t/(1{-}\bar\alpha_t)`$，式 (9)；linear $\bar\alpha_T = 4.04{\times}10^{-5}$ | 频谱读法 §2.3 |
| 后验 | $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$、 $\tilde\mu_t$ 式 (14)(15) | Bayes + 配方 §3.2 |
| $\varepsilon$-参数化 | $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon)$，式 (17)(18) | §3.3 |
| ELBO 三项 | $L_T + \sum L_{t-1} + L_0$，式 (21)； $L_T \approx 0$、 $L_0$ 离散高斯 | §4.1 |
| 逐时 KL | $L_{t-1} = w_t\thinspace \mathbb{E}\lVert \varepsilon - \varepsilon_\theta \rVert^2 + C$，式 (24) | 高斯 KL 式 (22) |
| SNR 权重 | $`w_t = \frac{1}{2\alpha_t}(1 - \mathrm{SNR}_t/\mathrm{SNR}_{t-1})`$，式 (25)； $w_1/w_T \approx 49$ | §4.3 |
| $L_{simple}$ | $\mathbb{E}\lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2$，式 (26) | 丢 $w_t$，FID 更优 |
| $v$-prediction | $v = \sqrt{\bar\alpha_t}\varepsilon - \sqrt{1{-}\bar\alpha_t}x_0$，式 (27)(28) | Salimans & Ho |
| $\sigma_t$ | $\beta_t$ vs $\tilde\beta_t$ 式 (29)；iDDPM 学习式 (30) | DDPM / iDDPM |

## 练习

1.（推导）跳步边缘分布。用 §2.2 的引理式 (4) 与归纳法证明：对 $s < t$， $q(x_t \mid x_s) = \mathcal{N}(\sqrt{\bar\alpha_t / \bar\alpha_s}\thinspace x_s,\ (1 - \bar\alpha_t / \bar\alpha_s)\thinspace I)$。提示：把 $x_s$ 当作「新的 $x_0$」，注意累计保留率变成比值。这个结果在 7.4 的跳步采样（DDIM）里直接复用。
2.（推导）补全 §3.2 的 $d$ 维细节：(a) 写出式 (12) 的 $d$ 维版本并验证归一化常数确实给出式 (14) 的方差；(b) 验证 $\tilde\beta_t \le \beta_t$ 与 $\tilde\beta_t \le 1 - \bar\alpha_{t-1}$（后验方差不超过任何一个母方差）；(c) 证明 $\tilde\mu_t$ 的两系数满足：当 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\thinspace \varepsilon$ 代回， $\mathbb{E}[\tilde\mu_t \mid x_0] \to x_0$ 于 $t \to 1$。
3.（计算）linear schedule（式 (7)， $T = 1000$）下：(a) 计算 $\bar\alpha_{500}$ 与 $`\mathrm{SNR}_{500}`$（dB），对照 §2.3 表；(b) 计算 $w_{500}$（式 (25)）并与 $w_1, w_T$ 比较，说明严格 ELBO 的权重分布形状；(c) 同一 $t$，cosine schedule 的 $\bar\alpha_t$ 是多少？两条曲线在哪个 $t$ 附近 $\mathrm{SNR}$ 差距最大？
4.（代码）教学用，非优化实现：二维 swiss-roll 数据上的迷你 DDPM。(a) 按式 (7) 预计算 $\lbrace\beta_t, \alpha_t, \bar\alpha_t\rbrace$ 三张表；(b) 用本节 `training_loss` 训练一个 4 层 MLP（ $t$ 经 sinusoidal embedding 拼接输入）；(c) 按 §4.4 采样算法生成 1000 点，与数据散点对比；(d) 对照实验： $\sigma_t^2 = \beta_t$ vs $\tilde\beta_t$（式 (29)），观察样本散布差异。

## 参考

- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——式 (1)(5)(6)(7)(10)(17)(18)(21)(24)(26)(29)， $T = 1000$、linear schedule、 $\varepsilon$-参数化、 $L_{simple}$、 $\sigma_t$ 两选择
- [Nichol & Dhariwal, 2021](../references.md)：Improved Denoising Diffusion Probabilistic Models，arXiv:2102.09672，[A]（key: iddpm）——式 (8)(30)，cosine schedule、可学方差与 hybrid objective
- [Salimans & Ho, 2022](../references.md)：Progressive Distillation for Fast Sampling of Diffusion Models，arXiv:2202.00512，[A]（key: progressive-distillation）——式 (27)(28)，v-prediction 原始出处
- [Kingma & Welling, 2013](../references.md)：Auto-Encoding Variational Bayes，arXiv:1312.6114，[A]（key: vae）——ELBO 与 reparameterization 工具（经 6.3）
- [Jonathan Ho, hojonathanho/diffusion](../references.md)：DDPM 官方 TF 实现（`diffusion_tf/` 目录），[A]（key: ddpm-repo）
- [lucidrains/denoising-diffusion-pytorch](../references.md)：DDPM 教学实现（key: ddpm-pytorch），教学参照
