# 7.1 生成模型家族统一对照（A Unified Map of Generative Model Families）

> 本章给出五大家族生成模型的统一坐标系——各自优化什么、怎么采样、似然是否可解、mode 行为如何——并以此定位本 Part 其余六章 | 前置：1.1（AR 分解、MLE 与 KL 方向）、6.3（VAE/ELBO、visual tokenizer） | 锚点：VAE [A]、GAN [A]、Glow [A]、DDPM [A]

## 1. 一个问题，五种算法

DDPM 的默认配置从纯噪声采出一张图要 1000 次网络前向（[Ho et al., 2020](../references.md)，arXiv:2006.11239，[A]）；GAN 的生成器只要 1 次。这个三个数量级的采样账单差距不是工程实现的优劣，而是目标函数选择的直接后果——五大家族在「用什么量逼近 $p_{data}$」上的分歧，决定了它们各自能算什么、采样要走几步。

先钉死本章的考察对象。一个生成模型要交付两样东西：

1. **一个可优化的训练目标**：给定来自未知真实分布 $p_{data}$ 的样本集，目标函数必须能给出关于参数 $\theta$ 的梯度，且梯度方向指向「模型分布 $p_\theta$ 更接近 $p_{data}$」；
2. **一条可执行的采样路径**：训练完成后，能从某个已知的简单分布（标准高斯、bos token）出发，产出近似服从 $p_\theta$ 的样本，并标明这条路径要走几步、每步几次网络前向。

五大家族——自回归（autoregressive, AR）、变分自编码器（variational autoencoder, VAE）、生成对抗网络（generative adversarial network, GAN）、归一化流（normalizing flow）、扩散模型（diffusion model）与 flow matching（FM）——对这两条的回答各不相同。本章用四个坐标把它们放进同一张表：

- **优化什么**：精确似然、似然的下界、minimax 对抗目标，还是一个回归目标；
- **怎么采样**：一次前向、顺序串行，还是沿时间轴迭代；
- **似然可解性**：$\log p_\theta(x)$ 能不能算出来、多贵——它决定模型能不能做密度估计、异常检测、按似然排序这类任务；
- **mode 行为**：模型覆盖数据全部 mode 还是坍缩到少数几个。

记号约定在此登记一次，本 Part 全部章节遵守（与 7.2 及锚点 SD3 的口径一致）：**$x_0$ = 数据端、$t = 1$ = 纯噪声端；前向过程（加噪）方向 $t: 0 \to 1$，反向过程（采样）方向 $t: 1 \to 0$**；噪声 $\varepsilon \sim \mathcal{N}(0, I)$；$\alpha_t = 1 - \beta_t$、$\bar\alpha_t = \prod_{s \le t} \alpha_s$。扩散语境下「建模对象 $x_0$」按场景指像素或 6.3 的 latent $z$，7.6 展开这条分工。五个家族里只有扩散与 FM 需要这条时间轴约定，其余四节只用通用概率记号；7.5 另附 FM 原文献的约定映射框，本章不提前引入。

## 2. AR：精确似然与顺序采样

AR 家族的训练目标在 1.1 已经钉死：链式法则给出联合分布的精确分解 $p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$（1.1 式 (3)），训练就是极大似然

$$
L_{AR}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \qquad (1)
$$

**符号**：$x_{<t} = (x_1, \ldots, x_{t-1})$：位置 $t$ 之前的全部 token；$T$：序列长度。式 (1) 与 1.1 式 (4) 的 per-token NLL 差常数因子 $1/T$——本章取未归一化的整序列求和口径，1.1 取逐位置平均口径；与 cross-entropy 的同一性论证见 1.1 §4，此处只列结论。

把式 (1) 放进四坐标：

- **优化什么**：精确似然。$\log p_\theta(x)$ 逐位相加即得，没有近似项、没有下界缝隙。
- **怎么采样**：祖先采样（ancestral sampling，1.1 §8）——从 bos 出发逐位置采 $x_t \sim p_\theta(\cdot \mid x_{<t})$，$T$ 步严格串行，第 $t$ 步必须等前 $t-1$ 步落地。一张 $256 \times 256$ 图像经 f16 离散 tokenizer 切成 $16 \times 16 = 256$ 个 token，就是 256 次串行前向；训练侧靠教师强制（teacher forcing）一次前向并行算完全部 $T$ 个位置，训练并行、生成串行的不对称由此而来（1.1 §4）。
- **似然**：精确且便宜，评估一个样本只要一次前向。
- **mode 行为**：MLE 优化的是 $\mathrm{KL}(p_{data} \| p_\theta)$（1.1 式 (6)），数据里出现的每个 mode 都必须分到概率，否则 $\log p_\theta \to -\infty$ 的惩罚无界——行为是全覆盖（mean-seeking）。

AR 进图像的前置工序是离散化：连续像素必须先经 6.3 的 VQ 族 tokenizer 变成 codebook 下标网格，图像才成为「词表 $K$、长度 $hw$ 的序列」，与文本同构。代价是生成质量先被 tokenizer 的重建上限卡一道（6.3 §1 的账）。收益是与 LLM 的原生可组合性：同一套 decoder-only 主干、同一个 next-token 目标，图像只是另一种语言——这条线连同 raster 顺序之外的选择（next-scale、mask-predict）在 7.7 展开。

## 3. VAE：只给似然的下界

VAE 的目标在 6.3 已完整推导，此处只登记结论。潜变量模型 $p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$ 的高维积分不可解，引入摊销后验 $q_\phi(z \mid x)$ 后得到证据下界

$$
\log p_\theta(x) \ge \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[ \log p_\theta(x \mid z) \right]}_{\text{recon}} - \mathrm{KL}\left( q_\phi(z \mid x)\, \|\, p(z) \right) = \mathrm{ELBO}(\theta, \phi; x) \qquad (2)
$$

**符号**：$z$：latent；$\phi$：encoder 参数；recon 项：后验采样 $z$ 后 decoder 给 $x$ 的对数似然；第二项把后验往先验 $p(z) = \mathcal{N}(0, I)$ 上拉。gap 精确等于 $\mathrm{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$（6.3 式 (4)）：encoder 容量决定下界松紧。推导（Jensen 与 KL 非负性两条路径、reparameterization、Gaussian KL 闭式）见 6.3 §2，本章不重复。

四坐标读法：

- **优化什么**：似然的下界。最大化 ELBO 只在 $q_\phi$ 等于真后验时才等价于最大化似然本身。
- **怎么采样**：一次前向——$z \sim \mathcal{N}(0, I)$，$\hat{x} = \mu_\theta(z)$，全并行，无串行结构。
- **似然**：不可解，只有下界；逐点精确似然无从谈起。
- **mode 行为**：目标仍是 MLE 方向的近似，覆盖行为同 AR 族；但高斯 decoder 的 recon 项是 MSE，对多 mode 的条件分布取条件均值——输出偏向各 mode 的平均，视觉上即模糊。decoder 过强时的 posterior collapse（后验塌成先验、latent 被忽略）机制见 6.3 §2.6。

VAE 在 2024 年以后的主用形态不是独立生成器，而是两个配套角色：连续版作为扩散的感知压缩器（扩散在 $z$ 空间工作，7.6 的 LDM 主线），离散版作为 AR 的 tokenizer（7.7）。式 (2) 的「重建 + 先验匹配」骨架同时是 7.2 推导扩散 ELBO 的模板——扩散可以看作 latent 序列固定为加噪轨迹、encoder 没有参数的层级 VAE，这个对应关系在 7.2 精确化。

## 4. GAN：目标函数里没有似然

GAN（[Goodfellow et al., 2014](../references.md)，arXiv:1406.2661，[A]）把密度估计整个绕开：生成器 $G$ 把噪声 $z$ 映成样本，判别器 $D$ 学真假二分类，两者对同一个价值函数拉锯

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}\left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)}\left[ \log \left( 1 - D(G(z)) \right) \right] \qquad (3)
$$

**符号**：$D(x) \in (0, 1)$：$x$ 被判为真实样本的概率；$p_z$：噪声先验（通常 $\mathcal{N}(0, I)$）；$G(z)$：生成样本。$D$ 的方向是分类正确率，$G$ 的方向是让 $D$ 在假样本上打高分。

四坐标：

- **优化什么**：minimax 目标。式 (3) 从头到尾没有出现 $p_\theta(x)$——密度从未被显式表示，更没有似然梯度。固定 $G$ 时最优判别器有闭式，代回后 $G$ 端等价于最小化一个 JS 散度（练习 1 推导），这就是 GAN 与「距离 $p_{data}$」之间全部的联系。
- **怎么采样**：一次前向，$z \to G(z)$，与 VAE 同为最快一档。
- **似然**：不存在，连下界都没有。
- **mode 行为**：**模式坍缩（mode dropping）**——式 (3) 只奖励「生成的样本骗过 $D$」，不奖励「覆盖 $p_{data}$ 的全部 mode」；$G$ 把全部质量堆到少数几个逼真 mode 上时，$D$ 拿不出任何梯度去惩罚那些缺席的 mode，因为没有一项损失在度量覆盖度。

按本书范围决策，GAN 不展开：训练技巧、架构演化、稳定性方案均不专列章节。对抗 loss 作为组件在用处就地讲——VQGAN 的 PatchGAN 感知项已在 6.3 §4 拆解，扩散蒸馏里的对抗分量（DMD2）在 7.4 就地给公式。

## 5. Flow：可逆性换精确似然

Flow 家族的回答是最直接的：不绕开密度，把它参数化成对一个已知密度的变量替换。取双射 $f_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d}$，记 $z = f_\theta(x)$，change of variables 公式给出精确似然

$$
\log p_X(x) = \log p_Z\left( f_\theta(x) \right) + \log \left| \det \frac{\partial f_\theta(x)}{\partial x} \right| \qquad (4)
$$

**符号**：$p_Z$：latent 空间的简单先验（通常 $\mathcal{N}(0, I)$）；$\frac{\partial f_\theta}{\partial x}$：$f_\theta$ 在 $x$ 处的 Jacobian 矩阵；$\det$：行列式，度量该映射在 $x$ 处的局部体积缩放。第二项是变量替换的全部修正：映射把体积撑大处，密度相应压低。

式 (4) 的两项都必须可算，这把架构设计卡进一条窄缝：$f_\theta$ 必须双射（维度不变、可逆）、且每层 Jacobian 的 $\log|\det|$ 必须便宜。稠密层的行列式是 $O(d^{3})$，图像维度下不可行；Glow（[Kingma & Dhariwal, 2018](../references.md)，arXiv:1807.03039，[A]）的代表性解法是 invertible $1 \times 1$ convolution——把通道混合写成每个空间位置共享的一个 $c \times c$ 矩阵，全图的 $\log|\det|$ 坍缩成 $hw \cdot \log|\det W|$，一次行列式计算服务所有空间位置（练习 2 算这笔账）。

双射还附带一个其他家族没有的副产品：编码方向 $z = f_\theta(x)$ 免费。任何真实样本都能精确反演回 latent 并读出自己的似然，插值、属性编辑在 $z$ 空间是精确坐标操作而不是近似投影——VAE 的 encoder 只给近似后验，GAN 连 encoder 都没有（要反演一个真实样本得额外做优化或训一个编码器）。

四坐标：

- **优化什么**：精确似然，直接对式 (4) 做 MLE，与 AR 同为「密度可算」一族。
- **怎么采样**：一次逆映射——$z \sim p_Z$，$x = f_\theta^{-1}(z)$，单方向一遍前向。
- **似然**：精确，这是 flow 的主卖点：异常检测、密度排序、精确 bits/dim 评估都直接可用。
- **mode 行为**：精确 MLE，覆盖行为同 AR；但双射约束是硬性的——同维、处处可逆的架构在表达任意扭曲的密度时比无约束网络吃力，样本质量受架构可逆性的天花板限制。

Flow 的框架在另一条世界线里复活：把离散的双射层换成连续时间向量场，change of variables 变成沿 ODE 的散度积分，就得到 continuous normalizing flow——这正是 7.5 flow matching 的入口，届时 trace 公式的成本账会解释为什么连续版一度也昂贵。

## 6. Diffusion 与 FM：回归目标换迭代采样

扩散把「一步生成」拆成一条时间轴。前向过程（加噪）是固定的、无参数的：按调度把数据逐步混入高斯噪声，其闭式（7.2 推导）为

$$
q(x_t \mid x_0) = \mathcal{N}\left( \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t)\, I \right), \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon \qquad (5)
$$

**符号**：$x_0 \sim p_{data}$：数据；$\varepsilon \sim \mathcal{N}(0, I)$；$\bar\alpha_t$：从 1 单调衰减到约 0 的调度量（$\bar\alpha_0 = 1$ 为数据端、$\bar\alpha_1 \approx 0$ 为纯噪声端，与第 1 节的约定一致）。采样式给出任意 $t$ 的一步到位加噪，不需要逐层递推。

训练目标不是似然而是一个回归：网络 $\varepsilon_\theta(x_t, t)$ 学习从加噪样本里预测被混入的噪声

$$
L_{simple} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\left[ \left\| \varepsilon - \varepsilon_\theta(x_t, t) \right\|^2 \right] \qquad (6)
$$

**符号**：$t$：均匀采样的时间；$x_t$ 按式 (5) 现场构造。式 (6) 看起来与似然无关，7.2 将推导它是扩散 ELBO 去掉一项 SNR 加权后的简化形——「回归噪声」与「最大化似然下界」是同一个目标的两种权重。等价地，$\varepsilon_\theta$ 与 score（7.3 定义）只差一个时间相关的标量因子：$s_\theta = -\varepsilon_\theta / \sqrt{1 - \bar\alpha_t}$。

flow matching 是同族的另一种参数化。取线性插值路径与恒定目标速度

$$
x_t = (1 - t)\, x_0 + t\, \varepsilon, \qquad u_t = \varepsilon - x_0 \qquad (7)
$$

$$
L_{FM} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\left[ \left\| v_\theta(x_t, t) - u_t \right\|^2 \right] \qquad (8)
$$

**符号**：$v_\theta$：回归速度场（velocity field）的网络；$u_t$：沿直线路径的恒定速度。式 (7)(8) 与式 (5)(6) 是同一 Gaussian 路径族在不同坐标下的写法，换算表在 7.5 给出（含与 Lipman 原文「$x_1$ = 数据」约定的映射框，本书一律用 $x_0$ = 数据）。

四坐标：

- **优化什么**：噪声/速度回归——结论级属于「加权 MLE 族」，推导在 7.2 / 7.5。
- **怎么采样**：反向过程（采样）沿 $t: 1 \to 0$ 迭代——从 $x_1 = \varepsilon$ 出发，逐步调用网络并更新 $x_t$，每步至少一次网络前向。DDPM 原始配置 $T = 1000$ 步；把步数压下来的全套技术（DDIM、高阶 solver、蒸馏）构成 7.4 整章；网络评估次数的 NFE 记号 7.2 已引入，本书自此统一使用。
- **似然**：两条口径——ELBO 下界（与 VAE 同族，7.2），或经 probability-flow ODE 的连续 change of variables 算精确似然但每点要付一条 ODE 轨迹的积分成本（结论级见 7.3）。日常训练与采样都不需要它。
- **mode 行为**：覆盖好。回归目标在每个噪声水平上都要求网络拟合全体数据的条件期望结构，没有 GAN 式「缺席的 mode 无惩罚」的漏洞；迭代细化本身也对多 mode 分布友好。

迭代还附带一个运行期自由度：同一份训练好的权重，采样时用步数换质量——预览走几步、出图走几十步，训练侧零改动。一次前向家族没有这个旋钮，要快只能在训练期换目标（7.4 的蒸馏族正是把迭代教师的步数压进少步学生）。

扩散用迭代采样换来的是另外两档收益：质量（当前开放 T2I 系统的主力路线，7.7 拆解）与可控性——条件信息经 cross-attention 等机制注入每一步去噪（7.6），引导强度还可以在采样时连续调节（7.4 的 CFG）。与 LLM 的组合不是参数级的（目标函数不同），而是接口级的：text encoder 产出 embedding 序列，扩散主干消费它（7.6 的三种条件化机制对比）。

## 7. 五家族对照总表

| 家族 | 优化什么 | 似然 | 采样步数 | mode 行为 | 条件化 / 可控性 | 与 LLM 可组合性 |
| --- | --- | --- | --- | --- | --- | --- |
| AR（式 (1)） | 精确 MLE | 精确、便宜 | $T$ 步串行 | 全覆盖（mean-seeking） | 前缀条件天然；图像需先离散化 | 原生：同主干同目标，图像 = 另一种语言（7.7） |
| VAE（式 (2)） | ELBO 下界 | 仅下界 | 1 步 | 覆盖，但 MSE 项致模糊 | latent 插值可用；控制粒度粗 | 配套角色：给 diffusion / AR 供 latent 与 tokenizer（6.3） |
| GAN（式 (3)） | minimax（非似然） | 不存在 | 1 步 | mode dropping | 条件 GAN 可行；本书不展开 | 弱：作为损失组件出现（6.3 VQGAN、7.4 DMD2） |
| Flow（式 (4)） | 精确 MLE | 精确 | 1 步（逆映射） | 覆盖，受可逆架构上限约束 | 密度任务直接可用 | 弱 |
| Diffusion / FM（式 (5)-(8)） | 噪声 / 速度回归 | ELBO；精确可算但贵（7.3） | 迭代，原始 1000 步、可压至个位数（7.4） | 覆盖好 | 每步注入条件 + 采样期引导强度（7.4/7.6） | 接口级：text encoder embedding → 扩散主干（7.6） |

读表的两个抓手：似然一列区分「密度可算族」（AR / Flow）与「密度绕开族」（GAN、工程形态的 diffusion）；采样步数一列区分「一步到位族」（VAE / GAN / Flow）与「迭代族」（diffusion、AR 的串行本质上也属迭代）。Diffusion 的独特位置在于：它付迭代账单，换来的是回归目标的训练稳定性与 mode 覆盖——GAN 的一次前向以训练稳定性与覆盖度为代价，这笔交易的两端在 7.7 的系统对照表里会再次用真实数字结账。

这张表的另一半价值在组合关系上：2022 年以后的主力图像系统几乎都不是单一家族。LDM（7.6）= 6.3 连续 VAE latent + 扩散的迭代生成，把扩散从像素空间搬进压缩空间；VQGAN（6.3）与 DMD2（7.4）把式 (3) 的对抗项当感知损失组件挂在其他目标骨架上；LlamaGen、Emu3（7.7）= 6.3 离散 tokenizer + 式 (1) 的 AR 主干；MaskGIT 式的 mask-predict（7.7）则是 AR 坐标里「顺序任意性」（1.1 §3 性质 2）的另一取点。读懂一个新系统的标准动作，就是把它拆到这张表的若干行上——本 Part 后六章就是在逐行提供拆分所需的数学。

## 8. 本 Part 章节地图

本 Part 共七章，依赖序即写作与阅读序：

```
                        ┌────────────────────────┐
                        │ 7.1 家族地图（本章）    │
                        └───────────┬────────────┘
                                    │
              ┌─────────────────────┼───────────────────────┐
              v                     v                       │
   ┌────────────────────┐  ┌────────────────────┐           │ 6.3 已备：
   │ 7.2 DDPM 完整推导   │  │ 7.5 Flow Matching  │           │ latent / tokenizer
   │ 前向闭式、后验配方、│  │ conditional FM 定理、│           │
   │ ELBO → L_simple    │  │ 线性路径、FM↔扩散   │           v
   └─────────┬──────────┘  │ 映射               │  ┌─────────────────────┐
             │             └─────────┬──────────┘  │ 7.6 LDM / DiT /     │
             v                       │             │ MMDiT：latent 空间、 │
   ┌────────────────────┐            │             │ 条件化、网络架构     │
   │ 7.3 Score 视角与    │            │             └──────────┬──────────┘
   │ SDE/ODE：DSM 等价、 │◄───────────┘                        │
   │ PF-ODE、精确似然    │                                     │
   └─────────┬──────────┘                                     │
             v                                                │
   ┌────────────────────┐                                     │
   │ 7.4 采样器与        │                                     │
   │ Guidance：DDIM、CFG、│                                     │
   │ solver、蒸馏族      │                                     │
   └─────────┬──────────┘                                     │
             └────────────────────────┬───────────────────────┘
                                      v
                    ┌──────────────────────────────────┐
                    │ 7.7 T2I 系统与 AR 生图（收口）     │
                    │ SD3 / SDXL / Imagen；LlamaGen /   │
                    │ VAR / Infinity；MaskGIT；Emu3     │
                    └──────────────────────────────────┘
```

各章定位一句话：

- **7.2** 是数学基座：式 (5)(6) 的完整推导在这里，后验配方与 ELBO 三项分解是后续一切的地基；
- **7.3** 换连续视角：score、SDE、probability-flow ODE，把 DDPM/DDIM/solver 统一成同一对象的不同离散化；
- **7.4** 对付采样账单：1000 步压到几十步再到个位数的全部机制；
- **7.5** 给同族的第二套参数化：式 (7)(8) 的严格版本与 FM↔扩散换算；
- **7.6** 回答「网络是什么、在哪个空间工作」：latent 空间的账与 DiT/MMDiT 架构；
- **7.7** 收口：真实 T2I 系统拆解 + AR/mask-predict 家族的实战对照，本章式 (1)-(8) 的家族坐标在系统层面再结一次账。

## 总结

| 家族 | 目标 | 似然 | 采样 | mode | 详拆去处 |
| --- | --- | --- | --- | --- | --- |
| AR | 式 (1)：精确 MLE | 精确 | $T$ 步串行 | 全覆盖 | 1.1；图像侧 7.7 |
| VAE | 式 (2)：ELBO | 下界 | 1 步 | 覆盖 + 模糊 | 6.3；配套 7.6/7.7 |
| GAN | 式 (3)：minimax | 无 | 1 步 | mode dropping | 不展开；组件见 6.3/7.4 |
| Flow | 式 (4)：change of variables | 精确 | 1 步逆映射 | 覆盖，架构受限 | 7.5 连续版入口 |
| Diffusion/FM | 式 (6)(8)：噪声/速度回归 | ELBO / 精确但贵 | 迭代 $t: 1 \to 0$ | 覆盖好 | 7.2-7.6 |

公式清单：式 (1) AR 负对数似然；式 (2) VAE 的 ELBO（recap 6.3）；式 (3) GAN minimax；式 (4) flow 的 change of variables；式 (5) 扩散前向闭式 $q(x_t \mid x_0)$；式 (6) $L_{simple}$；式 (7) 线性路径与目标速度 $u_t = \varepsilon - x_0$；式 (8) FM 回归目标。约定登记：$x_0$ = 数据、$t = 1$ = 噪声端、采样 $t: 1 \to 0$。

## 练习

1.（推导）固定 $G$，证明式 (3) 的最优判别器为 $D^{*}(x) = p_{data}(x) / \big(p_{data}(x) + p_g(x)\big)$（提示：$V$ 对 $D(x)$ 逐点取极值）；代回 $V(D^{*}, G)$，证明 $\min_G$ 端等价于最小化 $2 \cdot \mathrm{JSD}(p_{data} \| p_g) - \log 4$。用该结论解释 mode dropping 为何不被惩罚：JSD 有界（$\le \log 2$），$p_g$ 放弃一个 mode 至多付出有界代价，对比 MLE 中 $\log p_\theta \to -\infty$ 的无界惩罚。
2.（计算）Flow 的行列式账：$32 \times 32 \times 4$ 的 latent（f8），比较 (a) 稠密双射层 $d = 4096$，$\log|\det|$ 一次 $O(d^{3})$；(b) Glow 的 invertible $1 \times 1$ conv：单矩阵 $W \in \mathbb{R}^{4 \times 4}$，全图 $\log|\det| = 32 \cdot 32 \cdot \log|\det W|$。各算一次前向中行列式项的标量运算量级，并说明 (b) 把什么计算换成了什么。
3.（计算）采样总账：$256 \times 256$ 图像经 f16 tokenizer 得 256 个 token。AR 顺序采样 256 步（KV cache 下第 $t$ 步只算 1 个新 token），diffusion 以 NFE = 20 计、每步处理全部 256 个 latent 位置。比较两者的「网络处理 token 总数」量级；再指出这个账忽略了哪两个使比较不严格的因素（提示：每步的计算深度、序列并行性）。
4.（分析）三个场景各选一个家族并给机制理由：(a) 工业质检的异常检测，需要逐样本精确 $\log p(x)$；(b) 与现有 LLM 共享主干的图文统一生成；(c) 端侧实时预览，只允许 1-2 次网络前向。说明每个选择放弃了什么。

## 参考

- [Goodfellow et al., 2014](../references.md)：Generative Adversarial Networks，arXiv:1406.2661，[A]（key: gan）——式 (3) minimax 目标；练习 1 的 $D^{*}$ 与 JSD 结论
- [Kingma & Dhariwal, 2018](../references.md)：Glow: Generative Flow with Invertible 1x1 Convolutions，arXiv:1807.03039，[A]（key: glow）——式 (4) 的工程化：invertible $1 \times 1$ conv 与精确似然
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——式 (5)(6) 出处；$T = 1000$ 默认配置；推导见 7.2
- [Kingma & Welling, 2013](../references.md)：Auto-Encoding Variational Bayes，arXiv:1312.6114，[A]（key: vae）——式 (2) ELBO；完整推导见 6.3
