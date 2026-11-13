# 6.3 Autoencoder 与 Visual Tokenizer：图像的两种压缩表示（Autoencoders and Visual Tokenizers）

> 本章回答「图像在进生成模型之前先被压成什么」：VAE 的 ELBO 完整推导（Jensen 与 KL 非负性两条路径）、reparameterization 的梯度流通推导、Gaussian KL 闭式、posterior collapse 机制；VQ-VAE 的最近邻量化、straight-through 梯度拷贝推导、commitment loss 与 EMA 更新、codebook collapse 机制与缓解；VQGAN 的 LPIPS 感知项与 PatchGAN 对抗项（对抗 loss 就地讲）；RVQ 残差递推与 bitrate 账（P10 音频 codec 直接复用）；LFQ 的 sign 量化与熵正则；现代 AE 三档（SD-VAE f8/4ch 与 scaling factor 0.18215 的单位方差化、SD3/FLUX f8 16ch、DC-AE f32/f64 深压缩——SANA 基石）；离散 vs 连续选择表（P7/P8 伏笔）| 前置：1.1（KL 散度与 CE 分解式 (6)）、2.6（逐位置账的口径）、6.1（patchify 与 token 数账） | 锚点：VAE [A]、VQ-VAE [A]、VQGAN [A]、MAGVIT-v2 [A]、DC-AE [A]；CompVis/taming-transformers [A]、diffusers AutoencoderKL [A]、mit-han-lab/efficientvit [A]、lucidrains/vector-quantize-pytorch（教学参照）

## 1. 压缩的账：为什么不在像素空间生成

一张 $512 \times 512$ 的 RGB 图像是 $786432$ 个标量。把它直接塞进 1.1 的样本空间框架，就是一条长 26 万的序列（每像素 3 个 8-bit 值），任何自回归模型都无法在这个长度上工作；6.1 式 (3) 的硬切补丁 $N = HW/p^{2}$ 也只是折中—— $p$ 取 8 时仍有 4096 个 token，且每个 patch 内部的高频结构被一次线性投影（6.1 式 (2)）提前丢掉，找不回来。

autoencoder 的回答是把「下采样」从固定算子换成可学映射：encoder $E$ 把 $x \in \mathbb{R}^{H \times W \times C}$ 压成 latent $z \in \mathbb{R}^{h \times w \times c}$，空间压缩率 $f = H/h$（全书约定：f8 即 $h = H/8$，f32 即 $h = H/32$）；decoder $D$ 学近似逆映射 $\hat{x} = D(z)$，训练目标让重建误差足够小。同一张 $512^{2}$ 图：f8 留 $64 \times 64 = 4096$ 个位点，f32 留 $16 \times 16 = 256$ 个，f64 只剩 $8 \times 8 = 64$ 个——已经低于 ViT 224/16 的 196 这一量级。与硬切 patch 的差别在信息论意义上：patch 投影是一个固定的线性算子，丢什么由网格决定； $E$ 是被重建目标监督过的非线性压缩，丢什么、留什么由数据决定。

压缩出来的 latent 分两族接口，本章沿这条线展开：

- **连续 latent**（§2、§7）： $z$ 是连续张量，配显式的概率语义（高斯后验 + 高斯先验），下游生成器（diffusion，P7 主线）在连续空间里回归它；
- **离散 token**（§3–§6）： $z$ 被量化成 codebook 下标的网格，图像变成「词表 $K$、长度 $hw$ 的离散序列」，与 1.1 的 token 序列同构，下游是 AR/transformer 生成（P8 多模态生成）。

两族都活着、互为竞争方案；§7.4 的选择表给出判定依据。visual tokenizer 的命名也由此而来：它之于图像就是 2.2 的 BPE 之于文本——决定生成模型的「词表」，后续一切生成质量先被它卡一道上限。

## 2. VAE：latent variable 模型与 ELBO

### 2.1 模型设定与似然的不可解性

潜变量模型（latent variable model）把图像 $x$ 的生成拆成两步：先从先验采 latent $z \sim p(z)$，再由 decoder 从 $z$ 生成 $x \sim p_{\theta}(x \mid z)$。全章取标准设定 $p(z) = \mathcal{N}(0, I)$，连续图像配高斯 decoder $p_{\theta}(x \mid z) = \mathcal{N}(\mu_{\theta}(z), \sigma_{x}^{2} I)$。这个模型的边缘似然是

$$
p_{\theta}(x) = \int p_{\theta}(x \mid z)\thinspace p(z)\thinspace dz \qquad (1)
$$

**符号**： $x \in \mathbb{R}^{H \times W \times C}$：观测图像； $z \in \mathbb{R}^{d_z}$：latent（本节先压成向量写，空间结构在 §7 恢复）； $\theta$：decoder 参数。

式 (1) 不可解：高维积分没有闭式，逐点数值求积要在 $d_z$ 维空间里枚举。训练要 $\nabla_{\theta} \log p_{\theta}(x)$、生成后要采样，两条路都被这个积分堵死。VAE（[Kingma & Welling, 2013](../references.md)，arXiv:1312.6114，[A]）的解法：引入一个可学的近似后验 $q_{\phi}(z \mid x) = \mathcal{N}(\mu_{\phi}(x),\ \mathrm{diag}\thinspace\sigma_{\phi}^{2}(x))$（encoder，参数 $\phi$），把「对每个 $x$ 各求一次后验」摊销（amortize）成一张共享网络的一次前向——标题里的 Auto-Encoding 就是指这个摊销结构。

### 2.2 ELBO 推导 I：Jensen 不等式路径

从式 (1) 出发，被积函数里乘除 $q_{\phi}(z \mid x)$（要求 $q_{\phi}$ 在 $p_{\theta}(x \mid z) p(z) > 0$ 处严格正，高斯设定自动满足）：

$$
\log p_{\theta}(x) = \log \int q_{\phi}(z \mid x)\thinspace \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)}\thinspace dz = \log \mathbb{E}_{q_{\phi}}\left[ \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)} \right] \qquad (2)
$$

**符号**： $p_{\theta}(x, z) = p_{\theta}(x \mid z) p(z)$：联合密度；期望对 $q_{\phi}(z \mid x)$ 取。

$\log$ 是凹函数，Jensen 不等式给出 $\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log \cdot]$，于是

$$
\log p_{\theta}(x) \ge \mathbb{E}_{q_{\phi}(z \mid x)}\left[ \log \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)} \right] =: \mathrm{ELBO}(\theta, \phi; x) \qquad (3)
$$

**符号**： $\mathrm{ELBO}$：证据下界（evidence lower bound），「证据」是 $\log p_{\theta}(x)$ 的别名。等号条件（Jensen 取等）是被取期望的随机变量几乎处处为常数，即 $p_{\theta}(x, z) / q_{\phi}(z \mid x)$ 不依赖 $z$，等价于 $q_{\phi}(z \mid x) = p_{\theta}(z \mid x)$（真后验）。

### 2.3 ELBO 推导 II：KL 非负性路径（gap 的精确含义）

第二条路径得到一个恒等式，不只是不等式。 $\log p_{\theta}(x)$ 不含 $z$，对任意 $q_{\phi}$ 有 $`\log p_{\theta}(x) = \mathbb{E}_{q_{\phi}}[\log p_{\theta}(x)]`$；再用联合分解 $p_{\theta}(x) = p_{\theta}(x, z) / p_{\theta}(z \mid x)$ 代入：

$$
\begin{aligned}
\log p_{\theta}(x) &= \mathbb{E}_{q_{\phi}}\left[ \log p_{\theta}(x, z) - \log p_{\theta}(z \mid x) \right] \\
&= \mathbb{E}_{q_{\phi}}\left[ \log \frac{p_{\theta}(x, z)}{q_{\phi}(z \mid x)} \right] + \mathbb{E}_{q_{\phi}}\left[ \log \frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)} \right] \\
&= \mathrm{ELBO}(\theta, \phi; x) + \mathrm{KL}\left( q_{\phi}(z \mid x)\thinspace \Vert\thinspace p_{\theta}(z \mid x) \right)
\end{aligned} \qquad (4)
$$

第一步在 $\log$ 内同时加减 $q_{\phi}$（等价地，把第二项的分子分母都配出来），第二步按期望线性拆开，第三步识别出两个定义：前一项正是式 (3) 的 ELBO，后一项按定义是近似后验与真后验之间的 KL 散度。KL 非负（1.1 式 (6) 注记：非负，当且仅当两分布相等时为 0），所以 $\log p_{\theta}(x) \ge \mathrm{ELBO}$，且**gap 精确等于后验近似误差** $\mathrm{KL}(q_{\phi} \Vert p_{\theta}(z \mid x))$。两条路径在等号条件处会师：Jensen 取等 $\Leftrightarrow$ 这个 KL 为 0。

式 (3) 的坐标读法：最大化 ELBO 同时做两件事—— $\theta$ 方向抬高似然下界、 $\phi$ 方向把 gap（式 (4) 的 KL）压向 0。 $q_{\phi}$ 越逼近真后验，ELBO 对 $\theta$ 越是紧的下界；这就是为什么 encoder 的容量直接决定下界的松紧，摊销网络不是可省的附件。

### 2.4 重建-先验分解与 Gaussian KL 闭式

把 ELBO 的联合密度按 $p_{\theta}(x, z) = p_{\theta}(x \mid z)\thinspace p(z)$ 再拆一层，配出第二个 KL：

$$
\begin{aligned}
\mathrm{ELBO} &= \mathbb{E}_{q_{\phi}}\left[ \log p_{\theta}(x \mid z) \right] + \mathbb{E}_{q_{\phi}}\left[ \log p(z) \right] - \mathbb{E}_{q_{\phi}}\left[ \log q_{\phi}(z \mid x) \right] \\
&= \underbrace{\mathbb{E}_{q_{\phi}(z \mid x)}\left[ \log p_{\theta}(x \mid z) \right]}_{\text{recon}} - \underbrace{\mathrm{KL}\left( q_{\phi}(z \mid x)\thinspace \Vert\thinspace p(z) \right)}_{\text{prior matching}}
\end{aligned} \qquad (5)
$$

**符号**：recon 项：后验采样 $z$ 后 decoder 给 $x$ 的对数似然，衡量重建质量；prior matching 项：后验与先验 $p(z)$ 的 KL，把后验往先验上拉。

两项都落在可算闭式上。recon 项用高斯 decoder： $\log p_{\theta}(x \mid z) = -\Vert x - \mu_{\theta}(z)\Vert^{2} / (2\sigma_{x}^{2}) + \mathrm{const}$——固定 $\sigma_{x}$ 时，最大化它就是最小化 MSE，重建损失由此而来。prior matching 项是两个对角高斯之间的 KL，有闭式。逐维推导：设 $q = \mathcal{N}(\mu, \sigma^{2})$、 $p = \mathcal{N}(0, 1)$（单维），

$$
\begin{aligned}
\mathrm{KL}(q \Vert p) &= \mathbb{E}_{q}\left[ \log q(z) - \log p(z) \right] \\
&= \mathbb{E}_{q}\left[ -\log \sigma - \frac{(z - \mu)^{2}}{2\sigma^{2}} + \frac{z^{2}}{2} \right] \\
&= -\log \sigma - \frac{1}{2} + \frac{\mu^{2} + \sigma^{2}}{2} = \frac{1}{2}\left( \mu^{2} + \sigma^{2} - \log \sigma^{2} - 1 \right)
\end{aligned} \qquad (6)
$$

第二步写出两个 log 密度并把常数 $-\frac{1}{2}\log 2\pi$ 消掉；第三步用 $`\mathbb{E}_{q}[(z-\mu)^{2}] = \sigma^{2}`$ 与 $`\mathbb{E}_{q}[z^{2}] = \mu^{2} + \sigma^{2}`$；第四步整理。多维对角情形逐维相加：

$$
\mathrm{KL}\left( \mathcal{N}(\mu, \mathrm{diag}\thinspace\sigma^{2})\thinspace \Vert\thinspace \mathcal{N}(0, I) \right) = \frac{1}{2} \sum_{i=1}^{d_z} \left( \mu_{i}^{2} + \sigma_{i}^{2} - \log \sigma_{i}^{2} - 1 \right) \qquad (7)
$$

**符号**： $\mu, \sigma \in \mathbb{R}^{d_z}$：encoder 输出的均值与标准差向量。式 (7) 是 $\phi$ 的显式可微函数，不涉采样——prior matching 项的梯度一分钱采样噪声都不用付。

写成要最小化的 loss（ $\beta$ 加权版）：

$$
L_{\mathrm{VAE}} = -\mathbb{E}_{q_{\phi}}\left[ \log p_{\theta}(x \mid z) \right] + \beta \cdot \mathrm{KL}\left( q_{\phi}(z \mid x)\thinspace \Vert\thinspace p(z) \right) \qquad (8)
$$

**符号**： $\beta > 0$：两项的相对权重； $\beta = 1$ 即负 ELBO。 $\beta$ 不是理论对象而是设计旋钮：调大则后验更贴先验（latent 更「规整」、重建更差），调小则重建优先。式 (8) 就是全部现代连续 AE（§7）的目标函数骨架。

### 2.5 reparameterization：梯度怎么穿过采样

式 (8) 的 recon 项要对 $\phi$ 求导，但期望号下的分布 $q_{\phi}(z \mid x)$ 本身依赖 $\phi$——采样是一个无梯度的操作，链式法则在「采一个 $z$」这一步断掉。

通用解是 log-derivative trick： $`\nabla_{\phi} \mathbb{E}_{q_{\phi}}[f(z)] = \mathbb{E}_{q_{\phi}}[ f(z)\thinspace \nabla_{\phi} \log q_{\phi}(z \mid x) ]`$。它对任何分布成立，但梯度估计把逐样本的 $f(z)$ 值直接乘上去，方差大—— $f$ 的大幅波动原样进入梯度。

高斯族有更好的 pathwise 解（reparameterization，VAE 论文的核心机制）：把采样改写成对无参噪声的确定性变换，

$$
z = g_{\phi}(x, \varepsilon) = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) \qquad (9)
$$

**符号**： $\odot$：逐元素乘； $\varepsilon$：与 $\phi$ 无关的标准高斯噪声。分布正确性由仿射变换保证： $\varepsilon \sim \mathcal{N}(0, I) \Rightarrow \mu + \sigma \odot \varepsilon \sim \mathcal{N}(\mu, \mathrm{diag}\thinspace\sigma^{2})$，与直接从 $q_{\phi}$ 采样的 $z$ 同分布。

换变量后期望的外层分布不再含 $\phi$，梯度直进期望号：

$$
\nabla_{\phi}\thinspace \mathbb{E}_{q_{\phi}}\left[ f(z) \right] = \mathbb{E}_{\varepsilon}\left[ \nabla_{z} f(z)\thinspace \frac{\partial g_{\phi}}{\partial \phi} \right] = \mathbb{E}_{\varepsilon}\left[ \nabla_{z} f(z) \odot \frac{\partial (\mu_{\phi} + \sigma_{\phi} \odot \varepsilon)}{\partial \phi} \right] \qquad (10)
$$

逐分量看雅可比： $\partial z_i / \partial \mu_i = 1$， $\partial z_i / \partial \sigma_i = \varepsilon_i$。与 log-derivative 的实质差别：梯度经由 $f$ 对 $z$ 的局部斜率传导——用上了 $f$ 的光滑性（decoder 关于 $z$ 可微），采样值本身不进入梯度表达式，方差因此远低。工程上这就是一行代码：`z = mu + sigma * torch.randn_like(sigma)`，反向传播自动走式 (10)。

### 2.6 posterior collapse：KL 项的暗面

式 (8) 的 KL 项单独取最优的解是 $q_{\phi}(z \mid x) = p(z)$ 对所有 $x$ 成立——后验塌成先验， $z$ 不带 $x$ 的任何信息。这个解在 recon 项不抵抗时就是全局最优，称为 posterior collapse（后验坍缩）。

机制拆解：collapse 发生的条件是 decoder 强到 recon 项对 $z$ 不敏感。极端情形是自回归 decoder—— $p_{\theta}(x_t \mid x_{<t}, z)$ 里 $x_{<t}$ 已经足以预测 $x_t$， $z$ 被旁路， $\mathbb{E}[\log p_{\theta}(x \mid z)]$ 在 $z$ 被忽略时照样取到高值，于是 KL 项把 $q_{\phi}$ 一路拉向先验，KL → 0，latent 通道报废。训练中的可观测信号：KL 项单调降到 0 附近；从 $\mathcal{N}(0, I)$ 采样 $z$ 生成的结果与 encoder 编码真实图像得到的 $z$ 不可区分——后验已经不带信息。

缓解手段全部从「打破 recon 与 KL 的失衡」出发：(a) $\beta$ 退火：训练前期 $\beta = 0$（纯 AE），让 latent 先变得有用，再线性升到目标值；(b) free bits：给每维 KL 一个免罚额度 $\lambda$，把式 (7) 换成 $`\sum_{i} \max(\lambda,\ \mathrm{KL}_i)`$——低于额度的维度不产生梯度，强制每维至少保留 $\lambda$ nats 的信息通道；(c) 限制 decoder 的条件容量（砍掉自回归通路）。对图像 conv decoder（非自回归），collapse 压力天然较小；§7 会看到的实践口径是把 $\beta$ 压到远小于 1，让式 (8) 行为上接近确定性 AE，概率语义只剩一层薄正则。

VAE 数据流（教学口径）：

```
x [B, H, W, C]
 │  E_φ：conv 下采样 → 两个线性头
 ▼
μ_φ(x), σ_φ(x)  [B, h, w, c]
 │  式 (9)：z = μ + σ ⊙ ε，ε ~ N(0, I)   ← 采样在梯度图内
 ▼
z  [B, h, w, c]
 │  D_θ：conv 上采样
 ▼
x̂ [B, H, W, C]
loss = −log p_θ(x|z)（≈ MSE） + β·KL（式 (7)，闭式无采样）
```

## 3. VQ-VAE：离散 latent 与 straight-through

### 3.1 最近邻量化：从连续 latent 到 codebook 下标

VQ-VAE（[van den Oord et al., 2017](../references.md)，arXiv:1711.00937，[A]）把 §2 的连续 $z$ 换成离散符号。三件套：encoder $E$ 输出连续 latent $z_e = E(x) \in \mathbb{R}^{h \times w \times d_e}$；codebook（码本） $E_{cb} = \lbrace e_k \rbrace_{k=1}^{K}$， $e_k \in \mathbb{R}^{d_e}$，共 $K$ 个 entry；量化器在每个空间位点独立做最近邻查表：

$$
k^{\ast }(i, j) = \operatorname{argmin}_{k}\hspace{5mu} \left\Vert z_e(i, j) - e_k \right\Vert_{2}, \qquad z_q(i, j) = e_{k^{\ast }(i, j)} \qquad (11)
$$

**符号**： $z_e(i, j) \in \mathbb{R}^{d_e}$：位点 $(i, j)$ 的 encoder 输出； $z_q(i, j)$：量化后的 latent，即被选中的 codebook entry。

decoder 只吃量化结果： $\hat{x} = D(z_q)$。量化之后，图像的表示是 $h \times w$ 网格、每格一个 $\lbrace1, \ldots, K\rbrace$ 里的下标——这正是 1.1 的「词表 $K$、长度 $hw$」的样本空间结构，2.x 的全套文本模型机制可以原样搬上来。VQ-VAE 论文就是这样接龙的：先训 VQ-VAE，再在离散网格上训一个 PixelCNN 先验做采样——「离散 latent + AR 先验」这条路线（P8 视觉生成的原型）从这里开始。

### 3.2 straight-through estimator：梯度拷贝推导

式 (11) 的 $\operatorname{argmin}$ 不可微，且断得很干净：在 $z_e$ 的几乎所有取值处，微小扰动不改变最近邻编号， $\partial z_q / \partial z_e = 0$；在切换面上偏导无定义。反向传播到 $z_q$ 为止——decoder 与重建项有梯度，encoder 与 codebook 之外的参数一分梯度都收不到。

straight-through estimator（STE，直通估计器）的构造只有一步：定义一个前向值等于 $z_q$、反向雅可比等于恒等的替代节点，

$$
z_q^{\mathrm{ST}} = z_e + \mathrm{sg}\left[ z_q - z_e \right] \qquad (12)
$$

**符号**： $\mathrm{sg}[\cdot]$：stop-gradient，前向取括号内的值、反向梯度恒为 0（实现上是 `.detach()`）。前向数值： $z_q^{\mathrm{ST}} = z_e + (z_q - z_e) = z_q$，与硬量化一致；反向： $\partial z_q^{\mathrm{ST}} / \partial z_e = I$，于是

$$
\frac{\partial L}{\partial z_e} := \frac{\partial L}{\partial z_q^{\mathrm{ST}}} \cdot I = \frac{\partial L}{\partial z_q} \qquad (13)
$$

decoder 输入端的梯度被原样拷贝到 encoder 输出端，式 (11) 的查表在梯度图里被当成恒等映射。数据流与梯度流对照：

```
前向：  z_e ──argmin 查表──▶ z_q ──▶ D ──▶ x̂ ──▶ L
反向：  ∂L/∂z_e ◀── 原样拷贝（式 (13)）── ∂L/∂z_q ◀── ∂L/∂x̂
```

这个拷贝是有误差的近似：真实的 $\partial z_q / \partial z_e$ 几乎处处为 0，STE 把它换成 $I$，误差量级正比于量化残差 $\Vert z_q - z_e \Vert$—— $z_e$ 离所选 code 越近，恒等近似越准。这就是为什么 STE 必须与下一条 commitment loss 配对使用：commitment 项把 $z_e$ 拉向 $`e_{k^{*}}`$，直接压住式 (13) 的近似误差。两者不是两个独立 trick，是同一个近似及其误差控制器。

### 3.3 三项 loss 与 EMA 更新

VQ-VAE 的训练目标（论文 eq. 3 口径）：

$$
L = \log p(x \mid z_q) + \left\Vert \mathrm{sg}[z_e] - e \right\Vert_{2}^{2} + \beta \left\Vert z_e - \mathrm{sg}[e] \right\Vert_{2}^{2} \qquad (14)
$$

**符号**：第一项：重建项（最小化时即 MSE/对数似然，位点逐格求和）；$`e = e_{k^{*}}`$：被选中的 codebook entry；第二项 codebook loss 把被选 entry 拉向 $z_e$（ $\mathrm{sg}$ 挡住对 $z_e$ 的梯度——encoder 的梯度只走式 (13) 一条道）；第三项 commitment loss 把 $z_e$ 拉向 entry， $\beta$ 控制承诺强度，论文全实验取 $\beta = 0.25$。注意式 (14) 以最大化似然的符号习惯书写（实现时取负）。

与式 (8) 的对照值得停一步：离散化之后不再有解析后验，VAE 框架里的 KL 项去了哪里？VQ-VAE 论文的接法：取确定性后验 $`q(k^{*} \mid x) = 1`$（one-hot 在 argmin 上）、先验为 $K$ 类均匀分布，则 ELBO 的 KL 项退化成常数 $\log K$，不进任何梯度——式 (14) 的三项正是这个退化 ELBO 剩下的可优化部分。正则的角色由 codebook loss 与 commitment loss 接管：一个管 code 跟着数据走，一个管数据贴着 code 走。

codebook loss 其实不必走梯度。对 entry $e_i$ 而言，式 (14) 第二项关于 $e_i$ 的极小有闭式：设 batch 内被分到 $e_i$ 的 encoder 输出集合为 $S_i$，对 $\sum_{z \in S_i} \Vert z - e_i \Vert^{2}$ 求导置零得 $e_i = |S_i|^{-1} \sum_{z \in S_i} z$——被分配点的均值，即 K-Means 更新（练习 2 推这一步）。全量数据的闭式解不能直接用于小 batch 训练，论文在 Appendix A.1 给出它的在线版——EMA 更新：

$$
N_i \leftarrow \gamma N_i + (1 - \gamma)\thinspace n_i, \qquad m_i \leftarrow \gamma m_i + (1 - \gamma) \sum_{z \in S_i} z, \qquad e_i = \frac{m_i}{N_i} \qquad (15)
$$

**符号**： $n_i = |S_i|$：本 batch 分到 code $i$ 的位点数； $N_i, m_i$：滑动平均的计数与向量和； $\gamma$：衰减系数，论文报告 $\gamma = 0.99$ 在实践中效果稳定。机制读法：EMA 把 codebook 从梯度循环里摘出来——code 的更新不与 encoder 共享学习率、不被重建梯度的噪声污染。定位要钉准：EMA 在论文里是式 (14) 第二项的替代方案（Appendix A.1），正文明说主实验未用它（"not used for the experiments in this work"）——主实验走的是 codebook loss 的梯度下降，两版不要混为一谈。顺带补一条 commitment 项的原文动机：embedding 空间没有天然的尺度约束，若 code 学得比 encoder 慢， $\Vert z_e \Vert$ 会任意膨胀，commitment 项就是压住这个膨胀的配重。

### 3.4 codebook collapse：死锁机制与缓解

codebook 的失效模式是一个正反馈死锁。设初始化后某些 entry 离数据流形偏远：argmin 永不选它们 $\Rightarrow$ EMA 的 $n_i = 0$、梯度支路的 $\mathrm{sg}$ 也不给它们流量 $\Rightarrow$ 与数据的距离不缩小 $\Rightarrow$ 下一轮更不会被选中。有效词表缩到远小于 $K$，极端时只剩几个 code——这就是 codebook collapse（码本坍缩）。诊断指标：各位点 code 使用分布 $u_k$（ $K$ 维频率向量）的 perplexity $\exp(H(u))$，健康值应与 $K$ 同量级，collapse 时掉到个位数。

缓解按作用层分三类：(a) EMA 本身——code 更新与梯度噪声解耦，死锁链条里的「更新依赖梯度」一环先被拆掉一半（式 (15) 的 $n_i = 0$ 时只是原地不动，不会被梯度推到更偏的位置）；(b) 死码重启：使用率长期低于阈值的 entry 直接重置为当前 batch 的某个 $z_e$，强制回到数据流形上（[lucidrains/vector-quantize-pytorch](../references.md) 教学实现的标准配置，教学参照）；(c) 目标函数层面强制均匀使用——LFQ 的熵正则是这条路线的完整形态，见 §6。

## 4. VQGAN：感知损失与 PatchGAN 对抗项

式 (14) 的重建项是逐像素距离，而逐像素监督有一个结构性盲区：当真实解是多峰的（一片草地的像素排列有无穷多种同样合理的实现），MSE 学到的是众解的均值——模糊。VQGAN（[Esser et al., 2020](../references.md)，arXiv:2012.09841，[A]）给 VQ-VAE 打两项补丁，把「像素保真」升级为「感知保真」。

**补丁一：LPIPS 感知损失**。用预训练分类网络（VGG）的中间层特征代替像素算距离：设 $\phi_l(\cdot)$ 为第 $l$ 层特征图（逐空间位置、逐通道单位化）， $w_l$ 为逐层权重，

$$
L_{\mathrm{per}}(x, \hat{x}) = \sum_{l} w_l \cdot \frac{1}{H_l W_l} \sum_{i, j} \left\Vert \phi_l(x)_{i,j} - \phi_l(\hat{x})_{i,j} \right\Vert_{2}^{2} \qquad (16)
$$

**符号**： $H_l, W_l$：第 $l$ 层特征图的空间尺寸。机制：分类网络的特征对「纹理相同、位置差几个像素」天然不敏感（这是它训练目标的副产物），在特征空间算距离就把监督从像素对齐换成语义对齐——模糊解在像素空间是「平均草地」，在特征空间仍然是错误答案。

**补丁二：PatchGAN 对抗项**（全书范围决策：GAN 不单独展开，对抗 loss 在用处就地讲）。判别器 $D$ 不把整图压成一个真/假标量，而是输出 $h \times w$ 的 patch 级 logits——每个输出位点只感受输入的一个局部块，对全图取均值。minimax 目标：

$$
\min_{E, D_{ec}, E_{cb}}\hspace{5mu} \max_{D}\hspace{5mu} L_{\mathrm{rec}} + \lambda\hspace{5mu} \mathbb{E}_{x}\left[ \log D(x) \right] + \mathbb{E}_{\hat{x}}\left[ \log\left( 1 - D(\hat{x}) \right) \right] \qquad (17)
$$

**符号**： $E, D_{ec}, E_{cb}$：encoder、decoder、codebook； $L_{\mathrm{rec}}$：式 (14) 加式 (16) 的重建侧全部项； $\lambda$：对抗项权重，见下。判别器侧把真图的 $D(x)$ 推高、把重建图 $D(\hat{x})$ 压低；生成侧（ $E, D_{ec}, E_{cb}$）在式 (17) 里被 $D$ 的梯度反向塑造——重建图被判得越假，生成侧收到的「往真里修」的梯度越大。官方实现（[CompVis/taming-transformers](../references.md)，[A]）用 hinge 变体（loss 定义在 `taming/modules/losses/vqperceptual.py`，`taming/models/vqgan.py` 是调用方）：

$$
L_{D} = \mathbb{E}_{x}\left[ \max\left( 0,\ 1 - D(x) \right) \right] + \mathbb{E}_{\hat{x}}\left[ \max\left( 0,\ 1 + D(\hat{x}) \right) \right], \qquad L_{G} = -\mathbb{E}_{\hat{x}}\left[ D(\hat{x}) \right] \qquad (18)
$$

hinge 与对数版的机制同构：真图打分推过 $+1$、假图压过 $-1$ 之后梯度归零，判别器不在已分对的样本上继续用力，对抗训练更稳。

对抗项引入一个新问题：它与重建项的量级互不可知，固定 $\lambda$ 不是压垮重建就是被重建淹没。VQGAN 的自适应权重按两边对 decoder 最后一层参数的梯度范数现场配比：

$$
\lambda = \frac{\left\Vert \nabla_{G_L} L_{\mathrm{rec}} \right\Vert}{\left\Vert \nabla_{G_L} L_{\mathrm{GAN}} \right\Vert + \delta}, \qquad \delta = 10^{-6} \qquad (19)
$$

**符号**： $G_L$：decoder 最后一层的参数； $\nabla_{G_L}$ 取梯度后按范数计。机制一目了然：让两项在「对输出端的拉力」上始终同阶——训练早期重建梯度大、 $\lambda$ 自动缩小对抗项；重建收敛后对抗项接管细节。量化环节沿用 §3 的 STE（式 (12)(13)）与 EMA（式 (15)），原样嵌进式 (17) 的最小化侧。

落地坐标：VQGAN 的常用档是 f16、 $K = 1024$（官方 checkpoint 命名 `vqgan_imagenet_f16_1024` 即此口径）—— $256^{2}$ 图被压成 $16 \times 16 = 256$ 个 code 的网格，「图像 = 256 个离散 token」由此成为可用实现，后来一大票 T2I 系统的 tokenizer 由此起步。

## 5. RVQ：残差量化递推与 bitrate 账

单级 VQ 的量化误差由词表大小钉死：误差半径量级是 code 间距，要把误差减半， $K$ 得按维度的幂次涨—— $d_e$ 维空间里覆盖同样密度，词表要指数级变大，而 §3.4 的 collapse 风险随 $K$ 一起涨。RVQ（residual vector quantization，残差向量量化）换一条加法路线：用 $n_q$ 个小 codebook 逐级量化残差，

$$
r_1 = z_e, \qquad e_1 = Q_1(r_1); \qquad r_{i+1} = r_i - e_i, \quad e_{i} = Q_{i}(r_{i}), \quad i = 1, \ldots, n_q; \qquad \hat{z} = \sum_{i=1}^{n_q} e_i \qquad (20)
$$

**符号**： $Q_i$：第 $i$ 级的最近邻量化器（codebook $C_i$，各 $K$ 个 entry、维 $d_e$）； $r_i$：第 $i$ 级输入残差； $\hat{z}$：量化结果，各位点是 $n_q$ 个 entry 的和。STE 逐级原样适用：每级 $Q_i$ 处做一次式 (12) 的拷贝。

级联的数据流：

```
z_e ─▶ Q1 ─▶ e1 ─┐
  └─ r2 = z_e − e1 ─▶ Q2 ─▶ e2 ─┤
       └─ r3 = r2 − e2 ─▶ Q3 ─▶ e3 ─┤        ẑ = e1 + e2 + e3
            ...（共 n_q 级）──────┘
误差 = ‖r_{n_q+1}‖ = ‖z_e − ẑ‖
```

总误差就是最后一段残差 $\Vert r_{n_q + 1} \Vert$；第 $i$ 级 codebook 学习的是「前 $i-1$ 级量化完之后剩下的残差」的条件分布，级数越深、残差分布越集中、同样 $K$ 的量化半径越小——实践中误差逐级递减（这是训练使然的经验性质，无严格单调保证）。

两笔账，都按 2.6 的逐位置口径算（先定每位点的量，再乘位点率）：

**容量账**。每位点的量化结果是 $n_q$ 元组 $(k_1, \ldots, k_{n_q})$，组合空间大小 $K^{n_q}$——有效词表随级数指数涨，参数只线性涨（ $n_q$ 张 $K \times d_e$ 的表）。 $K = 1024$、 $n_q = 8$：有效词表 $2^{80}$，每位点等效 80 bit，而单级 VQ 要到同样容量得塞 $2^{80}$ 个 entry。加法结构换指数容量，这是 RVQ 的全部卖点。

**bitrate 账**。每位点占 $n_q \log_2 K$ bit，乘上 latent 位点的时间密度 $f_{\mathrm{rate}}$（图像：每图位点数按图均摊；音频：每秒 latent 帧数，即 fps）：

$$
b = n_q \cdot \log_{2} K \cdot f_{\mathrm{rate}} \qquad (21)
$$

**符号**： $b$：码率（bit/s）； $f_{\mathrm{rate}}$：latent 帧率（fps）。实例： $n_q = 8$、 $K = 1024$、75 fps $\Rightarrow$ $8 \times 10 \times 75 = 6000$ bps $\approx$ 6 kbps——这是神经音频 codec 的标准档位。**式 (21) 连同它的实例账在 P10 音频 codec 章直接复用**：SoundStream/Encodec 系 codec 的全部码率标注都是这个公式的取值表，届时只需把 $f_{\mathrm{rate}}$ 的物理含义换成音频帧率，本章的推导原样生效。分级结构还白送一个性质：只传前 $m < n_q$ 级即得低码率版本，码率伸缩不需要重训。

## 6. LFQ（MAGVIT-v2）：lookup-free 量化与熵正则

把 §3.4 的 collapse 问题再往前推一层：既然死锁的根源是「entry 要先离数据近、才会被选中」，能不能干脆取消查表？MAGVIT-v2 的 LFQ（lookup-free quantization，免查表量化，[Yu et al., 2023](../references.md)，arXiv:2310.05737，[A]）把量化压到逐维符号：

$$
q(z)_i = \operatorname{sign}(z_i) \in \lbrace -1, +1 \rbrace, \qquad i = 1, \ldots, d \qquad (22)
$$

**符号**： $z \in \mathbb{R}^{d}$：encoder 输出（单 latent 位点展平）； $q(z) \in \lbrace -1, +1 \rbrace^{d}$：量化结果。codebook 不再是一张可学表，而是超立方体的顶点集 $\lbrace -1, +1 \rbrace^{d}$——隐式 codebook，大小 $2^{d}$，位点的 token id 就是 sign 模式的二进制编号。没有最近邻检索（lookup-free 的名字由此来），量化是逐维独立的一次符号函数；STE 照旧： $q^{\mathrm{ST}} = z + \mathrm{sg}[\operatorname{sign}(z) - z]$，梯度拷贝同式 (12)(13)。

代价： $2^{d}$ 个顶点里能被 argmin「自然选中」的概念不复存在——没有距离几何，使用率必须由目标函数显式管理。LFQ 的熵正则由两项组成。对第 $i$ 维，样本 $z$ 诱导一个 Bernoulli： $p_i(z) = (z_i + 1)/2$（设 $z_i$ 经有界化落在 $[-1, 1]$， $p_i$ 是「第 $i$ 维取 $+1$」的软概率——该参数化是实现层的操作化写法：原文的 $q(z_i)$ 保持抽象分布记号，官方 magvit 实现按顶点距离的负值过 softmax 得概率再算熵，数值细节以实现为准），其熵 $H(p) = -p \log p - (1 - p) \log(1 - p)$。记 batch 大小 $B$、batch 聚合概率 $`\bar{p}_i = \frac{1}{B} \sum_{b=1}^{B} p_i(z^{(b)})`$，则

$$
L_{\mathrm{ent}} = \frac{1}{B} \sum_{b=1}^{B} \sum_{i=1}^{d} H\left( p_i(z^{(b)}) \right) \hspace{5mu} - \hspace{5mu} \sum_{i=1}^{d} H\left( \bar{p}_i \right) \qquad (23)
$$

**符号**：第一项：逐样本熵，最小化它把每个样本的每维推向 $0$ 或 $1$—— $z_i$ 贴向 $\pm 1$ 的顶点，不许「骑墙」（ $p_i \approx 1/2$ 时熵最大）；第二项：batch 聚合熵，式 (23) 里带负号即最大化它，把逐维的边缘使用推向均匀（ $`\bar{p}_i = 1/2`$），进而让 $2^{d}$ 个顶点被整批样本铺开。一压一抬，正是 §3.4 collapse 的目标函数级解法：逐样本要确定、逐维边缘要均匀。总目标是 VQGAN 家族三项（重建 + LPIPS + 对抗）再加 $L_{\mathrm{ent}}$。

定位：MAGVIT-v2 用 LFQ 把视频与图像量化进同一个 vocabulary（摘要口径，[A]），并在该论文的评测上让语言模型路线的生成质量超过同期扩散模型——标题主张即此；注意其主结果的采样范式是 masked LM（掩码迭代解码）而非严格逐 token 自回归，「tokenizer 质量是语言模型视觉路线的杠杆点」这一判断由此坐实，直接影响 P7 的「AR vs diffusion」讨论与 P8 的多模态 tokenizer 选择。

## 7. 现代 AE：SD-VAE、SD3/FLUX、DC-AE

连续族（§2 的 $\beta$ 极小口径）是当代扩散模型的标配前端。三档实例按压缩率递进，数值出处均为 W3 research pass 的 config/代码实测。

### 7.1 SD-VAE：f8/4ch 与 scaling factor 的单位方差化

Stable Diffusion 1.x 的 VAE： $f = 8$（encoder 三次 stride-2 下采样， $H/2 \to H/4 \to H/8$），latent 通道 $c = 4$（runwayml/stable-diffusion-v1-5 的 `vae/config.json` 实测 `latent_channels = 4`）；后验是对角高斯，走读锚点是 diffusers 的 `autoencoder_kl.py`（[AutoencoderKL](../references.md)，[A]）：`encode` 产出后验分布，`sample`/`mode` 取 $z$，随后乘一个常数 scale factor。

这个常数的数学值得单独写：训练好的 AE 的 latent 标准差实测 $`\hat{\sigma}_z \approx 5.49`$，而下游扩散模型的 noise schedule 按单位方差输入标定（P7 展开），于是把 latent 归一到 $\mathcal{O}(1)$：

$$
z' = s \cdot z, \qquad s = \frac{1}{\hat{\sigma}_z} = 0.18215 \qquad (24)
$$

**符号**： $s$：scaling factor。数值自检： $1 / 0.18215 \approx 5.49$，与 $`\hat{\sigma}_z`$ 互倒——式 (24) 就是单位方差化，不是模型结构的一部分，纯粹是分布标定。出处注意：0.18215 不在 diffusers 版 config 里，它的出处是 CompVis/stable-diffusion 的 `configs/stable-diffusion/v1-inference.yaml`（W3 实测）；decode 侧先除以 $s$ 再进 decoder。 $\beta$ 口径上，这条线的 KL 权重压得极小，式 (8) 行为上接近确定性 AE——§2.6 的概率语义只剩一层薄正则，这与「latent 规整」而非「后验推断」的目标一致。

### 7.2 SD3/FLUX：f8 16ch

同 $f = 8$、 $c$ 从 4 提到 16：每 latent 位点的容量乘 4，重建保真（尤其文字渲染与高频纹理这类 f8/4ch 的传统弱项）随之抬升；代价是 AE 自身变重，而下游扩散主干的序列长度不变（通道在 patchify 时被吃掉，6.1 式 (2) 的 $p^{2}C$ 项吸收）。FLUX 的归一化比式 (24) 多一项 shift： $z' = s\thinspace(z - m)$，实测 $s = 0.3611$、 $m = 0.1159$（black-forest-labs/flux 的 `src/flux/util.py` 代码实测；HF 的 FLUX 权重 gated，以代码为准）——多出来的 $m$ 把 latent 均值也一并归零，是式 (24) 的「减均值 + 除方差」完整版。SD3 同属 16ch 家族（论文细节归 P7/W4 范围，本章只登记 16ch 这一架构事实，证据挂 FLUX 代码实测）。

### 7.3 DC-AE：f32/f64 深压缩

$f$ 从 8 继续往上翻，瓶颈不在表示而在优化：decoder 要从 $H/32$ 的网格还原全部高频细节，式 (8) 的重建项在 f32 起步就训不动。DC-AE（deep compression autoencoder，[Chen et al., 2024](../references.md)，arXiv:2410.10733，[A]）用两个技术把压缩率推到 f32/f64：

**(a) residual autoencoding（残差自编码）**。捷径不是 ResNet 的恒等映射，而是无参数的 space-to-channel 映射（6.1 式 (1) 的无投影版： $f \times f$ 空间块摊进通道维），并且不是整图级一条，而是挂在**每个下采样/上采样 block 上**并联。encoder 的下采样 block（输入 $H \times W \times C$、输出 $\frac{H}{2} \times \frac{W}{2} \times 2C$）配的非参捷径为

$$
\underbrace{H \times W \times C \hspace{5mu}\rightarrow\hspace{5mu} \frac{H}{2} \times \frac{W}{2} \times 4C}_{\text{space-to-channel}} \quad \underbrace{\rightarrow\hspace{5mu} \frac{H}{2} \times \frac{W}{2} \times 2C}_{\text{channel averaging}} \qquad (25)
$$

即 space-to-channel 之后把通道拆成两组逐组平均，对齐主路的输出通道。decoder 的上采样 block 对称（输入 $\frac{H}{2} \times \frac{W}{2} \times 2C$）：

$$
\underbrace{\frac{H}{2} \times \frac{W}{2} \times 2C \hspace{5mu}\rightarrow\hspace{5mu} H \times W \times \frac{C}{2}}_{\text{channel-to-space}} \quad \underbrace{\rightarrow\hspace{5mu} H \times W \times C}_{\text{duplicate + concat}} \qquad (26)
$$

block 输出 = 卷积主路 + 捷径输出，中间 stage 按同一原则改造。机制：捷径把可保结构的低频分量直通下去，每个 block 的可学支路只需拟合本级的局部残差，优化景观从「学整图」变成「逐级学差分」——深压缩下这是训得动与训不动的分界（论文 Figure 3 的消融对照）。

**(b) decoupled high-resolution adaptation（解耦高分辨率适配）**。动机是一笔实测账：f64 AE 从 $256^{2}$ 直接泛化到 $1024^{2}$ 时 rFID 从 0.50 掉到 7.40，而同设置下 f8 AE 反而从 0.51 改善到 0.19——论文称之为高空间压缩 AE 的 generalization penalty，压缩率越高越重。直接上高分辨率训练则两头挨打：训练成本高，且高分辨率 GAN 训练不稳定。DC-AE 的解法是三阶段（论文 Figure 6 口径）：(i) **phase 1 低分辨率全量训练**：完整 AE 在低分辨率（ $256^{2}$ 口径）上训，loss 为 L1 + LPIPS，不含 GAN；(ii) **phase 2 高分辨率 latent adaptation**：高分辨率下只调中间层——encoder 头部与 decoder 的**输入层**——其余冻结，loss 同 phase 1；作用是适配 latent 空间以缓解 generalization penalty，实验上只调中间层即够，显存从全量高分辨率训练的 153.98 GB 降到 67.81 GB（ $1024^{2}$、batch 12 口径）；(iii) **phase 3 低分辨率 local refinement**：GAN（PatchGAN）loss 解耦到本阶段才进场，只调 decoder 的头部层、其余全部冻结，并且在**低分辨率**上跑。phase 3 的机制依据有两条：重建 loss 单独已足够学到内容与语义，GAN loss 的边际贡献集中在局部细节与去 artifact（论文 Figure 5 对照），只调 decoder 头部即可拿到这份收益且比全量训练更准；同时冻结其余层保证 GAN 训练不改动 latent 空间，于是本阶段可以留在低分辨率、不必担心 generalization penalty——既绕开高分辨率 GAN 的不稳定，又再省一份训练成本。

账： $512^{2}$ 图经 f32 得 $16 \times 16 = 256$ 个 latent 位点、f64 得 $8 \times 8 = 64$ 个——对照 §1 的像素账，序列长度再省一到两个量级。DC-AE 是 SANA 文生图系统的 latent 基石（SANA 用 f32 档，P7 展开）；官方实现在 mit-han-lab/efficientvit 的 `applications/dc_ae` 目录（[efficientvit](../references.md)，[A]）。

### 7.4 离散 vs 连续：选择表

| 维度 | 连续 latent（KL-VAE 系，§2、§7.1–7.3） | 离散 token（VQ/LFQ 系，§3–§6） |
| --- | --- | --- |
| latent 形态 | 连续张量 $h \times w \times c$，高斯后验采样 | 每位点 1 个（RVQ 为 $n_q$ 个）codebook 下标 |
| 配套生成范式 | diffusion / flow 在连续空间回归（P7 主线） | AR / transformer 逐 token 预测（P8 视觉生成） |
| 先验语义 | 显式 $\mathcal{N}(0, I)$，ELBO 可算（式 (3)(4)） | 无连续先验；先验由离散网格上的 AR 模型学（PixelCNN 是原型） |
| 正则机制 | $\beta \cdot$ KL，闭式（式 (7)） | commitment + codebook/EMA（式 (14)(15)）或熵正则（式 (23)） |
| 压缩前沿 | f8（SD 系）→ f32/f64（DC-AE） | f8–f16 主流；大词表走 LFQ（ $2^{d}$ 隐式） |
| 失效模式 | posterior collapse（§2.6） | codebook collapse（§3.4） |
| 代表 | SD-VAE f8/4ch、SD3/FLUX f8/16ch、DC-AE f32/f64 | VQ-VAE、VQGAN f16/K=1024、MAGVIT-v2 LFQ |

选择不由「谁重建更好」定，由下游生成器吃哪种序列定：diffusion 的 score 回归天然作用在连续向量上，离散 token 反而要额外嵌入层；AR 的 softmax 输出头需要有限词表，连续 latent 无法直接进。两族因此长期并存——P7 默认连续族、P8 的 AR 多模态路线默认离散族，本章全部公式在那两处直接复用。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 压缩的账 | $z \in \mathbb{R}^{h \times w \times c}$， $f = H/h$ | $512^{2}$ 图：f8 → 4096 位点、f32 → 256、f64 → 64；可学压缩 vs 硬切 patch |
| ELBO | 式 (3)(4)(5) | Jensen 与 KL 两条路径；gap = KL(q‖真后验)；= recon − KL(q‖先验) |
| Gaussian KL | 式 (7)： $\frac{1}{2}\sum(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1)$ | 闭式可微，无采样噪声 |
| reparameterization | 式 (9)(10)： $z = \mu + \sigma \odot \varepsilon$ | 期望外层换成无参噪声；梯度经 $\nabla_z f$ 传导，方差远低于 log-derivative |
| posterior collapse | §2.6 | KL 单独最优 = 后塌成先验；decoder 过强时 recon 不抵抗；缓解：β 退火 / free bits / 弱 decoder |
| VQ-VAE 量化 | 式 (11)：argmin 最近邻 | latent = K 值词表上的 $hw$ 网格；接 PixelCNN 先验 |
| STE | 式 (12)(13)： $z_e + \mathrm{sg}[z_q - z_e]$ | 前向硬量化、反向梯度拷贝；误差正比于量化残差，由 commitment 项压住 |
| VQ-VAE loss | 式 (14)(15) | recon + codebook + commitment（β=0.25）；EMA（γ=0.99，Appendix A.1 替代方案，主实验未用）= K-Means 闭式解的在线版 |
| codebook collapse | §3.4 | 不选中 → 不更新的死锁；诊断用使用分布 perplexity；缓解：EMA / 死码重启 / 熵正则 |
| VQGAN | 式 (16)(17)(18)(19) | LPIPS 换特征空间监督；PatchGAN + hinge；自适应 λ = 梯度范数比 |
| RVQ | 式 (20)(21) | 残差递推 $r_{i+1} = r_i - e_i$；容量 $K^{n_q}$；bitrate $= n_q \log_2 K \cdot f_{\mathrm{rate}}$，P10 直接复用 |
| LFQ | 式 (22)(23) | sign 量化、隐式 codebook $2^{d}$；逐样本熵压、聚合熵抬 |
| 现代 AE | 式 (24)(25)(26) | SD-VAE f8/4ch、s=0.18215 单位方差化；FLUX 16ch、加 shift；DC-AE f32/f64：逐 block 非参捷径 + 三阶段适配（phase 2 调中间层高分辨率适配、phase 3 低分辨率只调 decoder 头部上 GAN） |
| 离散 vs 连续 | §7.4 表 | 由下游生成范式定：diffusion 用连续、AR 用离散 |

## 练习

1.（推导）补全式 (6) 到式 (7)：对 $q = \mathcal{N}(\mu, \sigma^{2})$、 $p = \mathcal{N}(0, 1)$，从 $`\mathrm{KL}(q \Vert p) = \mathbb{E}_q[\log q - \log p]`$ 出发，写出两个密度的完整形式、逐项取期望，得到单维闭式；再说明多维对角情形为何是逐维求和。验证 $\mu = 0, \sigma = 1$ 时取 0、 $\sigma = 1$ 时退化为 $\mu^{2}/2$。
2.（推导）EMA 与均值闭式解的关系：证明式 (14) 第二项 $\sum_{z \in S_i} \Vert z - e_i \Vert^{2}$ 关于 $e_i$ 的全局极小是 $`\bar{z}_i = |S_i|^{-1} \sum_{z \in S_i} z`$；再说明式 (15) 在 $N_i$ 收敛（ $`N_i \approx \bar{n}_i`$ 的稳态）时等价于对历史 batch 的 $`\bar{z}_i`$ 做指数滑动平均，并讨论 $\gamma = 0.99$ 对应的有效窗口长度量级。
3.（计算）(a) 用式 (21) 复算文中实例（ $n_q = 8$、 $K = 1024$、75 fps）；(b) 码率预算压到 1.5 kbps、 $K$ 不变，给出两组可行的 $(n_q, f_{\mathrm{rate}})$ 组合并比较各自的重建风险（提示： $n_q$ 与 $f_{\mathrm{rate}}$ 分别伤什么）；(c) $K = 1024$、 $n_q = 8$ 时有效词表是 $2^{80}$，若换成单级 VQ 达到同容量，词表多大、 $d_e = 256$ 时参数多少？以此说明加法结构的参数效率。
4.（代码）教学用，非优化实现：实现式 (11)(12) 的 VQ 层——`z_q = z_e + (z_q - z_e).detach()`，随机初始化 codebook、encoder 用两层 MLP，在合成二维数据上训练；(a) 打印训练前后 codebook 使用分布的 perplexity $\exp(H(u))$；(b) 把 codebook 初始化拉到远离数据的位置复训，观察 collapse 的发生与 perplexity 曲线；(c) 加入死码重启（使用率低于阈值的 entry 重置为 batch 内随机 $z_e$）对照。验证 STE：对同一输入，比较 `loss.backward()` 后 `z_e.grad` 与手动拷贝的 decoder 输入端梯度是否一致。

## 参考

- [Kingma & Welling, 2013](../references.md)：Auto-Encoding Variational Bayes，arXiv:1312.6114，[A]（key: vae）——式 (1)(2)(3)(4)(5)(6)(7)(8)(9)(10)，ELBO 双路径与 reparameterization
- [van den Oord et al., 2017](../references.md)：Neural Discrete Representation Learning，arXiv:1711.00937，[A]（key: vq-vae）——式 (11)(12)(13)(14)(15)，STE、commitment（β=0.25）、EMA（γ=0.99，Appendix A.1 替代方案、主实验未用）、PixelCNN 先验
- [Esser et al., 2020](../references.md)：Taming Transformers for High-Resolution Image Synthesis，arXiv:2012.09841，[A]（key: vqgan）——式 (16)(17)(18)(19)，LPIPS、PatchGAN、自适应 λ、f16/K=1024 档
- [Yu et al., 2023](../references.md)：Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation，arXiv:2310.05737，[A]（key: magvit-v2）——式 (22)(23)，LFQ 与熵正则、图像视频统一 vocabulary
- [Chen et al., 2024](../references.md)：Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models，arXiv:2410.10733，[A]（key: dc-ae）——式 (25)(26)，f32/f64、residual autoencoding 与三阶段适配
- [CompVis/taming-transformers](../references.md)：[A]（key: taming-transformers-repo）——`taming/models/vqgan.py`（模型）、`taming/modules/losses/vqperceptual.py`（LPIPS/hinge loss），VQGAN 官方实现
- [mit-han-lab/efficientvit](../references.md)：[A]（key: efficientvit-repo）——`applications/dc_ae`，DC-AE 官方实现
- [diffusers AutoencoderKL](../references.md)：[A]（key: autoencoderkl）——`src/diffusers/models/autoencoders/autoencoder_kl.py`，SD 系 VAE 走读锚点
- [lucidrains/vector-quantize-pytorch](../references.md)：教学参照（key: vector-quantize-pytorch）——VQ/RVQ/LFQ 教学实现，死码重启配置对照
