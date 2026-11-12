# 10.3 语音合成（Text-to-Speech Synthesis）

> 合成是识别的逆问题，但不是它的镜像：$P(x \mid y)$ 是一对多映射——同一文本有无数种合法读法——必须有一个潜变量来吸收「说话方式」（时长、韵律、音高、音色），三条路线正是这个潜变量的三种安置方案：级联管线把它拆成显式监督模块（duration/pitch/energy），LM 路线把它压进 codec token 上的 softmax 分布，FM 路线把它交给噪声与 infilling 构造。本章给每条路线一个推导锚点：HiFi-GAN 的 MPD 质数周期折叠与 feature matching（vocoder 的对抗工程）、VITS 的 VAE+flow+MAS 统一目标、VALL-E 的 AR+NAR 两阶段分解、F5-TTS 的 infilling CFM 与 Sway Sampling；最后落级联 / LM / FM 三路线对照表 | 前置：1.1（CE 与采样）、2.4（KV cache 账）、6.3（ELBO 式 (3)、VQ/RVQ、LFQ）、7.1（GAN minimax 式 (3)、flow change-of-variables 式 (4)）、7.5（FM 式 (7)、CFM 式 (9)、rectified flow、timestep shift）、7.6（DiT）、7.7（mask-predict）、10.1（codec 六家账与式 (7) bitrate、semantic vs acoustic 对照）、10.2（对齐作为潜变量的另一种处理、前缀条件化式 (21)） | 锚点：jik876/hifi-gan（models.py）[A]、jaywalnut310/vits [A]、QwenAudio/CosyVoice [A]、SWivid/F5-TTS [A]、open-mmlab/Amphion（MaskGCT）[A]；VALL-E / VALL-E 2 / E2-TTS / VoiceBox / NS3 均 [B]（无官方代码，公式引论文原文）

记号约定（沿用 10.1/10.2）：文本序列 $y = (y_1, \ldots, y_U)$（音素或字符级，$U$ 约 10 fps 量级）；声学侧两种形态——mel 帧序列 $x \in \mathbb{R}^{T \times K}$（$T$ 帧，100 fps 量级）与 codec 码矩阵 $C \in \{0, \ldots, K-1\}^{T \times n_q}$（$c_{t,j}$ = 帧 $t$ 第 $j$ 个量化器的码，RVQ 记号同 10.1）；波形 $x[n]$。两处 $K$ 各自定义（mel 通道数 / 码本大小），随 10.1 惯例并置、按上下文区分。

## 1. 形式化：一对多映射是 TTS 的第一约束

10.2 式 (1) 把识别写成 $P(y \mid x)$ 且 $U \ll T$；合成是它的反向：

$$
P(x \mid y), \qquad T \gg U \qquad (1)
$$

**符号**：$x$：mel 帧序列或波形（本节按 mel 口径）；$y$：文本序列。方向一反过来，10.2 的两个困难各自变形。其一，长度不匹配从「多对一」变「一对多」：约 10 fps 的文本要撑出 100 fps 的声学帧，每个音素占多少帧（duration/alignment）从「要边缘化掉的冗余」变成「必须显式或隐式决定的生成量」。其二，识别的输出在多数情形下近似唯一（转写错一字即错），合成的输出天然一对多：同一句话可以快慢、升降、喜悲，全部由说话方式决定。把这个自由度写成潜变量 $z$：

$$
P(x \mid y) = \int P(x \mid y, z)\, p(z \mid y)\, dz \qquad (2)
$$

**符号**：$z$：未观测的说话方式潜变量（时长选择、韵律轮廓、音高、音色的合称）；$p(z \mid y)$：说话方式先验。

式 (2) 是全章的形式骨架，三条路线对 $z$ 的安置互不兼容：

1. **级联管线**（§2–§5）：$z$ 拆成一组显式模块——duration model、pitch/energy 预测器、speaker embedding——各自带监督训练，推理时逐模块取值。可控性最好（每个旋钮可人工覆盖），代价是误差沿模块链累积。
2. **LM 路线**（§6–§7）：$x$ 先离散化成 codec token（10.1 的骨架在此兑现），$z$ 不再显式出现，全部压进条件 softmax $P(c_{t,j} \mid \cdot)$ 的分布形状里；音色等条件以 prompt 前缀注入。
3. **FM/diffusion 路线**（§8）：$z$ 由采样噪声与 ODE 轨迹吸收（7.5 的框架），对齐这个 $z$ 中最硬的部分则靠 infilling 的输入构造直接消掉。

三条路线共用同一个出口：无论 mel 还是 codec token，最后都要回波形——vocoder 因此是三条路线共用的部件，也是 §3–§4 先讲它的原因。

## 2. 级联管线：把 $z$ 拆成显式模块

级联管线的数据流（各模块独立训练、推理时串联）：

```
文本 ──G2P──► 音素序列 y₁..y_U
                 │
                 ▼
        duration model（forced alignment 监督）
                 │  每音素帧数 d_u
                 ▼
        length regulator：h_u 重复 d_u 次 → T 帧
                 │  variance adaptor 再加 pitch/energy
                 ▼
        acoustic model ──► mel 序列 x ∈ R^{T×K}
                 │
                 ▼
        vocoder ──► 波形 x̂[n]
```

G2P（grapheme-to-phoneme，字素转音素）把正字法文本转成音素序列；duration model 的训练监督来自 forced alignment（强制对齐——用现成对齐工具把音素序列与录音逐帧对齐，常用实现为 Montreal Forced Aligner）——10.2 里 CTC 当作潜变量边缘化掉的「对齐」，在级联 TTS 里被当成可离线生产的监督信号。length regulator 按时长把音素级 hidden 复制展开到帧级：

$$
\mathrm{LR}\left( (h_1, \ldots, h_U),\, (d_1, \ldots, d_U) \right) = (\underbrace{h_1, \ldots, h_1}_{d_1\ \text{times}},\ \ldots,\ \underbrace{h_U, \ldots, h_U}_{d_U\ \text{times}}), \qquad T = \sum_{u=1}^{U} d_u \qquad (3)
$$

**符号**：$h_u$：第 $u$ 个音素的隐向量（文本编码器输出）；$d_u$：该音素占据的帧数；$\mathrm{LR}$：length regulator，纯复制展开，无参数。

FastSpeech2（[Ren et al., 2020](../references.md)，arXiv:2006.04558，[B]——微软未释官方代码，机制按论文口径）在这条链上补 variance adaptor：「the variance adaptor adds different variance information such as duration, pitch and energy into the hidden sequence」——音素级、非自回归；pitch 与 energy 从语音中提取、各配一个小预测器，展开后逐帧加进 hidden：

$$
\tilde{x}_t = \mathrm{LR}(h, d)_t + E_p(\hat{p}_t) + E_e(\hat{e}_t) \qquad (4)
$$

**符号**：$\hat{p}_t, \hat{e}_t$：预测的帧级 pitch / energy 标量；$E_p, E_e$：各自的嵌入变换；$\tilde{x}_t$：加上韵律信息后的帧级表示，送 acoustic model 出 mel。

机制分工：duration 钉节奏骨架（式 (3)），pitch/energy 补韵律色彩（式 (4)）——式 (2) 的 $z$ 在这里被拆成了三个可分离监督的标量场。收益与代价同根：每个模块的输入在训练时是真值、推理时是上游预测——跨模块的 exposure bias；但每个旋钮也因此可人工干预（改 $d_u$ 即改时长），这条性质在 §10 的对照表里是级联路线至今未丢的一格。

## 3. HiFi-GAN：vocoder 的判别器工程

vocoder 的账：100 fps 量级的 mel 到 24 kHz 波形，每秒要撑起 24000 个采样点——约 240 倍上采样，且输出是准周期信号（浊音），周期保真度直接决定听感。HiFi-GAN（[Kong et al., 2020](../references.md)，arXiv:2010.05646，[A]；repo jik876/hifi-gan，models.py 文件级实测）的组成：一个 generator + 两组判别器（MPD 与 MSD）。

**Generator**：上采样栈（转置卷积级联，各级 stride 之积等于 mel 帧的 hop）把帧级特征逐级升到采样率，沿途挂 MRF（multi-receptive field fusion：多组不同 kernel/dilation 组合的残差块并联、输出求和）——不同感受野分支各盯一个时间尺度（音高周期级、包络级），并联求和相当于让生成器在每个尺度上同时过一遍质量控制。

**MPD（multi-period discriminator）**是本节要精算的部件。语音的关键结构是周期性，但周期性在 1D 时间轴上的呈现是「相距 $p$ 个样本的点强相关」——小 kernel 的 1D 卷积一次只看到相邻十几个样本，看不见相距一个基音周期的两个样本之间的关系。MPD 的处理是折叠：把 1D 波形按周期 $p$ 重排成 2D（原文：「we first reshape 1D raw audio of length $T$ into 2D data of height $T/p$ and width $p$ and then apply 2D convolutions」）：

$$
X_p[i, j] = x[i \cdot p + j], \qquad i = 0, \ldots, T/p - 1, \quad j = 0, \ldots, p - 1 \qquad (5)
$$

**符号**：$T$：波形样本数；$p$：折叠周期；$X_p \in \mathbb{R}^{(T/p) \times p}$；$T$ 不整除 $p$ 时尾部补齐至整行。

折叠的几何读法：第 $j$ 列 = 每周期内相位 $j$ 处的采样序列（沿周期轴演化），第 $i$ 行 = 第 $i$ 个周期的一次完整采样。准周期信号折完后呈现「竖直条纹」结构——同相位列彼此相似、行间缓变——2D 卷积的一次 kernel 操作同时看到「周期内相位轴（横向）× 周期间演化轴（纵向）」，对周期保真度的感知由此变成局部的。

周期取质数：原文「We set the periods to $[2, 3, 5, 7, 11]$」。机制读法：合数周期的折叠结构可由质因子的折叠推断——周期 4 的条纹模式是周期 2 折叠后再两两合并的特例，周期 6 同理归约到 2 与 3；取质数让五个判别器各盯一族不可互相归约的周期结构，同样的判别器预算下覆盖互不相同。

**MSD（multi-scale discriminator）**：同一波形在多个平均池化降采样尺度上各设一个判别器——粗尺度盯包络与长程结构，细尺度盯波形细节。与 10.1 式 (9) 的 multi-scale STFT 判别器同族：一个在时域池化，一个在时频域变窗，目的一致——单个判别器只有一对时频分辨率（10.1 式 (3) 的不确定性约束），一族判别器才能把两维同时盯住。

**损失**三件。对抗项按 7.1 式 (3) 的 minimax 方向（判别器族 $\{D_d\}$ 最大化、生成器最小化；实现取 least-squares 变体——models.py 实测），此处把另外两个稳定件就地讲清。feature matching loss（Larsen et al. 2016 式）：

$$
L_{\mathrm{fm}}(G) = \mathbb{E}\left[ \sum_{d} \sum_{l} \frac{1}{N_{d,l}} \left\| D_d^{(l)}(x) - D_d^{(l)}\left( G(s) \right) \right\|_1 \right] \qquad (6)
$$

**符号**：$s$：输入 mel；$x$：真实波形；$D_d^{(l)}$：第 $d$ 个判别器第 $l$ 层的中间激活；$N_{d,l}$：该层激活的元素数；期望取遍数据。

机制：纯对抗项只奖励「骗过判别器」——判别器的盲区就是生成器的免费午餐，7.1 §4 的 mode dropping 由此而来。feature matching 把生成器锚到真实数据在判别器特征空间里的逐层统计上：判别器每学一步、锚随之更新，形成一条自我稳定回路——它奖励的不是「骗过」而是「统计匹配」，覆盖度由此进入目标函数。

mel-spectrogram loss：

$$
L_{\mathrm{mel}}(G) = \left\| \phi(x) - \phi\left( G(s) \right) \right\|_1 \qquad (7)
$$

**符号**：$\phi$：10.1 式 (2)(5) 的 STFT → mel 提取链。把感知上最重要的幅度谱直接做成回归项，对抗项得以专注补回归学不到的细节（相位相干的纹理、谐波精细结构）。生成器总目标：

$$
L_G = L_{\mathrm{adv}}(G) + \lambda_{\mathrm{fm}} L_{\mathrm{fm}}(G) + \lambda_{\mathrm{mel}} L_{\mathrm{mel}}(G) \qquad (8)
$$

**符号**：$\lambda_{\mathrm{fm}}, \lambda_{\mathrm{mel}}$：两项辅助损失的权重（实现常数，见 repo）。式 (8) 的结构——一个对抗项 + 一个特征空间匹配项 + 一个感知域回归项——此后成为 vocoder 与 codec decoder（10.1 §4）的共享模板。

## 4. BigVGAN：snake 激活与混叠账

BigVGAN（[Lee et al., 2022](../references.md)，arXiv:2206.04658，[A]；repo NVIDIA/BigVGAN）把 vocoder 推向「universal」档（大规模训练、跨说话人/跨录音条件泛化），核心改动之一是把激活函数换成带周期先验的 snake：

$$
f_{\alpha}(x) = x + \frac{1}{\alpha} \sin^2(\alpha x) \qquad (9)
$$

**符号**：$\alpha > 0$：可训练标量（逐通道），控制周期分量的频率。

两项分工：$x$ 保留恒等通路——信息与梯度直通，输入不被压弯；$\frac{1}{\alpha}\sin^2(\alpha x)$ 是有界周期扰动（幅值不超过 $1/\alpha$）——激活响应呈「单调主趋势 + 周期起伏」，正是准周期音频信号的形状先验。求导一步：

$$
f'_{\alpha}(x) = 1 + \frac{1}{\alpha} \cdot 2 \sin(\alpha x) \cos(\alpha x) \cdot \alpha = 1 + \sin(2 \alpha x) \;\in\; [0, 2] \qquad (10)
$$

三条读法：(i) 导数处处非负——无 ReLU 式死区，且 $f_\alpha$ 单调、不把输入折叠（折叠会让不同输入挤到同一输出，丢信息）；(ii) 导数本身以频率 $2\alpha$ 振荡——梯度场携带周期结构，网络学周期函数时不必从零拼出正弦；(iii) $\alpha$ 逐通道可学，不同通道可以分工到不同频率档——这给了上采样栈一条「逐层搭谐波」的组织方式。

周期先验的另一面是混叠。非线性作用在离散信号上会产生谐波：$\sin^2$ 作用在正弦上由 $\sin^2\theta = (1 - \cos 2\theta)/2$ 直接倍频。采样率 $f_s$ 下，任何高于 $f_s/2$ 的分量折叠回 $[0, f_s/2]$（10.1 式 (1) Nyquist 的直接推论）：24 kHz 系统中一个 15 kHz 的生成谐波混叠到 $24 - 15 = 9$ kHz——落在可听带内的虚假分量。上采样与非线性交替的栈把这个效应逐级放大。BigVGAN 的对策是给 snake 配 low-pass filter 做抗混叠（原文消融记录该组件引入约 19 倍的差距量级），把谐波能量在折叠发生前截掉——「universal」泛化档与这个细节直接相关：混叠分量是录音条件相关的，截掉它等于删掉一类跨域分布偏移。

## 5. VITS：把整条级联并成一个 ELBO

§2 管线每个模块各自训练、推理串联消费——误差逐级累积是结构性的，解法是把对齐、声学、vocoder 并进同一个生成目标。VITS（[Kim et al., 2021](../references.md)，arXiv:2106.06103，[A]；repo jaywalnut310/vits，models.py 文件级实测）是这条路的代表：「variational inference augmented with normalizing flows and an adversarial training process」，外加 Monotonic Alignment Search（MAS）对齐估计与 stochastic duration predictor。潜变量设定：帧级 latent $z$（后验编码器吃线性谱给出）、文本条件 $c$（音素编码）、对齐 $A$（潜变量）。骨架是条件 VAE（ELBO 推导 recap 6.3 式 (3)）：

$$
\log p_{\theta}(x \mid c) \;\ge\; \mathbb{E}_{q_{\phi}(z \mid x)}\left[ \log p_{\theta}(x \mid z) \right] - \mathrm{KL}\left( q_{\phi}(z \mid x)\, \|\, p_{\theta}(z \mid c, A) \right) \qquad (11)
$$

**符号**：$q_{\phi}(z \mid x)$：后验编码器（训练时可见声学）；$p_{\theta}(x \mid z)$：decoder——HiFi-GAN 式生成器，$z \to$ 波形；$p_{\theta}(z \mid c, A)$：文本条件下的先验。

先验若只是逐帧独立高斯，式 (11) 的 KL 项会把后验也压向逐帧独立高斯——韵律的帧间连续性无处安放。VITS 的处理是在条件先验上叠 normalizing flow（change of variables，recap 7.1 式 (4)）：

$$
\log p_{\theta}(z \mid c, A) = \log \mathcal{N}\left( f^{-1}(z);\, \mu_{\theta}(c, A),\, \sigma_{\theta}(c, A)^2 \right) + \log \left| \det \frac{\partial f^{-1}(z)}{\partial z} \right| \qquad (12)
$$

**符号**：$f$：可逆变换栈（affine coupling 族——Jacobian 三角、log-det 便宜，7.1 §5 的架构约束在此兑现）；$\mu_{\theta}, \sigma_{\theta}$：先验编码器在文本与对齐 $A$ 下给出的逐帧高斯参数。

机制：flow 把简单高斯弯成有能力逼近后验族的分布——6.3 §2.3 的口径里 ELBO 的 gap 是后验近似误差，先验族越灵活、同样的 $q$ 能被承接得越紧。方向性细节：训练沿 KL 梯度估值要用 $f^{-1}$（把后验样本 $z$ 映回简单密度下评估），推理采样要用 $f$（$\varepsilon \sim \mathcal{N}(\mu, \sigma^2)$，$z = f(\varepsilon)$）——两个方向都要便宜，这正是 coupling 层设计的动机。

**MAS**：对齐 $A$ 是潜变量，VITS 不走 10.2 式 (6) 的边缘化（对全部对齐求和），而是取硬最优——在单调对齐集合上搜使后验样本在先验下似然最大的那条：

$$
A^{*} = \operatorname{arg\,max}_{A \in \mathcal{A}_{\mathrm{mono}}} \sum_{t=1}^{T} \log \mathcal{N}\left( z_t;\, \mu_{a(t)},\, \sigma_{a(t)}^2 \right) \qquad (13)
$$

**符号**：$\mathcal{A}_{\mathrm{mono}}$：单调对齐集合——每帧恰好指派一个音素、音素下标随帧不减，等价于给每个音素分配正整数时长 $d_u \ge 1$ 且 $\sum_u d_u = T$；$a(t)$：帧 $t$ 被指派的音素；$z_t$：后验采样的第 $t$ 帧 latent。

求解结构：单调约束给出与 CTC 同族的格点（10.2 §2.3 的扩展序列在这里换成「音素段」），把 10.2 式 (6) 的「对全部路径求和」换成「对全部对齐取 max」即 Viterbi 式 DP，$O(TU)$ 精确求解——每步训练迭代先跑一次 MAS 得 $A^{*}$，再按式 (11) 的 ELBO 更新。与 CTC 的对照读法值得停一拍：同一个「对齐未知」问题，识别侧求和（软，保留全部对齐的不确定性），合成侧取 max（硬，只要一条最可能的）——硬指派恰好给出式 (3) 所需的显式 $d_u$，它也是 stochastic duration predictor 的训练监督来源。duration predictor 本身用 flow 建时长分布（条件于音素编码），推理时采样 $d_u$——同一文本的韵律多样性由这个采样给出，而不是由确定性的时长回归钉死。

训练目标五项并列：

$$
L_{\mathrm{VITS}} = L_{\mathrm{recon}} + L_{\mathrm{kl}} + L_{\mathrm{dur}} + L_{\mathrm{adv}}(G) + \lambda_{\mathrm{fm}} L_{\mathrm{fm}}(G) \qquad (14)
$$

**符号**：$L_{\mathrm{recon}}$：式 (7) 的 mel L1（$G(z)$ 与 $x$ 之间）；$L_{\mathrm{kl}}$：式 (11) 第二项（含式 (12) 的 flow 修正）；$L_{\mathrm{dur}}$：时长预测损失；后两项是 §3 式 (6)(8) 的对抗工程原样搬进端到端框架（decoder 即 HiFi-GAN 式生成器）。

统一目标的读法：§2 的模块边界在 VITS 里全部变成同一计算图里的不同损失项，梯度从波形判别器一路回流到文本编码器。代价对称地清楚：中间表示不再可单独替换（想换 vocoder 得整体重训），可控性也从「逐旋钮覆盖」退化为「对潜变量与时长采样」——§10 的对照表里，这两格正好与级联路线互补。

## 6. LM 路线：VALL-E 的两阶段分解

10.1 §6.3 的离散 token 进 LM 分析在此兑现其生成侧一半：codec token 把语音变成离散序列后，TTS 与文本条件生成同构——对 LM 而言，acoustic token 等价于又一种待建模的离散语言。设定：10.1 §4 骨架的 RVQ codec，$n_q = 8$ 个量化器（论文公式中 $j \in [2, 8]$ 的口径），码矩阵 $C \in \{0, \ldots, K-1\}^{T \times 8}$；文本侧转为音素序列 $y$；另有一段 acoustic prompt $\tilde{C}$（参考录音的 codec 码）。

VALL-E（[Wang et al., 2023](../references.md)，arXiv:2301.02111，[B]——微软未释官方代码与权重，公式引论文原文）把联合分布按量化器层做链式分解，再在每层内选择时间结构：

$$
P(C \mid y, \tilde{C}) = P\left( c_{:,1} \mid y,\, \tilde{C}_{:,1} \right) \cdot \prod_{j=2}^{8} P\left( c_{:,j} \mid c_{:,<j},\, y,\, \tilde{C} \right) \qquad (15)
$$

**符号**：$c_{:,j} = (c_{1,j}, \ldots, c_{T,j})$：第 $j$ 层的整条时间序列；$c_{:,<j}$：第 $1$ 至 $j-1$ 层的全部码；$\tilde{C}_{:,1}$：prompt 的第一层码；$\tilde{C}$：prompt 的全部层。

第一层用 AR（原文：「For the discrete tokens from the first quantizer $c_{:,1}$, we train an autoregressive (AR) decoder-only language model……conditioned on the phoneme sequence and the acoustic prompt $\tilde{C}_{:,1}$」）：

$$
P\left( c_{:,1} \mid y,\, \tilde{C}_{:,1} \right) = \prod_{t=1}^{T} P\left( c_{t,1} \mid c_{<t,1},\, y,\, \tilde{C}_{:,1} \right) \qquad (16)
$$

**符号**：$c_{<t,1}$：第一层前 $t-1$ 帧的码；条件中的 prompt 只含第一层——高层级 prompt 信息留给第二阶段。

第 2 至 8 层用 NAR（原文：「we train a non-autoregressive (NAR) language model」建模 $c_{:,j \in [2,8]}$）：

$$
P\left( c_{:,j} \mid c_{:,<j},\, y,\, \tilde{C} \right) = \prod_{t=1}^{T} P\left( c_{t,j} \mid c_{:,<j},\, y,\, \tilde{C} \right), \qquad j = 2, \ldots, 8 \qquad (17)
$$

**符号**：右端的乘积沿时间并行——层内各帧联合预测、互不依赖；帧 $t$ 第 $j$ 层的码只依赖「前 $j-1$ 层的全部帧」，不依赖「第 $j$ 层的前 $t-1$ 帧」。层间仍串行：第 $j$ 层全部采样完才进第 $j+1$ 层。

```
输入：音素序列 y + prompt 第一层码 C̃_{:,1}
  │
  ▼
AR LM（decoder-only，逐帧 t=1..T）──► c_{:,1}（第一层码）
  │（第一层全部 T 帧就绪）
  ▼
NAR LM：j = 2..8 逐层推进，层内 T 帧并行 ──► c_{:,2}, …, c_{:,8}
  │
  ▼
codec decoder ──► 波形 x̂[n]
```

为什么这样切，三笔账。(i) RVQ 的第一级承载最粗结构（6.3 §5 残差递推：$e_1$ 最接近 $z$ 本身，后续各级只补残差细节）——内容与韵律的全局连贯主要在第 1 层，自回归的全局一致性花在刀刃上。(ii) 第 2–8 层是逐级精修：给定全部低层码后，高层码的帧间依赖弱，并行化的建模误差代价小。(iii) 步数账：把 $8T$ 个码全部摊平自回归是 $8T$ 步严格串行；两阶段是 $T$ 步 AR 加 7 趟层内并行前向。按 10.1 §4.2 的 8 码本档帧率（75 fps）估：每秒音频的串行步数从约 600 降到 75 步加 7 趟——延迟结构完全不同。

**3 秒 prompt 的 in-context cloning**（原文：「only a 3-second enrolled recording of an unseen speaker as an acoustic prompt」）：机制是前缀条件化——与 10.2 式 (21) 的 Whisper 前缀同一原理、方向相反（那里前缀指定任务，这里前缀指定音色）。训练样本里 prompt 与目标是同一说话人同一录音的片段，LM 学到的是「延续 prompt 的音色与声学环境」；推理时把前缀换成 3 秒未见说话人的录音，条件分布的音色轴随前缀移动。无 speaker embedding、无微调——LM 路线对级联路线 d-vector 模块的结构性替代就在这一步：音色控制从「一个外挂模块的输出」变成「序列条件本身的属性」。

**VALL-E 2**（[Chen et al., 2024](../references.md)，arXiv:2406.05370，[B]）的两个补丁都打在式 (16)(17) 的已知病上。**Repetition Aware Sampling**：原文「refines the original nucleus sampling process by accounting for token repetition in the decoding history……stabilizes the decoding……circumvents the infinite loop issue」——AR 长序列在 codec 码上的重复坍缩（同一码段无限循环）是该路线的典型失败，补丁在采样器层：按解码历史中的重复情况在 random 与 nucleus 采样间自适应切换。**Grouped Code Modeling**：「organizes codec codes into groups to effectively shorten the sequence length」——把一帧的多个码编组建模，机制上等价于把式 (16) 的步长按组放大、压缩 AR 段长度。标题级主张 human parity，证据等级维持 [B]。

## 7. CosyVoice 系：监督语义 token + LM + FM decoder

VALL-E 的 token 是声学重建导向的（10.1 §6.2 对照表：纯声学码本不保证音素可分性）——CosyVoice（[Du et al., 2024](../references.md)，arXiv:2407.05407，[A]；repo QwenAudio/CosyVoice，FunAudioLLM/CosyVoice 301 重定向链，cosyvoice/{tokenizer, flow, llm, hifigan} 目录实测）的回答是换 token 来源：**supervised semantic tokens**，「derived from a multilingual speech recognition model by inserting vector quantization into the encoder」——在一个多语 ASR encoder 内部插入 VQ 瓶颈层，量化下标即 token。命名纪律先立：三篇论文正文均无「S3Tokenizer」一词（grep=0）；该名见于官方生态（s3tokenizer pip 包与 speech_tokenizer_v*.onnx 模型文件命名，二手渠道一致佐证）——本书口径：官方生态称 S3Tokenizer（Supervised Semantic Speech Tokenizer），论文正文只写 supervised semantic tokens。

机制：ASR 的转写监督把 encoder 状态压向「识别所需信息 = 语音学内容」——音色、信道、韵律波动对转写是 nuisance，梯度天然抑制它们；VQ 瓶颈再把这个方向压成离散。与 10.1 §3 的 HuBERT 对照：一个靠 masked prediction 自举内容标签，一个靠转写监督直取——殊途同归到「内容纯化」，但监督版的 token 直接进入 LM 词表时分布更干净（与文本 token 的统计距离更小）。

管线三段式：

```
文本 ──► LLM（text-to-token 生成，自回归）
             │  semantic tokens
             ▼
     conditional FM decoder ──► mel
             │  条件：semantic tokens + speaker embedding
             ▼
     HiFi-GAN vocoder（§3 部件复用）──► 波形
```

token-to-speech 段是 conditional flow matching（recap 7.5 式 (9)）：以 semantic token 序列（加说话人条件）为条件、在 mel 空间回归条件速度场，再经 §3 的 HiFi-GAN 部件回波形。结构注记：这条链是「LM 路线」与「级联路线」的混血——文本→token 用 LM（拿 in-context 与语言先验的好处），token→mel→波形回到连续域级联（拿 FM 与 vocoder 的音质好处）；codec 的 RVQ 层级结构（VALL-E 的 8 层账）被整个绕开——semantic token 单流逐帧，没有式 (17) 的层间分解问题。

**CosyVoice 2**（[Du et al., 2024](../references.md)，arXiv:2412.10117，[A]）四个改动各值一读。(i) **chunk-aware causal FM**：「enabling both streaming and non-streaming synthesis within a single model」——FM decoder 按 chunk 因果化（第 $k$ 块 mel 只可见第 $\le k$ 块的 token 条件与已生成 mel），同一权重整句输入即非流式、逐块输入即流式——流式与非流式从两套系统并成一套（mask 语义与 10.2 §7 的 chunk attention 同族）。(ii) tokenizer 的 VQ 换 **FSQ**（finite scalar quantization）：「Replacing vector quantization (VQ) in the speech tokenizer with finite scalar quantization (FSQ), improving codebook utilization」——lookup-free 量化族（6.3 §6 的 LFQ 近亲）：每维标量各自取有限档、码本为各维档数之积，无学习码本即无利用率塌缩。(iii) **去掉 text encoder 与 speaker embedding**，直用预训练文本 LLM 主干（Qwen2.5-0.5B）——文本理解整个外包给语言模型先验；speaker 信息随之改由序列条件携带（机制读法：与 §6 的 in-context prompt 逻辑合流）。(iv) token rate 25 Hz（原文：「Our speech tokenizer works at a token rate of 25 Hz, i.e., 25 speech tokens per second」）——LM 侧序列长度是 75 fps 档的三分之一（单码本流的 token/s = fps 本身，10.1 §6.1 对照表的 fps 列），式 (16) 的 AR 段每秒 25 步。

**CosyVoice 3**（[Du et al., 2025](../references.md)，arXiv:2505.17589，[A]）：FSQ 模块改插 **MinMo** voice encoder（CV2 为 SenseVoice-Large ASR encoder——两代差异原文直述），主线转向 scale-up + post-training，定位 in-the-wild 语音生成。

## 8. FM 路线：VoiceBox → E2-TTS → F5-TTS（infilling 走读）

这条路线从一个更激进的问题出发：级联要 duration model、LM 要 codec——两个都不要行不行？**VoiceBox**（[Le et al., 2023](../references.md)，arXiv:2306.15687，[B]——Meta 未开源）把 TTS 改写成填空：「text-guided speech infilling task, where the goal is to generate masked speech given its surrounding audio and text transcript」——任意位置挖掉一段语音，条件为上下文音频 + 文本转写，用 flow matching（optimal transport path——7.5 §6 的直线路径族）学回填。任意 mask 位置意味着编辑、去噪等多任务共用同一模型；更关键的是对齐成本被重新安置：构造自带的是「挖空段内容必与其文本段对应」的锚定关系（边界处的文本-音频配对），帧级对齐并不在输入中、由模型内学——duration 问题因此从显式建模降为隐式学习。

**E2-TTS**（[Eskimez et al., 2024](../references.md)，arXiv:2406.18009，[B]——微软未开源）把这个任务推到极简：「text input is converted into a character sequence with filler tokens……flow-matching-based mel spectrogram generator……trained based on the audio infilling task……does not require additional components (e.g., duration model, grapheme-to-phoneme)」——逐字符 + filler 的输入形态在此成型，G2P 一并省掉。

**F5-TTS**（[Chen et al., 2024](../references.md)，arXiv:2410.06885，[A]；repo SWivid/F5-TTS，src/f5_tts 目录实测）是这条路线的完整开放配方：infilling + CFM + DiT + Sway Sampling + ConvNeXt V2 文本精炼，全 NAR。逐件走读。

**Filler 对齐**（全路线的关键构造）：文本逐字符用 filler token 补齐到与语音等长（原文：「the text input is simply padded with filler tokens to the same length as input speech」）：

$$
\tilde{y} \in \left( \mathcal{V}_{\mathrm{char}} \cup \{\text{filler}\} \right)^{T}, \qquad \tilde{y}_t = \begin{cases} y_u, & t \in S_u \\ \text{filler}, & \text{otherwise} \end{cases} \qquad (18)
$$

**符号**：$T$：mel 帧数；$S_u$：第 $u$ 个字符实际占据的帧段（训练时即该字符的真实跨度，数据侧构造、无需模型）——帧 $t$ 落在字符 $u$ 的跨度内则该位取字符 $y_u$，全部跨度之外的帧位取 filler。

机制读法：§1 式 (2) 的 $z$ 中最硬的一块——每个音素占多少帧——被这一构造直接消解：对齐不再是潜变量，而是输入张量的位置对应。停在这里回看全章：CTC 的边缘化、VITS 的 MAS、FastSpeech 的 duration model 是同一个问题的三种回答，filler 补齐是第四种——把答案写进输入。代价同步转移到推理侧：目标长度 $T$ 从「模型要学」变成「调用方要给」，zero-shot 用法里按参考音频语速与文本长度比估计（机制推论）。

**CFM 目标**（recap 7.5 式 (9) 与 §6 rectified 路径）：取线性插值路径与闭式目标速度

$$
x_t = (1 - t)\, x_0 + t\, \varepsilon, \qquad u_t(x_t \mid x_0) = \varepsilon - x_0 \qquad (19)
$$

**符号**：$x_0$：数据端 mel（沿用 7.5 约定：$t = 0$ 数据端、$t = 1$ 噪声端）；$\varepsilon \sim \mathcal{N}(0, I)$；$x_t$：$t$ 时刻的路径点；$u_t$：条件速度场——直线路径下是与 $x_t$ 无关的常向量 $\varepsilon - x_0$。

infilling 版 CFM 损失：

$$
L_{\mathrm{F5}}(\theta) = \mathbb{E}_{t,\, x_0,\, \varepsilon,\, m}\left[ \left\| \left( v_{\theta}\left( x_t^{(m)},\, t,\, c \right) - (\varepsilon - x_0) \right) \odot m \right\|^2 \right] \qquad (20)
$$

**符号**：$m \in \{0,1\}^{T}$：挖空段指示（1 = 待生成），损失只在挖空段计；$x_t^{(m)}$：模型输入——挖空段取路径点 $x_t$、上下文段保留干净 $x_0$（infilling 的标准构造：未挖部分作为条件一并输入）；$c$：条件 = 式 (18) 的补齐文本（经 ConvNeXt V2 精炼——F5 相对 E2 的第一处增量）连同参考段信息；$v_\theta$：速度网络，DiT 主干（7.6 的骨干选择：序列条件生成 + 全并行前向，与逐帧 mask 结构天然相容）。

```
mel 帧轴  t = 1..T
┌────────────────────┬─────────────────────────────┐
│ 上下文段：干净 x0    │ 挖空段：x_t（噪声 → 数据）    │
│（作为条件输入）       │（损失只算这里，m = 1）        │
└────────────────────┴─────────────────────────────┘
文本轴：参考文本 chars + 目标文本 chars，filler 补齐至 T（式 (18)）
```

训练与推理的任务同构：训练 = 随机挖一段学回填；zero-shot 推理 = 把目标整句挖掉，条件给「参考音频 + 参考文本 + 目标文本」——音色从参考段经双向上下文条件流入。与 VALL-E 的对照：一个靠 AR 前缀延续（式 (16)），一个靠掩码上下文场（式 (20)）——「3 秒克隆」的两种机制实现。全 NAR 的延迟结构：生成一句 mel 的串行步数 = ODE 求积步数，与句长无关——式 (16) 的 $T$ 步 AR 税在这里整个不存在。

**Sway Sampling**（推理时策略；原文：「inference-time Sway Sampling strategy……can be easily applied to existing flow matching based models without retraining」）：ODE 求积的均匀步点 $t_i = i/N$ 换成 warped 步点 $\tau_i = g(t_i)$，其中 $g: [0,1] \to [0,1]$ 单调、$g(0) = 0$、$g(1) = 1$。机制账：第 $i$ 步的有效步长 $\Delta \tau_i \approx g'(\xi_i)/N$——$g'$ 小处步点密、局部截断误差小；$g'$ 大处步点稀。sway 形曲线把密度预算向某一端倾斜（端点固定保证积分区间不变），倾斜方向由一个 sway 系数控制。与 7.5 §8 训练侧 timestep shift 是同一思想搬到推理调度上：不改权重、只改求积节点，所以任何 FM 模型免重训即插——这正是原文强调的迁移性。方向性读法：数据端（$t \to 0$）附近轨迹决定细节纹理，噪声端（$t \to 1$）附近决定全局结构；往哪端倾斜是质量-步数权衡上的可调旋钮，最优方向随数据与骨干而变，属推理超参而非训练配方。

收束一行三条路线的任务观：infilling 把 TTS、编辑、去噪统一成同一个条件生成——mask 位置即任务指定符。与 10.2 §6.2 Whisper 的前缀 token 对照：一个用前缀指定任务，一个用 mask 几何指定任务，都是「单一模型、多任务」的便宜实现。

## 9. 其他路线两件与 2025 一段式

**NaturalSpeech 3 / FACodec**（[Ju et al., 2024](../references.md)，arXiv:2403.03100，[B]——微软未开源）：「factorized vector quantization (FVQ)……decompose speech waveform into distinct subspaces of content, prosody, timbre, and acoustic details」——10.1 §3.3 分离视角的完整产品形态：四个子流各配码本，生成侧各属性流各配一个 diffusion model。克隆 = 换 timbre 子流、保 content 流；可控性走到极端形态——内容、韵律、音色、声学细节各有独立旋钮；代价是四条生成链的复合复杂度。

**MaskGCT**（[Wang et al., 2024](../references.md)，arXiv:2409.00750，[A]——官方实现在 open-mmlab/Amphion）：声学 codec token 上的 mask-predict NAR（7.7 §7 的解码范式搬进语音：全部位点并行出价、按置信度逐轮兑现），两级结构：文本 → 语义 token → 声学 token。与 VALL-E 的对照直接：同样建 codec token，把「AR 第一层 + NAR 其余层」换成 mask-predict 全并行——式 (16) 的 $T$ 步串行税再降一档；代价是伪似然族的老问题（7.7 §7：迭代伪似然乘积不等于联合似然）。

**2025 批一段式**——从「管线」回归「一个模型」的工程化收束，四家各带一个标志性决策：

- **IndexTTS**（[Deng et al., 2025](../references.md)，arXiv:2502.05512，[A]；repo index-tts/index-tts）：「GPT-style」AR TTS；工程件落在中文场景——拼音纠错（多音字消歧）与标点控停。
- **IndexTTS2**（[Zhou et al., 2025](../references.md)，arXiv:2506.21619，[A]）：情感表达 + 时长可控 AR zero-shot——把 §2 时代 duration 旋钮的可控性请回 AR 路线（对照 §10 表的可控性行）。
- **MegaTTS3**（[Jiang et al., 2025](../references.md)，arXiv:2502.18924，[A]；repo bytedance/MegaTTS3）：「Sparse Alignment Enhanced Latent Diffusion Transformer」——稀疏显式对齐 + latent diffusion 的混合形态：对齐不全交给输入构造（对照 §8），留一个稀疏的显式监督。
- **Spark-TTS**（[Wang et al., 2025](../references.md)，arXiv:2503.01710，[A]；repo SparkAudio/Spark-TTS）：BiCodec 单流 codec——「a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes」：内容走低码率逐帧流、音色走定长全局 token（一句话一组），LLM 单流建模——10.1 §6.2 的 semantic/acoustic 分工在单流内的折中实现，音色控制又是一种「全局 token 条件」形态。

## 10. 三路线对照表

| 维度 | 级联（FastSpeech2 / VITS + §3 vocoder） | LM 系（VALL-E / CosyVoice） | FM/diffusion 系（VoiceBox / E2 / F5 / NS3） |
| --- | --- | --- | --- |
| 生成单元 | mel 帧 + 波形（vocoder 出口） | codec token（式 (15)） | mel（infilling，式 (20)） |
| 对齐来源 | 显式：duration model / forced alignment / MAS（式 (3)(13)） | LM 隐式学（前缀 + AR 延续） | filler 补齐构造（式 (18)） |
| 长度 $T$ 决策 | duration predictor（VITS 随机采样，多样韵律） | AR 学到自行终止 | 外生给定（按参考语速估计） |
| 音色克隆 | speaker embedding / d-vector | 3s prompt in-context（式 (16)） | infilling 上下文参考段（式 (20)） |
| 串行步数（每秒音频，教学账） | 各模块全并行；vocoder 帧级可流式 | AR $T$ 步 + 7 趟层间（75 fps 档 → 75+7；CV2 25 Hz → 25 步） | ODE $N$ 步 × 全句并行 |
| 流式 | vocoder 天然帧级流式 | chunk-aware causal FM（CV2 单模型双流式） | 整句 ODE，流式需切块改造 |
| 可控性 | 逐旋钮可人工覆盖（duration/pitch/energy） | prompt 工程 + 采样器补丁（VALL-E 2） | mask 几何 + 参考段；NS3 子流旋钮 |
| 典型失败模式 | 误差沿模块链累积、拼接感 | 重复坍缩/漏字（repetition aware sampling 的对象） | 长度外生错配（$T$ 给错全盘偏） |
| 本章证据形态 | HiFi-GAN/VITS [A]，FastSpeech2 [B] | VALL-E 系 [B]，CosyVoice 系 [A] | VoiceBox/E2/NS3 [B]，F5-TTS [A] |

读法两笔。其一，「音色克隆」列是同一需求的三种机制答案——模块嵌入、上下文条件、掩码条件——共同点是都不微调权重；这一格也是三条路线互相借件最多的地方（CosyVoice 2 把 speaker embedding 拆掉改走 prompt，正是 LM 列方案向 FM 解码器侧的渗透）。其二，「对齐」行是本章主线：显式监督（级联）→ 模型内隐式学（LM）→ 写进输入构造（infilling）——对齐这个潜变量的处理方向是逐层外包给数据与构造。音质数字的横向对比刻意缺席：各家的 MOS / 偏好评估口径互不兼容，且多家无官方代码可复现（[B] 行），本书只引各家摘要级主张。

## 总结

| 对象 | 公式 / 结论 | 位置 |
| --- | --- | --- |
| TTS 形式化 | $P(x \mid y)$，$T \gg U$；一对多 → 潜变量 $z$ 吸收说话方式 | 式 (1)(2) |
| 级联管线 | length regulator 式 (3)（$T = \sum d_u$）；variance adaptor 式 (4)（duration/pitch/energy） | §2 |
| HiFi-GAN MPD | 式 (5)：1D 按质数周期 $p \in \{2,3,5,7,11\}$ 折成 2D，周期性变竖直条纹 | §3 |
| feature matching | 式 (6)：判别器中间激活的 L1 匹配——对抗训练的自我稳定回路 | §3 |
| HiFi-GAN 总目标 | 式 (8)：对抗 + FM + mel 回归三件 | §3 |
| BigVGAN snake | 式 (9)：$x + \sin^2(\alpha x)/\alpha$；式 (10)：$f' = 1 + \sin(2\alpha x) \in [0,2]$；配低通抗混叠（消融量级 ×19） | §4 |
| VITS | 式 (11) 条件 ELBO；式 (12) flow 增强先验（训练用 $f^{-1}$、采样用 $f$）；式 (13) MAS 硬对齐（Viterbi 式 DP）；式 (14) 五项总目标 | §5 |
| VALL-E | 式 (15) 按层链式；式 (16) AR 第一层；式 (17) NAR 第 2–8 层层内并行；3s prompt = 前缀条件化 | §6 |
| VALL-E 2 [B] | repetition aware sampling（重复坍缩补丁）+ grouped code modeling（步长按组放大） | §6 |
| CosyVoice 系 | 监督语义 token（ASR encoder 内插 VQ；官方生态称 S3Tokenizer）；CV2：chunk-aware causal FM 双流式、FSQ、直用 Qwen2.5-0.5B、25 Hz；CV3：MinMo voice encoder | §7 |
| F5-TTS | 式 (18) filler 对齐；式 (19) 线性路径 $u_t = \varepsilon - x_0$；式 (20) infilling CFM；Sway Sampling = 免重训的求积步点 warping | §8 |
| 三路线对照 | 对齐来源 / 音色克隆 / 串行步数 / 流式 / 可控性 / 失败模式 | §10 表 |

## 练习

1.（推导）VALL-E 两阶段的依赖结构。(a) 把 $8T$ 个码按层主序摊平（先第 1 层全部 $T$ 帧、再第 2 层……），写出纯 AR 分解（每个码条件于摊平序下的全部前驱），给出其严格串行步数；(b) 证明式 (15)–(17) 的联合分布与 (a) 的摊平 AR 相比，恰好只丢掉了层内跨帧依赖 $\{P(c_{t,j} \mid c_{<t,j}, \cdot)\}_{j \ge 2}$，其余条件集相同（逐层核对链式展开的因子）；(c) 帧率 75 fps、$n_q = 8$、5 秒音频：摊平 AR 与两阶段的串行步数各是多少（KV cache 账按 2.4 口径，NAR 趟内视为一次并行前向），并指出两阶段账里被忽略的一项工程开销。

2.（推导）snake 激活。(a) 由式 (9) 推出式 (10) 并给出 $f'_\alpha$ 的值域；(b) 用 $\sin^2\theta = (1 - \cos 2\theta)/2$ 说明 $\sin^2$ 作用于 $\cos(\omega n)$ 产生倍频分量，进而给出 24 kHz 采样下 15 kHz 谐波的混叠频率（折叠公式：$f \mapsto |f - k f_s|$ 取落入 $[0, f_s/2]$ 者）；(c) $\alpha \to \infty$ 时扰动幅值 $1/\alpha \to 0$ 而振荡频率 $\to \infty$——说明此时 snake 退化成什么函数，并论证训练为何不会把 $\alpha$ 推向无穷（提示：高频周期分量的混叠与 §4 低通的对抗关系）。

3.（计算）MPD 折叠账。取 $T = 24000$（1 秒 @ 24 kHz）：(a) 按式 (5) 对 $p \in \{2, 3, 5, 7, 11\}$ 分别给出 $X_p$ 的 shape 与尾部需补齐的样本数；(b) 某浊音基频 120 Hz，其周期合多少样本？该周期结构能被哪些 $p$ 的折叠「整除对齐」地捕获、哪些不能（用整除关系论证）？(c) 若把质数周期换成合数周期 $\{2, 4, 8\}$，论证它比 $\{2, 3, 5\}$ 少覆盖哪一族周期结构（举出一个具体基频例子）。

4.（分析）三路线选型。给定三个产品需求——(i) 有声书批量合成（固定播音员、超长文本、要求逐句时长可预估）；(ii) 实时语音助手的零样本音色定制（用户录 3 秒即用）；(iii) 播客后期编辑（改几个字不重录整句）——从 §10 对照表的维度逐项论证各应选哪条路线（含具体系统举例），并指出每种选择放弃了哪一格。

## 参考

- [Kong et al., 2020](../references.md)：HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis，arXiv:2010.05646，[A]（key: hifi-gan）——式 (5) MPD 质数周期 $[2,3,5,7,11]$ 2D reshape、式 (6) feature matching、式 (7)(8) mel 损失与总目标
- [Lee et al., 2022](../references.md)：BigVGAN: A Universal Neural Vocoder with Large-Scale Training，arXiv:2206.04658，[A]（key: bigvgan；repo NVIDIA/BigVGAN）——式 (9)(10) snake 激活与导数、低通抗混叠（消融量级口径）
- [Kim et al., 2021](../references.md)：Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech（VITS），arXiv:2106.06103，[A]（key: vits）——式 (11)–(14)：VAE+flow 先验增强、MAS、stochastic duration predictor、对抗统一目标
- [Ren et al., 2020](../references.md)：FastSpeech 2: Fast and High-Quality End-to-End Text to Speech，arXiv:2006.04558，[B]（key: fastspeech2）——式 (3)(4)：variance adaptor（duration/pitch/energy）；微软未释官方代码
- [Wang et al., 2023](../references.md)：Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers（VALL-E），arXiv:2301.02111，[B]（key: vall-e）——式 (15)–(17) 两阶段分解与 3s prompt；微软未开源
- [Chen et al., 2024](../references.md)：VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers，arXiv:2406.05370，[B]（key: vall-e-2）——repetition aware sampling、grouped code modeling；未开源
- [Du et al., 2024](../references.md)：CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens，arXiv:2407.05407，[A]（key: cosyvoice）——监督语义 token（ASR encoder 内插 VQ）+ LLM + conditional FM decoder
- [Du et al., 2024](../references.md)：CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models，arXiv:2412.10117，[A]（key: cosyvoice2）——chunk-aware causal FM、FSQ、25 Hz、Qwen2.5-0.5B 主干
- [Du et al., 2025](../references.md)：CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training，arXiv:2505.17589，[A]（key: cosyvoice3）——FSQ 改插 MinMo voice encoder
- [Chen et al., 2024](../references.md)：F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching，arXiv:2410.06885，[A]（key: f5-tts）——式 (18)–(20)：filler 补齐、infilling CFM、ConvNeXt V2、Sway Sampling
- [Eskimez et al., 2024](../references.md)：E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS，arXiv:2406.18009，[B]（key: e2-tts）——逐字符 filler + FM mel + audio infilling 前驱形态；微软未开源
- [Le et al., 2023](../references.md)：Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale，arXiv:2306.15687，[B]（key: voicebox）——text-guided speech infilling + FM/OT path；Meta 未开源
- [Ju et al., 2024](../references.md)：NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models，arXiv:2403.03100，[B]（key: naturalspeech3）——FACodec FVQ 四子流分解 + 各流 diffusion；微软未开源
- [Wang et al., 2024](../references.md)：MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer，arXiv:2409.00750，[A]（key: maskgct）——声学 token 上 mask-predict NAR，语义→声学两级；实现在 Amphion
- [Deng et al., 2025](../references.md)：IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System，arXiv:2502.05512，[A]（key: indextts）——GPT 式 AR、拼音纠错、标点控停
- [Zhou et al., 2025](../references.md)：IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech，arXiv:2506.21619，[A]（key: indextts2）——情感表达 + 时长可控 AR
- [Jiang et al., 2025](../references.md)：MegaTTS 3: Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis，arXiv:2502.18924，[A]（key: megatts3）——稀疏对齐 + latent diffusion
- [Wang et al., 2025](../references.md)：Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens，arXiv:2503.01710，[A]（key: spark-tts）——BiCodec 单流：低码率语义 token + 定长全局 speaker token
- [jik876/hifi-gan](../references.md)：[A]（key: hifigan-repo）——models.py 文件级实测（MPD/MRF 走读锚点）；[jaywalnut310/vits](../references.md)：[A]（key: vits-repo，models.py）；[QwenAudio/CosyVoice](../references.md)：[A]（key: cosyvoice-repo，FunAudioLLM 301 重定向链，cosyvoice/{tokenizer, flow, llm, hifigan} 目录实测）；[SWivid/F5-TTS](../references.md)：[A]（key: f5-tts-repo，src/f5_tts 目录实测）；[open-mmlab/Amphion](../references.md)：[A]（key: amphion-repo，MaskGCT 官方实现所在）
