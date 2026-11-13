# 10.1 音频表征：特征与 Neural Codec（Audio Representations: Features and Neural Codecs）

> 音频进模型之前只有两种形态——连续谱特征（log-mel，100 fps 量级）与离散 token（semantic / acoustic 两类）；本章把前者（STFT/mel 公式链）与后者（HuBERT 系 semantic token、SoundStream 到 X-codec2 六家 neural codec）的公式与参数账一次结清，codec 六家的 fps / $n_q$ / $K$ / bitrate 对照表是 Part 10 其余三章的公共底座 | 前置：1.1（CE/NLL）、2.4（KV cache 账）、3.1（masked prediction/MLM）、6.2（InfoNCE 式 (2)-(5)）、6.3（VQ/RVQ 式 (20)(21)、离散 vs 连续选择表 §7.4）、8.1（VLM connector 同构） | 锚点：facebookresearch/encodec [A]、kyutai-labs/moshi（Mimi 实现 moshi/moshi/models/compression.py）[A]、descriptinc/descript-audio-codec [A]、facebookresearch/fairseq（HuBERT：examples/hubert）[A]、kyutai/mimi config [A]；SoundStream [B]、w2v-BERT [B]

记号约定（本 Part 统一）：波形 $x[n]$（离散时间采样，采样率 $f_s$）；STFT 谱 $X[m, k]$（$m$ 帧下标、$k$ 频点下标）；**帧率 fps 与 token rate 区分**——fps = codec/encoder 每秒帧数，token rate = fps × 每秒并发的 codebook 数（如 Mimi $8 \times 12.5 = 100$ tok/s）；RVQ 记号沿用 6.3——$r_1 = z$，$e_i = Q_i(r_i)$，$r_{i+1} = r_i - e_i$，codebook 数 $n_q$、大小 $K$。

## 1. 数字音频：采样率三档与 Nyquist 账

连续声波经均匀采样成为序列 $x[n]$，采样率 $f_s$ 决定可表示的频率上限（Nyquist）：

$$f_{\max} = f_s / 2 \qquad (1)$$

**符号**：$f_{\max}$：无混叠可表示的最高频率。三档常用采样率的账：16 kHz（语音识别档，上限 8 kHz）、24 kHz（语音合成与 neural codec 主流档，上限 12 kHz）、44.1 kHz（音乐档，上限 22.05 kHz）。语音信息集中在低频段，8 kHz 上限对识别与可懂度已富余；音乐的高频泛音才需要 44.1 kHz 的带宽。选 $f_s$ 就是选带宽预算：采样率直接乘进每秒样本数，进而乘进后续每个环节的算力与码率账。

原始 PCM 的码率账给出压缩任务的量级参照：24 kHz、16 bit、单声道波形是 $24000 \times 16 = 384$ kbps；本章 §4 的 SoundStream 在同采样率下做到 6 kbps（64 倍压缩），§5 的 X-codec2 在 16 kHz 下做到 0.8 kbps（对 $16000 \times 16 = 256$ kbps 的 PCM 是 320 倍）。neural codec 的设计空间就是在「bitrate × 重建质量 × token 结构」三个旋钮之间做交易——第三个旋钮是 LLM 时代才出现的，§5 与 §6 会看清它的分量。

## 2. STFT 与 mel：从波形到特征序列

### 2.1 STFT 完整公式

波形本身对模型不友好：采样点级时间序列（24 kHz → 每秒 24000 个点）既长又没有直接的频率可读性。短时傅里叶变换（STFT）把波形切成重叠的帧、逐帧做 DFT，得到时频联合表征：

$$X[m, k] = \sum_{n} x[n]\, w[n - mH]\, e^{-j 2\pi k n / N} \qquad (2)$$

**符号**：$w$：窗函数（长度 $N_w$ 个采样点）；$H$：hop（相邻帧起点间距，样本数）；$N$：DFT 点数（$N \ge N_w$）；$k = 0, \ldots, N/2$ 为有效单边频点。$X[m, k]$ 是复数，幅度谱 $|X[m, k]|$ 与功率谱 $|X[m, k]|^2$ 是下游特征的原料。

hop 决定帧率：$fps = f_s / H$。标准语音档 16 kHz / 25 ms 窗 / 10 ms hop：$N_w = 0.025 \times 16000 = 400$，$H = 160$，帧率 $= 16000/160 = 100$ fps。窗大于 hop（$N_w > H$）时相邻帧共享一段波形，帧间相位连续可读。

式 (2) 的输出是复数，下游特征链只取幅度：相位谱 $\angle X[m, k]$ 逐帧逐频点摆动剧烈、没有幅度谱那样的可读结构，建模价值低；而它丢掉的信息（精细定时）在感知上本就弱。代价是单向的——log-mel 这条路走不到波形重建，回波形要靠 codec 的 decoder（§4）或 vocoder 从幅度谱反推相位（10.3 的 HiFi-GAN 系）。理解与压缩两条路线在「相位」这个岔口分开。

### 2.2 时频不确定性：一个旋钮管两个分辨率

窗长 $N_w$ 同时钉死两个方向的分辨率：频率分辨率 $\Delta f \sim f_s / N_w$（DFT 主瓣宽度量级），时间分辨率 $\Delta t \sim N_w / f_s$（窗内事件不可区分）。两者乘积是常数：

$$\Delta t \cdot \Delta f \sim \frac{N_w}{f_s} \cdot \frac{f_s}{N_w} = 1 \qquad (3)$$

**符号**：$\sim$ 表量级关系（常数因子由窗形决定）。25 ms @ 16 kHz：$\Delta f \approx 40$ Hz、$\Delta t \approx 25$ ms。频率分辨率要提一倍，窗就得加长一倍、时间分辨率同倍率让出——不存在「两个方向都更细」的窗长选择。语音取 25 ms 的理由在信号特性：音素持续数十毫秒量级，25 ms 窗内谱近似平稳（短时平稳假设），而 40 Hz 的频率分辨率足以读清谱包络。

### 2.3 mel filterbank 与 log 压缩

STFT 的 $N/2 + 1$ 个频点在线性频率轴上等间距，与听感的非线性不匹配。mel 刻度把频率轴按感知重排，两种常用定义点名：HTK 式

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right) \qquad (4)$$

与 Slaney 式（另一套常用刻度，差异在分段形式与归一化，不展开推导）。mel filterbank 是一组在 mel 轴上等距、换算回 Hz 轴后低频密高频疏的三角滤波器，写成矩阵 $M \in \mathbb{R}^{K \times (N/2 + 1)}$（$K$ = mel 通道数），作用在功率谱上再加 log：

$$S_{\text{mel}}[m, :] = \log\left(M \cdot |X[m, :]|^2\right) \qquad (5)$$

**符号**：$S_{\text{mel}}[m, :] \in \mathbb{R}^{K}$：第 $m$ 帧的 log-mel 特征向量。两步各有分工：mel 重排把表征密度移到感知敏感的低频段；log 压缩把声强的乘法关系变加法（Weber-Fechner 式的感知对数律），同时压缩动态范围，数值尺度更合优化的胃口。

实例锚定：Whisper 前端就是这条链的标准取值——16 kHz 重采样、25 ms 窗 / 10 ms hop、80 通道 log-mel，帧率 100 fps（[Radford et al., 2022](../references.md)，arXiv:2212.04356，[A]）；10.2 的 Whisper 走读直接沿用这套参数。

### 2.4 特征账：100 fps 的序列长度问题

| 模态 | 单位 | 速率 | 10 秒对应序列长度 |
| --- | --- | --- | --- |
| 波形（16 kHz） | 采样点 | 16000 /s | 160,000 |
| log-mel（100 fps） | 帧 × $K$ 维向量 | 100 /s | 1,000 帧 |
| codec token（SoundStream 档） | $n_q$ 个码/帧 | $8 \times 75 = 600$ tok/s | 6,000 token |
| codec token（Mimi 档） | 同上 | $8 \times 12.5 = 100$ tok/s | 1,000 token |
| 文本（对照） | token | $\sim$ 10 /s 量级 | $\sim$ 100 |

这张表是 Part 10 后半程两个难点的根。其一，10.2 的 ASR 要把 100 fps 的声学序列映到 $\sim$ 10 fps 的文本序列——长度差一个数量级且没有逐帧对齐标注，CTC 的对齐路径边缘化就是为这个 mismatch 准备的。其二，10.3/10.4 要让 LM 逐帧建模 codec token，token rate 直接乘进序列长度与 KV cache 账——600 tok/s 与 100 tok/s 之间是 6 倍成本差，这正是 §5 各家压 fps、减码本数的动机。

## 3. Semantic tokens：自监督表征的离散化

连续特征之外的另一条路是把语音变成离散 id 序列，直接复用文本侧的 softmax 词表机制（1.1 的 CE 框架原样生效）。semantic token 路线的目标不是压缩重建，而是把「这段语音说了什么」抽成离散标签——波形细节、音色、信道噪声是要被抹掉的干扰项。

### 3.1 HuBERT：k-means 教师 + masked prediction 迭代

HuBERT（[Hsu et al., 2021](../references.md)，arXiv:2106.07447，[A]；官方实现在 fairseq 的 examples/hubert，[fairseq-repo](../references.md)，[A]）用聚类标签自举出监督信号。第 $b$ 轮迭代：对当前表征做 k-means 得到逐帧离散标签 $z^{(b)}$，mask 掉一部分帧，训练模型预测被 mask 位置的标签：

$$L_{\text{MP}} = - \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_{\theta}\left(z_t \mid \tilde{x}\right) \qquad (6)$$

**符号**：$\mathcal{M}$：被 mask 的帧集合；$\tilde{x}$：加 mask 后的输入；$z_t$：第 $t$ 帧的簇标签；$P_\theta$：模型输出的簇分布。式 (6) 与 3.1 式 (4) 的 MLM 同构——把「词表」换成「k-means 簇表」，掩码预测机制照搬。符号约定一处：HuBERT 原文按 log-likelihood 最大化写法（不带负号、不归一化），本书从 1.1 起 CE/NLL 一律取最小化口径，式 (6) 据此取负并按 $|\mathcal{M}|$ 归一——两种写法差一个符号约定，优化方向相同。迭代闭环：本轮训出的隐层表征比上一轮更干净，用它重新聚类得到更好的标签再训下一轮。论文口径：起步教师是一个 100 簇的 k-means，共做两轮聚类迭代。

### 3.2 w2v-BERT：对比 + masked 两段式

w2v-BERT（[Chung et al., 2021](../references.md)，arXiv:2108.06209，[B]——Google 未释官方实现）把离散化与上下文学习拆成两个目标：对比模块先把连续表征量化成离散 token（对比目标一族的机制推导见 6.2 式 (2)-(5) 的 InfoNCE），主目标再对这些离散 token 做 masked language modeling。原文口径即「combines contrastive learning and masked language modeling」：对比项负责训练码本候选的分离度（原文配方中该项缺席会导致表征塌缩），MLM 项使表征带上上下文语义。

### 3.3 content 与 speaker 的分离视角

semantic token 想要 content（语音学内容），自监督标签里却混着说话人、韵律、声学环境。FACodec 系（NaturalSpeech 3，[Ju et al., 2024](../references.md)，arXiv:2403.03100，[B]）把这个问题摆到架构层：用 factorized vector quantization 把语音显式分解成 content / prosody / timbre / acoustic details 四个子空间、各配独立码本——「哪些算语义」不再是训练目标的隐式期望，而是码本级的分工。10.3 会看到这条分解线在 TTS 里的直接用途：音色克隆只换 timbre 子流，content 流保持文本对齐。

semantic token 的共性收一笔：bitrate 极低（单码本、低帧率），但没有重建目标、不配 decoder——它们不是压缩格式，不能解码回波形；衡量标准是下游识别/理解性能，不是保真度。

## 4. Acoustic tokens I：SoundStream / EnCodec / DAC

acoustic token（neural codec）的目标相反：把波形**可逆地**压成离散 token，decoder 要重建出人耳难辨的音频。骨架三家共用：

```
x[n]（24 kHz 波形）                            x̂[n]（重建波形）
  │                                               ▲
  │ Encoder：stride 卷积级联                       │ Decoder：转置卷积镜像
  │ 24000 → 75 Hz（总 stride 320，SoundStream 档）  │
  ▼                                               │
z ∈ R^{T'×d}（latent，fps = 75）                  │
  │                                               │
  │ RVQ（6.3 式 (20)）：r₁=z, eᵢ=Qᵢ(rᵢ),           │ ẑ = Σᵢ eᵢ
  │ rᵢ₊₁=rᵢ−eᵢ，共 n_q 级                          │
  ▼                                               │
每帧 n_q 个码 (k₁,…,k_{n_q}) ─────────────────────┘
训练期另挂判别器（multi-scale STFT 系）作用于 x̂
```

量化与码率的数学在 6.3 §5 已推完：残差递推 $r_{i+1} = r_i - e_i$（6.3 式 (20)）、有效词表 $K^{n_q}$、以及本章反复套用的 bitrate 公式（6.3 式 (21)，$f_{\text{rate}}$ 换成音频帧率 fps）：

$$b = n_q \cdot \log_2 K \cdot fps \qquad (7)$$

**符号**：$b$：码率（bit/s）。式 (7) 不是本书的通用化——SoundStream 原文即此式（「The number of bits used to encode a single frame is equal to $N_q \log_2 N$」，乘帧率即码率）。RVQ 的分级结构顺带给出码率伸缩机制：只传前 $m < n_q$ 级即得低码率版本，不必重训。

### 4.1 SoundStream：骨架与 6 kbps 实例账

SoundStream（[Zeghidour et al., 2021](../references.md)，arXiv:2107.03312，[B]——Google 未开源官方代码，数字引论文原文）确立了上面的骨架：24 kHz 输入，encoder 各级 stride 乘积 320 → latent 帧率 $24000 / 320 = 75$ Hz（原文：「each second of audio at sampling rate $f_s = 24000$ Hz is represented by $S = 75$ frames」）。式 (7) 的原文实例账：6000 bps 档每帧 80 bit，$n_q = 8$ 个量化器、各配 $2^{80/8} = 1024$ 码本：

$$8 \times \log_2 1024 \times 75 = 8 \times 10 \times 75 = 6000 \text{ bps} = 6 \text{ kbps} \qquad (8)$$

可变码率靠 quantizer dropout 训练：训练时随机丢弃部分量化器级，推理时按预算选级数，覆盖 3–18 kbps。机制读法：随机丢级等价于同时在所有前缀截断档上训练同一组码本，第 $i$ 级学到的是「前 $i-1$ 级残差」的条件分布，任意前 $m$ 级截取都是训练见过的配置——码率伸缩因此不需要重训，这是 RVQ 分级结构白送的性质（6.3 §5）。论文的对照结论：3 kbps 的 SoundStream 优于 12 kbps 的 Opus、逼近 9.6 kbps 的 EVS——neural codec 相对传统信号处理 codec 的代差证据。

### 4.2 EnCodec：挡位账与 multi-scale STFT 判别器

EnCodec（[Défossez et al., 2022](../references.md)，arXiv:2210.13438，[A]；官方代码 [encodec-repo](../references.md)，[A]）同骨架，带宽旋钮摆在码本数上：24 kHz 模型每秒 75 个 latent step（48 kHz 模型 150 个），最多 32 个 codebook（48 kHz 模型 16 个）、每个 1024 entry（10 bit）。带宽挡位 = 码本数 2/4/8/16/32，代入式 (7)：$n_q \times 10 \times 75$ = 1.5 / 3 / 6 / 12 / 24 kbps，五挡逐项可复算。

重建质量由判别器顶住。multi-scale STFT discriminator 是一族判别器 $\{D_s\}$，各自作用在不同窗长的 STFT 幅度谱上，对抗项按尺度求和：

$$L_{\text{adv}} = \sum_{s=1}^{5} \ell\left(D_s\left(\left|\mathrm{STFT}_{N_s}(x)\right|\right),\; D_s\left(\left|\mathrm{STFT}_{N_s}(\hat{x})\right|\right)\right), \qquad N_s \in \{2048, 1024, 512, 256, 128\} \qquad (9)$$

**符号**：$\mathrm{STFT}_{N_s}$：窗长 $N_s$ 的 STFT；$\ell$：具体 GAN 损失形式（各家实现有差异，机制不依赖其选择）。式 (9) 与式 (3) 连起来读：单个判别器只钉一对 $(\Delta t, \Delta f)$，五个窗长就是五对互补分辨率——长窗抓谐波与音高结构（频率细），短窗抓瞬态与爆破音（时间细）——重建谱在时频两维同时被盯住。另有一条可选的 Transformer LM 熵编码支路（对码流再压一档），此处一笔带过。

### 4.3 DAC：factorized codes 与 dropout 修正

DAC（[Kumar et al., 2023](../references.md)，arXiv:2306.06546，[A]；官方代码 [dac-repo](../references.md)，[A]）在同一骨架上改三处。其一，factorized codes：码的查找在低维空间（8d 或 32d）进行、码的 embedding 驻留高维（1024d）——查找维与表达维解耦，大码本不再要求 latent 同宽；配 L2-normalized codes 把量化几何钉到单位球面。其二，指出 quantizer dropout 的副作用（伤满带宽档的重建质量）并换用改进版训练。其三，把采样率推到音乐档：主模型 44.1 kHz、总 stride 512 → 帧率 $44100 / 512 \approx 86$ Hz、9 个 10-bit 码本：

$$9 \times 10 \times 86 = 7740 \text{ bps} \approx 8 \text{ kbps} \qquad (10)$$

（原文目标档 8 kbps）。重建侧配 multi-scale mel loss，窗长从 32 到 2048 逐级加倍。

## 5. Acoustic tokens II：低帧率与单码本——Mimi / WavTokenizer / X-codec2

§4 三家的 fps 在 75–86、$n_q$ 在 8–32，token rate 600 tok/s 起步。codec token 要进 LM 逐帧自回归（10.3/10.4 的用法），token rate 直接乘进序列长度与 KV cache 账——降 fps、减码本数由此成为主攻方向。

### 5.1 Mimi：split-RVQ 与语义蒸馏（Moshi 的 codec）

Mimi（Moshi 论文 §3.3，[Défossez et al., 2024](../references.md)，arXiv:2410.00037，[A]；HF config 实测 codebook_size 2048 / frame_rate 12.5 / hidden_size 512，[mimi-config](../references.md)，[A]；实现走读锚点 [moshi-repo](../references.md) 的 `MimiModel`）把帧率砍到 12.5 Hz（24 kHz 输入，总下采样比 $24000 / 12.5 = 1920$）——§4 三家的约六分之一。

结构上它不是一条 8 层 RVQ 串行链，而是 split-RVQ：一个普通 VQ 专管语义、一个 7 层 RVQ 专管声学，两路并联、输出求和（原文：「Rather than a single RVQ with 8 levels, we distill semantic information into a plain VQ and apply an RVQ with 7 levels in parallel. We sum their outputs」）：

$$\hat{z} = e_{\text{sem}} + \sum_{i=1}^{7} e_i, \qquad e_{\text{sem}} = Q_{\text{sem}}(z) \qquad (11)$$

**符号**：$Q_{\text{sem}}$：语义 VQ（plain VQ，单级）；$e_i$：声学 RVQ 第 $i$ 级输出（级内残差递推同 6.3 式 (20)）。

```
latent z（12.5 fps）
  ├─ 语义 VQ：e_sem = Q_sem(z)
  │    └─ 蒸馏目标：cos 距离对齐 WavLM 表征（式 (12)）
  └─ 声学 RVQ ×7：逐级残差 e_1..e_7
  ẑ = e_sem + Σᵢ eᵢ（两路求和 → decoder）
```

语义支路的训练信号是对 WavLM 表征的蒸馏——余弦距离口径：

$$L_{\text{distill}} = \sum_t \left(1 - \cos\left(g(e_{\text{sem}, t}),\, u_t\right)\right) \qquad (12)$$

**符号**：$u_t$：WavLM 在第 $t$ 帧的 embedding（50 Hz 输出经 stride 4 average pooling 对齐到 12.5 Hz，non-causal pooling 是原文强调的细节）；$g$：可学线性投影（原文口径：作用于量化器输出侧、投到 1024 d——「a linear projection with an output dimension of 1024 to the output of the first RVQ level」）；$e_{\text{sem}, t}$：语义量化器输出。拆开的原因是目标冲突（原文：蒸馏显著提升第一量化器的音素可分性，但「it also affects audio quality negatively」）——把蒸馏压进同一条 RVQ 链的第 1 级，语义目标会拖着整条链牺牲音质；split 之后蒸馏只作用在 $e_{\text{sem}}$ 一路，声学 RVQ 专心补重建。这正是 §3.3 的 content/speaker 分离视角在 codec 内部的重演。

bitrate 账（原文实例）：$n_q = 8$（1 语义 + 7 声学）、$K = 2048$（11 bit）、12.5 Hz：

$$8 \times \log_2 2048 \times 12.5 = 8 \times 11 \times 12.5 = 1100 \text{ bps} = 1.1 \text{ kbps} \qquad (13)$$

token rate $= 8 \times 12.5 = 100$ tok/s——LM 侧成本是 §4 档的六分之一。帧率的另一面是延迟粒度：12.5 Hz 意味着每帧 80 ms，LM 的输出节奏被钉在这个栅格上——10.4 的 Moshi 全双工数学（含端到端延迟预算）全部建立在 12.5 Hz / 100 tok/s 这组数字上。

### 5.2 WavTokenizer：单码本极压缩

WavTokenizer（[Ji et al., 2024](../references.md)，arXiv:2408.16532，[A]）把结构简化到底：24 kHz 音频每秒只要单个量化器的 75（极端档 40）个 token。单码本 $K = 4096$（论文码本表中该行利用率 100%；另有 1024（亦 100% 利用率）/ 8192 / 16384 三档）。按式 (7) 现推：$\log_2 4096 = 12$ bit，$1 \times 12 \times 75 = 900$ bps，40 fps 档 $480$ bps。单码本顺手消掉了 RVQ 的层级结构——LM 侧不必处理「一帧多码」的层间依赖，这个依赖在生成侧的代价 10.4 会细算。

### 5.3 X-codec2：语义-声学统一单码本

X-codec2 出自 Llasa 论文（[Ye et al., 2025](../references.md)，arXiv:2502.04128，[A]——该 ID 的论文标题是 *Llasa*，X-codec2 是其中提出的 speech tokenizer，引用时勿张冠李戴）：延续 X-codec 的思路，把语义与声学特征融合进同一个码本（原文：「fuse these semantic and acoustic features into a unified codebook」）——§3 与 §4 两条路线在同一个码本里合流。配置：16 kHz、总下采样比 320 → 帧率 $16000 / 320 = 50$ fps、单码本 $K = 65536$（16 bit）：

$$1 \times 16 \times 50 = 800 \text{ bps} = 0.8 \text{ kbps} \qquad (14)$$

## 6. 对照与选择：六家 codec、两类 token、进 LM 的形态

### 6.1 六家 neural codec 对照表

| codec | 采样率 | fps | $n_q$ | $K$ | bitrate（式 (7) 复算） | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| SoundStream | 24 kHz | 75 | 8（dropout 可变） | 1024 | 6 kbps（3–18 可变） | 式 (8)，[B] |
| EnCodec | 24 kHz | 75 | 2–32（五挡） | 1024 | 1.5–24 kbps | §4.2 挡位账，[A] |
| DAC | 44.1 kHz | 86 | 9 | 1024 | 7.75 kbps（目标档 8） | 式 (10)，[A] |
| Mimi | 24 kHz | 12.5 | 1+7（split） | 2048 | 1.1 kbps | 式 (13)，[A] |
| WavTokenizer | 24 kHz | 75（40） | 1 | 4096 | 0.9（0.48）kbps，现推 | §5.2，[A] |
| X-codec2 | 16 kHz | 50 | 1 | 65536 | 0.8 kbps | 式 (14)，[A] |

读表三条：其一，三个旋钮各有分工——fps 定 LM 侧序列长度，$n_q$ 定每帧的结构复杂度，$K$ 定单码容量；其二，六家 bitrate 全部落在式 (7) 的取值网格上，报码率不报 $(n_q, K, fps)$ 的论文无从复算，本表只收可复算的行；其三，75+ fps 与 12.5 fps 之间是设计目标的代差——Mimi 之后「codec 为 LM 服务」成为显式目标，低 token rate 优先于保真冗余。

### 6.2 semantic vs acoustic 对照表

| 维度 | semantic tokens（HuBERT 系） | acoustic tokens（neural codec） |
| --- | --- | --- |
| 训练目标 | masked prediction / 对比（自监督） | 重建 + 对抗（压缩保真） |
| 语音学内容 | 高（目标即此） | 隐式携带；纯声学码本不保证音素可分性 |
| 音色/声学细节 | 被聚类与 mask 目标抹掉 | 保留（重建要求充分统计量） |
| 可解码回波形 | 不可（无 decoder） | 可（本职） |
| bitrate | 极低（单码本、低帧率） | 0.5–24 kbps 全谱系 |
| LM 建模难度 | token 语义干净、分布近文本 | 多码本 × 高 fps，序列长且带层级依赖 |

混合设计是两条线的收敛点：Mimi 把 semantic 一层嵌进 acoustic 框架（split-RVQ），X-codec2 融合统一码本，FACodec 显式分解子空间——三种工程回答对应同一个判断：纯 acoustic token 对 LM 不干净，纯 semantic token 丢了可解码性。

### 6.3 离散 token vs 连续特征进 LM（10.4 伏笔）

语音进 LLM 的最后一道形态选择，两条路的账分开记：

- **离散 codec token**：直接进词表，复用 next-token CE 与文本侧全部基建（采样、KV cache 原样生效）；代价是量化瓶颈——信息上限被 $n_q \log_2 K \cdot fps$ 钉死（与 8.5 式 (4) 同一笔账），多码本展开的层级结构在生成侧另有一笔结构成本。
- **连续特征（log-mel / encoder latent）**：经 adapter 投影进 LLM 的 embedding 空间（与 8.1 的 VLM connector 同构）；无硬性信息上限，但连续到离散的表征对齐全靠训练学出。且理解侧成立不等于生成侧可行——LLM 的 softmax 输出头吐不出连续向量，生成方向仍要回到离散 codec 或扩散头。

这个选择的各家答卷（全离散、混合、闭源方案）是 10.4 的主题。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 采样账 | 式 (1)：$f_{\max} = f_s / 2$；24k/16bit 单声道 PCM = 384 kbps | 三档 16k/24k/44.1k；codec 6 kbps = 64× 压缩 |
| STFT | 式 (2)；帧率 $f_s / H$；16k/25ms/10ms → 100 fps | 式 (3) 时频不确定性：$\Delta t \cdot \Delta f \sim 1$，一个旋钮管两个分辨率 |
| log-mel | 式 (4)(5)：HTK 刻度 + 三角滤波器组 + log | Whisper 档 80 通道 100 fps；100 fps vs 文本 ~10 fps = CTC 的动机 |
| RVQ bitrate | 式 (7)：$b = n_q \log_2 K \cdot fps$（6.3 式 (21) 复用） | 六家对照表全部落在此式网格上 |
| semantic tokens | 式 (6)：HuBERT 两轮 k-means + masked prediction | 抓内容、弃细节、不可解码；w2v-BERT = 对比 + MLM 两段式 |
| SoundStream 系 | 式 (8)：8×10×75 = 6 kbps；dropout 覆盖 3–18 kbps | 骨架：conv encoder（stride 320 → 75 Hz）+ RVQ + 判别器 |
| EnCodec / DAC | 五挡 1.5–24 kbps；式 (10)：9×10×86 ≈ 8 kbps @ 44.1k | 式 (9) multi-scale STFT 判别器 5 窗长；DAC factorized codes |
| 低帧率系 | 式 (13)：Mimi 8×11×12.5 = 1.1 kbps；式 (14)：X-codec2 0.8 kbps | split-RVQ = 1 语义 VQ（WavLM 余弦蒸馏，式 (12)）+ 7 声学 RVQ 并联；单码本两例 |
| 进 LM 形态 | token rate = fps × $n_q$（Mimi 100 tok/s） | 离散复用 softmax 基建但有信息上限；连续无硬上限但生成侧要回 codec/扩散——10.4 |

## 练习

1.（计算）bitrate 账逐项复算。(a) 用式 (7) 复算 §6.1 表六家的 bitrate 列，验证与原文值/现推值一致；(b) SoundStream 的 3–18 kbps 可变范围在 $K = 1024$、75 fps 下对应 $n_q$ 的取值范围；(c) EnCodec 五挡 1.5/3/6/12/24 kbps 各对应的码本数；(d) 若 Mimi 把码本从 2048 加到 4096、其余不动，bitrate 与 token rate 各变为多少（注意区分两者）。
2.（计算）STFT 特征账。44.1 kHz 音乐、40 ms 窗、20 ms hop：(a) $N_w$、$H$、帧率、$\Delta f$ 各是多少；(b) 1 分钟音频的 log-mel 序列多长；(c) 同一段音频过 DAC（86 fps、$n_q = 9$），LM 侧每秒要建模多少 token，与 (b) 的帧率口径对比。
3.（推导）时频不确定性。从 $\Delta f \sim f_s / N_w$ 与窗持续 $N_w / f_s$ 出发：(a) 说明为什么 16 kHz 下要 5 Hz 频率分辨率就必须放弃 10 ms 级时间分辨率（给出所需窗长）；(b) 算式 (9) 的 5 个窗长 [2048, 1024, 512, 256, 128] 在 24 kHz 下各自的 $\Delta f$ 与 $\Delta t$ 量级，说明多分辨率判别各盯什么现象。
4.（分析）token 形态选择。给定三个任务——(i) ASR 前端特征、(ii) 零样本 TTS 音色克隆的表征、(iii) 全双工 speech LM 的生成单元——从 §6.2 对照表的维度逐项论证各应选哪类 token，并指出每种选择放弃了什么。

## 参考

- [Zeghidour et al., 2021](../references.md)：SoundStream: An End-to-End Neural Audio Codec，arXiv:2107.03312，[B]（key: soundstream）——75 Hz @ 24 kHz（stride 乘积 320）、式 (8) 的 6 kbps 实例账、quantizer dropout 3–18 kbps、3 kbps 优于 Opus 12 kbps；Google 未开源
- [Défossez et al., 2022](../references.md)：High Fidelity Neural Audio Compression，arXiv:2210.13438，[A]（key: encodec）——75 fps @ 24 kHz（48 kHz 模型 150 fps）、≤32 × 1024 码本、五挡 1.5–24 kbps、multi-scale STFT 判别器（窗长 [2048, 1024, 512, 256, 128]）
- [Kumar et al., 2023](../references.md)：High-Fidelity Audio Compression with Improved RVQGAN，arXiv:2306.06546，[A]（key: dac）——factorized + L2-normalized codes（8d/32d 查找、1024d embedding）、dropout 副作用修正、44.1 kHz / 86 Hz / 9 × 10 bit ≈ 8 kbps
- [Défossez et al., 2024](../references.md)：Moshi: a speech-text foundation model for real-time dialogue，arXiv:2410.00037，[A]（key: moshi）——§3.3 Mimi：12.5 Hz、split-RVQ（1 语义 VQ + 7 声学 RVQ 并联求和，式 (11)）、WavLM 余弦蒸馏（式 (12)）、8 × 2048 @ 12.5 Hz = 1.1 kbps（式 (13)）
- [Hsu et al., 2021](../references.md)：HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units，arXiv:2106.07447，[A]（key: hubert）——100 簇 k-means 起步教师、两轮聚类迭代、masked prediction（式 (6) 口径）
- [Chung et al., 2021](../references.md)：W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training，arXiv:2108.06209，[B]（key: w2v-bert）——对比产离散 token + MLM 主目标；Google 未释官方实现
- [Ji et al., 2024](../references.md)：WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling，arXiv:2408.16532，[A]（key: wavtokenizer）——单码本 $K = 4096$（利用率 100%）、24 kHz 每秒 75（40）token
- [Ye et al., 2025](../references.md)：Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis（X-codec2 出处），arXiv:2502.04128，[A]（key: xcodec2）——语义-声学统一单码本、16 kHz / 320 下采样 = 50 fps、$K = 65536$ → 0.8 kbps（式 (14)）
- [Radford et al., 2022](../references.md)：Robust Speech Recognition via Large-Scale Weak Supervision（Whisper），arXiv:2212.04356，[A]（key: whisper）——16 kHz / 25 ms / 10 ms / 80 通道 log-mel 标准档实例
- [Ju et al., 2024](../references.md)：NaturalSpeech 3，arXiv:2403.03100，[B]（key: naturalspeech3）——FACodec 的 content / prosody / timbre / acoustic details 子空间分解视角
- [facebookresearch/encodec](../references.md)：[A]（key: encodec-repo）；[kyutai-labs/moshi](../references.md)：[A]（key: moshi-repo，moshi/moshi/models/compression.py 的 `MimiModel`）；[descriptinc/descript-audio-codec](../references.md)：[A]（key: dac-repo）；[facebookresearch/fairseq](../references.md)：[A]（key: fairseq-repo，examples/hubert）；[kyutai/mimi config](../references.md)：[A]（key: mimi-config）——codebook_size 2048、frame_rate 12.5、hidden_size 512
