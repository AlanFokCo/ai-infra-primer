# 7.6 Latent Diffusion、DiT 与 MMDiT（Latent Diffusion and Diffusion Transformers）

> 本章回答「扩散模型的主干网络长在什么空间、长什么样」：latent 空间的 token 账（f8 位点削减 64×）与两阶段分工（VAE = 感知压缩、diffusion = 语义生成）、scaling factor 0.18215 的单位方差化数学（错误尺度的代价 = log-SNR 轴整体平移）、conditioning 三机制（cross-attention / concat / in-context）对比、text encoder 从 CLIP 到 T5 的选择依据（Imagen/SD3/PixArt 消融）、LDM 世代 U-Net block 走读（ResBlock 的 time embedding 注入、spatial transformer）；随后换主干：DiT 的 patchify 账（p=2/4/8）、**adaLN-Zero 推导**（gate 零初始化 = 每个 block 以 identity 起步）、DiT scaling 账（Gflops ↔ FID 对数线性）；MMDiT（SD3）双流各自 QKV + **联合 attention 矩阵**公式走读、QK-norm 的 logit 上界推导、FLUX double→single 变体 [C]、multi-aspect bucketing；效率线收尾：PixArt-α 三阶段训练与 adaLN-single、**SANA = Linear DiT + DC-AE f32**（linear attention 进扩散，呼应 4.3）| 前置：6.3（VAE/DC-AE、scaling factor 式 (24)）、7.2（DDPM 前向闭式与 $L_{simple}$）、7.5（rectified flow 目标式 (22)、logit-normal 式 (27)）、4.3（linear attention 结合律账式 (4)）、6.2（CLIP 训练机制）| 锚点：LDM [A]、DiT [A]、SD3 [A]、PixArt-α [A]、PixArt-Σ [A]、SANA [A]；CompVis/latent-diffusion [A]、facebookresearch/DiT [A]、Stability-AI/sd3.5 [A]、Stability-AI/generative-models [A]、black-forest-labs/flux [C]、NVlabs/Sana [A]、PixArt-alpha/PixArt-alpha [A]

## 1. 为什么不在像素空间扩散：latent 的账与两阶段分工

先摆一笔账。一张 $512 \times 512$ 的 RGB 图是 $786432$ 个标量、 $262144$ 个空间位点。扩散模型的每一步去噪都要在这张网格上做一次完整前向：conv 主干的 FLOPs 与位点数 $P$ 成正比，attention 与 $P^2$ 成正比；采样 $T$ 步（或 $K$ 步 solver）就把这笔账乘上步数。在像素空间直接做 $512^2$ 扩散， $P = 262144$，attention 的配对数是 $6.9 \times 10^{10}$——这就是为什么像素空间的扩散 U-Net（DDPM、ADM）只敢在 $32^2$ 以下的低分辨率级插 attention。

LDM（latent diffusion model，[Rombach et al., 2022](../references.md)，arXiv:2112.10752，[A]）的做法是把扩散的工作空间整个换掉：先用 6.3 训好的 autoencoder 把图像压成 latent，扩散只在 latent 空间进行。设 encoder $E$、decoder $D$、空间压缩率 $f = H/h$（全书约定，6.3 §1）：

$$
z = E(x) \in \mathbb{R}^{h \times w \times c}, \qquad \hat{x} = D(z), \qquad f = H/h = W/w \qquad (1)
$$

**符号**： $x \in \mathbb{R}^{H \times W \times 3}$：像素图像； $z$：latent； $c$：latent 通道数（SD 1.x 为 4，SD3/FLUX 为 16）。

位点数从 $HW$ 降到 $HW/f^2$：f8 档下削减恰为 $f^2 = 64$ 倍——这就是「64× 削减」的出处。同一张 $512^2$ 图，各档的账：

| 工作空间 | 位点数 $P$ | 标量数（ $c$ 通道） | patchify $p=2$ 后 token 数 $N = P/4$ | attention 配对数 $P^2$ 相对值 |
| --- | --- | --- | --- | --- |
| 像素（f1） | $512^2 = 262144$ | $786432$（ $c=3$） | $65536$ | $4096\times$ |
| f4 latent | $128^2 = 16384$ | $65536$（ $c=4$） | $4096$ | $256\times$ |
| f8 latent | $64^2 = 4096$ | $16384$（ $c=4$） | $1024$ | $1\times$ |
| f16 latent | $32^2 = 1024$ | — | $256$ | $1/16\times$ |
| f32 latent | $16^2 = 256$ | $8192$（ $c=32$，DC-AE） | $64$ | $1/256\times$ |

两个方向的收益同时成立：(i) **训练与采样的每步成本**按 $f^2$（conv 部分）到 $f^4$（attention 部分）下降；(ii) latent 已经够小，**全分辨率堆 attention 变得可行**——像素空间做不到的全程全局感受野，在 $64^2$ 的网格上是默认配置。autoencoder 的训练是一次性开销，训好后被任意多轮扩散实验复用（摊销），解码只在采样结束时跑一次。

把这笔账挂到 7.2 立的成本结构上读更清楚：扩散的采样是串行的（7.2 §1：训练并行、采样串行），总成本 = 步数 × 每步前向；步数由 solver 决定（7.4），每步前向由工作空间决定——latent 压缩砍的是后者，与前者正交，两个杠杆可以叠乘。训练目标本身一行不改：7.2 的 $L_{simple}$ 与 7.5 的 $L_{RF}$ 把 $x_0$ 换成 $z_0$ 原样成立，CFG（7.4）同样原样适用——**latent 化对扩散数学是透明的**，全部代价集中在 autoencoder 的重建质量这一处（下一段的 $f$ 消融）。LDM 论文里像素空间的 LDM-1（ $f = 1$）对照组把这点坐实：同预算下它的 FID 显著差于 f4/f8 档——省下来的位点不是白省的，扩散在像素空间要为感知冗余付容量税。

但 $f$ 不是越大越好，也不是越小越坏。LDM 论文对 $f \in \lbrace1, 4, 8, 16, 32\rbrace$ 的同预算消融（ImageNet、2M 步、单卡 A100，Figure 6/7 口径，2026-11-12 经 ar5iv 原文核实）给出的边界是：像素基线 LDM-1 的训练时间显著长于一切 latent 档，且同预算下 FID 明显更差；LDM-{4–16} 在质量—吞吐上基本拉平；只有 LDM-32 因感知压缩过度而封顶整体样本质量——论文结论是 LDM-4 与 LDM-8 给出最佳合成条件，复杂数据集（ImageNet）需要更保守的压缩率。机制读法： $f$ 过大时 autoencoder 的重建误差成为生成质量的天花板——decoder 画不回去的细节，扩散模型生成得再好也无从落地。这个「压缩率—质量」权衡在 6.3 §7 的 DC-AE 处再次出现：把 $f$ 推到 32/64 同时保住重建，需要专门的优化技术（residual autoencoding + 三阶段适配），不是白来的。

这套结构的分工，LDM 论文给了两个名字：**感知压缩**（perceptual compression）归 autoencoder——把人眼不敏感的高频冗余从表示里剥掉，监督信号是重建 + LPIPS + 对抗项（6.3 §7）；**语义压缩**（semantic compression）归 latent 上的扩散模型——构图、物体、文本语义在这个低维空间里被生成。为什么分开比端到端（一个网络同时学压缩与生成）好，机制上有两条：其一，去噪目标（score matching，7.2/7.3）的容量不该浪费在「学一个感知无损的编解码」上——那是重建损失的强项；其二，两个阶段的数据与损失解耦后，autoencoder 可以在不含文本标注的纯图像上训，扩散模型只面对规整的低维分布。

```
像素空间 x [3, 512, 512]                    latent 空间（扩散的工作空间）
     │
     │  E（感知压缩，6.3）── 训练时一次前向
     ▼
  z₀ = s·E(x) [4, 64, 64]  ── 前向加噪闭式直达任意 t（7.2 式 (5)(6) 的 latent 版）
     │                          z_t = √ᾱ_t z₀ + √(1−ᾱ_t) ε
     │  采样：去噪网络在 latent 上走 K 步（7.4 的 solver / 7.5 式 (23) 的 ODE）
     ▼
  ẑ₀ ── D(ẑ₀ / s) 一次 decoder 前向 ──► x̂ [3, 512, 512]

     （s = 0.18215：scaling factor，§2）
```

一句约定桥接：按 W4 约定页，扩散语境下建模对象统一记 $x_0$；本章它具体化为 latent，故记 $z_0$（缩放后的干净 latent）， $z_t$ 为其加噪版。7.2/7.5 的全部公式把 $x_0$ 换成 $z_0$ 原样成立。

## 2. Scaling factor：0.18215 的单位方差化数学

6.3 式 (24) 已经立过这个常数： $z' = s \cdot z$， $s = 0.18215$，出处是 CompVis/stable-diffusion 的 `configs/stable-diffusion/v1-inference.yaml`（W3 锁定，勿重查）。本节补它在扩散语境下的精确含义：**错误的 latent 尺度等价于悄悄平移整条 noise schedule**。

设 $E(x)$ 的每个分量方差为 $\sigma_z^2$（训练好的 SD-VAE 实测 $\hat\sigma_z \approx 5.49$，6.3 §7.1），不缩放直接进前向加噪（7.2 式 (6) 的 latent 版）：

$$
z_t = \sqrt{\bar\alpha_t}\thinspace z_0 + \sqrt{1-\bar\alpha_t}\thinspace \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) \qquad (2)
$$

逐分量的实际信噪比为

$$
\mathrm{SNR}_{\mathrm{eff}}(t) = \frac{\bar\alpha_t\thinspace \sigma_z^2}{1 - \bar\alpha_t} = \sigma_z^2 \cdot \mathrm{SNR}_{\mathrm{nominal}}(t) \qquad (3)
$$

**符号**： $\mathrm{SNR}_{\mathrm{nominal}}(t) = \bar\alpha_t / (1-\bar\alpha_t)$：schedule 设计时的名义信噪比（7.2 式 (9)、7.5 §7 的同一对象）。

取对数： $\log \mathrm{SNR}_{\mathrm{eff}} = \log \mathrm{SNR}_{\mathrm{nominal}} + 2\ln \sigma_z$。也就是说， $\sigma_z \ne 1$ 的效果是在 log-SNR 轴上把整条 schedule 平移一个常数——linear/cosine schedule（7.2 §2.3）、logit-normal $t$ 采样（7.5 式 (27)）的全部标定都建立在「输入单位方差」的假设上，尺度错了等于整套超参对着另一条 schedule 调。取 $s = 1/\hat\sigma_z$ 预乘 $z$，式 (3) 回到名义值，schedule 恢复其设计语义；decode 侧先除以 $s$ 再进 $D$。

数值自检： $1/0.18215 \approx 5.49$，与实测 $\hat\sigma_z$ 互倒（6.3 §7.1）。SD3/FLUX 世代的 16 通道 VAE 同机制、另加一个 shift 项（scale 0.3611 / shift 0.1159，FLUX 代码实测口径，6.3 §7.2）——机制不变：把 latent 的逐分量均值与方差都钉到标定点上。

这个常数不是模型结构的一部分，是**分布标定**——但它回答了一个真实的工程问题：换 VAE（f8 4ch → f8 16ch → f32 32ch）时，扩散侧不需要重调 schedule，只要重测 $\hat\sigma_z$、换 $s$。

## 3. 条件注入三机制（加第四个：调制）

条件 $c$（text token 序列、class 标签、或空间对齐的条件图）进入去噪网络的路线，工程上收敛到三类加一类。

**(a) cross-attention：文本作 K、V。** LDM 的选择：把 latent 特征图拍平成 token 作 query，文本 token 序列作 key/value：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\negthinspace\left( \frac{QK^{\top}}{\sqrt{d_k}} \right) V, \qquad Q = Z W_Q,\hspace{5mu} K = C W_K,\hspace{5mu} V = C W_V \qquad (4)
$$

**符号**： $Z \in \mathbb{R}^{N \times d}$：latent 特征（ $N = hw$ 个位点拍平）； $C \in \mathbb{R}^{M \times d_c}$：text encoder 的 sequence 输出（ $M$ 个文本 token）； $W_Q \in \mathbb{R}^{d \times d_k}$、 $W_K, W_V \in \mathbb{R}^{d_c \times d_k}$。每个图像位点自由检索全部文本 token——空间结构（「左边的狗」）由 attention 权重显式绑定到空间位置，这是 cross-attention 路线图文对齐强的机制根源。文本表示在扩散网络内部是**静态只读**的：每层拿同一组 $K, V$ 用，不更新。走读锚点：CompVis/latent-diffusion 的 `ldm/models/diffusion/ddpm.py`（conditioning 注入）[A]。

**(b) 通道 concat：条件与 latent 同网格拼接。** 把条件张量（inpainting 的 mask 与被 mask 图像、超分的低分辨率图上采样）与 $z_t$ 沿通道维拼好进第一层 conv。代价几乎为零（只有首层输入通道变宽），但要求条件与 latent 空间逐点对齐——文本向量没有空间形状，走不了这条路。

**(c) in-context：条件 token 拼接进序列。** transformer 主干的原生做法：条件（class token、文本 token）与图像 token 拼成一条长序列过 self-attention，softmax 矩阵里自然长出图文交叉块。无新增投影参数，但 attention 从 $N^2$ 涨到 $(N+M)^2$，且条件 token 挤占序列预算。DiT 论文把它列为对照路线之一；MMDiT（§7）是它的完全体。

**(d) 调制（adaLN 族）：条件不进序列，改归一化层的系数。** 把条件压成一个全局向量，逐 block 回归 LayerNorm 的 scale/shift/gate。几乎零开销，但条件被压成单向量、丢了 token 级细节——§6 展开推导。

四路对比（「条件是否被逐层更新」一列是 §7 MMDiT 的伏笔）：

| 机制 | 注入位置 | 增量参数 | 逐层开销 | 条件表示 | 代表 |
| --- | --- | --- | --- | --- | --- |
| cross-attention | 每 block 插一层 | 每层 $W_Q, W_K, W_V$ 投影 | $O(NM)$ attention + 投影 | 静态（每层复用同一 $K,V$） | LDM / SD 1.x / SDXL |
| 通道 concat | 仅输入层 | $\approx 0$ | $\approx 0$ | 静态、需空间对齐 | inpainting 等空间条件（LDM） |
| in-context 拼接 | 整条序列 | $\approx 0$（条件 embedding） | $N^2 \to (N+M)^2$ | 逐层更新（双向） | DiT 消融对照路 |
| adaLN(-Zero) 调制 | 不进序列，调 LN | 每 block 一个调制 MLP | $\approx 0$ | 静态全局向量 | DiT / MMDiT 的时间与 pooled 条件 |

DiT 论文对 (a)(c)(d) 与朴素 adaLN 做了同主干四路消融（论文原表口径）：adaLN-Zero（(d) 的零初始化版）FID 最好且几乎免费——cross-attention 每层多付一组投影与 $O(NM)$ 的 attention，in-context 把序列拉长，两者都没换来质量。这个结论直接决定了 DiT 系主干把「时间/全局条件全走调制、文本条件另议」的格局。

## 4. Text encoder：从 CLIP 到 T5 的选择依据

条件 $C$ 的质量上限由 text encoder 决定。现役系统的选择分两族，差异可以用训练目标直接解释（6.2）。

**CLIP 系（CLIP-L、OpenCLIP-G）。** 6.2 的 InfoNCE 只监督「整图 ↔ 整句」的全局对齐：pooled 向量（EOS 位置读出）承载全局语义，sequence 输出（77 个 token）的逐 token 表示没有被细粒度监督过——它够用，是因为对齐目标只需要它够用。工程上两个读出口都用：pooled 向量进调制（时间 embedding 相加后做 adaLN 的全局条件），sequence 输出进 cross-attention 或 token 序列。SD3 的具体接法（sd3.5 实现口径）：CLIP-L 与 OpenCLIP-G 的 sequence 沿特征维拼接（每位置 $768 + 1280 = 2048$ 维）、pooled 两向量拼接后与 $t$ embedding 相加；再与 T5 的 sequence 沿序列维拼接成条件 token 序列（各段截断长度由 config 定）。三个 encoder 的参数量：CLIP-L 123M、OpenCLIP-G 695M（SDXL 公开 config 口径）、T5-XXL encoder 4.7B（SD3 论文口径）。

**T5-XXL 为什么有帮助。** 三条机制：

1. **数据规模**：纯文本语料比图文对齐数据大几个数量级，语言理解（长句法、数量、属性绑定、否定）的监督密度高得多。
2. **训练目标的颗粒度**：T5 的去噪式目标逐 token 监督表示；CLIP 的对比目标在 sequence 层面是欠监督的。文字渲染、多主体关系这类「逐 token 语义」需求，恰好打在 CLIP 的弱项上。
3. **覆盖**：罕见词、引号内字符串在 T5 的语料与分词下保留更完整。

消融证据（全部论文口径，2026-11-12 锁定；不新造数字）：Imagen（[Saharia et al., 2022](../references.md)，arXiv:2205.11487，[B]）的核心结论是放大冻结 text encoder（T5-L → XL → XXL）的收益大于放大 U-Net 本身——text encoder 是那个阶段的瓶颈变量；SD3 论文在 8B 档做了去 T5-XXL 对照：移除 T5 后人类偏好评测胜率降至 50%/46%/38%（随评测项不同，论文原表口径），损失集中在复杂组合 prompt 与文字渲染上；PixArt-α 的三阶段（§8）把「换 T5 + 重写高质量 caption」作为文图对齐阶段的核心变量。成本侧一笔账：T5-XXL 级 encoder 的推理显存与延迟不可忽略，工程折中是训练/推理侧预计算文本 embedding 缓存（文本分布远窄于图像，cache 命中率高）。

## 5. LDM 世代的主干：U-Net block 走读

LDM/SD 1.x 的去噪网络是多尺度 U-Net：下采样路径逐级减半分辨率，上采样路径镜像，同级 skip connection 拼接；每个分辨率级堆若干 ResBlock，低分辨率级（ $32^2$、 $16^2$ 等）再插 spatial transformer。逐块公式如下。

**ResBlock 与时间注入。** 时间 $t$ 先过 sinusoidal embedding（2.5 式 (1) 同款）再过两层 MLP 得 $\tau(t) \in \mathbb{R}^{C}$，在第一个 conv 之后按通道 broadcast 注入：

$$
\begin{aligned}
h_1 &= \mathrm{Conv}_{3\times3}\negthinspace\left( \mathrm{SiLU}\left( \mathrm{GN}(x) \right) \right) \\
h_2 &= h_1 + \underbrace{\tau(t)}_{\text{broadcast } [B, C, 1, 1]} \qquad (5)\\
\mathrm{out} &= \mathrm{Conv}_{3\times3}\negthinspace\left( \mathrm{SiLU}\left( \mathrm{GN}(h_2) \right) \right) + x
\end{aligned}
$$

**符号**： $\mathrm{GN}$：GroupNorm； $x$ 维度变化时 shortcut 用 $1\times1$ conv。式 (5) 是 LDM 代码的加法注入（latent-diffusion repo 实测）。等价族里的另一支是 FiLM 式 scale-shift——ADM（[Dhariwal & Nichol, 2021](../references.md)，arXiv:2105.05233，[A]）把归一化层的仿射交给时间条件：

$$
h_2 = \left( 1 + \gamma(t) \right) \odot \mathrm{GN}(h_1) + \beta(t) \qquad (6)
$$

**符号**： $\gamma(t), \beta(t) \in \mathbb{R}^{C}$：由 $\tau(t)$ 回归出的逐通道系数； $1+\gamma$ 的写法使 $\gamma = 0$ 对应恒等。式 (5) 是式 (6) 取 $\gamma \equiv 0$、把 $\beta$ 挪到卷积后的退化形态。注意这条「条件向量调制归一化层」的线索——DiT 的 adaLN（§6）就是把它从 U-Net 的 GroupNorm 搬到 LayerNorm、再补上 gate。

**spatial transformer。** 在选定的分辨率级上，特征图拍平成 $N = hw$ 个 token 过标准 transformer block：self-attention（2.3，图像位点互看）+ cross-attention（式 (4)，文本条件在此注入）+ GEGLU FFN。一个 U-Net 因此同时是卷积金字塔（多尺度归纳偏置）与若干 Transformer 级（全局混合 + 文本绑定）。

U-Net 的代价是设计空间手工化：哪一级插 attention、通道倍率怎么排，全是先验。DiT 的问题是：既然 latent 只有 $64^2$，能不能把整张网格拍平成一条序列，多尺度金字塔整个扔掉？

## 6. DiT：latent patchify、adaLN-Zero 与 scaling 账

### 6.1 patchify 的账

DiT（[Peebles & Xie, 2023](../references.md)，arXiv:2212.09748，[A]）的回答是 6.1 式 (2)(3) 的直接复用：latent 按 $p \times p$ 切块、线性投影成 token 序列，进标准 ViT 式主干，末端线性层逐 token 回归 $\hat\varepsilon$（或 $v$）再 unpatchify 回网格。ImageNet $256^2$ 经 f8 VAE 得 $32 \times 32$ latent， $p$ 的三个档（论文 Table 1 口径）：

| patch $p$ | token 数 $N = (32/p)^2$ | 单 token 输入维 $p^2 c$ | DiT-XL 单次前向 Gflops（论文口径，乘加计 1） |
| --- | --- | --- | --- |
| 8 | $16$ | $256$ | $\approx 7.4$ |
| 4 | $64$ | $64$ | $\approx 29.1$ |
| 2 | $256$ | $16$ | $\approx 118.6$ |

模型档（同表）：DiT-S/B/L/XL = 隐藏维 $d = 384/768/1024/1152$、层数 $12/12/24/28$、参数量 $33\text{M}/130\text{M}/458\text{M}/675\text{M}$。单 block 前向 FLOPs 按本书 $2mnk$ 口径为

$$
C_{\text{block}} \approx \underbrace{24 N d^2}_{\text{QKV+proj } 8Nd^2\ +\ \text{MLP } 16Nd^2} + \underbrace{4 N^2 d}_{\text{QK}^{\top}\ \text{and}\ AV} \qquad (7)
$$

**符号**：MLP 隐藏维 $4d$。论文表的 Gflops 按乘加计 1，约为本书口径的一半，两口径下结论一致。 $p$ 减半则 $N$ 乘 4：投影/MLP 项乘 4，attention 项乘 16—— $p$ 是「用算力换粒度」的旋钮，且 patchify 的线性投影只丢 $p^2 c$ 维内部的混合自由度（由投影矩阵学回），与 6.1 对 ViT 的同一笔账。

位置编码沿 6.1 的三族选择：DiT 原版用固定正弦 2D 编码（无参数、对网格形状外推友好——换分辨率时对位置网格做插值即可），这也是 §7.3 multi-aspect 变长训练在 DiT 系主干上成立的前提之一。末端层把每个 token 线性投回 $p^2 c$（学习方差时为 $2 p^2 c$）再 unpatchify——去噪预测在 latent 网格上完成，图像空间只经 decoder 出现一次。

### 6.2 adaLN-Zero：每个 block 从 identity 起步

条件注入走 §3 的 (d) 路：条件向量 $c$（class 标签 embedding + sinusoidal 时间 embedding 相加）不进序列，每个 block 由一个共享形状的调制 MLP 回归六组逐通道系数。adaLN-Zero 的 block 为

$$
\begin{aligned}
x' &= x + \alpha_1(c) \odot \mathrm{MSA}\negthinspace\left( \left(1 + \gamma_1(c)\right) \odot \mathrm{LN}(x) + \beta_1(c) \right) \\
x'' &= x' + \alpha_2(c) \odot \mathrm{MLP}\negthinspace\left( \left(1 + \gamma_2(c)\right) \odot \mathrm{LN}(x') + \beta_2(c) \right)
\end{aligned} \qquad (8)
$$

**符号**： $\gamma_i, \beta_i, \alpha_i \in \mathbb{R}^{d}$：由 $c$ 经调制 MLP 回归的 scale / shift / gate； $\mathrm{LN}$ 无可学仿射（其参数职能被 $\gamma, \beta$ 接管）。DiT 原文等价写法是 $\gamma \odot \mathrm{LN} + \beta$，此处取 $1+\gamma$ 使 $\gamma = 0$ 对应恒等，两者差一个参数化平移。

**零初始化推导。** 把调制 MLP 的最后一层线性层零初始化，则训练起点处 $\gamma_i = \beta_i = \alpha_i = 0$，逐条代入式 (8)： $(1+0)\odot\mathrm{LN}(x) + 0 = \mathrm{LN}(x)$，而

$$
x' = x + 0 \odot \mathrm{MSA}(\mathrm{LN}(x)) = x, \qquad x'' = x' \qquad (9)
$$

即**每个 DiT block 在初始化时严格等于 identity**。端到端后果：patchify 后过 $L$ 个恒等 block，末端层同样零初始化，整网输出恒为零——初始预测 $\hat\varepsilon \equiv 0$，初始 $L_{simple} = \mathbb{E}\Vert\varepsilon\Vert^2 = $ 元素个数（每元素恰为 1），损失从一个良定义的标定点出发，没有随机初始化调制系数造成的尺度冲击。梯度侧的对价同样明确：残差通路全程恒等，反向梯度从末端层无衰减直达每个 block 的调制参数（2.7 Pre-LN 残差通路的同款论证）， $\alpha_i$ 从零开始的每一次更新都在「开通」一条子通路——网络深度是被逐 block 学出来的，而不是初始化时就全额承担。DiT 论文的同主干消融（论文原表口径）：adaLN-Zero 优于非零初始化 adaLN、优于 cross-attention 与 in-context（§3 表），且调制参数的代价是每 block 一个 $d \to 6d$ 的 MLP——这笔参数账到 §8 的 PixArt-α 还要再算一次。

### 6.3 DiT scaling：Gflops ↔ FID 的对数线性

DiT 论文的标题贡献是一条 scaling 曲线：固定数据与训练配方，FID 随单次前向 Gflops 近似对数线性下降、在扫描范围内未见饱和。两个杠杆沿同一条曲线起作用——加大模型档（ $d \times$ 层数，式 (7) 第一项按 $d^2$ 长）或缩小 $p$（ $N$ 按 $1/p^2$ 长）——即「质量由投给每个样本的计算量决定」，与 LLM 的 scaling law（1.3）同构。锚点数字（论文口径，2026-11-12 锁定）：DiT-XL/2 加 classifier-free guidance 在 ImageNet $256^2$ 类别条件生成上 FID 2.27，把此前 LDM-4 的 3.60 压过去——同样的 latent 空间、同样的 f8 VAE，换的是主干与注入方式。这条曲线的工程含义是：扩散模型的质量预算第一次可以像 LLM 一样按算力外推，7.7 的系统拆解里各家模型档的排布都是这条曲线的实例。

## 7. MMDiT（SD3）：双流、联合 attention 与 QK-norm

### 7.1 从 DiT 到 MMDiT：文本也要逐层更新

DiT 的 adaLN 把文本条件压成一个全局向量——token 级语义在调制里被压扁；cross-attention 版则把文本冻结成只读源（§3 (a)）。SD3（[Esser et al., 2024](../references.md)，arXiv:2403.03206，[A]）的 MMDiT 取第三条：图像与文本**各持一套 block 参数**（各自的 QKV 投影、各自的 MLP、各自的 adaLN 系数），但 attention 在两流拼接的序列上联合计算——文本 token 因此逐层被图像去噪状态修正，图像 token 逐层重新读取更新后的文本。双向，是它与 §3 (a) 的本质差异。

设图像流 $Z \in \mathbb{R}^{N \times d}$（patchify 后的 latent token）、文本流 $Y \in \mathbb{R}^{M \times d}$（§4 的 CLIP+T5 拼接序列，经投影到 $d$）。每个 block：两流各自的 adaLN 系数由共享条件向量（pooled 文向 + $t$ embedding）回归；然后

$$
Q = \begin{bmatrix} Z W_Q^{z} \\ Y W_Q^{y} \end{bmatrix}, \qquad K = \begin{bmatrix} Z W_K^{z} \\ Y W_K^{y} \end{bmatrix}, \qquad V = \begin{bmatrix} Z W_V^{z} \\ Y W_V^{y} \end{bmatrix} \qquad (10)
$$

**符号**：堆叠是序列维拼接， $Q, K, V \in \mathbb{R}^{(N+M) \times d}$； $W^{z}$、 $W^{y}$ 是两流各自的投影参数（MMDiT 的「MM」：同构不同参）； $d_h = d / n_h$：单个 attention head 的维度（式 (11) 起用）。

联合 attention 矩阵写出 $2 \times 2$ 分块：

$$
S = \frac{QK^{\top}}{\sqrt{d_h}} = \begin{bmatrix} S_{zz} & S_{zy} \\ S_{yz} & S_{yy} \end{bmatrix}, \qquad O = \mathrm{softmax}(S)\thinspace V, \qquad O_z = O_{[1:N]},\hspace{5mu}\hspace{5mu} O_y = O_{[N+1:N+M]} \qquad (11)
$$

**符号**：softmax 在长度为 $N+M$ 的行上归一化； $S_{zz} \in \mathbb{R}^{N \times N}$：图像自注意力； $S_{zy}$：图像 query 读文本 key——cross-attention 的那一份语义原样保留在右上块； $S_{yz}$：文本 query 读图像——cross-attention 没有的对称块，文本表示借此随去噪进程逐层细化； $S_{yy}$：文本自注意力。拆回后两流各自过输出投影、gate 残差与各自 MLP：

$$
Z \leftarrow Z + \alpha_z \odot O_z W_O^{z}, \qquad Y \leftarrow Y + \alpha_y \odot O_y W_O^{y} \qquad (12)
$$

**符号**： $W_O^{z}, W_O^{y}$：两流各自的 attention 输出投影； $\alpha_z, \alpha_y$：两流各自的 adaLN gate（式 (8) 同款零初始化）。（两流各自的 MLP 子层同构续上，式 (8) 形状。）走读锚点：Stability-AI/sd3.5 的 `mmditx.py`（双流 QKV 与拼接 attention）[A]。一个结构性细节：网络出口只有图像流—— $Z$ 经末端 adaLN 与线性层回归速度 $v_\theta$（7.5 的 rectified flow 目标）， $Y$ 从头到尾「只被读、不出图」，它在每个 block 被更新的全部意义是让下一个 block 的图像 token 读到更贴合当前去噪状态的文本表示。文本流是手段，不是产出。

```
   Z ∈ R[N,d]（图像 latent token）        Y ∈ R[M,d]（文本 token）
      │                                     │
   adaLN_z(c)                            adaLN_y(c)      c = pooled 文向 + emb(t)
      │                                     │
   Qz,Kz,Vz ← Z·W_z                      Qy,Ky,Vy ← Y·W_y     ← 两流各自权重
      └────────────────┬────────────────────┘
                       ▼  序列维拼接 → [N+M, ·]
        S = QKᵀ/√d_h = ┌ S_zz   S_zy ┐    softmax 按行（跨 N+M 个 key 归一）
                       └ S_yz   S_yy ┘
                       ▼
        O = softmax(S)·V  →  拆回 Oz = O[:N]、Oy = O[N:]
                       ▼
   Z ← Z + αz·Oz·Wz_out ；Z ← Z + αz'·MLP_z(adaLN_z(Z))
   Y ← Y + αy·Oy·Wy_out ；Y ← Y + αy'·MLP_y(adaLN_y(Y))
```

**账与性质。** 交叉项开销 $2NM$ 与 cross-attention 路线的 $NM$ 同量级；真正多付的是文本流每 block 的 QKV/MLP（ $O(Md^2)$）与 $M^2$ 自注意力——买的是 $Y$ 的逐层更新。另一个不显然的性质：softmax 的分母在图文共 $N+M$ 个 key 上联合归一，图像 token 分给文本的注意力权重与文本 token 自身的存在感直接竞争——「条件强度」本身变成一个被注意力几何化的可学量，而不只是投影矩阵里的静态权重。

SD3 系统的完整配方至此闭环：MMDiT 主干 + rectified flow 目标（7.5 式 (22)，回归 $\varepsilon - z_0$）+ logit-normal $t$ 采样（7.5 式 (27)）+ 分辨率相关 shift（7.5 式 (30)）+ §4 的三 text encoder。训练侧走读锚点另有 Stability-AI/generative-models 的 `sgm/modules/diffusionmodules/` [A]。

### 7.2 QK-norm：把 attention logit 的上界钉死

模型档推到 2B 量级时，SD3 论文（Figure 5 口径）记录了注意力训练失稳： $q, k$ 的范数随训练增长，logit 绝对值变大，softmax 进入饱和区、梯度消失（注意力熵坍缩）。解法是 query/key 在进入点积前各过一次 per-head RMSNorm（无可学仿射）：

$$
\hat{q} = \frac{q}{\mathrm{RMS}(q)}, \quad \hat{k} = \frac{k}{\mathrm{RMS}(k)}, \qquad \mathrm{RMS}(q) = \sqrt{\tfrac{1}{d_h}\sum_{i=1}^{d_h} q_i^2} \qquad (13)
$$

归一化后 $\Vert\hat{q}\Vert_2 = \Vert\hat{k}\Vert_2 = \sqrt{d_h}$，于是每个 logit 满足

$$
\left| \frac{\hat{q}^{\top} \hat{k}}{\sqrt{d_h}} \right| \le \frac{\Vert\hat{q}\Vert\thinspace \Vert\hat{k}\Vert}{\sqrt{d_h}} = \sqrt{d_h} \qquad (14)
$$

（Cauchy-Schwarz。）logit 的幅度上界被钉在 $\sqrt{d_h}$（ $d_h = 64$ 时约 8），与权重尺度解耦——失稳通道从源头关闭。实现见 sd3.5 `mmditx.py` 的 QK-norm 选项 [A]。

### 7.3 FLUX 的 double→single stream 变体 [C] 与 multi-aspect bucketing

FLUX（[black-forest-labs/flux](../references.md)，[C]——官方推理代码公开、HF 权重 gated，只述已披露信息）在 MMDiT 骨架上做了一个结构变体：前段是 double-stream block（式 (10)-(12) 同构，图文各自参数），后段切换为 single-stream block——两流 token 拼接后**共用**一套 QKV/MLP 参数。读法：双流分工在前段学成之后，后段退回参数共享的 in-context 形态（§3 (c)），参数与算力省一份；文本侧为 CLIP-L pooled + T5 序列（[C]，已披露口径）。

**multi-aspect bucketing。** 高分辨率 T2I 训练要覆盖从竖图到横图的全长宽比，而 attention 成本随 token 数平方变。工程惯例（无单一出处）：把数据集按长宽比分桶，每个 batch 只装同一桶的样本（同形同 size），于是无 padding 浪费、per-sample FLOPs 按桶可预测；位置编码对变长网格做插值（6.1 的 2D 位置编码族），NaViT 式 packing（6.1）是免桶的替代路线。SD3/FLUX 世代的多分辨率训练都建立在这类调度上。

## 8. 效率线：PixArt-α 三阶段与 SANA 的 Linear DiT

### 8.1 PixArt-α：把 DiT 系统的训练账单拆成三段

PixArt-α（[Chen et al., 2023](../references.md)，arXiv:2310.00426，[A]）的前提是 §6.3 的 scaling 曲线反过来读：既然质量由算力决定，就把每一份算力花在刀刃上。三阶段训练，每段各由一个消融背书（论文原表口径）：

1. **像素依赖预训练**：先在 ImageNet 类别条件任务上训 DiT 主干（ $z_0$ 空间、无文本）。自然图像的低层统计（纹理、边缘、局部结构）与文本无关，不该占用文图数据的算力；消融显示该阶段显著加速后续文图收敛。
2. **文图对齐**：换上 Flan-T5-XXL text encoder（约 4.3B 参数，论文原文口径）并把标签条件头换成文本条件；关键变量是 caption 质量——用 LLaVA 重写的高密度 caption 替代原生 alt-text，消融显示对齐质量的上限由 caption 信息量决定（§4 的 T5 论据在此落地）。
3. **高分辨率美学微调**：在高质量美学数据上把分辨率顶上去；消融显示美学质量主要来自这一段的数据筛选，而非架构改动。

配套两个结构/工程件：**adaLN-single**——全部 block 共享一个调制 MLP，block 间差异由一个可学的逐 block embedding 提供。动机是一笔参数账：逐 block 的 adaLN 调制线性层（ $d \to 6d$）在 DiT 上的累计开销约占主干参数量的 27%（PixArt-α 论文原文口径）；adaLN-single 把 $L$ 份调制 MLP 换成一份共享 MLP 加 $L$ 个 $6d$ 的可学 embedding，把这项开销压到常数级，省下的参数改投主干。以及文本 embedding 预计算缓存（§4 已述）。净效果（论文口径：正文 12%，v3 摘要作 10.8%）：PixArt-α 的训练成本约为 Stable Diffusion v1.5 的 12%，质量对标同期大一个量级的系统。后续 PixArt-Σ（[Chen et al., 2024](../references.md)，arXiv:2403.04692，[A]）走 weak-to-strong：继承 α 的权重，用更高质量数据与 KV 压缩把生成推到 4K——三阶段的哲学（用旧模型的产出喂养新模型的高效训练）延伸到分辨率维度。

### 8.2 SANA：Linear DiT + DC-AE f32

SANA（[Xie et al., 2024](../references.md)，arXiv:2410.10629，[A]；官方实现 NVlabs/Sana [A]）把效率问题拆成两个独立杠杆，各自顶到头。

**杠杆一：更深的压缩。** 换 6.3 §7.3 的 DC-AE（[Chen et al., 2024](../references.md)，arXiv:2410.10733，[A]）， $f = 32$： $1024^2$ 图像的 latent 只有 $32 \times 32 = 1024$ 个位点（f8 下同图为 $128^2 = 16384$——16 倍差距），patchify 取 $p = 1$，token 数 $N = 1024$。

**杠杆二：更便宜的序列混合。** DiT block 里的 softmax attention 换成 linear attention（4.3 的机制直接进扩散）：特征映射取 $\phi = \mathrm{ReLU}$，保留归一化分母，结合律重排

$$
O_i = \frac{\phi(Q_i) \left( \sum_{j=1}^{N} \phi(K_j)^{\top} V_j \right)}{\phi(Q_i) \left( \sum_{j=1}^{N} \phi(K_j)^{\top} \right)}, \qquad \phi = \mathrm{ReLU} \qquad (15)
$$

**符号**：式 (15) 即 SANA 论文 Eq. (1)（2026-11-12 经 ar5iv 原文核实）：分子的共享项 $\sum_j \phi(K_j)^{\top} V_j \in \mathbb{R}^{d_h \times d_h}$ 与分母的共享项 $\sum_j \phi(K_j)^{\top} \in \mathbb{R}^{d_h}$ 全序列各算一次、被全部 query 复用——省力的来源是共享与重排，分母并不省。非因果双向场景下左序为 $4N^2 d_h$、右序为 $4N d_h^2$（本书 $2mnk$ 口径，分母与逐 query 读出为低阶项），交叉点 $N = d_h$； $N = 1024$、 $d_h = 64$ 时 attention 项省约 16 倍——4.3 式 (4) 的账从 LLM 原样搬到 DiT。占 block 总 FLOPs 的比例用式 (7) 估（ $d = 1152$ 档）：f8 下 $1024^2$ 图（ $N = 4096$）attention 项约占 37%，f32 下（ $N = 1024$）约 13%——两个杠杆是乘性的：先靠 DC-AE 把 $N$ 压下来，再靠 linear attention 把剩下的二次项抹掉。

删掉 softmax 丢掉的是局部归纳偏置（softmax 的锐化天然倾向近邻），SANA 的补偿是 **Mix-FFN**：FFN 换成 inverted residual block + $3\times3$ depthwise conv + GLU 的组合，把局部混合职责从 attention 挪给卷积——conv 的 $3\times3$ 邻域混合是逐 token 常数成本，不进式 (7) 的二次项。论文消融（Table 8/12 口径）把这条因果链钉死：self-attention 直接换 linear attention 会掉 FID 与 CLIP Score，Mix-FFN 把损失补回；同一份消融还坐实了 NoPE——零填充 $3\times3$ 卷积隐式携带位置信息，SANA 是首个完全去掉位置编码的 DiT，论文报告无质量损失。text encoder 用 Gemma-2-2B（decoder-only LM 的表示直接作条件，§4 的「纯文本监督密度」论据的又一实例）；训练目标为 FM 线性路径（7.5 式 (22)），采样配论文提出的 Flow-DPM-Solver（7.4 指数积分族在速度场坐标下的对应物：直轨迹 → 大步长）。DC-AE 的深压缩还带来一条隐性收益： $1024^2$ 的训练可以在原生分辨率端到端进行（latent 只有 $32^2$），免掉「低分辨率预训练 + 高分辨率微调」课程的一个档。系统账（论文口径）：SANA-0.6B 在 $1024^2$ 上对标大其约 20 倍的 FLUX 级系统，笔记本 GPU 单图 0.88s（FP16，论文 Table 5），吞吐高约两个数量级。

把 §6-§8 合起来看，效率线的全部动作只有三个，且都不改动扩散数学骨架：更深的 latent 压缩（f8 → f32）、更便宜的序列混合（softmax → linear）、更聪明的训练课程（三阶段 + weak-to-strong）。

## 总结

| 对象 | 公式/数字 | 要点 |
| --- | --- | --- |
| latent 账 | 式 (1)；f8 位点削减 $f^2 = 64\times$ | 感知压缩（AE）与语义生成（diffusion）两阶段分工；LDM 消融：LDM-{4–16} 拉平、LDM-1 慢、LDM-32 限质，f4/f8 最佳 |
| scaling factor | $s = 1/\hat\sigma_z = 0.18215$，式 (2)(3) | 错误尺度 = log-SNR 轴平移 $2\ln\sigma_z$；单位方差化保 schedule 语义 |
| cross-attention | 式 (4) | 文本作 K/V、静态只读；§3 四路对比表，DiT 消融：adaLN-Zero 最优 |
| text encoder | CLIP pooled+sequence vs T5 | T5 强在数据规模、逐 token 监督、长尾覆盖；Imagen：放 text encoder 收益 > 放 U-Net；SD3 去 T5 掉复杂 prompt |
| U-Net block | 式 (5)(6) | ResBlock 加法/FiLM 注入时间；spatial transformer = self + cross + GEGLU |
| DiT patchify | 式 (7)： $24Nd^2 + 4N^2d$ | $p$ 减半 → $N \times 4$、attention 项 $\times 16$；S/B/L/XL = 33M/130M/458M/675M |
| adaLN-Zero | 式 (8)(9) | 六系数调制、gate 零初始化 ⇒ 每 block identity 起步、初始损失良定义、残差通路全程恒等 |
| DiT scaling | FID 对数线性随 Gflops 降 | XL/2-G FID 2.27 vs LDM-4 3.60（ImageNet $256^2$，论文口径） |
| MMDiT 联合 attention | 式 (10)(11)(12) | 双流各自 QKV/MLP/adaLN； $S$ 的 $2\times2$ 分块， $S_{yz}$ = 文本逐层被图像修正 |
| QK-norm | 式 (13)(14) | per-head RMSNorm ⇒ logit 上界 $\sqrt{d_h}$，与权重尺度解耦 |
| FLUX 变体 [C] | double → single stream | 前段双流分工、后段参数共享（已披露口径） |
| PixArt-α | 三阶段 + adaLN-single | adaLN 调制占 DiT 参数约 27% → single 版压至常数级（论文口径）；训练成本约 SD1.5 的 12%（论文口径） |
| SANA | 式 (15) + DC-AE f32 | $N = 1024$ @ $1024^2$；ReLU linear attention（分子/分母共享项，分母保留）省 attention 项约 $16\times$；Mix-FFN 补局部性并带来 NoPE |

## 练习

1.（计算）latent 与 patchify 联合账：对 $1024^2$ 图像、 $d = 1152$ 的主干，分别取 (a) f8 + $p=2$、(b) f8 + $p=1$、(c) f32 + $p=1$（SANA 档），算 $N$、单 block 前向 FLOPs（式 (7)， $2mnk$ 口径）与 attention 项占比；再算 (c) 换 ReLU linear attention（式 (15)， $d_h = 64$）后 attention 项的降幅倍数。验证交叉点 $N = d_h$ 的推导（非因果、两侧满算）。
2.（推导）scaling factor 的另一视角：设 v-prediction（7.2 式 (27)）目标 $v = \sqrt{\bar\alpha_t}\thinspace\varepsilon - \sqrt{1-\bar\alpha_t}\thinspace z_0$。若 latent 未缩放（ $\mathrm{Var}(z_0) = \sigma_z^2 \ne 1$），把 $z_t$ 用真实方差展开，求网络在最优解处回归的目标与名义 $v$ 的偏差；由此说明 $s = 1/\hat\sigma_z$ 同时修正了前向边际与回归目标两处。
3.（推导）adaLN-Zero 的恒等起点：在式 (8) 下证明 (a) 零初始化时全网络输出恒为零（设末端层同样零初始化）；(b) 初始 $L_{simple}$ 等于 $z_0$ 的元素个数；(c) 若把 gate 改为随机初始化（小方差），反向一阶分析中 $\partial L / \partial \alpha_i$ 与残差通路梯度的相对尺度如何变化——为什么零初始化更稳。
4.（推导 + 代码）MMDiT 联合 attention：把式 (11) 的 softmax 按行展开，证明文本流的输出 $O_y$ 恰等于「以图像为 K/V 的 cross-attention 与文本自注意力的 softmax 混合」（写出分母合并的形式）；再用教学用 PyTorch（非优化实现）写出单个 MMDiT block 的前向（两流各 4 个投影 + 拼接 attention + 拆回），并在 $N = 256$、 $M = 77$、 $d = 768$ 上验证输出 shape 与参数计数。

## 参考

- [Rombach et al., 2022](../references.md)：High-Resolution Image Synthesis with Latent Diffusion Models，arXiv:2112.10752，[A]（key: ldm）——式 (1)(4)：latent 空间扩散、两阶段分工、 $f$ 消融、cross-attention 注入
- [Peebles & Xie, 2023](../references.md)：Scalable Diffusion Models with Transformers，arXiv:2212.09748，[A]（key: dit）——式 (7)(8)(9)：patchify 账、adaLN-Zero、四路 conditioning 消融、Gflops↔FID scaling、FID 2.27
- [Esser et al., 2024](../references.md)：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis，arXiv:2403.03206，[A]（key: sd3）——式 (10)-(14)：MMDiT 双流与联合 attention、三 text encoder 配置、QK-norm；训练目标 recap 见 7.5
- [Chen et al., 2023](../references.md)：PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis，arXiv:2310.00426，[A]（key: pixart-alpha）——三阶段训练、adaLN-single、T5 条件、训练成本口径
- [Chen et al., 2024](../references.md)：PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation，arXiv:2403.04692，[A]（key: pixart-sigma）——weak-to-strong 4K
- [Xie et al., 2024](../references.md)：SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers，arXiv:2410.10629，[A]（key: sana）——式 (15)：Linear DiT（ReLU linear attention、Mix-FFN）、DC-AE f32、Gemma-2-2B 条件、效率账（摘要口径）
- [Chen et al., 2024](../references.md)：Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models，arXiv:2410.10733，[A]（key: dc-ae）——f32 latent 出处（6.3 §7.3 完整拆解）
- [Kingma & Welling, 2013](../references.md)：Auto-Encoding Variational Bayes，arXiv:1312.6114，[A]（key: vae）——两阶段分工中 AE 一侧的目标函数（6.3）
- [Saharia et al., 2022](../references.md)：Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding，arXiv:2205.11487，[B]（key: imagen）——「放大 text encoder > 放大 U-Net」消融（[B]：论文公式可引、无官方代码；系统拆解在 7.7）
- [Radford et al., 2021](../references.md)：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020，[A]（key: clip）——CLIP pooled/sequence 表示的训练机制（6.2）
- [Dhariwal & Nichol, 2021](../references.md)：Diffusion Models Beat GANs on Image Synthesis，arXiv:2105.05233，[A]（key: classifier-guidance）——式 (6) FiLM 式时间注入（ADM）
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——前向闭式与 $L_{simple}$（7.2 全链推导）
- [CompVis/latent-diffusion](../references.md)：LDM 官方实现（`ldm/models/diffusion/ddpm.py`），[A]（key: latent-diffusion-repo）——式 (4)(5) 走读锚点
- [facebookresearch/DiT](../references.md)：DiT 官方实现（`models.py`），[A]（key: dit-repo）——式 (8) adaLN-Zero 走读锚点
- [Stability-AI/sd3.5](../references.md)：SD3.5 官方推理代码（`mmditx.py`），[A]（key: sd3.5-repo）——式 (10)-(14) 走读锚点
- [Stability-AI/generative-models](../references.md)：SGM 训练代码（`sgm/modules/diffusionmodules/`），[A]（key: sgm-repo）——训练侧走读锚点
- [black-forest-labs/flux](../references.md)：FLUX 官方推理代码，[C]（key: flux-repo）——§7.3 double→single 变体（权重 gated，仅已披露口径）
- [NVlabs/Sana](../references.md)：SANA 官方实现，[A]（key: sana-repo）——式 (15) 与 configs 走读锚点
- [PixArt-alpha/PixArt-alpha](../references.md)：PixArt 官方实现（含 Σ），[A]（key: pixart-repo）——§8.1 走读锚点
