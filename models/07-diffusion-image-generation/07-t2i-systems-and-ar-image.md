# 7.7 T2I 系统拆解与 AR 生图（T2I Systems and Autoregressive Image Generation）

> 本章把 7.1–7.6 推导过的每个部件装回整机：SD3 = 现代开放参考实现的完整配方汇总（MMDiT + rectified flow + logit-normal，每一项指向前文章节）、SDXL（micro-conditioning 把数据质量标注写进条件、base+refiner 沿 t 轴分工）、Imagen [B]（像素级联 64→256→1024、T5-only text encoder、dynamic thresholding）；然后整条换掉生成目标：LlamaGen（「图像 = 另一种语言」的公式论证）、VAR（next-scale prediction 的顺序设计）、Infinity（bitwise 词表）；再换一族解码范式：MaskGIT / MUSE（mask-predict）；Emu3（图、视、文全 token 统一的账）；最后落一张 diffusion vs AR vs mask-predict 对照总表 | 前置：7.1（家族坐标与对照表）、7.2（DDPM 约定、频谱读法）、7.3（PF-ODE）、7.4（CFG 与 dynamic thresholding 推导、蒸馏族）、7.5（rectified flow、logit-normal、timestep shift）、7.6（latent 账、text encoder 消融、MMDiT）、6.3（VQ/RVQ/LFQ tokenizer）、2.8（decoder block 标准配方）、2.4（KV cache）、1.1（AR 分解与精确似然）、1.3（scaling law 账）、4.5（MoE 专家分工） | 锚点：SD3 [A]、SDXL [A]、LlamaGen [A]、VAR [A]、Infinity [A]、MaskGIT [A]、Emu3 [A]、Imagen [B]、MUSE [B]；Stability-AI/sd3.5 [A]、Stability-AI/generative-models [A]、stabilityai/stable-diffusion-xl-base-1.0 [A]、FoundationVision/LlamaGen [A]、FoundationVision/VAR [A]、FoundationVision/Infinity [A]、baaivision/Emu3 [A]

## 1. SD3：现代开放参考实现的配方汇总

先给一个判据：SD3（[Esser et al., 2024](../references.md)，arXiv:2403.03206，[A]）的配方表里没有一行是本书前六章没推导过的。权重、推理代码、论文三者齐（走读锚点 [Stability-AI/sd3.5](../references.md) 的 `mmditx.py` 与 `sd3_impls.py`，[A]；训练侧 [Stability-AI/generative-models](../references.md) 的 `sgm/modules/diffusionmodules/`，[A]），且每个选择都能在前文找到推导出处——这就是「现代开放参考实现」定位的全部内容：它不是新部件的组合，而是已推导部件的装配图。

| 部件 | SD3 的选择 | 推导出处 |
| --- | --- | --- |
| 表示空间 | f8 16ch VAE latent： $1024^2$ 像素 → $128^2$ latent 网格 | 6.3 §7.2 |
| 生成目标 | rectified flow： $z_t = (1-t) z_0 + t \varepsilon$，回归目标速度 $\varepsilon - z_0$ | 7.5 §6 |
| $t$ 的采样分布 | logit-normal$(0, 1)$ = log-SNR 轴上的高斯，质量压中段 | 7.5 §8 |
| 分辨率 shift | $t' = s t / (1 + (s-1) t)$， $1024^2$ 档取 $s = 3$ | 7.5 §8 |
| 主干 | MMDiT：图文双流各自 QKV/MLP/adaLN + 拼接序列联合 attention | 7.6 §7.1 |
| 训练稳定化 | QK-norm：per-head RMSNorm 把 logit 上界钉在 $\sqrt{d_h}$ | 7.6 §7.2 |
| 文本条件 | CLIP-L + OpenCLIP-G + T5-XXL 三 encoder（123M + 695M + 4.7B） | 7.6 §4 |
| 采样 | PF-ODE solver（7.3 §6 族）+ CFG： $\tilde v = v_\theta(\varnothing) + w\thinspace(v_\theta(c) - v_\theta(\varnothing))$ | 7.4 §4.2 |

整机数据流走一遍（推理方向， $t: 1 \to 0$）：

```
prompt ─► CLIP-L ─────┐ sequence 沿特征维拼接 ──┐
      ─► OpenCLIP-G ──┘                          ├─► 条件序列 Y（文本流 token）
      └►（两者 pooled ⊕ emb(t) = adaLN 条件 c）──┘
prompt ─► T5-XXL ───► sequence 沿序列维拼进 Y ────┘
                                                       │ 每个 block：Y 被读也被更新
ε ~ N(0,I) = z_1 ─► patchify ─► Z（图像流）─► MMDiT × L 层 ─► v_θ(Z, t, c)
   每步：ṽ = v(∅) + w·(v(c) − v(∅))（CFG） ，z ← z − h·ṽ
z_0 ─► 除以 scaling factor ─► VAE decoder ─► 像素图
```

**scaling 证据。** SD3 论文报告验证 loss 随主干规模（depth 15–38，最大 8B 档）持续下降，且与人类偏好评测同向改善；8B 档在人类偏好上超过 SDXL 等前代开放模型（均论文 Figure 口径，2026-11-12）。机制读法：DiT 在 ImageNet 上的 Gflops↔FID 对数线性（7.6 §6.3）延续到了 T2I 尺度——「换 transformer 主干就能吃 scaling law」这个 DiT 的核心论点，由 SD3 在开放权重系统上坐实。后续的 SD3.5 与 FLUX（[C]，权重 gated、仅述已披露口径）都是这台整机的直接变体：前者同配方的权重更新，后者 double→single stream 的结构变体（7.6 §7.3）。

本章其余系统都可以读作对这张配方表的改写：SDXL 改的是条件通路与阶段分工，Imagen 改的是表示空间与模型分解，AR 系把生成目标整个换掉。

## 2. SDXL：micro-conditioning 与 base+refiner 两阶段

先看一个数据管线的矛盾。训练 $1024^2$ 模型时，长宽比各异的原图要切成方形训练样本（multi-aspect bucketing，7.6 §7.3）：直接中心裁剪会产出大量「构图被切」的样本——人物只剩半身、物体被边框切断；模型学到这些样本后，生成时也会随机断头。SDXL 之前的对策是丢数据（按尺寸与裁剪质量过滤），SDXL（[Podell et al., 2023](../references.md)，arXiv:2307.01952，[A]）反过来：**不丢数据，把每张样本的原始尺寸与裁剪坐标作为条件标注给模型**。

**微条件化（micro-conditioning）。** 设训练样本来自原图尺寸 $(h_0, w_0)$、裁剪窗口左上角坐标 $(c_y, c_x)$、目标分辨率 $(h_t, w_t)$。每个标量各自过一组 Fourier 特征（与 timestep embedding 同族的 sinusoidal 展开，注入通路见 7.6 §5）：

$$
\phi(v) = \left[ \sin(\omega_1 v),\thinspace \cos(\omega_1 v),\thinspace \ldots,\thinspace \sin(\omega_m v),\thinspace \cos(\omega_m v) \right], \qquad \omega_k = \exp\left( -\tfrac{k}{m} \ln \Omega \right) \qquad (1)
$$

**符号**： $v$：待嵌入的标量（尺寸或坐标）； $\phi: \mathbb{R} \to \mathbb{R}^{2m}$：Fourier 特征映射； $\Omega$：基频常数（与 2.5 的 RoPE 基频同族设计， $10^4$ 量级）； $m$：频率个数。各标量的嵌入与 pooled 文向拼接后线性投影进 time embedding 通路：

$$
\tau = W \left[ \tau_t \thinspace\Vert\thinspace \phi(h_0) \thinspace\Vert\thinspace \phi(w_0) \thinspace\Vert\thinspace \phi(c_y) \thinspace\Vert\thinspace \phi(c_x) \thinspace\Vert\thinspace \phi(h_t) \thinspace\Vert\thinspace \phi(w_t) \thinspace\Vert\thinspace e_{\mathrm{pooled}} \right] \qquad (2)
$$

**符号**： $\tau_t$：timestep 的 sinusoidal embedding； $\Vert$：向量拼接； $e_{\mathrm{pooled}}$：OpenCLIP-G 的 pooled 文向； $W$：可学投影（把拼接向量映到 time embedding 维度，随后按 7.6 §5 的 scale-shift 注入每个 ResBlock）。实现锚点：[stabilityai/stable-diffusion-xl-base-1.0](../references.md)（model_index.json 实测 `StableDiffusionXLPipeline`、双 text encoder CLIP-L + OpenCLIP-G，2026-11-12），训练侧 [Stability-AI/generative-models](../references.md)，均 [A]。

机制读法：模型学到的是条件分布 $p(x_0 \mid c,\thinspace h_0, w_0, c_y, c_x)$——「断头构图」不再混进主分布，而是被收进 $c_y > 0$ 的条件分支；推理时固定 $(c_y, c_x) = (0, 0)$、 $(h_0, w_0) = (h_t, w_t)$，采样落在「全幅构图」的条件模式上。本质是把数据清洗的判断题改成条件生成的选择题：**标注替代丢弃**——小图、裁剪图全部保留（数据量与多样性保住），质量信号以条件形式进模型，推理时把条件拨到「好」的角上。同一招可挂任何可标注的信号（美学分、水印标记等），是数据工程进条件通路的通用范式。

**base + refiner 两阶段。** SDXL 的整机由两个独立权重的 latent diffusion 模型组成（同 f8/4ch latent 空间，6.3 §7.1）：base 是 2.6B 的 U-Net 主干（论文口径；text encoder 侧 CLIP-L 123M + OpenCLIP-G 695M，7.6 §4 已核），refiner 是同族 U-Net，只在**低噪声段**（ $t$ 近 0 的一端）训练与工作——正是 7.2 §2.3 频谱读法里高频细节成形的区间。两种用法（论文口径）：(a) 切换式（ensemble of expert denoisers）：base 从 $t = 1$ 走到切换时刻 $t_s$，refiner 接管 $t_s \to 0$；(b) 后处理式：base 出完整 latent，加少量噪声回到 $t_s$ 再让 refiner 去回 0（SDEdit 风格的细节精修）。账的读法与 4.5 的 MoE 同构：MoE 沿 token 轴分工，这里沿 $t$ 轴分工——两个专家各自只在自己擅长的噪声区间分配容量，refiner 不必学构图、base 不必精修纹理。refiner 是可选项：base 单用即可出图，加 refiner 换细节与纹理的偏好分（论文口径）。

定位一句话：SDXL 的生成目标与主干家族全是 7.2/7.6 §5 的旧部件（latent + U-Net + cross-attention），新意集中在**条件通路**（式 (1)(2)）与**沿 $t$ 的系统分工**——它是一个「系统工程」而非「算法」意义上的升级。

## 3. Imagen [B]：像素级联、T5-only 与 dynamic thresholding

Imagen（[Saharia et al., 2022](../references.md)，arXiv:2205.11487，[B]——Google 论文、无官方代码，本节只引论文公式与口径，实现细节不断言）的三个选择与 SD 系全部相反：留在**像素空间**而非 latent、用**级联**三个模型而非单模型、text encoder **只用 T5 不用 CLIP**。三个相反选择各对应一笔账。

**级联分解。** 整机是三个像素空间 diffusion 的链：base $64^2$ text-conditional U-Net → SR $64 \to 256$ → SR $256 \to 1024$，联合分布被分解为三个条件分布的乘积：

$$
p(x^{1024} \mid c) = p(x^{64} \mid c) \hspace{5mu} p(x^{256} \mid x^{64}, c) \hspace{5mu} p(x^{1024} \mid x^{256}, c) \qquad (3)
$$

**符号**： $x^{r}$： $r \times r$ 分辨率的图像； $c$：文本条件。机制读法：每个因子只承担一个尺度的信息量。base 在 $64^2$ 网格上决定全局构图——这个规模连同 7.6 §1 的位点账都付得起（ $4096$ 个像素位点，像素空间 attention 也未爆）；高频纹理由 SR 阶段局部补全，SR 的输入条件是上采样的低分辨率图加文本。与 7.6 §1 的 LDM 分工对照着看：LDM 用 VAE 把「感知压缩」从扩散手里拿走、单模型在 latent 空间一次到位；Imagen 留在像素空间、把「分辨率递进」拆给级联。同一笔复杂度账，一个压进表示空间，一个摊进模型链。

**误差隔离：noise conditioning augmentation。** 级联的结构性风险是伪影放大：base 输出的瑕疵若被 SR 当作忠实结构，会逐级放大。Imagen 的对策（论文口径）：训练 SR 时对低分辨率条件图加少量高斯噪声、并把增广的噪声水平作为额外条件给模型；推理时输入干净低分辨率图、条件拨到最低增广档。效果：训练分布覆盖了 base 输出的不完美，SR 学会「不全信输入的高频、按文本自己重画细节」。这是级联系统的标准接口设计——下游按「上游输出有噪」建模，而非假设上游完美。

**T5-only。** text encoder 只用冻结的 T5-XXL，不配 CLIP。机制依据与消融结论已在 7.6 §4 核过：纯文本语料的监督密度、逐 token 的去噪目标、长尾词覆盖，三条都打在 CLIP 对比目标的弱项上；Imagen 的核心消融结论是放大 text encoder（T5-L → XL → XXL）的收益大于放大 U-Net 本身——文本理解是那个阶段的瓶颈变量。

**dynamic thresholding。** 像素空间有一个 latent 空间没有的硬约束： $x_0$ 的值域是 $[-1, 1]$。大 CFG 权重下 score norm 膨胀（7.4 §4.3 的失效机制）直接把 $\hat x_0$ 推出值域，static clip 又把出界变成贴边堆积。Imagen 的 dynamic thresholding（推导与对比在 7.4 §4.4）按当前步样本自身的分位数自适应压缩：

$$
\hat x_0 \leftarrow \frac{\mathrm{clip}(\hat x_0, -s, s)}{s}, \qquad s = \mathrm{percentile}(|\hat x_0|, p) \qquad (4)
$$

**符号**： $p$：高档位分位数；仅当 $s > 1$ 时动手，除以 $s$ 保持相对对比度、避免贴边。它是 Imagen 能用大 $w$ 换文本对齐的前提手术。对照：latent 扩散（SDXL/SD3）的 latent 无硬值域，thresholding 的必要性下降，rescaled CFG（7.4 §4.4）是那边更常用的尺度修正。

**附带产物。** Imagen 论文同时提出 DrawBench 评测集（论文口径）：按计数、空间关系、文字渲染等结构化类别组织的 prompt 集，此后成为 T2I 对比的公共基准之一——本章 §9 对照表的质量证据有相当一部分落在这类基准的口径上。

## 4. AR 生图 I：LlamaGen —— 图像 = 另一种语言

先摆结果再讲论证。LlamaGen（[Sun et al., 2024](../references.md)，arXiv:2406.06525，[A]，走读锚点 [FoundationVision/LlamaGen](../references.md)）在 ImageNet $256^2$ 类条件生成上跑到 FID 2.18（3.1B 参数、CFG，论文口径，2026-11-12）——低于同基准 DiT-XL/2-G 的 2.27 与 LDM-4 的 3.60（7.6 §6.3 的同口径数字）。用的主干是不带任何视觉归纳偏置的 Llama（论文摘要口径：「vanilla autoregressive models, e.g., Llama, without inductive biases」）。

「图像 = 另一种语言」不是比喻，是一条逐项核验的论证链：

**(1) 样本空间同构。** VQ tokenizer（6.3 §3–§4 的 VQGAN 家族：f16 下采样、codebook $K = 16384$）把 $256^2$ 图压成 $16 \times 16 = 256$ 个 code 的网格，raster 顺序展平成序列 $z = (z_1, \ldots, z_{256})$， $z_i \in \lbrace1, \ldots, K\rbrace$。这与 1.1 的文本样本空间——有限字母表上的序列——逐点同构。

**(2) 目标函数同式。** 同样的链式分解、同样的精确负对数似然：

$$
p_\theta(z) = \prod_{i=1}^{N} p_\theta(z_i \mid z_{<i}), \qquad L = -\thinspace\mathbb{E}_{z \sim p_{data}} \sum_{i=1}^{N} \log p_\theta(z_i \mid z_{<i}) \qquad (5)
$$

**符号**： $N$：token 数（ $256^2$ 档为 256）； $z_{<i}$：raster 前缀。没有 ELBO 下界、没有回归替代目标——7.1 对照表里 AR 行的「精确似然」在图像上原样成立，这是它对 diffusion 的硬优势（diffusion 侧精确似然要走 PF-ODE 的 trace 积分，7.3 §6.3 / 7.5 §2，贵到不实用）。

**(3) 架构原样复用。** 2.8 §1 的标准配方整块搬来：Pre-LN RMSNorm + RoPE + SwiGLU 的 decoder-only block，无卷积、无 2D 位置编码、无视觉专属组件。规模档 111M → 3.1B。

**(4) 基建零改动。** KV cache（2.4）、温度/top-k/top-p 采样、CFG（条件与无条件两次前向，与 7.4 §4.2 同款构造，只是 score 换成 logits 差）、serving 栈——论文报告接入 vLLM 后吞吐数倍提升（论文口径）。P2 积累的全部推理工程直接变成图像生成的工程。

**代价账。** raster 顺序的 $N$ 步严格串行： $256^2$ 是 256 步、 $512^2$ 是 1024 步。KV cache 省的是重复计算，省不掉依赖链；同质量 diffusion 采样在 20–50 步量级且每步是全图并行前向（7.4 §1 的 NFE 账）。AR 的串行步数与单 token 信息量是绑定的——这正是下面两节（VAR 换顺序、Infinity 换词表）与 mask-predict 族（换解码范式）各自的动机。

## 5. AR 生图 II：VAR —— next-scale prediction

先问 raster 展平丢了什么。位点 $(i, j)$ 与 $(i, j+1)$ 在序列里相邻，但 $(i, j)$ 与 $(i+1, j)$ 隔了 $w$ 个位置——2D 邻接关系在 1D 化后只剩一半；更要命的是前缀条件的信息不完备：预测右下角时，右半张与下半张图的「历史」根本不存在，模型只能凭左上的 L 形邻域外推全局结构。VAR（[Tian et al., 2024](../references.md)，arXiv:2404.02905，[A]，走读锚点 [FoundationVision/VAR](../references.md) 的 `models/var.py`、`models/vqvae.py`）的答法不是换模型，而是换「下一个」的定义：next-token 改成 **下一尺度预测（next-scale prediction）**。

**形式化。** tokenizer 是多尺度 VQ（残差量化结构同 6.3 §5 的 RVQ 家族，共享 codebook，VAR 档词表 $V = 4096$）：图像被编码成 $K$ 个逐级加细的 token map $r_1, \ldots, r_K$， $r_k \in \lbrace1, \ldots, V\rbrace^{h_k \times w_k}$， $256^2$ 档的尺度序列 $(h_k, w_k) \in \lbrace1, 2, 3, 4, 5, 6, 8, 10, 13, 16\rbrace$（ $K = 10$）；重建侧把各尺度 map 查表、上采样对齐后相加（tokenizer 账 recap 6.3，此处不重推）。生成分解按尺度自回归：

$$
p(r_1, \ldots, r_K) = \prod_{k=1}^{K} p(r_k \mid r_{<k}) \qquad (6)
$$

**符号**：条件 $r_{<k} = (r_1, \ldots, r_{k-1})$：所有更粗尺度的**完整** map； $r_k$ 内部全部 $h_k w_k$ 个位点并行预测。注意力 mask 因此是 block 因果形：尺度间因果（第 $k$ 块只看前 $k-1$ 块）、尺度内双向。

```
r_1    r_2    r_3              r_10
1×1 →  2×2 →  3×3 →  …  →    16×16        共享 codebook V = 4096
└──────── 条件前缀 = 全部更粗尺度的完整 map ────────┘
          p(r_k | r_{<k})：r_k 内 h_k·w_k 个位点一步并行
重建：ẑ = Σ_k upsample(r_k 的查表向量)
```

**raster vs scale 的信息结构论证。** (a) 条件完备性：每步的条件是整张图的粗粒度全貌——全局构图在 $r_1$（ $1 \times 1$）一步定死，之后逐步补细节；raster 在任何一步都看不到全图。(b) 2D 结构保留：步内并行 + 双向注意力，token map 的空间邻接不被展平破坏。(c) 依赖链长度： $256^2$ 档总 token 数 $\sum_k h_k w_k = 680$，但串行步数只有 $K = 10$——raster 的 256 步压到 10 步，且每步是一次完整的并行前向。注意这个顺序与扩散同构：coarse-to-fine 正是 7.2 §2.3 频谱读法里 $t: 1 \to 0$ 的进程，VAR 把扩散的时间轴离散化成了尺度轴——它兼有 AR 的精确似然与扩散的「先全局后局部」结构。

**scaling law。** VAR 论文给出 test loss 与 FID 随参数量（至 2B）与训练算力的幂律拟合（论文 Figure 口径，2026-11-12）——图像 AR 生成第一次接入 1.3 的 scaling-law 账，且幂律形态干净；VAR-2B（depth 30）在 ImageNet $256^2$ 报 FID 1.73（VAR-d30-re，rejection sampling 变体；裸 d30 为 1.92，均论文口径），压过同基准全部扩散与 GAN 对照。附带品：由于每步条件是完整粗图，in-painting / out-painting 变成「改条件前缀」的特例，论文演示了免微调的 zero-shot 编辑（论文口径）。

## 6. AR 生图 III：Infinity —— bitwise tokenizer 把词表推向 $2^d$

先算一笔输出头的账。index-wise 分类器对词表 $K$ 的每个 code 输出一个 logit：参数 $K \times d_{model}$、softmax 归一化随 $K$ 线性变贵；而 tokenizer 的保真上限恰恰随 $K$ 涨（6.3 §5 的容量账：量化半径随词表缩小）。LFQ（6.3 §6）已经把 codebook 变成隐式顶点集 $\lbrace0, 1\rbrace^d$—— $K = 2^d$ 随位数 $d$ 指数涨，不需要查表；但预测侧若仍做 index-wise softmax， $2^d$ 个 logit 会把输出头直接撑爆。Infinity（[Han et al., 2024](../references.md)，arXiv:2412.04431，[A]，走读锚点 [FoundationVision/Infinity](../references.md)）把分类器也改成 bitwise：每个 token 视为 $d$ 位二进制向量 $(b_1, \ldots, b_d)$，逐位独立二分类：

$$
p_\theta(z \mid \cdot) = \prod_{j=1}^{d} \sigma\negthinspace\left( w_j^{\top} h \right)^{b_j} \left( 1 - \sigma\negthinspace\left( w_j^{\top} h \right) \right)^{1 - b_j} \qquad (7)
$$

**符号**： $h$：transformer 在该位点的输出向量； $w_j \in \mathbb{R}^{d_{model}}$：第 $j$ 位的二分类头； $\sigma$：sigmoid； $b_j \in \lbrace0, 1\rbrace$：目标位。参数量 $O(d \cdot d_{model})$：词表 $2^d$ 随 $d$ 指数膨胀、预测头只随 $d$ 线性膨胀——这就是论文摘要口径的「theoretically scaling the tokenizer vocabulary size to infinity」与「infinite-vocabulary tokenizer & classifier」。代价写在分解里：式 (7) 假设各位条件独立（共享 $h$ 只能近似建模位间相关）， $d$ 加大时逼近误差由主干容量吸收——这是「词表换 bit 数」交易的另一面。

**bitwise self-correction（位级自纠错）。** teacher forcing 的暴露偏差（exposure bias）：训练时前缀全是真 token，推理时前缀是模型自己的预测，误差沿依赖链累积。Infinity 的机制（论文摘要口径）：训练时以一定概率随机翻转输入 token 的若干 bit，人为制造「错误前缀」，让模型学习在被污染的历史下纠正后续预测。这与 7.4 §5 蒸馏族的自一致性思想同源——都把「推理分布 ≠ 训练分布」的落差显式搬进训练目标，只是这里落点是离散 bit 而非连续轨迹。

**系统形态。** 主干沿用 VAR 的 next-scale 框架（论文口径），bitwise 换掉 index-wise 预测头，面向 $1024^2$ 级高分辨率 T2I（论文摘要口径的高分辨率合成主张）。机制链条合起来读：tokenizer 侧 $d$ 可以大胆加深（词表边际成本从指数降到线性）→ tokenizer 保真抬升 → 6.3 §6 的「tokenizer 质量是语言模型视觉路线的杠杆点」判断再次被兑现。

## 7. mask-predict 族：MaskGIT 与 MUSE [B]

先看第三个回答：AR 的串行来自「顺序承诺」——每个 token 一旦生成就不可改。mask-predict 族换个承诺方式：全部位点同时出价、逐轮收敛，每轮只兑现最有把握的一部分。MaskGIT（[Chang et al., 2022](../references.md)，arXiv:2202.04200，[A]）用 8 轮迭代完成一张 $256^2$ 图（论文报告甜点 8–12 轮、更多迭代反而有害；raster AR 要 256 步严格串行），ImageNet $256^2$ FID 6.18（论文口径，2026-11-12）。代价在似然列：每步预测的是「给定未 mask 位点、被 mask 位点的条件分布」，整轮迭代的伪似然乘积不等于联合似然——7.1 对照表里这一格要让给 AR 与 diffusion（ELBO）。

**训练。** BERT 式目标：VQ token 网格（VQGAN f16 tokenizer，6.3 §4 的 $16 \times 16$ 档）随机 mask 一部分，双向 transformer 预测被 mask 位点；训练时的 mask 比例按与解码同一族的 schedule 采样（ $r = \gamma(u)$， $u$ 均匀，论文口径）。

**迭代并行解码。** 从全 mask 网格出发，共 $T$ 轮，第 $t$ 轮三步：(a) 对所有被 mask 位点**并行**预测分布并采样；(b) 给每个刚预测的位点算置信度（预测概率，可加温度噪声调多样性）；(c) 置信度最低的一批重新 mask（confidence remasking，置信度重掩码），留下的 mask 数由 schedule 定：

$$
n_t = \left\lceil \gamma\negthinspace\left( \tfrac{t}{T} \right) \cdot N \right\rceil, \qquad \gamma(t) = \cos\negthinspace\left( \frac{\pi t}{2} \right) \qquad (8)
$$

**符号**： $n_t$：第 $t$ 轮后仍被 mask 的位点数； $N$：总位点数； $\gamma$ 从 $\gamma(0) = 1$ 单调降到 $\gamma(1) = 0$。 $\cos$ 形的读法：前期 $\gamma$ 降得快、每轮大批兑现（此时上下文少、只有高置信位点敢定）；后期降速放缓、小步精修（低置信位点留到上下文充分时再定）。机制上与 AR 的顺序承诺相反：AR 早步的错误没有撤回通道，mask-predict 每轮都保留「收回重出」的权利。

```
轮次   t=0        t=1         t=2                  t=T
      ■■■■■■■    ■▓▓■▓■▓     ▓▓▓■▓▓▓      …      ▓▓▓▓▓▓▓    ■ = mask
      ■■■■■■■ →  ■■▓■■▓■  →  ▓■▓▓▓▓▓  →   …  →   ▓▓▓▓▓▓▓    ▓ = 已兑现
      n = N      n = ⌈γN⌉    n = ⌈γN⌉             n = 0
每轮：全位点并行预测 → 按置信度保留 n_t 个继续 mask
```

**MUSE [B]。** MUSE（[Chang et al., 2023](../references.md)，arXiv:2301.00704，[B]——Google 未开源，`google-research/muse` 404 实测 2026-11-12，不作代码锚点，机制按论文公式讲、实现细节不断言）把 mask-predict 装进 T2I 系统：T5-XXL 文本条件拼进序列，base $256^2$ masked transformer + token 空间级联超分到 $512^2$——与 Imagen 的级联同位，但每级是 mask-predict 而非 diffusion。步数优势直接兑现成延迟（论文表 3 口径）：MUSE $512^2$ 出图 1.3 秒、 $256^2$ 出图 0.5 秒（TPUv4）；同表的 Imagen $256^2$ 为 9.1 秒（同硬件）、SD v1.4（LDM 50 步） $512^2$ 为 3.7 秒（A100 口径）——数字跨硬件、跨分辨率，只宜读作量级证据：并行解码把单图延迟压进秒级，而迭代去噪系统在十秒级。同一时代的三个系统恰好各占本表一族，这组数字是三路线延迟结构差异的直接证据。

## 8. Emu3：图、视、文全 token 统一的账

LlamaGen 证明了「AR 主干够」、VAR 证明了「顺序可以换」、Infinity 证明了「词表可以推到 $2^d$」；Emu3（[Wang et al., 2024](../references.md)，arXiv:2409.18869，[A]，走读锚点 [baaivision/Emu3](../references.md)）走的是最后一步：模态也可以并。它用同一个 next-token 目标从头训练一个 8B transformer，输入混合文本 BPE token、图像与视频的离散 token（SBER-MoVQGAN 系 visual tokenizer，codebook 32768、空间 f8： $512^2$ 图 = $64 \times 64 = 4096$ token；视频加时域下采样，论文口径）——生成与理解统一在一个 softmax 下，论文摘要主张无 diffusion、无组合架构即在生成与感知任务上超过 SDXL 等任务专用模型（摘要口径，2026-11-12）。

「统一」的含金量拆成三条账：

**(a) 接口账。** 单 softmax 覆盖文本 + 32768 个视觉 code + 控制符；7.6 §3 的 cross-attention、7.6 §7.1 的双流 MMDiT 这类「跨模态接口」整体消失——模态对齐不再由架构显式搭桥，而由共享序列里的 next-token 监督自发学成。系统复杂度从「多组件组合」塌缩回「一个 transformer」。

**(b) 序列账。** 一张 $512^2$ 图 4096 token，对一段几十 token 的 prompt 是两个数量级的碾压——视觉统治序列长度、KV cache 与解码延迟；视频更甚。含义直接：P2 的全部长上下文与推理优化（KV cache 2.4、稀疏/线性注意力 P4、投机采样）从「文本基建」变成「生成质量与成本的第一变量」，这也是 AR 视觉路线对 infra 侧的真实拉力。

**(c) tokenizer 账。** 6.3 §6 的判断在此完全落地：离散 tokenizer 的重建保真是全系统的质量上限——LLM 侧的任何改进都穿不过 tokenizer 的有损瓶颈；Emu3 把视频也压进同一 tokenizer，tokenizer 的每一笔重建损失摊给三个模态，tokenizer 训练的优先级被推到系统最高位。对照扩散侧：LDM 的两阶段分工（7.6 §1）里 VAE 同样是上限，但连续 latent 的瓶颈软得多（f8 16ch 的保真对多数任务已足够）——离散统一付出的信息瓶颈代价，是 Emu3 路线必须持续偿还的债。

四步合看，AR 路线各卸掉一个历史包袱：主干（LlamaGen）、顺序（VAR）、词表（Infinity）、模态（Emu3）；代价始终在同一处结算——串行或准串行解码的延迟，与离散化的信息瓶颈。

## 9. 对照总表：diffusion vs AR vs mask-predict

三族路线的结构性差异收进一张表（似然与步数列的推导出处见 7.1–7.5；质量列一律论文口径、2026-11-12 锁定，跨基准数字不可互比）：

| 维度 | Diffusion / FM（SD3、SDXL、Imagen） | AR raster（LlamaGen） | AR next-scale（VAR、Infinity） | mask-predict（MaskGIT、MUSE） |
| --- | --- | --- | --- | --- |
| 似然 | ELBO 下界；精确似然走 PF-ODE trace（7.3 §6.3），贵到不实用 | 精确 NLL（式 (5)） | 精确 NLL（式 (6)） | 无精确似然：迭代伪似然 ≠ 联合似然 |
| 串行步数（ $256^2$ 档） | 20–50 步 solver（7.4 §3）；蒸馏后 1–4 步（7.4 §5.6） | 256 步严格串行；KV cache 省重算不省依赖链 | $K = 10$ 步 | $T = 8$ 轮（甜点 8–12，论文口径），轮内全并行 |
| 每步形态 | 全图并行前向 | 单 token 前向 + cache | 一整张 token map 并行 | 全位点并行预测 |
| 可控性 / 编辑 | 连续 latent：CFG（7.4 §4）与 img2img / inpainting 生态最完整 | 条件 = 前缀；交错图文天然；编辑靠补全 | coarse-to-fine；in/out-painting 免微调（论文口径） | mask 即编辑：inpainting 是原生形态 |
| 与 LLM 组合性 | 外挂 text encoder + cross-attn / MMDiT 接口（7.6 §3/§7）；权重不共享 | 2.8 block 原样复用；serving/KV/投机采样零改动 | 同左（LLM 基建复用） | 双向 transformer：decoder-only LLM 的 KV cache/投机采样不直接适用 |
| 质量现状证据（论文口径，2026-11-12） | SD3 8B 人类偏好超 SDXL 系；Imagen 立 DrawBench [B] | ImageNet $256^2$ FID 2.18 @ 3.1B（CFG） | ImageNet $256^2$ FID 1.73 @ 2B（VAR-re 变体；裸 d30 = 1.92） | MaskGIT FID 6.18；MUSE $512^2$ 1.3s / $256^2$ 0.5s（TPUv4），同表 Imagen $256^2$ 9.1s、SD v1.4 $512^2$ 3.7s（A100）[B] |

**选型读法。** (a) 要精确似然与 LLM 基建复用 → AR 两型；(b) 要成熟控制生态与少步采样 → diffusion/FM（步数短板由 7.4 §5 的蒸馏族补）；(c) 要并行解码与编辑原生 → mask-predict。质量列说明一个未固化的格局：单看 ImageNet FID，AR 两型已压过同基准扩散（2.18 / 1.73 对 2.27）；开放 T2I 的人类偏好战场（DrawBench/GenEval 类口径）扩散仍占多数席位，但 Infinity 与 Emu3 的论文口径已把高分辨率 T2I 拉进 AR 射程。分野的变量在 tokenizer 与顺序设计，不在「扩散 vs AR」的口号上。

## 总结

| 对象 | 公式/数字 | 要点 |
| --- | --- | --- |
| SD3 配方汇总 | §1 表 | 16ch VAE + MMDiT + rectified flow + logit-normal + shift + 三 encoder + CFG；每项出处 6.3/7.4/7.5/7.6 |
| SDXL micro-conditioning | 式 (1)(2) | 原尺寸/裁剪坐标/目标尺寸 Fourier 嵌入进 $\tau$；标注替代丢弃，推理把好角 |
| SDXL base+refiner | base U-Net 2.6B（论文口径） | 沿 $t$ 轴的专家分工（同构 4.5 MoE 沿 token 轴）；refiner 只吃低噪声段 |
| Imagen 级联 | 式 (3) | $64 \to 256 \to 1024$ 像素空间三因子分解；noise conditioning augmentation 做误差隔离；T5-only（7.6 §4 消融） |
| dynamic thresholding | 式 (4) | 分位数 $s$ 自适应压回 $[-1,1]$；像素空间大 $w$ 的前提（推导在 7.4 §4.4） |
| LlamaGen | 式 (5)；FID 2.18 @ 3.1B（论文口径） | VQ f16、 $K = 16384$、256 token；2.8 配方原样复用；精确似然 + P2 基建零改动 |
| VAR next-scale | 式 (6)；FID 1.73 @ 2B（-re 变体；裸 d30 = 1.92，论文口径） | $K = 10$ 尺度、680 token、串行 10 步；coarse-to-fine 同构扩散频谱方向；scaling 幂律 |
| Infinity bitwise | 式 (7) | 词表 $2^d$、分类头 $O(d \cdot d_{model})$；bit 翻转自纠错治 exposure bias |
| MaskGIT | 式 (8)；FID 6.18（论文口径） | $\gamma(t) = \cos(\pi t / 2)$、confidence remasking；8 轮对 256 步（甜点 8–12 轮，论文口径） |
| MUSE [B] | $512^2$ 1.3s / $256^2$ 0.5s（TPUv4）；同表 Imagen $256^2$ 9.1s、SD v1.4 $512^2$ 3.7s（A100）——跨硬件跨分辨率的量级证据（论文表 3 口径） | mask-predict 进 T2I 两级级联（ $256^2$ base + $512^2$ superres）；Google 未开源（404 实测），不作代码锚点 |
| Emu3 | $512^2$ = 4096 token、codebook 32768（论文口径） | 图视文统一 next-token；接口账消失、序列账视觉统治、tokenizer 账成系统上限 |
| 三路线对照 | §9 表 | 似然 / 步数 / 每步形态 / 可控性 / LLM 组合性 / 质量证据 |

## 练习

1.（推导）micro-conditioning 的条件语义：设训练集中「断头构图」样本全部满足 $c_y > 0$。写出推理时把式 (2) 中 $(c_y, c_x)$ 设为 $(0, 0)$ 所对应的条件分布，并说明它等价于在「无断头」子分布上采样。再讨论反向情形：若训练数据已全量清洗（无任何裁剪样本），micro-conditioning 的 $(c_y, c_x)$ 通道退化成什么？这解释了为什么该技巧的价值随数据质量过滤策略反向变化。
2.（计算）步数与并行度联合账：验证 VAR $256^2$ 档尺度序列 $\lbrace1, 2, 3, 4, 5, 6, 8, 10, 13, 16\rbrace$ 的总 token 数为 680。对四条路线——raster AR（256 步，每步 1 token）、VAR（10 步，第 $k$ 步 $h_k w_k$ 位点）、MaskGIT（ $T$ 轮，每轮至多 $N = 256$ 位点）、DDIM 25 步（每步全 latent）——列表给出「串行步数 × 峰值并行位点数」，并指出哪条路线的「串行深度/总 token 数」比最低。
3.（推导）bitwise 分解的表示力：设词表 $K = 2^d$、 $d_{model} = 4096$，计算 $d = 14$（LlamaGen 档）、 $d = 15$（Emu3 档）、 $d = 32$ 时 index-wise softmax 头与 Infinity bitwise 头的参数量。再证明：式 (7) 的逐位独立分解能精确表示任意联合分布 $p(z)$ 的充要条件是各位在给定前缀下条件独立；并说明 Infinity 中共享表示 $h$ 为什么只能近似建模位间相关（提示： $h$ 是 $d_{model}$ 维瓶颈）。
4.（代码）MaskGIT 一轮解码：用教学 PyTorch（非优化实现）实现「并行预测 + confidence remasking」单轮：输入 logits $[N, K]$ 与当前 mask 布尔向量，对被 mask 位点采样，按置信度排序保留式 (8) 的 $n_t$ 个继续 mask； $N = 256$、 $K = 1024$、 $T = 12$ 跑完整循环，打印每轮兑现的位点数序列，并验证其近似 $N \cdot (\gamma(t/T) - \gamma((t{+}1)/T))$ 的形状。

## 参考

- [Esser et al., 2024](../references.md)：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis，arXiv:2403.03206，[A]（key: sd3）——§1 配方汇总、scaling 与人类偏好口径；部件推导见 7.5/7.6
- [Podell et al., 2023](../references.md)：SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis，arXiv:2307.01952，[A]（key: sdxl）——式 (1)(2) micro-conditioning、base+refiner 两阶段、2.6B U-Net 口径
- [Saharia et al., 2022](../references.md)：Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding，arXiv:2205.11487，[B]（key: imagen）——式 (3)(4) 级联分解与 dynamic thresholding、T5-only、noise conditioning augmentation、DrawBench（无官方代码，公式引论文）
- [Sun et al., 2024](../references.md)：Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation，arXiv:2406.06525，[A]（key: llamagen）——式 (5)「图像 = 另一种语言」论证、FID 2.18、tokenizer 档
- [Tian et al., 2024](../references.md)：Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction，arXiv:2404.02905，[A]（key: var）——式 (6) next-scale 分解、尺度序列与 680 token 账、scaling 幂律、FID 1.73（-re 变体；裸 d30 = 1.92）、zero-shot 编辑
- [Han et al., 2024](../references.md)：Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis，arXiv:2412.04431，[A]（key: infinity）——式 (7) bitwise 分类器、无限词表主张、bitwise self-correction
- [Chang et al., 2022](../references.md)：MaskGIT: Masked Generative Image Transformer，arXiv:2202.04200，[A]（key: maskgit）——式 (8) mask schedule 与 confidence remasking、FID 6.18
- [Chang et al., 2023](../references.md)：Muse: Text-To-Image Generation via Masked Generative Transformers，arXiv:2301.00704，[B]（key: muse）——mask-predict 进 T2I 系统与延迟对照（Google 未开源，不作代码锚点）
- [Wang et al., 2024](../references.md)：Emu3: Next-Token Prediction is All You Need，arXiv:2409.18869，[A]（key: emu3）——全 token 统一的接口/序列/tokenizer 三笔账
- [Yu et al., 2023](../references.md)：Language Model Beats Diffusion — Tokenizer is Key to Visual Generation，arXiv:2310.05737，[A]（key: magvit-v2）——LFQ 与「tokenizer 是杠杆点」判断（6.3 §6 完整拆解）
- [Esser et al., 2020](../references.md)：Taming Transformers for High-Resolution Image Synthesis，arXiv:2012.09841，[A]（key: vqgan）——VQ tokenizer 家族（6.3 §4）
- [Stability-AI/sd3.5](../references.md)：SD3.5 官方推理代码（`mmditx.py`、`sd3_impls.py`），[A]（key: sd3.5-repo）——§1 走读锚点
- [Stability-AI/generative-models](../references.md)：SGM 训练代码（`sgm/modules/diffusionmodules/`），[A]（key: sgm-repo）——§1/§2 训练侧走读锚点
- [stabilityai/stable-diffusion-xl-base-1.0](../references.md)：SDXL HF 权重与 config（`StableDiffusionXLPipeline`、双 text encoder），[A]（key: sdxl-hf）——§2 条件通路锚点
- [FoundationVision/LlamaGen](../references.md)：LlamaGen 官方实现，[A]（key: llamagen-repo）——§4 走读锚点
- [FoundationVision/VAR](../references.md)：VAR 官方实现（`models/var.py`、`models/vqvae.py`），[A]（key: var-repo）——§5 next-scale 解码与多尺度 tokenizer 走读锚点
- [FoundationVision/Infinity](../references.md)：Infinity 官方实现，[A]（key: infinity-repo）——§6 走读锚点
- [baaivision/Emu3](../references.md)：Emu3 官方实现，[A]（key: emu3-repo）——§8 走读锚点
