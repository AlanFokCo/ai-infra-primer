# 9.1 视频 VAE（Video VAE）

> 一段 5 秒 720p 视频是 $3.3 \times 10^{8}$ 个数值，直接进 DiT 的 attention 不可行；Video VAE 的职责是把它压成 $t \times h \times w \times c$ 的连续 latent（通用档 $(\tau, s, s) = (4, 8, 8)$、 $c = 16$，数值体积缩约 48 倍），并用「时间维因果卷积 + 首帧只做空间压缩」的结构让图像与视频共享同一套 latent——全章按「数据账 → 3D causal conv 结构与因果性公式 → 典型配置与 token 账 → 各家对照 → 压缩率边界（CogVideoX ablation 与 DC-AE 式深压缩）→ 720p 实例内存账」展开 | 前置：6.3（VAE/ELBO 式 (3)(8)、scaling factor 式 (24)、DC-AE §7.3）、7.6（latent 空间两阶段分工 §1、scaling factor 的扩散语义 §2、DC-AE f32 §8.2）、2.4（KV cache——feature caching 的同构物）、2.7（RMSNorm） | 锚点：CogVideoX [A]、HunyuanVideo [A]、Wan [A]（论文三家）；zai-org/CogVideo、Tencent-Hunyuan/HunyuanVideo、Wan-Video/Wan2.1（官方 repo）[A]；cogvideox-config / hunyuanvideo-vae-config / autoencoderkl-wan（config 与 diffusers 实现实测）[A]；Wan2.2（高压缩 VAE，官方 README）[A]；LTX-Video [A]、StepVideo-T2V [A]、Open-Sora 2.0 [A]、Open-Sora-Plan / WF-VAE [A]；Mochi [C]

记号约定（本 Part 统一）：原始视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$；latent $z \in \mathbb{R}^{t \times h \times w \times c}$；压缩率记 $(\tau, s, s)$——时间 $\tau$ ×、空间 $s \times s$；token 数 $N = t \cdot h \cdot w$（patchify 等效边长 1 的统一口径；各家在 latent 之上的 patchify 逐家不同，是 9.2 的主题）。

## 1. 视频数据账： $T \times H \times W \times 3$ 的乘法

一段 5 秒、720p（ $1280 \times 720$）、24 fps 的视频含 $T = 120$ 帧、 $1.1 \times 10^{8}$ 个像素，写成张量账：

$$|V| = T \cdot H \cdot W \cdot 3 = 120 \times 720 \times 1280 \times 3 \approx 3.32 \times 10^{8} \qquad (1)$$

**符号**： $T$：帧数； $H \times W$：空间分辨率；3：RGB 通道。uint8 存储约 332 MB，bf16 约 664 MB；一分钟同规格 uint8 约 4.0 GB。

这组数字本身不稀奇，稀奇的是它进 attention 的方式。扩散主干（9.2 的时空 DiT）的成本由 token 数 $N$ 决定，full attention 随 $N^{2}$ 长。若不做任何压缩、把像素网格直接当 token（每 $8 \times 8$ 像素块一个 token 的粗放假设），这段 5 秒视频的 $N = 120 \times 90 \times 160 = 1{,}728{,}000$， $N^{2} \approx 3 \times 10^{12}$——单层 attention 的得分矩阵按 bf16 计约 6 TB。压缩因此不是优化项而是可行性前提：先把 $V$ 编码进 latent $z$，扩散模型只在 $z$ 的空间里工作。

与 8.4 的理解侧对照可以看清生成侧为什么更苛刻：视频理解可以抽稀帧（1 min @ 1 fps × 256 token/帧 = 15,360 token，8.4 式 (1) 的账；更高帧率档直接吃掉 128K context 预算的过半），但生成必须逐帧产出——输出端没有「跳帧」的自由，latent 帧密度由压缩率钉死，帧率与时长动不了。理解侧的省 token 三位置（8.4 §4）对应到生成侧只剩一个：VAE 压缩率本身。这是 Video VAE 在整个系统里的位置：它是 token 预算的唯一闸门。

Video VAE 的数学骨架与 6.3 的图像 VAE 完全一致：对角高斯后验 $q_\phi(z \mid V)$、reparameterization 采样 $z = \mu + \sigma \odot \varepsilon$、目标函数 $L = -\mathbb{E}_{q_\phi}\left[ \log p_\theta(V \mid z) \right] + \beta \cdot \mathrm{KL}\left( q_\phi(z \mid V)\thinspace \Vert\thinspace p(z) \right)$（6.3 式 (8)，推导不重复）。其中 KL 项在视频语境下的角色值得单独点出：它把后验往标准高斯先验上拉，保证「从 $\mathcal{N}(0, I)$ 采样 $z$ → decoder」这条生成路径与 encoder 产出的 latent 同分布——视频 latent 的数值体积比单帧图像大一到两个数量级，标定不良的后果（扩散模型在失配分布上回归）也被同等放大。两阶段分工沿用 7.6 §1 的读法：VAE 负责感知压缩，扩散负责语义生成——LDM 的同预算消融同时给出警示：压缩率存在质量上限，压缩因子过大时 autoencoder 的重建误差封顶整体样本质量。这个上限在视频侧的量化证据见 §5。

latent 形状由压缩率直接写出：

$$t = 1 + \left\lceil \frac{T - 1}{\tau} \right\rceil, \qquad h = \frac{H}{s}, \qquad w = \frac{W}{s} \qquad (2)$$

**符号**：首帧特判（Wan 式约定，机制见 §2.3）：第 0 帧只做空间压缩、独占一个 latent 帧；其余 $T - 1$ 帧每 $\tau$ 帧合成一个。 $\tau = 4$ 时 $T = 81 \to t = 21$， $T = 120 \to t = 31$。

$$N = t \cdot h \cdot w \qquad (3)$$

**符号**： $t, h, w$ 回指式 (2)——latent 的时间帧数与空间分辨率； $N$ 即 DiT 侧 attention 面对的序列长（约定口径，patchify 等效边长 1）。

数值压缩率（取长视频极限、忽略首帧 $+1$）：

$$R = \frac{T \cdot H \cdot W \cdot 3}{t \cdot h \cdot w \cdot c} \approx \frac{3 \tau s^{2}}{c} \qquad (4)$$

**符号**： $c$：latent 通道数。通用档 $(\tau, s, c) = (4, 8, 16)$ 代入得 $R = 48$。

式 (3) 与式 (4) 是两本不同的账：attention 关心 $N$（按 $\tau s^{2}$ 缩），存储与带宽关心数值体积（按 $3 \tau s^{2} / c$ 缩）， $c$ 是两本账之间的兑换率——同 $R$ 下可以用更大的 $c$ 换更小的 $N$。§4 的 WFVAE 行是这个兑换的实例。

还有一个路线选择问题埋在骨架里：为什么视频生成主流用连续 latent，而不是 6.3 §3-§6 的离散 token（VQ-VAE/RVQ 族）？6.3 §7.4 的选择表在视频侧原样成立：离散化的收益（可以直接套 LLM 的 softmax 建模）在视频上被两件事抵消——量化误差在时间维累积成闪烁与漂移，而连续 latent + 扩散的组合已经在图像侧验证过质量上限。离散视频 tokenizer 存在（MagViT-v2，[Yu et al., 2023](../references.md)，[A]，6.3/8.5 已拆；它的首帧特判约定正是 §2.3 的源头），但本章的主角——CogVideoX/HunyuanVideo/Wan 一系——全部走连续 latent + 扩散路线。

## 2. 3D 因果卷积 VAE：结构与时间因果性

Video VAE 与 6.3 图像 VAE 的结构差异归结为一处：全部卷积换成 3D 因果卷积（causal conv3d），下采样栈从空间两维扩到时空三维。这一结构在两家的官方实现与论文里各有物证：HunyuanVideo 的 VAE 实现类名直书为 `AutoencoderKLCausal3D`（[hunyuanvideo-vae-config](../references.md)，[A]）；CogVideoX 侧类名为 `AutoencoderKLCogVideoX`（[cogvideox-config](../references.md)，[A]），其 3D causal VAE 的出处是论文（[Yang et al., 2024](../references.md)，arXiv:2408.06072，[A]）。其余部件原样继承：encoder 末层分出后验参数 $\mu$ 与 $\sigma$（对角高斯，6.3 §2.1），decoder 与 encoder 镜像、输入是采样后的 $z$；变的是卷积核多出一个时间维，以及时间维上的 padding 与下采样规则——后者是 §2.1 的公式对象。

```
V ∈ R^{T×H×W×3}                    例：121×720×1280×3（Wan 式「首帧+120 帧」写法，bf16 ≈ 669 MB）
  │
  │  Encoder：3D 因果卷积级联
  │    空间下采样（stride (1,2,2) 级联）：H→H/8，W→W/8
  │    时间下采样（stride (2,1,1) 级联）：首帧独立；其余 τ=4 帧合 1
  │    通道 3 → c（通用档 16）；归一化用 RMSNorm（Wan-VAE，见 2.7）
  ▼
q(z|V) 的 μ, σ ∈ R^{t×h×w×c}              对角高斯后验（6.3 §2）
  │   z = μ + σ ⊙ ε（reparameterization，6.3 §2.5），× s（scaling factor，§3）
  ▼
z ∈ R^{31×90×160×16}（bf16 ≈ 14.3 MB） ──► DiT 在此空间去噪（9.2）
  │
  │  Decoder：镜像上采样栈（同样只向过去看）
  ▼
V̂ ∈ R^{T×H×W×3}
```

### 2.1 因果卷积公式：时间维的非对称 padding

对时间核长 $k_{t}$、空间核 $k_{h} \times k_{w}$ 的 3D 卷积，因果版本保留空间维的对称 padding、时间维改为只在左端补 $k_{t} - 1$ 帧：

$$y[c_o, t, i, j] = \sum_{c_i} \sum_{a = 0}^{k_t - 1} \sum_{b = 0}^{k_h - 1} \sum_{d = 0}^{k_w - 1} w[c_o, c_i, a, b, d] \cdot \tilde{x}[c_i, t + a, i + b, j + d] \qquad (5)$$

**符号**： $x$：输入特征， $[c_i, T, H, W]$； $\tilde{x}$：补零版本——时间维左侧补 $k_{t} - 1$ 帧、空间维两侧各补 $(k_{h} - 1)/2$ 与 $(k_{w} - 1)/2$（ $k_{h}, k_{w}$ 取奇数），即 $\tilde{x}[\cdot, t', i', j'] = x[\cdot,\ t' - (k_{t} - 1),\ i' - (k_{h} - 1)/2,\ j' - (k_{w} - 1)/2]$，越界处为零； $w$：卷积核； $y$：输出特征。把补零代入求和指标： $y[\cdot, t, i, j]$ 在时间维只读第 $t - k_{t} + 1$ 至第 $t$ 帧、在空间维读以 $(i, j)$ 为中心的 $k_{h} \times k_{w}$ 邻域——时间感受野只向过去延伸（不多不少 $k_{t}$ 帧），空间感受野双向对称，与下文「空间维仍是对称 padding」一致。

注意因果性只加在时间轴上：空间维仍是对称 padding。原因是两个维度的消费方式不同——DiT 对同一帧内的空间位置是双向可见的（9.2 的 full 3D attention 没有空间因果 mask），空间感受野向两个方向延伸没有泄漏问题；而时间轴承载的是「帧序」这个被生成范式（分块自回归、流式）依赖的语义，只有它需要单向约束。

```
k_t = 3 的时间卷积（空间维略去）：

对称 padding（非因果）              因果 padding（左补 k_t−1 = 2 帧）
输入:  [x0][x1][x2][x3][x4]         [· ][· ][x0][x1][x2][x3][x4]
y[2] ←  x1, x2, x3    ← 读到未来    y[2] ←      x0, x1, x2  ← 只见过去与现在
y[0] ←  pad,x0, x1    ← 首帧读未来  y[0] ←      ·,  ·, x0   ← 首帧只含自身
```

### 2.2 为什么必须因果：三个机制理由

**图像 = 单帧视频兼容。** Wan 论文 §4.1.1 原文：「the first frame is only spatially compressed… following MagViT-v2」（[Team Wan, 2025](../references.md)，arXiv:2503.20314，[A]；MagViT-v2 见 [Yu et al., 2023](../references.md)，arXiv:2310.05737，[A]，6.3/8.5 已拆）。机制链条：因果结构下第 0 帧的 latent 只依赖第 0 帧自己（上图右列 $y[0]$），于是一张孤立图像过 encoder 的输出与它作为视频首帧的输出逐位一致——同一套 VAE、同一个 latent 空间服务两种数据。这是图视频混训与 T2I 权重继承（9.3 的训练配方）成立的结构前提。反例：对称 padding 下 $y[0]$ 读进第 1 帧（上图左列），孤立图像没有「第 1 帧」，补零或复制得到的 latent 与视频首帧的 latent 属于两个分布，两种数据无法共享一个生成空间。

**分块流式编解码（feature caching）。** 因果卷积按时间分块推进时，块间只需保留前块末尾 $k_{t} - 1$ 帧的特征作状态——与 2.4 KV cache 同构的「状态 = 有限过去窗口」。状态大小的账：单层因果卷积的跨块状态是 $k_{t} - 1$ 帧特征， $L$ 层堆叠的总状态是各层 $k_{t}^{(i)} - 1$ 帧特征之和——与层数线性、与视频总长无关。diffusers 的 Wan-VAE 实现即此结构：`WanCausalConv3d` 自定义 3D 因果卷积 + 时间维因果 padding + feature caching（[autoencoderkl-wan](../references.md)，[A]）。编解码内存从 $O(T)$ 降到 $O(\text{chunk})$——长视频可编解码的前提；decoder 侧对称成立，生成完的 latent 可以逐块解成像素流式吐出。生成侧的分块推进范式（9.3）在 VAE 侧也靠这个接口。

**latent 帧序无未来泄漏。** latent 帧 $j$ 只含第 $j\tau$ 帧及之前的输入信息（具体覆盖区间依赖层栈，是练习 2 的对象），任何「以过去 latent 为条件推进下一块」的生成范式因此良定义；非因果 VAE 下第 $j$ 块 latent 混入第 $j + 1$ 块的像素，自回归条件被污染。

### 2.3 首帧特判与 latent 时间长度

式 (2) 的 $t = 1 + \lceil (T-1)/\tau \rceil$ 就是首帧特判的代数形态。Wan 论文的等价写法：输入记作 $(1 + T) \times H \times W \times 3$（首帧 + $T$ 帧， $T$ 取 4 的倍数），latent 时间维 $= 1 + T/4$；统一口径的 $T = 120$ 对应论文口径的 $121 = 1 + 120$ 帧、 $t = 1 + 120/4 = 31$。在因果 padding 的下采样栈里这个特判是自然落出来的：时间维 stride 级联逐级左补零，帧 0 的输出位置始终只含帧 0 的贡献， $\tau$ 帧合 1 的分组从帧 1 起算。解码侧镜像：latent 帧 0 解出第 0 帧图像，latent 帧 $j \ge 1$ 各解出 $\tau$ 帧——图像与视频的 decoder 也是同一个。

实现细节一则：Wan-VAE 把图像 VAE 通行的 GroupNorm 换成 RMSNorm（2.7；Wan 论文 §4.1.1 口径）。

## 3. 典型配置与 token 账

2024–2025 年的主流通用档是 $(\tau, s, s) = (4, 8, 8)$、 $c = 16$，三家开源模型的论文与 config 互证（2026-11-13 锁定）：

- **CogVideoX**（2B/5B 档）：论文的 VAE ablation 选定空间 8×8、时间 4×（§5 的边界证据）；HF config 实测 `temporal_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.7`（[cogvideox-config](../references.md)，[A]；官方代码 [zai-org/CogVideo](../references.md)，[A]，原 THUDM/CogVideo 已 301 重定向）。
- **HunyuanVideo**（13B+）：官方 README 原文「compression ratios of video length, space, and channel to 4, 8, and 16」——时间 4×、空间 8×、通道 3→16（[Kong et al., 2024](../references.md)，arXiv:2412.03603，[A]）；HF config 实测 `AutoencoderKLCausal3D`、`time_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.476986`（[hunyuanvideo-vae-config](../references.md)，[A]）。
- **Wan**：论文 §4.1.1，latent $[1 + T/4,\ H/8,\ W/8]$、通道 3→16（[Team Wan, 2025](../references.md)，[A]；官方代码 [Wan-Video/Wan2.1](../references.md)，[A]）。

token 账两例（式 (2)(3) 直接代入）：

- Wan 式 81 帧： $t = 1 + 80/4 = 21$ latent 帧——「81 帧 → 21 latent 帧」是 $(4, 8, 8)$ 档的标准换算。
- CogVideoX 档：49 帧、 $480 \times 720$ → latent $13 \times 60 \times 90 \times 16$（config 实测 sample_frames 49、特征图 60×90）， $N = 13 \times 60 \times 90 = 70{,}200$。全链核验：原始 $49 \times 480 \times 720 \times 3 \approx 5.08 \times 10^{7}$ 个数值 → latent $1.12 \times 10^{6}$ 个，精确压缩比 45.2×（极限 48×，差额同首帧 $+1$）。

时长方向的成本曲线可以用一行一行地列出来看清：720p（ $h \times w = 90 \times 160$，每个 latent 帧 14,400 token）、 $(4, 8, 8)$ 档下——

| $T$（帧） | $t$（式 (2)） | $N = t \times 14{,}400$ | $N^{2}$ |
| --- | --- | --- | --- |
| 49 | 13 | 187,200 | $3.50 \times 10^{10}$ |
| 81 | 21 | 302,400 | $9.14 \times 10^{10}$ |
| 120 | 31 | 446,400 | $1.99 \times 10^{11}$ |
| 240 | 61 | 878,400 | $7.72 \times 10^{11}$ |

$N$ 随 $T$ 线性长（首帧之后每 4 帧加 $1.44 \times 10^{4}$ token），attention 的二次项随 $T^{2}$ 长——时长翻倍的 attention 代价是 4 倍，这是长视频贵的代数根源，也是 9.2 的拓扑取舍与 9.3 的分块路线各自的出发点。

scaling factor 是 VAE 与扩散之间的分布标定接口： $z' = s \cdot z$ 把 latent 的逐分量统计钉到标定点，换 VAE 只需重测 $s$、扩散侧的 noise schedule 不动；错误尺度的代价是 log-SNR 轴整体平移（机制推导见 7.6 §2 式 (2)(3)）。0.7 与 0.476986 这类常数的出处是各家 config 而非论文——换权重文件时以 config 实测为准。

## 4. 各家 Video VAE 对照

把八家九档的参数摆进同一张表，分歧只在 $(\tau, s, s)$ 与 $c$ 两个旋钮上（2026-11-13 锁定；「—」= 事实源未登记，不推测）：

| 模型（VAE） | $(\tau, s, s)$ | $c$ | $R$（式 (4)） | scaling factor | 出处 |
| --- | --- | --- | --- | --- | --- |
| CogVideoX-2B/5B | $(4, 8, 8)$ | 16 | 48× | 0.7 | 论文 ablation + HF config [A] |
| HunyuanVideo-13B | $(4, 8, 8)$ | 16 | 48× | 0.476986 | 官方 README + HF config [A] |
| Wan2.1（1.3B/14B） | $(4, 8, 8)$ | 16 | 48× | — | 论文 §4.1.1 [A] |
| Wan2.2-VAE（TI2V-5B） | $(4, 16, 16)$ | 48 | 64× | — | 官方 README（无论文）+ HF config 实测 [A] |
| LTX-Video | $(8, 32, 32)$ | 128（由 1:192 反推） | 192× | — | 论文摘要 [A] |
| StepVideo-T2V-30B | $(8, 16, 16)$ | — | — | — | 论文摘要 [A] |
| Mochi 1-10B | $(6, 8, 8)$ | 12 | 96×（README 称 128×） | — | repo README [C] |
| Open-Sora-Plan v1.5（WFVAE） | $(8, 8, 8)$ | 32 | 48× | — | repo README（v1.5.0）+ WF-VAE 论文 [A] |
| Open-Sora 2.0-11B | Video DC-AE（§5） | — | — | — | 论文 [A] |

读表的五条：

1. **通用档 $(4, 8, 8, 16)$ = 48× 是质量与收敛的安全区**（§5 的 CogVideoX ablation 边界），分歧全部发生在「要不要更狠」上。
2. **深时间压缩的两种付账方式**：WFVAE 把时间压缩翻倍到 $\tau = 8$、代价是通道翻倍到 $c = 32$——数值压缩 $R$ 仍是 48× 不变，但 $N$ 的 $t$ 因子减半、attention 二次项省约 4×。这正是式 (3)(4) 两本账的分离实例： $N$ 按 $\tau s^{2}$ 缩、数值体积按 $3 \tau s^{2} / c$ 缩， $c$ 是兑换旋钮。WF-VAE 的做法是用小波分解组织时间信息流（[Li et al., 2024](../references.md)，arXiv:2411.17459，[A]；Open-Sora-Plan v1.3 报告 [Lin et al., 2024](../references.md)，arXiv:2412.00131，[A]；v1.5.0 配置为 [repo README](../references.md) 口径）。
3. **LTX 把 patchify 挪进 VAE**：摘要原文「compression ratio of 1:192, with spatiotemporal downscaling of 32×32×8 pixels per token, enabled by relocating the patchifying operation from the transformer's input to the VAE's input」（[HaCohen et al., 2025](../references.md)，arXiv:2501.00103，[A]）——transformer 输入侧不再有独立 patchify，VAE 直接吐出「32×32×8 像素/token」的网格；由 $3 \cdot 8 \cdot 32^{2} / 192 = 128$ 反推 latent 通道。token 网格与通用档相比： $\tau s^{2}$ 从 256 涨到 8192，同一段视频的 $N$ 缩到 $1/32$。这笔交易的结构是：VAE 变深（一次性编解码成本），换 DiT 在每个去噪步上都省的二次项——去噪步数越多，交易越划算。换到的是端到端速度：2B 级参数的 DiT 在 H100 上 2 秒生成 5 秒 768×512@24fps 视频，快于实时（论文摘要口径）。
4. **StepVideo 把两个旋钮同时拧狠**： $(8, 16, 16)$——空间 $s = 16$ 与 Wan2.2 同档、时间 $\tau = 8$ 与 WFVAE 同档（[Ma et al., 2025](../references.md)，arXiv:2502.10248，[A]；摘要口径， $c$ 未登记）。 $\tau s^{2} = 2048$，同段视频的 $N$ 是通用档的 $1/8$；它服务的负载是 30B 主干、最长 204 帧——长时长预算正是从 token 账里省出来的。注意 $\tau s^{2} = 2048$ 恰与 §5 撞墙的 CogVideoX E 档同密度；摘要口径未披露其 $c$ 与训练配方，该密度下收敛可行的条件本书无从登记。
5. **Mochi 的口径差**（[Genmo, Mochi 1 repo](../references.md)，[C]）：AsymmVAE 因果压缩 8×8 空间 × 6× 时间 → 12 通道，README 称「总压缩 128×」；按数值体积算得 $6 \cdot 8^{2} \cdot 3 / 12 = 96$ 倍，两处口径之差官方未说明——本书以机制参数 $(6, 8, 8, 12)$ 为准。表中所有「—」同理：事实源未登记的数字不推测、不补全。Open-Sora 2.0 行填的是族名而非参数：Video DC-AE 表明它属于深压缩系（§5），但其 $(\tau, s, s)$ 与 $c$ 未在事实源登记，留空。读这张表时把证据等级一起读：Wan2.2 行是官方 README 一手（无论文），Mochi 行是 [C]——只登记已披露信息。

## 5. 压缩率的边界：收敛、重建与 DC-AE 式深压缩

压缩率不是免费旋钮。本书事实源收录的同系 VAE 压缩率对照只有 CogVideoX 论文的 ablation 一组（论文 Table 1，记号 = 空间 × 空间 × 时间；此处按本书 $(\tau, s, s, c)$ 记号实列通道）：Baseline $(1, 8, 8, 4)$、A $(4, 8, 8, 8)$、B $(4, 8, 8, 16)$（预训练采用）、C $(4, 8, 8, 32)$、D $(8, 8, 8, 32)$、E $(8, 16, 16, 128)$。逐档代入式 (4)，数值压缩 $R$ = 48 / 96 / 48 / 24 / 48 / 48——通道随压缩率同步加宽， $R$ 始终停在 24–96 区间；真正被推高的是 token 密度 $\tau s^{2}$ = 64 / 256 / 256 / 256 / 512 / 2048。撞墙的是 E 档，原文结论「even if the channel dimensions are correspondingly increased, the convergence of the model also becomes extremely difficult」——128 通道把 $R$ 压回 48×，仍救不回 $\tau s^{2} = 2048$（8 倍于采用的 B 档）带来的收敛崩塌。最终选定 B 档 $(4, 8, 8, 16)$（[Yang et al., 2024](../references.md)，arXiv:2408.06072，[A]）。

机制读法两条：

1. **重建误差是生成质量的天花板**——7.6 §1 的 LDM 压缩率消融结论在视频侧原样适用：decoder 画不回去的细节，DiT 生成得再好也落不了地。时间维的重建误差集中暴露在快速运动与动态纹理上： $\tau$ 帧合 1 意味着帧间差分必须全部塞进 $c$ 个通道，运动越剧烈、每个 latent 位点要携带的信息越多， $c$ 固定时 $\tau$ 的信息瓶颈先于空间维到达。式 (4) 的 $R$ 换个读法就是「每个 latent 数值要还原的像素数值数」：Wan 档 48、Wan2.2 档 64、Mochi 档 96、LTX 档 192——压缩越狠单点信息越稠密，与 §4 的 $c$ 兑换是同一枚硬币。
2. **深压缩的 latent 本身更难被生成**：每位点信息量上升、位点间统计更尖锐，DiT 的回归目标变难——E 档把这条钉死：通道加到 128、 $R$ 回到 48×，收敛仍然崩，说明撞墙的量是 token 密度 $\tau s^{2}$（决定 $N$ 与每位点信息量）而非数值体积比 $R$；CogVideoX 说的「convergence extremely difficult」是 DiT 侧收敛困难，不只是 VAE 侧重建变差。压缩率的选择因此是 VAE 重建与 DiT 可学性的联合约束。

两条机制各自给出一个可观测预测（机制推理，非收录的实证结论）：VAE 侧撞边界时，重建视频的运动区域发虚、高频动态纹理被抹平——重建误差集中在帧间差分大的区域；DiT 侧撞边界时，训练 loss 停在高位平台——latent 逐位点信息量被拉大后，扩散目标中难回归分量的占比上升。两个症状指向相反的修法（加 $c$ 或降 $\tau$ 救前者，降 $R$ 救后者），排查时先定位再动手。

DC-AE 式回答：把压缩率继续推高同时保住重建，需要专门的优化技术——图像版 DC-AE（6.3 §7.3：residual autoencoding 把 shortcut 挂进每个下采样/上采样 block，可学支路只拟合本级残差，外加三阶段适配，把压缩率推到 $f = 32/64$；[Chen et al., 2024](../references.md)，arXiv:2410.10733，[A]）已给出范式。视频侧的官方实例是 Open-Sora 2.0：Video DC-AE + MMDiT（[Zheng et al., 2025](../references.md)，arXiv:2503.09642，[A]）。Wan2.2 走同方向的另一解： $(4, 16, 16)$、 $c = 48$——空间压缩从 8 翻倍到 16 由通道从 16 加宽到 48 买单， $R = 64$；TI2V-5B 在 VAE 之上再叠 patchify，token 网格等效 $4 \times 32 \times 32$，5B dense 主干即可跑 720P@24fps（[Wan2.2 官方 README](../references.md)，[A]，无论文；48 通道为 HF config 实测，[wan2.2-a14b-hf](../references.md)，[A]）。两条路线的分工：DC-AE 式把优化技术砸在 VAE 训练上，Wan2.2 式把信息量摊到更多通道上——目标相同，都是把 $R$ 推过 48× 而不撞 CogVideoX ablation 的收敛墙。

重建质量的评估方法论：对「编码 → 解码」往返的重建样本算 FID/FVD（与生成样本同指标、同特征空间），记作 rFID/rFVD 以示区分。它度量的是 VAE 单模块的保真上限，与端到端生成质量解耦——是 §4 各档选型的可比依据，但只在同数据集同口径下可比，跨论文比较没有意义；本书不引未收录的基准数字。

## 6. 实例内存账：720p × 5s × 24fps 全链路

把式 (1)–(4) 代进一个具体负载：720p（ $1280 \times 720$）、5 秒、24 fps → $T = 120$，Wan-VAE 档 $(\tau, s, s) = (4, 8, 8)$、 $c = 16$。

| 链路环节 | shape / 公式 | 数值量 | 字节 | 注 |
| --- | --- | --- | --- | --- |
| 原始视频 | $120 \times 720 \times 1280 \times 3$，式 (1) | $3.32 \times 10^{8}$ | 332 MB（uint8）/ 664 MB（bf16） | 磁盘与解码后张量 |
| latent | $31 \times 90 \times 160 \times 16$，式 (2) | $7.14 \times 10^{6}$ | 14.3 MB（bf16） | $t = 1 + \lceil 119/4 \rceil = 31$ |
| token 序列 | $N = t \cdot h \cdot w$，式 (3) | 446,400 | — | patchify 等效边长 1 口径 |
| attention 得分矩阵（单层若实体化） | $N^{2}$ | $1.99 \times 10^{11}$ | ≈ 399 GB（bf16） | online softmax 分块不实体化（4.1 FlashAttention），FLOPs 照付 |
| attention matmul（单层） | $4 N^{2} d$， $d = 5120$ | ≈ $4.1 \times 10^{15}$ FLOPs | — | Wan2.1-14B 主干 dim 5120（[wan2.1-repo](../references.md)，[A]）；40 层合计 ≈ 163 PFLOPs/去噪步 |
| 对照： $\tau = 1$ 逐帧 2D VAE | $N = 120 \times 90 \times 160$ | 1,728,000 | $N^{2}$ bf16 ≈ 5.97 TB | 时间 4× 压缩把二次项砍约 15× |
| 对照：Wan2.2 档（同负载） | latent $31 \times 45 \times 80 \times 48$； $N = 111{,}600$ | $N^{2} = 1.25 \times 10^{10}$ | $N^{2}$ bf16 ≈ 24.9 GB | 配 TI2V-5B 主干 $d = 3072$：单层 $4N^{2}d \approx 1.5 \times 10^{14}$ FLOPs |

末行与 Wan2.1-14B 行的比值约 27×，可分解为 VAE 贡献的 16×（ $N^{2}$： $(446{,}400/111{,}600)^{2} = 4^{2}$）与主干宽度贡献的约 1.7×（ $5120/3072$）——深压缩 VAE 与小主干是乘性配合（该档 HF config 实测 dim 3072/30 层，[wan2.2-a14b-hf](../references.md)，[A]；注意 TI2V-5B 实机还叠 patchify $2 \times 2$，token 数在此口径上再除 4，§4）。

压缩比核验： $3.32 \times 10^{8} / 7.14 \times 10^{6} \approx 46.5$（长视频极限 48×，差额即首帧特判的 $+1$）。读法：VAE 把存储账砍掉约 46×，但留下的 $N = 4.46 \times 10^{5}$ 仍让 full attention 的单层 FLOPs 处在 PFLOPs 级——VAE 决定 DiT 成本的数量级，9.2 的 attention 拓扑与逐家 patchify 决定剩下的常数。两个灵敏度顺手记下：时长方向 $N$ 线性、attention 项二次（§3 的表）；分辨率方向 $h \cdot w$ 随边长二次长， $N$ 二次、attention 项四次——1080p 相对 720p 的 attention 代价是 $(3/2)^{4} \approx 5$ 倍。编码器/解码器自身的激活内存与 DiT 训练内存另有专账，不属本章。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 数据账 | 式 (1)；720p 5s ≈ $3.32 \times 10^{8}$ 值 / 332 MB（uint8） | 不压缩时 $N^{2} \sim 10^{12}$；压缩是可行性前提 |
| latent 形状 | 式 (2)(3)： $t = 1 + \lceil (T-1)/\tau \rceil$、 $N = t h w$ | 首帧只空间压缩（Wan 式，源自 MagViT-v2）；81 帧 → 21 latent 帧 |
| 两本账 | 式 (4)： $R \approx 3 \tau s^{2} / c$ | $N$ 按 $\tau s^{2}$ 缩、数值体积按 $3 \tau s^{2}/c$ 缩； $c$ 是兑换率（WFVAE 例） |
| 因果 3D conv | 式 (5)：时间维左补 $k_{t} - 1$ | 三机制：图像兼容 / feature caching 流式 / 无未来泄漏 |
| 通用档 | $(4, 8, 8, 16)$、 $R = 48$×；scaling 0.7 / 0.476986 | 三家 config 实测互证 |
| 深压缩 | Wan2.2 $(4, 16, 16, 48)$ 64×；LTX 1:192（ $c = 128$ 反推）；WFVAE $(8, 8, 8, 32)$；Video DC-AE | 边界证据：CogVideoX E 档 16×16×8（128 通道， $\tau s^{2} = 2048$）「convergence extremely difficult」——通道加倍救不回 token 密度 8× |
| 实例账 | $N = 446{,}400$ → $N^{2}$ ≈ 399 GB；单层 $4N^{2}d$ ≈ 4.1 PFLOPs（ $d = 5120$） | $\tau = 4$ 把二次项砍约 15×；VAE 定数量级 |

## 练习

1.（计算）latent 与 token 账。对 10 秒、24 fps（ $T = 240$）视频分别用三档 VAE 计算 $t$、 $h \times w$、 $N$ 与数值压缩比的精确值（式 (2)(3)(4) 的非极限形式）：(a) Wan2.1 档 $(4, 8, 8, 16)$、分辨率 $1280 \times 720$；(b) Wan2.2 档 $(4, 16, 16, 48)$、同分辨率；(c) LTX 档 $(8, 32, 32, 128)$、分辨率换成其论文口径 $768 \times 512$。再算三档单层 attention 得分矩阵的 bf16 字节数，并验证 (b)(c) 的 $R$ 分别收敛到 64× 与 192×。
2.（推导）因果感受野。设 encoder 时间路径依次堆叠：层 A（ $k_t = 3$、stride 1、因果）→ 层 B（ $k_t = 4$、stride 4、因果时间下采样）。取定补零对齐约定（式 (5)），求 latent 帧 $j$（ $j \ge 1$）读取的输入帧区间；验证上界不越过第 $4j$ 帧，并说明首帧特判（ $j = 0$ 只含帧 0）在该堆叠下如何实现。
3.（分析）非因果反事实。设 VAE 时间维改用对称 padding（ $k_t = 3$），其余不变。(a) 写出视频首帧 latent $z_0$ 的依赖集；(b) 说明孤立图像（无未来帧，只能补零或复制边界帧）经同一 encoder 得到的 latent 与「同一图像作为视频首帧」的 latent 为何不逐位一致；(c) 由此说明非因果 VAE 要做图视频混训须付出什么（两条 latent 分布、或为图像路径单独关闭时间卷积的工程补丁），并对照 §2.2 的因果方案。
4.（计算）深压缩的 DiT 侧代价。取 §6 负载（ $T = 120$、 $1280 \times 720$）：(a) 算 Wan2.2 档 $(4, 16, 16, 48)$ 的 $N$ 与 $N^{2}$ 相对 Wan2.1 档的降幅；(b) 算两档 latent 的数值体积与每 token 通道数，用式 (3)(4) 的两本账解释「token 更少、单 token 更稠密」为何不矛盾；(c) 若深压缩使重建误差上升，按 §5 的两条机制说明它如何传导到生成质量与 DiT 收敛。

## 参考

- [Yang et al., 2024](../references.md)：CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer，arXiv:2408.06072，[A]（key: cogvideox）——3D causal VAE 出处、VAE ablation（8×8×1 / 8×8×4 / 16×16×8）与「convergence extremely difficult」原文、2B/5B 档
- [Kong et al., 2024](../references.md)：HunyuanVideo: A Systematic Framework For Large Video Generative Models，arXiv:2412.03603，[A]（key: hunyuanvideo）——Causal 3D VAE（CausalConv3D）、README「compression ratios of video length, space, and channel to 4, 8, and 16」
- [Team Wan, 2025](../references.md)：Wan: Open and Advanced Large-Scale Video Generative Models，arXiv:2503.20314，[A]（key: wan）——§4.1.1 Wan-VAE：首帧仅空间压缩（following MagViT-v2）、 $[1+T/4, H/8, W/8]$、GroupNorm→RMSNorm
- [Yu et al., 2023](../references.md)：Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation，arXiv:2310.05737，[A]（key: magvit-v2）——首帧特判约定的源头（6.3/8.5 已拆，本章只引兼容机制）
- [Team Wan, Wan2.2 官方 repo README](../references.md)：[A]（key: wan2.2）——Wan2.2-VAE 4×16×16（总 64×）、TI2V-5B 叠 patchify 后 token 网格 4×32×32、5B dense 720P@24fps；无论文，表述引 README 原文
- [HaCohen et al., 2025](../references.md)：LTX-Video: Realtime Video Latent Diffusion，arXiv:2501.00103，[A]（key: ltx-video）——1:192、32×32×8 pixels/token、patchify 挪入 VAE、2B 级（摘要口径）、H100 上 2 秒生成 5 秒 768×512@24fps
- [Ma et al., 2025](../references.md)：Step-Video-T2V Technical Report，arXiv:2502.10248，[A]（key: stepvideo-t2v）——Video-VAE 16×16 空间 + 8× 时间（摘要口径）
- [Zheng et al., 2025](../references.md)：Open-Sora 2.0: Training a Commercial-Level Video Generation Model in \$200k，arXiv:2503.09642，[A]（key: open-sora-2）——Video DC-AE + MMDiT，DC-AE 式深压缩的视频官方实例
- [Lin et al., 2024](../references.md)：Open-Sora Plan: Open-Source Large Video Generation Model，arXiv:2412.00131，[A]（key: open-sora-plan）——v1.3 报告；v1.5.0 配置以 repo README 为准
- [Li et al., 2024](../references.md)：WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model，arXiv:2411.17459，[A]（key: wfvae）——WFVAE 8×8×8、32-dim latent（v1.5.0 口径）
- [Genmo, Mochi 1 官方 repo](../references.md)：[C]（key: mochi）——AsymmVAE 因果 8×8×6 → 12ch（无论文，只登记 README 披露；genmoai/models 已 301 → genmoai/mochi）
- [Chen et al., 2024](../references.md)：Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models，arXiv:2410.10733，[A]（key: dc-ae）——深压缩优化技术（residual autoencoding + 三阶段适配）出处，6.3 §7.3 已拆
- [Z.ai, CogVideoX-5b config](../references.md)：[A]（key: cogvideox-config）——`temporal_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.7`、sample_frames 49、特征图 60×90
- [Tencent, HunyuanVideo VAE config](../references.md)：[A]（key: hunyuanvideo-vae-config）——`AutoencoderKLCausal3D`、`time_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.476986`
- [HuggingFace, diffusers AutoencoderKLWan](../references.md)：[A]（key: autoencoderkl-wan）——`WanCausalConv3d`：时间维因果 padding + feature caching 走读锚点
- [Z.ai, zai-org/CogVideo](../references.md)：[A]（key: cogvideo-repo）；[Tencent, Tencent-Hunyuan/HunyuanVideo](../references.md)：[A]（key: hunyuanvideo-repo）；[Alibaba, Wan-Video/Wan2.1](../references.md)：[A]（key: wan2.1-repo）；[Alibaba, Wan2.2-T2V-A14B HF 仓](../references.md)：[A]（key: wan2.2-a14b-hf）——官方代码与 config 实测锚点（TI2V-5B `in_dim/out_dim: 48`）
