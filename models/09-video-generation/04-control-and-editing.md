# 9.4 控制与编辑（Control and Editing）

> 控制一个视频 DiT，能动的只有三种东西：喂给它的条件（I2V 的 latent concat + mask 通道、CLIP 图像 embedding、ControlNet 可训练副本与 zero-conv、相机 Plücker 射线图、drag 轨迹图）、它走的去噪轨迹（DDIM inversion 与 null-text 逐步修正）、它的权重（LoRA 低秩增量与 DreamBooth 双损失微调）；多镜头叙事是三者之上的组合层（keyframe 链）| 前置：9.1（latent 几何 $(\tau, s, s)$、首帧约定）、9.2（时空 DiT 结构、cross-attn 成本账）、7.4（DDIM 更新式 (5)、CFG 式 (18)、DDIM=PF-ODE §2.4）、7.6（adaLN-Zero 式 (8)、conditioning 三机制）、7.3（Tweedie 式 (10)）、7.5（FM 目标式 (22)） | 锚点：Wan-AI/Wan2.1-I2V-14B-720P config [A]、Wan-Video/Wan2.1（`clip_fea` 走读）[A]、Wan-Video/Wan2.2 [A]、lllyasviel/ControlNet [A]、hehao13/CameraCtrl [A]；ControlNet [A]、CameraCtrl [A]、MotionCtrl [A]、null-text inversion [A]、LoRA [A]、DDIM [A]、DreamBooth [B]

9.2 §7 把文本条件的成本钉死在主干账的 2% 以内——文本通道的设计空间是全开的。但文本表达不了的东西恰是视频生成的重头：首帧画面长什么样、逐帧的深度/边缘结构、相机怎么动、画面里某个点往哪拖、主体必须是「这一只」。本章按「控制信号改的是什么」把控制机构分成四族：改输入条件（§1–§4）、改去噪轨迹（§5）、改权重（§6）、改生成的组合结构（§7）。四族共享同一条工程原则——**起点即原模型**：一切控制机构在训练起点必须严格不改变基座的输出；§2 把这条原则的三个实例（adaLN-Zero、zero-conv、LoRA 的 $B$ 因子）归到同一条公式性质上。

## 1. I2V：条件注入公式家族

I2V（image-to-video）的问题形式：给定条件图像 $I_c$（与文本 $c_{txt}$），生成以 $I_c$ 为首帧（或内容锚点）的视频。在 latent 侧（9.1），条件图像是 $z_c^{img} = s \cdot E(I_c) \in \mathbb{R}^{1 \times h \times w \times c}$（$s$：VAE scaling factor，9.1 §3 口径——与压缩率记号 $(\tau, s, s)$ 里的空间 $s$ 同名不同义）——首帧特判约定（9.1 式 (2)）使单张图像与视频首帧共享同一 latent 空间，I2V 因此不需要任何新编码器。剩下的问题只有一个：这个 latent 怎么进 DiT。开放系统的答案是一个三件套公式家族，Wan2.1-I2V 是代码级证据最全的实例。

### 1.1 latent concat + mask 通道

Wan-AI/Wan2.1-I2V-14B-720P 的 config 实测 `in_dim: 36`（[wan21-i2v-config](../references.md)，[A]）——36 = 16 + 16 + 4：噪声 latent（16 通道，9.1 的通用档）+ 条件图像 latent（16 通道）+ mask（4 通道），沿通道维拼接。每个去噪步的输入为

$$
\tilde x_t = [\,x_t \,;\, z_c \,;\, m\,] \in \mathbb{R}^{t \times h \times w \times 36} \qquad (1)
$$

**符号**：$x_t \in \mathbb{R}^{t \times h \times w \times 16}$：当前去噪步的噪声 latent；$z_c$：条件 latent——锚点帧位填 $z_c^{img}$、其余位填零；$m$：逐 latent 帧的锚点指示（1 = 该帧由条件钉死，标准读法；官方未展开 mask 的逐位语义说明），空间维广播。给定锚点集合 $\mathcal{A} \subseteq \{0, \ldots, t-1\}$：

$$
z_c^{(j)} = \begin{cases} z_c^{img,(j)} & j \in \mathcal{A} \\ 0 & j \notin \mathcal{A} \end{cases}, \qquad m^{(j)} = \begin{cases} 1 & j \in \mathcal{A} \\ 0 & j \notin \mathcal{A} \end{cases} \qquad (2)
$$

**符号**：上标 $(j)$：第 $j$ 个 latent 帧；$z_c^{img,(j)}$：第 $j$ 位锚点的条件图像 latent——锚点图按位替换，单图 I2V 是各位同图的特例，首尾帧形态两端各一张（§1.4）。首帧 I2V 对应 $\mathcal{A} = \{0\}$。

```
输入侧（每个去噪步）                        通道维拼接
x_t  [t × h × w × 16]  噪声 latent ───┐
z_c  [t × h × w × 16]  条件 latent ───┼─► [x_t; z_c; m]  [t × h × w × 36]
m    [t × h × w ×  4]  锚点 mask ─────┘         │
  z_c：锚点帧填 s·E(I_c)，其余帧全 0             ▼
  m：  帧 0 全 1（首尾帧形态再加帧 t−1）   patchify + 线性投影（仅第一层换权重 16→36）
                                                ▼
                                     X [N × d]  进时空 DiT（9.2）
```

拼接后的 $\tilde x_t$ 过常规 patchify + 线性投影（9.2 §1），第一层权重从 $16 \times d$ 换成 $36 \times d$——这是唯一需要新参数的层。账（patch=1 等效口径、$d$ = 5,120 的 Wan 14B 档）：第一层参数增量 $20 \times 5120 \approx 1.0 \times 10^{5}$，相对 14B 主干约百万分之七；每 token 投影 FLOPs 增量 $2 \times 20 \times 5120$，乘 $N$ = 302,400 约 0.06 TFLOPs，对 9.2 §2.5 单层 2.06 PFLOPs 的总账是十万分之三。I2V 的成本全在数据与训练步数上，不在架构上。

mask 占 4 通道而非 1 是 config 实测值，官方未说明缘由——机制读法（本书分析，不坐实）：多通道广播给「这是锚点帧」这个比特一份冗余编码，使它在投影进 $d$ 维后不易被 latent 特征淹没。

一个退化观察（本书分析）：$\mathcal{A} = \varnothing$ 时式 (1)(2) 退回纯 T2V 输入（条件通道全零、mask 全零）——T2V 与 I2V 在同一输入接口下统一，只差一个 mask 的取值。Wan2.2 仓即同时发布 T2V 与 I2V 形态（[Wan-Video/Wan2.2](../references.md)，[A]；TI2V-5B 档的 VAE 与主干账见 9.1）。

### 1.2 noise augmentation：首帧加噪防复制粘贴

直接把干净的 $z_c^{img}$ 喂进去有一个训练动力学问题：训练样本的条件帧取自视频自身，与后续帧天然一致；最小化去噪损失的一条捷径是把条件帧复制到输出首帧区域、再从此出发外推——「复制粘贴」策略在训练集上损失很低，但它没有学到「从一张不一致的图出发组织运动」。推理时用户给的图与训练分布几乎必然不一致（画风、分辨率、内容），按复制粘贴策略学出的模型在首帧交接处露馅。Wan 论文的配方（[Team Wan, 2025](../references.md)，arXiv:2503.20314，[A]）是给条件 latent 加噪：

$$
\hat z_c = z_c^{img} + \sigma_c\, \eta, \qquad \eta \sim \mathcal{N}(0, I) \qquad (3)
$$

**符号**：$\sigma_c$：条件噪声强度（加噪的调度细节以 Wan 论文原文为准）。像素级复制被噪声破坏后，条件帧唯一可用的角色是语义与构图锚点——模型被迫学「保留身份、重画细节」，恰是推理时需要的泛化方向；训练时见过的条件分布（带噪、有损）也覆盖了推理时的条件分布（用户图 ≠ 任何视频的真实首帧），train/test 的条件失配被同时缓解。

### 1.3 CLIP image embedding cross-attn

第三件走语义通道：条件图像过 CLIP 视觉塔（[Radford et al., 2021](../references.md)，arXiv:2103.00020，[A]；训练机制见 6.2），取 image embedding（Wan 代码侧为 1280 维），经 MLP 投影到主干宽度后作为独立条件注入 cross-attention 子层。代码证据：wan/modules/model.py 实测 `clip_fea` 独立条件输入、`img_emb = MLPProj(1280, dim)`（[Wan-Video/Wan2.1](../references.md)，[A]）；diffusers 官方管线 WanImageToVideoPipeline 用 CLIPVisionModel image_encoder + AutoencoderKLWan（Wan2.1 README 官方示例口径）。

两条图像条件通道的分工（机制读法）：latent concat 携带像素级身份——构图、颜色、边缘的精确位置，且只占首帧一个锚点位；CLIP embedding 携带语义级身份——「这是什么」，对空间错位鲁棒，且在全部去噪步的 cross-attn 里持续可读。前者管「首帧长得一样」，后者管「整段视频里主体还是它」。这套「像素通道 + 语义通道」的双条件与 7.6 §4 的多编码器分工同族：不同抽象层级的条件各出各的编码器、汇合进同一条件面；成本上两者都不到主干账的零头（9.2 §2.5：cross-attn 全部占 1.7%），不存在二选一的预算压力。

### 1.4 首尾帧控制

把锚点集合从 $\{0\}$ 换成 $\{0, t-1\}$，式 (1)(2) 原样成立——首尾帧控制（first-last-frame-to-video）是同一公式的直接推广：$z_c$ 现在有两个非零位，mask 两端置 1，输入通道数不变。变的是任务性质：单锚点是外推（初值问题），双锚点是插值（边界值问题）——运动轨迹两端被同时钉死，模型要生成一条两端吻合的中间轨迹。对生成质量的要求从「首帧保真」变成「两端保真 + 中间轨迹合理」：首尾差异越大，中间帧必须承载的运动幅度越大，越容易顶到模型运动先验的边界。本节按公式级推广给出；首尾帧形态的具体产品配置以各家发布为准，diffusers 官方示例覆盖的是单锚点 I2V 形态（§1.3 锚点）。

## 2. ControlNet-for-DiT：可训练副本与 zero-conv

§1 的条件通道解决「内容锚点」，不解决「结构控制」——边缘图、深度图、姿态序列这类逐帧 dense map 需要另一条通路。ControlNet（[Zhang et al., 2023](../references.md)，arXiv:2302.05543，[A]；官方 repo [lllyasviel/ControlNet](../references.md)，[A]，ldm/ 目录为走读锚点）的构造两步：(i) 冻结基座参数 $\Theta$，把基座编码侧复制出一份可训练副本 $\Theta_c$；(ii) 副本的输出经 zero-conv 加回基座通路。zero-conv 是权重与偏置都零初始化的 1×1 卷积 $\mathcal{Z}(\cdot\,; \{W, b\})$，整网为

$$
y = F(x;\, \Theta) + \mathcal{Z}\Big( F_c\big( x + \mathcal{Z}(c;\, \{W_1, b_1\});\, \Theta_c \big);\, \{W_2, b_2\} \Big) \qquad (4)
$$

**符号**：$F$：基座 block；$F_c$：其可训练副本；$c$：dense 条件图（与 $x$ 同空间网格）；$\mathcal{Z}$：zero-conv，初始化 $W_i = 0$、$b_i = 0$。

**零初始化 = 起点即原模型。** 代入 $W_2 = 0, b_2 = 0$：外层 $\mathcal{Z}$ 恒输出零，式 (4) 严格等于 $F(x; \Theta)$——训练起点处整个控制分支的贡献为零，组合网络逐位等于预训练基座。这不是「扰动很小」的近似，是严格恒等：控制训练从一台完好的生成器出发，要学的只有「控制带来的增量」。

梯度的不对称性是该构造训得起来的关键。对 $\mathcal{Z}(u) = Wu + b$：$\partial \mathcal{Z} / \partial W = u$ 与 $\partial \mathcal{Z} / \partial b = 1$ 在零初始化处非零（$\Theta_c$ 随机初始化保证副本输出 $u \neq 0$）——**门先开**；但流进副本 $\Theta_c$ 与内层 $W_1$ 的梯度要乘 $W_2$，第一步严格为零——**支路后动**。训练顺序被自动排好：先学「门开多大」，再学「门里给什么」。

**与 adaLN-Zero 同思想**（7.6 式 (8) 的推导原样适用）：adaLN-Zero 把调制 MLP 末层零初始化，使每个 DiT block 以 identity 起步、整网初始预测为零（[Peebles & Xie, 2023](../references.md)，arXiv:2212.09748，[A]）；zero-conv 把支路末层零初始化，使控制分支初始输出为零、组合网初始等于基座。同一个「零初始化门控」原则——可训练增量串一个零初始化的门 ⇒ 起点即原模型，梯度先开门、后训支路。§6 的 LoRA（$B = 0$ 初始化）是第三个实例：

| 实例 | 零初始化的门 | 门的位置 | 起点的直接后果 |
| --- | --- | --- | --- |
| adaLN-Zero（7.6 式 (8)） | 调制 MLP 末层（gate $\alpha$） | 每个 block 的调制系数 | block = identity，整网初始预测为零 |
| zero-conv（式 (4)） | 支路进出口的 1×1 conv $W, b$ | 控制支路两端 | 组合网逐位等于基座 |
| LoRA（式 (10)） | 低秩右因子 $B$ | 权重增量 $\Delta W$ | $\Delta W = 0$，$W = W_0$ |

**DiT 语境。** 原论文的 $F$ 是 U-Net 编码级；搬到时空 DiT（7.6/9.2）时构造原话照读：基座 DiT 冻结，可训练副本取基座的若干 block（原论文对应 U-Net 编码侧全部尺度；DiT 上取哪几层是设计自由度，本书不坐实具体产品的选择），dense 条件（逐帧深度/边缘/姿态，编码或下采样到 latent 网格）从副本入口进，各副本 block 的输出经零初始化线性层加回基座对应位置的残差流。视频语境的两个增量事实：(i) 条件图是逐帧的，$c \in \mathbb{R}^{t \times h \times w \times c_c}$，与 latent 同网格——§1 的 concat 接口与本节的支路接口可叠加；(ii) 成本账：训练时为副本多存一份参数与激活；推理时副本不能合并进基座（它依赖逐样本条件 $c$），每个去噪步多付副本各层的前向——复制几层就是「控制力 vs 每步成本」的旋钮。

## 3. 相机控制：Plücker ray embedding

相机运动（推拉升降、环绕、平移）用文本几乎无法精确指定。把相机轨迹变成条件的标准表示是 Plücker 射线参数化，出处是 CameraCtrl（[He et al., 2024](../references.md)，arXiv:2404.02101，[A]——「Plücker embeddings」在原文出现 32 处；官方 repo [hehao13/CameraCtrl](../references.md)，[A]）。

**从内外参构造。** 设帧 $f$ 的相机内参 $K \in \mathbb{R}^{3 \times 3}$、外参旋转 $R_f$ 与平移 $t_f$（世界系到相机系 $x_{cam} = R_f x + t_f$）。像素 $(u, v)$ 对应的射线：相机系里反投影、转到世界系取单位方向 $d_f$；相机光心在世界系的位置 $o_f$。Plücker 坐标把这条射线表示为 6 维向量：

$$
d_f(u,v) = \frac{R_f^{\top} K^{-1} [u, v, 1]^{\top}}{\lVert R_f^{\top} K^{-1} [u, v, 1]^{\top} \rVert}, \qquad o_f = -R_f^{\top} t_f, \qquad p_f(u,v) = \big(d_f(u,v),\; o_f \times d_f(u,v)\big) \in \mathbb{R}^{6} \qquad (5)
$$

**符号**：$d_f$：世界系单位方向（相机系方向 $K^{-1}[u, v, 1]^{\top}$ 经逆旋转 $R_f^{\top}$ 转回世界系——与 w2c 约定自洽：$q_{world} = R_f^{\top}(q_{cam} - t_f)$）；$o_f$：相机光心（世界系）；$o_f \times d_f$：moment——射线与世界原点张成平面的法向。逐帧逐像素构造、下采样对齐 latent 网格，得条件张量 $P \in \mathbb{R}^{t \times h \times w \times 6}$。

Plücker 表示的三个数学性质，各自对应一个工程收益：

1. **moment 与射线上取点无关**：射线上任意点 $q = o + \lambda d$ 满足 $q \times d = o \times d + \lambda\,(d \times d) = o \times d$——表示只依赖射线本身，不依赖在射线上选哪个点当原点。
2. **旋转与平移分置两半**：$d$ 不含平移（平移不进式 (5) 第一式），moment 含平移——$\lVert o \times d \rVert$ 恰为世界原点到射线的距离（$d$ 单位时）；推拉的平移量与环绕的旋转量在条件空间里占据不同通道，可区分、可复合。
3. **与像素网格同形**：6 通道 dense map 直接对齐 latent 网格，「哪个像素走哪条射线」的对应关系由网格共享自动给出——不需要把相机矩阵压成一个全局向量再广播（那会丢掉全部逐像素对应）。6 通道相对 16 通道 latent 也便宜：条件通道的引入成本与 §1 同量级。

CameraCtrl 的系统形态（论文口径）：Plücker 图序列经一个独立的相机编码器提取特征后注入基座；相机模块 plug-and-play——**不动基座权重**，与 §2 的「起点即原模型」同一纪律。MotionCtrl（[Wang et al., 2023](../references.md)，arXiv:2312.03641，[A]；官方 repo TencentARC/MotionCtrl）把控制面再扩一维：相机运动与物体运动统一控制——相机侧以相机位姿序列为条件（内部构造未在本轮事实源展开，以论文为准），物体侧用轨迹条件（§4），两个条件通道可独立启用也可组合，覆盖「只动相机」「只动物体」「两者复合」三种工况。

## 4. 轨迹条件：drag point embedding

物体级运动控制的输入形态是 drag 轨迹：用户在首帧点选 $K$ 个点，逐帧拖动给出 2D 轨迹 $\{(u_k^{f}, v_k^{f})\}_{f=1}^{T}$（$k = 1, \ldots, K$）。MotionCtrl 的物体运动侧即消费此类条件（§3），但其内部构造未在本轮事实源展开；以下按公式级一般性表述给出标准构造：把稀疏轨迹点转成 dense 条件图——逐帧在每个轨迹点位置溅落一个高斯斑：

$$
G_k^{f}(u, v) = \exp\left( -\frac{(u - u_k^{f})^{2} + (v - v_k^{f})^{2}}{2\sigma_g^{2}} \right) \qquad (6)
$$

**符号**：$\sigma_g$：高斯斑宽度，即「软约束」的空间半径。$K$ 条轨迹叠成 $K$ 个通道（或按语义合并为少量通道），逐帧堆叠成 $T \times H \times W \times K'$ 的条件张量，下采样到 latent 网格后经条件编码器或 concat 注入——与 §1 的 mask、§3 的 Plücker 图共享同一类 dense 条件通道接口。

机制读法（本书分析）：用高斯斑而不是 delta 尖峰，因为控制目标本来就是软的——用户拖的是「大致往这走」，$\sigma_g$ 把指令的置信半径显式编码进条件；多点轨迹的组合表达力大于单点——两个点的相对运动可以区分平移与旋转/形变；轨迹中途消失（遮挡）的处理约定（斑消失还是停在末位）本身是条件语义的一部分，各实现口径不同。

## 5. 视频编辑：DDIM inversion 与 null-text

§1–§4 的控制都在生成新视频；编辑的对象是已有视频：保持运动与结构，按新 prompt 改外观。扩散编辑的两阶段骨架：先把源视频反演（inversion）到噪声端，再带新条件从该噪声重新生成。两个环节各有公式对象。

### 5.1 DDIM inversion 推广到视频

本节记号沿用 7.4 的 DDPM 离散步约定（$t$ 增大 = 噪声增大，$x_0$ 为数据端；与全书 FM 记号——W4 约定页与 7.5 的 $t \in (0,1]$——相差一个时间轴重标定）。自本节起 $t$ 为扩散步号，与 §1–§4 的 latent 帧数同名不同义（9.2 亦有此双载，语境自明处不另立符号）。7.4 式 (5) 的 DDIM 更新是可逆的——把下标方向反过来，就得到从数据端向噪声端的积分：

$$
x_{t+1} = \sqrt{\bar\alpha_{t+1}}\; \hat x_0(x_t, t) + \sqrt{1 - \bar\alpha_{t+1}}\; \varepsilon_\theta(x_t, t), \qquad t:\; 0 \to T \qquad (7)
$$

**符号**：$\hat x_0(x_t, t) = (x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t))/\sqrt{\bar\alpha_t}$（Tweedie，7.3 式 (10)）。式 (7) 与去噪更新共享同一个网络场、方向相反。7.4 §2.4 已证 DDIM（$\sigma = 0$）= PF-ODE 的一阶离散化；inversion 就是同一条 ODE 的反向积分——反演的合法性不是新假设，而是 ODE 的可逆性。FM/rectified-flow 主干（7.5）的对应物是同一路径 ODE 的反向 Euler：「反演 = 沿同一场从数据端积到噪声端」的结构与参数化无关；本章用 VP/DDIM 记号写，因为 7.4 已把它的误差性质钉死（[Song et al., 2021](../references.md)，arXiv:2010.02502，[A]）。

视频推广在形式上零修改：$x_t$ 换成整段 latent（$N = t \cdot h \cdot w$ 个 token 一起过时空 DiT，9.2），式 (7) 逐字成立。变的是误差账本——以下为本书的扩展分析（非收录条目的原文结论）：式 (7) 每步的局部误差有两部分，离散截断（7.4 §2.4：局部 $O((\Delta\lambda)^{2})$）与网络误差（$\varepsilon_\theta \neq \varepsilon$）；两者都改变下一步的出发点。记 $e_t = x_t - x_t^{*}$（相对精确 ODE 轨迹的偏差），一阶展开给出递推

$$
e_{t+1} \approx J_t\, e_t + \delta_t, \qquad e_T \approx \sum_{t=0}^{T-1} \left( \prod_{s > t} J_s \right) \delta_t \qquad (8)
$$

**符号**：$J_t$：更新映射在精确轨迹处的 Jacobian；$\delta_t$：第 $t$ 步局部误差。$T$ 步累积不是局部误差之和，而是经各步 Jacobian 放大或缩小后的加权和——反演误差是复合的，不是独立的。视频语境把两件事放大：(i) full 3D attention 下任何局部位点的误差下一步扩散到全部 token（9.2 §2 的拓扑账）；(ii) 误差在时间维的表现是闪烁与漂移——帧间不一致的偏差在视频里比单图显眼得多。工程推论：inversion 步数给足（压截断项）、反演与再生成用同一调度（标定不漂移）。

### 5.2 null-text inversion：把引导场的轨迹偏差逐步吸收掉

编辑采样需要 CFG（$w > 1$，7.4 式 (18)）抬高 prompt 跟随度；但式 (7) 的反演是在 $w = 1$ 的场上积出来的——换到 $w > 1$ 的场重新生成，轨迹系统性偏离反演轨迹，源视频的结构随之丢失。null-text inversion（[Mokady et al., 2022](../references.md)，arXiv:2211.09794，[A]；官方实现在 google/prompt-to-prompt 仓的 `null_text_w_ptp.ipynb`）的修法是把无条件 embedding 从常量变成逐步可优化变量。两阶段：(i) 以 $w = 1$、源 prompt $c$ 跑式 (7)，存下 pivot 轨迹 $\{x_t^{*}\}_{t=0}^{T}$；(ii) 按采样序逐去噪步优化该步的空条件 embedding $\varnothing_t$，使带 CFG 的一步更新落回 pivot：

$$
\min_{\varnothing_t} \left\lVert x_{t-1}^{*} - \mathrm{DDIM}\big( x_t^{*},\, t,\, \tilde\varepsilon_\theta(x_t^{*}, t, c, \varnothing_t;\, w) \big) \right\rVert^{2} \qquad (9)
$$

**符号**：$\tilde\varepsilon_\theta(\cdot;\, w) = \varepsilon_\theta(\cdot, \varnothing_t) + w\,\big(\varepsilon_\theta(\cdot, c) - \varepsilon_\theta(\cdot, \varnothing_t)\big)$：7.4 式 (18) 的逐步变体；$\varnothing_t$ 初始化取上一步的最优值（沿采样序热启动）。机制读法：$\varnothing_t$ 是条件空间里每步一个的自由偏置，恰好吸收「引导场 vs 反演场」在该步的残差——不碰权重、不碰轨迹，只修正每步无条件分支的读数。编辑时保持 $x_T$ 与 $\{\varnothing_t\}$ 不变、把 $c$ 换成目标 prompt $c'$：结构由 pivot 轨迹锚住，外观跟 $c'$ 走。

视频形态（本书扩展分析）：式 (9) 逐字成立，$c$ 与 $\varnothing_t$ 走视频模型的文本编码器（9.2 §7）。代价账：inversion $K$ 步 + 逐步优化（每步一个小内循环）+ 再生成 $K$ 步，全部按视频的 $N$ 计费（练习 4）——「零训练、逐样本付费」的编辑路线，与 §6 的权重侧路线（一次训练、多次复用）互补。

## 6. Subject 一致性：LoRA 与 DreamBooth 的视频形态

「让某个特定主体出现在生成的所有视频里」是权重侧问题——文本与 dense 条件都表达不了「就是这一只狗」。

**LoRA（低秩适配）**（[Hu et al., 2021](../references.md)，arXiv:2106.09685，[A]；官方 repo microsoft/LoRA）：冻结 $W_0 \in \mathbb{R}^{d \times k}$，学低秩增量

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k) \qquad (10)
$$

**符号**：$\alpha / r$：缩放系数（换秩不必重调学习率）；初始化 $A$ 小随机、$B = 0$——$\Delta W = 0$ 于起点，「起点即原模型」的第三个实例（§2 的表）。参数账：$r(d + k)$ vs $dk$；方阵（$d = k$）时省幅 $d / (2r)$。代入 Wan 14B 几何（$d$ = 5,120，9.2 §2.5）：attention 的 QKVO 四个方阵各挂 $r$ = 16 的 LoRA，每层 $4 \times 16 \times 2 \times 5120 \approx 6.6 \times 10^{5}$ 参数，40 层合计约 26M——14B 基座的 0.19%。推理时 $W = W_0 + (\alpha/r)BA$ 合并进基座权重，零额外 FLOPs；不合并则换主体 = 换适配器，多主体 = 叠加多份增量。视频 DiT 的挂载点沿用 LLM 实践：attention 的 QKV/O 投影（9.2 结构走读里那四个 $d \times d$ 矩阵）。

**DreamBooth**（[Ruiz et al., 2022](../references.md)，arXiv:2208.12242，**[B]**——google/dreambooth 仓仅为主题数据集、无官方训练代码，按论文拆公式）：全量微调，用少量主体图绑定一个稀有 token（论文记 [V]），双损失：

$$
L = \mathbb{E}\,\lVert \varepsilon_\theta(x_t, t, c_{subj}) - \varepsilon \rVert^{2} + \lambda\, \mathbb{E}\,\lVert \varepsilon_\theta(x_t^{pr}, t, c_{pr}) - \varepsilon \rVert^{2} \qquad (11)
$$

**符号**：第一项：主体样本（「a [V] dog」）上的标准去噪损失；第二项 prior preservation：类 prompt（「a dog」）样本上的去噪损失，防稀有 token 劫持整个类的先验（语言漂移）与小样本过拟合；$\lambda$：两项配比。

视频形态（本书按公式级推广）：回归目标换成视频 DiT 的速度场（FM 目标，7.5 式 (22) 口径），双损失结构逐字平移。一个结构性便利来自 9.1：首帧特判约定使「图像 = 单帧视频」对同一模型良定义——主体照片本身就是合法训练样本，不必非要主体视频。LoRA vs 全量微调的取舍：LoRA 的容量上限由 $r$ 钉死（主体细节复杂度超秩则保真降级），但数据需求小、可插拔、合并后零成本；DreamBooth 式全量微调容量大，过拟合与漂移风险靠式 (11) 第二项与 $\lambda$ 管，且一个主体占一份完整权重。

## 7. 多镜头与叙事结构：keyframe 链

单段生成的时长有上限（9.1/9.2 的 token 账、9.3 的长视频路线），其上的叙事内容靠镜头级组合。公式级骨架：把成片分解为关键帧（keyframe）锚定的段链——每段以关键帧图像 $I_k$ 为内容锚点、以镜头 prompt $c_k$ 为文本条件独立生成：

$$
p(V_{1:K} \mid I_{1:K}, c_{1:K}) = \prod_{k=1}^{K} p\big(V_k \mid I_k, c_k\big), \qquad I_{k+1} = \mathrm{last}\big(\mathrm{Dec}(V_k)\big) \;\text{or independent} \qquad (12)
$$

**符号**：$V_k$：第 $k$ 段视频；$\mathrm{last}(\cdot)$：取末帧（经 VAE decoder 回像素）。$I_{k+1}$ 取上一段末帧时段间连续（由 §1.4 的首尾帧接口承载）；取独立 T2I 图时是硬切镜头，蒙太奇语义由关键帧序列承担。

```
I_1 ─► [段 1: p(V_1 | I_1, c_1)] ─► V_1 ─► last(Dec(V_1)) = I_2 ─► [段 2: p(V_2 | I_2, c_2)] ─► V_2 ─► ...
          首尾帧接口（§1.4）                （或 I_2 独立 T2I：硬切镜头）
```

两个性质直接读出来：(i) **段间无 attention**——全局一致性只经 $I_k$ 这一个压缩接口传递，凡不在关键帧里的信息（主体身份细节、光照、画风）都有逐段漂移的自由度，且漂移沿链复合——与 9.3 分块自回归的 exposure bias 同族（条件分布失配的链式累积），差别是这里失配发生在「段分布 vs 接口图像」之间而非帧级；(ii) **成本可加、大体可并行**——链式依赖只在 $I_{k+1}$ 取末帧时存在，独立关键帧的段可全部并行；总成本 = $K$ 段各自成本之和（9.2 §2.5 的单段账逐段乘），无跨段二次项。关键帧本身的生产（T2I、人工、上一段末帧）不在本章范围；9.3 的「keyframe+插值」长视频路线是同一思想在单次生成内部的形态。

## 总结

| 族 | 机制 | 公式 | 关键性质 |
| --- | --- | --- | --- |
| I2V 输入拼接 | latent concat + mask | 式 (1)(2)：36 = 16+16+4（Wan2.1-I2V config） | 首层增量 $\sim 10^{5}$ 参数；$\mathcal{A} = \varnothing$ 退化回 T2V |
| 条件加噪 | noise augmentation | 式 (3)：$\hat z_c = z_c + \sigma_c \eta$ | 破复制粘贴捷径；对齐训练/推理条件分布 |
| 语义图像条件 | CLIP embedding cross-attn | `clip_fea`：MLPProj(1280→$d$) 注入（Wan 代码实测） | 像素级（concat）与语义级（CLIP）双通道分工 |
| 结构控制 | ControlNet 可训练副本 + zero-conv | 式 (4)：零初始化 ⇒ 起点即原模型 | 门先开、支路后动；与 adaLN-Zero / LoRA 同族 |
| 相机控制 | Plücker 射线图 | 式 (5)：$(d,\; o \times d) \in \mathbb{R}^{6}$ / pixel | moment 与取点无关；旋转/平移分置；与网格同形 |
| 轨迹控制 | drag point 高斯图 | 式 (6)：$\exp(-\lVert \cdot \rVert^{2} / 2\sigma_g^{2})$ | $\sigma_g$ = 软约束半径；多点可表旋转/形变 |
| 编辑 | DDIM inversion + null-text | 式 (7)(8)(9) | 误差经 Jacobian 复合；$\varnothing_t$ 吸收引导场偏差 |
| 主体一致性 | LoRA + DreamBooth | 式 (10)(11) | $r(d{+}k)$ 参数、可合并；prior preservation 防漂移 |
| 叙事组合 | keyframe 链 | 式 (12)：$\prod_k p(V_k \mid I_k, c_k)$ | 一致性只过接口帧；成本可加、可并行 |

## 练习

1.（计算）I2V 输入账。按 patch=1 等效口径与 Wan 14B 几何（$d$ = 5,120，$N$ = 302,400）：(a) 算 T2V→I2V 首层投影的参数与 FLOPs 增量（16→36 通道），及其相对 9.2 §2.5 单层总账（2.06 PFLOPs）的占比；(b) 首尾帧控制（$\mathcal{A} = \{0, t-1\}$）相对单锚点，输入通道数与首层参数变不变？说明理由；(c) 若把 mask 从 4 通道改成 1 通道，首层省多少参数——这个省法为什么不值得做？

2.（推导）零初始化门控三实例。(a) 对式 (4) 证明：$W_2 = 0$ 时 $\partial y / \partial W_2 = F_c(x + \mathcal{Z}(c);\, \Theta_c) \neq 0$ 而 $\partial y / \partial \Theta_c = 0$，并说明一步梯度更新后这两个梯度如何变化；(b) 对 LoRA（式 (10)，$B = 0$ 初始化）证明对称结论：$\partial L / \partial B \neq 0$、$\partial L / \partial A = 0$；(c) 对照 7.6 式 (8) 的 adaLN-Zero 推导，抽出三者共享的充分条件（门参数零初始化 + 门处梯度非零），并构造一个「零初始化但门梯度也为零」的反例说明该条件不可缺。

3.（推导）Plücker 几何。取 $K = \mathrm{diag}(f, f, 1)$（主点在原点）、$R = I$、$t = [t_x, 0, 0]^{\top}$：(a) 写出像素 $(u, v)$ 的 $d(u, v)$ 与 moment $o \times d$ 的闭式；(b) 验证 moment 与射线上取点无关（§3 性质 1）；(c) 两个平行相机（同 $R$、$t$ 相差 $x$ 向平移 $\Delta$）在同一像素处的 Plücker 向量差是多少——视差信息落在哪几个通道？(d) $\lVert o \times d \rVert$ 等于什么几何量？

4.（分析）inversion 误差与编辑预算。(a) 设每步局部误差有界 $\lVert \delta_t \rVert \le \delta$、Jacobian 谱范数有界 $\lVert J_t \rVert \le \gamma$，由式 (8) 给出 $\lVert e_T \rVert$ 的界，并讨论 $\gamma < 1$、$\gamma = 1$、$\gamma > 1$ 三种情形下步数 $T$ 的角色；(b) 从「反演是在哪个场上积的」论证 $w > 1$ 的引导场为什么使式 (7) 的反演失效，以及式 (9) 的 $\varnothing_t$ 为什么能修复；(c) 一段 81 帧视频做 null-text 编辑（$K$ = 50 步、每步内循环 $n$ 次迭代、每迭代计 2 NFE）的总 NFE 账（inversion + 优化 + 再生成三项分列），并对照 DreamBooth 式「一次训练多次复用」的固定成本，说明逐样本付费路线的适用边界。

## 参考

- [Team Wan, 2025](../references.md)：Wan: Open and Advanced Large-Scale Video Generative Models，arXiv:2503.20314，[A]（key: wan）——§1.2 noise augmentation 配方
- [Alibaba, Wan-AI/Wan2.1-I2V-14B-720P config](../references.md)：[A]（key: wan21-i2v-config）——§1.1 `in_dim: 36` = 16+16+4
- [Alibaba, Wan-Video/Wan2.1](../references.md)：[A]（key: wan2.1-repo）——§1.3 `clip_fea` 与 `img_emb = MLPProj(1280, dim)` 走读锚点
- [Team Wan, Wan-Video/Wan2.2](../references.md)：[A]（key: wan2.2）——§1.1 T2V/I2V 同仓发布形态（无论文，官方 README 口径）
- [Radford et al., 2021](../references.md)：CLIP，arXiv:2103.00020，[A]（key: clip）——§1.3 图像语义条件通道的编码器（训练机制 6.2 已拆）
- [Zhang et al., 2023](../references.md)：Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet），arXiv:2302.05543，[A]（key: controlnet）——§2 可训练副本 + zero-conv 出处
- [Zhang, lllyasviel/ControlNet](../references.md)：[A]（key: controlnet-repo）——§2 走读锚点（ldm/ 目录）
- [Peebles & Xie, 2023](../references.md)：DiT，arXiv:2212.09748，[A]（key: dit）——§2 adaLN-Zero 同思想对照（7.6 式 (8)）
- [He et al., 2024](../references.md)：CameraCtrl: Enabling Camera Control for Text-to-Video Generation，arXiv:2404.02101，[A]（key: cameractrl）——§3 Plücker embedding 出处
- [He et al., hehao13/CameraCtrl](../references.md)：[A]（key: cameractrl-repo）——§3 走读锚点
- [Wang et al., 2023](../references.md)：MotionCtrl: A Unified and Flexible Motion Controller for Video Generation，arXiv:2312.03641，[A]（key: motionctrl）——§3/§4 相机+物体运动统一控制
- [Song et al., 2021](../references.md)：DDIM，arXiv:2010.02502，[A]（key: ddim）——§5.1 inversion 出处（7.4 式 (5) 的逆向使用）
- [Mokady et al., 2022](../references.md)：Null-text Inversion for Editing Real Images using Guided Diffusion Models，arXiv:2211.09794，[A]（key: null-text-inversion）——§5.2 逐步空条件优化出处（官方实现在 google/prompt-to-prompt 的 `null_text_w_ptp.ipynb`）
- [Hu et al., 2021](../references.md)：LoRA，arXiv:2106.09685，[A]（key: lora）——§6 $\Delta W = (\alpha/r)BA$ 出处
- [Ruiz et al., 2022](../references.md)：DreamBooth，arXiv:2208.12242，[B]（key: dreambooth）——§6 双损失微调出处（google/dreambooth 仅数据集、无官方训练代码，按论文拆公式）
