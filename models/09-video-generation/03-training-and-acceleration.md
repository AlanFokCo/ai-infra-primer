# 9.3 训练配方与加速/流式生成（Training Recipes and Accelerated/Streaming Generation）

> 9.2 把成本钉在了桌面上：81 帧 720p 的单步去噪 82.3 PFLOPs、50 步 ≈ 4.1 EFLOPs。本章回答两问——这笔账在训练侧怎么组织（图视频混训的三条结构兼容条件、渐进分辨率/时长课程、logit-normal 接口、视频 CFG 的移植与变体），以及在推理侧怎么砍掉一个数量级以上（DMD2 的视频化、CausVid 非对称蒸馏 50 → 4 步、KV cache 流式推理——2.4 的 prefill/decode 数学在块因果视频 DiT 上逐项重放）；随后两个改写训练目标本身的框架：Diffusion Forcing（per-token 独立噪声水平，teacher forcing 与全序列扩散的统一）与 Self-Forcing（训练时自 rollout，exposure bias 的形式化与消除）；收尾是 2026 现状表（Causal-Forcing 系、rCM 系）、长视频三路线的账与交互式世界模型展望 | 前置：7.5（FM 目标式 (22)、Euler 采样式 (23)、logit-normal 式 (27)-(29)、timestep shift 式 (30)(31)）、7.4（CFG 式 (18)(19)、guidance 蒸馏 §4.5、DMD 梯度式 (30)-(32)、DMD2 注记 §5.5）、7.3（DSM §3）、2.4（causal mask §1、前缀不可变命题 §4.2、cache 账式 (9)、prefill/decode §3、teacher forcing §7）、9.1（latent 几何式 (2)、首帧兼容 §2.2、scaling factor §3）、9.2（attention FLOPs 式 (2)、Wan 14B 单步账 §2.5、bucket 表 §9、3D RoPE §8、SNR 式 (7)） | 锚点：tianweiy/CausVid（configs/wan_causal_dmd.yaml）[A]、guandeh17/Self-Forcing [A]、thu-ml/Causal-Forcing [A]、NVlabs/rcm [A]、buoyancy99/diffusion-forcing [A]、SkyworkAI/SkyReels-V2 [A]；CausVid [A]、Self-Forcing [A]、Diffusion Forcing [A]、Causal-Forcing / Causal-Forcing++ [A]、rCM / Causal-rCM [A]、SkyReels-V2 [A]、DMD / DMD2 [A]（recap 7.4）、Movie Gen [B]

记号口径（本章声明）：扩散噪声水平 $t$ 沿用 W4 约定（ $x_0$ = 数据端、 $t = 1$ = 纯噪声端、采样方向 $t : 1 \to 0$；FM 线性路径 $x_t = (1-t) x_0 + t \varepsilon$、目标速度 $\varepsilon - x_0$）；序列元素记 $x^{(i)}$——视频语境下即第 $i$ 个 latent 帧（9.1 的 $z$ 的时间切片， $h \times w \times c$ 网格）；chunk（若干连续 latent 帧组成的分块）用下标 $k = 1..M$；latent 帧数只在时长账语境出现，沿用 9.1 式 (2)（81 帧 → 21 latent 帧），与噪声水平语境不重叠。

## 1. 训练配方：图像-视频联合训练与渐进课程

9.2 §9 的分桶表先给出本节问题的成本底色：同一模型从最低桶（480p × 33 帧）到最高桶（720p × 120 帧）的单步 attention 二次项成本差 63.2 倍。训练配方的全部设计——混训、课程、采样分布——都是在这个成本地形上组织的。

### 1.1 T2I 权重作初始化：三个结构兼容条件

视频 DiT 用 T2I 权重初始化之所以合法（而非工程玄学），由前三章各出一条结构性质拼成：

1. **latent 空间同一**（9.1 §2.2）：因果 3D VAE 下，一张孤立图像的编码与它作为视频首帧的编码逐位一致——图像与视频共享同一 latent 空间，scaling factor 的标定（9.1 §3）对两种数据同一次成立。
2. **位置编码退化**（9.2 §8）：3D RoPE 在 $t$ 坐标恒定时自动退化——全部 token 的 $\Delta t = 0$， $J_t$ 频段对 score 的贡献恒为零位移项，退化为两轴 RoPE-2D；「 $t = 1$ 的图像桶」与「 $t = 21$ 的视频桶」共用同一套频段切分，零改造。
3. **attention 退化**：full 3D attention 作用在单帧序列上就是帧内空间 full attention——9.2 式 (2) 的成本账在 $t = 1$ 时逐字退化为图像 DiT 的账。

三条合起来的含义：T2I DiT 的权重在视频架构上从第 0 步就是良定义的函数，图像数据过视频架构的前向在数学上是普通前向的特例。初始化买到的是空间先验——纹理、构图、局部结构，这些从量大价低的图像数据学得；买不到的是时间轴上的耦合——帧间一致性只能由视频数据提供。联合训练（图像-视频混训）因此不是「顺便用一下图像数据」，而是两种数据各管一个正交能力轴的分工：图像桶继续钉住空间质量，防止规模小得多的视频数据把空间先验冲刷掉；视频桶独供时间监督。目标函数同形——图像样本以「单 latent 帧序列」的形态进入同一个 FM 损失：

$$L_{joint}(\theta) = \mathbb{E}_{(x_0,\thinspace c) \sim \mathcal{D}} \hspace{5mu} \mathbb{E}_{t \sim \pi,\hspace{5mu} \varepsilon \sim \mathcal{N}(0, I)} \left\Vert v_\theta\left( (1 - t)\thinspace x_0 + t\thinspace \varepsilon,\hspace{5mu} t,\hspace{5mu} c \right) - \left( \varepsilon - x_0 \right) \right\Vert^2 \qquad (1)$$

**符号**： $`\mathcal{D} = \mathcal{D}_{img} \cup \mathcal{D}_{vid}`$：图像与视频的混合数据分布（混合比例是配方自由度）； $x_0$：latent（图像样本的 $x_0$ 是 $1 \times h \times w \times c$）；其余同 7.5 式 (22)。混训在目标层面零改动——全部差异在数据分布与桶调度里。

诚实边界：各家具体的混训比例、图像桶占比与课程参数，本书事实源未逐一登记，本节给的是该配方族成立的结构条件与成本论证；机制断言（三条兼容条件）逐条有 9.1/9.2 的已登记事实支撑。

### 1.2 渐进分辨率/时长课程的成本账

把 9.2 §9 的桶成本表换个读法，它就是课程表。设课程分若干阶段、阶段 $s$ 在桶成本 $C_s$ 上花 $K_s$ 步，总成本 $= \sum_s K_s C_s$。渐进课程（低分辨率 + 短时长先行、高分辨率 + 长时长收尾）的机制依据是训练动态的结构：去噪任务的学习进度有明确的噪声水平-分辨率分工——构图与运动学的粗结构在中高噪声、低分辨率段就能被学到（7.5 §8 的 log-SNR 论据：语义结构在 SNR 中段定型），这些进度在 1/63 单步成本的低桶上以同质量推进；高桶课程段只需负责细节带的收尾。反事实账：全程在最高桶训练，同预算下步数被砍到 1/63——训练不充分；同步数下成本放大 63 倍——预算不可行。9.2 §9 的判词原样成立：渐进课程不是优化项，是让训练总成本落在可行域的必要条件。

配套旋钮是分辨率相关的 timestep shift（7.5 式 (30)(31)）：分辨率升高后固定 $t$ 的去噪任务变易（像素冗余度上升），shift 在 logit 空间做常数平移 $\operatorname{logit}(t') = \operatorname{logit}(t) + \log s$ 补偿——视频侧同样适用，各家的具体 shift 取值事实源未登记，不杜撰。

### 1.3 timestep 采样：logit-normal 接口与逐 token 推广伏笔

$t$ 的采样分布 $\pi$ 是纯加权设计：不改最优解，只改各噪声水平分到的训练样本数。标准取点 recap 7.5 式 (27)-(29)： $t = \operatorname{sigmoid}(u)$、 $u \sim \mathcal{N}(m, s^{2})$，等价于在 log-SNR 轴上放高斯 $\log \mathrm{SNR} \sim \mathcal{N}(-2m, 4s^{2})$——预算压向 SNR ≈ 1 的中段（语义结构定型处），避开两端（近纯数据的微小残差、近纯噪声的退化任务）。

视频侧同一个旋钮，两处新语境值得点名：(i) 与 9.2 §6 的 SNR-split MoE 在同一坐标系联动——路由阈值按 SNR 定（9.2 式 (8)），训练样本按 log-SNR 高斯分布，两个 expert 各自分到的梯度份额由 $\pi$ 在阈值两侧的质量直接决定；(ii) 它是 §5 的退化角点——Diffusion Forcing 把这个标量分布推广为逐 token 的乘积分布 $\pi^{\otimes n}$，单标量版本正是「全序列共享 $t$」截面上的特例。

## 2. 视频 CFG：公式移植、成本账与变体自由度

CFG 公式（7.4 式 (18) 的推导）对「 $x$ 是什么形状」无感知——把图像 latent 换成整段视频 latent，公式逐字成立。速度场坐标（FM 系模型，9.2 §5 的 Wan 谱系）下的写法：

$$\tilde v_\theta(x_t, t) = v_\theta(x_t, t, \varnothing) + w \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing) \right) \qquad (2)$$

**符号**： $x_t$：噪声水平 $t$ 处的整段视频 latent； $c$ / $\varnothing$：文本条件 / 空条件（condition dropout 训出的同一网络两次求值，7.4 §4.2）； $w$：引导强度（guidance scale）； $\tilde v_\theta$：替换采样更新里的速度预测。

分布语义原样继承：式 (2) 对应的采样分布是 tilt 后的 $q_w(x) \propto p(x)\thinspace p(c \mid x)^{w}$（7.4 式 (19)），视频侧的读法只剩一处不同——分类器项的「匹配」现在是整段 clip 与 prompt 的匹配，运动描述、镜头语言都在 $p(c \mid x)$ 的覆盖范围内；引导方向 $\Delta = v_\theta(c) - v_\theta(\varnothing)$ 是一个定义在全部 latent 帧上的时空场， $w$ 对它统一放大。

成本账是硬的：NFE 逐步 ×2。代入 9.2 §2.5 的 Wan 14B 几何：单步 82.3 PFLOPs → CFG 后 164.6 PFLOPs/步，50 步一段 81 帧视频 ≈ 8.2 EFLOPs（9.2 的 4.1 EFLOPs 翻倍）。失效模式（7.4 式 (20) 的 score norm 随 $w$ 膨胀 → 过饱和、分布失配累积、多样性塌缩）逐帧适用；时间维新增一条风险（本书机制分析，非收录实证）：若逐帧独立变化引导强度 $w_i$，不同帧被 tilt 的程度不同，时空场的时间切片间一致性被打破，表现为帧间风格漂移——标准做法是全程统一标量 $w$。

变体的公式级自由度： $w$ 推广为噪声水平的函数

$$\tilde v_\theta(x_t, t) = v_\theta(x_t, t, \varnothing) + w(t)\thinspace \Delta_\theta(x_t, t), \qquad \Delta_\theta(x_t, t) = v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing) \qquad (3)$$

**符号**： $w(t)$：逐噪声水平的引导强度 schedule。依据：7.4 §4.2 的推导在逐 $t$ 的 score 空间进行， $w$ 本不必须是常数——每个噪声水平的 tilt 指数可独立选择；引导的收益与失效沿噪声轴分布不均（构图在高噪声段定型——7.4 §4.1 的隐患 (iii) 与 7.5 §8 的 log-SNR 论据同向）， $w(t)$ 把「在哪段噪声水平买对齐」变成显式旋钮。视频系统具体使用的 $w(t)$ schedule 事实源未登记，本节只给框架。

流式语境的交互（§4 的前置账）：chunk-causal 推理下跑 CFG，两个分支的文本侧计算可共享——文本编码一次，cross-attn 的文本侧 K/V 由冻结文本决定、两分支相同；视频侧不可共享——cross-attn 输出从第一层起在两分支间不同，下游 self-attention 的 K/V 随之全部分叉，cache 与二次项成本双双 ×2。蒸馏语境的标准解法是把引导折进学生：分布匹配的目标分布是教师（含引导）的输出分布，学生整体模仿后即 CFG-free（机制即 7.4 §4.5 guidance 蒸馏；LCM 在图像侧把 CFG 与步数同蒸，7.4 §5.3）——§3 的 CausVid 属同一谱系。机制动机很直接：4 步学生若推理时再叠 CFG 就是 8 NFE/段，蒸馏省下的步数被吃掉一半。

## 3. 步数蒸馏进视频：DMD2 组件映射与 CausVid 非对称蒸馏

### 3.1 DMD 族组件盘点与视频化映射

7.4 §5.5 的组件清单（recap，不重推）：DMD（[Yin et al., 2023](../references.md)，arXiv:2311.18828，[A]）三件套 = (i) 分布匹配梯度——fake score 与 real score 之差经生成器 Jacobian 回传（7.4 式 (32)）；(ii) 在线 fake score 网络——随生成器并行训练、在生成器当前输出上做 DSM（7.3 §3）；(iii) 教师 ODE 配对回归——作初始化与稳定器。DMD2（[Yin et al., 2024](../references.md)，arXiv:2405.14867，[A]）改动两条：去掉 (iii)（免教师数据），**加 GAN 分支**。

CausVid（[Yin et al., 2024](../references.md)，arXiv:2412.07772，[A]；官方 repo [tianweiy/CausVid](../references.md)，[A]）是这套机器的视频化实例，组件映射：

| 组件 | DMD（图像） | DMD2（图像） | CausVid（视频） |
| --- | --- | --- | --- |
| 分布匹配梯度（reverse KL，逐随机噪声水平） | ✓ | ✓ | ✓（式 (5)(6)） |
| 在线 fake score 网络（DSM） | ✓ | ✓ | ✓ |
| 教师 ODE 配对回归 | ✓（稳定器） | 删除 | 恢复为预训阶段（式 (4)，论文 §4.3） |
| GAN 分支 | 无 | 加 | **无**（见下） |
| 教师-学生架构关系 | 不限定 | 不限定 | **双向教师 → 因果学生**（非对称） |

**GAN 归属澄清**（spec 明令陷阱）：CausVid 方法部分没有 GAN 分量——ar5iv 全文口径，「GAN」字样只出现在 related work 对早期工作的引述中；GAN 分支是 DMD2 的组件，与 CausVid 无关，勿张冠李戴。CausVid 的全部训练信号 = 式 (4) 的 ODE 配对回归（预训）+ 式 (5)(6) 的 DMD 梯度（主阶段）。

### 3.2 CausVid 的目标函数：ODE 初始化 + DMD 主阶段

```
双向教师（50 步 FM 采样；官方 repo 释出基座 Wan2.1-T2V-1.3B）
   │  ① 教师跑 ODE 生成配对（小集合）          ② 冻结，提供 s_data（real score）
   ▼                                            │
因果学生 G_ψ（块因果 mask）── 训练：全体 chunk 各采 t^j（含 0），以带噪历史为条件逐 chunk 一步预测、拼接成 x
   │                                            │
   │            x 加噪到随机 t ────────────────► s_ψ（在线 fake score）
   ▲                                            │
   └──── 梯度 = (s_ψ − s_data) · ∂x/∂ψ ◄────────┘   （式 (6)，无 GAN）
        （反传只穿过每 chunk 的一次前向；4 步复合只在推理，论文 Algorithm 2）
```

**阶段一：ODE 轨迹初始化**（论文 §4.3）。教师用自己的确定性采样从噪声生成解轨迹，学生在其锚点时间上对轨迹终点做回归预训：

$$L_{ODE}(\psi) = \mathbb{E}_{x_1 \sim \mathcal{N}(0, I),\hspace{5mu} j \in \lbrace1, \ldots, K\rbrace} \left\Vert G_\psi\left( x^{teacher}_{s_j},\hspace{5mu} s_j \right) - x^{teacher}_{0} \right\Vert^2 \qquad (4)$$

**符号**： $`x^{teacher}_{s}`$：教师从 $x_1 \sim \mathcal{N}(0, I)$ 出发的 ODE 解轨迹在噪声水平 $s$ 处的取值； $\lbrace s_j\rbrace_{j=1..K}$：学生的 $K = 4$ 个锚点时间（采样网格，方向 $s : 1 \to 0$）； $G_\psi(x, s)$：学生从 $(x, s)$ 一步映射到数据端的预测（ $K$ = 4 步采样由该一步映射经重噪迭代复合而成——few-step 学生的标准用法，7.4 §5 谱系）； $`x^{teacher}_{0}`$：该轨迹的教师终态。这一构造与 DMD 初代的配对回归同族（7.4 §5.5），DMD2 删除、CausVid 恢复为独立预训阶段。机制读法（本书分析）：DMD 主阶段的 real score 由教师在「噪声化的学生输出」上评估——学生输出若离数据流形太远，加噪后的输入落在教师的训练分布之外、score 信号失真；ODE 预训先把学生钉到教师解流形附近，主阶段的监督才良性。

**阶段二：非对称 DMD 主目标**（论文 Algorithm 1「Asymmetric Distillation with DMD」）。训练输入的构造先钉死（这是后面式 (6) 的 $x$ 的出处）：逐 chunk 采样——**全体** chunk 各自采噪声时间步 $t^{j} \sim \mathrm{Uniform}(\mathcal{T})$（含 0）并各自加噪，历史 chunk 同为带噪的 ground truth；学生以此带噪历史为条件、把当前（ground-truth）chunk 的加噪输入**一步**映到数据端 $`\hat x_0^{(i)} = G_\psi\left( x^{(i)}_{t^{i}},\thinspace t^{i};\thinspace x_{t^{<i}}^{(<i)} \right)`$，各 chunk 的预测拼接成整段视频 $x$（论文 Algorithm 1 lines 5-9 口径）。输入侧的这个构造在 chunk 粒度上已是 Diffusion Forcing 方案——论文 §4.2 原文「Each chunk has its own noise time step, following Diffusion Forcing」（Self-Forcing 论文 §2 亦以此定性 CausVid，其对 DF 的刻画为「denoising each frame based on noisy context frames」）；§5 的 DF 框架是同一思想在 token 粒度的推广、并给出变分下界口径。记 $p_\psi$ 为学生输出分布（上述逐 chunk 一步预测拼接的分布）、 $p_{data}$ 为数据分布， $p_{\psi, t}$ / $p_{data, t}$ 为各自加噪到水平 $t$ 的平滑分布（FM 线性路径 $x_t = (1-t)x + t\varepsilon$ 的边际）。目标是在随机采样的噪声水平上最小化 reverse KL：

$$L_{DMD}(\psi) = \mathbb{E}_{t \sim \pi} \left[ \mathrm{KL}\left( p_{\psi, t}\thinspace \Vert\thinspace p_{data, t} \right) \right] \qquad (5)$$

**符号**： $\pi$：噪声水平采样分布（随机 timestep 口径与 7.4 §5.5 一致）。reverse KL 的方向语义：学生把质量放到数据无质量处会被严惩（mode-seeking；与 §7 rCM 条目的「forward KL mode-covering + reverse KL mode-seeking 互补」口径互证）。

梯度由 7.4 式 (30)-(32) 的推导在视频记号下重放：交叉项经重参数化链式展开、熵项拆出测度项（ $`\mathbb{E}_{p_\psi}[\partial_\psi \log p_\psi] = 0`$，密度归一的恒等推论）与样本点链式项，合并得

$$\nabla_\psi\thinspace \mathrm{KL}\left( p_{\psi, t}\thinspace \Vert\thinspace p_{data, t} \right) = \mathbb{E}_{x \sim p_\psi,\hspace{5mu} \varepsilon} \left[ \left( s_{\psi, t}(x_t) - s_{data, t}(x_t) \right)\thinspace \frac{\partial x}{\partial \psi} \right], \qquad x_t = (1 - t)\thinspace x + t\thinspace \varepsilon \qquad (6)$$

**符号**： $x \sim p_\psi$：学生输出——上文训练构造的逐 chunk 一步预测拼接（每 chunk 一次前向）， $\partial x / \partial \psi$ 只穿过这一次前向；推理侧的 4 步复合（论文 Algorithm 2）不进训练计算图。 $s_{data, t}$：水平 $t$ 处的真实 score，由**冻结的双向教师**经 score↔ε 桥（约定页第 2 条）在 $x_t$ 上求值给出； $s_{\psi, t}$：水平 $t$ 处学生分布的 score，由并行训练的 fake score 网络在学生的加噪输出上经 DSM 回归给出。逐项与 7.4 式 (32) 同构，唯一变化是「 $x$ 是什么」与「两个 score 各自由什么架构的网络给出」。

**非对称的精确含义**（论文原词 asymmetric distillation 落点）： $s_{data, t}$ 由双向教师计算——教师在每个位置见到整段视频； $G_\psi$ 是因果学生——第 $k$ 个 chunk 的生成只见到 $1, \ldots, k-1$（§4 的 mask）。监督信号的信息量严格多于学生架构可用的上下文：教师 score 场在学生输出的每个位置上都编码了全视频统计（含该位置的未来帧），这个信号经 $\partial x / \partial \psi$ 回传，把全局一致性约束蒸馏进只能向过去看的因果权重。不动点论证不受非对称影响：式 (6) 为零 $\Longleftrightarrow$ $s_{\psi, t} = s_{data, t}$ 几乎处处成立 $\Longleftrightarrow$ $p_{\psi, t} = p_{data, t}$（KL 的唯一不动点，7.4 §5.5 的逐样本力读法）——目标分布里没有「学生能看多少上下文」这个自由度，因果性只限制学生逼近不动点的方式（全局协调必须预先编码进因果权重），不改变不动点本身。

### 3.3 数字账：50 → 4 步与延迟/吞吐

论文端到端口径（10 秒 / 120 帧负载，[Yin et al., 2024](../references.md)，arXiv:2412.07772，[A]）：50 步 → 4 步；流式吞吐 9.4 FPS（frames per second）、首帧延迟 1.3 s；160× 延迟削减 / 16× 吞吐提升是 vs CogVideoX-5B 的口径（论文口径）；vs 双向教师为 219.2 s → 1.3 s ≈ 169×、0.6 → 9.4 FPS ≈ 16×（Table 3，640×352）；质量侧 VBench-Long 84.27。官方 repo 的释出基座是 Wan2.1-T2V-1.3B（README 口径：实现基于 Wan 模型族；4 步配置锚点 configs/wan_causal_dmd.yaml，[tianweiy/CausVid](../references.md)，[A]）。

两个加速比不在同一量纲：延迟是 time-to-first-frame（只付第一个 chunk 的 4 步），吞吐是稳态帧率（每个 chunk 4 步、历史走 cache）。步数比 12.5× 是两个数字共同的第一来源，剩余部分由 §4 的流式账分解。

## 4. KV cache 流式推理：视频生成与 LLM 推理数学的合流点

### 4.1 块因果 mask 与前缀不可变性的重放

因果化改造的唯一手术：把 9.2 的全双向 attention 换成块因果（block-causal）mask——latent 帧序列分 chunk $C_1, \ldots, C_M$（连续帧段），token 间的可见规则为「chunk 内双向、chunk 间只见过去」。

**命题（块版前缀不可变）**：块因果 mask 下，chunk $k$ 内任一 token 在任意层的隐状态是 chunk $1, \ldots, k$ 的函数，与 $k+1, \ldots, M$ 无关。证明是 2.4 §4.2 的逐层归纳原样重放：基例（embedding 逐 token）不变；归纳步里 attention 的求和上限从「位置 $s$」换成「chunk $k$ 的边界」，可见集仍不含未来，RMSNorm / FFN / 残差加的逐位置性不变。∎

推论与 2.4 逐字平行：chunk $k$ 的逐层 K/V 一经算出永不改变（未来 chunk 不回流），可存可复用——**可缓存性与因果性在视频 DiT 上还是同一个性质的两面**。这条链向上游还接着一环：9.1 §2.2 的 VAE 时间因果保证 latent 帧序本身无未来泄漏（「以过去 latent 为条件推进」的输入侧良定义），本节的 DiT 块因果保证计算侧可流式——两层因果拼出端到端的流式生成。

### 4.2 双层循环：chunk 自回归 × K 步去噪

```
prefill：文本条件编码（一次）＋（可选）初始帧 latent 的 K/V 写入 cache
──────────────────────────────────────────────────────────────
外层（自回归，chunk k = 1..M）：
    x ← ε ~ N(0, I)                    # 当前 chunk 初始化在噪声端（t = 1）
    内层（去噪，步 j = 1..K，K = 4；s_j: 1 → 0）：
        v ← G_ψ(x, s_j ; cache)        # 当前 chunk 的 K/V 每内层步重算
        x ← x − h · v                   # 历史 chunk 的 K/V 只读，全程不变
    emit chunk k（去噪到 t = 0）；其干净态 K/V append 进 cache
──────────────────────────────────────────────────────────────
2.4 §3 的逐项映射：prefill ↔ 文本条件与初始帧；decode 一步 ↔ 外层一个 chunk；
唯一增量：「decode 一步」内部嵌了一个 K 步扩散循环
（示意按 FM 确定性 Euler 写；CausVid 论文 Algorithm 2 的采样更新为祖先式——
  每步注入新噪声 x ← α·x̂₀ + σ·ε′，KV caching 下推理不再需要显式块因果 mask）
```

cache 的读写语义有一处扩散特有的细节值得钉死：历史 chunk 是已 emit 的干净 latent（ $t = 0$），其 K/V 在 emit 时计算一次，之后被全部后续 chunk × 每 chunk 全部 $K$ 个内层步复用；当前 chunk 的输入随内层步变化（ $x$ 在退火），其 K/V 每步重算、不进长期 cache。这是 2.4 §4.3「列被未来读、行只被自己读」的视频版：干净历史是被未来反复读的列，退火中的当前 chunk 是一次性消费品。

**FLOPs 账**（逐层，只写 attention 二次项）：chunk $k$ 的每个内层步，queries 是当前 chunk 的 $n_k$ 个 token，keys 是 $T_{k-1} + n_k$（ $T_{k-1}$ = 历史 token 数）， $QK^{\top}$ 与 PV 各 $2 d\thinspace n_k (T_{k-1} + n_k)$。均匀 chunk（ $n_k = n$、 $N = Mn$）下全视频、每内层步合计：

$$\sum_{k=1}^{M} 4 d\thinspace n \big( (k-1)n + n \big) = 4 d n^{2} \cdot \frac{M(M+1)}{2} = 2 d N^{2} \left( 1 + \frac{1}{M} \right) \qquad (7)$$

**符号**： $d$：宽度； $n$：chunk token 数； $M$：chunk 数； $N$：全视频 token 数（约定页 patch=1 等效口径）。

$K$ 个内层步把式 (7) 乘 $K$。对照双向教师：每步 $4 d N^{2}$（9.2 式 (2) 二次项）、 $K_T = 50$ 步。二次项总账比值：

$$\frac{K \cdot 2 d N^{2} (1 + 1/M)}{K_T \cdot 4 d N^{2}} = \frac{K (1 + 1/M)}{2 K_T} \approx \frac{4}{100} = \frac{1}{25}$$

（末步取 $M \gg 1$ 的极限； $M$ = 7 的精确值为 1/21.9。）

——两个乘性来源分解干净：**步数比 12.5×（蒸馏）× 三角因子 2×（因果 mask 砍掉 score 矩阵上三角）**。结构与 2.4 式 (8) 的恒等式同型：流式串行与一次性因果并行前向的 FLOPs 账恒等，cache 买的是墙钟时间与延迟，不是渐近算力。

**延迟账**（time-to-first-frame）：双向教师须走完全部 50 步才存在任何一帧，latency $\approx K_T \cdot C(N)$（ $C$ = 单步全序列前向成本）；因果学生的首帧延迟 $\approx K \cdot C(n_1)$（ $n_1$ = 首 chunk token 数）。比值上界 $\approx (K_T / K) \cdot C(N) / C(n_1)$； $C$ 在二次项主导区（9.2 §2.5：91%）随 token 数近平方缩放， $C(N) / C(n_1) \approx (N/n_1)^{2} = M^{2}$ 量级——延迟削减的第二来源是「首帧前必须处理的 token 数」从全视频缩到首 chunk。论文实测 160× 低于这个上界估计：小 chunk 下 kernel 效率回落、VAE 逐块解码与文本编码等固定开销吃掉一部分（本书机制分析；端到端数字以论文为准）。

**吞吐与 cache 账**：稳态每 chunk 付 $K$ 步 × 当前 chunk 与历史 cache 的前向成本（queries $n$、keys $T_{k-1} + n$）；cache 字节数随历史线性长——2.4 式 (9) 逐项平移：每层 K、V 各 $H_{kv} \cdot d_{head} \cdot T_k \cdot B \cdot b$ bytes（bf16 $b = 2$），attention 二次项随历史线性长（式 (7) 的 $T_{k-1}$ 项）——无限长流下吞吐随历史缓降。把历史上限钉死（滑窗只留最近若干 chunk 的 cache）即得有界上下文流式，与 §5 Diffusion Forcing 的滑窗调度家族在推理侧互为表里。cache 量级的一个直观参照（用已登记的 14B config 算， $L = 40$、40 头、 $d_{head} = 128$）：81 帧 720p 全历史（ $N$ = 302,400 token）的 self-attn K/V ≈ 248 GB——对照本书 LLM 侧锚点（2.4 §5 的 cache 账表：Llama-3.1-8B @ 131K token 满窗 = 16 GiB、128 KiB/token）：总量口径约 14 倍、per-token 口径（约 800 KiB vs 128 KiB）约 6 倍；视频 cache 的 per-token 压力显著高于同代 LLM，这是流式视频系统选小基座（1.3B）、短 chunk、滑窗的账面动因（机制分析；具体系统的 chunk 配置以各 repo 为准）。

### 4.3 「合流」的严格表述

视频流式生成与 LLM 推理的合流不是类比，是同一组数学对象的逐项重放：causal mask 的加性 $-\infty$ 语义（2.4 §1）→ 前缀不可变（2.4 §4.2 ↔ 本节命题）→ K/V 可缓存的读写模式（2.4 §4.4）→ prefill/decode 二相分解（2.4 §3 ↔ 本节双层循环）→ cache 线性增长与 decode 的访存主导（2.4 式 (9)、2.4 §6）。视频侧的实质增量只有两个：decode 的「一步」内嵌 $K$ 步扩散退火（离散采样动作换成 ODE 段子）；cache 的内容是连续 latent 的 K/V 而非离散 token 的（数学结构相同）。加上 9.1 §2.2 的 VAE feature caching（因果卷积的跨块状态 = 有限过去窗口，同一原理的卷积侧同构物），VAE、DiT attention、采样循环三层共享同一条「因果 ⇒ 可缓存」原理——这是全书第一次同一个机制在三个不同层级上各出现一次。

## 5. Diffusion Forcing：per-token 噪声水平的训练目标（核心推导）

### 5.1 从全局标量 $t$ 到逐 token 向量 $t_{1:n}$

式 (1) 的标准训练里，一段序列共享一个噪声水平 $t$。Diffusion Forcing（[Chen et al., 2024](../references.md)，arXiv:2407.01392，[A]；官方 repo [buoyancy99/diffusion-forcing](../references.md)，[A]）把噪声水平挂到 token 级——摘要原文口径：「a diffusion model is trained to denoise a set of tokens with independent per-token noise levels」。逐 token 加噪（FM 线性路径，约定页第 3、4 条）：

$$x_{t_i}^{(i)} = (1 - t_i)\thinspace x^{(i)} + t_i\thinspace \varepsilon^{(i)}, \qquad t_i \stackrel{iid}{\sim} \pi, \qquad \varepsilon^{(i)} \stackrel{iid}{\sim} \mathcal{N}(0, I) \qquad (8)$$

**符号**： $x^{(i)} \in \mathbb{R}^{h \times w \times c}$：第 $i$ 个 token（视频语境 = latent 帧）； $t_i \in [0, 1]$：该 token 自己的噪声水平（ $t_i = 1$ 纯噪声、 $t_i = 0$ 干净）； $\pi$：逐 token 独立同分布的噪声水平采样分布（uniform、logit-normal（7.5 式 (27)）均为合法取点，选择是框架的自由旋钮）。

训练目标：

$$L_{DF}(\theta) = \mathbb{E}_{x^{1:n} \sim p_{data}} \hspace{5mu} \mathbb{E}_{t_{1:n} \stackrel{iid}{\sim} \pi} \hspace{5mu} \mathbb{E}_{\varepsilon^{1:n} \stackrel{iid}{\sim} \mathcal{N}(0, I)} \left[ \sum_{i=1}^{n} \left\Vert \varepsilon_\theta\left( x_{t_{1:n}}^{1:n},\hspace{5mu} t_{1:n},\hspace{5mu} c \right)^{(i)} - \varepsilon^{(i)} \right\Vert^2 \right] \qquad (9)$$

**符号**： $x^{1:n}$：长度 $n$ 的 latent 帧序列； $t_{1:n} = (t_1, \ldots, t_n)$：逐 token 噪声水平向量； $x_{t_{1:n}}^{1:n}$：按式 (8) 逐 token 加噪后的整序列； $\varepsilon_\theta(\cdot)^{(i)}$：模型对第 $i$ 个 token 的噪声预测（默认 ε-prediction，约定页第 2 条；换成 FM 速度目标只是坐标变换，7.5 式 (25)）； $c$：文本条件。与式 (1) 的唯一差异：标量 $t$ 换成向量 $t_{1:n}$——实现侧的对应改动是时间调制从全局广播改为逐 token（9.2 §3-§5 的 adaLN 族调制接口本就逐 token 作用，把全局 $t$ 换成逐 token $t_i$ 不改架构、只改调制输入；机制注记）。

### 5.2 两个退化角点：全序列扩散与 next-token 预测

式 (9) 的训练分布覆盖 $t_{1:n}$ 的整个乘积空间 $\pi^{\otimes n}$。两个角点把这个空间的语义钉死：

**角点一（全序列扩散）**：约束 $t_1 = \cdots = t_n = t$ 并联合退火——式 (9) 逐字退化为式 (1) 的 ε 坐标版，即本章 §1 与 9.2 的标准视频扩散目标。方向值得反过来读：标准训练是 DF 训练分布的一个低维截面；只在共享 $t$ 下训练的模型（§3 谱系之前的双向基座）从未见过「历史干净 + 当前有噪」的输入，而 §4 流式推理的输入（cache 里 clean、当前 chunk 在噪）对它恰是未见配置（支撑集口径，测度注记见 §5.3）。DF 补上的第一个洞就是这个配置。限定一笔：CausVid 的因果学生已在 chunk 粒度按 DF 方案采样训练输入（§3.2 的训练构造），该缺口对它不存在——它的残余失配只在 §6 的测度轴。

**角点二（next-token 预测）**：取 $t_{1..i-1} = 0$（历史干净）、只对 $t_i$ 加噪——该配置下模型学的是条件族 $p(x^{(i)} \mid x^{<i})$ 在各噪声水平下的平滑版；采样时逐 token 执行「 $t_i : 1 \to 0$ 内层退火 + 历史钉 0」即自回归逐帧生成。这正是 teacher forcing（2.4 §7：训练见 ground-truth 上下文）的扩散版——摘要原文「training a causal next-token prediction model to generate … without fully diffusing past ones」。把内层退火塌缩成 $t_i = 0$ 处的点估计（直接输出条件均值 $\mathbb{E}[x^{(i)} \mid x^{<i}]$），就退回 LLM 式确定性 next-token 预测。论文标题「Next-token Prediction Meets Full-Sequence Diffusion」的公式含义至此完整：同一组权重、同一个目标，两个采样 schedule；teacher forcing 与全序列扩散被统一为同一配置空间上的两个角点。

```
t_1..t_n 的配置空间（每 token 一个噪声水平，0 = 干净，1 = 纯噪声）：

角点一：全序列扩散        中间带（DF 训练覆盖）          角点二：next-token
[ s  s  s  s  s  s ]      [ 0  0  0  s  s  s ]         [ 0  0  0  0  0  s ]
共享 t，联合退火           流式配置：历史钉 0            逐帧 AR（teacher forcing
（9.2 的标准训练只         [ 0  0  s' s  s  s ]          的扩散版），内层退火
 覆盖这一截面）            滑窗：近端历史留小噪声 s'>0    塌缩到点估计 = LLM 式
```

中间带里有一个论文实际使用的家族值得点名：历史不必钉在 0——「不把过去完全去噪」（摘要原文 without fully diffusing past ones）的调度让近端历史保留小噪声水平 $s' > 0$，同样落在式 (9) 的训练分布内。机制读法（本书分析）：rollout 的历史本就是自生成输出、而非 ground truth，训练时见过带噪历史的模型对历史里的微小误差更鲁棒——DF 从输入分布侧拓宽训练覆盖，与 §6 Self-Forcing 从测度侧直接换成 on-policy 是互补的两个方向。

### 5.3 变分下界口径与长视频含义

论文给的理论口径（摘要原文）：式 (9) 的目标「proven to optimize a variational lower bound on the likelihoods of all subsequences」——所有子序列似然的变分下界。读法：扩散目标的 ELBO 性质（7.2 的变分推导谱系）在逐 token 噪声水平下推广为对「任意干净/带噪 token 组合对应的边缘与条件分布」的一致覆盖——模型同时为全部噪声水平配置定义了良定义的生成过程，这正是推理侧可自由选择任意 $t_{1:n}$ 退火顺序（全序列联合 / 逐帧 AR / 滑窗）的许可证。本书按摘要口径引用该定理陈述，证明细节回原论文（[Chen et al., 2024](../references.md)，[A]）。

长视频含义是与 §4 的直接接口：DF 的训练分布天然覆盖流式推理的输入配置——「长历史 clean（或低噪）+ 当前 chunk 有噪 + KV cache 推进」对它不是外推。测度注记（口径校准）：DF 论文原式取离散噪声水平集合（含 0），「历史钉 0」类配置带正概率质量；本章按约定页第 4 条连续化（ $t_i \stackrel{iid}{\sim} \pi$）后，任何具体配置都是零测集——精确表述是「落在支撑集边界上」而非内点；两种口径下结论相同：不是外推。SkyReels-V2 把这条路做成无限长电影生成系统：MLLM captioner（SkyCaptioner-V1）+ 多阶段预训练 + RL + Diffusion Forcing 框架（全文 23 处提及），[Chen et al., 2025](../references.md)，arXiv:2504.13074，[A]（官方 repo [SkyworkAI/SkyReels-V2](../references.md)，[A]）——长视频「分块 AR」路线的当代代表（§8 路线二）。

## 6. Self-Forcing：自 rollout 训练与 exposure bias 的消除（核心推导）

### 6.1 exposure bias 的形式化：KL 链式分解定位失配

把 chunk 级联合分布写成条件乘积： $p(x^{1:M}) = \prod_{k=1}^{M} p(x^{(k)} \mid x^{<k})$，模型侧 $q_\theta$ 同形。KL 的链式分解（展开对数比、对 $p$ 取全期望、逐项用 tower property 把 $x^{(\ge k)}$ 边缘化）：

$$\mathrm{KL}\left( p\left( x^{1:M} \right)\thinspace \Vert\thinspace q_\theta\left( x^{1:M} \right) \right) = \sum_{k=1}^{M} \mathbb{E}_{x^{<k} \sim p} \left[ \mathrm{KL}\left( p\left( \cdot \mid x^{<k} \right)\thinspace \Vert\thinspace q_\theta\left( \cdot \mid x^{<k} \right) \right) \right] \qquad (10)$$

**符号**： $x^{(k)}$：第 $k$ 个 chunk； $x^{<k}$：其前全部 chunk（历史）；右端第 $k$ 项：在**数据分布的历史边际**下评估的第 $k$ 个条件失配。

式 (10) 把训练-推理失配钉到一个精确位置。teacher forcing 式训练最小化的是右端——历史测度取 $p$（ground-truth 上下文）；推理 rollout 评估的是同一批条件、但历史测度换成了模型自己的 $q_\theta$：

$$L_{deploy}(\theta) = \sum_{k=1}^{M} \mathbb{E}_{x^{<k} \sim q_\theta} \left[ \ell_k\left( x^{<k} \right) \right] \qquad (11)$$

**符号**： $\ell_k$：第 $k$ 个条件在自生成历史下的失配泛函（与式 (10) 的逐条件 KL 同形，或其实用替代）。复合机制：chunk $j$ 的任何残差失配使 $q_\theta$ 的历史边际偏离 $p$，该偏移进入全部 $k > j$ 的条件的输入分布——历史测度的错位随 $k$ 累积，而训练目标对这些区域的覆盖为零。Self-Forcing 摘要的指控对象正是这个错位——训练以 ground-truth 为上下文、推理须以自生成输出为条件（摘要原文：「models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference」，[Huang et al., 2025](../references.md)，arXiv:2506.08009，[A]）。注意 §3 的 DMD 阶段不改这个诊断：分布匹配的目标分布确实是学生输出分布，但因果学生的**条件上下文**在训练时仍是 ground truth——失配不在被监督的输出侧，在未被监督的输入侧。长视频上 $M$ 大、复合长，这就是因果学生长跑漂移而双向教师不漂移的原因链中属于训练侧的那一环（架构侧的因果约束分析见 §3.2）。

### 6.2 on-policy 修复的结构

Self-Forcing 的修复一句话版本：把式 (11) 的测度直接拿来当训练测度——训练时就用模型自己做自回归 rollout（KV caching 推进，与推理同一算法），逐帧以自生成输出为条件，监督改为施加在整段自生成视频上的分布匹配：

$$L_{SF}(\theta) = \mathbb{E}_{x^{1:M} \sim q_\theta^{rollout}} \left[ \ell_{DM}\left( x^{1:M} \right) \right] \qquad (12)$$

**符号**： $q_\theta^{rollout}$：模型自 rollout 诱导的整视频分布——与式 (11) 的部署测度恒等，训练期望的测度与部署测度按构造一致，exposure bias 的失配位置被消除； $\ell_{DM}$：视频级分布匹配项——§3 式 (5)(6) 的同一台 DMD 机器（fake/real score 差经 Jacobian 回传），施加在完整自 rollout 产出的整段视频上；反传按论文口径做梯度截断（gradient truncation）：只穿过每帧最后一个去噪步，且历史 KV cache embedding 的梯度被 detach——长 rollout 的训练内存由此可付，与本节的 cache 成本论证是同一处设计的两半。摘要口径三件套：训练时自回归 rollout + KV caching、逐帧自生成条件、视频级整体分布匹配监督（[Huang et al., 2025](../references.md)，[A]；官方 repo [guandeh17/Self-Forcing](../references.md)，[A]，README 明示建于 CausVid 代码 + Wan2.1-T2V-1.3B 基座）。

一处设计耦合值得点名：KV cache 在 Self-Forcing 里既是推理加速件也是训练机制——自 rollout 若不用 cache，每推进一帧都要重算全历史的前向，训练成本不可行；cache 使「训练时跑推理算法本身」变得可付。2.4 的推理数学反向进入训练循环，这是 §4.3 合流点的另一面。

### 6.3 谱系定位

把 §3-§6 的修正按「修的是什么」排开：CausVid 修架构与步数（双向 → 因果、50 → 4 步，监督信号仍来自见全局的教师；训练输入同时已在 chunk 粒度 DF 化——支撑集轴它跨了一半，§3.2）；Self-Forcing 修训练测度（off-policy → on-policy，与 CausVid 正交、可叠加，代码上直接建于其 repo）；Diffusion Forcing 给出支撑集轴的完整框架（共享 $t$ 截面 → 乘积空间 $\pi^{\otimes n}$ 的逐 token 推广 + 变分下界口径）。三者攻击的是同一条链上的不同环节：架构（看得见过往多少）× 步数（每环付多少）× 测度（在谁的历史上训练）——支撑集轴贯穿其间（DF 给框架，CausVid 取了它在 chunk 粒度的一个实例）。

## 7. 2026 后续现状表：Causal-Forcing 系与 rCM 系

| 系统（均 [A]） | arXiv（时间） | 初始化 | 主训练信号 | 步数 | 关键声明（出处口径） |
| --- | --- | --- | --- | --- | --- |
| CausVid | 2412.07772（2024-12） | 教师 ODE 配对回归（式 (4)） | 非对称 DMD（式 (5)(6)），无 GAN | 4 | 9.4 FPS、首帧延迟 1.3 s、160×/16× vs CogVideoX-5B（vs 双向教师 ≈169×/16×，Table 3 640×352）、VBench-Long 84.27（论文） |
| Self-Forcing | 2506.08009（2025-06） | 建于 CausVid（代码 + Wan2.1-1.3B 基座） | 自 rollout + 视频级分布匹配（式 (12)） | 4 步谱系（建于 CausVid 4 步代码，README 口径） | 消除 exposure bias（摘要口径） |
| Causal-Forcing | 2602.02214（2026-02） | **Causal ODE 驱动 asymmetric DMD 初始化** | 非对称 DMD | chunk-wise / frame-wise | 官方 README 口径：同训练预算、同推理效率下视觉质量与运动动态显著超 Self-Forcing |
| Causal-Forcing++ | 2605.15141（2026-05） | **causal consistency distillation 取代 ODE 配对数据** | 同上谱系 | 首个 frame-wise 1/2 步模型 | 官方 README/论文口径 |
| rCM | 2510.08431（2025-10，ICLR 2026） | —（连续时间一致性谱系） | **sCM + score distillation 正则**（forward KL mode-covering + reverse KL mode-seeking 互补） | 1–4 | >10B 规模、Cosmos-Predict2/Wan2.1（≤14B）验证、15–50× 加速、**无 GAN tuning**；FA2 JVP kernel |
| Causal-rCM | 2606.25473（2026-06） | — | **teacher-forcing（forward/offline CM）与 self-forcing（reverse/on-policy DMD）统一开放配方**（repo README 原文） | few-step | NVlabs/rcm 官方实现 |

读表三条收敛线：(i) **GAN 全体退场**——CausVid 起方法部分无 GAN，rCM 明示无 GAN tuning；7.4 里 DMD2 的 GAN 分支在视频侧没有被继承（§3.1 的澄清在此收账）。(ii) **初始化路线的演进**——ODE 配对回归（CausVid）→ Causal ODE（Causal-Forcing）→ causal consistency distillation（CF++，把 ODE 配对数据本身也取消）。(iii) **训练测度的统一**——Causal-rCM 把 teacher-forcing（forward/offline CM 一侧）与 self-forcing（reverse/on-policy DMD 一侧）写成同一开放配方的两个模式（repo README 口径），§6.1 的测度之争在配方层被工程化收编。rCM 线另记一笔机制：score-regularized continuous-time consistency = sCM（一致性模型的连续时间化，谱系接 7.4 §5.2，[Song et al., 2023](../references.md)，arXiv:2303.01469，[A]）+ score 蒸馏正则，forward KL 与 reverse KL 的互补分工与式 (5) 的方向语义互为印证（[Zheng et al., 2025](../references.md)，arXiv:2510.08431，[A]；[Zheng et al., 2026](../references.md)，arXiv:2606.25473，[A]；官方实现 [NVlabs/rcm](../references.md)，[A]）。

## 8. 长视频三路线：keyframe+插值 / 分块 AR / context 扩展

**路线一：keyframe + 插值。** 稀疏关键帧先行，区间后填。公式形态是两级：keyframe 链 $p(K_1) \prod_i p(K_{i+1} \mid K_i, c_i)$（关键帧粒度的高层自回归）+ 区间内填充 $p(x^{(k_i + 1..k_{i+1} - 1)} \mid K_i, K_{i+1}, c)$（首尾帧条件生成——机制归下一章：9.4 的 I2V/kf2v 家族）。账：区间之间可并行（误差不在时间轴上复合），墙钟友好；上限是区间内的运动幅度与内容连续性受关键帧密度约束，长程一致性等于关键帧链的质量。

**路线二：分块 AR。** 即 §4-§6 的全部内容：历史条件 $q_\theta(x^{(k)} \mid x^{<k})$ + KV cache 流式；SkyReels-V2（Diffusion Forcing 框架，§5.3）为当代代表。已知敌人是 exposure bias（§6 修复）；推理上下文 = cache 长度，滑窗截断换有界成本（§4.2 账）。时长在原则上无界（流一直推），实际约束是长程一致性随 rollout 复合的残差。

**路线三：context 扩展。** 把双向窗口直接拉长，账用 9.1/9.2 的口径现推：1 分钟 720p@24fps → $T = 1440$、latent 帧 $361$、 $N = 361 \times 90 \times 160 \approx 5.20 \times 10^{6}$ token， $N^{2} \approx 2.70 \times 10^{13}$；单层二次项 $4 N^{2} d$（ $d = 5120$）≈ 553 PFLOPs，×40 层 ≈ 22.1 EFLOPs **一个去噪步**——已 5.4 倍于 81 帧整段视频全部 50 步的总和（9.2 §2.5 的 4.1 EFLOPs）；×50 步 ≈ 1.1 ZFLOPs 一段一分钟视频，约为 5 秒段的 270 倍。披露侧的规模点：Movie Gen 最大上下文 73K video token（= 16s@16fps，[Polyak et al., 2024](../references.md)，arXiv:2410.13720，[B]——未开源，只引论文口径）。裸 full attention 的 context 扩展在分钟级不可行；要么稀疏化（9.2 §2.4 的 SUV 一系），要么退回路线一/二。三条路线不互斥——按「时长 × 一致性 × 预算」的权衡面取点，当代系统多为组合（SkyReels-V2 = 路线二主干 + 多阶段配方）。

## 9. 展望：交互式世界模型（一段式）

实时流式 + 逐 chunk 推进的另一个读法：动作条件下的下一 chunk 预测——视频模型由此进入交互式世界模型（interactive world model）的形态，Causal-Forcing 的标题（Real-Time Interactive Video Generation）与 Causal-rCM 的副标题定位（interactive world models）都把方向写在了标题里（[Zhu et al., 2026](../references.md)，[A]；[Zheng et al., 2026](../references.md)，[A]）。成熟度标注：配方层已有 [A] 级开放实现（thu-ml/Causal-Forcing、NVlabs/rcm 两个 repo）；质量-时延的帕累托前沿仍在 2026 年活跃移动（§7 表是快照）；「交互动作条件化 + 长时一致性」的系统级验证尚处研究阶段，本书不作能力外推。

## 总结

| 主题 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| 图视频混训 | 式 (1)；三条兼容条件（9.1 首帧 / 9.2 RoPE 退化 / attention 退化） | 图像钉空间先验、视频供时间监督；渐进课程由 63.2× 桶成本差驱动（9.2 §9） |
| 视频 CFG | 式 (2)(3)；NFE ×2（Wan 14B 几何 50 步 ≈ 8.2 EFLOPs） | tilt 语义不变； $w(t)$ 是公式级自由度；蒸馏把引导折进学生 |
| CausVid | 式 (4)(5)(6)；50 → 4 步、9.4 FPS、1.3 s 首帧延迟、160×/16× | 非对称 = 双向教师 score → 因果学生；训练 = 逐 chunk 一步预测（4 步复合只在推理）；无 GAN（GAN 属 DMD2）；ODE 配对预训恢复 |
| 流式合流 | 式 (7)；cache 2.4 式 (9) 平移；81 帧 720p 全历史 K/V ≈ 248 GB（14B config） | 块因果 ⇒ 前缀不可变 ⇒ KV cache；双层循环 = decode 步内嵌 K 步退火 |
| Diffusion Forcing | 式 (8)(9)；两角点（共享 $t$ / 逐帧 AR） | per-token $t_i$；统一 teacher forcing 与全序列扩散；所有子序列的变分下界（摘要口径） |
| Self-Forcing | 式 (10)(11)(12) | 失配在输入侧测度；训练测度 = 部署测度（on-policy rollout + 视频级分布匹配） |
| 2026 现状 | §7 表 | GAN 退场；初始化 ODE→Causal ODE→causal consistency；teacher/self-forcing 统一（Causal-rCM） |
| 长视频 | §8；分钟级 full attention ≈ 1.1 ZFLOPs/段 | keyframe+插值 / 分块 AR / context 扩展按权衡面取点 |

## 练习

1.（推导）Diffusion Forcing 的角点与缺口。从式 (8)(9) 出发：(a) 约束 $t_1 = \cdots = t_n = t$（每序列采一个标量），把式 (9) 化简为式 (1) 的 ε 坐标形态，写出每一步；(b) 取 $t_{<i} = 0$，写出第 $i$ 个 token 的条件学习目标的显式形式，并描述「逐 token 内层退火 + 历史钉 0」的采样算法；(c) 论证：只在共享 $t$ 截面训练的双向基座，面对 §4 双层循环的输入（cache 内 clean 历史 + 当前 chunk 在噪）时缺了哪部分支撑集——注意连续 $\pi$ 下任何具体配置都是零测集，请用支撑集（而非概率质量）的语言陈述缺口；并说明 DF 论文的离散噪声水平口径（含 0）如何使该类配置获得正概率质量。

2.（推导）CausVid 的非对称 DMD 梯度。从式 (5) 出发重推式 (6)：写出 reverse KL 的熵-交叉熵分解、测度项 $`\mathbb{E}_{p_\psi}[\partial_\psi \log p_\psi] = 0`$ 的依据、以及 score 差的合并步（对照 7.4 式 (30)-(32)）。然后回答两个架构问题：(a) 冻结教师的双向性出现在梯度的哪个因子里，学生的因果性出现在哪里；(b) 论证不动点 $p_\psi = p_{data}$ 与学生上下文窗口无关，并解释「非对称」因此改变的是逼近路径而非目标。

3.（计算）流式账。取 81 帧 720p（21 latent 帧、 $N$ = 302,400、 $d$ = 5120、40 层、40 头、 $d_{head}$ = 128，Wan 14B config 实测口径），chunk = 3 latent 帧（ $n$ = 43,200、 $M$ = 7）、 $K$ = 4：(a) 用式 (7) 算每内层步的 attention 二次项总量与全视频（ $K$ 步合计）总量，对照双向教师 50 步的 $K_T \cdot 4 d N^{2}$，验证比值 ≈ 1/21.9（ $M$ = 7 精确值； $M \to \infty$ 极限 1/25）；(b) 算 chunk $k$ = 7 时 cache 的字节数（2.4 式 (9) 口径）与全历史上限，并对照 §4.2 的 248 GB 参照；(c) 用 §4.2 的延迟分解估计 time-to-first-frame 比值的量级，说明实测 160× 与上界估计之间的差值来源。

4.（分析）长视频路线选择。目标：10 分钟 720p@24fps。(a) 按 §8 路线三的口径算 context 扩展的 $N$、 $N^{2}$ 与单层二次项 FLOPs；(b) 同负载按路线二（chunk = 3 latent 帧 + 滑窗 $W$ = 10 chunk）估算每 chunk 成本与 cache 上限；(c) 给出两路线在「总 FLOPs / 首帧延迟 / 长程一致性风险」三维下的取舍陈述，各用一条本章公式支撑。

## 参考

- [Chen et al., 2024](../references.md)：Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion，arXiv:2407.01392，[A]（key: diffusion-forcing）——per-token 独立噪声水平、统一 teacher forcing 与全序列扩散、所有子序列的变分下界（§5）
- [Boyuan Chen 等，diffusion-forcing 官方 repo](../references.md)：[A]（key: diffusion-forcing，条目含 buoyancy99/diffusion-forcing 的 URL）——§5 实现锚点
- [Yin et al., 2024](../references.md)：From Slow Bidirectional to Fast Autoregressive Video Diffusion Models（CausVid），arXiv:2412.07772，[A]（key: causvid）——非对称蒸馏、ODE 初始化、无 GAN、KV cache 流式与全部端到端数字（§3、§4）
- [Yin et al., CausVid 官方 repo](../references.md)：[A]（key: causvid-repo）——configs/wan_causal_dmd.yaml 四步配置锚点（§3）
- [Huang et al., 2025](../references.md)：Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion，arXiv:2506.08009，[A]（key: self-forcing）——exposure bias 动机、自 rollout + KV cache + 视频级分布匹配（§6）
- [He et al., Self-Forcing 官方 repo](../references.md)：[A]（key: self-forcing-repo）——建于 CausVid 代码 + Wan2.1-1.3B 基座（§6）
- [Chen et al., 2025](../references.md)：SkyReels-V2: Infinite-length Film Generative Model，arXiv:2504.13074，[A]（key: skyreels-v2；勘误：2504.02436 是 SkyReels-A2，勿用）——Diffusion Forcing 框架的无限长电影系统（§5.3、§8）
- [SkyworkAI, SkyReels-V2 官方 repo](../references.md)：[A]（key: skyreels-v2-repo）——§8
- [Zhu et al., 2026](../references.md)：Causal Forcing，arXiv:2602.02214，[A]（key: causal-forcing）——Causal ODE 驱动初始化、超 Self-Forcing 的 README 口径（§7）
- [Zhao et al., 2026](../references.md)：Causal Forcing++，arXiv:2605.15141，[A]（key: causal-forcing-pp）——causal consistency distillation、首个 frame-wise 1/2 步（§7）
- [Tsinghua, thu-ml/Causal-Forcing 官方 repo](../references.md)：[A]（key: causal-forcing-repo）——§7
- [Zheng et al., 2025](../references.md)：rCM: Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency，arXiv:2510.08431，[A]（key: rcm）——sCM+score 正则、15–50×、无 GAN tuning（§7）
- [Zheng et al., 2026](../references.md)：Causal-rCM，arXiv:2606.25473，[A]（key: causal-rcm）——teacher-forcing 与 self-forcing 统一开放配方（§7、§9）
- [NVIDIA, NVlabs/rcm 官方 repo](../references.md)：[A]（key: rcm-repo）——§7、§9
- [Yin et al., 2023](../references.md)：DMD，arXiv:2311.18828，[A]（key: dmd）——分布匹配梯度与 ODE 配对回归的图像侧出处（recap 7.4 §5.5，§3.1）
- [Yin et al., 2024](../references.md)：DMD2，arXiv:2405.14867，[A]（key: dmd2）——免教师数据 + GAN 分支（归属澄清见 §3.1）
- [Ho & Salimans, 2022](../references.md)：Classifier-Free Diffusion Guidance，arXiv:2207.12598，[A]（key: cfg）——CFG 出处（recap 7.4，§2）
- [Song et al., 2023](../references.md)：Consistency Models，arXiv:2303.01469，[A]（key: consistency-models）——sCM 谱系源头（recap 7.4 §5.2，§7）
- [Lipman et al., 2022](../references.md)：Flow Matching，arXiv:2210.02747，[A]（key: flow-matching）；[Liu et al., 2022](../references.md)：Rectified Flow，arXiv:2209.03003，[A]（key: rectified-flow）——FM/线性路径与 Euler 采样（recap 7.5，§1、§3）
- [Team Wan, 2025](../references.md)：Wan，arXiv:2503.20314，[A]（key: wan）——CausVid/Self-Forcing 的基座族（Wan2.1-T2V-1.3B）与 14B config 账的出处（§3、§4）
- [Polyak et al., 2024](../references.md)：Movie Gen，arXiv:2410.13720，[B]（key: moviegen）——73K video token 上下文规模点（§8）
