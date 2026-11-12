# W4 调研笔记：P7 Diffusion & Image Generation 事实源

调研日期：2026-11-12（W4 Task 1，research pass）。本文件是 T2–T8（7.1–7.7 共 7 章）写作的唯一事实源；写作 agent 不做额外调研，公式与数值一律以本文件与 `models/references.md` 为准。本 Part 是全书数学密度最高的 Part，所有推导以「W4 扩散全书记号约定」节为唯一口径。

## 访问路径增量记录（沿用 W1–W3 runbook，本轮新增标注）

- arXiv API：**一律 `https://export.arxiv.org/api/query`**——`http://` 裸协议在本机会 hang 至超时（本轮实测，首请求 120s 无响应）；https 正常。间歇性 0 字节响应同 W3，重试即恢复。批量 10 条 + `max_results` 参数。
- GitHub：repo 存在性 api.github.com/repos（本轮 17 查：16×200 + google-research/muse 404）；目录列取 api.github.com/repos/<org>/<repo>/contents/<path>（raw.githubusercontent.com 直连仍不可用，W2 已记）。
- HF：huggingface.co 直连不可用，一律 hf-mirror。stabilityai/stable-diffusion-xl-base-1.0 model_index.json 实测 200（非 gated）。
- Vincent 2011 出处勘误：spec 7.3 称「Vincent 2011（JMLR，无 arXiv）」，**实测期刊为 Neural Computation 23(7):1661–1674**（DOI 10.1162/NECO_a_00142，doi.org 302 → direct.mit.edu/neco/article/23/7/1661-1674/7677 验证存在；MIT Press 页有 Cloudflare challenge，curl 无法取正文，DOI 解析即存在性证据）。
- 论文摘要取证口径：arXiv API `<summary>` 字段（本轮用于 Infinity bitwise、LlamaGen 两案），正文细节由写作者读论文。

## W4 扩散全书记号约定（写作 agent 必须遵守）

本书扩散章节统一采用 DDPM/SD3 式记号。**核心：x₀ = 数据端，t = 1 = 纯噪声端；采样方向 t: 1 → 0。**

### 基本记号

- 数据 $x_0 \sim p_{data}$（图像或 latent——P6 的 $x$/$z$ 约定在扩散语境统一改称 $x_0$，指「扩散模型的建模对象」：像素扩散时 $x_0$ = 像素，latent diffusion 时 $x_0$ = VAE latent $z$）。
- 时间 $t\in[0,1]$（连续）或 $t\in\{1,\dots,T\}$（DDPM 离散记号，$T=1000$）；**$t=1$（或 $T$）= 纯噪声端，$t=0$ = 数据端**。
- 前向过程（加噪）方向 t: 0→1；反向过程（采样/生成）方向 t: 1→0。
- 噪声 $\varepsilon \sim \mathcal{N}(0, I)$；schedule 记号 $\alpha_t = 1-\beta_t$、$\bar\alpha_t = \prod_{s\le t}\alpha_s$（离散）或其连续对应。

### 前向闭式与采样式（7.2 推导终点）

$$q(x_t|x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,I\right),\qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

### 三种预测参数化与换算（7.2 / 7.5 共用）

- **ε-prediction**（DDPM 原始口径，本书默认）：网络 $\varepsilon_\theta(x_t, t)$ 回归噪声 $\varepsilon$；训练目标 $L_{simple} = \mathbb{E}\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$。
- **x₀-prediction**：回归数据 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$。
- **v-prediction**（iDDPM/SD 系常用）：定义 $v \equiv \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0$。换算（给定 $v_\theta$ 与 $x_t$）：
$$\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v_\theta,\qquad \hat\varepsilon = \sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,v_\theta$$
- **score ↔ ε**（7.3 桥）：$s_\theta(x_t, t) = \nabla_{x_t}\log p_t(x_t) = -\varepsilon_\theta(x_t, t)/\sqrt{1-\bar\alpha_t}$。

### Rectified flow / 线性路径约定（7.5 主口径）

- 线性插值路径：$x_t = (1-t)\,x_0 + t\,\varepsilon$（$t\in[0,1]$，t=0 数据、t=1 噪声）。
- 目标速度（恒定，沿直线）：$u_t = \varepsilon - x_0$；网络 $v_\theta(x_t, t)$ 回归该速度。
- 采样：从 $x_1 = \varepsilon$ 出发沿 ODE $dx/dt = v_\theta(x, t)$ 积分至 $t=0$（**t: 1→0**）。
- 与 DDPM 记号的同族映射：令 $\bar\alpha_t = (1-t)^2/\big((1-t)^2 + t^2\big)$ 即得一一对应（SNR 匹配），7.5 给换算表。

### Lipman（FM 原文）约定映射框（防混淆，每处引用 FM 论文公式时适用）

Lipman et al. 原论文约定 **x₁ = 数据、x₀ = 噪声**，路径 $p_t$ 从 t=0（噪声）演化到 t=1（数据）。映射到本书记号：

$$x_0^{\text{Lipman}} = \varepsilon\ (\text{本书噪声}),\qquad x_1^{\text{Lipman}} = x_0^{\text{本书}},\qquad t^{\text{Lipman}} = 1 - t^{\text{本书}}$$

7.5 正文首次引用 FM 目标时出一次映射框，此后全部用本书记号重述，不混写。

### Guidance 与采样约定（7.4）

- CFG：$\tilde\varepsilon = \varepsilon_\theta(x_t, \varnothing) + w\,\big(\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing)\big)$，$w=1$ 退化为纯条件；工业典型 $w\in[2, 15]$。
- logit-normal t 采样（SD3 口径）：$t = \mathrm{sigmoid}(u),\ u\sim\mathcal{N}(m, s^2)$。
- 「NFE」= network function evaluations（采样步数账的统一单位）。

### 与 P6 约定的衔接

latent 记号 $z$、压缩率 f8/f32、scale factor 0.18215（LDM yaml 出处）、SD3/FLUX 16ch VAE 等均沿用 W3 约定页；本章（P7）不再重复定义，引用时交叉链接 6.3。

## 7.1 生成模型家族统一对照

- **GAN**（`gan`，arXiv:1406.2661，Goodfellow，本轮核实标题 *Generative Adversarial Networks*——arXiv 版标题为复数 "Networks"，NIPS 2014 版为 "Nets"，引用以 arXiv 标题为准）：minimax 目标出处。
- **Glow**（`glow`，arXiv:1807.03039，Kingma & Dhariwal，本轮核实）：invertible 1×1 conv flow、精确似然 change-of-variables 出处——7.1 Flow 家族代表。
- **DDPM**（`ddpm`，arXiv:2006.11239，Ho，本轮核实）：扩散家族代表，详拆在 7.2。
- **VAE**（`vae`，W3 已收录，被引章节本轮 +7.1）：ELBO recap → 6.3。
- 写作注意：本章是对照章（五家族统一表），所有公式给「结论级」即可，推导指向 7.2–7.5 与 6.3；GAN 按 spec 范围决策不展开目标函数以外的内容。

## 7.2 DDPM 完整推导

- **DDPM**（`ddpm`，2006.11239）：前向过程、$L_{simple}$、T=1000、linear schedule 出处（ε-prediction 与 β̃ 方差选择均在本文）。
- **iDDPM**（`iddpm`，arXiv:2102.09672，Nichol & Dhariwal，本轮核实）：**cosine noise schedule 出处**、learned σ（hybrid objective）、strided sampling——7.2 schedule 对比节与 σ_t 选择的直接来源。
- 锚点：`ddpm-repo`（hojonathanho/diffusion，200；官方 TF 实现在 `diffusion_tf/` 目录，根目录实测）；`ddpm-pytorch`（lucidrains/denoising-diffusion-pytorch，200，**教学参照**——代码对照用，勿当官方实现）。
- 约定遵循：全书 x₀=数据、t=1=噪声（见约定页）；DDPM 原文离散下标 t∈{1..T}，写作时用「离散↔连续」一句桥接。

## 7.3 Score-Based 视角与 SDE/ODE

- **score-SDE**（`score-sde`，arXiv:2011.13456，Yang Song，本轮核实）：VP/VE-SDE、reverse-time SDE（Anderson）、probability-flow ODE 出处。
- **EDM**（`edm`，arXiv:2206.00364，Karras，本轮核实）：设计空间梳理（preconditioning、σ-参数化、Heun 2 阶采样器）——7.3 末尾「工程化收口」节来源。
- **Vincent 2011**（`vincent-2011`，**勘误：Neural Computation 非 JMLR**，DOI 10.1162/NECO_a_00142，doi.org 解析验证）：denoising score matching 等价性定理原始出处（$\nabla\log q(x_t|x_0)$ 条件 score 回归 ≡ 真 score 回归的关键引理）。
- 锚点：`score-sde-repo`（yang-song/score_sde_pytorch，200）。
- 写作注意：ε↔score 换算 $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ 按约定页统一记号；Fokker-Planck 只给结论级（spec 口径）。

## 7.4 采样器与 Guidance

- **DDIM**（`ddim`，arXiv:2010.02502，Jiaming Song，本轮核实）：非马尔可夫前向族、σ=0 确定性采样（= PF-ODE 一阶离散化）出处。锚点 `ddim-repo`（ermongroup/ddim，200）。
- **classifier guidance**（`classifier-guidance`，arXiv:2105.05233，Dhariwal & Nichol，本轮核实——标题即 *Diffusion Models Beat GANs on Image Synthesis*，classifier guidance 是该文技术贡献，写作时标题与贡献勿张冠李戴）：$\nabla\log p(y|x_t)$ 梯度引导出处；ADM/ADM-G 模型细节属本文。
- **CFG**（`cfg`，arXiv:2207.12598，Ho & Salimans，本轮核实）：classifier-free 推导与 w 的 tilt 解释出处。
- **consistency models**（`consistency-models`，arXiv:2303.01469，Yang Song，本轮核实）：同 ODE 轨迹自一致性 $f(x_t,t)=f(x_{t'},t')$、CT/CD 两种训练出处。锚点 `consistency-models-repo`（openai/consistency_models，200；`cm/karras_diffusion.py` 与 `cm/losses.py` 目录实测——CT/CD loss 走读锚点）。
- **LCM**（`lcm`，arXiv:2310.04378，Simian Luo，本轮核实——spec ⟨verify⟩ 结案：标题 *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*）：latent 空间 consistency 蒸馏出处。
- **DMD**（`dmd`，arXiv:2311.18828，Tianwei Yin，本轮核实——spec ⟨verify⟩ 结案：标题 *One-step Diffusion with Distribution Matching Distillation*）：分布匹配梯度 $\mathbb{E}[(s_{fake}-s_{real})\,\partial x/\partial\theta]$ 出处（real score = 冻结 teacher、fake score 在线训练）。
- **DMD2**（`dmd2`，arXiv:2405.14867，Tianwei Yin，本轮核实——spec ⟨verify⟩ 结案：标题 *Improved Distribution Matching Distillation for Fast Image Synthesis*）：+ GAN loss + 免教师真实数据回归的改进版；**GAN 分量属 DMD2 不属 DMD**（spec 9.2 同款口径，勿混淆）。锚点 `dmd2-repo`（tianweiy/DMD2，200）。
- **rectified flow**（`rectified-flow`，arXiv:2209.03003，Xingchao Liu，本轮核实）：reflow 拉直 + 1-rectified flow 出处；详拆在 7.5，本章只在蒸馏语境引「reflow→few-step」。
- **DPM-Solver**（`dpm-solver`，arXiv:2206.00927，Cheng Lu，本轮核实）：半线性结构 + 指数积分 + Taylor 展开的高阶 ODE solver 出处（~10 NFE）。
- 写作注意：CFG 完整推导是本章关键推导之一（evaluator 强制复推清单含 7.4 CFG）；few-step 现状表的数字（4-step/1-step FID）一律引 DMD/DMD2/LCM 论文原表，标日期。
