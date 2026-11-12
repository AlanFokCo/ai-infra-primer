# models 维度 W4 实施计划（P7 Diffusion & Image Generation，7 章）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W4：P7 扩散与图像生成（7 章），全书数学密度最高的 Part——每个公式完整推导。

**Architecture:** 沿用已验证流水线（research pass → 分批并行写作（opus）+ 批量审查（opus）→ 波末 evaluator（fable，强制重推导）→ 修复复审 → 清扫收尾）。

**Tech Stack:** 同前。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P7 条目）

## Global Constraints

（沿用 + 累积裁决）：
- 字数 = 正文 CJK 汉字数（不含标点/公式/代码）：标准章 ≥3500、推导重章 ≥5500（软）、短篇 ≥2500
- **全书约定：x₀=数据、t=1=噪声端**（与 7.2 DDPM 及锚点 SD3 一致；附 Lipman 原论文「x₁=数据」的约定映射框防混淆）
- 白名单现行锁定版（含 \lceil \circ \perp \square \leftarrow \ne \cup 等收编项）
- 写作者不做网络调研；references.md 回写由写作者在各自 commit 内做（P7 内顺序派发则自控，并行则收归 controller）
- 并行 git 纪律：禁用 --amend / reset；index.lock 等 5 秒重试
- 防崩纪律（子 agent）：curl 抓网页后不把大段原文粘贴进输出/思考

## Review Focus

1. **扩散数学的约定一致性**（x₀/x₁、α/ᾱ、t 方向、ε/v-prediction）跨 7 章 → research pass 写死约定页，写作者遵守，evaluator 抽查
2. **推导正确性**（本书数学最密部分）→ 波末强制重推导：7.2 DDPM 全链、7.4 CFG、7.5 conditional FM 定理
3. **系统拆解的真实性**（SD3/SDXL/PixArt/SANA/VAR/LlamaGen/Emu3 的论文级细节）→ research pass + evaluator 抽查
4. **evaluator rubber-stamp** → verification log 硬要求

---

## W4 任务

### Task 1: W4 research pass（事实源锁定）

**Files:** Modify `models/references.md`；Create `docs/research/w4-sources.md`

- [ ] **Step 1: 核实与收录**——DDPM 2006.11239、iDDPM 2102.09672、score-SDE 2011.13456、EDM 2206.00364、DDIM 2010.02502、classifier guidance 2105.05233、CFG 2207.12598、consistency 2303.01469、LCM 2310.04378、DMD 2311.18828、DMD2 2405.14867、rectified flow 2209.03003、DPM-Solver 2206.00927、FM 2210.02747、SD3 2403.03206、PixArt-α 2310.00426、PixArt-Σ 2403.04692、SANA 2410.10629、LDM 2112.10752、DiT 2212.09748、VAR 2404.02905、LlamaGen 2406.06525、Infinity 2412.04431、MaskGIT 2202.04200、MUSE 2301.00704、Emu3 2409.18869、Imagen 2205.11487、SDXL 2307.01952、Glow 1807.03039、GAN 1406.2661、DC-AE 2410.10733（已有）；repo 存在性：hojonathanho/diffusion、yang-song/score_sde_pytorch、ermongroup/ddim、openai/consistency_models、tianweiy/DMD2、facebookresearch/flow_matching、facebookresearch/DiT、CompVis/latent-diffusion、Stability-AI/generative-models、sd3.5、black-forest-labs/flux、NVlabs/Sana、PixArt-alpha/PixArt-alpha、FoundationVision/VAR+LlamaGen+Infinity、baaivision/Emu3、google-research/muse（不存在——[B] 处理先例）
- [ ] **Step 2: 约定页**——扩散全书约定写死：x₀=数据、t=1=噪声、q(x_t|x₀)=N(√ᾱ_t x_0, (1−ᾱ_t)I)、ε-prediction、v-prediction 换算、rectified flow x_t=(1−t)x₀+tε、目标速度 ε−x₀、采样方向 t:1→0、Lipman x₁=数据 映射框
- [ ] **Step 3: 写 w4-sources.md**（每章一节）+ references.md 入库
- [ ] **Step 4: 自检 + Commit**（`research: lock W4 sources (P7)`）

### Task 2-8: P7 七章（分批并行写作）

**Files:** Create `models/07-diffusion-image-generation/{01-generative-model-map,02-ddpm,03-score-sde,04-sampling-and-guidance,05-flow-matching,06-latent-diffusion-and-dit,07-t2i-systems-and-ar-image}.md`

分批（依赖序）：batch 1 = 7.1+7.2（7.1 概念地图独立、7.2 是 DDPM 基石）→ batch 2 = 7.3+7.5（SDE 视角、FM——依赖 7.2）→ batch 3 = 7.4+7.6（采样器/guidance 依赖 7.2/7.3、LDM/DiT 依赖 7.5）→ batch 4 = 7.7（系统拆解收口，依赖全部）

**内容要求（spec §5-P7 各节全量）**：
- 7.1 generative-model-map：五家族统一框架（AR/VAE/GAN/Flow/Diffusion-FM 各自优化什么、怎么采样、似然可解性、mode 行为）+ 对照总表 + 本 Part 定位图；GAN 不展开（对抗 loss 在用处就地讲）
- 7.2 ddpm（推导重章）：前向过程 → **闭式解推导**（高斯卷积递推逐步展开）→ 采样式 → noise schedule（linear/cosine）→ 反向过程 → **后验 q(x_{t−1}|x_t,x_0) 完整推导**（Bayes+配方）→ **ELBO 分解三项** → ε-参数化 → **L_simple**（与严格 ELBO 的 SNR weighting 差异推导）→ x₀/v-prediction 等价换算 → σ_t 选择
- 7.3 score-sde（推导重章）：score 定义 → score matching 不可解 → **denoising score matching 等价性（Vincent 定理）推导** → 与 DDPM 换算 ε_θ=−√(1−ᾱ_t)·s_θ → **SDE 表述**（VP/VE）→ Fokker-Planck（结论级）→ **probability-flow ODE 推导** → 精确似然（连续 change of variables）→ reverse-time SDE（Anderson）→ 统一视图表
- 7.4 sampling-and-guidance（推导重章）：ancestral sampling 伪码 → **DDIM 推导**（非马尔可夫前向族、σ=0 确定性、= PF-ODE 一阶离散化）→ **classifier guidance**（Bayes 展开）→ **CFG 完整推导**（隐式 classifier p(x|y)∝p(x)p(y|x)^w、w 的数学含义、过饱和失效模式与缓解——dynamic thresholding/rescaled CFG）→ 求解器（DPM-Solver 指数积分 outline）→ **蒸馏族**（progressive distillation、consistency models 自一致性性质公式、LCM、**DMD 分布匹配梯度**、reflow）→ **CFG/guidance 蒸馏**（把 w 蒸进单模型的目标公式——工业默认做法）→ few-step 现状表
- 7.5 flow-matching（推导重章）：CNF 定义与 pushforward → trace 公式（贵的原因）→ FM 目标 → **conditional FM 定理推导**（条件速度场回归与边际场同梯度）→ Gaussian 条件路径 → **rectified flow/线性路径**（x_t=(1−t)x₀+tε、目标速度 ε−x₀、与 DDPM ε-prediction 同族映射）→ logit-normal t 采样 → **diffusion↔FM 统一表** → FM 优势（轨迹直→少步 solver→蒸馏友好）→ 高分辨率 timestep shift
- 7.6 latent-diffusion-and-dit（推导重章）：LDM（latent 空间账 64×、两阶段分工、scaling factor 数学）→ **conditioning 三机制**（cross-attention 公式/concat/in-context 对比表）→ text encoder 选择（CLIP-L/G、T5-XXL 为什么有帮助——SD3/PixArt ablation）→ U-Net block（ResBlock time embedding 注入）→ **DiT**（latent patchify 账、**adaLN-Zero 推导**（零初始化=identity 起点）、DiT scaling 账 Gflops↔FID）→ **MMDiT（SD3）**（双流各自 QKV + 联合 attention 矩阵公式走读）→ FLUX double→single 变体 [C] → QK-norm → multi-aspect bucketing → **效率线**（PixArt-α 三阶段训练、**SANA = Linear DiT**——呼应 4.3 + DC-AE f32）
- 7.7 t2i-systems-and-ar-image（收口章）：**SD3**（MMDiT+FM recap 汇总=现代开放参考实现）→ **SDXL**（base+refiner、**micro-conditioning**：original-size/crop-coords 公式）→ **Imagen** [B]（级联 64→256→1024、T5-only、dynamic thresholding 公式）→ **AR 生图**：LlamaGen（VQ tokens + plain decoder-only：图像=另一种语言的公式论证）、**VAR next-scale prediction**（K 个分辨率 token map 序列公式；raster vs scale 顺序的信息结构论证；scaling law 曲线）、**Infinity**（bitwise tokenizer）、**mask-predict 族**：MaskGIT（mask ratio schedule γ(t)=cos(πt/2)、confidence remasking）、MUSE（[B] 无官方实现）、**Emu3**（全 token 统一账）→ **对照总表**（diffusion vs AR vs mask-predict：似然/步数/可控性/与 LLM 组合性/质量现状）

- [ ] **Step 1-5：标准章节流程**（各章 commit message "content(7.x): ..."）

### Task 9: W4 波末对抗审查

**Files:** Create `docs/reviews/w4-review-report.md`

**内容要求**：独立 fresh-context evaluator（fable），三任务 + **强制重推导：7.2 DDPM 全链（闭式解/后验/ELBO→L_simple）、7.4 CFG、7.5 conditional FM 定理**；约定一致性专项（x₀/x₁/α/ᾱ/t 方向跨 7 章抽查）；时效性专项（SD3.5/FLUX 等 2025-2026 现状）；verification log 硬要求。

### Task 10: W4 修复、复审与收尾

**内容要求**：S1/S2 修复 + 复审；S3 清零；进度表 7 行关闭 + 审查记录区追加；spec 波次完成记录追加；术语表累积登记；commit。

---

## 完成定义（W4）

- 7 章全部通过：lint OK + 批量审查无未关闭 Important+ + evaluator 无未关闭 S1/S2 + verification log 落盘
- 进度表 7 行已关闭；spec 波次记录已追加
