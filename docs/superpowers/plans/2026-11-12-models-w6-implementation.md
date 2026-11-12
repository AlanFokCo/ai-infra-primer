# models 维度 W6 实施计划（P9 Video Generation，4 章）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W6：P9 视频生成（4 章：video VAE、时空 DiT、训练配方与加速/流式生成、控制与编辑）。领域迭代快，research pass 权重最高。

**Architecture:** 沿用已验证流水线（research pass → 分批并行写作（opus）+ 批量审查（opus）→ 波末 evaluator（fable）→ 修复复审 → 清扫收尾）。

**Tech Stack:** 同前。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P9 条目）

## Global Constraints

（沿用 + 累积裁决）：
- 字数 = 正文 CJK 汉字数（标准章 ≥3500、推导重章 ≥5500 软、短篇 ≥2500）
- 白名单现行锁定版
- 写作者不做网络调研；references.md 回写由写作者在各自 commit 内做（并行时收归 controller）
- 并行 git 纪律：禁用 --amend / reset；index.lock 等 5 秒重试
- 防崩纪律：curl 抓到网页内容不粘贴大段原文
- P9 特查：时空 token 账（T×H×W×3 → latent → attention 矩阵）、各家 VAE/DiT 参数对照、闭源系统（Sora/Veo/Kling/Movie Gen）只讲已披露信息并标 [C]

## Review Focus

1. **时空 attention 拓扑的 FLOPs 账**（full 3D vs factorized——9.2 核心）→ 波末强制重推导
2. **Diffusion Forcing / CausVid / Self-Forcing 的公式**（9.3 核心——视频生成与 LLM 推理数学的合流点）→ 波末强制重推导
3. **各家 VAE/DiT 参数对照的真实性**（CogVideoX/HunyuanVideo/Wan/Open-Sora/LTX）→ research pass + evaluator 抽查
4. **evaluator rubber-stamp** → verification log 硬要求

---

## W6 任务

### Task 1: W6 research pass（事实源锁定）

**Files:** Modify `models/references.md`；Create `docs/research/w6-sources.md`

- [ ] **Step 1: 核实与收录**——CogVideoX 2408.06072、HunyuanVideo 2412.03603、Wan 2503.20314、Wan2.2（repo/blog——SNR-split MoE 细节）、Open-Sora（需定位 2.0 报告）、Open-Sora-Plan、Mochi（无论文 [C]）、LTX-Video 2501.00103、StepVideo-T2V 2502.10248、Diffusion Forcing 2407.01392、CausVid 2412.07772、Self-Forcing 2506.08009、Causal-Forcing（需定位）、rCM（NVlabs）、SkyReels-V2 2504.02436、Movie Gen 2410.13720 [B/C]、ControlNet 2302.05543、LoRA 2106.09685、DreamBooth 2208.12242、null-text inversion 2211.09794、CameraCtrl（需定位）；repo 存在性与关键文件路径（THUDM/CogVideo→zai-org、Tencent-Hunyuan/HunyuanVideo、Wan-Video/Wan2.1+2.2、hpcaitech/Open-Sora、PKU-YuanGroup/Open-Sora-Plan、genmoai/models、Lightricks/LTX-Video、tianweiy/CausVid、guandeh17/Self-Forcing、thu-ml/Causal-Forcing、NVlabs/rcm、lllyasviel/ControlNet）
- [ ] **Step 2: 约定页**——时空记号（T×H×W×C、latent (t,h,w,c)、压缩率）；diffusion 约定沿用 W4 约定页（x₀=数据等）
- [ ] **Step 3: 写 w6-sources.md**（每章一节）+ references.md 入库
- [ ] **Step 4: 自检 + Commit**（`research: lock W6 sources (P9)`）

### Task 2-5: P9 四章（分批并行写作）

**Files:** Create `models/09-video-generation/{01-video-vae,02-spatiotemporal-dit,03-training-and-acceleration,04-control-and-editing}.md`

分批（依赖序）：batch 1 = 9.1+9.2（VAE + 时空 DiT——9.2 的 attention 账是核心）→ batch 2 = 9.3+9.4（训练/加速 + 控制——依赖 9.1/9.2）

**内容要求（spec §5-P9 各节全量）**：
- 9.1 video-vae：视频数据账（T×H×W×3 raw vs latent——压缩率预算）；**3D causal conv VAE**（时间维 causal 结构公式、为什么 causal：图像=首帧兼容）；典型配置（spatial 8×8、temporal 4×；token 账：81 帧→21 latent 帧）；各家 VAE 对照（CogVideoX/HunyuanVideo/Wan 高压缩/LTX 更高压缩比换实时）；DC-AE 式深压缩在视频上的 tradeoff；重建质量评估（rFID/FVD 方法论一句话）；实例内存账（720p×5s×24fps 全链路数字表）
- 9.2 spatiotemporal-dit（推导重章）：spacetime patchify（latent (t,h,w,c) → 3D patches——Sora report formulation [C]）；**attention 拓扑账**（full 3D O((T·H·W)²) vs factorized spatial O(T·(HW)²)+temporal O(HW·T²)——FLOPs 对比表 + 质量差异证据）；**CogVideoX**（expert-adaLN 公式、3D full attention）；**HunyuanVideo**（dual-stream→single-stream MMDiT 变体结构走读、text encoder = MLLM 的原因、token refiner）；**Wan**（DiT+FM+cross-attn 走读；**Wan2.2 SNR-split MoE**：按 timestep SNR 确定性路由 high-noise expert（布局/运动）vs low-noise expert（细节）——公式（SNR(t) 阈值路由；对比 4.5 token 级路由：这是 timestep 级 MoE）、27B/14B active 账）；text conditioning 对照表（cross-attn vs joint stream vs in-context）；**时空位置编码**（3D/分解式 RoPE——t/h/w 三轴，与 8.2 M-RoPE 数学同构）；分辨率/时长 bucketing；其他开放系统一段式（StepVideo-T2V 30B 账、Open-Sora 2.0、LTX-Video、Mochi [C]）
- 9.3 training-and-acceleration：训练配方（图像-视频联合训练：T2I 权重作 init 的原因、渐进分辨率/时长 schedule、timestep 采样 FM logit-normal recap → 7.5）；视频 CFG 变体公式；**step distillation 在视频**：DMD2 视频版；**CausVid**（双向 teacher → 4-step causal student 的非对称蒸馏公式——DMD 梯度 + ODE 轨迹初始化；注意 CausVid 无 GAN loss——GAN 分量属 DMD2，勿混淆；KV cache 流式推理——**视频生成与 LLM 推理数学的合流点**：prefill/decode 概念直接复用）；**Diffusion Forcing**（per-token 独立噪声水平——t 逐 token 而非全局，训练目标改写公式；统一 teacher forcing 与 diffusion 的框架）；**Self-Forcing**（训练时自 rollout + 分布匹配——exposure bias 的数学表述与消除）；2026 后续（Causal-Forcing、rCM——现状表）；长视频三路线（keyframe+插值 / 分块 AR / context 扩展账）；世界模型/交互式生成一句话展望（标注成熟度）
- 9.4 control-and-editing：**I2V**（条件注入公式家族：latent concat（首帧 latent + mask 通道）、noise augmentation（首帧加噪防复制粘贴）、CLIP image embedding cross-attn——Wan/Hunyuan 变体）；首尾帧控制（kf2v）；**ControlNet-for-DiT**（可训练复制分支 + zero-conv 公式——零初始化 = 起点即原模型——与 adaLN-Zero 同思想）；**相机控制**：**Plücker ray embedding**（射线 (d, o×d) ∈ R^6 per pixel——从相机内外参构造 Plücker map 的公式、作为条件通道注入）、CameraCtrl/MotionCtrl 走读；轨迹条件（drag point embedding 公式）；**视频编辑**（DDIM inversion 推广到视频——inversion ODE 公式、误差累积；null-text inversion）；subject 一致性（LoRA ΔW=(α/r)BA 低秩公式 + DreamBooth 微调在视频上的形态）；多镜头/叙事结构（keyframe 链公式级）

- [ ] **Step 1-5：标准章节流程**（各章 commit message "content(9.x): ..."）

### Task 6: W6 波末对抗审查

**Files:** Create `docs/reviews/w6-review-report.md`

**内容要求**：独立 fresh-context evaluator（fable），三任务 + **强制重推导：9.2 时空 attention 拓扑 FLOPs 账、9.3 Diffusion Forcing 目标 + CausVid 非对称蒸馏**；时效性专项（Wan2.2/后续、Causal-Forcing、rCM 等 2026 现状）；verification log 硬要求。

### Task 7: W6 修复、复审与收尾

**内容要求**：S1/S2 修复 + 复审；S3 清零；进度表 4 行关闭 + 审查记录区追加；spec 波次完成记录追加；术语表累积登记；commit。

---

## 完成定义（W6）

- 4 章全部通过：lint OK + 批量审查无未关闭 Important+ + evaluator 无未关闭 S1/S2 + verification log 落盘
- 进度表 4 行已关闭；spec 波次记录已追加
