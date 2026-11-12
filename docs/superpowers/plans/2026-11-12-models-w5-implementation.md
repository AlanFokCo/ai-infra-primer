# models 维度 W5 实施计划（P8 Multimodal LLMs，5 章）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W5：P8 多模态 LLM（5 章：VLM 架构分类学、connector 与分辨率、训练目标、视频理解、统一理解生成）。

**Architecture:** 沿用已验证流水线（research pass → 分批并行写作（opus）+ 批量审查（opus）→ 波末 evaluator（fable）→ 修复复审 → 清扫收尾）。

**Tech Stack:** 同前。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P8 条目）

## Global Constraints

（沿用 + 累积裁决）：
- 字数 = 正文 CJK 汉字数（标准章 ≥3500、推导重章 ≥5500 软、短篇 ≥2500）
- 白名单现行锁定版（T10 收编 \det/\varnothing/\middle/\iint 若已落地则用）
- 写作者不做网络调研；references.md 回写由写作者在各自 commit 内做（并行时收归 controller）
- 并行 git 纪律：禁用 --amend / reset；index.lock 等 5 秒重试
- 防崩纪律：curl 抓到网页内容不粘贴大段原文
- P8 特查：token 预算账与 M-RoPE 公式的精确性（Qwen2.5-VL 是主锚点）

## Review Focus

1. **M-RoPE 三元组分解公式**（Qwen2-VL/2.5-VL 的 t/h/w 分解——8.2 核心）→ 波末强制重推导
2. **Transfusion 完整 loss**（8.5 核心：$L=L_{LM}+\lambda L_{DDPM}$、mask 矩阵、BOI/EOI）→ 波末强制重推导
3. **统一生成各家机制区分**（Chameleon early-fusion vs Janus 解耦 vs Show-o 混合 vs Transfusion 连续 latent）→ 批量审查 + evaluator 抽查
4. **evaluator rubber-stamp** → verification log 硬要求

---

## W5 任务

### Task 1: W5 research pass（事实源锁定）

**Files:** Modify `models/references.md`；Create `docs/research/w5-sources.md`

- [ ] **Step 1: 核实与收录**——LLaVA 2304.08485、Flamingo 2204.14198、BLIP-2 2301.12597、Qwen2-VL 2409.12191、Qwen2.5-VL 2502.13923、InternVL 2312.14238、InternVL3（需定位）、LLaVA-OneVision 2408.03326、Pixtral 2410.07073、Molmo 2409.17146、MiniCPM-V 2408.01800、Ovis（需定位）、Chameleon 2405.09818、Transfusion 2408.11039、Janus 2410.13848、Janus-Pro 2501.17811、Show-o 2408.12528、Show-o2（需定位）、Bagel 2505.14683、Emu3 2409.18869、Apollo 2412.10360、VCD 2311.16922；repo 存在性与关键文件路径（haotian-liu/LLaVA、mlfoundations/open_flamingo、salesforce/LAVIS、QwenLM/Qwen2.5-VL、OpenGVLab/InternVL、deepseek-ai/Janus、showlab/show-o、baaivision/Emu3、ByteDance-Seed/Bagel）
- [ ] **Step 2: 约定页**——VLM 记号（vision encoder 输出 z_img、connector、token 预算账 N=HW/p² 复用 P6）；M-RoPE 三元组（t/h/w）写法
- [ ] **Step 3: 写 w5-sources.md**（每章一节）+ references.md 入库
- [ ] **Step 4: 自检 + Commit**（`research: lock W5 sources (P8)`）

### Task 2-6: P8 五章（分批并行写作）

**Files:** Create `models/08-multimodal-llm/{01-vlm-architecture-taxonomy,02-connectors-and-resolution,03-training-objectives,04-video-understanding,05-unified-understanding-generation}.md`

分批（依赖序）：batch 1 = 8.1+8.2（分类学 + connector/分辨率——8.2 的 M-RoPE 是核心）→ batch 2 = 8.3+8.4（训练目标 + 视频理解——依赖 8.1/8.2）→ batch 3 = 8.5（统一理解生成——依赖全部）

**内容要求（spec §5-P8 各节全量）**：
- 8.1 vlm-architecture-taxonomy：形式化（VLM = p(text|image, prompt)）；组件链（vision encoder → connector → LLM）；**三个接入点公式**（input-level LLaVA 范式、cross-attention-level Flamingo gated xattn h ← h + tanh(α)·Attn(h, z_img) α zero-init、混合 Q-Former 输出 tokens）；**token 预算账**（CLIP-336/14 → 576 tokens；AnyRes 4 切 → 2880；视频帧数账）；冻结策略与表征分析（LLM embedding 空间缺什么视觉特征——各家 ablation 证据）；参数规模组合表
- 8.2 connectors-and-resolution（推导重章）：**projector 家族**（MLP 2 层 GELU——LLaVA-1.5 的数据 ablation：小数据下 Q-Former 不划算）；**Q-Former/BLIP-2**（B 个 learnable queries + self-attn 文本可见性 mask + cross-attn 到冻结 ViT 输出 + ITC/ITM/ITG 三目标公式、两阶段训练）；Perceiver resampler；**token 压缩**：pixel shuffle/space-to-depth（2×2 相邻 token → 1 token × 4 通道公式，InternVL/Qwen 用法）、token merging；**分辨率处理**：固定分辨率信息损失账、**AnyRes tiling 数学**（LLaVA-NeXT 网格选择 + 缩略图拼接 + 位置编码处理公式）、**Qwen2-VL naive dynamic resolution**、**M-RoPE 完整公式**（position id 分解 (t,h,w) 三元组——文本三值相同、图像 t 恒定+(h,w) 网格、视频 t 递增——旋转作用在各频段）；Pixtral native resolution（2D RoPE + 免 tiling）；connector 选型决策表
- 8.3 training-objectives：**loss masking 公式**（L=−Σ_{t∈response} log p——mask 掉 prompt/图像 token，代码走读）；**多阶段配方**（stage1 alignment 只训 projector、stage2 multi-task/instruction 解冻哪些、stage3 偏好对齐）；interleaved 图文格式（`<image>` 占位符展开机制、多图位置编码排布）；**幻觉形式化**（定义、机制假设：language prior 压过视觉证据——p(text|img) vs p(text) KL 分析、缓解：数据侧 + 解码侧 contrastive decoding 公式 p̃=softmax((1+α)logit_img − α·logit_noise)——VCD）；**评测方法论**（benchmark 类型学、多选题局限、claimed vs 真实能力）
- 8.4 video-understanding：视频 = 图像 × 时间账（帧数 F × 每帧 tokens N——总预算表：1min@1fps×256 = 15K tokens 级）；**帧采样**（uniform vs dynamic FPS——Qwen2.5-VL 绝对时间编码进 M-RoPE t 轴公式）；**时间压缩**（3D patch、时空 pooling、temporal queries resampler）；流式/长视频（chunk + 循环 memory 公式 outline）；输出格式（temporal grounding 时间戳 special token 化）；视频-图像训练数据混合；视频评测方法论
- 8.5 unified-understanding-generation（推导重章）：**路线 1 离散 token**：VQ tokenizer + 全 AR（Emu3 公式）；**Chameleon**（early-fusion token 级混合；**稳定性技术全套**：QK-norm、norm 重排、z-loss、token dropout——混合模态训练发散问题与 ablation 证据）；**Janus/Janus-Pro**（解耦理解 SigLIP + 生成 VQ 双 encoder——表征冲突论证）；Show-o/Show-o2（AR + 离散 diffusion 混合目标公式）；**路线 2 连续 latent**：**Transfusion 完整公式**（同样本内 text tokens AR loss + image patches diffusion loss：L=L_LM+λL_DDPM，无 U-Net、同一 Transformer；attention mask = text causal + image 块内双向——mask 矩阵图；BOI/EOI special tokens；沿用 Llama 式 1D RoPE——原文无 2D RoPE 勿杜撰）；**路线 3 diffusion head**（LLM hidden state → 小型 diffusion/FM head 解码——SEED-X/Emu2 式）；GPT-4o 生图 [C]：已披露信息分析；**对照总表**（离散 vs 连续 vs diffusion head：保真/编辑/训练稳定/推理成本/与 LLM scaling 兼容）

- [ ] **Step 1-5：标准章节流程**（各章 commit message "content(8.x): ..."）

### Task 7: W5 波末对抗审查

**Files:** Create `docs/reviews/w5-review-report.md`

**内容要求**：独立 fresh-context evaluator（fable），三任务 + **强制重推导：8.2 M-RoPE 三元组分解、8.5 Transfusion 完整 loss + mask 矩阵、8.3 loss masking**；时效性专项（Qwen2.5-VL/Qwen3-Omni 等 2025-2026 现状）；verification log 硬要求。

### Task 8: W5 修复、复审与收尾

**内容要求**：S1/S2 修复 + 复审；S3 清零；进度表 5 行关闭 + 审查记录区追加；spec 波次完成记录追加；术语表累积登记；commit。

---

## 完成定义（W5）

- 5 章全部通过：lint OK + 批量审查无未关闭 Important+ + evaluator 无未关闭 S1/S2 + verification log 落盘
- 进度表 5 行已关闭；spec 波次记录已追加
