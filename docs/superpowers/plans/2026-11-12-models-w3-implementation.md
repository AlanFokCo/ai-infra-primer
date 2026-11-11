# models 维度 W3 实施计划（P5 Case Studies + P6 Vision Representations）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W3：P5 案例拆解（6 章：Llama/DeepSeek/Qwen/GLM/Kimi 谱系 + 跨家对照方法论）+ P6 视觉表征基础（3 章：ViT、CLIP 对比学习、Autoencoder 与 visual tokenizer），含 research pass 与波末 evaluator 对抗审查。

**Architecture:** 沿用 W1/W2 已验证流水线（research pass 锁定事实源 → 分批并行写作（opus）+ 批量审查（opus）→ 波末 evaluator（fable，强制重推导）→ 修复复审 → 清扫收尾）。过程模式、lint 脚本、references.md schema、进度表、审查协议全部沿用。

**Tech Stack:** 同前（Markdown + GitHub MathJax 白名单、PyTorch 教学片段、curl 调研、子 agent 审查）。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P5/P6 条目）

## Global Constraints

（spec §4 全量沿用 + W1/W2 裁决：字数=CJK 汉字数（标准章 ≥3500、推导重章 ≥5500 软目标、短篇 ≥2500）；白名单按现行锁定版（T14 收编 \lceil \circ \perp \square \leftarrow \prime 等待定项已收编则用之）；写作者不做网络调研；references.md 回写由写作者在各自 commit 内做（不再收归 controller——已被证明平顺））

- P5 各章固定模板（无背景/文化/影响评述）：(a) 配置谱系表；(b) 标志性架构决策的公式级分析；(c) 旗舰 config.json 逐字段精读；(d) 技术开放问题清单
- 时效性条款：P5 涉及 2025-2026 最新旗舰（GLM-4.6/4.7、Qwen3-Max、Kimi K3、Llama-4），一律以 research pass 锁定的调研为准；正文避免「最新/首个」类时效断言，除非标日期

## Review Focus

1. **P5 谱系表的事实密度**（每家多代模型的 config 数值）→ research pass 逐 config 实测 + 波末 evaluator 抽查
2. **跨 Part 引用一致性**（P5 大量回引 P2/P4 公式与账）→ 写作者必读相关章节；批量审查抽查
3. **闭源/无 paper 模型的证据分级**（Llama-4 [W]、GPT-4o [C] 等）→ evaluator 专项
4. **P6 的推导**（InfoNCE、ELBO、straight-through、RVQ 递推）→ 波末强制重推导
5. **evaluator rubber-stamp** → verification log 硬要求

---

## W3 任务

### Task 1: W3 research pass（事实源锁定）

**Files:** Modify `models/references.md`；Create `docs/research/w3-sources.md`

**Interfaces:** Consumes spec §5 P5/P6 条目 + 既有 references.md/w1-sources.md/w2-sources.md；Produces T2-T10 写作的唯一事实源

- [ ] **Step 1: P5 源核实与收录**——Llama-1 2302.13971、Llama-2 2307.09288、Llama-3 herd 2407.21783、Llama-4 [W]（blog/config 实测：Scout/Maverick 16/128 experts、early-fusion、iRoPE ⟨需核实⟩）；DeepSeek-V2 2405.04434、V3 2412.19437、V3.2 2512.02556（config 已有）；Qwen1.5⟨需定位⟩、Qwen2 2407.10671、Qwen2.5 2412.15115、Qwen3 2505.09388、Qwen3-Next [W]（blog）、Qwen3-Max⟨需定位核实⟩；GLM 2103.10360、GLM-130B 2210.02414、GLM-4 2406.12793⟨核实⟩、GLM-4.5 text 2508.06471、GLM-4.5V 2507.01006、GLM-4.6 [W]（config 实测 355B/32B）、GLM-4.7⟨需定位核实⟩；Moonlight/Muon 2502.16982、K1.5 2501.12599、K2⟨需定位：有无论文，无则 [W] model card⟩、Kimi Linear 2510.26692、K3 2607.24653；各旗舰 config.json 实拉（vocab/层数/heads/ MoE 字段/ctx）
- [ ] **Step 2: P6 源核实与收录**——ViT 2010.11929、SigLIP 2303.15343、SigLIP 2 2502.14786、NaViT 2307.06304、Swin 2103.14030、CLIP 2103.00020、alignment/uniformity 2005.10242、VAE 1312.6114、VQ-VAE 1711.00937、VQGAN 2012.09841、MAGVIT-v2 2310.05737、DC-AE 2410.10733；open_clip/vision_transformer/big_vision/taming-transformers/efficientvit 各 repo 存在性与关键文件路径
- [ ] **Step 3: 约定页**——P6 视觉记号（H×W×C、patch p、token 数 N=HW/p²）；P5 谱系表字段约定
- [ ] **Step 4: 写 w3-sources.md**（每章一节：可用数值/config 摘录/来源 URL/注意事项）+ references.md 入库
- [ ] **Step 5: 自检**（spec P5/P6 各章 Sources 清单逐一 grep references.md；w3-sources.md ≥ 150 行）
- [ ] **Step 6: Commit**——`git commit -m "research: lock W3 sources (P5+P6)"`

### Task 2-7: P5 案例六章（并行写作，模板统一）

**Files:** Create `models/05-case-studies/{01-llama-lineage,02-deepseek,03-qwen,04-glm,05-kimi-moonshot,06-cross-lab-comparison}.md`

**统一内容要求**（spec §5 各节 + 章固定模板 (a)-(d)）：
- 5.1 llama-lineage：谱系（Llama-1 标准配方 → 2 GQA → 3.1 vocab/15T/405B → 4 MoE+early-fusion [W]）；每代 config diff 表 + 技术原因；open-weights 研究 baseline 意义
- 5.2 deepseek：V2（MLA/DeepSeekMoE recap）→ V3（671B/37B、aux-loss-free、MTP、FP8、14.8T、成本严谨表述=全训练 2788K GPU-hours GPU 时口径）→ V3.2（DSA）→ R1 一句话交叉链接 training；旗舰 config 逐字段（first_k_dense_replace=3 等）
- 5.3 qwen：dense/MoE 双轨 → Qwen3（dense 6 + MoE 2、thinking 模式交叉链接 training、QK-Norm 替代 QKV bias⟨核实⟩）→ Qwen3-Next（hybrid 逐组件 recap → P4）→ Qwen3-Max⟨T1 核实⟩；vocab 150K 账
- 5.4 glm：GLM 2021（blank infilling + 2D 位置编码详拆——唯一自研 objective 谱系）→ GLM-130B → ChatGLM → GLM-4.5/4.6（355B/32B、200K；4.6 无独立 report 的事实标注 + 方法论示范）→ GLM-4.7⟨T1 核实⟩
- 5.5 kimi-moonshot：Moonlight（shared expert、sigmoid routing、upcycling 公式）→ K1.5（交叉链接 training）→ K2（1T/A32B、MuonClip qk-clip 公式详拆）→ Kimi Linear/KDA recap → K3（LatentMoE、Attention Residuals——T1 锁定 report）
- 5.6 cross-lab-comparison：跨家对照总表（attention/位置编码/MoE 粒度与路由/norm/vocab/ctx/cache 账/激活比/MTP/hybrid）+ 收敛/分歧机制分析 + **方法论章：如何独立读懂一个新模型**（config.json 字段字典表、HF modeling 阅读路径、tech report 五步读法、30 分钟拆解 checklist）

- [ ] **Step 1-5：标准章节流程**（各章 `git commit -m "content(5.x): ..."`；共享文件纪律按各 writer 惯例：README/references 回写由写作者在各自 commit 内做，顺序派发防冲突——若并行派发则收归 controller）

### Task 8-10: P6 三章（并行写作）

**Files:** Create `models/06-vision-representations/{01-vit,02-clip-and-contrastive,03-autoencoders-and-tokenizers}.md`

**内容要求（spec §5-P6）**：
- 6.1 vit：patchify=conv≡linear 公式、token 数账、2D 位置编码（learned/sin-cos/RoPE-2D——为 8.2 M-RoPE 埋伏笔）、与文本 decoder 结构对照表、Swin window attention（简要）、NaViT patch-n-pack、SigLIP loss 公式、SigLIP 2 多目标融合
- 6.2 clip-and-contrastive：**InfoNCE 完整推导**（batch 相似度矩阵双向 CE）、温度 τ 数学作用、batch size scaling、**zero-shot 分类=检索公式**、prompt ensembling、alignment/uniformity 几何诊断、CLIP 的两个下游角色伏笔（VLM tower → P8；T2I text encoder → P7）
- 6.3 autoencoders-and-tokenizers：**VAE ELBO 完整推导**（Jensen/KL 两路径、reparameterization、Gaussian KL 闭式、β、posterior collapse）；**VQ-VAE**（straight-through 梯度拷贝推导、commitment loss、EMA、codebook collapse）；VQGAN（+LPIPS+PatchGAN 对抗项——对抗 loss 就地讲）；**RVQ 残差递推 + bitrate 账（P10 复用）**；**LFQ（MAGVIT-v2）**；现代 AE：SD-VAE（f8/4ch/scaling 0.18215）、SD3/FLUX VAE（f8 16ch）、**DC-AE**（f32/f64 深压缩——SANA 基石）；离散 vs 连续选择表（AR 需离散、diffusion 惯用连续——P7/P8 伏笔）

- [ ] **Step 1-5：标准章节流程**

### Task 11: W3 波末对抗审查

**Files:** Create `docs/reviews/w3-review-report.md`

**内容要求**：独立 fresh-context evaluator（fable），三任务 + **强制重推导：6.2 InfoNCE、6.3 ELBO+straight-through+RVQ 递推、5.5 MuonClip qk-clip**；P5 时效性专项（各旗舰 config 抽查 + 2026 新动态）；verification log 硬要求。

### Task 12: W3 修复、复审与收尾

**内容要求**：S1/S2 修复 + 复审；S3 清零；w1-sources.md:118 指针修正（W1 遗留）；进度表 9 行关闭 + 审查记录区追加；spec 波次完成记录追加；术语表累积登记；commit。

---

## 完成定义（W3）

- 9 章全部通过：lint OK + 批量审查无未关闭 Important+ + evaluator 无未关闭 S1/S2 + verification log 落盘
- 进度表 9 行已关闭；spec 波次记录已追加
