# models 维度 W2 实施计划（P3 Objectives & Variants + P4 Modern Architectures）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W2：P3（3 章 objective/架构形态）+ P4（8 章现代架构专题），含 research pass 与独立 evaluator 对抗审查质量门。

**Architecture:** 与 W1 相同的流水线（research pass 锁定事实源 → 逐章写作（opus）+ 逐章审查（opus）→ 波末 evaluator 对抗审查（fable，强制重推导）→ 修复复审 → 清扫收尾）。过程模式、lint 脚本、references.md schema、进度表、审查协议全部沿用 W1 已交付物，本计划只定义 W2 的内容与研究增量。

**Tech Stack:** 同 W1（Markdown + GitHub MathJax 白名单、PyTorch 教学片段、curl 调研、子 agent 审查）。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P3/P4 条目为章节内容要求唯一来源）

## Global Constraints

（spec §4 全量沿用；W1 期间新增/裁决一并生效）

- 字数口径 = 总字符（含标点与公式代码）；推导重章 6000-9000、标准章 4000-7000、短篇 2500-4000
- 公式白名单以 STYLE_GUIDE §3.1 现行锁定版为准（含 round 1-4 扩展：希腊字母/箭头族/间距/\binom/\Pr/\big 族/\mid 等；裸 \{ \} | 为 base symbol）
- 写作者不做网络调研——一切事实来自 research pass 产出（references.md + w2-sources.md）；先收录后引用；references.md 可由写作者回写被引章节，新增条目需 controller 授权
- 代码走读链接用 references.md 已收录条目或按 W1 先例补登（[A]、URL 字段、核实日期）
- 每章模板：metadata 块 → 正文 → 总结（表格/公式清单）→ 练习（2-4）→ 参考；lint 必过

## Review Focus

1. **2025-2026 新内容的时效与真实性**（NSA 无官方代码、DSA/KDA/GDN-2/Qwen3-Next/Kimi K3 等）→ T1 research pass 全部 fetch 一手来源 + 波末 evaluator 复核
2. **数学约定一致性跨章漂移**（GDN/KDA 状态约定、MoE 路由公式、RoPE 缩放族）→ spec 已统一约定（S∈R^{dv×dk} 等），T1 在 w2-sources.md 里写死约定页，写作者遵守，evaluator 抽查
3. **推导重章的推导正确性** → 波末强制重推导：4.1 MLA、4.3 DeltaNet-GDN-KDA、4.5 MoE routing
4. **与 W1 章节的接口一致性**（P1-P2 的公式编号/记号/口径被 P3/P4 引用处）→ 写作者必读相关 W1 章节；章节审查抽查跨 Part 引用
5. **evaluator rubber-stamp** → verification log 硬要求（同 W1）

---

## W2 任务

### Task 1: W2 research pass（事实源锁定）

**Files:**
- Modify: `models/references.md`（P3/P4 条目入库）
- Create: `docs/research/w2-sources.md`

**Interfaces:**
- Consumes: spec §5 P3/P4 条目、W1 的 references.md/w1-sources.md
- Produces: T2-T12 写作的唯一事实源

- [ ] **Step 1: P3 源核实与收录**——BERT 1810.04805、RoBERTa 1907.11692、ALBERT 1909.11942、ELECTRA 2003.10555、T5 1910.10683、UL2 2205.05131、GLM 2103.10360、ModernBERT 2412.13663、GPT-3 2005.14165、von Oswald 2212.07677、Akyürek 2211.15661、Dai 2212.10559、Min 2202.12837、Pythia 2304.01373、Whisper 2212.04356；Anthropic induction-heads（transformer-circuits.pub，[B] 无 arXiv）；ModernBERT / Pythia 的 config 与 repo 核实
- [ ] **Step 2: P4 源核实与收录（重头，2025-2026 内容一律 fetch 一手来源）**——MQA 1911.02150、GQA 2305.13245、DeepSeek-V2 2405.04434、V3 2412.19437、V3.2 2512.02556、NSA 2502.11089、MoBA 2502.13189、StreamingLLM 2309.17453、Longformer 2004.05150、Mistral 2310.06825、Katharopoulos 2006.16236、Performer 2009.14794、RetNet 2307.08621、DeltaNet 2406.06484、GDN 2412.06464、GDN-2 2605.22791、KDA 2510.26692、Mamba 2312.00752、Mamba-2 2405.21060、MiniMax-01 2501.08313、Jamba 2403.19887、Shazeer 1701.06538、Switch 2101.03961、GShard 2006.16668、expert choice 2202.09368、ST-MoE 2202.08906、DeepSeekMoE 2401.06066、Mixtral 2401.04088、Qwen3 2505.09388、PI 2306.15595、YaRN 2309.00071、LongRoPE 2402.13753、RULER 2404.06654、IOI 2211.00593、Gemma Scope 2408.05147、LLaDA 2502.09992、SEDD 2310.16834、differential attention 2410.05258、FlashAttention 2205.14135、FA2 2307.08691、speculative decoding 2211.17192；**Qwen3-Next**（HF config 实测 + 官方 tech report/blog 定位）、**Kimi K3**（report/博客定位）、**Qwen2.5-1M**（tech report 定位）；config 实测：Qwen3-Next-80B-A3B-Instruct（48 层 36+12、512 experts top-10+shared、zero-centered 字段）、Kimi-Linear-48B-A3B（20 KDA + 7 full-attn、mla_use_nope）、MiniMax-01（attn_type_list）
- [ ] **Step 3: 约定页**——在 w2-sources.md 开头写「W2 数学约定」：linear attention 状态约定（S∈R^{d_v×d_k}、o=S_t q、写入 v kᵀ、擦除右乘 (I−βkkᵀ)，沿用 2.7 修复后口径）；MoE 记号（N_e、k、f_i、P_i、b_i bias）；RoPE 缩放记法（s=扩展因子、θ_j、λ_j）；KV cache 公式引用 2.4 式号
- [ ] **Step 4: 写 w2-sources.md**（每章一节：可用数值/config 摘录/来源 URL/注意事项）+ references.md 按 schema 入库全部条目
- [ ] **Step 5: 自检**——spec P3/P4 各章 Sources 清单逐一 grep references.md 存在；w2-sources.md ≥ 150 行
- [ ] **Step 6: Commit**——`git add models/references.md docs/research/w2-sources.md && git commit -m "research: lock W2 sources (P3+P4)"`

### Task 2: 章节 3.1 `01-objectives-and-masks.md` — Objective 与 Mask 统一形式化

**Files:** Create `models/03-objectives-and-variants/01-objectives-and-masks.md`

**Interfaces:** Consumes 1.1（(x',M,ℓ) 形式化）；Produces mask/objective 统一框架（2.4 已埋点、3.2/3.3/4.x 引用）

**内容要求（spec §5-3.1）**：统一框架（mask 矩阵 M∈{0,−∞}^{T×T} + loss mask + corruption 方案）；五种 objective 公式：CLM、MLM（15% masking 与 80/10/10 的分布 mismatch 数学分析）、prefix-LM（**GLM blank infilling + 2D 位置编码公式详拆**）、span corruption（T5 sentinel、span 长度分布）、UL2 mixture-of-denoisers；objective↔能力对应表（可 factorize 的数学判定）；perplexity 跨 objective 不可比论证；真实 masking 代码走读（BERT 官方/T5 span corruption/GLM infilling）。锚点：google-research/bert、THUDM GLM（zai-org）、ModernBERT。Sources：1810.04805、1910.10683、2205.05131、2103.10360、2412.13663

- [ ] **Step 1-5：标准章节流程**（取数 w2-sources §3.1 → 写作 → `bash scripts/lint-chapter.sh models/03-objectives-and-variants/01-objectives-and-masks.md` → 进度表+被引回写 → `git commit -m "content(3.1): objectives and masks"`）

### Task 3: 章节 3.2 `02-encoder-architectures.md` — Encoder-only 与 Encoder-Decoder

**Files:** Create `models/03-objectives-and-variants/02-encoder-architectures.md`

**Interfaces:** Consumes 3.1；Produces encoder 形态分析（P10 ASR Whisper 引用）

**内容要求（spec §5-3.2）**：encoder-only：双向 attention（M=0）；**无法自回归生成的证明**；输出接口（[CLS]/mean pooling/token-level head 公式）；NSP 公式与失败机制（RoBERTa ablation）；ALBERT 跨层参数共享 + embedding 分解；ELECTRA RTD（generator/discriminator 公式、样本效率论证）；encoder-decoder（T5 类）：cross-attention 机制公式（K,V 来自 encoder、一次计算复用）、参数分配账、token 效率分析；何时 enc-dec 占优（翻译/摘要/ASR——Whisper 活例，交叉链接 P10.2）；ModernBERT 走读（RoPE、local/global 交替、unpadding——encoder 的现代化形态）。Sources：2412.13663、1810.04805、1907.11692、1909.11942、2003.10555、1910.10683、2212.04356

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(3.2): encoder architectures"`）

### Task 4: 章节 3.3 `03-decoder-only-and-icl.md` — Decoder-only 与 In-Context Learning

**Files:** Create `models/03-objectives-and-variants/03-decoder-only-and-icl.md`

**Interfaces:** Consumes 2.4（KV cache）、3.1；Produces ICL/decoder-only 论证（4.x 各章引用）

**内容要求（spec §5-3.3）**：自回归为通用接口的三条机制论证（任务空间表述、KV cache 经济学、训练-推理一致性）；**ICL 形式化**；三层理论解释：(a) linear attention ≈ in-context 一步 GD（von Oswald/Akyürek 定理 sketch + Dai et al. 并列引用）；(b) induction heads 两层回路（attention pattern 组合公式、induction score 可测量）；(c) 任务识别视角（Min et al. ablation）；Pythia 实证（154 checkpoints、训练相变）；诚实标注开放领域。锚点：Pythia、TransformerLens、GPT-2。Sources：2005.14165、2212.07677、2211.15661、2212.10559、2202.12837、2304.01373、transformer-circuits.pub [B]

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(3.3): decoder-only and in-context learning"`）

### Task 5: 章节 4.1 `01-kv-compression-and-exact-attention.md` — KV 压缩与精确 Attention（推导重章）

**Files:** Create `models/04-modern-architectures/01-kv-compression-and-exact-attention.md`

**Interfaces:** Consumes 2.3/2.4（attention、KV cache）、2.5（RoPE）；Produces MLA/GQA 账（4.6/5.x/P5 复用）

**内容要求（spec §5-4.1）**：MQA（H_kv→1 公式、cache 账、质量损失机制）；GQA（g 组、H_kv=H_q/g、uptraining、Pareto 论文数据）；**MLA 完整推导**（RoPE 与低秩压缩的数学冲突 → c_kv=W_DKV·h 低秩 + decoupled RoPE（k^R 单独 64 维）→ W_UQ/W_UK/W_UV 上投影 → cache 账 512+64=576/token/层 vs MHA 实数对比 → **W_UK/W_UV absorption 对称推导**）；**FlashAttention 数学**（online softmax running max/sum 重缩放公式推导、tiling ASCII 图、**精确等价非近似**、IO 复杂度 O(T²d²/M) vs naive）；FA2/FA3 一句话（kernel 交叉链接 gpu）。锚点：DeepSeek-V3（官方 MLA 代码）、Llama-3.1-70B（GQA 8 KV）、flash-attention repo。Sources：1911.02150、2305.13245、2405.04434、2412.19437、2205.14135、2307.08691

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.1): KV compression and exact attention"`）

### Task 6: 章节 4.2 `02-sparse-attention.md` — Sparse Attention

**Files:** Create `models/04-modern-architectures/02-sparse-attention.md`

**Interfaces:** Consumes 2.3/2.4/2.5/4.1；Produces 稀疏方案谱系（5.2/5.5 案例复用）

**内容要求（spec §5-4.2）**：分类学（静态 vs 动态）；**SWA**（窗口 w、L·w 感受野推导、Mistral-7B config）；**SWA+full 混合层模式**（Gemma 3 5:1 local:global config 走读，与 4.4 hybrid 对照）；**attention sinks**（StreamingLLM 数学解释：softmax 归一化必须为 1 → 首 token 吸收剩余 mass）；**NSA**（compression/selection/sliding 三分支公式、block=64 硬件对齐；**[B]：有论文无官方 repo**）；**DSA**（lightning indexer 打分公式 → top-K=2048 选择 → MLA 只在选中位置算；sparse-from-dense 继续训练配方、~50% 长上下文算力削减账）；**MoBA**（块划分 + query 级块路由公式、SWA+MoBA hybrid）；质量风险（needle 类召回数据）。锚点：DeepSeek-V3.2-Exp（DSA 官方代码）、MoonshotAI/MoBA、Mistral-7B、Gemma 3。Sources：2502.11089、2512.02556、2502.13189、2309.17453、2004.05150、2310.06825、2503.19786

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.2): sparse attention"`）

### Task 7: 章节 4.3 `03-linear-attention-and-ssm.md` — Linear Attention 与 SSM（推导重章）

**Files:** Create `models/04-modern-architectures/03-linear-attention-and-ssm.md`

**Interfaces:** Consumes 2.3/2.4；Produces linear/hybrid 理论基座（4.4/4.6/5.5 复用）

**内容要求（spec §5-4.3，约定按 w2-sources 约定页）**：kernel 视角（结合律重排 (φ(Q)φ(K)ᵀ)V=φ(Q)(φ(K)ᵀV) 完整推导、FLOPs O(T²d)→O(Td²)）；Performer FAVOR+；**recurrent 形式**（S_t=S_{t−1}+v_t φ(k_t)ᵀ、z_t、o_t=S_t φ(q_t)/(z_tᵀφ(q_t))）——**linear attention ≡ 矩阵状态 RNN 等价性证明**；状态容量含义；RetNet（指数衰减 γ、parallel/recurrent/chunkwise 三形式公式）；**delta rule 家族**：DeltaNet（S_t=S_{t−1}+β_t(v_t−S_{t−1}k_t)k_tᵀ，从 min‖Sk_t−v_t‖² 一步 GD 推导）；**GDN**（S_t=α_t S_{t−1}(I−β_t k_t k_tᵀ)+β_t v_t k_tᵀ 逐项解释）；**GDN-2**（2026，erase/write 解耦公式）；**KDA**（channel-wise gating α_t∈R^d 对角 vs GDN 标量，更新公式与对角矩阵形式）；chunkwise 并行要点（WY representation outline，kernel 交叉链接 fla repo）；**Mamba/S6**（连续系统离散化 Ā=exp(ΔA)、B̄≈ΔB ZOH 公式；selectivity）；SSM↔attention 对照表；**理论边界**：d_v×d_k 固定状态的 associative recall 下界论证。锚点：fla-org/flash-linear-attention、NVlabs/GatedDeltaNet(+2)、MoonshotAI/Kimi-Linear、state-spaces/mamba、microsoft/unilm。Sources：2006.16236、2009.14794、2307.08621、2406.06484、2412.06464、2605.22791、2510.26692、2312.00752、2405.21060

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.3): linear attention and SSM"`）

### Task 8: 章节 4.4 `04-hybrid-architectures.md` — Hybrid 架构

**Files:** Create `models/04-modern-architectures/04-hybrid-architectures.md`

**Interfaces:** Consumes 4.1/4.3；Produces hybrid 形态对照（5.3/5.5 复用）

**内容要求（spec §5-4.4）**：互补性论证（linear state 省内存但 recall 受限 + 少数 full attention 保检索——4.3 结论的形式化）；交错模式对照表：**Qwen3-Next**（3:1 GDN:gated attention；gated attention 输出门控 o=σ(W_g h)⊙attn 公式；48 层=36+12 config 走读；zero-centered norm；MTP）、**Kimi Linear**（3:1 KDA:gated-MLA；**full-attn 层 NoPE 论证**（KDA 层已破位置对称性）；KV cache −75%、1M ctx decode ~6× 论文数字）、**Kimi K3**（KDA 沿用 + LatentMoE + Attention Residuals——以 T1 锁定的 report 为准）、Jamba（SSM:attention:MoE 交错）、Nemotron-H、MiniMax-01（lightning attention 7:1 交替）；**内存账统一表**（固定 state vs 线性 KV——各模型真实数字）；质量边界（hybrid 强弱任务——各论文数据）；设计空间（比例/位置/门控方式）。锚点：Qwen3-Next-80B-A3B-Instruct（HF modeling_qwen3_next.py）、Kimi-Linear、MiniMax-01、Jamba。Sources：2510.26692、2501.08313、2403.19887、Qwen3-Next report（T1 锁定）、Kimi K3 report（T1 锁定）

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.4): hybrid architectures"`）

### Task 9: 章节 4.5 `05-moe.md` — MoE 与 MTP（推导重章）

**Files:** Create `models/04-modern-architectures/05-moe.md`

**Interfaces:** Consumes 2.6（FFN）、1.3（scaling）；Produces MoE 路由数学（P5 各章复用）、EP 预告（cluster 交叉链接源）

**内容要求（spec §5-4.5）**：参数/计算解耦公式（FLOPs ∝ activated）；Chinchilla-inference 联合优化视角；**Shazeer 2017 noisy top-k gating** 公式；**load balancing aux loss α·N_e·Σf_i·P_i 完整推导**（乘积在均匀分布最小的证明）；**Switch**（top-1、capacity factor、token 溢出机制与质量影响）；z-loss（router logits 幅度惩罚，ST-MoE 出处）；**expert choice**（公式、天然均衡、自回归不适配）；**DeepSeekMoE**（fine-grained + shared experts 参数分配公式）；**DeepSeek-V3**（256+1、top-8、seq-mask balancing）；**aux-loss-free routing**（持久 bias 更新规则 b_i ← b_i+Δ·(target−observed)_i 公式走读、收敛行为、不扭曲路由分布的论证）；**MTP**（目标 L_MTP=Σ_k CE(f^{(k)}_θ(...))、模块结构、训练辅助/推理 self-speculative——接受率公式、系统细节交叉链接 inference）；**Medusa 式并行 MTP heads 对照**（一段）；router 病理（塌缩动力学、利用率不均、振荡——诊断指标）；MoE×scaling laws（fine-grained scaling，V3 曲线）；EP/all-to-all 预告。锚点：DeepSeek-V3（官方 MoE 模块代码）、Mixtral-8x7B、Qwen3-30B-A3B。Sources：1701.06538、2101.03961、2006.16668、2202.09368、2202.08906、2401.06066、2412.19437、2401.04088、2505.09388、2211.17192

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.5): MoE and MTP"`）

### Task 10: 章节 4.6 `06-long-context.md` — Long Context

**Files:** Create `models/04-modern-architectures/06-long-context.md`

**Interfaces:** Consumes 2.5（RoPE）、4.1/4.2/4.3（cache 经济学）；Produces 长上下文方法体系（P5 复用）

**内容要求（spec §5-4.6）**：四堵墙公式化（O(T²) FLOPs、KV O(T) 实数、softmax 熵稀释 log T 推导、RoPE 高频外推失败机制）；**RoPE 缩放族全推导**：PI（m→m/s、高频过度压缩损伤）、NTK-aware（base→base·s^{d/(d−2)} 谱分析）、**YaRN**（波长分段 ramp α/β 公式 + 温度修正 1/t=1/t₀+log(s)·c 推导：熵不变性）、LongRoPE（逐维搜索目标公式）、dual-chunk attention（三段编码公式）；**架构手段**（SWA 堆叠、Ring Attention 分块跨设备机制图——交叉链接 cluster、MLA/DSA/KDA cache 经济学对比表）；**训练配方**（staged extension schedule 要点、长文档 upsampling——Qwen2.5-1M 与 Llama-3.1 128K 配方）；**评测方法论**（RULER、needle 方法与局限、claimed vs effective context 对照）。锚点：Qwen2.5-1M、Llama-3.1（herd paper 128K 配方）、DeepSeek-V3（YaRN config 块——research pass 已实测）、Kimi-Linear（1M）。Sources：2306.15595、2309.00071、2402.13753、2404.06654、2309.17453、Qwen2.5-1M（T1 锁定）、2407.21783

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.6): long context"`）

### Task 11: 章节 4.7 `07-interpretability-primer.md` — 机制可解释性基础

**Files:** Create `models/04-modern-architectures/07-interpretability-primer.md`

**Interfaces:** Consumes 2.1（residual stream）、3.3（induction heads）

**内容要求（spec §5-4.7）**：residual stream 回顾（read-modify-write 数学框架）；**logit lens**（p̃_l=softmax(W_U h_l) 公式、用途与局限）；**superposition**（n>d 时近似正交叠加编码的数学表述、干扰误差公式）；polysemanticity；**SAE**（字典学习 h≈Dz、loss=‖h−Dz‖²+λ‖z‖₁、k-SAE/gating SAE 变体公式、monosemanticity 实证）；**circuits**（induction head 回链 3.3、IOI circuit 方法论）；**activation patching**（干预 do(x) 与因果追踪数学定义）；方法工具箱对照表（patching/path patching/probing 各测什么）；诚实边界（当前结论可信度分级）。锚点：TransformerLens、decoderesearch/SAELens + Gemma Scope SAEs、Pythia。Sources：transformer-circuits.pub 系列 [B]、2211.00593、2408.05147、2304.01373

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.7): interpretability primer"`）

### Task 12: 章节 4.8 `08-frontier-outlook.md` — 前沿方向（短篇）

**Files:** Create `models/04-modern-architectures/08-frontier-outlook.md`

**Interfaces:** Consumes 4.1-4.7；Produces P5 预告

**内容要求（spec §5-4.8）**：**diffusion LLM（文本）**（absorbing-state diffusion formulation：forward corruption=逐 token mask、reverse 参数化、ELBO→逐步 CE；与 AR 的公式级对比：并行解码/无 KV cache/当前质量差距证据；LLaDA、SEDD）；**MoD（mixture-of-depths）/early-exit**、**differential attention**（双 softmax 消噪公式，2410.05258）、**TTT/looped-depth recurrence**（各一段）；**test-time scaling 的架构含义**（推理链长度 → context 需求 → 架构压力）；**agentic 化的架构含义**（多轮工具调用=长上下文+状态管理，交叉链接 inference）；native multimodal → P8.5；每项标注成熟度（研究原型/工业验证/主流）。短篇 2500-4000 字。Sources：2502.09992、2310.16834、2410.05258

- [ ] **Step 1-5：标准章节流程**（`git commit -m "content(4.8): frontier outlook"`）

### Task 13: W2 波末对抗审查

**Files:** Create `docs/reviews/w2-review-report.md`

**内容要求**：起独立 fresh-context evaluator（fable），输入=11 章成品 + STYLE_GUIDE + references.md + spec §5 P3/P4 条目；三任务（事实核查自 fetch / 教学审查 / 规范审查）+ **强制重推导：4.1 MLA（含 W_UK/W_UV absorption）、4.3 DeltaNet-GDN-KDA 状态更新链、4.5 aux-loss/aux-loss-free 路由**；输出 findings + verification log；S1/S2 必修、S3 累计清零。波末另查：2025-2026 新内容的时效性（Qwen3-Next/Kimi K3 等锚点的最新状态）。

- [ ] **Step 1: 派发 evaluator → 收取报告落盘**（`docs/reviews/w2-review-report.md`，commit）

### Task 14: W2 修复、复审与收尾

**内容要求**：S1/S2 修复（resume 原作者；含 S1 章由 evaluator 复审）；S3 清零；w2-sources.md 残留修正；进度表 11 行关闭 + 审查记录区追加；spec 波次完成记录追加；references.md 术语表累积登记（3.1-4.8 首现术语）；lint 回归全过；commit。

- [ ] **Step 1-5：修复 → 复审 → lint 回归 → 进度与记录 → `git commit -m "content: W2 fixes and review closure (P3+P4 complete)"`**

---

## 完成定义（W2）

- 11 章全部通过：lint OK + 逐章 review 无未关闭 Important+ + evaluator 审查无未关闭 S1/S2 + verification log 落盘
- models/README.md 进度表 11 行已关闭；spec 波次记录已追加
