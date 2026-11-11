# ai-infra-primer · models 维度设计 Spec

- 日期：2026-11-09
- 状态：用户已确认设计（2026-11-09）；4 路独立对抗审查完成，全部 findings 已修复（2026-11-10，记录见文末附录）
- 范围：models 维度（五维度中的第一个）：文本 LLM 主干 + 视觉表征 + 扩散生图 + 多模态 LLM + 视频生成 + 语音音频，共 10 个 Part / 51 章

---

## 1. 目标与读者

**目标**：构建纯技术、公式级、可动手验证的生成式模型学习资料库的 models 维度。读完的计算机从业人员应当：

1. 能从零推导 Transformer 各组件的公式（含 √d_k 方差推导、RoPE 内积不变性、KV cache 公式、Pre/Post-LN 梯度分析）
2. 能读懂任意新模型的 tech report 与 config.json，独立判断其架构选择的代价与收益
3. 掌握扩散/流匹配/VAE/VQ/对比学习的完整数学体系（DDPM 闭式解、CFG、FM 目标、InfoNCE 均能独立推导）
4. 理解多模态接入（connector / M-RoPE / 统一生成）与音视频生成（时空 DiT / RVQ codec / RQ-Transformer）的公式级机制

**读者假设**：会编程、懂线性代数与概率论基础、用过但不一定懂 LLM 内部机制的工程师。

**成功判据**：
- 每章通过独立 evaluator 对抗审查，无未关闭的 S1/S2 级 findings
- 所有数值事实（参数量/数据量/日期/benchmark/config 值）带出处，标注证据等级 [A]/[B]/[C]
- 所有关键公式：推导不跳关键步 + 符号表 + 与官方代码对照
- 每章含练习（2-4 题）与公式清单式总结（禁空洞评论段）

## 2. 范围边界

**In scope**：模型结构、训练目标的数学形式、推理时的架构行为（KV cache 公式等模型侧内容）、各模态的表征与生成模型数学。

**Out of scope（交叉链接到兄弟维度，不展开）**：
- 训练算法细节（optimizer / PPO / GRPO / FP8 工程 / 分布式训练策略）→ `training/`
- serving 系统（continuous batching / PagedAttention / speculative decoding 系统实现）→ `inference/`
- GPU kernel / 硬件（FlashAttention 只讲数学与 IO 复杂度，kernel 实现 → `gpu/`）
- 集群（EP / all-to-all / Ring Attention 通信细节）→ `cluster/`
- 3D 生成 / 具身智能（未来可加 Part）
- GAN 专门章节（目标函数在 P7.1 讲；对抗 loss 在 VQGAN / vocoder / 蒸馏等用处就地讲）

**闭源系统处理**：Sora / Veo / Kling / Midjourney / GPT-4o / Gemini / FLUX 等无论文或无代码的系统，只讲已披露技术信息，标注 [C] 级，并用同级开源 [A] 锚点对照分析。禁止把媒体转述当事实。

## 3. 仓库结构

```
ai-infra-primer/
├── README.md                          # 仓库总览、五维度导航、阅读路径（重写）
├── STYLE_GUIDE.md                     # 写作规范（写作与审查 agent 的唯一标准）
├── docs/superpowers/specs/            # 本 spec 及后续 spec
├── models/
│   ├── README.md                      # 目录、前置知识、Part 依赖图、进度表
│   ├── references.md                  # 全维度文献汇总（Part 分组；chapter → sources 映射）
│   ├── 01-foundations/                # 3 章
│   ├── 02-transformer-anatomy/        # 8 章
│   ├── 03-objectives-and-variants/    # 3 章
│   ├── 04-modern-architectures/       # 8 章
│   ├── 05-case-studies/               # 6 章
│   ├── 06-vision-representations/     # 3 章
│   ├── 07-diffusion-image-generation/ # 7 章
│   ├── 08-multimodal-llm/             # 5 章
│   ├── 09-video-generation/           # 4 章
│   └── 10-audio-speech/               # 4 章
├── training/README.md                 # 占位：范围说明 + 与 models 的接口
├── inference/README.md                # 占位
├── gpu/README.md                      # 占位
└── cluster/README.md                  # 占位
```

章节文件名用英文（如 `01-formulation.md`），每 Part 内两位数编号。

## 4. 写作规范（STYLE_GUIDE.md 条款定义）

STYLE_GUIDE.md 是写作 agent 与 evaluator agent 的共同唯一标准，必须包含以下条款：

### 4.1 语言
- 中文正文；专业术语保持英文：attention、token、embedding、MoE、router、KV cache、score、latent、codec 等不翻译
- 首次出现的概念用「中文（English Term）」格式并给出定义；全书术语表由 references.md 附带
- 语气：资深工程师对同事讲话，直接、具体

### 4.2 纯技术规则（用户硬要求）
- **禁止历史叙事**：不写时间线、不写人物故事、不写「某年某司发布某模型」式背景段
- 论文只以 citation 形式出现，作为公式与论断的证据来源
- 「为什么」一律用机制、数学、账（FLOPs / 内存 / cache / 数据效率）回答，不用年代顺序回答
- 章末禁止空洞评论段；总结只允许用表格或公式清单

### 4.3 公式
- GitHub 原生 LaTeX：行内 `$...$`，独立 `$$...$$`；W1 首章完成后做渲染测试（align/cases/矩阵环境）
- 每个公式后跟符号表；推导不跳关键步；重要公式编号（章内 `(1)(2)...`）便于交叉引用
- 张量 shape 统一 `[B, T, H, D]` 记法，关键运算处标注 shape

### 4.4 代码
- 讲解性 PyTorch 片段，清晰优先于效率，标注「教学用，非优化实现」
- 引用官方实现时链接到 repo + 文件路径（+ 关键函数名），写作时核实路径存在
- 教学重写版与官方实现并列展示时，明确差异点

### 4.5 图
- ASCII 文本图为主（架构图、数据流、shape 走读）；流程图/依赖图可用 Mermaid
- 禁止引用外部图片链接（防失效）；论文原图不复制，用 ASCII 重绘并注明出处

### 4.6 引用与证据分级
- 正文内 `[Vaswani et al., 2017]` 格式，链接到 `models/references.md` 对应条目
- 证据等级显式标注：
  - **[A]** = 正式 paper（arXiv）+ official repo + open weights，三者齐——公式拆解可对照真实代码
  - **[B]** = 有 paper、无官方代码（或代码不全）——按论文拆公式，社区实现辅助并注明
  - **[C]** = 闭源（仅 blog / model card / 媒体披露）——只讲已披露信息，禁止推测当事实
  - **[W]** = 开放权重 + config 但无正式 paper（如 Mistral-7B、GLM-4.6、Llama-4）——架构事实以 config / modeling 代码为准，训练配方类断言降级谨慎处理
  - **教学参照** = 无 paper 的教学 repo（nanoGPT、lucidrains 复现系列）——不占证据等级，仅作代码对照
- **所有数值事实必须有出处**；arXiv ID 一律经该波 research pass 复核后才写入正文

### 4.7 锚点规则
- 每章开头 metadata 块列 anchors：锚点模型（repo 链接到具体文件）+ 关键论文（arXiv）+ 证据等级
- **代码/config 走读锚点优先 2024-2026 模型**且满足 [A] 级；老经典（Transformer / BERT / DDPM / CLIP / VQ-VAE / Whisper 原文）只作公式源头，不作代码走读对象
- 无论文的流行模型（FLUX、Mochi）只作 [B]/[C] 级补充对照

### 4.8 章节模板
```
# <Part编号>.<章编号> <中文标题>（English Title）

> 一句话定位 | 前置章节 | 锚点模型与论文（含证据等级）

## 正文各节（以事实或问题开场）

## 总结（表格或公式清单）

## 练习（2-4 题：推导题 / 计算题 / 代码题）

## 参考（本章引用列表，链接 references.md）
```

篇幅约定：推导重章 6000-9000 字（如 7.2 / 4.3 / 7.5）、标准章 4000-7000 字、短篇/展望章 2500-4000 字（如 4.8）；每章至少 1 张 ASCII 结构图或 shape 走读图。references.md 条目 schema 在 W0 定义（字段：key / 标题 / 作者 / arXiv ID 或 URL / 证据等级 / 核实日期 / 被引章节）。

### 4.9 去 AI 味 checklist（evaluator 审查项）
- 禁用模板化开头与过渡：「在当今…」「值得注意的是」「综上所述」「总的来说」「让我们…」
- 禁用排比堆砌、空洞形容词（「强大的」「革命性的」——除非引用原文并标注）
- 每节以具体事实、公式或问题开场
- 数字优先于形容词；机制优先于评价

---

## 5. 章节大纲（10 Part / 51 章）

标记说明：`⟨verify⟩` = 2025 年后的高不确定项（新模型状态 / 新报告编号）子集——叠加在 §4.6 的全量复核之上，表示最高优先级核查对象。

### Part 1 · Foundations（01-foundations，3 章）

#### 1.1 `01-formulation.md` — 语言建模的数学形式化
- **内容**：样本空间 = token 序列；模型定义 $f_\theta: X_{<t} \to \Delta^V$（概率单纯形）；概率链式法则 → autoregressive factorization $P(x_{1:T})=\prod_t P(x_t|x_{<t})$ 完整推导；训练目标 = NLL = cross-entropy；信息论视角 CE = H(p) + KL(p‖q_θ)、bits/token；perplexity = exp(CE) 及其陷阱（vocab 相关、跨 tokenizer 不可比）；teacher forcing 定义；三种 objective（CLM/MLM/prefix-LM）统一为 attention mask 矩阵 + loss mask 的形式化（细节 → P3）；token / vocabulary / context window 形式化定义；采样生成（temperature/top-p 公式简介，系统细节交叉链接 inference）
- **关键推导**：链式法则→AR 分解；CE↔KL；perplexity 性质
- **锚点**：Llama-3.1-8B、Qwen3-8B 的真实 config（vocab/ctx 数值算例）[A]
- **Sources**：Kaplan [arXiv:2001.08361]、GPT-3 [arXiv:2005.14165]、GPT-2（OpenAI 技术报告，无 arXiv）

#### 1.2 `02-compute-accounting.md` — 参数、计算、内存的账
- **内容**：matmul FLOPs = 2mnk 推导；前向 2N/token（每参数 1 次 MAC）；反向 = 2× 前向（对输入梯度 + 对权重梯度各一次）→ **C ≈ 6ND 完整推导**与适用条件（不含 attention 二次项）；attention 的 O(L·T²·d) 项单算；每层 FLOPs 分解表（QKV proj / attn matmul / out proj / FFN）；内存四账：参数（bf16 = 2N bytes）、激活（逐层 shape 账）、KV cache（完整公式，详见 2.4）、优化器态（6N~12N bytes，交叉链接 training）；实例演算：Llama-3.1-8B 与 DeepSeek-V3（671B 总 / 37B 激活）的 per-token 账
- **关键推导**：6ND；attention FLOPs；每层参数→FLOPs 换算
- **锚点**：Llama-3.1-8B、DeepSeek-V3 config [A]
- **Sources**：Kaplan [arXiv:2001.08361] 附录、DeepSeek-V3 [arXiv:2412.19437]

#### 1.3 `03-scaling-laws.md` — Scaling Laws（作为理论）
- **内容**：幂律形式与 log-log 线性；Kaplan：$L(N)=(N_c/N)^{\alpha_N}$ 及 L(D)、L(C) 形式；Chinchilla 联合形式 $L(N,D)=E+A/N^\alpha+B/D^\beta$；**compute-optimal 分配的 Lagrangian 推导**：约束 C=6ND 下最小化 L → $N_{opt}\propto C^a, D_{opt}\propto C^b$；Kaplan vs Chinchilla 结论差异及拟合方法原因；inference-aware tradeoff：小模型 over-training 的账（Llama-3.1-8B 用 15T tokens）；MoE scaling：fine-grained scaling law（activated params 主导，DeepSeekMoE 曲线）；emergence：定义（能力随规模非线性出现）、Wei 分类；**Schaeffer 测量学解释**：连续 vs 离散 metric 如何人为制造「跳变」（公式级分析）；局限（数据质量、能力维度不平滑）
- **关键推导**：Chinchilla 最优分配；emergence metric artifact
- **锚点**：Pythia（每模型 154 个中间 checkpoint，scaling/动态研究标配）[A]、OLMo 2（开放训练数据/日志）[A]
- **Sources**：[arXiv:2001.08361]、[arXiv:2203.15556]、[arXiv:2206.07682]、[arXiv:2304.15004]、DeepSeekMoE [arXiv:2401.06066]

### Part 2 · Transformer Anatomy（02-transformer-anatomy，8 章）

每章固定结构：动机（机制/账，非历史）→ 公式推导 → shape 走读 → PyTorch 教学片段 ↔ 官方代码对照 → 常见误解 → 练习。

#### 2.1 `01-overview.md` — 架构总览与设计约束
- **内容**：架构 = tensor 变换的组合（ASCII 全景图）；residual stream 形式化 $x \leftarrow x + f(\mathrm{LN}(x))$（Pre-LN 现代形式）；residual stream 视角：各层读写同一向量总线（为 P4.7 interpretability 埋伏笔）；一次 forward 完整 shape 表；**参数量 ≈12Ld² 推导**（attention 4d² + FFN 8d²，embedding 单算）；设计约束三指标（Transformer 原文 Table 1 框架）：每层计算量 / sequential operations / 任意两位置最大路径长度 O(1) vs O(n)；recurrence 的两个数学代价：不可并行（隐状态依赖链）、信息瓶颈（固定 d_h 携带 T 个 token 的信息量论证）；三种架构形态结构图（细节 → P3）
- **锚点**：Llama-3.1-8B config.json 逐字段 ↔ 结构图映射 [A]；OLMo 2 [A]
- **Sources**：[arXiv:1706.03762]、[arXiv:2407.21783]、OLMo 2 [arXiv:2501.00656]

#### 2.2 `02-tokenization-and-embeddings.md` — Tokenization 与 Embedding
- **内容**：BPE 训练算法（频率统计→merge 循环）+ 手算小例子 + 压缩视角；byte-level BPE：无 OOV 的机制；WordPiece：似然驱动的 merge；**Unigram LM（SentencePiece）：目标 $\max_\theta \sum_x \log \sum_{z\in S(x)} p(z)$，EM 推导 outline**、Viterbi 切分；tokenizer 质量指标：fertility / compression ratio 公式；**vocab size 三方权衡**：T×V×d 的账（序列长度 × embedding 参数 × softmax 计算），32K/128K/150K 真实对照表（Llama-2 vs Llama-3 vs Qwen3 vs Gemma）；embedding = one-hot × W_E 等价性；初始化 scale（0.02 惯例及依据）；**weight tying**：W_U = W_E^⊤ 的梯度耦合分析、tied vs untied 的账；现代锚点走读：Llama-3.1 与 Qwen3 tokenizer 真实代码；**softmax bottleneck**（输出头 rank 限制理论，一小节）；**tokenizer-free 对照段**：BLT byte-level entropy patching（一小节）⟨verify⟩
- **关键推导**：Unigram EM 目标；vocab 账；tying 梯度
- **锚点**：Qwen3（150K vocab）、Llama-3.1（128K）、HF tokenizers / sentencepiece [A]
- **Sources**：BPE [arXiv:1508.07909]、Unigram [arXiv:1804.10959]、WordPiece（Google NMT [arXiv:1609.08144]）、tying [arXiv:1608.05859]

#### 2.3 `03-attention-mechanism.md` — Attention 机制解剖
- **内容**：检索直觉：hard retrieval（argmax，不可导）→ soft retrieval（softmax，可导）；QKV 投影 $W^Q,W^K,W^V \in \mathbb{R}^{d\times d_k}$；$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$；**√d_k 完整推导**：q,k 分量 iid 零均值单位方差 → Var(q·k)=d_k → 不缩放时 |q·k|~√d_k 进入 softmax 饱和区 → softmax Jacobian $\mathrm{diag}(p)-pp^\top$ → 梯度 ∝ p_i(δ−p_j) → 饱和时梯度消失；multi-head：$\mathrm{head}_i$、concat·W^O、参数账、子空间解释；**permutation equivariance 证明**（无位置信息时 Attention(πX)=πAttention(X)）→ 位置编码必要性（→ 2.5）；复杂度：FLOPs O(T²d + Td²) 两项拆分、attention 矩阵内存 O(T²)；数值稳定：softmax max 减法；手写 PyTorch（~40 行）↔ F.scaled_dot_product_attention
- **关键推导**：Var(q·k)=d_k；softmax Jacobian；permutation equivariance
- **锚点**：Llama-3.1-8B attention config（32 Q heads；GQA 部分 → P4.1，本章讲 MHA 本体）、OLMo 2 modeling 代码 [A]
- **Sources**：[arXiv:1706.03762]

#### 2.4 `04-causality-and-kv-cache.md` — Causality 与 KV Cache
- **内容**：mask 数学：$M_{ij}=0\ (j\le i),\ -\infty\ (j>i)$，softmax 前的加性形式与极限语义；三种 mask（bidirectional / causal / prefix-LM）↔ P1 三 objective 对照表；autoregressive 生成算法伪码（prefill 一次 + decode 循环）；**KV cache 从零推导**：decode 步 t 的 attention 展开 → k_i,v_i (i<t) 与 t 无关 → 可缓存；q 不缓存的依赖结构论证；**cache size 公式** bytes = 2·L·H_kv·d_head·T·B·b 逐项解释（MHA 时 H_kv=H_q；GQA 的 H_kv<H_q 详 → P4.1）；实例账：Llama-3.1-8B/70B @128K（MLA/DSA 特例 → P4.1/P4.2，案例 → P5.2）；prefill vs decode：算术强度分析（prefill compute-bound 大 matmul；decode memory-bound 读大算小）→ roofline 入口（交叉链接 gpu/inference）；teacher forcing 与 causality 的一致性（训练时无信息泄漏论证）
- **关键推导**：KV 可缓存性（依赖结构）；cache size；算术强度
- **锚点**：Llama-3.1-8B、OLMo 2 generate 循环代码 [A]
- **Sources**：[arXiv:1706.03762]、MQA [arXiv:1911.02150]

#### 2.5 `05-positional-encoding.md` — 位置编码全谱系
- **内容**：必要性 = 2.3 permutation equivariance 推论；**sinusoidal**：$PE(pos,2i)=\sin(pos/10000^{2i/d})$，频率几何级数 $\omega_i=10000^{-2i/d}$，相对位移性质 PE(pos+k)=R_k·PE(pos)（旋转矩阵表示，推导）；learned absolute：参数表、外推 = 未定义；**相对位置族**：T5 bias（log 分桶公式）、ALiBi（score − m_h·|i−j|，slope $m_h=2^{-8h/H}$ 几何级数、长度外推机制分析）；**RoPE 完整推导**：2D 旋转 R(mθ) → d 维块对角 diag(R(mθ_1),…,R(mθ_{d/2})) → 复数视角 q_m = q·e^{imθ} → **核心定理 ⟨R_m q, R_n k⟩ 只依赖 (m−n) 的完整证明** → 频率 schedule θ_j = base^{−2j/d} → 波长 λ_j = 2π·base^{2j/d} 与频率覆盖 → 远程衰减性质 → partial RoPE（只旋转前 d_r 维：DeepSeek-V3 64/192、Qwen 用法，账）；两种等价实现代码（half-split / interleave）；**NoPE**：causal mask 本身提供隐式位置信号（机制论证 + 文献证据）；RoPE 外推失败模式预告（→ P4.6）
- **关键推导**：sinusoidal 相对位移；RoPE 内积不变性；波长账
- **锚点**：Llama-3.1（rope_theta=500000 真实 config）、DeepSeek-V3（partial RoPE 64/192 + rope_scaling 块——2026-11-10 核实：Qwen2.5 config 无 YaRN 字段，勿用作锚点）[A]
- **Sources**：RoFormer [arXiv:2104.09864]、ALiBi [arXiv:2108.12409]、T5 [arXiv:1910.10683]、NoPE [arXiv:2305.19466]

#### 2.6 `06-ffn.md` — FFN 与非线性
- **内容**：$\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2$；维度分析 d→4d→d（8d²/层 ≈ 2/3 层参数）；激活演化：ReLU → GELU（x·Φ(x)，高斯 CDF 视角、平滑性/梯度性质）→ **GLU 族 / SwiGLU**：$(\mathrm{Swish}(xW)\odot xV)W_2$，gating 的数学作用（乘性交互的表达力论证）、三矩阵参数账（d_ff 从 4d 调至 (8/3)d 保持参数恒定的推导）；**FFN-as-key-value-memory**：W_1 行 = key、W_2 列 = value，$\mathrm{FFN}(x)=\sum_i v_i\,\sigma(k_i\cdot x)$（Geva 分解），神经元激活稀疏性经验证据；MoE 预告：FFN → N 个并行 FFN + routing（→ P4.5）
- **关键推导**：SwiGLU 参数等价；FFN key-value 分解
- **锚点**：Llama-3.1（intermediate_size=14336 的真实数字账：为什么不是 4×4096）、OLMo 2 [A]
- **Sources**：GLU variants [arXiv:2002.05202]、GELU [arXiv:1606.08415]、FFN memory [arXiv:2012.14913]

#### 2.7 `07-norm-and-residual.md` — Normalization 与 Residual
- **内容**：LayerNorm 公式（特征维 μ,σ²，γ/β）与 scale/shift invariance；为什么 NLP 不用 BatchNorm（变长序列 / inference 统计漂移 / 时间维独立性，三条数学原因）；RMSNorm：y = γx/RMS(x)，去掉 re-centering 后数学上丢了什么（re-scaling only 的论证）；**Pre-LN vs Post-LN 梯度分析**（Xiong et al. 2002.04745 定理精确形式——2026-11-10 research pass 核实：Theorem 1 覆盖初始化时最后 FFN 层参数梯度：Post-LN $O(d\sqrt{\ln d})$ 与深度 L 无关、Pre-LN $O(d\sqrt{\ln d / L})$ 随深度衰减；且 Post-LN 梯度**靠近输出端大、向输入端衰减**——推导骨架 → warmup 的数学必然性。注意：勿写成「Post-LN 浅层梯度爆炸」，方向相反）；QK-Norm：对 q,k 归一化（解决 attention logits 爆炸 → softmax 饱和；出处 Henry et al. 2020，采用者 ViT-22B / Gemma 3——Gemma 2 用的是 soft-capping，勿张冠李戴）；residual：梯度流 $\partial L/\partial x_l = \partial L/\partial x_L\,(1+\partial/\partial x_l \sum f_i)$，identity 项保证梯度传输；zero-centered norm（RMSNorm−1 变体：Gemma / Qwen3-Next，公式与动机）；DeepNorm 残差缩放系数（深层稳定性）；logit soft-capping（Gemma-2：c·tanh(z/c) 公式，动机）
- **关键推导**：梯度流 identity 项；Pre/Post-LN 梯度尺度
- **锚点**：Gemma 3（QK-norm——Gemma 3 以 QK-norm 取代 Gemma 2 的 soft-capping；pre+post RMSNorm 真实 config）[A]、Gemma 2（logit soft-capping 出处）[A]、Llama-3.1（RMSNorm + Pre-LN）[A]、Qwen3-Next（zero-centered）[A]
- **Sources**：LayerNorm [arXiv:1607.06450]、RMSNorm [arXiv:1910.07467]、Pre-LN [arXiv:2002.04745]、DeepNorm [arXiv:2203.00555]、Gemma 2 [arXiv:2408.00118]、Gemma 3 ⟨verify⟩ [arXiv:2503.19786]、QK-norm ⟨verify⟩ [arXiv:2010.04245]

#### 2.8 `08-assembly.md` — 组装完整 Decoder Block
- **内容**：现代标准配方完整 block 公式（h ← h + Attn(RMSNorm(h)); h ← h + FFN(RMSNorm(h))，全下标写出，GQA+RoPE+SwiGLU 组合）；**config.json ↔ 公式映射表**（num_hidden_layers / hidden_size / num_attention_heads / num_key_value_heads / intermediate_size / rope_theta / rms_norm_eps 逐字段）；~150 行 minimal decoder-only PyTorch（整合本 Part 全部组件）；官方实现走读：transformers `modeling_llama.py`（LlamaAttention.forward / LlamaMLP / LlamaRMSNorm 代码摘录 ↔ 公式对照）；单层单 token FLOPs 全算（Llama-3.1-8B 数值例子）；参数量对账练习（config → 总参数 ↔ 官方数字）；常见误解清单（head_dim ≠ hidden/heads 的反例、bias 有无、tied embeddings）；**结构变体：parallel attention/FFN block**（PaLM / GPT-J / Falcon：h ← h + Attn(LN(x)) + FFN(LN(x)) 单 LN 并行分支公式与梯度/吞吐含义）
- **锚点**：Llama-3.1-8B（主）、OLMo 2（对照）、nanoGPT（教学参照）
- **Sources**：[arXiv:1706.03762]、[arXiv:2407.21783]、[arXiv:2501.00656]

### Part 3 · Objectives & Variants（03-objectives-and-variants，3 章）

三种架构形态的技术分析（无历史叙事：BERT/T5 原文只作公式源头，走读用现代锚点）。

#### 3.1 `01-objectives-and-masks.md` — Objective 与 Mask 的统一形式化
- **内容**：统一框架：架构差异 = attention mask 矩阵 $M\in\{0,-\infty\}^{T\times T}$ + loss mask + 输入 corruption 方案的组合；五种 objective 公式：**CLM**（causal CE over 全 token）、**MLM**（BERT：15% masking 与 80/10/10 策略的分布 mismatch 数学分析）、**prefix-LM**（GLM blank infilling：2D 位置编码公式——GLM 谱系的独门 objective，详拆）、**span corruption**（T5：sentinel token、span 长度分布公式）、**UL2 mixture-of-denoisers**（regular/extreme 模式）；objective ↔ 能力对应表：哪些可自回归生成（条件分布可 factorize 的数学判定）；perplexity 跨 objective 不可比的论证（loss 支撑集不同）；真实 masking 代码走读（BERT 官方 / T5 span corruption / GLM infilling）
- **关键推导**：MLM 期望公式；GLM infilling objective；不可比性论证
- **锚点**：google-research/bert（官方 masking 代码）、THUDM GLM（infilling 实现）、ModernBERT [A]
- **Sources**：BERT [arXiv:1810.04805]、T5 [arXiv:1910.10683]、UL2 [arXiv:2205.05131]、GLM [arXiv:2103.10360]

#### 3.2 `02-encoder-architectures.md` — Encoder-only 与 Encoder-Decoder
- **内容**：**encoder-only（BERT 类）**：双向 attention（M=0）；**能力边界**：无法自回归生成的证明（无条件分布 factorization、无增量计算、每 token 表征依赖全序列）；输出接口：[CLS] / mean pooling / token-level head 公式；NSP objective 公式及失败机制（任务混淆：topic vs coherence——RoBERTa ablation 证据）；**ALBERT**：跨层参数共享公式（参数账）、embedding 分解 V≪H；**ELECTRA**：replaced-token detection（小 generator 腐蚀输入 + discriminator 逐 token 二分类，目标公式、样本效率论证：每 token 都有学习信号 vs MLM 只有 15%）；**encoder-decoder（T5 类）**：cross-attention 机制公式（Q 来自 decoder、K,V 来自 encoder 输出、encoder 一次计算重复使用——一种「cache」）；参数分配账（enc/dec 分置 vs decoder-only 同参数对比）；token 效率分析（双向编码每参数信息密度）；何时 enc-dec 占优（输入固定输出变化任务：翻译/摘要/ASR——Whisper 为活例，交叉链接 P10.2）
- **关键推导**：NSP 公式；RTD 目标；cross-attention 依赖结构
- **锚点**：**ModernBERT**（现代 encoder：RoPE、local/global 交替 attention、unpadding）[A] [arXiv:2412.13663]、BERT/RoBERTa/ELECTRA 官方 repo、T5、Whisper（enc-dec 例证）
- **Sources**：[arXiv:2412.13663]、[arXiv:1810.04805]、RoBERTa [arXiv:1907.11692]、ALBERT [arXiv:1909.11942]、ELECTRA [arXiv:2003.10555]、[arXiv:1910.10683]、Whisper [arXiv:2212.04356]

#### 3.3 `03-decoder-only-and-icl.md` — Decoder-only 与 In-Context Learning
- **内容**：自回归为通用接口的三条机制论证：(1) 任意条件建模任务可写成序列续写（任务空间数学表述）；(2) KV cache 经济学只适配 causal（增量计算）；(3) 训练-推理一致性（teacher forcing 与生成共用参数）；**ICL 形式化**：推理时条件于 demonstration 序列、无参数更新；三层理论解释：(a) **隐式梯度下降**：linear attention 层 ≈ 对 in-context 样本做一步 GD（von Oswald / Akyürek 定理公式 sketch：in-context 线性回归 ↔ attention 权重实现 GD 更新）；(b) **induction heads**：previous-token head 写入 + induction head prefix-match & copy 的两层回路（attention pattern 组合可写成公式；induction score 可测量）；(c) 任务识别视角（Min et al.：label 分布与 input 分布可分开学习——ablation 证据）；Pythia 实证：induction head 形成时机（训练中的相变）；诚实标注：ICL 理论仍是开放领域
- **关键推导**：linear attention↔GD 等价（sketch）；induction head 组合公式
- **锚点**：Pythia（154 intermediate checkpoints/model）[A]、TransformerLens（回路测量工具）[A]、GPT-2 官方代码
- **Sources**：GPT-3 [arXiv:2005.14165]、von Oswald [arXiv:2212.07677]、Akyürek [arXiv:2211.15661]、Dai et al. [arXiv:2212.10559]、Min et al. [arXiv:2202.12837]、Anthropic induction heads（transformer-circuits.pub, 2022，[B] 无 arXiv）

### Part 4 · Modern Architectures（04-modern-architectures，8 章）

#### 4.1 `01-kv-compression-and-exact-attention.md` — KV 压缩与精确 Attention 加速
- **内容**：**MQA**：单 KV head 公式、cache 账（H_kv: H_q→1）、质量损失机制；**GQA**：g 组、H_kv=H_q/g；uptraining（MQA→GQA 继续训练）；cache/质量 Pareto（论文数据表）；**MLA 完整推导**：动机——RoPE 与低秩 KV 压缩的数学冲突（旋转作用于压缩向量会破坏低秩结构）；解法：低秩联合压缩 $c_{kv}=W_{DKV}h\in\mathbb{R}^{d_c}$ + decoupled RoPE（$k^R$ 单独小维压缩）；上投影 W_UQ/W_UK/W_UV 公式；cache 账：512+64=576/token/层 vs MHA 2·H·d_head 真实数字对比；**W_UK / W_UV absorption**：W_UK 吸进 Q 投影、W_UV 吸进 W_O 的对称数学技巧（推导，推理侧优化）；**FlashAttention 数学**：online softmax（running max/sum 增量更新与重缩放因子 $e^{m_{old}-m_{new}}$ 的正确性推导）；tiling 分块流程 ASCII 图；**结论：数学上精确等价于标准 attention（非近似）**；IO 复杂度 O(T²d²/M) vs naive（M=SRAM）；FA2/FA3 改进要点（kernel 细节交叉链接 gpu）
- **关键推导**：MLA 压缩+decoupled RoPE；W_UK absorption；online softmax 正确性
- **锚点**：DeepSeek-V3（官方 MLA 实现）[A]、Llama-3.1-70B（GQA 8 KV heads）[A]、Dao-AILab/flash-attention [A]
- **Sources**：MQA [arXiv:1911.02150]、GQA [arXiv:2305.13245]、DeepSeek-V2 [arXiv:2405.04434]、V3 [arXiv:2412.19437]、FlashAttention [arXiv:2205.14135]、FA2 [arXiv:2307.08691]

#### 4.2 `02-sparse-attention.md` — Sparse Attention
- **内容**：分类学：静态 pattern（编译期确定）vs 动态选择（输入相关）；**SWA**：窗口 w、L 层堆叠有效感受野 = L·w（推导）、Mistral-7B 真实 config；**SWA+full 混合层模式**：Gemma 3 的 5:1 local:global 交错 config 走读（与 4.4 的 linear+full 混合对照——另一条混合路线）；**attention sinks**：StreamingLLM 现象的数学解释（causal softmax 下首 token 吸收「剩余」attention mass——softmax 归一化必须为 1）；sink token 设计；**NSA**：compression（块级聚合表示公式）+ selection（top-n 块选择打分公式）+ sliding 三分支及加权合并；硬件对齐设计（block=64）；**DSA**：lightning indexer（独立轻量多头点积 indexer 打分公式）→ top-K（K=2048）token 选择 → MLA 只在选中位置算；sparse-from-dense 继续训练配方（从 V3.1-Terminus 出发，长上下文算力 ~50% 削减的账面数字）；**MoBA**：块划分 + query 级块路由（top-k 块公式）、无训练插入、SWA+MoBA hybrid 模式；质量风险评估：召回类任务（needle）表现数据
- **关键推导**：NSA 三分支；DSA indexer→top-K；感受野 L·w
- **锚点**：NSA（[B]：有论文、无官方 repo——DeepSeek 未发布 NSA 代码）、DeepSeek-V3.2-Exp（DSA）[A]、MoonshotAI/MoBA [A]、Mistral-7B（SWA）[A]、Gemma 3（5:1 混合层 config）[A]
- **Sources**：NSA [arXiv:2502.11089]、DSA/V3.2 [arXiv:2512.02556]、MoBA ⟨verify⟩ [arXiv:2502.13189]、StreamingLLM [arXiv:2309.17453]、Longformer [arXiv:2004.05150]、Mistral 7B [arXiv:2310.06825]

#### 4.3 `03-linear-attention-and-ssm.md` — Linear Attention 与 State-Space Models
- **内容**：**kernel 视角**：sim(q,k) 替换为 φ(q)ᵀφ(k) → 结合律重排 $(\phi(Q)\phi(K)^\top)V=\phi(Q)(\phi(K)^\top V)$，FLOPs O(T²d)→O(Td²) 完整推导；Performer FAVOR+（随机特征近似 softmax attention 公式）；**recurrent 形式**（**全章统一约定**：$S\in\mathbb{R}^{d_v\times d_k}$、$o_t=S_t\phi(q_t)$，与 GDN 论文一致——linear / DeltaNet / GDN / KDA 四式对齐并标注 shape）：$S_t=S_{t-1}+v_t\phi(k_t)^\top$、$z_t=z_{t-1}+\phi(k_t)$、$o_t=S_t\phi(q_t)\,/\,(z_t^\top\phi(q_t))$ —— **linear attention ≡ 矩阵状态 RNN 的等价性证明**；状态容量含义（固定 $d_v\times d_k$ 矩阵）；**RetNet**：指数衰减 γ，三种计算形式（parallel / recurrent / chunkwise 公式，chunkwise = 训练关键）；**delta rule 家族**：DeltaNet 状态更新 = 在线 GD 步 $S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top$（从 $\min_S\|Sk_t-v_t\|^2$ 的一步梯度下降推导）；**Gated DeltaNet**：$S_t=\alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top$ 逐项解释（α=全局遗忘、作用于整个旧状态；(I−βkkᵀ) 右乘 = 定点擦除；v kᵀ = 写入——约定同 GDN 论文 2412.06464）；**GDN-2**（2026）：erase/write 解耦公式；**KDA**：channel-wise gating（α_t ∈ R^d 对角 vs GDN 标量 α），更新公式与对角矩阵形式；chunkwise 并行实现要点（WY representation outline，kernel 交叉链接 fla repo）；**Mamba/S6**：连续系统 h'=Ah+Bx → 离散化 Ā=exp(ΔA)、B̄≈ΔB（ZOH 公式）；selectivity（Δ,B,C 输入依赖 vs LTI）；SSM↔attention 对照表；**理论边界**：d×d 固定状态 vs 增长 KV 的 associative recall 下界论证（纯 linear 在精确检索任务失败的数学与实证）
- **关键推导**：结合律重排；linear↔RNN 等价；delta rule=在线 GD；GDN/KDA 状态更新
- **锚点**：fla-org/flash-linear-attention [A]、NVlabs/GatedDeltaNet + GatedDeltaNet-2 [A]、MoonshotAI/Kimi-Linear（KDA）[A]、state-spaces/mamba [A]、microsoft/unilm（RetNet）[A]
- **Sources**：[arXiv:2006.16236]、Performer [arXiv:2009.14794]、RetNet [arXiv:2307.08621]、DeltaNet [arXiv:2406.06484]、GDN [arXiv:2412.06464]、GDN-2 ⟨verify⟩ [arXiv:2605.22791]、KDA [arXiv:2510.26692]、Mamba [arXiv:2312.00752]、Mamba-2/SSD [arXiv:2405.21060]

#### 4.4 `04-hybrid-architectures.md` — Hybrid 架构（linear + full attention 交错）
- **内容**：互补性论证：linear state 省内存但 recall 受限（4.3 结论）+ 少数 full attention 层保精确检索；交错模式对照表：**Qwen3-Next**（3:1 GDN:gated attention；gated attention = 输出门控 $o=\sigma(W_g h)\odot \mathrm{attn}$ 公式；48 层=36+12 真实 config 走读；zero-centered norm；MTP）、**Kimi Linear**（3:1 KDA:gated-MLA；**full-attn 层用 NoPE**——为什么可行：KDA 层已破位置对称性的论证；论文数字：KV cache −75%、1M ctx decode 吞吐 ~6×）、**Kimi K3**（沿用 KDA + LatentMoE + Attention Residuals——⟨verify⟩ research pass 锁定细节）、Jamba（SSM:attention:MoE 交错公式）、Nemotron-H（Mamba-2 hybrid）、MiniMax-01（lightning attention 7:1 交替）；**内存账统一表**：每层固定 state（d_state×d_head×H）vs 线性增长 KV——各模型真实数字；质量边界：hybrid 的强项（长上下文 PPL/吞吐）与弱项（精确复制/检索任务——各论文数据）；设计空间：比例/位置（哪些层 full attn）/门控方式
- **锚点**：Qwen3-Next-80B-A3B（HF `modeling_qwen3_next.py`）[A]、MoonshotAI/Kimi-Linear [A]、MiniMax-01 ⟨verify⟩、Jamba ⟨verify⟩
- **Sources**：Kimi Linear [arXiv:2510.26692]、MiniMax-01 [arXiv:2501.08313]、Jamba [arXiv:2403.19887]、Qwen3-Next tech report/blog ⟨verify⟩、Kimi K3 ⟨verify⟩

#### 4.5 `05-moe.md` — Mixture-of-Experts 与 MTP
- **内容**：动机公式：N_total 与 N_activated 解耦（FLOPs ∝ activated）；Chinchilla-inference 联合优化视角（固定算力预算下 MoE 的最优性论证——DeepSeek-V2 框架）；**Shazeer 2017 noisy top-k gating**：$G(x)=\mathrm{TopK}(\mathrm{softmax}(W_gx+\epsilon\cdot\mathrm{softplus}(xW_{noise})),k)$；**load balancing aux loss**：$\alpha N_e\sum_i f_i P_i$（f=token 分数、P=平均路由概率——为什么乘积在均匀分布时最小的推导）；**Switch**：top-1、capacity factor（capacity=CF·T/N_e，token 溢出/丢弃机制与质量影响）；z-loss（router logits 幅度惩罚公式，训练稳定性）；**expert choice**：专家选 token（公式、天然均衡、自回归推理的不适配问题）；**DeepSeekMoE**：fine-grained experts + shared experts（参数分配公式）；**DeepSeek-V3**：256 routed + 1 shared、top-8、seq-mask balancing；**aux-loss-free routing**：持久 bias 更新规则 $b_i \leftarrow b_i+\Delta\cdot(\mathrm{target}-\mathrm{observed})_i$（公式走读、收敛行为、vs aux loss 优劣：不扭曲路由分布）；**MTP**：目标 $L_{MTP}=\sum_k \mathrm{CE}(f^{(k)}_\theta(h_{<t},x_{t+1..t+k}))$、模块结构（附加 block + 输入移位）、训练=辅助 loss / 推理=self-speculative decoding（接受率公式，系统细节交叉链接 inference）；**Medusa 式并行 MTP heads** 设计对照（一段）；router 病理：塌缩（rich-get-richer 动力学）、利用率不均、振荡——诊断指标；MoE×scaling laws（fine-grained scaling：性能 ∝ activated params——V3 论文曲线）；EP 预告：all-to-all dispatch 公式级（细节交叉链接 cluster）
- **关键推导**：aux loss 最小点；bias 更新规则；MTP 目标
- **锚点**：DeepSeek-V3（官方 MoE 模块代码）[A]、Mixtral-8x7B（HF modeling_mixtral）[A]、Qwen3-30B-A3B [A]
- **Sources**：[arXiv:1701.06538]、Switch [arXiv:2101.03961]、GShard [arXiv:2006.16668]、expert choice [arXiv:2202.09368]、z-loss/ST-MoE [arXiv:2202.08906]、DeepSeekMoE [arXiv:2401.06066]、V3 [arXiv:2412.19437]、Mixtral [arXiv:2401.04088]、Qwen3 [arXiv:2505.09388]

#### 4.6 `06-long-context.md` — Long Context
- **内容**：四堵墙公式化：(1) FLOPs O(T²)（T=1M 单层数字账）；(2) KV 内存 O(T)（GQA vs MLA vs KDA 在 1M 下真实数字）；(3) **attention dilution**：softmax 熵随 T 增长（均匀分布熵 = log T；分数差距稀释推导）；(4) 位置外推失败：RoPE 高频分量在未见位置的波长溢出分析；**RoPE 缩放族全推导**：PI（m→m/s 线性插值，高频过度压缩损伤）、NTK-aware（base→base·s^{d/(d−2)}：低频插值高频不动的谱分析）、**YaRN**（按波长分段 ramp α,β 公式 + attention 温度修正 1/t = 1/t₀ + log(s)·c 推导：熵不变性）、LongRoPE（逐维搜索目标公式）、dual-chunk attention（intra/inter/successive 三段编码公式）；**架构手段**：SWA 堆叠、Ring Attention（分块跨设备机制图，交叉链接 cluster）、MLA/DSA/KDA 的 cache 经济学对比表；**训练配方**：staged extension（4K→32K→128K→1M schedule 要点、长文档 upsampling）；**评测方法论**：RULER（合成多任务测什么）、needle-in-haystack 方法与局限、claimed vs effective context（各家报告数字对照）
- **关键推导**：NTK 频谱；YaRN ramp+温度；熵稀释账
- **锚点**：Qwen2.5-1M ⟨verify⟩、Llama-3.1（herd paper 128K 配方）、DeepSeek-V3（YaRN config 字段）、Kimi-Linear（1M）[A]
- **Sources**：PI [arXiv:2306.15595]、YaRN [arXiv:2309.00071]、LongRoPE [arXiv:2402.13753]、RULER [arXiv:2404.06654]、StreamingLLM [arXiv:2309.17453]、Qwen2.5-1M ⟨verify⟩

#### 4.7 `07-interpretability-primer.md` — 机制可解释性基础
- **内容**：residual stream 视角回顾（层 = 对流的 read-modify-write 数学框架）；**logit lens**：中间层投影 $\tilde p_l=\mathrm{softmax}(W_U h_l)$（公式、用途与局限）；**superposition**：特征数 n > 维度 d 时的近似正交叠加编码（toy models 的数学表述：随机单位向量叠加 + 干扰误差公式）；polysemanticity；**SAE**：字典学习 h ≈ Dz，loss = ‖h−Dz‖²+λ‖z‖₁（L1 稀疏、k-SAE / gating SAE 变体公式）、特征拆分实证（monosemanticity）；**circuits**：induction head（回链 P3.3）、IOI circuit（方法论级）；**activation patching**：干预 do(x) 与因果追踪的数学定义；方法工具箱对照表（patching / path patching / probing 各自测什么）；诚实边界：当前结论的可信度分级
- **锚点**：TransformerLens [A]、decoderesearch/SAELens + Gemma Scope SAEs ⟨verify⟩ [A]、Pythia [A]
- **Sources**：transformer-circuits.pub 系列（2022-2024，[B] 无 arXiv）、IOI [arXiv:2211.00593]、Gemma Scope ⟨verify⟩ [arXiv:2408.05147]

#### 4.8 `08-frontier-outlook.md` — 前沿方向（短篇，标注成熟度）
- **内容**：**diffusion LLM（文本）**：absorbing-state diffusion formulation（forward corruption = 逐 token mask、reverse 参数化、ELBO → 逐步 CE）、与 AR 的公式级对比（并行解码 / 无 KV cache / 当前质量差距的证据数字）、LLaDA / SEDD；**test-time scaling 的架构含义**：推理链长度 → context 需求 → 架构压力（long context / cache 效率 / MTP 的关联）；**agentic 化的架构含义**：多轮工具调用 = 长上下文 + 状态管理需求（交叉链接 inference）；**MoD（mixture-of-depths）/ early-exit**、**differential attention**（双 softmax 消噪公式，[arXiv:2410.05258]）、**TTT / looped-depth recurrence**（各一段）；native multimodal → P8.5；每项标注成熟度（研究原型 / 工业验证 / 主流）
- **Sources**：LLaDA ⟨verify⟩ [arXiv:2502.09992]、SEDD [arXiv:2310.16834]

### Part 5 · Case Studies（05-case-studies，6 章）

每章固定模板（无背景/文化/影响评述）：(a) 配置谱系表（model × params × attention × MoE × norm × vocab × ctx）；(b) 标志性架构决策的公式级分析（什么公式、解决什么问题、换掉什么代价）；(c) 旗舰 config.json 逐字段精读；(d) 技术开放问题清单。

#### 5.1 `01-llama-lineage.md` — Llama 谱系（开源参照系）
- **内容**：Llama-1「标准配方」确立（RMSNorm+SwiGLU+RoPE+non-tied）→ Llama-2（70B 引入 GQA、4K ctx）→ Llama-3.1（vocab 32K→128K 的账、15T tokens、405B：GQA 8 KV heads、128K ctx——herd paper 架构表要点）→ Llama-4（Scout/Maverick：MoE 16/128 experts、early-fusion 多模态、Scout 长 ctx claim——[C] 级标注：无 paper，blog+config 为准）；每代 config diff 表 + 每个改动的技术原因；open-weights 作为研究 baseline 的机制层意义
- **锚点**：meta-llama/llama-models、Llama-3.1 HF config [A] / Llama-4 HF config [W]
- **Sources**：Llama-1 [arXiv:2302.13971]、Llama-2 [arXiv:2307.09288]、Llama-3 herd [arXiv:2407.21783]、Llama-4 [W 无 paper]

#### 5.2 `02-deepseek.md` — DeepSeek
- **内容**：谱系：V2（MLA 首秀 recap → P4.1；DeepSeekMoE fine-grained）→ **V3**（671B/37B、256+1 experts top-8、aux-loss-free recap、MTP recap、FP8 训练模型侧要点（quantization scheme，工程细节交叉链接 training）、14.8T tokens、YaRN 128K；成本数字的严谨表述：2.788M H800 GPU-hours 仅为 paper Table 1 的口径）→ V3.1-Terminus → **V3.2**（DSA recap → P4.2：lightning indexer、K=2048、sparse-from-dense；正式版整合 reasoning）→ R1（训练侧一句话，交叉链接 training）；旗舰 config 精读：DeepSeek-V3 config.json 逐字段（n_routed_experts / moe_layer_freq / **first_k_dense_replace=3**（前 3 层 dense！）/ kv_lora_rank=512 / qk_rope_head_dim=64）
- **锚点**：deepseek-ai/DeepSeek-V3、DeepSeek-V3.2-Exp 官方 repo [A]
- **Sources**：V2 [arXiv:2405.04434]、V3 [arXiv:2412.19437]、V3.2 [arXiv:2512.02556]

#### 5.3 `03-qwen.md` — Qwen
- **内容**：谱系：Qwen1.5（标准配方）→ Qwen2（GQA）→ Qwen2.5（全系列 + 1M 长上下文版）→ Qwen3（dense 6 尺寸 + MoE 2 尺寸：235B-A22B / 30B-A3B；thinking 模式 = 训练侧交叉链接；QK-Norm 替代 QKV bias ⟨verify⟩）→ **Qwen3-Next**（hybrid：GDN 3:1 + 高稀疏 MoE 80B/A3B + MTP + zero-centered norm——逐组件 recap 指向 P4 对应章）→ Qwen3-Max（>1T ⟨verify⟩）；vocab 150K 的多语言压缩率账；旗舰走读：Qwen3-Next-80B-A3B config（layer_types 列表字段）+ modeling 代码
- **锚点**：QwenLM/Qwen3、Qwen/Qwen3-Next-80B-A3B-Instruct HF [A]
- **Sources**：Qwen2 [arXiv:2407.10671]、Qwen2.5 [arXiv:2412.15115]、Qwen3 [arXiv:2505.09388]、Qwen3-Next ⟨verify⟩

#### 5.4 `04-glm.md` — GLM
- **内容**：谱系：**GLM 2021**（blank infilling objective 公式 recap → P3.1 + 2D positional encoding 公式详拆——唯一从自研 objective 出发的谱系，技术后果分析：理解生成同体）→ GLM-130B（双语、稳定性工程记录）→ ChatGLM → GLM-4 → **GLM-4.5/4.6**（355B/32B MoE、200K ctx；**事实标注：4.6 无独立 tech report，细节以 4.5 report + HF config 为准**——这本身作为「如何对待无 paper 模型」的方法论示范）→ GLM-4.7 ⟨verify⟩；GLM-4.5 文本 report = arXiv:2508.06471（已核实；2507.01006 为 4.5V 视觉版）
- **锚点**：zai-org/GLM（原 THUDM）、zai-org/GLM-4.5 [A]、GLM-4.6 HF config [W]
- **Sources**：GLM [arXiv:2103.10360]、GLM-130B [arXiv:2210.02414]、GLM-4.5 text report [arXiv:2508.06471]、GLM-4.5V [arXiv:2507.01006]

#### 5.5 `05-kimi-moonshot.md` — Kimi / Moonshot
- **内容**：谱系：长上下文服务时代（128K→200K，工程为主）→ **Moonlight**（MoE 架构论文：Muon optimizer 训练侧交叉链接；架构要点：shared expert、sigmoid routing ⟨verify⟩、SwiGLU、**upcycling**：dense→MoE 权重初始化技术公式）→ K1.5（RL 交叉链接 training）→ **K2**（1T/A32B；**MuonClip：Muon + QK-clip**——qk-logit 超阈值时缩放 γ 的公式：模型层稳定性技术，详拆；agentic 导向设计）→ **Kimi-Linear/KDA**（recap → P4.3/4.4）→ **K3**（2.8T、LatentMoE、Attention Residuals、native multimodal——⟨verify⟩ research pass 锁定 report 与证据等级）
- **锚点**：MoonshotAI/Kimi-K2、MoonshotAI/Kimi-Linear、MoonshotAI/Moonlight ⟨verify⟩ [A]
- **Sources**：Moonlight/Muon [arXiv:2502.16982]（Muon 原始出处为 Keller Jordan 博客，无 arXiv）、KDA [arXiv:2510.26692]、K1.5 [arXiv:2501.12599]、K2 model card ⟨verify⟩、K3 report ⟨verify⟩

#### 5.6 `06-cross-lab-comparison.md` — 跨家对照与方法论
- **内容**：总表（行=五家旗舰，列=attention 方案 / 位置编码 / MoE 粒度与路由 / norm / vocab / ctx / per-token cache bytes / 激活比 / MTP / hybrid）；收敛点机制分析：为什么都上 MoE（推理经济公式 recap）、为什么都探索 subquadratic（1M ctx 的 KV 账）、为什么 MTP 普及（speculative 收益）；分歧点：MLA（DeepSeek）vs GQA-hybrid（Qwen/Kimi）vs GQA（Llama/GLM）——cache/计算/实现复杂度三维对比；**方法论章：如何独立读懂一个新模型**——config.json 字段字典表（字段→公式/结构）、HF modeling 文件阅读路径、tech report 五步读法（架构表→ablation→数据→评测→局限）、30 分钟拆解 checklist
- **锚点**：前五章全部
- **Sources**：各家 report recap

### Part 6 · Vision Representations（06-vision-representations，3 章）

后续一切多模态内容的公共前置：ViT = Transformer 用于图像；VAE/VQ = 图像如何变成 latent/token。

#### 6.1 `01-vit.md` — Vision Transformer
- **内容**：patchify：$x\in\mathbb{R}^{H\times W\times 3}$ → N=HW/p² tokens；patch projection = conv(k=p, s=p) ≡ linear 的等价性公式；[CLS] token 与 pooling 方式；2D 位置编码（learned / sin-cos 2D / RoPE-2D 轴向旋转公式——为 P8 M-RoPE 埋伏笔）；与文本 decoder 的结构对照表（无 causal mask、双向、FFN ratio、norm 位置）；token 数账（224px/16 → 196）；变体：Swin window attention（window partition + shifted window mask 公式，简要）、**NaViT**（patch n' pack：变分辨率 packing 的 attention mask 构造公式）；**SigLIP**：sigmoid pairwise loss $-\frac{1}{N}\sum\log\sigma(z_{ij}(\langle v_i,t_j\rangle+b))$（vs InfoNCE：免跨样本归一化 → 小 batch 友好）；**SigLIP 2**（2025：多目标融合——sigmoid + captioning + self-distillation ⟨verify⟩；当代 VLM 默认 vision tower）；AIMv2（AR 预训练视觉表征 ⟨verify⟩，选读）
- **关键推导**：conv≡linear patch 投影；SigLIP loss
- **锚点**：google-research/vision_transformer [A]、HF siglip2 ⟨verify⟩ [A]、big_vision [A]
- **Sources**：ViT [arXiv:2010.11929]、SigLIP [arXiv:2303.15343]、SigLIP 2 ⟨verify⟩ [arXiv:2502.14786]、NaViT [arXiv:2307.06304]、Swin [arXiv:2103.14030]

#### 6.2 `02-clip-and-contrastive.md` — CLIP 与对比学习
- **内容**：问题定义：图文表征空间对齐；**InfoNCE 完整推导**：batch B 对 → 相似度矩阵 S=VTᵀ/τ → 行方向 CE（image→text）+ 列方向（text→image）取平均；对比学习视角：batch 内其余 B−1 个 = 负样本；温度 τ 的数学作用（梯度锐度影响；learnable τ = e^t 及 clamp）；batch size scaling（负样本数 → uniformity 提升，论文数据）；**zero-shot 分类 = 检索**：$p(y|x)\propto\exp(\cos(v_x,t_{prompt(y)})/\tau)$；prompt engineering / prompt ensembling 公式；表征几何诊断：alignment & uniformity 指标公式；embedding 空间的两个下游角色（VLM vision tower → P8；T2I text encoder → P7）；数据工程要点一句话（交叉链接 training）
- **关键推导**：InfoNCE 双向 CE；zero-shot 检索公式
- **锚点**：openai/CLIP [A]、mlfoundations/open_clip（训练 loss 代码摘录）[A]
- **Sources**：CLIP [arXiv:2103.00020]、alignment/uniformity ⟨verify⟩ [arXiv:2005.10242]

#### 6.3 `03-autoencoders-and-tokenizers.md` — Autoencoder 与 Visual Tokenizer
- **内容**：**VAE**：latent variable 模型 p(x)=∫p(x|z)p(z)dz 不可解 → 变分后验 q_φ(z|x)；**ELBO 完整推导**（Jensen 与 KL 非负性两条路径）；reparameterization z=μ+σ⊙ε（梯度流通推导）；Gaussian KL 闭式解；loss = 重建项 + β·KL；posterior collapse 机制；**VQ-VAE**：$z_q=\arg\min_k\|z_e-e_k\|$；**straight-through estimator**（梯度拷贝公式 ∂L/∂z_e := ∂L/∂z_q）；commitment loss β‖sg[e]−z_e‖² + codebook loss（EMA 更新公式替代）；codebook collapse 机制与缓解；**VQGAN**：+PatchGAN 对抗项 + LPIPS perceptual loss（公式定义）——对抗 loss 就地讲处之一；**RVQ**：残差递推 r₁=z, e₁=Q(r₁), r₂=r₁−e₁,…（codebook 容量 K^{n_q}、bitrate = n_q·log2K·fps 账——**P10 音频 codec 直接复用**）；**LFQ（MAGVIT-v2）**：lookup-free quantization z_i=sign(z_i)（隐式 codebook 2^d、熵正则公式）；**现代 AE**：SD-VAE（f8、4ch、scaling factor 0.18215 的数学：单位方差化）、SD3/FLUX VAE（f8 16ch）、**DC-AE**（f32/f64 深压缩：residual autoencoding + decoupled high-resolution adaptation 公式——SANA 基石）；**离散 vs 连续选择表**（AR 需离散、diffusion 惯用连续——P7/P8 伏笔）
- **关键推导**：ELBO；straight-through；RVQ 递推与 bitrate 账；LFQ
- **锚点**：CompVis/taming-transformers（VQGAN 官方）[A]、diffusers AutoencoderKL [A]、mit-han-lab/efficientvit（DC-AE 在 applications/dc_ae）[A]、lucidrains/vector-quantize-pytorch（教学参照）
- **Sources**：VAE [arXiv:1312.6114]、VQ-VAE [arXiv:1711.00937]、VQGAN [arXiv:2012.09841]、MAGVIT-v2 [arXiv:2310.05737]、DC-AE [arXiv:2410.10733]

### Part 7 · Diffusion & Image Generation（07-diffusion-image-generation，7 章）

全书数学密度最高的 Part：每个公式完整推导。

#### 7.1 `01-generative-model-map.md` — 生成模型家族统一对照
- **内容**：五家族统一框架（各自优化什么 / 怎么采样 / 似然可解性 / mode 行为）：**AR**：max Σ log p(x_t|x_<t)，精确似然、顺序采样；**VAE**：ELBO 下界（recap → P6.3）；**GAN**：minimax $\mathbb{E}[\log D(x)]+\mathbb{E}[\log(1-D(G(z)))]$（非似然、mode dropping 机制一句话）；**Flow**：change of variables $\log p(x)=\log p(z)+\log|\det \partial z/\partial x|$（可逆性约束）；**Diffusion/FM**：score/velocity 回归、迭代采样；对照总表（似然/步数/质量/可控性/与 LLM 可组合性）；本 Part 各章定位图；GAN 不展开（范围决策：对抗 loss 在用处就地讲）
- **Sources**：VAE [arXiv:1312.6114]、GAN [arXiv:1406.2661]、Glow [arXiv:1807.03039]、DDPM [arXiv:2006.11239]

#### 7.2 `02-ddpm.md` — DDPM 完整推导
- **内容**：**前向过程**：q(x_t|x_{t−1})=N(√(1−β_t)x_{t−1}, β_t I)；**闭式解推导**：α_t=1−β_t、ᾱ_t=∏α_s → q(x_t|x_0)=N(√ᾱ_t x_0, (1−ᾱ_t)I)（高斯卷积递推逐步展开）；采样式 x_t=√ᾱ_t x_0+√(1−ᾱ_t)ε；noise schedule：linear vs cosine（公式与频谱分析）；**反向过程**：p_θ(x_{t−1}|x_t)=N(μ_θ, σ_t²I)；**后验 q(x_{t−1}|x_t,x_0) 完整推导**（Bayes + 配方：μ̃ = √ᾱ_{t−1}β_t/(1−ᾱ_t)·x_0 + √α_t(1−ᾱ_{t−1})/(1−ᾱ_t)·x_t）；**ELBO 分解**：L_T + Σ L_{t−1} + L_0 三项写出；高斯 KL 闭式 → ε-参数化 $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta)$ → **简化 loss $L_{simple}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$**（与严格 ELBO 的 SNR weighting 差异推导）；x₀/v-prediction 等价换算（v=√ᾱ ε−√(1−ᾱ)x₀）；σ_t 选择（β_t vs β̃_t）；ε_θ 网络 = U-Net（结构预告 → 7.6）
- **关键推导**：前向闭式解；后验配方；ELBO→L_simple
- **锚点**：hojonathanho/diffusion（官方 TF）[A]、lucidrains/denoising-diffusion-pytorch（教学参照）
- **Sources**：DDPM [arXiv:2006.11239]、iDDPM/cosine [arXiv:2102.09672]

#### 7.3 `03-score-sde.md` — Score-Based 视角与 SDE/ODE
- **内容**：score 定义 s(x)=∇_x log p(x)（几何直觉：指向密度上升方向）；score matching 目标不可解（p 未知）→ **denoising score matching 等价性（Vincent 定理）推导**：∇log q(x_t|x_0) = −(x_t−√ᾱ x_0)/(1−ᾱ) 已知 → 用条件 score 回归与用真 score 回归同梯度（证明关键步骤）；与 DDPM 换算：ε_θ = −√(1−ᾱ_t)·s_θ；**SDE 表述**：VP-SDE dx=−½β(t)x dt+√β(t) dW、VE-SDE；Fokker-Planck（密度演化，结论级）；**probability-flow ODE**：dx/dt=[f−½g²∇log p]（推导思路：同密度路径的确定性 ODE）→ 确定性采样 + 精确似然（连续 change of variables）；reverse-time SDE（Anderson 定理公式）；统一视图表：DDPM / DDIM / ODE solver 都是同一家族的不同离散化
- **关键推导**：denoising SM 等价；PF-ODE
- **锚点**：yang-song/score_sde_pytorch（官方）[A]
- **Sources**：score-SDE [arXiv:2011.13456]、EDM [arXiv:2206.00364]、Vincent 2011（JMLR，无 arXiv）

#### 7.4 `04-sampling-and-guidance.md` — 采样器与 Guidance
- **内容**：ancestral sampling 伪码；**DDIM 推导**：非马尔可夫前向族（自由 σ）；σ=0 确定性形式 $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta$（= PF-ODE 一阶离散化）；跳步子序列；**classifier guidance**：∇log p(x|y)=∇log p(x)+∇log p(y|x)（Bayes）→ $\tilde\epsilon=\epsilon_\theta-w\sqrt{1-\bar\alpha}\nabla\log p_\phi(y|x_t)$；**CFG 完整推导**：隐式 classifier p(x|y)∝p(x)p(y|x)^w → $\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+w(\epsilon_\theta(x_t,y)-\epsilon_\theta(x_t,\varnothing))$；w 的数学含义（log-likelihood tilt）；失效模式：过饱和/多样性下降（score norm 膨胀机制）与缓解（dynamic thresholding 公式——Imagen、rescaled CFG）；**求解器**：PF-ODE 的指数积分思想（DPM-Solver：半线性结构 + Taylor 展开 outline）；步数-质量曲线；**蒸馏族**：progressive distillation（T→T/2 迭代公式）；**consistency models**（同 ODE 轨迹自一致性 f(x_t,t)=f(x_{t'},t') 性质公式；CT/CD）；LCM（latent 空间一致性）；**DMD**：分布匹配梯度 $\mathbb{E}[(s_{fake}-s_{real})\partial x/\partial\theta]$（real score = 冻结 teacher、fake score = 在线训练）；reflow/rectified flow（轨迹拉直公式）；**CFG / guidance 蒸馏**（把 w 蒸进单模型的目标公式——工业默认做法）；few-step 现状表（4-step/1-step 质量数字）
- **关键推导**：DDIM；CFG；DMD 梯度
- **锚点**：ermongroup/ddim [A]、openai/consistency_models [A]、tianweiy/DMD2 [A]
- **Sources**：DDIM [arXiv:2010.02502]、classifier guidance [arXiv:2105.05233]、CFG [arXiv:2207.12598]、consistency [arXiv:2303.01469]、LCM ⟨verify⟩ [arXiv:2310.04378]、DMD ⟨verify⟩ [arXiv:2311.18828]、DMD2 ⟨verify⟩ [arXiv:2405.14867]、rectified flow [arXiv:2209.03003]、DPM-Solver [arXiv:2206.00927]

#### 7.5 `05-flow-matching.md` — Flow Matching
- **内容**：CNF 定义 dx/dt=v_θ(x,t)、pushforward 密度；精确似然的 trace 公式（贵的原因）；**FM 目标** $L_{FM}=\mathbb{E}_{t,x\sim p_t}\|v_\theta(x,t)-u_t(x)\|^2$（u_t 真实速度场不可解）；**conditional FM 定理推导**：用条件速度场 u_t(x|x_1) 回归与用边际场同梯度（期望交换关键步）；Gaussian 条件路径 p_t(x|x₀)=N(α_t x₀, σ_t²I) → u_t 闭式；**全书约定：x₀=数据、t=1=噪声端**（与 7.2 DDPM 及锚点 SD3 一致；附 Lipman 原论文「x₁=数据」的约定映射框防混淆）；**rectified flow / 线性路径**：x_t=(1−t)·x₀+t·ε（ε~N(0,I)）、目标速度 = ε−x₀（与 DDPM ε-prediction 的同族映射公式；采样方向 t:1→0）；t 采样分布：logit-normal t~sigmoid(N(m,s))（SD3：为什么中间 timestep 更重要）；**diffusion↔FM 统一表**：同一 Gaussian 路径的两种参数化映射；FM 优势：轨迹直 → 少步 solver → 蒸馏友好；高分辨率的 timestep shift（SD3 resolution-dependent shift 公式）
- **关键推导**：conditional FM 定理；线性路径速度目标；FM↔diffusion 映射
- **锚点**：facebookresearch/flow_matching（官方库代码摘录）[A]、Stability-AI sd3.5 [A]
- **Sources**：FM [arXiv:2210.02747]、rectified flow [arXiv:2209.03003]、SD3 [arXiv:2403.03206]

#### 7.6 `06-latent-diffusion-and-dit.md` — Latent Diffusion、DiT 与 MMDiT
- **内容**：**LDM**：为什么 latent 空间（像素 vs f8 latent 的 token 数账：64× 削减；两阶段分工：VAE=感知压缩、diffusion=语义生成）；scaling factor 数学（z·0.18215：单位方差化）；**conditioning 三机制**：cross-attention $\mathrm{Attn}(W_Qz, W_Kc, W_Vc)$（c=text embedding 序列）、concat（通道）、in-context（token 拼接）对比表；text encoder 选择：CLIP-L/G（pooled+sequence）、**T5-XXL 为什么有帮助**（长文本/字符理解——SD3/PixArt ablation 数据）；**U-Net block**：ResBlock（time embedding 注入 scale-shift 公式）+ spatial/cross attention；**DiT**：latent patchify（p=2/4/8 的账）；**adaLN-Zero**：(1+γ)·LN(x)+β、gate α 零初始化（= identity 起点的推导）；DiT scaling 账（Gflops ↔ FID 线性关系）；**MMDiT（SD3）**：text/image 双流各自 QKV 投影 + **联合 attention 矩阵**（拼接 softmax 再拆回，公式走读）；FLUX 的 double→single stream 变体 [C]；QK-norm；multi-aspect bucketing；**效率线**：PixArt-α 三阶段训练（各 ablation 证明什么）、**SANA = Linear DiT**（linear attention 进扩散——呼应 P4.3 + DC-AE f32）
- **关键推导**：adaLN-Zero；MMDiT 联合 attention；latent token 账
- **锚点**：facebookresearch/DiT [A]、CompVis/latent-diffusion [A]、Stability-AI generative-models + sd3.5 [A]、black-forest-labs/flux [C]、NVlabs/Sana ⟨verify⟩ [A]、PixArt-alpha ⟨verify⟩ [A]
- **Sources**：LDM [arXiv:2112.10752]、DiT [arXiv:2212.09748]、SD3 [arXiv:2403.03206]、PixArt-α [arXiv:2310.00426]、PixArt-Σ [arXiv:2403.04692]、SANA [arXiv:2410.10629]

#### 7.7 `07-t2i-systems-and-ar-image.md` — T2I 系统拆解与 AR 生图
- **内容**：**SD3**（MMDiT+FM recap 汇总 = 现代开放参考实现 [A]）；**SDXL**（base+refiner 两阶段、**micro-conditioning**：original-size/crop-coords 作为 embedding 条件的公式——数据质量标注进模型）；**Imagen** [B]（级联 64→256→1024、T5-only encoder、dynamic thresholding 公式）；**AR 生图**：**LlamaGen**（VQ tokens + plain decoder-only：图像 = 另一种语言的公式论证——P2 全套直接复用）、**VAR next-scale prediction**（K 个分辨率 token map 序列 r₁,…,r_K 公式；raster vs scale 顺序的信息结构论证；scaling law 曲线）、**Infinity**（bitwise tokenizer 无限 vocab 机制 ⟨verify⟩）；**mask-predict 族**：MaskGIT 迭代并行解码（mask ratio schedule γ(t)=cos(πt/2) 公式、confidence remasking）、MUSE（[B]：有论文、Google 未开源）；**全 token 统一**：Emu3（图/视/文全 next-token：tokenizer 选择账）；**对照总表**：diffusion vs AR vs mask-predict（似然/步数/可控性/与 LLM 组合性/质量现状证据）
- **锚点**：FoundationVision/VAR + LlamaGen + Infinity ⟨verify⟩ [A]、baaivision/Emu3 [A]、sd3.5 [A]（MUSE 无官方实现，不作锚点）
- **Sources**：VAR [arXiv:2404.02905]、LlamaGen ⟨verify⟩ [arXiv:2406.06525]、Infinity ⟨verify⟩ [arXiv:2412.04431]、MaskGIT [arXiv:2202.04200]、MUSE [arXiv:2301.00704]、Emu3 [arXiv:2409.18869]、Imagen [arXiv:2205.11487]、SDXL ⟨verify⟩ [arXiv:2307.01952]

### Part 8 · Multimodal LLMs（08-multimodal-llm，5 章）

#### 8.1 `01-vlm-architecture-taxonomy.md` — VLM 架构分类学
- **内容**：形式化：VLM = 建模 p(text | image, text_prompt)（+生成方向）；组件链：vision encoder（冻结/微调）→ connector → LLM；**三个接入点公式**：(a) input-level：visual tokens 拼进 LLM 输入序列（LLaVA 范式，主流）；(b) cross-attention-level：Flamingo gated xattn h ← h + tanh(α)·Attn(h, z_img)（α zero-init：不破坏预训练 LLM 的机制）；(c) 混合（Q-Former 输出 tokens 进 input-level）；**token 预算账**：CLIP-336/14 → 576 tokens；AnyRes 4 切 → 2880；高分辨率/视频帧数的账——context 压力的核心矛盾（8.2 压缩技术伏笔）；冻结策略与表征分析：LLM embedding 空间缺什么视觉特征（语义 vs 像素抽象层级——各家 ablation 证据）；参数规模组合表（vision 0.4B × LLM 8B 等真实组合）
- **锚点**：haotian-liu/LLaVA [A]、mlfoundations/open_flamingo [A]、Qwen2.5-VL [A]
- **Sources**：LLaVA [arXiv:2304.08485]、Flamingo [arXiv:2204.14198]、Qwen2.5-VL [arXiv:2502.13923]

#### 8.2 `02-connectors-and-resolution.md` — Connector 与分辨率处理
- **内容**：**projector 家族**：MLP（2 层 GELU——LLaVA-1.5 的数据 ablation：为什么小数据下 Q-Former 不划算）；**Q-Former/BLIP-2**（B 个 learnable queries + self-attn 文本可见性 mask + cross-attn 到冻结 ViT 输出 + ITC/ITM/ITG 三目标公式、两阶段训练）；Perceiver resampler（Flamingo：固定输出长度公式）；**token 压缩**：pixel shuffle / space-to-depth（2×2 相邻 token → 1 token × 4 通道公式，InternVL/Qwen 用法）、token merging（相似度合并 outline）；**分辨率处理**：固定分辨率的信息损失账（小字识别）；**AnyRes tiling 数学**（LLaVA-NeXT：网格选择规则 + 缩略图拼接 + 位置编码处理公式）；**Qwen2-VL naive dynamic resolution**：任意 H×W → 动态 token 数；**M-RoPE 完整公式**：position id 分解为 (temporal, height, width) 三元组——文本三值相同、图像 t 恒定 + (h,w) 网格、视频 t 递增——旋转作用在各频段（公式写出；视频应用 → 8.4/9.2）；Pixtral native resolution（2D RoPE + 免 tiling）；connector 选型决策表（token 数/灵活性/训练成本）
- **关键推导**：Q-Former 三目标；pixel shuffle；M-RoPE 三元组分解
- **锚点**：Qwen2.5-VL 官方 modeling 代码（M-RoPE 实现摘录）[A]、OpenGVLab/InternVL（pixel shuffle）[A]、salesforce/LAVIS（Q-Former）[A]
- **Sources**：Qwen2-VL [arXiv:2409.12191]、Qwen2.5-VL [arXiv:2502.13923]、BLIP-2 [arXiv:2301.12597]、LLaVA-OneVision [arXiv:2408.03326]、Pixtral [arXiv:2410.07073]、InternVL [arXiv:2312.14238]

#### 8.3 `03-training-objectives.md` — VLM 训练目标与数据形式
- **内容**：**loss masking 公式** $L=-\sum_{t\in response}\log p(x_t|x_{<t},img)$（mask 掉 prompt/图像 token——代码走读）；**多阶段配方**（各家 report 公约数）：stage1 alignment（只训 projector、frozen ViT+LLM、caption 数据——学什么：映射到 LLM embedding 空间）→ stage2 multi-task/instruction（解冻哪些、数据配比表）→ stage3 偏好对齐（DPO/RLHF 一句话交叉链接 training + 幻觉专项：对比偏好数据构造）；interleaved 图文格式（`<image>` 占位符展开机制、多图位置编码排布）；**幻觉形式化**：定义（生成图像不支持的内容）、机制假设（language prior 压过视觉证据——p(text|img) vs p(text) 的 KL 分析）、缓解（数据侧 + 解码侧 contrastive decoding：$\tilde p=\mathrm{softmax}((1+\alpha)\,\mathrm{logit}_{img}-\alpha\,\mathrm{logit}_{noise})$ ⟨verify⟩）；**评测方法论**：benchmark 类型学（感知/OCR/推理/幻觉率）、多选题局限（选项偏置、猜测基线）、claimed vs 真实能力
- **锚点**：LLaVA-OneVision（训练配方全开放）[A]、Qwen2.5-VL [A]、Molmo（数据全开放）[A]
- **Sources**：LLaVA-OV [arXiv:2408.03326]、Qwen2.5-VL [arXiv:2502.13923]、Molmo [arXiv:2409.17146]、VCD ⟨verify⟩ [arXiv:2311.16922]

#### 8.4 `04-video-understanding.md` — 视频理解
- **内容**：视频 = 图像 × 时间的账：帧数 F × 每帧 tokens N——总预算表（1min@1fps×256 = 15K tokens 级别数字）；**帧采样**：uniform vs dynamic FPS（Qwen2.5-VL：绝对时间编码进 M-RoPE t 轴公式——对事件定位的意义）；**时间压缩**：3D patch（temporal patch=2 相邻帧合并公式）、时空 pooling、temporal queries resampler；流式/长视频：chunk + 循环 memory（recurrent memory 公式 outline ⟨verify⟩）；输出格式：temporal grounding（时间戳 special token 化方案）；视频-图像训练数据混合（ablation 要点）；视频评测方法论（长视频检索/事件计数）
- **锚点**：Qwen2.5-VL（dynamic FPS + 绝对时间官方实现）[A]、LLaVA-OneVision [A]、Apollo ⟨verify⟩（视频 LMM 设计空间 ablation 研究）[A]
- **Sources**：[arXiv:2502.13923]、[arXiv:2408.03326]、Apollo [arXiv:2412.10360]

#### 8.5 `05-unified-understanding-generation.md` — 统一理解与生成
- **内容**：问题定义：同一模型做理解（image→text）与生成（text→image）；**路线 1 离散 token**：VQ tokenizer + 全 AR（Emu3：图像=语言的公式；质量 vs VQ 信息瓶颈的账）；**Chameleon**：early-fusion token 级混合（图文同词表同序列）；**稳定性技术全套**（重要架构教训）：QK-norm、norm 重排、z-loss、token dropout ⟨verify⟩——混合模态训练发散问题与 ablation 证据；**Janus/Janus-Pro**：解耦理解（SigLIP）与生成（VQ）双 encoder——表征冲突论证（两任务需要不同抽象层级的公式级证据）；Show-o/Show-o2：AR + 离散 diffusion 混合目标公式；**路线 2 连续 latent**：**Transfusion 完整公式**：同样本内 text tokens AR loss + image patches diffusion loss（$L=L_{LM}+\lambda L_{DDPM}$，无 U-Net、同一 Transformer）；attention mask = text causal + image 块内双向（mask 矩阵图）；BOI/EOI special tokens；沿用 Llama 式 1D RoPE（原文无 2D RoPE，勿杜撰）；**路线 3 diffusion head**：LLM hidden state → 小型 diffusion/FM head 解码（SEED-X ⟨verify⟩ / Emu2 式）；GPT-4o 生图 [C]：已披露信息分析（AR+diffusion 后级的推断证据）；**对照总表**：离散 vs 连续 vs diffusion head（保真/编辑/训练稳定/推理成本/与 LLM scaling 兼容）
- **锚点**：deepseek-ai/Janus [A]、showlab/show-o [A]、baaivision/Emu3 [A]、ByteDance-Seed/Bagel ⟨verify⟩ [A]、transformers chameleon modeling [B：有 paper、权重撤回]
- **Sources**：Transfusion [arXiv:2408.11039]、Chameleon [arXiv:2405.09818]、Janus [arXiv:2410.13848]、Janus-Pro ⟨verify⟩ [arXiv:2501.17811]、Emu3 [arXiv:2409.18869]、Show-o ⟨verify⟩ [arXiv:2408.12528]、Bagel ⟨verify⟩ [arXiv:2505.14683]

### Part 9 · Video Generation（09-video-generation，4 章）

#### 9.1 `01-video-vae.md` — Video VAE
- **内容**：视频数据账：T×H×W×3 raw vs latent——压缩率预算（token 数直接决定 DiT 成本）；**3D causal conv VAE**：时间维 causal 结构公式（conv3d time kernel 非对称 padding 图）；为什么 causal：图像 = 单帧视频兼容（首帧独立编码——图视频混训必需）；典型配置：spatial 8×8、temporal 4×（token 账：81 帧→21 latent 帧）；各家 VAE 对照：HunyuanVideo / CogVideoX / Wan（高压缩 ⟨verify⟩）/ LTX（更高压缩比换实时）；DC-AE 式深压缩在视频上的 tradeoff（运动细节 vs token 数）；重建质量评估（rFID/FVD 方法论一句话）；实例内存账：720p×5s×24fps 全链路数字表（raw→latent→attention 矩阵）
- **锚点**：zai-org/CogVideo（原 THUDM，VAE 官方代码）[A]、Tencent-Hunyuan/HunyuanVideo [A]、Wan-Video/Wan2.1 [A]
- **Sources**：CogVideoX [arXiv:2408.06072]、HunyuanVideo [arXiv:2412.03603]、Wan [arXiv:2503.20314]

#### 9.2 `02-spatiotemporal-dit.md` — 时空 DiT
- **内容**：spacetime patchify：latent (t,h,w,c) → 3D patches（token 化公式——Sora report 的 formulation [C]）；**attention 拓扑账**：full 3D O((T·H·W)²) vs factorized spatial O(T·(HW)²)+temporal O(HW·T²)（FLOPs 对比表 + 质量差异证据）；**CogVideoX**：expert-adaLN（文本/图像 expert 分开调制参数公式）、3D full attention；**HunyuanVideo**：dual-stream→single-stream MMDiT 变体结构走读、text encoder = MLLM 的原因、token refiner ⟨verify⟩；**Wan**：DiT+FM+cross-attn 走读；**Wan2.2 SNR-split MoE**：按 timestep SNR 确定性路由 high-noise expert（布局/运动）vs low-noise expert（细节）——公式（SNR(t) 阈值路由；对比 P4.5 token 级路由：这是 timestep 级 MoE）、27B/14B active 账；text conditioning 对照表（cross-attn vs joint stream vs in-context）；**时空位置编码**：3D/分解式 RoPE（t/h/w 三轴，与 P8.2 M-RoPE 数学同构）vs learned absolute 3D——各家选择对照；分辨率/时长 bucketing；其他开放系统一段式：StepVideo-T2V（30B 账 ⟨verify⟩）、Open-Sora 2.0 ⟨verify⟩、LTX-Video ⟨verify⟩、Mochi [C 无论文]
- **关键推导**：full vs factorized FLOPs；SNR 路由
- **锚点**：CogVideoX / HunyuanVideo / Wan2.2 官方 repo 代码走读 [A]
- **Sources**：[arXiv:2408.06072]、[arXiv:2412.03603]、[arXiv:2503.20314]、Wan2.2 README/blog [A−]、StepVideo ⟨verify⟩ [arXiv:2502.10248]、LTX-Video ⟨verify⟩ [arXiv:2501.00103]

#### 9.3 `03-training-and-acceleration.md` — 训练配方与加速/流式生成
- **内容**：训练配方（模型层）：图像-视频联合训练（T2I 权重作 init 的原因）、渐进分辨率/时长 schedule、timestep 采样（FM logit-normal recap → 7.5）；视频 CFG 变体公式 ⟨verify⟩；**step distillation 在视频**：DMD2 视频版；**CausVid**：双向 teacher → 4-step causal student 的非对称蒸馏公式（DMD 梯度 + ODE 轨迹初始化；注意 CausVid 无 GAN loss——GAN 分量属 DMD2，勿混淆）、KV cache 流式推理——**视频生成与 LLM 推理数学的合流点**（prefill/decode 概念直接复用）；**Diffusion Forcing**：per-token 独立噪声水平（t 逐 token 而非全局——训练目标改写公式；统一 teacher forcing 与 diffusion 的框架）；**Self-Forcing**：训练时自 rollout + 分布匹配（exposure bias 的数学表述与消除）；2026 后续：Causal-Forcing、rCM（现状表）；长视频三路线：keyframe+插值 / 分块 AR（历史条件公式）/ context 扩展账；世界模型/交互式生成一句话展望（标注成熟度）
- **关键推导**：Diffusion Forcing 目标；Self-Forcing exposure bias
- **锚点**：tianweiy/CausVid [A]、guandeh17/Self-Forcing [A]、thu-ml/Causal-Forcing [A]、NVlabs/rcm [A]、diffusion-forcing 官方 repo ⟨verify⟩ [A]
- **Sources**：Diffusion Forcing [arXiv:2407.01392]、CausVid [arXiv:2412.07772]、Self-Forcing [arXiv:2506.08009]、Causal-Forcing ⟨verify⟩、rCM ⟨verify⟩

#### 9.4 `04-control-and-editing.md` — 控制与编辑
- **内容**：**I2V**：条件注入公式家族：latent concat（首帧 latent + mask 通道）、noise augmentation（首帧加噪防复制粘贴）、CLIP image embedding cross-attn——Wan/Hunyuan 变体代码摘录；首尾帧控制（kf2v）；**ControlNet-for-DiT**：可训练复制分支 + zero-conv 公式（零初始化 = 起点即原模型——与 adaLN-Zero 同思想）；**相机控制**：**Plücker ray embedding**：射线 (d, o×d) ∈ R^6 per pixel（从相机内外参构造 Plücker map 的公式、作为条件通道注入）、CameraCtrl/MotionCtrl 走读 ⟨verify⟩；轨迹条件（drag point embedding 公式）；**视频编辑**：DDIM inversion 推广到视频（inversion ODE 公式、误差累积）、null-text inversion ⟨verify⟩；subject 一致性：**LoRA ΔW=(α/r)BA 低秩公式** + DreamBooth 微调在视频上的形态；多镜头/叙事结构（keyframe 链公式级）
- **锚点**：Wan2.2-I2V 官方代码 [A]、lllyasviel/ControlNet [A]、CameraCtrl ⟨verify⟩ [A]
- **Sources**：ControlNet [arXiv:2302.05543]、LoRA [arXiv:2106.09685]、DreamBooth [arXiv:2208.12242]、null-text inversion ⟨verify⟩ [arXiv:2211.09794]、CameraCtrl ⟨verify⟩

### Part 10 · Audio & Speech（10-audio-speech，4 章）

#### 10.1 `01-audio-representations.md` — 音频表征：特征与 Neural Codec
- **内容**：数字音频基础：采样定理（Nyquist 公式）、16k/24k/44.1k 账；**STFT 完整公式** $X[m,k]=\sum_n x[n]w[n-mH]e^{-j2\pi kn/N}$（窗函数、hop/overlap、时频分辨率不确定性）；**mel filterbank**：mel 刻度定义（HTK/Slaney 公式）、三角滤波器组矩阵、log 压缩（Weber-Fechner）；特征账：16kHz/25ms 窗/10ms hop → 100fps → 序列长度与文本对比表；**semantic tokens**：自监督语音模型——HuBERT（离线 k-means 标签 + masked prediction 迭代公式）、w2v-BERT（对比+masked 两阶段）；content vs speaker 分离（FACodec 视角）；**acoustic tokens（neural codec）**：SoundStream/Encodec 架构：conv encoder（stride 堆叠下采样账：24kHz→75Hz）+ **RVQ（recap P6.3 公式直接复用；bitrate = n_q·log2K·fps 实例账：8×log2(1024)×75≈6kbps）** + conv decoder + multi-scale STFT discriminator 公式；DAC 改进（factorized codes ⟨verify⟩）；**Mimi**：split-RVQ + **第一 quantizer 语义蒸馏**（distillation loss 公式）、12.5Hz 低帧率——Moshi 基石；WavTokenizer（单 codebook 极压缩 ⟨verify⟩）；XCodec2（语义-声学统一单 codebook ⟨verify⟩）；**semantic vs acoustic 对照表**（内容保真/音色/bitrate/LM 建模难度）；离散 token vs 连续特征进 LM 的分析（10.4 伏笔）
- **关键推导**：STFT/mel；RVQ bitrate 账；Mimi split-RVQ
- **锚点**：facebookresearch/encodec [A]、kyutai-labs/moshi（Mimi 代码）[A]、descriptinc/descript-audio-codec [A]、fairseq（HuBERT）[A]
- **Sources**：SoundStream [arXiv:2107.03312]、Encodec [arXiv:2210.13438]、DAC [arXiv:2306.06546]、Moshi/Mimi [arXiv:2410.00037]、HuBERT [arXiv:2106.07447]、w2v-BERT [arXiv:2108.06209]、WavTokenizer ⟨verify⟩ [arXiv:2408.16532]、XCodec2 ⟨verify⟩ [arXiv:2502.04128]

#### 10.2 `02-asr.md` — 语音识别
- **内容**：ASR 形式化：p(y|x)，x=声学特征（100fps）、y=文本（~10fps）——长度不匹配问题；**CTC 完整数学**：条件独立假设、对齐路径集合 $\mathcal{B}^{-1}(y)$、边缘化 $P(y|x)=\sum_{\pi\in\mathcal{B}^{-1}(y)}P(\pi|x)$；**forward-backward 递推公式**（α_t/β_t 定义与 DP 推导）；blank 的边界语义；CTC peaky 行为与 LM 能力弱分析；CTC+attention hybrid（$L=\lambda L_{CTC}+(1-\lambda)L_{att}$，ESPnet）；**RNN-T/Transducer**：2D lattice（时间×标签）forward 算法公式、joint network；对照表（CTC/RNN-T/AED：流式/成本/精度）；**Whisper 走读**：log-mel-80（recap 10.1）、30s 窗 padding 账、decoder prefix = task/language special tokens（控制机制公式级）、680k hrs 弱监督一句话（交叉链接 training）、鲁棒性来源分析；流式约束：chunk attention 公式（lookahead 账）；现代开放锚点：SenseVoice（非 AR 多任务 ⟨verify⟩）、FireRedASR（AED+LLM hybrid ⟨verify⟩）、Qwen3-ASR ⟨verify⟩；**LLM-based ASR**：audio encoder → projector → LLM（与 VLM connector 同构 recap → P8.1）
- **关键推导**：CTC 边缘化与 forward-backward；RNN-T lattice
- **锚点**：openai/whisper [A]、espnet/espnet（CTC/RNN-T 标准实现）[A]、QwenAudio/SenseVoice（原 FunAudioLLM）[A]
- **Sources**：Whisper [arXiv:2212.04356]、CTC（Graves et al., ICML 2006，无 arXiv）、RNN-T [arXiv:1211.3711]、SenseVoice（FunAudioLLM 报告）[arXiv:2407.04051]、FireRedASR ⟨verify⟩ [arXiv:2501.14350]

#### 10.3 `03-tts.md` — 语音合成
- **内容**：TTS 形式化：p(audio|text)——一对多映射（风格/韵律 latent）；**级联管线**：G2P、duration model（MFA forced alignment）、acoustic model（mel）、vocoder；**HiFi-GAN**：generator（上采样栈 + MRF）+ **multi-period discriminator（质数周期 2D reshape 公式）** + multi-scale discriminator + feature matching loss——对抗 loss 就地讲处之二；**BigVGAN**：snake 激活 $x+\sin^2(ax)/a$ 的抗混叠公式（现代默认 vocoder）；**VITS**：VAE + normalizing flow（后验灵活化公式）+ 对抗 + MAS 单调对齐搜索——端到端统一目标走读；FastSpeech2（variance adapter：duration/pitch/energy 公式）；**LM-based TTS**：**VALL-E**：acoustic tokens as a foreign language——AR 建 RVQ 第一层 + NAR 逐层建其余层（两阶段公式）；in-context TTS：3s prompt = 条件（zero-shot cloning 机制）；VALL-E 2（repetition-aware sampling、grouped code ⟨verify⟩）[B 无官方代码]；**CosyVoice**：监督语义 token（S3Tokenizer：ASR encoder + 量化公式）+ LM + **FM decoder（semantic→mel：conditional FM recap → 7.5）**；chunk-aware 流式（causal chunk 公式）；CosyVoice3（scale-up ⟨verify⟩）；**F5-TTS/E2-TTS**：免 duration model：text infilling formulation（pad 到目标长度 + masked infilling）、**CFM 目标 + DiT + Sway sampling 公式走读**；VoiceBox [B]：任意 mask FM infilling 先驱；NaturalSpeech-3 FACodec（content/prosody/timbre/detail 子流分解公式）；MaskGCT（mask-predict 进 codec tokens——recap 7.7）；2025 批一段式：IndexTTS / MegaTTS3 / Spark-TTS ⟨verify⟩；**三路线对照表**：级联 vs LM vs FM/diffusion（延迟/克隆/可控性/音质证据）
- **关键推导**：VALL-E 两阶段；CosyVoice FM decoder；F5 infilling
- **锚点**：QwenAudio/CosyVoice（原 FunAudioLLM）[A]、SWivid/F5-TTS [A]、jaywalnut310/vits [A]、jik876/hifi-gan [A]
- **Sources**：VALL-E [arXiv:2301.02111]、VALL-E 2 ⟨verify⟩ [arXiv:2406.05370]、CosyVoice [arXiv:2407.05407]、CosyVoice 2 [arXiv:2412.10117]、CosyVoice 3 [arXiv:2505.17589]、F5-TTS [arXiv:2410.06885]、E2-TTS ⟨verify⟩ [arXiv:2406.18009]、VoiceBox [arXiv:2306.15687]、NS3 [arXiv:2403.03100]、MaskGCT ⟨verify⟩ [arXiv:2409.00750]、HiFi-GAN [arXiv:2010.05646]、VITS [arXiv:2106.06103]、FastSpeech2 [arXiv:2006.04558]

#### 10.4 `04-speech-llm-and-duplex.md` — Speech LLM 与全双工
- **内容**：**speech 理解 LLM**：audio encoder（Whisper 式）+ adapter + LLM（Qwen2-Audio/Qwen2.5-Omni Thinker 公式）；音频-文本时间对齐（interleaved chunk 公式）；**Qwen2.5/3-Omni Thinker-Talker**：Thinker（理解+文本生成）→ Talker（流式语音 token：接 Thinker hidden 而非最终文本——低延迟机制公式级分析）、**TMRoPE**（绝对时间轴对齐音/视/文——M-RoPE 的扩展公式，recap → 8.2）；Qwen3-Omni 的 Code2Wav 流式 vocoder（chunk 账；Qwen2.5-Omni 为 block-wise DiT 方案，勿混淆）；**Moshi 全公式**（本章深拆范例）：**multi-stream**（用户流+自身流并行建模——无 turn-taking 假设的 full-duplex 数学）、**Inner Monologue**（文本 token 流对齐前置：联合目标公式、为什么文本前缀提升语音质量）、**RQ-Transformer**（时间 Transformer 12.5Hz 帧级 + depth Transformer RVQ 层级内自回归——两级公式与参数账）；GLM-4-Voice：单 codebook 12.5Hz + 文本/语音交错 token 公式（interleave pattern、流式）；AudioLM：semantic→coarse acoustic→fine acoustic 分层生成公式（token 层级 recap）；Kimi-Audio / Step-Audio / LLaMA-Omni ⟨verify⟩ 一段式（audio 表征选择 + 流式设计）；**full-duplex 数学**：always-listen/always-talk（输出分布含 silence token）、延迟预算账表（mic→VAD→LM→codec→speaker ~200ms 逐项）；**离散 token vs 连续特征进 LLM**：adapter 选择分析（token 信息损失 vs embedding 对齐难度——各家 ablation）；GPT-4o voice [C]：已披露信息分析
- **关键推导**：RQ-Transformer 两级分解；multi-stream 联合分布；TMRoPE
- **锚点**：kyutai-labs/moshi（官方全栈）[A]、QwenLM/Qwen3-Omni [A]、zai-org/GLM-4-Voice（原 THUDM）[A]、MoonshotAI/Kimi-Audio ⟨verify⟩ [A]
- **Sources**：Moshi [arXiv:2410.00037]、Qwen2.5-Omni [arXiv:2503.20215]、Qwen3-Omni [arXiv:2509.17765]、GLM-4-Voice ⟨verify⟩ [arXiv:2412.02612]、AudioLM [arXiv:2209.03143]、Kimi-Audio ⟨verify⟩ [arXiv:2504.18425]、Step-Audio ⟨verify⟩ [arXiv:2502.11946]

---

## 6. 生产流程与质量门（每波一个循环）

### 6.1 Research pass（写前，每波必做）
1. 列出该波全部章节涉及的 claims / 论文 / 模型 / config 数值
2. WebSearch + WebFetch 核实：arXiv ID 与标题逐一对上（**⟨verify⟩ 项全部解决**）；2025-2026 内容以调研结果为准，不凭记忆
3. 拉取 primary sources：arXiv 原文（abs/html）、HF config.json、GitHub modeling 文件路径核实存在
4. source list 写入 `models/references.md`（按 Part 分组，含证据等级与核实日期）

### 6.2 Write pass
- 逐章写作，严格遵守 STYLE_GUIDE.md 与章节模板
- 公式推导与论文原文对照；教学代码片段自行验证可运行（片段级，非项目级）
- 每章完成即自查一遍去 AI 味 checklist

### 6.3 Adversarial review（用户硬要求）
- 每波写作完成后，起**独立 fresh-context evaluator agent**（不继承写作上下文）
- 输入：该波章节文件 + STYLE_GUIDE.md + references.md 对应 source list + 本 spec 对应章节要求
- 三项任务：
  1. **事实核查**：抽查所有公式、数值、日期、arXiv ID、repo 路径、attribution；要求 evaluator 自己 WebFetch sources 对照；**独立重新推导数学重章的关键公式**（**强制清单**：1.3 Chinchilla Lagrangian / 2.3 attention √d_k / 2.4 KV cache / 2.5 RoPE / 2.7 Pre-Post-LN 梯度 / 4.1 MLA / 4.3 DeltaNet-GDN-KDA / 4.5 MoE routing / 6.2 InfoNCE / 6.3 ELBO / 7.2 DDPM / 7.4 CFG / 7.5 FM / 10.2 CTC——每波按覆盖章节指定子集）
  2. **教学审查**：推导断步、未定义符号、循环解释、前置知识缺口（对照 Part 依赖图）
  3. **规范审查**：STYLE_GUIDE 逐条对照——重点抓历史叙事违规、AI 味、术语翻译、证据等级缺失、练习/总结缺失
- severity 判定规则：S1 = 事实性致命（张冠李戴的引用、错误公式、不存在的 repo/模型、虚构数值）；S2 = 实质问题（约定冲突、推导断裂、协议缺口、局部事实错误）；S3 = minor（措辞、格式、可选增补）
- 输出：findings 表（severity + 位置 + 断言 + 证据 + 建议修正）**+ verification log（抽查项 × 来源 URL 或原文引句 × 结论 × 抽查覆盖率）——无 log 的「确认无误」一律视为未审查（防 rubber-stamp）**

### 6.4 Fix + re-review
- S1/S2 必须修复；S3 累计，下一波复审时统一清零（逐条判定：修复 / 接受 / 驳回）
- **writer 反驳通道**：writer 对某条 finding 持异议时可附证据（来源引文）提交 evaluator 复核；复核仍维持才强制修复——防 evaluator 误报造成错误改动
- 含 S1 findings 的章节修复后由 evaluator 复审该章
- 审查结论（轮次、findings 数、关闭状态）记录到 `models/README.md` 进度表

### 6.5 Commit gate
- 一波通过审查才 git commit（commit message 注明波次与章节）
- `models/README.md` 进度表更新后随同提交

## 7. 交付波次

| 波次 | 内容 | 章数 | 备注 |
|---|---|---|---|
| W0 | 脚手架：根 README 重写、STYLE_GUIDE.md、models/README.md（含进度表与依赖图）、references.md 骨架、四个兄弟维度占位 README | — | 一次性 |
| W1 | P1 Foundations + P2 Transformer Anatomy | 11 | 全书基石；完成后做 GitHub 公式渲染测试 |
| W2 | P3 Objectives + P4 Modern Architectures | 11 | 文本主干推导密度最高 |
| W3 | P5 Case Studies + P6 Vision Representations | 9 | P5 需最新调研（K3/GLM-4.7/Qwen3-Max） |
| W4 | P7 Diffusion & Image Generation | 7 | 推导量最大 |
| W5 | P8 Multimodal LLM | 5 | |
| W6 | P9 Video Generation | 4 | 领域迭代快，research pass 权重最高 |
| W7 | P10 Audio & Speech | 4 | |

每波可跨多个 session；`models/README.md` 进度表为唯一进度事实源（章节 × 状态：未开始/已写/审查中/已关闭）。

## 8. 风险与对策

1. **领域快速演化**（2025-2026 新模型频出）：每波 research pass 强制核实最新动态；references.md 记录核实日期；正文避免「最新/首个」类时效性断言，除非标注日期
2. **arXiv ID / repo 路径记错**：⟨verify⟩ 标记制度——未经 research pass 核实的 ID 不得写入正文；evaluator 事实核查项包含 ID 有效性
3. **上下文限制**（51 章无法单 session 完成）：波次制 + 进度表 + 每章自包含（前置知识用链接而非重复）
4. **evaluator 与写作者同源偏差**：fresh context + 只给成品与 sources + 要求自行 fetch 与独立重推导，禁止信任正文自述
5. **GitHub 公式渲染兼容**：W1 首章完成后立即渲染测试（$$ 环境、align/cases、矩阵）；STYLE_GUIDE 固化可用语法子集
6. **版权**：只引用与链接，不复制论文图表；ASCII 图全部自绘并注明出处

## 9. Spec 维护

- 本 spec 是唯一设计事实源；后续波次若需调整大纲（如新模型取代旧锚点），先改 spec 并注明日期，再执行
- 每波结束时在 spec 末尾追加一行「波次完成记录」（日期 + 波次 + 偏差说明）

### 波次完成记录

- 2026-11-11 W0+W1 完成：11 章（P1 三章 + P2 八章）+ W0 脚手架（STYLE_GUIDE / references.md / lint 脚本 / math-render-probe）；审查为逐章 review + 波末独立对抗审查；S1 2 起 / S2 2 起全部关闭，S3 5 起于波末 sweep 清零；强制重推导 5 项（1.3 Chinchilla Lagrangian、2.3 √d_k、2.4 KV cache、2.5 RoPE、2.7 Pre-Post-LN 梯度）全部 pass；无大纲偏差。审查档案：docs/reviews/w1-review-report.md
- 2026-11-12 W2 完成：11 章（P3 三章 + P4 八章）；审查为逐章批量审查 3 批 + 波末独立对抗审查 1 轮；S1 0 起 / S2 2 起全部关闭（ModernBERT 交错比 3:1→2:1 跨章传播、README 进度表未随波更新），S3 9 起于波末 sweep 清零；强制重推导 3 项（4.1 MLA absorption、4.3 DeltaNet-GDN-KDA 状态更新链、4.5 MoE routing）全部 pass；无大纲偏差。篇幅口径裁定：STYLE_GUIDE §8 由「总字符数」改为「正文 CJK 汉字数」（标准章 ≥3500 / 推导重章 ≥5500 软目标 / 短篇 ≥2500）。审查档案：docs/reviews/w2-review-report.md

---

## 附录：Spec 对抗审查记录（2026-11-10）

四个独立 fresh-context evaluator 并行审查本 spec，全部 findings 已修复：

**A（P1-P5 事实核查）**：80 个 arXiv ID 77 个正确；41 项模型字段断言 39 项正确；22 个 repo 20 个存在。
- S1×2：Muon 引用 ID 张冠李戴（2410.19753 实为无关论文 → 改 Moonlight [arXiv:2502.16982]，原始出处 Keller Jordan 博客）；`deepseek-ai/NSA` repo 不存在（NSA 降 [B]，仅以论文为准）
- S2×5：Pythia checkpoint 数 16→154（两处）；expert choice ID 2202.08906→2202.09368（2202.08906 实为 ST-MoE/z-loss 出处，已拆分）；`jbloomX/SAELens`→`decoderesearch/SAELens`；Gemma 3 弃用 soft-capping（与 D 的 S2-3 独立互证）；SED→SEDD [arXiv:2310.16834]
- S3×3：Mistral-7B 有正式 paper（[arXiv:2310.06825]，[W]→[A]）；Qwen3 阵容 dense 6 + MoE 2；Qwen3-Next 锚点补 -Instruct 后缀
- 顺手解决自留疑问：GLM-4.5 文本 report = [arXiv:2508.06471]；Kimi Linear `mla_use_nope=true` 与 Qwen3-Next 48 层=36+12 经 config 实测确认

**B（P6-P7 事实核查）**：42 个 arXiv ID 41 个正确；12/13 repo 存在。
- S1×2：Glow ID 1811.00154→1807.03039（原 ID 为天体物理手册）；`google-research/muse` 不存在（MUSE 未开源，撤锚点降 [B]）
- S2×1：DC-AE 实际位于 `mit-han-lab/efficientvit`（applications/dc_ae）
- 确认：SD3 = MMDiT + rectified flow + logit-normal 采样；VAR next-scale；SANA = Linear DiT + DC-AE f32；SD-VAE 0.18215 / SD3 VAE 16ch；CLIP-336→576 tokens

**C（P8-P10 事实核查）**：61 个 arXiv ID 58 个正确；29 个 repo 28 个存在。
- S1×3：Pixtral 2410.05258→2410.07073（原 ID 为 Differential Transformer——该 ID 转用于 4.8 differential attention）；Apollo 2412.05271→2412.10360（原 ID 为 InternVL 2.5）；SenseVoice 2405.02132→2407.04051（FunAudioLLM 报告）
- S2×2：Transfusion「图内 2D RoPE」系杜撰（原文仅 1D RoPE，已删）；CausVid 无 GAN loss（GAN 分量属 DMD2，已纠正）
- S3×3：BOI/BOE→BOI/EOI；Code2Wav 归属 Qwen3-Omni（非 2.5）；组织更名 THUDM→zai-org、FunAudioLLM→QwenAudio
- 确认：Moshi/Mimi 全链路（12.5Hz、split-RVQ、双 stream、RQ-Transformer）、Encodec 75Hz/6kbps 账、HunyuanVideo dual→single、M-RoPE 三元组、Chameleon 稳定性技术、Transfusion 主干

**D（设计合理性审查）**：0 S1；章数账/依赖拓扑/波次/流水线骨架全部成立。
- S2×6：GDN/DeltaNet 状态约定统一为 $S\in\mathbb{R}^{d_v\times d_k}$（写入项 vkᵀ、擦除右乘，同 GDN 论文）；rectified flow 全书约定 x₀=数据、t=1=噪声（对齐 SD3，附 Lipman 映射框）；Gemma 3 锚点纠错 + 4.2 增设 SWA:full 混合层小节；强制重推导清单补全（KV cache/Pre-Post-LN/CFG/InfoNCE/ELBO/Chinchilla）；evaluator 输出强制附 verification log（防 rubber-stamp）+ severity 判定规则；证据分级增设 [W] 档与「教学参照」标签（Mistral 后经 A 核实有 paper 回升 [A]）
- S3×11 全修；G 项主题增补：parallel attention/FFN block（2.8）、SWA:full 混合（4.2）、BigVGAN（10.3）、BLT 对照段 + softmax bottleneck（2.2）、Medusa 对照（4.5）、CFG 蒸馏（7.4）、MoD/differential attention/TTT（4.8）
- 10 项 spec 内公式断言抽查全过（√d_k、12Ld²、KV cache、MLA 576、DDPM 后验系数、rectified flow、Encodec 账、Xiong 方向、80/10/10；GDN α 作用域）
