# Training 维度 Spec 对抗审查 · Evaluator A 报告

- 日期：2026-11-16
- 审查对象：`docs/superpowers/specs/2026-11-16-training-dimension-design.md` §1-§4 + §5 Part 1-3（章节 1.1-3.2）
- 立场：fresh context，与 spec 作者无共享上下文；所有核实均自己动手（arXiv API / ar5iv 全文 / GitHub contents API / 独立推导），未使用 spec 附录预核实记录
- 范围说明：µP 数学声称（7.2）归 evaluator C；本报告只覆盖 2.3 的 LR×batch 内容

---

## 一、Findings 表

### S1（事实性致命）：无

全部 29 个抽查 arXiv ID 与标题一一对上（含 3 个 ⟨verify⟩ 项：1502.05767 / 1908.03265 / 2209.05433），全部 7 个 GitHub repo/路径存在，头条归因（Polyak / Sutskever / Williams / Bradley-Terry / label smoothing / ST-MoE / Muon / Moonlight / MuonClip）未发现致命错误。

### S2（实质）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S2-1 | §5 1.2 锚点、§5 3.1 锚点（两处） | 「linkedin/Liger-Kernel repo ⟨verify⟩…（**无独立论文**，按 §4.1 口径）」 | **Liger-Kernel 有正式论文**：arXiv:2410.10989「Liger Kernel: Efficient Triton Kernels for LLM Training」（Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song 等，LinkedIn 团队，即 repo 所属组织；摘要：「we introduce Liger-Kernel, an open-sourced set of Triton kernels…20% increase in training throughput and 60% reduction in GPU memory」）。arXiv API `id_list=2410.10989` 与全库搜索 `all:"Liger-Kernel"` 双向确认 | 两处锚点改为引用 arXiv:2410.10989 并按 §4.1 重分类为 [A]（正式 paper + 官方代码）；§4.1 教学参照示例清单不受影响（TRL 的「无独立论文」经核实成立，见 verification log） |
| S2-2 | §5 2.2 「超参结构与 LLM 惯例对照表」 | 「β1=0.9、β2=0.95、ε=10⁻⁸、wd=0.1 的各家出处：GPT-3 / **Llama-3** / DeepSeek-V3 / OLMo 2」 | **Llama-3 herd 论文（2407.21783）全文未披露 Adam β1/β2/ε**：ar5iv 全文检索 `beta` 仅 1 命中（DPO 的 KL 系数）、`0.95` 仅 1 命中（Prompt Guard AUC 表）、MathML alttext 中 `\beta` 仅 1 处。§3.4.1 只给 peak lr 8×10⁻⁵ / warmup 8000 / cosine 至 8×10⁻⁷ / batch 4M→8M→16M tokens。β2=0.95 的 Llama 谱系出处是 Llama-1/2 论文（2302.13971 / 2307.09288），不是 herd 论文。对照：GPT-3 ✓（Appendix B 原文「we use Adam with β1=0.9, β2=0.95, and ϵ=10⁻⁸」）、DeepSeek-V3 ✓（原文「β1=0.9, β2=0.95, and weight_decay=0.1」）、OLMo 2 论文正文亦未写 β，但官方 OLMo-core repo `src/scripts/train/OLMo2/OLMo2-7B.py` 有 `betas=(0.9, 0.95), weight_decay=0.1`（config 级证据） | 表格中 Llama-3 的 β/ε 字段标注「herd 论文未披露」并把谱系出处改引 Llama-2（2307.09288）；OLMo 2 的 β 出处注明为 OLMo-core config（[W]/repo 级）而非论文正文；2.2 Sources 补 2307.09288 |
| S2-3 | §5 1.3 「FP8 训练」bullet | 「E4M3（前向权重/激活）与 E5M2（梯度）的分工、per-tensor scaling → blockwise scaling（DeepSeek-V3…）；DeepSeek-V3 FP8 方案全拆（论文 §3.3 口径…）」 | **DeepSeek-V3 §3.3 明确放弃 E4M3/E5M2 混合分工**：原文「In contrast to the hybrid FP8 format adopted by prior work…which uses E4M3 in Fprop and E5M2 in Dgrad and Wgrad, **we adopt the E4M3 format on all tensors** for higher precision. We attribute the feasibility of this approach to our fine-grained quantization strategy, i.e., tile and block-wise scaling.」（另对 attention 相关激活用定制 E5M6）。即 E4M3/E5M2 分工是 NVIDIA TE 系前作（= 2209.05433 口径），DeepSeek-V3 恰恰是该分工的反例。bullet 把二者并置且以 DeepSeek-V3 为锚，「§3.3 口径全拆」与「E4M3/E5M2 分工」框架存在潜在矛盾 | 明确分层：E4M3/E5M2 混合分工 = 前作惯例（锚 2209.05433）；DeepSeek-V3 方案 = E4M3 全量 + 细粒度缩放使其可行（+ E5M6 定制）。128×128 权重块 / 1×128 激活 tile、在线量化、N_C=128 间隔提升 CUDA core FP32 累加，均经原文核实无误，可保留 |

### S3（minor）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S3-1 | §5 2.2 Sources | 内容 bullet 点名 GPT-3 出处，Sources 行无 GPT-3 ID | 2.2 Sources = [1412.6980, 1711.05101, 1904.09237, 2407.21783, 2412.19437, 2501.00656]，缺 GPT-3 论文 2005.14165（「Language Models are Few-Shot Learners」，Appendix B 含 β/ε/clip=1.0 原文，已核实） | Sources 补 [arXiv:2005.14165] |
| S3-2 | §5 3.1 「z-loss」 | 「z-loss（log Z 惩罚公式、抑制 logit 爆炸的机制…）」 | ST-MoE（2202.08906）Eq. (5) 为 **L_z = (1/B)Σᵢ (log Σⱼ e^{x_j})²——log Z 的平方**，且原文是 **router logits** 上的 z-loss，并自承「an adaptation of the z-loss used for final softmax logits in the Mesh Tensorflow codebase (Shazeer et al. 2018)」。3.1 章场景是最终 softmax logits（PaLM/OLMo 2 式用法）；「z-loss 梯度」列为关键推导，漏掉平方则梯度公式直接错 | 章内写全 (log Z)² 并注明谱系：Mesh TensorFlow（final logits 源头）→ ST-MoE（router 版、稳定化动机与标准引用点）；injection point 对照表不受影响 |
| S3-3 | §5 2.3 「critical batch size」 | 「B_noise = tr(Σ)/gᵀg」 | McCandlish et al. 1812.06162 原文有两式：**B_simple = tr(Σ)/|G|²**（Hessian-free 简化版）与 **B_noise = tr(HΣ)/GᵀHG**（完整版，alttext 原文核对）。spec 把简化公式挂了完整版的名字 | 公式保留，命名对齐论文：正文以 B_simple = tr(Σ)/gᵀg 为主、注明论文另定义 Hessian 加权版 B_noise |

---

## 二、Verification log（强制项）

### 2.1 arXiv ID ↔ 标题核对（arXiv API `id_list` 批量，2026-11-16）

| arXiv ID | spec 用途 | API 返回标题 | 结论 |
|---|---|---|---|
| 1502.05767 ⟨verify⟩ | 1.1 AD survey | Automatic differentiation in machine learning: a survey | ✓ |
| 1908.03265 ⟨verify⟩ | 2.3 RAdam | On the Variance of the Adaptive Learning Rate and Beyond | ✓ |
| 2209.05433 ⟨verify⟩ | 1.3 FP8 格式 | FP8 Formats for Deep Learning | ✓ |
| 2412.19437 | 1.3/2.2 DeepSeek-V3 | DeepSeek-V3 Technical Report | ✓ |
| 2502.16982 | 2.5 Moonlight | Muon is Scalable for LLM Training | ✓ |
| 1604.06174 | 1.2/1.4 Chen checkpointing | Training Deep Nets with Sublinear Memory Cost | ✓ |
| 1710.03740 | 1.3 Mixed Precision | Mixed Precision Training | ✓ |
| 2205.14135 | 1.2 FlashAttention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | ✓ |
| 2205.05198 | 1.4 Megatron SP/重计算 | Reducing Activation Recomputation in Large Transformer Models | ✓ |
| 1812.06162 | 1.5/2.1/2.3 critical batch | An Empirical Model of Large-Batch Training | ✓ |
| 1706.02677 | 2.1/2.3 linear scaling | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | ✓ |
| 1412.6980 | 2.2 Adam | Adam: A Method for Stochastic Optimization | ✓ |
| 1711.05101 | 2.2 AdamW | Decoupled Weight Decay Regularization | ✓ |
| 1904.09237 | 2.2 AMSGrad | On the Convergence of Adam and Beyond | ✓ |
| 2407.21783 | 2.2/2.3 Llama-3 | The Llama 3 Herd of Models | ✓ |
| 2501.00656 | 2.2/2.3 OLMo 2 | 2 OLMo 2 Furious | ✓ |
| 2404.06395 | 2.3 MiniCPM/WSD | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | ✓ |
| 2210.02414 | 2.4 GLM-130B | GLM-130B: An Open Bilingual Pre-trained Model | ✓ |
| 2202.08906 | 2.4/3.1 ST-MoE | ST-MoE: Designing Stable and Transferable Sparse Expert Models | ✓ |
| 2507.20534 | 2.5 Kimi K2 | Kimi K2: Open Agentic Intelligence | ✓ |
| 1512.00567 | 3.1 label smoothing | Rethinking the Inception Architecture for Computer Vision | ✓（label smoothing 即此文提出） |
| 1706.03741 | 3.2 Christiano RLHF | Deep reinforcement learning from human preferences | ✓ |
| 2305.18290 | 3.2 DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | ✓ |
| 2006.15704 | 1.1/1.5 DDP | PyTorch Distributed: Experiences on Accelerating Data Parallel Training | ✓ |
| 2409.19256 | §4.1 veRL 例 | HybridFlow: A Flexible and Efficient RLHF Framework | ✓ |
| 2405.11143 | §4.1 OpenRLHF 例 | OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework | ✓ |
| 2304.11277 | §4.1 FSDP 例 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | ✓ |
| 1910.02054 | §4.1 ZeRO 例 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | ✓ |
| 1909.08053 | §4.1 Megatron 例 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | ✓ |
| 2203.02155 | §4.1 InstructGPT 例 | Training language models to follow instructions with human feedback | ✓ |
| 2005.14165 | （补查 GPT-3） | Language Models are Few-Shot Learners | ✓（→ finding S3-1） |
| 2203.03466 | µP 边界确认 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | ✓（数学归 evaluator C） |
| 2402.03300 | （顺带）DeepSeekMath | DeepSeekMath: Pushing the Limits of Mathematical Reasoning | ✓ |

**覆盖率：范围内 22/22 全核 + 范围外 §4.1 示例 7 个 + 边界 2 个 = 31 查 31 中（其中 Liger 反向命中，见下）**

### 2.2 GitHub repo/路径核对（contents API，2026-11-16）

| 路径 | HTTP | 结论 |
|---|---|---|
| repos/linkedin/Liger-Kernel | 200 | ✓ 存在（但「无独立论文」断言错误 → S2-1） |
| repos/KellerJordan/Muon | 200 | ✓ |
| repos/MoonshotAI/Moonlight | 200 | ✓ |
| pytorch/pytorch torch/optim/adamw.py | 200 | ✓ |
| pytorch/pytorch torch/utils/checkpoint.py | 200 | ✓ |
| pytorch/pytorch torch/nn/parallel/distributed.py | 200 | ✓ |
| pytorch/pytorch torch/nn/utils/clip_grad.py | 200 | ✓（2.4 锚点 clip_grad_norm_ 所在文件） |
| huggingface/trl CITATION.cff | 200 | ✓ `type: software`——「TRL 无独立论文」成立；arXiv:2004.14279 为同名误撞（水动力学论文），排除 |

### 2.3 论文内容断言核对（ar5iv 全文 / repo 源码，逐条引句）

| # | spec 断言 | 一手来源证据 | 结论 |
|---|---|---|---|
| C1 | DeepSeek-V3：权重 128×128 块、激活 1×128 粒度 | 2412.19437 §3.3：「for activations, we group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and (2) for weights…on a 128x128 block basis」 | ✓ |
| C2 | E4M3/E5M2 分工 | 2412.19437 §3.3：「In contrast to the hybrid FP8 format adopted by prior work…which uses E4M3 in Fprop and E5M2 in Dgrad and Wgrad, we adopt the E4M3 format on all tensors」 | ✗ spec 框架矛盾 → S2-3 |
| C3 | FP8 GEMM fp32 累加 + 周期性提升 | 2412.19437：「Once an interval of N_C is reached, these partial results will be copied to FP32 registers on CUDA Cores, where full-precision FP32 accumulation is performed…setting N_C=128」 | ✓ |
| C4 | Muon NS 系数 (3.4445, −4.7750, 2.0315) | KellerJordan/Muon `muon.py` 源码：`a, b, c = (3.4445, -4.7750, 2.0315)`（quintic，bf16，默认 steps=5） | ✓ |
| C5 | Moonlight 约 2× 计算效率口径 | 2502.16982：「Scaling law experiments indicate that Muon achieves ~2× computational efficiency compared to AdamW with compute optimal training」 | ✓（口径 = scaling law 实验 + compute-optimal 对照） |
| C6 | Moonlight 尺度匹配 √max(m,n) 对齐 AdamW RMS | 2502.16982：更新式 W_t = W_{t-1} − η(0.2·O_t·√max(A,B) + λW)；「AdamW's update RMS is usually around 0.2 to 0.4」；并注明 Keller Jordan 原版 √max(1,A/B) 与之等价（同第二维时至多差全局尺度） | ✓ |
| C7 | Muon/AdamW 分工（隐藏矩阵 vs embedding/head/norm/scalar） | Muon repo docstring：「Muon should only be used for hidden weight layers. The input embedding, final output layer, and any internal gains or biases should be optimized using a standard method such as AdamW」 | ✓ |
| C8 | MuonClip = Muon + QK-clip（K2） | 2507.20534 abstract：「We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique…pre-trained on 15.5 trillion tokens with zero loss spike」 | ✓ |
| C9 | DDP no_sync 在论文 §3 | 2006.15704 §3 含 `with ddp.no_sync():` 代码例与语义说明 | ✓ |
| C10 | Chen √L：内存 O(√L)、一次额外前向 | 1604.06174 abstract：「O(sqrt(n)) memory to train a n layer network, with only the computational cost of an extra forward pass per mini-batch」 | ✓ |
| C11 | Megatron selective recompute「重算量 <10%」 | 2205.05198：full recompute「30-40% execution time overhead」；本方法「reducing execution time overhead from activation recomputation by over 90%」（残余 ~2-4%）+「reduces activation memory by 5×」 | ✓（<10% 一致且偏保守） |
| C12 | DeepSeek-V3 β1=0.9, β2=0.95, wd=0.1 | 2412.19437：「We employ the AdamW optimizer…β1=0.9, β2=0.95, and weight_decay=0.1」 | ✓ |
| C13 | GPT-3 β2=0.95, ε=10⁻⁸, clip 1.0 | 2005.14165 Appendix B：「we use Adam with β1=0.9, β2=0.95, and ϵ=10⁻⁸, we clip the global norm of the gradient at 1.0」 | ✓ |
| C14 | Llama-3 β2=0.95 出处 | 2407.21783 全文无 β1/β2/ε（`beta`/`0.95`/MathML alttext 三重检索） | ✗ → S2-2 |
| C15 | OLMo 2 β=(0.9,0.95), wd=0.1 | 论文正文未写 β；OLMo-core `OLMo2-7B.py`：`weight_decay=0.1, betas=(0.9, 0.95)` + embeddings no-wd（OptimGroupOverride） | ✓（config 级，出处层级需注明 → 并入 S2-2 修正） |
| C16 | MiniCPM = WSD 出处 | 2404.06395：「we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler…makes the reusing of model intermediate checkpoints highly feasible」 | ✓ |
| C17 | critical batch 公式 | 1812.06162 alttext：B_simple = tr(Σ)/\|G\|²；B_noise = tr(HΣ)/GᵀHG | 公式 ✓、命名偏差 → S3-3 |
| C18 | ST-MoE z-loss | 2202.08906 Eq.(5)：L_z = (1/B)Σ(log Σⱼ e^x)²，router logits；「an adaptation of the z-loss used for final softmax logits in the Mesh Tensorflow codebase」 | 谱系/平方 → S3-2 |
| C19 | Liger-Kernel「无独立论文」 | arXiv:2410.10989「Liger Kernel: Efficient Triton Kernels for LLM Training」存在且为 LinkedIn 官方 | ✗ → S2-1 |
| C20 | RAdam 视角（Adam 早期 v̂ 方差） | 1908.03265 标题即「On the Variance of the Adaptive Learning Rate and Beyond」 | ✓ |

### 2.4 数学声称独立复核（推导/数值）

| # | 声称 | 复核方式 | 结论 |
|---|---|---|---|
| M1 | matmul 反向 ∂L/∂A=GBᵀ、∂L/∂B=AᵀG | 逐元素链式法则独立推导 | ✓ |
| M2 | softmax-CE 融合梯度 p−δ_y | 独立推导（∂/∂zᵢ[−z_y+logsumexp(z)]=pᵢ−δ_{iy}） | ✓ |
| M3 | fp16 最小正规数 2⁻¹⁴、最小次正规 2⁻²⁴ | IEEE 754 half：5 位指数 bias 15 → E_min=1−15=−14；10 位尾数 → 次正规 2⁻¹⁴·2⁻¹⁰=2⁻²⁴ | ✓ |
| M4 | bf16 免 scaling（8 位指数 = fp32 同动态范围） | IEEE bfloat16 定义 | ✓ |
| M5 | loss scaling 恒等（scale S 乘 loss、更新前除回） | 梯度线性性 | ✓ |
| M6 | momentum 两形式等价（EMA ↔ 速度形式，变量代换） | 独立代数：v_t = m_t/(1−β)，步长吸收因子 | ✓ |
| M7 | 二次碗 GD (κ−1)/(κ+1)、momentum (√κ−1)/(√κ+1) | 最优定步长 GD 与 Polyak heavy ball 经典结果（与 Polyak 1964 一致） | ✓ |
| M8 | Adam bias correction 几何级数 E[m_t]=(1−β₁ᵗ)g | 独立求和 (1−β₁)Σβ₁^{t−i}=1−β₁ᵗ（平稳假设） | ✓ |
| M9 | AdamW 解耦（L2 被 √v̂ 逐坐标扭曲 → 解耦 wd） | 与 1711.05101 §2 论证一致，独立重推 | ✓ |
| M10 | linear scaling rule（小步近似 η∝B + gradual warmup） | 1706.02677 abstract 与已知结论 | ✓ |
| M11 | 谱范数最速下降 = msign（min ⟨G,Δ⟩ s.t. ‖Δ‖₂≤η ⇒ Δ=−ηUVᵀ） | von Neumann 迹不等式/奇异值对齐独立推导 | ✓ |
| M12 | Bradley-Terry 从随机效用（Gumbel iid ⇒ σ(r_w−r_l)） | 独立推导（Gumbel 差 = logistic） | ✓ |
| M13 | reward 平移不可识别（r 与 r+c(x) 同偏好） | σ 差分不变性，平凡验证 | ✓ |
| M14 | 反向 FLOPs ≈ 2× 前向（matmul 反向两次同形 matmul） | C=AB 反向 dA=G·Bᵀ、dB=Aᵀ·G 各一次同形 matmul | ✓ |
| M15 | 融合 CE 内存账：V=128256 × T=4096 bf16 ≈ 1 GiB/序列 | 数值：128256×4096×2 B = 1,050,673,152 B = 0.978 GiB | ✓（「约 1 GiB」成立；V=128256 即 Llama-3 词表） |

### 2.5 归因核对

| 归因 | 结论 |
|---|---|
| Polyak 1964 momentum（无 arXiv，标注） | ✓（"Some methods of speeding up the convergence of iteration methods"） |
| Sutskever et al. 2013 ICML（无 arXiv，标注） | ✓（ICML 2013，JMLR W&CP） |
| Williams 1992 REINFORCE（§4.1 例） | ✓ |
| Bradley-Terry 1952（§4.1 例、3.2） | ✓ |
| label smoothing → 1512.00567（Szegedy et al.） | ✓ |
| z-loss → ST-MoE 2202.08906 | ✓（但需注意 router/final-logits 谱系 → S3-2） |
| AMSGrad 反例 → 1904.09237 | ✓ |
| WSD → MiniCPM 2404.06395 | ✓ |
| GRPO 出处 = DeepSeekMath（顺带核 ID） | ✓ |
| 组织更名两例（deepspeedai / verl-project） | 未在本轮范围内逐一路径复核，repo 级重定向机制可信；P4+ 归 evaluator B |

### 2.6 覆盖率声明

- 范围内（1.1-3.2 + §1-§4）全部 22 个 arXiv ID：22/22 核实
- §4.1 示例 ID 7 个 + 边界/补查 2 个：9/9 核实
- GitHub 路径 8 项：8/8 核实
- 论文内容断言抽查 20 项：17 通过、3 失败（→ S2-1/S2-2/S2-3）、2 项命名/谱系偏差（→ S3-2/S3-3）
- 数学声称独立复核 15 项：15/15 通过
- 未覆盖：µP 缩放结论表（约定归 evaluator C）、Part 4-9 全部内容（归 evaluator B/C）

---

## 三、结论

- **S1 = 0，S2 = 3，S3 = 3**
- **放行结论：有条件放行**——无 S1；3 项 S2（Liger-Kernel 论文存在性、Llama-3 β₂ 出处、E4M3/E5M2 与 DeepSeek-V3 框架矛盾）均为 spec 级可局部修复项，修复后无需复审全 spec；建议 W0 内修复 S2-1/2/3 与 S3-1（ Sources 补 ID），S3-2/S3-3 随 W1/W2 章写作时落实并在该波 evaluator 复核中验证。
