# Training 维度 Spec 对抗审查报告 — Evaluator C（Part 7-9：预训练全流程 / RL 数学详解 / 后训练全流程）

- **审查对象**：`docs/superpowers/specs/2026-11-16-training-dimension-design.md` §5 Part 7-9（章节 7.1-9.5）
- **审查人**：evaluator C（fresh context，与 spec 作者无共享上下文）
- **日期**：2026-11-16
- **方法**：全部一手核实——arXiv API 批量核对 ID↔标题（33/33 全覆盖）、ar5iv 全文 grep 关键论断（23 篇论文 + Schulman 博客）、GitHub API 核 repo 与文件路径（8 个）、数学独立重推导（GRPO 组基线偏差、k3 无偏性、DPO 闭式链、GAE 形式）

## Findings 表

### S1（与一手来源直接矛盾，须修复后才可放行）

| # | 位置 | 断言 | 证据（一手来源） | 建议修正 |
|---|---|---|---|---|
| S1-1 | §7.2（02-recipes.md 内容行） | 「µTransfer 工程用法（小代理调参 → 大模型直迁的成本账——**Moonlight 实践口径**）」 | Moonlight（arXiv:2502.16982）摘要原文："These techniques allow Muon to work **out-of-the-box** on large-scale training **without the need of hyper-parameter tuning**"；全文（ar5iv 全量）"transfer"/"Tensor Programs"/"parametriz" **0 hits**。Moonlight 的核心主张恰是 wd + per-parameter update RMS 对齐（≈0.2）后**免**代理调参——与 µTransfer 的「小代理调参→大模型直迁」是相反路线 | µTransfer 工程口径归 TP-V（2203.03466，GPT-3 µP 迁移实验为其原生出处）；Moonlight 改写为「update RMS 匹配→免调参」的**对照路线**（µP 哲学 vs RMS 匹配哲学），两路线并陈 |
| S1-2 | §9.5（05-rlvr-and-reasoning.md 内容行） | 「**Kimi K1.5** 训练侧要点（长 CoT RL 实践：长度惩罚、**语言混杂处理**）」 | K1.5（arXiv:2501.12599）全文 "mixing"/"mixed" **0 hits**，无 language consistency reward；其 RL 侧要点为 length penalty（§2.3.3 已核实）、online mirror descent 变体、sampling strategy、long2short。**语言混杂处理是 DeepSeek-R1 的**：2501.12948 原文 "To mitigate the issue of language mixing, we introduce a **language consistency reward** during RL training"（reasoning-oriented RL 阶段） | 「语言混杂处理」移到 R1 链条（R1-Zero 的 language mixing 问题 → R1 的 language consistency reward）；K1.5 括注改为「长度惩罚、online mirror descent 变体、long2short」 |

### S2（不精确 / 缺限定 / 缺来源，应修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S2-1 | §8.3 锚点 | 「TRL `ppo_trainer`（官方实现走读）」 | GitHub API：`huggingface/trl` 的 `trl/trainer/ppo_trainer.py` 现 **404**；删除 commit 为 2026-02-10 "Remove deprecated PPO after moved to experimental (#5051)"；当前 `trl/trainer/`（dpo/grpo/sft/kto/rloo/reward/distillation 等）与 `trl/experimental/`（a2po/gmpo/cpo 等）均无任何 ppo 文件。TRL main 已不含 PPO 实现 | 8.3 走读锚点主换 CleanRL `ppo.py`（已列，repo 已核实存在且含 ppo.py 全家族）；或钉 TRL 历史 tag（v0.23 之前）注明版本；或以 OpenRLHF / veRL 的 PPO 实现顶替（与 8.5 走读复用）。§6.1 research pass 的「文件路径核实」项应把 ppo_trainer 列入 |
| S2-2 | §8.4 内容行 | 「组均值作为 baseline 的**无偏性** / 方差分析」 | 独立重推导：同 prompt 采 G 个 i.i.d. 样本、baseline = 全组均值 b̄=(1/G)Σrⱼ 时，E[∇log π(oᵢ)·b̄] = (1/G)·∇J（j=i 项贡献；j≠i 项因 E[∇log π]=0 消失），得 **E[ĝ] = (G−1)/G · ∇J**——严格有偏（(G−1)/G 缩放）；leave-one-out baseline 才精确无偏。叠加 std 归一化后另有难度偏置（Dr.GRPO 2503.20783 原文：question-level normalization 给全对/全错问题更高权重，"leading to a difficulty-level bias"——已核实） | 措辞改为「组均值 baseline 的**偏差-方差分析**：全组均值的 (G−1)/G 缩放偏差与 LOO 无偏版对照；std 归一化引入的难度偏置（Dr.GRPO）——与 8.1 最优 baseline 对照」。该修正项本就在 §6.3 强制重推导清单（8.4 GRPO 组基线）覆盖范围内，spec 正文应先写对 |
| S2-3 | §8.4 Sources | 「k3 低方差无偏 KL 估计推导」未列一手来源 | k1/k2/k3 估计器命名与结论出自 **John Schulman 博客 "Approximating KL Divergence"（joschu.net/blog/kl-approx.html，2020）**——已 curl 核实原文："k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator"。**DeepSeekMath 原文（2402.03300）对该估计器的引用即为 "(Schulman 2020)"**——ar5iv 全文核实 | 8.4 Sources 加 Schulman blog 2020（按 §4.1 博客/老经典口径标注，无 arXiv）；DeepSeekMath 保留为 GRPO 应用出处。断言本身（k3 无偏、低方差）无误 |
| S2-4 | §7.5 锚点 + 内容行 | 「Qwen2.5-1M ⟨verify⟩」未解；「4K → 32K → 128K → 1M 逐段配方」复合箭头 | **Qwen2.5-1M Technical Report = arXiv:2501.15383**（2025-01-26，arXiv API 核实存在）——⟨verify⟩ 可销。但该文长上下文**训练只到 262,144 tokens**（四阶段 32,768→65,536→131,072→262,144，ar5iv 全文 Table 2 核实），**1M 是推理侧 DCA+YaRN 外推**（"over 80% accuracy on sequences up to 1M... despite being trained only on sequences up to 32K"指 DCA 消融）。另 Llama-3 实为 8K→128K 六阶段（~800B tokens），DSv3 为 4K→32K→128K——复合箭头把三家不同口径混排 | 7.5 Sources 加 2501.15383；内容行改为「训练长度逐段扩展（DSv3 4K→32K→128K 两阶段各 1000 步；Llama-3 8K→128K 六阶段 800B tokens；Qwen2.5-1M 训到 256K）+ 推理侧外推到 1M（DCA）」 |
| S2-5 | §8.5 内容行 | 「veRL（HybridFlow：**single-controller 编程模型**、3D-HybridEngine 权重 resharding…）」 | HybridFlow 论文（2409.19256）摘要原文："We propose HybridFlow, which **combines single-controller and multi-controller paradigms in a hybrid manner**"——hybrid 控制模型是该文核心贡献，单写 single-controller 遗漏一半且误导（既有 RL 框架 RLLib 才是纯 single-controller，论文批评的正是它） | 改为「hybrid 控制模型（数据流编排 single-controller + 模型内分布式计算 multi-controller）」 |
| S2-6 | §7.4 内容行 | 「异步 checkpoint（…——**Llama-3 的异步方案口径**）」 | herd paper（2407.21783）§3.3 仅说 "we **reduced job startup and checkpointing time**"、"We aim to **minimize GPU pause time** during checkpointing"、"highly **bursty** checkpoint writes"（每 GPU 1MB–4GB 状态、存储可持续 2TB/s/峰值 7TB/s）——未描述显式异步 checkpoint 机制；PyTorch DCP 的 async_save 是文档级出处，非 herd 论文 | 软化为「herd 的 checkpoint 开销口径（bursty writes 账 + minimize GPU pause time）」，异步 checkpoint 机制另引 torch.distributed.checkpoint 文档（恰好本章锚点已有 DCP，可内联） |

### S3（建议性 / 一致性问题）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S3-1 | §7.4 锚点 | `torch.distributed.checkpoint` 标 **[A]** | DCP 无正式 paper；§4.1 明确把「PyTorch 框架本体」列为**教学参照**档。同 spec builder-check 第 5 条已修过 TRL/Liger 同类问题，此处漏网 | 改标「教学参照 + 官方实现」 |
| S3-2 | §7.2 内容行 | 「GPT-3 / Llama-2 / Llama-3 / DeepSeek-V3 / **Qwen3** / OLMo 2 / GLM-4.5 的 lr / bsz / warmup / decay / wd / β2 / clip **全字段对照表**」 | Qwen3 TR（2505.09388）**不披露具体数值**："we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions... we set the predicted optimal learning rate and batch size strategy for each model"——无数值。GLM-4.5（2508.06471）则全披露（Muon、cold 0→2.5e-4→2.5e-5 cosine、bsz 16M→64M tokens/首 500B、wd 0.1、seq 4K→32K→128K mid-training）已核实 | 内容行加注「未披露字段标 未披露 并注明披露口径差异」；把 GLM-4.5 的 batch ramp 与 DSv3 并列（两家 ramp 实证） |
| S3-3 | §9.1 内容行 | 「LR 量级 = 预训练峰值**约 1/10** 的惯例与出处」 | Llama-3 实证：SFT LR = 1e-5（herd paper §4.1.3 核实），405B 预训练峰值 LR = 8e-5（Table 3）→ 实际 **1/8**；「约 1/10」为数量级口径 | 写作时逐家给出处（Llama-3 1/8、其他家 research pass 核实），避免把 1/10 写成普适规则 |
| S3-4 | §9.5 内容行 | 「长 CoT 训练动力学（response 长度随训练增长、**entropy 塌缩风险——R1-Zero 分析论文口径**）」 | Dr.GRPO（2503.20783）主打长度/std 偏置（response length increase ✓ 核实），全文 entropy 仅 1 处提及；**entropy collapse 的主出处是 DAPO**（Clip-Higher 动机即 "avoids entropy collapse"，2503.14476 已核实） | 9.5 Sources 加 2503.14476 或注明「entropy 塌缩回链 8.4 DAPO」 |
| S3-5 | §9.4 内容行 | 「STaR 式迭代：bootstrapping 收敛问题与**多样性塌缩风险**」 | STaR 原文（2203.14465）实证的是「loop 无法解新题→无训练信号→停滞」（其 rationalization 机制的动机，已核实）；「多样性塌缩」是 self-training 文献的外延担忧，非 STaR 原文内容 | 行内区分「STaR 原文的停滞问题」与「self-training 文献的多样性塌缩风险（research pass 补出处）」 |

## Verification log（抽查项 × 来源 × 结论）

### A. arXiv ID ↔ 标题（arXiv API `id_list` 批量核对，2026-11-16）

**P7-9 全部 33 个唯一 arXiv ID，33/33 核对命中（覆盖率 100%）**：

| ID | 期望 | API 返回标题 | 结论 |
|---|---|---|---|
| 2305.10429 ⟨verify⟩ | DoReMi | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | ✓ |
| 2203.14465 ⟨verify⟩ | STaR | STaR: Bootstrapping Reasoning With Reasoning | ✓ |
| 2501.12599 ⟨verify⟩ | Kimi K1.5 | Kimi k1.5: Scaling Reinforcement Learning with LLMs | ✓ |
| 2305.11206 ⟨verify⟩ | LIMA | LIMA: Less Is More for Alignment | ✓ |
| 2406.17557 | FineWeb | The FineWeb Datasets… | ✓ |
| 2406.11794 | DCLM | DataComp-LM… | ✓ |
| 2203.03466 | µP/TP-V | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | ✓ |
| 2210.02414 | GLM-130B | GLM-130B: An Open Bilingual Pre-trained Model | ✓ |
| 2407.21783 | Llama-3 herd | The Llama 3 Herd of Models | ✓ |
| 2412.19437 | DeepSeek-V3 | DeepSeek-V3 Technical Report | ✓ |
| 2501.00656 | OLMo 2 | 2 OLMo 2 Furious | ✓ |
| 2508.06471 | GLM-4.5 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | ✓ |
| 2304.11277 | FSDP | PyTorch FSDP… | ✓ |
| 2309.00071 | YaRN | YaRN: Efficient Context Window Extension… | ✓ |
| 1706.03741 | Christiano RLHF | Deep reinforcement learning from human preferences | ✓ |
| 1506.02438 | GAE | High-Dimensional Continuous Control Using Generalized Advantage Estimation | ✓ |
| 1707.06347 | PPO | Proximal Policy Optimization Algorithms | ✓ |
| 1502.05477 | TRPO | Trust Region Policy Optimization | ✓ |
| 2402.03300 | DeepSeekMath/GRPO | DeepSeekMath… | ✓ |
| 2501.12948 | DeepSeek-R1 | DeepSeek-R1… | ✓ |
| 2503.14476 | DAPO | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | ✓ |
| 2503.20783 | R1-Zero 分析 | Understanding R1-Zero-Like Training: A Critical Perspective | ✓ |
| 2507.18071 | GSPO | Group Sequence Policy Optimization | ✓ |
| 2409.19256 | HybridFlow/veRL | HybridFlow: A Flexible and Efficient RLHF Framework | ✓ |
| 2405.11143 | OpenRLHF | OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework | ✓ |
| 2309.06180 | vLLM | Efficient Memory Management for Large Language Model Serving with PagedAttention | ✓ |
| 2203.02155 | InstructGPT | Training language models to follow instructions with human feedback | ✓ |
| 2305.18290 | DPO | Direct Preference Optimization… | ✓ |
| 2310.12036 | IPO | A General Theoretical Paradigm to Understand Learning from Human Preferences | ✓（spec 已注正式标题） |
| 2402.01306 | KTO | KTO: Model Alignment as Prospect Theoretic Optimization | ✓ |
| 2405.14734 | SimPO | SimPO: Simple Preference Optimization with a Reference-Free Reward | ✓ |
| 2403.07691 | ORPO | ORPO: Monolithic Preference Optimization without Reference Model | ✓ |
| 2505.09388 | Qwen3 | Qwen3 Technical Report | ✓ |

补充新发现：**Qwen2.5-1M Technical Report = arXiv:2501.15383**（spec 7.5 ⟨verify⟩ 应销记并补入 Sources）。

### B. GitHub repo / 文件路径（GitHub API，2026-11-16）

| 项 | 结果 | 结论 |
|---|---|---|
| vwxyzjn/cleanrl ⟨verify⟩ | 存在；`cleanrl/ppo.py` + 8 个 PPO 变体在列 | ✓ 销记 |
| allenai/OLMo-core ⟨verify⟩ | 存在（"PyTorch building blocks for the OLMo ecosystem"） | ✓ 销记 |
| eric-mitchell/direct-preference-optimization ⟨verify⟩ | 存在（"Reference implementation for DPO"，第一作者官方 repo） | ✓ 销记 |
| verl-project/verl | 存在；`volcengine/verl` API 返回 **301 Moved Permanently**（spec 更名记录属实）；`verl/trainer/main_ppo.py` 200 | ✓ |
| huggingface/trl | 存在；`dpo_trainer.py` ✓ `grpo_trainer.py` ✓ `sft_trainer.py` ✓；**`ppo_trainer.py` 404（2026-02-10 删除）** | ✗ → S2-1 |
| OpenRLHF/OpenRLHF | 存在 | ✓ |
| BytedTsinghua-SIA/DAPO | 存在（官方实现） | ✓ |
| openai/instructgpt | **不存在**（OpenAI 无 InstructGPT 训练代码 repo） | ✓ §9.2 [B]「代码未开放」准确 |

### C. 论文内容断言（ar5iv 全文 grep + 引句）

| 断言 | 一手来源核对结论 |
|---|---|
| DeepSeekMath GRPO 用 k3 形式 KL 估计器 | ✓ 原文公式 D_KL = (π_ref/π_θ) − log(π_ref/π_θ) − 1，且**自引 "(Schulman 2020)"** → S2-3 |
| DeepSeekMath：KL 加在 loss（vs PPO 加在 reward） | ✓ 原文 "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence… to the loss" |
| Dr.GRPO 修 std 归一化 + 长度项偏置 | ✓ 原文 "removing the length and std normalization terms"；两类偏置（response-level length bias / question-level difficulty bias）原文俱在 |
| GSPO ratio = 长度归一化几何平均 | ✓ 原文 sᵢ(θ) = (π_θ(yᵢ|x)/π_old(yᵢ|x))^{1/\|yᵢ\|} = exp((1/\|yᵢ\|)Σ_t log ratio) |
| DAPO 四件套 | ✓ 原文 "4 key techniques"：Clip-Higher / Dynamic Sampling / Token-Level Policy Gradient Loss / Overlong Reward Shaping；实现基于 verl 原文明示 |
| DeepSeek-R1 多阶段配方 | ✓ 原文四阶段：Cold Start → Reasoning-oriented RL → Rejection Sampling + SFT（**重训 V3-Base**）→ RL for all Scenarios；R1-Zero = V3-Base 上直接 GRPO |
| R1 语言一致性奖励 | ✓ 原文 "language consistency reward during RL training" → S1-2 |
| DeepSeek-V3 bsz ramp 3072→15360 | ✓ 原文 "gradually increased from 3072 to 15360 in the training of the first 469B tokens" |
| DeepSeek-V3 配方字段 | ✓ AdamW β₁=0.9 β₂=0.95 wd=0.1 clip=1.0；LR 0→2.2e-4（2K 步 warmup）→10T 常量→4.3T cosine 降至 2.2e-5→分段常数；seq 4K |
| DeepSeek-V3 YaRN 两阶段 | ✓ 4K→32K→128K，每阶段 1000 步，s=40 α=1 β=32 √t=0.1·ln s+1，仅作用于 decoupled shared key |
| DeepSeek-V3 零不可恢复 spike | ✓ 原文 "we did not encounter any irrecoverable loss spikes or have to roll back" |
| Llama-3 405B 中断统计 ⟨verify 精确数字⟩ | ✓ §3.3.4：**54 天 snapshot、466 次中断（47 计划内 + 419 意外）、~78% 硬件相关、GPU 占意外 58.7%（Faulty GPU 148=30.1%、HBM3 72=17.2%、SDC 6=1.4%）、每天 ≥1 次、>90% effective training time、人工介入仅 3 次**；405B 末段 16,384 GPU（Table 4：TP8×CP16×PP16×DP8、128K seq、16M tokens/batch、380 TFLOPs/GPU、38% MFU） |
| Llama-3 post-training RFT 用法 | ✓ SFT+RS+DPO（明示不用 PPO）；RS 每 prompt 采 K=10–30、RM 选优、多轮迭代 |
| Llama-3 长上下文 | ✓ 8K→128K 六阶段、~800B tokens；annealing 末 40M tokens 线性降 0 + 高质量上采样 + Polyak 权重平均 |
| Llama-3 SFT LR | ✓ 1e-5（最大模型）→ S3-3 |
| DoReMi proxy + group DRO | ✓ 原文 "trains a small proxy model using group distributionally robust optimization (Group DRO)…280M proxy → 8B" |
| DCLM 管线 | ✓ 412M–7B benchmark；dedup 对照 MinHash vs Bloom filter（baseline 用 Bloom）；fastText OH-2.5+ELI5 分类器最优；perplexity filtering 在被比较方法之列 |
| FineWeb 小模型 proxy ablation | ✓ 原文 data ablation 方法学（每版本两模型两种子、训练于等量子样）；MinHash 去重；datatrove 开源 |
| OLMo 2 全开放 | ✓ 原文 "full training data, training code and recipes, training logs and thousands of intermediate checkpoints"（Dolmino 1124 mix、OLMo-core、WandB logs） |
| Qwen3 thinking 模式训练 | ✓ 四阶段 post-training：Long-CoT Cold Start → Reasoning RL → Thinking Mode Fusion（/think、/no_think flag 混插）→ General RL；小模型 Strong-to-Weak 蒸馏（1/10 GPU hours） |
| Qwen3 预训练 HP 披露 | ✗ 不披露数值（scaling-law 预测口径）→ S3-2 |
| K1.5 长度惩罚 | ✓ §2.3.3 Length Penalty（length reward 抑制长度增长） |
| K1.5 语言混杂处理 | ✗ 全文无 → S1-2 |
| DPO 闭式解链 | ✓ Eq.4 π*(y\|x)=(1/Z(x))π_ref·exp(r/β)；Eq.5 r=β log(π*/π_ref)+β log Z(x)；代入 BT 后 Z(x) 抵消（只依赖 x）；DPO 目标 −log σ(βΔw−βΔl) 与 spec 一致 |
| IPO ∈ 2310.12036 | ✓ ΨPO 框架 + DPO 过拟合分析（BT 假设 + 确定性偏好 ⇒ 过拟合）+ IPO=identity mapping |
| STaR 迭代机制 | ✓ rationale 生成→留正确→rationalization 补失败题→SFT→迭代；原文停滞问题动机核实 → S3-5 |
| LIMA 口径 | ✓ 65B LLaMa + 1,000 精选样本、无 RLHF；Superficial Alignment Hypothesis |
| HybridFlow 控制模型 | 原文为 hybrid（single+multi controller）→ S2-5；3D-HybridEngine resharding ✓（zero memory redundancy；70B actor 每 iteration 传 140GB、占迭代时间 36.4%） |
| OpenRLHF 架构 | ✓ 原文 "re-designs scheduling… using Ray, vLLM, and DeepSpeed"；四模型协调挑战明示 |
| GAE 公式 | ✓ 原文 Â^GAE(γ,λ) = (1−λ)Σλ^k Â^(k) = Σ_l (γλ)^l δ_{t+l}（spec 8.2 形式一致） |
| GLM-130B spike 记录 | ✓ DeepNorm(Post-LN α=(2N)^{1/2}) + Embedding Gradient Shrink(α=0.1) + 三次 late-stage divergence 记录俱在 |
| Moonlight µTransfer | ✗ 全文无 → S1-1 |
| Schulman k1/k2/k3 博客 | ✓ joschu.net/blog/kl-approx.html 原文核实（k3 无偏且方差低于 k1/k2） |
| InstructGPT [B] 分级 | ✓ OpenAI 未开放训练代码（GitHub API 无该 repo） |

### D. 数学独立重推导

| 项 | 结论 |
|---|---|
| GRPO 组均值 baseline 无偏性 | **E[ĝ] = (G−1)/G·∇J**，严格有偏；LOO 无偏 → S2-2（推导见 findings 表） |
| k3 无偏性 | E_p[r − log r − 1]（r=q/p）= 1 + KL − 1 = KL(p‖q) ✓ 无偏成立 |
| DPO 闭式解 | 变分法（对 π 逐 x 优化 KL 约束目标）得 π*∝π_ref·exp(r/β) ✓ 与原文 Eq.4 一致；Z(x) 只依赖 x ⇒ BT 差分中抵消 ✓ |
| BT 损失形式 | p(y_w≻y_l)=σ(r_w−r_l) ⇒ NLL = −log σ(r_w−r_l) ✓（9.2 与 3.2 自洽） |
| GAE | spec 形式与原文一致 ✓ |
| PPO clip | 公式标准；1707.06347 无 ar5iv 全文（2017 年文），摘要核实「surrogate objective + 多 epoch minibatch」口径 ✓ |
| µP 结论表方向 | spec 7.2 正文未断言具体方向（仅「Tensor Programs V 结论表」+ §6.3 强制核对），spec 级无方向错误可审；ID 正确 ✓ |

### E. 覆盖率

- arXiv ID：P7-9 Sources 出现的 **33/33（100%）** 全部 API 核对
- GitHub 锚点：8/8 核实（1 个失效 → S2-1）
- 论文内容断言：23 篇 ar5iv 全文 + 1 篇博客全文 grep；任务书点名校查项 1–13 全覆盖
- ⟨verify⟩ 项：7 项全部解决（DoReMi ✓、STaR ✓、K1.5 ✓、LIMA ✓、CleanRL ✓、OLMo-core ✓、DPO 官方 repo ✓）+ 附录「Llama-3 中断频率精确数字」✓ 已锁定（466/54 天/16,384 卡）

## 结论

**不放行**（2 个 S1：Moonlight 与 K1.5 两处归因与一手来源直接矛盾）。S1 修复量均为一行级改写（换归因对象），S2 修复后本评估范围可通过。arXiv ID 层零错误（33/33），spec 附录预核实记录属实；主要风险集中在「内容行把技术归错论文」与「锚点时效（TRL PPO 已删）」两类。
