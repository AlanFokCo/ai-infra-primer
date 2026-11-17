# references.md — training 维度文献索引与术语表

本文件是 training 维度全部章节的引用唯一事实源。核心规则：**先收录后引用**——条目由各波 research pass 核实后写入本文件，章节正文才能引用；`scripts/lint-chapter.sh` 检查项 4 强制执行：正文出现的 arXiv ID 在本文件 grep 不到即 lint FAIL。写作 agent 不新增条目，只引用已有条目。

## 写入规则

- 条目只由 research pass 写入；「核实日期」= 当次核实 ID / URL 的日期（YYYY-MM-DD）。spec 中 `⟨verify⟩` 标记的条目必须经 WebFetch 复核（arXiv abs 页 / HF config / repo 路径）后才收录。
- 非 arXiv 来源（blog、官方文档、repo 文件、无 arXiv 的经典文献）在「arXiv ID / URL」字段写完整 URL 或标准出处（如「Polyak 1964, USSR Computational Mathematics」）。lint 只机械核对 arXiv ID，URL 条目的真实性由 research pass 与 evaluator 把关。
- 「证据等级」五档：[A] / [B] / [C] / [W] / 教学参照，定义见 STYLE_GUIDE §6；training 维度对方法/框架锚点的口径细化（paper + official code 即 [A]、无 paper 的官方库为教学参照）见 training spec §4.1。[W] 级找到正式 paper 后的回升判定同 STYLE_GUIDE §6。
- key 为小写 slug，全文件唯一，取论文或资源的通用短名（如 `zero`、`megatron-lm`）。
- 章节写完（或引用关系变化）后回写「被引章节」；一条目被多章引用时逗号分隔累加（如 `4.1, 4.2`）。
- 正文引用格式 `[Rafailov et al., 2023]`，链接到本文件对应条目（spec §4.6 / STYLE_GUIDE §6）。

## 条目 schema

每条目七个字段：

| 字段 | 含义 |
| --- | --- |
| key | 条目唯一 slug（小写） |
| 标题 | 论文 / 资源标题 |
| 作者 | 第一作者 et al.，或机构名 |
| arXiv ID / URL | arXiv 编号；非 arXiv 来源写完整 URL |
| 证据等级 | [A] / [B] / [C] / [W] / 教学参照 |
| 核实日期 | research pass 核实日期（YYYY-MM-DD） |
| 被引章节 | 引用本条目的章节号，逗号分隔 |

条目写法（表头 + 分隔行 + 数据行；下行示例同时是 lint grep 的格式基准）：

```
| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
```

## 条目

按 Part 分组；research pass 向对应 Part 的表追加，chapter → sources 映射体现在「被引章节」字段。

> **W1 research pass 收录说明（2026-11-16）**：下列条目的 arXiv ID↔标题经 arXiv API（export.arxiv.org）复核——大部分经 spec 预核实（2026-11-16，见 spec 附录）与 spec 四路对抗审查（A/B/C 三路）双重核实；repo 路径经 GitHub contents API 核实。W1 写作只需要 Part 1-2 组；跨波共享锚点（deepseekv3 / llama3 / olmo2 等）随本表先行收录，后续波次扩充其余 Part 组。

### Part 1 · Autograd 与反向机制

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| baydin-ad-survey | Automatic Differentiation in Machine Learning: a Survey | Baydin et al. | arXiv:1502.05767 | [B] | 2026-11-16 | 1.1 |
| pytorch-repo | pytorch/pytorch repo（torch.autograd、torch.optim、torch.utils.checkpoint、torch.nn.utils.clip_grad_norm_、torch.distributed.checkpoint） | PyTorch | https://github.com/pytorch/pytorch | 教学参照 | 2026-11-16 | 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.4, 4.1, 4.3, 7.4 |
| pytorch-ddp | PyTorch Distributed: Experiences on Accelerating Data Parallel Training | Li et al. | arXiv:2006.15704 | [A] | 2026-11-16 | 1.1, 1.5, 4.1 |
| flashattention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | arXiv:2205.14135 | [A] | 2026-11-16 | 1.2, 7.1 |
| liger-kernel | Liger Kernel: Efficient Triton Kernels for LLM Training | Hsu et al.（LinkedIn） | arXiv:2410.10989 ; https://github.com/linkedin/Liger-Kernel | [A] | 2026-11-16 | 1.2, 3.1 |
| gradient-checkpointing | Training Deep Nets with Sublinear Memory Cost | Chen et al. | arXiv:1604.06174 | [B] | 2026-11-16 | 1.2, 1.4 |
| megatron-sp | Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al. | arXiv:2205.05198 | [A] | 2026-11-16 | 1.4, 4.4, 5.2, 6.1 |
| mixed-precision | Mixed Precision Training | Micikevicius et al. | arXiv:1710.03740 | [B] | 2026-11-16 | 1.3 |
| fp8-formats | FP8 Formats for Deep Learning | Micikevicius et al. | arXiv:2209.05433 | [B] | 2026-11-16 | 1.3 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 | [A] | 2026-11-16 | 1.3, 2.2, 2.3, 2.4, 4.4, 5.1, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.5 |

### Part 2 · Optimizer 与 LR 动力学

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| polyak1964 | Some Methods of Speeding up the Convergence of Iteration Methods | Polyak | USSR Computational Mathematics and Mathematical Physics 4(5):1-17（无 arXiv） | [B] | 2026-11-16 | 2.1 |
| sutskever2013 | On the Importance of Initialization and Momentum in Deep Learning | Sutskever et al. | ICML 2013（PMLR 28(3):1139-1147，无 arXiv） | [B] | 2026-11-16 | 2.1 |
| adam | Adam: A Method for Stochastic Optimization | Kingma & Ba | arXiv:1412.6980 | [B] | 2026-11-16 | 2.2 |
| adamw | Decoupled Weight Decay Regularization | Loshchilov & Hutter | arXiv:1711.05101 | [A] | 2026-11-16 | 2.2, 2.4 |
| amsgrad | On the Convergence of Adam and Beyond | Reddi et al. | arXiv:1904.09237 | [B] | 2026-11-16 | 2.2 |
| gpt3 | Language Models are Few-Shot Learners | Brown et al. | arXiv:2005.14165 | [B] | 2026-11-16 | 2.2, 2.3, 2.4, 7.2 |
| llama2 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | arXiv:2307.09288 | [A] | 2026-11-16 | 2.2, 4.4, 7.2 |
| llama3 | The Llama 3 Herd of Models | Grattafiori et al. | arXiv:2407.21783 | [A] | 2026-11-16 | 2.2, 2.3, 2.5, 4.3, 4.4, 5.1, 5.2, 5.5, 7.2, 7.3, 7.4, 7.5, 9.1, 9.4 |
| olmo2 | 2 OLMo 2 Furious | Team OLMo et al. | arXiv:2501.00656 | [A] | 2026-11-16 | 2.2, 2.3, 2.4, 4.4, 5.5, 7.2, 7.3 |
| critical-batch | An Empirical Model of Large-Batch Training | McCandlish et al. | arXiv:1812.06162 | [B] | 2026-11-16 | 1.5, 2.1, 2.3 |
| linear-scaling | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Goyal et al. | arXiv:1706.02677 | [B] | 2026-11-16 | 2.1, 2.3 |
| minicpm | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Hu et al. | arXiv:2404.06395 | [A] | 2026-11-16 | 2.3, 7.1, 7.2 |
| radam | On the Variance of the Adaptive Learning Rate and Beyond | Liu et al. | arXiv:1908.03265 | [A] | 2026-11-16 | 2.3 |
| glm-130b | GLM-130B: An Open Bilingual Pre-trained Model | Zeng et al. | arXiv:2210.02414 | [A] | 2026-11-16 | 2.4, 7.3 |
| st-moe | ST-MoE: Designing Stable and Transferable Sparse Expert Models | Zoph et al. | arXiv:2202.08906 | [B] | 2026-11-16 | 2.4, 3.1 |
| muon-blog | Muon: An optimizer for hidden layers（Keller Jordan 博客，Muon 原始出处） | Jordan | https://kellerjordan.github.io/posts/muon/ | 教学参照 | 2026-11-16 | 2.5 |
| muon-repo | KellerJordan/Muon repo（Newton-Schulz 迭代官方实现） | Jordan | https://github.com/KellerJordan/Muon | 教学参照 | 2026-11-16 | 2.5 |
| moonlight | Muon is Scalable for LLM Training（Moonlight） | Liu et al.（Moonshot AI） | arXiv:2502.16982 ; https://github.com/MoonshotAI/Moonlight | [A] | 2026-11-16 | 2.5, 7.2, 7.3 |
| kimi-k2 | Kimi K2: Open Agentic Intelligence | Kimi Team | arXiv:2507.20534 | [A] | 2026-11-16 | 2.5, 5.5, 7.3 |

### Part 3 · Loss 函数族

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| label-smoothing | Rethinking the Inception Architecture for Computer Vision（label smoothing 出处） | Szegedy et al. | arXiv:1512.00567 | [B] | 2026-11-17 | 3.1 |
| focal-loss | Focal Loss for Dense Object Detection | Lin et al. | arXiv:1708.02002 | [B] | 2026-11-17 | 3.1 |
| chameleon | Chameleon: Mixed-Modal Early-Fusion Foundation Models（z-loss 输出侧出处） | Chameleon Team | arXiv:2405.09818 | [B] | 2026-11-17 | 3.1 |
| bradley-terry | Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons | Bradley & Terry | Biometrika 39(3/4):324-345（1952，无 arXiv） | [B] | 2026-11-17 | 3.2, 9.2, 9.3 |
| christiano | Deep Reinforcement Learning from Human Preferences | Christiano et al. | arXiv:1706.03741 | [B] | 2026-11-17 | 3.2, 8.1, 9.2 |
| dpo | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | arXiv:2305.18290 ; https://github.com/eric-mitchell/direct-preference-optimization | [A] | 2026-11-17 | 3.2, 9.3 |
| trl-repo | huggingface/trl repo（SFT/DPO/GRPO 等 trainer；TRL 无独立论文） | HuggingFace | https://github.com/huggingface/trl | 教学参照 | 2026-11-17 | 3.2, 6.3, 9.1, 9.3 |

### Part 4 · 数据并行与显存切分

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| zero | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al. | arXiv:1910.02054 | [A] | 2026-11-17 | 4.1, 4.2, 4.4, 5.5, 6.2 |
| zero-offload | ZeRO-Offload: Democratizing Billion-Scale Model Training | Ren et al. | arXiv:2101.06840 | [A] | 2026-11-17 | 4.2 |
| zero-infinity | ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning | Rajbhandari et al. | arXiv:2104.07857 | [A] | 2026-11-17 | 4.2 |
| zero-plusplus | ZeRO++: Extremely Efficient Collective Communication for Giant Model Training | Wang et al. | arXiv:2306.10209 | [A] | 2026-11-17 | 4.2 |
| deepspeed-repo | deepspeedai/DeepSpeed repo（ZeRO 各阶段实现；原 microsoft/DeepSpeed，组织已更名） | DeepSpeed | https://github.com/deepspeedai/DeepSpeed | [A] | 2026-11-17 | 4.2, 6.3 |
| fsdp | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Zhao et al. | arXiv:2304.11277 | [A] | 2026-11-17 | 4.2, 4.3, 4.4, 5.5, 7.4 |
| torchtitan-repo | pytorch/torchtitan repo（PyTorch 原生最小训练栈） | PyTorch | https://github.com/pytorch/torchtitan | 教学参照 | 2026-11-17 | 4.3, 6.3 |
| megatron-lm | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Shoeybi et al. | arXiv:1909.08053 ; https://github.com/NVIDIA/Megatron-LM | [A] | 2026-11-17 | 4.1, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 7.4 |
| megatron-cluster | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | Narayanan et al. | arXiv:2104.04473 | [A] | 2026-11-17 | 5.3, 5.5, 6.2 |
| gshard | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Lepikhin et al. | arXiv:2006.16668 | [B] | 2026-11-17 | 5.4 |


### Part 5 · 模型并行全家桶

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| gpipe | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Huang et al. | arXiv:1811.06965 | [B] | 2026-11-17 | 5.3 |
| zero-bubble | Zero Bubble Pipeline Parallelism | Qi et al. | arXiv:2401.10241 | [A] | 2026-11-17 | 5.3 |
| ring-attention | Ring Attention with Blockwise Transformers for Near-Infinite Context | Liu et al. | arXiv:2310.01889 | [A] | 2026-11-17 | 5.2 |
| striped-attention | Striped Attention: Faster Ring Attention for Causal Transformers | Brandon et al. | arXiv:2311.09431 ; https://github.com/exists-forall/striped_attention | [A]（官方 repo 在 exists-forall org——论文首页脚注出处；2026-11-17 复审核实。教训：初查 togethercomputer 系查错 org） | 2026-11-17 | 5.2 |
| ulysses | DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models | Jacobs et al. | arXiv:2309.14509 | [A] | 2026-11-17 | 5.2 |

### Part 6 · Megatron-LM 与现代训练栈

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| nemo | NVIDIA NeMo Framework（Megatron-Core 的上游封装；注意：NeMo 主仓已重组，github.com/NVIDIA/NeMo 现 301 至 NVIDIA-NeMo/Speech） | NVIDIA | https://github.com/NVIDIA-NeMo/Speech | 教学参照 | 2026-11-17 | 6.3 |
| megatron-bridge | NVIDIA-NeMo/Megatron-Bridge repo（Megatron 模型训练库 + HF 双向转换；NeMo 重组后的 LLM 训练入口） | NVIDIA | https://github.com/NVIDIA-NeMo/Megatron-Bridge | 教学参照 | 2026-11-17 | 6.3 |
| verl | HybridFlow: A Flexible and Efficient RLHF Framework（veRL 论文） | Sheng et al. | arXiv:2409.19256 ; https://github.com/verl-project/verl | [A] | 2026-11-17 | 6.3, 8.5 |
| openrlhf | OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework | Hu et al. | arXiv:2405.11143 ; https://github.com/OpenRLHF/OpenRLHF | [A] | 2026-11-17 | 6.3, 8.3, 8.5, 9.2 |
| olmo-core | allenai/OLMo-core repo（OLMo 2 的全开放训练栈） | AllenAI | https://github.com/allenai/OLMo-core | [A] | 2026-11-17 | 5.5, 6.3 |


### Part 7 · 预训练全流程

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| fineweb | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | Penedo et al. | arXiv:2406.17557 | [A] | 2026-11-17 | 7.1 |
| dclm | DataComp-LM: In search of the next generation of training sets for language models | Li et al. | arXiv:2406.11794 | [A] | 2026-11-17 | 7.1 |
| doremi | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | Xie et al. | arXiv:2305.10429 | [B] | 2026-11-17 | 7.1 |
| tensor-programs-v | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer（µP / µTransfer） | Yang et al. | arXiv:2203.03466 | [B] | 2026-11-17 | 7.2 |
| glm-4.5 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | GLM Team | arXiv:2508.06471 | [A] | 2026-11-17 | 7.1, 7.2, 7.5 |
| qwen3 | Qwen3 Technical Report | Qwen Team | arXiv:2505.09388 | [A] | 2026-11-17 | 7.2 |
| qwen2.5-1m | Qwen2.5-1M Technical Report | Qwen Team | arXiv:2501.15383 | [A] | 2026-11-17 | 7.5 |
| yarn | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | arXiv:2309.00071 | [A] | 2026-11-17 | 7.5 |


### Part 8 · RL 数学详解

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| williams1992 | Simple statistical gradient-following algorithms for connectionist reinforcement learning（REINFORCE） | Williams | Machine Learning 8:229-256（1992，无 arXiv） | [B] | 2026-11-17 | 8.1 |
| sutton1999 | Policy Gradient Methods for Reinforcement Learning with Function Approximation | Sutton et al. | NeurIPS 1999（无 arXiv） | [B] | 2026-11-17 | 8.1 |
| sutton-barto | Reinforcement Learning: An Introduction（第 2 版） | Sutton & Barto | MIT Press 2018，http://incompleteideas.net/book/the-book-2nd.html | [B] | 2026-11-17 | 8.1, 8.2 |
| greensmith2004 | Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning（最优 baseline 方差最小化的标准出处） | Greensmith et al. | JMLR 5:1471-1530（2004，无 arXiv） | [B] | 2026-11-18 | 8.1 |
| gae | High-Dimensional Continuous Control Using Generalized Advantage Estimation | Schulman et al. | arXiv:1506.02438 | [B] | 2026-11-17 | 8.2, 8.3 |
| ppo | Proximal Policy Optimization Algorithms | Schulman et al. | arXiv:1707.06347 | [B] | 2026-11-17 | 8.3 |
| trpo | Trust Region Policy Optimization | Schulman et al. | arXiv:1502.05477 | [B] | 2026-11-17 | 8.3 |
| cleanrl | vwxyzjn/cleanrl repo（PPO 等单文件教学实现） | Huang et al. | https://github.com/vwxyzjn/cleanrl | 教学参照 | 2026-11-17 | 8.2, 8.3 |
| deepseekmath | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 出处） | Shao et al. | arXiv:2402.03300 | [A] | 2026-11-17 | 8.4 |
| deepseek-r1 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | arXiv:2501.12948 | [A] | 2026-11-17 | 8.4, 9.5 |
| dapo | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | Yu et al. | arXiv:2503.14476 | [A] | 2026-11-17 | 8.4 |
| dr-grpo | Understanding R1-Zero-Like Training: A Critical Perspective（Dr.GRPO 出处） | Liu et al. | arXiv:2503.20783 | [A] | 2026-11-17 | 8.4, 9.5 |
| gspo | Group Sequence Policy Optimization | Zheng et al. | arXiv:2507.18071 | [A] | 2026-11-17 | 8.4 |
| schulman-kl | Approximating KL Divergence（k1/k2/k3 KL 估计器出处） | Schulman | http://joschu.net/blog/kl-approx.html（2020 博客，无 arXiv） | 教学参照 | 2026-11-17 | 8.4, 9.2 |
| vllm | Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al. | arXiv:2309.06180 | [A] | 2026-11-17 | 8.5 |


### Part 9 · 后训练全流程

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| instructgpt | Training language models to follow instructions with human feedback | Ouyang et al. | arXiv:2203.02155 | [B] | 2026-11-17 | 9.2 |
| ipo | A General Theoretical Paradigm to Understand Learning from Human Preferences（IPO / ΨPO） | Azar et al. | arXiv:2310.12036 | [B] | 2026-11-17 | 9.3 |
| kto | KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh et al. | arXiv:2402.01306 | [A] | 2026-11-17 | 9.3 |
| simpo | SimPO: Simple Preference Optimization with a Reference-Free Reward | Meng et al. | arXiv:2405.14734 | [A] | 2026-11-17 | 9.3 |
| orpo | ORPO: Monolithic Preference Optimization without Reference Model | Hong et al. | arXiv:2403.07691 | [A] | 2026-11-17 | 9.3 |
| star | STaR: Bootstrapping Reasoning with Reasoning | Zelikman et al. | arXiv:2203.14465 | [B] | 2026-11-17 | 9.4 |
| lima | LIMA: Less Is More for Alignment | Zhou et al. | arXiv:2305.11206 | [B] | 2026-11-17 | 9.1 |
| kimi-k1.5 | Kimi k1.5: Scaling Reinforcement Learning with LLMs | Kimi Team | arXiv:2501.12599 | [A] | 2026-11-17 | 9.5 |

## 术语表

（各波 writing pass 累积：章节首次出现的「中文（English Term）」对在此登记，格式与 models/references.md 术语表一致。）
