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
| pytorch-repo | pytorch/pytorch repo（torch.autograd、torch.optim、torch.utils.checkpoint、torch.nn.utils.clip_grad_norm_、torch.distributed.checkpoint） | PyTorch | https://github.com/pytorch/pytorch | 教学参照 | 2026-11-16 | 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.4 |
| pytorch-ddp | PyTorch Distributed: Experiences on Accelerating Data Parallel Training | Li et al. | arXiv:2006.15704 | [A] | 2026-11-16 | 1.1, 1.5, 4.1 |
| flashattention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | arXiv:2205.14135 | [A] | 2026-11-16 | 1.2 |
| liger-kernel | Liger Kernel: Efficient Triton Kernels for LLM Training | Hsu et al.（LinkedIn） | arXiv:2410.10989 ; https://github.com/linkedin/Liger-Kernel | [A] | 2026-11-16 | 1.2, 3.1 |
| gradient-checkpointing | Training Deep Nets with Sublinear Memory Cost | Chen et al. | arXiv:1604.06174 | [B] | 2026-11-16 | 1.2, 1.4 |
| megatron-sp | Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al. | arXiv:2205.05198 | [A] | 2026-11-16 | 1.4, 5.2, 6.1 |
| mixed-precision | Mixed Precision Training | Micikevicius et al. | arXiv:1710.03740 | [B] | 2026-11-16 | 1.3 |
| fp8-formats | FP8 Formats for Deep Learning | Micikevicius et al. | arXiv:2209.05433 | [B] | 2026-11-16 | 1.3 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 | [A] | 2026-11-16 | 1.3, 2.2, 2.3, 2.4, 4.4, 5.1, 5.3, 5.4, 5.5, 7.2, 7.5 |

### Part 2 · Optimizer 与 LR 动力学

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| polyak1964 | Some Methods of Speeding up the Convergence of Iteration Methods | Polyak | USSR Computational Mathematics and Mathematical Physics 4(5):1-17（无 arXiv） | [B] | 2026-11-16 | 2.1 |
| sutskever2013 | On the Importance of Initialization and Momentum in Deep Learning | Sutskever et al. | ICML 2013（PMLR 28(3):1139-1147，无 arXiv） | [B] | 2026-11-16 | 2.1 |
| adam | Adam: A Method for Stochastic Optimization | Kingma & Ba | arXiv:1412.6980 | [B] | 2026-11-16 | 2.2 |
| adamw | Decoupled Weight Decay Regularization | Loshchilov & Hutter | arXiv:1711.05101 | [A] | 2026-11-16 | 2.2, 2.4 |
| amsgrad | On the Convergence of Adam and Beyond | Reddi et al. | arXiv:1904.09237 | [B] | 2026-11-16 | 2.2 |
| gpt3 | Language Models are Few-Shot Learners | Brown et al. | arXiv:2005.14165 | [B] | 2026-11-16 | 2.2, 2.3, 2.4, 7.2 |
| llama2 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | arXiv:2307.09288 | [A] | 2026-11-16 | 2.2, 7.2 |
| llama3 | The Llama 3 Herd of Models | Grattafiori et al. | arXiv:2407.21783 | [A] | 2026-11-16 | 2.2, 2.3, 2.5, 7.2, 7.3, 7.4, 7.5, 9.1, 9.4 |
| olmo2 | 2 OLMo 2 Furious | Team OLMo et al. | arXiv:2501.00656 | [A] | 2026-11-16 | 2.2, 2.3, 2.4, 4.4, 5.5, 7.2, 7.3 |
| critical-batch | An Empirical Model of Large-Batch Training | McCandlish et al. | arXiv:1812.06162 | [B] | 2026-11-16 | 1.5, 2.1, 2.3 |
| linear-scaling | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Goyal et al. | arXiv:1706.02677 | [B] | 2026-11-16 | 2.1, 2.3 |
| minicpm | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Hu et al. | arXiv:2404.06395 | [A] | 2026-11-16 | 2.3, 7.1 |
| radam | On the Variance of the Adaptive Learning Rate and Beyond | Liu et al. | arXiv:1908.03265 | [A] | 2026-11-16 | 2.3 |
| glm-130b | GLM-130B: An Open Bilingual Pre-trained Model | Zeng et al. | arXiv:2210.02414 | [A] | 2026-11-16 | 2.4, 7.3 |
| st-moe | ST-MoE: Designing Stable and Transferable Sparse Expert Models | Zoph et al. | arXiv:2202.08906 | [B] | 2026-11-16 | 2.4, 3.1 |
| muon-blog | Muon: An optimizer for hidden layers（Keller Jordan 博客，Muon 原始出处） | Jordan | https://kellerjordan.github.io/posts/muon/ | 教学参照 | 2026-11-16 | 2.5 |
| muon-repo | KellerJordan/Muon repo（Newton-Schulz 迭代官方实现） | Jordan | https://github.com/KellerJordan/Muon | 教学参照 | 2026-11-16 | 2.5 |
| moonlight | Muon is Scalable for LLM Training（Moonlight） | Liu et al.（Moonshot AI） | arXiv:2502.16982 ; https://github.com/MoonshotAI/Moonlight | [A] | 2026-11-16 | 2.5, 7.2 |
| kimi-k2 | Kimi K2: Open Agentic Intelligence | Kimi Team | arXiv:2507.20534 | [A] | 2026-11-16 | 2.5, 5.5 |

## 术语表

（各波 writing pass 累积：章节首次出现的「中文（English Term）」对在此登记，格式与 models/references.md 术语表一致。）
