# references.md — models 维度文献索引与术语表

本文件是 models 维度全部章节的引用唯一事实源。核心规则：**先收录后引用**——条目由各波 research pass（W1 为计划 T7）核实后写入本文件，章节正文才能引用；`scripts/lint-chapter.sh` 检查 4 强制执行：正文出现的 arXiv ID 在本文件 grep 不到即 lint FAIL。写作 agent 不新增条目，只引用已有条目。

## 写入规则

- 条目只由 research pass 写入；「核实日期」= 当次核实 ID / URL 的日期（YYYY-MM-DD）。spec 中 `⟨verify⟩` 标记的条目必须经 WebFetch 复核（arXiv abs 页 / HF config / repo 路径）后才收录。
- 非 arXiv 来源（blog、model card、HF config、repo 文件）在「arXiv ID / URL」字段写完整 URL。lint 只机械核对 arXiv ID，URL 条目的真实性由 research pass 与 evaluator 把关。
- 「证据等级」五档：[A] / [B] / [C] / [W] / 教学参照，定义见 STYLE_GUIDE §6；[W] 级找到正式 paper 后的回升判定同见该节。
- key 为小写 slug，全文件唯一，取论文或模型的通用短名（如 `transformer`、`chinchilla`）。
- 章节写完（或引用关系变化）后回写「被引章节」；一条目被多章引用时逗号分隔累加（如 `2.1, 2.3, 2.4`）。
- 正文引用格式 `[Vaswani et al., 2017]`，链接到本文件对应条目（spec §4.6）。

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
| transformer | Attention Is All You Need | Vaswani et al. | arXiv:1706.03762 | [A] | 2026-11-10 | 2.1, 2.3, 2.4 |
```

## 条目

按 Part 分组；research pass 向对应 Part 的表追加，chapter → sources 映射体现在「被引章节」字段。

> **W1 research pass 收录说明（2026-11-10）**：P1/P2 及 mistral-7b、qwen3、qwen2.5 各表条目的 arXiv ID↔标题已于 2026-11-10 经 arXiv API（export.arxiv.org）全量复核。模型数值类事实（Llama-3.1 系列 / DeepSeek-V3 / Gemma 2-3 / Pythia 154 checkpoints / Mistral sliding_window 等）来源为 spec 附录对抗审查记录（2026-11-10，Evaluator A/D）。新鲜核实项（gemma3、qk-norm、softmax-bottleneck、blt、pythia、gpt2、qwen3-8b-config、olmo-2-7b-config、deepseek-v3-config 的 YaRN 字段）为本轮 curl 实测，证据与访问路径见 `docs/research/w1-sources.md`。

### Part 1 · Foundations

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| kaplan | Scaling Laws for Neural Language Models | Kaplan et al. | arXiv:2001.08361 | [B] | 2026-11-10 | 1.1, 1.2, 1.3 |
| gpt3 | Language Models are Few-Shot Learners | Brown et al. | arXiv:2005.14165 | [B] | 2026-11-10 | 1.1 |
| gpt2 | Language Models are Unsupervised Multitask Learners | Radford et al. | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf ; https://huggingface.co/openai-community/gpt2 | [W] | 2026-11-10 | 1.1, 2.1, 2.2, 2.5, 2.6, 2.7, 2.8 |
| chinchilla | Training Compute-Optimal Large Language Models | Hoffmann et al. | arXiv:2203.15556 | [B] | 2026-11-10 | 1.3 |
| emergent-abilities | Emergent Abilities of Large Language Models | Wei et al. | arXiv:2206.07682 | [B] | 2026-11-10 | 1.3 |
| emergence-mirage | Are Emergent Abilities of Large Language Models a Mirage? | Schaeffer et al. | arXiv:2304.15004 | [B] | 2026-11-10 | 1.3 |
| deepseekmoe | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Dai et al. | arXiv:2401.06066 | [A] | 2026-11-10 | 1.3 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 | [A] | 2026-11-10 | 1.2, 2.5, 2.6 |
| pythia | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | Biderman et al. | arXiv:2304.01373 | [A] | 2026-11-10 | 1.3 |
| llama-3.1-8b-config | Llama-3.1-8B config.json（HF，gated repo） | Meta | https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8 |
| qwen3-8b-config | Qwen3-8B config.json 与 tokenizer 文件（HF） | Qwen | https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 2.2, 2.5, 2.6, 2.7, 2.8 |
| deepseek-v3-config | DeepSeek-V3 config.json（HF） | DeepSeek-AI | https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json | [A] | 2026-11-10 | 1.2, 2.3, 2.5, 2.6, 2.8 |

### Part 2 · Transformer Anatomy

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | Attention Is All You Need | Vaswani et al. | arXiv:1706.03762 | [A] | 2026-11-10 | 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8 |
| llama3 | The Llama 3 Herd of Models | Grattafiori et al. | arXiv:2407.21783 | [A] | 2026-11-10 | 1.3, 2.1, 2.2, 2.7, 2.8 |
| olmo2 | 2 OLMo 2 Furious | Team OLMo et al. | arXiv:2501.00656 | [A] | 2026-11-10 | 1.3, 2.1, 2.3, 2.4, 2.6, 2.7, 2.8 |
| olmo-2-7b-config | OLMo-2-7B（allenai/OLMo-2-1124-7B）config.json | AllenAI | https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json | [A] | 2026-11-10 | 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8 |
| olmo2-modeling | transformers repo：OLMo 2 modeling 代码（modeling_olmo2.py，Olmo2Attention / eager_attention_forward） | HuggingFace | https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py | [A] | 2026-11-10 | 2.3, 2.4, 2.6, 2.7, 2.8 |
| llama-modeling | transformers repo：Llama modeling 代码（modeling_llama.py，LlamaAttention 等） | HuggingFace | https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py | [A] | 2026-11-10 | 2.3, 2.4, 2.5, 2.6, 2.7, 2.8 |
| bpe | Neural Machine Translation of Rare Words with Subword Units | Sennrich et al. | arXiv:1508.07909 | [A] | 2026-11-10 | 2.2 |
| unigram | Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates | Kudo | arXiv:1804.10959 | [A] | 2026-11-10 | 2.2 |
| wordpiece | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation | Wu et al. | arXiv:1609.08144 | [B] | 2026-11-10 | 2.2 |
| weight-tying | Using the Output Embedding to Improve Language Models | Press & Wolf | arXiv:1608.05859 | [B] | 2026-11-10 | 2.2 |
| softmax-bottleneck | Breaking the Softmax Bottleneck: A High-Rank RNN Language Model | Yang et al. | arXiv:1711.03953 | [A] | 2026-11-10 | 1.1, 2.2 |
| blt | Byte Latent Transformer: Patches Scale Better Than Tokens | Pagnoni et al. | arXiv:2412.09871 | [A] | 2026-11-10 | 2.2 |
| mqa | Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | arXiv:1911.02150 | [B] | 2026-11-10 | 2.4 |
| llama-3.1-70b-config | Llama-3.1-70B config.json（HF，gated repo） | Meta | https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json | [A] | 2026-11-10 | 2.4 |
| rope | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | arXiv:2104.09864 | [A] | 2026-11-10 | 2.5 |
| alibi | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | Press et al. | arXiv:2108.12409 | [A] | 2026-11-10 | 2.5 |
| t5 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Raffel et al. | arXiv:1910.10683 | [A] | 2026-11-10 | 2.5 |
| nope | The Impact of Positional Encoding on Length Generalization in Transformers | Kazemnejad et al. | arXiv:2305.19466 | [B] | 2026-11-10 | 2.5 |
| glu-variants | GLU Variants Improve Transformer | Shazeer | arXiv:2002.05202 | [B] | 2026-11-10 | 2.1, 2.6 |
| gelu | Gaussian Error Linear Units (GELUs) | Hendrycks & Gimpel | arXiv:1606.08415 | [B] | 2026-11-10 | 2.6 |
| ffn-memory | Transformer Feed-Forward Layers Are Key-Value Memories | Geva et al. | arXiv:2012.14913 | [B] | 2026-11-10 | 2.6 |
| layernorm | Layer Normalization | Ba et al. | arXiv:1607.06450 | [B] | 2026-11-10 | 2.7 |
| rmsnorm | Root Mean Square Layer Normalization | Zhang & Sennrich | arXiv:1910.07467 | [B] | 2026-11-10 | 2.7 |
| pre-ln | On Layer Normalization in the Transformer Architecture | Xiong et al. | arXiv:2002.04745 | [B] | 2026-11-10 | 2.7 |
| deepnorm | DeepNet: Scaling Transformers to 1,000 Layers | Wang et al. | arXiv:2203.00555 | [A] | 2026-11-10 | 2.7 |
| gemma2 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team | arXiv:2408.00118 | [A] | 2026-11-10 | 2.2, 2.7 |
| gemma3 | Gemma 3 Technical Report | Gemma Team | arXiv:2503.19786 | [A] | 2026-11-10 | 2.2, 2.7 |
| qk-norm | Query-Key Normalization for Transformers | Henry et al. | arXiv:2010.04245 | [B] | 2026-11-10 | 2.7 |
| nanogpt | nanoGPT（教学实现 repo） | Karpathy | https://github.com/karpathy/nanoGPT | 教学参照 | 2026-11-10 | 2.8 |

### Part 3 · Objectives & Variants

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 4 · Modern Architectures

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| mistral-7b | Mistral 7B | Jiang et al. | arXiv:2310.06825 | [A] | 2026-11-10 | 4.2 |

### Part 5 · Case Studies

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3 | Qwen3 Technical Report | Yang et al. | arXiv:2505.09388 | [A] | 2026-11-10 | 5.3 |
| qwen2.5 | Qwen2.5 Technical Report | Qwen Team | arXiv:2412.15115 | [A] | 2026-11-10 | 5.3 |

### Part 6 · Vision Representations

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 7 · Diffusion & Image Generation

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 8 · Multimodal LLMs

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 9 · Video Generation

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 10 · Audio & Speech

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

## 术语表

全书术语累积表：章节首现概念按「中文（English Term）」格式处理（STYLE_GUIDE §1）后在此登记——「中文处理」记录该术语的全书统一写法（译名或保留英文），保证跨章一致。各波次追加。

| 英文术语 | 中文处理 | 首次出现章节 |
| --- | --- | --- |
| vocabulary | 词表（vocabulary）；token 保留英文 | 1.1 |
| tokenizer | 分词器（tokenizer） | 1.1 |
| context window | 上下文窗口 | 1.1 |
| probability simplex | 概率单纯形 | 1.1 |
| output head / unembedding | 输出头 | 1.1 |
| hidden state | 隐状态 | 1.1 |
| support（分布语境） | 支持集 | 1.1 |
| chain rule | 链式法则 | 1.1 |
| autoregressive factorization | 自回归分解 | 1.1 |
| maximum likelihood estimation (MLE) | 极大似然估计（MLE） | 1.1 |
| negative log-likelihood (NLL) | 负对数似然（NLL） | 1.1 |
| cross-entropy (CE) | 交叉熵（cross-entropy） | 1.1 |
| KL divergence | KL 散度 | 1.1 |
| teacher forcing | 教师强制（teacher forcing） | 1.1 |
| exposure bias | 暴露偏差 | 1.1 |
| entropy | 熵 | 1.1 |
| perplexity (PPL) | 困惑度（perplexity） | 1.1 |
| bits per byte | 保留英文（bits per byte） | 1.1 |
| ancestral sampling | 祖先采样 | 1.1 |
| temperature | 温度（temperature） | 1.1 |
| softmax bottleneck | softmax 瓶颈 | 1.1 |
| mixture of softmaxes (MoS) | 保留英文（Mixture of Softmaxes，MoS） | 1.1 |
| KV cache | 保留英文（KV cache） | 1.1 |
| FLOPs / FLOPS | 保留英文（FLOPs 计次数、FLOPS 计速率） | 1.2 |
| fused multiply-add (FMA) | 融合乘加（FMA） | 1.2 |
| multiply-accumulate (MAC) | 保留英文（MAC） | 1.2 |
| multi-head attention (MHA) | 保留英文（MHA） | 1.2 |
| grouped-query attention (GQA) | 保留英文（GQA） | 1.2 |
| gradient checkpointing（= activation checkpointing） | 梯度检查点（亦称激活检查点） | 1.2 |
| arithmetic intensity | 算术强度 | 1.2 |
| model FLOPs utilization (MFU) | 算力利用率（MFU） | 1.2 |
| compute-optimal | 保留英文（compute-optimal，算力最优分配） | 1.3 |
| over-training | 过度训练 | 1.3 |
| emergence / emergent ability | 涌现（emergence）/ 涌现能力（emergent ability） | 1.3 |
| residual stream | 保留英文（residual stream；残差流） | 2.1 |
| Pre-LN / Post-LN | 保留英文（Pre-LN / Post-LN） | 2.1 |
| sequential operations | 串行操作数 | 2.1 |
| byte-pair encoding (BPE) | 保留英文（BPE） | 2.2 |
| WordPiece | 保留英文（WordPiece） | 2.2 |
| Unigram | 保留英文（Unigram） | 2.2 |
| subword regularization | 保留英文（subword regularization） | 2.2 |
| out-of-vocabulary (OOV) | 未登录词（OOV） | 2.2 |
| weight tying | 权重绑定（weight tying） | 2.2 |
| byte latent transformer (BLT) | 保留英文（BLT） | 2.2 |
| attention weights | attention 权重（首现作「注意力权重（attention weights）」；attention 为 STYLE_GUIDE §1 保护术语，不译） | 2.3 |
| bilinear form | 双线性型 | 2.3 |
| saturation（softmax 语境） | 饱和 | 2.3 |
| Jacobian | 雅可比（Jacobian） | 2.3 |
| permutation matrix / permutation equivariance | 置换矩阵 / 置换等变 | 2.3 |
| online softmax | 保留英文（online softmax） | 2.3 |
| causal mask | 因果掩码（causal mask） | 2.4 |
| prefill / decode | 保留英文（prefill / decode） | 2.4 |
| chunked prefill | 保留英文（chunked prefill，分块 prefill） | 2.4 |
| multi-query attention (MQA) | 保留英文（MQA） | 2.4 |
| roofline model | 屋顶线模型（roofline） | 2.4 |
| ridge point | 分界线（ridge point） | 2.4 |
| positional encoding | 位置编码 | 2.5 |
| rotary position embedding (RoPE) | 旋转位置编码（RoPE） | 2.5 |
| relative position bias | 相对位置偏置 | 2.5 |
| extrapolation | 外推 | 2.5 |
| wavelength（RoPE 语境） | 波长 | 2.5 |
| partial / decoupled RoPE | 部分旋转（partial RoPE） | 2.5 |
| NoPE | 无显式位置编码（NoPE） | 2.5 |
| sinusoidal position encoding | 保留英文（sinusoidal） | 2.5 |
| learned absolute position embedding | 可学绝对位置嵌入（learned absolute） | 2.5 |
| ALiBi (attention with linear biases) | 保留英文（ALiBi） | 2.5 |
| position interpolation (PI) | 位置插值（PI） | 2.5 |
| half-split / interleaved | 保留英文（RoPE 两种维度配对约定） | 2.5 |
| position-wise feed-forward network (FFN) | 逐位置前馈网络（FFN） | 2.6 |
| gated linear unit (GLU) | 门控线性单元（GLU） | 2.6 |
| gating | 门控（gating） | 2.6 |
| SiLU / Swish | 保留英文（SiLU / Swish） | 2.6 |
| key-value memory（FFN 语境） | 键值记忆 | 2.6 |
| dead zone | 死区 | 2.6 |
| Gaussian error linear unit (GELU) | 保留英文（GELU） | 2.6 |
| cumulative distribution function (CDF) | 累积分布函数（CDF） | 2.6 |
| SwiGLU / ReGLU / GEGLU | 保留英文（GLU 族变体） | 2.6 |
| mixture-of-experts (MoE) | 保留英文（MoE） | 2.6 |
| normalization | 归一化 | 2.7 |
| layer normalization (LayerNorm) | 层归一化（LayerNorm） | 2.7 |
| batch normalization (BatchNorm) | 批归一化（BatchNorm） | 2.7 |
| RMSNorm | 保留英文（RMSNorm） | 2.7 |
| re-centering / re-scaling | 保留英文（re-centering / re-scaling） | 2.7 |
| warmup | 学习率预热（warmup） | 2.7 |
| QK-Norm | 保留英文（QK-Norm） | 2.7 |
| zero-centered norm | zero-centered 增益（增益存成相对恒等的偏差） | 2.7 |
| soft-capping | 保留英文（soft-capping，软钳制） | 2.7 |
| invariance | 不变性 | 2.7 |
| weight decay | 权重衰减 | 2.7 |
| automatic gain control | 自动增益控制 | 2.7 |
| DeepNorm | 保留英文（DeepNorm） | 2.7 |
| parallel (attention/FFN) block | 并行块（parallel block） | 2.8 |
| tensor parallelism | 张量并行 | 2.8 |
