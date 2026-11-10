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
| gpt2 | Language Models are Unsupervised Multitask Learners | Radford et al. | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf ; https://huggingface.co/openai-community/gpt2 | [W] | 2026-11-10 | 1.1, 2.2, 2.8 |
| chinchilla | Training Compute-Optimal Large Language Models | Hoffmann et al. | arXiv:2203.15556 | [B] | 2026-11-10 | 1.3 |
| emergent-abilities | Emergent Abilities of Large Language Models | Wei et al. | arXiv:2206.07682 | [B] | 2026-11-10 | 1.3 |
| emergence-mirage | Are Emergent Abilities of Large Language Models a Mirage? | Schaeffer et al. | arXiv:2304.15004 | [B] | 2026-11-10 | 1.3 |
| deepseekmoe | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Dai et al. | arXiv:2401.06066 | [A] | 2026-11-10 | 1.3 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 | [A] | 2026-11-10 | 1.2 |
| pythia | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | Biderman et al. | arXiv:2304.01373 | [A] | 2026-11-10 | 1.3 |
| llama-3.1-8b-config | Llama-3.1-8B config.json（HF，gated repo） | Meta | https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8 |
| qwen3-8b-config | Qwen3-8B config.json 与 tokenizer 文件（HF） | Qwen | https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 2.2, 2.5, 2.6 |
| deepseek-v3-config | DeepSeek-V3 config.json（HF） | DeepSeek-AI | https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json | [A] | 2026-11-10 | 1.2, 2.5 |

### Part 2 · Transformer Anatomy

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | Attention Is All You Need | Vaswani et al. | arXiv:1706.03762 | [A] | 2026-11-10 | 2.1, 2.3, 2.4, 2.8 |
| llama3 | The Llama 3 Herd of Models | Grattafiori et al. | arXiv:2407.21783 | [A] | 2026-11-10 | 1.3, 2.1, 2.8 |
| olmo2 | 2 OLMo 2 Furious | Team OLMo et al. | arXiv:2501.00656 | [A] | 2026-11-10 | 1.3, 2.1, 2.3, 2.4, 2.6, 2.8 |
| olmo-2-7b-config | OLMo-2-7B（allenai/OLMo-2-1124-7B）config.json | AllenAI | https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json | [A] | 2026-11-10 | 2.1, 2.3, 2.4, 2.6, 2.8 |
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
| glu-variants | GLU Variants Improve Transformer | Shazeer | arXiv:2002.05202 | [B] | 2026-11-10 | 2.6 |
| gelu | Gaussian Error Linear Units (GELUs) | Hendrycks & Gimpel | arXiv:1606.08415 | [B] | 2026-11-10 | 2.6 |
| ffn-memory | Transformer Feed-Forward Layers Are Key-Value Memories | Geva et al. | arXiv:2012.14913 | [B] | 2026-11-10 | 2.6 |
| layernorm | Layer Normalization | Ba et al. | arXiv:1607.06450 | [B] | 2026-11-10 | 2.7 |
| rmsnorm | Root Mean Square Layer Normalization | Zhang & Sennrich | arXiv:1910.07467 | [B] | 2026-11-10 | 2.7 |
| pre-ln | On Layer Normalization in the Transformer Architecture | Xiong et al. | arXiv:2002.04745 | [B] | 2026-11-10 | 2.7 |
| deepnorm | DeepNet: Scaling Transformers to 1,000 Layers | Wang et al. | arXiv:2203.00555 | [A] | 2026-11-10 | 2.7 |
| gemma2 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team | arXiv:2408.00118 | [A] | 2026-11-10 | 2.7 |
| gemma3 | Gemma 3 Technical Report | Gemma Team | arXiv:2503.19786 | [A] | 2026-11-10 | 2.7 |
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
