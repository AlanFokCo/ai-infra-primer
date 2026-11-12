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
| gpt3 | Language Models are Few-Shot Learners | Brown et al. | arXiv:2005.14165 | [B] | 2026-11-10 | 1.1, 3.3 |
| gpt2 | Language Models are Unsupervised Multitask Learners | Radford et al. | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf ; https://huggingface.co/openai-community/gpt2 | [W] | 2026-11-10 | 1.1, 2.1, 2.2, 2.5, 2.6, 2.7, 2.8, 3.1, 3.3, 4.7 |
| chinchilla | Training Compute-Optimal Large Language Models | Hoffmann et al. | arXiv:2203.15556 | [B] | 2026-11-10 | 1.3, 4.5 |
| emergent-abilities | Emergent Abilities of Large Language Models | Wei et al. | arXiv:2206.07682 | [B] | 2026-11-10 | 1.3 |
| emergence-mirage | Are Emergent Abilities of Large Language Models a Mirage? | Schaeffer et al. | arXiv:2304.15004 | [B] | 2026-11-10 | 1.3 |
| deepseekmoe | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Dai et al. | arXiv:2401.06066 | [A] | 2026-11-10 | 1.3, 4.5, 5.2, 5.5, 5.6 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 | [A] | 2026-11-10 | 1.2, 2.5, 2.6, 4.1, 4.2, 4.5, 4.6, 5.2, 5.5, 5.6 |
| pythia | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | Biderman et al. | arXiv:2304.01373 | [A] | 2026-11-10 | 1.3, 3.3, 4.7 |
| llama-3.1-8b-config | Llama-3.1-8B config.json（HF，gated repo） | Meta | https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 4.1, 4.6, 5.1 |
| qwen3-8b-config | Qwen3-8B config.json 与 tokenizer 文件（HF） | Qwen | https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json | [A] | 2026-11-10 | 1.1, 2.2, 2.5, 2.6, 2.7, 2.8, 5.3 |
| deepseek-v3-config | DeepSeek-V3 config.json（HF） | DeepSeek-AI | https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json | [A] | 2026-11-10 | 1.2, 2.3, 2.5, 2.6, 2.8, 4.1, 4.5, 4.6, 5.2, 5.5, 5.6 |

### Part 2 · Transformer Anatomy

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | Attention Is All You Need | Vaswani et al. | arXiv:1706.03762 | [A] | 2026-11-10 | 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 6.1 |
| llama3 | The Llama 3 Herd of Models | Grattafiori et al. | arXiv:2407.21783 | [A] | 2026-11-10 | 1.3, 2.1, 2.2, 2.7, 2.8, 4.6, 5.1, 5.6 |
| olmo2 | 2 OLMo 2 Furious | Team OLMo et al. | arXiv:2501.00656 | [A] | 2026-11-10 | 1.3, 2.1, 2.3, 2.4, 2.6, 2.7, 2.8 |
| olmo-2-7b-config | OLMo-2-7B（allenai/OLMo-2-1124-7B）config.json | AllenAI | https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json | [A] | 2026-11-10 | 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8 |
| olmo2-modeling | transformers repo：OLMo 2 modeling 代码（modeling_olmo2.py，Olmo2Attention / eager_attention_forward） | HuggingFace | https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py | [A] | 2026-11-10 | 2.3, 2.4, 2.6, 2.7, 2.8 |
| llama-modeling | transformers repo：Llama modeling 代码（modeling_llama.py，LlamaAttention 等） | HuggingFace | https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py | [A] | 2026-11-10 | 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 4.1, 5.6 |
| bpe | Neural Machine Translation of Rare Words with Subword Units | Sennrich et al. | arXiv:1508.07909 | [A] | 2026-11-10 | 2.2 |
| unigram | Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates | Kudo | arXiv:1804.10959 | [A] | 2026-11-10 | 2.2 |
| wordpiece | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation | Wu et al. | arXiv:1609.08144 | [B] | 2026-11-10 | 2.2 |
| weight-tying | Using the Output Embedding to Improve Language Models | Press & Wolf | arXiv:1608.05859 | [B] | 2026-11-10 | 2.2, 5.1 |
| softmax-bottleneck | Breaking the Softmax Bottleneck: A High-Rank RNN Language Model | Yang et al. | arXiv:1711.03953 | [A] | 2026-11-10 | 1.1, 2.2 |
| blt | Byte Latent Transformer: Patches Scale Better Than Tokens | Pagnoni et al. | arXiv:2412.09871 | [A] | 2026-11-10 | 2.2 |
| mqa | Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | arXiv:1911.02150 | [B] | 2026-11-10 | 2.4, 4.1 |
| llama-3.1-70b-config | Llama-3.1-70B config.json（HF，gated repo） | Meta | https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json | [A] | 2026-11-10 | 2.4, 4.1, 5.1 |
| rope | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | arXiv:2104.09864 | [A] | 2026-11-10 | 2.5, 4.6, 5.1, 6.1 |
| alibi | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | Press et al. | arXiv:2108.12409 | [A] | 2026-11-10 | 2.5 |
| t5 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Raffel et al. | arXiv:1910.10683 | [A] | 2026-11-10 | 2.5, 3.1, 3.2 |
| nope | The Impact of Positional Encoding on Length Generalization in Transformers | Kazemnejad et al. | arXiv:2305.19466 | [B] | 2026-11-10 | 2.5, 4.4, 5.1 |
| glu-variants | GLU Variants Improve Transformer | Shazeer | arXiv:2002.05202 | [B] | 2026-11-10 | 2.1, 2.6, 5.1 |
| gelu | Gaussian Error Linear Units (GELUs) | Hendrycks & Gimpel | arXiv:1606.08415 | [B] | 2026-11-10 | 2.6 |
| ffn-memory | Transformer Feed-Forward Layers Are Key-Value Memories | Geva et al. | arXiv:2012.14913 | [B] | 2026-11-10 | 2.6 |
| layernorm | Layer Normalization | Ba et al. | arXiv:1607.06450 | [B] | 2026-11-10 | 2.7 |
| rmsnorm | Root Mean Square Layer Normalization | Zhang & Sennrich | arXiv:1910.07467 | [B] | 2026-11-10 | 2.7, 5.1 |
| pre-ln | On Layer Normalization in the Transformer Architecture | Xiong et al. | arXiv:2002.04745 | [B] | 2026-11-10 | 2.7 |
| deepnorm | DeepNet: Scaling Transformers to 1,000 Layers | Wang et al. | arXiv:2203.00555 | [A] | 2026-11-10 | 2.7 |
| gemma2 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team | arXiv:2408.00118 | [A] | 2026-11-10 | 2.2, 2.7, 5.5 |
| gemma3 | Gemma 3 Technical Report | Gemma Team | arXiv:2503.19786 | [A] | 2026-11-10 | 2.2, 2.7, 4.2, 4.6 |
| qk-norm | Query-Key Normalization for Transformers | Henry et al. | arXiv:2010.04245 | [B] | 2026-11-10 | 2.7, 5.3, 5.5 |
| nanogpt | nanoGPT（教学实现 repo） | Karpathy | https://github.com/karpathy/nanoGPT | 教学参照 | 2026-11-10 | 2.8 |

### Part 3 · Objectives & Variants

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| bert | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin et al. | arXiv:1810.04805 | [A] | 2026-11-11 | 3.1, 3.2 |
| bert-repo | BERT 官方 repo（create_pretraining_data.py：15% / 80-10-10 masking 实现） | Google | https://github.com/google-research/bert | [A] | 2026-11-11 | 3.1 |
| roberta | RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al. | arXiv:1907.11692 | [A] | 2026-11-11 | 3.2 |
| albert | ALBERT: A Lite BERT for Self-supervised Learning of Language Representations | Lan et al. | arXiv:1909.11942 | [A] | 2026-11-11 | 3.2 |
| electra | ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators | Clark et al. | arXiv:2003.10555 | [A] | 2026-11-11 | 3.2 |
| electra-repo | ELECTRA 官方 repo（RTD generator/discriminator 实现） | Google | https://github.com/google-research/electra | [A] | 2026-11-11 | 3.2 |
| ul2 | UL2: Unifying Language Learning Paradigms | Tay et al. | arXiv:2205.05131 | [B] | 2026-11-11 | 3.1 |
| glm | GLM: General Language Model Pretraining with Autoregressive Blank Infilling | Du et al. | arXiv:2103.10360 | [A] | 2026-11-11 | 3.1, 5.4 |
| glm-repo | GLM 官方 repo（blank infilling 实现；注意 zai-org/GLM 不存在，2021 代码在 THUDM/GLM） | THUDM | https://github.com/THUDM/GLM | [A] | 2026-11-11 | 3.1, 5.4 |
| glm-130b | GLM-130B: An Open Bilingual Pre-trained Model | Zeng et al. | arXiv:2210.02414 | [A] | 2026-11-11 | 3.1, 5.4 |
| modernbert | Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference | Warner et al. | arXiv:2412.13663 | [A] | 2026-11-11 | 3.1, 3.2 |
| modernbert-config | ModernBERT-base / ModernBERT-large config.json（HF）+ 官方 repo | Answer.AI | https://huggingface.co/answerdotai/ModernBERT-base/blob/main/config.json ; https://github.com/answerdotai/ModernBERT | [A] | 2026-11-11 | 3.2 |
| whisper | Robust Speech Recognition via Large-Scale Weak Supervision | Radford et al. | arXiv:2212.04356 ; https://github.com/openai/whisper | [A] | 2026-11-11 | 3.2 |
| von-oswald | Transformers Learn In-Context by Gradient Descent | von Oswald et al. | arXiv:2212.07677 | [B] | 2026-11-11 | 3.3 |
| akyurek | What Learning Algorithm Is In-Context Learning? Investigations with Linear Models | Akyürek et al. | arXiv:2211.15661 | [B] | 2026-11-11 | 3.3 |
| dai-icl | Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers | Dai et al. | arXiv:2212.10559 | [B] | 2026-11-11 | 3.3 |
| min-icl | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? | Min et al. | arXiv:2202.12837 | [B] | 2026-11-11 | 3.3 |
| induction-heads | In-context Learning and Induction Heads（transformer-circuits.pub，无 arXiv） | Olsson et al. | https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html | [B] | 2026-11-11 | 3.3, 4.7 |
| transformerlens | TransformerLens（回路测量工具 repo；IOI/induction head 实验基础） | Nanda & Bloom | https://github.com/TransformerLensOrg/TransformerLens | [A] | 2026-11-11 | 3.3, 4.7 |

### Part 4 · Modern Architectures

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| mistral-7b | Mistral 7B | Jiang et al. | arXiv:2310.06825 | [A] | 2026-11-10 | 4.2 |
| gqa | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie et al. | arXiv:2305.13245 | [B] | 2026-11-11 | 4.1, 5.1, 5.3, 5.6 |
| deepseekv2 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | arXiv:2405.04434 | [A] | 2026-11-11 | 4.1, 4.5, 5.2, 5.6 |
| flashattention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | arXiv:2205.14135 | [A] | 2026-11-11 | 4.1 |
| fa2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | arXiv:2307.08691 | [A] | 2026-11-11 | 4.1 |
| flash-attention-repo | FlashAttention 官方 repo（kernel 走读锚点） | Dao-AILab | https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-11 | 4.1 |
| nsa | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | Yuan et al. | arXiv:2502.11089 | [B] | 2026-11-11 | 4.2 |
| deepseek-v3.2 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models（DSA 出处） | DeepSeek-AI | arXiv:2512.02556 ; https://github.com/deepseek-ai/DeepSeek-V3.2-Exp | [A] | 2026-11-11 | 4.2, 4.6, 4.8, 5.2, 5.6 |
| moba | MoBA: Mixture of Block Attention for Long-Context LLMs（官方 repo 仅代码、无开放权重，2026-11-12 波末审查降为 [B]） | Lu et al. | arXiv:2502.13189 ; https://github.com/MoonshotAI/MoBA | [B] | 2026-11-11 | 4.2 |
| streamingllm | Efficient Streaming Language Models with Attention Sinks | Xiao et al. | arXiv:2309.17453 ; https://github.com/mit-han-lab/streaming-llm | [A] | 2026-11-11 | 4.2, 4.6 |
| longformer | Longformer: The Long-Document Transformer | Beltagy et al. | arXiv:2004.05150 | [A] | 2026-11-11 | 4.2 |
| katharopoulos | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | Katharopoulos et al. | arXiv:2006.16236 ; https://github.com/idiap/fast-transformers | [A] | 2026-11-11 | 4.3 |
| performer | Rethinking Attention with Performers | Choromanski et al. | arXiv:2009.14794 | [A] | 2026-11-11 | 4.3 |
| retnet | Retentive Network: A Successor to Transformer for Large Language Models | Sun et al. | arXiv:2307.08621 ; https://github.com/microsoft/unilm | [A] | 2026-11-11 | 4.3 |
| deltanet | Parallelizing Linear Transformers with the Delta Rule over Sequence Length | Yang et al. | arXiv:2406.06484 | [A] | 2026-11-11 | 4.3 |
| gdn | Gated Delta Networks: Improving Mamba2 with Delta Rule | Yang et al. | arXiv:2412.06464 ; https://github.com/NVlabs/GatedDeltaNet | [A] | 2026-11-11 | 4.3, 4.4, 5.3 |
| gdn2 | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | Hatamizadeh et al. | arXiv:2605.22791 ; https://github.com/NVlabs/GatedDeltaNet-2 | [A] | 2026-11-11 | 4.3 |
| kimi-linear | Kimi Linear: An Expressive, Efficient Attention Architecture（KDA 出处） | Kimi Team | arXiv:2510.26692 ; https://github.com/MoonshotAI/Kimi-Linear | [A] | 2026-11-11 | 4.3, 4.4, 4.6, 5.5, 5.6 |
| kimi-linear-config | Kimi-Linear-48B-A3B-Instruct config.json（HF：20 KDA + 7 full-attn、mla_use_nope=true） | Moonshot AI | https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json | [A] | 2026-11-11 | 4.4, 4.6, 5.5 |
| mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | arXiv:2312.00752 ; https://github.com/state-spaces/mamba | [A] | 2026-11-11 | 4.3 |
| mamba2 | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | Dao & Gu | arXiv:2405.21060 | [A] | 2026-11-11 | 4.3 |
| fla-repo | flash-linear-attention（fla）官方库：GDN/KDA/DeltaNet chunkwise kernel 走读锚点 | fla-org | https://github.com/fla-org/flash-linear-attention | [A] | 2026-11-11 | 4.3, 4.4, 5.3, 5.6 |
| minimax01 | MiniMax-01: Scaling Foundation Models with Lightning Attention | MiniMax | arXiv:2501.08313 ; https://github.com/MiniMax-AI/MiniMax-01 | [A] | 2026-11-11 | 4.4, 4.6, 5.6 |
| minimax-text-01-config | MiniMaxAI/MiniMax-Text-01 config.json（HF：attn_type_list 70 linear + 10 full） | MiniMax | https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json | [A] | 2026-11-11 | 4.4, 5.6 |
| jamba | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber et al. | arXiv:2403.19887 ; https://huggingface.co/ai21labs/Jamba-v0.1 | [A] | 2026-11-11 | 4.4 |
| qwen3-next | Qwen3-Next-80B-A3B-Instruct model card + 官方 blog + config.json（**无 arXiv 论文**，证据以官方 config/blog 为准） | Qwen | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct ; https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd | [W] | 2026-11-11 | 4.4, 4.5, 4.6, 5.3, 5.6 |
| kimi-k3 | Kimi K3: Open Frontier Intelligence | Kimi Team | arXiv:2607.24653 ; https://github.com/MoonshotAI/Kimi-K3 | [A] | 2026-11-11 | 4.4, 4.8, 5.5, 5.6 |
| shazeer-moe | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Shazeer et al. | arXiv:1701.06538 | [B] | 2026-11-11 | 4.5 |
| switch | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | Fedus et al. | arXiv:2101.03961 ; https://github.com/google-research/t5x | [A] | 2026-11-11 | 4.5, 5.1 |
| gshard | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Lepikhin et al. | arXiv:2006.16668 | [B] | 2026-11-11 | 4.5 |
| expert-choice | Mixture-of-Experts with Expert Choice Routing | Zhou et al. | arXiv:2202.09368 | [B] | 2026-11-11 | 4.5 |
| stmoe | ST-MoE: Designing Stable and Transferable Sparse Expert Models（z-loss 出处） | Zoph et al. | arXiv:2202.08906 | [B] | 2026-11-11 | 4.5 |
| mixtral | Mixtral of Experts | Jiang et al. | arXiv:2401.04088 | [A] | 2026-11-11 | 4.5 |
| medusa | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Cai et al. | arXiv:2401.10774 ; https://github.com/FasterDecoding/Medusa | [A] | 2026-11-11 | 4.5 |
| speculative-decoding | Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | arXiv:2211.17192 | [B] | 2026-11-11 | 4.5, 5.6 |
| pi | Extending Context Window of Large Language Models via Positional Interpolation | Chen et al. | arXiv:2306.15595 | [B] | 2026-11-11 | 4.6 |
| yarn | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | arXiv:2309.00071 ; https://github.com/jquesnelle/yarn | [A] | 2026-11-11 | 4.6, 5.5, 5.6 |
| longrope | LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | Ding et al. | arXiv:2402.13753 | [B] | 2026-11-11 | 4.6 |
| ruler | RULER: What's the Real Context Size of Your Long-Context Language Models? | Hsieh et al. | arXiv:2404.06654 ; https://github.com/NVIDIA/RULER | [A] | 2026-11-11 | 4.6 |
| qwen2.5-1m | Qwen2.5-1M Technical Report | Yang et al. | arXiv:2501.15383 | [A] | 2026-11-11 | 4.6, 5.3 |
| ioi | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small | Wang et al. | arXiv:2211.00593 | [A] | 2026-11-11 | 4.7 |
| gemma-scope | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | Lieberum et al. | arXiv:2408.05147 | [A] | 2026-11-11 | 4.7 |
| saelens | SAELens（SAE 训练/分析库；Gemma Scope 配套工具） | Bloom et al. | https://github.com/decoderesearch/SAELens | [A] | 2026-11-11 | 4.7 |
| toy-models | Toy Models of Superposition（transformer-circuits.pub，无 arXiv） | Elhage et al. | https://transformer-circuits.pub/2022/toy_model/index.html | [B] | 2026-11-11 | 4.7 |
| llada | Large Language Diffusion Models | Nie et al. | arXiv:2502.09992 ; https://github.com/ML-GSAI/LLaDA | [A] | 2026-11-11 | 4.8 |
| sedd | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution | Lou et al. | arXiv:2310.16834 ; https://github.com/louaaron/Score-Entropy-Discrete-Diffusion | [A] | 2026-11-11 | 4.8 |
| differential-transformer | Differential Transformer | Ye et al. | arXiv:2410.05258 ; https://github.com/microsoft/unilm | [A] | 2026-11-11 | 4.8 |
| mod | Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models | Raposo et al. | arXiv:2404.02258 | [B] | 2026-11-11 | 4.8 |
| ttt | Learning to (Learn at Test Time): RNNs with Expressive Hidden States | Sun et al. | arXiv:2407.04620 | [B] | 2026-11-11 | 4.8 |

### Part 5 · Case Studies

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| llama-1 | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | arXiv:2302.13971 | [A] | 2026-11-12 | 5.1 |
| llama-2 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | arXiv:2307.09288 | [A] | 2026-11-12 | 5.1 |
| qwen1 | Qwen Technical Report | Bai et al. | arXiv:2309.16609 | [A] | 2026-11-12 | 5.3 |
| qwen2 | Qwen2 Technical Report | Yang et al. | arXiv:2407.10671 | [A] | 2026-11-12 | 5.3, 5.5 |
| glm-4 | ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools | Team GLM | arXiv:2406.12793 | [A] | 2026-11-12 | 5.4 |
| glm-4.5 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | GLM-4.5 Team | arXiv:2508.06471 | [A] | 2026-11-12 | 5.4, 5.6 |
| glm-4.5v | GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning | GLM-V Team | arXiv:2507.01006 | [A] | 2026-11-12 | 5.4 |
| moonlight | Muon is Scalable for LLM Training（Moonlight 模型出处论文） | Liu et al. | arXiv:2502.16982 | [A] | 2026-11-12 | 5.5, 5.6 |
| kimi-k1.5 | Kimi k1.5: Scaling Reinforcement Learning with LLMs | Kimi Team | arXiv:2501.12599 | [A] | 2026-11-12 | 5.5 |
| kimi-k2 | Kimi K2: Open Agentic Intelligence | Kimi Team | arXiv:2507.20534 | [A] | 2026-11-12 | 5.5, 5.6 |
| llama-4 | Llama 4 官方 blog（Scout/Maverick 发布）+ Scout/Maverick config.json（HF gated，经 unsloth 镜像实测：16/128 experts、no_rope_layers iRoPE、early-fusion vision tower） | Meta | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ ; https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json | [W] | 2026-11-12 | 5.1, 5.6 |
| llama-models-repo | meta-llama/llama-models 官方 repo（Llama 1-4 模型定义与 tokenizer） | Meta | https://github.com/meta-llama/llama-models | [A] | 2026-11-12 | 5.1 |
| qwen1.5 | Qwen1.5 发布博客（**无 arXiv 论文**；谱系定位以博客 + Qwen2 report 回溯为准） | Qwen | https://qwenlm.github.io/blog/qwen1.5/ | [W] | 2026-11-12 | 5.3 |
| qwen3-max | Qwen3-Max 官方公告（API-only 旗舰；**无 arXiv、无开放权重**——HF Qwen/Qwen3-Max 404 实测 2026-11-12；参数规模为厂商宣称） | Qwen | https://qwen.ai/blog?id=qwen3-max | [C] | 2026-11-12 | 5.3 |
| glm-4.6 | GLM-4.6 HF config.json + model card（**无独立 tech report**；架构字段实测同 GLM-4.5，ctx 202752） | Z.ai | https://huggingface.co/zai-org/GLM-4.6/blob/main/config.json | [W] | 2026-11-12 | 5.4, 5.6 |
| glm-4.7 | GLM-4.7 HF config.json + model card（**无 arXiv**；config 实测架构同 4.6） | Z.ai | https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json | [W] | 2026-11-12 | 5.4, 5.6 |
| glm-4-repo | zai-org/GLM-4 官方 repo（GLM-4 系列；**THUDM/GLM-4 已 301 重定向至此**） | Z.ai | https://github.com/zai-org/GLM-4 | [A] | 2026-11-12 | 5.4 |
| glm-4.5-repo | zai-org/GLM-4.5 官方 GitHub repo | Z.ai | https://github.com/zai-org/GLM-4.5 | [A] | 2026-11-12 | 5.4 |
| muon-blog | Muon: An optimizer for hidden layers in neural networks（Muon 原始出处，Keller Jordan 博客，无 arXiv） | Keller Jordan | https://kellerjordan.github.io/posts/muon/ | [W] | 2026-11-12 | 5.5 |
| moonlight-repo | MoonshotAI/Moonlight 官方 repo（Moonlight-16B-A3B 权重 + Muon 分布式实现） | Moonshot AI | https://github.com/MoonshotAI/Moonlight | [A] | 2026-11-12 | 5.5 |
| kimi-k2-config | Kimi-K2-Instruct config.json（HF：architectures=DeepseekV3ForCausalLM、384 routed+1 shared top-8、MLA q_lora_rank=1536、YaRN factor 32） | Moonshot AI | https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json | [A] | 2026-11-12 | 5.5, 5.6 |
| kimi-k2-repo | MoonshotAI/Kimi-K2 官方 repo（K2 model card 与部署说明） | Moonshot AI | https://github.com/MoonshotAI/Kimi-K2 | [A] | 2026-11-12 | 5.5 |
| qwen3 | Qwen3 Technical Report | Yang et al. | arXiv:2505.09388 | [A] | 2026-11-10 | 4.5, 5.3, 5.6 |
| qwen2.5 | Qwen2.5 Technical Report | Qwen Team | arXiv:2412.15115 | [A] | 2026-11-10 | 5.3 |

### Part 6 · Vision Representations

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| vit | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | Dosovitskiy et al. | arXiv:2010.11929 | [A] | 2026-11-12 | 6.1 |
| siglip | Sigmoid Loss for Language Image Pre-Training | Zhai et al. | arXiv:2303.15343 | [A] | 2026-11-12 | 6.1 |
| siglip2 | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features | Tschannen et al. | arXiv:2502.14786 | [A] | 2026-11-12 | 6.1 |
| navit | Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution | Dehghani et al. | arXiv:2307.06304 | [A] | 2026-11-12 | 6.1 |
| swin | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Liu et al. | arXiv:2103.14030 | [A] | 2026-11-12 | 6.1 |
| clip | Learning Transferable Visual Models From Natural Language Supervision | Radford et al. | arXiv:2103.00020 | [A] | 2026-11-12 | 6.1, 6.2 |
| alignment-uniformity | Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere | Wang & Isola | arXiv:2005.10242 | [A] | 2026-11-12 | 6.2 |
| vae | Auto-Encoding Variational Bayes | Kingma & Welling | arXiv:1312.6114 | [A] | 2026-11-12 | 6.3, 7.1 |
| vq-vae | Neural Discrete Representation Learning | van den Oord et al. | arXiv:1711.00937 | [A] | 2026-11-12 | 6.3 |
| vqgan | Taming Transformers for High-Resolution Image Synthesis | Esser et al. | arXiv:2012.09841 | [A] | 2026-11-12 | 6.3 |
| magvit-v2 | Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation | Yu et al. | arXiv:2310.05737 | [A] | 2026-11-12 | 6.3 |
| dc-ae | Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models | Chen et al. | arXiv:2410.10733 | [A] | 2026-11-12 | 6.3, 7.6 |
| vision-transformer-repo | google-research/vision_transformer（ViT 官方 JAX 实现；vit_jax/models_vit.py） | Google | https://github.com/google-research/vision_transformer | [A] | 2026-11-12 | 6.1 |
| big-vision | google-research/big_vision（SigLIP/SigLIP 2/NaViT 官方训练代码；big_vision/models/vit.py） | Google | https://github.com/google-research/big_vision | [A] | 2026-11-12 | 6.1 |
| clip-repo | openai/CLIP 官方 repo（模型与 zero-shot 协议） | OpenAI | https://github.com/openai/CLIP | [A] | 2026-11-12 | 6.2 |
| open-clip | mlfoundations/open_clip（开放复现训练；src/open_clip/loss.py 为 InfoNCE/SigLIP loss 走读锚点） | LAION et al. | https://github.com/mlfoundations/open_clip | [A] | 2026-11-12 | 6.2 |
| taming-transformers-repo | CompVis/taming-transformers（VQGAN 官方实现；taming/models/vqgan.py） | CompVis | https://github.com/CompVis/taming-transformers | [A] | 2026-11-12 | 6.3 |
| efficientvit-repo | mit-han-lab/efficientvit（DC-AE 官方实现；applications/dc_ae） | MIT Han Lab | https://github.com/mit-han-lab/efficientvit | [A] | 2026-11-12 | 6.3 |
| autoencoderkl | diffusers AutoencoderKL 实现（SD/SD3/FLUX VAE 走读锚点；src/diffusers/models/autoencoders/autoencoder_kl.py） | HuggingFace | https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py | [A] | 2026-11-12 | 6.3 |
| vector-quantize-pytorch | lucidrains/vector-quantize-pytorch（VQ/RVQ/LFQ 教学实现） | lucidrains | https://github.com/lucidrains/vector-quantize-pytorch | 教学参照 | 2026-11-12 | 6.3 |

### Part 7 · Diffusion & Image Generation

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| gan | Generative Adversarial Networks | Goodfellow et al. | arXiv:1406.2661 | [A] | 2026-11-12 | 7.1 |
| glow | Glow: Generative Flow with Invertible 1x1 Convolutions | Kingma & Dhariwal | arXiv:1807.03039 | [A] | 2026-11-12 | 7.1 |
| ddpm | Denoising Diffusion Probabilistic Models | Ho et al. | arXiv:2006.11239 | [A] | 2026-11-12 | 7.1, 7.2 |
| iddpm | Improved Denoising Diffusion Probabilistic Models | Nichol & Dhariwal | arXiv:2102.09672 | [A] | 2026-11-12 | 7.2 |
| ddpm-repo | hojonathanho/diffusion（DDPM 官方 TF 实现；diffusion_tf/ 目录） | Jonathan Ho | https://github.com/hojonathanho/diffusion | [A] | 2026-11-12 | 7.2 |
| ddpm-pytorch | lucidrains/denoising-diffusion-pytorch（DDPM 教学实现） | lucidrains | https://github.com/lucidrains/denoising-diffusion-pytorch | 教学参照 | 2026-11-12 | 7.2 |
| score-sde | Score-Based Generative Modeling through Stochastic Differential Equations | Song et al. | arXiv:2011.13456 | [A] | 2026-11-12 | 7.3 |
| edm | Elucidating the Design Space of Diffusion-Based Generative Models | Karras et al. | arXiv:2206.00364 | [A] | 2026-11-12 | 7.3 |
| vincent-2011 | A Connection Between Score Matching and Denoising Autoencoders（Neural Computation 23(7):1661–1674；**非 JMLR**——spec 7.3 勘误） | Vincent | https://doi.org/10.1162/NECO_a_00142 | [A] | 2026-11-12 | 7.3 |
| score-sde-repo | yang-song/score_sde_pytorch（score-SDE 官方 PyTorch 实现） | Yang Song | https://github.com/yang-song/score_sde_pytorch | [A] | 2026-11-12 | 7.3 |
| ddim | Denoising Diffusion Implicit Models | Song et al. | arXiv:2010.02502 | [A] | 2026-11-12 | 7.4 |
| classifier-guidance | Diffusion Models Beat GANs on Image Synthesis（classifier guidance 出处） | Dhariwal & Nichol | arXiv:2105.05233 | [A] | 2026-11-12 | 7.4 |
| cfg | Classifier-Free Diffusion Guidance | Ho & Salimans | arXiv:2207.12598 | [A] | 2026-11-12 | 7.4 |
| consistency-models | Consistency Models | Song et al. | arXiv:2303.01469 | [A] | 2026-11-12 | 7.4 |
| lcm | Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference | Luo et al. | arXiv:2310.04378 | [A] | 2026-11-12 | 7.4 |
| dmd | One-step Diffusion with Distribution Matching Distillation | Yin et al. | arXiv:2311.18828 | [A] | 2026-11-12 | 7.4 |
| dmd2 | Improved Distribution Matching Distillation for Fast Image Synthesis | Yin et al. | arXiv:2405.14867 | [A] | 2026-11-12 | 7.4 |
| rectified-flow | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | Liu et al. | arXiv:2209.03003 | [A] | 2026-11-12 | 7.4, 7.5 |
| dpm-solver | DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps | Lu et al. | arXiv:2206.00927 | [A] | 2026-11-12 | 7.4 |
| ddim-repo | ermongroup/ddim（DDIM 官方实现） | Song et al. | https://github.com/ermongroup/ddim | [A] | 2026-11-12 | 7.4 |
| consistency-models-repo | openai/consistency_models（官方；cm/karras_diffusion.py、cm/losses.py 走读锚点） | OpenAI | https://github.com/openai/consistency_models | [A] | 2026-11-12 | 7.4 |
| dmd2-repo | tianweiy/DMD2（DMD/DMD2 官方实现） | Yin et al. | https://github.com/tianweiy/DMD2 | [A] | 2026-11-12 | 7.4 |
| flow-matching | Flow Matching for Generative Modeling | Lipman et al. | arXiv:2210.02747 | [A] | 2026-11-12 | 7.5 |
| sd3 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | Esser et al. | arXiv:2403.03206 | [A] | 2026-11-12 | 7.5, 7.6, 7.7 |
| flow-matching-repo | facebookresearch/flow_matching（FM 官方库；flow_matching/path/、solver/、loss/ 包结构实测） | Meta | https://github.com/facebookresearch/flow_matching | [A] | 2026-11-12 | 7.5 |
| sd3.5-repo | Stability-AI/sd3.5（SD3.5 官方推理代码；mmditx.py、sd3_impls.py——MMDiT 走读锚点） | Stability AI | https://github.com/Stability-AI/sd3.5 | [A] | 2026-11-12 | 7.5, 7.6, 7.7 |
| ldm | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach et al. | arXiv:2112.10752 | [A] | 2026-11-12 | 7.6 |
| dit | Scalable Diffusion Models with Transformers | Peebles & Xie | arXiv:2212.09748 | [A] | 2026-11-12 | 7.6 |
| pixart-alpha | PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis | Chen et al. | arXiv:2310.00426 | [A] | 2026-11-12 | 7.6 |
| pixart-sigma | PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation | Chen et al. | arXiv:2403.04692 | [A] | 2026-11-12 | 7.6 |
| sana | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers | Xie et al. | arXiv:2410.10629 | [A] | 2026-11-12 | 7.6 |
| dit-repo | facebookresearch/DiT（官方；models.py 含 adaLN-Zero 实现） | Meta | https://github.com/facebookresearch/DiT | [A] | 2026-11-12 | 7.6 |
| latent-diffusion-repo | CompVis/latent-diffusion（LDM 官方；ldm/models/diffusion/ddpm.py、ddim.py 走读锚点） | CompVis | https://github.com/CompVis/latent-diffusion | [A] | 2026-11-12 | 7.6 |
| sgm-repo | Stability-AI/generative-models（SGM：SDXL/SD3 训练代码；sgm/modules/diffusionmodules/ 目录实测） | Stability AI | https://github.com/Stability-AI/generative-models | [A] | 2026-11-12 | 7.6, 7.7 |
| flux-repo | black-forest-labs/flux（官方推理代码；HF 权重 gated——spec 口径 [C]） | Black Forest Labs | https://github.com/black-forest-labs/flux | [C] | 2026-11-12 | 7.6 |
| sana-repo | NVlabs/Sana（SANA 官方实现） | NVIDIA | https://github.com/NVlabs/Sana | [A] | 2026-11-12 | 7.6 |
| pixart-repo | PixArt-alpha/PixArt-alpha（官方，含 PixArt-Σ） | PixArt team | https://github.com/PixArt-alpha/PixArt-alpha | [A] | 2026-11-12 | 7.6 |
| var | Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction | Tian et al. | arXiv:2404.02905 | [A] | 2026-11-12 | 7.7 |
| llamagen | Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation | Sun et al. | arXiv:2406.06525 | [A] | 2026-11-12 | 7.7 |
| infinity | Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis | Han et al. | arXiv:2412.04431 | [A] | 2026-11-12 | 7.7 |
| maskgit | MaskGIT: Masked Generative Image Transformer | Chang et al. | arXiv:2202.04200 | [A] | 2026-11-12 | 7.7 |
| muse | Muse: Text-To-Image Generation via Masked Generative Transformers（**Google 未开源**：google-research/muse 404 实测 2026-11-12，不作代码锚点） | Chang et al. | arXiv:2301.00704 | [B] | 2026-11-12 | 7.7 |
| emu3 | Emu3: Next-Token Prediction is All You Need | Wang et al. | arXiv:2409.18869 | [A] | 2026-11-12 | 7.7 |
| imagen | Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding（Google，无官方代码） | Saharia et al. | arXiv:2205.11487 | [B] | 2026-11-12 | 7.7 |
| sdxl | SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis | Podell et al. | arXiv:2307.01952 | [A] | 2026-11-12 | 7.7 |
| var-repo | FoundationVision/VAR（官方；models/var.py、models/vqvae.py 走读锚点） | FoundationVision | https://github.com/FoundationVision/VAR | [A] | 2026-11-12 | 7.7 |
| llamagen-repo | FoundationVision/LlamaGen（官方） | FoundationVision | https://github.com/FoundationVision/LlamaGen | [A] | 2026-11-12 | 7.7 |
| infinity-repo | FoundationVision/Infinity（官方） | FoundationVision | https://github.com/FoundationVision/Infinity | [A] | 2026-11-12 | 7.7 |
| emu3-repo | baaivision/Emu3（官方） | BAAI | https://github.com/baaivision/Emu3 | [A] | 2026-11-12 | 7.7 |
| sdxl-hf | stabilityai/stable-diffusion-xl-base-1.0（HF；model_index.json 实测：StableDiffusionXLPipeline、双 text encoder CLIP-L+OpenCLIP-G） | Stability AI | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | [A] | 2026-11-12 | 7.7 |

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
| corruption scheme | 腐蚀方案（corruption scheme） | 3.1 |
| blank infilling | 空白填充（blank infilling） | 3.1 |
| sentinel token | 哨兵 token（sentinel token） | 3.1 |
| span corruption | 片段腐蚀（span corruption） | 3.1 |
| whole-word masking | 整词掩码（whole-word masking） | 3.1 |
| denoiser | 去噪器（denoiser） | 3.1 |
| mode-switch token | 模式切换符（mode-switch token） | 3.1 |
| pseudo-likelihood | 伪似然（pseudo-likelihood） | 3.1 |
| pseudo-perplexity | 伪困惑度（pseudo-perplexity） | 3.1 |
| next sentence prediction (NSP) | 下一句预测（next sentence prediction，NSP） | 3.2 |
| replaced token detection (RTD) | 替换 token 检测（replaced token detection，RTD） | 3.2 |
| factorized embedding | embedding 分解（factorized embedding） | 3.2 |
| cross-layer parameter sharing | 跨层参数共享 | 3.2 |
| unpadding | 反填充（unpadding） | 3.2 |
| in-context learning (ICL) | 保留英文（in-context learning，ICL；上下文学习） | 3.3 |
| induction head | 归纳头（induction head） | 3.3 |
| task recognition | 任务识别（task recognition） | 3.3 |
| multi-head latent attention (MLA) | 保留英文（MLA） | 4.1 |
| sparse attention | 稀疏注意力（sparse attention） | 4.2 |
| sliding window attention (SWA) | 滑窗 attention（sliding window attention，SWA） | 4.2 |
| attention sink | 保留英文（attention sink） | 4.2 |
| lightning indexer | 保留英文（lightning indexer） | 4.2 |
| linear attention | 保留英文（linear attention） | 4.3 |
| state-space model (SSM) | 保留英文（state-space model，SSM） | 4.3 |
| feature map（linear attention 语境） | 特征映射（feature map） | 4.3 |
| delta rule | 保留英文（delta rule） | 4.3 |
| chunkwise | 保留英文（chunkwise） | 4.3 |
| WY representation | 保留英文（WY 表示） | 4.3 |
| semiseparable matrix | 半可分（semiseparable）矩阵 | 4.3 |
| associative recall | 联想回忆（associative recall） | 4.3 |
| multi-token prediction (MTP) | 保留英文（Multi-Token Prediction，MTP） | 4.4 |
| self-speculative decoding | 保留英文（self-speculative） | 4.4 |
| affinity（routing 语境） | 亲和度（affinity） | 4.5 |
| load-balancing auxiliary loss | 负载均衡辅助损失（aux loss） | 4.5 |
| aux-loss-free | 保留英文（aux-loss-free） | 4.5 |
| capacity factor | 容量因子（capacity factor） | 4.5 |
| expert parallelism (EP) | 保留英文（expert parallelism，EP） | 4.5 |
| attention dilution | 熵稀释（attention dilution） | 4.6 |
| NTK-aware scaling | 保留英文（NTK-aware） | 4.6 |
| dual-chunk attention (DCA) | 双块注意力（dual-chunk attention，DCA） | 4.6 |
| staged extension | 分阶段扩窗（staged extension） | 4.6 |
| long-document upsampling | 长文档上采样（long-document upsampling） | 4.6 |
| needle-in-a-haystack | 大海捞针（needle-in-a-haystack） | 4.6 |
| effective context | 有效上下文（effective context） | 4.6 |
| logit lens | 保留英文（logit lens） | 4.7 |
| superposition | 叠加（superposition） | 4.7 |
| polysemanticity | 保留英文（polysemanticity，多义性） | 4.7 |
| sparse autoencoder (SAE) | 稀疏自编码器（sparse autoencoder，SAE） | 4.7 |
| circuit（可解释性语境） | 保留英文（circuit；回路） | 4.7 |
| activation patching | 保留英文（activation patching） | 4.7 |
| do-operator | do 算子（do-operator） | 4.7 |
| masked diffusion | 掩码扩散（masked diffusion） | 4.8 |
| absorbing state | 吸收态（absorbing state） | 4.8 |
| concrete score | 具体分数（concrete score） | 4.8 |
| score entropy | 分数熵（score entropy） | 4.8 |
| DSA (DeepSeek Sparse Attention) | 保留英文（DSA） | 4.2 |
| sparse-from-dense | 保留英文（sparse-from-dense） | 4.2 |
| iRoPE (interleaved RoPE) | 保留英文（iRoPE） | 5.1 |
| early fusion / late fusion | 保留英文（early fusion / late fusion） | 5.1 |
| online quantization | 保留英文（online quantization，在线量化） | 5.2 |
| delayed scaling | 保留英文（delayed scaling，延迟缩放） | 5.2 |
| upcycling | 保留英文（upcycling；dense→MoE 初始化路线） | 5.3 |
| MuonClip | 保留英文（MuonClip = Muon + QK-clip） | 5.5 |
| QK-clip | 保留英文（QK-clip） | 5.5 |
| native vision | 原生多模态（native vision） | 5.5 |
| patchify | 保留英文（patchify，图像分块成 patch） | 6.1 |
| GAP (global average pooling) | 全局均值池化（GAP） | 6.1 |
| attention pooling | 保留英文（attention pooling） | 6.1 |
| packing（视觉语境） | 保留英文（packing，patch n' pack） | 6.1 |
| token dropping | 保留英文（token dropping，逐图随机丢 patch） | 6.1 |
| self-distillation | 自蒸馏（self-distillation） | 6.1 |
| contrastive learning | 对比学习（contrastive learning） | 6.2 |
| positive pair / negative sample | 正样本对（positive pair）/ 负样本（negative sample） | 6.2 |
| zero-shot | 零样本（zero-shot） | 6.2 |
| linear probe | 保留英文（linear probe，线性探针） | 6.2 |
| prompt ensembling | 模板集成（prompt ensembling） | 6.2 |
| latent variable model | 潜变量模型（latent variable model） | 6.3 |
| amortize（推断语境） | 摊销（amortize） | 6.3 |
| evidence lower bound (ELBO) | 证据下界（ELBO） | 6.3 |
| reparameterization | 重参数化（reparameterization） | 6.3 |
| posterior collapse | 后验坍缩（posterior collapse） | 6.3 |
| straight-through estimator (STE) | 直通估计器（STE） | 6.3 |
| commitment loss | 保留英文（commitment loss，承诺损失） | 6.3 |
| codebook collapse | 码本坍缩（codebook collapse） | 6.3 |
| EMA (exponential moving average) | 指数滑动平均（EMA） | 6.3 |
| perceptual loss / LPIPS | 感知损失（LPIPS） | 6.3 |
| PatchGAN | 保留英文（PatchGAN） | 6.3 |
| residual vector quantization (RVQ) | 残差向量量化（RVQ） | 6.3 |
| lookup-free quantization (LFQ) | 保留英文（LFQ） | 6.3 |
