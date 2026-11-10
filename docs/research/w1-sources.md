# W1 调研笔记：P1 Foundations + P2 Transformer Anatomy 事实源

调研日期：2026-11-10（Task 7，W1 research pass）。本文件是 T8–T18（1.1–1.3、2.1–2.8 共 11 章）写作的唯一事实源；写作 agent 不做额外调研，数值一律以本文件与 `models/references.md` 为准。

## 访问路径记录（后续 research pass 复用）

- arXiv：`https://export.arxiv.org/api/query?id_list=...`（**必须 https**，http 无返回）——ID↔标题↔第一作者核对。
- arXiv 全文 HTML：`https://ar5iv.labs.arxiv.org/html/<id>`（定理原文取证用）。
- HF：**huggingface.co 在本环境直连超时（curl 状态 000），一律用镜像 `https://hf-mirror.com/<org>/<model>/raw/main/config.json`**；repo 文件列表用 `https://hf-mirror.com/api/models/<org>/<model>`。
- HF gated 仓库（meta-llama/*、google/gemma-*）config 即使镜像也需鉴权——本轮 Llama/Gemma 数值直接采用 spec 审查（Evaluator A）已核实值，引用时挂论文条目或 canonical config URL。
- openai.com 页面对 curl 返回 403（WAF，浏览器可开）；GPT-2 报告用 CDN PDF（200）：https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GitHub repo 存在性：`https://api.github.com/repos/<org>/<repo>`（本轮 8 个 repo 全 200）。

## 通用锚点 config 数值（已核实）

**Llama-3.1-8B**（spec 审查 Evaluator A 核实，HF gated）：hidden_size=4096，num_hidden_layers=32，num_attention_heads=32，num_key_value_heads=8（GQA 4:1），intermediate_size=14336，rope_theta=500000，vocab_size=128256，max_position_embeddings=131072。head_dim = 4096/32 = 128。tie_word_embeddings=false（Llama 谱系不 tying）。来源：references.md `llama-3.1-8b-config` + `llama3`（arXiv:2407.21783）。

**Llama-3.1-70B**（spec 审查）：hidden 8192，80 层，64 Q heads，8 KV heads。来源：`llama-3.1-70b-config`。

**DeepSeek-V3**（spec 审查 + 本轮 config 复核一致）：61 层，n_routed_experts=256 + n_shared_experts=1，num_experts_per_tok=8，first_k_dense_replace=3（前 3 层 dense），kv_lora_rank=512，qk_rope_head_dim=64，qk_nope_head_dim=128，vocab_size=129280，max_position_embeddings=163840，rope_theta=10000；671B 总参数 / 37B 激活；14.8T tokens 预训练；2.788M H800 GPU-hours（**仅为 paper Table 1 的 pre-training 口径，不含前期研究与 post-training**）。rope_scaling = `{type: yarn, factor: 40, beta_fast: 32, beta_slow: 1, mscale: 1.0, mscale_all_dim: 1.0, original_max_position_embeddings: 4096}`（本轮 hf-mirror 实测，DeepSeek-V3 config 即 YaRN 字段的标准实例）。来源：`deepseekv3`（arXiv:2412.19437）+ `deepseek-v3-config`。

**Qwen3-8B**（本轮 hf-mirror 实测 config.json）：vocab_size=151936，hidden_size=4096，num_hidden_layers=36，num_attention_heads=32，num_key_value_heads=8，head_dim=128，intermediate_size=12288，rope_theta=1000000，max_position_embeddings=40960，rms_norm_eps=1e-6，attention_bias=false，tie_word_embeddings=false。来源：`qwen3-8b-config`。

**Qwen3-8B tokenizer**（本轮实测 tokenizer_config.json + vocab.json + 文件列表）：byte-level BPE，tokenizer_class=Qwen2Tokenizer（与 Qwen2/2.5 同一 tokenizer 家族）；vocab.json 恰 151,643 条 merge 后基础 token；added_tokens_decoder 26 个（id 151643–151668，含 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`）；config vocab_size 151936 = 151643+26 再向上 padding 对齐；bos_token_id=151643（`<|endoftext|>`），eos_token_id=151645（`<|im_end|>`）；repo 内含 vocab.json + merges.txt + tokenizer.json；model_max_length=131072。注意区分：**config.max_position_embeddings=40960（原生 32K 量级）vs tokenizer model_max_length=131072（YaRN 扩展上限）**。

**OLMo-2-7B = allenai/OLMo-2-1124-7B**（本轮 hf-mirror 实测）：hidden_size=4096，num_hidden_layers=32，num_attention_heads=32，**num_key_value_heads=32（纯 MHA，无 GQA）**，intermediate_size=11008，vocab_size=100352，max_position_embeddings=4096，rope_theta=500000，rms_norm_eps=1e-6，attention_bias=false，tie_word_embeddings=false。来源：`olmo-2-7b-config` + `olmo2`（arXiv:2501.00656）。

**GPT-2 small（openai-community/gpt2）**（本轮实测 config）：n_layer=12，n_head=12，n_embd=768，n_ctx=n_positions=1024，vocab_size=50257，bos=eos=50256（`<|endoftext|>`），activation=gelu_new，initializer_range=0.02，layer_norm_epsilon=1e-5（Pre-LN + 末尾 final LN + learned absolute position 的老配方活化石——GPT-2 报告 §2.2：LN 移到每个子块输入端、末尾再加一层 LN；W1 初版误写 Post-LN，2.7 审查轮修正）。来源：`gpt2`（[W]：OpenAI 技术报告 PDF + HF 权重，无 arXiv）。注意 openai.com 正文页 curl 403，引用走 CDN PDF URL。

## 1.1 语言建模的数学形式化

- 可用数值：Llama-3.1-8B vocab 128256 / ctx 131072；Qwen3-8B vocab 151936 / max_position_embeddings 40960；GPT-2 vocab 50257 / ctx 1024——vocab × ctx 的真实对照算例。
- perplexity 跨 tokenizer 不可比的算例素材：同一文本在 50257 vs 128256 vs 151936 vocab 下 token 数不同 → CE/token 与 PPL 不可直接比。
- 来源：kaplan（arXiv:2001.08361）、gpt3（arXiv:2005.14165）、gpt2（[W]，URL 条目）、softmax-bottleneck（arXiv:1711.03953——详拆在 §2.2（spec 的承接分工），1.1 只做一小节引入：perplexity/vocab 讨论顺带点名输出头 rank 限制）。
- 注意：GPT-2 无 arXiv，正文引用格式 `[Radford et al., 2019]` 链到 references.md `gpt2` 条目，证据等级标 [W]；不要给 GPT-2 编 arXiv ID。

## 1.2 参数、计算、内存的账

- 可用数值：Llama-3.1-8B 全套 config（上表）；DeepSeek-V3 671B 总 / 37B 激活 / 61 层 / 14.8T tokens；attention 二次项单算时的 head 数与 head_dim=128。
- per-token 前向 FLOPs ≈ 2N（N=非 embedding 参数）的算例用 Llama-3.1-8B：非 embedding 参数可课堂推导（12Ld² 分解 → 2.1），embedding 128256×4096×2（untied）单算。
- 成本数字口径：2.788M H800 GPU-hours 是 DeepSeek-V3 paper Table 1 的 pre-training 口径；正文引用时必须带此限定，勿写成「总训练成本」。
- 来源：kaplan（附录 FLOPs 账框架）、deepseekv3、llama-3.1-8b-config、deepseek-v3-config。

## 1.3 Scaling Laws

- Chinchilla 联合形式常数（本轮对原文 ar5iv 全文核实，Eq. (10)）：L(N,D) = E + A/N^0.34 + B/D^0.28，E=1.69，A=406.4，B=410.7。compute-optimal：N_opt ∝ C^a、D_opt ∝ C^b，a≈0.50/b≈0.50（论文 Table 3 三方法略有出入，写作以论文为准）。
- Pythia（本轮核实摘要原文）：16 个模型，70M–12B，**每模型 154 个 checkpoints**，公开数据同序训练——scaling/动态研究锚点。注意 spec 旧版曾写 16 个 checkpoints，系错误，已改为 154（spec 附录 Evaluator A S2 修复）。
- OLMo 2：论文 + 权重 + 数据 + 训练日志全开放（arXiv:2501.00656）。
- 其他来源：emergent-abilities（arXiv:2206.07682）、emergence-mirage（arXiv:2304.15004）、deepseekmoe（arXiv:2401.06066，fine-grained MoE scaling）。
- 注意：Kaplan（2001.08361）与 Chinchilla（2203.15556）均无官方开源代码/权重，标 [B]；L(D) 数据项指数 Kaplan vs Chinchilla 结论差异（Kaplan N_opt 增长更快）写作时用论文原图数据，勿凭记忆编系数。

## 2.1 架构总览与设计约束

- 参数量 ≈12Ld² 推导算例：Llama-3.1-8B（L=32, d=4096 → 12×32×4096² ≈ 6.44B；attention 4d²/层 + FFN 8d²/层 的分解与真实 GQA/SwiGLU 有偏差——GQA 使 QKV 投影 <4d²，14336 FFN 使 FFN >8d²，课堂对账素材）。
- OLMo-2-7B 是对照组：MHA（32 KV heads）、intermediate 11008（=2.6875d，接近 Shazeer 8d/3≈10922 的参数等价取整）、untied。
- 设计约束三指标（每层计算/sequential ops/路径长度 O(1)）出自 transformer（arXiv:1706.03762 Table 1）。
- 来源：transformer、llama3、olmo2、llama-3.1-8b-config、olmo-2-7b-config。

## 2.2 Tokenization 与 Embedding

- 算法四源头：bpe（arXiv:1508.07909，频率 merge）、unigram（arXiv:1804.10959——注意该 ID 标题是 *Subword Regularization*，即 Unigram LM 切分的原始论文，SentencePiece 为其官方实现 google/sentencepiece）、wordpiece（arXiv:1609.08144，GNMT 论文，WordPiece 无独立开源实现，标 [B]）、weight-tying（arXiv:1608.05859）。
- vocab 对照表可用硬数字：GPT-2 50257（本轮实测）/ Llama-3.1 128256（spec 审查）/ Qwen3 151936（本轮实测）。
- **注意**：spec 提到的 Llama-2 32K 与 Gemma 256K/262K 本轮未核实（Gemma HF 仓库 gated）——对照表若用 Gemma 数值，引用 gemma2/gemma3 论文条目并注明出处为论文；勿写无出处数字。
- init scale 0.02 惯例：GPT-2 config initializer_range=0.02（本轮实测）、Qwen3-8B 同（本轮实测）。
- softmax bottleneck：softmax-bottleneck（arXiv:1711.03953，本轮核实；输出头 rank 限制 + MoS 解法，官方代码 zihangdai/mos 存在）。
- tokenizer-free 对照：blt（arXiv:2412.09871，本轮核实摘要：byte→按下一 byte 熵动态切 patch；FLOP-controlled scaling study 至 8B 参数 / 4T bytes；官方 repo facebookresearch/blt 存在）。
- 来源：bpe、unigram、wordpiece、weight-tying、softmax-bottleneck、blt、qwen3-8b-config、llama-3.1-8b-config、gpt2。

## 2.3 Attention 机制解剖

- MHA 本体算例用 Llama-3.1-8B 的 32 Q heads / head_dim=128（GQA 部分本章不展开，留给 P4.1）；OLMo-2-7B（32/32 MHA）可作「无 GQA」对照。
- √d_k 推导设定：q,k 分量 iid 零均值单位方差 → Var(q·k)=d_k；softmax Jacobian = diag(p)−ppᵀ。出处均为 transformer 论文设定（缩放在论文 §3.2.1）。
- **head_dim ≠ hidden/heads 的反例预告**（2.8 详讲）：DeepSeek-V3 qk head dim = 128(nope)+64(rope)=192 ≠ hidden/heads。
- 来源：transformer、llama-3.1-8b-config、olmo-2-7b-config。

## 2.4 Causality 与 KV Cache

- KV cache 公式 bytes = 2·L·H_kv·d_head·T·B·b（b=bytes/elem，bf16=2）。
- 实例账（用已核实 config 自算，bf16、T=131072、B=1）：
  - Llama-3.1-8B：2×32×8×128×131072×2 = 16 GiB
  - Llama-3.1-70B：2×80×8×128×131072×2 = 40 GiB
  - 对照 MHA 情形（H_kv=H_q）：若 8B 是 MHA 则 64 GiB——GQA 的 4× 削减一目了然。
- MQA（arXiv:1911.02150）是 H_kv=1 的极端；GQA 论文（2305.13245）属 P4.1，本章只到 MQA 为止。
- 来源：transformer、mqa、llama-3.1-8b-config、llama-3.1-70b-config、olmo-2-7b-config。

## 2.5 位置编码全谱系

- RoPE base 真实值谱：DeepSeek-V3 rope_theta=10000（+YaRN）/ OLMo-2-7B 500000 / Llama-3.1 500000 / Qwen3-8B 1000000 / Qwen2.5-1M 10000000——θ 逐年抬升是长上下文的主手段之一。
- partial RoPE 实例：DeepSeek-V3 qk_rope_head_dim=64 / qk_nope_head_dim=128（只旋 64/192 维，config 已复核）。
- **注意（spec 锚点纠错）**：spec 2.5 写「Qwen2.5（YaRN 字段）」——本轮实测 Qwen/Qwen2.5-7B-Instruct 与 Qwen2.5-7B-Instruct-1M 的 config **均无 rope_scaling 字段**（1M 版只是 rope_theta=10M + max_position_embeddings=1010000）。YaRN config 字段实例一律改用 DeepSeek-V3（上表 rope_scaling 块，本轮实测）。写作时不要再写「Qwen2.5 config 的 YaRN 字段」。
- ALiBi slope：m_h = 2^(−8h/H) 几何级数（alibi 论文，arXiv:2108.12409）；T5 bias log 分桶（t5，arXiv:1910.10683）；NoPE 证据（nope，arXiv:2305.19466，[B]）。
- 来源：rope、alibi、t5、nope、llama-3.1-8b-config、deepseek-v3-config、qwen3-8b-config、olmo-2-7b-config。

## 2.6 FFN 与非线性

- SwiGLU 三矩阵参数等价：与两矩阵 4d FFN 等参数取 d_ff=(8/3)d（glu-variants，arXiv:2002.05202）。
- 真实 intermediate_size 对照（d=4096 家族）：OLMo-2-7B 11008（≈2.69d，贴 8/3 规则、256 倍数取整）/ Qwen3-8B 12288（=3d，本轮实测）/ Llama-3.1 14336（=3.5d，spec 审查）——「Llama-3.1 为什么不是 4×4096=16384」的账用三家对照讲，结论：等参数规则是起点不是铁律，厂商会再加宽。
- GELU（arXiv:1606.08415）：x·Φ(x) 高斯 CDF 视角；FFN-as-KV-memory（ffn-memory，arXiv:2012.14913，Geva 分解）。
- 来源：glu-variants、gelu、ffn-memory、llama-3.1-8b-config、olmo-2-7b-config、qwen3-8b-config。

## 2.7 Normalization 与 Residual

- **Xiong et al.（arXiv:2002.04745）§3 Theorem 1 精确表述**（本轮对 ar5iv 全文逐字取证，这是 spec 2.7 的 ⟨verify⟩ 项）：
  > Theorem 1 (Gradients of the last layer in the Transformer). 假设 ‖x^{post,5}_{L,i}‖² 与 ‖x^{pre}_{L+1,i}‖² 对所有 i 均 (ε,δ)-bounded（ε、δ=δ(ε) 为小量）。则以至少 0.99−δ−ε/(0.9+ε) 的概率：Post-LN（L 层）最后一层 FFN 参数梯度 ‖∂L̃/∂W^{2,L}‖_F ≤ O(d√(ln d))（**与 L 无关**）；Pre-LN ≤ O(d√(ln d / L))（**随 L 衰减**）。
  - 机制（论文 §3 正文）：LN 的 Jacobian 谱范数 ‖J_LN(x)‖ = O(√d/‖x‖)；Pre-LN 中最终 LN 的输入范数随 L 线性增长 → 梯度被 √L 归一化。
  - **spec 措辞纠错**：spec 2.7 写「Post-LN 浅层梯度随深度爆炸」——与原文不符。原文正式定理只覆盖**最后一层 FFN** 的梯度（初始化时）；其他层的结论在 supplementary 的 extended analysis（非正式）：Post-LN **近输出层梯度大、向输入层（浅层）衰减**，Pre-LN 各层大致均匀。warmup 的数学必然性来自「初始化时 Post-LN 输出侧梯度过大且各层不均」，不是「浅层爆炸」。2.7 写作必须按此纠正。
- Gemma 2 logit soft-capping：attn 50.0 / final 30.0，c·tanh(z/c)（gemma2，arXiv:2408.00118，spec 审查）。
- **Gemma 3（gemma3，arXiv:2503.19786，本轮核实 ID↔标题）：以 QK-norm 取代 Gemma 2 的 soft-capping——无 soft-capping，勿沿用 Gemma 2 的 50.0/30.0 数字**；pre+post RMSNorm；5:1 local:global 层交错（4.2 混合层素材）。
- QK-Norm 出处：qk-norm（arXiv:2010.04245，Henry et al.，本轮核实；对 q,k 做 LayerNorm）。
- DeepNorm（arXiv:2203.00555）：残差缩放（microsoft/torchscale 有官方实现）。
- 来源：layernorm、rmsnorm、pre-ln、deepnorm、gemma2、gemma3、qk-norm。
- 注意：google/gemma-* HF 仓库 gated，Gemma 数值引用一律挂论文条目，不挂 config URL。

## 2.8 组装完整 Decoder Block

- 官方实现走读路径（本轮核实存在）：huggingface/transformers repo 内 `src/transformers/models/llama/modeling_llama.py`（LlamaAttention.forward / LlamaMLP / LlamaRMSNorm）。
- config.json ↔ 公式映射表素材：Llama-3.1-8B 全字段（num_hidden_layers=32 / hidden_size=4096 / num_attention_heads=32 / num_key_value_heads=8 / intermediate_size=14336 / rope_theta=500000 / rms_norm_eps / vocab_size=128256）。
- 教学参照：nanogpt（https://github.com/karpathy/nanoGPT，repo 已核实存在；GPT-2 配方：Post-LN + learned abs pos + bias=True + tied——与现代配方逐项对照的好素材，配合 gpt2 config 数值）。
- parallel block（PaLM/GPT-J/Falcon 变体）：h ← h + Attn(LN(x)) + FFN(LN(x))——本章只给公式与梯度/吞吐含义，无单独引用条目（PaLM 论文属后续波次）。
- 常见误解清单素材：head_dim≠hidden/heads 反例（DeepSeek-V3 192）；bias 有无（Llama/Qwen3/OLMo-2 全 false vs GPT-2 true）；tied vs untied（Llama-3.1-8B/Qwen3-8B/OLMo-2-7B 均 false，GPT-2 true）。
- 来源：transformer、llama3、olmo2、nanogpt、llama-3.1-8b-config、olmo-2-7b-config、gpt2（GPT-2 老配方对照数值）。

## 附录：W2+ 提前锁定的数值（spec 审查已核实，本波仅登记不用）

- Mistral-7B sliding_window=4096（SWA 窗口；mistral-7b 条目已入 Part 4 表，被引 4.2）。
- Qwen3 MoE：128 experts top-8（Qwen3-30B-A3B/235B-A22B；4.5/5.3 用）。
- MiniMax-01：lightning attention 与 full attention 7:1 交替（4.4 用；arXiv:2501.08313 条目待 W2 research pass 收录）。
- Gemma 3 5:1 local:global 混合层（4.2 用）。
