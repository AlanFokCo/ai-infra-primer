# W2 调研笔记：P3 Objectives & Variants + P4 Modern Architectures 事实源

调研日期：2026-11-11（W2 Task 1，research pass）。本文件是 T2–T12（3.1–3.3、4.1–4.8 共 11 章）写作的唯一事实源；写作 agent 不做额外调研，数值一律以本文件与 `models/references.md` 为准。

## 访问路径记录（沿用 W1 runbook，本轮新增标注）

- arXiv ID↔标题↔第一作者：`https://export.arxiv.org/api/query?id_list=...`（https 必须）。**注意：默认只返回 10 条，批量查询必须加 `&max_results=N`**（W1 runbook 未记此坑）。
- arXiv 全文 HTML：`https://ar5iv.labs.arxiv.org/html/<id>`——本轮用于 V3.2（DSA 细节取证）与 Kimi Linear（摘要数字取证），均完整转换可用。
- HF：**huggingface.co 直连超时，一律 hf-mirror**（`https://hf-mirror.com/<org>/<model>/raw/main/config.json`）。本轮 hf-mirror 多次间歇 502/503（下午时段），重试即可；`MiniMaxAI/MiniMax-01` 在 HF 上不存在——正确 repo 名是 `MiniMaxAI/MiniMax-Text-01`。
- GitHub repo 存在性：`https://api.github.com/repos/<org>/<repo>`。**raw.githubusercontent.com 本轮直连失败（000）**，取 GitHub 文件用 `api.github.com/repos/<org>/<repo>/contents/<path>` 加 header `Accept: application/vnd.github.raw`。
- qwen.ai/blog 页面为 JS 渲染，curl 抓不到文章列表；Qwen 官方 blog 链接从 HF model card README 里取。

## W2 数学约定（全书 P3/P4 统一，写作 agent 必须遵守）

**Linear attention / delta-rule 状态约定**（沿用 2.7 修复后口径与 GDN 论文 arXiv:2412.06464 的记号，spec 附录 D-S2 裁定）：

- 状态矩阵 $S_t\in\mathbb{R}^{d_v\times d_k}$（每 head 一个；多 head 时逐 head 成立）。
- 读出：$o_t = S_t\,\phi(q_t)$（$q_t\in\mathbb{R}^{d_k}$，$\phi$ 为特征映射；delta-rule 族 $\phi$=恒等）。
- 写入项：$v_t k_t^{\top}$（外积，shape $d_v\times d_k$）。
- 擦除右乘：$S_{t-1}(I-\beta_t k_t k_t^{\top})$——右乘 $(I-\beta_t k_tk_t^{\top})$ 实现定点擦除（沿 $k_t$ 方向）。
- GDN：$S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^{\top})+\beta_t v_tk_t^{\top}$（$\alpha_t$ 标量全局遗忘；$\beta_t$ 写入强度）。
- KDA：把标量 $\alpha_t$ 换为 channel-wise 对角门 $\mathrm{diag}(\alpha_t)$，$\alpha_t\in\mathbb{R}^{d_k}$（**key 侧**逐 channel 遗忘），本书约定布局下写作 $S_t=S_{t-1}\mathrm{diag}(\alpha_t)(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top}$（右乘对角门）。**2026-11-12 纠错**（4.3 fix round 1）：原稿写 $\alpha_t\in\mathbb{R}^{d_v}$ 有误——KDA 论文（arXiv:2510.26692 §3 eq (1)）原式布局 $S\in\mathbb{R}^{d_k\times d_v}$：$S_t=(I-\beta_tk_tk_t^{\top})\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^{\top}$，其 chunkwise 记号 $\Gamma\in\mathbb{R}^{C\times d_k}$ 佐证 $\alpha$ 的 channel 轴是 key 维；GDN-2 论文的退化链（$\alpha_t=\alpha_t\mathbf{1}_{d_k}$ 回 GDN）同证。
- **实现对照（已核实）**：HF transformers `modeling_qwen3_next.py` 的 `torch_recurrent_gated_delta_rule` 把状态存成转置布局 $(d_k, d_v)$（`k.unsqueeze(-1) * delta.unsqueeze(-2)` 左乘写入、`(S * q).sum(dim=-2)` 读出）。正文公式一律用论文约定 $S\in\mathbb{R}^{d_v\times d_k}$；若引用 kernel 代码，必须注明布局互为转置。

**MoE 记号**（4.5 全章统一）：$N_e$=专家总数；$k$=top-k（每 token 激活数）；$f_i$=路由到专家 $i$ 的 token 分数（token fraction）；$P_i$=router 对专家 $i$ 的平均概率；aux loss $=\alpha N_e\sum_i f_i P_i$；aux-loss-free routing 的持久偏置记 $b_i$（更新 $b_i\leftarrow b_i+\Delta\cdot(\mathrm{target}-\mathrm{observed})_i$，DeepSeek-V3 口径）。

**RoPE 缩放记法**（4.6 统一）：$s$=上下文扩展因子（目标长度/原始长度）；$\theta_j$=第 $j$ 个频率分量 base；$\lambda_j$=对应波长。PI：$m\rightarrow m/s$；NTK-aware：base $\rightarrow \theta\cdot s^{|D|/(|D|-2)}$；YaRN：按波长分段的 ramp 系数 $\alpha,\beta$ + attention 温度修正。DeepSeek-V3 config 的 `rope_scaling` 块是 YaRN 字段标准实例（W1 已实测，见 w1-sources.md §2.5）。

**KV cache 公式**：引用 2.4 已建立的公式 bytes $=2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot b$（bf16 时 $b=2$），4.1/4.4/4.6 直接复用不重复推导。

## 3.1 Objective 与 Mask 的统一形式化

- 五种 objective 源头（全部本轮 arXiv API 核实 ID↔标题↔一作）：BERT `bert`（1810.04805，Devlin）、T5 `t5`（1910.10683，Raffel，W1 已收录）、UL2 `ul2`（2205.05131，Tay）、GLM `glm`（2103.10360，Du，blank infilling + 2D 位置编码出处）、ModernBERT `modernbert`（2412.13663，Warner——注意论文标题是 *Smarter, Better, Faster, Longer...*，不叫 "ModernBERT"）。
- masking 代码走读锚点：BERT 官方 `create_pretraining_data.py`（15% 选中、80/10/10 替换策略）在 google-research/bert（repo 本轮核实 200）；GLM infilling 实现在 **THUDM/GLM**（200；**注意 zai-org/GLM 是 404**，spec 5.4 锚点的写法对 2021 GLM 不适用，3.1 引用一律 THUDM/GLM）；T5 span corruption 实现在 google-research/text-to-text-transfer-transformer（200）。
- UL2 无官方开源实现 → 证据等级 [B]，公式按论文拆，勿引用社区复现当官方。
- perplexity 跨 objective 不可比的论证素材：MLM 的 loss 支撑集是 masked 15% token，CLM 是全序列——结合 1.1 的 CE/PPL 定义展开。

## 3.2 Encoder-only 与 Encoder-Decoder

- 现代 encoder 锚点 **ModernBERT**（config 本轮 hf-mirror 实测，2026-11-11）：
  - base：22 层、hidden 768、12 heads、intermediate 1152、vocab 50368、ctx 8192、`global_attn_every_n_layers=3`（每 3 层一个 global attention，其余 local）、`local_attention=128`（local 窗口 ±128）、`global_rope_theta=160000 / local_rope_theta=10000`、`tie_word_embeddings=true`、attention_bias=false。
  - large：28 层、hidden 1024、16 heads、intermediate 2624，其余同 base。
  - config 里 `position_embedding_type="absolute"` 是残留字段，实际用 RoPE（rope_theta 字段为证）——写作勿被误导。
- RoBERTa（1907.11692，Liu）：NSP 失败证据出处（移除 NSP 的 ablation）；ALBERT（1909.11942，Lan）：跨层参数共享 + embedding 分解（V×E→E×H）；ELECTRA（2003.10555，Clark）：RTD 目标，官方 repo google-research/electra（200）；RoBERTa 官方实现在 facebookresearch/fairseq（200）。
- encoder-decoder 锚点：T5（cross-attention 公式源头）；**Whisper**（2212.04356，Radford）——enc-dec 在 ASR 的活例，openai/whisper repo 核实 200。Whisper 详细走读在 P10.2，本章只做 enc-dec 结构例证，勿展开音频细节。

## 3.3 Decoder-only 与 In-Context Learning

- GPT-3（`gpt3`，2005.14165，W1 已收录 [B]）：ICL 原始出处。
- 隐式 GD 理论三篇（本轮核实）：von Oswald `von-oswald`（2212.07677，linear self-attention 层可实现一步 GD 的构造性证明）、Akyürek `akyurek`（2211.15661，线性模型上的学习算法识别）、Dai `dai-icl`（2212.10559，meta-optimizer 视角）。三篇均无官方开源代码 → [B]，按论文拆公式。
- Min et al. `min-icl`（2202.12837）：任务识别视角——label 分布与 input 分布可分开学习的 ablation 出处。
- induction heads：`induction-heads`（transformer-circuits.pub 2022，Olsson et al.，URL 本轮核实 200，无 arXiv → [B]）。previous-token head + induction head 两层回路公式出自该文。
- Pythia（`pythia`，W1 已核实 [A]）：**每模型 154 个 checkpoints**（spec 附录 A-S2 修正后的口径，勿写 16）；induction head 训练相变的实证锚点。EleutherAI/pythia repo 本轮核实 200。
- TransformerLens（`transformerlens`，repo 200）：induction score 测量工具锚点。
- 诚实标注：ICL 理论是开放领域，三层解释（GD/回路/任务识别）不是互斥终论。

## 4.1 KV 压缩与精确 Attention 加速

- MQA（`mqa`，1911.02150，W1 已收录 [B]）；GQA `gqa`（2305.13245，Ainslie，[B]：无官方代码——cache/质量 Pareto 数据表按论文引）。
- MLA 出处：DeepSeek-V2 `deepseekv2`（2405.04434）+ V3 `deepseekv3`（2412.19437，W1 已收录）。数值（W1 已锁定，直接沿用）：kv_lora_rank=512、qk_rope_head_dim=64、qk_nope_head_dim=128 → per-token per-layer cache = 512+64=576（bf16 1152 bytes）；对照 MHA 2·H·d_head（Llama-3.1-8B：2×32×128=8192）。W_UK/W_UV absorption 推导按 V2 论文 §2.1.2。
- FlashAttention `flashattention`（2205.14135）/ FA2 `fa2`（2307.08691，均 Tri Dao，官方 repo Dao-AILab/flash-attention 200）：online softmax（running max/sum、重缩放因子 $e^{m_{old}-m_{new}}$）、tiling、IO 复杂度 $O(T^2d^2/M)$。结论口径：**数学上精确等价**，不是近似。
- GQA 实例账（W1 锁定）：Llama-3.1-70B 8 KV heads（H_kv=8, H_q=64）；Llama-3.1-8B 8/32。

## 4.2 Sparse Attention

- SWA 锚点：Mistral-7B `mistral-7b`（2310.06825，W1 收录）sliding_window=4096（spec 附录 A 核实值，本轮 hf-mirror 间歇故障未复测——沿用即可）；L 层堆叠有效感受野 L·w 的推导本章自做。
- SWA+full 混合：Gemma 3 5:1 local:global（`gemma3`，2503.19786，W1 核实；HF gated，引用挂论文条目）。
- attention sinks：StreamingLLM `streamingllm`（2309.17453，Xiao；官方 repo mit-han-lab/streaming-llm 200）；sink 机制 = causal softmax 归一化下首 token 吸收剩余 mass；常用 4 个 sink token（论文数字）。
- NSA `nsa`（2502.11089，Yuan）：compression + selection + sliding 三分支；**[B]：deepseek-ai/NSA repo 不存在（spec 附录 A-S1），无官方代码，公式按论文拆**。block=64 的硬件对齐设计见论文。
- DSA / DeepSeek-V3.2 `deepseek-v3.2`（2512.02556，本轮 arXiv + ar5iv 全文取证）：lightning indexer 打分 $I_{t,s}=\sum_{j=1}^{H^I} w^I_{t,j}\cdot\mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)$（多头、ReLU 为吞吐考虑、可 FP8）；top-k 检索，**sparse training 阶段每 query 选 2048 个 KV token（论文原文核实）**；主 attention 复杂度 O(L²)→O(L·k)；从 V3.1-Terminus 继续训练：dense warm-up（冻结主模型、只训 indexer，目标 = 对主 attention 分布的 KL 蒸馏）→ sparse 训练（lr 7.3e-6、15000 步、每步 480 条序列、128K 数据）。官方 repo deepseek-ai/DeepSeek-V3.2-Exp（200）；README 有一条实现勘误：indexer 的 RoPE 输入要 non-interleaved 布局、MLA 是 interleaved（2025-11-17 公告，写 kernel 细节时注意）。
- MoBA `moba`（2502.13189，Lu；官方 repo MoonshotAI/MoBA 200）：块划分 + query 级 top-k 块路由、免训练插入、SWA+MoBA hybrid。
- Longformer `longformer`（2004.05150，Beltagy；allenai/longformer 200）：静态 pattern 谱系起点（滑窗 + 任务相关 global token）。

## 4.3 Linear Attention 与 State-Space Models

- kernel 视角源头：Katharopoulos `katharopoulos`（2006.16236；官方代码 idiap/fast-transformers 200）：$\phi(q)^{\top}\phi(k)$ 结合律重排 O(T²d)→O(Td²)。Performer `performer`（2009.14794，Choromanski）：FAVOR+ 随机特征近似 softmax。
- RetNet `retnet`（2307.08621，Sun；microsoft/unilm 200）：指数衰减 γ、parallel/recurrent/chunkwise 三形式。
- DeltaNet `deltanet`（2406.06484，Yang）：状态更新 = $\min_S\|Sk_t-v_t\|^2$ 的一步在线 GD。
- GDN `gdn`（2412.06464，Yang；NVlabs/GatedDeltaNet 200）：$S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top}$（约定见本文件「W2 数学约定」）。
- GDN-2 `gdn2`（2605.22791，Hatamizadeh，2026-05；NVlabs/GatedDeltaNet-2 200）：**erase/write 解耦**。**2026-11-12 ar5iv 全文核实（4.3 fix round 1，升级自摘要口径）**：精确更新式为论文 eq (10)（其布局 $S\in\mathbb{R}^{d_k\times d_v}$）：$S_t=(I-k_t(b_t\odot k_t)^{\top})\mathrm{Diag}(\alpha_t)S_{t-1}+k_t(w_t\odot v_t)^{\top}$——channel-wise erase gate $b_t\in[0,1]^{d_k}$（key 侧：门控旧内容读出的坐标）与 channel-wise write gate $w_t\in[0,1]^{d_v}$（value 侧：门控提交的坐标），保留 KDA 的逐通道衰减 $\mathrm{Diag}(\alpha_t)$（key 侧，$\alpha_t\in\mathbb{R}^{d_k}$）。退化链（论文原文）：$b_t=\beta_t\mathbf{1}_{d_k}$、$w_t=\beta_t\mathbf{1}_{d_v}$ 精确回 KDA；$\alpha_t$ 再标量化回 GDN；$b=0, w=\mathbf{1}, \alpha=\gamma\mathbf{1}$ 即 RetNet（耦合形式表达不了此档，正是解耦动机）。负特征值变体把擦除门放宽到 $[0,2]^{d_k}$（write 门保持 $[0,1]^{d_v}$）。chunkwise：逐通道衰减吸收进非对称 erase 因子，WY 结构与 KDA 同款（UT transform + fused Triton kernel）。实证口径：1.3B 参数 / 100B FineWeb-Edu tokens，对 Mamba-2 / GDN / KDA / Mamba-3 变体总体占优，multi-key 检索任务优势最大。换算到本书约定布局（转置）：$S_t=S_{t-1}\mathrm{diag}(\alpha_t)(I-(b_t\odot k_t)k_t^{\top})+(w_t\odot v_t)k_t^{\top}$。
- KDA / Kimi Linear `kimi-linear`（2510.26692，Kimi Team，2025-10-30；MoonshotAI/Kimi-Linear 200）：channel-wise gating（对角 $\alpha_t$）；chunkwise 用 DPLR 转移矩阵的特化变体（摘要原文核实）。**论文数字（ar5iv 取证）：摘要原文为 KV cache 最多 −75%、1M ctx decode 吞吐 "up to 6×"；精确值 6.3× 出自正文 Fig 1(b)（1M decode 长度的 TPOT 对照）——正文引用时写「最高 6.3×（Fig 1(b)）」或按摘要写 6×，勿写成摘要原数 6.3×；KDA:full-attention 名义 3:1 交错**。
- Mamba `mamba`（2312.00752，Gu & Dao；state-spaces/mamba 200）：S6，selectivity（Δ,B,C 输入依赖）。Mamba-2 `mamba2`（2405.21060，Dao & Gu）：SSD 对偶框架。
- kernel 走读锚点：`fla-repo`（fla-org/flash-linear-attention，200）——GDN/KDA 的 chunkwise 实现都在此（transformers 的 Qwen3-Next 也直接调 fla 的 `chunk_gated_delta_rule`，本轮读码确认）。
- 理论边界（associative recall 下界）：论证素材用 RetNet/GDN/KDA 论文的 retrieval ablation + 4.4 hybrid 动机，数学表述自行推导（固定 d×d 状态的信息论上界）。

## 4.4 Hybrid 架构（linear + full attention 交错）

**Qwen3-Next-80B-A3B-Instruct**（`qwen3-next`，[W]：**无 arXiv 论文**（arXiv 搜索确认无），证据 = HF config + 官方 model card + blog，本轮全部实测 2026-11-11）：

- config.json 实测：48 层、`full_attention_interval=4`（transformers 配置代码确认 layer_types：每第 4 层 full → **36 GDN + 12 full attention**）；512 experts、top-10、`shared_expert_intermediate_size=512`（1 个 shared expert）；hidden 2048、head_dim 256、16 Q heads / 2 KV heads、`partial_rotary_factor=0.25`（RoPE 只旋 64/256 维）；GDN 侧 `linear_num_value_heads=32 / linear_num_key_heads=16 / linear_key_head_dim=128 / linear_value_head_dim=128 / linear_conv_kernel_dim=4`；rope_theta=10⁷、max_position_embeddings=262144；vocab 151936（与 Qwen3 同 tokenizer）。
- model card 数字：80B 总参数 / 3B 激活（非 embedding 79B）；15T tokens 预训练；布局原文「12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))」；ctx 原生 262,144、可扩到 1,010,000；官方 blog `https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd`。
- **zero-centered norm 实现级证据**（transformers `modeling_qwen3_next.py` 读码）：`Qwen3NextRMSNorm` 的 weight 以 `torch.zeros` 初始化、前向 `output * (1.0 + self.weight)`——增益存成相对恒等的偏差（与 2.7 的 zero-centered 定义一致）；另有 `Qwen3NextRMSNormGated`（norm 后乘 SiLU 门）。
- gated attention 实现级证据：attention 输出 `attn_output * torch.sigmoid(gate)`（gate 由 Q 投影一并切出）——即 $o=\sigma(W_gh)\odot\mathrm{attn}$，与 spec 4.4 公式一致。

**Kimi-Linear-48B-A3B-Instruct**（`kimi-linear-config`，config 本轮实测 2026-11-11）：

- 27 层 = **20 KDA + 7 full-attn**：`linear_attn_config.kda_layers=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26]`、`linear_attn_config.full_attn_layers=[4,8,12,16,20,24,27]`——**两列表嵌套在 config.json 的 `linear_attn_config` 字段下**（同层还有 `head_dim=128 / num_heads=32 / short_conv_kernel_size=4`），写作引用 config 字段时给全路径（注意：名义 3:1 交错 + 最后一层额外 full attention，27 层凑不满整数周期——写「3:1」时须带此注脚）。
- full-attn 层用 MLA 且 **`mla_use_nope=true`**（RoPE 关闭——NoPE；KDA 层已破位置对称性的论证素材）；kv_lora_rank=512、qk_rope_head_dim=64、qk_nope_head_dim=128、v_head_dim=128（MLA 字段与 DeepSeek 谱系一致）。
- MoE：256 experts、top-8、1 shared（`num_shared_experts=1`）、`moe_router_activation_func=sigmoid`、`routed_scaling_factor=2.446`、`first_k_dense_replace=1`。
- 其他：hidden 2304、head_dim 72（=2304/32）、intermediate 9216、moe_intermediate_size 1024、rope_theta=10⁴、model_max_length=1048576（1M）、vocab 163840、tie=false。
- 论文（`kimi-linear`）数字：48B 总 / 3B 激活；**训练 token 量分两档（勿写单一数字）**：公平对照实验用 **1.4T tokens**（Kimi Linear / GDN-H / MLA 三模型共享语料与 recipe，论文 Fig 1 及实验节口径）、**最终发布 checkpoint 用 5.7T tokens**（对齐 Moonlight 预训练 token 量；Kimi Linear@5.7T 与 Moonlight 的对照见论文 Appendix D）；KV cache −75%（摘要口径）、decode 吞吐摘要口径 up to 6×（精确值 6.3× 见 Fig 1(b)）、**首称在公平对照下全面超过 full attention**（摘要原文）。

**MiniMax-Text-01**（`minimax01` 2501.08313 + `minimax-text-01-config`，config 本轮实测）：**HF repo 名是 MiniMaxAI/MiniMax-Text-01（MiniMaxAI/MiniMax-01 不存在，GitHub 侧是 MiniMax-AI/MiniMax-01，200）**。80 层，`attn_type_list` 实测 = 70 linear（lightning）+ 10 full（位置 8,16,…,80，1-indexed）→ **7:1**；456B 总 / 45.9B 激活（论文摘要）；32 experts top-2；hidden 6144、64 Q heads / 8 KV heads、head_dim 128、`rotary_dim=64`（partial RoPE 一半维度）；rope_theta=10⁷、max_position_embeddings=10,240,000；vocab 200064；`postnorm=true` + `layernorm_*_alpha≈3.557 / beta=1.0`（DeepNorm 式残差缩放）；训练 ctx 1M、推理外推 4M（摘要原文）。

- Jamba `jamba`（2403.19887，Lieber；HF ai21labs/Jamba-v0.1 本轮 API 核实存在，tag 挂 arXiv:2403.19887）：SSM:attention:MoE 交错，层比例/专家数等细节写作时以论文为准（本轮未逐字段实测 config，hf-mirror 间歇故障）。
- Kimi K3 `kimi-k3`（arXiv:2607.24653，2026-07-27，摘要原文核实）：2.8T 总参数 / 104B 激活；KDA + Attention Residuals；LatentMoE 16/896 routed experts；1M ctx；native vision；发布全部权重；GitHub MoonshotAI/Kimi-K3（200）。4.4 只写架构三件套（KDA/LatentMoE/Attention Residuals）各一段，训练/RL 细节属 training 维度。
- 内存账统一表素材：每层固定 state（Qwen3-Next GDN：32 v-heads×128×128；Kimi KDA：32 heads×128×128）vs full-attn 层 KV（用 2.4 公式逐模型算）。

## 4.5 Mixture-of-Experts 与 MTP

- 谱系源头（本轮核实）：Shazeer 2017 `shazeer-moe`（1701.06538，noisy top-k gating 出处，[B] 无官方代码）；GShard `gshard`（2006.16668，[B]）；Switch `switch`（2101.03961，top-1 + capacity factor；t5x 200 → [A]）；expert choice `expert-choice`（2202.09368，Zhou，[B]）；ST-MoE `stmoe`（2202.08906，Zoph，**z-loss 出处**，[B]）。
- DeepSeekMoE `deepseekmoe`（2401.06066，W1 收录）：fine-grained + shared expert 参数分配公式。DeepSeek-V3 `deepseekv3`（W1 收录）：256 routed + 1 shared、top-8、**aux-loss-free** 偏置更新规则 + seq-mask balancing（公式按 V3 论文 §2.1.2/2.2）。
- Mixtral `mixtral`（2401.04088，Jiang）：8 experts top-2，HF modeling_mixtral 走读锚点。Qwen3 `qwen3`（2505.09388，W1 收录）：Qwen3 MoE 128 experts top-8（spec 附录 A 核实值，4.5 对照表用）。
- MTP：DeepSeek-V3 MTP 模块（附加 block + 输入移位、训练辅助 loss、推理 self-speculative）；Medusa `medusa`（2401.10774，Cai，本轮新增核实；FasterDecoding/Medusa 200）并行多头对照；speculative decoding `speculative-decoding`（2211.17192，Leviathan，[B]）接受率公式出处。
- router 病理（塌缩/利用率不均/振荡）：素材取 Switch + ST-MoE + DeepSeekMoE 论文的诊断讨论，无单一出处，写作时分散引用。
- Qwen3-Next 的 MoE 字段（512 experts top-10 + 1 shared、`norm_topk_prob=true`）可作现代高稀疏度对照（引用 `qwen3-next`，注意其 [W] 等级）。

## 4.6 Long Context

- RoPE 缩放族（本轮核实 ID）：PI `pi`（2306.15595，Chen，[B] 无官方 repo）：$m\rightarrow m/s$；NTK-aware（出处为社区推导，PI 论文同期——写作时按 PI 论文与 YaRN 论文的综述处理，无独立条目）；YaRN `yarn`（2309.00071，Peng；jquesnelle/yarn 200）：波长分段 ramp + 温度修正；LongRoPE `longrope`（2402.13753，Ding，[B]）：逐维搜索。
- 评测：RULER `ruler`（2404.06654，Hsieh；官方 repo **NVIDIA/RULER**（200）——注意 hsiehjackson/RULER 已 301 重定向到 NVIDIA/RULER，引用写新地址）；needle-in-haystack 方法论局限按 RULER 论文 §1 讨论。
- Qwen2.5-1M `qwen2.5-1m`（**arXiv:2501.15383，本轮核实定位**——Qwen2.5-1M Technical Report）：**config 口径沿用 W1 实测纠错（w1-sources.md §2.5）：Qwen2.5-7B-Instruct-1M 无 rope_scaling 字段，只是 rope_theta=10⁷ + max_position_embeddings=1010000；勿写「Qwen2.5 config 的 YaRN 字段」**。dual-chunk attention 出自该报告。
- DeepSeek-V3 的 YaRN config 字段实例（W1 实测）：factor 40、original_max_position_embeddings 4096 → 163840。
- RoPE base 谱系对照（W1 锁定）：DeepSeek-V3 10⁴(+YaRN) / OLMo-2 5×10⁵ / Llama-3.1 5×10⁵ / Qwen3-8B 10⁶ / Qwen2.5-1M 10⁷ / Qwen3-Next 10⁷ / MiniMax-01 10⁷（后两者本轮实测新增）。
- attention dilution（熵稀释）：均匀分布熵 log T 的推导自行做；StreamingLLM `streamingllm` 兼任 sink 证据。
- Kimi-Linear 的 1M ctx 数字（6.3×、cache −75%）见 §4.4，本章 cross-link 不重复。

## 4.7 机制可解释性基础

- induction head：`induction-heads`（同 3.3，[B]）；IOI `ioi`（2211.00593，Wang et al.）：GPT-2 small 上的完整 circuit 案例（redwoodresearch/Easy-Transformer 200 为当时代码基础，现行走读用 TransformerLens）。
- superposition：`toy-models`（transformer-circuits.pub 2022，Elhage et al.，URL 200，[B] 无 arXiv）——随机方向叠加 + 干扰误差公式的出处。
- SAE：`gemma-scope`（2408.05147，Lieberum，本轮核实）——Gemma 2 全尺寸开放 SAE 套件；`saelens`（decoderesearch/SAELens，200——spec 附录 A-S2 已从 jbloomX 改名）。
- 工具锚点：`transformerlens`（3.3 同条）、`pythia`（154 checkpoints，W1 锁定）。
- logit lens / activation patching：方法论概念无单一 arXiv 出处——logit lens 源头是 nostalgebraist 博客（2020），patching 源头可归 IOI 论文方法论章；写作时按「方法描述 + IOI 案例引用」处理，勿杜撰 arXiv 条目。
- 诚实边界：SAE/回路结论的可信度分级一段，按 gemma-scope 与 transformer-circuits.pub 系列的自述局限写。

## 4.8 前沿方向（短篇，标注成熟度）

- 文本 diffusion LLM：`llada`（2502.09992，Nie，本轮核实；ML-GSAI/LLaDA 200）：8B 规模 absorbing-state（mask）diffusion LLM；`sedd`（2310.16834，Lou；louaaron/Score-Entropy-Discrete-Diffusion 200）：score entropy 离散扩散。与 AR 的公式级对比（并行解码/无 KV cache/质量差距）按两篇论文数字。
- differential attention `differential-transformer`（2410.05258，Ye，本轮核实——**此 ID 即 spec 附录 C-S1 从 Pixtral 纠错回收的条目**；实现代码在 microsoft/unilm 的 Diff-Transformer 目录）：双 softmax 相减消噪公式。
- MoD `mod`（2404.02258，Raposo，本轮新增核实，[B] 无官方代码）：按 token 动态跳层的 mixture-of-depths。TTT `ttt`（2407.04620，Sun，本轮新增核实，[B]）：测试时学习的 RNN 层。
- 成熟度标注口径：diffusion LLM = 研究原型→早期工业验证（LLaDA 系）；differential attention = 研究原型；MoD/early-exit = 研究原型；test-time scaling 与 agentic 的架构含义 = 工业现实（K3/V3.2 报告佐证，引 `kimi-k3`/`deepseek-v3.2`）。

## 通用注意事项（写作 agent 必读）

1. **先收录后引用**：正文只能引用 references.md 已收录条目；本轮新增条目的核实日期统一 2026-11-11。
2. **hf-mirror 间歇故障**（本轮多次 502/503）：evaluator 复核 HF config 时重试即可；huggingface.co 直连在本环境不可用。
3. **[B]/[W] 条目写作约束**：NSA、UL2、GQA、von Oswald 等 [B] 条目只有论文，勿引用社区实现当官方；Qwen3-Next 是 [W]（无 arXiv 论文），架构事实以 config/model card 为准，训练配方类断言要降级措辞。
4. **状态约定转置陷阱**：公式用 $S\in\mathbb{R}^{d_v\times d_k}$（论文约定），HF kernel 实现是 $(d_k,d_v)$——4.3/4.4 引用 kernel 代码时必须注明。
5. **勿张冠李戴**：Kimi Linear 的 6.3× 是 decode 吞吐（非 6×）；MiniMax-01 的 HF repo 名带 `-Text-`；GLM 2021 的 repo 在 THUDM 不在 zai-org；RULER 的官方 repo 已迁到 NVIDIA/RULER。
6. **W1 已锁定数值直接沿用**（勿重查）：Llama-3.1-8B/70B config、DeepSeek-V3 config 全套（含 YaRN 字段、MLA 维度、MoE 256+1 top-8、first_k_dense_replace=3、2.788M H800 GPU-hours 全训练口径）、Qwen3-8B config、OLMo-2-7B config、GPT-2 配方、Mistral sliding_window=4096、Gemma 3 5:1、Pythia 154 checkpoints。
