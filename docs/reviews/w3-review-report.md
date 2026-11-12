# W3 波审查报告：P5 Case Studies + P6 Vision Representations（9 章）

- 审查日期：2026-11-12；evaluator：独立 fresh-context（W3 对抗审查，立场：假设有错直到一手来源证明无误）
- 审查对象：`models/05-case-studies/01–06`（6 章）+ `models/06-vision-representations/01–03`（3 章）
- 标准：`STYLE_GUIDE.md`、`models/references.md`、`docs/research/w3-sources.md`、spec §5 Part 5-6
- 机械检查：`bash scripts/lint-chapter.sh` 9/9 LINT OK（2026-11-12 实测）

## 一、Findings 表

| # | severity | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S1** | 5.1 §5.1 式 (5) 及 §1 谱系表 Llama-4 两行（「16 top-1（层层）」「128 top-1（隔层）」）；波及 5.6 §1 对照表 Scout 行「16 top-1」与 §2.1「Scout 16/1 = 16×」 | 「Llama-4 取 top-1，每 token 的 FFN 输出为 $y = g_{i^{*}}(x) E_{i^{*}}(x)$」——式 (5) 把 MoE 层输出写成纯 routed 单项，漏掉无条件经过的 shared expert | ① 本章自己引用的官方 blog 原文：「MoE layers use 128 routed experts and a shared expert. Each token is sent to the shared expert and also to one of the 128 routed experts」（ai.meta.com/blog/llama-4-multimodal-intelligence，2026-11-12 复核）；② transformers `modeling_llama4.py`：`Llama4TextMoe.__init__` 第 163 行 `self.shared_expert = Llama4TextMLP(config)`，forward 第 171 行 `out = self.shared_expert(hidden_states)` 再 `add_` routed 和（2026-11-12 复核） | 式 (5) 改为 $y = E_{\mathrm{shared}}(x) + g_{i^{*}}(x) E_{i^{*}}(x)$ 并加符号行；谱系表 MoE 列改「16+1 top-1」「128+1 top-1」；5.6 同步；§7 开放问题 3 措辞从「是否含 shared FFN」降级为「blog 未给逐项数字拆解」（存在性已由 blog+代码锁定，开放的是入账口径）；§6 的 101.7B→109B 差额假说可顺势转正为「shared expert 6.0B + vision tower 0.8B 闭合」并注明出处 |
| F2 | **S2** | 5.3 §2.3（及总结表 upcycling 行） | 「report §2.2.2 的原文只写 leveraging the weights of a dense model——**没有点名源模型**」并据此把 shape 取证当作归属的主证据 | Qwen2 report（arXiv:2407.10671，ar5iv 全文复核 2026-11-12）§2.2.3 明写：「Particularly, Qwen2-57B-A14B is upscaled from Qwen2-7B」——源模型被点名，只是不在 §2.2.2 而在 §2.2.3。本章的 shape 一致性结论（57B-A14B = 3584/28/18944 对齐 Qwen2-7B、排除 Qwen1.5-32B 5120/64/27392）经 HF config 实测成立，但「未点名」为假 | 改为：源 dense 由 report §2.2.3 点名（Qwen2-7B），shape 对账作为一致性互证保留；「Qwen1.5-32B 只作 baseline」的排除论证保留（该句属实，12 处提及均为评测对照）。研究笔记 w3-sources.md §5.3 的「由 Qwen1.5-32B upcycle 初始化」一句亦应一并更正 |
| F3 | **S2** | 5.1 §7 开放问题 2 与 §5.1 末 | 「Llama-4 的 router 均衡机制不可见：config 只锁定 top-1 与 expert 数」 | Scout config（unsloth 镜像 2026-11-12 复核）含 `router_aux_loss_coef = 0.001`、`router_jitter_noise = 0.0`；官方 modeling 代码 `Llama4Router.forward` 用 sigmoid 打分（`torch.nn.functional.sigmoid(router_scores)`）。打分函数形态与 aux-loss 系数字段均可见，不可见的只是训练期是否实际启用与曲线 | 改写开放问题 2：登记 sigmoid 打分（代码级）与 aux_loss_coef=0.001 / jitter=0.0（config 级）两个签名，开放项收窄为「训练期启用与否与均衡效果无公开曲线」 |
| F4 | S3 | 5.1 §3 | 「MHA 下 B=32 的 cache 是权重的 2.3 倍」「7B…对 13.4 GB 权重，B=8 才追平」 | 重算：320 GiB 对 128.5 GiB（=138 GB）为 2.49 倍（GB 口径 343.6/138 = 2.49 同值）；7B 档 12.47 GiB ÷ 2 GiB = 6.2，B=7 已超 | 统一 GiB/GB 口径后改为 ≈2.5 倍与 B=7 |
| F5 | S3 | 5.4 式 (4) | `\text{experts（89 MoE 层 + MTP 层）}`、`\text{attention（92 层 + MTP 层）}` 两处 `\text{}` 内含中文 | STYLE_GUIDE §3.1 探测项 P7b 未复核前 `\text{}` 只放 ASCII | 改为 ASCII（如 `\text{experts (89 MoE + 1 MTP layers)}`）或移出公式进符号行 |
| F6 | S3 | 5.4 §3 | 「4.6 README 只有 58 行：0 个参数数字、6 处指向 GLM-4.5」 | 实测（hf-mirror 2026-11-12）：58 行 ✓、0 参数数字 ✓；但「指向 4.5」按 URL 链接计为 5 处（logo、arXiv 2508.06471、bench 图、GitHub repo、TIR 文档），按文本提及计为 8–9 行——6 处对不上任一口径 | 改为「5 个链接指向 GLM-4.5 资源」 |
| F7 | S3 | 5.1 §6 use_qk_norm 行 | 「q/k 投影后各过一次 RMSNorm」未注明层型条件 | `modeling_llama4.py` 第 331 行：`if self.config.use_qk_norm and self.use_rope: self.qk_norm = ...`——QK-Norm 只在 3:1 中的 RoPE 层启用，NoPE 层无 | 该行补「仅 RoPE 层」四字 |

未发现的问题类别（逐条排查过）：历史叙事违规（P5 特查通过：无「某年某司发布」式背景段、无文化/影响评述段；谱系分段均以机制为依据）、AI 味禁令词（9 章 grep 零命中）、术语翻译违规、证据等级缺失、练习/总结缺失（9 章均 4 题 + 表格总结）、裸「最新/首个」断言（零命中；时效性表述均带日期）。

## 二、Verification log

### A. arXiv ID 全量复核（48/48 = 100%）

方法：`curl -sL "https://export.arxiv.org/api/query?id_list=..."` 分 7 批，逐条比对返回标题与正文/参考列表声称的标题。全部命中，无张冠李戴。关键新条目：

| arXiv ID | 正文声称 | API 返回标题 | 结论 |
| --- | --- | --- | --- |
| 2512.02556 | DeepSeek-V3.2（DSA 出处，5.2/5.6） | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | ✓ |
| 2607.24653 | Kimi K3（2026，5.5/5.6） | Kimi K3: Open Frontier Intelligence（2026-07-27 公开） | ✓ |
| 2507.20534 | Kimi K2: Open Agentic Intelligence | 同名 | ✓ |
| 2502.16982 | Moonlight = Muon is Scalable for LLM Training | 同名（标题核对通过，「勿称 Moonlight paper」成立） | ✓ |
| 2508.06471 / 2507.01006 / 2406.12793 | GLM-4.5 / GLM-4.5V / ChatGLM 代际报告 | 标题逐字命中 | ✓ |
| 2302.13971 / 2307.09288 / 2407.21783 | Llama-1/2/3 herd | 命中 | ✓ |
| 其余 39 个 | —（列表略，含 1312.6114、1711.00937、2012.09841、2310.05737、2410.10733、2103.00020、2005.10242、2010.11929、2103.14030、2307.06304、2303.15343、2502.14786 等） | 全部命中 | ✓ |

### B. HF config / repo / 代码实测（逐项）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| Llama-4 Scout config 全字段（48L/5120/40Q8KV/head_dim 128/16 experts top-1/step 1/no_rope_layers [1,1,1,0]×12/chunk 8192/attn_scale 0.1/floor_scale 8192/use_qk_norm/θ 5e5/max_pos 10485760/vocab 202048/llama3 scaling f16 orig 8192/vision 34L·1408·p14·ps 0.5·16 heads/architectures=Llama4ForConditionalGeneration） | hf-mirror unsloth 镜像 config.json | 与 5.1 §6 逐字段一致 ✓；另发现 `router_aux_loss_coef=0.001`、`router_jitter_noise=0.0`（→ F3）与代码级 shared expert（→ F1） |
| Llama-4 Maverick | 同上 | 128 experts、step 2、use_qk_norm=false、1M ctx、rope_scaling null ✓ |
| Llama-4 对数温度形态 | transformers `modeling_llama4.py` L358-366：`log1p(floor((pos+1)/floor_scale)) * attn_scale + 1.0`，仅 NoPE 层、乘在 query 上 | 5.1 式 (8) 重构形态一致 ✓（pos+1 vs t 的差为索引约定级） |
| Llama4TextL2Norm | 同文件 L105-114：`x * rsqrt(mean(x²)+eps)` 无增益 | = 单位增益 RMSNorm，5.1 §6 表述成立 ✓（但仅 RoPE 层 → F7） |
| Qwen3-235B-A22B-Instruct-2507 config | hf-mirror | 94L/4096/64Q4KV/128 experts top-8/moe_int 1536/`mlp_only_layers=[]`/262144/θ 5e6/151936/bias false/tie false ✓（5.3 谱系表） |
| Qwen3-8B config + tokenizer | 同上 | 36L/4096/32Q8KV/40960/θ 1e6/151936 ✓；vocab.json 实测 **151643** 条 ✓；added_tokens 26 个（id 151643–151668，含 `<think>`/`</think>`）✓；tokenizer_config model_max_length=131072 ✓（5.3 §3 与表注三） |
| Qwen3-Next config | 同上 | 48L、interval 4、2048、16Q2KV、head_dim 256、partial 0.25、θ 1e7、262144、GDN 16/32 头 ×128、conv 4、512 experts top-10、shared_int 512、norm_topk_prob ✓（5.3 §4.1） |
| Qwen3-Next modeling 三处证据 | transformers `modeling_qwen3_next.py` L130/139（zeros 初始化 + `output*(1.0+weight)`）、L300（`attn_output * torch.sigmoid(gate)`，gate 由 q_proj chunk 切出）、L369（`chunk_gated_delta_rule` 走 fla）；`configuration_qwen3_next.py` L131-133（interval 生成 layer_types） | 5.3 §4.2/§4.3 全部坐实 ✓ |
| Qwen2-57B-A14B vs Qwen2-7B vs Qwen1.5-32B shape | 三份 config 实测 | 3584/28/18944 vs 3584/28/18944 vs 5120/64/27392 ✓（5.3 §2.3 的 shape 论证成立；归属表述见 F2） |
| Qwen2.5-7B-1M config | 同上 | max_pos 1010000、θ 1e7、**rope_scaling 字段存在但为 null**——5.3 表注三「无 YaRN ramp」成立 ✓ |
| GLM-4.5/4.6/4.7 三 config 逐字段 diff | 同上 | 4.5→4.6 唯一差异 max_position_embeddings 131072→202752；4.6≡4.7 全字段一致（28 字段全核）✓（5.4 核心断言） |
| GLM-4.6 精读全字段 | 同上 | 92L/5120/96Q8KV/head_dim 128/qk_norm+bias 双 true/160+1 top-8/int 12288 vs moe 1536/first3dense/norm_topk+2.5/n_group=1/nextn=1/θ 1e6/partial 0.5/rope_scaling null/eps 1e-5/silu/151552/tie false/Glm4MoeForCausalLM ✓（5.4 §4 全表一致） |
| GLM-4.6 README | 同上 | 58 行 ✓、0 参数数字 ✓、指针计数口径偏差（→ F6） |
| Kimi-K2-Instruct config | 同上 | architectures=DeepseekV3ForCausalLM ✓、61L/7168/64 heads（kv=64 占位）✓、MLA 五字段逐值同 V3 ✓、384+1 top-8、first_k_dense=1、sigmoid+noaux_tc+norm_topk_prob、rsf 2.827、n_group=1、vocab 163840、131072、θ 5e4、YaRN f32/4096 ✓（5.5 §2.3/§4 全表一致）；nextn=0（无 MTP 签名 ✓，支持 5.6 MTP 列） |
| DeepSeek-V3 config | 同上 | 与 5.2 §6 摘录逐字段一致（含 n_group 8/topk_group 4/rsf 2.5/nextn 1/129280/163840/θ 1e4/YaRN f40 beta 32-1/eps 1e-6/bias false/tie false/bf16）✓ |
| DeepSeek-V2 config | 同上 | 60L/5120/128 头/MLA 五字段/160+2 top-6/first_k_dense=1/8 组选 3/102400/θ 1e4 ✓（5.2 V2 行） |
| Kimi-Linear config | 同上 | 27L/2304/32 头/kda_layers 20/full_attn_layers=[4,8,12,16,20,24,27]（前 24 层严格 3:1 + 第 27 层例外 ✓）/mla_use_nope=true/first_k_dense=1/sigmoid router/model_max_length 1048576/163840/θ 1e4 ✓ |
| Qwen3-Max 存在性 | hf-mirror `api/models/Qwen/Qwen3-Max` | 404 ✓（[C] 定级成立） |
| SD v1.5 VAE | runwayml/stable-diffusion-v1-5 `vae/config.json` | latent_channels=4 ✓、**无 scaling_factor 字段** ✓（6.3 §7.1 出处辨析成立） |
| SD scaling factor 0.18215 | CompVis/stable-diffusion `configs/stable-diffusion/v1-inference.yaml` L17 | `scale_factor: 0.18215` ✓ |
| FLUX scale/shift | black-forest-labs/flux `src/flux/util.py` L326-327 等 | 0.3611 / 0.1159 ✓（6.3 §7.2） |
| GitHub repo 存在性（19 个） | api.github.com | meta-llama/llama-models、deepseek-ai/DeepSeek-V3.2-Exp、zai-org/GLM-4、zai-org/GLM-4.5、MoonshotAI/{Moonlight,Kimi-K2,Kimi-Linear,Kimi-K3}、google-research/{vision_transformer,big_vision}、openai/CLIP、mlfoundations/open_clip、CompVis/{taming-transformers,stable-diffusion}、mit-han-lab/efficientvit、lucidrains/vector-quantize-pytorch、fla-org/flash-linear-attention、THUDM/GLM 全部 200 ✓；**THUDM/GLM-4 → 301 → zai-org/GLM-4** ✓（5.4 参考条目注记属实） |
| 代码路径 | 同上 | vit_jax/models_vit.py、big_vision/models/vit.py、open_clip `src/open_clip/loss.py`（ClipLoss L57、SigLipLoss L384 均在）、taming `taming/models/vqgan.py` + `taming/modules/losses/vqperceptual.py`、efficientvit `applications/dc_ae`、diffusers `autoencoder_kl.py` 全部 200 ✓ |

### C. 论文原文级抽查（ar5iv / PDF 全文取证）

| 章 | 抽查断言 | 一手来源与关键引句 | 结论 |
| --- | --- | --- | --- |
| 5.1 | Llama-1 四档参数表与 1.0T/1.4T tokens | arXiv:2302.13971 Table 2：6.7B/13.0B/32.5B/65.2B、4096–8192、1.0T（小两档）/1.4T（大两档） | ✓ |
| 5.1 | herd 405B 架构表；3.17→3.94 chars/token；六阶段 8K→128K、约 800B tokens | arXiv:2407.21783 Table 3（126L/16384/53248/128Q8KV/θ 5e5）；「improves compression rates … from 3.17 to 3.94 characters per token」；「increased context length gradually in six stages … approximately 800B training tokens」 | ✓ |
| 5.1 | Llama-4 blog 口径 109B/17B Scout、400B/17B Maverick、10M | ai.meta.com blog：「a 17 billion active parameter model with 16 experts」「109 billion total parameters」「industry-leading context window of 10M」「17B active parameters and 400B total parameters」 | ✓（数字全中；blog 同句含 shared expert → F1） |
| 5.2 | 成本三口径 2664K+119K+5K=2788K | arXiv:2412.19437 Table 1 原表数字 | ✓ |
| 5.2 | MTP 接受率 85–90%、约 1.8× decode | 同上：「acceptance rate of the second token prediction ranges between 85% and 90%」「1.8 tim[es]」 | ✓ |
| 5.2 | FP8：E4M3 全量、1×128 / 128×128、online quantization | 同上 §3.3：「we adopt the E4M3 format on all tensors」、tile/block 粒度原句 | ✓ |
| 5.2 | V3.2：DSA top-2048、lr 7.3e-6、15000 步 ×480×128K ≈ 943.7B tokens、V3.1-Terminus 续训起点、仅 DSA 一项架构增量 | arXiv:2512.02556：「select 2048 key-value tokens… learning rate of 7.3e-6… 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens」；「the only architectural modification … is the introduction of DeepSeek Sparse Attention」 | ✓ |
| 5.2 | RoPE 布局勘误（2025-11-17） | deepseek-ai/DeepSeek-V3.2-Exp README L77：「the input tensor to RoPE in the indexer module requires a non-interleaved layout, whereas RoPE in the MLA module expects an interleaved layout」，日期行 2025.11.17 | ✓ |
| 5.3 | Qwen2 upcycling 三处提及、§2.2.2 程序（shuffle + 50% 重初始化） | arXiv:2407.10671 ar5iv：「upcycl」×3 ✓；「parameters are shuffled along the intermediate dimension… 50% of its parameters are randomly reinitialized」✓；但 §2.2.3「Qwen2-57B-A14B is upscaled from Qwen2-7B」点名为 Qwen2-7B | 程序 ✓ / 归属表述 ✗（→ F2） |
| 5.3 | Qwen3：QK-Norm 替代 QKV bias；dense 6 档 + MoE 2 档；235B/22B | arXiv:2505.09388：「we remove QKV-bias used in Qwen2 and introduce QK-Norm」；「6 dense models… and 2 MoE models」；「235B parameters with 22B activated」 | ✓ |
| 5.3 | Qwen3-Next model card：MTP 列装、80B/3B、非 embedding 79B、15T、262144 原生可扩 1,010,000 | HF model card 逐项命中 | ✓ |
| 5.4 | GLM-4.5：355B/32B、23T、ARC、「we add an MoE layer as the MTP layer」、Table 1 口径「含 MTP、不含 word embeddings 与输出层」、QK-Norm Yes | arXiv:2508.06471 摘要 + §2.1 + Table 1 注逐字命中 | ✓ |
| 5.5 | Moonlight：16B/3B、5.7T、sigmoid routing ×2、shared expert ×2、SwiGLU ×2、upcycling ×0、from-scratch + deepseek-v3-small（§3.3） | arXiv:2502.16982 ar5iv 全文关键词计数与原句 | ✓（取证计数复核一致） |
| 5.5 | K2：MuonClip=Muon+QK-clip、1T/32B、15.5T zero loss spike；γ=min(1,τ/Smax) 逐头；MLA 特化（qC/kC 各乘 √γ、qR 乘 γ、共享 kR 不动）；τ=100、附录 τ=30；前 70k 步 12.7% 头触发、其后自退役；「QK-Norm is not applicable to MLA because its Key matrices are not fully materialized」 | arXiv:2507.20534 摘要 + §2.1 + Algorithm 1 + Appendix D 逐句命中 | ✓ |
| 5.5 | Kimi Linear：cache −75%、decode 6.3×（Fig 1(b)）、1.4T 公平档 / 5.7T 对齐 Moonlight（Appendix D）、3:1 | arXiv:2510.26692：「reducing KV cache usage by up to 75%」、Fig 1(b) 1M 处 6.3×、「with an expanded total of 5.7 trillion tokens to match the pretraining tokens of Moonlight」 | ✓ |
| 5.5 | K3：2.8T/104B、LatentMoE 16/896、Attention Residuals、1M ctx、native vision | arXiv:2607.24653 摘要逐句命中（「2.8T… 104 billion activated… native vision… 1-million-token… activates 16 of 896 routed experts」） | ✓ |
| 6.1 | ViT Table 1：B=12L/768/3072/86M、L=24L/1024/4096/307M；Swin MACs 公式与 M=7 | arXiv:2010.11929 Table 1；arXiv:2103.14030：「Ω(MSA)=4hwC²+2(hw)²C」「Ω(W-MSA)=4hwC²+2M²hwC」「M=7 by default」 | ✓ |
| 6.1 | SigLIP：b=−10 初值、动机为负样本失衡 | arXiv:2303.15343 PDF 正文抽取：「We initialize t₀ and b to log 10 and −10 respectively」「the heavy imbalance coming from the many negatives dominates the loss」 | ✓ |
| 6.2 | CLIP：τ init 0.07、clamp 100、batch 32768、400M 对、80 模板、zero-shot 匹配 ResNet-50、50 万 query × 至多 2 万条 | arXiv:2103.00020：「initialized to the equivalent of 0.07… clipped to prevent scaling the logits by more than 100」「minibatch size of 32,768」「400 million (image, text) pairs」「we ensemble 80 different context prompts」「we match the accuracy of the original ResNet-50 on ImageNet zero-shot」「500,000 queries… up to 20,000 (image, text) pairs per query」 | ✓ |
| 6.2 | Wang & Isola Theorem 1 渐近式与 O(M^-1/2)、uniformity 核 t=2、均匀分布为最小化者 | arXiv:2005.10242 Theorem 1 原文（式 (13) 为其逐符号翻译） | ✓ |
| 6.3 | VQ-VAE：β=0.25、EMA γ=0.99、EMA 为 Appendix A.1 替代方案且主实验未用、均匀先验下 KL 为常数、commitment 动机（embedding 空间无尺度约束） | arXiv:1711.00937：「We use β=0.25 in all our experiments」「γ=0.99 to work well in practice」「(not used for the experiments in this work)」「the KL term… is constant」 | ✓ |
| 6.3 | VQGAN：自适应 λ = ∇L_rec/(∇L_GAN+δ)、δ=1e-6、K=1024、f16 | arXiv:2012.09841 eq. (7) 与 δ=10⁻⁶ 原句；「|Z|=1024… sequences of length 16·16」 | ✓ |
| 6.3 | DC-AE：rFID 0.50→7.40（f64）对 0.51→0.19（f8）、generalization penalty、三阶段（p1 低清全量 L1+LPIPS 无 GAN、p2 高清只调中间层、p3 低清只调 decoder 头部上 GAN）、显存 153.98→67.81 GB（1024²·batch 12） | arXiv:2410.10733 Figure 3/5/6 配套正文逐句命中 | ✓ |

### D. 强制独立重推导（W3 子集）

**1. 6.2 InfoNCE — verdict：PASS（含全部关键中间步）**

- 双向 CE：行方向 $L_{i2t}=-\frac1B\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}$、列方向同构（对行下标归一化），总损失 $\frac12$ 平均。对 $S_{ij}$ 求导：行方向 $\frac1B(p_{ij}-\delta_{ij})$、列方向 $\frac1B(p^c_{ij}-\delta_{ij})$，合并式 (7) $=\frac{1}{2B}(p_{ij}+p^c_{ij}-2\delta_{ij})$ ✓（独立求导复现）。
- embedding 梯度（式 (8)）：$\partial L_{i2t}/\partial v_i=\frac{1}{B\tau}[(p_{ii}-1)t_i+\sum_{j\ne i}p_{ij}t_j]=\frac{1}{B\tau}\sum_{j\ne i}p_{ij}(t_j-t_i)$，末步用 $\sum_jp_{ij}=1$ ✓（独立复现，吸引/排斥分解成立）。
- τ 的数学作用：两极限（one-hot vs 1/B）✓；温度梯度恒等式（式 (10)）$\partial L_i/\partial t=\sum_j(p_{ij}-\delta_{ij})S_{ij}=\gamma(\sum_jp_{ij}c_{ij}-c_{ii})$ ✓（链式 $\partial S/\partial t=S$ 独立复现）。
- zero-shot=检索：式 (14) 与式 (3) 同对象（候选集由 batch 换成类别表）✓。
- 数值算例（B=3）全部重算：$p_{11}=0.945$、$L_{i2t}=0.040$、$L_{t2i}=0.019$、$L=0.029$、随机基线 $\log3=1.099$ ✓；Monte Carlo 误差 $1/\sqrt{32767}\approx0.55\%$、$1/\sqrt{255}\approx6.3\%$ ✓。

**2. 6.3 VAE ELBO + STE + RVQ — verdict：PASS**

- Jensen 路径：$\log p=\log\mathbb{E}_q[p(x,z)/q]\ge\mathbb{E}_q[\log p(x,z)/q]$，取等条件 $\Leftrightarrow q=p(z|x)$ ✓。
- KL 路径恒等式：$\log p(x)=\mathrm{ELBO}+\mathrm{KL}(q\|p(z|x))$ ✓（独立展开复现，gap=后验近似误差）。
- Gaussian KL：$\mathrm{KL}(\mathcal N(\mu,\sigma^2)\|\mathcal N(0,1))=\frac12(\mu^2+\sigma^2-\log\sigma^2-1)$ ✓（逐步取期望复现；$\mu{=}0,\sigma{=}1$ 归零校验通过）。
- reparameterization：仿射变换分布正确性 + 逐分量雅可比（$\partial z_i/\partial\mu_i=1$、$\partial z_i/\partial\sigma_i=\varepsilon_i$）✓。
- STE：$z_q^{ST}=z_e+\mathrm{sg}[z_q-z_e]$ 前向 $=z_q$、反向雅可比 $=I$、$\partial L/\partial z_e=\partial L/\partial z_q$ ✓；与 commitment 的误差控制配对论证自洽 ✓。
- EMA↔K-Means：$\sum_{z\in S_i}\|z-e_i\|^2$ 极小解为均值 ✓（求导置零复现）。
- RVQ 递推：$r_{n_q+1}=z_e-\sum_ie_i$、误差=末段残差 ✓；容量 $K^{n_q}$（$1024^8=2^{80}$ ✓）；bitrate $8\times10\times75=6000$ bps ✓。

**3. 5.5 MuonClip qk-clip — verdict：PASS**

- $\gamma^h=\min(1,\tau/S_{\max}^h)$ 与论文 §2.1 逐字一致（先验的 all-head 版与 per-head 版均核对）；监控域 $\max_{X\in\mathcal B}\max_{i,j}$ 与论文定义一致 ✓。
- MLA 特化施加方式：论文原文「qC、kC 各乘 $\sqrt{\gamma_h}$；qR 乘 $\gamma_h$；共享 kR 不动以避免跨头串扰」——章中式 (4) 逐项对应 ✓。
- 对账恒等式独立复现：内容项 $\propto W^{QC}(W^{KC})^\top$ 缩 $\sqrt\gamma\cdot\sqrt\gamma=\gamma$；位置项缩 $\gamma\times1=\gamma$；两项同缩 $\Rightarrow$ 总分缩 $\gamma$，触发后新峰值 $=\gamma\cdot S_{\max}=\tau$，未触发头恒等 ✓。
- 联合二次/各自线性的代数性质（式 (2)）独立验证 ✓。

### E. 规范审查

| 项 | 方法 | 结果 |
| --- | --- | --- |
| lint 6 项 | `scripts/lint-chapter.sh` × 9 | 全过 ✓ |
| AI 味禁令词 | grep 全词表 × 9 章 | 零命中 ✓ |
| 历史叙事（P5 特查） | 逐章通读：无背景/文化/影响评述段；谱系期段（5.4「两段式」等）均以机制（objective、config 字段）为分期依据 | ✓ |
| 模板齐全（总结/练习/参考/metadata 块） | 逐章核对 | 9/9 ✓（练习均 4 题） |
| 篇幅（CJK 口径） | 去代码块/公式后汉字计数 | 3673–6591，全部 ≥3500 ✓（6.2=5606、6.3=6591 满足推导重章软目标） |
| 公式白名单 | grep 违禁环境/命令（align*/equation/\boxed/\color/\operatorname*/\lfloor 等） | 零命中；唯一越界 = 5.4 两处 `\text{}` 内中文（→ F5） |
| 时效性条款 | grep「最新/首个/首次」裸断言 | 零命中（命中的「第一行/第一步」等为序数用法） |
| ASCII 图 ≥1/章 | 逐章核对 | 9/9 ✓ |

### F. 波末时效性（2025-2026 锚点现状）

| 锚点 | 现状核查 | 结论 |
| --- | --- | --- |
| Llama-4 Scout/Maverick | config 经 unsloth 镜像复核仍与正文一致；仍无 arXiv 论文 | [W] 定级与正文断言仍成立 ✓ |
| GLM-4.7 | arXiv 搜索「GLM-4.7」仅第三方评测文、无官方 report ✓；config 与 4.6 逐字段一致（本 evaluator 独立 diff 复核）✓；OpenRouter `z-ai/glm-4.7` ctx 标称 204800、描述「latest flagship… enhanced programming + multi-step reasoning」与 5.4 §3 引文逐字一致 ✓ | [W] 成立 ✓ |
| Qwen3-Max | HF `Qwen/Qwen3-Max` 404（本轮实测）✓；OpenRouter 现有 qwen3.7-max / qwen3.8-max-prime（ctx 1M）✓ | [C] 成立 ✓ |
| Kimi K3 | arXiv:2607.24653 存在且标题命中、官方 repo 200 ✓ | [A] 成立 ✓ |
| 各章 [W]/[C] 分级 | 与上述现状一致；5.3 表注的「Qwen1.5 [W] 无 arXiv」经搜索复核成立 | ✓ |

### G. 数值重算覆盖率

独立重算的账（全部用 config 实测值代入复算）：5.1 式 (1) cache 表（2.5 MiB/320 KiB、10 GiB/1.25 GiB、320/40 GiB）✓、$2Vd$ 三档与 13.1%/1.0% ✓、chars/token 八折账 ✓、式 (4) $\lambda_{\max}$ 两档 ✓、式 (8) $m(10\mathrm M)\approx1.72$ ✓、§6 Scout 复算 101.7B ✓；5.2 671B 五项分解（656.5+11.4+1.19+1.85+0.69≈671.6B）✓、激活 35.6B ✓、$6ND$ 对账 $3.43\times10^{14}$ FLOPS ✓、DSA 144 MB / 9.2 GB ✓；5.3 式 (1) 47 GiB / 752 GiB / 144 KiB / 5.6 GiB ✓、$|z|\le\sqrt{128}\approx11.3$ ✓、词表三笔账（1.24B、15%/0.5%/0.8%、ln 151936≈11.9、输出头 8%）✓、GDN 状态 1 MiB/层 ✓；5.4 式 (1) 564/77/7.3× ✓、式 (2) 368 KiB 与 71.1 GiB/854 GiB ✓、式 (3) $\lambda_{\max}\approx4.1$M 与 5.0% ✓、式 (4) 355.2B 闭合（341.9+12.7+0.57+0.07）✓、激活 32.4B/弃 MTP 后 32.0B ✓；5.5 101.1M/层 attention 逐项 ✓、1T 复算 1026B ✓、激活 30.6B ✓、68.6 KiB/9.2 GB ✓、Kimi Linear 7.9 KiB/7.9 GiB/30.4 GiB/3.9× ✓；5.6 五条斜率与 1M 整窗（192/68.6/24/368 GiB、1.9 TiB）✓、打分 $10^{12}$ FLOPs/token ✓、FLOPs 倍数 16×–56× ✓；6.1 patch 嵌入 0.59M 与 $2HWCd$ ✓、Swin 1/64 ✓、NaViT 1152 vs 772 ✓；6.2 全部数值 ✓；6.3 $2^{80}$/6000 bps/0.18215 互倒 ✓。唯二不符：F4 的两个商（2.3 vs 2.49；B=8 vs B=7）——单位混用导致的偏一格，列入 S3。

抽查总覆盖：arXiv ID 48/48（100%）；HF config 精读章的全部逐字段表（5.1 §6、5.2 §6、5.4 §4、5.5 §4）100% 字段复核；论文级断言抽查 24 项命中 23（1 项不符 → F2）；repo/代码路径 19+8 全部命中。

## 三、一句话总结

W3 九章的证据纪律与推导质量整体过硬——48 个 arXiv ID、四套 config 精读、三份强制重推导全部独立复核通过——但 5.1 的 Llama-4 MoE 公式漏掉了官方 blog 与官方代码都明写的 shared expert（S1），5.3 对 Qwen2 report 的「未点名源模型」表述与该报告 §2.2.3 原文直接相反（S2），两处都必须修复。

## 四、放行结论（初轮）

**有条件放行**：5.1 含 S1（F1）+ S2（F3），修复后需复审该章；5.3 含 S2（F2），必须修复；其余 7 章（5.2、5.4、5.5、5.6、6.1、6.2、6.3）无 S1/S2，放行。S3 共 4 条（F4–F7）按协议累计、下一波复审时统一清零。

## 五、复审记录（2026-11-12，修复轮 0765e41..661392e）

复审对象：5.1 全文重读（当前 HEAD）+ 5.3 / 5.6 修复 diff 抽查。三章 lint 全部 LINT OK。

| finding | verdict | 核验细节 |
| --- | --- | --- |
| F1（S1，5.1 shared expert） | **已修复，关闭** | 式 (5) 现为 $y = g_{i^{*}}E_{i^{*}}(x) + E_s(x)$，$E_s$ 符号行注明「不经路由、不乘门控」，正文引官方 blog 原句与 modeling 代码双证；谱系表改「16+1 top-1」「128+1 top-1」；§5 diff 表、§5.1 标题、§5.2 图、§6 字段表（num_local_experts / num_experts_per_tok 两行均注明 shared 不在字段内）、总结表同步。§6 参数账重算核验：routed 96.6B + shared 6.0B + attn 3.0B + 词表 2.07B = 107.7B，加 vision tower ≈0.8B 对 109B 闭合（残差 ≈0.5B）；active 侧 3.0+12.1+2.07 ≈ 17.2B ≈ 17B ✓。§7 问题 3 已改写为余量归属问题 ✓ |
| F3（S2，5.1 开放问题 2） | **已修复，关闭** | 现文「训练侧行为不可见：打分形态是可见的——config 有 router_aux_loss_coef=0.001 与 jitter=0.0，官方实现用 sigmoid 亲和度；但负载分布、塌缩控制与系数调度无 paper 可查」——与建议修正一致；§5.1 同步补一句打分形态登记 ✓ |
| F2（S2，5.3 upcycling 归属；连带复核 23fee78） | **已修复，关闭** | §2.3 现以 report §2.2.3 明文「upscaled from Qwen2-7B」为主证据、shape 对账降为互证，并补齐论文的复制 $\lceil nh_E/h_{\mathrm{FFN}}\rceil$ 份细节（式 (4) 与符号行同步）；谱系表行与总结行同步 ✓ |
| F4（S3，5.1 单位两处） | **已修复** | 「320 GiB ≈ 344 GB…约 2.5 倍」✓；「B = 7 起反超」✓（13.4 GB / 2.15 GB = 6.2） |
| F7（S3，5.1 use_qk_norm） | **已修复** | §6 该行现注「仅 RoPE 层」✓ |
| F1 波及 5.6（对照表 Scout 行、FLOPs 倍数、字典表） | **已修复**（3bc203e，5.6 作者另行修复，evaluator 复核 diff） | Scout 行「16 routed + 1 shared」；FLOPs 倍数改「Scout 17/2 ≈ 8.5×」与总结「8.5×–56×」；字典表补 Scout 例 ✓ |
| F5（S3，5.4 `\text{}` 内中文 ×2） | **未修，按协议转入下波清零**（line 150 仍在） | S3 允许累计；复审确认该断言仍成立 |
| F6（S3，5.4「6 处指向 4.5」计数） | **未修，按协议转入下波清零**（line 100/175/193 仍为「6 处」） | 同上 |

**复审新登记（S3，累计下波）**：

- F8：5.1 式 (5) 符号行「$g_{i^{*}}$：选中者的**归一化**门控权重」——`modeling_llama4.py` 中 router 输出为原始 sigmoid 分且直接乘进 routed expert（无 top-k 后再归一化），top-1 下「归一化」严格读法会推出 $g\equiv1$，与实现不符。建议改「门控权重（官方实现为原始 sigmoid 分，不归一化）」。该措辞为初稿既有，初轮漏抓，复审补登。
- 附带观察（非 finding）：§7 问题 3 的子项 (a)（shared expert 中间维）在代码层可答——`Llama4TextMLP(config)` 默认走 `config.intermediate_size`（8192），与 routed 同宽；现按练习 4 留给读者做字段级核对，教学上可接受。

**复审放行结论**：5.1 的 S1/S2 全部关闭、S3 两处修复落实，lint 通过——**5.1 放行**；连同初轮已放行的 7 章与已修复的 5.3，**W3 全波 9 章放行**。开放项：S3 三条（F5、F6、F8）转入 W4 波末统一清零。
