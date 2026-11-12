# W3 调研笔记：P5 Case Studies + P6 Vision Representations 事实源

调研日期：2026-11-12（W3 Task 1，research pass）。本文件是 T2–T10（5.1–5.6、6.1–6.3 共 9 章）写作的唯一事实源；写作 agent 不做额外调研，数值一律以本文件与 `models/references.md` 为准。

## 访问路径增量记录（沿用 W1/W2 runbook，本轮新增标注）

- arXiv API：`https://export.arxiv.org/api/query?id_list=...&max_results=N`。本轮出现**间歇性空响应（0 字节）**，sleep 5–10s 重试即恢复（AIMv2 第三次成功）。批量查 10 条以上务必 `max_results`。
- arXiv 全文 HTML：ar5iv（本轮用于 Moonlight/Qwen2/Qwen3/SigLIP2/MAGVIT-v2 关键词取证，均完整转换可用）。取证口径：grep 关键词计数 + 摘要数字抽取，正文细节由写作者读论文。
- HF：huggingface.co 直连不可用，一律 hf-mirror。**gated repo（meta-llama/Llama-4-*、stabilityai/SD3、black-forest-labs/FLUX.*）hf-mirror 同样返回 restricted**；绕行方案 = unsloth 公开镜像（Llama-4 Scout/Maverick 成功）或官方 GitHub 代码（FLUX）。
- GitHub：repo 存在性 api.github.com/repos；raw 文件用 api.github.com/contents + `Accept: application/vnd.github.raw`（raw.githubusercontent.com 直连失败，W2 已记）。**THUDM/GLM-4 已 301 重定向至 zai-org/GLM-4**（组织改名，引用一律写新地址）。
- qwen.ai：全站 SPA，curl 只得 JS shell（任何 id 都返回 200，状态码不代表内容存在）；官方 blog 链接以 HF model card README 为准（W2 口径）。Qwen3-Max 无 HF model card → 本轮无法取 blog 正文，见 §5.3。
- 旁证渠道：OpenRouter `/api/v1/models`（JSON 可 curl）可作 API-only 模型的存在性/ctx/厂商定位旁证（本轮用于 qwen3.7/3.8-max 与 glm-4.7）。

## W3 数学与记号约定（全书 P5/P6 统一，写作 agent 必须遵守）

### P6 视觉记号

- 图像张量 $x\in\mathbb{R}^{H\times W\times C}$（$C=3$）；patch 边长 $p$；**token 数 $N=HW/p^2$**；有 [CLS] 时序列长 $N+1$。实例账：$224/16\Rightarrow N=196$。
- patch 投影：$\mathrm{conv}(k=p,\ s=p)\equiv$ reshape 后单线性层（权重形状 $(p^2C)\times d$）——6.1 推导起点。
- latent 记号：$z\in\mathbb{R}^{h\times w\times c}$，空间压缩率 $f=H/h$（f8=8×，f32=32×）；通道 $c$（SD1.x=4、SD3/FLUX=16）。
- codebook：$K$ 个 entry、维 $d_e$；RVQ $n_q$ 级残差；bitrate $=n_q\log_2 K\cdot\mathrm{fps}$（P10 音频复用同一账）。
- 对比学习记号：视觉 embedding $v_i$、文本 $t_j$（均单位化）、温度 $\tau$；相似度矩阵 $S=VT^{\top}/\tau$；CLIP 的 $\tau=e^{t}$ 可学（init 0.07、clamp 上限 100，论文口径）。
- 2D 位置：patch 坐标 $(h,w)$；RoPE-2D = 轴向分解（半个维度走 $h$、半个走 $w$），为 8.2 M-RoPE 埋伏笔。
- 归一化约定：SigLIP/CLIP 的 embedding 默认 $\ell_2$ 归一化后再点积（$\cos$ 相似度）。

### P5 谱系表字段约定

- 列序固定：model | total/active 参数 | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE（experts/top-k/shared）| norm | vocab | ctx（max_position_embeddings）| rope_theta | 证据条目 key。
- **数值口径优先级**：config.json 实测 > tech report 架构表 > model card > blog。冲突时以 config 为准并加脚注。
- total/active 参数不是 config 字段 → 引用时必须标出处（report 摘要 / blog 原文）与日期。
- active 参数各家口径不一（是否含 embedding、shared expert）：Qwen3-Next 官标「3B 激活（非 embedding 79B）」、K2 摘要「32B activated」、GLM-4.5 摘要「32B activated」——表中照抄官方口径，不自行换算。
- MoE 列写法：`160+1 top-8`（routed+shared、每 token 激活数）；dense 前导层另注（如 `first 3 dense`）。
- ctx 列写 max_position_embeddings 实测值；外推宣称（如 MiniMax 4M、Qwen3-Next 1.01M）另注并降级为厂商宣称。
- config 来源若是镜像（Llama-4 经 unsloth），谱系表脚注注明「meta-llama gated，经 unsloth 镜像实测」。

## 5.1 Llama 谱系

- **Llama-1**（`llama-1`，arXiv:2302.13971，Touvron，2023-02，本轮核实）：标准配方出处（RMSNorm pre-norm + SwiGLU + RoPE + non-tied embedding）；7B–65B 四档；vocab 32000；ctx 2048。
- **Llama-2**（`llama-2`，arXiv:2307.09288，Touvron，2023-07，本轮核实）：70B 引入 GQA（8 KV groups）；ctx 4096；vocab 32000；chat 线 RLHF 细节属 training 维度。
- **Llama-3 herd**（`llama3`，arXiv:2407.21783，W1 已收录）：405B；15T tokens；vocab 128K（128256）；128K ctx；GQA 8 KV heads 全尺寸。8B/70B config 数值 W1 已实测锁定（`llama-3.1-8b-config`/`llama-3.1-70b-config`），直接沿用勿重查。405B config gated 且无公开镜像（unsloth 405B 需鉴权，本轮实测失败）——405B 架构数字以 herd paper 架构表为源。
- **Llama-4 Scout/Maverick**（`llama-4`，**[W] 无 paper**；config 经 unsloth 镜像实测 2026-11-12，blog 同日核实）：
  - 共同：model_type `llama4`，`Llama4ForConditionalGeneration`（单模型内含 vision tower + projector = early-fusion 多模态的 config 级证据）；48 层、hidden 5120、40 Q heads / 8 KV heads（GQA）、head_dim 128、intermediate 8192；**num_experts_per_tok=1（top-1 routing）**；vocab 202048；rope_theta 5×10⁵；`attention_chunk_size=8192`；**`no_rope_layers=[1,1,1,0,...]`（3:1，每第 4 层无 RoPE = iRoPE 的 config 级实证）**；`attn_temperature_tuning=true`、`attn_scale=0.1`；vision tower 34 层 / hidden 1408 / patch 14 / `pixel_shuffle_ratio=0.5` / 16 heads。
  - Scout：16 experts、`interleave_moe_layer_step=1`（层层 MoE）、max_position_embeddings=**10,485,760（10M）**、`rope_scaling={rope_type: llama3, factor 16, original 8192}`。blog 原文「a 17 billion active parameter model with 16 experts」「industry-leading context window of 10M」。
  - Maverick：128 experts、`interleave_moe_layer_step=2`（隔层 MoE）、max_position_embeddings=1,048,576（1M）、rope_scaling=null。blog 原文「400B total parameters」（17B active 同 Scout）。
  - Scout 总参数 109B：blog 中出现「109」字样；写作时引 blog 口径。
  - **[W] 写作约束**：训练数据/配方/评测一律不断言（无 paper）；架构以 config 字段为准并给全字段名。
- repo 锚点：`llama-models-repo`（meta-llama/llama-models，200）。

## 5.2 DeepSeek

全部事实 W1/W2 已锁定，本章为指针汇总（勿重查）：

- V2（`deepseekv2`，2405.04434）：MLA 首秀 + DeepSeekMoE fine-grained——recap 指向 4.1/4.5。
- V3（`deepseekv3`，2412.19437 + `deepseek-v3-config` W1 实测）：671B/37B；256 routed + 1 shared、top-8；aux-loss-free；MTP；14.8T tokens；**first_k_dense_replace=3**；kv_lora_rank=512、qk_rope_head_dim=64；YaRN factor 40（4096→163840）；成本口径 = paper Table 1「2.788M H800 GPU-hours 全训练」。
- V3.2（`deepseek-v3.2`，2512.02556，W2 ar5iv 取证）：DSA lightning indexer、sparse 阶段 top-2048、从 V3.1-Terminus 续训（dense warm-up → sparse，lr 7.3e-6、15000 步、480 seq/步、128K）；官方 repo deepseek-ai/DeepSeek-V3.2-Exp（含 indexer RoPE non-interleaved 勘误公告）。
- R1：训练侧一句话，交叉链接 training，无需新条目。

## 5.3 Qwen

- `qwen1`（arXiv:2309.16609，Bai et al.，本轮核实）：Qwen v1 report，谱系起点锚点。
- **Qwen1.5**（`qwen1.5`，**[W] 无 arXiv 论文**（arXiv 搜索无），官方博客 https://qwenlm.github.io/blog/qwen1.5/ 实测 200）：定位 = 标准配方承接代（dense 全尺寸）；写作时架构断言引 Qwen2 report 的回溯描述 + 博客。
- `qwen2`（arXiv:2407.10671，Yang，本轮核实）：GQA 全系普及；**upcycling 出处**（ar5iv 全文「upcycl」×3：Qwen2-57B-A14B 由 Qwen1.5-32B upcycle 初始化）——**注意：spec 5.5 把 upcycling 写在 Moonlight 条目下，实际 Moonlight 论文（ar5iv 全文）0 次提及 upcycling；upcycling 归 Qwen2（S2 级纠正，5.5 写作时交叉引用本节）**。
- `qwen2.5`（2412.15115，W1 收录）+ `qwen2.5-1m`（2501.15383，W2 收录；config 无 rope_scaling 字段，勿写 YaRN——W1 纠错口径）。
- `qwen3`（2505.09388，W1 收录）：dense 6 尺寸 + MoE 2 尺寸（235B-A22B / 30B-A3B）；**QK-Norm 替代 QKV bias 已核实**（ar5iv 原文「and introduce QK-Norm」；`attention_bias=False` 与 config 互证）；thinking 模式 → 交叉链接 training。
- **Qwen3-235B-A22B-Instruct-2507 config 本轮实测**（2026-11-12）：94 层、hidden 4096、64 Q / **4 KV**、head_dim 128、128 experts top-8、moe_intermediate 1536、`mlp_only_layers=[]`、ctx 262144、rope_theta 5×10⁶、vocab 151936、tie=false。旗舰谱系表直接采用。
- **Qwen3-Next**（`qwen3-next`，[W]，W2 全套锁定）：48 层 = 36 GDN + 12 full（`full_attention_interval=4`）；512 experts top-10 + 1 shared；80B/3B（非 embedding 79B）；15T tokens；ctx 262144→1,010,000；vocab 151936。逐组件 recap 指向 4.3/4.4/4.5。
- **Qwen3-Max**（`qwen3-max`，**[C]**，本轮定位核实）：**无 arXiv（搜索 0 条）、无开放权重（HF Qwen/Qwen3-Max 404 实测 2026-11-12）**；API-only 旗舰；「>1T 参数」为厂商宣称。qwen.ai blog JS 渲染无法 curl 取证（URL 在案：qwen.ai/blog?id=qwen3-max，evaluator 需浏览器复核）。旁证（OpenRouter 模型清单 2026-11-12）：Max 线已迭代至 qwen3.7-max / qwen3.8-max-prime（ctx 1M）——**写作只写一段定位（2025-09 起 API-only 旗舰线、>1T 厂商宣称），标日期、用降级措辞，不写任何架构字段**。

## 5.4 GLM

- `glm`（2103.10360）与 `glm-130b`（2210.02414）：W2 已收录；2021 代码在 THUDM/GLM（W2 核实；zai-org/GLM 是另一 repo，勿混）。
- `glm-4`（arXiv:2406.12793，Team GLM，本轮核实）：标题为 *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools*——GLM-4 代际论文；repo `glm-4-repo`（zai-org/GLM-4，200；**THUDM/GLM-4 已 301 至此**）。
- `glm-4.5`（arXiv:2508.06471，本轮核实）：**355B total / 32B activated**、23T tokens 多阶段训练、hybrid reasoning（thinking/direct 双模式）、ARC 定位——摘要原文。GitHub `glm-4.5-repo`（zai-org/GLM-4.5，200）。
- `glm-4.5v`（arXiv:2507.01006）：视觉版（GLM-4.5V + GLM-4.1V-Thinking），本章只一句定位，细节属 P8。
- **GLM-4.6**（`glm-4.6`，**[W] 无独立 tech report**，config 本轮实测 2026-11-12）：92 层、hidden 5120、96 Q / 8 KV（GQA）、head_dim 128、160 routed + 1 shared、top-8、moe_intermediate 1536、**first_k_dense_replace=3**、ctx **202752**（model card「200K」互证）、rope_theta 10⁶、vocab 151552、**attention_bias=True（QKV bias）**、tie=false。**与 GLM-4.5 config 逐字段对照：唯一差异 = max_position_embeddings 131072→202752**（两 config 本轮同拉对照）。**355B/32B 数字出处 = GLM-4.5 report 摘要**（4.6 README 58 行、无参数数字、6 处引 4.5——「如何对待无 paper 模型」方法论示范素材）。**5.4 fix round 1 补录（2026-11-12，hf-mirror config 复核）**：`use_qk_norm=true`（与 attention_bias **叠加**——4.5 report Table 1「QK-Norm: Yes」+ 正文 "incorporate QK-Norm to stabilize the range of attention logits" 互证）、`intermediate_size=12288`（前 3 dense 层 FFN）、`num_nextn_predict_layers=1`（MTP；report 原文 "we add an MoE layer as the MTP layer"）、`partial_rotary_factor=0.5`（旋 64/128 维）、`hidden_act="silu"`、`rms_norm_eps=1e-5`、`norm_topk_prob=true`、`routed_scaling_factor=2.5`、`n_group=1`/`topk_group=1`（组限定退化为全域路由）、`rope_scaling=null`、`architectures=Glm4MoeForCausalLM`/`model_type=glm4_moe`（自有架构类，借 DeepSeek schema 不借类）；report Table 1 注：355B/32B 口径「含 MTP 层参数、不含 word embeddings 与输出层」。
- **GLM-4.7**（`glm-4.7`，**[W] 无 arXiv**（搜索 0 条），HF zai-org/GLM-4.7 存在（200）、config 本轮实测）：**架构字段与 4.6 逐字段完全一致**（同 checkpoint 迭代线，能力升级非架构变更）。OpenRouter 旁证（2026-11-12）：「Z.ai's latest flagship…enhanced programming + multi-step reasoning」，ctx 204800。写作口径：4.5 report 架构 + 4.6/4.7 config 实测 + model card 能力宣称 [W]，正文标日期。

## 5.5 Kimi / Moonshot

- **Moonlight**（`moonlight`，arXiv:2502.16982，一作 Jingyuan Liu，本轮核实）：**论文标题是 *Muon is Scalable for LLM Training***（勿称 "Moonlight paper"）；Moonlight = **3B active / 16B total** MoE、**5.7T tokens**、Muon 训练（摘要原文）；ar5iv 全文实证：sigmoid routing（×2）、shared expert（×2）、SwiGLU（×2）；**upcycling 不在此文（×0）——归 Qwen2（见 §5.3）**。repo `moonlight-repo`（MoonshotAI/Moonlight，200，README 自述 16B-parameter + Muon 实现）。
- **Muon 原始出处**（`muon-blog`，[W]）：Keller Jordan 博客 https://kellerjordan.github.io/posts/muon/（200），无 arXiv——公式细节属 training 维度，本章只引出处。
- **K1.5**（`kimi-k1.5`，arXiv:2501.12599，本轮核实）：RL 论文，本章一句定位 + 交叉链接 training。
- **K2**（`kimi-k2`，arXiv:2507.20534，本轮核实——spec「有无论文」之问的答案：**有**，2025-07，[A]）：**1T total / 32B activated**（摘要）；**MuonClip = Muon + QK-clip**（摘要原文「a novel QK-clip technique to address training instability」）；15.5T tokens、zero loss spike（摘要）；agentic 定位。`kimi-k2-config` 本轮实测：**architectures=DeepseekV3ForCausalLM（架构类复用 DeepSeek-V3，非笔误——MLA + DeepSeekMoE 谱系归属的直接证据）**；61 层、hidden 7168、64 heads；MLA：kv_lora_rank=512、**q_lora_rank=1536**、qk_rope 64、qk_nope 128、v 128；**384 routed + 1 shared、top-8**、moe_intermediate 2048、first_k_dense_replace=1；rope_theta 5×10⁴ + **YaRN factor 32（4096→131072）**；vocab 163840。repo `kimi-k2-repo`（MoonshotAI/Kimi-K2，200）。
- **Kimi Linear / KDA**（`kimi-linear` + `kimi-linear-config`）：W2 全套锁定（27 层 = 20 KDA + 7 full、mla_use_nope、1.4T 公平对照 / 5.7T 发布档、cache −75%、decode 最高 6.3× 见 Fig 1(b)）——recap 指向 4.3/4.4。
- **K3**（`kimi-k3`，arXiv:2607.24653，W2 锁定）：2.8T / 104B active；KDA + LatentMoE（16/896）+ Attention Residuals；1M ctx；native vision。本章给谱系定位段，架构组件 recap → 4.4。

## 5.6 跨家对照与方法论

- 无新增事实源——对照表数值全部来自 §5.1–5.5 各条目；方法论章（config.json 字段字典表、modeling 阅读路径、tech report 五步读法、30 分钟 checklist）的字段实例素材直接取本文件各 config 摘录（Llama-4 的 no_rope_layers、K2 的 DeepseekV3ForCausalLM 复用、GLM-4.6/4.5 逐字段 diff、DeepSeek-V3 的 YaRN 块均为现成教学案例）。
- 写作注意：跨家对照表的每一格都要能回指本文件某条 config 摘录；激活比 = active/total 用各家官方口径（注意 §5.5 Moonlight 3B/16B 与 K2 32B/1T 分母口径不同，表中注明）。

## 6.1 ViT

- `vit`（arXiv:2010.11929，Dosovitskiy，本轮核实）：patchify、learned 1D 位置嵌入、[CLS]；token 账 224/16→196。
- `swin`（arXiv:2103.14030，Liu，本轮核实）：window attention + shifted window、层级化——6.1 简要变体。
- `navit`（arXiv:2307.06304，Dehghani，本轮核实）：sequence packing 处理任意分辨率/宽高比（摘要原文）；推理期分辨率-成本权衡。
- `siglip`（arXiv:2303.15343，Zhai，本轮核实）：pairwise sigmoid loss、免全局归一化、小 batch 友好（摘要原文）——loss 公式 6.1 推导（对照 6.2 InfoNCE）。
- `siglip2`（arXiv:2502.14786，Tschannen，本轮核实）：multilingual 全家系；ar5iv 实证多目标 = sigmoid + **LocCa captioning（×8）+ self-distillation（×8）**；含 NaViT-style 多分辨率 native-aspect 变体（摘要原文）。当代 VLM 默认 vision tower（P8 伏笔）。
- AIMv2（选读，arXiv:2411.14402，Fini et al.，2024-11，本轮核实存在）：AR 预训练视觉表征。spec 标「选读」——若引用，证据等级按 [A]（Apple 官方，代码在 apple/ml-aim 未核）。
- repo 锚点：`vision-transformer-repo`（google-research/vision_transformer，200；vit_jax/models_vit.py 200）、`big-vision`（google-research/big_vision，200；big_vision/models/vit.py 200——SigLIP/SigLIP2/NaViT 训练代码所在）。

## 6.2 CLIP 与对比学习

- `clip`（arXiv:2103.00020，Radford，本轮核实）：400M 图文对、30+ 数据集评测、zero-shot 匹配 ResNet-50 ImageNet（摘要原文）；InfoNCE 双向 CE + learnable τ（init 0.07、e^t 参数化、clamp ≤100，论文 §2/§3）。
- `alignment-uniformity`（arXiv:2005.10242，Wang & Isola，本轮核实，n=2）：hypersphere 上 alignment/uniformity 两性质与指标公式出处——6.2 几何诊断节直接引。
- repo 锚点：`clip-repo`（openai/CLIP，200）；`open-clip`（mlfoundations/open_clip，200；**src/open_clip/loss.py 200 实测**——InfoNCE/SigLIP loss 代码走读锚点）。
- 写作注意：batch size scaling 的 uniformity 论证 = clip + alignment-uniformity 两篇合引；prompt ensembling 公式按 CLIP 论文 §2.4。

## 6.3 Autoencoder 与 Visual Tokenizer

- `vae`（arXiv:1312.6114，Kingma & Welling，本轮核实）：ELBO/reparameterization 原始出处。
- `vq-vae`（arXiv:1711.00937，van den Oord，本轮核实）：straight-through estimator、commitment loss、EMA 更新出处。
- `vqgan`（arXiv:2012.09841，Esser，本轮核实）：+PatchGAN 对抗 + LPIPS 感知损失；官方实现 `taming-transformers-repo`（CompVis/taming-transformers，200；taming/models/vqgan.py 200）。
- `magvit-v2`（arXiv:2310.05737，Yu，本轮核实）：ar5iv 实证 **LFQ（lookup-free quantization）×27**——隐式 2^d codebook + 熵正则出处；video+image 统一 vocabulary（摘要）。
- `dc-ae`（arXiv:2410.10733，Chen，本轮核实）：深压缩 autoencoder（摘要：突破 8× 空间压缩的重建瓶颈——f32/f64）；官方实现在 `efficientvit-repo`（mit-han-lab/efficientvit，200；applications/dc_ae 目录 200）。
- **现代 AE 实测**（2026-11-12）：SD1.x VAE = f8、**latent_channels=4**（runwayml/stable-diffusion-v1-5 vae/config.json 实测）、**scale_factor=0.18215**（CompVis/stable-diffusion `configs/stable-diffusion/v1-inference.yaml` 实测——注意 0.18215 不在 diffusers 版 config 里，出处是 LDM 原始 yaml）；**FLUX VAE = 16ch、scale_factor=0.3611、shift_factor=0.1159**（black-forest-labs/flux `src/flux/util.py` 代码实测；HF FLUX.* 全 gated）；SD3 VAE 同属 16ch 家族（SD3 论文属 P7/W4 锚定范围，本章引「SD3/FLUX 16ch」时挂 FLUX 代码实测即可）。
- 走读锚点：`autoencoderkl`（diffusers src/diffusers/models/autoencoders/autoencoder_kl.py，200）；`vector-quantize-pytorch`（lucidrains，200，**教学参照**——VQ/RVQ/LFQ 代码对照用，勿当官方实现）。
- RVQ 递推与 bitrate 账：公式出处 = vq-vae/magvit-v2（LFQ 特例）；bitrate = n_q·log2K·fps 的实例账 P10 直接复用（本章给出公式即可）。

## P5 旗舰 config 速查表（本轮实测值，5.6 对照表直接取用）

| 模型 | layers | hidden | Q/KV heads | head_dim | MoE | vocab | max_pos | rope_theta | 实测日期 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-4 Scout | 48 | 5120 | 40/8 | 128 | 16 top-1（层层） | 202048 | 10,485,760 | 5e5（+llama3 式 factor16） | 2026-11-12 |
| Llama-4 Maverick | 48 | 5120 | 40/8 | 128 | 128 top-1（隔层） | 202048 | 1,048,576 | 5e5 | 2026-11-12 |
| DeepSeek-V3 | 61 | 7168 | 128/—（MLA） | — | 256+1 top-8，first 3 dense | 129280 | 163840（YaRN f40） | 1e4 | W1 已锁定 |
| Qwen3-235B-A22B（2507） | 94 | 4096 | 64/4 | 128 | 128 top-8 | 151936 | 262144 | 5e6 | 2026-11-12 |
| Qwen3-Next-80B-A3B | 48（36 GDN+12 full） | 2048 | 16/2 | 256 | 512+1 top-10 | 151936 | 262144 | 1e7 | W2 已锁定 |
| GLM-4.5 / 4.6 / 4.7 | 92 | 5120 | 96/8 | 128 | 160+1 top-8，first 3 dense | 151552 | 131072 / 202752 / 202752 | 1e6 | 2026-11-12 |
| Kimi-K2 | 61 | 7168 | 64（MLA） | — | 384+1 top-8，first 1 dense | 163840 | 131072（YaRN f32） | 5e4 | 2026-11-12 |
| Kimi-Linear-48B-A3B | 27（20 KDA+7 full） | 2304 | 32/— | 72/128 | 256+1 top-8，first 1 dense | 163840 | 1048576 | 1e4 | W2 已锁定 |
| MiniMax-Text-01 | 80（70 linear+10 full） | 6144 | 64/8 | 128 | 32 top-2 | 200064 | 10,240,000 | 1e7 | W2 已锁定 |

total/active 参数（出处均为论文摘要或官方 blog，非 config）：DeepSeek-V3 671B/37B；Qwen3-235B 235B/22B；Qwen3-Next 80B/3B（非 embedding 79B）；GLM-4.5/4.6 355B/32B；K2 1T/32B；Kimi-Linear 48B/3B；MiniMax-01 456B/45.9B；Llama-4 Scout 109B/17B（blog）、Maverick 400B/17B（blog）；K3 2.8T/104B；Moonlight 16B/3B。

## spec P5/P6 Sources 清单核对结果（Step 5 自检，2026-11-12）

- 5.1：Llama-1 2302.13971 ✓ / Llama-2 2307.09288 ✓ / herd 2407.21783 ✓ / Llama-4 [W] = `llama-4` ✓（+`llama-models-repo`、`llama-3.1-8b-config`）
- 5.2：V2 2405.04434 ✓ / V3 2412.19437 ✓ / V3.2 2512.02556 ✓（+`deepseek-v3-config`）
- 5.3：Qwen2 2407.10671 ✓ / Qwen2.5 2412.15115 ✓ / Qwen3 2505.09388 ✓ / Qwen3-Next [W] ✓（本轮补充 `qwen1`、`qwen1.5` [W]、`qwen3-max` [C]）
- 5.4：GLM 2103.10360 ✓ / GLM-130B 2210.02414 ✓ / GLM-4.5 2508.06471 ✓ / GLM-4.5V 2507.01006 ✓（本轮补充 `glm-4` 2406.12793、`glm-4.6`/`glm-4.7` [W]、`glm-4-repo`、`glm-4.5-repo`）
- 5.5：Moonlight/Muon 2502.16982 ✓ / Muon 博客 `muon-blog` [W] ✓ / KDA 2510.26692 ✓ / K1.5 2501.12599 ✓ / **K2 = arXiv:2507.20534 论文存在（[A]，非 model card）** ✓ / K3 2607.24653 ✓
- 5.6：recap 章，无新源。
- 6.1：ViT 2010.11929 ✓ / SigLIP 2303.15343 ✓ / SigLIP 2 2502.14786 ✓ / NaViT 2307.06304 ✓ / Swin 2103.14030 ✓（+`vision-transformer-repo`、`big-vision`；HF SigLIP2 权重由写作时引用 google/siglip2-* model card，本章不新增条目）
- 6.2：CLIP 2103.00020 ✓ / alignment-uniformity 2005.10242 ✓（+`clip-repo`、`open-clip` loss.py 走读锚点）
- 6.3：VAE 1312.6114 ✓ / VQ-VAE 1711.00937 ✓ / VQGAN 2012.09841 ✓ / MAGVIT-v2 2310.05737 ✓ / DC-AE 2410.10733 ✓（+`taming-transformers-repo`、`efficientvit-repo`、`autoencoderkl`、`vector-quantize-pytorch` 教学参照）
- 全部 32 个 arXiv ID + 23 个非 arXiv 锚点 grep references.md 命中（2026-11-12 机械核对）。

## references.md 本轮变更清单（2026-11-12）

- Part 5 新增 22 条：`llama-1`、`llama-2`、`llama-4` [W]、`llama-models-repo`、`qwen1`、`qwen1.5` [W]、`qwen2`、`qwen3-max` [C]、`glm-4`、`glm-4.5`、`glm-4.5v`、`glm-4.6` [W]、`glm-4.7` [W]、`glm-4-repo`、`glm-4.5-repo`、`moonlight`、`muon-blog` [W]、`moonlight-repo`、`kimi-k1.5`、`kimi-k2`、`kimi-k2-config`、`kimi-k2-repo`。
- Part 6 新增 20 条：`vit`、`swin`、`navit`、`siglip`、`siglip2`、`clip`、`alignment-uniformity`、`vae`、`vq-vae`、`vqgan`、`magvit-v2`、`dc-ae`、`vision-transformer-repo`、`big-vision`、`clip-repo`、`open-clip`、`taming-transformers-repo`、`efficientvit-repo`、`autoencoderkl`、`vector-quantize-pytorch`。
- 既有条目「被引章节」回写 11 处：llama3、llama-3.1-8b-config（+5.1）；deepseekv2、deepseekv3、deepseek-v3-config、deepseek-v3.2（+5.2）；qwen3-next（+5.3）；glm、glm-130b（+5.4）；kimi-linear、kimi-k3（+5.5）。
- spec ⟨verify⟩/⟨需定位⟩项全部结案：GLM-4 = 2406.12793 ✓；QK-Norm 替代 QKV bias ✓（Qwen3）；Qwen3-Next = [W] 维持；Qwen3-Max = [C]（API-only）；GLM-4.7 = 存在、[W]、架构同 4.6；K2 = 有 arXiv（2507.20534）；K3 = `kimi-k3` [A] 维持（W2 已锁）；sigmoid routing（Moonlight）✓；SigLIP 2 多目标（sigmoid+LocCa+self-distillation）✓；AIMv2 = 2411.14402（选读，未收录 references.md——若 6.1 引用需补收录）。

## 通用注意事项（写作 agent 必读）

1. **先收录后引用**：正文只能引用 references.md 已收录条目；本轮新增条目核实日期统一 2026-11-12。
2. **[W]/[C] 写作约束**：Llama-4、GLM-4.6、GLM-4.7、Qwen3-Next、Qwen1.5 为 [W]（config/blog/model card 证据）；Qwen3-Max 为 [C]（厂商宣称）——架构字段以 config 实测为准，训练配方/评测断言一律降级措辞 + 标日期（时效性条款）。
3. **勿张冠李戴（本轮新增）**：Moonlight 论文标题 = *Muon is Scalable for LLM Training*；K2 config 的 architectures=DeepseekV3ForCausalLM 是架构复用证据非笔误；THUDM/GLM-4 → zai-org/GLM-4（301）；upcycling 归 Qwen2 不归 Moonlight（spec 5.5 纠正）；Llama-4 是 top-1 routing（16/128 experts 配 top-1，勿写成 top-2）；SD-VAE 的 0.18215 出处在 LDM yaml 不在 diffusers config。
4. **W1/W2 已锁定数值直接沿用（勿重查）**：Llama-3.1-8B/70B config、DeepSeek-V3 config 全套、Qwen3-8B、OLMo-2-7B、Qwen3-Next 全套、Kimi-Linear 全套、MiniMax-Text-01、K3 摘要数字、Mistral sliding_window、Gemma 3 5:1。
5. **激活参数口径**：照抄官方（Qwen3-Next「非 embedding 79B」式注法保留），不自行换算；跨家对照表注明各家口径差异。
6. **hf-mirror gated 限制**：meta-llama/Llama-4、stabilityai SD3/FLUX、unsloth 405B 均 restricted——evaluator 复核 Llama-4 用 unsloth 镜像链接，FLUX 用 GitHub 代码链接。
7. **405B config 缺口**：Llama-3.1-405B 无公开 config（gated + 无镜像），谱系表 405B 行数字引 herd paper 架构表（`llama3`），并在脚注说明。
