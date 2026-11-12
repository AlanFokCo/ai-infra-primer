# W5 调研笔记：P8 Multimodal LLMs 事实源

调研日期：2026-11-12（W5 Task 1，research pass）。本文件是 T2–T6（8.1–8.5 共 5 章）写作的唯一事实源；写作 agent 不做额外调研，公式与数值一律以本文件与 `models/references.md` 为准。VLM 全书记号以「W5 VLM 记号约定」节为唯一口径。

## 访问路径增量记录（沿用 W1–W4 runbook，本轮新增标注）

- arXiv API：`https://export.arxiv.org/api/query`（https 裸协议纪律沿用 W4：http:// 会 hang）。批量 10 条 + `max_results`，本轮两批共 19 条全部一次成功。「需定位」三项用 `search_query=all:"<名称>"` 定位（InternVL3 / Show-o2 / Ovis 均首条命中，见各章）。
- GitHub：repo 存在性 api.github.com/repos（本轮 17 查：12×200、2×301（QwenLM/Qwen2.5-VL→QwenLM/Qwen3-VL；AIDC-AI/Ovis→ATH-MaaS/Ovis）、3×404（facebookresearch/Transfusion、Apollo-LMMs/Apollo、showlab/Show-o2））；文件存在性 api.github.com/contents（10 查全 200）；raw 文件走 `Accept: application/vnd.github.raw`（raw.githubusercontent.com 直连不可用，W2 已记）。**301 重定向需 curl -L 才能解出最终名**（api.github.com 的 301 响应体只给 "Moved Permanently"）。
- HF：一律 hf-mirror。**hf-mirror `/api/models` 端点会 429 限流**（本轮连发 4 查 3 个 429）；绕行 = 直接拉模型页 URL（`https://hf-mirror.com/<org>/<model>`）或 sleep 20s 重试。config.json 拉取 `https://hf-mirror.com/<org>/<model>/raw/main/config.json` 正常（Qwen2.5-VL-7B-Instruct 实测 200）。
- ar5iv 全文取证：本轮用于 Chameleon（2405.09818）与 Transfusion（2408.11039）关键词计数，均完整转换可用。取证口径同 W3：grep 关键词计数 + 短句摘录，正文细节由写作者读论文。
- Apollo 权重取证：hf-mirror `/api/models/Apollo-LMMs/Apollo-{7B,3B,1.5B}` 全 404；org 现存仅 2 个 vision encoder 权重；社区镜像（GoodiesHere/*-t32 等）存在——原始权重已下架的判据。

## W5 VLM 记号约定（写作 agent 必须遵守）

### VLM 组件链与接入点记号

- 形式化：VLM 建模 $p(\text{text} \mid \text{image}, \text{text\_prompt})$（理解方向；生成方向见 8.5）。
- 图像 $x\in\mathbb{R}^{H\times W\times 3}$ 经 vision encoder（ViT 族，P6 记号沿用：patch 边长 $p$、**token 数 $N=HW/p^2$**——复用 P6 约定，勿重定义）得视觉特征 $z_{img}\in\mathbb{R}^{N\times d_v}$（[CLS] 是否保留逐家标注；进 LLM 的一般只取 patch tokens）。
- **connector $\phi$**：$v=\phi(z_{img})\in\mathbb{R}^{M\times d_{llm}}$，$M$ = 每图视觉 token 数（connector 可能改 token 数，如 Q-Former $M$=query 数、pixel shuffle $M=N\cdot r^2$）。
- **三个接入点**：(a) input-level：$v$ 直接拼进 LLM 输入序列（LLaVA 范式，主流）；(b) cross-attention-level：Flamingo gated xattn $h\leftarrow h+\tanh(\alpha)\cdot\mathrm{XAttn}(h,z_{img})$，$\alpha$ zero-init（不破坏预训练 LLM 的机制）；(c) 混合：Q-Former 输出 tokens 走 input-level。
- 全书统一称谓：vision encoder（视觉塔）→ connector（连接器）→ LLM（语言塔）。

### token 预算账（统一口径，实例一律用此账）

- 单图基线：$N=HW/p^2$。实例：CLIP-336/p14 → $576$；224/p14 → 256。
- pixel shuffle / space-to-depth（压缩率 $r$，2×2 相邻合并 ⇒ $r=0.5$）：$M=N\cdot r^2$，通道维 ×$1/r^2$ 后接线性投影（InternVL 代码实测：`pixel_shuffle(scale_factor=downsample_ratio)`、`num_image_token=(H/p)^2·r^2`——`internvl-repo`）。
- AnyRes（LLaVA-NeXT/LLaVA-OneVision）：切 $k$ 块 + 1 全局缩略图 ⇒ $(k+1)\cdot N$；实例账：4 切 → $5\times576=2880$。
- 视频：$F$ 帧 × 每帧 token 数；Qwen2.5-VL 另有 temporal patch（相邻帧合并，t 轴压缩），写作时引 tech report 原文常数。
- context 压力账 = $\sum_{图/帧} M$ + 文本 token，对照 LLM ctx（如 Qwen2.5-VL-7B 的 128000——`qwen25vl-config` 实测）。

### M-RoPE 约定（8.2 关键推导；与 P9 视频 DiT 三轴 RoPE 数学同构）

- position id 分解为三元组 **$(t,h,w)$**（temporal/height/width）；head_dim 的 rotary 频段切三段，实测 Qwen2.5-VL-7B `rope_scaling={type: mrope, mrope_section: [16,24,24]}`——三段和 = head_dim/2（128/2=64=16+24+24，rotary 成对旋转），按序对应 t/h/w（config/代码实测，`qwen25vl-config` + `qwen25vl-modeling`）。
- 构造规则（Qwen2-VL 论文口径 + transformers `get_rope_index` 实现互证）：文本 token 三值同值递增；图像 $t$ 恒定、$(h,w)$ = patch 网格坐标；视频 $t$ 随时间递增。
- **Qwen2.5-VL 增量：t 轴与绝对时间对齐**（dynamic FPS 采样的配套——frame 的 t-id 按实际秒数算而非帧序号），写作时引 tech report 原文，勿凭记忆写常数。
- RoPE 旋转公式本身（轴向分解）recap → 2.5/6.1（RoPE-2D 伏笔在 P6 已埋），本章只补三元组构造与频段切分。

### 与 P6/P7 约定的衔接

- ViT patchify、$N=HW/p^2$、SigLIP tower、对比学习记号 $v_i/t_j/\tau$：沿用 W3 约定页（6.1/6.2），本章不重复定义。
- 统一生成章（8.5）涉及 diffusion loss 时沿用 W4 扩散记号（x₀=数据、t=1=噪声、ε-prediction 默认）；Transfusion 的 $L=L_{LM}+\lambda L_{DDPM}$ 中 $L_{DDPM}$ 即 W4 口径的 $L_{simple}$。
- VQ tokenizer 账（Emu3/Show-o 离散视觉 token）recap → 6.3（VQ-VAE/MAGVIT-v2），勿重推。

## 8.1 VLM 架构分类学

- **LLaVA**（`llava`，arXiv:2304.08485，Haotian Liu，本轮核实——标题 *Visual Instruction Tuning*）：input-level 拼接范式出处（线性 projector）。锚点 `llava-repo`（haotian-liu/LLaVA，200；llava/model 目录实测）。
- **Flamingo**（`flamingo`，arXiv:2204.14198，Alayrac，本轮核实，**[B]——DeepMind 未开源**）：gated xattn（tanh(α) zero-init）+ Perceiver resampler 出处。开放复现锚点 `open-flamingo-repo`（mlfoundations/open_flamingo，200；open_flamingo/src/flamingo.py 实测含 `gated_cross_attn_layers`——gated xattn 走读锚点）。
- **BLIP-2**（`blip2`，arXiv:2301.12597，Junnan Li，本轮核实）：Q-Former 出处（混合接入点 (c)），详拆在 8.2。
- **Qwen2.5-VL**（`qwen2.5-vl`，arXiv:2502.13923，Shuai Bai，本轮核实）：当代开放旗舰锚点。HF 配置实测见 `qwen25vl-config`（7B：vision 32L/hidden 1280/patch 14/spatial_merge 2/window attention 112+fullattn 层 [7,15,23,31]；LLM 28L/3584/28Q-4KV/ctx 128000/vocab 152064/rope_theta 1e6——「vision 0.6B 级 × LLM 7B 级」参数组合表的素材。**vision 参数量修正（2026-11-12 W5 审查）**：tech report Table 1 仅结构参数、无 ViT 参数总量行，原锁「0.4B 级」有误；按 config 现推（32L、d=1280、intermediate 3456、SwiGLU）≈0.63B，含 merger ≈0.67B——写作引「0.6B 级（config 现推）」。
- **InternVL3**（`internvl3`，arXiv:2504.10479，Jinguo Zhu，本轮 search_query 定位核实——标题 *InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models*）：当代开放旗舰对照。
- repo 重定向记录：**QwenLM/Qwen2.5-VL 已 301 → QwenLM/Qwen3-VL**（系列 repo 归并）；引用一律写 `qwen-vl-repo`（QwenLM/Qwen3-VL，-L 解出 200）。
- 写作注意：三接入点公式是本章骨架；参数规模组合表的数字引各家 tech report 原表并标日期，不自行加总。

## 8.2 Connector 与分辨率处理

- **BLIP-2 / Q-Former**（`blip2`，2301.12597）：B 个 learnable queries + ITC/ITM/ITG 三目标 + 两阶段训练出处。锚点 `lavis-repo`（salesforce/LAVIS，200；lavis/models/blip2_models/blip2_qformer.py 实测——Q-Former 走读锚点）。
- **LLaVA-1.5**（`llava-1.5`，arXiv:2310.03744，Haotian Liu，本轮补录核实——标题 *Improved Baselines with Visual Instruction Tuning*）：**MLP（2 层）connector 的数据 ablation 出处**（spec 8.2 点名；brief 清单外，按 spec 内容需要补录）。
- **pixel shuffle**：`internvl-repo` 代码实测（见约定页）；论文侧 `internvl`（2312.14238，Zhe Chen，本轮核实）。Qwen 系 spatial_merge=2 同族（`qwen25vl-config` 字段实测）。
- **Qwen2-VL**（`qwen2-vl`，arXiv:2409.12191，Peng Wang，本轮核实）：naive dynamic resolution + M-RoPE 出处（标题即 *...at Any Resolution*）。
- **M-RoPE 走读锚点**：`qwen25vl-modeling`（transformers modeling_qwen2_5_vl.py，200；实测 mrope_section [16,24,24] split 与 get_rope_index 三元组构造）。约定页公式是唯一写法口径。
- **Pixtral**（`pixtral`，arXiv:2410.07073，Agrawal，本轮核实——标题 *Pixtral 12B*）：native resolution + 2D RoPE、免 tiling 出处；HF 权重 mistralai/Pixtral-12B-2409 实测 200。
- **LLaVA-OneVision**（`llava-onevision`，arXiv:2408.03326，Bo Li，本轮核实——标题 *Easy Visual Task Transfer*）：AnyRes 承继（4 切 2880 账）。
- **MiniCPM-V**（`minicpm-v`，arXiv:2408.01800，Yuan Yao，本轮核实）：端侧高分辨率方案对照；repo `minicpm-v-repo`（OpenBMB/MiniCPM-V，200）。
- **Ovis**（`ovis`，arXiv:2405.20797，Shiyin Lu，本轮 search_query 定位核实——标题 *Ovis: Structural Embedding Alignment for Multimodal Large Language Model*）：visual embedding table（结构化 embedding 对齐）出处。repo `ovis-repo`（**AIDC-AI/Ovis 已 301 → ATH-MaaS/Ovis**，引用写新地址）。后续代际 Ovis2.5 tech report = arXiv:2508.11737（本轮定位、未展开核实，写作如引用须先补录）。
- 写作注意：connector 选型决策表（token 数/灵活性/训练成本）是本章收口；LLaVA-1.5「小数据下 MLP 够、Q-Former 不划算」的 ablation 结论引论文原表。

## 8.3 VLM 训练目标与数据形式

- **LLaVA-OneVision**（`llava-onevision`，2408.03326）：训练配方全开放锚点（stage 划分、数据配比引论文/repo 原文）。锚点 `llava-next-repo`（LLaVA-VL/LLaVA-NeXT，200；README 实测含 llava_onevision 锚点——LLaVA-OneVision 代码家）。
- **Qwen2.5-VL**（`qwen2.5-vl`，2502.13923）：多阶段配方对照。
- **Molmo**（`molmo`，arXiv:2409.17146，Matt Deitke，本轮核实——标题 *Molmo and PixMo: Open Weights and Open Data...*）：数据全开放（PixMo）锚点。repo `molmo-repo`（allenai/Molmo，200）。
- **VCD**（`vcd`，arXiv:2311.16922，Sicong Leng，本轮核实——标题 *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*；spec ⟨verify⟩ 结案）：幻觉解码侧缓解出处，公式 $\tilde p=\mathrm{softmax}((1+\alpha)\,\mathrm{logit}_{img}-\alpha\,\mathrm{logit}_{noise})$（spec 口径，写作时引论文原文核对记号）。官方代码 DAMO-NLP-SG/VCD 200（URL 已并入条目）。
- 写作注意：loss masking 公式 $L=-\sum_{t\in response}\log p(x_t|x_{<t},img)$ 是本章起点；幻觉形式化（language prior vs 视觉证据）给机制假设级表述，不断言超出论文的结论；评测方法论的多选题局限引各家原表。

## 8.4 视频理解

- **Qwen2.5-VL**（`qwen2.5-vl` + `qwen25vl-config` + `qwen-vl-repo`）：dynamic FPS + 绝对时间编码（M-RoPE t 轴）官方实现出处——本章主锚点。absolute-time t-id 的具体常数引 tech report 原文（勿凭记忆）。
- **LLaVA-OneVision**（`llava-onevision` + `llava-next-repo`）：单图→多图→视频迁移锚点。
- **Apollo**（`apollo`，arXiv:2412.10360，Orr Zohar，本轮核实——标题 *Apollo: An Exploration of Video Understanding in Large Multimodal Models*；spec ⟨verify⟩ 结案但**降级 [B]**）：视频 LMM 设计空间 ablation 研究。**降级原因（2026-11-12 实测）**：Apollo-LMMs/Apollo repo 404（org 仅存 .github.io）；HF 原权重 Apollo-LMMs/Apollo-{1.5B,3B,7B} 全 404（社区镜像存在，非官方）；arXiv comment 仅给项目页 apollo-lmms.github.io（200，无代码链接）。写作只引论文 ablation 结论（帧采样/时间压缩/各组件贡献），不引实现。
- 写作注意：帧数 × 每帧 token 总预算表用约定页账（$F\times M$ + 文本）；1min@1fps×256 ≈ 15K 级数字照此推（256 tokens/帧 × 60 帧 = 15360）；「recurrent memory 公式 outline ⟨verify⟩」本轮未找到合用的开放论文锚点——写作时降为定性段落，不给公式。

## 8.5 统一理解与生成

- **Emu3**（`emu3`，2409.18869，W4 已收录，被引 +8.5；锚点 `emu3-repo` 同）：路线 1（VQ + 全 AR）出处，图像=语言的论证直接引摘要口径（W4 已记）。
- **Chameleon**（`chameleon`，arXiv:2405.09818，Chameleon Team (Meta)，本轮核实，**[B]——repo 在（facebookresearch/chameleon，200）但权重撤回**）：early-fusion token 级混合出处。**稳定性全套 ar5iv 全文实证（2026-11-12）**：原文明确「query-key normalization (QK-Norm)」「revised placement of layer norms」「crucial for stable training in the mixed-modal setting」；z-loss ×2、dropout ×11（spec ⟨verify⟩ 结案——原文为标准子层后 dropout：attention 与 FFN 输出后施加、rate 0.1、仅 7B 预训练（SFT 0.05），非「token dropout」；8.5 正文已按此口径）。transformers 走读锚点 `chameleon-modeling`（modeling_chameleon.py 200——看结构、不对权重）。
- **Transfusion**（`transfusion`，arXiv:2408.11039，Chunting Zhou，本轮核实，**[B]——Meta 未开源**：facebookresearch/Transfusion 404 实测）：路线 2（连续 latent）出处。**ar5iv 全文实证**：BOI ×5、EOI ×3（BOI/EOI special tokens 成立）；RoPE 仅在 Llama 架构组件引用处出现（SwiGLU+RoPE 并列引用）、**全文无 "rotary" 细节、无 2D RoPE**——spec「沿用 Llama 式 1D RoPE、勿杜撰 2D」口径成立。$L=L_{LM}+\lambda L_{DDPM}$ 与 attention mask（text causal + image 块内双向）写作时引论文原式。
- **Janus / Janus-Pro**（`janus` 2410.13848 Chengyue Wu、`janus-pro` 2501.17811 Xiaokang Chen，均本轮核实）：解耦理解（SigLIP）/生成（VQ）双 encoder 出处——表征冲突论证引论文原文。锚点 `janus-repo`（deepseek-ai/Janus，200；janus/models 目录实测；两代同 repo；HF Janus-1.3B、Janus-Pro-7B 实测 200）。
- **Show-o / Show-o2**（`show-o` 2408.12528、`show-o2` 2506.15564，均 Jinheng Xie，本轮核实——Show-o2 经 search_query 定位，标题 *Improved Native Unified Multimodal Models*）：AR + 离散 diffusion 混合目标出处。锚点 `showo-repo`（showlab/Show-o，200；**Show-o2 无独立 repo，代码在 show-o2/ 子目录**；README 载 Show-o2 NeurIPS 2025 收录）。
- **Bagel**（`bagel`，arXiv:2505.14683，Chaorui Deng，本轮核实；spec ⟨verify⟩ 结案）：**论文标题是 *Emerging Properties in Unified Multimodal Pretraining*——arXiv 标题不含 "BAGEL"，引用勿张冠李戴**。锚点 `bagel-repo`（ByteDance-Seed/Bagel，200；README 原文「7B active parameters (14B total)」、MoT 架构；HF ByteDance-Seed/BAGEL-7B-MoT 实测 200）。
- 写作注意：对照总表（离散 vs 连续 vs diffusion head）是本章收口；「路线 3 diffusion head（SEED-X/Emu2）」与「GPT-4o 生图 [C]」本轮未收录新条目——写作按 spec 给一段式 [C] 处理（只讲已披露信息），若需正式引用须先走收录流程。

## spec P8 Sources 清单核对结果（Step 4 自检，2026-11-12）

- 8.1：LLaVA 2304.08485 ✓ / Flamingo 2204.14198 ✓（[B]）/ Qwen2.5-VL 2502.13923 ✓（+`llava-repo`、`open-flamingo-repo`、`qwen-vl-repo`）
- 8.2：Qwen2-VL 2409.12191 ✓ / Qwen2.5-VL 2502.13923 ✓ / BLIP-2 2301.12597 ✓ / LLaVA-OneVision 2408.03326 ✓ / Pixtral 2410.07073 ✓ / InternVL 2312.14238 ✓（+`llava-1.5` 补录、`internvl3`、`minicpm-v`、`ovis`、`qwen25vl-modeling`、`lavis-repo`、`internvl-repo`、`minicpm-v-repo`、`ovis-repo`）
- 8.3：LLaVA-OV 2408.03326 ✓ / Qwen2.5-VL 2502.13923 ✓ / Molmo 2409.17146 ✓ / VCD 2311.16922 ✓⟨verify 结案⟩（+`llava-next-repo`、`molmo-repo`、`llava-repo`）
- 8.4：2502.13923 ✓ / 2408.03326 ✓ / Apollo 2412.10360 ✓⟨verify 结案，**[B] 降级**⟩（+`qwen-vl-repo`、`qwen25vl-config`、`llava-next-repo`）
- 8.5：Transfusion 2408.11039 ✓（[B]）/ Chameleon 2405.09818 ✓（[B]）/ Janus 2410.13848 ✓ / Janus-Pro 2501.17811 ✓⟨verify 结案⟩ / Emu3 2409.18869 ✓（W4 收录，+8.5）/ Show-o 2408.12528 ✓⟨verify 结案⟩ / Bagel 2505.14683 ✓⟨verify 结案⟩（+`show-o2`、`janus-repo`、`showo-repo`、`bagel-repo`、`chameleon-modeling` [B]、`emu3-repo`）
- 全部 23 个 arXiv ID（含 brief 外补录 llava-1.5 2310.03744；含「需定位」三项：InternVL3 2504.10479 / Show-o2 2506.15564 / Ovis 2405.20797）+ 15 个非 arXiv 锚点 grep references.md 命中（2026-11-12 机械核对）。

## spec P8 锚点清单核对结果（Step 4 自检续，2026-11-12）

- 8.1 锚点：haotian-liu/LLaVA ✓（`llava-repo`）/ mlfoundations/open_flamingo ✓（`open-flamingo-repo`）/ Qwen2.5-VL ✓（`qwen-vl-repo` + `qwen25vl-config`——repo 已归并 Qwen3-VL）
- 8.2 锚点：Qwen2.5-VL 官方 modeling ✓（`qwen25vl-modeling`）/ OpenGVLab/InternVL ✓（`internvl-repo`，pixel shuffle 实测）/ salesforce/LAVIS ✓（`lavis-repo`，blip2_qformer.py 实测）
- 8.3 锚点：LLaVA-OneVision ✓（`llava-next-repo`）/ Qwen2.5-VL ✓ / Molmo ✓（`molmo-repo`）
- 8.4 锚点：Qwen2.5-VL ✓ / LLaVA-OneVision ✓ / Apollo ✓（`apollo` **paper-only [B]**——repo/权重实测缺失）
- 8.5 锚点：deepseek-ai/Janus ✓（`janus-repo`）/ showlab/show-o ✓（`showo-repo`）/ baaivision/Emu3 ✓（`emu3-repo`，W4 已收录）/ ByteDance-Seed/Bagel ✓（`bagel-repo`）/ transformers chameleon modeling ✓（`chameleon-modeling` [B]）

## references.md 本轮变更清单（2026-11-12）

- Part 8 新增 37 条：22 篇论文（`llava`、`llava-1.5`、`flamingo` [B]、`blip2`、`qwen2-vl`、`qwen2.5-vl`、`internvl`、`internvl3`、`llava-onevision`、`pixtral`、`molmo`、`minicpm-v`、`ovis`、`vcd`、`apollo` [B]、`chameleon` [B]、`transfusion` [B]、`janus`、`janus-pro`、`show-o`、`show-o2`、`bagel`）+ 15 个锚点（`llava-repo`、`open-flamingo-repo`、`qwen-vl-repo`、`qwen25vl-config`、`qwen25vl-modeling`、`lavis-repo`、`internvl-repo`、`llava-next-repo`、`molmo-repo`、`minicpm-v-repo`、`ovis-repo`、`janus-repo`、`showo-repo`、`bagel-repo`、`chameleon-modeling` [B]）。
- 既有条目被引章节回写 2 处：`emu3`（+8.5）、`emu3-repo`（+8.5）。
- 术语表新增 20 条 P8 术语（首次出现章节 8.1–8.5）。
- spec ⟨verify⟩ 项全部结案：VCD 标题 ✓；Apollo 存在但代码/权重下架 → [B] ✓；Janus-Pro 标题 ✓；Show-o 标题 ✓；Bagel 标题（≠"BAGEL"）✓；Chameleon 稳定性全套（QK-Norm/norm 重排/z-loss/dropout）ar5iv 实证 ✓；Transfusion BOI/EOI + 无 2D RoPE 实证 ✓；ByteDance-Seed/Bagel repo 存在 ✓。

## 通用注意事项（P8 写作 agent 必读）

1. **先收录后引用**：正文只能引用 references.md 已收录条目；本轮新增条目核实日期统一 2026-11-12。「路线 3 diffusion head（SEED-X/Emu2）」与「GPT-4o 生图」无收录条目——8.5 按 [C] 一段式处理，不引具体 arXiv。
2. **约定页是唯一记号口径**：$z_{img}$/$\phi$/$v$/$M$ 组件链记号、$N=HW/p^2$（复用 P6）、pixel shuffle $M=Nr^2$、AnyRes $(k+1)N$、M-RoPE $(t,h,w)$ 三元组与 mrope_section [16,24,24]（和 = head_dim/2）——全部照约定页写。
3. **标题与归属勿张冠李戴**：Bagel 论文标题是 *Emerging Properties in Unified Multimodal Pretraining*（无 "BAGEL" 字样）；LLaVA 论文标题是 *Visual Instruction Tuning*；Pixtral 论文标题即 *Pixtral 12B*；Chameleon 署名是 "Chameleon Team (Meta)"。
4. **repo 重定向与缺失（引用一律写新地址/降级措辞）**：QwenLM/Qwen2.5-VL → 301 → QwenLM/Qwen3-VL；AIDC-AI/Ovis → 301 → ATH-MaaS/Ovis；facebookresearch/Transfusion 404（Meta 未开源）；Apollo-LMMs/Apollo 404 且 HF 权重下架；Show-o2 无独立 repo（在 showlab/Show-o 的 show-o2/ 子目录）。
5. **[B] 写作约束**：`flamingo`（DeepMind 未开源——公式可引论文，代码走读走 open_flamingo 复现并注明「复现」）、`chameleon`（权重撤回——结构可看 transformers modeling，不可对权重）、`transfusion`（无官方代码——不引实现细节）、`apollo`（无代码无权重——只引论文 ablation 结论）。
6. **数字引用一律标日期并回原表**：token 预算账用约定页公式现推（576/2880/15K 均为公式推导值）；参数组合表、stage 数据配比、幻觉率、视频 ablation 数字全部引论文原表/repo README 原文，不凭记忆写。
7. **hf-mirror 限流**：/api/models 连发会 429，写作期如需自查用模型页 URL 或 sleep 20s。
