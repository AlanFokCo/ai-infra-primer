# W5 波次独立对抗审查报告（P8 Multimodal LLM，8.1–8.5）

审查日期：2026-11-12。审查对象：`models/08-multimodal-llm/01–05` 五章成品。审查方式：fresh-context 独立 evaluator，全部关键断言回一手来源（arXiv API / ar5iv 全文 / HF config / GitHub API + raw 文件）复核，数学重推导独立执行，不信任正文自述。

## 一句话总结

五章事实密度高、核心公式（M-RoPE / Transfusion / loss masking）经独立重推导全部成立、23 个 arXiv ID 与全部 repo 路径/HF config 今日实测无误、三条强制重推导全部通过；未发现 S1；存在 2 条 S2（LLaVA-1.5「ablation 表」归因错误、LLaVA-OneVision「三阶段课程」描述与论文不符）与 9 条 S3。

## 放行结论

**有条件放行**：S1 = 0；S2 = 2 条必须修复（修复后按协议无需复审——仅含 S1 的章节才需复审）；S3 = 9 条累计、W6 复审时统一清零。

## Findings 表

| # | severity | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| 1 | S2 | 8.2 §2（亦及 §3.4 与 8.2 参考条目） | 「LLaVA-1.5 论文的 ablation 表给了…对比（引原表）：(a) 线性换 2 层 MLP 稳定提升；(b) Q-Former 类在该预算下不占优」 | ar5iv 2310.03744 全文：线性→MLP 是 §3.3 正文陈述（"we find that improving the vision-language connector's representation power with a two-layer MLP can improve…"），无对照表行；Q-Former 类判断只见于 limitations 讨论（"visual resamplers…cannot achieve convergence as efficiently as LLaVA with a comparable amount of training data"）与 InstructBLIP 对照段；论文实际 ablation 为 global context（§4.2）、LLM choices（§4.4）、data efficiency（Fig. 4），**无 connector 消融表**。8.1 §3.3 的正确口径是「论文讨论口径，非表内消融行」，与 8.2 自相矛盾 | 8.2 §2 改「论文正文（§3.3）与 limitations 讨论口径」、删「ablation 表」「引原表」；§3.4「其 ablation 表结论」同步改「其讨论结论」；参考条目「数据规模 ablation」改「数据规模讨论」 |
| 2 | S2 | 8.4 §7 及总结表「课程：单图→多图→视频 [A]」；8.3 §2.2 同族表述 | 「LLaVA-OneVision…单图 → 多图 → 视频三阶段…阶段划分与数据配比引论文原文」 | ar5iv 2408.03326：论文配方为**两阶段**——Single-Image stage（内部 Stage-1/1.5/2 为分辨率渐进）→ OneVision stage（560K 多图 + 350K 视频 + 800K 单图**混合**同训，"we further fine-tune the model using a mixture of video, image, and multi-image data"）；论文无独立的多图阶段，「三阶段」是 blog 产品线的演化史而非论文训练阶段 | 改为「单图 stage → OneVision stage（多图+视频混合同训）」；课程论据改写为「图像阶段打底、混合阶段叠加时间监督」 |
| 3 | S3 | 8.1 §4、8.1 参考条目、8.2 §7 | 「AnyRes 4 切实例 5×576=2880（LLaVA-NeXT / LLaVA-OneVision 口径）」 | 2880 是 LLaVA-NeXT（CLIP-336、576 token/块）实例；LLaVA-OneVision 换 SigLIP-384、每块 729 token（ar5iv 2408.03326 Table 1：「Max 729 × 5」），同公式 (k+1)·N 不同实例数 | 2880 后标注「LLaVA-NeXT 口径」；LLaVA-OneVision 单列「5×729 = 3645」或删其括号归属 |
| 4 | S3 | 8.1 §5 | 「stage-1 通常冻结双塔、只训 connector（LLaVA 与 BLIP-2 的共同选择）——几十万级图文对就能把 v 对齐进 LLM 的 embedding 空间」 | BLIP-2 stage 1 无 LLM 在环（仅冻结 ViT + 训 Q-Former，LLM 于 stage 2 才冻结接入），且其 stage 1 数据为 129M 对（ar5iv 2301.12597 §3.4）；「几十万级」只对 LLaVA 成立。现写法易被读成 BLIP-2 也是几十万级对齐进 LLM | 拆两句：LLaVA 证据（几十万级、LLM 在环冻结）与 BLIP-2（LLM 不入环、129M）分开陈述 |
| 5 | S3 | 8.5 §6.2 式 (14) 符号表 | 「𝒯：文本与 BOI/EOI 位置集」（LM loss 作用位置含 EOI） | Transfusion 论文 footnote 6："When the input is a BOI token, we do not compute any loss"；且 patch 位置输出走噪声头（unshared heads），EOI 从不是 CE 目标——与本章 §6.4 自述「EOI 由解码算法机械追加、非模型采样产出」自证矛盾。BOI 是 CE 目标（推理时模型产出 BOI 触发成图） | 改为「𝒯：文本 token 位置（含以 BOI 为目标的预测位）；EOI 由协议追加、非 CE 目标」 |
| 6 | S3 | 8.4 §3 要点 3 | 「相邻采样帧的 id 差 = ν/fps」 | Qwen2.5-VL τ_p = 2（同章 §4.1 与 config `temporal_patch_size=2` 双重口径），t-id 挂在 merge 后帧组上；HF main `time_interval = tokens_per_second × second_per_grid_t` = ν·τ_p/fps。现写法与 §4.1「t 轴挂在 merge 后的帧组上」粒度不一致 | 注明「merge 后帧组粒度下 id 差 = ν·τ_p/fps」 |
| 7 | S3 | 8.2 §9.1、8.4 §3 式 (4) 符号表 | 「换算常数以 tech report 原文为准」 | ar5iv 2502.13923 全文无 per-second 数值常数（仅机制表述 "aligns MRoPE IDs directly with the timestamps"）；常数实际在 HF config（`tokens_per_second=2`，今日实测）与 HF 代码 | 改引「HF config `tokens_per_second=2`（[A]，qwen25vl-config）」 |
| 8 | S3 | references.md（非正文章节文件） | 「被引章节」字段 8+ 处漏回写 | 今日逐条 grep 核对：flamingo 缺 8.3/8.4；llava 缺 8.3；llava-1.5 缺 8.1/8.3；qwen2-vl 缺 8.4；qwen25vl-modeling 缺 8.4；llava-next-repo 缺 8.2；internvl3 多写 8.2（8.2 正文无 InternVL3）；rope 缺 8.2；qk-norm/siglip/vq-vae/vqgan/magvit-v2/ddpm/maskgit 缺 8.5；vit/clip 缺 8.1 | 按 grep 结果回写 |
| 9 | S3 | 8.3 §2.4 表 Molmo 行 stage 3 | 「（引论文原表）」 | ar5iv 2409.17146 全文无偏好对齐阶段（训练止于 caption 预训练 + SFT；文中 "preference" 仅指人工评测口径），该单元格形同空答 | 写「无（论文配方止于 caption 预训练 + SFT）」 |
| 10 | S3 | 8.1 §3.2 | 「score 矩阵 shape [B, H, T, N]」 | B（batch）与 H（heads）未在符号表定义，且 H 与 §1 式 (1) 的图像高 H 冲突 | 补定义或改记 [B, n_h, T, N] |
| 11 | S3 | 8.1 §3.4（时效性） | 「当代开放旗舰——Qwen2.5-VL 与 InternVL3——清一色选 (a)」 | 截至 2026-11-12，Qwen3-VL Technical Report（arXiv:2511.21631）与 InternVL3.5（arXiv:2508.18265）已发布（今日 arXiv API 实测）；两者仍为 input-level + pixel shuffle 族，主张实质成立，但「当代」未标日期 | 句末加「（2026-11-12 口径）」或脚注后继型号同族 |

无 S1。

## Verification log（抽查项 × 来源 × 结论）

### 机械基线

| 项 | 来源 | 结论 |
| --- | --- | --- |
| lint 五章 | `scripts/lint-chapter.sh` | 5/5 LINT OK（禁令词、模板节、metadata、arXiv 收录、证据等级、`$$` 配平全过） |
| 篇幅（CJK 字数，不含公式/代码） | 自算 | 8.1=4088 / 8.2=6477 / 8.3=5242 / 8.4=4968 / 8.5=7646，均 ≥3500 达标 |
| ASCII 图 | 逐章目检 | 五章均 ≥1（组件链/序列布局/mask 矩阵/shape 走读/管线图） |

### arXiv ID ↔ 标题（arXiv API 两批 23 条，2026-11-12）

2304.08485=Visual Instruction Tuning ✓；2310.03744=Improved Baselines with Visual Instruction Tuning ✓；2204.14198=Flamingo ✓；2301.12597=BLIP-2 ✓；2502.13923=Qwen2.5-VL Technical Report ✓；2504.10479=InternVL3 ✓；2408.03326=LLaVA-OneVision ✓；2103.00020=CLIP ✓；2010.11929=ViT ✓；2405.20797=Ovis ✓；2312.14238=InternVL ✓；2409.12191=Qwen2-VL ✓；2410.07073=Pixtral 12B ✓；2408.01800=MiniCPM-V ✓；2409.17146=Molmo ✓；2311.16922=VCD ✓；2412.10360=Apollo ✓；2409.18869=Emu3 ✓；2405.09818=Chameleon ✓；2410.13848=Janus ✓；2501.17811=Janus-Pro ✓；2408.12528=Show-o ✓；2506.15564=Show-o2 ✓；2408.11039=Transfusion ✓；2505.14683=Emerging Properties in Unified Multimodal Pretraining（Bagel，标题无 "BAGEL" ✓）；1711.00937/2012.09841/2310.05737/2006.11239/2202.04200/2303.15343/2010.04245/2104.09864 经典引用全部 ✓。**23/23 命中，0 错。**

### repo / 文件路径（GitHub API，2026-11-12 实测）

- 存在性：haotian-liu/LLaVA、mlfoundations/open_flamingo、salesforce/LAVIS、OpenGVLab/InternVL、LLaVA-VL/LLaVA-NeXT、allenai/Molmo、OpenBMB/MiniCPM-V、deepseek-ai/Janus、showlab/Show-o、baaivision/Emu3、ByteDance-Seed/Bagel、facebookresearch/chameleon、DAMO-NLP-SG/VCD 全部 200 ✓
- 重定向：QwenLM/Qwen2.5-VL → 301 → QwenLM/Qwen3-VL ✓；AIDC-AI/Ovis → 301 → ATH-MaaS/Ovis ✓（章节引用均写新地址 ✓）
- 404 实测：facebookresearch/Transfusion ✓、Apollo-LMMs/Apollo ✓、showlab/Show-o2 ✓（三章的降级/归并措辞全部成立）
- 文件级：llava/model 目录 ✓；open_flamingo/src/flamingo.py 含 `gated_cross_attn`（×6）✓；lavis/.../blip2_qformer.py ✓；internvl_chat modeling 含 `num_image_token = int((image_size // patch_size)² × downsample_ratio²)` 与 `pixel_shuffle` ✓；transformers modeling_qwen2_5_vl.py / modeling_chameleon.py ✓；janus/models 目录 ✓；showlab/Show-o 的 show-o2/ 子目录 ✓；LLaVA-NeXT `select_best_resolution`（mm_utils.py L119-149：scale=min、effective=min(缩放面积,原面积)、wasted=画布−effective、字典序 max e → min ρ）与 8.2 式 (11) 逐行一致 ✓；llava_arch.py `image_newline = nn.Parameter(...)` ✓；llava/train/train.py 多处 `IGNORE_INDEX` 掩盖 ✓
- README：ByteDance-Seed/Bagel 含 "7B active parameters (14B total)" 原文 ✓；showlab/Show-o README 载 "Show-o2 has been accepted to NeurIPS 2025" ✓；LLaVA-NeXT README 含 onevision 锚点 ✓

### HF config / 权重（hf-mirror，2026-11-12 实测）

- Qwen/Qwen2.5-VL-7B-Instruct config.json 全字段核对：LLM 28L / hidden 3584 / 28 Q heads / 4 KV heads / ctx 128000 / vocab 152064 / rope_theta 1e6 ✓；`rope_scaling={type: mrope, mrope_section: [16,24,24]}` ✓；vision depth 32 / hidden 1280 / patch 14 / spatial_merge 2 / window 112 / fullatt [7,15,23,31] / temporal_patch 2 / tokens_per_second 2 ✓。视觉塔参数现推：32×(4×1280² + 3×1280×3420) ≈ 0.63B ——「0.6B 级」✓（config intermediate=3420 与 tech report 表 3456 有微小出入，不影响章节结论量级）
- 权重页：deepseek-ai/Janus-1.3B ✓、Janus-Pro-7B ✓、ByteDance-Seed/BAGEL-7B-MoT ✓、mistralai/Pixtral-12B-2409（页面 200 ✓；config.json 404 系 repo 布局，非缺失）、OpenGVLab/InternVL3-8B ✓、BAAI/Emu3-Gen ✓、showlab/Show-o ✓
- Apollo-LMMs/Apollo-7B、Apollo-3B：页面 401 + tree API "Repository not found"——权重下架判断成立，[B] 分级维持 ✓

### 强制重推导 1：8.2 M-RoPE 三元组分解 —— PASS

- 式 (14) 构造规则 vs 实现：transformers main `get_rope_index`（modeling_qwen2_5_vl.py L928-1034）逐行核对——`time_interval = tokens_per_second × second_per_grid_ts`（绝对时间）、`current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`（**视频 t 轴行程不进 max**）——8.2 §9.4 的版本注记与当前 main 完全一致 ✓；图像（T=1）下两口径一致的断言 ✓
- 式 (15)：(16+24+24)=64=128/2 ✓（config mrope_section [16,24,24] + head_dim=3584/28=128 实测）；§9.2 频率表 λ_j = 2π·10^{3j/32} 重算：j=0→2π、j=15→160、j=16→199、j=39→2.85×10⁴、j=40→3.53×10⁴、j=63→5.06×10⁶，与章节「2π→1.6×10² / 2.0×10²→2.8×10⁴ / 3.5×10⁴→5.1×10⁶」全对 ✓
- 式 (16) 与 HF 实现的等价性：`recomposition_frequencies` = 按 [16,24,24] split 64 通道、取 t/h/w 三段、`cat((freq,freq))` 复制到前后两半维 → 与「三段模式复制到前后两半维再成对旋转」逐行吻合 ✓
- 式 (17) 主命题：逐块继承 2.5 式 (9)(11)(13)（R(a)ᵀR(b)=R(b−a) 群性质），内积按块分解求和，得三段 B_j 加权和 ✓；推论 1（文本三值同值 ⇒ 逐通道回到 64 通道标准 RoPE）✓——与 tech report 原文 "all three components use identical position IDs, making MRoPE functionally equivalent to traditional 1D RoPE" 互证；推论 2（Δt=0 ⇒ B_j(·;0)=q_{2j}k_{2j}+q_{2j+1}k_{2j+1} 纯内容项）✓；推论 3（三段相位差 (t₀−m, t₀+i−m, t₀+j−m)）✓。与 2.5 式 (13) 的符号约定逐符号一致 ✓

### 强制重推导 2：8.5 Transfusion 完整公式 —— PASS（1 条 S3）

- L = L_LM + λL_DDPM：ar5iv 2408.11039 原文 "ℒ_Transfusion = ℒ_LM + λ·ℒ_DDPM"（Eq 4）✓
- 式 (13) 加噪：与 7.2 式 (6) 同形 ✓；论文 "we add noise ε to each input latent image x₀…before patchification" ⇒ 逐 patch 独立噪声 + 同图共享 t 的章节表述成立（latent 网格上逐元素独立噪声经 patchify 后仍逐 patch 独立）✓；ε-参数化 ✓（"predicts the noise accumulated at timestep t"）
- mask 式 (15)：论文 "causal attention for text tokens and bidirectional attention for image patches" + Fig. 4 ✓；章节 8 token 实例矩阵逐行核对无误（patch 行块内双向、EOI/w3 纯因果）✓
- BOI/EOI：原文 BOI×4/EOI×3 命中 ✓；"we append an EOI token to the predicted image, and switch back to LM mode" 逐字命中 ✓；patch 接口「线性层 或 U-Net up/down 块」✓；footnote 4 "We add an embedding of the timestep t to every patch vector before the linear layer" 逐字命中 ✓；RoPE 全文仅 2 处、均在 Llama 组件引用语境，**无 2D RoPE** ✓
- 无泄漏命题式 (16)：ε⁽ⁱ⁾ ⊥ {ε⁽ʲ⁾, x₀⁽ʲ⁾}_{j≠i} 且独立于 prefix ⇒ H(ε⁽ⁱ⁾|{x_t⁽ʲ⁾}_{j≠i}, prefix) = H(ε⁽ⁱ⁾) ✓ 严格成立；唯一信息通道是共享 x₀ 结构 ✓。唯一瑕疵：式 (14) 符号表把 EOI 列入 CE 位置集，与论文 footnote 6 及本章 §6.4 机械追加口径矛盾（finding #5，S3）

### 强制重推导 3：8.3 loss masking —— PASS

- 式 (1) 掩码 NLL：与标准 SFT 形式一致 ✓；视觉 token 无 ground truth 的论证（连续向量行不在文本词表上、softmax 无法给标签 ⇒ 角色是条件而非预测目标）成立 ✓；prompt 是条件非目标 ✓
- 代码走读：haotian-liu/LLaVA train.py 实测——`mask_human_words` 系函数以 `IGNORE_INDEX`(-100) 掩盖 prompt/指令段、padding_value=IGNORE_INDEX ✓；shift-by-one 在 HF causal LM forward 内部完成的表述 ✓；「被 mask 位置不省 forward、梯度沿 attention 回传到 connector」机制表述正确 ✓
- 有效 token 账：η = 150/3530 = 4.25% ≈ 4% ✓（重算）

### 论文级断言抽查（ar5iv 全文取证，短句核对）

- Chameleon（2405.09818）："crucial for stable training in the mixed-modal setting" ✓；z-loss 10⁻⁵ log²Z ✓；"For Chameleon-7B it was important to use both dropout and z-loss" ✓；"Chameleon-34B only required z-loss"+追加 norm 重排（Swin Liu et al. 2021/2022 策略，"bounds the norm growth of the feedforward block"）✓；"Chameleon-7B can also be stably trained without dropout, when using norm-reordering, but QK-norm is essential in both cases" ✓；dropout 0.1 / 施加在 attention 与 FFN 层后 / 34B 不用 / SFT 0.05 沿用 z-loss ✓；"divergences can manifest after as much as even 20-30% of training progress"+"uncontrolled growth of output norms…strongly correlated" ✓；"each modality will try to 'compete'…by increasing its norms…outside the effective representation range of bf16" ✓；去图生成支路不发散 ✓；logit drift（Wortsman et al. 2023）✓
- Show-o（2408.12528）：Q_t = Q_t^a Q_t^u（Eq 4）、Q_t^a=(1−α_t)I+α_t1e_m^⊤、Q_t^u 均匀分量 β_t/(K+1) ✓；L = L_MTP + αL_NTP（Eq 11，α 配平 NTP 项）✓；L_MTP 裸求和、只施于被掩位置、无 1/t ✓；omni-attention（text causal / image full）✓；built upon pre-trained LLM ✓；MaskGIT/Muse 调度 ✓
- Apollo（2412.10360）：Finding 2（fps 优于 uniform，训练与推理两侧；Fig. 4 middle 控制测试时帧数、差距归因训练采样方式）✓；Finding 3（8–32 tokens/帧强性能带、"little dependence on fps, tpf 与 tps 更具决定性"）✓；Finding 6（Perceiver resampler 降每帧 token 优于 average pooling）✓；Finding 7（clip 间文本时间戳最优、"do not require learning any new token embeddings"）✓；Finding 10（text 10–14% 必要、>25% 或 <7% 掉点、catastrophic forgetting 解读、slightly video-heavy 最优、higher-quality image datasets 解读）✓；Video-MME 短长组 R²=0.94、LongVideoBench R²>0.92 ✓；ApolloBench 按任务维度（temporal reasoning/perception）组织 ✓
- Qwen2.5-VL（2502.13923）：dynamic FPS ✓；MRoPE 对齐绝对时间 ✓；预训练三阶段、phase 1 只训 ViT、phase 2 全部解冻 ✓；post-training（SFT+DPO）ViT 全程冻结 ✓；DPO 阶段存在 ✓；grounding 数据含秒级与 hmsf 时间戳格式 ✓；**无绝对时间换算常数**（支撑 finding #7）
- Qwen2-VL（2409.12191）：Naive Dynamic Resolution 原名 ✓；ViT 去绝对位置嵌入换 2D-RoPE ✓；3D conv 深度 2（两帧一组 tubelet）✓；M-RoPE ✓
- BLIP-2（2301.12597）：32 queries × 768 ✓；188M（含 queries）✓；BERT_base 初始化 + cross-attn 随机初始化 + 隔层插入 ✓；ITC max-over-queries 相似度 + unimodal mask ✓；ITM 逐 query logit 取均值 ✓；ITG multimodal causal mask ✓；129M 图文对 ✓；stage 2 冻双塔 ✓
- Flamingo（2204.14198）：0-init tanh gating 消融掉 4.2% 且致训练不稳 ✓；Perceiver resampler 固定 64 输出 ✓；NFNet-F6 435M 冻结对比预训练 ✓；70B 冻 + 10B 新增（every 7th）+ resampler 194M ✓；视频帧特征加可学时间位置编码后作 K/V ✓
- LLaVA-1.5（2310.03744）：558K+665K ✓；MLP 改进为正文陈述（**无 connector 消融表**——finding #1 的来源）✓；resamplers「数亿级数据」讨论 ✓
- LLaVA（2304.08485）：CLIP ViT-L/14 + 线性 W + grid features ✓
- LLaVA-OneVision（2408.03326）：SI stage → OneVision stage（多图+视频+单图 1.6M 混合）两阶段（finding #2 的来源）✓；视频管线逐帧过塔 + 2×2 bilinear 插值至 196 token/帧、≤32 帧 ✓；task transfer from images to videos ✓；AnyRes 承继 LLaVA-NeXT ✓
- Pixtral（2410.07073）：RoPE-2D 进 ViT、native 分辨率、patch_size 16（Table 1）✓、block-diagonal packing mask ✓
- Ovis（2405.20797）：learnable visual embedding table + 概率组合 ✓；MiniCPM-V（2408.01800）：LLaVA-UHD 式切片 + perceiver resampler 压缩 ✓；Molmo（2409.17146）：口述 60-90s 转写 dense caption、pointing 数据、无偏好阶段（finding #9）✓；Janus（2410.13848）：SigLIP-L-384 理解 + VQ 16384/f16 生成、两层 MLP adaptor、解耦消冲突 ✓；Show-o2（2506.15564）：扩展至视频模态、代码在 showlab/Show-o ✓；Emu3（2409.18869）：8B、codebook 32768、f8、512²→4096、时域 4× 下采样 ✓；VCD（2311.16922）：式 (4) 与论文 Eq 3 逐符号一致 ✓、POPE/MME 判别协议 ✓、扩散前向加噪 T=500/999 ✓、β 截断超参 ✓、语言先验+统计偏差机制分析 ✓

### 数值重算（python 独立执行，~25 项全对）

576/256/2880/15360/9408=16.33×576/1152/3528/61440 bit/MLP 20.97M≈21M/线性 W 5.24M/KV 57344 B=56 KiB/2880 tok→165.15 MB/15360→0.88 GB/86400→4.95 GB/prefill 单层 5.35×10¹³、28 层 1.50×10¹⁵/AnyRes 800×1200 例 e=169344 vs 301056、util 0.75>0.67/视觉塔 0.63B/η=4.25% 全部与正文一致 ✓

### 跨章引用完整性

2.5 式 (9)(11)(13)(16)、2.4 式 (1)(9)、2.3 式 (9)(12)、2.7 式 (15a)(16) 与 §9 OLMo-2、4.3 固定状态、4.6 式 (5) 与 §3/§5、4.8 式 (1)-(3)、6.1 式 (2)(3)(7)(8)(11)(13)(14) 与 §3.1/§4/§6/§8、6.2 双向 InfoNCE、6.3 式 (11)(14)(23) 与 §2/§3.4/§6/§7.2/§7.4、7.2 式 (6) 与 §3.3/§4.4、7.6 adaLN-Zero、7.7 式 (8) 与 §8/§9（Emu3 K=32768/f8/4096、MaskGIT 甜点 8–12 轮）——逐一对上，无断链 ✓

### 时效性与证据分级（2026-11-12 波末另查）

- 后继存在性（arXiv API 今日实测）：Qwen3-VL Technical Report = 2511.21631、InternVL3.5 = 2508.18265、Qwen3-Omni = 2509.17765、Ovis2.5 = 2508.11737。章节锚点（Qwen2.5-VL/InternVL3/Janus-Pro/Show-o2/Bagel）的 paper+code+weights 三件套今日全部仍在线，[A] 成立；「当代旗舰」措辞见 finding #11（S3）
- [B] 分级维持：Flamingo（DeepMind 未开源）✓、Transfusion（facebookresearch/Transfusion 今日 404）✓、Chameleon（repo 200、权重撤回状态未变）✓、Apollo（repo 404 + HF 权重 not found）✓
- [C]：GPT-4o 生图仍无论文，章节 [C] 纪律（只登记已披露信息与行为兼容性、不断言实现）执行到位 ✓
- 无 [W] 条目

### 规范审查（STYLE_GUIDE 逐条）

- §1 语言：术语英文保留、首现「中文（English）」格式抽查（VLM/连接器/交错图文/视觉对比解码/帧采样器/gated xattn）均合规 ✓
- §2：无时间线叙事、无人物故事；「初代 LLaVA」「LLaVA-1.5 起的主流档」为事实性指称非叙事 ✓；章末总结均为表格 ✓
- §3 公式：白名单内（`\operatorname{floor/round}`、`\Longleftrightarrow`、`\underbrace`、`\mathcal`、`\tilde`/`\bar` 均白名单成员）✓；符号表随行 ✓；章内编号连续 ✓
- §4 代码：均标「教学用，非优化实现」、官方对照给 repo+文件路径+函数名且路径今日实测存在 ✓
- §5 图：全 ASCII 自绘、无外链图 ✓
- §6 证据分级：全文 [A]/[B]/[C] 标注齐；数值均挂出处与日期 ✓（finding #1/#2 为出处形态错标，非缺标）
- §7 锚点：各章 metadata 块列锚点+等级 ✓；走读锚点为 2024-2026 [A] 级 ✓
- §8 模板：metadata/总结/练习/参考齐 ✓；篇幅达标 ✓
- §9 AI 味：lint 禁令词零命中；无「强大的/革命性的」类空词；各节以事实/公式/问题开场 ✓

### 覆盖率声明

5/5 章逐行通读；23/23 个正文 arXiv ID 全量复核（非抽样）；17 个 repo 状态断言全量复测；8 个文件级锚点全量复测（其中 5 个做了内容级 grep）；18 篇论文 ar5iv 全文取证（覆盖全部承重断言）；3 条强制重推导完成；约 25 项数值独立重算；references.md 被引章节字段全量核对。未覆盖：各 benchmark 分数（章节明确「不转载原表分数」）、练习题的完整解答（只核对了题面中可作事实断言的数值）。

## 修复闭环要求

- S2 ×2（finding #1、#2）必须修复；修复后无需 evaluator 复审（无 S1）。
- S3 ×9 累计至 W6 波末统一清零。
