# W6 波次独立对抗审查报告（P9 Video Generation，9.1–9.4）

审查日期：2026-11-13。审查对象：`models/09-video-generation/01–04` 四章成品。审查方式：fresh-context 独立 evaluator，全部关键断言回一手来源（arXiv API / ar5iv 全文 / HF config / GitHub API + 文件内容 grep）复核，数学重推导独立执行，不信任正文自述。

## 一句话总结

四章的 FLOPs 账与全部数值经独立重算 40+ 项零误差、38 个 arXiv ID 与 22 个 repo 断言今日实测全对、三条强制重推导（9.2 拓扑账 / 9.3 DF 目标 / 9.3 CausVid 非对称蒸馏）核心公式全部通过；但发现 1 条 S1——9.4 §1.2 把 SVD 谱系的「条件帧 noise augmentation」配方张冠李戴到 Wan 论文（ar5iv 全文取证：Wan 论文无此配方，其对首帧失配的实际回答是 SigLIP 相似度数据过滤），另有 3 条 S2（9.3 CausVid 训练输入「clean 历史」与 Algorithm 1 矛盾、9.1 CogVideoX ablation 的 R=384× 映射与该变体实际 128 通道矛盾、9.4 两处「官方未说明」被 Wan 论文 §5.1.1 证伪）与 5 条 S3。

## 放行结论

**暂不放行（波末初查）**：S1 = 1（9.4），该章修复后须由 evaluator 复审；S2 = 3 条必须修复（9.1/9.3 两条所在章无 S1，修复后按协议无需复审）；S3 = 5 条累计至 W7 波末统一清零。

## Findings 表

| # | severity | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| 1 | S1 | 9.4 §1.2（式 (3) 及全节） | 「Wan 论文的配方（[Team Wan, 2025]，arXiv:2503.20314）是给条件 latent 加噪：ẑ_c = z_c^img + σ_c η」（noise augmentation 防复制粘贴） | ar5iv 2503.20314 全文检索：「noise augmentation / add noise / noisy condition / σ_c」零命中；§5.1.1 Model Design 的 I2V 配方 = 条件图与零填充帧时间维拼接 → VAE → condition latent z_c + binary mask（1=preserved frame）+ 噪声 latent 通道拼接 + 新增投影层零初始化 + CLIP image encoder → 三层 MLP → decoupled cross-attention；§5.1 对首帧失配的实际回答是数据侧：「we retain only the videos with a similarity greater than a predefined threshold」（SigLIP 特征余弦相似度过滤）。noise augmentation 的真实出处是 SVD 谱系：ar5iv 2311.15127 §4.3「we concatenate a noise-augmented (Ho et al. 2021) version of the conditioning frame channel-wise」、D.5「we use noise augmentation for the two conditioning frames, which we apply in the latent space」（上溯 Ho et al. 2022 / Singer et al. 2022） | 式 (3) 改挂 SVD（arXiv:2311.15127，[B] 无官方代码……其配方级断言按论文引用即可；或引 Ho et al. 2022）；§1.2 的「Wan 配方」改为对照陈述：Wan 用首帧相似度数据过滤（§5.1 原文），noise augmentation 是 SVD 谱系配方——两者都是破复制粘贴/条件失配的回答 |
| 2 | S2 | 9.3 §3.2（式 (6) 前训练构造段） | 「学生以 clean 历史 chunk（ground truth）为条件、把当前 chunk 的加噪输入一步映到数据端 x̂_0^{(i)} = G_ψ(x^{(i)}_{t^i}, t^i; x^{(<i)})」 | CausVid ar5iv Algorithm 1 lines 5-7：全体 chunk 各自采 t^i ~ Uniform(𝒯) 并加噪（历史 chunk 同为带噪 ground truth），§4.2 正文「Each chunk has its own noise time step, following Diffusion Forcing」；Self-Forcing（2506.08009）对 CausVid 的定性：「trains… using the DF scheme」而 DF = 「denoising each frame based on noisy context frames」（Fig. 1(b)）。本章 §5.2 亦自述「CausVid 的因果学生已在 chunk 粒度按 DF 方案采样训练输入」——§3.2 的「clean 历史」与之自相矛盾 | 改为「以各自带噪的历史 ground-truth chunk 为条件（t^j ~ Uniform(𝒯)，含 0）」；条件记号改 x_{t^{<i}}^{(<i)}。不动点论证与 DMD 梯度不受影响（错误只在输入构造描述） |
| 3 | S2 | 9.1 §5（ablation 段与总结表「384× 档」行） | CogVideoX ablation「变体 8×8×1 / 8×8×4 / 16×16×8…对应 R = 12× / 48× / 384×」 | ar5iv 2408.06072 Table 1：变体共六行四档压缩率——Baseline 8×8×1（4ch）、A 8×8×4（8ch）、B 8×8×4（16ch，预训练采用）、C 8×8×4（32ch）、D 8×8×8（32ch）、E 16×16×8（**128ch**）；正文「even if the channel dimensions are correspondingly increased, the convergence of the model also becomes extremely difficult」。失败的 E 档实际数值压缩 R = 3·8·16²/128 = 48×，不是按统一 c=16 外推的 384×；真正撞墙的量是 token 密度（τs² = 2048，8 倍于通用档）而非数值体积比 R。枚举亦漏 8×8×8（D）档 | 按 Table 1 实列通道重写：六变体 R 落在 24–96 区间，E 档 R=48 而 τs²=2048；边界教训改写为「通道加倍救不回 token 密度 8× 的收敛崩塌」——这恰好加强本节机制 2（DiT 侧收敛）的论据；总结表「384× 档」同步改为「16×16×8（E 档，128 通道）」 |
| 4 | S2 | 9.4 §1.1（式 (2) 符号表与 4 通道注记） | 「m：逐 latent 帧的锚点指示（…官方未展开 mask 的逐位语义说明）」；「mask 占 4 通道而非 1 是 config 实测值，官方未说明缘由」 | Wan 论文 §5.1.1 原文：「we introduce a binary mask M ∈ {0,1}^{1×T×h×w} where 1 indicates the preserved frame and 0 denotes the frames to be generated」（逐位语义官方已给）；「M shares the same temporal length with the target video, which is then rearranged as shape of s×t×h×w where s is temporal stride of Wan-VAE」——4 通道 = 帧级 mask 按时间 stride 4 重排到 latent 时间分辨率的官方解释。两处「官方未说明」均被原文证伪 | 符号表改为「官方 §5.1.1：binary mask，1=preserved frame，0=to be generated」；4 通道注记改为「帧级 1 通道 mask 经 temporal-stride 重排成 s=4 通道（论文 §5.1.1）」；删冗余编码猜测或降级为对照。顺带可补：新增投影层零初始化（论文原文「an additional projection layer is employed, which is initialized with zero values」）——与 §2「起点即原模型」主題直接互证 |
| 5 | S3 | 9.3 §3.3 与 §7 表 CausVid 行 | 「相对双向 teacher 160× 延迟削减、16× 吞吐提升」 | CausVid 论文正文：「a 160× reduction in latency and a 16× improvement in throughput compared to the similarly scaled **CogVideoX**」；Table 3 实测：Bidirectional Teacher 219.2s/0.6 FPS、CogVideoX-5B 208.6s/0.6 FPS、CausVid 1.3s/9.4 FPS——对教师实为 168.6×/15.7×（≈160×/16× 同数量级成立，但 160×/16× 这组数论文是配给 CogVideoX 的）；另该表口径为 640×352、10s/120 帧 | 改为「vs CogVideoX-5B 160×/16×（论文口径）；vs 双向教师 219.2s→1.3s ≈ 169×、0.6→9.4 FPS ≈ 16×（Table 3，640×352）」 |
| 6 | S3 | 9.2 §7 表注、§10 Mochi 段、总结表 | 「视觉流宽度 ≈4× 文本流」（三处同族表述） | genmoai/mochi README 原文：「our visual stream has nearly 4 times as many **parameters** as the text stream via a larger hidden dimension」——4× 是参数量（经更大 hidden 维度达成），不是宽度；宽度 4× 则参数约 16× | 三处均改「视觉流参数 ≈4× 文本流（更大 hidden 维度）」 |
| 7 | S3 | references.md（非正文章节文件） | 「被引章节」字段 17 处漏回写/1 处多写 | 逐条 grep 四章参考节核对：magvit-v2 缺 9.1；dc-ae 缺 9.1；flashattention 缺 9.2；navit 缺 9.2；qwen2-vl 缺 9.2；dit 缺 9.2/9.4；sd3 缺 9.2；flow-matching 缺 9.2/9.3；rectified-flow 缺 9.2/9.3；cfg 缺 9.3；consistency-models 缺 9.3；dmd 缺 9.3；clip 缺 9.4；stepvideo-t2v 缺 9.1；wan2.2-a14b-hf 缺 9.1；open-sora-plan-repo 缺 9.2；hunyuanvideo-repo 多写 9.4（9.4 未引） | 按 grep 结果回写 |
| 8 | S3 | 9.1 §4 表 LTX 行 | 「<2B」 | LTX 论文摘要口径为「2B parameters, before distillation」（outperforming all available models of similar scale） | 改「2B 级（论文摘要口径）」 |
| 9 | S3 | 9.2 §4 | 「层数切分点官方 README 未给数值，本章不杜撰」 | 狭义成立（README 确无双流/单流层数），但同章代码锚点 hyvideo/modules/models.py 的默认参数即答案：`mm_double_blocks_depth: int = 20, mm_single_blocks_depth: int = 40`（今日实测） | 补「代码默认 20 双流 + 40 单流（hyvideo/modules/models.py 实测）」或保留现状并注明「README 未给、代码默认 20/40」 |

无其他 S1。

## Verification log（抽查项 × 来源 × 结论）

### 机械基线

| 项 | 来源 | 结论 |
| --- | --- | --- |
| lint 四章 | `scripts/lint-chapter.sh` | 4/4 LINT OK（禁令词、模板节、metadata、arXiv 收录、证据等级、`$$` 配平全过） |
| 篇幅（CJK 字数，不含公式/代码） | 自算（去 code block 与 math 后正则计字） | 9.1=4426 / 9.2=5923 / 9.3=8272 / 9.4=5582；9.1/9.4 ≥3500 达标，9.2/9.3 ≥5500 软目标达标 |
| ASCII 图 | 逐章目检 | 四章均 ≥2（9.1 VAE 管线+因果 padding；9.2 shape 走读+dual→single+SNR 时间轴+RoPE 频段；9.3 CausVid 训练图+双层循环+DF 配置空间；9.4 I2V 拼接+keyframe 链） |
| 模板五件套 | 逐章目检 | metadata 块/总结（表格）/练习（各 4 题）/参考 四章齐全 |
| STYLE_GUIDE 抽查 | grep + 目检 | 无外链图、无 Mermaid、禁令词零命中、无「强大的/革命性的」类空词、术语英文保留、首现「中文（English）」格式抽查（因果卷积/feature caching/分桶/exposure bias 等）合规、章末无空洞评论段 |

### arXiv ID ↔ 标题（arXiv API 两批 38 条，2026-11-13）

2010.02502=DDIM ✓；2103.00020=CLIP ✓；2106.09685=LoRA ✓；2205.14135=FlashAttention ✓；2207.12598=CFG ✓；2208.12242=DreamBooth ✓；2209.03003=Rectified Flow ✓；2210.02747=Flow Matching ✓；2211.09794=Null-text Inversion ✓；2212.09748=DiT ✓；2302.05543=ControlNet ✓；2303.01469=Consistency Models ✓；2307.06304=NaViT ✓；2310.05737=MagViT-v2 ✓；2311.18828=DMD ✓；2312.03641=MotionCtrl ✓；2403.03206=SD3 ✓；2404.02101=CameraCtrl ✓；2405.14867=DMD2 ✓；2407.01392=Diffusion Forcing ✓；2408.06072=CogVideoX ✓；2409.12191=Qwen2-VL ✓；2410.10733=DC-AE ✓；2410.13720=Movie Gen ✓；2411.17459=WF-VAE ✓；2412.00131=Open-Sora Plan ✓；2412.03603=HunyuanVideo ✓；2412.07772=CausVid（From Slow Bidirectional to Fast Autoregressive Video Diffusion Models）✓；2501.00103=LTX-Video ✓；2502.10248=Step-Video-T2V ✓；2503.09642=Open-Sora 2.0 ✓；2503.20314=Wan ✓；2504.13074=SkyReels-V2 ✓（勘误执行到位：正文未用 2504.02436）；2506.08009=Self Forcing ✓；2510.08431=rCM ✓（comment 字段 "ICLR 2026" 实测 ✓）；2602.02214=Causal Forcing ✓（abstract 含 "surpassing the SOTA Self Forcing by 19.3%…" ✓）；2605.15141=Causal Forcing++ ✓；2606.25473=Causal-rCM ✓（标题即 "A Unified Teacher-Forcing and Self-Forcing Open Recipe…" ✓）。**38/38 命中，0 错。**

### repo / 文件路径（GitHub API，2026-11-13 实测）

- 存在性 22/22：zai-org/CogVideo、Tencent-Hunyuan/HunyuanVideo、Wan-Video/Wan2.1、Wan-Video/Wan2.2、Lightricks/LTX-Video、stepfun-ai/Step-Video-T2V、hpcaitech/Open-Sora、PKU-YuanGroup/Open-Sora-Plan、genmoai/mochi、tianweiy/CausVid、guandeh17/Self-Forcing、thu-ml/Causal-Forcing、NVlabs/rcm、buoyancy99/diffusion-forcing、SkyworkAI/SkyReels-V2、lllyasviel/ControlNet、hehao13/CameraCtrl、TencentARC/MotionCtrl、microsoft/LoRA、google/prompt-to-prompt、google/dreambooth 全部 200 ✓；genmoai/models → 301 → genmoai/mochi ✓（引用写新地址 ✓）
- 文件/目录级 8/8：Wan2.1 `wan/modules/model.py` ✓（内容级 grep：`def rope_apply` L43、`freqs.split([c - 2*(c//3), c//3, c//3])` L47——c=64 时即 [22,21,21]，与 9.2 §8 表逐字一致 ✓；`img_emb = MLPProj(1280, dim)` L488、`clip_fea` 入参 L499 与 9.4 §1.3 一致 ✓）；HunyuanVideo `hyvideo/modules/models.py` ✓（`rope_dim_list: List[int] = [16, 56, 56]` L461、docstring "for t, h, w" 轴序 ✓、`mm_double_blocks_depth=20 / mm_single_blocks_depth=40` L459-460——finding #9）；tianweiy/CausVid `configs/wan_causal_dmd.yaml` ✓（README 实测引用该 config + `wan_models/Wan2.1-T2V-1.3B/` 基座下载指令 ✓）；google/prompt-to-prompt `null_text_w_ptp.ipynb` ✓；lllyasviel/ControlNet `ldm/` ✓；zai-org/CogVideo `sat/` ✓；Wan2.2 `wan/modules` ✓；HunyuanVideo `hyvideo/` ✓
- README 内容级：Wan2.2 README 逐字命中——「A14B model series adopts a two-expert design… high-noise expert for the early stages… low-noise expert… refining video details. Each expert model has about 14B parameters, resulting in a total of 27B parameters but only 14B active parameters per step」✓；「SNR… decreases monotonically as the denoising step t increases… threshold step t_moe corresponding to half of the SNR_min, and switch to the low-noise expert when t<t_moe」✓；「Wan2.2-VAE… T×H×W compression ratio of 4×16×16… overall compression rate to 64… additional patchification layer… 4×32×32」✓。HunyuanVideo README：「over 13 billion parameters… the largest among all open-source models」✓；「Dual-stream to Single-stream」✓；「extra bidirectional token refiner」✓；「compression ratios of video length, space, and channel to 4, 8, and 16」✓。Mochi README：「AsymmVAE causally compresses… 128x smaller… 8x8 spatial and 6x temporal… 12-channel latent」✓（128× vs 机制算 96× 的口径差确实存在，9.1 表注成立）；「nearly 4 times as many **parameters**… via a larger hidden dimension」（finding #6）；「single T5-XXL」✓。Self-Forcing README：「built on top of… CausVid… and the Wan2.1 repo」+ Wan2.1-T2V-1.3B 下载指令 ✓。Open-Sora-Plan README：v1.5.0（2025-06-05）「higher-compression WFVAE… 8×8×8 downsampling… SUV… 8B-scale… Ascend 910」、`Anysize_8x8x8_32dim` 权重文件名 ✓
- diffusers 实现走读：huggingface/diffusers `autoencoder_kl_wan.py` 实测——`class WanCausalConv3d` L135、`self._padding = (…, 2*self.padding[0], 0)`（时间维左补 2p、右补 0 = 因果）L167、`cache_x` 特征缓存 L172-175 ✓（9.1 §2.2 三件套全部落实）

### HF config / 权重（hf-mirror，2026-11-13 实测）

- zai-org/CogVideoX-5b vae/config.json：`AutoencoderKLCogVideoX`、`temporal_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.7` ✓；transformer/config.json：42L、48 头、head_dim 64、patch_size 2、text_embed_dim 4096、max_text_seq_length 226、sample_frames 49、sample 60×90 ✓（9.2 §2.3/§3 全字段命中）
- tencent/HunyuanVideo hunyuan-video-t2v-720p/vae/config.json：`AutoencoderKLCausal3D`、`time_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.476986` ✓；transformers/ 目录仅三个裸 .pt、无 config（与 sources 记录一致，结构参数以 GitHub README/论文为准的降级处理成立 ✓）
- Wan-AI/Wan2.1-T2V-14B config：in_dim 16、dim 5120、40L、40 头、ffn 13824、text_len 512 ✓（9.2 §2.5 全字段）
- Wan-AI/Wan2.1-I2V-14B-720P config：`in_dim: 36` ✓（=16+16+4，9.4 §1.1 物证）
- Wan-AI/Wan2.2-T2V-A14B tree：`high_noise_model/` + `low_noise_model/` 双目录 ✓（9.2 §6 物证）；high_noise_model/config.json = WanModel dim 5120/40L/40H ✓（「与 Wan2.1-14B 同构」✓）；仓内 `models_t5_umt5-xxl-enc-bf16.pth` 文件名互证 umT5-XXL ✓；注意仓内 VAE 为 `Wan2.1_VAE.pth`（A14B 沿用 Wan2.1 VAE，章节未误判此点 ✓）
- Wan-AI/Wan2.2-TI2V-5B config：in_dim/out_dim 48 ✓（高压缩 VAE 48 通道物证）、dim 3072、30L、24H ✓

### 强制重推导 1：9.2 时空 attention 拓扑 FLOPs 账 —— PASS

- 式 (2) 独立重推：QKV 3×2Nd² + O 2Nd² = 8Nd²；QK^T 与 PV 各 2N²d → 4N²d ✓（softmax 低 d_h 倍不入主账的说明成立）
- 式 (3)：两子层各带全套投影 16Nd²；score 项 t·4S²d + S·4t²d = 4Nd(S+t)（N=tS 提因子）✓
- 式 (4) 省幅 = N/(S+t) = tS/(t+S) = 调和均值之半 ✓；「t≪S 时省幅趋近 t」✓
- 式 (5) 交叉点独立联立：C_fact < C_full ⟺ N > S+t+2d ✓（三行几何全部满足且大一个数量级 ✓）
- 三行表逐格 python 重算：CogVideoX（t=13、30×45、N=17550、d=3072）3.785 vs 3.79 TFLOPs ✓、fact 0.294 ✓、比 12.88 vs 12.9 ✓；Wan（N=302400、d=5120）1872.8 vs 1.87 PFLOPs ✓、89.3 ✓、20.97 vs 21.0 ✓；120 帧行 4081 vs 4.08 ✓、131.9 ✓、30.93 vs 30.9 ✓；×层数：159/12.3 TFLOPs ✓、74.9/3.6 PFLOPs ✓
- §2.5 Wan 14B 全层账：63.4/1872.8/34.9/85.6 TFLOPs、合计 2.0568 PFLOPs、×40 = **82.27 PFLOPs/步**（正文 82.3 ✓）、×50 = 4.11 EFLOPs ✓；占比 3.08/91.06/1.70/4.16% vs 3.1/91.1/1.7/4.2 ✓；参数自洽校验 351.3M×40=14.05B ✓；480p 敏感点 351.7/79.7/81.5% ✓（正文 79.8 为舍入内差）
- §2.4 直接边账：tS²+St²−tS = 4.36×10⁹、N²=9.14×10¹⁰、缺边 95.2% ✓；S+t−1=14420 = 4.77% ✓（「4.8% 恰为 95.2% 之补」✓）；score 矩阵 40 头 bf16 = 7.32 TB ✓
- §9 五桶表逐格重算：N=56160/131040/129600/302400/446400 ✓、相对 1.0/5.4/5.3/29.0/63.2 ✓；激活内存 40×446400×5120×2B = 182.7 GB ✓（183）
- 9.1 配合账全部重算一致：|V|=3.318×10⁸、latent 7.142×10⁶、46.5×、45.2×（CogVideoX 全链）、15×（τ=1 对照）、27×=16×（VAE）×1.67×（宽度）、LTX c=128 反推、Mochi 96×、WFVAE/Wan2.2 48×/64×、LTX N 缩 32×、StepVideo 缩 8× —— 全部 ✓

### 强制重推导 2：9.3 Diffusion Forcing 目标 —— PASS

- 式 (8)(9) vs DF 论文 Algorithm 1 逐行：独立噪声水平 k_t ∈ {0,1,…,K}（含 0——9.3 测度注记所依赖的离散口径 ✓）、逐 token ForwardDiffuse、ε-参数化（line 6）、全体位置 MSE（line 10）——正文 FM 连续化改写已声明为本书约定、原口径离散含 0 亦如实注记 ✓
- 角点一（共享 t）：t_i=t 代入式 (9) 逐字得式 (1) 的 ε 坐标版 ✓；「标准训练 = DF 训练分布的低维截面」与「历史干净+当前有噪 = 共享-t 训练未见配置」的支撑集论证成立 ✓
- 角点二（t_{<i}=0）：条件族 p(x^{(i)}|x^{<i}) 的平滑版 + 逐 token 内层退火采样 = teacher forcing 的扩散版，与摘要「training a causal next-token prediction model… without fully diffusing past ones」逐字互证 ✓
- 「variational lower bound on the likelihoods of all subsequences」摘要逐字命中 ✓；Theorem A.1（uniformly sampled noise levels 上的 ELBO）存在 ✓
- 逐字引文复核：「a diffusion model is trained to denoise a set of tokens with independent per-token noise levels」✓、「without fully diffusing past ones」✓

### 强制重推导 3：9.3 CausVid 非对称蒸馏 —— PASS（附 1 条 S2）

- 式 (5) reverse KL 目标：论文逐字「minimize the reverse KL divergence across randomly sampled timesteps between the smoothed data distribution and the student generator's output distribution」✓；方向语义 mode-seeking ✓（reverse KL = E_{p_ψ} 下期望，模型乱放质量被严惩 ✓；与 rCM 的 forward/backward 互补口径互证 ✓）
- 式 (6) DMD 梯度独立重推：∇KL = E[∇log p_ψ]（=0，密度归一恒等推论 ✓）+ E[(∇_x log p_ψ,t − ∇_x log p_data,t)·∂x/∂ψ] = E[(s_ψ,t − s_data,t)∂x/∂ψ] ✓——符号、方向、测度项处理与 7.4 式 (30)-(32) 同构；ar5iv 命中论文 Eq 3.2 同形表述 ✓
- 逐 chunk 一步预测 vs rollout 反传：Algorithm 1 lines 5-9（逐 chunk 独立噪声、一步预测、拼接）✓、lines 10-12（拼接后单 t 加噪、DMD 更新）✓、lines 13-16（fake score 在线 DSM）✓；Algorithm 2（祖先式更新 x ← α·x̂₀+σ·ε′、KV caching 下推理不再需要显式块因果 mask、干净态 G_φ(x_0,0) 前向入 cache）✓——正文「反传只穿过每 chunk 的一次前向；4 步复合只在推理」的区分与论文结构一致 ✓
- ODE 初始化式 (4)：论文 §4.3 Eq (6) = 教师 ODE 解轨迹上取学生锚点时间子集、向轨迹终点回归 ✓（方向 t:T→0、「few-step generator initialized from the teacher」✓）；「小集合/少迭代」口径逐字命中 ✓
- 非对称含义：「supervises a causal student model with a bidirectional teacher」摘要逐字 ✓；Table 4 消融（无教师 92.9/48.1、因果教师 91.9、双向教师 94.7/64.4/30.1）支撑「监督信号信息量多于学生可用上下文」的读法 ✓；不动点论证（KL 唯一不动点与学生上下文窗口无关）成立 ✓
- 无 GAN：ar5iv 全文「GAN」仅 related work 1 处（早期工作引述）✓——spec 陷阱结案正确
- 端到端数字：50→4 步 ✓、9.4 FPS ✓、1.3s 首帧延迟 ✓、VBench-Long 84.27 ✓、Table 3（219.2s 教师 / 1.3s、640×352、10s/120 帧）✓——160×/16× 的配对错配见 finding #5（S3）
- 唯一错误：训练输入构造句「clean 历史 chunk 为条件」与 Algorithm 1 per-chunk 噪声矛盾（finding #2，S2）；Self-Forcing 侧三件套（自 rollout+KV cache、逐帧自生成条件、视频级分布匹配）✓、梯度截断（「limits the backpropagation to only the final denoising step of each frame」+「detach the gradients… by restricting gradient flow into KV cache embeddings」）逐字命中 ✓
- 9.3 §4.2 式 (7) 流式账 brute-force 验证（M=7、n=43200）：Σ 4dn((k−1)n+n) = 2dN²(1+1/M) 恒等 ✓；比值 K(1+1/M)/2K_T：M=7 → 1/21.87（正文 1/21.9 ✓）、M→∞ → 1/25 ✓；cache 账 248 GB / 800 KiB·token⁻¹ / 与 Llama-3.1-8B 的 14×/6× 对照 ✓；§8 分钟级账（t=361、N=5.20×10⁶、553 PFLOPs/层/步、22.1 EFLOPs/步、5.4×、1.1 ZFLOPs、270×）全部重算一致 ✓

### 论文级断言抽查（ar5iv 全文取证，短句核对）

- CogVideoX（2408.06072）：「3D full attention」正文 5 处 ✓；「expert transformer with expert adaptive LayerNorm」摘要 ✓；3D RoPE ×2 ✓；VAE ablation Table 1 六变体（finding #3 来源）与「convergence… extremely difficult」逐字 ✓；2B/5B 档 ✓
- Wan（2503.20314）：§4.1.1 首帧仅空间压缩 following MagViT-v2、[1+T/4,H/8,W/8]、C→16 ✓；GroupNorm→RMSNorm「to preserve temporal causality」✓；feature cache（§4.1.3）✓；umT5 ×15 ✓；§4.2 DiT+FM+cross-attn ✓；§5.1.1 mask 语义与 s×t×h×w 重排（finding #4 来源）、新增投影层零初始化 ✓；§5.1 I2V 数据过滤（finding #1 来源）✓
- LTX（2501.00103）：1:192、32×32×8 pixels/token、patchify 挪入 VAE、full spatiotemporal self-attention、H100 2s/5s/768×512@24fps 全部逐字 ✓
- StepVideo（2502.10248）：30B、204 帧、Video-VAE 16×16 空间+8× 时间、双语文本编码器 ×2、DiT 3D full attention + FM 逐字 ✓
- Open-Sora 2.0（2503.09642）：$200k、11B、Video DC-AE + FLUX 式 MMDiT hybrid ✓
- Movie Gen（2410.13720）：30B、73K video token = 16s@16fps、Flow Matching ✓
- SkyReels-V2（2504.13074）：Diffusion Forcing 全文计数以大小写不敏感口径 = 23 ✓；SkyCaptioner-V1 ✓；progressive-resolution pretraining + four-stage post-training ✓；RL ✓
- CameraCtrl（2404.02101）：「Plücker」全文 32 处 ✓；plug-and-play「leaving other modules of the base model untouched」✓
- ControlNet（2302.05543）：locked + trainable copy、zero-conv = 权重偏置双零初始化 1×1 conv、两个 zero-conv 实例 ✓（式 (4) 同形）
- Self-Forcing（2506.08009）：见强制重推导 3；exposure bias 摘要指控逐字 ✓
- rCM（2510.08431）：sCM + score distillation long-skip 正则、forward「mode-covering」/ reverse「mode-seeking」互补、FA2 JVP kernel、>10B、Cosmos-Predict2/Wan2.1 ≤14B、1–4 步、15–50×、「without GAN tuning」全部逐字 ✓
- Sora 引文（spacetime patches / diffusion transformer）：本环境 openai.com 全站不可达（curl 403、WebFetch 域验证失败），采用 research pass 的 WebFetch+镜像复核记录；两条引文为 [C] 级短句、标注纪律执行到位

### 数值重算（python 独立执行，40+ 项全对，除 findings 所列外零误差）

9.1：3.32×10⁸/332MB/664MB/4.0GB/1,728,000/5.97TB/31/7.14×10⁶/14.3MB/48×/46.5/21/70,200/5.08×10⁷/45.2/时长表四行 N 与 N²/399GB/4.1PFLOPs/163PFLOPs/15×/111,600/24.9GB/1.5×10¹⁴/27×=16×·1.67×/WFVAE-48×/LTX-128/96×/64×/32×/8×；9.2：4.9×10¹⁰ patchify/全部三行表/2.06PFLOPs/82.3PFLOPs/4.1EFLOPs/91.1%/351.3M/14.05B/480p 81.5%/95.2%/4.8%/7.3TB/17,776/3.88TFLOPs/2.6%/五桶表/183GB/[22,21,21]/1.47（dense-27B 宽度对照）/32%；9.3：164.6PFLOPs/8.2EFLOPs/92%/式 (7) brute-force 恒等/1/21.9 与 1/25/248GB/800KiB/14×/6×/361/5.20×10⁶/553PFLOPs/22.1EFLOPs/5.4×/1.1ZFLOPs/270×；9.4：1.02×10⁵（首层参数增量）/百万分之七/0.06TFLOPs/十万分之三/LoRA 6.55×10⁵·层⁻¹/26.2M/0.19% —— 全部与正文一致 ✓

### 跨章引用完整性

9.1 式 (2)(3)(4) ↔ 9.2 §1/§9、9.3 §1/§8、9.4 §1；9.2 式 (2) ↔ 9.3 §4.2；9.2 §2.5 ↔ 9.3 §1/§2、9.4 §1.1/§6；9.2 式 (7)(8)(9)（SNR/t_moe）↔ 9.3 §1.3；7.5 式 (22)(23)(27)-(29) ↔ 9.2 §5/§6、9.3 §1；7.4 式 (18)(19)(30)-(32) ↔ 9.3 §2/§3；2.4 式 (8)(9) 与 §3/§4.2 ↔ 9.3 §4；8.2 式 (14)-(17) ↔ 9.2 §8；6.3 式 (8)/§7.3-7.4 ↔ 9.1 §1/§5——逐一对上，无断链；记号双载（t = latent 帧数 vs 噪声水平；s = scaling factor vs 空间压缩）两章均在开场声明消歧 ✓

### 时效性与证据分级（2026-11-13 波末实测）

- 后继存在性：arXiv 检索 "Wan2.5/Wan3" 零命中；HunyuanVideo 相关近期命中均为推理加速类（MG-Cache/ESTRA 等）非后继模型；视频蒸馏谱系 2026-07 后确有 LongForcing/Mask Forcing/History Forcing 等更新工作，但 9.3 §7 表自标「快照」、§9 明示「帕累托前沿仍在 2026 年活跃移动」——快照纪律合规 ✓；章节无「最新/首个」类裸时效断言
- [C] 纪律：Sora 仅引已披露原话两句 + [C] ✓；Mochi 仅 README 披露 + 口径差标注 ✓；Movie Gen [B]（未开源）只引摘要口径 ✓；Wan2.2 全段标「官方 README 一手（无论文）」✓
- [A] 三件套：CogVideoX/HunyuanVideo/Wan2.1/Wan2.2/CausVid/Self-Forcing/DF/Causal-Forcing/rCM/SkyReels-V2/Open-Sora 系/ControlNet/CameraCtrl/MotionCtrl 今日全部 repo+权重在线 ✓；DreamBooth 降 [B]（google/dreambooth 仅数据集）执行 ✓

### 覆盖率声明

4/4 章逐行通读；38/38 个正文 arXiv ID 全量复核（非抽样）；22 个 repo 存在性 + 8 个文件/目录锚点全量复测（其中 6 个做了内容级 grep）；6 个 HF config/仓树全字段核对；14 篇论文 ar5iv 全文取证（CogVideoX/Wan/LTX/StepVideo/Open-Sora2.0/MovieGen/SkyReels-V2/CausVid/Self-Forcing/DF/rCM/CameraCtrl/ControlNet/SVD——覆盖全部承重断言）；3 条强制重推导完成（含 brute-force 数值验证）；40+ 项数值独立重算；references.md 被引章节字段全量核对；STYLE_GUIDE 逐条抽查。未覆盖：练习题的完整解答推导（核对了题面事实断言与部分数值口径）；HunyuanVideo 论文本身（ar5iv 不可用，按 sources 记录的 README 一手路径复核）；Sora 页直连（网络不可达，采 research pass 记录）；Causal-Forcing/rCM repo README 逐字复核（GitHub API 限流后以 arXiv 摘要互证关键声明）。

## 修复闭环要求

- S1 ×1（finding #1，9.4 §1.2）必须修复；**9.4 修复后须由 evaluator 复审该章**（含 S1 章节复审协议）。
- S2 ×3（finding #2 9.3、#3 9.1、#4 9.4）必须修复；9.1/9.3 修复后按协议无需复审。
- S3 ×5（finding #5–#9）累计至 W7 波末统一清零。
