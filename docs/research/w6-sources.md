# W6 调研笔记：P9 Video Generation 事实源

调研日期：2026-11-13（W6 Task 1，research pass）。本文件是 T2–T5（9.1–9.4 共 4 章）写作的唯一事实源；写作 agent 不做额外调研，公式与数值一律以本文件与 `models/references.md` 为准。时空记号以「W6 视频生成记号约定」节为唯一口径；扩散记号沿用 W4 约定页。

## 访问路径增量记录（沿用 W1–W5 runbook，本轮新增标注）

- arXiv API：`https://export.arxiv.org/api/query` 批量 id_list 全部一次成功（两轮共 24 条 ID↔标题核对 + 作者）。「需定位」用 `search_query=all:"<名称>"`：SkyReels-V2、CameraCtrl、MotionCtrl 均首条命中。
- **spec/brief 勘误两起**：① brief/spec 所列 SkyReels-V2 ID `2504.02436` 实为 *SkyReels-A2: Compose Anything in Video Diffusion Transformers*；SkyReels-V2 正确 ID = **2504.13074**（*Infinite-length Film Generative Model*）。② Diffusion Forcing 官方 repo 不在作者姓氏名下（boyuanchen/、mit-han-lab/ 均 404），经论文 comment 字段项目页 boyuan.space/diffusion-forcing 定位到 **buoyancy99/diffusion-forcing**（200）。
- GitHub：repo 存在性 api.github.com/repos 共 18 查：15×200、2×301（**THUDM/CogVideo → zai-org/CogVideo**；**genmoai/models → genmoai/mochi**）、0×404（google/dreambooth 存在但只是数据集 repo，见 9.4）。目录/文件级锚点另查 8 处全 200（hyvideo/、sat/、wan/modules、ldm/、causvid/ 等）。
- HF hf-mirror：**THUDM/CogVideoX-5b 页面 307**（权重已迁 zai-org/CogVideoX-5b，config 可拉）；**tencent/HunyuanVideo 官方仓只发裸 .pt**（`hunyuan-video-t2v-720p/transformers/` 内无 diffusers config.json，结构参数以 GitHub README/论文为准）；Wan-AI/* 仓 config.json 直接可读；**Wan2.2-T2V-A14B 仓目录即 high_noise_model/ + low_noise_model/ 双 expert**（MoE 的物证）。
- ar5iv：**2412.03603（HunyuanVideo）转换失败**（"No content available"）→ 该文事实改经 GitHub 官方 README 一手取证；其余 8 篇（CogVideoX/Wan/LTX/CausVid/Self-Forcing/Diffusion Forcing/SkyReels-V2/Open-Sora 2.0/Movie Gen）ar5iv 均成功。
- openai.com 对本机 curl 一律 403（JS checkpoint 对任意 URL 都返 403，无法据此判存在性）；archive.org 本环境 DNS 黑洞不可用。Sora 页存在性与 spacetime patches 原文经 WebFetch 复核 + 全文镜像（cloud.tencent.com 转载）确认，见 9.2。

## W6 视频生成记号约定（写作 agent 必须遵守）

### 时空记号

- 原始视频 $V\in\mathbb{R}^{T\times H\times W\times 3}$（$T$ 帧、$H\times W$ 空间、RGB）；Wan-VAE 式输入写作 $(1+T)\times H\times W\times 3$ 时指「首帧 + $T$ 帧」约定（见下）。
- VAE latent $z\in\mathbb{R}^{t\times h\times w\times c}$；压缩率记 **$(\tau, s, s)$** = 时间 $\tau$×、空间 $s\times s$；token 化后序列长 $N = t\cdot h\cdot w$（patchify 后另除 $p^3$ 或 $p\cdot p^2$，逐家标注）。
- **首帧特判约定（Wan 式，全书统一写法）**：$z$ 时间维 $= 1 + (T-1)/\tau$——首帧只做空间压缩、后续帧按 $\tau$ 时间压缩。实例：$\tau{=}4$、$T{=}81$ → $1+80/4=21$ latent 帧。
- latent 通道 $c$：CogVideoX/HunyuanVideo/Wan 均 16；Mochi 12；Wan2.2-TI2V-5B 高压缩 VAE 48；Open-Sora-Plan WFVAE 32。
- attention FLOPs 账统一以 latent token 数 $N=t\cdot h\cdot w$（patch 边长 1 等效）记：full 3D $=O(N^2)$；factorized $=O(t\cdot(hw)^2 + hw\cdot t^2)$。

### 扩散记号：沿用 W4 约定页（前三条为 W4 recap，第 4 条为本 Part 增量）

1. **$x_0$ = 数据端、$t=1$ = 纯噪声端；采样方向 $t:1\to0$**（Wan/CausVid 等视频论文亦用此向，个别论文 $t$ 大=噪声大与此一致）。
2. 默认 ε-prediction；score↔ε 桥 $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$。
3. FM/rectified flow 口径：$x_t=(1-t)x_0+t\varepsilon$、$u_t=\varepsilon-x_0$；Wan 系论文用 FM 时 SNR 随去噪步 $t$ 递减（$t$ 大=噪声大=SNR 小）。
4. Diffusion Forcing 扩展约定：噪声水平改挂 token 级——序列 $x^{(1..n)}$ 各有独立 $t_i$（逐 token 而非全局单一 $t$），9.3 专用。

## 9.1 Video VAE

- **CogVideoX**（`cogvideox`，arXiv:2408.06072，Zhuoyi Yang，本轮核实，[A]）：3D causal VAE 出处之一。ar5iv 实证：VAE ablation 表变体 8×8×1 / 8×8×4 / 16×16×8，原文「when spatial-temporal compression is too aggressive (16×16×8)… convergence… extremely difficult」→ 选用 **8×8×4（空间 8×8、时间 4×）**。HF config 实测（`cogvideox-config`，zai-org/CogVideoX-5b/vae/config.json）：`AutoencoderKLCogVideoX`、`temporal_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.7`。模型档 2B/5B（论文正文）。锚点 `cogvideo-repo`（**THUDM/CogVideo 已 301 → zai-org/CogVideo**，sat/ 目录实测）。
- **HunyuanVideo**（`hunyuanvideo`，arXiv:2412.03603，Weijie Kong，本轮核实，[A]）：**Causal 3D VAE（CausalConv3D）**，官方 README 原文「compression ratios of video length, space, and channel to 4, 8, and 16」——时间 4×、空间 8×8、通道 3→16。HF config 实测（`hunyuanvideo-vae-config`）：`AutoencoderKLCausal3D`、`time_compression_ratio: 4`、`latent_channels: 16`、`scaling_factor: 0.476986`。锚点 `hunyuanvideo-repo`（hyvideo/ 目录实测）。
- **Wan / Wan-VAE**（`wan`，arXiv:2503.20314，Team Wan，本轮核实，[A]）：论文 §4.1.1 原文：输入 $V\in\mathbb{R}^{(1+T)\times H\times W\times 3}$ → latent $[1+T/4,\ H/8,\ W/8]$、通道 C→16；**「the first frame is only spatially compressed… following MagViT-v2」**（图像=单帧视频兼容的一手出处）；架构上 GroupNorm→RMSNorm。diffusers 实现锚点 `autoencoderkl-wan`（autoencoder_kl_wan.py 实测：`WanCausalConv3d` 自定义 3D 因果卷积、time 维 causal padding + feature caching）。锚点 `wan2.1-repo`。
- **Wan2.2 高压缩 VAE**（`wan2.2`，Wan-Video/Wan2.2 README，本轮核实，[A]——官方 repo+权重、无论文，MoE/VAE 表述引 README 原文）：**Wan2.2-VAE 压缩率 $T\times H\times W = 4\times16\times16$（总 64×）**；TI2V-5B 加 patchification 后总压缩 $4\times32\times32$；5B dense、720P@24fps。HF 实测（`wan2.2-a14b-hf`）：TI2V-5B config `in_dim/out_dim: 48`（高压缩 VAE latent 48 通道）、dim 3072/30L/24H。
- **LTX-Video**（`ltx-video`，arXiv:2501.00103，Yoav HaCohen，本轮核实，[A]）：摘要原文「compression ratio of **1:192**, with spatiotemporal downscaling of **32×32×8** pixels per token, enabled by **relocating the patchifying operation from the transformer's input to the VAE's input**」——VAE 与 patchify 一体化设计出处；<2B 参数、H100 上 5s@24fps 768×512 视频 2 秒（faster-than-real-time）。锚点 `ltx-video-repo`（Lightricks/LTX-Video，200）。
- **StepVideo-T2V**（`stepvideo-t2v`，arXiv:2502.10248，Guoqing Ma，本轮核实，[A]）：摘要实证：30B 参数、最长 204 帧、Video-VAE **16×16 空间 + 8×时间**压缩、双语文本编码器×2、DiT 3D full attention + Flow Matching。repo stepfun-ai/Step-Video-T2V（200）。
- **Open-Sora 2.0**（`open-sora-2`，arXiv:2503.09642，Zangwei Zheng，本轮核实，[A]）：$200k 训练成本、11B；ar5iv 实证架构节为 **Video DC-AE + MMDiT**——9.1「DC-AE 式深压缩在视频上的 tradeoff」的官方实例（DC-AE 图像版见 `dc-ae`，P6 已收录）。锚点 `open-sora-repo`（hpcaitech/Open-Sora，200）。
- **Open-Sora-Plan**（`open-sora-plan`，arXiv:2412.00131，Bin Lin，本轮核实，[A]）+ **WF-VAE**（`wfvae`，arXiv:2411.17459，Zongjian Li，本轮核实，[A]）：v1.3 论文与 WF-VAE 出处；repo README 实证 v1.5.0（2025-06）= 更高压缩 WFVAE（8×8×8、32-dim latent）+ SUV sparse DiT、8B 级、昇腾 910 训练。锚点 `open-sora-plan-repo`（PKU-YuanGroup/Open-Sora-Plan，200）。
- **Mochi 1**（`mochi`，genmoai/mochi，本轮核实，**[C]——无论文**，只登记 repo/README 已披露信息）：10B AsymmDiT；**AsymmVAE 因果压缩 8×8 空间 × 6× 时间 → 12 通道 latent（128× 总压缩）**；单 T5-XXL 文本编码。**重定向记录：genmoai/models 已 301 → genmoai/mochi**。
- 写作注意：① 对照表口径 = (τ,s,s)+latent 通道 $c$，各行注明出处（论文§/config/README）；② 实例内存账用约定页公式现推，720p×5s×24fps → $T{=}120$、Wan-VAE latent $=31\times90\times160\times16$；③ 重建评估（rFID/FVD）方法论一句带过，不引未收录基准数字。

## 9.2 时空 DiT

- **attention 拓扑账**（9.2 核心推导素材）：full 3D（CogVideoX 论文「3D full attention」×5；StepVideo 摘要「DiT with 3D full attention」；LTX 摘要「full spatiotemporal self-attention」）vs factorized/其他；FLOPs 公式按约定页 $N$ 推导，质量差异证据引各家论文原表。
- **CogVideoX**（`cogvideox` + `cogvideox-config` + `cogvideo-repo`）：**expert transformer + expert adaptive LayerNorm（论文摘要原文，文本/视频两模态各自调制参数）**；3D full attention；3D RoPE（论文「3D RoPE」×2）。transformer config 实测：42L、48 头、head_dim 64（→hidden 3072）、patch_size 2、text_embed_dim 4096（T5-XXL）、max_text_seq_length 226、sample_frames 49、sample 60×90（=480×720/8）。
- **HunyuanVideo**（`hunyuanvideo` + `hunyuanvideo-repo`）：**13B+**（README 原文「over 13 billion parameters… the largest among all open-source models」）；**Dual-stream to Single-stream MMDiT**（README §3.2 原文：双流阶段视频/文本 token 各自过独立 block、单流阶段拼接）；**MLLM 文本编码器**（Decoder-Only MLLM 取代 CLIP/T5-XXL 的三条理由 + **额外 bidirectional token refiner**——README 原文，spec ⟨verify⟩ 结案）；3D RoPE 实测锚点 hyvideo/modules/models.py：`rope_dim_list: List[int] = [16, 56, 56]`（t/h/w 三轴）。
- **Wan**（`wan` + `wan2.1-repo`）：DiT + FM + cross-attn（论文 §4.2）；**umT5-XXL 文本编码器**（论文 umT5 ×15 + 权重文件名 models_t5_umt5-xxl-enc-bf16.pth 互证）；14B config 实测 dim 5120/40L/40 头/ffn 13824/text_len 512；**3D RoPE 代码实证**（wan/modules/model.py `rope_apply`：`freqs.split([c - 2*(c//3), c//3, c//3], dim=1)`——head_dim/2 频段切 t/h/w 三段，与 8.2 M-RoPE 数学同构）。
- **Wan2.2 SNR-split MoE**（`wan2.2` + `wan2.2-a14b-hf`——9.2 承重断言，README 原文逐字备查）：「A14B model series adopts a **two-expert design**… a **high-noise expert** for the early stages, focusing on overall layout; and a **low-noise expert** for the later stages, refining video details. Each expert model has about 14B parameters, resulting in a **total of 27B parameters but only 14B active parameters per step**」；路由：「transition point… determined by the **signal-to-noise ratio (SNR)**, a metric that decreases monotonically as the denoising step $t$ increases… high-noise expert is activated [at $t$ large / SNR min]. We define a **threshold step $t_{moe}$ corresponding to half of the $SNR_{min}$, and switch to the low-noise expert when $t<t_{moe}$**」（引号内为 README 原文，注意 $t$ 向与 W4 约定一致）。物证：HF Wan-AI/Wan2.2-T2V-A14B 仓 = high_noise_model/ + low_noise_model/ 两目录，各自 config = WanModel dim 5120/40L/40H（与 Wan2.1-14B 同构）。**写作对照点：这是 timestep 级确定性路由（全 token 共享同一 expert），与 4.5 token 级 MoE 路由不同。**
- **Movie Gen**（`moviegen`，arXiv:2410.13720，Adam Polyak，本轮核实，**[B]——Meta 未开源代码/权重**）：摘要实证：最大视频模型 30B transformer、最大上下文 73K video tokens（=16s@16fps）、Flow Matching；另有同步音频生成。
- **Sora**（`sora`，OpenAI 技术页，**[C]**——spacetime patchify 出处）：原文「we turn videos into patches by first compressing videos into a lower-dimensional latent space, and subsequently decomposing the representation into **spacetime patches**」「Importantly, Sora is a **diffusion transformer**」（curl 403 为全站 JS checkpoint，存在性经 WebFetch + 全文镜像复核）。
- 其他开放系统一段式素材：StepVideo 30B / Open-Sora 2.0 11B / LTX <2B / Mochi 10B（AsymmDiT：non-square QKV、visual stream ≈4× text stream、逐模态独立 MLP——均 [C] repo README 口径）。
- 写作注意：① 时空位置编码对照 = CogVideoX 3D RoPE / HunyuanVideo [16,56,56] / Wan [c−2⌊c/3⌋,⌊c/3⌋,⌊c/3⌋]（后两者代码实测）；② bucketing/渐进时长引各家论文/repo 原文；③ Mochi 无参数表一手论文，只引 README spec 表。

## 9.3 训练配方与加速/流式生成

- **Diffusion Forcing**（`diffusion-forcing`，arXiv:2407.01392，Boyuan Chen，本轮核实，[A]——spec ⟨verify⟩ 结案）：摘要原文「a diffusion model is trained to **denoise a set of tokens with independent per-token noise levels**… training a **causal next-token prediction** model to generate… without fully diffusing past ones」「proven to optimize a **variational lower bound on the likelihoods of all subsequences**」——per-token 噪声水平 + 统一 teacher forcing 与全序列扩散的一手表述；训练目标公式写作时引论文原式（ar5iv 可用）。官方 repo **buoyancy99/diffusion-forcing**（200，经论文 comment→项目页定位，作者姓氏直下无此 repo）。
- **CausVid**（`causvid`，arXiv:2412.07772，Tianwei Yin，本轮核实，[A]）：ar5iv 全文实证——① **非对称蒸馏**：bidirectional teacher → causal student（「asymmetric distillation strategy that supervises a causal student model with a bidirectional teacher」）；② **DMD 目标**：minimize **reverse KL** across randomly sampled timesteps between smoothed data distribution 与 student output distribution（Algorithm 1 "Asymmetric Distillation with DMD"）；③ **ODE 轨迹初始化**：student 先在 teacher 生成的 ODE solution pairs 小集合上回归预训（§4.3）；④ **无 GAN loss**（全文 "GAN" 仅出现在 related work 对早期工作的引述，方法部分无 GAN 分量——spec 警告结案：GAN 分量属 DMD2，勿混淆）；⑤ **KV cache 流式推理**：9.4 FPS 流式、延迟 1.3s（10s/120 帧，160× 延迟削减、16× 吞吐 vs 双向 teacher）。repo `causvid-repo`（tianweiy/CausVid，causvid/ 目录实测；基座 = Wan2.1-T2V-1.3B，50 步→4 步；VBench-Long 84.27）。⑥ **训练输入 = chunk 粒度的 DF 方案**（2026-11-13 审查轮 ar5iv 取证补登）：全体 chunk 各自采噪声时间步 t^i ~ Uniform(𝒯)（含 0）并各自加噪，历史 chunk 同为带噪 ground truth、非 clean 条件——论文 §4.2 原文「Each chunk has its own noise time step, following Diffusion Forcing」、Algorithm 1 lines 5-9；Self-Forcing（2506.08009）对 DF 的定性「denoising each frame based on noisy context frames」（Fig. 1(b)）同口径。
- **Self-Forcing**（`self-forcing`，arXiv:2506.08009，Xun Huang，本轮核实，[A]）：摘要原文——针对 exposure bias（训练见 ground-truth 上下文、推理须以自生成输出为条件），**训练时即以 KV caching 做自回归 rollout、逐帧以自生成输出为条件**，从而在**视频级**施加分部匹配式整体监督（distribution matching ×17、exposure bias ×11、KV cach ×29）。repo `self-forcing-repo`（guandeh17/Self-Forcing，README 明示建于 CausVid 代码 + Wan2.1-1.3B 基座）。**梯度截断口径**（2026-11-13 审查轮论文原文取证补登）：反传只穿过每帧最后一个去噪步（「limits the backpropagation to only the final denoising step of each frame」），且历史 KV cache embedding 的梯度被 detach（「detach the gradients… by restricting gradient flow into KV cache embeddings」）——长 rollout 训练内存可付的另一半机制。
- **SkyReels-V2**（`skyreels-v2`，arXiv:**2504.13074**，Guibin Chen，本轮核实——**勘误：2504.02436 = SkyReels-A2，勿用**）：Infinite-length Film Generative Model；MLLM captioner（SkyCaptioner-V1）+ multi-stage pretraining + RL + **Diffusion Forcing 框架**（全文 23 处）。repo SkyworkAI/SkyReels-V2（200）。长视频「分块 AR」路线的当代代表。
- **Causal-Forcing / Causal-Forcing++**（`causal-forcing` 2602.02214、`causal-forcing-pp` 2605.15141，Hongzhou Zhu，本轮核实，[A]）：thu-ml/Causal-Forcing README 实证——**Causal ODE（或 Causal Consistency Distillation）驱动 asymmetric DMD 初始化**；「significantly outperforms Self Forcing in both visual quality and motion dynamics, same training budget and inference efficiency」；支持 chunk-wise 与 frame-wise；CF++（2026-05）以 causal consistency distillation 取代 ODE 配对数据，发布首个 frame-wise 1/2 步模型。2026 后续现状表素材（RAVEN、Rolling Forcing 等派生见 repo News，不逐一收录）。
- **rCM / Causal-rCM**（`rcm` 2510.08431、`causal-rcm` 2606.25473，Kaiwen Zheng，本轮核实，[A]）：NVIDIA，ICLR 2026。rCM = score-regularized continuous-time consistency（sCM + score distillation 正则，forward KL mode-covering + reverse KL mode-seeking 互补）；FA2 JVP kernel、>10B 规模、Cosmos-Predict2/Wan2.1（≤14B）验证；1–4 步、15–50× 加速、**无 GAN tuning**。Causal-rCM = **teacher-forcing（forward/offline CM）与 self-forcing（reverse/on-policy DMD）统一开放配方**（repo README 原文）。锚点 `rcm-repo`（NVlabs/rcm，200）。
- DMD2 视频版：DMD2 本体已收录（`dmd2`，7.4）；视频化表述以 CausVid（DMD→视频）为准。被引章节回写：`dmd2` +9.3、`ddim` +9.4（inversion 出处）。
- 写作注意：① CausVid vs DMD2 的 GAN 归属是 spec 明文陷阱，正文须显式澄清；② prefill/decode 与 KV cache 概念 recap → 2.4，本章只讲视频语境映射；③ 「世界模型/交互式生成」展望只给一段式并标成熟度（Causal-rCM 副标题即 interactive world models，可作 [A] 锚点）。

## 9.4 控制与编辑

- **I2V 条件注入（Wan 变体代码实证）**：Wan2.1-I2V-14B-720P config 实测（`wan21-i2v-config`）`in_dim: 36` = 16（噪声 latent）+ 16（条件图像 latent）+ 4（mask）——**latent concat + mask 通道**；wan/modules/model.py 实测 `clip_fea` 独立条件输入（CLIP image embedding 经 `img_emb = MLPProj(1280, dim)` 后注入 cross-attn）。diffusers WanImageToVideoPipeline 用 CLIPVisionModel image_encoder + AutoencoderKLWan（Wan2.1 README 官方示例）。noise augmentation（首帧加噪防复制粘贴）为 Wan 论文所述配方，写作引原文。
- **ControlNet**（`controlnet`，arXiv:2302.05543，Lvmin Zhang，本轮核实，[A]）：可训练副本 + zero-conv 出处（零初始化=起点即原模型）。锚点 `controlnet-repo`（lllyasviel/ControlNet，ldm/ 目录实测）。
- **相机控制**：**CameraCtrl**（`cameractrl`，arXiv:2404.02101，Hao He，本轮核实——spec ⟨verify⟩ 结案）：ar5iv 实证 **Plücker embeddings ×32**（per-pixel 射线表示作条件；相机模块 plug-and-play、不动基座）；锚点 `cameractrl-repo`（hehao13/CameraCtrl，200）。**MotionCtrl**（`motionctrl`，arXiv:2312.03641，Zhouxia Wang，本轮核实——相机+物体运动统一控制；repo TencentARC/MotionCtrl，200）。Plücker map 构造公式（$(d, o\times d)\in\mathbb{R}^6$ per pixel）写作时引 CameraCtrl 论文原式。
- **LoRA**（`lora`，arXiv:2106.09685，Edward J. Hu，本轮核实，[A]——microsoft/LoRA 200）：$\Delta W=(\alpha/r)BA$ 低秩公式出处。
- **DreamBooth**（`dreambooth`，arXiv:2208.12242，Nataniel Ruiz，本轮核实，**[B]**——google/dreambooth repo 存在但仅为主题数据集、无官方训练代码；公式引论文原文即可）。
- **null-text inversion**（`null-text-inversion`，arXiv:2211.09794，Ron Mokady，本轮核实，[A]——官方实现在 google/prompt-to-prompt 的 `null_text_w_ptp.ipynb`，实测存在）：DDIM inversion + 每步 null-text embedding 优化出处；视频推广（误差累积讨论）写作时注明是本书的扩展分析。
- 写作注意：① ControlNet-for-DiT 段落在原 U-Net 公式后接 DiT 语境说明（adaLN-Zero 同思想——`dit` P7 已收录）；② 轨迹条件（drag point embedding）若无收录条目支撑具体系统，按公式级一般性表述写；③ 多镜头/叙事结构给 keyframe 链公式级即可。

## spec P9 Sources 清单核对结果（Step 4 自检，2026-11-13）

- 9.1：CogVideoX 2408.06072 ✓ / HunyuanVideo 2412.03603 ✓ / Wan 2503.20314 ✓（+`cogvideo-repo`、`hunyuanvideo-repo`、`wan2.1-repo`、`cogvideox-config`、`hunyuanvideo-vae-config`、`autoencoderkl-wan`、`open-sora-2`、`open-sora-plan`、`wfvae`、`ltx-video`、`stepvideo-t2v`、`mochi` [C]）
- 9.2：2408.06072 ✓ / 2412.03603 ✓ / 2503.20314 ✓ / Wan2.2 README ✓⟨verify 结案，[A] repo 一手⟩ / StepVideo 2502.10248 ✓ / LTX-Video 2501.00103 ✓（+`moviegen` [B]、`sora` [C]、`wan2.2-a14b-hf`、`open-sora-repo`、`ltx-video-repo`）
- 9.3：Diffusion Forcing 2407.01392 ✓⟨verify 结案：repo=buoyancy99/diffusion-forcing⟩ / CausVid 2412.07772 ✓ / Self-Forcing 2506.08009 ✓ / Causal-Forcing 2602.02214 ✓⟨verify 结案⟩+CF++ 2605.15141 / rCM 2510.08431 ✓⟨verify 结案⟩+Causal-rCM 2606.25473（+`skyreels-v2` 2504.13074 **勘误**、`causvid-repo`、`self-forcing-repo`、`causal-forcing-repo`、`rcm-repo`、`skyreels-v2-repo`）
- 9.4：ControlNet 2302.05543 ✓ / LoRA 2106.09685 ✓ / DreamBooth 2208.12242 ✓（[B] 降级说明见上）/ null-text 2211.09794 ✓⟨verify 结案：官方实现在 prompt-to-prompt repo⟩ / CameraCtrl 2404.02101 ✓⟨verify 结案⟩（+`motionctrl` 2312.03641、`controlnet-repo`、`cameractrl-repo`、`wan21-i2v-config`、`wan2.2` I2V）
- 全部 23 个 arXiv ID（14 原列 + 9 新增定位/勘误/补录）+ 18 个 repo/配置锚点，2026-11-13 逐一实测。

## spec P9 锚点清单核对结果（Step 4 自检续，2026-11-13）

- 9.1 锚点：zai-org/CogVideo ✓（**THUDM/CogVideo 301 → zai-org**，sat/ 实测）/ Tencent-Hunyuan/HunyuanVideo ✓（hyvideo/ 实测）/ Wan-Video/Wan2.1 ✓
- 9.2 锚点：CogVideoX/HunyuanVideo/Wan2.2 官方 repo ✓（wan/modules、hyvideo/modules、Wan2.2 仓双 expert 目录均实测）
- 9.3 锚点：tianweiy/CausVid ✓ / guandeh17/Self-Forcing ✓ / thu-ml/Causal-Forcing ✓ / NVlabs/rcm ✓ / diffusion-forcing 官方 repo ✓（buoyancy99/diffusion-forcing）
- 9.4 锚点：Wan2.2-I2V 官方代码 ✓（wan2.2 仓 wan/modules；I2V 机制另以 Wan2.1-I2V config+代码互证）/ lllyasviel/ControlNet ✓ / CameraCtrl ✓（hehao13/CameraCtrl）

## references.md 本轮变更清单（2026-11-13）

- Part 9 新增 44 条：26 篇论文/官方页面（`cogvideox`、`hunyuanvideo`、`wan`、`wan2.2`、`ltx-video`、`stepvideo-t2v`、`open-sora-2`、`open-sora-plan`、`wfvae`、`mochi` [C]、`moviegen` [B]、`sora` [C]、`diffusion-forcing`、`causvid`、`self-forcing`、`skyreels-v2`、`causal-forcing`、`causal-forcing-pp`、`rcm`、`causal-rcm`、`controlnet`、`lora`、`dreambooth` [B]、`null-text-inversion`、`cameractrl`、`motionctrl`）+ 18 个锚点（`cogvideo-repo`、`hunyuanvideo-repo`、`wan2.1-repo`、`open-sora-repo`、`open-sora-plan-repo`、`ltx-video-repo`、`causvid-repo`、`self-forcing-repo`、`causal-forcing-repo`、`rcm-repo`、`controlnet-repo`、`cameractrl-repo`、`cogvideox-config`、`hunyuanvideo-vae-config`、`wan21-i2v-config`、`wan2.2-a14b-hf`、`autoencoderkl-wan`、`skyreels-v2-repo`）。
- 既有条目被引章节回写 2 处：`dmd2`（+9.3）、`ddim`（+9.4）。
- spec ⟨verify⟩ 全部结案：Wan2.2（repo README 一手）✓、StepVideo 2502.10248 ✓、LTX 2501.00103 ✓、diffusion-forcing repo ✓、Causal-Forcing ✓、rCM ✓、CameraCtrl ✓、null-text ✓。
- 勘误入册：SkyReels-V2 = 2504.13074（2504.02436 为 SkyReels-A2）。

## 通用注意事项（P9 写作 agent 必读）

1. **先收录后引用**：正文只能引用 references.md 已收录条目；本轮核实日期统一 2026-11-13。
2. **约定页是唯一记号口径**：$(\tau,s,s)$ 压缩率、$1+(T-1)/\tau$ latent 帧数（81→21）、$N=t\cdot h\cdot w$、W4 扩散三条 recap——照约定页写。
3. **承重断言的出处等级**：Wan2.2 MoE 的 27B/14B 与 SNR 路由只有 repo README 一手（无论文）——正文引原文措辞并注明「官方 README」；Mochi/Sora 全 [C]；Movie Gen [B] 不引实现。
4. **勘误与陷阱**：SkyReels-V2 ID 勘误（勿用 2504.02436）；CausVid 无 GAN loss（GAN 属 DMD2）；DreamBooth [B]（repo 仅数据集）；THUDM→zai-org、genmoai/models→genmoai/mochi 重定向引用写新地址。
5. **数字一律标日期回原表**：VAE 压缩率/config 字段/参数量均引论文原表、README 原文或 config 实测（本文件已给出处）；token 账/FLOPs 账用约定页公式现推。
6. **ar5iv 缺口**：HunyuanVideo（2412.03603）ar5iv 不可用——需要论文细节时读 arXiv PDF/HTML 或本文件给的 README 摘录，勿凭记忆。
