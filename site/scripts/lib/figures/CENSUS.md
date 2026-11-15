# 块普查：每章的无语言围栏块（块序号从 0 计，供 replacesBlock 引用）

### 01-foundations/01-formulation
  - block 0（8 行）: `位置 t     :   1      2      3      4      5`
### 01-foundations/02-compute-accounting
  - block 0（10 行）: `x (4096) ──RMSNorm──► (4096) ──QKV proj──► Q(4096) K(1024) V(102`
### 01-foundations/03-scaling-laws
  - block 0（15 行）: `log L`
### 02-transformer-anatomy/01-overview
  - block 0（20 行）: `token ids  x_1..x_T             [B, T]         ← 整数下标（1.1 第 1 节）`
  - block 1（9 行）: `encoder-only              encoder-decoder                     de`
### 02-transformer-anatomy/02-tokenization-and-embeddings
  - block 0（10 行）: `原始字符串   "newest"`
  - block 1（9 行）: `bytes   b1 b2 b3 b4 b5 b6 b7 b8 ...`
### 02-transformer-anatomy/03-attention-mechanism
  - block 0（10 行）: `X [B, T, d]                          ← 2.1 式 (1) 中 RMSNorm(x) 的输`
  - block 1（10 行）: `X [B, T, 4096]`
### 02-transformer-anatomy/04-causality-and-kv-cache
  - block 0（12 行）: `输入：prompt token ids x_1..x_{T_p}，模型 f_θ，采样器 sample(·)`
  - block 1（12 行）: `decode 步 t，单层`
### 02-transformer-anatomy/05-positional-encoding
  - block 0（10 行）: `R_m 结构（d = 8 示意，θ_0 > θ_1 > θ_2 > θ_3）          频率轴（log 刻度）`
  - block 1（3 行）: `rope_scaling = { type: "yarn", factor: 40.0, beta_fast: 32, beta`
### 02-transformer-anatomy/06-ffn
  - block 0（10 行）: `x   [B, T, d]                        ← RMSNorm₂ 的输出（2.1 式 (1)）`
  - block 1（4 行）: `x [B, T, d]`
### 02-transformer-anatomy/07-norm-and-residual
  - block 0（8 行）: `Post-LN                              Pre-LN`
### 02-transformer-anatomy/08-assembly
  - block 0（6 行）: `x⁽ˡ⁻¹⁾ [B, T, 4096] ──┬─────────────────────────────────────► (+`
  - block 1（11 行）: `ids [B, T]`
### 03-objectives-and-variants/01-objectives-and-masks
  - block 0（18 行）: `原序列 x :  the  cat  sat  on  the  mat  and  slept        (T = 8)`
  - block 1（8 行）: `原序列 :  x1  x2  x3  x4  x5  x6        spans: [x2 x3]（起点 1）、[x5]（起`
  - block 2（3 行）: `原序列    :  the  cat  sat  on  the  mat  and  slept`
### 03-objectives-and-variants/02-encoder-architectures
  - block 0（9 行）: `enc-dec 数据流（decode 第 t 步，逐层，省略 B/H 维）`
  - block 1（11 行）: `ModernBERT-base 结构一瞥（22 层，global 周期 3，示意）`
### 03-objectives-and-variants/03-decoder-only-and-icl
  - block 0（6 行）: `s_K 的布局（K = 2，情感分类例）：`
  - block 1（8 行）: `序列:      …  [A]_i  [B]_{i+1}  ………  [A]_t  (下一位置)`
### 04-modern-architectures/01-kv-compression-and-exact-attention
  - block 0（6 行）: `MHA (g = 1)              GQA (g = 4)                 MQA (g = 32`
  - block 1（10 行）: `u_t [d = 7168]（RMSNorm 后的 stream，2.4 式 (6) 口径）`
  - block 2（8 行）: `Q [T, d] 切成 T/B_r 行块          K, V [T, d] 切成 T/B_c 列块`
### 04-modern-architectures/02-sparse-attention
  - block 0（9 行）: `1 2 3 4 5 6 7 8      第 1 层：位置 8 直接看到 {6,7,8}`
  - block 1（12 行）: `query q_t`
  - block 2（9 行）: `q_t ──► lightning indexer（H^I 个小维度头，ReLU 打分，可 FP8）`
### 04-modern-architectures/03-linear-attention-and-ssm
  - block 0（8 行）: `同一状态方程的三种算图（T = 6、chunk 长 C = 3 示意）：`
### 04-modern-architectures/04-hybrid-architectures
  - block 0（3 行）: `一个周期（4 层，model card 布局重绘）：`
### 04-modern-architectures/05-moe
  - block 0（4 行）: `h_i [d]（backbone 输出）  Emb(t_{i+1}) [d]（真实下一 token）`
### 04-modern-architectures/06-long-context
  - block 0（8 行）: `设备 p（常驻 Q_p 与 KV_p，P 步轮转）：`
### 04-modern-architectures/07-interpretability-primer
  - block 0（7 行）: `logit lens（§2）：       ▼          ▼                    ▼`
  - block 1（4 行）: `h ∈ R^d ─► encoder（ReLU / TopK / 门控）─► z ∈ R^n（稀疏：L0 ≪ n）`
  - block 2（8 行）: `duplicate token heads（早层）：在重复名 John 的第二次出现处标记「此 token 出现过」`
### 04-modern-architectures/08-frontier-outlook
  - block 0（8 行）: `前向（式 (1)），t: 0 → 1，逐位置独立腐蚀：`
### 05-case-studies/01-llama-lineage
  - block 0（5 行）: `Llama-4 文本侧 48 层 = 12 × 4 层周期（no_rope_layers 3:1）`
  - block 1（7 行）: `image [H, W, 3]`
### 05-case-studies/02-deepseek
  - block 0（5 行）: `DeepSeek-V3 层栈（61 层，d = 7168，untied 词表）`
### 05-case-studies/03-qwen
  - block 0（7 行）: `config.json                          modeling_qwen3_next.py`
### 05-case-studies/04-glm
  - block 0（5 行）: `Part A（双向）       ‖  Part B（乱序 span，span 内因果）`
  - block 1（5 行）: `GLM-4.6（92 层，d = 5120，untied 词表）`
### 05-case-studies/05-kimi-moonshot
  - block 0（4 行）: `Kimi-K2 层栈（61 层，d = 7168；architectures = DeepseekV3ForCausalLM）`
### 05-case-studies/06-cross-lab-comparison
  - block 0（6 行）: `per-token cache 斜率（bf16，全量驻留口径）与 1M 窗整窗账`
  - block 1（4 行）: `config.json ──► configuration_*.py ──► modeling_*.py`
### 06-vision-representations/01-vit
  - block 0（16 行）: `x [B, H, W, C]`
### 06-vision-representations/02-clip-and-contrastive
  - block 0（8 行）: `图像 x_1..x_B ─► 视觉塔 f ─► W_v ─► 单位化 ─► V [B, d] ─┐`
### 06-vision-representations/03-autoencoders-and-tokenizers
  - block 0（11 行）: `x [B, H, W, C]`
  - block 1（2 行）: `前向：  z_e ──argmin 查表──▶ z_q ──▶ D ──▶ x̂ ──▶ L`
  - block 2（5 行）: `z_e ─▶ Q1 ─▶ e1 ─┐`
### 07-diffusion-image-generation/01-generative-model-map
  - block 0（31 行）: `┌────────────────────────┐`
### 07-diffusion-image-generation/02-ddpm
  - block 0（9 行）: `前向 q（固定，无参数）：  q(x_t | x_{t-1}) = N(√α_t · x_{t-1}, β_t · I)`
  - block 1（9 行）: `训练（ε-参数化 + L_simple）                     采样（ancestral，T 步串行）`
  - block 2（8 行）: `ε_θ 的 I/O 契约（DDPM U-Net；结构细节 → 7.6）`
### 07-diffusion-image-generation/03-score-sde
  - block 0（14 行）: `前向 SDE（固定，无参数，加噪 t: 0 → 1）`
### 07-diffusion-image-generation/04-sampling-and-guidance
  - block 0（9 行）: `DDPM ancestral 采样（7.2 §4.4，T = 1000 步串行，随机）`
  - block 1（11 行）: `质量 ↑`
### 07-diffusion-image-generation/05-flow-matching
  - block 0（7 行）: `t: 0（数据端）────────────────► 1（噪声端）     采样方向相反：t: 1 → 0`
  - block 1（9 行）: `条件轨迹（固定端点对：直线、恒速 ε−x₀）          边际 ODE 轨迹（后验平均：弯曲）`
### 07-diffusion-image-generation/06-latent-diffusion-and-dit
  - block 0（11 行）: `像素空间 x [3, 512, 512]                    latent 空间（扩散的工作空间）`
  - block 1（14 行）: `Z ∈ R[N,d]（图像 latent token）        Y ∈ R[M,d]（文本 token）`
### 07-diffusion-image-generation/07-t2i-systems-and-ar-image
  - block 0（8 行）: `prompt ─► CLIP-L ─────┐ sequence 沿特征维拼接 ──┐`
  - block 1（5 行）: `r_1    r_2    r_3              r_10`
  - block 2（5 行）: `轮次   t=0        t=1         t=2                  t=T`
### 08-multimodal-llm/01-vlm-architecture-taxonomy
  - block 0（10 行）: `x_img [H, W, 3]`
  - block 1（3 行）: `序列布局（input-level）：`
### 08-multimodal-llm/02-connectors-and-resolution
  - block 0（7 行）: `z_img [N × d_v]（冻结 ViT，N = 576）        U [T × 768]（文本 token）`
  - block 1（6 行）: `ITC（对比）              ITM（匹配）               ITG（生成）`
  - block 2（7 行）: `z_img: 24×24 × d_v  （N = 576）`
  - block 3（10 行）: `原图 1568×1176                缩略图：整图 → 336×336 → 576 tokens（全局）`
  - block 4（7 行）: `head 维 128 = 64 个 2D 旋转块（角度 = 轴坐标 × θ_j）`
### 08-multimodal-llm/03-training-objectives
  - block 0（4 行）: `一条单轮样本的序列布局与 m_s：`
  - block 1（4 行）: `stage 1 对齐             stage 2 指令微调                stage 3 偏好对齐`
### 08-multimodal-llm/04-video-understanding
  - block 0（12 行）: `视频 T 秒`
### 08-multimodal-llm/05-unified-understanding-generation
  - block 0（4 行）: `理解方向:  x ──► SigLIP 塔（连续特征）──► MLP ──► v ──┐`
  - block 1（4 行）: `序列布局（式 (12)）：`
  - block 2（9 行）: `w1  w2  BOI  p1  p2  p3  EOI  w3`
  - block 3（8 行）: `1. 文本 AR decode …… 直到模型产出 BOI`
### 09-video-generation/01-video-vae
  - block 0（15 行）: `V ∈ R^{T×H×W×3}                    例：121×720×1280×3（Wan 式「首帧+120`
  - block 1（6 行）: `k_t = 3 的时间卷积（空间维略去）：`
### 09-video-generation/02-spatiotemporal-dit
  - block 0（10 行）: `z [t × h × w × c]   9.1 实例：Wan-VAE，81 帧 720p → 21 × 90 × 160 × 1`
  - block 1（5 行）: `dual-stream 段（前段）                  single-stream 段（后段）`
  - block 2（6 行）: `x [N × 5120]        N = t·h·w（patch=1 等效）；时间条件走 adaLN 族调制`
  - block 3（4 行）: `采样方向   t: 1 ───────────────────────────────► 0`
  - block 4（5 行）: `64 个旋转对，θ_j = base^(−2j/128)（j 小 = 高频短波长，j 大 = 低频长波长）`
### 09-video-generation/03-training-and-acceleration
  - block 0（9 行）: `双向教师（50 步 FM 采样；官方 repo 释出基座 Wan2.1-T2V-1.3B）`
  - block 1（13 行）: `prefill：文本条件编码（一次）＋（可选）初始帧 latent 的 K/V 写入 cache`
  - block 2（7 行）: `t_1..t_n 的配置空间（每 token 一个噪声水平，0 = 干净，1 = 纯噪声）：`
### 09-video-generation/04-control-and-editing
  - block 0（8 行）: `输入侧（每个去噪步）                        通道维拼接`
  - block 1（2 行）: `I_1 ─► [段 1: p(V_1 | I_1, c_1)] ─► V_1 ─► last(Dec(V_1)) = I_2 ─`
### 10-audio-speech/01-audio-representations
  - block 0（12 行）: `x[n]（24 kHz 波形）                            x̂[n]（重建波形）`
  - block 1（5 行）: `latent z（12.5 fps）`
### 10-audio-speech/02-asr
  - block 0（11 行）: `CTC 状态网格与三类步进（y = ab → y' = (∅, a, ∅, b, ∅)，S = 5；横轴 t = 1…T，纵轴 `
  - block 1（12 行）: `RNN-T 二维格点（T 帧 × U 个标签；路径 = (1,0) → (T,U) 的右移/上移交错）`
### 10-audio-speech/03-tts
  - block 0（13 行）: `文本 ──G2P──► 音素序列 y₁..y_U`
  - block 1（10 行）: `输入：音素序列 y + prompt 第一层码 C̃_{:,1}`
  - block 2（7 行）: `文本 ──► LLM（text-to-token 生成，自回归）`
  - block 3（6 行）: `mel 帧轴  t = 1..T`
### 10-audio-speech/04-speech-llm-and-duplex
  - block 0（9 行）: `12.5 Hz 栅格（每格 80 ms）上的三流并行（↓= 该帧生成，● = 输入观测）`
  - block 1（11 行）: `RQ-Transformer 一帧的生成流（帧 s；c_s 由 §4.3 式 (5) 在帧 s-1 末备好）`