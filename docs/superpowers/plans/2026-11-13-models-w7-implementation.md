# models 维度 W7 实施计划（P10 Audio & Speech，4 章）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 完成 models 维度 W7（收官波）：P10 语音音频（4 章：音频表征与 neural codec、ASR、TTS、Speech LLM 与全双工）。CTC 与 Moshi 是全波数学重心；codec 参数对照与 2025-2026 时效性是事实重心。

**Architecture:** 沿用已验证流水线（research pass → 分批并行写作（opus）+ 批量审查（opus）→ 波末 evaluator（fable）→ 修复复审 → 清扫收尾）。

**Tech Stack:** 同前。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（§5 P10 条目）

## Global Constraints

（沿用 + 累积裁决）：
- 字数 = 正文 CJK 汉字数（标准章 ≥3500、推导重章 ≥5500 软、短篇 ≥2500）
- 白名单现行锁定版（含 W4 收编 `\det`/`\varnothing`）
- 写作者不做网络调研；references.md 回写由写作者在各自 commit 内做（并行时收归 controller）
- 并行 git 纪律：禁用 --amend / reset；index.lock 等 5 秒重试
- 防崩纪律：curl 抓到网页内容不粘贴大段原文
- P10 特查：RVQ bitrate 账（n_q·log2K·fps）逐家重算、CTC/RNN-T 递推公式符号精确性、各家 codec 帧率与 codebook 参数对照真实性、Qwen2.5-Omni（block-wise DiT）与 Qwen3-Omni（Code2Wav）vocoder 方案勿混淆

## Review Focus

1. **CTC 边缘化与 forward-backward 递推**（10.2 核心——对齐路径集合、α/β DP 推导）→ 波末强制重推导
2. **Moshi RQ-Transformer 两级分解 + multi-stream 联合分布**（10.4 核心——全双工数学）→ 波末强制重推导
3. **codec 参数对照真实性**（SoundStream/Encodec/DAC/Mimi/WavTokenizer/XCodec2 的帧率、codebook 数、bitrate）→ research pass + evaluator 抽查
4. **evaluator rubber-stamp** → verification log 硬要求

---

## W7 任务

### Task 1: W7 research pass（事实源锁定）

**Files:** Modify `models/references.md`；Create `docs/research/w7-sources.md`

- [ ] **Step 1: 核实与收录**——SoundStream 2107.03312、Encodec 2210.13438、DAC 2306.06546、Moshi/Mimi 2410.00037、HuBERT 2106.07447、w2v-BERT 2108.06209、WavTokenizer 2408.16532、XCodec2 2502.04128、Whisper 2212.04356、CTC（Graves et al., ICML 2006，无 arXiv——按 Vincent 2011 先例用 venue+DOI 条目）、RNN-T 1211.3711、SenseVoice 2407.04051、FireRedASR 2501.14350、Qwen3-ASR（需定位）、VALL-E 2301.02111、VALL-E 2 2406.05370、CosyVoice 2407.05407、CosyVoice 2 2412.10117、CosyVoice 3 2505.17589、F5-TTS 2410.06885、E2-TTS 2406.18009、VoiceBox 2306.15687、NaturalSpeech-3 2403.03100、MaskGCT 2409.00750、HiFi-GAN 2010.05646、BigVGAN（需定位 arXiv ID）、VITS 2106.06103、FastSpeech2 2006.04558、IndexTTS/MegaTTS3/Spark-TTS（需定位）、Qwen2.5-Omni 2503.20215、Qwen3-Omni 2509.17765、GLM-4-Voice 2412.02612、AudioLM 2209.03143、Kimi-Audio 2504.18425、Step-Audio 2502.11946、LLaMA-Omni（需定位）；repo 存在性与关键文件路径（openai/whisper、espnet/espnet、FunAudioLLM/SenseVoice、FunAudioLLM/CosyVoice、SWivid/F5-TTS、jaywalnut310/vits、jik876/hifi-gan、kyutai-labs/moshi、QwenLM/Qwen3-Omni、zai-org/GLM-4-Voice、MoonshotAI/Kimi-Audio）；spec ⟨verify⟩ 项全部解决
- [ ] **Step 2: 约定页**——音频记号（采样率/帧率 fps、波形 x[n]、谱 X[m,k]、RVQ 记号沿用 6.3）；CTC 记号（对齐路径 π、blank、$\mathcal{B}^{-1}$）
- [ ] **Step 3: 写 w7-sources.md**（每章一节）+ references.md 入库
- [ ] **Step 4: 自检 + Commit**（`research: lock W7 sources (P10)`）

### Task 2-5: P10 四章（分批并行写作）

**Files:** Create `models/10-audio-speech/{01-audio-representations,02-asr,03-tts,04-speech-llm-and-duplex}.md`

分批（依赖序）：batch 1 = 10.1+10.2（表征/codec + ASR——10.2 依赖 10.1 特征账）→ batch 2 = 10.3+10.4（TTS 依赖 10.1 codec；10.4 依赖全部——Moshi 用 Mimi、TMRoPE recap 8.2）

**内容要求（spec §5-P10 各节全量）**：
- 10.1 audio-representations：数字音频基础（Nyquist 公式、16k/24k/44.1k 账）；**STFT 完整公式**（窗、hop/overlap、时频不确定性）；**mel filterbank**（HTK/Slaney 公式、三角滤波器组、log 压缩）；特征账（16kHz/25ms/10ms → 100fps，与文本序列长度对比表）；**semantic tokens**（HuBERT 离线 k-means + masked prediction 迭代公式、w2v-BERT、FACodec content/speaker 分离视角）；**acoustic tokens**：SoundStream/Encodec（conv encoder stride 下采样账 24kHz→75Hz + **RVQ（recap 6.3）+ bitrate = n_q·log2K·fps 实例账 8×10×75≈6kbps** + multi-scale STFT discriminator）；DAC 改进；**Mimi**（split-RVQ + 第一 quantizer 语义蒸馏 loss、12.5Hz——Moshi 基石）；WavTokenizer、XCodec2；**semantic vs acoustic 对照表**；离散 token vs 连续特征进 LM 分析（10.4 伏笔）
- 10.2 asr（推导重章）：ASR 形式化（长度不匹配：100fps 声学 vs ~10fps 文本）；**CTC 完整数学**（条件独立假设、对齐路径集合 $\mathcal{B}^{-1}(y)$、边缘化求和、**forward-backward α/β 递推 DP 推导**、blank 边界语义、peaky 行为与 LM 能力弱分析）；CTC+attention hybrid（ESPnet，$L=\lambda L_{CTC}+(1-\lambda)L_{att}$）；**RNN-T**（2D lattice forward、joint network）；CTC/RNN-T/AED 对照表（流式/成本/精度）；**Whisper 走读**（log-mel-80 recap、30s 窗 padding 账、decoder prefix special tokens 控制机制、680k hrs 弱监督、鲁棒性来源）；流式 chunk attention（lookahead 账）；SenseVoice、FireRedASR、Qwen3-ASR；**LLM-based ASR**（audio encoder → projector → LLM，与 8.1 connector 同构 recap）
- 10.3 tts（推导重章）：TTS 形式化（一对多映射）；级联管线（G2P/duration/MFA/acoustic model/vocoder）；**HiFi-GAN**（上采样栈 + MRF + **MPD 质数周期 2D reshape 公式** + feature matching loss）；**BigVGAN**（snake 激活抗混叠公式）；**VITS**（VAE + flow 后验灵活化 + MAS 单调对齐 + 对抗，端到端统一目标）；FastSpeech2 variance adapter；**VALL-E**（AR 建 RVQ 第一层 + NAR 逐层其余——两阶段公式；3s prompt in-context cloning 机制）；VALL-E 2 [B]；**CosyVoice**（S3Tokenizer 监督语义 token + LM + **FM decoder（recap 7.5）** + chunk-aware 流式）；CosyVoice3；**F5-TTS/E2-TTS**（text infilling formulation + **CFM 目标 + DiT + Sway sampling 走读**）；VoiceBox [B]；NaturalSpeech-3 FACodec 子流分解；MaskGCT（recap 7.7）；2025 批一段式（IndexTTS/MegaTTS3/Spark-TTS）；**三路线对照表**（级联 vs LM vs FM/diffusion）
- 10.4 speech-llm-and-duplex（推导重章）：speech 理解 LLM（Whisper 式 encoder + adapter + LLM）；音频-文本时间对齐（interleaved chunk）；**Qwen2.5/3-Omni Thinker-Talker**（Talker 接 Thinker hidden 的延迟机制、**TMRoPE**（recap 8.2 M-RoPE 扩展）、Code2Wav 流式 vocoder（Qwen3-Omni）vs block-wise DiT（Qwen2.5-Omni）勿混淆）；**Moshi 全公式**（本章深拆：**multi-stream** 并行建模无 turn-taking 假设、**Inner Monologue** 联合目标、**RQ-Transformer** 时间 12.5Hz 帧级 + depth RVQ 层级两级公式与参数账）；GLM-4-Voice（单 codebook 12.5Hz + 交错 token）；AudioLM 分层生成；Kimi-Audio/Step-Audio/LLaMA-Omni 一段式；**full-duplex 数学**（always-listen/always-talk、silence token、延迟预算账表 mic→VAD→LM→codec→speaker ~200ms 逐项）；**离散 vs 连续进 LLM 对照**（adapter 选择分析）；GPT-4o voice [C]

- [ ] **Step 1-5：标准章节流程**（各章 commit message "content(10.x): ..."）

### Task 6: W7 波末对抗审查

**Files:** Create `docs/reviews/w7-review-report.md`

**内容要求**：独立 fresh-context evaluator（fable），三任务 + **强制重推导：10.2 CTC 边缘化 + forward-backward、10.4 RQ-Transformer 两级分解 + multi-stream 联合分布**；时效性专项（Qwen3-Omni/Qwen3-ASR/CosyVoice3/Spark-TTS 等 2025-2026 现状）；codec 参数对照抽查；verification log 硬要求。

### Task 7: W7 修复、复审与收尾（全维度收官）

**内容要求**：S1/S2 修复 + 复审；S3 清零；进度表 4 行关闭（51/51）+ 审查记录区追加；spec 波次完成记录追加；术语表累积登记；commit。

---

## 完成定义（W7）

- 4 章全部通过：lint OK + 批量审查无未关闭 Important+ + evaluator 无未关闭 S1/S2 + verification log 落盘
- 进度表 4 行已关闭（全维度 51/51）；spec 波次记录已追加
