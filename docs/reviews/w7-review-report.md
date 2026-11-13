# W7 波次独立对抗审查报告（P10 Audio & Speech，10.1–10.4）

审查日期：2026-11-13。审查对象：`models/10-audio-speech/01–04` 四章成品。审查方式：fresh-context 独立 evaluator，全部关键断言回一手来源（arXiv API / ar5iv 全文 / HF config / GitHub API + 文件内容 grep）复核，数学重推导独立执行，不信任正文自述。

## 一句话总结

四章的 39 个 arXiv ID 与 23 个 repo、15 个文件/目录锚点今日实测全对，codec 六家参数表逐格回原文复核通过，全部 bitrate/参数/延迟账独立重算零误差，两条强制重推导（10.2 CTC 边缘化+forward-backward 全链、10.4 Moshi RQ-Transformer 两级分解+multi-stream 联合分布）核心数学全部通过且 CTC 数例与枚举逐项相符；但发现 1 条 S1——10.3 §4 称 BigVGAN 低通抗混叠「原文消融记录该组件引入约 19 倍的差距量级」，ar5iv 全文取证证伪（全文无任何 19× 消融数字，唯一的 19.59× 是 WaveFlow 的合成速度，Table 4/7/8 的 w/o-filter 消融均为小数点级差异），另有 2 条 S2（均在 10.4 §4.3：「原文仅给定性 smaller」被 Moshi Table 1 的 6L/1024d/16H 证伪；「原文按音频码本计深度步数（K=8 步）」误读原文 K=2Q+1=17 的全流口径）与 5 条 S3。

## 放行结论

**暂不放行（波末初查）**：S1 = 1（10.3），该章修复后须由 evaluator 复审；S2 = 2 条必须修复（所在章 10.4 无 S1，修复后按协议无需复审）；S3 = 5 条——收官波无下一波，建议随修复轮一并清零（逐条判定：修复/接受/驳回）。

## Findings 表

| # | severity | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| 1 | S1 | 10.3 §4（抗混叠段） | 「BigVGAN 的对策是给 snake 配 low-pass filter 做抗混叠（原文消融记录该组件引入约 19 倍的差距量级）」 | ar5iv 2206.04658 全文检索：不存在任何「19 倍」量级的抗混叠消融数字——全文独立「19」仅两处：Table 2 合成速度行 WaveFlow 的 19.59×（与滤波器无关）与参考文献 arXiv:1909.09577。真实的消融证据：Table 4（MUSDB18 OOD SMOS）BigVGAN-base 4.20 vs w/o filter 4.15 vs w/o filter & snake ≈4.1（差 ~0.05，Wilcoxon p<0.01）；Table 7/8（LibriTTS）PESQ 3.574→3.472、MCD 0.418→0.462 等，全部为小数点级差异。w7-sources.md 的「anti-alias ×19」调研记录同误（本章照抄） | 删去「约 19 倍的差距量级」；改写为原文口径：消融（Table 4/7/8）显示去滤波器显著掉点（MUSDB18 SMOS 4.20→4.15、统计显著），或只保留机制陈述不引数字 |
| 2 | S2 | 10.4 §4.3 参数账 (ii) | 「Depth——小模型，原文仅给定性『smaller』，不给未收录参数数」 | Moshi 论文（ar5iv 2410.00037）§3.4.1 正文：「As detailed in Table 1, our Depth Transformer has 6 layers, a dimension of 1024, and 16 attention heads」；Table 1 同载 MLP 4096。Depth 规模原文给全（6L/1024d/16H/MLP 4096 ≈ 75M 量级），「仅给定性」被证伪 | 改为引 Table 1 实测值（6 层、d=1024、16 头、MLP 4096），并可顺势补一笔参数对比账（≈7.5×10⁷ vs Temporal 7B，约 1%）——反而加强本节「税从宽度转移到小模型」的论点 |
| 3 | S2 | 10.4 §4.3 计数口径说明段 | 「原文按音频码本计深度步数（K = 8 步）」 | 原文 Figure 3 说明与 §3.4.1 的 K 是**子序列总数**：「breaks down a flattened sequence of length K·S … Depth Transformer over K steps」；§3.4.2 Eq (6) 明确「amounting to a total number of K = 2Q+1 streams, with Q = 8」即 K = 17（含用户流 8 条），「the number of steps in the Depth Transformer is at most K」按 17 计。本章的 8 步口径对应的实际是官方实现（kyutai-labs/moshi lm.py 实测：`dep_q = 8`、`n_q = 16`，文本由 `text_linear` 从 temporal 输出直出、用户流推理只喂不采） | 改为：「原文深度链按全流 K = 2Q+1 = 17 计（训练时用户流也建模——『modeling the user stream as output allows generating simulated dialogues』；部署时用户流喂入不采样）。官方实现 lm.py dep_q = 8：Depth 本体只跑 Moshi 侧 8 个码本步，文本步是 c_s 上的线性头——本章 Q′ = 9 由此而来」 |
| 4 | S3 | 10.2 式 (17) | joint network 写作 h(t,u) = tanh(W^f f_t + W^g g_u + b)、P = softmax(W^o h) | Graves 2012 原文（ar5iv 1211.3711 §2.3 式 (12)(13)）的 joint 是无隐层直接加性：h(k,t,u) = exp(f_t^k + g_u^k) 再归一化；tanh 隐层联合是后续实践（Graves 2013 系 / ESPnet 默认）的标准形。式 (18)(19) 与原文逐字相符，仅 (17) 越出原文口径 | 加一行口径注记：「式 (17) 取现代标准形（tanh 联合，ESPnet 默认）；2012 原文的联合是逐类直接相加 exp(f^k+g^k) 归一，无隐层」 |
| 5 | S3 | 10.1 §5.2 | 「单码本 K = 4096（论文码本表中该行利用率 100%；另有 8192 / 16384 两档）」 | ar5iv 2408.16532 码本表实为四行：16384（27%）、8192（68%）、4096（100%）、1024（100%）——漏 1024 行 | 补「另有 1024（亦 100% 利用率）/ 8192 / 16384 三档」 |
| 6 | S3 | 10.4 §4.2 式 (4) | 帧内序为「用户流 → 文本 → 模型音频」（用户流因子在前） | 原文 Eq (6) 的深度序为文本 → Moshi 音频（含 delay τ）→ 用户音频——用户流殿后，即训练时用户流预测条件于 Moshi 同帧 token；本章的因式序是链式法则下的合法重排且部署语义正确（模型流不条件于同帧用户流这一点两口径一致），但与原文 Eq (6) 字面链序不同且未声明 | 加一行注记：「原文 Eq (6) 的深度序把用户流排在模型流之后（训练口径，配 delay τ）；式 (4) 取部署相关的条件链重排——模型流不读同帧用户流的结论在两口径下相同」 |
| 7 | S3 | 10.1 式 (12) | L_distill = Σ_t (1 − cos(e_sem,t, g(u_t)))，g 标作「可学变换」作用于 WavLM 侧 | 原文：「We apply a linear projection with an output dimension of 1024 to the output of the first RVQ level… We then compute a cosine distance between the output of the first quantizer and the transformed WavLM embeddings」——可学投影加在量化器输出侧（投到 1024d），不是 WavLM 侧 | 把 g 移到 e_sem 侧：cos(g(e_sem,t), u_t)，或注明「投影位置两实现等价、原文在量化器输出侧」 |
| 8 | S3 | references.md（非正文章节文件） | `whisper` 行被引章节 =「10.2, 10.4」 | grep 四章：10.1 §2.3 正文与参考节均引（key: whisper）；10.4 全文无 whisper 引用。应为「10.1, 10.2」 | 回写为「10.1, 10.2」 |

无其他 S1。

## Verification log（抽查项 × 来源 × 结论）

### 机械基线

| 项 | 来源 | 结论 |
| --- | --- | --- |
| lint 四章 | `scripts/lint-chapter.sh` | 4/4 LINT OK（禁令词、模板节、metadata、arXiv 收录、证据等级、`$$` 配平全过） |
| 篇幅（CJK 字数，去 code block 与 math 后正则计字） | 自算 | 10.1=4036 / 10.2=6245 / 10.3=7075 / 10.4=7041；10.1 ≥3500 达标，10.2/10.3/10.4 ≥5500 软目标达标 |
| ASCII 图 | 逐章目检 | 四章均 ≥2（10.1 codec 骨架+split-RVQ；10.2 CTC 状态网格+RNN-T 格点；10.3 级联管线+VALL-E 流程+F5 infilling；10.4 三流栅格+RQ 帧内生成流） |
| 模板五件套 | 逐章目检 | metadata 块/总结（表格）/练习（各 4 题）/参考 四章齐全 |
| STYLE_GUIDE 抽查 | grep + 目检 | 无外链图、无 Mermaid、禁令词零命中（「强大的/革命性的/值得注意的是/综上所述/总的来说/众所周知/在当今」全零）；「第一个」仅出现于 10.4 §4 且逐字引原文 first multi-stream 主张、限定「开放系统」✓；首现「中文（English）」格式抽查（连接时序分类/坍缩映射/前向变量/占有后验/折叠/字素转音素等）合规；章末无空洞评论段；SoundStream/w2v-BERT/RNN-T/FastSpeech2/VALL-E 系/E2-TTS/VoiceBox/NS3/AudioLM 的 [B] 标注与「未开源」注记逐处在位；GPT-4o voice 全段 [C] 纪律执行（不引数字、不推测架构）✓ |

### arXiv ID ↔ 标题（arXiv API 两批 39 条，2026-11-13）

1211.3711=RNN-T ✓；2006.04558=FastSpeech 2 ✓；2010.05646=HiFi-GAN ✓；2106.06103=VITS ✓；2106.07447=HuBERT ✓；2107.03312=SoundStream ✓；2108.06209=W2v-BERT ✓；2206.04658=BigVGAN ✓；2209.03143=AudioLM ✓；2210.13438=EnCodec ✓；2212.04356=Whisper ✓；2301.02111=VALL-E ✓；2306.06546=DAC ✓；2306.15687=Voicebox ✓；2403.03100=NaturalSpeech 3 ✓；2406.05370=VALL-E 2 ✓；2406.18009=E2 TTS ✓；2407.04051=FunAudioLLM ✓；2407.05407=CosyVoice ✓；2407.10759=Qwen2-Audio ✓；2408.16532=WavTokenizer ✓；2409.00750=MaskGCT ✓；2409.06666=LLaMA-Omni ✓；2410.00037=Moshi ✓；2410.06885=F5-TTS ✓；2412.02612=GLM-4-Voice ✓；2412.10117=CosyVoice 2 ✓；2501.14350=FireRedASR ✓；2502.04128=Llasa ✓（标题勘误执行到位：正文与参考节均写 Llasa 并注 X-codec2 出处）；2502.05512=IndexTTS ✓；2502.11946=Step-Audio ✓；2502.18924=MegaTTS 3 ✓；2503.01710=Spark-TTS ✓；2503.20215=Qwen2.5-Omni ✓；2504.18425=Kimi-Audio ✓；2505.17589=CosyVoice 3 ✓；2506.21619=IndexTTS2 ✓；2509.17765=Qwen3-Omni ✓；2601.21337=Qwen3-ASR Technical Report ✓。**39/39 命中，0 错。** CTC 无 arXiv 走 venue+DOI：doi.org/10.1145/1143844.1143891 今日实测 302 → portal.acm.org 对应条目 ✓。作者抽查 12 条（2025-2026 新条目全部）：Deng/Zhou/Jiang/Wang/Xu(×2)/Du/Shi/Fang/Huang/KimiTeam/Ye 与正文署名全对 ✓。

### repo / 文件路径（GitHub API，2026-11-13 实测）

- 存在性 23/23：facebookresearch/encodec、kyutai-labs/moshi、descriptinc/descript-audio-codec、facebookresearch/fairseq、openai/whisper、espnet/espnet、QwenAudio/SenseVoice、FireRedTeam/FireRedASR、jik876/hifi-gan、jaywalnut310/vits、QwenAudio/CosyVoice、SWivid/F5-TTS、open-mmlab/Amphion、NVIDIA/BigVGAN、index-tts/index-tts、bytedance/MegaTTS3、SparkAudio/Spark-TTS、QwenLM/Qwen2.5-Omni、QwenLM/Qwen3-Omni、zai-org/GLM-4-Voice、MoonshotAI/Kimi-Audio、stepfun-ai/Step-Audio、ictnlp/LLaMA-Omni 全部 200 ✓
- 文件/目录级 15/15：moshi `moshi/moshi/models/compression.py` ✓（内容级：`class MimiModel(CompressionModel)` L105、frame_rate 接口 ✓）；whisper `whisper/model.py` ✓（内容级：conv1 kernel 3 + `conv2 ... stride=2` L186-187——10.2 式 (20) 的 stride-2 物证 ✓）；hifi-gan `models.py` ✓（内容级：`feature_loss` L1 逐层激活 L251、`discriminator_loss`/`generator_loss` 为 (1−d)²/d² 最小二乘形 L260-274——10.3 §3「least-squares 变体 models.py 实测」✓）；vits `models.py` ✓；CosyVoice `cosyvoice/{tokenizer, flow, llm, hifigan}` 四目录 ✓；F5-TTS `src/f5_tts` ✓；GLM-4-Voice `flow_inference.py` + `speech_tokenizer/` ✓；Kimi-Audio `kimia_infer/` + `infer.py` + `finetune.py` ✓；fairseq `examples/hubert` ✓
- 代码内容级（moshi lm.py 实测，finding #2/#3 物证）：`n_q: 16, dep_q: 8`（loaders.py）→ num_codebooks = 17 = 2Q+1 ✓；`LMGen._step` 中 `needed_tokens = num_codebooks − dep_q − 1 = 8`（用户流 8 码本只喂不采）✓；`forward_depformer_training` 的 depformer 步 0 吃 `depformer_text_emb(sequence[:, 0])`（文本）、步 j≥1 吃码本 embedding，`linears[0..7]` 预测 Moshi 的 8 个音频码本；文本由 `text_linear(transformer_out)` 直出——与 10.4 式 (5)(6) 的结构（求和嵌入 17 流、文本第 0 步、8 码本深度步）逐项一致 ✓

### HF config（hf-mirror，2026-11-13 实测）

- kyutai/mimi config.json：codebook_size 2048、frame_rate 12.5、hidden_size 512、`num_semantic_quantizers: 1` ✓（10.1 §5.1 三字段物证 + split-RVQ 1+7 的 config 级旁证）
- openai/whisper-large-v3 config.json：d_model 1280、decoder_layers 32、decoder_attention_heads 20 ✓（10.2 §6.1 逐字一致）

### 强制重推导 1：10.2 CTC 边缘化 + forward-backward —— PASS

- 对齐路径集合：$y = ab$、$T = 3$ 的 27 条候选路径 brute-force 枚举，$\mathcal{B}^{-1}(ab)$ 恰 5 条——$(a,a,b)$/$(a,b,b)$ 各 0.12、$(a,b,\varnothing)$/$(a,\varnothing,b)$ 各 0.06、$(\varnothing,a,b)$ 0.072，合计 **0.432**——与正文逐条逐项相符 ✓
- α 递推独立实现（式 (9)(10) 含两格步进合法性条件与越界归零）：α₁=(0.3,0.5,0,0,0)、α₂(2)=0.32、α₂(3)=0.10、α₂(4)=0.20、α₃(4)=0.372、α₃(5)=0.06——与正文全等 ✓；终止式 (11) 0.372+0.06=0.432 = 枚举 ✓
- β 按式 (13) 独立实现（含 δ_s 开关）：β 全表算出后与 α 逐帧相乘求和，**t=1/2/3 三处均得 0.432**——恒等式 (14) 全帧数值验证 ✓
- 式 (10) 条件与 §2.3 规则的「到点/出点」等价性独立论证：奇位恒 blank 蕴含两写法等价 ✓；长度约束 T ≥ U + r（重复处数）独立论证成立 ✓（无重复时 2(T−1) ≥ 2U−2 ⟺ T ≥ U ✓）
- logit 梯度式 (15) 独立推导：∂P/∂P_t(k) = Σ_{s:y'_s=k} αβ/P_t(k)（经式 (14)），过 softmax 雅可比合并两项得 P_t(k) − Σ αβ/P(y|x) ✓——符号与求和范围与正文一致
- RNN-T 配合项：式 (18)(19) 与 1211.3711 §2.4 式 (16)(17) 逐字对照（α(1,0)=1、终止乘 ∅(T,U)）✓；forward 变量定义引文逐字命中 ✓；唯一偏差在式 (17) 的 joint 形态（finding #4，S3）

### 强制重推导 2：10.4 Moshi RQ-Transformer 两级分解 + multi-stream 联合分布 —— PASS（数学），附 2 条 S2 出处口径

- 式 (4) 独立重推：展平链式法则按帧分组 + 帧内固定序（用户码 → 文本 → 模型码）展开；**同帧跨流独立的进入位置**定位在两个因子上——P(w_s|·) 删去 a^u_{s,1:K}、P(a^m_{s,k}|·) 删去 a^u_{s,1:K}——正文「模型流只条件于历史」与此一致 ✓；该前提与原文结构互证（z_s = Tr_Temp(V_0..V_{s-1}) 不含 V_s；深度序中 Moshi 流在用户流之前，同帧用户码本不可见）✓
- 式 (5)(6) 两级公式与原文 §3.4.1 式 (1)(2)(3) 逐项对照：c_s ↔ z_s（Temporal 只读 V_{≤s-1}）✓；文本步 l_{s,1} = Lin(z_s) ↔ 正文「文本步是 c_s 上的线性头」✓；深度步 l_{s,k} = Tr_Depth(z_s, V_{s,1..k-1}) ↔ 式 (6) ✓；kyutai-labs/moshi lm.py 实测（depformer_in 投影 dim→depformer_dim ↔ c 的 W_o；dep_q=8；文本头独立）代码级落实 ✓
- 式 (7) 瓶颈替换：帧内一半 = 链式法则逐字展开无近似 ✓；跨帧一半 = 历史强制流过 c_s 的架构约束，正文定性正确 ✓
- 效率账独立重算：展平基线每帧 attention FLOPs ∝ L·Q·(QS)·d = LQ²Sd ✓；RQ = L(S·d_t + Q′²d_d) ✓；累计 O(LQ²S²d) vs O(LS²d_t + LSQ′²d_d) ✓；「Q² → 1」读法成立 ✓；10 分钟实例 S = 600×12.5 = 7500、展平 cache 读 17×7500 ≈ 1.3×10⁵ 条 vs RQ 7.5×10³ 条、17× 差 ✓；cache 增长 +17/帧 vs +1/帧 ✓
- 参数账：16×2048×4096 = 1.34×10⁸ ≈ 1.3 亿 ✓；/7B ≈ 1.9%「百分位级」✓；Helium 7B / 2.1T tokens 原文命中 ✓
- 两条出处口径错误（finding #2/#3，S2）：「原文仅给定性 smaller」被 Table 1（Depth 6L/1024d/16H/MLP4096）证伪；「原文按音频码本计深度步数（K=8）」误读——原文 K=2Q+1=17 全流计，8 步是官方实现 dep_q=8 的口径——均属源文献表述错误，不动本章自身数学
- Inner Monologue/delay：式 (8) 强制对齐构造与原文 PAD/EPAD 铺法一致（「padding tokens represent about 65%」原文旁证）✓；「文本超前→流式 TTS、文本滞后→流式 ASR」与原文 delay 翻转段逐句对应 ✓；四阶段训练（无监督预训→simulated multi-stream→Fisher→instruction）与原文一致 ✓

### 论文级断言抽查（ar5iv 全文取证，短句核对，36 篇）

- codec 六家（10.1 承重表逐格）：SoundStream「represented by S=75 frames」「bits … equal to N_q log_2 N」「3 kbps to 18 kbps」「3 kbps outperforms Opus at 12 kbps and approaches EVS at 9.6 kbps」✓；EnCodec「75 latent steps per second… 150 at 48 kHz」「at most 32 codebooks (16 for the 48 kHz models) with 1024 entries」「1.5, 3, 6, 12 or 24 kbps」「5 different scales… [2048, 1024, 512, 256, 128]」✓；DAC Table 1「44.1 / 8 kbps / 512 / 86 / 9 / 91.16」✓、factorized codes「8d or 32d … 1024d」✓、quantizer dropout「actually hurts the full-bandwidth audio quality」✓；Mimi split-RVQ/余弦蒸馏/8×2048@12.5Hz=1.1kbps/WavLM 池化（stride 4、non-causal 关键性）✓；WavTokenizer「single quantizer with 40 or 75 tokens」+ 码本表 ✓（finding #5）；Llasa/X-codec2「unified codebook」、16kHz、R=320、K=65536 ✓
- Whisper 走读（10.2 §6 全节）：「680,000 hours」✓、「80-channel log-magnitude Mel… 25-millisecond windows with a stride of 10 milliseconds」✓、「30-second segments」✓、「special tokens that serve as task specifiers or classification targets」✓、「no-speech probability threshold of 0.6 and the average log-probability threshold of −1」✓
- ASR 余项：HuBERT「100 clusters… two iterations」✓；w2v-BERT 标题即「Combining Contrastive Learning and Masked Language Modeling」✓；RNN-T「extends CTC… all lengths… jointly modelling」✓；SenseVoice 非 AR + 5×/15× ✓ + 多语/情感/事件多任务 ✓；FireRedASR「serve as an effective speech representation module in LLM-based speech models」✓；Qwen3-ASR 1.7B/0.6B/ForcedAligner-0.6B/52 语言/AuT+Qwen3-Omni 基座 ✓
- TTS：HiFi-GAN MPD「We set the periods to [2, 3, 5, 7, 11]」「reshape 1D raw audio of length T into 2D data of height T/p and width p」✓、one generator + MSD + MPD ✓；BigVGAN snake 式与 α 可训练 ✓、抗混叠结构（2× 上采样+snake+低通）✓——**「19 倍」证伪见 finding #1**；VITS「variational inference augmented with normalizing flows and an adversarial training process」+ MAS + stochastic duration predictor ✓；FastSpeech2 variance adaptor 引文 ✓；VALL-E AR 第一层/NAR c_{:,j∈[2,8]}/3s prompt 三处引文 ✓；VALL-E 2 repetition aware sampling + grouped code modeling ✓；CosyVoice「inserting vector quantization into the encoder」+ LLM + conditional FM ✓、「S3Tokenizer」三篇正文零命中（命名纪律执行正确）✓；CV2 chunk-aware causal FM/FSQ/25Hz/Qwen2.5-0.5B/去 text encoder 与 speaker embedding 五项 ✓；CV3 MinMo 换 SenseVoice-Large 原文直述 ✓；F5-TTS infilling+CFM/filler 补齐/ConvNeXt V2/Sway Sampling（式 (7) f_sway 方向与正文定性读法一致；注意原文 t→0 为噪声端，本书 7.5 约定 t=0 数据端，正文方向性陈述已按本书约定正确翻转）✓；E2-TTS 逐字符 filler + audio infilling ✓；VoiceBox text-guided speech infilling + OT path ✓；NS3 FVQ 四子空间 ✓；MaskGCT mask-predict 两级 ✓；IndexTTS GPT-style+拼音+标点 ✓；IndexTTS2 情感+时长可控 ✓；MegaTTS3 标题级 ✓；Spark-TTS BiCodec 单流引文 ✓
- Speech LLM：Qwen2.5-Omni TMRoPE 命名/Thinker-Talker dual-track 直读 hidden/「transformed into a mel-spectrogram using Flow-Matching」（sliding-window block attention DiT）+「modified BigVGAN」/音频编码器「attention in blocks of 2 seconds each」✓；Qwen3-Omni「Code2Wav replaces block-wise DiT with a lightweight convolutional network」/12.5Hz/「single-frame, immediate」/MTP 产残差码本/MoE/119 语言/TM-RoPE 沿用 ✓——**两代 vocoder 区分（S1 级陷阱）正文锁死正确**；GLM-4-Voice 175bps 单码本 12.5Hz VQ 瓶颈 + FM decoder ✓；AudioLM 三层引文 ✓；Kimi-Audio 13M hrs + FM detokenizer + 预训 LLM 初始化 ✓；Step-Audio 130B ✓；LLaMA-Omni 引文 ✓

### 数值重算（python 独立执行，除 findings 所列外零误差）

bitrate 账：SoundStream 8×10×75=6000 ✓；EnCodec 五挡 1500/3000/6000/12000/24000 ✓；DAC 9×10×86=7740（44100/512=86.13≈86 ✓）；Mimi 8×11×12.5=1100 ✓；WavTokenizer 900/480 ✓；X-codec2 16×50=800、16000/320=50 ✓；GLM-4-Voice 175/12.5=14bit→K=16384 ✓。PCM 参照：24k×16=384kbps（6kbps=64×）✓、16k×16=256kbps（0.8kbps=320×）✓。STFT 账：Nw=400、H=160、100fps、Δf=40Hz ✓；时频积为常数 ✓。Whisper：3000→1500 ✓。token rate：600 vs 100 tok/s（6×）✓。10.4：160ms=2×80ms ✓；17×/1.3×10⁵/7.5×10³ ✓；1.3 亿/1.9% ✓。练习题面数值抽查：10.1 练习 1(d) 4096 码本→1.2kbps 而 token rate 不变 ✓；练习 2 Nw=1764/H=882/50fps ✓；10.2 练习 4(c) 80ms/240ms ✓；10.4 练习 2(a) 25Hz→80ms ✓

### 跨章引用完整性

10.1 引 6.3 式 (20)(21)（RVQ/bitrate）✓、6.3 §7.4 离散 vs 连续选择表 ✓、6.2 式 (2)-(5) InfoNCE ✓、8.1 connector ✓、8.5 式 (4)（I ≤ N log2 K）✓；10.2 引 3.2 式 (11)-(13)（enc-dec 三式）✓、8.1 式 (2)（v = φ(z_img)）✓；10.3 引 7.5 式 (9)（CFM）✓、7.1 式 (3)（GAN minimax）✓、7.6 DiT ✓、7.7 mask-predict ✓；10.4 引 8.2 式 (14)-(17)（M-RoPE 三元组与 [16,24,24] 频段切分、文本一致性推论）✓、8.2 §9.1（Qwen2.5-VL t 轴绝对秒对齐）✓、2.4 式 (9)（KV cache 账）✓、4.5（MTP——4.5 章确含 MTP §6 与接受率账）✓。记号双载（K = mel 通道数/码本大小；t = 帧下标/文本下标/flow 时间）各章开场均有声明 ✓。references.md 被引章节字段逐条 grep 核对：除 `whisper` 行（finding #8）外 Part 10 全部 40 论文行 + 18 锚点行与四章实际引用一致 ✓

### 时效性与证据分级（2026-11-13 波末实测）

- 后继存在性：arXiv 检索 "Qwen4-Omni" / "CosyVoice 4" / "GLM-5-Voice" 零命中；title:"Moshi" 检索无后继模型（仅对抗攻击与环境足迹两类旁证论文）；10.4 §7 表「当代系统的收敛解几乎都是混合」为表内系统的归纳陈述、非裸时效断言 ✓；章节无「最新/首个」未标注日期的裸断言（10.4 §4「第一个」逐字引原文且限定开放系统）✓
- 2025-2026 条目现状：Qwen3-Omni（2509.17765）/Qwen3-ASR（2601.21337）/CosyVoice3（2505.17589）/Spark-TTS（2503.01710）/IndexTTS2（2506.21619）今日 API 复核标题、作者、关键口径全部在线一致 ✓
- [C] 纪律：GPT-4o voice 全段只陈述已公开定位 + 三轴分析框架，未引任何未披露数字、未推测架构 ✓
- [B] 纪律：SoundStream/w2v-BERT/RNN-T/FastSpeech2/VALL-E 系/E2-TTS/VoiceBox/NS3/AudioLM 九处「无官方代码」注记逐处在位；VALL-E 2「标题级 human parity 主张，证据等级维持 [B]」注记在位 ✓

### 覆盖率声明

4/4 章逐行通读；39/39 个正文 arXiv ID 全量复核（非抽样）+ 12 条作者抽查；23 个 repo 存在性 + 15 个文件/目录锚点全量复测（其中 4 个做了内容级 grep：moshi compression.py 与 lm.py、whisper model.py、hifi-gan models.py）；2 个 HF config 全字段核对；36 篇论文 ar5iv 全文取证（覆盖 codec 六家参数表、Mimi split-RVQ+蒸馏、Qwen2.5/3-Omni vocoder 区分、VALL-E 两阶段、F5 infilling、HiFi-GAN MPD、Moshi 三机制、Whisper 走读全部指定承重断言）；2 条强制重推导完成（CTC 数例 brute-force + α/β/恒等式/梯度全链独立复算；Moshi 两级分解与联合分布独立推导 + 效率/参数账重算 + 官方代码交叉验证）；references.md 被引章节字段全量核对；STYLE_GUIDE 逐条抽查。未覆盖：练习题的完整解答推导（核对了题面数值口径）；CosyVoice/F5-TTS 等 repo 的 README 逐字复核（GitHub API 限流预算约束，存在性与目录级已覆盖）；10.3 §9 四家 2025 一段式的论文正文细读（均做到摘要级引文取证）。

## 修复闭环要求

- S1 ×1（finding #1，10.3 §4）必须修复；**10.3 修复后须由 evaluator 复审该章**（含 S1 章节复审协议）。
- S2 ×2（finding #2/#3，均在 10.4 §4.3）必须修复；10.4 无 S1，修复后按协议无需复审。
- S3 ×5（finding #4–#8）——W7 为收官波、无下一波 sweep 窗口，建议随本次修复轮一并清零（逐条判定：修复/接受/驳回）。
