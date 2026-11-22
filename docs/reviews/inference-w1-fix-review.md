# inference W1 修复轮复审报告（fix-review，fresh-context）

- 日期：2026-11-23
- 角色：修复轮独立复审者（不继承写作与首轮审查上下文；对 S1/S2 逐条 verdict + 强制重推导独立重推 + cascade 核 + 轻量修复抽查）
- 输入：evaluator-brief.md、inference-w1-review-p1.md（S1=0/S2=1/S3=5）、inference-w1-review-p2.md（S1=2/S2=3/S3=6）、writer-{A,B,C,D}-report.md Fix round 节、progress.md（命题编号裁定 = 章内重启）、8 章修后成品、spec §4.3/§5/§6.3、references.md、research notes、STYLE_GUIDE/R2/rubric
- 方法：全部修后数值 python3 独立重算；2.4 分层命中成本账独立重推（含 causal 系数对 models 2.4 式 (8) 精确求和交叉核）；一手来源自行 fetch（DeepInfer doc contents API、KIVI arXiv HTML、cuBLAS 官方 doc、NVIDIA H200 datasheet PDF、vLLM 论文 HTML、sglang contents API）；lint×8 + 篇幅 + 禁令词 + 习题分级 + 命题编号 + stale grep 机械核。**只读复审：未改章节、未 commit、未派 subagent。**

## 结论

**Status：原 S1×2 / S2×4 全部 ADDRESSED；2.4 强制重推导 PASS（原 FAIL 项）。修复文本引入新 S2×1（2.3 命题 1 边界情形句，与自身等号条件矛盾）+ 新 S3×1，commit 前须修 NB-1（§6.4 S2 必修）。**

| 项 | 结果 |
| --- | --- |
| 原 S1①（2.3 假命题） | **ADDRESSED**（假普适命题已删；命题 1 收窄 per-tensor 版 + 反例段 + 条件命题 2 数学正确；KIVI 引文一手逐字；回指全同步） |
| 原 S1②（2.3 FP4 κ=612B） | **ADDRESSED**（0.5625 B/元素 → κ=324B / 3.56× / 12.7KiB / 12.7GiB，doc + cuBLAS 一手证；全传播无 stale；StateBlock 528B MXFP8 未动） |
| 原 S2（2.4 双重 causal） | **ADDRESSED**（再取半已删；8.06e14 FLOPs / 1.63s；全 cascade 重算一致；练习锚同步） |
| 原 S2（P1 F1 H200 NVL TDP） | **ADDRESSED**（up to 600W configurable，官方 datasheet PDF 一手核；research notes L94 源头已纠） |
| 原 S2（2.2 命题编号 + 练习 3 锚） | **ADDRESSED**（章内重启裁定全 8 章落地；基数树模拟 hit+insert = R1/R2=4、R3=3、树 6 token） |
| 2.4 分层命中成本强制重推导 | **PASS**（独立重推，causal 系数与 1.2 式 (1) / models 2.4 式 (8) 对齐，无额外取半） |
| 新 breakage | **NB-1（S2）** 2.3 §2 L43 边界情形句；**NB-2（S3）** 2.3 §2 L49 「只剩符号」措辞 |
| lint / 篇幅 / 禁令词 / 习题 / 编号 | 8/8 LINT OK；重章 5825/5886/5756 ≥5500、标准章 ≥3500；rubric A 零命中；习题分布全合规；命题章内重启 8/8 |

## 1. S1/S2 逐条 verdict（file:line 证据 + 重推导输出）

### S1① 2.3 命题 1 假命题 — ADDRESSED

- **假普适命题已 gone**：2.3:43 命题 1 现为「per-channel 误差界占优 **per-tensor**（整组单 scale）」；2.3:41 引入第三粒度 per-tensor + 横截性声明（per-token 行切 / per-channel 列切互不细化）；2.3:45 明言「命题 1 不能推广到 per-token」+「『per-channel 更好』不是普适界定理」。全文 grep 无残留无条件「per-channel ≤ per-token」断言。
- **命题 1 重推（PASS）**：通道取值集合 ⊆ 整组取值集合 ⇒ R_c ≤ R_group ⇒ max_c Δ_c/2 ≤ Δ_group/2（Δ=R/(2^q−1) 单调）。等号条件「某通道同时取到全局两极」正确（R_c=R_group ⇔ 该通道含全局 max 与 min）。python 多组随机实例验证子集界全成立。
- **反例段重算（PASS）**：g=2、2 通道、q=2，token (0,1)/(2,3)：per-token 每 token 极差 1 → 界 1/6=0.1667；per-channel 每通道极差 2 → 界 1/3=0.3333 > 0.1667。与 evaluator 原反例逐字一致（2.3:45）。
- **命题 2（outlier 隔离）重推（PASS）**：前提 x_{t,c*} ≥ M（每 token）、正常 |x| ≤ m、M ≥ 3m。token 内 max ≥ M、min ≤ m（正常通道值 ≤ m，方向正确——writer 自查已修初稿「min ≥ −m」反向步）⇒ R_t ≥ M−m ⇒ per-token 界 ≥ (M−m)/(2(2^q−1))；per-channel 正常通道 R_c ≤ 2m ⇒ 界 ≤ m/(2^q−1)；M ≥ 3m ⇒ (M−m)/2 ≥ m ⇒ 链成立。M=3m 取等，「不小于」措辞正确。口径注（条件命题、非普适界）就位。
- **数值例重算（PASS）**：m=1、M=8、q=2：per-token 界 ≥ 7/6=1.1667（相对正常极差 2 为 58.3%≈58%）；最坏 token R=11、Δ=3.6667、格点 −1/2.667/6.333/10、界 1.8333（相对 91.7%≈92%）；per-channel 正常界 0.3333（16.7%）；outlier 通道 [8,10] 跨 token 极差 2。全部与 2.3:49 一致。
- **KIVI 引文一手核（PASS）**：自行 fetch arxiv.org/html/2402.02750v2，「The persistence of outliers within each channel means that per-channel quantization can confine the quantization error to each individual channel without impacting the other normal channels」逐字命中，位于 §3.2（§3 Methodology 子节，章引「§3」正确）；residual 窗口引句「the residual key cache and value cache are kept in full precision」亦逐字（2.3:51）。
- **回指同步（PASS）**：§3 item 1 → 命题 2（2.3:69）；召回命题 → 命题 3（2.3:112 + 练习 4(a):151 + 总结:143）；总结拆两行（:138 per-tensor 占优 + 反例 / :139 条件命题 + 数值）；练习 2 重写（:149，重证 per-tensor 版 + 等号条件 + 反例代入 + 指向命题 2，锚 1/3 > 1/6）。
- **但**：命题 1 证明后的「边界情形」句引入新错（NB-1，见 §2）——不影响本条 verdict（原假命题已除、新命题本体正确），但 commit 前须修。

### S1② 2.3 FP4 κ — ADDRESSED

- **一手证（自行 fetch）**：DeepInfer doc（contents API，main，372 行）L269-275 在「Where is FP4 used?」标题下逐字「Main KV: per-16, e4m3 — like NVFP4 but without the global scale.」「Indexer KV: per-32, ue8m0, MXFP4.」→ 主 KV 是 FP4 使用点，元素 = E2M1、e4m3 = scale 类型。cuBLAS 官方 doc（docs.nvidia.com/cuda/cublas/index.html，2.49MB 全页）「CUDA_R_4F_E2M1 The data type is a 4-bit real floating point in E2M1 format」+「CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 for all FP4-precision tensors」→ 元素 0.5B 确认。
- **重算（PASS）**：(16×0.5+1)/16 = 0.5625 B/元素；576/16=36 块 scale 36B + 数据 288B → κ=324 B/token/层；1152/324=3.5556≈3.56×；40 层 12960 B=12.65625 KiB≈12.7 KiB；满窗 1048576 → 12.65625 GiB≈12.7 GiB；bf16 45 GiB 不变。2.3:90 代入段 + 中间步（36 块/288+36）全对。
- **传播核（PASS）**：2.3 总结:142（324B/3.56×/12.7GiB）；坑注:92（误读虚高 1.89×，612/324=1.8889）；2.4:175（主 KV 低精度 12.7 KiB/token + 「10× 于其主 KV 低精度」128/12.65625=10.11）；2.4 总结:199（12.7 KiB/token）。stale grep：全 inference/ 无 1.88×/23.9/1.0625；612 仅现于坑注（故意保留作误读对照）。
- **StateBlock 未连带改（PASS）**：2.4:159 / 练习 1:204 仍 20×134×528=1,415,040 B、528=512+16 MXFP8；doc L221 一手「The 528 B/token is the MXFP8 SWA KV format」证实口径正确。

### S2 2.4 双重 causal — ADDRESSED（强制重推导 PASS，独立重推见 §3）

- recap 句 2.4:52 改「二次项系数 2 **已含 causal 减半**……不得再取半」+ 1.2 式 (1) / models 2.4 §6 指针 + 双向系数 4 对照；「上界口径取半」全 repo 零残留。
- 2.4:54：attention 2×32×32768²×4096 = 2.81e14（=2^48，不再取半）；合计 8.06e14；t_recompute≈1.63s；N=8e9 圆整数跨章口径注就位（S3-2 同修）。
- cascade 全链重算一致：15–24×（SSD 15.17 / 网络 18.96 / PCIe 24.27）、含 attention 临界 2.64 GB/s（2.4:66）、加权 198 ms / 8.2× / miss 163 ms / 82%（2.4:72）、逐层 51 ms（2.4:88）、总结:190-191、练习 2(a):205 改「已含 causal 口径，1.2 式 (1)」、练习 3(c):206 锚 ≈198 ms。练习 2(c)/4 锚（4.05/21.5/6.25）不受影响，核过不变。
- stale grep：1.34 s/1340/169/3.2 GB/42 ms/12–20/6.65/7.9× 零残留（1.41 仅 2.4:9 的 V4.1 bf16 容量 1.41 GiB，合法）。

### S2 P1 F1 H200 NVL TDP — ADDRESSED

- 1.3:31 现「H200 NVL 变体（BF16 1671 / FP8 3341 TFLOPS sparse、141 GB、4.8 TB/s、TDP up to 600 W configurable，H200 官方 datasheet NVL 列）勿与 H200 SXM（700 W）混」。
- 一手核（自行 fetch dam-cdn b2kt2580e5b6424if22lg65223m6ghjb.pdf，pdftotext -layout）：Technical Specifications 表 NVL 列「Max Thermal Design Power (TDP): Up to 600W (configurable)」；BF16 1,671 / FP8 3,341（sparse 脚注）/ 141GB / 4.8TB/s 全对；SXM 列 Up to 700W。
- research notes L94 源头已纠（「TDP up to 600 W configurable」+ 纠正注引 datasheet 原文）；全 repo「350-400」仅存于该纠正注（引旧误，合法）。变体不入算账，无数值级联。

### S2 2.2 命题编号 + 练习 3 锚 — ADDRESSED

- 编号：2.2:70 命题 1（基数树匹配复杂度）/ :103 命题 2（前缀命中节省）；总结:173/:175、练习 2:183 回指同步；全文无「命题 3/4」残留。
- **裁定全 8 章落地（grep 命题定义行）**：1.1={1}、1.2={1,2,3}、1.3={1}、1.4={1..6}、2.1={1,2}、2.2={1,2}、2.3={1,2,3}、2.4={1}——全部章内重启。1.2 三处跨章回指「1.1 命题 1」显式带章号（:3/:91/:144）；1.2 练习 3(a)→命题 1 闭环保持。
- 练习 3 锚（2.2:184）：「命中长度 + 新插入长度 = 该请求 token 数（R1/R2 = 4，R3 = 3）」。
- **基数树独立模拟（page=1 + split 语义，python）**：R1=[s,a,b,c] hit 0 + insert 4；R2=[s,a,b,d] 匹配 s,a,b 后分裂 → hit 3 + insert 1 = 4；R3=[s,a,e] 匹配 s,a 后分裂 → hit 2 + insert 1 = 3；树 token 总数 6（s,a,b,c,d,e 各一次）。锚正确，叶优先不变式锚（c）可校验。

## 2. 新 breakage（修复文本引入）

| ID | severity | 位置 | 断言 | 证据（独立重算） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| NB-1 | **S2** | 2.3 §2 L43 末句（命题 1 证明后「边界情形」） | 「边界情形：单个 outlier 通道含全局两极、其余通道极差窄，per-channel **严格占优**」 | 在命题 1 自己的**最大误差界**口径下为假，且与同段等号条件直接矛盾：outlier 通道含全局两极 ⇒ R_outlier=R_group ⇒ max_c Δ_c/2 = Δ_group/2，**等号成立**而非严格。数值反证：通道 1={0,10}（含全局两极）、通道 2={4,6}，q=2：per-tensor 界 10/6=1.667，per-channel 界 max(10/6, 2/6)=1.667，不严格。真正的严格情形是全局两极**分落不同通道**（如 ch1={0,4}、ch2={6,10}：max R_c=4 < R_group=10 → 0.667<1.667）。此句恰把 S1① 要澄清的「max bound vs 正常通道误差」区分再次混掉（outlier 含两极时 per-channel 的好处只在正常通道，属命题 2 论域） | 边界情形改「全局最大与全局最小分落不同通道、各通道极差严格小于整组极差时，per-channel 最大界严格更小」；outlier 含两极情形改列为**等号例**并注「此时 max bound 相等，per-channel 的收益在正常通道（命题 2）」。一句之修，修后 2.3 须 scoped 复核 |
| NB-2 | S3 | 2.3 §2 L49 数值例 | 「正常通道取值塌到 −1 / 2.67 两个格点，幅度信息尽失、**只剩符号**」 | 塌缩阈值是格点中点 (−1+2.667)/2 = 0.833 而非 0：正常值 (0, 0.833) 亦落 −1，符号并非严格保留。量化主张（格点/界 1.833/92%）全对，仅定性 gloss 不精确 | 改「幅度信息基本尽失（以格点中点 0.833 为界粗分两档）」或等价给阈值措辞 |

## 3. 强制重推导：2.4 分层命中成本账 — PASS（独立重推，不信任 writer）

结构：E[t_KV] = h1·0 + h2·t2 + h3·t3 + h0·t_recompute，t_i = κLT/BW_i + ℓ_i（大 KV 忽略 ℓ_i，章已声明）。

**causal 系数独立验证**：causal prefill 每层 attention = QK^T 因果对 T(T+1)/2 ≈ T²/2 × 每对 2·d_model（乘加计 2）+ PV 同量 ⇒ 每层 2·d_model·T²，全层 2LT²d_model——**系数 2 已含 causal**。精确求和交叉核：L·2·d_model·T(T+1)/2·2 与 2LT²d_model 相对差 <1e-4（T=32768）。与 1.2 式 (1)（1.2:28，§2:31「系数 2 来自 causal 平均可见长 T/2，双向回到 4」）及 models 2.4 式 (8)（本地 L148：2NT+2Ld·T(T+1)≈2NT+2LT²d；L153 teacher-forced 恒等；§6 L218 prefill 同式）逐项对齐。**无额外取半。**

数值独立重算（章脊 L=32/H_kv=8/d_head=128/d_model=4096/bf16，T=32768，F_eff=50%×989.5=494.75 TFLOPS，N=8e9 圆整数已声明）：

| 量 | 重算 | 章节值 | 判定 |
| --- | --- | --- | --- |
| κ / κL / KV | 4096 B / 131072 B=128KiB / 4.294967296 GB | 同 | ✓ |
| t2 / t3 / t400G | 67.1 / 107.4 / 85.9 ms | 67 / 107 / 86 ms | ✓ |
| 权重项 2NT | 5.24288e14 | 5.24e14 | ✓ |
| attention 2LT²d | 2.81475e14（=2^48） | 2.81e14 | ✓ |
| 合计 / t_recompute | 8.05763e14 / **1.6286 s** | 8.06e14 / 1.63 s | ✓ |
| 便宜倍数 | SSD 15.17× / 网络 18.96× / PCIe 24.27× | 15–24× | ✓ |
| 含 attention 临界 | 4.295/1.6286 = **2.637 GB/s** | 2.64 GB/s | ✓ |
| 加权 E | 0.2×67.1+0.2×107.4+0.1×1628.6 = **197.8 ms** | 198 ms | ✓ |
| 改善 / miss 贡献 / 占比 | 8.24× / 162.9 ms / 82.4% | 8.2× / 163 ms / 82% | ✓ |
| 权重项 BW*（命题 1） | 131072×4.9475e14/1.6e10 = **4.053 GB/s**，T 无关证明独立重推成立 | 4.05 GB/s | ✓ |
| 逐层 | KV 128MiB / PCIe 2.10 ms / 计算 50.9 ms | 128MiB / 2.1 / 51 ms | ✓ |
| 练习锚 | ex2(c) 4.05>2.64 方向 ✓；ex3(c) ≈198 ✓；ex4(a) 20–25ms / (b) 6.25>4.05 / (c) 21.5/4=5.375 ✓ | 同 | ✓ |

连带核（不受系数修正影响项）：BW_req 21.5 GB/s、压缩 5.0–6.1、跨 DC 6.25、写容量 3.8e4 token/s、16384 次小拷贝、FUSE 1.64 GB/s 差 24×、EC 1.2×/83.3%/2.5×、StateBlock 1,415,040 B、512GiB≈550GB=6.9×、S3-4 单盘限定（4×4.3=17.2<21.5，≥6.1 GB/s 级可行）——全部重算一致。

## 4. 轻量修复抽查（P1 S3 + 2.1/2.2 S3 + 跨章）

| 项 | 位置 | 结论 |
| --- | --- | --- |
| P1 F2 N 口径跨章注 | 1.2:19 + 1.3:11/:126 | ✓ 双向同注（matmul 7.50e9 vs storage 8.03e9、+7.07%/−6.68% 相消）；重算 1.2=66.54ms / 1.3=66.48ms 均 ≈66.5ms；W=1.606e10 两章同取 |
| P1 F3 W_soj 消歧 | 1.3:152/:160/:163/:166/:172/:210 | ✓ §5 全 W_soj + 消歧注；裸 W 仅权重字节语境（:13/:47/:104/:126/:196）；W_q 不动 |
| P1 F4 τ 互指 | 1.3:11 + 1.4:9 | ✓ 两章互指注齐全 |
| P1 F5 KV 写量级 | 1.2:91 | ✓ 「小约 T 倍——逾三个数量级」；log10(4096)=3.61>3 |
| P1 F6 Recipe 逐字 | 1.4:176 + research notes L111 | ✓ 自行 fetch doc L35 逐字比对「…it prevents special-token injection…」命中；notes 纠正注就位 |
| 2.1 引句小节号 | 2.1:56/:85/:89/:292 | ✓ 自行 fetch 2309.06180v1 HTML 按标题偏移定位：block-table 句 @94143 ∈ §4.2（91919-94742）、within-one-block @99134 / Storing-multiple @98212 ∈ §4.3（94742-100886）；PagedAttention 定义 §4.1 保留正确 |
| 2.1 拷贝 2× 流量注 | 2.1:37/:230 | ✓ 两处口径注（单向折算、读+写约 2×、比值不变），数值不改 |
| 2.2 N 切 matmul | 2.2:7/:20/:44/:113/:121/:146/:177/:182 | ✓ 11 值全重算：46.08/2.47/48.55/65.84 TFLOP、73.7%、49.1ms、0.48TFLOP/0.485ms、61.9%、1.94ms、14.8×；全量 65.8 与 1.2 实例同值；练习 1 锚 56.25<73.7<75 仍成立 |
| 2.3 S3-3 k/R 局部注 | 2.3:33/:80 | ✓ R 极差局部注（非 ridge）+ k OCP 原记号注（非 MoE top-k） |
| 2.4 S3-4 单盘限定 | 2.4:86 | ✓ ≥6.1 GB/s 级可行 / 4 GB/s 压到 4.3× 仍不达标，与练习 4(c) 张力消除 |
| 2.4 S3-5 HiSparse 路径 | 2.4:173 | ✓ 三路径全前缀 + kernels/ 非 srt/ 显式注；contents API 核 python/sglang/kernels/ops/kvcache/hisparse.py 存在（22579B）、srt/kernels 404 |
| 跨章符号 §4.3 | 全 8 章 | ✓ b=block size / bytes 文字或 \mathrm{bytes} / B=batch / d_model/d_latent/d_rope / κ / λ 到达率声明（1.1:40、1.3:11）/ k_top vs k（1.4:9）/ α 下标（1.4）/ goodput 𝒢（1.1/1.3）/ 2.3-2.4 局部符号均声明 |
| attention 系数 1.2↔2.2↔2.4 | 1.2:28/:31、2.2:20/:101、2.4:52 | ✓ 三章同 2LT²d_model（causal），2.2 每层 2·d_model·T_p² 钉死注与 1.2 式 (1) 显式对齐 |
| N 口径声明 | 1.2:19 / 1.3:11 / 2.2:20 / 2.4:54 | ✓ 四章各声明（7.50e9 matmul / 8.03e9 storage / 7.50e9 沿 1.2 / 8e9 圆整数 + 跨章注） |
| recap 指针 | 2.1:9 / 2.3:7 / 2.4:9 / 1.2:9 / 1.3:13 | ✓ 全指 models 2.4 式 (9)（非式 5）+ b→bytes 翻译注 |

## 5. 残留观察（非修复引入；首轮已存在或移交项，不阻塞本轮 verdict）

| ID | severity | 位置 | 内容 | 处置建议 |
| --- | --- | --- | --- | --- |
| R-1 | S3（可升 S2 通道） | 2.2:146 vs 2.4:19 | 2.2 式 (4) 例以「PCIe Gen5 128 GB/s」作 H2D 单向带宽（16MiB→131μs、裕量 14.8×）；2.4 §1 显式钉死 128 GB/s 为**双向合计**、单向取 64 GB/s。按 2.4 口径应为 262μs、裕量 7.4×——重叠结论不变，但同一 datasheet 数跨章口径不一。**首轮已存在**（P2 log 记 131μs/15.8×），非本轮引入 | 2.2 补一句双向/单向口径注或改 64 GB/s 重算（262μs/7.4×）；controller 如按跨章口径从严可升 S2 |
| R-2 | S3（移交 research pass） | research notes L117/L119 | vLLM 引句小节仍误标 §4.1（章节侧已修 §4.2/§4.3；错误源头在 research pass，P2 §8.6 已建议同步更正，writer C 明示不越权） | research/controller pass 更正，防 W2+ 继承 |
| R-3 | 观察 | 1.2:239（\gtrsim）、2.1×6 / 2.2×2（\gt） | \gt/\gtrsim 未显式列入 STYLE_GUIDE §3.1 白名单表（\lt 由 §3 硬规则强制但同样未列表）；MathJax 可渲染、lint 过、首轮 evaluator 放行 | 白名单补账（\lt/\gt/\gtrsim 条目）或统一裸 >；非阻塞 |
| R-4 | 观察 | 2.1:136 | 「更大显著变差」含 rubric B1 词「显著」，但系论文 §7.2/Fig.18(b) 实验结论转述（citational 语域，首轮放行） | 可选改「更大明显变差（论文口径）」；不强制 |
| R-5 | S3 可选 | 2.3:29 §2 标题 | 「per-channel 为什么压过 per-token」预设仅在 KIVI 分布下成立；正文已框为 KIVI 选择并明否普适性 | 可选收窄「per-channel 为什么在 K cache 上压过 per-token」 |

## 6. 机械核（修后终态）

- **lint**：8/8 LINT OK（scripts/lint-chapter.sh，含检查 7 反转义/`}_`/禁宏 + 检查 8 lint-math-boundary flanking）。
- **篇幅（CJK，剥代码/display/行内公式/code span）**：1.1=4025 / 1.2=5825【重】/ 1.3=4284 / 1.4=4303 / 2.1=5886【重】/ 2.2=4337 / 2.3=4361 / 2.4=5756【重】——重章 ≥5500、标准章 ≥3500 全达（raw 口径 4136/5948/4314/4335/6245/4417/4386/5830 亦达）。
- **禁令词**：rubric A 全表 8 章零命中；B1 仅 2.1:136「显著」（citational，R-4）；语气词 呢/吧/嘛 零。
- **习题分级**：8 章均 4 题、难度非递减、≥1 题 [10]/[20]、无 [40]、每章 ≥1 题以正文命题为证明目标（1.1 练2→命题1 / 1.2 练3→命题1 / 1.3 练2→命题1 / 1.4 练2→命题3 / 2.1 练3→命题2 / 2.2 练2→命题2 / 2.3 练2→命题1+练4→命题3 / 2.4 练3→命题1）；计算/代码题校验锚齐全。
- **命题编号**：8 章全部章内重启（§1 裁定核）；跨章回指带章号。
- **stale grep**：612（仅坑注）/1.88×/23.9/1.0625/1.34s/1340/169/3.2GB/42ms/12–20/6.65/7.9×/上界口径取半/350-400（仅纠正注）——零非法残留。
- **references 键**：本章所引 30 个 key 全部在 inference/references.md（被引章节字段回写仍留集成 pass，writers 禁动 refs 纪律遵守）。

## 7. Verification log（抽查项 × 来源 × 结论 × 覆盖率）

| 抽查项 | 来源（自行 fetch / 重算） | 原文引句 / 数值 | 结论 |
| --- | --- | --- | --- |
| DeepInfer「Where is FP4 used?」 | api.github.com deepseek-ai/DeepInfer/contents/doc/inference_deployment_qa.md?ref=main（Accept: raw，372 行） | L269-275「Main KV: per-16, e4m3 — like NVFP4 but without the global scale.」「Indexer KV: per-32, ue8m0, MXFP4.」 | 逐字 ✓（S1② 元素=E2M1 证） |
| DeepInfer Recipe L35 | 同上 | 「…efficiently and accurately; by segmenting on special tokens and encoding the segments in parallel, it prevents special-token injection while reducing tokenization latency for long inputs.」 | 逐字 ✓（F6） |
| DeepInfer EP-强度 L295 | 同上 | 「Increasing EP raises arithmetic intensity to avoid being memory-bound. Once memory bandwidth is no longer the bottleneck, a larger EP size introduces additional communication and synchronization overhead.」 | 逐字 ✓（1.2 L295 锚） |
| DeepInfer StateBlock / EC | 同上 L221 / L111-113 | 「state_size_to_disk = 20 layers × 134 tokens × 528 B = 1,415,040 B…The 528 B/token is the MXFP8 SWA KV format」；「10+2 erasure coding…all 11 storage nodes…any 10 responses」 | 逐字 ✓（528B 未动正确；2.4 §8 引文） |
| KIVI 分布性引句 + 节位 | arxiv.org/html/2402.02750v2（322KB） | 「The persistence of outliers within each channel…without impacting the other normal channels」@char 80873 ∈ §3.2（h3 @79140，§3 Methodology 下）；residual 句逐字 | 逐字 ✓、§3 归属 ✓（S1①） |
| cuBLAS FP4 类型 + scale | docs.nvidia.com/cuda/cublas/index.html（2.49MB） | 「CUDA_R_4F_E2M1 The data type is a 4-bit real floating point in E2M1 format」；「a dedicated scaling factor stored as an 8-bit CUDA_R_8F_UE4M3 value for each 16-element block in the innermost dimension…」；「…VEC16_UE4M3 for all FP4-precision tensors」 | 逐字 ✓（0.5B/元素证） |
| H200 datasheet NVL TDP | dam-cdn.nvd.orangelogic.com/AssetLink/b2kt2580e5b6424if22lg65223m6ghjb.pdf（pdftotext -layout） | Technical Specifications：NVL「Up to 600W (configurable)」、SXM「Up to 700W (configurable)」；NVL BF16 1,671 / FP8 3,341（sparse）/141GB/4.8TB/s | 全 ✓（F1） |
| vLLM 论文三引句节号 | arxiv.org/html/2309.06180v1（raw HTML 标题偏移） | §4.1@80812 / §4.2@91919 / §4.3@94742 / §4.4@100886；block-table@94143→§4.2、within-one-block@99134→§4.3、Storing-multiple@98212→§4.3 | 2.1 修后节号全 ✓（S3-1） |
| HiSparse 路径 | api.github.com sgl-project/sglang/contents（main） | python/sglang/kernels/ops/kvcache/hisparse.py 200（22579B）；python/sglang/srt/kernels 404 | ✓（S3-5） |
| models 2.4 causal 系数 | 本地 models/02-transformer-anatomy/04-causality-and-kv-cache.md L148/L153/L218 | 式 (8) 2NT+2Ld·T(T+1)≈2NT+2LT²d；teacher-forced 恒等句；§6 prefill 2NT_p+2LT_p²d | ✓（2.4 recap 锚） |
| 2.4 全链重推 | python3 独立重算（/tmp/iw1-fix-review-scripts/rederive.py） | §3 表全部 12 量 + 精确 causal 求和交叉核 + 练习锚 + 连带 10 项 | 全 PASS |
| 2.3 命题重推 | 同上 | 命题 1 子集界多实例；反例 1/6 vs 1/3；命题 2 链 (M−m)/2≥m；数值例 7/6、格点、91.7%、16.7%；FP4 κ 链 324/3.5556/12.65625；NB-1 数值反证 | 全 PASS（NB-1 除外，见 §2） |
| 2.2 重算 + 基数树模拟 | 同上 | 11 值逐值断言；R1/R2/R3 hit+insert=4/4/3、树 6 token、split 语义 | 全 PASS |
| 1.2/1.3 跨章一致 | 同上 | 66.54 vs 66.48 ms；+7.07%/−6.68%；log10(4096)=3.61 | PASS |
| lint / 篇幅 / 禁令词 / 习题 / 编号 / stale | scripts/lint-chapter.sh×8 + python 扫描 + /usr/bin/grep | §6 全项 | 8/8 过 |

**覆盖率**：5/5 原 S1/S2 逐条 verdict + 独立重推导；2.4 强制重推导全链 100% 重算；修后 touched 数值 100% 重算（2.4 cascade×12 + 2.3 命题/FP4×15 + 2.2×11 + 1.2/1.3 口径×4）；一手来源 6 类自行 fetch 逐字核（doc 5 段 / KIVI 2 句 / cuBLAS 3 句 / datasheet 全表 / 论文 3 引句节号 / repo 路径 2 条）。未重核项（首轮已验且本轮未触）：1.1 全文、1.4 未触数值例、2.1 代码锚点行号、2.4 §5-§8/§10 机制内容、Ascend/B200 datasheet 数、arXiv ID↔标题（lint 白名单过）。无「无 log 的确认无误」。

## 8. 处置建议

1. **NB-1（S2）必修**：2.3:43 边界情形句按 §2 建议一句之修；修后 2.3 scoped 复核（S1 章纪律）。NB-2/R-5 可顺手同修。
2. R-1（2.2 PCIe 口径）建议本轮顺手加口径注（foundational 跨章一致性收益高）；R-2 移交 research pass；R-3/R-4 白名单/措辞补账留 controller。
3. NB-1 修复 + 复核通过后，按 progress.md 管线续行：产图入 QA → refs 被引章节回写 → 断言 {8,2} → commit+push。
