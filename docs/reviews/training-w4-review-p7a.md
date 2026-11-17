# training W4 批 1（7.1–7.3）对抗审查报告

日期：2026-11-17。evaluator：独立 fresh-context（与写作 agent 无共享上下文）。对象：`training/07-pretraining/01-data-pipeline.md`（7.1）、`02-recipes.md`（7.2，推导重章）、`03-training-stability.md`（7.3）。方法：全部一手来源自行下载 ar5iv 全文转文本逐条核对（14 篇，清单见 verification log），关键数值本地复算，lint 实跑。

## 结论先行

- **S1 × 0，S2 × 1，S3 × 5（含 1 条 bookkeeping）。**
- **放行结论：不放行（待修 1 条 S2 后放行）**。S2 是 7.2 §2 一句话的归因错误，单行可修；其余内容全部通过一手核查。
- **GLM-4.5 冲突裁定：writer 对，spec 审查 C 的「全披露」错。** 7.2 配方表口径无需改动（详见 §4）。
- **Qwen3 引用已正式落地**（references.md `qwen3` 条目 + 7.2 两处正式引用，lint 检查 4 通过）。

## 1. Findings 表

| # | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | 7.2 §2（工作例段末） | 「Llama-3 405B 的 1.2M 步日程按末值 16M tokens/step 折算约 19.2T tokens 的吞吐上限——与披露的 15.6T 同量级，差额由 ramp 期的小 batch**与长上下文阶段**吸收」 | herd Table 4（arXiv:2407.21783）显示长上下文各阶段（8K、128K）batch 恒为 16M tokens/step，不吸收缺口；缺口 ≈3.6T 中 ramp 段（4M→8M→16M，节点 252M/2.87T）单独即可解释 ≈3.3T（327k 步按均值约 6M 计），余量可由 1.2M 步的约整口径吸收。「长上下文阶段吸收」与原文 Table 4 矛盾 | 删「与长上下文阶段」，改为「差额由 ramp 期的小 batch 吸收（ramp 段约 33 万步运行在 16M 以下）」；或加「长上下文阶段 batch 恒 16M（Table 4），不参与缺口」的注记 |
| F2 | S3 | 7.3 §3 推导一、7.3 ## 参考 moonlight 条 | 「RMS(u) 的实证区间为 0.2–0.4（Moonlight **脚注**口径）」 | 出处位置误标——Moonlight（2502.16982）的「AdamW's update RMS is usually around 0.2 to 0.4」在 §2.2 **正文段落**（式 (4) 前一句），不是脚注；论文脚注 2 的内容是「β₁<β₂ 与 ε>0 把实际 update RMS 压到 1 以下」。2.5 §4 的转引口径本身是对的（脚注只对应 <1 那句） | 「脚注口径」改「§2.2 正文口径」（两处） |
| F3 | S3 | 7.1 §2（DCLM 段） | fastText「负例 = 随机网页」 | DCLM 原文：「fix negative data as a random sample from a version of our RefinedWeb reproduction」——负例是 **RefinedWeb 复现版（已过基础过滤）的随机抽样**，不是原始网页；对数据章读者这是有效信息 | 改「负例 = RefinedWeb 复现的随机网页抽样」 |
| F4 | S3 | 7.3 §5.1 | 「一个 batch 约 10⁷ token」 | 表中各家末值 batch：Llama-3 16M ≈ 1.6×10⁷、DSv3 62.9M ≈ 6×10⁷、GLM-4.5 64M——旗舰端落在 10⁷–10⁸ 上沿 | 改「约 10⁷–10⁸ token」 |
| F5 | S3 | 7.1 §3 / §5 / §7 | 三处 rubric-B 量词「显著」：「显著低于 RefinedWeb 对照」「decay 段本来就有显著 loss 下降」「该规模下显著高于随机基线」 | 均跟随原文定性口径（FineWeb「far below RefinedWeb」为 Figure 3/4 读数无点值；MiniCPM「pronounced loss decrease」；FineWeb 基准判据原文为「above random baseline」——「显著高于」略升格）。不计 S2：源本身无数值可引 | 可选：第一、三处去「显著」或括注「原文为图形/定性口径」 |
| F6 | S3（bookkeeping，非章节内容） | training/references.md「被引章节」字段 | 回写缺失：glm-4.5 缺 7.1、flashattention 缺 7.1、deepseekv3 / kimi-k2 / moonlight 缺 7.3、minicpm 缺 7.2 | 两位 writer 报告均已自报该缺口；lint 只核 arXiv ID 不核被引字段，故不影响 LINT | 归 controller/research pass 补登，不在本批章节修复范围 |

无 S1。7.2 §4 存疑项 6（SGD 隐藏层 lr=1 的「梯度坐标 Θ(1/n) 经输出层尺度回推」）已复核：与 TP-V Table 3 的 SGD hidden lr = 1 一致，回推链（输出层坐标 1/n ⇒ 隐层 preactivation 梯度 δ 坐标 Θ(1/n) ⇒ g = δxᵀ 坐标 Θ(1/n)）成立，驳回存疑、维持原文。

## 2. 强制独立重推导（逐步中间过程）

### 2.1 7.2 µP 缩放结论表（对 TP-V 2203.03466 Table 3 逐格）

ar5iv 全文 Table 3 原文转录（input weights & all biases / output weights / hidden weights 三列）：

- Init. Var.：`1/fan_in` ｜ `1/fan_in²`（括注 SP `1/fan_in`）｜ `1/fan_in`（括注同 SP）
- SGD LR：`fan_out`（括注 SP `1`）｜ `1/fan_in`（括注 SP `1`）｜ `1`
- Adam LR：`1` ｜ `1/fan_in`（括注 SP `1`）｜ `1/fan_in`（括注 SP `1`）

7.2 §4 表逐格比对：**9 格全部一致**（含 SP 对照括注）。表注「Transformer µP requires one more modification (1/d attention instead of 1/√d)」与 7.2 §4 的 Definition 4.1 引用一致；Definition 4.1 原文（q、k 训练中相关 ⇒ qᵀk 按大数律 ∝ d；初始化处不相关 ⇒ CLT ∝ √d）与章节机制读法逐句对应。Desiderata J.1 三条原文核对：(1) 任意时刻（预）激活坐标 Θ(1)；(2) 输出 O(1)；(3) 所有参数在宽度极限下尽可能多地更新而不发散——与章节转述一致（章节对 (ii) 补的「初始化处允许趋于零」与 abc 体系兼容，练习 1 提示把该点做成推导义务，处置得当）。Lemma J.1 原文：SGD `(Aθ, B/θ, C/θ²)`、Adam `(Aθ, B/θ, C/θ)` 保持 f_t(ξ) 逐点不变——与章节一致。坐标尺度推导 outline 复算：输出层 W∈ℝ^{1×n}、x 坐标 Θ(1)，训练后相关成分需 n·(1/n)=Θ(1) 故坐标取 Θ(1/n)（init 方差 1/n² 时 init 输出 O(1/√n)→0，desideratum (ii) 允许）；隐藏层 Var((Wx)_i)=nσ²·Θ(1) ⇒ σ²=Θ(1/n)；Adam 逐坐标更新幅度 = lr ⇒ lr=Θ(1/n)；SGD 隐层梯度坐标 Θ(1/n) ⇒ lr Θ(1) 即足。**结论：结论表转录与 outline 均无误。**

### 2.2 7.1 MinHash 相似度估计（式 (1)–(4) 与数值验算）

- 式 (2)：S∪T 在随机置换下最小元均匀 ∈ 并集；h(S)=h(T) ⟺ 最小元 ∈ S∩T ⇒ Pr = |S∩T|/|S∪T| = J。**成立**（经典结论，一行证明完整）。
- 式 (3)：m 个独立伯努利(J) 的均值 ⇒ E[Ĵ]=J、Var=J(1−J)/m、std=√(J(1−J)/m)。代入 m=112、J=0.75：√(0.1875/112)=√0.001674=0.0409 ≈ 0.041 ✓。
- 式 (4)：b 桶各 r 位，单桶全同概率 s^r，候选 = 至少一桶全同 = 1−(1−s^r)^b ✓。代入 b=14、r=8：
  - s=0.75：0.75⁸=0.10011；(1−0.10011)¹⁴=0.89989¹⁴，ln=−1.4756 ⇒ 0.2288；P=**0.7712** ≈ 0.771 ✓（FineWeb §3.4 自报同点 77% ✓，双源一致）
  - s=0.5：0.5⁸=0.003906；0.99609¹⁴=0.9468；P=**0.0532** ≈ 0.053 ✓
  - s=0.9：0.9⁸=0.43047；0.56953¹⁴≈3.8×10⁻⁴；P=**0.99962** ≈ 0.9995 ✓
  - 拐点近似 s*=(1/b)^{1/r}=14^{−1/8}=e^{−0.330}=0.720，与目标阈值 0.75 邻近 ✓
- 练习 1(c) 自洽性：1.96·√(0.1875/m) ≤ 0.02 ⇒ m ≥ 1801，题目自洽。

## 3. Verification log（抽查项 × 来源 × 结论）

来源获取方式：14 篇 arXiv 全部 curl ar5iv 全文转纯文本本地 grep/摘录（2026-11-17）；OLMo-core 配置与 flash-attention 接口经 GitHub API。

### 7.1（FineWeb / DCLM / DoReMi / Llama-3 数据侧 / GLM-4.5 数据侧 / DSv3 数据侧 / MiniCPM / GPT-3 App / FA repo）

| 抽查项 | 来源定位 | 结论 |
| --- | --- | --- |
| 96 快照、WARC+trafilatura 弃 WET、基础过滤后 ≈36T | FineWeb §3.2/§3.3（「roughly 36 trillion tokens」） | ✓ |
| 逐快照 MinHash → 20T；最终 15T；五段（§3.2→3.3→3.4→3.5→3.6） | FineWeb §3.4「individually deduplicating… resulted in 20 trillion」、§3.7 五段枚举 | ✓ |
| MinHash 参数 5-gram、112 哈希、14 桶 ×8、≥75%；传递聚类随机留一 | FineWeb §3.4 + App. E.1 原文 | ✓ |
| 三过滤器 0.12/10.14%（C4 原版 30%）、0.1/12.47%（MassiveText 0.2）、0.67/3.73%；并集 ~22%、+1pp、28B 消融 | FineWeb §3.6 原文逐数 | ✓ |
| 全局去重：最老快照删至 90%、剩 4T、350B 消融几无提升且远逊 RefinedWeb；2013-48：留下 ~31B vs 删掉的 ~460B 单独去重得 171B、留下的更差（广告/关键词堆砌） | FineWeb §3.4 原文逐数（Fig. 3/4） | ✓ |
| 消融协议：1.71B（含 embedding）、Llama 架构、seq 2048、batch ~2M tok、GPT-2 tokenizer、过滤 28B（≈Chinchilla 最优）、去重/累计 350B、双种子取均值、八基准截断 1000、三判据、70+ 模型、8 万 H100 时 | FineWeb §3.1 原文逐项 | ✓ |
| DCLM Table 4（1B-1x）：fastText OH-2.5+ELI5 30.2 居首；top-k 29.2、perplexity 29.0、AskLLM 28.6、BGE 27.2、SemDedup 27.1、PageRank top20% 26.1；训练集 ~400k 两类各半；负例=RefinedWeb 复现随机抽样（见 F3） | DCLM Table 4 + §3.3 原文 | ✓（除 F3 措辞） |
| Table 5（7B-1x）：top-10% 优于 15%/20%；正例集消融 34.7–37.5（OWT2 34.7 / Wikipedia 35.7 / GPT-3 Approx 37.5） | DCLM Table 5 原文 | ✓ |
| perplexity 过滤器：154M causal Transformer、英文 Wikipedia + RedPajama-v1 books + peS2o | DCLM §3.3 原文 | ✓ |
| Bloom filter vs MinHash：7B-2x 差 <0.2 Core pp；>10TB 更好扩展；baseline 用 Bloom、其余用 MinHash；§3.3 + App. L | DCLM §3.3 原文 | ✓ |
| Mixing（Table 6）：67% CC / 33% 高质量源；弱过滤子集受益、DCLM-baseline 平均掉分（31.1→29.9） | DCLM §3.3 + Table 6 原文 | ✓ |
| 240T pool；阶梯 412M→7B（7B-1x=138B、7B-2x=276B）；416 组基线；7B/2.6T→64% MMLU 5-shot | DCLM §1/§3.1/Table 1 原文 | ✓ |
| DoReMi：280M 参考+代理 → 8B（30×）；+6.5pp；2.6×（跨规模）/4×（同规模）；Algorithm 1（均匀域采 batch、max{·,0} 截断、指数更新、c=1e-3 平滑、轨迹平均 = Nemirovski 口径）；式 (5)(6) 逐式对回 | DoReMi 摘要/§2/§3/Algorithm 1 原文 | ✓ |
| Llama-3 §3.1.1：n-gram 覆盖率、脏词、token-KL 散度三启发式；fastText「是否被 Wikipedia 引用」+ DistilRoberta 学 Llama 2 判定；三级去重（URL 留最新 / 全局 MinHash / ccNet 式 30M 桶 >6 次删行、人工抽查净收益为正） | herd §3.1.1 原文 | ✓ |
| Llama-3 §3.1.2：知识分类器降采样过度代表类目（娱乐）、配比 scaling-law 筛选流程；50/25/17/8 | herd §3.1.2 原文 | ✓ |
| Llama-3 §3.1.3：8B annealing GSM8k +24.0%、MATH +6.4%、405B 几乎无提升；50% 训 8B + lr 线性归零 + 40B tokens + 30% 权重测量法 | herd §3.1.3 原文 | ✓ |
| GLM-4.5：SemDedup 补 MinHash（「Such webpages cannot be removed by MinHash deduplication」）；两阶段预训练上采样；mid-training 三段；best-fit packing 取舍原文 | GLM-4.5 §2.2/§2.3 原文 | ✓ |
| DSv3：BBPE 128K、预分词器为多语压缩修改；文档 packing 不加跨样本 mask；FIM 0.1；数学/编程占比提升、多语扩展；Ding et al. 转引成立 | DSv3 §4.1 原文 | ✓ |
| MiniCPM §5：decay 段混高质量+SFT 数据；A-1/A-2、B-1/2/3 消融；「decay 段 loss 骤降」机制解释 | MiniCPM §5 + Table 1 原文 | ✓ |
| GPT-3 App. B：多文档 pack 进 2048、仅 eot 分隔、不做 mask；App. A Spark MinHashLSH 10 哈希模糊去重 | GPT-3 App. A/B 原文 | ✓ |
| FlashAttention varlen：`flash_attn_varlen_func` + `cu_seqlens_q/k` 存在 | GitHub API：Dao-AILab/flash-attention `flash_attn/flash_attn_interface.py` | ✓ |

### 7.2（七家配方表逐字段 + ramp + µP/µTransfer + Moonlight + tokens 预算 + 初始化）

| 抽查项 | 来源定位 | 结论 |
| --- | --- | --- |
| GPT-3：Table 2.1 八档 lr 单调（125M 0.5M/6.0e-4 → 175B 3.2M/0.6e-4）；App. B：Adam β0.9/0.95、ε1e-8、clip 1.0、cosine 260B 至 10% 后恒定、warmup 375M tokens、ramp 32k→全值前 4–12B、wd 0.1、300B tokens | GPT-3 Table 2.1/App. B 原文逐字段 | ✓ |
| Llama-2：AdamW β0.9/0.95、ε1e-5、warmup 2000、cosine 至 10%、wd 0.1、clip 1.0（§2.2）；4M tokens（Table 1 表注）；7B/13B=3e-4、34B/70B=1.5e-4；2.0T | Llama-2 §2.1/§2.2/Table 1 原文 | ✓ |
| Llama-3 405B：AdamW、8e-5、warmup 8000、cosine 至 8e-7/1.2M 步、4M→8M→16M（252M/2.87T 节点，「252M」照录存疑与 2.3 §5 一致）；wd=0.1×lr 为 §3.2.1 口径；β/ε/clip 未披露；15.6T（§1） | herd §3.4.1/§3.2.1/§1/Table 3 原文 | ✓ |
| DSv3：AdamW β0.9/0.95、wd 0.1、warmup 2000 步、2.2e-4、恒定 10T→cosine 4.3T→2.2e-5、末 500B 两段（2.2e-5/333B + 7.3e-6/167B）、clip 1.0、batch 3072→15360 序列（×4096 = 12.6M→62.9M）前 469B、init std 0.006、ε 未披露、14.8T | DSv3 §4.2 原文逐字段 | ✓ |
| OLMo 2 7B：batch 1024×4096、clip 1.0、3e-4、warmup 2000、cosine 5T 标定 4T 截断；13B：2048、9e-4、不截断；β0.9/0.95 与 wd 0.1（embedding 免）= OLMo-core `OLMo2-7B.py` 发布配置（GitHub API 实拉：lr 3e-4、wd 0.1、betas (0.9,0.95)、embedding wd=0 override）；ε 1e-8（§3.4.1）；init 0.02 截断正态（§2.3/§3.2）；mid-training 3×50B 线性归零 + 平均（§2.3/§4） | OLMo 2 Table 4/§2.3/§3.2/§3.4/§4 原文 + OLMo-core repo | ✓ |
| GLM-4.5：Muon（NS 5 步、μ=0.95、RMS 0.2，embedding/bias/RMSNorm 增益除外且其优化器未点名）、cosine 非 WSD（early experiments 欠拟合原句）、warmup 0→2.5e-4、衰减至 2.5e-5 至 mid-training 末、batch 16M→64M 前 500B、wd 0.1、无 dropout；warmup 步数与 clip 未披露；23T（摘要，含 mid-training）；355B/32B | GLM-4.5 §2.4/摘要/§2.1 Table 1 原文逐字段 | ✓（详见 §4 裁定） |
| Qwen3：36T、119 语言；「we develop scaling laws for optimal hyper-parameters…we set the predicted optimal learning rate and batch size strategy for each dense or MoE model」引句逐字一致；数值未披露 | Qwen3 TR §3.2/§1 原文 | ✓ |
| ramp 式 (1) 积分与数值：DSv3 469e9/50.3e6×ln5 = 9320×1.609 ≈ 1.50 万步（恒定口径 7456，2.0×）；GLM-4.5 500e9/48e6×ln4 = 10417×1.386 ≈ 1.44 万步（恒定 7813，1.84×）；ramp 占比 3.2% / 2.2% | 本书计算复算 | ✓ |
| 式 (2) ΔL_opt/B 单调减；23.5 万步（14.8T/62.9M=235,294）、19.2T（1.2M×16M）交叉验算；warmup 占比 0.85%≈0.9%、0.67%≈0.7% | 本书计算复算 | ✓ |
| µP Table 3 九格、Desiderata J.1、Definition 4.1、Lemma J.1、wider-is-better（Fig. 1/3/7 口径） | TP-V 原文（见 §2.1 重推导） | ✓ |
| µTransfer：40M/width 256 代理 → 6.7B/width 4096（168×）；调参成本 7%；迁移 6.7B 超原配方重训基线、评测达 13B 水平（Table 7/11）；迁移边界：wd/dropout 不迁移（Table 2）、最小门槛宽 256/深 4/batch 32/seq 128/步 5000、init std 不随深度迁移（固定 init 调其余）、深度迁移仅 Pre-LN | TP-V §6/§7.4/Table 2 原文 | ✓ |
| Moonlight：wd 动机（权重与层输出 RMS 涨出 bf16 高精度区间）、800M/100B（~5× compute-optimal）对照、附录 A 五点扫描 {0.05,0.1,0.2,0.4,0.8} 中 0.2/0.4 最优且接近、AdamW RMS 0.2–0.4 | Moonlight §2.2/§3.2/附录 A 原文 | ✓（脚注误植见 F2） |
| tokens 预算：DSv3 14.8T/671B≈22、405B 38.5、8B 1950 tok/param；herd §1 引言过训练原句逐字；MiniCPM「much higher data size/model size ratio than Chinchilla」 | 复算 + herd §1/MiniCPM §1 原文 | ✓ |
| 初始化：OLMo 2 从 Zhang et al. 2019 深度缩放（OLMo-0424：输出投影 1/√(2·d·layer_idx)）换 0.02 平坦；DSv3 0.006；√(4096/7168)×0.02≈0.015 对照账 | OLMo 2 §3.2/DSv3 §4.2 原文 + 复算 | ✓ |
| MiniCPM 风洞族 0.009B–0.17B | MiniCPM §3.2 原文 | ✓ |

### 7.3（GLM-130B / OLMo 2 / DSv3 / herd Table 5 / K2 / Moonlight + 两条推导）

| 抽查项 | 来源定位 | 结论 |
| --- | --- | --- |
| 「a training collapse usually lags behind a "spike" in gradient norm by a few training steps」引句；spike 两形态；embedding 梯度大几个数量级；Apex O2；DeepNorm 型 Post-LN；EGS α=0.1「wipes out most spikes」；3 次晚期发散 vs 硬件失败更多；PB-Relax 不能避免发散；BF16 +~15% 显存/不支持 V100；embedding norm 损 zero-shot；OPT-175B 手动跳数据+调超参 | GLM-130B §3 原文逐项 | ✓ |
| OLMo 2 spike score 定义（滚动 1000 窗、7σ、百分比、用于 loss 与 grad L2 norm）；八项干预与节号；0.40→0.03（§3.2 init）；0.108→0.069（§3.3.2）；z-loss 1e-4·(log Z)²（§3.3.3）；ε 1e-5→1e-8「permanently lower」（§3.4.1）；embedding 免 wd 0.16 vs 0.092（§3.4.2）；§3.5 LR 专项；n-gram 32+/1–13 token、trainer mask、三条非决定论反例 | OLMo 2 §3 全节原文逐数 | ✓ |
| DSv3「did not experience any irrecoverable loss spikes or perform any rollbacks」（摘要+§1）、14.8T、2.788M H800 时 | DSv3 摘要/§1 原文 | ✓ |
| herd §3.3.4+Table 5：54 天 466=47+419、~78% 硬件、正文 GPU 58.7%、Table 5 九行逐数（148/30.1%、72/17.2%、54/12.9%、35/8.4%、32/7.6%、19/4.5%、17/4.1%、SDC 6/1.4%、Thermal 6/1.4%）、3 次人工干预、>90% 有效训练时间、单 GPU 故障可致全作业重启；表文口径差（268/419≈64%）注记 | herd §3.3.4/Table 5 原文 | ✓ |
| K2 §2.1：MuonClip=Muon+wd+RMS 匹配+QK-Clip（Algorithm 1）、τ=100、~30% 步后回落、τ 全程不调、15.5T 零 loss spike；中规模裸 Muon logit 破 1000；WSD（10T 恒定 2e-4/500 步 warmup + 5.5T cosine 2e-4→2e-5） | K2 §2.1/§2.5/Fig. 2-3 原文 | ✓ |
| Moonlight 附录 D：max attention logit 特定层破 100 而 loss/grad norm 平稳、AdamW 更健康 | Moonlight App. D 原文 | ✓ |
| 检测判据式 (1) 与 k=7 误报账：Φ̄(7)=1.28e-12 ⇒ 10⁴ 步期望误报 ≈1.3e-8 | 复算 | ✓ |
| 推导一（update ratio 式 (2)(3)）：主项比值 0.2/0.02=10 – 0.4/0.02=20；r≈8e-5×(10–20)=8e-4–1.6e-3；wd 份额 λ²/15²=4.4e-5 | 复算 | ✓ |
| 推导二（param norm 准稳态式 (4)）：RMS(θ*)=√(8e-5/0.2)×0.3=0.02×0.3=6e-3；假设与诚实边界在文 | 复算 | ✓ |
| 宿醉账（1−β₂)k²v₀=500 倍、20 步/10 步时间常数 | 复算 + 2.4 §4 回链存在性 | ✓ |
| §5.2 代价账：1000 步 ×6s≈1.7h；54 天窗总步数 ≈7.8×10⁵ ⇒ 千分之一量级 | 复算 | ✓（batch 量级见 F4） |

**覆盖率**：三章全部带数值/出处的断言 100% 过一手来源或复算（抽查率：配方表 11 字段 × 7 家 = 77 格全字段逐格；7.1 管线数值全项；7.3 Table 5 摘录九行全数；推导式全部重算）。跨章回链存在性抽查 2.2 §7 / 2.3 §3/§5 / 2.4 §4/§5 / 2.5 §4/§7 / 3.1 §5 全部命中。

## 4. GLM-4.5 冲突裁定（writer vs spec 审查 C）

**裁定：writer 对；spec 审查 C 的「GLM-4.5 全披露」为错误概括；7.2 配方表口径无需改动。**

ar5iv 2508.06471 §2.4 全文逐字段核对结果：

| 字段 | 披露状态 | 原文 |
| --- | --- | --- |
| optimizer | 披露（Muon）但排除项的优化器**未点名** | 「We employed the Muon optimizer for all parameters except word embedding, bias, and weights for RMSNorm.」 |
| Muon 超参 | 披露 | NS steps N=5、momentum μ=0.95、update RMS 0.2 |
| schedule | 披露形态（cosine、明确不用 WSD 及理由） | 「We used cosine decay schedule… instead of WSD… Our early experiments showed that models trained with the WSD schedule perform worse…」 |
| warmup | **只披露区间 0→2.5e-4，步数/tokens 未披露** | 「The learning rate went through a warm-up stage from 0 to 2.5e-4」 |
| decay 终点 | 披露 | 「a decaying stage to 2.5e-5 until the end of mid-training」 |
| batch ramp | 披露 | 「16M tokens to 64M tokens in the training of the first 500B tokens」 |
| wd / dropout | 披露 | 「weight decay ratio to 0.1 and did not use dropout」 |
| 梯度裁剪 | **未披露**（全文无 clip 字样） | — |
| seq 长度 | 披露 | 4K（预训练）→32K→128K（mid-training） |
| 23T tokens | 披露（摘要，multi-stage 口径含 mid-training） | 「Through multi-stage training on 23T tokens」 |

spec 审查 C 括号里列的字段（Muon、0→2.5e-4→2.5e-5 cosine、16M→64M/500B、wd 0.1、seq 4K→32K→128K）全部为真，但「全披露」的定性越过了证据：warmup 长度与 clip 两个表内字段无原文支撑，非矩阵参数的优化器未点名。7.2 表把 warmup 标「有 warmup 段，步数未披露」、clip 标「未披露」、β/ε 标「—（Muon）」并加注意外排除项——与原文严格一致，**处置正确，无需修改**。建议下一轮 spec 修订把 §5 7.2 内容行的「GLM-4.5 全披露」改为「GLM-4.5 高披露（warmup 步数/clip/非矩阵参数优化器未披露）」（spec 文本，不在本批章节修复范围）。

## 5. Qwen3 引用落地核查

- references.md：`qwen3 | Qwen3 Technical Report | Qwen Team | arXiv:2505.09388 | [A] | 2026-11-17 | 7.2` ✓ 存在。
- 7.2 §1 逐家出处 bullet：`[Qwen Team, 2025](../references.md)，arXiv:2505.09388，[A]` + 披露口径引句 ✓；## 参考 节同格式含 key ✓。
- lint 检查 4（arXiv ID 回查 references.md）通过 ✓。**落地确认。**

## 6. 机械检查

```
bash scripts/lint-chapter.sh training/07-pretraining/01-data-pipeline.md → LINT OK (exit 0)
bash scripts/lint-chapter.sh training/07-pretraining/02-recipes.md      → LINT OK (exit 0)
bash scripts/lint-chapter.sh training/07-pretraining/03-training-stability.md → LINT OK (exit 0)
```

rubric A 类禁令词（含 lint 词表外的补充词表）三章 grep 零命中；无历史叙事/人物故事；每节以事实/公式/问题开场；模板小节、metadata、ASCII 图（7.1 五段管线图、7.2 decay 形态图、7.3 检测流水线图）齐全。

## 7. 教学审查摘要

- 推导链完整：MinHash（Jaccard→碰撞恒等式含证明→签名方差→banding）无断步；DoReMi 式 (5)(6) 与 Algorithm 1 逐行对应且符号全定义；ramp 积分有中间步；7.3 两条推导假设显式、诚实边界在文、练习 1 主动探假设放松。
- µP 推导 outline 的严格性边界（附录 J Table 14 随机/相关矩阵分类）已显式声明不替代原文——处置得当，writer 存疑项 6 经复核驳回（见 §1 末注）。
- 前置依赖全部存在且内容匹配（2.2/2.3/2.4/2.5/3.1 各回链抽查命中）。
- 未定义符号零发现（每个公式后跟符号表）。

## 8. 放行条件

1. **（必须，S2）** 修 F1：7.2 §2 删/改「与长上下文阶段吸收」半句。
2. （S3，累计下波清零）F2–F6。

修完 F1 后本批即可放行；F1 为单行修改，修复后可由 controller 直接复核该句（herd Table 4 即证据），不必重跑全量审查。
