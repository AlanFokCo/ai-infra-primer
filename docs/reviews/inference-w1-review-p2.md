# W1 对抗审查报告 · P2（KV cache 系统：2.1 / 2.2 / 2.3 / 2.4）

- 日期：2026-11-23
- 审查者：fresh-context evaluator（P2 batch，不继承写作上下文）
- 输入：四章成品 + inference spec（§4/§5 P2/§6.3）+ STYLE_GUIDE + R2 style sheet + rubric + inference/references.md + docs/research/inference-w1-sources.md + 1.2（跨章对照）+ models 2.4 / 4.1（recap 忠实性）
- 一手来源：evaluator 自行取得——arXiv HTML（2309.06180v1 / 2312.07104v2 / 2402.02750v2 / 2310.07240v6）+ arXiv API 10 摘要批量；codeload tarball（vllm main / sglang main / DeepInfer main / 3FS main / LMCache dev / Mooncake main，均 2026-11-23 取）；DeepInfer doc 全文；cuBLAS / TensorRT-LLM 官方文档；OCP MX v1.0 PDF（pdftotext）；hf-mirror V4.1-Flash config；GitHub API（YaoJiayi/CacheBlend、WukLab/preble）
- 只读审查：未改任何章节文件、未 commit、未派 subagent

## 结论

**Status：NOT READY（含 S1，2.3 / 2.4 修复后须复审）**

| severity | 计数 | 分布 |
| --- | --- | --- |
| S1 | 2 | 2.3 ×2（其一传播至 2.4 §9 / 总结） |
| S2 | 3 | 2.4 ×1（attention 系数）、2.2 ×2（命题编号约定、练习 3 校验锚） |
| S3 | 6 | 2.1 ×2、2.2 ×1（与 2.4 共因）、2.3 ×1、2.4 ×2 |

- 强制重推导：2.1 三项全 PASS；2.4 传输带宽需求 PASS、专用集群 vs 分层权衡 PASS、**分层命中成本账 FAIL**（attention causal 系数误读，S2-1）。
- 跨章系数必查项：**2.2 每层 2·d_model·T² 与 1.2 式 (1) 2LT²d_model 对齐（PASS）**；但 2.4 把已含 causal 的 2LT²d 再取半，与 1.2 / 2.2 冲突（S2-1）。
- KV 容量 recap 指针：四章全部指向 models 2.4 式 (9)（非式 5）✓。
- W-C / W-D writer 待核项：代码锚点与口径分离全部核实成立（见 §5）；但 2.3 主 KV 元素字节误读独立成 S1（S1-2）。
- lint 四章全绿；篇幅四章达标（2.1 = 5848 / 2.4 = 5676 CJK ≥5500 软目标；2.2 = 4315 / 2.3 = 3797 ≥3500）；禁令词零命中；R2-1…R2-12 逐条过（详见 §6）。

## 1. Findings 表

| ID | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| P2-S1-1 | S1 | 2.3 §2 L43（命题 1 + 证明）；传播：§3 L65、总结 L132、练习 2 L142 | 「per-channel 量化的最大误差界不超过 per-token 的最大误差界」作为无条件命题**为假**，证明的子集论证无效 | 按本章自己的粒度定义（per-token = 一个 token 向量的 d_head 通道共享 scale；per-channel = 一个通道跨 g 个 token 共享 scale），两种划分是**横截的**，互不细化。反例：g=2、d=2、q=2，token1=(0,1)、token2=(2,3)：per-token 每 token 极差 1 → 最大误差界 1/6≈0.167；per-channel 每通道极差 2 → 最大误差界 1/3≈0.333 > 0.167。证明步「通道的取值集合是全向量取值集合的子集」只对**整组**（per-tensor）极差成立，对单个 token 的极差不成立。练习 2 的「全通道极差相同时两粒度误差界一致」同被此反例族否证（R_c1=R_c2=2 而界不等）。KIVI 原文（2402.02750v2 §3，evaluator 一手）只给分布性论断：「The persistence of outliers within each channel means that per-channel quantization can confine the quantization error to each individual channel without impacting the other normal channels」——无普适 max-bound 占优声明 | 重述命题：(a) 无条件版改为「per-channel 误差界 ≤ per-tensor（整组单 scale）误差界」（子集证明平凡成立）；(b) K cache 的持久 outlier 结构前提下（每 token 的全局两极同落一个 outlier 通道），per-token scale 被 outlier 绑架、per-channel 把正常通道误差限回自身极差——§2 的 16.7% vs 167% 数值例已是正确机制，命题应比较**正常通道误差**而非组 max bound（outlier 通道跨 token 极差可 ≥ 其单 token 极差，max bound 口径下 per-channel 未必占优）。同步改练习 2 的等号条件与反例、总结行、§3 item 1 引用 |
| P2-S1-2 | S1 | 2.3 §4 L86（主 KV 字节账）；传播：2.3 总结 L135、2.4 §9 L175、2.4 总结 L199 | DeepInfer 主 KV κ = 612 B/token/层（1.88×、23.9 KiB/token、满窗 23.9 GiB）**错误**：把 FP4 元素当 1 字节代入 (16×1+1)/16 | doc L269-271 在「**Where is FP4 used?**」标题下逐字：「Main KV: per-16, e4m3 — like NVFP4 but without the global scale.」→ 元素 = FP4 E2M1（0.5 B，cuBLAS Table 4 CUDA_R_4F_E2M1 一手核实），scale = per-16 e4m3（1 B）。正确：(16×0.5+1)/16 = 0.5625 B/元素，κ = 576×0.5625 = **324 B/token/层**，对 bf16 1152/324 = **3.56×**，40 层 12960 B/token ≈ **12.7 KiB**，满窗 1048576 ≈ **12.7 GiB**。章节自己的行文「主 KV 与 NVFP4 的差别正是 global scale 一项：per-16 的 e4m3 scale 照用」与代入矛盾（NVFP4 元素 0.5 B 被换成 1 B）。注意区分：SWA StateBlock 的 528 B 是 **MXFP8**（doc L223 明示），2.4 §8 该账正确、勿连带改 | 2.3 §4 代入改 (16×0.5+1)/16，κ=324 B、3.56×、12.7 KiB/token、12.7 GiB 满窗；总结行同步；2.4 §9「主 KV 低精度 23.9 KiB/token」→ 12.7 KiB/token、总结行同步。576/16=36 块 scale 字节的中间步保留 |
| P2-S2-1 | S2 | 2.4 §2 L52-54（recap 句 + 重算账）、L66（3.2 GB/s）、L72（169 ms / 7.9×）、§3 L88（42 ms）、总结 L190-191、练习 2(a) L205 / 练习 3(c) L206 | 「models 2.4 §6 的 2LT²d_model 上界口径，causal 实际约其半」**误读来源**：models 2.4 §6 的 2LT²d 已是 causal 值，再取半把 attention 项低估 2×，与 1.2 式 (1) / 2.2 式 (2) 系数冲突 | models 2.4 §4.4 式 (8)（L148）：decode T 步总账 = 2NT + 2Ld·T(T+1) ≈ 2NT + 2LT²d，且明言「与一次 teacher-forced 并行前向打 T 个位置的 FLOPs（参数 2NT + attention 2LT²d）逐项相同」；§6（L218）prefill FLOPs = 2NT_p + 2LT_p²d 即此 causal 账。1.2 §2 显式钉死「二次项系数 2 来自 causal mask 的平均可见长 T/2……不打 causal mask 的双向场景系数回到 4」。2.4 取半后 attention = 1.41e14，正确 causal 值 = 2.81e14 → 重算总量 6.65e14 → **8.06e14** FLOPs，t_recompute 1.34 s → **≈1.63 s**；连带：取回便宜倍数 12–20× → ≈15–24×；含 attention 临界 3.2 → ≈2.6 GB/s；加权期望 169 → ≈198 ms、改善 7.9 → ≈8.2×；逐层计算 42 → ≈51 ms。结构式 (1)(2)(3) 与命题 1（权重项 regime）本身正确，结论方向不变 | recap 句改「attention 二次项 2LT²d_model（1.2 式 (1) / models 2.4 §6，已含 causal 减半）」；全部下游数值与练习锚按 8.06e14 / 1.63 s 重算；练习 2(a) 删「上界口径取半」 |
| P2-S2-2 | S2 | 2.2 §3 L70（命题 3）、§4 L103（命题 4）+ 总结 / 练习回指；跨 batch：1.2 命题 2-4 同型 | 命题编号约定冲突：2.2 续接 2.1 编号（3/4）且无声明，而 2.3 / 2.4 从 1 重启；P1 内部也分裂（1.1 命题 1 → 1.2 命题 2-4 续编，1.3 / 1.4 全部无编号）。「命题 1」跨章歧义（2.1 碎片上界 / 2.3 per-channel / 2.4 临界带宽 / 1.1 吞吐守恒） | evaluator grep 全 P1+P2 命题定义行：1.1 命题 1；1.2 命题 2/3/4；1.3 命题（无号）；1.4 命题 ×6（无号）；2.1 命题 1/2；2.2 命题 3/4；2.3 命题 1/2；2.4 命题 1。STYLE_GUIDE §3 公式编号明定「章内 (1)(2)…」，命题编号_scope_ 无维度级裁定 | 需维度级裁定（controller）：建议章内编号（与公式编号同 scope）——2.2 命题 3/4 → 1/2（连同总结 / 练习回指），1.2 命题 2-4 → 1-3 由 P1 batch 同步；或统一续编但全维度声明 + 跨章引用一律带章号。本 batch 内先修 2.2 的无声明续编 |
| P2-S2-3 | S2 | 2.2 练习 3 校验锚 L184 | 「(b) 每条请求的命中长度 + 新插入长度 = 4」对 R3 = [s, a, e] **为假**：R3 长 3 token（命中 2 + 插入 1 = 3） | 题面自设 R1=[s,a,b,c]、R2=[s,a,b,d]、R3=[s,a,e]；R1/R2 长 4、R3 长 3。命中 + 插入 = 请求自身长度是恒等式，锚把 R3 也钉成 4，读者按锚自查必失败 | 锚改「每条请求的命中长度 + 新插入长度 = 该请求 token 数（R1/R2 = 4，R3 = 3）」 |
| P2-S3-1 | S3 | 2.1 §2 L56（block table 引句标 §4.1）、§2 L85（Storing multiple tokens 引句标 §4.1）、§3 L89（within one block 引句标 §4.1）、参考 L292 | 三处引句的**小节号**偏移：block table 句实在 §4.2（KV Cache Manager），within-one-block 与 Storing-multiple-tokens 两句实在 §4.3（Decoding walkthrough 段）；引文本身逐字无误 | evaluator 对 2309.06180v1 HTML 按节标题字符偏移定位：§4.1 @24478 / §4.2 @27216 / §4.3 @28796 / §4.4 @32361；「Each block table entry records」@28457（§4.2）、「within one block」@31597（§4.3）、「Storing multiple tokens」@30976（§4.3）。research notes §2.1 同样误标 §4.1（错误源头在 research pass，章节沿袭） | 三处小节号更正（§4.2 / §4.3 / §4.3），参考行「碎片上界 §4.1」→ §4.3；COW §4.4 / 调度 §4.5 / block size §7.2 / recompute-swap §7.3 均核实无误不动 |
| P2-S3-2 | S3 | 2.2 §1 L20（N≈8e9 声明处）、2.4 §2 L54（8e9 未声明） | FLOPs 参数量口径与 1.2 漂移：1.2 钉死 N=7.50e9（matmul 口径）/ N_full=8.03e9（存储口径）；2.2 用 8e9「教学圆整数」（已声明），2.4 直接用 8e9（未声明）。全量 prefill 实例 69.9 vs 1.2 的 65.8 TFLOP（~7%） | 1.2 §1 L19 两口径声明；2.2 L20 括注；2.4 §1 章脊只声明几何未声明 N。R2-2 要求换口径声明 | 建议统一到 1.2 的 7.50e9 matmul 口径（实例重算），或至少 2.4 补「N≈8e9 圆整数，1.2 用 7.50e9 matmul 口径，差约 7%」跨章口径注 |
| P2-S3-3 | S3 | 2.3 §2 L33（R = 极差）、§4 L76-82（k = MX 块大小） | 局部符号复用他章已占字母：spec §4.3 表 k = top-k（MoE），1.2 / 1.3 R = ridge point；2.3 的 k（OCP 原记号）与 R（量化极差）虽在符号表定义，但未注「局部 / 原规范记号」 | spec §4.3 统一表；1.2 L19「R = ridge point」；OCP MX v1.0 §5.2 用 k 为 scaling block size（PDF 一手） | 各加一句括注（「k 沿 OCP 规范原记号，本节局部，与 MoE top-k 的 k 无关」「R 为本章极差局部记号，非 1.2 的 ridge point」），或改 k_MX / R_range |
| P2-S3-4 | S3 | 2.4 §3 L86 | 「单盘变得可行」无盘级带宽限定：压缩后 BW_req 5.0–6.1 GB/s，而 §2 把单块 NVMe 锚在临界值 4.05 GB/s 附近、练习 4(c) 设单盘 4 GB/s——4 GB/s 盘需 5.4× 压缩 > CacheGen 上限 4.3×，仍不达标 | 2.4 L66（单盘 GB/s 量级在临界值附近）、L86（5.0–6.1 GB/s）、练习 4(c) L207（21.5/4 = 5.375×）；章节内部张力 | 改「≥6 GB/s 级单盘（Gen4 旗舰 / Gen5）变得可行；4 GB/s 级单盘仍差（练习 4(c)）」 |
| P2-S3-5 | S3 | 2.4 §9 L173 | HiSparse kernel 路径前缀混用：同列三路径中前两个是 srt/ 相对（mem_cache/…、managers/…），第三个「kernels/ops/kvcache/hisparse.py」实在 **python/sglang/kernels/**（非 srt/）下；按 srt/ 读则路径不存在 | tarball 一手：python/sglang/srt/kernels 不存在；python/sglang/kernels/ops/kvcache/hisparse.py 存在（另 hisparse_slot_mapping.py） | 给全路径 python/sglang/kernels/ops/kvcache/hisparse.py，或三路径统一带 python/sglang/ 前缀 |
| P2-S3-6 | S3 | 2.1 §1 L37（动态扩容 0.33 s / 336 s）、§5 L230（COW 0.63 μs / 80 μs） | 拷贝时间把「拷贝字节」当单向 HBM 流量折算；memcpy 实为读+写双份流量，时间约 2×。量级结论（二次增长不可行 / 单块拷贝 ≪ 整段）不变 | HBM 带宽 3.35 TB/s 为读写合计口径（datasheet）；1.1 TB 拷贝 = 2.2 TB 流量 → ≈0.66 s | 可选：加一句口径注「拷贝含读写两份流量，时间约 2×，量级结论不变」，或数值直接 ×2 |

## 2. 强制重推导（独立重推，不信任 writer）

### 2.1 碎片上界 — PASS
不变式「除最后一块外所有块皆满」对分配事件归纳：prefill 左到右填充基例成立；decode 写尾块（未满则写入、已满则新块成唯一尾块）保持。故 waste(T,b) = ⌈T/b⌉b − T = b − r(T)（r>0）或 0 ≤ b−1，相对 waste/T ≤ (b−1)/T < b/T。边界：b|T 时 0（不取到）、r=1 时 b−1（取到）。数值复核：T=1000、b=16 → 63 块 1008 slot、waste 8 slot = 1 MiB、0.8% < 1.6%；b=64 → 16 块 1024 slot、waste 24 = 3 MiB、2.4% < 6.4% ✓。「<4% 是 Fig.2 经验均值非 stated bound」与论文原文核对成立（论文文字只有 within one block；20.4%–38.2% 为 Fig.2 实测）✓。

### 2.1 block size 权衡账 — PASS
r(T) 均匀 ⇒ E[waste] = (b−1)/2（b=16 → 7.5 slot = 960 KiB；b=64 → 31.5 = 3.94 MiB ✓）。W(b) ≈ bκL/2 + T·s_table/b，dW/db = κL/2 − T·s_table/b² = 0 ⇒ b* = √(2T·s_table/(κL))。代入 κL=131072、s_table=8：T=4096 → √0.5 ≈ 0.71；T=131072 → √16 = 4 ✓。两档 < 16 ⇒ 内存账最优被压到 1 以下、真实下界来自 kernel 并行度 / swap 效率——与论文 §7.2「16 is large enough… small enough」及 Fig.18(b) ShareGPT 16–128 / Alpaca 16/32 实验口径一致 ✓。块表内存量级（128K/b=16 → 64 KiB/请求，B=256 → 16 MiB = 池 0.03%）✓。共享概率量化（分叉点 D → ⌊D/b⌋ 满块可共享、平均作废 b/2）✓。

### 2.1 COW 正确性（命题 2）— PASS
命题陈述（块表映射正确 + 共享块内容不可变）与证明骨架成立：基例 fork 时刻共享块由公共前缀唯一确定（前缀不可变，models 2.4 §4.2 命题一手核对）；保持步两情形（块边界 → 新独占块零拷贝；共享半满块 → 先复制 ref_cnt=1 再写、原块只减引用）均不写共享块；终止归纳得任意步正确。保持步留练习 3，练习存在且脚手架完整、校验锚（b 整除前缀零拷贝特例）有效 ✓。代码侧语义核对：_apply_cow 登记 (source, cow) 拷贝对 + take_pending_cow_copies 交 worker（single_type_kv_cache_manager.py L585/L553 一手）✓；beam 继承 = 块表映射复制 + 引用加一（论文 Fig.9 语义）✓；边界反例「每次 fork 都拷贝」否证 ✓。

### 2.4 分层命中成本账 — FAIL（S2-1）
结构独立重推：E[t_KV] = h1·0 + h2·t2 + h3·t3 + h0·t_recompute，t_i = κLT/BW_i + ℓ_i ✓。κLT = 32768×131072 = 4.295 GB ✓；t2 = 67 ms、t3 = 107 ms、400G = 86 ms ✓；加权 0.2×67 + 0.2×107 + 0.1×t_rec ✓（算术无误）。**但 t_recompute 的 attention 项误读系数**：独立重推 causal prefill = 2N T + 2LT²d（models 2.4 §4.4 式 (8) 恒等式 + 1.2 式 (1) 钉死）= 5.24e14 + 2.81e14 = 8.06e14 FLOPs → /4.9475e14 = 1.63 s；章节取 6.65e14 → 1.34 s（attention 再取半，双重 causal 折扣）。下游 12–20× / 3.2 GB/s / 169 ms / 7.9× / 42 ms 全部连带偏移（正确 ≈15–24× / 2.6 GB/s / 198 ms / 8.2× / 51 ms）。命题 1（权重项 regime BW* = κL·F_eff/(2N_act)，T 无关）独立重推 PASS：131072×4.9475e14/1.6e10 = 4.05 GB/s ✓，两边界注（attention 使 BW*(T) 降、小 KV 延迟项主导）方向正确 ✓。

### 2.4 传输带宽需求 — PASS
BW_req = κLT/t_budget 独立重推：4.295 GB/0.2 s = 21.5 GB/s ✓；PCIe 64 余量 3×、40 GB/s 余量 1.9× ✓；CacheGen 3.5–4.3× → 5.0–6.1 GB/s ✓；跨 DC 400 Gbps/8 GPU = 6.25 GB/s > 4.05 ✓；逐层 κT = 128 MiB → PCIe 2.1 ms ≪ 每层计算（结论对系数误差不敏感）✓。t_budget = TTFT×份额、份额 10% 教学示例已标注 ✓。

### 2.4 专用集群 vs 分层权衡 — PASS
三条理由逐字对 doc L175 核实（any engine any prefix / autoscaling 解耦 / 故障率高一个量级以上）✓；代价面（多一跳网络 ms 延迟 + 独立硬件）与 Mooncake stranded-capacity 成本结构对照（§5）成立 ✓；前提是 §2 带宽账（网络取回 ≫ 重算，系数修正后结论更强）✓；EC 账 12/10 = 1.2×、可用 10/12 = 83.3%、对三副本容量效率 2.5×、any-10 尾延迟 = 第 10 快顺序统计量（教学推导已标注）✓。

## 3. 跨章一致性（必查项）

| 检查 | 结论 |
| --- | --- |
| **2.2 attention 系数 vs 1.2** | **PASS / 对齐**：2.2 §4 每层 2·d_model·T_p²、全层 2L·d_model·T_p²（乘加计 2、causal 减半、score+PV 两 GEMM）＝ 1.2 式 (1) 2LT²d_model（causal 系数 2 显式声明）。命题 4 节省账 2Ld·δ² 与每对 4d FLOPs × δ²/2 对数独立重推一致 |
| 2.4 attention 系数 vs 1.2 / 2.2 | **FAIL / 需对齐**（S2-1）：2.4 把 2LT²d 当非 causal 上界再取半 |
| κ 几何跨章 | PASS：2.1 式 (1) / 2.3 式 (1) / 2.4 §1 与 1.2 式 (3) 同形（MHA/GQA 2H_kv·d_head·bytes；MLA (d_latent+d_rope)·bytes 无因子 2 无 H_kv）；2.1 §6 / 2.3 §4 的 V4.1 代入 512+64 与 config 一手一致 |
| KV 容量 recap 指针 | PASS：2.1 §1 / 2.3 §1 / 2.4 §1 全部「models 2.4 式 (9)」+ b→bytes 翻译注；无式 (5) 误引 |
| 章脊口径 | 2.1（Llama-3.1-8B GQA κL=128KiB、b=16/64、池 48GiB）→ 2.2 / 2.3 / 2.4 沿用一致；2.3 §4 / 2.4 生产口径换 V4.1 均声明原因 ✓；例外：N 圆整数漂移（S3-2）、2.4 §6 反事实层分布已声明 ✓ |
| 符号 §4.3 | b=block size / B=batch / bytes 不占字母 / d_model / d_latent / d_rope / κ / L / T 全 batch 一致 ✓；γ / 𝒢 本 batch 未出现 ✓；局部符号 q / η / g / w / M / n / δ / h_i / ℓ_i / F_eff / t_budget / s_table 均定义 ✓；例外：命题编号 scope（S2-2）、k / R 复用（S3-3） |
| recap 忠实性 | 2.1「I≈B」recap 带饱和 caveat（models 2.4 式 (12)）✓；2.3 §6 MLA 56.9×（576 vs 32768）与 models 4.1 L160 逐字一致 ✓；2.4 §2 重算 recap 误读（S2-1）；2.2 CachedAttention 四机制与摘要逐字一致 ✓ |
| 两层走读纪律 | PASS：2.1 / 2.2 / 2.4 代码均机制例证片段级 + 「模块全量走读归 5.2 / 6.1」指针；2.4 hisparse/kv_hints 语义未核实留 5.3 显式声明 ✓ |

## 4. 教学审查

- 推导断步：无（除 S1-1 无效证明、S2-1 系数误读）。2.1 命题 1/2、2.2 命题 3/4、2.4 命题 1 路线句 + 三段骨架完整；留作练习的证明（2.1 练习 3、2.2 练习 2、2.3 练习 4）练习均存在且脚手架完整。
- 未定义符号：无（S3-3 为复用而非未定义）。
- 前置缺口：无——四章前置（models 2.4 / 4.1 / 4.2、1.2 / 1.3、2.1→2.2→2.3→2.4）与依赖图 P1→P2 一致；跨维度指引纯文字 ✓。
- 习题校验锚：2.1 ×4 全可校验（练习 4 源码注释逐字核实）；2.2 练习 1/2/4 可校验、**练习 3 锚错（S2-3）**；2.3 ×4 可校验（练习 2 内容受 S1-1 牵连）；2.4 ×4 可校验（练习 2/3 数值受 S2-1 牵连）。难度分布四章均合规（非递减、≥1 题 [10]/[20]、无 [40]、每章 ≥1 题以正文命题为证明目标）。
- 数值实例密度（R2-3）：四章每个 display 公式后 300 字内均有算到底的实例 ✓。

## 5. Writer 待核项裁定（W-C / W-D）

### W-C（2.1 / 2.2 代码锚点）— 全部成立
| 项 | 一手证据（vllm / sglang main tarball，2026-11-23） | 裁定 |
| --- | --- | --- |
| watermark 默认 0.0 @ config/scheduler.py L232 | L232 `watermark: float = Field(default=0.0, ge=0.0, lt=1.0)`，docstring「0.0 (the default) disables the watermark」 | ✓ |
| 协商函数 @ vllm/platforms/interface.py L711（非 v1/） | L710-711 `@classmethod / def _preferred_block_size_for_backends`；docstring 含「CPU_MLA takes 16 and no multiple of it」 | ✓ |
| block 16（论文）vs 64（当前 main） | 论文 §7.2「vLLM sets its default block size as 16」逐字；config/cache.py L185 `DEFAULT_BLOCK_SIZE: ClassVar[int] = 64` | ✓ 两口径章节均给 |
| SWA 256 / CPU_MLA 16 | sparse_swa.py L166-167 `get_preferred_block_size → return 256`；cpu_mla.py L25-26「the platform layer forces block_size=16 for MLA models」+ get_supported_kernel_block_sizes | ✓ |
| TTL 0.5–1% chatbot-3d 反事实限定 | doc L187「a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate」vs L211 生产「72 hours for agent… 7 days for chatbot… almost all reusable」（L187 的 agent「3 days」= 72h，两处自洽）；2.2 §5 两口径分列 + 坑注 | ✓ 限定正确 |
| HiRadixCache → UnifiedRadixCache 漂移 | sglang main 全 mem_cache/ grep `class HiRadixCache` 零命中；unified_radix_cache.py L224 `class UnifiedRadixCache`；radix_cache.py L282 `class RadixCache` 仍在；2.2 §3 / 2.4 §9 漂移注 | ✓ |
| 其余锚点（抽全） | block_pool.py：get_new_blocks L681 / touch L774 / free_blocks L798 / cache_full_blocks L203 / null block L161-165 / NOTE #1「block tables are append-only」逐字 / LIFO-FIFO 注释 L827/L831 逐字 / prepend_n-append_n ✓；kv_cache_manager.py：watermark_blocks L200 / WAITING-PREEMPTED 条件 L529-536 / 失败路径 L595-599 / remove_skipped_blocks L575 / get_computed_blocks L278 / make_prefix_cache_stats L228（stats.py「queries: number of tokens」token 口径 ✓）；kv_cache_utils.py hash_block_tokens L697 签名与返回逐字 + NONE_HASH L156；cache.py L269 sha256 + L177 四算法；coordinator L106-107 LCM assert + find_longest_cache_hit ✓；single_type：FullAttention/SlidingWindow/Mamba Manager + _apply_cow L585 + take_pending_cow_copies L553 ✓；parallel_sampling.py ParentRequest L13 ✓；entrypoints/generate/beam_search/ ✓；radix_cache.py：_match_prefix_helper L658 / _split_node L684（value[:split_len].clone() L693）/ inc_lock_ref L589 / evict L559 堆语义逐行 / page_aligned / extra_key docstring「never share」✓；schedule_policy.py L191-192 LPM/DFS_WEIGHT ✓ | ✓ |

### W-D（2.3 / 2.4 口径与限定）— 口径分离成立；主 KV 字节误读独立成 S1-2
| 项 | 一手证据 | 裁定 |
| --- | --- | --- |
| 3FS EC 10+2 / 「11 节点 any 10」= 生产披露 vs 开源 CRAQ-only | doc L111-113 三句逐字（2.4 §8 引文无误）；3FS main tarball 全 repo grep -ri "erasure" **零命中**；design_notes.md L11/L105/L151 CRAQ write-all-read-any；2.4 §7/§8 两口径分列 + 张力注（12 节点组 vs 组内 11 节点，不自行修复 doc）+ 坑注 | ✓ 分离正确，未把 EC 归开源 repo |
| CacheGen multiprocess 未实现限定 | LMCache dev tarball docs/source/mp/cachegen.rst 逐字「not implemented yet (coming soon)… LMCache_inprocess」；kernels csrc/cuda/cachegen_kernels.cuh + torch_ops/cachegen_kernels.py + musa/sycl 变体存在；blend/ 与 integration/{vllm,sglang}/ 存在；2.4 §6 限定完整 | ✓ |
| NVFP4 per cuBLAS / TRT-LLM 官方文档 | cuBLAS 逐字「a dedicated scaling factor stored as an 8-bit CUDA_R_8F_UE4M3 value for each 16-element block in the innermost dimension」+ SCALAR_32F tensor-wide（仅 UE4M3 时可用）+「Blackwell GPUs (compute capability 10.0 and above)… 4-bit」+ Table 4 CUDA_R_4F_E2M1；TRT-LLM kv_cache dtype='nvfp4'（ModelOpt 离线）+ NVFP4_PLUS「On Rubin (SM107)… one UE5M3 scale per 32 values and no FP32 global scale」；2.3 §4 引文与坑注逐字无误 | ✓ |
| 教学假设值标注 | F_eff 50%（2.4 L54「教学假设口径」+ MFU 说明）✓；传输份额 10%（L84「教学示例…不是上限」）✓；命中率例 (0.5,0.2,0.2,0.1)（L72「教学示例值」）✓；单盘 4 GB/s（练习 4「教学口径」）✓；528 B 构成（「推导口径，research 侧算术观察」）✓；2.1 §6 反事实层分布（声明原因）✓ | ✓ 充分 |
| DeepInfer 主 KV 元素字节 | doc「Where is FP4 used?」→ E2M1 0.5 B；章节代入 1 B | **✗ S1-2** |
| 其余 doc 引句（抽全） | L97 跨 DC 400Gbps/8GPU、L101 20×30.73TB/40GB/s 读 5GB/s 写/24h、L105 14.4TB/GPU 逻辑、L115 GC Worker LRU 水位、L139 any-prefix 逐字、L165-169 三层放置表、L175 三理由 + 「more than an order of magnitude higher」逐字、L183 decode 不 offload 逐字、L191 GDR 逐字 + CHECK metadata-only + 三段流水、L195 write-back 读校验、L199 版本元组、L201 未做格式迁移、L203-207 CXL 三理由 + SSD 写开放问题、L211 TTL 72h 跨周末、L217-223 StateBlock 20 层/134 token/528 B MXFP8/128 粒度/+6 位置/越界 ≤5/内存目录 | ✓ 全部逐字一致 |

## 6. 规范审查（STYLE_GUIDE + R2 + rubric）

- lint：四章 LINT OK（禁令词 / 模板节 / metadata / arXiv 白名单 / 证据等级 / $$ 配平）。
- 篇幅（CJK 不含代码）：2.1 = 5848、2.2 = 4315、2.3 = 3797、2.4 = 5676；【重】章 ≥5500 软目标达标，标准章 ≥3500 达标。ASCII 结构图每章 ≥1 ✓。
- 证据等级：全部数值事实带出处 + 等级（[A]/[B]/[C]/[W]/教学参照 + 官方披露口径注 + 核实日期）；CachedAttention [B] 无 repo 限定 ✓；NVFP4 blog 反爬以 cuBLAS/TRT-LLM 为据的口径注 ✓；引用格式 [X et al., 年] 链接 ../references.md ✓；章间引用纯文字 ✓。
- 历史叙事：零（论文只作 citation 证据）；rubric A 禁令词手工 + 机械双扫零命中；空洞形容词零。
- R2-1 节首（问题/数值/元问题）四章逐节 ✓；R2-2 章脊锁定 + 换口径声明 ✓（例外 S3-2）；R2-3 公式后实例 ✓；R2-4 命题骨架 ✓（内容问题见 S1-1/S2-1，编号 scope 见 S2-2）；R2-5 路线句 ✓；R2-6 定义配例 + 边界反例 ✓；R2-7 口径注 ✓；R2-8 判断句定义 ✓；R2-9 关节词 ✓；R2-10 提点块引用可删 ✓；R2-11 坑注（触发-现象-规避）✓；R2-12 总结表回指列 ✓。
- 公式硬规则：\lt 用法 ✓、反转义替代（\thinspace/\lbrace/\Vert）✓、backtick 保护形 ✓、flanking 抽查未见违规、表格内公式无裸 | ✓。
- 术语：首现「中文（English）」+ 此后英文统一 ✓（block table / COW / radix tree / TTL / GDR / CRAQ / USRBIO 等）；与 references.md 术语表一致 ✓。

## 7. Verification log

覆盖率：四章全部编号公式（2.1 ×9、2.2 ×4、2.3 ×5、2.4 ×3 + 命题 1）独立重推/代入复核 = 100%；全部数值实例（约 85 处）重算 = 100%；全部论文引句（vLLM ×10、SGLang ×8）一手 HTML 逐字核对 = 100%；全部代码锚点（vLLM ×20、SGLang ×12）tarball 行号级核实 = 100%；DeepInfer doc 引句 ×22 行逐字 = 100%；arXiv ID ×12（2309.06180 / 2312.07104 / 2403.19708 / 2405.16444 / 2407.00023 / 2402.02750 / 2401.18079 / 2306.14048 / 2309.17453 / 2404.14469 / 2407.00079 / 2310.07240）标题↔第一作者↔摘要数字 API/HTML 核对 = 100%。

### 7.1 论文与摘要（一手 URL + 原文引句）

| 抽查项 | 来源 | 原文/证据 | 结论 |
| --- | --- | --- | --- |
| 碎片上界引句 | arxiv.org/html/2309.06180v1 | 「vLLM limits all the memory wastes for a request within one block」 | ✓ 逐字；**小节实在 §4.3**（S3-1） |
| 20.4%–38.2% | 同上 §2 | 「only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems」 | ✓ Fig.2/§2 口径正确 |
| 三浪费分类 | 同上 Fig.3 caption + §2 | 「Three types of memory wastes – reserved, internal fragmentation, and external fragmentation」；「external… known before serving a request」「internal… only realized after a request has finished」 | ✓ 2.1 §1 时间性辨析忠实 |
| block table 引句 | 同上 | 「Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions」 | ✓ 逐字；**实在 §4.2**（S3-1） |
| b>1 动机引句 | 同上 | 「Storing multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel」 | ✓ 逐字；**实在 §4.3**（S3-1） |
| Fig.6 走读 | 同上 §4.3 | 7 token prompt、逻辑 0/1 → 物理 7/1、前 4 token 块 0、后 3 块 1、第 1 decode 步填空 slot、第 2 步分配物理 3 | ✓ 2.1 ASCII 重绘忠实 |
| COW 引句 | 同上 §4.4 | 「The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block, as in parallel decoding. This involves only copying one block of data.」 | ✓（章节省略号处理合规） |
| 共享前缀末块 COW | 同上 §4.4 | 「with the last block marked copy-on-write」 | ✓ |
| 55% beam 节省 | 同上 §2 + §6.3 | §2「up to 55% memory saving, see §6.3」；§6.3「37.6% - 55.2% on beam search」（Alpaca/OPT-13B；ShareGPT 44.3%–66.3%） | ✓ 章节「workload 依赖，非普适上界」限定正确 |
| FCFS / all-or-nothing / gang | 同上 §4.5 | 「first-come-first-serve (FCFS)」「latest requests are preempted first」「all-or-nothing eviction policy」「gang-scheduled as a sequence group」 | ✓ |
| 分布式块表 | 同上 §4.6 | 「a single KV cache manager within the centralized scheduler」「GPU workers do not need to synchronize on memory management… at the beginning of each decoding iteration」 | ✓ 2.1 §2「映射共享、内容分片」忠实 |
| block size 权衡引句 + 默认 16 | 同上 §7.2 | 「If the block size is too small… If the block size is too large, internal fragmentation increases and the probability of sharing decreases」「vLLM sets its default block size as 16」「16 is large enough… small enough」 | ✓ 逐字 |
| ShareGPT/Alpaca + Fig.18(b) | 同上 §7.2 | 「block sizes from 16 to 128 lead to the best performance」「Alpaca… 16 and 32 work well, larger… significantly degrade… sequences become shorter than the block sizes」「In Fig. 18(b)」 | ✓ |
| recompute ≤20% swap | 同上 §7.3 | 「recomputation overhead is never higher than 20% of swapping's latency」 | ✓ |
| SGLang 6.4× | arxiv.org/html/2312.07104v2 摘要/§1/§6.2 | 「up to 6.4× higher throughput」 | ✓ workload 依赖限定在章 |
| page = 1 token | 同上 §3.1 | 「the size of each page is equivalent to one token」 | ✓ 逐字 |
| LRU 叶优先 | 同上 §3.1 | 「evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves」 | ✓ 逐字 |
| 共池无固定配额 | 同上 §3.1 | 「we do not preallocate a fixed-size memory pool as a cache… evict all cached tokens in favor of a larger batch size」 | ✓ |
| Theorem 3.1（DFS 最优 + LPM≡DFS） | 同上 §3.1 + A.3 | 「optimal cache hit rate by visiting… depth-first search order, with a cache size ≥ the maximum request length. The longest-shared-prefix-first order is equivalent to a depth-first search order」；A.3 归纳证明 | ✓ 2.2 §3 表述忠实 |
| 饥饿 future work | 同上 §3.1 | 「greedy cache-aware scheduling… can lead to starvation. We leave its integration with other fair scheduling methods as future work」 | ✓ |
| Fig.3 九时点 | 同上 Fig.3 caption | 「illustrated across nine time points… two chat sessions, a batch of few-shot learning inquiries, and a self-consistency sampling」 | ✓ ASCII 重绘出处正确 |
| KIVI 粒度/收益 | arXiv API 2402.02750 + html v2 | 摘要「key… per-channel… value… per-token」「2.6× less peak memory (including model weight)」「4× larger batch」「2.35×~3.47× throughput」；§3「the residual key cache and value cache are kept in full precision」「The persistence of outliers within each channel… confine… without impacting the other normal channels」 | ✓ 数字与引句逐字；**论文无普适 max-bound 占优声明**（S1-1 佐证） |
| KVQuant 四技术/数字 | arXiv API 2401.18079 | Per-Channel Key / Pre-RoPE Key / per-layer sensitivity-weighted NUQ / Per-Vector Dense-and-Sparse；「< 0.1 perplexity… 3-bit… Wikitext-2 and C4」「1 million… single A100-80GB… 10 million… 8-GPU」「~1.7x speedups」 | ✓ |
| H2O 数字 | arXiv API 2306.14048 | 「20% heavy hitters… 29×, 29×, and 3×… OPT-6.7B and OPT-30B」「same batch size… reduce the latency by up to 1.9×」「dynamic submodular… theoretical guarantee」 | ✓ |
| StreamingLLM | arXiv API 2309.17453 | attention sink 初始 token 强 score「even if they are not semantically important」「4 million tokens and more」「Llama-2, MPT, Falcon, and Pythia」「placeholder token as a dedicated attention sink」「22.2x speedup」 | ✓ |
| SnapKV | arXiv API 2404.14469 | 「observation window located at the end of the prompts」「clustered important KV positions for each attention head」「3.6x… 8.2x… 16K」「380K… single A100-80GB… minor changes」「negligible accuracy drop in the Needle-in-a-Haystack」 | ✓ |
| Mooncake | arXiv API 2407.00079 | 「serving platform for Kimi」「KVCache-centric disaggregated」「underutilized CPU, DRAM, and SSD」「prediction-based early rejection」「525% increase in throughput… simulated」「75% more requests」 | ✓ |
| CacheGen | arXiv API 2310.07240 + html v6 | 摘要「3.5-4.3x… 3.2-3.7x… negligible impact」；§5.2「lossless arithmetic coder… compresses the values in each layer and channel separately」「delta tensors」；§5.1.1 token-wise locality | ✓ 2.4 §6 机制全核对 |
| CachedAttention | arXiv API 2403.19708 | 「hierarchical KV caching… cost-effective memory/storage」「layer-wise pre-loading and asynchronous saving」「scheduler-aware fetching and eviction」「decoupling the positional encoding and effectively truncating」；摘要无代码链接 | ✓ 四机制 + [B] 维持 |
| CacheBlend | arXiv API 2405.16444 + GitHub API | 「reuses… regardless prefix or not… selectively recomputes… a small subset of tokens」「without compromising generation quality」；YaoJiayi/CacheBlend API 200 | ✓ |
| Preble | arXiv API 2407.00023 + GitHub API | 「first distributed LLM serving platform… prompt sharing」「co-optimizes KV state reuse and computation load-balancing」；WukLab/preble API 200 | ✓ |

### 7.2 代码锚点（codeload tarball，2026-11-23 main/dev）

| 抽查项 | 来源 | 证据 | 结论 |
| --- | --- | --- | --- |
| DEFAULT_BLOCK_SIZE=64 | vllm config/cache.py L185 | `DEFAULT_BLOCK_SIZE: ClassVar[int] = 64` | ✓ |
| watermark 0.0 | vllm config/scheduler.py L232 | `Field(default=0.0, ge=0.0, lt=1.0)` + docstring | ✓ |
| 协商函数位置 | vllm platforms/interface.py L711 | `_preferred_block_size_for_backends`（非 v1/） | ✓ |
| SWA 256 / CPU_MLA 16 | vllm v1/attention/backends/mla/{sparse_swa,cpu_mla}.py | L166-167 return 256；「platform layer forces block_size=16」 | ✓ |
| block_pool 函数族 | vllm v1/core/block_pool.py | L681/L774/L798/L203/L161-165；NOTE #1 append-only 逐字；LIFO/FIFO 注释逐字；touch/free 语义逐行 | ✓ |
| kv_cache_manager | vllm v1/core/kv_cache_manager.py | L200/L228/L278/L529-536/L575/L595-599 逐行 | ✓ |
| hash 链 | vllm v1/core/kv_cache_utils.py L697 + config/cache.py L269/L177 | hash_block_tokens 签名/返回逐字；NONE_HASH；sha256 默认 + 四算法 | ✓ |
| coordinator / 逐类型 | vllm v1/core/{kv_cache_coordinator,single_type_kv_cache_manager}.py | LCM assert L106-107；FullAttention/SlidingWindow/Mamba；_apply_cow L585 拷贝对语义；take_pending_cow_copies L553 | ✓ |
| parallel sampling / beam | vllm v1/engine/parallel_sampling.py L13；entrypoints/generate/beam_search/ | ParentRequest；目录存在 | ✓ |
| offload 模块树 | vllm v1/kv_offload/ + simple_kv_offload/ | base.py docstring 逐字；cpu/tiering/policies/{lru,arc}/p2p{nixl 数据面+zmq 控制面+session}；SimpleCPUOffloadConnector docstring 逐字；hisparse/ kv_hints/ 存在 | ✓ |
| radix_cache 函数族 | sglang srt/mem_cache/radix_cache.py | L282/L658/L684/L693/L589/L559；evict 堆语义逐行；last_access_time L260；page_aligned；extra_key「never share」docstring | ✓ |
| UnifiedRadixCache 漂移 | sglang srt/mem_cache/ | HiRadixCache 零命中；unified_radix_cache.py L224 | ✓ |
| schedule_policy | sglang srt/managers/schedule_policy.py L191-192 | LPM / DFS_WEIGHT | ✓ |
| HiCache 层级/共享 | sglang docs hicache_design.mdx + mem_cache/storage/ | L1/L2 private、L3「Only if the backend is set up for it」tier sharing 表；backends file/mooncake_store/hf3fs/nixl/aibrix/lmcache/eic/flexkv/seaweedfs 全在 | ✓ |
| HiSparse 前提/direct-to-host | sglang docs hisparse_guide.mdx + 源码 | DSA 模型列表（V3.2/GLM-5.1/Kimi-K3.1 DSA 层/V4/MiniMax M3）+ PD + decode-only 逐字；「writes only C4 KV… c4_indexer and C128 KV remain device-to-device」逐字；hot buffer「4KB tokens」；kernel 实在 python/sglang/kernels/ops/kvcache/hisparse.py | ✓（路径前缀 S3-5） |
| 3FS CRAQ / 无 EC | 3FS README + docs/design_notes.md | CRAQ write-all-read-any L11/L105/L151；全 repo grep erasure 零命中；四组件 L5；FoundationDB 无状态 L9 | ✓ |
| 3FS FUSE / USRBIO | 3FS docs/design_notes.md L29-33 + src/lib/api/UsrbIo.md | 「approximately 400K 4KiB reads per second」逐字；kernel-user 拷贝耗带宽；Linux 5.x 同文件并发写不支持；USRBIO/Iov/Ior/io_depth/fd 注册逐字；benchmarks/fio_usrbio/ 存在 | ✓ |
| LMCache CacheGen | LMCache dev docs/source/mp/cachegen.rst + csrc/ | 「not implemented yet (coming soon)… LMCache_inprocess」逐字；cuda/musa/sycl kernels；blend/ + integration/{vllm,sglang}/ | ✓ |
| Mooncake TE | Mooncake docs/source/design/transfer-engine/index.md | Segment/BatchTransfer「asynchronous and more flexible AllScatter/AllGather」逐字；每进程一 Segment；Buffer (GPUDirect) RDMA；NVMeof 零拷贝；Transport 八种逐字；topology matrix 广播 + preferred/secondary + 本地 NUMA/PCIe switch GDR（L62-64）；本地 memcpy/cudaMemcpy（L50）；transfer_engine.h 存在 | ✓ |

### 7.3 官方文档 / config / 规范

| 抽查项 | 来源 | 证据 | 结论 |
| --- | --- | --- | --- |
| NVFP4 格式 | docs.nvidia.com/cuda/cublas/index.html | VEC16_UE4M3 引句逐字；SCALAR_32F tensor-wide 仅 UE4M3 可用；FP4 → VEC16_UE4M3；「Blackwell… compute capability 10.0 and above… 4-bit」；Table 4 CUDA_R_4F_E2M1 | ✓（并证 S1-2：元素 0.5 B） |
| TRT-LLM KV nvfp4 / NVFP4_PLUS | nvidia.github.io/TensorRT-LLM/features/quantization.html | KvCacheConfig(dtype='nvfp4') + ModelOpt 离线；NVFP4_PLUS「Rubin (SM107)… UE5M3 scale per 32 values and no FP32 global scale」 | ✓ 2.3 坑注逐字 |
| OCP MX v1.0 | opencompute.org PDF（pdftotext） | Table 1：MXFP8(E5M2/E4M3)/MXFP6/MXFP4(E2M1)/MXINT8 一律 k=32 + E8M0 8bit；§5.4.1「E8M0 is an unsigned representation of a conventional biased Float32 exponent」逐字；「Different encodings… considered different formats」 | ✓ |
| V4.1-Flash config | hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json | 40L / hidden 5120 / kv_heads 1 / head_dim 512 / qk_rope 64 / sliding_window 128 / max_position_embeddings 1048576 / index_topk 512 / index_n_heads 32 / index_head_dim 128 / nextn 3 / top-6 / fp8+ue8m0+[32,32]+expert fp4；kv_lora_rank 字段不存在（head_dim 512 表达，1.2 坑注口径） | ✓ |
| DeepInfer doc ×22 行 | DeepInfer main tarball doc/inference_deployment_qa.md | 见 §5 W-D 表（L97/101/105/111-113/115/139/165-169/175/183/187/191/195/199/201/203-207/211/217-223/269-275 逐字） | ✓（L271 元素口径 → S1-2） |
| Llama-3.1-8B 几何 | models 2.4 §5 表（L205）+ models references llama-3.1-8b-config [A] 2026-11-10 | L=32/H_kv=8/d_head=128/T=131072/128KiB/16GiB | ✓ 章脊与已核实 config 账一致 |
| models 2.4 式 (9)/(8)/§6 | models/02-transformer-anatomy/04-causality-and-kv-cache.md L148/L188/L218 | 式 (9) 2LH_kv d_head TBb；式 (8) 2NT+2Ld·T(T+1) 与 teacher-forced 前向恒等；§6 prefill 2NT_p+2LT_p²d | ✓ recap 指针正确；**证 S2-1：2LT²d 已含 causal** |
| models 4.1 MLA 56.9× | models/04-modern-architectures/01-*.md L160 | 「576 对 32768，56.9 倍」 | ✓ 2.3 §6 recap 忠实 |
| models 2.4 §4.2 前缀不可变 | 同上 L112-114 | 命题 + 对 ℓ 归纳证明 | ✓ 2.1/2.2 recap 指针（§4）正确 |

### 7.4 数值重算抽查（全部实例重算，列代表性样本）

| 项 | 章节 | 重算 | 结论 |
| --- | --- | --- | --- |
| κ=4096B、κL=128KiB、满窗 16GiB、B=32 512GiB | 2.1/2.3/2.4 | 2×8×128×2=4096；×32=131072；×131072=2^34=16GiB | ✓ |
| 预留浪费 M=2048/T=256 → 224MiB/87.5% | 2.1 | 1792×128KiB=224MiB；1792/2048=87.5% | ✓ |
| 动态扩容 T²κL/2：T=4096→1.1TB/0.33s；T=131072→1125TB/336s | 2.1 | 8.39e6×131072≈1.1e12；/3.35e12≈0.33；8.59e9×131072≈1.126e15→336s | ✓（读+写 2× 口径注 S3-6） |
| 并发账 192 vs 768（4×） | 2.1 | 48GiB/256MiB=192；48GiB/64MiB=768 | ✓ |
| b*=0.71/4；E[waste] 960KiB/3.94MiB；块表 64KiB/16MiB/0.03% | 2.1 | √(2×4096×8/131072)=√0.5；√16=4；7.5×128KiB；31.5×128KiB；8192×8B | ✓ |
| watermark 6144 块×0.01=61 块=488MiB | 2.1 | 48GiB/8MiB=6144；⌊61.44⌋×8MiB | ✓ |
| SWA ≤⌈w/b⌉+1=3 块；混合 480+7.5=487.5MiB vs 640MiB 省 23.8%；B=32 15.2/20GiB | 2.1 | 30 层×4096×4KiB=480MiB；3×64×10×4KiB=7.5MiB；(640−487.5)/640=23.8% | ✓ |
| COW 2MiB/0.63μs；256MiB/80μs；128×；n=4 省 768MiB=75% | 2.1 | 16×128KiB=2MiB；2.097e6/3.35e12=0.63μs；3×2048×128KiB=768MiB | ✓（S3-6 同注） |
| hash 链 64 块/48 命中=3072；尾损 32token=4MiB=0.512TFLOP=0.52ms | 2.2 | 4096/64=64；32×128KiB=4MiB；2×8e9×32/989.5e12=0.52ms | ✓ |
| 命题 4 实例 49.2+2.47=51.6TFLOP；全量 69.9；73.8%；52ms | 2.2 | 2×8e9×3072=4.92e13；2×32×4096×3072²=2.47e12；51.6/69.9=73.8%；/989.5e12=52ms | ✓（N 口径 S3-2） |
| 式 (3) 极限 h/h²；T=131072 62.2% | 2.2 | 0.75×(8e9+0.75×1.718e10)/(8e9+1.718e10)=62.2% | ✓ |
| 重叠不等式 16MiB/131μs vs 2.07ms=15.8× | 2.2 | 4096×4096=16MiB；/128e9=131μs；2×(8e9/32)×4096/989.5e12=2.07ms | ✓ |
| 量化 Δ=20/3=6.67 界 3.33；正常通道 0.667/0.333=16.7%；开销 16/(gq)=25%/6.25% | 2.3 | 逐式代入 | ✓（命题 1 普适性 S1-1） |
| MXFP8 1.03125/1.94×；MXFP4 0.53125/3.76×；NVFP4 0.5625/3.56× | 2.3 | (32+1)/32；(16+1)/32；(8+1)/16 | ✓ |
| DeepInfer 主 KV 612B/1.88×/23.9KiB/23.9GiB | 2.3 | (16×1+1)/16=1.0625→576×1.0625=612 | **✗ 元素应 0.5B → 324B/3.56×/12.7KiB/12.7GiB（S1-2）** |
| 召回 4100/32768=12.5%；组合 0.2×2/16=1/40→0.4GiB；MLA 56.9× | 2.3 | 逐式 | ✓ |
| 三层账 PCIe 64GB/s（32GT/s×16/8，128b/130b −1.5%）；100PiB=1.13e17B；512GiB≈550GB=6.9× | 2.4 | 逐式 | ✓ |
| t2/t3/400G=67/107/86ms；重算 6.65e14→1.34s | 2.4 | 4.295/64、/40、/50；5.24e14+1.41e14=6.65e14 | **✗ attention 应 2.81e14 → 8.06e14 → 1.63s（S2-1）** |
| BW*=4.05GB/s；含 attention 3.2GB/s；加权 169ms/7.9×/80% | 2.4 | 131072×4.9475e14/1.6e10=4.05e9；4.295/1.34=3.2；0.2×67+0.2×107+0.1×1340=168.8 | 结构 ✓；含 attention 项与加权值随 S2-1 修正（→2.6GB/s、198ms、8.2×、82%） |
| BW_req 21.5GB/s；压缩 5.0–6.1；跨 DC 6.25；逐层 128MiB/2.1ms/42ms | 2.4 | 4.295/0.2；21.5/4.3、/3.5；50/8；1.34e8/64e9=2.1ms | ✓（42ms 随 S2-1 →51ms） |
| 写容量 3.8e4 token/s；16384 次小拷贝；FUSE 1.64GB/s 差 24×；EC 1.2×/83.3%/2.5×；StateBlock 1,415,040B=1.35MiB/528=512+16 | 2.4 | 5e9/131072；512×32；400000×4096；40/1.64；12/10、10/12、3/1.2；20×134×528 | ✓ |
| CacheGen 1.0–1.23GB/20–25ms；Mooncake 525%/75% | 2.4 | 4.295/4.3、/3.5；/50 | ✓ |

## 8. 修复与复审要求

1. **2.3**：S1-1（命题 1 重述 + 练习 2 + 总结 + §3 引用）、S1-2（主 KV 字节账 + 总结）修复后**复审**（含 S1 章硬要求）。
2. **2.4**：S1-2 传播值（§9 + 总结）、S2-1（attention 系数全链重算 + 练习锚）修复后**复审**（含 S1 传播值）。
3. **2.2**：S2-2（命题编号，待维度级裁定后执行）、S2-3（练习 3 锚）必须修复；S3-2 建议随修。
4. **2.1**：S3-1 / S3-6 累计下波清零（本轮可随修）。
5. S3 全量逐条判定（修复/接受/驳回）记入 inference/README.md 审查记录区；writer 反驳通道附证据复核。
6. 跨 batch 协调项：命题编号 scope 裁定（涉 1.2）、N 口径统一（涉 1.2）、research notes §2.1 的 §4.1 小节误标建议 research pass 同步更正（错误源头）。
