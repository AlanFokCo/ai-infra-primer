# training W3 批 2 审查报告：5.4-5.5 + P6 三章 + 全波 8 章跨章 sweep

日期：2026-11-17。evaluator：task-20（独立 fresh context，与写作 agent、批 1 evaluator 无共享上下文）。
对象：training/05-model-parallelism/{04,05}、training/06-megatron-and-stacks/{01,02,03} 逐字审查；05-model-parallelism/{01,02,03} 跨章扫读；回链对照 1.2/1.4/1.5/2.4/3.2/4.1-4.4 与 models/04-modern-architectures/05-moe.md。

**结论：S1 = 0，S2 = 7，S3 = 10 → 不放行（S2 修复后轮复审，S3 随下一波清零）。**

## 0. 机械检查

| 章 | lint | CJK 字数（篇幅线） |
| --- | --- | --- |
| 5.4 | LINT OK | 3654（≥3500 ✓） |
| 5.5 | LINT OK | 3758（≥3500 ✓） |
| 6.1 | LINT OK | 3702（≥3500 ✓） |
| 6.2 | LINT OK | 3519（≥3500 ✓） |
| 6.3 | LINT OK | 2540（短篇 ≥2500 ✓） |
| 5.1-5.3（sweep） | 3× LINT OK | 5566 / 3581 / 5932（5.1、5.3 重章 ≥5500 ✓） |

全 8 章 lint 通过；每章 ≥1 张 ASCII 图 ✓；模板三节齐 ✓；证据等级标注齐 ✓；篇幅全达标 ✓。

## 1. Findings 表

### S1（0 起）

无。全部一手来源核查命中（见 §3 verification log）：herd Table 4 三行逐字命中；DSv3 部署/内核/偏置控制器/推理冗余参数逐字命中；K2 §2.4 逐段命中；GShard 三处原文命中；Megatron-LM main（2026-11-17 tarball）行号级锚点 30+ 处全部命中（含 6.2 引用的全部行号）；NeMo 重定向、Megatron-Bridge、Megatron-DeepSpeed pushed_at、OLMo-core 文件级配置、torchtitan README 与目录树均命中。

### S2（7 起，必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 5.4 §6 末段 | 「跨节点流量相比无限制路由（期望约 5.3 个节点）再降约 1/4」 | 实际 8(1−(7/8)⁸)=5.2511 → 4(1−(3/4)⁸)=3.5995，降幅 (5.2511−3.5995)/5.2511 = **31.5% ≈ 1/3**，非 1/4（本章 §2 自己给出的两个数当场可复算） | 「再降约 1/4」改「再降约 1/3」（或写 30%） |
| S2-2 | 5.4 §5 叠加行 + 5.5 §5 表 EP 行 | EP 通信按「前向 dispatch+combine」计，与同章/同表的全程（前向+反向）口径 siblings 不一致：5.4 §5 写「每 step EP 通信 = 2kd bytes × token 数 × 层数」（既漏 §1 自立的 bf16 ×2，又漏 §7 自立的反向对称 ×2——全程应为 8kd bytes 量级）；5.5 §5 EP 行「≤4kh·L_moe,p·τ（每 token 每 MoE 层 2kh 元素 ×2 bytes）」= 前向值，而 TP 行 8h（明写「前向 2 + 反向 2」）、PP 行 8h、CP-ring 行 8h_kv(c−1) 均为全程值 | 5.4 §1「每 token 每层 2kd 个元素，bf16 下 2kd×2 bytes」+ §7「EP 的通信账因此前向反向对称」；5.5 §5 表头「每 GPU 每 step 字节数」与 TP 行的「前向 2 + 反向 2」注记 | 统一为全程口径：5.4 §5 改「每 token 每 MoE 层全程 ≤ 8kd bytes（前向 4kd：dispatch+combine 各 kd 元素 ×2 bytes；反向对称，§7）」；5.5 EP 行改「≤ 8kh·L_moe,p·τ」或在表注声明该行为前向口径——但同表混口径不可留 |
| S2-3 | 5.4 §6「1:1 复算」段 | 复算分子取前向通信（2M·d·2 = 115 KB，dispatch+combine 一程），分母取全程计算（6ND 口径 = 前向+反向）——非对称 pairing 得 1.1 并以此「相合」论文的 1:1；若通信同样按全程计（§7 自立反向对称），比值为 2.2，「1:1 含义：无重叠时 all-to-all 吃掉约一半算力」的读数随之变为约 2/3 | 本章 §7 vjp 段「两次 all-to-all 在反向再走一遍」；复算原文自标「本书复算，口径假设随注」但未注这条非对称 | 补一句口径注记：分子为前向一程、分母为全程 6ND，通信若对称翻倍则比值约 2.2——论文 1:1 的分子口径未知；或把复算改成全全程口径并改写结论读数 |
| S2-4 | 5.5 §5 表 CP（Ulysses）行 | 「8h·L_p·(c−1)/c·τ（每层 4 次 all-to-all）」：8h(c−1)/c 是**前向一程**值（5.2 式 (5)：8bsh(c−1)/c² per layer forward，按 5.5 表注 (ii) 的本地 τ 口径折算恰为 8h(c−1)/c/token）；括号注「每层 4 次」指前向 2 + 反向 2，反向两次同量未计入——行内自相矛盾 2× | 5.2 式 (5) 及其 §6 表头「每层前向每卡发送」（5.2 自声明前向口径，内部一致；问题在 5.5 把前向值搬进了全程表）；独立重推：前向 QKV 换维 6bsh(c−1)/c² + 输出换回 2bsh(c−1)/c²，反向同量 → 全程 16bsh(c−1)/c² → 16h(c−1)/c per local token | 行值改「16h·L_p·(c−1)/c·τ」 |
| S2-5 | 5.2 §2 vs 6.1 §2/§6（跨章 sweep 命中） | 同一对 SP 转换算子两套命名无桥接：5.2 记 ḡ（all-gather 前向，处 f 位）/ f̄（reduce-scatter 前向，处 g 位）；6.1 按论文记 g（all-gather 前向）/ ḡ（reduce-scatter 前向）——**ḡ 在两章意义相反** | megatron-sp 论文（arXiv:2205.05198）§4.2.2 与 Figure 5 原文："g is all-gather in the forward pass and reduce-scatter in the backward pass. ḡ is reduce-scatter in forward pass and all-gather in backward pass"——6.1 与论文一致，5.2 的 ḡ = 论文的 g、5.2 的 f̄ = 论文的 ḡ | 任选：(a) 5.2 改挂论文命名并注明「即论文的 g/ḡ」；(b) 6.1 §2 变体表下加一行桥接「5.2 记作 ḡ/f̄」——至少消除ḡ 的双义 |
| S2-6 | 5.4 §1（正文 + metadata 行） | recap 句单位标签混用：「每 token 每层 2kd 个元素，bf16 下 2kd×2 bytes。这正是 models 4.5 §7 交付的『每 token 每 MoE 层 2kd bytes……』」——同句中「2kd×2 bytes」与「2kd bytes」相差 2×；models 4.5 §7 末行原文「2kd 字节」自身即与其公式 2·k·d·b（b=2 bytes，得 229 KB）标签不一 | 5.4 §1 原文；models/04-modern-architectures/05-moe.md §7 末行与总结表（公式行为 2kd·b bytes，文字行为「2kd 字节」）；数值（229 KB）两边一致，纯标签问题 | 5.4 引述改写为「每 token 每层 2kd 个元素（bf16 合 4kd bytes）的上界」；metadata 行同步；建议 controller 回修 models 4.5 §7 末行措辞（本批不动 models 章节，登记待办） |
| S2-7 | 6.3 §2 | torchtitan README 引文以原话形式给出但不精确：章文 "A PyTorch native platform for training generative AI models"；实际 README（2026-11-17 tarball）为 "**A PyTorch-native platform for pretraining and post-training generative AI models at scale**" | torchtitan README 第 6 行附近（本批 tarball 实测） | 改为原文（保留连字符与 at scale），或去掉引号改为转述 |

### S3（10 起，记录；下一波统一清零）

| # | 位置 | 事项 |
| --- | --- | --- |
| S3-1 | 5.5 §3 vs 6.2 §1 | 同一 docstring 引述不一：5.5 称 `generate_masked_orthogonal_rank_groups`「docstring 以 tp-pp-dp 为例给出同一公式」，6.2 称其算例为 tp-dp-pp 2-3-4。实测（main 2026-11-17）：该 docstring 两处并存且**源码自身不一致**（prose 例写 order 'tp-pp-dp'，但其式 (1) 的进制链与 [2,3,4] 算例均为 tp-dp-pp）。建议两章统一引 tp-dp-pp 的 2-3-4 算例，并各加一句「docstring prose 与公式存在 order 表述矛盾」的读法注记 |
| S3-2 | 6.2 §5 | 「serialization.py 的 save / load（行 78 起）」：实测 load 在行 78、save 在行 364——「行 78 起」只适用于 load |
| S3-3 | training/references.md | 被引章节回写缺口（本波 8 章引用实况 vs 条目字段）：llama3 缺 5.1/5.2/5.5；zero 缺 5.5/6.2；fsdp 缺 5.5；megatron-lm 缺 5.4/5.5；megatron-cluster 缺 5.5；trl-repo 缺 6.3。lint 不查此字段，归 controller 回写 |
| S3-4 | 6.3 §5 | veRL / OpenRLHF 无 references 条目（writer 存疑 #3，处置合规：未附链接、未写 arXiv ID）；建议下一波 research pass 补录 verl（2409.19256）/ openrlhf（2405.11143）后回填链接 |
| S3-5 | 6.3 §2 | 「代码组织为 torchtitan/{models, components, distributed, config, experiments, rl}」以 {} 全量记法呈现，实测顶层另有 observability/、ops/、quantization/、tools/；建议改「主要目录为…」或补全 |
| S3-6 | 6.1 §2 变体表 | 「同文件还有四个变体族」为选编：main 上另有 `_ReduceScatterToTensorParallelRegion`、`_GatherFromSequenceParallelRegionAsync`、`_AsyncCollectiveHandle` 等；建议「四个」改「主要」或注明选编 |
| S3-7 | 5.4 §1/§3 | 措辞 minor：§1「把入口带宽花在大量不会命中的 token 上」（含糊量词，同句虽有 n·d vs k 的数，建议删「大量」）；§3「完美均衡」（建议「理想均衡」） |
| S3-8 | 跨章 sweep 观察 | 符号 $M$ 复用：5.4/5.5 §2.2 的 $M$ = node-limited 扇出上限（DSv3 论文记号）；4.3/5.5 §5 hybrid 行的 $M$ = 模型元素数（FSDP 论文记号）——各自局部声明、无公式混用，登记观察（同 W1 对局部符号复用的接受口径） |
| S3-9 | 6.3 §1 / references.md | nemo 条目 URL（github.com/NVIDIA/NeMo）经 301 仍可达但已非真身（重定向至 NVIDIA-NeMo/Speech，本批实测 301）——建议下次 research pass 更新 URL 并考虑增补 megatron-bridge 条目（writer 存疑 #4 同议） |
| S3-10 | 5.5 §2.1 ↔ 4.4 §5（跨章） | 「16M」基数不一致：5.5 的 τ = 16M/64 = 262144 按 2²⁴ 读（与 Table 4 的 32 seq × 8192 × 64 = 2²⁴ 精确相符 ✓）；4.4 §5 的「step ≈ 11.0 s」按 16×10⁶ 算（2²⁴ 口径应为 ≈11.6 s）。5.5 引用 4.4 的 11.0 s 时两基数混用（结论不变：44 vs 41.7 GB/s 同判）。建议 5.5 §2.1 加「16M = 2²⁴」注记；4.4 的 11.0 改 11.6 或同样加注（回修 4.4 归 controller 裁定） |

## 2. 强制独立重推导（两项，中间过程全留）

### 2.1（5.4）EP all-to-all 通信量细化 与 models 4.5 §7 上界口径一致性

- 式 (2)：去重后 token t 发往 $D_t$ 张不同卡，$D_t \le \min(k, e)$；dispatch+combine 两程 → $2 D_t d$ 元素，bf16 ×2 bytes ✓（$D_t = k$ 时退回逐对口径，与 models 4.5 §7 上界逐点一致——上界保持）。
- 式 (3)：均衡假设下卡 j 不被命中概率 $(1-1/e)^k$；对 e 张卡用期望线性 → $\mathbb{E}[D] = e(1-(1-1/e)^k)$ ✓。Taylor：$(1-1/e)^k \approx 1 - k/e + \binom{k}{2}/e^2$ → $\mathbb{E}[D] \approx k - \binom{k}{2}/e$ ✓。e=64、k=8：精确 $64(1-(63/64)^8) = 64 \times 0.11838 = 7.576$；二阶展开 8 − 28/64 = 7.5625；章文 7.6 ✓；「只省约 5%」= (8−7.576)/8 = 5.3% ✓。
- 节点粒度：8 节点、$8(1-(7/8)^8) = 8 \times 0.656391 = 5.251 \approx 5.3$ ✓；「省约 1/3」= (8−5.25)/8 = 34.4% ✓。
- 组限定 M=4：$4(1-(3/4)^8) = 4 \times 0.899887 = 3.5995 \approx 3.6$ ✓——**但 §6 的「再降约 1/4」复算为 31.5%（见 S2-1）**。
- 偏斜：式 (4) 定义 $\Rightarrow \sum_j R_j = nk$ ✓（每 (t,i) 对恰落一卡）；同步点下 $T \propto \max_j R_j$ ✓；塌缩极限 $\max_j R_j \to nk$ 即 e 倍劣化 ✓。
- 容量 padding：CF=1.25、完美均衡 → 浪费率 $1 - \bar R/C_e = 1 - 1/1.25 = 20\%$ ✓。

### 2.2（6.2）Distributed Optimizer ↔ DeepSpeed ZeRO 逐项对照账

- 显存（官方 dist_optimizer.md 表逐字核实，main 2026-11-17）：bf16 参数 + fp32 梯度行 18 → 6+12/d：非分布列 18 = 2(param) + 4(grad) + 4(master) + 4(m) + 4(v) ✓；分布列 6+12/d = 2 + 4 + (4+4+4)/d ✓（参数与梯度缓冲全量常驻、优化器侧 ÷d）。章文「换成 bf16 参数 + bf16 梯度口径即 4N + 12N/d」与 4.2 式 (1) $\mathrm{Mem}_1 = 2N + 2N + 12N/d$ 逐项相同 ✓（fp16+fp16 行的 4+16/d 多出的一项是 fp32 main-grad 分片，bf16-grad 口径下无此项——章文换算正确）。
- 通信：每 step 每 rank 梯度 RS ≈ N 元素 + 参数 AG ≈ N 元素 = 2N 元素 ✓；与 DP all-reduce 的 ring 分解（RS+AG 两段，4.1 式 (2) 因子 2(d−1)/d → ≈2N）严格同构 ✓ = ZeRO-1/2（4.2 §2-3 同量论证逐字适用 ✓）；ZeRO-3 为 3N（前向 AG 2N + 反向 AG 2N + 梯度 RS 2N = 6N bytes vs DP 4N bytes，4.2 式 (3) 的 ×1.5）✓。对照表六行逐行与 4.2 推导相符（梯度驻留：ZeRO-1 全量 / ZeRO-2 ÷d 归约后即弃；参数：ZeRO-3 ÷d 用时 gather；更新后 AG：ZeRO-1/2 有、ZeRO-3 无）✓。
- 代码链逐项核实：per-bucket RS（param_and_grad_buffer.py:662，`reduce_scatter_tensor` ✓）、参数 AG（:441 `start_param_sync` → `all_gather_into_tensor` ✓）、main_grad view 机制 ✓、padding 到 d 倍数后连续均分（`shard_buffer` 断言可整除、rank r 取第 r 段 ✓）、`prepare_grads`→`_copy_model_grads_to_main_grads`（optimizer.py:1375 ✓）、`_copy_main_params_to_param_buffer`（distrib_optimizer.py:3305 ✓）、16 元素示例与 "remaining 12 elements are garbage"（dist_optimizer.md 原文 ✓）、`get_grad_stats_parallel_group` docstring（distrib_optimizer.py:1111 ✓ 与章文转述一致）。
- √d 注记（writer 存疑 #2）：各 rank 局部范数 ≈ 全局/√d（分片元素数 1/d、iid 假设），局部裁剪使有效阈值升至 √d 倍量级——推导成立，章文已标「2.4 练习可推」，量级口径可接受 ✓。

## 3. Verification log（抽查项 × 来源 × 结论）

覆盖率说明：5 章正文逐字读；每章全部对外数值/引文/路径断言列入核查集（约 120 项），其中高优先项（⟨verify⟩ 标记 + writer 存疑 + 强制重推导 + 全部行号锚点）100% 覆盖，其余按 ≥70% 抽查。批 1 章节（5.1-5.3）仅做跨章一致性扫读（口径、链接、符号），不重复其逐条事实核查。

### 论文原文（arXiv HTML / ar5iv 全文）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| Llama-3 Table 4 三行（8192/TP8/CP1/PP16/DP64/16M/43%；16384/DP128/41%；16384/CP16/DP8/131072/38%） | arxiv.org/html/2407.21783v3 Table 4 逐字 | ✓ 全中（5.5 §2.1/§4 表；spec 裁定与原文一致） |
| 「43→41 归因于 batch/DP 降」（5.5 §4） | 同上正文："...due to the lower batch size per DP group needed to keep the global tokens per batch constant" | ✓ 原文口径 |
| herd Figure 5 顺序 [TP,CP,PP,DP] 与 GPU0/1/2/4/8 同组实例（5.5 §3 ASCII） | 同上 Figure 5 caption 逐字 | ✓ 逐字一致 |
| 「for model shards we do not reshard after forward computation」（5.5 §2.1 引文） | 同上 §3.3.2 | ✓ 逐字 |
| RoCE/IB「400 Gbps interconnects between GPUs」；Grand Teton 8 GPU/节点 NVLink；466 次中断（47 计划 + 419 意外）、54 天快照、>90% 有效训练时间（5.5 §2 表/§4/§7） | 同上 §3.3.1/§3.3.4 | ✓ 逐字（含小节号归属） |
| herd §3.3.3 是否披露 DP 集合通信拓扑（writer 存疑 #1） | 同上 §3.3.3 全节读 | ✓ 只披露 NCCLX 调优（chunking/优先级），未披露分层拓扑组织——章文「论文未披露」成立；4N≈1.62e12 bytes / 50 GB/s ≈ 32 s vs step≈11 s 的账面矛盾复算成立，章文克制表述（机制分析 + 明标未披露）通过 |
| DSv3：671B/37B、61 层首 3 dense、256+1 expert、d_ff 2048、d 7168、top-8 | arxiv.org/html/2412.19437v1 §2/§4.2 | ✓ 逐字 |
| DSv3 部署：16-way PP × 64-way EP（跨 8 节点）× ZeRO-1、2048 H800、expert 均布 64 卡 | 同上 §3.1/§3.2/§4.2（"uniformly deployed on 64 GPUs belonging to 8 nodes"） | ✓ 逐字（每卡 4 expert 为 256/64 算术 ✓） |
| M=4 组限定、「NVLink 160 GB/s, roughly 3.2 times that of IB (50 GB/s)」、两级转发（IB 到同序号 GPU → NVLink 转发）、20/132 SM、warp specialization、10 通信通道、动态调 warp、平均 3.2 expert/节点、上限 13 = 4×3.2、"near-zero all-to-all"、"without using costly tensor parallelism"、DualPipe Table 2 三行（含 (PP/2−1)(F&B+B−3W)、2×、PP+1）、「divisible by 2」、"1/PP times"、气泡激活不随 m 涨 | 同上 §3.2.1/§3.2.2/§4.2/Table 2 | ✓ 全部逐字命中 |
| 偏置控制器 γ=10⁻³ 前 14.3T、末 500B 关闭、α=10⁻⁴ sequence-wise 兜底；推理侧 32 冗余 expert prefill、约 10 分钟调整 | 同上 §4.2/§3.4 | ✓ 逐字 |
| 1:1 复算（5.4 §6）：通信 2×4×7168×2=114,688 B；计算 6×8×3×7168×2048=2.114e9 FLOPs；比 5.43e-5 vs 机器平衡 50e9/989e12=5.06e-5 → 1.07≈1.1 | 算术（本报告 §2） | 数值 ✓；**口径非对称（前向通信 ÷ 全程计算）→ S2-3** |
| K2：1.04T/32.6B/384 expert/top-8/61 层；PP16(virtual)×EP16×ZeRO-1；32 节点倍数弹性；6 TB/256 卡；每卡全状态约 30 GB；拒 DualPipe（参数梯度显存翻倍、1T 太贵）；interleaved 1F1B 加 warmup 藏 all-to-all；weight-grad 解耦与 PP 通信并行；「smallest feasible EP=16 relaxes expert-balance constraints」；8×400 Gbps RoCE/节点 | arxiv.org/html/2507.20534v1 §2.4.1-2.4.3 与 Table 2 | ✓ 全部逐字命中（「未提及 TP/CP」成立：§2.4.2 只列 PP/EP/ZeRO-1——writer 存疑 #3 的措辞正确） |
| OLMo 2：Jupiter 每节点 8×400 Gbps IB；正文零 FSDP/ZeRO 字样 | arxiv.org/html/2501.00656v2 §6.1.1 与全文 grep（FSDP/HSDP/ZeRO 均 0 命中） | ✓（5.5「以 repo 为一手来源」的口径注记必要且正确） |
| GShard：容量 O(N/E)、双 expert 满员 → 零向量经 residual 穿层、random routing 按 g₂ 概率派第二 expert | ar5iv 2006.16668 §2.2 | ✓ 逐字 |
| megatron-sp：g/ḡ 定义（g = all-gather 前向 / reduce-scatter 反向）；7.7→7.2 ms；「RS+AG 合起来比单次 all-reduce 慢」；激活账 sbh(10+24/t+5as/(ht)) 与 sbh(34+5as/h)/t | ar5iv 2205.05198 §4.2.1/§4.2.2/Figure 5/§6.2/Table 2/Table 4 | ✓ 逐字（并据以确立 S2-5 的命名冲突） |
| ZeRO-3 反向二次 gather 机制（6.2 练习 4 所依） | 4.2 §4.1 章文 | ✓ 跨章一致 |

### 代码（tarball 快照，均 2026-11-17）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| mappings.py：`_CopyToModelParallelRegion`（fwd 恒等 / bwd `_reduce`）与 `_ReduceFromModelParallelRegion`（fwd `_reduce` / bwd 直通）逐字；`_reduce` 的 group-size-1 短路；公开包装两函数；变体族 5 行 | NVIDIA/Megatron-LM main tarball（行 26/205/225/244/264/284/304/359/388/428/496/502） | ✓ 逐字一致（归属 mappings.py ✓） |
| layers.py：`allreduce_dgrad = (world_size > 1 and not sequence_parallel and not disable_grad_reduce)`；ColumnParallelLinear.forward 融合分支跳过独立 f；backward 内 `all_reduce(grad_input, async_op=True)` + CUDA_DEVICE_MAX_CONNECTIONS=1 注释；SP 时 forward 内 `dist_all_gather_func(all_gather_buffer, input, ...)`；`_forward_impl`；`gather_tensor_parallel_output`；RowParallelLinear `input_is_parallel` / SP 换 `reduce_scatter_to_sequence_parallel_region` / bias 在 g 后加 / skip_bias_add；VocabParallelEmbedding 掩码-查表-清零-all-reduce 四步、`deterministic_mode`（"F.embedding currently has a non-deterministic backward function"）、`EmbeddingWithGradAccumulationFusion` | 同 tarball layers.py（行 244/287/308/495/500/506/1319/1347/1406-1415/1435/1461/1731/1735/1765/1769/1771；842/861/874 注释） | ✓ 全部命中 |
| mlp.py：GLU 时 `fc1_stride = 2`、源码注释 "each TP rank stores interleaved [gate, up] portions"；fc1 `gather_output=False` | 同 tarball mlp.py 行 221-244 | ✓ 逐字 |
| attention.py：linear_qkv `gather_output=False`（SelfAttention）、linear_proj `input_is_parallel=True`；GQA 两分支（num_query_groups < tp 时每 rank 1 KV 组 + q_heads/kv_groups 个 Q head，KV 复制）；`apply_rotary_pos_emb(..., cp_group=...)` 在 linear_qkv 后、core attention 前 | 同 tarball attention.py 行 368-379/432-438/1647-1683/1932 | ✓ 全部命中 |
| random.py：s_dp = s₀、s_tp = s₀+2718+i_tp（"2718 is just for fun"）、s_ep = s₀+1024+100·i_ep+i_etp；`_initialize_affine_weight_gpu` 的 `get_cuda_rng_tracker().fork()`；CPU 路径 `torch.split` + `weight_list[rank::world_size]` | 同 tarball random.py 行 556-618；layers.py 行 236-291 | ✓ 式 (6) 逐项一致 |
| ops/norm/torch_norm.py：WrappedTorchNorm 对 sequence_parallel 的 assert 拒绝；RMSNorm 要求 torch ≥ 2.4.0a0 | 同 tarball 行 35/58/66-69 | ✓ 逐字 |
| parallel_state.py：order 默认 "tp-cp-ep-dp-pp"（行 672）；model_size = t·c·p·(gtp) 与整除报错、data_parallel_size 推出（行 907-919）；expert_data_parallel_size（行 984）；两 generator PP 组相等断言；`ep == 1 or cp == 1`（行 536）；RankGenerator 行 522 / get_ranks 行 589 / generate_masked 行 301 / default_embedding_ranks 行 615；decompose/前缀积机制；两 docstring 算例（tp-dp-pp 2-3-4 的 DP 组 {0,2,4}…{19,21,23}；8 GPU tp4 cp2 的 CP 组 {0,4}…）；piggyback 原话；gtp_remat「NOT activation recomputation」；create_gloo_process_groups 默认 True；num_distributed_optimizer_instances 行 668；模块 docstring 弃用声明；process_groups_config.py 的 ProcessGroupCollection | 同 tarball parallel_state.py / process_groups_config.py | ✓ 全部命中（行号 12/12 全中；docstring 内部矛盾另记 S3-1） |
| schedules.py：get_forward_backward_func 行 53、三分支行 161-167、no_pipelining 735 / with_interleaving 1013 / without_interleaving 2169、forward_step 404、backward_step 551、custom_backward 204、get_schedule_table 983、no_sync_func disable/enable 行 2272 起、clear_embedding_activation_buffer 888、finish_embedding_wgrad_compute 907、SP 形状 docstring 注记；P2PCommunicator 行 148 | 同 tarball | ✓ 全部命中（12/12 行号全中） |
| optimizer/：DistributedOptimizer 行 202 及 docstring、checkpoint_fully_reshardable_formats 行 216（三值 ✓）、_build_model_gbuf_param_range_map 行 281（"each created range references a shard (or subset) of the full parameter" ✓）、get_grad_stats_parallel_group 行 1111、sharded_state_dict 行 1774、_copy_main_params_to_param_buffer 行 3305；MixedPrecisionOptimizer.step 行 1076；shard_buffer 整除断言与连续分段；DistributedDataParallelConfig：use_distributed_optimizer 注释（"reduce-scatter … otherwise all-reduce" ✓）、bucket_size 默认 max(4×10⁷, 10⁶×dp_size) ✓；dist_optimizer.md 三行显存表与 16 元素示例逐字 ✓ | 同 tarball + docs/user-guide/features/dist_optimizer.md | ✓ 全部命中（10/10 行号全中） |
| dist_checkpointing/：ShardedTensor（mapping.py 行 64）、save/load（serialization.py 行 364/78）、get_default_save_sharded_strategy 行 67（backend="torch_dist" ✓）、load_sharded_metadata 行 286 ✓；"Distributed checkpoint saving, loading, and resharding"（repo README 行 73 树注 ✓） | 同 tarball | ✓（S3-2：save 行号 364） |
| moe/：router.py / token_dispatcher.py / experts.py / moe_layer.py 均在；MoEAlltoAllTokenDispatcher（行 390）/ MoEAllGatherTokenDispatcher（行 248）/ MoEFlexTokenDispatcher（行 2226，含 DeepEP/HybridEP 后端引用 ✓）；dispatcher 的 `all_to_all` 经 `megatron.core.tensor_parallel` 导入——**即 mappings.py 的 `_AllToAll`**（writer 存疑 #4 的「EP dispatch/combine 底层」判定成立 ✓）；TEGroupedMLP/grouped_linear（experts.py ✓） | 同 tarball | ✓ 全部命中 |
| OLMo-core：OLMo2-7B.py 的 `dp_config = HSDP`（bf16 参数、fp32 归约、blocks 包装）✓；OLMo2-32B.py 同为 HSDP、FSDP 注释备选 ✓；distributed/parallel/ 五文件（data/tensor/pipeline/context/expert）✓；包内 data/nn/optim/distributed/train/eval/launch + float8 + mxfp8_config + kernels + ops + generate ✓ | allenai/OLMo-core main tarball | ✓ 全部命中 |
| torchtitan：README 特性清单逐项（FSDP2/HSDP/DDP、async TP、EP、pipelining 的 1F1B/交错/GPipe、CP 的 all-gather KV 与 Ulysses、checkpointing full/selective/regional、FA3/4 + FlexAttention + varlen、BF16 optimizer states、FSDP CPU offload、Grain packing/mixing/可 checkpoint 状态、DCP 异步 save、HF↔DCP 转换）✓；"Models declare how they shard" ✓；distributed/ 六文件 ✓；parallelism_context.py 导出 ParallelismContext 与 DistributedTopology ✓；components/ 六件套 ✓；train.py/trainer.py/training_engine.py ✓；rl/ 目录存在 ✓；BSD ✓ | pytorch/torchtitan main tarball | ✓ 命中（**README 标语引文不精确 → S2-7**；顶层目录遗漏 → S3-5） |

### Repo 现状（GitHub API，2026-11-17 实测）

| 抽查项 | 结论 |
| --- | --- |
| github.com/NVIDIA/NeMo → 301 → NVIDIA-NeMo/Speech | ✓（curl -sI 实测；writer 存疑 #4 属实） |
| NVIDIA-NeMo 子库 RL/Curator/Evaluator/Automodel/Guardrails/Speech/Export-Deploy | ✓ 7/7 存在（6.3 §1 职能读法成立） |
| NVIDIA-NeMo/Megatron-Bridge 自述 "Training library for Megatron-based models with bidirectional Hugging Face conversion capability" | ✓ 逐字（pushed_at 2026-11-16，活跃） |
| deepspeedai/Megatron-DeepSpeed pushed_at = 2025-08-14；deepspeedai/DeepSpeed pushed_at = 2026-11-16 | ✓（writer 存疑 #5 的口径注记可接受） |

### 教学代码实跑

| 项 | 结论 |
| --- | --- |
| 6.1 §6 教学实现对拍代码（torch 2.8.0，float64） | ✓ 四个 assert 全过（前向/输入梯度/两权重梯度与单机参考一致） |
| 5.4 §7 `ep_moe_forward` 语义（与逐 token 朴素参考对照） | ✓ 一致（max err ~2e-7，float32 噪声） |

### 跨章一致性 sweep（全 8 章 + 回链章）

| 项 | 结论 |
| --- | --- |
| 符号约定（spec §4.3）：d/t/p/e/c 全波一致；混场章 5.5 无 $G$ 表 DP（grep 零命中）；world 展开写法 d·t·c·p·e（5.4 §5）✓ | ✓ |
| 1.2 ↔ 5.1（matmul 行切/列切母式）双向 | ✓（1.2 §5.1 前瞻 + 5.1 前置行） |
| 1.4 ↔ 6.2（checkpointing 两档 ↔ recompute_granularity 配置面）双向 | ✓（1.4 §5「实现细节归 5.1/6.2」+ 6.2 §3 回链） |
| 4.x ↔ 5.5 总账：4.4 §4 表（ZeRO-2 档 ≈4N）= 5.5 §2.1 DP 行 ✓；4.1 式 (2) ring 因子 2(ν−1)/ν = 5.5 §2.1 用法 ✓；4.3 式 (1) hybrid 2M/d_s 与 5.5 §5 行一致（4.3 已声明 W→d、G→d_s 桥）✓；4.4「跨家族组合总账归 5.5」双向 ✓ | ✓（16M 基数不一致另记 S3-10） |
| 5.x ↔ 6.x：5.4 §5/5.5 §3 → 6.2 group 树走读 ✓；6.1 ↔ 5.1/5.2 公式-代码对照 ✓；5.3 ↔ 5.4（DualPipe 重叠 all-to-all ↔ 内核细节）双向 ✓ | ✓ |
| TP 每层通信口径：5.1（每层 4 次、消息 2bsh、前向 4bsh 反向翻倍）= 5.5 §2.1（每 token 每层 8h bytes、前向 2 + 反向 2）✓；5.3 式 (11) PP 发送口径（每 step 4mbTh 发送）与 5.5 PP 行 8hτ（消息体过链含收发）各自声明、不矛盾 ✓ | ✓ |
| 气泡公式：(p−1)/(m+p−1) 在 5.3 式 (5) 与 5.5 §5 表一致 ✓；interleaved「再除 v」两章同向（5.3 式 (8) 用 (p−1)/(v·m) 口径、5.5 相对自身行值除 v——分母约定异、方向一致，两章各自成立） | ✓（登记观察） |
| ZeRO/dist-opt 账：4.2 式 (1)(3) ↔ 6.2 §4 表逐项一致（本报告 §2.2）✓；6.3 §2 对照表「dist-opt = ZeRO-1 族 / FSDP2 = ZeRO-3 语义」与 6.2/4.3 一致 ✓ | ✓ |
| models 4.5 接口：容量式 (11)、§7 上界与扇出、§4 诊断口径、§2.2/§3.2 均衡机制回链 | ✓（「2kd bytes」标签问题 → S2-6） |
| SP 转换算子命名 | ✗ → S2-5 |
| EP/Ulysses 行口径 | ✗ → S2-2 / S2-4 |
| references.md 被引章节回写 | ✗ → S3-3 |

### Writer 存疑项复核结论（13 项全裁）

5.5/6.1 writer（task-17）6 项：#1 DP 4N 账面矛盾——算术复核成立、论文未披露拓扑属实、章文措辞克制 → 通过不修；#2 DSv3 d=128/d_e=2 本书换算已显式标注 → 通过；#3 K2「未提及」措辞准确（§2.4.2 只列三项）→ 通过；#4 `_AllToAll` 确为 dispatcher 底层（经 tensor_parallel 包导入链核实）→ 断言成立；#5 模块树选编已声明 → 通过；#6 消息体口径注记在文 → 通过。

6.2/6.3 writer（task-18）7 项：#1 dist-opt「显存 ZeRO-1 / 通信 ZeRO-2 同形」经官方文档表 + 代码逐项核实成立 → 通过；#2 √d 量级推导成立、标注充分 → 通过；#3 veRL/OpenRLHF 无条目——处置合规，转 S3-4 跟踪；#4 NeMo 重定向属实 → 转 S3-9；#5 pushed_at 口径注记 → 通过；#6 行号漂移以函数名锚定兜底（本批 30+ 行号当日全中）→ 通过；#7 torchtitan 顶层结构经本批 tarball 独立核实 → 通过（遗漏四级目录转 S3-5）。

## 4. 教学审查摘要

推导断步：未发现硬性断步——5.4 式 (1)-(4) 符号表齐全、假设显式（均衡独立性已声明为近似）；5.5 §2.1 算例数字链完整；6.2 式 (1)(2) 与 16 卡实例自洽（练习 1 的 rank 23 五维坐标 (3,1,0,1) 与四个组 {20-23}/{19,23}/{23,31}/{7,23} 经独立取模复算全中）；6.2 练习 2 的 order 反例存在性成立（dp-tp-pp 下 TP 组 {0,2} 等非连续）。前置缺口：无新增（5.4 依赖 models 4.5 §2-3 路由定义、4.1 集合通信账，均已成章）。循环解释：无。可改进点已并入 S2/S3（口径混用对读者的可复现性影响最大——5.5 §5 表是其核心交付物，尤应统一口径）。

## 5. 规范审查摘要

rubric A 类：零命中（禁令词、空洞形容词、历史叙事均无；NeMo 组织变动以「核实于 2026-11-17」的事实登记口径呈现，非年代叙事 ✓）。B 类：S3-7 两处边缘措辞；各节均以事实/公式/问题开场 ✓；术语无漂移 ✓。模板/证据等级/练习数（5.4:4、5.5:4、6.1:4、6.2:4、6.3:3）/篇幅/ASCII 图：全达标。

## 6. 放行结论（批 2 初审）

**不放行。** S1 = 0；S2 = 7（S2-1 数值、S2-2/S2-3/S2-4 通信口径族、S2-5 跨章命名、S2-6 单位标签、S2-7 引文），按协议 S2 必须修复后进复审；S3 = 10 随下一波清零（S3-3/S3-4/S3-9 归 controller / research pass；models 4.5 §7 与 4.4 §5 的回修建议归 controller 裁定，本批不动两章文件）。

## 7. 修复复审记录（2026-11-17 复审轮）

方式：五章当前全文重读（非 diff）+ 修复触及处新问题扫描 + 10 个文件 lint 重跑。

### 逐条判定

| finding | 判定 | 当前文本证据 |
| --- | --- | --- |
| S2-1（「再降约 1/4」） | **ADDRESSED** | 5.4 §6 现作「再降约 1/3（5.25 → 3.6 个节点，降幅约 31%）」——与复审复算 31.5% 一致 ✓ |
| S2-2（EP 全程口径，两侧） | **ADDRESSED** | 5.4 §5：「每 token 每 MoE 层全程 ≤ 8kd bytes（前向 4kd：dispatch+combine 各 kd 个元素 ×2 bytes；反向对称，§7）」；5.5 §5 EP 行：「≤ 8kh·L_moe,p·τ（全程口径：前向 dispatch + combine 共 2kh 元素/token/层，反向对称再一倍；bf16）」。算术复核：2kd 元素 ×2B = 4kd（前向）×2 = 8kd（全程）✓；5.5 行 2kh 元素 ×2B ×2 = 8kh ✓；与 TP 行 8h（前向 2+反向 2）口径一致 ✓；5.4 总结表行同步 ✓ |
| S2-3（1:1 复算非对称 pairing） | **ADDRESSED** | 5.4 §6 新增口径注记：「分子只是前向一程的通信、分母却是全程……是非对称 pairing；通信按反向对称翻倍时比值约 2.2，……本书复算只支撑『同量级』结论」；结论句改为「暴露时间与计算同量级」；顺带修正原「未计入分子」→「未计入分母」（更正确）。复核：1.07×2 = 2.14 ≈ 2.2 ✓ |
| S2-4（Ulysses 行 2×） | **ADDRESSED** | 5.5 §5 表现作「16h·L_p·(c−1)/c·τ（全程口径：前向 8h = QKV 换维 6h + 输出换回 2h，反向同量）」。算术复核：前向 (6+2)bsh(c−1)/c² ÷ 本地 token bs/c = 8h(c−1)/c ✓ 与 5.2 式 (5) 分解一致；×2 反向 = 16h(c−1)/c ✓ |
| S2-5（ḡ/f̄ vs g/ḡ 命名冲突） | **ADDRESSED** | 5.2 式 (1) 及全文改挂论文原记号：g = all-gather 前向 / reduce-scatter 反向（占 5.1 的 f 位）、ḡ = reduce-scatter 前向 / all-gather 反向（占 5.1 的 g 位），并显式点破「此 g 非 5.1 的 g」的同名易混；行 19/24/26/35/126/135 全部一致；6.1 维持论文记号 g/ḡ 不动——两章与 megatron-sp 论文三方一致 ✓ |
| S2-6（「2kd bytes」标签） | **ADDRESSED** | 5.4 metadata 行与 §1 改「每 token 每层 2kd 个元素（bf16 合 4kd bytes）」；models 4.5 §7 末行已做最小修订为「2kd 个元素（bf16 合 4kd 字节）」——两侧措辞一致，229 KB 数值不变 ✓ |
| S2-7（torchtitan README 引文） | **ADDRESSED** | 6.3 §2 现引原文逐字 "A PyTorch-native platform for pretraining and post-training generative AI models at scale"（与 2026-11-17 tarball 逐字一致）✓ |
| S3-1（docstring 算例引述） | ADDRESSED | 5.5 §3 改引 tp-dp-pp 的 [2,3,4] 算例并加注「该 docstring 的 prose 举例写 order 'tp-pp-dp'，而其式 (1) 的进制链与算例均为 tp-dp-pp，源码自身表述不一致，以公式与算例为准」——与 6.2 §1 一致 ✓ |
| S3-2（save/load 行号） | ADDRESSED | 6.2 §5 现作「load（行 78）与 save（行 364）」✓（与源码一致） |
| S3-3（references 回写） | ADDRESSED | llama3 +5.1/5.2/5.5、zero +5.5/6.2、fsdp +5.5、megatron-lm +5.4/5.5、megatron-cluster +5.5、trl-repo +6.3 全部回补 ✓ |
| S3-4（verl/openrlhf 条目） | ADDRESSED（残留登记） | references.md 已补录 verl（2409.19256）与 openrlhf（2405.11143）✓；残留：6.3 §5 正文未挂新条目链接（cosmetic，下一波随手清零） |
| S3-5（torchtitan 目录枚举） | ADDRESSED | 6.3 §2 现列 10 个目录的「完整清单」（models/components/distributed/config/experiments/rl/observability/ops/quantization/tools）——与 tarball 实测一致 ✓ |
| S3-6（变体表选编措辞） | ADDRESSED | 6.1 §2 改「一族变体算子……下表为主要选编」并点名从略变体（_ReduceScatterToTensorParallelRegion、_GatherFromSequenceParallelRegionAsync 等）✓ |
| S3-7（措辞） | ADDRESSED | 5.4 §1「大量」已删、§3「完美均衡」改「理想均衡」✓ |
| S3-8（M 符号复用） | 接受登记 | 维持观察登记，不修 |
| S3-9（nemo 条目） | ADDRESSED | nemo 条目 URL 更新为 NVIDIA-NeMo/Speech 并加重组注记；megatron-bridge 条目补录 ✓ |
| S3-10（16M 基数） | ADDRESSED（见 N1） | 5.5 §2.1 加「16M 为二进制口径 2^24，恰为 batch/DP 32 条 × seq 8192 × d 64 之积」✓；4.4 §5 改「约 11.6 s」并注明 2^24 基数 ✓——但见 N1 |

### 修复引入的新问题（只查修复触及处）

- **N1（S3，新登记）**：4.4 §5 的 step 时间由 11.0 s 改为 11.6 s 后，5.5 §2.1 未跟进：行 25 仍写「约 11.0 s（4.4 §5 的数字链）」（引用断裂——4.4 现作 11.6）；行 33 的 25 GB/s 与 44 GB/s、行 43 的「step ≈ 11 s」、§5 回代段「TP ≈ 44 GB/s」、练习 2 的「11 s」同基于旧值。按 11.6 s 重算：23.7 GB/s 消息体、ring 口径 41.5 GB/s（占 50 GB/s 的 83%）、PP 3.0 GB/s——全部结论方向不变。建议：5.5 §2.1 行 25 改「约 11.6 s」并重算三个派生数（或加注「按 11 s 估」）。随下一波 S3 清零，不阻塞放行。
- 其余修复触及处（5.2 式 (1) 及全文命名、5.4 §1/§5/§6/总结表、5.5 §2.1/§3/§5 表、6.1 §2、6.2 §5、6.3 §2、references.md、models 4.5 §7、4.4 §5）逐处复读，未发现其他新问题；5.2 式号序列 (1)-(5) 完整无碰撞。

### lint 重跑

training 全 8 章（5.1-5.5、6.1-6.3）+ 修复触及的 4.4 与 models 4.5：10/10 LINT OK。

### 复审结论

S2 7/7 ADDRESSED、S3 10/10 关闭（9 修复 + 1 接受登记）；新登记 N1（S3）与 S3-4 残留链接回填随下一波清零。**批 2 五章（5.4/5.5/6.1/6.2/6.3）放行。**
