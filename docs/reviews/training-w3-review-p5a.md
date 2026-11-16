# training W3 批 1（5.1-5.3）对抗审查报告

日期：2026-11-17。Evaluator：独立 fresh-context（未读写作上下文，先读 task-19 brief）。立场：假设有错，逐项回源。
审查对象：`training/05-model-parallelism/01-tensor-parallelism.md`（5.1）、`02-sequence-and-context-parallelism.md`（5.2）、`03-pipeline-parallelism.md`（5.3）。
一手来源：arXiv abs 页 + ar5iv 全文（curl 直拉，math alttext 保留 LaTeX 后核对原文句子）、NVIDIA/Megatron-LM repo main 分支文件（经 jsdelivr CDN 镜像拉取，`api.github.com` 当日限流、`raw.githubusercontent.com` 被重置——镜像内容逐项与章节断言对照）、论文 SVG/PNG 原图（Figure 2/3/4 像素级重建，见 verification log #12/#13）。

## 结论

**不放行**。S1×1（striped-attention 的 arXiv ID 张冠李戴到一篇宇宙学论文）、S2×5、S3×5。5.1/5.3 的主体推导与数字链经独立重推导与回源全部成立；5.2 §4（striped 均衡）的机制归属与参数化需要实质改写。含 S1 的 5.2 修复后需复审。

## 机械检查（lint）

| 章节 | 命令 | 结果 |
| --- | --- | --- |
| 5.1 | `bash scripts/lint-chapter.sh training/05-model-parallelism/01-tensor-parallelism.md` | LINT OK |
| 5.2 | `bash scripts/lint-chapter.sh training/05-model-parallelism/02-sequence-and-context-parallelism.md` | LINT OK（注：lint 的 arXiv 回查是「references.md 存在该 ID」的机械 grep，查不出 ID↔论文错配——S1-1 正是从 lint 盲区漏过的） |
| 5.3 | `bash scripts/lint-chapter.sh training/05-model-parallelism/03-pipeline-parallelism.md` | LINT OK |

补充手工扫描（lint 不覆盖项）：三章无 `\{`/`\,`/`\;`/`\!`/`\%`/`\#` 反转义陷阱写法（实际命中均为白名单的 `\mid`/`\thinspace`/`\hspace`）；行内公式无裸 `*`、无 `<`+字母；rubric A 类禁令词全量 grep 零命中；「显著」唯一命中处（5.3 §6「这份 ×2 不显著」）是 V3 论文原意的转述且有归属，按 rubric A.2 豁免不计。

## Findings

### S1（事实性致命）

**S1-1. striped-attention 的 arXiv ID 错配到无关论文。**
位置：5.2 metadata 行、§4 首段、总结表、参考条；`training/references.md` 第 115 行 `striped-attention` 条；上游源头在 spec（`2026-11-16-training-dimension-design.md` 第 258/261/499 行，且 261 行标注「spec 审查 B 核实」——该核实记录为假阳性）。
断言：「Striped Attention: Faster Ring Attention for Causal Transformers，arXiv:2405.13119」。
证据：arXiv abs 页直拉——2405.13119 实为《Cosmology from Point Clouds with Dark Matter Halos from the Quijote Simulations》（Chatterjee & Villaescusa-Navarro），与本文毫不相干。真正的 Striped Attention 是 **arXiv:2311.09431**（William Brandon、Aniruddha Nrusimha 等，v1 2023-11-14，标题逐字相同，官方代码 exists-forall/striped_attention，论文首页自引）。标题与作者写对了、ID 错了——典型的「凭记忆写 ID」事故，且 spec 审查 B 的「核实」背书是假的。
建议修正：ID 全部改为 2311.09431（5.2 四处 + references.md + spec 三处）；作者年随之改 2023（见 S3-2）；并在修复流程上追加一条：references.md 的 arXiv 条目写入时必须以 abs 页标题双向核对（lint 做不到）。

### S2（实质问题）

**S2-1. 5.2 §4 把 zigzag 的机制安到 Striped Attention 头上，且参数化与不变量断言自相矛盾。**
位置：5.2 §4 全节（式 (3)(4)、「首末配对」「倾斜比恒等于 1」「驻留显存不变」「轮转收发同式 (2)」「奇数 c 的中间块自成一对」、练习 2(b)(c)）。
断言：Striped Attention 的做法是「rank r 持块 r 与块 c−1−r 的首末配对」，每 rank 工作量恒为 (c+1)u、倾斜比恒等于 1、显存与通信不变。
证据（2311.09431 原文 + 其 Figure 1 的 SVG 像素级重建）：论文的划分是**按 token 对设备数取余的等距条带**——原文：「Rather than partitioning the tokens into contiguous blocks, we partition them into sets of evenly-spaced stripes based on their residues modulo the device count N. For example, given a device count of 4, device 0 would own tokens 0,4,8,…」；Figure 1 的 stripe permutation 重建为 {0,4,8,12}/{1,5,9,13}/{2,6,10,14}/{3,7,11,15}（n=16、N=4），即步长 N 的跨步取样，不是首末配对（首末配对会给 {0,15}）。论文的均衡结论也是**近似**口径（"approximately half of the query/key interactions on each device will be inhibited"、"almost entirely eliminates the workload imbalance"），不存在「恒等于 (c+1)u、倾斜比恒等于 1」的精确结论。章节真正推导的「块 r 与块 c−1−r」是工程界通称 **zigzag** 的方案（Megatron-Core CP 的实现口径：2c 个半块、rank r 持 r 与 2c−1−r）。
参数化自相矛盾处：§3 把序列切成 c 个连续块（每块 s/c），§4 若每 rank 持其中两个整块，则每 rank Q 驻留从 s/c 翻倍到 2s/c、每步 K/V 收发同样翻倍——「驻留显存不变（仍是两块 Q）、轮转收发同式 (2)」在该记号下不成立；自洽的版本是切成 **2c 个 s/(2c) 小块**、rank r 持 {r, 2c−1−r}，此时工作量 (r+1)u′+(2c−r)u′=(2c+1)u′ 恒定、显存与式 (2) 通信账不变，且 2c 恒偶、「奇数 c 中间块自成一对」的问题整个消失（它纯属 c 块参数化的产物）。
建议修正：§4 改写为——(a) 机制归属改为「Striped Attention（2311.09431）：按位置 mod c 的等距条带，近似均衡（论文原文口径）」；(b) 精确均衡的首末配对单独命名 zigzag（实现侧通行方案，可引 Megatron-Core CP 实现为教学参照），式 (4) 改按 2c 半块记号重立（(2c+1)u′）；(c) 「显存不变 / 同式 (2)」两条断言只在 2c 半块版本下成立，随改写一并落地；练习 2 同步改。式 (3)（连续切分 2c/(c+1) 倾斜比）本身数学无误，保留。

**S2-2. 5.3 §3 的 1F1B 时序图「与 Megatron 论文 Figure 4 同画法」为伪，且图的 B 级联列对齐违反式 (1)。**
位置：5.3 §3 的 ASCII 时序图及其括号声明（task-16 存疑项 1）。
证据：把 2104.04473 的 Figure 4（interleaved_schedule.svg）逐像素重建——该图反向块宽恰为前向块 2 倍（t_b=2t_f，与 Figure 3 注一致），且 **B1 严格逐级错开**：dev4 的 B1 占 x∈[182.6,216.3]、dev3 [216.3,250.1]、dev2 [250.1,283.9]、dev1 [283.9,317.5]，恰为无缝级联；dev1 在 B1 前有 6 个单位空格 = (p−1)·t_b（与正文推导一致）。章节 ASCII 图却把四个 stage 的 B1 画在同一列（slot 5）——即使在「拓扑示意」口径下也违反式 (1) 的 B(i,j)←B(i+1,j)（同列 = 同时刻，而依赖要求严格居后）。正文确实声明了「等宽格为拓扑示意」并给出了精确等待推导（这部分教学处理合格），但「与 Megatron 论文 Figure 4 同画法」这句归属断言是假的。
建议修正：删掉「与 Megatron 论文 Figure 4 同画法」；图内把各 stage 的 B 级联逐段错开一格（stage4 B1 在 slot 5、stage3 在 6、…stage1 在 8，等待格以 · 标出），或明确改写为「依赖结构压缩示意（B 级联等待压掉，精确时序见模拟器）」。对照之下 §2 的 GPipe 图 B 级联是错开的、与 GPipe Figure 2c（已下载 PNG 核对：等宽格、逐级错开）画法一致，无问题。

**S2-3. 5.2 §2 引用 5.1 的「RNG 同步注记」——5.1 全文不存在该注记（断链引用）。**
位置：5.2 §2 末「5.1 §3 的 RNG 同步注记（TP 组内共享种子）在这里是同一份纪律」。
证据：对 5.1 全文 grep RNG/种子/seed 零命中；Megatron 论文确有 Appendix B.2「Model Parallel Random Number Generation」，但 5.1 §3 的两条实现注记是「求和语义」与「反向同步步进」，没有 RNG。
建议修正：二选一——在 5.1 §3 补上该注记（TP 组内 dropout 等 RNG 按分片共享种子/状态的事实，引 Megatron 论文附录 B.2），或把 5.2 该句改为直接陈述（不挂 5.1 回指）。

**S2-4. 5.3 对 2104.04473 的三处小节号指引全部指错。**
位置与证据（章节结构已从 ar5iv 全文提取核对）：
(a) §1「各 b·T·h 个元素（…§2.1 的口径）」——该句（"the total amount of communication … between every pair of consecutive devices … for each microbatch is bsh"）实际在 **§3.2**（Tensor and Pipeline Model Parallelism）；§2.1 是 Data Parallelism。
(b) §8「Megatron 集群论文则明确只考虑均匀切分…（arXiv:2104.04473 §2.1）」——"We do not consider more asymmetric model architectures…" 实际在 **§2.2**（Pipeline Model Parallelism）。
(c) §4「（原文 §3.4 / Figure 9；实测…11%）」——scatter/gather 机制在 **§4.1**（Communication Optimizations，Figure 9 正确），11% 数字在 **§5.7**（Figure 18）；§3.4 是 Microbatch Size。
建议修正：三处小节号改为 §3.2 / §2.2 / §4.1（11% 可补引 §5.7）。内容性断言本身全部属实（见 verification log #8），只是路标错了。

**S2-5. 5.1 §7「恰充满一台 DGX-2H 的 8 卡 NVLink 域」——DGX-2H 是 16 卡机。**
位置：5.1 §7。
证据：Megatron 论文原文「All of our experiments use up to 32 DGX-2H servers (a total of 512 Tesla V100 SXM3 32GB GPUs)」——512/32 = 16 卡/台；DGX-2/2H 公开规格即 16×V100（NVSwitch 全互连域 16 卡，不存在「8 卡 NVLink 域」）。8 路 TP × 64 路 DP = 512 的数字本身正确（Figure 1 注口径核实），每台 16 卡机装两个 TP 组。
建议修正：改为「8 路 TP 收在单台 DGX-2H（16×V100 NVSwitch 域）内」一类表述。论证方向（TP 收节点内）不受影响。

### S3（minor）

- **S3-1**：5.2 §2 的算子命名 ḡ/f̄ 与 SP 论文互换（论文 Figure 5/6：入口转换子 = g（前向 all-gather/反向 reduce-scatter）、出口 = ḡ；章节写作 ḡ/f̄）。章节自定义自洽、mappings.py 类名对照正确，但与原文对照的读者会错位；建议在式 (1) 符号行加一句「论文记号为 g/ḡ，本书换记为 ḡ/f̄」。
- **S3-2**：5.2 引用年「[Brandon et al., 2024]」——2311.09431 v1 为 2023-11-14；references.md 其余条目按 v1 年（Ring 2310.01889 → Liu et al., 2023）。随 S1-1 一并改 2023。
- **S3-3**：5.3 §5 括注「与 GPipe/Megatron 两文的等时长画法同源」——Megatron Figure 3/4 实际按 t_b=2t_f 块宽绘制（像素重建确认），等时长画法只有 GPipe Figure 2c；改为「与 GPipe 的等时长画法同源」。
- **S3-4**：5.3 §9「V3 的 DualPipe 未随技术报告开源调度器本体」——字面为真（TR 2024-12 未含），但 2026 当下具有误导性：deepseek-ai/DualPipe repo 已公开（2025 年开源周，核到 repo 页标题「A bidirectional pipeline parallelism algorithm for computation-communication overlap in DeepSeek V3/R1 training」）。建议补半句「调度器实现后于 2025 年单独开源（deepseek-ai/DualPipe）」。
- **S3-5**：references.md 的 `megatron-lm` 条「被引章节」字段缺 5.2/5.3（当前为 4.1, 5.1, 6.1, 6.2）；写作方已按纪律标注「留待 research pass 回写」，此处登记督办。

## 强制独立重推导（中间过程逐步写出）

### 5.1-A：f/g 恒等算子的 autograd 语义

设定 Y = XA，X∈R^{m×k}，A∈R^{k×n}，1.2 式 (2)（∂L/∂A = X^T G、∂L/∂X = G A^T，已从 1.2 文件第 22 行核对原文）为母式。

列切 A=[A_1|…|A_t]：前向 Y_p = XA_p 本地、Y 逻辑拼接，无通信。反向：∂L/∂A_p = X^T G_p 本地；∂L/∂X = GA^T，按 A 的列分块展开 GA^T = Σ_p G_p A_p^T——第 p 项只含 rank p 的本地量，跨卡求和即一次 all-reduce（消息 [m,k] = 2mk bytes bf16）。**f 的反向 all-reduce 正是这个 Σ**：f 前向恒等 ⇒ 列切层输入是未动过的 X；链式上 ∂L/∂X 在 f 的 backward 处收口为全量。

行切 A^T=[A_1^T|…]、X 沿收缩维同步切：Y = Σ_p X_pA_p，前向一次 all-reduce（[m,n] = 2mn bytes）。反向：∂L/∂X_p = GA_p^T、∂L/∂A_p = X_p^T G 皆本地（G 在 all-reduce 之后各卡一致）。**g 前向 = 这个 Σ、反向恒等**（G 不需再动）。

复合链验证（MLP）：前向 g(φ(f(X)W_1^{(p)})W_2^{(p)}) = Σ_p φ(XW_1^{(p)})W_2^{(p)} = φ(XW_1)W_2，与单卡逐项相等（φ 逐位置 ⇒ φ(H) 第 p 段 = φ(H_p)，精确非近似）。反向：g 恒等 ⇒ G 原样入行切反向（全本地）；φ 逐元素本地；W_1 列切的输入梯度部分和由 f 的反向 all-reduce 收口。全链无前向函数改动、无第二个通信点。**与实现核对**：Megatron repo main `megatron/core/tensor_parallel/mappings.py`（jsdelivr 镜像拉取，722 行）——`_CopyToModelParallelRegion.forward` 原样返回 input_、`backward` 调 `_reduce`（`torch.distributed.all_reduce`）= 式 (5)；`_ReduceFromModelParallelRegion.forward` = `_reduce`、`backward` 原样回传 = 式 (6)；公开包装 `copy_to_tensor_model_parallel_region` / `reduce_from_tensor_model_parallel_region` 均在；`_ScatterToModelParallelRegion`（前向 `_split_along_last_dim`/反向 `_gather_along_last_dim`）与 `_GatherFromModelParallelRegion`（反之）对称对存在；SP 族 `_GatherFromSequenceParallelRegion`（forward `_gather_along_first_dim`/backward `_reduce_scatter_along_first_dim`）与 `_ReduceScatterToSequenceParallelRegion`（反之）存在。`layers.py`（1814 行）：`ColumnParallelLinear` 调 copy_to（行 1414）、`RowParallelLinear` 调 reduce_from（行 1769）、`VocabParallelEmbedding` 掩码置零 + `reduce_from_tensor_model_parallel_region`（行 480-522）、`LinearWithGradAccumulationAndAsyncCommunication`（行 744）反向内 `all_reduce(grad_input, async_op=True)`（行 860，注释明示与 wgrad GEMM 重叠）。章节的教学版两个 Function 与官方逐行同构——成立。

### 5.1-B：TP 每层通信账

每层：attention 块 g（前向 1 次）+ MLP 块 g（前向 1 次）+ 两处 f（反向各 1 次）= 前向 2 次、反向 2 次 all-reduce。消息为完整 [b,s,h] 激活 bf16 = 2bsh bytes。前向每层 2×2bsh = 4bsh；反向对称 4bsh；单层单 micro-batch 8bsh；L 层 8bshL。ring 每 rank 发送系数 2(t−1)/t（4.1 式 (2)，已从 4.1 文件第 58 行核对：C_rank = 2(d−1)/d·M）⇒ C_rank = 8bshL·2(t−1)/t → 16bshL（t→∞）。t=2 时系数为 1、饱和到 2——「每卡通信量与 t 几乎无关」成立。论文侧核对：Megatron 原文「only two all-reduces in the forward path and two in the backward path」+ Figure 4 注「4 total communication operations…per layer」逐字命中。
式 (11)：每层每卡通信 16bsh(t−1)/t、计算 C_1/t ⇒ 比值 = 16bsh(t−1)/C_1 ∝ (t−1)——代数无误。
405B 数值例复算：M_msg = 2×1×8192×16384 = 268.4 MB ✓；每卡每层 8bsh·2·7/8 = 1.879 GB ✓×8 层（126/16≈7.9 的均分假设，写作方已声明）= 15.0 GB ✓；/900 GB/s = 16.7 ms ≈ 17 ✓；计算 6×(405e9/128)×8192 = 1.56e14 FLOPs /430 TFLOPS = 362 ms ≈ 360 ✓；17/360 = 4.7% <5% ✓；跨节点 15.0 GB/50 GB/s = 300 ms ✓。数字链全闭合。

### 5.3-A：气泡公式 (p−1)/(m+p−1)

逐步：stage p 的 F(·,1) 结束于 p·t_f（前向链 p 跳）；随后 F(·,2..m) 连续 ⇒ t_F,end = (p−1)t_f + m·t_f ✓式 (2)。GPipe 下 stage p 的反向紧跟：B(p,j) 结束于 t_F,end + j·t_b；B 沿 stage 逆序每段一跳 ⇒ B(1,j) 结束于 t_F,end + (j+p−1)·t_b；j=m ⇒ T = (m+p−1)(t_f+t_b) ✓式 (3)。每 stage 有效工作 m(t_f+t_b) ⇒ t_pb = (p−1)(t_f+t_b) ✓式 (4)。/T = (p−1)/(m+p−1) ✓式 (5)（GPipe 原文 O((K−1)/(M+K−1)) 逐字命中）；/t_id = (p−1)/m ✓式 (6)（Megatron 原文 t_pb/t_id 公式逐字命中）。换算：式 (5) = 式 (6)/(1+式 (6)) ✓（x=(p−1)/m ⇒ x/(1+x) = (p−1)/(m+p−1)）。m=4p ⇒ (p−1)/(5p−1) → 1/5 ✓「约 20%」成立。模拟器数值复核（自写运行）：p=4,m=8,t_f=1,t_b=2 ⇒ T=33=(8+3)×3，GPipe 与 1F1B 相同；全网格 (t_b∈{1,2,4})×(p∈{4,8})×(m∈{8,16,64}) 式 (3) 全成立。

### 5.3-B：1F1B 峰值激活 ≤ p 份

in-flight(i,t) := 已 F 未 B 的 micro-batch 数（须存激活的份数）。warmup：stage i 连发 p−i 个 F、零 B ⇒ in-flight = p−i。稳态严格 F/B 交替：每个 F 使 in-flight +1、紧随的 B −1 ⇒ in-flight ∈ {p−i, p−i+1}，峰值 p−i+1。cooldown 只减不增。全机最大在 stage 1：p ✓式 (7)。可行性检查：首个 B 到达 stage 1 的时刻 = p·t_f + (p−1)·t_b（F 链 p 跳 + B 链 p−1 跳），stage 1 此时已排完 warmup 的 p−1 个加稳态第 1 个共 p 个前向（耗时 p·t_f），等待恰 (p−1)t_b——等待只加空闲不加在飞份数，上界与梯度到达早晚无关 ✓。模拟器复核：p=4 各 stage 峰值 [4,3,2,1] = p−i+1 ✓；GPipe 全 stage = m=8 ✓。另与 Megatron Figure 4 像素重建独立印证：dev1 在 B1 前恰有 6 个 t_f 单位空闲 = (p−1)·t_b（t_b=2t_f、p=4 ⇒ 3 个反向块宽）✓。论文原话「activations to be stashed for p or fewer microbatches (compared to m microbatches for the GPipe schedule)」「the time spent in the bubble is the same for this new schedule」逐字命中 ✓。

## Verification log（抽查项 × 来源 × 结论）

| # | 抽查项 | 来源（方式） | 结论 |
| --- | --- | --- | --- |
| 1 | Megatron 1909.08053 摘要数字：8.3B / 512 GPU / 15.1 PFLOPS / 76% / 单卡基线 39 TFLOPS(30% peak) | arXiv abs 页（curl） | 逐字命中 ✓；「512 张 V100」见正文「up to 32 DGX-2H servers (a total of 512 Tesla V100 SXM3 32GB)」✓；但 DGX-2H 为 16 卡机 ⇒ S2-5 |
| 2 | f/g 定义与共轭、MLP 列切+行切、attention 按 head 切、每层 2+2 次 all-reduce、词表 padding、TP 组内收 | 1909.08053 ar5iv 全文 | 全部逐字命中（"f is an identity operator in the forward pass and all reduce in the backward pass…"「conjugates」；"partition the first GEMM in this column parallel fashion and split the second GEMM along its rows…"；「two all-reduces in the forward path and two in the backward path」）✓ |
| 3 | mappings.py / layers.py 类名与前后向语义、SP 算子族、VocabParallelEmbedding 掩码+置零+all-reduce、async dgrad 重叠 | NVIDIA/Megatron-LM main（fastly.jsdelivr 镜像，api 限流/raw 被墙时的备用通道） | 逐项命中 ✓（行号见重推导节）；写作方「/tmp 缓存核实」的存疑项 4 由本通道独立复核通过 |
| 4 | Megatron-SP 2205.05198：sbh(34+5as/h)、TP sbh(10+24/t+5as/(ht))、TP+SP sbh(34+5as/h)/t、all-reduce=RS+AG 通信恒等、「10sbh 即复制区段」、22B 4% → 530B/1T 2%（Table 4 + Figure 8） | ar5iv 全文 | 全部命中 ✓（Eq 1/2/4、Table 2、"the communication bandwidth used for tensor parallelism and tensor together with sequence parallelism are the same"） |
| 5 | Ring Attention 2310.01889：每卡一段 Q + K/V 轮转、环上只与邻卡交互、块大小 ≥ F/B 的重叠条件、"near-infinite"；**全文零命中 striped/zigzag/imbalance** | ar5iv 全文 + 关键词扫描 | 机制命中 ✓；striped 不在 Ring 原文的归属边界成立 ✓ |
| 6 | Striped Attention 真实出处 2311.09431（Brandon et al., 2023）：mod-N 等距条带、近似均衡、官方 repo | arXiv abs 页 + ar5iv 全文 + Figure 1 SVG 像素重建 | **章节引的 2405.13119 是宇宙学论文 ⇒ S1-1**；机制非首末配对 ⇒ S2-1 |
| 7 | Ulysses 2309.14509：all-to-all 换维（全序列×部分 head）、通信量随 s、c 同比放大时恒定 | ar5iv 全文 | 命中 ✓；c ≤ a 的 head 原子约束原文未明写（章节以推导口径给出，未伪托原文）✓；式 (5) 8bsh(c−1)/c² 复算 ✓（QKV 三项 6bsh(c−1)/c² + 输出换回 2bsh(c−1)/c²） |
| 8 | Megatron 集群 2104.04473：(p−1)/m 公式、≤p 份原话、气泡相同原话、Figure 3 注（效率不依赖 t_b/t_f 比）、interleaved 气泡÷v 与通信 ×v 原话、m 为 p 整数倍、scatter/gather 与 11%、1T@3072 A100@502 PFLOP/s@52%、bsh 边界口径、logit 层 2BshV/4BshV | ar5iv 全文 + abs 页 | 内容全部命中 ✓；**三处小节号指错 ⇒ S2-4** |
| 9 | Zero Bubble 2401.10241：Table 1（T_B+T_W=2T_F、T_W<T_F<T_B、M_W<M_B）、Table 2 三行、worker 峰值公式 (p−i+1)M_B+(i−1)M_W / (2p−2i+1)M_B+(2i−2)M_W、「reduced to a third」、§3 启发式、§4 post-validation、§5.1（首末 stage 少一层、bit-to-bit identical）、ZB-1p/ZB-2p（pM_B / 2pM_B 阈值）、Table 5 多数 <1%、§6 ZB-V（T_F=T_B=T_W 下 pM_B 零气泡）、摘要 23%/31%、32×A100-80G 4 节点 RoCE、sail-sg repo | ar5iv 全文 + 表格 HTML 抽取 | 全部逐字命中 ✓；5.3 §5 的 V3↔ZB 记号换算（V3 的 B=全反向，ZB1P 行展开 = ZB-H1 公式）独立代数验证成立 ✓ |
| 10 | DeepSeek-V3 2412.19437：「without using costly tensor parallelism」两处（§1 与 §3.2 开头，显存工程语境）、PP16×EP64(8 节点)×ZeRO-1、2048 H800、通信-计算约 1:1 原话、DualPipe 四组件+反向再拆两路、Figure 4 五色、Figure 5（8 PP rank/20 micro-batch）、Table 2 三行、「only increasing the peak activation memory by 1/PP times」、被 2 整除 vs Chimera、不随 m 增长、§3.2.3 embedding/head 同 rank、61 层、h=7168、d_c=512 | ar5iv 全文 + 表格 HTML 抽取 | 全部命中 ✓（写作方存疑项 1 的出处即 §3.2 开头一句，章节「训练框架一节的口径」准确）；§8「V3 未披露 61 层划分、本书不推测」与论文一致 ✓；DualPipe 后续开源 ⇒ S3-4 |
| 11 | Llama-3 herd 2407.21783：Table 3（405B：126 层/16384/128 head/8 KV head；8B：32 head/8 KV head）、Table 4（8192 GPU TP8×CP1×PP16×DP64 @8192 430 TFLOPS 43%；16384 GPU TP8×CP16×PP16×DP8 @131072 380 38%） | ar5iv 全文 + 表格 HTML 抽取 | 全部命中 ✓；5.1/5.2 引用与 4.4 §5 的表一致 ✓ |
| 12 | GPipe 1811.06965：气泡 O((K−1)/(M+K−1))、M≥4K 规则及「重计算反向可提前调度」注、re-materialization O(N+(L/K)(N/M)) vs O(NL)、同步 flush（"Gradients are applied synchronously at the end"）、Figure 2b naive 利用率的图 | ar5iv 全文 + Figure 2 PNG 目检 | 全部命中 ✓；Figure 2c 等宽格画法属实（与章节 GPipe 图兼容）✓ |
| 13 | 2104.04473 Figure 4 画法 | interleaved_schedule.svg 逐元素重建（块宽、颜色、x 坐标） | 反向块 = 2× 前向块宽、B1 逐级严格错开 ⇒ 5.3「同画法」为伪 ⇒ S2-2 |
| 14 | 5.3 模拟器断言（T=33、峰值 [8,8,8,8] vs [4,3,2,1]）与练习 3a 网格 | 本地 python3 -I 运行（/tmp/w3review/sim） | 全部复现 ✓（18 组 (t_b,p,m) 网格式 (3) 与峰值断言全过） |
| 15 | 跨章回指：4.1 式 (2)(3)、4.2 式 (3)（ZeRO-3 3N）、1.2 式 (2)/§5/§6/§10、1.4 式 (1) 与 TP 表、1.5 式 (3)、4.4 §4/§5/§6、models 4.1 §5.2、models 2.3 式 (4) | 仓库内文件直读 | 全部存在且内容对口 ✓；唯 5.2→5.1 的「RNG 注记」断链 ⇒ S2-3 |
| 16 | NVLink 900 GB/s（H100）、IB 每卡 50 GB/s 量级 | 公开规格（H100 NVLink4 900 GB/s；NDR400=50 GB/s×8 卡/节点） | 口径正确 ✓ |
| 17 | 5.2 式 (2)(3)(5) 与 5.3 式 (8)(11)、练习数值（5.1 §8 的 a_kv/(2a)=1/8；5.3 §8 的 3.8 GB/76 ms；练习 4c 的 16M_B/31 M_B） | 独立复算 | 全部成立 ✓ |

覆盖率声明：三章的全部**编号公式 (1)–(11)** 均经独立复算或一手来源逐字核对；全部 arXiv ID（1909.08053、2205.05198、2310.01889、2309.14509、2407.21783、1811.06965、2104.04473、2401.10241、2412.19437、2405.13119）逐个经 arXiv/ar5iv 回查——9 个标题-内容吻合、1 个错配（S1-1）；全部 repo 路径/类名/函数名经 main 分支文件核对；全部直接引号（英文原话 6 处）逐字命中；全部数字实例复算闭合。教学代码（5.1 f/g 教学版、5.3 模拟器）语义核对 + 模拟器实际运行通过。

## 写作方存疑项的独立裁决

1. （task-16 #1）1F1B 图口径：**裁决为 S2-2**——「声明+示意」的教学处理可接受，但「与 Megatron Figure 4 同画法」为假断言，必须修。
2. （task-16 #2）V3 未披露 61 层划分：属实，章节的「不推测」处理正确 ✓。
3. （task-16 #3-5）MoE 走读停在类级、DeepEP 一句、2048 算术：属 5.4 范围，本批不审。
4. （task-15 #1）「without using costly tensor parallelism」位置：论文 §1 与 §3.2 开头各出现一次，章节「训练框架一节」口径准确 ✓。
5. （task-15 #2）Megatron 摘要数字：逐项命中 ✓（唯 DGX-2H 卡数错 ⇒ S2-5）。
6. （task-15 #3）记号 2BTd→2bsh：章节内自洽、与 4.x/1.4 全书口径一致，且式 (9) 符号行有定义；接受 ✓。
7. （task-15 #4）repo 缓存核实：已由本审查经 jsdelivr 镜像独立重核，全部命中 ✓。
8. （task-15 #5）405B 每卡约 8 层的均分假设：已声明、不影响方向性结论；接受 ✓。
9. （task-15 #6）分析性结论归属：Ring 重叠口径、Ulysses GQA 约束、MLA 压缩分析的行文归属检查——均以推导/本书分析口径写出，未伪托原文 ✓；但 5.2 §4 的 striped 机制归属本身错误（S2-1），不在写作方已申报的分析性结论之列。

## 修复闭环要求

S1-1、S2-1～S2-5 必修；5.2 含 S1，修复后由 evaluator 复审该章。S3-1～S3-5 记入累积清单随波末清零。references.md/spec 的联动修正（S1-1 的三处 spec 记录、S3-5 的回写）需 controller 授权后执行（写作 agent 无 references.md 写权限）。

## 修复复审记录（2026-11-17 复审轮，fresh 全文重读 + 逐项回源）

### 逐条判定

| Finding | 判定 | 当前文本证据 |
| --- | --- | --- |
| S1-1（striped ID 错配） | **ADDRESSED** | 5.2 四处（metadata/§4/总结/参考）与 references.md、spec 三处均已改 arXiv:2311.09431、年份 2023，spec 附诚实勘误注记；2405.13119 全仓零残留（grep 确认） |
| S2-1（striped 机制张冠李戴 + 参数化矛盾） | **ADDRESSED** | 5.2 §4 全文重写：striped = 按位置 mod c 等距条带、近似均衡（两处论文原话引用核对一致，Figure 1 条带排列 {0,4,8,12}/… 与我当日的 SVG 像素重建一致，Figure 编号复核为 Figure 1）；zigzag = 2c 个 s/(2c) 半块、rank r 持 {r, 2c−1−r}——式 (4) w=(2c+1)u′ 独立重推成立（总账 Σ_r w_r = c(2c+1)u′ = Σ_{b=0}^{2c-1}(b+1)u′ 自洽，u′=u/4 正确）；「显存 s/c 不变、式 (2) 通信账不改、奇数 c 无遗留问题」三条现在参数化下全部成立；zigzag 归 Megatron-Core CP 实现（教学参照）——已在 repo main `megatron/core/utils.py` 核到 "zigzag load-balanced chunking…split into 2 * cp_size equal chunks" 原文 |
| S2-2（1F1B 图「同画法」为伪） | **ADDRESSED** | §3 图体重绘：B1 在 stage4/3/2/1 的第 5/6/7/8 格逐级错开，式 (1) 依赖逐格满足（我逐格核验四条 stage 行）；图注如实改写为「Megatron 论文 Figure 4 按 t_b=2t_f 块宽绘制，本图是它的等宽格重绘」；图注「与模拟器 t_f=t_b=1 输出逐格一致、T=22、每 stage 空闲 6 格」经我重跑模拟器逐格比对**全部吻合**（含峰值 [4,3,2,1]） |
| S2-3（RNG 注记断链） | **ADDRESSED** | 5.1 §3 新增「其三 RNG 同步注记」（TP 区段内独立流、复制区段共享 mask、Megatron 附录 B.2 口径）——与论文 B.2 原文逐句对上（"same seed…identical dropout patterns across all model parallel workers" / "separate random number generator…uniquely seeded for each model parallel worker"）；5.2 §2 的回指现在解析成立 |
| S2-4（三处小节号） | **ADDRESSED** | §1→§3.2、§8→§2.2、§4→§4.1/Figure 9 + §5.7（11%），正文与参考条同步修正 |
| S2-5（DGX-2H 卡数） | **ADDRESSED** | 5.1 §7 改「512 卡 = 32 台 DGX-2H，8 路 TP 组收在单台 16 卡的 NVSwitch 全互连域内、每机装两个 TP 组」——准确 |
| S3-1（g/ḡ 命名映射） | **ADDRESSED** | 5.2 式 (1) 已改用论文原记号 g/ḡ，并显式点破「此 g 非 5.1 的 g」的同名易混；逐节一致性检查通过（式 (1)、块图、总结、练习 1） |
| S3-2（年份） | **ADDRESSED**（2023） | 但见下方新 finding——证据等级配套改错 |
| S3-3（等时长画法同源） | **ADDRESSED** | 5.3 §5 已删 Megatron，只留 GPipe Figure 2c |
| S3-4（DualPipe 开源状态） | **ADDRESSED** | §9 改「调度器实现后于 2025 年单独开源（deepseek-ai/DualPipe repo，2025-02 创建）」——repo 实况复核一致 |
| S3-5（references 回写） | **ADDRESSED** | megatron-lm 条被引章节已含 5.2/5.3 |

### 修复引入的新问题（只查修复触及处）——发现 1 条

**S2-6（新）：striped-attention 的 [B] 降级与「无官方代码」失实。** 位置：5.2 metadata / §4 / 参考尾注的 [B] ×3 与「无官方代码（2026-11-17 核实）」；references.md 第 115 行「[B]（有 paper、无官方代码——2026-11-17 核实 togethercomputer 无该 repo）」。证据：论文第 1 页脚注明写 "We release the code for our experiments as open source. https://github.com/exists-forall/striped_attention/"——该 repo 今日（2026-11-17）在线（github.com 200，含 README.md / bpt.py 等，JAX、基于 Liu et al. Ring Attention codebase，与论文自述一致）；references.md 注记自曝查错了 org（togethercomputer ≠ 作者 exists-forall = William Brandon 的个人账号）。按 training spec §4.1（paper + official code 即 [A]，方法锚点无需 open weights）应为 **[A]** 并补 repo URL。建议修正：references.md 第 115 行改 [A] 并附 `https://github.com/exists-forall/striped_attention`；5.2 三处 [B] 改 [A]，参考尾注改为「官方 JAX 实验代码 exists-forall/striped_attention」。

### lint 重跑

`bash scripts/lint-chapter.sh` × 3 章：**LINT OK × 3**（2026-11-17）。

### 复审结论

原 S1×1 + S2×5 + S3×5 全部 ADDRESSED；修复引入 1 条新 S2（S2-6，证据等级与「无官方代码」失实，两处文件 4 个点位）。**S2-6 修复后放行**（改动小、无扩散风险，修复后无需再次全量复审，核对 4 个点位即可）。
