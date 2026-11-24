# gpu 维度 Spec 对抗审查 · 路线 B：P5–P6 kernel 计算优化 / 低精度与注意力 kernel

- 日期：2026-11-24
- 审查对象：`docs/superpowers/specs/2026-11-24-gpu-dimension-design.md`（§5 Part 5–6 大纲 L282–333、§6.3 强制重推导清单 5.1/5.2/6.3 条、§4.2 两层走读纪律、§2 边界、§4.3 符号表、附录预核实 + builder-check 修复 #1/#6）
- 审查人：fresh-context adversarial evaluator（路线 B，不继承写作上下文；未写本 spec；未派发子代理）
- 标准：STYLE_GUIDE.md §6 / §7 / §10（S1 事实致命 / S2 实质 / S3 minor；verification log 强制）
- recap 对照源（全文读）：models 4.1（`models/04-modern-architectures/01-kv-compression-and-exact-attention.md`）、inference 3.1（`inference/03-inference-operators/01-decode-attention-kernels.md`）；inference 3.2/3.3 定点 grep；inference README X-1 errata 专段
- 任务：事实核查（WebFetch/curl 一手）+ 独立数学重推导 + 代码走读锚点核实 + 边界检查 + MLA 几何最高优先核查

## 0. 核查通道与约束（影响覆盖率，先声明）

| 通道 | 状态 | 用途 |
| --- | --- | --- |
| WebFetch arxiv.org | 本机被挡（domain verify 失败） | 改 curl abs 页（spec risk 8 已列 fallback 通道） |
| arxiv.org/abs（curl） | 可用；2307.08691 两次返回 stub（4.2KB），title 可解析、abstract 不可得 | FA/FA3/Marlin/MX/FP8/FlashInfer abs 全文；FA2 abstract 沿 inference 3.1 已核实记录（2026-11-23 arXiv API）+ 本次 title 核实 |
| export.arxiv.org API | 超时（120s 无响应，后台任务失败退出） | 弃用 |
| api.github.com（repos / contents） | 可用，匿名限额内（~20 次调用） | 13 repo 存在性 + 目录/README 锚点 |
| docs.nvidia.com（curl） | 可用 | PTX ISA 全文 3.07MB（grep 定位逐字引句）、cuBLAS 1.02MB |
| cs.cmu.edu（curl） | 可用但慢（Ble90.pdf 1.35MB 两次下载截断） | CMU-CS-90-190 摘要页（1.5KB）完整取得；报告 PDF 正文未读，scan 算法 work/depth 以 GPU Gems 3 ch39（NVIDIA 官方，完整取得）+ 独立 Python 模拟互证 |
| raw.githubusercontent.com | 未使用 | 沿 spec risk 8 直接走 contents API |

抓取件存 `/tmp/gpu-review-b/`（审查输入，非仓库交付物）。数学重算脚本为 inline Python（Bash heredoc），全部断言 PASS/FAIL 见 §2。

**总判定：S1 = 0，S2 = 4，S3 = 5。**

- **MLA 几何最高优先核查：PASS**——spec 6.4 内容行与关键推导行均正确写 V4.1 512 = 448 latent + 64 rope、V3 576 = 512 + 64 两代区分 + 「勿写 512+64=576」坑注，与 inference README X-1 四方一手证据口径逐字一致；builder-check 修复 #1 已落地，无 X-1 重犯（§3）。
- **6.3 recap 纪律：PASS**——IO 定理 O(T²d²/M) 在 §2 / §4.2 / §6.3 / 6.3 大纲四处均声明 recap 不重推；inference 3.1 策略级 recap + 指令级增量声明完整（§4）。
- S2 集中在：6.3 shared 预算式 K/V 少计一份（数学）、5.3 online softmax 正确性重推导越界（边界）、§4.3 符号专归规则与 P5 大纲冲突 + IO 定理 recap 翻译缺口（约定）、6.1 NVFP4 scale 误写 E4M3（事实，PTX 逐字 UE4M3 = 7-bit unsigned）。
- 全部 ⟨verify⟩ 延迟项按任务书不计 findings；本路顺带解决的列 §5 供 W3 research pass 引用。

## 1. Findings 表

| ID | severity | 位置 | spec 断言 | 证据（一手） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| B-1 | **S2** | §5 6.3 内容行（L323）+ §6.3 清单 6.3 条（L477） | 「$T_M \times d$ Q tile + $T_N \times d$ KV tile ≤ shared 容量」——KV 按**一份** $T_N d$ 计 | 标准 FA 的 K 与 V 是**两份独立** tile。recap 源 models 4.1 §5.3 逐字：「SRAM 要同时驻留 $K_j, V_j$（**各** $B_c d$）、当前 $Q_i$ 与累加器 $O_i$（各 $B_r d$）、中间块 $S_{ij}$（$B_r B_c$）」。FA2/FA3 kernel 的 smem 布局 Q/K/V 分开（flash-attention repo `csrc/flash_attn/src`、`hopper/` 本次核实存在）。独立重算：bf16、d=128、T_M=T_N=128 时 spec 式给 64KB，正确 (T_M+2T_N)·d·2 = 96KB，**少计 32KB**（§2.4） | 预算式改 $(T_M + 2T_N)\,d\cdot\mathrm{bytes} \le S_{\mathrm{sh}}$（单缓冲；multi-stage 流水再乘 stage 数），或显式定义「KV tile = K、V 各一份 $T_N d$」。此项在强制重推导清单内，照抄大纲式会推出错误 tile 上界，须写作前修 |
| B-2 | **S2** | §5 5.3 内容行 + 关键推导行（L297–298） | 「**online softmax 推导**……rescale 正确性」；关键推导「online softmax 增量更新正确性（命题：running max/sum 与 two-pass 等价）」 | spec §2（L36）明列「FlashAttention 前向数学与 **online softmax 正确性**……（models 4.1）→ models/（recap 一句 + 纯文字指引）」为 out of scope；§4.2（L110）把 online softmax 正确性放在 **recap 侧**（增量 = tile 预算 + HBM 流量 + 指令级走读）；§6.3 强制清单无 5.3 online softmax 条；risk 3（L535）「重推已交付公式即 S2」。models 4.1 §5.2 已交付同一命题（不变量 + 归纳证明，式 (15)(16)），standalone softmax 只是去掉 v_s 的标量特例，归纳结构全同。5.3 自己的锚点行也写「models 4.1（online softmax **recap**）」——内容行与锚点行自相矛盾 | 关键推导改「online softmax 正确性 recap models 4.1 式 (15)(16)（不重证）；增量 = standalone softmax two-pass vs online single-pass 访存账（2 读 vs 1 读）+ warp/block reduction 映射（5.1 recap）+ Welford online variance（新）+ norm 带宽下界（新）」。Welford 与访存账是真增量，保留 |
| B-3 | **S2** | §4.3 消歧硬规则（L136）vs §5 5.1（L285–286）/ 5.2（L292）/ §4.2 recap 行（L110） | 硬规则「$N$ 专归参数量；$n$ 专归 rank 数；……」无条件全书生效；但 5.1 用 $n$ = 元素数（「$n$ 元素求和……$O(n)$ work」）、5.2 用 $M,N,K$ = GEMM 问题维（「HBM traffic = $O(MNK/T)$」，$M$ 与表中「消息总字节」、$N$ 与「专归参数量」冲突）；§4.2 recap 式 $O(T^2 d^2/M)$ 的 $M$ = SRAM 容量（元素计）、$d$ = head 维，而翻译规则只给「models $d$（hidden）→ $d_{\mathrm{model}}$」 | spec 内部互证：§4.3 表 $M$ 行 =「被归约/传输消息总字节」、$n$ 行 =「rank/GPU 数」；models 4.1 §5.3 逐字「$M$ 为可用 SRAM、元素计」；inference 3.1 §1 已处理同一翻译：「其 d = head 维，按 spec §4.3 翻译规则译为本章 $d_{\mathrm{head}}$」——gpu spec 无对应注，盲用「d（hidden）→ d_model」会把 IO 定理的 head 维误译成 hidden 维。§4.3 无 head 维 / GEMM 问题维行；builder-check「符号表内部一致」结论只查了表内、未查表 vs P5 大纲 | §4.3 加章域句：「P5/P6 kernel 章：$n$ = 元素数、$M,N,K$ = GEMM 问题维、$d$ = head 维，章首声明；通信章 8.x/10.x 按表内通信语义」+ 补 GEMM 问题维 / head 维两行；§4.2 recap 行补 IO 定理翻译注（$d \to d_{\mathrm{head}}$、$M \to$ SRAM 符号或章首声明 $M$ = SRAM 元素数，沿 inference 3.1 先例）。约定冲突属 STYLE_GUIDE §10 S2 例示 |
| B-4 | **S2** | §5 6.1 内容行（L311） | 「NVFP4 per-16 **E4M3** + global FP32」 | PTX ISA（docs.nvidia.com/cuda/parallel-thread-execution/，本次全文抓取）逐字：「ue4m3 data format: This data format is a **7-bit unsigned** floating-point format with 4 bits for exponent and 3 bits for mantissa. The ue4m3 encoding does not support infinity. NaN value is limited to 0x7f. A register variable containing single ue4m3 value must be declared with .b8 type having **MSB bit padded with zero**.」——无 sign bit，与 E4M3（1+4+3，有 sign、NaN 编码不同）位布局不同。tcgen05 组合表逐字：「.kind::mxf4nvf4 .e2m1 **.ue4m3** .scale_vec::4X」（block16）。recap 源 inference 3.2 §2 表行与规范事实段均写 **UE4M3**（cuBLAS 记法 CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3） | 6.1 改写「NVFP4 per-16 **UE4M3**（7-bit unsigned，8-bit 容器 MSB 补零）+ global FP32」。6.1 的交付物就是 sign/exponent/mantissa 位布局表，照大纲写 E4M3 会多画一个 sign bit、动态范围/NaN 账同错；recap 忠实性也破（inference 3.2 已交付 UE4M3） |
| B-5 | S3 | §5 6.1 内容行（L311） | 「**MX 格式**（MXFP8 per-32 E8M0 scale、MXFP4、NVFP4 per-16 E4M3 + global FP32；**OCP MX 规范 [C]**，inference 3.2 已锚 recap）」——NVFP4 与 OCP MX 规范绑在同一出处括注内 | NVFP4 不在 OCP MX v1.0：MX 合规格式 = 块 32 + E8M0（inference 3.2 引规范原文「an unsigned representation of a conventional biased Float32 exponent」；MX paper arXiv:2310.10537 本次核实）；PTX 把 .kind::mxf4（E2M1+ue8m0，block32）与 .kind::mxf4nvf4（E2M1+ue4m3，block16）分为两种 kind。inference 3.2 正确分源：MXFP8/MXFP4 → OCP MX v1.0 [C]，NVFP4 → NVIDIA cuBLAS/TensorRT-LLM 官方文档 [C] | 括注拆源：「MX 族（MXFP8/MXFP4：OCP MX v1.0 [C]）；NVFP4（NVIDIA 官方文档 / PTX ISA [C]，非 OCP MX v1.0 格式）」。防 writer 把 NVFP4 挂 OCP 规范成张冠李戴引用（届时升 S1） |
| B-6 | S3 | §5 5.2 关键推导行（L292） | 「HBM traffic = $O(MNK / T)$」未声明假设 | 独立重推导（§2.3）：一般式 traffic = $MNK(1/T_M + 1/T_N)$（A 读 $MK\cdot N/T_N$ 遍 + B 读 $KN\cdot M/T_M$ 遍），**方阵 tile** $T_M=T_N=T$ 才得 $2MNK/T$；前提是每 tile 从 HBM 读（**无 L2 复用**）、C 写 $O(MN)$ 低阶略去。H100 L2 50MB（spec 2.2）可捕获 B/KV 复用，实际 traffic 低于该式；split-K 会增 traffic | 大纲补「方阵 tile、无 L2 复用模型；一般式 $MNK(1/T_M+1/T_N)$；L2 命中与 split-K 修正入走读对账」。与 4.3 强度式 $I = 2T_M T_N/(T_M+T_N)$ 的一致性本次已数值验证（§2.3），无需改式本身 |
| B-7 | S3 | §5 6.3 内容行（L323） | 「accumulator $T_M \times d$ FP32 ≤ register file → tile size 上界」——以整 SM register file 为约束 | 绑定约束是**每线程** register 上限（255）：accumulator 按线程分摊 $T_M d / n_{\mathrm{threads}}$，FA2 的 $S_{ij}$ tile 也在 register（$T_M T_N / n_{\mathrm{threads}}$）。数值例（§2.4）：T_M=T_N=128、d=128、4 warps（128 线程）→ acc+S = 128+128 = 256 regs/thread > 255，tile 不可行；8 warps → 128 regs/thread 可行。register file（64K regs/SM）约束的是 blocks/SM（occupancy 层），不是单 tile 上界。inference 3.1 §5 的 MLA 实例（32768/65536）是 block 总量口径，gpu 增量应下到 per-thread | 大纲改「accumulator + S tile 按线程分摊 ≤ 255 regs/thread（绑定）；block 总量 × blocks/SM ≤ register file（occupancy 层，3.4 回链）」。命题「tile 上界由 shared 与 register 决定」方向正确，保留 |
| B-8 | S3 | §5 6.2（L317）/ 6.4（L329）大纲 vs §4.2（L115） | §4.2 要求「每章大纲自行声明代码 scope……6.2 Marlin / 6.4 FlashMLA 等含 repo 走读的章**在大纲声明其片段范围**」；builder-check 修复 #6 声称已落地。实际 6.2 只写「repo 走读」、6.4 只写「FlashMLA repo 走读」，均无显式片段级声明 / 文件范围 | §4.2 逐字（L115）；6.2/6.4 大纲全文无「片段级 / 模块级」字样（5.2 有「模块全量走读」、6.3 有「指令级 kernel 走读增量」，对照明显）。主题清单（Marlin: mixed-input tiling / lop3/bfe；FlashMLA: kernel 几何 / paged gather）已隐式收窄，但显式声明缺失 | 6.2/6.4 各加一行「代码 scope：片段级（Marlin: dequant mainloop 片段，文件锚 marlin/marlin_cuda_kernel.cu；FlashMLA: 见 B-9 版本锚）」，与 5.2/6.3 主走读章划界。顺带一句 6.3↔6.4 split-KV 分工（6.3 = 流量账修正、6.4 = reduction kernel 指令走读），防两章重复推导 |
| B-9 | S3 | §5 6.4 锚点行（L331）+ §6.4（L486） | 「FlashMLA repo 走读」无版本/commit 范围；§6.4 版本漂移坑注清单「CUDA/NCCL/CUTLASS/FA/PTX 版本」不含 FlashMLA/DeepGEMM | FlashMLA README（本次 contents API 全文读）逐字：「Breaking change notice (2026.09.30): …we **removed support for the Hopper architecture** and for earlier models (including DeepSeek V3 / V3.2 / V4.0), and we changed the FP8 / FP4 KV cache format」，旧机制自链 commit ba89a34；当前 main = V4.1（B200 + Ascend 950）sparse / fused kernel。inference 3.1 §5 已立坑注「把 FlashMLA 当前 main 当 Hopper MLA decode 读……代码已移除」并分工「Hopper 机制锚 ba89a34 + deep-dive doc，当前 main 归 7.3」。gpu 6.4 若要走读 absorbed decode seesaw 的指令级实现，main 上已不存在 | 6.4 大纲声明 FlashMLA 走读版本锚：V4.1 sparse/fused kernel → 当前 main（与 6.4 的 V4.1 512 几何一致）；Hopper absorbed 机制指令级 → ba89a34 + docs/20250422-new-kernel-deep-dive.md（沿 inference 3.1 口径，纯文字指引）。§6.4 漂移清单补 FlashMLA/DeepGEMM/DeepEP |

## 2. 独立数学重推导（强制清单 5.1 / 5.2 / 6.3，全部独立重算）

### 2.1 reduction tree 带宽最优性（5.1）——PASS

- 下界：任何求和算法必须读全部 $n$ 个输入至少一次 → HBM 流量 $\Omega(n\cdot\mathrm{bytes})$、work $\Omega(n)$。
- tree reduction：$n-1$ 次加法（每元素恰读一次）→ work $O(n)$ 达下界；fan-in 2 时 depth $\lceil\log_2 n\rceil$（亦为 bounded fan-in 下界阶）；naive 串行 depth $n-1$。
- Python 数值：n=8/1024/1048576 → work 7/1023/1048575，depth 3/10/20 = ceil(log2 n)。spec「每元素至少读一次 → Ω(n) 下界，tree 达下界」正确。
- 口径注（非 finding）：「带宽最优」指 HBM 流量下界；章内宜把流量下界（Ω(n) bytes）与并行 depth 下界（Ω(log n)，fan-in 2）分开陈述。5.1「消息量账（reduction 通信量 vs scan）」在单 GPU kernel 语境宜写明是 shared/片上 traffic 还是多 GPU 预告。

### 2.2 Blelloch scan work O(n) / depth O(log n)（5.1）——PASS

- up-sweep（reduce）：$d = 0..\log_2 n - 1$，每层 $n/2^{d+1}$ 次加法，合计 $n-1$；down-sweep：同层数，每层一次 swap + 一次加法，加法合计 $n-1$。总 work $2(n-1) = O(n)$，depth $2\log_2 n = O(\log n)$。exclusive scan（root 置 0）。
- Python 模拟（随机输入，n=8/16/1024）：输出与直接前缀和逐位相等；加法计数 14/30/2046 = 2(n-1)；depth 6/8/20 = 2 log2 n。全 PASS。
- 一手互证：GPU Gems 3 ch39（NVIDIA 官方）逐字「work-efficient … the same work complexity, O(n)」「the reduce phase (also known as the up-sweep phase) and the down-sweep phase」「After Blelloch 1990」；CMU-CS-90-190 摘要页确认报告出处（§6 log）。
- 对照：Hillis-Steele 形 scan 是 O(n log n) work / O(log n) depth（GPU Gems 3 同章述 Horn/Hensley O(n log n)）——spec 把 work-efficient 形归 Blelloch 正确。

### 2.3 GEMM tile 层级流量（5.2）——PASS（带 B-6 假设注）

- 设 C[M,N] = A[M,K]·B[K,N]，threadblock tile $T_M\times T_N$，K 维步长 $T_K$。block 数 $(M/T_M)(N/T_N)$；每 block 沿 K 读 A tile $T_M T_K$、B tile $T_K T_N$，共 $K/T_K$ 步。
- A 总流量 $= (M/T_M)(N/T_N)\cdot T_M K = MKN/T_N$；B 总流量 $= KNM/T_M$；合计 $MNK(1/T_M + 1/T_N)$。方阵 tile $T$：$2MNK/T = O(MNK/T)$ ✓。C 写 $MN$ 低阶。
- 强度 $I = 2MNK / [MNK(1/T_M+1/T_N)] = 2T_M T_N/(T_M+T_N)$，与 4.3 大纲式 $O(T_M T_N/(T_M+T_N))$ 一致 ✓。
- Python 数值（M=N=K=4096）：(128,128) → 1073.7M elem、I=128.0；(64,256)/(256,64) → 1342.2M elem、I=102.4；与解析式逐位吻合。
- 假设：无 L2 复用、方阵 tile（B-6）；register tiling 不改 HBM 流量（只降 shared→register 级），与 5.2「层级流量账」分层一致。

### 2.4 FA tile↔shared/register 预算（6.3）——shared 式 FAIL（B-1），register 式粒度不足（B-7）

- shared（bf16，单缓冲）：Q tile $T_M d$ + K tile $T_N d$ + V tile $T_N d$ → $(T_M + 2T_N)d\cdot 2$ B。spec 式 $(T_M+T_N)d\cdot 2$ 少计 $T_N d\cdot 2$。数值（d=128）：(128,128) spec 64KB vs 正确 96KB；(64,64) 32 vs 48KB；(128,64) 48 vs 64KB。H100 228KB shared 下两式都装得下该例，但 tile 上界求解会系统性偏大——强制重推导项照抄即错。
- register：accumulator $T_M d$ FP32 按线程分摊；FA2 的 $S_{ij}$（$T_M T_N$）亦在 register。per-thread = $(T_M d + T_M T_N)/n_{\mathrm{threads}}$ + 操作数/统计量 ≤ 255。数值：(128,128,d=128) 4 warps → 256 regs/thread 超限；8 warps → 128 可行。整 SM register file 64K regs = 256KB 约束 blocks/SM（occupancy），非单 tile 上界（B-7）。
- multi-stage 流水（FA3 TMA）shared 再乘 stage 数；Q 常驻 register 后 smem 可释放——章内走读时按实际 kernel 布局对账（大纲式是一阶约束）。

### 2.5 FA GPU 侧精确 HBM 流量（6.3）——PASS（recap 对账成立）

- FA2（Q outer / KV inner，每 block 一个 Q tile）：Q 读一遍 $Td$、O 写一遍 $Td$；K/V 被全部 $T/T_M$ 个 Q block 各读一遍 → $2T^2 d/T_M$（non-causal）；causal 跳上三角块 ≈ 减半。split-KV 另加 partial O+LSE 写读（inference 3.1 式 (4) 已交付账，recap）；GQA 组内共享 KV 读 → KV 项按 $H_{kv}$ 而非 $H_q$ 计。
- 与 models IO 定理对账：$T_M = \Theta(M/d)$（M = SRAM 元素数）时 $2T^2 d/T_M = \Theta(T^2 d^2/M)$ ✓。Python 数值（T=8192, d=128, M=116736 elem）：T²d²/M = 9.4M；T_M = M/(4d) = 228 时 2T²d/T_M = 75.4M = 8·T²d²/M——同渐近阶，常数差来自 tile 取法与 FA1/FA2 循环序（FA1 重读 Q/O、FA2 重读 K/V，models 4.1 §5.3 交付的是 FA1 序）。
- 下界 Ω(Td)（Q/K/V 至少读一遍 + O 写一遍）；「实际 traffic 与理论下界对账」增量定义良好。L2 捕获 K/V 重读是实测低于模型的主因（2.2 回链）。
- recap 纪律核查：§2 L36 / §4.2 L110 / §6.3 L477 / 6.3 大纲 L323 四处均声明 IO 定理 recap 不重推 ✓；6.3 大纲「Q outer / KV inner」是 FA2/FA3 序，FA1 序（KV outer）已在 models 4.1 §5.3 交付，章内 recap 即可（log 注，非 finding）。

## 3. MLA 几何专项核查（最高优先）——PASS，无 X-1 重犯

| 位置 | spec 原文（摘） | 对照 | 结论 |
| --- | --- | --- | --- |
| 6.4 内容行 L329 | 「V4.1 共享 KV 向量 512 元素 = $d_{\mathrm{latent}}$ 448 + $d_{\mathrm{rope}}$ 64（inference W4 errata 口径，config 无 kv_lora_rank 字段、引擎导出 448；**勿写 512+64=576，那是 V3 几何**）」 | inference README X-1 专段：「真值 = 512 元素 = kv_lora_rank 448 + rope 64，四方一手互证」（config 无字段 / DeepInfer config.py L288-289 / cache_common.py quantized_elements=512 / FlashMLA README 528B） | ✓ 逐字一致 |
| 6.4 关键推导行 L330 | 「V4.1 512 元素 = 448 latent + 64 rope，V3 576 = 512 latent + 64 rope，两代区分」 | X-1：V3 行（576/1152/68.6KiB）不动；两代显式区分 | ✓ |
| §4.1 [W] 例 L99 | 「如 V4.1 MLA 512 元素」 | 同上 | ✓ |
| builder-check 修复 #1 L597 | 记录初稿「d_latent 512 + d_rope 64」（=576）错误已修 | 本次核对修复已落地于 L329/L330 | ✓ 闭环 |

任务书警戒情形（「d_latent 512 + d_rope 64」=576 误套 V4.1）**未出现**。6.4 的 FlashMLA 走读版本锚问题另见 B-9（非几何错误）。

## 4. 边界与两层走读核查

| 检查项 | 结论 | 证据 |
| --- | --- | --- |
| P5/P6 不重推 models 4.1 FA 数学 / IO 定理 | **PASS（除 5.3，B-2）** | 6.3 大纲「models 4.1 FA 数学 + IO 定理 recap……models IO 定理 recap 不重推」；5.4「recap models 4.1 IO 定理，不重推」；5.3 online softmax 正确性列关键推导 → B-2 |
| P5/P6 不重推 inference 3.1 策略级 | PASS | 6.3「inference 3.1 算法/策略级 recap + 指引 → 指令级 kernel 走读增量」「FA2 warp partitioning / seq 维并行（inference 3.1 策略级 recap → 指令级 warp 分工）」；6.4「inference 3.1/3.3……recap + 指引 → GPU 侧 kernel 增量」；split-KV / DSA / paged 均声明 recap → 指令级增量。inference 3.1 已交付件（命题 1 split-KV 正确性、FA2/FA3 三技术策略表、FlashMLA seesaw 策略级、命题 4 DSA 预选）在 gpu 大纲中均无重推导声明 |
| 不跨 serving 调度 / 引擎编排 | PASS | 5.2 cuBLASLt heuristic = 库级 algo 选择（硬件/库机制）；6.4「变体 kernel 选择决策表（几何 × batch × hardware）」= kernel 选型非请求调度；6.4 paged attention = block table gather 指令实现（inference 2.1/3.1 recap），无 batching / 投机解码 / 引擎编排内容 |
| 走读 scope 声明 | 5.2 模块级 ✓（「模块全量走读」）；6.3 指令级主走读 ✓；**6.2/6.4 片段范围未显式声明 → B-8**；其余章机制片段级（教学 kernel 片段）✓ | §4.2 L115 要求 vs 各章大纲逐行核对 |
| recap 翻译硬规则 | IO 定理 M/d 翻译缺口 → B-3；其余（training d→n 等）非本路 | §4.2 L116 |

## 5. ⟨verify⟩ 延迟项处理（按任务书不计 findings；本路顺带解决供 W3 引用）

| spec ⟨verify⟩ 项 | 本路结论（2026-11-24） | 处置 |
| --- | --- | --- |
| FA4 是否存在 / tcgen05（6.3） | **存在**：flash-attention README 逐字「FlashAttention-4 (CuTeDSL) … written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200)」，接口 `flash_attn.cute.flash_attn_func` | 延迟项不计 finding；W3 research pass 核 file:line 与 tcgen05 细节 |
| NVIDIA/cub（5.1） | repo 200 但 **archived=True**；活跃开发并入 NVIDIA/cccl（200，"CUDA Core Compute Libraries"） | 延迟项不计 finding；建议 W3 锚改 CCCL（CUB 子库）或注明 archived |
| openai/triton（5.4） | **301 → triton-lang/triton**（200，main） | 延迟项不计 finding；W3 锚用现名 |
| CUTLASS 当前版本架构（5.2） | 未核版本（延迟项）；repo 200、media/docs 存在（cpp / cutlass_compiler / operators / pythonDSL 四目录） | 留 W3 |
| FP4 4× vs BF16 datasheet（6.1） | 与 inference 3.3 已核实口径一致（B200 FP4 9000 TFLOPS dense = BF16 2250 的 4×，[C] datasheet） | 留 W1/W3 datasheet 一手复核 |
| TransformerEngine / Apex norm kernel（5.3） | NVIDIA/TransformerEngine 200；Apex 未核 | 留 W3 |
| cuda-samples reduction/scan 路径（5.1 Sources） | repo 200；**目录已重构**：旧 Samples/ 不存在，现 `cpp/2_Concepts_and_Techniques/{reduction, scan, shfl_scan, reductionMultiBlockCG, threadFenceReduction}` | spec 未断言具体路径，不计 finding；W3 file:line 锚用新路径 |

## 6. Verification log（抽查项 × 来源 × 结论 × 覆盖率）

### 6.1 arXiv / 论文出处（curl abs 页，2026-11-24）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| FA 2205.14135 | https://arxiv.org/abs/2205.14135 | ✓ title 逐字「FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness」；abstract 确认 IO-aware exact attention / tiling / IO complexity optimal for a range of SRAM sizes |
| FA2 2307.08691 | https://arxiv.org/abs/2307.08691（两次） | ✓ title 逐字「FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning」；abstract 页 stub，内容沿 inference 3.1 已核实记录（2026-11-23 API：~2× FA1、A100 50–73%、225 TFLOPs/s） |
| FA3 2407.08608 | https://arxiv.org/abs/2407.08608 | ✓ title 逐字「FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision」；abstract 逐字：FA2 only 35% util on H100、warp-specialization、interleave matmul/softmax、FP8 block quantization + incoherent processing、1.5–2.0×、FP16 740 TFLOPs/s (75%)、FP8 close to 1.2 PFLOPs/s、2.6× lower error——与 spec 6.3 / inference 3.1 recap 全合 |
| Marlin 2408.11743 | https://arxiv.org/abs/2408.11743 | ✓ title 逐字「MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models」；abstract 逐字「batchsizes up to 16-32 … close to maximum (4×) quantization speedup … 64-128 … still significant」——spec 6.2「near-ideal ~4×」成立（inference 3.3 精确口径 3.88 = 16/4.125 含 g128 scale） |
| MX paper 2310.10537 | https://arxiv.org/abs/2310.10537 | ✓ title「Microscaling Data Formats for Deep Learning」；abstract 确认 per-block scaling + narrow FP/INT 元素（MX 概念）；块 32 + E8M0 细节沿 inference 3.2 OCP v1.0 逐字引句 |
| FP8 paper 2209.05433 | https://arxiv.org/abs/2209.05433 | ✓ title「FP8 Formats for Deep Learning」；abstract 逐字「E4M3 (4-bit exponent and 3-bit mantissa) and E5M2 (5-bit exponent and 2-bit mantissa) … E4M3 … not representing infinities and … one mantissa bit-pattern for NaNs」——6.1 位布局账源头成立 |
| FlashInfer 2501.01005 | https://arxiv.org/abs/2501.01005 | ✓ title 逐字「FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving」（6.4 [A] 锚） |
| Blelloch 1990 CMU-CS-90-190，无 arXiv | https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html | ✓ 页面逐字「Guy E. Blelloch. CMU-CS-90-190, November 1990. 311k PDF」+ BibTeX（institution CMU SCS, month nov, year 1990）；无 arXiv 版本（WebSearch 多源一致：CMU 页 / KiltHub / Semantic Scholar）。**注**：~guyb/papers/Ble90.pdf 实为 MIT Press 书《Vector Models for Data-Parallel Computing》（本次下载文本识别），报告 PDF 在 CMU 页链接（Ble93.pdf 为 1993 Synthesis 章节版）——spec 引报告号正确，W3 取 PDF 时勿混书/报告 |
| Blelloch scan work/depth 算法内容 | https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda | ✓ 逐字「work-efficient … O(n)」「reduce phase (also known as the up-sweep phase) and the down-sweep phase」「After Blelloch 1990」+ exclusive/inclusive 定义；与 §2.2 独立模拟互证 |

### 6.2 GitHub repo / 代码锚点（api.github.com，2026-11-24）

| 抽查项 | HTTP / 分支 | 结论 |
| --- | --- | --- |
| IST-DASLab/marlin（6.2 [A]） | 200 / master | ✓；`marlin/marlin_cuda_kernel.cu` + `marlin_cuda.cpp` 存在（指令级 dequant 走读锚） |
| Dao-AILab/flash-attention（6.3 [A]） | 200 / main | ✓；`csrc/flash_attn/{flash_api.cpp, src/}`（FA2）+ `hopper/`（FA3）存在；README 含 FA4 CuTeDSL 节（§5） |
| deepseek-ai/DeepGEMM（5.2/6.2 教学参照） | 200 / main | ✓ README 逐字「SM90 requires scaling factors in FP32 format. SM100 requires scaling factors in packed UE8M0 … packs 4 UE8M0 into a single torch.int」+ DeepJIT / NT layout / TMA-aligned LHS SF——spec「JIT/UE8M0 for Blackwell」增量成立 |
| deepseek-ai/FlashMLA（6.4 教学参照） | 200 / main | ✓ README 逐字 2026.09.30 breaking change（移除 Hopper + V3/V3.2/V4.0、改 FP8/FP4 KV 格式、自链 ba89a34）；当前 main = V4.1 sparse/fused；`csrc/ docs/ flash_mla/` 存在 → B-9 版本锚建议 |
| NVIDIA/cutlass（5.2） | 200 / main | ✓；`media/docs/` 存在（cpp / cutlass_compiler / operators / pythonDSL） |
| NVIDIA/cub（5.1 ⟨verify⟩） | 200 / main，**archived** | 存在但已归档 → §5 延迟项表（CCCL 200 核实） |
| NVIDIA/cuda-samples（5.1） | 200 / master | ✓；reduction/scan/shfl_scan 现 `cpp/2_Concepts_and_Techniques/`（旧 Samples/ 404）→ §5 |
| flashinfer-ai/flashinfer（6.4） | 200 / main | ✓ |
| vllm-project/vllm（6.4） | 200 / main | ✓（arXiv 2309.06180 沿 inference 已核实记录） |
| NVIDIA/TransformerEngine（5.3 ⟨verify⟩） | 200 / main | ✓ |
| openai/triton（5.4 ⟨verify⟩） | **301** → triton-lang/triton 200 / main | 组织迁移 → §5 |
| NVIDIA/cccl | 200 / main | ✓（CUB 活跃 home） |
| PTX 指令锚（5.1 shfl / 6.2 lop3·prmt·bfe / 6.3 wgmma·TMA / 1.3↔6.1 tcgen05） | PTX ISA 全文 grep | ✓ shfl.sync / lop3 / prmt / bfe / wgmma.mma_async / cp.async.bulk.tensor / tcgen05.mma / mbarrier 全部存在 |

### 6.3 低精度格式事实（PTX ISA 全文 + recap 源互证）

| 抽查项 | 来源 / 引句 | 结论 |
| --- | --- | --- |
| FP8 E4M3 = 1+4+3、无 inf、单 NaN 型 | 2209.05433 abstract（§6.1）+ training 1.3 / inference 3.2 recap（E4M3 max 448） | ✓ |
| FP8 E5M2 = 1+5+2、IEEE inf/NaN | 同上（E5M2 max 57344 沿 inference 3.2） | ✓ |
| FP4 E2M1 = 1+2+1、无 inf/NaN | PTX 逐字「e2m1 … 4-bit floating point format with 2 bits for exponent and 1 bit for mantissa … does not support infinity and NaN」 | ✓ |
| MXFP8 per-32 E8M0 | OCP MX v1.0（inference 3.2 逐字引句）+ MX paper + PTX「ue8m0 … 8-bit unsigned … 8 bits for exponent and 0 bits for mantissa … NaN … 0xff」 | ✓ |
| NVFP4 per-16 + global FP32 结构 | PTX 组合表「.kind::mxf4nvf4 .e2m1 .ue4m3 .scale_vec::4X」+ block16 layout 节 + inference 3.2（cuBLAS VEC16_UE4M3 + SCALAR_32F） | ✓ 结构；**scale 类型 spec 误写 E4M3 → B-4** |
| UE4M3 位布局 | PTX 逐字「7-bit unsigned … 4 bits for exponent and 3 bits for mantissa … NaN … 0x7f … .b8 … MSB bit padded with zero」 | ✓ 一手；与 E4M3（signed 8-bit）不同 → B-4 证据 |
| NVFP4 ≠ OCP MX v1.0 | PTX kind::mxf4（ue8m0/block32）vs kind::mxf4nvf4（ue4m3/block16）分立 + inference 3.2 分源 | ✓ → B-5 |
| Marlin ~4× / batch 16–32 | 2408.11743 abstract + inference 3.3（3.88 精确口径） | ✓ |
| DeepGEMM SM90 FP32 SF / SM100 packed UE8M0 | DeepGEMM README 逐字（§6.2） | ✓ |
| cuBLAS 现行 docs 结构 | docs.nvidia.com/cuda/cublas/（1.02MB 抓取）：含「16/32/128-Element 1D Block Scaling for FP8 and FP4」节；VEC16_UE4M3 枚举在子页（本次未取），NVFP4 事实以 PTX + inference 3.2 已核实 cuBLAS 引句互证 | ✓（覆盖率注：cuBLAS 子页未逐字核，沿 inference 3.2 记录） |

### 6.4 数学 / 几何 / 边界（独立重算与文本核对）

| 抽查项 | 方法 / 来源 | 结论 |
| --- | --- | --- |
| 5.1 reduction Ω(n) 下界 + tree 达界 | 独立推导 + Python（n=8/1024/1048576） | ✓ PASS（§2.1） |
| 5.1 Blelloch work 2(n-1) / depth 2log2 n | 独立推导 + Python 模拟 n=8/16/1024 逐位校验 + GPU Gems 3 互证 | ✓ PASS（§2.2） |
| 5.2 HBM traffic O(MNK/T) | 独立推导 MNK(1/T_M+1/T_N) + Python 3 组 tile 数值 | ✓ PASS，假设注 B-6（§2.3） |
| 5.2 ↔ 4.3 强度式一致性 | I = 2T_M T_N/(T_M+T_N) 数值对账 | ✓ |
| 6.3 shared 预算式 | 独立推导 + Python 3 组几何 + models 4.1 §5.3 逐字对照 | ✗ K/V 少计一份 → **B-1**（§2.4） |
| 6.3 register 预算式 | per-thread 分摊推导 + 255 上限数值例 | 粒度不足 → B-7（§2.4） |
| 6.3 GPU 侧 HBM 流量 + IO 定理 recap 对账 | FA2 流量逐项推导 + Python（T=8192 例）+ 渐近对账 | ✓ PASS（§2.5） |
| 6.3 IO 定理 recap 不重推声明 | spec §2 L36 / §4.2 L110 / §6.3 L477 / 6.3 L323 四处核对 | ✓ 四处一致 |
| 6.3 inference 3.1 策略级 recap → 指令级增量 | 6.3 大纲 vs inference 3.1 §3（FA2/FA3 三技术策略表）逐件对照 | ✓ |
| MLA V4.1 512=448+64 / V3 576=512+64 | spec L329/L330/L99/L597 vs inference README X-1 专段（四方证据） | ✓ PASS（§3），无 S1 |
| 5.3 online softmax 重推导越界 | spec §2 L36 / §4.2 L110 / §6.3 清单 / risk 3 vs 5.3 L297–298 | ✗ → **B-2** |
| 符号专归 vs P5 大纲 | §4.3 L125–136 vs 5.1 L285 / 5.2 L292 / §4.2 L110；inference 3.1 §1 d_head 翻译先例 | ✗ → **B-3** |
| 6.1 NVFP4 scale 类型 | PTX 逐字 vs spec L311 vs inference 3.2 L70/L75 | ✗ → **B-4** |
| serving 调度越界 | P5/P6 八章大纲逐行扫（batching / 投机 / 引擎编排 / K8S 关键词） | ✓ 无越界 |
| 走读 scope 声明 | §4.2 L115 vs 八章大纲 | 5.2/6.3 ✓；6.2/6.4 ✗ → B-8 |
| FlashMLA 版本锚 | README 全文 vs 6.4 大纲 / §6.4 漂移清单 | ✗ → B-9 |

### 6.5 覆盖率声明

- **大纲覆盖**：P5 四章 + P6 四章共 8 章 × 4 行（内容 / 关键推导 / 锚点 / Sources）= 32 行全读全核（100%）；关联条款 §2 / §4.1（P5/P6 相关条）/ §4.2 / §4.3 / §4.4 / §6.3（5.1/5.2/6.3 条）/ §7 W3 / 附录预核实 / builder-check #1/#6 全核。
- **强制重推导清单（本路子集）**：5.1 reduction 下界、5.1 Blelloch、5.2 流量账、6.3 tile 预算、6.3 HBM 流量 = 5/5 独立重推导（100%），Python 数值断言全跑（scan 正确性 3 组、GEMM 3 组、FA 预算 3 组、FA 流量 1 组）。
- **事实核查**：arXiv ID 7/7 解析（FA2 abstract 沿已核实记录 + title 本次核）；repo 13/13 HTTP 确定码；格式事实 6/6（E4M3/E5M2/E2M1/E8M0/UE4M3/NVFP4 结构）一手或互证；Blelloch 出处 1/1；MLA 几何 4 位置全核。
- **recap 源**：models 4.1 全文（339 行）、inference 3.1 全文（350 行）读；inference 3.2/3.3 定点 grep（格式 / Marlin / DeepGEMM 锚）；inference README X-1 专段读。
- **未覆盖（延迟，非 findings）**：CUTLASS 当前版本架构细节、FA4 file:line、Blackwell datasheet 精确数值、Apex repo、cuBLAS 子页枚举逐字、Blelloch 报告 PDF 正文（以 CMU 摘要页 + GPU Gems 3 + 独立模拟三方互证替代）——均 ⟨verify⟩ 留 W3 research pass 或属他路。
- 估算：本路可核断言族 ~50，已决 ~44（≈88%），余为显式延迟项；无「未核却判无误」项。

## 7. 汇总

| severity | 数 | ID |
| --- | --- | --- |
| S1 | 0 | —（MLA 几何 PASS，无 X-1 重犯；无虚构数值 / 不存在 repo） |
| S2 | 4 | B-1（6.3 shared 预算 K/V 少计）、B-2（5.3 online softmax 重推导越界）、B-3（符号专归冲突 + IO 定理 recap 翻译缺口）、B-4（NVFP4 scale E4M3→UE4M3） |
| S3 | 5 | B-5（NVFP4 出处归属）、B-6（O(MNK/T) 假设注）、B-7（register 约束粒度）、B-8（6.2/6.4 片段 scope 声明）、B-9（FlashMLA 版本锚 + 漂移清单） |

修复优先级：B-1/B-4 在强制重推导 / 位布局表交付路径上，写作前必修；B-2/B-3 属 spec 内部纪律一致性，W0 修订期一并修；S3 五条随 W3 writer brief 落地。全部 findings 附一手证据，writer 反驳通道沿 spec §6.4。
