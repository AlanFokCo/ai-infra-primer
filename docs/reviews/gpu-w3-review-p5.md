# gpu W3 Part 5 对抗审查（T6）— kernel 计算原语 5.1–5.4

日期：2026-11-25。审查者：fresh-context adversarial evaluator（不继承写作上下文，未派 subagent，全部工作自做）。
对象：`gpu/05-kernel-compute-primitives/` 01-reduction-scan / 02-gemm-kernel-design / 03-softmax-layernorm-rmsnorm / 04-operator-fusion。
标准：STYLE_GUIDE.md + gpu spec §4（符号/证据/recap）§5 Part 5 §6.3（强制重推导）+ rubric + R2 style sheet（R2-1…12 + 习题分级）。
方法：独立 Python 重算全部数值账 + Blelloch/教学 kernel/warp shuffle/Welford/Chan 模拟；CUTLASS 4.9.0 fresh tarball（`/tmp/gpu-w3-NVIDIA_cutlass/cutlass-main`）file:line 逐锚核；DeepGEMM/FA2 tarball + TransformerEngine contents-API 缓存核；WebFetch/curl 一手官方文档（PG 13.5 / PTX 9.5 / BP 13.5 / cuBLASLt / cuda-samples / H100 datasheet / arXiv export API / Blelloch 报告页 / GitHub API）。

## 0. 摘要计数

| severity | 数 | 处置 |
| --- | --- | --- |
| S1（致命事实/公式错误） | **0** | — |
| S2（实质问题，必须修复） | **4** | S2-1 ridge 口径 / S2-2 符号 S 撞 / S2-3 reduce7 锚误归 / S2-4 rubric A.3 禁令词 |
| S3（minor，下波清零） | **7** | 见下表 |

**强制重推导**：5.1 reduction Ω(n_el) 下界 + Blelloch work O(n_el)/depth O(log n_el) = **PASS**；5.2 GEMM tile 流量 MNK(1/T_M+1/T_N) + MMA 利用率分解 = **PASS**。
**Recap 边界**：5.2 recap 4.3 tiling（不重推）= **PASS**；5.3 online softmax 正确性 recap models 4.1 §5.2（不重证）= **PASS**；5.4 recap models 4.1 IO 定理（不重推）+ inference 3.4 机制 = **PASS**（fusion IO 计数重列为 spec §5 5.4 关键推导「推导 + 实例」授权，见 verification log §C）。
无越界（serving 调度/引擎编排/K8S 零命中）。无 §6.3 强制 recap 清单内公式重推越界。

## 1. Findings 表

| # | sev | 位置 | 断言（问题） | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S2-1 | S2 | 5.1 §2 L19 | fp32 reduction 用 BF16 TC ridge 295 算 roofline 缺口「离 ridge 295 差约 1180 倍」，精度类错配 | reduction 是 fp32 CUDA-core 加法，适用 ridge = FP32 峰值/带宽 = 66.9/3.352 ≈ 20（1.2 §2 式：128×2×1.98 GHz×132 = 66.9 TFLOPS；5.3 §1 已正确立「fp32 ridge R ≈ 20」并对 fp32 softmax 用 20）。I = 0.25 对 fp32 ridge 20 缺口 ≈ 80×，非 1180×（1180 = 295/0.25 用的是 BF16 TC ridge）。2.4 ridge 表只列 BF16/FP8/FP4 TC 类，无 FP32 行；5.1 引「2.4 recap」的 295 套到 fp32 工作属口径错配。memory-bound 结论不受影响（0.25 ≪ 20 ≪ 295），但定量缺口虚高 14.75×，且与同 Part 5.3 不一致 | 改「离 fp32 ridge ≈ 20 差约 80 倍（1.2 §2 / 5.3 §1 口径）」；若保留 295 须显式注「对 BF16 TC ridge 的上界比较，非 fp32 执行单元 ridge」 |
| S2-2 | S2 | 5.4 §2 L30 / §3 L54 / 练习 3 L132 / 总结 L118,L120；符号声明 L11 缺 S | 字母 S 章内双义未声明：§2/练习 3 = tensor 字节（inference 3.4 语义），§3 = FA score 矩阵（models 4.1 语义）；章首符号声明只声明 S_sh，未声明 S，亦未作混场声明 | L30「residual+norm 未融合 5S…（S 为对应 tensor 字节）」；L54「score 矩阵 S 与概率矩阵 P」；L132「tensor 字节 S」；L11 符号声明含 R/R_th 混场声明却漏 S。spec §4.3 消歧纪律 + R2-7 要求同章双义字母章首声明（本章已对 R 做，对 S 漏）；S_ff 亦未在章首定义 | 符号声明补「S = 单 tensor 字节（§2/练习 3，沿 inference 3.4）；§3 S/P 为 FA score/概率矩阵（沿 models 4.1），两义分局部」混场注；或把 tensor 字节译作 S_tensor 避撞 |
| S2-3 | S2 | 5.1 练习 4(c) L247 | 「找出 reduce7 的 `tile32.shfl_down` 循环」函数/标识符误归 | fresh cuda-samples `reduction_kernel.cu`：reduce7（L448+）调 `warpReduceSum<T>(mask,mySum)`（L489/L504），其内为 `__shfl_down_sync` 循环（L75-81）；字面 `tile32.shfl_down` 循环在 reduce6（L436-438，及 reduce5 L351）。reduce7 体内无 `tile32.shfl_down`。校验锚「5 步、lane 0」对 reduce7 的 warpReduceSum 仍成立，但代码标识符归错函数 | 改「reduce7 的 warpReduceSum（`__shfl_down_sync`）循环」或「reduce6 的 `tile32.shfl_down` 循环」 |
| S2-4 | S2 | 5.3 §3 L115 | rubric A.3 语气松散禁令词「其实」命中 | L115「TE 的 per-warp 基例**其实**是 two-pass」。rubric A.3 硬禁「其实」（A 类命中即修）；前序 gpu-w2-review-p4 将「其实」列入 rubric A 禁令词扫描（该波零命中）。按 STYLE_GUIDE §10 S1 专指致命事实，本条属必修文风问题定 S2 | 删「其实」：「TE 的 per-warp 基例是 two-pass」 |
| S3-1 | S3 | 5.1 §3 ASCII 图 L60-66 | shuffle-down 归约图展示的是 shuffle 返回值语义，与本章 `v += __shfl_down_sync` 代码的累加态不符 | 模拟（x=[1..8]）：off=4 后 lane4-7 实持 2·x_i=[10,12,14,16]（图标 x4..x7=[5,6,7,8]）；off=2 后 lane2=x2+x6+2x4=20（图标 x2+x6=10），且「只有 lane 0-3 有效」实为只有 lane0-1 持干净部分和。lane0 轨迹 6→16→36 正确，最终结果正确；坑注 L73 已言高位 lane「加自己的旧值…偏大」，与图矛盾 | 图注「示意 shuffle 源值语义；`+=` 下高位 lane 实翻倍、仅 lane0 结果有效」，或改画 predicated 形式与代码对齐 |
| S3-2 | S3 | 5.1 §4 L108 | two-pass pass2「depth 约 12 步」与本章计数口径不符 | 256 线程归约 1024 partial（4/线程）：串行 4 元素 = 3 加法（沿 §2「64 元素 = 63 步」口径）+ warp 5 + block 3 = 11 步，非 12 | 改「depth 约 11 步」 |
| S3-3 | S3 | 5.3 §4 L154 | RMSNorm general `rsqrtf(sqsigma*rn + eps)`（L249）代码引文不确 | TE `rmsnorm_fwd_kernels.cuh` L248 `sqsigma = reducer.allreduce(sqsigma,sum)*rn`（rn 已折入），L249 实为 `rsqrtf(sqsigma + params.epsilon)`。章把 *rn 写进 rsqrtf 内，数学等价但非逐字代码 | 改「`rsqrtf(sqsigma + eps)`（rn 已于 L248 折入 sqsigma）」 |
| S3-4 | S3 | 5.2 §7 L140 / 参考 L223 | LinCombPerColBiasEltAct「注释逐字」行号归 L240，实在 L229 | fresh tarball `epilogue/fusion/operations.hpp`：注释「D = activation(alpha * acc + beta * C + per-column bias)」在 L229，struct LinCombPerColBiasEltAct 在 L240（注释属其声明块上方） | 行号改 L229（注释）/L240（struct），或注「L229 注释，L240 struct」 |
| S3-5 | S3 | 5.4 命题 2 L60 | 「带宽掉约 160 倍（1.2 §3 / 3.2 §1 比值 recap）」中 3.2 §1 指针不确 | 160× 带宽比在 1.2 §3（4055:25.4）与 2.3 §6（L135「约 160 倍落差」）；3.2 §1 给的是 16.5× **延迟**比（478.8/29.0），非 160× 带宽 | 指针改「1.2 §3 / 2.3 §6」 |
| S3-6 | S3 | 5.2 参考 L225 | process-leak「本轮再核」；「fresh tarball」为 W3 波内约定（边界） | 「2026-11-25 **本轮**再核」——「本轮」不见于任何已发布 W1/W2 章正文（仅 references.md 内部用），属生产轮次内部语；「fresh tarball」P5/P6/references 一致使用（W3 代码走读溯源约定），可接受但较 W1/W2「一手核实」更内部 | 「本轮再核」改「再核」或仅留日期；「fresh tarball」建议统一为「一手核实」（波内约定，低优先） |
| S3-7 | S3 | gpu/references.md 被引章节 | 多个共享锚点被引未回写 P5 章 | 实测被引字段缺：cuda-programming-guide（缺 5.1/5.3）、cuda-best-practices（缺 5.1/5.3/5.4）、hopper-microbench（缺 5.2）、ptx-isa（缺 5.3）、nvidia-datasheet（缺 5.3）、flashattention（缺 5.3/5.4）、transformerengine（缺 5.4）、cutlass（缺 5.4）。research §6 声称「共享锚点被引已含 P5/P6 维持」不全确。引用本身有效（条目存在、链接解析），仅被引元数据缺 | references.md 对应条目被引章节补 P5 章号 |

## 2. Verification log（强制）

### A. 强制重推导 5.1 — reduction 带宽最优性 + Blelloch scan（独立重算 + 模拟）

**命题 1（Ω(n_el) 读下界 + tree 达界）— PASS。**
- 下界反证逻辑核：未读元素 x_j → 改 x_j+δ 输出不变而真和变 δ → 矛盾；每元素至少读一次 → 读流量 ≥ n_el·bytes。论证有效（正确求和算法口径）。
- 达界构造核：grid-stride 每元素恰一次 HBM 读 + 片上 tree + n_blk partial 写 → 总流量 (n_el+n_blk)·bytes+O(1)，对 n_el 比值→1。有效。
- 数值独立重算：n_el=2^24, fp32：读 67108864 B = 67.11 MB ✓；67.11e6/3352e9 = 20.02 μs ✓；partial 1024×4 = 4096 B = 4 KB，占 0.0061% ✓。

**命题 2（work/depth）— PASS。** W = n_el−1 = 16777215 ✓；D = ceil(log2 n_el) = 24 ✓；串行 depth 16777215 ✓；depth 比 ≈ 70 万 ✓；层级 depth 63+5+3+10 = 81 ✓；Brent W/p+D = 16777215/262144+24 ≈ 64+24 = 88 ✓。depth 下界扇入翻倍论证（D 步覆盖 ≤ 2^D 输入）有效。

**命题 3（shuffle shared 流量）— PASS。** shuffle 方案 2·w_b·bytes = 2×8×4 = 64 B ✓；朴素 tree 256×4 + 3×4×(128+64+…+1) = 1024+3060 = 4084 B ✓；比 4084/64 = 63.8125 ≈ 63.8 ✓。

**命题 4（Blelloch work O(n_el)/depth O(log n_el)）— PASS（独立模拟）。**
- 独立实现 up-sweep（右+=左，步长 2^{d+1}）+ 置根 0 + down-sweep（左↔右：左=右旧、右=左旧+右旧）。
- n=8, x=[1..8]：输出 [0,1,3,6,10,15,21,28] = exclusive scan ✓；加法 14 = 2(n−1) ✓；交换 7 = n−1 ✓。
- 随机幂 2 长度 n=2,4,…,1024（200+ 例）：全部与 numpy 前缀和逐位一致，加法恒 2(n−1)、交换恒 n−1 ✓。
- 等比求和核：up 加法 Σ_{d=0}^{L−1} n/2^{d+1} = n(1−2^{−L}) = n−1 ✓；down 同；depth 2L = 48（n=2^24）✓；总加法 33554430 ✓。
- offset-doubling work n·log2 n = 4.03 亿 ✓；对 Blelloch 比 12 ✓。
- 章 §5 逐层数组状态（d=0/1/2 up，置根，d=2/1/0 down）独立复算**逐位吻合** ✓。
- inclusive 互转 z_i=y_i+x_i：[1,3,6,10,15,21,28,36] ✓，work n_el、depth 1 ✓。

**命题 5（正确性不变式）— PASS。** up 不变式（x[i+2^{d+1}−1] = 原数组 i 起 2^{d+1} 元素和）与 down 不变式（块边界 exclusive 前缀）经 8 元素逐层态 + 随机模拟验证成立。

**教学 kernel 模拟 — PASS。** 章 §5 Blelloch 教学代码（n=1024，up d=1..512 / down d=512..1 索引算术）独立重实现，对随机 float 数组与 numpy exclusive cumsum 最大绝对误差 2.8e−13 ✓。§3 warp/block 教学 kernel 逻辑核（warp_reduce_sum 5 步、block partial 8 字、warp0 读回 3 有效步）正确。

**精度账 — PASS（fp32 模拟）。** 2^24+2 个 1：串行 fp32 累加 = 16777216（末两 1 被吞）✓；pairwise tree = 16777218 = 真值 ✓。误差系数 tree 24×2^−24 = 1.43e−6 ✓，串行 (n−1)×2^−24 ≈ 1 ✓。

**应用账 — PASS。** compaction p=[1,0,1,1,0,0,1,0] scan=[0,1,1,2,3,3,3,4] ✓，选 x0/x2/x3/x6 写 0/1/2/3 ✓，长 4 ✓；选择率 25% 流量 83.89 MB / 25.03 μs ✓。scan HBM 2× = 134.22 MB / 40.04 μs ✓。

**warp shuffle 语义模拟。** 8-lane/32-lane down 归约 lane0 = 全和（36/528）✓；xor butterfly 每 lane 全和 ✓。**发现 S3-1**：高位 lane 在 `+=` 下翻倍，章图展示 shuffle 返回语义而非累加态。

### B. 强制重推导 5.2 — GEMM tile 流量 + MMA 利用率分解（独立重算）

**tile 流量 MNK(1/T_M+1/T_N)·bytes — PASS。** 独立计数：(M/T_M)(N/T_N) block × 每 block 沿 K 读 (T_M+T_N)K 元素 = MNK(1/T_M+1/T_N)。代入 4096³, T_M=T_N=128, bf16：2.147 GB ✓；640.7 μs ✓；I_blk = 2·128·128/(256·2) = 64 ✓；算力地板 137.4 GFLOP/989.5 TFLOPS = 138.9 μs ✓；ridge 989.5/3.352 = 295.2 ✓；缺口 R/I = 4.61 ✓。（此式为 4.3 式 (3) recap，章不重推，evaluator 独立重算确认。）

**命题 1（L2 波前复用因子 2b/(r+c)）— PASS。** 独立计数：b=rc block，无复用每 K 步 b·2T_tile·T_K 元素；完美命中 (r+c) 个 distinct panel → (r+c)T_tile·T_K；因子 2rc/(r+c)。方阵 r=c=√b → √b。b=132：方阵 √132 = 11.489 ✓，流量 186.9 MB ✓，强度 735.3 ✓；行优先 c=32, r=4.125 → 7.308 ✓，293.8 MB ✓，467.7 ✓。C 写 33.55 MB ✓；加 C 带宽时间 65.8 μs ✓。工作集 panel 2r×(128×4096×2) = 24.1 MB < L2 50 MB ✓；每 K 步 (r+c) tile ≈ 368 KB ✓。

**命题 2（每 tile wgmma 计数）— PASS。** (T_M/64)(T_N/128)(T_K/16) = 2×1×4 = 8 ✓；FLOPs 8×2×64×128×16 = 2097152 = 2×128×128×64 ✓。512 cycles/1.83 GHz = 279.8 ns = t_comp ✓（闭环）。累加器 2×(64×128/128) = 128 regs ✓。操作数 8×(2048+4096) = 49152 B ✓；96 B/cycle = 128 峰值 75% ✓。

**命题 3（MMA 利用率分解 U = U_geom·U_wave·U_main·U_epi）— PASS（telescoping 恒等式独立验证）。**
- 符号核：U = 2MNK/(n_SM·F_SM·n_wave·(t_main+t_epi))；四因子乘积 = [MN/(n_blk·T_M·T_N)]·[n_blk/(n_wave·n_SM)]·[t_comp/τ]·[t_main/(t_main+t_epi)]，用 t_main=(K/T_K)τ 与 t_comp=2T_M T_N T_K/F_SM 化简逐项相消 = U。**恒等式精确成立**（compute-bound / padded M=4000 / load-bound 三态数值验证 match < 1e−12；初测 harness 因 t_comp 舍入 279.8 出现 1e−4 偏差，用精确 t_comp=279.769 ns 后精确吻合——非章错）。
- 数值：n_blk=1024 ✓，n_wave=ceil(1024/132)=8 ✓，U_geom=1 ✓，U_wave=1024/1056=0.9697 ✓（末波 100 block，32 SM 空）✓，U_main=1（L2 修正）✓，U_epi=1/1.2=0.833 ✓，U=0.808 ✓，F_ach=799.6 TFLOPS ✓。无 L2：U_main=279.8/1290.4=0.217 ✓，U=0.175 ✓，173 TFLOPS ✓。DeepGEMM 1550/1979=0.783 ✓。

**pipeline / occupancy 账 — PASS。** t_comp=279.8 ns ✓；t_load=32768/(3352/132)e9=1290.4 ns（用未舍入每 SM 份额 25.394，沿 4.3 口径；声明值 25.4 为舍入显示）✓；t_lat=478.8/1.83=261.6 ns ✓。无 L2 τ_4=1290.4（装载主导）✓；L2 修正 t_load=112.3，τ_4=279.8（计算主导）✓；S_min=1+ceil((261.6+112.3)/279.8)=3 ✓。occupancy：shared 4×32768+1024=132096，floor(233472/132096)=1 ✓；register 128×168+128×40=26624，floor(65536/26624)=2 ✓；warp floor(64/8)=8 ✓；min=1 shared 绑定，O=8/64=0.125 ✓。Stages=3：floor(233472/99328)=2，2×26624=53248≤65536，O=0.25 ✓。

**split-K / cluster — PASS。** split-K partial 2(n_split−1)MN×4 = 2×3×16777216×4 = 402.7 MB > 186.9 ✓。cluster 2×1：MNK(1/128+1/256)×2 = 1.610 GB，省 25% ✓。

### C. Recap 边界审查 — PASS（全三章）

- **5.2 recap 4.3 tiling（不重推）= PASS。** §2 明写「先 recap 理论侧（4.3 §3，不重推）」，式 (2)(3) 仅陈述 + 代入实例化（I=64、2.147 GB），无强度/流量证明重推。增量命题 1（L2 波前修正）、命题 3（利用率分解）为真增量（4.3 未交付）。命题 2（wgmma 计数）为 1.3 几何实例化。4.4 命题 1 τ_S 式 (1)、S_min 式 (2) 均 recap + 指引（核 4.4 原文一致）。
- **5.3 online softmax 正确性 recap models 4.1 §5.2（不重证）= PASS。** §2 式 (2) 标「models 4.1 §5.2 式 (15) 已交付（本章 recap，不重证）」，recap 翻译注明「归纳证明见 models 4.1 §5.2，本章不复述」，无正确性归纳重证。增量 = 命题 1 访存账（two-pass/online/行驻留读写计数）+ warp/block 归约映射 + Welford/Chan（命题 2/3，models 4.1 未交付，合法新推导）。warp 合并「softmax 版 Chan」作为映射陈述，正确性仍指向 models 4.1。
- **5.4 recap models 4.1 IO 定理（不重推）+ inference 3.4 机制 = PASS（附注）。** §3 FA IO 定理 O(T²d_head²/M_SRAM) 明写「models 4.1 式 (17) recap…推导见该章 §5.4，本章不重推」✓。torch.compile/NanoFlow/TileLang 机制 recap inference 3.4 §4/§5/§7 + 指引 ✓。多输入算子账（SiLU+mul 5→3、residual+norm 5→4）recap inference 3.4 §2 ✓。
  - **附注（fusion IO 计数重列）**：5.4 命题 1 给完整读写计数证明（2(k−1)n_el·bytes），与 inference 3.4 命题 1（2(n−1)S）近乎同构（n→k、S→n_el·bytes 翻译）。**判为 spec 授权而非越界**：spec §5 5.4 内容明列「fusion 带宽节省账（…；**推导 + 实例**）」、关键推导列「fusion IO 节省 (k−1)×bytes」**无 recap-only 标记**，且与同句显式标 recap-only 的「FlashAttention fusion 视角 HBM 节省（recap models 4.1 IO 定理，不重推）」形成对照；§4.2/§6.3 强制 recap-only 清单不含 inference 3.4 fusion IO。章边界声明亦明写「按 spec 关键推导重列计数并实例化到 kernel 侧」，增量 = kernel 侧实例化（decode/prefill 表、(k−1) 中间 tensor 结构读法）+ register 预算（命题 2）+ 三类策略实体。故 fusion IO 计数重列合规，models 4.1 IO 定理与 inference 3.4 机制均正确 recap。**不标 S2。**

### D. CUTLASS 4.9.0 代码锚 spot-check（fresh tarball file:line，/tmp/gpu-w3-NVIDIA_cutlass/cutlass-main）

版本锚：version.h L41-43 MAJOR 4/MINOR 9/PATCH 0 ✓；CHANGELOG L5「[4.9.0] (2026-11-13)」✓。

| 锚（章引） | tarball 实测 | 结论 |
| --- | --- | --- |
| device gemm_universal.h L142/L289/L355/L398 | 2.x class L142、3.x 特化 L289、GemmKernel L355、initialize L398 | ✓ |
| kernel gemm_universal.hpp L42-57 IsCutlass3ArrayKernel | L42-57 分流 struct | ✓ |
| sm90_gemm_tma_warpspecialized.hpp class L65 | L65 | ✓ |
| SharedStorage union L120-138 | L120 struct，union TensorStorage（注释 non-persistent）| ✓ |
| NumLoad/NumMmaWarpGroups L140-141 = 1/1 | L140/141 | ✓ |
| MaxThreadsPerBlock L142 = 256 | L142 size(TiledMma)+NumLoad×NumThreadsPerWarpGroup | ✓ |
| operator() L269 | L269 | ✓ |
| WarpGroupRole/ProducerWarpRole | L284 Producer=0/Consumer=1，L288 MainloopEpilogue=0 | ✓ |
| roles L317-327 / transaction_bytes L326 | L318/321/326 | ✓ |
| blockIdx→M/N L406-407 | m_coord=idx2crd(blockIdx.x) L406、n_coord(blockIdx.y) L407 | ✓ |
| collective_epilogue L392 | L392 | ✓ |
| mainloop.load L436 / load_tail L450 / epi.load L455 | L436/450/455 | ✓ |
| partition_fragment_C L470 / mma L472 / mma_tail L483 / epi.store L496 / store_tail L510 | L470/472/483/496/510 | ✓ |
| make_producer_start_state + 「skip all waits…empty」 | L359 + 注释 L358 | ✓ |
| cooperative Load/MmaRegisterRequirement L150-151 = 40/232 | L150 `!HeavyRegisterPressure?40:24`、L151 `?232:240`（默认 40/232）| ✓（条件式，章取默认值） |
| reg_dealloc L562 / reg_alloc L772 | L562 dealloc<LoadRegisterRequirement>、L772 alloc<MmaRegisterRequirement> | ✓ |
| CollectiveMma L72 / DispatchPolicy L92 | L72/92 | ✓ |
| MainloopPipeline=PipelineTmaAsync L111 | L111 cutlass::PipelineTmaAsync<Stages> | ✓ |
| SmemLayoutA/B L130-134 | tile_to_shape L127-134（A L127/B L131 起）| ✓（范围锚，起始略宽） |
| Stages≥2 L137 / smem_desc assert L138-140 | L137 / L138-140 | ✓ |
| K_PIPE_MAX L263=Stages / K_PIPE_MMAS L264=1 | L263/264 | ✓ |
| TmaTransactionBytes L265-269 | L265 MK + L267 NK = L269 | ✓ |
| load L310 / producer_acquire L374 / TMA copy L384-385 / load_tail L396 | L310/374/384-385/396 | ✓（research §3.1 记 load L312，章 L310 正确） |
| mcast_mask_a/b L355-370 | mask 构造 L351-366（a 沿 N L359、b 沿 M L366）| ✓（范围锚） |
| mma L417 / consumer_wait L487 / warpgroup_arrive L490 / cute::gemm L496 / commit L500 | L417/487/490/496/500 | ✓ |
| prologue ScaleOut::Zero→One k_block 循环 L494-498 | L491 Zero、L494-498 循环、L497 One | ✓ |
| warpgroup_wait<K_PIPE_MMAS> L547 先于 consumer_release L551 | L547 wait / L551 release | ✓ |
| mma_tail L563 / warpgroup_wait<0> L571 | L563/571 | ✓ |
| static_assert (0<=K_PIPE_MMAS)&&(K_PIPE_MMAS<K_PIPE_MAX) | L473 | ✓ |
| epilogue CollectiveEpilogue L83 / EpilogueTile L68 / ThreadEpilogueOp L122 / 整除 assert L128-130 | L83/68/122/128-130 | ✓ |
| operations.hpp LinCombEltAct L143 / LinCombPerColBiasEltAct L240 | L143/240；注释「D = activation(alpha*acc+beta*C+per-column bias)」L229 | ✓ struct（注释行 S3-4） |
| cute/layout.hpp Shape L48 / Stride L51 / Layout L99 | L48/51/99 | ✓ |
| cute/atom/mma_atom.hpp MMA_Atom L49 / TiledMMA L211 / get_slice L359 / ThrMMA L460 / partition_A L478 / make_tiled_mma L531 | L49/211/359/460/478/531 | ✓ |
| cute/arch/mma_sm90_gmma.hpp m64n8k16 L158 / n16 L249 / n32 L341 / n64 L438 / bf16 m64n128k16 L2842 | L158/249/341/438/2842 | ✓ |
| mma_traits_sm90_gmma.hpp SM90_64x128x16_F32BF16BF16_SS L1716 | L1716 | ✓ |
| cute/swizzle.hpp L52「AA = ZZ xor YY」 | L52 | ✓ |
| sm90_pipeline.hpp PipelineState L171 / PipelineTmaAsync L289 / full-empty L299-300 / producer_acquire L443 | L171/289/299-300/443 | ✓ |
| sm100_pipeline.hpp PipelineTmaUmmaAsync L538（Blackwell，非 sm90） | L538 | ✓（漂移订正确） |
| sm90_tile_scheduler.hpp RasterOrder L47-48 / swizzle L105-121 | L48 / L105-121 | ✓ |
| sm90_tile_scheduler_group.hpp AlongM 默认 L134 | L134 | ✓ |
| fp8_accumulation.hpp L40-43 promote/scale 逐字 | L40-43 注释逐字吻合 | ✓ |
| efficient_gemm.md L16-32 嵌套循环 | cta_n/cta_m→cta_k(mainloop/stage)→warp→warp_k(k Group)→mma_k/n/m→mma_instruction | ✓ |

**结论：CUTLASS 全部 spot-check 锚（40+）命中，无虚构/错归。** 仅 S3-4（operations.hpp 注释行 L229 vs L240）与范围锚起始略宽（SmemLayout L127 vs L130、mcast L351 vs L355）属可接受范围锚。

### E. 其他 repo 代码锚（DeepGEMM / FA2 / TransformerEngine）

- **DeepGEMM**（tarball commit 5081f851）：sm90_fp8_gemm_1d2d.cuh「Accumulation for WGMMA or CUDA promotion」L251 ✓、accum/final_accum L254 ✓、BLOCK_K==128「Only support per-128-channel FP8 scaling」L56 ✓、「Promote with scales」L329 ✓、scale_0_0=scale_a_0*scale_b_0 L331 ✓、shifted_accum[i*4+0]+=(predicate?scale_0_0:scale_0_1)*accum L341-344 ✓。README DeepJIT L3 ✓、1550 TFLOPS H800 L25 ✓、SM90 FP32/SM100 packed UE8M0 L67-68 ✓。jit.hpp LazyInit L12/init_jit L14 ✓。scaling-factor-format.md gran_k SM90 仅 128 L29 ✓。
- **FA2**（tarball b038b497）：softmax.h quad_allreduce_ L39 ✓、scale_apply_exp2 L67 ✓、「Instead of computing exp(x - max)…exp2(x*log_2(e)」L79 ✓、exp2f(__fmul_rn(tensor,scale)-max_scaled) L86 ✓、softmax_rescale_o L137 ✓、scores_scale=exp2f((max_prev-max_cur)*scale_log2) L157 ✓、row_sum*= L158 ✓、acc_o_rowcol*=scores_scale L160 ✓。
- **TransformerEngine**（contents-API 缓存 2026-11-25，normalization/{layernorm,rmsnorm}）：ln_fwd_tuned_kernel L26 ✓、gamma/beta register L66-74 ✓、grid-stride L85 ✓、向量化 load+fp32 L86-98（compute_t L94）✓、stats.compute L100 ✓、rs=rsqrtf(rn*m2+eps) L109 ✓、mu/rs 写回 L105-113 ✓、normalize L121 ✓、zero_centered_gamma L123 ✓、gamma+beta L127 ✓、amax L129 ✓、fp8 scale L140 ✓、store L144-146 ✓、reduce_max+atomicMaxFloat L152-157 ✓；general L169、two-pass mean L246-255/var L258-270/rs L271 ✓。rmsnorm tuned L25、「we could optimize here to count mean square directly」L101 ✓、rsqrtf(rn*m2+mu*mu+eps) L102 ✓、general L161/Σx² L236-248/rs L249（S3-3）✓。utils.cuh parallel Welford L660/Chan L662 ✓、warp_chan_upd_dynamic L672-701（n_ab L687/delta L690/m2_ab L691/m_ab L692）✓、Stats<T,1,WARPS_M,1>::compute per-warp two-pass L844-864 ✓、warp 间 Chan L819/CTA 间 L758 ✓。rmsnorm_fwd_mxfp8.cu 存在（contents API 目录列）✓。路径漂移（layer_norm/rms_norm 404 → normalization）确 ✓。

### F. Web 一手事实核查（curl/WebFetch）

| 断言 | 一手来源 | 结论 |
| --- | --- | --- |
| PG「Warp shuffle functions exchange a value between non-exited threads within a warp without the use of shared memory.」 | PG 13.5 cpp-language-extensions（缓存 569 KB 实页）| ✓ 逐字 |
| __shfl_down_sync「calculates a source lane ID by adding delta to the caller's lane ID」+「the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will effectively remain unchanged」 | 同上 | ✓ 逐字（down 节，前缀 As for __shfl_up_sync）|
| __shfl_xor_sync「butterfly addressing pattern, which is used in tree reduction and broadcast」 | 同上 | ✓ 逐字 |
| 「If the target thread is inactive, the retrieved value is undefined」+「These intrinsics do not imply a memory barrier.」 | 同上 | ✓ 逐字 |
| BP Table 5 CC 9.0：warp shuffle 32 / SFU(lg2.approx) 16 / FP32 add 128 results/cycle/SM | BP 13.5 Table 5（HTML colspan 解析：shuffle 7.5=16, colspan6=32；SFU colspan7=16；add.f32 colspan2=64,colspan5=128）| ✓ |
| PTX redux.sync「performs a reduction operation .op of the 32 bit source register src across all non-exited threads in the membermask」+ .op={.add,.min,.max} .type={.u32,.s32}；.f32 仅 min/max（PTX 8.6，sm_100a）；PTX 7.0 sm_80+ | PTX 9.5（缓存 3.9 MB 实页）| ✓ 逐字（.f32 syntax `.op={.min,.max}`；Introduced 7.0，.f32 8.6，Requires sm_80，.f32 sm_100a）|
| cuBLASLt「The output is placed in heuristicResultsArray[] in the order of increasing estimated compute time.」 | cuBLASLt docs（缓存 2.5 MB）| ✓ 逐字 |
| heuristics cache「LRU-like eviction policy and is thread-safe」+ 默认 8192 条/每条至多 1024 B + CUBLASLT_HEURISTICS_CACHE_CAPACITY | 同上 | ✓ 逐字 |
| cuda-samples reduce6「(Brent's Theorem optimization)」+ work O(n)/step O(log n)；reduce2「no divergence or bank conflicts」 | reduction_kernel.cu L361-363 / L167 | ✓ 逐字 |
| reduction README「using shared memory, __shfl_down_sync, __reduce_add_sync and cooperative_groups reduce」；shfl_scan README「demonstrates how to use the shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block」 | contents API master 实时 fetch | ✓ 逐字 |
| threadFenceReduction / scan / reductionMultiBlockCG 路径 | GitHub contents API | ✓ 存在 |
| H100 SXM：HBM 3.35 TB/s、BF16 dense 989.5（sparse 1979/2）、FP8 dense 1979（sparse 3958/2）、FP32 67 | nvidia.com H100 页（实时 fetch，* With sparsity 脚注）| ✓ |
| Luo 2402.13499：H800 HBM(global) 478.8 cycles、L1 40.7/shared 29.0/L2 263.0；dense wgmma N=128 延迟 128.0、sparse SS 144.0 | arXiv HTML Table IV + wgmma 节（实时 fetch）| ✓ |
| NanoFlow 2408.12757：1.91×、50%–72% optimal、end-to-end compute bound | arXiv export API abstract | ✓ 逐字 |
| TileLang 2504.17577（Lei Wang）/ NanoFlow（Kan Zhu）/ Luo（Weile Luo）ID↔标题↔一作 | arXiv export API totalResults=3 | ✓ |
| Blelloch「Prefix Sums and Their Applications」CMU-CS-90-190 November 1990 | cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html（实时 fetch）| ✓ |
| NVIDIA/cub archived → cccl；openai/triton 301 → triton-lang | GitHub API（cub archived=True；openai/triton 301）| ✓ |
| PTX wgmma.mma_async.m64nNk32（f8f6f4） | PTX 9.5 缓存 | ✓ |

### G. 规范审查（R2-1…12 + rubric + 模板 + 习题 + 公式）

- **模板**：四章均含 metadata `> ` 行（定位|前置|锚点，L3）+ `## 总结`/`## 练习`/`## 参考` ✓。
- **习题分级**：四章均 [10]/[20]/[30]/[30]，难度非递减 ✓，≥1 [10]/[20] ✓，≤1 [40]（无 [40]）✓，每计算/代码题带校验锚 ✓，≥1 证明题闭环（5.1 ex3→命题 5、5.2 ex3→命题 3、5.3 ex3→命题 3、5.4 ex3→命题 1 fan-out 边界，均「留作练习 3」存在）✓。
- **R2-1 节首**：各 ## 节首段以设问/数值/元问题开场（「先回答一个更基本的问题」式），无「本节介绍/讲解/讨论」✓（5.1 §6/5.2 §5/§8 以具体事实/表格开场，合规）。
- **R2-2 章脊口径**：四章「本章口径（章脊锁定）」+ 换口径声明 ✓。
- **R2-3 数字实例**：公式后 300 字内全数字代入密度高 ✓。
- **R2-4/5 命题骨架 + 路线句**：命题/证明（路线）骨架齐，≥4 步证明有路线句 ✓。
- **R2-6 定义配例 + 反例**：scan exclusive/inclusive 正例 + compaction 反例、catastrophic cancellation 实例、fan-out 反例 ✓。
- **R2-7/8/9**：口径注、术语判断句（「所谓 X，是指」）、逻辑关节词 ✓。
- **R2-10/11 提点/坑**：5.1 提点 L51/坑 L73,L194；5.2 坑 L38；5.3 坑 L89,L117；5.4 提点 L66/坑 L68 ✓。
- **R2-12 总结回指**：四章总结表每行含回指列（式 N/§N/命题 N/练习 N）✓。
- **rubric A 禁令词**：命中「其实」1 处（S2-4）；其余（值得注意的是/综上所述/让我们/营销形容词/历史叙事）零命中 ✓。
- **公式白名单 / rescape**：`\, \; \: \! \{ \} \% \#` 零命中 ✓；`\lt` 用于公式内 <（5.1×1、5.3×2）✓；无 align/equation/宏定义 ✓。
- **flanking（开口/闭口）+ 公式内 bare <**：独立 checker（区分开/闭口 delimiter，保护 code span）四章 **0 命中** ✓（writer 已跑 fix_flanking）。
- **prose bare <**：四章 0 命中 ✓。
- **符号 §4.3**：n_el/n=rank、T_M/T_N/T_K/T_tile（禁裸 T）、GEMM M/N/K 章域声明、bytes 不占字母、B=batch、M_SRAM/d_head 翻译注 — 各章章首声明齐 ✓；**唯 5.4 S 双义未声明（S2-2）**。
- **证据等级**：[A]/[B]/[C]/教学参照标注齐，[C] 官方规格披露·一手权威 + 核实日期 ✓；Blelloch [B] 老经典无 arXiv ✓；无 [A/B] 混合档 ✓。
- **process-leak**：「spec §4.3」为全书约定（9 个已发布章同用）非 leak；「章脊锁定」R2 术语 ✓；「本轮再核」leak（S3-6）；「fresh tarball」W3 波内约定（S3-6 边界）。
- **越界**：serving 调度/引擎编排/K8S 零命中 ✓（NanoFlow/torch.compile 均 kernel 侧互补/recap + 指引，引擎机制归 inference）。

### H. 数值重算覆盖率

独立 Python 重算覆盖：5.1 全部命题数值（26 项）+ Blelloch/教学 kernel/shuffle/Welford/Chan 模拟 + fp32 精度例；5.2 全部命题数值（40+ 项）+ telescoping 三态；5.3 全部访存/SFU/强度/方差数值（24 项）+ fp32 朴素方差 −8.0 实算确认 + Welford 1.25 + Chan 42 + RMS 25.5；5.4 全部节省/预算/NanoFlow 数值（17 项）。**全部 PASS**（唯 t_load 1290.4 用未舍入份额 25.394 与声明 25.4 之别，沿 4.3 口径，非错；pass-2 depth 12 vs 11 见 S3-2）。fp32 朴素方差 [10001..10004] 实算 = **−8.0**（章断言确）✓，Welford = 1.25 ✓。

## 3. 结论

S1=0 / S2=4 / S3=7。S2 四条必须修复（S2-1 ridge 口径、S2-2 符号 S 撞、S2-3 reduce7 锚误归、S2-4「其实」禁令词）；S3 七条下波逐条判定（修复成本低，建议全修）。强制重推导 5.1/5.2 全 PASS，recap 边界全 PASS，CUTLASS 40+ 代码锚全命中，Web 一手事实全确。章节数值账与公式推导经独立重算/模拟无一错误；主要问题为局部口径/符号/锚归/文风，非致命事实错误。
