# 6.3 FlashAttention kernel 走读（FlashAttention Kernel Walkthrough）

> models 4.1 的 IO 定理给 $O(T^{2} d_{\mathrm{head}}^{2} / M_{\mathrm{SRAM}})$，可真实 FA2 kernel 的 Q 只读一遍、K/V 却被重读 $T/T_M$ 趟——定理的循环序与代码的循环序方向相反，账怎么对上？ $M_{\mathrm{SRAM}}$ 说「SRAM 越大流量越小」，但 128×128 的 tile 在 H100 上先撞的不是 shared 容量而是 255 寄存器上限——tile 到底能开多大？本章 recap models 4.1 的 online softmax 正确性与 IO 定理、inference 3.1 的并行策略（均不重推），交付指令级增量：tile↔shared/register 预算约束（命题 1，K/V 分计）、flash-attention repo 模块级走读（FA2 Q outer / gemm_rs / softmax_rescale_o / exp2，FA3 warp specialization / TMA / named barrier pingpong）、GPU 侧精确 HBM 流量账（命题 2，causal 跳过 + GQA 共享 + split-KV 修正，对账 IO 定理）、FA4 Blackwell tcgen05 与软件 exp | 前置：models 4.1 §5（FA 数学与 IO 定理 recap 源）、inference 3.1（split-KV/cascade/FA2/FA3 策略 recap 源）、5.3（online softmax 访存账与 exp2 锚）、5.2（warp specialization 与 pipeline 同构原型）、4.3（tiling 三预算）、4.4（cp.async/TMA/multi-stage）、1.3（mma.sync/wgmma/tcgen05 几何与累加器）、3.4（occupancy 四约束）、2.4（roofline 与带宽库存）；下游 6.4（MLA/GQA/paged/split-KV reduction 变体 kernel） | 锚点：FlashAttention（[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]）、FA2（[Dao, 2023](../references.md)，arXiv:2307.08691，[A]）、FA3（[Shah et al., 2024](../references.md)，arXiv:2407.08608，[A]）、FA4（[Zadouri et al., 2026](../references.md)，arXiv:2603.05451，[A]）、Dao-AILab/flash-attention repo b038b4974f26（FA2 csrc/flash_attn/src/ + FA3 hopper/ + FA4 flash_attn/cute/）、FlashMLA（[DeepSeek-AI, FlashMLA](../references.md)，教学参照 + 官方实现，双版本锚）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C] 官方规格披露·一手权威）——代码锚 2026-11-25 fresh tarball 一手核实

## 1. 定理给阶，谁给常数：本章口径与两层纪律

同一个 $T$ = 4096、 $d_{\mathrm{head}}$ = 128 的 causal attention，IO 定理说流量阶是 $O(T^{2} d_{\mathrm{head}}^{2} / M_{\mathrm{SRAM}})$，代入 H100 的 227 KB shared 得 $M_{\mathrm{SRAM}}$ = 116224 元素、 $T^{2} d_{\mathrm{head}}^{2} / M_{\mathrm{SRAM}} \approx 2.37 \times 10^{6}$ 元素；可 FA2 kernel 精确计数的 KV 流量（单头、非 causal）是 $2 T^{2} d_{\mathrm{head}} / T_M = 3.36 \times 10^{7}$ 元素——差 14.2 倍。定理错了吗？没错：定理假设 tile 受 SRAM 容量限制（ $B_c = \Theta(M_{\mathrm{SRAM}}/d)$），而 GPU 上累加器必须驻寄存器， $T_M$ = 128 先被 255 regs/thread 钉死，只有 $M_{\mathrm{SRAM}}/d$ = 908 的 1/7.1。**定理给阶，预算给可达 tile，两者合起来才给实际流量**——这条接缝就是本章的全部增量。

本章回答三个问题：tile 几何被 shared 与 register 两个预算怎样约束，上界是多少（命题 1）？FA2/FA3 把 online softmax 的每一步映射到哪些指令与同步原语（§4/§5 模块级走读）？GPU 侧精确 HBM 流量怎么计数，causal、GQA、split-KV 各修正哪一项，与 IO 定理怎么对账（命题 2）？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0（sm_90a），bf16 输入、fp32 累加**； $n_{\mathrm{SM}} = 132$、 $\mathrm{BW} = 3352$ GB/s、 $F_{\mathrm{peak}} = 989.5$ TFLOPS（BF16 dense）、ridge $R \approx 295$、每 SM 份额 $F_{\mathrm{SM}} = 7.496$ TFLOPS（2.4 recap）；每 SM shared 228 KB（233472 B）、单 block opt-in 上限 227 KB（232448 B）、register file 65536 个 32-bit 寄存器、255 regs/thread（2.3/3.4 recap）。running example：**$T$ = 4096、 $d_{\mathrm{head}}$ = 128、 $H_q$ = 32、 $H_{kv}$ = 8（GQA-4， $g = H_q/H_{kv} = 4$）、 $B$ = 1、causal**（Llama-3.1-8B 形态的 attention 几何，模型侧账归 models 4.1）。FA2 走读配置取 repo 的 hdim128 默认 tile： $T_M$ = 128、 $T_N$ = 64、4 warps（`flash_fwd_launch_template.h` 注释逐字「1st ones are good for H100, A100」——sm80 代码路径，A100/H100 通用；A100 shared 164 KB 更小，64 KB 配置两代都装得下，预算代入统一用 H100 库存）。FA3 走读配置取 hdim128 causal： $T_M = T_N$ = 128、 $n_{\mathrm{stage}}$ = 2、2 个 MMA warpgroup + 1 个 load warpgroup（`tile_size.h` L31-33）。**换口径声明**：§5 的 SFU/TC 比值账与 §7 的 FA4 对照用芯片级峰值比（B200 BF16 dense 2250 TFLOPS 对 H100 989.5，datasheet 口径；B200 SM 数官方未披露，不做每 SM 分解）；§6 split-KV 修正沿 inference 3.1 的 decode 口径但 $T$ 取本章 4096。片上 KB = 1024 B，GB/s 按 $10^{9}$ B/s。

符号声明（spec §4.3）： $T$ = 序列长；**$T_M, T_N$ = tile 维（Q tile 行数 / KV tile 行数），非序列长 $T$，禁裸 $T$ 表 tile**——models 4.1 的 $B_r/B_c$ 译为本章 $T_M/T_N$； $d_{\mathrm{head}}$ = head 维（models 4.1 FA 的 $d$ 翻译）； $B$ = batch； $H_q, H_{kv}$ = query/KV head 数， $g = H_q/H_{kv}$；**$M_{\mathrm{SRAM}}$ = 可用 SRAM 元素计（章首声明，单缓冲口径 $= S_{\mathrm{sh}}/\mathrm{bytes}$），非消息字节 $M$**； $S_{\mathrm{sh}}$ = shared bytes/block； $R_{\mathrm{th}}$ = registers/thread， $R_{\mathrm{file}}$ = 65536；混场声明： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread； $n_{\mathrm{stage}}$ = pipeline stage 数（4.3/4.4 口径，避免与 split 数 $S$ 冲突）； $S$ = split-KV 段数（沿 inference 3.1）； $\mathrm{bytes}$ = 每元素字节数（bf16 取 2，bytes 不占字母）； $m, \ell$ = running max / 指数和（沿 models 4.1）； $\mathrm{LSE}$ = log-sum-exp； $n_{\mathrm{th}}$ = block 内线程数。B/KB/MB 单位豁免。

**recap 边界**（spec §4.2 强制清单，不重推）：online softmax 正确性（running max/sum 增量更新 + 换基准恒等，models 4.1 §5.2 式 (15)(16) 的归纳证明）与 IO 复杂度定理（models 4.1 §5.4 式 (17)）本章只 recap + 翻译；split-KV/cascade 的段级归约正确性（inference 3.1 命题 1/2）与 FA2/FA3 并行策略（inference 3.1 §3）只 recap 结论与指针；standalone softmax 的访存账与 exp2 trick 机制（5.3 §2）recap；warp specialization 角色分离与 pipeline 同步骨架（5.2 §4/§5、4.4）recap 作同构原型。**本章增量 = tile 预算约束推导 + 指令级走读 + GPU 侧精确流量对账**。

**版本锚与代码 scope**：Dao-AILab/flash-attention commit **b038b4974f26**（main，2026-11-23），全部 file:line 按该版本 fresh tarball 核实（2026-11-25）。repo 含三代实现：FA2 在 `csrc/flash_attn/src/`（sm80 路径）、FA3 在 `hopper/`（sm90）、FA4 在 `flash_attn/cute/`（CuTeDSL，Hopper+Blackwell）。**代码 scope：模块走读级**（spec §4.2 四个主走读章之一）——FA2 前向 mainloop 全量 + FA3 mainloop/kernel 骨架；反向 kernel、dropout/softcap/alibi 分支、varlen 细节不在走读范围。FlashMLA/FlashInfer/vLLM 的变体 kernel 片段走读归 6.4，本章只在 §7 作对照；GEMM 模块全量走读归 5.2，本章重访其片段一律指引。

## 2. recap：FA 数学、IO 定理与并行策略

整行的 max 拿不到，还能不能做 softmax？数学侧已有答案（models 4.1 §5.2，本章 recap 不重证）：online softmax 把行归一化拆成 running max 与 running 指数和，每来一块换基准重缩放：

$$
m^{(j)} = \max\left(m^{(j-1)}, \max_{s \in B_j} x_s\right), \qquad \ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_{s \in B_j} e^{x_s - m^{(j)}} \qquad (1)
$$

$$
O^{(j)} = e^{m^{(j-1)} - m^{(j)}} O^{(j-1)} + \sum_{s \in B_j} e^{x_s - m^{(j)}} v_s \qquad (2)
$$

**符号与翻译注**：式 (1)(2) 即 models 4.1 式 (15)(16)， $B_j$ = 第 $j$ 块 logit， $x_s$ = score， $v_s$ = value 向量；该章已证不变量「 $m^{(j)}$ 是已见 logit 全局 max、 $\ell^{(j)}$ 与 $O^{(j)}$ 是以它为基准的指数和/加权和」，换基准因子 $e^{m_{\mathrm{old}} - m_{\mathrm{new}}} \le 1$ 是恒等变换——**FlashAttention 精确等价而非近似**，归纳证明见 models 4.1 §5.2，本章不复述。kernel 侧读法：式 (2) 的第一项就是 FA2 `softmax_rescale_o` 里 `acc_o *= scores_scale` 那一行（§4），第二项就是 PV GEMM 的累加——**一条数学恒等式拆成「一次逐元素乘 + 一次矩阵乘加」**，这是全部走读的对照基准。

IO 定理 recap（models 4.1 §5.4 式 (17)，不重推）：FA1 循环序（KV 外循环、Q 内循环）下 K/V 各读一遍、Q/O 每趟外循环重读/重写，代入 $B_c = \Theta(M/d)$ 得

$$
\mathrm{HBM\ traffic}_{\mathrm{FA}} = O\left( \frac{T^{2} d_{\mathrm{head}}^{2}}{M_{\mathrm{SRAM}}} \right) \qquad (3)
$$

**翻译注**：models 的 $d$（head 维）译 $d_{\mathrm{head}}$； $M$（可用 SRAM、元素计）译 $M_{\mathrm{SRAM}}$，章首单缓冲口径 $= S_{\mathrm{sh}}/\mathrm{bytes}$，禁混消息字节 $M$。代入本章口径： $M_{\mathrm{SRAM}} = 232448/2 = 116224$ 元素， $d_{\mathrm{head}}^{2}/M_{\mathrm{SRAM}} \approx 0.141$——定理说流量随 $T^{2}$ 涨、系数被 SRAM 容量压。但注意定理的两个假设：tile 只受 SRAM 限制、循环序是 KV outer。GPU 实现两条都改了（§3 寄存器绑定、§4 Q outer），对账留到 §6。

策略侧 recap（inference 3.1，不重推）：FA2 三技术——减 non-matmul FLOPs、seq 维跨 thread block 并行、warp 间 work partitioning（A100 50–73% 峰值）；FA3 三技术——warp specialization 异步重叠、pingpong 交错、FP8 block quant + incoherent processing（H100 FP16 740 TFLOPs/s、75% util，FP8 近 1.2 PFLOPs/s）；decode 侧 split-KV 把 KV 维变成并行维、cascade 把共享前缀的读取跨请求摊薄，正确性由段级归约命题保证（inference 3.1 命题 1/2）。本章把其中四条落到指令：non-matmul 削减 → §4 的延后归约与倒数乘；warp partitioning → §4 的 Q 行切分与 quad 归约；warp specialization/pingpong → §5 的角色分配与 named barrier 环；split-KV 的流量代价 → §6 的 $\eta$ 修正。策略到指令的映射就是「同一数学，改谁算哪块、何时算」。

## 3. tile 预算：shared 三份与 register 上限（命题 1）

一个 FA threadblock 要同时驻留哪些数据？Q tile（ $T_M \times d_{\mathrm{head}}$）、K tile（ $T_N \times d_{\mathrm{head}}$）、V tile（ $T_N \times d_{\mathrm{head}}$）——**K 与 V 是两份独立 tile**，虽然逻辑上成对出现，物理上各占各的 shared。单缓冲口径的 shared 预算：

$$
(T_M + 2 T_N) \cdot d_{\mathrm{head}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}} \qquad (4)
$$

**符号**： $T_M$ = Q tile 行数， $T_N$ = KV tile 行数（K、V 各一份 $T_N \times d_{\mathrm{head}}$）， $S_{\mathrm{sh}}$ = 单 block 可用 shared 字节（H100 opt-in 上限 232448 B）。代码锚一字不差：`kernel_traits.h` 的 `kSmemQSize = size(SmemLayoutQ) * sizeof(Element)`（L107）、`kSmemKVSize = size(SmemLayoutKV) * 2 * sizeof(Element)`（L108，**乘 2 就是 K/V 分计**）、`kSmemSize = kSmemQSize + kSmemKVSize`（L109）。代入走读配置： $(128 + 2 \times 64) \times 128 \times 2 = 65536$ B = 64 KB——只占 227 KB 的 28%，shared 远不是绑定约束。两个省容量的结构细节：V 的转置视图 `sVt` 是同一块 shared 的 layout 复合（`flash_fwd_kernel.h` L175、`kernel_traits.h` L93-95），**不复制第二份 V**；epilogue 的 O staging 复用 Q 的缓冲（L446 `make_tensor(sQ.data(), SmemLayoutO)`，注释「sO has the same size as sQ」）——Q 算完即废，缓冲转给 O，预算不加项。

multi-stage 流水把 K/V 缓冲乘份数（4.4 recap：Q 驻留单份，K/V 预取 $n_{\mathrm{stage}}$ 份）：

$$
(T_M + 2 \cdot n_{\mathrm{stage}} \cdot T_N) \cdot d_{\mathrm{head}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}} \qquad (5)
$$

FA3 走读配置代入： $(128 + 2 \times 2 \times 128) \times 128 \times 2 = 163840$ B = 160 KB，落在 227 KB 内、只剩 1 block/SM 的余量——`tile_size.h` L35 注释逐字「128 x 192 hits the limit of smem if MmaPV_is_RS」，说的就是式 (5) 触顶。FA3 的 SmemLayoutK/V 用 `tile_to_shape(..., Int<kStages>)` 把 stage 维铺进 layout（mainloop L135-166）， $n_{\mathrm{stage}}$ = 2 由 launch template 钉死（`flash_fwd_launch_template.h` L47）。

shared 松、register 紧。累加器必须驻寄存器（mma.sync/wgmma 的 D 操作数语义，1.3 recap），FA2 每线程的寄存器主项：

$$
R_{\mathrm{th}} \approx \underbrace{\frac{T_M (d_{\mathrm{head}} + T_N)}{n_{\mathrm{th}}}}_{\text{O + S accum, fp32}} + \underbrace{\frac{T_M T_N}{2 \cdot n_{\mathrm{th}}}}_{\text{P, bf16}} + R_{\mathrm{aux}} \le 255 \qquad (6)
$$

**符号**： $n_{\mathrm{th}}$ = block 线程数（FA2 走读配置 128）；acc_o 是 $T_M \times d_{\mathrm{head}}$ 的 fp32 输出累加器、acc_s 是 $T_M \times T_N$ 的 fp32 score 累加器，各元素摊到线程（每 fp32 占 1 寄存器）；P 是 acc_s 转 bf16 后的第二 GEMM A 操作数（每 2 元素占 1 寄存器）； $R_{\mathrm{aux}}$ = 行统计量与地址/谓词/循环变量（教学假设：row_max/row_sum 8 个 + 其余约 20 个，合计 28）。代入： $128 \times (128+64)/128 = 192$，P 项 $128 \times 64/(2 \times 128) = 32$，合计 $192 + 32 + 28 = 252$——**距 255 上限只剩 3 个寄存器**。

**命题 1**（tile 上界）。单缓冲、累加器驻寄存器的 FA kernel，可行 tile 区域由式 (4)(6) 与 occupancy 层约束（block 寄存器总量 × blocks/SM $\le R_{\mathrm{file}}$，3.4 式 (1) recap）共同决定；固定 $T_N$ 时 $T_M$ 的上界为

$$
T_M \le \min\left( \frac{S_{\mathrm{sh}}}{d_{\mathrm{head}} \cdot \mathrm{bytes}} - 2 T_N,\ \frac{(255 - R_{\mathrm{aux}}) \cdot n_{\mathrm{th}}}{d_{\mathrm{head}} + 1.5 T_N} \right) \qquad (7)
$$

且 $T_M$ 须为 MMA atom 铺砌的整数倍（FA2 TiledMma 沿 M 铺 $16 \times n_{\mathrm{warps}}$ = 64 行）。**证明**（路线：两个预算各解一个线性不等式，取交，两步）。其一，shared 侧：式 (4) 对 $T_M$ 线性，解出 $T_M \le S_{\mathrm{sh}}/(d_{\mathrm{head}} \cdot \mathrm{bytes}) - 2T_N$；其二，register 侧：式 (6) 合并同类项得 $T_M (d_{\mathrm{head}} + 1.5 T_N)/n_{\mathrm{th}} \le 255 - R_{\mathrm{aux}}$，解出第二项；两约束须同时满足取 min，atom 整除是几何约束（§4 TiledMma）。occupancy 层不收紧单 block 上界、只决定驻留 block 数（下段代入）。∎

全数字代入（ $T_N$ = 64、 $d_{\mathrm{head}}$ = 128、bf16、 $n_{\mathrm{th}}$ = 128）：shared 项给 $T_M \le 232448/256 - 128 = 780$；register 项给 $T_M \le (255-28) \times 128/(128+96) = 129.7$。**register 绑定， $T_M$ = 128 是上界内最大的 64 倍数**——下一档 192 需要 $192 \times 224/128 + 28 = 364$ 寄存器，超限 43%。这解释了 repo 注释里被注掉的候选：`run_mha_fwd_hdim128` 里 128×128、256×64 的 8-warp 配置全部注释掉（「Using 8 warps ... is 28% slower for seqlen=2k」），可行域里只剩 128×64 与 128×32 两档。occupancy 层（3.4 式 (1) 代入）：每 warp 寄存器按 256 粒度取整， $252 \times 32 = 8064 \to 8192$，block 总量 $4 \times 8192 = 32768$，register 项 $\lfloor 65536/32768 \rfloor = 2$ blocks；shared 项 $\lfloor 233472/(65536+1024) \rfloor = 3$；warp 项 16——取 min = **2 blocks/SM，register 绑定**， $\mathcal{O} = 8/64 = 0.125$。低 occupancy 可接受：mainloop 靠 cp.async 双缓冲与 ILP 藏延迟（3.4 命题 2 recap），不靠 warp 数。

> **坑**：把 shared 预算写成 $(T_M + T_N) \cdot d_{\mathrm{head}} \cdot \mathrm{bytes}$。触发条件：K/V 当成一份 tile 计；现象：预算低估一半， $T_N$ 选型偏大，编译期 shared 超限或运行期 launch 失败；规避：K、V 各一份（式 (4) 的系数 2，`kSmemKVSize` 的 `* 2` 逐字），multi-stage 再乘 $n_{\mathrm{stage}}$（式 (5)）。反向的坑：把 V 转置视图当第二份 V 计——`sVt` 与 `sV` 同址（L175），转置是 layout 不是数据。

FA3 的寄存器账换一种解法。2 个 MMA warpgroup 沿 M 切分 tile（AtomLayoutQK = `Shape<kBlockM/64,1,1>`，mainloop L88），每 WG 持 64 行：acc_o $64 \times 128/128 = 64$ regs、acc_s $64 \times 128/128 = 64$ regs、P 32 regs，合计 160——比 FA2 的 224 松，因为 tile 的 M 维被两个 WG 分摊。但 block 总量仍紧：producer WG 让出寄存器、consumer WG 多占（`flash_fwd_kernel_sm90.h` L82-83：`LoadRegisterRequirement` = 24、`MmaRegisterRequirement` = 240，2 WG + TMA 配置）， $128 \times 24 + 256 \times 240 = 64512 \le 65536$，**1 block/SM，寄存器与 shared 双双绑定**。寄存器再分配指令是 `setmaxnreg`（warpgroup 级 dealloc/alloc，5.2 §4 cooperative 变体同构）：producer 只发 TMA 不需要累加器，把寄存器让给 consumer 的 acc_o/acc_s——角色分离（§5）在寄存器预算上的兑现。

对照预告：MLA decode 的输出矩阵 64×512 fp32 = 32768 寄存器，恰为 $R_{\mathrm{file}}$ 的一半——两份输出（FA3 pingpong 的前提）放不下，FlashMLA 被迫单输出 seesaw（§7 对照、6.4 走读）。式 (6) 的约束在 $d_{\mathrm{head}}$ = 512 时从「紧」变「死」，这是变体 kernel 分岔的预算根源。

## 4. FA2 走读：Q 外循环与 online softmax 的指令级实现

FA2 的 grid 怎么铺？`dim3 grid(num_m_block, params.b, params.h)`（`flash_fwd_launch_template.h` L64）——blockIdx.x = Q 块号 m_block、y = batch、z = query head，每 block 算一个 $T_M \times d_{\mathrm{head}}$ 的输出 tile。代入章脊口径： $32 \times 1 \times 32 = 1024$ 个 CTA，对 132 个 SM 铺 7.8 波——**Q outer 把 seq 维变成并行维**（FA1 的 KV outer 下 grid 只有 $B \times H_q$，seq 维串行；inference 3.1 策略 recap 的指令级兑现）。入口 `compute_attn_1rowblock(params, bidb, bidh, m_block)`（`flash_fwd_kernel.h` L57），走读路线：causal 范围 → shared 布局 → 逆序 KV 循环 → 循环体五段 → epilogue。

**causal 范围先算**（L92-96）：`n_block_max = min(ceil(seqlen_k/kBlockN), ceil(((m_block+1)*kBlockM + seqlen_k - seqlen_q)/kBlockN))`——第 m_block 个 Q tile 只看得到对角线以下的 KV 块，上三角整块不算也不读（§6 流量修正的代码源头）。`n_block_min` 对 local attention 非零（滑窗左界）。空范围早退并写零（L103-136），防 OOB 读。

**shared 布局与装载**：sQ/sK/sV 三块连续排布（L169-174），swizzle atom `Swizzle<kSwizzle,3,3>`（4.2 recap，bank conflict 消除）；gmem→shared 用 `SM80_CP_ASYNC_CACHEGLOBAL`（`kernel_traits.h` L127，cp.async 16 B 向量，4.4 recap），注释逐字「We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading from the same address by the same threadblock」——同 block 不复读，绕 L1 省污染。prologue 装 Q 与第一个 K tile（L259-280），`cp_async_fence` 分组。

**KV 逆序内循环**，注释逐字（L140-142）：「We iterate over the blocks in reverse order. This is because the last block is the only one that needs masking when we read K and V from global memory.」——causal 下只有最右的 KV 块骑在对角线上需要逐元素 mask，逆序把 masking 迭代集中在循环开头，其余迭代免 mask 分支。代码分两段：`n_masking_steps = ceil_div(kBlockM, kBlockN)` = 2 个 masking 迭代（L307-311，`#pragma unroll` 展开）+ 无 mask 稳态循环 `for (; n_block >= n_block_min; --n_block)`（L387）。循环体五段（稳态段 L387-438）：

```
FA2 mainloop 每迭代（T_M=128, T_N=64, d_head=128, 4 warps）
────────────────────────────────────────────────────────────
① cp_async_wait<0> + __syncthreads        ← 等上一轮预取的 K 落地
② copy V[n_block] → sV; cp_async_fence    ← 发射本轮 V 预取（异步）
③ gemm(acc_s, Q, K)                       ← S = Q·Kᵀ，mma.sync m16n8k16
   mask.apply(acc_s)                      ← 仅 masking 迭代
④ cp_async_wait<0>; copy K[n_block-1]     ← 发射下一轮 K 预取
   softmax_rescale_o(acc_s, acc_o)        ← 式(1)(2) 的换基准（下详）
   rP = convert_type<bf16>(acc_s)         ← P fp32→bf16
   tOrP = convert_layout_acc_Aregs(rP)    ← C 布局 → A 布局（寄存器内）
⑤ gemm_rs(acc_o, tOrP, tOrVt)             ← O += P·V，A 来自寄存器
```

两个 GEMM 的操作数路径不对称，这是 FA kernel 与纯 GEMM（5.2）的结构差异。S GEMM 的 A（Q）B（K）都从 shared 经 ldmatrix 进寄存器（`SmemCopyAtom = SM75_U32x4_LDSM_N`，`kernel_traits.h` L40）；PV GEMM 走 **`gemm_rs`**（L376）——**A 操作数 P 直接来自寄存器**（register-source），B 操作数 V 从 shared 经转置 ldmatrix（`SM75_U16x8_LDSM_T`，L41）读入。为什么 P 不落 shared：P 出生在 acc_s 寄存器里（S GEMM 的 D），写 shared 再读回是纯浪费的往返；`convert_layout_acc_Aregs`（L374）在寄存器内把 mma.sync 的 C fragment 布局重排成 A fragment 布局——注释「Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)」，零数据移动、纯 layout 视图变换。V 则需要转置：PV 的 B 是 $V^{\top}$（ $d_{\mathrm{head}} \times T_N$），shared 里存的是 $V$（ $T_N \times d_{\mathrm{head}}$），ldmatrix.trans 在装载时硬件转置（4.2 recap），免独立转置 kernel。

**warp 分工**：TiledMma = `Layout<Shape<Int<kNWarps>,_1,_1>>`（`kernel_traits.h` L74-77）——4 个 warp 沿 M 维各持 16 行（`Tile<16*kNWarps,16,16>` 铺砌， $T_M$ = 128 需要 MMA_M = 2 趟），K/V 被全部 warp 共享。每 warp 算自己 Q 行的完整 score 行，行统计量不出 warp：m16n8k16 的 C fragment 里同一行由 lane%4 相同的 4 个线程持有，行归约 = 线程内列归约（`thread_reduce_`，softmax.h L24）+ **quad allreduce**（`quad_allreduce_` L39 → `Allreduce<4>`，utils.h L111-117，两条 `__shfl_xor_sync` offset 1/2）——warp 内免 shared（5.1 命题 3 同构）。对照 FA1 的 split-K 式分工（warp 切 KV、行统计量跨 warp 经 shared 归约），FA2 把 shared 通信从 mainloop 里整个删掉——这就是「warp 间 work partitioning」的指令级内容（[Dao, 2023](../references.md)，[A]）。mask 的行索引公式 `m_block*kBlockM + (tidx/32)*16 + (tidx%32)/4`（L338）正是这个 fragment 布局的坐标反解。

**online softmax 的指令级实现**（`softmax.h` `softmax_rescale_o` L137-167，式 (1)(2) 逐行对照）：

```cpp
// csrc/flash_attn/src/softmax.h L146-165（机制摘录，稳态分支）
Tensor scores_max_prev = make_fragment_like(row_max);
cute::copy(row_max, scores_max_prev);                 // 存旧基准 m_old
reduce_max</*zero_init=*/false>(scores, row_max);     // m_new = max(m_old, 块内 max)
float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur)
                           * softmax_scale_log2);     // L157: e^{m_old-m_new} 的 exp2 形
row_sum(mi) *= scores_scale;                          // L158: ℓ 换基准（式 (1) 第一项）
for (ni) acc_o_rowcol(mi, ni) *= scores_scale;        // L160: O 换基准（式 (2) 第一项）
scale_apply_exp2(scores, row_max, softmax_scale_log2);// L162: 新块 exp（式 (1)(2) 第二项的系数）
reduce_sum</*zero_init=*/false>(scores, row_sum);     // L165: ℓ 累加（线程内，跨线程延后）
```

三处指令级增量值得钉死。其一，**exp2 trick**（5.3 §2 recap，此处补全链路）：scale 在 host 侧预折 $\log_2 e$（`params.scale_softmax_log2` = $1/\sqrt{d_{\mathrm{head}}} \times \log_2 e$），运行期 `exp2f(x * scale - max_scaled)` 一条 FFMA + 一条 MUFU.EX2；注释逐字（L79-81）「This allows the compiler to use the ffma instruction instead of fadd and fmul separately」——换底数对齐 SFU 原生指令，还顺手把乘减融成单条 FFMA。rescale 因子同底（L157），全链路无自然 exp。其二，**跨线程 sum 归约延后**：L163-164 注释逐字「We don't do the reduce across threads here since we don't need to use the row_sum. We do that reduce at the end when we need to normalize the softmax.」——mainloop 里 $\ell$ 只做线程内累加，quad allreduce 推迟到 epilogue 的 `normalize_softmax_lse`（L172）一次性做；每迭代省 2 条 shfl × 行数，这是 FA2「减 non-matmul FLOPs」的第一笔。其三，**终除变倒数乘**：epilogue 算 `inv_sum = 1/sum`（L179，带 0/NaN 哨兵守卫）后 `acc_o *= inv_sum`（L183）——每元素一次除法变一次乘法， $\mathrm{LSE} = m \cdot \mathrm{scale} + \log \ell$（L180）顺带产出（split-KV/cascade 的状态分量，6.4）。第二笔 non-matmul 削减是结构性的：O 累加器全程驻寄存器、epilogue 才写回一次（FA1 每趟外循环写回 HBM），除法与 rescale 的总次数随块数线性、但每次都在寄存器上完成。

> **提点**：式 (2) 的两项在代码里分居两处——换基准乘在 `softmax_rescale_o`（CUDA core），新块加权和在 `gemm_rs`（Tensor Core）。online softmax 的「精确等价」因此不依赖任何特殊指令：它只是把一条恒等式拆给两条管线执行。FA3 的全部调度设计（§5）就是让这两条管线重叠起来。

## 5. FA3 走读：warp specialization、TMA 与 pingpong

FA2 在 H100 只跑到 35% util（inference 3.1 recap）——异步硬件（TMA、wgmma）被同步调度浪费。FA3 的解法分三层：角色分离、流水交接、warpgroup 间交错。走读配置：hdim128 causal， $T_M = T_N$ = 128、 $n_{\mathrm{stage}}$ = 2、2 MMA WG + 1 load WG = 384 线程（`tile_size.h` L31-33 返回 `{128, 128, true, true}`，第 3/4 元素是 MmaPV_is_RS 与 IntraWGOverlap，下详）。

**角色分离**（`flash_fwd_kernel_sm90.h`）：`operator()`（L179）按 `canonical_warp_group_idx()`（L208）分角色——`pipeline_params_k.role = warp_group_idx == 0 ? Producer : Consumer`（L220-222）：WG0 是 TMA producer，WG1/WG2 是 MMA consumer。producer 线程数可以少到一个 warp：`NumProducerThreads = NumThreadsPerWarp`（TMA 单线程发射语义，mainloop L121；4.4 §3 recap）。寄存器再分配（L82-83）：`LoadRegisterRequirement` = 24 / `MmaRegisterRequirement` = 240——§3 已算账，producer 让寄存器给 consumer 累加器。与 5.2 §4 CUTLASS base kernel 的同构关系：角色枚举、pipeline 参数注入、寄存器 dealloc/alloc 三件套一致；差异在 FA3 的 consumer 是 2 个 WG（GEMM 切 M）且 mainloop 里夹着 softmax。

**流水交接**：`PipelineTmaAsync = PipelineTmaAsyncNoCluster<kStages>`（cluster = 1 时）或 `PipelineTmaAsync<kStages>`（mainloop L283）——**是 cutlass::PipelineTmaAsync 系，不是 PipelineTmaUmmaAsync**（后者在 sm100_pipeline.hpp L538，Blackwell UMMA 专用，5.2 版本漂移注）。K/V 各一条 pipeline（L284-286），transaction bytes 按 tile 字节注入（L278-281：K tile $128 \times 128 \times 2 = 32768$ B，mbarrier expect_tx 按字节记账，4.4 recap）。producer 的 load_K lambda（L749-760）：`producer_acquire`（L750，等 empty barrier）→ `tma_load_K.with(barrier, mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST)`（L753）→ 硬件 complete_tx 记账。**EVICT_LAST 是 L2 逐出优先级提示**：K/V tile 会被同 KV head 的其他 Q 块 CTA 重读（§6），标「最后逐出」让 L2 尽量留住——2.2 缓存策略的 kernel 侧旋钮；对照 FlashMLA decode 用 EVICT_FIRST（每 KV block 只被一个 CTA 消费，流式数据先逐出，6.4 §2）。cluster 大于 1 时 KV 走 TMA multicast（`GmemTiledCopyKV = sm90_cluster_shape_to_tma_atom`，L209；一次 global 读落多个 CTA 的 shared，5.2 §5 同构）。

**warpgroup 内重叠（IntraWGOverlap）**：先钉一个容易读错的锚——**IntraWGOverlap 是 bool 模板参数**（mainloop L33），值由 `tile_size_fwd_sm90` 派生（`tile_size.h` L9-54，返回 tuple 第 4 元素；`flash_fwd_launch_template.h` L40 调用、L45 `std::get<3>` 提取后传入）。FP16 hdim128 取 true；FP8 headdim 大于 192 时取 `!paged_kv_non_TMA`（L51 注释逐字「PagedKV uses more registers so we disabled IntraWGOverlap」——paged gather 的地址状态吃寄存器，重叠窗口被迫关闭）。它启用的是 mainloop 的软件流水 `fwd_step`，注释逐字（L1199）：「Each step does gemm0 for iter n_block, gemm1 for iter n_block + 1, and softmax for iter n_block.」——每步发射 S-GEMM(n) 与 PV-GEMM(n+1) 两条 wgmma（`wg_wait=-1` 不等完成），然后做 block n 的 softmax；`warpgroup_wait<1>`（L1214）保证 S-GEMM(n) 完成后才读 acc_s，`warpgroup_wait<0>` 后才释放 V 缓冲（wgmma 异步读 shared，提前 release 会覆写在读 tile，5.2 §5 时序同构）。**Tensor Core 嚼 block n+1 的矩阵乘时，CUDA core/SFU 在做 block n 的 softmax**——式 (2) 两项的管线重叠在单 WG 内先做一层。

**warpgroup 间 pingpong**：两个 MMA WG 各持 64 行输出，若同时发 GEMM 则 TC 排队、若同时做 softmax 则 SFU 排队——交错才是最优。实化在 named barrier（`named_barrier.hpp` L50-58）：`FwdNamedBarriers` 枚举 `WarpSchedulerWG1 = 1 / WG2 = 2 / WG3 = 3` 与 `PFull = 6 / PEmpty = 7`。调度环（mainloop L921-937）：`warp_scheduler_barrier_sync()` 在发射 GEMM 对之前等自己的 barrier，`warp_scheduler_barrier_arrive()` 发射后 arrive 下一个 WG 的 barrier（2 WG 时 next = 1-cur，3 WG 时轮转）——**token 环强制两个 WG 的 TC 相位交替**，WG1 做 softmax 时 WG2 占 TC，反之亦然；`mma_init`（L955）给 WG1 发首个信号启动环。门控条件逐字（L352-355）：

```cpp
// hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp L351-355（逐字）
// These are tuned for speed. They don't affect correctness.
static constexpr bool UseSchedulerBarrier = (IntraWGOverlap
    ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128)
    : NumMmaWarpGroups == 2)
    && !LargeHeadDimV;
```

**这是 UseSchedulerBarrier 的条件，不是 IntraWGOverlap 的定义**（后者是模板参数，见上）——两个名字管两件事：IntraWGOverlap 管 WG 内 GEMM-softmax 流水，UseSchedulerBarrier 管 WG 间调度环是否启用。注释「tuned for speed, don't affect correctness」钉死性质：调度只改执行顺序，不改数学对象（式 (1)(2) 的恒等性与顺序无关）。PFull/PEmpty 是另一条 pipeline：MmaPV 走 SS（P 经 shared 传给另一个 WG，LargeHeadDimV 路径）时，写 P 的 WG arrive PFull、读 P 的 WG sync PFull 后 arrive PEmpty——P tile 的生产消费同步；走读配置 MmaPV_is_RS = true，P 驻寄存器，这条 pipeline 不启用。paper 的「ping-pong scheduling」在代码里没有字面 pingpong 命名，实化就是这个 named barrier 环 + 2 WG 交错（CUTLASS 侧的 pingpong 命名变体归 5.2 §4）。

**为什么必须重叠：SFU/TC 比值账**。每 KV tile 迭代（ $T_M = T_N$ = 128）：exp2 次数 $T_M T_N = 16384$，SFU 吞吐 16 results/cycle/SM（1.2 Table 5 recap）→ 1024 cycles = 559.6 ns @1.83 GHz；S GEMM FLOPs $2 \times 128 \times 128 \times 128 = 4.19$ MFLOP，每 SM 7.496 TFLOPS → 559.5 ns，PV 同值，TC 合计 1119 ns。**SFU 时间 = TC 时间的 50%**（rescale 乘、max/sum 归约、P 转换等 CUDA core 项为低阶，教学口径略）。若串行执行，利用率上限 $1119/(1119+559.6) \approx 0.667$（66.7%）——低于 FA3 报告的 74.8%（740/989.5，inference 3.1 recap）。差额只能来自重叠：pingpong 把 WG1 的 SFU 相位藏进 WG2 的 TC 相位，IntraWGOverlap 再藏一层。这是「调度收益」的定量下界：**不重叠，H100 上 FA 的 util 天花板就是 2/3**。

**FP8 路径**（算法级 recap inference 3.1：block quant + incoherent processing；指令级增量如下）：`Is_FP8 = float_e4m3_t || float_e5m2_t`（mainloop L44）；FP8 wgmma 的 B 操作数只支持 K-major，而 V 自然布局对 PV 是 MN-major，所以 `Transpose_V = Is_FP8 && !V_colmajor`（L55）触发 kernel 内转置（STSM 路径），且 `static_assert`「MmaPV must be RS if FP8」（L83）——P 必须寄存器源，省掉 P 的 shared 往返给转置腾带宽。descale 指针 q/k/v 各一份（L386），反量化乘在 score 上（格式位布局与 scale 粒度账归 6.1）。FP8 的 exp2 结果转 FP8 前还要 `permute_Cregs_fp8` / `permute_Aregs_fp8`（L1185/L1189）——FP8 MMA fragment 的寄存器序与 FP16 不同，转换时顺手重排。QK 侧 `MmaQK_is_RS = false` 钉死（L81），注释逐字「Register bandwidth is actually a bottleneck so we don't want Q to be in registers」——寄存器带宽也是预算（1.2 §3），Q 走 shared 描述符直喂。

调度收尾：persistent tile scheduler（`flash_fwd_launch_template.h` L63-76，`UsePersistentScheduler` L75）让 block 数 = SM 数、循环领 tile，causal 下按 LPT（最长处理时间优先）排序块号均衡各 SM 工作量——causal 的三角形工作量不均（§6）在调度层的对策。split-KV 的 combine kernel（`flash_fwd_combine_kernel.h`）把各 split 的 (O, LSE) 归约成最终输出，流量账归 §6、reduction kernel 指令走读归 6.4（FlashInfer 侧）。

## 6. GPU 侧精确 HBM 流量（命题 2）与定理对账

KV 到底被读了几遍？定理给阶，精确账给常数。FA2 前向的 HBM 流量按对象计数（无 L2 复用口径，沿 4.3 式 (3) 假设）：Q 读一遍、O 写一遍、LSE 写一遍，K/V 被每个 Q 块 CTA 重读——causal 下第 $i$ 个 Q 块读 $\lceil (i+1) T_M / T_N \rceil$ 个 KV 块（§4 的 n_block_max 公式）。

**命题 2**（FA2 前向精确流量）。设 $n_m = T/T_M$ 个 Q 块、causal、 $T$ 被 tile 整除，单个 query head 列的 KV 块装载次数为 $n_{\mathrm{load}} = \sum_{i=0}^{n_m-1} \lceil (i+1) T_M/T_N \rceil$；设 KV 读共享因子 $\sigma \in [1, g]$（ $\sigma = 1$ 无共享， $\sigma = g$ 完美共享），则全 kernel HBM 流量为

$$
\mathrm{traffic} = \frac{H_q}{\sigma} \cdot n_{\mathrm{load}} \cdot 2 T_N d_{\mathrm{head}} \cdot \mathrm{bytes} + 2 T H_q d_{\mathrm{head}} \cdot \mathrm{bytes} + 4 T H_q \qquad (8)
$$

**符号**：第一项 KV 读（系数 2 = K、V 各一份），第二项 Q 读 + O 写，第三项 LSE（每行一个 fp32）； $\sigma$ 的机制见下段。**证明**（路线：逐对象计数 + causal 块数求和 + 共享因子除，三步）。其一，每 CTA 的 KV 装载范围由 n_block_max 钉死（代码 L92-96），块数求和即 $n_{\mathrm{load}}$，每块 $2 T_N d_{\mathrm{head}}$ 元素；其二，Q/O/LSE 各一遍（Q outer 下无重读，O 驻寄存器到 epilogue）；其三， $H_q$ 个 query head 列中每 $g$ 列读同一份 KV（GQA），完美共享下 HBM 只付 $H_q/g = H_{kv}$ 列，故除 $\sigma$。完整证明（含非整除边界与 $\sigma$ 的成立条件）留作练习 3。∎

全数字代入（章脊口径， $n_m$ = 32、 $T_M/T_N$ = 2）： $n_{\mathrm{load}} = \sum_{i=0}^{31} 2(i+1) = 1056$；每列 KV = $1056 \times 2 \times 64 \times 128 \times 2 = 34.60$ MB。

| 口径 | KV 读 | Q 读 + O 写 + LSE | 合计 | 强度 $I$（FLOP/byte） | bound（ridge 295） |
| --- | --- | --- | --- | --- | --- |
| 无共享（ $\sigma = 1$） | 1107.30 MB | 67.63 MB | 1174.93 MB | 117 | memory-bound |
| GQA 完美共享（ $\sigma = g = 4$） | 276.82 MB | 67.63 MB | 344.46 MB | 399 | compute-bound |
| naive（S/P 物化，causal） | — | — | 2281.70 MB | 60 | memory-bound |

FLOPs 侧：causal attention $4 \times (T^{2}/2) \times d_{\mathrm{head}} \times H_q = 137.44$ GFLOP（两个 GEMM 各 $2 \times T^{2}/2 \times d_{\mathrm{head}} \times H_q$），算力地板 $137.44/989.5 \times 10^{3} = 138.9$ μs。带宽口径：共享 102.8 μs、无共享 350.5 μs——**GQA 共享一项就把 bound 从 memory 翻到 compute**（5.2 的 L2 翻转同构：流量假设换一档，roofline 位置换一区）。naive 对照：S/P 各一写一读 $4 \times (T^{2}/2) \times H_q \times 2$ B = 2147.48 MB，加 QKV/O 读写 134.22 MB，合 2281.70 MB——FA（共享口径）省 6.62 倍，且 FLOPs 一分未省（models 4.1 recap）。

$\sigma$ 的两个机制，各有代码实体。其一，**L2 共享**：同一 KV head 的 $g$ 个 query head 列读同一份 KV，tile scheduler 的 L2 swizzle 把它们排进相邻波前（`hopper/tile_scheduler.hpp` L261：swizzle 因子乘 `qhead_per_khead`；L693-699 显式计算 `nheads_in_l2`）——KV tile 在 L2 命中，HBM 只读一次（5.2 命题 1 的波前计数同法，FA 的波前是「同 KV head 的 $g$ 列 × 若干 m_block」）。其二，**PackGQA**：把 $g$ 个 query head 打包进 Q tile 的 M 维（`tile_scheduler.hpp` L108/L605：`seqlen *= qhead_per_khead`），一个 CTA 读一次 KV 服务 $g$ 个头——decode（每头 $q$ = 1）下从 $H_q$ 个 CTA 各读全量 KV 变成 $H_{kv}$ 个 CTA， $\sigma = g$ 精确成立（6.4 §3 走读）。causal prefill 下打包把 token 粒度变细（ $T_M/g$），KV 重读次数近似守恒（练习 3(d) 计数），收益主要在 decode 侧；prefill 的 $\sigma$ 介于 1 与 $g$ 之间，取决于 L2 命中率（实测归 11.2）。**式 (8) 的 $\sigma = g$ 是理想化上界口径**（教学假设：完美共享），无共享是下界口径，实测落在两者之间。

split-KV 修正（decode 侧，inference 3.1 式 (4) recap，不重推）： $S$ 段 partial state 写+读额外 $M_{\mathrm{red}} \approx 2 S H_q (4 d_{\mathrm{head}} + 4)$ 字节（fp32 partial O + LSE），占 KV 读的比例 $\eta = M_{\mathrm{red}}/(T \kappa)$。代入本章口径（ $S$ = 14、 $T$ = 4096、 $\kappa$ = 4096 B/token/层）： $M_{\mathrm{red}}$ = 451.5 KiB，KV 读 16 MiB， $\eta \approx 0.0276$（2.76%）——比 inference 3.1 满窗口径（ $T$ = 131072，0.086%）大 32 倍，因为 $\eta$ 反比于 $T$：**短 KV 切 split 的归约占比不可忽略**，这正是 scheduler 短序列不切的账侧理由（inference 3.1 §2 recap）。reduction kernel 的指令走读归 6.4。

**对账 IO 定理**，三句话。其一，**阶的恢复**：式 (8) 的 KV 项单头非 causal 为 $2 T^{2} d_{\mathrm{head}} / T_M$ 元素，代入 $T_M = \Theta(M_{\mathrm{SRAM}}/d_{\mathrm{head}})$（定理的 SRAM 绑定假设）即 $O(T^{2} d_{\mathrm{head}}^{2} / M_{\mathrm{SRAM}})$——定理是精确账在「tile 只受 SRAM 限制」假设下的推论，Q outer 换的是重读对象（K/V 换 Q/O），阶不变。其二，**常数的来源**：GPU 上 register 绑定（命题 1）给 $T_M$ = 128，只有 $M_{\mathrm{SRAM}}/d_{\mathrm{head}}$ = 908 的 1/7.1，K/V 两份再乘 2——精确 KV 项是直接代入定理表达式的 14.2 倍（§1 的缺口）。定理没有错，它的最优 tile 假设在 GPU 上不可达；**预算推导补上的正是「可达 tile」这一环**。其三，**结构差**：FA2 的 O 驻寄存器到 epilogue（写一遍），FA1 的 O 每趟外循环读改写（ $T/B_c$ 趟）——精确账里 FA2 的 Q/O 项是 $2 T d_{\mathrm{head}}$ 低阶，FA1 是 $3 T^{2} d_{\mathrm{head}}/B_c$ 高阶；加上并行度与 non-matmul 削减（§4），FA2 用同阶流量换执行效率。L2 跨 CTA 复用再把实际 KV 流量压到式 (8) 之下（5.2 命题 1 同法，方向注：本章给无 L2 上界，实测修正归 11.2）。

> **坑**：拿 IO 定理的 $M_{\mathrm{SRAM}}$ 直接代入预测实测流量。触发条件：用 227 KB shared 算 $M_{\mathrm{SRAM}}$ = 116224 再代 $T^{2} d^{2}/M$；现象：预测值比精确计数小一个数量级（本章 14.2 倍），因为定理假设的 tile（ $B_c \approx 908$）被寄存器预算否决；规避：流量预测用命题 2 的精确账 + 命题 1 的可达 tile，定理只用于阶的判断与「SRAM 越大越省」的方向性结论。

## 7. FA4 Blackwell：asymmetric scaling 与 tcgen05

H100 上 SFU 时间 = TC 时间的 50%（§5），Blackwell 把这个比值再翻倍——FA4 的出发点是一句硬件事实（[Zadouri et al., 2026](../references.md)，arXiv:2603.05451，[A]，摘要逐字）：「tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged」——**非对称缩放**：TC 吞吐翻倍，shared 带宽与指数单元慢涨或不涨。数字代入（芯片级峰值比，B200 BF16 dense 2250 对 H100 989.5 = 2.274×，datasheet 口径；B200 SM 数未披露，不做每 SM 分解）：SFU/TC 比值从 50% 涨到 $0.5 \times 2.274 \approx 1.137$（113.7%）——**softmax 仅 exp 一项就超过 TC 时间**，§5 的 pingpong 也藏不住（两条管线一样长，交错无空档可填）。FA3 的调度在 B200 上结构性失效，这是 FA4 存在的理由。

FA4 的两条对策（论文摘要口径）：(1) 全异步 MMA + 更大 tile——tcgen05 的累加器驻 TMEM（1.3 §6 recap：512 列 × 128 行 × 32-bit = 256 KB/CTA），**命题 1 的寄存器约束在 Blackwell 上解除**， $T_M$ 不再被 255 regs/thread 钉死，大 tile 抬 shared 侧复用（4.3 强度式同法）对冲 shared 带宽慢涨；(2) 软件仿真 exp（摘要逐字「software-emulated exponential」）+ 条件 rescale——把 exp 从 SFU 搬到 FP32 管线用多项式算。教学估算（口径：FP32 管线 128 results/cycle/SM 对 SFU 16，8× 吞吐差）：每元素 8 条 FFMA 的多项式占 $8/128 = 0.0625$ cycle，与 SFU 路径 $1/16 = 0.0625$ cycle 持平——TC 翻倍后 SFU 路径成了瓶颈，FMA 管线却还有 8× 余量可吸收这项工作；条件 rescale 则在 max 未变（scale = 1）时跳过 acc_o 的乘法，省 FP32 管线开销（多项式次数与条件判定的实现细节以论文为准，此处为量级估算）。

代码侧（repo b038b4974f26，`flash_attn/cute/`）：FA4 用 **CuTeDSL**（Python DSL 生成 kernel，CUTLASS 4.x 的 DSL 前端）写成，README L82 逐字「written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200)」——同一份算法描述跨两代硬件，代际差异落在 MMA atom 选择上：`blackwell_helpers.py` L8 `from cutlass.cute.nvgpu import tcgen05`，`_tcgen05_mma_kind`（L14-25）按格式分派 f16/tf32/i8/f8f6f4/mxf8f6f4/mxf4/mxf4nvf4——第 5 代 Tensor Core 的逐格式 MMA（1.3/6.1 recap）。前向 kernel 按架构分文件：`flash_fwd_sm90.py`（Hopper）/ `flash_fwd_sm100.py`（B200）/ `flash_fwd_sm120.py`（RTX Blackwell），另有 `flash_fwd_mla_sm100.py` / `flash_fwd_decode_sm100.py`（MLA 与 decode 变体，6.4 对照）。安装与接口：`pip install flash-attn-4`、`from flash_attn.cute import flash_attn_func`（README L86/L96）。

三代 overlap 方案并排收口：**FA3 pingpong 要两份输出矩阵**（两个 WG 各持一份 acc_o 交替），寄存器预算够（ $d_{\mathrm{head}}$ = 128 时每份 64 regs/thread）；**FlashMLA seesaw 只有一份输出**（64×512 = 32768 寄存器 = $R_{\mathrm{file}}$ 一半，两份放不下），竖切 O_L/O_R + 两个 KV block 跷跷板，deep-dive 逐字「mathematically equivalent to FlashAttention's online softmax algorithm」（6.4 §2 走读；FlashMLA 当前 main 与 open vs internal 差距归 inference 7.3）；**FA4 用 TMEM 把约束整个移除**，累加器不占寄存器，大 tile + 全异步 MMA 重新设计流水。三条路是同一道预算题（命题 1）在不同硬件/几何下的三个解——正确性都锚在式 (1)(2) 的恒等式上，变的只是执行顺序。

## 8. 边界：本章不做什么

走读收口，剩下的归谁？本章交付 tile 预算约束（命题 1 + §3）、FA2 指令级走读（§4）、FA3 warp specialization/TMA/pingpong（§5）、精确 HBM 流量与定理对账（命题 2 + §6）、FA4 Blackwell 机制（§7）；以下各有专章。MLA absorbed kernel 几何与 FlashMLA 双锚走读、GQA/PackGQA、paged gather、DSA indexer、split-KV reduction kernel 指令走读（FlashInfer MergeState/scheduler）归 6.4；格式位布局与逐格式吞吐归 6.1；量化 kernel（Marlin/DeepGEMM）归 6.2；GEMM 模块全量走读归 5.2；反向 kernel（重算 $S/P$ 的 backward mainloop）与 dropout/softcap 分支不在本章 scope；实测流量与利用率 drill-down 归 11.2。代码 scope：模块走读级（FA2 前向 mainloop 全量 + FA3 骨架 + FA4 机制锚）。

本章交付清单（下游 recap 接口）：命题 1（§3）、FA2 五段循环与 softmax 指令链（§4）、FA3 角色/barrier 环/SFU 比值账（§5）、命题 2 + 对账三句（§6）、FA4 非对称缩放（§7）。6.4 recap 命题 1 作 MLA seesaw 的预算根源、recap §6 的 $\eta$ 作 split-KV reduction 的流量入口、recap §5 的 EVICT_LAST 作 FlashMLA EVICT_FIRST 的对照；5.4 的 fusion 视角 recap 本章作「tiling + online softmax 融合免 $T^{2}$ 往返」的代码实体。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| online softmax 正确性与 IO 定理 recap models 4.1（不重证）；翻译 $d \to d_{\mathrm{head}}$、 $M \to M_{\mathrm{SRAM}} = S_{\mathrm{sh}}/\mathrm{bytes}$ = 116224 元素 | 式 (1)(2)(3) | §2 |
| shared 预算 K/V 分计： $(T_M + 2T_N) d_{\mathrm{head}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}}$；FA2 64 KB、FA3 multi-stage 160 KB；sVt 是视图、O 复用 Q 缓冲 | 式 (4)(5)，kSmemKVSize L108 | §3、练习 1 |
| 命题 1：register 绑定 $T_M \le 129.7 \to 128$（252/255 regs），shared 界 780 松；FA2 occupancy 2 blocks/SM、FA3 1 block（24/240 再分配，64512 regs） | 式 (6)(7) | §3、练习 1 |
| FA2 Q outer：grid = $n_m \times B \times H_q$ = 1024 CTA；逆序 KV、masking 2 迭代 + 稳态循环；gemm_rs P 寄存器源、V ldmatrix.trans | §4 走读 | §4、练习 4 |
| softmax 指令链：exp2f 换基准（L157）→ acc_o *=（L160）→ sum 延后归约（L163-164）→ epilogue 倒数乘 + LSE（L179-183）；quad Allreduce 2 条 shfl 免 shared | §4 摘录 | §4、练习 2 |
| FA3 角色 WG0 Producer / WG1-2 Consumer（L220-222）；PipelineTmaAsync（L283，非 TmaUmmaAsync）；TMA EVICT_LAST（L753） | §5 走读 | §5、练习 4 |
| IntraWGOverlap = 模板参数（L33，tile_size 派生）；UseSchedulerBarrier = L352-355 门控式；pingpong = WarpSchedulerWG1/2/3 token 环 + PFull/PEmpty（named_barrier L50-58） | §5 逐字 | §5、练习 4 |
| SFU/TC 比值 50%（16384 exp2 = 559.6 ns 对 TC 1119 ns）→ 串行天花板 66.7%，低于报告 74.8%——重叠是必需不是打磨 | §5 账 | §5、练习 2 |
| 命题 2：精确流量 = KV 重读（causal 1056 块/列）+ Q/O/LSE 一遍； $\sigma = 4$ 共享 344.46 MB（ $I$ = 399，compute-bound）对无共享 1174.93 MB（ $I$ = 117，memory-bound）；naive 2281.70 MB，省 6.62× | 式 (8) + §6 表 | §6、练习 3 |
| 对账三句： $T_M = \Theta(M_{\mathrm{SRAM}}/d)$ 恢复定理阶；register 绑定使常数 14.2×；FA2 免 O 读改写。split-KV $\eta$ = 2.76%（ $T$ = 4096，反比 $T$） | §6 对账段 | §6、练习 3 |
| FA4：TC 2.274× 而 SFU 慢涨 → 比值 113.7% → 软件 exp + 条件 rescale；tcgen05/TMEM 解除寄存器约束 → 大 tile；CuTeDSL 跨代（flash_attn/cute/） | §7 | §7 |
| 三代 overlap 三解：FA3 双输出 pingpong / FlashMLA 单输出 seesaw（32768 regs = $R_{\mathrm{file}}$/2）/ FA4 TMEM；正确性同锚式 (1)(2) | §7 收口 | §7、6.4 |

## 练习

1.（计算 · [10]）tile 预算代入（FA2 走读配置 $T_M$ = 128、 $T_N$ = 64、 $d_{\mathrm{head}}$ = 128、bf16、4 warps）：(a) 单缓冲 shared 字节（式 (4)），K/V 分计；(b) acc_o、acc_s、P 各占多少 regs/thread，加 $R_{\mathrm{aux}}$ = 28 后对 255 的余量；(c) 若 $T_N$ 改 128，register 项是否超限；(d) occupancy 四约束取 min（3.4 式 (1)，每 warp 寄存器按 256 粒度取整）。校验：(a) 65536 B = 64 KB（§3）；(b) 128/64/32，余量 3（式 (6)）；(c) $128 \times (128+192)/128 + 28 = 348$ 超限；(d) 2 blocks/SM、 $\mathcal{O}$ = 0.125（§3）。
2.（计算 · [20]）SFU/TC 比值与 non-matmul 账：(a) FA3 走读配置每 KV tile 的 exp2 次数、SFU cycles 与 ns（16 results/cycle/SM、1.83 GHz）；(b) S/PV 两个 GEMM 的 TC 时间（每 SM 7.496 TFLOPS），SFU/TC 比值与串行利用率天花板；(c) 对照 FA3 报告 740 TFLOPs/s（74.8%），先预测「不重叠能否达到」再算；(d) B200 峰值比 2.274× 下比值变为多少，判断 pingpong 是否还藏得住。校验：(a) 16384、1024 cycles、559.6 ns（§5）；(b) 1119 ns、50%、66.7%；(c) 不能（66.7% 低于 74.8%）；(d) 113.7%，SFU 反超 TC（§7）。
3.（推导 · [30]）补全命题 2 的证明并对账定理：(a) 由 n_block_max 公式（L92-96）推出 causal 块数 $n_{\mathrm{load}} = \sum_{i} \lceil (i+1) T_M/T_N \rceil$，代入章脊口径验证 1056；(b) 逐对象计数推出式 (8)（含 LSE 项），说明 $\sigma = g$ 的成立条件（PackGQA decode 精确 / prefill L2 理想化）；(c) 证明 KV 项在 $T_M = \Theta(M_{\mathrm{SRAM}}/d_{\mathrm{head}})$ 假设下恢复定理阶 $O(T^{2} d_{\mathrm{head}}^{2}/M_{\mathrm{SRAM}})$，并算出 register 绑定（ $T_M$ = 128）下的常数比 14.2；(d) PackGQA causal prefill 计数： $g$ 头打包后每 KV head 的装载次数 $\sum_{i=0}^{127} \lceil (i+1)/2 \rceil = 4160$ 对不打包 $4 \times 1056 = 4224$，解释「近似守恒」与 decode 侧除以 $g$ 的差异。校验：(a) 1056 = 2×32×33/2（§6）；(c) $2 \times 908/128 = 14.19$（§1/§6）；(d) 4160/4224 = 0.985（§6 $\sigma$ 段）。
4.（代码 · [30]）FA2/FA3 走读（版本锚 b038b4974f26）：先写出预测再对照源码——(a) FA2 的 KV 循环为什么逆序？masking 迭代数由什么决定？(b) `softmax_rescale_o` 里跨线程 sum 归约在哪一步做，为什么 mainloop 里不做？(c) FA3 的 IntraWGOverlap 与 UseSchedulerBarrier 各是什么（模板参数还是计算式）、各管哪层重叠？(d) pingpong 的 token 环用哪些 named barrier，首个信号发给谁？校验：(a) L140-142 注释（末块唯一需 mask）+ L307-309（ceil(kBlockM/kBlockN)）；(b) epilogue normalize_softmax_lse L172（L163-164 注释逐字）；(c) L33 模板参数 + tile_size.h 派生 / L352-355 门控式（勿混，§5）；(d) WarpSchedulerWG1/2/3（named_barrier.hpp L52-54），mma_init L955 给 WG1。

## 参考

- [Dao et al., 2022](../references.md)：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，arXiv:2205.14135，[A]（key: flashattention）——online softmax 与 IO 定理的数学源头归 models 4.1 §5（本章 recap 不重推）；repo b038b4974f26（2026-11-25 fresh tarball 核实）：csrc/flash_attn/src/flash_fwd_kernel.h（compute_attn_1rowblock L57、n_block_min/max L92-96、逆序注释 L140-142、sQ/sK/sV L169-176、sVt 视图 L175、prologue L259-280、masking 循环 L307-311、softmax_rescale_o 调用 L352-353、convert_layout_acc_Aregs L374、gemm_rs L376、稳态循环 L387、epilogue sO 复用 L446、normalize L442）；softmax.h（thread_reduce_ L24、quad_allreduce_ L39、scale_apply_exp2 L67、FFMA 注释 L79-81、exp2f L86、softmax_rescale_o L137、scores_scale L157、row_sum L158、acc_o L160、延后注释 L163-164、normalize_softmax_lse L170/L172/L179-183）；kernel_traits.h（Flash_kernel_traits L15、MMA atom L30-37、SmemCopyAtom L40-41、kNWarps/kNThreads L63-64、TiledMma L74-77、SmemLayoutKV L88-90、Vtransposed L93-95、kSmemQSize/KVSize/Size L107-109、CP_ASYNC_CACHEGLOBAL L127）；utils.h Allreduce L111-117；flash_fwd_launch_template.h（grid L64、hdim128 配置与注释）
- [Dao, 2023](../references.md)：FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning，arXiv:2307.08691，[A]（key: flashattention2）——Q outer / warp partitioning / non-matmul 削减的策略源头（§4 指令级兑现）；A100 50–73% 峰值报告口径
- [Shah et al., 2024](../references.md)：FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision，arXiv:2407.08608，[A]（key: flashattention3）——warp specialization / pingpong / FP8 策略源头（§5 指令级兑现）；hopper/ 代码锚：mainloop_fwd_sm90_tma_gmma_ws.hpp（模板参数 IntraWGOverlap L33、Is_FP8 L44、Transpose_V L55、MmaQK_is_RS L81、RS assert L83、AtomLayoutQK L88、NumProducerThreads L121、SmemLayout kStages L135-166、GmemTiledCopyQ/KV L208-209、TmaTransactionBytes L278-281、PipelineTmaAsync L283-289、UseSchedulerBarrier L352-355、RescaleOBeforeGemm L356、load_K L749-760/EVICT_LAST L753、warp_scheduler_barrier_sync/arrive L921-937、mma_init L955、fwd_step 注释 L1199、consumer 循环 L1205-1236）；flash_fwd_kernel_sm90.h（NumMmaWarpGroups L75、Load/MmaRegisterRequirement L82-83、operator() L179、warp_group_idx L208、role L220-222）；named_barrier.hpp（FwdNamedBarriers L50-58）；tile_size.h（tile_size_fwd_sm90 L9-54、hdim128 L31-33、smem 注释 L35、FP8 PagedKV L51）；flash_fwd_launch_template.h（L40/L45 提取、kStages L47、TileScheduler L63-76）；tile_scheduler.hpp（L2 swizzle L261、PackGQA seqlen L108/L605、nheads_in_l2 L693-699）；flash_fwd_combine_kernel.h（split-KV combine）
- [Zadouri et al., 2026](../references.md)：FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling，arXiv:2603.05451，[A]（key: flashattention4）——非对称缩放摘要逐字（§7）；flash_attn/cute/ 代码锚：README L80-99（CuTeDSL、pip install flash-attn-4、flash_attn_func）、blackwell_helpers.py（tcgen05 import L8、_tcgen05_mma_kind L14-25）、flash_fwd_sm90.py / flash_fwd_sm100.py / flash_fwd_sm120.py / flash_fwd_mla_sm100.py / flash_fwd_decode_sm100.py
- [DeepSeek-AI, FlashMLA](../references.md)：deepseek-ai/FlashMLA repo（key: flashmla），教学参照 + 官方实现，双版本锚 main 395cb3afbb33 + ba89a3466e9470ad08ab39738d4e7bb66989e1e7（2026-11-25）——§7 seesaw 对照（64×512 = 32768 regs、deep-dive L21/L42 逐字），详走读归 6.4
- [Ye et al., 2025](../references.md)：FlashInfer，arXiv:2501.01005，[A]（key: flashinfer）——split-KV/cascade 策略与 (O, LSE) 状态语义（inference 3.1 recap 源之一），reduction kernel 走读归 6.4
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——mma.sync m16n8k16 / wgmma m64nNk16（FP8 m64nNk32 kind::f8f6f4）/ setmaxnreg / bar.arrive named barrier / cp.async / TMA cache hint（1.3/4.4 recap 源）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100/H200/Blackwell datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM BF16 dense 989.5 TFLOPS / 3.35 TB/s、HGX B200 BF16 dense 2250 TFLOPS（§7 芯片级峰值比口径）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide 13.5（key: cuda-best-practices），[C] 官方规格披露·一手权威——Table 5 SFU 16 / FP32 128 results/cycle/SM（§5/§7 比值账）、寄存器 256/warp 粒度（§3 occupancy）
- 跨维度指引（纯文字）：models 4.1 §5.2（online softmax 式 (15)(16) 正确性证明）、§5.3（FA1 tiling 与 $B_r/B_c$）、§5.4（IO 定理式 (17)）；inference 3.1 §2（split-KV 式 (4) 与 $\eta$）、§3（FA2/FA3 策略与报告数字）、§4（cascade/plan-run）；gpu 5.3 §2（exp2 trick 与 SFU 地板）、5.2 §4/§5（warp specialization 与 pipeline 同构）、4.3（三预算）、4.4（cp.async/TMA/multi-stage）、1.3（MMA 几何与 TMEM）、3.4（occupancy 四约束）、2.4（roofline 库存）
