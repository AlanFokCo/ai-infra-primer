# 3.4 Occupancy 与资源约束（Occupancy and Resource Constraints）

> 同一个 kernel，每线程多用一个 register，occupancy 从 50% 掉到 25%——一定更慢吗？compute-bound kernel 只要 6.25% occupancy 就能喂满 FP32 管线，memory-bound kernel 却要 81%——「够用的 occupancy」怎么从延迟隐藏账反解？本章交付编程模型侧 occupancy 精确计算（分配粒度 + cudaOccupancy API）、register pressure 与 ILP 的权衡（launch_bounds 寄存器上限命题）、latency hiding 所需最小 occupancy（recap 1.4 Little's law）、compute-bound 低 occupancy 满吞吐命题、shared carveout 配置与 NCU Occupancy Calculator 读法 | 前置：1.2（资源库存、occupancy 物理上界式 (3)-(5)、命题 3 粒度上界性、228/227/256 口径）、1.4（Little's law、208 sector / 52 warp load、ILP-TLP 乘积替代）、3.1（波次量化、running example）、3.2（local/spill 硬件 backing）、2.4（roofline 与 ridge）；下游 4.3（tile 预算）、5.2（GEMM register 分摊）、6.3（FA tile 约束）、11.2（occupancy 实测） | 锚点：CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C]）、Nsight Compute（[NVIDIA, Nsight Compute](../references.md)，[C]）、Hopper/Blackwell Tuning Guide（[NVIDIA, Hopper tuning guide](../references.md)、[NVIDIA, Blackwell tuning guide](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 延迟实测口径）——官方规格 2026-11-25 一手核实

## 1. 50% 对 25%：occupancy 不是越高越好，也不是越低越好

每 block 512 线程、每线程 64 个 register 的 kernel 能驻留 2 个 block（occupancy 50%）；编译器多用一个 register（65 个），驻留 block 数掉到 1（occupancy 25%）——这是官方文档里的真实悬崖（§3）。但 25% 一定比 50% 慢吗？不一定：若 kernel 是 compute-bound 且 ILP 充足，6.25% occupancy 就能喂满 FP32 管线（§4 命题 2）。本章回答三个问题：给定 kernel 的 register/shared 用量与 block 尺寸，编程模型侧怎么**精确**算出 occupancy（1.2 的物理上界往下修哪些项）？register 预算怎么在 ILP 与 occupancy 之间分配，launch_bounds 怎么控制？「够用的 occupancy」怎么从延迟隐藏账反解，NCU 又怎么读出绑定约束？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 1.2/3.1）；**CC 10.0 作官方算例对照**——PG §2.3.7 的算例给在 CC 10.0 上，两代资源库存逐行相同（1.2 §1），故算例对 CC 9.0 同形，换口径原因：官方算例出处；**CC 6.x / 7.0 作 register 粒度算例**——官方 register 悬崖与分配粒度算例给在这两代上，机制跨代同形、具体粒度数值随架构变（BP 逐字，§3），不代入 CC 9.0。running example 两个：3.1 的 elementwise kernel（256 threads/block、 $R_{\mathrm{th}} = 32$、 $S_{\mathrm{sh}} = 0$， $\mathcal{O} = 1.0$，沿 3.1 章脊教学假设）作满 occupancy 基线；教学 kernel K2（256 threads/block、 $R_{\mathrm{th}} = 96$、 $S_{\mathrm{sh}} = 32$ KB，1.4 练习 4 口径）作 register 绑定案例。cycle ↔ 时间换算取 1.83 GHz（沿 1.4）。

符号声明（spec §4.3，**混场章声明： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread**，两者勿混）： $n_b$ = 每 SM resident blocks（沿 1.2）； $B_{\max}$ = 1.2 式 (3) 理想上界（recap）； $b_{\mathrm{act}}$ = 编程模型侧精确 resident blocks（本章推导量）； $B_{\max,\mathrm{slot}}$ = block slot 上限（32）； $W_{\mathrm{slots}}$ = warp slots（64）； $w_b$ = warps/block； $W_{\mathrm{active}}$ = active warps； $R_{\mathrm{file}}$ = 65536（每 SM register 数）； $R_{\mathrm{blk}}$ = 每 block 实际 register 分配（warp 粒度取整后，本章局部）； $S_{\mathrm{sh}}$ = shared bytes/block（请求值）； $S_{\mathrm{total}}$ = 233472 B（carveout 最大档，carveout 档位与 API 见 §5）； $S_{\mathrm{alloc}}$ = 每 block 实际 shared 占用（取整 + 预留后，本章局部）； $\mathcal{O}$ = occupancy； $`\mathcal{O}_{\min}`$ = 延迟隐藏所需最小 occupancy（1.4 式 (5)）； $L_{\mathrm{SM}}$ = 每 SM 所需在飞 sector 并发（≈208，1.4 式 (3)）； $s_{\mathrm{warp}}$ = 每条 warp load 的 sector 数（coalesced fp32 = 4）； $\mathrm{ILP}$ = 每 warp 在飞独立指令数； $L_{\mathrm{reg}}$ = launch_bounds 推导的寄存器上限（PG 记号 L，本章加下标区别于 1.4 的在飞并发 $L$）； $F_{\mathrm{peak}}$ = 峰值 FLOPS； $\mathrm{BW}$ = HBM 带宽。裸字母 n 专归 rank 数、N 专归参数量、B 专归 batch（ $B_{\max}$ 为 spec §4.3 表内上界符号），本章不占用；每 block 线程数写作 $`\mathrm{threads}_{\mathrm{block}} = 32 w_b`$。

## 2. 从物理上界到精确计算：粒度修正与 occupancy API

物理上界往下修多少，才是实际驻留的 block 数？答案藏在两个分配粒度里：register 按 warp 取整、shared 按 128 B 取整外加每 block 1 KB 预留。1.2 交付了硬件资源库存与 occupancy 物理上界（1.2 式 (3)-(5) recap，不重推）：四约束取 min（单位一致，皆 blocks/SM）→ active warps → $\mathcal{O}$，

$$
B_{\max} = \min\left( B_{\max,\mathrm{slot}},\ \left\lfloor \frac{W_{\mathrm{slots}}}{w_b} \right\rfloor,\ \left\lfloor \frac{R_{\mathrm{file}}}{R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}} \right\rfloor,\ \left\lfloor \frac{S_{\mathrm{total}}}{S_{\mathrm{sh}}} \right\rfloor \right)
$$

并证明实际 resident blocks 不超过它（1.2 命题 3：register 以 warp 粒度向上取整、shared 以 128 B 粒度取整且每 block 预留 1 KB，占用只增不减）。本节增量是把「理想口径」换成「分配口径」：register 项的分母从 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}`$ 换成实际分配 $R_{\mathrm{blk}}$，shared 项的分母从请求值 $S_{\mathrm{sh}}$ 换成实际占用 $S_{\mathrm{alloc}}$：

$$
b_{\mathrm{act}} = \min\left( B_{\max,\mathrm{slot}},\ \left\lfloor \frac{W_{\mathrm{slots}}}{w_b} \right\rfloor,\ \left\lfloor \frac{R_{\mathrm{file}}}{R_{\mathrm{blk}}} \right\rfloor,\ \left\lfloor \frac{S_{\mathrm{total}}}{S_{\mathrm{alloc}}} \right\rfloor \right) \qquad (1)
$$

$$
W_{\mathrm{active}} = b_{\mathrm{act}} \cdot w_b \qquad (2)
$$

$$
\mathcal{O} = \frac{W_{\mathrm{active}}}{W_{\mathrm{slots}}} = \frac{b_{\mathrm{act}} \cdot w_b}{W_{\mathrm{slots}}} \qquad (3)
$$

**符号**： $R_{\mathrm{blk}}$ = 每 block 实际 register 分配（warp 粒度向上取整，粒度数值随架构变，见下）； $S_{\mathrm{alloc}} = \lceil S_{\mathrm{sh}}/128 \rceil \times 128 + 1024$ B（128 B 粒度取整 + 每 block 1 KB 预留，1.2 命题 3 recap；1 KB 预留的官方逐字见 Blackwell tuning guide「CUDA reserves 1 KB of shared memory per thread block」，[NVIDIA, Blackwell tuning guide](../references.md)，[C]）。由取整方向， $b_{\mathrm{act}} \le B_{\max}$（1.2 命题 3 方向一致）：式 (1) 从理想上界**往下修**，不会往上修。

最小实例（3.1 elementwise 基线）：256 threads/block（ $w_b = 8$）、 $R_{\mathrm{th}} = 32$、 $S_{\mathrm{sh}} = 0$ → 四项 min(32, 8, $\lfloor 65536/8192 \rfloor = 8$, $\infty$) = 8 blocks， $W_{\mathrm{active}} = 64$， $\mathcal{O} = 1.0$——3.1 章脊假设的式 (1)-(3) 实例化，register 与 warp 项并列最紧。

register 粒度数值须钉住口径。BP 逐字「register allocations are rounded up to the nearest 256 registers per warp」（[NVIDIA, Best Practices](../references.md)，[C]）——这是 **CC 7.0 算例口径**；BP 同节逐字「The number of registers available, the maximum number of simultaneous threads resident on each multiprocessor, and the register allocation granularity vary over different compute capabilities」，且「the exact relationship between register usage and occupancy can be difficult to determine」。CC 9.0 的 register 分配单位官方未逐字披露（沿 1.2 §3 口径），所以手算式 (1) 在 CC 9.0 上取理想 register 项作上界估计，**精确值以 occupancy API 为准**——API 内部实现各架构的分配规则。shared 侧 128 B 粒度与 1 KB 预留则可直接代入（1.2 命题 3 已交付）。

occupancy API 的官方语义逐字：「The occupancy calculator API, cudaOccupancyMaxActiveBlocksPerMultiprocessor, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.」，换算逐字「Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.」（PG archive/13.0.0 §8.2.3.1，[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——API 返回的就是式 (1) 的 $b_{\mathrm{act}}$，式 (2)(3) 是它的官方换算。教学片段（PG 官方例同形）：

```cpp
// 教学片段，非优化实现：occupancy API 三步换算
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, dynamicSmemBytes);
int activeWarps = numBlocks * blockSize / prop.warpSize;          // 式 (2)
int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;  // 64
double occupancy = (double)activeWarps / maxWarps;                // 式 (3)
```

配套 API 三件：`cudaOccupancyMaxPotentialBlockSize` / `cudaOccupancyMaxPotentialBlockSizeVariableSMem` 启发式反解最大 occupancy 的 launch 配置（block 尺寸扫描，§6 曲线的手算替代）；`cudaOccupancyMaxActiveClusters` 给 cluster 尺寸的 occupancy 预测（cluster 语义见 3.1 §6，TMA multicast 用法归 5.2）。kernel 侧资源用量从编译报告读：nvcc `--resource-usage`（PG 13.5 §2.3.7）或 `--ptxas-options=-v`（PG archive/13.0.0）报每 kernel 的 register 与 shared 用量——这是式 (1) 的输入 $R_{\mathrm{th}}$ 与 $S_{\mathrm{sh}}$ 的来源。硬件库存侧七个设备属性（`maxBlocksPerMultiProcessor` / `sharedMemPerMultiprocessor` / `regsPerMultiprocessor` / `maxThreadsPerMultiProcessor` / `sharedMemPerBlock` / `regsPerBlock` / `maxThreadsPerBlock`，PG §2.3.7）运行时可查，编译期同值由 `cuda::arch_traits` 提供。

```
kernel 资源用量                     硬件库存（1.2 §1）
--resource-usage / -Xptxas -v       device properties / arch_traits
  R_th, S_sh, threads_block           R_file, S_total, W_slots, B_max,slot
        │                                   │
        ▼                                   ▼
   粒度修正（R_blk warp 取整 / S_alloc 128B + 1KB）
        │
        ▼
   式 (1) 四约束取 min ──► b_act ──► 式 (2) W_active ──► 式 (3) O
        ▲
   cudaOccupancyMaxActiveBlocksPerMultiprocessor（权威精确值，内含各架构粒度）
```

官方算例三则（PG §2.3.7，CC 10.0 库存 = CC 9.0，[NVIDIA, CUDA Programming Guide](../references.md)，[C]；全数字代入式 (1)-(3)）：

| 算例 | 配置 | 绑定约束 | $b_{\mathrm{act}}$ | $W_{\mathrm{active}}$ | $\mathcal{O}$ |
| --- | --- | --- | --- | --- | --- |
| 768 threads/block | $w_b = 24$ | warp slots（ $\lfloor 64/24 \rfloor = 2$；线程口径 $2048/768 = 2.67$ 取 2） | 2 | 48 | 75% |
| 32 threads/block | $w_b = 1$ | block slots（32；warp 项 64 更宽） | 32 | 32 | 50% |
| 100 KB shared/block | 256 threads | shared（ $3 \times 102400 = 307200 > 233472$） | 2 | 16 | 25% |

逐例核对：768 线程例 PG 逐字「each SM would only be able to execute 2 thread blocks at a time ... the occupancy would be (768 * 2) / 2048, or 75%」；32 线程例「32 blocks * 32 threads per block, or 1024 total threads ... 1024 / 2048, or 50%」——小 block 撞 block slot 上限、warp slot 浪费一半（1.2 案例 C4 同构）；100 KB 例「the third thread block ... would require ... 300KB, which is more than the 233472 bytes available per SM」——含 1 KB 预留核对： $102400 + 1024 = 103424$ B， $\lfloor 233472/103424 \rfloor = 2$，与理想口径同值。三则算例是式 (1)-(3) 的公开校验锚（练习 1）。

## 3. register pressure vs ILP：一个 register 的悬崖与 launch_bounds

为什么多用一个 register，occupancy 就腰斩？因为 register 约束是整 block 粒度的不等式，跨过临界点就掉一个 block。PG 官方算例逐字（CC 6.x）：「if a kernel uses 64 registers and each block has 512 threads ... then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor.」（archive/13.0.0 §8.2.3）。全数字： $2 \times 512 \times 64 = 65536$ 恰好占满 → 2 blocks、32 warps、 $\mathcal{O} = 0.5$； $2 \times 512 \times 65 = 66560 > 65536$ → 1 block、16 warps、 $\mathcal{O} = 0.25$。register 需求涨 1.6%，occupancy 掉一半——这就是 register 粒度悬崖。

粒度取整让悬崖更密。BP 的 CC 7.0 算例（逐字）：「a kernel with 128-thread blocks using 37 registers per thread results in an occupancy of 75% with 12 active 128-thread blocks per multi-processor, whereas a kernel with 320-thread blocks using the same 37 registers per thread results in an occupancy of 63% because only four 320-thread blocks can reside on a multiprocessor」。手算对账：37 regs/thread → 每 warp $\lceil 37 \times 32/256 \rceil \times 256 = 1280$ regs；128 线程 block（4 warps）每 block 5120 regs， $\lfloor 65536/5120 \rfloor = 12$ blocks → 48 warps → 75%，与官方逐字对上；320 线程 block（10 warps）每 block 12800 regs，warp 粒度手算 $\lfloor 65536/12800 \rfloor = 5$ blocks（78%），**官方值却是 4 blocks（63%）**——warp 级取整之外还有架构特定的分配细节（官方未逐字披露 CC 7.0 的逐项规则），手算式 (1) 给上界、API/calculator 给精确值（§2 口径）。方向与 1.2 命题 3 一致：实际 4 ≤ 理想 5。这个差距不是公式错，是「理想口径上界」与「分配口径精确值」的差——3.4 的精确计算以 API 为准的理由在此。

所谓 register pressure（寄存器压力），是指 kernel 的 register 需求超出可分配预算的状态——编译器要么 spill、要么压 occupancy。悬崖的两端是 register pressure 与 ILP 的权衡（recap 1.2 §3 / 1.4 §6，不重推）： $R_{\mathrm{th}}$ 大，单线程驻留的独立指令与中间变量多、ILP 高，但每 block 占用多、 $b_{\mathrm{act}}$ 小； $R_{\mathrm{th}}$ 小则反之。压 $R_{\mathrm{th}}$ 若压到低于 kernel 需求，编译器 spill——把 register 值落到 local memory（HBM backing，3.2 §1），带宽掉约 160 倍、延迟掉 16.5 倍（1.2 §3 / 3.2 §1 比值 recap）。BP 把另一端的收益说透（逐字）：「A lower occupancy kernel will have more registers available per thread than a higher occupancy kernel, which may result in less register spilling to local memory; in particular, with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy.」——低 occupancy 高 ILP 可以免 spill 且藏住延迟，所以「压 register 提 occupancy」不是无条件正确。

控制手段三件，粒度从粗到细：`-maxrregcount=N`（编译选项，全局压所有 kernel 的 register 上限，粒度粗）；`__launch_bounds__`（per-kernel qualifier）；`__maxnreg__`（per-kernel 上限，PG §8.2.3 并列）。launch_bounds 的官方语义逐字（PG 13.5 §5.4.3.2）：「If launch bounds are specified, the compiler first derives the upper limit, L, on the number of registers that the kernel should use. This ensures that minBlocksPerMultiprocessor blocks (or a single block, if minBlocksPerMultiprocessor is not specified) of maxThreadsPerBlock threads can reside on the multiprocessor.」超上限则「the compiler reduces it until it is less than or equal to L. This usually results in increased local memory usage and/or a higher number of instructions.」；低于上限且双参齐备则「the compiler may increase register usage up to L in order to reduce the number of instructions and better hide the latency of single-threaded instructions.」——注意第二个方向：launch_bounds 不是只压不抬，它给编译器一个**目标区间**，余量可换 ILP。

**命题 1**（launch_bounds 寄存器上限）。理想分配口径下，`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 的寄存器上限为

$$
L_{\mathrm{reg}} = \left\lfloor \frac{R_{\mathrm{file}}}{\mathrm{minBlocks} \cdot \mathrm{maxThreads}} \right\rfloor \qquad (4)
$$

（ $\mathrm{minBlocks}$ / $\mathrm{maxThreads}$ 为 qualifier 两参；warp 粒度取整只会收紧此值）。**证明**（必要条件重排，一步）：register 约束 $`n_b \cdot R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}} \le R_{\mathrm{file}}`$（1.2 §5）对 $n_b = \mathrm{minBlocks}$、 $`\mathrm{threads}_{\mathrm{block}} = \mathrm{maxThreads}`$ 解出 $R_{\mathrm{th}}$ 即得；编译器以此为目标上限（上文 PG 逐字）。∎ 全数字代入：`__launch_bounds__(256, 4)` → $L_{\mathrm{reg}} = \lfloor 65536/(4 \times 256) \rfloor = 64$ regs/thread；warp 粒度核对（CC 7.0 口径）：4 blocks × 8 warps = 32 warps， $65536/32 = 2048$ regs/warp 恰为 256 的倍数，取整无损耗。K2 kernel（自然需求 96 regs）加此 qualifier → 压到 64，超出需求的 32 regs/thread 走 spill（教学假设：超额需求全部 spill），换 4 blocks 驻留——值不值由 §4 的隐藏账判。边界：launch 时 block 线程数超过 maxThreadsPerBlock 直接 launch 失败（PG 逐字「A kernel will fail to launch if it is executed with more threads per block than its launch bound maxThreadsPerBlock」）；最优 launch_bounds 随架构变（PG 逐字「The optimal launch bounds for a kernel typically differ across major architecture revisions」），跨代代码用 `__CUDA_ARCH__` 分档。

> **坑**：用 `-maxrregcount` 全局压 register。触发条件：为救一个 register 绑定 kernel 设 `-maxrregcount=64`，编译单元里所有 kernel 同被压；现象：本来 register 宽裕的 compute-bound kernel 也被压出 spill，整体变慢；规避：per-kernel 用 `__launch_bounds__` / `__maxnreg__`，全局选项只作最后手段（BP register pressure 节口径）。

## 4. latency hiding 账：够用的 occupancy 怎么反解

occupancy 要多少才够？答案不在 occupancy 本身，在延迟隐藏账（recap 1.4，不重推）：藏住 HBM 延迟需每 SM 约 208 个 sector 在飞（ $L_{\mathrm{SM}}$，1.4 式 (3)，HBM 478.8 cycles 实测 [B] 口径；服务率取 HBM 每 SM 带宽份额 ÷ 32 B sector，**不是** warp 发射率——取错高估约 9 倍，1.4 坑注 recap），折 coalesced fp32 warp load（ $s_{\mathrm{warp}} = 4$）为 52 条；隐藏条件 $W_{\mathrm{active}} \cdot \mathrm{ILP} \cdot s_{\mathrm{warp}} \ge L_{\mathrm{SM}}$（1.4 式 (4)），解出最小 occupancy $`\mathcal{O}_{\min} = L_{\mathrm{SM}}/(s_{\mathrm{warp}} \cdot \mathrm{ILP} \cdot W_{\mathrm{slots}})`$（1.4 式 (5)）：ILP 1 → 0.81，ILP 2 → 0.41，ILP 4 → 0.20。本节增量是**反解资源预算**：从 $`\mathcal{O}_{\min}`$ 倒推需要几个 block、每线程至多几个 register——

$$
n_b^{\mathrm{req}} = \left\lceil \frac{\mathcal{O}_{\min} \cdot W_{\mathrm{slots}}}{w_b} \right\rceil, \qquad R_{\mathrm{th}}^{\max} = \left\lfloor \frac{R_{\mathrm{file}}}{n_b^{\mathrm{req}} \cdot \mathrm{threads}_{\mathrm{block}}} \right\rfloor \qquad (5)
$$

**符号**： $n_b^{\mathrm{req}}$ = 达到 $`\mathcal{O}_{\min}`$ 所需 resident blocks； $R_{\mathrm{th}}^{\max}$ = 该 block 数下的每线程 register 预算上限（理想口径，warp 取整只收紧）。全数字代入（memory-bound、ILP 1、256 threads/block 即 $w_b = 8$）： $`\mathcal{O}_{\min} = 0.81`$ → 需 $0.81 \times 64 = 52$ warps → $n_b^{\mathrm{req}} = \lceil 52/8 \rceil = 7$ blocks（56 warps， $\mathcal{O} = 0.875$）；register 预算 $R_{\mathrm{th}}^{\max} = \lfloor 65536/(7 \times 256) \rfloor = \lfloor 36.57 \rfloor = 36$ regs/thread。粒度收紧示例（CC 7.0 的 256/warp 披露口径）：36 regs → 每 warp $\lceil 1152/256 \rceil \times 256 = 1280$， $7 \times 8 \times 1280 = 71680 > 65536$，装不下——须降到 32 regs/thread（1024/warp， $56 \times 1024 = 57344 \le 65536$）。CC 9.0 的精确预算以 API 为准（§2 口径），手算 36 是理想上界。

K2 kernel 的完整判定（256 threads、 $R_{\mathrm{th}} = 96$、 $S_{\mathrm{sh}} = 32$ KB）：式 (1) 四项 = min(32, 8, $\lfloor 65536/24576 \rfloor = 2$, $\lfloor 233472/33792 \rfloor = 6$) = 2 blocks（register 绑定；96 为 8 的倍数，warp 取整无损耗）， $W_{\mathrm{active}} = 16$， $\mathcal{O} = 0.25$。ILP 1 时 $16 \lt 52$，藏不住 HBM；最小 ILP = $\lceil 52/16 \rceil = 4$（ $16 \times 4 = 64 \ge 52$， $16 \times 3 = 48 \lt 52$）。两条出路：其一，保 96 regs 提 ILP 到 4——但 ILP 4 要更多独立地址/目的寄存器，与 96 regs 预算竞争（1.4 §6 权衡）；其二，`__launch_bounds__(256, 4)` 压到 64 regs → 4 blocks、32 warps、 $\mathcal{O} = 0.5$，ILP 2 即 $32 \times 2 = 64 \ge 52$ 藏住——代价是 32 regs/thread 的 spill（§3）。哪条路赢取决于 spill 流量与 ILP 收益的实测对比（11.2），没有先验答案。

compute-bound 是另一番景象。**命题 2**（compute-bound 最小并发）。FP32 FMA 依赖链 kernel（ILP 1）喂满 SM 需 16 个 active warps（ $\mathcal{O} = 0.25$）；ILP 4 时仅需 4 个 warps（ $\mathcal{O} = 0.0625$）。**证明**（路线：per-scheduler Little's law + 单发射宽度，三步）：单 scheduler 每周期发射 1 条 warp FMA（1.2 命题 1 单发射），FMA 延迟 4 cycles，故每 scheduler 需 $1 \times 4 = 4$ 条在飞 warp FMA（1.4 §4 热身例）；4 个 scheduler 共 16 条；ILP 1 时每 warp 贡献 1 条在飞 → 16 warps，ILP 4 时每 warp 贡献 4 条 → 4 warps。完整证明（含依赖链吞吐塌缩与 Tensor Core 延迟代入的边界）留作练习 3。∎ 官方锚：PG 逐字「This means that 16 active warps per multiprocessor (4 cycles, 4 warp schedulers) are required to hide arithmetic instruction latencies ... If the individual warps exhibit instruction-level parallelism ... fewer warps are needed because multiple independent instructions from a single warp can be issued back to back.」（archive/13.0.0 §8.2.3）。全数字：16/64 = 25%，4/64 = 6.25%——compute-bound kernel 的「够用 occupancy」比 memory-bound 的 81% 低一个量级。

所以「occupancy 越高越好」是错误目标。BP 逐字「Higher occupancy does not always equate to higher performance」，「However, low occupancy always interferes with the ability to hide memory latency, resulting in performance degradation.」（[NVIDIA, Best Practices](../references.md)，[C]）。**口径注**：这两句与命题 2 不矛盾——BP 的「low occupancy」指**隐藏并发不足**（ $W_{\mathrm{active}} \cdot \mathrm{ILP}$ 乘积低于需求），不是 occupancy 数值本身低；1.4 命题 2 的乘积替代说的是同一件事：ILP 补上，低 occupancy 数值照样藏住延迟。occupancy 是库存比例，隐藏并发是库存 × ILP，目标是后者达标（1.2 §5 末段口径 recap）。

bound 类型决定用哪笔账（recap 2.4 roofline 入口，不重推）：elementwise add 每元素 1 FLOP / 12 B（读 a、b 写 c）， $I = 1/12 \approx 0.083$ FLOPs/B，远小于 FP32 ridge $R = F_{\mathrm{peak}}/\mathrm{BW} = 66.9/3.352 \approx 20$ FLOPs/B（本章口径实例化， $F_{\mathrm{peak}}$ 与 $\mathrm{BW}$ 取 1.1/2.1 值）→ memory-bound，用 $`\mathcal{O}_{\min} = 0.81`$（ILP 1）； $I \gg R$ 的 compute-bound kernel 用命题 2 的 6.25%–25%。先判 bound 类型再定 occupancy 目标，顺序不能反。

> **提点**：register 预算是 ILP 与 TLP 之间的零和分配——同一个 256 KB register file，给单线程多分（ILP 轴）就给驻留 warp 少分（TLP 轴）。launch_bounds 与 carveout 是这个分配的两个旋钮，隐藏条件（1.4 式 (4)）是分配的目标函数。删掉这条类比，式 (4)(5) 与命题 1/2 的推导链依然完整。

## 5. shared 配置：carveout 档位与容量悬崖

同一个 kernel，换个 carveout 档 occupancy 就变——为什么？因为 shared 约束的分母有两个可调项：每 block 请求 $S_{\mathrm{sh}}$（kernel 侧）与每 SM 总量 $S_{\mathrm{total}}$（carveout 侧）。recap 1.2 §4（不重推）：L1/shared 统一物理 256 KB，carveout 档位 {0, 8, 16, 32, 64, 100, 132, 164, 196, 228} KB，最大档 $S_{\mathrm{total}} = 228$ KB = 233472 B；每 block 用户上限 227 KB（1 KB 预留）；228/227/256 三个数三个口径（1.2 坑注）。

carveout 的 API 语义（1.2 留待本章交付）：`cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout)` 在 launch 前按 kernel 设定，carveout 取最大 shared 容量的整数百分比或三个枚举（`cudaSharedmemCarveoutDefault` / `MaxL1` / `MaxShared`）；百分比不恰好落在支持档位时取**下一档更大容量**（PG §3.2.6 逐字「the next larger capacity is used」）；且「cudaFuncSetAttribute is interpreted by the driver as a hint, and the driver may choose a different carveout size if required to execute the kernel.」——是 hint 不是硬约束。旧 API `cudaFuncSetCacheConfig` 是硬要求，交错 launch 不同配置的 kernel 会在重配置后串行化，PG 明示 preferred 用前者。每 block 超 48 KB 的 shared 须动态 shared + 显式 opt-in：`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes)`（PG 官方例 96 KB = 98304），静态数组上限 48 KB（1.2 §4 recap）。

shared 容量悬崖表（ $S_{\mathrm{total}} = 233472$ B， $S_{\mathrm{alloc}} = S_{\mathrm{sh}} + 1024$；KB 倍数均 128 B 对齐，取整无额外损耗；式 (1) shared 项）：

| $S_{\mathrm{sh}}$ | 理想项 $\lfloor S_{\mathrm{total}}/S_{\mathrm{sh}} \rfloor$ | 精确项 $\lfloor S_{\mathrm{total}}/S_{\mathrm{alloc}} \rfloor$ | 备注 |
| --- | --- | --- | --- |
| 32 KB | 7 | 6 | 1 KB 预留吃掉 1 block（1.2 §6 例 recap） |
| 48 KB | 4 | 4 | 静态上限档 |
| 64 KB | 3 | 3 | 须 opt-in |
| 96 KB | 2 | 2 | 须 opt-in |
| 112 KB | 2 | 2 | $233472/115712 = 2.018$，贴着悬崖 |
| 116 KB | 1 | 1 | 跨悬崖：+4 KB 掉 1 block |
| 227 KB | 1 | 1 | 每 block 上限，恰好占满 |

carveout 换档的灵敏度（式 (1) shared 项， $S_{\mathrm{sh}} = 64$ KB、 $S_{\mathrm{alloc}} = 66560$ B）：228 KB 档 $\lfloor 233472/66560 \rfloor = 3$ blocks；100 KB 档（ $S_{\mathrm{total}} = 102400$ B） $\lfloor 102400/66560 \rfloor = 1$ block——同一 kernel，carveout 从最大档换到 100 KB 档，shared 约束从 3 压到 1。若 register/warp 项更紧则不变（K2 的 register 项 2 比 shared 项紧，换 100 KB 档后 shared 项 $\lfloor 102400/33792 \rfloor = 3$ 仍宽于 2，occupancy 不变，练习 4(d)）：**只有绑定约束的松紧才改变 occupancy**，这是四约束取 min 的直接推论。

权衡一笔（指针 4.3）：shared 切得大，单 block 装得下大 tile、数据复用高（tiling 强度账归 4.3），但 $S_{\mathrm{total}}$ 被单 block 吃掉更多、驻留 block 少；shared 切得小则反之。double buffering 把 shared 需求翻倍（4.3），multi-stage pipeline 再乘 stage 数（4.4）——大 tile kernel 的 occupancy 天然低，是否够用仍由 §4 隐藏账判，不由 occupancy 数值判。4.3/6.3 的 tile 预算约束（ $(T_M + 2T_N) \cdot d_{\mathrm{head}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}}$ 等）以本章式 (1) 为 occupancy 层回代点。

> **坑**：动态 shared 超 48 KB 未 opt-in。触发条件：launch 配置第三参给 96 KB 但未调 `cudaFuncAttributeMaxDynamicSharedMemorySize`；现象：launch 失败（invalid argument），不是 occupancy 低；规避：opt-in 先行，且每 block 总量不超 227 KB（1 KB 预留，1.2 §4）；把 combined 256 KB 当 shared 总量代入会高估上界（1.2 坑注 recap）。

## 6. NCU Occupancy Calculator 读法

手算式 (1) 与 API 之外，图形化读法在哪？Nsight Compute 的 Occupancy Calculator。官方逐字「NVIDIA Nsight Compute provides an Occupancy Calculator that allows you to compute the multiprocessor occupancy of a GPU for a given CUDA kernel. It replaces the previously provided CUDA Occupancy Calculator spreadsheet.」（[NVIDIA, Nsight Compute](../references.md) §4.13，[C]）——独立 Excel spreadsheet 已废，本书不引旧 spreadsheet。四个视图（NCU §4.13.1-4）：Tables 给 occupancy 与每 SM active threads/warps/blocks、全 GPU max active blocks；Utilization 给资源堆叠条；Graphs 给 occupancy 随 block 尺寸的曲线；GPU Data 给各设备库存。输入 kernel 的 block 尺寸/register/shared 用量，Apply 即出结果；数据可存 `.ncu-occ` 文件，另有 Python interface 供脚本化扫描。

Utilization 三态是四约束取 min 的可视化（NCU 逐字）：「These stacked bar graphs show the portion of resources that is allocated to the blocks, the unallocated portion due to some other resource being a limiter, and lastly the unused portion that has the size less than the minimum amount required by a block.」拆三态：allocated（已分给驻留 block）、unallocated due to other limiter（被**别的**约束卡住而分不出去——绑定约束的直接指示）、unused（不足一个 block 最小需求的零头，粒度取整的可视化）。读法：哪种资源的 unallocated 段最大，哪种就是绑定约束；register 条满而 shared 条大段 unallocated，即 register 绑定（K2 案例）。

```
Utilization 堆叠条示意（K2：register 绑定）
register │████████████████████│░░│        ← 满 + 零头：绑定约束
shared   │██████│░░░░░░░░░░░░░░░░│        ← 大段 unallocated：被 register 卡住
warp slot│████████│░░░░░░░░░░░░░░│
block slot│██│░░░░░░░░░░░░░░░░░░░│
█ = allocated   ░ = unallocated due to other limiter / unused
```

Graphs 视图的曲线是锯齿状——每个台阶跳变对应一个粒度悬崖（§3 register 悬崖、§5 shared 悬崖）；选定 block 尺寸以蓝点标出，其余尺寸连成曲线，Y 轴可切百分比或绝对 warp 数。读法：在曲线上找「occupancy 平台」与「悬崖位置」，block 尺寸选平台区、避开悬崖右侧（多用一个 register 掉一档的位置）。理论 occupancy（calculator/API 预测）与 achieved occupancy（NCU profiling 实测，occupancy metrics）是两个口径：achieved ≤ 理论，差距来自尾波量化（3.1 §5：grid 不是驻留容量整数倍时末波 SM 空转）、block 间负载不均与 launch/退出瞬态——理论值是库存上界，实测值是时间平均，勿混。

> **坑**：引旧 CUDA Occupancy Calculator spreadsheet 的 URL 或截图。触发条件：照旧书/旧 blog 用独立 Excel 工具；现象：工具已废（NCU 逐字 replaces），口径与现行架构脱节；规避：用 NCU Occupancy Calculator（§4.13）或运行时 API，两者同源。

## 7. 边界：本章不做什么

精确计算与权衡账交付完，剩下的归谁？本章交付粒度修正式 (1)-(3)、occupancy API 语义、register 悬崖与 launch_bounds 命题 1、隐藏账反解式 (5) 与 compute-bound 命题 2、carveout API 与 shared 悬崖、NCU 读法；以下各有专章。tile 几何 ↔ shared/register 预算的推导归 4.3（tiling）与 6.3（FA tile 约束），本章式 (1) 是它们的 occupancy 层回代点；GEMM register 分摊与 mainloop 走读归 5.2。roofline / MFU 与 bound 类型判定归 2.4（本章只 recap 入口）；spill 的精确带宽-延迟代价归 2.3；stall reason 实测与 occupancy drill-down 归 11.2；bank conflict 归 4.2；cluster occupancy（`cudaOccupancyMaxActiveClusters`）语义见 3.1 §6，DSMEM/TMA multicast 用法归 5.2。代码 scope：机制片段级（§2 API 换算、§3 launch_bounds），不做模块走读。

本章交付清单（下游 recap 接口）：粒度修正式 (1)-(3) + API 语义（§2）、命题 1 + 式 (4)（§3）、式 (5) 反解 + 命题 2（§4）、carveout API + shared 悬崖表（§5）、NCU 三态读法（§6）。4.3 recap 式 (1) 作 tile 预算的 occupancy 约束；6.3 recap §4 隐藏账作 FA warp 分工的并发目标；11.2 recap §6 理论/实测两口径作 drill-down 起点；5.2 recap §3 register 权衡作 GEMM 分摊动机。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 精确 occupancy = 四约束取 min（register 换 R_blk warp 取整、shared 换 S_alloc 128B + 1KB）→ active warps → O；b_act ≤ 1.2 理想上界 | 式 (1)(2)(3)，1.2 命题 3 方向 | §2、练习 1/2 |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor 返回 blocks/SM，×w_b ÷max warps 得 O；粒度随架构变，精确值以 API 为准 | §2 PG 逐字 | §2、练习 1 |
| 官方算例：768 threads → 2 blocks 75%；32 threads → 32 blocks 50%（block slot 绑定）；100 KB shared → 2 blocks | §2 表（PG §2.3.7） | §2、练习 1 |
| register 悬崖：CC 6.x 64→65 regs × 512 threads，2 blocks 50% → 1 block 25%；CC 7.0 37 regs 128/320 threads 官方 12/4 blocks（手算 320 给 5，粒度细节架构特定） | §3 PG/BP 逐字 | §3、练习 2 |
| launch_bounds 寄存器上限 L_reg = floor(R_file/(minBlocks·maxThreads))；超则压（spill/指令增），低于且双参可抬至 L 换 ILP；(256,4) → 64 regs | 式 (4)，命题 1 | §3、练习 4 |
| 隐藏账反解：O_min（1.4 式 (5)）→ n_b_req → R_th_max；ILP1 256-thread 需 7 blocks、预算 36 regs（CC 7.0 粒度收紧到 32） | 式 (5) | §4、练习 4 |
| K2（96 regs/32KB/256）：2 blocks 16 warps 25%，ILP1 藏不住、最小 ILP 4；launch_bounds(256,4) → 32 warps ILP2 藏住，代价 spill | §4 判定 | §4、练习 4 |
| compute-bound：FP32 FMA 满吞吐需 16 在飞 warp FMA；ILP1 → 16 warps 25%，ILP4 → 4 warps 6.25%（memory-bound ILP1 需 81%） | 命题 2 | §4、练习 3 |
| occupancy 非越高越好：目标是隐藏并发 W_active×ILP 达标，不是 occupancy 数值；bound 类型先判（I 对 R） | §4 BP 逐字 + 口径注 | §4 |
| carveout 是 per-kernel hint（百分比取下一档更大容量）；超 48 KB 动态 shared 须 opt-in；112→116 KB 掉 1 block；只有绑定约束松紧改变 O | §5 PG 逐字 + 表 | §5、练习 2/4 |
| NCU Occupancy Calculator 替代 spreadsheet；Utilization 三态 = 绑定约束可视化；Graphs 锯齿 = 粒度悬崖；理论 ≠ achieved（尾波/不均） | §6 NCU 逐字 | §6 |

## 练习

1.（计算 · [10]）官方算例代入（CC 9.0/10.0 库存）：(a) 768 threads/block（ $R_{\mathrm{th}}$ 宽裕、无 shared）用式 (1)-(3) 算 $b_{\mathrm{act}}$ 与 $\mathcal{O}$；(b) 32 threads/block 同算，指出绑定约束；(c) 256 threads/block、100 KB shared 同算。校验：与 PG §2.3.7 官方算例 75% / 50% / 2 blocks 逐字一致（公开数值锚）；(b) 与 1.2 案例 C4 的 block-slot 绑定同构。
2.（计算 · [20]）register 悬崖与 shared 悬崖：(a) CC 6.x 口径，512 threads/block、64 与 65 regs/thread 各自的 blocks/warps/ $\mathcal{O}$（精确算术，PG 例）；(b) CC 7.0 口径（256/warp 取整），128 threads/block、37 regs 的每 warp 分配与 blocks；(c) 用 §5 表核对 $S_{\mathrm{sh}}$ = 32 / 112 / 116 KB 的精确 shared 项，指出 112→116 的悬崖；(d) carveout 从 228 KB 换 100 KB 档， $S_{\mathrm{sh}} = 64$ KB 的 shared 项怎么变。先预测 (a) 的方向再算。校验：(a) 2 blocks 32 warps 50% → 1 block 16 warps 25%，与 PG §8.2.3 逐字一致；(b) 1280/warp、12 blocks 75%，与 BP 逐字一致；(c) 6 / 2 / 1，与 1.2 §6 例（7→6）方向一致；(d) 3 → 1。
3.（推导 · [30]）补全命题 2 的证明（compute-bound 最小并发）：(a) 由 1.2 命题 1 单发射宽度与 FMA 延迟 4 cycles，用 Little's law 算每 scheduler 所需在飞 warp FMA；(b) 乘 4 scheduler 得每 SM 16 条，ILP 1 折 16 warps、 $\mathcal{O} = 0.25$；(c) ILP 4 折 4 warps、 $\mathcal{O} = 0.0625$；(d) 边界：单 warp FMA 依赖链（ILP 1）的吞吐为何塌到 1/4（1.4 §2 时序图），以及把 FMA 延迟换成 wgmma 128 cycles（1.3/1.4 实测 [B]）时结论怎么变。校验：(b) 与 PG「16 active warps per multiprocessor (4 cycles, 4 warp schedulers)」逐字一致（公开数值锚）；(c) 与 1.4 §4 热身例 16 条在飞一致；(d) 依赖链每 4 周期 1 条（1.4 §2）。
4.（计算 · [30]）K2 综合账（256 threads、 $R_{\mathrm{th}} = 96$、 $S_{\mathrm{sh}} = 32$ KB，memory-bound、ILP 2）：(a) 先预测绑定约束，再用式 (1)-(3) 算 $b_{\mathrm{act}}$ / $W_{\mathrm{active}}$ / $\mathcal{O}$；(b) ILP 2 能否藏住 HBM（52 warp load）？若不能，最小 ILP 多少；(c) 加 `__launch_bounds__(256, 4)`：用式 (4) 算 $L_{\mathrm{reg}}$，重算 occupancy 与隐藏条件，指出代价方向；(d) 若改 carveout 100 KB 档（不加 launch_bounds），occupancy 变不变，为什么。校验：(a) 2 blocks / 16 warps / 25%，register 绑定，与 1.4 练习 4(a) 一致；(b) $16 \times 2 = 32 \lt 52$ 藏不住，最小 ILP 4；(c) $L_{\mathrm{reg}} = 64$、4 blocks / 32 warps / 50%， $32 \times 2 = 64 \ge 52$，代价 spill（§3 方向）；(d) 不变——shared 项 3 仍宽于 register 项 2，非绑定约束（§5 推论）。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——13.5 §2.3.7 Kernel Launch and Occupancy（occupancy 定义逐字 + 七设备属性 + CC 10.0 Table 2 算例 768/32/100KB 逐字 + cuda::arch_traits + --resource-usage + spill 净收益逐字）；§3.2.6 carveout API（百分比取下一档逐字 + driver hint 逐字 + cudaFuncSetCacheConfig 硬要求串行化 + MaxDynamicSharedMemorySize opt-in 96KB 例）；§5.4.3.2 Launch Bounds（L 推导逐字 + 超 L 压/spill 逐字 + 双参抬至 L 逐字 + launch 失败条件 + 架构分档逐字）；archive/13.0.0 §8.2.3（CC 6.x 64/65 regs 悬崖逐字 + 16 active warps 算术延迟逐字 + --ptxas-options=-v + maxrregcount/__launch_bounds__/__maxnreg__）+ §8.2.3.1 Occupancy Calculator API 逐字（blocks/SM 口径 + 换算逐字 + MaxPotentialBlockSize/VariableSMem + MaxActiveClusters）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Occupancy 定义 + 「Higher occupancy does not always equate to higher performance」与 low occupancy 逐字 + CC 7.0 37 regs 128/320-thread 算例逐字 + 256 regs/warp 取整逐字 + 粒度随架构变/exact relationship difficult 逐字 + NCU calculator 交叉引用；Register Pressure（maxrregcount/launch bounds）+ ILP 低 occupancy 满 latency cover 逐字
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——§4.13 Occupancy Calculator「replaces the previously provided CUDA Occupancy Calculator spreadsheet」逐字 + Tables/Utilization 三态逐字/Graphs/GPU Data + .ncu-occ 文件 + Python interface；occupancy metrics（理论/实测两口径，§6）
- [NVIDIA, Hopper tuning guide](../references.md)：NVIDIA Hopper Tuning Guide（key: hopper-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 9.0 occupancy 64 warps / shared 228 KB（1.2 库存 recap 互证）
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——「CUDA reserves 1 KB of shared memory per thread block ... up to 227 KB」逐字（§2 S_alloc 预留项）+ static 48 KB
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——HBM 478.8 cycles 延迟实测口径（§4 L_SM recap 1.4 的输入，非 H100 官方规格）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——132 SM、HBM3 3352 GB/s（§4 ridge 实例化带宽口径；FP32 峰值 66.9 TFLOPS 沿 1.1 §3）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：NVIDIA 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM FP32 67 TFLOPS / HBM 3.35 TB/s（§4 ridge R ≈ 20 FLOPs/B 实例化，沿 1.1 §3 口径）
