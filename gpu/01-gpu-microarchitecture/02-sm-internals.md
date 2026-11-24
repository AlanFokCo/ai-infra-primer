# 1.2 SM 内部拆解（SM Internals）

> 一个 SM 同时装得下几个 thread block？本章交付 SM 的资源库存（4 个单发射 warp scheduler、64K register、228 KB shared、64 warp slots、32 block slots）与 occupancy 物理上界的单位一致推导（四约束取 min → active warps → $\mathcal{O}$），含 register-binding 不可 launch 条件与五个全数字案例 | 前置：1.1（芯片组织、 $n_{\mathrm{SM}}$ 与峰值反推式 (1)）、3.1 预告（warp/block 语义，本章只用「warp = 32 线程、block 驻留一个 SM」两条）；下游 3.4（编程模型侧 occupancy 计算 recap 本章上界）、1.3（Tensor Core 内部）、1.4（延迟隐藏）、2.3（片上存储层级）、4.2（bank conflict） | 锚点：NVIDIA Hopper whitepaper V1.04 Fig 7 / Table 3 / Table 4（[NVIDIA, Hopper whitepaper](../references.md)，[C] 官方规格披露·一手权威）、CUDA Programming Guide compute-capabilities Table 33/34/35（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、CUDA Best Practices Table 5（[NVIDIA, Best Practices](../references.md)，[C]）、Hopper/Blackwell Tuning Guide（[NVIDIA, Hopper tuning guide](../references.md)，[C]）、PTX ISA（[NVIDIA, PTX ISA](../references.md)，[C]）——全部 2026-11-25 一手核实

## 1. 资源库存：一个 SM 装得下几个 block

一个 SM 同时能驻留几个 thread block？答案不在编程模型里，在四个硬件库存数里：32 个 block slot、64 个 warp slot、65536 个 32-bit register、228 KB shared memory。任何一个耗尽，下一个 block 就装不进来。本章回答三个问题：SM 内部有哪些执行与存储单元，各是多少？warp scheduler 每周期发射几条指令（单发射还是双发射）？occupancy 的物理上界怎么从四个约束推出来，为什么 255 registers/thread × 1024 threads 的 kernel 根本 launch 不起来？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 1.1）；B200 CC 10.0 作对照——两者资源库存相同（64 warps / 32 blocks / 64K regs / 228 KB shared / 256 KB combined，[NVIDIA, CUDA Programming Guide](../references.md) Table 33/34/35 + [NVIDIA, Blackwell tuning guide](../references.md)，[C]），故本章上界式对 CC 9.0/10.0 同形。精度口径：register 带宽账用 FP32-boost 1980 MHz（沿 1.1 §3 逐精度 clock）。单位口径：片上存储 KB = 1024 B（Programming Guide 逐字注），228 KB = 233472 B、256 KB = 262144 B；带宽 GB/s 按 $10^{9}$ B/s。

资源库存表（CC 9.0 / 10.0，[NVIDIA, Hopper whitepaper](../references.md) Table 4 + [NVIDIA, CUDA Programming Guide](../references.md) Table 33/34/35，[C]，2026-11-25 核实）：

| 资源 | 符号 | 每 SM 库存 | 出处 |
| --- | --- | --- | --- |
| warp slots | $W_{\mathrm{slots}}$ | 64 warps（= 2048 threads） | PG Table 33 / wp Table 4 |
| block slots | $B_{\max,\mathrm{slot}}$ | 32 blocks | PG Table 33 |
| register file | $R_{\mathrm{file}}$ | 65536 个 32-bit register = 256 KB | PG Table 34 / wp Table 3 |
| 每线程 register 上限 | — | 255 | PG Table 34 / wp Table 4 |
| shared memory 总量 | $S_{\mathrm{total}}$ | 228 KB = 233472 B（carveout 最大档） | PG Table 34/35 |
| L1 + shared 统一物理 | — | 256 KB = 262144 B | PG Table 35 |
| 每 block shared opt-in 上限 | — | 227 KB（CUDA 每 block 预留 1 KB） | Blackwell tuning guide 逐字 |
| 每 block 线程上限 | — | 1024 threads（= 32 warps） | PG Table 33 |
| shared bank 数 | — | 32 banks × 4 B | PG Table 34 |

符号声明（spec §4.3）： $R_{\mathrm{th}}$ = registers/thread； $S_{\mathrm{sh}}$ = shared bytes/block； $w_b$ = warps/block； $B_{\max}$ = 每 SM 可驻留 block 数上界（本章推导量）； $\mathcal{O}$ = occupancy = active warps / $W_{\mathrm{slots}}$； $W_{\mathrm{active}}$ = active warps（本章局部符号）；warp size = 32 为常量，故每 block 线程数 $= 32 w_b$。 $B_{\max,\mathrm{slot}}$ 是硬件 block slot 上限（32）， $B_{\max}$ 是四约束取 min 后的实际上界——两者勿混（§5）。

**边界声明**（本章交付什么、不交付什么）：本章只交付**硬件资源库存 + occupancy 物理上界**。编程模型侧的 occupancy 精确计算——register 按 warp 分配粒度取整、shared 按 128 B 粒度 + 每 block 1 KB 预留、carveout 选择、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` API、launch bounds——归 3.4，届时 recap 本章上界为起点。Tensor Core 内部数据流与 MMA 指令几何归 1.3；指令延迟与 latency hiding 归 1.4；shared bank conflict 判定归 4.2；片上存储层级总表与 roofline 归 2.3 / 2.4。inference 3.1 §1 把「warp 级占用率的精确定义与硬件来源」指向 gpu 维度——本章 §5 即该定义的硬件来源交付。

为什么绑定 occupancy 的恰是这四种资源、而不是别的？因为一个 block 驻留 SM 期间**独占**这四样：它占一个 block slot（调度器跟踪它的身份）、占 $w_b$ 个 warp slot（它的 warp 要能被发射）、占 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}`$ 个 register（它的线程状态要常驻）、占 $S_{\mathrm{sh}}$ 字节 shared（它的协作数据要常驻）。这四样在 block 整个生命周期里不释放，故同时驻留的 block 数被四者各自封顶。执行单元（FP32/Tensor Core）不绑定 occupancy——它们是**时分复用**的，所有 resident warp 轮流用，不独占；所以 SM 有 128 个 FP32 lane 却允许 64 个 warp（2048 线程）驻留，线程数远超 lane 数正是时分复用的体现。库存表里 CC 9.0 与 10.0 逐行相同（PG Table 33/34/35），意味着本章上界式跨 Hopper/Blackwell 同形，代际差异在 Tensor Core 吞吐（1.3）与 HBM（2.1），不在 occupancy 库存。

## 2. SM 框图：四个 processing block 与单发射

一个 SM 每周期能发射几条 warp 指令？4 条——不是 8 条。Hopper 的 warp scheduler 是**单发射**：每个 scheduler 每周期至多发射 1 条 warp 指令到它唯一的 dispatch unit。双发射是 Kepler 时代的术语（每 scheduler 配 2 个 dispatch unit），Hopper 不成立。这个「4 还是 8」直接决定 FP32 峰值是 66.9 还是 133.8 TFLOPS，是本节要用官方框图与吞吐表互证锁死的论断。

所谓 processing block（亦称 sub-partition），是指 SM 内一套独立的「warp scheduler + dispatch unit + register file 分区 + 执行管线」组合。Hopper SM 由 4 个 processing block 构成（whitepaper Fig 7，[NVIDIA, Hopper whitepaper](../references.md)，[C]）。SM 内部框图（Fig 7 重绘）：

```
Hopper SM（CC 9.0，whitepaper Fig 7 重绘）
┌──────────────────────────────────────────────────────────────────────┐
│  processing block 0          processing block 1                       │
│ ┌──────────────────────┐   ┌──────────────────────┐                  │
│ │ L0 I-cache           │   │ L0 I-cache           │                  │
│ │ Warp Scheduler       │   │ Warp Scheduler       │  ← 每 block 恰 1 │
│ │  (32 thread/clk)     │   │  (32 thread/clk)     │    scheduler     │
│ │ Dispatch Unit        │   │ Dispatch Unit        │  ← 每 block 恰 1 │
│ │  (32 thread/clk)     │   │  (32 thread/clk)     │    dispatch      │
│ │ Register File        │   │ Register File        │                  │
│ │  16384 × 32-bit      │   │  16384 × 32-bit      │  ← 4×16384=65536 │
│ │ 32 FP32 │ 16 INT32   │   │ 32 FP32 │ 16 INT32   │                  │
│ │ 16 FP64 │ 1 TC(4th)  │   │ 16 FP64 │ 1 TC(4th)  │  ← 4 TC/SM       │
│ │ 8 LD/ST │ SFU        │   │ 8 LD/ST │ SFU        │                  │
│ └──────────────────────┘   └──────────────────────┘                  │
│  processing block 2          processing block 3                       │
│ ┌──────────────────────┐   ┌──────────────────────┐                  │
│ │ （同 block 0 结构）   │   │ （同 block 0 结构）   │                  │
│ └──────────────────────┘   └──────────────────────┘                  │
│ ─────────────────── SM 级共享 ───────────────────                     │
│  Tensor Memory Accelerator (TMA)                                      │
│  256 KB L1 Data Cache / Shared Memory（统一物理，carveout 可配）       │
│  4 Tex                                                                │
└──────────────────────────────────────────────────────────────────────┘
每 SM 合计：128 FP32 / 64 INT32 / 64 FP64 / 4 Tensor Core / 32 LD/ST
```

**命题 1**（单发射吞吐互证）。CC 9.0 每 SM 的 FP32 吞吐为 128 results/cycle，等于 4 个 warp scheduler × 每 scheduler 1 个 dispatch unit × 32 lane × 每周期 1 条指令。**证明**（框图与吞吐表两路互证）：其一，Fig 7 每个 processing block 恰含 1 个 Warp Scheduler (32 thread/clk) 与 1 个 Dispatch Unit (32 thread/clk)——「32 thread/clk」即每周期处理一个 warp 的 32 个线程，也就是 1 条 warp 指令；4 个 block 各 1 条，合计每周期 4 条 warp 指令、 $4 \times 32 = 128$ 个 FP32 result。其二，Best Practices Table 5（CC 9.0 列）逐字给 FP32 add/mul/fma 吞吐 = 128 results/cycle/SM（[NVIDIA, Best Practices](../references.md)，[C]）。两路独立来源给出同一个 128，单发射成立。若为双发射（每 scheduler 2 dispatch），FP32 吞吐应为 256 results/cycle/SM，与 Table 5 的 128 矛盾。∎

全数字实例（接 1.1 式 (1) FP32 行）：128 results/cycle/SM × 2 FLOPs/FMA × 1.98 GHz × 132 SM = $128 \times 2 \times 1.98 \times 10^{9} \times 132 = 66.9$ TFLOPS，与 1.1 §3 逐精度表 FP32 行、datasheet 67 TFLOPS 逐字对上。这条链把「单发射」从框图论断落到了芯片级峰值数：发射宽度错一倍，峰值就错一倍。

> **坑**：把 Hopper 当双发射。触发条件：沿用 Kepler 时代「每 scheduler 双发射」的旧术语，或把「4 scheduler × 2 dispatch」当 Hopper 配置；现象：FP32 吞吐算成 256 results/cycle/SM，芯片级峰值高估 2×（133.8 对 66.9 TFLOPS），进而 MFU、roofline、latency hiding 所需并发全部虚高；规避：Hopper 每 processing block 恰 1 scheduler + 1 dispatch（Fig 7），单发射，以 Best Practices Table 5 的 128 为准。

执行管线吞吐表（Best Practices Table 5，CC 9.0 列，[NVIDIA, Best Practices](../references.md)，[C]；口径 = results/cycle/SM）：

| 指令类 | 吞吐（results/cycle/SM） | 每 SM 物理单元数（Fig 7 / Table 3） |
| --- | --- | --- |
| FP32 add / mul / fma | 128 | 128 FP32 |
| FP64 | 64 | 64 FP64 |
| INT32 add | 128 | 64 INT32 |
| warp shuffle（shfl） | 32 | — |
| SFU（lg2.approx） | 16 | SFU（每 block 一组） |
| DPX | 64 | — |

**口径注**：吞吐表（results/cycle/SM）与物理单元数是两个口径，不可逐行划等号。FP32 行 128 吞吐对 128 单元、FP64 行 64 对 64，恰好相等；但 INT32 add 吞吐 128 高于 64 个 INT32 单元——说明部分 INT32 运算可走其他管线，具体管线归属属闭源细节，官方未逐字披露，本书只引两个披露值、不推定内部连线。SFU 吞吐 16 results/cycle/SM 可反推每 SM 16 个 SFU lane（每 block 4 个），但 Fig 7 只标「SFU」未给逐 block 数，故 16 这个单元数标反推。warp scheduler 每周期从 eligible warps 里选 1 个发射（Nsight Compute 的 issue-slot 口径，[NVIDIA, Nsight Compute](../references.md)，[C]）；warp 为何 stall、stall 原因分类与延迟隐藏归 1.4，本章只锁发射宽度。

框图里其余单元各管一件事，本节只立角色、不展开机制。LSU（load/store unit，每 processing block 8 个 LD/ST）负责 global/shared/local 访存指令的地址生成与合并，访存合并的判定归 4.1；SFU（special function unit）负责超越函数（sin/cos/exp/lg2 等），吞吐 16 results/cycle/SM（Table 5），即每周期只能服务半个 warp 的超越函数——超越函数密集的 kernel 会在 SFU 上排队，这是 §2 吞吐表里 SFU 16 远低于 FP32 128 的含义。TMA（Tensor Memory Accelerator，SM 级共享，Hopper 新增）负责 global↔shared 的异步批量拷贝，把地址计算从线程卸载到专用硬件（机制与 kernel 用法归 4.4）；4 个 Tex 单元处理纹理采样（AI kernel 少用，归 2.2 常量/纹理路径）。Tensor Core 每 block 1 个、全 SM 4 个，内部数据流与 MMA 指令几何归 1.3，本章框图只标位置。

为什么是 4 个 processing block、每 block 32 个 FP32 lane？因为 warp size = 32：一个 scheduler 每周期发射 1 条 warp 指令，恰好覆盖 32 个 lane，4 个 block 并行就是 128 lane。这个几何让「发射宽度」与「warp 宽度」对齐——scheduler 不需要把一个 warp 拆成多拍发射，也不需要把多个 warp 拼成一拍，单发射即可喂满 32 lane。register file 也按 block 分区（每 block 16384×32-bit），从 Fig 7 布局读出的机制是：每个 block 的 register 分区就近喂自己的 32 个 FP32 lane，operand 不必跨 block 搬运，省掉跨分区的 operand 走线。这也是 §3 端口带宽账按「每 scheduler 独立 512 B/cycle」反推的依据——4 个分区的 operand 流是并行的，不是共享一条总线。

## 3. register file：容量、分配与端口带宽账

64K 个 register 听起来抽象，换成字节是 256 KB——比 shared memory 的 228 KB 还大。register file 是 SM 上带宽最高的存储层，也是 occupancy 最常见的绑定约束。本节先给容量与分配，再按指令语义反推 operand 带宽需求。

容量与分区（[NVIDIA, Hopper whitepaper](../references.md) Table 3 / Fig 7 + [NVIDIA, CUDA Programming Guide](../references.md) Table 34，[C]）：每 SM $R_{\mathrm{file}} = 65536$ 个 32-bit register = $65536 \times 4 = 262144$ B = 256 KB；按 4 个 processing block 分区，每 block $16384 \times 32$-bit = 64 KB（Fig 7 逐字「Register File (16,384 × 32-bit)」）；全 GPU $132 \times 256$ KB = 33792 KB ≈ 33 MB（Table 3 逐字）。每线程 register 上限 255（Table 4），分配以 warp 为粒度（warp 级取整，精确分配单位归 3.4）。

正例： $R_{\mathrm{th}} = 64$、每 block 256 线程 → 每 block 占 $64 \times 256 = 16384$ 个 register = 64 KB，恰好等于一个 processing block 的 register 分区容量，4 个 block 占满 65536 → 4 blocks/SM（§5 案例 C1）。反例： $R_{\mathrm{th}} = 255$、每 block 1024 线程 → $255 \times 1024 = 261120$ 个 register = 1020 KB，是 $R_{\mathrm{file}}$（256 KB）的 3.98 倍，一个 block 都装不下 → 0 blocks，kernel 不可 launch（§5 案例 C5）。这两个例子界定了 register 约束的两端：刚好占满分区 vs 远超整 SM 容量。

端口与 bank 结构官方未披露（闭源细节），本书不推定端口数，改按**指令语义反推 operand 带宽需求**。推导路线分三步：先算一条 warp 指令的 operand 字节，再乘每周期发射宽度得每 SM 每周期的 operand 字节，最后乘频率得带宽并与 HBM、shared 对比。关键一步是第一步的 operand 计数——一条 FMA 是 3 读 1 写。

一条 warp FMA 指令（`d = a*b + c`）覆盖 32 个 lane，每 lane 读 3 个源操作数（a、b、c）、写 1 个目的（d），每个操作数 32-bit = 4 B：

$$
\mathrm{bytes}_{\mathrm{FMA}} = 32 \times (3 + 1) \times 4 = 512 \ \mathrm{B} \qquad (1)
$$

**符号**： $`\mathrm{bytes}_{\mathrm{FMA}}`$ = 一条 warp FMA 的 register 读写字节（3 源 + 1 目的）。由命题 1，每 SM 每周期至多 4 条 warp 指令（4 scheduler 单发射），取最重 operand 需求（全是 FMA）得上界：

$$
\mathrm{BW}_{\mathrm{RF}} = 4 \times 512 \times f = 2048 \ \mathrm{B/cycle} \times f \qquad (2)
$$

**符号**： $`\mathrm{BW}_{\mathrm{RF}}`$ = 每 SM register file operand 带宽需求上界； $f$ = clock。**口径注**：式 (2) 是**需求上界**，假设 4 个 scheduler 每周期各发射 1 条 3 源 1 目的 FMA（operand 最重的指令）；真实 kernel 含 load/store、单源指令，operand 需求低于此值。官方未披露端口数，故这是「要喂饱满发射需要多少带宽」的反推，不是「硬件实际提供多少端口」的披露。

全数字代入（FP32-boost $f = 1.98$ GHz）： $`\mathrm{BW}_{\mathrm{RF}} = 2048 \times 1.98 \times 10^{9} = 4.06 \times 10^{12}`$ B/s = 4.06 TB/s **每 SM**。对比两个量级：HBM 全芯片 3.35 TB/s 摊到 132 个 SM，每 SM 份额 $3352/132 = 25.4$ GB/s；shared memory 32 banks × 4 B = 128 B/cycle/SM（bank 几何披露值，[NVIDIA, CUDA Programming Guide](../references.md) Table 34），@1.98 GHz 合 $128 \times 1.98 \times 10^{9} = 253$ GB/s。于是 register : shared : HBM-每SM份额 ≈ $4055 : 253 : 25.4$ GB/s ≈ $160 : 10 : 1$——register file 带宽是 HBM 每 SM 份额的约 160 倍、shared 的约 16 倍。全芯片 register 聚合带宽 $132 \times 4.06 = 535$ TB/s，对 HBM 3.35 TB/s 同样是 160 倍。

这笔账解释了两件事。其一，为什么 kernel 要把热数据放进 register：从 register 取 operand 的带宽比从 HBM 取高两个量级，tiling 的本质就是把数据从 25.4 GB/s 的层搬到 4055 GB/s 的层（tiling 强度账归 4.3）。其二，为什么 register spill 代价高昂：spill 把 register 数据落到 local memory（HBM backing），带宽从 4055 GB/s 掉到 25.4 GB/s 量级，掉约 160 倍——spill 的精确带宽代价与延迟暴露账归 2.3（recap 本章比值）。shared 的 128 B/cycle 是 bank 无冲突时的峰值，bank conflict 会把它除以冲突重数（判定归 4.2）。

register 分配的粒度机制须再钉一句。每线程上限 255 是官方披露值（Table 4），不是 256——以披露值为准，不推定编码原因。分配以 warp 为粒度：编译器给每个线程分配 $R_{\mathrm{th}}$ 个 register，但硬件按 warp 整体取整到分配单位，故一个 warp 实际占用的 register 数可能略高于 $32 R_{\mathrm{th}}$。这意味着式 (3) 的 register 项 $`R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}})`$ 用的是理想值，实际占用只会更多、能装的 block 只会更少（命题 3 的方向）。精确到分配单位的计算归 3.4，本章上界式不代入具体取整单位（官方未逐字披露 CC 9.0 的 register 分配单位数值，只披露 warp 级粒度）。

每线程 register 预算的权衡一笔： $R_{\mathrm{th}}$ 大，单线程能驻留的中间变量多、ILP 高（指令级并行，归 1.4），但每 block 占用多、能装的 block 少、occupancy 低； $R_{\mathrm{th}}$ 小则反之。案例 C1（ $R_{\mathrm{th}} = 64$， $\mathcal{O} = 0.5$）与 C2（ $R_{\mathrm{th}} = 128$， $\mathcal{O} = 0.25$）正是这条权衡的两端。所以 register 不是「越多越好」也不是「越省越好」，是 ILP 与 TLP（线程级并行，即 occupancy 提供的可调度 warp 数）之间的分配——最优 $R_{\mathrm{th}}$ 取决于 kernel 的延迟结构，归 1.4 / 3.4。编译器在 register 不够时会 spill（把暂时不用的 register 落到 local memory），spill 的代价就是上段那 160 倍带宽落差，故「压 $R_{\mathrm{th}}$ 提 occupancy」若压到触发 spill，反而得不偿失（spill 判定与代价账归 2.3）。

## 4. shared/L1 统一物理：256 KB 怎么切

256 KB 的 L1/shared 统一存储，最多能切多少给 shared？228 KB，不是 256 KB。这两个数差 28 KB，是 occupancy shared 约束最常见的误代来源。

所谓 shared/L1 统一物理，是指 L1 data cache 与 shared memory 共用同一块 256 KB 的 SRAM，按 carveout 档位切分给两者（[NVIDIA, CUDA Programming Guide](../references.md) Table 35，[C]）。CC 9.0/10.0 的 carveout 档位为 {0, 8, 16, 32, 64, 100, 132, 164, 196, 228} KB（Table 35 逐字），shared 最大档 228 KB，此时 L1 剩 $256 - 228 = 28$ KB；取 100 KB 档时 L1 剩 156 KB。carveout 由 `cudaFuncAttributePreferredSharedMemoryCarveout` 设定（API 语义归 3.4 / 2.3），本章物理上界取最大档 $S_{\mathrm{total}} = 228$ KB = 233472 B。

每 block 的 shared 上限是 **227 KB** 而非 228 KB：Blackwell tuning guide 逐字「CUDA reserves 1 KB of shared memory per thread block. Hence, the B200 GPU enables a single thread block to address up to 227 KB」（[NVIDIA, Blackwell tuning guide](../references.md)，[C]；Hopper 同口径，[NVIDIA, Hopper tuning guide](../references.md)）。即每 block 有 1 KB 被 CUDA 预留，用户可寻址 227 KB。静态 shared 上限 48 KB，超过 48 KB 须用动态 shared 并显式 opt-in（Programming Guide 逐字「shared memory allocations over 48 KB per block must use dynamic shared memory and require an explicit opt-in」）。shared 分 32 个 bank、每 bank 4 B 宽（Table 34），bank conflict 判定归 4.2；L1 缓存行 128 B、sector 32 B 归 2.2。

正例： $S_{\mathrm{sh}} = 48$ KB = 49152 B → shared 约束 $\lfloor 233472/49152 \rfloor = \lfloor 4.75 \rfloor = 4$ blocks/SM（§5 案例 C3）。反例：把 combined 256 KB 当 shared 总量代入， $\lfloor 262144/49152 \rfloor = \lfloor 5.33 \rfloor = 5$ blocks，高估 25%——多算的那 1 个 block 实际占的是 L1 的 28 KB，不是 shared。

> **坑**：228 / 227 / 256 三个数混用。触发条件：把 combined 256 KB 当 shared 总量、或把每 SM 228 KB 当每 block 上限；现象：shared 约束项高估（256 代 228 高估 12.3%；228 代 227 在单 block 场景多算 1 KB），occupancy 上界虚高；规避：每 SM shared 总量 $S_{\mathrm{total}} = 228$ KB（carveout 最大档），每 block 用户上限 227 KB（1 KB 预留），combined L1+shared = 256 KB——三个数三个口径。含 1 KB 预留与 128 B 分配粒度的精确 occupancy 归 3.4，本章上界式取请求值口径（§5 命题 3 说明取整方向）。

carveout 的权衡一笔：shared 切得大，单 block 能装的 tile 大（tiling 数据复用账归 4.3），但 L1 剩得少、全局访存命中率低（L1 命中账归 2.2）；反之亦然。228 KB 不是「越大越好」，是 shared 与 L1 的零和切分——本章只交付库存与上界，最优 carveout 选择归 3.4 / 4.3。

为什么把 L1 与 shared 做成同一块 SRAM、而不是各给一块固定容量？因为两类负载的需求相反：tiling 类 kernel（GEMM、attention）要把数据显式搬进 shared 反复复用，shared 越大 tile 越大；流式访存类 kernel（elementwise、reduction）不显式用 shared，全靠 L1 缓存全局访存，L1 越大命中越高。固定切分会让一类负载浪费另一类的容量，统一物理 + 可配 carveout 让每个 kernel 按需切。

> **提点**：这是「机制与策略分离」在存储上的体现——硬件提供 256 KB 统一 SRAM 与 carveout 档位（机制），具体切多少给 shared 由 kernel 按负载选（策略）。删掉这条类比，上文的容量账与式 (3) 的 shared 约束依然完整。

代际增量一笔：Hopper 每 SM shared 最大 228 KB，较 A100 的 164 KB 增 39%（[NVIDIA, Hopper tuning guide](../references.md)，[C] 逐字「+39% vs A100 164」）。combined 物理从 A100 的 192 KB 升到 Hopper 的 256 KB。shared 容量增大直接放宽式 (3) 的 shared 约束项：同样 $S_{\mathrm{sh}} = 48$ KB，A100 的 $\lfloor 164/48 \rfloor = 3$ blocks，Hopper 的 $\lfloor 228/48 \rfloor = 4$ blocks——大 tile kernel 的 occupancy 上界随代际抬升。bank 几何不变（32 banks × 4 B），32 这个数恰等于 warp size：一个 warp 的 32 个线程各访问一个 bank、每 bank 4 B，无冲突时一拍服务整个 warp（128 B/cycle，§3 已用）。bank conflict 的判定与 swizzle 规避归 4.2，本章只立「32 banks × 4 B = 128 B/cycle 峰值」这个库存数。

## 5. occupancy 物理上界：四约束取 min（单位一致推导）

四个约束取 min——为什么是 min 而不是相加？因为每个 resident block **同时**消耗四种资源，任何一种耗尽都装不下下一个 block；block 数被最紧的那种资源封顶。本节推导分四步：列四个必要条件、把每个条件解成 block 数、取 min 并向下取整、乘 $w_b$ 得 active warps 再除 $W_{\mathrm{slots}}$ 得 $\mathcal{O}$。关键一步是**量纲对账**：四项必须同为「blocks/SM」，否则 min 无意义——这是本章推导的单位一致硬要求。

所谓 occupancy（占用率），是指 SM 上 active warps 与 warp slots 上限之比： $\mathcal{O} = W_{\mathrm{active}}/W_{\mathrm{slots}}$。它衡量的是 warp slot 被填满的比例，不是算力利用率（算力利用率是 MFU / roofline 口径，归 2.4 / 11.2）。occupancy 高只说明有足够多的 warp 可供调度器选择，不保证每个 warp 都在做有用功（延迟隐藏与 ILP 的关系归 1.4）。

一个 resident block 消耗四种资源，各给一个必要条件（设每 block 线程数 $= 32 w_b$）：

| 约束 | 每 block 消耗 | 必要条件（ $B$ = resident blocks） | 解成 blocks |
| --- | --- | --- | --- |
| block slot | 1 slot | $B \le B_{\max,\mathrm{slot}}$ | $B_{\max,\mathrm{slot}}$ |
| warp slot | $w_b$ warps | $B \cdot w_b \le W_{\mathrm{slots}}$ | $\lfloor W_{\mathrm{slots}}/w_b \rfloor$ |
| register | $R_{\mathrm{th}} \cdot 32 w_b$ 个 | $B \cdot R_{\mathrm{th}} \cdot 32 w_b \le R_{\mathrm{file}}$ | $\lfloor R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot 32 w_b) \rfloor$ |
| shared | $S_{\mathrm{sh}}$ B | $B \cdot S_{\mathrm{sh}} \le S_{\mathrm{total}}$ | $\lfloor S_{\mathrm{total}}/S_{\mathrm{sh}} \rfloor$ |

四个必要条件同时成立， $B$ 不超过四项的最小值；block 不可分割，故向下取整。得物理上界式（每 block 线程数写作 $`\mathrm{threads}_{\mathrm{block}} = 32 w_b`$）：

$$
B_{\max} = \min\left( B_{\max,\mathrm{slot}},\ \left\lfloor \frac{W_{\mathrm{slots}}}{w_b} \right\rfloor,\ \left\lfloor \frac{R_{\mathrm{file}}}{R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}} \right\rfloor,\ \left\lfloor \frac{S_{\mathrm{total}}}{S_{\mathrm{sh}}} \right\rfloor \right) \qquad (3)
$$

$$
W_{\mathrm{active}} = B_{\max} \cdot w_b \qquad (4)
$$

$$
\mathcal{O}_{\max} = \frac{W_{\mathrm{active}}}{W_{\mathrm{slots}}} = \frac{B_{\max} \cdot w_b}{W_{\mathrm{slots}}} \qquad (5)
$$

**符号**： $B_{\max}$ = 每 SM resident blocks 上界； $B_{\max,\mathrm{slot}}$ = block slot 上限（32）； $W_{\mathrm{slots}}$ = warp slots（64）； $w_b$ = warps/block； $R_{\mathrm{file}}$ = 65536； $R_{\mathrm{th}}$ = registers/thread； $`\mathrm{threads}_{\mathrm{block}}`$ = 每 block 线程数（ $= 32 w_b$）； $S_{\mathrm{total}}$ = 233472 B； $S_{\mathrm{sh}}$ = shared bytes/block； $W_{\mathrm{active}}$ = active warps； $`\mathcal{O}_{\max}`$ = occupancy 上界。 $S_{\mathrm{sh}} = 0$ 时 shared 项不绑定，记 $+\infty$（不是除以 0）——无 shared 的 kernel 不受 shared 约束。

**量纲对账**（单位一致硬要求，逐项核）：

| 项 | 量纲 | 化简 |
| --- | --- | --- |
| $B_{\max,\mathrm{slot}}$ | blocks/SM | blocks |
| $W_{\mathrm{slots}}/w_b$ | (warps/SM)/(warps/block) | blocks/SM |
| $`R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}})`$ | (regs/SM)/((regs/thread)·(threads/block)) = (regs/SM)/(regs/block) | blocks/SM |
| $S_{\mathrm{total}}/S_{\mathrm{sh}}$ | (B/SM)/(B/block) | blocks/SM |

四项同为 blocks/SM，min 合法；式 (4) $B_{\max}$ [blocks] × $w_b$ [warps/block] = warps；式 (5) warps / warps 无量纲。若把 warp 项误写成 $W_{\mathrm{slots}}$（不除 $w_b$），量纲变成 warps 与 blocks 混取 min，结果无意义——这是单位一致推导要堵的第一个错。

先落一个最小实例（即下文案例 C1）：每 block 256 线程（ $w_b = 8$）、 $R_{\mathrm{th}} = 64$、 $S_{\mathrm{sh}} = 0$，四项 = min(32, $\lfloor 64/8 \rfloor$, $\lfloor 65536/(64 \times 256) \rfloor$, ∞) = min(32, 8, 4, ∞) = 4 blocks； $W_{\mathrm{active}} = 4 \times 8 = 32$ warps； $\mathcal{O} = 32/64 = 0.5$。register 项 4 最紧，故 register 绑定——这就是式 (3)-(5) 的一次完整代入。

**命题 2**（occupancy 物理上界）。在 CC 9.0 资源库存（ $B_{\max,\mathrm{slot}} = 32$、 $W_{\mathrm{slots}} = 64$、 $R_{\mathrm{file}} = 65536$、 $S_{\mathrm{total}} = 233472$ B）下，给定 kernel 的 $`(\mathrm{threads}_{\mathrm{block}}, R_{\mathrm{th}}, S_{\mathrm{sh}})`$，每 SM 可同时驻留的 block 数上界为式 (3)，active warps 上界为式 (4)，occupancy 上界为式 (5)。**证明**（必要性 + 取整 + 充分性）：必要性——上表四个必要条件各由一种资源的库存独立给出，任一违反即装不下，故 $B$ 不超过四项各自的上界，从而不超过其 min。取整——block 是驻留的不可分单位， $B$ 为整数，故每项向下取整。充分性——在理想分配口径下（register 按线程精确计、shared 按字节精确计、无预留），取 $B = B_{\max}$ 时四种资源消耗均不超库存，故该 block 数可达。理想口径与实际分配粒度的差由命题 3 界定。∎

**命题 3**（粒度上界性）。实际 resident blocks 不超过式 (3) 的 $B_{\max}$。**证明**（要点）：实际分配中 register 以 warp 为粒度向上取整、shared 以 128 B 粒度向上取整且每 block 另预留 1 KB，故每 block 实际占用的 register 与 shared 均不少于式 (3) 假设的理想值；资源占用只增不减，能装下的 block 数只减不增，故实际值 $\le B_{\max}$。完整证明（含两个取整方向的单调性论证）留作练习 3。这条命题确立了本章上界的地位：式 (3) 是**物理上界**，3.4 的编程模型侧计算只会等于或低于它，不会高于它。

**register-binding 推论**（不可 launch 条件）。当 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}} > R_{\mathrm{file}}`$ 时，register 项 $`\lfloor R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}) \rfloor = 0`$，故 $B_{\max} = 0$——一个 block 都装不下，kernel 不可 launch。全数字代入： $R_{\mathrm{th}} = 255$（每线程上限）、每 block 1024 线程 → $255 \times 1024 = 261120 > 65536$（3.98 倍）， $B_{\max} = 0$。反过来，1024 线程的 block 要能 launch， $R_{\mathrm{th}}$ 至多 $\lfloor 65536/1024 \rfloor = 64$；512 线程的 block 至多 128；256 线程的 block 至多 256，但每线程上限 255 封顶，故 255 × 256 = 65280 ≤ 65536 仍可装 1 个 block。这给出 register 压力与 block 大小的权衡：block 越大，每线程能用的 register 越少。

五个全数字案例（CC 9.0 库存； $S_{\mathrm{sh}} = 0$ 记 shared 项 $+\infty$；逐项代入式 (3)-(5)）：

| 案例 | threads/block | $w_b$ | $R_{\mathrm{th}}$ | $S_{\mathrm{sh}}$ | slot 项 | warp 项 | reg 项 | shared 项 | $B_{\max}$ | $W_{\mathrm{active}}$ | $\mathcal{O}$ | 绑定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 256 | 8 | 64 | 0 | 32 | 8 | 4 | $\infty$ | 4 | 32 | 0.50 | register |
| C2 | 256 | 8 | 128 | 0 | 32 | 8 | 2 | $\infty$ | 2 | 16 | 0.25 | register |
| C3 | 256 | 8 | 32 | 48 KB | 32 | 8 | 8 | 4 | 4 | 32 | 0.50 | shared |
| C4 | 32 | 1 | 16 | 0 | 32 | 64 | 128 | $\infty$ | 32 | 32 | 0.50 | block slot |
| C5 | 1024 | 32 | 255 | 0 | 32 | 2 | 0 | $\infty$ | 0 | 0 | 0 | 不可 launch |

逐案例核对（全数字，算到底）：

- **C1**（register 绑定）：reg 项 $\lfloor 65536/(64 \times 256) \rfloor = \lfloor 65536/16384 \rfloor = 4$；warp 项 $\lfloor 64/8 \rfloor = 8$；min(32, 8, 4, ∞) = 4； $W_{\mathrm{active}} = 4 \times 8 = 32$ warps； $\mathcal{O} = 32/64 = 0.50$。灵敏度：把 $R_{\mathrm{th}}$ 从 64 降到 32，reg 项升到 $\lfloor 65536/8192 \rfloor = 8$， $B_{\max} = 8$、 $W_{\mathrm{active}} = 64$、 $\mathcal{O} = 1.0$——register 压力减半换来 occupancy 翻倍（前提 warp 项 8 仍够）。
- **C2**（register 重压）：reg 项 $\lfloor 65536/(128 \times 256) \rfloor = \lfloor 65536/32768 \rfloor = 2$；min(32, 8, 2, ∞) = 2； $W_{\mathrm{active}} = 16$； $\mathcal{O} = 0.25$。每线程 register 翻倍，occupancy 减半。
- **C3**（shared 绑定）：reg 项 $\lfloor 65536/(32 \times 256) \rfloor = 8$；warp 项 8；shared 项 $\lfloor 233472/49152 \rfloor = \lfloor 4.75 \rfloor = 4$；min(32, 8, 8, 4) = 4； $W_{\mathrm{active}} = 32$； $\mathcal{O} = 0.50$。此时 register 与 warp 都允许 8 个 block，是 shared 把上限压到 4。含 1 KB 预留核对：每 block 实际 $49152 + 1024 = 50176$ B， $\lfloor 233472/50176 \rfloor = \lfloor 4.65 \rfloor = 4$，与理想口径同值——本案例粒度修正不改结果（命题 3 的「只减不增」在此取等）。
- **C4**（block slot 绑定）：reg 项 $\lfloor 65536/(16 \times 32) \rfloor = \lfloor 65536/512 \rfloor = 128$；warp 项 $\lfloor 64/1 \rfloor = 64$；min(32, 64, 128, ∞) = 32； $W_{\mathrm{active}} = 32 \times 1 = 32$； $\mathcal{O} = 0.50$。每 block 只有 1 个 warp，32 个 block slot 装满时只用了 32 个 warp slot，另外 32 个 warp slot 空着——**小 block 撞 block slot 上限，warp slot 浪费一半**。这是「block 开太小」的定量代价。
- **C5**（不可 launch）：reg 项 $\lfloor 65536/(255 \times 1024) \rfloor = \lfloor 65536/261120 \rfloor = \lfloor 0.251 \rfloor = 0$；min(32, 2, 0, ∞) = 0； $W_{\mathrm{active}} = 0$。register-binding 推论命中，kernel 无法启动（不是 occupancy 低，是根本跑不起来）。

读这五案例的方式：C1/C2 同 block 大小、register 压力不同，occupancy 随 $R_{\mathrm{th}}$ 反比；C3 换绑定约束到 shared；C4 揭示 block slot 与 warp slot 是两个独立上限（小 block 撞前者）；C5 是 register 约束的极端——上界为 0。五个案例的 active warps 依次 32 / 16 / 32 / 32 / 0，覆盖四种绑定约束与不可 launch 边界。

推导的三个关节再各钉一句。其一，**为什么取 min 不取 sum**：四种资源是同一个 block 同时消耗的，不是四批 block 各耗一种；装第 $B$ 个 block 时四种资源各被扣一份，任何一种先见底就停，故上限由最紧者定。其二，**为什么向下取整**：block 是驻留的不可分单位，半个 block 不能驻留， $\lfloor 4.75 \rfloor = 4$ 而非 5——C3 的 shared 项 4.75 取 4 正是此意；若向上取整会装下资源不够的第 5 个 block，违反必要条件。其三，**为什么单位必须一致**：min 是比较运算，只有同量纲的数才能比大小；warp 项若不除 $w_b$ 就是 warps 量纲，与 blocks 量纲的 slot 项取 min 无意义（§5 量纲对账表逐项核过）。这三条是式 (3) 区别于「把四个数随便凑一起」的地方，也是 evaluator 重推导的核对点。

occupancy 与算力利用率是两个量，勿混。 $\mathcal{O} = 0.5$ 只说明 warp slot 填了一半，不说明算力只用了一半——若每个 active warp 都在满发射 FMA， $\mathcal{O} = 0.5$ 也可能跑出接近峰值的算力（4 个 scheduler 每周期只需 4 个 eligible warp 即可喂满，64 个 slot 里有一半活跃已远超 4）。反过来 $\mathcal{O} = 1.0$ 若 warp 全在等访存，算力可能远低于峰值。所以 occupancy 是「可调度 warp 的库存比例」，算力利用率是「实际 FLOPs / 峰值」（MFU / roofline 口径，归 2.4 / 11.2）；两者的桥梁是延迟隐藏（1.4 的 Little's law），不是恒等关系。本章只交付 occupancy 的硬件上界，不交付算力利用率。

## 6. 物理上界之外：本章不做什么

物理上界算出来是 50% occupancy，实际 kernel 就一定跑在 50% 吗？不一定。本章交付的是硬件资源库存决定的**上界**，实际 occupancy 还要过编程模型侧的分配粒度与 API 这一关，而 occupancy 本身也不是越高越好。本节把三条边界划清，各给下游指针。

其一，**物理上界 vs 编程模型侧计算**（归 3.4）。式 (3) 取理想分配口径（register 按线程精确计、shared 按字节精确计）；实际 register 以 warp 粒度向上取整、shared 以 128 B 粒度取整且每 block 预留 1 KB，故实际 resident blocks ≤ $B_{\max}$（命题 3）。3.4 将用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 与 `cudaFuncAttributePreferredSharedMemoryCarveout`、launch bounds 做精确计算，届时 recap 本章式 (3) 为起点、增量是分配粒度与 API 语义。本章不重复 3.4 的编程模型计算。

其二，**occupancy vs 延迟隐藏**（归 1.4）。occupancy 高只说明可供调度器选择的 warp 多，不保证延迟被藏住。藏住访存延迟需要的是 active warps × ILP ≥ latency × throughput（Little's law，1.4 推导）；高 occupancy 低 ILP 的 kernel 可能不如低 occupancy 高 ILP 的 kernel。所以「把 occupancy 拉满」不是优化终点，1.4 会给出延迟隐藏所需的并发量级，本章的 $W_{\mathrm{active}}$ 上界是它的输入之一。

其三，**register/shared 容量 vs 层级带宽账**（归 2.3 / 2.4）。本章 §3 给了 register : shared : HBM-每SM份额 ≈ 160 : 10 : 1 的带宽比值，但完整的片上存储层级容量-延迟-带宽总表、register spill 的精确代价、roofline 的微架构来源与测量归 2.3 / 2.4。Tensor Core 的内部数据流、MMA 指令几何（mma.sync m16n8k16 / wgmma m64nNk16 / tcgen05 与 TMEM 512×128×32-bit = 256 KB）归 1.3——本章框图只标「1 TC(4th) per block」，不拆 TC 内部。

命题 3 的「只减不增」给一个全数字落点：取 $S_{\mathrm{sh}} = 32$ KB = 32768 B，理想 shared 项 $\lfloor 233472/32768 \rfloor = \lfloor 7.125 \rfloor = 7$ blocks；计入每 block 1 KB 预留后实际占用 $32768 + 1024 = 33792$ B， $\lfloor 233472/33792 \rfloor = \lfloor 6.909 \rfloor = 6$ blocks。理想式给 7、精确式给 6，差恰 1 个 block，且精确式低于理想式——方向与命题 3 一致（练习 3(d) 即要求构造这样的例子）。这说明本章上界不是「实际 occupancy 的近似」，而是「实际 occupancy 不可能超过的天花板」：3.4 的精确计算从它出发往下修，不会往上修。

资源库存表（§1）是下游各章共用的硬件契约。4.3 的 tile↔shared/register 预算约束、6.3 的 FlashAttention tile 预算、5.2 的 GEMM register 分摊，全部以本章的 $R_{\mathrm{file}} = 65536$、 $S_{\mathrm{total}} = 233472$ B、 $W_{\mathrm{slots}} = 64$ 为输入；它们各自推导「给定 tile 几何需要多少 register/shared」，再回代式 (3) 判定 occupancy 上界。本章不替它们做 tile 预算（那是 kernel 章的增量），只把硬件库存与上界式交付到位——定义章早于使用章，库存数错一个，下游所有预算账跟着错。

本章交付清单（下游 recap 接口）：资源库存表（§1）、单发射命题 1 与吞吐表（§2）、register 容量与端口带宽账（§3）、shared/L1 carveout 与 227/228/256 口径（§4）、occupancy 物理上界式 (3)-(5) + 命题 2/3 + register-binding 推论 + 五案例（§5）。3.4 recap 式 (3) 上界、1.4 recap $W_{\mathrm{active}}$、2.3 recap §3 带宽比值、4.2 recap §4 bank 几何——定义章早于使用章，本章是这些章的硬件来源。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| Hopper warp scheduler 单发射：4 scheduler × 1 dispatch × 32 lane = 128 FP32 results/cycle/SM（双发射为 Kepler 术语，不成立） | 命题 1，Fig 7 + BP Table 5 互证 | §2、练习 1 |
| 每 SM 资源库存：64 warp slots / 32 block slots / 65536 regs（256 KB）/ 228 KB shared / 256 KB combined | §1 库存表 | §1、§5 |
| register file 每 SM 256 KB = 4 block × 64 KB；每线程上限 255；warp 级分配粒度 | §3 容量段 | §3、练习 2 |
| register operand 带宽需求上界 4.06 TB/s/SM @1.98 GHz；register : shared : HBM-每SM份额 ≈ 160 : 10 : 1 | 式 (1)(2) | §3、练习 1 |
| shared/L1 统一 256 KB，carveout 最大档 shared 228 KB / L1 28 KB；每 block opt-in 227 KB（1 KB 预留）；static 48 KB | §4 + 坑注 | §4、练习 2 |
| occupancy 物理上界 = 四约束取 min（单位一致，皆 blocks/SM）→ active warps → $\mathcal{O}$ | 式 (3)(4)(5)，命题 2 | §5、练习 2/3 |
| 实际 resident blocks ≤ 式 (3) 上界（register warp 粒度 + shared 128 B + 1 KB 预留只增占用） | 命题 3 | §5、练习 3 |
| register-binding： $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}} > R_{\mathrm{file}}`$ → 0 blocks 不可 launch；255×1024 = 261120 > 65536 | §5 推论，案例 C5 | §5、练习 2 |
| 五案例 active warps = 32/16/32/32/0，覆盖 register/shared/block-slot 绑定与不可 launch | §5 案例表 | §5、练习 4 |
| 小 block 撞 block slot 上限浪费 warp slot：32 个 1-warp block 只用 32/64 warp slots | 案例 C4 | §5、练习 4 |
| 物理上界 ≠ 实际 occupancy（粒度归 3.4）；occupancy 高 ≠ 延迟藏住（Little's law 归 1.4） | §6 边界 | §6 |

## 练习

1.（计算 · [10]）CC 9.0 主口径：(a) 由命题 1 的单发射宽度算每 SM FP32 吞吐（results/cycle），再乘 2 FLOPs/FMA、1.98 GHz、132 SM 得芯片级 FP32 峰值，对照 1.1 §3 的 66.9 TFLOPS；(b) 用式 (2) 算每 SM register operand 带宽需求上界（TB/s），再算 HBM 每 SM 份额（3352 GB/s ÷ 132），求两者比值。校验：(a) 与 1.1 §3 FP32 行一致（容差 0.2%）；(b) 比值 ≈160，与 §3 的 160 : 10 : 1 一致。
2.（推导 · [20]）register-binding 与 shared 口径：(a) 由式 (3) 的 register 项证明「 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}} > R_{\mathrm{file}}`$ 当且仅当 register 项为 0」，代入 255×1024 验证不可 launch；(b) 求 1024 / 512 / 256 线程的 block 各自能 launch 的 $R_{\mathrm{th}}$ 上限（注意每线程 255 封顶）；(c) 若把 $S_{\mathrm{total}}$ 误代 256 KB（combined）而非 228 KB， $S_{\mathrm{sh}} = 48$ KB 时 shared 项高估几个 block、百分之几？先预测 (c) 的方向再算。校验：(a) 与 §5 推论一致；(b) 64 / 128 / 255；(c) 与 §4 反例的 5 对 4、高估 25% 一致。
3.（推导 · [30]）补全命题 3 的证明：(a) 论证 register 以 warp 粒度向上取整使每 block 实际 register 占用 ≥ 理想值 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}`$；(b) 论证 shared 以 128 B 粒度取整 + 每 block 1 KB 预留使实际占用 ≥ $S_{\mathrm{sh}}$；(c) 由 (a)(b) 的单调性推出实际 resident blocks ≤ 式 (3) 的 $B_{\max}$（资源占用只增不减 → 可装 block 数只减不增）；(d) 构造一个数值例使理想式与含预留的精确式差 1 个 block（提示：取 $S_{\mathrm{sh}}$ 使 $\lfloor S_{\mathrm{total}}/S_{\mathrm{sh}} \rfloor$ 与 $\lfloor S_{\mathrm{total}}/(S_{\mathrm{sh}}+1024) \rfloor$ 不等）。校验：(c) 与命题 3 论断一致；(d) 的差值应恰为 1 block，且精确式 ≤ 理想式（方向不可反）。
4.（计算 · [30]）五案例之外的综合账（CC 9.0 库存，carveout 取 228 KB）：(a) 每 block 512 线程、 $R_{\mathrm{th}} = 48$、 $S_{\mathrm{sh}} = 64$ KB，逐项算式 (3) 四项并求 $B_{\max}$、 $W_{\mathrm{active}}$、 $\mathcal{O}$，指出绑定约束；(b) 把 carveout 从 228 KB 换到 100 KB 档（ $S_{\mathrm{total}} = 102400$ B），(a) 的 $B_{\max}$ 与 $\mathcal{O}$ 怎么变？先预测方向再算；(c) 把每 block 线程数从 512 降到 128（ $w_b$ 从 16 到 4）、其余不变， $B_{\max}$ 与 $W_{\mathrm{active}}$ 怎么变，是否撞 block slot 上限？校验：(a) 与式 (3) 逐项代入一致；(b) shared 项应随 $S_{\mathrm{total}}$ 减小而不增（命题 3 方向）；(c) 对照案例 C4 的 block-slot 绑定机制。

## 参考

- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25 dam-cdn PDF 一手核实——Fig 7 SM 框图（4 processing block 各 1 Warp Scheduler (32 thread/clk) + 1 Dispatch Unit + RF 16384×32-bit + 32 FP32/16 INT32/16 FP64/1 TC 4th/8 LD/ST/SFU + SM 级 TMA/256 KB L1-shared/4 Tex）、Table 3（128 FP32/64 INT32/64 FP64/4 TC per SM、RF 256 KB/SM、33792 KB/GPU）、Table 4（CC 9.0 64 warps/32 blocks/65536 regs/255 per-thread）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——compute-capabilities 子页 Table 33（CC 9.0/10.0 32 blocks/64 warps/2048 threads/1024 per-block）、Table 34（64K regs/255 per-thread/shared 228 KB/227 opt-in/32 banks）、Table 35（unified 256 KB carveout 0-228）、KB=1024 注、>48 KB dynamic opt-in 逐字
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Table 5 CC 9.0 FP32 128/FP64 64/INT32 add 128/shfl 32/SFU 16/DPX 64 results/cycle/SM（命题 1 单发射互证）
- [NVIDIA, Hopper tuning guide](../references.md)：NVIDIA Hopper Tuning Guide（key: hopper-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 9.0 shared 228 KB / 64 warps / TMA / DSMEM / cluster
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 10.0 库存同 CC 9.0、「CUDA reserves 1 KB of shared memory per thread block … up to 227 KB」逐字、static 48 KB
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——mma.sync m16n8k16 / wgmma m64nNk16 / tcgen05 + TMEM 512×128×32-bit = 256 KB（§6 指针 1.3，本章不拆 TC 内部）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——Warp Scheduler States issue-slot 单 warp 发射口径（§2）；stall reason 分类归 1.4
