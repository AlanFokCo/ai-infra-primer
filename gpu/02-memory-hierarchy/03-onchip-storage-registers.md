# 2.3 片上存储与 register（On-Chip Storage and Registers）

> register file 每 SM 256 KB，比 shared 的 228 KB 还大——最快的层为什么也是片上最大的层？kernel 的 register 预算不够时，被挤出去的数据要花多少带宽与延迟买回来？本章交付 register file 容量/分区与 bank/operand collector 机制（教学口径）、shared 的 32 bank 结构与容量配置（opt-in / carveout）、常量内存 64 KB 与 broadcast、纹理/surface 路径及其现代边界、register spill 到 local memory 的带宽-延迟双代价账（命题 1）与片上存储层级总结表 | 前置：1.2（RF 256 KB / 4 分区、operand 带宽账式 (1)(2)、shared/L1 carveout 228/227/256 口径、occupancy 上界式 (3)）、2.2（延迟表实测口径、常量/纹理路径在层级中的位置）、1.4（延迟隐藏与 ILP）；下游 4.2（bank conflict 判定与 swizzle）、3.4（occupancy 精确计算与 launch bounds）、2.4（层级带宽-延迟-容量全表与 roofline）、4.3（tile↔shared/register 预算） | 锚点：CUDA Programming Guide device memory 节与 legacy archive（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、PTX ISA .local 状态空间（[NVIDIA, PTX ISA](../references.md)，[C]）、Hopper whitepaper Table 3/4 + Fig 7（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、Hopper/Blackwell Tuning Guide（[NVIDIA, Hopper tuning guide](../references.md) / [NVIDIA, Blackwell tuning guide](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）、Nsight Compute memory chart / spill metrics（[NVIDIA, Nsight Compute](../references.md)，[C]）、Best Practices 延迟定性（[NVIDIA, Best Practices](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 256 KB 对 228 KB：片上存储的库存与边界

每 SM 的 register file 有 256 KB，比 shared memory 最大档 228 KB 还大（recap 1.2 §1 库存表）——片上最快的一层同时也是容量最大的一层，这不是巧合：register 要常驻全部 2048 个线程的架构状态，容量必须按线程数配。但 register 的容量是按线程切死的：每线程上限 255 个 32-bit register，预算不够时编译器把值 spill 到 local memory——物理上就是 HBM。本章回答三个问题：register file 的端口/bank 怎么组织，operand collector 解决什么矛盾？shared 的 bank 结构与容量配置各有哪些档位与上限？spill 的带宽与延迟代价怎么算，什么时候「压 register 提 occupancy」反而得不偿失？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 1.1/1.2）；B200 CC 10.0 资源库存相同（64K regs / 228 KB shared / 256 KB combined，recap 1.2），本章结论跨两代同形。单位口径：片上存储 KB = 1024 B（Programming Guide 逐字注，recap 1.2）；显存 GB = $10^{9}$ B、带宽 GB/s 按 $10^{9}$ B/s（2.1 口径）。**实测口径声明**（沿 1.4/2.2）：精确延迟 cycles（shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8）来自 H800 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B]），非 H100 官方规格；周期 ↔ 时间换算取 TC-boost 1830 MHz（BF16 kernel 口径，沿 1.4），register operand 带宽账沿用 1.2 的 FP32-boost 1980 MHz——两个 clock 各归其精度类，不混用（1.1 §3 坑注）。shared 峰值 GB/s 换算沿 1.2 的 1.98 GHz（§7 表 253 GB/s），最坏例与延迟换算同取 1.83 GHz（§3 stride-32 例 7.3 GB/s）——两 clock 下最坏 shared 均低于 HBM 每 SM 份额 25.4 GB/s，结论同向。

符号声明（spec §4.3，recap 1.2）： $R_{\mathrm{th}}$ = registers/thread； $R_{\mathrm{file}}$ = 每 SM register file 容量（65536 个 32-bit register）； $S_{\mathrm{sh}}$ = shared bytes/block； $S_{\mathrm{total}}$ = 每 SM shared 总量（carveout 最大档 228 KB = 233472 B）； $M_{\mathrm{spill}}$ = spill 总流量字节（本章局部符号，读写合计）； $n_{\mathrm{spill}}$ = 单 warp 每轮 spill load 条数； $\mathrm{MLP}$ = memory-level parallelism，每 warp 在飞独立 load 数（1.4 $\mathrm{ILP}$ 的访存口径）； $k$ = bank 冲突重数； $t_{\mathrm{HBM}}$ = 478.8 cycles（实测 [B]）。bytes 不占字母，B/KB/MB/GB 是单位符号豁免。

**边界声明**：本章交付片上存储的**容量、机制与 spill 代价账**。shared bank conflict 的完整判定规则与 swizzle 规避归 4.2（本章只立 bank 结构与一个最坏实例）；occupancy 的编程模型侧精确计算（分配粒度、launch bounds、occupancy API）归 3.4；层级带宽-延迟-容量全表（加 HBM/NVLink/PCIe）与带宽测量归 2.4；常量/纹理路径在缓存层级中的位置已 recap 2.2 §7，本章交付其容量与编程语义；TMA/异步拷贝与 DSMEM 的 kernel 用法归 4.4。代码 scope：机制片段级（§3 shared 配置片段）。

## 2. register file：容量、分区与 operand 供给

先回答一个更基本的问题：为什么 register 访问不算「访存延迟」？因为 register 读出不是一条 load 指令，而是执行管线内的 operand 读出阶段——FMA 发射时源操作数从 register file 取出送执行单元，这段走线在管线时序内消化（recap 2.2 §2 口径注：register 层不列入 load-to-use 延迟表）。所以 register 的代价不在延迟表里，在**容量约束**（occupancy，1.2 式 (3)）与**operand 供给带宽**（1.2 式 (2)）两笔账里。

容量与分区 recap 1.2 §3（不重推）：每 SM $R_{\mathrm{file}} = 65536$ 个 32-bit register = 256 KB；按 4 个 processing block 分区，每 block 16384 × 32-bit = 64 KB（whitepaper Fig 7 逐字「Register File (16,384 × 32-bit)」，[NVIDIA, Hopper whitepaper](../references.md)，[C]）；全 GPU $132 \times 256$ KB = 33792 KB ≈ 33 MB（Table 3 逐字）；每线程上限 255（Table 4），分配以 warp 为粒度。分区的机制理由 recap 1.2 §2：每个 block 的 register 分区就近喂自己的 32 个 FP32 lane，operand 不跨 block 搬运。

端口与 bank 结构官方未披露（闭源细节，recap 1.2 §3 口径）。本节给**机制层教学模型**（口径：教学模型描述 bank 分区与 operand 收集的一般机制，不是对 Hopper 具体 bank 数/collector 实现的断言；官方披露值只有容量与分区）。所谓 register bank 冲突，是指一条指令的多个源操作数落在同一 bank，而一个 bank 每周期只能服务一个操作数读出，operand 读出被串行化。所谓 operand collector（操作数收集器），是指 register file 与执行管线之间的缓冲结构，跨多个周期把一条指令所需的 operand 收集齐——收集期间调度器可以先发射 operand 已就绪的其他指令，把「收集 operand」与「发射指令」解耦。没有 collector 时，一条指令的 bank 冲突会卡住发射槽；有 collector 时，冲突的代价被缓冲吸收，发射侧尽量不空转。

```
register file operand 供给（教学模型，bank 数为假设口径）
┌────────────────────────────────────────────────────┐
│ register file 分区（每 processing block 64 KB）     │
│  bank 0   bank 1   bank 2   bank 3   ← 假设 4 bank │
│  r0,r4,…  r1,r5,…  r2,r6,…  r3,r7,…               │
└────┬────────┬────────┬────────┬────────────────────┘
     │ 同 bank 多 operand → 冲突，逐周期串行读出
     ▼        ▼        ▼        ▼
┌────────────────────────────────────────────────────┐
│ operand collector（缓冲，跨周期收集齐 3 源 operand）│
│  收集未完成 → 该指令等待；调度器先发别的就绪指令    │
└──────────────────────┬─────────────────────────────┘
                       ▼ operand 齐 → dispatch → 执行管线
```

教学实例（**假设 4 bank**，bank 号 = register 号除 4 的余数；Hopper 实际 bank 数未披露）：FMA 读 r1、r2、r3 三个源——bank 1/2/3 各一个，无冲突，operand 一拍取齐；改读 r4、r8、r12——三个都在 bank 0，3 路冲突，读出要 3 拍。这就是 1.2 式 (2) 口径注里「需求上界假设无冲突」的含义：2048 B/cycle 是 4 条 FMA 满发射的 operand 需求，bank 冲突把有效供给压到其下，发射侧表现为 stall（1.4 依赖追踪口径）。Hopper 在指令侧给了一条 disclosed 的减压路径：wgmma 的 B 操作数直接从 shared 经描述符喂 Tensor Core、跳过 register 中转（recap 1.3 §4，PTX ISA 逐字，[NVIDIA, PTX ISA](../references.md)，[C]）——operand 压力从 register 堆转给 shared，MMA 每 FLOP 的 register 字节降到约 0.25 B/FLOP（1.3 §4）。

register 与 shared 的容量-延迟权衡一笔：register 线程私有、operand 读出零 load 延迟，但容量按线程切死，且多占一个 register 就少装一个 block（1.2 式 (3) register 项）；shared block 内共享、一份数据服务全 block，但访问要显式 load/store（29.0 cycles [B]）。全数字对照：32 个线程各需要同一个 128 B 的 tile 片段——放 register 是 32 份副本共 $32 \times 128 = 4096$ B = 4 KB 的 RF 占用；放 shared 是 1 份 128 B，代价是每次访问 29 cycles。**线程私有的热数据放 register，线程共享的数据放 shared**——这条放置原则的账面根据就是「副本数 × 容量」对「一份容量 + 访问延迟」的权衡（tiling 的预算约束归 4.3）。

## 3. shared memory：bank 结构与容量配置

shared 的 128 B/cycle 峰值带宽从哪来？从 bank 几何来。Programming Guide 逐字：「Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle.」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——32 个 bank、连续 32-bit 字映射到连续 bank、每 bank 每周期 32 bit。所以 bank 号 = 字地址（字节地址 ÷ 4）除 32 的余数，一个 warp 的 32 个线程各访问一个 bank、每 bank 4 B，无冲突时一拍服务整个 warp： $32 \times 4 = 128$ B/cycle（recap 1.2 §4 库存数）。

冲突与例外（PG 逐字口径）：同一 warp 的多个线程访问**同一 bank 的不同地址** → bank conflict，该 bank 的访问串行化直到所有请求线程拿到数据；两个例外是访问**同一地址**——读访问把该字 broadcast 给全部请求线程，写访问由其中一个线程写入（哪个线程写是 undefined）。全数字三模式对照（fp32 warp load，峰值 128 B/cycle）：

| 访问模式 | bank 分布 | 冲突重数 $k$ | 有效带宽 |
| --- | --- | --- | --- |
| stride-1（连续） | 32 线程落 32 个不同 bank | 1（无冲突） | 128 B/cycle |
| stride-2（隔一个字） | 落 16 个 bank、每 bank 2 地址 | 2 | 64 B/cycle |
| stride-32（列访问 32 宽数组） | 全落 bank 0 | 32 | 4 B/cycle |
| 全线程同一地址 | 同 bank 同地址 | broadcast | 一拍（例外） |

stride-32 的 32 路冲突把有效带宽压到峰值的 $1/32$： $128/32 = 4$ B/cycle——这是「shared 比 HBM 快」的最坏反例，4 B/cycle @1.83 GHz 合 7.3 GB/s，甚至低于 HBM 每 SM 份额 25.4 GB/s（2.4 §4）。**bank 结构给的是峰值，访问模式决定拿到几成**。完整判定规则（stride 饱和、64-bit 访问、swizzle 规避）归 4.2，本节只立结构与最坏账；实测侧 Nsight Compute 有 bank conflict 计数器 `l1tex__data_bank_conflicts_pipe_lsu.sum`（[NVIDIA, Nsight Compute](../references.md)，[C]），读法归 4.2/11.2。

容量配置三个旋钮（recap 1.2 §4 的 228/227/256 口径，本节补 API 语义）：

- **静态上限 48 KB**：`__shared__` 静态数组超过 48 KB 不可用；PG 逐字「Kernels relying on shared memory allocations over 48 KB per block are architecture-specific … must use dynamic shared memory … require an explicit opt-in using cudaFuncSetAttribute」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。
- **动态 opt-in**：`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` 显式抬高每 block 动态 shared 上限，launch 第三参传字节数；每 block 用户可寻址上限 227 KB（CUDA 预留 1 KB，Blackwell tuning guide 逐字，Hopper 同口径）。
- **carveout**：`cudaFuncAttributePreferredSharedMemoryCarveout` 设 shared/L1 切分偏好（整数百分比或 Default/MaxL1/MaxShared 枚举），是 hint 非保证——driver 可选不同配置，百分比向上圆整到最近档位（PG 逐字，档位 {0, 8, 16, 32, 64, 100, 132, 164, 196, 228} KB recap 1.2）。

代码片段（教学用，非优化实现；机制片段级，PG 示例同形）：

```cuda
// 动态 shared 96 KB opt-in（机制例证，非优化实现）
__global__ void tile_kernel(float* out) {
    extern __shared__ float buf[];   // 动态 shared，大小由 launch 第三参定
    // ... tile 数据搬入 buf、block 内复用（tiling 账归 4.3）
}
// host 侧：
int maxbytes = 98304;                // 96 KB > 48 KB，必须 opt-in
cudaFuncSetAttribute(tile_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
tile_kernel<<<grid, block, maxbytes>>>(out);
```

全数字 occupancy 落点（recap 1.2 式 (3) shared 项，理想口径）：opt-in 96 KB = 98304 B，carveout 须切到不小于 96 KB 的档位 → 100 KB 档（ $S_{\mathrm{total}} = 102400$ B），shared 项 $\lfloor 102400/98304 \rfloor = 1$ block/SM——大 tile 把 occupancy 压到 1 个 block；对照 48 KB（1.2 案例 C3）的 $\lfloor 233472/49152 \rfloor = 4$ blocks。**shared 切得大，tile 大、复用好，但能装的 block 少**——这是 1.2 §4 零和切分的 occupancy 侧代价，最优 carveout 选择归 3.4/4.3。含 1 KB 预留核对： $98304 + 1024 = 99328 \le 102400$，单 block 装得下，与 227 KB 上限口径自洽。

Hopper 增量一笔：CC 9.0 的 thread block cluster 让 block 访问同 cluster 内其他 block 的 shared——distributed shared memory（DSMEM，PG §2.3.3.8 逐字，[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。实测 SM-to-SM DSMEM 延迟 180 cycles（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B]），是本地 shared 29.0 的约 6.2 倍——跨 SM 共享仍比 HBM（478.8）快得多，但比本地 shared 贵；cluster 与 TMA 的 kernel 用法归 4.4，本章只立库存与延迟量级。

## 4. 常量内存：64 KB 与 broadcast

全 warp 读同一个系数，走哪条路最便宜？常量路径。常量内存总量 64 KB（PG compute capabilities 表「Constant memory size 64 KB」，[NVIDIA, CUDA Programming Guide](../references.md)，[C]），每 SM 常量缓存工作集 8 KB（同表逐字）；变量用 `__constant__` 声明、host 侧 `cudaMemcpyToSymbol` 写入，grid 作用域、应用生命周期、kernel 只读（PG §2.3.3.5 逐字口径）。

broadcast 机制（legacy Programming Guide 逐字：「A request is then split into as many separate requests as there are different memory addresses in the initial request, decreasing throughput by a factor equal to the number of separate requests.」，[NVIDIA, CUDA Programming Guide](../references.md)，[C]）：一条 warp 请求按 distinct 地址数拆分——全 warp 同一地址拆成 1 个请求、broadcast 给 32 线程；32 个不同地址拆成 32 个请求串行服务，吞吐降 32 倍。这与 shared 的 broadcast 例外（§3）同构：**uniform 访问是常量路径的适用区，分歧访问是其反例**。

全数字对照：warp 读同一系数 $c$ → 1 次常量缓存访问服务 32 线程；每线程查各自的表项（32 个不同地址）→ 32 次串行访问，吞吐为 uniform 的 $1/32$。后者甚至不如走 global 的 coalesced 读（一次 128 B、4 sector，4.1）——所以「per-thread 查找表」不该放常量内存。适用判据一句话：**数据小（装进 64 KB）、只读、warp 内 uniform**（系数表、超参、kernel 配置）；三个条件缺一个就换路径（per-thread 数据走 global/L1，block 共享走 shared）。常量路径在缓存层级中的位置（DCC/IDC/IMC 常量缓存层级）recap 2.2 §7，不重述。

## 5. 纹理/surface：空间局部性路径与现代边界

纹理路径为 2D 空间局部性优化。legacy Programming Guide 逐字：「The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency.」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——注意后半句的口径：纹理缓存命中省的是 DRAM 带宽，**不省 fetch 延迟**（constant-latency 设计），这与 L1 命中直接降延迟（40.7 cycles）是两种机制。

纹理硬件三件套（legacy PG texture object 节逐字口径）：

- **寻址模式**：Border / Clamp / Wrap / Mirror 四种，越界坐标由硬件处理（Wrap/Mirror 只支持归一化坐标）；归一化坐标把纹理映射到 $[0, 1 - 1/N]$ 区间，与纹理尺寸解耦。
- **过滤模式**：Point（取最近 texel）或 Linear（硬件插值）——1D 线性、2D 双线性、3D 三线性，官方口径是「low-precision interpolation」（低精度插值，非 FP32 精确算术）。插值由 Tex 单元硬件完成（每 SM 4 个 Tex，recap 1.2 Fig 7），一条 fetch 指令出插值结果，不占 CUDA Core。
- **对象 API**：`cudaCreateTextureObject` 创建纹理对象；surface 是其可读写对应物——CUDA array 带 `cudaArraySurfaceLoadStore` 标志后经 `cudaCreateSurfaceObject` 创建，`surf2Dread`/`surf2Dwrite` 读写，**字节寻址**（纹理按元素寻址，surface 的 x 坐标要乘元素字节数，legacy PG 逐字）。

现代边界（现行 Programming Guide 逐字，必须钉死）：「On all currently supported GPUs … use of texture and surface memory instructions no longer provides any performance benefit for non-texture loads.」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——在 H100/Blackwell 上，把普通 global 数据伪装成纹理去读**没有性能收益**；AI kernel 的只读数据走 `__ldg` / `ld.global.nc` 只读缓存路径（recap 2.2 §7），不走纹理采样。纹理/surface 指令的现存适用区是真纹理数据（渲染、OptiX hit shader 类负载）与需要硬件插值/边界处理的场景。所以本节交付的是机制库存与历史路径的现代表述：旧代码里的「纹理加速」口径在 Hopper 上不成立，迁移时按只读路径重写。

## 6. register spill：local memory 的带宽-延迟双代价账

register 预算不够时数据去哪？Programming Guide 逐字定义：「Register spilling means that values currently stored on-chip in registers must be written out to global memory and then read back later to make space for other values.」local memory 是线程私有存储，「The 'local' label refers to its logical scope, not its physical location」——物理位置在 global memory 空间（HBM backing），「local memory accesses have the same latency and bandwidth as global memory accesses」（均 PG §2.3.3 逐字，[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。触发条件三类（PG 逐字）：编译期无法确定常量索引的数组、过大的结构/数组、register 超预算（即 spill）。每线程 local memory 上限 512 KB（PG compute capabilities 表逐字）。

spill 的访存是 coalesced 的：local memory 按「连续 32-bit 字由连续 thread ID 访问」排布，warp 内各线程访问同一相对地址（同一数组下标/结构成员）时完全合并（PG 逐字）——一条 warp spill load 搬 $32 \times 4 = 128$ B = 4 sector，与 coalesced global load 同形（4.1）。所以 spill 的代价不在散乱访存，在**流量本身**：本来零 HBM 流量的 register 数据，现在每次用都要走一遍 L1/L2/HBM 路径。

检测与控制的官方手段：编译侧用 `-ptx`/`-keep` 看 PTX 汇编——spill 变量以 `.local` 声明、以 `ld.local`/`st.local` 访问（legacy PG 逐字；PTX 指令语义见 [NVIDIA, PTX ISA](../references.md)，[C]）；profiling 侧 Nsight memory chart 的 local 路径与 spill 请求计数器（`derived__shared_spilling_requests` 是 shared spill 口径，local spill 看 memory chart local 流量，[NVIDIA, Nsight Compute](../references.md)，[C]）。控制旋钮：`-maxrregcount`、`__launch_bounds__`、`__maxnreg__`（PG 逐字）——三者都是压 $R_{\mathrm{th}}$，方向与 occupancy 收益一致、与 spill 风险相反（1.2 §3 权衡）。

代价账推导路线分三步：先数 spill 流量得带宽口径时间，再数在飞批次得延迟口径暴露，最后说明两口径独立、实际耗时取其大者。关键一步是**延迟按批次计**——MLP 条独立 load 同批在飞、一批一个 $t_{\mathrm{HBM}}$，不是每条 load 各付一个全程延迟（1.4 Little's law 口径）。

**命题 1**（spill 双代价账）。(a) 带宽口径：设到达 HBM 的 spill 总流量为 $M_{\mathrm{spill}}$ 字节（读写合计），则其占用 HBM 的时间 $t_{\mathrm{BW}} = M_{\mathrm{spill}}/\mathrm{BW}$（芯片口径；per-SM 口径除以份额 $\mathrm{BW}/n_{\mathrm{SM}}$）。(b) 延迟口径：单 warp 每轮 $n_{\mathrm{spill}}$ 条 spill load、每 warp 至多 $\mathrm{MLP}$ 条独立 load 在飞，全 miss 最坏口径下暴露延迟为

$$
t_{\mathrm{lat}} = \left\lceil \frac{n_{\mathrm{spill}}}{\mathrm{MLP}} \right\rceil \cdot t_{\mathrm{HBM}} \qquad (1)
$$

**证明**（要点）：(a) 带宽 = 字节/时间的定义式重排，HBM 服务率上限即 $\mathrm{BW}$；(b) load 按 $\mathrm{MLP}$ 条一批在飞，每批从发射到数据返回付一个全程延迟 $t_{\mathrm{HBM}}$（load-to-use 口径，2.2）， $n_{\mathrm{spill}}$ 条需 $\lceil n_{\mathrm{spill}}/\mathrm{MLP} \rceil$ 批。完整证明（含取整方向论证与两口径独立性）留作练习 3。∎

带宽口径式记为式 (2)：

$$
t_{\mathrm{BW}} = \frac{M_{\mathrm{spill}}}{\mathrm{BW}} \qquad (2)
$$

**符号**： $M_{\mathrm{spill}}$ = spill 总流量字节（store + load 合计）； $\mathrm{BW}$ = HBM 标称带宽 3352 GB/s（2.1 式 (1)）； $n_{\mathrm{spill}}$ = 每轮 spill load 条数； $\mathrm{MLP}$ = 每 warp 在飞独立 load 数； $t_{\mathrm{HBM}}$ = 478.8 cycles（实测 [B]）。**口径注**：式 (2) 是 spill 流量**独占**带宽的时间（上界口径）——实际 kernel 的 spill 与正常访存共享带宽，spill 的边际代价取决于它挤占了多少正常流量；式 (1) 是全 miss 最坏口径，local 访问经 L1/L2（PG：L2 缓存 register spill；Nsight glossary：local 与 global 同延迟口径），spill 工作集小、常命中 L1（40.7 cycles），实际暴露低于式 (1)。两口径各给一个界，实际耗时不低于其大者、不高于两者之和的粗界。

全数字实例（H100 SXM 主口径；教学假设的 kernel 形态）：某 SM 驻留 $W_{\mathrm{active}} = 32$ warps（ $\mathcal{O} = 0.5$），每 warp 循环 1000 次，每次迭代每线程 spill 4 个 register（4 条 store + 4 条 load）：

- **流量**：每 warp 每迭代 $32 \times 4 \times 4 \times 2 = 1024$ B（32 线程 × 4 register × 4 B × 读写 2 份）；每 SM 每迭代 $32 \times 1024 = 32768$ B；1000 迭代合计 $32.768$ MB。
- **带宽账**（式 (2)，per-SM 份额口径）：份额 $3352/132 = 25.4$ GB/s（recap 1.2 §3）， $t_{\mathrm{BW}} = 32.768 \times 10^{6}/(25.4 \times 10^{9}) \approx 1.29$ ms。
- **延迟账**（式 (1)）： $n_{\mathrm{spill}} = 4$。MLP 4 → $\lceil 4/4 \rceil = 1$ 批 → 每迭代 478.8 cycles，1000 迭代 478800 cycles @1.83 GHz ≈ 0.262 ms；MLP 1 → 4 批 → 每迭代 1915.2 cycles，合计 ≈ 1.05 ms。

读这笔账的方式：每线程只 spill 4 个 register，spill 时间就到毫秒量级——与 kernel 本体同量级；带宽侧从 register 的 4055 GB/s/SM 掉到 HBM 份额 25.4 GB/s/SM，约 160 倍落差（recap 1.2 §3 比值）。MLP 从 1 提到 4，延迟暴露降 4 倍（1.05 → 0.262 ms）——这是 1.4 ILP 账在 spill 侧的直接应用：spill load 之间若无依赖，编译器/硬件可让它们同批在飞。

> **坑**：压 $R_{\mathrm{th}}$ 提 occupancy 压到触发 spill。触发条件：`-maxrregcount`/`__launch_bounds__` 把 register 预算压得低于 kernel 实际需求；现象：occupancy 上界升了，但 spill 流量把 HBM 份额吃掉、kernel 反而变慢——1.2 §3「压 $R_{\mathrm{th}}$ 若触发 spill 得不偿失」的定量兑现；规避：改预算前后对比 PTX `.local` 声明与 Nsight local 流量，spill 字节数显著上升即回退；最优 $R_{\mathrm{th}}$ 的判定归 3.4（register pressure vs ILP 权衡）。

## 7. 片上存储层级总结表

片上四层放在一起，容量、延迟、带宽各怎么排？总表如下（H100 SXM 主口径；延迟实测 [B]、容量官方 [C]、带宽口径分列）：

| 层 | 作用域 / 管理 | 容量 | 延迟（实测 [B]） | 带宽口径 |
| --- | --- | --- | --- | --- |
| register | 线程私有 / 编译器分配 | 256 KB/SM（65536 × 32-bit，4 分区） | operand 读出在执行管线内（非 load 延迟口径） | 需求上界 2048 B/cycle/SM ≈ 4055 GB/s @1.98 GHz（1.2 式 (2)，无冲突假设） |
| shared | block 共享 / 用户管理 | 228 KB/SM 最大档（每 block 227 KB opt-in） | 29.0 cycles | 128 B/cycle/SM = 253 GB/s @1.98 GHz（32 banks × 4 B，无冲突峰值） |
| L1 | SM 私有 / 硬件管理 | 256 KB 统一 − shared carveout | 40.7 cycles | 官方未披露每周期带宽（与 shared 同一 SRAM；测量归 2.4） |
| L2 | 全芯片共享 / 硬件管理 | 50 MB（两分区各 25 MB） | 263.0 cycles | 官方未披露（测量归 2.4） |

（常量缓存工作集 8 KB/SM、纹理路径见 §4/§5；HBM 行与 NVLink/PCIe 行、各层带宽的归一化对比与测量方法归 2.4 全表。）

读表的方式沿 2.2 §4：每往下一层，容量大一个量级以上（256 KB → 50 MB → 80 GB），延迟逐级抬升（管线内 → 29 → 263 → 478.8 cycles），没有「又快又大」的层。数据放置的决策因此是：**线程私有热数据 → register；block 内共享 → shared；全 warp uniform 只读 → 常量；流式只读 → L1/L2 主路径或只读旁路；其余 → HBM**。spill 是这条放置链的失败模式——register 装不下的值被强制降到 HBM backing 的 local，跨越整个层级（§6 双代价账）。tiling 把 HBM 访问换成 shared/register 访问（4.3），本质是把工作集沿这张表往上搬；搬动的预算约束（tile 几何 ↔ $S_{\mathrm{sh}}$ / $R_{\mathrm{th}}$）与收益账（强度 $O(1) \to O(T)$）归 4.3，occupancy 侧的精确取舍归 3.4。

本章交付清单（下游 recap 接口）：register 容量/分区与 bank/operand collector 教学模型（§2）、shared bank 结构与配置三旋钮（§3）、常量 64 KB broadcast 判据（§4）、纹理/surface 机制与现代边界（§5）、spill 双代价账式 (1)(2) + 命题 1（§6）、片上层级总结表（§7）。4.2 recap §3 bank 几何作 conflict 判定的硬件侧；3.4 recap §3 配置旋钮与 §6 spill 风险作 occupancy 精确计算的输入；2.4 recap §7 表作层级全表的片上部分；4.3 recap §2 放置原则作 tile 预算的出发点。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| register file 256 KB/SM = 4 分区 × 64 KB，每线程上限 255、warp 粒度分配；operand 读出在执行管线内，非 load 延迟口径 | §2 + 1.2 §3 recap | §2、§7 |
| bank 冲突 + operand collector 为教学模型（官方未披露 bank 数）；假设 4 bank 下 r4/r8/r12 同 bank 0 → 3 路冲突 3 拍 | §2 教学实例 | §2 |
| wgmma B 操作数走 shared 描述符跳过 register（disclosed 减压路径），MMA 约 0.25 B/FLOP register 字节 | §2 + 1.3 §4 recap | §2 |
| shared 32 banks × 4 B = 128 B/cycle 峰值；stride-1 无冲突 / stride-2 二路 64 B/cycle / stride-32 全落 bank 0 → 4 B/cycle（1/32）；同地址读 broadcast | §3 表（PG 逐字） | §3、练习 1 |
| shared 配置三旋钮：静态 48 KB 上限、>48 KB 动态 opt-in（MaxDynamicSharedMemorySize）、carveout hint 向上圆整；96 KB opt-in → 100 KB 档 → 1 block/SM | §3 + 1.2 式 (3) shared 项 | §3、练习 2 |
| DSMEM（CC 9.0 cluster）跨 SM shared 访问 180 cycles [B] ≈ 本地 shared 6.2 倍 | §3 增量段 | §3 |
| 常量 64 KB、每 SM 常量缓存工作集 8 KB；uniform 1 请求 broadcast vs 32 distinct 地址 32 路串行（1/32 吞吐）；per-thread 查找表是反例 | §4（PG/legacy 逐字） | §4 |
| 纹理缓存 2D 空间局部性、命中省带宽不省 fetch 延迟；硬件插值 low-precision；现代 GPU 上非纹理 load 无收益（只读走 __ldg） | §5（PG 逐字） | §5 |
| spill = register 值写 global 再读回；local 物理在 HBM、与 global 同延迟带宽、同相对地址完全 coalesced；每线程 local 上限 512 KB | §6（PG 逐字） | §6、练习 4 |
| spill 双代价账：带宽 t_BW = M_spill/BW；延迟 t_lat = ceil(n_spill/MLP)·t_HBM；32 warps × 1000 迭代 × 4 reg → 32.768 MB → 1.29 ms（带宽）/ 0.262–1.05 ms（延迟，MLP 4/1） | 式 (1)(2)，命题 1 | §6、练习 3/4 |
| 压 R_th 触发 spill 得不偿失：检测看 PTX .local 与 Nsight local 流量 | §6 坑注 | §6 |
| 片上层级表：register 256 KB/管线内/4055 GB/s 需求上界；shared 228 KB/29.0/253 GB/s；L1 28–256 KB/40.7/未披露；L2 50 MB/263.0/未披露 | §7 表 | §7、2.4 |

## 练习

1.（计算 · [10]）shared bank 结构代入（32 banks × 4 B，峰值 128 B/cycle）：(a) fp32 数组 stride-1、stride-2、stride-32 三种 warp 访问的冲突重数与有效带宽各多少？(b) 全线程读同一地址走什么机制、几拍？(c) stride-32 的有效带宽 @1.83 GHz 折合多少 GB/s，与 HBM 每 SM 份额 25.4 GB/s 比谁大？先预测 (c) 的方向再算。校验：(a) 1/2/32 路，128/64/4 B/cycle，与 §3 表一致（PG Figure 15/17 口径）；(b) broadcast 一拍；(c) $4 \times 1.83 \times 10^{9} = 7.3$ GB/s，低于 25.4——最坏 bank 冲突比 HBM 还慢。
2.（计算 · [20]）shared 配置与 occupancy（CC 9.0，理想口径 recap 1.2 式 (3)）：(a) 动态 shared 96 KB 须不须 opt-in？carveout 最小切到哪档？(b) 该档下 shared 项给几个 block/SM？(c) 换 48 KB 静态 shared（carveout 228 KB 档）给几个 block？(d) 96 KB + 1 KB 预留是否超每 block 227 KB 用户上限？校验：(a) 须 opt-in（>48 KB），100 KB 档；(b) $\lfloor 102400/98304 \rfloor = 1$；(c) $\lfloor 233472/49152 \rfloor = 4$，与 1.2 案例 C3 一致；(d) $99328 \le 232448$（227 KB），不超。
3.（推导 · [30]）补全命题 1 的证明（spill 双代价账）：(a) 带宽口径——由带宽定义推式 (2)，做量纲对账（字节 ÷ (字节/s) = s），并说明 per-SM 口径为何除份额 $\mathrm{BW}/n_{\mathrm{SM}}$；(b) 延迟口径——论证 load 按 MLP 条一批在飞、每批付一个全程 $t_{\mathrm{HBM}}$，批数为什么向上取整（ $\lceil \cdot \rceil$ 而非 $\lfloor \cdot \rfloor$ ，提示：剩 1 条也要付一批延迟）；(c) 数值验证 §6 实例的 MLP 4 与 MLP 1 两列（0.262 / 1.05 ms）；(d) 说明两口径为什么独立、实际耗时为什么取其大者而非相加。校验：(b) $\lceil 4/4 \rceil = 1$ 、 $\lceil 4/1 \rceil = 4$ ，方向不可反；(c) 与 §6 逐数一致（容差 1%）；(d) 与 §6 口径注「各给一个界」一致。
4.（计算 · [30]）spill 综合账（H100 SXM，教学假设形态）：kernel 每 SM 驻留 16 warps（ $\mathcal{O} = 0.25$），每 warp 循环 2000 次，每迭代每线程 spill 8 个 register（8 store + 8 load）：(a) 每 SM spill 总流量多少 MB？(b) 带宽口径时间（份额 25.4 GB/s）？(c) MLP 2 的延迟口径时间 @1.83 GHz？(d) 与 §6 实例（32 warps/1000 迭代/4 reg）比，流量与两口径时间各怎么变，方向与命题 1 一致吗？先预测 (d) 再算。校验：(a) $16 \times 2000 \times 32 \times 8 \times 4 \times 2 = 65.536$ MB；(b) ≈2.58 ms；(c) $\lceil 8/2 \rceil = 4$ 批 × 478.8 × 2000 / $1.83 \times 10^{9}$ ≈ 2.09 ms；(d) 流量翻倍、带宽账翻倍、延迟账随批数与迭代数变化，与式 (1)(2) 的线性/取整结构一致。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——writing-cuda-kernels 子页 §2.3.3 device memory（Registers/spill 定义、Local Memory「logical scope not physical location」+ 与 global 同延迟带宽 + coalesced 排布、Constant Memory __constant__/cudaMemcpyToSymbol/64 KB、Caches、Texture and Surface「no longer provides any performance benefit for non-texture loads」逐字、DSMEM §2.3.3.8）；advanced-kernel-programming 子页（>48 KB dynamic opt-in 逐字 + cudaFuncAttributeMaxDynamicSharedMemorySize 96 KB 示例、carveout hint）；compute-capabilities 子页 Table 34（local/thread 512 KB、constant 64 KB、32 banks）；legacy archive 13.0.0（constant broadcast「split into as many separate requests」逐字、texture 2D spatial locality + constant-latency 逐字、filtering low-precision interpolation、addressing modes、surface 字节寻址、constant cache working set 8 KB、PTX .local/ld.local/st.local 检测逐字）。**URL 漂移注**：CUDA 13.5 子页重组，device memory 节在 02-basics/writing-cuda-kernels.html（旧单页 understanding-memory.html 现为 Unified and System Memory），opt-in/carveout 在 03-advanced/advanced-kernel-programming.html；legacy 细节在 archive/13.0.0 单页
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——.local 状态空间与 ld.local/st.local 指令语义（§6 spill 检测）；wgmma shared 描述符 operand 路径（§2，recap 1.3）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——Table 3（RF 256 KB/SM、33792 KB/GPU）、Table 4（65536 regs、255/thread）、Fig 7（RF 16384×32-bit × 4 分区、4 Tex）
- [NVIDIA, Hopper tuning guide](../references.md)：NVIDIA Hopper Tuning Guide（key: hopper-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 9.0 shared 228 KB、TMA/DSMEM/cluster（§3 增量段）
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——「CUDA reserves 1 KB of shared memory per thread block … up to 227 KB」逐字、static 48 KB（§3）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV：shared 29.0 / L1 40.7 / L2 263.0 / global 478.8 cycles；SM-to-SM DSMEM 180 cycles（§3/§6/§7 实测口径）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——memory chart local 路径、glossary「Local memory has the same latency as global memory」、derived__shared_spilling_requests（shared spill 口径）、l1tex__data_bank_conflicts_pipe_lsu.sum（§3/§6）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——global/local「hundreds of clock cycles」定性（§1 实测口径声明，精确 cycles 挂 [B]）
