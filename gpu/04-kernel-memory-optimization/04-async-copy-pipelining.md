# 4.4 异步拷贝与流水线（Asynchronous Copy and Pipelining）

> 4.3 的 tiled GEMM 每 tile 同步等待 906.8 ns、计算只有 139.9 ns——双缓冲把每 tile 压到 906.8 ns 仍被装载主导；把搬运交给 cp.async/TMA 后台执行、开 3 个 stage 让两份装载同时在飞，每 tile 降到带宽地板 645.2 ns（对同步 1.62×）。本章交付 cp.async（sm_80+，global→shared 免中间 register，commit/wait group 完成语义）、TMA（sm_90+，copy descriptor 硬件地址生成，tensor-map 128 B，mbarrier tx-count 按字节记账）、multi-stage 软件流水的延迟隐藏条件（命题 1： $(S-1)\tau \ge t_{\mathrm{lat}} + t_{\mathrm{load}}$，stage 数下界）、producer-consumer warp specialization、kernel 内 tile 级对 3.3 stream 级 overlap 的增量、CUTLASS mainloop pipeline 机制片段（5.2 全量回链）与 cp.async/TMA 适用场景账 | 前置：4.3（tiling 强度、双缓冲命题 4 与时间账式 (6)-(8)、tile 预算式 (5)）、3.3（stream 级 copy-compute overlap 与三资源流水）、1.3（wgmma 异步语义）、4.1（16 B 向量化粒度）、4.2（swizzle 与 TMA 硬件模式）、3.4（shared 预算与 occupancy）、1.4（Little's law）；下游 5.2（CUTLASS mainloop 全量走读，本章 recap 源）、6.3（FA3 warp specialization + TMA） | 锚点：PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C] 官方规格披露·一手权威）、Hopper whitepaper V1.04（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C]）、CUTLASS sm90_pipeline.hpp（[NVIDIA, CUTLASS](../references.md)，教学参照 + 官方实现，版本锚 4.9.0）、CUTLASS pipeline.md（[NVIDIA, CUTLASS docs](../references.md)，教学参照）、cuda-samples globalToShmemAsyncCopy（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）、Hopper Tuning Guide（[NVIDIA, Hopper tuning guide](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 139.9 ns 计算对 906.8 ns 等待：搬运为什么要异步

4.3 §6 的账留了一个缺口：双缓冲每 tile 906.8 ns 里，计算只占 139.9 ns，其余全在等装载——因为同步装载把「发射—等待—计算」串成一条依赖链，warp 发射 load 后只能 stall（1.4 的 execution dependency），等待期间执行资源空转。异步拷贝把这条链剪断：发射后立即返回，搬运在后台进行，warp 继续算当前 tile。本章回答三个问题：搬运怎么发射才不占执行资源（cp.async 免 register、TMA 硬件地址生成）？完成怎么判定（async-group 与 mbarrier 两套语义，为什么 `__syncthreads()` 不够）？几个 stage 才能把延迟藏住（命题 1 与 stage 数下界）？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0（sm_90，TMA 可用），bf16 输入、fp32 累加**；cp.async 机制自 sm_80（Ampere）起可用，本章数值例一律取 H100 口径、不代入 A100 绝对值。running example 沿 4.3： $M = N = K = 4096$ GEMM（**本章 $M, N, K$ = GEMM 问题维，非消息字节/参数量/序列长**），threadblock tile $T_M = T_N = 128$、 $T_K = 32$（tile 维，非序列长 $T$），256 threads/block，每 tile 装载 $(T_M + T_N) T_K \cdot \mathrm{bytes} = 16384$ B = 16 KB（A/B 各 8 KB）。时间账沿 4.3 式 (6)： $t_{\mathrm{comp}} = 139.9$ ns、 $t_{\mathrm{load}} = 645.2$ ns、 $t_{\mathrm{lat}} = 261.6$ ns（478.8 cycles @1.83 GHz，H800 微基准实测 [B]，沿 4.3 声明）； $\mathrm{BW} = 3352$ GB/s、每 SM 份额 25.4 GB/s（2.4 recap）。片上 KB = 1024 B。

符号声明： $S$ = pipeline stage 数（**含正在计算的 1 份**，预取缓冲 $S - 1$ 份；4.3 的 $n_{\mathrm{stage}}$ 即本章 $S$）； $\tau_S$ = $S$ stage 稳态每 tile 时间； $m_{\mathrm{tile}}$ = 每 tile 装载字节（本章 16384 B；小写 $m$，别于 GEMM 问题维 $M$）； $t_{\mathrm{comp}}, t_{\mathrm{load}}, t_{\mathrm{lat}}$ 沿 4.3 §6； $S_{\mathrm{sh}}$ = shared bytes/block； $\mathrm{bytes}$ = 每元素字节（bf16 取 2，bytes 不占字母）；tx-count 为 mbarrier 的字节计数（§3，文字术语不占符号）。B/KB 单位豁免。

**边界声明**：本章交付 kernel 内 tile 级的异步搬运与流水。stream 级（kernel 之间、copy engine、H2D/D2H）overlap 归 3.3（§4 末两级对照）；wgmma 指令语义与累加器几何归 1.3；CUTLASS mainloop 模块全量走读（CuTe layout、mma atom、epilogue、利用率账）归 5.2——**本章代码 scope 为机制片段级**（§2/§3 装载骨架、§5 pipeline 同步骨架），5.2 重访本章片段一律 recap + 指引；cluster 与 DSMEM 的编程模型细节归 3.1/5.2（本章只到 multicast 机制一句）；FA3 的 warp specialization 指令级走读归 6.3。

## 2. cp.async：global→shared 免中间 register

同步搬运贵在哪？BP 把两条路径并排说清（逐字，[NVIDIA, Best Practices](../references.md) §10.2.3.4，[C]）：「The synchronous version for the kernel loads an element from global memory to an intermediate register and then stores the intermediate register value to shared memory.」——同步版是 load 进 register、再 store 进 shared 两条指令，register 是必经中转。异步版逐字：「Using asynchronous copies does not use any intermediate register. Not using intermediate registers can help reduce register pressure and can increase kernel occupancy.」——**免中间 register 一笔、异步发射一笔**，两笔收益独立：前者降 register 压力（3.4 权衡 recap），后者藏延迟（§4）。

PTX 侧语义（[NVIDIA, PTX ISA](../references.md) §9.7.10.28.3，[C]）：`cp.async.ca.shared{::cta}.global [dst], [src], cp-size`，cp-size 取 4/8/16 字节，PTX 7.0、**sm_80+**；「non-blocking instruction」——发射后控制立即返回。缓存行为两档：`.ca` 全层级缓存、`.cg` 只缓 L2（bypass L1）；BP 逐字「If individual CUDA threads are copying elements of 16 bytes, the L1 cache can be bypassed.」——16 B 粒度正是 4.1 §4 的 128-bit 向量化上限，cp.async 把它接到 shared 侧。完成机制官方钉死（逐字）：「The executing thread can then use async-group based completion mechanism or the mbarrier based completion mechanism to wait for completion of the asynchronous copy operation. No other synchronization mechanism guarantees the completion of the asynchronous copy operations.」——**除 async-group 与 mbarrier 外，任何同步原语都不保证 cp.async 完成**，`__syncthreads()` 不在其列（坑注见下）。

group 语义三件套（PTX 逐字）：`cp.async.commit_group`「creates a new cp.async-group per thread and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group」——把未提交的拷贝打包成组；`cp.async.wait_group N`「wait till only N or fewer of the most recent cp.async-groups are pending ...」——等到至多 N 个最新组未完成，其余全完成；`wait_all` 等价 commit + wait_group 0。排序语义同样钉死（逐字）：「There is no ordering guarantee between two cp.async operations if they are not explicitly synchronized using cp.async.wait_all or cp.async.wait_group or mbarrier instructions.」——组间无序，依赖必须显式同步。

指令数账（本章口径）：16 KB tile 按 16 B 粒度需 $16384/16 = 1024$ 条 cp.async，256 线程摊 4 条/线程；每条仍要线程自己算地址——地址生成开销随 tile 字节线性涨，这是 TMA 的动机（§3）。教学骨架（CUDA pipeline API，BP 官方例同形；机制片段级）：

```cpp
// 教学片段，非优化实现：cp.async 双缓冲装载骨架（每线程 16 B，256 线程 × 4 条 = 16 KB tile）
load_tile_async(smem[0], A, B, k0);          // 序章：装 tile 0（内部 4 条 __pipeline_memcpy_async）
__pipeline_commit();                       // 打包成 group
for (int k0 = 0; k0 < K; k0 += TK) {
    if (k0 + TK < K) {
        load_tile_async(smem[buf ^ 1], A, B, k0 + TK);   // 预取下一 tile
        __pipeline_commit();
    }
    __pipeline_wait_prior(1);              // 等当前 tile 组完成，留最新 1 组在飞
    __syncthreads();                       // 跨线程可见：wait 只管本线程的拷贝
    compute_tile(smem[buf]);               // 外积 / wgmma（4.3 §7）
    __syncthreads();                       // 读毕，下一轮可覆写
    buf ^= 1;
}
```

`__pipeline_wait_prior(1)` 是双缓冲的关键一手：留最新 1 组（预取组）在飞、等其余组（当前 tile）完成——若写 `wait_prior(0)` 则预取也被等完，双缓冲退化成同步。cuda-samples `cpp/3_CUDA_Features/globalToShmemAsyncCopy`（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）是官方完整教学例，README 逐字「implements matrix multiplication which uses asynchronous copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.」——它用 arrive-wait barrier（mbarrier 路径）做完成同步，与 §3 衔接。

> **坑**：用 `__syncthreads()` 当 cp.async 的完成同步。触发条件：发射 cp.async 后只调 `__syncthreads()` 就读 shared；现象：PTX 逐字「No other synchronization mechanism guarantees the completion」——barrier 只同步线程到达，不保证异步拷贝落地，读到旧数据是未定义行为且可能静默；规避：先 `wait_group`/`wait_prior`（或 mbarrier wait）再 `__syncthreads()`——前者保本线程拷贝完成，后者保跨线程可见（wait 是 per-thread 语义，block 内其他线程的拷贝它不管），两步缺一不可。

## 3. TMA：descriptor 驱动的硬件搬运

cp.async 的地址生成还在软件手里，能不能也交给硬件？Hopper 的 TMA（Tensor Memory Accelerator，张量内存加速器）就是这一步。whitepaper 逐字（[NVIDIA, Hopper whitepaper](../references.md) TMA 节，[C]）：「TMA operations are launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing」——用张量维数与块坐标描述搬运，不再逐元素算地址；支持「1D-5D tensors」与大块数据（「up to the shared memory capacity」）。编程模型逐字：「the TMA programming model is single-threaded, where a single thread in a warp is elected to issue an asynchronous TMA operation (cuda::memcpy_async) to copy a tensor, and subsequently multiple threads can wait on a cuda::barrier for completion of the data transfer」——**warp 内选一个线程发射，其余线程等 barrier**。对 A100 的增量逐字：「On A100 ... asynchronous memory copies were executed using a special LoadGlobalStoreShared instruction, so the threads were responsible for generating all addresses and looping across the whole copy region. On Hopper, TMA takes care of everything.」——stride、offset、边界计算全部硬件接管（LDGSTS 即 cp.async 的指令形态，PG async-copies 子页同口径）。

PTX 侧（[NVIDIA, PTX ISA](../references.md) §9.7.10.28.5，[C]）：`cp.async.bulk.tensor.dim.dst.src...`，dim 取 1d–5d，PTX 8.0、**sm_90+**；global→shared 方向完成机制 `.mbarrier::complete_tx::bytes`，shared→global 方向 `.bulk_group`；load 模式含 `.tile`（本章 GEMM 用）与 `.im2col`（卷积）等。descriptor 的官方形态逐字（§5.5.8）：「The tensor-map is a 128-byte opaque object either in .const space or .param (kernel function parameter) space or .global space which describes the tensor properties and the access properties of the tensor data ...」——**tensor-map 是 128 B 不透明对象**，host 侧用 `cuTensorMapEncodeTiled` driver API 创建（PG async-copies 子页），kernel 侧以 `__grid_constant__` const 参数传入；搬运时硬件按 descriptor 生成地址，软件只给块坐标。

完成判定走 mbarrier（异步事务屏障）。whitepaper Async Transaction Barrier 节逐字：「instead of counting just thread arrivals, it also counts transactions ... The transaction count is essentially a byte count. The asynchronous transaction barrier will block threads at the Wait command until all the producer threads have performed an Arrive, and the sum of all the transaction counts reaches an expected value.」——**tx-count 本质是字节计数**：发射侧 expect_tx 记入应收字节，TMA 每完成一笔硬件自动 complete_tx 扣减，归零且到达数齐才放行。PTX 侧 mbarrier 是 shared 内 8 B 对齐的 `.b64` 对象，追踪 phase、pending arrivals 与 tx-count 三件套（§9.7.15.16），数量只受 shared 容量限（不像 bar.sync 固定 barrier 数）。Hopper 还把等待从 spin 改成 sleep（逐字：「New for Hopper is the ability for 'Waiting' threads to sleep until all other threads arrive. On previous chips, Waiting threads would spin on the barrier object in shared memory.」）——等待线程不再烧发射槽。

全数字账（本章口径）：每 stage 的 expect_tx = A tile $128 \times 32 \times 2 = 8192$ B + B tile $32 \times 128 \times 2 = 8192$ B = 16384 B，与 $m_{\mathrm{tile}}$ 一致——**tx-count 记的就是流量账的字节**，4.3 的强度账与这里的完成判定同一本账。descriptor 开销：A/B 两个 tensor-map 共 $2 \times 128 = 256$ B，host 创建一次、全 kernel 复用，对 2.147 GB 的 tile 流量占比 $1.2 \times 10^{-7}$——descriptor 本身可略，真正的节省在发射侧（§6）。硬件 swizzle：tensor-map 带 32/64/128 B swizzle 模式（4.2 §4 recap），TMA 写 shared 时直接按 swizzle 布局落位，软件免手工置换。multicast 一句：`.multicast::cluster` 让一次 global 读同时落多个 CTA 的 shared（cluster 内 DSMEM），HBM 流量按 cluster 大小摊薄——机制细节归 3.1/5.2。教学骨架（语义级，非可编译实现）：

```cpp
// 机制片段（语义骨架）：TMA + mbarrier（sm_90+）
// host: cuTensorMapEncodeTiled(...) → 128 B tensor-map
// kernel 参数: __grid_constant__ const CUtensorMap tensorMap
if (threadIdx.x == 0) {                    // elected 单线程发射（wp 逐字）
    // mbarrier.expect_tx(mbar, 16384);    // 记入本 stage 应收字节
    // cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
    //     [smem], [tensorMap, {coord_x, coord_y}], [mbar];
}
// 消费线程: mbarrier.try_wait(phase) —— 到达数与 tx-count 均达期望才放行
```

## 4. 软件流水线：multi-stage 与隐藏条件

双缓冲只有一份预取在飞，盖不住 906.8 ns 的装载完成时间——多开几份缓冲行不行？所谓软件流水线（software pipelining），是指把循环体的装载与计算拆成 producer/consumer 两个角色、用多份缓冲让多个迭代同时在飞的编程结构（CUTLASS 文档逐字：「In order to write a performant GEMM Kernel, software pipelining is critical to hide the latency of global memory loads.」，[NVIDIA, CUTLASS docs](../references.md) pipeline.md，教学参照）。推导路线分三步：先钉发射时序（第 $k+S-1$ 块的装载何时发出），再写稳态每 tile 时间的三项 max，最后解出 stage 数下界；关键一步是「 $S$ 份缓冲允许 $S-1$ 份装载同时在飞，每份的完成时间被 $S-1$ 个 tile 的计算覆盖」。

**命题 1**（multi-stage 稳态时间与 stage 数下界）。 $S$ stage 流水（ $S \ge 2$，含计算 stage）稳态每 tile 时间为

$$
\tau_S = \max\left( t_{\mathrm{comp}},\ t_{\mathrm{load}},\ \frac{t_{\mathrm{lat}} + t_{\mathrm{load}}}{S - 1} \right) \qquad (1)
$$

代入 $T_K = 32$、 $S = 3$： $\tau_3 = \max(139.9, 645.2, 453.4) = 645.2$ ns——带宽地板。consumer 不等待的充分条件是 $(S - 1) \cdot \tau_S \ge t_{\mathrm{lat}} + t_{\mathrm{load}}$；达到带宽/计算地板 $\max(t_{\mathrm{comp}}, t_{\mathrm{load}})$ 的最小 stage 数为

$$
S_{\min} = 1 + \left\lceil \frac{t_{\mathrm{lat}} + t_{\mathrm{load}}}{\max(t_{\mathrm{comp}}, t_{\mathrm{load}})} \right\rceil \qquad (2)
$$

**证明**（要点）：发射时序——consumer 开始算 tile $k$ 时，缓冲 $(k-1) \bmod S$ 刚释放，producer 立即发射 tile $k+S-1$ 的装载；该装载在 $S-1$ 个 tile 之后（ $(S-1)\tau_S$ 时间后）才被消费，故不等待条件为 $(S-1)\tau_S \ge t_{\mathrm{lat}} + t_{\mathrm{load}}$。稳态时间——每 tile 至少耗 $t_{\mathrm{comp}}$（计算串行）与 $t_{\mathrm{load}}$（带宽服务率），且 $S-1$ 份在飞装载摊到的完成时间下界为 $(t_{\mathrm{lat}} + t_{\mathrm{load}})/(S-1)$，三项取 max 即式 (1)；式 (2) 由不等待条件解出。完整证明（含发射时序归纳与三项下界的逐条论证）留作练习 3。∎

全数字代入（4.3 时间账， $t_{\mathrm{lat}} + t_{\mathrm{load}} = 906.8$ ns）：

| $T_K$ | $S = 2$ | $S = 3$ | $S = 4$ | $S_{\min}$ | shared（ $S_{\min}$ 份） | 对同步加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 906.8 ns | 645.2 ns | 645.2 ns | 3 | 48 KB | 1.622 |
| 64 | 1552.0 ns | 1290.4 ns | 1290.4 ns | 3 | 96 KB | 1.420 |

$T_K = 32$ 逐数核对： $S = 2$ 时第三项 $906.8/1 = 906.8$ 主导（双缓冲，退化为 4.3 命题 4）； $S = 3$ 时 $906.8/2 = 453.4$ 低于 $t_{\mathrm{load}} = 645.2$，带宽项主导——**再加 stage 无收益**（ $S = 4$ 同值），地板由强度决定（4.3 式 (2)），流水只能逼近、不能突破。对同步版（1046.7 ns）加速 $1046.7/645.2 = 1.622$。延迟主导推论：当 $t_{\mathrm{load}}$ 远小于 $t_{\mathrm{lat}}$（小 tile 或高有效带宽）且计算主导稳态（ $\tau_S = t_{\mathrm{comp}}$）时，不等待条件约化为 $(S-1) t_{\mathrm{comp}} \ge t_{\mathrm{lat}}$——**预取 stage 数 × tile 计算时间 ≥ HBM 延迟**（口径：stage 数指预取份数 $S-1$，不含计算份；若把计算份也计入覆盖延迟的份数，得 $S \cdot t_{\mathrm{comp}} \ge t_{\mathrm{lat}}$ 的宽松形）。代入： $T_K = 32$ 需 $S - 1 \ge 261.6/139.9 = 1.87$ 即 $S \ge 3$； $T_K = 64$ 需 $S - 1 \ge 261.6/279.8 = 0.93$ 即 $S \ge 2$——与 4.3 §6「 $T_K = 64$ 单 tile 计算盖住首字节延迟」的结论一致。

```
3-stage 流水（S = 3：2 份预取 + 1 份计算，T_K = 32）：
序章: 装 tile 0、tile 1（填满预取份）
计算: │ tile 0 │ tile 1 │ tile 2 │ tile 3 │ ...
装载:    │ tile 2 │ tile 3 │ tile 4 │ ...      ← 算 tile k 之初发射 tile k+2
在飞: 恒有 2 份装载重叠；2 × 645.2 = 1290.4 ≥ 906.8，完成被计算盖住
稳态 τ = max(139.9, 645.2, 453.4) = 645.2 ns（带宽地板）
```

stage 的容量代价（4.3 式 (5) recap）： $S_{\mathrm{sh}} = S \cdot m_{\mathrm{tile}}$， $S = 3$、 $T_K = 32$ 给 48 KB（ $S_{\mathrm{alloc}} = 50176$ B，shared 项 $\lfloor 233472/50176 \rfloor = 4$ blocks）； $T_K = 64$ 给 96 KB（shared 项 2 blocks）。本章配置 register 项 2 更紧（4.3 §5），occupancy 仍 0.25 不变——**stage 数在 register 绑定配置下免费加到 shared 项触顶为止**；shared 绑定配置下每加一份掉驻留 block，值不值由命题 1 的收益对 3.4 隐藏账判。

**producer-consumer warp specialization**。上面默认同一批线程既装载又计算；Hopper 的异步原语允许把两个角色拆给不同 warp：producer warp（elected 单线程发 TMA）专管搬运，consumer warp-group 专管 wgmma 计算（1.3 异步语义 recap），两者用 full/empty 两类 barrier 交接缓冲——full barrier 记「数据就绪」（tx-count 达期望），empty barrier 记「缓冲释放」（consumer 用毕）。CUTLASS 文档对角色的定义逐字（pipeline.md）：「Some are 'producers' that load data or perform computations to satisfy other threads' input data dependencies. The same or different threads may be 'consumers' that do other work with those input data dependencies, once they are satisfied.」收益是三条管线并行：TMA 搬运、Tensor Core wgmma、CUDA Core 辅助工作（地址/softmax 等）互不占发射槽——wgmma 发射后立即返回（1.3），consumer 无需 stall 等 MMA 结果。FA3 的 warp specialization + pingpong 是这条结构的注意力侧实例（指令级走读归 6.3）。

两级 overlap 对照（3.3 recap，不重推）：

| 维度 | 3.3 stream 级 | 本章 tile 级 |
| --- | --- | --- |
| 位置 | kernel 之间（host 提交） | kernel 内 mainloop |
| 搬运执行者 | copy engine（独立 DMA，不占 SM） | cp.async（SM 访存路径）/ TMA（专用硬件单元） |
| 数据路径 | H2D/D2H（PCIe） | HBM→shared |
| 粒度 | chunk（MB 级） | tile（KB 级） |
| 完成同步 | event / stream 顺序 | async-group / mbarrier |
| 重叠对象 | copy engine 对 SM | 装载对计算（SM 内） |

两级正交可叠加：3.3 把 H2D 切 chunk 藏进 kernel 执行，本章把 kernel 内的 HBM 装载藏进 tile 计算——前者治「数据进 GPU」，后者治「数据进 SM」。copy engine 不搬 HBM→shared（那是 SM 侧路径），所以 stream 级 overlap 替代不了本章流水。

## 5. CUTLASS mainloop pipeline 片段（机制片段级）

工业实现怎么组织这套同步？CUTLASS 把 producer/consumer 流水封装成 pipeline 类（[NVIDIA, CUTLASS](../references.md) `include/cutlass/pipeline/sm90_pipeline.hpp`，教学参照 + 官方实现，版本锚 4.9.0）：`PipelineState`（index/phase 两个字段，stage 索引循环、phase 翻转）、`PipelineTmaAsync<Stages, ClusterTransactionBarrier>`（TMA producer 专用），内部持 `full_barrier_[Stages]` 与 `empty_barrier_[Stages]` 两组 mbarrier——正是 §4 的两类 barrier。方法名与语义逐一对应：`producer_acquire`（等 empty barrier，阻塞到 consumer 释放该 stage）、`producer_expect_transaction`（mbarrier expect_tx 记账）、`producer_commit`、`consumer_wait`（等 full barrier）、`consumer_release`（empty barrier arrive）。结构骨架（机制片段级，非完整 mainloop）：

```cpp
// 机制片段：PipelineTmaAsync producer/consumer 同步骨架（对应 sm90_pipeline.hpp）
PipelineTmaAsync<Stages> pipe(...);        // full/empty barrier 各 Stages 份
// producer warp（elected 单线程发 TMA）：
for (int k = 0; k < K_tiles; ++k) {
    pipe.producer_acquire(state);          // 等 empty：缓冲已被 consumer 释放
    pipe.producer_expect_transaction(state, m_tile);   // expect_tx 记字节
    tma_load(smem[state.index], tensorMap, coords_k);  // 硬件搬运，完成自动 complete_tx
    pipe.producer_commit(state);           // TMA 场景可 No-Op（硬件已更新 barrier）
    ++state;                               // index 循环、phase 翻转
}
// consumer warp-group（wgmma 计算）：
for (int k = 0; k < K_tiles; ++k) {
    pipe.consumer_wait(state);             // 等 full：tx-count 达期望
    wgmma(acc, smem[state.index]);         // 异步 MMA（1.3）
    pipe.consumer_release(state);          // empty arrive：缓冲可复用于下一轮
    ++state;
}
```

`producer_commit` 在 TMA 场景可能是 No-Op，官方文档逐字（pipeline.md）：「This API may result in a No-Op in some cases, if the producer instructions also update the barrier stage associated automatically (e.g., TMA_based producer threads using the PipelineTmaAsync class).」——TMA 硬件完成时自动 complete_tx，软件无需再补一刀；cp.async 路径则须软件 commit（§2）。**scope 声明**：本片段只展示 pipeline 同步骨架，mainloop 全量（CuTe layout/tile、collective mainloop、mma atom、epilogue、利用率账）归 5.2 模块走读，5.2 recap 本片段作同步侧机制。

> **坑**：按旧版类名读当前 CUTLASS。触发条件：拿 FA3 时代（CUTLASS 3.x）文章里的 `PipelineTmaUmmaAsync` 去当前 main 找；现象：该类不在当前 sm90_pipeline.hpp（4.9.0 核实），warp-specialized UMMA 流水的类组织已变；规避：引 pipeline 类名以当前版本清单为准（PipelineTmaAsync / PipelineTmaStore / PipelineTransactionAsync / PipelineAsync），跨版本阅读先对 CHANGELOG。

## 6. cp.async 对 TMA：适用场景与开销账

两者怎么选？先把开销差算清。发射侧（本章 16 KB tile）：cp.async 需 1024 条指令、256 线程各 4 条、每条附地址算术；TMA 需 1 条指令（elected 线程）、地址由 descriptor 硬件生成——**发射槽与地址 register 两笔都省**，tile 越大差距越大（指令数随字节线性涨，TMA 恒 1）。边界处理：cp.async 越界须软件 predicate 或 src-size 零填充（PTX 语义），TMA 的边界计算硬件接管（wp 逐字「computing stride, offset, and boundary calculations」）。descriptor 成本：128 B/tensor、host 一次性创建（§3 账：占流量 $1.2 \times 10^{-7}$），对不规则小张量反而不划算。对照表：

| 维度 | cp.async（sm_80+） | TMA（sm_90+） |
| --- | --- | --- |
| 粒度 | 每线程 4/8/16 B | bulk tile（至 shared 容量） |
| 地址生成 | 软件逐线程 | descriptor 硬件 |
| 发射 | 全体线程各发 | elected 单线程 1 条 |
| 完成机制 | async-group（commit/wait） | mbarrier tx-count（字节记账） |
| 维数 | 1D 地址 | 1D–5D 张量 |
| 边界处理 | 软件 predicate/零填充 | 硬件 |
| swizzle | 软件布局 | 硬件模式（32/64/128 B） |
| multicast | 无 | cluster 内一次读多 CTA |

适用判断（教学口径，非官方规则）：tile 字节 KB 级以上、布局规则、边界多的场景（GEMM/FA/卷积 im2col）用 TMA——发射与边界开销摊得薄；细粒度、小 tile、散乱或非张量布局的场景（gather/scatter、reduction 的稀疏装载）用 cp.async——descriptor 建不起来或建了不值；要兼容 sm_80 的代码只有 cp.async 一条路。Hopper Tuning Guide 把 TMA 列为 CC 9.0 的调优要点（1D-5D global↔shared + cluster DSMEM，[NVIDIA, Hopper tuning guide](../references.md)，[C]），与本章判断同向。

## 7. 边界：本章不做什么

流水账算完，剩下的归谁？本章交付 cp.async 免 register 与 group 语义（§2）、TMA descriptor/tensor-map 128 B/mbarrier tx-count（§3）、multi-stage 稳态与 stage 下界（命题 1 + 式 (1)(2)）、warp specialization 与两级 overlap 对照（§4）、CUTLASS pipeline 同步骨架（§5）、cp.async/TMA 适用账（§6）；以下各有专章。CUTLASS mainloop 全量走读与 GEMM 利用率账归 5.2（本章命题 1 与 §5 片段是它的 recap 源）；FA3 warp specialization + pingpong 指令级走读归 6.3；wgmma/tcgen05 指令几何归 1.3；stream 级 overlap 与 CUDA Graph 归 3.3；cluster/DSMEM 编程模型归 3.1/5.2；occupancy 精确计算归 3.4；实测 drill-down 归 11.2。代码 scope：机制片段级（§2/§3/§5），不做模块走读。

本章交付清单（下游 recap 接口）：cp.async group 语义与免 register（§2）、TMA descriptor + tx-count（§3）、命题 1 + 式 (1)(2)（§4）、pipeline 同步骨架（§5）、适用对照表（§6）。5.2 recap 命题 1 作 mainloop pipeline 效率账的理论侧、recap §5 骨架作 collective mainloop 走读的同步侧；6.3 recap §4 warp specialization 作 FA3 的机制原型。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| cp.async（sm_80+）global→shared 免中间 register，降 register 压力；16 B 粒度可 bypass L1；非阻塞发射 | §2 BP/PTX 逐字 | §2、练习 1 |
| 完成只认 async-group 与 mbarrier：commit_group 打包、wait_group N 留最新 N 组；`__syncthreads()` 不保证完成；组间无序 | §2 PTX 逐字 + 坑注 | §2、练习 4 |
| 16 KB tile：cp.async 1024 条（4 条/线程）对 TMA 1 条；地址生成软件对硬件 | §2/§6 账 | §6、练习 1 |
| TMA（sm_90+）copy descriptor 按张量维数 + 块坐标搬运，1D–5D；elected 单线程发射；tensor-map 128 B opaque，host cuTensorMapEncodeTiled 创建 | §3 wp/PTX/PG 逐字 | §3、练习 1 |
| mbarrier tx-count 本质字节计数：expect_tx 记账、硬件 complete_tx 扣减；每 stage 16384 B（A/B 各 8192）；Hopper wait sleep 对 Ampere spin | §3 wp 逐字 | §3、练习 1 |
| multi-stage $\tau_S = \max(t_{\mathrm{comp}}, t_{\mathrm{load}}, (t_{\mathrm{lat}}+t_{\mathrm{load}})/(S-1))$；不等待条件 $(S-1)\tau_S \ge t_{\mathrm{lat}}+t_{\mathrm{load}}$ | 式 (1)，命题 1 | §4、练习 2/3 |
| $S_{\min} = 1 + \lceil (t_{\mathrm{lat}}+t_{\mathrm{load}})/\max(t_{\mathrm{comp}},t_{\mathrm{load}}) \rceil$；本章 tile 给 3，达带宽地板 645.2 ns、对同步 1.622×；再加 stage 无收益 | 式 (2) + §4 表 | §4、练习 2 |
| 延迟主导推论：预取 stage 数 × tile 计算时间 ≥ HBM 延迟；T_K=32 需 S≥3、T_K=64 需 S≥2 | §4 推论 | §4、练习 2 |
| stage 容量代价 $S \cdot m_{\mathrm{tile}}$：48/96 KB；register 绑定下 occupancy 无损；warp specialization 三管线并行（TMA/wgmma/CUDA Core） | §4 容量段 + 角色段 | §4、练习 2 |
| tile 级对 stream 级：kernel 内 HBM→shared 对 kernel 间 H2D/D2H；copy engine 不搬 HBM→shared，两级正交可叠加 | §4 对照表 | §4 |
| CUTLASS PipelineTmaAsync：full/empty barrier 各 Stages 份，producer_acquire/expect_transaction/commit 对 consumer_wait/release；TMA commit 可 No-Op；旧类名 PipelineTmaUmmaAsync 不在 4.9.0 | §5 骨架 + 逐字 + 坑注 | §5 |
| 适用：大 tile/多维/边界多用 TMA，细粒度/散乱/sm_80 兼容用 cp.async；descriptor 256 B 占流量 $1.2 \times 10^{-7}$ | §6 表 + 账 | §6 |

## 练习

1.（计算 · [10]）发射与记账代入（本章口径）：(a) 16 KB tile 按 16 B 粒度的 cp.async 条数与 256 线程的人均条数；(b) TMA 发射条数；(c) 每 stage mbarrier expect_tx 的字节值（A/B tile 分列）；(d) 两个 tensor-map 的 descriptor 总字节。校验：(a) 1024/4（§2）；(b) 1（§3）；(c) 8192 + 8192 = 16384 B（§3）；(d) 256 B（§3）。
2.（计算 · [20]）stage 数账（4.3 时间账： $t_{\mathrm{comp}} = 139.9$、 $t_{\mathrm{load}} = 645.2$、 $t_{\mathrm{lat}} = 261.6$ ns）：(a) $S = 2, 3, 4$ 的 $\tau_S$（式 (1) 三项逐一代入）；(b) $S_{\min}$ 与 shared 字节、shared 项 blocks（ $S_{\mathrm{alloc}}$ 含 1 KB 预留）；(c) 对同步版 1046.7 ns 的加速比；(d) 延迟主导推论下 $T_K = 64$（ $t_{\mathrm{comp}} = 279.8$ ns）的最小 $S$。先预测 (a) 中 $S = 4$ 对 $S = 3$ 的方向再算。校验：(a) 906.8/645.2/645.2 ns（§4 表）；(b) 3、48 KB、4 blocks；(c) 1.622；(d) $S \ge 2$（ $279.8 \ge 261.6$，与 4.3 §6 一致）。
3.（推导 · [30]）补全命题 1 的证明：(a) 证发射时序：consumer 开始 tile $k$ 时缓冲 $(k-1) \bmod S$ 释放、producer 发射 tile $k+S-1$，该装载被消费前经历 $(S-1)\tau_S$；(b) 证式 (1) 三项各为下界（计算串行、带宽服务率、 $S-1$ 份在飞摊完成时间），并说明贪心调度可达；(c) 验证 $S = 2$ 退化为 4.3 命题 4 的 $\max(t_{\mathrm{comp}}, t_{\mathrm{lat}}+t_{\mathrm{load}})$；(d) 求 $S \to \infty$ 极限并解释为什么流水不能突破带宽地板。校验：(c) 与 4.3 式 (7) 逐项一致；(d) $\max(t_{\mathrm{comp}}, t_{\mathrm{load}}) = 645.2$ ns（§4）。
4.（代码 · [30]）cuda-samples `cpp/3_CUDA_Features/globalToShmemAsyncCopy` 走读：(a) 找出拷贝完成同步用的原语（arrive-wait barrier），说明它对应本章哪套完成机制；(b) 解释为什么不能用 `__syncthreads()` 替代该原语（引 PTX 逐字）；(c) 若把该 sample 的 cp.async 装载换成 TMA，列出须替换的三个部件（descriptor 创建、发射方式、完成记账）；(d) 先预测该 sample 对同步版的方向（快/慢/不变）再对照 README 描述验证。校验：(a) mbarrier 路径（§3，README「arrive-wait barrier」逐字，公开文档锚）；(b)「No other synchronization mechanism guarantees the completion of the asynchronous copy operations」（§2 PTX 逐字）；(c) cuTensorMapEncodeTiled / elected 单线程 / expect_tx（§3）；(d) README 逐字「uses asynchronous copy ... when on compute capability 8.0 or higher」（方向与 §4 加速账一致）。

## 参考

- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——§9.7.10.28.3 cp.async（cp-size 4/8/16、PTX 7.0 sm_80+、.ca/.cg、non-blocking 逐字 + 「No other synchronization mechanism guarantees the completion of the asynchronous copy operations.」逐字 + 「There is no ordering guarantee between two cp.async operations ...」逐字）；§9.7.10.28.3.2/3 commit_group（per-thread batch uncommitted 逐字）/ wait_group N（「wait till only N or fewer of the most recent cp.async-groups are pending ...」逐字）；§9.7.10.28.5 cp.async.bulk.tensor（1d–5d、PTX 8.0 sm_90+、.tile/.im2col、global→shared .mbarrier::complete_tx::bytes、shared→global .bulk_group、.multicast::cluster）；§5.5.8 tensor-map（「The tensor-map is a 128-byte opaque object either in .const space or .param ... or .global space」逐字）+ swizzle 模式；§9.7.15.16 mbarrier（.b64 8B-aligned、phase + pending arrivals + tx-count、expect_tx/complete_tx、user-defined 数量仅受 shared 容量限）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——TMA 节（p.32-34）：「TMA operations are launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing」逐字 + 1D-5D tensors + single-threaded programming model（elected single thread issue cuda::memcpy_async、多线程 cuda::barrier wait）逐字 + A100 LoadGlobalStoreShared 线程负责全部地址生成 vs Hopper TMA 硬件接管 stride/offset/boundary 逐字；Async Transaction Barrier 节：「instead of counting just thread arrivals, it also counts transactions ... The transaction count is essentially a byte count.」逐字 + sum transaction counts reaches expected value + Hopper waiting threads sleep vs 前代 spin 逐字
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——async-copies 子页（LDGSTS CC 8.0+ / TMA CC 9.0+：tensor map 用 cuTensorMapEncodeTiled driver API host 创建、__grid_constant__ const 参数传入、global→shared mbarrier 完成 / shared→global bulk async-group 完成机制矩阵）；pipelines 子页（memcpy_async + barrier 组合）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——§10.2.3.4 Asynchronous Copy from Global Memory to Shared Memory（同步版 intermediate register 逐字 + 「Using asynchronous copies does not use any intermediate register. Not using intermediate registers can help reduce register pressure and can increase kernel occupancy.」逐字 + 「If individual CUDA threads are copying elements of 16 bytes, the L1 cache can be bypassed.」逐字 + __pipeline_memcpy_async/__pipeline_commit/__pipeline_wait_prior 官方微基准例）
- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，2026-11-25——include/cutlass/pipeline/sm90_pipeline.hpp（版本锚 4.9.0；PipelineState index/phase + `PipelineTmaAsync<Stages, ClusterTransactionBarrier>` + full_barrier_/empty_barrier_[Stages] + producer_acquire/producer_commit/producer_expect_transaction/consumer_wait/consumer_release 路径级核实；PipelineTmaStore/PipelineTransactionAsync/PipelineAsync 同文件；漂移：PipelineTmaUmmaAsync 旧 3.x 类不在当前 main）；mainloop 全量走读归 5.2
- [NVIDIA, CUTLASS docs](../references.md)：CUTLASS media/docs/cpp 文档（key: cutlass-docs），教学参照 + 官方实现，2026-11-25——pipeline.md（「In order to write a performant GEMM Kernel, software pipelining is critical to hide the latency of global memory loads.」逐字 + producer/consumer 角色定义逐字 + producer_acquire 阻塞语义 + producer_commit TMA 场景 No-Op 逐字）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/3_CUDA_Features/globalToShmemAsyncCopy（README 逐字「implements matrix multiplication which uses asynchronous copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.」；§2/练习 4 教学例）
- [NVIDIA, Hopper tuning guide](../references.md)：NVIDIA Hopper Tuning Guide（key: hopper-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 9.0 TMA 1D-5D global↔shared + cluster DSMEM 调优要点（§6 适用判断官方同向）
