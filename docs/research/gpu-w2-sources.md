# gpu W2 research pass — P3/P4 CUDA/PTX 一手核实证据记录（2026-11-25）

Task 1：P3（CUDA 编程模型 3.1-3.4）+ P4（kernel 访存优化 4.1-4.4）一手来源核实 + `gpu/references.md` P3/P4 扩充。本文件记录核实通道、逐源证据（URL/引句）、⟨verify⟩ 销项/残留、版本漂移注、网络 fallback。writer（T2-T5）引用白名单 = `gpu/references.md` P3/P4 条目 + 跨 Part 共享锚点（P1 cuda-programming-guide / ptx-isa / cuda-best-practices / nsight-compute / cutlass / hopper-whitepaper / hopper-tuning-guide，P3 cuda-samples）。

## 0. 结果摘要

- **parse-check（pure ESM）**：`groups 11 entries 68`（≥64 地板过；W1 64 → +4）。
- **per-Part**：P1=12 / P2=4 / P3=4 / P4=1 / P5=9 / P6=10 / P7=6 / P8=6 / P9=6 / P10=4 / P11=6 = 68。
- **证据等级分布**：[C]=32 / 教学参照=17 / [A]=10 / [B]=8 / [W]=1（W1 [C]29/教学参照16 → 本轮 +3 [C] fermi-whitepaper/cuda-graphs-blog/libcudacxx-memory-model、+1 教学参照 cutlass-docs）。
- **污染检查全过**：68 数据行均恰 7 字段；无 escaped `\|`；无重复 key；日期均 YYYY-MM-DD；等级单值无混合档。
- **新增 4 条**：P3 `fermi-whitepaper`（GigaThread 逐字，[C] 官方规格）/ `cuda-graphs-blog`（launch 开销 μs 实测，[C] 开放博客）/ `libcudacxx-memory-model`（thread_scope 五 scope，[C] 官方规格）；P4 `cutlass-docs`（media/docs/cpp，教学参照，4.3 首引，自 cutlass repo 条目拆分）。
- **更新 7 条**：cuda-programming-guide（13.5 子页地图 + PDF/archive URL）/ ptx-isa（9.5 fence/atom/cp.async/TMA/mbarrier/ldmatrix/ld.v4/cache-ops/state-spaces/divergence）/ cuda-best-practices（13.5 coalescing/banks/occupancy/register-pressure）/ nsight-compute（Occupancy Calculator + Sectors/Req，**被引 += 3.4**）/ hopper-whitepaper（TMA + Async Transaction Barrier 节）/ cutlass（4.9.0 版本锚 + swizzle/pipeline 路径级，**被引 4.3 → cutlass-docs**）/ cuda-samples（cpp/ 重组路径级）。核实日期统一 2026-11-25。
- **⟨verify⟩ W2 销项**：CUTLASS swizzle 路径（spec 4.2）、CUTLASS pipeline 路径（spec 4.4）、occupancy calculator 现行形态（spec 3.4）、GigaThread 出处（spec 3.1 内容）、graph launch ~μs 出处（spec 3.3 内容）、PG 13.5 子页地图（spec 3.1 ⟨verify URL⟩ W1 已销本轮再确认）、cuda-samples 现行路径。**残留**：CUTLASS 3.x/4.x 架构全量走读留 W3（5.2）；B200 SM 数 W3/W6（W1 残留不变）。

## 1. 核实通道与网络 fallback（本轮实际使用）

本轮网络整体稳定（与 W1 高度波动不同）：

- **docs.nvidia.com = 本轮主通道，全部首轮 200**（PG 根 + 14 子页 + 官方 PDF、archive/13.0.0 / 11.8.0 / 9.0 / 8.0 legacy 单页、Best Practices 13.5、PTX ISA 9.5、Runtime API 根 + apis、Nsight Compute + ProfilingGuide）。W1「TLS handshake 挂起须重试」本轮未复现。
- **PG 官方 PDF 新通道**：`https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf`（12.78 MB，Release 13.5）一手取得 → pdftotext 全文 1.99 MB，为现行版本主 corpus；archive/13.0.0 单页（3.92 MB HTML → 1.18 MB txt）互证。**13.0 → 13.5 章节号再重组**（见 §4 漂移注），writer 锚以 13.5 子页 + PDF 节号为准。
- **archive 版本路径规则**：`archive/9.0/`、`archive/8.0/`、`archive/11.8.0/`、`archive/13.0.0/` 200；**带 patch 后缀的 `archive/9.2.148/`、`archive/10.2.89/` 404**（archive 索引只列 major.minor[.0] 目录）。GigaThread 逐版 grep：8.0/9.0/11.8/13.0/13.5 均无 → Fermi whitepaper 为术语一手出处。
- **www.nvidia.com 本轮可达**（Fermi whitepaper PDF 876380 B 首轮 200；W1 曾记挂起，波动）。`developer.nvidia.com` 可达（CUDA Graphs blog 237 KB 200）。
- **dam-cdn.nvd.orangelogic.com 稳定 200**（Hopper whitepaper V1.04 7.88 MB 重取，沿 W1 主通道）。
- **nvidia.github.io/cccl/ 根 redirect → unstable/**；旧 `/cccl/libcudacxx/extended_api/memory_model.html` **404** → 现行 `/cccl/unstable/libcudacxx/extended_api/memory_model.html` 200（URL 漂移注）。
- **GitHub API 稳定 200**：cuda-samples（master，pushed 2026-11-04）git trees recursive 565 KB 全树；cutlass（main，pushed 2026-11-23）contents API 目录级 + raw 文件级（swizzle.hpp 16120 B / sm90_pipeline.hpp 48142 B / pipeline.md 8954 B / efficient_gemm.md 20550 B / CHANGELOG.md 173897 B）。
- **WebFetch 未使用**（W1 记录全域名失败「Unable to verify if domain is safe」，本轮沿 curl 通道，未重试 WebFetch）。
- PDF 文本：`/opt/homebrew/bin/pdftotext -layout`；HTML 文本：macOS `textutil -convert txt`。下载缓存 `/tmp/gpu-w2-research/`（pg-current.txt / pg-13.0.0.txt / ptx.txt / bp.txt / ncu.txt / ncu-pg.txt / hopper-wp.txt / fermi-wp.txt / libcudacxx-mm.txt / cuda-graphs-blog.txt / cs-tree.json / cute-swizzle.hpp / sm90-pipeline.hpp 等）。

## 2. 逐源证据（URL × 引句 × 结论）

### 2.1 CUDA Programming Guide 13.5（[C] 一手，PDF + 子页）

URL：根 `https://docs.nvidia.com/cuda/cuda-programming-guide/`（200）；PDF `.../pdf/cuda-programming-guide.pdf`（200，12784076 B，Release 13.5）；legacy `https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/`（200，3921458 B）。

**3.1 线程层级与 SIMT**：

- **SIMT Execution Model（13.5 §3.2.2.1，子页 03-advanced/advanced-kernel-programming.html；13.0 §7.1）逐字**：「A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.」→ divergence 代价账（分支两侧串行）官方语义。
- SIMT vs SIMD 逐字：「SIMT instructions specify the execution and branching behavior of a single thread... SIMD vector organizations expose the SIMD width to the software」。
- **Volta Independent Thread Scheduling 逐字**（13.0 §7.1）：「Prior to NVIDIA Volta, warps used a single program counter shared amongst all 32 threads... Starting with the NVIDIA Volta architecture... the GPU maintains execution state per thread, including a program counter and call stack... A schedule optimizer determines how to group active threads from the same warp together into SIMT units... threads can now diverge and reconverge at sub-warp granularity.」→ 独立 PC 语义 + warp-synchronous 代码须重审（warp-synchronous 定义脚注）。
- **warp 内同址写/atomic 语义逐字**：「If a non-atomic instruction executed by a warp writes to the same location... which thread performs the final write is undefined.」「If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.」→ 3.2 atomic 非确定性（recap inference 1.4 batch-invariance）官方锚。
- **Thread Hierarchy（13.5 §2.3.2，子页 02-basics/writing-cuda-kernels.html；13.0 §5.2）**：tid = x + y·Dx + z·Dx·Dy；block ≤ 1024 threads（「all threads of a block are expected to reside on the same streaming multiprocessor core」→ block 不可跨 SM）；grid 1D/2D/3D；blockIdx/blockDim/gridDim；全局 tid = blockIdx.x*blockDim.x+threadIdx.x 例。
- **Hardware Implementation / block 调度（13.5 §3.2.2；13.0 §7）逐字**：「When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor... As thread blocks terminate, new blocks are launched on the vacated multiprocessors.」+ 13.5 §2.3.7「The details of which thread blocks are scheduled to execute on which SMs cannot be controlled or queried by the application and no ordering guarantees are made by the scheduler」。**GigaThread 术语不出现**（见 §2.5 Fermi whitepaper）。
- **Hardware Multithreading（13.0 §7.2）**：warp execution context on-chip 全生命周期 → 切换零成本；warp scheduler 每 issue 选 ready warp；resident blocks/warps 上限 + register/shared 约束 → 「If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.」；warps/block = ceil(T/32)。
- **Cooperative Groups（13.0 §11，子页 04-special-topics/cooperative-groups.html 200）**：CUDA 9 引入；`grid_group grid = this_grid(); grid.sync();` + 「it is necessary to use, instead of the <<<...>>> execution configuration syntax, the cudaLaunchCooperativeKernel CUDA runtime launch API」+ co-residency 保证（blocks 数须 ≤ 可同时驻留数，occupancy calculator 算）；`tiled_partition<32>(block)` / `thread_block_tile<Size>`；`coalesced_threads()` opportunistic（「returns the set of active threads at that point in time, and makes no guarantee...」）。
- PTX 侧 divergence 见 §2.2。

**3.2 内存模型与地址空间**：

- **Memory Fence Functions（13.5 §5.4.4.3，子页 05-appendices/cpp-language-extensions.html；13.0 §10.5）逐字**：
  - 「void __threadfence_block(); is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)」——prior writes 对 block 内所有线程 observed before subsequent writes + reads ordered。
  - 「void __threadfence(); is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device) and ensures that no writes to all memory made by the calling thread after the call to __threadfence() are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to __threadfence().」
  - 「void __threadfence_system(); is equivalent to cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)... observed by all threads in the device, host threads, and all threads in peer devices」。
  - 「Memory fence functions only affect the ordering of memory operations by a thread; they do not, by themselves, ensure that these memory operations are visible to other threads (like __syncthreads() does for threads within a block)」→ **spec 3.2 命题骨架销项**：fence 只 ordering，跨线程 happens-before 须配 release/acquire 或 atomic flag；__threadfence() = membar.gl = fence.sc.gpu（PTX 侧 §2.2），grid ⊂ device。
  - 「The memory fence functions differ in the scope in which the orderings are enforced but they are independent of the accessed memory space (shared memory, global memory, page-locked host memory, and the memory of a peer device).」
  - 13.5 新增建议逐字：「It is suggested to use cuda::atomic_thread_fence provided by libcu++ whenever possible」。
  - last-block done 模式（atomicInc counter + __threadfence + volatile result）示例仍在 fence 节——**与 Volatile Qualifier 节内部不一致**（见下漂移注）。
- **Atomic Functions（13.0 §10.14）逐字**：「An atomic function performs a read-modify-write atomic operation on one 32-bit, 64-bit, or 128-bit word residing in global or shared memory.」「The atomic functions described in this section have ordering cuda::memory_order_relaxed and are only atomic at a particular scope: _system suffix → thread_scope_system（条件）; 无后缀 → thread_scope_device; _block suffix → thread_scope_block.」→ atomicAdd 默认 device scope relaxed；非确定性见 SIMT 节引句。
- **Synchronization Functions（13.0 §10.6）**：__syncthreads()「waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to __syncthreads() are visible to all threads in the block」+ conditional 须全 block 同值否则 hang；__syncwarp(mask)「guarantees memory ordering among threads participating in the barrier」；__syncthreads_count/and/or。
- **Volatile Qualifier（13.5 §5.3.10.4.3；13.0 §18.5.3.3）逐字**：「The volatile keyword is supported to maintain compatibility with ISO C++; however, few if any of its remaining non-deprecated uses apply to GPUs. Reads and writes to volatile qualified objects are not atomic, and are compiled to one or more .volatile instructions which do NOT guarantee: ordering of memory operations, or that the number of memory operations performed by the HW matches the number of PTX instructions.」「CUDA C++ volatile is NOT suitable for: Inter-Thread Synchronization: Use atomic operations via cuda::atomic_ref, cuda::atomic, or Atomic Functions instead.」+ 13.5「In tile code, the volatile keyword has no effect on the behavior of memory accesses.」→ **3.2 volatile 语义新口径**（旧「volatile bypass L1 跨 block 可见」deprecated）。
- **Variable Memory Space Specifiers（13.0 §10.2）**：无 specifier 自动变量「generally resides in a register. However in some cases the compiler might choose to place it in local memory」；__device__ → global（context lifetime、per-device、grid 全线程 + host runtime 可访问）；__constant__ → constant（host 修改与 grid 访问并发 = undefined）；__shared__ → block shared（block lifetime）。
- **Device Memory Accesses（13.0 §8.3.2）硬件 backing 逐字**：
  - global：「device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned」+「if a 32-byte memory transaction is generated for each thread's 4-byte access, throughput is divided by 8」。
  - Size/Alignment：1/2/4/8/16 B words 单指令 iff 自然对齐；「Any address of a variable residing in global memory or returned by any of the memory allocation routines... is always aligned to at least 256 bytes」；built-in vector types 自动对齐；__align__(8/16)；2D array width 须 warp size 倍数（cudaMallocPitch padding）。
  - local：「The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses... On devices of compute capability 5.x onwards, local memory accesses are always cached in L2」+ spill 条件（动态索引数组 / 大结构 / register 超限）+ ld.local/st.local + lmem 报告。
  - shared：「Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory」+ banks n-way conflict 串行化。
  - constant：「The constant memory space resides in device memory and is cached in the constant cache. A request is then split into as many separate requests as there are different memory addresses in the initial request... serviced at the throughput of the constant cache in case of a cache hit, or at the throughput of device memory otherwise.」
  - texture：「The texture and surface memory spaces reside in device memory and are cached in texture cache... The texture cache is optimized for 2D spatial locality... Addressing calculations are performed outside the kernel by dedicated units」。
- **Memory Hierarchy（13.0 §5.3）**：per-thread local / per-block shared（cluster 内跨 block read/write/atomics）/ global 全线程 / constant+texture read-only；global/constant/texture persistent across kernel launches。
- **L1 非一致 across SM**：PTX cache operators 逐字（§2.2）为官方披露；PG 侧 W1 2.2 已锚 L2 一致。
- **Unified Memory（13.0 §24.x，子页 04-special-topics/unified-memory.html 200）**：cudaMallocManaged 单一 coherent 地址空间 + oversubscription；**page fault 三步逐字**（software-coherent 系统）：「it needs to be ensured that the currently owning processor... cannot access this page anymore... the processor requesting access can access this page... the physical page backing this virtual page must be moved/migrated to the processor requesting access: this can be an expensive operation, and the amount of work is proportional to the page size.」；hardware-coherent（Grace Hopper）「less page-faults... coherent at cache-line granularity instead of page-size granularity」；cudaMemPrefetchAsync stream-ordered 迁移 + cudaMemAdviseSetPreferredLocation/SetAccessedBy；cudaDevAttrDirectManagedMemAccessFromHost。
- **Memory Pool / cudaMallocAsync（13.0 §15.3，子页 04-special-topics/stream-ordered-memory-allocation.html 200）**：「cudaMallocAsync and cudaFreeAsync form the core of the allocator... Both APIs accept stream arguments to define when the allocation will become and stop being available for use. The pointer value returned by cudaMallocAsync is determined synchronously... determines the resident device based on the specified memory pool or the supplied stream.」；default pool non-migratable；BP §10.3 推荐 cudaMallocAsync 替代 cudaMalloc（expensive）。
- **CUDA C++ Memory Model（13.5 §5.8，子页 05-appendices/cuda-cpp-memory-model.html 200）**：thread_scope enum + libcu++ 扩展（详见 §2.7）。

**3.3 Stream / Event / Graph**：

- **Default Stream（13.0 §6.2.8.5.2）逐字**：无 stream 参数/=0 → default stream 顺序执行；`--default-stream per-thread`（或 CUDA_API_PER_THREAD_DEFAULT_STREAM 宏）→ 每 host thread 独立 regular default stream；`--default-stream legacy` → NULL stream（per-device 单条，implicit synchronization）。
- **Implicit Synchronization（13.0 §6.2.8.5.4）逐字**：「Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them, unless the streams are non-blocking streams (created with the cudaStreamNonBlocking flag).」
- **Event / 跨 stream 依赖**：「cudaStreamWaitEvent() takes a stream and an event as parameters... makes all the commands added to the given stream after the call... delay their execution until the given event has completed.」；cudaEventRecord 跨 stream capture 依赖（须同一 capture graph）。
- **Copy engine / overlap（13.0 §6.2.8.5.5 + Overlap of Data Transfer and Kernel Execution）逐字**：「Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the asyncEngineCount device property... greater than zero... If host memory is involved in the copy, it must be page-locked.」；「Some devices... can overlap copies to and from the device... asyncEngineCount... equal to 2」→ 双向 PCIe overlap；intra-device copy 与 kernel 并发（concurrentKernels）。
- **Hardware queue（13.0 §CUDA Environment Variables）逐字**：CUDA_DEVICE_MAX_CONNECTIONS「Sets the number of compute and copy engine concurrent connections (work queues) from the host to each device of compute capability 3.5 and above.」；CUDA_DEVICE_MAX_COPY_CONNECTIONS（CC 8.0+，per async copy engine）。
- **Stream priority / 抢占（13.0 §6.2.8.5.7）逐字**：cudaStreamCreateWithPriority + cudaDeviceGetStreamPriorityRange；「these priorities serve as hints rather than guarantees... Higher-priority tasks do not preempt already running lower-priority tasks. The GPU does not reassess work queues during task execution」；Compute Preemption（Pascal+ CC6.0+）「preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler」+ cudaDevAttrComputePreemptionSupported。
- **CUDA Graphs（13.0 §6.2.8.7，子页 04-special-topics/cuda-graphs.html 200）逐字**：「CUDA Graphs present a new model for work submission in CUDA. A graph is a series of operations, such as kernel launches, connected by dependencies, which is defined separately from its execution... first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams.」；「when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU... an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time.」；三阶段「definition, instantiation, and execution... the overhead of definition and instantiation can be amortized over many executions」；stream capture cudaStreamBeginCapture/EndCapture（capture mode 不入队执行、append 到 capture graph；cross-stream event 依赖须 rejoin origin stream）；graph update（cudaGraphExecUpdate / 逐节点 update，「eliminates the overhead of instantiation」）；conditional nodes（CUDA 12.4+，empty body graph handle）；device graph launch（cudaStreamGraphFireAndForget / cudaStreamGraphTailLaunch）。**~μs 数值官方 PG 只定性** → 数值出处 CUDA Graphs blog（§2.6）。

**3.4 Occupancy 与资源约束**：

- **Kernel Launch and Occupancy（13.5 §2.3.7，子页 02-basics/writing-cuda-kernels.html）逐字**：「The occupancy of a CUDA kernel is the ratio of the number of active warps to the maximum number of active warps supported by the SM.」；device properties 七项（maxBlocksPerMultiProcessor / sharedMemPerMultiprocessor / regsPerMultiprocessor / maxThreadsPerMultiProcessor / sharedMemPerBlock / regsPerBlock / maxThreadsPerBlock）；**CC 10.0 Table 2 例**：32 / 233472 B / 65536 / 2048 / 49152 B / 65536 / 1024；算例：768 threads/block → 2 blocks（2048 限）→ 75%；32 threads/block → 32 blocks（block slots 限）→ 1024/2048 = 50%；100 KB shared/block → 2 blocks（300 KB > 233472 B）；cuda::arch_traits<cuda::arch_id::sm_100> 编译期同值；register spill 权衡「limiting registers allows more thread blocks to be scheduled which in turn increases occupancy and may result in a net increase in performance」→ **spec 3.4 四约束取 min 实例化官方算例**。
- **Occupancy Calculator（13.0 §8.2.3.1）逐字**：「The occupancy calculator API, cudaOccupancyMaxActiveBlocksPerMultiprocessor, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.」「Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.」；cudaOccupancyMaxPotentialBlockSize / VariableSMem（heuristic 最大 occupancy launch config）；cudaOccupancyMaxActiveClusters。
- **Register 约束例（13.0 §8.2.3）逐字**：CC 6.x 64 regs/thread × 512 threads/block → 2 blocks（2×512×64 = 65536 恰满）；65 regs → 1 block（16 warps）→ register 粒度悬崖；maxrregcount / __launch_bounds__ / __maxnreg__ 控制。
- **Launch Bounds（13.5 §5.4.3.2；13.0 §10.38）逐字**：`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)` → .maxntid / .minnctapersm / .maxclusterrank PTX directives；compiler 推导 register 上限 L 保证 minBlocksPerMultiprocessor 驻留；超 L → 降 register「usually at the expense of more local memory usage and/or higher number of instructions」；低于 L 且双参 → 可升 register 至 L 减指令数 hide latency；超 maxThreadsPerBlock launch → fail；「Optimal launch bounds for a given kernel will usually differ across major architecture revisions」（__CUDA_ARCH__ 分档）。
- **occupancy calculator 现行形态**：BP 逐字「NVIDIA provides an occupancy calculator as part of Nsight Compute; refer to https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator」+ NCU §4.13「It replaces the previously provided CUDA Occupancy Calculator spreadsheet.」→ **spec 3.4 Source「occupancy calculator」归 nsight-compute 条目（被引 += 3.4）**，不另立条目。
- **register pressure vs ILP**：BP §10.2.7.1 + §11.1.1 逐字（§2.3）。

### 2.2 PTX ISA 9.5（[C] 一手）

URL：`https://docs.nvidia.com/cuda/parallel-thread-execution/`（200，3965886 B → 1453587 B txt；title「1. Introduction — PTX ISA 9.5 documentation」）。

**3.1 divergence**：§9.5 Divergence of Threads in Control Constructs 逐字：「If threads execute down different control flow paths, the threads are called divergent. If all of the threads act in unison and follow a single control flow path, the threads are called uniform.」「All control constructs are assumed to be divergent points unless the control-flow instruction is marked as uniform, using the .uni suffix. For divergent control flow, the optimizing code generator automatically determines points of re-convergence.」

**3.2 fence / atom / memory model**：

- §9.7.15.4 membar/fence 逐字：「fence{.sem}.scope; .sem = { .sc, .acq_rel, .acquire, .release }; .scope = { .cta, .cluster, .gpu, .sys };」「membar.level; .level = { .cta, .gl, .sys };」「**On sm_70 and higher membar is a synonym for fence.sc, and the membar levels cta, gl and sys are synonymous with the fence scopes cta, gpu and sys respectively.**」→ __threadfence_block/device/system ↔ membar.cta/gl/sys ↔ fence.sc.cta/gpu/sys 三级映射销项（spec 3.2 命题 membar.gl 锚）。fence requires sm_70+；membar.{cta,gl} PTX 1.4 / membar.sys PTX 2.0 sm_20+；fence.proxy.async（PTX 8.0 sm_90+）async proxy ↔ generic proxy 排序（cp.async/TMA 数据可见性须 fence.proxy.async.shared::cta）。
- Memory Consistency Model（§8）：strong/weak/acquire/release/synchronizing operation 定义；**morally strong 三条件逐字**（program order 同线程 or 各 strong 且 scope 含对方线程；same proxy；memory ops 完全 overlap）；data-race = conflicting ops 无 causality order 且非 morally strong；Fence-SC order + SB litmus（fence.sc 保证 sequential consistency，fence.acq_rel 不保证）；「The memory consistency model does not apply to texture (including ld.global.nc) and surface accesses.」
- §9.7.15.5 atom 逐字：`atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b;` `.space = { .global, .shared{::cta, ::cluster} }; .sem = { .relaxed, .acquire, .release, .acq_rel }; .scope = { .cta, .cluster, .gpu, .sys };`（.op and/or/xor/add/inc/dec/min/max；cas/exch；f16x2/bf16x2 add）。red = 无返回值 atomic reduction。
- bar.sync / barrier.cta.sync（§9.7.15.1）：CTA 内 named barrier，固定数量；mbarrier user-defined（见下）。

**3.2/4.1 地址空间与 load**：

- §5.1 State Spaces Table 6/7 逐字：.reg「Registers, fast.」per-thread R/W；.sreg RO per-CTA；.const「Shared, read-only memory.」per-grid；.global「Global memory, shared by all threads.」Context；.local「Local memory, private to each thread.」；.param kernel parameters per-grid / function per-thread；.shared「Addressable memory, defined per CTA, accessible to all threads in the cluster throughout the lifetime of the CTA that defines it.」；.tex「Global texture memory (deprecated).」
- ld 语法（§9.7.10.8）：`ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type d, [a];` `.cop = { .ca, .cg, .cs, .lu, .cv };` `.vec = { .v2, .v4 }`；**ld.global.v4.f32 例逐字**；「Vectors cannot exceed 128-bits in length; for example, .v4 .f64 is not allowed.」；「the access size of ld.v4.b32 is 16 bytes... The address must be naturally aligned to a multiple of the access size. If an address is not properly aligned, the resulting behavior is undefined.」→ 4.1 向量化 128bit/thread + 对齐要求。
- **Cache Operators（§9.7.10.1）逐字（3.2 L1 非一致官方披露）**：.ca「Cache at all levels... The default load instruction cache operation is ld.ca... **Global data is coherent at the L2 level, but multiple L1 caches are not coherent for global data.** If one thread stores to global memory via one L1 cache, and a second thread loads that address via a second L1 cache with ld.ca, the second thread may get stale L1 cache data... The driver must invalidate global L1 cache lines between dependent grids of parallel threads.」；.cg「Cache at global level (cache in L2 and below, not L1)」；.cs evict-first streaming；.lu last use（spill restore）；.cv don't cache fetch again（system memory）；st.wb/.cg/.cs/.wt（write-through to system memory）。「Cache operators on load or store instructions are treated as performance hints only... does not change the memory consistency behavior」。
- ld.global.nc（§9.7.10.9）：「Load a register variable from global state space via non-coherent cache.」.vec={.v2,.v4,.v8}；read-only data cache 路径（texture 路径，不适用 memory model）。

**4.2 ldmatrix**：§9.7.16.5.15 逐字：`ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];` `.shape = {.m8n8, .m16n16}; .num = {.x1, .x2, .x4}; .ss = {.shared{::cta}}; .type = {.b16, .b8};`「Collectively load one or more matrices across all threads in a warp from... .shared state space into destination register r」；m8n8 = 8×8 16-bit；m16n16 = 8/6/4-bit；.trans 转置；for mma instruction（Tensor Core shared→register 路径，1.3/5.2 接口）。

**4.4 cp.async / TMA / mbarrier**：

- §9.7.10.28 Asynchronous copy 总述逐字：「An asynchronous copy operation performs the underlying operation asynchronously in the background... can be a bulk operation... or a non-bulk operation... The amount of data handled by a bulk asynchronous operation must be a multiple of 16 bytes.」；两种完成机制：Async-group / mbarrier-based；「A thread must explicitly wait for the completion... modifying the source memory location or tensor descriptor or reading from the destination memory location before the asynchronous operation completes, exhibits undefined behavior.」
- **cp.async（§9.7.10.28.3.1）逐字**：`cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size} [dst], [src], cp-size{, src-size}{, cache_policy};` `cp.async.cg... 16`；cp-size = {4, 8, 16}；「non-blocking instruction... src specifies a location in the global state space and dst specifies a location in the shared state space」；「The mandatory .async qualifier indicates... control will return to the executing thread before the copy operation is complete... async-group based completion mechanism or the mbarrier based completion mechanism... **No other synchronization mechanism guarantees the completion of the asynchronous copy operations.**」；「**There is no ordering guarantee between two cp.async operations if they are not explicitly synchronized using cp.async.wait_all or cp.async.wait_group or mbarrier instructions.**」；.cg = L2 only（bypass L1）/ .ca = all levels；「cp.async is treated as a weak memory operation performed in the generic proxy」；PTX 7.0，**sm_80+（Ampere）**；src-size < cp-size 零填充；ignore-src 全零。
- **cp.async.commit_group（§9.7.10.28.3.2）逐字**：「creates a new cp.async-group per thread and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group」；空 group trivially complete；同 group 内两 cp.async 写同址 = undefined。
- **cp.async.wait_group / wait_all（§9.7.10.28.3.3）逐字**：「wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups... are complete」；wait_all ≡ commit_group + wait_group 0；writes visible only after wait_all / wait_group / mbarrier.test_wait/try_wait True；「does not provide any ordering and visibility guarantees for any other memory operation apart from cp.async」。
- **cp.async.bulk.tensor（§9.7.10.28.5.3，TMA）逐字**：`cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}... [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo};` `.dim = { .1d, .2d, .3d, .4d, .5d }`；global→shared::cta/::cluster 完成机制 `.mbarrier::complete_tx::bytes`；shared::cta→global `.bulk_group`；`.load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }`（+ scatter4 store）；`.multicast::cluster`（global→多 CTA shared）；`.cta_group::1/::2`；「The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space... The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side.」；PTX 8.0，**sm_90+（Hopper）**；multicast advised sm_90a/sm_100f 等。
- **Tensor-map（§5.5.8）逐字**：「The tensor-map is a **128-byte opaque object** either in .const space or .param (kernel function parameter) space or .global space which describes the tensor properties and the access properties of the tensor data... Tensor-Map can be created using CUDA APIs.」；tensormap.replace（device 侧修改 descriptor）；fence.proxy.tensormap::generic.acquire（kernel start 须显式 acquire tensormap proxy）。
- **TMA swizzle modes（§5.5.8 附近）逐字**：32-byte / 64-byte / 128-byte swizzle modes；128-byte 含 32-byte atomicity sub-mode + 32-byte atomicity with 8-byte flip sub-mode（「flip of adjacent 8-bytes within the 16-byte data at every alternate shared memory line」，仅 shared::cta/cluster.global 方向合法）→ 4.2 swizzle 硬件模式 + 4.4 TMA descriptor 开销账。
- **mbarrier（§9.7.15.16）逐字**：「mbarrier is a barrier created in shared memory that supports: Synchronizing any subset of threads within a CTA; One-way synchronization of threads across CTAs of a cluster...; Waiting for completion of asynchronous memory operations initiated by a thread and making them visible to other threads.」；opaque object .b64 8B-aligned .shared；track「Current primary and conditional phases... Count of pending arrivals... Count of pending asynchronous memory operations (or transactions) tracked by the current phase... also referred to as **tx-count**」；ops init/inval/expect_tx/complete_tx/arrive/arrive_drop/test_wait/try_wait/pending_count；「Unlike bar{.cta}/barrier{.cta} instructions which can access a limited number of barriers per CTA, mbarrier objects are user defined and are only limited by the total shared memory size available.」；「mbarrier operations enable threads to perform useful work after the arrival... and before waiting」→ split arrive/wait 异步语义。
- cp.async.mbarrier.arrive（§9.7.15.16.18）：cp.async 完成触发 mbarrier arrive-on（async proxy）。

### 2.3 CUDA Best Practices Guide 13.5（[C] 一手）

URL：`https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/`（200，433269 B；title「CUDA C++ Best Practices Guide 13.5 documentation」）。

**4.1 coalescing（§10.2.1）逐字**：

- 「For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: **the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.**」→ spec 4.1 官方规则（distinct 32B sector 数）销项。
- 「On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not.」
- 「if the threads of a warp access adjacent 4-byte words (e.g., adjacent float values), four coalesced 32-byte transactions will service that memory access」；「If from any of the four 32-byte segments only a subset of the words are requested... the full segment is fetched anyway. Furthermore, if accesses... had been permuted within or across the four segments, still only four 32-byte transactions」；misaligned sequential → five 32-byte segments（V100 offset 非 8 words 倍数 → 4/5 吞吐）。
- **Strided Accesses（§10.2.1.4）逐字**：strideCopy kernel（xid = global_tid × stride）；「A stride of 2 results in a 50% of load/store efficiency since half the elements in the transaction are not used... **As the stride increases, the effective bandwidth decreases until the point where 32 32-byte segments are loaded for the 32 threads in a warp**」→ spec 4.1 饱和律 1/min(s,8)（fp32）官方锚：s≥8 每 thread 独立 sector → 32 sectors 地板。
- Global Memory Load/Store Efficiency metrics（§9.1.1 附近）：「this comparison of requested memory bandwidth to actual memory bandwidth is reported by the Global Memory Load Efficiency and Global Memory Store Efficiency metrics」→ legacy 指标名；NCU 现行 Sectors/Req（§2.8）。

**4.2 shared banks（§10.2.3.1）逐字**：

- 「shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously... any memory load or store of n addresses that spans n distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is n times as high as the bandwidth of a single bank.」
- 「if multiple addresses of a memory request map to the same memory bank, the accesses are serialized. The hardware splits a memory request that has bank conflicts into as many separate conflict-free requests as necessary, **decreasing the effective bandwidth by a factor equal to the number of separate memory requests**. The one exception here is when multiple threads in a warp address the same shared memory location, resulting in a **broadcast**. In this case, multiple broadcasts from different banks are coalesced into a single **multicast**」→ k-way conflict → k 周期 + broadcast 免冲突销项。
- 「On devices of compute capability 5.x or newer, **each bank has a bandwidth of 32 bits every clock cycle, and successive 32-bit words are assigned to successive banks. The warp size is 32 threads and the number of banks is also 32**」→ 32 banks × 4B、bank = (addr/4) mod 32 销项。
- 「Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses by a warp in shared memory.」
- **padding 证明（§10.2.3.3 transpose 例）逐字**：w×w tile 列写 shared → stride w banks 全 warp 同 bank（w=32）；「The simple remedy is to pad the shared memory array so that it has an extra column」「This padding eliminates the conflicts entirely, because now the stride between threads is w+1 banks (i.e., 33 for current devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a unit stride.」V100 实测 140.2 → 199.4 GB/s → spec 4.2 padding 命题官方锚。

**3.4 occupancy / register pressure（§11.1 / §10.2.7.1）逐字**：

- 「Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps.」
- 「**Higher occupancy does not always equate to higher performance**-there is a point above which additional occupancy does not improve performance. However, low occupancy always interferes with the ability to hide memory latency, resulting in performance degradation.」
- CC 7.0 例：65536 regs / 2048 threads → 32 regs/thread@100%；「register allocations are rounded up to the nearest 256 registers per warp」；128-thread block 37 regs → 75%（12 blocks）vs 320-thread block 37 regs → 63%（4 blocks）。
- 「**NVIDIA provides an occupancy calculator as part of Nsight Compute**; refer to https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator」+「An application can also use the Occupancy API from the CUDA Runtime, e.g. cudaOccupancyMaxActiveBlocksPerMultiprocessor, to dynamically select launch configurations based on runtime parameters.」
- Register Pressure：「Register pressure occurs when there are not enough registers available for a given task... use the -maxrregcount=N compiler command-line option or the launch bounds kernel definition qualifier」。
- **ILP 权衡逐字**：「A lower occupancy kernel will have more registers available per thread than a higher occupancy kernel, which may result in less register spilling to local memory; in particular, **with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy.**」→ spec 3.4 register pressure vs ILP + occupancy 非越高越好。

**4.4 async copy（§10.2.4 附近）逐字**：「The synchronous version... loads an element from global memory to an intermediate register and then stores... to shared memory. The asynchronous version... __pipeline_memcpy_async()... __pipeline_wait_prior(0)... **Using asynchronous copies does not use any intermediate register. Not using intermediate registers can help reduce register pressure and can increase kernel occupancy.**... If individual CUDA threads are copying elements of 16 bytes, the L1 cache can be bypassed.」→ cp.async bypass register 官方口径。

**4.3 tiling**：matrixMul shared tile 例（§10.2.3.2/3）：每 tile 元素 global 读一次 fully coalesced → shared broadcast；V100 144.4 / 195.5 GB/s 实测；「the performance improvement is not due to improved coalescing in either case, but to avoiding redundant transfers from global memory」→ 4.3 复用账官方教学例。

### 2.4 Hopper whitepaper V1.04 TMA 节（[C] 一手，4.4）

URL：`https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf`（200，7880278 B，W1 同版重取）。TMA 节 p.32-34（Fig 18/19/20）：

- 「a new Tensor Memory Accelerator (TMA) that can transfer large blocks of data and multi-dimensional tensors from global memory to shared memory and vice-versa.」
- 「**TMA operations are launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing**... Large blocks of data (up to the shared memory capacity)... support for different tensor layouts (**1D-5D tensors**), different memory access modes, reductions, and other features.」
- 「The TMA operation is asynchronous and leverages the shared memory-based asynchronous barriers introduced in A100. Additionally, **the TMA programming model is single-threaded, where a single thread in a warp is elected to issue an asynchronous TMA operation (cuda::memcpy_async)** to copy a tensor, and subsequently multiple threads can wait on a cuda::barrier for completion... the H100 SM adds hardware to accelerate these asynchronous barrier wait operations.」
- 「On A100... asynchronous memory copies were executed using a special **LoadGlobalStoreShared** instruction, so the threads were responsible for generating all addresses and looping across the whole copy region. On Hopper, TMA takes care of everything. A single thread creates a copy descriptor before launching the TMA, and from then on **address generation and data movement are handled in hardware**... computing stride, offset, and boundary calculations」→ 4.4 TMA descriptor 开销 vs 带宽节省账 + cp.async（LDGSTS）vs TMA 适用场景官方锚。
- **Asynchronous Transaction Barrier 节逐字**：「Asynchronous Barriers were originally introduced in the Ampere GPU architecture... split the synchronization process into two steps... 'Arrive'... non-blocking... 'Wait' which blocks them until every thread has signaled 'Arrive'.」「New for Hopper is the ability for 'Waiting' threads to **sleep** until all other threads arrive. On previous chips, Waiting threads would **spin** on the barrier object in shared memory.」「Hopper adds a new form of barrier called an Asynchronous Transaction Barrier... instead of counting just thread arrivals, **it also counts transactions**... **The transaction count is essentially a byte count.** The asynchronous transaction barrier will block threads at the Wait command until all the producer threads have performed an Arrive, and **the sum of all the transaction counts reaches an expected value**.」→ mbarrier expect_tx/complete_tx 硬件 backing 逐字销项。

### 2.5 Fermi whitepaper GigaThread（[C] 一手，3.1 新增条目）

URL：`https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf`（200，876380 B；www.nvidia.com 本轮可达）。另一 URL `.../content/dam/develop/external/us/en/documents/fermi_white_paper.pdf` 404（勿用）。

- 「The **GigaThread global scheduler distributes thread blocks to SM thread schedulers**.」（An Overview of the Fermi Architecture）
- 「One of the most important technologies of the Fermi architecture is its **two-level, distributed thread scheduler**. At the chip level, a global work distribution engine schedules thread blocks to various SMs, while at the SM level, each warp scheduler distributes warps of 32 threads to its execution units. The first generation GigaThread engine introduced in G80 managed up to 12,288 threads in realtime.」（GigaThread Thread Scheduler 节 p.18）
- **术语时代核实**：GigaThread 在 PG archive 8.0 / 9.0 / 11.8.0 / 13.0.0 / 13.5 PDF 全文 grep 均无 → spec 3.1「GigaThread engine」为 legacy 术语，writer 须注「Fermi 时代术语，现行 PG 无名描述 block 分发」；block 不可跨 SM 由 PG Thread Hierarchy「all threads of a block... reside on the same streaming multiprocessor」锚。

### 2.6 CUDA Graphs blog（[C] 开放博客，3.3 新增条目）

URL：`https://developer.nvidia.com/blog/cuda-graphs/`（200，237053 B；title「Getting Started with CUDA Graphs」；article:published_time 2019-09-05）。

- 「there are overheads associated with the submission of each operation to the GPU – **also at the microsecond scale**」。
- **V100 / CUDA 10.1 实测逐字**：kernel「tuned such that kernel takes a few microseconds... measure the time taken to be **2.9μs**」（N=500000，512 threads/block）；sync launch（每 kernel 后 cudaStreamSynchronize）「**9.6μs per kernel** (including overheads): much higher that the kernel execution time of 2.9μs」；async stream「**3.8μs** (vs 2.9μs kernel execution time)... allows a kernel to be launched before the previous kernel completes, allowing launch overhead to be hidden behind kernel execution」；graph「**3.4μs** (vs 2.9μs kernel execution time)... the time to create and instantiate the graph is relatively large at around **400μs**, but this is only performed a single time, so this is only contributes around **0.02μs** to our per-kernel cost... the first graph launch is around **33% slower** that all subsequent launches」。
- 「cudaGraphInstantiate... creates and pre-initialises all the kernel work descriptors so that they can be launched repeatedly as rapidly as possible」。
- → spec 3.3「launch 开销摊销 ~μs × kernel 数」官方数值出处；**时代注**：V100/CUDA 10.1 绝对值，现行硬件会漂移，量级口径可用。PG 只定性（§2.1），数值挂本条目。

### 2.7 libcu++ memory model（[C] 官方规格，3.2 新增条目）

URL：`https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html`（200，93352 B）。**旧 URL `/cccl/libcudacxx/extended_api/memory_model.html` 404**（cccl/ 根 302 → unstable/）。

- 「CUDA C++ is different: the cost to synchronize threads grows as threads are further apart... CUDA C++ extends the standard C++ memory model and concurrency facilities in the cuda:: namespace with **thread scopes**, retaining the syntax and semantics of standard C++ by default.」
- **enum 逐字**：`enum thread_scope { thread_scope_system, thread_scope_device, thread_scope_cluster, thread_scope_block, thread_scope_thread };`
- scope 定义逐字：system = 全系统 threads；device =「each other GPU thread in the same CUDA device **and within the same memory synchronization domain**」；cluster = same thread block cluster；block = same thread block；thread = self。「On architectures before SM90, thread_scope_cluster operations are widened to thread_scope_device.」
- release/acquire flag message-passing 例（flag.store(1, memory_order_release) / while(flag.load(memory_order_acquire) != 1)）+ 反例 data race UB。
- → 3.2 memory_order / scoped synchronization 官方口径；PG §5.4.4.3 / §5.8 交叉引用同库。

### 2.8 Nsight Compute（[C]，3.4 occupancy calculator + 4.1 metrics）

URL：`https://docs.nvidia.com/nsight-compute/NsightCompute/index.html`（200，347513 B）；`https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html`（200，383205 B）。

- **Occupancy Calculator（NsightCompute §4.13）逐字**：「NVIDIA Nsight Compute provides an Occupancy Calculator that allows you to compute the multiprocessor occupancy of a GPU for a given CUDA kernel. **It replaces the previously provided CUDA Occupancy Calculator spreadsheet.**」；.ncu-occ 文件 + Python interface（§4 Occupancy Calculator Python Interface）；Tables（occupancy + active threads/warps/blocks per SM + max active blocks on GPU）/ Utilization（stacked bar：allocated / unallocated due to other limiter / unused < block minimum）/ Graphs（occupancy vs block size）/ GPU Data。→ **spec 3.4「occupancy calculator 读法」销项**：归 nsight-compute 条目（被引 += 3.4），Utilization 三态即四约束取 min 的可视化。
- **ProfilingGuide Memory Workload Analysis 逐字**：
  - Requests：「Each instruction generates exactly one request for LSU traffic (global, local, …). For texture (TEX) traffic, more than one request may be generated.」
  - Wavefronts：「Number of wavefronts required to service the requested memory operation. Wavefronts are serialized and processed on different cycles.」+「a notion of processing one wavefront per cycle in L1TEX... the number of sectors per request is a property of the access pattern」。
  - **Sectors/Req**：「The average ratio of sectors to requests for the L1 cache... **For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16.**... Higher numbers can imply uncoalesced memory accesses and will result in increased memory traffic.」+ 例「32 sectors per request... each thread needs to access a different sector... ideally... 4, as every 8 consecutive threads access the same sector」。
  - Bytes：「identical to the number of sectors multiplied by 32 byte, since the minimum access size in L1 is one sector.」
  - 「Atomics and reductions are passed through to the L2 cache.」
  - L1/L2 Sectors/Wavefronts/Requests achieved-ideal-excessive 派生指标（NsightCompute §deprecated counter 节附近）：「Ideal metrics indicate the number that would needed, given each not predicated-off thread performed the operation of given width. Excessive metrics indicate the required surplus over the ideal case.」
  - → 4.1 coalescing 效率现行指标（替代 legacy gld_efficiency）；4.2 L1 Wavefronts Shared Excessive（shared conflict 指标）。

### 2.9 cuda-samples（教学参照，cpp/ 重组路径级）

GitHub API：repo 200（NVIDIA/cuda-samples，default_branch **master**，pushed 2026-11-04）；git trees/master?recursive=1 200（565487 B，1791 paths）；README raw 200（「This version supports CUDA Toolkit 13.5」）。

- **重组漂移**：顶层无 Samples/ 目录；现行 `cpp/0_Introduction … cpp/10_Safety` + `python/` + `Common/`。旧 `Samples/N_Name/` 路径全部失效 → writer 用 cpp/ 路径。
- 路径级核实（tree 逐字）：
  - 3.1：cpp/0_Introduction/simpleCooperativeGroups；cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG。
  - 3.2：cpp/2_Concepts_and_Techniques/threadFenceReduction（fence + atomic last-block 模式教学例）。
  - 3.3：cpp/3_CUDA_Features/{simpleCudaGraphs, graphConditionalNodes, graphMemoryNodes, graphMemoryFootprint, jacobiCudaGraphs}；cpp/6_Performance/cudaGraphsPerfScaling（README 逐字：cudaStreamBeginCapture / cudaGraphInstantiate / cudaGraphLaunch / cudaGraphUpload）；cpp/0_Introduction/asyncAPI。
  - 4.1：cpp/6_Performance/alignedTypes。
  - 4.3：cpp/0_Introduction/{matrixMul, matrixMulDrv, matrixMulDynlinkJIT}。
  - 4.4：cpp/3_CUDA_Features/globalToShmemAsyncCopy（README 逐字：「implements matrix multiplication which uses asynchronous copy of data from global to shared memory when on compute capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.」）。
  - 5.1（W3 预核实）：cpp/2_Concepts_and_Techniques/{reduction, reductionMultiBlockCG, scan, shfl_scan}。

### 2.10 CUTLASS 4.9.0（教学参照，4.2/4.3/4.4 路径级）

GitHub API：repo 200（NVIDIA/cutlass，default_branch **main**，pushed 2026-11-23，description「CUDA Templates and Python DSLs for High-Performance Linear Algebra」）；CHANGELOG raw 200：**「## [4.9.0](https://github.com/NVIDIA/cutlass/tree/main) (2026-11-13)」**（CUTLASS 4.x，CuTe DSL + Rubin 支持）。

- **4.2 swizzle ⟨verify⟩ 销项**：contents include/cute/ 200 → `swizzle.hpp` / `swizzle_layout.hpp` / `pointer_swizzle.hpp` 存在；swizzle.hpp raw 200（16120 B）逐字：「0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where **AA = ZZ xor YY**」+ `struct Swizzle`（template <int B, int M, int S>）+ `return offset ^ shiftr(offset & yyy_msk{}, msk_sft{}); // ZZZ ^= YYY` → spec 4.2 swizzle 异或映射 + 双射性代码锚。
- **4.4 pipeline ⟨verify⟩ 销项**：contents include/cutlass/pipeline/ 200 → `pipeline.hpp` / `sm90_pipeline.hpp` / `sm100_pipeline.hpp` / `sm107_pipeline.hpp`；sm90_pipeline.hpp raw 200（48142 B）逐字：`struct PipelineState`（L171，index/phase）；`class PipelineTmaAsync`（L289，template <int Stages_, class FullBarrier_ = cutlass::arch::ClusterTransactionBarrier>）；`FullBarrier full_barrier_[Stages]; EmptyBarrier empty_barrier_[Stages];`（L299-300）；producer_acquire（L443）/ producer_commit（L453）/ producer_expect_transaction（L480，mbarrier expect_tx）/ consumer_wait（L498）/ consumer_release；`class PipelineTmaStore`（L673）/ `class PipelineTransactionAsync`（L784）/ `class PipelineAsync`（L1033）。
- **版本漂移**：`PipelineTmaUmmaAsync`（旧 3.x FA3 时代 warp-specialized UMMA pipeline 类）**不在当前 main sm90_pipeline.hpp**（全文 grep Umma/UMMA 零命中）→ writer 4.4/5.2 引 pipeline 类名须用当前清单；FA3 走读（6.3 W3）若引旧类名须注版本。
- **4.3 docs 拆分**：contents media/docs/cpp/ 200 → `efficient_gemm.md` / `gemm_api_3x.md` / `pipeline.md` / `cutlass_3x_design.md` / `cute/` 等；**`gemm_tutorial.md` 不存在**（全目录清单核实）→ spec 4.3「CUTLASS GEMM tutorial」现行锚 = efficient_gemm.md。efficient_gemm.md raw 200（20550 B）逐字：GEMM 层级嵌套循环伪代码（cta_n/cta_m threadblock-level concurrency → warp_n/warp_m/warp_k warp-level parallelism → mma_k/mma_n/mma_m instruction-level parallelism）。pipeline.md raw 200（8954 B）：「Synchronization primitives」+ Hopper clusters/async barriers + CUTLASS abstractions。**新条目 cutlass-docs（P4，4.3 首引）**；cutlass repo 条目被引 4.3 移彼（代码路径 4.2/4.4/5.2 维持）。

## 3. ⟨verify⟩ W2 销项 / 残留

| 项 | 状态 | 证据 / 留波 |
| --- | --- | --- |
| CUTLASS swizzle 路径（spec 4.2 Sources include/cute/swizzle*.hpp ⟨verify⟩） | **销** | contents API 200 + swizzle.hpp raw Swizzle<B,M,S> XOR 逐字（§2.10） |
| CUTLASS pipeline 路径（spec 4.4 Sources include/cutlass/pipeline/ ⟨verify⟩） | **销** | contents API 200 + sm90_pipeline.hpp raw PipelineTmaAsync/PipelineState 逐字（§2.10） |
| CUTLASS 当前版本（W0 留 W3） | **部分销** | 4.9.0（2026-11-13）版本锚本轮取得；3.x/4.x 架构全量走读留 W3 5.2 |
| occupancy calculator 现行形态（spec 3.4 Source） | **销** | NCU §4.13 replaces spreadsheet 逐字 + BP 交叉引用；归 nsight-compute 条目 += 3.4 |
| GigaThread 出处（spec 3.1 内容） | **销** | Fermi whitepaper 逐字（§2.5）；PG 8.0-13.5 无此术语 → legacy term 注 |
| graph launch ~μs 出处（spec 3.3 内容） | **销** | CUDA Graphs blog V100/CUDA 10.1 实测 2.9/9.6/3.8/3.4 μs + 400 μs instantiate（§2.6）；时代注 |
| PG URL（spec 3.1 ⟨verify URL⟩，W1 已销） | **再确认** | 根 200 + 13.5 子页地图 14/14 200 + 官方 PDF 一手（§2.1） |
| cuda-samples 教学例路径（spec 3.1/3.3/4.1/4.3 Sources 200） | **销** | GitHub API master 树 cpp/ 重组路径级（§2.9） |
| SIMT/warp divergence 语义 | **销** | PG 13.5 §3.2.2.1 + 13.0 §7.1 逐字 + PTX §9.5 .uni/reconvergence |
| fence 三级 scope + memory_order + atomics | **销** | PG §5.4.4.3/§10.14 + PTX fence.sc/membar 同义逐字 + atom .sem/.scope + libcu++ thread_scope |
| 地址空间硬件 backing | **销** | PG Device Memory Accesses（local→device mem cached L2 / constant→constant cache / texture→texture cache / shared on-chip）+ PTX State Spaces Table 6/7 |
| unified memory page fault / 迁移 | **销** | PG §24.x 三步逐字 + Grace Hopper 硬件一致对照 |
| stream/event/graph 机制（copy engine / hardware queue / capture/replay） | **销** | PG asyncEngineCount / CUDA_DEVICE_MAX_CONNECTIONS work queues / capture 三阶段 + blog 数值 |
| occupancy API / launch_bounds / register 约束 | **销** | PG §2.3.7 CC10.0 算例 + §8.2.3.1 API + §5.4.3.2 launch_bounds + BP register pressure ILP |
| coalescing distinct sector / stride 饱和 / ld.global.v4 | **销** | BP CC6.0+ 32B transaction 规则逐字 + stride 32 segments 饱和 + PTX ld.v4 ≤128bit 对齐 |
| shared 32 banks×4B / conflict / broadcast / padding / ldmatrix | **销** | BP §10.2.3.1 逐字 + padding 33≡1 证明 + PTX ldmatrix.m8n8 |
| cp.async commit/wait group / TMA descriptor / mbarrier | **销** | PTX §9.7.10.28 逐字（sm_80+/sm_90+、tensor-map 128B、tx-count）+ Hopper wp TMA/Async Transaction Barrier 节 |
| B200 SM 数 | **残留**（W1 不变） | 官方未披露 → W3/W6 |
| JEDEC / PCI-SIG 精确 URL | **残留**（W1 不变） | 403 → W4 |

## 4. 版本漂移注

- **PG 13.0 → 13.5 章节再重组**（W1 已注 13.x 拆子页，本轮 13.5 再变）：SIMT Architecture（13.0 §7.1）→ **SIMT Execution Model（13.5 §3.2.2.1）**，子页 03-advanced/advanced-kernel-programming.html；Thread Hierarchy（13.0 §5.2）→ 13.5 §2.3.2（02-basics/writing-cuda-kernels.html）；新增 §2.3.7 Kernel Launch and Occupancy（CC 10.0 算例 + cuda::arch_traits）；Memory Fence Functions → 13.5 §5.4.4.3（05-appendices/cpp-language-extensions.html）；CUDA C++ Memory Model → 13.5 §5.8（05-appendices/cuda-cpp-memory-model.html）；Launch Bounds → 13.5 §5.4.3.2；TMA 用法 → 13.5 §4.13.2（Table 25 完成机制矩阵）；LDGSTS → §4.13.1。**writer 锚以 13.5 子页 URL + PDF 节号为准；archive/13.0.0 单页为互证 corpus（节号不同）**。
- **volatile 语义漂移（重要）**：PG 13.x Volatile Qualifier 节明确 volatile 不适合 inter-thread synchronization（用 cuda::atomic_ref/atomic），「In tile code, the volatile keyword has no effect」；但 Memory Fence Functions 节 last-block 示例仍用 volatile result（内部不一致）。**3.2 writer 以 Volatile Qualifier 节新口径为准**；旧书「volatile bypass L1 跨 block 可见」模式 deprecated。fence 节建议 libcu++ cuda::atomic_thread_fence。
- **GigaThread 术语时代**：Fermi whitepaper（2009）/ G80 时代术语；PG 8.0-13.5 均无 → 3.1 须注 legacy term，现行机制描述用 PG「blocks enumerated and distributed」。
- **cuda-samples 重组**：Samples/N_Name → cpp/N_Name（CUDA 13.5 版）；旧书/旧章 Samples/ 路径全部失效。
- **CUTLASS 4.9.0**：gemm_tutorial.md → efficient_gemm.md；PipelineTmaUmmaAsync 不在当前 main sm90_pipeline.hpp（旧 3.x 类）；repo description 改「CUDA Templates and Python DSLs」（CuTe DSL 时代）。
- **occupancy calculator**：独立 Excel spreadsheet 已废 → Nsight Compute §4.13（BP/NCU 逐字）；3.4 不引旧 spreadsheet URL。
- **libcu++ docs URL**：/cccl/libcudacxx/extended_api/ 404 → /cccl/unstable/libcudacxx/extended_api/（cccl 根 redirect unstable/）。
- **Nsight Compute 指标更名**：BP 仍提 Global Memory Load/Store Efficiency（legacy gld_efficiency/gst_efficiency）；NCU 现行 Memory Workload Analysis Sectors/Req（最优 4/8/16）+ achieved-ideal-excessive 派生指标。4.1 两口径并注。
- **CUDA Graphs blog 数值时代**：V100/CUDA 10.1（2019）实测；现行硬件绝对值漂移，~μs 量级口径可用；PG 只定性。
- **archive 路径规则**：带 patch 后缀目录（9.2.148/10.2.89）404；用 major.minor[.0]（9.0/10.2/11.8.0/13.0.0）。
- **PTX 9.5 / BP 13.5 / PG 13.5 版本同步**（CUDA 13.5 toolkit）；PTX fence.alias.sync_restrict::read.release.sys 为 9.5 新增（sm_90+）。

## 5. 种入纪律与被引回写

- 跨 Part 共享锚点收录首引 Part、被引跨组累加；新增 fermi-whitepaper / cuda-graphs-blog / libcudacxx-memory-model 首引 3.1/3.3/3.2 归 P3；cutlass-docs 首引 4.3 归 P4（docs 自 repo 拆分，沿 inference deepinfer-doc/deepinfer-repo 先例）。
- 等级单值：官方规格/官方文档 [C]「官方规格披露·一手权威」；NVIDIA developer blog [C]「开放技术博客、非闭源」；官方 repo/docs 教学参照（官方实现）；无混合档。
- 禁竖线：多 URL 用 ` ; `；单元格无 `|` 含 escaped。68 行污染检查全过。
- **被引回写（spec §5 Sources）**：nsight-compute += 3.4（occupancy calculator）；cutlass 被引 4.3 → cutlass-docs（docs 拆分，代码 4.2/4.4/5.2 维持）；cuda-programming-guide / ptx-isa / cuda-best-practices / cuda-samples / hopper-whitepaper / hopper-tuning-guide 被引已含 3.1-3.4/4.1-4.4 对应章不变；新增条目被引 = 首引章。
- **writer 白名单提示**：3.1 GigaThread 引 fermi-whitepaper + legacy term 注；3.2 memory_order/thread_scope 引 libcudacxx-memory-model + volatile 新口径；3.3 ~μs 数值引 cuda-graphs-blog + 时代注；3.4 occupancy calculator 引 nsight-compute（非独立条目）；4.3 GEMM tutorial 引 cutlass-docs（efficient_gemm.md，非 gemm_tutorial.md）；4.4 pipeline 类名用当前 main 清单（PipelineTmaAsync 等，非 PipelineTmaUmmaAsync）。
