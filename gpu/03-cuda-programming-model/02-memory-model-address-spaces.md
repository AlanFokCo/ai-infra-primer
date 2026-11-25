# 3.2 内存模型与地址空间（Memory Model and Address Spaces）

> SM 0 写了一个 global 地址，SM 1 一定读得到新值吗？——L1 跨 SM 非一致，答案取决于你用了什么同步语义。本章交付六个地址空间的语义与硬件 backing、CUDA 内存模型（thread_scope 五级 + memory_order + fence 三级映射）、命题 1（fence 只管 ordering、可见性须配对）、volatile 新口径、同步原语（barrier + 可见性双语义）、atomic 非确定性（命题 2，recap inference 1.4 batch-invariance）、unified memory page fault 迁移账与 cudaMallocAsync 内存池 | 前置：2.2（L1 非一致 / L2 一致点、延迟表实测口径）、2.3（local/spill 双代价、常量/纹理路径）、3.1（block/warp 语义、调度无顺序保证、divergence 代价账）、1.2（shared/register 库存）；下游 3.3（stream-ordered 语义）、3.4（occupancy）、4.1/4.2（coalescing/bank conflict）、4.4（cp.async/TMA 与 mbarrier）、5.1（确定性归约） | 锚点：CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、libcu++ memory model（[NVIDIA, libcu++ memory model](../references.md)，[C]）、Nsight Compute（[NVIDIA, Nsight Compute](../references.md)，[C]）、cuda-samples threadFenceReduction（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 延迟实测口径）、PCI-SIG PCIe（[PCI-SIG, PCIe](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 地址空间：六个逻辑空间，四种物理 backing

`__shared__ float tile[256];`、`float x;`、`__constant__ float c[1024];`——三行声明，三种命运：片上 SRAM、register（装不下就 spill 到 HBM）、HBM 加常量缓存。地址空间的逻辑作用域不等于物理位置，local memory 是最误导的名字：线程私有，物理上却在 HBM。本章回答三个问题：六个地址空间各是什么语义、落在什么硬件上？跨线程读写要什么语义才保证可见（fence/atomic/scope）？unified memory 与内存池把分配和迁移的开销藏在哪？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（沿 3.1）；running example 沿用 3.1 的 elementwise grid（65536 blocks × 256 threads、 $n_{\mathrm{vec}} = 16777216$ 个 fp32 元素），本章新增 last-block 归约与 histogram 两个机制片段；histogram 账换 grid-stride 1056 blocks 配置，原因见 §5（两阶段 atomic 的节省因子 = 每线程元素数，1 元素/线程配置无节省）。延迟口径 recap 2.2 实测 [B]（shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8 cycles，H800 微基准，非 H100 官方规格），cycle ↔ 时间换算取 1.83 GHz。

符号声明： $n_{\mathrm{blk}}$ = grid block 数、 $n_{\mathrm{vec}}$ = 向量元素数（recap 3.1）； $n_{\mathrm{th}}$ = 参与同址 atomic 的线程数（§5 命题 2 局部；裸字母 n 专归 rank 数，不占用）； $n_{\mathrm{bins}}$ = histogram bin 数（本章局部）； $\rho$ = global atomic 节省因子（§5）； $P_{\mathrm{page}}$ = 迁移页大小、 $M$ = 迁移总字节（沿全书 $M$ = 传输字节语义）、 $\mathrm{BW}$ = 互连带宽（本章取 PCIe Gen5 x16 理论单向 64 GB/s，2.4 §4 表口径）； $t_{\mathrm{page}}$、 $t_{\mathrm{mig}}$ = 单页迁移带宽项与总迁移时间下界（§6）。bytes 不占字母，B/KB/MB/GB 是单位符号豁免。

所谓地址空间（address space），是指 CUDA 按作用域与生命周期划分的逻辑存储类别，共六个：register、local、shared、global、constant、texture/surface（[NVIDIA, CUDA Programming Guide](../references.md) Variable Memory Space Specifiers 与 Device Memory Accesses 节，[C]）。语义与硬件 backing 总表：

| 地址空间 | 作用域 / 生命周期 | 硬件 backing | 延迟口径（实测 [B]） |
| --- | --- | --- | --- |
| register | 每线程 | 片上 register file，256 KB/SM（1.2） | 管线内 |
| local | 每线程（逻辑私有） | HBM（device memory），CC 5.x+ 恒经 L2 缓存 | 与 global 同（478.8 cycles 全 miss） |
| shared | 每 block（Hopper cluster 内可跨 block 互访） | 片上 SRAM，与 L1 统一 256 KB（1.2 §4） | 29.0 cycles |
| global | 全 grid 线程 + host，context 生命周期 | HBM，经 L1/L2 | L1 40.7 / L2 263.0 / HBM 478.8 |
| constant | 每 grid 只读，64 KB 空间 | HBM + constant cache | 命中按常量缓存吞吐 |
| texture / surface | 只读路径 | HBM + texture cache | 2D 空间局部性优化 |

backing 的官方逐字（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）：local——「The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses」，且「On devices of compute capability 5.x onwards, local memory accesses are always cached in L2」；shared——「Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory」；constant——「The constant memory space resides in device memory and is cached in the constant cache. A request is then split into as many separate requests as there are different memory addresses in the initial request」，命中按常量缓存吞吐、miss 按 device memory 吞吐；texture——「The texture and surface memory spaces reside in device memory and are cached in texture cache」，「The texture cache is optimized for 2D spatial locality」，「Addressing calculations are performed outside the kernel by dedicated units」。

**引文版本归属注**：本章 PG 逐字引文除显式注 13.5 者外均为 **archive/13.0.0 逐字**——13.5 PDF 与子页对相应节（backing/fence/syncthreads/atomic relaxed/volatile/cudaMallocAsync/UM page fault）已改写措辞，语义两版全同、互证（如 fence「independent of the accessed memory space」13.5 作「These functions differ in the thread scope ... including shared memory ...」，volatile「Reads and writes to volatile qualified objects」13.5 作「Reading and writing to volatile-qualified objects」）。§5 SIMT 同址写/atomic 顺序两句与 §2 libcu++ fence 建议句为 13.5 逐字，随文注。节号给 13.5 子页现行编号、引文措辞以 archive 为准。

PTX 侧对应 state space（[NVIDIA, PTX ISA](../references.md) State Spaces 表，[C]）：`.reg`（每线程读写）、`.local`（每线程私有）、`.shared`（per-CTA，cluster 内可访问）、`.global`（context 级全线程）、`.const`（per-grid 只读）、`.tex`（已 deprecated）、`.param`（kernel 参数）。逻辑空间与物理 backing 的映射图：

```
逻辑空间（编程模型）              物理 backing（H100 SXM）
register ───────────────────►  片上 RF 256 KB/SM（1.2）
local ──────────┐
global ─────────┼───────────►  HBM 80 GB（经 L1/L2）
constant ───────┤              + constant cache（64 KB 空间）
texture/surface ┘              + texture cache（只读路径）
shared ─────────────────────►  片上 SRAM（与 L1 统一 256 KB，carveout）
```

全数字落点：HBM 对 shared 的延迟比 $478.8/29.0 = 16.5$ 倍——把 block 协作数据从 global 搬进 shared，延迟账缩 16.5 倍（带宽账与 tiling 强度归 4.3）；constant 空间 64 KB = $65536/4 = 16384$ 个 fp32，warp 内全线程读同一地址时一次广播服务（2.3 §4 recap），读不同地址则按地址数拆分请求（上文 PG 逐字）；local 每线程上限 512 KB（2.3 recap）。放置决策 recap 2.3 §7（不重推）：线程私有热数据 → register，block 共享 → shared，warp-uniform 只读 → constant，流式只读 → L1/L2 主路径或只读旁路，其余 → HBM。

local 与 register spill 的关系 recap 2.3 §6（不重推）：编译器装不下的值 spill 到 local，物理走 HBM，带宽-延迟双代价（2.3 命题 1）；触发条件三类——动态索引数组、过大结构、register 超预算。所以「局部变量数组开个动态下标」不是免费的：它可能把 register 数据整批赶到 16.5 倍延迟的层。

> **提点**：逻辑作用域与物理 backing 的分离，是存储侧的「机制与策略分离」——编程模型只承诺作用域语义（机制），数据落在哪一层由声明与编译器决定（策略）。删掉这条类比，backing 表与放置决策链依然完整。

只读旁路的边界须钉一句：`ld.global.nc`（`__ldg` / `const __restrict__` 路由此路径）走非一致缓存，PTX 逐字「The memory consistency model does not apply to texture (including ld.global.nc) and surface accesses.」（[NVIDIA, PTX ISA](../references.md)，[C]）——内存模型不管这条路径，所以只读旁路上的数据**不得与写并发**，读写重叠须程序员自己保证（2.2 §7 已立此边界，此处交付其模型侧根据）。

## 2. CUDA 内存模型：scope、memory_order 与 fence

为什么 GPU 需要一套与 CPU 不同的内存模型？两个机制原因：其一，L1 跨 SM 非一致（§3，2.2 §6 recap），「写了就可见」在 GPU 上不成立；其二，同步成本随线程距离增长——libcu++ 逐字「the cost to synchronize threads grows as threads are further apart」（[NVIDIA, libcu++ memory model](../references.md)，[C]），所以 CUDA 把 **scope（作用域）**提升为一等公民：每个同步操作显式声明它覆盖多大范围的线程。

thread_scope 五级（libcu++ enum，成员与顺序逐字，注释为本书所加）：

```cpp
enum thread_scope {
    thread_scope_system,    // 全系统：本 GPU + host + peer GPU
    thread_scope_device,    // 本 GPU 全部线程（同一 memory synchronization domain）
    thread_scope_cluster,   // 同一 thread block cluster（Hopper）
    thread_scope_block,     // 同一 thread block
    thread_scope_thread     // 仅自身
};
```

device scope 的官方定义逐字「each other GPU thread in the same CUDA device and within the same memory synchronization domain」；「On architectures before SM90, thread_scope_cluster operations are widened to thread_scope_device.」（[NVIDIA, libcu++ memory model](../references.md)，[C]）——Hopper 之前 cluster scope 自动放宽为 device scope。scope 嵌套：thread ⊂ block ⊂ cluster ⊂ device ⊂ system，选最小够用的 scope 是性能原则（§5 给延迟账）。

memory_order 沿用 C++ 标准子集：relaxed（只保原子性、不保顺序）、acquire / release（配对建立跨线程顺序）、acq_rel、seq_cst（最强，全序）。fence 三级与 PTX 的映射（[NVIDIA, CUDA Programming Guide](../references.md) Memory Fence Functions 节 + [NVIDIA, PTX ISA](../references.md) fence/membar 节，[C]）：

| CUDA C++ | libcu++ 等价（均 seq_cst） | PTX（sm_70+） | scope |
| --- | --- | --- | --- |
| `__threadfence_block()` | `thread_scope_block` | `membar.cta` ≡ `fence.sc.cta` | block |
| `__threadfence()` | `thread_scope_device` | `membar.gl` ≡ `fence.sc.gpu` | device |
| `__threadfence_system()` | `thread_scope_system` | `membar.sys` ≡ `fence.sc.sys` | system |

等价性的官方逐字：「void `__threadfence_block()`; is equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`」；device 级「void `__threadfence()`; is equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)` and ensures that no writes to all memory made by the calling thread after the call to `__threadfence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `__threadfence()`.」；system 级覆盖「all threads in the device, host threads, and all threads in peer devices」。PTX 侧逐字：「On sm_70 and higher membar is a synonym for fence.sc, and the membar levels cta, gl and sys are synonymous with the fence scopes cta, gpu and sys respectively.」——三级映射在两套官方文档里逐字对上。fence 与访存空间无关：「The memory fence functions differ in the scope in which the orderings are enforced but they are independent of the accessed memory space (shared memory, global memory, page-locked host memory, and the memory of a peer device).」PG 13.5 并建议「It is suggested to use cuda::atomic_thread_fence provided by libcu++ whenever possible」。

scope 的量化含义（本章口径）：device scope 覆盖 132 SM × 2048 threads = 270336 个线程，block scope 只覆盖 256 个——差 1056 倍。「同步成本随距离增长」不是修辞：device 级 atomic 落到全芯片一致点 L2（§5），block 级 shared atomic 留在片上，延迟差 9 倍（§5 数值）。

fence 保证的精确边界是什么？先给一个定义。**所谓 morally strong**（PTX Memory Consistency Model 节，[NVIDIA, PTX ISA](../references.md)，[C]），是指两个内存操作满足全部三条：(i) 同一线程的 program order，或两者各为 strong 操作且各自 scope 包含对方线程；(ii) 同一 proxy；(iii) 访问地址完全重叠。两个冲突操作既无 causality order 又非 morally strong，即 data race，行为未定义。正例：两线程对同一 global 地址做 device scope atomicAdd——各为 strong、scope 互含、地址重叠，morally strong，硬件保证串行。反例：block scope 的 atomic 写与另一 block 的普通读——scope 不含对方，非 morally strong，race。

**命题 1**（fence 的 ordering-可见性边界）。(a) `__threadfence()` 在 device scope 建立 calling 线程内存操作的 seq_cst 顺序：对 device 内任一线程，calling 线程 fence 后的写不可能被观察为发生在 fence 前的写之前——等价地，若 fence 后的写被（morally strong 地）观察到，fence 前的写必已可观察。(b) fence 单独**不构成**跨线程 happens-before：PG 逐字「Memory fence functions only affect the ordering of memory operations by a thread; they do not, by themselves, ensure that these memory operations are visible to other threads (like `__syncthreads()` does for threads within a block)」；可见性须配对——读侧以 acquire load 或 morally strong 的 atomic 观察建立同步边，与 fence 的排序边合成因果链。(c) 正确的消息传递模式：生产者「写数据 → fence → flag 原子写」或「写数据 → flag release store」（release 自带排序），消费者「flag acquire load → 读数据」。**证明**（路线）：三步——先由 PG 等价逐字把 fence 归约为 `fence.sc.gpu`；再按 PTX 内存模型，causality order 由 program order、fence 序与同步边（release/acquire 配对或 morally strong 观察）合成，fence 只贡献排序边、不贡献同步边；最后对 (c) 两模式各验证一条完整因果链。关键一步是区分**排序边**（fence）与**同步边**（配对操作）——只有排序边没有同步边，读侧仍可能读旧值。完整证明（含 morally strong 三条件逐条验证与缺配对反例构造）留作练习 3。∎

两个模式的教学片段。libcu++ release/acquire flag 模式（[NVIDIA, libcu++ memory model](../references.md) 官方例同形）：

```cpp
// 教学片段，非优化实现：跨 block 消息传递（release/acquire flag）
__device__ int data = 0;
__device__ cuda::atomic<int, cuda::thread_scope_device> flag{0};
// 生产者 block：                          // 消费者 block：
data = 42;                                 while (flag.load(cuda::memory_order_acquire) != 1) {}
flag.store(1, cuda::memory_order_release); int r = data;    // r == 42 有保证
```

经典 last-block 模式（PG fence 节语义骨架，官方教学例 cuda-samples `cpp/2_Concepts_and_Techniques/threadFenceReduction`，教学参照）：

```cpp
// 教学片段，非优化实现：last-block 归约（fence + relaxed atomic 计数）
__device__ unsigned int count = 0;
__global__ void last_block_reduce(float* partials, float* out, unsigned int n_blk) {
    float partial = block_reduce();                  // block 内归约（5.1）
    partials[blockIdx.x] = partial;                  // 普通 store
    __threadfence();                                 // 排序：partial 写先于计数增量
    unsigned int ticket = atomicInc(&count, n_blk);  // relaxed、device scope，返回旧值
    if (ticket == n_blk - 1) {                       // 最后一个 block
        float total = 0;
        for (unsigned int i = 0; i < n_blk; i++) total += partials[i];
        *out = total;
    }
}
```

ordering 链这样读：每个 block 的 partial 写被 fence 排在计数增量之前；最后一个 block 的 atomicInc 返回 $n_{\mathrm{blk}} - 1$，意味着它观察到其余全部增量已发生（同址 atomic 串行，§5），沿 fence 排序边回溯，全部 partial 写对它可观察。grid ⊂ device，所以 device fence 覆盖 block 间通信绰绰有余——但 scope 大于必要即成本大于必要，block 内通信用 `__threadfence_block()` 或 `__syncthreads()` 就够（§4）。libcu++ 现代重写：把 fence + relaxed 组合换成一条 acq_rel 的 RMW——

```cpp
// 教学片段：acq_rel fetch_add 一条替代 fence + relaxed 组合
// counter 为 cuda::atomic<unsigned int, cuda::thread_scope_device>
unsigned int ticket = counter.fetch_add(1, cuda::memory_order_acq_rel);
```

release 侧把 partial 写排进增量之前，acquire 侧让读到票号的线程接住全部先前写（release sequence 上每个 RMW 延续 release），等价性论证留作练习 3(c)。官方旧示例用 volatile 读 result——按 volatile 新口径（§3）该写法已不推荐，排序链本身由 fence + atomic 保证，与 volatile 无关。

## 3. 一致性：L1 非一致、L2 一致点、volatile 新口径

SM 0 写了 flag，SM 1 自旋读，可能永远读不到新值——为什么？recap 2.2 §6 的四层一致性（不重推）：L1 每 SM 私有、跨 SM 非一致；L2 是全芯片一致点；跨 GPU 非一致；Blackwell 封装内双 die 经 NV-HBI 例外一致。本节补编程模型侧：代码要写什么才保证可见。

L1 非一致的官方披露在 PTX cache operators 节（[NVIDIA, PTX ISA](../references.md)，[C]），逐字：「Global data is coherent at the L2 level, but multiple L1 caches are not coherent for global data.」并给机制例：「If one thread stores to global memory via one L1 cache, and a second thread loads that address via a second L1 cache with ld.ca, the second thread may get stale L1 cache data」；跨 kernel 的可见性由 driver 兜底——「The driver must invalidate global L1 cache lines between dependent grids of parallel threads.」所以边界是：**kernel 之间**的 global 可见性由 driver 失效 L1 保证，**kernel 内跨 SM** 的可见性必须靠 fence/atomic 语义，硬件不自动失效。

cache operator 一览（PTX 逐字语义，[NVIDIA, PTX ISA](../references.md)，[C]）：

| 操作符 | 语义 | 典型用途 |
| --- | --- | --- |
| `ld.ca`（默认） | 各级缓存都进（L1 + L2） | 普通 load |
| `ld.cg` | 只进 L2 及以下，bypass L1 | 不想污染 L1 的流式读 |
| `ld.cs` | evict-first 流式 | 一次性数据 |
| `ld.lu` | last use | spill restore |
| `ld.cv` | 不缓存、每次重取 | system memory 读 |
| `st.wb` / `st.cg` / `st.cs` / `st.wt` | write-back / L2 / 流式 / write-through | store 侧对应 |

关键逐字：「Cache operators on load or store instructions are treated as performance hints only」，「does not change the memory consistency behavior」——**cache operator 是性能提示，不是同步机制**。`ld.cg` 绕过 L1 只是改变缓存行为，正确性仍须由 fence/atomic 的语义保证；把「bypass L1」当「跨 SM 可见」用是语义错位。

> **坑**：拿 `ld.cg` / `__ldcg` 当跨 SM 可见性手段。触发条件：知道 L1 非一致后，改用 bypass L1 的 load 读别的 SM 写的 flag，以为绕过 L1 就拿到新值；现象：cache operator 只是性能提示（PTX 逐字），不改变内存一致性行为，读侧仍可能因重排或缺同步边读到旧值；规避：可见性一律走 fence/atomic 语义（§2 命题 1），cache operator 只作性能调优。

volatile 的口径在 CUDA 13.x 已变，PG Volatile Qualifier 节逐字（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）：「The volatile keyword is supported to maintain compatibility with ISO C++; however, few if any of its remaining non-deprecated uses apply to GPUs. Reads and writes to volatile qualified objects are not atomic, and are compiled to one or more .volatile instructions which do NOT guarantee: ordering of memory operations, or that the number of memory operations performed by the HW matches the number of PTX instructions.」并明确「CUDA C++ volatile is NOT suitable for: Inter-Thread Synchronization: Use atomic operations via cuda::atomic_ref, cuda::atomic, or Atomic Functions instead.」，tile 代码里「the volatile keyword has no effect on the behavior of memory accesses」。三条钉死：volatile 读写不原子；不保证排序、也不保证硬件操作数与 PTX 指令数一致；**不适合线程间同步**——旧书「volatile bypass L1 实现跨 block 可见」的模式 deprecated，跨线程同步一律 atomic / fence。

> **坑**：照抄旧示例的 volatile 同步模式。触发条件：旧版 PG fence 节的 last-block 示例仍用 volatile 读 result，与同书 Volatile Qualifier 节新口径内部不一致；现象：把 volatile 当可见性保证写新代码，在编译器优化与硬件重排下偶发读旧值；规避：以 Volatile Qualifier 节新口径为准——同步用 `cuda::atomic_ref` / `cuda::atomic` / Atomic Functions，volatile 只留给 ISO C++ 兼容场景（本书 §2 经典片段的排序链由 fence + atomic 保证，不依赖 volatile）。

数值落点（本章口径）：自旋读 flag 用 atomic load，每次轮询落到 L2（NCU 逐字「Atomics and reductions are passed through to the L2 cache.」，[NVIDIA, Nsight Compute](../references.md)，[C]），轮询一轮至少 L2 往返 263.0 cycles ≈ 143.7 ns @1.83 GHz（L2 load-to-use 实测口径作 atomic 往返的量级下界）；若命中自己 L1 的旧值只要 40.7 cycles——快 6.46 倍（ $263.0/40.7$ ）但**错**。正确性的价格就是这个延迟差：一致性点在哪，正确读法就得付到哪一层的延迟。

## 4. 同步原语：barrier 与可见性双语义

`__syncthreads()` 做几件事？两件——barrier（全 block 到达）加可见性（prior 访存对 block 内全线程可见），少算一件就会写错。PG 逐字：「waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `__syncthreads()` are visible to all threads in the block」（[NVIDIA, CUDA Programming Guide](../references.md) Synchronization Functions 节，[C]）。所以它等价于 block 级 barrier + block scope 的可见性保证，shared 数据交换的标准写法「写 shared → `__syncthreads()` → 读邻居的 shared」正确性全在这条双语义上。条件分支里的 `__syncthreads()` 须全 block 同值到达，否则 hang（PG）——divergence（3.1 §4）使部分线程永远到不了 barrier 就是死锁。

> **坑**：divergent 分支里放 `__syncthreads()`。触发条件：barrier 写在 if/else 一侧，而分支条件在 block 内不一致（warp 间或 warp 内分歧）；现象：走另一侧的线程永远不到达，全 block 卡死（不是算错，是 hang）；规避：barrier 必须放在 block 内所有线程一致执行的路径上；warp 级分歧用 `__syncwarp(mask)`（只要求 mask 内线程到达）。

`__syncwarp(mask)` 逐字「guarantees memory ordering among threads participating in the barrier」（PG）——mask 指定参与 lane，保证参与线程间的内存排序。Volta+ 每线程独立 PC（recap 3.1 §3），warp 内「天然锁步」的隐式假设失效，warp-synchronous 代码必须显式 `__syncwarp` 或带 `_sync` 后缀的原语。`__syncthreads_count/and/or` 是 barrier 加谓词归约的变体（统计到达线程中谓词为真的数量等），语义同族。

PTX 与硬件 backing（[NVIDIA, PTX ISA](../references.md)，[C]）：`bar.sync` / `barrier.cta.sync` 是 CTA 内 named barrier，数量有限；mbarrier 是 shared memory 里的用户定义 barrier 对象，支持 arrive/wait 拆分与异步事务计数（tx-count），「mbarrier is a barrier created in shared memory」且可同步 CTA 内任意线程子集、等待异步访存完成——cp.async/TMA 的完成机制与 Hopper 异步事务屏障的硬件账归 4.4，本章只立语义。cooperative groups 侧：`block.sync()` 等价 block barrier，`grid.sync()` 须共驻留（recap 3.1 命题 3，不重证），tile 的 `.sync()` 即 warp 级同步。

同步原语总表：

| 原语 | scope | 语义 | backing |
| --- | --- | --- | --- |
| `__syncthreads()` | block | barrier + prior global/shared 可见 | SM 内硬件 barrier |
| `__syncwarp(mask)` | warp（mask 内） | 参与线程间内存排序 | warp 级汇合 |
| `__threadfence_block()` | block | 仅排序（不保证可见，命题 1） | `fence.sc.cta` |
| `grid.sync()` | grid | 全 grid barrier（须共驻留） | cooperative launch |
| mbarrier | CTA 子集 / cluster | arrive/wait 拆分 + tx-count | shared 对象（4.4） |

barrier 把 divergence 代价从 warp 级放大到 block 级（跨章数值，recap 3.1 §4 D4 口径）：256 线程 block = 8 warps，barrier 的等待时间 = 最慢 warp 的到达时间；若 1 个 warp 走 10 条指令的重路径、其余 7 个走 2 条的轻路径，7 个 warp 在 barrier 前空等 8 条指令的时间。divergence 本身折的是 warp 内吞吐（3.1 式 (3)），加上 barrier 后折的是整 block 的进度——两笔账叠加，这是「divergent kernel 加 barrier 格外慢」的机制。

## 5. atomic：scope、串行化与非确定性

atomicAdd 保证「不丢更新」，但不保证「每次跑出同一个结果」——对 fp32，这两件事不同。所谓原子操作（atomic operation），是指把 read-modify-write 三步合成一个不可分割的事务，PG 逐字「An atomic function performs a read-modify-write atomic operation on one 32-bit, 64-bit, or 128-bit word residing in global or shared memory.」（[NVIDIA, CUDA Programming Guide](../references.md) Atomic Functions 节，[C]）。CUDA atomic 函数默认 relaxed 序、按后缀定 scope：「The atomic functions described in this section have ordering cuda::memory_order_relaxed and are only atomic at a particular scope」——无后缀 device scope，`_block` 后缀 block scope，`_system` 后缀 system scope（有条件）。relaxed 的含义：原子性有、顺序无——要排序须显式 memory_order 或配 fence（§2）。PTX 侧语法（[NVIDIA, PTX ISA](../references.md) atom 节，[C]）：

```
atom{.sem}{.scope}{.space}.op.type d, [a], b;
.sem   = { .relaxed, .acquire, .release, .acq_rel }
.scope = { .cta, .cluster, .gpu, .sys }
.space = { .global, .shared{::cta, ::cluster} }
```

warp 内同址的两条官方语义（PG SIMT 节逐字，[NVIDIA, CUDA Programming Guide](../references.md)，[C]）：非原子写——「If a non-atomic instruction executed by a warp writes to the same location... which thread performs the final write is undefined.」；原子写——「If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.」前者连「哪个写活下来」都不定，后者保每个 RMW 都发生、但顺序不定。

**命题 2**（atomic 累加非确定性）。 $n_{\mathrm{th}}$ 个线程对同一 fp32 地址 atomicAdd，结果的比特值依赖各 RMW 的完成顺序；该顺序官方未定义（上文逐字），故运行间比特级不可复现——只要浮点加法非结合使不同顺序产生不同舍入。**证明**（要点）：串行化保证每个 RMW 恰好发生一次（不丢更新），但 $n_{\mathrm{th}}$ 个加数的求和顺序有 $n_{\mathrm{th}}!$ 种可能；fp32 加法非结合，不同顺序的中间舍入不同，结果比特值随之不同。∎

全数字反例（fp32，先预测再算）：取 $a = 10^{8}$、 $b = -10^{8}$、 $c = 1$，

$$
(a + b) + c = 1, \qquad a + (b + c) = 0 \qquad (1)
$$

**符号**： $a, b, c$ 均 fp32。机制：fp32 有效位 24 bit（23 存储 + 1 隐含）， $10^{8}$ 落在 $2^{26}$ 与 $2^{27}$ 之间，该量级 ulp（相邻可表示值间距）为 $2^{26-23} = 8$； $b + c = -99999999$ 距 $-10^{8}$ 为 1、距 $-99999992$ 为 7，舍入到 $-10^{8}$，于是 $a + (b+c) = 0$——加数 1 被整个吞掉。另一经典形态：fp32 顺序累加 1，到 $2^{24} = 16777216$ 后 ulp 变 2，加 1 舍入回原值，2000 万个 1 累加停在 16777216。非确定性不是「小误差」，是加数级别的丢失。

这条语义直通 serving 侧的可复现性问题，recap inference 1.4 §6（纯文字指引，不重推）：浮点非结合是 reduction 漂移的根源；「concurrency + floating point」假说（atomicAdd 累加顺序随并发完成顺序不定）**只是部分解释**——batch 组成改变 reduction 的分区与顺序，无需并发非确定性同样改变结果；解法是 batch-invariant kernels 固定 reduction order（split-K 固定分区、禁 atomicAdd 非确定累加），引擎模式入口与 kernel 实现归 inference 1.4 / 3.4。本章交付的是它的硬件-语义来源：顺序未定义是官方披露的硬件行为，不是实现瑕疵。确定性替代路线是固定顺序的树形归约（5.1），代价是放弃 atomic 的任意顺序并行度。

atomic 的延迟账（本章口径）：global atomic 落到 L2（NCU 逐字，§3），device scope 一次 RMW 至少 L2 往返 263.0 cycles ≈ 143.7 ns；shared atomic（`atomicAdd_block`）留在片上，按 shared 访问延迟 29.0 cycles ≈ 15.8 ns 作量级估计（RMW 略高于纯 load，官方未披露 atomic 专属延迟，两值均实测 [B] 口径）——scope 选小一级，延迟缩 $263.0/29.0 = 9.07$ 倍。同址串行化另计：32 线程同址 atomicAdd 串行成 32 次 RMW（顺序不定），热点地址的 atomic 吞吐被串行化封顶。

histogram 两阶段账（式 (2)；配置换 grid-stride 1056 blocks × 256 threads，原因见章脊口径声明）： $n_{\mathrm{vec}} = 16777216$ 个元素、 $n_{\mathrm{bins}} = 256$ 个 bin。直接法每元素一次 global atomicAdd，共 16777216 次 L2 RMW；两阶段法先做 block 私有 shared histogram（256 bin × 4 B = 1 KB shared/block，shared atomic 片上），block 结束后每 bin 一次 global atomic，共 $n_{\mathrm{blk}} \cdot n_{\mathrm{bins}}$ 次。节省因子：

$$
\rho = \frac{n_{\mathrm{vec}}}{n_{\mathrm{blk}} \cdot n_{\mathrm{bins}}} \qquad (2)
$$

**符号**： $\rho$ = global atomic 次数节省因子； $n_{\mathrm{blk}}$ = block 数（1056）； $n_{\mathrm{bins}}$ = bin 数（256）。代入： $\rho = 16777216/(1056 \times 256) = 16777216/270336 = 62.06$——等于每线程处理的平均元素数（最大迭代次数向上取整为 63，recap 3.1 §2 grid-stride 口径）。机制一句话：把「每元素一次远端 RMW」换成「每元素一次片上 RMW + 每 block 每 bin 一次远端 RMW」，远端流量按每线程元素数摊薄。bin 数增大或 block 数增多都摊薄节省（式 (2) 分母），shared 容量（1 KB/block 对 228 KB 库存）是 bin 数的上界约束。

## 6. unified memory 与内存池：迁移与分配的开销

`cudaMallocManaged` 给你一个 CPU/GPU 都能解引用的指针——代价藏在 page fault 里。所谓统一内存（unified memory / managed memory），是指单一一致地址空间、按需在处理器间迁移物理页的分配方式，支持 oversubscription（分配总量可超显存容量，[NVIDIA, CUDA Programming Guide](../references.md) Unified Memory 节，[C]）。software-coherent 系统上，一次 page fault 的三步官方逐字：「it needs to be ensured that the currently owning processor... cannot access this page anymore... the processor requesting access can access this page... the physical page backing this virtual page must be moved/migrated to the processor requesting access: this can be an expensive operation, and the amount of work is proportional to the page size.」拆三步：owner 侧页表失效（不能再访问）→ requester 侧页表建立（可以访问）→ 物理页拷贝迁移，工作量与页大小成正比。hardware-coherent 平台（Grace Hopper）fault 更少、一致性粒度细到 cache line 而非页（PG）——平台不同，开销结构不同。

迁移账推导路线分两步：先算单页迁移的带宽项，再对总字节求下界；关键一步是分清**带宽项**（物理拷贝，不可省）与**fault 处理开销**（页表操作与中断，按页计、可预取摊销）。单页带宽项：

$$
t_{\mathrm{page}} = \frac{P_{\mathrm{page}}}{\mathrm{BW}} \qquad (3)
$$

**符号**： $P_{\mathrm{page}}$ = 迁移页大小（教学假设取 2 MiB = 2097152 B；PG 只披露 cost ∝ page size，实际页大小由驱动/平台定）； $\mathrm{BW}$ = 互连有效带宽（取 PCIe Gen5 x16 理论单向 64 GB/s，2.4 §4 表口径，[PCI-SIG, PCIe](../references.md)，[C]；GB/s 按 $10^{9}$ B/s）。代入： $t_{\mathrm{page}} = 2097152/(64 \times 10^{9}) = 32.768$ μs。总迁移时间下界（ $M$ = 迁移总字节）：

$$
t_{\mathrm{mig}} \ge \frac{M}{\mathrm{BW}} \qquad (4)
$$

全数字代入：1 GiB buffer（ $M = 1073741824$ B = 512 个 2 MiB 页）→ 下界 $1073741824/(64 \times 10^{9}) = 16.777$ ms，与 512 × 32.768 μs 逐字对上。fault 模式在下界之上再付每页的 fault 处理开销，且 fault 是 demand 触发——计算线程停下来等页，串行暴露在关键路径上；`cudaMemPrefetchAsync` 把迁移变成 stream-ordered 的批量操作、`cudaMemAdvise`（SetPreferredLocation / SetAccessedBy）给迁移 hint，二者不改变式 (4) 的带宽下界，改变的是「谁在等」：预取可与计算重叠（stream 语义归 3.3），把迁移从关键路径挪走。

> **坑**：把 managed memory 当「免费迁移」。触发条件：kernel 首次 touch 大块 managed buffer，无 prefetch/advise；现象：按页 demand fault，三步串行（失效 → 建表 → 拷贝），计算停等，1 GiB 首触至少 16.8 ms 带宽项外加每页 fault 开销；规避：热 buffer 用 `cudaMemPrefetchAsync` 预取 + `cudaMemAdvise` 定首选位置，或干脆显式 `cudaMalloc` + 拷贝（overlap 账归 3.3）。

分配侧的开销另一笔：`cudaMalloc` / `cudaFree` 是昂贵的同步驱动操作，Best Practices 推荐以 `cudaMallocAsync` 替代（[NVIDIA, Best Practices](../references.md)，[C]）。所谓 stream-ordered 内存池（memory pool），是指分配/释放按 stream 顺序入队、从池中复用空闲块的分配器，PG 逐字「cudaMallocAsync and cudaFreeAsync form the core of the allocator」，「Both APIs accept stream arguments to define when the allocation will become and stop being available for use. The pointer value returned by cudaMallocAsync is determined synchronously」（[NVIDIA, CUDA Programming Guide](../references.md) Stream Ordered Memory Allocation 节，[C]）。机制账：指针值同步确定（host 立即可用），但内存的**可用性**由 stream 顺序定义——同 stream 内前序操作完成后才可用；`cudaFreeAsync` 把块还池而非立即还驱动，后续同 stream 分配从池里拿，驱动级分配开销被摊销。default pool 不可迁移（PG）。serving 循环里每步 alloc/free 的场景，这笔摊销直接落在步时上（stream 语义与 overlap 归 3.3）。

## 7. 边界：本章不做什么

语义边界划清之后，剩下的账归谁？本章交付地址空间语义与 backing、内存模型（scope/order/fence）、一致性的编程模型侧、同步原语语义、atomic 非确定性来源、UM 迁移与内存池机制账；以下各有专章。coalescing 与 sector 账归 4.1，bank conflict 归 4.2（本章只引 shared 29.0 cycles 延迟口径）；cp.async/TMA 的完成机制与 mbarrier 硬件账归 4.4；确定性树形归约实现归 5.1，batch-invariant kernel 与引擎模式归 inference 1.4 / 3.4；stream/event/graph 与拷贝重叠归 3.3；occupancy 与 shared 配置归 3.4；L1/L2 容量-延迟-带宽全表与测量归 2.2 / 2.4。代码 scope：机制片段级（§2 消息传递与 last-block、§5 histogram 口径），不做模块走读。

本章交付清单（下游 recap 接口）：地址空间 backing 表（§1）、scope 五级 + fence 三级映射 + 命题 1（§2）、volatile 新口径与 cache operator 边界（§3）、同步原语双语义表（§4）、命题 2 + 式 (1)(2)（§5）、迁移账式 (3)(4)（§6）。4.4 recap §4 mbarrier 语义作异步完成机制起点；5.1 recap §5 非确定性作确定性归约动机；3.3 recap §6 stream-ordered 语义；4.1/4.2 recap §1 backing 表作物访存优化的硬件侧。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 六地址空间四种 backing：register/shared 片上，local/global/constant/texture 落 HBM（各带专属缓存）；local 逻辑私有 ≠ 物理片上 | §1 表 + PG 逐字 | §1、练习 1 |
| HBM 对 shared 延迟比 16.5 倍；constant 64 KB = 16384 fp32；只读旁路（ld.global.nc）不适用内存模型 | §1 数值 + PTX 逐字 | §1、练习 1 |
| thread_scope 五级（system/device/cluster/block/thread）；pre-SM90 cluster 放宽为 device | §2 enum 逐字 | §2 |
| fence 三级映射：__threadfence_block/device/system ≡ seq_cst fence block/device/system ≡ membar.cta/gl/sys ≡ fence.sc.cta/gpu/sys（sm_70+） | §2 映射表 + 双逐字 | §2、练习 2 |
| fence 只管 ordering 不保证 visibility；跨线程 happens-before 须 release/acquire 或 atomic flag 配对 | 命题 1 | §2、练习 2/3 |
| L1 跨 SM 非一致、L2 一致点；kernel 间 driver 失效 L1，kernel 内靠 fence/atomic；cache operator 只是性能提示 | §3 PTX 逐字 + 坑注 | §3 |
| volatile 13.x 新口径：不原子、不保序、不适合线程间同步；旧 bypass-L1 模式 deprecated | §3 PG 逐字 + 坑注 | §3、练习 3 |
| __syncthreads = barrier + block 可见性双语义；divergent 分支内放置即 hang；__syncwarp 保 mask 内排序 | §4 PG 逐字 + 坑注 | §4 |
| atomic 默认 relaxed + scope 后缀；同址 warp atomic 串行但顺序未定义 → fp32 累加比特级非确定 | 命题 2，式 (1) | §5、练习 4 |
| fp32 非结合实锤：(1e8−1e8)+1 = 1 对 1e8+(−1e8+1) = 0；顺序累加 1 停在 $2^{24}$ | 式 (1) + ulp 账 | §5、练习 4 |
| global atomic 落 L2（263.0 cycles）对 shared atomic 片上（29.0）= 9.07 倍；histogram 两阶段节省 ρ = 62.06 | 式 (2) + NCU 逐字 | §5、练习 4 |
| UM page fault 三步（失效 → 建表 → 迁移），cost ∝ page size；2 MiB 页 32.768 μs、1 GiB 下界 16.777 ms | 式 (3)(4) | §6、练习 4 |
| cudaMallocAsync stream-ordered 池：指针同步定、可用性按 stream 序、free 还池摊销驱动开销 | §6 PG 逐字 | §6 |

## 练习

1.（计算 · [10]）地址空间与延迟（本章口径）：(a) HBM 对 shared 的延迟比（478.8 对 29.0 cycles）；(b) constant 空间 64 KB 装得下几个 fp32；(c) CC 9.0 上 local memory 的物理 backing 与是否经 L2。校验：(a) ≈16.5，与 §1 一致；(b) 16384；(c) HBM（device memory）、恒经 L2（PG 逐字，§1）。
2.（推导 · [20]）fence litmus（命题 1 应用）：(a) §2 libcu++ 片段里消费者为什么保证读到 r == 42——指出排序边与同步边各来自哪条操作；(b) 经典 last-block 片段若去掉 `__threadfence()`，最后一个 block 可能读到什么、为什么；(c) 消费者把 acquire load 换成 relaxed load，保证还在吗？校验：(a) 与命题 1(c) 一致；(b) partial 旧值（排序边缺失，命题 1(a) 的反面）；(c) 不在——缺同步边（命题 1(b)）。
3.（推导 · [30]）补全命题 1 的证明与 last-block 正确性：(a) 逐条验证「两线程 device scope atomicAdd 同址」满足 morally strong 三条件；(b) 写出经典 last-block 模式的完整 ordering 链（partial 写 → fence → atomicInc；ticket = n_blk−1 → 读 partials），说明每一跳的依据；(c) 论证 acq_rel fetch_add 重写与 fence + relaxed 组合等价（release sequence：RMW 延续 release）；(d) 说明旧示例 volatile 读 result 为何按新口径不推荐、排序链是否依赖 volatile。校验：(b) 与 §2 经典片段及 cuda-samples threadFenceReduction 语义一致；(c) 与命题 1(c) 两模式等价；(d) 与 §3 volatile 逐字一致（不原子、不保序）。
4.（计算 · [30]）atomic 非确定性与迁移账：(a) fp32 下算 $(10^{8} - 10^{8}) + 1$ 与 $10^{8} + (-10^{8} + 1)$——先预测两者是否相等再算，用 ulp 解释；(b) histogram 两阶段（1056 blocks × 256 bins）的 global atomic 次数与节省因子 ρ；(c) 2 MiB 单页迁移带宽项与 1 GiB 总下界（64 GB/s 口径）。校验：(a) 1 对 0，与式 (1) 一致；(b) 270336 次、ρ ≈ 62.06，与式 (2) 一致；(c) 32.768 μs 与 16.777 ms，与式 (3)(4) 一致（512 页对账）。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——**引文版本归属见 §1 注**：下列逐字引文除 SIMT 同址 atomic 两句与 libcu++ 建议句为 13.5 逐字外，均为 archive/13.0.0 逐字（13.5 子页改写措辞、语义相同，互证；节号为 13.5 现行）：05-appendices/cpp-language-extensions.html §5.4.4.3 Memory Fence Functions（三级 fence ≡ cuda::atomic_thread_fence seq_cst block/device/system 逐字 + fence 只 ordering 不保证 visibility 逐字 + 与访存空间无关逐字 + libcu++ 建议逐字〔13.5〕）+ Atomic Functions（32/64/128-bit word global/shared 逐字 + relaxed 默认与 scope 后缀逐字）+ Synchronization Functions（__syncthreads 双语义逐字 + conditional 须全 block 同值 + __syncwarp mask 逐字）+ Volatile Qualifier（13.x 新口径逐字：不原子、不保序、NOT suitable for Inter-Thread Synchronization、tile 代码无效）+ Variable Memory Space Specifiers；02-basics/writing-cuda-kernels.html Device Memory Accesses（local resides in device memory + CC5.x+ 恒 L2、shared on-chip、constant cache 拆分请求、texture 2D locality + dedicated units 逐字）；04-special-topics/unified-memory.html（page fault 三步逐字 + Grace Hopper 硬件一致 + cudaMemPrefetchAsync/cudaMemAdvise）；04-special-topics/stream-ordered-memory-allocation.html（cudaMallocAsync/cudaFreeAsync 逐字 + default pool non-migratable）；warp 同址写/atomic 顺序未定义逐字（SIMT 节，13.5 逐字）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——§9.7.15.4 fence/membar（sm_70+ membar ≡ fence.sc、cta/gl/sys ≡ cta/gpu/sys 逐字 + fence.sem/.scope 语法）；§8 Memory Consistency Model（morally strong 三条件 + causality order + data race 定义 + texture 含 ld.global.nc 不适用逐字）；§9.7.15.5 atom（.sem/.scope/.space 语法）；§9.7.10.1 Cache Operators（.ca「multiple L1 caches are not coherent for global data」逐字 + stale L1 例 + driver invalidate between dependent grids + performance hints only 逐字）；§5.1 State Spaces 表（.reg/.local/.shared/.global/.const/.tex/.param）；mbarrier 节（shared 对象 + arrive/wait + tx-count，§4 预告 4.4）
- [NVIDIA, libcu++ memory model](../references.md)：CCCL libcu++ Extended API — Memory model（key: libcudacxx-memory-model），[C] 官方规格披露·一手权威，2026-11-25——thread_scope 五级 enum 逐字 + device scope 定义（same device + memory synchronization domain）+ pre-SM90 cluster widened 逐字 + 同步成本随距离增长逐字 + release/acquire flag 消息传递官方例（现行 URL /cccl/unstable/libcudacxx/extended_api/memory_model.html，旧路径 404 漂移注）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——ProfilingGuide Memory Workload Analysis「Atomics and reductions are passed through to the L2 cache」逐字（§3/§5 atomic 延迟口径）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/2_Concepts_and_Techniques/threadFenceReduction（fence + atomic last-block 模式官方教学例，§2；现行 cpp/ 路径）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——cudaMallocAsync 替代 cudaMalloc（expensive）推荐（§6）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV 延迟 cycles（shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8），本章延迟口径为实测 [B] 非 H100 官方规格（沿 2.2 口径声明）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露·一手权威，2026-11-24——Gen5 x16 = 64 GB/s 单向理论口径（§6 迁移账带宽取值，沿 2.4 §4 表；精确 URL W4 复核残留）
