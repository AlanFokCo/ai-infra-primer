# 3.1 线程层级与 SIMT（Thread Hierarchy and SIMT）

> 一行 launch 配置把 1677 万个线程跑在 132 个 SM 上——thread/warp/block/grid 四级抽象各映射到什么硬件，divergence 为什么把有效吞吐砍半、且代价与分支人数比例无关？本章交付层级语义与索引公式（命题 1 双射）、SIMT vs SIMD 与 Volta 独立 PC、divergence 代价账（命题 2：等代价 k 路径吞吐 1/k）、block 调度与波次量化（GigaThread 术语时代注）、cooperative groups 共驻留命题与 warp 原语预告 | 前置：1.2（warp = 32、block 驻留单 SM、资源库存与 occupancy 上界式 (3)）、1.4（warp scheduler 单发射、延迟隐藏）；下游 3.2（内存模型与同步）、3.3（stream/event/graph）、3.4（occupancy 精确计算）、4.1（coalescing）、5.1（shuffle/vote 与 reduction/scan） | 锚点：CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、Fermi whitepaper（GigaThread 一手出处，[NVIDIA, Fermi whitepaper](../references.md)，[C]）、cuda-samples（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）——全部 2026-11-25 一手核实

## 1. 四级层级：1677 万个线程怎么装进 132 个 SM

一个 elementwise kernel 要处理 16777216 个 fp32 元素，每元素一个线程——1677 万个线程，而 H100 SXM 只有 132 个 SM、每 SM 至多驻留 2048 个线程（recap 1.2 库存表），全芯片同时装得下的线程不过 27 万。1677 万对 27 万，差 62 倍，怎么跑？答案是线程并不全部同时驻留：grid/block/warp/thread 四级层级把「逻辑线程总数」与「物理驻留线程数」拆开，block 是驻留与调度的单位，warp 是执行的单位，逻辑上无限的 grid 按波次流过物理有限的 SM。本章回答三个问题：四级层级各是什么语义、索引怎么算？SIMT 与 SIMD 差在哪、divergence 的代价怎么算？block 怎么分到 SM、编程模型给了什么保证、没给什么保证？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 1.1）；running example 为 elementwise 向量 kernel，向量长 $n_{\mathrm{vec}} = 16777216$（ $2^{24}$ ）个 fp32 元素，launch 配置 65536 blocks × 256 threads/block（1D）；cycle ↔ 时间换算取 1.83 GHz（沿 1.4/2.2）。资源口径：该 kernel 设 $R_{\mathrm{th}} = 32$、 $S_{\mathrm{sh}} = 0$（教学假设，elementwise 典型形态），代入 1.2 式 (3) 得每 SM 驻留 min(32, 8, 8, $\infty$) = 8 blocks、 $\mathcal{O} = 1.0$——此假设只用于 §5 的调度波次账，不作一般性结论。

符号声明： $n_{\mathrm{vec}}$ = 向量元素数（本章局部符号；归约/扫描元素数 $n_{\mathrm{el}}$ 归 5.1，此处不占用）； $n_{\mathrm{blk}}$ = grid 的 block 数（65536）； $w_b$ = warps/block（recap 1.2）； $n_{\mathrm{SM}}$ = SM 数； $\mathcal{O}$ = occupancy； $R_{\mathrm{th}}$ = registers/thread； $S_{\mathrm{sh}}$ = shared bytes/block；warp size = 32 为常量。divergence 账的 $k$、 $m_j$、 $c_j$、 $\eta$ 见 §4。

所谓线程层级（thread hierarchy），是指 CUDA 把一次 kernel launch 的全部线程组织成 grid → thread block → warp → thread 四级嵌套（[NVIDIA, CUDA Programming Guide](../references.md) Thread Hierarchy 节，[C]）。各级语义与硬件映射：

| 层级 | 对象 | 协作范围 | 硬件映射 |
| --- | --- | --- | --- |
| grid | 一次 kernel launch 的全部 block | block 间无顺序保证、无直接通信 | block 调度器分发到有空闲容量的 SM（§5） |
| thread block（线程块） | 至多 1024 线程 | shared memory + block 级 barrier | 整体驻留单个 SM，独占四类资源（1.2） |
| warp | 32 个连续编号线程 | warp 内原语（§6 预告，5.1） | scheduler 的发射单位，1 条 warp 指令覆盖 32 lane（1.2 命题 1） |
| thread | 标量执行流 | 私有 register / local | 一个 lane；Volta+ 每线程独立 PC（§3） |

层级结构图（本章口径数字）：

```
grid：65536 blocks × 256 threads = 16777216 threads（逻辑总量）
├─ block 0 ─── 驻留 SM a ──┐
│   ├─ warp 0：thread 0..31        │
│   ├─ warp 1：thread 32..63       │ 8 warps/block（256/32）
│   └─ ... warp 7：thread 224..255 │
├─ block 1 ─── 驻留 SM b           │ block 不可跨 SM（PG 逐字）
│   └─ ...                         │
└─ block 65535 ────────────────────┘
物理驻留：132 SM × 8 blocks = 1056 blocks = 8448 warps = 270336 threads
逻辑总量 524288 warps ÷ 驻留 8448 warps ≈ 62 个波次（§5）
```

所谓 warp（32 线程），是指 block 内连续 32 个线程构成的硬件调度单位：threadIdx 编号 0 到 31 是 warp 0，32 到 63 是 warp 1，依此类推；每 block 的 warp 数 $`w_b = \lceil \mathrm{threads}_{\mathrm{block}} / 32 \rceil`$（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。warp 不是编程对象：CUDA C++ 没有 warp 类型，threadIdx 也不直接给 warp 编号——它是硬件调度粒度向编程模型的「泄漏」，warp 级原语（§6）全部建立在这个泄漏之上。正例：256 threads/block → $w_b = 8$，65536 blocks → 全 grid 共 $65536 \times 8 = 524288$ 个 warp。边界反例：257 threads/block → $w_b = \lceil 257/32 \rceil = 9$，最后一个 warp 只有 1 个活跃线程、其余 31 个 lane 生来空转——block 大小不是 32 的倍数时，多出的 warp slot 照占、lane 白占，这是「block 尺寸取 32 倍数」的定量理由。

所谓线程块（thread block），是指能用 shared memory 与 block 级 barrier 协作的一组线程，PG 逐字「all threads of a block are expected to reside on the same streaming multiprocessor core」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——**block 不可跨 SM**。这不是调度偏好而是硬件约束：shared memory 是每 SM 的片上 SRAM（1.2 §4），block 级 barrier 是 SM 内的硬件屏障（3.2 §4），跨 SM 两者都失去物理载体。每 block 至多 1024 线程（PG）。grid 是全部 block 的集合，block 之间相互独立：无执行顺序保证、无直接通信手段（§5；跨 block 同步的例外见 §6 cooperative groups 与 3.2 atomic）。

## 2. 索引：从 threadIdx 到全局 tid

线程怎么知道自己该处理第几个元素？靠四个内置变量：threadIdx（block 内坐标）、blockIdx（grid 内 block 坐标）、blockDim（block 尺寸）、gridDim（grid 尺寸），都是 dim3 型（x/y/z 三分量，1D/2D/3D launch 对应启用几个分量）。block 内线性编号按 x 最快、z 最慢展开（PG 逐字公式）：

$$
t = x + y \cdot D_x + z \cdot D_x \cdot D_y \qquad (1)
$$

**符号**： $t$ = block 内线性线程号； $x, y, z$ = threadIdx 三分量； $D_x, D_y, D_z$ = blockDim 三分量。1D 情形退化为 $t = x$。全局线程号（1D）是 block 线性号乘 block 尺寸加 block 内号（PG 例）：

$$
t_{\mathrm{global}} = b_x \cdot D_x + t_x \qquad (2)
$$

**符号**： $t_{\mathrm{global}}$ = 全局线程号； $b_x$ = blockIdx.x； $t_x$ = threadIdx.x； $D_x$ = blockDim.x。全数字代入： $b_x = 100$、 $t_x = 37$、 $D_x = 256$ → $t_{\mathrm{global}} = 100 \times 256 + 37 = 25637$，即该线程处理向量第 25637 个元素。2D 代入：block 尺寸 (16, 16)、线程 (5, 3) → 式 (1) 给 $t = 5 + 3 \times 16 = 53$；grid 尺寸 (4, 4)、block (2, 1) → block 线性号 $2 + 1 \times 4 = 6$，全局号 $6 \times 256 + 53 = 1589$。

**命题 1**（索引双射）。1D 配置下，映射 $(b_x, t_x) \mapsto b_x D_x + t_x$ 是 $\lbrace 0, \ldots, n_{\mathrm{blk}} - 1 \rbrace \times \lbrace 0, \ldots, D_x - 1 \rbrace$ 到 $\lbrace 0, \ldots, n_{\mathrm{blk}} D_x - 1 \rbrace$ 的双射。**证明**（带余除法）：任一整数 $t_{\mathrm{global}}$ 除以 $D_x$ 的商与余数唯一（带余除法），商即 $b_x$、余数即 $t_x$，故映射既单又满。∎ 正例：25637 = 100×256 + 37 唯一分解。边界反例：双射的值域是**线程空间**不是元素空间—— $n_{\mathrm{vec}}$ 不是 $D_x$ 倍数时，末尾 block 有线程的 tid 超出向量长度，必须边界掩码，否则越界访存。

教学片段（边界掩码 + grid-stride loop）：

```cpp
// 教学片段，非优化实现：1D elementwise + grid-stride loop
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   // 式 (2)
    int stride = gridDim.x * blockDim.x;               // 全 grid 线程数
    for (int i = tid; i < n; i += stride)              // 边界由循环条件保证
        c[i] = a[i] + b[i];
}
```

grid-stride loop 让每个线程处理多个元素：launch 配置与问题规模解耦，block 数可以按 SM 数与 occupancy 调（3.4），而不是被 $n_{\mathrm{vec}}$ 绑死。全数字对比：本章口径 65536 × 256 = 16777216 恰等于 $n_{\mathrm{vec}}$，每线程 1 个元素、循环 1 次；若只 launch 1056 blocks（§5 的驻留容量），每线程处理 $\lceil 16777216 / 270336 \rceil = 63$ 个元素——总工作量不变，波次从 62 压到 1，代价是每线程串行迭代变长。两种配置的取舍归 3.4 / 11.2。

## 3. SIMT vs SIMD：独立 PC 与 divergence 语义

同样是「一条指令驱动多路数据」，SIMT 与 CPU 的 SIMD 差在哪？一句话：SIMD 把向量宽度暴露给软件，SIMT 把宽度藏在硬件里——你写标量代码，硬件把 32 个线程拼成一条 warp 指令。PG 的对照逐字：「SIMT instructions specify the execution and branching behavior of a single thread」，而「SIMD vector organizations expose the SIMD width to the software」（[NVIDIA, CUDA Programming Guide](../references.md) SIMT Execution Model 节，[C]）。SIMD 程序显式操作向量 lane，分支要软件自己管 mask；SIMT 程序按单线程写分支，硬件管分歧与汇合。

所谓 SIMT（单指令多线程），是指以 warp 为单位发射一条公共指令、32 个线程各自维护执行状态的执行模型。Volta 起每线程有独立程序计数器与调用栈，PG 逐字「the GPU maintains execution state per thread, including a program counter and call stack」，且「A schedule optimizer determines how to group active threads from the same warp together into SIMT units」，「threads can now diverge and reconverge at sub-warp granularity」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。Volta 之前全 warp 共享一个 PC，分歧只能整 warp 锁步；Volta 之后调度优化器把活跃线程重新拼成 SIMT 单元，分歧与汇合可以发生在 sub-warp 粒度。独立 PC 改变的是**正确性边界**不是免费午餐：分歧路径依旧串行执行（§4），且旧式 warp-synchronous 代码（隐式假设 32 线程永远锁步、不加显式同步）在 Volta+ 不再安全，必须补 `__syncwarp`（3.2 §4）。

> **坑**：Volta+ 上沿用 warp-synchronous 旧代码。触发条件：kernel 假设 warp 内 32 线程天然锁步，跨线程交换数据不加 `__syncwarp`（典型如旧式 warp 内 reduction 直接读邻居写的 shared）；现象：独立 PC 允许线程进度错开，读到未写入的旧值，结果偶发错误且随架构/编译器漂移；规避：warp 内协作一律显式 `__syncwarp(mask)` 或用带 `_sync` 后缀的原语（语义归 3.2，实现归 5.1）。

divergence（分支分歧）的官方语义，PG 逐字（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）：

> A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.

拆成三条论断：其一，32 线程走同一路径时满效率；其二，数据依赖分支使 warp 分歧时，**每条被走到的路径各执行一遍**，不在该路径上的线程被 disable（mask 掉）；其三，分歧只发生在 warp 内部，warp 之间无论走相同还是不同路径都独立执行——warp 间「分歧」不付代价。PTX 侧对应定义（[NVIDIA, PTX ISA](../references.md) Divergence of Threads 节，[C]）：「If threads execute down different control flow paths, the threads are called divergent. If all of the threads act in unison and follow a single control flow path, the threads are called uniform.」且「All control constructs are assumed to be divergent points unless the control-flow instruction is marked as uniform, using the .uni suffix. For divergent control flow, the optimizing code generator automatically determines points of re-convergence.」——除非显式标 `.uni`，一切控制构造默认按分歧点处理；汇合点由编译器决定，程序员不能假设「if/else 一出就汇合」。

分歧时序图（if/else 两侧，warp 内 m 个线程条件为真）：

```
if (cond) { 路径 A } else { 路径 B }      cond：m 真 / 32−m 假
发射槽 →  [路径 A：m lane 活跃，32−m 被 mask]
          [路径 B：32−m lane 活跃，m 被 mask]
          └──────── 两侧串行，共 2 个发射槽 ────────┘
uniform（m = 32 或 m = 0）：只执行被走到的那一侧，1 个发射槽
```

正例：分支条件只依赖 blockIdx 或 tid/32（warp 内 32 线程同值）→ warp-uniform，零分歧。反例：条件依赖 threadIdx.x 的奇偶 → warp 内 16/16 分裂，每个 warp 都分歧。数据分布决定分歧频率的定量账见 §4。

## 4. divergence 代价账：有效吞吐 = 有用 lane-指令占比

divergence 到底亏多少？本节推导分三步：先数发射槽（每条非空路径占整 warp 的槽），再数有用 lane-指令（每线程只走自己那条路径），两者相除得有效吞吐；关键一步是「槽数与路径上的人数无关，只与路径是否非空有关」——这推出一个反直觉结论：1 对 31 分裂与 16 对 16 分裂，等代价时一样贵。

**口径注**：有效吞吐 $\eta$ = 有用 lane-指令数 / 发射 lane-指令数（指令槽口径，量的是计算管线被 mask 浪费的比例）；不是 FLOPs/峰值（那是 MFU 口径，2.4），也不是带宽利用率（那是 coalescing 口径，4.1）。三种口径各管一类损失，勿混。

设分支把 warp 的 32 个线程划成 $k$ 条非空路径，路径 $j$ 上有 $m_j$ 个线程、含 $c_j$ 条 warp 指令， $\sum_{j} m_j = 32$。

**命题 2**（divergence 代价账）。该分支的 warp 发射槽数为 $\sum_{j=1}^{k} c_j$（各路径串行）；有效吞吐为

$$
\eta = \frac{\sum_{j=1}^{k} m_j \cdot c_j}{32 \cdot \sum_{j=1}^{k} c_j} \qquad (3)
$$

**符号**： $k$ = 非空分支路径数； $m_j$ = 路径 $j$ 的线程数； $c_j$ = 路径 $j$ 的指令数； $\eta$ = 有效吞吐（口径见上）。当各路径等代价（ $c_j = c$）时，执行时间为 uniform 情形的 $k$ 倍， $\eta = 1/k$，**与 $m_j$ 分布无关**；分布只通过两个渠道影响代价——路径是否空（ $m_j = 0$ 的路径不执行）与不等代价时的加权。**证明**（要点）：槽数——由 PG「executes each branch path taken, disabling threads that are not on that path」，每条被走到的路径的 $c_j$ 条指令各按整 warp 发射一遍（被 mask 的 lane 仍占槽）， $k$ 条路径串行故总槽数 $\sum c_j$；有用——每线程只执行自己路径的指令，有用 lane-指令合计 $\sum m_j c_j$，发射 lane-指令为 $32 \sum c_j$，相除得式 (3)；等代价—— $\sum c_j = kc$、 $\sum m_j c_j = 32c$，故 $\eta = 32c/(32kc) = 1/k$。完整证明（含嵌套分支的深度归纳与空路径边界讨论）留作练习 3。∎

全数字案例表（本章口径； $c_j$ 以 warp 指令计）：

| 案例 | 分支形态 | $m_j$ 分布 | $c_j$ | 发射槽 | $\eta$ |
| --- | --- | --- | --- | --- | --- |
| D1 | tid 奇偶（tid % 2） | 16 / 16 | 1 / 1 | 2 | 1/2 |
| D2 | tid % 32 == 0 | 1 / 31 | 1 / 1 | 2 | 1/2 |
| D3 | (tid/32) 奇偶（warp-uniform） | 32 / 0 | 1 / — | 1 | 1 |
| D4 | 重路径 10 条 / 轻路径 2 条 | 1 / 31 | 10 / 2 | 12 | 0.1875 |
| D4' | 同 D4 | 31 / 1 | 10 / 2 | 12 | 0.8125 |
| D5 | 两层嵌套二分（四路全非空） | 8 / 8 / 8 / 8 | 1 / 1 / 1 / 1 | 4 | 1/4 |

逐案核对：D1 两侧各 16 人、各 1 条指令，槽数 2、有用 32、发射 64， $\eta = 32/64 = 1/2$。D2 只有 1 个线程走另一侧，但两侧都非空、照样串行， $\eta$ 仍是 1/2——**「只分歧 1 个线程所以代价小」是错误直觉**，这是命题 2「与分布无关」的直接兑现。D3 条件在 warp 内同值，空路径不执行，槽数 1、 $\eta = 1$：把分支粒度从 thread 级提到 warp 级，分歧代价清零。D4 代入式 (3)： $(1 \times 10 + 31 \times 2)/(32 \times 12) = 72/384 = 0.1875$；D4' 换分布： $(31 \times 10 + 1 \times 2)/384 = 312/384 = 0.8125$——不等代价时分布敏感度回来了，两者差 4.33 倍，重路径上的人越多越划算。D5 嵌套两层二分产生 4 条非空路径， $\eta = 1/4$；一般地，深度 $d$ 的嵌套二分至多 $2^{d}$ 条路径，等代价全非空时 $\eta = 1/2^{d}$（练习 3(c) 归纳证明）。

数据依赖分支的分歧频率由数据分布定。正例（教学假设：元素符号 i.i.d.）：ReLU 形分支按元素正负走两侧，正元素占比 $p$，则一个 warp 全正的概率 $p^{32}$、全负 $(1-p)^{32}$，其余分歧。期望发射槽（以单侧代价 $c$ 为单位）为 $2 - p^{32} - (1-p)^{32}$。代入 $p = 0.5$： $p^{32} = (1-p)^{32} \approx 2.33 \times 10^{-10}$，期望槽 $\approx 2$， $\eta \approx 0.5$——几乎每个 warp 都分歧。代入 $p = 0.99$： $0.99^{32} \approx 0.7250$，期望槽 $2 - 0.7250 = 1.275$， $\eta = 1/1.275 \approx 0.784$——七成 warp 全正走 uniform 快路径。数据分布直接决定分歧代价，这是「同一 kernel 换个输入吞吐就变」的机制侧原因之一。**口径注**：此处 $\eta \approx 0.784$ 是 kernel 聚合口径（总有用 lane-指令 / 总发射 lane-指令，即 $1/\mathrm{E}[\text{slot}]$）；与逐 warp $\eta$ 的期望 $\mathrm{E}[\eta] = 0.7250 \times 1 + 0.2750 \times (1/2) = 0.8625$ 不同——比值的期望不等于期望的比值，吞吐账取聚合口径。

divergence 账与 coalescing 账是两笔独立的罚单：前者折计算槽（本节），后者折带宽利用率（4.1）；kernel 对哪笔敏感取决于 bound 类型（2.4 roofline 判定）。compute-bound kernel 的 $\eta$ 直接折峰值吞吐；memory-bound kernel 的分歧可能被访存延迟藏住（1.4 延迟隐藏），两笔账不叠加相乘。

> **坑**：把 warp-uniform 的分支写成 thread 级条件。触发条件：分支条件依赖 threadIdx.x 的低位（如奇偶、模 4），相邻线程交替走两侧；现象：每个 warp 都分歧，等代价两侧时吞吐恒折半（D1/D2），且与数据无关、无法靠输入分布改善；规避：重排数据或分支粒度，让条件在 warp 内同值（如每 warp 处理连续 32 元素、分支依赖 tid/32），或把分支提到循环外。

## 5. block 调度到 SM：GigaThread 术语与波次量化

65536 个 block、同时只驻留 1056 个——谁决定哪个 block 上哪个 SM？PG 给机制描述但不给名字：「When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity.」且「As thread blocks terminate, new blocks are launched on the vacated multiprocessors.」（[NVIDIA, CUDA Programming Guide](../references.md) Hardware Implementation 节，[C]）。拆三条：block 被枚举后分发到**有空闲执行容量**的 SM；block 内线程在单个 SM 上并发执行；block 终止腾出资源，新 block 补位。调度器不保证任何顺序，PG 逐字「The details of which thread blocks are scheduled to execute on which SMs cannot be controlled or queried by the application and no ordering guarantees are made by the scheduler」——所以不能假设 block 0 先跑、不能按 blockIdx 推断 SM 编号、不能在同一 kernel 内做 block 间生产者-消费者（除非 §6 cooperative groups 或 3.2 的 atomic flag 模式）。

这个 chip 级分发机制的历史名字是 GigaThread。Fermi whitepaper 逐字「The GigaThread global scheduler distributes thread blocks to SM thread schedulers.」，并描述两级分布式调度：「At the chip level, a global work distribution engine schedules thread blocks to various SMs, while at the SM level, each warp scheduler distributes warps of 32 threads to its execution units.」（[NVIDIA, Fermi whitepaper](../references.md)，[C]）。**术语时代注**：GigaThread 一词在 Programming Guide 代表版本 8.0 / 9.0 / 11.8 / 13.0 / 13.5 全文均不出现（2026-11-25 代表版本逐版核实），现行官方文档只无名描述「blocks enumerated and distributed」；本书以「block 调度器」指称该机制，GigaThread 仅作术语出处收录——旧书与代码注释里的 GigaThread 指的就是这个 chip 级 block 分发 + SM 级 warp 调度的两级结构，机制未变、名字退役。

调度示意图（block → SM 的分配仅为示意，无顺序保证）：

```
grid：65536 blocks
   │  block 调度器（chip 级，按空闲容量分发）
   ▼
SM 0 [8 blocks]   SM 1 [8 blocks]   ...   SM 131 [8 blocks]
   └────────── 驻留 1056 blocks = 8448 warps ──────────┘
   │  block 终止 → 腾出 slot/register/shared → 下一 block 补位
   ▼
波次 1（1056）→ 波次 2 → ... → 波次 62 → 尾部 64 blocks
```

block 不可跨 SM（§1 PG 逐字）的另一面是资源约束：block 驻留期间独占 block slot / warp slot / register / shared 四类资源（recap 1.2 §1，不重推），装不下至少一个 block 时 kernel 直接 launch 失败——PG 逐字「If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]），1.2 案例 C5（255 registers/thread × 1024 threads）即此情形。

波次量化账（本章口径，全数字）：驻留容量 132 × 8 = 1056 blocks，65536 / 1056 = 62.06 → 62 个满波次 + 尾部 64 blocks（65536 − 62×1056 = 64）。尾部波次只有 64 / 1056 ≈ 6.1% 的驻留容量在干活，其余 SM 空转等尾——grid 尺寸不是驻留容量整数倍时的**尾波量化损失**。两条缓解路：grid-stride loop 把 block 数压到驻留容量（§2，每线程 63 次迭代），或把 grid 尺寸凑成驻留容量的倍数（配置取舍归 3.4 / 11.2）。warp 视角同账：524288 warps ÷ 8448 驻留 warps = 62.06，两口径一致。

> **提点**：逻辑总量与物理驻留的分离，是「编程模型承诺无限线程、硬件交付有限槽位」的机制——grid 尺寸是程序员的策略，波次调度是硬件的机制，两者互不绑定。删掉这条类比，上文的波次账与命题 3 依然完整。

## 6. cooperative groups：显式线程组与 grid sync

想让全部 block 一起到达一个同步点再往下走，普通 launch 做不到——grid 级同步需要 cooperative launch 与共驻留保证。所谓协作组（cooperative groups），是指 CUDA 9 引入的显式线程组抽象：一个组 = 一组线程 + 组级同步操作（[NVIDIA, CUDA Programming Guide](../references.md) Cooperative Groups 节，[C]）。三个常用层级：`this_thread_block()` 取当前 block 组；`tiled_partition` 把 block 切成固定尺寸 tile（32 线程 tile 即 warp）；`this_grid()` 取全 grid 组。

教学片段（grid sync 两阶段 kernel）：

```cpp
// 教学片段，非优化实现：cooperative groups grid sync
// 须用 cudaLaunchCooperativeKernel 启动，block 数不得超过共驻留容量
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void two_phase(float* data) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    phase1(data);          // 各 block 处理自己的 tile
    grid.sync();           // 全 grid 到达后才放行
    phase2(data);          // 跨 block 依赖的计算
    warp.sync();           // warp 级同步（__syncwarp 语义，3.2）
}
```

cooperative launch 的共驻留要求是 PG 逐字「it is necessary to use, instead of the … execution configuration syntax, the cudaLaunchCooperativeKernel CUDA runtime launch API」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]），且 grid 的 block 数须不超过可同时驻留数（用 occupancy API 算，归 3.4）。为什么必须共驻留？

**命题 3**（共驻留必要性）。若 grid 的 block 数超过可同时驻留数，`grid.sync()` 死锁。**证明**（反证）：设某 block 未驻留。由 §5 调度语义，它只能在某个驻留 block 终止后补位 launch；而所有驻留 block 都停在 `grid.sync()` 等**全部** block 到达，谁也不能终止。未驻留者等驻留者终止，驻留者等未驻留者到达——循环等待，死锁。∎ 正例：本章口径 1056 blocks 恰等于驻留容量，可 cooperative launch；反例：65536 blocks 超容量 62 倍，launch 即失败（API 在 launch 时检查，不会跑起来再死锁）。

两个边界须钉住。其一，`coalesced_threads()` 是机会主义组：PG 逐字「returns the set of active threads at that point in time, and makes no guarantee...」（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——它返回当下活跃线程集合、不保证成员稳定，只能作优化提示，不能作正确性依赖的同步单位。其二，Hopper 的 thread block cluster 是另一条跨 block 协作路：cluster 内 block 可互访 shared（DSMEM），不要求全 grid 共驻留，机制与 kernel 用法归 4.4，本章不展开。

warp 原语预告：warp 内寄存器级交换（`__shfl_down_sync` 等 shuffle 族）与投票（`__ballot_sync` 等 vote 族）让 32 线程不经过 shared 直接交换数据——32 线程的 warp 内归约用 shuffle 走 $\log_2 32 = 5$ 步蝶形，免 shared 读写与 bank conflict 风险；树形归约/扫描的完整实现与带宽最优性证明归 5.1，`_sync` 后缀的 mask 与内存排序语义归 3.2 §4。

## 7. 边界：本章不做什么

代价账算完，剩下的问题归谁？本章交付层级语义、索引、SIMT/divergence 语义与代价账、调度机制、cooperative groups 语义；以下各有专章，此处只留指针。occupancy 的编程模型侧精确计算（分配粒度、occupancy API、launch bounds）归 3.4，本章 §5 的驻留容量是 1.2 式 (3) 的理想口径实例化；内存语义（地址空间、fence、atomic、同步原语）归 3.2；stream/event/graph 与 launch 开销归 3.3；访存合并归 4.1（divergence 折计算槽、coalescing 折带宽，两笔账独立）；bank conflict 归 4.2；shuffle/vote 的归约/扫描实现归 5.1；cluster/DSMEM 与异步拷贝归 4.4。代码 scope：机制片段级（§2 elementwise、§6 cooperative groups），不做模块走读。

本章交付清单（下游 recap 接口）：四级层级表与 warp/block 定义（§1）、索引式 (1)(2) + 命题 1（§2）、SIMT vs SIMD 与 Volta 独立 PC（§3）、divergence 代价账式 (3) + 命题 2 + D1-D5 案例（§4）、block 调度语义与波次量化（§5）、命题 3 共驻留（§6）。3.2 recap §1 block/warp 语义作同步 scope 的物理载体；3.4 recap §5 驻留容量作 occupancy 计算起点；5.1 recap §4 代价账作 warp 级归约的动机。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 四级层级：grid → block → warp → thread；block 是驻留单位（不可跨 SM、≤1024 线程），warp 是发射单位（32 线程） | §1 层级表 + PG 逐字 | §1、练习 1 |
| $`w_b = \lceil \mathrm{threads}_{\mathrm{block}}/32 \rceil`$；257 线程 → 9 warps、末 warp 仅 1 活跃 lane | §1 定义与反例 | §1、练习 1 |
| block 内线性号 $t = x + y D_x + z D_x D_y$；1D 全局 tid 双射（带余除法） | 式 (1)(2)，命题 1 | §2、练习 1 |
| SIMT 写标量代码、硬件拼 warp；Volta+ 每线程独立 PC，warp-synchronous 旧代码须补 `__syncwarp` | §3 PG 逐字 + 坑注 | §3 |
| divergence 只在 warp 内；每条被走到的路径串行整 warp 发射，mask lane 仍占槽 | §3 PG/PTX 逐字 | §3、练习 2 |
| 有效吞吐 $\eta = \sum m_j c_j / (32 \sum c_j)$；等代价 k 非空路径 $\eta = 1/k$，与人数分布无关 | 式 (3)，命题 2 | §4、练习 2/3 |
| 1/31 分裂与 16/16 分裂等代价同价（D1/D2 均 1/2）；不等代价时分布敏感（D4 0.1875 对 D4' 0.8125） | §4 案例表 | §4、练习 2 |
| 嵌套深度 d 二分至多 $2^{d}$ 路径；数据依赖分支分歧频率由分布定（p=0.99 时 η≈0.784） | §4 D5 + 概率例 | §4、练习 4 |
| block 调度无顺序保证；GigaThread 为 Fermi 时代术语，现行 PG 无名描述 | §5 PG/Fermi 逐字 | §5 |
| 波次量化：65536 blocks ÷ 1056 驻留 = 62 满波 + 64 尾 block（6.1% 容量） | §5 波次账 | §5、练习 4 |
| grid.sync 须共驻留，否则循环等待死锁；coalesced_threads 无成员保证 | 命题 3 | §6 |

## 练习

1.（计算 · [10]）索引与 warp 几何（本章口径）：(a) 用式 (2) 算 $b_x = 100$、 $t_x = 37$、 $D_x = 256$ 的全局 tid；(b) 256 与 257 threads/block 各自的 $w_b$，257 时最后一个 warp 的活跃线程数；(c) block 尺寸 (16, 16)、线程 (5, 3) 的 block 内线性号。校验：(a) 与 §2 全数字实例一致；(b) 8 与 9、末 warp 1 个活跃线程（§1 反例）；(c) 与式 (1) 代入一致。
2.（计算 · [20]）divergence 账（式 (3)）：(a) D1（16/16、各 1 条指令）与 D2（1/31、各 1 条）的发射槽与 $\eta$——先预测 D2 是否比 D1 便宜再算；(b) D4（1/31、代价 10/2）与 D4'（31/1）的 $\eta$；(c) warp-uniform 分支（D3）的 $\eta$。校验：(a) 两者均 1/2，与命题 2「等代价与分布无关」一致；(b) 0.1875 与 0.8125，与 §4 案例表一致；(c) 1。
3.（推导 · [30]）补全命题 2 的证明：(a) 由 PG「executes each branch path taken」论证 k 条非空路径的发射槽数为 $\sum c_j$，且 $m_j = 0$ 的路径不执行（空路径边界）；(b) 推出等代价时 $\eta = 1/k$ 与 $m_j$ 分布无关；(c) 用归纳三段（基例/保持/终止）证明深度 d 的嵌套二分至多 $2^{d}$ 条非空路径，代入 d=2 对账 D5；(d) 用式 (3) 说明不等代价时分布敏感度（D4 对 D4' 差 4.33 倍）。校验：(b) 与命题 2 论断一致；(c) d=2 给 4 路径、 $\eta = 1/4$ 与 D5 一致；(d) 0.8125/0.1875 ≈ 4.33。
4.（计算 · [30]）概率分歧与调度波次综合账：(a) ReLU 形分支、正元素占比 p=0.99（i.i.d. 教学假设），算期望发射槽（单侧代价为单位）与 $\eta$；(b) p=0.5 重复；(c) 65536 blocks、驻留 1056，算满波次数与尾波驻留容量利用率；(d) 改 grid-stride 配置 1056 blocks，每线程迭代几次？先预测 (a) 的 $\eta$ 高于还是低于 0.5 再算。校验：(a) 期望槽 ≈1.275、 $\eta \approx 0.784$，与 §4 概率例一致；(b) ≈2 与 0.5；(c) 62 满波 + 64 尾 block ≈6.1%；(d) 63 次（ $\lceil 16777216/270336 \rceil$）。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——13.5 子页 02-basics/writing-cuda-kernels.html §2.3.2 Thread Hierarchy（tid = x + y·Dx + z·Dx·Dy、block ≤1024 同 SM 逐字、blockIdx/blockDim/gridDim）+ §2.3.7（scheduler 无 ordering 保证逐字）；03-advanced/advanced-kernel-programming.html §3.2.2.1 SIMT Execution Model（warp 公共指令、divergence 各路径串行 disabling、warp 间独立逐字；SIMT vs SIMD 逐字；Volta Independent Thread Scheduling per-thread PC/call stack + schedule optimizer + sub-warp reconverge 逐字）；Hardware Implementation（blocks enumerated and distributed + 终止补位逐字）+ launch 失败逐字；Cooperative Groups 子页（grid.sync + cudaLaunchCooperativeKernel 共驻留 + tiled_partition + coalesced_threads 无保证逐字）；archive/13.0.0 互证
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——§9.5 Divergence of Threads in Control Constructs（divergent/uniform 定义、.uni suffix、optimizing code generator 决定 reconvergence 点逐字）；shfl.sync/vote.sync 指令族（§6 预告，实现归 5.1）
- [NVIDIA, Fermi whitepaper](../references.md)：NVIDIA Fermi Compute Architecture Whitepaper（key: fermi-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——GigaThread 逐字「The GigaThread global scheduler distributes thread blocks to SM thread schedulers」+ two-level distributed thread scheduler（chip 级 block 分发 / SM 级 warp 调度）；术语时代注：PG 代表版本（8.0/9.0/11.8/13.0/13.5）均无此词（§5）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/0_Introduction/simpleCooperativeGroups 与 cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG（cooperative groups 教学例，§6；现行 cpp/ 路径，旧 Samples/ 已重组）
