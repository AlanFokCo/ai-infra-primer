# 3.3 Stream / Event / Graph 异步（Streams, Events, and CUDA Graphs）

> 拷 192 MiB 数据约 3.15 ms，算它的 kernel 只要 60 μs——重叠能省多少，上界由什么定？重放 330 个 kernel，CUDA Graph 把 CPU 提交开销抹掉多少、一次性成本要几次重放才摊平？本章交付 stream 顺序语义与默认 stream 陷阱、event 计时与跨 stream 依赖、copy-compute 三资源流水账（命题 1 makespan 式）、CUDA Graph 三阶段与 launch 摊销（命题 2 盈亏平衡）、hardware work queue 假依赖、stream 优先级与抢占边界 | 前置：3.1（block 调度无顺序保证、波次量化、running example）、3.2（stream-ordered 内存池、page-locked、PCIe 64 GB/s 口径）、1.4（延迟隐藏 52 warp load）、2.4（roofline 与 PCIe 带宽表）；下游 3.4（occupancy）、4.4（kernel 内 tile 级重叠）、5.4（fusion 与图捕获）、11.2（Nsight Systems timeline）；inference 3.4（full vs piecewise 图捕获 recap，纯文字指引） | 锚点：CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C]）、CUDA Graphs blog（[NVIDIA, CUDA Graphs blog](../references.md)，[C] 开放技术博客、非闭源，V100/CUDA 10.1 实测时代注）、cuda-samples（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、PCI-SIG PCIe（[PCI-SIG, PCIe](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 传输 3.15 ms、计算 60 μs：异步要回答的三笔账

3.1 的 elementwise kernel 算 16777216 个 fp32 元素只要 60 μs，但把 a、b 两个输入向量拷进 GPU、把结果 c 拷回来，PCIe 上纯拷贝要走约 3.15 ms（192 MiB ÷ 64 GB/s；加上 kernel 的串行端到端是 3.2 ms，§4 分解）——计算只占端到端的 1.9%。数据不动，算力再强也空转。本章回答三个问题：stream 与 event 怎么把「拷贝」和「计算」排成重叠的时间线，重叠的收益上界怎么算？CUDA Graph 把几百次 kernel 提交变成一次重放，省下的 CPU 开销与一次性建图成本怎么摊销？多 stream 的优先级与抢占，硬件到底承诺了什么、没承诺什么？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 3.1/3.2）；running example 沿用 3.1 的 elementwise 向量 kernel（ $n_{\mathrm{vec}} = 16777216$ 个 fp32 元素，a/b/c 三个 buffer 各 67108864 B = 64 MiB， $R_{\mathrm{th}} = 32$、 $S_{\mathrm{sh}} = 0$、256 threads/block， $\mathcal{O} = 1.0$ 沿 3.1 章脊假设）；互连带宽取 **PCIe Gen5 x16 理论单向 64 GB/s**（沿 2.4 §4 表与 3.2 §6 口径，[PCI-SIG, PCIe](../references.md)，[C]；GB/s 按 $10^{9}$ B/s，有效带宽低于理论值，测量方法归 2.4）；HBM 带宽 3352 GB/s（[NVIDIA, Hopper whitepaper](../references.md)，[C]）。§5 的 launch 开销数值换 **V100 / CUDA 10.1 实测口径**（[NVIDIA, CUDA Graphs blog](../references.md)，[C] 开放博客），原因：官方 Programming Guide 对 launch 开销只定性（「CPU launch costs are reduced」），~μs 数值的唯一官方实测出处是该 blog——绝对值随硬件世代漂移，量级口径可用，随文注。

符号声明（spec §4.3）： $n_{\mathrm{kernel}}$ = 一次工作流（或一步 decode）的 kernel 数（recap inference 3.4 的 $n_{\mathrm{ker}}$ 译名，同一对象）； $n_{\mathrm{chunk}}$ = 数据切分的 chunk 数； $n_{\mathrm{rep}}$ = graph 重放次数； $t_{\mathrm{launch}}$ = 每 kernel 的 CPU 提交开销； $t_{\mathrm{graph}}$ = 单次 graph launch 开销； $t_{\mathrm{inst}}$ = 建图 + 实例化一次性成本； $m_{\mathrm{in}}, m_{\mathrm{out}}, m_{\mathrm{comp}}$ = 每 chunk 的 H2D 字节、D2H 字节、kernel HBM 流量字节（本章局部；整 buffer 传输字节沿全书 $M$ 语义）； $t_{\mathrm{H2D}}, t_{\mathrm{D2H}}, t_{\mathrm{comp}}$ = 每 chunk 三阶段时间； $\tau_{\mathrm{seq}}, \tau_{\mathrm{pipe}}$ = 串行与流水总时间。裸字母 n 专归 rank 数、N 专归参数量、B 专归 batch，本章均不占用；时间一律 $t/\tau$ 带下标，不用裸 T（序列长专用）。

## 2. stream：顺序队列与默认 stream 陷阱

两个 kernel 提交到 GPU，谁先跑、能不能同时跑？答案取决于它们进了哪条 stream。所谓 stream，是指一条按提交顺序执行的 GPU 命令队列（kernel launch、memcpy、memset 等操作入队），Programming Guide 逐字「CUDA streams are what are known as in-order streams. This means that the order of execution of the operations in a stream is the same as the order in which those operations were enqueued.」（[NVIDIA, CUDA Programming Guide](../references.md) §2.5.5，[C]）。拆成两条语义：**stream 内有序**（后一个操作不前越）、**stream 间无序**（不同 stream 的操作可交错、可并发，除非有显式依赖或隐式同步）。正例：同一 stream 里「H2D 拷贝 → kernel」，kernel 必等拷贝完成，无需显式同步（PG：「Memory operations (such as copies) are tracked by the runtime and will always complete before the next operation ...」）。反例：两条 stream 各跑一个 kernel，执行顺序与完成顺序都无保证——要顺序就得显式建依赖（§3 event）。

并发 kernel 的硬件基础是 block 调度器按空闲容量分发（recap 3.1 §5，不重推）：两个 grid 的 block 可以同驻一个 SM，只要资源装得下（`concurrentKernels` 设备属性）。所以「stream 间可并发」不是一句空话，它落到 SM 的 block slot / register / shared 库存上（1.2）——两个 kernel 并发时各自能驻留的 block 数互相挤占，这是 §4 干扰折扣的来源之一。

默认 stream 是最常见的并发杀手。CUDA 的 legacy default stream（NULL stream，stream ID 0）是 blocking stream：不指定 stream 的 kernel launch 与 blocking `cudaMemcpy` 都进它。PG 逐字「When an operation is launched into this default stream, it will synchronize with all other blocking streams, in other words it will wait for all other blocking streams to complete before it can execute.」（§2.5.6.1）。隐式同步的完整规则逐字：「Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them, unless the streams are non-blocking streams (created with the cudaStreamNonBlocking flag).」（§2.5.8）。两条出路：其一，`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)` 建非阻塞 stream，与 NULL stream 互不同步；其二，编译选项 `--default-stream per-thread`（或宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM`）让每个 host 线程有独立的 regular default stream，不再共享 legacy NULL stream（PG §2.5.6.2，CUDA 7 起）。

> **坑**：两条非默认 stream 之间夹了一个默认 stream 操作。触发条件：`kernel1<<<...,stream1>>>` 之后写了不带 stream 参数的 `cudaMemcpy` 或 kernel，再 `kernel3<<<...,stream2>>>`；现象：kernel3 被迫等 kernel1 完成，三 kernel 串行，多 stream 白写；规避：全部操作显式指定非阻塞 stream，或开 per-thread default stream；隐式同步只认 NULL stream 与 blocking stream 的组合（PG 逐字）。

stream 是编程对象，hardware work queue（工作队列）是它的物理载体——两者不是一一对应。PG 环境变量表逐字：CUDA_DEVICE_MAX_CONNECTIONS「Sets the number of compute and copy engine concurrent connections (work queues) from the host to each device of compute capability 3.5 and above.」，取值 1 到 32、默认 8（archive/13.0.0 表）；CC 8.0+ 另有 CUDA_DEVICE_MAX_COPY_CONNECTIONS 按 async copy engine 设拷贝队列数。stream 数超过 work queue 数时，多条 stream 共享同一队列，队列内顺序语义把本可并行的操作串行化——PG 逐字「If independent GPU work tasks (e.g., kernels submitted on different CUDA streams) map to the same workqueue, a false dependence between these tasks may be introduced, which can lead to their serialized execution.」（§4.7），并建议「increasing the value of the CUDA_DEVICE_MAX_CONNECTIONS environment variable may be necessary to reduce the possibility that independent work from different CUDA streams gets serialized due to false dependencies」（§3.1.6）。所谓假依赖（false dependency），是指操作间本无数据依赖、仅因共享队列而被强加的顺序约束。

```
host 线程                编程对象              物理载体（默认 8 条 work queue）
cudaStreamCreate ──► stream 1 ──┐
cudaStreamCreate ──► stream 2 ──┼──► work queue 0 ──► SM（kernel）/ copy engine（memcpy）
cudaStreamCreate ──► stream 3 ──┘        ▲
...                                    共享队列 → 假依赖 → 串行
stream 9（超出 8 条）──► 与某 stream 共 queue
```

> **坑**：把「stream 数」当「并发度」。触发条件：开 16 条 stream 跑 16 个独立小 kernel，默认 8 条 work queue；现象：部分 kernel 因共享队列串行，timeline 上看不到预期重叠，且与 NULL stream 无关；规避：先排除隐式同步与数据依赖，再试调 CUDA_DEVICE_MAX_CONNECTIONS（PG 建议从默认值出发、只在出现无法归因的串行时再调）。

## 3. event：GPU 时间戳与跨 stream 依赖

怎么知道一个 kernel 在 GPU 上跑了多久、怎么让 stream B 等 stream A 的某个节点？两个问题同一个原语：event。所谓 event，是指插入 stream 的完成标记——PG 逐字「When an event reaches the front of a stream it records a timestamp.」（§2.5.3.3）。`cudaEventRecord(event, stream)` 把 event 排进 stream，它前面的操作全部完成时 event 触发并记 GPU 时间戳。

计时用法（教学片段，非优化实现）：

```cpp
// 教学片段：event 计时（PG §2.5.3.3 官方例同形）
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);
cudaStreamSynchronize(stream);          // 等两个 event 都触发
float ms;
cudaEventElapsedTime(&ms, start, stop); // GPU 时间戳差，单位 ms
```

**口径注**：`cudaEventElapsedTime` 量的是两个 GPU 时间戳之差——kernel 在 GPU 上的执行时间（含它在队列里的等待），**不含** CPU 提交开销；§5 的 launch 开销账用的是 CPU wallclock 端到端口径（blog 测法），两者量的是不同段，勿混。依赖用法逐字：「cudaStreamWaitEvent() takes a stream and an event as parameters ... and makes all the commands added to the given stream after the call to cudaStreamWaitEvent() delay their execution until the given event has completed.」（PG §2.5.7）。所以跨 stream 依赖的标准写法是：A stream 先 record event，B stream 再 wait 该 event，B 中后续操作全部推迟到 event 完成：

```cpp
// 教学片段：fork/join 跨 stream 依赖（PG §4.2.2.1.2.1 官方例同形）
cudaEventRecord(e1, streamA);      // A 的节点 1 完成标记
cudaStreamWaitEvent(streamB, e1);  // B 等 A 的节点 1
kernel_B<<<..., streamB>>>(...);   // 与 A 的后续节点并发
cudaEventRecord(e2, streamB);
cudaStreamWaitEvent(streamA, e2);  // A 的后续节点等 B（join）
```

两个边界须钉住。其一，纯依赖用途的 event 建议关计时：PG 逐字「CUDA events carry, by default, timing information」，「it is recommended to create events with timing information disabled for improved performance. This is possible using cudaEventCreateWithFlags() API with the cudaEventDisableTiming flag.」（§3.1.3）——计时信息有成本，只表达依赖时不需要。其二，wait 一个从未 record 的 event 是空操作：PG 逐字「a non-recorded event will always return success」——依赖边静默缺失，不报错。

> **坑**：`cudaStreamWaitEvent` 写在 `cudaEventRecord` 之前（或忘了 record）。触发条件：跨 stream 依赖的 record/wait 顺序颠倒，或 event 创建后未 record 就 wait；现象：wait 立即返回（non-recorded event always success），下游 kernel 读到未就绪数据，结果偶发错误；规避：record 必须先于 wait 提交，依赖链用 §5 的 capture 规则校验（cross-stream 依赖须在同一 capture graph 内）。

event 还是 graph capture 的依赖边载体：capture 模式下 record 的 event 成为 captured event，wait 它的 stream 被拉进同一张 capture graph（§5）。

## 4. copy-compute overlap：三资源流水账

拷贝和计算凭什么能同时跑？因为它们用的是不同的硬件。所谓拷贝引擎（copy engine），是指独立于 SM 的 DMA 搬运单元，H2D（host-to-device，主机→GPU）与 D2H（device-to-host，GPU→主机）拷贝由它执行，不占 SM 的执行管线。PG 逐字「Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the asyncEngineCount device property ... which is greater than zero for devices that support it. If host memory is involved in the copy, it must be page-locked.」（archive/13.0.0 §6.2.8.3）；双向能力逐字「Some devices of compute capability 2.x and higher can overlap copies to and from the device ... which is equal to 2 for devices that support it.」（§6.2.8.4）——`asyncEngineCount` = 2 时 H2D 与 D2H 各用一个引擎，加上 SM，恰是三个可并行的资源。带宽侧的对应事实：PCIe 全双工，Gen5 x16 两个方向各 64 GB/s 理论带宽（§1 口径），H2D 与 D2H 同时跑各占一个方向、互不分薄——双 copy engine 正是把这条全双工能力用起来的硬件前提。Best Practices 把单/双引擎的差别说得更直白（逐字）：「GPUs with a single copy engine can perform one asynchronous data transfer and execute kernels whereas GPUs with two copy engines can simultaneously perform one asynchronous data transfer from the host to the device, one asynchronous data transfer from the device to the host, and execute kernels.」（[NVIDIA, Best Practices](../references.md) §10.1.2，[C]）。

**定义**（overlap 三条件）。拷贝与 kernel 重叠须同时满足：(i) 设备 `asyncEngineCount` 大于 0；(ii) host 侧 buffer page-locked（pinned）；(iii) 拷贝与 kernel 在**不同的非默认 stream**。BP 逐字：「the overlap once again requires pinned host memory, and, in addition, the data transfer and kernel must use different, non-default streams (streams with non-zero stream IDs).」正例：`cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, stream1)` 与 `kernel<<<..., stream2>>>` 并发（BP 官方例）。边界反例：pageable host memory 的 `cudaMemcpyAsync` 退化为同步行为（PG：「it will revert to a synchronous behavior」），重叠静默失效；blocking `cudaMemcpy` 走默认 stream，与任何操作都串行（BP 逐字「it is not possible to overlap a blocking transfer with an asynchronous transfer」）。

整块拷贝无法重叠——kernel 要等全部输入就绪。机制是切 chunk：数据分 $n_{\mathrm{chunk}}$ 份，每份「H2D → kernel → D2H」走一条 stream，stream 内顺序保证依赖，stream 间不同 chunk 的三个阶段错开重叠（BP「Staged concurrent copy and execute」官方模式）：

```cpp
// 教学片段，非优化实现：chunk 化 overlap（BP §10.1.2 模式，双向版）
for (int i = 0; i < n_chunk; i++) {
    int off = i * chunk_elems;
    cudaMemcpyAsync(d_a + off, h_a + off, bytes, cudaMemcpyHostToDevice, s[i]);
    cudaMemcpyAsync(d_b + off, h_b + off, bytes, cudaMemcpyHostToDevice, s[i]);
    kernel<<<chunk_blocks, 256, 0, s[i]>>>(d_a + off, d_b + off, d_c + off);
    cudaMemcpyAsync(h_c + off, d_c + off, bytes, cudaMemcpyDeviceToHost, s[i]);
}
```

时间线对比（本章口径， $n_{\mathrm{chunk}} = 8$ ，每 chunk 三阶段时间见下）：

```
串行（整块或逐 chunk 串行）:
  copy引擎 │ H2D(全部) │            │ D2H(全部) │
  SM       │           │ kernel(全部)│           │
  时间 ────────────────────────────────────────► 3.206 ms
三资源流水（双 copy engine，chunk 错开）:
  H2D引擎 │c1│c2│c3│c4│c5│c6│c7│c8│
  SM      │  │k1│k2│k3│k4│k5│k6│k7│k8│
  D2H引擎 │  │  │d1│d2│d3│d4│d5│d6│d7│d8│
  时间 ──────────────────────────────► 2.236 ms
  c/k/d = chunk 的 H2D / kernel / D2H；稳态由最慢阶段（H2D）封顶
```

推导路线分三步：先把每 chunk 三阶段时间写成带宽账，再写串行总时间，最后用瓶颈资源的忙碌时间给流水 makespan——关键一步是「瓶颈阶段串行处理全部 chunk，首尾 chunk 各付一遍上下游串行链」。每 chunk 时间（ $m$ 为字节、 $\mathrm{BW}$ 为带宽）：

$$
t_{\mathrm{H2D}} = \frac{m_{\mathrm{in}}}{\mathrm{BW}_{\mathrm{PCIe}}}, \qquad t_{\mathrm{D2H}} = \frac{m_{\mathrm{out}}}{\mathrm{BW}_{\mathrm{PCIe}}}, \qquad t_{\mathrm{comp}} = \frac{m_{\mathrm{comp}}}{\mathrm{BW}_{\mathrm{HBM}}} \qquad (1)
$$

**符号**： $m_{\mathrm{in}}, m_{\mathrm{out}}$ = 每 chunk H2D / D2H 字节； $m_{\mathrm{comp}}$ = 每 chunk kernel 的 HBM 流量字节； $`\mathrm{BW}_{\mathrm{PCIe}} = 64`$ GB/s（理论单向）； $`\mathrm{BW}_{\mathrm{HBM}} = 3352`$ GB/s。全数字代入（ $n_{\mathrm{chunk}} = 8$ ，每 chunk 2097152 个元素）： $m_{\mathrm{in}} = 2 \times 8388608 = 16777216$ B（a+b 两个输入）， $m_{\mathrm{out}} = 8388608$ B（c）， $m_{\mathrm{comp}} = 3 \times 8388608 = 25165824$ B（读 a、b 写 c）； $t_{\mathrm{H2D}} = 16777216/(64 \times 10^{9}) = 262.144$ μs， $t_{\mathrm{D2H}} = 131.072$ μs， $t_{\mathrm{comp}} = 25165824/(3352 \times 10^{9}) = 7.508$ μs。**口径注**： $t_{\mathrm{comp}}$ 取 HBM 带宽下界（roofline 口径，2.4），前提是延迟藏住——本 kernel $\mathcal{O} = 1.0$ 、64 warps × ILP 1 = 64 ≥ 52（recap 1.4 式 (4)，不重推），满足。

串行总时间（逐 chunk 三阶段顺序执行，或整块一次执行，同值）：

$$
\tau_{\mathrm{seq}} = n_{\mathrm{chunk}} \left( t_{\mathrm{H2D}} + t_{\mathrm{comp}} + t_{\mathrm{D2H}} \right) \qquad (2)
$$

代入：每 chunk 三阶段和 $\Sigma = 262.144 + 7.508 + 131.072 = 400.724$ μs， $\tau_{\mathrm{seq}} = 8 \times 400.724 = 3205.79$ μs ≈ 3.206 ms。整块口径对账：H2D 128 MiB → 2097.152 μs，kernel 192 MiB HBM 流量 → 60.062 μs，D2H 64 MiB → 1048.576 μs，合计 3205.79 μs，逐项对上。

**命题 1**（三资源流水 makespan）。三阶段各由独立资源执行（H2D 引擎 / SM / D2H 引擎）、chunk 间无依赖、缓冲无限的理想口径下，流水总时间为

$$
\tau_{\mathrm{pipe}} = \Sigma + (n_{\mathrm{chunk}} - 1) \cdot \max\left( t_{\mathrm{H2D}},\ t_{\mathrm{comp}},\ t_{\mathrm{D2H}} \right) \qquad (3)
$$

其中 $\Sigma = t_{\mathrm{H2D}} + t_{\mathrm{comp}} + t_{\mathrm{D2H}}$ 。**证明**（路线：下界 + 上界，两步）：下界——设瓶颈阶段时间 $t_{\max}$ ，它串行处理 $n_{\mathrm{chunk}}$ 个 chunk（单资源），首个 chunk 须先串行走完上游阶段（ $\Sigma_{\mathrm{up}}$ ），末个 chunk 完成后还须串行走完下游阶段（ $\Sigma_{\mathrm{down}}$ ），故 $\tau \ge \Sigma_{\mathrm{up}} + n_{\mathrm{chunk}} t_{\max} + \Sigma_{\mathrm{down}} = \Sigma + (n_{\mathrm{chunk}} - 1) t_{\max}$ ；上界——贪心调度（各阶段就绪即按 chunk 序处理）使瓶颈阶段自首个 chunk 后不再空闲，因为其余两阶段时间均不超过 $t_{\max}$ ，供给与排空都不慢于瓶颈（对 chunk 序号归纳）。两界夹逼得式 (3)。完整证明（含归纳保持步与等时退化）留作练习 2。∎

全数字代入：瓶颈是 H2D（262.144 μs）， $\tau_{\mathrm{pipe}} = 400.724 + 7 \times 262.144 = 2235.73$ μs ≈ 2.236 ms。节省与加速比：

$$
\Delta\tau = (n_{\mathrm{chunk}} - 1) \left( \Sigma - \max \right) \qquad (4)
$$

**符号**： $\Delta\tau$ = 节省时间。加速比 $\tau_{\mathrm{seq}}/\tau_{\mathrm{pipe}}$ 随 $n_{\mathrm{chunk}}$ 增大趋于 $\Sigma/\max$，上界 3（三阶段， $\Sigma \le 3 \max$ ）。代入： $\Delta\tau = 7 \times (400.724 - 262.144) = 970.06$ μs ≈ 0.970 ms，占串行的 30.3%；加速比 $3205.79/2235.73 = 1.434$ ；渐近上界 $\Sigma/\max = 400.724/262.144 = 1.529$ ——**不是 3**。诚实的读数：elementwise 过 PCIe 是传输主导（H2D 一项占 $\Sigma$ 的 65%），重叠只能把另外两项藏进 H2D 背后，加速比上界 1.53；只有三阶段时间均衡时才逼近 3（练习 4）。chunk 数灵敏度： $n_{\mathrm{chunk}} = 4$ 加速 1.350、 $n_{\mathrm{chunk}} = 16$ 加速 1.480，向 1.529 收敛——切得越细省得越多，但收益有顶。

单 copy engine 的对照账（`asyncEngineCount` = 1）：H2D 与 D2H 共享一个引擎，拷贝总忙碌时间 $n_{\mathrm{chunk}} (t_{\mathrm{H2D}} + t_{\mathrm{D2H}}) = 8 \times 393.216 = 3145.73$ μs 是 makespan 下界，故节省至多 $\tau_{\mathrm{seq}} - 3145.73 = 60.06$ μs（1.9%）——恰好等于全部 kernel 时间，即最好情况也只是把计算藏进拷贝。双引擎把节省从 60 μs 抬到 970 μs，差 16 倍：`asyncEngineCount` 这个设备属性值直接决定 overlap 的收益量级。

与官方估算对账：BP 给的两阶段（拷贝 vs 计算）粗估逐字「when the execution time (tE) exceeds the transfer time (tT), a rough estimate for the overall time is tE + tT/nStreams for the staged version versus tE + tT for the sequential version. If the transfer time exceeds the execution time, a rough estimate for the overall time is tT + tE/nStreams.」——式 (3) 令 $t_{\mathrm{D2H}} = 0$ 即退化为该估算（练习 2(d) 验证），本式是它向三资源的推广。

> **坑**：把式 (3) 当实测承诺。触发条件：忽略干扰与 CPU 侧成本，直接按理想 makespan 预估收益；现象：实测加速低于 1.434——并发 kernel 与拷贝共享 HBM 带宽和 L2（本例 kernel + 拷贝合计 HBM 流量 384 MiB → 120.12 μs，远小于 PCIe 3146 μs，干扰可忽略；但 compute 重的 workload 不可忽略），且 chunk 化把 launch 次数从 4（整块 2 次 H2D + 1 kernel + 1 次 D2H）抬到 $4 n_{\mathrm{chunk}}$ ，CPU 提交成本随 $n_{\mathrm{chunk}}$ 线性涨；规避：式 (3) 是 GPU 侧时间下界（口径：理想无干扰、缓冲无限、不计 launch 开销），CPU 侧成本用 §5 的 graph 摊销对付，实测偏差用 Nsight Systems timeline 归因（11.2）。

## 5. CUDA Graph：把几百次 launch 摊成一次

先 recap 引擎侧的账（inference 3.4 §3，纯文字指引，不重推）：8B decode 每步约 330 个 kernel，每 kernel CPU 提交开销教学值 5 μs，launch 项 $n_{\mathrm{kernel}} \cdot t_{\mathrm{launch}} = 330 \times 5$ μs = 1.65 ms——batch 1 步时 4.81 ms（权重-only 口径）里占 34.3%，batch 64 的 15.1 ms 里占 10.9%。CUDA Graph 把整段前向录成一张图整体重放，省 ≈ 1.645 ms/step；静态三要求（指针固定、shape 与 grid 固定、kernel 序列固定），内容动态合法（replay 读 buffer 当前值）。full graph 把整个前向捕进一张图，要求 attention backend 支持静态 shape；decode attention 的 KV 长度与 block table 每步在变，故 piecewise 在 attention 算子处切分——非 attention 段各自捕获成子图，attention 在两次 replay 之间 eager 跑（动态 shape 排除）。引擎配置面（vLLM CompilationMode / CUDAGraphMode、splitting_ops）与 piecewise 正确性命题归 inference 3.4，本节不重复。**本章增量**：GPU 侧机制—— $t_{\mathrm{launch}}$ 到底花在哪、graph 的三阶段路径怎么把它省掉、一次性成本怎么摊销（符号翻译注：inference 3.4 的 $n_{\mathrm{ker}}$ 即本章 $n_{\mathrm{kernel}}$ ）。

每 kernel 的 launch 开销花在哪？PG 逐字「when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU. These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time.」（§4.2）。即 driver 为每次提交做一套准备工作（工作描述符设置与入队），kernel 越短，这笔固定开销占比越大——这就是 §1 那笔 34.3% 的机制来源。

CUDA Graph 的解法是把「定义」与「执行」分离。PG 逐字「A graph is a series of operations such as kernel launches, data movement, etc., connected by dependencies. A graph is defined separately from its execution. This allows a graph to be defined once and then launched repeatedly.」，收益两条逐字「first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams.」三阶段路径：

```
definition（定义）        instantiation（实例化）         execution（执行）
stream capture 或 graph API ──► cudaGraphInstantiate ──► cudaGraphLaunch × n_rep
记录节点 + 依赖边              预建全部 kernel 工作描述符     一次提交整张 DAG
（不执行）                    一次性成本 t_inst            每次成本 t_graph
```

capture（捕获）机制逐字：「When a stream is being captured, work launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built up.」（PG §4.2.2.1.2）——`cudaStreamBeginCapture(stream)` 与 `cudaStreamEndCapture(stream, &graph)` 之间，kernel launch 只录不跑。instantiate（实例化）把图变成可执行对象，blog 逐字「cudaGraphInstantiate ... creates and pre-initialises all the kernel work descriptors so that they can be launched repeatedly as rapidly as possible」（[NVIDIA, CUDA Graphs blog](../references.md)，[C]）——每 kernel 的准备工作在实例化时一次做完，replay（重放）时不再付。教学片段（三阶段全链，PG 官方例同形）：

```cpp
// 教学片段，非优化实现：capture → instantiate → replay
cudaStreamBeginCapture(stream);
kernel_A<<<..., stream>>>(...);
kernel_B<<<..., stream>>>(...);
cudaStreamEndCapture(stream, &graph);
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, 0);   // 一次性 t_inst
for (int i = 0; i < n_rep; i++)
    cudaGraphLaunch(graphExec, stream);       // 每次 t_graph
```

cross-stream 依赖可以入图，但有 rejoin 规则：capture 中 record 的 event 是 captured event，wait 它的 stream 被拉进同一张 capture graph；PG 逐字「cudaStreamEndCapture() must still be called in the same stream where cudaStreamBeginCapture() was called; this is the origin stream」，且「Failure to rejoin to the origin stream will result in failure of the overall capture operation.」（§4.2.2.1.2.1）——fork 出去的 stream 必须用 event join 回 origin stream，否则整个 capture 失败。

摊销账。stream 路径每次执行的 CPU 提交成本（recap inference 3.4 式 (2)，符号译名）：

$$
\tau_{\mathrm{CPU,stream}} = n_{\mathrm{kernel}} \cdot t_{\mathrm{launch}} \qquad (5)
$$

代入（inference 3.4 recap 口径）： $n_{\mathrm{kernel}} = 330$、 $t_{\mathrm{launch}} = 5$ μs → 1.65 ms/step。graph 路径把一次性成本 $t_{\mathrm{inst}}$ 摊到 $n_{\mathrm{rep}}$ 次重放，每次执行的摊销成本：

$$
\tau_{\mathrm{CPU,graph}} = t_{\mathrm{graph}} + \frac{t_{\mathrm{inst}}}{n_{\mathrm{rep}}} \qquad (6)
$$

**符号**： $t_{\mathrm{graph}}$ = 单次 graph launch 开销（与单 kernel launch 同量级）； $t_{\mathrm{inst}}$ = 建图 + 实例化一次性成本。**口径注**：式 (6) 是 CPU 提交侧摊销口径；GPU 执行时间 $\tau_{\mathrm{GPU}}$ 不被 graph 改变（inference 3.4 式 (3) 口径），端到端收益在 CPU-bound 区（kernel 短、launch 藏不住）最确定。代入（blog 单 kernel 图，数值出处见下）： $t_{\mathrm{inst}} = 400$ μs、 $n_{\mathrm{rep}} = 20000$ → 摊销项 0.02 μs。盈亏平衡：

**命题 2**（graph 盈亏平衡）。当 $n_{\mathrm{kernel}} \cdot t_{\mathrm{launch}} > t_{\mathrm{graph}}$ 时，graph 摊销成本低于 stream 当且仅当

$$
n_{\mathrm{rep}} > n_{\mathrm{rep}}^{\ast} = \frac{t_{\mathrm{inst}}}{n_{\mathrm{kernel}} \cdot t_{\mathrm{launch}} - t_{\mathrm{graph}}} \qquad (7)
$$

**证明**（代数重排，一步）：式 (6) 低于式 (5) 即 $t_{\mathrm{graph}} + t_{\mathrm{inst}}/n_{\mathrm{rep}} \lt n_{\mathrm{kernel}} t_{\mathrm{launch}}$ ，移项除正数 $n_{\mathrm{kernel}} t_{\mathrm{launch}} - t_{\mathrm{graph}}$ 即得。∎ 边界反例：若每步只 launch 1 个长 kernel（ $t_{\mathrm{launch}}$ 已被 GPU 执行藏住），分母趋零或为负，graph 无收益——它省的是暴露的提交开销，不是 GPU 时间。

全数字代入（blog 实测，**V100 / CUDA 10.1 时代注**：绝对值随硬件世代漂移，~μs 量级为官方口径，PG 只定性）：kernel 执行 2.9 μs（数组 500000 个元素、512 threads/block）；每 kernel 后同步的 sync launch 9.6 μs；异步 stream 3.8 μs；graph 3.4 μs。暴露开销三档：sync 9.6 − 2.9 = 6.7 μs（每 kernel 一次 CPU-GPU 往返，流水全断），async 3.8 − 2.9 = 0.9 μs（launch 与上一 kernel 执行部分重叠），graph 3.4 − 2.9 = 0.5 μs。async → graph 每 kernel 省 $\Delta = 0.4$ μs；create + instantiate 约 400 μs 一次性，代入式 (7)： $n_{\mathrm{rep}}^{\ast} = 400/0.4 = 1000$ 次重放才摊平。blog 用 20000 次重放： $400/20000 = 0.02$ μs/kernel，与 blog 逐字「only contributes around 0.02μs to our per-kernel cost」对上。首次 graph launch 比后续慢约 33%（blog 实测）——机制是内存映射在首次 launch 才做：PG 逐字「Physical memory cannot be allocated or mapped during graph instantiation because the stream in which the graph will execute is unknown. Mapping is done instead during graph launch. Calling cudaGraphUpload can separate out the cost of allocation from the launch」（§4.2.5.4.1），`cudaGraphUpload` 把映射成本从首次 launch 挪到预热阶段。多 kernel 图的账更陡：inference 口径 $n_{\mathrm{kernel}} = 330$ 、 $t_{\mathrm{launch}} = 5$ μs（引擎栈教学值，含框架分发；blog 0.9 μs 是 V100 纯 driver 层实测，两口径栈深不同、量级同为 ~μs），每步省 1.645 ms，对 400 μs 量级的一次性成本（**口径注**：400 μs 是单节点图实测，多节点图实例化成本随节点数涨，官方未给逐节点成本），盈亏平衡远小于 1 次重放——decode 这种每步重复的工作流，graph 从第一步就赚。

图不是一成不变的。workflow 变了怎么办？PG 逐字「Major changes to graph structure (such as topology or node types) require re-instantiation because topology-related optimizations must be re-applied.」，而「CUDA provides a lightweight "Graph Update" mechanism that allows certain node parameters to be modified in-place without rebuilding the entire graph, which is much more efficient than re-instantiation.」（§4.2.3）——拓扑不变、只改 kernel 参数或地址时，`cudaGraphExecUpdate`（整图更新，要求拓扑与依赖序一致）或逐节点更新即可，免掉重新实例化；更新在下次 launch 生效，不影响正在跑的 launch（PG 逐字「Updates take effect the next time the graph is launched」）。这正是 inference 3.4 piecewise 之外另一条对付「参数每步变」的路：shape 固定的算子用 graph update 改参数，shape 变的算子仍须切分排除。

条件节点（conditional node）把控制流也搬进图内：PG 逐字「Conditional nodes allow conditional execution and looping of a graph contained within the conditional node.」，「Evaluation of the condition value is performed on the device when the dependencies of the conditional node have been met.」（§4.2.4）——IF / WHILE / SWITCH 三型，条件值由 device 代码经 `cudaGraphSetConditional` 设置，body graph 可嵌套。意义：迭代与分支不再需要 host 往返决策，CPU 彻底解放（「frees up the host CPU to perform other work in parallel」）。device graph launch（`cudaStreamGraphFireAndForget` / `cudaStreamGraphTailLaunch` 等模式）进一步允许从 device 侧发起 graph launch，免 host 往返，机制细节归 PG §4.2.6，本章只立语义。

> **坑**：replay 读的是捕获时的指针与 launch 参数（recap inference 3.4 坑注，机制侧根据在此）。触发条件：capture 后换了输入 tensor 地址、或 grid 依赖的量每步在变；现象：replay 按旧参数执行，输出不随输入变或越界；规避：输入 copy 进捕获 buffer（内容动态合法），参数变化走 graph update，shape 变化走 piecewise 切分（inference 3.4）——三者的边界都是「capture 录的是参数值，不是求值过程」。

## 6. 多 stream 优先级与抢占：hint，不是 guarantee

高优先级 stream 的 kernel 能插队吗？能影响**待启动**工作的选择，不能打断**正在跑**的工作。PG 逐字「these priorities serve as hints rather than guarantees. When selecting work to launch, pending tasks in higher-priority streams take precedence over those in lower-priority streams. Higher-priority tasks do not preempt already running lower-priority tasks. The GPU does not reassess work queues during task execution, and increasing a stream's priority will not interrupt ongoing work.」（§3.1.3.1）。用法：`cudaStreamCreateWithPriority` 建带优先级的 stream，`cudaDeviceGetStreamPriorityRange` 查范围，数值越小优先级越高（PG §2.5.9.1 逐字「lower numbers correspond to higher priorities」），默认优先级 0。所以优先级是调度选择 hint：延迟敏感的短 kernel 放高优先级 stream，能在资源腾出时先被选中，但若低优先级 kernel 已占满 SM，高优先级者照样等——要硬隔离得用 green context 的 SM 静态分区（PG §4.7，本章不展开）。

抢占是另一层机制，别与 stream 优先级混。Compute Preemption（Pascal+，CC 6.0+）逐字「This allows compute tasks to be preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture」，收益是长 kernel 不再垄断系统或触发超时，代价是「there will be context switch overheads associated with Compute Preemption」（PG archive/13.0.0）；能力查询用 `cudaDevAttrComputePreemptionSupported`。两层机制的边界：stream 优先级作用于**同一 context 内待启动工作的选择**，不抢占运行中任务；Compute Preemption 是 **context 级**切换能力（指令级粒度），伴随 context switch 开销，不是 stream 优先级触发的 per-kernel 抢占。

## 7. 边界：本章不做什么

三笔账算完，剩下的归谁？本章交付 stream 顺序语义与默认 stream 陷阱、event 计时与依赖、copy engine 与 overlap 三条件、三资源流水 makespan（命题 1）、graph 三阶段与摊销（命题 2）、work queue 假依赖、优先级与抢占边界；以下各有专章。kernel **内部**的 tile 级 copy-compute 重叠（cp.async / TMA / multi-stage pipeline）归 4.4——本章是 stream 级（kernel 之间），4.4 是 kernel 内（tile 之间），两层重叠正交可叠加。occupancy 与资源约束归 3.4（并发 kernel 互相挤占驻留容量的精确账）。引擎侧图捕获配置面与 piecewise 正确性归 inference 3.4，fusion 与图捕获的算子层协同归 5.4。overlap 与 launch 的实测 timeline 归因（Nsight Systems）归 11.2；PCIe 有效带宽测量归 2.4。代码 scope：机制片段级（§3 计时与 fork/join、§4 chunk 化、§5 capture 三阶段），不做模块走读。

本章交付清单（下游 recap 接口）：stream 双语义与隐式同步规则（§2）、event 时间戳口径与 wait 语义（§3）、overlap 三条件 + 式 (1)-(4) + 命题 1（§4）、graph 三阶段 + 式 (5)-(7) + 命题 2 + blog ~μs 数值（§5）、优先级 hint 边界（§6）。4.4 recap §4 作 stream 级 vs kernel 内级重叠的对照起点；3.4 recap §2 并发挤占作 occupancy 计算动机之一；11.2 recap §4 干扰口径作 timeline 归因起点。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| stream 内有序、stream 间无序；NULL stream 隐式同步阻断并发，除非 cudaStreamNonBlocking 或 per-thread default stream | §2 PG 逐字 | §2、练习 1 |
| stream 映射到有限 work queue（CUDA_DEVICE_MAX_CONNECTIONS 默认 8、上限 32）；共享队列引入假依赖致串行 | §2 PG 逐字 + 坑注 | §2 |
| event = GPU 时间戳；cudaEventElapsedTime 量 GPU 段不含 CPU 提交；cudaStreamWaitEvent 建跨 stream 依赖；non-recorded event wait 恒成功 | §3 PG 逐字 | §3、练习 1 |
| overlap 三条件：asyncEngineCount 大于 0 + page-locked + 不同非默认 stream；双引擎 = H2D + D2H + SM 三资源 | §4 定义 + BP 逐字 | §4、练习 4 |
| 三资源流水 makespan = Σ + (n_chunk − 1)·max；节省 (n_chunk − 1)(Σ − max)；加速上界 Σ/max ≤ 3 | 式 (3)(4)，命题 1 | §4、练习 2 |
| 章脊实例：τ_seq 3.206 ms → τ_pipe 2.236 ms，省 970 μs（30.3%），加速 1.434，上界 1.529（传输主导，非 3） | §4 全数字 | §4、练习 1/4 |
| 单 copy engine 节省至多 60.06 μs（1.9%）= 全部 kernel 时间；双引擎抬到 970 μs，差 16 倍 | §4 对照账 | §4、练习 4 |
| graph 三阶段 definition / instantiation / execution；capture 只录不跑；instantiate 预建工作描述符；cross-stream 须 rejoin origin | §5 PG/blog 逐字 | §5 |
| launch 摊销：stream n_kernel·t_launch 对 graph t_graph + t_inst/n_rep；盈亏平衡 n_rep* = t_inst/(n_kernel·t_launch − t_graph) | 式 (5)(6)(7)，命题 2 | §5、练习 3 |
| blog 实测（V100/CUDA 10.1 时代注）：2.9/9.6/3.8/3.4 μs；instantiate 400 μs；break-even 1000 次；20000 次摊 0.02 μs；首 launch 慢 33%（映射在 launch，cudaGraphUpload 可前移） | §5 全数字 | §5、练习 3 |
| graph update 免重新实例化（拓扑须不变）；条件节点 IF/WHILE/SWITCH 在 device 侧求值；replay 读捕获时指针 | §5 PG 逐字 + 坑注 | §5 |
| stream 优先级是 hint：影响待启动选择、不抢占运行中工作；Compute Preemption 是 context 级指令粒度切换，另一层 | §6 PG 逐字 | §6 |

## 练习

1.（计算 · [10]）overlap 带宽账代入（本章口径）：(a) 每 chunk 2097152 个 fp32 元素，用式 (1) 算 $t_{\mathrm{H2D}}$（a+b 两输入）、 $t_{\mathrm{D2H}}$（c）、 $t_{\mathrm{comp}}$（读 a、b 写 c 的 HBM 流量）；(b) 用式 (2) 算 $n_{\mathrm{chunk}} = 8$ 的 $\tau_{\mathrm{seq}}$；(c) event 计时与 CPU wallclock 各量哪一段？校验：(a) 262.144 / 131.072 / 7.508 μs，与 §4 一致；(b) 3.206 ms，与整块口径 2097.152 + 60.062 + 1048.576 μs 对账；(c) 与 §3 口径注一致（GPU 时间戳差对端到端）。
2.（推导 · [20]）补全命题 1 的证明：(a) 下界——论证瓶颈阶段串行处理 $n_{\mathrm{chunk}}$ 个 chunk、首 chunk 付上游串行链、末 chunk 付下游串行链，得 $\tau \ge \Sigma + (n_{\mathrm{chunk}} - 1) t_{\max}$；(b) 上界——对 chunk 序号归纳证明贪心调度下瓶颈阶段自首个 chunk 后不空闲（其余阶段时间均 $\le t_{\max}$）；(c) $n_{\mathrm{chunk}} = 2$ 手画时间线枚举验证式 (3)；(d) 令 $t_{\mathrm{D2H}} = 0$，验证式 (3) 退化为 BP 两阶段估算 tT + tE/nStreams（传输主导情形）。校验：(c) 与式 (3) 代入一致；(d) 与 §4 引 BP 逐字估算一致（特例锚）； $n_{\mathrm{chunk}} = 1$ 时 $\tau_{\mathrm{pipe}} = \tau_{\mathrm{seq}}$（无节省，退化锚）。
3.（计算 · [20]）graph 摊销账（blog V100/CUDA 10.1 口径）：(a) 算 sync / async / graph 三档的暴露开销（各减 kernel 2.9 μs）；(b) 用式 (7) 算 async → graph 的盈亏平衡重放次数（ $t_{\mathrm{inst}} = 400$ μs）；(c) 20000 次重放的摊销成本；(d) recap inference 3.4 口径（ $n_{\mathrm{kernel}} = 330$ 、 $t_{\mathrm{launch}} = 5$ μs）每步省多少。先预测 (b) 的量级（百次对千次对万次）再算。校验：(a) 6.7 / 0.9 / 0.5 μs；(b) 1000 次；(c) 0.02 μs，与 blog 逐字一致（公开数值锚）；(d) 1.645 ms，与 inference 3.4 式 (3) 一致。
4.（计算 · [30]）overlap 综合灵敏度（本章口径）：(a) 先预测 $n_{\mathrm{chunk}}$ 从 4 增到 16 加速比的方向与上界，再用式 (3)(4) 算两档加速比；(b) 单 copy engine 的节省上界（拷贝忙碌时间下界论证）；(c) 三阶段时间均衡（各 100 μs）时的加速上界，与章脊实例 1.529 对照；(d) 说明 chunk 化的 CPU 侧代价（launch 次数）与 §5 graph 的关系。校验：(a) 1.350 与 1.480，上界 1.529，与 §4 一致；(b) 60.06 μs（1.9%）；(c) Σ/max = 3（三阶段上界取等）；(d) launch 次数 4 → 4 n_chunk，与 §4 坑注一致。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——13.5 §2.5.5 stream in-order 逐字、§2.5.6 blocking/non-blocking 与 legacy default stream 逐字、§2.5.6.2 per-thread default stream、§2.5.7 cudaStreamWaitEvent 逐字、§2.5.8 implicit synchronization 逐字、§2.5.3.3 event 时间戳逐字、§2.5.9.1 优先级数值语义、§3.1.3 cudaEventDisableTiming 与 non-recorded event 逐字、§3.1.3.1 stream priorities hints 逐字、§3.1.6 与 §4.7 work queue 假依赖逐字、§4.2 graph 定义/收益逐字 + §4.2.2.1.2 capture 不入队逐字 + §4.2.2.1.2.1 rejoin origin 逐字 + §4.2.2.2/3 instantiate/launch + §4.2.3 graph update 逐字 + §4.2.4 条件节点逐字 + §4.2.5.4.1 cudaGraphUpload 映射逐字 + §4.2.6 device graph launch；archive/13.0.0 §6.2.8.3 asyncEngineCount 大于 0 逐字 + §6.2.8.4 等于 2 双向逐字 + CUDA_DEVICE_MAX_CONNECTIONS 1 到 32 默认 8 + Compute Preemption 指令级逐字
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——§10.1.2 overlap 三条件逐字（pinned + 不同非默认 stream）+ 单/双 copy engine 逐字 + staged concurrent copy and execute 官方模式 + tE + tT/nStreams 两阶段估算逐字 + blocking transfer 不可 overlap 逐字；§10.1.1 pinned memory
- [NVIDIA, CUDA Graphs blog](../references.md)：Getting Started with CUDA Graphs（key: cuda-graphs-blog），[C] 开放技术博客、非闭源，2026-11-25——V100 / CUDA 10.1 实测 kernel 2.9 μs / sync 9.6 μs / async 3.8 μs / graph 3.4 μs + create+instantiate ~400 μs + 20000 次摊销 0.02 μs + 首 launch 慢 33% + cudaGraphInstantiate 预建工作描述符逐字；时代注：绝对值随硬件漂移，~μs 量级口径（§5）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/0_Introduction/asyncAPI（异步拷贝与 overlap）、cpp/3_CUDA_Features/simpleCudaGraphs 与 graphConditionalNodes（graph 与条件节点）、cpp/6_Performance/cudaGraphsPerfScaling（capture/instantiate/launch/upload 全链；现行 cpp/ 路径）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——132 SM、HBM3 3352 GB/s（§4 t_comp 带宽口径）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露·一手权威，2026-11-24——Gen5 x16 = 64 GB/s 单向理论口径（§4 带宽取值，沿 2.4 §4 表与 3.2 §6；精确 URL W4 复核残留）
