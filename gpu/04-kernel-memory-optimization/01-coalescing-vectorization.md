# 4.1 访存合并与向量化（Coalescing and Vectorization）

> 同一个 3.35 TB/s 的 HBM，为什么一个 copy kernel 40 μs 跑完、另一个要 320 μs？——地址模式决定 warp 触及几个 32 B sector，sector 数决定流量，标称带宽只是分母。本章交付 coalescing 判定规则（命题 1：transaction 数 = distinct sector 数，跨度公式双向反例 63 对 32、4 对 5）、stride 饱和律（命题 2：fp32 利用率 $\eta = 1/\min(s, 8)$，不是无界 $1/s$）、向量化加载（ld.global.v4 / float4，128-bit 上限与自然对齐）、AoS vs SoA 布局账、misaligned 的两层代价、Nsight 两代指标（legacy efficiency 对 Sectors/Req）与教学 kernel strideCopy 对 coalesced copy 的带宽对比 | 前置：2.2（line 128 B / sector 32 B、流量 = distinct sector × 32 B 命题 2、 $\eta$ 带宽口径）、2.1（HBM 3352 GB/s 式 (1)）、3.1（warp = 32、running example 向量长）、2.4（带宽账与测量方法）；下游 4.2（shared 侧重排）、4.3（tiling 的 coalesced 装载）、4.4（cp.async 16 B 粒度）、5.2/6.3（GEMM/FA 的 global 侧流量账）、11.2（实测 drill-down） | 锚点：CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）、CUDA Programming Guide（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、Nsight Compute（[NVIDIA, Nsight Compute](../references.md)，[C]）、cuda-samples alignedTypes（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）——官方规格 2026-11-25 一手核实

## 1. 40 μs 对 320 μs：流量由地址模式决定

拷贝 16777216 个 fp32 元素（读 64 MiB + 写 64 MiB = 128 MiB 有用流量），按 HBM 标称 3352 GB/s 算，带宽口径下界 40.04 μs；同一个 kernel 把访问改成 stride-16，时间变 320.33 μs——8 倍。标称带宽没变、FLOPs 没变、元素数没变，变的只有一件事：每条 warp load 触及的 sector 数。本章回答三个问题：一条 warp global 访问的 transaction 数怎么判定，为什么不能按地址跨度算？strided 访问把有用带宽折到几成，折扣在哪里饱和？向量化、数据布局（AoS/SoA）与对齐各改变哪一笔账？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（沿 3.1/3.2）；running example 为向量拷贝 c[i] = a[i]，向量长 $n_{\mathrm{vec}} = 16777216$（ $2^{24}$ ）个 fp32 元素（沿 3.1 的 elementwise 向量口径，add 改 copy：流量只计读 + 写两项），launch 65536 blocks × 256 threads；strided 版本沿 Best Practices 的 strideCopy 教学 kernel（§6），buffer 按 stride 扩到 $s \cdot n_{\mathrm{vec}}$ 元素（教学假设， $s = 16$ 时每侧 1 GiB 地址空间，H100 80 GB 装得下）。 $\mathrm{BW} = 3352$ GB/s（2.1 式 (1)，GB/s 按 $10^{9}$ B/s；MiB 按 $2^{20}$ B）。copy 是流式访问、每 sector 只触及一次、无缓存复用，sector 流量账即 HBM 流量口径（教学口径；缓存命中的延迟账归 2.2，本章不重复）。章内出现的 V100 实测值（790 GB/s、4/5 吞吐）是 BP 官方教学例的时代口径，机制跨代同形、绝对值不代入 H100。

符号声明： $s$ = stride（以 32-bit 字计，fp32 即元素数；**翻译注**：2.2 的 $s$ = distinct sector 数在本章写作 $n_{\mathrm{sec}}$， $s$ 专归 stride，两章符号不混用）； $n_{\mathrm{sec}}$ = 一条 warp 访问触及的 distinct 32 B sector 数； $a_i$ = lane $i$ 的访问字节地址（ $i = 0, \ldots, 31$ ）； $\eta$ = 有用字节占比（**口径注**：沿 2.2 式 (2) 的带宽口径 = 有用字节 ÷ 下层流量；3.1 式 (3) 的 $\eta$ 是指令槽口径，两口径各量一类损失，勿混）； $n_{\mathrm{vec}}$ = 向量元素数（recap 3.1）； $\mathrm{bytes}$ = 每元素字节数（fp32 取 4；bytes 不占字母）； $\mathrm{BW}$ = HBM 标称带宽； $t_{\mathrm{copy}}$ = copy 的带宽口径时间（§6）； $n_{\mathrm{v4}}$ = float4 向量个数（§4）。B/KB/MB/GB/MiB 是单位符号豁免。

**边界声明**：本章交付 global 侧的访问模式账。shared 侧的 bank conflict 归 4.2（global 按 32 B sector 计流量、shared 按 4 B bank 计冲突，两套粒度两条定律，§5 末对照）；tiling 复用账归 4.3；cp.async/TMA 归 4.4；L1/L2 命中延迟账归 2.2；带宽测量方法归 2.4/11.2。代码 scope：机制片段级（§4 float4 copy、§6 strideCopy），不做模块走读。

粒度机制 recap 2.2 §3（不重推）：L1/L2 的 cache line 128 B 是 tag 管理粒度，sector 32 B 是数据搬运粒度，一条 line 含 4 个 sector；miss 按需要的 sector 取数、不搬整条 line；一条 warp load 的下层流量 = 32 B × distinct sector 数（2.2 命题 2）。本节回答 2.2 留下的判定问题：**给定地址模式，distinct sector 数怎么数**。

## 2. coalescing 判定：transaction 数 = distinct sector 数

32 条 lane 各报一个地址，硬件怎么决定搬几个 sector？所谓访存合并（coalescing），是指硬件把 warp 内 32 个线程的 global 访问合并成尽可能少的 memory transaction。CC 6.0+ 的官方规则一句话（[NVIDIA, Best Practices](../references.md) §10.2.1，[C]）：

> the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.

即 transaction 数 = 服务 warp 全部线程所需的 32 B transaction 数——每个被触及的 sector 整体搬一次，数的是 **distinct sector 个数**。访问单位与 L1 缓存是否启用无关：「the data access unit is 32-byte regardless of whether global loads are cached in L1 or not」（BP 逐字）。PG 侧同口径：「device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned」（[NVIDIA, CUDA Programming Guide](../references.md) Device Memory Accesses 节，archive/13.0.0 §8.3.2，[C]），并给最坏例逐字「if a 32-byte memory transaction is generated for each thread's 4-byte access, throughput is divided by 8」——每线程 4 B 各占一个 sector 时吞吐折 8 倍，这正是 §3 饱和律的官方预告。

**命题 1**（coalescing transaction 数）。CC 6.0+ 上，一条 warp global 访问的 transaction 数为

$$
n_{\mathrm{sec}} = \left\vert \lbrace \lfloor a_i / 32 \rfloor : i = 0, \ldots, 31 \rbrace \right\vert \qquad (1)
$$

**符号**： $a_i$ = lane $i$ 的访问字节地址； $n_{\mathrm{sec}}$ = distinct sector 数； $\lfloor \cdot \rfloor$ 向下取整。**证明**（两步）：sector 是最小搬运粒度，每个被触及的 sector 整体搬 32 B 一次（2.2 命题 2 recap），故下层流量 $32 n_{\mathrm{sec}}$ 字节；官方规则把 transaction 数定义为「number of 32-byte transactions necessary to service all of the threads」，一个 transaction 服务一个 sector，故 transaction 数 = $n_{\mathrm{sec}}$。∎

正例（BP 逐字）：「if the threads of a warp access adjacent 4-byte words (e.g., adjacent float values), four coalesced 32-byte transactions will service that memory access」——32 线程 × 4 B = 128 B 连续，恰 4 个 sector， $\eta = 1$。两条不变量把规则的边界钉死（BP 逐字）：「If from any of the four 32-byte segments only a subset of the words are requested ... the full segment is fetched anyway. Furthermore, if accesses by the threads of the warp had been permuted within or across the four segments, still only four 32-byte transactions would have been performed」——**只请求 sector 内部分字，仍搬整个 sector；lane 顺序任意置换，transaction 数不变**。coalescing 关心的是触及 sector 的集合，不是哪个 lane 触及哪个字，也不是 sector 内每个字是否都被用到。

反例两个方向，钉死「为什么不能按跨度算」。设跨度 = 最大地址 − 最小地址 + 访问宽度， naive 公式 $\lceil \text{跨度}/32 \rceil$ 两头都错：

- **高估**（大 stride）：stride-16 fp32（字节 stride 64），lane $i$ 地址 $64i$，跨度 = $64 \times 31 + 4 = 1988$ B， $\lceil 1988/32 \rceil = 63$。但 sector 号 $\lfloor 64i/32 \rfloor = 2i$，32 条 lane 落在编号 0, 2, ..., 62 的 **32 个 distinct sector**——真值 32，跨度公式给 63，高估 31 个 sector（近 2 倍）。机制：跨度把地址之间的空洞也计成流量，而未触及的 sector 根本不搬。
- **低估**（misaligned）：连续 128 B 但基址偏移 4 B，访问区间 [4, 132) 跨编号 0–4 共 5 个 sector，跨度公式 $\lceil 128/32 \rceil = 4$ 漏掉首尾两个残 sector。BP 逐字：「If sequential threads in a warp access memory that is sequential but not aligned with a 32-byte segment, five 32-byte segments will be requested」（§10.2.1.2）。

```
fp32 warp load：32 lane × 4 B，基址 32 B 对齐（数字 = 占据该字的 lane 号）
s=1（连续）：
  sector 0        sector 1        sector 2        sector 3
  |0 1 2 ... 7|   |8 9 ... 15|    |16 ... 23|     |24 ... 31|
  → 4 sector，流量 128 B，η = 1
s=2（隔字）：
  sector 0        sector 1        ...
  |0 . 1 . 2 . 3 .|4 . 5 . 6 . 7 .|...        （. = 搬来但无用的字）
  → 8 sector，流量 256 B，η = 1/2
s=16（64 B stride）：
  lane i → 字节 64i → sector 2i：0, 2, 4, ..., 62（每 lane 独占一个 sector）
  → 32 sector，流量 1024 B，η = 1/8
  跨度 1988 B，ceil(跨度/32) = 63 ≠ 32 —— 跨度公式高估
```

跨度公式只在**对齐基址、stride 不超过 8 字的稠密访问**下恰好成立（§3 证明该情形 $n_{\mathrm{sec}} = 4s = \lceil \text{跨度}/32 \rceil$）——这是「连续访问 128 B = 4 sector」直觉的来源，但直觉不能外推到大 stride 与 misaligned。

> **坑**：按「地址跨度 ÷ 32」或「触及 line 数 × 128 B」估流量。触发条件：给 strided 或 misaligned kernel 做带宽账时套跨度公式；现象：大 stride 高估（stride-16 给 63 对真值 32）、misaligned 低估（4 对 5），优化收益误判；规避：一律按式 (1) 逐 sector 数（2.2 §3 坑注的 coalescing 侧补全），line 只是 tag 粒度、不参与流量账。

## 3. stride 饱和律：利用率 1/min(s, 8)

strided 访问把有用带宽折到几成？BP 给定性方向（§10.2.1.4 逐字）：「A stride of 2 results in a 50% of load/store efficiency since half the elements in the transaction are not used and represent wasted bandwidth. As the stride increases, the effective bandwidth decreases until the point where 32 32-byte segments are loaded for the 32 threads in a warp」——注意终点：**降到 32 个 sector 为止**，不是无限降。本节把这句话变成公式。

推导路线分两步：先对对齐基址的 fp32 stride 访问数 $n_{\mathrm{sec}}$（分 $s \le 8$ 与 $s \ge 8$ 两情形），再代入 $\eta$ = 有用字节 ÷ 流量；关键一步是 $s \ge 8$ 时字节 stride 不小于 32 B、每 lane 独占一个 sector，sector 数被 warp 的 32 条 lane 封顶——**饱和地板来自 lane 数，不来自 stride**。

**命题 2**（stride 饱和律）。基址 32 B 对齐、fp32（ $\mathrm{bytes} = 4$ ）、stride $s$ 字（ $s \ge 1$ 整数）的 warp load，有用字节占比为

$$
\eta = \frac{1}{\min(s, 8)} \qquad (2)
$$

等价地： $n_{\mathrm{sec}} = 4s$（ $1 \le s \le 8$ ）或 $n_{\mathrm{sec}} = 32$（ $s \ge 8$ ）。**证明**（要点）：有用字节恒 $32 \times 4 = 128$ B，流量 $32 n_{\mathrm{sec}}$，故 $\eta = 4/n_{\mathrm{sec}}$。情形 $s \ge 8$：字节 stride $4s \ge 32$，sector 号 $\lfloor 4si/32 \rfloor$ 随 $i$ 严格递增，32 lane 给 32 个 distinct sector， $\eta = 4/32 = 1/8$。情形 $1 \le s \le 8$：sector 号 $\lfloor si/8 \rfloor$ 每步至多增 1（因 $s/8 \le 1$），取遍 0 到 $\lfloor 31s/8 \rfloor$ 的全部整数， $n_{\mathrm{sec}} = \lfloor 31s/8 \rfloor + 1 = 4s$， $\eta = 1/s$。完整证明（含取整恒等式 $\lfloor 31s/8 \rfloor + 1 = 4s$ 与 $s = 8$ 边界两情形衔接）留作练习 3。∎

一般元素宽度形： $\eta = 1/\min(s, 32/\mathrm{bytes})$——饱和点是「字节 stride 达到一个 sector」（ $s \cdot \mathrm{bytes} = 32$ ）。fp64（ $\mathrm{bytes} = 8$ ）在 $s = 4$ 饱和： $s = 2$ 给 $\eta = 1/2$， $s = 4$ 与 $s = 8$ 都给 $1/4$（练习 3(c) 数值验证）。**口径注**：式 (2) 的教学假设是基址 32 B 对齐、stride 按整字、访问不跨 sector 边界；misaligned 基址按式 (1) 逐 sector 数（§5），字节级 gather 同样不适用式 (2)。

全数字表（对齐 fp32 warp load，128 B 有用；有效有用带宽 = $\eta \cdot \mathrm{BW}$，2.2 式 (3) recap）：

| $s$ | $n_{\mathrm{sec}}$ | 流量 | $\eta$ | 有效有用带宽 |
| --- | --- | --- | --- | --- |
| 1 | 4 | 128 B | 1 | 3352 GB/s |
| 2 | 8 | 256 B | 1/2 | 1676 GB/s |
| 4 | 16 | 512 B | 1/4 | 838 GB/s |
| 8 | 32 | 1024 B | 1/8 | 419 GB/s |
| 16 | 32 | 1024 B | 1/8 | 419 GB/s |

代入演示 $s = 2$：lane $i$ 读字 $2i$，sector 号 $\lfloor 2i/8 \rfloor = \lfloor i/4 \rfloor$，取 0–7 共 8 个，每 sector 8 个字里只有 4 个有用， $\eta = 128/256 = 1/2$——与 BP「stride of 2 results in a 50%」逐字对上。 $s = 16$： $\eta = 1/\min(16, 8) = 1/8$，**不是** $1/16$。无界 $1/s$ 公式在 $s \ge 8$ 处错误：它假设流量随 stride 线性涨，但 sector 地板把流量封在 1024 B； $s = 16$ 处两公式差 2 倍（ $1/16$ 对 $1/8$ ），stride 越大差距越大。表末两行同价是饱和律的直接兑现：stride 从 8 加到 16，带宽不再降。

> **坑**：把有用带宽外推成无界 $1/s$。触发条件：给大 stride（ $s \ge 8$ ）的 gather/scatter kernel 估 $\eta = 1/s$；现象： $s = 16$ 估出 209.5 GB/s，实际地板 419 GB/s，低估 2 倍——反过来会高估「把 stride 16 改到 8」的优化收益（两者同价，改了白改）；规避：用式 (2) 饱和形； $s \ge 8$ 区想提带宽只能改访问模式（SoA 布局 §5、shared 重排 4.2），在饱和区内调 stride 无带宽收益。

## 4. 向量化：ld.global.v4 与 float4

一条指令能不能搬更多字节？能——向量化加载。PTX 的 ld 支持 `.v2`/`.v4` 向量限定符，上限逐字（[NVIDIA, PTX ISA](../references.md) §9.7.10.8，[C]）：「Vectors cannot exceed 128-bits in length; for example, .v4 .f64 is not allowed.」；访问大小逐字「the access size of ld.v4.b32 is 16 bytes」。CUDA C++ 侧的 float4 是 4 个 fp32 的 built-in 向量类型，16 B 对齐由类型保证（PG 向量类型表），编译器把 float4 读写映射到 128-bit 访存指令。

对齐在这里是**正确性边界**，不是性能提示。PTX 逐字：「The address must be naturally aligned to a multiple of the access size. If an address is not properly aligned, the resulting behavior is undefined.」——16 B 访问必须 16 B 对齐，违反是未定义行为（可能静默掩掉低位地址、可能 fault）。两层对齐勿混：自然对齐（对齐到访问宽度）管正确性，32 B sector 对齐管性能（§5）。分配侧的保障：cudaMalloc 返回地址至少 256 B 对齐（PG 逐字「Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes.」），float4 数组基址天然满足；危险来自指针偏移与类型双关（§5 坑注）。

向量化省的是什么？**口径注**：省 request 数与指令数，不省 sector 流量——对已 coalesced 的模式，两者流量相等。账（每线程处理 4 个 fp32，warp 有用 512 B）：

| 口径 | 标量 4 B × 4 条 | float4 16 B × 1 条 |
| --- | --- | --- |
| 每线程 load 指令数 | 4 | 1 |
| 每 warp request 数 | 4 | 1 |
| 每 warp sector 数 | 16 | 16 |
| Sectors/Req | 4 | 16 |
| 每 request 字节 | 128 B | 512 B |

request 数的官方口径（[NVIDIA, Nsight Compute](../references.md) ProfilingGuide §2.11.2，[C]）：「Each instruction generates exactly one request for LSU traffic (global, local, …)」——一条指令一个 request，4 条标量 load 就是 4 个 request。Sectors/Req 的最优值逐字：「For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16.」——**4、8、16 都是最优**，随访问宽度变；float4 的 16 不是 uncoalesced。机制收益两笔：其一，发射槽与 LSU 队列压力 ÷4（同样字节更少指令，1.4 延迟隐藏口径下请求队列更不容易 throttled）；其二，每 request 携带 512 B，在飞请求的字节密度高，藏延迟所需的请求条数少。对 memory-bound kernel，收益落在指令侧与延迟隐藏，**不落在 HBM 流量**——流量已在 $\eta = 1$ 地板，向量化压不出更多。

教学片段（float4 版 copy，机制片段级）：

```cpp
// 教学片段，非优化实现：float4 向量化 copy（128-bit/thread）
__global__ void copy_v4(const float4* a, float4* c, int n_v4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_v4) c[i] = a[i];   // 一条 ld.global.v4.f32 + 一条 st.global.v4.f32
}
```

全数字代入（本章口径）： $n_{\mathrm{vec}} = 16777216$ fp32 → $n_{\mathrm{v4}} = 4194304$ 个 float4，launch 16384 blocks × 256 threads；有用流量仍 128 MiB，带宽口径下界仍 40.04 μs——每线程指令从 4 load + 4 store 降到 1 load + 1 store（8 条对 2 条）。边界两条： $n_{\mathrm{vec}}$ 不是 4 的倍数时须标量收尾（教学假设整除）；float4 要求 16 B 对齐，偏移指针先验对齐（§5）。官方教学例对照：cuda-samples `cpp/6_Performance/alignedTypes`（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）演示对齐类型与非对齐类型的访存差，可与本节账对跑。

## 5. AoS vs SoA 与 misaligned 的两层代价

为什么 struct-of-arrays 比 array-of-structs 合并友好？因为 stride 由布局决定。所谓 AoS（array of structures），是指每个对象的各字段连续存成一个 struct、对象排成数组；所谓 SoA（structure of arrays），是指每个字段各存一个数组。教学例（fp32 口径）：4 字段 struct {x, y, z, w}（16 B），warp 跨对象读各对象的 x 字段——AoS 布局字节 stride 16（ $s = 4$ ），式 (2) 给 $\eta = 1/4$、16 个 sector，有效有用带宽 $3352/4 = 838$ GB/s；SoA 布局的 x 数组 stride 1， $\eta = 1$、4 个 sector，3352 GB/s。**4 倍差**，机制与 §3 的 stride-4 行完全相同。反过来，若 kernel 一次消费每个对象的全部 4 字段，AoS 每线程 16 B 连续、等价 float4， $\eta = 1$——AoS 不是天然坏，**坏的是「warp 跨对象取单字段」这一种访问**。判据一句话：访问粒度与布局粒度一致则合并，不一致则改布局（SoA）或改访问（整对象取回再在 register/shared 内重排，重排侧归 4.2/5.1）。

misaligned 的代价分两层，正确性与性能必须分开：

- **自然对齐违反（正确性层）**：4 B float 的地址不是 4 的倍数、16 B float4 不是 16 B 对齐——PG/PTX 口径未定义行为（§4 逐字）。典型触发：带字节偏移的指针 reinterpret_cast 成 float4* 再解引用。
- **32 B sector 对齐缺失（性能层）**：地址合法但 warp 访问区间不对齐 32 B 边界，首尾残 sector 各整体搬。连续 fp32 偏移 4 B：5 个 sector， $\eta = 128/160 = 0.8$。BP 的 V100 教学例逐字（§10.2.1.3）：「global memory accesses with no offset or with offsets that are multiples of 8 words result in four 32-byte transactions. The achieved bandwidth is approximately 790 GB/s. Otherwise, five 32-byte segments are loaded per warp, and we would expect approximately 4/5th of the memory throughput achieved with no offsets.」—— $790 \times 4/5 = 632$ GB/s（V100 时代口径，机制跨代）。float4 版本：基址 16 B 对齐但偏移 16 B（非 32 B 对齐），warp 覆盖 [16, 528)，17 个 sector， $\eta = 512/544 = 16/17 \approx 0.94$——访问越宽，misaligned 的相对代价越小（首尾浪费被更多有用字节摊薄）。

两条官方缓解：分配侧 cudaMalloc 的 256 B 对齐（§4）保证基址干净；2D 数组侧 `cudaMallocPitch` 把行距 pad 到对齐要求（PG 逐字「padded to meet the alignment requirements described in Device Memory Accesses」），每行行首保持对齐——宽度不是 32 B 倍数的 2D 数组不要手排紧凑行距，否则每行的 warp 访问都付 5-sector 账。

> **坑**：把字节偏移指针 reinterpret_cast 成 float4* 做向量化。触发条件：对带 header 或奇数偏移的 buffer 直接 float4 加载；现象：自然对齐违反是未定义行为——可能静默掩掉低位地址读错数据、可能直接 fault，不是「慢一点」；规避：向量化前验偏移是 16 的倍数，否则退回标量或调整布局。性能层的 32 B 对齐缺失（本节第二层）才是「慢但正确」的情形，两层勿混。

两套粒度对照（4.2 预告）：global 侧 stride-16 fp32 饱和于 $\eta = 1/8$（32 sector 地板，式 (2)）；shared 侧 stride-16 字访问却是 16 路 bank conflict（bank 按 4 B 字取模，无 sector 地板，4.2 式 (3)）——同一个 stride 数字，两套粒度两条定律，勿把 global 的饱和直觉搬进 shared。

## 6. 教学 kernel：strideCopy 对 coalesced copy

三种模式实测差多少？BP 官方教学 kernel strideCopy（逐字同形，[NVIDIA, Best Practices](../references.md) §10.2.1.4，[C]）：

```cpp
// BP strideCopy 教学 kernel（官方例同形）：线程 tid 拷贝元素 tid*stride
__global__ void strideCopy(float* odata, const float* idata, int stride) {
    int xid = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    odata[xid] = idata[xid];
}
```

stride = 1 即 coalesced copy；stride 大于 1 时读、写两侧同 stride。带宽账路线分三步：先算每 warp 的 $\eta$（式 (2)），再放大到全 kernel——有用流量固定 128 MiB（ $n_{\mathrm{vec}}$ 元素读 + 写），实际流量 = 有用 ÷ $\eta$，时间 = 流量 ÷ $\mathrm{BW}$；关键一步是**有用字节不变、浪费字节随 $\eta$ 倒数涨**。带宽口径时间（有效带宽定义沿 BP §9.2.2：读字节 + 写字节 ÷ 时间）：

$$
t_{\mathrm{copy}} = \frac{2 n_{\mathrm{vec}} \cdot \mathrm{bytes}}{\eta \cdot \mathrm{BW}} \qquad (3)
$$

**符号**： $n_{\mathrm{vec}}$ = 拷贝元素数（16777216）； $\mathrm{bytes} = 4$； $\eta$ 由式 (2)； $\mathrm{BW} = 3352$ GB/s。全数字代入（读写两侧同 $\eta$）：

| $s$ | $\eta$ | 实际流量 | $t_{\mathrm{copy}}$ |
| --- | --- | --- | --- |
| 1 | 1 | 128 MiB | 40.04 μs |
| 2 | 1/2 | 256 MiB | 80.08 μs |
| 4 | 1/4 | 512 MiB | 160.16 μs |
| 8 | 1/8 | 1024 MiB | 320.33 μs |
| 16 | 1/8 | 1024 MiB | 320.33 μs |

表读三句： $s$ 从 1 到 8，stride 每翻倍时间翻倍（ $\eta = 1/s$ 区）； $s$ 从 8 到 16，时间不变（饱和区，命题 2）；40.04 对 320.33 μs 的 8 倍差全部由 sector 浪费支付，FLOPs 一个没多算。float4 版（§4）与 $s = 1$ 行同流量，收益在指令侧。buffer 口径：strideCopy 的 idata/odata 各需 $s \cdot n_{\mathrm{vec}}$ 元素（教学假设）， $s = 16$ 时每侧 1 GiB——strided 访问的地址空间膨胀本身也是容量成本。

这笔账在 Nsight 里怎么读？两代指标并注：

- **legacy 口径**（BP §9.2.3）：Global Memory Load Efficiency / Global Memory Store Efficiency，逐字「this comparison of requested memory bandwidth to actual memory bandwidth is reported by the Global Memory Load Efficiency and Global Memory Store Efficiency metrics」——请求（有用）带宽对实际（sector 流量）带宽之比，即本章 $\eta$； $s = 2$ 的 load efficiency 为 50%。
- **现行口径**（NCU Memory Workload Analysis）：**Sectors/Req** = 每 request 平均 sector 数，逐字「Higher numbers can imply uncoalesced memory accesses and will result in increased memory traffic.」——fp32 全活跃 warp 最优 4， $s = 2$ 给 8， $s \ge 8$ 给 32；Bytes 指标 = sector 数 × 32 B（NCU 逐字「identical to the number of sectors multiplied by 32 byte, since the minimum access size in L1 is one sector」）。派生指标 achieved/ideal/excessive：ideal 是每活跃线程按给定宽度访问所需的 sector 数，excessive 是超出 ideal 的盈余—— $s = 2$ fp32 load：ideal 4、achieved 8、excessive 4。

**口径注**：Sectors/Req 的最优值随访问宽度变（4/8/16，§4），「大于 4 即 uncoalesced」是误读——float4 的 16 是最优。全 32 lane 活跃、同宽度访问下，legacy efficiency = 最优 Sectors/Req ÷ 实际 Sectors/Req（ $s = 2$： $4/8 = 1/2$ ），两代指标是同一笔账的两种表述。profiling 工作流与 drill-down 归 11.2，带宽测量方法归 2.4。

## 7. 边界：本章不做什么

地址模式账算完，剩下的归谁？本章交付 coalescing 判定（命题 1 + 式 (1)）、stride 饱和律（命题 2 + 式 (2)）、向量化 request/sector 双口径账、AoS/SoA 布局账、misaligned 两层代价、copy 带宽账（式 (3)）与 Nsight 两代指标读法；以下各有专章。shared 侧 bank conflict 判定与消除归 4.2（本章只留两套粒度对照）；tiling 把 strided global 访问换成 coalesced + shared 重排的复用账归 4.3；cp.async/TMA 的异步搬运归 4.4；GEMM/FA 的 global 侧流量账归 5.2/6.3；L1/L2 命中延迟与命中率账归 2.2；实测 profiling 归 11.2。代码 scope：机制片段级（§4 float4 copy、§6 strideCopy），不做模块走读。

本章交付清单（下游 recap 接口）：命题 1 + 式 (1) distinct sector 判定（§2）、命题 2 + 式 (2) 饱和律（§3）、向量化双口径表与 128-bit/对齐边界（§4）、AoS/SoA 与 misaligned 两层账（§5）、式 (3) copy 带宽账 + 两代指标（§6）。4.2 recap §5 末两套粒度对照作 bank 定律的引入；4.3 recap 式 (1) 作 tiling 装载侧的 coalesced 判定；4.4 recap §4 的 16 B 粒度作 cp.async 16 B 拷贝的起点；5.2/6.3 recap 式 (2)(3) 作 global 流量账的判定工具。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| CC 6.0+ transaction 数 = warp 触及的 distinct 32 B sector 数；访问单位 32 B 与 L1 caching 无关 | 式 (1)，命题 1，BP 逐字 | §2、练习 1/2 |
| 跨度公式双向错：stride-16 fp32 给 63 对真值 32（高估）；misaligned 连续给 4 对 5（低估） | §2 反例 | §2、练习 2 |
| 部分字请求仍搬整 sector；lane 置换不改变 transaction 数 | §2 BP 逐字 | §2 |
| fp32 对齐 stride 访问 $\eta = 1/\min(s, 8)$： $s \le 8$ 按 $1/s$ 降， $s \ge 8$ 饱和 1/8（32 sector 地板）；无界 $1/s$ 错 | 式 (2)，命题 2 | §3、练习 1/3 |
| 一般元素宽度饱和点 $s \cdot \mathrm{bytes} = 32$；fp64 在 $s = 4$ 饱和 | §3 一般形 | §3、练习 3 |
| 向量化省 request/指令（4→1）不省 sector（16→16）；Sectors/Req 最优 4/8/16 随宽度变；128-bit 上限、16 B 自然对齐违反 = UB | §4 表 + PTX/NCU 逐字 | §4、练习 2/4 |
| AoS 跨对象取单字段 = stride-4 访问 $\eta = 1/4$（838 GB/s）；SoA $\eta = 1$；整对象消费时 AoS 等价 float4 | §5 布局账 | §5、练习 4 |
| misaligned 两层：自然对齐违反 = 正确性（UB）；32 B 对齐缺失 = 性能（5 sector， $\eta = 0.8$；V100 790→632 GB/s） | §5 + BP 逐字 | §5、练习 2 |
| copy 带宽账 $t = 2 n_{\mathrm{vec}} \cdot \mathrm{bytes}/(\eta \mathrm{BW})$：128 MiB 有用，40.04 μs（s=1）对 320.33 μs（s≥8） | 式 (3) + §6 表 | §6、练习 4 |
| legacy efficiency = 请求/实际带宽；现行 Sectors/Req + achieved/ideal/excessive；全活跃同宽度下两者等价 | §6 两代指标 | §6 |

## 练习

1.（计算 · [10]）sector 数代入（对齐 fp32 warp load，128 B 有用）：(a) stride-1、stride-2、stride-16 各自的 $n_{\mathrm{sec}}$、流量与 $\eta$；(b) stride-8 与 stride-16 的有效有用带宽各多少，谁大？先预测 (b) 再算。校验：(a) 4/8/32 sector，128/256/1024 B， $\eta$ = 1、1/2、1/8，与 §3 表一致；(b) 两者同为 419 GB/s（饱和律，命题 2）。
2.（计算 · [20]）跨度反例与 misaligned：(a) stride-16 fp32 的跨度与 $\lceil \text{跨度}/32 \rceil$，对比 distinct sector 数，指出高估还是低估；(b) 连续 fp32 但基址偏移 4 B 的 $n_{\mathrm{sec}}$ 与 $\eta$；(c) float4 连续 warp load（16 B/thread）的 $n_{\mathrm{sec}}$ 与 Sectors/Req；(d) float4 基址偏移 16 B 的 $n_{\mathrm{sec}}$ 与 $\eta$。校验：(a) 1988 B → 63 对 32，高估（§2 反例）；(b) 5 sector、 $\eta = 0.8$，与 BP「4/5th」逐字一致（公开数值锚）；(c) 16，与 NCU 128-bit 最优 16 一致（公开数值锚）；(d) 17、 $\eta = 16/17$（§5）。
3.（推导 · [30]）补全命题 2 的证明（stride 饱和律）：(a) $s \ge 8$ 情形：证明 sector 号 $\lfloor si/8 \rfloor$ 随 $i$ 严格递增，得 $n_{\mathrm{sec}} = 32$；(b) $1 \le s \le 8$ 情形：证明每步增量至多 1、取遍 0 到 $\lfloor 31s/8 \rfloor$，并证取整恒等式 $\lfloor 31s/8 \rfloor + 1 = 4s$（提示： $4s - 1 \le 31s/8 \lt 4s$ 当且仅当 $1 \le s \le 8$ ）；(c) 推广到 fp64（ $\mathrm{bytes} = 8$ ）：饱和点 $s = 4$，算 $s = 2, 4, 8$ 的 $\eta$；(d) 说明无界 $1/s$ 在 $s = 16$ 处错多少倍、为什么饱和地板来自 lane 数。校验：(b) $s = 8$ 两情形同给 32（边界衔接）；(c) 1/2、1/4、1/4，与 $1/\min(s, 4)$ 一致；(d) 2 倍（ $1/8$ 对 $1/16$ ），与 §3 坑注一致。
4.（计算 · [30]）copy 综合账（本章口径， $\mathrm{BW} = 3352$ GB/s）：(a) 用式 (3) 算 $s = 1$ 的 $t_{\mathrm{copy}}$（128 MiB 有用）；(b) $s = 2$ 与 $s = 16$ 的实际流量与时间；(c) AoS 4 字段 struct 跨对象读 x 字段的 $\eta$ 与有效有用带宽；(d) float4 版相对标量版（每线程 4 元素）的每 warp request 数之比与 sector 数之比。先预测 (b) 中 $s = 16$ 对 $s = 8$ 的时间比再算。校验：(a) ≈40.04 μs，与 §6 表一致；(b) 256 MiB/80.08 μs 与 1024 MiB/320.33 μs， $s = 16$ 对 $s = 8$ 时间比 1（饱和）；(c) 1/4、838 GB/s，与 §5 一致；(d) request 比 1/4、sector 比 1（§4 表）。

## 参考

- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——§10.2.1 Coalesced Access to Global Memory（CC 6.0+「number of 32-byte transactions necessary to service all of the threads of the warp」逐字 + access unit 32-byte 与 L1 caching 无关逐字 + adjacent 4-byte words → four transactions 逐字 + subset 仍搬整 segment / permuted 仍 four 逐字）；§10.2.1.2（misaligned sequential → five 32-byte segments 逐字）；§10.2.1.3（V100 offset 非 8 words 倍数 → 4/5 吞吐、790 GB/s 逐字，时代口径）；§10.2.1.4 Strided Accesses（strideCopy kernel 逐字同形 + stride 2 → 50% + 「32 32-byte segments」饱和逐字）；§9.2.2 有效带宽定义（读 + 写字节 ÷ 时间）；§9.2.3（Global Memory Load/Store Efficiency legacy 指标逐字）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——Device Memory Accesses 节（archive/13.0.0 §8.3.2；13.5 子页 02-basics/writing-cuda-kernels.html）：32/64/128 B naturally-aligned transactions 逐字 + 「throughput is divided by 8」逐字 + cudaMalloc ≥256 B 对齐逐字 + cudaMallocPitch pad 逐字 + built-in 向量类型对齐表
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——§9.7.10.8 ld（.v2/.v4 向量 + 「Vectors cannot exceed 128-bits in length; for example, .v4 .f64 is not allowed.」逐字 + ld.v4.b32 access size 16 bytes 逐字 + 自然对齐违反 undefined behavior 逐字）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——ProfilingGuide §2.11.2 L1/TEX Cache memory table（「Each instruction generates exactly one request for LSU traffic」逐字 + Sectors/Req 最优 32-bit:4 / 64-bit:8 / 128-bit:16 逐字 + higher numbers → uncoalesced 逐字 + Bytes = sectors × 32 B 逐字）；achieved/ideal/excessive 派生指标口径
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/6_Performance/alignedTypes（对齐与向量化访存官方教学例，§4；现行 cpp/ 路径，旧 Samples/ 已重组）
