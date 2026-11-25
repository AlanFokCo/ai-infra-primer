# 4.3 Tiling 与数据复用（Tiling and Data Reuse）

> 同一个 4096³ 的 bf16 GEMM，FLOPs 一个不多一个不少，naive kernel 要搬 274.9 GB、带宽口径 82 ms，shared tiling 只搬 2.15 GB、640.7 μs——128 倍的流量差全部来自「一个元素搬上来之后被用几次」。本章交付 GEMM tiling 的算术强度推导（命题 1：naive 每元素 $O(1)$ 复用 → 强度 $1/\mathrm{bytes}$；命题 2：shared tile 每元素 $O(T)$ 复用 → 强度 $I = 2T_M T_N/((T_M+T_N)\mathrm{bytes})$，方阵 tile 即 $T_{\mathrm{tile}}/\mathrm{bytes}$）、register tiling 再升复用与 global→shared→register 三级层级账、tile size 的 shared/register/occupancy 三预算约束（命题 3：固定周长方阵 tile 最优）、double buffering 的延迟隐藏条件与容量翻倍代价（命题 4）、教学 GEMM 的 naive 对 tiled 强度对比 | 前置：2.4（强度/roofline/ridge 入口与层级带宽表、每 SM 份额 25.4 GB/s）、2.3（shared 228 KB/227 KB/256 KB 口径、160:10:1 带宽比值）、4.1（coalescing 判定与 16 B 向量化）、4.2（bank conflict 与 swizzle）、3.4（occupancy 四约束式 (1) 与 compute-bound 低 occupancy 命题 2）、1.3（MMA 操作数复用与 wgmma 异步）、1.4（Little's law 与 6.6 KB 带宽延迟积）；下游 4.4（cp.async/TMA 异步流水）、5.2（CUTLASS 全量走读，本章理论 recap 源）、6.3（FA tile 预算约束同法） | 锚点：CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）、CUTLASS efficient_gemm.md（[NVIDIA, CUTLASS docs](../references.md)，教学参照 + 官方实现）、cuda-samples matrixMul（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 延迟实测口径）——官方规格 2026-11-25 一手核实

## 1. 0.5 对 64 FLOP/byte：复用因子决定强度

4096³ 的 bf16 GEMM 共 $2 \times 4096^{3} = 137.4$ GFLOP，H100 的 BF16 峰值算力 138.9 μs 就能算完；但 naive kernel 每个输出元素都从 HBM 重读一行 A、一列 B，总流量 274.9 GB，带宽口径要 82 ms——比算力上限慢约 590 倍。FLOPs 没变、带宽没变，变的只有一件事：**每个从 HBM 搬上来的元素被消费几次**。本章回答三个问题：复用因子怎么换算成算术强度，naive 的 $O(1)$ 怎么变成 tiling 的 $O(T)$？global→shared→register 三级各付多少流量、各拿多少复用？tile 尺寸被哪几个预算约束钉死，double buffering 拿什么换什么？

本 Part 前三章各管一笔正交的账：4.1 问「一次 warp 搬运浪费几个 sector」（地址模式），4.2 问「一次 shared 访问串行几拍」（bank 模式），本章问「同一份数据从 HBM 搬几次」（复用结构）。三者可以独立优化：coalescing 不改复用次数，消 bank conflict 不改流量，tiling 也不改单次访问的地址模式。tiling 的账之所以最大——开场 128 倍的流量差是 sector 级与周期级优化都给不出的量级——是因为它动的是「搬几次」这个乘数，而不是「每次搬多宽」或「每拍服务几个」。

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0，bf16 输入、fp32 累加**（沿 2.4/3.4）；running example 为方阵 GEMM $M = N = K = 4096$，threadblock tile $T_M = T_N = 128$、 $T_K = 32$（§6 对比 $T_K = 64$，换口径原因：展示 $T_K$ 对延迟账而非强度账的影响），256 threads/block，每线程 register tile $r_M \times r_N = 8 \times 8$。 $\mathrm{BW} = 3352$ GB/s、 $F_{\mathrm{peak}} = 989.5$ TFLOPS、ridge $R \approx 295$ FLOP/byte（2.4 §3 recap，datasheet dense 口径）；每 SM 份额 $`\mathrm{BW}_{\mathrm{SM}} = 25.4`$ GB/s、 $F_{\mathrm{SM}} = 7.496$ TFLOPS（2.4 式 (4) 归一化）。周期↔时间换算取 TC-boost 1.83 GHz，shared 峰值 GB/s 换算沿 2.3 的 1.98 GHz（253 GB/s）——两个 clock 各归其口径（2.3 章脊声明 recap）。HBM 延迟 478.8 cycles 为 H800 微基准实测 [B]（[Luo et al., 2024](../references.md)），非 H100 官方规格。片上 KB = 1024 B，带宽 GB/s 按 $10^{9}$ B/s（2.1/2.3 口径）。

符号声明：**本章 $M, N, K$ = GEMM 问题维（行/列/归约维），非消息字节/参数量/序列长**（kernel 章局部声明，通信章按通信语义）； $T_M, T_N, T_K$ = threadblock tile 三维、 $T_{\mathrm{tile}}$ = 方阵 tile 边长（**tile 维，非序列长 $T$；禁裸 $T$ 表 tile**）； $r_M, r_N$ = 每线程 register tile 维（§4 局部）； $I$ = 算术强度（FLOP/byte，沿 models/inference）；混场声明： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread（3.4 recap 出现时同章区分）； $\mathrm{bytes}$ = 每元素字节数（bf16 取 2；bytes 不占字母， $B$ 专归 batch 不表 tile）； $S_{\mathrm{sh}}$ = shared bytes/block； $n_{\mathrm{stage}}$ = shared 缓冲份数（§5，单缓冲 1、双缓冲 2）； $t_{\mathrm{comp}}, t_{\mathrm{load}}, t_{\mathrm{lat}}$ = 每 tile 计算/装载服务/首字节延迟时间， $\tau$ = 稳态每 tile 时间（§6）； $b_{\mathrm{act}}$ = resident blocks/SM（3.4 式 (1)）。B/KB/MB/GB 是单位符号豁免。

**边界声明**：本章交付 tiling 的复用与强度理论账。装载侧 coalescing 判定归 4.1（本章装载一律按 coalesced 模式设计）；shared 侧 bank conflict 判定与消除归 4.2（本章强度账按无冲突峰值计——冲突乘服务周期、不改流量，两笔账勿混）；异步搬运 cp.async/TMA 与 multi-stage 流水归 4.4；occupancy 精确计算归 3.4（本章只代式 (1)）。**两层走读纪律（4.3↔5.2）**：本章代码 scope 为机制片段级（§7 教学 GEMM），CUTLASS/cuBLAS 模块全量走读 + 公式逐行对照归 5.2；5.2 重访本章片段一律 recap + 指引，本章不预支走读内容。L2 跨 block 复用与 block 调度（rasterization）的流量修正归 5.2 走读对账（§3 假设注）；FA 的 tile↔shared/register 预算约束归 6.3（本章 §5 方法是它的 GEMM 侧原型）。

## 2. naive GEMM：O(1) 复用钉死 O(1) 强度

一个输出元素要消费多少数据？ $C[i,j] = \sum_{k} A[i,k] B[k,j]$ 共 $K$ 次乘加，naive kernel 一个线程负责一个输出，每次乘加从 global 读 $A[i,k]$ 与 $B[k,j]$ 各一个元素。所谓复用因子（reuse factor），是指一个数据元素从某一层级装载上来之后被消费的次数（口径：按「每次装载服务几次消费」计，不是逻辑上被几个输出需要——逻辑复用与装载复用之差正是 tiling 的收益来源）。naive 的每次装载只服务 1 次乘加，复用因子 1 = $O(1)$。

**命题 1**（naive GEMM 强度）。无缓存复用口径下，naive GEMM 的算术强度为

$$
I_{\mathrm{naive}} = \frac{2K}{2K \cdot \mathrm{bytes}} = \frac{1}{\mathrm{bytes}} \qquad (1)
$$

与问题规模 $M, N, K$ 无关，即 $O(1)$。**证明**（一步计数）：每输出 $2K$ FLOPs（ $K$ 次乘加、每次 2 FLOPs），每输出读 $2K$ 个元素（A 行 $K$ 个 + B 列 $K$ 个），流量 $2K \cdot \mathrm{bytes}$，相除即得；对全部 $MN$ 个输出求和，分子分母同乘 $MN$，强度不变。∎ 代入 bf16（ $\mathrm{bytes} = 2$）： $I_{\mathrm{naive}} = 0.5$ FLOP/byte，离 ridge 295 差 590 倍——**naive GEMM 是结构性 memory-bound，与卡多快无关**。全 kernel 流量 $2MNK \cdot \mathrm{bytes} = 274.88$ GB，带宽口径时间 $274.88 \times 10^{9}/(3352 \times 10^{9}) = 82.00$ ms；算力口径上限 $137.4 \times 10^{9}/(989.5 \times 10^{12}) = 138.9$ μs，两者之比 590 恰等于 $R/I_{\mathrm{naive}} = 295/0.5$（roofline 斜线区，2.4 recap）。

**口径注**：式 (1) 按每线程请求计（每次 load 计一次）；warp 内 A 同址 broadcast 省的是 sector 流量（4.1 服务口径），不改「一次装载服务一次乘加」的复用结构——请求口径与服务口径差常数倍， $O(1)$ 的阶相同。

边界反例（缓存不改阶）：warp 内 32 线程读同一个 $A[i,k]$（同址 broadcast，4.2），sector 流量省 32 倍；L1 若留住 A 行，后续迭代还能命中。但 BP 对官方教学例的分析逐字（[NVIDIA, Best Practices](../references.md) §10.2.3.2，[C]）：「when many warps execute on the same multiprocessor simultaneously, as is generally the case, the cache line may easily be evicted from the cache between iterations i and i+1」——多 warp 并发时 cache line 在两次迭代之间就被逐出，跨迭代复用靠不住。所以式 (1) 的「无缓存复用」是上界口径：缓存命中能降实际流量，但**每输出独立装载的结构不变，强度不随问题规模上升**， $O(1)$ 的阶钉死。BP 官方 V100 教学例实测 119.9 GB/s（时代口径，机制跨代），正是这个结构的兑现。

## 3. shared tiling：O(T) 复用与强度公式

怎么把复用因子从 1 抬到 $O(T)$？所谓分块（tiling），是指把输出矩阵切成 $T_M \times T_N$ 的 tile、每个 threadblock 负责一块，沿 $K$ 维每次取 $T_K$ 深的 A/B 子块进 shared，让**一次装载服务整个 tile 的消费**。推导路线分三步：先数一个 K 步的装载流量与 FLOPs，再算每个元素的复用因子，最后对 block 求和得全 kernel 流量；关键一步是「A tile 的每个元素被 $T_N$ 个输出消费、B tile 的每个元素被 $T_M$ 个输出消费」——复用因子从 1 变成 tile 边长。

```
C tile (T_M x T_N) = sum_k A tile 行 x B tile 列（K 维循环，每步 T_K 深）

      A tile T_M x T_K          B tile T_K x T_N
      ┌─────────────┐           ┌───────────────────┐
      │  HBM 读一次  │           │    HBM 读一次      │
      │  进 shared   │     ×     │    进 shared       │
      └─────────────┘           └───────────────────┘
            │  每元素被 T_N 个输出消费   │  每元素被 T_M 个输出消费
            ▼                          ▼
      ┌─────────────────────────────────────┐
      │  C tile T_M x T_N：register 累加器    │
      │  K/T_K 步全程驻留，最后写回 HBM 一次   │
      └─────────────────────────────────────┘
每 K 步装载：(T_M + T_N)·T_K 元素；每 K 步计算：2·T_M·T_N·T_K FLOPs
```

**命题 2**（shared tiling 的 per-block 强度）。threadblock tile $T_M \times T_N$、K 步深 $T_K$ 的 shared tiling GEMM，每 threadblock 的算术强度为

$$
I_{\mathrm{blk}} = \frac{2 T_M T_N}{(T_M + T_N) \cdot \mathrm{bytes}} \qquad (2)
$$

方阵 tile（ $T_M = T_N = T_{\mathrm{tile}}$）化简为 $I_{\mathrm{blk}} = T_{\mathrm{tile}}/\mathrm{bytes} = O(T)$。**证明**（三步计数）：其一，一个 K 步从 HBM 装载 A tile $T_M T_K$ 元素 + B tile $T_K T_N$ 元素，流量 $(T_M + T_N) T_K \cdot \mathrm{bytes}$；其二，同一步内计算 $T_M T_N$ 个输出各 $T_K$ 次乘加，共 $2 T_M T_N T_K$ FLOPs——A tile 每个元素参与 $T_N$ 个输出的乘加（复用 $T_N$），B tile 每个元素参与 $T_M$ 个（复用 $T_M$）；其三，强度 = FLOPs ÷ 流量 = $2 T_M T_N T_K / ((T_M + T_N) T_K \cdot \mathrm{bytes})$， $T_K$ 约去即式 (2)。 $K/T_K$ 步求和分子分母同乘步数，强度不变；C 写回 $T_M T_N$ 元素是低阶项（下段口径注）。∎

式 (2) 的两个推论值得钉死。其一，**强度与 $K$、 $T_K$ 无关**：归约维深度在分子分母同现约去——问题加深不改复用结构，K 步加深只改单次装载的粒度、不改每元素被消费的次数。其二，强度反比于 $\mathrm{bytes}$：同 tile fp32 给 32、bf16 给 64、fp8 给 128；但对应精度类的 ridge 也随算力峰值同比抬（FP8 峰值 2× BF16，2.4 recap），所以 $I/R$ 比值在精度切换下不变——**tiling 的 bound 判定对精度不变**，低精度改变的是绝对流量（减半）与绝对时间，不是 roofline 上的相对位置。

全数字代入（本章口径 bf16）： $T_{\mathrm{tile}} = 128$ 给 $I_{\mathrm{blk}} = 128/2 = 64$ FLOP/byte——比 naive 的 0.5 抬 128 倍，恰等于复用因子之比（ $T_N = 128$ 对 1）。tile 尺寸扫描表（式 (2) + 全 kernel 流量式 (3)）：

| $T_{\mathrm{tile}}$ | $I_{\mathrm{blk}}$（FLOP/byte） | HBM 流量 | 带宽口径时间 | 对 naive 流量比 |
| --- | --- | --- | --- | --- |
| 32 | 16 | 8.590 GB | 2562.6 μs | 32× |
| 64 | 32 | 4.295 GB | 1281.3 μs | 64× |
| 128 | 64 | 2.147 GB | 640.7 μs | 128× |
| 256 | 128 | 1.074 GB | 320.3 μs | 256× |

全 kernel 流量由 block 数求和： $(M/T_M)(N/T_N)$ 个 block，每 block 沿 $K$ 读 $(T_M + T_N) K$ 元素，故

$$
\mathrm{traffic}_{\mathrm{HBM}} = MNK \left( \frac{1}{T_M} + \frac{1}{T_N} \right) \cdot \mathrm{bytes} \qquad (3)
$$

**假设注**（口径钉死）：式 (3) 假设**每 tile 从 HBM 读、无跨 block L2 复用**，C 写 $O(MN)$ 低阶略。代入 $T_M = T_N = 128$： $4096^{3} \times (2/128) \times 2 = 2.147$ GB；C 写回 $4096^{2} \times 2 = 33.55$ MB，占 1.56%，确为低阶。带宽口径时间 640.7 μs，加 C 写 650.7 μs，仍是算力上限 138.9 μs 的 4.61 倍——**per-block 口径下 $I = 64 \lt R = 295$，tiling 之后依然 memory-bound**。这不是 tiling 失效：式 (2) 要抬到 ridge 需 $T_{\mathrm{tile}} \ge 590$（bf16），远超 shared 预算（§5）；真实 GEMM 靠 **L2 跨 block 复用**补上缺口——同一 B tile 被多个行 block 消费、同一 A tile 被多个列 block 消费，L2 命中把实际 HBM 流量压到式 (3) 之下，kernel 进 compute-bound 区。L2 命中率与 block 调度（rasterization）的修正账归 5.2 走读对账，本章不预支。几何解读（2.4 §7 recap）：tiling 把 roofline 上的点从 $I = 0.5$ 右移到 $I = 64$，沿斜线爬升 128 倍；L2 复用再右移进平台区。

官方教学锚（BP §10.2.3.2，V100 时代口径）：naive 119.9 GB/s → A tile 进 shared 144.4 GB/s（1.204×）→ A/B tile 都进 shared 195.5 GB/s（1.354×），合计 1.631×。BP 把收益结构一句话钉死（逐字）：「the performance improvement is not due to improved coalescing in either case, but to avoiding redundant transfers from global memory」——**收益来自消除冗余搬运（复用），不是来自合并**。这与 4.1 的边界一致：coalescing 治「一次搬运浪费几个 sector」，tiling 治「同一数据搬几次」，两笔账正交。官方例的 shared 读侧用 broadcast（「a value in shared memory is broadcast to all threads in a warp」），bank 冲突账归 4.2。

> **坑**：把 per-block 强度当整 kernel 实测强度。触发条件：用式 (2) 算出 $I = 64$ 就断言 GEMM 是 memory-bound、或反过来拿实测 compute-bound 否定式 (2)；现象：两个结论都对不上实测——式 (2) 是「无 L2 复用」上界口径，实测流量被 L2 跨 block 复用压低；规避：per-block 账用于 tile 尺寸选型（式 (2) 单调升），整 kernel bound 判定用实测流量（2.4 式 (8) 口径），L2 修正归 5.2。

## 4. register tiling：把复用再抬一级，瓶颈跟着上移

数据进了 shared 就够快了吗？shared 峰值 253 GB/s 只是 HBM 每 SM 份额的 10 倍（2.3 比值 recap），若每个线程每次乘加都从 shared 重读操作数，shared 侧流量会成为新瓶颈。register tiling 把复用再抬一级：每线程负责 $r_M \times r_N$ 个输出（本章 $8 \times 8 = 64$ 个，fp32 累加器驻留 register），每个 k 步从 shared 读 $r_M$ 个 A 值 + $r_N$ 个 B 值，做 $2 r_M r_N$ FLOPs 的外积——A 值复用 $r_N$ 次、B 值复用 $r_M$ 次。每线程强度：

$$
I_{\mathrm{reg}} = \frac{2 r_M r_N}{(r_M + r_N) \cdot \mathrm{bytes}} \qquad (4)
$$

代入 $r_M = r_N = 8$、bf16： $I_{\mathrm{reg}} = 8/2 = 4$ FLOP/byte。结构同式 (2)：把 tile 边长换成 register tile 边长，强度随边长线性升。256 线程 $\times$ 64 输出 = 16384 = $128 \times 128$，与 threadblock tile 恰好闭合（练习 3(d) 的一般约束）。

三级层级账（每 block 每 k 步， $T_M = T_N = 128$、 $r = 8$、256 线程；各层带宽库存 recap 2.4 §4 表）：

| 层级 | 每 k 步流量 | 复用因子 | 强度（FLOP/byte） | 该层 ridge | 判定 |
| --- | --- | --- | --- | --- | --- |
| HBM→shared | $(128+128) \times 2 = 512$ B | A×128、B×128 | 64（式 (2)） | 295（HBM） | 紧（待 L2 补） |
| shared→register | $256 \times (8+8) \times 2 = 8192$ B | 相对 HBM 放大 16× | 4（式 (4)） | 29.6（shared） | 紧 |
| register→MMA | 执行管线内读出 | 指令内 A×8、B×16 | — | 1.85（register） | 宽裕 |

**符号与口径**：shared ridge = $F_{\mathrm{SM}}/253$ GB/s = $7496/253 \approx 29.6$；register ridge = $7496/4055 \approx 1.85$（register 4055 GB/s 是需求上界口径，2.4 表注 recap）；register→MMA 行的复用是 MMA 指令内部的操作数复用（1.3 §4 recap：mma.sync m16n8k16 的 A 元素被 n=8 个输出消费、B 元素被 m=16 个消费），不产生用户可见 load。shared→register 行是**每线程请求口径**（每线程读自己的操作数，各计一次）；实际服务口径更低——warp 内同址读走 broadcast（4.2 命题 1 例外）：本章线程布局下 warp = 2 个 ty 行 × 16 个 tx，A 侧 2 个 distinct 片段 × 8 值、B 侧 16 个 distinct 片段 × 8 值，每 warp 每 k 步 distinct 流量 $(2 \times 8 + 16 \times 8) \times 2 = 288$ B，对请求口径 1024 B 省 3.56 倍，有效强度 $4096/288 \approx 14.2$ FLOP/byte——仍低于 shared ridge 29.6。**结论：tiling 把瓶颈逐级上移**——naive 卡在 HBM（0.5），shared tiling 后 HBM 侧仍紧（64，待 L2），register tiling 后 shared 侧成为新约束（4–14.2 对 29.6）。结构性解法是 Hopper 的 wgmma：B 操作数从 shared 经描述符直喂 Tensor Core、跳过每线程 register 中转（1.3 §5 recap），shared 每 warp-group 只被硬件读一遍；配合 ldmatrix 与 swizzle 布局（4.2 §4）把冲突也消掉。这条路径的模块级走读归 5.2，本章只立层级账动机。

表里「相对 HBM 放大 16×」一列是读法的关键：shared 侧每 k 步要供给 HBM 侧 16 倍的流量（8192 对 512 B），而 shared 的带宽库存只是 HBM 每 SM 份额的 10 倍（253 对 25.4 GB/s，2.3 比值 recap）。需求放大超过供给放大，所以 shared 侧的强度对 ridge 之比（ $4/29.6 \approx 0.14$）比 HBM 侧（ $64/295 \approx 0.22$）更紧——瓶颈没有消失，只是换了地址。这就是「数据搬进 shared 就完事」不成立的原因：补缺口的工具是 broadcast（降服务流量）、swizzle（消冲突乘数）与 wgmma（改供给结构），各归 4.2/1.3/5.2。

> **提点**：层级账的读法是「每层各付各的流量、各对各的 ridge」——tiling 不是把数据搬进快层就完事，而是把瓶颈从慢层推到快层，推到哪里就要用哪层的工具（HBM 侧靠复用与 L2，shared 侧靠 broadcast/swizzle/wgmma，register 侧靠指令内复用）。删掉这条类比，上表与式 (2)(4) 的推导链依然完整。

## 5. tile size 选择：三个预算钉死一个目标

tile 越大强度越高（式 (2) 单调升），为什么不直接取最大？因为三个预算约束各砍一刀：shared 容量、register 容量、occupancy。目标函数一句话：**在预算内取最大 tile**（强度最大化），约束逐条代入。

**shared 预算**。每 block 的 shared 占用 = 缓冲份数 × 每份 tile 字节：

$$
S_{\mathrm{sh}} = n_{\mathrm{stage}} \cdot (T_M + T_N) \cdot T_K \cdot \mathrm{bytes} \qquad (5)
$$

**符号**： $n_{\mathrm{stage}}$ = shared 缓冲份数（单缓冲 1；双缓冲 2，§6；multi-stage 大于 2 归 4.4）。代入 $128 \times 128 \times 32$、bf16：单缓冲 $(128+128) \times 32 \times 2 = 16384$ B = 16 KB，双缓冲 32 KB； $T_K = 64$ 则 32/64 KB。上限两条（2.3/3.4 recap）：每 block 用户上限 227 KB（1 KB 预留），静态声明上限 48 KB、超出须动态 shared + opt-in。 $T_K$ 在式 (2) 里约去、**不改强度**，但线性吃 shared 预算—— $T_K$ 的选择是延迟账与预算账（§6），不是强度账。

**register 预算**。累加器是主项： $r_M r_N$ 个 fp32 输出 = $r_M r_N$ 个 register（本章 64 个），加操作数片段与地址变量（教学假设合计 $R_{\mathrm{th}} = 128$）。约束两条：每线程 $R_{\mathrm{th}} \le 255$（1.2 Table 4）；block 总量 $`R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}} \cdot b_{\mathrm{act}} \le R_{\mathrm{file}} = 65536`$（3.4 式 (1) register 项）。

**occupancy 回代**（3.4 式 (1) recap，不重推）。双缓冲 $T_K = 32$ 配置： $S_{\mathrm{sh}} = 32$ KB → $S_{\mathrm{alloc}} = 33792$ B → shared 项 $\lfloor 233472/33792 \rfloor = 6$；register 项 $\lfloor 65536/(128 \times 256) \rfloor = 2$；warp 项 $\lfloor 64/8 \rfloor = 8$；block slots 32。四项取 min = **2 blocks，register 绑定**， $W_{\mathrm{active}} = 16$， $\mathcal{O} = 0.25$。compute-bound GEMM 低 occupancy 可接受（3.4 命题 2 recap：FP32 依赖链 16 warps 即满吞吐，Tensor Core kernel 靠 ILP 与异步流水补并发），但延迟隐藏必须达标——双缓冲与 multi-stage 正是用流水并发替代 warp 并发（§6/4.4）。

三个预算并不独立： $`T_M T_N = \mathrm{threads}_{\mathrm{block}} \cdot r_M r_N`$ 把 tile 几何与 launch 配置耦合——加大 tile 要么加大 register tile（累加器 register 按平方涨），要么加线程（block register 总量按线性涨），两条路都挤 register 项；加大 $T_K$ 不改强度却线性吃 shared 项。所以 tile 选型是一个小约束优化：目标函数是式 (2)，约束是式 (5)、register 两条不等式与 3.4 式 (1) 四约束，可行域通常只剩少数几个配置。CUTLASS 把候选 tile 做成模板参数枚举、选择交给 heuristic 与 autotune（cuBLASLt 的 algo 选择机制归 5.2），本章给目标函数与约束，不给求解器。

**命题 3**（固定周长方阵 tile 最优）。shared 预算固定（即 $T_M + T_N$ 固定、 $T_K$ 固定）时， $I_{\mathrm{blk}}$ 在 $T_M = T_N$ 处取最大。**证明留作练习 3(a)**（提示：式 (2) 分子 $T_M T_N$ 在和固定时相等取最大，分母不变）。∎ 这解释了 CUTLASS 教学文档的默认取向：threadblock tile 以方阵为基准（`ThreadblockShape::{kM, kN, kK}`，[NVIDIA, CUTLASS docs](../references.md) efficient_gemm.md，教学参照），矩形 tile 只在问题形状偏斜时启用。代入反例： $T_M = 128, T_N = 256$（周长 384 对 方阵 192×2=384 同预算）给 $I = 2 \times 128 \times 256/(384 \times 2) = 85.33$，低于同周长方阵 192 的 $192/2 = 96$——偏斜白付容量。

大 tile 的 register 悬崖（ $T_{\mathrm{tile}} = 256$ 的两条路都死）：保 256 线程则 $r = 16$，累加器 $16 \times 16 = 256$ regs 超每线程上限 255，spill 到 local（HBM backing，2.3 §6 代价账）；保 $r = 8$ 则需 $256 \times 256/64 = 1024$ 线程（恰撞 block 上限），block register 总量 $128 \times 1024 = 131072 > 65536$，一个 block 都装不下、launch 失败。**所以大 tile 不是「多开线程」能解决的**——它要求 warp 级 MMA 几何把累加器摊到 warp-group（wgmma m64nNk16 的 $64 \times N$ 累加器摊 128 线程，1.3 式 (4) recap），这条路的 tile 选型归 5.2。CUTLASS 文档对大 tile 的取舍逐字（efficient_gemm.md）：「A larger threadblock means fewer fetches from global memory, thereby ensuring that DRAM bandwidth does not become a bottleneck.」——收益方向与式 (2) 一致；同节也给出大 tile 的形状失配代价（M/N 小时线程空转、block 数不足喂不满 SM），split-K 修正归 5.2。

## 6. double buffering：预取藏延迟，容量翻倍

同步装载每 tile 暴露多少等待？本章口径下每 tile 装载 16 KB，HBM 首字节延迟 $t_{\mathrm{lat}} = 478.8/1.83$ GHz = 261.6 ns，带宽口径装载服务时间 645.2 ns，而 tile 计算只要 139.9 ns——同步版每 tile 1046.7 ns 里计算只占 13.4%。所谓双缓冲（double buffering），是指开两份 shared tile 缓冲、计算第 $k$ 块的同时预取第 $k+1$ 块，把装载时间藏进计算时间的机制。时间账先钉定义（per-block、单 block 独占 SM 的理想口径）：

$$
t_{\mathrm{comp}} = \frac{2 T_M T_N T_K}{F_{\mathrm{SM}}}, \qquad t_{\mathrm{load}} = \frac{(T_M + T_N) T_K \cdot \mathrm{bytes}}{\mathrm{BW}_{\mathrm{SM}}} \qquad (6)
$$

**符号**： $F_{\mathrm{SM}} = 7.496$ TFLOPS、$`\mathrm{BW}_{\mathrm{SM}} = 25.4`$ GB/s（2.4 份额口径）；装载完成时间近似 $t_{\mathrm{lat}} + t_{\mathrm{load}}$（教学近似：首字节延迟 + 传输服务时间，上界口径）。代入 $T_K = 32$： $t_{\mathrm{comp}} = 2 \times 128 \times 128 \times 32/(7.496 \times 10^{12}) = 139.9$ ns， $t_{\mathrm{load}} = 16384/(25.4 \times 10^{9}) = 645.2$ ns。**口径注**：多 block 共驻时算力与带宽份额同比例摊薄， $t_{\mathrm{comp}}$ 与 $t_{\mathrm{load}}$ 同乘 block 数、比值结论不变； $t_{\mathrm{load}}$ 是带宽口径，假定在飞并发足够——单份 16 KB tile 在飞已超每 SM 6.6 KB 带宽延迟积（1.4 recap），一条装载流即可喂满份额。

```
同步版（每 tile 串行，等待全暴露）：
  │ 装载 tile k (t_lat + t_load) │ 计算 k (t_comp) │ 装载 k+1 ...
双缓冲（计算与下一块装载重叠）：
  计算: │ tile k (t_comp) │        tile k+1         │ ...
  装载: │ tile k+1 (t_lat + t_load) │ tile k+2 ...  │
  稳态每 tile τ = max(t_comp, t_lat + t_load)
```

**命题 4**（双缓冲稳态时间与隐藏条件）。双缓冲稳态每 tile 时间为

$$
\tau_{\mathrm{db}} = \max\left( t_{\mathrm{comp}},\ t_{\mathrm{lat}} + t_{\mathrm{load}} \right) \qquad (7)
$$

代入本章口径： $\tau_{\mathrm{db}} = \max(139.9, 906.8) = 906.8$ ns——装载主导。装载被完全隐藏当且仅当 $t_{\mathrm{comp}} \ge t_{\mathrm{lat}} + t_{\mathrm{load}}$；其必要条件是带宽平衡 $t_{\mathrm{comp}} \ge t_{\mathrm{load}}$，等价于 $I_{\mathrm{blk}} \ge R$（per-SM ridge = 芯片 ridge，分子分母同除 $n_{\mathrm{SM}}$）。对同步版的加速比

$$
\mathrm{speedup} = \frac{t_{\mathrm{comp}} + t_{\mathrm{lat}} + t_{\mathrm{load}}}{\tau_{\mathrm{db}}} \qquad (8)
$$

上界 2（ $t_{\mathrm{lat}} \to 0$ 且两时间平衡时取到）。**证明**（路线：循环时序分析，两步）：其一，双缓冲循环在计算 tile $k$ 之初发射 tile $k+1$ 的装载（缓冲 $k-1$ 刚释放），装载单份在飞、完成需 $t_{\mathrm{lat}} + t_{\mathrm{load}}$；计算耗时 $t_{\mathrm{comp}}$ 后若装载未完则等待剩余，故每 tile 循环时间 = $t_{\mathrm{comp}} + \max(0, t_{\mathrm{lat}} + t_{\mathrm{load}} - t_{\mathrm{comp}})$ = 式 (7)。其二，同步版每 tile 串行三段 $t_{\mathrm{comp}} + t_{\mathrm{lat}} + t_{\mathrm{load}}$，相除得式 (8)；分母至少是分子的一半（ $\max(a,b) \ge (a+b)/2$），故上界 2。带宽平衡等价性： $t_{\mathrm{comp}} \ge t_{\mathrm{load}}$ 两边代入式 (6) 整理即 $`I_{\mathrm{blk}} \ge F_{\mathrm{SM}}/\mathrm{BW}_{\mathrm{SM}} = R`$。∎

全数字表（式 (6)-(8)，本章口径）：

| $T_K$ | $t_{\mathrm{comp}}$ | $t_{\mathrm{load}}$ | $t_{\mathrm{lat}} + t_{\mathrm{load}}$ | $\tau_{\mathrm{sync}}$ | $\tau_{\mathrm{db}}$ | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 139.9 ns | 645.2 ns | 906.8 ns | 1046.7 ns | 906.8 ns | 1.154 |
| 64 | 279.8 ns | 1290.4 ns | 1552.0 ns | 1831.8 ns | 1552.0 ns | 1.180 |

诚实读数：per-block 口径下 $I = 64 \ll 295$，装载主导（ $t_{\mathrm{load}} \gg t_{\mathrm{comp}}$），双缓冲只藏掉计算那一段，加速 1.15–1.18 有限；真实 GEMM 的更大收益来自两处——L2 复用抬有效带宽（ $t_{\mathrm{load}}$ 下降，5.2 对账）与 multi-stage 流水让多份装载同时在飞（4.4：本章 tile 三 stage 即达带宽地板 645.2 ns，对同步 1.62×）。延迟侧另有一笔： $t_{\mathrm{comp}} \ge t_{\mathrm{lat}}$ 是否成立决定单 tile 计算能否盖住首字节延迟—— $T_K = 32$ 时 139.9 低于 261.6（盖不住）， $T_K = 64$ 时 279.8 高于 261.6（盖住）。**这是 $T_K$ 不宜过小的延迟账理由**（另一理由：每 tile 的同步与调度开销被更长的计算摊薄）； $T_K$ 再大则吃 shared 预算（式 (5)），两头夹出中间值。

两个口径注把命题 4 的读法钉死。其一，**稳态口径**：首个 tile 的装载没有计算可藏，付全额 $t_{\mathrm{lat}} + t_{\mathrm{load}}$（流水填充段）；式 (7) 只计稳态循环段，填充段占比随 K 步数增多而降——本章 $K/T_K = 128$ 步，填充约 1/128，可略。其二，**延迟隐藏与带宽平衡是两笔账**： $t_{\mathrm{comp}} \ge t_{\mathrm{lat}}$ 只说明「计算够长、盖得住首字节等待」，不说明「装载被完全隐藏」——完全隐藏要盖住整个 $t_{\mathrm{lat}} + t_{\mathrm{load}}$。前者是延迟账（1.4），后者是带宽账（2.4）； $T_K = 64$ 时前者成立而后者不成立，正是「延迟藏住、带宽仍紧」的中间态，也是 4.4 multi-stage 要处理的剩余缺口。

容量代价与 occupancy（式 (5) + 3.4 式 (1)）：双缓冲把 $S_{\mathrm{sh}}$ 从 16 KB 翻到 32 KB，shared 项从 13 blocks 掉到 6 blocks——但本章配置 register 项 2 更紧， $b_{\mathrm{act}}$ 不变、occupancy 不变（3.4 §5 结论 recap：只有绑定约束的松紧才改变 occupancy）。**双缓冲在 register 绑定配置下是「免费」的**（口径：免费指 occupancy 无损，shared 容量仍实付翻倍）；shared 绑定配置下则实打实掉驻留 block，须用隐藏账判值不值。

> **坑**：把双缓冲当带宽优化。触发条件：期望双缓冲突破 $t_{\mathrm{load}}$ 地板；现象：式 (7) 的 $\max$ 里 $t_{\mathrm{load}}$ 一项纹丝不动——双缓冲藏的是等待（延迟与串行段），不是流量；带宽地板由强度与 L2 复用决定（式 (2)/5.2）；规避：先判 $I_{\mathrm{blk}}$ 对 $R$ 的位置，load-bound 时优先提强度（大 tile/L2 调度）或加 stage 数（4.4），双缓冲只是流水线的最小份数。

## 7. 教学 GEMM：naive 对 tiled 的强度对比

两个 kernel 差多少？教学片段（机制片段级；bf16 输入 fp32 累加， $M, N, K$ 被 tile 整除的教学假设，边界处理略）：

```cpp
// 教学片段，非优化实现：naive GEMM（每线程 1 输出，O(1) 复用）
__global__ void gemm_naive(const __nv_bfloat16* A, const __nv_bfloat16* B,
                           float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int k = 0; k < K; k++)
        acc += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    C[row * N + col] = acc;
}
```

```cpp
// 教学片段，非优化实现：shared tiling + register tiling（128x128x32，256 线程，每线程 8x8）
constexpr int TM = 128, TN = 128, TK = 32, RM = 8, RN = 8;
__global__ void gemm_tiled(const __nv_bfloat16* A, const __nv_bfloat16* B,
                           float* C, int M, int N, int K) {
    __shared__ __nv_bfloat16 As[TM][TK];   // A tile 转置存：行 m 列 k，装载 k 最快
    __shared__ __nv_bfloat16 Bs[TK][TN];
    int ty = threadIdx.x / 16, tx = threadIdx.x % 16;   // 16x16 线程网格
    int row0 = blockIdx.y * TM + ty * RM, col0 = blockIdx.x * TN + tx * RN;
    float acc[RM][RN] = {};                             // 64 个 fp32 累加器
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int i = threadIdx.x; i < TM * TK; i += 256) {   // 协作装载 A tile
            int k = i % TK, m = i / TK;                      // k 最快 → global 合并
            As[m][k] = A[(blockIdx.y * TM + m) * K + k0 + k];
        }
        for (int i = threadIdx.x; i < TK * TN; i += 256) {   // 协作装载 B tile
            int k = i / TN, n = i % TN;                      // n 最快 → global 合并
            Bs[k][n] = B[(k0 + k) * N + blockIdx.x * TN + n];
        }
        __syncthreads();
        for (int k = 0; k < TK; k++) {
            float a[RM], b[RN];
            for (int m = 0; m < RM; m++) a[m] = __bfloat162float(As[ty * RM + m][k]);
            for (int n = 0; n < RN; n++) b[n] = __bfloat162float(Bs[k][tx * RN + n]);
            for (int m = 0; m < RM; m++)
                for (int n = 0; n < RN; n++) acc[m][n] += a[m] * b[n];   // 外积
        }
        __syncthreads();
    }
    for (int m = 0; m < RM; m++)
        for (int n = 0; n < RN; n++) C[(row0 + m) * N + col0 + n] = acc[m][n];
}
```

强度对比表（本章口径，式 (1)-(4)）：

| 口径 | naive | tiled（128×128×32 + 8×8） |
| --- | --- | --- |
| 每输出 global 读 | $2K = 8192$ 元素 | $(T_M+T_N)K/(T_M T_N) = 64$ 元素 |
| 复用因子 | 1 | A×128、B×128（shared 级）；A×8、B×8（register 级） |
| 强度 | 0.5 FLOP/byte | 64 FLOP/byte（HBM 侧） |
| HBM 流量 | 274.88 GB | 2.147 GB（+C 写 33.55 MB） |
| 带宽口径时间 | 82.00 ms | 640.7 μs（+C 写 650.7 μs） |

逐相位走读（tiled 版）：相位一协作装载，256 线程各搬 32 个元素（A tile 4096 + B tile 4096 = 8192 元素 ÷ 256），global 侧 k/n 最快保合并；相位二 `__syncthreads()` 后全 block 可见（3.2 同步语义 recap）；相位三每线程沿 k 做 32 步外积，每步读 8+8 个 shared 值、算 64 次乘加，累加器全程驻留 register；相位四再同步后换下一 K 步，末步写回 64 个 fp32。两个 `__syncthreads()` 各挡一次「写后读」与「读后写」依赖——少了第二个，下一 K 步的装载会覆盖还在被读的 tile（正确性边界，不是性能提示）。

装载侧判定（4.1 式 (1) recap）：tiled 版 A tile 装载 k 最快、B tile 装载 n 最快，warp 内连续地址 → 各 4 sector/128 B， $\eta = 1$；naive 版 B 读 coalesced、A 读 warp 内同址 broadcast（§2）。shared 侧冲突账（4.2 命题 1/2 recap）：**教学片段朴素布局的 A/B 读各带 2 路冲突**——As 读：warp 内两个 ty 行相距 8 行、行距 16 字， $8 \times 16 = 128$ 字恰为 32 的倍数，两 ty 落同 bank 不同字（同 ty 16 线程同址走 broadcast，不增重数）；Bs 读：lane tx 的字偏移含 $4tx$ 项，tx 与 tx+8 同 bank 不同字（练习 4(e) 逐 lane 验证）。冲突使服务周期乘 2、流量不变，故强度表不受影响（吞吐账与流量账分离，4.2 口径）；消除用 swizzle（4.2 §4，行距保 128 B 倍数）或改布局，向量化/Tensor Core 路径的完整布局走读归 5.2。官方教学例对照：cuda-samples `cpp/0_Introduction/matrixMul`（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）与 BP §10.2.3.2 三级例同形，可与本节账对跑。

## 8. 边界：本章不做什么

强度账算完，剩下的归谁？本章交付 naive 强度（命题 1 + 式 (1)）、shared tiling 强度与总流量（命题 2 + 式 (2)(3)，无 L2 复用口径）、register tiling 与三级层级账（式 (4) + §4 表）、tile 三预算与方阵最优（式 (5) + 命题 3）、双缓冲条件与加速比（命题 4 + 式 (6)-(8)）、教学 GEMM 对比（§7）；以下各有专章。异步搬运与 multi-stage 流水归 4.4（本章命题 4 是它 $n_{\mathrm{stage}} = 2$ 的特例）；L2 跨 block 复用、rasterization、split-K 与 CUTLASS 模块全量走读归 5.2（本章式 (2)(3) 是它的理论 recap 源）；FA 的 tile 预算约束（Q/K/V 三份独立 tile）归 6.3；bank conflict 消除细节归 4.2；occupancy 精确计算归 3.4；实测 drill-down 归 11.2。代码 scope：机制片段级（§7），不做模块走读。

本章交付清单（下游 recap 接口）：命题 1/2 + 式 (1)-(3)（§2/§3）、式 (4) 层级账（§4）、式 (5) + 命题 3 预算约束（§5）、命题 4 + 式 (6)-(8) 双缓冲（§6）。4.4 recap 命题 4 作 multi-stage 的 $n_{\mathrm{stage}} = 2$ 基线、recap 式 (5) 作 stage 数预算；5.2 recap 命题 2 与式 (3) 作 tile 层级流量账的理论侧、recap §4 层级账作 CuTe 层级的机制侧；6.3 recap §5 三预算方法作 FA tile 约束的原型。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| naive 每元素 $O(1)$ 复用 → $I = 1/\mathrm{bytes} = 0.5$（bf16），与规模无关；缓存命中不改阶 | 式 (1)，命题 1 | §2、练习 1/2 |
| shared tile 每元素复用 $T_N$/ $T_M$ → $I_{\mathrm{blk}} = 2T_M T_N/((T_M+T_N)\mathrm{bytes})$；方阵 $T_{\mathrm{tile}}/\mathrm{bytes}$，128 bf16 给 64 | 式 (2)，命题 2 | §3、练习 1/3 |
| 总流量 $MNK(1/T_M+1/T_N)\mathrm{bytes}$（每 tile 从 HBM 读、无 L2 复用口径）；128 tile 给 2.147 GB 对 naive 274.88 GB，128× | 式 (3) + §3 表 | §3、练习 2/3 |
| $I = 64 \lt R = 295$：per-block 仍 memory-bound；达 ridge 需 $T \ge 590$ 超预算 → 真实 GEMM 靠 L2 跨 block 复用进 compute-bound（修正归 5.2） | §3 假设注 | §3、练习 2 |
| BP 官方三级锚（V100）：119.9 → 144.4 → 195.5 GB/s；收益来自消除冗余搬运，不是 coalescing | §3 BP 逐字 | §3 |
| register tiling $I_{\mathrm{reg}} = r/\mathrm{bytes} = 4$（r=8）；三级账 HBM 64 / shared 4–14.2 / register 宽裕，瓶颈逐级上移；wgmma B 直喂 shared 是结构解 | 式 (4) + §4 表 | §4、练习 3 |
| tile 三预算：shared $n_{\mathrm{stage}}(T_M+T_N)T_K\mathrm{bytes}$、register $r_M r_N$ 累加器 + 总量、occupancy 四约束取 min；固定周长方阵最优 | 式 (5)，命题 3 | §5、练习 3/4 |
| $T_{\mathrm{tile}} = 256$ 两条路都死：r=16 累加器 256 超 255；r=8 需 1024 线程、block regs 131072 超 65536 | §5 悬崖 | §5、练习 4 |
| 双缓冲 $\tau = \max(t_{\mathrm{comp}}, t_{\mathrm{lat}}+t_{\mathrm{load}})$；完全隐藏 iff $t_{\mathrm{comp}} \ge t_{\mathrm{lat}}+t_{\mathrm{load}}$，必要条件 $I \ge R$；加速比上界 2 | 式 (7)(8)，命题 4 | §6、练习 4 |
| 双缓冲数值：139.9/645.2/906.8 ns，加速 1.154（T_K=32）；容量翻倍但 register 绑定下 occupancy 无损； $T_K$ 不改强度、改延迟账 | §6 表 + 口径注 | §6、练习 4 |
| 教学 GEMM：naive 82.00 ms 对 tiled 640.7 μs（带宽口径）；装载两侧 coalesced，朴素布局 A/B 读各 2 路冲突（服务周期 ×2、流量不变），swizzle 归 4.2 | §7 表 + 冲突账 | §7、练习 4 |

## 练习

1.（计算 · [10]）强度代入（式 (1)(2)）：(a) bf16 方阵 tile $T_{\mathrm{tile}} = 32, 64, 128, 256$ 的 $I_{\mathrm{blk}}$；(b) naive 的 $I$；(c) fp32（ $\mathrm{bytes} = 4$） $T_{\mathrm{tile}} = 128$ 的 $I_{\mathrm{blk}}$，与 bf16 同 tile 比谁大。先预测 (c) 的方向再算。校验：(a) 16/32/64/128 FLOP/byte，与 §3 表一致；(b) 0.5；(c) 32，bf16 的一半（强度反比于 bytes，式 (2)）。
2.（计算 · [20]）流量与时间账（本章口径 $M=N=K=4096$， $\mathrm{BW} = 3352$ GB/s）：(a) naive 与 tiled-128 的 HBM 流量与带宽口径时间；(b) C 写回字节与占 tiled 流量的百分比；(c) 流量比与强度比各多少，两者什么关系；(d) tiled-128 时间对算力上限 138.9 μs 的比值，用 $R/I$ 核对。先预测 (c) 再算。校验：(a) 274.88 GB/82.00 ms 对 2.147 GB/640.7 μs（§2/§3）；(b) 33.55 MB、1.56%；(c) 同为 128（流量比 = 强度反比）；(d) 4.61 ≈ 295/64（roofline 斜线区，2.4 recap）。
3.（推导 · [30]）强度公式补全与推广：(a) 证命题 3（固定 $T_M + T_N$ 时方阵 tile 强度最大），并构造 $T_M = 128, T_N = 256$ 对同周长方阵 192 的数值反例；(b) 把式 (2) 写成调和平均形 $I = \frac{2}{1/T_M + 1/T_N} \cdot \frac{1}{\mathrm{bytes}}$ 并验证与式 (2) 恒等；(c) 推导式 (4)（register 级强度），说明 $T_K$ 为何在两级强度式里都约去；(d) 推导式 (3)（block 数求和），并给出线程数闭合约束 $`\mathrm{threads}_{\mathrm{block}} \cdot r_M r_N = T_M T_N`$，验证本章配置。校验：(a) 85.33 低于 96（§5）；(b) 代入 128/256 两形同给 85.33；(c) r=8 bf16 给 4，与 §4 一致；(d) 2.147 GB 与 §3 表一致、 $256 \times 64 = 16384$。
4.（计算 · [30]）预算与双缓冲综合账（ $T_M = T_N = 128$、256 线程、 $R_{\mathrm{th}} = 128$ 教学假设）：(a) $T_K = 64$ 双缓冲的 $S_{\mathrm{sh}}$、 $S_{\mathrm{alloc}}$（128 B 取整 + 1 KB 预留）与 shared 项 blocks；(b) 四约束取 min 的 $b_{\mathrm{act}}$ 与 $\mathcal{O}$，指出绑定约束；(c) $T_K = 64$ 的 $t_{\mathrm{comp}}$、 $t_{\mathrm{load}}$、 $\tau_{\mathrm{db}}$ 与加速比；(d) $T_{\mathrm{tile}} = 256$ 两条路（r=16 保 256 线程 / r=8 保线程数）各撞哪个上限；(e) tiled 片段 shared 读冲突账：As 读（固定 $m = 0, k = 0$，warp 内两 ty）的两个字号与 bank 号、Bs 读（ $n = 0$、tx = 0..15）前 8 个 tx 的 bank 号序列，各判冲突重数。先预测 (c) 加速比对 $T_K = 32$ 的方向再算。校验：(a) 64 KB、66560 B、3 blocks（3.4 式 (1) 口径）；(b) 2 blocks、0.25、register 绑定（§5）；(c) 279.8/1290.4/1552.0 ns、1.180（§6 表）；(d) 256 超 255 与 131072 超 65536（§5）；(e) As 字号 0 与 128 同落 bank 0、2 路；Bs bank 号 $4tx \bmod 32$ = 0,4,8,...,28 循环、2 路（4.2 命题 1/2）。

## 参考

- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——§10.2.3.2 Shared Memory in Matrix Multiplication (C=AB)（naive/A tile/B tile 三级教学例 + V100 119.9/144.4/195.5 GB/s + 「the performance improvement is not due to improved coalescing in either case, but to avoiding redundant transfers from global memory」逐字 + A 同址 broadcast 与 cache line 逐出逐字 + 「a value in shared memory is broadcast to all threads in a warp」逐字）
- [NVIDIA, CUTLASS docs](../references.md)：CUTLASS media/docs/cpp 文档（key: cutlass-docs），教学参照 + 官方实现，2026-11-25——efficient_gemm.md（GEMM 层级嵌套循环 threadblock-level concurrency → warp-level parallelism → instruction-level parallelism；「A larger threadblock means fewer fetches from global memory, thereby ensuring that DRAM bandwidth does not become a bottleneck.」逐字 + 大 tile 形状失配与 split-K 段 + `ThreadblockShape::{kM, kN, kK}`；旧 gemm_tutorial.md 已不存在，现行锚 efficient_gemm.md）；模块全量走读归 5.2
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/0_Introduction/matrixMul（shared tiling GEMM 官方教学例，§7 对照；现行 cpp/ 路径）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——HBM 478.8 cycles 延迟实测口径（H800 微基准 Table IV，非 H100 官方规格，沿 2.2/2.3 声明；§6 $t_{\mathrm{lat}}$ 换算输入）
