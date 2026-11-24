# 2.2 缓存层级（Cache Hierarchy）

> 同一条 global load，为什么 L1 命中 40.7 个周期就回数据，miss 到 HBM 却要 478.8 个周期？本章交付缓存层级的延迟账（命中率加权 AMAT，实测 [B] 口径）、L1/shared 统一的缓存口径（line 128 B / sector 32 B）与 sector 有效带宽账、L2 两分区（H100 50 MB，partitioned crossbar）与跨分区访问、L2 persistence（cudaAccessPolicyWindow）、一致性模型（片内 L2 一致、跨 GPU 非一致、Blackwell 双 die NV-HBI 例外）与常量/纹理缓存路径 | 前置：1.2（shared/L1 统一 256 KB、carveout、228/227/256 口径）、1.4（延迟表实测口径、Little's law 并发）、2.1（HBM 带宽式 (1)、MC/通道库存）；下游 2.3（片上存储总表与常量/纹理编程细节）、2.4（层级带宽账与测量）、3.2（内存模型 fence/atomic）、4.1（coalescing 判定）、7.1/7.3（NVLink 一致性与 P2P） | 锚点：Hopper whitepaper Fig 6 / L2 prose（[NVIDIA, Hopper whitepaper](../references.md)，[C] 官方规格披露·一手权威）、CUDA Programming Guide memory / l2-cache-control 子页（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）、Blackwell 微基准（[Jarmusch et al., 2025](../references.md)，arXiv:2507.10789，[B]）、Blackwell Technical Brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、Blackwell Tuning Guide（[NVIDIA, Blackwell tuning guide](../references.md)，[C]）、Nsight Compute memory chart（[NVIDIA, Nsight Compute](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 40.7 对 478.8 个周期：命中决定延迟

同一条 global load，L1 命中与 miss 到 HBM 的实测延迟差 11.8 倍（40.7 对 478.8 cycles，[Luo et al., 2024](../references.md)，arXiv:2402.13499，H800 微基准 Table IV，[B]）。硬件不区分「重要的数据」——每条访问的延迟由它命中哪一层决定。本章回答三个问题：平均延迟怎么从命中率算出来，上下界是什么？32 B 的 sector 粒度让散乱访存的有效带宽掉多少？L2 分区、persistence、一致性模型各管什么，边界在哪？

**本章口径**（章脊锁定）：主口径 **H100 SXM**（132 SM，L2 50 MB 两分区，每 SM shared/L1 统一 256 KB，recap 1.2）；GB200（L2 126 MB）作世代对比行，表外不再换。**实测口径声明**（沿 1.4）：精确 cycles（shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8）来自 H800 微基准 [B]，非 H100 官方规格——官方 Best Practices 只给定性「hundreds of clock cycles」（[NVIDIA, Best Practices](../references.md)，[C]）；周期 ↔ 时间换算取 TC-boost 1830 MHz（BF16 kernel 口径，沿 1.4）。单位口径：片上 KB = 1024 B（Programming Guide 逐字注）；L2 50 MB 沿 whitepaper 印法（MB 行不强行二进制换算）；显存 GB = $10^{9}$ B（2.1）。

符号声明： $h_1$ = L1 命中率； $h_2$ = L2 命中率（**条件于 L1 miss**）； $t_{\mathrm{L1}}$、 $t_{\mathrm{L2}}$、 $t_{\mathrm{HBM}}$ = 各层命中（或到达）的全程 load-to-use 延迟（cycles）； $s$ = 一条 warp load 触及的 distinct sector 数（1.4 的 $s_{\mathrm{warp}}$ 是其 coalesced 特例 = 4）； $\eta$ = 有用字节占比； $\mathrm{BW}$ = HBM 标称带宽（2.1 式 (1)）。

**边界声明**：本章交付缓存层级的机制与延迟/带宽账。编程模型侧内存模型（fence/atomic 语义）归 3.2；coalescing 判定规则与 stride 饱和归 4.1；常量/纹理容量与编程细节、register spill 归 2.3；层级带宽-延迟-容量全表与测量归 2.4；NVLink/P2P transport 归 7.1/7.3。代码 scope：机制片段级（§5 L2 persistence API 片段）。

## 2. 层级总览与命中延迟加权账

一条 global load 从 SM 出发，最远要走到哪？路径与各级延迟（实测 [B]，容量 [C]）：

```
SM（×132）
 ├─ register file 256 KB ── operand 读出在执行管线内（1.2 §3，非 load 延迟口径）
 ├─ shared/L1 统一 256 KB（carveout 切分，1.2 §4）
 │    shared 访问 29.0 cycles / L1 命中 40.7 cycles [B]
 │        │ L1 miss（按 sector 32 B 粒度向下取数）
 ▼        ▼
L2 分区 0（25 MB）◄────► L2 分区 1（25 MB）
 │  partitioned crossbar（Fig 6）；L2 命中 263.0 cycles [B]
 │        │ L2 miss
 ▼        ▼
MC ×10（每控制器 512-bit，2.1 §5）──► HBM3 5 栈 × 16 ch
                                     478.8 cycles [B] / 3352 GB/s（2.1 式 (1)）
```

延迟表（recap 1.4 §2 表的缓存行；延迟实测 [B]，容量官方 [C]）：

| 层 | 容量（H100 SXM） | 命中延迟（实测 [B]） | 带宽口径 |
| --- | --- | --- | --- |
| shared | 228 KB/SM（carveout 最大档，1.2） | 29.0 cycles | 128 B/cycle/SM（32 banks × 4 B，1.2） |
| L1 | 256 KB 统一 − shared carveout | 40.7 cycles | 官方未披露每周期带宽（测量归 2.4） |
| L2 | 50 MB（两分区各 25 MB） | 263.0 cycles | 同上 |
| HBM | 80 GB（2.1） | 478.8 cycles（L2 miss） | 3352 GB/s（2.1 式 (1)） |

**口径注**：各层延迟是**全程 load-to-use 口径**（从发射到数据返回），不是层间增量——263 是 L2 命中的全程延迟，478.8 是 L2 miss 到 HBM 的全程延迟；加权账用「全程延迟 × 命中概率」，不能把 263 与 478.8 相加。register 层不列入：register 访问是执行管线内的 operand 读出（1.2 §3 带宽账），不是独立的 load 延迟口径。

平均延迟怎么算？推导路线分三步：把全部访问按命中层级分成三个互斥类（L1 命中 / L1 miss 但 L2 命中 / 两级都 miss），写出每类的概率，再按各层全程延迟加权求和。关键一步是概率相乘——L2 命中的概率是 $(1-h_1) h_2$，因为 $h_2$ 是条件于 L1 miss 的命中率。

$$
t_{\mathrm{avg}} = h_1 t_{\mathrm{L1}} + (1-h_1) h_2 t_{\mathrm{L2}} + (1-h_1)(1-h_2) t_{\mathrm{HBM}} \qquad (1)
$$

**符号**： $t_{\mathrm{avg}}$ = 平均访存延迟（cycles）； $h_1$、 $h_2$、 $t_{\mathrm{L1}}$、 $t_{\mathrm{L2}}$、 $t_{\mathrm{HBM}}$ 见 §1。三个权重 $h_1$、 $(1-h_1) h_2$、 $(1-h_1)(1-h_2)$ 非负且和为 1——式 (1) 是三个延迟的凸组合。

全数字实例（ $t_{\mathrm{L1}} = 40.7$、 $t_{\mathrm{L2}} = 263.0$、 $t_{\mathrm{HBM}} = 478.8$，实测 [B]；换算 @1.83 GHz）：

| $h_1$ | $h_2$ | $t_{\mathrm{avg}}$（cycles） | 折合 ns |
| --- | --- | --- | --- |
| 0.5 | 0.5 | 205.8 | 112.5 |
| 0.8 | 0.5 | 106.7 | 58.3 |
| 0.9 | 0.5 | 73.7 | 40.3 |
| 0 | 0 | 478.8 | 261.6 |
| 1 | — | 40.7 | 22.2 |

代入演示（ $h_1 = h_2 = 0.5$）： $0.5 \times 40.7 + 0.25 \times 263 + 0.25 \times 478.8 = 20.35 + 65.75 + 119.7 = 205.8$ cycles。全 miss（ $h_1 = h_2 = 0$）退化为 478.8，与 1.4 §4 的 261.6 ns 换算逐字一致。

**命题 1**（AMAT 界与单调性）。当 $0 \le h_1, h_2 \le 1$ 且 $t_{\mathrm{L1}} \lt t_{\mathrm{L2}} \lt t_{\mathrm{HBM}}$ 时，式 (1) 满足 $t_{\mathrm{L1}} \le t_{\mathrm{avg}} \le t_{\mathrm{HBM}}$，且 $t_{\mathrm{avg}}$ 对 $h_1$、 $h_2$ 各自单调不增。**证明**（要点）：三个权重非负、和为 1，凸组合落在最小与最大项之间，得界；单调性用差值法——固定 $h_2$， $t_{\mathrm{avg}}$ 是 $h_1$ 的线性函数，斜率 $t_{\mathrm{L1}} - [h_2 t_{\mathrm{L2}} + (1-h_2) t_{\mathrm{HBM}}]$ 由延迟序为负。完整证明留作练习 3。

数值落点： $h_1$ 从 0.5 提到 0.9（ $h_2$ 固定 0.5），平均延迟从 205.8 降到 73.7 cycles，降 64%——这是 tiling 的延迟侧收益（把 HBM 访问换成 shared/L1 访问，归 4.3）；带宽侧收益是 §3 的 sector 账。两个收益是同一件事的两面：命中替代 miss，有用字节替代浪费的 sector。

与 1.4 的连接（recap，不重推）：本章 $t_{\mathrm{avg}}$ 是 1.4 并发账的延迟侧输入——平均延迟越低，藏住它所需的在飞并发越少（Little's law，1.4 式 (1)）。1.4 按全 miss 口径 478.8 cycles 算出每 SM 需 ≈208 个 sector（52 条 warp load）在飞，那是 $h_1 = h_2 = 0$ 的最坏并发需求；缓存命中把实际需求压到最坏值之下。

命中还有带宽侧的收益，与延迟侧互补。每笔 L1/L2 命中都省掉一次 HBM 访问：到达 HBM 的请求比例 = $(1-h_1)(1-h_2)$。代入 $h_1 = 0.8$、 $h_2 = 0.5$：只有 $0.2 \times 0.5 = 0.1$（10%）的请求到达 HBM——同样的总请求率下，HBM 带宽占用降到 1/10，省出的带宽余量可以喂更大的工作集或更高的请求率。所以命中率是延迟账（式 (1)）与带宽账（HBM 流量比例）的共同输入：tiling 把访问往上层搬，两笔账同时改善（4.3）。

## 3. L1/shared 统一与 line/sector 粒度

L1 与 shared 共用一块 SRAM，缓存侧的粒度是什么？recap 1.2 §4（不重推）：L1 data cache 与 shared memory 统一物理 256 KB，carveout 档位 {0, 8, 16, 32, 64, 100, 132, 164, 196, 228} KB，最大档 shared 228 KB、L1 剩 28 KB；每 block opt-in 上限 227 KB（1 KB 预留）。本节补缓存侧的两个粒度：line 与 sector。

所谓 cache line（缓存行），是指缓存的 tag 管理粒度——H100 的 L1/L2 line 为 128 B；所谓 sector，是指数据搬运粒度——32 B，一条 line 含 4 个 sector（[NVIDIA, Nsight Compute](../references.md) memory chart 口径，[C]）。miss 时按需要的 sector 取数，不必搬整条 line：warp load 只触及 1 个 sector 就只搬 32 B，不是 128 B。line 管「在哪」（tag 查找），sector 管「搬多少」（流量账）——两个粒度各管一件事。

```
一条 128 B cache line = 4 个 32 B sector
┌───────────────┬───────────────┬───────────────┬───────────────┐
│  sector 0     │  sector 1     │  sector 2     │  sector 3     │
│  [0,32) B     │  [32,64) B    │  [64,96) B    │  [96,128) B   │
└───────────────┴───────────────┴───────────────┴───────────────┘
coalesced warp load（32 线程 × 4 B 连续）：覆盖 4 个 sector，搬 128 B，全有用
散乱 warp load（每线程隔 128 B）：触及 32 个 sector，搬 1024 B，只有 128 B 有用
```

sector 带宽账的推导路线：先算一条 warp load 的有用字节，再算触及的 distinct sector 数（下层流量 = 32 B × sector 数），两者之比即有用字节占比。关键一步是**流量按 sector 计、不按 line 计也不按地址跨度计**——只触及一个 sector 的 line 不搬整条。

**命题 2**（sector 流量下界）。一条 warp load 的下层流量为 $32 s$ 字节（ $s$ = 触及的 distinct sector 数），有用字节占比 $\eta \le 1$，等号当且仅当搬到的每个字节都有用。**证明**：sector 是最小搬运粒度，每个被触及的 sector 整体搬 32 B，故流量 $= 32 s$；有用字节不超过流量，故 $\eta \le 1$；等号条件即无浪费字节。∎

$$
\mathrm{bytes}_{\mathrm{traffic}} = 32 s, \qquad \eta = \frac{\mathrm{bytes}_{\mathrm{useful}}}{32 s} \qquad (2)
$$

$$
\mathrm{BW}_{\mathrm{eff}} = \eta \cdot \mathrm{BW} \qquad (3)
$$

**符号**： $`\mathrm{bytes}_{\mathrm{traffic}}`$ = 下层搬运字节； $`\mathrm{bytes}_{\mathrm{useful}}`$ = kernel 实际用到的字节； $s$ = distinct sector 数； $\eta$ = 有用字节占比； $`\mathrm{BW}_{\mathrm{eff}}`$ = 有效有用带宽（口径注：「有效」指有用字节口径，不是官方标称带宽的修正）； $\mathrm{BW}$ = 标称带宽 3352 GB/s。

全数字两模式对照（fp32 warp load，32 线程 × 4 B = 128 B 有用）：

- **coalesced**（地址连续）：触及 $128/32 = 4$ 个 sector，流量 $32 \times 4 = 128$ B， $\eta = 1$——recap 1.4 的 $s_{\mathrm{warp}} = 4$。
- **全散乱**（每线程隔 128 B）：每线程落不同 sector， $s = 32$，流量 $32 \times 32 = 1024$ B， $\eta = 128/1024 = 0.125$（12.5%）。代入式 (3)：有效有用带宽 $3352 \times 0.125 = 419$ GB/s——标称带宽没变，有用带宽掉到 1/8。

这就是「同一个 3.35 TB/s，不同 kernel 拿到不同有效带宽」的带宽侧代价。coalescing 的判定规则（什么地址模式合并成几个 sector、stride 饱和）归 4.1，本节只交付粒度机制与账式。sector 口径也解释了 1.4 式 (2) 服务率为什么除 32 B：HBM 队列按 sector 服务，不按 line、不按字节。

> **坑**：流量按 line 数或地址跨度算。触发条件：把流量算成「触及 line 数 × 128 B」或「地址跨度向上取整」；现象：散乱访存流量高估（每 line 只触及 1 个 sector 却按整条 128 B 计，高估 4×），coalescing 收益误判；规避：流量 = distinct sector 数 × 32 B（命题 2），line 只是 tag 粒度；完整 coalescing 判定归 4.1（distinct sector 口径）。

## 4. L2 两分区：locality 与跨分区访问

H100 的 50 MB L2 为什么不是一整块？whitepaper Fig 6 画的是左右两个 L2 Cache 块、中间双向箭头（[NVIDIA, Hopper whitepaper](../references.md)，[C]）；微基准互证逐字「In the GH100, the L2 cache is divided into two independent partitions」（[Jarmusch et al., 2025](../references.md)，arXiv:2507.10789，[B]），对照消费级 GB203 的单片 L2。分区的机制理由，whitepaper prose 逐字：「Using a partitioned crossbar structure, the L2 cache localizes and caches data for memory accesses from SMs in GPCs directly connected to the partition.」——每个分区优先服务直连 GPC 的 SM，把访问 localize，缩短 crossbar 路径。

分区带来三个后果：

- **locality 与跨分区延迟**：SM 访问本地分区路径短；跨分区访问经分区间互连（Fig 6 双向箭头），延迟更高。跨分区的精确 cycle 数官方与微基准均未逐字披露——Table IV 的 263 cycles 是 L2 命中的实测口径（未分区解析），本书不推定跨分区增量。
- **容量份额与 camping**：每分区 25 MB。若工作集集中映射到一个分区（partition camping 的 L2 层表现，机制 recap 2.1 §5），可用容量减半、命中率下降。实例：工作集 30 MB，总容量 50 MB 装得下；但若映射集中到一个分区（25 MB），至少 5 MB 被挤出到 HBM， $h_2$ 下降、 $t_{\mathrm{avg}}$ 上升（式 (1)）。地址 hash 属闭源，最坏账按 2.1 的简化模型口径。
- **世代增量**：GB200 的 L2 为 126 MB（[NVIDIA, Blackwell tuning guide](../references.md)，[C]），是 H100 50 MB 的 2.52 倍——更大 L2 装得下更大工作集，把式 (1) 的 $h_2$ 推高、平均延迟压低。Blackwell 双 die 的 L2 一致性经 NV-HBI 维持（§6）。

容量-延迟权衡贯穿整个层级：shared 29 cycles 但每 SM 只有 228 KB，L2 263 cycles 但全芯片共享 50 MB，HBM 478.8 cycles 但有 80 GB——每往下一层，容量大一个量级以上，延迟逐级抬升（29 → 263 → 478.8，先约 9 倍再约 1.8 倍）。没有「又快又大」的层，所以 kernel 优化的主线是把工作集沿金字塔往上搬：tiling 搬进 shared（4.3）、persisting 留在 L2（§5）、只读数据走常量/纹理旁路（§7）。本章的延迟表与命中率账就是这条主线的度量尺。

## 5. L2 persistence：access policy window

L2 默认驱逐不区分「马上还要用的数据」——流式访问会把热数据挤出去。CUDA 给的办法是 access policy window：把一段地址标成 persisting，L2 为它留一块 set-aside 区域，热数据不被流式访问驱逐（[NVIDIA, CUDA Programming Guide](../references.md) l2-cache-control 子页，[C]）。

机制三参数：

- **set-aside**：`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` 划出 L2 的一部分专供 persisting 访问；上限是设备属性 `cudaDevAttrMaxPersistingL2CacheSize`（运行时查询，本书不断言具体值）；剩余部分服务正常访问。
- **window**：`cudaAccessPolicyWindow` 指定 base_ptr + num_bytes（上限 `cudaDevAttrMaxAccessPolicyWindowSize`）+ hitRatio + hitProp/missProp，设在 stream 或 kernel launch 上。
- **hitRatio**：window 内访问被标 persisting 的比例（概率性选取）——不是「数据驻留比例」。window 大于 set-aside 时，hitRatio 低于 1 让期望驻留足迹装进 set-aside，避免「都想留、谁都留不住」的 thrashing。

代码片段（教学用，非优化实现；机制片段级）：

```cuda
// L2 persistence 窗口设置（机制例证，非优化实现）
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_bytes);
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = hot_ptr;
attr.accessPolicyWindow.num_bytes = hot_bytes;    // ≤ 设备属性上限
attr.accessPolicyWindow.hitRatio  = 0.6f;         // 概率性标 persisting
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

全数字实例（教学假设口径：set-aside 取 25 MB，即 50 MB 的一半；实际上限以设备属性查询为准）：window 覆盖 20 MB 热数据、hitRatio 0.5 → 期望 persisting 足迹 $20 \times 0.5 = 10$ MB，装进 25 MB set-aside；hitRatio 提到 1.0 → 足迹 20 MB 仍装得下。若 window 是 40 MB，hitRatio 至多 $25/40 = 0.625$ 才能让期望足迹不超 set-aside。set-aside 与正常区是零和切分——set-aside 越大正常区越小，与 shared/L1 carveout 同构（1.2 §4）。

两句边界：persistence 是驱逐优先级 hint，不是驻留保证——硬件不承诺 window 数据永不驱逐；哪些数据值得标 persisting（KV cache 热页、embedding 表）是 serving/kernel 侧决策，归 inference 与 4.x，本节只交付硬件机制与 API 语义。

## 6. 一致性模型：片内一致、跨 GPU 非一致

SM 0 写了一个 global 地址，SM 1 能读到新值吗？片内可以，但不是无条件的；跨 GPU 则没有硬件保证。一致性模型分四层：

- **L1 之间非一致**：L1 每 SM 私有（132 份），相互之间没有硬件一致性协议——SM 0 的 L1 缓存的旧值不会被 SM 1 的写自动失效。跨 SM 可见性由内存模型语义保证（fence/atomic，编程模型侧归 3.2）。
- **L2 是全芯片一致点**：所有 SM 的 global 访问汇聚到 L2（两分区构成统一地址空间），「GPU 内部 L2 一致」的口径指这一层。
- **跨 GPU 非一致**：独立 GPU 封装之间没有硬件 cache 一致性协议；P2P 访问（NVLink load/store 或 copy）不自动失效对方缓存，可见性由程序员/框架管理（transport 语义归 7.3 GPUDirect P2P）。
- **Blackwell 双 die 例外**：B200 封装内两 die 经 NV-HBI 10 TB/s 互连，brief 逐字「providing one fully coherent chip」（[NVIDIA, Blackwell brief](../references.md)，[C]）——封装内一致。所以「跨 GPU 非一致」指独立封装之间，不是封装内 die 之间（recap 1.1 §5：NV-HBI 是 die-to-die，勿与 NVLink-C2C 的 Grace↔Blackwell CPU-GPU 互连混；CPU-GPU 一致性归 7.1）。

机制账一笔：若 L1 跨 SM 一致，每次写都要失效其余 131 个 SM 的副本，一致性流量随 SM 数增长；把一致点放在 L2，L1 保持私有、快（40.7 cycles），跨 SM 可见性收敛到 L2 一层。典型模式：SM 0 写 flag、SM 1 自旋读 flag——若读侧命中自己 L1 的旧值又没有 fence/atomic 语义，自旋可能永远读不到新值；volatile/atomic 访问绕过 L1 旧值，语义归 3.2。

## 7. 常量/纹理缓存路径

global 访存除 L1/L2 主路径外还有两条旁路：常量与纹理。

- **常量路径**：constant memory 64 KB（[NVIDIA, CUDA Programming Guide](../references.md) Table 34，[C]），经每 SM 的常量缓存访问（Nsight memory chart 的 DCC/IDC/IMC 常量缓存层级，[NVIDIA, Nsight Compute](../references.md)，[C]）。broadcast 优化：warp 32 线程读同一地址时一次访问广播全 warp；地址分歧则按 distinct 地址串行——最坏 32 个不同地址串行 32 次，与 uniform 读差 32 倍。所以常量路径适合全 warp 共享的只读数据（kernel 参数、超参表）；编程细节归 2.3。
- **纹理路径**：每 SM 4 个 Tex 单元（recap 1.2 Fig 7），纹理缓存只读、为 2D 空间局部性优化、带硬件插值；AI kernel 少用纹理采样，但只读 load 路径（`ld.global.nc` / `__ldg`）走只读缓存——编译器把标注只读的数据（const __restrict__）路由此路径。只读路径不参与写一致性，故只读数据可安全走；有写则须程序员保证读写不重叠（语义归 3.2）。

本节只立两条路径在层级中的位置与粒度；常量/纹理的容量、bank 与编程细节归 2.3，层级总表归 2.4。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 命中决定延迟：shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8 cycles（实测 [B]，全程 load-to-use 口径，非层间增量） | §2 延迟表 | §2、练习 1 |
| AMAT = 三类互斥命中概率 × 各层全程延迟的凸组合； $t_{\mathrm{L1}} \le t_{\mathrm{avg}} \le t_{\mathrm{HBM}}$，对 $h_1$、 $h_2$ 单调不增 | 式 (1)，命题 1 | §2、练习 1/3 |
| $h_1$ 0.5→0.9（ $h_2$ = 0.5）平均延迟 205.8→73.7 cycles（降 64%）；全 miss 478.8 = 1.4 最坏并发口径 | §2 实例表 | §2、练习 1 |
| 到达 HBM 的请求比例 = $(1-h_1)(1-h_2)$； $h_1$ = 0.8、 $h_2$ = 0.5 时仅 10% 到 HBM（带宽侧收益） | §2 带宽段 | §2 |
| L1/shared 统一 256 KB、carveout 最大档 shared 228 KB / L1 28 KB（recap 1.2）；line 128 B = 4 × sector 32 B | §3 + 1.2 §4 | §3 |
| 流量 = distinct sector × 32 B；coalesced η=1，全散乱 η=12.5% → 有效有用带宽 419 GB/s（3352/8） | 式 (2)(3)，命题 2 | §3、练习 2 |
| H100 L2 50 MB 两分区各 25 MB，partitioned crossbar localize 直连 GPC；跨分区延迟更高（精确值未披露） | §4 + Fig 6 | §4 |
| L2 camping：工作集 30 MB 集中单分区 → 至少 5 MB 挤出；GB200 L2 126 MB = 2.52× H100 | §4 实例 | §4、练习 4 |
| L2 persistence：set-aside + window + hitRatio；期望足迹 = num_bytes × hitRatio ≤ set-aside；40 MB window 需 hitRatio ≤ 0.625 | §5 实例 | §5、练习 4 |
| 一致性四层：L1 非一致 / L2 片内一致点 / 跨 GPU 非一致（7.3）/ Blackwell 双 die NV-HBI 封装内一致 | §6 | §6 |
| 常量路径 broadcast（uniform 1 次 vs 分歧最坏 32 次）；纹理/只读路径 ld.global.nc 不参与写一致性 | §7 | §7 |

## 练习

1.（计算 · [10]）AMAT 代入（H100 SXM 主口径，实测延迟 40.7 / 263.0 / 478.8 cycles）：(a) $h_1 = 0.8$、 $h_2 = 0.5$ 时用式 (1) 求 $t_{\mathrm{avg}}$（cycles），按 1.83 GHz 折合 ns；(b) 全 miss（ $h_1 = h_2 = 0$）的延迟与 ns；(c) 求 (b) 对 (a) 的倍数。校验：(a) ≈106.7 cycles ≈ 58.3 ns，与 §2 实例表一致（容差 1%）；(b) 478.8 cycles ≈ 261.6 ns，与 1.4 §4 换算一致。
2.（计算 · [20]）sector 账（fp32 warp load，32 线程 × 4 B）：(a) coalesced 时触及几个 sector、流量与 $\eta$ 各多少？(b) 每线程隔 128 B 的全散乱模式， $s$、流量、 $\eta$ 各多少？(c) 用式 (3) 求 (b) 的有效有用带宽。先预测 (b) 的 $\eta$ 数量级再算。校验：(a) $s = 4$、 $\eta = 1$，与 1.4 $s_{\mathrm{warp}}$ 一致；(b) $\eta = 0.125$（12.5%），与 §3 一致；(c) 419 GB/s = 3352/8。
3.（推导 · [30]）补全命题 1 的证明（AMAT 界与单调性）：(a) 验证式 (1) 三权重非负、和为 1，推出 $t_{\mathrm{L1}} \le t_{\mathrm{avg}} \le t_{\mathrm{HBM}}$；(b) 固定 $h_2$，把 $t_{\mathrm{avg}}$ 写成 $h_1$ 的线性函数，求斜率并证明其为负（用 $t_{\mathrm{L1}} \lt t_{\mathrm{L2}} \lt t_{\mathrm{HBM}}$）；同法对 $h_2$；(c) 数值验证： $h_2 = 0.5$ 时 $h_1$ 从 0 到 1， $t_{\mathrm{avg}}$ 从 370.9 单调降到 40.7。校验：(b) $h_2 = 0.5$ 时斜率 $= 40.7 - 370.9 = -330.2$，为负；(c) 端点与 §2 实例表一致，方向不可反。
4.（计算 · [30]）persistence 与分区综合账（教学假设 set-aside 25 MB）：(a) window 20 MB、hitRatio 0.5，期望 persisting 足迹多少，装得下吗？(b) window 40 MB 时 hitRatio 至多多少才不超 set-aside？(c) GB200 L2 126 MB 是 H100 50 MB 的几倍？(d) 工作集 30 MB 集中映射到单分区（25 MB），至少多少 MB 被挤出到 HBM？校验：(a) 10 MB ≤ 25 MB；(b) 0.625 = 25/40，与 §5 一致；(c) 2.52，与 §4 一致；(d) 5 MB = 30 − 25。

## 参考

- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25 dam-cdn PDF 一手核实——Table 3 L2 50 MB、Fig 6 两 L2 分区（左右各一块、中间双向箭头）、L2 prose「partitioned crossbar structure … localizes and caches data … directly connected to the partition」逐字
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——Table 35 unified 256 KB carveout、Table 34 constant 64 KB、understanding-memory 与 l2-cache-control 子页（accessPolicyWindow / hitRatio / persisting L2 set-aside）、KB = 1024 逐字注
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV 延迟 cycles：shared 29.0 / L1 40.7 / L2 263.0 / global 478.8（实测口径，§2 表与式 (1) 代入）
- [Jarmusch et al., 2025](../references.md)：Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks（key: blackwell-microbench），arXiv:2507.10789，[B] 有 paper 无官方代码，2026-11-25——「In the GH100, the L2 cache is divided into two independent partitions」vs GB203 monolithic L2（§4 分区互证）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NV-HBI 10 TB/s die-to-die「providing one fully coherent chip」逐字（§6 双 die 一致性例外）
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——GB200 L2 126 MB（§4 世代对比）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——memory chart sector 32 B 口径（§3）、DCC/IDC/IMC 常量缓存层级（§7）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——global「hundreds of clock cycles」定性（§1 实测口径声明）
