# 7.2 节点内拓扑与带宽账（Node-Internal Topology and Bandwidth Accounting）

> 单对 P2P 拷贝的峰值是 450 GB/s 单向，不是头条 900——带宽矩阵的每个格子该按哪个方向口径读？8-GPU ring all-reduce 每 rank 发 $2(n-1)/n \cdot M$ 字节，除以哪个带宽才得到时间？本章 recap 7.1（NVLink/NVSwitch 几何与域聚合恒等式）+ 2.4 §6（nvbandwidth 测量口径）+ training 4.1 式 (2)(3)（ring 消息量与 α-β 模型， $d \to n$ 翻译不重推），交付五块增量：HGX 8-GPU 拓扑与 18 links 聚合账、P2P 带宽/延迟账（命题 1 每对上界 + read/write 语义 + 有效带宽折扣）、ring 映射 NVLink 域、PCIe 拓扑与 NUMA affinity 的 min 瓶颈账、nvbandwidth P2P 矩阵测量方法与节点内 vs 机间数量级差 | 前置：7.1（NVLink 五代/方向口径/域聚合）、2.4（层级带宽表与 nvbandwidth 口径）、1.1（SKU 口径）、training 4.1 §2.2（recap 源）；下游 7.3（GPUDirect P2P API 与 ACS/IOMMU）、8.2/8.4（ring 带宽最优性与 NCCL 环构建）、9.2（GDR 与 NIC 瓶颈）、10.2（分层 ring 与 fabric）、11.1（服务器形态） | 锚点：NVLink/NVSwitch blog（[NVIDIA, NVLink/NVSwitch blog](../references.md)，[C] 开放技术博客）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C] 官方规格披露·一手权威）、HGX 规格（[NVIDIA, HGX/DGX](../references.md)，[C]）、datasheet（[NVIDIA, datasheet](../references.md)，[C]）、nvbandwidth（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现）、PCIe 规范（[PCI-SIG, PCIe](../references.md)，[C]）、NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1）——官方规格 2026-11-25 一手核实

## 1. 三个问题与本章口径

nvbandwidth 跑出一张 8×8 的 P2P 带宽矩阵，NVLink 域内非对角格子大约四百 GB/s——这个数是单向还是双向？为什么不是头条 900？ring all-reduce 每 rank 发送 $2(n-1)/n \cdot M$ 字节（training 4.1 式 (2)），除以 900 还是 450 还是别的数才得到通信时间？跨 socket 的 H2D（host-to-device，host 内存到 GPU 显存）拷贝为什么可能比本地慢一截？本章回答三个问题：单对 P2P 的带宽上界由什么决定、方向口径怎么锁？拓扑（NVSwitch 全互连 / PCIe / NUMA）怎么进入集合通信的时间账？带宽矩阵怎么测、怎么读？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（H100 SXM，NVLink4 + 4 颗第三代 NVSwitch，7.1 §3 几何）；B200（NVLink5）作对比行，表外不再换。方向口径沿 7.1 硬规则（spec §4.3）：头条 900 GB/s / 1.8 TB/s 为双向总和，per-link 25 / 50 GB/s 为单向，per-GPU 单向 = 450 / 900 GB/s。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，与 models/inference $N$ = 参数量严格区分；recap training 4.1 时 $d \to n$ 翻译）； $M$ = 被归约/传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数（recap training 4.1 式 (3)）； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s），下标 uni / bidir / eff 标单向 / 双向 / 有效； $\eta$ = 协议效率（本章局部符号）； $n_{\mathrm{NIC}}$ = 节点 NIC 数。 $N$ 专归参数量、 $B$ 专归 batch，本章均不占用。

**实测口径声明**（沿 2.4，spec §6.2 裁定）：本机无 NVIDIA GPU，测量项交付的是方法 + 官方披露口径，不是新跑 benchmark；凡未引官方披露/已发表基准的数值一律标**教学假设口径**。**边界声明**：P2P API（cudaDeviceEnablePeerAccess / cudaMemcpyPeer）与 PCIe ACS/IOMMU 限制归 7.3；ring 带宽最优性下界证明归 8.2；NCCL 环构建与 cost model 代码走读归 8.4；GDR 与 NIC→HBM 瓶颈账归 9.2；机间 fabric 归 10.x。**errata 重判**：training 4.1 §2.2「节点内 NVLink 与跨节点 IB 的有效带宽差一个量级（硬件侧归 cluster 维度）」按 gpu 维度边界裁定重判——节点内有效带宽硬件侧归本章，机间归 9.x/10.x；本章 §3 交付有效带宽的硬件侧账。

## 2. HGX 8-GPU 拓扑：18 links 去哪了

8 GPU × 18 links = 144 条 link，4 颗 NVSwitch × 64 ports = 256 个端口——144 条 link 占掉 144 个交换端口，平均每颗 36 个（7.1 §3 聚合账）。每 GPU 的 18 links 分布到 4 颗 switch 上，逐颗分布属闭源细节、官方未披露，本书只做聚合账；任意 GPU 对可经交换矩阵线速接通是 blog 的非阻塞披露口径（7.1 §3），逐颗分布本书不推定。拓扑如下（示意，非按比例）：

```
GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 │     │     │     │     │     │     │     │
 │18   │18   │18   │18   │18   │18   │18   │18    ← 每 GPU 18 NVLink4 links
 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼      （25 GB/s 单向/link，7.1 式 (2)）
┌────────────────────────────────────────────────┐
│ NVSwitch0   NVSwitch1   NVSwitch2   NVSwitch3  │ ← 4 颗第三代 NVSwitch
│ (64 ports)  (64 ports)  (64 ports)  (64 ports) │   每 chip 3.2 TB/s 双向
│ 非阻塞 crossbar：任意输入端口 ↔ 任意输出端口线速 │
└────────────────────────────────────────────────┘
单对路径：GPU_A → NVSwitch_i → GPU_B（单对峰值 900 GB/s 双向，命题 1）
ring 一跳 = 一条这样的 switch 路径（§4）；PCIe 另走 root complex（§5）
```

两个 SKU 边界注（datasheet 口径，勿混）：其一，**4-GPU HGX H100/H200 无 NVSwitch**（datasheet NVSwitch 行 4-GPU = N/A），GPU 间走直连 P2P——每对带宽按直连均分账（§3 命题 1 的直连情形）；其二，**H100 PCIe 版 NVLink = 600 GB/s 双向**（bridge 连接，datasheet），与 SXM 版 900 GB/s 是不同 SKU，本章主口径一律 SXM。「全互连」指逻辑全连接（任意对可直通），物理接线是星型汇接 + crossbar，不是每对专用链路（7.1 §4 定义）。

## 3. P2P 带宽账：每对上界、方向与有效折扣

单对 P2P 拷贝的峰值是 900、450 还是 128 GB/s？三个数都对，各是一个口径：900 是双向总和峰值，450 是单向拷贝峰值，128 是无 switch 直连均分的每对双向带宽。所谓 P2P（peer-to-peer，点对点），是指 GPU 之间不经 host bounce buffer 的直接数据传输（GPUDirect P2P 的 API 与地址映射语义归 7.3）。本节先证每对上界（命题 1），再钉 read/write 方向语义，最后给有效带宽折扣与延迟账。（口径：注入带宽 = GPU 向互连发送与接收的带宽之和，即双向总和，沿 7.1 §3。）

**命题 1**（每对 P2P 峰值上界与两种达界拓扑）。设 GPU A、B 的双向注入带宽各为 $`\mathrm{BW}_{\mathrm{GPU,bidir}}`$（同构 HGX 口径 900 GB/s），则：

$$
\mathrm{BW}_{\mathrm{pair,bidir}} \le \min\left(\mathrm{BW}_{A,\mathrm{bidir}},\ \mathrm{BW}_{B,\mathrm{bidir}}\right) \qquad (1)
$$

且两种拓扑分别达界：直连全互连均匀拆分时 $`\mathrm{BW}_{\mathrm{pair,bidir}} = \mathrm{BW}_{\mathrm{GPU,bidir}} / (n-1)`$；非阻塞 NVSwitch 全互连时单对可达 $`\mathrm{BW}_{\mathrm{GPU,bidir}}`$。

**证明**（路线：注入上界 → 直连链路计数 → switch 非阻塞披露，三步）：其一，上界——A↔B 的每字节都要占 A 的一条发送方向与 B 的一条接收方向（或反向），故单对双向流量不超过两端各自注入带宽的较小者，式 (1) 对任意拓扑成立。其二，直连情形——每 GPU 的 $n_{\mathrm{link}}$ 条 link 分给 $n-1$ 个 peer，均匀拆分时每对得 $1/(n-1)$ 的注入带宽，即 $`\mathrm{BW}_{\mathrm{GPU,bidir}}/(n-1)`$。其三，switch 情形——链路不再按 peer 切死，crossbar 非阻塞（7.1 §3 blog 逐字披露），单对通信可占满两端全部 link，达式 (1) 上界。∎

全数字代入（ $n = 8$，HGX H100）：直连均分 $900/7 = 128.57 \approx 128$ GB/s 双向每对——blog Table 1 印 128 GB/s（[NVIDIA, NVLink/NVSwitch blog](../references.md)，[C]，取整口径）；NVSwitch 单对 900 GB/s 双向。blog 的两行对照账：2-GPU 通信，直连 128 vs switch 900 GB/s（7×）；8-GPU 全对通信，直连每 GPU 7 × 128 ≈ 900 vs switch 每 GPU 900 GB/s——**总注入两者相同，switch 的增益在单对峰值与调度灵活性，不在总注入**。边界反例（上界的另一侧）：「任意对同时各跑 900」不成立——一个 GPU 同时与 7 个 peer 各跑 900 GB/s 需要 7 × 900 = 6300 GB/s 注入，超出 900 上限；非阻塞承诺的是每 GPU 总注入不随通信对数缩水（7.1 §3 口径注）。

方向语义（read/write，P2P 拷贝的两个方向口径）：write 语义 = 发起方把数据写进目标 HBM，payload 走发起方 → 目标；read 语义 = 发起方读目标 HBM，请求走发起方 → 目标、payload 走目标 → 发起方。所以**单向拷贝峰值 = 450 GB/s**（payload 只占一个方向， $`\mathrm{BW}_{\mathrm{NVLink}}`$ 单向口径），**双向并发拷贝 = 2 × 450 = 900 GB/s**（两方向同时满载）。nvbandwidth 的 read_sm / write_ce 测试族分别对应这两个语义（§6）；双向测试报值是**每方向**带宽、非双向总和（README 逐字，recap 2.4 §6 坑注）——拿双向报值对账头条 900 须 ×2。

有效带宽折扣（硬件侧账，training 4.1 errata 重判的交付）：raw 带宽是信令几何上限（7.1 式 (1)(2)），通信库实际可用的是扣除协议/编码开销的有效值。NCCL 拓扑模型对 SM90 取 per-link **20.6 GB/s 单向**（topo.h SM90_NVLINK_BW，版本锚 2.32.7-1 / commit 351f33fd，[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现；SM100 取 40.1），于是

$$
\mathrm{BW}_{\mathrm{eff}} = n_{\mathrm{link}} \cdot \mathrm{BW}_{\mathrm{link,model}} = 18 \times 20.6 = 370.8 \ \mathrm{GB/s}\ (\text{单向}), \qquad \eta = \frac{20.6}{25} = 82.4\text{％} \qquad (2)
$$

**符号**： $`\mathrm{BW}_{\mathrm{link,model}}`$ = NCCL 带宽模型的 per-link 单向值（含协议/编码折扣）； $\eta$ = 模型效率。口径分离：硬件 raw 25 GB/s（7.1 规格口径）vs NCCL 模型 20.6 GB/s（库内有效口径，8.4 走读 cost model）——两个数不是矛盾，是两层账；本章时间账代入用模型值，规格对账用 raw 值。B200 模型口径：18 × 40.1 = 721.8 GB/s 单向（对比行）。

延迟账：NVLink P2P 延迟官方未披露精确值，测量方法 = nvbandwidth `device_to_device_latency_sm`（pointer chasing，§6）。量级对照（**教学假设口径**，供 α-β 代入）：取 $\alpha = 2$ μs（μs 量级假设）；对比 HBM load-to-use 478.8 cycles @1.83 GHz ≈ 0.26 μs（recap 2.4 §4 实测 [B] 口径）——P2P 访问比本地 HBM 高约一个量级，因为多了互连往返与协议处理。PCIe 路径经 root complex/switch、协议开销更大，延迟更高（定性；精确值测量归 §6）。

α-β 时间账代入（recap training 4.1 式 (3)， $d \to n$ 翻译，不重推）：

$$
T_{\mathrm{ring}} = 2(n-1)\thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \beta \qquad (3)
$$

**符号**： $\alpha$ = 每消息固定延迟； $`\beta = 1/\mathrm{BW}_{\mathrm{eff}}`$ = 每字节传输时间（式 (2) 单向有效口径）。全数字实例（ $n = 8$， $M = 1$ GB = $10^{9}$ B， $\beta = 1/(370.8 \times 10^{9})$ s/B， $\alpha = 2$ μs 教学假设）：带宽项 $2 \times (7/8) \times 10^{9} / (370.8 \times 10^{9}) \approx 4.72$ ms；延迟项 $2 \times 7 \times 2 = 28$ μs = 0.028 ms；合计 ≈ 4.75 ms——带宽项是延迟项的约 169 倍，大消息牢固处于带宽主导 regime（training 4.1 §2.2 结论 recap）。这个 4.72 ms 就是 §4 ring 映射与 7.1 §6 NVLS 对比账的时间基准。

## 4. 拓扑与集合通信：ring 映射到 NVLink 域

ring all-reduce 的 8 个 rank，每一跳走的是哪条物理路径？逻辑环 0 → 1 → … → 7 → 0 的每条边，映射为「源 GPU → NVSwitch crossbar → 目标 GPU」的一条 switch 路径（§2 图）——不是直连链路，因为 HGX 8-GPU 的 link 全汇 NVSwitch。非阻塞保证任意环排列线速接通：每跳单向 raw 450 GB/s、模型有效 370.8 GB/s（式 (2)），每 rank 发送 $2(n-1)/n \cdot M$（recap training 4.1 式 (2)，不重推），时间账即式 (3)。ring 的带宽最优性下界证明归 8.2，NCCL 环构建代码（rings.cc ncclBuildRings）归 8.4。

all-to-all 的流量模式不同：每 rank 向 $n-1$ 个 peer 各发一份（每 peer $M/n$、每 rank 总发送 $(n-1)/n \cdot M$，消息量账归 8.1），通信图是完全图而非环。NVSwitch 非阻塞对任意排列线速——all-to-all 的每 GPU 总注入仍是 900 GB/s 双向；直连全互连下每对只有 128 GB/s，但 7 对并发合计仍 ≈ 900（命题 1 代入）。所以 switch 对 all-to-all 的增益不在稳态总带宽，而在**单对突发与调度灵活性**：任何一对可以在任何时刻占满 900，不必按 peer 预切链路。NVLS（NVSwitch 网内归约，7.1 §6 recap）进一步把归约搬进交换芯片：每 rank 发送从 $2(n-1)/n \cdot M$ 降到 $M$， $n = 8$ 时带宽项从 4.72 ms 降到 2.70 ms（7.1 §6 教学推导口径）；NVLS 算法与 kernel 走读归 8.4。

拓扑路径类型（NCCL 拓扑检测的口径，代码走读归 8.4）：NVLink 直连路径（PATH_NVL）带宽最高；经 CPU root complex（PCIe 树的主机侧根节点，衔接 CPU/内存控制器与 PCIe 设备）的 PCIe 路径（PATH_PHB）带宽受 PCIe 与 NUMA 约束（§5）；经中间 GPU 到 NIC 的路径（PATH_PXN）用于 GPU-NIC 亲和优化（9.2/10.2）。NCCL 按路径类型与带宽搜索环/树图（src/graph/，8.4），本章只交付路径的硬件带宽账。分层 ring（节点内 NVLink 环 + 机间 RDMA 环）的 fabric 工程归 10.2——节点内与机间的数量级差（§7）正是分层设计的算术依据。

## 5. PCIe 拓扑与 NUMA affinity

Gen5 x16 理论 64 GB/s 单向，可跨 socket 的 H2D 拷贝可能只拿到 40 GB/s（教学假设例）——瓶颈不在 PCIe 本身，在路径上多出来的那一跳。本节把 PCIe 路径拆开，给 min 瓶颈账。

PCIe Gen5 x16 = **64 GB/s 单向 / 128 GB/s 双向**（Hopper wp p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」；datasheet Interconnect 行互证；[PCI-SIG, PCIe](../references.md)，[C]，URL 403 残留、数值经 NVIDIA 一手闭合）。H2D 路径：host DRAM → CPU 内存控制器 → root complex → PCIe x16 → GPU。双 socket 服务器是 NUMA（Non-Uniform Memory Access，非一致内存访问）结构：每个 socket 的 CPU 挂本地 DRAM，访问另一 socket 的内存要经 inter-socket link，带宽与延迟都劣于本地。GPU 挂在某个 socket 的 root complex 下；若 host buffer 分配在另一个 socket 的 NUMA node，路径多一跳 inter-socket link——带宽取路径各段的最小值：

$$
\mathrm{BW}_{\mathrm{H2D}} = \min\left(\mathrm{BW}_{\mathrm{PCIe}},\ \mathrm{BW}_{\mathrm{inter\text{-}socket}}\right) \qquad (4)
$$

**符号**： $`\mathrm{BW}_{\mathrm{PCIe}}`$ = GPU 所挂 x16 的单向带宽（Gen5 = 64 GB/s）； $`\mathrm{BW}_{\mathrm{inter\text{-}socket}}`$ = 跨 socket 互连的有效单向带宽（平台相关，官方未披露统一值）。全数字实例（**教学假设口径**：设 inter-socket 有效 40 GB/s）：本地 NUMA 绑定不经 inter-socket link，理论上限 64 GB/s；跨 socket $\min(64, 40) = 40$ GB/s——只剩本地的 62.5%。所以 H2D 密集负载（数据加载、checkpoint 写回）要把 host buffer 绑到 GPU 本地 NUMA node；测量方法 = nvbandwidth `host_to_device_memcpy_ce` 分别在两种绑定下跑（§6），拓扑查询用 `nvidia-smi topo -m`（教学命令片段），绑定用 `numactl --membind`。

PCIe switch 与 P2P 路径：PCIe switch 做 lane 扇出，挂在同一 switch 下的 GPU/NIC 可不经 root complex 直通；GPU↔GPU PCIe P2P 要求同 root complex 或经 PCIe switch，且受 ACS/IOMMU 限制——ACS 把 P2P 流量重定向经 CPU root complex，bisection 带宽下降（CUDA PG 逐字口径，精确定义与限制清单归 7.3）。NIC 亲和同理：GPU↔NIC 同 PCIe switch/root complex 时 GDR 路径最优（9.2 瓶颈账 $`\min(\mathrm{BW}_{\mathrm{NIC}}, \mathrm{BW}_{\mathrm{PCIe}})`$）；rail-optimized 拓扑把每 GPU 配一张同 rail NIC（10.1）。

## 6. nvbandwidth P2P 矩阵：测量方法

8×8 的 P2P 带宽矩阵怎么测出来、对角线为什么不是 P2P？工具与口径 recap 2.4 §6（不重述全量）：nvbandwidth（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现，README 2026-11-25 一手核实）分 CE / SM 两类拷贝（README 逐字「CE copies use memcpy APIs. SM copies use kernels.」）；spin kernel + CUDA events 把 enqueue 开销排除在传输计时外；默认 3 次采样取中位数；SM 拷贝把 copy size 截断到「512 threads/block × SM 数」的整数倍；延迟测试用 pointer chasing 防 prefetch/防 cache 命中。

P2P 矩阵方法：对每对 GPU (i, j) 跑 `device_to_device_memcpy_read_sm` / `device_to_device_memcpy_write_ce` 等测试，输出 $n \times n$ 矩阵。读法三条：**对角线** = 同卡 HBM 拷贝（源与目的都在本地 HBM），payload $M$ 字节对应 HBM 读 $M$ + 写 $M$ = $2M$ 流量，对账 HBM 峰值须 ×2（recap 2.4 §6 坑注）；**非对角线（NVLink 域内）** = P2P 拷贝，单向口径期望值 ≈ 450 GB/s × 效率（**教学期望口径**，非实测报告；raw 拷贝效率与 NCCL 通信库模型效率 82.4% 是两层账，前者以本机实测为准）；**非对角线（PCIe 路径）** 显著更低（理论上限 64 GB/s 单向，再扣协议开销与 NUMA 折扣，§5）。延迟矩阵用 `device_to_device_latency_sm` 逐对跑，口径与 §3 延迟账一致。

命令片段（教学参照，测试名以 README 为准）：

```bash
./nvbandwidth -t device_to_device_memcpy_read_sm   # P2P 带宽矩阵（SM read 语义）
./nvbandwidth -t device_to_device_memcpy_write_ce  # P2P 带宽矩阵（CE write 语义）
./nvbandwidth -t device_to_device_latency_sm       # P2P 延迟矩阵（pointer chasing）
./nvbandwidth -t host_to_device_memcpy_ce          # H2D（配 numactl --membind 对比 NUMA）
```

> **坑**：拿矩阵报值直接对账头条 900 GB/s。触发条件一：把单向 read/write 报值（≈450 × 效率）当双向总和，对账 900 得「利用率 50%」的假象；触发条件二：把 bidirectional 测试报值当双向聚合——README 逐字「The reported bidirectional GB/s represents per-direction bandwidth under concurrent opposite-direction copy, not an aggregated sum.」，报值是每方向带宽；现象：两种误读都差 2 倍；规避：对账前先问「报值是 payload 还是总流量、单向还是双向总和」，NVLink 头条 900 是双向总和，单向报值 ×2 再比（spec §4.3 方向口径，recap 2.4 §6）。

本节交付的是方法 + 官方披露口径（spec §6.2 裁定）：本机无 NVIDIA GPU，不新跑 benchmark；读者在真机跑出的矩阵，用上面三条读法对账 §3 的账。

## 7. 节点内 vs 机间：数量级差决定放置

每 GPU NVLink 单向 450 GB/s，一张 400 Gb/s NIC 的线速是多少？400 ÷ 8 = 50 GB/s——差 9 倍。这条数量级差是并行放置规则的算术依据（training 4.1 §2.2 的硬件侧账，本章交付）。

单位换算与逐笔账（NIC 口径归 9.2，此处只做换算）：400 Gb/s ÷ 8 = **50 GB/s 单向**线速。每 GPU 对比：NVLink4 单向 450 / NIC 50 = **9×**；B200 NVLink5 单向 900 / 50 = **18×**。节点聚合对比（rail-optimized 每 GPU 配一张 NIC，10.1）：节点内 NVLink 注入 8 × 450 = **3600 GB/s 单向**，节点出口 $n_{\mathrm{NIC}}$ × 50 = 8 × 50 = **400 GB/s 单向**——比值仍是 9×。域聚合带宽（双向口径）recap 7.1 式 (4)：HGX H100 7.2 TB/s，折单向 3.6 TB/s = 3600 GB/s——与上面的节点注入账逐数闭合（方向口径先统一再比）。

后果（放置规则的硬件依据，组合规则归 training 5.x / 10.2）：高频集合（TP 的逐层 all-reduce）尽量留在 NVLink 域内——域内每跳 370.8 GB/s 有效（式 (2)），跨机每跳只有 NIC 线速量级；DP/PP 的跨机流量按 rail 与分层 ring 组织（10.2），把跨域字节数压到最小。NVL72 把域从 8 GPU 扩到 72 GPU，正是把这条数量级差的边界往外推——72 GPU 内的 all-reduce 仍走 NVLink，不必下 NIC（7.1 §4 域账）。RDMA 与 GDR 的硬件机制归 P9，机间 fabric 与大规模通信工程归 P10。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| HGX 8-GPU 拓扑：8×18 = 144 links 占 144 交换端口（4×64 = 256，平均 36/chip）；逐颗分布闭源未披露，只做聚合账；4-GPU HGX 无 NVSwitch、H100 PCIe 版 NVLink 600 GB/s 勿混 SXM 900 | §2 拓扑图 + SKU 注 | §2、7.1 §3 |
| 每对 P2P 峰值上界 = min(两端注入)；直连均分 900/7 ≈ 128 GB/s 双向（blog Table 1）；NVSwitch 单对 900；总注入两者相同，switch 增益在单对峰值与调度灵活性 | 式 (1)，命题 1 | §3、练习 3 |
| 方向语义：单向拷贝峰值 450 GB/s（payload 单方向），双向并发 900；read = payload 目标→发起方，write = 发起方→目标；bidirectional 报值是每方向非总和 | §3 方向段 | §3、§6、练习 1 |
| 有效带宽 = raw × 协议效率：NCCL 模型 SM90 per-link 20.6 GB/s 单向 → 18×20.6 = 370.8 GB/s，η = 82.4%；SM100 40.1 → 721.8；raw 25 与模型 20.6 是两层账 | 式 (2) | §3、8.4 |
| α-β 时间账：n=8、M=1 GB、β=1/370.8、α=2 μs（教学假设）→ 带宽项 4.72 ms、延迟项 0.028 ms、合计 4.75 ms，带宽主导（training 4.1 regime recap） | 式 (3) | §3、练习 2 |
| ring 每跳 = GPU→NVSwitch→GPU 路径，每跳单向 raw 450 / 模型 370.8；每 rank 发送 2(n-1)/n·M（recap 不重推）；all-to-all 每 peer M/n；NVLS 每 rank 发送 M → 2.70 ms | §4 映射段 | §4、8.2/8.4、7.1 §6 |
| PCIe Gen5 x16 = 64 单向/128 双向；H2D = min(PCIe, inter-socket)；教学假设 inter-socket 40 → 跨 socket 40 GB/s = 本地 62.5%；NUMA 绑定 + nvbandwidth 对比测量 | 式 (4) | §5、练习 4 |
| nvbandwidth P2P 矩阵：逐对 read_sm/write_ce；对角线 = 同卡 HBM（对账 ×2）；非对角 NVLink ≈ 450×效率（教学期望）；延迟 pointer chasing；报值先问 payload/总流量、单向/双向 | §6 方法 + 坑注 | §6、2.4 §6 |
| 节点内 vs 机间：400G NIC = 50 GB/s 单向；每 GPU 450/50 = 9×（B200 18×）；节点 3600 vs 出口 400 GB/s = 9×；高频集合留域内，跨域按 rail 分层 | §7 数量级账 | §7、练习 4、9.2/10.2 |

## 练习

1.（计算 · [10]）P2P 方向口径代入（§3）：(a) H100 单对单向拷贝峰值多少 GB/s；(b) 双向并发拷贝峰值多少；(c) PCIe Gen5 x16 单向理论值多少；(d) NVLink4 单向对 PCIe 单向的比值。校验：(a) 450（头条 900 ÷ 2）；(b) 900；(c) 64（Hopper wp p.49 逐字）；(d) 450/64 ≈ 7.03，与 7.1 式 (5) 的 7× 一致。
2.（计算 · [20]）ring 时间账（式 (2)(3)，教学假设口径）： $n = 8$、 $M = 1$ GB、 $\beta = 1/370.8$ GB/s、 $\alpha = 2$ μs。(a) 带宽项多少 ms；(b) 延迟项多少 ms；(c) 合计多少、带宽项是延迟项的几倍？先预测哪项主导再算。校验：(a) 4.72 ms；(b) 0.028 ms；(c) ≈4.75 ms、约 169 倍，带宽主导与 training 4.1 §2.2 regime 结论一致，与 §3 实例逐数一致（容差 1%）。
3.（推导 · [30]）证明命题 1（式 (1) 每对上界与两种达界拓扑）：(a) 写出注入上界论证（每字节占两端各一个方向）；(b) 直连均分情形代 $n = 8$ 算每对双向带宽，核对 blog Table 1 的 128 GB/s；(c) switch 情形说明为什么单对可达 900（引非阻塞披露）；(d) 给边界反例：一个 GPU 同时与 7 个 peer 各跑 900 GB/s 为什么不可行；(e) 对账 blog 8-GPU 行：直连 7 × 128 vs switch 900，说明总注入为什么相同。校验：(b) 900/7 = 128.57 ≈ 128（blog 取整）；(d) 7 × 900 = 6300 超出 900 注入上限；(e) 7 × 128 ≈ 900，与 §3 代入一致。
4.（计算 · [30]）NUMA min 瓶颈与节点出口账（式 (4)，教学假设口径）：(a) 设 inter-socket 有效 40 GB/s，跨 socket H2D 多少 GB/s、是本地的百分之几；(b) 400 Gb/s NIC 线速多少 GB/s 单向，H100 每 GPU NVLink 单向对其比值；(c) 8 NIC 节点出口聚合 vs 8 GPU NVLink 注入聚合（均单向）各多少、比值几倍；(d) B200 每 GPU 比值变多少？校验：(a) min(64, 40) = 40 GB/s = 62.5%；(b) 50 GB/s、9×；(c) 400 vs 3600 GB/s、9×；(d) 900/50 = 18×，与 §7 逐数一致。

## 参考

- [NVIDIA, NVLink/NVSwitch blog](../references.md)：NVIDIA NVLink and NVSwitch Supercharge Large Language Model Inference blog（key: nvlink-nvswitch-blog），[C] 开放技术博客、非闭源，2026-11-25——非阻塞 crossbar 逐字、P2P 对照 Table 1（无 switch 每对 900/7 ≈ 128 GB/s、2-GPU 128 vs 900、8-GPU 7×128 vs 900）、HGX 8-GPU 4 颗第三代 NVSwitch（§2/§3）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink4 18×25 = 450 单向 = 900 双向（p.47）、PCIe Gen5 节 p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」（§3/§5）
- [NVIDIA, HGX/DGX](../references.md)：NVIDIA HGX / DGX 服务器规格（key: hgx-dgx），[C] 官方规格披露·一手权威，2026-11-25——HGX H100/H200 datasheet NVSwitch 行（8-GPU 第三代、4-GPU N/A 无 switch P2P）、HGX B200 8-GPU NVLink5 聚合 14.4 TB/s（§2）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100 / H200 / HGX H100-H200 / Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM NVLink 900 GB/s + PCIe Gen5 128 GB/s、H100 PCIe 版 NVLink 600 GB/s（bridge）勿混 SXM、Blackwell NVLink 1.8 TB/s（§2/§5/§7）
- [NVIDIA, nvbandwidth](../references.md)：NVIDIA/nvbandwidth repo（key: nvbandwidth），教学参照 + 官方实现，README 2026-11-25 经 GitHub API 一手核实——CE/SM 两类拷贝逐字、测试名族（device_to_device_memcpy_read_sm / write_ce / bidirectional / latency_sm、host_to_device_memcpy_ce）、spin kernel + CUDA events、3 采样中位数、512 threads/block 截断、pointer chasing、双向报值「per-direction … not an aggregated sum」逐字（§3/§5/§6）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露·一手权威，2026-11-25——Gen5 x16 = 64 GB/s 单向 / 128 GB/s 双向；URL 403 bot 拦截残留，数值口径经 Hopper wp p.49 逐字 + datasheet 互证闭合（§5）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e，master 2026-11-24）fresh tarball 核实——topo.h SM90_NVLINK_BW 20.6 / SM100_NVLINK_BW 40.1（per-link 单向有效模型值，vs raw 25/50 勿混）、paths.cc PATH_NVL / PATH_PHB / PATH_PXN 路径类型、rings.cc ncclBuildRings（§3/§4，代码全量走读归 8.4）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink5 18×50 = 900 单向 = 1.8 TB/s 双向（p.10 逐字）、HGX B200 聚合 14.4 TB/s（§3/§7 对比行）
