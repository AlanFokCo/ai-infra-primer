# 10.2 拓扑感知集合与网内聚合（Topology-Aware Collectives and In-Network Aggregation）

> 同一个 ncclAllReduce，8 卡域内 4.75 ms、64 卡跨机为什么单通道环要 39 ms——环怎么摆到 fabric 上才能把 8 张 NIC 全用上？SHARP 把归约搬进 IB 交换机，每 rank 出口流量从约 2M 降到 M，省的是哪一笔？万卡规模下 ring 的延迟项 286 ms、tree 只要 0.39 ms，测量与调优怎么落？本章 recap 8.4 NCCL 机制走读（不重推），交付 fabric 工程增量：拓扑检测与 rail 映射（NCCL_TOPO_FILE / PXN / connectRings）、分层 ring 的流量与延迟账（命题 1）、SHARP 网内聚合与 NVLS 对照（命题 2）、CollNet 集成与部署、多 channel/多 NIC/algo/proto 调优、多作业干扰与拥塞交互（9.3 recap）、万卡 all-reduce 测量账（nccl-tests busbw） | 前置：8.4（NCCL 机制走读，recap 源）、8.1（消息量账）、8.2（ring 下界）、8.3（tree 延迟）、7.1 §6（NVLS）、7.2（有效带宽 370.8 / 节点出口 9×）、10.1（fabric 拓扑/rail/收敛比，recap）、9.2（GDR/多 NIC）、9.3（拥塞，recap）；下游 10.3（a2a/DeepEP）、10.4（sendrecv）、11.2（drill-down） | 锚点：NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，2.32.7-1 / 351f33fd）、NCCL docs（[NVIDIA, NCCL docs](../references.md)，[C] 官方规格披露·一手权威）、SHARP docs（[NVIDIA, SHARP docs](../references.md)，[C]）、nccl-rdma-sharp-plugins（[Mellanox, nccl-rdma-sharp-plugins](../references.md)，教学参照）、nccl-tests（[NVIDIA, nccl-tests](../references.md)，教学参照）、Quantum switch（[NVIDIA, Quantum switch](../references.md)，[C]）、IB cluster topology（[NVIDIA, IB cluster topology](../references.md)，[C]）、NCCL 2.12 blog（[NVIDIA, NCCL 2.12 blog](../references.md)，[C] 开放技术博客）、NCCL 2.4 blog（[NVIDIA, NCCL 2.4 blog](../references.md)，[C]）、ConnectX-7（[NVIDIA, ConnectX-7](../references.md)，[C]）——代码锚 2026-11-25 fresh tarball 核实

## 1. 三个问题与本章口径

8 卡域内 all-reduce 4.75 ms（7.2 §3），把组扩到 64 卡跨 8 节点，单通道环为什么涨到约 39 ms——瓶颈换到了哪一段，怎么把节点里 8 张 NIC 全用上而不是只喂一张？SHARP 宣称把归约卸载进交换机、"eliminating the need to send data multiple times between endpoints"，每 rank 出口流量到底从多少降到多少，延迟项又省在哪？万卡规模下小消息 all-reduce 的延迟项 ring 要 286 ms、tree 族只要 0.39 ms，这笔账怎么用 nccl-tests 测出来、怎么调？本章回答三个问题：拓扑感知的 ring/tree 怎么把逻辑环映射到物理 rail（§3）？SHARP/NVLS 网内聚合的流量与延迟账怎么列（§4-5）？大规模部署的调优旋钮与多作业干扰怎么管（§6-7），万卡测量怎么落（§8）？

**本章口径**（章脊锁定，R2-2）：主口径 **HGX H100 8-GPU 节点**（NVLink4 + 4 颗第三代 NVSwitch，几何 recap 7.1/7.2）+ **rail-optimized fabric，每 GPU 一张单口 ConnectX-7 NDR 400G**（线速 50 GB/s 单向，[NVIDIA, ConnectX-7](../references.md)，[C]）。有效带宽取 **NCCL 模型 370.8 GB/s 单向**（7.2 式 (2)，18 × 20.6，raw 25 与模型 20.6 两层账分离）；NIC 线速 50 GB/s（400 ÷ 8）。延迟常数两套口径分开： $\alpha_{\mathrm{NVL}} = 2$ μs 为**教学假设**（沿 7.2 §3 / 8.1）， $\alpha_{\mathrm{NET}} = 14.0$ μs 取 **NCCL cost model 内置校准值**（8.4 §4，ring + Simple 协议 NET 每跳硬件延迟，非本章教学假设）。贯穿实例 $M = 1$ GB = $10^{9}$ B、bf16；跨机例 $k = 8$ 节点、 $n = 64$ rank。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，recap training 4.1 时 $d \to n$ 翻译）； $k$ = 节点数（ $n = 8k$，每节点 8 GPU，本章局部符号）； $M$ = 被归约消息总字节； $\alpha_{\mathrm{NVL}}$ / $\alpha_{\mathrm{NET}}$ = NVLink / NET 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $`\beta_{\mathrm{NVL}} = 1/\mathrm{BW}_{\mathrm{NVL}}`$、 $`\beta_{\mathrm{NIC}} = 1/\mathrm{BW}_{\mathrm{NIC}}`$ = 每字节传输时间； $C$ = 每 rank 流量字节（局部）。 $N$ 专归参数量、 $B$ 专归 batch、 $T$ 专归序列长，本章不占用；时间一律 $T_{\mathrm{comm}}$ / $T_{\mathrm{hier}}$ 下标。

**recap 边界声明**（强制，spec §4.2）：8.4 NCCL 模块全量走读（init 流程 / channel / transport / proxy / 三协议 / cost model / NVLS kernel）、8.1 八原语消息量账、8.2 ring 带宽最优性下界、8.3 双树带宽最优与 ring/tree 交叉、7.1 §6 NVLS 流量账、7.2 有效带宽折扣、10.1 fat-tree 收敛比与 rail-optimized 拓扑、9.2 GDR 带宽 min 与多 NIC 聚合、9.3 DCQCN/PFC/IB CC 机制——**均已交付，本章只 recap + 指引，不重推**。本章增量 = fabric 工程层：拓扑检测的部署口径、环到 rail 的映射、分层 ring 账、SHARP 部署与 CollNet 集成、调优旋钮、多作业干扰、万卡测量方法。

**两层走读纪律声明（8.4 ↔ 10.2，强制）**：8.4 = **代码机制层**（repo 每个模块怎么实现）；本章 = **fabric 工程层**（怎么把环摆到物理 fabric、SHARP 交换机怎么配、NCCL_TOPO_FILE 怎么调、多作业怎么隔离）。**代码 scope 声明**：本章 = 机制片段级（topo.cc / paths.cc / connect.cc / net_ib / coll_net.cc 的 fabric 相关片段，非模块全量走读；全量走读已在 8.4 交付）。**测量口径**沿 7.2 / spec §6.2：本机无 NVIDIA GPU，万卡 all-reduce 不可能新跑 benchmark，本章交付**测量方法 + 官方披露/已发表数值**，数值账为 Python 独立重算。

## 2. Recap：NCCL 机制链与消息量账

fabric 工程到底要动 NCCL 的哪几个自由度？一个 ncclAllReduce 从入队到数据落地经过初始化、channel、transport、proxy、协议、cost model、NVLS 七个环节（8.4 已全量走读），其中绝大多数是代码机制、本章不重推；真正留给 fabric 工程的只有「环怎么摆到物理链路」「归约在哪做」两个自由度。先把 8.4 机制链 recap 成表（指针回 8.4 各节），标出本章增量落点：

| 环节 | 8.4 已交付机制 | 本章 fabric 工程增量 |
| --- | --- | --- |
| 初始化 | ncclCommInitRank → bootstrap → 拓扑 → 图搜索 → transport → devComm（8.4 §2） | 拓扑检测的部署口径（NCCL_TOPO_FILE，§3） |
| channel | 一 channel 一 CTA，rings.cc/trees.cc，MAXCHANNELS 64（8.4 §3） | 多 channel 跨 NIC 的 rail 映射（§3/§6） |
| transport | p2p/shm/net/collNet，NVLS 独立（8.4 §4） | net 的 rail/PXN 放置、collNet 的 SHARP 集成（§3/§5） |
| proxy | GPU-NIC 协调，posted/transmitted/done（8.4 §5） | GDR 协商的部署前提（§6，9.2 recap） |
| 协议 | LL 50% / LL128 93.75% / Simple（8.4 §6） | algo/proto 调优旋钮（§6） |
| cost model | ncclTuningGetTime 的 α-β 代码形态（8.4 §7） | 拓扑带宽喂入 bwIntra/bwInter（§3） |
| NVLS | cuMulticastCreate，归约进 NVSwitch（8.4 §8） | NVLS vs IB SHARP 对照（§4） |

消息量账 recap（8.1 / 8.2，不重推）：ring all-reduce 每 rank 发送 $C_{\mathrm{rank}} = 2(n-1)/n \cdot M$，带宽最优性下界已证（8.2 命题 1，总流量 $\ge 2(n-1)M$）；α-β 时间模型 $T_{\mathrm{ring}} = 2(n-1)\alpha + 2\frac{n-1}{n} M \beta$（training 4.1 式 (3) recap， $d \to n$）。代入 $n=8$、 $M=1$ GB、 $\beta_{\mathrm{NVL}} = 1/370.8$ GB/s、 $\alpha_{\mathrm{NVL}}=2$ μs：带宽项 4.72 ms、延迟项 0.028 ms、合计 4.75 ms（7.2 §3 逐数一致）。**关键结构性质**：带宽项系数 $2(n-1)/n$ 随 $n$ 趋于 2、与 $n$ 基本无关——这是 ring 撑到万卡的带宽侧依据；但延迟项 $2(n-1)\alpha$ 随 $n$ 线性涨，万卡小消息下它会主导（§8 定量）。fabric 工程要解决的就是：带宽项落在哪条物理链路（NIC 还是 NVLink）、延迟项怎么压。

## 3. 拓扑检测与 rail 映射：环怎么摆到 fabric 上

先回答一个更基本的问题：NCCL 怎么知道机器里有哪些 GPU/NIC、谁离谁近？答案是拓扑检测把机器读成一棵 XML 树。`ncclTopoGetSystem`（src/graph/topo.cc L1994，[NVIDIA, NCCL](../references.md)，教学参照，2.32.7-1）经 NVML/PCI 枚举 GPU、NIC、CPU 与桥，节点类型登记在 topo.h L47-51：GPU 0 / PCI 1 / NVS 2 / CPU 3（NUMA）/ NIC 4。检测默认从硬件读，但部署上可以用文件覆盖：

**NCCL_TOPO_FILE**（topo.cc L2006-2013）：环境变量指向一个 XML 文件，`ncclTopoGetXmlFromFile` 在硬件检测前加载它；未设时默认尝试 `/var/run/nvidia-topologyd/virtualTopology.xml`（L2012，与 NCCL docs 逐字一致）。docs 口径：拓扑文件**只描述单 host**（even for multi-node NVLink systems）。用途：虚拟机/容器里 NVML 读不到真实 NVLink 拓扑时，由 topologyd 生成虚拟拓扑喂给 NCCL；`NCCL_TOPO_DUMP_FILE`（L2170）反向把检测到的拓扑 dump 成 XML 供核对。per-link 带宽取 SM90_NVLINK_BW 20.6 / SM100 40.1（topo.h L24/L26，NCCL 有效模型值 vs raw 25/50，8.4 §2 口径注 recap）。

**路径梯子与 PXN**（recap 8.4 §2，本章增量是 rail 对齐的工程读法）。paths.cc 给每对 GPU/NIC 定路径类型：PATH_NVL 1（NVLink 直连）/ PATH_PXN 7（经中间 GPU 到 NIC）/ PATH_PHB 8（经 CPU root complex）。**PXN（PCI × NVLink）**是 rail 对齐的代码实化：`ncclTopoCheckP2p` 注释逐字「The connection may go through an intermediate rank」（paths.cc L319），函数签名带 `intermediateRank` 出参（L320-321），L319-378 段计算中间 rank。机制（NCCL 2.12 blog 逐字，[NVIDIA, NCCL 2.12 blog](../references.md)，[C]）：「PXN leverages NVIDIA NVSwitch connectivity between GPUs within the node to first move data on a GPU on the same rail as the destination, then send it to the destination without crossing rails」——发送 GPU 先经 NVLink 把数据搬到「与目的同 rail」的本节点 GPU，再由那张 GPU 的 NIC 发出，于是跨机只过一跳 leaf 交换机，而不是三跳（L0 → spine → L3）。

**rail-optimized 拓扑**（recap 10.1，本章只用其结论做映射）。IB cluster bring-up doc 逐字（[NVIDIA, IB cluster topology](../references.md)，[C]）：「Rail-optimized design means a GPU node with multiple interfaces will put each GPU "rail" (IB network interface) onto a different first level (LEAF) switch」——节点里第 $j$ 张 GPU 的 NIC 接第 $j$ 台 leaf，所有节点的第 $j$ 张 NIC 同属 rail $j$。NCCL 2.12 blog 逐字：「NIC-0 of each DGX system is connected to the same leaf switch (L0), NIC-1s... (L1)... rail-optimized... maximize all-reduce performance while minimizing network interference between flows」。

**环到 rail 的映射**。跨节点环由 `connectRings`（src/graph/connect.cc L105，`ncclTopoPostset` 内调用 L462）把图搜索得到的 prev/next 邻接落成每 channel 的环数组。逻辑环 0 → 1 → … → 63 → 0 若按节点连续排（每节点 8 GPU 相邻），则环上每节点恰有一条出边跨机——**单通道环只用到每节点一张 NIC**。要把 8 张 NIC 全用上，有两条等价路线：其一，多 channel，每 channel 一条环、各环的跨机边映射到不同 rail（§6 多 channel 调优）；其二，分层 ring，节点内 NVLink 环 + 机间 IB 环显式分相。下面给分层 ring 的定量账。

```
分层 ring all-reduce（k=8 节点 × 8 GPU，rail-optimized）
相 1  节点内 reduce-scatter（NVLink 环，8 rank）
        每 GPU 持 M/8 分片的「节点内全和」
        GPU0 ─NVLink─► GPU1 ─► … ─► GPU7 ─► GPU0   （每节点独立）
相 2  机间 all-reduce（IB 环，k=8 节点，8 rail 并行）
        rail j：所有节点的第 j 张 GPU 组一条 IB 环，归约 M/8 分片
        node0.GPUj ─NIC_j─► node1.GPUj ─► … ─► node7.GPUj ─► node0.GPUj
        （每 rail 一台 leaf，跨机一跳；8 rail 同时跑 → 8 NIC 全用）
相 3  节点内 all-gather（NVLink 环，8 rank）
        把 M/8 全和分片扩回全量 M
```

**命题 1**（分层 ring 的流量与延迟）。设 $k$ 节点、每节点 8 GPU、 $n = 8k$，消息 $M$。分层三相（节点内 RS + 机间 AR + 节点内 AG）下：

(a) 每 NIC 流量 $C_{\mathrm{NIC}} = 2\frac{k-1}{k} \cdot \frac{M}{8}$；单通道节点连续平环每 NIC 流量 $C_{\mathrm{NIC}}^{\mathrm{flat}} = 2\frac{n-1}{n} \cdot M$。两者之比 $\frac{C_{\mathrm{NIC}}^{\mathrm{flat}}}{C_{\mathrm{NIC}}} = \frac{n-1}{n} \cdot \frac{k}{k-1} \cdot 8$。

(b) 延迟项 $T_{\mathrm{lat}} = 14\alpha_{\mathrm{NVL}} + 2(k-1)\alpha_{\mathrm{NET}}$，带宽项 $T_{\mathrm{bw}} = \frac{7}{4} M \beta_{\mathrm{NVL}} + \frac{k-1}{4k} M \beta_{\mathrm{NIC}}$，合计 $T_{\mathrm{hier}} = T_{\mathrm{lat}} + T_{\mathrm{bw}}$。

**证明 (a)**（路线：逐相数跨机字节，三步）。其一，相 1 节点内 RS 只在 NVLink 上搬，跨机字节 0；相 3 AG 同理。其二，相 2 机间 AR 的消息是 $M/8$ 分片，每 rail 一条 $k$ rank 环，每 rank 发送 $2(k-1)/k \cdot (M/8)$（ring 消息量 recap 8.1），全走该 rail 的 NIC，故每 NIC 流量即式 (a)。其三，平环每 rank 发送 $2(n-1)/n \cdot M$，节点连续排时每节点出边（一张 NIC）承载该节点全部跨机流量，即每 NIC $2(n-1)/n \cdot M$；相比即得比值。∎

**证明 (b) 路线**（完整推导留作练习 3）：相 1 RS 走 7 步 NVLink、带宽 $(7/8)M\beta_{\mathrm{NVL}}$；相 2 AR 走 $2(k-1)$ 步 NET、带宽 $2(k-1)/k \cdot (M/8)\beta_{\mathrm{NIC}}$；相 3 AG 走 7 步 NVLink、带宽 $(7/8)M\beta_{\mathrm{NVL}}$；三相串行相加，延迟项 $7\alpha_{\mathrm{NVL}} + 2(k-1)\alpha_{\mathrm{NET}} + 7\alpha_{\mathrm{NVL}}$、带宽项合并即式 (b)。

全数字代入（ $k=8$、 $n=64$、 $M=1$ GB、 $`\mathrm{BW}_{\mathrm{NVL}}=370.8`$、 $`\mathrm{BW}_{\mathrm{NIC}}=50`$ GB/s、 $\alpha_{\mathrm{NVL}}=2$ μs、 $\alpha_{\mathrm{NET}}=14$ μs）：每 NIC 流量分层 $2 \times 7/8 \times 0.125 = 0.21875$ GB = **218.75 MB**，平环 $2 \times 63/64 = 1.96875$ GB——比值 **9.0×**（平环只喂一张 NIC，分层 8 张全用）。时间： $T_{\mathrm{bw}}$ NVLink 项 $1.75/370.8 \times 10^{3} = 4.72$ ms、NIC 项 $0.21875/50 \times 10^{3} = 4.375$ ms； $T_{\mathrm{lat}} = 14 \times 2 + 2 \times 7 \times 14 = 224$ μs = 0.224 ms； $T_{\mathrm{hier}} = 4.72 + 4.375 + 0.224$ = **9.32 ms**。平环单通道带宽项 $1.96875/50 \times 10^{3} = 39.375$ ms（NIC 瓶颈）——分层对平环加速 **4.2×**。**口径注**：分层三相是教学推导口径（机制级，沿 7.1 §6 / 8.2）；NCCL 实际用多 channel 平环 + rail 映射逼近同一效果（每 channel 跨机边落不同 NIC，每 NIC ≈ $2(n-1)/n \cdot M/8 = 0.246$ GB，与分层 0.219 GB 同量级），并非显式三相 kernel。边界反例： $k=1$（单节点）时式 (a) 机间项为 0、式 (b) 退化为纯节点内 RS+AG，分层无意义——分层的收益完全来自「把跨机流量摊到多 rail」。

> **提点**：分层 ring 与「多 channel 平环 + rail 映射」是同一笔带宽账的两种实现——前者显式分相、后者靠 channel 并行隐式摊 NIC。工程上 NCCL 选后者（kernel 简单、channel 数可调），但账要按前者列才看得清「每 NIC 流量与 $n$ 解耦、只与 $k$ 和 $M/8$ 有关」这个结构。

## 4. SHARP 网内聚合：把归约搬进 IB 交换机

ring/tree 的归约都在 GPU 的 SM 上做（NVLS 例外，搬进 NVSwitch）——跨机归约能不能搬进 IB 交换机？SHARP 就是这条 bypass。所谓 **SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）**，是指在 IB 交换机硬件里做聚合与归约、把集合操作从端点 CPU/GPU 卸载到网络的技术（[NVIDIA, SHARP docs](../references.md)，[C] 官方规格披露·一手权威，v3.17.0）。overview 逐字：「offloading collective operations from CPUs and GPUs to the network... eliminating the need to send data multiple times between endpoints... decreases the amount of data traversing the network as aggregation nodes are reached」。硬件代际：Quantum-2（NDR）SHARPv3、Quantum-X800（XDR）SHARPv4（[NVIDIA, Quantum switch](../references.md)，[C]）；SHARPv3 起「multiple aggregation trees can be built over the same topology」——同一物理拓扑上建多棵聚合树，多作业并行（§7）。

**命题 2**（SHARP 每 rank 出口流量与延迟）。设 $n$ rank 各持 $M$ 字节做 all-reduce，SHARP 沿聚合树归约。机制级图景：每 rank 把自有 $M$ 字节向根方向上传一次，聚合节点在交换机内把子树部分和归约后再上传，根得全和后向下广播 $M$。则每 rank **发送** $M$（上传），对比 ring 的 $2(n-1)/n \cdot M$，比值 $2(n-1)/n \to 2$（ $n \to \infty$）；延迟项为树深 $O(\log n)$ 跳，对比 ring 的 $2(n-1)$ 步。

**证明**（路线：数每 rank 上传字节 + 数树深，两步）。其一，每 rank 只上传自有贡献 $M$ 一次（归约在交换机内做，端点不必像 ring 那样逐 chunk 转发 $2(n-1)/n \cdot M$），故发送 $M$；接收为广播下来的全和 $M$。其二，聚合树深 $\lceil \log_2 n \rceil$ 量级，上传 + 广播约 $2\lceil \log_2 n \rceil$ 跳，对比 ring $2(n-1)$ 步。∎ **口径注**：这是机制级教学推导（沿 7.1 §6 NVLS 同口径），「up to 2×」是上界披露非保证值；增益以端点注入带宽为瓶颈口径，交换侧复制流量由非阻塞端口容量承担。SHARP 聚合树是 8.3 双树 $O(\log n)$ 延迟的网内实现，带宽最优性骨架同源。

全数字代入（ $n=64$、 $M=1$ GB、NIC 50 GB/s）：ring 每 rank 出口 $1.96875$ GB → 出口时间 $39.375$ ms；SHARP 每 rank 出口 $1$ GB → **20 ms**，比值 1.97。延迟项万卡（ $n=10240$、 $\alpha_{\mathrm{NET}}=14$ μs）：ring $2(n-1)\alpha_{\mathrm{NET}} = 2 \times 10239 \times 14$ μs = **286.7 ms**，tree 族 $2\lceil \log_2 10240 \rceil \alpha_{\mathrm{NET}} = 2 \times 14 \times 14$ μs = **392 μs**——比值 **731×**。这就是万卡小消息 all-reduce 必须走 tree/SHARP 而非 ring 的定量依据（§8 测量兑现）。

**SAT 大消息流式聚合**。SHARP collective library 支持 Barrier/Allreduce/Reduce/Bcast/Reduce-scatter/Allgather，数据类型含 16/32/64-bit int/float + BF16 + INT8（sharp-docs）。大消息走 **SAT（Streaming Aggregation Tree）**：`SHARP_COLL_ENABLE_SAT`（默认 0）、`SHARP_COLL_SAT_THRESHOLD` 默认 **16384 B**（16 KB，超过即流式）、max message **1073741792 B**（1 GB − 32 B）。口径：小消息走 OST（offload subtree）低延迟路径，大消息走 SAT 流式——阈值 16 KB 与 8.1 regime 临界 $M_c$（5.93 MB，ring 内部）不同源，勿混。

**NVLS vs IB SHARP 对照**（recap 7.1 §6 / 8.4 §8，不重推流量账）：

| 维度 | NVLS（NVLink SHARP） | IB SHARP |
| --- | --- | --- |
| 域 | NVLink 域内（节点内 / NVL72 超节点） | 跨机 IB fabric |
| 硬件 | 第三代 NVSwitch（NVLink4）+ Hopper+ | Quantum HDR+ 交换机 + ConnectX-6 HDR+ |
| 归约位置 | NVSwitch 内 multicast + reduce | IB 交换机内 aggregation tree |
| 每 rank 发送 | $M$（7.1 §6） | $M$（命题 2） |
| NCCL 路径 | 独立 nvls.cc / NVLS_TREE | collNet transport（§5） |
| 多节点 | NVLS_TREE（2.18+） | CollNet chain/direct |

两者机制同族（网内归约）、层级不同：NVLS 管域内、IB SHARP 管跨机；NCCL 的 NVLS_TREE 与 CollNet 可组合（域内 NVLS + 跨机 SHARP）。可用性逐字（NCCL docs NCCL_NVLS_ENABLE）：「NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures」。

## 5. CollNet 集成：SHARP 怎么接进 NCCL

SHARP 交换机有了，NCCL 怎么用上？答案是 CollNet transport + NCCL-SHARP plugin。CollNet 是 NCCL 四个 transport 之一（8.4 §4 recap，transport.cc L15-20），`sharedBuffersInit`（src/transport/coll_net.cc L416）建 CollNet 共享缓冲；两种算法 setup：**COLLNET_CHAIN**（L1613）与 **COLLNET_DIRECT**（L1648）。chain = 节点内链式归约 + 跨机 SHARP；direct = 端点直连 SHARP 聚合树。开关 `NCCL_COLLNET_ENABLE`（src/init.cc L67，每次 comm init 重载）；docs 口径默认 0，须显式设 1（[NVIDIA, NCCL docs](../references.md)，[C]）。

**plugin 分发与依赖**（sharp-docs NCCL 集成页逐字）：「RDMA and SHARP collectives are enabled with NVIDIA NCCL through the NCCL-SHARP plugin」。分发两渠道：HPC-X binary（自动加载）/ source **[Mellanox/nccl-rdma-sharp-plugins](../references.md)**（教学参照 + 官方实现，autotools 构建 `configure --with-sharp=PATH`，经 LD_LIBRARY_PATH 加载）。Requirements 逐字：ConnectX-6 HDR+ / Quantum HDR switch+ / DOCA-Host GPUDirectRDMA Plugin / **nv_peer_mem module**（`lsmod grep nv_peer_mem`）/ NCCL 2.7.3+。启用：`NCCL_COLLNET_ENABLE=1`（≤2.7.8 有 bug 须配 `NCCL_ALGO=CollNet`）；`SHARP_COLL_LOCK_ON_COMM_INIT=1` 保证 SAT 资源；`NCCL_SHARP_DISABLE` / `NCCL_SHARP_GROUP_SIZE_THRESH` 可禁用或设组规模阈值。

**rail 拓扑硬要求**（集成页逐字，部署关键）：「On systems with multiple GPUs and multiple HCAs, NCCL creates an aggregation streaming flow (NCCL Ring/Channel) per HCA rail. It is required to build the cluster topology in such a way that leaf level switches connected to same HCA rail from each server」——**每 HCA rail 一条 aggregation streaming flow，各 server 同 rail 的 NIC 必须接同一台 leaf**。这正是 §3 rail-optimized 的 SHARP 侧兑现：rail 不对齐，SAT flow 建不起来。另一限制逐字：「NCCL SHARP Streaming aggregation is supported on a single NCCL communicator/process group」——SAT 流式聚合限单 communicator。

> **坑**：leaf 不同 rail 时强开 CollNet SAT。触发条件：rail-optimized 没做对（同 rail NIC 接了不同 leaf）却设 NCCL_COLLNET_ENABLE=1 + SHARP_COLL_ENABLE_SAT=1；现象：aggregation streaming flow 建不起来，SHARP 初始化失败或回退 ring，日志报资源/拓扑错；规避：先按 IB bring-up doc 核对「同 rail 同 leaf」（§3），用 NCCL_TOPO_DUMP_FILE 导出拓扑核对 rail 映射，再开 SAT。nv_peer_mem 未加载（9.2 GDR 前提）同样挡 SHARP——GDR 是 SHARP 端点直访 HBM 的底座。

## 6. 大规模 NCCL 部署：调优旋钮

部署到千卡万卡，哪些旋钮真正动带宽与延迟？按「拓扑 → 通道 → 算法/协议 → GDR/路由 → 可靠性」五层列（机制 recap 8.4，本章给部署口径）：

| 层 | 旋钮 | 代码 / docs 锚 | 部署口径 |
| --- | --- | --- | --- |
| 拓扑 | NCCL_TOPO_FILE / DUMP_FILE | topo.cc L2006 / L2170 | 虚拟拓扑喂入 / 导出核对（§3） |
| 通道 | NCCL_MAX_NCHANNELS / MIN_NCHANNELS | docs env.html | 多 channel 跨 NIC，rail 映射（§3） |
| 多 NIC | NCCL_IB_HCA / NCCL_IB_RAIL_POLICY | net_ib/init.cc L464 / docs | HCA 过滤 `<hca>[:<port>[:<rail>[:<plane>]]]`；SuperNIC 自动 rail/plane 分配（2.30.5+） |
| 多 QP | NCCL_IB_QPS_PER_CONNECTION / SPLIT_DATA_ON_QPS | net_ib/connect.cc L106（默认 1） | 每连接多 QP 提并发，数据按 QP 拆分 |
| 算法/协议 | NCCL_ALGO / NCCL_PROTO | cost_model.cc L365-379（8.4 §7） | Ring/Tree/NVLS/CollNet/PAT × LL/LL128/Simple |
| GDR | NCCL_NET_GDR_LEVEL / IB_CUDA_SUPPORT | paths.cc L542 / L494 | 拓扑允许默认开 GDR；GDR_LEVEL 强制距离阈值 |
| 路由 | NCCL_IB_ADAPTIVE_ROUTING | net_ib/init.cc L12（默认 -2） | IB 默认开、RoCE 默认关（L594-596） |
| 可靠性 | NCCL_IB_TIMEOUT / RETRY_CNT | net_ib/connect.cc L26 / L32 | 默认 20 / 7（9.3 recap） |

几个关键点的部署读法。**多 channel 跨 NIC**：ringGraph maxChannels = MAXCHANNELS/2 = 32（8.4 §3），实际 nChannels 由图搜索按拓扑定；channel 数增大到每 GPU 注入带宽饱和后不再增益（7.2 §3）。rail-optimized 下多 channel 的跨机边落不同 NIC，是 §3 分层账的工程实现。**GDR 协商**：`ncclTopoCheckGdr`（paths.cc L494）判 GPU-NIC GDR 可行性，`NCCL_NET_GDR_LEVEL`（L542，ncclTopoUserGdrLevel）覆盖距离阈值；net_ib/gdr.cc L18 经 sysfs `/sys/kernel/mm/memory_peers/nv_mem/version` 探测 nvidia-peermem 是否加载（9.2 §3 互证，DRV_NAME nv_mem）。docs 逐字：「By default, NCCL enables GPU Direct RDMA if the topology permits it」——GDR 默认开，前提是拓扑允许（同 root complex / IOMMU 1:1 / ACS 不重定向，9.2 §7 三道门 recap）。**自适应路由**：`NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2)`（init.cc L12），L594-596 逐字「Enable ADAPTIVE_ROUTING by default on IB networks」，`comm->ar` 取所有 merged dev 的 AND（connect.cc L989）；docs 口径 IB 默认开 / RoCE 默认关，且须选系统已启用 AR 的 SL（NCCL_IB_SL）。AR 让跨机流量在 fat-tree 多路径间散列，缓解 §7 的拥塞热点（10.1 路由 recap）。**可靠性**：IB_TIMEOUT 默认 20，注释逐字「4.096us * 2^timeout... 4.096 * 2^20 = 4,294,967.296us (or ~4.3s)」（connect.cc L26-31），RETRY_CNT 默认 7（L32）——IB 可靠连接重传机制（9.3 recap），超时过大掩盖链路故障、过小误判拥塞丢包。

> **坑**：在不支持 128 B 写原子性的 PCIe 平台强开 NCCL_PROTO=LL128。触发条件：跨机或 PCIe 路径设 LL128；现象：flag 与 data 非原子到达、接收侧读半 line，**数据损坏**（docs 逐字 data corruption，8.4 §6 坑注 recap）；规避：LL128 只在 NVLink（intraType 不高于 PATH_NVB）启用，isLL128Enabled 已门控，部署勿绕过。algo/proto 调优只在门控允许范围内选。

## 7. 集合调度与多作业干扰

同一张 fabric 上跑多个作业，带宽怎么分、拥塞怎么交互？先 recap 9.3 拥塞机制（不重推）：RoCE 用 **DCQCN**（ECN 标记 + CNP 反馈 + 速率调节，RP/CP/NP 三角色，[Zhu et al., 2015](../references.md)，[B] SIGCOMM 2015 无 arXiv）；**PFC**（802.1Qbb per-priority pause）保无损但有 head-of-line blocking 与 PFC storm 风险；IB 用 **CCA**（FECN → BECN → CCT 包间延迟，[Crupnicoff et al., Mellanox](../references.md)，[C]）。拥塞的后果是 RDMA 有效带宽 $\beta_{\mathrm{eff}}$ 降 → 集合通信延迟涨 → training/inference 通信账恶化（9.3 GPU 侧影响 recap）。

**fabric 带宽共享与 rail 隔离**。rail-optimized 的隔离价值（NCCL 2.12 blog 逐字）：「maximize all-reduce performance while minimizing network interference between flows... reduce the cost of the network by having lighter connections between rails」——同 rail 流量集中在本 rail 的 leaf 子树，跨 rail 干扰小；rail 间连接可以更轻（省成本）。多作业若各占不同 rail/分区，流量天然隔离；若共享 rail，则 a2a 的突发性与全 bisection 需求（8.1 §4 recap）会互相挤占。**SHARPv3 多聚合树**（sharp-docs 逐字「many parallel jobs over the same infrastructure」）：同一拓扑建多棵聚合树，每作业一棵，归约流量在交换侧隔离——这是网内聚合对多作业的独特价值（ring/tree 端点归约做不到交换侧隔离）。

**NVLink 域内多作业干扰**（recap inference 7.4，不重推）。DeepInfer doc L297 逐字（官方披露）：「Different jobs sharing NVLink within a supernode can interfere with one another and slow down. Using the whole supernode for a single EP group avoids that interference」——超节点 NVLink 是共享带宽，多作业 a2a 互相干扰；整超节点作单 EP 组把干扰内部化。这与 fabric 侧 rail 隔离同构：共享介质上，要么空间分区（rail/整超节点），要么时间分片（调度错开集合窗口）。

**SM 与调度交互**。通信 kernel 占 SM（8.4 device dispatch；DeepEP V2.5「EP dispatch and combine require GPU SMs」，10.3），多作业并发时通信与计算争 SM。集合调度（何时发哪个集合）与计算重叠（training 5.3 DualPipe / inference Mega MoE recap）是压暴露时间的两手段；fabric 侧能管的是带宽隔离与拥塞调参（ECN threshold / PFC priority / rate limiter，9.3 调参概念 recap），SM 侧归 kernel 调度（11.2）。口径：本章管 fabric 带宽与拥塞的工程，作业级编排（K8S 调度 / device plugin）归 cluster 维度。

## 8. 万卡 all-reduce 测量账：nccl-tests 方法

万卡 all-reduce 的带宽怎么测、读数怎么对硬件峰值？工具是 nccl-tests（[NVIDIA, nccl-tests](../references.md)，教学参照 + 官方实现）。**测量口径**（spec §6.2）：本机无 GPU、万卡不可能新跑，本节交付方法 + 官方披露/已发表数值，非新跑 benchmark。

**两个带宽口径**（doc/PERFORMANCE.md 逐字）：「NCCL tests report... algorithm bandwidth and bus bandwidth」。**algbw = S/t**（ $S$ = 消息字节、 $t$ = 实测时间）；**busbw** 修正因子使读数「compare it with the hardware peak bandwidth, independently of the number of ranks」。AllReduce 逐字推导：「we need 2(n-1) data transfers... t = (S*2*(n-1)) / (n*B)... B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)」——即 busbw = algbw × $2(n-1)/n$，正是 8.1 ring 每 rank 消息量系数的倒数兑现。Summary 逐字因子表：AllReduce **2(n-1)/n**、ReduceScatter / AllGather / **AlltoAll (n-1)/n**、Broadcast / Reduce **1**。「The bus bandwidth should reflect the speed of the hardware bottleneck: NVLink, PCI, QPI, or network」——busbw 对的是硬件瓶颈段，不是算法理论值。

**命令与读法**。all_reduce_perf 扫消息尺寸：`all_reduce_perf -b 8 -e 8G -f 2 -g 1`（8 B 到 8 GB、倍率 2、每进程 1 GPU）；多机经 MPI 拉起，SHARP UM 给的 1024-rank 例：`mpirun -np 1024 -map-by ppr:8:node -x NCCL_COLLNET_ENABLE=1 ...`（每节点 8 进程）。读法三条：小消息段 algbw 低、延迟项主导（busbw 也低）；大消息段 busbw 趋硬件瓶颈——rail-optimized 跨机 all-reduce 大消息 busbw 应趋 **per-rail NIC 线速 50 GB/s**（每 NIC 流量与 $n$ 解耦，§3 命题 1），域内则趋 NVLink 有效带宽；若 busbw 远低于 50，查 rail 映射/多 channel/GDR/拥塞（§6/§7）。

**已发表大规模锚**（官方披露，非新跑）：NCCL 2.4 blog（[NVIDIA, NCCL 2.4 blog](../references.md)，[C]）Summit **24,576 GPUs** 双树对 ring 延迟最高 **180×** 改善——这是 tree 族 $O(\log n)$ 延迟在万卡规模的官方兑现，与命题 2 延迟账（ring 286.7 ms vs tree 392 μs，731× 理论比）同方向（180× 是实测系统级、含带宽项与重叠，731× 是纯延迟项理论比，两口径勿混）。

**万卡规模账**（Python 独立重算， $n=10240$、 $\alpha_{\mathrm{NET}}=14$ μs）：ring 延迟项 $2(n-1)\alpha_{\mathrm{NET}} = 286.7$ ms，tree 族 $2\lceil \log_2 n \rceil \alpha_{\mathrm{NET}} = 392$ μs；带宽项 ring $2(n-1)/n \cdot M \approx 2M$ 与 $n$ 解耦（8.2 recap）。结论：万卡**大消息**带宽项主导、ring 仍可用（带宽最优）；万卡**小消息**延迟项主导、必须 tree/SHARP（286.7 ms 不可接受）。regime 临界 $M_c = n\alpha/\beta$（8.1 命题 2）随 $n$ 涨——万卡下 $M_c$ 巨大，更多消息落延迟主导侧，这是万卡偏向 tree/网内聚合的结构性原因。

## 9. 边界：本章不做什么

fabric 工程收口，剩下的归谁？本章交付：拓扑检测部署口径与 rail 映射（§3）、分层 ring 流量/延迟账（命题 1）、SHARP 网内聚合与 NVLS 对照（命题 2，§4）、CollNet 集成与部署（§5）、调优旋钮（§6）、多作业干扰（§7）、万卡测量方法（§8）。以下各有专章：**fabric 拓扑族与收敛比**（fat-tree 1:1 / rail-optimized 推导 / 端口 radix 规模）归 10.1；**a2a 拥塞与 DeepEP 实现**归 10.3；**sendrecv 与 Ring Attention / CP** 归 10.4；**GDR 硬件机制**归 9.2；**拥塞控制机制**（DCQCN 式 / PFC storm / IB CCA）归 9.3；**NCCL 模块全量走读**归 8.4；**实测 drill-down**（Nsight timeline）归 11.2；**作业编排**（K8S 调度）归 cluster。代码 scope：机制片段级。

下游 recap 接口：10.3 recap §3 rail 映射 + §7 拥塞交互（a2a 放置）；10.4 recap §3 环映射（ring pass 拓扑）；11.2 recap §8 nccl-tests 方法（通信基准）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 拓扑检测 ncclTopoGetSystem L1994；NCCL_TOPO_FILE L2006 默认 /var/run/nvidia-topologyd/virtualTopology.xml；DUMP L2170；节点类型 GPU0/PCI1/NVS2/CPU3/NIC4 | topo.cc / topo.h L47-51 | §3、练习 4 |
| PXN 经中间 GPU 到 NIC（paths.cc L319 intermediateRank），rail 对齐跨机一跳；connectRings L105/L462 落环；rail-optimized 每 GPU rail 接不同 leaf | paths.cc / connect.cc / IB topology doc | §3、10.1 |
| 分层 ring 每 NIC 流量 2(k-1)/k·M/8 = 218.75 MB vs 平环 2(n-1)/n·M = 1.969 GB，比 9.0×；T_hier = 9.32 ms vs 平环 39.375 ms，加速 4.2× | 命题 1、式 (a)(b) | §3、练习 1/2/3 |
| 分层延迟项 14α_NVL + 2(k-1)α_NET = 224 μs；带宽项 NVLink 4.72 + NIC 4.375 ms；多 channel 平环 rail 映射 ≈ 分层（每 NIC 0.246 GB） | 命题 1(b)、§3 口径注 | §3、练习 3 |
| SHARP 每 rank 出口 M vs ring 2(n-1)/n·M，比 →2；n=64 出口 20 vs 39.375 ms；SAT threshold 16384 B / max 1GB−32B | 命题 2、sharp-docs | §4、练习 2 |
| 万卡延迟项 ring 2(n-1)α_NET = 286.7 ms vs tree 2⌈log2 n⌉α_NET = 392 μs（731×）；Summit 24576 GPU 180× 实测 | 命题 2、NCCL 2.4 blog | §4/§8、练习 2 |
| NVLS（NVSwitch 域内）vs IB SHARP（跨机）：每 rank 发送均 M；NVLS_TREE + CollNet 可组合 | §4 对照表、7.1 §6 | §4、8.4 |
| CollNet chain L1613 / direct L1648；NCCL_COLLNET_ENABLE init.cc L67 默认 0；plugin nccl-rdma-sharp-plugins；每 HCA rail 一条 SAT flow、leaf 须同 rail | coll_net.cc / sharp-docs 集成页 | §5、练习 4 |
| 调优五层：TOPO_FILE / 多 channel / IB_HCA rail-plane / QPS_PER_CONN 默认 1 / ALGO-PROTO / GDR_LEVEL / AR IB 开 RoCE 关 / IB_TIMEOUT 20 RETRY 7 | §6 表、net_ib 锚 | §6、练习 4 |
| 多作业：rail 隔离 + SHARPv3 多聚合树 + 整超节点单 EP 组；拥塞 → β_eff 降 → 通信账恶化（9.3 recap） | §7、doc L297 | §7、9.3/10.3 |
| nccl-tests algbw=S/t、busbw=algbw×2(n-1)/n（AR）/ (n-1)/n（RS/AG/a2a）/ 1（Bcast/Reduce）；busbw 对硬件瓶颈；大消息趋 per-rail 50 GB/s | PERFORMANCE.md 逐字 | §8、练习 2 |

## 练习

1.（计算 · [10]）分层 ring 流量代入（命题 1(a)， $k=8$、 $n=64$、 $M=1$ GB）：(a) 分层每 NIC 流量多少 MB；(b) 平环单通道每 NIC 流量多少 GB；(c) 两者比值；(d) rail-optimized 8 NIC 节点出口聚合多少 GB/s、对 NVLink 注入 8×450 比值。校验：(a) 218.75 MB；(b) 1.96875 GB；(c) 9.0×；(d) 400 GB/s、9×（7.2 §7 / 9.2 命题 2 recap），与 §3 逐数一致。
2.（计算 · [20]）分层延迟与 SHARP/万卡账（命题 1(b)、命题 2， $\alpha_{\mathrm{NVL}}=2$ μs、 $\alpha_{\mathrm{NET}}=14$ μs、 $`\mathrm{BW}_{\mathrm{NVL}}=370.8`$、 $`\mathrm{BW}_{\mathrm{NIC}}=50`$ GB/s）：(a) $T_{\mathrm{hier}}$ 多少 ms（延迟项 + 两带宽项）；(b) SHARP $n=64$ 每 rank 出口时间 vs ring；(c) 万卡 $n=10240$ ring 延迟项 vs tree 族延迟项各多少、比值；(d) nccl-tests AllReduce busbw 因子（ $n=64$）。先预测 (c) 哪个主导再算。校验：(a) 9.32 ms；(b) 20 vs 39.375 ms；(c) 286.7 ms vs 392 μs、731×；(d) 2×63/64 = 1.96875，与 §3/§4/§8 一致（容差 1%）。
3.（推导 · [30]）证明命题 1(b)（分层 ring 延迟/带宽公式，正文留作练习）：(a) 逐相写 RS/AR/AG 的步数与每 rank 字节（RS 7 步 NVLink $(7/8)M$、AR $2(k-1)$ 步 NET $2(k-1)/k \cdot M/8$、AG 7 步 NVLink $(7/8)M$）；(b) 三相串行相加得 $T_{\mathrm{lat}} = 14\alpha_{\mathrm{NVL}} + 2(k-1)\alpha_{\mathrm{NET}}$、 $T_{\mathrm{bw}} = (7/4)M\beta_{\mathrm{NVL}} + (k-1)/(4k) M\beta_{\mathrm{NIC}}$；(c) 给边界反例 $k=1$ 退化；(d) 说明多 channel 平环 rail 映射为什么逼近分层（每 NIC ≈ $2(n-1)/n \cdot M/8$）。校验：(b) 与 §3 式 (b) 逐项一致；(c) 机间项为 0；(d) 0.246 GB vs 分层 0.219 GB 同量级。
4.（代码 · [30]）NCCL fabric 锚走读（版本锚 2.32.7-1 / 351f33fd）：(a) NCCL_TOPO_FILE 默认路径与加载函数（topo.cc）；(b) PXN intermediateRank 注释与出参（paths.cc）；(c) connectRings 调用点（connect.cc）；(d) IB_ADAPTIVE_ROUTING 默认值与 IB/RoCE 分派（net_ib/init.cc）、IB_TIMEOUT/RETRY_CNT 默认（net_ib/connect.cc）；(e) CollNet chain/direct setup 与 NCCL_COLLNET_ENABLE（coll_net.cc / init.cc）。校验：(a) L2006 / /var/run/nvidia-topologyd/virtualTopology.xml / ncclTopoGetXmlFromFile；(b) L319「The connection may go through an intermediate rank」/ L320-321；(c) L105 / L462；(d) L12 默认 -2 / L594-596 IB 开 RoCE 关 / L26 默认 20 / L32 默认 7；(e) L1613 / L1648 / init.cc L67，与 §3/§5/§6 一致。

## 参考

- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e，master 2026-11-24）fresh tarball 核实——graph/topo.cc ncclTopoGetSystem L1994 / NCCL_TOPO_FILE L2006-2013（默认 /var/run/nvidia-topologyd/virtualTopology.xml）/ NCCL_TOPO_DUMP_FILE L2170；topo.h 节点类型 L47-51 / SM90_NVLINK_BW 20.6 L24 / SM100 40.1 L26；paths.cc PXN L243 / intermediateRank 注释 L319 / ncclTopoCheckP2p L320-321 / L319-378 段 / ncclTopoCheckGdr L494 / NCCL_NET_GDR_LEVEL L542；connect.cc connectRings L105 / ncclTopoPostset 调用 L462；rings.cc ncclBuildRings L29 / trees.cc ncclGetDtree L90 / search.cc ncclTopoCompute L1245（8.4 全量走读 recap）；transport/coll_net.cc sharedBuffersInit L416 / COLLNET_CHAIN L1613 / COLLNET_DIRECT L1648；init.cc NCCL_COLLNET_ENABLE L67；transport/net_ib/init.cc IB_ADAPTIVE_ROUTING L12（默认 -2）/ IB 开 RoCE 关 L594-596 / NCCL_IB_HCA L464；net_ib/connect.cc IB_TIMEOUT 20 L26（4.096μs×2^20≈4.3s 逐字）/ RETRY_CNT 7 L32 / QPS_PER_CONNECTION 1 L106 / AR merge L989；net_ib/gdr.cc nv_mem sysfs L18 / ncclIbGdrSupport L24；gin/gin_host.cc ncclGetGinType L28 / ncclGetRailedGinType L36（§3/§5/§6）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2.32.3（滞后 repo 2.32.7 注）——NCCL_TOPO_FILE（单 host 描述）/ NCCL_COLLNET_ENABLE 默认 0 / NCCL_NET_GDR_LEVEL / NCCL_IB_CUDA_SUPPORT「By default... if the topology permits it」/ NCCL_IB_ADAPTIVE_ROUTING（IB 开 RoCE 关）/ NCCL_IB_HCA rail-plane 语法 / NCCL_IB_RAIL_POLICY（2.30.5+）/ NCCL_ALGO（CollnetChain/CollnetDirect 2.14+ / NVLS 2.17+ / NVLSTree 2.18+ / PAT 2.23+）/ NCCL_PROTO / NCCL_NVLS_ENABLE 逐字（§3/§4/§5/§6）
- [NVIDIA, SHARP docs](../references.md)：NVIDIA SHARP docs（key: sharp-docs），[C] 官方规格披露·一手权威，v3.17.0（2026-11-16）——网内聚合 overview 逐字（offload / eliminating multiple sends / SHARPv3 multiple aggregation trees / many parallel jobs）；collective library Barrier/Allreduce/Reduce/Bcast/RS/AG + BF16/INT8；SAT SHARP_COLL_ENABLE_SAT 默认 0 / threshold 16384 / max 1GB−32B；NCCL-SHARP plugin 集成页（HPC-X / source / CX-6 HDR+ / Quantum HDR+ / nv_peer_mem / NCCL 2.7.3+ / NCCL_COLLNET_ENABLE=1 / 每 HCA rail 一条 SAT flow leaf 同 rail / 单 communicator）（§4/§5）
- [Mellanox, nccl-rdma-sharp-plugins](../references.md)：Mellanox/nccl-rdma-sharp-plugins repo（key: nccl-rdma-sharp-plugins），教学参照 + 官方实现——「enables RDMA and Switch based collectives(SHARP) with NVIDIA's NCCL」；autotools configure --with-sharp=PATH；CollNet/SHARP 集成实现（§5）
- [NVIDIA, nccl-tests](../references.md)：NVIDIA/nccl-tests repo（key: nccl-tests），教学参照 + 官方实现，README + doc/PERFORMANCE.md 2026-11-25 重核——algbw=S/t / busbw 因子 AllReduce 2(n-1)/n / RS/AG/AlltoAll (n-1)/n / Bcast/Reduce 1 / 「bus bandwidth... hardware bottleneck: NVLink, PCI, QPI, or network」逐字（§8）
- [NVIDIA, Quantum switch](../references.md)：NVIDIA Quantum InfiniBand 交换机（key: quantum-switch），[C] 官方规格披露·一手权威——Quantum-2（NDR）64×400G / 51.2 Tb/s 双向 / SHARPv3；Quantum-X800（XDR）Q3400-RA 144×800G / SHARPv4 / 2 级 fat-tree 10368 NICs（§4，10.1 fabric 规模 recap）
- [NVIDIA, IB cluster topology](../references.md)：NVIDIA IB Cluster Bring-Up — Setting the InfiniBand Cluster Topology（key: ib-cluster-topology），[C] 官方规格披露·一手权威——「Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized」/「each GPU rail onto a different first level (LEAF) switch」/「NVSwitch path... one switch hop away」逐字（§3，10.1 recap）
- [NVIDIA, NCCL 2.12 blog](../references.md)：Doubling all2all Performance with NCCL 2.12（key: nccl-all2all-blog），[C] 开放技术博客、非闭源，2022-02-28——rail-optimized「NIC-0... same leaf switch (L0)... minimize network interference between flows... lighter connections between rails」/ PXN「first move data on a GPU on the same rail as the destination... without crossing rails」逐字（§3/§7）
- [NVIDIA, NCCL 2.4 blog](../references.md)：Massively Scale Deep Learning Training with NCCL 2.4（key: nccl-tree-blog），[C] 开放技术博客——Summit 24576 GPUs 双树对 ring 延迟最高 180× 改善（§8，8.3 已引）
- [NVIDIA, ConnectX-7](../references.md)：ConnectX-7 NDR 400G adapter（key: connectx7），[C] 官方规格披露·一手权威——Max total 400Gb/s = 50 GB/s 单向 / PCIe Gen5 / In-Network Computing（SHARP）（§1/§6，9.2 recap）
- [Zhu et al., 2015](../references.md)：Congestion Control for Large-Scale RDMA Deployments（DCQCN，key: dcqcn），[B] 老经典无 arXiv，SIGCOMM 2015——RP/CP/NP + ECN/CNP 速率调节 + PFC 局限（§7 recap，9.3 机制详展）
- [Crupnicoff et al., Mellanox](../references.md)：Deploying QoS and Congestion Control in InfiniBand-based Data Center Networks（key: ib-cc-whitepaper），[C] 官方规格披露·一手权威——IB CCA FECN/BECN/CCT（§7 recap，9.3 详展）
- 8.1 / 8.2 / 8.3 / 8.4 / 7.1 / 7.2 / 9.2 / 9.3 / 10.1（纯文字指引，不挂链接）：八原语消息量与 a2a 流量模式（8.1）、ring 下界与多 ring（8.2）、双树与 ring/tree 交叉（8.3）、NCCL 模块全量走读（8.4）、NVLS 流量账（7.1 §6）、有效带宽 370.8 与节点出口 9×（7.2）、GDR min 与多 NIC 聚合（9.2）、DCQCN/PFC/IB CC 机制（9.3）、fat-tree 收敛比与 rail-optimized（10.1）——recap 不重推（§2/§3/§4/§6/§7/§8）
- inference 7.4（纯文字指引，不挂链接）：超节点多 job NVLink 干扰 doc L297、整超节点单 EP 组（§7 recap）；10.3 / 10.4 / 11.2 / cluster（纯文字指引）：a2a/DeepEP（10.3）、sendrecv/Ring Attention（10.4）、drill-down（11.2）、作业编排（cluster）——下游边界（§9）
