# 8.4 NCCL 实现走读（NCCL Implementation Walkthrough）

> 8.1 说 ncclAllReduce 返回只是入队、不是传输完成——那入队之后，是谁把 1.75 GB 真正搬过 NVLink？同一个调用，NCCL 怎么在 ring/tree/NVLS 等七种算法与 LL/LL128/Simple 三种协议里挑出一组、再落到一个 kernel？LL 协议为什么宁可扔掉一半带宽也要用？本章对 NVIDIA/nccl **2.32.7-1**（commit 351f33fd）做模块全量走读：初始化流程（ncclCommInitRank → bootstrap → 拓扑 → 图搜索 → transport → devComm）、channel 构建（rings.cc/trees.cc，MAXCHANNELS 64）、transport 抽象（p2p/shm/net/collNet，NVLS 独立）、proxy 线程（GPU-NIC 协调）、三协议（LL 16B line 50% / LL128 128B line 93.75% / Simple 512B，命题 1 效率 + 命题 2 延迟-带宽交叉）、算法选择 cost model（ncclTuningGetTime 的 α-β 代码形态）、NVLS（cuMulticastCreate）、kernel 实现（src/device/ runRing/runTreeUpDown/runTreeSplit） | 前置：8.1（API 与 regime）、8.2（ring 下界与多 ring 式 (6)）、8.3（双树与 ring/tree 交叉）、7.1 §6（NVLS 流量账）、7.2（有效带宽 370.8 GB/s）、7.3（peer 映射）；下游 9.2（NET/GDR）、10.2（fabric 工程）、10.3（a2a）、10.4（sendrecv） | 锚点：NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd087e，fresh tarball 全量 file:line 核实）、NCCL docs（[NVIDIA, NCCL docs](../references.md)，[C] 官方规格披露·一手权威，2.32.3 滞后 repo 注）、NCCL 2.4 blog（[NVIDIA, NCCL 2.4 blog](../references.md)，[C] 开放技术博客）——代码锚 2026-11-25 一手核实

## 1. 三个问题与本章口径

ncclAllReduce 返回时数据还没动（8.1 §7 stream 入队语义），那之后是谁在搬数据？一个调用要经过「选算法 → 选协议 → 切 channel → 发射 kernel → proxy 推进」整条链，这条链在 repo 里各落在哪个文件？LL 协议每 16 B 只搬 8 B 有效数据、带宽效率只有一半，为什么小消息还非用它不可？本章回答三个问题：ncclCommInitRank 到 kernel 发射的完整代码路径是什么？transport/proxy/协议三层各解决什么硬件约束？算法与协议的延迟-带宽账怎么在 cost model 里逐行兑现？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（ $n = 8$，NVLink4 + 4 颗第三代 NVSwitch，几何 recap 7.1/7.2）；有效带宽取 **NCCL 模型 370.8 GB/s 单向**（7.2 式 (2)，18 × 20.6）； $\alpha = 2$ μs 为**教学假设口径**（7.2 §3）；贯穿实例 $M = 1$ GB = $10^{9}$ B、bf16。**版本锚**：NVIDIA/nccl **2.32.7-1**，commit **351f33fd087e0c4d0cfb99f1c61dc13da7996c28**（master，2026-11-24「Merge v2.32.7-1 release into master」，makefiles/version.mk L9-11 MAJOR 2 / MINOR 32 / PATCH 7，+L13 PKG_REVISION 1 为「2.32.7-1」的「-1」出处）；本章全部 file:line 按该 commit 的 fresh tarball 核实（2026-11-25）。**结构漂移注**（vs 旧布局，阅读老资料必对）：`src/collectives/device/` → **`src/device/`**；`src/graph/tuning.cc` → **`src/tuning/`**（per-algo cost model）；`src/enqueue.cc` → **`src/enqueue/`**；`devcomm.h` → `device.h` + `plugin/nccl_tuner.h`；算法枚举新增 **NCCL_ALGO_PAT=6 / NVLS_TREE=5**。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，recap training 4.1 时 $d \to n$ 翻译）； $M$ = 被归约向量总字节； $C$ = 每 rank 流量字节（ring all-reduce $C = 2(n-1)/n \cdot M$，本章局部符号）； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $k$ = 并行 channel 数（沿 8.2）； $\eta_{\mathrm{LL}}$ / $\eta_{\mathrm{LL128}}$ = 协议带宽效率、 $C^{\ast}$ = 协议交叉流量（§6 局部符号）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 $T_{\mathrm{comm}}$ / $T_{\mathrm{ring}}$ 下标。cost model 代码变量 lat（μs）/ bw（GB/s）以 code span 给出，与 $\alpha/\beta$ 的教学口径分开（§7 口径注）。

**recap 边界声明**（强制，spec §4.2）：8.1 原语语义与 API、8.2 ring 带宽最优性下界（命题 1）与多 ring 式 (6)、8.3 双树带宽最优（命题 1）与 ring/tree 交叉 $M^{\ast}$、7.1 §6 NVLS 流量账、7.2 有效带宽折扣——**均已交付，本章只 recap + 指引，不重推**。本章增量 = NCCL repo 模块全量走读 + 协议效率与交叉的定量账（命题 1/2）+ cost model 逐行对照。

**两层走读纪律声明（8.4 ↔ 10.2，强制）**：本章 = **代码机制层**——repo 里每个模块怎么实现、公式怎么落进代码；10.2 = **fabric 工程层**——拓扑感知 ring/tree 的跨机映射、rail 对齐、IB SHARP 部署、多 channel/多 NIC 调优、多作业干扰（W5）。凡涉及「怎么把环摆到物理 fabric 上」「SHARP 交换机怎么配」「NCCL_TOPO_FILE 怎么调」一律归 10.2，本章只走读代码机制、不越界到工程。**代码 scope 声明**：本章 = 模块走读级（init/graph/transport/proxy/tuning/device 全量）；GDR 硬件机制归 9.2，a2a/DeepEP 归 10.3，sendrecv/Ring Attention 归 10.4，实测 drill-down 归 11.2。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. 初始化：ncclCommInitRank 的完整流程

一个 communicator 从无到所要经过哪些步骤？公共 API 入口是 `ncclCommInitRank`（src/init.cc L3071，[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1），它把建组拆成一条异步 job 链，主体在 `initTransportsRank`（L1219，调用点 L2320）。全流程如下（file:line 均 fresh tarball 核实）：

```
ncclCommInitRank            init.cc L3071   公共 API（newcomm, nranks, commId, myrank）
  └ ncclCommInitRankDev     init.cc L2975   → 异步 job（ncclCommInitRankAsyncJob L2132）
     └ ncclCommInitRankFunc init.cc L2229   job 主体
        ├ bootstrapInit     init.cc L2314   带外 allgather，建 rank 间 TCP 连接
        └ initTransportsRank init.cc L1219  初始化主体：
           ├ ncclTopoGetSystem   init.cc L1447 / topo.cc L1994  NVML/PCI/NVLink → XML → topo
           ├ ncclTopoComputePaths init.cc L1453 / paths.cc      GPU/NIC/CPU 路径类型 + 带宽
           ├ ncclTopoCompute（逐算法图搜索，search.cc L1245）：
           │    ringGraph        init.cc L1488  （maxChannels = MAXCHANNELS/2 = 32）
           │    treeGraph        init.cc L1496  （channels = ring 的 nChannels）
           │    collNetChainGraph init.cc L1513 / collNetDirectGraph L1515
           │    nvlsGraph        init.cc L1525  （nvlsSupport 时）
           ├ ncclTransportP2pSetup init.cc L1957  transport 连接建立
           └ devCommSetup      init.cc L2076 / L714  host→device 拷贝（L847）
```

**拓扑检测**把机器读成一棵 XML 树再转成 topo system：`ncclTopoGetSystem`（topo.cc L1994）经 NVML/PCI 枚举 GPU、NIC、CPU 与桥，`ncclTopoGetSystemFromXml`（L1072）解析；NVLink 边按 `nvlBw = ncclTopoNVLinkBw(compCap)`（topo.cc L944）构建，per-link 带宽取 **SM90_NVLINK_BW 20.6 / SM100_NVLINK_BW 40.1 GB/s 单向**（topo.h L24/L26，按 cudaCompCap 分派 L315-323）。**口径注**：20.6/40.1 是 NCCL 内部**有效带宽模型值**（含协议/编码折扣），低于硬件 raw 25/50 GB/s（7.1 规格口径）——两层账分离，7.2 式 (2) 的 370.8 = 18 × 20.6 正是这个模型值，勿与 raw 混。

**路径类型**是一架梯子（graph.h L120-148，数值越小越近）：PATH_LOC 0（自身）/ PATH_NVL 1（NVLink 直连）/ PATH_NVB 2（经中间 GPU 的 NVLink）/ PATH_C2C 3 / PATH_PIX 4（单 PCIe 桥）/ PATH_PXB 5（多 PCIe 桥）/ PATH_P2C 6 / PATH_PXN 7（经中间 GPU 到 NIC，rail 聚合）/ PATH_PHB 8（经 CPU root complex）/ PATH_SYS 9（跨 NUMA）。`ncclTopoComputePaths`（paths.cc）给每对 GPU/NIC 定路径：PATH_PHB（L104）、PATH_PXN（L243）、p2pLevel = PATH_NVL 判 P2P 可行（L341）。这架梯子后面两处直接用：transport 选择（§4）与 LL128 门控（§6）。

**图搜索**对每种算法各搜一张 `ncclTopoGraph`（graph.h L173-187）：结构体持 `pattern / minChannels / nChannels / bwIntra / bwInter / latencyInter / typeIntra`——**这就是 8.2/8.3 带宽账的代码载体**（bwIntra/bwInter 喂 cost model，nChannels 喂多 channel 聚合）。搜索入口 `ncclTopoCompute`（search.cc L1245，每算法算一次缓存），递归体 `ncclTopoSearchRec`（L331）/ `ncclTopoSearchRecGpu`（L357）按路径类型与带宽约束找环/树。搜索得到的 prev/next 邻接怎么落成环数组、双树怎么互补，§3 走读。

## 3. channel 构建：ring/tree 图落到多 channel

一张环图怎么变成多个并行 channel？channel 是 NCCL 的并行单位——**一个 channel 对应 kernel 里一个 CTA**（blockIdx.x 映 channelId，common.h L436），多 channel 就是把消息拆到多个 CTA 同时搬。

**ring 落 channel**：`ncclBuildRings`（src/graph/rings.cc L29）从本 rank 出发沿 next 指针走 $n$ 步生成环数组，并做两道校验——「ring does not loop back to start」（走 $n$ 步须回起点，L51）与全 rank 覆盖校验（每 rank 恰出现一次，L55-60 位图快查）。这两道校验保证逻辑环是覆盖全 rank 的单一循环（8.2 §4 已 recap，不重推）；本章增量是 **nrings 维度**：函数对 `r = 0..nrings-1` 各建一条环，每条环落一个 channel。ringGraph 的 `maxChannels = MAXCHANNELS/2 = 32`（init.cc L1487，MAXCHANNELS 64 见 device.h L80），即环最多 32 个 channel。

**tree 落 channel**：`ncclGetBtree`（trees.cc L32）按 power-of-two 位模式建单棵二叉树，`ncclGetDtree`（L90）建双树——第一棵用 btree，第二棵在 nranks 为偶数时用 mirror、nranks 为奇数时用 shift，注释逐字「mirror tree (if nranks is even)... shift it by one rank (if nranks is odd)」（L69-88，含 12/13-rank 双树图）。双树互补性与带宽最优性已在 8.3 命题 1 证明，本章不重证，只走读构建入口。treeGraph 的 channel 数直接取 ring 的 nChannels（init.cc L1494-1495），保证 ring/tree 切换时 channel 布局一致。

**图 → channel 连接**：`ncclTopoPreset`（connect.cc L22）预设、`ncclTopoPostset`（L395）把搜好的图落成 channel 的 ring/tree 邻接（每 channel 的 prev/next、tree 的 up/down）。至此每个 channel 拿到自己的环排列或树位置，kernel 发射时按 channelId 取用（§9）。

**多 channel 带宽聚合**（recap 8.2 式 (6)，不重推）： $k$ 个 channel 各搬 $M/k$，带宽项除以 $k$、延迟项不变：

$$
T_{\mathrm{ring}}(k) = 2(n-1)\thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \frac{\beta}{k} \qquad (\text{8.2 式 (6) recap})
$$

代入 $n=8$、 $M=1$ GB、 $\alpha=2$ μs： $k=1$ 带宽项 4.72 ms、 $k=4$ 1.18 ms、 $k=8$ 0.59 ms（延迟项恒 0.028 ms）。**口径注**：线性聚合假设各 channel 走不同 link 子集且 NVSwitch 非阻塞； $k$ 增大到每 GPU 注入带宽（370.8 GB/s 模型口径）饱和后不再增益（7.2 §3）。MAXCHANNELS 64 是结构上限，实际 nChannels 由图搜索按拓扑定（通常远小于 64）。多 channel 的 fabric 侧映射（跨 NIC、rail 对齐）归 10.2。

## 4. transport 抽象：p2p / shm / net / collNet

同一对 rank 之间，数据走哪条物理路径由谁定？transport 层回答这个。注册表只有四项（src/transport.cc L15-20）：

```c
struct ncclTransport* ncclTransports[NTRANSPORTS] = {
  &p2pTransport, &shmTransport, &netTransport, &collNetTransport,
};
```

**NVLS 不在此表**——它走独立的 nvls.cc/multicast.cc 路径（§8），因为 NVLS 是网内归约而非点对点搬运。每个 transport 实现 `canConnect`（判可行）/ `sendSetup` / `recvSetup`（建连接）/ `proxyProgress`（推进）四组回调，选择时按拓扑逐一试 canConnect。

**p2p（NVLink / PCIe 直访）**：`p2pCanConnect`（p2p.cc L130）按路径类型判 P2P 可行；`p2pMap`（L358）把 peer 显存映射进本进程地址空间——同进程不同 GPU 走 `cudaDeviceEnablePeerAccess`，跨进程走 **`cudaIpcOpenMemHandle(cudaIpcMemLazyEnablePeerAccess)`**（L331，IPC 句柄导入）。映射建成后 GPU kernel 直接 load/store peer HBM，**数据路径不经 CPU 拷贝**（7.3 peer 映射 recap）。`p2pSendSetup`（L403）里 `useRead`（L410）决定 read 还是 write 语义：NVLink 上 read（拉）与 write（推）带宽不对称，`p2pGetInfo` 按拓扑选；`useMemcpy` 时强制 read=0。read/write 的带宽差账 recap 7.2 §3。

**shm（同机共享内存）**：`shmCanConnect`（shm.cc L61）要求同 host（hostHash 相等）且非 net；用于**同机 PCIe 但 P2P 不可行**时（如 ACS/IOMMU 阻断，7.3 §5），经 host 共享内存中转。`shmSendSetup`（L88）/ `shmRecvSetup`（L121）建共享缓冲，数据由 **proxy 线程拷贝**（§5）——比 p2p 多一次 host  bounce，带宽延迟都差。

**net（跨机 RDMA / socket）**：`sendSetup`（net.cc L296）/ `recvSetup`（L356）经 `ncclProxyCallBlocking` 把建连请求交给 proxy 线程，协商 net plugin 属性与 GDR 能力。日志逐字带 `/GDRDMA` 标记（L325，useGdr）；**GDRCOPY_SYNC_ENABLE 默认 1 / GDRCOPY_FLUSH_ENABLE 默认 0**（L338/L340，gdrcopy 同步字 + PCIe read flush）。GDR 让 NIC 直访 GPU HBM、省 host bounce（硬件机制归 9.2），本章只走读 net transport 怎么协商与推进它（§5）。

**collNet（网内归约链）**：`canConnect`（coll_net.cc L143）/ `sharedBuffersInit`（L416）建 CollNet 共享缓冲，是 IB SHARP 集成的 NCCL 侧链（SHARP 部署归 10.2）。

**transport 对延迟的影响**（cost model 内置校准值，cost_model.cc hwLatencies，NVLink L164 / PCI L174 / NET L184）：ring + Simple 协议的每跳硬件延迟，NVLink **3.4 μs** / PCI **5.7 μs** / NET **14.0 μs**——NET 是 NVLink 的约 4.1×、PCI 约 1.7×。这就是「高频集合留 NVLink 域内」的延迟侧定量依据（带宽侧 9× 差 recap 7.2 §7）。**口径注**：这些是 NCCL 内置校准常数（非本章 $\alpha=2$ μs 教学假设），两套口径不混；真实值随平台/驱动变，以实测为准。

## 5. proxy 线程：GPU 与 NIC 之间的 CPU 代理

GPU kernel 可以直接 load/store peer HBM（p2p），为什么跨机还要一个 CPU 线程？因为 **NIC 的 verbs/socket 接口是 host 侧 API，GPU kernel 调不了**——必须有一个 CPU 线程替 GPU 向 NIC 提交请求、轮询完成。这就是 proxy 线程。

**线程入口**：`ncclProxyProgress`（src/proxy.cc L956，每 comm 一线程）循环调 `progressOps`（L801），后者遍历活跃 proxy args、逐个调 `op->progress`（即各 transport 的 proxyProgress 回调）。GPU kernel 侧经 `ncclProxyPost`（L481）把 op 提交进池，proxy 线程取出推进。

**net 进度状态机**：`sendProxyProgress`（net.cc L1327）/ `recvProxyProgress`（L1496）是 GPU-NIC 协调的核心。每个 sub 维护三个计数器 **posted / transmitted / done**，在 `NCCL_STEPS = 8`（device.h L26）个缓冲槽上轮转：posted（向 GPU 发布可写槽）→ transmitted（向 NIC 提交 isend/irecv）→ done（test 轮询到完成）。`maxDepth = min(NCCL_STEPS, NCCL_SHARED_STEPS/nsubs)` 限制在途深度，防止槽位覆写。数据流如下（GDR vs 非 GDR 两条路径）：

```
GDR 路径（NIC 直访 GPU HBM，9.2）：
  GPU kernel ──写数据──► GPU HBM 缓冲
                          │  proxy 只提交描述符（不拷数据）
  proxy 线程 ──isend──► NIC ──RDMA READ──► GPU HBM ──► fabric
  proxy 线程 ◄──test 完成── NIC（done++）

非 GDR 路径（host bounce）：
  GPU kernel ──写──► GPU HBM ──proxy 拷──► host 缓冲 ──NIC──► fabric
                              （多一次 D2H + CPU 拷贝，带宽延迟都差）
```

**关键区分**：GDR 下 proxy 只做「提交 + 轮询」的控制面，数据面 NIC 直读 GPU HBM（零拷贝）；非 GDR 下 proxy 还要做 D2H 搬运的数据面。p2p transport 也有 proxy 辅助回调（p2pSendProxySetup，p2p.cc L697），但**数据路径 GPU 直访、不经 proxy 拷贝**——proxy 只在 P2P 建连/同步时介入。SHM 则数据面也经 proxy 拷贝（§4）。gdrcopy 同步字（GDRCOPY_SYNC_ENABLE）让 proxy 把 RX tail 放进 CUDA 内存、省一次 PCIe read flush（net.cc L337-340）。GDR 的 NIC→HBM 带宽账与 peer memory 机制归 9.2，本章只走读 proxy 怎么协调。

## 6. 协议：LL / LL128 / Simple 的延迟-带宽权衡

搬同样一份数据，为什么有三种协议？因为**同步机制的开销与带宽利用率不可兼得**：小消息要低延迟（同步要轻），大消息要高带宽（同步开销要摊薄）。三协议各占一端。

**LL（Low Latency）**：line 布局是 `union ncclLLFifoLine`（ll__types.h L14-26），`static_assert(sizeof == 16)`（L26）钉死 16 B：

```
LL line（16 B）：  [data1 4B][flag1 4B][data2 4B][flag2 4B]
                    └── 有效数据 8 B ──┘└── flag 8 B ──┘   → 效率 8/16 = 50%
LL128 line（128 B）：[data 15×8B = 120 B][flag 1×8B = 8 B]   → 效率 120/128 = 93.75%
Simple：无 per-line flag，512 B grain，proxy + fence 同步    → 效率趋 100%
```

注释逐字（L15-16）：「Flags follow data so that a partial network receive cannot expose a flag before its data. This assumes contiguous network writes or eight-byte write atomicity」——**flag 跟在 data 后面**，保证部分接收不会先暴露 flag 后到 data。接收侧 `readLL`（prims_ll.h L102-115）用 `ld.volatile.global.v4.u32` 一次读 16 B（L108），**自旋直到 flag1 == flag 且 flag2 == flag**（L113）；flag 是 step 序号，**flag 即同步，无独立 fence**——这就是 LL 低延迟的来源（省掉 memory fence 与 proxy 往返）。代价是每 16 B 只搬 8 B 有效数据。

**LL128**：line 128 B = 16 个 8 B 槽，`NCCL_LL128_LINEELEMS = 16`、`DATAELEMS = 15`（ll128__types.h L13-15），即 120 B data + 8 B flag。每 line 最后一个 thread 持 flag（`NCCL_LL128_FLAGTHREAD = LINEELEMS-1`，prims_ll128.h L10）。**LL128 要求 128 B 写原子性**——只有 NVLink 能保证，故 `isLL128Enabled`（cost_model.cc L125）门控：intraType 不高于 PATH_NVB、interType 不高于 PATH_PXN/PXB（§2 路径梯子）。NCCL docs 逐字警告：「enabling LL128 on platforms that don't support it can lead to data corruption」（[NVIDIA, NCCL docs](../references.md)，[C]）。LL128 的 warp 粒度 = 32 × 8 / 16 × 15 × 8 = **1920 B**（device.h ncclProtoGrainSize L304-310）。

**Simple**：`Primitives<ProtoSimple<...>>` 特化（prims_simple.h L20），**无 per-line flag**，同步靠 proxy + memory fence + connFifo step 计数器（`waitPeer` L104 自旋等 step）；粒度 512 B（device.h L308）。大消息下 flag 开销摊薄到趋零、满带宽，代价是每跳延迟最高（§4 表 Simple 3.4 μs vs LL 0.6 μs）。`genericOp`（L179）统一 send/recv/recvReduceSend/recvReduceCopySend 数据路径。

**命题 1**（协议带宽效率）。设 line 携带的有效数据字节为 $D$、line 总字节为 $L$，则带宽效率 $\eta = D/L$。三协议各为：

$$
\eta_{\mathrm{LL}} = \frac{8}{16} = 0.5, \qquad \eta_{\mathrm{LL128}} = \frac{120}{128} = 0.9375, \qquad \eta_{\mathrm{Simple}} \to 1 \qquad (1)
$$

**证明**（路线：逐 line 数 data 与 flag 字节，三步）。其一，LL line 16 B = data1/flag1/data2/flag2 各 4 B（ll__types.h L17-22），data 共 8 B， $\eta = 8/16 = 0.5$。其二，LL128 line 128 B = 16 槽 × 8 B，DATAELEMS = 15（ll128__types.h L15），data 120 B、flag 8 B， $\eta = 120/128 = 0.9375$。其三，Simple 无 per-line flag，512 B grain 全为数据，flag/同步开销按 grain 摊薄趋零， $\eta \to 1$。∎ 代入 $M=1$ GB、 $n=8$ ring（每 rank $C = 1.75$ GB）：带宽项 Simple 4.72 ms、LL128 $4.72/0.9375 = 5.03$ ms、LL $4.72/0.5 = 9.44$ ms——**LL 用一半带宽换最低延迟**。

**命题 2**（协议延迟-带宽交叉）。设两协议固定延迟 $`\mathrm{lat}_{\mathrm{LL}} \lt \mathrm{lat}_{\mathrm{Simple}}`$、效率 $\eta_{\mathrm{LL}} \lt \eta_{\mathrm{Simple}} = 1$，每 rank 流量 $C$，带宽 $\mathrm{BW}$。时间模型 $`T_{\mathrm{LL}} = \mathrm{lat}_{\mathrm{LL}} + C/(\eta_{\mathrm{LL}} \mathrm{BW})`$、 $`T_{\mathrm{Simple}} = \mathrm{lat}_{\mathrm{Simple}} + C/\mathrm{BW}`$，交叉流量：

$$
C^{\ast} = \Delta\mathrm{lat} \cdot \mathrm{BW} \cdot \frac{\eta_{\mathrm{LL}}}{1 - \eta_{\mathrm{LL}}}, \qquad \Delta\mathrm{lat} = \mathrm{lat}_{\mathrm{Simple}} - \mathrm{lat}_{\mathrm{LL}} \qquad (2)
$$

$C \lt C^{\ast}$ 时 LL 赢（延迟主导）、 $C \gt C^{\ast}$ 时 Simple 赢（带宽主导）。

**证明**（路线：令两式相等 → 解 $C$，三步）。其一，令 $T_{\mathrm{LL}} = T_{\mathrm{Simple}}$，移项 $\Delta\mathrm{lat} = C/\mathrm{BW} \cdot (1/\eta_{\mathrm{LL}} - 1)$。其二， $1/\eta - 1 = (1-\eta)/\eta$，故 $\Delta\mathrm{lat} = C(1-\eta_{\mathrm{LL}})/(\eta_{\mathrm{LL}} \mathrm{BW})$。其三，解出 $C^{\ast} = \Delta\mathrm{lat} \cdot \mathrm{BW} \cdot \eta_{\mathrm{LL}}/(1-\eta_{\mathrm{LL}})$。∎

全数字代入（ring baseLatencies Simple 8.4 − LL 6.6 = 1.8 μs，cost_model.cc L152； $\mathrm{BW} = 370.8$ GB/s； $\eta_{\mathrm{LL}} = 0.5$）： $C^{\ast} = 1.8 \times 10^{-6} \times 370.8 \times 10^{9} \times 0.5/0.5 = 667440$ B ≈ **667.4 KB**（每 rank 流量）；对应消息尺寸 $M^{\ast} = C^{\ast} \cdot n/(2(n-1)) = 667440 \times 8/14 \approx 381.4$ KB。验证： $C = C^{\ast}$ 时 $T_{\mathrm{LL}} = 6.6 + 3.6 = 10.2$ μs、 $T_{\mathrm{Simple}} = 8.4 + 1.8 = 10.2$ μs，两式逐数闭合。**口径注**（三个临界勿混）：本章 $C^{\ast}$（协议间，LL vs Simple）≠ 8.3 $M^{\ast} = 1.83$ MB（算法间，ring vs tree，实测 $\beta_{\mathrm{eff}}$ 差）≠ 8.1 $M_c = 5.93$ MB（ring 内部延迟/带宽两项临界）——三者驱动机制不同，数值也不同。式 (2) 是教学简化模型（固定延迟取 baseLatencies 差），真实选择走 §7 完整 cost model。

**缓冲尺寸**（init.cc L1008-1011）：LL = 8 × 512 × 8 × 16 = 512 KiB、LL128 = 120 × 640 × 8 × 8 = 4.6875 MiB、Simple = 4 MiB（DEFAULT_BUFFSIZE = 1<<22）。LL 缓冲小（line 小、step 浅），Simple 缓冲大（满带宽流水）。

> **坑**：在不支持 128 B 原子性的平台强开 LL128。触发条件：PCIe 平台设 NCCL_PROTO=LL128；现象：flag 与 data 非原子到达，接收侧读到半 line，**数据损坏**（docs 逐字 data corruption）；规避：LL128 只在 NVLink（intraType 不高于 PATH_NVB）启用，isLL128Enabled 已门控，勿绕过。

## 7. 算法选择与 cost model：α-β 模型的代码形态

七种算法 × 三种协议，NCCL 怎么挑？答案是对每个 (算法, 协议) 组合算一个预估时间，取最小。算法枚举（plugin/nccl_tuner.h L28-35）：**TREE 0 / RING 1 / COLLNET_DIRECT 2 / COLLNET_CHAIN 3 / NVLS 4 / NVLS_TREE 5 / PAT 6**；协议枚举（L39-41）：LL 0 / LL128 1 / SIMPLE 2。PAT（Parallel Aggregated Tree，2.23+）与 NVLS_TREE（多节点 NVLS，2.18+）是新增算法；symmetric kernels / GIN / RAS / CE 为 2.28+ 新特性（本章走读 scope 以 ring/tree/NVLS/CollNet + 三协议为主，新特性注存在即可）。

**时间公式**：`ncclTuningGetTime`（tuning_general.cc L58）是 α-β 模型的代码形态：

```c
float time = lat * latCount * precision_ratio;
if (bw > 0) time += inputs->nBytes / (1000 * bw) * precision_ratio;
```

**单位对账**（lat 单位 μs、bw 单位 GB/s）：带宽项 `nBytes/(1000*bw)`——nBytes 单位 B、bw 单位 GB/s = $10^{9}$ B/s，故 nBytes/bw 单位 $10^{-9}$ s = ns，再除 1000 得 **μs**，与 lat 项同单位相加。这正是 $\alpha$（延迟项）+ $M\beta$（带宽项）的结构：lat×latCount 对应跳数×每跳延迟，nBytes/(1000×bw) 对应字节×每字节时间。代入 `nBytes` $= 1.75 \times 10^{9}$ B（ring 每 rank 流量 $C$）、 `bw` $= 370.8$ GB/s：带宽项 $= 1.75 \times 10^{9} / (1000 \times 370.8) = 4719.5$ μs ≈ 4.72 ms，与 §6 Simple 带宽项逐数一致。`latCount`：ring = numPipeOps，tree = DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)（L59）。**FP8 降级**：ring 且 `nRanks > 8` 时 precision_ratio = 1024（L65-67），注释逐字「Relegate fp8 reduction trees of sufficient depth that they incur precision loss to be least preferred」——深环 FP8 归约精度损失大，乘大因子把它排到最末。

**per-algo 模型**（src/tuning/ 各文件）把拓扑带宽折成算法带宽：

- **ring.cc**：busBw = bw × nChannels；LL 取 `min(llMaxBw, busBw×0.5)`（L40，**50% 效率的代码兑现**，命题 1）；LL128 取 `busBw×0.92`（L43，注释 120/128，略保守）；`generalBandwidths = busBw×nRanks/nSteps`（L47，all-reduce nSteps = 2(n-1)，n=8 时 ×8/14 ≈ 0.571，对应 2M 流量）。延迟项 `(nSteps−nInterSteps)×intraLat + nInterSteps×interLat`（L86-87）。`ringSimplePlateau`（L109-139）：多节点 AllReduce 或 Blackwell 单节点 NVLink AR/AG/RS 时 lat × plateauFactor（≥sm80 取 1.4、低于 sm80 取 1.9），按 per-rank/channel 字节与 step 数调制——环的延迟平台效应。
- **tree.cc**：**仅 AllReduce**（其余 func hard disable，L23-28）；busBw ×0.92（L33）；LL ×1/3.8（L35）；LL128 单节点 ×7/9、多节点 ×120/128（L38-40，**单节点 7/9 ≈ 0.778 比理论 93.75% 更保守，含 tree incast 余量**）；`generalBandwidths = busBw×0.5`（L44，**all-reduce 2M 流量 → 算法带宽 = bus 带宽之半**，8.3 §4 回链）；延迟项 `+= 2×((ppn−1)×intraLat + log2i(nNodes)×interLat)`（L57-58，**log2 树深延迟项的代码实化**，8.3 O(log n) 锚）；TreeModelSim 再按 `treeCorrectionFactor[proto][logSize]` 尺寸修正（L79-81，logSize = log2i(nBytes>>6)）。

**选择**：`ncclTuningSelectBestTuning`（tuning.cc L153）遍历候选取最小 timeUs，注释逐字「The tuning with the lowest simulated time/cost is determined to be best」。环境变量 **NCCL_ALGO / NCCL_PROTO** 经 parseList 解析（cost_model.cc L365-379，支持 ^ 排除与 per-function 分号语法）强制候选集。选择结果落 task：`task->algorithm = tuningOut.algo; task->protocol = tuningOut.proto`（task_posttuning.cc L547-548）；输入装配 fillCollTuningInput（task_pretuning.cc L24，调用 L248）。

**口径注**（模型值 vs 理论效率）：命题 1 的 $\eta$ 是 line 布局的**理论上界**；cost model 再乘算法/平台修正（ring LL128 ×0.92、tree 单节点 ×7/9、ring Simple plateau ×1.4）——两层账，理论效率管「线上最多搬多少」，模型修正管「这个算法/平台实际拿多少」。LL 调度侧还把 trafficBytes ×4（enqueue.cc L554）：line 开销 2×（16 B 搬 8 B）之上再计 2× 保守调度余量（含双 flag 开销口径），让调度器按膨胀流量分配 channel，勿把 LL 任务塞太密。

## 8. NVLS 与 CollNet：把归约搬进交换芯片

ring/tree 的归约都在 GPU 的 SM 上做，能不能让交换芯片代劳？NVLS 就是这条 bypass（7.1 §6 机制 recap，不重推流量账）。

**multicast 组建立**：`ncclNvlsInit`（nvls.cc L246）探测硬件支持——经 **`cuMulticastCreate`**（L270）查 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`，日志逐字「NVLS multicast support is %savailable」（L277）；`ncclNvlsSetup`（L494）建 multicast 组与缓冲。可用性：第三代 NVSwitch（NVLink4）+ Hopper 及以后（NCCL docs NCCL_NVLS_ENABLE 节逐字「NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures」，[C]）。多节点变体 NVLS_TREE 经 `ncclNvlsTreeConnect`（L283）连节点间树。

**kernel 侧**：`RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>`（all_reduce.h L390）把 warp 分成 scatter / gather / reduce / bcast 四组：scatter 组用 `FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>`（L421，只发不收，把数据写进 multicast 组，NCCL_MAX_NVLS_ARITY 32 见 device.h L192），gather 组反向收全和，reduce 组经 `nvls->down` 做网内归约-广播。**归约计算从 SM 移进 NVSwitch**，通信 kernel 的 SM 占用下降（7.1 §6 wp 逐字「significantly reduces the load on SMs」）。

**流量账 recap**（7.1 §6，不重推）：NVLS 每 rank 发送 $M$（写 multicast 组），对比 ring 的 $2(n-1)/n \cdot M$，比值 $2(n-1)/n$： $n=8$ 时 1.75。代入 $M=1$ GB：NVLS 带宽项 $10^{9}/(370.8 \times 10^{9}) \approx 2.70$ ms，ring 4.72 ms——这是 wp「up to 2x throughput gain」的流量侧来源。**CollNet**（coll_net.cc）是 IB SHARP 的 NCCL 侧链（§4），把归约搬进 IB 交换机；NVLS（NVLink 域）vs IB SHARP（跨机）的对照与部署归 10.2。

## 9. kernel 实现：src/device/ 的 dispatch 与三算法骨架

选好的 (算法, 协议) 怎么变成一个 kernel？NCCL 用**编译期模板特化 + 函数指针表**做 dispatch，运行期零分支。

**入口链**：`ncclDevKernel_Generic`（common.cu L23，`__global__`，args4K）→ `ncclKernelMain`（common.h L420，按 SpecializedFnId dispatch）→ `RunWorkBatch`（L303，work 批处理）→ `RunWorkColl<Fn, T, RedOp, Algo, Proto>` 特化 → 具体算法骨架。函数指针表 `ncclDevFuncTable`（common.h L29-31）按 (coll × redop × type × algo × proto) 索引。**实例化生成**：generate.py L16 `all_protos = ["LL","LL128","SIMPLE"]`、L17 all_algos 七种，per 组合生成独立 kernel（编译期 dispatch，kernel 数爆炸式特化，按架构/精度裁剪）。

**runRing**（all_reduce.h L14，template<T, RedOp, Proto>）：`Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>`（L31，prev/next 各 1，环邻居）；循环体逐 chunk 走两阶段——step 0 `directSend`（推自有块），k-2 步 `directRecvReduceDirectSend`（收-归约-发），step k-1 `directRecvReduceCopyDirectSend(postOp=true)`（产出全和块），再 k-2 步 `directRecvCopyDirectSend`（all-gather 转发）+ 末步 `directRecv`。**这与 8.2 的两阶段消息量账逐行对应**（scatter-reduce + all-gather，每 rank 2(n-1)/n·M，recap 不重推）。

**runTreeUpDown**（L86）：归约段 `FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>`（L97，收 3 发 1，NCCL_MAX_TREE_ARITY 3 = 二叉 + local，device.h L163），广播段 `FanAsymmetric<1, NCCL_MAX_TREE_ARITY>`（L121，收 1 发 3）；按 root（up==-1）/ leaf（down[0]==-1）/ 内部节点三分支各走 recvReduceCopy / send / recvReduceSend。

**runTreeSplit**（L146）：把线程拆成归约组与广播组**同时**跑（上下行重叠）。Simple 取 nthreads/2（≥256 再 +64）；LL/LL128 取 70% 归约 / 30% 广播，注释逐字「Receiving from up to 3 sources is more compute intensive than sending to 3 dests. Use 70% for reduce and 30% for bcast」（L161-162）——收 3 路归约比发 3 路更耗计算，故多分线程。

**dispatch 表**（all_reduce.h）：ProtoSimple 的 ring/tree 走 L229-244（tree 在多数 CUDA 版本走 runTreeSplit）；**LL/LL128 只有 ring + treeSplit，无 treeUpDown**（L762-787）——LL 协议下树用 split 重叠而非 updown 串行。channel 与 CTA 一一对应（§3），每 CTA 从 ncclShmem.channel 取自己的 ring/tree/nvls 邻接。

```
ncclDevKernel_Generic (common.cu L23)
  └ ncclKernelMain (common.h L420)  按 SpecializedFnId
     └ RunWorkBatch (L303) → RunWorkColl<Fn,T,RedOp,Algo,Proto>
        ├ RING  + Simple → runRing        (all_reduce.h L14)
        ├ TREE  + Simple → runTreeSplit   (L146)
        ├ TREE  + LL/LL128 → runTreeSplit （无 treeUpDown）
        ├ NVLS  + Simple → scatter/gather/reduce-bcast (L390)
        └ ...（per coll×redop×type×algo×proto 特化，generate.py）
```

> **提点**：per (coll × redop × type × algo × proto) 一份 kernel 的编译期特化看似二进制爆炸，换来的是运行期 dispatch 只剩一次函数指针查表、零分支——选择成本在编译期一次性付清，热路径（每次迭代通信）不再付。这与 CUTLASS 的 manifest 实例化同构（5.2 §9）：把组合成本前置到编译期，运行期只留查表。

## 10. 边界：本章不做什么

走读收口，剩下的归谁？本章交付：初始化全流程（§2）、channel 构建与多 channel 聚合（§3）、transport 四件套与延迟账（§4）、proxy 线程 GPU-NIC 协调（§5）、三协议效率与交叉（命题 1/2，§6）、cost model α-β 代码形态与 per-algo 修正（§7）、NVLS multicast 与 kernel（§8）、device dispatch 与三算法骨架（§9）。以下各有专章：**fabric 工程**（拓扑感知 ring/tree 跨机映射、rail 对齐、IB SHARP 部署、NCCL_TOPO_FILE/多 NIC 调优、多作业干扰）归 10.2；**GDR 硬件机制**（NIC→HBM 带宽账、peer memory、BAR1）归 9.2；**a2a 拥塞与 DeepEP** 归 10.3；**sendrecv 与 Ring Attention** 归 10.4；**实测 drill-down**（Nsight、nccl-tests）归 11.2。代码 scope：模块走读级。

下游 recap 接口：9.2 recap §4 net transport + §5 proxy（GDR 协调）；10.2 recap §2 拓扑检测 + §3 channel + §8 NVLS/CollNet（fabric 侧）；10.3 recap §4 transport（a2a 放置）；10.4 recap §4 p2p + §9 kernel（sendrecv）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| ncclCommInitRank L3071 → bootstrap L2314 → initTransportsRank L1219 → topo L1447 / 图搜索 L1488-1525 / transport L1957 / devComm L2076；版本锚 2.32.7-1 / 351f33fd | §2 流程图 | §2、练习 4 |
| 路径梯子 PATH_NVL 1 / PXN 7 / PHB 8 / SYS 9；SM90_NVLINK_BW 20.6（模型值 vs raw 25 两层账）；ncclTopoGraph 持 bwIntra/bwInter/nChannels | topo.h L24 / graph.h L173 | §2、7.2 |
| channel = 一个 CTA；ncclBuildRings 两道校验 + nrings 维；ncclGetDtree mirror（nranks 偶）/shift（nranks 奇）；MAXCHANNELS 64、ringGraph max 32；多 ring 带宽项 /k（k=4 → 1.18 ms） | 8.2 式 (6) recap | §3、8.2/8.3 |
| transport 表 p2p/shm/net/collNet（NVLS 独立）；p2pMap cudaIpcOpenMemHandle、useRead；shm 同机无 P2P 经 proxy 拷贝；net GDR 协商 | transport.cc L15-20 | §4、8.1 |
| transport 延迟 ring+Simple：NVLink 3.4 / PCI 5.7 / NET 14.0 μs（NET≈4.1× NVLink）；高频集合留域内 | cost_model.cc hwLatencies | §4、7.2 |
| proxy 线程 ncclProxyProgress L956 循环 progressOps；net sendProxyProgress L1327 posted/transmitted/done + NCCL_STEPS 8；GDR 控制面 vs 非 GDR 数据面 | proxy.cc / net.cc | §5、9.2 |
| LL 16B line 8B data 50% / LL128 128B line 120B data 93.75% / Simple 512B 趋 100%；flag 即同步无 fence；LL128 需 NVLink 128B 原子性 | 式 (1)，命题 1 | §6、练习 1/3 |
| 协议交叉 C* = Δlat·BW·η/(1−η) = 667.4 KB（M* ≈ 381.4 KB）；≠ 8.3 M* 1.83 MB ≠ 8.1 $M_c$ 5.93 MB | 式 (2)，命题 2 | §6、练习 2 |
| ncclTuningGetTime time = lat×latCount + nBytes/(1000×bw)（μs，α-β 代码形态）；FP8 ring `nRanks > 8` ×1024 降级；tree busBw×0.5 + log2i 延迟项 | tuning_general.cc L58 | §7、8.2/8.3 |
| ncclTuningSelectBestTuning 取最小 timeUs；NCCL_ALGO/PROTO env；LL trafficBytes×4 调度；七算法含 PAT/NVLS_TREE | tuning.cc L153 | §7 |
| NVLS cuMulticastCreate L270 + kernel scatter/gather/reduce（all_reduce.h L390）；每 rank 发送 M → 2.70 ms（recap 7.1）；CollNet = IB SHARP 链 | nvls.cc / 7.1 §6 | §8、10.2 |
| device dispatch：ncclDevKernel_Generic → ncclKernelMain → RunWorkColl；runRing L14 / runTreeUpDown L86 / runTreeSplit L146；LL/LL128 无 treeUpDown | all_reduce.h / generate.py | §9、练习 4 |

## 练习

1.（计算 · [10]）协议效率与带宽项代入（式 (1)， $n=8$、 $M=1$ GB、 $\mathrm{BW}=370.8$ GB/s）：(a) LL / LL128 / Simple 的带宽效率各多少；(b) ring 每 rank 流量 $C$ 多少 GB；(c) 三协议各自的带宽项多少 ms。校验：(a) 0.5 / 0.9375 / 趋 1（ll__types.h sizeof 16、ll128__types.h DATAELEMS 15）；(b) 1.75 GB；(c) 9.44 / 5.03 / 4.72 ms，与 §6 逐数一致（容差 1%）。
2.（计算 · [20]）协议交叉（式 (2)，ring baseLatencies Simple 8.4 − LL 6.6 = 1.8 μs， $\eta_{\mathrm{LL}}=0.5$， $\mathrm{BW}=370.8$ GB/s）：(a) $C^{\ast}$ 多少 KB；(b) 对应 $M^{\ast}$ 多少 KB（ $n=8$）；(c) $M=64$ KB 与 $M=1$ MB 各哪个协议赢？先预测 (c) 再算。校验：(a) 667.4 KB；(b) ≈381.4 KB；(c) 64 KB LL 赢、1 MB Simple 赢； $C=C^{\ast}$ 时两式均 10.2 μs，与 §6 闭合（容差 1%）。辨析：此 $C^{\ast}$ 与 8.3 $M^{\ast}$（1.83 MB）、8.1 $M_c$（5.93 MB）各驱动什么。
3.（推导 · [30]）证明命题 1（协议效率）并辨析 flag 语义：(a) 逐字段数 LL line 16 B 的 data/flag 字节得 0.5；(b) 数 LL128 line 128 B 的 15 data 槽 + 1 flag 槽得 0.9375；(c) 解释为什么 flag 必须跟在 data 后（partial receive）；(d) 解释 LL128 为什么要求 128 B 写原子性、isLL128Enabled 按什么门控。校验：(a) static_assert sizeof==16（ll__types.h L26）；(b) LINESIZE 128 / DATAELEMS 15（ll128__types.h L13-15）；(c) 注释逐字「Flags follow data」；(d) intraType 不高于 PATH_NVB（cost_model.cc L125），与 §6 一致。
4.（代码 · [30]）NCCL 2.32.7-1 走读（版本锚 351f33fd）：(a) 在 tuning_general.cc 找出 FP8 降级因子与触发条件；(b) tree.cc generalBandwidths = busBw×0.5 对应哪笔消息量账；(c) runRing 的 step k-1 调哪个 primitive、为什么 postOp=true；(d) 为什么 LL/LL128 dispatch 无 treeUpDown。校验：(a) 1024、ring 且 `nRanks > 8`（L65-67）；(b) all-reduce 2M 流量 → 算法带宽 = bus 之半（8.3 §4）；(c) directRecvReduceCopyDirectSend（all_reduce.h L64），产出全和块；(d) LL/LL128 树走 runTreeSplit 重叠（L762-787），与 §9 一致。

## 参考

- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e0c4d0cfb99f1c61dc13da7996c28，master 2026-11-24）fresh tarball 全量核实——init.cc ncclCommInitRank L3071 / bootstrapInit L2314 / initTransportsRank L1219 / ncclTopoGetSystem L1447 / 图搜索 L1488-1525 / ncclTransportP2pSetup L1957 / devCommSetup L2076·L714·L847；graph/topo.h SM90_NVLINK_BW 20.6 L24 / topo.cc L944·L1994；graph.h PATH 梯子 L120-148 / ncclTopoGraph L173；paths.cc PATH_PHB L104 / PATH_PXN L243 / PATH_NVL L341；search.cc ncclTopoSearchRec L331 / ncclTopoCompute L1245；rings.cc ncclBuildRings L29；trees.cc ncclGetBtree L32 / ncclGetDtree L90（mirror/shift L69-88）；connect.cc ncclTopoPostset L395；transport.cc ncclTransports L15-20；p2p.cc p2pCanConnect L130 / p2pMap L358 / cudaIpcOpenMemHandle L331 / useRead L410；shm.cc shmCanConnect L61；net.cc sendSetup L296 / sendProxyProgress L1327 / recvProxyProgress L1496 / GDRCOPY L338-340；coll_net.cc sharedBuffersInit L416；proxy.cc ncclProxyProgress L956 / progressOps L801 / ncclProxyPost L481；ll__types.h L14-26（sizeof 16）/ ll128__types.h L13-15（DATAELEMS 15）/ prims_ll.h readLL L102-115 / prims_ll128.h FLAGTHREAD L10 / prims_simple.h L20·L104·L179；device.h MAXCHANNELS 64 L80 / NCCL_STEPS 8 L26 / ncclProtoGrainSize L304-310；plugin/nccl_tuner.h 算法枚举 L28-35 / 协议 L39-41；tuning_general.cc ncclTuningGetTime L58（FP8 ×1024 L65-67）；tuning.cc ncclTuningSelectBestTuning L153；cost_model.cc isLL128Enabled L125 / baseLatencies L152（Ring）/ hwLatencies L164·L174·L184 / NCCL_ALGO·PROTO L365-379；ring.cc LL×0.5 L40 / LL128×0.92 L43 / generalBandwidths L47 / ringSimplePlateau L109-139；tree.cc busBw×0.5 L44 / LL128 7/9·120/128 L38-40 / log2i 延迟项 L57-58 / treeCorrectionFactor L79-81；enqueue.cc LL trafficBytes×4 L554；task_posttuning.cc L547-548；nvls.cc cuMulticastCreate L270 / ncclNvlsSetup L494；all_reduce.h runRing L14 / runTreeUpDown L86 / runTreeSplit L146 / NVLS L390 / dispatch L229-244·L762-787；common.cu ncclDevKernel_Generic L23 / common.h ncclKernelMain L420 / RunWorkBatch L303；generate.py L16-17；init.cc DEFAULT_BUFFSIZE L1008-1011（§2-§9）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2.32.3（滞后 repo 2.32.7 注）——NCCL_ALGO（Ring/Tree/NVLS/NVLSTree/PAT）/ NCCL_PROTO（LL/LL128/Simple，「enabling LL128 on platforms that don't support it can lead to data corruption」逐字）/ NCCL_NVLS_ENABLE（「third-generation NVSwitch systems (NVLink4) with Hopper and later」逐字）（§6/§7/§8）
- [NVIDIA, NCCL 2.4 blog](../references.md)：Massively Scale Deep Learning Training with NCCL 2.4（key: nccl-tree-blog），[C] 开放技术博客、非闭源，2026-11-25——double binary tree 与 ring/tree 实测交叉（8.3 已引，本章 §3 双树构建出处）
- 7.1 / 7.2 / 7.3 / 8.1 / 8.2 / 8.3（纯文字指引，不挂链接）：NVLS 流量账每 rank 发送 M（7.1 §6）、有效带宽 370.8 GB/s 与 raw/模型两层账（7.2 §3）、peer 映射与 ACS/IOMMU（7.3）、API 入队语义与 regime $M_c$（8.1 §5/§7）、ring 下界与多 ring 式 (6)（8.2 §3/§5）、双树带宽最优与 ring/tree 交叉 M*（8.3 §3/§4）——recap 不重推（§1/§3/§6/§8）
- 9.2 / 10.2 / 10.3 / 10.4 / 11.2（纯文字指引，不挂链接）：GDR 硬件机制（9.2）、fabric 工程与 IB SHARP（10.2）、a2a 拥塞与 DeepEP（10.3）、sendrecv 与 Ring Attention（10.4）、实测 drill-down（11.2）——两层纪律下游（§10）
