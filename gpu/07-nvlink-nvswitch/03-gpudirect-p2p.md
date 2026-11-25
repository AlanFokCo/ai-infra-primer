# 7.3 GPUDirect P2P（GPUDirect Peer-to-Peer）

> 同一条 cudaMemcpyPeer，为什么在一台机器上 1 GB 拷 2.22 ms、另一台要 31.25 ms——差的不是 API，是路径走没走 host 内存？cudaDeviceEnablePeerAccess 启用之后，驱动到底映射了什么，指针为什么两边通用？PCIe P2P 被 IOMMU 与 ACS 各挡在哪一层，后果是变慢还是别的？GDS 把 NVMe 直连 HBM，省下的两跳值多少带宽？本章交付四块：GPUDirect 家族定位与 P2P 形式定义、peer access 的 API 语义与地址映射（含 8-peer 上限与一致性 scope）、P2P vs host bounce buffer 的带宽账（命题 1：串行半谐波均值 / 流水 min）、PCIe P2P 限制清单与 GDS cuFile 接口及 min 带宽账 | 前置：7.1（NVLink/NVSwitch 几何与方向口径）、7.2（P2P 带宽账、PCIe 拓扑与 NUMA、nvbandwidth 测量口径）、3.2（fence 与 memory model scope）；下游 8.1/8.4（NCCL p2p transport 建立在 peer 映射上）、9.2（GPUDirect RDMA 与 BAR1）、10.2（compute fabric transport 与 MNNVL） | 锚点：CUDA Programming Guide（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威，13.5 §3.4.2 P2P 节）、GPUDirect docs（[NVIDIA, GPUDirect docs](../references.md)，[C] 官方规格披露·一手权威，家族页 + GDR docs）、GDS docs（[NVIDIA, GDS docs](../references.md)，[C] 官方规格披露·一手权威，cuFile API v1.20）、cuda-samples（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现，simpleP2P）、NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C]，PCIe Gen5 数值）——官方文档 2026-11-25 一手核实

## 1. 三个问题与本章口径

两台机器都跑 cudaMemcpyPeer 拷 1 GB，一台报 2.2 ms、一台报 31 ms，差 14 倍——API 相同，差在哪？cudaDeviceEnablePeerAccess 之后，一个在 GPU 1 上分配的指针为什么能被 GPU 0 的 kernel 直接解引用？Linux 服务器上某个 BIOS 选项开着，PCIe P2P 就不是变慢而是根本不能用——是哪个选项、后果是什么？本章回答三个问题：P2P 的「直接」到底直接在哪（路径与地址映射）？不经 P2P 的拷贝多付哪两跳、各值多少带宽？PCIe 路径上 IOMMU/ACS/root complex 三道门各挡什么？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（NVLink4 + 4 颗第三代 NVSwitch，几何 recap 7.1 §3 / 7.2 §2）；**PCIe Gen5 x16 = 64 GB/s 单向**作同机对比路径（Hopper wp p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」，recap 7.2 §5）。带宽账取 **raw 峰值口径**（有效带宽折扣见 7.2 式 (2)，两边同扣协议效率、比值结论不变）；贯穿实例 $M = 1$ GB = $10^{9}$ B。方向口径沿 7.1 硬规则：头条 900 GB/s 为双向总和，单向拷贝峰值 450 GB/s。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，取一 GPU 一 rank 口径）； $M$ = 被传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s），下标 uni / serial / pipe 标单向 / 串行端到端 / 流水稳态； $T_{\mathrm{serial}}$ = 串行两跳端到端总时间（§4 证明）。 $N$ 专归参数量、 $B$ 专归 batch，本章均不占用。

**边界声明**：GPUDirect RDMA（NIC 直访 HBM、nvidia-peermem、BAR1 注册机制）归 9.2；NCCL p2p transport 模块走读归 8.4；ring 带宽最优性归 8.2；NVLink 硬件几何与有效带宽模型归 7.1/7.2，本章只引其结论。**测量口径**沿 7.2：本机无 NVIDIA GPU，测量项交付方法 + 官方披露口径，非一手来源数值一律标**教学假设口径**。

## 2. GPUDirect 家族：P2P 是哪个成员

GPUDirect 这个名字下面挂着四项技术，各搬什么、在谁和谁之间搬？家族页逐字：「GPUDirect is a family of technologies, part of Magnum IO」，总目标是「network adapters and storage drives can directly read and write to/from GPU memory, eliminating unnecessary memory copies, decreasing CPU overheads and reducing latency」（[NVIDIA, GPUDirect docs](../references.md)，[C]）。四个成员按「直连路径的两端」区分：

| 成员 | 直连路径两端 | 消掉的中间环节 | 归属 |
| --- | --- | --- | --- |
| **P2P** | GPU ↔ GPU（PCIe / NVLink） | host bounce buffer | 本章 §3-5 |
| **RDMA** | NIC ↔ GPU HBM | CPU 与缓冲拷贝 | 9.2 |
| **Storage** | NVMe / NVMe-oF ↔ GPU HBM | CPU 内存 bounce buffer | 本章 §6 |
| **Video** | 帧采集设备 ↔ GPU HBM | 帧传输中间环节（仅 Windows） | 本书不展开 |

Video 一行按家族页逐字登记：「Offers an optimized pipeline for frame-based devices such as frame grabbers, video switchers, HD-SDI capture, and CameraLink devices ... on Windows only」——帧采集/切换设备到 GPU 显存的优化管线，平台限定 Windows，与数据中心训练/serving 路径无关，本书只列名。RDMA 的机制（NIC 经 PCIe 直访 HBM、免 CPU 拷贝）与 P2P 共享同一套 PCIe 拓扑约束（§5），硬件账归 9.2。

**定义**（GPUDirect P2P）：所谓 GPUDirect P2P（peer-to-peer，点对点），是指 GPU 之间不经 host 内存中转、直接经内存 fabric（PCIe 或 NVLink）完成的拷贝与 load/store 访问——家族页逐字「Enables GPU-to-GPU copies as well as loads and stores directly over the memory fabric (PCIe, NVLink). GPUDirect Peer to Peer is supported natively by the CUDA Driver」。7.2 §3 给过工作定义（不经 host bounce buffer 的直接传输），此处是形式定义。定义里有两个层次，后文分开讲：**bulk 拷贝**（copy engine 搬整块，cudaMemcpyPeer 族，§3）与 **fine-grained 访问**（kernel 逐字 load/store/atomic 直接解引用 peer 指针，§3）。正例：HGX H100 上 GPU0 到 GPU1 的 cudaMemcpyPeer 经 NVSwitch 直通，1 GB 约 2.22 ms（§4 账）。边界反例：**API 名不保证路径**——peer access 未启用时，同一条 cudaMemcpyPeer 调用会经 host 内存中转（PG §3.4.2.1 逐字：启用后「no longer need to be staged through the host」，反读即未启用时 staged），此时走的不是 P2P 路径；是不是 P2P 由启用状态与拓扑决定，不由 API 名字决定。

## 3. Peer access：API 语义与地址映射

让一张 GPU 直访另一张的 HBM，要几步？启用之后指针指向什么？三步：查可行、启用、用。

**第一步，查可行**：`cudaDeviceCanAccessPeer(&can, dev, peer)` 返回 dev 能否直访 peer 的内存。可行性由系统拓扑决定——PG §3.4.2.2 逐字「Depending on the system properties, specifically the PCIe and/or NVLink topology, devices are able to address each other's memory」。NVLink 域内任意对可行（7.2 §2 全互连）；PCIe 路径受 §5 三道门约束。

**第二步，启用**：`cudaDeviceEnablePeerAccess(peer, 0)` 在**当前设备**上启用对 peer 内存的访问。方向是单向的：当前设备获得访问 peer 的能力，反向不自动成立；双向直访要在两个设备上各启用一次。cuda-samples 的 simpleP2P 例正是这么写的（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现，`cpp/0_Introduction/simpleP2P/simpleP2P.cu`，版本漂移注：旧 p2pBandwidthLatencyTest 已从 master 移除，P2P 教学例现行锚 simpleP2P）：

```cpp
// 教学用片段，摘自 simpleP2P.cu（行号经 master 树 contents API 核实）
cudaDeviceCanAccessPeer(&can_access_peer, i, j);   // L99  查可行
cudaSetDevice(gpuid[0]);
cudaDeviceEnablePeerAccess(gpuid[1], 0);           // L130 GPU0 获访问 GPU1
cudaSetDevice(gpuid[1]);
cudaDeviceEnablePeerAccess(gpuid[0], 0);           // L132 GPU1 获访问 GPU0（双向各一次）
// L158-168：UVA 下 ping-pong 拷贝，cudaMemcpyDefault 免指定设备号
cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault);   // 运行时从指针自判源/目的设备
// L173：cudaMemcpyPeer / cudaMemcpy 带宽打印
```

**第三步，用**：两种用法。bulk 拷贝走 `cudaMemcpyPeer(dst, dstDev, src, srcDev, bytes)`（及 Async / 3D 变体）；UVA（Unified Virtual Addressing，统一虚拟地址）下 `cudaMemcpyDefault` / `cudaMemcpyDeviceToDevice` 也能跨设备——simpleP2P 注释逐字「With UVA we don't need to specify source and target devices, the runtime figures this out by itself from the pointers」。fine-grained 访问则直接在 kernel 里解引用 peer 指针做 load/store/atomic（PG §3.4.2.2）。NULL stream 上的跨设备拷贝有强排序语义：PG 逐字，隐式 NULL stream 的跨设备拷贝「does not start until all commands previously issued to either device have completed」，且跑完之前两设备后续命令都不能开始——要重叠须用显式 stream（3.3 异步执行口径）。

**地址映射：指针为什么两边通用**。UVA 给所有设备一个统一虚拟地址空间，PG 逐字「A unified virtual address space is used for both devices ... so the same pointer can be used to address memory from both devices」。机制层图景：启用 peer access 后，peer 设备上的分配被映射进当前设备的地址空间——§3.4.2.6 逐字「Peer-to-peer memory access reaches a peer allocation through a virtual address that the issuing process has reserved and mapped」。于是同一个指针值在两张卡上都指向同一份物理数据：kernel 解引用时，地址翻译落到 peer HBM，事务经 NVLink/PCIe 送达（口径：机制级描述；页表/TLB 的逐层结构闭源，本书不推定）。

```
启用前：GPU0 kernel 解引用 p1 ──► 无效（p1 不在 GPU0 地址空间）
启用后：
  GPU0 虚拟地址空间 ──映射──► GPU1 HBM 物理页（p1 所指）
  GPU0 kernel load/store p1 ──► NVLink/PCIe 事务 ──► GPU1 HBM
  （同一指针值，两卡通用；NVSwitch 域内任意对可映射，7.2 §2）
```

启用的副作用与上限，两条工程事实（PG §3.4.2.2 Note 逐字口径）。其一，**全局生效**：cudaDeviceEnablePeerAccess「operates globally on all previous and subsequent GPU memory allocations on the peer device」，并给 peer 的分配操作加上随 peer 数乘性增长的开销（要令每个分配对所有已启用 peer 立即可访）；更可扩展的替代是 CUDA Virtual Memory Management API，在分配时按需显式申请 peer-accessible 区域，不牵连其余分配。其二，**8-peer 上限**：PG 逐字「On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections」。代入：非 NVSwitch 系统全互连 P2P 要求每设备 $n-1$ 个 peer 连接， $n-1 \le 8$ 即 $n \le 9$——10 GPU 全互连需要 9 个 peer，超限不可行；NVSwitch 系统不受此限（HGX 8-GPU 有 NVSwitch，7.1 §3），且 8 GPU 本就只需 7 个 peer。

> **坑**：多卡启用 peer access 后频繁小分配变慢。触发条件：cudaDeviceEnablePeerAccess 全局生效后，在 peer 设备上高频 cudaMalloc 小块；现象：每次分配都要对所有已启用 peer 建立映射，开销随 peer 数乘性涨（PG Note 逐字「adding multiplicative overhead that scales with the number of peer devices」）；规避：改用 VMM API 按需显式分配 peer-accessible 区域，或集中大块分配、池化复用。

**一致性 scope**（PG §3.4.2.3）：跨设备并发线程的同步必须用 `thread_scope_system` scope——跨卡同步落在 system 域（recap 3.2 fence scope 层级：block / device / system）；peer 内存上的 atomic 读改写仅在**单个 GPU 访问该对象**时保证原子性，多卡并发 atomic 的原子性要求见 CUDA memory model（3.2 回链）。跨卡写后读要可见，同步原语不可省。

NCCL 的域内 transport 正建立在这套映射上：p2p transport 的 `p2pMap` 做 peer 内存映射（`src/transport/p2p.cc` L358），跨进程经 `cudaIpcOpenMemHandle`（L331，cudaIpcMemLazyEnablePeerAccess 旗标）；read/write 语义选择（`useRead`，L410）对应 7.2 §3 的方向账。模块全量走读归 8.4，本章只登记接口落点。另登记一条新路径：PG §3.4.2.6 的 **compute fabric transport** 为大型 NVLink fabric 提供「逻辑 endpoint id + offset」命名替代映射地址，面向通信库与高级应用（MNNVL 场景，10.2 回链）。

## 4. P2P vs bounce buffer：省两跳的带宽账

不启用 P2P 时拷贝走哪条路？多出的两跳各值多少？PG §3.4.2.1 逐字给了答案的反面：启用后「peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster」——未启用时经 host 中转（staged through the host）。两条路径画出来：

```
bounce 路径（peer access 未启用 / 不可行）：
  GPU A HBM ──D2H──► host DRAM buffer ──H2D──► GPU B HBM
     读 M        PCIe 跳 1：写 M       PCIe 跳 2：写 M
  host DRAM 流量 = 写 M + 读 M = 2M；CPU/copy engine 参与中转

P2P 路径（启用，NVLink 域内）：
  GPU A HBM ─────────NVLink─────────► GPU B HBM
     读 M          单跳 M                写 M
  host DRAM 流量 = 0；省 D2H + H2D 两跳
```

**命题 1**（bounce 两跳带宽账）。设 D2H、H2D 两跳的有效单向带宽各为 $`\mathrm{BW}_1`$、 $`\mathrm{BW}_2`$，消息 $M$ 字节，则端到端有效带宽分两种口径：

$$
\mathrm{BW}_{\mathrm{serial}} = \left( \frac{1}{\mathrm{BW}_1} + \frac{1}{\mathrm{BW}_2} \right)^{-1}, \qquad \mathrm{BW}_{\mathrm{pipe}} = \min\left(\mathrm{BW}_1,\ \mathrm{BW}_2\right) \qquad (1)
$$

**符号**： $`\mathrm{BW}_{\mathrm{serial}}`$ = 串行（两跳不重叠）端到端有效带宽，为两跳带宽的**谐波均值之半**（半谐波均值，并联电阻式——谐波均值为本式的 2 倍）； $`\mathrm{BW}_{\mathrm{pipe}}`$ = 全流水（双缓冲、稳态）端到端有效带宽，取两跳较小者。P2P 直连路径把两跳换成单跳，带宽即该路径单向带宽 $`\mathrm{BW}_{\mathrm{P2P}}`$。

**证明**（路线：时间相加 → 倒数和取倒数 → 流水瓶颈，三步）：其一，串行两跳时间相加， $`T_{\mathrm{serial}} = M/\mathrm{BW}_1 + M/\mathrm{BW}_2`$，端到端带宽按定义 $M/T_{\mathrm{serial}}$ 即得倒数和的倒数，亦即谐波均值之半。其二，流水下每个 chunk 须依次过两跳，稳态吞吐由较慢一跳决定——快跳的产出在中间缓冲堆积，慢跳喂不满则整条线速降到慢跳，即 min。其三，P2P 路径只有一跳， $M$ 字节只占该路径一次，时间 $`M/\mathrm{BW}_{\mathrm{P2P}}`$。∎

全数字代入（两跳均 Gen5 x16 = 64 GB/s， $M = 1$ GB，raw 峰值口径）：串行 $1/(1/64 + 1/64) = 32$ GB/s，耗时 $10^{9}/(32 \times 10^{9}) = 31.25$ ms；流水 $\min(64, 64) = 64$ GB/s，耗时 15.625 ms。NVLink P2P 单向 raw 450 GB/s（7.1 式 (2)），耗时 $10^{9}/(450 \times 10^{9}) \approx 2.22$ ms——**对串行 bounce 快 14.06 倍，对流水 bounce 快 7.03 倍**。PCIe P2P（同 root complex、ACS 关，§5）64 GB/s，耗时 15.625 ms——对串行 bounce 快 2 倍，与流水 bounce 稳态持平。所以 P2P 的增益分两层：**路径增益**（NVLink 换 PCIe，450/64 ≈ 7×）与**跳数增益**（单跳换两跳，串行口径 2×）；PCIe P2P 只拿跳数增益，NVLink P2P 两层都拿。不等跳例（命题 1 代入）：跨 socket H2D 有效 40 GB/s（7.2 §5 教学假设）、D2H 64 GB/s，串行半谐波均值 $1/(1/64 + 1/40) \approx 24.62$ GB/s，流水 $\min = 40$ GB/s。

另一笔是 host 侧流量账：

$$
\mathrm{Traffic}_{\mathrm{host}}^{\mathrm{bounce}} = 2M, \qquad \mathrm{Traffic}_{\mathrm{host}}^{\mathrm{P2P}} = 0 \qquad (2)
$$

**符号**： $`\mathrm{Traffic}_{\mathrm{host}}`$ = 一次拷贝在 host DRAM 上产生的读写流量。bounce 路径 host buffer 先被写 $M$（D2H 落地）再被读 $M$（H2D 取走），合计 $2M$；P2P 路径 host DRAM 零流量。代入 $M = 1$ GB：每拷 1 GB 省 2 GB host DRAM 流量，并免去 CPU/copy engine 的中转参与。多卡并发拷贝时 host 内存带宽是共享瓶颈，这笔节省是结构性的——不是单拷贝快一点，而是把 host 从数据路径上整个摘掉。口径注：串行与流水是同一路径的两种读数，勿混；两跳若共享同一物理链路（如同一 PCIe switch 的上行口）， $`\mathrm{BW}_1`$、 $`\mathrm{BW}_2`$ 不独立，min 账须按共享链路带宽计（边界反例：流水不突破 min 上界，共享链路时两跳带宽不能各按 64 独立取）。

> **坑**：把 cudaMemcpyPeer 的报值当必然 P2P。触发条件：peer access 未启用，或被 IOMMU 挡掉（§5）而 API 仍返回成功；现象：路径实际 staged through host，带宽只有串行 ~32 GB/s 量级，与 NVLink 期望差 14 倍。与 ACS 区分：ACS 开时 P2P 仍可行、但事务绕行 root complex（§5），仍是 device-to-device，降的是 bisection 带宽，不是退化为 host 中转 ~32 GB/s；规避：先 cudaDeviceCanAccessPeer 查可行、cudaDeviceEnablePeerAccess 启用，再用 nvbandwidth P2P 矩阵核对路径（测量方法 recap 7.2 §6）。

## 5. PCIe P2P：IOMMU、ACS、root complex 三道门

NVLink P2P 由硬件原生支持（§2 家族页逐字 supported natively）；PCIe P2P 则要过三道门，每道门的后果不同——有的挡性能，有的挡正确性。逐道给一手出处。

**第一道，拓扑门：同上游 root complex**。所谓 root complex（PCIe 树的主机侧根节点，衔接 CPU/内存控制器与 PCIe 设备，recap 7.2 §5 定义），是 PCIe 树的根。GPU↔GPU PCIe P2P 要求两设备共享同一上游 root complex（或经 PCIe switch 直通）——GDR docs 逐字「the two devices must share the same upstream PCI Express root complex」（[NVIDIA, GPUDirect docs](../references.md)，[C]；GDR 与 P2P 共用同一 PCIe 拓扑约束，互证）。跨 root complex（典型为跨 socket）的流量须绕 inter-socket link，P2P 直通不成立，退化回 host 中转或跨 socket 路径（NUMA min 账 recap 7.2 §5 式 (4)）。

**第二道，IOMMU 门：Linux bare-metal 必须关**。所谓 IOMMU（I/O Memory Management Unit，I/O 内存管理单元），是指对 PCIe 设备 DMA 做地址翻译的硬件；GDR docs 口径：任何非 1:1 的翻译都与 GPUDirect 不兼容，「must be disabled or configured for pass-through」。PG §3.4.2.5 逐字更硬：「CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer」，且「The IOMMU must be disabled when running Linux on a bare metal system **to prevent silent device memory corruption**」——注意后果是**静默数据损坏**，不是变慢、不是报错。两个例外同节逐字：VM 场景「do support IOMMU via virtual machine pass through」（IOMMU 应开、用 VFIO 驱动做 PCIe passthrough）；Windows「the IOMMU limitation above does not exist」。

**第三道，ACS 门：重定向杀 bisection**。所谓 ACS（Access Control Services，PCIe 访问控制服务），是指强制 P2P 流量上行经 root complex 的 PCIe 访问控制能力——PG §3.4.2.5 逐字披露其效果：「The PCI ACS feature redirects all PCI point-to-point traffic through the CPU root complex, which can cause significant performance loss due to the reduction in overall bisection bandwidth」。机制：本可在 PCIe switch 本地交换的 P2P 事务，被 ACS 强制上行到 root complex 再下行折返——直通变绕行，bisection 带宽（二分带宽，拓扑两半之间的总交换容量）从 switch 本地交换降为 root complex 转发口径。ACS 与 IOMMU 常如影随形（PG 逐字「ACS can be enabled on systems that support IOMMU」），但两道门后果不同：IOMMU 挡正确性（bare-metal 不支持），ACS 挡性能（重定向降带宽）。

三道门合成一张判定表（Linux 口径，PG §3.4.2.5 + GDR docs）：

| 配置 | PCIe P2P 可行性 | 实际路径 |
| --- | --- | --- |
| bare-metal + IOMMU 开 | **不支持**（静默损坏风险） | 须关 IOMMU 方可 P2P |
| IOMMU 关 + ACS 开 | 可行但绕行 | P2P 事务 → root complex → 目标，bisection 降 |
| IOMMU 关 + ACS 关 + 同 root complex | 直通 | switch 本地 / root complex 内交换 |
| 跨 root complex（跨 socket） | 无直通 P2P | host 中转或 inter-socket（7.2 §5） |
| VM passthrough（VFIO） | 支持 | IOMMU 开为合法配置 |

另两条 PCIe 侧事实登记。其一，**BAR1 孔径**：PCIe 侧设备（peer GPU / NIC）访问 GPU HBM 经 BAR1 孔径，GDR docs 口径 large BAR「BAR1 size is set to 16GB or larger」；孔径大小与注册机制的硬件账归 9.2，本章只登记存在。其二，**8-peer 上限**（§3 recap）：非 NVSwitch 系统每设备至多 8 个 peer 连接，是 CUDA 驱动限制，与上面三道 PCIe 门独立——即使三道门全过，10 GPU 全互连仍超限。

> **坑**：把 IOMMU 当性能问题。触发条件：Linux bare-metal 开着 IOMMU（部分发行版默认开）跑 PCIe P2P；现象：不是变慢而是 silent device memory corruption（PG 逐字），数据错而无报错；规避：bare-metal 关 IOMMU（或改 VM passthrough 合法配置），改配置后用 nvbandwidth 与数据校验双重核对 P2P 路径。

## 6. GPUDirect Storage：NVMe 到 HBM 的直路

训练数据从 NVMe 读进 GPU，传统路径走几跳？GDS 摘掉哪一跳？传统路径：NVMe → host DRAM（page cache / bounce buffer）→ GPU HBM，两跳、host DRAM 付 $2M$ 流量、CPU 参与拷贝。GDS 家族页逐字：「enables a direct data path between local or remote storage, such as NVMe or NVMe over Fabric (NVMe-oF), and GPU memory. It avoids extra copies through a bounce buffer in the CPU's memory, enabling a direct memory access (DMA) engine near the NIC or storage to move data on a direct path into or out of GPU memory — all without burdening the CPU」（[NVIDIA, GDS docs](../references.md)，[C]）。即存储侧 DMA 引擎直接把数据搬进/搬出 GPU HBM，host bounce buffer 与 CPU 中转整个摘掉——与 §4 P2P 省两跳同构，只是两端换成 NVMe 与 GPU。

cuFile API 是 GDS 的编程接口（cuFile API Reference v1.20，[NVIDIA, GDS docs](../references.md)，[C]）。核心五个函数按调用序列：

| API | 作用 |
| --- | --- |
| cuFileDriverOpen | 打开 GDS 驱动会话（加载配置） |
| cuFileHandleRegister | 注册已打开的文件（必需，带 memoized 检查加速后续操作） |
| cuFileBufRegister | 注册 GPU 缓冲（供 DMA 直访） |
| cuFileRead / cuFileWrite | 同步读/写：文件 ↔ GPU 缓冲 |

调用序列（教学骨架，API 名与次序按 v1.20；异步变体 cuFileReadAsync / cuFileWriteAsync 另见 docs）：

```
cuFileDriverOpen()          ← 1. 打开驱动会话
open(file, O_DIRECT)        ← 2. 打开文件（O_DIRECT 直路）
cuFileHandleRegister(...)   ← 3. 注册文件句柄（必需）
cuFileBufRegister(...)      ← 4. 注册 GPU 缓冲
cuFileRead(...)             ← 5. NVMe → HBM 直读
```

> **坑**：假设 cuFileRead 与默认流上的 kernel 有序。触发条件：把 cuFileRead/Write 当 cuMemcpy 非 async 变体那样与 NULL stream 其他工作自动排序；现象：docs 逐字「the cuFileHandleRegister, cuFileHandleDeregister, cuFileRead, and cuFileWrite APIs do not have the semantic of being ordered with respect to other work in the null stream」——读与计算竞态，数据未就绪即被消费；规避：显式 stream/event 同步，或用带 stream 参数的 async 变体。

带宽账沿用命题 1 的 min 结构。GDS 直路是单跳，瓶颈取路径各段最小：

$$
\mathrm{BW}_{\mathrm{GDS}} = \min\left(\mathrm{BW}_{\mathrm{NVMe}},\ \mathrm{BW}_{\mathrm{PCIe}}\right) \qquad (3)
$$

**符号**： $`\mathrm{BW}_{\mathrm{NVMe}}`$ = 存储侧有效读带宽； $`\mathrm{BW}_{\mathrm{PCIe}}`$ = GPU 所挂 PCIe 链路单向带宽。全数字代入（**教学假设口径**：设 Gen5 x4 NVMe SSD 顺序读 14 GB/s，GPU 侧 Gen5 x16 = 64 GB/s）：直路 $\min(14, 64) = 14$ GB/s；bounce 串行按式 (1) 半谐波均值 $1/(1/14 + 1/64) \approx 11.49$ GB/s。增益 $14/11.49 \approx 1.22\times$——**带宽增益有限**，因为 NVMe 本身就是瓶颈，直路并没有把存储带宽变快。代入 $M = 1$ GB：直路 71.43 ms，bounce 串行 87.05 ms。所以 GDS 的主战场不在单盘单卡提速，而在两处结构性收益：其一，host DRAM $2M$ 流量与 CPU 中转摘掉（式 (2) 同构），多 GPU 并发读时不再争抢 host 内存带宽——聚合存储带宽随盘数扩展，不被 host 内存带宽封顶；其二，CPU 从数据路径卸载（家族页逐字「all without burdening the CPU」），让出 CPU 给数据预处理。口径注：14 GB/s 为教学假设值（Gen5 x4 SSD 量级），非官方披露；实测以本机 nvbandwidth / fio 为准。O_DIRECT 一条兼容注（docs Note）：GDS 1.7.x（CUDA 12.2）起支持非 O_DIRECT 打开（compat 模式），装有 nvidia-fs.ko 时若存在 O_DIRECT 直路仍走直路。

## 7. P2P 是集合通信的地基

8-GPU all-reduce 和 P2P 有什么关系？ring 的每一跳，就是一次 P2P 传输。recap 7.2 §4：ring 每跳 = 「源 GPU → NVSwitch crossbar → 目标 GPU」路径；这一跳的数据搬运正是 §3 的 peer access write/read（NCCL p2p transport，useRead 决定 read/write 语义，p2p.cc L410）。所以**域内集合通信 = 结构化的 P2P 传输序列**：原语语义与每 rank 消息量账归 8.1，ring 构建与协议走读归 8.4。两条 bypass 登记：NVLS 把归约搬进 NVSwitch、不再逐 peer P2P（7.1 §6，每 rank 发送从 $2(n-1)/n \cdot M$ 降到 $M$）；跨节点段走 NIC/RDMA（9.2），P2P 只管域内段。compute fabric transport（§3.4.2.6）为大型 NVLink fabric 提供 endpoint id + offset 命名，是 MNNVL 域集合通信的寻址替代（10.2）。本章交付的 peer 映射、bounce 账与 PCIe 三道门，就是 8.1 原语→硬件映射表里「域内 p2p transport」一行的机制底座。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| GPUDirect 四成员按直连两端区分：P2P（GPU↔GPU）/ RDMA（NIC↔HBM，9.2）/ Storage（NVMe↔HBM）/ Video（帧设备，仅 Windows）；P2P = 不经 host 中转、经 PCIe/NVLink 的拷贝与 load/store，CUDA Driver 原生支持 | 家族页逐字 | §2、练习 1 |
| API 名不保证路径：peer access 未启用时 cudaMemcpyPeer staged through host；启用三步 CanAccessPeer → EnablePeerAccess（单向，双向各一次）→ 拷贝/解引用 | PG §3.4.2.1-2 逐字；simpleP2P L99/L130/L132 | §2、§3 |
| 地址映射：UVA 同一指针两卡通用；peer 分配经 issuing process 预留并映射的虚拟地址到达；非 NVSwitch 每设备 ≤8 peer 连接 → 全互连 n ≤ 9 | PG §3.4.2.2/§3.4.2.6 逐字 | §3、练习 4 |
| bounce 两跳：串行端到端 = 半谐波均值（并联电阻式）、流水 = min；Gen5 64/64 → 串行 32 / 流水 64 GB/s；NVLink P2P 450 → 1 GB 2.22 ms，对串行快 14.06× / 对流水快 7.03×；PCIe P2P 只拿跳数增益 2× | 式 (1)，命题 1 | §4、练习 2/3 |
| host DRAM 流量：bounce 2M vs P2P 0；1 GB 拷贝省 2 GB host 流量 + CPU 中转；多卡并发时 host 带宽是共享瓶颈 | 式 (2) | §4、练习 2 |
| PCIe 三道门：同 root complex（GDR docs 逐字）/ IOMMU bare-metal 不支持（后果 silent corruption，VM VFIO 与 Windows 例外）/ ACS 重定向经 root complex 降 bisection | PG §3.4.2.5 逐字 | §5、练习 4 |
| GDS：NVMe→HBM 直路免 bounce；cuFile DriverOpen/HandleRegister/BufRegister/Read/Write；null stream 无排序语义；BW = min(NVMe, PCIe)，教学假设 14/64 → 14 vs bounce 11.49 GB/s = 1.22×，主战场在多卡聚合与 CPU 卸载 | 式 (3)；cuFile v1.20 | §6、练习 4 |
| P2P 是域内集合通信地基：ring 每跳 = 一次 peer write/read（NCCL p2p transport）；NVLS 与跨机 RDMA 是两条 bypass | 7.2 §4 recap；p2p.cc L358/L410 | §7、8.1/8.4 |

## 练习

1.（计算 · [10]）带宽口径代入（§4，raw 峰值口径）：(a) NVLink4 单向拷贝峰值多少 GB/s；(b) PCIe Gen5 x16 单向理论值多少；(c) 两者比值；(d) 两跳均 64 GB/s 时 bounce 串行端到端有效带宽多少。校验：(a) 450（头条 900 ÷ 2，7.1 方向口径）；(b) 64（Hopper wp p.49 逐字）；(c) 450/64 ≈ 7.03，与 7.1 式 (5) 的 7× 一致；(d) 32 = 64/2（命题 1 半谐波均值，等跳特例）。
2.（计算 · [20]）bounce 账（命题 1， $M = 1$ GB）：(a) 串行 bounce（两跳 64 GB/s）耗时多少 ms；(b) NVLink P2P（450 GB/s）耗时多少 ms、对串行 bounce 快几倍；(c) PCIe P2P（64 GB/s）耗时多少 ms、对串行 bounce 快几倍；(d) 一次 1 GB 拷贝 P2P 比 bounce 省多少 GB host DRAM 流量？先预测 (b) 对 (c) 的增益来自路径还是跳数，再算。校验：(a) 31.25 ms；(b) ≈2.22 ms、14.06×；(c) 15.625 ms、2×；(d) 2 GB（式 (2)）；预测：路径增益（450/64 ≈ 7×），跳数增益已计入串行分母。
3.（推导 · [30]）证明命题 1（式 (1)）：(a) 写出串行两跳时间相加 → 端到端带宽 = 倒数和取倒数（半谐波均值）的推导；(b) 论证流水稳态吞吐 = min（慢跳决定线速、快跳产出堆积）；(c) 给边界反例：两跳共享同一物理上行链路时为什么 $`\mathrm{BW}_1`$、 $`\mathrm{BW}_2`$ 不能各按 64 独立取；(d) 代入不等跳 64/40 GB/s 算串行与流水值。校验：(a) $`(1/\mathrm{BW}_1 + 1/\mathrm{BW}_2)^{-1}`$；(d) 串行 ≈24.62、流水 40 GB/s，与 §4 不等跳例一致（容差 1%）。
4.（计算 · [30]）PCIe 三道门与 GDS 账（§5/§6）：(a) 判定三配置下 PCIe P2P 可行性与实际路径：Linux bare-metal IOMMU 开 / IOMMU 关 + ACS 开 / VM VFIO passthrough；(b) 非 NVSwitch 系统 10 GPU 全互连 P2P 是否可行，至多几 GPU；(c) GDS 直路与 bounce 串行有效带宽各多少（NVMe 14 GB/s 教学假设、PCIe 64 GB/s）、1 GB 耗时各多少 ms、增益几倍？校验：(a) 不支持（silent corruption 风险）/ 可行但绕行 root complex / 支持；(b) 不可行（需 9 peer 超 8 上限），至多 9 GPU；(c) 14 vs ≈11.49 GB/s、71.43 vs 87.05 ms、≈1.22×，与 §6 逐数一致。

## 参考

- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide 13.5（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——§3.4.2.1 P2P Memory Transfers（cudaMemcpyPeer/PeerAsync/3DPeer 族、UVA 下 cudaMemcpyDefault/DeviceToDevice 跨设备、NULL stream 跨设备拷贝排序逐字、「no longer need to be staged through the host」逐字）；§3.4.2.2 P2P Memory Access（cudaDeviceCanAccessPeer + cudaDeviceEnablePeerAccess、拓扑决定可行逐字、UVA 同指针逐字、8 peer connections 上限逐字、enable 全局生效 + 乘性分配开销 + VMM 替代 Note）；§3.4.2.3 一致性（thread_scope_system、peer atomic 单 GPU 限制）；§3.4.2.5 IOMMU/ACS（bare-metal 不支持 + silent device memory corruption + VM VFIO + Windows 例外 + ACS 重定向降 bisection 逐字）；§3.4.2.6 compute fabric transport（VA reserved and mapped 逐字 + endpoint id/offset 替代）（§2-5、§7）
- [NVIDIA, GPUDirect docs](../references.md)：GPUDirect 家族页 + GPUDirect RDMA docs（key: gpudirect-rdma-docs），[C] 官方规格披露·一手权威，2026-11-25——家族页「family of technologies, part of Magnum IO」+ 四成员 + P2P 定义逐字（GPU-to-GPU copies/loads/stores over PCIe/NVLink、CUDA Driver 原生）+ GDS 直路逐字 + Video 仅 Windows 逐字；GDR docs 同 upstream root complex 逐字 + IOMMU 非 1:1 不兼容 + BAR1 16GB+ large BAR（§2/§5/§6）
- [NVIDIA, GDS docs](../references.md)：GPUDirect Storage docs + cuFile API Reference v1.20（key: gds-docs），[C] 官方规格披露·一手权威，2026-11-25——cuFileDriverOpen / cuFileHandleRegister（§3.2.1，必需 + memoized 检查）/ cuFileBufRegister / cuFileRead / cuFileWrite 逐字；null stream 无排序语义逐字；O_DIRECT compat 模式 Note（GDS 1.7.x / CUDA 12.2）（§6）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/0_Introduction/simpleP2P/simpleP2P.cu：cudaDeviceCanAccessPeer L99 / cudaDeviceEnablePeerAccess L130/L132（双向各一次）/ UVA ping-pong cudaMemcpyDefault L158-168 / 带宽打印 L173；版本漂移注：p2pBandwidthLatencyTest 已从 master 移除，带宽矩阵实测改锚 nvbandwidth（§3）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e）fresh tarball 核实——src/transport/p2p.cc：p2pMap L358（peer 内存映射）/ cudaIpcOpenMemHandle L331（跨进程 IPC）/ useRead L410（read/write 语义选择）；模块全量走读归 8.4（§3/§7）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——PCIe Gen5 节 p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」（§4/§5 带宽数值，recap 7.2 §5）
- [NVIDIA, nvbandwidth](../references.md)：NVIDIA/nvbandwidth repo（key: nvbandwidth），教学参照 + 官方实现，README 2026-11-25 一手核实——P2P 带宽/延迟矩阵测量方法（§4/§5 坑注的路径核对手段，方法 recap 7.2 §6）
