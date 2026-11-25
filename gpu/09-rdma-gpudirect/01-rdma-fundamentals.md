# 9.1 RDMA 原理（RDMA Fundamentals）

> 同一份 1 GB 数据，TCP 发送为什么每消息要 CPU 参与几十微秒、RDMA 却在提交请求后几乎不占 CPU？「远程直接内存访问」这六个字到底绕过了什么——NIC 凭什么能读远端内存而远端 CPU 毫不知情？verbs 的 QP/CQ/MR/WR 四个对象怎么分工，单边 READ/WRITE 与双边 SEND/RECV 为什么是两种语义？RC/UC/UD 三种 QP 与 IB/RoCE 两种链路层各差在哪、怎么选？本章交付四块：RDMA 三语义（zero-copy / kernel bypass / CPU offload）与 TCP/IP 延迟账（命题 1）、verbs 四对象与 doorbell 机制、单边/双边操作语义与 QP 类型/状态机（连接开销账）、IB vs RoCE 链路层差异与 ConnectX NIC 硬件 | 前置：7.2（PCIe Gen5 x16 = 64 GB/s 单向、NUMA、节点内 vs 机间数量级差）、7.3（GPUDirect 家族定位、bounce buffer 两跳账）、8.4（NCCL NET transport 与 proxy 线程）；下游 9.2（GPUDirect RDMA 与 NIC→HBM 带宽账）、9.3（拥塞控制与传输可靠性）、10.1（机间 fabric 与拓扑） | 锚点：RDMA Aware Programming Manual v1.7（[NVIDIA, RDMA programming manual](../references.md)，[C] 官方规格披露·一手权威，RDMA 定义/RC-UC-UD/操作×传输表/QP bringup 逐字）、rdma-core（[linux-rdma, rdma-core](../references.md)，教学参照 + 官方实现，v67.0 master 4b7b208c，verbs.h file:line）、IBTA 规范（[IBTA, ibta-spec](../references.md)，[C] 官方规格披露，Vol 1/2 Release 2.1）、NVIDIA networking docs（[NVIDIA, networking docs](../references.md)，[C]，OFED RoCE 页）、ConnectX-7/8（[NVIDIA, connectx7](../references.md) / [connectx8](../references.md)，[C]，datasheet + HW manual SKU 表）、Hopper whitepaper（[NVIDIA, hopper-whitepaper](../references.md)，[C]，PCIe Gen5 p.49）——官方规格与代码锚 2026-11-25 一手核实

## 1. 三个问题与本章口径

TCP socket 发一条消息，内核要拷两次、上下文切换两次、走一遍协议栈；RDMA 提交一个请求后，数据由 NIC 直接搬走，远端 CPU 甚至不知道这次访问发生过。差在哪三个环节？NIC 没有远端的页表、没有远端 OS 的配合，凭什么能 DMA 到远端内存？本章回答三个问题：RDMA 的三语义各省掉哪一笔开销、延迟账怎么列（§2）？verbs 四对象与单边/双边操作怎么把「直接」落到 API（§3-4）？QP 类型、状态机与 IB/RoCE 链路层怎么选型（§5-6）？

**本章口径**（章脊锁定）：主口径 **ConnectX-7 NDR 400G 单口**（线速 400 Gb/s = 50 GB/s 单向， $10^{9}$ 口径）+ **PCIe Gen5 x16 = 64 GB/s 单向**主机接口（Hopper wp p.49 逐字，recap 7.2 §5）；贯穿实例 $M = 1$ GB = $10^{9}$ B。延迟账的分量值（单次拷贝/上下文切换/协议栈/doorbell）为**教学假设口径**，逐处标注，结构结论（省掉哪些项）才是交付物，非实测拟合。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，§5 连接开销账用）； $M$ = 被传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s）。局部符号： $t_{\mathrm{copy}}$ = 单次内核拷贝时间、 $t_{\mathrm{ctx}}$ = 单次上下文切换、 $t_{\mathrm{stack}}$ = 协议栈处理、 $t_{\mathrm{db}}$ = 单次 doorbell 写（§2 命题 1）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用。

**边界声明**：本章只讲 **verbs 机制层**（代码 scope = 机制片段级：rdma-core verbs.h 接口片段 + 教学骨架，非模块全量走读）。机间 fabric 工程（拓扑族 / SM / 路由 / 收敛比 / rail）归 10.1；GDR（NIC 直访 GPU HBM、peer memory、BAR1、带宽账）归 9.2；拥塞控制（DCQCN / PFC / ECN）与传输可靠性归 9.3；NCCL NET transport 模块走读归 8.4。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为结构推导 + 教学假设分量，非新跑 benchmark。

## 2. RDMA 三语义：绕过的是什么

先回答一个更基本的问题：传统网络发送为什么慢、为什么吃 CPU？因为数据要经内核协议栈，每消息付拷贝、上下文切换、协议处理三笔软件开销。RDMA 的全部设计就是对这三笔开销动手。

**定义**（RDMA）：所谓 RDMA（Remote Direct Memory Access，远程直接内存访问），是指 NIC 直接读写两端主机的应用缓冲、数据路径不经内核拷贝、传输过程不涉及远端 OS 与 CPU 的通信方式。verbs 编程手册逐字：「Remote Direct Memory Access (RDMA) provides direct memory access from the memory of one host ... to the memory of another host **without involving the remote Operating System and CPU**」；对照面同句逐字：「TCP/IP communications typically require copy operations, which add latency and consume significant CPU and memory resources」（[NVIDIA, RDMA programming manual](../references.md)，[C]，§1）。三个语义各绕过一笔：

| 语义 | 绕过什么 | 硬件/机制落点 |
| --- | --- | --- |
| **zero-copy** | 内核缓冲拷贝（app↔kernel 两次 memcpy） | MR 预注册 pin 页，NIC DMA 直访应用缓冲（§3） |
| **kernel bypass** | 数据路径陷入内核 + 上下文切换 | post_send 用户态写 NIC doorbell，不进内核（§3） |
| **CPU offload** | 本端协议栈处理 + 远端 CPU/OS 参与 | 传输协议（ACK/重传/PSN）由 NIC 硬件引擎跑（§7） |

两条路径画出来（发送侧，接收侧对称）：

```
TCP/IP 发送路径（每消息）：
  app buffer ──copy1──► kernel socket buffer ──协议栈──► NIC DMA ──► wire
              CPU memcpy        TCP/IP 处理(CPU)      陷入内核 = 上下文切换
  接收：wire ──► NIC DMA ──► kernel buffer ──copy2──► app buffer（又一次 ctx + copy）

RDMA 发送路径（每消息）：
  app buffer(MR 已注册) ──NIC DMA 直读──► wire
       ▲ post_send = 用户态 doorbell 写（不进内核、无拷贝）
  接收：wire ──► NIC DMA 直写 ──► app buffer(MR)（远端 CPU/OS 不参与）
```

**命题 1**（数据路径软件开销账）：设单次内核拷贝时间 $t_{\mathrm{copy}}$、单次上下文切换 $t_{\mathrm{ctx}}$、协议栈处理 $t_{\mathrm{stack}}$、单次 doorbell 写 $t_{\mathrm{db}}$，则每消息 TCP/IP 在 wire 时间之外附加的软件开销为 $\Delta_{\mathrm{TCP}} = 2 t_{\mathrm{copy}} + 2 t_{\mathrm{ctx}} + t_{\mathrm{stack}}$，RDMA 附加的软件开销为 $\Delta_{\mathrm{RDMA}} = t_{\mathrm{db}}$；kernel bypass + zero-copy 省去的即 $\Delta_{\mathrm{TCP}} - \Delta_{\mathrm{RDMA}}$。

**证明**（路线：逐跳枚举发送/接收路径，标出 RDMA 删去的跳，三步）：其一，TCP 发送侧 app→kernel 一次拷贝 + 一次陷入内核（上下文切换），经协议栈处理交 NIC；接收侧 kernel→app 再一次拷贝 + 一次上下文切换——合计 2 拷贝 + 2 切换 + 1 协议栈。其二，RDMA 数据路径只有 post_send 的一次用户态 doorbell 写（ $t_{\mathrm{db}}$），NIC 凭 MR 注册信息 DMA 直访应用缓冲，无内核拷贝、无陷入内核、无本端协议栈（协议在 NIC 硬件），远端 CPU/OS 不参与。其三，两式相减即省项。证毕。该命题是「kernel bypass 省上下文切换 + 拷贝」的定量形式，结构结论不依赖分量取值。

全数字代入（**教学假设口径**：单核 memcpy 有效 10 GB/s、 $t_{\mathrm{ctx}} = 1$ μs、 $t_{\mathrm{stack}} = 1$ μs、 $t_{\mathrm{db}} = 0.2$ μs）： $M = 4$ KB = 4096 B 时 $t_{\mathrm{copy}} = 4096 / 10^{10} = 0.41$ μs， $\Delta_{\mathrm{TCP}} = 2 \times 0.41 + 2 \times 1 + 1 = 3.82$ μs， $\Delta_{\mathrm{RDMA}} = 0.2$ μs——TCP 软件开销是 RDMA 的 19.1 倍，省 3.62 μs。换 $M = 1$ GB： $t_{\mathrm{copy}} = 10^{9}/10^{10} = 100$ ms，拷贝项 $2 \times 100 = 200$ ms 远超固定项 3 μs——**大消息拷贝项主导，小消息固定项（切换 + 协议栈）主导**。这解释了为什么 RDMA 对大消息省带宽（免拷贝）、对小消息省延迟（免切换/协议栈），两个 regime 都受益。分量值为教学假设，非实测；省项的结构（2 拷贝 + 2 切换 + 协议栈）由路径枚举决定，是可靠结论。

> **坑**：把 CPU offload 读成「CPU 完全不参与」。触发条件：以为 RDMA 连接建立、内存注册、完成轮询都不花 CPU；现象：忽略控制路径开销，大规模连接时 CPU 仍被打满。事实：offload 的是**数据路径**（拷贝/协议/远端 CPU）；**控制路径**（QP 状态迁移 §5、MR 注册 §3、ibv_poll_cq 轮询）仍由 CPU 跑，只是都在用户态、不陷入内核。规避：数据面与控制面分开算账，连接开销另列（§5）。

## 3. verbs 四对象：QP / CQ / MR / WR

为什么 RDMA 不像 socket 那样一个 `send(fd, buf)` 就完事，要先建四个对象？因为 NIC 必须在数据路径启动**之前**就知道「哪块内存可 DMA、发到哪个远端端点、怎么判完成」——这些信息预注册进对象，数据路径只剩提交请求与轮询完成两个动作。四个对象各管一摊（[linux-rdma, rdma-core](../references.md)，教学参照 + 官方实现，v67.0，`libibverbs/verbs.h` file:line）：

| 对象 | 全称 | 职责 | 关键 API（verbs.h 行） |
| --- | --- | --- | --- |
| **MR** | Memory Region | 注册可 DMA 内存：pin 页 + 建地址翻译 + 发 lkey/rkey | `ibv_reg_mr` L2692 |
| **QP** | Queue Pair | 传输端点 = SQ（发送队列）+ RQ（接收队列），QPN 标识 | `ibv_create_qp` L3246 / `ibv_modify_qp` L3467 |
| **WR** | Work Request | 单次传输请求描述符（opcode + 本地 addr + len + 远端 addr/rkey） | `ibv_post_send` L3662 / `ibv_post_recv` L3671 |
| **CQ** | Completion Queue | NIC 写完成事件的队列，轮询取 CQE | `ibv_create_cq` L2962 / `ibv_poll_cq` L3040 |

**MR 与两个 key**：`ibv_reg_mr` 把一段应用内存注册给 NIC，返回 `lkey`（本地 DMA 凭证，post WR 时引用）与 `rkey`（远端访问凭证，单边操作时交给对端）。注册的实质是 pin 住物理页（防换出）+ 建立 NIC 可用的地址翻译 + 设访问权限。正例：注册 1 GB 缓冲后 NIC 可凭 lkey DMA 直读；边界反例：未注册的内存 NIC 不能 DMA——页可能被换出、地址翻译未建立，强行访问触发保护错误。rkey 是单边 READ/WRITE 的远端授权（§4），须经带外通道（TCP/CM）交换给对端。

**doorbell：kernel bypass 的硬件落点**。`ibv_post_send` 名为「提交」，实质是用户态把 WQE（Work Queue Element，WR 的硬件形态）写进队列内存、再写一次 NIC 的 **UAR**（User Access Region，mmap 到用户态的 NIC 寄存器页）——这次寄存器写就是 doorbell，通知 NIC 来取 WQE。全程不陷入内核：队列内存与 UAR 都在初始化时 mmap 好，数据路径只有用户态内存写。这就是 §2 命题 1 里 $t_{\mathrm{db}}$ 的来源，也是 zero-copy 与 kernel bypass 能同时成立的硬件前提。`ibv_post_send`/`ibv_post_recv` 在 verbs.h 是 static inline 直调 provider（L3662/L3671），无系统调用。

四对象协作的教学骨架（教学用，非优化实现；API 名与次序按 rdma-core verbs.h + manual §8 RDMA_RC 例，省略错误处理与建连细节）：

```c
// 1. 注册内存（pin + lkey/rkey）
struct ibv_mr *mr = ibv_reg_mr(pd, buf, len, IBV_ACCESS_LOCAL_WRITE | ...);
// 2. 建 CQ 与 QP（QP 状态机迁移见 §5）
struct ibv_cq *cq = ibv_create_cq(ctx, cqe, NULL, NULL, 0);
struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
// 3. 提交 WR（用户态 doorbell，不进内核）
struct ibv_send_wr wr = { .opcode = IBV_WR_RDMA_WRITE, .sg_list = &sge,
                          .wr.rdma = { .remote_addr = raddr, .rkey = rkey } };
ibv_post_send(qp, &wr, &bad_wr);          // 写 WQE + doorbell
// 4. 轮询完成（用户态，无中断/无系统调用）
struct ibv_wc wc;
while (ibv_poll_cq(cq, 1, &wc) == 0) { /* spin */ }
```

> **提点**：四对象的设计是「控制面预注册、数据面零系统调用」的分离——MR/QP/CQ 在初始化期建好（慢路径，可花 ms），WR 提交与 CQ 轮询在数据期跑（快路径，只有用户态内存写）。这与 8.4 NCCL proxy 线程「建连慢、推进快」的分层同构。

## 4. 操作语义：单边 READ/WRITE vs 双边 SEND/RECV

RDMA READ 为什么能在远端没 post 任何接收请求的情况下读到远端内存，而 SEND 必须远端先 post RECV？因为两类操作的「直接」程度不同：单边把远端地址与授权写进请求本身，NIC 硬件直达；双边仍需远端预先备好接收缓冲来匹配。

**定义**（单边/双边）：所谓**单边操作**（one-sided，RDMA READ/WRITE），是指 initiator 在 WR 里直接指定远端地址 + rkey、远端 NIC 硬件 DMA 直达、远端 CPU/OS 不参与也无需 post WR 的操作；所谓**双边操作**（two-sided，SEND/RECV），是指接收方必须预 post RECV、SEND 按序匹配接收缓冲的操作。操作 × 传输模式支持表（manual §3.2 逐字，[NVIDIA, RDMA programming manual](../references.md)，[C]）：

| 操作 | RC | UC | UD | 最大消息 | 语义 |
| --- | --- | --- | --- | --- | --- |
| SEND / RECV | 支持 | 支持 | 支持 | UD = MTU；UC/RC = 1 GB | 双边 |
| RDMA WRITE（含 imm） | 支持 | 支持 | **无** | UC/RC = 1 GB | 单边 |
| RDMA READ | 支持 | **无** | **无** | RC = 1 GB | 单边 |
| Atomic（CAS / Fetch&Add） | 支持 | **无** | **无** | RC = 1 GB | 单边 |

（RD 模式 manual 注「not supported by this API」，本书不展开。）读表三条判断句：**RDMA READ 与 Atomic 仅 RC**——读回数据与原子读改写要求可靠有序，UC/UD 的丢包无序会破坏语义；**UD 无 RDMA WRITE/READ**——UD 单包无连接，没有可指定的远端 QP 上下文；**UD 最大消息 = MTU**——单包语义，超过 MTU 须上层分片。

单边操作的远端授权：READ/WRITE 的 WR 携带 `remote_addr` + `rkey`（§3），rkey 是远端 MR 的访问凭证，经带外交换获得。远端 NIC 校验 rkey 与地址范围后直接 DMA，远端 CPU 全程不知。正例：RC RDMA WRITE 到对端已注册缓冲，对端无 post、无 CPU 参与即完成；边界反例：rkey 错误或地址越界 → NIC 保护错误（远端仍无 CPU 参与，错误经 CQ 报给 initiator）。这套「initiator 指定远端地址」语义正是 inference 8.3 push/pull（`NIXL_WRITE`/`NIXL_READ`，initiator 视角）的底层落点——传输语义归 inference 8.3，本章只交付其 verbs 机制底座，不重推。

双边操作的匹配与 RNR：SEND 必须匹配接收方预 post 的 RECV（按序）。若接收方未备好 RECV，RC 下触发 **RNR**（Receiver Not Ready）NAK，发送方按 `RNR_RETRY` 重试（§5 RTS 属性）；UD 下直接丢包（不可靠）。所以双边 SEND/RECV 的接收侧须持续补 post RECV，否则 RNR 重试拖慢甚至耗尽重试次数报错。NCCL NET transport 用双边 SEND/RECV（proxy 线程替 GPU post，8.4 §5 recap）；DeepEP low-latency 用 IBGDA put（单边 WRITE 语义，GPU 直构 WQE，9.2/10.3）——两类操作各有其适用场景，本章只交付语义区分。

## 5. QP 类型与状态机：连接开销账

RC/UC/UD 三种 QP 怎么选？判据是「可靠性 × 连接性」两个维度。三类型语义（manual §3.2 逐字 + rdma-core `verbs.h` L934-937 `enum ibv_qp_type { IBV_QPT_RC=2, IBV_QPT_UC, IBV_QPT_UD }`）：

| 类型 | 连接性 | 可靠性 | 有序 | 类比 | 典型用途 |
| --- | --- | --- | --- | --- | --- |
| **RC**（Reliable Connection） | 一对一 | 可靠（ACK + 重传） | 有序 | TCP | 跨机梯度/KV 可靠传输（NCCL/DeepEP） |
| **UC**（Unreliable Connection） | 一对一 | 不可靠（不重传） | — | — | 容忍丢包的大块写 |
| **UD**（Unreliable Datagram） | 任意对任意 | 不可靠 | 无序 | UDP | 多对多发现 / multicast |

manual 逐字：RC「Messages ... **reliably delivered** ... **Packets are delivered in order** ... very similar to a TCP connection」；UC「**not reliable** ... packets may be lost ... **not retried** by the transport」；UD「**single-packet messages** to/from **any other UD QP** ... **Multicast** ... very similar to a UDP connection」。选型判断句：可靠大消息选 RC；容忍丢包换低开销选 UC；无连接多对多或 multicast 选 UD。

**QP 状态机**：QP 创建后须逐态迁移才能收发，man `ibv_modify_qp.3` L128 逐字「state from: **Reset --> Init --> RTR --> RTS**」（[linux-rdma, rdma-core](../references.md)，教学参照）：

```
RESET ──ibv_modify_qp──► INIT ──► RTR ──► RTS
 创建态        设 port/pkey/      设远端 QPN/PSN/    设本地 PSN/timeout/
              access flags      GID/LID/PATH_MTU   RETRY_CNT/RNR_RETRY
              （本地属性）       （Receive Ready）  （Send Ready）
 RC：RTR/RTS 需带外交换对端 QPN/PSN/GID/LID/MTU；UD：INIT 需 pkey/port/qkey、RTS 需 SQ_PSN（无对端元数据）
```

每态属性（man 迁移属性表）：INIT 设本地端口/pkey/访问权限；RTR（Ready to Receive）设远端 QPN、接收 PSN、GID/LID、路径 MTU——RC 须先拿到对端这些元数据；RTS（Ready to Send）设本地发送 PSN、超时 `TIMEOUT`、重传次数 `RETRY_CNT`、`RNR_RETRY`、`MAX_QP_RD_ATOMIC`。UD 无连接，但仍需本地属性：INIT 设 pkey_index/port/qkey、RTR 仅 STATE、RTS 设 SQ_PSN（man ibv_modify_qp.3 UD 迁移表），只是不需对端元数据（无连接）。

**连接开销账**（RC 全互连）： $n$ 个 rank 两两建 RC 连接，每 rank 对每个对端建 QP，单 QP/对端口径全网共 $n(n-1)$ 个 QP；每个 QP 须 4 态迁移 + 带外交换元数据（QPN/PSN/GID/LID/MTU）。代入 $n = 1024$：每 rank 1023 个对端，全网 $1024 \times 1023 \approx 1.05 \times 10^{6}$ 个 QP。设每 QP bringup 需一次带外 RTT（**教学假设** 100 μs），串行建立约 $1.05 \times 10^{6} \times 100$ μs ≈ 105 s——控制面开销随 $n^{2}$ 涨，靠预建长连接复用摊销（连接是控制面慢路径、传输是数据面快路径，§3 提点）。**一个常见误解须订正**：DeepEP `num_qps_per_rank = 24`（v1.2.1，10.3 走读）不是把连接数压到 $O(n)$——它是**每对端**（per-peer）的 QP 池：`buffer.py` L101 设 `NVSHMEM_IBGDA_NUM_RC_PER_PE = 24`，`ibgda_device.cuh` L77/L80 按 `rcs[pe * num_rc_per_pe + id]` 取 QP，24 恰等于本地 expert 数（buffer.py L47-48 逐字「equals to the number of local experts」），`internode_ll.cu` L164 以 `dst_expert_local_idx` 作 qp_id 选 QP。所以全网 QP 总数 = $24 \cdot n(n-1)$，仍是 $O(n^{2})$（ $n = 1024$ 时约 $2.51 \times 10^{7}$）；多 QP/对端的目的是把不同 expert 的流量分散到同一对端的多条 NIC 队列/路径（per-expert traffic spreading），不是减少连接数。

## 6. IB vs RoCE：链路层差异

同一套 verbs API，InfiniBand 与 RoCE 底下差在哪？manual §2.4 逐字给了总纲：InfiniBand / Ethernet RoCE / Ethernet iWARP 三技术「**share a common user API** ... but have **different physical and link layers**」——verbs 语义（QP/WR/单边双边）三者通用，差异在物理层与链路层。IBTA 规范定义 InfiniBand 与 RoCE（RoCE 标准定义于 InfiniBand Architecture 规范内，[IBTA, ibta-spec](../references.md)，[C]，Vol 1 Release 2.1）。逐维对照（OFED RoCE 页逐字，[NVIDIA, networking docs](../references.md)，[C]）：

| 维度 | InfiniBand | RoCE（v2） |
| --- | --- | --- |
| 链路层流控 | **credit-based**（per-VL 信用，原生无损） | Ethernet best-effort，须 **PFC**（802.1Qbb）造无损 |
| 寻址 | LID（本地标识）+ GID | **GID/IP**，LID = 0（「LID ... is not set ... displayed as zero」） |
| 子网管理 | **须 SM**（Subnet Manager，发现/路由/QoS） | **无 SM**（「The presence of a Subnet Manager (SM) is not required」） |
| 路由 | IB LID forwarding | RoCEv2 UDP/IP 封装，可路由 |
| APM（自动路径迁移） | 支持 | **不支持**（RC QP 无 alternate path） |
| 拥塞控制 | IBTA CC Annex（FECN/BECN/CCT，9.3） | DCQCN（ECN + CNP，9.3） |

**IB credit-based 流控**：链路层按 VL（Virtual Lane，虚拟通道）发信用，接收方缓冲满则该 VL 停发——无损在链路层原生保证，不需 PFC。IB 还有 8 VL + VL15（管理 VL，ConnectX-7 datasheet）、VL 内强有序。**RoCE 的无损要靠 PFC**：Ethernet 默认 best-effort 会丢包，RoCE 须配 PFC per-priority pause 造无损以太网，代价是 PFC storm / victim flow / head-of-line blocking 风险（9.3 详）。所以 IB「原生无损 + SM 集中管理」，RoCE「复用 Ethernet 基础设施 + PFC/DCQCN 调参」——选型不是 verbs 语义之差（二者通用），而是链路层无损机制与运维模型之差。iWARP 第三技术一句登记：RDMA over TCP，同样共享 verbs API，本书不展开。fabric 拓扑/SM 工程/rail 对齐归 10.1，本章只交付链路层机制差异。

## 7. NIC 硬件：ConnectX 与 DMA 引擎

verbs 语义靠 NIC 里哪些硬件模块兑现？以 ConnectX-7 为例（datasheet + HW manual SKU 表逐字，[NVIDIA, connectx7](../references.md)，[C]）：

| 规格项 | ConnectX-7 值 | 出处 |
| --- | --- | --- |
| 最大总带宽 | **400 Gb/s**（max total） | datasheet 逐字 |
| 单口 NDR | 4 lanes × 100 Gb/s/lane = 400G | HW manual SKU MCX75310AAS-NEAT |
| 主机接口 | **PCIe Gen5，up to x32**（x16 + Socket-Direct 辅助卡） | datasheet |
| 端口数 | 1 / 2 / 4 | datasheet |
| 传输卸载 | **Hardware-based reliable transport + congestion control** | datasheet |
| VL / MTU | 8 VL + VL15 / 256–4 KB | datasheet |
| 网内计算 | In-Network Computing（SHARP，10.2） | datasheet |

**双口口径订正**（必读）：双口 QSFP112 SKU MCX755106AS-HEAT = **每口 NDR200**（2 lanes × 100 Gb/s/lane），2 × 200G = 400G 总，**非 2 × 400G**。datasheet portfolio「dual-port NDR (400Gb/s)」为 NDR 家族营销措辞，**以 HW manual SKU 表为准**——ConnectX-7 ASIC max total 400 Gb/s，双口不会翻倍到 800G。800G 是 **ConnectX-8 SuperNIC**（XDR，主机口 PCIe Gen6 @64 GT/s x16，[NVIDIA, connectx8](../references.md)，[C]），其带宽 regime 账归 9.2。

NIC 内部模块与 verbs 语义的对应：**PCIe 接口**（DMA 到 host 或 GPU 内存，Gen5 x16 = 64 GB/s 单向）；**DMA 引擎**（凭 MR 的 lkey/rkey 直读写应用缓冲，zero-copy 的执行者）；**传输引擎**（RC 的 ACK/重传/PSN 序号、可靠有序，CPU offload 的执行者，datasheet「Hardware-based reliable transport」）；**包处理**（封装/解封装、CRC）；**UAR/doorbell**（用户态提交入口，kernel bypass 的落点，§3）。WR 提交后落成 WQE 由 DMA 引擎取走执行，完成写 CQE 进 CQ。NIC 线速 400G = 50 GB/s 与主机 PCIe Gen5 x16 = 64 GB/s 的关系（50 低于 64，NIC 线速限）是 9.2 命题 1 的带宽账，本章只登记数值。NIC 往上接交换机、组 fabric——拓扑/rail/收敛比归 10.1，GDR（NIC 经 PCIe 直访 GPU HBM）归 9.2。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| RDMA 三语义各省一笔：zero-copy 免内核拷贝、kernel bypass 免陷入内核/上下文切换、CPU offload 免本端协议栈 + 远端 CPU | manual §1 逐字「without involving the remote Operating System and CPU」 | §2 |
| 数据路径软件开销：TCP = 2 拷贝 + 2 切换 + 协议栈，RDMA = 1 doorbell；4 KB 教学假设 3.82 vs 0.2 μs（19.1×），1 GB 拷贝项 200 ms 主导 | 命题 1 | §2、练习 1/3 |
| verbs 四对象：MR（lkey/rkey）/ QP（SQ+RQ）/ WR（WQE）/ CQ（CQE）；post_send = 用户态 doorbell 写 UAR，无系统调用 | verbs.h L2692/L3246/L3662/L2962 | §3 |
| 单边 READ/WRITE 携远端 addr+rkey、远端 CPU 不参与；双边 SEND/RECV 须预 post RECV，否则 RNR；READ/Atomic 仅 RC，UD 无 WRITE/READ、max = MTU | manual §3.2 操作×传输表 | §4、练习 4 |
| RC/UC/UD = 可靠一对一 / 不可靠一对一 / 无连接多对多；状态机 RESET→INIT→RTR→RTS；RC 全互连 n(n-1) QP，n=1024 ≈ 1.05M，控制面 O(n²) | man ibv_modify_qp.3 L128；verbs.h L934 | §5、练习 2 |
| IB vs RoCE 共 verbs API、差链路层：IB credit 流控原生无损 + SM + LID；RoCE 须 PFC 无损 + 无 SM + GID/IP + APM 不支持 | manual §2.4 + OFED RoCE 页逐字 | §6 |
| ConnectX-7 max total 400G（双口 = 2×NDR200 非 2×400G）、PCIe Gen5 up to x32、硬件可靠传输 + 拥塞控制；800G = CX8（Gen6 主机口） | datasheet + HW manual SKU 表 | §7、9.2 |

## 练习

1.（计算 · [10]）延迟账代入（命题 1，教学假设分量：单核 memcpy 10 GB/s、 $t_{\mathrm{ctx}} = 1$ μs、 $t_{\mathrm{stack}} = 1$ μs、 $t_{\mathrm{db}} = 0.2$ μs）： $M = 4$ KB。(a) 单次内核拷贝 $t_{\mathrm{copy}}$ 多少 μs；(b) TCP 软件开销 $\Delta_{\mathrm{TCP}}$；(c) RDMA 软件开销 $\Delta_{\mathrm{RDMA}}$ 与 TCP/RDMA 比值。校验：(a) 0.41 μs；(b) 3.82 μs；(c) 0.2 μs、19.1×，与 §2 逐数一致（容差 1%）。
2.（计算 · [20]）QP 连接开销（§5，RC 全互连，单 QP/对端口径）： $n = 1024$。(a) 每 rank 对端数；(b) 全网 QP 总数；(c) 设每 QP bringup 一次带外 RTT（教学假设 100 μs），串行建立约多少秒；(d) 先预测 DeepEP `num_qps_per_rank = 24` 是每对端 QP 池还是全网总数、全网 QP 随 $n$ 几阶，再对照 §5 求 $n = 1024$ 时 DeepEP 全网 QP 总数。校验：(a) 1023；(b) ≈1.05×10⁶；(c) ≈105 s；(d) per-peer 池（NVSHMEM_IBGDA_NUM_RC_PER_PE），全网 24·n(n-1) = O(n²) ≈ 2.51×10⁷，目的 per-expert 流量 spreading 非省连接（10.3）。
3.（推导 · [30]）证明命题 1（数据路径软件开销账）：(a) 逐跳枚举 TCP 发送/接收路径，标出 RDMA 删去的跳；(b) 写出 $\Delta_{\mathrm{TCP}}$/ $\Delta_{\mathrm{RDMA}}$ 表达式并说明每项来源；(c) 给边界：大消息哪项主导、小消息哪项主导；(d) 代入 $M = 1$ GB 求拷贝项。校验：(a) 2 拷贝 + 2 切换 + 协议栈被删；(d) 2×100 ms = 200 ms（10 GB/s 教学假设），与 §2 一致；先预测 (c) 再算。
4.（论证 · [30]）操作语义与 QP 选型（§4/§5）：(a) 为什么 RDMA READ 仅 RC、UD 不支持 RDMA WRITE；(b) 接收方未 post RECV 时 RC SEND 与 UD SEND 各发生什么；(c) 给三场景选型判断句：跨机梯度 all-reduce、多对多服务发现、容忍丢包的遥测大块写。校验：(a) manual §3.2 操作×传输表（READ 仅 RC、WRITE UC/RC、UD max MTU）；(b) RC RNR 重试 / UD 丢包；(c) RC / UD / UC，与 §4/§5 表一致。

## 参考

- [NVIDIA, RDMA programming manual](../references.md)：RDMA Aware Networks Programming User Manual v1.7（key: rdma-programming-manual），[C] 官方规格披露·一手权威，2026-11-25 PDF 一手——§1 RDMA 定义逐字（without involving the remote OS and CPU / TCP/IP require copy operations）；§2.4 三技术共 verbs API 不同物理/链路层；§3.2 RC/UC/UD 逐字 + 操作×传输表（READ/Atomic 仅 RC、UD max MTU、UC/RC 1GB）；§4.5 QP bringup RESET→INIT→RTR→RTS；§8 RDMA_RC 例（§2/§4/§5）
- [linux-rdma, rdma-core](../references.md)：linux-rdma/rdma-core（key: rdma-core），教学参照 + 官方实现，v67.0 master 4b7b208c（2026-11-22）fresh tarball——libibverbs/verbs.h：ibv_qp_type RC=2/UC/UD L934-937 / ibv_qp_state RESET/INIT/RTR/RTS L1084-1088 / ibv_wr_opcode L1137-1144 / ibv_reg_mr L2692 / ibv_create_cq L2962 / ibv_poll_cq L3040 / ibv_create_qp L3246 / ibv_modify_qp L3467 / ibv_post_send L3662 / ibv_post_recv L3671；man ibv_modify_qp.3 L128「Reset --> Init --> RTR --> RTS」+ 每 QP 类型迁移属性表（§3/§5）
- [IBTA, ibta-spec](../references.md)：IBTA InfiniBand Architecture Specification（key: ibta-spec），[C] 官方规格披露，2026-11-25——Vol 1 Release 2.1（2026-07-31，2294 页）定义 InfiniBand 与 RoCE（RoCE 定义于 IB Arch 规范内）/ Vol 2（835 页，XDR 200 Gb/s/lane）；全文 member-gated，公开 What's-New PDF 一手；RC/UC/UD 经 rdma-core + manual 互证（§6）
- [NVIDIA, networking docs](../references.md)：NVIDIA Networking docs（key: nvidia-networking-docs），[C] 官方规格披露，2026-11-25——MLNX_OFED SWUM 24.10 LTS RoCE 页逐字（无 SM / LID=0 / GID / APM 不支持）；IB credit 流控 + VL（§6）
- [NVIDIA, connectx7](../references.md)：ConnectX-7 NDR 400G adapter（key: connectx7），[C] 官方规格披露，2026-11-25——datasheet「Max total bandwidth 400Gb/s」/ PCIe Gen5 up to x32 / 硬件可靠传输 + 拥塞控制 / 8 VL+VL15 / MTU 256-4KB；HW manual SKU 表：单口 NDR 4×100G/lane、双口 QSFP112 = 每口 NDR200（2×200G=400G 总，非 2×400G）（§7）
- [NVIDIA, connectx8](../references.md)：ConnectX-8 SuperNIC（key: connectx8），[C] 官方规格披露，2026-11-25——up to 800 Gb/s XDR / 主机口 PCIe Gen6 @64GT/s x16；带宽 regime 账归 9.2（§7）
- [NVIDIA, hopper-whitepaper](../references.md)：H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露，2026-11-25——PCIe Gen5 节 p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」（§1/§7，recap 7.2 §5）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露，2026-11-25——Gen5 x16 = 64 GB/s 单向；URL 403 残留，数值经 Hopper wp p.49 互证闭合（§1/§7）
- 9.2 / 9.3 / 10.1 / 8.4 / inference 8.3（纯文字指引，不挂链接）：GDR 与 NIC→HBM 带宽账（9.2）、拥塞控制与可靠性（9.3）、fabric 工程（10.1）、NCCL NET transport 走读（8.4）、KV 传输语义（inference 8.3）——本章 verbs 机制的下游与 recap 源
