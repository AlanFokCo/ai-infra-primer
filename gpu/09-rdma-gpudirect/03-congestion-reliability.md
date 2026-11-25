# 9.3 拥塞控制与传输可靠性（Congestion Control and Transport Reliability）

> PFC 已在链路层保证不丢包，为什么还需要 DCQCN？交换机队列涨起来，怎么一步步变成 all-reduce 时间翻倍？IB 的 FECN/BECN/CCT 与 RoCE 的 ECN/CNP/速率调节各怎么让发送端降速？链路误码时 CRC 与重传各兜哪一层、RC 一次超时重传要等多久？本章交付五块：PFC 队列守恒与 pause 触发/传播条件（命题 1/2）、DCQCN 完整机制（RP/CP/NP 三角色 + 论文式 (1)-(4) + fast recovery 闭式命题 3）、IB IBTA CC（FECN→BECN→CCT 包间延迟）、传输可靠性（CRC/retry/RC 重传超时/链路降级）、GPU 侧影响账（拥塞 → $\beta$ 恶化 → 集合延迟涨 → 通信账恶化）与调参概念/NIC 卸载 | 前置：9.1（RC/UC/UD 与 QP 状态机、IB credit 流控 vs RoCE PFC、LID/GID）、9.2（GDR 路径与 NIC→HBM 带宽账、多 NIC 聚合）、8.1/8.2（α-β 模型与 ring 带宽项，recap 源）、7.2（机间 50 GB/s 口径）；下游 10.1（fabric 拓扑与收敛比）、10.2（大规模 fabric 调优与拓扑感知集合）、10.3（a2a incast 拥塞） | 锚点：DCQCN Zhu SIGCOMM 2015（[Zhu et al., 2015](../references.md)，[B] 老经典无 arXiv，pp 523-536，式 (1)-(4) 逐字）、IB CC whitepaper（[Crupnicoff et al., IB QoS/CC whitepaper](../references.md)，[C] 官方规格披露·一手权威，IBTA CC Annex 机制逐字）、IBTA 规范（[IBTA, ibta-spec](../references.md)，[C]）、NVIDIA networking docs（[NVIDIA, networking docs](../references.md)，[C]，OFED ECN/PFC/RoCE/Programming 页）、rdma-core（[linux-rdma, rdma-core](../references.md)，教学参照 + 官方实现，v67.0，TIMEOUT/RETRY_CNT 属性）、NCCL（[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，2.32.7-1 / 351f33fd，net_ib IB_TIMEOUT=20/RETRY_CNT=7）、ConnectX-7/8（[NVIDIA, connectx7](../references.md) / [connectx8](../references.md)，[C]，硬件拥塞控制）——官方规格与论文 2026-11-25 一手核实

## 1. 三个问题与本章口径

PFC 已经让以太网链路层不丢包，RoCE 为什么还要再叠一层 DCQCN？一条流被降速一半，跨机 all-reduce 的时间账会恶化多少——是只慢那条流，还是整个集合一起慢？IB 原生无损，它的拥塞控制又靠什么信号、什么执行器？本章回答三个问题：无损网络里拥塞以什么形式存在、PFC 的 pause 怎么触发与传播（§2-3）？DCQCN 与 IBTA CC 各用什么反馈回路把发送端速率压下来（§4-5）？误码与丢包靠哪几层可靠性机制兜住、拥塞最终怎么落到 GPU 通信账上（§6-7）？

**本章口径**（章脊锁定）：主口径 **ConnectX-7 NDR 400G 单口**（线速 400 Gb/s = 50 GB/s 单向， $10^{9}$ 口径，recap 9.1 §7）+ **RoCEv2（DCQCN）与 InfiniBand（IBTA CC）双机制口径**；贯穿实例 $M = 1$ GB = $10^{9}$ B；通信精度 bf16（梯度口径沿 recap 源）。DCQCN 部署参数取论文披露值（CNP 间隔 50 μs、权重 timer 55 μs、fast recovery 迭代 5 次）；标记阈值、缓冲大小、反应时间等未披露分量逐处标**教学假设口径**，结构结论（触发条件、收敛形式、传播链）才是交付物。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**）； $M$ = 被传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s）。局部符号： $\mathrm{RC}$ / $\mathrm{RT}$ = DCQCN 当前速率 / 目标速率（论文原名保留）； $w$ = DCQCN 降速权重（**论文原符号 $\alpha$，本章译 $w$—— $\alpha$ 在本章专归消息延迟，译名注见 §4**）； $g$ = 权重增益； $\mathrm{RAI}$ = 加性增量； $F$ = fast recovery 迭代次数阈值（无量纲计数）； $T_{\mathrm{CNP}}$ / $T_{w}$ = CNP 生成间隔 / 权重 timer（论文原符号 N/K，译 T 下标避免占用 $N$）； $A$ / $C$ = 端口聚合到达率 / 服务率； $Q$ / $Q_{\mathrm{th}}$ = 队列占用 / pause 阈值； $\mathrm{IPD}$ = 包间延迟。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 T 下标。

**边界声明**：本章讲**拥塞控制与可靠性的硬件/协议机制**（代码 scope = 机制片段级：rdma-core QP 属性 + NCCL net_ib 超时锚，非模块全量走读）。fabric 拓扑与收敛比归 10.1，大规模 fabric 调优工程与拓扑感知集合归 10.2，a2a incast 的 transport 级展开归 10.3；GDR 带宽账 recap 9.2 不重推，ring 消息量与 α-β 模型 recap 8.1/8.2（training 4.1 源）不重推。平台编排（K8S 调度/容器网络）归 cluster 维度。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方/论文披露口径，非新跑 benchmark。

## 2. 无损网络为什么还需要拥塞控制

先回答一个更基本的问题：不丢包的网络，拥塞去哪了？答案：堆在交换机队列里。丢包式网络（TCP）用丢包当拥塞信号，无损网络把丢包摘掉之后，拥塞改以**排队延迟**与**pause 反压**两种形式存在——队列涨、延迟涨、pause 沿链路向上游传播，有效带宽照样掉。

**定义**（拥塞）：所谓拥塞（congestion），是指某端口/链路的聚合到达率持续超过其服务率，导致队列堆积、排队延迟上升的状态。正例：两条 50 GB/s 的流同时发往一个 50 GB/s 出口（2:1 incast），队列以 50 GB/s 速率涨；边界反例：瞬时突发但总量低于服务率，队列涨而复消，不构成持续拥塞。**无损 ≠ 无拥塞**（口径：无损指链路层不丢包，PFC/credit 保证；拥塞指到达率超服务率，两者正交）——这是本章全部机制的出发点。

GPU 侧的影响链先立起来，后面逐节填机制：

```
拥塞（到达率 > 服务率）
  ├─► 队列堆积 ─► 排队延迟涨 ─► α（每消息延迟）涨
  ├─► PFC pause 反压 ─► 上游停发 ─► 有效带宽降（HoL/victim flow）
  └─► ECN/FECN 标记 ─► DCQCN/IBTA CC 降速 ─► BW_eff 降 ─► β_eff = 1/BW_eff 涨
        └─► 集合通信带宽项 2(n-1)/n·M·β_eff 涨（recap 8.2 式 (2)）
              └─► 同步语义取 max ─► 一条慢 rail 拖慢整个集合（§7）
                    └─► 通信暴露涨 ─► training step / inference 传输预算恶化
```

无损机制（PFC/credit）与拥塞控制（DCQCN/IBTA CC）是**两层不同的兜底**：前者兜瞬时缓冲溢出（微秒级，逐跳），后者兜持续过载（毫秒级，端到端）。只有前者没有后者，pause 会传播成风暴（§3）；只有后者没有前者，降速反应赶不上队列溢出、丢包触发秒级重传（§6）。两层都要，且参数要互相让路（§8）。

## 3. PFC：链路层无损的代价

PFC 凭什么让以太网不丢包，代价又是什么？所谓 PFC（Priority Flow Control，IEEE 802.1Qbb），是指按优先级（802.1p 的 8 个 traffic class）逐类发送 pause 帧、让上游暂停该类发送的链路层流控：接收侧缓冲快满时向发送侧发 pause，发送侧停发对应优先级，队列得以排空——不丢包由此造出来（OFED Flow Control 页逐字：「applies pause functionality to specific classes of traffic ... lossless service for the RoCE traffic and best-effort service for the standard Ethernet traffic」，[NVIDIA, networking docs](../references.md)，[C]）。RoCE 部署把 RDMA 流量放在一个专用无损优先级，其余优先级走 best-effort。

**命题 1**（队列守恒与 pause 触发）：设端口聚合到达率 $A$、服务率（线速） $C$、队列占用 $Q$。若 $A$ 持续大于 $C$，则队列按差率增长；当 $Q$ 达到 pause 阈值 $Q_{\mathrm{th}}$ 时，该端口向上游发 pause：

$$
Q(t + \Delta t) - Q(t) = (A - C)\thinspace \Delta t \quad (A > C \text{ 持续}), \qquad Q \ge Q_{\mathrm{th}} \Rightarrow \text{pause} \qquad (1)
$$

**证明**（路线：流体守恒 + 阈值比较，两步）：其一，队列是缓冲守恒量——到达字节减去服务字节即净增量， $A$ 持续大于 $C$ 时净增为正、队列单调涨（ $Q$ 为空时服务率取 $\min(A, C)$，不欠账）。其二，PFC 的触发规则就是阈值比较： $Q$ 达到 $Q_{\mathrm{th}}$ 即发 pause。证毕。正例：2:1 incast， $A = 100$ GB/s、 $C = 50$ GB/s，队列以 50 GB/s 涨；边界反例： $A = C$ 时队列不涨，pause 不触发——**pause 的根因是持续过载，不是瞬时突发**。

全数字代入（**教学假设口径**： $Q_{\mathrm{th}} = 512$ KB）：2:1 incast 下从空队列到触发 pause 只需 $512 \times 10^{3} / (50 \times 10^{9}) = 10.24$ μs——比 DCQCN 的 CNP 间隔 50 μs（§4）还短。这就是为什么 PFC 必须存在：端到端降速回路还没闭合，缓冲已经先满了。

**pause 之后还要留余量**（headroom 账）。pause 帧从接收侧传到发送侧、发送侧反应停发，需要一段时间 $t_{\mathrm{react}}$；这段时间里发送侧仍按线速注入，这些字节必须被缓冲吸收，否则照样溢出。所以缓冲须在 $Q_{\mathrm{th}}$ 之上再留 headroom：

$$
B_{\mathrm{headroom}} \ge C \cdot t_{\mathrm{react}} + \mathrm{MTU} \qquad (2)
$$

代入（**教学假设** $t_{\mathrm{react}} = 2$ μs，含传播与反应）： $50 \times 10^{9} \times 2 \times 10^{-6} = 100$ KB，加一个 MTU（4 KB）——headroom 至少 104 KB。阈值设得越高、headroom 留得越少，丢包风险越大；缓冲总量固定时，两者互相挤占。

**命题 2**（pause 传播与 PFC storm 条件）：端口 $i$ 的出口队列若无法排空（下游被 pause，或 $A_i$ 持续大于 $C_i$），则 $Q_i$ 涨至阈值、端口 $i$ 转而 pause 它的上游——pause 沿链路逐跳向上游传播。PFC storm 持续的条件是：传播链上**没有端点降速让路**（链持续存在），或依赖关系**成环**（A 等 B、B 等 A，死锁）。

**证明**（路线：对传播链逐跳递归应用命题 1，三步）：其一，下游被 pause 意味着端口 $i$ 的服务率 $C_i$ 被压到 0（或低于 $A_i$），由命题 1， $Q_i$ 单调涨。其二， $Q_i$ 达阈值即 pause 端口 $i$ 的上游，把同一条件递归传给上一跳。其三，若链上某端点（发送 NIC）降速使某跳 $A$ 降到 $C$ 以下，该跳队列排空、pause 解除，链断；若没有端点让路或依赖成环，链持续——storm 成立。证毕。

```
PFC pause 传播（拥塞点在 leaf 出口，逐跳反压）：
  src NIC ──► leaf 入口 ──► leaf 出口（拥塞：A > C，队列涨）
     ▲                          │
     └── pause ◄── 上游端口 Q 涨 ◄┘  （出口排不空 → 入口队列涨 → pause 上游）
  victim flow：不经拥塞点、但与拥塞流共享上游端口/优先级的流，一并被停
```

三个代价逐条登记（DCQCN 论文 §2.2 逐字口径，[Zhu et al., 2015](../references.md)，[B]）：其一，**head-of-line blocking**——pause 按优先级整类停，不区分流，同一优先级里不拥塞的流也被停（HoL）；其二，**victim flow**——不经拥塞点的流因共享上游链路被牵连（论文 Fig 4）；其三，**congestion spreading / PFC storm**——拥塞从一点沿 pause 链扩散到全网（论文 Fig 3），成环即死锁。**PFC 的结构性缺陷是把「逐流问题」当成「逐类问题」处理**：它不知道哪条流该降速，只能整类停。这正是端到端逐流拥塞控制（DCQCN）的动机——让端点降速替代逐跳 pause，把 PFC 压回「罕触发」的兜底位（论文设计目标逐字口径）。

> **坑**：把「无损」读成「无拥塞」，以为配了 PFC 就万事大吉。触发条件：只配 PFC 不配 DCQCN/ECN；现象：incast 下 pause 链扩散、victim flow 被停、吞吐塌陷甚至死锁。事实：PFC 兜瞬时溢出（10 μs 级），持续过载必须靠端点降速（ms 级回路）；规避：无损优先级 + ECN 标记 + DCQCN 三件一起配（§8），PFC 只作最后兜底。

## 4. ECN 与 DCQCN：端到端逐流降速

DCQCN 怎么知道该降速、降多少、何时恢复？所谓 DCQCN（Data Center Quantized Congestion Notification），是指 RoCE 部署的端到端拥塞控制：交换机用 ECN 标记拥塞包、接收 NIC 回 CNP 通知、发送 NIC 按权重乘性降速再逐步恢复——基于 QCN 与 DCTCP 的组合，交换机只需标准 RED + ECN 能力，速率调节全在 NIC 硬件实现（[Zhu et al., 2015](../references.md)，[B]，SIGCOMM 2015，无 arXiv）。

**定义**（ECN）：所谓 ECN（Explicit Congestion Notification，显式拥塞通知），是指交换机在 IP 头标记拥塞、接收端把标记反馈给发送端、**不丢包**即通知拥塞的机制（OFED ECN 页逐字：「notifying all ends ... when congestion occurs ... without dropping packets」，该实现指 RoCEv2，[NVIDIA, networking docs](../references.md)，[C]）。

三角色分工（论文 RP/CP/NP）：

```
发送 NIC = RP（Reaction Point）          接收 NIC = NP（Notification Point）
  按 RC 速率注入 ──► 交换机 = CP（Congestion Point）──► 收到 ECN 标记包
  收 CNP → 乘性降速      egress 队列 RED 式标记            │
  无 CNP → 逐步恢复      （Kmin/Kmax/Pmax）                ▼
       ▲                                          生成 CNP（每流每 T_CNP 至多一个）
       └──────────── CNP 高优先级回传 ◄────────────────┘
```

**CP（交换机标记）**：egress 队列按 RED 式规则标记——队列低于 $K_{\min}$ 不标记， $K_{\min}$ 到 $K_{\max}$ 之间按概率线性升至 $P_{\max}$，超过 $K_{\max}$ 全标记（论文 Fig 5 口径）。标记概率随队列深度上升，给端点一个「拥塞程度」的连续信号而非 0/1。

**NP（接收 NIC 生成 CNP）**：收到 ECN 标记包后，若该流最近 $T_{\mathrm{CNP}} = 50$ μs（论文部署值，原符号 N）内未发过 CNP（Congestion Notification Packet，拥塞通知包）则立即发一个，此后每 $T_{\mathrm{CNP}}$ 至多一个；CNP 走高优先级，确保反馈不被拥塞本身吞掉。

**RP（发送 NIC 速率调节）**：四个动作，论文式 (1)-(4) 逐字（**译名注**：论文原符号 $\alpha$ 是降速权重，本章译 $w$—— $\alpha$ 专归消息延迟，spec §4.3 符号纪律；其余符号 $\mathrm{RC}$ / $\mathrm{RT}$ / $g$ / $\mathrm{RAI}$ 保留原名）：

收到 CNP，乘性降速（论文式 (1)）：

$$
\mathrm{RT} = \mathrm{RC}, \qquad \mathrm{RC} = \mathrm{RC}\left(1 - \frac{w}{2}\right), \qquad w = (1-g)\thinspace w + g \qquad (3)
$$

$T_{w} = 55$ μs（论文部署值，原符号 K， $T_{w}$ 大于 $T_{\mathrm{CNP}}$）内无 CNP，权重衰减（论文式 (2)）：

$$
w = (1-g)\thinspace w \qquad (4)
$$

fast recovery，迭代 $F = 5$ 次（论文式 (3)）：

$$
\mathrm{RC} = \frac{\mathrm{RT} + \mathrm{RC}}{2} \qquad (5)
$$

additive increase（论文式 (4)）：

$$
\mathrm{RT} = \mathrm{RT} + \mathrm{RAI}, \qquad \mathrm{RC} = \frac{\mathrm{RT} + \mathrm{RC}}{2} \qquad (6)
$$

**符号**： $\mathrm{RC}$ = 当前注入速率； $\mathrm{RT}$ = 目标速率（降速前的速率快照）； $w$ = 降速权重，**初值 1**； $g$ = 权重增益（0 到 1 之间，论文 fluid model 整定参数）； $\mathrm{RAI}$ = 速率加性增量。机制读法：CNP 来了先把 $\mathrm{RT}$ 钉在当前速率（记住「拥塞前能跑多快」），再按 $w$ 乘性砍 $\mathrm{RC}$，同时把 $w$ 往 1 推（连续拥塞 → 砍得更狠）；长时间无 CNP 则 $w$ 指数衰减（恢复期再遇拥塞 → 砍得温和）。恢复分三档，由 timer（T）与 byte counter（BC）双触发计数（QCN 式）： $\max(T, BC)$ 小于 $F$ 走 fast recovery（式 (5)，向 $\mathrm{RT}$ 折半逼近）； $\min(T, BC)$ 大于 $F$ 走 hyper increase（更激进的快速 ramp）；其余走 additive increase（式 (6)， $\mathrm{RT}$ 加性涨、 $\mathrm{RC}$ 取中点）。**无 slow start**：流启动即线速（论文逐字口径）——这对 GPU 集合通信是双刃剑：突发立即打满队列，靠 ECN 回路在 RTT 尺度内压回来。

全数字走读（章脊口径 $\mathrm{RC}$ 初值 = 线速 50 GB/s， $w$ 初值 1）：**第一个 CNP**： $\mathrm{RT} = 50$， $\mathrm{RC} = 50 \times (1 - 1/2) = 25$ GB/s， $w = (1-g) \times 1 + g = 1$（恒等，与 $g$ 无关）——首个 CNP 恰好减半。**第二个 CNP 紧随**（ $w$ 尚未衰减）： $\mathrm{RT} = 25$， $\mathrm{RC} = 12.5$ GB/s。此后无 CNP，fast recovery 向 $\mathrm{RT} = 25$ 折半逼近：18.75 → 21.875 → 23.4375 → 24.21875 → 24.609375（5 次）。权重衰减的节奏（**教学假设** $g = 1/16$，论文 $g$ 为整定参数、部署值随固件）：一个 $T_{w}$ 无 CNP 后 $w = 15/16 = 0.9375$，此时再遇 CNP 砍幅因子 $1 - 0.9375/2 = 0.53125$（比减半温和）；5 个 timer 后 $w = (15/16)^{5} \approx 0.724$。

**命题 3**（fast recovery 闭式）：从 $`\mathrm{RC}_{0}`$ 出发、目标 $\mathrm{RT}$ 不变，式 (5) 迭代 $k$ 次后

$$
\mathrm{RC}_{k} = \mathrm{RT} - \frac{\mathrm{RT} - \mathrm{RC}_{0}}{2^{k}} \qquad (7)
$$

即与目标的差距每迭代减半（几何收敛）。**证明**（路线：对 $k$ 归纳，基例 $k=0$、归纳步代入式 (5)）——完整证明留作练习 3。代入 $`\mathrm{RC}_{0} = 25`$、 $\mathrm{RT} = 50$： $k=5$ 时 $`\mathrm{RC}_{5} = 50 - 25/32 = 49.21875`$ GB/s，差距 0.78 GB/s（占 $\mathrm{RT}$ 的 1.56%）；要把差距压到 $\mathrm{RT}$ 的 1% 以内需 $k \ge 6$（ $25/64 = 0.39$ GB/s）。这解释了 $F=5$ 的整定含义：5 次折半已收回九成以上差距，再往上交给加性增死去探新增带宽。

additive increase 一步（**教学假设** $\mathrm{RAI} = 2$ GB/s）： $\mathrm{RT} = 50 + 2 = 52$， $\mathrm{RC} = (52 + 49.21875)/2 = 50.609$ GB/s——超过线速的部分被物理端口封顶（注入率不可能超过线速，机制口径），实际回到 50。AIMD 结构（乘性减 + 加性增）的收敛行为是定性结论：多流竞争同一瓶颈时，乘性减让速率比例保持、加性增让总量爬升，反复迭代收敛到公平份额附近振荡（论文 §5 fluid model 整定 $\mathrm{RAI}$ / $F$ / $g$ 权衡收敛速度与吞吐振荡，16:1/32:1 incast 评估）。

> **提点**：DCQCN 与 PFC 的分工是「端点降速替代逐跳 pause」。CNP 回路时间尺度 50 μs，PFC 触发尺度 10 μs（§3）—— incast 瞬间 PFC 先兜住，DCQCN 随后把端点速率压下来、队列排空、pause 解除。调参的目标就是让 PFC 罕触发（论文设计目标）：ECN 阈值设得比 PFC 阈值低，标记先于 pause。

## 5. InfiniBand 拥塞控制：FECN → BECN → CCT

IB 原生 credit 流控已无损（recap 9.1 §6），为什么还要 IBTA CC？因为 credit 只兜瞬时缓冲溢出（逐跳、逐 VL），持续过载仍会让队列堆积、延迟上涨——IB 的答案是 IBTA Congestion Control Annex 定义的 CCA 机制（[Crupnicoff et al., IB QoS/CC whitepaper](../references.md)，[C] 官方规格披露·一手权威；IBTA 规范全文 member-gated，机制经 whitepaper 逐字，[IBTA, ibta-spec](../references.md)，[C]）。三阶段逐字口径：

```
① 交换机检测拥塞 ─► 置 FECN bit（Forward ECN，随数据包向前传）
② 目的 HCA 收 FECN ─► 回 BECN（Backward ECN）给源
     connected QP：piggyback 在 ACK 上；unconnected：专用 CN 包
③ 源 HCA 收 BECN ─► 查 CCT 表，插入包间延迟，throttle 注入率
```

**定义**（CCT/CCTI/CCM）：所谓 CCT（Congestion Control Table），是指每张 HCA 维护的降速表，每项是一个 **Inter-Packet Delay（包间延迟）**；CCTI（CCT Index）是表索引，每收一个 BECN 就加 $\mathrm{CCTI\_Increase}$（索引越深、延迟越大、注入越慢）；CCM（Congestion Control Manager）是设置全部参数的管理实体。交换机侧按 per-port per-VL 相对阈值标记（CCM 设，0 到 15，0 = 不标记、15 = 激进）。恢复：HCA timer 到期且无新 BECN，CCTI 递减，包间延迟变小，注入率逐步回升。降速可 per-QP 或 per-port per-SL（映射到 VL）施加。

**注入率账**（包间延迟怎么换算成带宽）：设包长 $L_{\mathrm{pkt}}$、线速 $`\mathrm{BW}_{\mathrm{line}}`$，包发送时间 $`t_{\mathrm{tx}} = L_{\mathrm{pkt}} / \mathrm{BW}_{\mathrm{line}}`$；插入包间延迟 $\mathrm{IPD}$ 后，每包占时 $t_{\mathrm{tx}} + \mathrm{IPD}$，有效注入率

$$
\mathrm{BW}_{\mathrm{inject}} = \frac{L_{\mathrm{pkt}}}{t_{\mathrm{tx}} + \mathrm{IPD}} = \mathrm{BW}_{\mathrm{line}} \cdot \frac{t_{\mathrm{tx}}}{t_{\mathrm{tx}} + \mathrm{IPD}} \qquad (8)
$$

全数字代入（4 KB 包、50 GB/s 线速）： $t_{\mathrm{tx}} = 4096 / (50 \times 10^{9}) = 81.92$ ns； $\mathrm{IPD} = 81.92$ ns（等于发送时间）时注入率恰好减半到 25 GB/s； $\mathrm{IPD} = 0$ 时回到 50。**IPD 与发送时间之比就是降速旋钮**——CCT 表把「BECN 计数」翻译成「包间延迟」，硬件逐包执行，CPU 不参与。

IB 与 RoCE 两套机制对照（机制层同构：标记 → 反馈 → 降速 → 恢复；执行器与无损底座不同）：

| 维度 | InfiniBand（IBTA CC） | RoCEv2（DCQCN） |
| --- | --- | --- |
| 拥塞信号 | FECN（随包向前） | ECN（IP 头标记） |
| 反馈 | BECN（piggyback ACK / CN 包） | CNP（专用包，每流每 50 μs 至多一个） |
| 执行器 | CCT 表包间延迟（CCTI 索引） | $\mathrm{RC}$ 速率（乘性减 / 折半恢复 / 加性增） |
| 恢复 | timer 到期 CCTI 递减 | $w$ 衰减 + fast recovery / AI / hyper |
| 无损底座 | credit-based 链路流控（per-VL，原生） | PFC（802.1Qbb，须配置） |
| 管理实体 | CCM | SM/固件参数 + OFED sysfs（§8） |

结构结论：两套都是「交换机标记 + 端点降速」的端到端回路，差别在 IB 用**包间延迟**（时域旋钮）、RoCE 用**速率调节**（频域旋钮），以及 IB 的无损是原生的、RoCE 的无损要 PFC 造。对 GPU 通信的含义相同：拥塞时有效带宽降、 $\beta$ 涨，账走 §7 同一条链。

## 6. 传输可靠性：CRC、重传与链路降级

误码和丢包靠哪几层兜住？分三层：链路层检错重传、传输层可靠交付、物理层降级运行。

**链路层 CRC 与 retry**：IB 链路层对每包算 CRC，检出错即在**链路层**重传（link-level retry），传输层不感知——误码被消化在逐跳范围内（IBTA 链路层机制，[IBTA, ibta-spec](../references.md)，[C]；verbs manual 互证 IB 传输层可靠性，[NVIDIA, RDMA programming manual](../references.md) 经 9.1 recap）。RoCE 的链路层是以太网：FCS 检错、坏帧丢弃，无损靠 PFC 兜缓冲溢出而非误码——误码丢帧最终由 RC 传输层重传兜底。**所以 RoCE 对链路误码更敏感**：IB 误码走链路层快速 retry，RoCE 误码可能落到传输层秒级超时（下文）。

**RC 重传与超时账**：RC 的可靠有序由 PSN 序号 + ACK/NAK + 超时重传保证（recap 9.1 §5：RTS 态设 TIMEOUT/RETRY_CNT/RNR_RETRY，man `ibv_modify_qp.3` 迁移属性表，[linux-rdma, rdma-core](../references.md)，教学参照）。超时值按指数编码（单位 μs）：

$$
T_{\mathrm{timeout}} = 4.096 \times 2^{\mathrm{timeout}} \qquad (9)
$$

NCCL 默认 `NCCL_IB_TIMEOUT` = 20（`src/transport/net_ib/connect.cc` L26，[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，2.32.7-1 / 351f33fd）： $4.096 \times 2^{20}$ μs = **4.295 s**； `NCCL_IB_RETRY_CNT` = 7（L32）——超时重传至多 7 次，仍失败则 QP 进错误态、通信报错。对比章脊口径：1 GB 消息正常传输 20 ms，一次超时等待 4.3 s 是它的 **215 倍**。**这就是无损网络对 RDMA 的意义**：丢包触发的不是「慢一点」，而是秒级尾延迟；PFC/credit 把丢包概率压下去，就是把这种尾延迟压下去。重传本身还注入重复流量、进一步加剧拥塞点负担——丢包 → 重传 → 更拥塞的正反馈，是「可靠性机制与拥塞控制必须协同」的又一层原因。

**链路降级**（link degradation）：链路误码率升高或训练失败时，端口可降宽/降速运行（如 4 lane 降 1 lane、NDR 降 HDR）。带宽账按 lane 几何重算：NDR 400G = 4 lane × 100 Gb/s/lane（recap 9.1 §7 SKU 表）；降到 1 lane 即 100G = 12.5 GB/s， $\beta$ 涨 4 倍——一条降级链路混在 rail 里，就是 §7 的 straggler。降级检测靠错误计数器（CRC error / link down 事件），诊断流程归 11.3，本章只登记带宽后果。

> **坑**：把 RC 重传当「无感兜底」。触发条件：无损配置缺失（PFC 未配/优先级错配）或链路误码率高；现象：偶发秒级尾延迟（4.3 s 超时 × 重试），集合通信 max 语义下全局被拖慢。规避：先查无损配置与链路计数器（§8/11.3），再谈性能调优；超时参数（TIMEOUT/RETRY_CNT）是兜底不是常态路径。

## 7. GPU 侧影响账：拥塞怎么落到通信账上

一条 rail 被降速一半，all-reduce 慢多少？先给结论：同步集合取 max，一条慢 rail 让整个集合翻倍——不是只慢那条流。下面把链补完整。

**有效带宽与 $\beta$ 恶化**（口径： $`\mathrm{BW}_{\mathrm{eff}}`$ = 拥塞折扣后的稳态有效带宽，非线速； $`\beta_{\mathrm{eff}} = 1/\mathrm{BW}_{\mathrm{eff}}`$）。拥塞从两个方向恶化 α-β 模型（recap 8.1/8.2，不重推）：排队延迟计入每消息延迟（ $\alpha_{\mathrm{eff}} = \alpha_{0} + t_{\mathrm{queue}}$），降速/停发抬高每字节时间（ $\beta_{\mathrm{eff}}$ 涨）。ring 带宽项 $2(n-1)/n \cdot M \beta$（8.2 式 (2)）里的 $\beta$ 换成 $\beta_{\mathrm{eff}}$，延迟项里的 $\alpha$ 换成 $\alpha_{\mathrm{eff}}$，结构不变、数值恶化。

**straggler 账**（同步语义）：集合完成时间取各 rank 的最大值——all-reduce 结束要求**每个** rank 都收齐全和，最慢的 rank 决定全局。代入（教学简化口径： $n = 16$ rank 各在不同节点、单 rail 跨机环，忽略节点内分层——分层账见 8.2 §7 / 10.2）：每 rank 发送 $2 \times 15/16 \times 1 = 1.875$ GB；健康 rail 50 GB/s → 带宽项 37.5 ms；**一条 rail 被降速到 25 GB/s** → 该 rank 75 ms；集合完成时间 = max = **75 ms，翻倍**。其余 15 条 rail 再快也无济于事——这就是「拥塞 → 集合延迟涨」的定量形式，也是 fabric 调优要保**每条 rail 都不拥塞**而非平均带宽的原因（10.2）。

**通信账恶化往上层传**：training 侧，通信暴露时间涨则 step time 涨（overlap 条件「通信时间不高于计算时间」被打破，training 4.1 recap，不重推）；inference 侧，KV 传输预算被挤占（inference 8.3 暴露时间账 recap，不重推）。拥塞的代价最终记在训练吞吐与 serving 延迟上，不是记在网络计数器上。

**GDR 流量与拥塞的交互**（recap 9.2，不重推带宽账）：三个机制要点。其一，**同步突发**——集合通信让所有 rank 同时收发，GDR 使 NIC 直 DMA HBM、发送侧无 CPU pacing，线速突发直接打向接收侧 leaf 出口，incast 是集合流量的常态而非异常（a2a 的 incast 更尖，10.3）。其二，**多 QP 逐流状态**——NCCL 多 channel × 每连接 QP 数（`NCCL_IB_QPS_PER_CONNECTION`）产生大量并发流，DCQCN 的逐流速率状态在 NIC 硬件维护（ConnectX-7 datasheet「Hardware-based congestion control」逐字，[NVIDIA, connectx7](../references.md)，[C]），流数涨不占 CPU。其三，**时间尺度错配**——CNP 间隔 50 μs，1 GB 消息传输 20 ms，大消息在生命周期内能看到 CC 收敛后的稳态；而百微秒级短消息（DeepEP low-latency 口径，10.3）可能在回路收敛前就发完，看到的是原始排队延迟而非降速后的稳态——**延迟敏感流量对拥塞的暴露形式与带宽敏感流量不同**，前者吃 $\alpha$ 恶化、后者吃 $\beta$ 恶化。

## 8. 调参概念与 NIC 卸载

调参调的是什么？三个旋钮，各管一层：**ECN 阈值**（ $K_{\min}$ / $K_{\max}$ / $P_{\max}$）管标记激进度——设得低则标记早、降速早、PFC 罕触发，但吞吐振荡风险涨；**PFC 优先级与 headroom**管无损兜底——哪个优先级无损、阈值与 headroom 怎么分缓冲（§3 式 (2)）；**rate limiter**（per-QP packet pacing，OFED Programming 页，[NVIDIA, networking docs](../references.md)，[C]）管注入上限——静态限速兜底突发。RoCE 侧 ECN 的 RP/NP 算法经 sysfs 配置（`/sys/class/net/<if>/ecn/roce_rp|roce_np/...`，per-device/port/priority，OFED ECN 页逐字）；PFC 经 `mlnx_qos --pfc` 设 8 优先级开关，DCBX/LLDP 自动协商。大规模 fabric 的调参工程（多作业干扰、拓扑与 rail 对齐、万卡实测）归 10.1/10.2，本章只交付旋钮的机制含义。

**NIC 卸载**：拥塞控制与可靠传输都跑在 NIC 硬件——ConnectX-7 datasheet 逐字「Hardware-based congestion control」+「Hardware-based reliable transport」（[NVIDIA, connectx7](../references.md)，[C]）；ConnectX-8 进一步「advanced routing and telemetry-based congestion control」（[NVIDIA, connectx8](../references.md)，[C]）。加上 checksum/TSO（OFED Programming 页），数据面的协议处理、降速执行、重传定时器全部不占 CPU 与 GPU SM——这是 9.1 CPU offload 语义在拥塞控制上的延伸：GPU 只管提交请求与轮询完成，拥塞回路在 NIC 里自治闭合。平台编排层（K8S 调度/容器网络策略）归 cluster 维度，本章不展开。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 无损 ≠ 无拥塞：拥塞 = 到达率持续超服务率，无损网络里以排队延迟 + pause 反压存在；PFC 兜瞬时（μs 级）、CC 兜持续（ms 级） | §2 定义与影响链 | §2、练习 4 |
| pause 触发： $A$ 持续大于 $C$ → $Q$ 按差率涨 → $Q \ge Q_{\mathrm{th}}$ 发 pause；2:1 incast 512 KB 阈值 10.24 μs 触发；headroom ≥ $C \cdot t_{\mathrm{react}}$ + MTU = 104 KB | 命题 1、式 (1)(2) | §3、练习 4 |
| pause 逐跳传播；storm 持续 = 无端点让路或依赖成环；HoL/victim flow 是逐类 pause 的结构性代价 | 命题 2 | §3 |
| DCQCN = ECN 标记 + CNP 反馈（每流每 50 μs 至多一个）+ NIC 速率调节；CNP 乘性减（ $w$ 初值 1 首砍减半）、timer 55 μs 权重衰减、fast recovery F=5 折半、AI 加性增、无 slow start | 式 (3)-(6)，论文式 (1)-(4) 逐字（ $\alpha$ 译 $w$） | §4、练习 1 |
| fast recovery 几何收敛：差距每迭代减半， $`\mathrm{RC}_{5} = 49.22`$ GB/s（25→50），差距小于 1% 需 $k \ge 6$ | 命题 3、式 (7) | §4、练习 3 |
| IB IBTA CC = FECN→BECN→CCT 包间延迟；IPD = 发送时间 → 注入率减半（4 KB 包 81.92 ns）；credit 原生无损 vs RoCE 须 PFC | 式 (8)、§5 对照表 | §5、练习 2 |
| RC 超时 = 4.096 μs × 2^timeout；NCCL 默认 20 → 4.295 s（1 GB 正常 20 ms 的 215×），RETRY_CNT=7；链路降级 4 lane→1 lane 带宽 ÷4 | 式 (9)、net_ib L26/L32 | §6、练习 2 |
| 拥塞 → $\beta_{\mathrm{eff}}$ 涨 + $\alpha_{\mathrm{eff}}$ 涨 → 集合同步取 max：一条 rail 减半 → 16 rank all-reduce 37.5→75 ms 翻倍；GDR 同步突发 + 多 QP 硬件 CC + 短消息吃 $\alpha$ 恶化 | §7 straggler 账 | §7、练习 4 |
| 三旋钮：ECN 阈值 / PFC 优先级+headroom / rate limiter；CC 与可靠传输 NIC 硬件卸载（CX7 逐字），不占 CPU/SM | §8 | §8、10.1/10.2 |

## 练习

1.（计算 · [10]）DCQCN 降速代入（式 (3)， $w$ 初值 1，线速 50 GB/s）：(a) 第一个 CNP 后 $\mathrm{RT}$ / $\mathrm{RC}$ / $w$ 各多少；(b) 紧随的第二个 CNP（ $w$ 未衰减）后 $\mathrm{RC}$ 多少；(c) 证明 $w$ 初值 1 时式 (3) 的更新与 $g$ 无关。校验：(a) 50 / 25 / 1；(b) 12.5 GB/s；(c) $(1-g) \times 1 + g = 1$ 恒等，与 §4 走读逐数一致。
2.（计算 · [20]）IB 注入率与 RC 超时（式 (8)(9)）：(a) 4 KB 包、50 GB/s 线速， $t_{\mathrm{tx}}$ 多少 ns； $\mathrm{IPD} = t_{\mathrm{tx}}$ 时注入率多少；(b) `NCCL_IB_TIMEOUT` = 20 对应超时多少秒，是 1 GB 正常传输时间（50 GB/s）的几倍；(c) NDR 4 lane 降级到 1 lane，有效带宽多少 GB/s、 $\beta$ 涨几倍。先预测 (b) 的数量级再算。校验：(a) 81.92 ns、25 GB/s；(b) 4.295 s、约 215×；(c) 12.5 GB/s、4×，与 §5/§6 逐数一致（容差 1%）。
3.（推导 · [30]）证明命题 3（fast recovery 闭式，式 (7)）并应用：(a) 对 $k$ 归纳写出基例（ $k=0$）、归纳步（代入式 (5)）、终止结论；(b) 代入 $`\mathrm{RC}_{0} = 25`$、 $\mathrm{RT} = 50$ 求 $k=5$ 的 $\mathrm{RC}$ 与差距占 $\mathrm{RT}$ 的百分比；(c) 求差距压到 $\mathrm{RT}$ 的 1% 以内所需最小 $k$；(d) 说明 $F=5$ 整定的含义。校验：(b) 49.21875 GB/s、1.56%；(c) $k \ge 6$（ $25/64 = 0.39$ GB/s 低于 0.5）；与 §4 逐数一致。
4.（计算 · [30]）拥塞影响全账（命题 1 + §7 straggler）：(a) 2:1 incast（ $A = 100$、 $C = 50$ GB/s），教学假设 $Q_{\mathrm{th}} = 512$ KB，空队列到 pause 多少 μs；(b) $n = 16$ 单 rail 跨机环、 $M = 1$ GB，每 rank 发送多少 GB、健康 rail 带宽项多少 ms；(c) 一条 rail 降速到 25 GB/s，集合完成时间多少 ms、相对健康翻几倍；(d) 先预测「只慢那条流」对不对，再用同步语义解释。校验：(a) 10.24 μs；(b) 1.875 GB、37.5 ms；(c) 75 ms、2×；(d) 错——max 语义全局拖慢，与 §3/§7 逐数一致。

## 参考

- [Zhu et al., 2015](../references.md)：Congestion Control for Large-Scale RDMA Deployments（key: dcqcn），[B] 老经典无 arXiv，SIGCOMM 2015 pp 523-536，DOI 10.1145/2785956.2787484，PDF 2026-11-25 重取一手——RP/CP/NP 三角色；式 (1)-(4) 逐字（本章式 (3)-(6)， $\alpha$ 译 $w$）； $w$ 初值 1、N=50 μs/K=55 μs/F=5 部署值；无 slow start；timer+byte counter 双触发；RED 式 Kmin/Kmax/Pmax 标记；PFC 局限（unfairness/victim flow/congestion spreading/storm，§2.2 Fig 3/4）；§5 fluid model 整定（§3/§4）
- [Crupnicoff et al., IB QoS/CC whitepaper](../references.md)：Deploying Quality of Service and Congestion Control in InfiniBand-based Data Center Networks（key: ib-cc-whitepaper），[C] 官方规格披露·一手权威，Mellanox Rev 1.0，PDF 2026-11-25 一手——IBTA Congestion Control Annex → CCA 三阶段（FECN→BECN→throttle）逐字；CCT/CCTI/Inter-Packet Delay/CCM；per-port per-VL 阈值 0-15；credit 流控 + SL→VL QoS（§5）
- [IBTA, ibta-spec](../references.md)：IBTA InfiniBand Architecture Specification（key: ibta-spec），[C] 官方规格披露，2026-11-25——Vol 1/2 Release 2.1；CC Annex 编号公开源未给（本章引「IBTA Congestion Control Annex」不标号）；链路层检错重传机制；全文 member-gated，机制经 whitepaper + verbs manual 互证（§5/§6）
- [NVIDIA, networking docs](../references.md)：NVIDIA Networking docs（key: nvidia-networking-docs），[C] 官方规格披露，2026-11-25——MLNX_OFED SWUM 24.10 LTS：ECN 页（RoCEv2 RP/NP sysfs、without dropping packets 逐字）/ Flow Control 页（PFC 802.1Qbb、mlnx_qos、DCBX/LLDP）/ Programming 页（packet pacing/TSO）（§3/§4/§8）
- [linux-rdma, rdma-core](../references.md)：linux-rdma/rdma-core（key: rdma-core），教学参照 + 官方实现，v67.0 master 4b7b208c——man `ibv_modify_qp.3` RTS 属性 TIMEOUT/RETRY_CNT/RNR_RETRY（RC 重传参数语义，recap 9.1 §5）（§6）
- [NVIDIA, nccl](../references.md)：NVIDIA/nccl（key: nccl），教学参照 + 官方实现，2.32.7-1 / 351f33fd fresh tarball——`src/transport/net_ib/connect.cc` `NCCL_IB_TIMEOUT` 默认 20 L26 / `NCCL_IB_RETRY_CNT` 默认 7 L32 / `NCCL_IB_QPS_PER_CONNECTION` L106（§6/§7）
- [NVIDIA, connectx7](../references.md)：ConnectX-7 NDR 400G（key: connectx7），[C] 官方规格披露，2026-11-25——datasheet「Hardware-based congestion control」+「Hardware-based reliable transport」逐字；NDR 4 lane × 100 Gb/s/lane SKU 几何（§6/§7/§8）
- [NVIDIA, connectx8](../references.md)：ConnectX-8 SuperNIC（key: connectx8），[C] 官方规格披露，2026-11-25——「advanced routing and telemetry-based congestion control」逐字（§8）
- 9.1 / 9.2 / 8.1 / 8.2 / 7.2 / 10.1 / 10.2 / 10.3 / 11.3 / training 4.1 / inference 8.3（纯文字指引，不挂链接）：RC/QP 状态机与 IB credit vs PFC（9.1）、GDR 路径与多 NIC 聚合（9.2）、α-β 模型与 ring 带宽项（8.1/8.2，training 4.1 源）、机间 50 GB/s 口径（7.2）、fabric 拓扑与收敛比（10.1）、大规模调优与拓扑感知集合（10.2）、a2a incast（10.3）、链路诊断（11.3）、通信暴露与 KV 传输预算（training 4.1 / inference 8.3）——recap 不重推
