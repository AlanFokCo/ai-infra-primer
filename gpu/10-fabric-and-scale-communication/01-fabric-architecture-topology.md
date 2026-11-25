# 10.1 机间 fabric 架构与拓扑（Fabric Architecture and Topology）

> 144 口的 XDR 交换机两级组网为什么恰好接 10368 张 NIC—— $144^{2}/2$ 这个数怎么从端口几何推出来？收敛比 1:1 到底保证什么、3:1 省了什么又丢了什么？rail-optimized 为什么让每张 GPU 各走一条独立 rail，集合带宽怎么随 NIC 数翻倍？IB 的 SM/LID 与 RoCE 的 ECMP/UDP 熵各怎么把包送到目的？本章交付六块：IB/RoCE fabric 架构（SM/LID-GID/交换机层级/故障恢复）、fat-tree/Clos 收敛比带宽账与规模账（命题 1 全二分条件 + 命题 2 两级规模上界 + 三级 pod 计数）、rail-optimized 拓扑与集合带宽 × NIC 数推导、路由（LID forwarding/ECMP/adaptive routing）、dragonfly/torus 对照与 GB200 NVL72 超节点、多轨多 NIC 工程与子网管理 | 前置：9.1（IB vs RoCE 链路层、LID/GID、credit vs PFC）、9.2（GDR 带宽账与多 NIC 聚合命题 2，recap 源）、9.3（拥塞控制与可靠性）、7.1/7.2（NVLink 域几何、节点出口 400 GB/s、9× 数量级差）、8.2/8.4（ring 带宽项与分层 ring、NCCL channel）；下游 10.2（拓扑感知集合与 SHARP）、10.3（a2a 拥塞）、10.4（点对点拓扑映射） | 锚点：Al-Fares fat-tree（[Al-Fares et al., 2008](../references.md)，[B] 老经典无 arXiv，SIGCOMM 2008 pp 63-74，元数据 Crossref 核实、公式独立重推）、Quantum 交换机（[NVIDIA, quantum-switch](../references.md)，[C] 官方规格披露·一手权威，Quantum-2/XDR 规格 + 10368 NICs 逐字）、IB cluster topology（[NVIDIA, ib-cluster-topology](../references.md)，[C]，rail-optimized 逐字）、NCCL 2.12 blog（[NVIDIA, nccl-all2all-blog](../references.md)，[C] 开放技术博客，rail/PXN）、IBTA 规范（[IBTA, ibta-spec](../references.md)，[C]，SM 路由+QoS）、NVIDIA networking docs（[NVIDIA, networking docs](../references.md)，[C]，OFED RoCE/OpenSM/UFM）、NCCL docs + repo（[NVIDIA, nccl-docs](../references.md) / [nccl](../references.md)，[C] / 教学参照，AR/HCA/RAIL_POLICY）、ConnectX-7/8（[NVIDIA, connectx7](../references.md) / [connectx8](../references.md)，[C]）、GB200 NVL72（[NVIDIA, gb200-nvl72](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 三个问题与本章口径

一台 144 口 800G 交换机，两级组网最多接多少张 NIC？官方手册写 10368——这个数为什么是 $144^{2}/2$ 而不是 $144^{2}$？收敛比 1:1 的「无阻塞」到底承诺什么，3:1 的超售又拿什么换什么？8 张 NIC 的节点，跨机 all-reduce 带宽凭什么乘 8？本章回答三个问题：fabric 由什么管理、寻址与转发怎么分层（§2）？拓扑族怎么决定收敛比与规模上界、rail-optimized 怎么把 NIC 数变成集合带宽（§3-4）？路由怎么选路、超节点与 fabric 怎么分工（§5-7）？

**本章口径**（章脊锁定；本章为多 SKU 对比章，口径在此一次声明）：交换机两代——**Quantum-2（NDR，64×400G 口）**与 **Quantum-X800（XDR，Q3400-RA，144×800G 口）**；NIC 两代——**ConnectX-7 NDR 400G = 50 GB/s 单向**与 **ConnectX-8 800G = 100 GB/s 单向**（ $10^{9}$ 口径，recap 9.1 §7 / 9.2 §4）；节点形态——**8-GPU rail-optimized 节点**（每 GPU 一张 NIC，各挂独立 PCIe x16）；贯穿实例 $M = 1$ GB = $10^{9}$ B，通信精度 bf16（梯度口径沿 recap 源）。超节点对照口径 GB200 NVL72（72-GPU NVLink 域）。拓扑推导取**图论结构口径**（端口计数与割容量，教学推导），不引特定厂商布线实现；dragonfly/torus 对照只作结构对比，无厂商数值断言。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**）； $M$ = 被传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s）。局部符号： $k$ = 交换机端口数（radix，**本章局部声明，非 GEMM 归约维 $K$**）； $k_{\mathrm{down}}$ / $k_{\mathrm{up}}$ = leaf 下行（接 NIC）/上行（接 spine）端口数； $\rho$ = 收敛比（oversubscription ratio）； $n_{\mathrm{leaf}}$ / $n_{\mathrm{NIC}}$ = leaf 交换机数 / NIC 数（ $n$ 下标变体，沿 9.2）； $n_{\mathrm{tier}}$ = 交换机层级数； $r$ = 单端口线速（GB/s）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 T 下标。

**边界声明**：本章讲 **fabric 架构与拓扑的硬件/结构机制**（代码 scope = 机制片段级：NCCL net_ib 路由与 rail 策略 env 锚，非模块全量走读）。拓扑感知集合构建/SHARP/万卡实测归 10.2，a2a 拥塞 transport 展开归 10.3，点对点拓扑映射归 10.4；GDR 带宽账与多 NIC 聚合的 PCIe 前提 recap 9.2 命题 2 不重推，ring 消息量与带宽项 recap 8.2（training 4.1 源）不重推，拥塞控制机制 recap 9.3 不重推。平台编排（K8S 调度/容器网络）归 cluster 维度。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. IB/RoCE fabric 架构：SM、LID/GID、交换机层级

跨机 fabric 靠谁把「哪台 NIC 在哪、包往哪走」配好？InfiniBand 的答案是一个集中控制面——subnet manager；RoCE 的答案是把这两件事交给 IP 路由与以太网配置。两套架构的管理模型不同，数据面都是交换机硬件线速转发。

**定义**（subnet manager）：所谓 SM（Subnet Manager，子网管理器），是指 IB 子网内负责发现拓扑、分配 LID、计算并下发转发表、配置 QoS 的集中管理实体（IBTA 规范口径：subnet manager 提供路由 + QoS，[IBTA, ibta-spec](../references.md)，[C]）。开源实现 OpenSM（linux-rdma/opensm），NVIDIA 管理面 UFM（[NVIDIA, networking docs](../references.md)，[C]）。SM 是**控制面软件**：拓扑发现（sweep 逐跳探测）、LID 分配、每交换机「目的 LID → 出端口」转发表计算、QoS 策略（SL→VL 映射）都由它算好下发；数据面转发是交换机硬件查表，SM 不在数据路径上。正例：链路故障后 SM 重新 sweep、重算路由下发，数据面按新表转发；边界反例：SM 宕机时已下发的转发表仍工作（数据面不依赖 SM 在线），但拓扑变化无人响应——所以生产部署配备用 SM。

**LID 与 GID**（recap 9.1 §6）：LID（Local Identifier）是 SM 分配的子网内本地标识，IB 转发按 LID 查表；GID（Global Identifier）是全局标识，跨子网寻址用。RoCE 的口径逐字（OFED RoCE 页）：「The presence of a Subnet Manager (SM) is not required in the fabric」、「LID ... is not set for a port and is displayed as zero」——**RoCE 无 SM、LID = 0，寻址走 GID/IP**，路由交给 IP 网络（§5 ECMP）。IB 另有 APM（Automatic Path Migration，自动路径迁移）做故障路径切换，RoCE 不支持（recap 9.1 §6）。

**交换机层级**：leaf（ToR，接服务器 NIC）→ spine（接 leaf）→ core（更多层级时）。两级 leaf-spine 的端口几何（1:1 口径，每 leaf 半口下行半口上行）：

```
        spine_1       spine_2      ...    spine_{k/2}     （k/2 台，每台 k 口接全部 leaf）
        / | \         / | \               / | \
   leaf_1   leaf_2  ...  leaf_k                          （k 台，每台 k/2 上行接全部 spine）
   / | \    / | \        / | \
 NIC NIC NIC ...                                          （每台 k/2 下行接 NIC）
  NIC 总数 = k 台 leaf × k/2 口 = k²/2（1:1，§3 命题 2）
```

**故障恢复与子网管理工程**：IB 侧，bring-up 流程按 topology 规划 → topology file → SM 配置 → 确认（[NVIDIA, ib-cluster-topology](../references.md)，[C]）；运行中链路故障由 SM 重算路由（IB APM 可自动迁移路径），UFM 监控告警。RoCE 侧无 SM，无损与优先级靠 PFC/DCBX/LLDP 配置（recap 9.3 §3/§8），路由靠 IP/ECMP——**管理模型的差异是 IB「集中控制面 + 原生无损」vs RoCE「复用以太网基础设施 + 配置造无损」**（recap 9.1 §6 选型结论的架构侧展开）。

## 3. fat-tree/Clos：收敛比带宽账与规模账

10368 这个数怎么来的？先立拓扑定义，再推收敛比条件（命题 1）与规模上界（命题 2），最后给三级与一般 pattern。fat-tree 的出处锚是 Al-Fares SIGCOMM 2008（[Al-Fares et al., 2008](../references.md)，[B] 老经典无 arXiv，pp 63-74，Crossref 元数据核实；PDF 多通道不可达残留，k-port 公式按本书骨架独立重推）。

**定义**（fat-tree/Clos 与收敛比）：所谓 fat-tree（胖树），是指上行带宽不低于下行带宽的多层交换树——「胖」在每层向上不收敛，任意割的容量不小于该割一侧的注入需求；Clos 是其 leaf-spine 折叠构造（叶脊两级或多级）。所谓收敛比（oversubscription ratio），是指 leaf 下行聚合带宽与上行聚合带宽之比：

$$
\rho = \frac{k_{\mathrm{down}}}{k_{\mathrm{up}}} \qquad (1)
$$

**符号**： $k_{\mathrm{down}}$ / $k_{\mathrm{up}}$ = leaf 下行/上行端口数（同速率端口，带宽比即端口数比）； $\rho = 1$ 即 1:1 无收敛， $\rho = 3$ 即 3:1 超售。代入：1:1 时 $k_{\mathrm{down}} = k_{\mathrm{up}} = k/2$（72/72，k=144）；3:1 时 48/16（k=64）。

**定义**（全二分带宽）：所谓二分带宽（bisection bandwidth），是指把 NIC 集合分成数量相等的两半、所有跨割链路容量之和的最小值（对所有二分取 min）；**全二分带宽**指该值等于 $n_{\mathrm{NIC}}/2 \times r$——一半 NIC 同时向另一半线速发送时网络不卡。（口径：二分带宽是**聚合割容量**口径，不是单对峰值——单对峰值与聚合的区分 recap 7.1 §5 提点。）正例：1:1 两级 fat-tree 任意对半割容量 = 一半 NIC 的注入总和；边界反例：3:1 时割容量只有注入需求的 1/3，全二分不成立。

**命题 1**（1:1 全二分条件）：两级 leaf-spine，k 口交换机，每 leaf $k_{\mathrm{down}}$ 下行 + $k_{\mathrm{up}}$ 上行（和为 k），每台 spine 接全部 leaf。则全二分带宽成立当且仅当 $k_{\mathrm{up}} \ge k_{\mathrm{down}}$（ $\rho \le 1$）；且在排列流量（每 NIC 向唯一目的 NIC 线速发送）下，1:1 可把全部流线速路由（可重排无阻塞）：

$$
\frac{n_{\mathrm{leaf}}}{2} \cdot k_{\mathrm{up}} \cdot r \ge \frac{n_{\mathrm{leaf}}}{2} \cdot k_{\mathrm{down}} \cdot r \Leftrightarrow k_{\mathrm{up}} \ge k_{\mathrm{down}} \qquad (2)
$$

**证明**（路线：必要性用割容量账，充分性用二部图边着色，三步）。其一，**必要性**：把 leaf 对半分（NIC 划分对齐 leaf），令一半的全部 NIC 线速发往另一半——需求 $(n_{\mathrm{leaf}}/2) \cdot k_{\mathrm{down}} \cdot r$；跨割流量必经 leaf 上行口，容量 $(n_{\mathrm{leaf}}/2) \cdot k_{\mathrm{up}} \cdot r$。全二分要求容量不低于需求，即式 (2)；若 $k_{\mathrm{up}}$ 低于 $k_{\mathrm{down}}$，该割就是瓶颈，二分带宽 = 容量，低于全值。其二，**充分性**（排列流量口径）：构造 leaf 间需求二部多重图——每条流是一条边，源/目的 leaf 是两端点；排列流量下每 leaf 发出与接收的流数都不超过 $k_{\mathrm{down}}$，故最大度 $\Delta \le k_{\mathrm{down}} = k/2$（1:1）。二部多重图的边色数等于最大度（König 定理，图论事实）——用 $k/2$ 种颜色给边着色，每种颜色对应一台 spine。其三，**逐链路核验**：每 leaf 每种颜色至多一条流 → leaf→spine 链路至多一条流（线速）；每 spine 到每目的 leaf 每种颜色至多一条流 → spine→leaf 链路至多一条流。全部流线速接通。证毕。

全数字代入（k=144 XDR，1:1）： $k_{\mathrm{down}} = k_{\mathrm{up}} = 72$，全二分成立；代入 3:1（k=64，48/16）：割容量只有需求的 $16/48 = 1/3$——超售省的是上行端口与 spine 台数，丢的是全二分带宽。

> **坑**：把 1:1 读成「严格无阻塞」。触发条件：以为任意时刻新增一条流都不影响已有流；现象：在线流量变化时误判「1:1 承诺失效」。事实：1:1（中平面数 = 下行口数）是**可重排无阻塞**（rearrangeably non-blocking）——对整个流量矩阵存在线速路由，但新流可能要重排已有流的路径；严格无阻塞 Clos 要求中平面数不少于 $2n-1$（图论口径）。规避：谈「无阻塞」先注口径（可重排 vs 严格），本书 1:1 均指可重排。

**命题 2**（两级 1:1 规模上界）：两级 1:1 leaf-spine（k 口交换机）可接 NIC 数满足

$$
n_{\mathrm{NIC}} \le \frac{k^{2}}{2} \qquad (3)
$$

且等号可达。**证明**（路线：端口计数——spine 台数 = 上行口数、每台 spine 的 k 口限制 leaf 台数，两步）——完整证明留作练习 3。代入 k=144： $144^{2}/2 = 10368$——XDR HW manual 逐字「With this very high radix, a two-level fat tree topology can connect up to **10,368** network interface cards (NICs) at lowest latency」（[NVIDIA, quantum-switch](../references.md)，[C]），官方数与推导逐数闭合。代入 k=64（Quantum-2）： $64^{2}/2 = 2048$。

**三级 fat-tree**（Al-Fares 构造，pod 计数）：k 个 pod，每 pod $k/2$ 台 leaf + $k/2$ 台 aggregation；core 层 $(k/2)^{2}$ 台。计数链：每 pod 的 agg 下行口 $k/2$ 接满 pod 内 leaf → 每 pod $k/2$ 台 leaf；core 每台 k 口、每口接一个 pod 的一台 agg → pod 数 = k；每 leaf $k/2$ 下行口接 NIC。合计：

$$
n_{\mathrm{NIC}} = k \cdot \frac{k}{2} \cdot \frac{k}{2} = \frac{k^{3}}{4} \qquad (4)
$$

代入 k=64： $64^{3}/4 = 65536$ 张 NIC——两级 2048 的 32 倍，代价是多一层交换机与跳数。**一般 pattern**（pod 递归计数）：每加一层，规模乘 $k/2$（半口下行扇出）， $n_{\mathrm{tier}}$ 层 1:1 fat-tree 最大 NIC 数

$$
n_{\mathrm{NIC,max}} = \frac{k^{n_{\mathrm{tier}}}}{2^{n_{\mathrm{tier}} - 1}} \qquad (5)
$$

$n_{\mathrm{tier}} = 2$ 给 $k^{2}/2$、 $n_{\mathrm{tier}} = 3$ 给 $k^{3}/4$，与式 (3)(4) 逐数闭合；一般归纳沿 pod 递归（每层把 $k/2$ 台下层交换机编成 pod、pod 数 k），本章交付两个已验证 case 与 pattern 式，完整归纳不展开。**规模账的读法**：radix k 是规模的杠杆——k 翻倍，两级规模翻 4 倍、三级翻 8 倍；这就是 XDR 把 radix 从 64 推到 144 的意义（两层接 10368 NIC，保持最低跳数）。

**超售的权衡账**（3:1 例，k=64）： $k_{\mathrm{down}} = 48$、 $k_{\mathrm{up}} = 16$，spine 16 台每台 64 口接满 64 台 leaf → NIC 数 $64 \times 48 = 3072$（比 1:1 的 2048 多 50%），但二分带宽降到 1/3。**超售拿「每 NIC 平均跨割带宽」换「NIC 总数/端口成本」**——集合通信是全局流量（all-reduce 跨割流量大），超售 fabric 上集合带宽被割容量封顶；这正是 AI factory 推荐 1:1 fat-tree 的带宽账依据（[NVIDIA, ib-cluster-topology](../references.md)，[C]：「Fat-Tree is NVIDIA's recommended topology」逐字）。

## 4. rail-optimized：每 GPU 一条独立 rail

8 张 NIC 的节点，跨机 all-reduce 带宽怎么乘 8？答案分两半：PCIe 侧的独立路径前提（recap 9.2 命题 2）与网络侧的 rail 对齐拓扑——本节交付后者。

**定义**（rail-optimized）：所谓 rail-optimized（轨优化），是指多 NIC GPU 节点把每张 GPU 的「rail」（IB 网络接口）接到**不同的**一级（leaf）交换机上的组网设计。官方逐字：「Rail-optimized design means a GPU node with multiple interfaces will put each GPU "rail" (IB network interface) onto a different first level (LEAF) switch」（[NVIDIA, ib-cluster-topology](../references.md)，[C]）；NCCL 2.12 blog 的等价描述：「NIC-0 of each DGX system is connected to the same leaf switch (L0), NIC-1s ... (L1), and so on」（[NVIDIA, nccl-all2all-blog](../references.md)，[C] 开放技术博客）。

```
node A                              node B
GPU0 ─ NIC0 ──── leaf(rail 0) ──── NIC0 ─ GPU0
GPU1 ─ NIC1 ──── leaf(rail 1) ──── NIC1 ─ GPU1
GPU2 ─ NIC2 ──── leaf(rail 2) ──── NIC2 ─ GPU2
GPU3 ─ NIC3 ──── leaf(rail 3) ──── NIC3 ─ GPU3
（示意 4 rail；8-GPU 节点 8 rail。节点内 GPU 经 NVSwitch 全互连；
  同号 rail 跨机只经本 rail 的 leaf，1 交换跳；各 rail 是独立 leaf-spine 平面）
```

**跳数账**（rail 对齐的延迟收益）：同号 GPU 跨机（GPU_i 对 GPU_i）走各自 rail 的 leaf，**1 交换跳**；跨 rail（GPU_i 对 GPU_j）若直接走源 rail 出口，要 leaf_i → spine → leaf_j **3 跳**。rail-optimized 配 NVSwitch 给出第三条路：官方逐字「utilize their internal NVSwitch path to talk across a NIC that is just one switch hop away」——源 GPU 先经节点内 NVSwitch 把数据搬到**目的 rail 对应的本地 GPU**，再从那张 NIC 出口，到目的节点同 rail NIC 仍只 1 交换跳（NCCL 的 PXN 路径，blog 逐字「first move data on a GPU on the same rail as the destination, then send it to the destination without crossing rails」；工程走读归 10.2）。**rail 对齐把「跨 rail 3 跳」变成「域内搬运 + 1 跳」**，域内 NVLink 带宽远高于机间（9×，recap 7.2 §7），多搬一跳域内不亏。

**集合带宽 × NIC 数推导**（本章关键账）。路线：各 rail 是独立 leaf-spine 平面 → 跨机环按 rail 映射、互不争抢 → 聚合带宽随 NIC 数线性（PCIe 前提 recap 9.2 命题 2），三步。其一，rail i 上的跨机流量只走 rail i 的 leaf/spine，与其他 rail 物理隔离——blog 逐字「maximize all-reduce performance while minimizing network interference between flows」。其二，NCCL 把 channel/环按 rail 映射（每 rail 一条或多条 channel，8.4 recap；拓扑感知构建归 10.2），消息 $M$ 拆到 $n_{\mathrm{NIC}}$ 条 rail 并行。其三，节点出口聚合（recap 9.2 式 (2)，独立 PCIe 路径前提）：

$$
\mathrm{BW}_{\mathrm{agg}} = n_{\mathrm{NIC}} \times \min(\mathrm{BW}_{\mathrm{NIC}}, \mathrm{BW}_{\mathrm{PCIe}}) = n_{\mathrm{NIC}} \times \mathrm{BW}_{\mathrm{NIC}} \quad (\text{NIC 线速限 regime}) \qquad (6)
$$

代入（8-GPU 节点，每 GPU 一张单口 CX7，NIC 线速限 50 GB/s）： $8 \times 50 = 400$ GB/s 节点出口——与 7.2 §7 / 9.2 §5 逐数闭合；对节点内 NVLink 注入 3600 GB/s 比值 **9×**（recap）。**all-reduce 带宽项**（节点粒度环口径：节点内分层聚合 recap 8.2 §7，机间段用节点出口聚合带宽；每节点视为一个超 rank）：

$$
T_{\mathrm{bw}} = 2 \thinspace \frac{n-1}{n} \cdot M \cdot \frac{\beta}{n_{\mathrm{NIC}}} \qquad (7)
$$

**符号**： $n$ = 节点数（节点粒度环的 rank 数）； $`\beta = 1/\mathrm{BW}_{\mathrm{NIC}}`$ = 单 NIC 每字节时间； $n_{\mathrm{NIC}}$ = 每节点 rail 数。代入 $n = 16$ 节点、 $M = 1$ GB、 $n_{\mathrm{NIC}} = 8$：每节点发送 $2 \times 15/16 \times 1 = 1.875$ GB，单 NIC 口径带宽项 37.5 ms，8 rail 聚合 $1.875/400 = 4.6875$ ms——**恰好 ÷8**。这就是「rail-optimized 集合带宽 × NIC 数」的定量形式：不是单条流变快，而是 $n_{\mathrm{NIC}}$ 条独立平面并行承载消息分片。

**成本侧**（blog 逐字）：「reduce the cost of the network by having lighter connections between rails」——rail 间流量被 PXN 压到域内，rail 之间的 spine 连接可以减轻（超售），把端口预算省下来。**rail-optimized 是「拓扑对齐换带宽与成本」的设计**：对齐则每 rail 独立、聚合线性、跨 rail 走域内；不对齐则跨 rail 流量挤 spine、聚合被共享路径封顶（9.2 命题 2 边界反例的网络侧对应）。

## 5. 路由：LID forwarding、ECMP 与 adaptive routing

包怎么从源 NIC 走到目的 NIC？IB 靠 SM 算好的 LID 转发表，RoCE 靠 IP 路由 + ECMP 哈希；两者的「选路自由度」不同，adaptive routing 是在其上叠加的动态选路。

**IB LID forwarding**：SM 为每台交换机算「目的 LID → 出端口」表（§2），数据面逐包查表转发。路径由 SM 集中决定——好处是全局最优可配（路由/QoS 一体），代价是拓扑变化要等 SM 重算（故障恢复 §2）。

**RoCE ECMP 与 UDP 熵**：RoCEv2 把 IB 传输封装进 UDP/IP，等价流量走 ECMP（Equal-Cost Multi-Path，按五元组哈希在等价路径间分流）。**UDP 源端口是逐 QP 的熵源**：不同 QP 配不同源端口 → 哈希到不同上行路径，多 QP 流量得以摊开（NCCL 多 channel × 多 QP 正是利用这一点，8.4 recap）。静态哈希的坑是**碰撞极化**：哈希不均时部分上行口热点、其余空闲。全数字账（**教学推导口径**，均匀哈希假设）：8 条流哈希到 8 条上行路径，全部错开的概率 $8!/8^{8} = 40320/16777216 \approx 0.0024$——**碰撞几乎必然**，期望意义下每路径 1 条流、但方差使最热路径显著超载。流数越多越接近均匀（大数定律），少量大象流则极易极化。

**adaptive routing**（AR）：按出口队列深度动态选路（逐包或逐 flowlet），绕开热点——Quantum-X800 官方规格含「adaptive routing, telemetry-based congestion control」（[NVIDIA, quantum-switch](../references.md)，[C]）。NCCL 侧逐字（[NVIDIA, nccl-docs](../references.md)，[C]）：`NCCL_IB_ADAPTIVE_ROUTING`「Enabled (1) by default on IB networks. Disabled (0) by default on RoCE」（repo 锚 `src/transport/net_ib/init.cc` L12，[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，2.32.7-1 / 351f33fd）。**AR 与 RC 有序语义的交互**：AR 下同一 QP 的包可能走不同路径、乱序到达，而 RC 承诺有序交付（recap 9.1 §5）——接收侧须按 PSN 重排序后再交付（机制推断口径：有序语义 + 多路径 ⇒ 重排序缓冲）。这是 AR 的传输层代价，也是 RoCE 默认关 AR 的原因之一（以太网侧乱序处理与 PFC 交互更复杂）。

| 维度 | InfiniBand | RoCEv2 |
| --- | --- | --- |
| 转发依据 | LID 表（SM 集中计算） | IP 路由 + ECMP 五元组哈希 |
| 路径熵 | SM 路由配置 / AR | UDP 源端口（逐 QP） |
| 动态选路 | AR（Quantum-X800，NCCL 默认开） | 交换机 AR（NCCL 默认关） |
| 有序保证 | RC PSN（AR 下接收侧重排） | RC PSN（同左） |
| 故障响应 | SM 重算 + APM | IP 路由收敛（无 APM，recap 9.1） |

## 6. 拓扑对照与超节点：dragonfly/torus、GB200 NVL72

fat-tree 之外还有什么拓扑族，超节点又把 fabric 的边界推到哪？本节作结构对照（**图论结构口径，教学推导**，不引厂商实现数值），再给 NVL72 的域/fabric 分工账。

**torus（环面）**：直连网络——每节点（或每交换机）只与固定邻居相连，k-ary n-cube 的节点按 n 维环面寻址。二维 $k \times k$ torus 的二分带宽：沿一维对半切，割断该行方向的 $k$ 条边 + 环绕回来的 $k$ 条边，共 $2k$ 条链路：

$$
\mathrm{Bisection}_{\mathrm{torus2D}} = 2k \text{ 条链路}, \qquad \text{每节点摊} = \frac{2k}{k^{2}} = \frac{2}{k} \qquad (8)
$$

代入 k=32（1024 节点）：torus 二分 64 条链路，同规模 1:1 fat-tree 二分 $1024/2 = 512$ 条——**8× 差**；每节点摊 torus $2/32 = 1/16$ 条 vs fat-tree 1/2 条。**torus 的每节点二分带宽随规模收缩（ $2/k$），fat-tree 恒定（1/2）**——这是全局集合通信（all-reduce 跨割流量大）选 fat-tree 不选 torus 的带宽账依据。torus 的补偿是邻居直连的低延迟与布线局部性（适合邻居通信为主的负载）；维度序路由简单但跨维流量绕路。

**dragonfly**：分层直连——组内交换机全互连（局部高带宽），组间靠全局链路（每交换机少量远距口）。结构动机是拿「组内全连接 + 组间稀疏全局」折中 fat-tree 的全局无阻塞与 torus 的布线成本；代价是对抗性流量模式下全局链路拥塞，须配自适应路由（结构对照口径，无厂商数值断言）。对 AI 集合通信：dragonfly/torus 的全局带宽都不及同成本 1:1 fat-tree，故 AI factory 推荐 fat-tree + rail-optimized（§3/§4 官方逐字）。

**GB200 NVL72 超节点 vs 传统 fat-tree**（recap 7.1，不重推域几何）：所谓超节点（supernode），是指把 NVLink 域扩到机架尺度、域内通信不经 NIC/fabric 的形态——NVL72 把 72 张 Blackwell GPU 经 9 个 NVLink switch tray 全互连成一个域（18 compute tray + 9 switch tray，聚合 130 TB/s 双向，[NVIDIA, gb200-nvl72](../references.md)，[C]）。**scale-up（域内扩）vs scale-out（fabric 扩）的分工账**：

| 维度 | HGX H100 + IB fat-tree | GB200 NVL72 + XDR fat-tree |
| --- | --- | --- |
| NVLink 域规模 | 8 GPU | 72 GPU（9×） |
| 每 GPU 域内带宽（单向） | 450 GB/s（NVLink4） | 900 GB/s（NVLink5，1.8 TB/s 双向 ÷2） |
| 每 GPU 跨 fabric 带宽（单向） | 50 GB/s（CX7） | 100 GB/s（CX8） |
| 域内/跨 fabric 比值 | 9× | 9×（900/100） |
| 分层环跨域字节（recap 8.2 §7） | $M/8$ | $M/72$ |

代入式 (9) 的比值账（NVLink5 单向 900 / CX8 单向 100）：

$$
\frac{\mathrm{BW}_{\mathrm{NVLink,unidir}}}{\mathrm{BW}_{\mathrm{NIC,unidir}}} = \frac{900}{100} = 9 \qquad (9)
$$

**超节点没有消灭 fabric，而是把 fabric 的边界外推**：72 GPU 以内走 NVLink（域内），以外仍走 IB fabric（NVL72 跨机架经 ConnectX-8 SuperNIC；Quantum-X800 平台口径每 GPU 至多 1.6 Tb/s = 2×CX8，[NVIDIA, connectx8](../references.md)，[C]）。规模账：两级 XDR fat-tree 接 10368 NIC，按每 GPU 一张 NIC 即 10368 GPU = **144 个 NVL72 机架**（ $10368/72 = 144$）。域扩大 9× 让分层环的跨域字节从 $M/8$ 压到 $M/72$（recap 8.2 §7 结构）——慢链路（fabric）承载的流量再降 9×，这是超节点对集合通信的算术价值。NVLink5 最大域 576 GPU（recap 7.1 §5），再往外仍是 fabric 的 scope。

## 7. 多轨多 NIC 工程：rail 对齐与子网管理

rail 对齐在软件侧怎么落地？三个落点：GPU-NIC 的 PCIe 亲和、NCCL 的 rail 策略、子网管理的配置流。

**GPU-NIC rail 对齐**（recap 9.2，不重推）：每 GPU 的 NIC 挂本地 PCIe switch/root complex 下（同上游 root complex，GDR 三道门之一），各 NIC 独立 x16——9.2 命题 2「独立 PCIe 路径」前提的物理落点。rail 编号对齐：GPU_i 配 NIC_i、NIC_i 接 leaf_i（§4），NCCL 拓扑检测据此把 channel 映射到 rail（`src/graph/topo.cc`/`paths.cc`，10.2 走读）。

**NCCL rail 策略**（[NVIDIA, nccl-docs](../references.md)，[C]）：`NCCL_IB_HCA` 过滤语法 `<hca>[:<port>[:<rail>[:<plane>]]]`——显式指定 rail/plane；`NCCL_IB_RAIL_POLICY`（2.30.5+）对 ConnectX SuperNIC 自动分配 rail/plane。多 channel 跨 NIC（MAXCHANNELS 64，8.4 recap）把消息拆到各 rail；SHARP 流式聚合同样要求每 HCA rail 一条聚合流、leaf 同 rail（10.2）。**rail 对齐是「拓扑—PCIe—通信库」三层一致的设计**：物理接线（§4）、PCIe 亲和（9.2）、channel 映射（10.2）任何一层错位，聚合带宽账（式 (6)）就打折。

**子网管理配置流**：IB 侧按 topology 规划 → PTP excel → topology file → SM 下发 → 确认（[NVIDIA, ib-cluster-topology](../references.md)，[C]）；UFM 承担运行监控。RoCE 侧无 SM，配置流是 PFC 优先级/DCBX 协商 + ECN 阈值 + IP 路由（recap 9.3 §8）——两套管理模型（§2）落到工程上是「topology file 驱动」vs「逐设备 QoS 配置」的差异。大规模 fabric 的调优实测（多作业干扰、万卡 all-reduce）归 10.2。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| IB fabric = SM 集中控制面（发现/LID/转发表/QoS）+ 硬件数据面；RoCE 无 SM、LID=0、GID/IP 寻址；故障恢复 SM 重算 + APM（RoCE 无） | IBTA + OFED 逐字 | §2 |
| 收敛比 $\rho = k_{\mathrm{down}}/k_{\mathrm{up}}$；全二分带宽 iff $k_{\mathrm{up}} \ge k_{\mathrm{down}}$（割容量账 + König 边着色）；1:1 是可重排无阻塞，非严格（严格需 $2n-1$ 中平面） | 命题 1、式 (1)(2) | §3、练习 2 |
| 两级 1:1 规模上界 $k^{2}/2$ 可达：k=144 → 10368（XDR 官方逐字闭合）、k=64 → 2048；三级 pod 计数 $k^{3}/4$ = 65536；pattern $k^{n_{\mathrm{tier}}}/2^{n_{\mathrm{tier}}-1}$ | 命题 2、式 (3)(4)(5) | §3、练习 1/3/4 |
| 超售 3:1（48/16）：NIC 数 3072（+50%）但二分带宽 ÷3——拿每 NIC 跨割带宽换端口成本；AI factory 推荐 1:1 fat-tree | §3 权衡账 | §3、练习 2 |
| rail-optimized = 每 GPU rail 接不同 leaf；同号跨机 1 跳、跨 rail 经 NVSwitch+PXN 仍 1 跳（否则 3 跳）；各 rail 独立平面 → 聚合 $`n_{\mathrm{NIC}} \times \mathrm{BW}_{\mathrm{NIC}}`$ = 400 GB/s、带宽项 ÷8（37.5→4.69 ms） | 式 (6)(7)，官方逐字 | §4、练习 2 |
| IB LID 表转发（SM 算）vs RoCE ECMP 五元组哈希（UDP 源端口逐 QP 熵）；8 流 8 路径全错开概率 0.0024；AR 动态选路（IB NCCL 默认开/RoCE 关），RC 有序须 PSN 重排 | §5 对照表 | §5 |
| torus 2D 二分 $2k$ 链路、每节点 $2/k$ 随规模收缩（k=32：64 vs fat-tree 512，8×）；dragonfly 组内全连接+组间全局链（结构对照）；全局集合选 fat-tree | 式 (8) | §6、练习 4 |
| NVL72 = 72-GPU 域（130 TB/s 双向），域内/跨 fabric 9×（900/100）；超节点外推 fabric 边界不消灭 fabric：10368 NIC = 144 机架；跨域字节 $M/8$ → $M/72$ | 式 (9)、recap 7.1/8.2 | §6 |
| rail 对齐三层一致：物理接线（每 rail 独立 leaf）+ PCIe 亲和（独立 x16，9.2）+ NCCL channel 映射（IB_HCA rail/plane、RAIL_POLICY）；IB topology file 驱动 vs RoCE 逐设备 QoS | §7 | §7、10.2 |

## 练习

1.（计算 · [10]）规模账代入（式 (3)(4)(5)）：(a) k=144 两级 1:1 最多接多少 NIC；(b) k=64 两级 1:1 多少；(c) k=64 三级多少；(d) 式 (5) 在 $n_{\mathrm{tier}} = 2/3$ 时退化为哪两式。校验：(a) 10368，与 XDR HW manual「10,368 NICs」逐字一致；(b) 2048；(c) 65536；(d) 式 (3)(4)。
2.（计算 · [20]）收敛比与 rail 带宽账（式 (1)(2)(6)(7)）：(a) k=64、 $k_{\mathrm{down}} = 48$ / $k_{\mathrm{up}} = 16$， $\rho$ 多少、NIC 总数多少、二分带宽是全值的几分之几；(b) 8-GPU 节点每 GPU 一张 CX7（NIC 线速限），出口聚合多少 GB/s、对 NVLink 注入 3600 GB/s 比值；(c) $n = 16$ 节点、 $M = 1$ GB 节点粒度环，单 NIC 口径带宽项多少 ms、8 rail 聚合多少 ms。先预测 (c) 的倍数再算。校验：(a) 3:1、3072、1/3；(b) 400 GB/s、9×（recap 7.2 §7）；(c) 37.5 ms、4.6875 ms（÷8），与 §3/§4 逐数一致（容差 1%）。
3.（推导 · [30]）证明命题 2（两级 1:1 规模上界，式 (3)）：(a) 由 1:1 得每 leaf 下行/上行各 k/2；(b) 证 spine 台数 = $k_{\mathrm{up}} = k/2$（每 leaf 上行接互异 spine），每台 spine k 口接互异 leaf → leaf 台数不超过 k；(c) 推出 $n_{\mathrm{NIC}} \le k \cdot k/2 = k^{2}/2$，并给等号可达的构造（leaf k 台、spine k/2 台、每 spine 接全部 leaf）；(d) 指出 1:1 条件在哪一步用到、超售时上界怎么变。校验：(c) k=144 → 10368、k=64 → 2048，与 §3 逐数一致；(d) 超售 $k_{\mathrm{down}}$ 大于 $k/2$ 时上界 $k \cdot k_{\mathrm{down}}$（但二分带宽降）。
4.（推导 · [30]）三级 pod 计数与 torus 对照（式 (4)(8)）：(a) 写出三级 fat-tree 计数链（pod 数 k × 每 pod leaf k/2 × 每 leaf 下行 k/2），代入 k=64；(b) 验证式 (5) 在 $n_{\mathrm{tier}} = 3$ 与 (a) 一致；(c) 推导二维 $k \times k$ torus 二分 = 2k 条链路（对半切 + 环绕边），代入 k=32 并与同规模 1:1 fat-tree 二分对比；(d) 用每节点摊带宽解释为什么全局集合通信选 fat-tree。校验：(a) 65536；(c) torus 64 链路 vs fat-tree 512 链路（8×）、每节点 1/16 vs 1/2，与 §6 逐数一致。

## 参考

- [Al-Fares et al., 2008](../references.md)：A Scalable, Commodity Data Center Network Architecture（key: al-fares），[B] 老经典无 arXiv，SIGCOMM 2008 pp 63-74，DOI 10.1145/1402958.1402967，Crossref 2026-11-25 再核——fat-tree/Clos 无阻塞收敛出处锚；PDF 可达出处残留（ACM 403 / UCSD TLS / CCR 404），k-port 公式按本书骨架独立重推（§3）
- [NVIDIA, quantum-switch](../references.md)：NVIDIA Quantum InfiniBand 交换机（key: quantum-switch），[C] 官方规格披露·一手权威，2026-11-25——Quantum-2：64×400G / 128×200G、32 OSFP、1U、51.2 Tb/s **双向**（=64×400G×2）、SHARPv3；Quantum-X800 Q3400-RA：144×800G、72 OSFP、4U、4×Quantum-3 ASIC、200G/lane SerDes、Switching 115.2 Tbps **单向**（口径差必注）、SHARPv4 + adaptive routing + telemetry CC、「two-level fat tree ... up to 10,368 NICs」逐字（§1/§3/§5）
- [NVIDIA, ib-cluster-topology](../references.md)：IB Cluster Bring-Up Procedure — Setting the InfiniBand Cluster Topology（key: ib-cluster-topology），[C] 官方规格披露，2026-10-22 更新——「Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized」/「each GPU rail ... onto a different first level (LEAF) switch」/「NVSwitch path ... just one switch hop away」逐字；PTP excel → topology file → confirm 工程流（§2/§3/§4/§7）
- [NVIDIA, nccl-all2all-blog](../references.md)：Doubling all2all Performance with NCCL 2.12（key: nccl-all2all-blog），[C] 开放技术博客、非闭源，2022-02-28——rail-optimized「NIC-0 ... same leaf switch (L0)」、PXN「first move data on a GPU on the same rail as the destination」、「lighter connections between rails」逐字（§4；PXN 工程走读归 10.2）
- [IBTA, ibta-spec](../references.md)：IBTA InfiniBand Architecture Specification（key: ibta-spec），[C] 官方规格披露，2026-11-25——Vol 1/2 Release 2.1；subnet manager 提供路由 + QoS；全文 member-gated，SM 机制经 networking docs/OpenSM 互证（§2）
- [NVIDIA, networking docs](../references.md)：NVIDIA Networking docs（key: nvidia-networking-docs），[C] 官方规格披露，2026-11-25——OFED RoCE 页（无 SM / LID=0 / GID / APM 不支持逐字）、OpenSM（linux-rdma/opensm）/ UFM 管理面（§2/§7）
- [NVIDIA, nccl-docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露，2.32.3（滞后 repo 2.32.7 注）——`NCCL_IB_ADAPTIVE_ROUTING` IB 默认开 / RoCE 默认关逐字、`NCCL_IB_HCA` rail/plane 语法、`NCCL_IB_RAIL_POLICY`（2.30.5+）（§5/§7）
- [NVIDIA, nccl](../references.md)：NVIDIA/nccl（key: nccl），教学参照 + 官方实现，2.32.7-1 / 351f33fd fresh tarball——`src/transport/net_ib/init.cc` `NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2)` L12；topo/paths rail 映射走读归 10.2（§5/§7）
- [NVIDIA, connectx7](../references.md)：ConnectX-7 NDR 400G（key: connectx7），[C] 官方规格披露，2026-11-25——单口 NDR 4×100G/lane = 400G = 50 GB/s 单向（NIC 线速限 regime，recap 9.2 §4）（§1/§4）
- [NVIDIA, connectx8](../references.md)：ConnectX-8 SuperNIC（key: connectx8），[C] 官方规格披露，2026-11-25——800G = 100 GB/s 单向；Quantum-X800 平台每 GPU 至多 1.6 Tb/s = 2×CX8（§6）
- [NVIDIA, gb200-nvl72](../references.md)：GB200 NVL72 规格（key: gb200-nvl72），[C] 官方规格披露，2026-11-25——72 Blackwell GPU + 36 Grace、18 compute tray + 9 switch tray、NVLink 聚合 130 TB/s 双向（=72×1.8 TB/s）；域几何 recap 7.1 不重推（§6）
- 9.1 / 9.2 / 9.3 / 7.1 / 7.2 / 8.2 / 8.4 / 10.2 / 10.3 / 10.4（纯文字指引，不挂链接）：IB vs RoCE 链路层与 LID/GID（9.1）、GDR 带宽账与多 NIC 聚合 PCIe 前提（9.2 命题 2）、拥塞控制与 PFC/DCBX 配置（9.3）、NVLink 域几何与单对/聚合口径（7.1）、节点出口 400 GB/s 与 9×（7.2 §7）、ring 带宽项与分层环跨域字节（8.2）、NCCL channel（8.4）、拓扑感知集合/SHARP/PXN 走读（10.2）、a2a 拥塞（10.3）、点对点拓扑映射（10.4）——recap 不重推
