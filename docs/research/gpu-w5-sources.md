# gpu W5 research pass — P9（RDMA/GPUDirect）+ P10（机间 fabric 与大规模通信）一手来源核实

Task 1：P9（9.1-9.3，9.2【重】GDR 带宽账）+ P10（10.1-10.4，10.3 DeepEP 模块走读）一手来源核实 + **DeepEP fresh-tarball 双锚清单**（v1.2.1 机制 / main V2.5 接口）+ **NCCL topo/net_ib/gin 代码锚**（10.2 fabric 工程，recap 8.4）+ `gpu/references.md` P9/P10 扩充 + W4 残留修（rdma-core+=9.3 / deepep+=10.4 citedBy）。本波承接原 cluster 通信 scope（fabric/SHARP/EP a2a/Ring Attention）。writer 引用白名单 = `gpu/references.md` P9/P10 条目 + 跨 Part 共享锚点（P1 hopper-whitepaper / nvidia-datasheet / cuda-programming-guide，P7 pcie-spec / gpudirect-rdma-docs / gb200-nvl72，P8 nccl / nccl-docs / nccl-tests）。

## 0. 摘要

- **DeepEP 双锚 fresh tarball 全量核实**（codeload，独立空目录）：**机制锚 tag v1.2.1**（commit 9af0e0d0，2025-09-15，V1 最后版本）——normal 两阶段 NVLink+RDMA（internode.cu WarpRole kRDMASender/kRDMAAndNVLForwarder/kNVLReceivers）+ low-latency 纯 RDMA hook（internode_ll.cu SEND/RECV phase 分离 + ibgda_device.cuh GPU 直构 mlx5 WQE）+ NVSHMEM 依赖；**接口锚 main V2.5**（commit f6aa38fe，2026-10-21）——统一 EPBuffer（hybrid/direct）+ NCCL Gin 后端（gin.put<ncclTeamTagRail>）+ NVSHMEM 全移除 + BufferBase/EngramBuffer/PPBuffer/BucketBuffer/BufferAllocator 拆分。**版本漂移硬注**：把 V1 两阶段/纯 RDMA hook 语义当当前 main 接口写 = 事实错误；V2.5 "IBGDA" 仅存注释，后端实为 NCCL Gin GDAKI（DOCA GPUNetIO）。
- **IBTA 规范**：ibta-specification/ 200（2026-09-21 更新）+ **公开 What's-New PDF 一手**（Vol 1 Release 2.1，2026-07-31，2294 页；Vol 2 Release 2.1，835 页，XDR 200 Gb/s/lane；MRC Annex 21 新增）；全文 member-gated（残留，同 PCI-SIG/JEDEC 类）。RC/UC/UD QP 语义经 **rdma-core v67.0 verbs.h + RDMA Aware Programming Manual v1.7 PDF** 逐字互证。
- **RDMA verbs 一手**：RDMA Aware Networks Programming User Manual **v1.7 PDF**（932451 B）——RDMA 定义（不经远端 OS/CPU）、RC/UC/UD §3.2 逐字、操作×传输模式表（RDMA Read/Atomic 仅 RC、UD 最大 MTU、UC/RC 1GB）、QP bringup RESET→INIT→RTR→RTS §4.5、RDMA_RC 例 §8；web 版已 deprecated → DOCA。**rdma-core v67.0**（master 4b7b208c，2026-11-22）fresh tarball verbs.h 全锚。
- **GPUDirect RDMA（9.2）**：GDR docs v13.5 重核（root complex / IOMMU 1:1 / BAR1 默认 256MB、large BAR 16GB+ / nvidia-peermem R470+ 内置）；**peer-memory 状态销项**：Mellanox/peer-memory + NVIDIA/peer-memory 404（名称不存在），实际 legacy off-tree repo = **Mellanox/nv_peer_memory（200）**，R470+ deprecated → nvidia-peermem（open-gpu-kernel-modules **620.73.04** kernel-open/nvidia-peermem/nvidia-peermem.c 696 行全锚）；**gdrcopy**（master 0d85ee38，post-v2.7）API 漂移 **gdr_copy_to_mapping/from_mapping**（旧 gdr_copy_to_bar 名已移除）+ ~0.1μs 拷贝延迟。
- **NIC 带宽账（9.2【重】关键）**：**ConnectX-7 max total 400Gb/s**（datasheet 逐字），单口 NDR 400G = 4×100G/lane = 50 GB/s < PCIe Gen5 x16 64 GB/s → **NIC 线速限**；**双口 QSFP112 CX7 = 每口 NDR200（2×200G=400G 总），非 2×400G**（HW manual SKU 表逐字，datasheet portfolio 营销措辞歧义注）；**800G = ConnectX-8 SuperNIC**（100 GB/s > 64 → PCIe Gen5 限；CX8 主机口 PCIe Gen6 @64GT/s x16 = 128 GB/s/方向，Gen6 主机下 NIC 线速限——regime 须注主机 PCIe 代际）。
- **拥塞控制（9.3）**：**DCQCN Zhu SIGCOMM 2015 PDF 重取**（1524272 B）——RP/CP/NP 三角色 + 式 (1)-(4) 逐字（CNP 降速 RT=RC/RC=RC(1−α/2)/α=(1−g)α+g；fast recovery F=5 RC=(RT+RC)/2；AI RT=RT+RAI；hyper increase；无 slow start）+ 部署参数 N=50μs/K=55μs；**IB CC 一手** = Mellanox whitepaper "Deploying QoS and Congestion Control in InfiniBand"（Crupnicoff/Das/Zahavi）——IBTA CC Annex → CCA FECN→BECN→CCT/CCTI Inter-Packet Delay + CCM + VL 阈值 0-15 + credit-based 链路流控；**NVIDIA OFED SWUM 24.10 LTS** ECN（RoCEv2 RP/NP sysfs）/ PFC（802.1Qbb mlnx_qos）/ RoCE（无 SM、LID=0、GID）页逐字。
- **fabric（10.1）**：**rail-optimized 一手销项** = NVIDIA IB Cluster Bring-Up "Setting the InfiniBand Cluster Topology"（2026-10-22）逐字（fat-tree 推荐 + 每 GPU rail 接不同 LEAF + NVSwitch 路径 1 跳）+ **NCCL 2.12 blog**（2022-02-28，PXN + rail-optimized + all2all 2×）；**Quantum-2**（64×400G NDR / 51.2 Tb/s 双向 / SHARPv3）+ **Quantum-X800 Q3400-RA**（144×800G XDR / 72 OSFP / 4× Quantum-3 ASIC / 115.2 Tbps switching / **2 级 fat-tree 10,368 NICs** / SHARPv4）；**Al-Fares fat-tree PDF 残留**（ACM 403 / UCSD TLS 证书失败 / CCR online 404 / UCLA 镜像仅 slides / archive.org 不可达；Crossref 元数据全核 pp 63-74）。
- **SHARP（10.2）**：**SHARP v3.17.0 UM**（2026-11-16 更新，legacy SHARPv300 URL 200 → 现行 networking-docs.nvidia.com/sharpum/3.17.0）——网内聚合 overview 逐字 + SAT（threshold 16384、max 1GB−32B）+ **NCCL-SHARP plugin 集成页**（NCCL_COLLNET_ENABLE=1 / CX-6 HDR+ / Quantum HDR+ / nv_peer_mem / **每 HCA rail 一条 aggregation streaming flow、leaf 须同 rail**）；**Mellanox/nccl-rdma-sharp-plugins repo 200**（CollNet 实现）。
- **NCCL topo/net_ib/gin（10.2/9.2）**：版本锚 **2.32.7-1 / 351f33fd 不变**（W4 同 commit，fresh tarball 重核 W4 全锚零偏移）；新增 topo.cc NCCL_TOPO_FILE L2006 / paths.cc ncclTopoCheckGdr L494 + PXN intermediate L319-378 / net_ib/ 目录（gdr.cc nv_mem sysfs L18 / init.cc IB_ADAPTIVE_ROUTING L12 / connect.cc IB_TIMEOUT=20 L26 RETRY_CNT=7 L32）/ gin_host.cc ncclGetRailedGinType L36 / connect.cc connectRings L105。
- **Ring Attention（10.4）**：arXiv export API 再核 2310.01889v4（Hao Liu/Matei Zaharia/Pieter Abbeel）+ **官方 code haoliuhl/ringattention**（原 lhao499/llm_large_context，arXiv abs 链接 301 redirect；JAX GPU/TPU 实现）→ [A] 确认。
- **nccl-tests busbw 公式**（PERFORMANCE.md 逐字）：AllReduce 2(n-1)/n、RS/AG/AlltoAll (n-1)/n、Broadcast/Reduce 1。
- **新增 8 条**：P9 connectx7 / connectx8 / ib-cc-whitepaper / rdma-programming-manual（[C]）；P10 quantum-switch / ib-cluster-topology / nccl-all2all-blog（[C]）+ nccl-rdma-sharp-plugins（教学参照）。**W4 残留修**：rdma-core citedBy += 9.3、deepep citedBy += 10.4。
- **残留**：Al-Fares PDF 可达出处（元数据已核）；IBTA 全文 member-gated（公开 what's-new + landing FAQ + rdma-core/verbs manual 互证）；IBTA CC Annex 编号（公开源未给编号，writer 引「IBTA Congestion Control Annex」不标号）；PCI-SIG 403（W4 不变，9.2 用 W4 已核 PCIe Gen5 数值）；JEDEC 403 / B200 SM 数（W1/W6 不变）。

## 1. 核实通道与网络 fallback（本轮实际使用）

- **codeload.github.com tarball 主通道**（不占 API 限额）：DeepEP main f6aa38fe（1332872 B）/ DeepEP v1.2.1 9af0e0d0（1249140 B）/ NCCL 351f33fd（4724874 B，与 W4 byte-size 一致）/ rdma-core 4b7b208c（2092839 B）/ gdrcopy 0d85ee38（112933 B），各解压独立空目录 `/tmp/gpu-w5-{deepep-main,deepep-v121,nccl,rdma-core,gdrcopy}/`。
- **GitHub API**（用量 ~15）：DeepEP commits/main + git/ref/tags/v1.2.1 + commits/9af0e0d0（日期）；NCCL commits/master（确认 351f33fd 未漂移）；rdma-core/gdrcopy commits/master；gdrcopy tags（v2.7=7344efbd）；open-gpu-kernel-modules tags（620.73.04=a4b95f6a）+ **contents API raw** nvidia-peermem.c（大 repo 不全量 tarball，沿 W3 FlashInfer 先例）；Mellanox/nv_peer_memory（200）/ Mellanox/peer-memory（404）/ NVIDIA/peer-memory（404）/ Mellanox/nccl-rdma-sharp-plugins（200）/ linux-rdma/opensm（200）；nccl-tests README + doc/PERFORMANCE.md contents raw；haoliuhl/ringattention（repositories/648145432 跟随 301）+ contents。
- **networking-docs.nvidia.com 新门户**（docs.nvidia.com/networking/ landing 200 → 门户）：**`__pagetree.json` 通道**（门户/各手册页树 JSON，绕 JS 渲染）——sharpum/3.17.0、mlnxofedswum/24.10-5.1.6.1lts、connectx7hw、xdrswitcheshw/1.6、ibclusterbringupproc 页树 + 内容页 HTML 内嵌正文（python3 -I 去标签提取）。
- **docs.nvidia.com 稳定 200**：GPUDirect RDMA v13.5 / NCCL User Guide env.html（2.32.3）/ **rdma-aware-networks-programming-user-manual-1-7.pdf**（932451 B 一手）。
- **nvidia.com 产品页**：Quantum-2（/networking/quantum2/ 200）/ Quantum-X800（/networking/products/infiniband/quantum-x800/ 200，旧 /networking/quantum-x800/ 404）/ ConnectX-7 datasheet PDF（content/dam 276574 B）。
- **network.nvidia.com whitepaper**：IB QoS/CC whitepaper PDF（397100 B 一手）。
- **developer.nvidia.com blog 200**：NCCL 2.12 all2all blog（2022-02-28）。
- **conferences.sigcomm.org**：DCQCN 2015 p523.pdf 200（1524272 B，断点续传）；**Al-Fares 2008 p63.pdf 404**（program.php 链接指向 ccr.sigcomm.org/online node/378 → 404）。
- **Crossref API 200**：Al-Fares 10.1145/1402958.1402967（pp 63-74）/ DCQCN 10.1145/2785956.2787484（pp 523-536）。
- **arXiv export API 200**：2310.01889（HTTPS；http 本轮空响应）。
- **403/404/000/证书**：dl.acm.org PDF **403**（bot 拦截，沿 W0）；cseweb.ucsd.edu **TLS 证书失败**（unable to get local issuer certificate，000；**未降级 TLS 验证**）；ccr.sigcomm.org/online **404**；web.archive.org / archive.org **连接超时**；UCLA web.cs.ucla.edu/classes/cs217/090505DataCenters.pdf 200 但**仅 24 页 slides**（Nanxi Chen 2009 presentation，非论文正文）；docs.nvidia.com/networking/display/rdaborg* **404**（verbs manual 旧 Confluence URL → v1.7 PDF + DOCA）；/networking/{software,adapters,switches} 相对链接 404（实在 networking-docs.nvidia.com 门户）。
- 工具：`/usr/bin/grep -n`（file:line）；`pdftotext -layout`（双栏论文另用无 -layout 阅读序）；`python3 -I`（JSON/HTML 解析，隔离）；下载缓存 `/tmp/gpu-w5-research/`。

## 2. IBTA / RDMA 基础（9.1）

### 2.1 IBTA 规范（[C] 官方规格，member-gated 全文 + 公开 what's-new 一手）

- **ibta-specification/ 200**（2026-09-21 更新）：IBTA 定义维护 InfiniBand 与 RoCE 规范；链路速率达 **800Gb/s**；**RoCE 标准定义于 InfiniBand Architecture 规范内**（2010 首发）；**subnet manager** 提供路由 + QoS（10.1 SM 锚）；全文下载 member-gated（cw.infinibandta.org Member_Specifications）。
- **公开 What's-New PDF 一手**（infinibandta.org/wp-content/uploads/2026/08/Whats-New-in-Vol.-1-and-2-Release-2.1-2026-07-31.pdf，919723 B）逐字：
  - 「Volume 1, Release 2.1, published July 31, 2026」「The specification defines InfiniBand and RoCE (RDMA over Converged Ethernet)」「**2294 pages versus 2156 in 2025**」「Available to IBTA Members」。
  - 「Volume 2, Release 2.1... physical layer electrical & optical communications interfaces at data rates up through **XDR at 200 Gb/s per lane**」「**835 pages**」。
  - 「Added **Multipath Reliable Connection (MRC) – Annex 21**」（2.1 新增）；comment 列表含 RC QP error state / UC responder payload / UC/UD SRQ completion / XRC / SRQ / protection domain / IB Verbs / RoCE UDP port（互证 QP 类型与 verbs 语义）。
- **残留**：全文 member-gated（同 PCI-SIG/JEDEC 403 类）；RC/UC/UD 精确语义经 rdma-core verbs.h + verbs manual v1.7 互证（§2.2/2.3）；**CC Annex 编号公开源未给**（writer 引「IBTA Congestion Control Annex」不标号，§4.2）。

### 2.2 rdma-core v67.0（教学参照，fresh tarball master 4b7b208c，2026-11-22）

> 版本锚：CMakeLists.txt L87 `PACKAGE_VERSION "67.0"`；commit 4b7b208c6ba521aef52c6d6205677cd70942dafd（2026-11-22，「mlx5: Expose NIC_RX_CLASS flow table type」）。tarball 2092839 B，解压 `/tmp/gpu-w5-rdma-core/rdma-core-4b7b208c.../`。

| file:line | 机制 |
| --- | --- |
| `libibverbs/verbs.h` L934-937 | **enum ibv_qp_type { IBV_QPT_RC=2, IBV_QPT_UC, IBV_QPT_UD }**（QP 类型；另有 XRC/RAW_PACKET 等） |
| `libibverbs/verbs.h` L1084-1088 | **enum ibv_qp_state { IBV_QPS_RESET, INIT, RTR, RTS, ... }**（QP 状态机） |
| `libibverbs/verbs.h` L1137-1144 | **enum ibv_wr_opcode { RDMA_WRITE, RDMA_WRITE_WITH_IMM, SEND, SEND_WITH_IMM, RDMA_READ, ATOMIC_CMP_AND_SWP, ATOMIC_FETCH_AND_ADD }**（单边 READ/WRITE vs 双边 SEND/RECV） |
| `libibverbs/verbs.h` L1157-1160 | **IBV_SEND_FENCE/SIGNALED/SOLICITED/INLINE**（send flags；INLINE 小消息免 DMA） |
| `libibverbs/verbs.h` L1322 / L2692 | struct ibv_qp / **ibv_reg_mr**（MR 注册，lkey/rkey） |
| `libibverbs/verbs.h` L2962 / L3040 | **ibv_create_cq** / **ibv_poll_cq**（CQ 完成轮询） |
| `libibverbs/verbs.h` L3246 / L3467 | **ibv_create_qp** / **ibv_modify_qp**（QP 创建/状态迁移） |
| `libibverbs/verbs.h` L3662 / L3671 | **ibv_post_send** / **ibv_post_recv**（WR 提交，static inline 直调 provider） |
| `libibverbs/man/ibv_modify_qp.3` L128 | **「state from: Reset --> Init --> RTR --> RTS」**逐字 + 每 QP 类型迁移属性表（RC L148-163：RTR 需 AV/PATH_MTU/DEST_QPN/RQ_PSN/MAX_DEST_RD_ATOMIC；RTS 需 SQ_PSN/TIMEOUT/RETRY_CNT/RNR_RETRY/MAX_QP_RD_ATOMIC；UD L173-174 仅 STATE） |
| `providers/mlx5/` | ConnectX 用户态 provider（buf.c/cq.c/dbrec.c/dr_*.c，Direct Rules） |

- 9.1 verbs API 教学参照 + QP 状态机/连接开销推导代码载体；man pages 为 verbs 语义权威（ibv_create_qp.3 / ibv_post_send.3 / ibv_modify_qp.3 等）。

### 2.3 RDMA Aware Networks Programming User Manual v1.7（[C]，verbs 编程手册 ⟨verify⟩ 销项）

> PDF 一手：docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf（932451 B）。**web 版 deprecated**（networking-docs.nvidia.com/rdmaawareprogramming/ 200，逐字「This document has been deprecated. Its contents have been integrated into DOCA documentation on the RDMA-Aware Networks Programming Guide page」，2026-06-24 更新）；旧 Confluence display/rdaborg* 404。**v1.7 为经典 verbs 教程（时代注：正文 speeds 举例 SDR..FDR 56Gb/s，QP/操作语义稳定）**。

- **RDMA 定义逐字**（§1）：「Remote Direct Memory Access (RDMA) provides direct memory access from the memory of one host (storage or compute) to the memory of another host **without involving the remote Operating System and CPU**, boosting network and host performance with lower latency, lower CPU load and higher bandwidth. In contrast, TCP/IP communications typically require copy operations, which add latency and consume significant CPU and memory resources.」→ 9.1 zero-copy/kernel bypass/CPU offload + RDMA vs TCP/IP 延迟账一手。
- **三技术共 API**（§2.4）：InfiniBand / Ethernet RoCE / Ethernet iWARP「share a common user API... but have different physical and link layers」；「RDMA use the OS to establish a channel and then allows applications to directly exchange messages without further OS intervention」；IB 链路层 credit-based flow control + VL + strong ordering within VL；IB 传输层 reliability。
- **RC/UC/UD 逐字**（§3.2，p21）：
  - RC：「Queue Pair is associated with only one other QP. Messages transmitted by the send queue of one QP are **reliably delivered** to receive queue of the other QP. **Packets are delivered in order**. A RC connection is very similar to a TCP connection.」
  - UC：「...associated with only one other QP. The connection is **not reliable** so packets may be lost. Messages with errors are **not retried** by the transport, and error handling must be provided by a higher level protocol.」
  - UD：「...may transmit and receive **single-packet messages** to/from **any other UD QP**. Ordering and delivery are not guaranteed... **Multicast** messages are supported (one to many). A UD connection is very similar to a UDP connection.」
- **操作×传输模式表**（§3.2，逐字）：Send/Receive UD+UC+RC+RD；**RDMA Write (with imm) UC+RC+RD（UD 无）**；**RDMA Read 仅 RC+RD**；**Atomic Fetch&Add/Cmp&Swap 仅 RC+RD**；Max message size **UD=MTU，UC/RC/RD=1GB**；「RD is not supported by this API」。
- **QP bringup**（§4.5，p44-46）：ibv_modify_qp RESET→INIT→RTR→RTS；§4.6.5 ibv_post_send；§8 RDMA_RC 例（modify_qp_to_rtr/rts + post_send + Send/Receive/RDMA Read/Write 代码）。
- SR/RR（ibv_send_wr/ibv_recv_wr）+ RNR（receive not ready）+ CQ 语义（§3.3）。

### 2.4 NVIDIA networking docs 门户 + OFED（[C]，⟨verify 精确子页⟩ 销项）

- **门户漂移**：docs.nvidia.com/networking/ landing 200 → **networking-docs.nvidia.com**（新 help-center 门户，JS 渲染；**`__pagetree.json` 取页树**）。类目：/software（adapter-software/MLNX_OFED、accelerator-software/SHARP、management-software/OpenSM/UFM）/ /adapters / /switches / /interconnect。
- **MLNX_OFED SWUM 24.10-5.1.6.1 LTS**（networking-docs.nvidia.com/mlnxofedswum/24.10-5.1.6.1lts）：ECN / PFC（Flow Control）/ RoCE / QCN / MAD Congestion Control / Programming / NVIDIA PeerDirect / NVIDIA SM 子页（§4.3/4.4）。
- **OpenSM**：linux-rdma/opensm repo 200（pushed 2026-06-04，IB 开源 subnet manager）；门户 /software/management-software/opensm → mlnxsmrn。10.1 SM 发现/配置/故障恢复工程锚（IBTA SM 定义 + IB bring-up doc）。

## 3. GPUDirect RDMA / peermem / gdrcopy（9.2）+ NIC 带宽账

### 3.1 GPUDirect RDMA docs v13.5（[C]，W4 已锚本轮重核 + 增量）

- 「GPUDirect RDMA is a technology introduced in **Kepler-class GPUs and CUDA 5.0** that enables a direct path for data exchange between the GPU and a third-party peer device using standard features of PCI Express」（NIC/video/storage）。
- **root complex 逐字**：「the two devices must share the same upstream PCI Express root complex」。
- **IOMMU 逐字**：「GPUDirect RDMA currently relies upon all physical addresses being the same from the different PCI devices' point of view. This makes it **incompatible with IOMMUs performing any form of translation other than 1:1**, hence they must be disabled or configured for pass-through」。
- **BAR1 容量约束（9.2 关键推导增量）**：「The default BAR1 size for Tesla-class GPUs is **256 MB**. Of that, **32MB are currently reserved** for internal use... On some Tesla-class GPUs a large BAR feature is enabled, e.g. **BAR1 size is set to 16GB or larger**. Large BARs can pose a problem for the BIOS...」。
- **nvidia-peermem / peer-memory 状态销项**：「Although **deprecated when running GPU drivers from the R470 branch and newer**, customers still using the off-tree **nv_peer_mem** module (**https://github.com/Mellanox/nv_peer_memory**) and needing the persistent mapping feature will have to switch to **nvidia-peermem**」。**Mellanox/nv_peer_memory repo 200**（legacy off-tree）；spec 所查 Mellanox/peer-memory + NVIDIA/peer-memory **404**（名称不存在，非删除）——GDR 内核锚 = open-gpu-kernel-modules nvidia-peermem（R470+ 内置）。
- **persistent mapping API 变更**：nvidia_p2p_get_pages 不再接受 NULL callback；nvidia_p2p_put_pages_persistent（无 invalidation callback）；free callback 竞态修复（§4.4）。
- **Known Issue**：「there is no service to automatically load nvidia-peermem. Users need to load the module manually」。
- CUDA 13.0 变更：nv-p2p API 在 Tegra Blackwell SoC 起不再支持（迁 Linux upstream kernel API）。
- ACS/IOMMU 主锚仍 cuda-programming-guide multi-gpu-systems §3.4.2.5（W4 逐字，9.2 被引 += 9.2）。

### 3.2 nvidia-peermem（open-gpu-kernel-modules 620.73.04，教学参照，contents API raw）

> 版本锚：tag **620.73.04**（commit a4b95f6a2db0bba10ccaccef79190a3c18dcf14a）。大 repo 不全量 tarball，contents API raw 取 `kernel-open/nvidia-peermem/nvidia-peermem.c`（22891 B，696 行）于独立目录 `/tmp/gpu-w5-peermem/`。同目录 nv-p2p.h（16093 B）/ peer_mem.h（7690 B）/ nvidia-peermem.Kbuild。

| file:line | 机制 |
| --- | --- |
| `nvidia-peermem.c` L46 / L50 | **DRV_NAME "nv_mem"**（→ NCCL gdr.cc sysfs `/sys/kernel/mm/memory_peers/nv_mem/version` 互证）/ MODULE_DESCRIPTION "NVIDIA GPU memory plug-in" |
| L99 / L129 | struct nv_mem_context / **nv_get_p2p_free_callback**（GPU 页失效回调 → mem_invalidate_callback 通知 IB 核心注销 MR） |
| L198 / L216 | **nv_mem_acquire**（「acquire return code: 1 mine, 0 - not mine」）→ **nvidia_p2p_get_pages**（pin GPU 页） |
| L246 / L265 | **nv_mem_acquire_nc**（no-callback persistent）→ **nvidia_p2p_get_pages_persistent** |
| L300 / L323 / L338 | **nv_dma_map** → **nvidia_p2p_dma_map_pages**（GPU BAR 页 → NIC DMA 地址）+ sg_alloc_table（scatterlist 给 IB 核心） |
| L357 | nv_dma_unmap |
| L570 / L601 | **ib_register_peer_memory_client**（legacy nv_mem_client_ex + nc nv_mem_client_nc 双客户端注册到 IB 子系统） |
| L641 / L695-696 | nv_mem_client_init / module_init/module_exit |

- 9.2 peer memory 机制走读（NIC 经 IB peer_memory_client 直访 GPU HBM，免 host bounce buffer）；legacy（invalidate callback）vs nc（persistent）双路径。

### 3.3 gdrcopy（教学参照，master 0d85ee38，2026-11-17，post-v2.7）

> 版本锚：master commit 0d85ee38d696495592c1d589ab42e021d35cda61（最新 tag v2.7=7344efbd，master 超前）。tarball 112933 B，解压 `/tmp/gpu-w5-gdrcopy/gdrcopy-0d85ee38.../`。

- **README 逐字**：「A low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA」；「GPUDirect RDMA is meant for direct access to GPU memory from [third-party devices]... perfectly valid **CPU mappings** of the GPU memory」；「very low overhead, as it is driven by the CPU. As a reference, currently a **cudaMemcpy can incur in a 6-7us overhead**」；「initial memory **pinning** phase... **10us-1ms** depending on the buffer size」；「**Fast H-D**, because of **write-combining**. H-D bandwidth is **6-8GB/s on Ivy Bridge Xeon** but subject to NUMA effects」（时代注：历史平台数值）；「**Slow D-H**, because the GPU BAR... **can't be prefetched** and so burst reads transactions are not generated through PCIE」；dma-buf export（L92）。
- **API 漂移（writer 必注）**：现行 **gdr_copy_to_mapping / gdr_copy_from_mapping**（include/gdrapi.h L175/L177）+ _v2（L180/L182）+ gdr_copy_fence（L184）；**旧 gdr_copy_to_bar/gdr_copy_from_bar 名已移除**。gdr_open L74 / gdr_pin_buffer L87 / gdr_map L152 / gdr_get_info_v2 L142。
- `src/gdrapi.c` L1109 gdr_copy_to_mapping_internal；memcpy_{avx,avx2,avx512,sse,sse41,ls64,movdir64b,neon}.c（write-combining 存储实现）。
- `src/gdrdrv/gdrdrv.c` L325 **use_persistent_mapping=1**（module_param）/ L501 NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING / L486 nvidia_p2p_cap_persistent_pages。
- README L286-289 benchmark：**gdr_copy_to_mapping ~0.10μs**（1-8B，copylat）→ 9.2 低延迟 CPU→GPU 拷贝 vs cudaMemcpy 6-7μs 对比账。
- 9.2 gdrcopy 定位：GDR 的 CPU 侧低延迟小拷贝补充（NIC DMA 大流量 + gdrcopy 小消息/同步字），NCCL GDRCOPY_SYNC_ENABLE（W4 net.cc L338）互证。

### 3.4 NIC 带宽账（9.2【重】关键，ConnectX-7/8 一手）

**ConnectX-7**（[C]，datasheet PDF 276574 B + HW manual networking-docs.nvidia.com/connectx7hw）：
- datasheet 逐字：「NDR 400G INFINIBAND ADAPTER CARD WITH IN-NETWORK COMPUTING」「**Max total bandwidth 400Gb/s**」「IBTA Spec 1.5」「Number of network ports **1/2/4**」「Host interface **PCIe Gen5, up to x32**」「Single-port or dual-port NDR (400Gb/s) or NDR200 (200Gb/s)」「**Hardware-based congestion control**」「**Hardware-based reliable transport**」「8x virtual lanes (VL) + VL15」「256 to 4Kbyte MTU」「PCIe Gen 5.0 compatible, 32 lanes... x1/x2/x4/x8/x16」「NVIDIA Multi-Host」。
- **HW manual SKU 表（权威，逐字）**：单口 OSFP MCX75310AAS-NEAT = **NDR（4 lanes × 100Gb/s/lane）= 400G**，PCIe Gen4/5 SERDES 16/32GT/s x16，25.9W；**双口 QSFP112 MCX755106AS-HEAT = NDR200（2 lanes × 100Gb/s/lane）/port**（即 2×200G=400G 总，**非 2×400G**）；单口 QSFP112 MCX715105AS-WEAT = NDR 400G；四口 SFP56。
- **口径订正（spec 9.2「双口 400G」歧义）**：CX7 ASIC **max total 400Gb/s**；双口 IB = 2×NDR200，总仍 400G=50 GB/s。datasheet portfolio「dual-port NDR (400Gb/s)」为 NDR 家族营销措辞，**writer 以 HW manual SKU 表为准**（双口无 2×400G CX7）。
- PCIe 特性：x16/x32（Socket-Direct 辅助卡）/ bifurcation / Multi-Host 4x / **ACS for peer-to-peer secure communication** / ATS/PASID / SR-IOV / TPH / DPC / AER。

**ConnectX-8 SuperNIC**（[C]，networking-docs.nvidia.com/connectx8hw/introduction）：
- 逐字：「support for both InfiniBand and Ethernet networking at **up to 800 gigabits per second (Gb/s)**」「end-to-end 800Gb/s connectivity」「**advanced routing and telemetry-based congestion control**」「extend... SHARP」；XDR InfiniBand（SDR/EDR/HDR100/HDR/NDR/XDR 互操作）。
- **主机口**：「PCIe **Gen6 @ 64GT/s** through x16 edge connector」（x16 配置）/「PCIe Gen5 SERDES @32GT/s」（Socket Direct / Switch DSP 扩展）。

**9.2 GDR 带宽账 regime（writer 推导骨架，数值全一手）**：
- 单口 CX7 NDR400 = 400 Gb/s = **50 GB/s** < PCIe Gen5 x16 **64 GB/s**（W4 Hopper wp p.49 逐字）→ **NIC 线速限**，GDR 有效带宽 ≈ 50 GB/s。
- CX8 800G = **100 GB/s** > PCIe Gen5 x16 64 GB/s → **PCIe 限**（H100/B200/GB200 主机均 Gen5）；**若主机 PCIe Gen6 x16 = 128 GB/s/方向 > 100 → NIC 线速限**（regime 须注主机 PCIe 代际，CX8 原生 Gen6）。
- 多 NIC 聚合（每 GPU 配 NIC，rail-optimized）：n_NIC × 50 GB/s 聚合，每 NIC 仍受自身 PCIe x16 限；vs HBM 3.35 TB/s（H100，W4）→ **GDR 路径瓶颈在 PCIe/NIC 而非 HBM**（数量级差 ~67×）。
- $\min(\mathrm{BW}_{\mathrm{NIC}}, \mathrm{BW}_{\mathrm{PCIe}})$ 单 NIC；多 NIC 聚合 $\sum$ 受 PCIe 总 lane / root complex 限。
- BAR1 映射容量约束：默认 256MB（32MB  reserved）vs large BAR 16GB+（§3.1）；gdrcopy pin 10μs-1ms 一次性。

## 4. 拥塞控制与传输可靠性（9.3）

### 4.1 DCQCN（Zhu et al. SIGCOMM 2015，[B] 老经典无 arXiv，PDF 重取一手）

> PDF：conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf（1524272 B，断点续传取得）。Crossref 再核：DOI 10.1145/2785956.2787484，**pp 523-536**，作者 Yibo Zhu / Haggai Eran / Daniel Firestone / Chuanxiong Guo / Marina Lipshteyn / Yehonatan Liron / Jitendra Padhye / Shachar Raindel / Mohamad Haj Yahia / Ming Zhang（Microsoft + Mellanox）。

- **三角色**：RP（Reaction Point，发送 NIC）/ CP（Congestion Point，交换机 ECN 标记）/ NP（Notification Point，接收 NIC 生成 CNP）。基于 QCN + DCTCP；交换机只需标准 RED + ECN；NIC 硬件实现。
- **NP**：ECN-marked 包到达，若该 flow 最近 N μs 未发 CNP 则立即发 CNP， thereafter 每 N μs 至多一个 CNP；**部署 N=50μs**；CNP 高优先级发送。
- **RP 式 (1)（收到 CNP）逐字**：`RT = RC`；`RC = RC(1 − α/2)`；`α = (1−g)α + g`；**α 初值 1**。
- **式 (2)（K 时间无 CNP，UpdateAlpha）**：`α = (1−g)α`；**K=55μs**（K > N）。
- **式 (3)（Fast Recovery，F=5 次迭代）**：`RC = (RT + RC)/2`。
- **式 (4)（Additive Increase）**：`RT = RT + RAI`；`RC = (RT + RC)/2`。
- **Hyper Increase** 快速 ramp；**无 slow start**（flow 启动即线速）；timer（每 T 时间）+ byte counter（每 B 字节）双触发增速（QCN 式）；RP 状态机 Fig 7：CutRate（CNP）→ Max(T,BC)<F ? FastRecovery : Min(T,BC)>F ? HyperIncrease : AdditiveIncrease。
- **交换机标记 Fig 5**：RED 式 Pmax/Kmax/Kmin egress 队列标记概率。
- **PFC 局限（§2.2，9.3 PFC storm/victim flow 一手）**：PFC（802.1Qbb）per-priority pause 致 unfairness（Fig 3）/ victim flow（Fig 4）/ congestion spreading / PFC storm；DCQCN 逐流端到端使 PFC 罕触发。
- §5 fluid model 参数整定（RAI/F/g 收敛速度 vs 吞吐振荡 trade-off；16:1/32:1 incast 评估）。
- **9.3 关键推导**：DCQCN rate adjust 收敛行为（定性 + 式 (1)-(4)）；拥塞 → β_eff 降 → 集合通信延迟涨（training/inference 通信账恶化）；GDR 流量与拥塞交互。

### 4.2 IB 拥塞控制（IBTA CC Annex，[C] Mellanox whitepaper 一手，⟨verify⟩ 销项）

> PDF：network.nvidia.com/pdf/whitepapers/deploying_qos_wp_10_19_2005.pdf（397100 B），"Deploying Quality of Service and Congestion Control in InfiniBand-based Data Center Networks"，**Diego Crupnicoff, Sujal Das, Eitan Zahavi**（Mellanox，Rev 1.0）。

- 逐字：「The **InfiniBand Congestion Control Annex from IBTA** defines a comprehensive and scalable method of congestion control that guarantees deterministic bandwidth and latency」（对比 TCP 丢包式）。
- **CCA 三阶段逐字**：① 交换机检测拥塞 → 置 **FECN**（Forward Explicit Congestion Notification）bit；② 目的 HCA 回 **BECN**（Backward ECN）给源（connected 可 piggyback ACK，unconnected 用 CN 包）；③ 源 HCA 收 BECN → **throttle injection rate**。
- **Congestion Marking**：交换机 per-port per-VL 相对阈值（CCM 设，**0-15**，0=不标记，15=激进）；marking rate 可配。
- **Injection Rate Reduction**：**CCT（Congestion Control Table）** per-HCA，CCM 加载；每收 BECN → **CCTI（index）+= CCTI_Increase**；CCT 每项 = **Inter-Packet Delay**（HCA 包间插入延迟）；可 per-QP 或 per-port per-SL（→VL）降速。
- **Injection Rate Recovery**：HCA timer 到期 → CCTI 减（CCM 设）→ 较小 Inter-Packet Delay；无 BECN 则逐步恢复。
- **CCM（Congestion Control Manager）** 设全部参数。
- **IB 链路层 credit-based flow control**（per-VL credit，接收资源满则该 VL 停发）+ **static rate control**（fabric manager 限速）+ SL→VL QoS。
- **残留**：IBTA CC Annex 编号公开源未给（writer 引「IBTA Congestion Control Annex」不标号）；全文 member-gated。
- **vs RoCE DCQCN 对照（9.3 骨架）**：IB CCA = 交换机 FECN/BECN + CCT 包间延迟（硬件 table）；RoCE DCQCN = ECN + CNP + α 速率调节（NIC 固件）；IB 原生 credit 流控 vs RoCE 需 PFC 无损以太网。

### 4.3 NVIDIA OFED ECN/PFC/RoCE 配置（[C]，Mellanox 拥塞控制文档 ⟨verify⟩ 销项）

> MLNX_OFED SWUM 24.10-5.1.6.1 LTS（networking-docs.nvidia.com/mlnxofedswum/24.10-5.1.6.1lts）。

- **ECN 页**（explicit-congestion-notification-ecn，2026-05-18 更新）逐字：「ECN is an extension to the IP protocol... notifying all ends... when congestion occurs... **without dropping packets**」「This ECN implementation refers to **RoCE v2**」；sysfs `/sys/class/net/<if>/ecn/<protocol>/enable/<X>`（X=priority 0..7，protocol=**roce_rp / roce_np**）；「ECN supports the following algorithms: **r_roce_ecn_rp - Reaction point / r_roce_ecn_np - Notification point**」（per-device/port/priority 参数 + 统计）。
- **Flow Control 页**（flow-control）逐字：「**Priority Flow Control (PFC) IEEE 802.1Qbb** applies pause functionality to specific classes of traffic on the Ethernet link... lossless service for the RoCE traffic and best-effort service for the standard Ethernet traffic... IEEE 802.1p traffic classes」；ConnectX-4+ `mlnx_qos -i <ethX> --pfc <0/1>×8`；VLAN egress mapping + 8 TC（tc_wrap.py）；**DCBX/LLDP 自动协商**（lldpad/lldptool/dcbtool）。
- **RoCE 页**（rdma-over-converged-ethernet-roce）逐字：「RoCE... efficient data transfer with very low latencies on **lossless Ethernet** networks... RDMA transport (InfiniBand) services over Ethernet」；**「The presence of a Subnet Manager (SM) is not required in the fabric」**（vs IB）；**「LID is a layer 2 attribute of the InfiniBand protocol stack, it is not set for a port and is displayed as zero」**（RoCE 用 GID/IP）；**APM（Automatic Path Migration）not supported**（RoCE RC QP 无 alternate path）。（时代注：正文 10GigE/40GigE 举例。）
- **MAD Congestion Control 页**：SA MAD（管理流量）拥塞控制（max_outstanding/queue_size/time_sa_mad sysfs）——**非数据路径 IB CC**（writer 勿混，数据路径 CC 见 §4.2）。
- **QCN 页**（quantized-congestion-notification-qcn）：Ethernet QCN（802.1Qau，L2 拥塞通知，DCQCN 前身之一）。
- **Programming 页**：Raw Ethernet programming（kernel bypass）/ Packet Pacing（per-QP 速率）/ TSO / ToS steering（9.3 NIC 卸载锚）。
- 9.3 调参概念：ECN threshold（Kmin/Kmax/Pmax）/ PFC priority / rate limiter；大规模 fabric 调参工程 → 10.1/10.2。

## 5. DeepEP 双锚清单（10.3 模块走读，fresh tarball）

> **双锚口径继承 inference 3.3**：机制讲解锚 **tag v1.2.1**（V1 最后版本），接口走读锚 **当前 main V2.5**。**版本漂移硬注**：spec 大纲「normal NVLink+RDMA 两阶段 / low-latency 纯 RDMA hook IBGDA」为 **V1 语义**；当前 main V2.5 已统一 EPBuffer + NCCL Gin + NVSHMEM 全移除——把 V1 机制当当前 main 接口写 = 事实错误。

### 5.0 版本锚与结构

| 锚 | commit | 日期 | tarball | 解压目录 |
| --- | --- | --- | --- | --- |
| **机制 tag v1.2.1** | 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee | 2025-09-15（「support topk10 in low latency kernel (#403)」） | 1249140 B | /tmp/gpu-w5-deepep-v121/DeepEP-9af0e0d0.../ |
| **接口 main V2.5** | f6aa38fe06cd4a0489e7730365d4e0114970a2f3 | 2026-10-21（「fix: use HTTPS for DeepJIT submodule (#788)」） | 1332872 B | /tmp/gpu-w5-deepep-main/DeepEP-f6aa38fe.../ |

**结构漂移（V1 → V2.5）**：
- V1：`csrc/deep_ep.cpp`（Buffer 单类）+ `csrc/kernels/{intranode,internode,internode_ll}.cu` + `ibgda_device.cuh` + `deep_ep/buffer.py`。
- V2.5：`csrc/buffers/{base,ep,engram,pp,bucket}.hpp` + `csrc/kernels/comm/context.cpp`（Gin）+ `deep_ep/buffers/{base,ep,engram,pp,bucket,allocator}.py` + `deep_ep/include/deep_ep/impls/ep/{dispatch,combine,hybrid_dispatch,hybrid_combine}.cuh` + `deep_ep/include/deep_ep/comm/handle.cuh`（ncclGin）；**ibgda_device.cuh / internode_ll.cu / buffer.py 已移除**。

### 5.1 机制锚 v1.2.1（normal 两阶段 + low-latency 纯 RDMA hook）

**README（v1.2.1）逐字**：
- L5：「kernels optimized for **asymmetric-domain bandwidth forwarding**, such as **forwarding data from NVLink domain to RDMA domain**... high throughput... training and inference prefilling... support **SM number control**」（normal）。
- L7：「For latency-sensitive inference decoding... **low-latency kernels with pure RDMA** to minimize delays... **hook-based communication-computation overlapping method that does not occupy any SM resource**」。
- L13-22 normal 性能表（H800 ~160 GB/s NVLink + CX7 IB 400Gb/s ~50 GB/s；V3/R1 pretrain 4096 tokens/7168 hidden/top-4 groups/top-8 experts/FP8 dispatch BF16 combine）：**Intranode EP8 153/158 GB/s（NVLink）；Internode EP16 43/43、EP32 58/57、EP64 51/50 GB/s（RDMA）**（dispatch/combine）。
- L26-37 low-latency 性能表（128 tokens/7168 hidden/top-8/FP8 dispatch BF16 combine）：**EP8 77/114μs（98/127 GB/s）、EP16 118/195、EP32 155/273、EP64 173/314、EP128 192/369、EP256 194/360μs（39/40 GB/s）**。
- L39 News（2025.06.05）：「low-latency kernels now **leverage NVLink as much as possible**（#173）」——**口径注**：LL 跨机纯 RDMA，节点内 LL 自 2025.06 可用 NVLink（allow_nvlink_for_low_latency_mode）；「纯 RDMA」指跨机路径。
- L43-52 Requirements：Ampere SM80/Hopper SM90（SM90 PTX）；CUDA 12.3+（SM90）；NVLink intranode / RDMA internode。
- L54-56：**「DeepEP also depends on NVSHMEM」**（third-party/README.md 安装指南）；L82 NVSHMEM_DIR 未设则禁 internode/LL。
- L91：「fully tested with InfiniBand... theoretically compatible with RoCE」；L103 NVSHMEM_IB_SL 虚拟 lane。
- L266-274 return_recv_hook=True 例；L290：「With our receiving hook interface, the **RDMA network traffic is happening in the background, without costing any GPU SMs** from the computation part」（两 micro-batch overlap）。

**代码锚（v1.2.1）**：

| file:line | 机制 |
| --- | --- |
| `deep_ep/buffer.py` L13 / L33-37 | **class Buffer**；ctor num_nvl_bytes/num_rdma_bytes/**low_latency_mode**/num_qps_per_rank=24/allow_nvlink_for_low_latency_mode/explicitly_destroy |
| `deep_ep/buffer.py` L96-99 | low_latency_mode → **NVSHMEM_DISABLE_P2P** env（allow_nvlink 控制） |
| `deep_ep/buffer.py` L305 / L387 | **dispatch / combine**（normal，intranode/internode 分派 L362-382/L421） |
| `deep_ep/buffer.py` L435 / L486 | internode_dispatch / internode_combine |
| `deep_ep/buffer.py` L530-535 / L600-602 | **low_latency_dispatch / low_latency_combine**（**return_recv_hook** 参数） |
| `deep_ep/buffer.py` L561-562 | 逐字：「return a receiving hook if set. If set, the kernel will just do the **RDMA request issues, but without actually receiving the data**. You must call the received hook to make sure the data's arrival」 |
| `csrc/deep_ep.cpp` L15 / L306 / L543 | Buffer ctor / **intranode_dispatch** / intranode_combine |
| `csrc/deep_ep.cpp` L654 / L935 | **internode_dispatch** / internode_combine |
| `csrc/deep_ep.cpp` L1091 / L1210 / L1313 | **low_latency_dispatch** / low_latency_combine / get_next_low_latency_combine_buffer |
| `csrc/kernels/configs.cuh` L3 / L20-21 | **NUM_MAX_NVL_PEERS 8** / **LOW_LATENCY_SEND_PHASE 1 / LOW_LATENCY_RECV_PHASE 2**（hook = phase 分离） |
| `csrc/kernels/internode.cu` L355-358 | **dispatch kernel** template<kLowLatencyMode, kNumRDMARanks, kCachedMode, ...> __launch_bounds__ |
| `csrc/kernels/internode.cu` **L370-376** | **enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers }**（**两阶段 warp 专化**） |
| `csrc/kernels/internode.cu` L462 / L615 | kRDMASender 分支（RDMA 发送到远端节点）/ kRDMASenderCoordinator |
| `csrc/kernels/internode.cu` **L690-691 / L738** | **kRDMAAndNVLForwarder**「**RDMA consumers and NVL producers**」/「**Forward tokens from RDMA buffer**」（**第二阶段 NVLink 转发到域内目的 GPU**） |
| `csrc/kernels/internode.cu` L995 / L1008 | dispatch host launcher / kNumDispatchRDMASenderWarps=7 |
| `csrc/kernels/internode.cu` L149 / L672 | **nvshmemi_ibgda_put_nbi_warp**（RDMA put，IBGDA） |
| `csrc/kernels/internode_ll.cu` **L40-41 / L555-556** | **LL dispatch / combine kernel**（__launch_bounds__(1024,1)） |
| `csrc/kernels/internode_ll.cu` L87 / L248 | **phases & LOW_LATENCY_SEND_PHASE / RECV_PHASE** 门控（send-only issue → hook recv） |
| `csrc/kernels/internode_ll.cu` L164 / L231 | **nvshmemi_ibgda_put_nbi_warp**（纯 RDMA put）/ **nvshmemi_ibgda_amo_nonfetch_add**（原子计数） |
| `csrc/kernels/internode_ll.cu` L337 / L920 | LL dispatch / combine host launcher（phases 参数） |
| `csrc/kernels/ibgda_device.cuh` **L63 / L72** | **mlx5_wqe_ctrl_seg typedef**（GPU 直构 mlx5 WQE）/ ibgda_get_state |
| `csrc/kernels/ibgda_device.cuh` L255 / **L339** / L433 / L471 / **L495** | ibgda_reserve_wqe_slots / **nvshmemi_ibgda_put_nbi_warp** / amo_nonfetch_add / ibgda_poll_cq / **nvshmemi_ibgda_quiet** |
| `setup.py` L19-25 | **NVSHMEM_DIR / nvidia.nvshmem wheel / libnvshmem_host.so**（NVSHMEM 依赖） |

- **normal 两阶段机制（9.2/10.3）**：节点内 NVLink dispatch（intranode.cu）；跨机 internode = **阶段 1 RDMA 发送到远端节点同 nvl_rank GPU（kRDMASender）→ 阶段 2 NVLink 转发到域内最终目的（kRDMAAndNVLForwarder）**——asymmetric-domain forwarding（NVLink 域 ~160 GB/s vs RDMA ~50 GB/s，先聚合再跨机省 RDMA 流量）。
- **low-latency hook 机制（9.2）**：纯 RDMA（IBGDA GPU 直构 WQE，免 CPU proxy）；**SEND phase 仅 issue RDMA 请求不接收 → return_recv_hook 返回 hook → 计算重叠 → 调 hook 触发 RECV phase**；不占 SM（RDMA 后台）；预注册 buffer（num_rdma_bytes）+ num_qps_per_rank。

### 5.2 接口锚 main V2.5（统一 EPBuffer + NCCL Gin + NVSHMEM 移除）

**README（main）逐字**：
- L11-18 **V2.5 release**：「Split **ElasticBuffer** into **EPBuffer, EngramBuffer, PPBuffer, and BucketBuffer**, sharing the **BufferBase** lifecycle」（L12）；「Add batched all-gather, reduce-scatter, and all-reduce through BucketBuffer」（L14）；「**EPBuffer.lb_prefetch_weights** and **EPBuffer.lb_reduce_grads** for dynamic redundant experts... prefetch expert weights and quantization scales over NVLink before expert computation... accumulate redundant experts' FP32 gradients... MoonEP / UltraEP」（L15）；「**Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation. NVSHMEM is no longer a dependency**」（L18）。
- L20 **V2 release**：「complete refactoring... larger scale-up and scale-out domains and the lightweight **NCCL Gin backend**」。
- L25-32 features：「**NCCL Gin backend** — Lightweight device-side communication APIs / Able to reuse existing NCCL communicators」；「**EPv2** — High-throughput and low-latency APIs **unified into a single EPBuffer interface**... Larger scale-up & scale-out domain support... **Analytical SM & QP count calculation — no more auto-tuning**... **Both hybrid & direct modes remain supported**」。
- L38 Notes：「**EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported**」（vs V1 LL hook 不占 SM——**版本漂移注**）。
- L41：「Engram requires a NCCL build providing **ncclGinOptFlagsWarpGet**」。
- L45-56 Requirements：**Linux / Python 3.10+ / NVIDIA Hopper or newer / CUDA Toolkit 13.1+（DeepJIT）/ C++20（std::format）/ PyTorch 2.10+ / NCCL 2.32.3+ / NVLink intranode / RDMA internode**。
- L57：**DeepJIT** 运行时编译（安装期无 CUDA 编译，无需可见 GPU/TORCH_CUDA_ARCH_LIST）；nvidia-smi + ibstat 自动带宽检测。
- L107：「High-throughput and low-latency EP operations **share a single EPBuffer interface**... SM and QP counts are **estimated analytically** and can be overridden per call」。
- L160-162：**do_cpu_sync=True** 精确输出尺寸 + CPU 侧 expert 计数（训练/prefill）；**False** GPU 侧接收计数（decode，输出按容量分配，expert GEMM 用 handle.psum_num_recv_tokens_per_expert 判有效范围）。
- L293-328 **BucketBuffer**（CP/DP batched all_gather/reduce_scatter/all_reduce，BufferAllocator 规划，NVLink/RDMA/hybrid，NVLink reduce 需 NCCL multimem，all-gather copy engine num_sms=0）。
- L332-334 **lb_prefetch_weights / lb_reduce_grads**（EPLB 冗余 expert 通信原语，inference 7.4 recap）。
- L384 **EngramBuffer**（experimental，GPU/CPU 多层 RDMA fetch）；L390 **PPBuffer**（RDMA-only pipeline send/recv）。
- L406 **EP_DISABLE_GIN**：「Skip Gin initialization for NVLink-only use. **RDMA operations require Gin**; this flag does not select another RDMA backend」。
- L428 EP_GIN_GDAKI_DEBUG（NCCL Gin **GDAKI** device debug）。

**代码锚（main V2.5）**：

| file:line | 机制 |
| --- | --- |
| `deep_ep/buffers/ep.py` **L208** | **class EPBuffer(BufferBase)** |
| `deep_ep/buffers/ep.py` L217-219 | docstring 逐字：「**Hybrid mode uses hierarchical RDMA + NVLink communication** to achieve higher bandwidth, and is more friendly to **multi-plane/multi-rail networks**」 |
| `deep_ep/buffers/ep.py` L242 / L255 | __init__ / **allow_hybrid_mode=True** |
| `deep_ep/buffers/ep.py` L553-557 | QP 数：**direct 少 QP 减 DB ringing / hybrid 每 channel 独立 QP** |
| `deep_ep/buffers/ep.py` **L562 / L770** | **dispatch / combine**（统一接口，do_cpu_sync L576） |
| `deep_ep/buffers/ep.py` L854 / L886 | **lb_prefetch_weights / lb_reduce_grads** |
| `deep_ep/buffers/ep.py` L27 | class EPHandle（token_metadata_at_forward / channel_linked_list hybrid only） |
| `deep_ep/buffers/base.py` L4 | **class BufferBase(ABC)**（共享生命周期） |
| `deep_ep/buffers/allocator.py` L10 | **class BufferAllocator** |
| `deep_ep/buffers/engram.py` L31 / `pp.py` L12 / `bucket.py` L184 | EngramBuffer / PPBuffer / BucketBuffer（均 BufferBase） |
| `csrc/buffers/ep.hpp` **L23** | **class EPBuffer: public BufferBase**（C++ 侧） |
| `csrc/buffers/ep.hpp` L183 / L192 / L199 | get_dispatch_buffer_size / **Direct dispatch** / **Hybrid dispatch** 尺寸 |
| `csrc/buffers/ep.hpp` L213 / L221 / L232 | get_combine_buffer_size / Direct / Hybrid combine |
| `deep_ep/include/deep_ep/comm/handle.cuh` **L21 / L31** | **ncclGin gin** 成员 / ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode) |
| `handle.cuh` L113 / L123-125 | **ncclGin_VASignalAdd** / gin.wait（Gin device API） |
| `deep_ep/include/deep_ep/impls/ep/dispatch.cuh` **L31-32** | **dispatch_impl** kernel（direct，template<kIsScaleupNVLink,...>） |
| `dispatch.cuh` L155-156 | **gin.put_value<team_t>(..., ncclGinOptFlagsAggregateRequests)** |
| `deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh` **L34-35** | **hybrid_dispatch_impl** kernel（hierarchical RDMA+NVLink） |
| `hybrid_dispatch.cuh` **L179-183 / L445-451** | **gin.put<ncclTeamTagRail>(...)**（rail team）/ 「Issue IBGDA requests」注释 → **gin.put<ncclTeamTagRail>**（**IBGDA 名存注释，实为 Gin GDAKI**） |
| `csrc/kernels/comm/context.cpp` **L126** | **(allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE**（Gin 后端探测） |
| `context.cpp` L65 / L219 | get_gin_min_stride / ncclWindow_t（RDMA storage windows） |

- **NVSHMEM 移除核实**：全 repo grep nvshmem/ibgda（*.py/*.cpp/*.hpp/*.cu/*.cuh/*.md）——**仅 README L18「NVSHMEM is no longer a dependency」+ hybrid_dispatch.cuh L441/444、hybrid_combine.cuh L482「IBGDA」注释**（指 Gin GDAKI GPU-initiated issue，非 NVSHMEM IBGDA）；**ibgda_device.cuh / internode_ll.cu / NVSHMEM setup 全移除**。
- **V2.5 hybrid vs direct（10.3 接口增量）**：hybrid = hierarchical RDMA+NVLink（V1 normal 两阶段后继，multi-rail 友好）；direct = 纯 RDMA（V1 LL 后继，但**占 SM**，无 V1 hook 不占 SM 语义）；统一 EPBuffer.dispatch/combine + do_cpu_sync 区分训练/prefill vs decode；**NCCL Gin 后端**（复用 NCCL communicator，GPU-initiated networking，NCCL src/gin/ + net_ib/gdaki/doca-gpunetio，§6.3）。
- **10.3 走读 scope**：V1 机制（v1.2.1 锚，§5.1）讲两阶段/hook 原理 + V2.5 接口（main 锚，§5.2）讲统一 EPBuffer/Gin/版本漂移；dispatch/combine kernel token 路由；a2a 拥塞（incast，9.3 recap）；EP>32 递减（inference 3.3 §5 命题 3 recap）；EPLB 通信（lb_* 原语，inference 7.4 recap）。

## 6. NCCL topo / net_ib / gin + SHARP（10.2，recap 8.4）

### 6.1 版本锚（W4 不变，fresh tarball 重核）

> NCCL master **351f33fd087e0c4d0cfb99f1c61dc13da7996c28**（2026-11-24，v2.32.7-1）——**与 W4 同 commit，本轮 API 确认未漂移**；fresh tarball 4724874 B 解压 `/tmp/gpu-w5-nccl/nccl-351f33fd.../`，W4 全锚（rings.cc L29 / trees.cc L90 / connect.cc L395 / search.cc L1245 / paths.cc L104/243/341）逐行重核**零偏移**。8.4 机制走读 recap，10.2 fabric 工程增量。

### 6.2 拓扑检测与 rail/PXN/GDR（src/graph/，10.2）

| file:line | 机制 |
| --- | --- |
| `src/graph/topo.cc` L1994 / L1072 | **ncclTopoGetSystem** / ncclTopoGetSystemFromXml（W4 锚重核） |
| `src/graph/topo.cc` **L2006-2013** | **NCCL_TOPO_FILE** env → **ncclTopoGetXmlFromFile**；默认 `/var/run/nvidia-topologyd/virtualTopology.xml`（与 docs 逐字一致） |
| `src/graph/topo.cc` L2170 | NCCL_TOPO_DUMP_FILE |
| `src/graph/topo.cc` L485-486 / L745-748 | NIC speed 从 XML（mbps，默认 10000）/ PCIe link_speed kvDictPciGen（拓扑带宽检测） |
| `src/graph/topo.h` L47-51 / L60-70 | node 类型 **GPU 0/PCI 1/NVS 2/CPU 3（NUMA）/NIC 4**；LINK_LOC/NVL/PCI/NET |
| `src/graph/topo.h` L24 / L26 / L317-318 | **SM90_NVLINK_BW 20.6 / SM100 40.1**（NCCL 有效模型值 vs raw 25/50，W4 注） |
| `src/graph/paths.cc` L104 / L243 / L341 | **PATH_PHB**（经 CPU root complex）/ **PATH_PXN**（tx==GPU，经中间 GPU 到 NIC）/ PATH_NVL（W4 锚重核） |
| `src/graph/paths.cc` **L319-378** | **「The connection may go through an intermediate rank」**→ intermediateRank / intermediateNode（**PXN rail 对齐代码实化**：GPU1-DEV1-DEV2-DEV3-GPU2，中间 DEV → rank） |
| `src/graph/paths.cc` **L494 / L542** | **ncclTopoCheckGdr**（GPU-NIC GDR 可行性）/ **NCCL_NET_GDR_LEVEL**（ncclTopoUserGdrLevel） |
| `src/graph/paths.cc` L716 / L839 / L860 | GDR check 调用（net/PXN 路径） |
| `src/graph/connect.cc` **L105 / L462** | **connectRings**（ring 跨节点连接，ringRecv/ringSend/ringPrev/ringNext）/ ncclTopoPostset 内调用 |
| `src/graph/rings.cc` L29 / `trees.cc` L90 / `search.cc` L331/L1245 | ncclBuildRings / ncclGetDtree / ncclTopoSearchRec / ncclTopoCompute（W4 锚重核） |

### 6.3 IB transport / GDR / Gin（src/transport/net_ib/ + src/gin/，9.2/10.2/10.3）

> **结构注**：2.32 `net_ib` 为**目录**（非单文件）：init.cc / connect.cc / gdr.cc / gin.cc / p2p.cc / reg.cc / common.cc + **gdaki/doca-gpunetio/**（GDAKI = GPU Direct Async Kernel-Initiated，DOCA GPUNetIO，**NCCL Gin RDMA 后端实现**，DeepEP V2.5 所依）。

| file:line | 机制 |
| --- | --- |
| `src/transport/net_ib/gdr.cc` **L18 / L24** | **ibGdrSupportInitOnce**：KNL_MODULE_LOADED("**/sys/kernel/mm/memory_peers/nv_mem/version**")（**nvidia-peermem DRV_NAME nv_mem 互证**，§3.2）/ **ncclIbGdrSupport** |
| `src/transport/net_ib/init.cc` **L12 / L594** | **NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2)** / 「Enable ADAPTIVE_ROUTING by default on IB networks」 |
| `src/transport/net_ib/init.cc` L464 | **NCCL_IB_HCA** env（HCA 过滤） |
| `src/transport/net_ib/connect.cc` L14 / **L26 / L32** | NCCL_IB_GID_INDEX / **NCCL_IB_TIMEOUT 默认 20**（4.096μs×2^20≈4.3s）/ **NCCL_IB_RETRY_CNT 默认 7**（IB 可靠传输重传，9.3） |
| `connect.cc` L42 / L49 / L106 | NCCL_IB_SL / NCCL_IB_TC / **NCCL_IB_QPS_PER_CONNECTION 默认 1** |
| `connect.cc` L989 | AR merge（comm->ar = 所有 merged devs AR enabled） |
| `src/gin/gin_host.cc` **L28 / L36** | **ncclGetGinType** / **ncclGetRailedGinType**（rail-aware Gin，DeepEP context.cpp L126 互证） |
| `src/transport/coll_net.cc` L416 / L1613 / L1648 | **sharedBuffersInit**（CollNet 共享缓冲，W4 锚）/ COLLNET_CHAIN / COLLNET_DIRECT setup |
| `src/init.cc` **L67 / L2581** | **NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", ...)** / 每次 comm init 重载 |
| `src/transport/net.cc` L1327 / L1496（W4） | sendProxyProgress / recvProxyProgress（proxy 线程，9.2 GDR 与 NCCL NET recap） |

### 6.4 NCCL docs env（[C]，2.32.3，10.2 调优锚）

- **NCCL_TOPO_FILE**（2.6+）逐字：「Path to an XML file to load before detecting the topology. By default, NCCL will load **/var/run/nvidia-topologyd/virtualTopology.xml** if present... The topology file must describe the topology of a **single host** (even for multi-node NVLink systems)」。
- **NCCL_COLLNET_ENABLE**（2.6+）：「Enable the use of the **CollNet plugin**. Default is 0, define and set to 1」；**NCCL_COLLNET_NODE_THRESHOLD**（2.9.9+，默认 2）。
- **NCCL_NET_GDR_LEVEL**（formerly NCCL_IB_GDR_LEVEL）；**NCCL_IB_CUDA_SUPPORT**：「force or disable the usage of GPU Direct RDMA. **By default, NCCL enables GPU Direct RDMA if the topology permits it**」；**NCCL_NET_GDR_READ**：send 侧 GDR read，2.4.2+ NVLink 平台默认开。
- **NCCL_IB_ADAPTIVE_ROUTING**（2.16+）逐字：「Enable... Adaptive Routing... **Enabled (1) by default on IB networks. Disabled (0) by default on RoCE**... system defined Adaptive Routing enabled SL has to be selected（cf. NCCL_IB_SL）」。
- **NCCL_IB_HCA**：过滤式 `<hca>[:<port>[:<rail>[:<plane>]]]`（**rail/plane 语法**）；**NCCL_IB_RAIL_POLICY**（2.30.5+）：「Controls automatic **rail and plane assignment** for supported **ConnectX SuperNIC** devices... NCCL_NET_MERGE_POLICY=RAIL... NONE disable」。
- NCCL_IB_TIMEOUT/RETRY_CNT/GID_INDEX/SL/TC/QPS_PER_CONNECTION/SPLIT_DATA_ON_QPS/PCI_RELAXED_ORDERING 均在 env.html ✓。
- NCCL_ALGO/PROTO/NVLS_ENABLE（W4 已核）；**docs 2.32.3 滞后 repo 2.32.7 注维持**。

### 6.5 SHARP（[C] v3.17.0 + 教学参照 plugin，10.2 网内聚合）

> **版本漂移**：spec/W0 锚 SHARPv300（docs.nvidia.com/networking/display/SHARPv300 仍 200）→ **现行 SHARP v3.17.0 UM**（networking-docs.nvidia.com/sharpum/3.17.0，2026-11-16 更新；版本列表 2.6.1..3.17.0）。

- **Overview 逐字**（sharpum/3.17.0/introduction）：「NVIDIA SHARP technology improves the performance of MPI and Machine Learning collective operation, by **offloading collective operations from CPUs and GPUs to the network** and **eliminating the need to send data multiple times between endpoints**... decreases the amount of data traversing the network as aggregation nodes are reached... **With the 3rd generation of SHARP, multiple aggregation trees can be built over the same topology**... In-Network Computing... many parallel jobs over the same infrastructure」。
- **Collective library**（nvidia-sharp-collective-library）：Barrier/Allreduce/Reduce/Bcast/Reduce-scatter/Allgather；16/32/64-bit int/float + **BF16 + INT8**；**sharpd daemon 2.7.0 起移除**（in-process）；**SAT（Streaming Aggregation Tree）**：SHARP_COLL_ENABLE_SAT（默认 0）、**max message 1073741792 B（1GB−32B）**、**SHARP_COLL_SAT_THRESHOLD 默认 16384**（16KB，大消息 SAT）；低延迟 OST quota 调参。
- **NCCL 集成页**（using-nvidia-sharp-with-nvidia-nccl）逐字：「RDMA and SHARP collectives are enabled with NVIDIA NCCL through the **NCCL-SHARP plugin**」；分发 = **HPC-X binary**（自动加载）/ **source github.com/Mellanox/nccl-rdma-sharp-plugins**（LD_LIBRARY_PATH）；Requirements：**ConnectX-6 HDR+ / Quantum HDR switch+ / DOCA-Host GPUDirectRDMA Plugin / nv_peer_mem module（service nv_peer_mem status / lsmod grep nv_peer_mem）/ NCCL 2.7.3+**；**NCCL_COLLNET_ENABLE=1**（+ NCCL_ALGO=CollNet 克服 ≤2.7.8 bug）；SHARP_COLL_LOCK_ON_COMM_INIT=1（SAT 资源保证）；NCCL_SHARP_DISABLE / NCCL_SHARP_GROUP_SIZE_THRESH / NCCL_IBEXT_DISABLE；**「NCCL SHARP Streaming aggregation is supported on a single NCCL communicator/process group」**；**rail 拓扑要求逐字**：「On systems with multiple GPUs and multiple HCAs, **NCCL creates an aggregation streaming flow (NCCL Ring/Channel) per HCA rail**. It is required to build the cluster topology in such a way that **leaf level switches connected to same HCA rail from each server**」；nccl-tests all_reduce_perf 例（mpirun -np 1024 -map-by ppr:8:node -x NCCL_COLLNET_ENABLE=1）。
- **nccl-rdma-sharp-plugins repo**（教学参照，API 200）：README「nccl-rdma-sharp plugin enables **RDMA and Switch based collectives(SHARP)** with NVIDIA's NCCL」；Requirements MOFED/CUDA/SHARP/NCCL/**GPUDirectRDMA plugin**；autotools（autogen.sh/configure --with-sharp=PATH/make）。
- **硬件代际**：Quantum-2 **SHARPv3**（产品页）/ Quantum-X800 **SHARPv4**（产品页）；NVLS（NVLink SHARP，NVSwitch 内）vs IB SHARP（跨机交换机内）——7.1/8.4 recap，10.2 对照。
- **10.2 关键推导**：SHARP 网内聚合带宽/延迟节省（交换机内 reduce → 流量降，endpoint 免多次发送）；SAT 大消息流式聚合；多 aggregation tree 多作业并行；CollNet chain/direct（NCCL coll_net.cc）。

### 6.6 nccl-tests busbw（教学参照，PERFORMANCE.md 一手，10.2 测量方法）

> NVIDIA/nccl-tests README（198 行）+ **doc/PERFORMANCE.md**（145 行）contents API raw 重核（核实日期 → 2026-11-25）。

- 逐字：「NCCL tests report... **algorithm bandwidth and bus bandwidth**」；**algbw = S/t**；「Bus Bandwidth... compare it with the **hardware peak bandwidth, independently of the number of ranks**」。
- **AllReduce**：「we need **2(n-1) data transfers** (x number of elements)... `t = (S*2*(n-1)) / (n*B)` = `(S/B) * (2*(n-1)/n)`... **B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)**」。
- **ReduceScatter / AllGather**：`t = S*(n-1)/(B*n)`；**B = algbw * (n-1)/n**（S = 总数组字节 = recvcount/sendcount×sizeof×n）。
- **Broadcast / Reduce**：root 瓶颈 `t = S/B`；**B = algbw（factor 1）**。
- **Summary 逐字**：AllReduce **2(n-1)/n**；ReduceScatter **(n-1)/n**；AllGather **(n-1)/n**；Broadcast **1**；Reduce **1**；**AlltoAll (n-1)/n**；「The bus bandwidth should reflect the speed of the **hardware bottleneck: NVLink, PCI, QPI, or network**」。
- 10.2 万卡 all-reduce 实测账 = 讲测量方法 + 引官方披露/已发表 nccl-tests 数值（spec §6.2，非新跑 benchmark）；10.3 a2a (n-1)/n recap 8.1。

## 7. 机间 fabric 架构与拓扑（10.1）

### 7.1 rail-optimized + fat-tree（[C] NVIDIA 官方，⟨verify⟩ 销项）

**IB Cluster Bring-Up Procedure — Setting the InfiniBand Cluster Topology**（networking-docs.nvidia.com/ibclusterbringupproc/setting-the-infiniband-cluster-topology，2026-10-22 更新）逐字：
- 「**Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized.**」
- 「**Rail-optimized design means a GPU node with multiple interfaces will put each GPU "rail" (IB network interface) onto a different first level (LEAF) switch** for cluster Interconnect.」
- 「This allows multiple nodes to **utilize their internal NVSwitch path to talk across a NIC that is just one switch hop away** (instead of having to cross multiple switches, incurring additional latency).」
- 3 Tier / 2 Tier 拓扑图；PTP excel → Topology file；**FNM port（XDR only，Q3400-RA 专用 OSFP in-band 管理口，UFM 连接不占数据口）**。
- Cluster Planning 页：topology type → PTP excel → topology file → confirm（工程流程）。

**NCCL 2.12 blog**（developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/，2022-02-28，Karthik Mandakolathur + Sylvain Jeaugey，[C] 开放技术博客）逐字：
- rail-optimized：「NIC-0 of each DGX system is connected to the same leaf switch (L0), NIC-1s... (L1), and so on. Such a design is often called **rail-optimized**」；「Rail-optimized network topology helps **maximize all-reduce performance while minimizing network interference between flows**. It can also **reduce the cost** of the network by having lighter connections between rails.」
- **PXN**：「PXN leverages NVIDIA **NVSwitch** connectivity between GPUs within the node to **first move data on a GPU on the same rail as the destination, then send it to the destination without crossing rails**」；before 2.12 三跳（L0,S1,L3）→ PXN 一跳；**message aggregation**（multireceive 8 buffers，remote CPU proxy 一次发 8 消息，connections shared）→ all2all 2×。
- **10.2/10.3 回链**：PXN = NCCL paths.cc PATH_PXN L243 + intermediate rank L319-378（§6.2）；rail-optimized = SHARP SAT per-rail flow（§6.5）+ NCCL_IB_RAIL_POLICY（§6.4）。

### 7.2 Quantum 交换机（[C]，⟨verify⟩ 销项）

**Quantum-2（NDR）**（nvidia.com/en-us/networking/quantum2/ 200）逐字：
- 「The NVIDIA Quantum-2 family of switches comprises **64 400Gb/s ports or 128 200Gb/s ports** on physical **32 octal small form-factor (OSFP) connectors**... compact **1U**... air-cooled and liquid-cooled」。
- 「aggregated **51.2 terabits per second (Tb/s) of bidirectional throughput**... more than **66.5 billion packets per second (bpps)**」（= 64×400G×2 = 51.2T 双向 ✓）。
- 「third generation... **SHARPv3**... MPI Tag Matching, MPI All-to-All, and programmable engines」。

**Quantum-X800（XDR）**（nvidia.com/en-us/networking/products/infiniband/quantum-x800/ 200，旧 /networking/quantum-x800/ 404）逐字：
- 「The NVIDIA Quantum-X800 InfiniBand switch provides **144 ports of 800 Gb/s connectivity per port**. It includes hardware-based In-Network Computing with **SHARP v4, adaptive routing, telemetry-based congestion control, performance isolation**」。
- ConnectX SuperNIC「up to **1.6 terabits per second (Tb/s)** of connectivity per GPU」（2×CX8 800G）。

**XDR Switch HW Manual**（networking-docs.nvidia.com/xdrswitcheshw/1.6，Q32xx/Q34xx）逐字：
- Introduction：「The NVIDIA **Quantum-3** family... **Q3400-RA 4U** switch... first to utilize **200 Gb/s-per-lane SerDes**... **144x 800 Gb/s ports distributed over 72 OSFP cages** and a dedicated management port for NVIDIA UFM... **With this very high radix, a two-level fat tree topology can connect up to 10,368 network interface cards (NICs) at lowest latency** while keeping maximum job locality」（**144²/2 = 10,368 ✓，10.1 fabric 规模账一手**）；Q3401-RD（48-54V DC busbar）/ Q3300-LD 3U 全液冷 144×800G。
- Specifications：Q3400-RA「Switch **4x Quantum-3 ASICs** / Throughput **Switching 115.2 Tbps** / Speed 400/800Gb/s per port」（**115.2T = 144×800G 单向口径**；Quantum-2 51.2T 为双向口径——**writer 须注口径差**）；Q3300-LD「2 independent 28.8Tb/s switches」。

### 7.3 fat-tree 经典（Al-Fares SIGCOMM 2008，[B]，PDF 残留）

- **Crossref 再核**（API 200）：DOI **10.1145/1402958.1402967**，「A scalable, commodity data center network architecture」，**Proceedings of ACM SIGCOMM 2008，pp 63-74**，作者 **Al-Fares / Loukissas / Vahdat**（UC San Diego），2008；亦发表于 SIGCOMM CCR 38(4)（DOI 10.1145/1402946.1402967）。
- **PDF 可达性残留（W0→W5）**：ACM DL PDF **403**（bot 拦截）；cseweb.ucsd.edu/~vahdat/papers/sigcomm08.pdf **TLS 证书失败**（unable to get local issuer certificate；**未降级验证**）；SIGCOMM 2008 program.php 链接 → ccr.sigcomm.org/online node/378 **404**；UCLA web.cs.ucla.edu/classes/cs217/090505DataCenters.pdf 200 但**仅 24 页 slides**（Nanxi Chen 2009 presentation，非论文）；archive.org / web.archive.org **连接超时**；Semantic Scholar openAccessPdf 仅指 ACM bronze（403）。
- **writer 口径**：fat-tree 理论（k-port switch Clos：k³/4 hosts、full bisection、two-level routing）按 spec 10.1 关键推导独立重推（k-port 构造），Al-Fares 为出处锚（[B] 元数据已核）；**精确公式逐字待 PDF 可达**（残留）。Quantum-X800 2 级 10,368 NICs（§7.2）为官方一手 fabric 规模账，可独立支撑 10.1 推导。

### 7.4 GB200 NVL72 vs fat-tree（10.1 对照，W4 已锚）

- gb200-nvl72 条目（W4）：72 GPU NVLink 域 130 TB/s 双向 vs 传统 IB fat-tree（每 GPU CX7 50 GB/s / CX8 100 GB/s）——NVLink 域扩展 vs IB fabric 数量级差；被引 10.1 已含。

## 8. Ring Attention（10.4，[A] 再核）

- **arXiv export API 200**（HTTPS）：2310.01889，「**Ring Attention with Blockwise Transformers for Near-Infinite Context**」，**Hao Liu / Matei Zaharia / Pieter Abbeel**，published 2023-10-03，updated 2023-11-27（**v4**）。
- **abstract 逐字**：「...leverages **blockwise computation of self-attention and feedforward** to distribute long sequences across multiple devices while **fully overlapping the communication of key-value blocks with the computation**...」（10.4 overlap 条件 t_comm ≤ t_compute 出处）。
- **官方 code 核实**：arXiv abs Comments「Code: this https URL」→ github.com/lhao499/llm_large_context **301 Moved Permanently** → **haoliuhl/ringattention**（repositories/648145432，「Large Context Attention」，778 stars，pushed 2025-10-13，not archived）。README 逐字：「**GPU/TPU Jax implementation of RingAttention**」；ringattention/ringattention_jax.py（17127 B）/ ringattention_jax_inference.py / ringattention_pallas_tpu.py（55312 B）；pip install ringattention；shard_map 分片。
- **[A] 确认**（paper + official code）；**版本/口径注**：官方实现为 **JAX**（GPU/TPU），非 PyTorch/NCCL——10.4 transport 级增量用 **NCCL ncclSend/ncclRecv**（8.4 P2P recap），Ring Attention 调度机制引论文 + JAX repo 佐证；分块机制归 models 4.6（recap），CP 账归 training 5.2（recap）。

## 9. ⟨verify⟩ 销项 / 残留

| 项 | 状态 | 依据 |
| --- | --- | --- |
| IBTA 规范 URL（spec 9.1 ⟨verify URL⟩） | **销** | ibta-specification/ 200 + 公开 What's-New Vol 1/2 Release 2.1 PDF（§2.1） |
| ConnectX 系列（spec 9.1 ⟨verify〉） | **销** | CX7 datasheet PDF + HW manual SKU 表 + CX8 SuperNIC intro（§3.4） |
| verbs 编程手册（spec 9.1 ⟨verify〉） | **销 + 漂移** | RDMA Aware Programming Manual v1.7 PDF 一手；web deprecated → DOCA；旧 display/rdaborg 404（§2.3） |
| rdma-core repo（spec 9.1 ⟨verify〉） | **销** | linux-rdma/rdma-core 200，v67.0 master 4b7b208c fresh tarball（§2.2） |
| peer-memory 状态（spec 9.2 ⟨verify W5〉） | **销** | Mellanox/peer-memory + NVIDIA/peer-memory 404（名称不存在）；legacy = Mellanox/nv_peer_memory 200（R470+ deprecated）；nvidia-peermem 内置 open-gpu-kernel-modules 620.73.04（§3.1/3.2） |
| IBTA CC 机制（spec 9.3 ⟨verify〉） | **销（机制）+ 残留（编号/全文）** | IB CC whitepaper FECN/BECN/CCT/CCM 逐字（§4.2）；CC Annex 编号公开源未给、全文 member-gated |
| Mellanox 拥塞控制文档（spec 9.3 ⟨verify〉） | **销** | OFED SWUM 24.10 LTS ECN/PFC/RoCE/QCN 页 + IB CC whitepaper（§4.2/4.3） |
| DCQCN 精确出处（spec 9.3） | **销** | PDF 重取 + 式 (1)-(4) + Crossref pp 523-536（§4.1） |
| Quantum switch docs（spec 10.1 ⟨verify〉） | **销** | Quantum-2 产品页 + Quantum-X800 产品页 + XDR HW manual（§7.2） |
| Al-Fares fat-tree 出处（spec 10.1 ⟨verify〉） | **残留（PDF）+ 元数据销** | Crossref pp 63-74 / 作者 / DOI 全核；PDF 多通道不可达（§7.3） |
| rail-optimized NVIDIA blog（spec 10.1 ⟨verify〉） | **销** | IB cluster topology doc 逐字 + NCCL 2.12 blog（§7.1） |
| SHARP docs（spec 10.2 ⟨verify〉） | **销 + 版本漂移** | SHARP v3.17.0 UM + NCCL 集成页 + plugin repo（§6.5） |
| DeepEP fresh tarball 双锚（spec 10.3 ⟨verify〉） | **销** | v1.2.1 + main V2.5 全量 file:line（§5） |
| Ring Attention arXiv（spec 10.4 ⟨verify W5 再核〉） | **销** | export API v4 + 官方 repo haoliuhl/ringattention（§8） |
| rdma-core+=9.3 / deepep+=10.4 citedBy（W4 残留） | **销** | 本轮回写（§11） |
| PCI-SIG 精确出处（W0→W4 残留） | **残留不变** | 403；9.2 PCIe Gen5 数值用 W4 已核 Hopper wp/datasheet |
| JEDEC HBM3 精确 URL（W1 残留） | **残留不变** | 403，非本波 scope |
| B200 SM 数（W1 残留 W6） | **残留不变** | 非本波 scope |
| IBTA 全文 member-gated | **残留** | 公开 what's-new + landing FAQ + rdma-core/verbs manual 互证；全文需会员 |

## 10. 版本漂移注（writer 必注）

1. **DeepEP V1 vs V2.5（最重要）**：spec 大纲 normal 两阶段/LL 纯 RDMA hook IBGDA = **V1 语义（v1.2.1 锚）**；当前 main V2.5 = 统一 EPBuffer（hybrid/direct）+ NCCL Gin + NVSHMEM 全移除 + **EP 占 SM（zero-SM RDMA EP 不支持）**。机制讲解锚 v1.2.1、接口走读锚 main，**禁把 V1 当当前 main 接口**。V2.5 "IBGDA" 仅存注释（hybrid_dispatch.cuh L441/444），实为 **NCCL Gin GDAKI**（DOCA GPUNetIO）。
2. **DeepEP LL NVLink 节点内（v1.2.1 内漂移）**：2025.06.05 #173 起 LL 节点内可用 NVLink（allow_nvlink_for_low_latency_mode）；「纯 RDMA」指跨机路径。
3. **gdrcopy API 名漂移**：gdr_copy_to_bar/from_bar → **gdr_copy_to_mapping/from_mapping**（master 0d85ee38，post-v2.7）。
4. **peer-memory repo 名**：spec 所查 peer-memory 404 系名称不存在；legacy off-tree = **Mellanox/nv_peer_memory**（200，R470+ deprecated）；现行 nvidia-peermem 内置 GPU 驱动。
5. **SHARP 版本**：W0 锚 SHARPv300 → **v3.17.0**（legacy URL 仍 200）；Quantum-2 SHARPv3 / XDR SHARPv4。
6. **NVIDIA networking docs 门户漂移**：docs.nvidia.com/networking/display/* 旧 Confluence → **networking-docs.nvidia.com** 新门户（__pagetree.json 通道）；verbs manual web deprecated → DOCA（v1.7 PDF 仍权威）。
7. **ConnectX-7 双口口径**：datasheet portfolio「dual-port NDR (400Gb/s)」营销歧义；**HW manual SKU 表权威：双口 QSFP112 = NDR200/port，max total 400G**；800G = CX8。
8. **CX8 PCIe 代际**：CX8 原生 **Gen6 @64GT/s x16**（128 GB/s/方向）；Gen5 主机（H100/B200/GB200）下 800G NIC > 64 GB/s PCIe → PCIe 限；Gen6 主机下 NIC 线速限——regime 须注主机代际。
9. **Quantum 吞吐口径**：Quantum-2「51.2 Tb/s bidirectional」（双向）vs XDR Q3400「Switching 115.2 Tbps」（单向，144×800G）——writer 须注口径差。
10. **NCCL 版本**：repo 2.32.7-1 / 351f33fd（W4 同 commit 未漂移）；docs 2.32.3 滞后注维持；net_ib 为目录（含 gdaki/doca-gpunetio Gin 后端）。
11. **Ring Attention 官方 repo 改名**：lhao499/llm_large_context → **haoliuhl/ringattention**（arXiv abs 链接 301 redirect）；JAX 实现（非 NCCL）。
12. **Al-Fares PDF 不可达**：元数据 Crossref 已核；fat-tree 公式按 spec 独立重推，精确逐字待 PDF（残留）。
13. **IBTA 全文 member-gated**：Vol 1 Release 2.1（2294p）/ Vol 2（835p）；公开 what's-new PDF + landing FAQ 一手；RC/UC/UD 经 rdma-core/verbs manual 互证；CC Annex 不标编号。

## 11. references 写入 / 被引回写（spec §5 Sources）

- **新增 8 条**：
  - P9 [C] × 4：**connectx7**（CX7 datasheet + HW manual，被引 9.1, 9.2）/ **connectx8**（CX8 SuperNIC 800G + Gen6，被引 9.2, 10.1）/ **ib-cc-whitepaper**（IB CCA FECN/BECN/CCT，被引 9.3）/ **rdma-programming-manual**（verbs manual v1.7，被引 9.1）。
  - P10 [C] × 3 + 教学参照 × 1：**quantum-switch**（Quantum-2/XDR，被引 10.1, 10.2）/ **ib-cluster-topology**（rail-optimized 逐字，被引 10.1）/ **nccl-all2all-blog**（NCCL 2.12 PXN/rail，被引 10.2, 10.3）/ **nccl-rdma-sharp-plugins**（CollNet/SHARP plugin，被引 10.2）。
- **更新条目（核实日期 2026-11-25）**：P9 ibta-spec（Vol 1/2 Release 2.1 + what's-new PDF + MRC Annex 21 + member-gated 注）/ nvidia-networking-docs（新门户 + OFED 24.10 LTS 子页 + verbs manual 漂移）/ rdma-core（v67.0 + verbs.h 锚 + **citedBy += 9.3**）/ open-gpu-kernel-modules（620.73.04 peermem 锚 + peer-memory 状态销）/ gdrcopy（API 漂移 + 0.1μs）/ dcqcn（式 (1)-(4) + pp 523-536）；P10 al-fares（Crossref 再核 + PDF 残留）/ sharp-docs（v3.17.0 + SAT + NCCL 集成）/ deepep（双锚 file:line + **citedBy += 10.4**）/ ring-attention（arXiv 再核 + 官方 repo）；P8 nccl-tests（busbw 公式 + 日期）/ nccl-docs（**citedBy += 10.2**）/ nccl（W5 topo/net_ib/gin 锚注，被引已含 9.2/10.2/10.3/10.4）。
- **跨 Part 被引回写**：pcie-spec += 9.2；hopper-whitepaper += 9.2（PCIe Gen5 p.49 + HBM 3.35）；nvidia-datasheet += 9.2（HBM 3.35）；cuda-programming-guide += 9.2（ACS/IOMMU §3.4.2.5）；gpudirect-rdma-docs（P7，被引 7.3, 9.2 已含，补 BAR1 256MB/16GB + nv_peer_memory 名 + R470）；gb200-nvl72（P7，被引 10.1 已含）。
- **W4 残留修**：rdma-core citedBy 9.1 → **9.1, 9.3**；deepep citedBy 9.2, 10.3 → **9.2, 10.3, 10.4**。
- parse-check：11 groups / **84 entries**（76 + 8）。

## 12. writer 白名单提示（T2-T5）

- **9.1 RDMA 原理**：RDMA 定义/三语义（verbs manual §2.3 逐字）；RC/UC/UD（verbs manual §3.2 + rdma-core verbs.h L934-937）；操作×传输表（RDMA Read/Atomic 仅 RC）；QP/CQ/MR/WR（rdma-core verbs.h 锚）；QP 状态机 RESET→INIT→RTR→RTS（man ibv_modify_qp.3 L128）；IB vs RoCE（OFED RoCE 页：无 SM/LID=0/GID/APM 不支持 + IBTA RoCE 定义）；IB 链路 credit 流控 + VL（IB CC whitepaper）；ConnectX-7 NIC 硬件（§3.4）；RDMA vs TCP/IP 延迟账（kernel bypass 省上下文切换 + 拷贝，verbs manual 逐字）。**fabric → 10.1，GDR → 9.2 指针**。
- **9.2 GPUDirect RDMA【重】**：inference 8.3 KV 传输 recap + 指引；GDR 机制（GDR docs root complex/IOMMU/BAR1）；nvidia-peermem（§3.2 全锚，nv_mem/acquire/dma_map/free callback/双客户端）；gdrcopy（§3.3，gdr_copy_to_mapping ~0.1μs vs cudaMemcpy 6-7μs）；**NIC→HBM 带宽账**（§3.4 regime：单口 CX7 50 GB/s NIC 限 / CX8 800G PCIe Gen5 限 / 多 NIC 聚合 / vs HBM 3.35 TB/s；$\min(\mathrm{BW}_{\mathrm{NIC}},\mathrm{BW}_{\mathrm{PCIe}})$；BAR1 256MB/16GB 约束）；GDR vs bounce（省 H2D+D2H 两跳）；NCCL NET transport（8.4 recap，net.cc proxy + gdr.cc nv_mem sysfs + NCCL_NET_GDR_LEVEL）；DeepEP LL 纯 RDMA hook（**v1.2.1 锚**，§5.1，10.3 双锚回链）；ACS/IOMMU（cuda-programming-guide §3.4.2.5，W4 逐字）。
- **9.3 拥塞控制与可靠性**：DCQCN（§4.1 式 (1)-(4) + RP/CP/NP + N=50μs/K=55μs + F=5 + 无 slow start）；PFC（802.1Qbb + storm/victim flow/congestion spreading，DCQCN §2.2 + OFED PFC 页）；ECN（OFED ECN 页 RoCEv2 RP/NP sysfs）；**IB CC**（§4.2 CCA FECN/BECN/CCT/CCTI/CCM + credit 流控，IBTA CC Annex 不标号）；可靠性（IB reliable connection 重传 NCCL_IB_TIMEOUT=20/RETRY_CNT=7 + CRC/链路降级）；GPU 侧影响（拥塞 → β_eff 降 → 通信账恶化）；调参概念（ECN threshold/PFC priority/rate limiter → 10.1/10.2）；NIC 卸载（OFED Programming 页 TSO/checksum/packet pacing）。
- **10.1 fabric 架构与拓扑**：IB/RoCE fabric（SM/LID/GID，IBTA + OFED RoCE + OpenSM）；**拓扑族**（fat-tree/Clos 无阻塞，Al-Fares [B] 元数据 + 公式独立重推；rail-optimized §7.1 逐字；dragonfly/torus 对照）；**收敛比带宽账**（oversubscription、fat-tree 全二分条件、rail-optimized 集合带宽优势）；路由（ECMP/AR，NCCL_IB_ADAPTIVE_ROUTING IB 默认开/RoCE 默认关 + IB LID forwarding/RoCE UDP src port entropy）；多轨多 NIC（每 GPU 配 NIC、rail 对齐、NCCL 多 channel 跨 NIC，NCCL_IB_HCA rail/plane 语法 + NCCL_IB_RAIL_POLICY）；子网管理（SM 发现/配置/故障恢复，IBTA SM + IB bring-up doc + OpenSM）；**fabric 规模账**（端口 radix → 最大 GPU 数：Quantum-X800 2 级 fat-tree **10,368 NICs**（144²/2）官方一手 + Quantum-2 64 ports；交换机层级数）；GB200 NVL72 超节点 vs fat-tree（gb200-nvl72 W4 锚，NVLink 域 130 TB/s vs IB 每 GPU 50/100 GB/s）。
- **10.2 拓扑感知集合与网内聚合**：8.4 NCCL 机制 recap + 指引；**拓扑感知 ring/tree**（NCCL topo.cc NCCL_TOPO_FILE L2006 + paths.cc PXN intermediate L319-378 + connect.cc connectRings L105，跨节点环映射 rail、节点内 NVLink 环 + 机间 IB 环分层）；**SHARP**（§6.5 v3.17.0 网内聚合 overview + SAT threshold 16384 + 多 aggregation tree + NCCL-SHARP plugin + 每 HCA rail 一条 streaming flow leaf 同 rail）；NVLS（NVLink SHARP，7.1/8.4 recap）vs IB SHARP（跨机）；CollNet（coll_net.cc sharedBuffersInit L416 + NCCL_COLLNET_ENABLE）；大规模部署（多 channel/多 NIC/NCCL_TOPO_FILE/NCCL_ALGO/NCCL_PROTO 调优）；多作业干扰（fabric 带宽共享、拥塞 9.3 recap）；**万卡 all-reduce 实测账**（nccl-tests busbw 公式 §6.6，讲测量方法 + 引官方/已发表数值，非新跑）；models 4.5 重判（带宽账 → 8.1/10.3，拓扑 → 本章）。**关键推导**：分层 ring 延迟（节点内 + 机间）；SHARP 流量降；多 NIC 聚合。
- **10.3 EP all-to-all 实现【模块走读】**：training 5.4 EP 训练账 + inference 3.3/7.4/8.1 serving recap + 指引；**DeepEP 双锚**（§5：机制 v1.2.1 / 接口 main V2.5，fresh tarball file:line）；normal 两阶段（v1.2.1 internode.cu WarpRole kRDMASender→kRDMAAndNVLForwarder，NVLink 域聚合再跨机）；LL 纯 RDMA hook（v1.2.1 internode_ll.cu SEND/RECV phase + ibgda_device.cuh，GDR 9.2 recap，预注册 buffer）；**V2.5 接口增量**（统一 EPBuffer hybrid/direct + NCCL Gin + NVSHMEM 移除 + EP 占 SM + do_cpu_sync + lb_* 原语 + Engram/PP/Bucket；**版本漂移注**）；dispatch/combine kernel token 路由；**a2a 拥塞**（all-to-all vs all-reduce 流量模式、incast、ECN/PFC 交互 9.3 recap）；EP>32 递减（inference 3.3 §5 命题 3 recap，7.4 兑现）；NVLink 域 vs 跨机 a2a 带宽差；EPLB 通信（inference 7.4 recap，lb_prefetch_weights/lb_reduce_grads）。**关键推导**：a2a 每 rank (n-1)/n·M（recap 8.1，nccl-tests AlltoAll factor）；normal 两阶段 vs LL 延迟账（v1.2.1 口径）；incast 拥塞条件。NCCL a2a 对照（ncclSend/Recv + PXN blog）。
- **10.4 点对点与序列并行通信**：models 4.6 Ring Attention 分块 + training 5.2 CP 账 recap + 指引；send/recv 点对点（NCCL ncclSend/ncclRecv，8.4 P2P transport recap）；**Ring Attention 通信调度**（ring pass KV 块轮转、compute-comm overlap、每步 send/recv 与 attention 重叠推导；论文 §8 abstract 逐字 + 官方 JAX repo 佐证）；CP 三模式（all-gather KV / ring pass / Ulysses all-to-all 对照，training 5.2 recap）；序列并行 transport 选择（NVLink 域 P2P vs 跨机 RDMA）；**overlap 条件**（t_comm ≤ t_compute per block → 完全隐藏）；拓扑映射（ring → 物理拓扑 10.2 recap）；models 4.6 重判到本章。DeepEP CP 对照（BucketBuffer all-gather/RS，§5.2）。