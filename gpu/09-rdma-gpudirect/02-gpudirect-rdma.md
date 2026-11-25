# 9.2 GPUDirect RDMA（GPUDirect RDMA）

> 单口 400G ConnectX-7 线速 50 GB/s、PCIe Gen5 x16 单向 64 GB/s——NIC 直访 GPU HBM 的有效带宽到底取哪个、瓶颈在 NIC 还是在 PCIe？换成 800G ConnectX-8 瓶颈为什么反而移到 PCIe？GDR 不经 host bounce buffer，省掉的 D2H/H2D 两跳值多少带宽与延迟？nvidia-peermem 怎么把 GPU 页注册给 NIC、BAR1 孔径又卡住多大的映射？本章交付五块：inference 8.3 KV 传输语义 recap（不重推）+ GDR 硬件路径与 peer memory 机制、NIC→HBM 带宽账（命题 1 min + regime 判定）、多 NIC 聚合（命题 2）、GDR vs bounce 延迟/带宽对比、GDR 限制（ACS/IOMMU/BAR1，命题 3）与 NCCL NET/DeepEP 集成 | 前置：9.1（RDMA verbs/QP/单边语义）、7.2（PCIe Gen5 x16 = 64 GB/s、NUMA、节点出口账）、7.3（GPUDirect 家族、bounce 两跳半谐波均值命题 1、PCIe 三道门）、8.4（NCCL NET transport 与 proxy 线程）、inference 8.3（KV 传输语义，recap 源）；下游 10.1（fabric/rail）、10.2（拓扑感知集合）、10.3（DeepEP a2a 走读，recap 本章 GDR） | 锚点：GPUDirect RDMA docs（[NVIDIA, gpudirect-rdma-docs](../references.md)，[C] 官方规格披露·一手权威，root complex/IOMMU/BAR1/nvidia-peermem 逐字）、open-gpu-kernel-modules（[NVIDIA, open-gpu-kernel-modules](../references.md)，教学参照 + 官方实现，tag 620.73.04，nvidia-peermem.c file:line）、gdrcopy（[NVIDIA, gdrcopy](../references.md)，教学参照 + 官方实现，master 0d85ee38，gdr_copy_to_mapping）、ConnectX-7/8（[NVIDIA, connectx7](../references.md) / [connectx8](../references.md)，[C]）、NCCL（[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，2.32.7-1 / 351f33fd）、DeepEP（[DeepSeek-AI, deepep](../references.md)，教学参照 + 官方实现，双锚 v1.2.1 / main V2.5）、Hopper whitepaper + datasheet（[NVIDIA, hopper-whitepaper](../references.md) / [nvidia-datasheet](../references.md)，[C]，PCIe Gen5 / HBM 3.35 TB/s）——官方规格与代码锚 2026-11-25 一手核实

## 1. 三个问题与本章口径

一张 400G NIC 线速 50 GB/s，一条 PCIe Gen5 x16 单向 64 GB/s——NIC 直接 DMA 到 GPU HBM，有效带宽是 50 还是 64，瓶颈在哪一段？换 800G NIC（100 GB/s）瓶颈为什么反而移到 PCIe？不经 host bounce buffer 省掉的两跳，值多少带宽、多少延迟？本章回答三个问题：GDR 的 NIC→HBM 带宽账怎么列、regime 怎么判（§4-5）？GDR 相对 bounce 省在哪、稳态带宽为什么可能持平（§6）？peer memory 怎么注册、BAR1 与 PCIe 三道门各卡什么（§3/§7）？

**本章口径**（章脊锁定）：主口径 **H100 SXM**（HBM 3.35 TB/s = 3350 GB/s，[NVIDIA, nvidia-datasheet](../references.md)，[C]）+ **单口 ConnectX-7 NDR 400G**（线速 50 GB/s 单向）+ **PCIe Gen5 x16 = 64 GB/s 单向**（Hopper wp p.49 逐字，recap 7.2 §5）；贯穿实例 $M = 1$ GB = $10^{9}$ B。带宽账取 **raw 峰值口径**（有效折扣两边同扣、比值结论不变，沿 7.3 §1）。对照口径：ConnectX-8 800G（100 GB/s）+ 主机 PCIe Gen5/Gen6 两代（regime 须注主机代际，§4）；多 NIC 聚合取 rail-optimized 每 GPU 一张 NIC（§5）。KV 容量对照引 inference 8.3 章脊 4.295 GB（32768-token，recap 不重推）。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**）； $M$ = 被传输消息总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s），下标 NIC / PCIe / HBM / GDR / D2H / H2D / serial / pipe / pool 标各段与各口径。局部符号： $n_{\mathrm{NIC}}$ = 参与聚合的 NIC 数（§5）。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；inference 8.3 的 $\kappa/L/T$ 几何本章不引入，KV 容量以文字数值 recap。

**边界声明**（recap 不重推）：inference 8.3 已交付的 KV 传输语义（粒度/方向/时序三自由度、push/pull、layer-wise 暴露时间账、egress/ingress 公式）本章只 recap + 指针，**不重推**；本章增量 = **GDR 硬件路径**（NIC 直访 HBM、peer memory、带宽账、BAR1）。inference 8.3 §5 边界声明把 RDMA 协议/NIC/PCIe 拓扑/NVLink 域/交换机/NCCL 指向 cluster 维度；重定界后归本维度——RDMA 协议与 QP 归 9.1，GDR 硬件路径归本章，PCIe 拓扑归 7.2/本章，NVLink 域归 P7，交换机与 fabric 归 10.1，NCCL 归 8.4/10.2。**代码 scope** = 机制片段级（nvidia-peermem/gdrcopy/NCCL net_ib 关键片段，非模块全量走读；DeepEP 全量走读归 10.3）。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. inference 8.3 KV 传输语义 recap：本章接哪一棒

PD 分离下 prefill 算出的 KV 要搬到 decode 卡，inference 8.3 已经把「什么时候传、谁发起、传多久」讲透——本章不重复那套语义，只接它留下的硬件问题：KV 在两端 GPU HBM 之间搬，NIC 凭什么不经 host 内存直访 HBM？

recap 三条已交付结论（inference 8.3，指针不重推）：其一，**传输三自由度**——粒度（full 一次性 / layer-wise 流式）、方向（push/pull，initiator 视角，`NIXL_WRITE`/`NIXL_READ`）、时序（serialized / overlapped）；layer-wise 是 overlapped 的前提。其二，**GDR 的定位**——inference 8.3 §5 逐字「GDR（GPUDirect RDMA）... RDMA NIC 绕过 host memory 与 CPU 拷贝栈、直接读写 GPU HBM 的 PCIe 直连数据路径」，并给了三条拷贝路径对照（SM kernel 拷贝占 SM / CUDA copy engine 细粒度差 / GDR 不占 SM 且 NIC 饱和 PCIe）；Dynamo NIXL 的 KV 传输是 **VRAM→VRAM 直传**（prefill HBM → decode HBM 不经 host）。其三，**多 NIC 池化**——Mooncake transfer engine 聚合带宽 $`\mathrm{BW}_{\mathrm{pool}} = \sum_j \mathrm{BW}_j`$（inference 8.3 式 (3)），4×400G = 200 GB/s。这三条是 serving 语义层的结论；本章增量是把「GDR 直访 HBM」从一句定位展开成**硬件机制 + 带宽账**：NIC 经什么路径直访（§3）、有效带宽取多少（§4）、多 NIC 怎么聚合（§5）、省掉 bounce 值多少（§6）、什么配置会挡掉它（§7）。inference 8.3 的暴露时间账（命题 1）与 egress 恒等式（命题 2）属 serving 预算层，本章不重推，只在 §6 引其 4.295 GB KV 数值做 BAR1 容量对照。

为什么要做这次边界重定界：inference 8.3 站在 serving 引擎视角，关心的是传输语义与预算（什么时候传、藏不藏得住、带宽够不够），硬件协议层当时指向 cluster 维度。但 GDR 的带宽账（min 取哪段、regime 怎么判）、peer memory 注册机制、BAR1 容量约束，本质是 GPU 侧硬件机制——与 NVLink 几何、PCIe 拓扑、NCCL transport 同属一个知识层，放在 cluster 会割裂「GPU 硬件→通信实现」的连贯线。重定界后这些归本维度：RDMA 协议与 QP 语义归 9.1（verbs 机制），GDR 硬件路径与带宽账归本章，PCIe 拓扑与 NUMA 归 7.2，NVLink 域归 P7，交换机与 fabric 工程归 10.1，NCCL 实现归 8.4/10.2。所以本章与 inference 8.3 是「语义层↔硬件层」的分工：serving 侧问「传多久不亏」，本章答「硬件能传多快、瓶颈在哪」，两层经 4.295 GB KV 与 50 GB/s 线速这类共享数值对接，不重复推导。

## 3. GDR 硬件路径：NIC 直访 HBM 与 peer memory

GDR 的「直接」直接在哪？bounce 路径与 GDR 路径画出来（跨机 GPU A → GPU B，发送侧）：

```
bounce 路径（非 GDR，NIC 不能直访 HBM）：
  GPU A HBM ──D2H──► host DRAM buffer ──NIC DMA 读──► NIC ──► fabric
     读 M      PCIe 跳1：写 M      PCIe 跳2：读 M        线速 M
  host DRAM 流量 = 写 M + 读 M = 2M；CPU/copy engine 参与 D2H 中转
  （接收侧对称：fabric ──► NIC ──DMA 写──► host DRAM ──H2D──► GPU B HBM，又 2M）

GDR 路径（NIC 经 PCIe 直访 HBM）：
  GPU A HBM ──────────NIC DMA 直读（经 BAR1）──────────► NIC ──► fabric
     读 M                  PCIe 单跳 M                    线速 M
  host DRAM 流量 = 0；D2H/H2D 两跳整个摘掉，CPU 不参与数据面
```

**定义**（GPUDirect RDMA）：所谓 GDR（GPUDirect RDMA），是指第三方 PCIe 设备（NIC / 视频 / 存储）经 PCIe 直接读写 GPU HBM、不经 host bounce buffer 与 CPU 拷贝的数据路径。GDR docs 逐字：「a technology ... that enables a **direct path for data exchange between the GPU and a third-party peer device** using standard features of PCI Express」（[NVIDIA, gpudirect-rdma-docs](../references.md)，[C]）。它是 7.3 GPUDirect 家族四成员之一（P2P / RDMA / Storage / Video），与 P2P 共享同一套 PCIe 拓扑约束（§7）。正例：H100 + ConnectX-7 同 root complex、IOMMU 关，NIC 凭注册的 GPU MR 直 DMA HBM；边界反例：IOMMU 开非 1:1 翻译时 GDR 不成立（§7），退回 bounce。

**peer memory 机制：GPU 页怎么注册给 NIC**。NIC 要 DMA 到 GPU HBM，须先拿到 GPU 物理页的 DMA 地址——这由 **nvidia-peermem** 内核模块桥接（R470+ 内置于 GPU 驱动；legacy off-tree 模块 = Mellanox/nv_peer_memory，R470+ deprecated；Mellanox/peer-memory 与 NVIDIA/peer-memory 这两个 repo 名则不存在，404）。走读 open-gpu-kernel-modules tag 620.73.04 的 `kernel-open/nvidia-peermem/nvidia-peermem.c`（696 行，[NVIDIA, open-gpu-kernel-modules](../references.md)，教学参照 + 官方实现）：

| 锚（file:line） | 机制 |
| --- | --- |
| `nvidia-peermem.c` L46 | `DRV_NAME "nv_mem"`——注册到 IB 子系统的 peer memory client 名（NCCL gdr.cc sysfs `/sys/kernel/mm/memory_peers/nv_mem/version` 互证，§8） |
| L198 / L216 | `nv_mem_acquire` → `nvidia_p2p_get_pages`：pin GPU 页、取物理页（「acquire return code: 1 mine, 0 - not mine」判定地址归 GPU） |
| L246 / L265 | `nv_mem_acquire_nc` → `nvidia_p2p_get_pages_persistent`：persistent 映射（无 invalidation callback） |
| L300 / L323 / L338 | `nv_dma_map` → `nvidia_p2p_dma_map_pages` + `sg_alloc_table`：GPU BAR 页 → NIC DMA 地址，scatterlist 交 IB 核心 |
| L129 | `nv_get_p2p_free_callback`：GPU 页失效回调 → 通知 IB 核心注销 MR（防 NIC DMA 到已释放页） |
| L570 / L601 | `ib_register_peer_memory_client`：legacy（invalidate callback）+ nc（persistent）双客户端注册到 IB 子系统 |

机制链：应用注册 GPU 缓冲（verbs `ibv_reg_mr`，9.1 §3）→ IB 核心经 peer memory client 调 `nv_mem_acquire` pin GPU 页 → `nv_dma_map` 把 GPU BAR 页映射成 NIC 可 DMA 的地址 → NIC 凭此直访 HBM。GPU 页释放时 free callback 通知 IB 注销 MR，防悬空 DMA。两条路径：legacy 带 invalidation callback（页失效即注销），nc 为 persistent（长驻映射，省反复 pin）。**部署事实**（GDR docs Known Issue 逐字）：「there is no service to automatically load nvidia-peermem. Users need to load the module manually」——须手动 modprobe，否则 GDR 静默退回 bounce。

这里要说清为什么「pin」不可省。NIC 的 DMA 引擎用的是总线地址，不是 CPU 虚拟地址；普通应用页可能被内核换出、或在内存整理时迁移，物理地址并不固定——若 NIC 拿到页表项后页被移走，DMA 会写到错误的物理位置，造成静默数据损坏。所以注册前必须把 GPU 页 pin 住（锁定在物理内存、禁止换出与迁移），并为 NIC 建立稳定的 DMA 地址。这正是 9.1 §3 MR 注册「pin 页 + 建地址翻译」在硬件侧的对应：host 内存注册 pin 的是 DRAM 页，GPU 内存注册经 nvidia-peermem pin 的是 HBM 页、再映射进 BAR1 孔径。两条路径的取舍也由此清楚：legacy 带 invalidation callback，GPU 页释放时能立即通知 IB 核心注销 MR，适合缓冲频繁分配/释放的场景；persistent 映射长驻，省去反复 pin/unpin 的开销（每次 pin 付 10 μs–1 ms，见下文 gdrcopy），适合 DeepEP 预注册 num_rdma_bytes 这类长驻通信缓冲，代价是映射存续期间页不能释放，否则须处理 free callback 竞态（GDR docs 记录了 persistent API 的 free callback 竞态修复）。

**gdrcopy：CPU 侧低延迟小拷贝补充**。GDR 的大流量由 NIC DMA 承载；CPU 到 GPU 的小消息/同步字另有低延迟路径——gdrcopy（[NVIDIA, gdrcopy](../references.md)，教学参照 + 官方实现，master 0d85ee38）。README 逐字：「A low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA」，把 GPU 内存做 **CPU 映射**（write-combining），CPU 直接 store 到映射即写 GPU。现行 API **`gdr_copy_to_mapping` / `gdr_copy_from_mapping`**（`include/gdrapi.h` L175/L177；**旧名 `gdr_copy_to_bar`/`gdr_copy_from_bar` 已移除**，版本漂移必注）；`gdr_open` L74 / `gdr_pin_buffer` L87 / `gdr_map` L152。延迟账（README 披露）：`gdr_copy_to_mapping` 约 **0.10 μs**（1–8 B，copylat），对照 cudaMemcpy 约 **6–7 μs** 开销——小拷贝快约 60–70×；pin 一次性 10 μs–1 ms（随缓冲大小）。方向不对称：H-D 快（write-combining），D-H 慢（GPU BAR 不可 prefetch，无 burst read）。NCCL 用 gdrcopy 放 RX tail 同步字（`GDRCOPY_SYNC_ENABLE`，8.4 §5 recap）——大流量 NIC DMA + 小同步字 gdrcopy，两者互补。

gdrcopy 的低延迟来自 write-combining 机制：CPU 把 GPU 的 BAR 空间映射成 write-combining（WC）内存类型，多个连续小 store 被 CPU 合并成一次突发 PCIe 写事务，不必逐次等待写完成应答——所以写一个 8 字节同步字只花约 0.10 μs，省掉了 cudaMemcpy 运行时的流调度、copy engine 描述符建立等固定开销（6–7 μs）。反向 D-H 慢，是因为 GPU BAR 空间不可 prefetch：CPU 读映射时 PCIe 读事务无法生成突发读（读请求须等 GPU 返回），延迟由 PCIe 往返主导，带宽远低于 H-D。这条不对称决定了 gdrcopy 的用途边界：它适合 CPU 向 GPU 写小同步字/控制消息（H-D 方向），不适合 CPU 大量读回 GPU 数据（D-H 方向）——后者应走 copy engine 或 NIC DMA。这也解释了为什么 NCCL 只把 gdrcopy 用于 RX tail 同步字（几字节、H-D），而不用它搬大数据：大数据的带宽账由 NIC DMA 承载（§4），gdrcopy 只补小消息的延迟账。

## 4. NIC→HBM 带宽账：min 与 regime 判定

GDR 有效带宽取 NIC 线速还是 PCIe 带宽？先给结论：取两者较小者，单口 400G CX7 是 NIC 线速限、不是 PCIe 限。下面推导这个 min 的结构来源。

**命题 1**（GDR 单 NIC 带宽）：NIC 经 PCIe 直访 GPU HBM 的稳态有效带宽为

$$
\mathrm{BW}_{\mathrm{GDR}} = \min\left( \mathrm{BW}_{\mathrm{NIC}},\ \mathrm{BW}_{\mathrm{PCIe}} \right) \qquad (1)
$$

符号表： $`\mathrm{BW}_{\mathrm{NIC}}`$ = NIC 线速（单向，GB/s）； $`\mathrm{BW}_{\mathrm{PCIe}}`$ = GPU/NIC 所挂 PCIe 链路单向带宽（Gen5 x16 = 64 GB/s）。**证明**（路线：把每字节路径拆成 NIC wire 段与 PCIe DMA 段，证两段串行、稳态取慢段，三步）：其一，GDR 路径上每字节须先经 PCIe 段（NIC DMA 读/写 HBM）再经 NIC wire 段（或反向），两段对同一字节串行。其二，稳态下快段产出在慢段前堆积、慢段喂不满则整条线速降到慢段——吞吐由瓶颈段决定，即 min（与 7.3 命题 1 流水口径同构，recap 不重推）。其三，NIC DMA 与 wire 在 NIC 内 cut-through 重叠，不另计串行跳，故有效带宽即两段较小者。证毕。

**regime 判定**（数值全一手，逐行 Python 重算见 §9 自查）：

| 配置 | $`\mathrm{BW}_{\mathrm{NIC}}`$ | $`\mathrm{BW}_{\mathrm{PCIe}}`$ | min | 瓶颈 regime |
| --- | --- | --- | --- | --- |
| 单口 CX7 NDR400 + Gen5 x16 主机 | 50 GB/s | 64 GB/s | **50** | **NIC 线速限**（50 低于 64） |
| 双口 CX7（2×NDR200）+ Gen5 | 50 GB/s（总 400G） | 64 GB/s | 50 | NIC 线速限（双口不翻倍，9.1 §7 口径订正） |
| CX8 800G + **Gen5** x16 主机（H100/B200/GB200） | 100 GB/s | 64 GB/s | **64** | **PCIe 限**（100 高于 64） |
| CX8 800G + **Gen6** x16 主机 | 100 GB/s | 128 GB/s | **100** | **NIC 线速限**（100 低于 128） |

关键裁定：**单口 400G ConnectX-7 = 50 GB/s 是 NIC 线速限，不是 PCIe 限**——50 低于 64，PCIe 还有 14 GB/s 余量，瓶颈在 NIC 线速。把 400G GDR 说成「PCIe 瓶颈」是错的。只有 NIC 线速超过 PCIe（CX8 800G = 100 高于 Gen5 x16 的 64）瓶颈才移到 PCIe；且 **regime 须注主机 PCIe 代际**——CX8 原生主机口 Gen6 @64 GT/s x16 = 128 GB/s/方向，Gen6 主机下 100 低于 128 又回到 NIC 线速限（[NVIDIA, connectx8](../references.md)，[C]）。数值出处：400G = 50 GB/s（400÷8， $10^{9}$ 口径）、800G = 100 GB/s（CX7/CX8 datasheet）；Gen5 x16 = 64 GB/s 单向（Hopper wp p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」，recap 7.2 §5）；Gen6 x16 = 128 GB/s/方向（CX8 HW manual）。

**vs HBM：瓶颈不在 HBM**。H100 HBM 3.35 TB/s = 3350 GB/s（datasheet），GDR 路径 50 GB/s——3350/50 = **67×**。HBM 带宽比 GDR 路径高约两个数量级，所以 GDR 的瓶颈牢固在 PCIe/NIC 段而非 HBM 段；NIC 直访 HBM 不会把 HBM 打满，HBM 余量充足。这解释了为什么 GDR 能「NIC 饱和 PCIe 而不占 SM」（inference 8.3 §5 recap）：搬运由 NIC DMA 引擎承担，HBM 侧带宽绰绰有余。

式 (1) 的结构有两点要澄清。其一，为什么取 min 而不是求和：NIC wire 段与 PCIe DMA 段是同一字节流必经的两段串行路径，不是可同时承载不同字节的并行通道——稳态吞吐由最慢段决定，快段的产出在慢段前堆积，整条线速被拖到慢段。这与 7.3 命题 1 的流水口径同构，也与 roofline「吞吐上限取瓶颈段」的结构同构（models 2.4 recap）：凡串行路径的稳态吞吐，都取瓶颈段而非各段之和。其二，为什么 NIC 的 DMA 与 wire 不另计两段串行：现代 NIC 用 cut-through 转发，DMA 引擎从内存读数据的同时即向 wire 发送，不必等整条消息读完再发——所以 NIC 内部 DMA 与 wire 重叠，有效出口带宽就是 NIC 线速（线速低于 PCIe 时）或 PCIe 带宽（PCIe 低于线速时），即 min。若 NIC 是 store-and-forward（读完整个消息再发），才会多一段串行跳、带宽进一步降到半谐波均值；ConnectX 是 cut-through，故按 min 账。这两点合起来说明：min 不是经验拟合，是「串行路径取瓶颈 + NIC 内 cut-through 重叠」两个机制的结构性结论。

> **坑**：把单口 400G GDR 的瓶颈判成 PCIe。触发条件：见「GDR 走 PCIe」就以为 PCIe 是瓶颈；现象：误以为升 PCIe 代际能提 400G GDR 带宽。事实：50 低于 64，NIC 线速限，升 PCIe 不动瓶颈；只有 NIC 线速超 PCIe（800G + Gen5）升 PCIe 才有收益。规避：先按式 (1) 判 regime，再决定优化哪一段（§4 表）。

## 5. 多 NIC 聚合：rail-optimized 的带宽账

单 NIC 50 GB/s，一台 8-GPU 服务器要喂饱跨机流量，聚合带宽从哪来？每 GPU 配一张 NIC（rail-optimized），聚合取各 NIC 之和——但有前提。

**命题 2**（多 NIC 聚合）：设 $n_{\mathrm{NIC}}$ 张 NIC 各经**独立 PCIe 路径**访存（无共享上行瓶颈），则聚合有效带宽

$$
\mathrm{BW}_{\mathrm{pool}} = \sum_{j=1}^{n_{\mathrm{NIC}}} \min\left( \mathrm{BW}_{\mathrm{NIC},j},\ \mathrm{BW}_{\mathrm{PCIe},j} \right) \qquad (2)
$$

符号表： $n_{\mathrm{NIC}}$ = 参与聚合的 NIC 数；下标 $j$ = 第 $j$ 张 NIC 及其 PCIe 路径。**证明**（路线：各 NIC 路径独立 → 流量可并行 → 求和，两步）：其一，每张 NIC 的 GDR 带宽按命题 1 取 min；其二，若各 NIC 挂独立 PCIe x16（不共享 switch 上行口/root complex 瓶颈），多张 NIC 的 DMA 并行互不争抢，聚合带宽即各路径之和。证毕。同构 NIC 各带独立 x16 时简化为 $`n_{\mathrm{NIC}} \times \min(\mathrm{BW}_{\mathrm{NIC}}, \mathrm{BW}_{\mathrm{PCIe}})`$。

全数字代入（rail-optimized 8-GPU，每 GPU 一张单口 CX7，各 Gen5 x16）： $8 \times 50 = 400$ GB/s 节点出口聚合——与 7.2 §7 节点出口账逐数闭合（recap）。对照节点内 NVLink 注入 8×450 = 3600 GB/s 单向，比值 3600/400 = **9×**（7.2 §7 recap）：跨机出口比域内注入低一个数量级，这是分层 ring（节点内 NVLink 环 + 机间 RDMA 环）与 rail 对齐的算术依据（10.1/10.2）。**边界反例**（命题 2 前提不成立）：若多张 NIC 共享同一 PCIe switch 上行口或同一 root complex 瓶颈，聚合被共享链路封顶，不能各按 64 独立取——例如 4 张 NIC 共享一条 Gen5 x16 上行，聚合上限 64 GB/s 而非 4×50 = 200。所以 rail-optimized 要求每 GPU 的 NIC 挂独立 PCIe 路径、各 rail 接不同 leaf switch（10.1 逐字），聚合才随 NIC 数线性扩展。inference 8.3 式 (3) 的 Mooncake 池化（4×400G = 200 GB/s）是同一求和结构在 serving 传输底座的实例，本章给出其 GPU 侧 PCIe 前提。

rail-optimized 的物理图景值得展开：8-GPU 服务器里每张 GPU 配一张 NIC（GPU0 配 NIC0、GPU1 配 NIC1，依此类推），每张 NIC 接不同的 leaf switch（NIC0 接 leaf0、NIC1 接 leaf1）——这条「rail」就是每张 GPU 的专属网络通道。好处有两层：PCIe 侧，每张 NIC 挂在本地 GPU 的 PCIe switch/root complex 下，DMA 路径彼此独立，命题 2 的「独立 PCIe 路径」前提成立，聚合才随 NIC 数线性扩展；网络侧，同号 GPU 跨机通信（GPU0 对 GPU0）只经各自 rail 的一跳 leaf，不必跨 rail，减少交换机跳数与流间干扰（10.1 逐字）。这正是 NCCL 跨节点环按 rail 映射（10.2）、DeepEP hybrid 模式「multi-rail 友好」（10.3）的硬件依据。反过来，若 NIC 与 GPU 不 rail 对齐（例如 GPU0 的流量要走 NIC3），数据须先经 NVLink 域搬到 GPU3 再从 NIC3 出口（NCCL PXN 路径，8.4/10.2），多一跳域内搬运——虽仍能聚合带宽，延迟与域内流量却增加了。所以「每 GPU 一张独立 PCIe 路径的 NIC + 各 rail 接不同 leaf」是聚合带宽账成立的物理前提，不是可选优化。

## 6. GDR vs bounce：省两跳的延迟与带宽账

GDR 摘掉 D2H/H2D 两跳，到底省多少？分带宽与延迟两笔，且须分串行/流水两种口径——稳态带宽可能持平，增益主要在延迟与 host 流量。

**带宽账（recap 7.3 命题 1，不重推）**：bounce 两跳串行端到端有效带宽 = 半谐波均值（7.3 式 (1)，并联电阻式），此处两跳 = D2H staging（PCIe Gen5 x16 = 64 GB/s）与 NIC egress（线速 50 GB/s，NIC DMA 与 wire cut-through 重叠计一段）。代入：

$$
\mathrm{BW}_{\mathrm{serial}} = \left( \frac{1}{64} + \frac{1}{50} \right)^{-1} = 28.07\text{ GB/s}, \qquad \mathrm{BW}_{\mathrm{pipe}} = \min(64, 50) = 50\text{ GB/s} \qquad (3)
$$

GDR 单跳 50 GB/s（命题 1）。所以**串行口径** GDR 对 bounce 快 50/28.07 = **1.78×**；**流水口径**（双缓冲稳态）bounce = min(64,50) = 50 = GDR，**稳态带宽持平**。计入接收侧对称 H2D 跳，bounce 端到端串行三跳：

$$
\mathrm{BW}_{\mathrm{serial,e2e}} = \left( \frac{1}{64} + \frac{1}{50} + \frac{1}{64} \right)^{-1} = 19.51\text{ GB/s} \qquad (4)
$$

GDR 端到端 ≈ 50 GB/s（两端 DMA cut-through），对 bounce 端到端串行快 50/19.51 = **2.56×**——这就是「省 D2H+D2H 两跳」的带宽侧定量（两跳 = 发送 D2H + 接收 H2D）。

**host 流量账**（recap 7.3 式 (2) 结构，扩到两端）：

$$
\mathrm{Traffic}_{\mathrm{host}}^{\mathrm{bounce}} = 4M, \qquad \mathrm{Traffic}_{\mathrm{host}}^{\mathrm{GDR}} = 0 \qquad (5)
$$

bounce 每侧 host buffer 写 $M$ + 读 $M$ = 2M，两侧合计 4M；GDR host DRAM 零流量。代入 $M = 1$ GB：每搬 1 GB 省 4 GB host DRAM 流量 + CPU/copy engine 中转。多 GPU 并发时 host 内存带宽是共享瓶颈，这笔节省是结构性的（与 7.3 §4 同论）。

**延迟账**：bounce 每消息额外付 staging 跳时间 $`M/\mathrm{BW}_{\mathrm{D2H}}`$（发送）+ $`M/\mathrm{BW}_{\mathrm{H2D}}`$（接收）+ 拷贝启动固定开销。代入 $M = 1$ GB、64 GB/s：每跳 15.625 ms，两跳 31.25 ms——GDR 全免。小消息下固定开销主导：cudaMemcpy 类拷贝约 6–7 μs 开销（gdrcopy README 披露），gdrcopy 映射拷贝约 0.10 μs（§3），小同步字快约 60–70×。所以 GDR 的增益分两层：**大消息**省 staging 带宽/延迟（串行 1.78×、端到端 2.56×）+ host 流量 4M→0；**小消息**省拷贝固定开销（μs 级）。流水稳态带宽虽持平，host 流量与 CPU 卸载与延迟三项结构性收益仍在——这与 7.3 §6 GDS「带宽增益有限、主战场在结构性收益」同论。

> **提点**：GDR 的价值不能只看稳态带宽。NIC 线速限 regime 下（400G）流水 bounce 也能到 50 GB/s，单看带宽 GDR 似无增益；但 host DRAM 4M 流量、CPU/copy engine 中转、staging 延迟三项是 bounce 永远付、GDR 永远免的——多卡并发抢 host 带宽时，GDR 把 host 从数据路径整个摘掉，聚合吞吐不被 host 内存带宽封顶。

把延迟账按消息大小拆开，能看到两个 regime。大消息（1 GB 口径）：带宽项主导，staging 两跳付 31.25 ms，GDR 省掉这一整段——但须注意流水稳态下 bounce 也能到 50 GB/s，所以大消息的稳态吞吐增益有限，主要增益落在 host 流量从 4 倍消息字节降到 0、以及 CPU 卸载（多卡并发时 host 带宽不再是共享瓶颈）。小消息（同步字、控制消息，KB 量级）：带宽项（消息字节除以 64 GB/s）可忽略，固定开销主导——bounce 每次 staging 拷贝付 cudaMemcpy 类 6–7 μs 启动开销，GDR 的 NIC DMA 没有这笔逐消息 CPU 拷贝启动；CPU 侧小写若走 gdrcopy 更只 0.10 μs。所以 GDR 的延迟增益在小消息最显著（省固定开销），带宽增益在串行大消息最显著（省 staging 跳）；流水大消息带宽持平，增益转向 host 流量与 CPU。这解释了为什么延迟敏感的 decode EP（DeepEP low-latency）特别强调 GDR + 纯 RDMA：它追求的正是小消息固定开销 regime，而非大消息带宽 regime。

## 7. GDR 限制：PCIe 三道门与 BAR1 容量

GDR 不是无条件成立——PCIe 拓扑三道门挡可行性，BAR1 孔径挡映射容量。三道门 recap 7.3 §5（不重推），此处登记 GDR 口径 + BAR1 增量。

**PCIe 三道门**（GDR docs + CUDA PG §3.4.2.5 逐字，recap 7.3 §5）：其一，**同上游 root complex**——GDR docs 逐字「the two devices must share the same upstream PCI Express root complex」，GPU 与 NIC 须同 root complex（或经 PCIe switch 直通），跨 socket 无直通。其二，**IOMMU 1:1**——GDR docs 逐字「incompatible with IOMMUs performing any form of translation other than 1:1 ... must be disabled or configured for pass-through」；CUDA PG 更硬：bare-metal IOMMU 开「to prevent silent device memory corruption」——后果是静默数据损坏，不是变慢。其三，**ACS 重定向**——ACS 把 P2P 流量强制上行经 root complex 再折返，降 bisection 带宽（性能门，非正确性门）。三道门判定表与 VM/Windows 例外见 7.3 §5，本章不重绘。

**BAR1 映射容量约束**（本章增量）。NIC DMA 到 GPU HBM 经 **BAR1** 孔径（Base Address Register 1，GPU 暴露给 PCIe 设备的内存窗口），同时可映射的 GPU 内存受孔径大小限制。

**命题 3**（BAR1 映射容量上界）：经 nvidia-peermem 同时注册供 NIC DMA 的 GPU 内存字节数不超过 BAR1 可用孔径：

$$
\mathrm{BAR1}_{\mathrm{usable}} = \mathrm{BAR1}_{\mathrm{size}} - \mathrm{reserved} \qquad (6)
$$

**证明**（路线：每映射页占 BAR1 地址空间 → 总映射 ≤ 孔径，两步）：其一，NIC 经 BAR1 窗口访 GPU HBM，每个 peer 映射页占用一段 BAR1 地址；其二，同时映射的总字节不能超过窗口可用大小，即式 (6)。证毕。GDR docs 逐字数值：「The default BAR1 size for Tesla-class GPUs is **256 MB**. Of that, **32MB are currently reserved** for internal use」→ 默认可用 256 − 32 = **224 MB**；「On some Tesla-class GPUs a large BAR feature is enabled, e.g. **BAR1 size is set to 16GB or larger**」→ large BAR 16 GB+。

全数字代入（KV 容量对照，inference 8.3 章脊 recap）：32768-token KV = 4.295 GB。默认 BAR1 可用 224 MB，4.295 GB = 4295 MB 远超 224 MB——4295/224 ≈ 19.2，须 **large BAR（16 GB+）** 才能整块映射，否则须分块注册/注销（≥20 块），每块 pin 付 10 μs–1 ms（gdrcopy §3）。所以大 KV 缓冲的 GDR 传输依赖 large BAR 配置；数据中心 GPU（H100 等）通常启用 large BAR，但**平台相关**，须以 `nvidia-smi -q` 的 BAR1 行核实，不能假设默认 256 MB 够用。

BAR1 孔径的物理意义：GPU 通过 Base Address Register（BAR）窗口把 HBM 暴露给 PCIe 设备，NIC DMA 到 GPU 内存时地址落在这个窗口里——窗口大小决定了能同时映射多少 GPU 内存供 peer 访问。默认 256 MB 是历史配置（早期 Tesla 卡 HBM 小、映射需求小）；现代数据中心 GPU 的 HBM 达数十 GB，KV/激活缓冲远超 224 MB，于是 large BAR（resizable BAR）特性把窗口扩到 16 GB 甚至更大。GDR docs 特意提醒 large BAR 可能给 BIOS 带来问题（「Large BARs can pose a problem for the BIOS」）——部分平台 BIOS 无法正确分配这么大的 PCIe 地址空间，需要固件支持并在 BIOS 里显式启用。所以「GDR 能不能映射大缓冲」不只是驱动问题（nvidia-peermem 是否加载），还是平台固件问题（large BAR 是否启用）；部署时须查 nvidia-smi -q 的 BAR1 大小行确认实际孔径，既不能假设默认值够用，也不能假设 large BAR 必然开着。这把 §7 的三道门（拓扑/翻译/重定向）之外又加了一道容量门：前三道决定 GDR 可不可行，BAR1 决定能映射多大。

> **坑**：假设 GDR 注册任意大 GPU 缓冲都成功。触发条件：默认 BAR1（256 MB/224 MB 可用）下注册数 GB KV 缓冲；现象：注册失败或被迫分块，pin 开销累积、传输退化为 bounce。规避：先查 BAR1 大小（nvidia-smi），大缓冲确认 large BAR 启用；IOMMU/ACS 配置一并核（§7 三道门），改配置后以带宽 + 数据校验双重核对 GDR 路径。

## 8. NCCL NET transport 与 DeepEP：谁在用 GDR

GDR 在集合通信与 EP 通信里怎么落地？NCCL 用 proxy 线程协调 GDR，DeepEP low-latency 用纯 RDMA hook——两者都建立在 §3-7 的硬件路径上。

**NCCL NET transport（recap 8.4 §5，不重推）**：跨机 transport 的 verbs 接口是 host 侧 API，GPU kernel 调不了，须 CPU **proxy 线程**替 GPU 向 NIC 提交请求、轮询完成（8.4 §5：`sendProxyProgress` net.cc L1327 / `recvProxyProgress` L1496，posted/transmitted/done 三计数器在 NCCL_STEPS=8 槽上轮转）。**GDR 改变的是 proxy 的职责**：GDR 下 proxy 只做「提交描述符 + 轮询完成」的**控制面**，数据面 NIC 直读 GPU HBM（零拷贝）；非 GDR 下 proxy 还要做 D2H 搬运的**数据面**（host bounce）。GDR 可行性探测：`net_ib/gdr.cc` L18 查 sysfs `/sys/kernel/mm/memory_peers/nv_mem/version`（即 nvidia-peermem 的 DRV_NAME，§3 互证）判模块是否加载；`paths.cc` `ncclTopoCheckGdr` L494 判 GPU-NIC 拓扑是否允许 GDR，`NCCL_NET_GDR_LEVEL` L542 控制启用阈值。NCCL docs 逐字：「By default, NCCL enables GPU Direct RDMA if the topology permits it」（`NCCL_IB_CUDA_SUPPORT`）；`NCCL_NET_GDR_READ` 控 send 侧 GDR read。gdrcopy 同步字（`GDRCOPY_SYNC_ENABLE` 默认 1，net.cc L338）让 proxy 把 RX tail 放 CUDA 内存、省一次 PCIe read flush（§3）。模块全量走读归 8.4，本章只登记 GDR 怎么被协商与推进。

proxy 线程为什么不可省，值得说透：NIC 的 verbs 接口（ibv_post_send / ibv_poll_cq）是 host 侧用户态 API，操作的是 host 内存映射的队列与 UAR 寄存器；GPU kernel 代码跑在设备上，调不了这些 host API——所以必须有一个 CPU 线程替 GPU 向 NIC 提交请求。GDR 没有消灭 proxy，而是把 proxy 从数据面降为控制面：非 GDR 时 proxy 要先把数据从 GPU HBM 拷到 host 缓冲（D2H，数据面），再 post isend；GDR 时 NIC 直接 DMA GPU HBM，proxy 只提交描述符 + 轮询完成（控制面），不再搬数据。「只控制面」的 CPU 开销远低于「数据面搬运」——这就是 GDR 在 NCCL 侧「不占 SM、降 CPU 负载」的机制。DeepEP V1 的 IBGDA 更进一步：GPU 直接构造 mlx5 WQE 并写 doorbell（ibgda_device.cuh），连 CPU proxy 的提交动作都省掉，RDMA 完全由 GPU kernel issue——这是「不占任何 GPU SM」的硬件前提（RDMA 后台跑，SM 只做计算）。但须再注：V2.5 已移除此路径、改 NCCL Gin GDAKI，且 EP 占 SM（见下文版本漂移硬注），不能把 V1 的零 SM 语义当当前接口。

**DeepEP low-latency 纯 RDMA hook（V1 机制，锚 tag v1.2.1）**：MoE decode 的 EP all-to-all 要极低延迟，DeepEP low-latency 模式用**纯 RDMA**（跨机路径）+ **hook 式通信-计算重叠**。机制锚 v1.2.1（[DeepSeek-AI, deepep](../references.md)，教学参照 + 官方实现，9af0e0d0）：`internode_ll.cu` dispatch L40 / combine L555，SEND/RECV phase 分离（`configs.cuh` L20-21 `LOW_LATENCY_SEND_PHASE 1 / RECV_PHASE 2`）；`ibgda_device.cuh` GPU 直构 mlx5 WQE（L63）+ `nvshmemi_ibgda_put_nbi_warp`（L339，单边 WRITE 语义，9.1 §4）——**IBGDA 让 GPU 直接 issue RDMA，免 CPU proxy**。`buffer.py` `return_recv_hook`（L535/L561-562 逐字：「the kernel will just do the RDMA request issues, but without actually receiving the data ... call the received hook to make sure the data's arrival」）：SEND phase 只 issue 请求不接收 → 返回 hook → 计算重叠 → 调 hook 触发 RECV phase。README L290 逐字：「the RDMA network traffic is happening in the background, **without costing any GPU SMs**」——预注册 buffer（`num_rdma_bytes`）+ GDR（§3）让 RDMA 后台跑、不占 SM。

**版本漂移硬注（必读）**：上述纯 RDMA hook / IBGDA / 不占 SM 是 **V1 语义（v1.2.1 锚）**。当前 main **V2.5**（f6aa38fe）已**全移除 V1**：统一 EPBuffer（hybrid/direct）+ NCCL Gin 后端（GDAKI，DOCA GPUNetIO）+ **NVSHMEM 不再依赖** + **EP dispatch/combine 占 SM（zero-SM RDMA EP 不支持）**；「IBGDA」在 V2.5 仅存注释，后端实为 NCCL Gin GDAKI。把 V1 hook 语义当当前 main 接口写 = 事实错误。本章引 V1 机制讲 GDR 硬件路径怎么被用，接口走读与 V2.5 增量归 10.3（双锚口径继承 inference 3.3）。另注：V1 内 2025.06 起 LL 节点内可用 NVLink（`allow_nvlink_for_low_latency_mode`），「纯 RDMA」指跨机路径。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| inference 8.3 KV 传输语义（三自由度/push-pull/layer-wise/多 NIC 池化）recap 不重推；本章增量 = GDR 硬件路径 + 带宽账；边界重定界归 gpu P7/P9/P10 | inference 8.3 §1/§5/式 (3) | §2 |
| GDR = NIC 经 PCIe 直访 HBM、免 host bounce；nvidia-peermem（nv_mem）acquire→pin GPU 页→dma_map→NIC DMA 地址，free callback 防悬空；须手动 modprobe | nvidia-peermem.c L46/L198/L300/L129 | §3 |
| gdrcopy CPU 侧小拷贝：gdr_copy_to_mapping（旧 to_bar 已移除）~0.10 μs vs cudaMemcpy 6–7 μs（60–70×）；H-D 快 D-H 慢 | gdrapi.h L175/L177；README | §3 |
| GDR 单 NIC 带宽 = min(NIC, PCIe)：单口 CX7 400G=50 低于 Gen5 x16 64 → NIC 线速限；CX8 800G=100 高于 64 → PCIe 限（Gen5）/ Gen6 128 又 NIC 限；vs HBM 3350 = 67× | 命题 1、式 (1) | §4、练习 1/2 |
| 多 NIC 聚合 = Σ min（独立 PCIe 路径前提）：8×50=400 GB/s 节点出口（recap 7.2 §7），vs NVLink 注入 3600 = 9×；共享上行封顶反例 | 命题 2、式 (2) | §5、练习 3 |
| GDR vs bounce：串行半谐波 28.07（1.78×）、端到端三跳 19.51（2.56×）、流水持平 50；host 流量 4M→0；延迟省 staging 两跳 31.25 ms/GB | 式 (3)(4)(5)，recap 7.3 命题 1 | §6、练习 4 |
| GDR 限制：同 root complex / IOMMU 1:1（bare-metal 静默损坏）/ ACS 降 bisection（recap 7.3 §5）；BAR1 可用 = size−reserved，默认 224 MB、large BAR 16GB+，4.295 GB KV 须 large BAR | 命题 3、式 (6) | §7、练习 2 |
| NCCL GDR：proxy 控制面（提交+轮询）vs 非 GDR 数据面（D2H）；gdr.cc nv_mem sysfs + ncclTopoCheckGdr；DeepEP LL 纯 RDMA hook = V1（v1.2.1），V2.5 移除占 SM | 8.4 §5 recap；deepep 双锚 | §8、10.3 |

## 练习

1.（计算 · [10]）regime 判定代入（命题 1，式 (1)）：(a) 单口 CX7 400G + Gen5 x16，min 多少、瓶颈哪段；(b) CX8 800G + Gen5 x16，min 多少、瓶颈哪段；(c) CX8 800G + Gen6 x16（128 GB/s/方向），min 多少、瓶颈哪段。校验：(a) 50 GB/s、NIC 线速限；(b) 64 GB/s、PCIe 限；(c) 100 GB/s、NIC 线速限，与 §4 表逐行一致。
2.（计算 · [20]）BAR1 容量与 HBM 对照（命题 3，式 (6)）：(a) 默认 BAR1 256 MB、reserved 32 MB，可用多少；(b) inference 8.3 章脊 KV 4.295 GB = 4295 MB，默认 BAR1 下须分几块（向上取整）；(c) H100 HBM 3350 GB/s 对 GDR 50 GB/s 比值多少，瓶颈在 HBM 吗？先预测 (c) 再算。校验：(a) 224 MB；(b) ≥20 块（4295/224≈19.2）；(c) 67×、瓶颈在 PCIe/NIC 非 HBM，与 §4/§7 一致。
3.（推导 · [30]）证明命题 2（多 NIC 聚合，式 (2)）并给边界：(a) 证各 NIC 独立 PCIe 路径时聚合 = Σ min；(b) 代入 8×CX7 求节点出口、对 NVLink 注入 3600 GB/s 比值；(c) 给边界反例：4 NIC 共享一条 Gen5 x16 上行时聚合上限多少、为什么不能 4×50；(d) 说明 rail-optimized 怎么保证独立路径。校验：(a) 命题 2；(b) 400 GB/s、9×（recap 7.2 §7）；(c) 64 GB/s（共享链路封顶）；(d) 每 GPU NIC 挂独立 x16、各 rail 接不同 leaf（10.1）。
4.（计算 · [30]）GDR vs bounce 全账（式 (3)(4)(5)，recap 7.3 命题 1）： $M = 1$ GB，D2H/H2D = 64 GB/s、NIC = 50 GB/s。(a) bounce 发送侧串行半谐波带宽、GDR 对其倍数；(b) bounce 端到端串行三跳带宽、GDR 对其倍数；(c) 流水稳态两者各多少、是否持平；(d) 一次 1 GB GDR 比 bounce 省多少 GB host DRAM 流量、省多少 ms staging 延迟（两跳）。校验：(a) 28.07 GB/s、1.78×；(b) 19.51 GB/s、2.56×；(c) 均 50、持平；(d) 4 GB、31.25 ms，与 §6 逐数一致（容差 1%）；先预测 (c) 再算。

## 参考

- [NVIDIA, gpudirect-rdma-docs](../references.md)：GPUDirect RDMA docs + 家族页（key: gpudirect-rdma-docs），[C] 官方规格披露·一手权威，2026-11-25——GDR 定义逐字（direct path GPU↔third-party peer device via PCIe）；同 upstream root complex 逐字；IOMMU 非 1:1 不兼容逐字；BAR1 默认 256 MB（32 MB reserved）/ large BAR 16GB+ 逐字；nvidia-peermem R470+ 内置 / legacy nv_peer_memory deprecated / 无自动加载 service 须手动 modprobe（§3/§7）
- [NVIDIA, open-gpu-kernel-modules](../references.md)：NVIDIA/open-gpu-kernel-modules（key: open-gpu-kernel-modules），教学参照 + 官方实现，tag 620.73.04（a4b95f6a）contents API raw——kernel-open/nvidia-peermem/nvidia-peermem.c（696 行）：DRV_NAME nv_mem L46 / nv_mem_acquire L198→nvidia_p2p_get_pages L216 / acquire_nc L246→persistent L265 / nv_dma_map L300→nvidia_p2p_dma_map_pages L323 + sg_alloc_table L338 / nv_get_p2p_free_callback L129 / ib_register_peer_memory_client L570+L601；legacy off-tree = Mellanox/nv_peer_memory（§3）
- [NVIDIA, gdrcopy](../references.md)：NVIDIA/gdrcopy（key: gdrcopy），教学参照 + 官方实现，master 0d85ee38（post-v2.7）——gdrapi.h gdr_copy_to_mapping/from_mapping L175/L177（旧 gdr_copy_to_bar/from_bar 已移除）/ gdr_open L74 / gdr_pin_buffer L87 / gdr_map L152；README cudaMemcpy 6–7 μs / pin 10 μs–1 ms / copy_to_mapping ~0.10 μs / H-D write-combining 快 D-H 慢（§3/§6）
- [NVIDIA, connectx7](../references.md)：ConnectX-7 NDR 400G（key: connectx7），[C] 官方规格披露，2026-11-25——max total 400 Gb/s = 50 GB/s、单口 NDR 4×100G/lane、双口 QSFP112 = 每口 NDR200（2×200G=400G 总非 2×400G，HW manual SKU 表权威）、PCIe Gen5 up to x32（§4/§5）
- [NVIDIA, connectx8](../references.md)：ConnectX-8 SuperNIC（key: connectx8），[C] 官方规格披露，2026-11-25——up to 800 Gb/s = 100 GB/s、主机口 PCIe Gen6 @64GT/s x16 = 128 GB/s/方向（或 Gen5）；regime 须注主机代际（§4）
- [NVIDIA, nccl](../references.md)：NVIDIA/nccl（key: nccl），教学参照 + 官方实现，2.32.7-1 / 351f33fd fresh tarball——net_ib/gdr.cc nv_mem sysfs L18 / ncclIbGdrSupport L24；paths.cc ncclTopoCheckGdr L494 / NCCL_NET_GDR_LEVEL L542；net.cc sendProxyProgress L1327 / recvProxyProgress L1496 / GDRCOPY_SYNC_ENABLE L338；模块全量走读归 8.4（§8）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2.32.3（滞后 repo 2.32.7 注）——NCCL_IB_CUDA_SUPPORT「By default, NCCL enables GPU Direct RDMA if the topology permits it」逐字 / NCCL_NET_GDR_READ 控 send 侧 GDR read（§8）
- [DeepSeek-AI, deepep](../references.md)：deepseek-ai/DeepEP（key: deepep），教学参照 + 官方实现，双锚——机制 tag v1.2.1（9af0e0d0）：internode_ll.cu dispatch L40 / combine L555 / configs.cuh SEND_PHASE 1 RECV_PHASE 2 L20-21 / ibgda_device.cuh mlx5 WQE L63 / put_nbi_warp L339 / buffer.py return_recv_hook L535·L561-562 / README L290 不占 SM；接口 main V2.5（f6aa38fe）：NVSHMEM 移除 / EP 占 SM / EPBuffer + NCCL Gin GDAKI；版本漂移硬注 V1 语义≠current main，走读归 10.3（§8）
- [NVIDIA, hopper-whitepaper](../references.md)：H100 Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露，2026-11-25——PCIe Gen5 p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」（§1/§4，recap 7.2 §5）
- [NVIDIA, nvidia-datasheet](../references.md)：NVIDIA 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露，2026-11-25——H100 SXM HBM 3.35 TB/s = 3350 GB/s（§4 vs HBM 67×）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露，2026-11-25——Gen5 x16 = 64 GB/s 单向；URL 403 残留，数值经 Hopper wp p.49 互证闭合（§4）
- [NVIDIA, cuda-programming-guide](../references.md)：CUDA C++ Programming Guide 13.5（key: cuda-programming-guide），[C] 官方规格披露，2026-11-25——§3.4.2.5 IOMMU/ACS 逐字（bare-metal 不支持 + silent device memory corruption + ACS 重定向降 bisection），三道门主锚（§7，recap 7.3 §5）
- 9.1 / 7.2 / 7.3 / 8.4 / 10.1 / 10.2 / 10.3 / inference 8.3（纯文字指引，不挂链接）：RDMA verbs 机制（9.1）、PCIe/NUMA/节点出口（7.2）、bounce 半谐波命题 1 与三道门（7.3）、NCCL NET/proxy 走读（8.4）、fabric/rail（10.1）、拓扑感知集合（10.2）、DeepEP a2a 走读（10.3）、KV 传输语义 recap 源（inference 8.3）
