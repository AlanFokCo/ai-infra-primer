# 10.3 EP all-to-all 通信实现（EP All-to-All Communication Implementation）

> MoE 的 dispatch/combine 是两次 all-to-all——为什么它不能用 SHARP 网内聚合、必须自己搬字节？DeepEP 的 normal 两阶段（NVLink 域聚合 + 机间 RDMA）与 low-latency 纯 RDMA hook 各解决什么 regime，V2.5 为什么把两族统一成 EPBuffer 又移除了 NVSHMEM？EP 扩过 32 收益递减，GPU 侧 transport 瓶颈增量在哪、incast 拥塞条件怎么列？本章 recap training 5.4 EP 训练账与 inference 3.3/7.4/8.1 serving 语义（不重推），对 deepseek-ai/DeepEP 做**双锚模块走读**（机制 tag v1.2.1 / 接口 main V2.5），交付 transport/kernel 级增量：normal 两阶段 WarpRole 走读、low-latency hook 与 IBGDA、V2.5 统一 EPBuffer + NCCL Gin、a2a 消息量与拥塞账（命题 1/2/3）、大规模 EP 通信账 | 前置：training 5.4（EP 训练账，recap）、inference 3.3（DeepEP serving 语义 + EP 拐点命题 3，recap）、inference 7.4（Wide-EP/EPLB，recap）、inference 8.1（a2a 共享出口时间模型，recap）、8.1（a2a 消息量命题 1）、9.2（GDR，recap）、9.3（拥塞，recap）、10.1（fabric/rail）、10.2（rail 映射/SHARP）；下游 10.4（CP/Ring Attention） | 锚点：DeepEP（[DeepSeek-AI, DeepEP](../references.md)，教学参照 + 官方实现，**双锚**机制 tag v1.2.1 / commit 9af0e0d0 + 接口 main V2.5 / commit f6aa38fe，fresh tarball 全量 file:line）、NCCL（[NVIDIA, NCCL](../references.md)，教学参照，2.32.7-1 / 351f33fd，a2a 对照 + Gin 后端）、NCCL 2.12 blog（[NVIDIA, NCCL 2.12 blog](../references.md)，[C]）、ConnectX-7（[NVIDIA, ConnectX-7](../references.md)，[C]）、DCQCN（[Zhu et al., 2015](../references.md)，[B]）——代码锚 2026-11-25 一手核实

## 1. 三个问题与本章口径

MoE 一层前向要两次 all-to-all（dispatch 把 token 按路由发到 expert 所在卡、combine 把输出发回原卡加权归约）——为什么这两次不能用 10.2 的 SHARP 网内聚合，非得自己搬字节？DeepEP 的 normal kernel 走「NVLink 域内聚合 + 机间 RDMA 两阶段」、low-latency kernel 走「纯 RDMA + hook」，各对应什么消息 regime，当前 main 为什么把两族统一成 EPBuffer 还把 NVSHMEM 整个移除？EP 从 8 扩到 64，inference 3.3 已证步时拐点 $e^{\ast}=32$，GPU 侧 transport 的瓶颈增量到底是什么、热 expert 的 incast 拥塞条件怎么列？本章回答三个问题：DeepEP 双锚的 transport/kernel 实现长什么样（§4-6）？a2a 的消息量、延迟与拥塞账怎么立（§3/§7）？大规模 EP 的通信瓶颈与 EPLB 影响怎么算（§8）？

**本章口径**（章脊锁定，R2-2）：主口径 **H800 8-GPU 节点**（NVLink 有效 ~160 GB/s，DeepEP README 性能表口径）+ **每 GPU 一张单口 ConnectX-7 NDR 400G**（线速 50 GB/s，[NVIDIA, ConnectX-7](../references.md)，[C]）；模型几何取 **DeepSeek-V3**（hidden 7168、top-8、256 routed expert、58 MoE 层，training 5.4 recap）；精度 **FP8 dispatch / BF16 combine**（README 性能表口径）。normal 测试 **4096 token/rank**、low-latency 测试 **128 token/rank**（tests/test_internode.py L297 默认 4096 / test_low_latency.py，README 性能表）。延迟常数 $\alpha_{\mathrm{NVL}} = 2$ μs 为**教学假设**（沿 7.2/8.1）， $\alpha_{\mathrm{NET}}$ 取 RDMA 单跳 μs 量级（inference 3.3 §6 recap）；H100 对照行 NVLink 单向 450 / NIC 50 = 9×（7.2 §7 recap）。

**双锚口径声明**（强制，继承 inference 3.3）：DeepEP 机制讲解锚 **tag v1.2.1**（commit 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee，2025-09-15，V1 最后版本）；接口走读锚 **当前 main V2.5**（commit f6aa38fe06cd4a0489e7730365d4e0114970a2f3，2026-10-21）。**版本漂移硬注**：spec 大纲「normal NVLink+RDMA 两阶段 / low-latency 纯 RDMA hook IBGDA」是 **V1 语义**；当前 main V2.5 已统一 EPBuffer + NCCL Gin + NVSHMEM 全移除——**把 V1 机制当当前 main 接口写 = 事实错误**。V2.5 的「IBGDA」仅存注释，后端实为 **NCCL Gin GDAKI**（DOCA GPUNetIO）。本章 §4-5 讲 V1 机制（v1.2.1 锚）、§6 讲 V2.5 接口（main 锚），两处分开标注。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**）； $e$ = EP 度（expert 并行卡数，沿 inference， $e$ 与 $n$ 在 EP 组内同值）； $k$ = top-k（每 token 选中 expert 数）； $B$ = batch（ $B_{\mathrm{local}}$ 单 DP 组本地 batch，recap inference 沿用）； $d_{\mathrm{model}}$ = hidden 维； $M$ = 消息总字节（a2a 每 rank 输入总量，§3）； $\alpha_{\mathrm{NVL}}$ / $\alpha_{\mathrm{NET}}$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta = 1/\mathrm{BW}$ = 每字节传输时间； $\sigma$ = 偏斜度（ $\max_j R_j / \bar{R}$，training 5.4 recap，§7 局部）； $C$ = NIC 线速（§7 局部）。 $N$ 专归参数量、 $T$ 专归序列长，本章不占用；时间一律 $T_{\mathrm{comm}}$ / $T_{\mathrm{normal}}$ 下标。

**recap 边界声明**（强制，spec §4.2）：training 5.4（dispatch/combine 置换矩阵 $P$ 与加权转置、每 token $2kd$ 元素上界、命中分布 $\mathbb{E}[D]$、木桶 $\max_j R_j$、V3 EP64 两阶段 20 SM）、inference 3.3（normal/LL serving 语义、dispatch 字节 $Bk d_{\mathrm{model}} \mathrm{bytes}$、decode 逐 peer KB 延迟主导、EP 拐点命题 3 $\tau(e)=\max(A/e,F)+g(e)$）、inference 7.4（Wide-EP prefill 最小/decode 最大、EPLB 冗余副本与 moe_lb、EP>32 doc L295/L297）、inference 8.1（a2a 共享出口时间模型）、8.1（a2a 每 rank/每 peer 命题 1）、9.2（GDR 硬件路径与带宽 min）、9.3（DCQCN/PFC/IB CC 机制）——**均已交付，本章只 recap + 指引，不重推**。本章增量 = **transport/kernel 级实现**：DeepEP repo 双锚走读 + a2a 拥塞 GPU 侧账 + 大规模 EP transport 瓶颈。

**代码 scope 声明**：本章 = **模块走读级**（DeepEP 双锚：V1 normal/LL kernel + V2.5 EPBuffer/Gin 全量；spec §4.2 四个主走读章之一）。NCCL 只作 a2a 对照与 Gin 后端片段（机制片段级，全量走读归 8.4）。**测量口径**沿 7.2 / spec §6.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + DeepEP README 已发表性能表，非新跑 benchmark。

## 2. Recap：EP 通信账与 serving 语义

训练侧和 serving 侧已经把 EP 的账算到哪一步，本章又接哪一棒？答案是：语义与预算（dispatch/combine 是什么、消息多大、拐点在哪）两侧都已交付，本章只增量「字节怎么在 GPU/NIC 上搬」这一 transport/kernel 层。先把两侧已交付结论 recap 成表（不重推，指针回各源）：

| 已交付结论 | 源 | 本章 transport 增量 |
| --- | --- | --- |
| dispatch = $PX$、combine = $P^{\top}(W \odot F(X'))$；a2a 无归约语义、归约在接收后本地 | training 5.4 §1 | 为什么 SHARP 不适用（§3）、kernel 怎么搬（§4-6） |
| 每 token 每层 $2kd$ 元素上界；节点去重 $\mathbb{E}[D]$：卡级 7.6 / 节点 5.3 / 组限定 3.6 | training 5.4 §2/§6 | 两阶段 RDMA 流量节省（命题 1，§4） |
| 木桶：层时间 $\propto \max_j R_j$，偏斜乘性 | training 5.4 §4 | incast 拥塞条件（命题 3，§7） |
| normal 两阶段 / LL 纯 RDMA hook；decode 逐 peer KB 延迟主导 | inference 3.3 §6 | WarpRole / hook / IBGDA 代码实化（§4-5） |
| EP 拐点 $\tau(e)=\max(A/e,F)+g(e)$， $e^{\ast}=A/F=32$ | inference 3.3 §5 命题 3 | $g(e)$ 的 transport 成分（§8） |
| Wide-EP prefill 最小 / decode 最大；EPLB 冗余 + moe_lb 1.090 | inference 7.4 §1-3 | 跨机占比 $1-8/e$ + lb_* 原语（§8） |
| a2a 每 rank $(n-1)/n \cdot M$、每 peer $M/n$ | 8.1 命题 1 | EP 特化消息量（§3） |

serving 语义的关键 recap（inference 3.3 §6，不重推）：V3 normal kernel 两阶段——IB 50 GB/s 发到目标节点同序号 GPU、NVLink 160 GB/s 转发到目标 expert 所在 GPU，占 20 SM、与计算重叠；decode 需要另一族 kernel，因为消息小、延迟主导，两阶段多一跳延迟、带宽优势在小消息上兑现不出来。dispatch 字节 $`\mathrm{bytes}_{\mathrm{dispatch}} = B k d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}}`$（inference 3.3 式 (6)）；V4.1 decode 每卡每层 1.97 MB、摊到 64 rank 逐 peer 仅 30.7 KB，IB 50 GB/s 下带宽时间 0.61 μs——与 RDMA 单跳延迟同量级，延迟主导。**这就是 low-latency kernel 存在的定量依据**，本章 §5 走读它的 hook 实现。

## 3. a2a 消息量与流量模式：为什么 SHARP 用不上

先回答一个更基本的问题：all-to-all 和 all-reduce 都是集合通信，为什么 10.2 的 SHARP 网内聚合救不了 EP？因为**语义不同**。SHARP collective library 支持 Barrier/Allreduce/Reduce/Bcast/Reduce-scatter/Allgather（10.2 §4，sharp-docs）——全是**有归约或广播语义**的操作，交换机能把多份数据求和成一份。a2a 是纯数据搬运：每对 rank 互换的内容各不相同，无归约语义（training 5.4 §1 recap，combine 的加权求和在接收后本地做），交换机无从聚合，只能逐字节搬。所以 EP 的 dispatch/combine 必须走定制 a2a kernel（DeepEP），不能走 CollNet/SHARP。

**消息量账**（recap 8.1 命题 1，不重推）：每 rank 持 $M$ 字节切 $n$ 份，第 $j$ 份发 rank $j$，每 rank 总发送 $(n-1)/n \cdot M$、每 peer $M/n$（自环份留本地不发）。代入 $n=64$、 $M=1$ GB：每 rank **0.984 GB**、每 peer **0.015625 GB = 15.625 MB**。

**EP 特化**：dispatch 的每 rank 输入 $M$ 不是任意向量，而是 $B_{\mathrm{local}} \cdot k$ 个 (token, expert) 副本的隐状态。按逐副本口径（FP8，每元素 1 B + scale 4 B/128 元素 = 1.03125 B）：

$$
M_{\mathrm{dispatch}} = B_{\mathrm{local}} \cdot k \cdot d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}} \qquad (1)
$$

**符号**： $B_{\mathrm{local}}$ = 每 rank token 数（normal 测试 4096）； $k$ = top-8； $d_{\mathrm{model}} = 7168$； $`\mathrm{bytes}_{\mathrm{act}} = 1.03125`$（FP8 + scale）。代入： $4096 \times 8 \times 7168 \times 1.03125 = 242.2$ MB/rank（逐副本口径）。**但 normal 两阶段按节点去重**（§4 命题 1）：RDMA 上每 token 只发「命中的不同节点数」份，组限定路由期望 3.6 节点（training 5.4 §2 recap），故 RDMA 字节 ≈ $4096 \times 3.6 \times 7168 \times 1.03125 = 109.0$ MB/rank——逐副本 242.2 MB 对节点去重 109.0 MB，比值 **2.22×**（= $8/3.6$）。口径注：逐副本口径是 NVLink 域内转发量（每 expert 一份），节点去重口径是 RDMA 跨机量（每节点一份），两笔账落在不同链路（§4）。

**a2a vs all-reduce 流量模式**（recap 8.1 §4，不重推）：ring all-reduce 每 rank 只与 prev/next 两邻居通信、 $2(n-1)$ 步转发、邻居链路持续均匀负载；a2a 是完全图，每 rank 同时向 $n-1$ 个 peer 各发 $M/n$、单跳直达、无多步转发。压力差两条：其一，a2a 要**全 bisection 带宽**（任意对同时满带宽，10.1 fat-tree 1:1 无阻塞 recap），ring 只要邻居链路；其二，a2a 有**突发性与偏斜 incast**（热 expert 多对一同时涌入，§7）。NVSwitch 非阻塞对均衡排列线速（7.2 §4 recap），但偏斜路由打破排列均衡——这是 EP 拥塞的根源。

## 4. DeepEP normal 模式走读（v1.2.1 锚）：两阶段 WarpRole

normal kernel 怎么把「NVLink 域聚合 + 机间 RDMA」落成代码？入口是 `Buffer` 类（deep_ep/buffer.py L13，[DeepSeek-AI, DeepEP](../references.md)，教学参照，**v1.2.1 锚**），ctor（L33-37）持 `num_nvl_bytes` / `num_rdma_bytes`（两域预注册缓冲）、`low_latency_mode`、`num_qps_per_rank=24`、`allow_nvlink_for_low_latency_mode`；`num_sms=20`（L30，V3 占 20/132 SM recap training 5.4）。`dispatch`（L305）/ `combine`（L387）按域分派：intranode 走 `intranode_dispatch`（csrc/deep_ep.cpp L306）、internode 走 `internode_dispatch`（L654）。

**两阶段的核心是 warp 专化**。internode dispatch kernel（csrc/kernels/internode.cu L355-358，template<kLowLatencyMode, kNumRDMARanks, ...>）把 warp 分成五种角色（**L370-376** 逐字）：

```c
enum class WarpRole {
    kRDMASender,            // 阶段1：RDMA 发送到远端节点
    kRDMASenderCoordinator, // 阶段1 协调
    kRDMAAndNVLForwarder,   // 阶段2：RDMA 消费 + NVLink 生产
    kForwarderCoordinator,  // 阶段2 协调
    kNVLReceivers           // 阶段2：NVLink 接收
};
```

**目的地址的译法**钉死了两阶段语义：`translate_dst_rdma_rank`（L74-75）逐字 `return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;`——normal 模式（kLowLatencyMode=false）RDMA 目的译成 **rdma_rank（节点号）**，LL 模式译成具体 GPU。rank 分解 `rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS`（L100，NUM_MAX_NVL_PEERS=8，configs.cuh L3）。所以 normal 的 RDMA 发到「远端节点里与发送卡同 nvl_rank 的那张 GPU」，再由它转发。

```
normal internode dispatch 两阶段（源 node A.GPU_i → 目的 node B.GPU_j）
阶段1  kRDMASender（node A.GPU_i）
        nvshmemi_ibgda_put_nbi_warp ──RDMA──► node B.GPU_i 的 rdma_recv_buffer
        （目的 = 同 nvl_rank=i，每命中节点一份，节点去重）
阶段2  kRDMAAndNVLForwarder（node B.GPU_i）「RDMA consumers and NVL producers」
        从 rdma_recv_buffer 取 token ──NVLink──► node B.GPU_j（最终目的 expert 卡）
        kNVLReceivers（node B.GPU_j）收进本地 expert 输入缓冲
```

阶段1 `kRDMASender` 分支（L462）经 `nvshmemi_ibgda_put_nbi_warp`（L149/L672）发 RDMA put；阶段2 `kRDMAAndNVLForwarder` 注释逐字「**RDMA consumers and NVL producers**」（L690-691）、「**Forward tokens from RDMA buffer**」（L738）——从 RDMA 缓冲消费、向 NVLink 生产。sender warp 数 `kNumDispatchRDMASenderWarps = 7`（L1008）。combine 对称（NVLink 归拢 + IB 回传，csrc/deep_ep.cpp internode_combine L935）。

**命题 1**（两阶段节点去重的 RDMA 流量节省）。设每 token top- $k$ 命中的不同节点数为 $D_{\mathrm{node}}$、不同卡数为 $D_{\mathrm{card}}$。两阶段 normal 的 RDMA 副本数 = $D_{\mathrm{node}}$（每命中节点一份，节点内再 NVLink 分发）；逐 expert 直发 RDMA 的副本数 = $D_{\mathrm{card}}$。节省比 $D_{\mathrm{card}} / D_{\mathrm{node}}$。

**证明**（路线：数 RDMA 副本粒度，两步）。其一，两阶段先把 token 按目的节点去重（`rdma_idx` 对节点号 `inplace_unique`，tests/test_internode.py L50-52），每命中节点发一份到同 nvl_rank GPU，故 RDMA 副本 = $D_{\mathrm{node}}$；节点内 NVLink 转发到各 expert 卡（ $D_{\mathrm{card}}$ 份）走 NVLink 不占 RDMA。其二，逐 expert 直发则每命中卡一份 RDMA，副本 = $D_{\mathrm{card}}$；相比即得。∎ 代入 V3 组限定（training 5.4 §2 recap）： $D_{\mathrm{card}} \approx 7.6$、 $D_{\mathrm{node}} \approx 3.6$，节省比 $7.6/3.6$ = **2.11×**（逐副本 8 对 3.6 则 2.22×）。这就是「先聚合再跨机」省 RDMA 流量的定量——NVLink 域 ~160 GB/s 远高于 RDMA ~50 GB/s，把多份副本的搬运留在域内、跨机只发去重后的一份。

**性能锚**（README v1.2.1 性能表，已发表，H800 + CX7）：intranode EP8 dispatch/combine **153/158 GB/s**（NVLink）；internode EP16 **43/43**、EP32 **58/57**、EP64 **51/50 GB/s**（RDMA）。口径注：internode「Bottleneck bandwidth」= 节点去重 RDMA 字节 / 时间（tests/test_internode.py L179 `dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2`，FP8 因子 $(1+4/128)/2$，L194），即命题 1 的去重口径；EP64 51 GB/s 趋近 CX7 线速 50，说明两阶段把 RDMA 段喂满了。

## 5. DeepEP low-latency 模式走读（v1.2.1 锚）：纯 RDMA hook

decode 消息小、延迟主导，normal 两阶段多一跳 NVLink 延迟不划算——low-latency kernel 怎么把延迟压下来？答案是**纯 RDMA 单跳 + hook 重叠**。LL dispatch/combine kernel（csrc/kernels/internode_ll.cu L40-41 / L555-556，`__launch_bounds__(1024, 1)`）绕过两阶段转发，RDMA 直发目的 GPU（`translate_dst_rdma_rank` 的 LL 分支译成具体 GPU，§4）。

**hook = SEND/RECV 两相分离**。configs.cuh L20-21 逐字 `#define LOW_LATENCY_SEND_PHASE 1` / `#define LOW_LATENCY_RECV_PHASE 2`。dispatch kernel 按 phases 门控：SEND 相（L87「Sending phase」）只 issue RDMA 请求、不接收；RECV 相（L248「Receiving phase」）才收数据。接口 `return_recv_hook`（buffer.py L535 low_latency_dispatch / L600-602 low_latency_combine）逐字（L561-562）：「return a receiving hook if set. If set, the kernel will just do the **RDMA request issues, but without actually receiving the data**. You must call the received hook to make sure the data's arrival」。README L270 逐字「the actual tensor will not be received only if you call `hook()`」。机制：dispatch 返回 hook → RDMA 在后台搬、**不占 SM**（README L290「without costing any GPU SMs」，指接收侧）→ 计算侧跑 expert GEMM 之外的活 → 需要数据时主动调 `hook()` 触发 RECV 相。这是 pull 式重叠，供 double-batch overlap（inference 3.3 §6 recap）。

**IBGDA：GPU 直构 WQE，免 CPU proxy**。LL 的 RDMA put 经 `nvshmemi_ibgda_put_nbi_warp`（internode_ll.cu L164）/ 原子计数 `nvshmemi_ibgda_amo_nonfetch_add`（L231）。底层 ibgda_device.cuh：`mlx5_wqe_ctrl_seg` typedef（**L63**，GPU 直接构造 mlx5 work queue entry）、`nvshmemi_ibgda_put_nbi_warp`（L339）、`nvshmemi_ibgda_quiet`（L495，等 WQE 完成）。**IBGDA（InfiniBand GPU Direct Async）**让 GPU kernel 自己向 NIC 提交 WQE，绕过 8.4 §5 的 CPU proxy 线程——decode 小消息下省掉 GPU→CPU→NIC 的代理往返延迟。这是 GDR（9.2 recap：NIC 直访 GPU HBM）之上的「GPU 直发起」：GDR 解决数据面零拷贝，IBGDA 解决控制面零 CPU。预注册缓冲 `num_rdma_bytes`（buffer.py ctor）+ `num_qps_per_rank=24` 长驻，免逐消息注册（9.2 §3 persistent mapping recap）。

**节点内 NVLink 优化（v1.2.1 内漂移）**：News 2025.06.05 #173「low-latency kernels now leverage NVLink as much as possible」——LL 跨机仍纯 RDMA，但节点内自 2025.06 可用 NVLink 直连（`allow_nvlink_for_low_latency_mode`，buffer.py L96-99 经 NVSHMEM_DISABLE_P2P 控制）。代码实化：internode_ll.cu L162-166 先 `nvshmemi_get_p2p_ptr` 探目的能否 P2P，能则 NVLink 直写、否则 RDMA put。**口径注**：「纯 RDMA」指跨机路径；节点内 LL 可用 NVLink，勿把「纯 RDMA」读成全部路径。NVSHMEM 依赖（setup.py L19-25，NVSHMEM_DIR / nvidia.nvshmem wheel）是 V1 LL 的底座——V2.5 已移除（§6）。

**命题 2**（normal vs LL 每消息延迟差）。设每消息 $m$ 字节，RDMA 单跳延迟 $\alpha_{\mathrm{NET}} + m\beta_{\mathrm{NIC}}$，NVLink 转发跳 $\alpha_{\mathrm{NVL}} + m\beta_{\mathrm{NVL}}$。normal 两阶段每消息临界路径 = RDMA 跳 + NVLink 跳，LL 纯 RDMA = RDMA 跳，差：

$$
\Delta T = T_{\mathrm{normal}} - T_{\mathrm{LL}} = \alpha_{\mathrm{NVL}} + m \beta_{\mathrm{NVL}} \qquad (2)
$$

**证明路线**（完整推导留作练习 3）：normal 每消息须先 RDMA 到远端同 nvl_rank GPU（ $\alpha_{\mathrm{NET}} + m\beta_{\mathrm{NIC}}$）、再 NVLink 转发到目的卡（ $\alpha_{\mathrm{NVL}} + m\beta_{\mathrm{NVL}}$），两跳串行；LL 单跳 RDMA 直达（ $\alpha_{\mathrm{NET}} + m\beta_{\mathrm{NIC}}$）；相减即式 (2)。

全数字代入（ $\alpha_{\mathrm{NVL}}=2$ μs 教学假设、 $`\mathrm{BW}_{\mathrm{NVL}}=160`$ GB/s H800）：decode 逐 peer $m = 30.72$ KB（§2 recap）→ $m\beta_{\mathrm{NVL}} = 0.192$ μs， $\Delta T = 2.19$ μs ≈ $\alpha_{\mathrm{NVL}}$（带宽项可略）；prefill $m = 1$ MB → $m\beta_{\mathrm{NVL}} = 6.25$ μs， $\Delta T = 8.25$ μs。**读数**：小消息下两阶段多付的几乎全是一跳固定延迟 $\alpha_{\mathrm{NVL}}$，带宽优势兑现不出——LL 省这一跳赢延迟；大消息下两阶段靠节点去重省 RDMA 流量（命题 1）赢带宽。口径注：式 (2) 是每消息临界路径的教学模型；README LL 整 kernel 延迟（EP8 77 μs 等）含 fan-out 与组同步，非单消息 $\Delta T$，两口径勿混。

**LL 带宽重算**（Python 独立重算，对 README 已发表延迟）。LL 测试 128 token/rank、逐 expert 副本（不去重），每副本 FP8 字节 $7168 + 7168/128 \times 4 + 16 = 7408$ B（数据 + scale + 16 B 元信息，tests/test_low_latency.py L138），dispatch 总字节 $128 \times 8 \times 7408 = 7.586$ MB；combine BF16 每副本 $7168 \times 2 = 14336$ B、总 $14.68$ MB。带宽 = 字节 / 延迟：

| EP | dispatch 延迟（README） | 重算 GB/s | combine 延迟 | 重算 GB/s |
| --- | --- | --- | --- | --- |
| 8 | 77 μs | 98.5（README 98） | 114 μs | 128.8（127） |
| 64 | 173 μs | 43.8（43） | 314 μs | 46.8（46） |
| 256 | 194 μs | 39.1（39） | 360 μs | 40.8（40） |

逐数闭合（容差 ~2%，README 延迟/带宽各自圆整、重算按显示延迟反推，最大偏差 1.97%）。口径注：EP8 行是节点内（NVLink，#173 后），EP≥16 行跨机 RDMA；测试带宽按逐 expert 副本字节计（含 NVLink 段），非纯 RDMA 段。EP256 dispatch 39 GB/s 远低于线速 50——延迟主导（每 peer 消息小、fan-out 256 peer 同步链长），这是 §8 大规模 EP 瓶颈的读数。

## 6. V2.5 接口增量（main 锚）：统一 EPBuffer + NCCL Gin

当前 main 把 V1 两族 API 怎么了？答案是**统一 + 换后端 + 移除 NVSHMEM**。README L18 逐字（**main 锚**）：「Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation. **NVSHMEM is no longer a dependency**」。全 repo grep nvshmem 仅命中 README L18 这一句（py/cpp/hpp/cu/cuh/md 全无代码依赖）；ibgda_device.cuh / internode_ll.cu / buffer.py 已移除。

**统一 EPBuffer**。high-throughput 与 low-latency 统一成单 `EPBuffer`（deep_ep/buffers/ep.py **L208** `class EPBuffer(BufferBase)`，BufferBase 共享生命周期 base.py L4；C++ 侧 csrc/buffers/ep.hpp L23）。README L25-32 features 逐字：「High-throughput and low-latency APIs unified into a single EPBuffer interface... Analytical SM & QP count calculation — no more auto-tuning... Both hybrid & direct modes remain supported」。两模式（ep.py L217-219 docstring 逐字）：「**Hybrid mode uses hierarchical RDMA + NVLink communication** to achieve higher bandwidth, and is more friendly to **multi-plane/multi-rail networks**」——hybrid = V1 normal 两阶段的后继（节点去重 + rail 友好）；direct = 纯 RDMA（V1 LL 后继）。`dispatch`（L562）/ `combine`（L770）统一接口；训练/prefill vs decode 的区分移到参数 **`do_cpu_sync`**（README L160-162）：`True` 精确输出尺寸 + CPU 侧 expert 计数（接 contiguous grouped GEMM）；`False` GPU 侧接收计数 + 容量分配，expert GEMM 用 `handle.psum_num_recv_tokens_per_expert` 定有效段（接 masked，inference 3.3 §4 recap）。

**NCCL Gin 后端**。V2.5 的 RDMA 不再经 NVSHMEM IBGDA，而是 **NCCL Gin**（GPU-initiated networking）。comm handle（deep_ep/include/deep_ep/comm/handle.cuh **L21** `ncclGin gin;` 成员，L31 ctor `gin(nccl_dev_comm, qp_idx, resource_sharing_mode)`）；device API `gin.put<ncclTeamTagRail>`（hybrid_dispatch.cuh **L179-183** rail team put / **L445-451** token put）、`gin.put_value<team_t>`（dispatch.cuh L155-156）、`ncclGin_VASignalAdd`（handle.cuh L113）/ `gin.wait`（L123-125）。后端探测（csrc/kernels/comm/context.cpp **L126**）逐字 `(allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE`——hybrid 用 railed Gin、direct 用普通 Gin。NCCL 侧 `ncclGetGinType`（src/gin/gin_host.cc L28）/ `ncclGetRailedGinType`（L36），RDMA 实现在 **net_ib/gdaki/doca-gpunetio**（GDAKI = GPU Direct Async Kernel-Initiated，DOCA GPUNetIO）。

> **坑**：把 V2.5 注释里的「IBGDA」当 NVSHMEM IBGDA 读。触发条件：见 hybrid_dispatch.cuh L441「overlapping with the IBGDA issues」/ L444「Issue IBGDA requests」/ hybrid_combine.cuh L482「Issue IBGDA」就引 NVSHMEM；现象：这些注释指的已是 **Gin GDAKI** 的 GPU-initiated issue（紧邻代码是 `gin.put<ncclTeamTagRail>`），NVSHMEM 已全移除，照注释引 NVSHMEM 会张冠李戴；规避：「IBGDA」在 V2.5 仅存注释（grep 全 repo 只这三处 + README），后端实为 NCCL Gin GDAKI，机制讲解锚 v1.2.1 的 ibgda_device.cuh 才是真 NVSHMEM IBGDA。

**EP 占 SM（版本 + 范围漂移）**。README L39 Notes 逐字：「**EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported**」——V2.5 通信 kernel 占 SM，与计算争 SM（§8 $g(e)$ 成分）。**对照 V1**：V1 LL hook 接收侧不占 SM（v1.2.1 README L290「without costing any GPU SMs」，指接收侧后台 RDMA）。两者版本与范围都不同：V1 说的是 LL 接收侧、V2.5 说的是 dispatch/combine 操作整体——勿混读成「V2.5 比 V1 多占 SM」的简单退化（inference 3.3 §6 口径分层注 recap）。QP 数解析计算（ep.py L553-558）：direct `min(num_sms, 8+1)`（L554，少 QP 减 doorbell ringing）、hybrid `num_sms * 16 + 1`（L558，每 channel 独立 QP）——免 V1 auto-tuning。

**冗余 expert 原语与 buffer 家族**。`lb_prefetch_weights`（ep.py L854）/ `lb_reduce_grads`（L886）：计算前经 NVLink 预取冗余 expert 权重 + 量化 scale、反向归约冗余 expert FP32 梯度回原 expert（README L15，MoonEP/UltraEP 动机）——EPLB 冗余副本的通信原语（§8，inference 7.4 recap）。V2.5 还把 ElasticBuffer 拆成 EPBuffer / EngramBuffer（GPU/CPU 多层 RDMA fetch）/ PPBuffer（RDMA-only pipeline send/recv）/ BucketBuffer（CP/DP batched all-gather/RS/AR），共享 BufferBase + BufferAllocator（allocator.py L10）。Requirements（README L45-56）：Hopper+ / CUDA 13.1+（DeepJIT）/ C++20 / PyTorch 2.10+ / **NCCL 2.32.3+** / NVLink intranode / RDMA internode；`EP_DISABLE_GIN`（L406）只跳 Gin 初始化供 NVLink-only，「RDMA operations require Gin; this flag does not select another RDMA backend」。

**V1 vs V2.5 对照表**（版本漂移总账）：

| 维度 | V1（v1.2.1 锚） | V2.5（main 锚） |
| --- | --- | --- |
| API | Buffer 单类，normal/LL 两族 | 统一 EPBuffer（hybrid/direct） |
| RDMA 后端 | NVSHMEM IBGDA（ibgda_device.cuh） | NCCL Gin GDAKI（net_ib/gdaki） |
| 两阶段 | internode.cu WarpRole 五角色 | hybrid_dispatch.cuh gin.put<ncclTeamTagRail> |
| LL hook | internode_ll.cu SEND/RECV 相分离，接收侧零 SM | direct 模式，EP 占 SM（zero-SM 不支持） |
| SM/QP | num_sms=20 + auto-tuning | 解析计算（direct min(sms,9) / hybrid sms×16+1） |
| 训练/decode 区分 | normal vs LL kernel | do_cpu_sync True/False |
| NVSHMEM | 依赖（setup.py L19-25） | 全移除（README L18） |

## 7. a2a 拥塞行为：incast 与 ECN/PFC 交互

热 expert 为什么把网络堵死？因为 a2a 的偏斜把「均衡流量」打破成「多对一 incast」。recap 9.3 拥塞机制（不重推）：RoCE DCQCN（ECN 标记 + CNP 反馈 + 速率调节）、PFC（per-priority pause，HoL blocking / storm 风险）、IB CCA（FECN → BECN → CCT 包间延迟）。本节给 GPU 侧的 incast 定量条件。

**命题 3**（incast 拥塞条件）。设各发送 rank NIC 出口线速 $C$，均衡 a2a 是均匀流量（每 rank 收发均 $(n-1)/n \cdot M$，每 sender 出口摊到 $n-1$ 个 receiver，每 receiver 入口恰 = $C$，恰好线速、非阻塞 fat-tree 1:1 可承载，10.1 recap；均匀流量是双随机矩阵、非排列——排列流量精确定义见 10.1 命题 1）。设热 receiver 偏斜度 $\sigma = \max_j R_j / \bar{R}$（training 5.4 木桶 recap），各 sender 仍满出口 $C$ 但路由向热卡集中，则热 receiver 入口 = $\sigma C$；**拥塞当且仅当 $\sigma > 1$ 且持续**（入口超线速，队列累积）。

**证明**（路线：均匀流量入口 → 偏斜集中 → 超线速判据，三步）。其一，均衡时每 sender 把出口 $C$ 均分 $n-1$ 份、每份 $C/(n-1)$，热 receiver 收 $n-1$ 个 sender 各一份 → 入口 $(n-1) \cdot C/(n-1) = C$，恰线速。其二，偏斜使每 sender 发往热卡的比例升到 $\sigma/(n-1)$，热卡入口 $(n-1) \cdot C\sigma/(n-1) = \sigma C$。其三， $\sigma > 1$ 时入口 $\sigma C > C$ 出口，差额 $(\sigma-1)C$ 在热卡 NIC 入口（或 leaf 下行端口）队列累积 → 拥塞。∎ 边界反例： $\sigma=1$（完全均衡）入口 = $C$ 不累积，均衡 a2a 本身不拥塞——拥塞全来自偏斜，不是 a2a 语义固有。

全数字代入（ $C=50$ GB/s，教学缓冲 $Q=64$ MB）： $\sigma=1.09$（EPLB 后残差，inference 7.4 moe_lb 1.090 recap）入口 54.5 GB/s、超额 4.5 GB/s、填满 64 MB 需 $64/4.5 \approx 14.2$ ms（MB 除以 GB/s 得 ms）； $\sigma=2$ 入口 100、超额 50、填 64 MB **1.28 ms**； $\sigma=4$（training 5.4 §2 偏斜例 $R_j$ 8192/2048）入口 200、超额 150、**0.427 ms**。读数：偏斜越大队列填得越快，ECN 标记越早触发。

**ECN/PFC 交互**（9.3 recap，不重推机制）。队列涨过 ECN 阈值（ $K_{\min}$）→ 交换机标记 → 接收 NIC 发 CNP → 发送 NIC DCQCN 降速（ $RC = RC(1-\alpha/2)$，[Zhu et al., 2015](../references.md)，[B]，CNP 生成周期 N=50 μs）；若 PFC 启用，队列涨过 pause 阈值 → 上游 pause frame → HoL blocking，极端成 PFC storm。DCQCN 论文评估 16:1/32:1 incast（§5）——正是 a2a 热卡多对一的形态。**a2a 与 all-reduce 的拥塞差**：ring all-reduce 邻居链路持续均匀负载、无 incast（每 rank 只收 2 邻居）；a2a 完全图 + 偏斜 → 突发 incast。所以 EP 的拥塞治理两手：模型侧 EPLB 压 $\sigma$（inference 7.4，§8）、网络侧 ECN/PFC 调参 + AR 散列（10.2 §6/§7）。口径：本章管 GPU 侧 incast 条件与后果，DCQCN 速率收敛完整推导归 9.3。

## 8. 大规模 EP 通信账：EP>32 递减的 transport 增量

EP 扩过 32 为什么收益停？inference 3.3 命题 3 已证步时 $\tau(e) = \max(A/e, F) + g(e)$、拐点 $e^{\ast} = A/F = 32$（recap，不重推）；inference 7.4 生产口径 doc L295「Once memory bandwidth is no longer the bottleneck, a larger EP size introduces additional communication and synchronization overhead」+ L297 超节点多 job NVLink 干扰。本节增量： $g(e)$ 的 **GPU 侧 transport 成分**到底是什么。

**成分一：跨机占比随 $e$ 涨**。均衡路由下目的 expert 均匀落在 $e$ 张卡，本节点（8 卡）命中概率 $8/e$，跨机占比 $1 - 8/e$：

| $e$ | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| 跨机占比 $1-8/e$ | 0% | 50% | 75% | 87.5% | 93.75% | 96.88% |

$e=8$ 全 NVLink（~160 GB/s）， $e=64$ 近 88% 流量下 RDMA（~50 GB/s）——**EP 扩张把 a2a 流量从高带宽域搬到低带宽域**，这是 $g(e)$ 的主成分。带宽差：H800 NVLink 160 / NIC 50 = **3.2×**（README 口径），H100 450/50 = **9×**（7.2 §7 recap）；README 实测 intranode EP8 153/158 vs internode EP64 51/50 GB/s ≈ **3×**，与 3.2× 带宽差闭合。节点去重（命题 1）压的是每 token 跨机副本数，压不掉「跨机占比随 $e$ 涨」这个结构。

**成分二：逐 peer 消息缩小 → 延迟主导**。每 rank a2a 发送 $(n-1)/n \cdot M \to M$ 饱和，但每 peer $M/n$ 随 $n$ 缩—— $e=256$ 时每 peer 消息极小，fan-out 256 peer 的同步链长、每消息固定开销占比高（§5 LL EP256 39 GB/s 远低于线速 50 即此读数）。**成分三：QP 与 SM 占用**。V2.5 hybrid QP 数 `num_sms * 16 + 1`（§6）， $e$ 大 → channel/peer 多 → QP 与同步计数多；EP 占 SM（README L39）与计算争 SM。**成分四：偏斜粒度变粗**。inference 7.4 命题 2（recap）卡负载变异系数 $\mathrm{CV} \propto \sqrt{e/E}$， $e$ 大每卡 expert 少、相对偏斜大 → incast 更频繁（§7）。四成分合起来就是 $g(e)$ 单调不减的 transport 机制。

**EPLB 通信影响**（recap inference 7.4，不重推）。EPLB 把热 expert 复制多份摊到不同卡压 $\sigma$（moe_lb decode 1.090 → 吞吐 91.7%），代价是冗余副本的权重/梯度通信：`lb_prefetch_weights` 计算前经 NVLink 推原 expert 权重 + 量化 scale 到冗余槽、`lb_reduce_grads` 反向归约 FP32 梯度回原 expert（§6，ep.py L854/L886）。权重量级锚：V4.1 单 expert 每层 18.8 MB（MXFP4，inference 7.4 §2 recap），decode EP64 每 GPU 1 冗余 → 权重 shuffle 每 rearrange 周期（10000 步）搬一次、摊薄到每步小；但 `lb_prefetch_weights` 是每前向预取，NVLink 域内带宽高可藏进计算。口径：EPLB 用「冗余副本通信」换「incast 拥塞下降」，是 §7 拥塞治理的模型侧手段；引擎侧 shuffle（symm_mem/NCCL P2P）归 inference 7.4，本章只走读 DeepEP 原语层。

**NVLink 域 vs 跨机 a2a 的放置结论**： $e \le 8$ 收节点内（全 NVLink，3.2×/9× 带宽优势）； $e > 8$ 必跨机，靠两阶段节点去重（命题 1）+ rail 对齐（10.2 §3）+ EPLB 压偏斜把 $g(e)$ 压低；Wide-EP decode 取最大 $e$ 是为减每卡权重流量（inference 7.4 recap），通信代价由 Mega MoE 重叠（inference 3.3 §5 recap）与 EPLB 兜底。整超节点单 EP 组（doc L297）把多 job NVLink 干扰内部化——与 10.2 §7 fabric rail 隔离同构。

## 9. 边界：本章不做什么

transport/kernel 走读收口，剩下的归谁？本章交付：a2a 消息量与 SHARP 不适用（§3）、normal 两阶段 WarpRole（命题 1，§4）、LL hook + IBGDA（命题 2，§5）、V2.5 EPBuffer + Gin（§6）、incast 拥塞（命题 3，§7）、大规模 EP transport 账（§8）。以下各有专章：**EP 训练账**（置换矩阵/木桶/容量）归 training 5.4；**serving 语义与 EP 拐点命题**归 inference 3.3；**Wide-EP/EPLB 引擎**归 inference 7.4；**GDR 硬件机制**归 9.2；**拥塞控制完整机制**归 9.3；**fabric 拓扑/rail/SHARP**归 10.1/10.2；**NCCL 模块全量走读**归 8.4；**CP/Ring Attention**归 10.4。代码 scope：DeepEP 模块走读级（双锚），NCCL 机制片段级。

下游 recap 接口：10.4 recap §3 a2a 语义（Ulysses a2a 换维对照）+ §6 BucketBuffer（CP all-gather/RS）；9.2 已 recap 本章 LL 纯 RDMA hook（v1.2.1 锚，双锚回链）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| a2a 无归约语义 → SHARP 不适用（library 无 a2a）；每 rank (n-1)/n·M、每 peer M/n（n=64 → 0.984/0.015625 GB） | 8.1 命题 1 recap、§3 | §3、练习 1 |
| EP dispatch 每 rank 输入 B_local·k·d·bytes = 242.2 MB（逐副本）；节点去重 109.0 MB，比 2.22× | 式 (1)、命题 1 | §3/§4、练习 1/3 |
| normal 两阶段 WarpRole 五角色（internode.cu L370-376）；translate_dst_rdma_rank normal=节点/LL=GPU（L74-75）；kRDMAAndNVLForwarder L690/L738；sender warps 7 L1008 | v1.2.1 锚 | §4、练习 4 |
| 两阶段节点去重节省 D_card/D_node = 7.6/3.6 = 2.11×；README internode EP64 51/50 GB/s 趋线速 | 命题 1、README 表 | §4、练习 3 |
| LL hook = SEND/RECV 相分离（configs.cuh L20-21）+ return_recv_hook（buffer.py L561-562）；IBGDA GPU 直构 mlx5 WQE（ibgda_device.cuh L63/L339/L495）免 CPU proxy；接收侧零 SM | v1.2.1 锚 | §5、练习 4 |
| normal vs LL 每消息差 ΔT = α_NVL + mβ_NVL；decode 30.72 KB → 2.19 μs ≈ α_NVL；LL 带宽重算 EP8 98.5/EP256 39.1 GB/s 对 README 98/39 | 命题 2、式 (2) | §5、练习 2/3 |
| V2.5 统一 EPBuffer（ep.py L208）hybrid/direct；NCCL Gin（handle.cuh L21/gin.put<ncclTeamTagRail> hybrid_dispatch.cuh L179/L445）；NVSHMEM 全移除（README L18）；IBGDA 仅注释实为 Gin GDAKI | main 锚 | §6、练习 4 |
| V2.5 EP 占 SM（README L39 zero-SM 不支持）vs V1 LL 接收侧零 SM；QP 解析 direct min(sms,9)/hybrid sms×16+1；do_cpu_sync 分训练/decode | §6 对照表 | §6、inference 3.3 |
| incast 条件 σ>1：热卡入口 σC，超额 (σ−1)C；σ=1.09/2/4 填 64 MB 14.2/1.28/0.427 ms；ECN/PFC 交互（9.3 recap） | 命题 3 | §7、练习 3 |
| EP>32 递减 transport 四成分：跨机占比 1−8/e（e64=87.5%）+ 逐 peer 缩小 + QP/SM + 偏斜 CV∝√(e/E)；NVLink/RDMA 3.2×（H800）/9×（H100） | §8 表、inference 3.3 命题 3 recap | §8、练习 2 |
| EPLB lb_prefetch_weights/lb_reduce_grads（ep.py L854/L886）冗余权重/梯度通信换 incast 下降；18.8 MB/expert/层、moe_lb 1.090→91.7% | inference 7.4 recap | §8、练习 2 |

## 练习

1.（计算 · [10]）a2a 消息量与 EP 输入代入（8.1 命题 1、式 (1)）：(a) $n=64$、 $M=1$ GB 每 rank/每 peer 各多少 GB；(b) normal 每 rank dispatch 输入（ $B_{\mathrm{local}}=4096$、 $k=8$、 $d_{\mathrm{model}}=7168$、FP8 1.03125 B）多少 MB；(c) LL 每副本 FP8 字节（ $7168 + 7168/128 \times 4 + 16$）与 128 token × top-8 dispatch 总字节。校验：(a) 0.98438 / 0.015625 GB；(b) 242.2 MB；(c) 7408 B / 7.586 MB，与 §3/§5 逐数一致。
2.（计算 · [20]）normal vs LL 延迟与大规模 EP 账（命题 2、§8， $\alpha_{\mathrm{NVL}}=2$ μs、 $`\mathrm{BW}_{\mathrm{NVL}}=160`$ GB/s）：(a) decode 逐 peer 30.72 KB 的 $\Delta T$ 多少 μs、prefill 1 MB 的 $\Delta T$；(b) LL EP256 dispatch 带宽（7.586 MB / 194 μs）多少 GB/s、对 README；(c) $e=64/128$ 跨机占比 $1-8/e$；(d) EPLB moe_lb 1.090 对应吞吐比。先预测 (a) 小消息哪项主导再算。校验：(a) 2.19 / 8.25 μs（小消息 ≈ α_NVL）；(b) 39.1 GB/s（README 39）；(c) 87.5% / 93.75%；(d) 91.7%，与 §5/§8 一致（容差 1%）。
3.（推导 · [30]）证明命题 2（normal vs LL 延迟差，正文留作练习）并闭环命题 1/3：(a) 写 normal 两跳临界路径（RDMA $\alpha_{\mathrm{NET}}+m\beta_{\mathrm{NIC}}$ + NVLink $\alpha_{\mathrm{NVL}}+m\beta_{\mathrm{NVL}}$）与 LL 单跳，相减得式 (2)；(b) 说明小消息 $\Delta T \approx \alpha_{\mathrm{NVL}}$、大消息两阶段靠命题 1 节点去重赢带宽；(c) 证命题 1 节省比 $D_{\mathrm{card}}/D_{\mathrm{node}}$（数 RDMA 副本粒度）；(d) 证命题 3 incast 条件（均衡入口 $C$ → 偏斜 $\sigma C$ → $\sigma>1$ 累积），给 $\sigma=1$ 反例。校验：(a) 与 §5 式 (2) 一致；(c) 7.6/3.6=2.11×；(d) $\sigma=1$ 入口 $C$ 不累积，与 §7 一致。
4.（代码 · [30]）DeepEP 双锚走读（机制 v1.2.1 / 接口 main V2.5）：(a) v1.2.1 internode.cu WarpRole 五角色与 translate_dst_rdma_rank 的 normal/LL 分支；(b) v1.2.1 internode_ll.cu SEND/RECV 相门控与 ibgda_device.cuh mlx5 WQE；(c) main ep.py EPBuffer 与 hybrid_dispatch.cuh gin.put<ncclTeamTagRail>；(d) 全 repo grep NVSHMEM/IBGDA 验证移除（仅 README L18 + 三处注释）；(e) context.cpp Gin 后端探测。校验：(a) L370-376 / L74-75；(b) L87/L248 + configs.cuh L20-21 / ibgda_device.cuh L63；(c) ep.py L208 / hybrid_dispatch.cuh L179/L445；(d) NVSHMEM 仅 README L18、IBGDA 注释 hybrid_dispatch.cuh L441/L444 + hybrid_combine.cuh L482；(e) context.cpp L126，与 §4/§5/§6 一致。

## 参考

- [DeepSeek-AI, DeepEP](../references.md)：deepseek-ai/DeepEP repo（key: deepep），教学参照 + 官方实现，**双锚 fresh tarball 全量 file:line**（2026-11-25）。**机制锚 tag v1.2.1**（commit 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee，2025-09-15）：deep_ep/buffer.py class Buffer L13 / num_sms 20 L30 / ctor L33-37（num_nvl_bytes/num_rdma_bytes/low_latency_mode/num_qps_per_rank 24/allow_nvlink_for_low_latency_mode）/ dispatch L305 / combine L387 / internode_dispatch L435 / low_latency_dispatch L530-535 / return_recv_hook 逐字 L561-562 / NVSHMEM_DISABLE_P2P L96-99；csrc/deep_ep.cpp intranode_dispatch L306 / internode_dispatch L654 / internode_combine L935 / low_latency_dispatch L1091；csrc/kernels/configs.cuh NUM_MAX_NVL_PEERS 8 L3 / SEND_PHASE 1 RECV_PHASE 2 L20-21；internode.cu dispatch template L355-358 / WarpRole L370-376 / translate_dst_rdma_rank L74-75 / rdma_rank·nvl_rank L100 / kRDMASender L462 / kRDMAAndNVLForwarder「RDMA consumers and NVL producers」L690-691 /「Forward tokens from RDMA buffer」L738 / nvshmemi_ibgda_put_nbi_warp L149·L672 / kNumDispatchRDMASenderWarps 7 L1008；internode_ll.cu dispatch L40-41 / combine L555-556（__launch_bounds__ 1024,1）/ SEND 相 L87 / RECV 相 L248 / put_nbi_warp L164 / amo_nonfetch_add L231 / nvshmemi_get_p2p_ptr NVLink L162-166；ibgda_device.cuh mlx5_wqe_ctrl_seg L63 / put_nbi_warp L339 / quiet L495；setup.py NVSHMEM L19-25；README normal L5 / LL L7 / intranode 153·158 L19 / internode EP16 43·EP32 58·EP64 51·50 / LL EP8 77·114 L32 / EP256 194·360 L37 / News #173 L39 / hook 逐字 L270 / zero-SM 逐字 L290 / NVSHMEM_DIR L82；tests/test_internode.py num_tokens 4096 L297 / rdma_idx 节点去重 L50-52 / rdma_send_bytes L179 / fp8_factor L194；test_low_latency.py 每副本字节 L138。**接口锚 main V2.5**（commit f6aa38fe06cd4a0489e7730365d4e0114970a2f3，2026-10-21）：deep_ep/buffers/ep.py EPBuffer L208 / hybrid docstring L217-219 / allow_hybrid_mode L255 / QP 解析 L553-558（direct L554 / hybrid L558）/ dispatch L562 / combine L770 / lb_prefetch_weights L854 / lb_reduce_grads L886 / EPHandle L27；base.py BufferBase L4 / allocator.py BufferAllocator L10；csrc/buffers/ep.hpp EPBuffer L23 / get_dispatch_buffer_size L183 / Direct L192 / Hybrid L199；comm/handle.cuh ncclGin L21 / ctor L31 / VASignalAdd L113 / wait L123-125；impls/ep/dispatch.cuh dispatch_impl L31-32 / gin.put_value L155-156；hybrid_dispatch.cuh hybrid_dispatch_impl L34-35 / gin.put<ncclTeamTagRail> L179-183·L445-451 / IBGDA 注释 L441·L444；hybrid_combine.cuh IBGDA 注释 L482；csrc/kernels/comm/context.cpp Gin 探测 L126 / get_gin_min_stride L65 / ncclWindow_t L219；README V2.5 L11-18（NVSHMEM 移除 L18）/ features L25-32 / EP 占 SM L39 / 单 EPBuffer L107 / do_cpu_sync L160-162 / EP_DISABLE_GIN L406 / EP_GIN_GDAKI_DEBUG L428 / Requirements L45-56（§4/§5/§6/§8）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，2.32.7-1 / 351f33fd——nccl.h.in ncclAlltoAll L648（a2a 对照，8.1）；gin/gin_host.cc ncclGetGinType L28 / ncclGetRailedGinType L36（V2.5 Gin 后端）；transport/net_ib/gdaki/doca-gpunetio（GDAKI = GPU Direct Async Kernel-Initiated，Gin RDMA 实现）；paths.cc PATH_PXN L243（a2a PXN）；模块全量走读归 8.4（§6/§8）
- [NVIDIA, NCCL 2.12 blog](../references.md)：Doubling all2all Performance with NCCL 2.12（key: nccl-all2all-blog），[C] 开放技术博客——PXN + message aggregation → all2all 2×（a2a 对照，10.2 §3 recap）（§8）
- [NVIDIA, ConnectX-7](../references.md)：ConnectX-7 NDR 400G（key: connectx7），[C] 官方规格披露·一手权威——Max total 400Gb/s = 50 GB/s 单向（§1/§5/§8，9.2 recap）
- [Zhu et al., 2015](../references.md)：Congestion Control for Large-Scale RDMA Deployments（DCQCN，key: dcqcn），[B] 老经典无 arXiv，SIGCOMM 2015——RP/CP/NP + ECN/CNP + 16:1/32:1 incast 评估（§7 recap，9.3 详展）
- training 5.4（纯文字指引，不挂链接）：dispatch/combine 置换矩阵 $P$ 与加权转置（§1）、每 token $2kd$ 上界与命中分布 $\mathbb{E}[D]$ 卡级 7.6/节点 5.3/组限定 3.6（§2）、木桶 $\max_j R_j$（§4）、V3 EP64 两阶段 20 SM 与组限定 $M=4$（§6）——EP 训练账 recap 不重推（§2/§3/§4/§7）
- inference 3.3 / 7.4 / 8.1（纯文字指引，不挂链接）：normal/LL serving 语义 + dispatch 字节式 (6) + decode 逐 peer KB 延迟主导（3.3 §6）、EP 拐点命题 3 $\tau(e)=\max(A/e,F)+g(e)$ $e^{\ast}=32$（3.3 §5）、Mega MoE 重叠（3.3 §5）；Wide-EP prefill 最小/decode 最大 + EPLB 冗余 moe_lb 1.090 + 负载粒度 CV∝√(e/E) 命题 2 + EP>32 doc L295/L297 + 单 expert 18.8 MB（7.4 §1-3）；a2a 共享出口时间模型（8.1 §4）——serving 语义 recap 不重推（§2/§5/§8）
- 8.1 / 9.2 / 9.3 / 10.1 / 10.2 / 8.4 / 10.4（纯文字指引，不挂链接）：a2a 每 rank/每 peer 命题 1 与流量模式（8.1）、GDR 硬件路径与带宽 min/persistent mapping（9.2）、DCQCN/PFC/IB CC 机制（9.3）、fat-tree 1:1 与 rail-optimized（10.1）、rail 映射/SHARP/多作业干扰（10.2）、NCCL 模块全量走读与 proxy（8.4）、CP/Ring Attention（10.4）——recap 不重推（§3/§5/§6/§7/§8/§9）
