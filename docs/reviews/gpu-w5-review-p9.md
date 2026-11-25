# gpu W5 adversarial review — Part 9（RDMA / GPUDirect）

- **Evaluator**: fresh-context adversarial（不继承写作上下文，未派 subagents），read-only（仅写本报告）。
- **Date**: 2026-11-25。
- **Scope**: `gpu/09-rdma-gpudirect/01-rdma-fundamentals.md`（9.1）、`02-gpudirect-rdma.md`（9.2【重】）、`03-congestion-reliability.md`（9.3）。
- **Authority**: spec `docs/superpowers/specs/2026-11-24-gpu-dimension-design.md` §5 P9 / §4.3 符号表 / §6.3 强制重推导；T1 `docs/research/gpu-w5-sources.md` §2-4/§5.1/§6.3；recap 源 `inference/08-distributed-and-pd/03-kv-transfer.md`（8.3）；fresh tarballs `/tmp/gpu-w5-{rdma-core,peermem,gdrcopy,nccl,deepep-v121,deepep-main}`。
- **Method**: 全部关键数值独立 Python（`-I`）重算（47 项，见 verification log）；代码锚 file:line 逐一对 fresh tarball；recap 边界逐句对 inference 8.3 / 7.3 / 8.4 / 8.2；`lint-chapter.sh` + `lint-math-boundary.py` 机械门；refs citedBy 程序化对账。
- **Diff package**: `.superpowers/sdd/2026-11-25-gpu-w5-rdma-fabric/review-a2ef471..2af2606.diff`（P9 hunks @ L508/704/915；P9 尚无 fix round，本审为首轮对抗）。

## 结论速览

| 项 | 结果 |
| --- | --- |
| S1（fatal） | **0** |
| S2（important） | **1**（9.1 DeepEP QP 池 O(n) 标度结论与锚代码矛盾） |
| S3（minor） | **5** |
| 强制重推导（9.2 GDR 带宽 / 9.3 DCQCN / 9.1 延迟账） | **全 PASS（47/47）** |
| Recap 边界（inference 8.3 / 7.3 / 8.4 / 8.2） | **PASS（不重推）** |
| 代码锚抽查 | **全命中**（rdma-core / peermem / gdrcopy / NCCL / DeepEP 双锚） |
| lint（chapter + math-boundary） | **3/3 OK** |
| CJK（9.2≥5500 / others≥3500） | 9.1=4006 / 9.2=6005 / 9.3=5856 **PASS** |
| display 9 禁形 / prose \lt / bare <= / \div | **0 / 0 / 0 / 0** |

**总体**：数值脊柱与机制推导可靠，recap 边界干净，代码锚全命中。唯一 S2 是 9.1 §5 连接开销账里对 DeepEP `num_qps_per_rank` 的 O(n) 标度论断——与 v1.2.1 锚代码（per-peer QP 数组）矛盾，且污染练习 2(d) 校验锚，须修。其余为 refs 回写缺口与措辞/格式小项。

---

## 1. 强制重推导 verification log（独立 Python，`python3 -I`）

脚本 `/tmp/gpu-w5-p9-review/rederive.py`。每项 = 独立重算值 vs 章内值。**不信任章内数字，全部从口径重新推。**

### 1.1 9.2 GDR NIC→HBM 带宽账（spec §6.3 强制项）

| # | 项 | 独立重算 | 章内 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | Gen5 x16 raw = 32GT/s×16/8 | 64.0 GB/s（128b/130b 有效 63.02，wp 名义 64） | 64（Hopper wp p.49 逐字） | PASS |
| 2 | Gen6 x16 raw = 64GT/s×16/8 | 128.0 GB/s | 128（CX8 HW manual） | PASS |
| 3 | CX7 400G = 400e9/8 | 50 GB/s | 50 | PASS |
| 4 | CX8 800G = 800e9/8 | 100 GB/s | 100 | PASS |
| 5 | **regime 单口 CX7+Gen5** | min(50,64)=**50，NIC 线速限** | 50 NIC 限 | PASS |
| 6 | regime 双口 CX7(2×NDR200)+Gen5 | min(50,64)=50，NIC 限（双口不翻倍） | 50 NIC 限 | PASS |
| 7 | **regime CX8+Gen5** | min(100,64)=**64，PCIe 限** | 64 PCIe 限 | PASS |
| 8 | **regime CX8+Gen6** | min(100,128)=**100，NIC 限** | 100 NIC 限 | PASS |
| 9 | PCIe 余量 64−50 | 14 GB/s | 14 | PASS |
| 10 | vs HBM 3350/50 | 67× | 67× | PASS |
| 11 | 多 NIC Σmin 8×50 | 400 GB/s | 400 | PASS |
| 12 | NVLink 注入 8×450 | 3600 GB/s | 3600 | PASS |
| 13 | 比值 3600/400 | 9× | 9× | PASS |
| 14 | **共享上行反例** min(4×50,64) | 64（非 200） | 64 封顶 | PASS |
| 15 | bounce 串行两跳 (1/64+1/50)^-1 | 28.0702 | 28.07 | PASS |
| 16 | GDR/串行两跳 50/28.07 | 1.7812× | 1.78× | PASS |
| 17 | 流水 min(64,50) | 50（持平） | 50 持平 | PASS |
| 18 | bounce e2e 三跳 (1/64+1/50+1/64)^-1 | 19.5122 | 19.51 | PASS |
| 19 | GDR/e2e 50/19.51 | 2.5625× | 2.56× | PASS |
| 20 | host 流量 bounce 4M / GDR 0 | 4 GB / 0（M=1GB） | 4M→0 | PASS |
| 21 | staging 单跳 1GB/64 | 15.625 ms | 15.625 | PASS |
| 22 | staging 两跳 | 31.25 ms | 31.25 | PASS |
| 23 | BAR1 可用 256−32 | 224 MB | 224 | PASS |
| 24 | KV 分块 ceil(4295/224) | 19.174→**20** | ≈19.2 / ≥20 | PASS |
| 25 | gdrcopy 6/0.10、7/0.10 | 60–70× | 60–70× | PASS |

**regime 判定核心结论独立确认**：单口 400G CX7 = 50 < 64 → **NIC 线速限，非 PCIe 限**（章 §4 关键裁定正确，"把 400G GDR 说成 PCIe 瓶颈是错的" 成立）；CX8 800G=100>64 → Gen5 PCIe 限 / Gen6 128 NIC 限，regime 注主机代际正确。多 NIC Σmin + 独立 PCIe 路径前提 + 共享上行封顶反例均正确。bounce 省两跳（发送 D2H + 接收 H2D）串行/流水分账自洽。

### 1.2 9.3 DCQCN rate adjust（spec §6.3 强制项）

| # | 项 | 独立重算 | 章内 | 判定 |
| --- | --- | --- | --- | --- |
| 26 | 首 CNP：RT=RC | RT=50 | 50 | PASS |
| 27 | 首 CNP：RC=50(1−1/2) | **25（恰减半）** | 25 | PASS |
| 28 | 首 CNP：w=(1−g)·1+g | =1，**与 g 无关**（g∈{0.01,1/16,0.5,0.99} 验证） | 1 恒等 | PASS |
| 29 | 次 CNP（w 未衰减） | RT=25, RC=12.5 | 25 / 12.5 | PASS |
| 30 | fast recovery 5 次向 RT=25 | 18.75/21.875/23.4375/24.21875/24.609375 | 同 | PASS |
| 31 | w 衰减 1 timer (g=1/16) | 15/16=0.9375 | 0.9375 | PASS |
| 32 | 砍幅因子 1−0.9375/2 | 0.53125 | 0.53125 | PASS |
| 33 | w 5 timer (15/16)^5 | 0.724196 | ≈0.724 | PASS |
| 34 | **命题 3 闭式** RC_k=RT−(RT−RC0)/2^k | 与迭代 k=1..8 逐项吻合（归纳基例 k=0=RC0，步代入式(5)） | 式(7) | PASS |
| 35 | RC0=25,RT=50,k=5 | 50−25/32=49.21875 | 49.21875 | PASS |
| 36 | 差距占 RT | 0.78125/50=1.5625% | 1.56% | PASS |
| 37 | <1% 需 k≥6 | gap5=0.78125>0.5, gap6=0.390625<0.5 | k≥6 | PASS |
| 38 | AI 一步 RAI=2 | RT=52, RC=(52+49.21875)/2=50.609375→封顶 50 | 50.609→50 | PASS |

论文式 (1)-(4)（经 T1 §4.1 逐字）→ 章式 (3)-(6) 映射正确；α 译 w（α 专归消息延迟）符号纪律正确；三档恢复 max(T,BC)<F→FR / min(T,BC)>F→hyper / else AI 与论文 Fig 7 一致；无 slow start、N=50μs/K=55μs/F=5 部署值与论文一致。

### 1.3 9.3 PFC / IB 注入 / RC 超时 / straggler

| # | 项 | 独立重算 | 章内 | 判定 |
| --- | --- | --- | --- | --- |
| 39 | pause 触发 512e3/(50e9) | 10.24 μs | 10.24 | PASS |
| 40 | headroom 50e9×2e-6+4KB | 104 KB | 104 | PASS |
| 41 | IB t_tx 4096/50e9 | 81.92 ns | 81.92 | PASS |
| 42 | IPD=t_tx 注入率 | 50×(81.92/163.84)=25 GB/s | 25 | PASS |
| 43 | RC 超时 4.096μs×2^20 | 4.294967 s | 4.295 s | PASS |
| 44 | 对 1GB 正常 20ms 倍数 | 214.75× | ≈215× | PASS |
| 45 | 4lane→1lane | 100G=12.5 GB/s, β×4 | 12.5 / 4× | PASS |
| 46 | n=16 每 rank 发送 2×15/16×1GB | 1.875 GB | 1.875 | PASS |
| 47 | 健康 37.5ms / 降级 75ms / max 翻倍 | 37.5 / 75 / 2× | 同 | PASS |

式(9) 超时公式与 NCCL connect.cc L28 注释逐字一致（"4.096us * 2^timeout"，default 20→4,294,967.296us≈4.3s）。straggler max 语义（一条慢 rail 拖慢整个集合）正确。

### 1.4 9.1 RDMA 延迟账 + QP 连接开销

| # | 项 | 独立重算 | 章内 | 判定 |
| --- | --- | --- | --- | --- |
| L1 | t_copy(4KB)=4096/1e10 | 0.4096 μs | 0.41 | PASS |
| L2 | Δ_TCP=2t_copy+2t_ctx+t_stack | 3.8192 μs | 3.82 | PASS |
| L3 | Δ_RDMA=t_db | 0.2 μs | 0.2 | PASS |
| L4 | 比值 Δ_TCP/Δ_RDMA | 19.096× | 19.1× | PASS |
| L5 | 省项 Δ_TCP−Δ_RDMA | 3.6192 μs | 3.62 | PASS |
| L6 | 1GB 拷贝项 2×(1e9/1e10) | 200 ms | 200 | PASS |
| L7 | n=1024 每 rank QP | 1023 | 1023 | PASS |
| L8 | 全网 n(n−1) | 1,047,552≈1.05e6 | ≈1.05e6 | PASS |
| L9 | 串行 bringup ×100μs | 104.76 s | ≈105 s | PASS |

命题 1 结构（TCP=2 拷贝+2 切换+协议栈，RDMA=1 doorbell）由路径枚举决定，分量值标教学假设口径，结构结论可靠。**注**：RC 全互连 n(n−1) QP 账本身正确；S2-1 仅针对其后 DeepEP 类比的 O(n) 结论（见 findings）。

---

## 2. Recap 边界核（verdict: **PASS**）

| 边界 | 要求 | 核对 | 判定 |
| --- | --- | --- | --- |
| **9.2 ← inference 8.3** | KV 传输语义只 recap（三自由度/push-pull/GDR 定位/池化），不重推 τ_c/τ_x/egress | §2 recap 三条逐句对 8.3 §1/§5/式(3)：三自由度✓、push/pull initiator 视角 NIXL_WRITE/READ✓、GDR 定位逐字✓、Mooncake 池化 ΣBW_j=200✓；τ_c/τ_x/egress/κL 仅出现在边界声明与指针句，**无重推**；4.295GB 仅作 BAR1 容量对照数值 | PASS |
| 9.2 增量 | = GDR 硬件路径 only | §3 peermem/gdrcopy、§4 带宽 min、§5 多 NIC PCIe 前提、§6 bounce 对比、§7 BAR1 —— 均硬件机制，非 serving 语义 | PASS |
| 9.2 多 NIC 命题 2 | 非重推 8.3 式(3) | 8.3 式(3)=ΣBW_j（serving 层）；9.2 命题 2=Σmin(BW_NIC,BW_PCIe)+独立 PCIe 前提（GPU 侧硬件增量，spec §5 9.2 关键推导明列"多 NIC 聚合"），章明言"本章给出其 GPU 侧 PCIe 前提" | PASS（属强制重推导增量，非越界） |
| **9.2 ← 7.3 半谐波** | recap 不重推 | §6 "带宽账（recap 7.3 命题 1，不重推）" 只代入式(1)，未重推半谐波公式本身；7.3 §命题1 确为半谐波/并联电阻式（W4 errata 口径） | PASS |
| **9.2 ← 8.4 proxy** | recap 不重推 | §8 "NCCL NET transport（recap 8.4 §5，不重推）" 引 sendProxyProgress L1327/recvProxyProgress L1496 锚，增量=GDR 对 proxy 职责的改变（控制面 vs 数据面），未重走 posted/transmitted/done 状态机 | PASS |
| **9.3 ← 8.1/8.2** | ring 消息量/α-β recap 不重推 | §7 直接用 2(n−1)/n·M·β_eff（recap 8.2 式(2)），未重推归纳；8.2 式(2) 确为 T_ring=2(n−1)α+2(n−1)/n·Mβ | PASS |
| 9.3 ← 9.2 | GDR 带宽账 recap 不重推 | §7 "GDR 流量与拥塞交互（recap 9.2，不重推带宽账）" | PASS |
| 越界（serving 调度/K8S） | 不出现 | 9.3 §1/§8 明言"平台编排（K8S 调度/容器网络）归 cluster 维度"；无 serving 调度/引擎编排内容 | PASS |

---

## 3. 代码锚抽查表（fresh tarball，file:line 逐一核）

| 章 | 锚 | tarball 实测 | 判定 |
| --- | --- | --- | --- |
| 9.1 | verbs.h ibv_qp_type RC=2/UC/UD L934-937 | L934 enum / L935 RC=2 / L936 UC / L937 UD | PASS |
| 9.1 | verbs.h ibv_qp_state L1084-1088 | L1084 enum / RESET..RTS L1085-1088 | PASS |
| 9.1 | verbs.h ibv_wr_opcode L1137-1144 | L1137 enum / 7 opcodes L1138-1144 | PASS |
| 9.1 | ibv_reg_mr L2692 / create_cq L2962 / poll_cq L3040 | 2692 / 2962 / 3040 | PASS |
| 9.1 | create_qp L3246 / modify_qp L3467 / post_send L3662 / post_recv L3671 | 3246 / 3467 / 3662 / 3671 | PASS |
| 9.1 | man ibv_modify_qp.3 L128 "Reset-->Init-->RTR-->RTS" | L128 逐字 | PASS |
| 9.1 | CMakeLists PACKAGE_VERSION 67.0 | L87 "67.0" | PASS |
| 9.2 | nvidia-peermem.c 696 行 / DRV_NAME nv_mem L46 | 696 行 / L46 | PASS |
| 9.2 | acquire L198→get_pages L216 / acquire_nc L246→persistent L265 | 198/216/246/265 | PASS |
| 9.2 | nv_dma_map L300→dma_map_pages L323 + sg_alloc_table L338 | 300/323/338 | PASS |
| 9.2 | free_callback L129 / ib_register_peer_memory_client L570+L601 | 129/570/601 | PASS |
| 9.2 | gdrapi.h to_mapping L175 / from_mapping L177（旧 to_bar 已移除） | L175/L177，无 to_bar | PASS |
| 9.2 | gdr_open L74 / pin_buffer L87 / map L152 | 74/87/152 | PASS |
| 9.2 | gdrcopy README 6-7us / pin 10us-1ms / copy_to_mapping ~0.10μs | L25/L28/L286-289 | PASS |
| 9.2 | NCCL gdr.cc nv_mem sysfs L18 / ncclIbGdrSupport L24 | L18/L24 | PASS |
| 9.2 | paths.cc ncclTopoCheckGdr L494 / NCCL_NET_GDR_LEVEL L542 | L494/L542 | PASS |
| 9.2 | net.cc sendProxyProgress L1327 / recvProxyProgress L1496 / GDRCOPY_SYNC_ENABLE L338 | 1327/1496/338 | PASS |
| 9.2 | DeepEP v1.2.1 internode_ll.cu dispatch L40 / combine L555 | __global__ 声明起 L40/L555（函数名 L41/L556；T1 口径 L40-41/L555-556） | PASS |
| 9.2 | configs.cuh SEND_PHASE 1 / RECV_PHASE 2 L20-21 | L20/L21 | PASS |
| 9.2 | ibgda_device.cuh mlx5 WQE L63 / put_nbi_warp L339 | L63/L339 | PASS |
| 9.2 | buffer.py return_recv_hook L535 / L561-562 逐字 | L535/L561-562 | PASS |
| 9.2 | README L290 "without costing any GPU SMs" | L290 逐字 | PASS |
| 9.2 | DeepEP main V2.5：V1 全移除 / NVSHMEM 不再依赖 / IBGDA 仅注释 / EP 占 SM | README L18/L39；ibgda_device.cuh/internode_ll.cu/buffer.py 不存在；IBGDA 仅 hybrid_dispatch.cuh L441/444 注释 | PASS |
| 9.3 | NCCL connect.cc NCCL_IB_TIMEOUT=20 L26 / RETRY_CNT=7 L32 / QPS_PER_CONNECTION L106 | L26/L32/L106；超时注释 "4.096us*2^timeout" | PASS |
| 9.1 | DeepEP num_qps_per_rank=24（buffer.py L34） | L34 default 24 | PASS（值正确；**标度结论错，见 S2-1**） |

版本锚：rdma-core v67.0 master 4b7b208c / peermem tag 620.73.04(a4b95f6a) / gdrcopy master 0d85ee38 / NCCL 2.32.7-1 351f33fd / DeepEP v1.2.1 9af0e0d0 + main V2.5 f6aa38fe —— 与章首/T1 一致。

---

## 4. Findings

### S1（fatal）— 0

无。全部强制重推导数值 PASS，regime 判定正确，recap 边界干净。

### S2（important）— 1

**S2-1 · 9.1 §5 L126 + 练习 2(d) L176 — DeepEP QP 池 "O(n²)→O(n)" 标度结论与锚代码矛盾**

- **章内断言**（L126）："DeepEP `num_qps_per_rank = 24`（v1.2.1，10.3 走读）即用固定 QP 数承载所有 token 路由，把 $O(n^{2})$ 建连成本压到 $O(n)$。" 练习 2(d) 校验锚（L176）重复："控制面 O(n²) 成本，固定 QP 池压到 O(n)（10.3）"。
- **独立核对（fresh tarball v1.2.1）**：
  - `deep_ep/buffer.py` L101：`os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'` —— 24 是 **per-PE** RC QP 数。
  - `csrc/kernels/ibgda_device.cuh` L79-80：`ibgda_get_rc(pe, id)` 返回 `state->globalmem.rcs[pe * num_rc_per_pe * num_devices_initialized + id % (...)]` —— QP 数组按 **(peer, qp_id)** 二维索引，每 peer 有 num_rc_per_pe 条 QP。
  - `csrc/kernels/internode_ll.cu` L164：`nvshmemi_ibgda_put_nbi_warp(..., dst_rank, dst_expert_local_idx, ...)` —— qp_id 取**目的 expert 本地索引**，与 dst_rank 无关。
  - docstring（buffer.py L47-48）："the low-latency mode requires that this number equals to the number of local experts."
- **结论**：每 rank 持 24×(n−1) 条 QP（每 peer 24 条），全网 24n(n−1) = **O(n²)**，不是 O(n)。24 QP/peer 的目的是**按 expert 索引分散流量到多条 QP**（并行/避单 QP HoL），不是减少连接总数。"固定 QP 池压到 O(n)" 的标度论断错误，且污染练习校验锚。
- **建议修正**：把该类比改为正确表述，例如"DeepEP 预注册长驻 QP 池（`num_qps_per_rank=24`，每 peer 24 条、按目的 expert 索引选 QP）摊销建连成本——连接是控制面慢路径、一次建好长驻复用，避免逐消息建连；QP 总数仍随 rank 数平方增长，但每消息不再付建连开销"。练习 2(d) 校验锚同步改（删 "压到 O(n)"，改 "长驻 QP 池摊销建连、免逐消息建连"）。一般原理（预建池复用、连接慢路径/传输快路径）正确，仅标度结论须修。
- **severity 理由**：属 9.1 关键推导（连接开销账）内的代码锚定事实论断，可由锚代码直接证伪，且进入练习校验锚；非核心数值交付物（RDMA 语义/延迟账/verbs 均正确）故非 S1。

### S3（minor）— 5

**S3-1 · 9.1 §5 L121/L124 — "UD 仅改 STATE" 过度简化**
章："UD：仅改 STATE（无连接）"/"UD 无连接，迁移仅改 STATE，不需对端元数据"。man `ibv_modify_qp.3` UD 表实测：Init 需 STATE+PKEY_INDEX+PORT+QKEY，RTR 仅 STATE，RTS 需 STATE+SQ_PSN。"不需对端元数据" 正确，但 "仅改 STATE" 对 INIT/RTS 不成立。建议改 "UD：RTR 仅改 STATE、无对端元数据；INIT 仍设本地 port/pkey/qkey，RTS 设 SQ_PSN"。（注：references.md rdma-core 条与 T1 §2.2 亦承此简化 "UD 仅 STATE"，宜一并核正。）核心对比（UD 无带外交换）正确，故 S3。

**S3-2 · refs citedBy 回写缺口（程序化对账，5 key）**
章参考节实际引用 vs `gpu/references.md` citedBy 列：
- `connectx7` citedBy=9.1,9.2 —— **缺 9.3**（9.3 §7/§8 引 hardware CC/reliable transport）。
- `connectx8` citedBy=9.2,10.1 —— **缺 9.1,9.3**（9.1 §7、9.3 §8 均引）。
- `hopper-whitepaper` citedBy=…7.3,9.2 —— **缺 9.1**（9.1 §1/§7 PCIe Gen5 p.49）。
- `pcie-spec` citedBy=…7.2,9.2 —— **缺 9.1**（9.1 参考节列 key: pcie-spec）。
- `nccl` citedBy=…8.4,9.2,10.2… —— **缺 9.3**（9.3 §6 net_ib TIMEOUT/RETRY_CNT）。
- （附）`rdma-programming-manual` citedBy=9.1，9.3 §6 L174 有 inline link "经 9.1 recap"——宜 citedBy+=9.3 或改纯文字指引。
W5 research 已做部分回写（rdma-core+=9.3、deepep+=10.4、pcie/hopper/datasheet/cuda-pg+=9.2）但漏上述。建议 T7 统一补。

**S3-3 · 9.2 §3 L42 — 过程泄漏措辞 "spec 旧查的"**
"spec 旧查的 Mellanox/peer-memory + NVIDIA/peer-memory 名称不存在，404" 叙述了 spec 的核查过程。先例章（3.2/5.3/7.1）漂移注均陈述事实（"旧路径 404"）不归因 spec。建议改事实句："社区常误写的 Mellanox/peer-memory 与 NVIDIA/peer-memory 仓库名不存在（404），legacy off-tree 实为 Mellanox/nv_peer_memory"。

**S3-4 · 9.2 §4 L78 — 交叉引格式 "§9.1 口径订正"**
表格 cell "NIC 线速限（双口不翻倍，§9.1 口径订正）" 应作 "9.1 §7 口径订正"（章.节，非 §章）。

**S3-5 · 9.2 L11 — 声明未用局部符号 T_serial（nit）**
符号声明列 "$T_{\mathrm{serial}}$ = 串行端到端总时间（§6）"，但 §6 实际只用 $\mathrm{BW}_{\mathrm{serial}}$/$\mathrm{BW}_{\mathrm{serial,e2e}}$，T_serial 全文未出现（grep 0 次）。无害；建议删该声明或在 §6 补一句 T_serial=M/BW_serial 代入。

---

## 5. 规范审查（rubric + R2 + 符号 + 机械门）

- **lint-chapter.sh**：3/3 LINT OK（禁令词/模板小节/metadata/arXiv-refs/证据等级/$$配平/反转义/边界）。
- **lint-math-boundary.py**：EXIT=0（open/close/space/matrix/pipe/range/lt/macro/rescape 全过）。
- **display 9 禁形**（\{ \} \| \, \; \: \! \% \#）：raw grep 全 0。
- **prose \lt**：0；**bare <=**：0；**\div**：0（白名单 /n 口径）。
- **CJK**：9.1=4006（≥3500）/ 9.2=6005（≥5500【重】）/ 9.3=5856（≥3500）PASS。
- **R2-1…12**：节首设问/数值开场✓、章脊口径锁定（CX7 NDR400+Gen5 x16 / H100 SXM）✓、公式后 300 字内全数字实例✓、命题-证明骨架（9.1 命题1 / 9.2 命题1-3 / 9.3 命题1-3）✓、推导路线句✓、定义配正例+边界反例✓、口径注（教学假设逐处标）✓、术语判断句✓、逻辑关节词✓、提点/坑块引用✓、总结表回指列✓。
- **练习**：三章均 [10]/[20]/[30]/[30] 非递减、≥1 [10]/[20]、无 [40]、每计算题带校验锚、≥1 题证正文命题（9.1 练习3 命题1 / 9.2 练习3 命题2 / 9.3 练习3 命题3）✓。9.3 命题3 完整证明留练习3（证明路线+归纳基例/步已给，合 proof-exercise 先例）。
- **符号 §4.3**：n=rank 章首声明✓、α/β 通信语义声明（"非 inference 接受率"）✓、M=消息字节✓、N 专归参数量/B 专归 batch 不占用✓、9.3 时间一律 T 下标（T_rec/T_CNP/T_w/T_timeout，无裸 T）✓、DCQCN α 译 w 译名注✓、9.2 κ/L/T 几何不引入✓。
- **证据分级**：DCQCN [B] 老经典无 arXiv（pp 523-536，DOI 10.1145/2785956.2787484）✓；IB CC whitepaper / GDR docs / ConnectX-7/8 / IBTA / RDMA manual [C] 官方规格✓；rdma-core/peermem/gdrcopy/NCCL/DeepEP 教学参照+官方实现✓。
- **过程泄漏**：除 S3-3 外无（"spec §4.3" 符号声明为 W1-W4 既有约定，非泄漏）。

## 6. 修复优先级（供 T7）

1. **S2-1 必修**：9.1 §5 L126 DeepEP O(n) 标度结论 + 练习 2(d) L176 校验锚（附证据 ibgda_device.cuh L79-80 / internode_ll.cu L164 / buffer.py L101）。
2. S3-2 refs citedBy 5 key 回写（connectx7/connectx8/hopper-whitepaper/pcie-spec/nccl）。
3. S3-1 UD 状态迁移措辞（9.1 + references.md + T1 同步）；S3-3 过程泄漏句；S3-4 交叉引格式；S3-5 T_serial nit。

**审查结论**：P9 数值与机制可靠（47/47 重推导 PASS、recap 边界 PASS、代码锚全命中），S1=0 / S2=1 / S3=5。S2-1 修复后 P9 可放行。