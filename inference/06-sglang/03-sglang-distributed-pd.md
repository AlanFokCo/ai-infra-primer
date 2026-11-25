# 6.3 SGLang 分布式与 PD 分离（SGLang Distributed Execution and PD Disaggregation）

> 同一组 8 张 GPU，attention 与 MoE 各按什么粒度切组？PD 分离后一条请求的 KV 怎么从 prefill 卡走到 decode 卡、谁保证一个字节都不缺？ | 前置：1.2（ $\kappa$ 与 prefill/decode 账）、2.2（prefix caching）、2.4（KV offload 与 Mooncake transfer engine）、3.4（piecewise CUDA Graph）、4.3（chunked prefill）、6.1（SGLang 架构与 ForwardMode）、6.2（overlap scheduler 与 DP controller）；training 5.1（TP 通信账）、5.4（EP 训练账）只 recap | 锚点：SGLang（arXiv:2312.07104）[A] + sgl-project/sglang repo `distributed/`、`disaggregation/`（含 `base/conn.py` KV 传输接口）、`eplb/`、`elastic_ep/`、`connector/`（权重/文件存储）、`checkpoint_engine/`、`compilation/` [A]、Mooncake（arXiv:2407.00079）[A]、NIXL（教学参照 + 官方实现）、SGLang v0.4 blog [C]（开放技术博客、非闭源）、DeepInfer doc（教学参照 + 官方实现，官方披露）；SGLang 代码路径相对 `python/sglang/srt/`，main fresh tarball 核实 2026-11-23

## 1. 配置面：tp_size 不再是「TP 度」

v0.4 blog 的 DP attention 基准命令是 `--tp 8 --enable-dp-attention`（[SGLang Team, 2024](../references.md)，[C]）；当前 main 里 `enable_dp_attention` 已经不是字段——这 8 张卡的并行度到底记在哪？答案是一次配置面迁移：并行度改由 `arg_groups/` 声明、`runtime_context.py` 派生，旧的布尔开关退化为派生量。

> **坑**：照旧 blog / 旧文档找 `enable_dp_attention` 字段——当前 `server_args.py` L345/L355 里它只出现在 `resolved_dict()`：`resolved["enable_dp_attention"] = resolving_view(self).attn_dp_size > 1`，专为仍从 `/server_info` 读该键的客户端保留。触发条件：版本漂移（2026-11-23 核实）；后果：按布尔字段写配置逻辑会拿到派生值而非输入值；规避：配置一律读 `attn_dp_size`，派生关系见下。

新配置面的三个输入字段（`arg_groups/fields/parallel.py`）：`dp_size` L77 = 整模型 DP 副本数（每副本一套独立引擎）；`attn_dp_size` L85 = TP 组**内部**的 attention DP 宽度，help 逐字「the number of data-parallel attention groups inside the TP group, one scheduler each, while the FFN stays tensor parallel」；`attn_cp_size` = attention CP 宽度。所谓 DP attention，是指 attention 按请求切给各 DP rank 独立算、MoE/FFN 仍在全 world 上切分的混合并行（通信账归 8.1，本节只讲配置与容量）。`tp_size` 的语义同时变了：它是 **world size**（总进程数），不是 attention 的实际 TP 度——`arg_groups/parallel_hook.py` L136/L149 两处注释逐字「The tp_size is the world size, not the real tensor parallel size」。

派生规则落在 `runtime_context.py` `derive_attn_tp_size` L73，源码一行 `return tp_size // attn_dp_size // attn_cp_size`。对正整数，链式整除等价于单次下取整：

$$
\mathrm{attn\_tp\_size} = \left\lfloor \frac{\mathrm{tp\_size}}{\mathrm{attn\_dp\_size} \cdot \mathrm{attn\_cp\_size}} \right\rfloor \qquad (1)
$$

符号表：三个量均为配置整数。本章数学记号 $t$ = `tp_size`（world size）、 $d$ = `attn_dp_size`、 $c$ = `attn_cp_size`（CP 度，沿 spec §4.3）、 $e$ = `ep_size`、 $a$ = `attn_tp_size`（attention TP 度，派生量）；代码字段名以 backtick 出现。整除不是假设而是断言：`attn_cp_size > 1` 时 `parallel_hook.py` L137-141 要求 `tp_size % attn_cp_size == 0` 且 `tp_size % (attn_dp_size * attn_cp_size) == 0`；`attn_dp_size > 1` 时另有 L389 断言 `tp_size % attn_dp_size == 0`—— $c$ = 1 情形（cp gate 不触发）的整除保证由它兜底；MoE 侧断言以 `moe_dp_size > 1` 为前提——L153 要求 `ep_size * moe_dp_size <= tp_size`，`ep_size > 1` 时 L159 收紧为等式（另断言 `pp_size == 1`）。代入 blog 的 8 卡口径： $t$ = 8、 $d$ = 8、 $c$ = 1，式 (1) 得 $a$ = 1——每张卡独立算自己那份请求的 attention；MoE 侧 `ep_size` = 8、`moe_dp_size` = 1，不触发该组断言，EP 组直接铺满 world。

rank（进程编号）布局由 `derive_attention_ranks` L79 给出，docstring 逐字「The rank layout is (dp, cp, tp), with tp changing fastest」：

$$
\mathrm{tp\_rank} = (\mathrm{attn\_dp\_rank} \cdot \mathrm{attn\_cp\_size} + \mathrm{attn\_cp\_rank}) \cdot \mathrm{attn\_tp\_size} + \mathrm{attn\_tp\_rank} \qquad (2)
$$

代入 $t$ = 8、 $d$ = 8、 $c$ = 1（此时 $a$ = 1）：rank 5 的坐标是 (5, 0, 0)——dp 坐标就是 rank 本身，每卡一个 attention 组。

**命题 1**（rank 布局双射）：若整除断言成立，则式 (2) 是 $[0, t)$ 到 $[0, d) \times [0, c) \times [0, a)$ 的双射——每个进程恰好属于一个 attention DP 组、一个 CP 组、一个 TP 组。**证明**（路线：混合进制分解的唯一性，三步）：断言保证 $t = d \cdot c \cdot a$（ $c$ > 1 时由 L140 整除断言、 $c$ = 1 时由 L389 保证）。第一步，对任意 $\mathrm{tp\_rank} \in [0, t)$ 做带余除法： $\mathrm{attn\_tp\_rank}$ 是 $\mathrm{tp\_rank}$ 除以 $a$ 的余数， $\mathrm{attn\_cp\_rank}$ 是 $\mathrm{tp\_rank} // a$ 除以 $c$ 的余数， $\mathrm{attn\_dp\_rank} = \mathrm{tp\_rank} // (a c)$——源码返回式正是第一、第三两项。第二步，以 $a$、 $c$、 $d$ 为基的混合进制表示唯一（带余除法逐位唯一），故映射单射。第三步，三个坐标的取值范围恰为 $[0, d)$、 $[0, c)$、 $[0, a)$，笛卡尔积大小 $d c a = t$ 与定义域等大，单射即双射。证毕。边界反例： $t$ = 8、 $d$ = 3、 $c$ = 1 不满足整除断言，`parallel_hook.py` L389 直接拒绝启动——双射的前提是配置合法，不是自动成立。

配置经 `runtime_context.py` `get_parallel()` L1314（`ParallelContext`）读取，通信组在 `distributed/parallel_state.py` 实例化：`GroupCoordinator` L271，`initialize_model_parallel` L2596。组访问器分两族——attention 族 `get_tp_group` L2258 / `get_attn_tp_group` L2262 / `get_attn_cp_group` L2270 / `get_dcp_group` L2278，MoE 族 `get_moe_dp_group` L2287 / `get_moe_ep_group` L2291 / `get_moe_tp_group` L2295，外加 `get_pp_group` L2311。**同一个 world，两套划分**：attention 层在 attention 族组内通信，MoE 层在 MoE 族组内通信，这是 DP attention 能「attention 独立、MoE 共享」的结构基础。

```
world = 8 ranks（tp_size = 8，attn_dp_size = 8，ep_size = 8）
attention 族:  [r0] [r1] [r2] [r3] [r4] [r5] [r6] [r7]
               8 个独立 attention 组（attn_tp_size = 1），各持 B/8 请求
MoE 族:        [r0 - r1 - r2 - r3 - r4 - r5 - r6 - r7]
               1 个 EP 组，expert 切 8 份
数据流:        attention 输出 ─ dispatch（EP all-to-all，3.3）─► MoE ─ combine ─► 各 DP rank
```

DP attention 为什么值得开？容量账。MLA 只有一个共享 latent KV head，TP 按 head 切分时 latent 无处可切、必须在每个 attention TP rank 复制（机制 recap training 5.1 §8，不重推）。每 rank 的 KV 驻留由 1.2 式 (3) 的 $\kappa$ 与本地 batch 决定：

$$
M_{\mathrm{rank}} = \kappa L T \cdot \frac{B}{d} \qquad (3)
$$

符号表： $M_{\mathrm{rank}}$ = 单 rank KV cache 字节数； $\kappa$ = 每 token 每层 KV 字节（1.2 式 (3)）； $L$ = 层数； $T$ = 序列长； $B$ = 全局 batch（DP 合计口径）； $d$ = `attn_dp_size`。要点在复制因子：纯 TP（ $d$ = 1）时每 rank 持全量 $B$ 条请求的 KV，全 world 聚合是必要量的 $t$ 倍； $d$ = $t$ 时复制因子 `attn_tp_size` = 1，聚合无冗余。本章数值口径就此锁定：**MLA 几何 recap 1.2 §3 几何表 V3 行**（ $L$ = 61、 $d_{\mathrm{latent}}$ = 512、 $d_{\mathrm{rope}}$ = 64、bf16 $\kappa$ = 1152 B/token/层， $\kappa L$ = 70272 B/token ≈ 68.6 KiB/token）；容量例 $T$ = 4096、 $B$ = 64（教学示例值），硬件 8×H100 节点（blog 基准硬件口径）；传输例换 $T$ = 32768，原因：chunk 划分演示需要多个 chunk（§4）。代入式 (3)：单序列 KV = 70272 × 4096 = 287834112 B ≈ 274.5 MiB； $d$ = 1 时每 rank 64 条 = 17.16 GiB， $d$ = 8 时每 rank 8 条 = 2.14 GiB——**同一全局 batch，每 rank KV 驻留降到 1/8**，腾出的 HBM 换更大 batch。blog 的实测回报是 8×H100 上 DeepSeek-Coder-V2-Instruct-FP8 decode 吞吐 1.9×（[SGLang Team, 2024](../references.md)，[C]，v0.4 vs v0.3）。all-gather/dispatch 的通信账与 DP controller（`data_parallel_controller.py` `DataParallelController` L159、`LoadBalanceMethod` L106 四值）分别归 8.1 与 6.2，此处不展开。

PP 与 EP 的实现落点各一句：PP 走 `managers/scheduler_pp_mixin.py` + `distributed/pp_sync_window.py`，PD × PP 的层范围元数据见 §5 `KVArgs`；EP 的 dispatch/combine 算子账 recap 3.3、生产部署归 7.4，`batch_overlap/` 的 TBO/SBO 计算-通信重叠（`MaybeTboDeepEPDispatcher` `two_batch_overlap.py` L1161）是另一机制、与 overlap scheduler 无关，走读归 6.2。

## 2. 通信原语：小消息走哪条路

decode batch 64、 $d_{\mathrm{model}}$ = 7168（V3 口径 recap training 5.1 §8）、bf16 时，一次 all-reduce 的消息量 = 64 × 7168 × 2 = 917504 B ≈ 0.875 MiB——亚 MB 级小消息，延迟项而非带宽项主导（decode TP 延迟全账归 8.1）。serving 与训练的通信模式差异 recap 一句：无 backward，每层只有 2 次前向 all-reduce（training 5.1 的每层 4 次含反向，serving 减半），推导不重复。

`distributed/device_communicators/` 是原语目录，按消息大小与拓扑选路：

| 原语 | 锚点 | 适用 |
| --- | --- | --- |
| `CustomAllreduce` | `custom_all_reduce.py` L40 | NVLink 域内小消息 custom kernel |
| `CustomAllReduceV2`（`AllReduceChoice` L111） | `custom_all_reduce_v2.py` L152 | 同上，选择逻辑重构版 |
| `TorchSymmMemCommunicator` | `torch_symm_mem.py` L34 | PyTorch symmetric memory 路径 |
| `pynccl` / `pymscclpp` | 同目录 | NCCL / MSCCL++ 大消息 |
| `quick_all_reduce` / `ring_simple_all_reduce` / `dpx_hier_all_reduce` | 同目录 | 轻量 / ring / 分层变体 |
| `multicast_all_gather` / `shm_broadcast` | 同目录 | all-gather 与共享内存广播 |

选择原则是机制级的：域内小消息走 custom kernel / symmetric memory 省 launch 与协议开销，大消息走 ring 类算法摊带宽；逐原语的延迟-带宽账与 NVLink 域边界归 8.1，RDMA / NVLink 硬件层归 gpu 7.1 / 9.1。同目录还有 `mooncake_transfer_engine.py` `MooncakeTransferEngine` L117（`KVTransferEngine` 子类）——它不做集合通信，是 PD 数据面的传输底座（§6），集合通信与 KV 传输共用一个 device_communicators 目录但语义两族。

## 3. eplb/ 与 elastic_ep/：expert 重分布与弹性

EP 把 expert 切到各卡后，负载不均随流量漂移——谁测量、谁决策、谁搬权重？所谓 EPLB（expert parallelism load balancer），是指按运行时负载统计重分布 expert 副本以均衡各卡计算的机制（算法账与生产配置归 7.4）。SGLang 把三件事拆成三个模块。

测量：`eplb/expert_distribution.py` `ExpertDistributionRecorder` L135，统计路径按 dispatch 模式分五种 gatherer——`_DetailSinglePassGatherer` L489 / `_LayerBasedCpuSinglePassGatherer` L576 / `_LayerBasedGpuSinglePassGatherer` L605 / `_DeepepNormalSinglePassGatherer` L660 / `_DeepepLowLatencySinglePassGatherer` L693。DeepEP normal 与 low-latency 两种模式的计数位置不同（CPU 侧精确尺寸 vs GPU 侧 masked 计数，3.3 recap），统计器必须跟着模式走，否则读到的计数语义不一致。决策：`eplb/eplb_manager.py` `EPLBManager` L29 持 model / expert_location_updater / expert_backup_client / weight_updater 四个协作对象（经 getter 延迟读取），构造时断言 `eplb_rebalance_num_iterations >= expert_distribution_recorder_buffer_size`——重分配窗口必须覆盖统计窗口，否则环形缓冲混入陈旧数据（源码注释逐字「Otherwise, the circular buffer will contain stale data」）。算法：`eplb/eplb_algorithms/deepseek.py` `rebalance_experts_hierarchical` L86 / `rebalance_experts` L171，即 DeepSeek EPLB 的冗余 expert 副本 + 分层均衡算法；执行侧 `expert_location.py` `ExpertLocationMetadata` L74 + `expert_location_dispatch.py` / `expert_location_updater.py` 把新布局落到 dispatch 表与权重搬运。EPLB 的算法账与生产配置（冗余数 / 统计窗 / 重分配周期 / moe_lb 实测）归 7.4 详展，本章只交付引擎侧模块结构；训练侧负载不均衡 recap training 5.4。另有 `lplb_solver.py` 与 `eplb_simulator/`（布局求解与仿真）。

弹性：`elastic_ep/elastic_ep.py` 把「扩缩 EP 域」做成状态机——`ScaleCohort` L30（一起扩缩的 rank 组，msgspec frozen Struct）/ `register_scale_cohort` L37 / `validate_scale_cohort_topology` L65 / `ElasticEPState` L99 / `ElasticEPStateManager` L132；`update_dp_attention_for_elastic_ep` L77 在扩缩后更新 attention DP 宽度——弹性 EP 与 §1 的配置面联动：world 变了，派生量跟着变。expert 权重备份在 `expert_backup_manager.py` / `expert_backup_client.py`（新 rank 加入时从备份取 expert，不重读 checkpoint）。弹性伸缩的触发账与容错哲学归 8.5；vLLM 对照件是 `distributed/elastic_ep/`（5.3 一句注范围）。

## 4. PD 角色与 bootstrap 时序

PD 分离（PD disaggregation）是指 prefill 与 decode 部署在不同实例、prefill 算完把 KV 传给 decode 继续生成的部署形态（干扰消除与 goodput 账归 8.2，本节走实现）。分离后同一条请求出现在两个引擎的队列里——它们靠什么认出彼此？答案是 `bootstrap_room`：`GenerateReqInput` 携带 `bootstrap_host` / `bootstrap_port` / `bootstrap_room` 三字段（`managers/io_struct.py` L350-352），路由层给同一请求的 P/D 两侧分配同一 room，room 是贯穿全程的关联键。所谓 bootstrap，是指 P/D 两侧按 room 交换端点与传输元数据的控制面握手——KV 数据本身不走这条通道（见下）。

角色是进程级配置：`disaggregation/mode.py` `DisaggregationMode` 三值 NULL / PREFILL / DECODE。两侧各有一条专用事件循环（overlap 变体与 overlap scheduler 机制本身归 6.2）：

- prefill 侧 `event_loop_normal_disagg_prefill`（`prefill.py` L856）：`ingest_requests` → `disagg_prefill_bootstrap_queue.move_bootstrapped(waiting_queue)` → `get_next_disagg_prefill_batch_to_run`（内部先对上一批调 `process_prefill_chunk` L1661 → `send_kv_chunk` L1753，再组新批）→ `run_batch` → `process_batch_result` → `process_disagg_prefill_transfer_queue`（poll 到 Success 才释放请求）。`PrefillBootstrapQueue` L270 为每条请求建 sender 并走 bootstrap 注册。
- decode 侧 `event_loop_normal_disagg_decode`（`decode.py` L4063）：`ingest_requests` → `process_decode_queue` L4541 → `get_next_disagg_decode_batch_to_run` → `run_batch`。`process_decode_queue` 内：`DecodePreallocQueue` L471 **先预分配 KV 槽**再收 KV，`DecodeTransferQueue` L3303 poll 传输状态，`move_transferred` L3610 只把 poll 到 Success 的请求放进 waiting_queue。

撮合发生在 bootstrap server：`BaseKVBootstrapServer`（`base/conn.py` L339）的公共实现 `CommonKVBootstrapServer`（`common/conn.py` L3244）是一个 aiohttp `web.Application` + dict store，由 `managers/disagg_service.py` `start_disagg_service` 在 prefill 侧启动（注释逐字「Start kv bootstrap server on prefill」）；prefill 的 KV manager 经 `register_to_bootstrap`（`base/conn.py` L173）注册服务器信息，decode 的 receiver 按 room 查询对端端点。`common/bootstrap.py` `BootstrapRoom` L18 / `DeferredBootstrap` L27 的模块 docstring 把边界说死（逐字）：「This table only joins a completed optimistic prefill with decode's control endpoint. It performs no I/O, including while holding its lock.」——**bootstrap 是纯控制面**，只做端点撮合；KV 数据面是 prefill GPU 到 decode GPU 的点对点 RDMA，不经过 bootstrap server。

```
时间 →
prefill 实例                 bootstrap server              decode 实例
sender.init(room) ──注册────► │                             │
                              │ ◄──查询(room)── receiver.init
                              │ ──撮合: room → 端点──►      │
                              │                    prealloc KV 槽 + metadata buf
send ◄──────────── send_metadata(目的索引) ──────────────── │
chunk 0 forward ─► send_kvcache ═══ RDMA 直写 ═══════════► │ poll: Transferring
chunk 1 forward ─► send_kvcache ═════════════════════════► │
...（chunk 流水，下图）                                     │
last chunk + aux ════════════════════════════════════════► │ poll: Success
poll: Success → 释放 KV                                   │ PREBUILT → DECODE
```

decode 侧的准入有一条不变式（`move_preallocated` L1653 注释逐字）：「the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request. Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.」——预分配必须给在跑请求的最坏增长留够余量，因为 transfer queue 里的请求已持有 KV 页、不可回收（retract 机制 recap 4.2）。这是设计规则而非调度策略：违反它不是慢，是死锁式 OOM。

KV 就绪后，decode 侧 `get_next_disagg_decode_batch_to_run` 调 `new_batch.prepare_for_prebuilt()`（`decode.py` L4458 → `decode_schedule_batch_mixin.py` L24），置 `ForwardMode.PREBUILT`（`forward_mode.py` L38，6.1 ForwardMode 表 recap）。源码注释逐字「PREBUILT never enters a model forward」：PREBUILT 只做 metadata 填充（从 `req_to_token_pool` 读已传输 KV 的槽位映射、算 seq_lens），不跑 prefill forward——因为**第一个输出 token 已在 prefill 侧采样**，其 id / logprobs 经 aux 通道传来（§5），decode 的第一次真实 forward 已是第二个 token 的 DECODE 步。

chunk 流水是传输与计算重叠的实现形态（口径：「暴露」指未被计算藏住、计入 TTFT 的传输时间）：

```
时间 →
prefill SM    [chunk 0 fwd][chunk 1 fwd][chunk 2 fwd][chunk 3 fwd]
RDMA NIC            [chunk 0 KV]  [chunk 1 KV]  [chunk 2 KV]  [chunk 3 KV]
decode 侧     prealloc ──────────────────────────────────► Success → DECODE
```

`send_kv_chunk`（`prefill.py` L1753）在每个 chunk 算完后的下一轮调度开头发送（`process_prefill_chunk` 处理上一批的 chunk），非末 chunk 的发送边界页对齐（`end_idx = end_idx - end_idx % page_size`，尾部不满页推迟到下一 chunk）；device 驻留前缀甚至可以在本步 forward 入队前早发（L1720-1751）。早发有一个 WAR 陷阱，源码注释逐字：「under overlap scheduling the PRIOR step's prefill forward may still be writing these prefix pages on forward_stream. Record a completion event now so the transfer worker can wait on those writes before the RDMA read, instead of racing them.」

> **坑**：overlap 调度下早发前缀 KV 不做 stream 同步——上一步 forward 可能还在写这些页，RDMA 读到半新半旧的字节。触发条件：`enable_overlap` + 早发路径；后果：decode 侧 KV 静默损坏（checksum 才兜得住）。规避：prefill 在 forward_stream 上 record event，transfer worker 等 event 再读（L1747-1750）；正确性兜底是末 chunk 的 KV checksum（§5）。

**命题 2**（chunk 划分恰好覆盖）：设发送范围 $[T_{\mathrm{hit}}, T)$（ $T_{\mathrm{hit}}$ = decode 侧 bootstrap 时承诺的已有前缀长，`prefill.py` L546 `start_send_idx = decode_prefix_len`），各次 `send_kv_chunk` 的边界为 $s_0 = T_{\mathrm{hit}} \le s_1 \le \cdots \le s_k = T$，则各 chunk 区间恰好覆盖 $[T_{\mathrm{hit}}, T)$——每 token 恰好发送一次、总字节 $\kappa L (T - T_{\mathrm{hit}})$ 与 chunk 数无关。**证明**（路线：telescoping + 对齐推迟 + 末 chunk 收全，三步）：基例：首次发送从 $s_0$ 起。保持：每次发送后 `req.start_send_idx = end_idx`（L2084），下一次从上一次终点起——区间首尾相接， $\sum_i (s_{i+1} - s_i) = s_k - s_0$（telescoping）；非末 chunk 的页对齐只把尾部不满页**推迟**到下一 chunk（边界仍是同一个 $s_i$ 序列），不产生重叠也不产生空洞；`clamp_kv_send_end` L148 把非末 chunk 边界夹在 radix 去重保护区内，同样只推迟、不跳段。终止：`last_chunk=True` 时 `end_idx = min(req.extend_end, len(origin_input_ids))` 不做对齐，收全剩余， $s_k = T$。故总字节 = $\kappa L (T - T_{\mathrm{hit}})$，与划分无关。证毕。边界反例：若非末 chunk 提前发送不满页，页是传输最小单元，页内未算出的槽位会随页一起传走（正确性错误）或下一 chunk 重发该页（带宽浪费）——页对齐推迟规则同时消掉两者。 $b$ = 1（SGLang 默认 page = 1 token，论文口径 [Zheng et al., 2023](../references.md)，[A]，6.1 走读）时对齐条件恒真，命题退化为平凡划分。代入本章口径： $T$ = 32768、 $T_{\mathrm{hit}}$ = 0、 $c_{\mathrm{chunk}}$ = 8192（示例值，`prefill.py` staging 代码的 fallback 同值； $c_{\mathrm{chunk}}$ = `chunked_prefill_size`，章内局部记号，与 CP 度 $c$ 区分）：4 个 chunk 各 8192 token × 70272 B = 575668224 B ≈ 549 MiB，合计 2302672896 B ≈ 2.14 GiB——与整段一次发送的字节数恒等。

## 5. KV 传输接口：base/conn.py 的四个抽象与一组状态

341 行的 `disaggregation/base/conn.py` 定义了 PD 数据面的全部契约——六个符号各管什么？

`KVArgs` L62 是注册描述符：每 rank 把本地 KV buffer 的指针 / 长度 / 单条目长（`kv_data_ptrs` / `kv_data_lens` / `kv_item_lens`）+ 层号（`kv_layer_ids`）+ dtype 与布局（`kv_cache_dtype_str` / `kv_layout`，注释逐字「KV byte layout both PD roles must share」）注册给传输引擎；aux 通道（`aux_data_ptrs` 等）与混合注意力状态（`state_types`，`StateType` L17 枚举 SWA / DSA / MAMBA / BLOCK_SCALE 等——SWA 窗口态、DSA indexer、Mamba 卷积态都走同一接口）并列注册。几何与拓扑字段：`page_size`、`kv_head_num` / `total_kv_head_num`（TP 切片信息）、`ib_device` / `gpu_id`、`pp_rank` / `prefill_start_layer` / `prefill_end_layer`（PD × PP：每 PP stage 只传自己的层范围）、`host_kv_data_ptrs`（direct-to-host，2.4 §9 HiSparse PD 集成 recap）。一个容易漏读的字段：`state_dim_per_tensor` L94 注释逐字「Per-tensor TP slice dim, used when prefill/decode attn_tp_size differ」——**P/D 两侧允许不同构并行度**（如 prefill TP4、decode DP8），切片元数据让 sender 按目的端宽度重排字节。

`KVPoll` L142 是五态状态机：`Failed = 0` / `Bootstrapping = 1` / `WaitingForInput = 2` / `Transferring = 3` / `Success = 4`。状态按 room 记在 manager 的 `request_status` 表里（`common/conn.py` `CommonKVManager` L311 `update_status`），sender 与 receiver 各自 poll 同一组词；正常路径 Bootstrapping → WaitingForInput → Transferring → Success，Failed 从任意状态可达（abort / 超时）。四个抽象类：

| 抽象 | 锚点 | 职责 |
| --- | --- | --- |
| `BaseKVManager` | L150（`register_to_bootstrap` L173） | 注册 buffer、管传输状态、撮合信息 |
| `BaseKVSender` | L191（`init` L202 / `send` L209 / `poll` L240） | prefill 侧：注册 room → 按目的索引写 KV |
| `BaseKVReceiver` | L266（`init` L289 / `send_metadata` L299 / `poll` L313） | decode 侧：解析 bootstrap → 发布目的索引 → poll |
| `BaseKVBootstrapServer` | L339 | 控制面撮合（§4） |

接口的关键语义在 `send_metadata` 的方向：**decode 指定位置，prefill 直写**。receiver 把预分配好的 `kv_indices`（目的槽位）经 `send_metadata` 发给 prefill（docstring 逐字「Notify the prefill server about the kv indices, aux index, and state_indices」），sender 的 `send` 随后把 KV 写到那些索引对应的远端地址——地址协商是 pull 式的询问，数据面是 push 式的 RDMA write。`KVTransferDestination` L57 再分 DEVICE / HOST 两个目的地（direct-to-host 时 KV 落 decode 的 host pool，绕过 decode GPU）。协议公共层在 `common/conn.py`（3631 行）：`CommonKVManager` L311 / `CommonKVSender` L2325 / `CommonKVReceiver` L2698 / `CommonKVBootstrapServer` L3244 实现 bootstrap 注册 / ack / abort 协议（`AbortNotification` L196 / `AbortAck` L264 / `SendRecord` L300 / `PrefillServerInfo` L126），backend 只实现传输本身。decode 侧发布目的索引还有一步跨 rank 协同：`publish_pending_metadata`（`decode.py` L3614）对 ready 标志做 all-reduce MIN，注释逐字「Publish on every rank together so an abort cannot leave only some ranks waiting for acknowledgements from a published destination」——要么全 rank 一起发布，要么都不发布。

接口对账（KV 接口 ↔ 传输账，带宽需求全账归 8.3）。发送字节由两部分构成：

$$
\mathrm{Bytes}_{\mathrm{send}} = \kappa L (T - T_{\mathrm{hit}}) + \mathrm{Bytes}_{\mathrm{aux}} \qquad (4)
$$

符号表： $\kappa L (T - T_{\mathrm{hit}})$ = KV 本体（命题 2， $\kappa$ recap 1.2 式 (3)）；$`\mathrm{Bytes}_{\mathrm{aux}}`$ = aux 通道字节。aux 装的是**第一个输出 token 的元数据**：`MetadataBuffers`（`disaggregation/utils.py` L639）注释逐字「We transfer the metadata of first output token to decode」「The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes」——`output_ids` 槽位 16 × int32 = 64 B，另有 cached_tokens / logprobs；需要 hidden states 时（投机续跑）`output_hidden_states` 每请求 7168 × 2 = 14336 B = 14 KiB。aux 合计约 14.4 KB，对 2.14 GiB 的 KV 本体占比约 $6 \times 10^{-6}$——传输账由 KV 本体主导，aux 忽略不计（口径：大 KV、带宽项主导，同 2.4 §3）。传输时间即 2.4 式 (3) 解出时间的 recap 形：

$$
t_{\mathrm{send}} = \frac{\kappa L (T - T_{\mathrm{hit}})}{\mathrm{BW}} \qquad (5)
$$

符号表： $\mathrm{BW}$ = 有效传输带宽（GB/s， $10^9$ 口径）。代入本章传输口径： $T$ = 32768、 $T_{\mathrm{hit}}$ = 8192（decode 侧 radix 命中，2.2 recap——PD 分离不消灭 prefix caching，反而让命中直接折成传输字节），发送 24576 token × 70272 B = 1727004672 B ≈ 1.61 GiB。带宽取 400 Gbps = 50 GB/s（DeepInfer 跨 DC 官方披露口径 recap 2.4 §3，[DeepSeek-AI, DeepInfer doc](../references.md)，教学参照，2026-11-23 核实）： $t_{\mathrm{send}}$ ≈ 34.5 ms；按每 8 GPU 共享 400 Gbps 的每 GPU 份额 6.25 GB/s 计则 ≈ 276 ms。chunk 流水（§4）把暴露时间压到「末 chunk + aux + poll 间隔」：每 chunk 575.7 MB 在 50 GB/s 下 11.5 ms，藏进下一 chunk 的 forward；decode 侧 poll 间隔由 `disaggregation_decode_polling_interval` 配置（`process_decode_queue`），是 Success 到进批之间的固定延迟项。带宽需求全账（TTFT 预算份额 / layer-wise 重叠 / 跨 DC 可行性）归 8.3，此处只交付接口与字节的对应。

页粒度决定描述符数量。 $b$ = 64 时 32768 token = 512 页，每页每层 1152 × 64 = 73728 B = 72 KiB；若逐页逐层发，描述符数 = 512 × 61 = 31232——正是 2.4 §4 细粒度拷贝坑的量级。mooncake backend 的对策是聚合：`_send_kvcache_generic`（`mooncake/conn.py` L1342）先 `group_concurrent_contiguous` 把连续页索引合并成块再提交 BatchTransfer（2.4 §5 Segment / BatchTransfer recap），连续布局下每层按连续段聚成大块、一段一个描述符。正确性兜底是 checksum：`send_kv_chunk` 末 chunk 对发送范围算 digest（`prefill.py` L1753-1779，注释逐字「decode digests the same range of its row, past the prefix it promised at bootstrap」），decode 侧对同范围校验；`should_verify_kv_checksum`（`base/conn.py` L156）在 P/D TP 宽度不一致时返回 False——digest 覆盖的字节范围与本地切片不可比，校验跳过而非误报。

## 6. mooncake / nixl：同一接口下的两个 backend

`TransferBackend` 枚举（`disaggregation/utils.py` L1006）列 mooncake / mori / nixl / ascend / fake 五值——两个主 backend 各实现什么？

**mooncake**（`disaggregation/mooncake/conn.py`，4060 行）：`MooncakeKVManager` L364（`StagingManagerMixin` + `CommonKVManager`）；`init_engine` L519 起 transfer engine、`register_buffer_to_engine` L547 把 `KVArgs` 的 buffer 注册进去；发送族按场景分件——`send_kvcache` L1675（整池页索引 → 目的指针）、`send_kvcache_staged` L814（staging buffer 中转）、`send_kvcache_dcp` L1845（decode 侧 CP）、`send_kvcache_slice` L2204（TP 切片重排）、`send_aux` L2344（元数据，另有 TCP 变体 `send_aux_tcp` L2383）；`MooncakeKVSender` L4015 / `MooncakeKVReceiver` L4055 / `MooncakeKVBootstrapServer` L4059 是四抽象的 mooncake 实现。传输底座即 §2 的 `MooncakeTransferEngine` L117；Segment / BatchTransfer / 拓扑感知多 NIC 池化的机制 recap 2.4 §5（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]），不重推。

> **坑**：给 MLA 模型开 `SGLANG_DISAGG_STAGING_BUFFER`——`prefill.py` L322-325 直接 raise，错误信息逐字「SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models (e.g. GQA, MHA). MLA models should not set this flag.」触发条件：MLA 单 latent buffer 布局与 staging 网格不匹配；规避：MLA 走 `send_kvcache` 直传路径。

**nixl**（`disaggregation/nixl/conn.py`，3856 行）：`NixlKVManager` L531（同 mixin 结构）；`register_buffer_to_engine` L2153 / `send_kvcache` L2505 / `send_kvcache_dcp` L2562 / `send_kvcache_mixed` L2633 / `send_kvcache_staged` L2782 / `send_aux` L2981；`NixlKVSender` L3843 / `NixlKVReceiver` L3851。NIXL 是 UCX / GDR / RDMA 的传输抽象层（[NVIDIA, ai-dynamo/nixl repo](../references.md)，教学参照 + 官方实现，2026-11-23 核实），agent API 与 descriptor 语义归 8.3 详展。两 backend 对上层完全同构：`KVArgs` 注册、`KVPoll` 状态、`send_metadata` 地址协商一字不差，差异只在传输引擎与 buffer 注册方式——选型跟着部署栈走（Mooncake 生态 / Dynamo 生态），不跟着接口走。fake backend 供测试（`FAKE_BOOTSTRAP_HOST = "2.2.2.2"`，`mode.py`）；mori / ascend / encoder / transport 子目录已扩展（2026-11-23 核实），超出本章范围。

## 7. connector/ 与 checkpoint_engine/：权重通道，不是 KV

`connector/base_connector.py` 里有个类叫 `BaseKVConnector`——它是 PD 的 KV 传输接口吗？不是。这是本章最容易踩的语义坑：类名同形，语义两族。

> **坑**：按类名搜 PD 接口会命中 `connector/base_connector.py` `BaseKVConnector` L75——它是 redis 型 key-value **存储** connector，装的是权重 / 文件字节，「KV」是存储语义不是 attention KV。触发条件：类名同形；后果：把权重加载通道误判为 PD 数据面。规避：PD KV 传输接口只在 `disaggregation/base/conn.py`（§5），两族接口无继承关系。

`connector/` 的真实职责是权重 / 文件存储：`BaseConnector` L13 的 docstring 给出 URL 形态（换行归一）「For fs connector such as s3: `<connector_type>://<path>/<filename>`；For kv connector such as redis: `<connector_type>://<host>:<port>/<model_name>/keys/<key>`」，抽象方法只有两个——`weight_iterator(rank)`（逐 rank 吐权重张量）与 `pull_files()`（拉文件到本地临时目录）。实现族：`S3Connector`（`s3.py` L69）/ `RedisConnector`（`redis.py` L16）/ `AzureBlobConnector`（`azure.py` L66）/ `RemoteInstanceConnector`（`remote_instance.py` L17，从远端实例取权重）。

`checkpoint_engine/` 是权重**更新**通道：`checkpoint_engine_worker.py` L15-16 docstring 逐字「Checkpoint-engine integration for SGLang. This module provides weight update functionality via IPC for checkpoint-engine compatibility.」它集成外部 MoonshotAI/checkpoint-engine（`pip install sglang[checkpoint-engine]`，import `checkpoint_engine.worker.update_weights_from_ipc`）；`checkpoint_engine_worker.py` `SGLangCheckpointEngineWorkerExtension` L39 / Impl L95 走 zmq IPC；`update.py` 是 torchrun CLI（`split_checkpoint_files` L79 / `split_tensors` L92 / `req_inference` L113 / `update_weights` L144 / `join` L182）。引擎侧的权重更新面在 `managers/scheduler_components/weight_updater.py` `SchedulerWeightUpdaterManager` L131（`update_weights_from_disk` L182 / `init_weights_update_group` L210 / `flush_cache_after_weight_update` L164）；`Engine.update_weights_from_ipc`（`entrypoints/engine.py` L1790）与 `release_memory_occupation` L1899 / `resume_memory_occupation` L1905（tags 分池释放 / 恢复显存）成对。权重更新后必须 flush KV cache——权重是 cache 版本元组的分量（2.4 §8 recap），旧权重算出的 KV 对新权重无效。vLLM 对照件是 sleep / wake_up（`CuMemAllocator` tags 分池）+ `weight_transfer/` 引擎族（5.4 走读）；RL rollout 权重同步的训练侧 recap training 8.5，训推共置的完整流程归 9.3。

## 8. compilation：piecewise 的引擎侧实现

3.4 已交付 piecewise 机制与 replay 正确性命题（指针固定 + 内容可见 + 顺序保持）——SGLang 的实现件长什么样？本节只做引擎调用级走读，机制与命题 recap 3.4，不重推。

配置面 `compilation/compilation_config.py` `CompilationConfig` L20：`capture_sizes`（捕获的 batch size 列表，shape 特化）、`compiler`（"eager" / "inductor"）、`split_ops`（初始化为 `SPLIT_OPS`，attention 算子列表——3.4 切分点的 SGLang 对应物）。实现件：`backend.py` `SGLangBackend` L371（torch.compile 的自定义 backend）/ `CompilerManager` L82 / `PiecewiseCompileInterpreter` L279（`torch.fx.Interpreter` 子类，按 `SplitItem` L223 切段）；`cuda_piecewise_backend.py` `CUDAPiecewiseBackend` L49（`ConcreteSizeEntry` L33 按具体 size 捕子图）；`compile_phase.py` / `pass_manager.py` / `inductor_pass.py` 管编译阶段与融合 pass。inductor 路线开 horizontal fusion（`configure_inductor`，`compilation_config.py` L50 起：`combo_kernels = True`，注释例「fusing q_norm + k_norm into a single triton kernel」）。与 vLLM 对照（3.4 已交付其配置面）：vLLM 是 CompilationMode 四级 × CUDAGraphMode 五模式的双枚举矩阵，SGLang 是 piecewise 单路线 + compiler 二值——表达力换简单性；vLLM 的 FULL_DECODE_ONLY 给 PD 的 D 实例省图内存（3.4 注）；SGLang 的 decode 实例 `capture_sizes` 同样只需覆盖 decode 批 size（结构推论）。npu / xpu 各有 piecewise backend 变体（`npu_piecewise_backend.py` / `xpu_piecewise_backend.py`，双平台一句注）。`batch_invariant_ops/` 的批不变 kernel 归 3.4 / 6.2，不展开。

## 9. dllm 与范围边界

`dllm/` 是 masked diffusion LLM 的 serving 支持（`DllmConfig` `config.py` L8：algorithm / block_size / mask_id / max_running_requests / first_done_first_out_mode；`DllmAlgorithm` `algorithm/base.py` L30 的 step / run 骨架，LowConfidence 等算法按置信度逐步 unmask）——非自回归 serving 的展望归 9.4，本章一句登记。范围外新模块一句注（2026-11-23 核实，目录已扩展）：`monokernels/` / `kv_canary/` / `rust_server/` / `weight_sync/` / `session/` / `beam_search/` 与 `disaggregation/{mori,ascend,encoder,transport}/`。

## 10. 与 vLLM / DeepInfer 对照：为 8.4 立基线

三个引擎的分布式 + PD 差异不在功能清单，而在接口形态。下表是 8.4 系统对照的基线（8.4 只 recap 本表并给选型论断，不重复展开）；vLLM 列锚点归 5.3 / 5.4 走读，DeepInfer 列已交付事实 recap 2.4、未交付部分纯文字指引。

| 轴 | SGLang（本章） | vLLM（5.3 / 5.4） | DeepInfer（2.4 recap / 7.x） |
| --- | --- | --- | --- |
| 并行配置面 | `attn_dp_size` / `ep_size` 派生面，`tp_size` = world（§1） | `config/parallel.py` 显式字段 + `enable_expert_parallel` | 不用 TP，Wide-EP + DP attention（rationale → 8.1，部署 → 7.4） |
| 组结构 | `GroupCoordinator` L271，同 world 双族划分（§1） | `GroupCoordinator` L427 + 每 DP rank 独立 EngineCore + DPCoordinator | → 7.1 |
| 小消息通信 | CustomAllreduce / V2 + TorchSymmMem（§2） | CustomAllreduce L120 + SymmMemCommunicator L29 | DeepEP + Mega MoE（3.3 / 7.3 recap） |
| EPLB | `eplb/` DeepSeek 算法 + 五统计路径（§3） | `distributed/eplb/` `EplbState` L321 | 生产 EPLB 配置 → 7.4 |
| 弹性 EP | `elastic_ep/` ScaleCohort（§3） | `distributed/elastic_ep/`（5.3 注） | EP failover → 8.5 |
| PD 接口 | `base/conn.py` 四抽象 + KVPoll，sender/receiver 双向（§5） | `KVConnectorBase_V1` L182 + KVConnectorRole，scheduler/worker 分件 | 存储为中心；编排层不在 release → 7.1 / 8.4 |
| KV 传输 backend | mooncake / nixl（+ mori / ascend）（§6） | nixl push/pull 双模 + MooncakeConnector L798 + MultiConnector L142 组合 | 3FS kvcache-store + GDR swap-in（2.4 §7-9 recap） |
| 权重更新 | checkpoint_engine + release/resume_memory_occupation（§7） | sleep/wake_up + weight_transfer 引擎族 | → 9.3 |
| 编译 | piecewise 单路线（§8） | CompilationMode × CUDAGraphMode 双枚举 | TileLang custom ops（3.4 / 7.3 recap） |

三条接口形态论断。其一，SGLang 的 PD 是**引擎内置的双向协议**：sender/receiver 在同一个 scheduler 事件循环里 poll，bootstrap 撮合与 KV 传输都是引擎自己的组件——PD 是一等部署模式，不是插件。其二，vLLM 的 PD 是**connector 抽象**：KVConnectorBase_V1 按 scheduler/worker 角色分件，nixl push/pull 双模 + MultiConnector 组合多后端（5.4 走读）——通用性换接口层数。其三，DeepInfer 走**存储为中心**：KV 经专用 3FS 集群中转而非点对点 RDMA 直传，任何 prefill 机命中任何前缀、存储与计算 autoscaling 解耦（2.4 §8 三理由 recap）——代价是多一跳网络与独立存储硬件。点对点直传适合集群内 P/D 低延迟交接，存储中转适合跨集群前缀共享与故障域隔离；PD 原理账（goodput / xPyD 配比 / 传输 vs 重算临界）归 8.2，传输带宽账归 8.3，系统级对照与选型归 8.4。

## 总结

| 论断 | 账 / 证据 | 回指 |
| --- | --- | --- |
| `tp_size` = world size；`enable_dp_attention` 是派生量（`attn_dp_size > 1`），不是字段 | `server_args.py` L345/L355；`parallel_hook.py` L136-158 / L389 断言 | §1、坑注 |
| attention TP 度 = `tp_size // attn_dp_size // attn_cp_size`；rank 布局 (dp, cp, tp) 混合进制双射 | 式 (1)(2)、命题 1 | §1、练习 1 |
| DP attention 每 rank KV = $\kappa L T B / d$，复制因子 = `attn_tp_size` | 17.16 GiB（ $d$=1）vs 2.14 GiB（ $d$=8）；blog 1.9× decode 吞吐 [C] | 式 (3)、§1 |
| serving 无 backward：每层 2 次前向 all-reduce；decode 消息亚 MB、延迟项主导 | 0.875 MiB @ B=64 / 7168 / bf16；全账 → 8.1 | §2 |
| EPLB 三件分离：统计（五 gatherer 按 DeepEP 模式）/ 决策（重分配窗 ≥ 统计窗）/ 执行 | `ExpertDistributionRecorder` L135、`EPLBManager` L29 断言 | §3 |
| bootstrap 是纯控制面（room 撮合、无 I/O）；KV 数据面点对点 RDMA 直写 | `BootstrapRoom` docstring 逐字；`CommonKVBootstrapServer` L3244 | §4 |
| decode 先预分配再收 KV；准入须留够在跑请求最坏增长（transfer queue 不可 retract） | `move_preallocated` L1653 注释逐字 | §4 |
| chunk 划分恰好覆盖 $[T_{\mathrm{hit}}, T)$，总字节与 chunk 数无关；页对齐推迟消重叠与空洞 | 命题 2；4 × 549 MiB = 2.14 GiB | §4、练习 3 |
| PREBUILT 不进 model forward；第一个输出 token 在 prefill 侧采样、经 aux 传递 | `decode_schedule_batch_mixin.py` L32 注释；`MetadataBuffers` L639 | §4、§5 |
| PD 接口 = KVArgs + KVPoll 五态 + 四抽象；decode 指定位置、prefill 直写；P/D 可异构并行度 | `base/conn.py` L62/L142/L150-339；`state_dim_per_tensor` L94 | §5、练习 4 |
| 发送字节 = $\kappa L (T - T_{\mathrm{hit}})$ + aux（占比 ~ $6 \times 10^{-6}$）；checksum 覆盖发送范围 | 式 (4)(5)；1.61 GiB → 34.5 ms @ 50 GB/s / 276 ms @ 6.25 GB/s | §5、练习 2 |
| mooncake / nixl 同接口异构传输；staging 仅限非 MLA；连续页聚合成块消细粒度描述符 | `TransferBackend` L1006；staging raise 逐字；`group_concurrent_contiguous` | §6 |
| `connector/` = 权重/文件存储（BaseKVConnector 是 redis 型 store）；checkpoint_engine = 权重更新 IPC | 类名同形语义纠正；docstring 口径 | §7、坑注 |
| SGLang piecewise 单路线 vs vLLM 双枚举矩阵；机制与 replay 命题 recap 3.4 | `CompilationConfig` L20 / `SGLangBackend` L371 | §8 |
| 接口形态三路线：引擎内置双向协议 / connector 抽象 / 存储为中心 | 对照表 + 三条论断；8.4 基线 | §10 |

## 练习

1.（计算 · [10]）配置派生代入：`tp_size` = 16、`attn_dp_size` = 4、`attn_cp_size` = 2。(a) 按式 (1) 求 attention TP 度；(b) 按式 (2) 求 `tp_rank` = 13 的 (dp, cp, tp) 坐标并回代验证；(c) 判断 `ep_size` = 4、`moe_dp_size` = 2 是否满足 `parallel_hook.py` L153/L159 断言。校验：`runtime_context.py` `derive_attn_tp_size` L73 / `derive_attention_ranks` L79 源码；(b) 回代应重构 13。
2.（计算 · [20]）传输账代入（本章口径 $\kappa L$ = 70272 B/token）： $T$ = 32768、 $T_{\mathrm{hit}}$ = 8192、 $c_{\mathrm{chunk}}$ = 8192、 $b$ = 64。(a) 按式 (4) 求发送字节（忽略 aux）并换算 GiB；(b) 按式 (5) 求 50 GB/s 与 6.25 GB/s 下的 $t_{\mathrm{send}}$，先写出 (b) 的数量级预测再计算；(c) 求页数与每页每层字节。校验：1.2 §3 几何表 V3 行；2.4 式 (3)；(a) 应等于 3 个整 chunk 之和（命中恰好折掉一个 chunk）。
3.（推导 · [30]）命题 2 证明闭环：(a) 写出 telescoping 三步证明（基例 / 保持 / 终止），指明每步对应的源码行；(b) 给出边界反例：非末 chunk 提前发送不满页时，页内未算出槽位与重发各造成什么后果；(c) 说明 $b$ = 1 时命题为何退化为平凡划分、页对齐规则为何恒真。校验：`prefill.py` `_send_kv_chunk` L1799 页对齐规则 + L2084 `start_send_idx` 更新；(c) 与 6.1 page = 1 token 口径一致。
4.（代码 · [30]）KVPoll 状态机走读：读 `base/conn.py` L142 与 `common/conn.py` 的 CommonKVManager / Sender / Receiver。(a) 列出 sender 与 receiver 各自的正常路径状态序列与触发事件（注册 / send_metadata / 传输完成）；(b) 定位 Failed 分支对应的协议消息（`AbortNotification` L196 / `AbortAck` L264）；(c) 解释 `decode.py` `publish_pending_metadata` L3614 为何对 ready 标志做 all-reduce MIN 而非各 rank 独立发布。校验：file:line 锚点（2026-11-23 tarball）；(c) 源码注释逐字「Publish on every rank together so an abort cannot leave only some ranks waiting」。

## 参考

- [Zheng et al., 2023](../references.md)：SGLang，arXiv:2312.07104，[A]（key: sglang）——论文锚点；page = 1 token 口径（6.1 走读 recap）
- [SGLang, sgl-project/sglang repo](../references.md)：key: sglang-repo，[A]——本章全部代码锚点（main，fresh tarball file:line 核实 2026-11-23）：`arg_groups/` + `runtime_context.py` 配置面、`distributed/parallel_state.py` + `device_communicators/`、`disaggregation/{base,common,mooncake,nixl}/conn.py` + `prefill.py` / `decode.py`、`eplb/` / `elastic_ep/`、`connector/`（权重/文件存储）、`checkpoint_engine/`、`compilation/`、`dllm/`
- [SGLang Team, 2024](../references.md)：SGLang v0.4 blog，[C]（开放技术博客、非闭源）（key: sglang-v04-blog）——DP attention 机制（all-gather / redistribute）与 1.9× decode 吞吐（8×H100，DeepSeek-Coder-V2-Instruct-FP8）；`--enable-dp-attention` 历史 flag
- [Qin et al., 2024](../references.md)：Mooncake，arXiv:2407.00079，[A]（key: mooncake）——transfer engine（Segment / BatchTransfer / 拓扑感知多 NIC 池化，2.4 §5 recap）；SGLang mooncake backend 传输底座
- [NVIDIA, ai-dynamo/nixl repo](../references.md)：key: nixl，教学参照 + 官方实现——NIXL UCX / GDR / RDMA 传输抽象（agent / descriptor 语义 → 8.3）
- [vLLM, vllm-project/vllm repo](../references.md)：key: vllm-repo，[A]——对照列锚点（`distributed/` / `eplb/` / `elastic_ep/` / `kv_connector/v1/` / compilation / sleep/wake_up；走读归 5.3 / 5.4）
- [DeepSeek-AI, DeepInfer doc](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——400 Gbps / 8 GPU 跨 DC 口径（2.4 §3 recap）；3FS + GDR 存储为中心路线（2.4 recap）
