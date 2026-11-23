# 8.4 PD 分离系统实现对照（PD Disaggregation Systems: A Comparative Walkthrough）

> 同样叫「PD 分离」，vLLM 的 connector、SGLang 的 bootstrap、Dynamo 的 router、TRT-LLM 的 transceiver、DeepInfer 的存储中转、MemServe 的 memory pool——它们到底差在哪？本章按六轴（KV 传输路径 / 调度编排 / 弹性 / 开源完整度 / 硬件支持 / 适用规模）对照六个系统，增量限定为对照表与选型论断；各系统实现要点只 recap + 指针（6.3 / 7.4 已走读内容不重复展开） | 前置：8.2（PD 原理：goodput / xPyD 配比 / 传输 vs 重算阈值）、8.3（KV 传输语义与带宽账）、6.3（SGLang disaggregation 走读）、7.1 / 7.4（DeepInfer 架构与部署）、5.3 / 5.4（vLLM connector）；training P5 只 recap | 锚点：vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、SGLang（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]）、Dynamo（[NVIDIA, ai-dynamo/dynamo](../references.md)，教学参照 + 官方实现，c41b6dfc）、TensorRT-LLM（[NVIDIA, TensorRT-LLM](../references.md)，教学参照 + 官方实现）、DeepInfer（[DeepSeek-AI, DeepInfer](../references.md)，教学参照 + 官方披露）、MemServe（[Hu et al., 2024](../references.md)，arXiv:2406.17565，[B]，无官方 repo）、NIXL / Mooncake（教学参照 / [A]）；代码锚点 2026-11-23 核实

## 1. 对照框架：原理一层，实现一层

六个系统都叫「PD 分离」，为什么不能拿一张功能清单打勾就选型？因为差异不在「有没有」，而在 KV 走哪条路、谁编排、缺哪块开源。8.2 回答了「该不该分离、配多少 P/D」，本章回答「分离之后，各系统怎么把 KV 搬过去、谁编排、能不能弹性」。两层不混：原理账（goodput、xPyD、 $\mathrm{BW}^{\ast}$ ）归 8.2，传输账（egress/ingress、暴露时间）归 8.3，本章只把各系统的实现件对回这两层账，再给选型论断。

**本章口径**：传输带宽沿用 8.3（400 Gbps = 50 GB/s、跨 DC 每 GPU 6.25 GB/s）；实例计数用 $n_P / n_D$ （8.2 约定，禁裸 P/D 当计数）；Dynamo 成本模型用 block 当量（与模型几何无关）；数值实例为教学构造值（逐处声明）。六轴对照框架：

| 轴 | 问什么 | 为什么这轴决定选型 |
| --- | --- | --- |
| KV 传输路径 | 点对点直传还是存储中转？走哪个传输库？ | 决定延迟跳数与带宽账形态（8.3） |
| 调度编排 | 谁选 worker、谁撮合 P/D、元数据怎么握手？ | 决定 KV 复用率与 decode 启动时序 |
| 弹性 | 能否运行时加减 worker、按什么信号伸缩？ | 决定流量摆动下的 SLO 保持（8.5） |
| 开源完整度 | 引擎侧 / 编排侧哪些在 release？ | 决定自建成本与可复现性 |
| 硬件支持 | 哪些 GPU / 互联 / 传输 backend？ | 决定部署栈约束 |
| 适用规模 | 单节点 / 集群内 / 跨 DC？ | 决定拓扑与可靠性预算 |

**两层走读纪律**（spec §4.2）：6.3（SGLang impl）/ 7.4（DeepInfer VE-P-D）/ 5.3 / 5.4（vLLM connector）已做模块全量走读，本章重访一律 recap + 指针；增量只有对照表与选型论断。9.2（其他引擎对照）只 recap 本章 + 一句选型定位，不双展开（C-S3-4）。

## 2. vLLM：connector 抽象与角色分件

vLLM 的 PD 是插件还是内置？答案是 connector 抽象——PD 数据面挂在 `kv_transfer/kv_connector/v1/` 下，按角色分件（recap 5.3 §9 / 5.4，模块走读不重复）。

接口基线（W3 同日 tarball，2026-11-23 再确认）：`base.py` `KVConnectorBase_V1` L182 + `KVConnectorRole` L140（scheduler / worker 两角色）；connector 族——nixl/（`NixlBaseConnector` L79 / `NixlPullConnector` L374 = READ / `NixlPushConnector` L415 = WRITE）、mooncake/（`MooncakeConnector` L798）、mooncake_store/、lmcache_connector.py、hf3fs/、multi_connector.py（`MultiConnector` L142，组合多后端）、hisparse/、offloading_connector.py。实例编排经 `kv_role` + `kv_transfer_config`：prefill 实例与 decode 实例各配 connector，角色决定它是发送方还是接收方。

传输语义对回 8.3：push/pull 双模即 NIXL `NIXL_WRITE` / `NIXL_READ` 的封装（8.3 §3/§7）；MultiConnector 把 PD connector 与 offload connector 组合——一条请求的 KV 既可以从 prefill 直传，也可以经 host/store 分层（2.4 §9 的 vLLM 路线）。

Dynamo 视角的时序口径（disagg-serving.md L60）：vLLM 用 `kv_transfer_params`（block IDs + remote worker 连接信息）作传输元数据，**prefill 同步、decode 等 prefill 完**才继续。这与 SGLang 的 bootstrap 异步形成对照（§3）。

开源完整度：引擎 + connector 全在 release（vllm-project/vllm，[A]）；编排层（多实例路由 / 弹性）不在引擎内，由外部编排器（Dynamo / 自建）承担。

## 3. SGLang：引擎内置的双向协议

SGLang 的 PD 为什么不是插件？因为 sender/receiver 在同一个 scheduler 事件循环里 poll，bootstrap 撮合与 KV 传输都是引擎自己的组件——PD 是一等部署模式（recap 6.3 §4-6 / §10，走读不重复）。

接口基线（W3 同日 tarball）：`disaggregation/base/conn.py` 四抽象 + 一组状态——`KVArgs` L62（注册描述符：ptr/len/item_len/layer/dtype/layout + TP 切片 + PP 层范围）/ `KVPoll` L142（Failed/Bootstrapping/WaitingForInput/Transferring/Success 五态）/ `BaseKVManager` L150 / `BaseKVSender` L191（send L209 / poll L240）/ `BaseKVReceiver` L266（send_metadata L299）/ `BaseKVBootstrapServer` L339；公共实现 `common/conn.py`，backend mooncake（`MooncakeKVManager` L364）/ nixl（`NixlKVManager` L531）同构。

bootstrap 时序 recap（6.3 §4 图，不重绘）：decode 注册 → bootstrap server 按 room 撮合端点 → prefill `send_kvcache` RDMA 直写 → decode poll 到 Success → PREBUILT forward。关键语义：**bootstrap 是纯控制面**（只撮合端点，KV 数据不经它）；地址协商 pull 式（decode 经 `send_metadata` 指定槽位）、数据面 push 式（prefill RDMA WRITE）；chunk 流水把暴露压到末 chunk + aux（6.3 命题 2：总字节 $\kappa L (T - T_{\mathrm{hit}})$ 与 chunk 数无关）。

Dynamo 视角的时序口径（disagg-serving.md L58）：SGLang 用 `bootstrap_info`（host, port, room_id），prefill 可作 background task、**decode 侧立即开始（预分配 + poll）而 KV 传输并行**。注意口径：「decode 立即开始」指控制面准入与预分配立即启动，生成仍须 KV 到齐才进 DECODE（PREBUILT 不跑 forward，6.3 §4）——与 vLLM「decode 等 prefill 完」的差异在控制面时序，不在数据面正确性。

开源完整度：引擎 + disaggregation 全在 release（sgl-project/sglang，[A]）；多实例路由由 sglang-router / Dynamo 承担。

## 4. Dynamo：引擎之上的编排层

Dynamo 替换引擎吗？不。README L36 逐字：「the orchestration layer above inference engines — it doesn't replace SGLang, TensorRT-LLM, or vLLM, it turns them into a coordinated multi-node inference system」（[NVIDIA, ai-dynamo/dynamo](../references.md)，教学参照 + 官方实现，c41b6dfc，2026-11-23 contents API 核实）。特性表 Disaggregated Serving / KV-Aware Routing / SLA-Based Planner 对 vLLM / SGLang / TRT-LLM 三引擎全 ✅。

### 4.1 disagg 编排：PrefillRouter

编排流程（docs/fern/.../system-architecture/disaggregated-serving.md）：

```
Client → Frontend → Router
  Router: select prefill worker（KV-aware：cache overlap + load）
  Router → Prefill worker: prefill request
  Prefill: compute KV → 返回 disaggregated_params（backend 传输元数据）
  Router: select decode worker + 注入 transfer metadata
  Decode ↔ Prefill: KV transfer（NIXL，VRAM→VRAM 直传，non-blocking）
  Decode: generate tokens → Frontend → Client（stream）
```

四步：worker 选择（KV-aware 或简单 LB）→ prefill 执行（算 KV 返 `disaggregated_params`）→ decode 路由（注入 prefill 结果）→ KV 传输（NIXL 选最优 transport：NVLink / InfiniBand / UCX）。**runtime-reconfigurable xPyD**：worker 运行时加入（register discovery + 发布 `RuntimeConfig` 含 KV capacity）/ 移除（drain 活跃请求 + deregister），router 经 discovery 自动发现——配比不是部署时定死（8.2 xPyD 原理的运行态实现）。

各 backend 的传输元数据握手名不同，时序也不同（disagg-serving.md L54-62，对照增量）：

| backend | 握手元数据 | 装什么 | prefill/decode 时序 |
| --- | --- | --- | --- |
| SGLang | `bootstrap_info` | host, port, room_id | prefill 可 background task，decode 侧立即开始 + KV 并行 |
| vLLM | `kv_transfer_params` | block IDs + remote worker 连接信息 | prefill 同步，decode 等 prefill 完 |
| TRT-LLM | `opaque_state` | 序列化 TRT-LLM 内部元数据 | prefill 同步，decode 等 prefill 完 |
| TRT-LLM（trtllm-serve） | `ctx_params` | ctx server 握手参数 | orchestrator 撮合 ctx↔gen（§5） |

握手名是控制面元数据（小字节），KV 数据面走 NIXL 直传、不经握手通道——与 8.3 §3 NIXL metadata「控制路径非数据路径」、6.3 bootstrap「纯控制面」同一条设计原则。时序差异（decode 立即开始 vs 等 prefill 完）是控制面准入时序，不改变「KV 到齐才进 DECODE」的数据面正确性（§3 口径注）。

### 4.2 KV-aware routing：成本模型

路由怎么选 worker？不是无条件亲和 cache-hot 机，而是算一笔成本账（routing-concepts.md L29-33）。tokenize/normalize → KV router（KV cache index 报每 worker prefix overlap + active-load tracker）→ filter → score → select 最低 cost 合格 worker。worker 发 KV create/release events 更新 index（`--no-router-kv-events` 则从路由决策预测）。代码 `lib/llm/src/kv_router.rs` + `kv_router/{indexer/mod.rs, shared_cache.rs, active_load.rs, prefill_router/mod.rs}`（Rust）。

成本模型（block 当量）：

$$
\mathrm{cost} = s_P \max(0, A_P + I_P - C_{\mathrm{ov}}) + (A_D + I_D) + w_R R \qquad (1)
$$

符号表： $A_P$ = active_prefill_blocks（worker 已指派的 prompt 侧 block）； $I_P$ = incoming_prompt_blocks（新请求 uncached prompt block）； $C_{\mathrm{ov}}$ = overlap_credit_blocks（device/host/disk/shared-cache 命中信用，block 当量）； $A_D$ = active_decode_blocks； $I_D$ = incoming_active_blocks（新请求投影 decode block）； $w_R$ = decode_active_request_weight（每活跃请求的 block 当量收费，默认 0）； $R$ = active_requests； $s_P$ = prefill_load_scale。三项语义：prefill 成本（命中信用折减、clamp 0 不为负）+ decode 成本（已指派 + 投影）+ 活跃请求成本（opt-in，decode 步延迟随 batch 数而非 KV 足迹时启用）。

数值实例（教学构造值，block 当量）：请求 $I_P$ = 100、 $I_D$ = 100。worker A（cache-hot 但重载）： $A_P$ = 200、 $C_{\mathrm{ov}}$ = 80、 $A_D$ = 500、 $R$ = 50、 $w_R$ = 0、 $s_P$ = 1 → cost = max(0, 200+100−80) + (500+100) = 220 + 600 = 820。worker B（cache-cold 但轻载）： $A_P$ = 0、 $C_{\mathrm{ov}}$ = 0、 $A_D$ = 100 → cost = 100 + 200 = 300。**B 胜出**（300 $\lt$ 820），尽管 A 有 80 block 命中——命中省的 80 block 被 A 的活跃负载（220+600）压倒。

**命题 1**（cache-hot 不保证选中）：存在合法负载配置，使 $C_{\mathrm{ov}} \gt 0$ 的 cache-hot worker 成本高于 $C_{\mathrm{ov}} = 0$ 的 cache-cold worker，路由选后者。**证明**（构造）：上例 A/B 即见证——A 的命中信用 80 只折减 prefill 项，decode 项 $A_D + I_D$ = 600 与活跃负载不受信用影响；B 虽无命中，总成本 300 $\lt$ 820。证毕。边界：cache-hot 且轻载（ $A_P$ = 0、 $A_D$ = 0、 $C_{\mathrm{ov}}$ = 80 → cost = 20 + 100 = 120）仍胜出——亲和不是被取消，而是被负载项压倒。overlap_credit 含 device/host/disk/shared-cache 四级信用，`overlap_score_credit_decay` 在 cache-rich worker 过载时折减 device-local 信用；conditional-disagg 例外式 cost = max(0, potential_decode_blocks − overlap_credit) + active_request_blocks，让 cache-hot decode worker 绕过 prefill。

**命题 2**（零信用退化）：当 $C_{\mathrm{ov}} = 0$ 且各 worker 活跃负载（ $A_P, A_D, R$ ）相等时，cost 排序退化为按 $s_P I_P + I_D$ 排序，与 cache 状态无关——KV-aware 路由退化为纯负载均衡。**证明**（要点）：代入 $C_{\mathrm{ov}} = 0$ ，prefill 项 = $s_P (A_P + I_P)$ ；活跃负载相等则 $s_P A_P + A_D + w_R R$ 为常数，cost 差只来自 $s_P I_P + I_D$ 。完整证明留作练习 2。含义：没有命中信用时，KV-aware 路由不比 shortest-prefill-first 的 block 版多知道任何事——index 的价值全在信用项。

prefill load model 默认静态至首 output token（`--router-prefill-load-model ais` 改用 AISimulate 预期时长 + elapsed 衰减）；decode 侧 `--router-track-output-blocks` + `nvext.agent_hints.osl` 分数权重。placement 决策非无条件亲和（命题 1）——这是 KV-aware 路由与朴素 cache-affinity 的本质区别。

### 4.3 planner：SLA 驱动伸缩（recap → 8.5）

planner（components/src/dynamo/planner/README.md + planner-design.md）是 SLA-driven autoscaling controller，双模：**throughput-based**（traffic prediction + engine perf model → 副本数下界保 TTFT/ITL SLA；bootstrap 来自 worker self-benchmark / AI Configurator / profiler / live FPM regression）+ **load-based**（ForwardPassMetrics FPM event plane reactive，optimization_target: sla → online regression 修正目标 TTFT/ITL；支持 vLLM/SGLang/TRT-LLM 含 attention-DP per-rank 信号）。throughput 设下界、load 实时调整；prefill/decode/aggregated 各有 PlannerEnginePerfModel；KV hit rate + spec decode accept length 预测器。伸缩触发账与 DeepInfer autoscaling 的对照归 8.5，本章只登记双模结构。

## 5. TensorRT-LLM：Python transceiver V2 + NIXL（当前形态）

TRT-LLM 的 disagg 是 C++ executor + MPI/UCX 吗？**不是**——这是本章最大的漂移坑。当前 main 默认 = **Python transceiver（`KvCacheTransceiverV2`）+ backend: NIXL + KV cache manager V2**（[NVIDIA, TensorRT-LLM](../references.md)，教学参照 + 官方实现，docs/source/features/disagg-serving.md，2026-11-23 contents API 核实）。

> **坑**：按 legacy 文档写 TRT-LLM disagg 形态。触发条件：读 legacy/advanced/disaggregated-service.md 或 blog05 的「MPI/UCX/NIXL 多 backend + C++ executor/transceiver」；现象：把已弃用路线当当前默认。事实（disagg-serving.md L169 逐字）：「The C++ cache transceiver (`transceiver_runtime: CPP`, or `null`) is deprecated as of TensorRT-LLM 1.4 (2026-11) and will be removed after the 3-month migration period. Use the Python transceiver, the default, with `backend: NIXL` and KV cache manager V2 on both the context and the generation worker.」规避：当前形态写 Python V2 + NIXL + KV manager V2；C++ 路线历史化（cpp cache_transmission/ 下的 mpi / ucx / nixl / mooncake utils 仍在树，但非默认）。

配置面（`cache_transceiver_config`）：`backend: NIXL`（RDMA/NVLink）**无默认值**——unset 则 worker 仍启动但不起 transceiver、拒绝被路由来的 disagg 请求（L162）；`max_tokens_in_buffer`（建议 unset，默认由模型 max seq len 派生）；`kv_transfer_timeout_ms`（默认 60000）；`kv_cache_bounce_size_mb`（默认 0 = off）。Python transceiver 要求 `use_kv_cache_manager_v2: true`，配 V1 初始化报错。

bounce buffer（L176-186）：ctx 与 gen 的 attention 切分不同（ctx TP ≠ gen TP / attention DP）时，一次 KV write 由大量小 fragment 组成、每 fragment 一个 NIXL descriptor。bounce buffer 让 sender 把小 fragment gather 进连续缓冲、按 chunk 发送，receiver scatter 回 KV cache。默认启用条件：**≥ 1024 descriptor 且平均 ≤ 16 KiB**（head-matched 布局如 MLA / 对称 TP 走标准 NIXL）。数值实例（教学构造）：一次 write 2048 descriptor × 8 KiB = 16 MiB → 满足条件（2048 ≥ 1024、8 ≤ 16），gather 后按默认 max_chunk_size 32 MiB 一个 chunk 发出，descriptor 2048 → 1。这是 8.3 §5「小块聚合成大拷贝」的 TRT-LLM 实现。

编排三路线：(1) **Dynamo**（smart router + pre/post-processing workers + K8s，§4）；(2) **trtllm-serve disaggregated**（ctx + gen 各 OpenAI-compatible server + disaggregated orchestrator server，`ctx_params` 握手；扩 orchestrator 用 **Coordinator + Worker Fleet**：fleet workers 经 `SO_REUSEPORT` 共享端口、无路由状态，coordinator 拥 cluster state 与 `/select` `/finish` `/renew` API、KV-aware routing 经 ZMQ event-ingest；output-length prediction ridge/ridge_gbm → generation priority shorter-first）；(3) **Triton BLS**（blog05，Executor API 仅 TensorRT backend）。

KV 传输机制：overlap（KV 传输与其他请求计算重叠 + 多 GPU 集并行传输）；cache layout transformation（device-to-device 直传，ctx/gen 不同并行策略需 block mapping 编排，TP2 ctx → PP2 gen 例）；unique global request ID（snowflake 64-bit，`DisaggregatedParams.disagg_request_id`，ctx↔gen 传输 keyed by it，全局 ID 范围 [1<<40, 2^63) 与 worker-local [0, 1<<40) 不相交）；`schedule_style` context_first / generation_first。

性能（blog05，官方披露）：DeepSeek R1 MTP off disagg 1.4–1.8×（低并发尤甚）/ MTP on 1.6–2.5×（平均高 20–30%）；GEN4 1.73× / GEN8 2×；50 tok/s/user 1.7×。rate-matching 方法（blog05 L186-189，recap 8.2 xPyD 原理）：测 ctx 各 TP/EP/DP/PP 满足 TTFT 的 request throughput 选最优 → 测 gen throughput/latency → 找 ctx:gen worker 比使两相聚合吞吐匹配（按 ISL/OSL）→ per-GPU throughput 公式 → rate-matched Pareto 曲线。E2E 与 SOL（rate-matched）差 0–25%（SOL 假设分数 ctx:gen 比 + 无 KV 传输开销）。IFB（in-flight batching）文档出处 = docs/source/features/paged-attention-ifb-scheduler.md（continuous/iteration-level batching、packed no-padding、context 序列排 generation 前）——9.2 只 recap 本章 + 一句选型定位，不展开。disagg-serving.md Motivation 引 Beyond the Buzz（arXiv:2506.05508，[B]）。

## 6. DeepInfer：存储为中心的 VE/P/D

DeepInfer 的 KV 走点对点直传吗？不——走专用 3FS 存储集群中转（recap 7.1 §2 / 7.4 §1 / 2.4 §8-9，走读不重复）。这是六系统里唯一的存储为中心路线。

架构 recap（doc 官方披露）：**E/P/D 分离 + 专用存储节点**——VE（vision encoder）/ PE（prefill engine）/ CE（completion engine）三角色 + 3FS kvcache-store。VE 单 GPU 一般够、推荐更便宜的 GPU（doc L51）；attention #DP = #EP（L33）；~1:1 P/D 经验配比 + 64P+64D 最小高效单元（L91，8.2 recap）。Entrypoint CHECK 角色对 local kvcache-store 做 **hash-chain 前缀匹配**（metadata only，hit length 路由前必算、不可隐藏，doc L191）；取 KV 异步 3FS → host shared memory；swap-in 引擎线程逐层 H2D 与 forward 重叠、用 GDR（NIC 直写 HBM 不占 SM，8.3 §5 recap）。

release 边界（recap 7.1 命题 1，README L38-42）：三项不在 release——KV cache 系统（Entrypoint directory/fetch 编排 + kvcache-store）/ VE-P-D disagg 编排（「most required engine functionality is already implemented」）/ prefill+decode DP LB。**引擎侧在 release**：PE/CE engine type（engine_type.py PROMPT/COMPLETION）+ StateBlock + EPLB；**编排侧不在**：Entrypoint CHECK / 三 DP LB / 3FS kvcache-store 编排 / 平台弹性。所以「DeepInfer 开源了 PD 分离」须收窄为「引擎侧功能已实现、编排层未开源」——自建编排成本是选型代价（§8）。

传输路径对回 8.3 账：存储中转 = 3FS 读（网络 + SSD）→ host shm → GDR H2D 三段流水（2.4 §4），相对点对点直传多一跳网络与独立存储硬件；代价换收益——任何 prefill 机命中任何前缀（跨集群前缀共享）、存储与计算 autoscaling 解耦、故障域隔离（6.3 §10 论断 recap）。带宽账形态是 2.4 式 (3) 的 offload 取回（无计算可重叠），不是 8.3 式 (6)(7) 的 P2D egress/ingress——这是存储路线与直传路线在账上的本质差异。

## 7. MemServe：elastic memory pool（论文机制，无 repo）

MemServe 有官方代码吗？没有（[Hu et al., 2024](../references.md)，arXiv:2406.17565，[B]，W5 search 12 结果均无关、论文唯一 github 链接为 TRT-LLM related work）。本节讲论文机制，不推测代码。

核心抽象 **MemPool**（§4）：管理集群全内存（CPU DRAM + GPU HBM）的 elastic memory pool，每 instance 内运行、fixed-size allocator；三类 API——memory（分配）/ indexing（prompt tokens → KV 映射，active + historical）/ distributed transfer（抽象三异构：parallelism / network / memory medium）。Table 1 `alloc_mem`：type（HBM-only/DRAM-only/mixed）、@instance id、返 addrList（address 编码 instance ID）、transfer flag 控 on-demand allocation。

transfer 三步（§4）：**allocation → transmission → insertion**。receiver 调 `alloc_mem` 分配 HBM/DRAM、返地址列表 + parallelism config（allocation）；sender 经最快可用路径传 KV、收完通知 sender（transmission）；receiver 若 `transfer_with_insert` 则把新传 prompt tokens + historical KV 插入本地 index（insertion）。**指定 dst addr 可跳 allocation → layer-by-layer 传输**（§4，与 8.3 §1 layer-wise 语义同构）。实现用 NCCL send/recv 对（GPU-GPU）、socket（任一侧 DRAM）（§7）。

context caching + disagg 首次结合（§5）：四步——distributed API 复现 disagg → prefill-only instance 加 caching → decode-only 加 caching → **decode-to-prefill 数据传输**（多轮对话历史 KV 回流）；huge pages 共优内存布局 + 网络传输；context caching cost model。global scheduler（§6）：**global prompt tree**（distributed radix tree + instance 持有字段）locality-aware best-effort routing 最大化 KV 复用——与 Dynamo KV-aware routing（§4.2）同目标、不同实现（论文树 vs 生产 index + 成本模型）。failure handling/scaling（§4.4）：cluster manager heartbeat 检测故障、广播 cluster info、instance 释放故障 instance 分配的 memory block 防泄漏。

实现与评估（§7-8，论文口径）：MemPool + global scheduler from scratch 5.6K SLOC Python + 1.6K C++；改 vLLM 200 SLOC Python + 400 CUDA C++。ShareGPT 上 PD-disagg + caching 最优：disagg vs colocated JCT 改善至多 42%，+caching 再 29%；LooGLE（长 prompt 短生成）disagg 10.8% + caching 26.9%。这些是论文模拟/原型数字，无官方 repo 可复现——选型时只作机制参照，不作生产部署选项（§8）。

## 8. 六轴对照总表与选型论断

六条轴里哪条真正决定选型？不是功能清单，而是 KV 传输路径（延迟跳数）与开源完整度（自建成本）——前者定性能账形态，后者定落地代价。下表按六轴对照（各格只 recap 要点 + 指针，增量在表后的选型论断）：

| 系统 | KV 传输路径 | 调度编排 | 弹性 | 开源完整度 | 硬件支持 | 适用规模 |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM | NIXL push/pull + Mooncake + MultiConnector 组合（§2，5.3/5.4） | connector 角色分件 + kv_transfer_config；外部编排 | 外部编排器承担；elastic_ep 引擎侧 | 引擎 + connector 全 [A]；编排外 | NIXL UCX/GDS 多 transport | 集群内 P/D |
| SGLang | mooncake/nixl backend，引擎内置双向协议（§3，6.3） | bootstrap room 撮合 + 四抽象；sglang-router/Dynamo | elastic_ep/ ScaleCohort（8.5） | 引擎 + disagg 全 [A]；路由外 | mooncake RDMA/NVLink + nixl | 集群内 P/D |
| Dynamo | NIXL VRAM→VRAM 直传（§4.1） | PrefillRouter + KV-aware 成本模型（式 (1)） | runtime xPyD + SLA planner 双模（8.5） | 编排层全（教学参照）；引擎用上游 | NIXL 多 transport（NVLink/IB/UCX） | 多引擎 fleet、集群内 |
| TRT-LLM | Python transceiver V2 + NIXL（§5）；bounce buffer 聚合 | trtllm-serve orchestrator + coordinator fleet / Dynamo / Triton BLS | 动态节点加入退出在建 | 引擎 + disagg 全（教学参照） | NVIDIA（RDMA/NVLink/MNNVL） | 集群内、NVIDIA 栈 |
| DeepInfer | 3FS kvcache-store 中转 + GDR swap-in（§6，2.4/7.4） | Entrypoint CHECK + 三 DP LB（release 外） | autoscaling + grace + EP failover（8.5，doc） | 引擎侧在、编排外（教学参照） | SM100 + Ascend 950 双平台 | 跨集群前缀共享、跨 DC |
| MemServe | MemPool transfer API（NCCL/socket），layer-by-layer（§7） | global prompt tree locality-aware | MemPool scaling + CM heartbeat | 无官方 repo [B] | 论文原型（GPU + DRAM） | 学术机制参照 |

各系统 KV 传输路径 ↔ 8.3 账对照（spec 关键推导）：vLLM / SGLang / Dynamo / TRT-LLM 走**点对点直传**，适用 8.3 式 (6)(7) egress/ingress 与命题 1 暴露时间账；DeepInfer 走**存储中转**，适用 2.4 式 (3) offload 取回账（无计算重叠）+ 三段流水；MemServe 走 **MemPool 统一传输**，layer-by-layer 与直传同构但经 pool 分配。编排层职责划分三型：**引擎内置**（SGLang，bootstrap 是引擎组件）/ **connector 插件**（vLLM，角色分件 + 外部编排）/ **外部编排器**（Dynamo PrefillRouter / TRT-LLM orchestrator / DeepInfer Entrypoint，引擎只管算与传）。

选型论断（可判定，每行回指）：

| 场景 | 选 | 论断依据 | 回指 |
| --- | --- | --- | --- |
| 集群内 P/D 低延迟交接 + 通用生态 | vLLM connector 或 SGLang 内置 | 直传适用 egress/ingress 账；SGLang 若已用 radix/结构化程序栈则内置协议省一层 | §2/§3、8.3 §6 |
| 多引擎混合 fleet + KV 复用 + SLA 弹性 | Dynamo 编排层 | 不替换引擎；成本模型非无条件亲和（命题 1）；runtime xPyD + planner | §4、命题 1/2 |
| NVIDIA 编译路线 + 生产 disagg 性能 | TRT-LLM（Python V2 + NIXL） | 1.4–2.5× 官方披露；勿按 legacy C++ 文档（漂移坑） | §5 |
| DeepSeek 专用 + 跨集群前缀共享 + 存储计算解耦 | DeepInfer 存储为中心 | 任何机命中任何前缀；代价：编排层不在 release、多一跳 + 独立存储硬件 | §6、7.1 命题 1 |
| 跨 DC P/D | 压缩 KV + 400 Gbps/8 GPU 预算 | V4.1 FP4 60.4 ms 达标余量 3.31×；可靠性代价（光纤易断、相邻 DC） | 8.3 §6.3、§6 |
| 弹性内存池 / context caching 机制研究 | MemServe 论文 | 无 repo 不可直接部署；JCT 42%/29% 是论文原型数字 | §7 |

一条边界论断：**开源完整度轴上，DeepInfer 与 MemServe 是两种不同的「不全」**——DeepInfer 引擎侧在 release、编排侧仅 doc 披露（可自建，成本已知）；MemServe 无官方 repo（机制可参照，实现须从头）。选型时前者是「自建编排」决策，后者是「是否值得复现论文」决策，不可混为一谈（7.1 命题 1 / §7）。

## 总结

| 论断 | 账 / 证据 | 回指 |
| --- | --- | --- |
| vLLM PD = connector 抽象（角色分件 + push/pull 双模 + MultiConnector 组合），编排在引擎外 | KVConnectorBase_V1 L182 / NixlPull L374 / NixlPush L415 / MultiConnector L142 | §2、5.3/5.4 |
| SGLang PD = 引擎内置双向协议（bootstrap 纯控制面 + 地址协商 pull 式 / 数据面 push 式） | base/conn.py 四抽象 + KVPoll；decode 立即开始指控制面准入 | §3、6.3 |
| Dynamo = 引擎之上编排层，不替换引擎；KV-aware 路由按成本模型选最低 cost worker | README L36；式 (1)；PrefillRouter + NIXL VRAM→VRAM | §4 |
| cache-hot 不保证选中：命中信用只折 prefill 项，活跃负载可压倒亲和 | 命题 1（A 820 vs B 300）；零信用退化为纯 LB（命题 2） | §4.2、练习 1/2 |
| TRT-LLM 当前 = Python transceiver V2 + NIXL + KV manager V2；C++ 1.4（2026-11）弃用 | disagg-serving.md L169 逐字；backend unset 拒 disagg；bounce ≥1024 desc ∧ 平均 ≤16 KiB | §5、练习 3 |
| DeepInfer = 存储为中心 VE/P/D，引擎侧在 release、编排侧仅 doc | README L38-42；Entrypoint CHECK hash-chain；3FS→shm→GDR 三段 | §6、7.1 命题 1 |
| MemServe = MemPool 三 API + transfer 三步 + global prompt tree；无 repo [B] | 5.6K+1.6K SLOC；JCT 42%/29%；指定 dst addr 跳 allocation → layer-by-layer | §7 |
| 传输路径三型：直传（egress/ingress 账）/ 存储中转（offload 账）/ pool 统一传输 | 8.3 式 (6)(7) vs 2.4 式 (3)；编排三型：内置/插件/外部 | §8、8.3 |
| 9.2 只 recap 本章 + 一句选型定位，不双展开 | C-S3-4 防双展开 | §1、§5 |

## 练习

1.（计算 · [10]）Dynamo 成本模型代入（式 (1)，block 当量）：worker A： $A_P$ = 200、 $I_P$ = 100、 $C_{\mathrm{ov}}$ = 80、 $A_D$ = 500、 $I_D$ = 100、 $w_R$ = 0、 $s_P$ = 1；worker B： $A_P$ = 0、 $I_P$ = 100、 $C_{\mathrm{ov}}$ = 0、 $A_D$ = 100、 $I_D$ = 100。(a) 各求 cost；(b) 路由选谁；(c) 若 A 的 $A_D$ 降到 0，cost 变多少、是否反超 B。校验：(a) 820 / 300；(b) B；(c) 220 + 100 = 320 $\gt$ 300，仍 B 胜（与 §4.2 命题 1 构造一致）。
2.（推导 · [20]）命题 2 证明闭环（零信用退化）：(a) 补全证明——代入 $C_{\mathrm{ov}} = 0$ 与活跃负载相等，说明 cost 差只来自 $s_P I_P + I_D$ ；(b) 给边界反例： $C_{\mathrm{ov}} \gt 0$ 时为什么排序不再与 cache 无关；(c) 说明 conditional-disagg 例外式与主式 (1) 的口径差。校验：(a) 与 §4.2 证明要点一致；(b) 信用项使 cache-hot 可胜出（命题 1 边界）；(c) 例外式用 potential_decode_blocks 减信用、绕过 prefill 项。
3.（代码 · [30]）TRT-LLM disagg 配置走读（docs/source/features/disagg-serving.md，2026-11-23 contents API）：(a) 当前默认 transceiver / backend / KV manager 各是什么；(b) backend unset 时 worker 行为；(c) bounce buffer 默认启用条件，判断 2048 descriptor × 8 KiB 是否启用；(d) C++ transceiver 弃用版本与迁移期。校验：(a) Python KvCacheTransceiverV2 / NIXL / KV manager V2；(b) 启动但不起 transceiver、拒 disagg 请求（L162）；(c) ≥1024 ∧ 平均 ≤16 KiB → 启用（L182）；(d) 1.4（2026-11）、3 个月（L169）；先预测 (c) 再对条件。
4.（论证 · [30]）六轴选型（§8 表）：给三个场景各选系统并给论断依据——(a) 多引擎混合 fleet、要 KV 复用 + SLA 弹性；(b) DeepSeek 专用、跨集群前缀共享、团队能自建编排；(c) NVIDIA 栈、要生产 disagg 性能、文档读到 C++ MPI/UCX 路线。校验：(a) Dynamo（命题 1 + runtime xPyD + planner）；(b) DeepInfer（存储为中心 + release 边界代价）；(c) TRT-LLM 但须按当前 Python V2 + NIXL 形态、C++ 路线已弃用（漂移坑）；每行回指 §8 表与对应节。

## 参考

- [Kwon et al., 2023](../references.md)：vLLM，arXiv:2309.06180，[A]（key: vllm-repo）——kv_transfer/kv_connector/v1/ KVConnectorBase_V1 L182 / KVConnectorRole L140 / NixlBaseConnector L79 / NixlPullConnector L374 / NixlPushConnector L415 / MooncakeConnector L798 / MultiConnector L142 / mooncake_store / lmcache_connector / hf3fs / hisparse / offloading_connector；W3 同日 tarball 2026-11-23 再确认
- [Zheng et al., 2023](../references.md)：SGLang，arXiv:2312.07104，[A]（key: sglang-repo）——disaggregation/base/conn.py KVArgs L62 / KVPoll L142 / BaseKVManager L150 / BaseKVSender L191 / BaseKVReceiver L266 / BaseKVBootstrapServer L339；common/conn.py；mooncake/conn.py MooncakeKVManager L364 / nixl/conn.py NixlKVManager L531；prefill.py PrefillBootstrapQueue L270 / decode.py DecodePreallocQueue L471 / DecodeTransferQueue L3303；W3 同日 tarball 再确认
- [NVIDIA, ai-dynamo/dynamo](../references.md)：key: dynamo，教学参照 + 官方实现（c41b6dfc，2026-11-23 contents API + trees 深核）——README L36 编排层定位；docs/fern/.../system-architecture/disaggregated-serving.md PrefillRouter + backend 元数据（bootstrap_info / kv_transfer_params / opaque_state）+ runtime xPyD；routing-concepts.md L29-33 成本模型；kv-aware-routing.md；lib/llm/src/kv_router.rs + kv_router/{indexer/mod.rs, shared_cache.rs, active_load.rs, prefill_router/mod.rs}；components/src/dynamo/planner/README.md + planner-design.md 双模
- [NVIDIA, TensorRT-LLM](../references.md)：key: tensorrt-llm，教学参照 + 官方实现（2026-11-23 trees + contents API）——docs/source/features/disagg-serving.md 当前 Python transceiver V2 + NIXL + KV manager V2（L162 backend unset / L169 C++ 1.4 弃用 / L176-186 bounce buffer）；Coordinator + Worker Fleet / snowflake ID / cache layout transformation / overlap；blogs/tech_blog/blog05 性能 1.4–2.5× + rate-matching；paged-attention-ifb-scheduler.md IFB；legacy/advanced/disaggregated-service.md 历史化
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——README L38-42 release 边界三项；engine_type.py PROMPT/COMPLETION + StateBlock + EPLB 在 release；Entrypoint / 三 DP LB / kvcache-store 编排不在；ve/multimodal_processor.py MultimodalProcessor L29；W4 同日 tarball 2026-11-23 再确认
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——L51 VE 单 GPU cheaper；L91 ~1:1 + 64P64D；L191 Entrypoint CHECK hash-chain + async fetch + GDR swap-in；L33 #DP=#EP；弹性容错 L251-265（8.5）
- [Hu et al., 2024](../references.md)：MemServe，arXiv:2406.17565，[B]（key: memserve，无官方 repo，W5 search + 论文全文再确认）——§4 MemPool 三 API + transfer 三步 + 指定 dst addr layer-by-layer；§5 context caching + disagg 四步 + huge pages；§6 global prompt tree；§4.4 CM heartbeat；§7 5.6K Python + 1.6K C++ / 改 vLLM 200+400 SLOC / NCCL send-recv + socket；§8 JCT 42%/29%、LooGLE 10.8%/26.9%
- [NVIDIA, ai-dynamo/nixl](../references.md)：key: nixl，教学参照 + 官方实现——NIXL VRAM→VRAM 直传 backend；agent/descriptor 语义详 8.3 §3
- [Qin et al., 2024](../references.md)：Mooncake，arXiv:2407.00079，[A]（key: mooncake）——transfer engine Segment/BatchTransfer/多 NIC 池化，SGLang/vLLM backend 底座；详 8.3 §4
- [Mitra et al., 2025](../references.md)：Beyond the Buzz，arXiv:2506.05508，[B]（key: beyond-buzz）——TRT-LLM disagg-serving.md Motivation 引用；rate matching + prefill-heavy 最有效（8.2 recap）
- [Zhong et al., 2024](../references.md)：DistServe，arXiv:2401.09670，[A]（key: distserve）——goodput 优化 + placement（8.2 原理层，本章只对照编排）
