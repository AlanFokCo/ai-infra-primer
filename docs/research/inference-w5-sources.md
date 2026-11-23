# W5 调研笔记：P8 分布式推理与 PD 分离 事实源

调研日期：2026-11-23（W5 research pass）。本文件是 W5 五章（8.1–8.5）写作的事实源之一；写作 agent 不做额外调研，数值与代码锚点一律以本文件与 `inference/references.md` 为准。引用 key 白名单 = references.md P8 全部条目（含本轮新增 `beyond-buzz`）+ 被引章节含 W5 章号的跨波锚点（distserve / splitwise / mooncake / llumnix / sarathi-serve / azure-llm-trace / vllm-repo / sglang-repo / deepinfer-doc / deepinfer-repo / nixl / dynamo / lmcache / tensorrt-llm）。**边界纪律（spec §2/§4.2）**：serving 侧并行 vs training P5（recap + 指引，无 backward 通信差异，不重推导训练账）；通信硬件 / RDMA 协议 / NVLink 拓扑 / NCCL 实现 → cluster 维度（本维度只讲传输语义、带宽账与系统设计）；PD 原理（8.2）vs 引擎实现（8.4 对照 / 6.3 W3 SGLang impl / 7.4 W4 DeepInfer VE-P-D）；8.4 / 9.2 对照章只 recap 各系统要点 + 指针，增量限定为对照表与选型论断（§4.2 两层走读纪律）。

**版本锚定总口径**：本轮代码锚点 = 2026-11-23 当日 codeload tarball（小 repo）+ GitHub contents API raw（大 repo 关键文件，tarball 超 ~7-10 MB 本机下载被截断，见访问路径）；vLLM / SGLang / DeepInfer 复用 W3/W4 同日 tarball（byte-identical，file:line 锚点维持）。repo HEAD：nixl **cb66ce836f82**（2026-11-22）/ dynamo **c41b6dfc5c5f**（2026-11-23）/ Mooncake **2cdbaeeb8e91**（2026-11-23）/ LoongServe **f6e8fc15f31e**（2024-11-11，论文时代代码稳定）/ llumnix-ray **3fb6c0376b3b**（2026-03-12，v0 末次更新）/ splitwise-sim **8f99e7dc9b40**（2024-04-25）/ DistServe **82831f1604cc**（2025-04-06）。

## 访问路径记录（后续 research pass 复用）

- **arXiv：https export API 本轮恢复可用**——`https://export.arxiv.org/api/query?id_list=<7 ID>&max_results=10` **批量一次 200**（18,636 B，7/7 entry；W3/W4 的 429 本轮未复现）。**http（非 https）export API 仍 000**——必须用 https。单 ID 补查 2506.05508 亦 200。论文全文用 `https://arxiv.org/html/<id>v<n>`（200）。
- **GitHub：api.github.com 200 可达**（core 限额本轮用量 ~20：rate_limit / repos×8 / commits/main×7 / search×2 / trees×3 / contents×13）；**codeload tarball 小 repo（<7 MB）200 完整**（nixl 6.8 MB / LoongServe 1.27 MB / llumnix-ray 0.79 MB / splitwise-sim 1.22 MB / DistServe 0.61 MB），**大 repo 下载被截断**（dynamo 7.7 MB / Mooncake 10.3 MB 固定字节数截断，curl --retry / -C - resume 6 次均卡同一尺寸 → 判定本机代理对大文件硬截断）→ **改用 git trees API（recursive）定位 + contents API raw 取关键文件**（dynamo-tree 9,890 entry / mooncake-tree 3,807 entry / trtllm-tree 25,568 entry 均 truncated=False）。raw.githubusercontent.com 仍 000（沿 W0-W4）。
- **TRT-LLM repo 2.25 GB**（size_KB 2,253,280）——不可 tarball，全程 trees + contents API。
- hf-mirror 200 / huggingface.co 000（本轮未涉及 config 重取，W4 第四次重取口径维持）。
- 证据存档：/tmp/iw5-evidence/（5 小 repo tarball+解压、dynamo/Mooncake 部分解压+API 关键文件、3 repo tree JSON、arXiv 批量 XML+metadata、LoongServe/MemServe/Llumnix/Beyond-the-Buzz HTML+txt、TRT-LLM 6 文档、Dynamo 4 文档、Mooncake 4 头文件）；脚本 /tmp/iw5-scripts/{fetch_repos.sh,refetch_big.sh}。复用 /tmp/iw1-evidence/{distserve.txt,splitwise.txt,arxiv-2401.09670v3.html,arxiv-2311.18677v2.html}（W1 同日）、/tmp/iw3-evidence/{vllm-x,sglang-x}（W3 同日 tarball，嵌套根 vllm-main/ sglang-main/）、/tmp/iw4-evidence/DeepInfer-x（W4 同日，嵌套根 DeepInfer-main/）。

## ⟨verify⟩ 销项总表（W5 范围）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 8.4 TRT-LLM disagg 实现形态（spec 附录留 W5） | **销**：当前 main = **Python transceiver（KvCacheTransceiverV2）+ backend: NIXL + KV cache manager V2** 为默认；**C++ cache transceiver 自 TensorRT-LLM 1.4（2026-11）弃用**、3 个月迁移期后移除（disagg-serving.md L169 逐字）。编排三路线：Dynamo / trtllm-serve disaggregated（ctx+gen+orchestrator 三服务器，ctx_params 元数据握手）/ Triton BLS（blog05，Executor API 仅 TensorRT backend）。KV 传输 backend 历史 MPI/UCX/NIXL（blog05 L144）→ 当前推荐 NIXL（RDMA/NVLink，动态节点加入/退出在建）；cpp cache_transmission/{mpi,ucx,nixl,mooncake}_utils 仍在树 | disagg-serving.md（contents API 60,489 B）+ blog05 + trtllm-tree.json |
| 9.2 inflight batching 文档出处（spec 附录留 W5） | **销**：docs/source/features/**paged-attention-ifb-scheduler.md**「In-flight batching (also known as continuous batching or iteration-level batching)」+ packed input（no padding）+ context 序列须排在 generation 序列前；disagg-serving.md Motivation 亦称 aggregated = IFB | contents API 12,232 B |
| 8.2 DistServe 官方 code | **维持 [A]**：LLMServe/DistServe tarball（82831f16）README = 论文摘要逐字 + SwiftTransformer C++ backend（LLMServe/SwiftTransformer）；论文脚注自链（W0 C-S2-1 / W1 摘要复核） | tarball README + arXiv v3 |
| 8.2 Splitwise 官方 code | **维持 [B]**：仅 mutinifni/splitwise-sim（8f99e7dc，README「SplitwiseSim … built to evaluate Splitwise」离散事件模拟器，cluster/scheduler/interconnect/performance_model/power_model）+ vLLM PR #2809；GitHub search splitwise 7,858 结果，LLM 相关仅 splitwise-sim（170 stars），余为记账 app/SDK | tarball + search API + 论文 ref |
| 8.1 LoongServe 官方 code | **维持 [A]**：LoongServe/LoongServe tarball（f6e8fc15）README 自认论文实现 + artifact-eval；longserve_c_scheduler / longserve_cuda_kernels / loongserve/ / rnccl | tarball README + arXiv v2 自链 |
| 8.4 MemServe 官方 code | **维持 [B]**：GitHub search memserve in:name 12 结果均无关（Xiaoxiao-Liu/MemServe 无描述且作者名不在论文 11 作者 list；HeatherHong/MemServe = GPU-DRAM-SSD 分层原型异项目；yangchenxi-nwpu/MEMSERVE 无描述）；论文全文唯一 github 链接 = NVIDIA/TensorRT-LLM（related work L691） | search API + 论文全文 grep |
| 8.5 Llumnix 官方 code | **维持 [A]**：llumnix-project/llumnix-ray tarball（3fb6c037）README 自链 arxiv.org/abs/2406.03243 + OSDI'24 citation（L28/L114）；v0 Ray 架构（论文实现），v1 另立 llumnix-project/llumnix（W2 注维持） | tarball README + arXiv v1 |
| 8.4 vLLM disagg connector 路径 ⟨verify 路径⟩ | **销（W3 已核本轮再确认）**：vllm/distributed/kv_transfer/kv_connector/v1/{base.py,nixl/,mooncake/,mooncake_store/,lmcache_connector.py,hf3fs/,multi_connector.py,hisparse/,offloading_connector.py,...}；nixl/connector.py NixlBaseConnector L79 / NixlPullConnector L374 / NixlPushConnector L415 | W3 同日 tarball 再 ls + grep |
| 8.3 NIXL agent API / descriptor 语义 | **销（C++ 级深核）**：见 8.3 节 nixl.h / nixl_descriptors.h / nixl_types.h file:line 锚 | nixl tarball cb66ce83 |
| 8.4 Dynamo planner / router / KV-aware / disagg | **销**：见 8.4 节 planner README + routing-concepts 成本模型 + disagg-serving PrefillRouter 编排 | dynamo contents API + tree |
| 8.3 Mooncake TE multi-NIC pooling / topology-aware | **销（源码级）**：见 8.3 节 transfer_engine.h / topology.h / transport.h + index.md | Mooncake contents API + tree |

## 语义纠正 / 命名注（research pass 新发现，writer 须知）

1. **TRT-LLM disagg 当前形态 ≠ 旧文档**：legacy/advanced/disaggregated-service.md 与 blog05 描述 MPI/UCX/NIXL 多 backend + C++ executor/transceiver；**当前 main 默认 Python transceiver V2 + NIXL + KV cache manager V2，C++ transceiver 1.4（2026-11）弃用**。写作必须给当前形态 + 漂移注（旧 C++ 路线历史化），勿按 legacy 文档写「TRT-LLM disagg = C++ executor + MPI/UCX」。backend: NIXL **无默认值**—— unset 则 worker 不起 transceiver、拒绝 disagg 请求（disagg-serving.md L162）。
2. **TRT-LLM 术语**：context phase = prefill、generation phase = decode；aggregated serving = IFB（in-flight batching）；NVIDIA 用 **FTL（First Token Latency）/ TTL（Token-to-Token Latency）**（Beyond-the-Buzz）= TTFT / TPOT 同义异名，引用时对照注明。ctx_params / disaggregated_params / opaque_state（TRTLLM）/ bootstrap_info（SGLang）/ kv_transfer_params（vLLM）= 各 backend 的 KV 传输元数据握手名（Dynamo disagg-serving.md L54-62）。
3. **NIXL ≠ 仅 UCX/GDR**：NIXL 是**插件化传输抽象库**（memory + storage 统一），当前核心 backend = UCX + GDS（GPUDirect Storage），插件 18 个（ucx / cuda_gds / **mooncake** / hf3fs / obj / posix / libfabric / gpunetio / ...；closeout 更正 19→18，evaluator tarball cb66ce83 裁定，S3-8）——**NIXL 可经 mooncake 插件用 Mooncake TE 作 backend**（8.3/8.4 对照素材）。数据面 RoCE/InfiniBand/GPUDirect RDMA/NVMe-oF/TCP/NVLink。GDR（GPUDirect RDMA）是 UCX backend 下的 GPU 直写路径，非独立 backend。
4. **NIXL descriptor 三型**：nixlBasicDesc（单连续区 addr+len+devId）/ nixlStrideDesc（跨步）/ nixlBlobDesc（带元数据 blob）；**nixl_xfer_dlist_t = 传输描述符列表 / nixl_reg_dlist_t = 注册描述符列表**（两者不同勿混）；操作 NIXL_READ / NIXL_WRITE（initiator 视角，对应 vLLM pull=READ / push=WRITE）。
5. **Mooncake TE Segment vs Buffer**：每进程**一个** RAM Segment（名 = local_hostname，逻辑覆盖全地址空间），内部划多个 Buffer（连续、同设备、各自 rkey 权限 + NIC 亲和）；远程 openSegment(name) 引用；NVMeof Segment 另型（文件→DRAM/VRAM 零拷贝不经 CPU）。BatchTransfer = READ/WRITE 请求数组（异步 getTransferStatus），「async flexible AllScatter/AllGather」。
6. **LoongServe ESP ≠ 训练 CP**：ESP = serving 侧**弹性序列并行**（每 iteration 动态 DoP），elastic instance = 持完整模型副本的统一模型并行单元，ESP 组跨 instance；**改 TP 度需重启 runtime（分钟级）**（论文 §2.4 L129）→ 弹性靠 SP 维而非 TP 维。scale-down = proactive migration（prefill SP ring 流通时选择性留 KV，零额外通信）；scale-up = multi-master distributed decoding（master 持本请求 KV + 发 Q 给其余 instance 并行 local attention，FFN 跨 master 并行 + Q 交换与 local attention 重叠）。**与 DeepInfer balanced CP（README L12，release 支持）/ 生产不用 CP（doc L67）是不同口径**——8.1 三者分清（LoongServe 弹性 SP 学术方案 / DeepInfer release balanced CP / DeepInfer 生产 CP 取舍）。
7. **DeepInfer CP 主因（C-S2-3 维持）**：doc L67 逐字「The main reason we do not currently use it is that V4.1 is fast enough to keep tail TTFT acceptable even on a complete cache miss」；e2e 至多慢 5% 是 doc 称「reasonable tradeoff」的**代价面非决策首因**；worsens EP balance + kernel/comm/CPU 开销。教学结论落「CP 取舍是模型/负载依赖」非「CP 慢 5% 所以不用」。
8. **DistServe goodput 定义要件（S3-9 维持）**：per-GPU 归一 + SLO attainment 目标（90%/99%）下最大请求率，**不得写成「达标请求的裸吞吐」**；优化三变量 = 每相 GPU 数 + 并行度 + placement，profiling+simulator 搜索再 replication。
9. **传输 vs 重算临界 T 相消（C-S3-2 维持）**：前向线性项下传输 κLT 与重算 2NT 同 ∝ T → T 相消，临界是**带宽阈值 BW* = κL·F_eff/(2N)** 非「临界上下文长度」；长度依赖由 attention 二次项 / TTFT 预算 / 重算对 decode 干扰引入；layer-wise 重叠（Splitwise §IV-C / Beyond-the-Buzz §5.1）进一步压低传输边际成本。
10. **Beyond-the-Buzz KV 带宽公式用 MHA/GQA 几何**（d_head × N_kv_heads）；MLA 须以 latent 维（κ）替换，论文显式以 DeepSeek-R1 MLA 为例（大模型 + MLA egress 带宽可能更低）。egress/ingress 归一只计**实际分片 KV 的 GPU**（DP attention 下非全部 GPU）。
11. **Llumnix 迁移用 Gloo 非 NCCL**：论文 §5 逐字——NCCL 并发调用 unsafe（迁移须与推理并行），Gloo Send/Recv + pipelined multi-stage 达 near-zero downtime；代码 RayColMigrationBackend（gloo，pygloo bf16→glooFloat16）。downtime ≈ 末次 iteration KV 拷贝（常数 20-30 ms，与序列长无关），naive recompute/copy 达 50-111× decode cost。
12. **Splitwise 三池**：prompt pool / token pool / **mixed pool**（动态进出减碎片 + 高负载保 SLO，mixed 机保留原身份、队列清空回原池；粗粒度 re-purposing 不频繁）；CLS 同时指派 prompt+token 机以重叠 KV 传输与 prompt 计算；小 prompt 用 serialized 传输、大 prompt 用 layer-wise（token 数启动已知）。

## 版本漂移总表（W5 vs W0-W4 / 论文时代）

| # | 漂移 | 写作口径 |
| --- | --- | --- |
| 1 | TRT-LLM C++ transceiver 1.4（2026-11）弃用 → Python V2 + NIXL + KV manager V2 默认（语义纠正 1） | 8.4 当前形态 + 漂移注；legacy 文档历史化 |
| 2 | NIXL 插件扩至 19（含 mooncake/hf3fs/gpunetio），核心 UCX+GDS；W3 仅核 Python API，本轮补 C++ agent/descriptor | 8.3 C++ 锚 + 插件抽象；mooncake 插件对照 |
| 3 | Dynamo 文档迁 docs/fern/（旧 docs/architecture 路径不存在）；planner 双模（throughput/load-based）+ router 成本模型 + runtime xPyD | 8.4/8.5 按 fern 文档锚 |
| 4 | Mooncake TE 新增 congestion API（getTaskCongestionState）/ Ascend Direct Transport / disconnect_all_peers（HEAD commit）；index.md 678 行（W1 57 KB 口径演进） | 8.3 核心 Segment/BatchTransfer/topology 不变 + 一句「目录已扩展」 |
| 5 | llumnix-ray v0 末次更新 2026-03（v1 另立）；Llumnix 论文 v1（2024-06）机制锚 v0 | 8.5 recap 4.2 + v0 代码锚 |
| 6 | LoongServe 论文 v2（2024-10）/ repo 2024-11 稳定；ESP 机制无漂移 | 8.1 直接复用 |
| 7 | vLLM/SGLang/DeepInfer 复用 W3/W4 同日 tarball，file:line 全维持 | 8.3/8.4 connector + 8.1/8.5 doc 锚照旧 |

## 8.1 Serving 侧并行策略与账【重】

**training P5 recap 锚（纯文字指引，不重推导）**：
- **TP 通信账（training 5.1 §6）**：每层 **4 次 all-reduce**（前向 g×2 + 反向 f×2），消息 [b,s,h]、bf16 合 2bsh bytes/次；前向每层 4bsh、反向对称 4bsh、单层 8bsh、全模型 8bshL；每卡乘 ring 系数 2(t-1)/t（4.1 式 (2)）。式 (9) M_msg=2bsh、式 (10) 每层账、式 (11) 通信-计算比随 t 线性涨（NVLink 边界）。**serving 增量 = 无 backward → 每层只 2 次前向 all-reduce × L**（C-S3-1 防 2× 低估）；decode s=1 → 消息 2·B·h bytes（B=batch token 数，training b=micro-batch 序列数须翻译 + 口径注 §4.3）。
- **GQA/MLA TP 复制（training 5.1 §8）**：a_kv < t 时 KV 侧无 head 可切 → 复制（每 rank 持完整 KV 投影 + 激活，参数/激活账 KV 侧不除 t）；MLA latent 投影（h→d_c）全 head 共享无处可切、复制 latent 激活随 s 线性涨 → attention 块 TP 可除 t 份额比 MHA/GQA 更低。**DP attention for MLA 即消此复制**（每 DP rank 独立处理请求、KV 不复制）。
- **EP all-to-all（training 5.4 §1-2）**：dispatch/combine 两次 all-to-all（无归约语义，combine 加权求和在接收后本地）；每 token 每层 2kd 元素（bf16 4kd bytes）上界（models 4.5 §7）；木桶命题 T_layer ∝ max_j R_j（热点卡尾延迟乘性）。**serving 增量 = dispatch/combine 延迟敏感（decode 小消息）+ Wide-EP 账 recap 7.4**。
- **组合规则（training 5.5）**：world size 固定 t/c/p/e/d 五度；TP 压关键路径（通信强度高）进节点、PP/DP/EP 可跨节点；GPU 数乘积约束 + group 树。**serving 对照 = 模型规模 × 上下文 × SLO × 拓扑 → t/p/e/d 决策表**（5.3/6.3 分布式实现回链）。

**serving 侧增量事实**：
- **TP decode 延迟账（强制重推导 §6.3）**：每步通信 ≈ 2L × (延迟项 α + 带宽项 2(t-1)/t·m/BW_link)；decode 小消息延迟项主导。route-C M-3 数值锚（V4.1 L=40、d_model=5120、t=8、B=60、bf16、NVLink 900 GB/s）：m=614,400 B、带宽项 ≈1.19 μs ≪ 延迟项 ~10-20 μs → 每步 ≈2L×15 μs=1.2 ms，占 TPOT 预算可观份额 → 解释 DeepInfer 不用 TP。**NVLink 域内边界**（跨节点延迟项飙升）。
- **DeepInfer 不用 TP（doc L69 逐字）**：「Tensor parallelism (TP) has high communication overhead. We do not use it when the model fits in memory and the latency SLA can be met without it.」
- **DP attention for MLA（doc L33）**：「attention uses data parallelism (#DP = #EP)」；SGLang v0.4 blog（W3 核）：单 KV head TP8 重复 KV → attention DP，MoE 前 all-gather / 后 redistribute，8×H100 DeepSeek-Coder-V2-FP8 1.9× decode 吞吐；SGLang 配置面 attn_dp_size>1 派生 enable_dp_attention（W3 漂移注）。**DP attention 通信账（强制重推导）**：attention 侧 DP 免 all-reduce（KV 不复制），MoE 侧 EP all-to-all + DP↔EP 边界 all-gather/reduce-scatter；#DP=#EP 使边界对齐。
- **Wide-EP（recap 7.4 + 指引）**：doc L47-49 prefill EP2-8/DP2-8 最小 HBM 允许 / decode EP64-128/DP64-128 最大 GPU 允许；每卡权重流量 ∝ E/e、compute ∝ Bk/e → 每步 ∝ 1/e 直到通信/同步主导（EP>32 递减 doc L293-297）；EPLB recap 7.4（W4 S3-11 errata 已收窄 8.1 recap）。
- **CP serving**：LoongServe ESP（2404.09526，见下）；DeepInfer **balanced CP**（README L12「Parallelism: Attention DP / balanced CP, MoE EP + EPLB」——release 支持）；**生产不用 CP**（doc L67 主因 V4.1 足够快 / 全 cache miss tail TTFT 可接受；代价面 e2e ≤5% 慢「reasonable tradeoff」+ worsens EP balance + kernel/comm/CPU 开销）。CP 度记 c（§4.3），仅 LoongServe 对照、不入默认决策元组 t/p/e/d。
- **LoongServe ESP（2404.09526v2，[A]）**：§3 ESP 每 iteration 动态 DoP（prefill 高 DoP 用全集群 / decode 降 DoP 减通信释资源）；elastic instance = 模型副本 + 统一模型并行；global manager + dispatcher + elasticity controller + 统一 KV cache pool。**§4.1 scale-down = proactive migration**（prefill SP ring 流通 KV 时选择性留在新组 R' 的 instance，零额外迁移；buffer O(bsh/d) 单层；消 reactive migration 的 O(blsh/d) 碎片约束 + 1M-token 7B KV>488GB 数秒迁移）；**§4.2 scale-up = multi-master distributed decoding**（master 持本请求 KV + 发 Q 给组内其余 instance 并行 local attention 回传，FFN 跨 master 并行 + Q 交换与 local attention 重叠；加 instance 不迁现有 KV）。§5 global manager：SIB（scaling information base）profiling + 解析模型 α_p+β_p·线性+γ_p·二次（最小二乘拟合）；DP batching（prefix sum 多项式时间）；decode DoP = 启动时最小 best DoP（模型并行度）。§7：3.85× vs chunked prefill（LightLLM SplitFuse）/ 5.81× vs DistServe（4P+4D）/ 4.64× vs vLLM；P90 goodput ablation。**改 TP 度需重启 runtime 分钟级（§2.4）→ 弹性走 SP 维**。
- **PP serving**：prefill micro-batch 流水（bubble vs 延迟）；decode PP token 级流水（每 token 过 L/p 层段）。training 5.3 PP 气泡 recap + 指引（serving 无 backward → 气泡结构差异）。来源：loongserve、deepinfer-doc、deepinfer-repo、vllm-repo、sglang-repo、training 5.1/5.3/5.4/5.5（纯文字指引）。

## 8.2 PD 分离原理【重】

**干扰问题**：prefill compute-bound 独占 → decode TPOT 尖刺；colocated TTFT-TPOT 耦合（DistServe §1 / Splitwise §II / Beyond-the-Buzz §2：colocated 强制单实例同时优化 FTL+TTL，资源调度内在张力）。

**DistServe（2401.09670v3，[A]，goodput 优化强制重推导锚）**：
- **goodput 定义（§1 逐字，W1 核本轮再确认）**：「maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query.」要件 = per-GPU 归一 + attainment 目标。
- **数值实例（§1）**：13B/A100/90% attainment，colocate 单卡 goodput ≈1.6 rps；分离 prefill 5.6 rps / decode 10 rps；2P+1D 整体 10 rps = 3.3 rps/GPU = 2.1× colocate。摘要：7.4× more requests 或 12.6× tighter SLO（>90% 达标）。
- **优化三变量（§4）**：(a) 每相并行度 (b) 每类实例数 (c) placement；目标 max per-GPU goodput。**§4.1 高带宽 Algorithm 1（两级）**：枚举每相可行并行配置 → simu_prefill/simu_decode **二分搜索**最大 goodput（SLO attainment 约束）→ 每相独立最优 → replication n,m = ⌈R/config_p.goodput⌉, ⌈R/config_d.goodput⌉；目标函数 config.goodput/config.num_gpus。**§4.2 低节点亲和 Algorithm 2**：同节点 colocate P/D 段（强制 NVLink 传输）；枚举 inter_op（PP）1..N + get_intra_node_configs，约束 P_p.num_gpus+P_d.num_gpus ≤ M（每节点 GPU 数，通常 8）；simulate 选最优 per-GPU goodput → replicate n=⌈R/goodput⌉。
- **simulator（§4.1 + Appendix A）**：FLOPs + 内存访问分析 + latency model；从历史 trace 拟合到达/长度分布、重采样估 SLO attainment（无解析式）；§6.4 验证与真实 profiling 吻合。
- **KV 传输开销（§3.3）**：OPT-66B 512-token KV ≈1.13 GB；10 rps → 11.3 GB/s = **90 Gbps** 使开销不可见；InfiniBand 800 Gbps 或 intra-node NVLink（A100 600 GB/s）→ negligible；placement 约束由此而来。
- **实现（§5）**：orchestration layer（request dispatch / KV transmission / results delivery）；**NCCL 跨节点 + async CudaMemcpy 节点内**（非阻塞）；Ray actor GPU workers；continuous batching/FlashAttention/PagedAttention。testbed 4 节点 32 A100-80GB、NVLink 节点内、跨节点 25 Gbps → 低节点亲和 placement。
- **官方 code**：LLMServe/DistServe（SwiftTransformer backend）。

**Splitwise（2311.18677v2，[B]）**：
- **phase splitting（摘要 + §II）**：compute-intensive prompt computation vs memory-intensive token generation；token generation 不需最新 GPU 算力、可低功率低成本运行；split 到独立机器、按相独立 provision + 异构硬件。
- **Azure trace（§3，W1 双重核）**：两个 Azure LLM inference 服务 2023-11-11 生产 trace（code/conv），释放子集 Azure/AzurePublicDataset AzureLLMInferenceDataset2023（CC-BY，无 prompt 内容只给尺寸）。
- **§IV-A cluster-level scheduling**：prompt pool / token pool / mixed pool；机动态进出 mixed pool 减碎片 + 高负载保 SLO（mixed 机保留身份、队列清空回原池）；粗粒度 re-purposing（不频繁）；CLS 同时指派 prompt+token 机以重叠 KV 传输与 prompt 计算；队列超阈值 → mixed pool → 反池。§IV-B machine-level scheduling。
- **§IV-C KV-cache transfer（layer-wise 重叠锚）**：naive serialized 传输（prompt 完后、下一 token 前）直接顶 max TBT + e2e；传输时间 ∝ KV size（∝ prompt tokens）/ 带宽；**优化 = 逐层异步传输**（每层算完触发该层 KV async transfer，prompt 继续下一层）→ 早启 token 相 + 早释 prompt 机 KV 内存；需逐层细粒度同步（小 prompt 可能干扰增 TTFT，但小 prompt KV 小不需 layer-wise）；**小 prompt serialized / 大 prompt layer-wise**（token 数启动已知）。InfiniBand 25-50 GBps/GPU pair（§II）。
- **cluster design（§IV-D）**：同质/异构、优化 throughput/cost/power；**1.4× 吞吐 @ 20% 低成本 / 2.35× 吞吐 @ 同 cost+power**（摘要）；H100 3.43× compute / 1.75× power vs A100（§III）；power cap 分析（§III-F）。
- **官方 code**：仅 mutinifni/splitwise-sim 模拟器 + vLLM PR #2809 → [B]。

**Mooncake（2407.00079v5，[A]，recap 2.4 + 指引）**：KVCache-centric disaggregated（prefill/decode 集群分离 + 利用 GPU 集群闲置 CPU/DRAM/SSD 做 disaggregated KVCache cache）；KVCache-centric scheduler（max effective throughput + 保 SLO）；prediction-based early rejection（过载）；摘要 525% 吞吐（模拟）/ 75% more requests（生产 Kimi）。transfer engine 详 8.3。

**xPyD 配比账（强制重推导 §6.3）**：
- route-C M-5 锚：到达率 λ、输入 T_in、输出 T_out；P 侧需求 λ·2N·T_in FLOPs/s → n_P ≈ λ·2N·T_in/(F_P·ρ_P)；D 侧并发 C ≈ λ·T_out·TPOT（Little's law），带宽需求 C·κ·L·T̄/TPOT → n_D ≈ C·κ·L·T̄/(TPOT·BW_D·ρ_D)（T̄≈T_in+T_out/2）；配比 n_P:n_D = [2N·T_in/(F_P·ρ_P)] : [T_out·κ·L·T̄/(BW_D·ρ_D)]——**负载与硬件依赖**（输入/输出比、κ、F/BW 比）；MLA 小 κ 使 D 侧便宜、长输出 agent 使 D 侧贵。计数用 n_P/n_D（§4.3，禁裸 P/D/G）。
- **DeepInfer ~1:1 = 生产流量经验配比**（doc L91「In practice the GPU ratio between P and D is roughly 1:1」）；最小高效单元 **64P+64D**（EP64 最小 decode 域量子化，doc L91）；1P+1D 可部署。
- **Beyond-the-Buzz rate matching（§3.2）**：integer solver 求 P:D 实例比平衡两相吞吐；**§4.3 动态 rate matching**：最优 ctx:gen 比随模型 + 目标延迟显著变化（ratio 3.5 利 relaxed latency / 0.5 利 tight latency；固定比退化）→ 须动态；**§4.2 disaggregation 最有效 = prefill-heavy（ISL≫OSL）+ 大模型（>10B）**；colocated context chunking 效果对 attention 机制敏感（MLA vs GQA）、最利 relaxed latency + generation-heavy。
- **Dynamo runtime-reconfigurable xPyD**（disagg-serving.md L65-72）：worker 运行时加入（register discovery + RuntimeConfig KV capacity）/ 移除（drain + deregister），router 自动发现。

**何时分离 vs colocated**：分离收益 = 干扰消除 − KV 传输成本；workload 依赖（Beyond-the-Buzz：prefill-heavy + 大模型最利；Dynamo disagg-serving.md「Capacity effects」：低并发 aggregated 省传输 + 池碎片开销 / 持续负载分离隔离延迟 / decode KV 容量瓶颈时加 prefill 副本反降总容量除非延迟收益补偿）；chunked prefill 替代方案 **Sarathi-Serve（2403.02310v3，[A]，对照）**：stall-free batching + chunk size 调优（recap 4.3 + 指引）。

**KV 传输 vs 重算临界（强制重推导 §6.3，C-S3-2 陷阱注）**：t_xfer = κLT/BW_net；t_recomp ≈ (2NT + attn 二次项)/F_eff；**线性项 T 相消 → 临界带宽阈值 BW* = κL·F_eff/(2N)**，非临界长度；长度依赖来自 attention O(LT²d) / TTFT 预算 / 重算占 decode 机干扰；layer-wise 重叠使传输边际成本 = max(0, t_xfer − 可重叠 prefill 计算)。route-C M-2 数值锚（V3 MLA κ=576 B FP8、L=61、N_act=37B、H100 FP8 1979 TFLOPS×MFU 0.4）：KV=35,136 B/token、t_recomp=93.5 μs/token、**BW*≈0.376 GB/s≈3 Gbps**；DeepInfer 跨 DC 400 Gbps/8 GPU=6.25 GB/s/GPU ≫ BW* → 传输远优于重算（与 doc L97 自洽）。V4.1 κ 更小（主 KV per-16 e4m3 288 B/token、L=40）结论同向更强。来源：distserve、splitwise、mooncake、sarathi-serve、beyond-buzz、deepinfer-doc、azure-llm-trace。

## 8.3 KV 传输与互联

**传输语义**：layer-wise 流式（Splitwise §IV-C / Beyond-the-Buzz §5.1 假设逐层即传重叠后续层计算）vs full 一次性；push（prefill WRITE）vs pull（decode READ）；P2D 时序与 prefill 计算重叠。

**NIXL（ai-dynamo/nixl，教学参照，cb66ce83）**：
- **定位（README + docs/nixl.md）**：NVIDIA Inference Xfer Library，AI 推理（Dynamo）点对点通信加速，**memory（CPU/GPU）+ storage（file/block/object）统一抽象、模块化插件架构**；核心 backend UCX + GDS；数据面 RoCE/InfiniBand/GPUDirect RDMA/NVMe-oF/TCP/NVLink；conductor 进程管理（每 agent 可管多 GPU，全局唯一 name）。
- **Transfer Agent 三抽象（docs/nixl.md Design）**：Memory Section（DRAM/VRAM/NVMe-oF/Object/File 段注册）/ Transfer Backend Interface（按 src/dst 内存类型 + 双方共有 backend 选最优）/ Metadata Handler（side-channel 或 etcd/Redis 中央；**控制路径交换非数据路径**；缓存；动态加 agent = 提供 metadata、删 agent = invalidate cache + disconnect）。
- **C++ agent API（src/api/cpp/nixl.h）**：nixlAgent **L35**；registerMem **L118** / deregisterMem L131；prepXferDlist **L184**（prepared descriptor list handle）；makeXferReq **L257** / createXferReq L306；postXferReq **L345**；getXferStatus **L355**（异步）；getLocalMD **L512** / loadRemoteMD **L543** / invalidateRemoteMD **L555**。
- **descriptor 语义（src/api/cpp/nixl_descriptors.h）**：nixlBasicDesc **L33**（addr uintptr_t + len size_t + devId uint64_t，单连续 memory/storage 区）/ nixlStrideDesc L168 / nixlBlobDesc L226；nixlDescList<T> L343；**nixl_xfer_dlist_t L547**（传输列表）/ **nixl_reg_dlist_t L559**（注册列表）/ nixl_stride_dlist_t L553 / nixl_remote_dlist_t L564。**nixl_xfer_op_t {NIXL_READ, NIXL_WRITE}（nixl_types.h L48）**。
- **插件（src/plugins/）19 个**：ucx / cuda_gds / **mooncake** / hf3fs / obj / posix / libfabric / gpunetio / gpudirect_tcpxo / doca_memos / azure_blob / redis / uccl / rocm_ais / infinia / gusli / telemetry / tracing。**NIXL 经 mooncake 插件可用 Mooncake TE**。
- **引擎集成（W3 核维持）**：vLLM NixlPullConnector L374（READ）/ NixlPushConnector L415（WRITE）；SGLang disaggregation/nixl/conn.py NixlKVManager L531。

**Mooncake transfer engine（kvcache-ai/Mooncake，[A] 继承，2cdbaeeb）**：
- **设计文档（docs/source/design/transfer-engine/index.md，678 行）**：零拷贝，**Segment + BatchTransfer** 两抽象。Segment = 可远程读写连续地址空间（RAM Segment DRAM/VRAM / NVMeof Segment 持久存储）；每进程一个 RAM Segment（名 local_hostname），内部 Buffer（连续同设备、rkey 权限 + NIC 亲和）；openSegment(name) 远程引用。BatchTransfer = READ/WRITE 请求数组（op/length/local/remote addr），异步 getTransferStatus，「async flexible AllScatter/AllGather」；local memcpy/cudaMemcpy / SHM / TCP / RDMA（**multi-NIC pooling + retry**）/ HIP / cuFile GDS。
- **topology-aware path selection（index.md L59-73）**：每服务器生成 **topology matrix** 广播集群；按内存类型分 **preferred / secondary NIC** 列表（NUMA / GPU Direct RDMA 经本地 PCIe switch）；正常用 preferred、故障用两者；**单请求 >64 KB 切多 slice、各 slice 可走不同路径 → 全 NIC 协同**。
- **endpoint management（L75-88）**：按需建连、endpoint pooling（cap 活跃连接）、**SIEVE 驱逐**、waiting_list_ 异步回收（~1 Hz monitorWorker heartbeat）。**fault handling（L90-98）**：NIC 暂不可用 → 自动选替代路径重提交；检测 RDMA context/CQ 问题暂避至恢复。
- **C++ API 源码锚**：transfer_engine.h **TransferEngine L90** / openSegment **L156** / registerLocalMemory **L187** / submitTransfer **L205** / submitTransferWithNotify L258 / allocateBatchID **L287** / getTransferStatus **L305** / getBatchTransferStatus L314 / getTaskCongestionState L308（新增 congestion API）。topology.h **TopologyEntry L38**（preferred_hca L40）/ **Topology L64** / discover **L74** / selectDevice **L109**（storage_type+retry_count）/ getMatrix **L126** / matrix_ L134。transport.h **TransferRequest L66**（**OpCode {READ, WRITE} L67** / opcode L72 / source L73 / target_id L74）。Transport 类：Tcp/Rdma/Efa/NVMeoF/Nvlink/IntraNodeNvlink/Hip/Shm。
- **引擎集成（W3 核维持）**：SGLang mooncake/conn.py MooncakeKVManager L364（send_kvcache L1675）+ device_communicators/mooncake_transfer_engine.py MooncakeTransferEngine L117；vLLM mooncake/mooncake_connector.py MooncakeConnector L798。

**GDR（recap 2.4 + 指引，DeepInfer doc L191 逐字）**：「Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.」RDMA NIC→HBM 直写不占 SM / copy engine。**硬件层（RDMA 协议 / NIC / PCIe 拓扑）归 cluster 维度**。

**带宽账推导（P2D 传输带宽需求）**：
- **Beyond-the-Buzz §5.1 公式（MHA/GQA 几何，MLA 以 κ 替）**：BW_egress = (N_layers × BS_prefill × ISL × d_head × N_kv_heads × bytes_element)/(FTL × NumGPU_prefill)；BW_ingress = (N_layers × BS_decode × ISL × d_head × N_kv_heads × bytes_element)/(TTL × OSL × NumGPU_decode)。**归一只计实际分片 KV 的 GPU**（DP attention 下非全部）；egress 随 ISL 增而降（FTL 涨更快）；ingress KV size 与 TTL 同 ∝ ISL 相消、∝ 1/OSL；大模型 + MLA egress 可能更低；结论「现有数据中心带宽足以支撑 KV 传输不成瓶颈」。
- DistServe §3.3：90 Gbps 使开销不可见（OPT-66B 512-token 10 rps）。
- **DeepInfer 跨 DC（doc L97 逐字）**：V4.1 压缩 KV 使跨 DC P/D 技术可行（可靠性代价，如光纤易被施工挖断）；**~400 Gbps/8 GPU**；DeepSeek 推理节点部署于不同但相邻 DC。
- **layer-wise 重叠时序**：Splitwise §IV-C（逐层 async transfer 重叠 prompt 下一层）；Beyond-the-Buzz §5.1 假设逐层即传；DeepInfer doc L191 swap-in 重叠逐层 H2D 与 forward。
- **vLLM/SGLang KV connector 接口（↔ 5.4/6.3，W3 核维持）**：vLLM KVConnectorBase_V1 base.py L182 + NixlPush/Pull + MooncakeConnector + MultiConnector L142；SGLang disaggregation/base/conn.py KVArgs L62 / KVPoll L142 / BaseKVManager L150 / BaseKVSender L191（send L209/poll L240）/ BaseKVReceiver L266 / BaseKVBootstrapServer L339；PD bootstrap 时序（decode 注册 → bootstrap 撮合 room → prefill send_kvcache RDMA → decode poll → PREBUILT forward）。来源：nixl、mooncake、deepinfer-doc、vllm-repo、sglang-repo、beyond-buzz、splitwise、distserve。

## 8.4 PD 分离系统实现对照

**两层纪律（§4.2）**：6.3（SGLang impl）/ 7.4（DeepInfer VE-P-D）已走读内容 recap + 指针；本章增量 = 对照总表（KV 传输路径 / 调度编排 / 弹性 / 开源完整度 / 硬件支持 / 适用规模六轴）+ 选型论断。

**vLLM disagg（W3 核维持）**：kv_transfer/kv_connector/v1/（nixl/ mooncake/ mooncake_store/ lmcache_connector.py hf3fs/ multi_connector.py hisparse/ offloading_connector.py ...）；KVConnectorBase_V1 + NixlPull/Push（READ/WRITE 双模）+ MooncakeConnector + MultiConnector（组合）；prefill/decode 实例编排经 kv_role + kv_transfer_config。Dynamo 视角：vLLM 用 kv_transfer_params（block IDs + remote worker 连接信息），**prefill 同步、decode 等 prefill 完**（disagg-serving.md L60）。

**SGLang disaggregation（W3 核维持，recap 6.3 + 指引）**：base/conn.py 四抽象（KVArgs/KVPoll/BaseKVManager/Sender/Receiver/BootstrapServer）+ common/conn.py 公共实现 + mooncake/nixl backend；prefill.py PrefillBootstrapQueue L270 / decode.py DecodePreallocQueue L471 + DecodeTransferQueue L3303；bootstrap room 撮合。Dynamo 视角：SGLang 用 **bootstrap_info（host, port, room_id）**，prefill 可作 background task、**decode 立即开始 + KV 传输并行**（disagg-serving.md L58）。

**Dynamo（ai-dynamo/dynamo，教学参照，c41b6dfc）**：
- **定位（README L36）**：「the orchestration layer above inference engines — it doesn't replace SGLang, TensorRT-LLM, or vLLM, it turns them into a coordinated multi-node inference system」；特性表 Disaggregated Serving / KV-Aware Routing / SLA-Based Planner 全 ✅（vLLM/SGLang/TRT-LLM）。
- **disagg 编排（disagg-serving.md）**：**PrefillRouter** 流程（select prefill worker KV-aware → prefill 算 KV 返 disaggregated_params → select decode worker + transfer metadata → NIXL VRAM→VRAM 直传 non-blocking → generate）；**NIXL KV 传输**（NVLink/InfiniBand/UCX 最优 transport）；backend 元数据 SGLang bootstrap_info / vLLM kv_transfer_params / TRTLLM opaque_state；**runtime-reconfigurable xPyD**（worker 加/删 + discovery 自动发现）。
- **KV-aware routing（kv-aware-routing.md + routing-concepts.md）**：tokenize/normalize → KV router（**KV cache index 报每 worker prefix overlap + active-load tracker**）→ filter → score → select；worker 发 KV create/release events 更新 index（--no-router-kv-events 则从路由决策预测）。**成本模型（routing-concepts.md L29-33）**：raw_prefill_blocks = active_prefill_blocks + incoming_prompt_blocks；adjusted_prefill_blocks = max(0, raw − overlap_credit_blocks)；potential_decode_blocks = active_decode_blocks + incoming_active_blocks；active_request_blocks = decode_active_request_weight × active_requests；**cost = prefill_load_scale × adjusted_prefill_blocks + potential_decode_blocks + active_request_blocks**；选最低 cost 合格 worker。overlap_credit = device/host/disk/shared-cache credits（overlap_score_credit_decay 减 device-local 当 cache-rich worker 过载）；conditional-disagg 例外 cost = max(0, potential_decode_blocks − overlap_credit) + active_request_blocks（cache-hot decode worker 绕过 prefill）。prefill load model 默认静态至首 output token / --router-prefill-load-model ais（AISimulate 预期时长 + elapsed 衰减）；decode --router-track-output-blocks + nvext.agent_hints.osl 分数权重。**placement 决策非无条件 affinity**（cache-hot 可输给 colder 若 active load 高）。代码 lib/llm/src/kv_router.rs + kv_router/{indexer/mod.rs, shared_cache.rs, active_load.rs, ...}（Rust）。
- **planner（planner/README.md + planner-design.md，↔ 8.5）**：SLA-driven autoscaling controller；**throughput-based**（traffic prediction + engine perf model → 副本数保 TTFT/ITL SLA；bootstrap from worker self-benchmark / AI Configurator / profiler / live FPM regression）+ **load-based**（ForwardPassMetrics FPM event plane reactive；optimization_target: sla → online regression  target TTFT/ITL；支持 vLLM/SGLang/TRT-LLM 含 attention-DP per-rank 信号）；throughput 设副本下界、load 实时调整；prefill/decode/aggregated PlannerEnginePerfModel；KV hit rate + spec decode accept length 预测器。

**TensorRT-LLM disagg（NVIDIA/TensorRT-LLM，教学参照，⟨verify⟩ 已销）**：
- **当前形态（disagg-serving.md）**：Python transceiver **KvCacheTransceiverV2** + **backend: NIXL** + KV cache manager V2 默认；C++ transceiver 1.4（2026-11）弃用（3 月迁移期）；backend NIXL 无默认（unset 拒 disagg 请求）；**bounce buffer**（kv_cache_bounce_size_mb，gather 小 fragment——ctx TP≠gen TP / attention DP 时多小 descriptor；fabric MNNVL 内存或 cudaMalloc；默认 ≥1024 descriptor 且平均 ≤16 KiB 才用；closeout 更正「均」→「平均」，一手 disagg-serving.md L182「averaging at most 16 KiB」，S2-4）。
- **编排三路线**：Dynamo（smart router + pre/post-processing workers + K8s）/ **trtllm-serve disaggregated**（ctx+gen 各 OpenAI-compatible server + disaggregated orchestrator server；ctx_params 握手；**Coordinator + Worker Fleet** 扩 orchestrator：SO_REUSEPORT 共享端口、coordinator 拥 cluster state /select /finish /renew、KV-aware routing ZMQ event-ingest、**output-length prediction ridge/ridge_gbm → generation priority shorter-first**）/ Triton BLS（blog05，Executor API 仅 TensorRT backend）。
- **KV 传输**：cache_transceiver_config（backend/max_tokens_in_buffer/kv_transfer_timeout_ms/kv_cache_bounce_size_mb）；cpp cache_transmission/{agent_utils,mpi_utils,ucx_utils,nixl_utils,mooncake_utils} + cacheSplitConcat.cu + batch_manager/cacheTransceiver.{h,cpp} + cacheFormatter；**overlap**（KV 传输与其他请求计算重叠 + 多 GPU 集并行传输）；**cache layout transformation**（device-to-device 直传、ctx/gen 不同并行策略需 block mapping 编排，TP2 ctx→PP2 gen 例）；**unique global request ID**（snowflake 64-bit，DisaggregatedParams.disagg_request_id，ctx↔gen 传输 keyed by it）；schedule_style context_first/generation_first。
- **性能（blog05，官方披露）**：DeepSeek R1 MTP off disagg 1.4-1.8×（低并发尤甚）/ MTP on 1.6-2.5×（平均高 20-30%）；GEN4 1.73× / GEN8 2×；50 tok/s/user 1.7×。**rate-matching 方法（blog05 L186-189）**：测 ctx 各 TP/EP/DP/PP 满足 TTFT 的 request throughput 选最优 → 测 gen throughput/latency → 找 ctx:gen worker 比使两相聚合吞吐匹配（按 ISL/OSL）→ per-GPU throughput 公式。
- **IFB（9.2 文档出处已销）**：paged-attention-ifb-scheduler.md（continuous/iteration-level batching、packed no-padding、context 序列排 generation 前）。
- **论文锚**：disagg-serving.md Motivation 引 **Beyond-the-Buzz（2506.05508）**。

**DeepInfer VE/P/D（recap 7.4 + 指引，W4 核维持）**：E/P/D 分离 + 专用存储节点；**Entrypoint CHECK 角色 hash-chain 前缀匹配**（doc L191，metadata only、hit length 路由前不可隐藏）；VE 单 GPU cheaper（doc L51）；async fetch 3FS→host shm→GDR swap-in；**release 未含编排层**（README L38-42 三项：KV cache 系统 / VE-P-D disagg「most required engine functionality is already implemented」/ prefill+decode DP LB）——引擎侧 PE/CE engine type + StateBlock + EPLB 在 release，Entrypoint/三 DP LB/3FS kvcache-store 编排/平台弹性不在（W4 release 边界代码现实）。

**MemServe（2406.17565v3，[B]）**：
- **MemPool elastic memory pool（§4）**：管理集群全内存（CPU DRAM + GPU HBM）；**三类 API（memory / indexing / distributed transfer）**；每 instance 内运行、fixed-size allocator；indexing API 映射 prompt tokens→KV（active + historical KV）；**transfer API 抽象三异构（parallelism / network / memory medium）**；Table 1 alloc_mem（type HBM-only/DRAM-only/mixed、@instance id、返 addrList、address 编码 instance ID、transfer flag 控 on-demand allocation）；transfer 三步（allocation→transmission→insertion，receiver alloc 返 addr+parallelism config；指定 dst addr 可跳 allocation → **layer-by-layer 传输**）。
- **context caching + disagg 首次结合（§5）**：四步（distributed API 复现 disagg → prefill-only instance 加 caching → decode-only 加 caching → **decode-to-prefill 数据传输**）；§5.2 huge pages 共优内存布局 + 网络传输；§5.3 context caching cost model。
- **global scheduler（§6）**：**global prompt tree**（distributed radix tree + instance 持有字段）locality-aware best-effort routing 最大化 KV 复用。
- **§4.4 failure handling/scaling**：cluster manager heartbeat 检测故障、广播 cluster info、instance 释放故障 instance 分配的 memory block 防泄漏。
- **实现（§7）**：MemPool + global scheduler from scratch **5.6K SLOC Python**；改 vLLM **200 SLOC Python + 400 SLOC CUDA C++**。
- **评估（§8）**：ShareGPT PD-disagg+caching 最优；disagg vs colocated JCT 改善至多 **42%**，+caching 再 **29%**；LooGLE disagg **10.8%** + caching **26.9%**。
- **无官方 repo** → [B]。

**对照总表素材（六轴）**：KV 传输路径（vLLM NixlPush/Pull+Mooncake / SGLang mooncake+nixl base/conn.py / Dynamo NIXL VRAM→VRAM / TRT-LLM NIXL+Python transceiver V2 / DeepInfer 3FS kvcache-store+GDR swap-in / MemServe MemPool transfer API）；调度编排（vLLM kv_role / SGLang bootstrap room / Dynamo PrefillRouter+KV-aware cost model / TRT-LLM trtllm-serve orchestrator+coordinator fleet / DeepInfer Entrypoint+三 LB（release 外）/ MemServe global prompt tree）；弹性（Dynamo runtime xPyD+SLA planner / TRT-LLM dynamic node joining 在建 / DeepInfer autoscaling+grace+EP failover / MemServe MemPool scaling / Llumnix migration）；开源完整度（vLLM/SGLang/Dynamo/TRT-LLM 全 / DeepInfer 引擎侧在编排外 / MemServe 无 repo / Splitwise 仅模拟器）；硬件支持（NIXL UCX/GDS 多 transport / Mooncake RDMA+NVLink+NVMeoF / DeepInfer SM100+Ascend 950）；适用规模。来源：vllm-repo、sglang-repo、dynamo、tensorrt-llm、deepinfer-repo、deepinfer-doc、memserve、nixl、mooncake、lmcache、beyond-buzz。

## 8.5 弹性伸缩与容错

**autoscaling（DeepInfer 官方口径 doc L251 逐字）**：「Prefill determines the number of machines it needs from the number of non-cached tokens; decode adjusts based on the number of running requests. Adjustments are rate-limited and bounded above and below.」**伸缩触发账（关键推导）**：non-cached tokens → prefill 机数（算力需求）/ running reqs → decode 机数（并发/带宽需求）；rate-limit + 上下界。
- **Dynamo SLA planner 对照（planner-design.md）**：throughput-based（traffic prediction + perf model → 副本下界保 TTFT/ITL）+ load-based（FPM online regression reactive SLA 修正）；prefill/decode 独立 perf model + worker inventory（ready/expected counts + scale in-progress）。**SLO-aware 伸缩策略对照**：DeepInfer 按 non-cached tokens/running reqs（负载信号）vs Dynamo 按 TTFT/ITL SLA（延迟信号 + 预测）。

**grace period 与请求迁移（doc L253 逐字）**：「a decode engine (CE) enters a grace period (1 min) after receiving a shutdown notice; any requests still unfinished, along with their KV cache, are forwarded in one shot to other CEs to continue decoding.」**grace 迁移成本（关键推导）**：一次性转发未完成请求 + KV（传输字节 = 剩余 KV）vs 重算；1 min grace 窗口。
- **Llumnix live migration（recap 4.2 + 指引，2406.03243v1，[A]）**：§4.2 **multi-stage migration**（append-only KV → stage 0 拷已完成 iteration KV 同时继续计算 → stage 1 拷 stage 0 期间新生 KV → ... → 末 stage drain 出 batch + 拷末 block = downtime）；**downtime ≈ 末次 iteration KV 拷贝，常数 20-30 ms 与序列长无关**（§6.2，短于单 decode step；naive recompute/copy 达 50-111×；8k LLaMA-30B recompute 3.5 s ≈ 54 decode steps）；每 stage 前 pre-allocate handshake（dst 预留 block，成功/失败通知 proceed/abort）、每 stage 后检查请求完成/抢占（abort 释放）；末 stage commit + resume。**VM live migration 借鉴但无 dirty page tracing**（KV append-only）。§4.3 distributed scheduling：global scheduler（instance-oriented、按 load 配对 src/dst）+ **llumlet**（local scheduler + migration coordinator；load = virtual usage 之和）。§4.4 **virtual usage** 统一四场景（load balancing / de-fragmentation / prioritization / drain-saturate for auto-scaling）；freeness metric F（可负）；迁移优先低优先级 + 短序列请求。**§5 Gloo Send/Recv 非 NCCL**（NCCL 并发 unsafe）；Ray actors；故障 = 故障 instance 请求 abort、Ray 重启 actor。§6.5 auto-scaling freeness 阈值 [10,60]；cost saving 至多 16%（同激进度）/ 36%（同 P99 prefill latency 目标）。摘要：尾延迟改善一个数量级 / 高优先级加速 1.5×。
- **代码锚（llumnix-ray v0，3fb6c037）**：llumnix/llumlet/migration_coordinator.py **MigrationCoordinator L228** / migrate_out **L251** / MigrationStatus L42；local_migration_scheduler.py；backends/vllm/migration_backend.py **RayRpcMigrationBackend L62** / **RayColMigrationBackend L220**（gloo L260，pygloo bf16→glooFloat16 L212-215）；llumnix/manager.py（global scheduler）/ **scaler.py Scaler L84** / load_computation.py（BaseLoad/KvBlocksRatioLoad）。

**冗余 quota（doc L257 逐字）**：「ready machines held unoccupied in the pool as scale-out inventory, and always keeping more instances online than the computed demand — concretely, scale-in is forbidden whenever quota utilization is above 0.8.」**冗余 quota 权衡（关键推导）**：ready 机池 + 在线实例 > 计算需求；quota 利用率 > 0.8 禁 scale-in（吸收流量摆动）。

**EP failover（doc L265 逐字）**：「all GPUs of an instance form a single communication domain, so a single failed GPU makes the whole instance unusable … the platform re-determines the EP size from all currently healthy GPUs, and the instance takes the number of GPUs it actually receives as its EP size.」大 EP 单通信域单 GPU 故障 → 整实例不可用；重启从健康 GPU 重定 EP size、实例以实收 GPU 数为 EP。

**容错哲学（doc L261 逐字）**：「Restarts are on the order of minutes. Fault tolerance comes from PE / CE quota redundancy plus request-level rerouting, rather than from shortening the restart time of a single engine.」重启分钟级 vs 请求级 rerouting（不靠缩短单引擎重启、靠 quota 冗余 + rerouting）。

**训推集群共调度（doc L119 逐字）**：「Training and inference clusters are managed together, and the resource allocation between them can be adjusted dynamically.」资源随流量动态分配（serving 侧资源视角，未越训练侧编排）。

**故障分类与检测**：健康检查（Llumnix heartbeat / MemPool CM heartbeat / Dynamo worker readiness）；SDC 一句 → training 7.3 指引。**弹性 EP 引擎侧（W3 核维持，指针）**：SGLang elastic_ep/ ElasticEPStateManager L132 / ScaleCohort L30 / update_dp_attention_for_elastic_ep L77；vLLM distributed/elastic_ep/（elastic_execute/elastic_state/standby_state）。来源：deepinfer-doc、llumnix、dynamo、memserve、training 7.3（纯文字指引）。

## serving 侧 vs training P5 并行差异（writer 须知的 recap 边界）

| 维度 | training P5（已交付，recap + 指引） | serving 侧增量（8.1） |
| --- | --- | --- |
| TP 通信次数 | 每层 4 次 all-reduce（前向 g×2 + 反向 f×2，5.1 §6 式 (10)） | **无 backward → 每层 2 次前向 × L**（C-S3-1 防 2× 低估） |
| TP 消息尺寸 | [b,s,h]，b=micro-batch 序列数、s=序列长（5.1 式 (9) 2bsh） | decode s=1 → 2·B·h bytes（B=batch token 数，翻译 + 口径注 §4.3）；小消息延迟项主导 |
| TP 度数约束 | 整除 head 数 + 通信-计算比随 t 线性涨 NVLink 边界（5.1 §7 式 (11)） | 同约束 + decode TPOT 延迟敏感；DeepInfer 装得下 + SLA 可达就不付 TP（doc L69） |
| GQA/MLA KV 复制 | a_kv<t KV 复制、MLA latent 无处可切（5.1 §8） | **DP attention for MLA 消复制**（#DP=#EP，doc L33） |
| EP all-to-all | dispatch/combine 2kd 元素上界 + 木桶 max_j R_j（5.4 §1-2） | 延迟敏感（decode 小消息）+ Wide-EP 每步 ∝ 1/e（recap 7.4） |
| CP | ring attention 训练账（5.2） | LoongServe 弹性 SP（DoP 每 iteration 变）/ DeepInfer balanced CP（release）vs 生产不用（doc L67） |
| PP | 气泡账（5.3） | prefill micro-batch 流水 / decode token 级流水（无 backward 气泡结构差异） |
| 组合 | t/c/p/e/d world size 乘积 + group 树（5.5） | 模型规模 × 上下文 × SLO × 拓扑 → t/p/e/d 决策表 |

**符号翻译硬规则（§4.3）**：training b（micro-batch 序列数）→ B（batch token 数）并声明口径；training 5.4 γ/α（偏置控制器步长/seq 级 aux 权重）→ 文字；models d（hidden）→ d_model；bytes 不占字母（文字 / \mathrm{bytes}）。**通信硬件（RDMA/NVLink 拓扑/NCCL 实现/交换机）→ cluster 维度**，8.1/8.3 只讲传输语义、带宽账、系统设计。

## arXiv 来源核实（W5 范围，https export API 批量 200，7/7 + 补查 1）

| ID | 标题（逐字） | 第一作者（作者数） | 版本（upd） | 通道 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 2401.09670 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | Yinmin Zhong（8） | v3（2024-06-06） | 批量 API + W1 HTML 全文 | 8.2 goodput 优化 [A] |
| 2311.18677 | Splitwise: Efficient generative LLM inference using phase splitting | Pratyush Patel（7） | v2（2024-05-20） | 批量 API + W1 HTML 全文 | 8.2 phase splitting [B] |
| 2407.00079 | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | Ruoyu Qin（7） | v5（2026-10-08） | 批量 API | 8.2/8.3 [A] |
| 2404.09526 | LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism | Bingyang Wu（6） | v2（2024-10-29） | 批量 API + HTML 全文 | 8.1 ESP [A] |
| 2406.17565 | MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool | Cunchen Hu（11） | v3（2024-12-21） | 批量 API + HTML 全文 | 8.4 MemPool [B] |
| 2406.03243 | Llumnix: Dynamic Scheduling for Large Language Model Serving | Biao Sun（7） | v1（2024-06-05） | 批量 API + HTML 全文 | 8.5 migration [A] |
| 2403.02310 | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | Amey Agrawal（8） | v3（2024-06-17） | 批量 API | 8.2 对照 [A] |
| 2506.05508 | Beyond the Buzz: A Pragmatic Take on Inference Disaggregation | Tiyasa Mitra（13，NVIDIA） | v1（2025-06-05） | 单 ID API + HTML 全文 | 8.2/8.3/8.4 新条目 [B] |

## 残留延迟项（writer / evaluator 须知）

1. **TRT-LLM disagg 当前形态 + 漂移注**（语义纠正 1）：8.4 必须写 Python transceiver V2 + NIXL 默认 + C++ 1.4 弃用，勿按 legacy 文档；9.2 disagg 只 recap 8.4 + 一句选型定位（C-S3-4 防双展开）。
2. **大 repo 下载截断**：dynamo/Mooncake/TRT-LLM 全程 trees + contents API（tarball 不可靠）；后续 research pass 复用此通道。部分解压（dynamo-x 止于 container/、Mooncake-x 止于 docs/source/image/）仅作辅助，关键文件以 API raw 为准。
3. **NIXL C++ vs Python API**：W3 核 Python _api.py，本轮补 C++ nixl.h/nixl_descriptors.h；8.3 走读以 C++ 为主（agent/descriptor 语义），Python 绑定一句注。
4. **Beyond-the-Buzz 无官方 repo**（proprietary simulator）→ [B]；KV 带宽公式 MHA/GQA 几何，MLA 须 κ 替换 + 口径注（语义纠正 10）；FTL/TTL = TTFT/TPOT 异名对照。
5. **LoongServe ESP ≠ 训练 CP ≠ DeepInfer balanced CP**（语义纠正 6）：8.1 三者分清；CP 度 c 仅 LoongServe 对照不入默认元组。
6. **DeepInfer CP 主因**（语义纠正 7，C-S2-3）：V4.1 足够快非「慢 5%」；release balanced CP vs 生产不用两口径。
7. **goodput 定义要件 + 传输临界 T 相消**（语义纠正 8/9，S3-9/C-S3-2）：8.2 强制重推导按此，防 naive 推导。
8. **Llumnix Gloo 非 NCCL + v0 锚**（语义纠正 11）：8.5 recap 4.2 + v0 代码锚（v1 另立）。
9. **MemServe/Splitwise 无完整官方 code** → [B]；8.4 MemServe 讲论文机制（MemPool API/global prompt tree），不推测代码。
10. **release 边界（W4 维持）**：DeepInfer DP LB/Entrypoint/kvcache-store/平台弹性只讲 doc 架构不推测代码；8.4 对照表「开源完整度」轴按此。
11. **通信硬件 → cluster**：8.3 GDR/RDMA/NVLink 只讲传输语义 + 带宽账，硬件层纯文字指引。
12. **W6+ 延迟项不在本轮**：veRL/OpenRLHF 引擎接口（W6）；B200/Ascend 950 datasheet（W1 已核）。
13. **arXiv https export API 本轮恢复**（批量 200）：后续优先 https 批量，http 仍 000。

## references.md 本轮变更清单

- **新增 1 条**：P8 `beyond-buzz`（[B]，arXiv:2506.05508，被引 8.2, 8.3, 8.4；TRT-LLM disagg-serving.md Motivation 引用 + rate matching + KV 带宽公式）。
- P8（4→5 条）：**loongserve**（日期 2026-11-22→2026-11-23，arXiv v2 API + repo f6e8fc15 + ESP 机制 proactive migration/multi-master decoding/SIB + 3.85×/5.81×）；**memserve**（日期→2026-11-23，arXiv v3 API + search 再确认无 repo + MemPool 三 API/global prompt tree/5.6K+200+400 SLOC/JCT 42%/29%）；**nixl**（W5 C++ 深核注：nixl.h agent API + nixl_descriptors.h 三型 + NIXL_READ/WRITE + 18 插件含 mooncake + metadata 控制路径）；**dynamo**（日期→2026-11-23，repo c41b6dfc + planner 双模 + router 成本模型 + PrefillRouter 编排 + NIXL VRAM→VRAM + backend 元数据 + runtime xPyD）。
- 跨波共享锚点追加 W5 复核注：**distserve**（arXiv v3 再核 + placement Algorithm 1/2 + goodput 三变量 + §3.3 90Gbps + §5 NCCL/async CudaMemcpy）/ **splitwise**（arXiv v2 + §IV-A 三池 + §IV-C layer-wise + 1.4×/2.35× + InfiniBand 25-50GBps）/ **mooncake**（arXiv v5 + TE 源码锚 transfer_engine.h/topology.h/transport.h + index.md topology-aware >64KB slice/SIEVE/multi-NIC）/ **llumnix**（arXiv v1 + repo 3fb6c037 + migration 代码锚 + multi-stage downtime 20-30ms + virtual usage/freeness + Gloo 非 NCCL）/ **sarathi-serve**（arXiv v3 批量再核）/ **vllm-repo**（被引 +8.1/8.3，kv_connector/v1 W3 同日 tarball 再确认）/ **sglang-repo**（被引 +8.3，disaggregation W3 同日 tarball 再确认）/ **deepinfer-repo**（被引 +8.1/8.4，README balanced CP L12 + release 边界）/ **deepinfer-doc**（W5 8.1/8.3/8.5 doc 行再确认 L33/L47-53/L67/L69/L97/L191/L251-265）/ **tensorrt-llm**（日期→2026-11-23，⟨verify disagg 形态 + IFB 文档⟩ 销项：Python V2+NIXL 默认/C++ 1.4 弃用/三编排路线/bounce buffer/snowflake ID/IFB doc）。
- **Closeout 更正登记（2026-11-23，W5 审查 S3-8 / S2-4 refs-notes 侧销项）**：① nixl 插件计数 19→18（本文件语义纠正 3 + 变更清单、references.md W5 精炼说明 + nixl 条目同步；evaluator fresh tarball cb66ce83 + contents API 双通道核 src/plugins/ = 18 目录 + meson.build，8.3 章侧原已正确）；② bounce 条件「且均 ≤16 KiB」→「且平均 ≤16 KiB」（本文件 TRT-LLM 当前形态节、references.md tensorrt-llm 条目同步；一手 disagg-serving.md L182「averaging at most 16 KiB」/ 参数名 max_average_descriptor_size；8.3/8.4 章侧已「平均」，8.4 总结 / 练习 3(c) 压缩表述 closeout 顺手补齐）。
- 新增 W5 精炼说明段。parse-check：**9 groups / 83 entries**（P1=11, P2=14, P3=16, P4=19, P5=4, P6=2, P7=9, P8=5, P9=3），无重复 key、无非法日期、无混合档、无竖线。