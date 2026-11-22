# inference 维度 Spec 对抗审查 · 路线 B：P5–P7 引擎走读与 DeepInfer 一手来源核查

- 日期：2026-11-22
- 审查对象：`docs/superpowers/specs/2026-11-22-inference-dimension-design.md`（§5 P5–P7 大纲、§2 边界、§4.1 证据分级、§4.2 V4.1 处理、预核实附录）
- 审查人：fresh-context adversarial evaluator（路线 B，不继承写作上下文；未写本 spec）
- 标准：STYLE_GUIDE.md §6 / §7 / §10（S1 事实致命 / S2 实质 / S3 minor；verification log 强制）
- 任务来源：`.superpowers/sdd/2026-11-22-inference-w0-implementation/task-1-brief.md` Step 1 路线 B

## 0. 核查通道与约束（影响覆盖率，先声明）

| 通道 | 状态 | 用途 |
| --- | --- | --- |
| api.github.com（contents / git-trees，raw Accept） | 可用；60/h 匿名限额本次耗尽 | vLLM / SGLang / DeepInfer 全量 tree + 文件内容 |
| raw.githubusercontent.com | 本机不可达（curl 000 / SSL err 35） | 弃用，改 API contents raw |
| github.com HTML | 可用但间歇超时（全部重试至确定码） | repo 存在性 / 组织更名 redirect |
| hf-mirror.com | 可用（307 → 跟随） | V4.1-Flash config.json 全文 |
| huggingface.co | 本机不可达（curl 000，与 spec risk 7 一致） | 官方渠道复核本机无法完成，留 W0 |
| zhuanlan.zhihu.com | 403（bot 拦截） | 知乎全文未读；URL 在 QA doc 内逐字核实 |

tree 快照：vllm-project/vllm@main 10,978 entries（truncated=false）；sgl-project/sglang@main 14,456 entries（truncated=false）；deepseek-ai/DeepInfer@main 398 entries（truncated=false）。DeepInfer README（4,660 B）与 doc/inference_deployment_qa.md（34,526 B / 373 行）**全文读取**；config.json（3,311 B）全文读取。抓取件存 `/tmp/route-b-verify/fetch/`（审查输入，非仓库交付物）。

**总判定：S1 = 0，S2 = 4，S3 = 6。** 任务书点名的 11 组生产数字、全部 V4.1 config 字段、全部必查 repo 路径均一手核实通过；无虚构数值、无不存在 repo。S2 集中在引擎代码锚点漂移（vLLM 目录迁移 ×2、SGLang connector 语义错挂）与 1 处生产数字流量范围缺失，须 W3/W4 写作前修复。

## 1. Findings 表

| ID | severity | 位置 | spec 断言 | 证据（一手） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| B-1 | **S2** | 附录「GitHub 路径核实」行 + §5 5.2 | `kv_cache_interface.py` / `kv_cache_layout.py` 列于 `v1/core/{…}` 集合内并标「2026-11-22 核实」 | vLLM 全量 tree：两文件实在 `vllm/v1/kv_cache_interface.py`、`vllm/v1/kv_cache_layout.py`（v1 根目录）；`vllm/v1/core/` 清单无此二文件 | 附录路径集合更正；5.2 大纲注明两文件在 `v1/` 根目录，避免 writer 按 core/ 锚定 |
| B-2 | **S2** | §5 4.5 / 5.4 | 「vLLM `v1/spec_decode/`：eagle、ngram、mtp ⟨verify 文件名⟩」 | tree：`vllm/v1/spec_decode/` 仅 `__init__.py`、`dynamic/utils.py`、`metrics.py`；eagle/ngram/mtp 实现在 `vllm/v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py`（同级另有 autoregressive / dflash / dspark 等） | 锚点改 `v1/worker/gpu/spec_decode/`，注明目录迁移与核实日期；`v1/spec_decode/`（metrics / dynamic）降为辅助锚点。⟨verify⟩ 标在文件名上，但目录归属本身错，不能只靠 research pass 兜底 |
| B-3 | **S2** | §5 6.3（及 8.3 的 ↔6.3 指针） | 「`connector/`（KV 传输接口）」 | `srt/connector/base_connector.py`：`BaseConnector` 抽象方法为 `weight_iterator` / `pull_files`，docstring 自述 fs（s3）/ kv（redis）存储 URL，实现为 azure / redis / s3 / serde——是**权重/文件存储 connector**；PD KV 传输接口在 `srt/disaggregation/base/conn.py`（`BaseKVManager` / `BaseKVSender` / `BaseKVReceiver` / `BaseKVBootstrapServer` / `KVPoll` / `KVArgs`） | 6.3 KV 传输接口锚点改 `disaggregation/base/conn.py`；`connector/` 若保留，语义改「权重/文件存储 connector（s3 / redis / azure），与 9.3 checkpoint_engine 相关」 |
| B-4 | **S2** | §5 2.2 / 9.1 / 附录 doc 事实行 | 「72h TTL 损失 0.5-1% 命中率」未限定流量类型 | QA doc 逐字：「Empirically, a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate.」；agent 72h 为生产设定且「captures almost all reusable cache entries」（doc TTL 节）。损失专指 chatbot 缩到 72h，非 agent | 三处均补「chatbot 流量缩至 72h TTL 损失 0.5–1% 命中率；agent 72h 近无损」。数字本身对，范围缺失会让章节写出与官方披露相反的含义 |
| B-5 | S3 | §5 7.1 | 「KV 管理与 LB 代码因内部 infra 耦合未完全开源，架构见官方知乎 + QA doc——**README 显式声明**」 | README 仅列 release 不含三项（KV cache system / VE-P-D disagg / DP LB）；「tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced」+ 知乎链接在 **QA doc 第 21 行**，不在 README | 归属改「README 列 release 边界；耦合原因与知乎架构指针见 QA doc」。同属官方一手锚点，影响小 |
| B-6 | S3 | §5 7.2 / 附录 | 「StateBlock：20 层 × 134 token × 528 B = 1,415,040 B/seq」 | 数字与 doc 逐字一致，但 doc 口径是 `state_size_to_disk`（**仅 stored region 落盘**）；StateBlock slot 本身还含 dropped region（YOCO consumer SWA + ca_state / ica_state）与 ring +6 余量 | 章节写「落盘大小 state_size_to_disk = 1,415,040 B/seq」，勿标成 slot 总尺寸；算式 20×134×528=1,415,040 复核无误 |
| B-7 | S3 | §9 risk 1 / 附录 | 「组织更名用重定向跟到现名（outlines / Mooncake 已各处理一例）」 | 实测 `MoonshotAI/Mooncake` → **404 无 redirect**（两次重试确认）；`kvcache-ai/Mooncake` 200 ✓；`outlines-dev/outlines` → 301 → dottxt-ai/outlines ✓ | 现名正确，但 Mooncake 旧 URL 已死、redirect 机制对其不成立；附录注一句，防后续 research pass 依赖旧链接 |
| B-8 | S3 | §5 8.3 锚点行 | 「NIXL / Mooncake transfer engine [教学参照 + 官方实现]」 | §4.1 自身例表：Mooncake = [A]（2407.00079 + kvcache-ai/Mooncake，repo 本次核实存在）。NIXL 无 paper → 教学参照正确；Mooncake transfer engine 作为 Mooncake 组件应继承 [A] | 锚点拆开标注：NIXL [教学参照 + 官方实现]、Mooncake transfer engine [A]。（8.3 属路线 C 章节，证据分级一致性归本路 item 4） |
| B-9 | S3（advisory） | §5 7.2 | CED「decoder block KV 由末 encoder 层 hidden 投影——config kv_source_layer_ids [W] + doc 口径」 | doc 逐字「projected from the last encoder layer's hidden state」；config `kv_source_layer_ids = [2, 8, 14, 20]`（4 个源层）。spec 两处引用各自无误，但两源存在表述张力 | 章节以 doc 口径为官方披露、config 字段原样列出，**不做模型侧调和**（§2 边界：形式化归 models，V4.1 待补）。另注：README `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 默认 140 是配置保留后缀（须覆盖 128 窗 + 最大回溯），与 doc 的 134 = 128+6 不矛盾，建议章内一句说明 |
| B-10 | S3 | §5 3.4 | 「vLLM / SGLang `batch_invariant_ops`」 | SGLang `srt/batch_invariant_ops/` ✓；vLLM 实现在 `vllm/model_executor/determinism/batch_invariant.py` + `batch_invariant_configs.py`，无 batch_invariant_ops 目录 | 按引擎分列路径；功能两边都有，仅名称归属不精确 |

## 2. ⟨verify⟩ 项本路解决表（供 spec 修订与 W3/W4 research pass 直接引用）

| spec ⟨verify⟩ 项 | 本路结论（2026-11-22 核实） |
| --- | --- |
| vLLM spec_decode 文件名（4.5 / 5.4） | `vllm/v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py`（见 B-2）；旧目录仅 metrics / dynamic |
| SGLang speculative 实现路径（4.5） | `python/sglang/srt/speculative/`（79 文件：eagle_info / eagle_worker 系、cpp_ngram/、dspark_components/ 等） |
| SGLang schedule_policy 策略名（4.2） | `CacheAwarePolicy.LPM / DFS_WEIGHT`；`CacheAgnosticPolicy.FCFS / LOF / RANDOM`（schedule_policy.py L188–203 逐字） |
| DeepInfer `_C` 内容（7.1） | `_C/` 为 native extension 类型 stub（`__init__.pyi` / `cuda_utils.pyi` / `engine.pyi` / `graph_timer.pyi`）；C++ 源码在根 `csrc/`（pybind.cpp、engine/、graph_timer/、cuda_utils.cpp） |
| vLLM disagg 路径（8.4） | `vllm/distributed/kv_transfer/kv_connector/v1/`：nixl/、mooncake/、mooncake_store/、lmcache_connector.py、hf3fs/、multi_connector.py、offloading_connector.py 等 |
| SGLang prefill_delayer / shortest_prefill_load / hisparse_coordinator / mm_schedule / io_struct（6.2 / 4.2 / 9.4 / 6.1） | 文件全部存在（managers/）；语义未逐行读，留 W3 research pass |
| vLLM hisparse / kv_hints 语义（5.3） | 目录存在：hisparse/{backend,binding,block_pool,coordinator,layout,runtime,types}.py、kv_hints/protocol.py；QA doc 提及 Hi-Sparse（DeepInfer decode 不用）；语义留 W3 |
| SafeAILab/EAGLE repo（4.1） | 存在（github.com 200） |
| SGLang req_to_token_pool / token_to_kv_pool（6.1） | `ReqToTokenPool`（memory_pool.py:382）；`MHATokenToKVPool` / `MLATokenToKVPool` / `DSATokenToKVPool` 族（抽象基类现名 `KVCache`，L1949） |
| DualPath 公开性（7.4） | **未解决**：QA doc 仅点名「See DualPath's intra-group scheduling algorithm for details」，无链接无引文；保留 ⟨verify⟩ |
| vLLM V1 重构 blog（5.1）、SGLang API speculative execution（6.1） | 未核（非本路必查项），保留 ⟨verify⟩ |

## 3. Verification log（抽查项 × 来源 URL/引句 × 结论 × 覆盖率）

### 3.1 引擎 repo 路径（任务 item 1 必查清单，逐项）

来源统一：`https://api.github.com/repos/<org>/<repo>/git/trees/main?recursive=1`（2026-11-22 抓取，truncated=false）。

**vllm-project/vllm**

| 抽查项 | tree 证据 | 结论 |
| --- | --- | --- |
| v1/{core,engine,executor,attention,spec_decode,structured_output,kv_offload,sample,worker} | 9 目录均在 v1 top-level 清单 | ✓ 全存在 |
| v1/core/sched/ | sched/{scheduler,async_scheduler,request_queue,interface,output,utils,prefix_lm_scheduler,diffusion_scheduler}.py | ✓（5.2「scheduler.py、policy、request 队列」中 policy 非独立文件，逻辑在 scheduler/request_queue，大纲未断言 policy.py，不计 finding） |
| v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager,encoder_cache_manager}.py | 5 文件均在 core/ | ✓ |
| v1/core/{kv_cache_config_builder,kv_cache_planning}.py（5.2） | 均在 core/ | ✓ |
| v1/core/{kv_cache_interface,kv_cache_layout}.py（附录断言） | **不在 core/，在 v1/ 根** | ✗ → B-1 |
| v1/{simple_kv_offload,metrics,hisparse,kv_hints}（附录 / 5.1 / 5.3 / 5.4） | 均存在 | ✓ |
| v1/spec_decode/ 含 eagle/ngram/mtp（4.5 / 5.4） | 仅 dynamic/ + metrics.py；实在 worker/gpu/spec_decode/ | ✗ → B-2 |
| v1/attention/ backends：FlashInfer / FlashAttention / Triton（5.3） | backends/{flashinfer,flash_attn,triton_attn}.py + selector.py + mla/ | ✓ |
| v1/structured_output/ xgrammar / guidance backend（5.4） | backend_xgrammar.py / backend_guidance.py（另有 outlines / lm_format_enforcer） | ✓ |
| vllm/distributed/、custom allreduce symm_mem（5.3） | distributed/ 271 文件；device_communicators/symm_mem.py | ✓ |
| piecewise CUDA Graph / compilation（3.4 / 5.3） | vllm/compilation/{piecewise_backend,breakable_cudagraph,cudagraph_pool}.py | ✓ |
| LLM / LLMEngine / AsyncLLM / OpenAI server（5.1） | entrypoints/llm.py、v1/engine/{llm_engine,async_llm,core,core_client}.py、entrypoints/openai/api_server.py | ✓ |
| LoRA / sleep mode / KV connector（5.4 / 8.4） | vllm/lora/、docs/features/sleep_mode.md + tests、kv_transfer/kv_connector/v1/{nixl,mooncake,lmcache_connector} | ✓ |
| batch_invariant_ops（3.4） | vLLM 无此目录；model_executor/determinism/batch_invariant.py | 名称不精确 → B-10 |

**sgl-project/sglang**

| 抽查项 | tree / 文件证据 | 结论 |
| --- | --- | --- |
| srt/{managers,mem_cache,disaggregation,eplb,elastic_ep,batch_overlap,batch_invariant_ops,connector,checkpoint_engine,dllm} | 10 目录均在 srt top-level 清单 | ✓ 全存在 |
| managers/{scheduler,schedule_policy,schedule_batch,data_parallel_controller,overlap_utils}.py | 5 文件均在 managers/ | ✓ |
| managers/{io_struct,prefill_delayer,shortest_prefill_load,hisparse_coordinator,mm_schedule}.py（附录 / 6.1 / 6.2 / 9.4） | 均存在 | ✓ |
| mem_cache/radix_cache.py（6.1） | 存在（另有 pure_swa_radix_cache / unified_radix_cache） | ✓ |
| req_to_token_pool / token_to_kv_pool（6.1） | memory_pool.py:382 ReqToTokenPool；MHA/MLA/DSA TokenToKVPool 族 | ✓（类名级） |
| batch_overlap/（6.2） | operations.py / single_batch_overlap.py / two_batch_overlap.py | ✓ |
| disaggregation/ mooncake / nixl backend（6.3） | disaggregation/{mooncake,nixl}/conn.py + prefill.py / decode.py / base/conn.py | ✓ |
| connector/ = KV 传输接口（6.3） | base_connector.py = weight_iterator / pull_files（s3/redis/azure） | ✗ → B-3 |
| eplb/ / elastic_ep/（6.3） | eplb/{eplb_manager,expert_location,expert_distribution}.py；elastic_ep/{elastic_ep,expert_backup_manager}.py | ✓ |
| checkpoint_engine/（6.3 / 9.3） | checkpoint_engine_worker.py / update.py | ✓ |
| dllm/（6.3 / 9.4） | dllm/{algorithm,attention}.py 等 | ✓ |
| constrained / compilation（附录 / 6.3） | 均存在；compilation/cuda_piecewise_backend.py | ✓ |
| frontend language（6.1） | python/sglang/lang/{api,interpreter,choices}.py | ✓ |
| schedule_policy 策略名（4.2 ⟨verify⟩） | LPM / DFS_WEIGHT / FCFS / LOF / RANDOM 逐字 | ✓ 解决 |

**deepseek-ai/DeepInfer**

| 抽查项 | tree 证据 | 结论 |
| --- | --- | --- |
| deepinfer/{_C,engine,model_executor,ve,worker,npu_ops,structure_output} | 7 目录全在 | ✓ 全存在 |
| deepinfer/{distributed,hugging_face,platform.py}（附录 / 7.1） | 均在；distributed/shm_broadcast.py | ✓ |
| engine/llm_engine.py（7.1） | 存在 | ✓ |
| model_executor/layers/tilelang_ops（3.4） | 存在，cuda/ + asc/ 双平台（kv_quant / state_cache / swa_index / candidate_blocks / schedule_mtp_varlen 等） | ✓ |
| ve/ = 多模态 vision encoder（7.1） | ve/{multimodal_processor,processor/image_processor}.py | ✓ |
| npu_ops/ = Ascend（7.1） | npu_ops/{deep_select,vit_attn}.py | ✓ |
| structure_output/（7.1，注意非 structured） | compiler.py / mask.py / matcher.py | ✓ 拼写与 repo 一致 |
| examples/benchmark.py（7.3） | 存在 | ✓ |
| doc/inference_deployment_qa.md（附录） | 存在，全文读取 | ✓ |
| _C = C++ 核心 ⟨verify⟩ | _C/ 仅 .pyi stub；C++ 源码在 csrc/ | 解决（见 §2 表） |

### 3.2 DeepInfer 生产数字逐字对照（任务 item 2；来源：`https://api.github.com/repos/deepseek-ai/DeepInfer/contents/doc/inference_deployment_qa.md` raw，2026-11-22 全文读取；行号为抓取件行号）

| # | spec 数字 | doc 逐字引句 | 结论 |
| --- | --- | --- | --- |
| 1 | 155T tokens/day 单集群（§2 / 9.1） | L17: "A single cluster currently serves 155 trillion tokens per day" | ✓ |
| 2 | $38/GPU-hour = **收入**（off-peak）非成本（9.1） | L17: "with overall revenue of US$38 per GPU-hour (at the off-peak price)" | ✓ 口径正确 |
| 3 | EPLB prefill 4 冗余 / 100 步窗 / 1000 步重分配，moe_lb 1.074(EP4DP4) / 1.045(EP2DP2)（7.4） | L127: "start from 4 redundant experts per GPU (rounded up …), a 100-step statistics window, and a re-assignment every 1,000 steps. Measured moe_lb (slowest rank / mean) = 1.074 (EP4DP4) / 1.045 (EP2DP2)" | ✓ |
| 4 | EPLB decode 1 冗余（EP64 → 7 expert/GPU）/ 1000 步窗 / 10000 步重分配，moe_lb 1.090（7.4） | L131: "start from 1 redundant expert per GPU … at EP64 … 7 experts per GPU after rounding, excluding the shared expert; a 1,000-step statistics window and a re-assignment every 10,000 steps. Measured moe_lb = 1.090"；算术复核 384/64+1=7 ✓ | ✓ |
| 5 | DSpark fixed-5 verified 5.00 → dynamic 3.2–4.4；accepted 3.29 vs 3.20；+4–7% @~60 conc（7.2） | L307: "cuts the number of draft tokens verified per step by about 30% (5.00 → 3.2–4.4) … (3.29 vs. 3.20) … +4–7% throughput at the production operating point (~60 concurrent sequences per rank)"；表内 conc60=+5.4% / conc72=+6.7%，范围限定与 doc 一致 | ✓ |
| 6 | StateBlock 20×134×528 B = 1,415,040 B/seq MXFP8；128 粒度；ring+6；YOCO dropped region；ca_state/ica_state（7.2） | L223: "state_size_to_disk = 20 layers × 134 tokens × 528 B = 1,415,040 B per sequence … The 528 B/token is the MXFP8 SWA KV format"；L219: "multiples of 128 tokens … 6 extra positions the slot's ring buffer"；L217: dropped region / ca_state / ica_state；算术 20×134×528=1,415,040 ✓ | ✓（落盘口径措辞 → B-6） |
| 7 | SWA replay 末 128 token；hit cap L−134（128+6）；L1 vs L1−128；encoder/decoder replay；近乎减半 prefill（7.2） | L237: "recomputes only the most recent 128 tokens … constant, independent of context length and layer count"；L245: "at most L − 134 (128-token window + 6 tokens of speculative backtracking) … hit length L1 … otherwise it is L1 − 128"；L239/241: encoder block KV 原样复用 / decoder 末 128 只供 decode 不入 prefix cache、"nearly halving total prefill computation" | ✓ |
| 8 | 3FS 100 PiB SSD、EC 10+2、LRU GC Worker（2.4） | L187: "roughly 100 PiB of SSD storage"；L111: "10+2 erasure coding; each EC group tolerates two simultaneous node failures"；L115: GC Worker LRU watermark | ✓ |
| 9 | ~80% of perfectly balanced offline（1.1 / 7.4） | L155: "we reach roughly 80% of a perfectly balanced offline benchmark" | ✓ |
| 10 | ~400 Gbps / 8 GPU 跨 DC（8.3） | L97: "Budget roughly 400 Gbps of interconnect bandwidth per 8 GPUs"（跨 DC 问答语境，spec 归属正确；另 L168/231 的 400G VLAN 是 SWA 跨机 pull，spec 未混用） | ✓ |
| 11 | 14.4 TB/GPU 逻辑容量；20×30.73 TB SSD/服务器；≥40 GB/s 读 / 5 GB/s 写；scale out（9.1） | L105: "about 14.4 TB per GPU measured as logical size; the physical capacity required is larger"；L101: "20 × 30.73 TB SSDs … at least 40 GB/s read and 5 GB/s write … Scale out …" | ✓ |
| 12 | ~1:1 P/D、最小高效单元 64P+64D（8.2 / 9.1） | L91: "1P + 1D is deployable … roughly 1:1 … the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode" | ✓ |
| 13 | TTL agent 3d / chatbot 7d；72h 损失 0.5–1% 命中（2.2 / 9.1 / 附录） | L187: "a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate"；L211: agent 72h + chatbot 7d "captures almost all reusable cache entries" | 数字 ✓、**流量范围缺失 → B-4** |
| 14 | compute_quota_limit ns / cost(q,c) 二次模型 / 32768 预算 / all-reduce MIN（4.3 / 7.4） | L143 逐字全对（"compute_quota_limit, in nanoseconds … fitted quadratic cost model cost(q, c) … largest q … 32,768-token budget … all-reduce MIN within the DP group"） | ✓ |
| 15 | Prefill LB 4 档 <16K/16–64K/64–256K/≥256K + pull；Decode LB seq count + total KV length + dispatch + DualPath 引用；VE LB machine 随机 + fewest patches（7.4） | L137/139/147/151/157–161 逐字对；VE patch 式 (width/patch_size)×(height/patch_size)，spec 的 /14 由 config patch_size=14 代入（§3.3 ✓） | ✓ |
| 16 | FlashMLA open vs internal：decode 慢 5–10%（SASS）、e2e ≤5%、indexer/mega 略差（7.3 / 3.1） | L289–291: "FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower … at most 5% apart … the indexer and mega series differ somewhat" | ✓ |
| 17 | 主 KV per-16 e4m3「类 NVFP4 无 global scale」/ indexer per-32 ue8m0 MXFP4 / routed expert MXFP4（2.3 / 3.2 / 7.3） | L271–275 逐字全对 | ✓ |
| 18 | Mega MoE scale-up 通用解 / EP>32 递减（带宽不再瓶颈 → 通信同步 + NVLink 多 job 干扰）（3.3 / 7.4） | L285 / L293–297 逐字对 | ✓ |
| 19 | Engram ENP vocab 分片 / host memory + one-sided RDMA read / scales NVLink / prefill ENP8 / decode ENP=DP/EP / NPU 无 offload（7.4） | L83–87 逐字对 | ✓ |
| 20 | autoscaling（prefill non-cached tokens / decode running reqs）、grace 1 min + KV 转发、quota>0.8 禁 scale-in、EP failover 健康 GPU 重定 EP、重启分钟级、训推共调度（8.5） | L251/253/257/261/265/119 逐字对 | ✓ |
| 21 | TP/CP 不用 rationale；balanced CP；#DP=#EP；prefill 最小 EP/DP、decode 最大 EP/DP（7.4 / 8.1） | L47–53/67–69/33；README L12 "Attention DP / balanced CP" | ✓ |
| 22 | Entrypoint CHECK hash-chain metadata 匹配 / 专用集群任何机命中任何前缀 / GDR swap-in 不占 SM / copy engine 细粒度差 / write-back + read validation / 版本元组 / CXL L2 收益有限（2.2 / 2.4 / 8.4） | L191/175/195/199/203–207 逐字对 | ✓ |
| 23 | async scheduling 仿 SGLang / NanoFlow；SO 专用机异步编译 + automaton mask；deepseek-recipe Rust 多协议 / 注入防护 / 分段并行 BPE / 流式 KMP+DSML（6.2 / 9.4 / 7.4 / 1.4） | L334/344–348/354–372 逐字对 | ✓ |
| 24 | DSpark Confidence Head / 排序选前缀子集 / draft 5 / TileLang cumprod+sort+scatter→pinned buffer / async 只定总预算（7.2） | L303/322–324；README L24（default draft 5） | ✓ |
| 25 | release 边界三项 + 内部 infra 耦合原因归属（7.1） | README L38–42 列三项；耦合原因在 doc L21 非 README | 边界 ✓、归属 → B-5 |

媒体来源检查：spec §2 / §4.1 明确「媒体转述一律不作事实来源，只作 [C] 线索」；上表全部数字均可在官方 repo doc 逐字定位，**未发现 media-only 支撑的数字**。知乎专栏 URL（`https://zhuanlan.zhihu.com/p/2095560295930508953`）在 doc L21 逐字存在 ✓；知乎正文 403 未读，但 doc 自述「This Q&A builds on that blog post」，本路全部数字以 doc 为一手即足。

### 3.3 V4.1-Flash config 字段（任务 item 3；来源：`https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json`，2026-11-22 全文读取）

| spec 声称字段 | config 实际值 | 结论 |
| --- | --- | --- |
| architectures DeepseekV41ForCausalLM（附录） | `["DeepseekV41ForCausalLM"]`，model_type deepseek_v41 | ✓ |
| 40L + 3 nextn（7.1 / 4.5） | text_config.num_hidden_layers 40；num_nextn_predict_layers 3 | ✓ |
| 384 routed + 1 shared top-6（7.1） | n_routed_experts 384；n_shared_experts 1；num_experts_per_tok 6 | ✓ |
| kv_heads 1 / head_dim 512（1.2 / 7.1） | num_key_value_heads 1；head_dim 512 | ✓ |
| q_lora_rank 1280 / o_lora_rank 1024（7.1） | 1280 / 1024（另 o_groups 8 ✓ 附录） | ✓ |
| qk_rope 64（1.2 / 附录） | qk_rope_head_dim 64 | ✓ |
| sliding_window 128（7.1） | 128 | ✓ |
| index_topk 512 / index_n_heads 32 / index_head_dim 128（3.1 / 附录） | 512 / 32 / 128 | ✓ |
| fp8 block [32,32] ue8m0 expert fp4（3.2 / 附录） | quantization_config: quant_method fp8、weight_block_size [32,32]、scale_fmt ue8m0、expert_dtype fp4 | ✓ |
| compress_ratios 43 项（7.1 / 附录） | 数组 43 项（0,0 + 2×18 + 1×20 + 0,0,0） | ✓ |
| dspark_* 字段族：小 MoE / markov_rank / noise_token / target_layer_ids（7.2） | dspark_n_routed_experts 128 / num_experts_per_tok 3、dspark_markov_rank 256、dspark_noise_token_id 128799、dspark_target_layer_ids [37,38,39]、dspark_block_size 5 | ✓ |
| kv_source_layer_ids（7.2 CED） | [2,8,14,20] 存在；与 doc「末 encoder 层」表述张力 → B-9 advisory | ✓ 字段存在 |
| engram_* / hc_* 字段族（7.1 / 附录） | engram_layer_ids [1,14] 等 8 字段；hc_mult 4 / hc_sinkhorn_iters 20 / hc_eps | ✓ |
| vision_config 32L/1024h/patch14/downsample3、image_token_id 129264（9.4 / 附录） | 32 / 1024 / 14 / 3；129264 | ✓（7.4 VE LB 的 /14 由此代入） |
| vocab 129280 / hidden 5120 / 64 heads / sqrtsoftplus / noaux_tc / max_position 1048576 YaRN ×16 from 65536（附录） | 全部逐字对 | ✓ |

**边界检查（§2 / §4.2）**：P5–P7 大纲对 V4.1 只做「config 字段名 + 数值 [W]」与「doc 官方口径」引用；7.1 显式「只列 serving 相关字段 + 一句 recap + 指针，不推导」，7.2 显式「模型侧不推导」，4.5 MTP「模型侧目标不展开」；7.2 关键推导（DSpark 期望吞吐最优 / SWA replay 恒定成本 / hit policy 正确性）均为 serving 侧账，1.2 的 κ 代入是 serving 带宽账——**未发现模型侧推导越界**。CED / Engram / DSpark draft / compress_ratios / hyper-connections 的形式化均未展开 ✓。hf-mirror 渠道已在 spec risk 7 声明；huggingface.co 本机仍 000，官方渠道复核留 W0（本机无法完成，如实记录）。

### 3.4 证据分级（任务 item 4）

| 检查项 | 结论 |
| --- | --- |
| DeepInfer / TRT-LLM / LMDeploy / FlashMLA / DeepGEMM / DeepEP / EPLB / 3FS / DeepSelect / TileKernels / deepseek-recipe / NIXL / Dynamo / LMCache / guidance / Marlin = 教学参照 + 官方实现（无 paper） | ✓ §4.1 列表与任务口径一致；7.1–7.4 / 3.3 / 9.2 / 2.4 / 8.4 各章锚点行逐一核对无误 |
| DeepInfer 生产数字挂「教学参照 + 官方实现」+ 官方披露注 + 核实日期，不升 [A] 不降 [C] | ✓ §2 / §4.1 末条口径正确 |
| vLLM / SGLang / FlashInfer / TileLang = [A]（paper + official code） | ✓ §4.1 例表 + 5.x / 6.x / 3.1 / 3.4 / 7.3 锚点行；repo 全部核实存在 |
| Mooncake = [A]（2407.00079 + kvcache-ai/Mooncake） | §4.1 ✓、2.4 / 8.2 ✓；**8.3 锚点把 Mooncake transfer engine 并入 [教学参照] → B-8** |
| Orca [B]（OSDI'22 有 paper 无官方代码）、Leviathan / Chen [B]、config [W] | ✓ 口径自洽 |
| 媒体转述只作 [C] 线索 | ✓ §2 明文；本路未发现 media-only 数字 |

### 3.5 repo 存在性（附录「25 repo 预核实」抽查复核）

API full_name 或 github.com 状态码（2026-11-22）：vllm-project/vllm ✓、sgl-project/sglang ✓、deepseek-ai/DeepInfer ✓、deepseek-ai/{FlashMLA,DeepGEMM,DeepEP,EPLB,3FS,DeepSelect,TileKernels,deepseek-recipe} ✓（8/8）、tile-ai/tilelang ✓、flashinfer-ai/flashinfer ✓、NVIDIA/TensorRT-LLM ✓、InternLM/lmdeploy ✓、ai-dynamo/{nixl,dynamo} ✓、LMCache/LMCache ✓、kvcache-ai/Mooncake ✓、IST-DASLab/marlin ✓、guidance-ai/guidance ✓、mlc-ai/xgrammar ✓、dottxt-ai/outlines ✓、SafeAILab/EAGLE ✓、Dao-AILab/flash-attention ✓。**25/25 存在**。更名：outlines-dev → dottxt-ai 301 redirect ✓；MoonshotAI/Mooncake 旧 URL 404（现名正确）→ B-7。

### 3.6 覆盖率与限制（如实声明）

- **覆盖**：任务 item 1 必查路径 100% 逐项核对（vLLM 9+6、SGLang 10+5、DeepInfer 7）；item 2 点名 11 组生产数字 100% 逐字对照（另加 14 组关联声明，共 25 行）；item 3 config 字段 100% 对照（含附录全字段）；item 4 分级全表核对；P5–P7 大纲其余路径级声明（5.1–5.4 / 6.1–6.3 / 7.1–7.4 含附录扩展清单）全部对 tree 核对。
- **深度限制**：vLLM / SGLang 为路径级 +  selective 文件级核查（读 4 个 SGLang 文件内容：base_connector.py、disaggregation/base/conn.py、memory_pool.py 类表、schedule_policy.py 策略枚举）；scheduler.py / schedule_batch.py / overlap_utils.py 的函数级语义（future token resolve、ScheduleBatch 类）未逐行读——API 限额耗尽且 raw 域不可达，留 W3 research pass（spec 本就要求 file+function 级锚定在写作波完成）。
- **未核项**（非本路必查，保留 ⟨verify⟩）：arXiv ID ↔ 标题（路线 A）、DistServe goodput 数学（路线 C）、vLLM V1 blog、SGLang API speculative execution、DualPath 公开性、硬件 datasheet、batch-invariance 博客 URL、OCP MX URL、知乎正文（403）、HF 官方渠道（000）。
- **观察项（不计 finding，供 W4 写作注意）**：① QA doc EC 读路径写「read issues requests to all **11** storage nodes … any 10 responses」而 EC 为 10+2（12 节点）——doc 内部表述待与 3FS repo 对照，spec 未引用此细节，章节若引用需先核；② spec 2.4「计算节点故障率高一个量级」doc 原文为「**more than** an order of magnitude higher」，章节宜写「高一个量级以上」。

## 4. 结论

- **S1 = 0，S2 = 4（B-1…B-4），S3 = 6（B-5…B-10）**。
- 无致命事实错误：全部生产数字官方一手逐字命中、全部 config 字段命中、全部必查 repo 路径存在、证据分级总体正确、V4.1 边界纪律在 P5–P7 无越界。
- S2 四条均为可快速修复的锚点/口径问题：vLLM 两处目录迁移（B-1 / B-2）、SGLang connector 语义错挂（B-3）、TTL 损失数字缺 chatbot 限定（B-4）。**须 W3（P5/P6）与 W4（P7）写作前修复**；B-4 同时影响 W1 的 2.2 与 W6 的 9.1，建议随本次 spec 修订一并关闭。
- S3 六条按 §6.4 逐条判定（修复 / 接受 / 驳回）；B-9 为预防性 advisory，建议采纳以免 W4 章节触发模型侧越界 S2。
- 本路 ⟨verify⟩ 解决 10 项（§2 表），可直接写入 spec 修订与 references.md 骨架。
