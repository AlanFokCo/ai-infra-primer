# W6 调研笔记：P9 生产全流程 事实源

调研日期：2026-11-24（W6 research pass）。本文件是 W6 四章（9.1–9.4）写作的事实源之一；写作 agent 不做额外调研，数值与代码锚点一律以本文件与 `inference/references.md` 为准。引用 key 白名单 = references.md P9 全部条目（tensorrt-llm / lmdeploy / hybridflow）+ 被引章节含 W6 章号的跨波锚点（deepinfer-doc / deepinfer-repo / distserve / vllm-repo / sglang-repo / v41-flash-config）。**边界纪律（spec §2/§4.2）**：9.1 容量成本算例强制重推导（§6.3 W6 子集）；9.2 引擎对照只 recap 8.4（PD 系统）+ P5/P6/P7 定位指针，增量限定选型决策表（防双展开 C-S3-4）；9.3 RL rollout 训练侧 recap training 8.5/8.4（引擎侧增量 = vLLM sleep/wake、SGLang checkpoint_engine、veRL/OpenRLHF 引擎接口）；9.4 多模态 VE recap 7.4/8.4，diffusion LLM 只展望一句。**生产数字挂 deepinfer-doc 官方披露 + doc 行号 + 核实日期**；$38/GPU-hour 是**收入**（off-peak）非成本；V4.1 模型侧只引 config 字段 [W] + recap + 指针不推导。

**版本锚定总口径**：本轮代码锚点 = 2026-11-24 当日 GitHub **contents API raw + git trees API**（codeload 大 tarball 本轮本机截断，沿 W5 通道）；DeepInfer doc / V4.1 config **byte-identical 复用 W4**（main@a679bb86 / config md5 e1e4964c…）。repo HEAD：vllm **6aeedab553dd**（2026-11-23）/ sglang **ac0243723b0d**（2026-11-23）/ verl **7b6c9cc0e19d**（2026-11-23）/ OpenRLHF **26494490bc23**（2026-11-03）/ lmdeploy **517cb6d73d88**（2026-11-23）/ TRT-LLM **a41224a3dee2**（2026-11-23）/ DeepInfer **a679bb86ce36**（2026-11-03，与 W4 同）。**vLLM/SGLang HEAD 自 W3 已移动 → sleep/wake 与 engine.py 行号有漂移（见漂移表），本轮 file:line 以本轮 contents 为准。**

## 访问路径记录（后续 research pass 复用）

- **arXiv：https export API 批量 200**（4 ID 一次，12,237 B）：2508.15601v2 / 2409.19256v2 / 2401.09670v3 / 2405.11143v6，ID↔标题↔第一作者全中。论文全文 `https://arxiv.org/html/2508.15601v2`（275,770 B）。http export 仍 000 须 https（沿 W5）。
- **GitHub：api.github.com 200 可达**（core 限额本轮用量 ~40：rate_limit / commits×7 / contents×~25 / trees×3 / 目录 listing×6）；**codeload 大 tarball 本轮截断**（vllm/sglang/verl 下载 ~400KB/min 且固定字节截断，verl.tar.gz 1.85MB unexpected EOF；OpenRLHF 432KB 小 repo 完整）→ **大 repo 全程 trees + contents API raw**（沿 W5 通道）。verl tree 1,553 entry / lmdeploy tree 2,423 entry 均 truncated=False；TRT-LLM 复用 W5 tree（25,568 entry）。raw.githubusercontent.com 仍 000。
- **hf-mirror**：`https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json` 首次 307（redirect stub）→ **加 -L 重试 200 / 3,311 B**，md5 e1e4964ca281289b9bc74de57a7ebcf5 与 W4 byte-identical（第五次重取）。**huggingface.co 本机仍 000**（官方渠道复核连续五波不可达，如实留痕）。
- **DeepInfer doc**：contents API raw 200 / 34,526 B / 372 行，md5 c805de234f944d526ee90185a93fba95 与 W4 byte-identical。
- 证据存档：/tmp/iw6-evidence/（DeepInfer doc API raw、v41 config、arXiv 批量 XML + LMDeploy HTML/txt、TRT-LLM README/overview/supported-hardware/legacy-workflow、verl/lmdeploy tree JSON、vLLM cumem/gpu_worker/llm/encoder_cache_manager/pooler、SGLang weight_updater/ckpt-update/ckpt-worker/engine/mm_schedule/dllm/server_args/serving_embedding、OpenRLHF 解压、verl sglang/vllm_rollout + async docs）；脚本 /tmp/iw6-scripts/{fetch_contents.sh,parse_refs_w6.py}。复用 /tmp/iw4-evidence/DeepInfer-x（ve/ + structure_output/）、/tmp/iw5-evidence/trtllm-tree.json。

## ⟨verify⟩ 销项总表（W6 范围）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 9.3 veRL SGLang backend（spec 附录留 W6） | **销**：verl-project/verl `workers/rollout/sglang_rollout/sglang_rollout.py` **ServerAdapter L128** 存在（+ async_sglang_server.py / http_server_engine.py / sglang_pd_replica.py / delta_loader.py）；**veRL 三引擎 rollout backend = vllm / sglang / trtllm**（trtllm_rollout/ 亦在树）；update_weights L305 tensor buckets CUDA IPC（参考 THUDM/slime）+ release L288 / resume L276 → release/resume_memory_occupation(tags) | verl tree 7b6c9cc0 + sglang_rollout.py contents |
| 9.3 OpenRLHF 引擎接口（spec 附录留 W6） | **销**：OpenRLHF **vLLM-only**（全 repo grep 无 sglang）；ppo_actor.py **broadcast_to_vllm L409**（NCCL broadcast / ray.util.collective + **CUDA IPC _handle_cuda_ipc L439**，ZeRO-3 GatheredParameters L472）；vllm_engine.py **update_weight L115 / update_weight_cuda_ipc L121 / sleep L136 / wake_up L139**。**经 training 8.5 §7 纯文字指引锚定（spec 9.3），不入 inference 独立条目** | OpenRLHF tarball 26494490 解压 grep |
| 9.2 TRT-LLM 编译器路线（spec 9.2 大纲口径） | **销 + 漂移**：当前 main = **PyTorch-native**（README L290「Architected on PyTorch」+ PyExecutor）；**legacy TensorRT backend 已移除**（legacy/architecture/workflow.md 逐字「The legacy TensorRT backend has been removed and is no longer supported」）——spec「TRT 编译器路线」须历史化为 legacy，当前是 PyTorch-native + 手写 kernel 混合路线 | contents API README + overview.md + workflow.md |
| 9.2 LMDeploy EuroSys 2027 / persistent batch / W4A16 | **销**：EuroSys 2027 accepted = **README L25 [2026/08] news**（arXiv metadata 无 journal_ref）；persistent batch = continuous batching（turbomind.md N slots）；W4A16 AWQ（w4a16.md lmdeploy lite auto_awq）；论文 v2 摘要双管线四技术 61%/156% | contents API README + turbomind.md + w4a16.md + 论文 HTML |
| 9.3 vLLM sleep/wake 路径（v1/worker/ or engine/） | **销（fresh）**：device_allocator/cumem.py **CuMemAllocator L62**（sleep L261 / wake_up L364 / use_memory_pool L394）；v1/worker/gpu_worker.py **Worker.sleep L361 / wake_up L402**（W3 L351/L389 漂移）；entrypoints/llm.py **sleep L853 / wake_up L897 / collective_rpc L587**；distributed/weight_transfer/ 六引擎 | contents API |
| 9.3 SGLang checkpoint_engine / update_weights（managers/scheduler_components/） | **销（fresh）**：checkpoint_engine/update.py **update_weights L144**；checkpoint_engine_worker.py **SGLangCheckpointEngineWorkerExtension L39**；scheduler_components/weight_updater.py **SchedulerWeightUpdaterManager L131**；engine.py **release_memory_occupation L1902 / resume L1908 / update_weights_from_ipc L1793**（W3 L1899/L1905/L1790 漂移 +3） | contents API |
| 9.4 SGLang mm_schedule / vLLM encoder_cache_manager（W3 已核再确认） | **维持**：mm_schedule.py get_embedding_chunk L33 / _get_chunked_prefill_embedding L593；encoder_cache_manager.py EncoderCacheManager L18（行号全同 W3） | contents API |
| 9.4 dllm 形态（W3 已核再确认） | **维持 + 结构注**：dllm/algorithm/{base.py DllmAlgorithm L30 step L83/run L98/_run_fdfo L149, low_confidence, joint_threshold, joint_threshold_indel, gemma4_renoise} + config.py DllmConfig L8（block_size/mask_id/first_done_first_out_mode）；W3 描述算法集一致，目录已重组为 algorithm/ + mixin/ 子目录 | contents API + 目录 listing |

## 语义纠正 / 命名注（research pass 新发现，writer 须知）

1. **$38/GPU-hour 是收入非成本（doc L17 逐字）**：「A single cluster currently serves 155 trillion tokens per day, with **overall revenue of US$38 per GPU-hour (at the off-peak price)**」。9.1 成本模型推导 USD/M tokens 用「USD/GPU-hour × GPU 数 ÷ tokens/h」通式时，**若代入 DeepInfer 38 须显式标注这是收入口径（off-peak 价）非成本**，不可混用。且 **155T tokens/day 的 token 计数口径（输入/输出/cache 命中计否）doc 未定义**、**单集群 GPU 数 doc 未披露** → 代入算例须显式声明假设（C-S3-7）。
2. **TRT-LLM 当前 = PyTorch-native，legacy TRT 编译器已移除**（⟨verify⟩ 表）：9.2 写「编译 vs 手写 kernel 路线对照」时，TRT-LLM 的历史身份是 TRT 编译器路线，但**当前 main 是 PyTorch-native（PyExecutor + _torch/ native PyTorch 模型 + 手写 kernel）**，与 vLLM/SGLang 路线趋同；legacy convert_checkpoint + trtllm-build engine-build 两步路线已 removed（workflow.md 保留仅供 cross-reference）。**必须给当前形态 + 历史化注**，勿按 spec 大纲字面写「TRT-LLM = TRT 编译器路线」。
3. **vLLM/SGLang sleep/wake 行号漂移**：HEAD 自 W3 移动，gpu_worker.py Worker.sleep L351→**L361** / wake_up L389→**L402**（Worker 类 L233→L243）；SGLang engine.py release L1899→**L1902** / resume L1905→**L1908** / update_weights_from_ipc L1790→**L1793**。cumem.py / llm.py / weight_updater.py / ckpt-update.py / mm_schedule.py / encoder_cache_manager.py 行号**全同 W3**（未漂移）。evaluator 复核以本轮 file:line 为准。
4. **vLLM sleep level 语义**（**closeout errata 2026-11-24**：本条初版口径「level 1 丢弃全部（weights + KV）」被 fresh HEAD fetch @6aeedab553dd 推翻——gpu_worker.py L368-371 level 1 discardable=∅ + sleep_mode_backend.py L58「nothing at level 1」+ cumem.py L292 offload 判据 + llm.py L864「Offload model weights to CPU, discard KV cache」四文件互证，evaluator P9 审查 S1-1 + fix 复审双通道 byte-identical 确认，见 docs/reviews/inference-w6-fix-review.md §1；初版并误标「(discardable parameters)」于保留集、LoRA 判据误挂 vllm_rollout，同轮纠正）：Worker.sleep(level)——**level 1 = 权重 offload 到 CPU（host 备份，wake_up 自动恢复、无需重新注入）+ 仅 KV 丢弃**（discardable=∅）；**level 2 = 丢弃被改写集（discardable = 全参数 − sleep_preserve_parameter_names 声明的 preserved 集，无 host 备份、wake_up 后空显存由权重更新通道填回）、非改写集（preserved）offload 到 CPU 恢复**；**KV 两 level 均丢弃**（kv_cache 池永不备份）。经 sleep_mode_backend.suspend(discardable)；wake_up(tags) 按 tags（weights/kv_cache deferrable）恢复 + 重建 kv_transfer group。veRL vllm_rollout sleep_level 由 config.layered_summon 判（开启取 level 1，L125-129 带 memory overflow 警告；否则 VLLM_SLEEP_LEVEL 常量）；「LoRA adapter 模式 level 1 只释 KV」是 **sglang_rollout** 判据（L206，勿混两 backend）。SGLang 对照 = release_memory_occupation(tags=[kv_cache,weights]) / resume_memory_occupation(tags)（tags 分池）。errata 前本条不可作 level 语义事实源。
5. **veRL fully_async / one_step_off 已迁 verl-recipe**：docs/advance/{fully_async,one_step_off}.md 现为 stub（「This implementation has moved to verl-recipe … pinned to verl v0.9.1 and use its V0 trainer」）；**mainline 异步训练 = V1 async trainer**（v1_async_trainer.md，colocate_async / separate_async）。9.3 写 async rollout 以 V1 为准 + 迁移注。
6. **veRL partial rollout 是 V1 async client 行为非配置开关**：abort 时保留 tokens+logprobs → LB 重试续跑 → 续跑用更新模型版本 → 保留前缀 KV 重建后继续 decode。代价 = 额外 prefill + 轨迹内策略版本变化。staleness 控制 = max_off_policy_threshold（默认 8 模型版本）/ strategy drop|wait（写作时竖线改文字）。
7. **OpenRLHF vLLM-only，veRL 三引擎**：9.3 引擎接口对照——veRL = vllm/sglang/trtllm 三 backend；OpenRLHF = vLLM-only（无 sglang）。这是选型论断素材。OpenRLHF 事实经 training 8.5 §7 指引（spec 9.3 锚点），**不在 inference 章直接引 arXiv:2405.11143**（该 ID 在 training/references.md，不在 inference/references.md，直接引会 lint FAIL）。
8. **vLLM pooler 已模块化**：旧 model_executor/layers/pooler.py 不存在（404）→ **pooler/ 目录**（abstract.py Pooler ABC L16 + seqwise/methods.py CLSPool L38/LastPool L51/MeanPool L61 + seqwise/poolers.py SequencePooler L44 + tokwise/ + special.py）。**CLS/MEAN pooling 不支持 partial prefill（chunked prefill）**——embedding prefill-only 但 CLS/MEAN 须整段 prefill。reranker late interaction 在 v1/pool/late_interaction.py（MaxSim）。
9. **SGLang embedding/reranker 端点**：entrypoints/openai/serving_{embedding,rerank,classify,score}.py 存在；serving_embedding.py 用 EmbeddingReqInput（io_struct）+ _handle_non_streaming_request（无 streaming = prefill-only 单次前向）。is_embedding flag 不在当前 server_args.py（已模块化，未逐一定位）——9.4 用端点存在性 + EmbeddingReqInput 锚定即可，不必引 is_embedding 字段。
10. **DeepInfer VE image patch 账代码级**：ve/processor/image_processor.py **downsampled_seqlens L81** = `((n_vit_w+r-1)//r)*((n_vit_h+r-1)//r)`（两维分别向上取整）；**_layout_from_pixels L87** = n_vit_h=best_height//patch_size, n_vit_w=best_width//patch_size, n_llm_h=ceil(n_vit_h/downsample_ratio), n_llm_w=ceil(n_vit_w/downsample_ratio)。**ViT 算 (w/14)(h/14) patches，LLM 见 ceil(n_vit/3)×ceil(n_vit/3) tokens**（downsample_ratio 3）。max_image_tokens 1024 上限。
11. **LMDeploy persistent batch ≠ 新机制**：turbomind.md 逐字「You may recognize this feature as 'continuous batching' in other repos」——persistent batch 是 continuous batching 的 LMDeploy 命名（N 预配置 slots，lifetime 跨整个 serving 进程）；KV manager LRU 驱逐转 token IDs、miss 重解码 = recompute 式（同 vLLM V1 recompute-only 抢占族，4.2 回链）。9.2 勿把 persistent batch 当独立新调度机制。
12. **存储 EC 张力维持 W1/W4 裁定**：doc L111 EC 10+2 / L113「all 11 storage nodes … any 10 responses」内部张力（10+2=12 节点 vs 读 11）维持；开源 3FS repo = CRAQ 链式复制无 EC，EC 属生产官方披露。9.1 存储 sizing 用 EC 10+2 → 物理 ×12/10，引用时区分口径。

## 版本漂移总表（W6 fresh vs W3/W4/W5 / spec 大纲）

| # | 漂移 | 写作口径 |
| --- | --- | --- |
| 1 | TRT-LLM legacy TensorRT backend 移除 → PyTorch-native（语义纠正 2） | 9.2 当前 PyTorch-native + 历史化注；勿按 spec 字面「TRT 编译器路线」 |
| 2 | vLLM gpu_worker sleep L351→L361 / wake_up L389→L402（语义纠正 3） | 9.3 按本轮 file:line |
| 3 | SGLang engine.py release L1899→L1902 / resume L1905→L1908 / update_weights_from_ipc L1790→L1793 | 9.3 按本轮 file:line |
| 4 | veRL fully_async/one_step_off 迁 verl-recipe（语义纠正 5） | 9.3 async 以 V1 async trainer 为准 + 迁移注 |
| 5 | vLLM pooler.py → pooler/ 目录（语义纠正 8） | 9.4 按 pooler/abstract.py + seqwise/methods.py |
| 6 | SGLang dllm/ 重组为 algorithm/ + mixin/ 子目录（⟨verify⟩ 表） | 9.4 展望一句按当前结构，算法集同 W3 |
| 7 | vLLM weight_transfer/ 新增 sparse_nccl_engine（W3 五引擎 → 六引擎） | 9.3 一句「目录已扩展」+ 日期 |
| 8 | DeepInfer doc / V4.1 config byte-identical W4 | 9.1/9.4 doc 行号 / config 字段照旧 |

## 9.1 容量规划与成本账【重】

**SLO 驱动 sizing 推导（强制重推导 §6.3 W6 子集）**：
- 到达率 λ（**章首声明「本章 λ = 到达率，非 GAE 参数」**§4.3）+ 长度分布 → prefill 算力需求 → P 实例数 n_P；decode 带宽需求 + 并发 → D 实例数 n_D。计数用 n_P/n_D（§4.3 禁裸 P/D/G）。
- **M/M/1 排队 recap 1.3 + 指引**（已交付锚）：逗留 W_soj=1/(μ−λ)、纯排队 W_q=ρ/(μ−λ)、ρ→1 延迟按 1/(1−ρ) 爆炸；Kingman VUT 近似 W_q（方差因子 (c_a²+c_s²)/2）；**余量规则 ρ≤q/(q+E[S])**（q=排队延迟预算）。1.3 提点：continuous batching 的 μ 状态依赖，M/M/1 是直觉基线非精确 serving 模型，生产用实测吞吐-延迟曲线 + Kingman 修正。
- **goodput 约束下冗余**：per-GPU goodput + SLO attainment 目标口径沿用 1.1/8.2（DistServe 定义要件，不得写成「达标请求裸吞吐」）。
- **8.2 xPyD 配比 recap**：n_P≈λ·2N·T_in/(F_P·ρ_P)、n_D≈C·κ·L·T̄/(TPOT·BW_D·ρ_D)（C≈λ·T_out·TPOT Little's law）；配比负载 + 硬件依赖。

**P/D GPU 配比（官方披露 doc L91 逐字）**：「1P + 1D is deployable. **In practice the GPU ratio between P and D is roughly 1:1.** Since the recommended decode configuration is EP64 or above, **the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.**」~1:1 = 生产流量经验配比（非最优定值）；最小高效单元 64P+64D（EP64 最小 decode 域量子化）；最优配比随输入/输出长度比与硬件 F/BW 比变化（8.2 推导）。

**成本模型（关键推导）**：USD/M tokens = (USD/GPU-hour × n_GPU) ÷ (tokens/h ÷ 1e6)。**分母必须是 tokens/h 速率与小时价对齐**；PD 分离时 n_GPU = n_P + n_D、两池可异价。**货币一律 USD 或 \$ 转义，禁裸 \$ 配对吞文本**（§4.3，C-S3-3）。
- **DeepInfer 38 USD/GPU-hour = 收入（off-peak）非成本**（doc L17，语义纠正 1）。155T tokens/day 单集群 = 6.458×10¹² tokens/h。**集群 GPU 数 G 未披露** → 收入 USD/M tokens = 38×G/(6.458×10⁶) = 5.884×10⁻⁶×G，代入算例须显式声明 G 假设。token 计数口径 doc 未定义须声明。

**存储 sizing（官方披露 doc 行号）**：
- L101：「We recommend **20 × 30.73 TB SSDs per KV cache storage server**, with each server providing **at least 40 GB/s read and 5 GB/s write** throughput and enough total capacity to **retain 24 hours of data**. **Scale out** the number of storage servers with the actual size of the inference deployment.」
- L105：「In practice, about **14.4 TB per GPU measured as logical size**; the physical capacity required is larger than that.」
- L111/L113/L115：3FS **EC 10+2** 容两节点 / 读 11 节点 any 10（张力注）/ GC Worker LRU watermark。L187：3FS **~100 PiB** SSD。
- **算术锚（本轮独立重算）**：raw/server = 20×30.73 = **614.6 TB**；EC 物理/GPU = 14.4×12/10 = **17.28 TB**；usable/server = 614.6×10/12 = **512.2 TB**；**GPUs/server ≈ 614.6/17.28 = 512.2/14.4 ≈ 35.6**（raw/物理 与 usable/逻辑 两口径同值）。100 PiB = 112.6 decimal PB → servers ≈183（若 100PiB=raw）或 ≈220（若 =usable/逻辑）——**口径歧义须声明**。**TB（十进制）/ PiB（二进制）换算注明**（§4.3）。

**缓存命中 → 成本节省（关键推导）**：命中率 → prefill 节省 → GPU 数节省账（命中 token 免 prefill 算力 → n_P 降）。**TTL vs 命中率（2.2 回链，官方披露）**：
- doc L211：「a TTL of **72 hours for agent traffic and 7 days for chatbot traffic captures almost all reusable cache entries**. The 72-hour TTL for agent traffic also **spans a weekend**, allowing requests on subsequent workdays to reuse the cache.」
- doc L187：「Empirically, **a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate**」——**chatbot 缩至 3d 的反事实口径，非生产损失**（生产 agent 72h / chatbot 7d 近乎无损）。chatbot 命中长尾重（少量 KV 读极多次）。引用必须带「chatbot 反事实」限定（B-4/C-S2-4）。

**workload mix**：agent vs chatbot 长度分布与命中形态（1.1 Azure trace / ShareGPT 归属 recap）。**autoscaling recap 8.5**（prefill non-cached tokens / decode running reqs，doc L251）。

**全数字容量算例（章脊口径锁定，算到底）**：给定 QPS/长度分布/SLO → P/D GPU 数 + 存储服务器数 + USD/M tokens。强制重推导按上述锚独立重算。来源：deepinfer-doc、distserve、1.3/2.2/8.2/8.5（纯文字指引）。

## 9.2 其他推理引擎对照与选型（短篇）

**两层纪律（§4.2）**：三主引擎（vLLM/SGLang/DeepInfer）定位 recap P5/P6/P7 指针；disagg recap 8.4 + 一句选型定位（**防双展开 C-S3-4**）；增量 = 选型决策表。

**TensorRT-LLM（教学参照，main a41224a3）**：
- **当前 = PyTorch-native**（README L290「Architected on PyTorch」+ high-level Python LLM API；overview.md PyExecutor(Worker) 每 rank 后台循环 = Scheduler/KVCacheManager/ModelEngine/Sampler；模型 native PyTorch 定义 _torch/models/）。**legacy TRT 编译器路线已移除**（workflow.md 逐字，语义纠正 2）——历史身份 convert_checkpoint + trtllm-build 两步 engine-build，现保留仅供 cross-reference。
- **inflight batching**（paged-attention-ifb-scheduler.md，W5 销）= continuous/iteration-level batching、packed no-padding、context 序列排 generation 前。
- **disagg**（recap 8.4）：当前 Python transceiver KvCacheTransceiverV2 + backend NIXL + KV manager V2（W5 销）；9.2 只一句选型定位。
- **Blackwell 支持**（supported-hardware.md）：B200/GB200/B300/GB300/DGX Spark + Hopper H100/H200/GH200 + Ada L20/L40/L40S + Ampere A100。
- Overlap Scheduler 默认开（docs/source/developer-guide/overview.md，GPU 第 n+1 步与 CPU 第 n 步重叠——doc prose L63-67 用 n 标号、代码注释 L77 用 n-1 标号，索引原点不同同一管线；closeout 与章 / refs 同口径对齐）+ CUDA Graph padding（至多 22% e2e 吞吐）。
- **选型定位 = NVIDIA 专属 + PyTorch-native 编译/手写 kernel 混合 + Dynamo/Triton 生态**。

**LMDeploy（[A] 2508.15601 + InternLM/lmdeploy，main 517cb6d7）**：
- **TurboMind 引擎**（turbomind.md）= 基于 NVIDIA FasterTransformer；**persistent batch = continuous batching**（N 预配置 slots，请求有空槽即加入、生成完释放复用、cache-hit 历史免重解码、batch 自动增减）；KV cache manager memory-pool + LRU「cache of KV caches」（驱逐转 token IDs、miss FMHA 重解码 = recompute 式）；cutlass FMHA Q/K 不等长 + INT8 KV。
- **W4A16 AWQ**（w4a16.md）：TurboMind 支持 AWQ/GPTQ INT4 推理（量化模块仅 AWQ）；`lmdeploy lite auto_awq`；V100(sm70)/Turing/Ampere/Ada。
- **InternLM 生态**：internlm2_5-7b-chat 示例 / InternVL2 / InternLM-XComposer2.5 / internlm2-20b 16+ RPS 1.8× vLLM。
- **论文 v2（EuroSys 2027，README L25 accepted）**：TurboMind 双 hardware-aware 混合精度管线（GEMM offline weight packing + online acceleration / attention 不同 Q/K/V 精度）+ 四技术（weight packing / adaptive head alignment / instruction-level parallelism / KV memory loading pipeline）；16 LLM × 4 GPU；至多 61% 更低延迟（均值 30%）/ 156% 更高吞吐（均值 58%）vs 现有混合精度框架。
- 双引擎 TurboMind + PyTorchEngine（纯 Python，Ascend）；MXFP4 NVIDIA V100+（gpt-oss H800 1.5× vLLM）；DeepSeek PD disagg 经 DLSlime+Mooncake。
- **选型定位 = InternLM 生态 + 混合精度 W4A16 部署 + FasterTransformer lineage**。

**选型决策表（论断式，每行回指）**：模型 × 硬件 × 特性需求 × 团队栈 → 引擎。vLLM 通用生态 / SGLang 结构化程序 + radix / DeepInfer DeepSeek 专用 + 双平台（SM100+Ascend 950）/ TRT-LLM NVIDIA PyTorch-native + Dynamo/Triton / LMDeploy InternLM 生态 + W4A16。**边界声明**：llama.cpp/端侧/CPU 不覆盖；非 LLM serving 不覆盖。来源：tensorrt-llm、lmdeploy、deepinfer-repo（P7 指针）、vllm-repo/sglang-repo（P5/P6 指针）、8.4（disagg recap）。

## 9.3 RL Rollout Serving

**training 8.5 recap 锚（纯文字指引，不重推导训练侧）**：
- **四模型显存**（§2 式 (1)）：Mem_PPO = 16N_a + 16N_c + 2N_a(ref) + 2N_r + 2N_a(rollout) + KV；同尺寸 **38N + KV**；GRPO 去 critic + RLVR 去 RM → **20N**。7B 266GB / 70B 2.66TB。
- **rollout/train 引擎拆分**（§3）：生成阶段约占 PPO 时间 80%（OpenRLHF §3.2 profile）；训练引擎（Megatron/FSDP）+ 推理引擎（vLLM）分置；layout 不一致引出 resharding；colocate（共卡时分复用）vs separate（分卡专用）。
- **权重同步训练侧**（§4 式 (2)）：t_sync ≥ 2N_a/B；70B NVLink 600GB/s ≈0.23s / 25GB/s ≈5.6s / 共享存储十几秒。**3D-HybridEngine**（式 (3)）micro-DP d_g=tp/(t_g p_g)，组内 all-gather 通信 (d_g−1)/(tp)·M、峰值 M/(t_g p_g)、冗余 0（Table 2）。
- **off-policy**（§5）：staleness + 引擎数值差两来源；rollout_corr_helper.py 截断 IS/拒采样；bypass（rollout_log_probs 当 π_old）vs decoupled（训练引擎重算 old logp 锚点）。
- **长 CoT**（§8）：R1 response 长度增长（2501.12948）；KV 随长度线性涨；max_len 截断切断 EOS；DAPO overlong/dynamic sampling（2503.14476，算法侧归 8.4）。

**引擎侧增量（本章核心，fresh 核实）**：
- **vLLM sleep mode / wake_up**：device_allocator/cumem.py **CuMemAllocator L62**（CUDA virtual memory management，tag 分池 weights/kv_cache；sleep L261 / wake_up L364 / use_memory_pool L394）；v1/worker/gpu_worker.py **Worker.sleep L361（level 1 权重 offload 到 CPU、仅 KV 丢弃 / level 2 丢弃被改写集、非改写集 offload——closeout errata，初版「level 1 丢全部」误，见语义纠正 4）/ wake_up L402（tags）**；entrypoints/llm.py **sleep L853 / wake_up L897 / collective_rpc L587**；EngineCore.sleep/wake（v1/engine/core.py）；**distributed/weight_transfer/ 六引擎**（nccl_engine / m2n_engine / modelexpress_engine / ipc_engine / sharded_rdt_engine / sparse_nccl_engine + base/clients/factory/packed_tensor——训练器→推理引擎权重传输后端族）。
- **SGLang checkpoint_engine / update_weights**：checkpoint_engine/update.py **update_weights L144**（torchrun CLI，split_checkpoint_files L79/split_tensors L92/req_inference L113/join L182）；checkpoint_engine_worker.py **SGLangCheckpointEngineWorkerExtension L39 / Impl L95**（zmq IPC，依赖外部 MoonshotAI/checkpoint-engine）；scheduler_components/weight_updater.py **SchedulerWeightUpdaterManager L131**（update_weights_from_disk L182 / init_weights_update_group L210 / destroy L215 / update_weights_from_distributed L232 / update_weights_from_tensor L287 / flush_cache_after_weight_update L164 / record_weight_version_after_update L179——函数全名 closeout 对齐 S3-3）；engine.py **release_memory_occupation L1902 / resume_memory_occupation L1908（tags 分池）/ update_weights_from_ipc L1793**（vLLM sleep/wake 对照）。
- **veRL 引擎接口（hybridflow 锚，⟨verify SGLang backend⟩ 销）**：三引擎 rollout backend（vllm_rollout/vllm_rollout.py **ServerAdapter L49** + sglang_rollout/sglang_rollout.py **ServerAdapter L128** + trtllm_rollout/）；vllm_rollout **update_weights L210**（collective_rpc update_weights_from_ipc + **BucketedWeightSender** zmq + bucket_size_mb + use_shm 共享内存）+ **release L204→sleep.remote / resume L195→wake_up.remote(tags)**；sglang_rollout **update_weights L305**（tensor buckets CUDA IPC MultiprocessingSerializer + UpdateWeightsFromTensorReqInput，参考 THUDM/slime）+ **release L288→release_memory_occupation(tags=[kv_cache,weights]，sleep_level 1 LoRA 只释 kv_cache) / resume L276**；**checkpoint_engine 多后端**（nccl/nixl/mooncake/kimi/hccl/delta + base + delta_sync/）。
- **OpenRLHF 引擎接口（training 8.5 §7 指引，vLLM-only）**：ppo_actor.py **broadcast_to_vllm L409**（_broadcast_param NCCL self._model_update_group.broadcast(src=0) / ray.util.collective.broadcast + **_handle_cuda_ipc L439** reduce_tensor→IPC handle→all_gather_object→update_weight_cuda_ipc；ZeRO-3 GatheredParameters L472 / TP GatherReplacedLayerParams）；vllm_engine.py **update_weight L115 / update_weight_cuda_ipc L121 / sleep L136 / wake_up L139**。**全 repo 无 sglang**。**不在 inference 章直接引 arXiv:2405.11143**（语义纠正 7）。

**colocate vs disagg rollout（关键推导）**：同卡训练+推理的显存切换（sleep/wake 错峰 KV+权重）vs 分离集群的权重广播（t_sync 下界 + 常驻 rollout 卡）。training 8.5 §3/§4 对照；veRL colocate_async（同卡池）vs separate_async（独立 rollout 卡 + hybrid 池）。

**partial rollout / async rollout（关键推导）**：长 CoT 截断续跑——veRL V1 partial rollout（abort 保留 tokens+logprobs → LB 重试 → 续跑用更新版本 → 保留前缀 KV 重建，语义纠正 6）；**staleness ↔ off-policy 校正 training 8.4（GSPO 序列级 ratio）/ 8.5 §5 指引**；veRL staleness 控制 max_off_policy_threshold=8 模型版本 / strategy drop 或 wait + trajectory_spans/staleness 监控；引擎侧 pause/resume 接口 = vLLM sleep/wake + SGLang release/resume_memory_occupation。**fully_async/one_step_off 迁 verl-recipe 注**（语义纠正 5）。

**长 CoT 长度爆炸 serving 含义**：response 长度漂移 → KV 容量涨 → 抢占（4.2 recompute 成本 recap）；rollout 时长由 batch 内最长序列决定。training 8.5 §8 指引。

**rollout 吞吐账（关键推导）**：采样并发 × 接受率 × 引擎吞吐（GRPO 每 prompt G 条 → 生成 token 数 ×G；投机解码接受率 recap 4.4）。来源：vllm-repo、sglang-repo、hybridflow、training 8.5/8.4（纯文字指引）。

## 9.4 多模态与特殊 Serving

**VE 分离与调度（recap 7.4/8.4 + 指引）**：
- **VE 单 GPU cheaper**（doc L51 逐字：「VE: a single GPU per instance is generally enough, and cheaper GPUs are recommended」）；DeepInfer ve/multimodal_processor.py **MultimodalProcessor L29**（W4 锚维持）。
- **SGLang mm_schedule.py**（managers/）：init_mm_embedding_cache L28（MultiModalStaticCache）/ **get_embedding_chunk L33**（chunked prefill embedding 切块）/ PrefetchedEmbeddingFunc L165 / PerImageRequestInfo L378 / _batch_encode_per_image_misses L389（cache miss 逐 image 批编码）/ **_get_chunked_prefill_embedding L593**。
- **vLLM encoder_cache_manager.py**（v1/core/）：**EncoderCacheManager L18**（check_and_update_cache L111 / can_allocate L148 / allocate L211 / free_encoder_input L259 / **compute_mm_encoder_budget L329**）/ EncoderDecoderCacheManager L392。
- **image patch 账（关键推导）**：ViT patches = (w/patch_size)(h/patch_size)（patch_size 14 由 config [W] 代入）；**downsample 后 LLM tokens = ceil(n_vit_h/3)×ceil(n_vit_w/3)**（downsample_ratio 3）。代码锚 DeepInfer ve/processor/image_processor.py **downsampled_seqlens L81 / _layout_from_pixels L87**（语义纠正 10）。**vision_config [W]**：patch_size 14 / downsample_ratio 3 / max_image_tokens 1024 / min_pixels 295936 / 32L / 1024h / 16 heads / 2816 inter。**数值锚**：896×896 → 4096 ViT patches → 484 LLM tokens（ceil(64/3)=22/维）。
- **VE LB fewest patches（doc L151 逐字，7.4 回链）**：machine 粒度随机选一台（每 GPU 一个 VE）→ 机内给当前处理 patch 最少 GPU；patch = (width/patch_size)×(height/patch_size)。
- **encoder cache 复用**（命中节省账）。

**结构化输出生产形态（SO，官方披露 doc L344-348）**：
- L344：「Compilation and inference are decoupled: **SO compilation runs asynchronously, and a request retrieves the compiled result before it enters the engine.**」
- L346：「Compilation stage (**on a dedicated SO machine**): the format constraint is compiled into a **grammar automaton**.」
- L348：「Runtime: at each decode step, the **automaton is advanced**, the set of legal tokens is determined, and a **mask** is generated and applied to the sampling logic.」
- 代码锚 DeepInfer structure_output/compiler.py **XGrammarCompiler L14**（compile_grammar L61，cache kill switch）。**1.4/5.4/6.1 回链**（SO 原理 / vLLM structured_output / SGLang constrained）。**SO 异步编译时序（关键推导）**。

**embedding / reranker serving（prefill-only、无 decode、pooling）**：
- **vLLM**：pooler/ 目录（**abstract.py Pooler ABC L16** + **seqwise/methods.py CLSPool L38 / LastPool L51 / MeanPool L61** + seqwise/poolers.py SequencePooler L44 + tokwise/）；**CLS/MEAN 不支持 partial prefill**（语义纠正 8）；encode API（AsyncLLM.encode）；**v1/pool/late_interaction.py（MaxSim reranker）**。
- **SGLang**：entrypoints/openai/**serving_embedding.py / serving_rerank.py / serving_classify.py / serving_score.py**；serving_embedding.py EmbeddingReqInput + _handle_non_streaming_request（无 streaming = prefill-only）。
- **batch 形态差异**：prefill-only 无 decode 循环、pooling 层取序列表示（CLS/last/mean）、无 KV cache 逐 token 增长。

**diffusion LLM serving 展望（一句 [C]，不展开）**：SGLang **dllm/algorithm/**（base.py **DllmAlgorithm L30** step L83/run L98/_run_fdfo L149 + low_confidence/joint_threshold/joint_threshold_indel/gemma4_renoise）+ config.py DllmConfig L8（block_size/mask_id/first_done_first_out_mode）；**并行解码 vs 自回归**；**models 4.8 §1 diffusion LLM recap 指引**（masked diffusion、LLaDA/SEDD、N 步全序列前向无 cache 复用 2N× attention FLOPs 账）。展望一句，不展开（spec §2 边界）。

**多模态请求生命周期（1.1 扩展）**：image → VE → embedding → LLM prefill。来源：deepinfer-doc、deepinfer-repo、sglang-repo、vllm-repo、v41-flash-config、models 4.8（纯文字指引）。

## 生产数字口径总表（官方披露 + doc 行号，2026-11-24 逐字再核全中）

| # | 数字/口径 | doc 行 |
| --- | --- | --- |
| 1 | 155T tokens/day 单集群；38 USD/GPU-hour = **收入**（off-peak）非成本 | L17 |
| 2 | prefill EP2–8/DP2–8 最小；decode EP64–128 Wide-EP 最大；**VE 单 GPU cheaper**；TP/CP 暂不用 | L47-53 |
| 3 | #DP=#EP attention DP；四 LB（Prefill/Decode/VE/EPLB）+ KVCache Storage | L33 |
| 4 | 1P+1D 可部署；**~1:1 经验配比**；最小高效单元 **64P+64D**（EP64+） | L91 |
| 5 | 跨 DC 压缩 KV 可行 + ~400Gbps/8GPU + 相邻 DC | L97 |
| 6 | 存储 **20×30.73 TB SSD/server**、**≥40 GB/s 读 / 5 GB/s 写**、**24h 保留**、scale out | L101 |
| 7 | **~14.4 TB/GPU 逻辑**（物理更大） | L105 |
| 8 | 3FS **EC 10+2** 容两节点 / 读 11 节点 any 10（张力注）/ GC LRU watermark | L111/L113/L115 |
| 9 | 3FS **~100 PiB** SSD；TTL agent 3d/chatbot 7d；**chatbot 3d 反事实 0.5–1% 命中损失** | L187 |
| 10 | **72h agent / 7d chatbot 近乎捕获全部可复用**（72h 跨周末） | L211 |
| 11 | **VE LB machine 随机 + fewest patches (w/p)×(h/p)** | L151 |
| 12 | ~80% of perfectly balanced offline | L155 |
| 13 | autoscaling prefill non-cached tokens / decode running reqs（8.5 recap） | L251 |
| 14 | 训推共调度 | L119 |
| 15 | **SO 专用机异步编译 grammar automaton + 请求入引擎前取编译结果 + 每步 advance+mask** | L344-348 |

媒体来源检查：本波全部数字均可在官方 repo doc / README / 论文逐字定位，**未发现 media-only 支撑的数字**。

## arXiv 来源核实（W6 范围，https export API 批量 200，4/4）

| ID | 标题（逐字） | 第一作者（作者数） | 版本（upd） | 用途 |
| --- | --- | --- | --- | --- |
| 2508.15601 | LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind | Li Zhang（9，Shanghai AI Lab） | v2（2026-05-15） | 9.2 [A]；论文 HTML 全文双管线四技术 61%/156% |
| 2409.19256 | HybridFlow: A Flexible and Efficient RLHF Framework | Guangming Sheng（9） | v2（2024-10-02） | 9.3 [A]；veRL 引擎接口锚 |
| 2401.09670 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | Yinmin Zhong（8） | v3（2024-06-06） | 9.1 goodput/sizing（W5 已深核本轮再验） |
| 2405.11143 | OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework | Jian Hu（15） | v6（2025-10-09） | 9.3 OpenRLHF 经 training 8.5 §7 指引（**不入 inference 独立条目**，ID 在 training/references.md） |

## 残留延迟项（writer / evaluator 须知）

1. **$38 收入非成本 + token 计数口径 + 集群 GPU 数未披露**（语义纠正 1）：9.1 成本算例代入须显式声明假设，不可把 38 当成本。
2. **TRT-LLM PyTorch-native + legacy TRT 移除**（语义纠正 2）：9.2 必写当前形态 + 历史化注，勿按 spec 字面「TRT 编译器路线」。
3. **vLLM/SGLang sleep/wake 行号漂移**（语义纠正 3）：evaluator 以本轮 file:line 为准（gpu_worker L361/L402、engine.py L1902/L1908/L1793）。
4. **veRL fully_async/one_step_off 迁 verl-recipe**（语义纠正 5）：9.3 async 以 V1 async trainer 为准。
5. **OpenRLHF vLLM-only + 经 training 8.5 §7 指引**（语义纠正 7）：不在 inference 章直接引 arXiv:2405.11143（lint FAIL 风险）。
6. **vLLM pooler 模块化 + CLS/MEAN 无 partial prefill**（语义纠正 8）：9.4 按 pooler/ 目录锚。
7. **存储 EC 张力 + 100PiB 口径歧义**（语义纠正 12 + 9.1 算术锚）：servers ≈183（raw）或 ≈220（usable）须声明口径；EC 属生产披露非开源 3FS。
8. **SGLang is_embedding flag 未逐一定位**（语义纠正 9）：9.4 用 serving_embedding/rerank 端点 + EmbeddingReqInput 锚定即可。
9. **dllm 只展望一句**（spec §2 边界）：9.4 不展开 diffusion LLM serving，models 4.8 §1 recap 指引。
10. **huggingface.co 官方渠道复核连续五波 000**：config 以 hf-mirror 传输核实 + 字段名/数值为准（spec risk 7 维持）。
11. **codeload 大 tarball 本轮截断**：大 repo 全程 trees + contents API（后续复用）；OpenRLHF 小 tarball 完整。
12. **W7 延迟项不在本轮**：lint display 反转义漏检（W7 加检查）；kit cap()/.dia-cap CSS（后续）；W4 carryover 2 指针收窄（models 4.1 §4.4→7.3 / models 4.5 §6.2→7.2，errata 通道，research 不改已发布文本）。

## references.md 本轮变更清单

- **P9 维持 3 条**（tensorrt-llm / lmdeploy / hybridflow，spec §5 P9 Sources）；**未新增条目**（OpenRLHF 经 training 8.5 §7 指引，spec 9.3 锚点）。
- **tensorrt-llm**（日期 2026-11-23→2026-11-24，W6 选型定位深核注：PyTorch-native + legacy TRT backend removed + Blackwell supported-hardware + Overlap Scheduler/CUDA Graph padding + 选型定位；被引 8.4, 9.2 不变）。
- **lmdeploy**（日期 2026-11-22→2026-11-24，W6 arXiv v2 + 论文 HTML 全文 + repo contents 深核注：TurboMind persistent batch + KV LRU recompute + W4A16 AWQ + InternLM 生态 + EuroSys 2027 README L25 + 双管线四技术 61%/156% + 双引擎/MXFP4/PD disagg；被引 9.2 不变）。
- **hybridflow**（日期 2026-11-23→2026-11-24，W6 ⟨verify SGLang backend⟩ 销项注：三引擎 rollout backend + vllm/sglang update_weights/release/resume + checkpoint_engine 六后端 + V1 async trainer partial rollout/staleness + fully_async/one_step_off 迁移注；被引 9.3 不变）。
- 跨波共享锚点被引已含 9.x 不变（deepinfer-doc 9.1/9.4、distserve 9.1、vllm-repo 9.3/9.4、sglang-repo 9.3/9.4、deepinfer-repo 9.4、v41-flash-config 9.4）；本轮未重取这些条目本体（doc/config byte-identical W4，vllm/sglang 关键文件本轮 contents 再核证据在本文件）。
- 新增 W6 精炼说明段。parse-check：**9 groups / 83 entries**（P1=11, P2=14, P3=16, P4=19, P5=4, P6=2, P7=9, P8=5, P9=3），无重复 key、无非法日期、无混合档、无竖线（含 escaped）；lint 抽查 8.4 LINT OK；P9 arXiv ID（2508.15601 / 2409.19256）grep-able。

## Closeout 更正登记（2026-11-24，W6 审查 S1-1 / S3-2 / S3-3 / S3-4 notes-refs 侧销项）

1. **vLLM sleep level 语义 errata（S1-1，必须项）**：本文件语义纠正 4（L33）与 9.3 引擎侧增量节（L115）初版口径「level 1 丢弃全部（weights + KV）/ level 2 保留权重更新不改写者」被 fresh HEAD fetch @6aeedab553dd 推翻（gpu_worker.py L368-371 discardable=∅ + sleep_mode_backend.py L58「nothing at level 1」+ cumem.py L292 offload→CPU backup / L386-387 wake_up cudaMemcpy 恢复 + llm.py L864-868 level 1 docstring 四文件互证；fix 复审 codeload tarball + contents API 双通道 byte-identical，当前 main 1d9f8058bf8b 对照 diff 0 行）。纠正后口径：**level 1 = 权重 offload 到 CPU（wake_up 自动恢复）+ 仅 KV 丢弃；level 2 = 被改写集（discardable）丢弃无备份、非改写集（preserved）offload；KV 两 level 均丢弃**。同条初版「(discardable parameters)」误标保留集（S3-1 同源）、LoRA level-1-只释-KV 判据误挂 vllm_rollout（S3-2 同源，实为 sglang_rollout L206；vllm_rollout = layered_summon 判据 L125-129）一并纠正。章侧 9.3 §2 已修复并留坑注痕迹；evaluator + fix 复审确认（docs/reviews/inference-w6-review-p9.md S1-1 / inference-w6-fix-review.md §1）。
2. **record_weight_version → record_weight_version_after_update（S3-3 同源对齐）**：9.3 引擎侧增量节 weight_updater 接口清单函数全名补齐（sgl weight_updater.py @ac0243723b0d L179 逐字，fix 审核实）。
3. **TRT-LLM overlap 标号 + overview 路径对齐（S3-4 同源）**：9.2 节「GPU n+1 与 CPU n-1 重叠」→「GPU 第 n+1 步与 CPU 第 n 步重叠」（doc prose L63-67 用 n 标号 / 代码注释 L77 用 n-1 标号，索引原点不同同一管线），overview.md 写全 docs/source/developer-guide/overview.md；references.md tensorrt-llm W6 注同步（fix-review advisory 2）。
4. **refs 被引回写（closeout 执行）**：10 条目 10 对——nvidia-datasheet / 3fs +9.1、vllm-repo / sglang-repo / deepinfer-repo / deepinfer-doc +9.2（fix-review advisory 3 flagged）、vllm / sglang / xgrammar / vllm-v1-blog +9.4。本文件变更清单「跨波共享锚点被引已含 9.x 不变」原注与条目实际不完全符（vllm-repo / sglang-repo / deepinfer-repo / deepinfer-doc 缺 9.2），references.md W6 精炼说明已按实际引用更正登记；forward + reverse 机械交叉核 clean（P9 四章参考节 key ↔ 被引字段）。