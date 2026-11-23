# W6 对抗审查报告 · P9 生产全流程（9.1 / 9.2 / 9.3 / 9.4）

日期：2026-11-24。Evaluator：fresh-context 独立审查（不继承写作上下文）。只读审查，未改章节、未 commit。
基准：evaluator-brief（W1 协议）+ inference spec §5 P9 / §2 / §4.1-4.3 / §6.3 + STYLE_GUIDE + R2 style sheet + rubric + inference/references.md + docs/research/inference-w6-sources.md + training 8.5。
一手来源：本轮自行 contents API raw / codeload / arXiv export API / hf-mirror 重取（不信任 writer 报告），证据存 /tmp/iw6-eval-evidence/，脚本 /tmp/iw6-eval-scripts/。

## 0. 总判定

- **Status：CONDITIONAL PASS（含 1 项 S1，修复后须复审 9.3）。**
- **S1 = 1，S2 = 0，S3 = 6。**
- **9.1 强制重推导：PASS**（全账独立重算，关键值 n_P=4 / n_D=45 / 4 服务器 / 0.0578 & 0.443 USD/M / 26.9% 全中；126 断言量级逐项复核无差）。
- 9.2 / 9.4 事实与教学审查通过（仅 S3 措辞/路径精度）；9.3 除 S1 外其余通过。
- 四项 writer concern 全部裁定成立（见 §3）；primary-source 口径（$38 收入 / TRT-LLM 历史化 / patch 账 / dllm 一行）全部核实。
- **唯一实质问题：9.3 §2 把 vLLM sleep level 1 的权重处置写反**（与本轮 fresh HEAD 代码、与 recap 源 5.4 §6、与本章 §2 自身 veRL 注三处矛盾）。研究笔记语义纠正 4 与 dispatch 预期同误，本轮 fresh fetch 推翻。

## 1. 强制重推导（9.1，独立重算，不信任 writer）

脚本 /tmp/iw6-eval-scripts/rederive_91.py（纯 Python 重实现章脊全链）。结论：**PASS，所有关键值与中间量逐一吻合**。

| 量 | 章值 | 独立重算 | 判定 |
| --- | --- | --- | --- |
| κ / κL | 4096 / 131072 B（128 KiB） | 2·8·128·2=4096；×32=131072 | ✓ |
| 服务时间 s_c / s_a | 15.9 / 76.7 ms | 15.94 / 76.70 ms | ✓ |
| E[S] / μ / c_s² / V | 28.1 ms / 35.6 / 0.748 / 0.874 | 28.09 / 35.60 / 0.7485 / 0.8742 | ✓ |
| 三 ρ 上界（均值/Kingman/attainment） | 0.877 / 0.891 / 0.722 | 0.8768 / 0.8906 / 0.7223 | ✓ |
| ρ_P | 0.72 | floor(min)=0.72 | ✓ |
| R_nc / D_P | 73728 / 1.390e15 | 73728 / 1.3898e15 | ✓ |
| n_P | 4（≥3.90） | 3.902→4 | ✓ |
| 线性-only 表观利用率 | 0.702→0.559 | 0.7023 / 0.5588 | ✓ |
| goodput 交叉核（尾/attainment/g_P） | 0.0843 / 91.6% / 25.0 | 0.0843 / 91.57% / 25.00 | ✓ |
| C / ΣC·T̄ / KV 总量 | 1843.2 / 2.210e7 / 2.897e12 B | 1843.2 / 2.2099e7 / 2.8965e12 | ✓ |
| 带宽闸 / 容量闸 → n_D | 37 / 45 → 45 | 36.026→37 / 44.96→45 → 45 | ✓ |
| 步时核（KV/权重/合） | 19.2 / 4.8 / 24.0 ms | 19.2 / 4.8 / 24.0 | ✓ |
| 存储 raw/usable/物理 | 614.6 / 512.2 / 17.28 TB | 614.6 / 512.17 / 17.28 | ✓ |
| GPUs/server（两口径恒等） | 35.6 | 35.567 = 35.567 | ✓ |
| 100 PiB → 服务器（raw/usable） | ~183 / ~220 | 183.2 / 219.8 | ✓ |
| 写/读率 / 24h 容量 | 17.72 / 44.02 GB/s / 1531 TB | 17.72 / 44.02 / 1531 | ✓ |
| 四约束 → n_stor | 2.99/3.54/1.10/1.38 → 4 | 2.989/3.543/1.101/1.378 → 4 | ✓ |
| 教学 TB/GPU vs 14.4 | 31.2（2.2×） | 31.24（2.17×） | ✓ |
| 成本 / USD/M（全/仅输出） | 98 / 0.0578 / 0.443 | 98 / 0.05779 / 0.4431 | ✓ |
| 口径差 / 每请求成本 | 7.67× / 2.72e-4 | 7.667 / 2.722e-4 | ✓ |
| 155T→tokens/h / 收入系数 / G=4096 | 6.458e12 / 5.884e-6 / 0.0241 | 6.4583e12 / 5.8839e-6 / 0.02410 | ✓ |
| h=0：n_P0 / 节省分数 / USD/M0 | 22 / 26.9% / 0.0790 | 21.26→22 / 26.87% / 0.07902 | ✓ |
| h=0 agent prefill / 写率闸 | 0.639 s / 13 台 | 0.6390 / 12.35→13 | ✓ |
| 练习 2(c) q=0.1 ρ 上界 | 0.531 | 0.5310 | ✓ |
| 练习 3 纯 chatbot（n_P/n_D/n_stor/USD/M） | 3 / 2 / 3 / 0.0217 | 3 / 2 / 3 / 0.02170 | ✓ |

命题 1（M/M/1 纯排队尾 ρe^{-(μ-λ)t}）证明独立复核：PASTA 几何队长 → Erlang(n,μ) 条件尾 → 交换求和 → 指数闭式，代数正确；t=0 给 ρ、积分给 ρ/(μ-λ) 自洽；attainment 上界唯一性（导数 e^{-μ(1-ρ)q}(1+ρμq)>0）正确。命题 2（两口径 GPU/server 恒等）EC 因子相消正确。命题 3（节省分数 Δn_P/(n_P0+n_D)）全 token 分母 + 同价口径代数正确。Kingman 方向（c_s²<1 时放松）正确，且恰当地细化了 1.3 L188「更保守」的全称表述（1.3 该句隐含 V>1，9.1 提点给出方向取决于方差——非矛盾，是精化）。

## 2. Findings 表

| # | sev | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S1-1 | **S1** | 9.3 §2 level 语义第 1 条（L30）+ 总结表 L192 + 练习 1(b) L205 | 「level 1：丢弃全部——weights 与 KV 的物理页都释放，wake_up 后权重须重新注入（走权重更新通道）」 | fresh HEAD 6aeedab553dd：gpu_worker.py L361-381 `discardable = _discardable_parameters(...) if level==2 else set()`（level 1 → 空集）；sleep_mode_backend.py CuMemBackend.suspend 调 `allocator.sleep(("weights","runtime","cudagraph"), discardable)`；cumem.py sleep L261 `offload = tag in offload_tags and ptr not in discarded`，offload→CPU backup，wake_up L364 从 cpu_backup_tensor 恢复。故 **level 1 权重 offload 到 CPU（保留、wake_up 自动恢复），仅 KV 丢弃**；非「丢弃全部权重 / 须重新注入」。与 recap 源 5.4 §6 命题 4 边界「weights 池若按 level 处置为保留到 host，则 wake_up 可恢复原权重」「权重按 level 可恢复」矛盾；与本章 §2 自身 veRL 注「level 1 只释 KV」矛盾 | 改为：level 1 = 权重 offload 到 host（wake_up 自动恢复，无需重新注入）+ KV 丢弃重建；level 2 = 被权重更新改写者丢弃、不改写者保留到 host。同步修总结表 L192 与练习 1(b) 校验锚。**含 S1，修复后复审 9.3** |
| S3-1 | S3 | 9.3 §2 level 2 条（L31） | 「保留权重更新不改写的一切（discardable parameters）」括注术语反了 | 代码 `discardable` = 被改写者（被丢弃），preserved = 不改写者（保留）；`return set(storage.values()) - preserved`。散文「保留不改写的一切」正确，括注把保留集误标为 discardable | 括注改「preserved parameters」或删；discardable 指被丢弃的改写集 |
| S3-2 | S3 | 9.3 §2 veRL 映射（L38） | 「vllm_rollout … 默认 sleep_level 见 config（LoRA adapter 模式 level 1 只释 KV）」 | LoRA-level-1-只释-KV 语义属 **sglang_rollout**（verl-sglang_rollout.py L206 `sleep_level = 1 if lora_served_as_adapter(...) else 2`、L291 docstring）；vllm_rollout L125-129 是 `layered_summon → sleep_level=1`（带 overflow 警告），非 LoRA 判据 | 把 LoRA 语义归 sglang_rollout；vllm_rollout 改述 layered_summon 判据 |
| S3-3 | S3 | 9.3 §3 权重更新面表（L55） | 「record_weight_version \| L179」 | sgl-weight_updater.py L179 实际函数名 `record_weight_version_after_update` | 补全函数名 |
| S3-4 | S3 | 9.2 §2（L31）+ 参考（L133） | PyExecutor / Overlap Scheduler / CUDA Graph 22% 出处标「overview.md」 | 内容实在 **docs/source/developer-guide/overview.md**（L17/21/23-31 四组件 Scheduler/KVCacheManager/ModelEngine/Sampler 逐字、L61 22%、L63-67 overlap n+1/n）；repo 另有 docs/source/overview.md（产品页）与 torch/arch_overview.md（组件作 Model Engine/Decoder/Scheduler），三文件同名易混 | 路径写全 developer-guide/overview.md（内容已核实正确，仅路径精度） |
| S3-5 | S3 | 9.4 §4 doc L348 引句（L131） | 引文止于「…applied to the sampling logic」 | doc L348 原句续「, forcing the output to conform to the target structure.」；截断未加省略号 | 补全句或加省略号 |
| S3-6 | S3 | 9.4 §1（L28） | 「_encode_images（L47）docstring 逐字『ViT -> downsample -> aligner…』」 | vl_aligner.py 方法起 L47，该 docstring 实在 L57 | 行号改 L57（或标「方法 L47，docstring L57」） |

无 S2。其余事实/公式/引用/教学/规范审查通过（见 §4 验证 log）。

## 3. Writer concern 裁定（独立复核，不照单全收）

| concern | 裁定 | 证据 |
| --- | --- | --- |
| 9.4：448×448 LB 直算 1024 vs resize 管线 1521 两层口径 | **成立，措辞清晰无矛盾** | 7.4 §7 L123 用 doc L151 尺寸直算 (448/14)²=1024；9.4 §2 L77 明写「LB 预估口径…与 resize 管线后实际 1521 是两层口径——LB 路由前估负载不跑 resize 管线」；本轮跑 image_processor.compute_image_layout 复现 448→546×546→1521/169/184 ✓。§3 命题 1 再谈第二层近似，前后一致 |
| 9.1：教学 31.2 TB/GPU > 官方 14.4 锚（KV 几何口径差） | **成立，已声明非矛盾** | 9.1 §6 L218 口径对照段明写「纯因 KV 几何口径不同…两口径不可互相代入」；重算 1531/49=31.24 ✓。μ/E[S] 前向指针 §3 已声明（L72「§3 算出」、L74「§3 表」）；u_P/u_D 与 8.5 quota u 异章异义已声明（L232），8.5 L122 u=quota 利用率核实 |
| 9.3：命题 1 GPU·时间判据 2r<s | **成立，代数与边界正确** | W_co−W_dis=n(2T+s)−2n(T+r)=n(s−2r)，disagg 省 iff 2r<s（T_r=T_t、m=n 前提声明）；64 卡例 39680 vs 39116.8 省 563.2（1.4%）✓；边界反例 r=15→2r=30>20 colocate 反超 ✓；练习 2 推广 m≠n 上界 m<n(T+s−r)/(T+r) 正确 |
| 9.1：删除主 KV 528B doc L223 误引 → 定性指针 2.3/7.2/7.3 | **成立，删除正确** | doc L223 逐字「state_size_to_disk = 20 layers × 134 tokens × 528 B…The 528 B/token is the MXFP8 SWA KV format」——L223 是 SWA state 非主 KV；9.1 全文 grep 无 528/L223，§6 改定性指针「量化 KV（2.3/7.3）+ 混合注意力（7.2）」✓ |

## 4. Verification log

### 4.1 Primary sources（本轮 fresh fetch，URL × 引句 × 结论）

| 项 | 来源 | 引句/事实 | 结论 |
| --- | --- | --- | --- |
| $38 = 收入 | DeepInfer doc contents API raw @a679bb86，md5 c805de23…（372 行，byte-identical W4）L17 | 「A single cluster currently serves 155 trillion tokens per day, with **overall revenue of US$38 per GPU-hour (at the off-peak price)**」 | ✓ 收入（off-peak）非成本；9.1 §7 三重声明（收入≠成本 / G 未披露 / 计数未定义）齐全 |
| P/D 配比 | doc L91 | 「1P + 1D is deployable. In practice the GPU ratio…roughly 1:1…smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode」 | ✓ 9.1 §5 逐字 |
| 存储锚 | doc L101/L105/L111/L113/L115/L187 | 20×30.73 TB / ≥40 GB/s 读 5 GB/s 写 / 24h / scale out；14.4 TB/GPU 逻辑；EC 10+2；读 11 any 10；GC LRU；~100 PiB | ✓ 9.1 §6 全中；EC 张力（10+2=12 vs 读 11）与开源 3FS CRAQ 无 EC 口径分离已注 |
| TTL | doc L187/L211 | L211「72 hours for agent…7 days for chatbot…captures almost all reusable…spans a weekend」；L187「a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate」 | ✓ chatbot 3d 反事实限定齐全（9.1 §8 + 坑注） |
| autoscaling / 80% / decode 写盘 / SO | doc L251/L155/L183/L344-348 | 逐字 | ✓ 9.1 §4 / §3 坑 / §6 / 9.4 §4 全中 |
| VE LB | doc L151 | 「fewest image patches — computed directly from the dimensions as (width/patch_size) × (height/patch_size)」 | ✓ 9.4 §3 + 7.4 §7 |
| vLLM sleep/wake 行号 | vllm contents @6aeedab553dd gpu_worker.py | Worker L243 / sleep L361 / wake_up L402；cumem CuMemAllocator L62 / sleep L261 / wake_up L364 / use_memory_pool L394 | ✓ 行号全中（W3 L351/L389 漂移 +10/+13 确认） |
| **vLLM level 语义** | gpu_worker.py L361-381 + sleep_mode_backend.py（CuMemBackend.suspend）+ cumem.py sleep/wake_up | level 1 discardable=∅ → weights offload CPU（保留）+ KV 丢弃；level 2 discardable=改写集 → 改写丢弃 + preserved offload | **✗ 9.3 level 1 写反（S1-1）**；level 2 散文正确 |
| SGLang 显存/权重 | sglang contents @ac0243723b0d engine.py / weight_updater.py | release L1902 / resume L1908 / update_weights_from_ipc L1793；SchedulerWeightUpdaterManager L131 + 六接口行号 | ✓ 全中（+3 漂移确认）；record_weight_version_after_update L179（S3-3） |
| SGLang mm_schedule | mm_schedule.py | init L28 / get_embedding_chunk L33 / PrefetchedEmbeddingFunc L165 / PerImageRequestInfo L378 / _batch_encode_per_image_misses L389 / _get_chunked_prefill_embedding L593 / _discard_mismatched L221；双键 docstring L400-402「prevents two colliding hashes with different placeholder spans from being deduplicated within the same batch」 | ✓ 9.4 §3 逐字 |
| SGLang embedding | serving_embedding.py | OpenAIServingEmbedding L30 / _handle_non_streaming_request L248 / EmbeddingReqInput / 单次 __anext__ L258 | ✓ prefill-only |
| SGLang dllm | dllm/algorithm/base.py + config.py | DllmAlgorithm L30 / step L83 / run L98 / FDFO；DllmConfig block_size/mask_id/first_done_first_out_mode | ✓ 9.4 §6 一行 [C] 不展开 |
| veRL 三引擎 | verl contents @7b6c9cc0e19d workers/rollout/ 目录 + checkpoint_engine/ 目录 | rollout = {vllm_rollout, sglang_rollout, trtllm_rollout}（+hf/naive）；checkpoint_engine = {nccl,nixl,mooncake,kimi,hccl,delta}+base+delta_sync | ✓ 9.3 §7 三引擎 + 六后端 |
| veRL 接口行号 | vllm_rollout.py / sglang_rollout.py | ServerAdapter L49/L128；vllm update_weights L210 / release L204 / resume L195；sglang update_weights L305 / release L288 / resume L276；sglang sleep_level L206 LoRA 判据 + L291 docstring；slime 参考 L321 | ✓ 全中（S3-2 LoRA 归属） |
| veRL async | docs/advance/v1_async_trainer.md / fully_async.md | colocate_async/separate_async + TransferQueue + partial rollout（abort 保留 tokens+logprobs、跨版本）；max_off_policy_threshold=8 / strategy drop|wait / trajectory_spans；fully_async stub「moved to verl-recipe…pinned to verl v0.9.1…V0 trainer」 | ✓ 9.3 §5 + 坑注逐字 |
| OpenRLHF | codeload tarball @26494490bc23（432135 B 完整）ppo_actor.py / vllm_engine.py | broadcast_to_vllm L409 / _handle_cuda_ipc L439 / GatheredParameters L472；update_weight L115 / update_weight_cuda_ipc L121 / sleep L136 / wake_up L139；全 repo grep 无 sglang | ✓ vLLM-only；9.3 经 training 8.5 §7 纯文字指引，不引 arXiv:2405.11143 |
| TRT-LLM 当前形态 | README @a41224a3dee2 L290 + legacy/architecture/workflow.md L9 | 「Architected on PyTorch」；「The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.」 | ✓ 9.2 历史化正确 |
| TRT-LLM PyExecutor/overlap/22% | developer-guide/overview.md L17-67 | 四组件 Scheduler/KVCacheManager/ModelEngine/Sampler 逐字；CUDA Graph padding「up to a 22% end-to-end throughput increase」；Overlap Scheduler n+1/n | ✓ 内容正确（S3-4 路径精度） |
| TRT-LLM IFB / 硬件 | features/paged-attention-ifb-scheduler.md / supported-hardware.md | IFB「also known as continuous batching or iteration-level batching」+ packed no-padding + context 排 generation 前；Blackwell B200/GB200/B300/GB300/DGX Spark + Hopper/Ada/Ampere | ✓ 9.2 |
| LMDeploy | README @517cb6d73d88 L25 + turbomind.md L29 + w4a16.md | EuroSys 2027 accepted（[2026/08] news）；「You may recognize this feature as "continuous batching" in other repos」+ N slots + KV LRU recompute + FasterTransformer；AWQ/GPTQ INT4 + lmdeploy lite auto_awq + V100(sm70)/Turing/Ampere/Ada | ✓ 9.2 persistent batch=continuous 非新机制 |
| LMDeploy 论文 | arXiv export API 2508.15601v2 摘要 | 双 hardware-aware 管线 + 四技术 + 16 LLM×4 GPU +「up to 61% lower serving latency (30% on average)…156% higher throughput (58% on average)…compared to existing mixed-precision frameworks」 | ✓ 9.2 基线限定正确 |
| arXiv IDs | export API 批量 4 ID | 2508.15601v2 LMDeploy/Li Zhang(9)；2409.19256v2 HybridFlow/Guangming Sheng(9)；2401.09670v3 DistServe/Yinmin Zhong(8)；2405.11143v6 OpenRLHF/Jian Hu(15) | ✓ ID↔标题↔一作全中 |
| V4.1 config | hf-mirror resolve/main/config.json，md5 e1e4964c…（3311 B byte-identical） | vision_config patch_size 14 / downsample_ratio 3 / 32L / 1024h / 16 heads / 2816 inter / max_image_tokens 1024 / min_pixels 295936；text_config hidden_size 5120 / 40L | ✓ 9.4 §2 字段表 + d_model 5120/40L |
| DeepInfer ve 代码 | contents @a679bb86 image_processor.py / vit.py / compiler.py / environ.py + W4 tarball（同 HEAD）multimodal_processor / mlp_projector / vl_aligner / vit_attn / prepare_tensor_input_vl / matcher / utils | downsampled_seqlens L81 两维 ceil；_layout_from_pixels L87；compute_image_layout L155；vit 常量 L27-32 / Attention L90 qkv+proj 4d² / causal=False L116 / SwiGLU L130-131；XGrammarCompiler L14 / dispatch_and_compile L54 / CVE-2025-32381 L27 / kill switch L17；environ timeout 20 L175 / cache 10GiB L177 / SO_BUILDER_CPUS 4 L178；matcher L261 / utils get_idle_cpus L154 | ✓ 9.4 全锚 |

### 4.2 9.4 image patch 账独立复算（跑代码逻辑，verify_94.py）

896²→4096/484/508 ✓；448²→546²→1521/169/184 ✓（LB 直算 1024 ✓）；1920×1080→1708×966→8418/943/968 ✓；672²→2304/256/274 ✓；224²→1521/169/184 ✓；896×448→2048/242/255 ✓。N_VE=411041792（4d²+3d·2816=12845056/层×32）✓；5.566 TFLOPs/图、att 39.5%、n*=6272、10764 att 63.2%、55.7 ms、18 图/s ✓；反例 1×4096 vs 4×1024 att 4×、总 1.421× ✓；emb 4.727 MiB、1GiB≈216 张 ✓；ceil(4096/9)=456≠484 差 28 ✓；8.46×/8.06× ✓。命题 1（单调代理非等价）证明独立复核正确（线性项精确、二次项交叉项非负、等号 iff 至多一图非零）。

### 4.3 跨章 recap 忠实性（对照源章原文）

| recap 接口 | 源章核实 | 结论 |
| --- | --- | --- |
| 9.1 ← 1.3 式 (7)-(11) | 1.3 L157/163/169/177/185 几何队长/W_soj/W_q/Kingman/余量规则逐字 | ✓ 忠实；Kingman 方向精化非矛盾 |
| 9.1 ← 8.2 式 (2)(5)-(8) 命题 4 | 8.2 L55/103/109/115/123 + 命题 4 双重无关性 | ✓ 混合推广忠实；带宽闸不含权重读项口径一致 |
| 9.1 ← 8.5 式 (1)-(4) 命题 1 / u / M_KV | 8.5 L35/41/51/57 + 命题 1「命中省 prefill 不省 decode」+ L122 u=quota + 60 GiB | ✓ 忠实；u 异章异义声明 |
| 9.1 ← 2.2 命题 2 | 2.2 L105 ΔFLOPs=2Nδ+2Ldδ² → 剩余 2N(T−δ)+2Ld(T²−δ²) | ✓ 9.1 式 (5) 正确 |
| 9.1/9.3 ← 1.2 式 (1)(3) / 章脊 | 1.2 L28 C_prefill=2N_act T+2LT²d_model（双向系数 4）+ κ=2H_kv d_head·bytes + N=7.50e9/W=1.606e10 | ✓ 章脊一致；9.4 ViT 系数 4 用双向口径正确 |
| 9.3 ← training 8.5 §2/§4/§5/§7/§8 | training 8.5 L43 38N+KV/20N、L84 t_sync≥2N_a/B、70B 0.23/5.6s 表、Table 2 (d_g−1)/(tp)·M、§7 OpenRLHF 70B+ 分卡、§4 ZeRO-3 坑 | ✓ recap 忠实；B→BW 翻译声明（§4.3） |
| 9.3 ← 4.2 式 (1) / 32768 表 | 4.2 L40 t_recompute=(2NT+2LT²d)/F_eff + L66「32768 时 36%」 | ✓ 284 ms / 144× 重算正确；「4.2 表」归属正确 |
| 9.3 ← 4.4 命题 3 / 章脊 | 4.4 L128 F=(1−α^{γ+1})/((1−α)(γc+1)) + α0.8/γ4/c0.1→2.40 + 5ms | ✓ 投机乘数正确 |
| 9.3 ← 5.4 §6 命题 4 / 16.1GB/3.35× | 5.4 L228-234 命题 4 + 16.1+40=56.1→23.9→80（3.35×） | ✓ 9.3 换 7B 口径声明（14+40=54→26→3.08×）；**但 5.4「权重按 level 可恢复」与 9.3 level 1 矛盾（S1-1）** |
| 9.3 ← 6.3 §7 / 2.4 §8 | 6.3 L165 checkpoint_engine + L173「权重是 cache 版本元组分量（2.4 §8）」；2.4 L161 版本元组 (deployment config, model weights, user group) | ✓ 指针正确 |
| 9.4 ← 7.4 §7 VE LB / 8.4 §6 / 7.1 命题 1 | 7.4 L123 448→1024 + L26 ∝E/e；8.4 §5/§6；7.1 L71 命题 1 release 边界 | ✓ 指针正确 |
| 9.4 ← models 4.8 §1 diffusion | models 08-frontier-outlook L51「N 步全序列双向前向，无 KV cache」+ L54「attention 项 2N 倍账…未工业验证」 | ✓ 一行 recap 忠实 |
| 9.2 ← 8.4 §5/§6/§8 + P5/P6/P7 | 8.4 L101 §5 TRT-LLM transceiver V2+NIXL；两层走读纪律 L20 | ✓ 防双展开，只 recap+指针 |

### 4.4 机械 / 规范检查

| 项 | 命令/方法 | 结果 |
| --- | --- | --- |
| lint 4 章 | bash scripts/lint-chapter.sh（仓库根） | 4/4 LINT OK |
| display 反转义 | /usr/bin/grep -F '\,' '\!' '\;' '\:' '\{' '\}' '\%' '\#' | 仅 9.1 L129/L254 两处 `\%`，均在 `$`…`$` backtick 保护行内形（STYLE_GUIDE §3 许可形）；display 块零命中 ✓ |
| CJK 篇幅 | python re [一-鿿] | 9.1=6459≥5500 ✓ / 9.2=2752≥2500 ✓ / 9.3=4461≥3500 ✓ / 9.4=5339≥3500 ✓ |
| arXiv 白名单 | grep arXiv: IDs × references.md | 引用 ID（2401.09670/2309.06180/2312.07104/2508.15601/2409.19256/2411.15100）全在 references；**2405.11143 仅现于 references W6 研究注，零章节引用** ✓ |
| references 竖线 | grep -F '\|' | 0 ✓ |
| 货币 | grep '\$' | 仅 doc 引句 US\$38 转义；余皆 USD ✓ |
| 公式内 `<` | grep '\$[^$]*<[a-zA-Z]' | 0（一律 \lt）✓ |
| rubric A 禁词 | grep 由此可见/不难发现/其实/强大的/… + 呢吧嘛句尾 | 0 ✓ |
| λ 声明 / n_P,n_D / bytes 不占字母 | 9.1 §1 | 「本章 λ=到达率，非 GAE」✓；n_P/n_D 计数 ✓；bytes 文字 ✓ |
| 命题 per-chapter | 逐章 | 9.1 命题 1/2/3 ✓ / 9.2 命题 1 ✓ / 9.3 命题 1/2/3 ✓ / 9.4 命题 1/2/3 ✓ |
| 练习分级 | 逐章 | 难度非递减、≥1 [10]/[20]、≤1 [40]、≥1 题以命题为证明目标、计算/代码题带校验锚 — 四章全合规 ✓ |
| R2-1…12 | 逐条 | 节首设问/数值 ✓、章脊锁定 ✓、公式后数字实例 ✓、命题-证明骨架 ✓、路线句 ✓、定义配例+反例 ✓、口径注 ✓、判断句定义 ✓、关节词 ✓、提点/坑块引用 ✓、总结回指列 ✓ |
| 总结/练习/参考模板 + ASCII 图 | 逐章 | 齐全；每章 ≥1 ASCII 图 ✓ |

## 5. 覆盖率与残留

- 强制重推导（9.1 全账）：100% 关键值 + 中间量独立重算。
- Primary sources：dispatch 点名项（$38 收入 / TRT-LLM 历史化 / vLLM sleep level / SGLang 行号 / OpenRLHF 不引 arXiv / veRL 三引擎 / patch 账 / dllm 一行）100% fresh fetch 核实；DeepInfer doc 25 组生产数字行号逐字再核。
- 跨章 recap：9.1/9.3/9.4 全部 recap 接口对照源章原文核实。
- 机械/规范：lint + 反转义 + 篇幅 + 白名单 + 竖线 + 货币 + rubric + R2 + 练习分级全覆盖。
- 残留延迟项（非本轮 finding，沿研究笔记）：huggingface.co 官方渠道连续不可达（config 经 hf-mirror，md5 byte-identical 五波一致）；W7 lint display 反转义检查待加。
- **S1-1 修复后须复审 9.3**（level 语义 + 总结表 + 练习 1(b)）。S3 累计下波清零。writer 反驳通道：附代码证据复核（S1-1 证据为 fresh HEAD 6aeedab553dd 三文件，研究笔记语义纠正 4 与 dispatch 预期同误，本轮推翻）。
