# W2 调研笔记：P3 推理算子 + P4 调度与 batching 事实源

调研日期：2026-11-23（W2 research pass）。本文件是 W2 九章（3.1–3.4、4.1–4.5）写作的事实源之一；写作 agent 不做额外调研，数值一律以本文件与 `inference/references.md` 为准。引用 key 白名单 = references.md P3/P4 全部条目 + 被引章节含 W2 章号的跨波锚点（vllm / vllm-repo / sglang / sglang-repo / orca / deepseekv3 / v41-flash-config / ocp-mx / nvidia-nvfp4 / deepinfer-repo / deepinfer-doc / tilekernels / batch-invariance）。

## 访问路径记录（后续 research pass 复用）

- arXiv：`https://export.arxiv.org/api/query?id_list=<逗号分隔>&max_results=N`——本轮 26 个 P3/P4 ID（含 3 个共享锚点 spot-check）批量一次取回，26/26 与 references 一致；另单独核实 Marlin 论文 2408.11743（1/1 中）。全文 HTML：`https://arxiv.org/html/<id>v<N>`（本轮 2211.17192v2 / 2305.09781v4 / 2501.01005v2 / 2210.17323v2 / 2306.00978v6 / 2211.10438v7 / 2308.16369v1 / 2403.02310v3 / 2305.05920v3 / 2208.07339v2 均 200）。
- GitHub：api.github.com 本机可达（未鉴权 core 60/h，本轮用量 <15）；深路径核实一律 **codeload tarball**（不占限额）：本轮取 21 个 P3/P4 repo + FastServe + FlexFlow inference 分支 + specinfer-ae；**vllm / sglang / DeepInfer 复用 W1 同日（2026-11-23）tarball**（/tmp/iw1-evidence/，版本漂移风险同日可忽略）。raw.githubusercontent.com 仍不可达。**组织更名新案例：flexflow/FlexFlow → flexflow/flexflow-train（301，API 返回 Moved Permanently，codeload 旧名可跟）**。
- HF：huggingface.co 本机仍不可达，V4.1 config 经 `https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json` 重取，W2 相关字段与 W1 完全一致。
- USENIX：NSDI'26 technical-sessions 页 200（FastServe 发表版标题 + 作者逐字核对）；Orca OSDI'22 复用 W1 PDF 核实。
- CRFM：FlashDecoding 博客 200 全文取得。
- 证据存档：/tmp/iw2-evidence/（arXiv XML / 论文 HTML+txt / 21 repo tarball 解压 / NSDI'26 页 / V4.1 config）。

## ⟨verify⟩ 销项总表（W2 范围）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 3.1 FlashInfer cascade / split-KV / JIT | **销**：cascade 是 repo API 名（MultiLevelCascadeAttentionWrapper + merge_state LSE 合并），论文口径 = composable formats 共享前缀分解；split-KV 决策 scheduler.cuh split_kv；JIT gen_jit_spec；NHD/HND utils.py；plan/run decode.py；tree mask = packed custom_mask | repo tarball + 论文 HTML §3.1 |
| 3.1 FlashMLA absorbed 矩阵 / 512+64 | **销（带版本漂移）**：64×576 K block（512 latent + 64 rope）见 docs/20250422-new-kernel-deep-dive.md（仍在 repo）；**2026.09.30 release 移除 Hopper / V3 系支持**，Hopper MLA decode 代码锚旧 commit ba89a34 | repo README + deep-dive doc |
| 3.1 DeepSelect top-k kernel | **销**：DSA indexer + sampler TopK，2-20× torch.topk，topk≤4096 主优化 512，bf16 indexer / fp32 sampling，CUDA+Ascend | repo README + deep-dive doc 存在 |
| 3.1 FlashDecoding split-KV 源头 | **销**：博客全文——三步 split-KV + LSE reduction、batch<SM occupancy 论断、8× e2e / 50× attention（CodeLlama-34B batch 1）；作者 Dao et al.（Meta）非 CRFM 本身 | crfm.stanford.edu 全文 |
| 3.1 FA2/FA3 算法级 | **销**：FA2 三技术（non-matmul FLOPs / seq 维跨 thread block / warp partitioning）；FA3 三技术（warp-specialization / matmul-softmax 交错 / FP8 block quantization + incoherent processing）+ 全部性能数 | arXiv 摘要原文 |
| 3.2 SmoothQuant 官方 repo URL | **销（补录）**：mit-han-lab/smoothquant——论文摘要逐字自链 + repo tarball 核实 | 论文 HTML L153 + codeload |
| 3.2 GPTQ/AWQ/LLM.int8 算法关键步 | **销**：H=2XX^T / lazy batch / Cholesky / act-order；AWQ s=s_X^α grid search [0,1] grid 20 / group 128；LLM.int8 vector-wise + outlier 维 fp16 分解（>99.9% int8） | 论文 HTML + repo README |
| 3.3 Marlin paper? | **销（升 [A]）**：arXiv:2408.11743 "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models"（Frantar et al.）存在，repo README Cite 块自引；~4× 保持到 batch 16-32 | arXiv API + README |
| 3.3 DeepGEMM FP8 scaling / JIT / UE8M0 / Mega MoE | **销**：DeepJIT 运行时编译；SM90 FP32 SF vs SM100 packed UE8M0（4/int）；Mega MoE mega-kernel（dispatch+GEMM+SwiGLU+combine、symm memory）⊂ DeepGEMM README；contiguous vs masked grouped GEMM | repo README |
| 3.3 DeepEP normal vs low-latency | **销（带版本漂移）**：spec 语义为 V1；当前 main V2.5 统一 EPBuffer（V1/NVSHMEM 全移除），V1 锚 tag v1.2.1；hybrid/direct 模式保留、do_cpu_sync 区分训练/prefill vs decode | repo README + tags API |
| 3.3 Mega MoE / EP>32 递减（DeepInfer doc） | **销**：L285 scale-up 通用解；L293-297 带宽不再瓶颈 → 通信/同步主导 + NVLink 多 job 干扰 | doc 逐字（W1 同日 tarball） |
| 3.4 TileLang / TileKernels / NanoFlow code | **销**：tilelang repo tarball；TileKernels = MoE routing / quant / Engram / mHC / RoPE TileLang 算子库（内部训推已用）；Nanoflow repo 含 gemv/ pipeline/ 实现 | codeload tarball |
| 3.4 CUDA Graph piecewise（vLLM/SGLang） | **销（带版本漂移）**：vLLM **CompilationLevel 已更名 CompilationMode**（NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE）+ CUDAGraphMode（NONE/PIECEWISE/FULL/...）+ compilation/piecewise_backend.py；SGLang srt/compilation/cuda_piecewise_backend.py | W1 同日 tarball |
| 3.4 batch-invariant kernels | **销（W1 已核路径本轮复用）**：vLLM determinism/{batch_invariant,batch_invariant_configs}.py、SGLang batch_invariant_ops/、DeepInfer DEEPINFER_BATCH_INVARIANT（attn.py 消费）；博客全文 W1 已核 | W1 证据 + 本轮 tarball 复核 |
| 4.1 Orca 机制 + 36.9× | **销（W1 提前取得）**：iteration-level scheduling + selective batching 原文、GPT-3 175B 同延迟 36.9× 吞吐 vs FasterTransformer——W2 writer 直接用 | W1 osdi22-yu.pdf |
| 4.1 vLLM/SGLang scheduler loop | **销**：vLLM v1/engine/core.py step()（schedule→execute→update_from_output）+ v1/core/sched/；SGLang scheduler.py event_loop_normal/overlap + get_next_batch_to_run + run_batch | W1 同日 tarball |
| 4.2 FastServe code? | **销（升 [A]，推翻 spec [B]）**：LLMServe/FastServe 官方 artifact 存在（AE README 自认 NSDI'26 paper artifact）；NSDI'26 发表版 "FastServe: Iteration-Level Preemptive Scheduling for LLM Inference"，作者与 arXiv 版逐字一致；C++ 后端 SwiftTransformer | repo tarball + USENIX NSDI'26 页 |
| 4.2 Llumnix repo | **销（带版本注）**：llumnix-ray = v0（论文实现）；2026-03 v1 另立 llumnix-project/llumnix | repo README |
| 4.2 SGLang schedule_policy 策略名 | **销（带版本漂移）**：LPM/DFS_WEIGHT/FCFS/LOF/RANDOM 仍在；**新增 HRRN / SHORTEST_PREFILL_FIRST / UNIBOOST（CacheAware）+ ROUTING_KEY（CacheAgnostic）** | W1 同日 tarball 逐字 |
| 4.2 vLLM sched policy / priority | **销**：SchedulingPolicy FCFS/PRIORITY；PriorityRequestQueue heap（priority 小者优先、同值早到优先） | tarball |
| 4.3 Sarathi code? | **销（[A] 依据收窄）**：原文无自链 repo；官方实现 = microsoft/sarathi-serve 的 sarathi scheduler provider（benchmark README 逐字引 2308.16369 + chunk_size 配置）；无独立 microsoft/sarathi repo（search 核实） | repo + GitHub search |
| 4.3 Sarathi-Serve repo | **销**：README 自链 OSDI'24（usenix.org/conference/osdi24/presentation/agrawal）；repo 含 orca/vllm/faster_transformer 对照 scheduler | repo README |
| 4.3 DeepInfer compute-quota 二次模型 | **销**：doc L143 逐字——compute_quota_limit ns + cost(q,c) 二次 + 解 q 不超 32,768 + DP all-reduce MIN；L137 四长度档 | doc（W1 同日 tarball） |
| 4.4 Leviathan Thm 3.8 无量纲 c | **销**：Def 3.7 c = M_q/M_p 单步时间比（cost coefficient）；Thm 3.8 (1-α^{γ+1})/((1-α)(γc+1)) + 证明；Eq 1 capped geometric；Cor 3.9 α>c | 论文 HTML 逐字 |
| 4.4 SpecInfer code + [A/B] 解 | **销（裁定 [A]）**：论文逐字自链 flexflow/FlexFlow；官方代码在 **inference 分支**（spec_infer.cc / incr_decoding.cc / python 绑定，与 Artifact Appendix 一致）+ AE goliaro/specinfer-ae；master 更名 flexflow-train 且无 spec 代码；ASPLOS'24 | 论文 HTML + 两 repo tarball + API 301 |
| 4.4 tree verification | **销**：token tree（expansion/merge 构造）+ Tree Attention + tree-based parallel decoding + MSS 随机解码；代码 TREE_VERIFY_MODE | 论文 §4 + spec_infer.cc |
| 4.5 EAGLE 1/2/3 + Medusa + REST + Draft&Verify | **销**：SafeAILab/EAGLE 覆盖三代（EAGLE-3 NeurIPS'25）；Medusa-1/2 + typical acceptance（2.2-3.6×）；REST NAACL 2024 datastore；Draft&Verify ACL 2024 layer skip | repo README + 摘要 |
| 4.5 vLLM spec_decode 路径 | **销（W0 纠正本轮再确认）**：v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py 三件均在；目录已扩（dspark/dflash/suffix 等） | W1 同日 tarball |
| 4.5 SGLang speculative 路径 | **销（带版本漂移）**：srt/speculative/ 在；**eagle_worker.py → eagle_worker_v2.py（EAGLEWorkerV2）+ eagle_worker_common.py**；dspark_components/ + cpp_ngram/ 在 | tarball |
| 4.5 MTP self-spec / V4.1 nextn | **销**：config num_nextn_predict_layers=3（hf-mirror 重取）；vLLM mtp/speculator.py 在；模型侧目标 recap models 4.5 | config + tarball |
| 3.1 NSA 官方 code? | **销（维持 [B]）**：GitHub search 仅社区实现（fla-org / lucidrains / triton port），无 DeepSeek 官方 repo | search API |

## 3.1 Decode Attention 算子族

**FlashDecoding 博客（[C] 开放技术博客，全文核实）**：作者 Tri Dao / Daniel Haziza / Francisco Massa / Grigory Sizov（Meta，发表于 CRFM Stanford 博客 2023-10-12）。核心论断逐字：
- occupancy 问题："During inference, the query length is typically 1: this means that if the batch size is smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100), the operation will only use a small part of the GPU!"；"With a batch size of 1, FlashAttention will use less than 1% of the GPU!"
- 三步 split-KV：(1) KV 切 chunk（"step (1) does not involve any GPU operation, as the key/value chunks are views of the full key/value tensors"）；(2) 各 split 并行 FlashAttention + "write 1 extra scalar per row and per split: the log-sum-exp of the attention values"；(3) "reducing over all the splits, using the log-sum-exp to scale the contribution of each split"。online softmax 两级使用（split 内 + split 间）。
- 数值：CodeLlama-34B batch 1、512→64k seq，"up to 8x speedups in decoding speed"（e2e）；"the attention itself is up to 50x faster than FlashAttention"；32k 以内 attention 时间近乎恒定。可用：FlashAttention ≥2.2、xFormers ≥0.0.22。来源：flashdecoding。

**FA2 / FA3（摘要原文核实，算法与策略级；指令级归 gpu）**：
- FA2（2307.08691，Tri Dao 独著）：FA1 仅 25-40% 理论峰值 → 三技术 (1) 减 non-matmul FLOPs (2) 单 head 也沿 seq 维跨 thread block 并行提 occupancy (3) thread block 内 warp 间 work partitioning 减 shared memory 通信；~2× FA1，A100 50-73% 峰值，训练至 225 TFLOPs/s（72% MFU）。
- FA3（2407.08608，Shah et al.）：FA2 在 H100 仅 35% util → 三技术 (1) warp-specialization 重叠计算与数据搬运（Tensor Core + TMA 异步）(2) block-wise matmul 与 softmax 交错（ping-pong）(3) FP8 block quantization + incoherent processing；H100 1.5-2.0×，FP16 至 740 TFLOPs/s（75% util）、FP8 近 1.2 PFLOPs/s；FP8 数值误差 2.6× 低于基线 FP8 attention。来源：fa2、fa3。

**FlashInfer（2501.01005 论文 + repo tarball 双核实）**：
- 论文口径：block-sparse row（BSR）统一 KV 格式（ragged Q/O 无 padding、任意 (Br,Bc) block size）；**composable formats**——共享前缀 KV 用大 Br block sparse 子矩阵（shared memory 复用）、独立后缀用 (1,1) block，无需数据移动只算 indices（Fig 3）；JIT 编译 attention variants（CUDA 字符串 functor 填模板）；load-balanced scheduler 兼容 CUDA Graph 静态配置；FA2 算法用于 ≤sm89、FA3 算法用于 Hopper。**论文无 "cascade" 一词——cascade 是 repo API 名**，写作时注明术语对应。
- repo 路径（main tarball）：`flashinfer/cascade.py` MultiLevelCascadeAttentionWrapper（L226，plan L401 / run L537）+ merge_state / merge_states（LSE 状态合并，两级 attention 正确性基础）；`include/flashinfer/attention/scheduler.cuh` split_kv 决策（L138-211）；`flashinfer/jit/core.py` gen_jit_spec（L760）；`flashinfer/utils.py` NHD=0 / HND=1（L57-58）；`flashinfer/decode.py` BatchDecodeWithPagedKVCacheWrapper（L1338）plan（L2179）/ run（L3237）；speculative tree mask = `flashinfer/prefill.py` maybe_packed_custom_mask。来源：flashinfer。

**FlashMLA（repo tarball，版本漂移重要）**：
- **2026.09.30 breaking change**：移除 Hopper 架构与 V3/V3.2/V4.0 支持、改 FP8/FP4 KV 格式；当前 main 面向 **V4.1**（B200 + Ascend 950）：sparse prefill/decode（FP8/FP4 KV）+ fused Q-norm/Q-RoPE/attention/O-RoPE/cast-to-FP8 kernel（B200 CUDA 13.3 prefill 至 1460 TFlops / decode 950 TFlops）+ dense prefill/backward。
- **3.1 的 Hopper MLA decode 走读锚**：旧 commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`（README 自链）+ `docs/20250422-new-kernel-deep-dive.md`（仍在当前 repo）：64×512 输出矩阵占 32,768 个 32-bit 寄存器（每 SM 65,536 → 只能容一个输出矩阵，FA3 ping-pong 不可直接用）→ 垂直切 O_L/O_R（各 64×256）+ 双 KV block 数学变换；**64×576 K block = 512 latent + 64 rope**（9 次 TMA copy 各 64×64，fine-grained TMA-GEMM pipelining）。absorbed 矩阵等价推导 recap models 4.1 §4 + 指引。
- open vs internal：DeepInfer doc L289 "FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower"、e2e 至多差 5%——**官方披露口径，详展归 7.3**，3.1 只一句 + 指针。来源：flashmla、deepinfer-doc。

**DSA serving / DeepSelect（repo README 核实）**：DeepSelect = DSA lightning indexer + sampler 的 TopK kernel，2-20× vs torch.topk；indexer 场景 bf16、topk≤4096、主优化 topk=512（= V4.1 config index_topk）；sampling 场景 fp32、vocab ~128K；CUDA + Ascend（Ascend 仅 bf16）；性能度量 = 有效内存带宽（TopK 无浮点运算，FLOP rate 无意义）；算法分析 docs/DeepSelect-deep-dive.md。V4.1 config（hf-mirror 本轮重取）：index_topk 512 / index_n_heads 32 / index_head_dim 128 [W]。candidate blocks 实现例证：DeepInfer `deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py`。indexer K cache / lightning indexer 模型侧 recap models 4.2 §6 + 指引。来源：deepselect、v41-flash-config、deepinfer-repo、nsa（背景，[B] 无官方 repo）。

**paged attention kernel / vLLM backends**：vllm/v1/attention/backends/{flash_attn,flashinfer,flex_attention,cpu_attn,...}.py + mla/ 子目录（W1 同日 tarball 核实）；block table gather 机制 recap 2.1 + 指引。来源：vllm-repo。

## 3.2 权重量化算法

**格式账（W1 已一手核实，本轮复用）**：OCP MX v1.0——MXFP8/MXFP6/MXFP4/MXINT8 一律 block 32 + E8M0 scale（8-bit 无符号纯指数），MXFP4 元素 E2M1；NVFP4——E2M1 + per-16 UE4M3 block scale + tensor-wide FP32 global scale（cuBLAS VEC16_UE4M3 + SCALAR_32F；TRT-LLM kv_cache dtype='nvfp4'）；**勿混 NVFP4_PLUS（Rubin SM107 per-32 UE5M3 无 global scale）**。V4.1 config [W]：weight_block_size [32,32]、scale_fmt ue8m0、expert_dtype fp4（本轮重取一致）；DeepInfer doc：主 KV per-16 e4m3「类 NVFP4 无 global scale」/ indexer per-32 ue8m0 MXFP4 / routed expert per-32 ue8m0 MXFP4。V3 blockwise 128×128 recap training 1.3 + 指引。来源：ocp-mx、nvidia-nvfp4、v41-flash-config、deepseekv3、deepinfer-doc。

**GPTQ（2210.17323v2 原文核实）**：谱系 OBS（二阶剪枝）→ OBQ（逐权重量化 + 误差补偿，O(d_row·d_col³) 立方复杂度）→ GPTQ 三步：(1) 任意序量化（OBQ 贪心序的松弛）(2) **Lazy Batch-Updates**（block 内逐列量化、block 末统一更新剩余权重，Fig 2）(3) **Cholesky Reformulation**（H^{-1} 的逐行消去 ≈ Cholesky 分解，预计算全部逆 Hessian 信息 + mild dampening 保数值稳定）。layer-wise MSE Hessian **H_F = 2X_F X_F^T**（原文逐字）。repo README：`--act-order`（按激活幅度降序量化列）、`--static-groups`（预定义 group grid 使 act-order 无需推理侧改动）、`--true-sequential`；act-order 对 OPT-66B outlier 模型 4bit PPL 9.55→9.34。来源：gptq。

**AWQ（2306.00978v6 原文核实）**：0.1%-1% salient weights（跳过量化显著降损，但 mixed-precision 硬件不友好）；salient 由**激活分布**（非权重分布）定——大激活幅度对应通道更重要；per-channel scaling 等价变换 (W·s)(X/s)=WX 保护 salient 通道；**s = s_X^α，α* = argmin_α L(s_X^α)**，s_X = 逐通道激活平均幅度，α 在 [0,1] **grid search（grid size 20）** + weight clipping 最小化 MSE；group size 128、INT4/INT3 weight-only、Pile 小校准集。来源：awq。

**SmoothQuant（2211.10438v7 原文核实 + 官方 repo 补录）**：W8A8（权重易量化、激活难）；等价迁移 **Y = (X diag(s)^{-1})(diag(s) W)**；**s_j = max|X_j|^α / max|W_j|^{1-α}**（migration strength α）；α=0.5 为 OPT/BLOOM sweet spot（0.4-0.6 区间，<0.4 激活难量化、>0.6 权重难量化）、GLM-130B α=0.75；530B 单节点 serving。**官方 repo = mit-han-lab/smoothquant（论文摘要逐字自链，本轮补入 refs）**。来源：smoothquant。

**LLM.int8()（2208.07339v2 原文核实）**：两部分 = vector-wise quantization（每内积独立归一化常数，行/列 absmax 缩放、Int32 输出外积反量化）+ **mixed-precision decomposition**（outlier 特征维度隔离走 fp16 matmul，>99.9% 值走 int8，两部分输出 fp16 累加）；outlier 现象：6.7B 起相变（所有层、75% 序列维受影响），幅度至 20×、150,000 outliers/seq 但集中于 6 个特征维（~0.1% 特征）；置零这些维度 → attention softmax top-1 概率质量降 >20%、PPL 恶化 600-1000%。repo TimDettmers/bitsandbytes。来源：llm-int8。

**量化质量评估口径**：perplexity / task benchmark，不写普适「无损」（spec 3.2 硬口径）；GPTQ/AWQ 论文 PPL 数可作算例（上文已给出处）。

## 3.3 量化 GEMM 与 Fused MoE 算子

**Marlin（升 [A]：论文 2408.11743 + repo）**：FP16xINT4 mixed-input GEMM；README 逐字 "close to ideal (4x) speedups up to batchsizes of 16-32 tokens (in contrast to the 1-2 tokens of prior work)"；带宽账素材：GPU FLOP:byte ≈ 100-200 → 每 4-bit 权重 <25-50 次 TC MAC 即可维持近理想 4×；技术 = 激活 L2 常驻 + 寄存器复用、async 全局权重加载（立即驱逐 cache policy）、shared memory double buffering、dequant 与 TC 指令交错饱和双流水线、权重/scale 离线 reshuffle、多 warp 同输出 tile partial results、striped partitioning、L2 内 global reduction。**指令级 lop3/bfe 归 gpu**。来源：marlin。

**DeepGEMM（README 核实）**：统一 tensor core kernel 库（FP8/FP4/BF16 GEMM + Mega MoE + indexer MQA logits + HC）；**DeepJIT** 运行时编译（安装期无 CUDA 编译）；命名 D = C + A@B、NT layout（SM90 仅 NT、SM100 全 layout）；**SM90 SF = FP32，SM100 SF = packed UE8M0（4 个 pack 进一个 torch.int）**，LHS SF 须 TMA-aligned transposed；**grouped GEMM 只 group M 轴（N/K 固定）**：contiguous layout（训练 fwd / prefill 变长 token 拼接、每 expert 段对齐 get_mk_alignment_for_contiguous_layout()）vs **masked layout**（decode + CUDA Graph、CPU 不知每 expert token 数、mask 指定有效部分、典型输入 = DeepEP low-latency 输出）；indexer MQA logits（fp8_fp4_mqa_logits non-paged / paged，weighted ReLU MQA）。fine-grained scaling 精度提升口径沿 spec（per-block scale 降量化误差，V3 blockwise 背景 recap training 1.3）。来源：deepgemm。

**Mega MoE（DeepGEMM README + DeepInfer doc 双口径）**：
- DeepGEMM README：mega-kernel 融合 **EP dispatch + linear1/linear2（FP8xFP4 或 FP8xFP8）+ SwiGLU + EP combine**，NVLink 通信与 tensor core 计算重叠；需 symmetric memory 多进程 launch（PyTorch ≥2.9）；fp8_fp4_mega_moe API + transform_weights_for_mega_moe。**Mega MoE ⊂ DeepGEMM 组件（README 口径，四路审查 cross-route 观察兑现）**。
- DeepInfer doc（官方披露）：L285 "Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE"；**EP>32 递减账**（L293-297）：扩 EP 提算术强度避免 memory-bound → 带宽不再瓶颈后，更大 EP 引入额外通信/同步开销；超节点内不同 job 共享 NVLink 互相干扰，整超节点单 EP 组避免干扰。来源：deepgemm、deepinfer-doc。

**DeepEP（README 核实，版本漂移重要）**：
- **当前 main = V2.5**：V1 API / NVSHMEM 后端 / 旧文档**全部移除**；high-throughput 与 low-latency 统一为单 **EPBuffer** 接口；NCCL Gin 后端（device-side 通信 API、复用 NCCL communicator）；DeepJIT 编译；SM/QP 数解析计算（免 auto-tuning）；hybrid / direct 两模式保留；另实验性 PP/CP/DP bucket collectives + Engram RDMA。
- **spec 大纲「normal 模式 NVLink+RDMA 两阶段 vs low-latency 模式纯 RDMA hook」是 V1 语义**——写作锚 **tag v1.2.1**（V1 最后版本，tags API 核实）讲机制，再注当前 V2.5 统一 API；V2.5 区分训练/prefill vs decode 的方式 = `do_cpu_sync=True`（精确输出尺寸 + CPU 侧 expert 计数）vs False（GPU 侧接收计数、按 capacity 分配、expert GEMM 用 handle.psum_num_recv_tokens_per_expert 定有效段 → 接 DeepGEMM masked layout）。
- README Notes：EP dispatch/combine 占 GPU SM，**零 SM RDMA EP 不支持**（与 DeepInfer GDR swap-in 不占 SM 对照素材）；FP8 dispatch 支持；NVLink 冗余 expert 权重/梯度交换（lb_prefetch_weights / lb_reduce_grads，MoonEP/UltraEP 动机）。硬件归 cluster。来源：deepep、deepseekv3（训练侧背景）。

**sampling / logits kernel**：DeepSelect top-k（sampler 场景 fp32 vocab ~128K）；vLLM v1/sample/（+ops/）、SGLang srt/layers/sampler.py（W1 核实路径）；fused penalties / argmax 机制级。来源：deepselect、vllm-repo、sglang-repo。

## 3.4 融合、图捕获与编译

**算子融合语义**：RMSNorm+quant / SiLU+mul / residual+norm 融合节省 = 中间 tensor IO 字节账（spec 关键推导，无外部数值依赖）；实例素材：TileKernels fused SwiGLU+quant、FlashMLA fused norm-RoPE-attn-RoPE-cast（"saves the overhead of those small kernels"，B200 prefill 1460 / decode 950 TFlops）。来源：tilekernels、flashmla。

**CUDA Graph（版本漂移重要）**：
- **vLLM（W1 同日 tarball）**：**CompilationLevel 已更名 CompilationMode**（config/compilation.py L38）：NONE=0（eager）/ STOCK_TORCH_COMPILE=1 / DYNAMO_TRACE_ONCE=2 / **VLLM_COMPILE=3**（自定义 Inductor backend：caching + piecewise compilation + shape specialization + custom passes）；**CUDAGraphMode**：NONE=0 / PIECEWISE=1 / FULL=2 / FULL_DECODE_ONLY=(FULL,NONE) / FULL_AND_PIECEWISE=(FULL,PIECEWISE)；实现 compilation/{piecewise_backend,cuda_graph,backends,partition_rules,breakable_cudagraph,cudagraph_pool}.py。**旧名 CompilationLevel / PIECEWISE=3 在当前 main 不存在——写作按新名 + 漂移注**。
- **SGLang**：srt/compilation/{cuda_piecewise_backend,backend,compilation_config,compile_phase,npu_piecewise_backend}.py（piecewise CUDA graph，attention 动态 shape 排除）。
- launch 开销账（~μs × 每步 kernel 数）为 spec 推导，无外部数值依赖；full vs piecewise 语义对照按 spec。来源：vllm-repo、sglang-repo。

**TileLang / TileKernels**：TileLang（2504.17577，Wang et al.）composable tiled programming model、tile 抽象 DSL、多硬件后端（repo tarball 核实）；TileKernels（README 核实）= 数十个 TileLang 优化算子：MoE routing top-k/scoring、per-token/per-block/per-channel FP8/FP4 cast/dequant + fused SwiGLU+quant、Engram gating（fused RMSNorm + fwd/bwd）、mHC（Sinkhorn normalization）、RoPE；多数接近硬件算力/带宽极限、已用于内部训推；2026-09-30 起 NVIDIA + Ascend 双后端运行时自动选择。DeepInfer custom ops：deepinfer/model_executor/layers/tilelang_ops/（candidate_blocks.py / compress.py 等，tarball 核实）。来源：tilelang、tilekernels、deepinfer-repo。

**batch-invariant kernels（W1 全文 + 路径核实，本轮复用）**：Thinking Machines 博客（Horace He，2025-09-10，[C] 开放技术博客）——浮点非结合 "original sin"；batch 组成改变 reduction 分区/顺序 → 结果变化；解法 = 固定 reduction order 的 RMSNorm/matmul/attention kernels（split-K 固定分区、禁 atomicAdd 非确定累加）。引擎路径：vLLM model_executor/determinism/{batch_invariant,batch_invariant_configs}.py（**无 batch_invariant_ops 目录**）；SGLang srt/batch_invariant_ops/batch_invariant_ops.py；DeepInfer DEEPINFER_BATCH_INVARIANT（attn.py 消费）。1.4 回链。来源：batch-invariance、vllm-repo、sglang-repo、deepinfer-repo。

**NanoFlow（2408.12757 摘要 + repo 核实）**：end-to-end serving 多数常见 workload 实为 **compute bound**（反直觉论断，摘要原文）；intra-device parallelism = nano-batch 切分 + operation 复制 + execution unit scheduling 重叠 compute/memory/network；自动搜索 nano-batch 数量/大小/顺序/资源分配（考虑并发干扰）；**1.91× vs SOTA、达 optimal 吞吐 50-72%（摘要口径；README 68.5% 为另一表述——引用以摘要为准并注明）**；repo 含 gemv/ pipeline/ 实现代码（efeslab/Nanoflow，[A] 维持）。为 6.2 / 7.1 埋伏笔（DeepInfer async scheduling 仿 SGLang/NanoFlow，doc 口径）。来源：nanoflow。

## 4.1 Continuous Batching 与迭代级调度

**Orca（W1 OSDI'22 PDF 原文核实，本轮复用；[B] 无官方代码）**：
- iteration-level scheduling："schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch"（摘要）。
- selective batching："splits the batch and processes each request individually for the Attention operation while applying batching to other operations"（§1）；"token-wise (instead of request-wise) batching"（§4，Fig 5：7 token 批 → 非 Attention 算子输入 [7,H]，Attention 前插 Split）。
- **36.9× 吞吐（⟨verify⟩ W1 已提前销）**：GPT-3 175B "36.9× throughput improvement at the same level of latency" vs NVIDIA FasterTransformer（摘要；PDF 为准）。writer 直接用，勿再标 ⟨verify⟩。
- 作者 Gyeong-In Yu et al.（Seoul National University / FriendliAI）。来源：orca。

**static batching 浪费账**：padding / head-of-line blocking / 早 finish 空转（spec 推导，无外部数值依赖；Orca Fig 1 可作示意出处）。

**vLLM scheduler loop（W1 同日 tarball）**：v1/engine/core.py **step()（L714）**：scheduler.schedule()（L724）→ execute → scheduler.update_from_output()（L738）；step_with_batch_queue（L759，async 变体）；v1/core/sched/{scheduler,async_scheduler,request_queue,output,interface,utils}.py（+diffusion_scheduler / prefix_lm_scheduler 新增）。机制例证级（两层纪律：模块全量走读归 5.2）。来源：vllm-repo、vllm。

**SGLang scheduler loop（tarball）**：managers/scheduler.py **event_loop_normal（L2231）/ event_loop_overlap（L2266）** + get_next_batch_to_run（L4386）+ run_batch（L5455）+ schedule_batch.py；overlap 机制详展归 6.2，4.1 只机制级。来源：sglang-repo、sglang。

## 4.2 调度器架构与抢占

**抢占策略成本账**：recompute FLOPs vs swap 带宽×KV 字节（spec 关键推导）；vLLM 论文 §4.3 recompute/swap 口径 recap 2.1 背景（vllm 条目）；watermark 触发。

**FastServe（升 [A]，spec 原判推翻——writer 须知）**：
- arXiv 2305.05920（v3，Bingyang Wu et al. 9 作者，Peking University）；**发表版 NSDI'26**："FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference"（USENIX 页逐字，作者完全一致；URL usenix.org/conference/nsdi26/presentation/wu-bingyang）。
- **官方 artifact = LLMServe/FastServe**（README：preemptive scheduling + continuous batching + proactive memory swapping + paged attention kernels + megatron-LM TP + streaming PP；C++ 后端 LLMServe/SwiftTransformer；benchmarks/artifact-evaluation/README.md 自认 NSDI'26 paper artifact）。
- 机制（论文原文）：每 output token 粒度抢占、**skip-join MLFQ**（Algorithm 1 伪代码）、proactive swapping；实现规模 2.9K Python（frontend+scheduler）+ 8.1K C++/CUDA（执行引擎，Ray actor GPU workers）；custom CUDA kernels 支持 Orca iteration-level scheduling + vLLM PagedAttention（论文 §Implementation 逐字）。来源：fastserve。

**Llumnix（2406.03243，[A]）**：live migration + 动态重调度 + 碎片感知（摘要：异构不可预测请求 → 排队延迟/尾延迟/SLO 违反）；repo llumnix-ray = **v0 Ray 架构（论文实现）**；**版本注：2026-03 v1 重构另立 llumnix-project/llumnix（cloud-native），v0 留本地部署/实验**——4.2 机制锚 v0；为 8.5 埋伏笔。来源：llumnix。

**SGLang schedule_policy（tarball 逐字，版本漂移）**：CacheAwarePolicy = **LPM**（longest prefix match）/ **DFS_WEIGHT**（depth-first search weighting）/ **HRRN**（highest response ratio next，token-based aging）/ **SHORTEST_PREFILL_FIRST** / **UNIBOOST**（tail-aware admission boosting）；CacheAgnosticPolicy = **FCFS / LOF**（longest output first）/ **RANDOM** / **ROUTING_KEY**（按 running batch 中 routing key 频率优先）。**spec W0 核实的五项仍在，新增四项（HRRN/SHORTEST_PREFILL_FIRST/UNIBOOST/ROUTING_KEY）——写作给全集 + 注明 W0 五项为 spec 口径子集**。LPM 有 BOUNDED_LPM_WINDOW_SIZE=64 窗口限制（L181）。来源：sglang-repo。

**vLLM sched policy / priority（tarball）**：SchedulingPolicy FCFS / PRIORITY（"priority"）；FCFSRequestQueue（deque）/ PriorityRequestQueue（heap，**priority 值小者优先、同值早到优先**，L131-144）；scheduler.policy 由 scheduler_config.policy 决定。encoder cache 调度（多模态 → 9.4）：v1/core/encoder_cache_manager.py（W0 核实存在）。来源：vllm-repo。

## 4.3 Chunked Prefill 与混合批

**Sarathi（2308.16369v1 原文核实）**：chunked-prefills + **decode-maximal batching**（piggyback decodes with prefills，把 memory-bound decode 转 compute-bound）；chunked-prefills 提供更多 prefill 供 decode piggyback + 统一 work unit 减 pipeline bubbles；decode batch 上限由 GPU 内存 / 参数内存 / 最大序列长决定（原文公式素材）。**[A] 依据**：原文无自链 repo；官方实现 = microsoft/sarathi-serve 的 **sarathi scheduler provider**（benchmark README 逐字引 2308.16369、chunk_size / enable_dynamic_chunking_schedule 配置）；无独立 microsoft/sarathi repo（search 核实）。来源：sarathi。

**Sarathi-Serve（2403.02310v3，OSDI'24）**：stall-free batching + throughput-latency 权衡 + chunk size 调优；repo README 自链 OSDI'24（usenix.org/conference/osdi24/presentation/agrawal）；repo 含 orca / vllm / faster_transformer scheduler 对照实现（sarathi/core/scheduler/）。来源：sarathi-serve。

**stall 账 / chunk size 权衡**：spec 关键推导（c_chunk 太小 attention 二次项效率低、太大 stall 长）；记号 c_chunk（c 保留 CP 度，spec §4.3）。

**DeepInfer compute-quota（doc L143 逐字，官方披露）**："Each prefill step is given a time budget (`compute_quota_limit`, in nanoseconds), and a fitted quadratic cost model `cost(q, c)` predicts how long the step will take (`q` = number of sequence tokens, `c` = the prefix length already computed for that sequence). The scheduler solves that quadratic for `q`, taking the largest `q` that stays within the 32,768-token budget while exactly consuming the remaining time budget. The budget itself is further reduced by an all-reduce MIN within the DP group."；L137 四长度档（<16K / 16-64K / 64-256K / ≥256K）各档 compute-quota cap、按上轮需求份额动态重分配引擎；L139 按剩余 quota pull work、无 prefix-affinity 路由。记号 cost(q,c) 沿 doc 原记号章首声明。为 7.4 埋伏笔。来源：deepinfer-doc。

**vLLM / SGLang chunked prefill 实现（tarball）**：vLLM config/scheduler.py **max_num_batched_tokens**（L47）/ **long_prefill_token_threshold**（L104）/ long_prefill_token_threshold_adaptive（L112，按 token 预算 / 请求数自适应）；SGLang **chunked_prefill_size 已从 server_args.py 迁出**——经 runtime_context.py get_schedule()（L1351）读取，scheduler.init_chunked_prefill（L1468）+ **scheduler_components/dynamic_chunk_sizer.py（动态 chunk size，spec 4.3 dynamic chunk size 素材）**。来源：vllm-repo、sglang-repo。

## 4.4 投机解码原理与账

**Leviathan（2211.17192v2 原文逐字核实，[B]）**：
- **Def 3.7**："Let c, the cost coefficient, be the ratio between the time for a single run of M_q and the time for a single run of M_p."——**无量纲 c = c_d/c_v**（spec §4.3 口径源头）。
- **Eq 1**：E[#generated tokens] = (1-α^{γ+1})/(1-α)（i.i.d. β 假设、α=E(β)，capped geometric：成功概率 1-α、cap γ+1）。
- **Thm 3.8**：walltime improvement factor = **(1-α^{γ+1})/((1-α)(γc+1))**；证明：设 M_p 单步成本 T，每轮成本 Tcγ+T（γ 次 M_q + 1 次 M_p），除以 Eq 1 期望产出得每 token 期望成本，与标准解码 T 相比即得。
- **Cor 3.9**：α>c 则存在 γ 使加速，factor ≥ (1+α)/(1+c)（γ=1 代入）。
- 前提：γ+1 并发 M_p 评估不增 walltime（足够算力，§3.4）；Thm 3.8 假设足够长生成；T5-XXL 实测 2-3× out-of-the-box。
- 接受概率 min(1, p(x)/q(x)) + 残差分布 norm(max(0, p-q)) 修正（分布保持证明骨架按 spec，不跳步）。来源：leviathan。

**Chen（2302.01318，[B]）**：modified rejection sampling 保目标分布（hardware numerics 内）；Chinchilla 70B 基准；与 Leviathan 独立同期——只作公式源头并列，不写历史叙事（spec 口径）。来源：chen。

**SpecInfer（2305.09781v4，[A] 裁定——writer 须知）**：
- **等级裁定**：spec 原 [A/B] 混合档销项 → **[A]**。依据：论文逐字 "SpecInfer is publicly available at https://github.com/flexflow/FlexFlow/" + Artifact Appendix 列程序 spec_infer.cc / incr_decoding.cc + 官方代码实在 FlexFlow **inference 分支**（inference/spec_infer/spec_infer.cc、inference/incr_decoding/incr_decoding.cc、inference/python/{spec_infer,incr_decoding}.py，tarball 核实）+ AE repo goliaro/specinfer-ae（ASPLOS'24 artifact，FlexFlow 子模块）。**路径注：flexflow/FlexFlow 已 301 更名 flexflow/flexflow-train，当前 master 无 spec 代码——引用锚 inference 分支**。
- 机制（论文 §3-4）：token tree（节点 = 候选 token 序列；**expansion-based / merge-based** 两种构造）；**Tree Attention**（tree mask 使多候选共享前缀计算）；**tree-based parallel decoding** 一次 forward 验证整树；**MSS（multi-step speculative sampling）** 随机解码验证（VerifyStochastic，Algorithm 2 / Fig 5）；LLM 作 token tree verifier 而非 incremental decoder，provably 保模型质量。代码例证：spec_infer.cc TREE_VERIFY_MODE / push_spec_infer_tree_width。kernel 归 3.1（FlashInfer custom mask）。来源：specinfer。

**tree verification 正确性**：spec 关键推导（树 mask 下各候选路径 attention 等价于独立序列验证）。

**draft 分类学**：独立小模型 / self-draft（Draft&Verify layer skip）/ Medusa heads / EAGLE feature / retrieval REST / n-gram——4.5 详展，4.4 分类表。REST（2311.08252，NAACL 2024）：datastore 检索 draft、免训练 plug-and-play。来源：rest、draft-verify、medusa、eagle。

**batch 效应**：接受率跨请求/位置方差 → 固定长度 verify 浪费 → DSpark 动机（DeepInfer doc L303 逐字："Because the acceptance probability of a draft token varies across requests and positions, fixed-length verification wastes verification compute on low-probability tokens"）——前瞻 7.2 + 回链。来源：deepinfer-doc。

## 4.5 投机解码系统实现

**EAGLE-1（2401.15077v3 摘要核实）**：feature（second-to-top-layer）级自回归比 token 级更直接；feature 不确定性约束性能 → **引入前移一时间步的 token 序列（shifted token embedding）** 解不确定性；LLaMA2-Chat 70B 2.7-3.5× 延迟加速、吞吐翻倍、分布保持。repo SafeAILab/EAGLE 覆盖三代。来源：eagle。

**EAGLE-2（2406.16858v2）**：接受率不仅依赖位置还**依赖上下文** → context-aware 动态 draft tree（confidence 剪枝），改进 EAGLE 静态树。来源：eagle2。

**EAGLE-3（2503.01840v3 + repo README）**：数据 scaling 下 EAGLE 收益受限（feature prediction 约束）→ **training-time test** 模拟 drafting + **低/中/高层语义特征融合**替代 top-layer feature；NeurIPS'25（repo README 2025.9.18）。来源：eagle3。

**Medusa（2401.10774 + repo README）**：多 decoding heads 预测多未来 token、原模型不动只训 heads（Medusa-1）/ 全模型训练 recipe（Medusa-2）；tree attention 组合候选 + **typical acceptance** 选最长合理前缀；2.2-3.6× 加速（README 更新版口径）。来源：medusa。

**MTP self-speculative**：models 4.5 MTP 目标 recap + 指引（模型侧不展开）；V4.1 config **num_nextn_predict_layers=3**（hf-mirror 本轮重取；V3 nextn 1 层对照见 deepseekv3 / models 4.5）；推理时 nextn 作 draft；vLLM 实现 v1/worker/gpu/spec_decode/**mtp/speculator.py**（tarball 核实）。来源：v41-flash-config、deepseekv3、vllm-repo。

**Draft & Verify（2309.08168，ACL 2024）**：self-speculative 两阶段——drafting 选择性跳过中间层（layer skip）+ 原模型单 forward verify；免辅助模型 / 免额外训练 / 免额外内存、输出一致。repo dilab-zju/self-speculative-decoding。来源：draft-verify。

**REST**：datastore 检索（4.4 已述），repo FasterDecoding/REST。来源：rest。

**引擎实现对照（tarball 核实，版本漂移重要）**：
- **vLLM**：v1/worker/gpu/spec_decode/{**eagle,ngram,mtp**}/speculator.py 三件均在（W0 纠正路径本轮再确认）；eagle/ 含 eagle3_utils.py；目录已大幅扩展（dspark/ dflash/ dflash2/ dfly/ suffix/ standalone_ar/ multi_module_mtp/ xpress 等 + acceptance_estimator.py / adaptive_verification.py / rejection_sampler.py）——4.5 只走读 eagle/ngram/mtp 三件（spec 大纲范围），其余一句「目录已扩展」+ 核实日期，全量走读归 5.4。旧 v1/spec_decode/ 仅 metrics/dynamic 辅助（W0 口径）。
- **SGLang**：srt/speculative/ 在；**eagle_worker.py 已重构为 eagle_worker_v2.py（class EAGLEWorkerV2(BaseSpecWorker)，L1564）+ eagle_worker_common.py + multi_layer_eagle_worker_v2.py**；eagle_{utils,info}.py / eagle_draft_cuda_graph_runner.py / eagle_draft_extend_cuda_graph_runner.py / ngram_worker.py / cpp_ngram/ / **dspark_components/**（dspark_draft.py / dspark_block_accept_estimator.py / dspark_config.py 等）/ dspark_disaggregation.py / spec_{info,utils}.py。**写作按新文件名锚定 + 漂移注**。
- **DeepInfer DSpark**：Confidence Head 变长 verify（doc L303 + 增益表：verified 5.00→3.2-4.4 约 -30%、accepted 3.29 vs 3.20、EP64/DP64 +4-7% @~60 conc/rank）——**4.5 只前瞻指针 + 回链，详展归 7.2**（spec 两层纪律）。来源：vllm-repo、sglang-repo、deepinfer-doc、deepinfer-repo。

**接受率测量与调优 / 调度交互**：γ 选择（Leviathan §3.5 数值搜索最优 γ）、batch 下接受率方差（DSpark 动机回链）；verify 是 prefill 形态 → 与 chunked prefill / PD 分离调度含义（spec 口径，推导无外部数值依赖）。

## 残留延迟项（writer / evaluator 须知）

1. **FlashMLA 版本漂移**：当前 main 无 Hopper MLA decode 代码——3.1 走读锚旧 commit ba89a34 + docs/20250422 deep-dive（仍在 repo）；当前 main 的内容（V4.1 sparse / fused kernel / Ascend）归 7.3 详展。evaluator 复核以锚定 commit 为准。
2. **DeepEP V1 vs V2.5**：spec 大纲 normal/low-latency 语义为 V1——机制讲解锚 tag v1.2.1，当前 main V2.5 统一 EPBuffer 作版本注；不得把 V1 两阶段语义写成当前 main API。
3. **vLLM CompilationMode 更名**：旧 CompilationLevel / PIECEWISE=3 不存在——3.4 / 5.3 按新名（NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE + CUDAGraphMode）+ 漂移注。
4. **SGLang 配置面迁移**：chunked_prefill_size 等已从 server_args.py 迁至 runtime_context get_schedule()；speculative 参数同理——4.3 / 4.5 引用配置时按新路径，勿写 server_args.chunked_prefill_size。
5. **SGLang eagle_worker_v2 更名 + schedule_policy 新增四策略**：按当前 main 文件名 / 策略全集写 + 漂移注（W0 五项为子集）。
6. **SpecInfer 代码在 inference 分支**：flexflow/FlexFlow 301 → flexflow/flexflow-train，master 无 spec 代码——引用 URL 用 flexflow/flexflow-train/tree/inference + AE repo；勿引 master 路径。
7. **FastServe [A] 升级依据**：LLMServe/FastServe artifact + NSDI'26 发表版标题与 arXiv 版不同（"FastServe: Iteration-Level Preemptive Scheduling..."）——引用注明两标题对应同一工作；spec 原 [B] 判「无官方 repo」已被推翻，evaluator 勿按 spec 旧 grade 判 S2。
8. **NanoFlow 吞吐口径**：摘要 50-72% of optimal vs README 68.5%——引用以摘要为准并注明 README 另一表述。
9. **huggingface.co 本机仍不可达**：V4.1 config 经 hf-mirror 重取（W0+W1+W2 三轮一致），引用注明传输口径。
10. **W3+ 延迟项不在本轮**：vLLM V1 重构 blog / hisparse / kv_hints 语义、SGLang prefill_delayer / shortest_prefill_load / hisparse_coordinator / mm_schedule 函数级语义、SGLang API speculative execution / dllm 形态（W3）；DeepInfer DualPath 公开性（W4）；TRT-LLM disagg（W5）；veRL / OpenRLHF（W6）。
11. **marlin 默认分支 master**（非 main）——codeload / 引用注意。
12. **Orca 36.9× / batch-invariance 博客 / OCP MX / NVFP4**：W1 已一手核实，本轮复用，writer 直接用勿再标 ⟨verify⟩。

## references.md 本轮变更清单

- **写入规则**：补「等级只取五个单值，不允许 [A/B] 类混合档；官方代码散落非默认分支 / 子模块 / AE repo 仍判 [A] + 注明路径」约定（SpecInfer 先例）。
- P3（16 条，全部更新，核实日期 2026-11-23）：flashinfer（cascade/split-KV/JIT/plan-run 路径 + 术语对应注）/ fa2 / fa3（算法级事实 + 性能数）/ gptq（H=2XX^T + 三步 + act-order）/ awq（grid search α）/ **smoothquant（补官方 repo URL mit-han-lab/smoothquant）**/ llm-int8（分解机制 + outlier 相变）/ tilelang / nanoflow（code 核实 + 吞吐口径注）/ **marlin（教学参照 → [A]，补 arXiv:2408.11743）**/ flashmla（版本漂移 + 旧 commit 锚）/ deepgemm（DeepJIT / UE8M0 / contiguous-masked / Mega MoE）/ deepep（V2.5 漂移 + v1.2.1 锚）/ deepselect（场景 + 性能口径）/ flashdecoding（作者更正 Dao et al. + 全文事实）/ nsa（[B] search 复核注）。
- P4（19 条，13 条更新 + 6 条共享锚点未动）：**fastserve（[B] → [A]，NSDI'26 URL + LLMServe/FastServe）**/ llumnix（v0/v1 版本注）/ sarathi（[A] 依据收窄至 sarathi scheduler provider）/ sarathi-serve（OSDI'24 自链）/ leviathan（Def 3.7 / Thm 3.8 / Eq 1 / Cor 3.9 逐字）/ chen / **specinfer（[A/B] → [A]，inference 分支 URL + AE + ASPLOS'24 + 更名注）**/ eagle / eagle2 / eagle3 / medusa / rest / draft-verify（机制事实 + venue）。top-p / min-p / xgrammar / outlines / guidance / batch-invariance 六个 P1 共享锚点未动（W1 已精炼）。
- 跨波共享锚点：vllm-repo（W2 路径核实注 + CompilationMode 漂移；被引 +3.1, 4.1, 4.5）/ sglang-repo（W2 路径 + 策略全集 + eagle_worker_v2 / 配置迁移漂移；被引 +4.1, 4.3, 4.5）/ deepinfer-doc（W2 compute-quota / Mega MoE / EP>32 / FlashMLA SASS / DSpark 逐字重核注；被引 +3.2——spec 3.2 FP8/FP4 部署对照引 doc）/ deepinfer-repo（tilelang_ops 路径核实注）/ tilekernels（算子清单核实注）更新。
- P1/P2 其余条目未动（orca / vllm / sglang / deepseekv3 / v41-flash-config / ocp-mx / nvidia-nvfp4 等共享锚点 W1 已核实、被引章节已含 W2 章号）。
- 新增 W2 精炼说明段。parse-check：**9 groups / 79 entries**（P1=11, P2=14, P3=16, P4=19, P5=3, P6=1, P7=8, P8=4, P9=3），无重复 key、无非法日期、无混合档。
