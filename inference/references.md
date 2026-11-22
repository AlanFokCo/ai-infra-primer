# references.md — inference 维度文献索引与术语表

本文件是 inference 维度全部章节的引用唯一事实源。核心规则：**先收录后引用**——条目由各波 research pass 核实后写入本文件，章节正文才能引用；`scripts/lint-chapter.sh` 检查项 4 强制执行：正文出现的 arXiv ID 在本文件 grep 不到即 lint FAIL。写作 agent 不新增条目，只引用已有条目。

## 写入规则

- 条目只由 research pass 写入；「核实日期」= 当次核实 ID / URL 的日期（YYYY-MM-DD）。spec 中 `⟨verify⟩` 标记的条目必须经 WebFetch 复核（arXiv abs 页 / HF config / repo 路径）后才收录。
- 非 arXiv 来源（blog、官方文档、repo 文件、无 arXiv 的经典文献）在「arXiv ID / URL」字段写完整 URL 或标准出处（如「USENIX OSDI'22」「Huawei Connect 2025-09 keynote」）。lint 只机械核对 arXiv ID，URL 条目的真实性由 research pass 与 evaluator 把关。
- 「证据等级」五档：[A] / [B] / [C] / [W] / 教学参照，定义见 STYLE_GUIDE §6；inference 维度口径细化见 inference spec §4.1：[A] = 正式 paper + official code（锚点是模型则另需 open weights）；[B] = 有 paper、无官方代码或代码不全；[C] = 闭源披露，含「开放技术博客、非闭源」子类（无 paper 无 repo 的官方 / 机构技术博客与官方规格披露，显式注明）；[W] = 开放权重 + config 但无正式 paper；教学参照 = 「教学参照 + 官方实现」（无独立论文的官方维护引擎 / 库）。DeepInfer 官方文档披露的生产数字挂教学参照锚点 + 显式「官方披露」口径注 + 核实日期，不升 [A]（无 paper）、不降 [C]（非媒体转述）。老经典无 arXiv 来源标注原始出处并注明「无 arXiv」。[W] 级找到正式 paper 后的回升判定同 STYLE_GUIDE §6。
- key 为小写 slug，全文件唯一，取论文或资源的通用短名（如 `vllm`、`distserve`）；paper 与 repo 分列条目时 repo 条目加 `-repo` 后缀（如 `vllm-repo`，先例 training `deepspeed-repo`）。
- 章节写完（或引用关系变化）后回写「被引章节」；一条目被多章引用时逗号分隔累加（如 `4.1, 4.2`）。
- 正文引用格式 `[Kwon et al., 2023]`，链接到本文件对应条目（inference spec §4.2 / STYLE_GUIDE §6）；章间引用一律纯文字，活链接唯一例外 `../references.md`。

## 条目 schema

每条目七个字段：

| 字段 | 含义 |
| --- | --- |
| key | 条目唯一 slug（小写） |
| 标题 | 论文 / 资源标题 |
| 作者 | 第一作者 et al.，或机构名 |
| arXiv ID / URL | arXiv 编号；非 arXiv 来源写完整 URL |
| 证据等级 | [A] / [B] / [C] / [W] / 教学参照 |
| 核实日期 | research pass 核实日期（YYYY-MM-DD） |
| 被引章节 | 引用本条目的章节号，逗号分隔 |

条目写法（表头 + 分隔行 + 数据行；下行示例同时是 lint grep 的格式基准）：

```
| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
```

## 条目

按 Part 分组；research pass 向对应 Part 的表追加，chapter → sources 映射体现在「被引章节」字段。

> **W0 骨架收录说明（2026-11-22）**：下列条目种入自 inference spec 附录「Spec 预核实记录」（arXiv API / GitHub contents API / hf-mirror 核实）与四路对抗审查销项记录，核实日期统一 2026-11-22；本次种入时 50 个 arXiv ID↔标题↔第一作者再经 arXiv API 批量复核，非 arXiv URL（Orca / FlashDecoding / batch-invariance / OCP MX / NVIDIA H100/H200 / hf-mirror config）经 HTTP 200 复核。GitHub 本机不可达，repo URL 沿用 spec 附录 2026-11-22 GitHub API 核实记录原值（例外：hybridflow repo URL 沿用 training references 已核实值；top-p / min-p / yoco / punica / s-lora repo URL 为 2026-11-23 终审修复补录，HTTP 200 核实）。`⟨verify⟩` 条目暂不收录，留各波 research pass 复核后追加。「被引章节」按 spec §5 大纲意图填写，各波 research / writing pass 回写精炼。跨 Part 共享锚点收录于首引 Part、被引章节跨组累加；引擎走读 Part 另立 repo 条目（`vllm-repo` / `sglang-repo`）。

> **W1 research pass 精炼说明（2026-11-23）**：P1/P2 全部条目经一手复核——19 个 arXiv ID↔标题↔第一作者经 arXiv API 批量再核（19/19 中）；GitHub 本机恢复可达，19 个 P1/P2 相关 repo 经 API 200 复核 + vllm / sglang / DeepInfer / 3FS / Mooncake / LMCache 六 repo codeload tarball 路径级核实；NVIDIA H100/H200/B200 精确 datasheet 数值经官方 PDF 一手取得（B200 ⟨verify⟩ 销项）；Ascend 950 经 Huawei 官方 keynote 全文 + hiascend 官方产品页核实（950PR 128 GB/1.6 TB/s 补齐）；NVFP4 格式经 cuBLAS / TensorRT-LLM 官方文档核实（新增 `nvidia-nvfp4` 条目）；新增 `sharegpt` 数据集条目（1.1 ShareGPT 归属与 Azure trace 分开）。⟨verify⟩ 销项与残留延迟项详见 `docs/research/inference-w1-sources.md`。P1/P2 被引章节按 spec §5 Sources 回写（vllm-repo +2.1/2.2、sglang-repo +1.1/2.4）；P3-P9 条目除跨波共享锚点（top-p / min-p / xgrammar / outlines / guidance / batch-invariance / vllm-repo / sglang-repo / deepinfer-doc / 3fs / deepseek-recipe）外未动。yoco / punica / s-lora 三个 W0 终审 repo URL 本轮 GitHub API 复核 200（条目本身未动，证据见 research notes）。

### Part 1 · 推理原理与性能账

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| vllm | Efficient Memory Management for Large Language Model Serving with PagedAttention（W1 原文核实：碎片上界原文「limits all the memory wastes for a request within one block」；论文 §7.2 默认 block size 16，非当前引擎默认 64） | Kwon et al. | arXiv:2309.06180 ; https://github.com/vllm-project/vllm | [A] | 2026-11-23 | 1.1, 1.2, 1.4, 2.1, 2.2, 4.1, 5.1, 5.2, 5.3, 5.4 |
| distserve | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（goodput 定义 W1 原文核实：per-GPU 归一 + SLO attainment 目标下最大请求率） | Zhong et al. | arXiv:2401.09670 ; https://github.com/LLMServe/DistServe | [A]（论文脚注自链官方实现，四路审查 C-S2-1 升 [A]；W1 摘要自链复核） | 2026-11-23 | 1.1, 1.3, 8.2, 9.1 |
| orca | Orca: A Distributed Serving System for Transformer-Based Generative Models（continuous batching 源头；iteration-level scheduling + selective batching：Attention 逐请求算、其余算子 token-wise batch——W1 OSDI'22 原文 PDF 核实；GPT-3 175B 同延迟水平 36.9× 吞吐 vs FasterTransformer） | Yu et al. | https://www.usenix.org/conference/osdi22/presentation/yu（USENIX OSDI'22，无 arXiv；PDF usenix.org/system/files/osdi22-yu.pdf） | [B]（有 paper 无官方代码） | 2026-11-23 | 1.1, 4.1 |
| splitwise | Splitwise: Efficient generative LLM inference using phase splitting（ISCA 2024；Azure trace 出处 W1 双重核实：论文原文「two Azure LLM inference services on November 11th 2023」+ 自链释放子集 Azure/AzurePublicDataset；characterization 用 20 分钟 code/conv 两 trace） | Patel et al. | arXiv:2311.18677 ; https://github.com/mutinifni/splitwise-sim | [B]（仅作者模拟器 + vLLM PR，代码不全） | 2026-11-23 | 1.1, 8.2 |
| deepseekv3 | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437 ; https://hf-mirror.com/deepseek-ai/DeepSeek-V3（开放权重，hf-mirror 传输核实，非 gated） | [A]（模型锚点：paper + open weights） | 2026-11-23 | 1.2, 3.2, 3.3 |
| v41-flash-config | DeepSeek-V4.1-Flash-1017 config.json（serving 相关字段：MLA 几何 / quantization_config / sliding_window / compress_ratios / indexer / nextn / dspark_* / engram_* / hc_* / vision_config；模型侧形式化归 models 维度，V4.1 待补） | DeepSeek-AI | https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json（huggingface.co 本机不可达，hf-mirror 传输核实；W1 2026-11-23 重取全字段与 spec 附录一致：kv_heads 1 / head_dim 512 / qk_rope 64 / q_lora_rank 1280 / o_lora_rank 1024 / o_groups 8 / 40L / 384 routed+1 shared top-6 / sliding_window 128 / index_topk 512 / index_n_heads 32 / index_head_dim 128 / nextn 3 / weight_block_size [32,32] / scale_fmt ue8m0 / expert_dtype fp4 / compress_ratios 43 项） | [W] | 2026-11-23 | 1.2, 2.1, 2.3, 2.4, 3.1, 3.2, 4.5, 7.1, 7.2, 7.4, 9.4 |
| nvidia-datasheet | NVIDIA H100 / H200 / B200 官方 datasheet（W1 PDF 一手核实，dense = 官方 sparse 表值之半：H100 SXM BF16 989.5 / FP8 1979 TFLOPS dense、80 GB HBM3、3.35 TB/s、NVLink 900 GB/s；H200 SXM 同算力、141 GB HBM3e、4.8 TB/s；B200（HGX B200，SM100）BF16 2250 / FP8 4500 / FP4 9000 TFLOPS dense、180 GB HBM3E、7.7 TB/s、NVLink5 1.8 TB/s；GB200 变体 186 GB、8 TB/s、FP4 10 PFLOPS dense——SKU 勿混；Hopper 无 FP4；B300/Rubin 勿混代际） | NVIDIA | https://www.nvidia.com/en-us/data-center/h200/ ; https://www.nvidia.com/en-us/data-center/h100/ ; https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23（H200/H100 datasheet PDF 官方落地页）; https://dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf（Blackwell architecture datasheet，B200/GB200 规格表） | [C]（官方规格披露，开放产品页 / datasheet、非闭源） | 2026-11-23 | 1.2, 1.3, 2.1, 2.2, 2.4 |
| ascend-950-keynote | Ascend 950 系规格（W1 官方一手核实：Huawei Connect 2025-09 keynote——950 系 1 PFLOPS FP8/MXFP8/HiF8、2 PFLOPS MXFP4/FP4、互联 2 TB/s（910C 的 2.5×）、SIMD+SIMT、访存粒度 512B→128B；950PR（prefill/推荐，HiBL 1.0）hiascend 官方页最大 128 GB、1.6 TB/s、灵衢 2.0 2 TB/s 双向、2026Q1；950DT（decode/训练，HiZQ 2.0）144 GB、4 TB/s、2026Q4；dense/sparse 口径 keynote 未注明；Atlas 350 为 950PR 基加速卡（2026-03 上市），卡级数与芯片级数须区分，媒体转述不作一手） | Huawei | https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech ; https://www.hiascend.com/hardware/processor（官方披露，无 arXiv） | [C]（官方 keynote / 产品页披露、非闭源） | 2026-11-23 | 1.3 |
| azure-llm-trace | Azure LLM Inference Dataset 2023（Azure/AzurePublicDataset AzureLLMInferenceDataset2023.md——.md 描述文件 + data/ CSV，非目录；归属 Splitwise ISCA 2024：两个 Azure LLM inference 服务（code/conv）2023-11-11 生产 trace，CC-BY；与 ShareGPT 分开标注） | Microsoft Azure | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md | 教学参照（官方数据集 repo） | 2026-11-23 | 1.1, 8.2 |
| genai-perf | genai-perf（triton-inference-server/perf_analyzer 下 genai-perf/；LLM serving benchmark 工具：output token throughput / TTFT / time to second token / ITL / request throughput；README 官方声明已停止新功能开发、后继工具 ai-dynamo/aiperf——W1 核实，写作须注明此状态） | NVIDIA | https://github.com/triton-inference-server/perf_analyzer ; https://github.com/ai-dynamo/aiperf（后继，README 自链） | 教学参照 | 2026-11-23 | 1.1 |
| sharegpt | ShareGPT_Vicuna_unfiltered（社区共享 ChatGPT 对话数据集；常用 benchmark 文件 ShareGPT_V3_unfiltered_cleaned_split.json；vLLM bench / SGLang bench serving 常用 workload 源，非 Splitwise Azure trace——归属分开标注） | 社区（anon8231489123，HF） | https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered（huggingface.co 本机不可达，hf-mirror 传输核实） | 教学参照（公开数据集、无 paper；benchmark 数据源，沿 azure-llm-trace 数据集条目先例） | 2026-11-23 | 1.1 |

### Part 2 · KV cache 系统

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| sglang | SGLang: Efficient Execution of Structured Language Model Programs（RadixAttention W1 原文核实：radix tree 存 KV + LRU 叶优先驱逐 + cache-aware 调度；论文 page = 1 token） | Zheng et al. | arXiv:2312.07104 ; https://github.com/sgl-project/sglang | [A] | 2026-11-23 | 2.2, 4.1, 6.1, 6.2, 6.3 |
| cachedattention | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention（layer-wise pre-loading + asynchronous saving 重叠 KV 访问与计算、scheduler-aware fetching/eviction、位置编码解耦截断——W1 原文核实） | Gao et al. | arXiv:2403.19708 | [B]（无官方 repo——W1 复核：GitHub search 0 结果、论文无代码链接） | 2026-11-23 | 2.2, 2.4 |
| cacheblend | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | Yao et al. | arXiv:2405.16444 ; https://github.com/YaoJiayi/CacheBlend | [A] | 2026-11-23 | 2.2, 2.4 |
| kivi | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache（K per-channel / V per-token + residual 全精度窗口——W1 原文核实） | Liu et al. | arXiv:2402.02750 ; https://github.com/jy-yuan/KIVI | [A] | 2026-11-23 | 2.3 |
| kvquant | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization（per-channel Key + Pre-RoPE Key 量化 + NUQ + per-vector dense-and-sparse outlier 隔离；1M ctx 单 A100-80GB / 10M 8-GPU——W1 摘要核实） | Hooper et al. | arXiv:2401.18079 ; https://github.com/SqueezeAILab/KVQuant | [A] | 2026-11-23 | 2.3 |
| h2o | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models（heavy-hitter + recent 动态平衡、dynamic submodular 保证——W1 摘要核实） | Zhang et al. | arXiv:2306.14048 ; https://github.com/FMInference/H2O | [A] | 2026-11-23 | 2.3 |
| streamingllm | Efficient Streaming Language Models with Attention Sinks（attention sink：初始 token KV + recent window——W1 摘要核实） | Xiao et al. | arXiv:2309.17453 ; https://github.com/mit-han-lab/streaming-llm | [A] | 2026-11-23 | 2.3 |
| snapkv | SnapKV: LLM Knows What You are Looking for Before Generation（prompt 末端 observation window 选 per-head 重要 KV 位置——W1 摘要核实） | Li et al. | arXiv:2404.14469 ; https://github.com/FasterDecoding/SnapKV | [A] | 2026-11-23 | 2.3 |
| mooncake | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（含 transfer engine：Segment / BatchTransfer 抽象 + 拓扑感知路径选择 + 多网卡池化，docs/source/design/transfer-engine/——W1 repo 核实；原 MoonshotAI/Mooncake 更名 kvcache-ai/Mooncake，旧 URL 404 无重定向） | Qin et al. | arXiv:2407.00079 ; https://github.com/kvcache-ai/Mooncake | [A]（transfer engine 继承本等级，四路审查 B-8） | 2026-11-23 | 2.4, 6.3, 8.2, 8.3 |
| cachegen | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving（量化 + 逐层逐通道算术编码压缩 bitstream、带宽自适应压缩级——W1 原文核实；3.5-4.3× KV 压缩 / 3.2-3.7× 取回延迟降低） | Liu et al. | arXiv:2310.07240 ; https://github.com/UChi-JCL/CacheGen | [A] | 2026-11-23 | 2.4 |
| preble | Preble: Efficient Distributed Prompt Scheduling for LLM Serving | Srivatsa et al. | arXiv:2407.00023 ; https://github.com/WukLab/preble | [A] | 2026-11-23 | 2.2 |
| lmcache | LMCache/LMCache repo（KV cache 层；W1 路径核实（dev 分支）：CacheGen 压缩 kernels 在 repo（csrc/cuda/cachegen_kernels.cuh + lmcache/v1/platform/torch_ops/cachegen_kernels.py），官方 docs/source/mp/cachegen.rst 注明 multiprocess 模式 CacheGen 尚未实现、原始实现维护于 LMCache/LMCache_inprocess；CacheBlend 集成 lmcache/v1/multiprocess/modules/blend/；vLLM / SGLang connector lmcache/integration/{vllm,sglang}/） | LMCache | https://github.com/LMCache/LMCache | 教学参照 | 2026-11-23 | 2.4, 8.4 |
| ocp-mx | OCP Microscaling Formats (MX) Specification v1.0（Date Sep 2023；W1 PDF 一手核实：MXFP8 / MXFP6 / MXFP4 / MXINT8 均 block size 32 + E8M0 scale（8 bit 无符号纯指数）；MXFP4 元素 FP4 E2M1） | OCP | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf（无 arXiv；URL 200，PDF 16 页） | [C]（官方格式规范，开放标准、非闭源） | 2026-11-23 | 2.3, 3.2 |
| nvidia-nvfp4 | NVIDIA NVFP4 格式官方文档（E2M1 元素 + per-16 UE4M3 block scale + tensor-wide FP32 global scale，Blackwell SM100；cuBLAS CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3（每 16 元素一个 8-bit UE4M3 scale）+ SCALAR_32F 全局 scale；TensorRT-LLM kv_cache dtype='nvfp4'；DeepInfer 主 KV per-16 e4m3「类 NVFP4 无 global scale」对照见 deepinfer-doc；blog 正文本机反爬不可达（URL HTTP 200），格式事实经 cuBLAS / TRT-LLM 官方文档一手核实） | NVIDIA | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ ; https://docs.nvidia.com/cuda/cublas/index.html（block scaling 节）; https://nvidia.github.io/TensorRT-LLM/features/quantization.html | [C]（官方格式文档，开放文档、非闭源） | 2026-11-23 | 2.3, 3.2 |

### Part 3 · 推理算子

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| flashinfer | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | Ye et al. | arXiv:2501.01005 ; https://github.com/flashinfer-ai/flashinfer | [A] | 2026-11-22 | 3.1, 5.3 |
| fa2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | arXiv:2307.08691 ; https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-22 | 3.1 |
| fa3 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Shah et al. | arXiv:2407.08608 ; https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-22 | 3.1 |
| gptq | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | arXiv:2210.17323 ; https://github.com/IST-DASLab/gptq | [A] | 2026-11-22 | 3.2 |
| awq | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | arXiv:2306.00978 ; https://github.com/mit-han-lab/llm-awq | [A] | 2026-11-22 | 3.2 |
| smoothquant | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | Xiao et al. | arXiv:2211.10438 | [A] | 2026-11-22 | 3.2 |
| llm-int8 | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers et al. | arXiv:2208.07339 ; https://github.com/TimDettmers/bitsandbytes | [A] | 2026-11-22 | 3.2 |
| tilelang | TileLang: A Composable Tiled Programming Model for AI Systems | Wang et al. | arXiv:2504.17577 ; https://github.com/tile-ai/tilelang | [A] | 2026-11-22 | 3.4, 7.3 |
| nanoflow | NanoFlow: Towards Optimal Large Language Model Serving Throughput | Zhu et al. | arXiv:2408.12757 ; https://github.com/efeslab/Nanoflow | [A] | 2026-11-22 | 3.4, 6.2 |
| marlin | IST-DASLab/marlin repo（W4A16 mixed-input GEMM；README ~4× 带宽节省口径） | IST-DASLab | https://github.com/IST-DASLab/marlin | 教学参照 | 2026-11-22 | 3.3 |
| flashmla | deepseek-ai/FlashMLA repo（MLA decode kernel；open vs internal 5-10% decode 差距见 7.3 官方披露） | DeepSeek-AI | https://github.com/deepseek-ai/FlashMLA | 教学参照 | 2026-11-22 | 3.1, 7.3 |
| deepgemm | deepseek-ai/DeepGEMM repo（FP8 fine-grained scaling GEMM、JIT、UE8M0；含 Mega MoE 组件——README 口径） | DeepSeek-AI | https://github.com/deepseek-ai/DeepGEMM | 教学参照 | 2026-11-22 | 3.3, 7.3 |
| deepep | deepseek-ai/DeepEP repo（MoE dispatch / combine：normal 模式 NVLink + RDMA 两阶段 vs low-latency 模式纯 RDMA hook） | DeepSeek-AI | https://github.com/deepseek-ai/DeepEP | 教学参照 | 2026-11-22 | 3.3, 7.3, 7.4 |
| deepselect | deepseek-ai/DeepSelect repo（DSA top-k 选择 kernel） | DeepSeek-AI | https://github.com/deepseek-ai/DeepSelect | 教学参照 | 2026-11-22 | 3.1, 3.3, 7.3 |
| flashdecoding | Flash-Decoding for long-context inference（CRFM Stanford 博客，split-KV 源头） | Stanford CRFM | https://crfm.stanford.edu/2023/10/12/flashdecoding.html（无 arXiv） | [C]（开放技术博客、非闭源） | 2026-11-22 | 3.1 |
| nsa | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention（sparse attention serving 背景；模型侧归 models 4.2） | Yuan et al. | arXiv:2502.11089 | [B] | 2026-11-22 | 3.1 |

### Part 4 · 调度与 batching

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| fastserve | Fast Distributed Inference Serving for Large Language Models | Wu et al. | arXiv:2305.05920 | [B]（无官方 repo） | 2026-11-22 | 4.2 |
| llumnix | Llumnix: Dynamic Scheduling for Large Language Model Serving | Sun et al. | arXiv:2406.03243 ; https://github.com/llumnix-project/llumnix-ray | [A]（原 AlibabaPAI/llumnix，301 更名） | 2026-11-22 | 4.2, 8.5 |
| sarathi | SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | Agrawal et al. | arXiv:2308.16369 ; https://github.com/microsoft/sarathi-serve | [A] | 2026-11-22 | 4.3 |
| sarathi-serve | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | Agrawal et al. | arXiv:2403.02310 ; https://github.com/microsoft/sarathi-serve | [A] | 2026-11-22 | 4.3, 8.2 |
| leviathan | Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | arXiv:2211.17192 | [B]（Google 无官方开源） | 2026-11-22 | 4.4 |
| chen | Accelerating Large Language Model Decoding with Speculative Sampling | Chen et al. | arXiv:2302.01318 | [B]（DeepMind 无官方开源） | 2026-11-22 | 4.4 |
| specinfer | SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification | Miao et al. | arXiv:2305.09781 ; https://github.com/goliaro/specinfer-ae | [A/B]（AE + FlexFlow 集成，实现散落） | 2026-11-22 | 4.4 |
| eagle | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Li et al. | arXiv:2401.15077 ; https://github.com/SafeAILab/EAGLE | [A] | 2026-11-22 | 4.5 |
| eagle2 | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | Li et al. | arXiv:2406.16858 ; https://github.com/SafeAILab/EAGLE | [A] | 2026-11-22 | 4.5 |
| eagle3 | EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test | Li et al. | arXiv:2503.01840 ; https://github.com/SafeAILab/EAGLE | [A] | 2026-11-22 | 4.5 |
| medusa | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Cai et al. | arXiv:2401.10774 ; https://github.com/FasterDecoding/Medusa | [A] | 2026-11-22 | 4.5 |
| rest | REST: Retrieval-Based Speculative Decoding | He et al. | arXiv:2311.08252 ; https://github.com/FasterDecoding/REST | [A] | 2026-11-22 | 4.4, 4.5 |
| draft-verify | Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding | Zhang et al. | arXiv:2309.08168 ; https://github.com/dilab-zju/self-speculative-decoding | [A] | 2026-11-22 | 4.5 |
| top-p | The Curious Case of Neural Text Degeneration（top-p / nucleus sampling 出处） | Holtzman et al. | arXiv:1904.09751 ; https://github.com/ari-holtzman/degen | [A]（官方 repo 2026-11-23 核实，终审升 [A]；W1 GitHub API 复核 200） | 2026-11-23 | 1.4 |
| min-p | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | Nguyen et al. | arXiv:2407.01082 ; https://github.com/menhguin/minp_paper | [A]（第一作者官方代码 2026-11-23 核实，终审升 [A]；W1 GitHub API 复核 200） | 2026-11-23 | 1.4 |
| xgrammar | XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models | Dong et al. | arXiv:2411.15100 ; https://github.com/mlc-ai/xgrammar | [A] | 2026-11-23 | 1.4, 5.4 |
| outlines | Efficient Guided Generation for Large Language Models（Outlines） | Willard & Louf | arXiv:2307.09702 ; https://github.com/dottxt-ai/outlines | [A]（原 outlines-dev/outlines，301 更名可跟） | 2026-11-23 | 1.4 |
| guidance | guidance-ai/guidance repo（结构化生成库；无独立论文） | Guidance | https://github.com/guidance-ai/guidance | 教学参照 | 2026-11-23 | 1.4, 5.4 |
| batch-invariance | Defeating Nondeterminism in LLM Inference（Thinking Machines 博客，2025-09-10，batch-invariant inference 源头；浮点非结合 → reduction order 随 batch 组成变化 → batch-invariant RMSNorm / matmul / attention kernels——W1 全文核实） | Horace He（Thinking Machines） | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | [C]（开放技术博客、非闭源） | 2026-11-23 | 1.4, 3.4 |

### Part 5 · vLLM 走读

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| vllm-repo | vllm-project/vllm repo（V1 模块树 vllm/v1/{engine,core,executor,worker,attention,sample,structured_output,kv_offload,simple_kv_offload,metrics}；spec_decode 实现在 v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py；kv_cache_interface / kv_cache_layout 在 v1/ 根目录——2026-11-22 路径核实；W1 2026-11-23 codeload tarball 复核：2.1 机制片段 v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager}.py 均在；CacheConfig.DEFAULT_BLOCK_SIZE=64（config/cache.py L185）；kv_offload/（cpu/ + tiering/ + fs/ 通用 offload 框架）与 simple_kv_offload/（SimpleCPUOffloadConnector）均在 v1/ 下；bench 入口 vllm/benchmarks/{serve,latency,throughput}.py + CLI vllm/entrypoints/cli/benchmark/；batch-invariant 在 model_executor/determinism/{batch_invariant,batch_invariant_configs}.py） | vLLM | https://github.com/vllm-project/vllm | [A]（2309.06180 官方实现） | 2026-11-23 | 1.1, 1.4, 2.1, 2.2, 2.4, 3.4, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 8.4, 9.3, 9.4 |
| punica | Punica: Multi-Tenant LoRA Serving | Chen et al. | arXiv:2310.18547 ; https://github.com/punica-ai/punica | [A]（官方实现存在性已核实——四路审查销项；repo URL 2026-11-23 终审补录） | 2026-11-22 | 5.4 |
| s-lora | S-LoRA: Serving Thousands of Concurrent LoRA Adapters | Sheng et al. | arXiv:2311.03285 ; https://github.com/S-LoRA/S-LoRA | [A]（官方实现存在性已核实——四路审查销项；repo URL 2026-11-23 终审补录） | 2026-11-22 | 5.4 |

### Part 6 · SGLang 走读

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| sglang-repo | sgl-project/sglang repo（srt/{managers,mem_cache,disaggregation,eplb,elastic_ep,batch_overlap,batch_invariant_ops,constrained,connector,checkpoint_engine,dllm}；PD KV 传输接口在 disaggregation/base/conn.py，connector/ 为权重 / 文件存储——2026-11-22 路径核实；W1 2026-11-23 codeload tarball 复核：HiCache = L1 GPU / L2 host / L3 storage 三层（mem_cache/hicache_storage.py + hybrid_cache/hybrid_cache_controller.py + memory_pool_host.py + storage/{hf3fs,mooncake_store,nixl,lmcache,file,...}，docs/docs/advanced_features/hicache_design.mdx）；HiSparse = DSA 系模型 decode 侧 top-k hot buffer + host 全量 KV（mem_cache/hisparse_memory_pool.py + managers/hisparse_coordinator.py，docs/docs/advanced_features/hisparse_guide.mdx）；HiRadixCache 类已重构为 UnifiedRadixCache（unified_radix_cache.py）+ RadixCache（radix_cache.py）；bench_serving 已迁 python -m sglang.benchmark.serving（python/sglang/benchmark/serving.py）；sampling 在 srt/layers/sampler.py） | SGLang | https://github.com/sgl-project/sglang | [A]（2312.07104 官方实现） | 2026-11-23 | 1.1, 1.4, 2.2, 2.4, 3.4, 4.2, 6.1, 6.2, 6.3, 8.1, 8.4, 9.3, 9.4 |

### Part 7 · DeepInfer 与 DeepSeek 栈

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| deepinfer-repo | deepseek-ai/DeepInfer repo（DeepSeek 生产引擎开源版，V4.1-Flash only，SM100 + Ascend 950，Apache-2.0；README + deepinfer/ 模块树；release 边界三项见 README） | DeepSeek-AI | https://github.com/deepseek-ai/DeepInfer | 教学参照 | 2026-11-22 | 1.4, 3.4, 4.5, 7.1, 7.2, 7.3, 9.4 |
| deepinfer-doc | DeepInfer doc/inference_deployment_qa.md（官方部署 QA 全文；生产数字官方披露口径：155T tokens/day、38 USD/GPU-hour 收入、moe_lb 实测、DSpark 增益表、TTL、存储 sizing、autoscaling 等；含官方知乎专栏链接——URL 在 doc 内，知乎原文留 W4 research pass 调取；W1 2026-11-23 codeload tarball 全文重读，P1/P2 相关事实（TTL agent 72h/chatbot 7d 近乎无损、chatbot 3d 反事实 0.5-1% 命中损失、主 KV per-16 e4m3 / indexer per-32 ue8m0 MXFP4、3FS 100 PiB / EC 10+2 / GC LRU、GDR swap-in 不占 SM、~80% perfectly balanced offline、TBT 用法）逐条与 spec 附录一致） | DeepSeek-AI | https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md | 教学参照（官方披露口径注：不升 [A]、不降 [C]） | 2026-11-23 | 1.1, 1.2, 1.4, 2.2, 2.3, 2.4, 3.3, 4.3, 6.2, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.4 |
| eplb | deepseek-ai/EPLB repo（expert parallelism load balancer：冗余 expert 副本 + 统计窗 + 重分配算法） | DeepSeek-AI | https://github.com/deepseek-ai/EPLB | 教学参照 | 2026-11-22 | 7.4 |
| 3fs | deepseek-ai/3FS repo（Fire-Flyer File System；DeepInfer kvcache-store 专用存储集群；W1 2026-11-23 tarball 核实：USRBIO = User Space Ring Based IO——Iov 共享内存零拷贝（IB 注册由 FUSE 进程管理）+ Ior io_uring 式共享环（io_depth 批处理），API 参考 src/lib/api/UsrbIo.md；FUSE 路径瓶颈：kernel-user 拷贝 + 共享队列自旋锁，~400K 4KiB reads/s（docs/design_notes.md）；开源 repo 存储侧为 CRAQ 链式复制（write-all-read-any），DeepInfer doc 的 10+2 EC 与「11 节点 any 10」读路径为生产官方披露、开源 repo 无 EC 实现——引用时区分口径） | DeepSeek-AI | https://github.com/deepseek-ai/3FS | 教学参照 | 2026-11-23 | 2.4 |
| deepseek-recipe | deepseek-ai/deepseek-recipe repo（Rust 多协议 Chat Completions / Responses / Messages 统一对话格式、special-token 注入防护、分段并行 BPE、流式状态机 + 增量 KMP + DSML tool args） | DeepSeek-AI | https://github.com/deepseek-ai/deepseek-recipe | 教学参照 | 2026-11-23 | 1.4, 7.4 |
| tilekernels | deepseek-ai/TileKernels repo（DeepInfer custom TileLang kernel 实例） | DeepSeek-AI | https://github.com/deepseek-ai/TileKernels | 教学参照 | 2026-11-22 | 3.4, 7.3 |
| yoco | You Only Cache Once: Decoder-Decoder Architectures for Language Models（YOCO 模型侧源头，指引不展开） | Sun et al. | arXiv:2405.05254 ; https://github.com/microsoft/unilm/tree/master/YOCO | [A]（官方代码目录 2026-11-23 核实，终审升 [A]） | 2026-11-22 | 7.2 |
| hyper-connections | Hyper-Connections（V4.1 hc_* 字段模型侧背景；模型侧形式化归 models 维度，V4.1 待补） | Zhu et al. | arXiv:2409.19606 | [B] | 2026-11-22 | 7.1 |

### Part 8 · 分布式推理与 PD 分离

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| loongserve | LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism | Wu et al. | arXiv:2404.09526 ; https://github.com/LoongServe/LoongServe | [A]（论文自链） | 2026-11-22 | 8.1 |
| memserve | MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool | Hu et al. | arXiv:2406.17565 | [B]（无官方 repo） | 2026-11-22 | 8.4 |
| nixl | ai-dynamo/nixl repo（NIXL：UCX / GDR / RDMA 传输抽象、agent API、descriptor 语义） | NVIDIA | https://github.com/ai-dynamo/nixl | 教学参照 | 2026-11-22 | 6.3, 8.3, 8.4 |
| dynamo | ai-dynamo/dynamo repo（planner、router、KV-aware 调度、disagg 编排） | NVIDIA | https://github.com/ai-dynamo/dynamo | 教学参照 | 2026-11-22 | 8.4 |

### Part 9 · 生产全流程

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| tensorrt-llm | NVIDIA/TensorRT-LLM repo（Python API 定义 + TRT 编译器路线、inflight batching；disagg 实现形态留 W5 research pass） | NVIDIA | https://github.com/NVIDIA/TensorRT-LLM | 教学参照 | 2026-11-22 | 8.4, 9.2 |
| lmdeploy | LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind（EuroSys 2027） | Zhang et al. | arXiv:2508.15601 ; https://github.com/InternLM/lmdeploy | [A]（四路审查 C-S2-2 升 [A]，2026-11-22 arXiv API 核实） | 2026-11-22 | 9.2 |
| hybridflow | HybridFlow: A Flexible and Efficient RLHF Framework（veRL） | Sheng et al. | arXiv:2409.19256 ; https://github.com/verl-project/verl | [A]（repo URL 沿用 training references 已核实值） | 2026-11-22 | 9.3 |

## 术语表

全书术语累积表：章节首现概念按「中文（English Term）」格式处理（STYLE_GUIDE §1）后在此登记——「中文处理」记录该术语的全书统一写法（译名或保留英文），保证跨章一致。各波次追加。W0 骨架先种核心 serving 术语，「首次出现章节」按 spec §5 大纲意图填写，writing pass 回写精炼；符号约定（B / b / κ / α / γ / 𝒢 等）以 inference spec §4.3 统一表为准，不在本表重复。

| 英文术语 | 中文处理 | 首次出现章节 |
| --- | --- | --- |
| TTFT (time to first token) | 首 token 延迟（TTFT）；缩写保留英文 | 1.1 |
| TPOT (time per output token) | 每输出 token 延迟（TPOT）；DistServe 用法 | 1.1 |
| TBT (time between tokens) | token 间延迟（TBT）；DeepInfer 用法，与 TPOT 对照 | 1.1 |
| ITL (inter-token latency) | token 间延迟（ITL） | 1.1 |
| goodput | SLO 达标吞吐（goodput）；保留英文。定义要件 = per-GPU 归一 + SLO attainment 目标（DistServe 口径），不得写成「达标请求的裸吞吐」 | 1.1 |
| SLO (service level objective) | 服务等级目标（SLO） | 1.1 |
| prefill / decode | 保留英文（prefill / decode） | 1.1 |
| arithmetic intensity | 算术强度 | 1.2 |
| critical batch | 临界 batch（compute-bound 临界 B*） | 1.2 |
| MHA / MQA / GQA | 保留英文（MHA / MQA / GQA） | 1.2 |
| MLA (multi-head latent attention) | 保留英文（MLA）；低秩推导归 models 4.1 | 1.2 |
| DSA (DeepSeek sparse attention) | 保留英文（DSA）；lightning indexer 模型侧归 models 4.2 | 1.2 |
| Wide-EP | 保留英文（Wide-EP，大 e） | 1.2 |
| roofline | 保留英文（roofline） | 1.3 |
| ridge point | 保留英文（ridge point） | 1.3 |
| M/M/1 queue | 保留英文（M/M/1）；逗留时间 W 与纯排队等待 W_q 区分 | 1.3 |
| Kingman VUT approximation | 保留英文（Kingman VUT）；近似的是 W_q | 1.3 |
| top-k / top-p / min-p | 保留英文（top-k / top-p / min-p 采样） | 1.4 |
| logits processor | 保留英文（logits processor） | 1.4 |
| structured output | 结构化输出（structured output） | 1.4 |
| grammar / FSM | 保留英文（grammar / FSM） | 1.4 |
| batch invariance | 批不变性（batch invariance）；正文保留英文 | 1.4 |
| incremental detokenization | 增量 detokenization（incremental detokenization） | 1.4 |
| PagedAttention | 保留英文（PagedAttention） | 2.1 |
| block table | 块表（block table） | 2.1 |
| block size | 块大小（block size）；符号 b 专归 block size（token 数） | 2.1 |
| copy-on-write (COW) | 写时复制（COW） | 2.1 |
| internal / external fragmentation | 内部 / 外部碎片 | 2.1 |
| SWA (sliding window attention) | 保留英文（SWA） | 2.1 |
| prefix caching | 前缀缓存（prefix caching）；正文保留英文 | 2.2 |
| RadixAttention | 保留英文（RadixAttention） | 2.2 |
| radix tree | 基数树（radix tree） | 2.2 |
| cache-aware scheduling | 缓存感知调度（cache-aware scheduling） | 2.2 |
| KV quantization | KV 量化（KV quantization） | 2.3 |
| eviction | 驱逐（eviction） | 2.3 |
| attention sink | 保留英文（attention sink） | 2.3 |
| heavy hitter | 保留英文（heavy hitter） | 2.3 |
| MXFP8 / NVFP4 / FP8 | 保留英文（低精度格式名）；格式账正式交付 3.2 | 2.3 |
| KV offload | 保留英文（KV offload） | 2.4 |
| tiered storage | 分层存储（tiered storage） | 2.4 |
| GDR (GPUDirect RDMA) | 保留英文（GDR） | 2.4 |
| transfer engine | 保留英文（transfer engine） | 2.4 |
| split-KV / FlashDecoding | 保留英文（split-KV / FlashDecoding） | 3.1 |
| cascade attention | 保留英文（cascade attention） | 3.1 |
| weight quantization | 权重量化（weight quantization） | 3.2 |
| GPTQ / AWQ / SmoothQuant | 保留英文（算法名） | 3.2 |
| fused MoE | 保留英文（fused MoE） | 3.3 |
| grouped GEMM | 保留英文（grouped GEMM） | 3.3 |
| dispatch / combine | 保留英文（dispatch / combine） | 3.3 |
| operator fusion | 算子融合（operator fusion） | 3.4 |
| CUDA Graph | 保留英文（CUDA Graph）；full graph vs piecewise 区分 | 3.4 |
| continuous batching | 连续批处理（continuous batching）；正文保留英文 | 4.1 |
| iteration-level scheduling | 迭代级调度（iteration-level scheduling） | 4.1 |
| selective batching | 保留英文（selective batching） | 4.1 |
| preemption | 抢占（preemption）；recompute vs swap 两策略 | 4.2 |
| MLFQ | 保留英文（MLFQ） | 4.2 |
| live migration | 在线迁移（live migration） | 4.2 |
| chunked prefill | 保留英文（chunked prefill） | 4.3 |
| mixed batch | 混合批（mixed batch） | 4.3 |
| speculative decoding | 投机解码（speculative decoding） | 4.4 |
| draft / verify | 保留英文（draft / verify） | 4.4 |
| acceptance rate | 接受率（acceptance rate）；符号 α | 4.4 |
| tree verification / tree attention | 保留英文（tree verification / tree attention） | 4.4 |
| MTP self-speculative | 保留英文（MTP self-speculative）；模型侧目标归 models 4.5 | 4.5 |
| LoRA serving | 保留英文（LoRA） | 5.4 |
| overlap scheduler | 保留英文（overlap scheduler） | 6.2 |
| zero-overhead scheduling | 零开销调度（zero-overhead scheduling） | 6.2 |
| EPLB | 保留英文（EPLB，expert parallelism load balancer） | 7.4 |
| TP / PP / EP / DP / CP | 保留英文（并行度缩写，沿用 training） | 8.1 |
| DP attention | 保留英文（DP attention） | 8.1 |
| PD disaggregation | PD 分离（PD disaggregation） | 8.2 |
| xPyD | 保留英文（xPyD 配比）；实例计数用 n_P / n_D | 8.2 |
| colocated | 保留英文（colocated，P/D 同卡部署） | 8.2 |
| NIXL | 保留英文（NIXL） | 8.3 |
| autoscaling | 弹性伸缩（autoscaling） | 8.5 |
| capacity planning | 容量规划（capacity planning） | 9.1 |
| RL rollout | 保留英文（RL rollout） | 9.3 |
| partial / async rollout | 保留英文（partial rollout / async rollout） | 9.3 |
| VE (vision encoder) 分离 | 保留英文（VE） | 9.4 |
| embedding / reranker serving | 保留英文（embedding / reranker） | 9.4 |
