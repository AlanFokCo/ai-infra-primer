# W1 调研笔记：P1 推理原理与性能账 + P2 KV cache 系统 事实源

调研日期：2026-11-23（W1 research pass）。本文件是 W1 八章（1.1–1.4、2.1–2.4）写作的事实源之一；写作 agent 不做额外调研，数值一律以本文件与 `inference/references.md` 为准。引用 key 白名单 = references.md P1/P2 全部条目 + 被引章节含 W1 章号的跨波锚点（top-p / min-p / xgrammar / outlines / guidance / batch-invariance / vllm-repo / sglang-repo / deepinfer-doc / 3fs / deepseek-recipe）。

## 访问路径记录（后续 research pass 复用）

- arXiv：`https://export.arxiv.org/api/query?id_list=<逗号分隔>&max_results=N`（https，Atom XML，含标题/作者/摘要）——本轮 19 个 P1/P2 ID 批量一次取回，19/19 与 references 一致。全文 HTML：`https://arxiv.org/html/<id>v<N>`（本轮 2311.18677v2 / 2401.09670v3 / 2309.06180v1 / 2312.07104v2 / 2403.19708v3 / 2402.02750v2 / 2310.07240v6 均 200）。
- **GitHub 本机本轮恢复可达**（W0 时不可达）：`api.github.com/repos/<org>/<repo>` 200；contents API 可列目录/取单文件（`Accept: application/vnd.github.raw`）。未鉴权 core 限额 60/h——深路径核实改用 **codeload tarball**：`https://codeload.github.com/<org>/<repo>/tar.gz/refs/heads/<branch>`（不占 API 限额），本轮取 vllm(main) / sglang(main) / DeepInfer(main) / 3FS(main) / Mooncake(main) / LMCache(**dev**，main 404) 六个。raw.githubusercontent.com 仍不可达。
- HF：huggingface.co 本机仍不可达，一律 `https://hf-mirror.com/<org>/<model>/raw/main/config.json` 与 `https://hf-mirror.com/api/{models,datasets}/<org>/<name>`。
- **NVIDIA 网页反爬**：www.nvidia.com / developer.nvidia.com 产品页与 blog 对 curl 返回 headers 200 但 body 挂起（0 字节超时）；WebFetch 亦被域名验证拦截。**可行路径**：`resources.nvidia.com` 落地页（200，HTML 内嵌 dam-cdn PDF 链接）→ `https://dam-cdn.nvd.orangelogic.com/AssetLink/<hash>.pdf`（NVIDIA 官方 DAM CDN，直接可下）；`images.nvidia.com/aem-dam/...pdf` 直接可下；`docs.nvidia.com/cuda/cublas/index.html` 可下；`nvidia.github.io/TensorRT-LLM/...`（GitHub Pages）可下。本轮 H100/H200/Blackwell 三份官方 datasheet PDF 均一手取得。
- OCP：opencompute.org 带浏览器 UA 反而 403，**裸 curl（无 -A）200**，PDF 直接可下。
- Huawei：`www.huawei.com/en/news/2025/9/hc-xu-keynote-speech`（keynote 全文）与 `www.hiascend.com/hardware/processor`（官方产品页）curl + 浏览器 UA 均 200。
- USENIX：Orca 页面 + PDF（`usenix.org/system/files/osdi22-yu.pdf`）均 200，pdftotext 可用（/opt/homebrew/bin/pdftotext）。
- thinkingmachines.ai 博客 200，全文可取。

## ⟨verify⟩ 销项总表（W1 范围）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 1.1 Azure trace 归属 | **销**：Splitwise 释放，两个 Azure LLM inference 服务 2023-11-11 生产 trace | 论文原文（2311.18677v2 §3）+ AzureLLMInferenceDataset2023.md 双重一手 |
| 1.1 genai-perf repo/tool | **销**：perf_analyzer/genai-perf/ 存在；**新发现：官方 README 声明停止新功能开发，后继 ai-dynamo/aiperf** | contents API + README 原文 |
| 1.1 Orca 机制细节 | **销**：iteration-level scheduling + selective batching（Attention 逐请求、其余 token-wise batch）；36.9× 吞吐数字一并取得（4.1 ⟨verify⟩ 提前销） | USENIX 摘要 + osdi22-yu.pdf 原文 |
| 1.3 B200(SM100) 精确 datasheet | **销**：官方 Blackwell architecture datasheet PDF 规格表（sparse/dense 脚注齐全） | dam-cdn.nvd.orangelogic.com PDF 一手 |
| 1.3 H100 SXM / H200 复核 | **销**：官方 datasheet PDF 与 W0 产品页核实值一致（dense = sparse 表值之半） | resources.nvidia.com hpc-datasheet-sc23 落地页内嵌 PDF |
| 1.3 Ascend 950 精确规格 | **部分销**：keynote 全文 + hiascend 官方页取得芯片级官方数（950PR 128 GB/1.6 TB/s 补齐）；dense/sparse 口径官方未注明；Atlas 350 卡级官方 datasheet 未获得（媒体数不作一手）——残留见下 | huawei.com keynote + hiascend.com |
| 1.4/2.3 batch-invariance 博客 canonical URL | **销**：URL 200 + 全文核实（Horace He，2025-09-10） | thinkingmachines.ai 一手 |
| 1.4/2.3 OCP MX 规范 URL | **销**：URL 200，PDF v1.0 / Date Sep 2023，MXFP8/MXFP4 block 32 + E8M0 逐字核实 | opencompute.org PDF 一手 |
| 1.4/2.3 NVFP4 官方文档 | **销**：per-16 UE4M3 + tensor-wide FP32 经 cuBLAS 官方文档 + TRT-LLM 官方文档一手核实；blog URL 存在（200）但正文反爬不可达——新增 nvidia-nvfp4 条目 | docs.nvidia.com/cuda/cublas + nvidia.github.io/TensorRT-LLM |
| 2.2 CachedAttention / CacheBlend / Preble repo | **销**：CachedAttention 无官方 repo（GitHub search 0 + 论文无代码链接）维持 [B]；CacheBlend YaoJiayi/CacheBlend、Preble WukLab/preble 均 200 | GitHub API + search |
| 2.3 KIVI / KVQuant / H2O / SnapKV repo | **销**：四 repo 均 200（+ StreamingLLM mit-han-lab/streaming-llm 200）；KIVI/H2O/StreamingLLM/CacheGen 摘要自链官方 repo | GitHub API + arXiv 摘要 |
| 2.4 vLLM kv_offload/ + simple_kv_offload/ | **销**：均在 vllm/v1/ 下，模块级路径全列（见 2.4 节） | codeload tarball |
| 2.4 SGLang HiCache / hisparse 特性名与路径 | **销**：HiCache = L1/L2/L3 三层 KV cache；HiSparse = DSA 系 decode 侧 top-k hot buffer + host 全量 KV；路径全列（见 2.4 节）；**HiRadixCache 类已重构**（版本漂移注） | tarball + 官方 docs |
| 2.4 Mooncake transfer engine | **销**：Segment/BatchTransfer 抽象 + 拓扑感知路径选择 + Transport 列表，官方 design doc 一手 | tarball docs/source/design/transfer-engine/ |
| 2.4 3FS USRBIO / FUSE | **销**：USRBIO = User Space Ring Based IO（Iov/Ior），FUSE 瓶颈 ~400K 4KiB reads/s；**EC 10+2 为 DeepInfer 生产披露、开源 repo 只有 CRAQ**（口径分离，见 2.4 节） | tarball README + UsrbIo.md + design_notes.md |
| 2.4 LMCache CacheGen 集成 | **销**：kernels 在 repo；**multiprocess 模式 CacheGen 官方 doc 注明尚未实现**，原始实现 LMCache/LMCache_inprocess；CacheBlend blend 模块 + vllm/sglang connector 路径核实 | tarball（dev 分支）+ docs/source/mp/cachegen.rst |
| W0 终审 5 repo URL 复核 | **全部 200**：ari-holtzman/degen、menhguin/minp_paper、microsoft/unilm/YOCO（contents API 列目录：README.md / yoco/ / scripts/）、punica-ai/punica、S-LoRA/S-LoRA | GitHub API（本轮可达） |

## 1.1 请求生命周期与 SLO 指标

**goodput 定义（DistServe 原文，arXiv:2401.09670v3 §1，逐字）**：
> "maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query."

定义要件 = per-GPU 归一 + SLO attainment 目标，不得写成「达标请求的裸吞吐」（spec 1.1 硬口径）。论文自链官方实现 github.com/LLMServe/DistServe（摘要内）。可用数值实例（论文 §1，13B 模型 A100、90% attainment）：colocate 单卡 goodput ≈1.6 rps；分离后 prefill 实例 5.6 rps、decode 实例 10 rps，2P+1D 配置整体 10 rps = 3.3 rps/GPU，2.1× 于 colocate。来源：distserve。

**TPOT vs TBT 用法对照**：DistServe 用 TPOT（TTFT and TPOT requirements，原文）；DeepInfer doc 用 TBT（"so TTFT and TBT can be controlled and optimized independently"，doc L31）。来源：distserve、deepinfer-doc。

**Orca 机制（OSDI'22 原文 PDF 核实，无 arXiv）**：
- iteration-level scheduling："schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch"（摘要）。
- selective batching："splits the batch and processes each request individually for the Attention operation while applying batching to other operations of the Transformer model"（§1）；"token-wise (instead of request-wise) batching to other operations"（§4，Fig 5：7 token 批 → 非 Attention 算子输入 [7, H]，Attention 前插 Split 逐请求算）。
- 实测口径（4.1 也用）：GPT-3 175B，"36.9× throughput improvement at the same level of latency" vs NVIDIA FasterTransformer（摘要；HTML 渲染为 "36:9×" 系排版伪影，PDF 为准）。
- 作者：Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun（Seoul National University / FriendliAI）。来源：orca。

**Azure LLM trace（归属 Splitwise，双重一手）**：
- 论文原文（2311.18677v2）："We use production traces taken from two Azure LLM inference services on November 11th 2023." + "We have released a subset of our traces at https://github.com/Azure/AzurePublicDataset"；characterization 用 20 分钟子集，字段 = arrival time / input size / output size；两服务 = coding 与 conversation。
- 数据集描述文件 AzureLLMInferenceDataset2023.md（**是 .md 文件不是目录**，repo 默认分支 master）："This dataset is the data described and analyzed in the ISCA 2024 paper 'Splitwise...'"；CSV = data/AzureLLMInferenceTrace_code.csv 与 data/AzureLLMInferenceTrace_conv.csv；schema = TIMESTAMP / ContextTokens / GeneratedTokens；CC-BY；**无 prompt 内容**（GDPR，只给尺寸）；Splitwise 发表于 ISCA 2024（引用时可注明 venue）。来源：splitwise、azure-llm-trace。

**ShareGPT（与 Azure trace 分开归属）**：社区共享 ChatGPT 对话数据集，常用 benchmark 版本 = anon8231489123/ShareGPT_Vicuna_unfiltered 中的 ShareGPT_V3_unfiltered_cleaned_split.json（hf-mirror API 核实存在，326K+ downloads）。vLLM bench 用 `--dataset-name sharegpt --dataset-path <本地路径或 HF ID>`（datasets.py ShareGPTDataset，无硬编码默认 URL）；SGLang bench serving 亦支持 sharegpt 数据集。**非 Splitwise 释放、非 Azure 生产 trace**。来源：sharegpt（新条目）、vllm-repo、sglang-repo。

**benchmark 工具（版本锚定 2026-11-23 main/dev）**：
- vLLM：`vllm bench serve / latency / throughput`——vllm/benchmarks/{serve,latency,throughput}.py + CLI vllm/entrypoints/cli/benchmark/（main.py 注册）。
- SGLang：**已迁移** `python -m sglang.benchmark.serving`（python/sglang/benchmark/serving.py；旧 `python -m sglang.bench_serving` 路径在当前 main 不存在）；官方指南 docs/docs/developer_guide/bench_serving.mdx；指标 = throughput / TTFT / ITL / per-request e2e；多 backend（sglang / vllm / lmdeploy / trt 等 OpenAI-compatible 端点）。
- genai-perf：triton-inference-server/perf_analyzer 下 genai-perf/（contents API 核实）；指标 = output token throughput / TTFT / **time to second token** / ITL / request throughput；负载 = concurrency 或 request rate。**官方 README 声明：「GenAI-Perf is being phased out... please use AIPerf instead」（ai-dynamo/aiperf，repo 200 核实）——写作须注明此状态与核实日期**。
- DeepInfer 在线/离线口径："we reach roughly 80% of a perfectly balanced offline benchmark"（doc L155，官方披露）。来源：vllm-repo、sglang-repo、genai-perf、deepinfer-doc。

## 1.2 Prefill / Decode 的算力与带宽账

**V4.1-Flash config（hf-mirror 2026-11-23 重取，全字段与 spec 附录一致）**：architectures = DeepseekV41ForCausalLM；text_config：40L、hidden 5120、64 Q heads、**num_key_value_heads 1、head_dim 512、qk_rope_head_dim 64**、q_lora_rank 1280、o_lora_rank 1024、o_groups 8、384 routed + 1 shared、top-6（num_experts_per_tok）、sliding_window 128、index_topk 512 / index_n_heads 32 / index_head_dim 128、num_nextn_predict_layers 3、vocab 129280、max_position_embeddings 1048576、compress_ratios 43 项；quantization_config：quant_method fp8、activation_scheme dynamic、**weight_block_size [32,32]、scale_fmt ue8m0、expert_dtype fp4**；多模态 vision_config + image_token_id 129264。κ（MLA 口径）= (512 + 64)·bytes = 576·bytes / token / 层（bf16 = 1152 B）；**勿代入 2·H_kv·d_head 形**（2×1×512 = 1024，较 576 高估 1.78×，spec 已给算例）。indexer K 另计。来源：v41-flash-config [W]。

**DeepSeek-V3**：paper arXiv:2412.19437（arXiv API 核实，DeepSeek-AI，200 作者）；开放权重 deepseek-ai/DeepSeek-V3（hf-mirror API 核实，非 gated，185 文件）——[A] 模型锚点 paper + open weights 齐。V3 MoE 账用论文口径（671B 总 / 37B 激活等数值以 paper 为准，本轮未重取 V3 config——writer 如需 V3 config 字段走 models/training references 已核实条目或论文）。来源：deepseekv3。

**DeepInfer Wide-EP / DP 口径（doc 一手）**：attention #DP = #EP（L33）；prefill EP2–8 / DP2–8 取 HBM 允许的最小（L47；理由 L55-57：prefill 每 expert batch 已足够大，小 DP/EP 组改善负载均衡 + 减 NVLink 流量）；decode EP64–128 / DP64–128 Wide-EP 取 GPU 允许的最大（L49）；"Wide-EP substantially reduces per-step memory traffic in the MoE expert part and, with good EPLB support, effectively raises TPS"（L61）。每 expert batch = Bk/E 与 EP 度无关的前提（B = 全局 batch + 均衡路由）与「扩 e 耦合扩 #DP=#EP → 全局 B 增 → 强度升」的 reconciling 注按 spec 1.2 口径写。来源：deepinfer-doc（官方披露）。

**recap 指引**：models 2.4 式 (9)（KV 容量）与 1.2 §10 / 2.4 §6 式 (10)-(12)（算术强度、decode I≈B、饱和上界）纯文字指引，符号按 spec §4.3 翻译规则。vLLM paper 作 KV 几何背景引用。来源：vllm。

## 1.3 Roofline 与延迟-吞吐权衡

**NVIDIA 官方 datasheet 精确数值（W1 PDF 一手，全部 dense = 官方 sparse 表值之半；脚注原文 "With sparsity" / "Specifications in sparse. Dense is one-half of the sparse spec shown."）**：

| 参数 | H100 SXM | H200 SXM | B200（HGX B200，SM100） | GB200（NVL72/NVL4 每 GPU） |
| --- | --- | --- | --- | --- |
| BF16/FP16 TC dense | 989.5 TFLOPS（表值 1,979*） | 989.5 TFLOPS（表值 1,979*） | 2,250 TFLOPS（表值 4.5 PF*） | 2,500 TFLOPS（表值 5 PF*） |
| FP8 TC dense | 1,979 TFLOPS（表值 3,958*） | 1,979 TFLOPS（表值 3,958*） | 4,500 TFLOPS（表值 9 PF*） | 5,000 TFLOPS（表值 10 PF*） |
| FP4 TC dense | 无（Hopper 不支持 FP4） | 无 | 9,000 TFLOPS（表值 18 PF*） | 10,000 TFLOPS（表值 20 PF*） |
| TF32 TC dense | 494.5 TFLOPS（表值 989*） | 494.5 TFLOPS | 1,100 TFLOPS（表值 2.2 PF*） | 1,250 TFLOPS（表值 2.5 PF*） |
| FP32 | 67 TFLOPS | 67 TFLOPS | 75 TFLOPS | 80 TFLOPS |
| FP64 / FP64 TC | 34 / 67 TFLOPS | 34 / 67 TFLOPS | 37 / 37 TFLOPS | 40 / 40 TFLOPS |
| 显存 | 80 GB HBM3 | 141 GB HBM3e | **180 GB HBM3E** | **186 GB HBM3E** |
| 显存带宽 | 3.35 TB/s | 4.8 TB/s | **7.7 TB/s** | **8 TB/s** |
| NVLink | 900 GB/s（NVLink4） | 900 GB/s | 1.8 TB/s（NVLink5） | 1.8 TB/s（NVLink5） |
| PCIe | Gen5 128 GB/s | Gen5 128 GB/s | Gen5 128 GB/s | Gen5 128 GB/s |
| TDP | ≤700 W | ≤700 W（SXM） | ≤1000 W 可配 | ≤1200 W 可配 |

- 出处：H100/H200 = resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23 落地页内嵌官方 PDF（H200 datasheet Nov24 doc 3512650；H100 datasheet © 2024）；B200/GB200 = NVIDIA Blackwell architecture datasheet PDF（dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf，Technical Specifications 表 + Individual Blackwell GPU Specifications 表）。交叉核对：HGX B200 PCF Summary（images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf）"eight NVIDIA Blackwell B200 GPUs, each with 180 GB of HBM3E"、FP4 144 PFLOPS*（8 卡）一致；PCF 的 "FP32 600 PFLOPS" 系单位笔误（应为 600 TFLOPS = 8×75，datasheet 为准）。
- **SKU / 代际纪律（spec ⟨verify⟩ 注的兑现）**：B200（HGX，180 GB / 7.7 TB/s）≠ GB200（Grace 配对，186 GB / 8 TB/s，频率更高）≠ B300（Blackwell Ultra，288 GB）≠ Rubin（SM107，产品页现主推）——1.3 参数表只用 B200 列，GB200 作对照注。H200 NVL 变体（1,671/3,341* TFLOPS、141 GB、4.8 TB/s、TDP up to 600 W configurable）勿与 SXM 混。（W1 fix round 纠正：原误植「350-400 W」系 H100 NVL 口径；H200 官方 datasheet PDF（dam-cdn b2kt2580e5b6424if22lg65223m6ghjb.pdf）Technical Specifications 表 H200 NVL 列「Max Thermal Design Power (TDP): Up to 600W (configurable)」，evaluator P1 F1 一手核实。）
- ridge point 算例素材（research 侧由上表直接算，writer 可复核）：H100 BF16 dense ridge ≈ 989.5/3.35 ≈ 295 FLOP/byte、FP8 ≈ 591；H200 BF16 ≈ 206、FP8 ≈ 412；B200 BF16 ≈ 292、FP8 ≈ 584、FP4 ≈ 1169。来源：nvidia-datasheet [C] 官方规格披露。

**Ascend 950（Huawei 官方一手，部分规格）**：
- keynote（huawei.com/en/news/2025/9/hc-xu-keynote-speech，Xu Zhijun / Eric Xu，Huawei Connect 2025-09）逐字要点：950PR 与 950DT 同 Ascend 950 Die；"deliver 1 PFLOPS in FP8, MXFP8, and HiF8, and 2 PFLOPS in MXFP4"；"2 TB/s interconnect bandwidth, which is 2.5 times higher than the Ascend 910C"；SIMD+SIMT 混合；访存粒度 512B→128B；HiF8 = 华为自有格式（精度近 FP16、效率比肩 FP8）；950PR = prefill + 推荐，HiBL 1.0（自有低成本 HBM，比 HBM3E/HBM4E 更具成本效益），2026Q1，卡 + SuperPoD 两形态；950DT = decode + 训练，HiZQ 2.0 **"delivering 144 GB of memory and 4 TB/s memory access bandwidth"**，2026Q4；Atlas 950 SuperPoD = 8,192 × 950DT、8 EFLOPS FP8 / 16 EFLOPS FP4、16 PB/s 互联、2026Q4。**keynote 未注明 1 PF/2 PF 的 dense/sparse 口径——写作只引 keynote 口径不推定**。
- hiascend 官方产品页（hiascend.com/hardware/processor，Ascend 950PR tab）："最大配备128GB、1.6TB/s高速片上内存"、"支持多种精度格式，最大支持1784TFLOPS"（**精度格式未注明，只可逐字引用不可解释**）、"支持灵衢2.0互联协议，最大支持2TB/s双向互联带宽"。950PR 的 128 GB / 1.6 TB/s 由此补齐（keynote 正文无此二数）。
- **Atlas 350 区分**：Atlas 350 是基于 950PR 的加速卡（2026-03 上市，媒体广泛报道卡级 1.56 PF FP4 / 112 GB / 1.4 TB/s / 600 W）——卡级数与芯片级数不同，且本轮未取得华为官方卡级 datasheet，**媒体数一律不作事实来源**（spec §2 闭源与生产披露处理条款）；1.3 只用芯片级官方数 + 注明卡级存在差异。来源：ascend-950-keynote [C] 官方披露。

**scale-up domain 素材**：NVLink 域——H100/H200 NVLink4 900 GB/s、B200 NVLink5 1.8 TB/s、GB200 NVL72 = 72 GPU 单 NVLink 域（datasheet：Total NVLink Bandwidth 130 TB/s）；超节点——Atlas 950 SuperPoD 8,192 NPU（UnifiedBus / 灵衢）。排队论（M/M/1 W = 1/(μ−λ)、W_q = ρ/(μ−λ)、Kingman VUT 近似 W_q）为标准数学，按 spec 1.3 口径写，λ 章首声明「到达率，非 GAE 参数」。来源：nvidia-datasheet、ascend-950-keynote、distserve。

## 1.4 采样与解码策略

- **top-p**：arXiv:1904.09751（Holtzman et al.，arXiv API 核实）+ 官方 repo ari-holtzman/degen（200，W0 终审 + W1 双核实）。
- **min-p**：arXiv:2407.01082（Nguyen et al.，v8）+ 第一作者官方代码 menhguin/minp_paper（200）。
- **XGrammar**：arXiv:2411.15100（Dong et al.）+ mlc-ai/xgrammar（200）；**Outlines**：arXiv:2307.09702（Willard & Louf，2 作者）+ dottxt-ai/outlines（200，原 outlines-dev 301 更名）；**guidance**：guidance-ai/guidance（200，无独立论文，教学参照）。三者对照按 spec 1.4 口径（XGrammar 自适应 mask 扩展 vs Outlines FSM vs guidance），引擎实现归 5.4 / 6.1。
- **batch-invariance 博客（[C] 开放技术博客、非闭源）**：canonical URL thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/（200 + 全文核实）；Horace He in collaboration with others at Thinking Machines，**2025-09-10**。核心论断（全文核实）：浮点非结合是 "the original sin"；"concurrency + floating point" 假说（atomicAdd 累加顺序随并发完成顺序不定）只是部分解释——**batch 组成改变 reduction 分区/顺序同样改变结果**；解法 = batch-invariant kernels（RMSNorm / matmul / attention 三类，固定 reduction order）。引擎模式对照路径（tarball 核实）：vLLM `vllm/model_executor/determinism/batch_invariant.py` + `batch_invariant_configs.py`（**无 batch_invariant_ops 目录**，spec 3.4 口径）；SGLang `srt/batch_invariant_ops/batch_invariant_ops.py`；DeepInfer 环境变量 `DEEPINFER_BATCH_INVARIANT`（deepinfer/model_executor/models/v4_modules/attn.py 消费）。kernel 层细节归 3.4。来源：batch-invariance、vllm-repo、sglang-repo、deepinfer-repo（路径经 deepinfer-doc 同 tarball 核实）。
- **sampling 实现路径**：vLLM `vllm/v1/sample/`（+ ops/）；SGLang `srt/layers/sampler.py`；DeepInfer sampling 走读归 7.x。
- **流式输出工程**：deepseek-ai/deepseek-recipe repo（200 核实；Rust 多协议 + special-token 注入防护 + 分段并行 BPE + 流式状态机 + 增量 KMP——spec 附录口径，详细走读归 7.4）；DeepInfer doc "DeepSeek Recipe handles this encoding and decoding efficiently and accurately; by segmenting on special tokens and encoding the segments in parallel, it prevents special-token injection while reducing tokenization latency for long inputs"（L35；W1 fix round 纠正：原省略号后误作 preventing，doc 逐字为 it prevents，evaluator P1 F6 一手核实）。来源：deepseek-recipe、deepinfer-doc。
- Gumbel-max 等价、penalty 公式、top-p 截断再归一化为标准数学推导，按 spec 关键推导清单写，无外部数值依赖。

## 2.1 PagedAttention 与块管理

**vLLM 论文口径（arXiv:2309.06180v1 全文核实）**：
- 碎片上界原文："vLLM limits all the memory wastes for a request within one block"（§4.1）——每请求浪费 ≤1 block（相对浪费 < b/T）。**「<4%」是 Fig.2 典型长度下经验均值，非论文 stated bound 亦非普适上界**（spec A-S3-5 裁定，本轮全文复核维持：论文文字只说 "near-zero waste"，Fig.2 为实验条形图）。
- 既有系统浪费实测（Fig.2 / §2）："only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems"——可作对照算例。
- block table："Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions"（§4.1）。
- COW 原文："The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block... This involves only copying one block of data."（§4.4，beam / parallel sampling 块共享）。
- block size 权衡（§7.2）："If the block size is too small, vLLM may not fully utilize the GPU's parallelism... If the block size is too large, internal fragmentation increases and the probability of sharing decreases."；**论文默认 16**："vLLM sets its default block size as 16"（§7.2）。
- **当前引擎默认 64（版本锚定 2026-11-23 main）**：`CacheConfig.DEFAULT_BLOCK_SIZE: ClassVar[int] = 64`（vllm/config/cache.py **L185**，与 spec 行号一致）；backend 偏好可调——**SWA 256**：v1/attention/backends/mla/sparse_swa.py L166-167 `get_preferred_block_size → return 256`；**CPU_MLA 16**：v1/attention/backends/mla/cpu_mla.py（"the platform layer forces block_size=16 for MLA models"，get_supported_kernel_block_sizes → [16]）；偏好协商机制 platforms/interface.py `_preferred_block_size_for_backends`。**论文 16 ≠ 当前默认 64，写作两个口径都要给**（spec A-S2-2）。
- 机制例证四文件（tarball 核实存在）：`vllm/v1/core/block_pool.py`、`kv_cache_manager.py`、`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`（hybrid KV 协调）；block hash 基础在 `v1/core/kv_cache_utils.py`（→2.2）。模块全量走读归 5.2（两层纪律）。来源：vllm、vllm-repo。

## 2.2 Prefix Caching 与 RadixAttention

**RadixAttention（arXiv:2312.07104v2 原文核实）**：
- "our system maintains an LRU cache of the KV cache for all requests within a radix tree... uses a radix tree for efficient matching, insertion, and eviction"（摘要）。
- "We implement an LRU eviction policy and a cache-aware scheduling policy to enhance the cache hit rate. RadixAttention is compatible with techniques like continuous batching, paged attention, and tensor parallelism."（§3）。
- "evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves"（§3.1）——LRU 叶优先驱逐语义。
- 论文 KV 布局："stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token"（§3.1，page = 1 token；与 vLLM block 16/64 对照素材）。
- 当前实现版本注：RadixCache（mem_cache/radix_cache.py L282）仍在；**HiRadixCache 类在当前 main 已不存在**，重构为 UnifiedRadixCache（unified_radix_cache.py L224）+ hybrid_cache/（HiCache 用，见 2.4）——2.2 机制级引论文 + radix_cache.py 即可，实现走读归 6.1。来源：sglang、sglang-repo。

**vLLM automatic prefix caching**：block hash 链机制，代码基础 v1/core/kv_cache_utils.py（存在核实）；hash 算法默认 sha256（config/cache.py L269 `prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"`，可选 sha256_cbor / xxhash / xxhash_cbor）。来源：vllm、vllm-repo。

**CachedAttention（arXiv:2403.19708v3，[B] 无官方 repo——W1 复核维持）**：多轮对话 session KV 保留；hierarchical KV caching（cost-effective memory/storage 介质）；**layer-wise pre-loading**（逐层 H2D 与计算重叠）+ **asynchronous saving**（保存与计算重叠）；scheduler-aware fetching / eviction（按调度器 hint 放置层级）；位置编码解耦 + KV 截断防 context window overflow 失效。GitHub search "CachedAttention" total_count = 0，论文全文无代码链接。来源：cachedattention。

**CacheBlend / Preble**：CacheBlend（2405.16444，Yao et al.）官方 repo YaoJiayi/CacheBlend 200——跨文档 KV 融合、选择性重算 cross-attention token（spec 口径）；Preble（2407.00023，Srivatsa et al.）官方 repo WukLab/preble 200——分布式 prompt 调度、prefix affinity。来源：cacheblend、preble。

**DeepInfer TTL / 命中率 / prefix affinity（doc 一手，官方披露）**：
- TTL 生产口径（L211）："a TTL of 72 hours for agent traffic and 7 days for chatbot traffic captures almost all reusable cache entries. The 72-hour TTL for agent traffic also spans a weekend"——近乎无损。
- **反事实口径（L187，勿混）**："a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate"——0.5-1% 损失是 **chatbot 流量缩至 3d TTL 的反事实**，非 agent 72h 生产损失（spec B-4 / C-S2-4 / A-S3-6 限定，本轮逐字复核）。chatbot 长尾命中重（少量 KV 被读极多次）。
- prefix affinity 解法（L139 / L175）："Since any engine can hit any prefix, there is no prefix-affinity routing"；专用 3FS 存储集群让任何 prefill 机命中任何前缀，避免 prefix affinity 与负载均衡路由冲突（→2.4 / 8.2 伏笔）。
- 匹配机制（L191）：Entrypoint CHECK 角色对 local kvcache-store 做 **hash-chain prefix match over metadata only**；hit length 须在路由前算出、不可隐藏。来源：deepinfer-doc。

## 2.3 KV 量化、驱逐与压缩

**KV 量化（摘要 / 原文核实）**：
- **KIVI**（2402.02750，Liu et al.）：key per-channel / value per-token 非对称 2bit、tuning-free；**residual 全精度窗口**原文（§3）："the residual key cache and value cache are kept in full precision"（不足一组的尾部 token）；2.6× 峰值内存降低（含权重）、4× batch、2.35–3.47× 吞吐（摘要）。repo jy-yuan/KIVI 200 + 摘要自链。
- **KVQuant**（2401.18079，Hooper et al.）：四技术 = Per-Channel Key Quantization / **Pre-RoPE Key Quantization** / Non-Uniform（per-layer sensitivity-weighted NUQ）/ Per-Vector Dense-and-Sparse（outlier 逐向量隔离）；3-bit <0.1 perplexity 退化（Wikitext-2 / C4）；LLaMA-7B 1M ctx 单 A100-80GB、10M 8-GPU；定制 CUDA kernel ~1.7× 加速（摘要）。repo SqueezeAILab/KVQuant 200。
- 来源：kivi、kvquant。

**驱逐（摘要核实）**：
- **H2O**（2306.14048，Zhang et al.）：heavy hitter（小部分 token 贡献大部分 attention value）+ recent 动态平衡；eviction 形式化为 dynamic submodular 问题并有理论保证；20% heavy hitters 下吞吐 vs DeepSpeed Zero-Inference / HF Accelerate / FlexGen 至多 29× / 29× / 3×（OPT-6.7B / 30B），同 batch 延迟降至 1.9×。repo FMInference/H2O 200 + 摘要自链。
- **StreamingLLM**（2309.17453，Xiao et al.）：attention sink 现象（初始 token 得强 attention score 即使语义不重要）；保留初始 token KV + recent window → 有限窗口模型泛化到无限长（4M+ tokens，Llama-2 / MPT / Falcon / Pythia）；预训练加专用 sink token 进一步改善；vs sliding window recomputation 至多 22.2× 加速。repo mit-han-lab/streaming-llm 200 + 摘要自链。
- **SnapKV**（2404.14469，Li et al.）：每 head 在生成中稳定关注特定 prompt 特征，可由 **prompt 末端 observation window** 观察得到；自动选 per-head clustered 重要 KV 位置压缩；16K 输入下 3.6× 生成速度 / 8.2× 内存效率；单 A100-80GB 380K ctx（HF 实现小改），NIAH 精度损失可忽略。repo FasterDecoding/SnapKV 200。
- 质量风险 recap models 4.2 §8（召回是稀疏软肋）+ 指引；三路线对照表按 spec。来源：h2o、streamingllm、snapkv。

**低精度格式（官方规范一手）**：
- **OCP MX v1.0**（Date Sep 2023，PDF 16 页，URL 200）：MX-compliant 格式 = 块大小 k + 元素类型 + scale 类型；**MXFP8 / MXFP6 / MXFP4 / MXINT8 一律 scaling block size 32 + scale E8M0（8 bit）**；MXFP4 元素 = FP4 E2M1；E8M0 = "an unsigned representation of a conventional biased Float32 exponent"（§5.4.1，无符号无尾数）。MXFP8 元素可 E4M3 或 E5M2（不同 encoding 算不同格式）。
- **NVFP4（NVIDIA 官方文档核实）**：E2M1 元素 + **每 16 元素一个 UE4M3（unsigned E4M3）block scale** + **tensor-wide FP32 global scale**，Blackwell（compute capability 10.0+）。cuBLAS 原文："a dedicated scaling factor stored as an 8-bit CUDA_R_8F_UE4M3 value for each 16-element block in the innermost dimension"（CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3）；全局 scale = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F（"tensor-wide scaling factor... available only when scaling factors use the CUDA_R_8F_UE4M3 data type"）。TRT-LLM 官方文档：KV cache `dtype='nvfp4'`（需 ModelOpt 离线量化）。**blog 正文本机反爬不可达（URL 200），格式事实以上述两份官方文档为准**。对照：MXFP8 = per-32 UE8M0（cuBLAS VEC32_UE8M0）。
- **DeepInfer 低精度 KV（doc L271-275 逐字，官方披露）**："Main KV: per-16, e4m3 — like NVFP4 but without the global scale." / "Indexer KV: per-32, ue8m0, MXFP4." / "MoE routed-expert weights: per-32, ue8m0, MXFP4."——主 KV「类 NVFP4 无 global scale」与 NVFP4 的差别正是 global scale 一项。
- 格式账正式交付在 3.2（本节前瞻引用 + 回链）；**勿混 TRT-LLM 文档中的 NVFP4_PLUS（Rubin SM107：per-32 UE5M3、无 FP32 global scale）——那是另一格式另一代际**。来源：ocp-mx、nvidia-nvfp4（新条目）、deepinfer-doc。

## 2.4 KV Offload 与分层存储

**三层存储账素材（官方规格 + 官方披露）**：HBM 层用 1.3 datasheet 数（TB/s、百 GB）；DRAM / SSD 层延迟列按 spec A-S3-7 口径 = **GPU 侧访问口径（含 PCIe / 网络栈）**，介质本征延迟（DRAM ~50-100 ns / NVMe ~10-100 μs）另注不混用。DeepInfer 存储集群官方披露（doc L101 / L105）：每存储服务器 20 × 30.73 TB SSD、≥40 GB/s 读 / 5 GB/s 写、保留 24h 数据；逻辑容量 ~14.4 TB/GPU（物理更大）；3FS ~100 PiB SSD（L187）。PCIe Gen5 128 GB/s（datasheet）作 D2H/H2D 上限素材。

**vLLM offload 模块（tarball 路径级核实，均在 vllm/v1/ 下）**：
- `kv_offload/`：通用 offload 框架——base.py（"Core abstractions for KV cache offloading in vLLM v1"）、config.py、factory.py、file_mapper.py、pointer_utils.py；`cpu/`（manager.py、gpu_worker.py、swap_blocks_triton.py、host_register.py、shared_offload_region.py、spec.py、policies/{lru,arc,base,factory}.py）；`tiering/`（manager.py、async_lookup.py、backpressure.py、metrics.py、spec.py、factory.py + fs/ obj/ p2p/ kvcr/ example/ 子后端，p2p 含 nixl 数据面 + zmq 控制面 + session）；`fs/worker.py`。
- `simple_kv_offload/`：**SimpleCPUOffloadConnector**（manager.py docstring "Scheduler-side manager for SimpleCPUOffloadConnector"）——worker.py、host_buffer.py、copy_backend.py、disk_backend.py、cuda_mem_ops.py、sizing.py、metadata.py、metrics.py。
- 两模块并存：kv_offload = 分层 / 多后端框架，simple_kv_offload = 简化 CPU offload connector；走读时先讲清分工。来源：vllm-repo。

**SGLang HiCache / HiSparse（tarball + 官方 docs 核实）**：
- **HiCache = Hierarchical KV Caching**：L1 GPU HBM / L2 host DRAM / L3 distributed storage；**L1/L2 每实例私有，L3 是否跨实例共享取决于 backend 配置**（hicache_design.mdx  tier sharing 表）；L3 backends：file / mooncake / hf3fs（3FS）/ nixl / aibrix / lmcache / eic / flexkv / seaweedfs 等（mem_cache/storage/ 目录核实）；HiRadixTree 元数据 + prefetch 策略 + write-back 策略 + zero-copy 传输。路径：mem_cache/hicache_storage.py（HiCacheStorage ABC / HiCacheFile 等）、hybrid_cache/hybrid_cache_controller.py、memory_pool_host.py、hicache_auto_size.py、hicache_key_scheme.py；docs/docs/advanced_features/{hicache,hicache_design,hicache_best_practices,hicache_storage_runtime_attach_detach}.mdx；官方博客 lmsys.org/blog/2025-09-10-sglang-hicache/（doc 自链，未取全文，只作指针）。
- **HiSparse = Hierarchical Sparse Attention**（hisparse_guide.mdx 逐字要点）：decode 侧 GPU 只留小 hot buffer（如 4K token），全量 KV 在 CPU pinned memory；CUDA kernel 按 attention score **on-demand top-k swap-in**（短序列 fast path / 长序列 hit detection → LRU reordering → miss handling）；eager backup 异步回写上一 token KV；**前提：DSA 系模型（DeepSeek-V3.2、GLM-5.1、Kimi-K3.1 DSA 层、DeepSeek V4、MiniMax M3）+ PD 分离模式 + 仅 decode 实例**；PD 集成 direct-to-host：prefill GPU 经 RDMA 直写 decode host pool（绕过 decode GPU，消除 KV 传输瞬时显存尖峰）；DeepSeek V4 direct-to-host 只写 C4 KV，c4_indexer 与 C128 KV 仍 D2D。路径：mem_cache/{base_hisparse_memory_pool,hisparse_memory_pool}.py、pool_host/hisparse.py、allocator/hisparse.py、managers/hisparse_coordinator.py、kernels/ops/kvcache/hisparse.py。
- **DeepInfer 对照口径（doc L183）**：decode 不开 KV offload、也不用 Hi-Sparse（"Decode is not bound by cache capacity"），但 decode 产生的 KV 仍写盘提命中率——与 SGLang HiSparse 路线形成对照。来源：sglang-repo、deepinfer-doc。

**Mooncake（paper 摘要 + repo design doc 一手）**：
- paper（2407.00079v5）：Kimi 生产 serving 平台；KVCache-centric 分离架构（prefill / decode 集群分离）；利用 GPU 集群闲置 CPU / DRAM / SSD 做 disaggregated KVCache cache；KVCache-centric scheduler 平衡有效吞吐与 SLO；过载场景 prediction-based early rejection；模拟场景至多 525% 吞吐提升、真实 workload 多服务 75% 请求。
- **Transfer Engine**（docs/source/design/transfer-engine/index.md，英文当前版）：零拷贝传输库，两核心抽象——**Segment**（RAM Segment = DRAM/VRAM 可远程读写地址空间，每进程一个，内部划 Buffer 注册 (GPUDirect) RDMA；NVMeof Segment = NVMe 文件偏移直传 DRAM/VRAM 零拷贝）+ **BatchTransfer**（异步 Read/Write，非连续空间同步，"类似异步且更灵活的 AllScatter/AllGather"）；接口 TransferEngine 类（mooncake-transfer-engine/include/transfer_engine.h）；Transport 当前列表：Tcp / Rdma / Efa / NVMeoF / Nvlink / IntraNodeNvlink / Hip / Shm；**拓扑感知路径选择**：每服务器生成 topology matrix 广播集群，NIC 按内存类型分 preferred / secondary 列表，正常走 preferred（本地 NUMA / 本地 PCIe switch GDR），故障时两列表皆可用；多网卡池化 + 重试；本地目标走 memcpy / cudaMemcpy。来源：mooncake。

**LMCache（dev 分支 tarball 核实）**：CacheGen 压缩 kernels 在 repo（csrc/cuda/cachegen_kernels.cuh + lmcache/v1/platform/torch_ops/cachegen_kernels.py，另有 musa / sycl 变体）；**官方 docs/source/mp/cachegen.rst 注明：multiprocess 模式 CacheGen 尚未实现（coming soon），原始实现维护于 LMCache/LMCache_inprocess**——写作必须带此限定，勿写「LMCache 已全模式集成 CacheGen」；CacheBlend 集成 = lmcache/v1/multiprocess/modules/blend/{module,reorder}.py；vLLM / SGLang connector = lmcache/integration/{vllm,sglang}/。CacheGen paper（2310.07240v6 原文）：量化 + **逐层逐通道算术编码**（lossless arithmetic coder 编码量化 delta tensor 成 bitstream，利用 KV 分布局部性与层敏感度差异）；带宽自适应压缩级；3.5-4.3× KV 尺寸压缩、3.2-3.7× 取回 + 处理总延迟降低、质量影响可忽略；repo UChi-JCL/CacheGen 200 + 摘要自链。来源：lmcache、cachegen、cacheblend。

**3FS（tarball 一手，USRBIO / FUSE ⟨verify⟩ 销项）**：
- **USRBIO = User Space Ring Based IO**（src/lib/api/UsrbIo.md）：应用直接向 FUSE 进程内的 3FS I/O 队列提交请求，绕过 FUSE 限制（如最大单 I/O 尺寸）；**Iov** = 用户/FUSE 进程间大段共享内存零拷贝区（IB 内存注册由 FUSE 进程管理，读入 Iov / 写先入 Iov）；**Ior** = io_uring 式小共享环（io_depth 控制批处理，多环推荐多线程）；fd 须注册后方可用 USRBIO；fio engine 在 benchmarks/fio_usrbio/。
- **FUSE 限制**（docs/design_notes.md）：kernel-user 数据拷贝耗内存带宽增延迟；共享队列自旋锁 → 并发不扩展，**实测 ~400K 4KiB reads/s 上限**；Linux 5.x FUSE 不支持同文件并发写。
- 架构：cluster manager / metadata service（无状态，FoundationDB）/ storage service / client（FUSE + native）四组件，RDMA 网络（IB / RoCE）；存储侧 **CRAQ 链式复制（write-all-read-any）**强一致；文件切 chunk 跨 replication chain 条带化。
- **EC 口径分离（spec 内部张力项的裁定素材）**：开源 3FS repo **无 erasure coding 实现 / 文档**（grep 全 repo 无 "erasure"）；DeepInfer doc 的 "10+2 erasure coding"、"each EC group tolerates two simultaneous node failures"、"A read issues requests to all 11 storage nodes in the group simultaneously; any 10 responses are enough to decode and return"（L111-113）是 **DeepInfer 生产 kvcache-store 的官方披露**。10+2 = 12 节点组 vs 「all 11 storage nodes in the group」措辞存在内部张力（12 节点组读 11 取 10 是可成立的 quorum 读策略，但 doc 字面说组内 11 节点）——**章节引用时逐字引 doc + 标官方披露 + 注明张力，不得把 EC 归给开源 3FS repo，不得自行「修复」doc 数字**。GC Worker LRU 回收（每存储节点一实例，水位以下，doc L115）同属官方披露。来源：3fs、deepinfer-doc。

**DeepInfer KV 存储 / 传输口径（doc 一手，官方披露，2.4 核心素材）**：
- 三层表（L165-169）：in-flight block KV + state KV → 推理 GPU HBM；SWA KV → 本地 host memory（单副本、不上 3FS，跨机命中走 400G VLAN 拉取）；paged block KV → 专用 3FS 集群（无应用层副本，靠 3FS 自身冗余，kvcache-store 管理）。
- 专用集群 vs 分层（L175）：任何 prefill 机命中任何前缀（避免 prefix affinity 与 LB 冲突）；存储与计算 autoscaling 解耦；计算节点故障率比存储节点高一个量级**以上**（"more than an order of magnitude higher"）。
- 重叠机制（L191）：匹配 = Entrypoint CHECK hash-chain（metadata only，hit length 路由前算、不可隐藏）；取 KV = 独立进程异步 3FS → host shared memory；swap-in = 引擎进程内线程逐层 H2D 与 forward 重叠；**"Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs."**
- write-back + 读校验（L195）：写失败不上报引擎，读时严格校验保正确。版本元组 (deployment configuration, model weights, user group)（L199）；权重 / block 字节布局 / tokenizer / 改 KV 值的框架变更均使旧 cache 失效；生产未做过 cache 格式迁移。
- CXL L2 展望（L203-207）：V4.1 KV 小、读已与排队/计算重叠 → TTFT 收益有限；SSD 远比 DRAM 便宜；CXL 减 SSD 写（延寿）是开放问题，当前不用。
- StateBlock（SWA state cache，L217-223）：每序列一固定 slot，[stored region: SWA][dropped region: YOCO consumer SWA + ca_state/ica_state]；stored = 前 20 层；**state_size_to_disk = 20 层 × 134 token × 528 B = 1,415,040 B/seq（MXFP8 SWA KV 格式，仅 stored region 落盘口径）**；128 token 粒度存储、ring buffer 窗口外 +6 位置（copy 位置至多越界 5 token）；SWA KV 不进持久 cache、留生产机内存，Entrypoint 内存目录（逻辑统一、物理按 key 分区）记录持有机。research 侧算术观察（非披露）：528 = 512 + 16，与 512 维 latent 的 MXFP8（512 数据字节 + 512/32 = 16 个 E8M0 scale 字节）一致——writer 如需解释 528 B 构成可用此观察并标注为推导。
- 跨 DC（L97）：压缩 KV 使跨 DC P/D 可行，~400 Gbps / 8 GPU（8.3 详展，2.4 只一句伏笔）。来源：deepinfer-doc。

## 残留延迟项（writer / evaluator 须知）

1. **Ascend 950 dense/sparse 口径**：keynote 的 1 PF FP8 / 2 PF FP4 未注明 dense/sparse；hiascend 页 "最大支持1784TFLOPS" 未注明精度格式——只可逐字引用 + 标官方披露，不得推定口径、不得与 NVIDIA dense 数直接同表对比（如需对比须注明口径未知）。
2. **Atlas 350 卡级官方 datasheet 未获得**：卡级数（1.56 PF FP4 / 112 GB / 1.4 TB/s / 600 W）仅媒体报道——不作事实来源；1.3 只用芯片级官方数。
3. **3FS EC 读路径内部张力**：doc「11 节点 any 10」vs 10+2（12 节点）——逐字引 + 官方披露标注 + 张力注；开源 repo 无 EC 可对照（CRAQ 是另一机制）。evaluator 复核时以 doc 原文为准。
4. **huggingface.co 官方渠道本机仍不可达**：V4.1-Flash config / DeepSeek-V3 weights / ShareGPT 均 hf-mirror 传输核实（W0 + W1 两轮一致）；引用注明传输口径。
5. **NVFP4 blog 正文反爬不可达**：URL 存在性 200 核实，格式事实以 cuBLAS / TRT-LLM 官方文档为据；evaluator 若需 blog 原文走浏览器或存档。
6. **genai-perf 已停止新功能开发**（官方 README）：1.1 介绍时注明状态 + 后继 ai-dynamo/aiperf + 核实日期。
7. **SGLang 版本漂移**（2026-11-23 main）：bench_serving → `python -m sglang.benchmark.serving`；HiRadixCache 类 → UnifiedRadixCache + hybrid_cache/；写作按新路径锚定 + 漂移注（spec 引擎版本锚定条款）。
8. **LMCache CacheGen multiprocess 模式未实现**（官方 doc）：2.4 写 CacheGen 集成必须带此限定；原始实现在 LMCache/LMCache_inprocess（repo 存在性未单独核实，doc 自链——引用时注明出处为 LMCache 官方 doc）。
9. **Orca 36.9× 已提前取得**（4.1 ⟨verify⟩ W2 项）：摘要口径「同延迟水平 36.9× 吞吐 vs FasterTransformer，GPT-3 175B」——W2 writer 直接用，勿再标 ⟨verify⟩。
10. **vLLM hisparse/ kv_hints/ 语义**（5.3 ⟨verify⟩）仍留 W3——本轮只确认目录存在（spec 附录），未做语义核实，不属 W1。

## references.md 本轮变更清单

- P1（10→11 条）：vllm / distserve / orca / splitwise / deepseekv3（补开放权重 URL，[A] 模型锚点补齐）/ v41-flash-config（W1 重取注）/ nvidia-datasheet（B200/GB200 精确值入条，⟨verify⟩ 销）/ ascend-950-keynote（keynote URL + hiascend 950PR 数补齐）/ azure-llm-trace（.md 文件路径纠正 + ISCA 2024）/ genai-perf（phased out + aiperf 注）更新；**新增 sharegpt**。
- P2（13→14 条）：sglang / cachedattention（[B] 复核注）/ cacheblend / kivi / kvquant / h2o / streamingllm / snapkv / mooncake（TE doc 路径）/ cachegen（算术编码核实注）/ preble / lmcache（dev 分支路径 + CacheGen MP 限定）/ ocp-mx（PDF 内容核实注）更新；**新增 nvidia-nvfp4**（被引 2.3, 3.2）。
- 跨波共享锚点（P1/P2 被引）：top-p / min-p（W1 复核 200 注）/ xgrammar / outlines / guidance / batch-invariance（作者 Horace He + 日期 + 全文核实注）/ vllm-repo（+simple_kv_offload、2.1 四文件、L185、bench 入口、determinism 路径；被引 +2.1, 2.2）/ sglang-repo（HiCache/HiSparse 路径 + 版本漂移注；被引 +1.1, 2.4）/ deepinfer-doc（W1 重读注）/ 3fs（USRBIO/FUSE/CRAQ-EC 口径分离注）/ deepseek-recipe（复核日期）更新。
- P3-P9 其余条目未动（yoco / punica / s-lora 的 W0 终审 URL 本轮 API 复核 200，证据在本文件，条目本身未动）。
- 全部更新条目核实日期 = 2026-11-23；新增 W1 精炼说明段。parse-check：9 groups / 79 entries（P1=11, P2=14）。
