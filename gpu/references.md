# references.md — gpu 维度文献索引与术语表

本文件是 gpu 维度全部章节的引用唯一事实源。核心规则：**先收录后引用**——条目由各波 research pass 核实后写入本文件，章节正文才能引用；`scripts/lint-chapter.sh` 检查项 4 强制执行：正文出现的 arXiv ID 在本文件 grep 不到即 lint FAIL。写作 agent 不新增条目，只引用已有条目。站点 build-content 对 gpu 设 refsMin:50 地板（W0 须种入 ≥50 条，空骨架不可过）。

## 写入规则

- 条目只由 research pass 写入；「核实日期」= 当次核实 ID / URL 的日期（YYYY-MM-DD）。spec 中 `⟨verify⟩` 标记的条目 W0 能核则核（arXiv abs / export API / GitHub API / HTTP curl）；不能核者在条目内标 `⟨verify⟩` + 留对应波 research pass，但**来源真实**（repo API 200 / arXiv ID↔标题确认 / Crossref 标题 venue 确认 / 官方 landing 200）仍收录计数。
- 非 arXiv 来源（官方文档、repo、无 arXiv 的经典文献）在「arXiv ID / URL」字段写完整 URL 或标准出处（如「CMU-CS-90-190」「Springer LNCS 3036」「JPDC 69(2)」「SIGCOMM 2015」）。lint 只机械核对 arXiv ID，URL 条目的真实性由 research pass 与 evaluator 把关。
- 「证据等级」五档，gpu 维度口径细化见 gpu spec §4.1（字母语义不变，沿 inference 先例）：
  - **[A]** = 正式 paper + official code（锚点是模型则另需 open weights）。例：FlashAttention / FA2 / FA3 / Marlin / NanoFlow / TileLang / FlashInfer / vLLM / Ring Attention。
  - **[B]** = 有 paper、无官方代码或代码不全；**含老经典无 arXiv 子类**（Blelloch 1990 / Rabenseifner 2004 / Patarasuk-Yuan 2009 / DCQCN Zhu 2015 / Al-Fares 2008）——标原始出处 + 注明「无 arXiv」，按论文拆公式，社区实现辅助并注明。
  - **[C]** = 闭源披露；**含「官方规格披露·一手权威」子类**（NVIDIA whitepaper / datasheet / CUDA Programming Guide / PTX ISA / Best Practices / NCCL docs / GPUDirect docs / IBTA 规范 / OCP MX / MIG / Nsight / XID / SHARP / JEDEC / PCI-SIG）——本维度硬件事实以官方规格为最高证据档，[C] 官方规格 ≠ 弱证据，显式注「官方规格披露·一手权威」+ 核实日期；另含「开放技术博客、非闭源」子类（NVIDIA developer blog，沿 inference 口径）。
  - **[W]** = 开放权重 + config 无正式 paper。**硬件不适用**；仅引模型 config 做 kernel 几何例（如 V4.1 MLA 512 元素）时沿 inference [W] 口径 + 「模型侧形式化归 models」指针。
  - **教学参照（+官方实现）** = 无独立论文的官方维护 repo / 库（NCCL / CUTLASS / cuda-samples / nvbandwidth / gdrcopy / nccl-tests / DCGM / open-gpu-kernel-modules / DeepEP / FlashMLA / DeepGEMM / CUB / Triton / rdma-core / TransformerEngine / PyTorch）。
  - **等级只取五个单值，禁 [A/B] 类混合档**（沿 inference references 写入规则）。
- key 为小写 slug，全文件唯一，取论文或资源的通用短名（如 `nccl`、`ptx-isa`）；paper 与 repo 分列条目时 repo 条目加 `-repo` 后缀。
- 章节写完（或引用关系变化）后回写「被引章节」；一条目被多章引用时逗号分隔累加（如 `8.1, 8.2`）。**跨 Part 共享锚点收录于首引 Part、被引章节跨组累加**（如 CUTLASS 首引 1.3 收录 P1，被引 1.3, 4.2, 4.3, 4.4, 5.2）。
- 正文引用格式 `[Dao et al., 2022]`，链接到本文件对应条目（gpu spec §4.2 / STYLE_GUIDE §6）；章间引用一律纯文字，活链接唯一例外 `../references.md`。
- **禁竖线**：条目单元格禁 `|` 含 escaped `\|`（naive parseReferences split('|') 丢条目，必须恰 7 字段）；arXiv ID 与 URL 之间、多 URL 之间一律用 ` ; ` 分隔（沿 inference 格式）。

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

> **W0 骨架收录说明（2026-11-24）**：下列 59 条种入自 gpu spec 附录「Spec 预核实记录」+ §5 各章 Sources 意图，核实日期统一 2026-11-24。核实通道：10 个 arXiv ID↔标题↔第一作者经 https export API 批量 200 复核（含 Ring Attention 2310.01889v4，W5 ⟨verify⟩ 提前销项）；22 个 repo 经 GitHub API 复核（21×200，openai/triton 301→triton-lang/triton、NVIDIA/cub archived→并入 NVIDIA/cccl）；NVIDIA 官方文档 URL 经 HTTP curl 复核（CUDA Programming Guide 旧 cuda-c-programming-guide/ 404→现行 cuda-programming-guide/ 200 销项、IBTA 旧 specifications-download/ 404→现行 ibta-specification/ 200 销项、MIG/Nsight Systems-Nsight Compute/XID/SHARP/GDS/vGPU 均 200）；老经典 4 篇标题 venue 经 Crossref 确认（Blelloch PDF 200 / Rabenseifner Springer 200 / Patarasuk-Yuan doi 200 / DCQCN 官方 proceedings PDF 200）；V4.1 config 经 hf-mirror 200 / 3311 B（text_config 嵌套，head_dim 512 / qk_rope 64 / 无 kv_lora_rank 字段，引擎导出 448 latent）。网络 fallback：huggingface.co 不可达→hf-mirror；ACM DL / JEDEC / PCI-SIG 403 bot 拦截→Crossref / doi / 官方 proceedings 互证；Hopper whitepaper PDF 镜像不可达→官方 landing 200 + Hopper blog / SM 框图 / Best Practices Table 5 互证。**⟨verify⟩ 残留延迟项**（来源真实、精确 URL / 数值留对应波）：JEDEC HBM3 精确出处（W1）、Hopper/Blackwell whitepaper + datasheet 精确数值（W1）、PCI-SIG 精确出处（W4）、NVLink 代际带宽 / GB200 NVL72 精确规格（W4）、Al-Fares fat-tree PDF 可达出处（W5）、peer-memory 状态（W5）、CUTLASS 当前版本架构 / TransformerEngine 路径（W3）。详见 `docs/research/gpu-w0-sources.md`。「被引章节」按 spec §5 Sources 意图填写，各波 research / writing pass 回写精炼。跨 Part 共享锚点收录首引 Part、被引跨组累加；P4 锚点均为跨 Part 共享（Best Practices / PTX / Nsight / CUTLASS → P1，cuda-samples → P3），本组空表、被引在对应条目累加 4.1-4.4。

### Part 1 · GPU 微架构拆解

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| hopper-whitepaper | NVIDIA Hopper Architecture Whitepaper（H100 SM 框图 / 4 warp scheduler 各单发射 / 4th Tensor Core / TMA；landing 200，PDF 镜像本机不可达⟨verify W1 精确数值〉，单发射以官方 SM 框图 + Best Practices Table 5 互证） | NVIDIA | https://resources.nvidia.com/en-us-hopper-architecture | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.1, 1.2, 2.1, 2.2, 4.4 |
| blackwell-whitepaper | NVIDIA Blackwell Architecture Whitepaper（B200 双 die + NVLink-C2C / 5th Tensor Core / tcgen05 / FP4；landing 200，精确数值⟨verify W1〉） | NVIDIA | https://resources.nvidia.com/en-us-blackwell-architecture | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.1, 1.3 |
| nvidia-datasheet | NVIDIA H100 / H200 / B200 官方 datasheet（dense = 官方 sparse 表值之半；H100 SXM BF16 989.5 / FP8 1979 TFLOPS dense、80 GB HBM3、3.35 TB/s、NVLink 900 GB/s 双向；H200 141 GB HBM3e、4.8 TB/s；HGX B200 BF16 2250 / FP8 4500 / FP4 9000 dense、180 GB、7.7 TB/s、NVLink5 1.8 TB/s；GB200 186 GB / 8 TB/s——SKU 勿混；沿 inference 1.3 已核实表，B200 精确数值⟨verify W1〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/h100/ ; https://www.nvidia.com/en-us/data-center/h200/ ; https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23 ; https://dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.1, 2.1, 2.4, 6.1, 7.1, 7.2, 11.1 |
| cuda-programming-guide | CUDA C++ Programming Guide（compute capability 表 / memory 节 / memory model / stream-event-graph / occupancy / P2P；**URL 漂移销项**：旧 cuda-c-programming-guide/ 404→现行 cuda-programming-guide/ 200） | NVIDIA | https://docs.nvidia.com/cuda/cuda-programming-guide/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.2, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 4.4, 7.3 |
| ptx-isa | NVIDIA PTX ISA（mma.sync / wgmma / tcgen05 / cp.async / TMA / ldmatrix / shfl / lop3-prmt-bfe / fence-atom / 低精度格式节） | NVIDIA | https://docs.nvidia.com/cuda/parallel-thread-execution/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.2, 1.3, 2.3, 3.1, 3.2, 4.1, 4.2, 4.4, 5.1, 5.2, 6.1, 6.2, 6.3 |
| cuda-best-practices | CUDA C++ Best Practices Guide（coalescing / shared / tiling / occupancy；Table 5 FP32 128 results/cycle/SM = 4 scheduler × 32 lane × 1 指令，Hopper 单发射互证） | NVIDIA | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.4, 2.2, 2.3, 3.4, 4.1, 4.2, 4.3 |
| nsight-compute | Nsight Compute docs（kernel profiling guide / stall reason / memory chart / roofline / occupancy metrics） | NVIDIA | https://docs.nvidia.com/nsight-compute/NsightCompute/index.html ; https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html | [C]（官方规格披露·一手权威） | 2026-11-24 | 1.4, 2.2, 2.4, 4.1, 11.2 |
| hopper-blog | NVIDIA Hopper Architecture In-Depth blog（SM 框图 / combined 256KB shared+L1 / shared 可配 228KB / 单发射；开放技术博客，whitepaper PDF 不可达时的一手互证源） | NVIDIA | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | [C]（开放技术博客、非闭源） | 2026-11-24 | 1.1, 1.2 |
| cutlass | NVIDIA/cutlass repo（CUTLASS GEMM 层级 device→kernel→threadblock→warp→instruction / CuTe layout-tile / MMA atom include/cute/arch/mma*.hpp / swizzle include/cute/swizzle*.hpp / pipeline include/cutlass/pipeline/；3.x/4.x 架构⟨verify W3 当前版本〉） | NVIDIA | https://github.com/NVIDIA/cutlass | 教学参照（官方实现） | 2026-11-24 | 1.3, 4.2, 4.3, 4.4, 5.2 |

### Part 2 · 内存层级与带宽账

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| jedec-hbm3 | JEDEC HBM3 规范（JESD238；栈结构 / channel-pseudo-channel / 1024bit per stack = 16 ch × 64bit；URL 403 bot 拦截⟨verify W1 精确出处〉，规范真实存在） | JEDEC | https://www.jedec.org/standards-documents/docs/jesd238a | [C]（官方规格披露·一手权威） | 2026-11-24 | 2.1 |
| nvbandwidth | NVIDIA/nvbandwidth repo（HBM / P2P / NVLink 带宽实测工具；2.4 / 7.2 / 11.2 测量方法） | NVIDIA | https://github.com/NVIDIA/nvbandwidth | 教学参照（官方实现） | 2026-11-24 | 2.4, 7.2, 11.2 |

### Part 3 · CUDA 编程模型

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| cuda-samples | NVIDIA/cuda-samples repo（reduction / scan / matrixMul / P2P / graph 教学例；3.1 / 3.3 / 4.1 / 4.3 / 5.1 / 5.3 / 7.3） | NVIDIA | https://github.com/NVIDIA/cuda-samples | 教学参照（官方实现） | 2026-11-24 | 3.1, 3.3, 4.1, 4.3, 5.1, 5.3, 7.3 |

### Part 4 · Kernel 访存优化

本 Part 锚点均为跨 Part 共享，收录于首引 Part：Best Practices / PTX ISA / Nsight Compute / CUTLASS → P1，cuda-samples → P3；被引章节在对应条目跨组累加 4.1-4.4，本组无独立首引条目。

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |

### Part 5 · Kernel 计算优化与并行原语

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| blelloch-scan | Prefix Sums and Their Applications（Blelloch scan up-sweep / down-sweep，work O(n_el) / depth O(log n_el)；CMU-CS-90-190，无 arXiv 无官方 repo） | Blelloch | https://www.cs.cmu.edu/~guyb/papers/Ble90.pdf（CMU-CS-90-190，无 arXiv） | [B]（老经典无 arXiv） | 2026-11-24 | 5.1 |
| cub | CUB（CUDA UnBound）cooperative primitives（block / warp reduction-scan；**NVIDIA/cub 已 archived→并入 NVIDIA/cccl**，CUB 为其子目录，原 repo README 自链 cccl） | NVIDIA | https://github.com/NVIDIA/cccl（原 NVIDIA/cub archived） | 教学参照（官方实现） | 2026-11-24 | 5.1 |
| cublaslt-docs | cuBLAS / cuBLASLt docs（GEMM heuristic 与 algo 选择） | NVIDIA | https://docs.nvidia.com/cuda/cublas/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 5.2 |
| deepgemm | deepseek-ai/DeepGEMM repo（FP8 / FP4 / BF16 GEMM + Mega MoE + grouped GEMM contiguous / masked layout；DeepJIT 运行时编译；SM90 FP32 scale vs SM100 packed UE8M0；沿 inference 已锚） | DeepSeek-AI | https://github.com/deepseek-ai/DeepGEMM | 教学参照（官方实现） | 2026-11-24 | 5.2, 6.2 |
| transformerengine | NVIDIA/TransformerEngine repo（norm / attention kernel，FP8；5.3 norm kernel 教学参照⟨verify W3 路径〉） | NVIDIA | https://github.com/NVIDIA/TransformerEngine | 教学参照（官方实现） | 2026-11-24 | 5.3 |
| nanoflow | NanoFlow: Towards Optimal Large Language Model Serving Throughput（intra-device nano-batch 流水 + execution unit scheduling 重叠 compute / memory / network；5.4 intra-device overlap recap，沿 inference 已锚） | Zhu et al. | arXiv:2408.12757 ; https://github.com/efeslab/Nanoflow | [A] | 2026-11-24 | 5.4 |
| triton | Triton repo（tile-level DSL / block-level 编程 / fusion pass；**openai/triton 301→triton-lang/triton**；Triton 论文 MAPL 2019 无 arXiv，spec §5 判教学参照） | triton-lang（原 OpenAI） | https://github.com/triton-lang/triton（原 openai/triton 301 重定向） | 教学参照（官方实现） | 2026-11-24 | 5.4 |
| tilelang | TileLang: A Composable Tiled Programming Model for AI Systems（tile 抽象 DSL / 多硬件后端；5.4 DSL fusion 对照，Triton 同类） | Wang et al. | arXiv:2504.17577 ; https://github.com/tile-ai/tilelang | [A] | 2026-11-24 | 5.4 |
| pytorch | PyTorch repo（Inductor fusion pass / dynamic shape guards / graph break；5.4 torch.compile recap） | PyTorch Team | https://github.com/pytorch/pytorch | 教学参照（官方实现） | 2026-11-24 | 5.4 |

### Part 6 · 低精度与注意力 kernel

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| ocp-mx | OCP Microscaling Formats (MX) Specification v1.0（MXFP8 / MXFP6 / MXFP4 / MXINT8 均 block size 32 + E8M0 scale；MXFP4 元素 FP4 E2M1；**NVFP4 非 OCP MX v1.0 格式**，归 NVIDIA 官方文档 / PTX kind::mxf4nvf4 .ue4m3 block16） | OCP | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf | [C]（官方规格披露·一手权威） | 2026-11-24 | 6.1 |
| marlin | MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models（FP16xINT4 mixed-input GEMM / dequant 融合 mainloop / lop3-bfe 指令级 / near-ideal 4×；inference W2 升 [A]，指令级走读归 gpu 6.2） | Frantar et al. | arXiv:2408.11743 ; https://github.com/IST-DASLab/marlin | [A] | 2026-11-24 | 6.2 |
| flashattention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（tiling + online softmax + rescale；GPU 侧精确 HBM 流量账；6.3 指令级走读，数学与 IO 定理归 models 4.1） | Dao et al. | arXiv:2205.14135 ; https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-24 | 6.3 |
| flashattention2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（warp partitioning / seq 维并行；6.3 指令级 warp 分工） | Dao | arXiv:2307.08691 ; https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-24 | 6.3 |
| flashattention3 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（warp specialization + pingpong + TMA + FP8，Hopper 异步；6.3） | Shah et al. | arXiv:2407.08608 ; https://github.com/Dao-AILab/flash-attention | [A] | 2026-11-24 | 6.3 |
| flashmla | deepseek-ai/FlashMLA repo（MLA decode kernel；**版本锚** main V4.1 sparse / fused + Hopper absorbed 机制 commit ba89a34 + docs/20250422-new-kernel-deep-dive.md；2026.09.30 移除 Hopper / V3 支持；沿 inference 已锚） | DeepSeek-AI | https://github.com/deepseek-ai/FlashMLA | 教学参照（官方实现） | 2026-11-24 | 6.3, 6.4 |
| flashinfer | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving（block-sparse BSR / composable formats / JIT / split-KV scheduler；6.4 变体 kernel，沿 inference 已锚） | Ye et al. | arXiv:2501.01005 ; https://github.com/flashinfer-ai/flashinfer | [A] | 2026-11-24 | 6.4 |
| vllm | Efficient Memory Management for Large Language Model Serving with PagedAttention（paged attention block table gather kernel；6.4 GPU 侧实现，serving 系统归 inference，沿 inference 已锚） | Kwon et al. | arXiv:2309.06180 ; https://github.com/vllm-project/vllm | [A] | 2026-11-24 | 6.4 |
| deepseekv3 | DeepSeek-V3 Technical Report（MLA 几何 V3 576 = 512 latent + 64 rope / EP DualPath 通信；6.4 MLA kernel 几何 + 10.3 EP a2a 锚；勿与 V4.1 512 = 448 + 64 混） | DeepSeek-AI | arXiv:2412.19437 ; https://hf-mirror.com/deepseek-ai/DeepSeek-V3 | [A]（模型锚点：paper + open weights） | 2026-11-24 | 6.4, 10.3 |
| v41-flash-config | DeepSeek-V4.1-Flash-1017 config.json（MLA 几何 head_dim 512 = 448 latent + 64 rope，config 无 kv_lora_rank 字段、引擎导出 448；q_lora_rank 1280 / o_lora_rank 1024 / o_groups 8 / 40L / 64 heads；模型侧形式化归 models） | DeepSeek-AI | https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json（huggingface.co 本机不可达，hf-mirror 传输核实 200 / 3311 B，text_config 嵌套） | [W] | 2026-11-24 | 6.4 |

### Part 7 · 节点内互连 NVLink/NVSwitch

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| nvlink-nvswitch | NVIDIA NVLink / NVSwitch 产品页（NVLink 世代 / link 带宽 / NVSwitch crossbar / NVLink 域；H100 18 links × 25 GB/s 单向 = 450 单向 = 900 双向；B200 NVLink5 1.8 TB/s 双向⟨verify W4 精确代际值〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/nvlink/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.1 |
| gb200-nvl72 | NVIDIA GB200 NVL72 规格（72 GPU NVLink 域超节点 / NVLink-C2C Grace-Blackwell；⟨verify W4 / W6 精确规格〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.1, 10.1, 11.1 |
| hgx-dgx | NVIDIA HGX / DGX 服务器规格（HGX H100 8-GPU baseboard / DGX H100 整机 / 供电散热；⟨verify W6 精确数值〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/hgx/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.2, 11.1 |
| pcie-spec | PCI-SIG PCIe 规范（Gen5 x16 = 64 GB/s 单向；URL 403 bot 拦截⟨verify W4 精确出处〉，规范真实存在） | PCI-SIG | https://pcisig.com/specifications | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.2 |
| gpudirect-rdma-docs | NVIDIA GPUDirect RDMA docs（GDR NIC 直访 HBM / peer memory / BAR1；GPUDirect 家族 P2P / RDMA / Storage / Video；7.3 总览 + 9.2 GDR） | NVIDIA | https://docs.nvidia.com/cuda/gpudirect-rdma/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.3, 9.2 |
| gds-docs | NVIDIA GPUDirect Storage docs（GDS NVMe→HBM 直访 / cuFile API） | NVIDIA | https://docs.nvidia.com/gpudirect-storage/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.3 |

### Part 8 · NCCL 集合通信

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| nccl | NVIDIA/nccl repo（channel 构建 src/graph/ / transport P2P-SHM-NET src/transport/ / proxy 线程 / LL-LL128-Simple 协议 / ring-tree-NVLS-CollNet 算法选择 / kernel src/collectives/device/；8.4 模块全量走读⟨verify W4 当前版本 2.x〉） | NVIDIA | https://github.com/NVIDIA/nccl | 教学参照（官方实现） | 2026-11-24 | 8.1, 8.2, 8.3, 8.4, 9.2, 10.2, 10.3, 10.4 |
| nccl-docs | NCCL User Guide（集合原语语义 / API ncclAllReduce / 算法协议 / env NCCL_ALGO / NCCL_PROTO / NCCL_TOPO_FILE） | NVIDIA | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 8.1, 8.3, 8.4 |
| patarasuk-yuan | Bandwidth optimal all-reduce algorithms for clusters of workstations（ring all-reduce 带宽最优；JPDC 69(2) 2009，无 arXiv） | Patarasuk et al. | https://doi.org/10.1016/j.jpdc.2008.09.002（JPDC 69(2)，无 arXiv；ScienceDirect pii S0743731508001767） | [B]（老经典无 arXiv） | 2026-11-24 | 8.2 |
| rabenseifner | Optimization of Collective Reduction Operations（recursive halving / doubling reduction；Springer LNCS 3036 ICCS 2004，无 arXiv） | Rabenseifner | https://link.springer.com/chapter/10.1007/978-3-540-24685-5_1（LNCS 3036，无 arXiv） | [B]（老经典无 arXiv） | 2026-11-24 | 8.2, 8.3 |
| nccl-tree-blog | Massively Scale Deep Learning Training with NCCL 2.4（double binary tree 两棵互补树覆盖所有 rank / 带宽项 ~2M / 延迟 O(log n)；开放技术博客） | NVIDIA | https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/ | [C]（开放技术博客、非闭源） | 2026-11-24 | 8.3, 8.4 |
| nccl-tests | NVIDIA/nccl-tests repo（all-reduce / all-gather 等集合带宽-延迟基准；8.3 tuning 表 / 10.2 万卡实测 / 11.2 通信基准） | NVIDIA | https://github.com/NVIDIA/nccl-tests | 教学参照（官方实现） | 2026-11-24 | 8.3, 10.2, 11.2 |

### Part 9 · RDMA 与 GPUDirect

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| ibta-spec | IBTA InfiniBand Architecture Specification（RDMA 语义 / QP-CQ-MR-WR / IB 拥塞控制 / 链路层；**URL 销项**：旧 specifications-download/ 404→现行 ibta-specification/ 200） | IBTA | https://www.infinibandta.org/ibta-specification/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 9.1, 9.3, 10.1, 10.2 |
| nvidia-networking-docs | NVIDIA Networking docs（ConnectX NIC / Quantum switch / RoCE / verbs 编程手册 / 拥塞控制文档；⟨verify W5 精确子页〉） | NVIDIA | https://docs.nvidia.com/networking/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 9.1, 9.3, 10.1, 10.2 |
| rdma-core | linux-rdma/rdma-core repo（verbs 用户态库 / libibverbs / man pages；9.1 verbs API 教学参照） | linux-rdma | https://github.com/linux-rdma/rdma-core | 教学参照（官方实现） | 2026-11-24 | 9.1 |
| open-gpu-kernel-modules | NVIDIA/open-gpu-kernel-modules repo（nvidia-peermem GDR 内核模块；原 Mellanox/peer-memory 404 改锚⟨verify W5 peer-memory 状态〉） | NVIDIA | https://github.com/NVIDIA/open-gpu-kernel-modules | 教学参照（官方实现） | 2026-11-24 | 9.2 |
| gdrcopy | NVIDIA/gdrcopy repo（低延迟 CPU→GPU 拷贝 / GDR 用户态） | NVIDIA | https://github.com/NVIDIA/gdrcopy | 教学参照（官方实现） | 2026-11-24 | 9.2 |
| dcqcn | Congestion Control for Large-Scale RDMA Deployments（DCQCN：ECN 标记 + CNP 反馈 + rate adjust；SIGCOMM 2015，无 arXiv；ACM DL 403 bot 拦截，官方 proceedings PDF 200 + Crossref 确认） | Zhu et al. | https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf（SIGCOMM 2015，无 arXiv；DOI 10.1145/2785956.2787484） | [B]（老经典无 arXiv） | 2026-11-24 | 9.3 |

### Part 10 · 机间 fabric 与大规模通信

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| al-fares | A Scalable, Commodity Data Center Network Architecture（fat-tree / Clos 无阻塞收敛；SIGCOMM 2008，无 arXiv；ACM DL 403 bot 拦截 + PDF 镜像不可达⟨verify W5 精确出处〉，Crossref 确认标题 venue） | Al-Fares et al. | https://doi.org/10.1145/1402958.1402967（SIGCOMM 2008，无 arXiv） | [B]（老经典无 arXiv） | 2026-11-24 | 10.1 |
| sharp-docs | NVIDIA SHARP docs（Scalable Hierarchical Aggregation and Reduction Protocol，IB 交换机内聚合 / CollNet 集成；v3.0 200） | NVIDIA | https://docs.nvidia.com/networking/display/SHARPv300 | [C]（官方规格披露·一手权威） | 2026-11-24 | 10.2 |
| deepep | deepseek-ai/DeepEP repo（MoE dispatch / combine；**双锚** tag v1.2.1 机制（normal NVLink+RDMA 两阶段 / low-latency 纯 RDMA hook IBGDA）+ main V2.5 接口（统一 EPBuffer / NCCL Gin / NVSHMEM 移除）；沿 inference 已锚，10.3 模块走读） | DeepSeek-AI | https://github.com/deepseek-ai/DeepEP | 教学参照（官方实现） | 2026-11-24 | 9.2, 10.3 |
| ring-attention | Ring Attention with Blockwise Transformers for Near-Infinite Context（ring pass KV 块轮转 / compute-comm overlap；10.4 通信调度，分块机制归 models 4.6；沿 training/references.md 已核实 2026-11-17，W0 arXiv API 再核 2310.01889v4） | Liu et al. | arXiv:2310.01889 | [A] | 2026-11-24 | 10.4 |

### Part 11 · 系统集成与生产

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| mig-user-guide | NVIDIA MIG User Guide（Multi-Instance GPU 硬件分区 / GPU instance + compute instance / SM-L2-HBM 切分 / 最多 7 实例） | NVIDIA | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 11.1 |
| vgpu-docs | NVIDIA vGPU docs（vGPU / SR-IOV 硬件虚拟化机制） | NVIDIA | https://docs.nvidia.com/vgpu/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 11.1 |
| nsight-systems | Nsight Systems docs（系统级 timeline / kernel launch-copy-API trace / CPU-GPU 重叠 / NVTX 标注） | NVIDIA | https://docs.nvidia.com/nsight-systems/UserGuide/index.html | [C]（官方规格披露·一手权威） | 2026-11-24 | 11.2 |
| dcgm | NVIDIA/DCGM repo（Data Center GPU Manager 监控 / diag 健康检查 / ECC-温度-功率） | NVIDIA | https://github.com/NVIDIA/DCGM | 教学参照（官方实现） | 2026-11-24 | 11.2, 11.3 |
| xid-docs | NVIDIA XID Error Codes docs（XID 31 / 43 / 48 / 63 / 74 / 79 语义 / GPU 故障分类） | NVIDIA | https://docs.nvidia.com/deploy/xid-errors/index.html | [C]（官方规格披露·一手权威） | 2026-11-24 | 11.3 |
| nvidia-smi-docs | NVIDIA nvidia-smi docs（GPU 状态查询 / ECC 计数 / 温度功率监控 / 健康检查） | NVIDIA | https://docs.nvidia.com/deploy/nvidia-smi/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 11.3 |

## 术语表

全书术语累积表：章节首现概念按「中文（English Term）」格式处理（STYLE_GUIDE §1）后在此登记——「中文处理」记录该术语的全书统一写法（译名或保留英文），保证跨章一致。各波次追加。W0 骨架先种核心 GPU / CUDA / 通信术语，「首次出现章节」按 gpu spec §5 大纲意图填写，writing pass 回写精炼；符号约定（n / n_el / M / bytes / T_M / α / β / I / R / 𝒪 等）以 gpu spec §4.3 统一表为准，不在本表重复。

| 英文术语 | 中文处理 | 首次出现章节 |
| --- | --- | --- |
| SM (streaming multiprocessor) | 保留英文（SM）；流多处理器 | 1.1 |
| GPC / TPC | 保留英文（GPC / TPC） | 1.1 |
| chiplet / CoWoS | 保留英文（chiplet / CoWoS 封装） | 1.1 |
| warp scheduler | warp 调度器（warp scheduler） | 1.2 |
| CUDA Core | 保留英文（CUDA Core） | 1.2 |
| Tensor Core | 保留英文（Tensor Core） | 1.2 |
| register file | 寄存器堆（register file） | 1.2 |
| occupancy | 占用率（occupancy） | 1.2 |
| MMA (matrix multiply-accumulate) | 保留英文（MMA） | 1.3 |
| wgmma / tcgen05 | 保留英文（Hopper wgmma / Blackwell tcgen05） | 1.3 |
| latency hiding | 延迟隐藏（latency hiding） | 1.4 |
| ILP / TLP | 保留英文（ILP / TLP，指令级 / 线程级并行） | 1.4 |
| HBM | 保留英文（HBM，高带宽内存） | 2.1 |
| TSV | 保留英文（TSV，硅通孔） | 2.1 |
| L2 cache | 保留英文（L2 cache） | 2.2 |
| shared memory | 共享内存（shared memory） | 2.2 |
| sector | 保留英文（sector，32B 访存粒度） | 2.2 |
| arithmetic intensity | 算术强度（arithmetic intensity） | 2.4 |
| roofline | 保留英文（roofline） | 2.4 |
| ridge point | 保留英文（ridge point） | 2.4 |
| warp | 保留英文（warp，32 线程） | 3.1 |
| SIMT | 保留英文（SIMT，单指令多线程） | 3.1 |
| divergence | 分支分歧（divergence） | 3.1 |
| thread block | 线程块（thread block） | 3.1 |
| memory model | 内存模型（memory model） | 3.2 |
| fence | 保留英文（fence） | 3.2 |
| stream / event | 保留英文（stream / event） | 3.3 |
| CUDA Graph | 保留英文（CUDA Graph） | 3.3 |
| copy engine | 拷贝引擎（copy engine） | 3.3 |
| coalescing | 访存合并（coalescing） | 4.1 |
| bank conflict | bank 冲突（bank conflict） | 4.2 |
| swizzle | 保留英文（swizzle） | 4.2 |
| tiling | 分块（tiling） | 4.3 |
| double buffering | 双缓冲（double buffering） | 4.3 |
| cp.async / TMA | 保留英文（cp.async / TMA，Tensor Memory Accelerator） | 4.4 |
| reduction / scan | 归约 / 扫描（reduction / scan） | 5.1 |
| warp shuffle | 保留英文（warp shuffle） | 5.1 |
| GEMM | 保留英文（GEMM） | 5.2 |
| epilogue | 保留英文（epilogue） | 5.2 |
| online softmax | 保留英文（online softmax）；正确性证明归 models 4.1 | 5.3 |
| operator fusion | 算子融合（operator fusion） | 5.4 |
| FP8 / FP4 / BF16 / TF32 | 保留英文（低精度格式名） | 6.1 |
| MXFP8 / NVFP4 | 保留英文（MX 族 / NVIDIA FP4）；NVFP4 scale UE4M3 非 OCP MX | 6.1 |
| dequant | 反量化（dequant） | 6.2 |
| FlashAttention | 保留英文（FlashAttention） | 6.3 |
| MLA (multi-head latent attention) | 保留英文（MLA）；低秩推导归 models 4.1 | 6.4 |
| GQA / MQA | 保留英文（GQA / MQA） | 6.4 |
| split-KV / FlashDecoding | 保留英文（split-KV / FlashDecoding） | 6.4 |
| NVLink / NVSwitch | 保留英文（NVLink / NVSwitch） | 7.1 |
| NVLink-C2C | 保留英文（NVLink-C2C） | 7.1 |
| NVLS (NVLink SHARP) | 保留英文（NVLS） | 7.1 |
| P2P (peer-to-peer) | 保留英文（P2P，点对点） | 7.3 |
| GPUDirect | 保留英文（GPUDirect） | 7.3 |
| all-reduce / all-gather / reduce-scatter | 保留英文（集合原语名） | 8.1 |
| all-to-all | 保留英文（all-to-all） | 8.1 |
| ring / tree | 保留英文（ring / tree 算法） | 8.2 |
| double binary tree | 双二叉树（double binary tree） | 8.3 |
| LL / LL128 / Simple | 保留英文（NCCL 协议名） | 8.4 |
| channel | 保留英文（channel） | 8.4 |
| RDMA | 保留英文（RDMA，远程直接内存访问） | 9.1 |
| verbs / QP / CQ / MR / WR | 保留英文（verbs / QP / CQ / MR / WR） | 9.1 |
| RoCE / InfiniBand | 保留英文（RoCE / InfiniBand） | 9.1 |
| GDR (GPUDirect RDMA) | 保留英文（GDR） | 9.2 |
| nvidia-peermem | 保留英文（nvidia-peermem） | 9.2 |
| DCQCN / PFC / ECN | 保留英文（拥塞控制机制名） | 9.3 |
| fat-tree / Clos | 保留英文（fat-tree / Clos） | 10.1 |
| rail-optimized | 保留英文（rail-optimized） | 10.1 |
| oversubscription | 收敛比（oversubscription） | 10.1 |
| SHARP | 保留英文（SHARP，网内聚合） | 10.2 |
| EP (expert parallelism) | 保留英文（EP）；训练账归 training 5.4 | 10.3 |
| dispatch / combine | 保留英文（dispatch / combine） | 10.3 |
| Ring Attention | 保留英文（Ring Attention）；分块机制归 models 4.6 | 10.4 |
| CP (context parallelism) | 保留英文（CP）；训练账归 training 5.2 | 10.4 |
| MIG (multi-instance GPU) | 保留英文（MIG，多实例 GPU） | 11.1 |
| vGPU / SR-IOV | 保留英文（vGPU / SR-IOV） | 11.1 |
| Nsight Systems / Nsight Compute | 保留英文（Nsight Systems / Nsight Compute） | 11.2 |
| XID | 保留英文（XID 错误码） | 11.3 |
| ECC | 保留英文（ECC，纠错码） | 11.3 |
