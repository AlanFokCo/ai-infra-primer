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

> **W1 research pass 精炼说明（2026-11-25）**：P1/P2 全部条目经一手复核——Hopper whitepaper V1.04（含 final clocks/TFLOPS）与 Blackwell Technical Brief、H100/H200/HGX-H100-H200/Blackwell 四份 datasheet 均经 **dam-cdn.nvd.orangelogic.com 官方 DAM CDN PDF 一手取得**（销 W0「Hopper PDF 镜像不可达」残留；resources.nvidia.com 落地页内嵌 AssetLink 为入口）；SM 单发射经 whitepaper Fig 7 SM 框图（4 processing block 各 1 Warp Scheduler + 1 Dispatch Unit，32 thread/clk）+ Best Practices Table 5（CC 9.0 FP32 128 results/cycle/SM）互证；CC 9.0/10.0 资源库存经 Programming Guide compute-capabilities 子页 Table 33/34/35 + Hopper/Blackwell Tuning Guide 互证（**CUDA 13.x Programming Guide 拆子页 URL 漂移**已注）；HBM 栈几何逐项（n_stack/n_ch/f_pin/w_ch）经 whitepaper Table 3（5 栈 5120-bit 2619 MHz DDR→3352 GB/s）+ Micron/Samsung/SK hynix 成员官方（16 ch×64-bit=1024 IO、32 pseudo、6.4 Gbps→819 GB/s）互证，JEDEC 原文 403 留残留；MMA 几何（mma.sync m16n8k16 / wgmma m64nNk16 / tcgen05 + TMEM 512×128×32bit）经 PTX ISA 逐字；延迟表（FMA 4 cycle 官方 prose + HBM/L2/shared 实测 cycles）经 Best Practices + Nsight + Hopper 微基准 [B] 支撑。**新增 5 条**：hopper-tuning-guide / blackwell-tuning-guide（[C]）、hopper-microbench / blackwell-microbench / b200-power-bench（[B] 微基准，延迟与 SKU raw-vs-usable 实测注）。**blelloch-scan URL 修**：改 CMU-CS-90-190 报告页 + 页链 Ble93.pdf（首页标题逐字），旧 Ble90.pdf（Vector Models 书锚/截断）移除。**SKU 口径**：HGX B200 180GB/7.7 vs GB200 186GB/8 vs raw 192（brief「Up to 192 GB」/ nvidia-smi 179 GiB）勿混。**残留**：B200 SM 数官方未披露（⟨verify W3/W6〉）、JEDEC 精确 URL（403，同 PCI-SIG W4）。详见 `docs/research/gpu-w1-sources.md`。parse-check：11 groups / 64 entries。**S3 fix 轮被引回写（2026-11-25）**：8 章实际引用 forward+reverse 交叉核——hopper-whitepaper / blackwell-whitepaper / hopper-microbench / cuda-programming-guide 等 12 条补 1.1/1.3/2.1-2.4 实际被引（hopper-whitepaper += 1.4 已在 W1 research pass 回写，本轮核实闭合），hopper-blog 移除未实引的 1.2；nvbandwidth README 核实日期回写 2026-11-25；nvidia-datasheet 脚注归属精度（逐字折半句仅 Blackwell）+ b200-power-bench 270 GB 归属 brief（T6 S3-1/S3-4/S3-5）。

> **W2 research pass 精炼说明（2026-11-25）**：P3/P4 全部锚点经一手复核——CUDA Programming Guide **13.5 官方 PDF 一手**（12.78 MB，docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf）+ archive/13.0.0 legacy 单页互证（**13.x 子页再重组**：SIMT Execution Model warp divergence 逐字 → 03-advanced/advanced-kernel-programming.html §3.2.2.1；Thread Hierarchy + Kernel Launch and Occupancy（CC 10.0 例 32 blocks/233472 B/65536 regs/2048 threads）→ 02-basics/writing-cuda-kernels.html §2.3.2/2.3.7；Memory Fence Functions（__threadfence_block/__threadfence/__threadfence_system = cuda::atomic_thread_fence seq_cst block/device/system scope，fence 只 ordering 不保证 visibility）→ 05-appendices/cpp-language-extensions.html §5.4.4.3；CUDA C++ Memory Model → 05-appendices/cuda-cpp-memory-model.html §5.8；cooperative-groups / cuda-graphs / async-copies（LDGSTS CC8.0+ / TMA CC9.0+ cuTensorMapEncodeTiled __grid_constant__）/ async-barriers / unified-memory（page fault 三步迁移）/ stream-ordered-memory-allocation（cudaMallocAsync）/ pipelines 子页均 200）；PTX ISA **9.5**（fence.sc/membar sm_70+ 同义逐字、atom .sem/.scope、cp.async.ca/cg + commit_group/wait_group、cp.async.bulk.tensor 1D-5D + tensor-map 128B opaque + mbarrier::complete_tx::bytes + multicast::cluster、mbarrier phase+tx-count、ldmatrix.m8n8、ld.global.v4 ≤128bit、cache operators .ca「multiple L1 caches are not coherent for global data」逐字、state spaces、divergence .uni）；Best Practices **13.5**（coalescing CC6.0+「number of 32-byte transactions necessary to service all of the threads of the warp」逐字 + stride 饱和 32 segments + 32 banks×32bit/cycle + broadcast/multicast + padding 33≡unit stride（V100 140.2→199.4 GB/s）+ occupancy 定义/非越高越好 + register pressure ILP + async copy 免中间 register）；Hopper whitepaper **TMA + Async Transaction Barrier 节逐字**（dam-cdn V1.04 重取：copy descriptor tensor dimensions+block coordinates / single-thread issue / A100 LDGSTS vs Hopper TMA 硬件地址生成 / transaction count byte count / Hopper wait sleep vs Ampere spin）；**Fermi whitepaper GigaThread 逐字**（www.nvidia.com PDF 本轮可达，销 spec 3.1 GigaThread 锚；**GigaThread 术语 PG 8.0-13.5 均无** → legacy term 注，现行 PG 无名描述 blocks enumerated and distributed）；**CUDA Graphs blog**（2019-09-05）V100/CUDA 10.1 launch 开销实测 2.9/9.6/3.8/3.4 μs + instantiate ~400 μs 一次性 + 首 launch ~33% 慢（3.3 ~μs 口径官方出处，时代注）；**libcu++ memory model** thread_scope 五 scope 逐字（**URL 漂移**：旧 /cccl/libcudacxx/extended_api/ 404 → 现行 /cccl/unstable/libcudacxx/extended_api/memory_model.html 200）；Nsight Compute **Occupancy Calculator §4.13「replaces the previously provided CUDA Occupancy Calculator spreadsheet」逐字**（BP 交叉引用同 URL → 3.4 occupancy calculator 归 nsight-compute 条目）+ ProfilingGuide **Sectors/Req 最优 32-bit:4 / 64-bit:8 / 128-bit:16** + Requests/Wavefronts/Bytes=sectors×32B；**cuda-samples 重组 Samples/N_Name → cpp/N_Name**（CUDA 13.5，GitHub API master 树路径级核实：matrixMul / simpleCooperativeGroups / reduction / scan / threadFenceReduction / globalToShmemAsyncCopy / graphConditionalNodes / graphMemoryNodes / simpleCudaGraphs / alignedTypes / cudaGraphsPerfScaling）；**CUTLASS 4.9.0（2026-11-13）版本锚** + swizzle.hpp（Swizzle<B,M,S> XOR 逐字）/ swizzle_layout.hpp / pointer_swizzle.hpp + pipeline/{pipeline,sm90_pipeline,sm100_pipeline,sm107_pipeline}.hpp（PipelineState + PipelineTmaAsync full/empty_barrier[Stages] producer_acquire/commit/expect_transaction + consumer_wait/release）路径级核实，**gemm_tutorial.md 已不存在 → efficient_gemm.md**（docs 拆 cutlass-docs 条目），PipelineTmaUmmaAsync 不在当前 main（版本漂移注）。**新增 4 条**：fermi-whitepaper / cuda-graphs-blog / libcudacxx-memory-model（P3 [C]）+ cutlass-docs（P4 教学参照，4.3 首引）。**⟨verify⟩ 销项**：CUTLASS swizzle/pipeline 路径、occupancy calculator 现行形态、GigaThread 出处、graph launch ~μs 出处、PG 13.5 子页地图、cuda-samples 现行路径。**残留**：CUTLASS 3.x/4.x 架构全量走读留 W3（5.2）；B200 SM 数 W3/W6（W1 残留不变）。**volatile 语义漂移**：PG 13.x 明确 volatile 不适合 inter-thread synchronization（用 cuda::atomic_ref/atomic 替代），旧 volatile 跨 block 可见性模式 deprecated——3.2 须用新口径（fence 节 last-block 示例仍留 volatile 为内部不一致，writer 以 Volatile Qualifier 节为准）。**被引回写**：nsight-compute += 3.4；cutlass 4.3 → cutlass-docs（docs 拆分）；其余按 spec §5 Sources 维持。parse-check：11 groups / 68 entries。详见 `docs/research/gpu-w2-sources.md`。

### Part 1 · GPU 微架构拆解

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| hopper-whitepaper | NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（Hopper；含 final GPU/memory clocks 与 final TFLOPS；Fig 6 GH100 full 144 SM + 两 L2 分区 + 6 HBM3 栈/12 控制器；Fig 7 SM 框图 4 processing block 各 1 Warp Scheduler (32 thread/clk) + 1 Dispatch Unit (32 thread/clk) + RF 16384×32bit + 1 Tensor Core 4th → 单发射互证；Table 3 H100 SXM5 132 SM / TC-boost 1830 MHz 与 FP32-FP64-boost 1980 MHz / BF16 dense 989.4 / 5120-bit HBM3 5 栈 2619 MHz DDR → 3352 GB/s / L2 50 MB / shared 228 KB / RF 256 KB；Table 4 CC 9.0 64 warps 32 blocks 65536 regs 255/thread；**W2 TMA 节（p.32-34）逐字**：「TMA operations are launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing」1D-5D tensors / single-threaded programming model（warp 内 elected single thread issue cuda::memcpy_async，多线程 cuda::barrier wait）/ A100 LoadGlobalStoreShared 线程负责全部地址生成+loop vs Hopper TMA 硬件接管 stride-offset-boundary 计算 / **Asynchronous Transaction Barrier**：split arrive/wait + 「instead of counting just thread arrivals, it also counts transactions... The transaction count is essentially a byte count」+ sum transaction counts reaches expected value + Hopper waiting threads sleep（vs Ampere spin on shared barrier）→ 4.4 mbarrier expect_tx/complete_tx 硬件 backing；PDF 一手取得销 W0 镜像不可达残留） | NVIDIA | https://resources.nvidia.com/en-us-hopper-architecture ; https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 4.4 |
| blackwell-whitepaper | NVIDIA Blackwell Architecture Technical Brief（含 Blackwell Ultra GB300 更新版；208B 晶体管 TSMC 4NP / 双 reticle-limit die + NV-HBI 10 TB/s 单一致芯片 / 5th Tensor Core FP4-FP6 + OCP MX / 18 NVLink5 links 1.8 TB/s / B200 raw Up to 192 GB HBM3e 7.7 TB/s TDP 1000W PCIe Gen5 / GB200 8 TB/s；B200 SM 数未披露⟨verify 残留 W3/W6〉；PDF 一手取得） | NVIDIA | https://resources.nvidia.com/en-us-blackwell-architecture ; https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.3, 2.1, 2.2, 2.4 |
| nvidia-datasheet | NVIDIA H100 / H200 / HGX H100-H200 / Blackwell 官方 datasheet（dense = 官方 sparse 表值之半；逐字折半脚注 "Specifications in sparse. Dense is one-half of the sparse spec shown." 仅 Blackwell datasheet，H100/H200/HGX 脚注为 "*With sparsity"；H100 SXM BF16 1979*→989.5 dense / FP8 3958*→1979 / TF32 989*→494.5 / FP32 67 / FP64 34 / 80 GB HBM3 / 3.35 TB/s / ≤700W；H200 SXM 141 GB HBM3e / 4.8 TB/s / ≤700W、NVL ≤600W；HGX B200 BF16 4.5PF*→2250 / FP8 9PF*→4500 / FP4 18PF*→9000 / TF32 2.2PF*→1100 / FP32 75 / FP64 37 / 180 GB / 7.7 TB/s / ≤1000W；GB200 186 GB / 8 TB/s / BF16 5PF*→2500 / ≤1200W——SKU 勿混；H100 NVL 94GB 3.9TB/s 350-400W 勿混；四份 PDF 一手取得销 W1 残留） | NVIDIA | https://www.nvidia.com/en-us/data-center/h100/ ; https://www.nvidia.com/en-us/data-center/h200/ ; https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23 ; https://dam-cdn.nvd.orangelogic.com/AssetLink/u5hh6fv4r7564i4y673484y3m20083nj.pdf ; https://dam-cdn.nvd.orangelogic.com/AssetLink/b2kt2580e5b6424if22lg65223m6ghjb.pdf ; https://dam-cdn.nvd.orangelogic.com/AssetLink/44wa8u63l12ps4ie2870214xa2exoyra.pdf ; https://dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.3, 2.1, 2.4, 6.1, 7.1, 7.2, 11.1 |
| cuda-programming-guide | CUDA C++ Programming Guide（**13.5 官方 PDF 一手 + archive/13.0.0 legacy 互证**；**URL 结构漂移**：CUDA 13.x 拆子页且 13.5 再重组——SIMT Execution Model（warp=32 threads、divergence 串行执行各分支路径 disabling 非路径线程、Volta Independent Thread Scheduling per-thread PC/call stack + schedule optimizer sub-warp reconverge）→ 03-advanced/advanced-kernel-programming.html §3.2.2.1；Thread Hierarchy（threadIdx/blockIdx/blockDim/gridDim、tid=x+y·Dx+z·Dx·Dy、block≤1024 threads 同 SM）+ Kernel Launch and Occupancy（§2.3.7：scheduler 分配 blocks 无 ordering 保证、occupancy=active warps/max warps、CC 10.0 例 32 blocks/233472 B shared/65536 regs/2048 threads、768 threads/block→2 blocks→75%、100KB shared→2 blocks、cuda::arch_traits）→ 02-basics/writing-cuda-kernels.html；Memory Fence Functions（§5.4.4.3：__threadfence_block/__threadfence/__threadfence_system = cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_block/device/system)，fence 只 affect ordering 不保证 visibility，建议 libcu++ fence）+ Atomic Functions（relaxed ordering、_system/无后缀/_block scope、warp 内同址 atomic 串行但 order undefined → 非确定性）+ Synchronization Functions（__syncthreads 全 block 到达 + prior global/shared 访问可见、conditional 须全 block 同值；__syncwarp mask）+ Volatile Qualifier（**13.x volatile 不适合 inter-thread synchronization，用 cuda::atomic_ref/atomic**）+ Launch Bounds（§5.4.3.2：__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster) → .maxntid/.minnctapersm/.maxclusterrank，register 上限 L 推导 + spill 权衡）→ 05-appendices/cpp-language-extensions.html；CUDA C++ Memory Model → 05-appendices/cuda-cpp-memory-model.html §5.8；Variable Memory Space Specifiers（__device__/__constant__/__shared__、无 specifier 自动变量→register 可能 spill local）+ Device Memory Accesses（global 32/64/128B naturally-aligned transactions、1/2/4/8/16B words、cudaMalloc ≥256B 对齐、local memory resides in device memory cached L2、constant cached constant cache、texture cached texture cache 2D locality、shared on-chip banks n-way conflict）；cooperative-groups（grid.sync() + cudaLaunchCooperativeKernel co-residency + tiled_partition<32> + coalesced_threads）/ cuda-graphs（definition/instantiation/execution 三阶段、CPU launch costs reduced、stream capture cudaStreamBeginCapture/EndCapture、graph update、conditional nodes、device graph launch）/ async-copies（LDGSTS CC8.0+ / TMA CC9.0+：tensor map cuTensorMapEncodeTiled host 创建 __grid_constant__ const 参数、1D bulk 免 tensor map、Table 25 global→shared mbarrier 完成 / shared→global bulk async-group）/ async-barriers / unified-memory（page fault 三步：owner PTE 失效 + requester PTE 建立 + 物理页迁移 cost∝page size；Grace Hopper 硬件一致免 fault）/ stream-ordered-memory-allocation（cudaMallocAsync/cudaFreeAsync stream-ordered pool）/ pipelines 子页均 200；Asynchronous Concurrent Execution（default stream legacy NULL vs --default-stream per-thread、implicit synchronization NULL stream 阻断并发除非 cudaStreamNonBlocking、cudaStreamWaitEvent 跨 stream 依赖、asyncEngineCount>0 copy-kernel overlap / =2 双向 overlap + page-locked、cudaStreamCreateWithPriority hints 非保证不抢占运行中任务、Compute Preemption Pascal+ instruction-level、CUDA_DEVICE_MAX_CONNECTIONS work queues）；compute capability 表 05-appendices/compute-capabilities.html Table 33/34/35（W1：CC 9.0/10.0 = 64 warps / 32 blocks / 2048 threads / 64K regs / 255 per-thread / shared 228 KB per-SM / 227 KB per-block opt-in / 32 banks / constant 64 KB；Table 35 unified 256 KB carveout 0-228；KB=1024 注）；根 URL 200 维持，旧 cuda-c-programming-guide/ 404 已销） | NVIDIA | https://docs.nvidia.com/cuda/cuda-programming-guide/ ; https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf ; https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/ ; https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.4, 7.3 |
| ptx-isa | NVIDIA PTX ISA（**9.5 版**；mma.sync m16n8k16 bf16/f16 f32-acc 逐字；wgmma.mma_async m64nNk16 N∈8..256 warpgroup 异步 / m64nNk8 tf32 / m64nNk32 f8f6f4；tcgen05.mma/.sp/.ws 5th-gen + Tensor Memory sm_100 512 columns×128 lanes×32bit=256 KB/CTA + cta_group::1/::2 双 CTA 协作 + kind::mxf4nvf4 block scaling；**W2 逐字核实**：§9.5 Divergence of Threads（divergent/uniform、.uni suffix、optimizing code generator 决定 reconvergence 点）；Memory Consistency Model（morally strong 三条件 + causality order + Fence-SC order + SB litmus、weak/strong/acquire/release operation 定义、texture 含 ld.global.nc 不适用 memory model）；fence{.sc,.acq_rel,.acquire,.release}.scope{.cta,.cluster,.gpu,.sys} + **sm_70+ membar≡fence.sc、membar.cta/gl/sys≡fence.sc.cta/gpu/sys 逐字** + fence.proxy.async（sm_90+，async proxy 与 generic proxy 排序）；atom{.sem}{.scope}{.space}.op：.space={.global,.shared::cta/::cluster}、.sem={.relaxed,.acquire,.release,.acq_rel}、.scope={.cta,.cluster,.gpu,.sys}；cp.async.ca/cg.shared{::cta}.global cp-size{4,8,16}（PTX 7.0 sm_80+，.cg cache L2 only / .ca all levels，non-blocking，weak generic proxy）+ cp.async.commit_group（per-thread batch uncommitted）+ cp.async.wait_group N（≤N most recent groups pending）/wait_all（=commit+wait_group 0）+ **无 wait/mbarrier 同步则 cp.async 间无 ordering 保证逐字**；cp.async.bulk.tensor{.1d-.5d}（PTX 8.0 sm_90+，.tile/.tile::gather4/.im2col load_mode、global→shared::cta/::cluster 完成机制 .mbarrier::complete_tx::bytes、shared→global .bulk_group、.multicast::cluster、cta_group::1/::2）+ **Tensor-map 128-byte opaque object（.const/.param/.global，tensormap proxy，host CUDA API 创建）逐字** + TMA swizzle modes（128B/64B/32B atomicity + 8-byte flip）；mbarrier（shared memory .b64 8B-aligned opaque object，phase + pending arrivals + **tx-count** transaction 计数，init/expect_tx/complete_tx/arrive/test_wait/try_wait，user-defined 数量仅受 shared 容量限 vs bar.sync 固定 barrier 数）；ldmatrix.sync.aligned.m8n8.x1/x2/x4{.trans}{.shared::cta}.b16（warp collective load 8×8 16-bit matrices for mma）+ m16n16 8/6/4-bit；ld.global.v4.f32（.v2/.v4 vector ≤128-bit、ld.v4.b32 access size 16B natural alignment）+ ld.global.nc（non-coherent cache read-only path，.vec={.v2,.v4,.v8}）；Cache Operators（**.ca「Global data is coherent at the L2 level, but multiple L1 caches are not coherent for global data」逐字** + driver invalidate L1 between dependent grids、.cg bypass L1、.cs evict-first、.cv don't cache fetch again、st.wb/.cg/.cs/.wt）；State Spaces Table 6/7（.reg per-thread R/W / .sreg RO / .const per-grid RO / .global context / .local per-thread / .param / .shared per-CTA cluster 可访问 / .tex deprecated）；shfl.sync / vote.sync / lop3-prmt-bfe / 低精度格式节） | NVIDIA | https://docs.nvidia.com/cuda/parallel-thread-execution/ | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.2, 1.3, 2.3, 3.1, 3.2, 4.1, 4.2, 4.4, 5.1, 5.2, 6.1, 6.2, 6.3 |
| cuda-best-practices | CUDA C++ Best Practices Guide（**13.5 版**；Table 5 Throughput of Native Arithmetic Instructions 维持 Table 5 编号、列漂移为 7.5/8.0/8.6-8.9/9.0/10.0/10.3/12.0-12.1；CC 9.0 FP32 add/mul/fma 128 results/cycle/SM = 4 scheduler×32 lane×1 指令 → Hopper 单发射互证、FP64 64 / INT32 add 128 / shfl 32 / SFU 16 / DPX 64；算术延迟 prose "typically 4 cycles"（CC 7.0 口径）/ global "hundreds of clock cycles"；**W2 逐字核实**：Coalesced Access to Global Memory——CC6.0+「the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp」、access unit 32-byte 无论 L1 caching 与否、adjacent float→four 32B transactions、misaligned sequential→five segments、permuted within/across segments 仍 four、subset requested 仍 full segment fetched；Strided Accesses——strideCopy kernel、stride 2→50% efficiency、stride 增大至 32 threads×32 segments 饱和；Shared Memory and Memory Banks——CC5.x+ each bank 32 bits/cycle、successive 32-bit words→successive banks、warp 32 = banks 32、conflict 串行化 factor=separate requests 数、broadcast 同址例外 + multicast、padding w+1=33 banks≡unit stride mod 32（V100 transpose 140.2→199.4 GB/s）；Occupancy——active warps/max warps 定义、「Higher occupancy does not always equate to higher performance... low occupancy always interferes with the ability to hide memory latency」、CC7.0 65536 regs/2048 threads→32 regs/thread@100%、allocation granularity rounded up 256 regs/warp、**「NVIDIA provides an occupancy calculator as part of Nsight Compute」逐字**（spreadsheet 已替换）、cudaOccupancyMaxActiveBlocksPerMultiprocessor 动态 launch config；Register Pressure——-maxrregcount/__launch_bounds__ 控制、低 occupancy 多 registers/thread 减 spill + 高 ILP 可低 occupancy 满 latency cover；async copy（__pipeline_memcpy_async）免中间 register 降 register pressure、16B copy 可 bypass L1；Global Memory Load/Store Efficiency metrics（legacy 名，NCU 现行 Sectors/Req）；tiling matrixMul shared 例） | NVIDIA | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.2, 1.4, 2.2, 2.3, 3.4, 4.1, 4.2, 4.3 |
| nsight-compute | Nsight Compute docs（ProfilingGuide §2.4.7 Warp Stall Reasons：long_scoreboard=L1TEX local/global 依赖 / mio_throttle=shared-MIO 队列 / wait / not_selected / barrier / membar / lg_throttle / math_pipe_throttle / imc_miss / drain / dispatch_stall；Warp Scheduler States issue-slot / memory chart DCC-IDC-IMC 常量缓存 / roofline / occupancy metrics；**W2 核实**：NsightCompute §4.13 Occupancy Calculator「It replaces the previously provided CUDA Occupancy Calculator spreadsheet」逐字（.ncu-occ 文件 + Python interface + Tables/Utilization/Graphs/GPU Data；**3.4 occupancy calculator 读法归本条目**，BP 交叉引用同 URL）；ProfilingGuide Memory Workload Analysis——Requests（each LSU instruction exactly one request）/ Wavefronts（L1TEX one wavefront/cycle 串行）/ **Sectors/Req「For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16」逐字** / Bytes=sectors×32B（minimum access size in L1 is one sector）/ atomics-reductions passed through to L2 / Sector Misses to L2；L1/L2 Sectors achieved-ideal-excessive 派生指标（4.1 coalescing 效率现行指标，替代 legacy gld_efficiency）） | NVIDIA | https://docs.nvidia.com/nsight-compute/NsightCompute/index.html ; https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.2, 1.4, 2.2, 2.3, 2.4, 3.4, 4.1, 11.2 |
| hopper-blog | NVIDIA Hopper Architecture In-Depth blog（132 SM / combined 256 KB shared+L1 / shared 可配 228 KB / RF 256 KB/SM / 5120-bit HBM3 / 50 MB L2 partitioned crossbar / 700W；**早期披露 3 TB/s 且注明 Memory Data Rate Not finalized**，与 whitepaper V1.04 Table 3 final 3352 GB/s 构成版本漂移；开放技术博客，whitepaper 互证源） | NVIDIA | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | [C]（开放技术博客、非闭源） | 2026-11-25 | 1.1 |
| hopper-tuning-guide | NVIDIA Hopper Tuning Guide（CC 9.0 occupancy 64 warps / shared 228 KB（+39% vs A100 164）/ TMA 1D-5D global↔shared + cluster DSMEM / Thread Block Clusters；1.2/2.3/4.4 官方调优口径） | NVIDIA | https://docs.nvidia.com/cuda/hopper-tuning-guide/ | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.2, 2.3, 4.4 |
| blackwell-tuning-guide | NVIDIA Blackwell Tuning Guide（CC 10.0 64 warps / 32 blocks / 64K regs / shared 228 KB / combined L1-tex-shared 256 KB / carveout 0-228；**CUDA reserves 1 KB shared per block → 单 block 227 KB opt-in**、static 48 KB 上限；GB200 L2 126 MB；B200 HBM3/HBM3e up to 180 GB；B200 nonportable cluster 16；1.1/1.2/2.1/2.2/2.3 官方调优口径） | NVIDIA | https://docs.nvidia.com/cuda/blackwell-tuning-guide/ | [C]（官方规格披露·一手权威） | 2026-11-25 | 1.1, 1.2, 2.1, 2.2, 2.3 |
| hopper-microbench | Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（H800 微基准 Table IV 延迟 cycles：L1 40.7 / shared 29.0 / L2 263.0 / global 478.8；wgmma m64n128k16 dense 128 cycles / sparse SS 144；SM-to-SM DSMEM 180 cycles；1.4/2.2 延迟表实测支撑，官方文档只给定性 hundreds-of-cycles；无官方 repo 判 [B]；扩展版 arXiv:2501.12084 同作者） | Luo et al. | arXiv:2402.13499 | [B]（有 paper 无官方代码） | 2026-11-25 | 1.3, 1.4, 2.2, 2.3, 2.4 |
| cutlass | NVIDIA/cutlass repo（**版本锚 4.9.0（2026-11-13，CHANGELOG main）**；CUTLASS GEMM 层级 device→kernel→threadblock→warp→instruction / CuTe layout-tile / MMA atom include/cute/arch/mma*.hpp；**W2 路径级核实（GitHub contents API main）**：swizzle include/cute/swizzle.hpp（Swizzle<B,M,S> template，「AA = ZZ xor YY」+ offset ^ shiftr(offset & yyy_msk, msk_sft) 逐字）+ swizzle_layout.hpp + pointer_swizzle.hpp；pipeline include/cutlass/pipeline/{pipeline,sm90_pipeline,sm100_pipeline,sm107_pipeline}.hpp——sm90_pipeline.hpp：PipelineState<Stages>（index/phase）+ PipelineTmaAsync<Stages, ClusterTransactionBarrier>（full_barrier_/empty_barrier_[Stages] + producer_acquire/producer_commit/producer_expect_transaction/consumer_wait/consumer_release）+ PipelineTmaStore + PipelineTransactionAsync + PipelineAsync；**漂移**：PipelineTmaUmmaAsync（旧 3.x FA3 时代 warp-specialized 类）不在当前 main sm90_pipeline.hpp；media/docs gemm_tutorial.md 已不存在→efficient_gemm.md（docs 拆 cutlass-docs 条目，4.3 被引移彼）；3.x/4.x 架构全量走读⟨verify W3 5.2〉） | NVIDIA | https://github.com/NVIDIA/cutlass | 教学参照（官方实现） | 2026-11-25 | 1.3, 4.2, 4.4, 5.2 |

### Part 2 · 内存层级与带宽账

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| jedec-hbm3 | JEDEC HBM3 规范（JESD238A；栈结构 1024-bit/stack = 16 ch × 64-bit/ch + 32 pseudo-channel；**JEDEC 原文 403 bot 拦截⟨verify 精确 URL 残留，同 PCI-SIG W4〉**，几何经成员官方互证：Micron "HBM3E offers 16 independent channels and 32 pseudo channels" + 1024 IO pins、Samsung/SK hynix HBM3 6.4 Gbps/pin → 819 GB/s/stack、24 GB 8-high 供 H200；HBM3e = vendor speed-bin 9.2/9.6 Gbps 非独立 JEDEC 编号） | JEDEC | https://www.jedec.org/standards-documents/docs/jesd238a ; https://www.micron.com/products/memory/hbm/hbm3e ; https://semiconductor.samsung.com/dram/hbm/ ; https://product.skhynix.com/products/dram/hbm.go | [C]（官方规格披露·一手权威） | 2026-11-25 | 2.1 |
| nvbandwidth | NVIDIA/nvbandwidth repo（HBM / P2P / NVLink 带宽实测工具；README 2026-11-25 经 GitHub API 一手核实：CE/SM 两类拷贝逐字、测试名族、spin kernel + CUDA events、3 采样中位数、SM 拷贝 512 threads/block × SM 数截断公式、pointer chasing 延迟口径、双向报值 per-direction 非总和；2.4 / 7.2 / 11.2 测量方法） | NVIDIA | https://github.com/NVIDIA/nvbandwidth | 教学参照（官方实现） | 2026-11-25 | 2.4, 7.2, 11.2 |
| blackwell-microbench | Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks（RTX 5080 GB203 + H100 PCIe 微基准；GH100 L2 两独立分区 vs GB203 单片 L2 对照；2.2 L2 分区实测互证；无官方 repo 判 [B]） | Jarmusch et al. | arXiv:2507.10789 | [B]（有 paper 无官方代码） | 2026-11-25 | 2.2 |
| b200-power-bench | Matching Workloads to Blackwell: Power-Resolved Benchmarking of NVIDIA B200 and B300 across HPC and AI Workloads（B200/B300 nvidia-smi HBM3e 179/269 GiB ≈ raw 192/288 GB decimal vs datasheet/brief usable 180/270 GB（270 出自 brief 对照表，datasheet 无 B300 行）、TDP 1000W 实测；2.1 SKU raw-vs-usable 口径实测注；无官方 repo 判 [B]） | Zand et al. | arXiv:2610.41919 | [B]（有 paper 无官方代码） | 2026-11-25 | 2.1 |

### Part 3 · CUDA 编程模型

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| cuda-samples | NVIDIA/cuda-samples repo（**重组漂移：Samples/N_Name → cpp/N_Name**，README 支持 CUDA Toolkit 13.5；**W2 GitHub API master 树路径级核实**：cpp/0_Introduction/{matrixMul, matrixMulDrv, simpleCooperativeGroups, asyncAPI} / cpp/2_Concepts_and_Techniques/{reduction, reductionMultiBlockCG, scan, shfl_scan, threadFenceReduction} / cpp/3_CUDA_Features/{globalToShmemAsyncCopy（matmul + async copy CC8.0+ + arrive-wait barrier）, graphConditionalNodes, graphMemoryNodes, graphMemoryFootprint, simpleCudaGraphs, jacobiCudaGraphs} / cpp/4_CUDA_Libraries/{conjugateGradientCudaGraphs, conjugateGradientMultiBlockCG, conjugateGradientMultiDeviceCG} / cpp/6_Performance/{alignedTypes, cudaGraphsPerfScaling（cudaStreamBeginCapture/cudaGraphInstantiate/cudaGraphLaunch/cudaGraphUpload）}；顶层 cpp/0_Introduction…cpp/10_Safety + python/；3.1 / 3.3 / 4.1 / 4.3 / 5.1 / 5.3 / 7.3 教学例） | NVIDIA | https://github.com/NVIDIA/cuda-samples | 教学参照（官方实现） | 2026-11-25 | 3.1, 3.3, 4.1, 4.3, 5.1, 5.3, 7.3 |
| fermi-whitepaper | NVIDIA Fermi Compute Architecture Whitepaper（GF100；**GigaThread 逐字**：「The GigaThread global scheduler distributes thread blocks to SM thread schedulers」+ two-level distributed thread scheduler——chip level global work distribution engine schedules thread blocks to various SMs / SM level each warp scheduler distributes warps of 32 threads to execution units；first-gen GigaThread（G80）managed up to 12288 threads；**legacy term 锚**：GigaThread 术语 PG 8.0/9.0/11.8/13.0/13.5 均无（本轮 archive 逐版 grep），现行 PG §3.2.2 无名描述「blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity」——3.1 block 调度到 SM 机制历史出处，writer 须注术语时代；www.nvidia.com PDF 本轮可达 876380 B 一手） | NVIDIA | https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf | [C]（官方规格披露·一手权威） | 2026-11-25 | 3.1 |
| cuda-graphs-blog | Getting Started with CUDA Graphs（NVIDIA Technical Blog 2019-09-05；**V100 / CUDA 10.1 launch 开销实测逐字**：kernel execution 2.9 μs / sync launch（cudaStreamSynchronize per kernel）9.6 μs per kernel / async stream launch 3.8 μs / **graph launch 3.4 μs per kernel** / create+instantiate ~400 μs 一次性（20000 launches 摊销 ~0.02 μs/kernel）/ first graph launch ~33% slower than subsequent；cudaStreamBeginCapture/EndCapture capture + cudaGraphInstantiate「creates and pre-initialises all the kernel work descriptors」+ cudaGraphLaunch replay；**时代注**：V100/CUDA 10.1 实测，现行硬件绝对值会漂移，~μs 量级为官方口径（PG 只定性「CPU launch costs are reduced」）；3.3 graph launch 开销摊销账官方数值出处；开放技术博客） | NVIDIA | https://developer.nvidia.com/blog/cuda-graphs/ | [C]（开放技术博客、非闭源） | 2026-11-25 | 3.3 |
| libcudacxx-memory-model | CCCL libcu++ Extended API — Memory model（CUDA C++ extends ISO C++ memory model with thread scopes；**enum thread_scope {thread_scope_system, thread_scope_device, thread_scope_cluster, thread_scope_block, thread_scope_thread} 逐字**；device scope = same device + same memory synchronization domain；pre-SM90 cluster operations widened to device；cuda::atomic/atomic_ref + memory_order_release/acquire flag message-passing 例 + cuda::atomic_thread_fence；PG 13.5 §5.4.4.3「It is suggested to use cuda::atomic_thread_fence provided by libcu++ whenever possible」交叉引用；**URL 漂移**：旧 nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html 404 → 现行 /cccl/unstable/libcudacxx/extended_api/memory_model.html 200（cccl/ 根 redirect unstable/）；3.2 memory_order / scoped synchronization 官方口径） | NVIDIA | https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html | [C]（官方规格披露·一手权威） | 2026-11-25 | 3.2 |

### Part 4 · Kernel 访存优化

本 Part 锚点多为跨 Part 共享，收录于首引 Part：Best Practices / PTX ISA / Nsight Compute / CUTLASS 代码 → P1，cuda-samples → P3；被引章节在对应条目跨组累加 4.1-4.4。**W2 新增 cutlass-docs**（CUTLASS media/docs 文档自 cutlass repo 条目拆分，首引 4.3 GEMM tutorial；沿 inference deepinfer-doc/deepinfer-repo 拆分先例）。

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| cutlass-docs | CUTLASS media/docs/cpp 文档（**自 cutlass repo 条目拆分**；W2 GitHub contents API main 路径级核实：efficient_gemm.md（GEMM 层级 device→threadblock→warp→instruction 嵌套循环伪代码逐字：CtaTileM/N threadblock-level concurrency → WarpTileM/N/K warp-level parallelism → MmaM/N/K instruction-level parallelism）+ pipeline.md（Synchronization primitives：Hopper clusters + async barriers + CUTLASS abstractions）+ gemm_api_3x.md + cutlass_3x_design.md + cute/ 子目录；**漂移**：旧 gemm_tutorial.md 已不存在 → efficient_gemm.md（4.3 tiling 教学参照现行锚）；repo 代码路径（swizzle/pipeline/mma atom）见 cutlass 条目） | NVIDIA | https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp | 教学参照（官方实现） | 2026-11-25 | 4.3 |

### Part 5 · Kernel 计算优化与并行原语

| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
| blelloch-scan | Prefix Sums and Their Applications（Blelloch scan up-sweep / down-sweep，work O(n_el) / depth O(log n_el)；CMU-CS-90-190 November 1990，无 arXiv 无官方 repo；**URL 修**：报告页 cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html（title + BibTeX number=CMU-CS-90-190 逐字）+ 页链报告 PDF Ble93.pdf（311166 B 首页标题逐字）；旧 Ble90.pdf 非本报告（Vector Models 书锚/本机截断）移除） | Blelloch | https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html ; https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf（CMU-CS-90-190，无 arXiv） | [B]（老经典无 arXiv） | 2026-11-25 | 5.1 |
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
| nvlink-nvswitch | NVIDIA NVLink / NVSwitch 产品页（NVLink 世代 / link 带宽 / NVSwitch crossbar / NVLink 域；H100 18 links × 25 GB/s 单向 = 450 单向 = 900 双向；B200 NVLink5 1.8 TB/s 双向⟨verify W4 精确代际值〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/nvlink/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 2.4, 7.1 |
| gb200-nvl72 | NVIDIA GB200 NVL72 规格（72 GPU NVLink 域超节点 / NVLink-C2C Grace-Blackwell；⟨verify W4 / W6 精确规格〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.1, 10.1, 11.1 |
| hgx-dgx | NVIDIA HGX / DGX 服务器规格（HGX H100 8-GPU baseboard / DGX H100 整机 / 供电散热；⟨verify W6 精确数值〉） | NVIDIA | https://www.nvidia.com/en-us/data-center/hgx/ | [C]（官方规格披露·一手权威） | 2026-11-24 | 7.2, 11.1 |
| pcie-spec | PCI-SIG PCIe 规范（Gen5 x16 = 64 GB/s 单向；URL 403 bot 拦截⟨verify W4 精确出处〉，规范真实存在） | PCI-SIG | https://pcisig.com/specifications | [C]（官方规格披露·一手权威） | 2026-11-24 | 2.4, 7.2 |
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
