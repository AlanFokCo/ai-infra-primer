# gpu W3 research pass — P5（kernel 计算优化）+ P6（低精度与注意力 kernel）一手来源核实

Task 1：P5（5.1-5.4）+ P6（6.1-6.4）论文核实 + **fresh-tarball 代码锚清单**（writer T2-T5 走读用）+ `gpu/references.md` P5/P6 扩充。本波为 **CODE-WALKTHROUGH DENSE**（5.2 GEMM / 6.3 FA 模块级走读，6.2 Marlin / 6.4 FlashMLA 片段级走读），故代码锚以 **fresh tarball file:line** 为准，版本锚 + 核实日期 2026-11-25。writer 引用白名单 = `gpu/references.md` P5/P6 条目 + 跨 Part 共享锚点（P1 cutlass / ptx-isa / nvidia-datasheet，P3 cuda-samples，P4 cutlass-docs）。

## 0. 摘要

- **论文**：7 篇 arXiv 经 export API 批量 200 复核（ID↔标题↔第一作者，totalResults=7）；Blelloch 1990（CMU-CS-90-190，无 arXiv）沿 W1/W2 已锚。**FA4 ⟨verify⟩ 销项**：FA4 存在（CuTeDSL 实现 + 正式论文 arXiv:2603.05451）。
- **代码锚**：6 repo fresh tarball（codeload.github.com）+ FlashInfer contents API（372 MB repo tarball 截断规避）。CUTLASS 4.9.0 / flash-attention b038b497 / Marlin 1f25790b / DeepGEMM 5081f851 / FlashMLA main 395cb3af + ba89a346 双锚 / FlashInfer 0.7.6 5228ea6b。TransformerEngine norm 路径漂移销项（5.3）。
- **新增 1 条**：`flashattention4`（P6 [A]，2603.05451 + flash_attn/cute/ CuTeDSL，6.3 首引）。
- **更新条目**：cutlass（P1，5.2 GEMM 层级全量代码锚 + PipelineTmaUmmaAsync 漂移订正）/ cutlass-docs（P4，被引 += 5.2）/ deepgemm / transformerengine / nanoflow / tilelang（P5）/ marlin / flashattention / flashattention2 / flashattention3 / flashmla / flashinfer（P6）。核实日期 2026-11-25。
- **⟨verify⟩ 销项**：FA4 存在性（6.3）、CUTLASS 3.x/4.x 架构全量走读锚（5.2，W2 残留）、TransformerEngine norm 路径（5.3）。**残留**：B200 SM 数官方未披露（W1 残留，W3/W6，本轮未销）。

## 1. 核实通道与网络 fallback（本轮实际使用）

- **arXiv export API 稳定 200**：`https://export.arxiv.org/api/query?id_list=...` 批量 7 ID 一次 200（19004 B，totalResults=7）；FA4 候选经 `search_query=ti:"FlashAttention-4"` 取得（totalResults=2，官方 2603.05451 vs 第三方 2609.04105）。未触 429，未用 abs/HTML fallback。
- **codeload.github.com tarball 主通道**：`https://codeload.github.com/<org>/<repo>/tar.gz/refs/heads/<branch>` 5 repo 全 200 一手取得并解压（marlin 732984 B / FlashMLA main 302549 B / DeepGEMM 336072 B / cutlass ~81 MB git→tarball / flash-attention 14662697 B）；FlashMLA ba89a34 commit tarball `tar.gz/ba89a34` 200（295426 B）。**每 tarball 独立空目录** `/tmp/gpu-w3-<org>_<repo>/`（untrusted 隔离）。
- **FlashInfer 372 MB → contents API**：repo size 372493 KB，tarball 必截断 → 改用 GitHub contents API 目录级（root / include/flashinfer / attention / flashinfer / jit）+ 文件级 base64 raw（cascade.cuh 35003 B / scheduler.cuh 90991 B / cascade.py 44558 B / decode.py 263202 B / jit/core.py 34808 B）。**raw.githubusercontent 沿 W2 记录 blocked → contents API base64 解码**（fetch.py helper，`python3 -I` 隔离运行）。
- **GitHub API repo/commits 200**：6 repo metadata（default_branch / size / pushed_at）+ HEAD commit SHA（版本锚）；TransformerEngine contents API 路径级（normalization/layernorm|rmsnorm）。
- **submodule 注意**：DeepGEMM tarball 含 `.gitmodules`（third-party/cutlass + third-party/deep_jit）但 **submodule 内容空**（codeload 不含 submodule）；DeepGEMM 自身 device kernel 在 `deep_gemm/include/deep_gemm/`（非 submodule），锚点取自此。
- 工具：`/usr/bin/grep -nE`（file:line）；`python3 -I`（JSON/base64 解析，隔离）；tar 解压。下载缓存 `/tmp/gpu-w3-research/`（arxiv-batch.xml / fa4-search.xml / meta-*.json / fi-*.json / fetch.py）+ `/tmp/gpu-w3-flashinfer-files/`（contents API raw）。

## 2. 论文核实（arXiv export API，ID↔标题↔第一作者，2026-11-25）

| arXiv ID | 版本 | 标题 | 第一作者 | 条目 key | 等级 |
| --- | --- | --- | --- | --- | --- |
| 2205.14135 | v2 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao | flashattention | [A] |
| 2307.08691 | v1 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Tri Dao | flashattention2 | [A] |
| 2407.08608 | v2 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Jay Shah | flashattention3 | [A] |
| 2603.05451 | v1 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling | Ted Zadouri | flashattention4（新增） | [A] |
| 2408.11743 | v1 | MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models | Elias Frantar | marlin | [A] |
| 2501.01005 | v2 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | Zihao Ye | flashinfer | [A] |
| 2408.12757 | v2 | NanoFlow: Towards Optimal Large Language Model Serving Throughput | Kan Zhu | nanoflow | [A] |
| 2504.17577 | v2 | TileLang: A Composable Tiled Programming Model for AI Systems | Lei Wang | tilelang | [A] |
| CMU-CS-90-190 | — | Prefix Sums and Their Applications（Blelloch scan，无 arXiv） | Blelloch | blelloch-scan | [B] |

- **FA4 论文甄别**：arXiv ti:"FlashAttention-4" 返回 2 条——**2603.05451**（Zadouri, Hoehnerbach, Shah, Liu, Thakkar, **Dao**；2026-03-05）为官方 FA4（FA3 原班 + NVIDIA CuTeDSL 团队）；2609.04105「Hardware-Aware FP4 FlashAttention-4」（Robert Hu，2026-09-03）为第三方同名，**非官方锚**。2603.05451 abstract 逐字：「FlashAttention-3 optimized attention for Hopper GPUs... primarily targets the H100... Blackwell-based systems such as the B200 and GB200... asymmetric hardware scaling: tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged... (1) redesigned pipelines that exploit fully asynchronous MMA operations and larger tile sizes, (2) software-emulated exponential and conditional softmax res[caling]」→ FA4 Blackwell tcgen05 全异步 MMA + 大 tile + 软件 exp（SFU 相对慢）机制锚。
- Blelloch 1990 沿 W1/W2：报告页 cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html + Ble93.pdf（首页标题逐字），无 arXiv 无官方 repo，[B] 老经典。

## 3. 代码锚清单（repo / 版本 / file:line / 机制）

> 版本锚 = HEAD commit SHA（GitHub commits API）+ 核实日期 2026-11-25。file:line 基于 fresh tarball（codeload main/master）或 contents API raw（FlashInfer）。**writer 走读引用一律带版本锚**；跨波 main 漂移须注。

### 3.1 NVIDIA/cutlass — 4.9.0（commit 5c179edeb3e2，2026-11-25；CHANGELOG 4.9.0 2026-11-13）· 5.2 模块级走读

版本：`include/cutlass/version.h` L41-43 `CUTLASS_MAJOR 4 / MINOR 9 / PATCH 0`；`CHANGELOG.md` L5「## [4.9.0] (2026-11-13)」。W2 已锚 swizzle/pipeline，本轮补 **5.2 GEMM 层级全量**。

| file:line | 机制 |
| --- | --- |
| `media/docs/cpp/efficient_gemm.md` L16-32 | GEMM 层级嵌套循环伪代码：cta_n/cta_m（threadblock-level concurrency）→ cta_k（"GEMM mainloop"）→ warp_n/warp_m（warp-level parallelism）→ warp_k → mma_k/mma_n/mma_m（instruction-level parallelism）→ mma_instruction。**device→kernel→threadblock→warp→instruction 概念层级一手** |
| `include/cutlass/gemm/device/gemm_universal.h` L142 / L289 / L398 | device 层：class GemmUniversal（2.x L142 / 3.x 特化 L289）+ initialize() L398（host 入口，GemmKernel = UnderlyingOperator::GemmKernel L355） |
| `include/cutlass/gemm/kernel/gemm_universal.hpp` L42-57 | kernel 层 dispatch：IsCutlass3ArrayKernel（2.x vs 3.x API 分流）+ include tile_scheduler.hpp / sm70_gemm.hpp |
| `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` L65 / L269 / L317-327 / L392 / L436 | **3.x kernel 层**：class GemmUniversal<CollectiveMainloop, CollectiveEpilogue, TileScheduler> L65；operator()(Params, smem) L269；MainloopPipeline Producer/Consumer role + transaction_bytes L317-327；CollectiveEpilogue L392；collective_mainloop.load(...) L436 + load_tail L450（producer warpgroup） |
| `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_{cooperative,pingpong}.hpp` | kernel schedule 变体（cooperative / pingpong，FA3 风格调度命名在 CUTLASS 侧） |
| `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` L72 / L92 / L111 / L130-137 / L312 / L417 / L496 | **collective mainloop**：struct CollectiveMma L72；DispatchPolicy=MainloopSm90TmaGmmaWarpSpecialized<Stages,ClusterShape,KernelSchedule> L92；**MainloopPipeline=cutlass::PipelineTmaAsync<Stages> L111**（非 TmaUmmaAsync）；SmemLayoutA/B=tile_to_shape(TileShape, Stages) L130-134 + static_assert Stages>=2 L137；load()（TMA producer）L312 / load_tail L396；mma()（consumer）L417：warpgroup_arrive L490 → **cute::gemm(tiled_mma, tCrA, tCrB, accum) L496** → warpgroup_commit_batch L500 |
| `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp` | SM90 FP8 blockwise scaling mainloop（DeepGEMM 对照，5.2/6.2） |
| `include/cutlass/gemm/collective/fp8_accumulation.hpp` L40-43 | FP8 promotion API 逐字：「promote (add) or scale (multiply_add) the results from the tensor core accumulators to the main accumulators when the number [of MMAs exceeds threshold]... the tensor core accumulators are zeroed」→ FP8 两级累加（DeepGEMM 同概念） |
| `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` L83 / L68 / L122 / L128-130 | **epilogue**：class CollectiveEpilogue L83；EpilogueTile (EPI_TILE_M,EPI_TILE_N) L68；ThreadEpilogueOp=FusionCallbacksTraits::Operation L122（fusion：bias/activation）；static_assert EpilogueTile rank-2 整除 CTA tile L128-130 |
| `include/cute/layout.hpp` L99 / L48 / L51 | **CuTe layout**：struct Layout L99；Shape=cute::tuple L48 / Stride=cute::tuple L51（layout = shape:stride 代数） |
| `include/cute/atom/mma_atom.hpp` L49 / L211 / L460 / L531 | **MMA atom 层级**：MMA_Atom<MMA_Traits<Op>> L49 → TiledMMA:MMA_Atom L211 → ThrMMA:TiledMMA L460；make_tiled_mma L531（atom→tiled→thread 三级） |
| `include/cute/arch/mma_sm90_gmma.hpp` L158 / L249 / L341 / L438 | **wgmma 指令**：wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 L158 / m64n16k16 L249 / m64n32k16 L341 / m64n64k16 L438（GMMA m64nNk16 族，1.3 recap 代码锚） |
| `include/cute/swizzle.hpp` L52 / L55 / L78 / L89 | swizzle（W2 已锚，fresh 再核）：「AA = ZZ xor YY」L52；struct Swizzle L55；`return offset ^ shiftr(offset & yyy_msk{}, msk_sft{}); // ZZZ ^= YYY` L78；template<int B,int M,int S> L89 |
| `include/cutlass/pipeline/sm90_pipeline.hpp` L171 / L289 / L299-300 / L443 / L453 / L480 | pipeline（W2 已锚，fresh 再核）：PipelineState L171；PipelineTmaAsync L289；full_barrier_[Stages]/empty_barrier_[Stages] L299-300；producer_acquire L443 / producer_commit（bytes / UserDefinedArriveOp 两重载）L453/459 / producer_expect_transaction L480 |
| `include/cutlass/pipeline/sm100_pipeline.hpp` L538 | **PipelineTmaUmmaAsync L538**（Blackwell UMMA pipeline）——**漂移订正见 §5** |
| 2.x 层级目录 `include/cutlass/gemm/{threadblock,warp,thread}/` | threadblock/default_mma_core_sm80.h、warp/default_mma_tensor_op.h、thread/mma.h（2.x SIMT/tensor-op 层级，概念对照 efficient_gemm.md；3.x/4.x 走读以 collective 为主） |

### 3.2 Dao-AILab/flash-attention — commit b038b4974f26（2026-11-23）· 6.3 模块级走读（FA2/FA3/FA4）

repo 含 FA2（`csrc/flash_attn/src/`，Ampere/sm80）、FA3（`hopper/`，Hopper/sm90）、**FA4（`flash_attn/cute/`，CuTeDSL，Hopper+Blackwell）**。

**FA2（csrc/flash_attn/src/）**

| file:line | 机制 |
| --- | --- |
| `flash_fwd_kernel.h` L57 | `compute_attn_1rowblock(params, bidb, bidh, m_block)` — **Q outer**（每 m_block 一个 Q tile，blockIdx 映射） |
| `flash_fwd_kernel.h` L92-93 / L276 / L387 | KV inner：n_block_min/n_block_max（causal/local 范围）L92-93；n_block=n_block_max-1 L276；**`for (; n_block >= n_block_min; --n_block)` L387**（KV 逆序内循环） |
| `flash_fwd_kernel.h` L352-353 | `softmax.softmax_rescale_o<Is_first>(acc_s, acc_o, scale_softmax_log2)` — online softmax rescale 调用 |
| `flash_fwd_kernel.h` L374 / L376 | tOrP=convert_layout_acc_Aregs（P→第二 GEMM A operand 布局）L374；**gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, ...) L376**（PV GEMM，register-source A） |
| `softmax.h` L67 / L79 / L86 | scale_apply_exp2 L67；「Instead of computing exp(x - max), we compute exp2(x * log_2(e) - ...」L79；`exp2f(__fmul_rn(tensor, scale) - max_scaled)` L86（**exp2 trick**，SFU exp2 快于 exp） |
| `softmax.h` L137 / L160 / L170 | **softmax_rescale_o L137**：acc_o_rowcol=convert_layout_acc_rowcol L150；**`acc_o_rowcol(mi,ni) *= scores_scale` L160**（O rescale by exp(old_max-new_max)）；normalize_softmax_lse L170（终除 row_sum） |
| `softmax.h` L24 / L39 / L48 / L54 / L60 | thread_reduce_ L24 → **quad_allreduce_ L39**（warp 内 4-thread quad 共享行归约）→ reduce_ L48；reduce_max L54 / reduce_sum L60（5.1 warp reduction 对照） |
| `kernel_traits.h` L15 / L63-66 | Flash_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps> L15；kNThreads=kNWarps*32 L64 / kBlockM L66（tile 几何） |

**FA3（hopper/）**

| file:line | 机制 |
| --- | --- |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L44 / L55 / L83 | **Is_FP8**=float_e4m3_t‖float_e5m2_t L44；Transpose_V=Is_FP8&&!V_colmajor L55；static_assert「MmaPV must be RS if FP8」L83 |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L121 / L135-166 | NumProducerThreads=NumThreadsPerWarp（TMA 单 warp producer）或 NumThreadsPerWarpGroup L121；SmemLayoutQ/K/V=tile_to_shape(...,kStages) L135-166（multi-stage） |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L208-209 | GmemTiledCopyQ=SM90_TMA_LOAD L208；GmemTiledCopyKV=sm90_cluster_shape_to_tma_atom（TMA multicast）L209 |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L283-289 | **PipelineTmaAsync=PipelineTmaAsyncNoCluster（cluster=1）/ PipelineTmaAsync（cluster>1）L283**；MainloopPipelineK/V/Vt L284-286；PipelineState<kStages> L289（**FA3 用 cutlass::PipelineTmaAsync 系，非 TmaUmmaAsync**） |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L353 / L356 | **IntraWGOverlap**=(NumMmaWarpGroups>=2)&&(...) L353（intra-warpgroup GEMM-softmax pipelining）；RescaleOBeforeGemm L356 |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L750 / L753 / L758 | load()：pipeline_k.producer_acquire L750；**tma_load_K.with(barrier, mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST) L753**；producer_commit(smem_pipe_write, cpasync_barrier_arrive) L758 |
| `mainloop_fwd_sm90_tma_gmma_ws.hpp` L386 / L442 | ptr_q_descale / ptr_k_descale / ptr_v_descale（FP8 descale 指针） |
| `flash_fwd_kernel_sm90.h` L75 / L78 / L82-83 | NumMmaWarpGroups=size(TiledMmaPV)/NumThreadsPerWarpGroup L75（==1‖2‖3 L78）；**LoadRegisterRequirement / MmaRegisterRequirement L82-83**（producer/consumer warpgroup 寄存器再分配，warp specialization 核心） |
| `flash_fwd_kernel_sm90.h` L179 / L208 / L220-222 | operator() L179；warp_group_idx=canonical_warp_group_idx() L208；**pipeline_params_k.role = warp_group_idx==0 ? Producer : Consumer L220-222**（WG0=TMA producer，WG1/2=MMA consumer） |
| `named_barrier.hpp` L50-58 | **FwdNamedBarriers**：QueryEmpty=0 / **WarpSchedulerWG1=1 / WG2=2 / WG3=3** / AppendKV=4 / QueryRotated=5 / **PFull=6 / PEmpty=7**（warpgroup 间 pingpong 调度 + P 矩阵 pipeline 同步 = FA3 paper pingpong 的代码实现） |
| `flash_fwd_launch_template.h` L63-76 | TileScheduler：VarlenDynamicPersistentTileScheduler / DynamicPersistentTileScheduler（Arch>=90 WarpSpecialized）/ SingleTileScheduler；UsePersistentScheduler L75 |
| `flash_fwd_combine_kernel.h` | split-KV combine kernel（6.3 流量账 split-KV reduction / 6.4 reduction 走读） |

**FA4（flash_attn/cute/，CuTeDSL）⟨verify⟩ 销项**

| file:line | 机制 |
| --- | --- |
| `README.md` L80-99 | 「## FlashAttention-4 (CuTeDSL)... written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200)」L82；`pip install flash-attn-4` L86；`from flash_attn.cute import flash_attn_func` L96；cu13 extra（CUDA 13）L91 |
| `flash_attn/cute/blackwell_helpers.py` L8 / L14-25 | **`from cutlass.cute.nvgpu import tcgen05` L8**；_tcgen05_mma_kind L14：MmaF16BF16Op / MmaTF32Op / MmaI8Op / MmaFP8Op‖MmaF8F6F4Op / MmaMXF8Op（**Blackwell tcgen05 MMA 逐格式**） |
| `flash_attn/cute/flash_fwd_sm90.py` / `flash_fwd_sm100.py` / `flash_fwd_sm120.py` | FA4 forward CuTeDSL（Hopper / Blackwell B200 / sm120 RTX Blackwell） |
| `flash_attn/cute/flash_fwd_mla_sm100.py` / `flash_fwd_decode_sm100.py` | FA4 MLA / decode Blackwell（6.4 对照） |
| 论文 arXiv:2603.05451 | asymmetric hardware scaling（TC 2× vs shared/exp 慢）+ fully-async MMA + 大 tile + 软件 exp（§2） |

### 3.3 IST-DASLab/marlin — commit 1f25790bdd49（master，2024-09-04）· 6.2 片段级走读

单文件 kernel `marlin/marlin_cuda_kernel.cu`（822 行）。**INT4 unpack 用 lop3（非 bfe）**，第二 nibble 用 `>>8` 移位。

| file:line | 机制 |
| --- | --- |
| `marlin/marlin_cuda_kernel.cu` L98 | **mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32**（FP16 MMA，dequant 后） |
| 同上 L111 | ldmatrix.sync.aligned.m8n8.x4.shared.b16（A operand shared→register，tensor core 布局） |
| 同上 L116-126 | **lop3.b32 L122**；注释「Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to automatically recognize it」L116-117；template<int lut> L118 |
| 同上 L131-152 | **dequant(int q)→FragB（4 fp16）**：LO=0x000f000f / HI=0x00f000f0 / EX=0x64006400 L132-134；`lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX)` L136-137；「fuse the -8 symmetric zero point directly into SUB and ADD」L138；SUB=0x64086408 / MUL=0x2c002c00 / ADD=0xd480d480 L139-141；__hsub2 / __hfma2 L143-150（**INT4→FP16 magic-number：EX=0x6400=1024.0，nibble+1024 再减 1032=x-8**） |
| 同上 L155-159 | scale(FragB, FragS)：grouped quantization __hmul2（per-group scale） |
| 同上 L443 / L447-458 | **matmul lambda：「We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations」L443**；b_quant_shift=b_quant>>8 L447；dequant L448/452 + scale L451/454 + mma L457-458（dequant↔MMA 双管线饱和） |
| 同上 L463-465 | 「slice across the k dimension... multiple warps accumulate partial sums of the same output location; which we have to reduce over... in shared memory」（split-K shared reduction） |
| 同上 L61 / L76 / L83 / L89 | cp.async.cg.shared.global（+L2::cache_hint）L61/76；commit_group L83；wait_group L89（A/B tile 异步拷贝） |
| `README.md` L7-8 / L14 / L22 / L43 | 「close to ideal (4x) speedups up to batchsizes of 16-32 tokens」L7-8；「less than 25-50 (tensor core) multiply-accumulates per 4-bit quantized weight... near ideal 4x speedup over FP16」L14；「carefully order dequantization and tensor core instructions... both GPU pipelines well saturated」L22；「maximum possible 3.87x speedup」L43 |

### 3.4 deepseek-ai/DeepGEMM — commit 5081f8513af5（main，2026-10-20）· 5.2 对照 / 6.2 片段级

device kernel 在 `deep_gemm/include/deep_gemm/`（非 submodule）；host launcher 在 `csrc/jit_kernels/impls/`；JIT 经 third-party/deep_jit（submodule，tarball 空）。

| file:line | 机制 |
| --- | --- |
| `README.md` L3 / L67-68 | 「All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation」L3；**「SM90 requires scaling factors in FP32 format」L67 / 「SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single torch.int」L68** |
| `README.md` L78 / L86 | grouped GEMM contiguous layout（M-axis grouping，expert 同 shape）L78；masked layout（decode，DeepEP low-latency 输出）L86 |
| `docs/scaling-factor-format.md` L29 / L39 / L89-92 | gran_k：32‖128 SM100 / 128 only SM90 L29；packed_sf_dtype=int32（4 UE8M0 packed）L39；transform 表 SM90 float32（no packing）vs SM100 broadcast→pack UE8M0 int32 MN-major L89-92 |
| `csrc/runtime/jit.hpp` L12-32 | DeepJIT：`deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit` L12；init_jit（library_root/include，"cutlass-"+CUTLASS_VERSION）L14-22；nvcc_flags（--diag-suppress / -Wno-deprecated）L26-29 |
| `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` L251 / L254 | **「Accumulation for WGMMA or CUDA promotion」L251**；`float accum[WGMMA::kNumAccum], final_accum[...]` L254（**两级累加：WGMMA FP32 accum + CUDA-core promotion final_accum**） |
| 同上 L285-302 / L313 | read B scales scale_b_0/1（ld_shared smem_sfb）L286-289；read A scales scale_a_0/1（smem_sfa）L301-302；WGMMA::wgmma(a_desc, b_desc, accum, k) L313 |
| 同上 L329-345 | **「Promote with scales」L329**：`scale_0_0 = scale_a_0 * scale_b_0` L331；`shifted_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0]` L341-344（**fine-grained scaling：每 BLOCK_K=128 把 FP8 WGMMA 部分积乘 scale_a×scale_b 提升到 FP32 final_accum**，1x128 A / 128x128 B） |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` L51-52 / L57 / L94-95 | MXF4 MMA（packed 4-bit，UMMA_K=64）vs MXF8F6F4 MMA（unpacked，UMMA_K=32）L51-52；「dynamically-scaled epilogue with per-32 UE8M0 SFD output」L57；kGranKA/B==32‖128 L94-95（**Blackwell tcgen05 + UE8M0**） |

### 3.5 deepseek-ai/FlashMLA — 双版本锚 · 6.3 对照 / 6.4 片段级

**main（V4.1，Blackwell）commit 395cb3afbb33（2026-10-27）** + **ba89a34（V3，Hopper）commit ba89a3466e9470ad08ab39738d4e7bb66989e1e7**（README L3 给出全 SHA）。

| 版本 / file:line | 机制 |
| --- | --- |
| main `README.md` L3 | **「Breaking change notice (2026.09.30): removed support for the Hopper architecture and for earlier models (DeepSeek V3 / V3.2 / V4.0), changed FP8 / FP4 KV cache format... switch to [commit ba89a3466e9470ad08ab39738d4e7bb66989e1e7]」**（双锚一手依据） |
| main `README.md` L7 / L25 / L41 | V4.1 token-level sparse prefill/decode（FP8/FP4 KV）+ fused Q-norm/Q-RoPE/attn/O-RoPE/cast kernel L7；2026.09.10 V4.1 release L25；fused kernel「1460 TFlops prefill / 950 TFlops decode on B200 CUDA 13.3」L41 |
| main `README.md` L79 / L140-143 | 「NVIDIA SM100 / SM103 GPU」L79（main 仅 Blackwell）；**V4.1 head_dim=512**，KV 528 B/token（512 float8_e4m3 + 16 float8_e8m0 scale per-32）或 288 B（V4.1 fp4：256 B=512 e2m1 + 32 float8_e4m3 scale per-16）L140-143（**64 RoPE 维也量化，无 bf16 部分**） |
| main `csrc/cuda_kernels/sm100/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp` | V4.1 Blackwell MLA mainloop（CUTLASS 风格 warp-specialized collective；sm100/ 下无 sm90 kernel） |
| ba89a34 `README.md` L81 | **「head_dim_k = 576 (for DeepSeek V3/V3.1/V3.2) or 512 (for DeepSeek V4/V4.1) with head_dim_v = 512」**（MQA mode；两代几何区分一手） |
| ba89a34 `csrc/kernels/params.h` L6-9 | enum：V32（d_qk=576）/ V4（d_qk=512）/ V41（d_qk=512, RoPE fp8, quant tile 32）/ V41_FP4（d_qk=512, fp4 e2m1, quant tile 16, e4m3 scales） |
| ba89a34 `csrc/kernels/sm90/decode/dense/config.h` L7-11 | **BLOCK_SIZE_M=64 / PAGE_BLOCK_SIZE=64 / HEAD_DIM_K=576 / HEAD_DIM_V=512**（absorbed kernel 几何） |
| ba89a34 `csrc/kernels/sm90/decode/dense/traits.h` L24 / L39 / L44 / L82-83 | NUM_THREADS=256（2 warpgroup）L24；GMMA rs/ss_op_selector Shape<BLOCK_M, **HEAD_DIM_V/2**, PAGE>（**O_L/O_R 垂直劈半=256**）L39/44；**TMABarrier barriers_K0[HEAD_DIM_K/64] / barriers_K1[...] L82-83（576/64=9 TMA barrier）** |
| ba89a34 `csrc/kernels/sm90/decode/dense/splitkv_mla.cuh` L20-23 / L36-49 | get_AorC_row_idx（WGMMA fragment 行布局 row_idx=(idx/32)*16+local*8+(idx%32/4)）L20-23；launch_kv_tiles_copy_tma（**cute::TMA::CacheHintSm90::EVICT_FIRST** L47，递归 head_dim tile）L36-49 |
| ba89a34 `csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k576.cu` / `phase1_k512.cu` | sparse prefill 576/512 几何实例化 |
| ba89a34 `docs/20250422-new-kernel-deep-dive.md` L21 / L25 / L42 / L54 | **「Each 64×512 output matrix occupies 32,768 32-bit registers. With only 65,536... per SM, we can store only one output matrix per SM」L21**（排除双输出 pingpong）；O_L/O_R（各 64×256）+ V0L/V0R/V1L/V1R 劈分 L25；**「seesaw scheduling... ping-pong variant using one output matrix... mathematically equivalent to FlashAttention's online softmax... interleaving the two warpgroups」L42**；**「For a 64×576 K block, we launch 9 TMA copies (each moving a 64×64 block). GEMM begins as soon as each TMA copy completes」L54**（fine-grained TMA-GEMM pipelining） |

> **注**：docs/20250422-new-kernel-deep-dive.md 在 main（8371 B）与 ba89a34（8351 B）均存在（main 保留文档但移除 sm90 kernel 代码）；spec 锚定 ba89a34 版本为 Hopper absorbed 机制一手。main `csrc/cuda_kernels/` 仅 sm100/smxx（无 sm90 kernel 目录），证实 Hopper 移除。

### 3.6 flashinfer-ai/flashinfer — 0.7.6（commit 5228ea6be8ec，2026-11-25）· 6.4 片段级（contents API）

| file:line | 机制 |
| --- | --- |
| `include/flashinfer/attention/cascade.cuh` L45 / L90 / L224 | **MergeStateKernel L45**（v_a,s_a + v_b,s_b → v_merged,s_merged）；MergeStateInPlaceKernel L90；MergeStatesKernel（多状态）L224 |
| 同上 L125 / L147 / L378 | threadblock_sync_state L125 / warp_sync_state L147（state 归约）；PersistentVariableLengthMergeStatesKernel L378 |
| 同上 L583 / L615 | MergeState host launcher L583（cudaLaunchKernel）/ MergeStateInPlace L615 |
| `include/flashinfer/attention/scheduler.cuh` L70-98 / L102-123 | **split-KV scheduler**：new_batch_size=Σ ceil_div(kv_len, chunk) L86/96；DecodePlan（qo_chunk_size, min_kv_chunk_size, max_batch_size_if_split）L102-123 |
| `flashinfer/cascade.py` L226 / L401 / L537 | **MultiLevelCascadeAttentionWrapper L226；plan() L401；run() L537**（cascade plan/run API） |
| 同上 L576 / L42 / L109 | BatchDecodeWithSharedPrefixPagedKVCacheWrapper L576；merge_state custom op L42 / merge_state_in_place L109 |
| `flashinfer/decode.py` L1338 / L2179 / L3237 | **BatchDecodeWithPagedKVCacheWrapper L1338；plan() L2179；run() L3237**（split-KV decode plan/run） |
| 同上 L541 / L115 / L797 | get_batch_decode_module（JIT）L541；disable_split_kv / fixed_split_size（split-KV 控制）L115/797 |
| `flashinfer/jit/core.py` L31 / L186-202 | **JIT**：generate_ninja_build_for_op / run_ninja L31；common_nvcc_flags L186；sm90a L194 / sm100a L198 / sm103a L199 / sm107a L201 / sm110a L202 nvcc arch flags |
| `include/flashinfer/attention/{decode,prefill,mla,mla_hopper,cutlass_mla}.cuh` | decode/prefill/MLA kernel（6.4 变体） |

### 3.7 NVIDIA/TransformerEngine — main（pushed 2026-11-25）· 5.3 norm kernel ⟨verify⟩ 销项

**路径漂移**：旧 `transformer_engine/common/layer_norm/`、`rms_norm/` 均 404 → 现行 `transformer_engine/common/normalization/{layernorm,rmsnorm}/`。

| file:line | 机制 |
| --- | --- |
| `transformer_engine/common/normalization/layernorm/ln_fwd_kernels.cuh` + `ln_fwd_cuda_kernel.cu` | LayerNorm forward kernel（mean/var reduction + normalize） |
| `transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_kernels.cuh` + `rmsnorm_fwd_cuda_kernel.cu` | RMSNorm forward kernel |
| `transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_mxfp8.cu` | RMSNorm + MXFP8 cast 融合（5.4 norm+quant 融合对照） |
| `transformer_engine/common/normalization/{kernel_traits.h,kernel_params.h}` | norm kernel traits/params |
| `transformer_engine/common/fused_softmax/` | fused softmax（5.3 对照） |

## 4. ⟨verify⟩ 销项 / 残留

| 项 | 状态 | 依据 |
| --- | --- | --- |
| FA4 是否存在 / tcgen05（spec 6.3） | **销** | FA4 存在：README L80-99 CuTeDSL（Hopper+Blackwell）+ flash_attn/cute/blackwell_helpers.py L8 tcgen05 + 论文 2603.05451（§2/§3.2） |
| CUTLASS 3.x/4.x 架构全量走读锚（spec 5.2，W2 残留） | **销** | 4.9.0 fresh tarball：device/kernel/collective mainloop/epilogue/CuTe layout/mma atom/wgmma 全量 file:line（§3.1） |
| TransformerEngine norm 路径（spec 5.3 ⟨verify W3〉） | **销** | normalization/layernorm\|rmsnorm（旧 layer_norm/rms_norm 404，§3.7） |
| CUB repo（spec 5.1 ⟨verify NVIDIA/cub〉） | 沿 W0 | NVIDIA/cub archived→并入 NVIDIA/cccl（W0 已核，本轮未重核） |
| Triton repo（spec 5.4 ⟨verify openai/triton〉） | 沿 W0 | openai/triton 301→triton-lang/triton（W0 已核） |
| B200 SM 数官方未披露（W1 残留 W3/W6） | **残留** | 本轮未销（blackwell-whitepaper 仍无 SM 数；留 W6 系统集成或官方后续披露） |
| Marlin bfe（spec 6.2「lop3/bfe」） | **精度订正** | Marlin 实际用 **lop3**（L136-137）+ `>>8` 移位（L447）unpack INT4，**非 bfe**；spec「lop3/bfe」为泛指 unpack 指令族，writer 6.2 须注 Marlin 具体用 lop3（§3.3） |

## 5. 版本漂移注（writer 必注）

1. **CUTLASS PipelineTmaUmmaAsync 位置订正（W2 注精炼）**：W2 记「PipelineTmaUmmaAsync 不在当前 main sm90_pipeline.hpp」——**正确但不完整**。本轮 fresh tarball 全 pipeline 目录 grep：**PipelineTmaUmmaAsync 在 `sm100_pipeline.hpp` L538 + `sm107_pipeline.hpp`**（Blackwell UMMA pipeline），**不在 sm90_pipeline.hpp**。SM90 mainloop（sm90_mma_tma_gmma_ss_warpspecialized.hpp L111）与 FA3 mainloop（mainloop_fwd_sm90_tma_gmma_ws.hpp L283）均用 **cutlass::PipelineTmaAsync / PipelineTmaAsyncNoCluster**。writer 5.2/6.3 引 SM90 pipeline 用 PipelineTmaAsync；引 Blackwell UMMA pipeline 用 sm100 PipelineTmaUmmaAsync（L538）。
2. **CUTLASS gemm_tutorial.md → efficient_gemm.md**（W2 已注，本轮再确认）：5.2 GEMM 层级教学锚 = `media/docs/cpp/efficient_gemm.md`（cutlass-docs 条目，被引 += 5.2）。
3. **FlashMLA main vs ba89a34 双锚**：main（395cb3af，2026-10-27）仅 SM100/SM103（V4.1，head_dim 512，FP8/FP4 KV 528/288 B）；Hopper absorbed 机制（V3，head_dim_k 576）→ **ba89a3466e9470ad08ab39738d4e7bb66989e1e7**（README L3 全 SHA）。**勿写 512+64=576 为 V4.1**——576 是 V3 几何（512 latent+64 rope），V4.1 是 512（448 latent+64 rope，全 FP8 量化）。docs/20250422-new-kernel-deep-dive.md 两版本均存（main 保留文档但移除 sm90 代码）。
4. **DeepGEMM 重构**：device kernel 从旧 `csrc/` 迁至 `deep_gemm/include/deep_gemm/impls/*.cuh`；JIT 经 third-party/deep_jit submodule（tarball 空，锚 csrc/runtime/jit.hpp 接口）；新增 FP4/MXF4、Mega MoE、MQA logits（lightning indexer）。SM90 FP32 scale vs SM100 packed UE8M0（README L67-68）。
5. **FlashInfer 0.7.6**：repo 372 MB（tarball 截断→contents API）；jit/ 含大量 cake_* / trtllm / cute_dsl 后端；plan/run API 在 decode.py（L2179/L3237）/ cascade.py（L401/L537）。
6. **TransformerEngine norm 路径**：layer_norm/rms_norm → normalization/layernorm|rmsnorm（§3.7）。
7. **FA3 pingpong 命名**：FA3 paper「pingpong scheduling」在当前代码无字面 pingpong 命名，实化为 **NumMmaWarpGroups==2 warpgroup 交错 + FwdNamedBarriers::WarpSchedulerWG1/2/3 + PFull/PEmpty**（named_barrier.hpp L52-58）+ IntraWGOverlap（mainloop L353）。CUTLASS 侧 kernel schedule 有 pingpong/cooperative 命名（sm90_gemm_tma_warpspecialized_pingpong.hpp）。writer 6.3 引 FA3 pingpong 须注代码实化名。
8. **FA4 vs FA3 硬件口径**：FA3=Hopper（wgmma+TMA+warp specialization）；FA4=Hopper+Blackwell CuTeDSL（tcgen05 全异步 MMA + 大 tile + 软件 exp，应对 Blackwell asymmetric scaling：TC 2× 但 shared 带宽/exp 单元慢）。

## 6. references 写入 / 被引回写（spec §5 Sources）

- **新增 1 条**：`flashattention4`（P6，[A]，arXiv:2603.05451 + Dao-AILab/flash-attention flash_attn/cute/，被引 6.3）。
- **更新条目（核实日期 2026-11-25 + 代码锚）**：cutlass（P1，5.2 层级全量 + PipelineTmaUmmaAsync 订正）/ deepgemm / transformerengine / nanoflow / tilelang（P5）/ marlin / flashattention / flashattention2 / flashattention3 / flashmla / flashinfer（P6）。
- **被引回写**：`cutlass-docs`（P4）被引 4.3, 4.4 → **4.3, 4.4, 5.2**（efficient_gemm.md 5.2 走读用）。其余跨 Part 共享锚点被引已含对应 P5/P6 章（ptx-isa 含 5.1/5.2/6.1/6.2/6.3；cuda-samples 含 5.1/5.3；cutlass 含 5.2；nvidia-datasheet 含 6.1），维持。
- **未重核条目维持原核实日期**：cub / cublaslt-docs / triton / pytorch（P5，2026-11-24，沿 W0）；ocp-mx / vllm / deepseekv3 / v41-flash-config（P6，2026-11-24，沿 W0/inference）。blelloch-scan 已 2026-11-25（W1）。
- **跨 Part 共享锚点收录首引 Part**：cutlass/ptx-isa/nvidia-datasheet → P1，cuda-samples → P3，cutlass-docs → P4；P5/P6 被引在对应条目跨组累加。

## 7. writer 白名单提示（T2-T5）

- **5.1 reduction/scan**：Blelloch [B]（CMU-CS-90-190，up/down-sweep work O(n_el)/depth O(log n_el)）；CUB→cccl；cuda-samples reduction/scan（cpp/2_Concepts_and_Techniques/）；PTX shfl；FA2 softmax.h quad_allreduce_（warp 4-thread 归约实例）。
- **5.2 GEMM 走读【重】**：cutlass 4.9.0（§3.1 全量：device gemm_universal.h → kernel sm90_gemm_tma_warpspecialized.hpp operator() L269 → collective sm90_mma_tma_gmma_ss_warpspecialized.hpp load L312/mma L417/cute::gemm L496 → epilogue sm90_epilogue_tma_warpspecialized.hpp L83；CuTe layout.hpp L99 / mma_atom.hpp L49/211/460；wgmma mma_sm90_gmma.hpp m64nNk16）；efficient_gemm.md（cutlass-docs，层级伪代码 L16-32）；cuBLASLt docs；DeepGEMM 对照（§3.4 promotion L329-345）；fp8_accumulation.hpp L40-43（FP8 两级累加）。**tile 流量账 recap 4.3，L2 复用/split-K 修正入走读**。
- **5.3 softmax/norm**：TransformerEngine normalization/layernorm|rmsnorm（§3.7，路径已修）；FA2 softmax.h（scale_apply_exp2 L67 / softmax_rescale_o L137）；online softmax 正确性 recap models 4.1（不重证）；cuda-samples。
- **5.4 fusion**：Triton（triton-lang/triton）/ PyTorch Inductor / NanoFlow [A] / TileLang [A]；FlashAttention fusion 视角 recap（6.3 正式走读）；TransformerEngine rmsnorm_fwd_mxfp8.cu（norm+quant 融合）。
- **6.1 低精度格式**：OCP MX [C] / datasheet [C] / PTX 格式节 [C]；MMA 吞吐逐格式 recap 1.3；DeepGEMM sm100 MXF4/MXF8F6F4（§3.4 L51-52）；FA4 blackwell_helpers.py tcgen05 逐格式（§3.2）。
- **6.2 量化 kernel【片段】**：Marlin（§3.3：lop3 L118-126 / dequant L131-152 magic-number / mma L98 / overlap L443；**用 lop3 非 bfe**）；DeepGEMM（§3.4：promotion L329-345 / UE8M0 / JIT）；PTX lop3/prmt/bfe [C]；GEMM 全量走读归 5.2（本章不重复）。
- **6.3 FA 走读【重】**：FA2（§3.2：compute_attn_1rowblock L57 Q outer / KV loop L387 / softmax_rescale_o L137/160 / gemm_rs L376）；FA3（warp specialization roles L220-222 / register realloc L82-83 / TMA L753 / FP8 L44 / pingpong=named_barrier WG1/2/3+PFull/PEmpty L52-58）；**FA4 [A] 2603.05451（CuTeDSL tcgen05，Blackwell）**；FlashMLA 对照（§3.5 seesaw）；tile↔shared/register 预算约束推导（Q/K/V 三份独立 tile）；GPU 侧精确 HBM 流量账（causal/split-KV/GQA 修正，recap models IO 定理不重推）。
- **6.4 变体 kernel【片段】**：FlashMLA 双锚（§3.5：main V4.1 sm100 + ba89a34 V3 sm90 splitkv_mla.cuh；几何 V4.1 512 / V3 576；seesaw L42 / 9 TMA L54）；FlashInfer（§3.6：cascade MergeStateKernel L45 / split-KV scheduler L70-123 / plan-run decode.py L2179/3237 / JIT core.py）；vLLM paged gather；split-KV reduction 指令走读（与 6.3 流量账分工：6.3=流量修正，6.4=reduction kernel）。