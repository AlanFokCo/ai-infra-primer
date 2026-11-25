# 5.2 GEMM kernel 设计与走读（GEMM Kernel Design and Walkthrough）

> 4.3 留了一个缺口：per-block 强度 $I = 64$ 低于 ridge 295，带宽口径 640.7 μs，可真实 GEMM 是 compute-bound、算力地板 138.9 μs——差的 4.61 倍由 L2 跨 block 复用补上（波前复用因子 $\sqrt{132} \approx 11.5$）。本章 recap 4.3 tiling 理论（不重推），对 CUTLASS 4.9.0 做模块全量走读：device→kernel→collective→CuTe→wgmma 五层实体逐文件对照（gemm_universal.h → sm90_gemm_tma_warpspecialized.hpp operator() L269 → CollectiveMma L72 / cute::gemm L496 → epilogue L83），mainloop pipeline（4.4 recap）与 warp specialization 角色分配，tile 层级流量账实例化 + L2 修正（命题 1）、MMA 利用率分解（命题 3）、cuBLASLt heuristic 与 DeepGEMM FP8 fine-grained scaling 对照 | 前置：4.3（命题 1-4 与式 (1)-(8)，tiling 理论 recap 源）、4.4（命题 1 与式 (1)(2)、pipeline 同步骨架）、1.3（wgmma 几何/异步语义/累加器式 (4)）、2.4（roofline 与带宽库存）、3.4（occupancy 四约束）、4.2（swizzle）、3.1（cluster 语义）；下游 6.2（量化 kernel 片段级走读 recap 本章 mainloop）、6.3（FA mainloop 同构走读）、5.4（epilogue fusion 带宽账） | 锚点：CUTLASS 4.9.0（[NVIDIA, CUTLASS](../references.md)，commit 5c179edeb3e2，教学参照 + 官方实现）、CUTLASS efficient_gemm.md（[NVIDIA, CUTLASS docs](../references.md)，教学参照）、cuBLAS/cuBLASLt docs（[NVIDIA, cuBLASLt](../references.md)，[C] 官方规格披露·一手权威）、DeepGEMM（[DeepSeek-AI, DeepGEMM](../references.md)，commit 5081f8513af5，教学参照 + 官方实现）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）——代码锚 2026-11-25 fresh tarball 一手核实

## 1. 4.61 倍的缺口：本章口径与两层纪律

同一个 4096³ bf16 GEMM，4.3 的 per-block 账给带宽口径 640.7 μs，算力地板只要 138.9 μs——per-block 强度 64 对 ridge 295 明明在斜线区，为什么实测 GEMM 落在平台区？缺口不在 tiling 理论，在理论的一条假设：「每 tile 从 HBM 读、无跨 block L2 复用」。真实 kernel 的 block 调度让同一份 A/B tile 被多个 block 在 L2 里共享，HBM 流量压到 per-block 账的约 1/11。本章回答三个问题：L2 复用与 block 调度怎么量化修正流量账（命题 1）？CUTLASS 4.9.0 的五层实体各在哪个文件、mainloop 的 producer/consumer 怎么交接（模块全量走读）？MMA 利用率怎么分解成可逐项核对的因子（命题 3），cuBLASLt 与 DeepGEMM 各在哪一层做选择？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0（sm_90a），bf16 输入、fp32 累加**（沿 4.3）；running example 沿 4.3 方阵 GEMM $M = N = K = 4096$。走读配置取 CUTLASS SM90 base warp-specialized kernel：TileShape $128 \times 128 \times 64$、Stages = 4、ClusterShape $1 \times 1$、1 个 MMA warp-group + 1 个 load warp-group = 256 threads。**换口径声明**：4.3 教学 tile 取 $T_K = 32$ 是机制口径；本章走读配置 $T_K = 64$，因为 wgmma m64n128k16 的 k16 几何下 64 恰为 4 个 k block，TMA box 与指令几何对齐（§6）——强度式 (2) 与 $T_K$ 无关（4.3 recap），换的是延迟账与 shared 预算。 $\mathrm{BW} = 3352$ GB/s、 $F_{\mathrm{peak}} = 989.5$ TFLOPS、ridge $R \approx 295$、每 SM 份额 $`\mathrm{BW}_{\mathrm{SM}} = 25.4`$ GB/s、 $F_{\mathrm{SM}} = 7.496$ TFLOPS（2.4/4.3 recap）；周期↔时间取 TC-boost 1.83 GHz，HBM 延迟 478.8 cycles 为 H800 微基准实测 [B]（[Luo et al., 2024](../references.md)，沿 4.3 声明）。片上 KB = 1024 B。

符号声明（spec §4.3）：**本章 $M, N, K$ = GEMM 问题维（行/列/归约维），非消息字节/参数量/序列长**； $T_M, T_N, T_K$ = threadblock tile 三维、 $T_{\mathrm{tile}}$ = 方阵 tile 边长（**tile 维，非序列长 $T$；禁裸 $T$ 表 tile**）； $n_{\mathrm{SM}} = 132$、 $n_{\mathrm{blk}}$ = grid block 数、 $n_{\mathrm{wave}}$ = 波次数、 $n_{\mathrm{split}}$ = split-K 段数； $b$ = 波前并发 block 数、 $r, c$ = 波前覆盖的 A 行 panel / B 列 panel 数（§2 局部）； $U$ = MMA 利用率（achieved/peak）， $U_{\mathrm{geom}}, U_{\mathrm{wave}}, U_{\mathrm{main}}, U_{\mathrm{epi}}$ = 分解因子（§8）； $S$ = pipeline stage 数（4.4 口径，CUTLASS 模板参数名 Stages）； $t_{\mathrm{comp}}, t_{\mathrm{load}}, t_{\mathrm{lat}}, \tau_S$ 沿 4.3/4.4； $t_{\mathrm{epi}}$ = epilogue 时间（§8 教学假设）； $\mathrm{bytes} = 2$（bytes 不占字母， $B$ 专归 batch）。**指令几何一律字面数**：wgmma m64n128k16 等；族名 m64nNk16 里的 N 是 PTX 指令字面占位（8 的倍数，8..256），**与本章 GEMM 维 $N$ 无关**。

**版本锚**：CUTLASS **4.9.0**（commit 5c179edeb3e2；`include/cutlass/version.h` L41-43 MAJOR 4 / MINOR 9 / PATCH 0，CHANGELOG 4.9.0 2026-11-13）。本章全部 file:line 按该版本 fresh tarball 核实（2026-11-25）；跨版本阅读先对 CHANGELOG——4.4 坑注的精确化：`PipelineTmaUmmaAsync` 在 `include/cutlass/pipeline/sm100_pipeline.hpp` L538（Blackwell UMMA 流水），**不在 sm90_pipeline.hpp**；SM90 mainloop 用 `PipelineTmaAsync`（§5）。

**边界声明（两层走读纪律，4.3↔5.2）**：4.3 交付 tiling 的复用与强度**理论**（代码 scope 机制片段级），本章做 CUTLASS **模块全量走读** + 公式逐行对照；4.3 的命题与公式本章一律 recap + 指引、不重推，增量 = L2 复用修正、五层代码实体、利用率分解与工程选择。1.3（MMA 指令几何/异步语义）与 4.4（cp.async/TMA/pipeline 机制）同为 recap 源，重访其片段一律指引。量化 kernel（Marlin/DeepGEMM 片段）归 6.2、FA mainloop 归 6.3、格式位布局归 6.1、epilogue fusion 的带宽节省账归 5.4；Blackwell tcgen05 全量走读不在本章（1.3 §6 机制 + 6.3 FA4 指针），本章聚焦 SM90。

## 2. 4.3 理论 recap 与 L2 波前修正

先 recap 理论侧（4.3 §3，不重推）：shared tiling 的 per-block 强度 $I_{\mathrm{blk}} = 2T_M T_N/((T_M+T_N)\mathrm{bytes})$（4.3 式 (2)），方阵 tile 即 $T_{\mathrm{tile}}/\mathrm{bytes}$；全 kernel HBM 流量 $`\mathrm{traffic}_{\mathrm{HBM}} = MNK(1/T_M + 1/T_N)\mathrm{bytes}`$（4.3 式 (3)），**假设每 tile 从 HBM 读、无跨 block L2 复用、C 写低阶略**。代入本章配置 $T_M = T_N = 128$： $I_{\mathrm{blk}} = 64$ FLOP/byte，流量 2.147 GB、带宽口径 640.7 μs——per-block 口径 memory-bound。4.3 把修正留给了本章：L2 跨 block 复用与 rasterization。

机制一句话：A tile $(i, k)$ 被第 $i$ 行的全部 $N/T_N = 32$ 个 block 消费，B tile $(k, j)$ 被第 $j$ 列的全部 $M/T_M = 32$ 个 block 消费；若这些 block 在时间上接近（同一波前），tile 在 L2 命中，HBM 只读一次。L2 容量 50 MB（2.2 recap）装得下波前工作集（下文核验），所以修正的量级由**波前形状**决定，不由容量决定。

**命题 1**（L2 波前复用因子）。设 $b$ 个并发 block 恰排成 $r$ 行 $\times$ $c$ 列（ $b = rc$），方阵 tile $T_{\mathrm{tile}}$，波前内 L2 完美命中（教学假设），则每 K 步 HBM 流量对无复用口径的缩减因子为 $2b/(r+c)$；方阵波前（ $r = c = \sqrt{b}$）给 $\sqrt{b}$。**证明**（计数，两步）。其一，无复用时 $b$ 个 block 各读 A/B tile 一份，每 K 步 $b \cdot 2T_{\mathrm{tile}} T_K$ 元素；波前只覆盖 $r$ 个不同的 A 行 panel 与 $c$ 个不同的 B 列 panel，完美命中下每个不同 tile 从 HBM 读一次，流量 $(r+c) T_{\mathrm{tile}} T_K$ 元素。其二，因子 $= 2b T_{\mathrm{tile}} T_K / ((r+c) T_{\mathrm{tile}} T_K) = 2b/(r+c)$； $r = c = \sqrt{b}$ 时 $= \sqrt{b}$。∎

全数字代入（ $b = n_{\mathrm{SM}} = 132$，1 block/SM，§4 occupancy）：

| 波前形状 | $r$ | $c$ | 因子 $2b/(r+c)$ | 有效 HBM 流量 | 有效强度 |
| --- | --- | --- | --- | --- | --- |
| 行优先 raster（ $c = N/T_N = 32$ 封顶） | 4.125 | 32 | 7.31 | 293.8 MB | 467.7 |
| 方阵波前（ $r = c = \sqrt{132}$） | 11.49 | 11.49 | 11.49 | 186.9 MB | 735.3 |

有效强度 = $64 \times$ 因子，两形都越过 ridge 295——**bound 判定翻转为 compute-bound**。带宽口径时间（方阵波前 + C 写 33.55 MB）： $(186.9+33.55) \times 10^{6}/(3352 \times 10^{9}) = 65.8$ μs，低于算力地板 138.9 μs，与实测 GEMM 落平台区一致。工作集核验（教学假设的成立条件）：每 K 步波前需 $(r+c)$ 个 tile 共约 368 KB；整条 K 深的 panel 工作集 $2r \times (128 \times 4096 \times 2)$ B ≈ 24.1 MB，低于 L2 50 MB——完美命中假设在容量侧成立（时间侧的逐出与调度抖动是实测损失，归 11.2）。

教学假设的边界一并钉死：命题 1 忽略末波不满（ $n_{\mathrm{blk}}$ 非 $n_{\mathrm{SM}}$ 整数倍时尾波复用因子下降，§8 的 $U_{\mathrm{wave}}$ 单独计这笔）、忽略 L2 逐出抖动（工作集核验只保容量侧）、忽略 C 写与 descriptor 流量（低阶）。persistent 调度（block 数 = $n_{\mathrm{SM}}$、每 block 循环领 tile）把波次量化摊平成连续流，是 cooperative 变体的取向；base kernel 非 persistent（§4 union 注释），波次账按式代入。

代码侧旋钮：SM90 tile scheduler 的 `RasterOrderOptions`（AlongM/AlongN/Heuristic）与 `log_swizzle_size` 决定波前形状（`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` L47-48、L105-121 的 swizzle 索引变换；group scheduler 默认 AlongM，`sm90_tile_scheduler_group.hpp` L134）——swizzle 把连续 cluster 编号折成近方阵的二维分布，正是命题 1 里「 $r \approx c$」的工程实现。

> **坑**：拿 per-block 强度当实测 bound 判定（4.3 坑注的兑现）。触发条件：用式 (2) 算出 64 就断言 GEMM memory-bound；现象：实测 compute-bound，两个结论对不上；规避：式 (2)(3) 是「无 L2 复用」上界口径，用于 tile 选型（单调升）；整 kernel bound 判定用本章波前修正或实测流量（2.4 式 (8) 口径）。反向同理：拿实测 compute-bound 否定式 (2) 也不成立——它没错，只是假设不含 L2。

## 3. GEMM 层级：概念五层与 device 层实体

一个 GEMM kernel 的并行性分几层？先钉动机：Transformer 的 FLOPs 大头是 GEMM——QKV/O 投影与 FFN 全是矩阵乘（models 1.2 计算账 recap），GEMM kernel 的利用率几乎直接折算成端到端算力利用率，这是它值得模块级走读的原因。CUTLASS 教学文档给出概念骨架（`media/docs/cpp/efficient_gemm.md` L16-32，[NVIDIA, CUTLASS docs](../references.md)，教学参照），嵌套循环逐字结构：

```
for cta_n / cta_m        ← threadblock-level concurrency（每 block 一个 C tile）
  for cta_k              ← "GEMM mainloop"，一次迭代 = 一个 "stage"
    for warp_n / warp_m  ← warp-level parallelism
      for warp_k         ← 沿 CtaTileK 全展开，一次迭代 = 一个 "k Group"
        for mma_k / mma_n / mma_m  ← instruction-level parallelism
          mma_instruction(d, a, b, c)  ← TensorCore 矩阵计算
```

3.x/4.x 把这五层映射到具体实体（本章走读路线，自顶向下）：

```
device 层    GemmUniversal（gemm/device/gemm_universal.h L142/L289）
             host 入口：initialize() L398 → can_implement / get_grid_shape
kernel 层    GemmUniversal<CollectiveMainloop, CollectiveEpilogue, TileScheduler>
             （gemm/kernel/sm90_gemm_tma_warpspecialized.hpp L65，operator() L269）
collective 层 CollectiveMma（gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp L72）
             + CollectiveEpilogue（epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp L83）
CuTe 层      TiledMMA / ThrMMA（cute/atom/mma_atom.hpp L211/L460）+ Layout（cute/layout.hpp L99）
instruction 层 wgmma.mma_async m64n128k16（cute/arch/mma_sm90_gmma.hpp，§6）
```

**2.x 与 3.x 的实体差异**：2.x 把层级做成显式目录——`include/cutlass/gemm/threadblock/`（default_mma_core_sm80.h 等）、`gemm/warp/`（default_mma_tensor_op.h）、`gemm/thread/`（mma.h），threadblock/warp/thread 各一类；3.x/4.x 改用 collective + CuTe 代数，层级藏在模板组合里。分流在 `gemm/kernel/gemm_universal.hpp` L42-57：`IsCutlass3ArrayKernel` 按 ProblemShape 类型把 2.x API 与 3.x API 分开 dispatch。本章走读 3.x/4.x 路径（SM90），2.x 目录只作概念对照。

device 层走读（`gemm/device/gemm_universal.h`）：2.x 基类 L142、3.x 特化 L289；3.x 侧 `GemmKernel = typename UnderlyingOperator::GemmKernel`（L355）把 kernel 层类型提上来，`initialize()`（L398）是 host 入口——参数转 underlying arguments、can_implement 检查、workspace 初始化、TMA descriptor 创建（A/B 各一个 128 B tensor-map，host `cuTensorMapEncodeTiled`，4.4 §3 recap）。**device 层不做计算**：它把问题形状、布局、融合参数翻译成 kernel 层的 Params，grid/block 形状由 kernel 层的 `get_grid_shape`/`get_block_shape` 给出。走读配置的翻译实例：问题 $(M, N, K) = (4096, 4096, 4096)$ 加 stride（A 行主序 $(K, 1)$、B 列主序 $(K, 1)$ 的 NT 布局）进 Arguments，`to_underlying_arguments` 转成 collective 的 Params（含 TMA descriptor 与 tma_transaction_bytes），grid = $(M/T_M, N/T_N) = (32, 32)$（blockIdx.x 映 M、blockIdx.y 映 N，operator() L406-407）、block = 256 线程。host 侧一次性付的成本是 descriptor 创建与 can_implement 检查（对齐/整除断言），运行期每 launch 只传 Params——4.4 的「descriptor 256 B 占流量 $1.2 \times 10^{-7}$」账在这里兑现。

## 4. kernel 层：warp specialization 与 operator() 走读

256 个线程怎么分工？SM90 base warp-specialized kernel 的答案是按 warp-group 分角色（`gemm/kernel/sm90_gemm_tma_warpspecialized.hpp`）：`NumLoadWarpGroups = 1`、`NumMmaWarpGroups = 1`（L140-141），`MaxThreadsPerBlock = size(TiledMma{}) + NumLoadWarpGroups * NumThreadsPerWarpGroup = 128 + 128 = 256`（L142）。`operator()`（L269）开头定义两个角色枚举：`WarpGroupRole{Producer, Consumer}` 按 warp-group 编号分——WG0 是 producer、WG1 是 consumer；producer WG 内再分 `ProducerWarpRole`，**只有 warp 0（MainloopEpilogue）发 TMA**，其余 warp 在 base kernel 里无装载任务。角色落到 pipeline 参数（L317-327）：producer/consumer 各自设 `MainloopPipeline::ThreadCategory`，`transaction_bytes = params.mainloop.tma_transaction_bytes`（L326）——4.4 的 expect_tx 字节账在这里注入。

两分支的执行体（L436-510）：

```
producer WG（warp0 elected lane）：
  collective_mainloop.load(...)      L436  ← K 维 TMA 装载循环（§5）
  collective_mainloop.load_tail(...) L450  ← 流水排空
  collective_epilogue.load(...)      L455  ← C tile 预取（若 fusion 需要 source C）
consumer WG：
  partition_fragment_C(tiled_mma, ...) L470 ← 累加器 fragment（register）
  collective_mainloop.mma(...)       L472  ← K 维 wgmma 循环（§5）
  collective_mainloop.mma_tail(...)  L483  ← warpgroup_wait<0> 排空
  collective_epilogue.store(...)     L496  ← fusion + TMA 写回（§7）
  collective_epilogue.store_tail(...) L510
```

一个读法疑问：producer 占整个 warp-group（128 线程）却只有 warp 0 干活，岂不浪费？base kernel 确实如此——这是模板结构的基线形态；cooperative 变体把 consumer 侧扩到 2 个 MMA warp-group 分摊更大 tile（producer 侧仍 warp0 发 TMA），并做寄存器再分配（见下文变体段）。warp specialization 的收益不在「每个 warp 都有活」，在**角色分离后三条管线互不占发射槽**：TMA 搬运、wgmma 计算、CUDA Core 辅助工作并行（4.4 §4 recap），空闲 warp 的成本是寄存器与 warp slot，换来的是装载与计算的重叠。

producer 的 pipeline 起始态用 `make_producer_start_state`（相位反转，跳过首轮 empty 等待——缓冲初始即空，注释逐字「we skip all waits since we know that the buffer is indeed empty」）。`SharedStorage`（L120-138）把 mainloop 与 epilogue 的 shared 做成 **union**：base kernel 非 persistent，mainloop 与 epilogue 不同时用 shared，容量按较大者计——这是 §7 epilogue subtile 缓冲能塞进预算的结构原因。

**occupancy 账**（走读配置，3.4 式 (1) recap 代入）：shared = Stages × 每 stage 字节 = $4 \times 32768 = 131072$ B = 128 KB（§5），加 1 KB 预留后 $S_{\mathrm{alloc}} = 132096$ B，shared 项 $\lfloor 233472/132096 \rfloor = 1$ block；register 侧 consumer 累加器 128 regs/thread（§6）加描述符/地址变量（教学假设合计约 168），producer 约 40，block 总量约 $128 \times 168 + 128 \times 40 = 26624$，register 项 $\lfloor 65536/26624 \rfloor = 2$；warp 项 $\lfloor 64/8 \rfloor = 8$、block slots 32。四项取 min = **1 block/SM，shared 绑定**， $\mathcal{O} = 8/64 = 0.125$。低 occupancy 可接受：compute-bound kernel 靠 TMA + wgmma 异步流水补并发（3.4 命题 2 recap），不靠 warp 数——§5 的 stage 账兑现这句话。绑定约束的松紧值得看一眼：Stages 降到 3 则 shared = 96 KB， $S_{\mathrm{alloc}} = 99328$ B，shared 项 $\lfloor 233472/99328 \rfloor = 2$ blocks，register 项 $2 \times 26624 = 53248 \le 65536$ 仍可行，occupancy 翻到 0.25——但预取份数恰落在 $S_{\min} = 3$ 地板（§5），一份余量都没有。Stages = 4 是用 occupancy 换流水余量：L2 修正口径下计算主导、occupancy 边际收益小，取 4 合理；若实测 L2 命中差（装载主导回归），先修 raster/swizzle 抬 L2 命中（§2）——stage 数只藏延迟、不突破带宽地板（4.4 命题 1 recap）。

**schedule 变体**：cooperative（`sm90_gemm_tma_warpspecialized_cooperative.hpp`）用 2 个 MMA warp-group 分摊更大 tile，并做寄存器再分配——`LoadRegisterRequirement = 40` / `MmaRegisterRequirement = 232`（L150-151），producer 侧 `warpgroup_reg_dealloc<40>`（L562）、consumer 侧 `warpgroup_reg_alloc<232>`（L772）：producer 让出寄存器给 consumer 的累加器。pingpong 变体（`sm90_gemm_tma_warpspecialized_pingpong.hpp`）让两个 MMA WG 交替做 mainloop 与 epilogue，藏 epilogue 延迟。FA3 的 warp specialization + pingpong 是这条结构在注意力侧的实例（指令级走读归 6.3）。

## 5. collective mainloop：load/mma 与 pipeline 交接

collective 层是 mainloop 的本体（`gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`）：`struct CollectiveMma`（L72）特化于 DispatchPolicy `MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>`（L92）；**`MainloopPipeline = cutlass::PipelineTmaAsync<Stages>`（L111）**——正是 4.4 §5 走读的 pipeline 类（full/empty barrier 各 Stages 份，`sm90_pipeline.hpp` L289/L299-300）。SmemLayoutA/B = `tile_to_shape(SmemLayoutAtom, (T_M, T_K, Stages))`（L130-134，PIPE 维即 stage 维），`static_assert Stages >= 2`（L137）；SS 变体的硬约束逐字（L138-140）：「MMA atom must source both A and B operand from smem_desc for this mainloop.」——A/B 都走 shared 描述符直喂（1.3 §5 wgmma 语义 recap），不经寄存器中转。

**字节账注入**：`TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK`（L265-269），按 SmemLayout 尺寸 × 元素位宽算。代入走读配置：A tile $128 \times 64 \times 2 = 16384$ B + B tile $64 \times 128 \times 2 = 16384$ B = **32768 B/stage**——4.4 的 expect_tx 16384 B 是 $T_K = 32$ 口径，本章 $T_K = 64$ 翻倍（换口径见 §1）。

`load()`（L310，producer 视角）：`elect_one_sync` 选单 lane（4.4 TMA 单线程发射 recap），K 维循环每轮 `producer_acquire`（L374，等 empty barrier）→ 两条 TMA copy（L384-385）：`tma_load_a.with(*tma_barrier, mcast_mask_a)` 与 `tma_load_b.with(...)`——**op 数按张量数计，A/B 各一条**（4.4 §3 口径），完成由硬件 complete_tx 记账。`load_tail`（L396）排空流水防 cluster 内 block 早退。

**cluster multicast**（3.1/4.4 留到本章的机制）：ClusterShape 大于 1 时，`mcast_mask_a` 沿 cluster 的 N 维收集目标 CTA、`mcast_mask_b` 沿 M 维（L355-370）——同一 A tile 被 cluster 内 $C_N$ 个 CTA 共享、同一 B tile 被 $C_M$ 个共享，TMA `.multicast::cluster` 让一次 global 读落多个 CTA 的 shared（DSMEM，PTX [C]）。流量修正（命题 1 同法计数；设 cluster 形状 $C_M \times C_N$ = 沿 M/N 维的 CTA 数）：A 侧 HBM 读除以 $C_N$、B 侧除以 $C_M$，总流量 $MNK(1/(T_N C_N) + 1/(T_M C_M))\mathrm{bytes}$。代入 cluster $2 \times 1$（ $C_M = 2, C_N = 1$）： $4096^{3} \times (1/128 + 1/256) \times 2 = 1.610$ GB，对 2.147 GB 省 25%——multicast 是 L2 复用之外的第二条流量压缩路径，两者可叠加。mask 位与 cluster 内 CTA rank 一一对应：置位 k 表示 tile 落 rank k 的 CTA shared（DSMEM 地址空间，3.1 cluster 语义），elected lane 发一条 multicast op、硬件复制到各目标 CTA 并各自完成 tx 记账。走读配置 cluster $1 \times 1$，mask 为 0、退化为普通 TMA_LOAD。

`mma()`（L417，consumer 视角）：`consumer_wait`（L487，等 full barrier）→ `warpgroup_arrive`（L490）→ **`cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(...), accum)`（L496）** → `warpgroup_commit_batch`（L500）。结构分三段：序章（prologue）发射 `min(K_PIPE_MMAS, k_tile_count)` 个 tile 的 MMA 把流水填满；稳态循环每轮 wait → gemm → commit → `wait<1>` → release；尾声 `mma_tail` 排空。首个 k block 用 `GMMA::ScaleOut::Zero`（累加器清零起跑），随后 `ScaleOut::One`（累加）——k 方向归约的硬件实现（5.1 §1 关系段）。序章为什么手动展开 K 维（L494-498 的 k_block 循环）：只有第一个 k block 需要 ScaleOut::Zero，其余全是 One，展开后把 scale 切换变成编译期常量序列，免运行期分支——wgmma 的 scale-D 谓词是指令操作数（PTX [C]），逐条设定。稳态循环里一个关键时序（L546-551）：`warpgroup_wait<K_PIPE_MMAS>()`（L547）**先于** `consumer_release`（L551）——`K_PIPE_MMAS = 1`（L264，`K_PIPE_MAX = Stages`，L263）：缓冲释放滞后一个 tile，因为 wgmma 异步读 shared，发射返回不代表操作数读完，提前 release 会让 producer 覆写还在被读的 tile（1.3 异步语义的同步侧兑现）。`mma_tail`（L563）以 `warpgroup_wait<0>`（L571）等全部 MMA 完成再进 epilogue。

**pipeline 效率账**（4.4 命题 1 recap： $\tau_S = \max(t_{\mathrm{comp}}, t_{\mathrm{load}}, (t_{\mathrm{lat}}+t_{\mathrm{load}})/(S-1))$，不重推）。代入走读配置 $T_K = 64$： $t_{\mathrm{comp}} = 2 \times 128 \times 128 \times 64/(7.496 \times 10^{12}) = 279.8$ ns， $t_{\mathrm{load}} = 32768/(25.4 \times 10^{9}) = 1290.4$ ns， $t_{\mathrm{lat}} = 261.6$ ns：

| 口径 | $t_{\mathrm{load}}$ | $(t_{\mathrm{lat}}+t_{\mathrm{load}})/3$ | $\tau_4$ | 主导 |
| --- | --- | --- | --- | --- |
| 无 L2 复用（4.4 口径） | 1290.4 ns | 517.3 ns | 1290.4 ns | 装载 |
| L2 波前修正（÷11.49） | 112.3 ns | 124.6 ns | 279.8 ns | 计算 |

L2 修正后 $S_{\min} = 1 + \lceil (261.6+112.3)/279.8 \rceil = 3$（4.4 式 (2)）——Stages = 4 留一份余量。**同一 kernel，流量假设换一档，主导项从装载翻到计算**：这就是 §2 bound 翻转在 pipeline 侧的读法。

```
SM90 warp-specialized mainloop（Stages = 4，T_K = 64）：
producer warp0: │acq│TMA A+B tile k│acq│TMA tile k+1│ ...   ← 2 条 op/stage，expect_tx 32768
consumer WG1  :      │wait│wgmma×8 tile k│wait│wgmma×8 tile k+1│ ...
shared 缓冲   : [stage 0][stage 1][stage 2][stage 3] 循环
release 时序  : consumer 算完 tile k 后 warpgroup_wait<1> 再 release（滞后 1 tile）
```

## 6. CuTe layout 与 MMA atom：从 tile 代数到 wgmma

tile 在代码里长什么样？CuTe 的核心抽象是 layout（`include/cute/layout.hpp`）：`Shape = cute::tuple`（L48）、`Stride = cute::tuple`（L51）、`struct Layout`（L99）持 (shape, stride) 对——**layout 是从坐标到偏移的代数**：偏移 = 各维坐标 × 对应 stride 之和。正例：行主序 A 的 layout $(4096, 4096):(4096, 1)$，坐标 $(i, k)$ 映到 $4096i + k$；A tile 的 layout $(128, 64):(64, 1)$ 同构。shared 侧的 SmemLayoutAtom 带 swizzle（4.2 recap：`Swizzle<B,M,S>` 异或映射，`cute/swizzle.hpp` L52「AA = ZZ xor YY」），`tile_to_shape` 把 atom 平铺成 $(T_M, T_K, \mathrm{Stages})$ 三维——stage 维是纯偏移加法，PIPE 循环不改 atom 内部布局。

MMA 侧三级实体（`include/cute/atom/mma_atom.hpp`）：**`MMA_Atom<MMA_Traits<Op>>`（L49）**封装一条指令的几何与操作数类型（ValTypeD/A/B/C）；**`TiledMMA : MMA_Atom`（L211）**把 atom 按线程布局铺成 warp-group/block 级 MMA；**`ThrMMA : TiledMMA`（L460）**是 `get_slice(thr_idx)`（L359）切出的每线程视角，`partition_A`（L478）/`make_fragment_A` 把 tile 切成该线程持有的片段；`make_tiled_mma`（L531）是构造入口。三级的分工一句话：atom 答「一条指令算什么形状」，TiledMMA 答「一个 warp-group 怎么铺满 CTA tile」，ThrMMA 答「我这个线程持有哪份」。`mma()` 里的 `tCsA = thread_mma.partition_A(sA)` 返回 (MMA, MMA_M, MMA_K, PIPE) 四维视图——MMA 维是 atom 内每线程份额，MMA_M/MMA_K 是 tile 内 atom 重复数，PIPE 是 stage 维；`make_fragment_A` 把它转成描述符迭代器，`cute::gemm` 沿 MMA_M/MMA_K 逐 atom 发射（§5 的 k_block 循环即 MMA_K 维）。走读配置的 atom 是 `SM90_64x128x16_F32BF16BF16_SS`（`mma_traits_sm90_gmma.hpp` L1716 别名），TiledMMA 铺满 128 线程（1 warp-group）。**SS 路径的 fragment 是描述符不是数据**：`make_fragment_A/B` 返回 GMMA::DescriptorIterator（L138-140 的 static_assert 钉死），寄存器里只有 64-bit shared 描述符（起始地址 + leading/stride byte offset + swizzle 模式，PTX Matrix Descriptor [C]），A/B 本体留在 shared——1.3 §4「B 直喂 shared 省寄存器」的代码兑现。描述符的读法：硬件按描述符生成操作数读序列，软件每 stage 只更新地址字段（描述符迭代器沿 PIPE 维前进），swizzle 模式字段与 SmemLayoutAtom 的 swizzle 一致（4.2）——TMA 按 swizzle 落位、wgmma 按 swizzle 读出，两侧布局约定在 descriptor 里对上，软件免手工置换。

**命题 2**（每 tile 的 wgmma 计数）。atom m64n128k16、1 warp-group、CTA tile $T_M \times T_N \times T_K$ 时，每 K tile 的 wgmma 条数 $= (T_M/64)(T_N/128)(T_K/16)$；走读配置给 8 条，FLOPs 与指令几何闭合。**证明**（计数）：M 维需 $T_M/64$ 个 atom 行、N 维 $T_N/128$ 个 atom 列、K 维 $T_K/16$ 个 k block，各组合恰一条指令；代入 $2 \times 1 \times 4 = 8$，FLOPs $8 \times 2 \times 64 \times 128 \times 16 = 2097152 = 2 \times 128 \times 128 \times 64$，与 tile FLOPs 恒等。∎

1.3 recap 接上吞吐与延迟：m64n128k16 单指令 262144 FLOPs，稳态发射间隔 64 cycles（ $c_{\mathrm{SM}} = 4096$），实测单条延迟 128 cycles [B]（延迟≠吞吐，1.3 坑注）；8 条 × 64 = 512 cycles = 279.8 ns @1.83 GHz——**与 §5 的 $t_{\mathrm{comp}}$ 逐位吻合**，指令几何账与 roofline 账闭环。累加器：单指令 $64 \times 128$ fp32 摊 128 线程 = 64 regs/thread（1.3 式 (4) 代入指令字面 128：128/2 = 64），2 个 M tile 合计 **128 regs/thread**——§4 occupancy 的累加器主项。指令代码锚：`cute/arch/mma_sm90_gmma.hpp` 的 `wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16`（L158）、m64n16k16（L249）、m64n32k16（L341）、m64n64k16（L438）；bf16 SS 形 `m64n128k16.f32.bf16.bf16`（L2842）。

操作数带宽账（1.3 §4 recap + 本章 tile 级实例化）：单指令 shared 读 A $64 \times 16 \times 2 = 2048$ B + B $16 \times 128 \times 2 = 4096$ B = 6144 B；每 tile 8 条共 49152 B——其中 A tile 读一遍（16384 B）、**B tile 读两遍**（每个 M tile 各读一次，32768 B）。折每周期 $49152/512 = 96$ B/cycle，占 shared 峰值 128 B/cycle 的 75%（1.3 结论复现）——wgmma 的 shared operand 流落在预算内，4.3 §4 的 shared 瓶颈由这条结构解掉。

## 7. epilogue：fusion 与写回

累加器算完，结果怎么出去？epilogue collective（`epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`）：`class CollectiveEpilogue`（L83）特化于 `Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>`；**EpilogueTile**（模板参数 L68，(EPI_TILE_M, EPI_TILE_N)）把 CTA tile 切成子块，static_assert 钉死 rank-2 且整除 CTA tile（L128-130）——子块是 shared 缓冲与写回流水的粒度。融合操作由 **`ThreadEpilogueOp = FusionCallbacksTraits<FusionCallbacks>::Operation`（L122）**注入：fusion callbacks 是逐 fragment 的 visitor 树（EVT，`epilogue/fusion/sm90_visitor_*.hpp`），把 bias/activation/量化 scale 等逐元素链组合进写回路径。现成组合如 `LinCombEltAct`（`epilogue/fusion/operations.hpp` L143， $D = \mathrm{activation}(\mathrm{alpha} \cdot \mathrm{acc} + \mathrm{beta} \cdot C)$）与 `LinCombPerColBiasEltAct`（struct L240，其上方 L229 注释逐字「D = activation(alpha * acc + beta * C + per-column bias)」）——线性组合、bias、激活一趟完成。

走读一个具体 fusion：bias + ReLU + bf16 转换（`LinCombPerColBiasEltAct`，operations.hpp L240）。visitor 链逐 fragment 执行：读累加器 fragment（fp32，register）→ 乘 alpha 加 bias（bias 经 TMA 预取进 shared，epilogue.load 分支，§4）→ ReLU 逐元素 → 转 bf16 → 写 shared D 子块 → TMA store。整条链的中间值全在 register/shared，**没有一个中间 tensor 落 HBM**——若 bias/activation 拆成独立 kernel，每个都要一趟「读 D + 写 D」的 HBM 往返（ $2MN \cdot \mathrm{bytes}$/趟），fusion 把这笔省掉（节省账的推导归 5.4）。epilogue 的 CUDA Core 工作与 mainloop 的 wgmma 在 base kernel 里串行（union 结构），pingpong 变体让两个 WG 交替做 mainloop/epilogue 把这段藏起来（§4）。

数据流：register 累加器（fp32）→ 逐 fragment ThreadEpilogueOp（CUDA Core 逐元素）→ shared D 子块 → TMA store 到 global。子块化的意义：CTA tile 的 fp32 累加器（128×128×4 = 64 KB）远超 shared 预算，按 EpilogueTile 分批转换/写回，StagesC/StagesD 让 C 预取与 D 写回流水；mainloop 结束后 shared union 转给 epilogue 用（§4）。数字账：每 block D 写回 bf16 $128 \times 128 \times 2 = 32$ KB，全 kernel C/D 写 $4096^{2} \times 2 = 33.55$ MB（4.3 recap，占 tile 流量低阶）。**fusion 的带宽节省账**（中间 tensor 免 HBM 往返， $(k-1)\times$ bytes）归 5.4，本章只走读注入点；GEMM epilogue 是 5.4 三类融合策略里「GEMM epilogue」一类的代码实体。

## 8. tile 层级流量账与 MMA 利用率

把三层流量并排（走读配置，每 block 每 K tile；4.3 §4 层级账的代码侧实例化）：

| 层级 | 每 tile 流量 | 复用结构 | 强度口径 |
| --- | --- | --- | --- |
| HBM→shared（TMA） | 32768 B（无 L2）；有效 ÷11.49 ≈ 2852 B | A× $T_N$、B× $T_M$（+L2 波前） | 64 → 735（命题 1） |
| shared→TC（描述符） | 49152 B（B 读两遍） | 指令内 A×128、B×64 | 96 B/cycle = 峰值 75% |
| register（累加器读写） | 524288 B（8 × 65536） | k 方向累加，D 驻留 | 1.3 §4 口径 |

利用率定义： $U = F_{\mathrm{ach}}/F_{\mathrm{peak}} = 2MNK/(F_{\mathrm{peak}} \cdot t_{\mathrm{kernel}})$——achieved 对 peak 的比值，实测口径归 11.2（Nsight SpeedOfLight）。本章给可逐项核对的分解。

**命题 3**（MMA 利用率分解）。波次串行、每波取最慢 block 的教学口径下， $U = U_{\mathrm{geom}} \cdot U_{\mathrm{wave}} \cdot U_{\mathrm{main}} \cdot U_{\mathrm{epi}}$，其中 $U_{\mathrm{geom}} = MN/(n_{\mathrm{blk}} T_M T_N)$（形状填充效率）、 $U_{\mathrm{wave}} = n_{\mathrm{blk}}/(n_{\mathrm{wave}} n_{\mathrm{SM}})$（波次量化）、 $U_{\mathrm{main}} = t_{\mathrm{comp}}/\tau_S$（mainloop TC 忙碌占比）、 $U_{\mathrm{epi}} = t_{\mathrm{main}}/(t_{\mathrm{main}} + t_{\mathrm{epi}})$（epilogue 占比）。**符号**： $t_{\mathrm{main}} = (K/T_K)\tau_S$ = mainloop 时间、 $t_{\mathrm{kernel}}$ = kernel 总时间、 $F_{\mathrm{ach}}$ = achieved 算力（2.4 口径）。**证明**（路线：时间账代入 + 比值链 telescoping，四步）：写 $t_{\mathrm{kernel}} = n_{\mathrm{wave}}(t_{\mathrm{main}} + t_{\mathrm{epi}})$、 $t_{\mathrm{main}} = (K/T_K)\tau_S$、每 block TC 忙碌 $(K/T_K)t_{\mathrm{comp}}$，代入 $U$ 定义式与 $F_{\mathrm{peak}} = n_{\mathrm{SM}} F_{\mathrm{SM}}$，四个比值逐项相消即得。完整证明留作练习 3。∎

全数字代入（教学口径）： $U_{\mathrm{geom}} = 1$（4096 被 128 整除，无 padding）； $n_{\mathrm{blk}} = 32 \times 32 = 1024$， $n_{\mathrm{wave}} = \lceil 1024/132 \rceil = 8$， $U_{\mathrm{wave}} = 1024/1056 = 0.9697$——末波只有 100 个 block，32 个 SM 空转； $U_{\mathrm{main}} = 279.8/279.8 = 1$（L2 修正后计算主导，§5）； $U_{\mathrm{epi}}$ 取教学假设 $t_{\mathrm{epi}} = 0.2\thinspace t_{\mathrm{main}}$（epilogue 与 mainloop 在 base kernel 串行，§4 union 结构），得 $1/1.2 = 0.833$。合计 $U \approx 0.808$， $F_{\mathrm{ach}} \approx 989.5 \times 0.808 = 799.6$ TFLOPS。对照无 L2 口径： $U_{\mathrm{main}} = 279.8/1290.4 = 0.217$（装载主导，TC 每 tile 只忙 21.7%），合计 $U \approx 0.175$、约 173 TFLOPS——**L2 复用一项就值 4.6 倍利用率**。这个 4.61 恰是 $R/I_{\mathrm{blk}} = 295/64$（roofline 缺口，4.3 recap）：L2 因子 11.49 只需越过缺口 4.61 就把 kernel 从斜线区推进平台区，超出部分不再抬利用率（平台封顶）。这给「GEMM 为什么必须调 raster/swizzle」一个量化答案：不是百分之几的打磨，是平台区与斜线区的差别。**公开数值锚**：DeepGEMM README 逐字「DeepGEMM now achieves up to 1550 TFLOPS on H800」（FP8，[DeepSeek-AI, DeepGEMM](../references.md)）；对 FP8 dense 峰值 1979 TFLOPS（H100 SXM datasheet 口径；教学口径：H800 为 H100 的 NVLink 缩减 variant，算力峰值取同值，微基准口径见 [Luo et al., 2024](../references.md)）得 $1550/1979 \approx 0.783$（78.3%）——与分解模型的 80.8% 同量级。模型给的是结构上限的逐项损失，实测偏差（调度抖动、L2 逐出、epilogue 实际占比）归 11.2 drill-down。

分解与实测的对照路径：Nsight Compute SpeedOfLight 的 compute/memory 吞吐百分比（2.4 recap）对应 roofline 位置；tensor pipe 活跃周期占比对应 $U_{\mathrm{main}}$；achieved FLOPs（指令计数 × 几何，或 FLOPs 计数器除时间）除 peak 给 $U$。实测 $U$ 显著低于分解值时逐项 drill-down： $U_{\mathrm{wave}}$ 低看尾波与 block 数（§9 split-K 取舍）， $U_{\mathrm{main}}$ 低看 stall reason（long_scoreboard = 等 shared/global 依赖、barrier = 等流水交接，1.4 recap）， $U_{\mathrm{epi}}$ 低看 epilogue 占比与 pingpong 是否启用。完整测量流程归 11.2，本章给因子到指标的映射。

## 9. tile 选型工程：split-K、cuBLASLt heuristic 与实例化

可行 tile 配置里怎么选点？理论侧（三预算约束、方阵最优）4.3 §5 已交付，工程侧有三个选择机制，各付各的成本。

**split-K**： $M, N$ 小而 $K$ 大时， $n_{\mathrm{blk}} = (M/T_M)(N/T_N)$ 喂不满 SM——极端例 $M = N = 128$、 $K = 4096$ 只有 1 个 block，132 个 SM 空 131 个。split-K 把 K 维切 $n_{\mathrm{split}}$ 段、各段独立 block 算部分和，末了归约（5.1 §4 grid 归约结构：atomic 或独立 kernel）。代价是 partial 流量：fp32 partial 写+读 $2(n_{\mathrm{split}}-1) MN \times 4$ B（教学口径）。代入 $M = N = 4096$、 $n_{\mathrm{split}} = 4$： $2 \times 3 \times 16777216 \times 4 = 402.7$ MB——**反超 tile 流量 186.9 MB，大 MN 下 split-K 纯亏**；它只在 block 数不足时换并行性。efficient_gemm.md 的大 tile 形状失配段（4.3 已引）给的正是这条取舍。

**cuBLASLt heuristic**：闭源库的选择接口（[NVIDIA, cuBLASLt](../references.md)，[C]）。`cublasLtMatmulAlgoGetHeuristic` 按问题形状返回候选 algo 配置，官方逐字：「The output is placed in heuristicResultsArray[] in the order of increasing estimated compute time.」——**按预估算力时间排序，不是实测**；heuristic 是预计算决策表（问题形状 → 候选 tile/stages/splitK 组合），查询一次可复用。复用机制官方逐字：「cuBLASLt implements a heuristics cache that maps matmul problems to kernels previously selected by heuristics. The heuristics cache uses an LRU-like eviction policy and is thread-safe.」，默认容量 8192 条、每条至多 1024 B（`CUBLASLT_HEURISTICS_CACHE_CAPACITY` 可调）。要实测最优则走 algo 枚举 + 逐一试跑（autotune 口径）；heuristic 省的是选择成本，付的是预估偏差。

**CUTLASS 侧对应物**：tile/stages/cluster/schedule 全是编译期模板参数，候选配置经 manifest 枚举实例化（profiler 逐一试跑 = 开源 autotune）；运行期选择交给 TileScheduler 的 raster/swizzle 旋钮（§2）与 cuBLASLt 式 heuristic。实例化的成本结构：每个候选配置是一份独立 kernel 二进制，枚举越多编译时间与二进制体积越大——所以 manifest 按架构/精度/布局族裁剪，不是全笛卡尔积；heuristic 的价值正在于把「运行期选点」的成本压到一次查表。4.3 §5「目标函数与约束在本章、求解器在工程侧」的分工在此兑现：式 (2) 单调升 + 三预算约束给出可行域，heuristic/autotune 在可行域里选点。

## 10. DeepGEMM 对照：FP8 fine-grained scaling

FP8 的 scale 沿 K 维每 128 元素一变，mainloop 怎么容纳一个沿归约维变化的 scale？量化算法与算子语义归 inference 3.2/3.3（recap 指针，不重推）；本节的增量是 **fine-grained scaling 逼出的两级累加**。数学原因先钉：per-block scale 沿 K 维每 128 元素一变（A 侧 1×128、B 侧 128×128），不同 K 段的 scale 不同，**无法折进一个事后全局 scale**——必须分段累加：每段在 Tensor Core 里无 scale 累加，段末乘 $`\mathrm{scale}_a \times \mathrm{scale}_b`$ 提升到 CUDA Core 的 fp32 主累加器，TC 累加器清零重起。

DeepGEMM（commit 5081f8513af5）的 SM90 实现（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`）：注释逐字「Accumulation for WGMMA or CUDA promotion」（L251），两级累加器声明 `float accum[WGMMA::kNumAccum], final_accum[...]`（L254）；`BLOCK_K == 128` 硬断言（L56，「Only support per-128-channel FP8 scaling」）；「Promote with scales」段（L329-345）：`scale_0_0 = scale_a_0 * scale_b_0`（L331），`shifted_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0]`（L341-344）——每 128 K 深把 FP8 WGMMA 部分积乘 scale 提升到 fp32 final_accum。FP8 wgmma 几何 m64nNk32（kind::f8f6f4，1.3 recap），128 K 深 = 4 条 k32 指令一段。

提升间隔是精度与开销的权衡：间隔越长，promotion 次数越少、CUDA Core 乘法加法开销越低，但 TC 内累加链越深——fp32 累加的舍入随链深累积（5.1 §4 精度账同构：误差系数随深度增长）；间隔越短则反之。CUTLASS API 把间隔交给用户指定（下文逐字「max number of MMA interval specified by user」），DeepGEMM 取 128 K 深，与 fine-grained scale 的粒度对齐——scale 每 128 元素一变，promotion 边界必须落在 scale 边界上，间隔没有自由选择的余地，这是「scaling 粒度钉死累加结构」的实例。

CUTLASS 侧同概念 API（`gemm/collective/fp8_accumulation.hpp` L40-43 注释逐字）：「This class provides API to promote (add) or scale (multiply_add) the results from the tensor core accumulators to the main accumulators when the number of MMAs reaches the max number of MMA interval specified by user, after that the tensor core accumulators are zeroed.」——提升间隔用户指定、TC 累加器周期清零，与 DeepGEMM 的 per-128 promotion 同构；现成 mainloop 变体 `sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`（本章 §5 走读的 FP8 blockwise 版）。

两代 scale 格式与编译路径（README 逐字）：「SM90 requires scaling factors in FP32 format.」/「SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single torch.int.」（L67-68）——Blackwell 侧 UE8M0 打包与 tcgen05/TMEM 衔接（1.3 §6 recap），格式位布局账归 6.1。编译路径对照：DeepGEMM「All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation.」（README L3；`csrc/runtime/jit.hpp` L12-32 的 Runtime/init_jit 接口）——运行时 JIT 对 CUTLASS 编译期模板实例化（§9），前者免安装期 nvcc、形状特化延迟到运行时，后者实例化成本前置、运行期零编译。Marlin 的 W4A16 dequant 融合与 DeepGEMM 的指令级片段走读归 6.2（本章不重复全量走读）。

## 11. 边界：本章不做什么

走读收口，剩下的归谁？本章交付 L2 波前修正（命题 1 + §2）、五层实体与 device/kernel 走读（§3/§4）、collective mainloop 与 pipeline 实例化（§5）、CuTe layout/MMA atom 与 wgmma 计数（命题 2 + §6）、epilogue fusion 注入点（§7）、层级流量账与利用率分解（命题 3 + §8）、split-K/cuBLASLt/实例化（§9）、DeepGEMM 两级累加对照（§10）；以下各有专章。量化 kernel 片段（Marlin lop3/dequant、DeepGEMM promotion 逐行）归 6.2；FA mainloop 同构走读（Q/K/V 三份 tile、online softmax 融合）归 6.3；格式位布局与逐格式吞吐归 6.1；fusion 带宽节省账归 5.4；Blackwell tcgen05 全量走读归 6.3（FA4）与后续版本；实测利用率 drill-down 归 11.2。代码 scope：模块走读级（CUTLASS SM90 五层 + DeepGEMM 对照片段）。

本章交付清单（下游 recap 接口）：命题 1（§2）、五层实体图谱（§3）、warp specialization 角色与 occupancy（§4）、mainloop load/mma + pipeline 账（§5）、命题 2 + CuTe 三级（§6）、epilogue 注入点（§7）、命题 3 + 流量表（§8）、split-K/heuristic（§9）、两级累加（§10）。6.2 recap §5 mainloop 与 §10 promotion 作量化 kernel 的底座；6.3 recap §4/§5 作 FA3 warp specialization 的同构原型；5.4 recap §7 作 GEMM epilogue fusion 的代码实体。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 4.3 recap： $I_{\mathrm{blk}} = 2T_M T_N/((T_M+T_N)\mathrm{bytes}) = 64$、流量 $MNK(1/T_M+1/T_N)\mathrm{bytes} = 2.147$ GB（无 L2 口径），不重推 | 4.3 式 (2)(3) | §2、练习 1 |
| L2 波前复用因子 $2b/(r+c)$，方阵波前 $\sqrt{b}$； $b = 132$ 给 11.49（行优先 7.31）；有效强度 735 越过 ridge 295，bound 翻转 | 命题 1 + §2 表 | §2、练习 2 |
| 工作集核验：波前 panel ≈ 24.1 MB 低于 L2 50 MB；raster/swizzle 是波前形状旋钮（sm90_tile_scheduler.hpp） | §2 核验段 | §2、练习 2 |
| 五层实体：device gemm_universal.h L142/L289/L398 → kernel L65/operator() L269 → collective L72 → CuTe L211/L460 → wgmma；2.x 目录对照 | §3 图谱 | §3、练习 4 |
| warp specialization：1 load WG + 1 MMA WG = 256 线程，producer warp0 elected 发 TMA；角色 L317-327、transaction_bytes L326 | §4 走读 | §4、练习 4 |
| occupancy：shared 128 KB → 1 block/SM 绑定， $\mathcal{O} = 0.125$；compute-bound 低 occupancy 靠异步流水（3.4 命题 2） | §4 账 | §4、练习 1 |
| MainloopPipeline = PipelineTmaAsync（L111）；TmaTransactionBytes = 32768 B/stage（L265-269）；load 2 条 TMA op（L384-385） | §5 走读 | §5、练习 1/4 |
| release 滞后：`warpgroup_wait<1>`（L547）先于 `consumer_release`（L551），wgmma 异步读 shared 不可提前覆写 | §5 时序 | §5、练习 4 |
| pipeline 翻转：无 L2 τ=1290.4 ns 装载主导；L2 修正 τ=279.8 ns 计算主导，S_min=3、Stages=4 留余量 | §5 表（4.4 命题 1 recap） | §5、练习 2 |
| cluster multicast：A÷ $C_N$、B÷ $C_M$；cluster 2×1 给 1.610 GB（省 25%）；DSMEM 机制 | §5 multicast 段 | §5 |
| CuTe layout = (shape,stride) 代数；MMA_Atom→TiledMMA→ThrMMA（L49/211/460）；SS fragment 是描述符 | §6 | §6、练习 4 |
| 每 tile wgmma = $(T_M/64)(T_N/128)(T_K/16)$ = 8 条；512 cycles = 279.8 ns 与 t_comp 闭环；累加器 128 regs/thread | 命题 2 + 1.3 recap | §6、练习 1/3 |
| epilogue：EpilogueTile 子块（L68/L128-130）+ ThreadEpilogueOp fusion 注入（L122）；shared union 转用；D 写 33.55 MB | §7 | §7 |
| 利用率分解 $U = U_{\mathrm{geom}} U_{\mathrm{wave}} U_{\mathrm{main}} U_{\mathrm{epi}}$；代入 1×0.9697×1×0.833 ≈ 0.808（约 800 TFLOPS）；DeepGEMM 1550/1979 = 78.3% 公开锚 | 命题 3 + §8 | §8、练习 3 |
| split-K：partial 流量 $2(n_{\mathrm{split}}-1)MN \times 4$ B；大 MN 反超（402.7 MB），只救 block 不足 | §9 | §9、练习 3 |
| cuBLASLt heuristic 按 estimated compute time 排序（逐字）+ LRU cache 8192 条；CUTLASS 编译期实例化对 DeepJIT 运行时 | §9/§10 逐字 | §9 |
| DeepGEMM 两级累加：per-128 promotion（L329-345）+ CUTLASS fp8_accumulation L40-43 同构；SM90 FP32 scale / SM100 packed UE8M0 | §10 | §10 |

## 练习

1.（计算 · [10]）走读配置代入（TileShape 128×128×64、Stages=4、bf16）：(a) 每 stage TMA 字节（A/B 分列）与 Stages 份 shared 总字节；(b) $I_{\mathrm{blk}}$（4.3 式 (2)）；(c) 每 K tile 的 wgmma 条数与 FLOPs 闭合；(d) shared 项 blocks 与 occupancy。校验：(a) 16384+16384 = 32768 B、128 KB（§5）；(b) 64（§2）；(c) 8 条、2097152 = 2×128×128×64（命题 2）；(d) 1 block、0.125（§4）。
2.（计算 · [20]）L2 与 pipeline 账（ $b = 132$）：(a) 行优先波前（ $c = 32$）与方阵波前的复用因子、有效 HBM 流量、有效强度；(b) 加 C 写后的带宽口径时间，与算力地板 138.9 μs 比较判 bound；(c) L2 修正口径的 $t_{\mathrm{load}}$、 $\tau_4$ 与 $S_{\min}$；(d) 先预测 cluster 2×1 对 HBM 流量的方向再按 §5 公式算。校验：(a) 7.31/11.49、293.8/186.9 MB、467.7/735.3（§2 表）；(b) 65.8 μs 低于 138.9 μs → compute-bound；(c) 112.3 ns、279.8 ns、3（§5 表）；(d) 省 25%、1.610 GB（§5）。
3.（推导 · [30]）补全命题 3 的证明（利用率分解）：(a) 写出时间账 $t_{\mathrm{kernel}} = n_{\mathrm{wave}}(t_{\mathrm{main}}+t_{\mathrm{epi}})$、 $t_{\mathrm{main}} = (K/T_K)\tau_S$ 与每 block TC 忙碌时间，代入 $U = 2MNK/(F_{\mathrm{peak}} t_{\mathrm{kernel}})$ 推出四因子乘积；(b) 用本章数值逐因子核对（含 $U_{\mathrm{wave}}$ 的末波 block 数）；(c) split-K 扩展： $n_{\mathrm{split}} = 4$ 时 $U_{\mathrm{wave}}$ 与 partial 流量各怎么变，说明大 MN 下为何不启用；(d) 指出分解的教学假设（波次串行、epilogue 不重叠）在 cooperative/persistent 变体下哪条失效。校验：(a) 四因子 telescoping（§8 路线）；(b) 0.9697 = 1024/1056、U ≈ 0.808；(c) 402.7 MB 反超 186.9 MB（§9）；(d) epilogue 与 mainloop 重叠（§4 pingpong）。
4.（代码 · [30]）CUTLASS 4.9.0 mainloop 走读（版本锚 5c179edeb3e2）：(a) 在 `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 找出 `K_PIPE_MMAS` 与 `K_PIPE_MAX` 的值，解释 consumer_release 为何必须在 `warpgroup_wait<K_PIPE_MMAS>` 之后；(b) 说明首个 k block 的 `ScaleOut::Zero` 与后续 `ScaleOut::One` 各管什么；(c) 若 Stages=2，`static_assert (0 <= K_PIPE_MMAS) && (K_PIPE_MMAS < K_PIPE_MAX)` 是否满足、流水退化为几份预取；(d) 指出 `PipelineTmaUmmaAsync` 的实际文件与架构归属（漂移注）。校验：(a) L263-264（1 与 Stages）、L547/L551（§5）；(b) 累加器清零起跑/累加（§5）；(c) 满足、1 份预取（4.4 双缓冲口径）；(d) sm100_pipeline.hpp L538、Blackwell UMMA（§1 版本锚）。

## 参考

- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，版本锚 4.9.0（commit 5c179edeb3e2，2026-11-25 fresh tarball 核实）——gemm/device/gemm_universal.h（GemmUniversal L142/L289、GemmKernel L355、initialize L398）；gemm/kernel/gemm_universal.hpp L42-57（IsCutlass3ArrayKernel 分流）；gemm/kernel/sm90_gemm_tma_warpspecialized.hpp（class L65、NumLoad/NumMmaWarpGroups L140-141、MaxThreadsPerBlock L142、SharedStorage union L120-138、operator() L269、roles L317-327、transaction_bytes L326、collective_epilogue L392、mainloop.load L436/load_tail L450、epilogue.load L455、partition_fragment_C L470、mma L472、epilogue.store L496）；cooperative 变体（Load/MmaRegisterRequirement L150-151、reg_dealloc L562/reg_alloc L772）与 pingpong 变体；gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp（CollectiveMma L72、DispatchPolicy L92、MainloopPipeline=PipelineTmaAsync L111、SmemLayout L130-134、Stages≥2 L137、smem_desc assert L138-140、K_PIPE_MAX/MMAS L263-264、TmaTransactionBytes L265-269、load L310/producer_acquire L374/TMA copy L384-385/load_tail L396、mma L417/consumer_wait L487/warpgroup_arrive L490/cute::gemm L496/commit L500/wait L547/release L551/mma_tail L563）；fp8_accumulation.hpp L40-43（promote/scale API 逐字）+ fp8_blockwise_scaling mainloop 变体；epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp（CollectiveEpilogue L83、EpilogueTile L68、ThreadEpilogueOp L122、整除 assert L128-130）；epilogue/fusion/operations.hpp L143（LinCombEltAct）/L229 注释·L240 struct（LinCombPerColBiasEltAct，bias+activation 注释逐字）；cute/layout.hpp（Shape L48/Stride L51/Layout L99）；cute/atom/mma_atom.hpp（MMA_Atom L49/TiledMMA L211/ThrMMA L460/get_slice L359/partition_A L478/make_tiled_mma L531）；cute/arch/mma_sm90_gmma.hpp（m64n8k16 L158/m64n16k16 L249/m64n32k16 L341/m64n64k16 L438/bf16 m64n128k16 L2842）+ mma_traits_sm90_gmma.hpp L1716（SS atom 别名）；cute/swizzle.hpp L52；pipeline/sm90_pipeline.hpp（PipelineState L171/PipelineTmaAsync L289/full-empty L299-300/producer_acquire L443）；pipeline/sm100_pipeline.hpp L538（PipelineTmaUmmaAsync，Blackwell）；gemm/kernel/sm90_tile_scheduler.hpp（RasterOrder L47-48/swizzle L105-121）+ sm90_tile_scheduler_group.hpp L134（AlongM 默认）；2.x 层级 gemm/{threadblock,warp,thread}/
- [NVIDIA, CUTLASS docs](../references.md)：CUTLASS media/docs/cpp 文档（key: cutlass-docs），教学参照 + 官方实现，2026-11-25——efficient_gemm.md L16-32（GEMM 层级嵌套循环：cta threadblock-level concurrency → cta_k mainloop/stage → warp-level parallelism → mma instruction-level parallelism → mma_instruction；大 tile 形状失配与 split-K 段，4.3 已引）
- [NVIDIA, cuBLASLt](../references.md)：cuBLAS/cuBLASLt docs（key: cublaslt-docs），[C] 官方规格披露·一手权威，2026-11-25 再核——cublasLtMatmulAlgoGetHeuristic（「The output is placed in heuristicResultsArray[] in the order of increasing estimated compute time.」逐字）；heuristics cache（「LRU-like eviction policy and is thread-safe」逐字 + 默认 8192 条/每条至多 1024 B + CUBLASLT_HEURISTICS_CACHE_CAPACITY）
- [DeepSeek-AI, DeepGEMM](../references.md)：deepseek-ai/DeepGEMM repo（key: deepgemm），教学参照 + 官方实现，版本锚 commit 5081f8513af5（2026-11-25）——sm90_fp8_gemm_1d2d.cuh（「Accumulation for WGMMA or CUDA promotion」L251 + accum/final_accum L254 + BLOCK_K==128 断言 L56 + 「Promote with scales」L329-345：scale_0_0 L331/shifted_accum L341-344）；README（DeepJIT L3 逐字 + 1550 TFLOPS H800 L25 逐字 + SM90 FP32/SM100 packed UE8M0 L67-68 逐字）；csrc/runtime/jit.hpp L12-32；docs/scaling-factor-format.md（gran_k SM90 仅 128 L29）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——wgmma.mma_async m64nNk16（bf16/f16）/ m64nNk32（kind::f8f6f4）+ shared 描述符 + commit/wait（1.3 recap 源）；cp.async.bulk.tensor.multicast::cluster（§5 multicast）；TMA swizzle 模式（4.2 recap 源）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——wgmma m64n128k16 dense 延迟 128 cycles、HBM 478.8 cycles（H800 微基准实测口径，§5/§6 延迟输入；H800 为 H100 NVLink 缩减 variant 的教学口径出处）
