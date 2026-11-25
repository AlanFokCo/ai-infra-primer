# 5.4 算子融合原理（Operator Fusion）

> 一条 $k = 4$ 的逐元素算子链作用在 prefill 的 33.55 MB 激活上，未融合要搬 $8 \times 33.55 = 268.44$ MB、融合后只剩首读尾写 67.11 MB——省 201.33 MB、60.06 μs，全部来自 $(k-1)$ 个中间 tensor 免落地 HBM；但融合把中间值压进 register，寄存器压力涨、occupancy 掉，不是无条件免费。本章交付融合带宽节省账（命题 1： $k$ 个单输入单输出逐元素算子省 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$）、融合的 register/shared 预算约束（命题 2）、三类融合策略（逐元素链 / reduction+逐元素 / GEMM epilogue）、FlashAttention 的 fusion 视角（recap models 4.1 IO 定理，6.3 回链）、torch.compile/Inductor 与 Triton 的自动化（recap inference 3.4）与 fusion 限制、NanoFlow 设备内重叠 | 前置：5.3（norm/softmax 行驻留与带宽下界、reduction+逐元素融合形态）、5.2 §7（GEMM epilogue fusion 代码实体）、3.4（occupancy 四约束、register pressure）、2.4（roofline）、4.3（tiling 复用）；下游 6.3（FA tiling+online softmax 融合全量走读）、6.2（dequant 融合进 GEMM mainloop） | 锚点：Triton（[triton-lang, Triton](../references.md)，教学参照 + 官方实现）、PyTorch Inductor（[PyTorch Team, PyTorch](../references.md)，教学参照）、NanoFlow（[Zhu et al., 2024](../references.md)，arXiv:2408.12757，[A]）、TileLang（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）、TransformerEngine（[NVIDIA, TransformerEngine](../references.md)，教学参照）、CUTLASS epilogue（[NVIDIA, CUTLASS](../references.md)，教学参照）、FlashAttention（[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]）、CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）——代码锚 2026-11-25 一手核实

## 1. 中间 tensor 的往返税：三个问题与本章口径

decode 一步要 launch 几百个 kernel，其中大量是逐元素小算子——残差加、激活、量化 cast、norm 的缩放平移。每个独立 kernel 都要把输入从 HBM 读进来、把输出写回 HBM；可这些算子的输出往往立刻被下一个算子消费，中间结果在 HBM 里只是「过路」。一条 $k = 4$ 的逐元素链，未融合时每个算子一读一写，中间三个 tensor 各被写一次读一次——白搬 $6$ 遍 tensor 字节。融合把它们并进一个 kernel，中间值留在 register，HBM 只剩链首读与链尾写。本章回答三个问题：融合到底省多少 HBM 字节，省的那笔怎么数（命题 1）？融合把中间值压进片上，register/shared/occupancy 各付什么代价（命题 2）？三类融合策略各融什么，torch.compile 与 Triton 怎么自动化，FlashAttention 为什么是最大的融合先例？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 5.2/5.3）； $\mathrm{BW} = 3352$ GB/s、bf16 ridge $R \approx 295$（2.4 recap）。running example 沿 inference 3.4 的 8B Llama 形态（ $d_{\mathrm{model}} = 4096$、bf16）：decode $B = 64$（每序列 1 token，激活 tensor $n_{\mathrm{el}} = 64 \times 4096 = 262144$ 元素、 $0.5$ MiB）与 prefill $T = 4096$（ $n_{\mathrm{el}} = 4096 \times 4096 = 2^{24}$ 元素、33.55 MB）两档——**换口径原因**：展示小 tensor（decode，launch 主导）与大 tensor（prefill，IO 主导）下融合收益的主次对调（inference 3.4 §2 结论 recap）。逐元素链算子数记 $k$（**本章 $k$ = 融合链算子数，非 rank $n$、非 GEMM 归约维 $K$**）。

符号声明（spec §4.3）： $k$ = 融合链上算子数； $n_{\mathrm{el}}$ = 单 tensor 元素数（5.1/5.3 recap， $n$ 专归 rank 数）； $\mathrm{bytes}$ = 每元素字节（bf16 取 2，bytes 不占字母）； $d_{\mathrm{model}}$ = hidden 宽（models $d$ 翻译）； $B$ = batch、 $T$ = 序列长（recap inference）； $R_{\mathrm{th}}$ = registers/thread、 $S_{\mathrm{sh}}$ = shared bytes/block、 $\mathcal{O}$ = occupancy（3.4 recap，**混场声明： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread**）； $I$ = 算术强度； $S$ = 单 tensor 字节（§2/练习 3，沿 inference 3.4 语义；**混场声明**：§3 的 $S/P$ 为 FA score/概率矩阵，沿 models 4.1 记号，两义分局部、随段声明）； $S_{\mathrm{ff}}$ = FFN 中间 tensor 字节（§2）； $\tau_c, \tau_m, \tau_n$ = NanoFlow compute/memory/network 段时间（recap inference 3.4 式 (4)）； $w$ = 融合链每线程同时驻留的中间值个数（§4 局部）。B/KB/MB/MiB 是单位符号豁免。

**边界声明**：融合 IO 节省的算法级账 inference 3.4 命题 1 已交付（ $2(n-1)S$，本章 $n \to k$ 翻译、按 spec 关键推导重列计数并实例化到 kernel 侧）；本章增量 = **register/shared 预算约束（命题 2）+ 三类融合策略的 kernel 实体 + Triton block-level 编程**。torch.compile/Inductor 的 Dynamo/guards/graph break 机制与 NanoFlow nano-batch 流水 recap inference 3.4（§4/§7），本章只补 kernel/codegen 侧读法。FlashAttention 的 IO 定理 recap models 4.1 §5.4（不重推），FA 全量指令级走读归 6.3；GEMM epilogue 代码实体 recap 5.2 §7；norm kernel 行驻留 recap 5.3。代码 scope：机制片段级（§5 Triton 教学 kernel），不做 Inductor/Triton 编译器模块全量走读。

## 2. 融合带宽节省账：(k−1) 个中间 tensor 免落地

先回答一个更基本的问题：融合省的是哪一笔字节？不是输入读、不是输出写——这两笔无论融不融合都得付；省的是**中间 tensor 的写 + 读**。所谓算子融合（operator fusion），是指把多个算子合并进一个 kernel、中间结果留在 register 或 shared memory 而不写回 HBM。

**命题 1**（逐元素链融合 IO 节省）。 $k$ 个单输入单输出逐元素算子（输出元素只依赖对应输入元素）作用在同 shape 的 $n_{\mathrm{el}}$ 元素 tensor 上，未融合 HBM 流量 $2k \cdot n_{\mathrm{el}} \cdot \mathrm{bytes}$，融合后 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$，节省 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$。**证明**（读写计数，三步；inference 3.4 命题 1 的算法级账同此， $n \to k$ 翻译）。未融合：每个算子读输入 $n_{\mathrm{el}} \cdot \mathrm{bytes}$、写输出 $n_{\mathrm{el}} \cdot \mathrm{bytes}$， $k$ 个算子合计 $2k \cdot n_{\mathrm{el}} \cdot \mathrm{bytes}$。融合：中间结果驻留片上，只有链首输入读与链尾输出写落 HBM，合计 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$。作差得 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$——结构上恰是 $(k-1)$ 个中间 tensor，每个省一次写 + 一次读（每元素 $2\mathrm{bytes}$）。∎

全数字代入（bf16， $\mathrm{bytes} = 2$）：

| 口径 | $n_{\mathrm{el}}$ | 单 tensor | $k=2$ 省 | $k=4$ 省 | $k=4$ 省时 |
| --- | --- | --- | --- | --- | --- |
| decode $B=64$ | 262144 | 0.5 MiB | 1.05 MB | 3.15 MB | 0.94 μs |
| prefill $T=4096$ | $2^{24}$ | 33.55 MB | 67.11 MB | 201.33 MB | 60.06 μs |

读法（inference 3.4 §2 结论 recap）：decode 的小 tensor 上 IO 项很小（ $k=4$ 才省 0.94 μs），**大的是 launch 项**——融合每省一个 kernel 就省一次 CPU 提交（约 5 μs/kernel，inference 3.4 式 (2)）， $k=4$ 链省 3 次 launch 约 15 μs，是 IO 项的 16 倍；prefill 的大 tensor 上 IO 项主导（60.06 μs）。判断一个融合值不值，先问 tensor 多大、链多长。

多输入算子的账稍有不同（inference 3.4 §2 recap，不重推）：SiLU+mul（SwiGLU，mul 有两个输入）未融合 $5 S_{\mathrm{ff}}$、融合 $3 S_{\mathrm{ff}}$、省 $2 S_{\mathrm{ff}}$；residual+norm 未融合 $5S$、融合 $4S$、省 $S$（ $S$ 为对应 tensor 字节）。**节省的通用数法是「未融合读写总数 − 融合读写总数」**，单输入单输出链是其中最规整的一类。

口径注与边界反例（inference 3.4 命题 1 口径 recap）：命题 1 是 HBM 流量**上界账**——中间 tensor 若部分命中 L2，未融合的实际 HBM 流量已低于 $2k \cdot n_{\mathrm{el}} \cdot \mathrm{bytes}$，节省相应缩水（L2 50 MB 装得下 decode 的小 tensor，所以 decode 的 IO 节省常被 L2 吃掉一部分，launch 项才是确定收益）。前提是被融合的中间结果不被链外算子复用。**边界反例（fan-out）**：中间结果被两个下游算子消费时，融进其中一个分支会迫使另一分支重算或重新物化中间结果——融合不是无条件免费。一般 DAG 的融合节省与 fan-out 取舍留作练习 3。

## 3. 三类融合策略：各融什么

融合只有逐元素链一种吗？不是——三类结构不同的合并，各融不同的东西、各驻留不同的片上层。先给分类，再各配 kernel 实体。

```
三类融合策略（中间值驻留层 + 代表实体）：
① 逐元素链      pointwise→pointwise→...   中间值在 register
   代表：residual+ReLU、SiLU+mul、量化 cast 链；Inductor/Triton 自动生成
② reduction+逐元素  归约→normalize→cast    整行驻留 register/shared
   代表：LayerNorm/RMSNorm+γ/β+quant（5.3 TE kernel）；行驻留是前提
③ GEMM epilogue   mainloop 累加器→bias/act/quant→store  累加器在 register
   代表：CUTLASS ThreadEpilogueOp（5.2 §7）；D=act(α·acc+β·C+bias) 一趟写回
```

**① 逐元素链**。最规整的一类：链上每个算子输出元素只依赖对应输入元素，无跨元素依赖，所以整条链可以逐元素并行、中间值全程在 register。命题 1 直接适用。Inductor 的 pointwise fusion pass 与 Triton 的手写融合 kernel（§5）都属此类。残差加 + ReLU、SiLU + mul（SwiGLU 激活）、FP8 cast 链是生产实例（inference 3.4 §2 的 TileKernels fused SwiGLU+quant）。

**② reduction + 逐元素**。归约算子（norm 的 mean/var、softmax 的 max/sum）后接逐元素链（normalize、 $\gamma/\beta$、量化 cast）。这类融合的前提是**行驻留**（5.3 命题 4）：整行装进 register/shared 后，归约与下游逐元素算子都能就地完成，中间值不落地。5.3 §4 走读的 TransformerEngine norm kernel 正是此类——Welford 归约 + normalize + $\gamma/\beta$ + FP8 scale/amax 在一个 kernel 里（`ln_fwd_kernels.cuh` L100-146），`rmsnorm_fwd_mxfp8.cu` 专做 RMSNorm + MXFP8 cast。归约类算子天生 memory-bound（5.3），融合下游逐元素算子省的是 normalize 输出的那一趟 HBM 往返。

**③ GEMM epilogue**。GEMM mainloop 算完，累加器还在 register，epilogue 把 bias/激活/量化 scale 接在写回路径上（5.2 §7 recap，不重走读）：CUTLASS 的 `ThreadEpilogueOp = FusionCallbacksTraits::Operation`（`sm90_epilogue_tma_warpspecialized.hpp` L122）注入逐 fragment 的 visitor 树，`LinCombPerColBiasEltAct`（operations.hpp L240）一趟完成 $D = \mathrm{activation}(\alpha \cdot \mathrm{acc} + \beta \cdot C + \mathrm{bias})$。中间值全在 register/shared，没有一个中间 tensor 落 HBM——若 bias/激活拆成独立 kernel，每个都要一趟「读 D + 写 D」往返（ $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$/趟， $n_{\mathrm{el}}$ = GEMM 输出元素数，5.2 口径；D/C 为该章 GEMM 输出/输入矩阵记号）。GEMM epilogue 融合的特殊性：它融进的是 **compute-bound** 主 kernel 的尾巴，省 IO 的同时不额外占 mainloop 的 Tensor Core 时间（epilogue 走 CUDA Core，5.2 §7）。dequant 融合进 GEMM mainloop（Marlin W4A16）是更激进的一类，归 6.2。

**FlashAttention：最大的融合先例**（fusion 视角，recap models 4.1 IO 定理不重推）。朴素 attention 把 $Q K^{\top}$ 的 score 矩阵 $S$ 与概率矩阵 $P$（此处 $S/P$ 为 FA 矩阵记号，沿 models 4.1，别于 §2 的 tensor 字节 $S$）各物化一份 $[H, T, T]$ 落 HBM，softmax 与 $PV$ 再读回—— $T^{2}$ 对象被写出读回共四趟（models 4.1 §5.1）。FlashAttention 把 tiling + online softmax + rescale 融进一个 kernel： $S, P$ 从生到死都在 SRAM，HBM 只读写 $Q, K, V, O$ 本身，IO 复杂度从 naive 的 $\Theta(T^{2})$ 降到 $O(T^{2} d_{\mathrm{head}}^{2}/M_{\mathrm{SRAM}})$（models 4.1 式 (17) recap， $d \to d_{\mathrm{head}}$、 $M \to M_{\mathrm{SRAM}}$ 翻译， $M_{\mathrm{SRAM}}$ = 可用 SRAM 元素计、非消息字节，推导见该章 §5.4，本章不重推）。这是融合原理的同一笔账——中间 tensor 免 HBM 往返——只是省的量级是 $T^{2}$ 而非小 tensor。FA 的 tiling + online softmax 指令级走读（Q outer / KV inner、shared tile 布局、register 累加器、tile↔预算约束）归 6.3；本章只立「FA 是融合」的视角与回链。

## 4. 融合的预算约束：register pressure 与 occupancy 降

融合省了 HBM 字节，代价是什么？中间值不落 HBM，就得驻留片上——register 或 shared。融合链越长、每线程同时持有的中间值越多，register 需求越大。所谓 register pressure（寄存器压力，3.4 recap），是指 kernel 的 register 需求超出可分配预算的状态——编译器要么 spill、要么压 occupancy。

**命题 2**（融合 register 预算与 occupancy）。设融合链每线程同时驻留 $w$ 个中间值（每个 fp32 占 1 register），基线 kernel 每线程 $R_0$ registers，则融合后 $R_{\mathrm{th}} \approx R_0 + w$；resident blocks 受 register 约束 $`n_b \le \lfloor R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}) \rfloor`$（3.4 式 (1) register 项）。当 $R_{\mathrm{th}}$ 跨过临界点， $n_b$ 掉一个、occupancy 阶跃下降；若压 $R_{\mathrm{th}}$ 低于需求则 spill 到 local memory（HBM backing）。**证明**（路线：register 计数 + 3.4 约束代入，两步）。计数：链上 $w$ 个中间值同时活跃（下一个算子消费前上一个不能释放），各占 1 register，加基线 $R_0$ 得 $R_{\mathrm{th}}$。约束：3.4 式 (1) 的 register 项 $`n_b \le \lfloor R_{\mathrm{file}}/(R_{\mathrm{th}} \cdot \mathrm{threads}_{\mathrm{block}}) \rfloor`$ 对 $R_{\mathrm{th}}$ 单调递减， $R_{\mathrm{th}}$ 增则 $n_b$ 减；整 block 粒度不等式跨过临界点即掉一个 block（3.4 §3 register 悬崖）。spill 侧：压 $R_{\mathrm{th}}$ 低于 $R_0 + w$，超额中间值落 local memory，带宽掉约 160 倍（1.2 §3 / 2.3 §6 比值 recap）——融合省的 HBM 流量被 spill 流量反噬。∎

全数字代入（3.4 K2 口径，256 threads/block、 $R_{\mathrm{file}} = 65536$）：基线逐元素 kernel $R_0 = 32$ → register 项 $\lfloor 65536/(32 \times 256) \rfloor = 8$ blocks；融合一条 $w = 64$ 的链（如行驻留 norm 每线程持 16 个 fp32 输入 + 中间值）→ $R_{\mathrm{th}} = 96$ → $\lfloor 65536/(96 \times 256) \rfloor = 2$ blocks。**occupancy 从 8 blocks 掉到 2 blocks**（register 绑定，3.4 §3 悬崖同构）。若用 `__launch_bounds__(256, 4)` 强压到 64 regs（3.4 命题 1： $L_{\mathrm{reg}} = \lfloor 65536/(4 \times 256) \rfloor = 64$），换 4 blocks 驻留，但超额的 32 regs/thread 走 spill——值不值由 spill 流量对融合省下的 HBM 流量比。

净收益判据（融合值不值的定量读法）：融合省 HBM 流量 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$（命题 1），代价是 occupancy 降（可能抬 latency-bound 风险，2.4 命题 1）或 spill 流量。**memory-bound kernel 融合净赚**——HBM 是瓶颈，省流量直接折时间，occupancy 边际收益小（5.3 norm/softmax 全 memory-bound，所以 TE 把 norm+quant 融合是稳赚）；**compute-bound kernel 融合要权衡**——occupancy 降可能压 ILP/延迟隐藏（3.4 命题 2），GEMM epilogue 融合之所以稳，是因为它不占 mainloop 的 TC 时间、且 epilogue 本身是 memory-bound 尾巴。shared 侧一笔：reduction+逐元素融合的 block 归约 partial 占 shared（5.1 命题 3 的 64 B），行驻留 norm 若用 shared 存行则 $S_{\mathrm{sh}} = n_{\mathrm{row}} \cdot \mathrm{bytes}$，受 3.4 式 (1) shared 项约束——大行（ $d_{\mathrm{model}}$ 大）会压 occupancy，这是 5.3 行驻留条件的预算侧读法。

> **提点**：融合与 tiling 是同一枚硬币的两面——tiling（4.3）把数据搬进快层换复用，fusion 把中间值留在快层免往返；两者都受 register/shared 预算约束（3.4 式 (1)），都在「省慢层流量」与「占快层容量」之间权衡。删掉这条类比，命题 1/2 的推导链依然完整。

> **坑**：为融合把 register 压到 spill。触发条件：长链融合后 $R_{\mathrm{th}}$ 超预算，用 `__launch_bounds__` 或 `-maxrregcount` 强压；现象：中间值 spill 到 local memory（HBM backing），融合省的流量被 spill 流量反超，kernel 反而变慢；规避：融合链长度按 register 预算裁（命题 2），或拆成两段融合各保行驻留，spill 流量用 Nsight Compute 的 local memory 指标核（11.2）。

## 5. 自动化：torch.compile / Inductor 与 Triton block-level

谁来自动决定融合哪些算子？手写融合 kernel（5.3 的 TE norm、§3 的 CUTLASS epilogue）覆盖关键路径，长尾逐元素链交给编译器。torch.compile 的两段分工（inference 3.4 §4 recap，不重推机制）：Dynamo（前端）把 Python 字节码捕获成 FX graph、沿途记录 guards（shape/dtype/常量假设）；Inductor（默认后端）把 FX graph lowering 成融合后的 Triton kernel。**本章 kernel 侧增量**：Inductor 的 fusion pass 在 FX graph 上把相邻的 pointwise 节点（及可融合的 reduction 节点）分组，每组生成一个 Triton kernel——命题 1 的节省由它自动执行，分组边界受 §4 的 register 预算与算子兼容性约束。graph break（数据依赖控制流或不支持 op）切断图、断点回退 eager，融合也就断在 break 处；dynamic shape 用符号维 + guards，guard 失效触发 recompile（inference 3.4 §4 recap）。vLLM 的 VLLM_COMPILE 在此之上加 caching/piecewise/shape specialization/custom passes（inference 3.4 §4 口径，引擎编译模块全量走读归 inference 5.3）。

Triton 提供手写融合的 block-level 路线（[triton-lang, Triton](../references.md)，教学参照 + 官方实现；**repo 漂移注**：openai/triton 301 → triton-lang/triton）。所谓 block-level 编程，是指用户以「一个 program 处理一个数据块」的粒度写 kernel，块内元素级并行由编译器映射到 warp/thread——比 CUDA 的 thread-level 高一层抽象，比 TileLang 的 tile 调度解耦（inference 3.4 §5）更贴近逐元素融合。教学片段（融合 residual add + ReLU， $k=2$ 逐元素链，中间值 $x+r$ 驻留 register）：

```python
# 教学片段，非优化实现：Triton 融合 add+relu（block-level）
import triton
import triton.language as tl

@triton.jit
def fused_add_relu(x_ptr, r_ptr, y_ptr, n_el, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # 一个 program 一个块
    mask = offs < n_el
    x = tl.load(x_ptr + offs, mask=mask)        # 读输入（HBM）
    r = tl.load(r_ptr + offs, mask=mask)        # 读残差（HBM）
    y = tl.maximum(x + r, 0.0)                  # add+relu 片上融合，中间值在 register
    tl.store(y_ptr + offs, y, mask=mask)        # 写输出（HBM）
```

读法：`BLOCK` 是块大小（编译期常量），决定每 program 处理的元素数与每线程 register 负载——`BLOCK` 越大，单线程摊到的元素越多、 $R_{\mathrm{th}}$ 越高（命题 2），occupancy 越低；Triton 的 autotune 在候选 `BLOCK`/`num_warps` 里选点，正是 §4 预算约束的工程求解。未融合的 add 与 relu 各是一个 kernel（中间 $x+r$ 落 HBM），融合后 `tl.maximum(x+r, 0)` 的中间值在 register，省一趟往返（命题 1， $k=2$ 省 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$）。Triton 编译器把块内 `tl.arange` 映射到线程、把 `tl.load/store` 映射到向量化访存（4.1 的 ld.global.v4），fusion 与 codegen 的内部机制归编译器（本章不做模块走读）。TileLang 是同类声明式路线（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]，tile 抽象 + 调度空间解耦，inference 3.4 §5 recap），TileKernels 的 fused SwiGLU+quant 是其算子库实证。

## 6. fusion 限制与 NanoFlow 设备内重叠

融合越多越好吗？不是——四条边界约束钉死可融合子图。命题 1/2 之外，fusion 还受四条约束：其一，**fan-out**（§2 边界反例）——中间结果被多下游消费时融合迫使重算或重新物化；其二，**register/shared 预算**（命题 2）——长链压 occupancy 或 spill；其三，**dynamic shape 与 graph break**（§5）——数据依赖控制流切断融合图；其四，**算子兼容性**——reduction 与 pointwise 的融合需行驻留（5.3），跨 block 的归约（grid 级）不能简单融进逐元素链。所以融合不是「越多越好」，是在预算与依赖约束下选最大可融合子图。

NanoFlow 提供另一条互补路线（inference 3.4 §7 recap，不重推）：fusion 减少**单个算子链内**的 HBM 流量，NanoFlow 的 intra-device overlap 重叠**异构算子之间**的执行——把输入切成 nano-batch、划分设备功能单元，让 compute（Tensor Core）、memory（HBM 带宽）、network（NIC）三类资源同时忙。流水加速上界（inference 3.4 式 (4) recap）：

$$
\tau_{\mathrm{seq}} = \tau_c + \tau_m + \tau_n, \qquad \tau_{\mathrm{pipe}} \approx \max(\tau_c, \tau_m, \tau_n), \qquad F_{\max} = \frac{\tau_c + \tau_m + \tau_n}{\max(\tau_c, \tau_m, \tau_n)} \le 3 \qquad (1)
$$

**符号**： $\tau_c, \tau_m, \tau_n$ = compute/memory/network 段时间（recap inference 3.4，教学参数 $\tau_c = 6$ ms、 $\tau_m = 4$ ms、 $\tau_n = 2$ ms 给串行 12 ms、流水稳态 6 ms、上界 2×）。实测较 state-of-the-art 吞吐提升 1.91×、达 optimal 的 50–72%（[Zhu et al., 2024](../references.md)，arXiv:2408.12757，[A]，摘要口径；README 68.5% 另注，inference 3.4 §7 已辨）。**fusion 与 overlap 的关系**：fusion 压单算子的流量下界（命题 1），overlap 藏异构算子的串行等待——前者降 $\tau_m$ 本身，后者让 $\tau_m$ 与 $\tau_c$ 并行；两者正交可叠加，但都受设备资源约束（fusion 占 register/shared，overlap 占 SM/L2 并发干扰，NanoFlow 搜索须计干扰）。设备内调度的引擎编排归 inference（overlap scheduler），本章只立 kernel 侧互补关系。

## 7. 边界：本章不做什么

节省账与预算约束交付完，剩下的归谁？本章交付融合 IO 节省（命题 1 + §2）、register/shared 预算（命题 2 + §4）、三类策略与 FA fusion 视角（§3）、torch.compile/Triton 自动化（§5）、限制与 NanoFlow 互补（§6）；以下各有专章。FA tiling+online softmax 全量指令级走读归 6.3（IO 定理 recap models 4.1）；dequant 融合进 GEMM mainloop（Marlin）归 6.2；GEMM epilogue 代码实体全量走读归 5.2 §7；norm/softmax 行驻留与带宽下界归 5.3；torch.compile/NanoFlow 引擎机制与编译模块走读归 inference 3.4/5.3；Inductor/Triton 编译器内部不做模块走读；实测 drill-down 归 11.2。代码 scope：机制片段级（§5 Triton 教学 kernel）。

本章交付清单（下游 recap 接口）：命题 1 + 节省表（§2）、三类策略图谱（§3）、FA fusion 视角（§3）、命题 2 + 预算账（§4）、Triton block-level 片段（§5）、NanoFlow 互补（§6）。6.3 recap §3 的 FA fusion 视角作走读入口（IO 定理仍指向 models 4.1）；6.2 recap §3 的 GEMM epilogue/mainloop 融合作 dequant 融合底座；5.3 已 recap 本章命题 1 作 norm+quant 融合带宽账。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| $k$ 个单输入单输出逐元素算子融合省 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$（ $(k-1)$ 个中间 tensor 各免一写一读）；recap inference 3.4 命题 1（ $n \to k$） | 命题 1 | §2、练习 1/3 |
| decode $B=64$： $k=4$ 省 3.15 MB / 0.94 μs（launch 主导）；prefill $T=4096$： $k=4$ 省 201.33 MB / 60.06 μs（IO 主导） | §2 表 | §2、练习 1 |
| 多输入算子：SiLU+mul 省 $2 S_{\mathrm{ff}}$（5→3）、residual+norm 省 $S$（5→4）；上界账（L2 命中折扣）+ fan-out 反例 | §2 recap | §2、练习 3 |
| 三类策略：逐元素链（register）/ reduction+逐元素（行驻留，5.3 TE）/ GEMM epilogue（累加器，5.2 §7） | §3 图谱 | §3、练习 4 |
| FA 是最大融合先例： $S,P$ 不落 HBM，IO $\Theta(T^{2}) \to O(T^{2} d_{\mathrm{head}}^{2}/M_{\mathrm{SRAM}})$（recap models 4.1 式 (17)，不重推） | §3 | §3、6.3 |
| 融合 register 预算 $R_{\mathrm{th}} \approx R_0 + w$； $R_0=32 \to 8$ blocks、 $R_{\mathrm{th}}=96 \to 2$ blocks（occupancy 悬崖）；spill 反噬 | 命题 2 | §4、练习 2 |
| 净收益：memory-bound 融合净赚（norm/softmax）、compute-bound 权衡（GEMM epilogue 不占 TC 时间） | §4 判据 | §4、练习 2 |
| torch.compile：Dynamo+guards、Inductor fusion pass → Triton；graph break 切断融合（recap inference 3.4 §4） | §5 | §5、练习 4 |
| Triton block-level：BLOCK 决定 register 负载、autotune 选点；融合 add+relu 中间值驻留 register | §5 片段 | §5、练习 4 |
| fusion 四限制：fan-out / register-shared 预算 / dynamic shape+graph break / 算子兼容性 | §6 | §6 |
| NanoFlow overlap 与 fusion 正交： $\tau_{\mathrm{pipe}} \approx \max(\tau_c,\tau_m,\tau_n)$、 $F_{\max} \le 3$；1.91× / 50–72%（recap inference 3.4 式 (4)） | 式 (1) | §6 |

## 练习

1.（计算 · [10]）融合节省代入（bf16， $\mathrm{BW} = 3352$ GB/s）：(a) decode $B=64$、 $d_{\mathrm{model}}=4096$， $k=2$ 与 $k=4$ 各省多少字节、折合多少 μs；(b) prefill $T=4096$ 同问；(c) 先预测 decode 与 prefill 哪个 IO 节省占比高再算。校验：(a) 1.05 / 3.15 MB、0.31 / 0.94 μs（§2 表）；(b) 67.11 / 201.33 MB、20.02 / 60.06 μs；(c) prefill IO 主导、decode launch 主导（§2）。
2.（计算 · [20]）融合预算账（256 threads/block、 $R_{\mathrm{file}} = 65536$）：(a) 基线 $R_0 = 32$ 与融合 $R_{\mathrm{th}} = 96$ 的 register 项 resident blocks 与 occupancy；(b) `__launch_bounds__(256, 4)` 的寄存器上限 $L_{\mathrm{reg}}$，强压后超额多少 regs/thread 走 spill；(c) 先预测 memory-bound norm 融合与 compute-bound GEMM 融合哪个净收益确定再按 §4 判据核对。校验：(a) 8 对 2 blocks（命题 2）；(b) $L_{\mathrm{reg}} = 64$、超额 32（3.4 命题 1）；(c) memory-bound 净赚（§4）。
3.（推导 · [30]）fan-out 融合节省（补全命题 1 边界）：设中间 tensor $m$ 被两个下游算子消费（fan-out=2），tensor 字节 $S$。(a) 写出未融合（ $m$ 物化一次、两下游各读一次）的 HBM 流量；(b) 写出「融进分支 1、分支 2 重算 $m$」与「融进分支 1、分支 2 重新物化 $m$」两种方案的流量，给出各自比重算/物化成本；(c) 推导融合净省为正的条件（重算 FLOPs 折算时间对物化 IO 时间），说明 fan-out 为何使融合非无条件免费；(d) 用 $k=2$ 单链（无 fan-out）验证退化回命题 1。校验：(d) fan-out=1 时节省 $2S$（命题 1 特例锚）；(a)(b) 读写计数与 §2 口径一致。
4.（代码 · [30]）Triton 融合与 graph break：(a) 用 §5 教学 kernel 写融合 add+relu（PyTorch 教学版或 Triton），对未融合版断言 allclose（容差 $10^{-5}$），先预测中间 tensor 是否落 HBM 及理由；(b) 对含数据依赖 `if y.sum() > 0` 的函数用 `torch._dynamo.explain` 观察 graph break，先预测 break 数与原因再运行；(c) 指出 Inductor fusion pass 在 FX graph 上分组的是哪类节点（§5）；(d) 说明 `BLOCK` 增大对 $R_{\mathrm{th}}$ 与 occupancy 的方向（命题 2）。校验：(a) allclose 容差 $10^{-5}$、中间值不落 HBM（命题 1）；(b) break 数 ≥ 1、原因数据依赖控制流（inference 3.4 §4）；(d) BLOCK 增 → $R_{\mathrm{th}}$ 增 → occupancy 降（§4）。

## 参考

- [triton-lang, Triton](../references.md)：Triton repo（key: triton），教学参照 + 官方实现，2026-11-24——block-level DSL（tl.load/tl.store/tl.arange、@triton.jit、BLOCK constexpr、autotune）；**repo 漂移注**：openai/triton 301 → triton-lang/triton；Triton 论文 MAPL 2019 无 arXiv（spec §5 判教学参照）；§5 教学融合 kernel 的 DSL  backing，编译器内部不做模块走读
- [PyTorch Team, PyTorch](../references.md)：PyTorch repo（key: pytorch），教学参照 + 官方实现，2026-11-24——Inductor fusion pass（pointwise/reduction 节点分组 → Triton codegen）、Dynamo guards、graph break、dynamic shape（§5，机制 recap inference 3.4 §4）
- [Zhu et al., 2024](../references.md)：NanoFlow: Towards Optimal Large Language Model Serving Throughput（key: nanoflow），arXiv:2408.12757，[A]，2026-11-25 export API 再核（第一作者 Kan Zhu）+ efeslab/Nanoflow——intra-device nano-batch 流水 + execution unit scheduling、 $\tau_{\mathrm{pipe}} \approx \max$、 $F_{\max} \le 3$、1.91× / 50–72% optimal（摘要口径，README 68.5% 另注）；§6 recap inference 3.4 §7
- [Wang et al., 2025](../references.md)：TileLang: A Composable Tiled Programming Model for AI Systems（key: tilelang），arXiv:2504.17577，[A]，2026-11-25 再核（第一作者 Lei Wang）+ tile-ai/tilelang——tile 抽象 DSL、调度空间与数据流解耦、多硬件后端；§5 声明式融合对照（recap inference 3.4 §5）
- [NVIDIA, TransformerEngine](../references.md)：NVIDIA/TransformerEngine repo（key: transformerengine），教学参照 + 官方实现，main pushed 2026-11-25——normalization/rmsnorm/rmsnorm_fwd_mxfp8.cu（RMSNorm+MXFP8 cast 融合）+ ln_fwd_kernels.cuh fp8_out 分支（amax/scale，norm+quant 融合）；§3 reduction+逐元素策略实体（5.3 走读 recap）
- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，版本锚 4.9.0（commit 5c179edeb3e2，2026-11-25）——epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp（ThreadEpilogueOp fusion L122）+ epilogue/fusion/operations.hpp L240（LinCombPerColBiasEltAct）；§3 GEMM epilogue 策略实体（5.2 §7 走读 recap，本章不重复全量）
- [Dao et al., 2022](../references.md)：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（key: flashattention），arXiv:2205.14135，[A]，2026-11-25——tiling + online softmax 融合、 $S,P$ 不落 HBM、IO 复杂度 $O(T^{2} d^{2}/M)$（§3 fusion 视角，IO 定理 recap models 4.1 §5.4 不重推，指令级走读归 6.3）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide 13.5（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Register Pressure（-maxrregcount/__launch_bounds__、低 occupancy 多 registers 减 spill + 高 ILP）；§4 融合预算约束的官方口径
