# 3.4 融合、图捕获与编译（Fusion, Graph Capture, and Compilation）

> decode 一步要 launch 三百个 kernel，CPU 提交开销吃掉步时的十分之一到三分之一——CUDA Graph 怎么把这笔开销抹掉，attention 的动态 shape 为什么必须排除在图外？融合两个 pointwise 算子省多少 HBM 字节，batch-invariant 模式固定 reduction 顺序又要付出多少吞吐？本章交付融合 IO 节省、launch 开销、piecewise 切分语义三笔账，以及 torch.compile / TileLang / NanoFlow 的算子层机制 | 前置：1.2（步时带宽账与三口径）、1.4（采样链与 batch invariance 现象，本章回链）、2.1（block table）、3.3（量化 GEMM 与 MoE 算子、Marlin 自适应 tiling）；models 4.1（FlashAttention IO 账）与 models 2.6（SwiGLU）只 recap 不重推 | 锚点：TileLang（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）、NanoFlow（[Zhu et al., 2024](../references.md)，arXiv:2408.12757，[A]）、vLLM / SGLang 编译与 batch-invariant 实现（[vLLM repo](../references.md)、[SGLang repo](../references.md)，[A]）、DeepInfer（[DeepSeek-AI, DeepInfer](../references.md)，教学参照 + 官方实现）、batch-invariance 博客（[He, 2025](../references.md)，[C] 开放技术博客、非闭源）、TileKernels（[DeepSeek-AI, TileKernels](../references.md)，教学参照 + 官方实现）

## 1. 主流量之外的开销与章脊口径

1.2 的步时账只算了两笔主流量：权重读与 KV 读。8B decode $B$ = 64、 $T$ = 4096 的带宽下界是 15.1 ms——但步时不止这两笔：每个 kernel 的 CPU launch（发射：CPU 把 kernel 提交给 GPU）、pointwise（逐元素）算子之间落 HBM 的中间结果、batch 组成变化带来的 reduction（归约）漂移，都是主流量之外的**开销**（口径：本章「开销」指主流量之外的 CPU 提交、中间 tensor IO、同步与编译成本）。本章回答算子层怎么对付这些开销：融合省中间 IO（§2）、CUDA Graph 省 launch（§3）、torch.compile 把两者自动化（§4）、TileLang 提供手写融合 kernel 的声明式路线（§5）、batch-invariant kernel 用固定 reduction 顺序换确定性（§6）、NanoFlow 用 nano-batch 流水把异构资源重叠起来（§7）。算子边界声明（spec §2）：本章讲算法、IO 与接口；warp / tensor core / SASS 指令级、Triton codegen 内部归 gpu 维度，引擎编译模块全量走读归 5.3 / 6.3。

本章口径（章脊锁定，沿用 1.2 的 8B dense 口径，换口径随文声明）：Llama-3.1-8B， $L$ = 32、 $d_{\mathrm{model}}$ = 4096、 $d_{\mathrm{ff}}$ = 14336（models 1.2 §9 config 账）；硬件 H100 SXM（HBM 3.35 TB/s、BF16 dense 989.5 TFLOPS，[NVIDIA datasheet](../references.md)，[C] 官方规格披露）；decode $B$ = 64、 $T$ = 4096 步时 15.1 ms、prefill $T$ = 4096 计算下界 66.5 ms——两者 recap 1.2 §2–3 的算例，不重推； $B$ = 1 步时 4.81 ms 为**本章权重-only 下界算例**（权重 16.1 GB ÷ 带宽；加上 KV 读 0.537 GB 的完整下界 ≈ 4.97 ms，KV 项 +3.3%——下文 launch 账用权重-only 口径，随文注）。局部记号： $S$ = 单个 tensor 的字节数、 $n$ = 融合链的算子数、 $n_{\mathrm{ker}}$ = 每步 kernel 数、 $t_{\mathrm{launch}}$ = 每 kernel 的 CPU 提交开销（声明教学值 5 μs，量级 ~μs；测量与微架构来源归 gpu 维度）、 $\tau_{\mathrm{step}}$ = 步时。 $B$ = batch 内序列数（decode 每序列 1 token）。

## 2. 算子融合：中间 tensor 的 IO 账

RMSNorm（均方根归一化）和量化两个算子融合成一个 kernel，省多少 HBM 流量？先数读写。所谓算子融合（operator fusion），是指把多个算子合并进一个 kernel、中间结果留在寄存器或 shared memory（片上共享内存）而不写回 HBM。三个典型融合对：RMSNorm + quant（norm 输出直接量化成 FP8 喂 GEMM——3.3 §3 的 DeepGEMM 接口要求「cast 融进前置 kernel」即此）、SiLU + mul（SwiGLU 激活， $\mathrm{SiLU}(g) \odot u$ ，结构 recap models 2.6）、residual + norm（残差加与 RMSNorm 合并，同时输出 norm 结果与供下一层用的残差和）。

路线：逐算子数 HBM 读写 → 融合后只剩首读尾写 → 作差。设链上 $n$ 个单输入单输出 pointwise 算子（逐元素算子：输出元素只依赖对应输入元素），每个处理的 tensor 字节数为 $S$ ：

$$
\mathrm{IO}_{\mathrm{unfused}} = 2nS, \qquad \mathrm{IO}_{\mathrm{fused}} = 2S, \qquad \Delta_{\mathrm{IO}} = 2(n-1)S \qquad (1)
$$

**符号**： $n$ = 链上算子数； $S$ = 单 tensor 字节数（链上各 tensor 同 shape 口径）； $\Delta_{\mathrm{IO}}$ = 节省的 HBM 字节。代入 $n$ = 2、 $S$ = 0.5 MiB：省 $2S$ = 1.05 MB（完整实例见下）。

**命题 1**（融合 IO 节省）。单输入单输出 pointwise 链融合后 HBM 流量从 $2nS$ 降到 $2S$ ，节省 $2(n-1)S$ 。**证明**（读写计数，三步）：未融合时每个算子读输入 $S$ 、写输出 $S$ ， $n$ 个算子合计 $2nS$ ；融合后中间结果驻留片上，只有链首输入读 $S$ 与链尾输出写 $S$ 落 HBM；作差即得。∎ 口径注：这是 HBM 流量上界账——中间 tensor 若部分命中 L2，未融合的实际 HBM 流量已低于 $2nS$ ，节省相应缩水；前提是被融合的中间结果不被链外算子复用。边界反例：中间结果有 fan-out（被两个下游算子消费）时，融进其中一个分支会迫使另一分支重算或重新物化中间结果——融合不是无条件免费。

全数字实例（decode $B$ = 64，激活 bf16；RMSNorm+quant 对 fp8 GEMM 部署，节省 = 中间结果的写 + 读 = $2S$ ，与输出 dtype 无关）： $S = 64 \times 4096 \times 2$ = 0.5 MiB。RMSNorm + quant（ $n$ = 2）省 $2S$ = 1.05 MB → 3.35 TB/s 下 0.31 μs；每层 2 个 norm 位点 × 32 层 = 64 处，IO 项合计省 20 μs/step——只占 15.1 ms 步时的 0.13%。诚实的读数：decode 的小 tensor 上 IO 项很小，**大的是 launch 项**：64 处融合各省 1 次 launch，64 × 5 μs = 320 μs/step（2.1%），是 IO 项的 16 倍。换 prefill 口径（ $T$ = 4096 token，原因：展示大 tensor 下两项主次对调）： $S$ = 33.6 MB，单位点省 67.1 MB → 20 μs，64 处合计 1.28 ms，占 66.5 ms prefill 的 1.9%——IO 项主导。

SiLU + mul 的账稍有不同（mul 有两个输入）：未融合 silu 读 gate 写 tmp（ $2S_{\mathrm{ff}}$ ）、mul 读 tmp 与 up 写 out（ $3S_{\mathrm{ff}}$ ），合计 $5S_{\mathrm{ff}}$ ；融合后读 gate 与 up、写 out， $3S_{\mathrm{ff}}$ ——省 $2S_{\mathrm{ff}}$ ， $S_{\mathrm{ff}}$ 为 FFN 中间维 tensor 字节。代入 decode： $S_{\mathrm{ff}} = 64 \times 14336 \times 2$ = 1.84 MB，省 3.67 MB → 1.10 μs/层 × 32 = 35.1 μs/step；prefill： $S_{\mathrm{ff}}$ = 117.4 MB，省 234.9 MB → 70.1 μs/层 × 32 = 2.24 ms（3.4%），还省掉 tmp 的物化容量（每层 117 MB 的峰值显存）。residual + norm：未融合 add 读 x、r 写 s（ $3S$ ）+ norm 读 s 写 y（ $2S$ ）= $5S$ ；融合读 x、r 写 y、s = $4S$ ，省 $S$ ——decode 口径 0.16 μs/处，launch 项照旧是大头。

生产实例两个。TileKernels（[DeepSeek-AI, TileKernels](../references.md)，教学参照 + 官方实现）的 fused SwiGLU + quant：激活与量化一个 kernel 完成（README 算子清单，§5 展开）。FlashMLA 的 fused Q-norm / Q-RoPE / attention / O-RoPE / cast-to-FP8 单 kernel（repo 口径：「saves the overhead of those small kernels」，B200 prefill 至 1460 TFlops / decode 950 TFlops；走读归 7.3）——把 attention 前后的小算子全部吸进主 kernel。最大的融合先例是 FlashAttention： $O(T^{2})$ 的 score 矩阵不落 HBM（IO 账 recap models 4.1，不重推）——本节的小算子融合与它同一原理，只是省的量级不同。

> **提点**：融合的三笔收益——IO 字节、launch 次数、中间容量——在不同 regime 权重不同：decode 小 tensor 上 launch 主导，prefill 大 tensor 上 IO 主导。判断一个融合值不值，先问 tensor 多大、链多长，再查式 (1) 与 §3 的 launch 账。

## 3. CUDA Graph：把三百次 launch 变成一次

8B decode 每步约 330 个 kernel、每次 launch CPU 开销约 5 μs——launch 吃掉多少步时？先估 kernel 数（声明教学口径，bf16；SiLU+mul 按 1 个 custom op 计）：每层 qkv 投影 1 + attention 2（split-KV partial + reduce，3.1）+ o 投影 1 + gate/up 1 + SiLU+mul 1 + down 1 + RMSNorm 2 + residual add 1 + KV cache 写入 1 = 10 个， × 32 层 = 320，加采样链约 10，合 $n_{\mathrm{ker}} \approx 330$ 。

$$
\tau_{\mathrm{launch}} = n_{\mathrm{ker}} \cdot t_{\mathrm{launch}} \qquad (2)
$$

**符号**： $n_{\mathrm{ker}}$ = 每步 kernel 数； $t_{\mathrm{launch}}$ = 每 kernel CPU 提交开销（教学值 5 μs；PyTorch eager（即时执行）栈含 Python 分发，实际更高，本账为下界口径）。代入：330 × 5 μs = 1.65 ms。 $B$ = 64 步时 15.1 ms → 占 10.9%； $B$ = 1 步时 4.81 ms（权重-only 口径，§1）→ 占 **34.3%**（按完整下界 4.97 ms 则 ≈ 33%）。小 batch decode 的步时里约三分之一是 CPU 提交——这就是图捕获的动机。口径注：GPU 时间远大于 CPU 提交且异步队列能跑在前面时，launch 项部分已被隐藏；图的收益在 CPU-bound 区（小 batch decode）最确定。

所谓 CUDA Graph，是指把一段 kernel 调用的 DAG（有向无环图）一次性录制、之后整体重放的机制。capture（捕获）：在 stream capture 模式下跑一遍前向，CUDA 记录每个 kernel 的 launch 参数与**固定指针**、以及 stream（GPU 命令队列）间的依赖边；replay（重放）：一次提交整张图，CPU 成本 ≈ 单次 graph launch。静态三要求：指针固定（输入输出 buffer 常驻）、shape 与 grid 固定、kernel 序列固定；**内容动态合法**——replay 读的是 buffer 的当前内容，把新数据写进同一 buffer 即可。

$$
\tau_{\mathrm{replay}} \approx \tau_{\mathrm{GPU}} + t_{\mathrm{graph}}, \qquad \Delta_{\mathrm{launch}} \approx (n_{\mathrm{ker}} - 1) \thinspace t_{\mathrm{launch}} \qquad (3)
$$

**符号**： $\tau_{\mathrm{GPU}}$ = GPU 侧执行时间（图不改变它）； $t_{\mathrm{graph}}$ = 单次 graph launch 开销（与单 kernel launch 同量级）。代入：省 ≈ 329 × 5 μs = 1.645 ms/step—— $B$ = 1 的步时从 4.81 ms 压向 3.2 ms 量级（权重-only 口径；CPU 提交项全抹的上界）。

full graph vs piecewise（分段图）：full 把整个前向（含 attention）捕进一张图，要求 attention backend 支持静态 shape；而 decode attention 的 KV 长度 $T$ 每步在变、block table 内容每步在变（2.1）、部分 backend 的 grid 依赖序列长——动态 shape 的算子捕进图里，replay 会按旧参数执行。piecewise 的解法：**在 attention 算子处切分**，非 attention 段各自编译 + 捕获成子图，attention 在两次 replay 之间 eager（即时执行）跑。切分点由 splitting_ops 配置，vLLM 默认取 attention 算子列表（`set_splitting_ops_for_v1`，2026-11-23 tarball 核实）。

**命题 2**（piecewise replay 正确性）。若切分点算子是仅有的 launch 参数随步变化的算子，且所有被捕获 kernel 只经固定指针 buffer 访问随步变化的数据，则「更新 buffer 内容 → 按原顺序 replay 各段、切分点算子 eager 执行」与 eager 全量执行结果一致。**证明**（路线：指针固定 → 内容可见 → 顺序保持，三步）：capture 记录的是指针值与 launch 配置，replay 原样使用——指针固定成立；graph 读内存当前值，向同一 buffer 写新数据对 replay 可见——内容可见成立；顺序保持步（stream 依赖边在 capture 中保留、replay 按原序执行，对段数归纳）**留作练习 2**。∎ 边界反例：把 grid 依赖 $T$ 的 kernel 捕进图而不作切分点，replay 按旧 grid 执行、结果错误——这正是 attention 必须排除在外的理由，也是命题前提「切分点覆盖全部动态 launch 参数算子」的必要性。

vLLM 的配置面（**版本漂移注**，`vllm/config/compilation.py`，2026-11-23 tarball 核实）：编译方式枚举已更名 **CompilationMode**（L38）——NONE = 0（eager）/ STOCK_TORCH_COMPILE = 1（标准 torch.compile）/ DYNAMO_TRACE_ONCE = 2（单次 Dynamo trace）/ **VLLM_COMPILE = 3**（自定义 Inductor backend：caching + piecewise compilation + shape specialization + custom passes）；图模式枚举 **CUDAGraphMode**（L165）——NONE / PIECEWISE / FULL / FULL_DECODE_ONLY = (FULL, NONE) / FULL_AND_PIECEWISE = (FULL, PIECEWISE)，V1 默认 FULL_AND_PIECEWISE：decode 批走 full graph、prefill 与混合批走 piecewise。FULL_DECODE_ONLY 只给 decode 批捕 full graph、混合批不捕——适合 P/D（prefill/decode 分离，8.2）的 D 实例省图内存。实现件：`compilation/{piecewise_backend,cuda_graph,backends,partition_rules,breakable_cudagraph,cudagraph_pool}.py`。SGLang 同构：`srt/compilation/cuda_piecewise_backend.py`（+ backend / compilation_config / compile_phase / npu_piecewise_backend），piecewise CUDA graph、attention 动态 shape 排除。

> **坑**：旧名 CompilationLevel 与「PIECEWISE = 3」在当前 main 不存在（2026-11-23 核实）——触发条件：照旧 blog / 旧版文档写配置；现象：枚举对不上、级别语义错位（旧 3 是 piecewise 编译，新 3 是 VLLM_COMPILE）；规避：按 CompilationMode + CUDAGraphMode 双枚举写，版本锚定核实日期。

捕获的代价两笔。启动时间：每个 capture size 各捕一张图 × warmup（预热跑）次数；显存：图内存池 + 常驻 buffer 占 HBM，与 KV 容量互斥（2.1 的容量账）。batch 向上 pad 到最小可用捕获 size（vLLM `decode_cudagraph_capture_shapes` docstring 逐字：「A batch pads to the smallest graph that holds it」）——pad 出的计算浪费与 3.3 §4 grouped GEMM 的 padding 账同构。

> **坑**：replay 读的是捕获时的**指针**——把新输入放进另一个 tensor 再 replay，读到的仍是旧 buffer 的旧数据。触发条件：输入 tensor 每步新分配；现象：输出不随输入变；规避：输入 copy 进捕获 buffer（vLLM `cudagraph_copy_inputs` 开关即为此），或保证调用方始终复用同一 buffer。

```
eager vs CUDA Graph（decode 一步，n_ker ≈ 330）
─────────────────────────────────────────────────────────────
eager:   CPU │launch│launch│launch│ ... ×330 ... │launch│
         GPU │ k1 │  k2  │ k3 │ ...（间隙 = CPU 提交）... │
         步时 = τ_GPU + 330 × t_launch
graph:   CPU │ replay（一次提交）│
         GPU │ k1 → k2 → k3 → ... → k330（录好的 DAG）│
         步时 = τ_GPU + t_graph
piecewise（attention 动态 shape 排除）:
  [norm+qkv+cast 子图 replay] → attention(eager, 读当前 T/block table)
  → [o+FFN+norm 子图 replay] → ... 逐层重复
  切分点 = splitting_ops（默认 attention 算子列表）
─────────────────────────────────────────────────────────────
```

## 4. torch.compile：融合的自动化与 graph break

谁来自动决定融合哪些算子？手写融合 kernel（3.3 的 Marlin、§5 的 TileKernels）覆盖关键路径，长尾 pointwise 链交给编译器。torch.compile 的两段分工：Dynamo（前端）把 Python 字节码捕获成 FX graph（torch 的算子图中间表示），沿途记录 guards（编译期假设条件：shape、dtype、常量值）；Inductor（默认后端）把 FX graph lowering（降级生成）成融合后的 Triton kernel（GPU kernel DSL，codegen 内部归 gpu 维度）——§2 的 pointwise 融合账由它自动执行。

graph break（图断裂）：遇到数据依赖控制流或不支持的 op，Dynamo 把图切断、断点段回退 eager。教学片段（教学用，非优化实现）：

```python
import torch

def f(x):
    y = x * 2 + 1          # pointwise：可进图、可融合
    if y.sum() > 0:        # 数据依赖分支：graph break
        return y.relu()
    return y.neg()

torch.compile(f)(torch.randn(64, 4096, device="cuda"))
```

`y.sum() > 0` 的值运行时才知道，编译期无法定分支——图在此切断，两侧各自成图、分支判断 eager 执行。dynamic shape 的处理：符号维（如 batch 维记 $s_0$ ）+ guards；guard 失效（batch 越过假设区间）触发 recompile（重编译），编译成本计入启动与抖动。vLLM 的 VLLM_COMPILE = 3 在此之上加四件（config docstring 逐字）：caching（编译缓存）、piecewise compilation（按 splitting_ops 切段编译，接 §3 的分段图）、shape specialization（按 `compile_sizes` / cudagraph capture size 特化，消掉符号维）、custom passes（自定义融合 pass，pass = 编译器对图的一轮变换——`pass_config` 的 fuse_attn_quant 等开关，2026-11-23 tarball 核实；§2 的 RMSNorm+quant / SiLU+mul 正是 pass 的目标形态）。

编译与图捕获正交：compile 减 $n_{\mathrm{ker}}$ （融合）并特化 shape，graph 减每次 launch 的成本；piecewise graph 依赖 piecewise compilation（mode = VLLM_COMPILE 且 splitting_ops 非空，config docstring 口径），full graph 则有无编译都支持。合并账（声明教学估计）：pointwise 融合把 $n_{\mathrm{ker}}$ 从 330 压到约 150，再整体 replay → launch 项 ≈ 单次 graph launch，式 (2) 的 1.65 ms 趋零。边界：Inductor / Triton 内部归 gpu 维度；vLLM 编译模块全量走读归 5.3，SGLang 归 6.3。

## 5. TileLang / TileKernels：声明式的融合 kernel

Marlin 的技术清单有八项（3.3 §2），每个 kernel 都手写一遍代价太高——有没有声明式路线？TileLang（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）的答案是 tile 抽象的 DSL（domain-specific language，领域专用语言）：AI kernel 的数据流模式高度规整——tile 在 DRAM（显存）与 SRAM（片上缓存）之间搬运、tile 上执行一串计算；TileLang 把**调度空间**（thread binding、layout、tensorize、pipeline）与**数据流**解耦，调度封装成一组可定制注解与原语（摘要口径），用户写数据流、编译器负责其余优化。多硬件后端（NVIDIA + Ascend）；tile 到 thread / warp（线程束）的映射与指令选择归 gpu 维度。

TileKernels（[DeepSeek-AI, TileKernels](../references.md)，教学参照 + 官方实现）是这条路线的算子库实证（README，2026-11-23 核实）：数十个 TileLang 优化算子——MoE routing top-k / scoring、per-token / per-block / per-channel FP8/FP4 cast / dequant + fused SwiGLU+quant（§2 的生产实例）、Engram gating（fused RMSNorm，前反向）、mHC（Sinkhorn normalization，迭代行列归一）、RoPE；多数接近硬件算力或带宽极限，已用于内部训练与推理；2026-09-30 起 NVIDIA + Ascend 双后端运行时自动选择。DeepInfer 的 custom ops 在 `deepinfer/model_executor/layers/tilelang_ops/`（candidate_blocks.py / compress.py / norm.py / fused_sample.py 等，2026-11-23 tarball 核实）——3.3 的 DSA（DeepSeek sparse attention）candidate blocks、量化 cast、采样在此都有 TileLang 实现；FlashMLA 的 fused kernel 与 DeepInfer 亦基于 TileLang（tilelang 条目口径）。Engram / mHC 的模型侧形式化归 models 维度（V4.1 待补），本章只列算子名。

## 6. Batch-invariant kernels：固定 reduction 顺序的代价

1.4 证明了 greedy 也不可复现：batch 组成改变 reduction（归约，沿某维求和类运算）顺序，浮点非结合使 logits 末位漂移。kernel 层怎么固定顺序、代价多少——1.4 留的「3.4 量化」在此兑现。源头口径 recap（[He, 2025](../references.md)，[C] 开放技术博客、非闭源，2025-09-10，URL 已核实）：pointwise 算子天然 batch-invariant，需要动的只有三类含 reduction 的算子——RMSNorm、matmul、attention。所谓 batch-invariant kernel，是指每个输出元素的 reduction 顺序不随 batch 组成变化的 kernel（博客口径：策略可以随 reduction 长度变，但不能随 batch size 变）。

三类构造（博客口径，算法级）：

- **RMSNorm**：data-parallel 策略——一个 block（thread block，线程块）算一行、大 batch 串行多行，每行的 reduction 全程在单 block 内、顺序固定；小 batch 时不得为凑并行度切换 split-reduction（会改变顺序）。
- **matmul**：固定 2D tile 分区 + 固定 split-K 分区（split-K：沿 reduction 维 K 切分并行，与 3.1 split-KV 同构），分区与 M（= batch）无关；禁 atomicAdd 累加（原子加：完成顺序不定 → 累加顺序不定）；不用 stream-k（其为负载均衡按输出 tile 变 K 切分，连 batch-position-invariance——结果与元素在 batch 内位置无关——都破坏，博客口径）；小 batch 补齐到固定 tensor core tile（padded TC 指令：batch 1 也按满 tile 算，算力浪费换顺序固定）。
- **attention**：固定 KV split 分区（split-KV 的份数与边界不随 batch 组成变，3.1 recap）+ 固定 reduction 树顺序。

**命题 3**（bitwise 可复现）。若前向全部 reduction 顺序与 batch 组成无关、且无原子累加，则相同输入的任意两次运行输出 bitwise（逐比特）一致。**证明**（路线：浮点确定性 → 顺序固定 → kernel DAG 归纳）：浮点运算不满足结合律，但给定操作数与顺序后结果唯一确定；reduction 顺序与 batch 无关 → 相同输入下每个 kernel 执行相同的运算序列 → 输出 bitwise 一致；前向是有限 kernel DAG，按拓扑归纳——基例：输入 embedding 相同；归纳步：任一 kernel 的全部输入 bitwise 一致 → 其输出 bitwise 一致；终止：末层 logits 一致。∎ 边界反例两条：固定随机种子固定不了 reduction 顺序（1.4 坑注 recap）；换硬件 / 软件版本会换指令选择与 reduction 策略——run-to-run deterministic 不等于 hardware/software version invariant（博客口径）。

引擎实现对照（路径 2026-11-23 tarball 核实；模式入口表 recap 1.4 §6，此处给 kernel 层）：

| 引擎 | 入口 | kernel 层事实 |
| --- | --- | --- |
| vLLM | `model_executor/determinism/batch_invariant.py` + `batch_invariant_configs.py`（无 `batch_invariant_ops` 目录） | 覆盖 aten matmul/linear；cuBLAS workspace 配置禁 split-K（`CUBLAS_WORKSPACE_CONFIG=":16:8"`、`CUBLASLT_WORKSPACE_SIZE="1"`）；禁 TF32（TensorFloat-32，10-bit 尾数内部格式） |
| SGLang | `srt/batch_invariant_ops/batch_invariant_ops.py` | matmul 固定 4 段 split-K（`_SEGMENT_COUNT = 4`）、段和按固定顺序相加（源码注释逐字「adds the segment sums in order, at every M」）；RMSNorm / fused_add_rms_norm；attention 固定 block size（`get_batch_invariant_attention_block_size`） |
| DeepInfer | `DEEPINFER_BATCH_INVARIANT` 环境变量 | `model_executor/models/v4_modules/attn.py` 消费（attention 侧开关） |

vLLM 的一件细节把 3.3 与本章钉在一起：batch-invariant 模式对 GPTQ 权重**先 dequant、绕开 Marlin**——docstring 逐字「the Marlin GEMM picks its tiling and K-split from the number of tokens」（`batch_invariant.py`，2026-11-23 核实）。Marlin 的自适应 tiling 是它快的原因（3.3 §2），也是它非 batch-invariant 的原因：确定性与自适应分区不可兼得，模式开关只能二选一。

代价算到数。机制：固定 split-K 放弃小 batch 的最优并行度——[1, K] × [K, 4096] 的 GEMV（矩阵-向量乘），128×128 输出 tile 只切出 32 个 block，对 132 个 SM（streaming multiprocessor；SM 数取 H100 SXM 教学口径）的占用率（口径：有输出 tile 可算的 SM 占比）仅 24%；split-K × 4 给 128 block ≈ 97%——但份数随 batch 自适应正是漂移源，batch-invariant 必须对所有 batch 用同一保守分区，小 batch 吞吐下降。博客实测（Performance 表，[C] 官方博客披露口径）：Qwen-3-8B 单 GPU、1000 条序列输出长 90–110，vLLM 默认 26 s、未优化确定性版 55 s、改进 attention kernel 后 42 s——即 2.1× / 1.6× 的端到端时间膨胀（吞吐相应下降）。用途：调试、评测、需要可复现的生产场景，以及 true on-policy RL（采样器与训练器 bitwise 一致 → KL 散度（分布差异度量）为 0 → 严格 on-policy，博客口径；训练侧编排归 training 8.5）。按需开关，纯吞吐场景关——这是 1.3 SLO 操作点选择的又一实例。

## 7. NanoFlow：设备内 nano-batch 流水

1.2 证明了 decode GEMM 是 memory-bound——NanoFlow 摘要却说 end-to-end serving 对多数常见 workload 是 compute bound，矛盾吗？不矛盾，口径不同：1.2 是**单算子**口径（decode 权重 GEMM 强度 $\approx B \ll R$ ）；NanoFlow 是**端到端混合负载**口径（prefill + decode、attention + FFN + 通信合计，摘要逐字「end-to-end LLM serving is compute bound for most common workloads and LLMs」，[Zhu et al., 2024](../references.md)，[A]，arXiv:2408.12757）。端到端算力利用率低的根源不是单算子 roofline，而是**异构算子在设备内串行执行**：compute（tensor core）、memory（HBM 带宽）、network（NIC，网卡）三类资源轮流忙、互相闲。

NanoFlow 的机制（摘要口径）：nano-batching——把输入切成更小的 nano-batch、复制算子让各份独立执行，打破顺序算子间的依赖；execution unit scheduling——划分设备功能单元、不同算子在不同单元同时执行；自动搜索 nano-batch 的数量、大小、顺序与资源分配，目标是最小化执行时间、约束是并发算子的互相干扰（共享 SM / L2 使单算子变慢）。repo 含 gemv / pipeline 实现代码（efeslab/Nanoflow，[A] 代码核实）。

流水的账。路线：串行时间 = 各段之和 → 流水稳态 = 最长段 → 比值给上界。三类资源各一段：

$$
\tau_{\mathrm{seq}} = \tau_c + \tau_m + \tau_n, \qquad \tau_{\mathrm{pipe}} \approx \max(\tau_c, \tau_m, \tau_n), \qquad F_{\max} = \frac{\tau_c + \tau_m + \tau_n}{\max(\tau_c, \tau_m, \tau_n)} \le 3 \qquad (4)
$$

**符号**： $\tau_c, \tau_m, \tau_n$ = compute / memory / network 段的时间； $F_{\max}$ = 理想流水加速上界（稳态口径，另有填充 / 排空瞬态；三段资源故 ≤ 3）。代入声明教学参数 $\tau_c$ = 6 ms、 $\tau_m$ = 4 ms、 $\tau_n$ = 2 ms：串行 12 ms、流水稳态 ≈ 6 ms、上界 2×——资源越不均衡上界越低，全均衡时取 3。干扰折扣：并发算子共享 SM 与 L2，单算子时间比独占时慢，搜索必须把干扰计进目标（摘要「considering the interference of concurrent operations」）——实测达不到理想上界。

实测数字（**摘要为准**）：较 state-of-the-art serving 系统吞吐提升 1.91×、达 optimal 吞吐的 50–72%（optimal = 资源充分利用的上界口径，摘要原文「achieving 50% to 72% of optimal throughput across popular models」）。版本注：README 另有表述「68.5% of optimal」（59–72% across ported models）——两处口径不同，引用以摘要为准、README 另注（W2 research concern 4）。

```
NanoFlow 设备内流水（示意，3 个 nano-batch × 3 类执行单元）
────────────────────────────────────────────────────────────
串行（eager 整批）:
  compute单元 │ GEMM(整批) │            │            │
  memory单元  │            │ attn(整批) │            │
  network单元 │            │            │ comm(整批) │
  时间 ────────────────────────────────────────────► 12 ms
nano-batch 流水:
  compute单元 │ G1 │ G2 │ G3 │
  memory单元  │    │ A1 │ A2 │ A3 │
  network单元 │    │    │ C1 │ C2 │ C3 │
  时间 ──────────────────────────────► ≈6 ms + 填充/排空
  G/A/C = nano-batch 的 GEMM / attention / comm 段
────────────────────────────────────────────────────────────
```

伏笔：nano-batch 流水是异步调度的算子层基础——SGLang overlap scheduler（6.2）与 DeepInfer 异步调度直接承此路线：「Following existing implementations (SGLang, NanoFlow), the pipeline is overlapped by scheduling asynchronously between GPU and CPU」（doc L334 逐字，官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，2026-11-23 核实；走读归 7.1）。搜索算法细节归论文，本章交付机制与账。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| pointwise 链融合省 $2(n-1)S$ HBM 字节（上界口径，L2 命中与 fan-out 折扣） | 命题 1、式 (1) | §2、练习 1 |
| decode 小 tensor 融合 launch 项主导（64 处省 320 μs vs IO 20 μs）；prefill 大 tensor IO 项主导（1.28 ms / 66.5 ms） | §2 实例 | §2 |
| SiLU+mul 省 $2S_{\mathrm{ff}}$ （5 遍 → 3 遍）；residual+norm 省 $S$ （5 遍 → 4 遍） | §2 读写计数 | §2 |
| launch 账 $n_{\mathrm{ker}} t_{\mathrm{launch}}$ ：330 × 5 μs = 1.65 ms，B=64 占 10.9%、B=1 占 34.3% | 式 (2) | §3、练习 3 |
| CUDA Graph 静态三要求（指针 / shape / 序列），内容动态合法；replay 省 ≈ 1.645 ms/step | 式 (3) | §3 |
| piecewise 在 attention 处切分（动态 shape 排除）；replay 正确性 = 指针固定 + 内容可见 + 顺序保持 | 命题 2 | §3、练习 2 |
| vLLM CompilationMode（VLLM_COMPILE=3）+ CUDAGraphMode（默认 FULL_AND_PIECEWISE）；旧名 CompilationLevel 不存在 | §3 漂移注 + 坑注 | §3 |
| torch.compile：Dynamo 捕获 + guards，Inductor 融合；graph break = 数据依赖控制流回退 eager；compile 与 graph 正交 | §4 | §4、练习 4 |
| TileLang 调度空间与数据流解耦；TileKernels 数十算子近硬件极限；DeepInfer tilelang_ops 为 custom ops 实例 | §5 | §5 |
| batch-invariant = reduction 顺序与 batch 无关 + 禁原子累加 → bitwise 可复现；三类算子构造（RMSNorm / matmul / attention） | 命题 3 | §6 |
| vLLM 确定性模式 dequant GPTQ 绕开 Marlin（自适应 tiling 与确定性不可兼得）；博客实测 2.1× / 1.6× 端到端时间膨胀 | §6 表 + 代价账 | §6、3.3 §2 |
| NanoFlow：端到端 compute bound（混合负载口径）与 1.2 单算子 memory-bound 不矛盾；nano-batch 流水上界 Σ/max ≤ 3；1.91×、50–72% optimal（摘要口径，README 68.5% 另注） | 式 (4) | §7 |

## 练习

1.（计算 · [10]）式 (1) 代入：decode $B$ = 64、 $d_{\mathrm{model}}$ = 4096、bf16，RMSNorm + quant 融合（ $n$ = 2）省多少字节、折合多少 μs（3.35 TB/s）？校验：与 §2 实例一致； $n$ = 1 时节省为 0（特例锚）。
2.（推导 · [20]）补全命题 2 的顺序保持步：对段数归纳（基例 / 保持 / 终止三段），论证 stream 依赖边在 capture 中保留使 replay 与 eager 同序；再说明把 grid 依赖 $T$ 的 kernel 捕进图违反了命题的哪条前提。校验：反例解释与 §3 边界反例一致；归纳的保持步只依赖「依赖边保留」这一 capture 性质。
3.（计算 · [30]）launch 账： $n_{\mathrm{ker}}$ = 330、 $t_{\mathrm{launch}}$ = 5 μs。(a) 先预测 $B$ = 1 与 $B$ = 64 哪个 launch 占比高，再用 §1 步时（4.81 权重-only / 15.1 ms）算占比；(b) graph replay 省多少；(c) compile 把 $n_{\mathrm{ker}}$ 减半后再 replay，launch 项剩多少？校验：(a) 与 §3 读数一致；(b) 与式 (3) 一致；(c) 代入式 (3) 与 (b) 对照（代入锚）。
4.（代码 · [30]）教学用 PyTorch（非优化实现）：(a) 写未融合 silu+mul（中间 tmp 物化）与融合版（单表达式），fp32 对照；(b) 对 §4 的含数据依赖 `if` 函数用 `torch._dynamo.explain` 观察 graph break——先预测 break 数与原因再运行，偏差用 guards 机制解释。校验：(a) allclose 容差 $10^{-5}$ ；(b) explain 输出的 break 数 ≥ 1、原因与 §4 一致。

## 参考

- [Wang et al., 2025](../references.md)：TileLang: A Composable Tiled Programming Model for AI Systems，arXiv:2504.17577，[A]（key: tilelang）+ tile-ai/tilelang——调度空间与数据流解耦、tile 抽象、多硬件后端（§5）
- [Zhu et al., 2024](../references.md)：NanoFlow: Towards Optimal Large Language Model Serving Throughput，arXiv:2408.12757，[A]（key: nanoflow）+ efeslab/Nanoflow——端到端 compute bound 论断、nano-batching + execution unit scheduling、1.91× / 50–72% optimal（摘要口径；README 68.5% 另注）、gemv / pipeline 实现（§7）
- [He, 2025](../references.md)：Defeating Nondeterminism in LLM Inference（Thinking Machines 博客，2025-09-10），[C] 开放技术博客、非闭源（key: batch-invariance）——三类 reduction 算子构造、stream-k 破坏 position invariance、padded TC 指令、Performance 表 26/55/42 s、true on-policy RL（§6）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——CompilationMode / CUDAGraphMode（config/compilation.py L38 / L165）、splitting_ops 默认 attention 列表、piecewise_backend 等实现件、batch_invariant.py（Marlin 绕开 docstring、cuBLAS workspace 配置）、cudagraph_copy_inputs（§3 / §4 / §6）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——srt/compilation/cuda_piecewise_backend.py 等、srt/batch_invariant_ops/batch_invariant_ops.py（§3 / §6）
- [DeepSeek-AI, DeepInfer](../references.md)：教学参照 + 官方实现（key: deepinfer-repo）——tilelang_ops/ custom ops（candidate_blocks / compress / norm / fused_sample）、DEEPINFER_BATCH_INVARIANT（attn.py 消费），2026-11-23 tarball 核实（§5 / §6）
- [DeepSeek-AI, TileKernels](../references.md)：教学参照 + 官方实现（key: tilekernels）——数十 TileLang 算子清单、fused SwiGLU+quant、双后端运行时选择（§2 / §5）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——异步调度承 SGLang / NanoFlow（L334 逐字），2026-11-23 核实（§7）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 HBM 3.35 TB/s、BF16 989.5 TFLOPS dense（§1–3 带宽换算）
