# 3.3 量化 GEMM 与 Fused MoE 算子（Quantized GEMM and Fused MoE Kernels）

> 权重压到 4 bit，带宽账给近 4×——为什么实测加速在 batch 稍大时就消失，Marlin 凭什么把近理想加速推到 batch 16–32？MoE 每个 expert 只收到几十个 token 时，grouped GEMM 在 padding 上浪费多少算力，EP 扩过 32 为什么收益递减？本章交付量化 GEMM 带宽节省、grouped GEMM padding 浪费、EP 扩张拐点三笔账，以及 DeepGEMM / DeepEP / DeepSelect 的算子接口 | 前置：1.2（算术强度与临界 batch、MoE 每 expert batch 与每卡流量）、1.3（datasheet 参数与 scale-up domain）、2.3（量化误差界与格式开销账）、3.2（权重量化算法与格式账）；models 4.5（MoE 路由）、training 1.3（FP8 blockwise）、training 5.4（EP all-to-all 训练账）只 recap 不重推 | 锚点：Marlin（[Frantar et al., 2024](../references.md)，arXiv:2408.11743，[A]）、DeepGEMM / DeepEP / DeepSelect（[DeepSeek-AI, DeepGEMM](../references.md)、[DeepSeek-AI, DeepEP](../references.md)、[DeepSeek-AI, DeepSelect](../references.md)，教学参照 + 官方实现）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、NVIDIA 官方 datasheet（[NVIDIA, H100/H200/B200 datasheet](../references.md)，[C] 官方规格披露）

## 1. 三笔账与章脊口径

3.2 把权重压到了 4 bit，字节数变成 bf16 的四分之一——但「字节少」不等于「时间短」：dequant（反量化，低比特权重还原为浮点，精确定义见 §2）要多花指令、batch 一大瓶颈就换边、MoE 把 GEMM（general matrix multiply，通用矩阵乘）切得又窄又碎。本章回答算子层怎么把字节账兑现成时间账，交付三笔账：量化 GEMM 的带宽节省及其成立条件（§2–3）、grouped GEMM 的 padding 浪费（§4）、EP（expert parallelism，专家并行）扩张的收益拐点（§5），再走读通信与采样算子的接口（§6–7）。算子边界声明（spec §2 三层划分）：本章讲算法、tiling（tile 划分）/ 分裂策略、IO 与算力账、引擎调用接口；warp / tensor core / SASS 指令级实现归 gpu 维度，RDMA / NVLink 硬件与拓扑归 cluster 维度，均纯文字指引不展开。

本章口径（章脊锁定，沿用 1.2 三口径，例题复用，换口径随文声明）：

| 口径 | 模型与几何 | 精度与硬件 | 来源 |
| --- | --- | --- | --- |
| 8B dense | Llama-3.1-8B： $L$ = 32、 $d_{\mathrm{model}}$ = 4096、 $d_{\mathrm{ff}}$ = 14336 | bf16 基线；H100 SXM（BF16 dense 989.5 TFLOPS、FP8 dense 1979 TFLOPS、HBM 3.35 TB/s） | models 1.2 §9 config 账；硬件 [NVIDIA datasheet](../references.md)，[C] |
| V3 MoE | DeepSeek-V3： $E$ = 256 routed + 1 shared、 $k$ = 8、 $d_{\mathrm{model}}$ = 7168、expert 中间维 2048、 $L_{\mathrm{moe}}$ = 58、单 expert 参数量 $W_{\mathrm{exp}} \approx 44.0$ M | 权重 fp8；带宽换算沿用 H100（教学口径，沿 1.2 §6） | [DeepSeek-AI, 2024](../references.md)，[A]；1.2 §5–6 账 |
| V4.1-Flash | 40 层、 $d_{\mathrm{model}}$ = 5120、 $E$ = 384 routed + 1 shared、 $k$ = 6、expert 中间维 2304、词表 $V$ = 129280 | config 声明 fp8 权重（block [32,32]、scale ue8m0）/ expert fp4；SM100 处引 B200（FP4 dense 9000 TFLOPS、BF16 2250、HBM 7.7 TB/s） | [DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]（hf-mirror 传输核实，2026-11-23）；[NVIDIA datasheet](../references.md)，[C] |

符号约定（spec §4.3）： $B$ = batch——§2–3 指本次 GEMM 的行数（decode 时 = batch 内序列数），§4 起 MoE 账指**全局 batch**（DP（data parallelism，数据并行）组合计，沿 1.2 §5 口径）； $E$ = routed expert 总数、 $k$ = top-k、 $e$ = EP 度（大小写严格区分）； $R$ = ridge point = 峰值算力 ÷ 带宽；bytes 不占字母，一律文字「字节」或 $\mathrm{bytes}$ 。本章局部记号： $g$ = 量化 group size（每组权重的个数）、 $M_j$ = expert $j$ 收到的 token 数、 $M_{\mathrm{blk}}$ = GEMM M 轴 tile 行数（与 KV block size $b$ 无关，2.1 定义）、 $C$ = masked layout 每 expert 容量、 $V$ = 词表大小。 $d_{\mathrm{ff}}$ 为 FFN 中间维（models 1.2 记号沿用）。

## 2. W4A16 GEMM：把 dequant 融进 mainloop

权重压到 4 bit，带宽账给近 4×——为什么此前的 kernel 只在 batch 1–2 时拿得到，batch 稍大加速就消失？先定义对象：所谓 W4A16（weight 4-bit / activation 16-bit），是指权重 INT4、激活保持 FP16/BF16 的 weight-only 量化部署形态；所谓 mixed-input GEMM，是指两个输入精度不同的矩阵乘——tensor core 按 FP16 计算，INT4 权重必须先反量化（dequantization：低比特整数乘 scale 还原为浮点值）。朴素做法是先把整个权重矩阵 dequant 成 FP16 再调 cuBLAS：dequant 本身读 4-bit、写 16-bit、物化中间矩阵，多出一整遍 HBM 流量，带宽节省被吃掉。正确做法是把 dequant 融进 GEMM mainloop（mainloop：沿 K 维分块、反复执行「载权重 tile → 计算 → 累加」的主循环；tile = 载入片上计算的数据块）：INT4 tile 载入后在寄存器里即时还原成 FP16、直接喂 tensor core，中间结果不落 HBM。Marlin（[Frantar et al., 2024](../references.md)，[A]，arXiv:2408.11743 + IST-DASLab/marlin）是这条路线的代表 kernel。

带宽账先算到底。路线：每权重字节分数据与 scale 两笔 → 与 bf16 字节相比 → 得理想加速。每 $g$ 个权重共享一个 scale（group 粒度，3.2 格式账），摊到每权重：

$$
\mathrm{bytes}_{\mathrm{weight}} = \frac{q}{8} + \frac{\mathrm{bytes}_{\mathrm{scale}}}{g}, \qquad S_{\mathrm{ideal}} = \frac{2}{\mathrm{bytes}_{\mathrm{weight}}} \qquad (1)
$$

**符号**： $q$ = 权重位宽； $g$ = 量化 group size（本章局部记号）； $`\mathrm{bytes}_{\mathrm{scale}}`$ = 每个 scale 的字节数（FP16 scale = 2）；分子 2 = bf16 每权重字节数； $S_{\mathrm{ideal}}$ = memory-bound 区的理想加速比（时间 ∝ 字节）。代入 $q$ = 4、 $g$ = 128、FP16 scale：每权重 0.5 + 2/128 = 0.515625 字节，有效位宽 4.125 bit， $S_{\mathrm{ideal}} = 16/4.125 \approx 3.88$ ——Marlin README 写「optimal 3.87x」并自注 group scale 的 0.125 bit 存储开销（16 bit ÷ 128 = 0.125 bit/权重），与本式同一笔账（3.87 为截断口径）。粒度越细 scale 开销（口径：此处指摊到每权重的额外字节）越大： $g$ = 32 给 3.56×、 $g$ = 256 给 3.94×——精度账（2.3 / 3.2：group 越细误差界越紧）与带宽账方向相反，group size 是两者的折中旋钮。

**命题 1**（W4A16 加速的成立条件）。memory-bound 区内 W4A16 GEMM 对 bf16 的理想加速由式 (1) 给出；该加速成立当且仅当 batch 不超过上界

$$
B_{\max} = \frac{R \cdot \mathrm{bytes}_{\mathrm{weight}}}{2} \qquad (2)
$$

**证明**（路线：强度 → roofline 比较 → 解出 $B$ ，三步）：GEMM 的 FLOPs = $2BKN$ （ $K$ 、 $N$ 为权重矩阵两维），权重字节 = $`\mathrm{bytes}_{\mathrm{weight}} \cdot KN`$ ，激活与输出字节为 $O(B(K+N))$ 量级、 $B \ll K, N$ 时可略（口径：忽略激活流量的权重主导区），强度 $`I = 2B/\mathrm{bytes}_{\mathrm{weight}}`$ ；memory-bound 要求 $I \le R$ ，解出 $B$ 即式 (2)。∎（口径：理想账，未计 dequant 指令成本；工程折扣见下文。）

代入 H100 BF16 口径 $R \approx 295$ ： $B_{\max} = 295 \times 0.515625/2 \approx 76$ 。Marlin README 用 MAC 语言给同一笔账：现代 GPU 的 FLOP:byte ≈ 100–200，每个 4-bit 权重（0.5 字节）「配得起」25–50 次 tensor core 乘加（MAC）——而每个权重恰好参与 $B$ 次 MAC（每 token 一次），故理论 batch 上限 25–50（其圆整口径，未计 scale）。理论上限是几十，实测近理想 4×（口径：「近理想」指接近式 (1) 的字节比上界 ≈ 3.88，而非绝对 4×）却只保持到 batch 16–32（此前 kernel 仅 1–2）：差距在工程——dequant 指令与 tensor core 指令竞争发射槽，任一流水线不饱和，加速就塌。MARLIN 论文把近理想 4× 推到 16–32、64–128 仍有显著加速（摘要原文「still significant」），集成进 vLLM 后端到端至多 2.8×（[Frantar et al., 2024](../references.md)，[A] 摘要口径）。

全数字实例（8B 口径单层线性 [4096, 4096]，H100；带宽下界口径 = 按峰值带宽折算的时间下界，沿 1.2）。正例： $B$ = 16 时 bf16 权重 33.6 MB → 带宽下界 10.0 μs；INT4（ $g$ = 128）8.65 MB → 2.58 μs；FLOPs $5.37 \times 10^{8}$ → 计算时间 0.54 μs ≪ 2.58 μs，两侧都 memory-bound，加速 ≈ 3.88 成立。边界反例（ $B$ = 128 > 76，越出命题 1 条件）：计算时间涨到 4.34 μs > 2.58 μs，INT4 侧翻转为 compute-bound、时间卡在 4.34 μs；bf16 侧仍 memory-bound 10.0 μs——加速缩到 2.31×。这就是「加速随 batch 消失」的完整读数：不是量化不再省字节，而是瓶颈从带宽换到了算力，字节节省兑现不出来。

Marlin 的技术清单（算法与策略级，README 口径；指令级归 gpu 维度）：激活 L2 常驻 + 寄存器内复用，避免反复读 shared memory（片上共享内存）；权重全局加载异步化、用立即驱逐 cache policy 不污染 L2；shared memory double buffering（双缓冲）重叠载入与计算；dequant 与 tensor core 指令精心交错、两条流水线同时饱和；权重与 scale 离线 reshuffle（重排布局）成理想访问布局（可直接 dequant 进 tensor core 组织）；多 warp（线程束，SM 内调度单元，细节归 gpu 维度）算同一输出 tile 的 partial results 提 warp 数而不增大 tile；striped partitioning 让每 SM（streaming multiprocessor，GPU 计算单元）的 tile 段跨列切片、兼顾 SM 利用率与归约步数；global reduction 直接在 L2 常驻的输出缓冲里做。dequant 本身用位操作指令（lop3 / bfe 一类）从打包的 INT4 里抽字段、拼 FP16——指令级实现归 gpu 维度，本章只记「dequant 必须便宜到不成为第二条瓶颈」这一算法约束。

> **坑**：Marlin 权重必须离线 reshuffle（`marlin.Layer.pack`），直接喂原始 GPTQ 张量不工作；且 Marlin 的 tiling 与 K-split 按 token 数自适应选择——这既是它快的原因，也是它**非 batch-invariant** 的原因：vLLM 的 batch-invariant 模式会 dequant GPTQ 权重、改走固定顺序 matmul 而绕开 Marlin（`vllm/model_executor/determinism/batch_invariant.py` docstring 逐字：「the Marlin GEMM picks its tiling and K-split from the number of tokens」，2026-11-23 tarball 核实；确定性账归 3.4 §6）。另注意硬件边界：README 要求 Ampere/Ada（compute capability ≥ 8.0，Hopper 未优化）——本节带宽账与架构无关，换卡代入其 datasheet 参数同形。

> **提点**：weight-only 量化的两笔账互相独立——带宽节省（本节）由字节比决定，精度代价（3.2 的 GPTQ/AWQ 误差补偿）由量化算法决定。Marlin 不改量化算法、只改执行方式：同一份 GPTQ 权重，朴素 kernel 拿 1–2 token 的加速，Marlin 拿 16–32。

## 3. W8A8 / FP8 GEMM：DeepGEMM 的细粒度 scaling

W4A16 的加速上限卡在 batch 几十——大 batch 想同时省带宽和省算力怎么办？把激活也量化：所谓 W8A8，是指权重与激活都 8-bit 的部署形态，tensor core 直接按 FP8/INT8 计算。字节减半的同时 FP8 峰值翻倍（H100：989.5 → 1979 TFLOPS dense，[NVIDIA datasheet](../references.md)，[C]），两个 regime 各拿 2×：小 batch 时间 ∝ 字节、减半；大 batch 时间 ∝ FLOPs ÷ 峰值、峰值翻倍。强度 $I = 2B/1 = 2B$ （每权重 1 字节，scale 摊薄见下），代入式 (2) 得 $B_{\max} = R_{\mathrm{FP8}}/2 \approx 591/2 \approx 295$ ——与 bf16 口径的 ≈ 295 相同：精度升级使峰值翻倍、字节减半，在临界 batch 式里恰好对消（recap 1.2 式 (6)，不重推）。所以 FP8 的价值不是「更早进 compute-bound」，而是同 batch 下两个 regime 都快 2×。全数字实例（同 §2 的 [4096, 4096] 层）： $B$ = 16 时 FP8 权重 16.8 MB → 5.01 μs，恰为 bf16 10.0 μs 的一半； $B$ = 512 时 FP8 计算时间 8.68 μs vs bf16 17.4 μs，仍是 2×。

精度侧的代价由 fine-grained scaling 压住。per-tensor 单一 scale 被 outlier 撑大动态范围、误差界随之松（recap 2.3 命题 1/2：均匀量化误差界由极差单独决定）；V3 用 128×128 blockwise scale（recap training 1.3，不重推），每块独立 scale、动态范围限制在块内。DeepGEMM（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照 + 官方实现）把这条路线推到硬件格式级，README 口径（2026-11-23 核实）：SM90（Hopper 架构代号）要求 scaling factor（SF，缩放因子）为 FP32 格式、rescale 在软件侧完成；SM100（Blackwell 架构代号）要求 packed UE8M0——4 个 UE8M0 pack 进一个 `torch.int`。UE8M0 是 8-bit 无符号纯指数格式（scale 恒为 2 的幂）：乘 scale = 指数加法、无尾数舍入；每 scale 1 字节、为 SM90 FP32 scale（4 字节）的 1/4，SM100 tensor core 的 block-scaling 路径原生消费该格式。LHS（left-hand side，左矩阵 A）SF 须 TMA-aligned 转置布局（README；TMA = Tensor Memory Accelerator，Hopper 起的异步批量搬运单元，指令细节归 gpu 维度）。V4.1-Flash config 把粒度再收紧：weight_block_size [32,32]、scale_fmt ue8m0、expert_dtype fp4 [W]；DeepInfer 生产 routed-expert 权重 per-32 ue8m0 MXFP4（官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，2026-11-23 核实）。FP4 的字节账 recap 2.3 式 (4)：MXFP4 每元素 (32×0.5+1)/32 = 0.53125 字节、对 bf16 3.76×；B200 FP4 峰值 9000 TFLOPS dense = BF16 2250 的 4×（[NVIDIA datasheet](../references.md)，[C]；Hopper 无 FP4）——FP4 在 SM100 上把 FP8 的「两个 regime 各赚一倍」复制成 4× 量级。

scale 的字节开销会不会吃掉节省？代入 4096×4096 FP8 矩阵、128×128 块、FP32 scale：(4096²/128²)×4 = 4096 字节，占数据字节的 0.024%——fine-grained scaling 的带宽代价可略，买的是精度（3.2）。DeepGEMM 的另两件工程事实。其一，DeepJIT（JIT，just-in-time compilation：运行时而非安装期编译）：全部 kernel 运行时编译、安装期无 CUDA 编译，按 shape 特化 + 磁盘缓存（默认 `$HOME/.dj`，`DG_JIT_CACHE_DIR` 可配）——serving 的 M 维（= batch）每步都变、prefill 的 M 千差万别，提前编译枚举不完 shape，JIT 按 shape 特化、编译成本被缓存摊掉。其二，接口约定：命名 D = C + A@B、NT layout（SM90 仅 NT，SM100 全 layout）；转置与 FP8 cast（类型转换）须用户自理或融进前置 kernel（README 明示，融合账归 3.4）。DeepGEMM 的组件不止 GEMM：indexer MQA logits（`fp8_fp4_mqa_logits`，non-paged / paged 两版，weighted ReLU MQA——DSA（DeepSeek sparse attention）serving 侧归 3.1）、HyperConnection、Mega MoE（§5）。warp specialization / TMA / SASS 级优化归 gpu 维度。

## 4. Grouped GEMM：variable-M 的 padding 账

V3 decode 全局 batch 1024 时每个 expert 平均只收 32 个 token（1.2 命题 2 recap： $Bk/E$ ），而 GEMM 的 M 轴 tile 是 128 行——浪费多少？先定义对象：所谓 grouped GEMM，是指一组共享 N/K、各自 M 独立的 GEMM 合并成一次 kernel 调用。MoE 的 $E$ 个 expert 权重 shape 相同（V3：gate/up 各 [7168, 2048]、down [2048, 7168]——linear1 与 linear2 是 N/K 不同的两次 grouped GEMM 调用），token 数 $M_j$ 却逐 expert 不同：逐个调 cuBLAS 要 $E$ 次 launch（CPU 提交 kernel 给 GPU，定义见 3.4 §1；开销账归 3.4）、每个小 M 又填不满 SM；grouped GEMM 把全部 expert 的 M 合起来 tiling、一次 launch。M 轴即 token 维：每个 expert 的 GEMM 是 [M_j, K] × [K, N]，padding（补齐到 tile 边界的无效行）发生在 M 轴。DeepGEMM 只 group M 轴（N/K 固定，README 逐字：为 MoE expert 同 shape 的场景定制），两种 layout 对应两种 serving 形态：

```
MoE grouped GEMM 两种 layout（每卡 E/e = 4 个本地 expert，M_blk = 128）
──────────────────────────────────────────────────────────────────
contiguous（训练 fwd / prefill：CPU 知道每个 M_j）
  token 按 expert 排序拼接，每段对齐 M_blk
  ┌─────────────┬─────────────┬─────────────┬─────────────┐
  │ expert 0    │ expert 1    │ expert 2    │ expert 3    │
  │ M_0=32      │ M_1=17      │ M_2=128     │ M_3=5       │
  │ +pad 96     │ +pad 111    │ +pad 0      │ +pad 123    │
  └─────────────┴─────────────┴─────────────┴─────────────┘
  offsets = [0,128,256,384,512]；形状 [512, K] 随路由变 → 动态 shape
masked（decode + CUDA Graph：CPU 不知道 M_j）
  ┌─────────────┬─────────────┬─────────────┬─────────────┐
  │ 容量 C      │ 容量 C      │ 容量 C      │ 容量 C      │ 形状 [4, C, K] 静态
  │ 有效 32     │ 有效 17     │ 有效 128    │ 有效 5      │ mask = GPU 侧计数
  └─────────────┴─────────────┴─────────────┴─────────────┘
  kernel 只算有效 block；buffer 形状不变 → 可被 CUDA Graph 捕获
──────────────────────────────────────────────────────────────────
```

contiguous layout：token 按 expert 排序拼成一个张量 + offsets（累积段边界），每段对齐 GEMM M block size——DeepGEMM 默认 128（`kLegacyMKAlignmentForContiguousLayout`，`csrc/jit_kernels/heuristics/runtime.hpp` L12，2026-11-23 tarball 核实；`get_mk_alignment_for_contiguous_layout()` 可查可设）。它要求 CPU 知道每个 $M_j$ （路由结果同步回 CPU），适合训练前向与 prefill。masked layout：预分配静态容量 buffer + mask 张量（每 expert 有效 token 数，GPU 侧），kernel 只算有效部分（README 口径）——decode 开 CUDA Graph 时 CPU 不知道每 expert 收到多少 token，而 CUDA Graph 要求静态 shape（3.4），masked 用「形状静态、内容动态」满足捕获条件；典型输入 = DeepEP low-latency kernel 的输出（README 逐字）。

padding 浪费算到闭式。路线：每段向上对齐到 $M_{\mathrm{blk}}$ 的倍数 → 数 padding 行 → 除以总计算行。

$$
\mathrm{pad}_j = \left\lceil \frac{M_j}{M_{\mathrm{blk}}} \right\rceil M_{\mathrm{blk}} - M_j, \qquad w = \frac{\sum_j \mathrm{pad}_j}{\sum_j (M_j + \mathrm{pad}_j)} \qquad (3)
$$

**符号**： $M_j$ = expert $j$ 的有效 token 数； $M_{\mathrm{blk}}$ = M 轴 tile 行数（局部记号，区别于 KV block size $b$ ）； $w$ = 计算浪费占比（口径：padding 行占总计算行的比例——padding 行照过 tensor core、结果丢弃）。

**命题 2**（padding 上界）。(a) 对任意 $M_j$ ， $`\mathrm{pad}_j \le M_{\mathrm{blk}} - 1`$ ；(b) 若每个 expert 非空且 $M_j \le M_{\mathrm{blk}}$ （均值 $\mu$ ），则 $w = 1 - \mu/M_{\mathrm{blk}}$ ，与 $M_j$ 的具体分布无关（全部 $M_j = \mu$ 是其特例）。

**证明**（路线：ceiling 性质 → 分两情形）：(a) $\lceil M_j/M_{\mathrm{blk}} \rceil M_{\mathrm{blk}}$ 是不小于 $M_j$ 的最小 $M_{\mathrm{blk}}$ 倍数，与 $M_j$ 之差小于一个周期 $M_{\mathrm{blk}}$ ，故 padding ≤ $M_{\mathrm{blk}} - 1$ 。(b) $0 < M_j \le M_{\mathrm{blk}}$ 时 $\lceil M_j/M_{\mathrm{blk}} \rceil = 1$ ，每段恰占一个整块：计算行 $M_{\mathrm{blk}}$ 、有效行 $M_j$ ，对全部 expert 求和得 $w = \sum_j (M_{\mathrm{blk}} - M_j)/\sum_j M_{\mathrm{blk}} = 1 - \mu/M_{\mathrm{blk}}$ ，分布无关。∎ 正例： $M_j$ = 513、 $M_{\mathrm{blk}}$ = 128 → pad = 127（触上界）； $M_j$ = 512 → pad = 0（整除零浪费）。分布无关正例：上方 layout 图四段 $M_j$ = [32, 17, 128, 5]（均值 45.5）→ $w$ = 1 − 45.5/128 = 64.5%，与图中 padding 行 330/512 一致。边界反例：「padding 浪费总是小」不成立—— $\mu \ll M_{\mathrm{blk}}$ 时 $w$ 趋近 1，算力大头花在 padding 上。

全数字实例（V3 口径， $M_{\mathrm{blk}}$ = 128）： $B$ = 1024 → $\mu = Bk/E$ = 32， $Bk$ = 8192 > $E$ 、均衡路由下每 expert 非空且 $M_j \approx 32 \le M_{\mathrm{blk}}$ ——每段恰占一个整块， $w$ = 1 − 32/128 = **75%，精确成立、与余数分布无关**（命题 2(b)）。注意均匀余数均值模型在此**不适用**：余数均匀 {0..127} 的均值 63.5 已超过 $\mu$ = 32，两个假设不相容；该模型只在 $\mu$ ≳ $M_{\mathrm{blk}}$ （ $M_j$ 跨多个 tile、余数近似均匀）时成立。含空 expert 时（ $Bk < E$ regime，1.2 §5）空段 pad = 0、不占块，按激活 expert 数另计。 $B$ = 16384 → $\mu$ = 512 ≳ $M_{\mathrm{blk}}$ ，平均 pad ≈ $(M_{\mathrm{blk}}-1)/2$ = 63.5， $w \approx 63.5/(512+63.5)$ = 11.0%（平均式与渐近式的推导留作练习 2）； $\mu \gg M_{\mathrm{blk}}$ 时渐近 $w \approx M_{\mathrm{blk}}/(2\mu)$ 。读数：padding 浪费是 MoE 强度稀释（1.2 式 (12)）的另一面——小 batch 不仅权重读摊不薄（强度低），算出的 FLOPs 也大头是 padding。prefill 的大 M 天然摊薄 padding，decode 的小 M 是重灾区：这解释了 decode 为什么要更小的 $M_{\mathrm{blk}}$ （JIT 按 shape 选），也解释了 masked layout 的定位——它**不消除**块粒度浪费（口径：tile 为最小计算单元，有效行不足一个 tile 仍按整块算——tile 粒度推论，指令细节归 gpu 维度），消除的是 CPU 同步与动态 shape。

masked 的容量代价单独一笔：

$$
\mathrm{bytes}_{\mathrm{buf}} = \frac{E}{e} \cdot C \cdot K \cdot \mathrm{bytes}_{\mathrm{act}} \qquad (4)
$$

**符号**： $C$ = 每 expert 容量（可接收 token 上限，由 `num_max_tokens_per_rank` 与 EP 组规模定）； $K$ = expert 输入维（= $d_{\mathrm{model}}$ ）； $`\mathrm{bytes}_{\mathrm{act}}`$ = 激活每元素字节。代入 V3 口径（ $e$ = 64、 $E/e$ = 4、 $C$ = 1024、 $K$ = 7168、FP8）：4×1024×7168×1 = 29.4 MB/层/rank（rank：通信组内进程编号）；实际有效数据 4×32×7168 = 0.92 MB——容量是实际的 32×，但绝对值小（HBM 80 GB 口径），内存开销不是 masked 的主成本，块粒度计算浪费才是。选型一句话：prefill / 训练（ $M_j$ 大、CPU 同步可接受）走 contiguous；decode + CUDA Graph（ $M_j$ 小且动态）走 masked。DeepEP V2.5 的接口恰好对上：`do_cpu_sync=True` 给精确输出尺寸 + CPU 侧 expert 计数（接 contiguous），`False` 给 GPU 侧接收计数 + 容量分配、expert GEMM 用 `handle.psum_num_recv_tokens_per_expert` 定有效段（接 masked）——§6 展开。

## 5. Fused MoE：Mega MoE 与 EP > 32 的拐点账

EP 从 8 扩到 128，每卡权重流量降 16×（1.2 命题 3 recap）——为什么 DeepInfer 官方说到 EP 32 以上收益就停了？先数未融合 MoE 的一步链：dispatch → grouped GEMM1 → SwiGLU → grouped GEMM2 → combine，每段独立 kernel（外加 quant/cast），通信与计算串行、通信时间全暴露（口径：暴露 = 没有被计算重叠盖住，training 5.4 §6 同口径）。所谓 fused MoE，是指把这条链融进更少的 kernel；Mega MoE（DeepGEMM 组件，README 口径）是融合最彻底的形态：单个 mega-kernel 融合 EP dispatch + linear1/linear2（FP8×FP4 或 FP8×FP8）+ SwiGLU + EP combine，NVLink 通信与 tensor core 计算重叠。接口事实（README，2026-11-23 核实）：需 symmetric memory（对称内存：各 rank 以相同布局分配、可被远端直接访问的 buffer）多进程 launch，PyTorch ≥ 2.9；`fp8_fp4_mega_moe` 执行、`transform_weights_for_mega_moe` 做权重布局变换、`get_symm_buffer_for_mega_moe` 分配 buffer。

重叠的账：未融合 $\tau \approx \tau_{\mathrm{comm}} + \tau_{\mathrm{gemm}}$ ；融合后 $\tau \approx \max(\tau_{\mathrm{comm}}, \tau_{\mathrm{gemm}}) + \varepsilon$ （ $\varepsilon$ = 同步与融合残余开销）。 $\tau_{\mathrm{comm}} \le \tau_{\mathrm{gemm}}$ 时通信被完全盖住——这是官方论断的机制：「Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE」（doc L285 逐字，官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，2026-11-23 核实）。限定词「within a scale-up domain」有账可依：scale-up domain（1.3 定义 recap：NVLink / 灵衢紧耦合互联域）内带宽高、 $\tau_{\mathrm{comm}}$ 小到能藏进计算；跨域 RDMA 的延迟藏不住，所以 Mega MoE 是**域内**通用解而非跨域通用解。

**命题 3**（EP 扩张拐点）。设每卡 expert 权重读时间为 $A/e$ （ $A$ = expert 权重总字节 ÷ 带宽，1.2 命题 3 的 $\propto 1/e$ 账），每卡计算与固定开销下界为 $F$ ，EP 通信/同步开销 $g(e)$ 对 $e$ 单调不减，则步时

$$
\tau(e) = \max\left( \frac{A}{e},\ F \right) + g(e) \qquad (5)
$$

存在拐点 $e^{\ast} = A/F$ ：当 $e \ge e^{\ast}$ 时带宽项不再贡献步时下降， $\tau(e) = F + g(e)$ 单调不减——EP 扩张的边际收益为零或负（代入 $A$ = 195.2 ms、 $F$ = 6.1 ms 得 $e^{\ast}$ = 32，两数来源见下方实例）。

**证明**（路线：max 换边 → 单调性，两步）： $A/e$ 对 $e$ 严格递减、与常数 $F$ 恰交于 $e^{\ast} = A/F$ ； $e \ge e^{\ast}$ 时 $A/e \le F$ ，max 取 $F$ ，故 $\tau = F + g(e)$ ； $g$ 单调不减 → $\tau$ 单调不减。∎ 边界注：若 $g \equiv 0$ 则 $e \ge e^{\ast}$ 后 $\tau$ 持平（零收益不亏损）；生产里 $g > 0$ （通信对端更多、同步链更长、跨节点转发更多），拐点即反转点。正例代入见下表；反例：把「每卡流量 $\propto 1/e$ 」外推成「步时 $\propto 1/e$ 」在 $e > e^{\ast}$ 后失效——流量还在降，步时不再降。 $F$ 取常数的前提：#DP = #EP 耦合扩张下每卡分配对数 $Bk/e = B_{\mathrm{local}} k$ 不变、每卡计算量不随 $e$ 变（1.2 §5 reconciling 注）。固定全局 batch 口径下每卡计算也 $\propto 1/e$ ，可并入 $A/e$ 项（ $A$ 换为权重读与计算时间的合并系数），「递减项 + 不减项」的结构不变、拐点位置移动。

全数字实例（V3 口径教学参数）：routed expert 合计 654 GB fp8（1.2 §6）、带宽 3.35 TB/s → $A$ = 195.2 ms；取 $F$ = 6.1 ms、 $g(e) = 0.02(e-1)$ ms（**两者为声明的教学参数**，生产下界由硬件/batch/通信实现决定，非官方数值）→ $e^{\ast}$ = 32：

| $e$ | 带宽项 $A/e$（ms） | 下界 $F$（ms） | 通信 $g(e)$（ms） | 步时 $\tau(e)$（ms） |
| --- | --- | --- | --- | --- |
| 8 | 24.4 | 6.1 | 0.14 | 24.5 |
| 16 | 12.2 | 6.1 | 0.30 | 12.5 |
| 32 | 6.1 | 6.1 | 0.62 | 6.72 |
| 64 | 3.05 | 6.1 | 1.26 | 7.36 |
| 128 | 1.52 | 6.1 | 2.54 | 8.64 |

最小值在 $e$ = 32，再扩反而变慢。DeepInfer 官方披露把生产拐点放在 EP 32 附近：「Increasing EP raises arithmetic intensity to avoid being memory-bound. Once memory bandwidth is no longer the bottleneck, a larger EP size introduces additional communication and synchronization overhead.」（doc L295 逐字，官方披露，2026-11-23 核实）。前半句是 1.2 §5 reconciling 注的机制（#DP = #EP 耦合扩全局 batch → 强度升），后半句是本命题的 max 换边——两半合起来才是「EP 扩张有最优值」的完整读数。32 不是普适常数：换硬件、换 batch、换通信实现， $e^{\ast}$ 随之移动；不变的是结构。第二笔代价是 NVLink 多 job 干扰：「Different jobs sharing NVLink within a supernode can interfere with one another and slow down. Using the whole supernode for a single EP group avoids that interference.」（doc L297 逐字，官方披露）——超节点（supernode，大规模 scale-up 域，如 GB200 NVL72，1.3 口径）的 NVLink 是共享带宽，多 job 的 all-to-all 互相干扰；整超节点作单 EP 组把干扰内部化（部署形态归 7.4）。Mega MoE 对拐点的回应是把 $g(e)$ 重叠进计算： $\tau \approx \max(A/e, F, g(e))$ ，拐点外推——但推不到无穷， $g(e)$ 自身的增长（同步链、SM 占用，§6）终会主导。

## 6. DeepEP：dispatch / combine 的语义与版本账

V4.1 decode 每卡每层 dispatch 只有 1.97 MB——这个消息尺寸下，瓶颈是带宽还是延迟？先 recap 训练侧（training 5.4 §6，不重推）：V3 的 normal kernel 走两阶段转发——IB（InfiniBand）50 GB/s 发到目标节点同序号 GPU、NVLink 160 GB/s 转发到目标 expert 所在 GPU，占 20 个 SM、与计算重叠（V3 集群口径，[DeepSeek-AI, 2024](../references.md)，[A]）。serving 侧增量：decode 需要另一族 kernel。DeepEP（[DeepSeek-AI, DeepEP](../references.md)，教学参照 + 官方实现）的 V1 语义（**锚 tag v1.2.1**，V1 最后版本，2026-11-23 tags API 核实；当前 main 已移除 V1，见下方漂移注）分两族：

- **normal（high-throughput）kernel**：训练与 prefill 用。NVLink + RDMA 两阶段：节点内走 NVLink、跨节点走 RDMA，利用两域带宽不对称（NVLink ≫ RDMA）做聚合转发——大消息带宽主导，两阶段把两段带宽都用满。
- **low-latency kernel**：decode 用。纯 RDMA 点对点、绕过 NVLink 两阶段转发——纯 RDMA 为原始设计口径，v1.2.1 已含节点内 NVLink 直连优化（News 2025.06.05「leverage NVLink as much as possible」，单跳直连，不改与两阶段转发的对比语义）。通信-计算重叠用 hook 机制：dispatch 返回 hook，RDMA 接收在后台进行、不占 SM，计算侧在需要数据时**主动调用 hook()** 完成接收（pull 式，供 double-batch overlap 用；v1.2.1 README 逐字「the actual tensor will not be received only if you call hook()」），免 CPU 同步往返。绕过的理由：decode 消息小、延迟主导——两阶段多一跳（hop，一次网络转发）延迟，带宽优势在小消息上兑现不出来。

通信量一笔账。每个 token 的 hidden 向量复制 $k$ 份（每选中 expert 一份），combine 对称（expert 输出发回原 rank 加权归约）：

$$
\mathrm{bytes}_{\mathrm{dispatch}} = B k \thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}} \qquad (6)
$$

**符号**： $B$ = 全局 batch； $k$ = top-k； $`\mathrm{bytes}_{\mathrm{act}}`$ = 激活每元素字节（FP8 dispatch = 1，scale 另计）。代入 V3 口径（ $B$ = 1024、 $k$ = 8、 $d_{\mathrm{model}}$ = 7168、FP8）：每 MoE 层 58.7 MB，58 层 dispatch + combine 合计 6.81 GB/step（全局口径，全部 rank 合计）； $e$ = 64 摊到每卡 0.92 MB/层——NVLink 900 GB/s（[NVIDIA datasheet](../references.md)，[C]）下带宽时间仅 1.02 μs。再看逐 peer（通信对端 rank）消息：V4.1 口径（ $B_{\mathrm{local}}$ = 64，单 DP 组本地 batch，1.2 §5 记号； $k$ = 6、 $d_{\mathrm{model}}$ = 5120、FP8）每卡每层 1.97 MB，均衡路由摊到 64 个 rank 后逐 peer 仅 30.7 KB，IB 50 GB/s 下带宽时间 0.61 μs——与 RDMA 单跳延迟同量级（μs 级；硬件测量归 cluster 维度）。延迟主导，这就是 low-latency kernel 存在的定量依据。FP8 dispatch 把数据字节减半（README 支持），延迟项不动：量化省带宽不省延迟。

> **坑**：V1 两阶段语义勿写成当前 main API。触发条件：引用 DeepEP 时不核对版本；现象：当前 main（V2.5）已无 normal/low-latency 两族 API 与 NVSHMEM 后端，照旧文档写接口会全部对不上；规避：机制讲解锚 tag v1.2.1，接口走读锚当前 main（本章口径 2026-11-23），两处分开标注。

V2.5 漂移注（当前 main，README 2026-11-23 核实）：V1 API、NVSHMEM 后端、旧文档全部移除；high-throughput 与 low-latency 统一为单 **EPBuffer** 接口；NCCL Gin 后端（device-side 通信 API、复用既有 NCCL communicator（通信句柄））；DeepJIT 运行时编译；SM 与 QP（queue pair，RDMA 连接单元，硬件归 cluster 维度）数解析计算、免 auto-tuning；hybrid / direct 两模式保留。训练/prefill 与 decode 的区分移到参数 `do_cpu_sync`：`True` = 精确输出尺寸 + CPU 侧 expert 计数（接 contiguous grouped GEMM）；`False` = GPU 侧接收计数、按容量分配输出，expert GEMM 用 `handle.psum_num_recv_tokens_per_expert` 定有效段（接 masked grouped GEMM，§4 的两种 layout 在此闭环）。

SM 账（V2.5 README Notes 逐字）：「EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported」——通信 kernel 占 SM，与计算竞争 SM 资源，这是命题 3 里 $g(e)$ 的一个成分；对照 2.4 的 GDR swap-in（RDMA NIC 直写 HBM、不占 SM）：KV 传输可以零 SM，EP dispatch/combine 不行（当前版本口径）。口径分层注：V1 low-latency 的 hook 接收是后台 RDMA、不占 SM（v1.2.1 README 逐字「without any SM occupation」，指接收侧）；V2.5 Notes 说的是 dispatch/combine 操作整体——两者版本与范围都不同，勿混读。冗余 expert 原语：`lb_prefetch_weights` / `lb_reduce_grads` 经 NVLink 为冗余 expert 预取权重与量化 scale、反向归约 FP32 梯度（MoonEP / UltraEP 动机，README 口径）——EPLB（expert parallelism load balancer）冗余副本的 serving 侧部署归 7.4，本章只记接口存在。

## 7. Sampling 与 logits 算子

$B$ = 64、词表 129280 时 logits 一遍是 33.1 MB——采样链要读几遍？logits（模型末层输出的未归一化分数，形状 [B, V]）的消费链 recap 1.4（公式不重推）：logits processors（repetition / frequency / presence penalty、ban mask）→ top-k / top-p / min-p 截断 → softmax → 采样；greedy 退化为 argmax。每个 processor 都是对 [B, V] 的一遍读-改-写，IO 账：

$$
\mathrm{bytes}_{\mathrm{logits}} = n_{\mathrm{pass}} \cdot B V \cdot \mathrm{bytes}_{\mathrm{logit}} \qquad (7)
$$

**符号**： $n_{\mathrm{pass}}$ = 对 logits 的遍数； $V$ = 词表大小； $`\mathrm{bytes}_{\mathrm{logit}}`$ = 每 logit 字节（采样场景 FP32 = 4）。代入 V4.1 词表 $V$ = 129280 [W]、 $B$ = 64、FP32：单遍 33.1 MB → 3.35 TB/s 下 9.88 μs；以 5 遍为例（教学假设；1.4 的链为 penalty×3 + ban mask，遍数随配置变）：5 个 processor 各走一遍 = 49.4 μs，fused penalties（全部 penalty + ban mask 单遍完成）把 penalty 链降到 9.88 μs，省 39.5 μs/step（top-k 选择另计一遍）。采样 IO 与上下文长 $T$ 无关——长 $T$ 下占步时比例小，短 $T$ 下占比升，融合 kernel 的价值随 $T$ 减小而上升。

DeepSelect（[DeepSeek-AI, DeepSelect](../references.md)，教学参照 + 官方实现）：DSA lightning indexer 与 sampler 的 TopK（按值选出前 k 个）kernel，对 `torch.topk` 2–20×（README 口径）。两场景：indexer bf16、topk ≤ 4096、主优化 topk = 512（= V4.1 config index_topk [W]，DSA serving 走读归 3.1）；sampling fp32、vocab ~128K（benchmark vocab_size = 129280，与 V4.1 config 同值）。性能度量是**有效内存带宽**（口径：TopK 无浮点运算，FLOP rate 无意义——README 逐字）：读 B×V 一遍、输出 B×k ≪ 输入，流量 ≈ 输入字节，算法是部分选择而非全排序。工程开关：`sorted=False`（不按值排序）、`return_value=False`（不输出值、约快 10%，README 口径）——省掉不需要的工作。argmax 是单遍归约、无需选择结构，greedy 路径最便宜。引擎接口：vLLM `v1/sample/`（+ `ops/`）、SGLang `srt/layers/sampler.py`（[vLLM repo](../references.md)、[SGLang repo](../references.md)，[A]，2026-11-23 tarball 核实；全量走读归 5.4 / 6.1）。DeepGEMM 的 indexer MQA logits kernel（§3）是 DSA 的打分侧、DeepSelect 是选择侧，DSA serving 全链归 3.1。

## 8. 算子选择决策表

走完七节，选型落在三个变量上：精度、硬件、batch 形态——哪个组合走哪条路线？下表汇总本章各节的选型论断：

| 场景 | 硬件 | batch 形态 | 路线 | 依据 |
| --- | --- | --- | --- | --- |
| W4A16 INT4 weight-only | Ampere/Ada | 16–32 内近理想 4× | Marlin（dequant 融 mainloop） | §2 |
| W8A8 FP8 dense | SM90 / SM100 | prefill / 大 batch | DeepGEMM `fp8_gemm_nt`（fine-grained SF） | §3 |
| FP4 routed expert | SM100 | MoE | DeepGEMM FP8×FP4 / Mega MoE | §3 / §5 |
| MoE 训练 fwd / prefill | SM90 / SM100 | $M_j$ 大、CPU 同步可接受 | contiguous grouped GEMM | §4 |
| MoE decode + CUDA Graph | SM90 / SM100 | $M_j$ 小且动态 | masked grouped GEMM + DeepEP `do_cpu_sync=False` | §4 / §6 |
| scale-up 域内 MoE | NVLink 域 | 任意 EP / batch | Mega MoE（通信重叠进计算） | §5 |
| EP 跨域 | RDMA 网络 | decode 小消息 | DeepEP low-latency 语义（V1 锚 v1.2.1 / V2.5 EPBuffer） | §6 |
| 采样 top-k | CUDA / Ascend | fp32、vocab ~128K | DeepSelect + fused penalties | §7 |

边界声明：本章交付算法、tiling / 分裂策略、IO 与算力账、引擎接口；指令级实现归 gpu 维度，通信硬件归 cluster 维度；量化算法与格式账归 3.2，decode attention 算子族归 3.1，融合、图捕获与 batch-invariant kernel 归 3.4；EPLB 与 Wide-EP 生产形态归 7.4，serving 并行组合归 8.1。V4.1-Flash 模型侧形式化归 models 维度（V4.1 待补），本章只引 config 字段做 serving 账。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| W4A16 每权重字节 = $`q/8 + \mathrm{bytes}_{\mathrm{scale}}/g`$ ； $g$ = 128 FP16 scale → 4.125 bit、理想加速 16/4.125 ≈ 3.88（README 3.87× 同账） | 式 (1) | §2、练习 1 |
| 加速成立当且仅当 $`B \le R \cdot \mathrm{bytes}_{\mathrm{weight}}/2`$ ；H100 INT4 上界 ≈ 76，Marlin 实测近理想保持到 16–32 | 命题 1、式 (2) | §2 |
| dequant 融进 mainloop、中间结果不落 HBM；Marlin 八项技术饱和双流水线；tiling 随 token 数自适应 → 非 batch-invariant（3.4 绕开） | §2 技术清单 + 坑注 | §2、3.4 §6 |
| FP8 两个 regime 各 2×（字节减半 + 峰值翻倍）；临界 batch 跨精度对消（≈ 295）；SM100 SF = packed UE8M0（4/int），SM90 = FP32 | §3；式 (2) 代入 | §3 |
| MXFP4 0.53125 B/元素、对 bf16 3.76×；B200 FP4 峰值 = BF16 4×；V4.1 expert fp4 + block [32,32] ue8m0 [W] | §3；2.3 式 (4) recap | §3 |
| grouped GEMM 每段 padding ≤ $M_{\mathrm{blk}}-1$ ；每段非空且 ≤ $M_{\mathrm{blk}}$ 时 $w = 1-\mu/M_{\mathrm{blk}}$ （分布无关）；V3 $\mu$ = 32 → 75% 精确；均匀余数均值式仅限 $\mu$ ≳ $M_{\mathrm{blk}}$ （ $\mu$ = 512 → 11.0%） | 命题 2、式 (3) | §4、练习 2 |
| contiguous（CPU 知 $M_j$ ，prefill）vs masked（静态容量 + GPU 计数，decode + CUDA Graph）；masked 不消除块粒度浪费；buffer = (E/e)·C·K·bytes | 式 (4)、§4 图 | §4、练习 4 |
| Mega MoE 单 mega-kernel 融合 dispatch + GEMM1/2 + SwiGLU + combine，NVLink 通信重叠进计算；scale-up 域内通用解（官方披露 doc L285） | §5 重叠账 | §5 |
| EP 步时 $\tau(e) = \max(A/e, F) + g(e)$ ，拐点 $e^{\ast} = A/F$ ； $e \ge e^{\ast}$ 后边际收益 ≤ 0；生产拐点 EP ≈ 32（官方披露 doc L295）+ NVLink 多 job 干扰（L297） | 命题 3、式 (5) | §5、练习 3 |
| dispatch 字节 = $`Bk \thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}}`$ ；decode 逐 peer 消息 KB 级 → 延迟主导 → low-latency 纯 RDMA；V2.5 统一 EPBuffer、 `do_cpu_sync` 分训练/prefill vs decode | 式 (6)、§6 | §6 |
| EP dispatch/combine 占 SM、零 SM RDMA EP 不支持（V2.5 Notes）；对照 GDR swap-in 零 SM（2.4） | §6 SM 账 | §6、2.4 |
| logits IO = $`n_{\mathrm{pass}} B V \cdot \mathrm{bytes}_{\mathrm{logit}}`$ ；fused penalties 5 遍 → 1 遍省 39.5 μs/step（B=64、V=129280；5 遍为教学假设）；DeepSelect 2–20×、度量 = 有效内存带宽 | 式 (7) | §7 |

## 练习

1.（计算 · [10]）式 (1) 代入： $q$ = 4、FP16 scale、 $g$ = 128，算有效位宽与理想加速；先预测 $g \to \infty$ 的极限再取极限。校验：与 Marlin README 的 3.87× 公开口径一致（16/4.125）；极限 = 4（纯数据字节比）。
2.（推导 · [20]）(a) 证明命题 2(b)（ $0 < M_j \le M_{\mathrm{blk}}$ ⇒ 每段恰占一个整块，对 $M_j$ 求和）；(b) 在 $\mu$ ≳ $M_{\mathrm{blk}}$ 、余数近似均匀（ $M_j$ 模 $M_{\mathrm{blk}}$ 均匀）条件下推导平均 padding ≈ $(M_{\mathrm{blk}}-1)/2$ 与平均浪费式，并说明该模型为什么在 $\mu$ = 32 < $M_{\mathrm{blk}}$ 时不适用；(c) 给 $\mu \gg M_{\mathrm{blk}}$ 时的渐近式。校验：代入 $\mu$ = 512、 $M_{\mathrm{blk}}$ = 128 与 §4 实例一致；渐近式与 $M_{\mathrm{blk}}/(2\mu)$ 一致；(b) 的适用边界与 §4 实例的 $\mu$ = 32 行一致。
3.（计算 · [30]）EP 拐点账： $A$ = 195.2 ms、 $F$ = 6.1 ms、 $g(e) = 0.02(e-1)$ ms。(a) 先预测 $\tau(64)$ 与 $\tau(32)$ 谁大并说明理由，再算 $e \in \lbrace 8, 16, 32, 64, 128 \rbrace$ 的步时；(b) 解出 $e^{\ast}$ ；(c) 若 Mega MoE 把 $g(e)$ 完全重叠进计算（ $\tau = \max(A/e, F, g(e))$ ），拐点怎么动？校验：(a) 与 §5 表一致；(b) $e^{\ast} = A/F$ ；(c) 代入 $e$ = 128，重叠式的 $\tau$ 应低于未重叠式（代入锚）。
4.（代码 · [30]）教学用 PyTorch 实现两种 grouped GEMM layout（非优化实现）： $E$ = 4 expert、 $K$ = $N$ = 64、 $M_j$ = [32, 17, 128, 5]、 $M_{\mathrm{blk}}$ = 128。(a) contiguous：拼接 + offsets、每段 pad 到 $M_{\mathrm{blk}}$ 、批量 matmul，数 padding 行；(b) masked：容量 $C$ = 128 静态 buffer + mask，只算有效段；(c) 两者有效行输出与逐 expert 循环参考对照。校验：fp32 allclose 容差 $10^{-5}$ ；(a) padding 行数与式 (3) 逐段手算一致，且总计算行数 = expert 数 × $M_{\mathrm{blk}}$ （守恒锚）。

## 参考

- [Frantar et al., 2024](../references.md)：MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models，arXiv:2408.11743，[A]（key: marlin，W2 升 [A]）+ IST-DASLab/marlin（默认分支 master）——近理想 4× 保持到 batch 16–32、64–128 递减仍显著、vLLM 端到端至多 2.8×；FLOP:byte 100–200 → 25–50 MAC/权重；八项技术清单（§2）
- [DeepSeek-AI, DeepGEMM](../references.md)：教学参照 + 官方实现（key: deepgemm）——DeepJIT、SM90 FP32 SF / SM100 packed UE8M0、NT layout、contiguous / masked grouped GEMM（M 轴 group、默认对齐 128）、Mega MoE mega-kernel 与 symmetric memory 接口、indexer MQA logits（§3–5）
- [DeepSeek-AI, DeepEP](../references.md)：教学参照 + 官方实现（key: deepep）——V1 normal/low-latency 语义锚 tag v1.2.1（hook pull 式接收「the actual tensor will not be received only if you call hook()」、接收侧零 SM「without any SM occupation」、News 2025.06.05 NVLink 直连优化）；当前 main V2.5 统一 EPBuffer、NCCL Gin、 `do_cpu_sync` 、 `handle.psum_num_recv_tokens_per_expert` 、SM 占用 Notes、lb_prefetch_weights / lb_reduce_grads（§4 / §6）
- [DeepSeek-AI, DeepSelect](../references.md)：教学参照 + 官方实现（key: deepselect）——indexer / sampling 两场景、2–20× vs torch.topk、有效内存带宽度量、sorted / return_value 开关（§7）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——Mega MoE scale-up 域通用解（L285）、EP > 32 递减（L295）、NVLink 多 job 干扰（L297）、routed-expert per-32 ue8m0 MXFP4，2026-11-23 核实（§3 / §5）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——EP 通信训练侧背景（normal kernel 两阶段、IB 50 / NVLink 160 GB/s 集群口径，recap training 5.4 §6）
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：[W]（key: v41-flash-config）——weight_block_size [32,32] / scale_fmt ue8m0 / expert_dtype fp4、E=384+1 top-6、词表 129280、index_topk 512（hf-mirror 传输核实，2026-11-23）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 BF16 989.5 / FP8 1979 TFLOPS dense、3.35 TB/s、NVLink 900 GB/s；B200 FP4 9000 / BF16 2250 TFLOPS dense、7.7 TB/s；Hopper 无 FP4（§2–3 / §6）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——batch-invariant 模式绕开 Marlin（model_executor/determinism/batch_invariant.py docstring）、sampling v1/sample/（§2 / §7）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——sampling srt/layers/sampler.py（§7）
