# 6.2 量化 kernel（Quantization Kernels）

> 字节账给近 4 倍，为什么实测加速在 batch 稍大时就消失，Marlin 凭什么把近理想加速推到 batch 16–32？本章 recap inference 3.2/3.3 的量化算法与算子语义（GPTQ/AWQ/SmoothQuant 算法侧不重推），交付 GPU 侧 kernel 增量：W4A16 dequant 融合 GEMM mainloop 的指令级实现（Marlin fresh tarball 走读——lop3.b32 + magic-number 把 INT4 在寄存器里即时还原成 FP16，命题 3）、dequant 融合的带宽节省账（命题 1）与 mixed-input tiling 强度（命题 2）、W8A8/FP8 的 fine-grained scaling 两级累加（DeepGEMM 走读）、INT4/FP4 unpack 指令族、grouped GEMM 的 variable-M 调度 | 前置：5.2（GEMM mainloop 与 DeepGEMM promotion recap）、1.3（mma.sync m16n8k16 几何）、4.4（cp.async）、5.1（shared 归约）、6.1（格式账与 scale 消费路径）；recap 源：inference 3.2（量化算法）、inference 3.3（带宽账与 grouped GEMM 算子语义）；下游 6.3（FA FP8）、5.4（融合） | 锚点：Marlin（[Frantar et al., 2024](../references.md)，arXiv:2408.11743，[A]，commit 1f25790bdd49）、DeepGEMM（[DeepSeek-AI, DeepGEMM](../references.md)，commit 5081f8513af5，教学参照 + 官方实现）、PTX ISA 9.5 lop3/prmt/bfe（[NVIDIA, PTX ISA](../references.md)，[C] 官方规格披露·一手权威）——代码锚 2026-11-25 fresh tarball 一手核实

## 1. 字节账怎么兑现成时间账：本章口径与代码 scope

inference 3.3 把 W4A16 的带宽账算到了近 4 倍理想加速，也列了 Marlin 的八项技术清单——但留了一句「dequant 必须便宜到不成为第二条瓶颈」给 gpu 维度。本章回答三个问题：dequant 在指令级长什么样，为什么几条位操作就能把 INT4 还原成 FP16（§4）？dequant 融合省下的带宽怎么算账、又付出多少指令开销（§2/§5）？fine-grained scale 与 variable-M 怎么进 mainloop 和调度器（§6/§7）？

**recap 边界**（spec §4.2，不重推）：量化**算法**侧——GPTQ 二阶误差补偿、AWQ 激活感知缩放、SmoothQuant outlier 迁移——归 inference 3.2 §3–5，本章只 recap 结论与指针；W4A16 带宽账（inference 3.3 式 (1) 理想加速、命题 1 临界 batch）与 grouped GEMM padding 账（inference 3.3 命题 2）recap 不重推；GEMM 模块全量走读归 5.2，本章不重复。

**代码 scope：片段级走读**（spec §4.2 两层走读纪律）。Marlin 取 dequant mainloop 片段（`marlin/marlin_cuda_kernel.cu` 的 lop3/dequant/scale/matmul lambda），DeepGEMM 取 scaling/JIT 片段（promotion 的 scale 读取与谓词选择、sm100 UE8M0 路径、DeepJIT 接口）；GEMM 五层实体与 collective mainloop 全量走读归 5.2，本章 recap 其 mainloop 骨架作底座。

**本章口径**（章脊锁定）：带宽账脊柱沿 inference 3.3——**H100 SXM**（ $\mathrm{BW} = 3352$ GB/s、BF16 dense 989.4 TFLOPS、ridge $R \approx 295$），单个线性层权重 $[4096, 4096]$、 $B = 16$。**换口径声明**：Marlin 代码面向 **Ampere/Ada**（compute capability ≥ 8.0，README L65 逐字「Marlin is not yet optimized for Hopper」），故指令走读用 mma.sync m16n8k16（Ampere 路径）而非 wgmma；带宽账与架构无关，换卡代入其 datasheet 参数同形。DeepGEMM 走读分 SM90（Hopper）与 SM100（Blackwell）两路。

**符号声明**（spec §4.3）：**本章 $M, N, K$ = GEMM 问题维（行/列/归约维），非消息字节/参数量/序列长**； $M$ = batch 行数（decode 时 = batch 内序列数）； $T_M, T_N, T_K$ = threadblock tile 三维（tile 维，非序列长 $T$，禁裸 $T$）； $B$ = batch； $B_{\mathrm{crit}}$ = W4A16 临界 batch（inference 3.3 命题 1 $B_{\max}$ 的翻译，别于 occupancy 符号 $B_{\max}$ = max blocks/SM，spec §4.3）； $g$ = 量化组大小、 $q$ = 权重位宽（沿 inference 3.2）；bytes 不占字母，一律文字「字节」或 $\mathrm{bytes}$。指令几何一律字面数（mma.sync m16n8k16）。

**版本锚**：Marlin **commit 1f25790bdd49**（master，2024-09-04），单文件 kernel `marlin/marlin_cuda_kernel.cu`（822 行）；DeepGEMM **commit 5081f8513af5**（main，2026-10-20），device kernel 在 `deep_gemm/include/deep_gemm/impls/`（非 submodule）。本章全部 file:line 按该两版本 fresh tarball 核实（2026-11-25）；跨版本阅读先对 README/CHANGELOG。

## 2. W4A16 dequant 融合：带宽节省账

权重压到 4 bit，带宽账给近 4 倍——可为什么不能先把整个权重矩阵 dequant 成 FP16 再调 cuBLAS？先 recap 对象（inference 3.3 §2，不重推）：所谓 W4A16（weight 4-bit / activation 16-bit），是指权重 INT4、激活保持 FP16/BF16 的 weight-only 量化部署形态；所谓 dequant（反量化），是指低比特整数乘 scale 还原为浮点值。朴素做法物化（materialize）一个 FP16 中间矩阵：dequant kernel 读 INT4、写 FP16，GEMM 再读 FP16——多出一整遍 HBM 流量。正确做法是把 dequant 融进 GEMM mainloop：INT4 tile 载入后在寄存器里即时还原成 FP16、直接喂 Tensor Core，中间结果不落 HBM。这笔「多出的一整遍」值多少，本节算成闭式。

推导路线：逐权重数 HBM 读写字节 → 物化三遍对融合一遍 → 与 bf16 基线比。关键一步是物化路径的 FP16 中间矩阵要**写一遍再读一遍**，两遍都计 HBM 流量。

**命题 1**（融合对物化的流量节省）。设每权重 INT4 数据加 scale 摊薄共 $`\mathrm{bytes}_{\mathrm{weight}}`$ 字节（ $g = 128$、FP16 scale 时 0.515625），则物化路径每权重 HBM 流量为 $`\mathrm{bytes}_{\mathrm{weight}} + 4`$ 字节（读 INT4 + 写 FP16 2 B + GEMM 读 FP16 2 B），融合路径为 $`\mathrm{bytes}_{\mathrm{weight}}`$ 字节；两者之比 $`1 + 4/\mathrm{bytes}_{\mathrm{weight}}`$。**证明**（计数，两步）：其一，物化三遍——dequant 读 $`\mathrm{bytes}_{\mathrm{weight}}`$、写 FP16 2 B，GEMM 读 FP16 2 B，合计 $`\mathrm{bytes}_{\mathrm{weight}} + 4`$；其二，融合只读 INT4 一遍 $`\mathrm{bytes}_{\mathrm{weight}}`$，FP16 中间值在寄存器里生灭、零 HBM 流量。相除即得。∎

全数字代入（ $g = 128$、FP16 scale，$`\mathrm{bytes}_{\mathrm{weight}} = 0.515625`$）：物化 $0.515625 + 4 = 4.515625$ B/权重，融合 0.515625 B/权重，比 $4.515625/0.515625 = 8.76$。**物化甚至比 bf16 基线（2 B/权重）还慢 $4.515625/2 = 2.26$ 倍**——量化本为省带宽，物化反而把带宽涨到 2.26 倍，这就是「dequant 必须融合」的定量根据。脊柱层 $[4096, 4096]$（16,777,216 权重）：物化 75.8 MB、融合 8.65 MB、bf16 33.55 MB；按 3352 GB/s 折时间下界 22.6 / 2.58 / 10.0 μs。融合对 bf16 的加速 $10.0/2.58 = 3.88$，正是 inference 3.3 式 (1) 的理想加速 $2/0.515625$（recap，不重推）。

```
物化（materialize）：HBM 三遍
  INT4 权重 ──读0.515625B──► dequant kernel ──写2B──► FP16 中间矩阵(HBM)
                                                          │ GEMM 读2B
                                                          ▼
                                                       cuBLAS GEMM
融合（fused）：HBM 一遍
  INT4 权重 ──读0.515625B──► GEMM mainloop 寄存器内 dequant ──► Tensor Core
                              （FP16 中间值不落 HBM）
```

理想加速成立有条件：inference 3.3 命题 1 给出临界 batch $`B_{\mathrm{crit}} = R \cdot \mathrm{bytes}_{\mathrm{weight}}/2 \approx 76`$（recap，不重推；inference 原记号 $B_{\max}$ 译 $B_{\mathrm{crit}}$，避与 occupancy 符号 $B_{\max}$ = max blocks/SM 撞，spec §4.3），超过则 INT4 侧翻转为 compute-bound、字节节省兑现不出来。Marlin README 实测近理想 4× 只保持到 batch 16–32（L7-8 逐字「close to ideal (4x) speedups up to batchsizes of 16-32 tokens」），远低于理论上界 76——差距在工程：dequant 指令与 Tensor Core 指令竞争发射槽，任一流水线不饱和加速就塌（§5）。README 另给「optimal 3.87x」并自注 group scale 的 0.125 bit 存储开销（L41-43），与 $16/4.125 = 3.88$ 同一笔账（3.87 为截断口径）。

## 3. Marlin mixed-input tiling 走读

Marlin 的 tile 怎么切，为什么小 batch 用一种切法、大 batch 用另一种？走读 host 侧配置（`marlin_cuda_kernel.cu`）：`THREADS = 256`（8 warp，L705）、`STAGES = 4`（4 级 async 流水，L706）、`SHARED_MEM = 96 * 1024`（CC 8.6 最大 shared，L707）。tile 选择按 batch 分档（L755-761）： $M \le 16$ 时 `thread_k = 128`、`thread_n = 128`（即 $T_K = 128$、 $T_N = 128$、 $T_M = 16$）； $M > 16$ 时 `thread_k = 64`、`thread_n = 256`。注释逐字（L756）：「For small batchizes, better partioning is slightly more important than better compute utilization」——小 batch 优先切分均匀（喂满 SM），大 batch 优先算力利用。kernel 配置经 `CALL_IF` 宏编译期枚举（L801 起 `CALL_IF(1, 8, 8, -1)` 等），只实例化测试中有用的组合（L798-799 注释「For compilation speed, we only define the kernel configurations that have seemed useful」）——这是编译期特化，与 DeepGEMM 的运行时 JIT 对照（§6）。

**mixed-input tiling 强度**（本章增量）。mixed-input GEMM 两个输入精度不同（A FP16、B INT4），tile 强度账与同精度 GEMM 不同。推导路线：数一个 tile 的 FLOPs 与 HBM 字节 → 分「A 走 HBM」与「A 常驻 L2」两口径 → 与 bf16 同 tile 比。关键一步是**小 $T_M$ 下激活 A 的流量占比不可忽略**，A 是否常驻 L2 直接决定强度比。

**命题 2**（mixed-input tile 强度）。tile $T_M \times T_N \times T_K$、A FP16（2 B）、B 每权重 $`\mathrm{bytes}_{\mathrm{weight}}`$ 字节。A 常驻 L2（权重主导 HBM）口径下 $`I = 2 T_M / \mathrm{bytes}_{\mathrm{weight}}`$；A 走 HBM 口径下 $`I = 2 T_M T_N / (2 T_M + \mathrm{bytes}_{\mathrm{weight}} T_N)`$。**证明**（计数）：tile FLOPs $= 2 T_M T_N T_K$；A 字节 $= T_M T_K \times 2$、B 字节 $`= T_N T_K \times \mathrm{bytes}_{\mathrm{weight}}`$。A 常驻 L2 时 HBM 只计 B，$`I = 2 T_M T_N T_K / (T_N T_K \mathrm{bytes}_{\mathrm{weight}}) = 2 T_M / \mathrm{bytes}_{\mathrm{weight}}`$；A 走 HBM 时两笔都计，$`I = 2 T_M T_N / (2 T_M + \mathrm{bytes}_{\mathrm{weight}} T_N)`$（ $T_K$ 约去）。∎

全数字代入（ $T_M = 16$、 $T_N = 128$、$`\mathrm{bytes}_{\mathrm{weight}} = 0.515625`$）：A 常驻 L2 口径 $I = 2 \times 16 / 0.515625 = 62.06$ FLOP/B，bf16 同口径 $2 \times 16 / 2 = 16$，比 $62.06/16 = 3.88$；A 走 HBM 口径 $I = 2 \times 16 \times 128 / (2 \times 16 + 0.515625 \times 128) = 4096/98 = 41.8$，bf16 同口径 $4096/288 = 14.2$，比 $41.8/14.2 = 2.94$。**A 不常驻 L2 时强度比从 3.88 掉到 2.94**——因为 $T_M = 16$ 时 A 字节（ $16 \times 128 \times 2 = 4096$ B）占 tile 总流量（ $4096 + 8448 = 12544$ B）的 33%，不可忽略。这正是 Marlin「激活常驻 L2」技术的账：README L19 逐字「all activations are essentially always fetched from L2 cache and are further reused several times within registers」。脊柱层 $B = 16$ 的 A 矩阵 $16 \times 4096 \times 2 = 128$ KB，远小于 L2（H100 50 MB），常驻成立；权重 8.65 MB 主导 HBM，整 GEMM 强度 60.2 FLOP/B（bf16 15.9），与命题 2 的 A-L2 口径一致。

数据流与异步装载（4.4 cp.async recap，增量是 cache policy 分化）：

```
Marlin mainloop（STAGES=4，T_M=16/T_N=128/T_K=128，batch≤16）
HBM            shared(4 stage)        register                 Tensor Core
A FP16 ─cp.async.cg(L61)─► sh_a ─ldmatrix m8n8.x4(L111)─► frag_a ─┐
B INT4 ─cp.async.cg+evict_first(L76)─► sh_b ─128bit─► frag_b_quant ─lop3/dequant(§4)─► FragB ─mma.m16n8k16(L98)─► frag_c
scale  ─cp.async(新组时,L405)─► sh_s ─► frag_s ─hmul2(L157)─┘
```

权重 B 用 `cp.async.cg.shared.global.L2::cache_hint` 配 `createpolicy.fractional.L2::evict_first`（L75-76），注释逐字（L66-68）：权重「only accessed precisely once and should thus not pollute the L2 cache which we need for inputs A and outputs C」——**B 立即驱逐、A/C 常驻 L2**，这是命题 2「A 常驻 L2」成立的工程手段（cache policy 分化，4.4 cache hint 机制 recap）。A 用 `ldmatrix.sync.aligned.m8n8.x4.shared.b16`（L111）按 Tensor Core fragment 布局搬进寄存器（4.2/1.3 recap）；计算用 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`（L98，FP16 输入 FP32 累加，1.3 几何 recap）。

切分与归约（5.1 recap，增量是 split-K 的 shared 归约）：Marlin 用 striped partitioning（L207-214 注释）让每 SM 的 tile 段跨列切片，兼顾 SM 利用率与归约步数。为增 warp 数而不增大输出 tile，多个 warp 算同一输出位置的 partial sums（split-K），注释逐字（L463-465）：「we slice across the k dimension ... multiple warps accumulate partial sums of the same output location; which we have to reduce over ... in shared memory」——`thread_block_reduce`（L466）做 shared 内对数归约（5.1 归约结构 recap）；跨 block 的 `global_reduce`（L514）直接在 L2 常驻的输出缓冲里做（L515 注释「reduce directly in the output buffer to maximize L2 cache utilization」）。

## 4. 指令级 dequant：lop3 + magic-number

几条位操作凭什么能把 INT4 还原成 FP16，还不需要整数转浮点指令？先给指令族（PTX ISA [C]）：`lop3.b32` 是三输入查找表位运算（一条指令完成任意三输入布尔组合），`prmt.b32` 是字节置换，`bfe` 是位域提取。**Marlin 用 lop3 而非 bfe**——第二 nibble 用 `>>8` 移位（L447）；unpack 指令族（lop3/prmt/bfe）里 Marlin 具体选 lop3（精度注）。

lop3 走读（L116-126）：注释逐字（L116-117）「Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to automatically recognize it in all cases」——显式写 lop3 是因为编译器不总能自动识别。`template<int lut>`（L118）把 8-bit 真值表作模板参数，`lop3.b32 %0, %1, %2, %3, %4`（L122）。dequant 用的 LUT 是 `(0xf0 & 0xcc) | 0xaa`（L136-137），算得 `0xea`——正是 `(a & b) | c` 的真值表（逐位核对：输入 $(a,b,c)$ 的 8 个组合，`0xea` 的第 `(a<<2)|(b<<1)|c` 位等于 `(a&b)|c`）。所以 `lop3<0xea>(q, LO, EX)` 等于 `(q & LO) | EX`，一条指令完成「掩码取 nibble + OR 上 magic number」。

**magic-number 的核心**（命题 3）。FP16 在 $[1024, 2048)$ 区间 ULP = 1（ $1024 = 2^{10}$，尾数 10 位， $2^{10-10} = 1$），故整数 $0..15$ 落进 1024 的尾数低位是**精确**的——把 nibble OR 进 `0x6400`（= FP16 1024.0）的尾数，直接得到 FP16 的 $1024 + n$，无需 I2F 转换指令。再用 FP16 减法去掉偏置。dequant 走读（L131-152）：

```
q (int32, 8 个 INT4 nibble): [n7][n6][n5][n4][n3][n2][n1][n0]
LO = 0x000f000f  取 n0(bits0-3) 与 n4(bits16-19)
HI = 0x00f000f0  取 n1(bits4-7) 与 n5(bits20-23)
EX = 0x64006400  两个 FP16 1024.0
lo = (q & LO) | EX  →  half0 = fp16(1024 + n0), half1 = fp16(1024 + n4)
hi = (q & HI) | EX  →  half0 = fp16(1024 + 16·n1), half1 = fp16(1024 + 16·n5)
SUB = 0x64086408 = fp16(1032)；MUL = 0x2c002c00 = fp16(1/16)；ADD = 0xd480d480 = fp16(-72)
frag_b[0] = lo - 1032        = n - 8     （__hsub2, L143）
frag_b[1] = hi × (1/16) - 72 = n - 8     （__hfma2, L147）
q >> 8 (L447) → 剩余 n2,n3,n6,n7 同法
```

**命题 3**（magic-number dequant 正确性）。对任意 nibble $n \in [0, 15]$，LO 路径 $(1024 + n) - 1032 = n - 8$，HI 路径 $(1024 + 16n) \times (1/16) - 72 = n - 8$，两者都给出对称 INT4 值 $n - 8 \in [-8, 7]$。**证明**（代数，两步）：其一， $1024 + n$ 与 $1024 + 16n$（ $n \le 15$ 时 $\le 1264 < 2048$）都在 FP16 精确整数区，OR 运算无舍入；其二，LO 路径减 $1032 = 1024 + 8$ 得 $n - 8$；HI 路径乘 $1/16$ 得 $64 + n$，减 $72 = 64 + 8$ 得 $n - 8$。∎ 正例： $n = 5$ 两路径都给 $-3$； $n = 15$ 给 $7$； $n = 0$ 给 $-8$。全 16 码位 Python 独立重算核对（`struct` 解 FP16 位模式）：`0x6400` = 1024、`0x6408` = 1032、`0x2c00` = 0.0625、`0xd480` = −72，16 个 nibble 两路径均精确等于 $n - 8$。

magic number 的常数账一眼对照：`EX = 0x6400` = 1024（基准），`SUB = 0x6408` = 1032 = 1024 + 8（LO 路径偏置），`MUL = 0x2c00` = $2^{-4} = 1/16$（HI 路径还原 `<<4`），`ADD = 0xd480` = −72 = −64 − 8（HI 路径偏置，−64 来自 $1024/16$、−8 是 zero-point）。注释逐字（L138）：「We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`」——**对称 zero-point −8 融进常数**，不另花指令。这就是 magic-number 的全部本质：用位操作 + FP16 算术替代整数→浮点转换，dequant 落到 lop3（INT 管线）与 hsub2/hfma2（FP16 管线）上。

scale 施加（grouped quantization）：`scale(FragB, FragS, i)`（L155-159）用 `__hmul2` 把 dequant 后的 FP16 乘 per-group scale（L157-158）。per-column（`group_blocks == -1`）时 scale 不在 mainloop 逐权重乘，而在写回时一次性乘（L590-591 `if (group_blocks == -1) res = __hmul2(res, s[0])`）——**scale 粒度决定施加点**（6.1 §6 路径①对路径②的 kernel 侧兑现）。权重与 scale 离线 reshuffle 成理想访问布局（README L23「reshuffled offline ... directly dequantizing weights into tensor core organization」），故 nibble 抽取顺序（n0,n1,n4,n5 → q>>8 → n2,n3,n6,n7）恰好对齐 Tensor Core B fragment 布局。

> **坑**：以为 Marlin 用 bfe 抽 nibble。触发条件：把 unpack 指令族的泛指（lop3/prmt/bfe）当成 Marlin 的实际选择；现象：Marlin 实际用 lop3（L136-137）+ `>>8`（L447），无 bfe；规避：unpack 指令族是 PTX 提供的工具集，具体 kernel 选哪条看代码——Marlin 选 lop3 因为一条指令完成「掩码 + OR」两步，bfe 只抽字段还需另 OR magic number。

教学参照实现（Python，教学用、非优化实现），供练习 4 对照：

```python
import struct
def f16(bits): return struct.unpack('>e', struct.pack('>H', bits))[0]
def dequant_nibble(n):              # n: 0..15，单个 INT4 码（教学用）
    lo = f16(0x6400 | n) - 1032     # LO 路径 fp16(1024+n) - 1032
    hi = f16(0x6400 | (n << 4)) / 16 - 72   # HI 路径 fp16(1024+16n)/16 - 72
    assert lo == hi == n - 8        # 命题 3：两路径均给 n-8
    return lo
def dequant_i4(q, scale):           # q: int32 装 8 个 INT4 nibble
    vals = []
    for shift in (0, 8):            # 低 4 nibble 与 q>>8 的高 4 nibble
        qq = q >> shift
        lo = (qq & 0x000f000f) | 0x64006400   # (q&LO)|EX
        hi = (qq & 0x00f000f0) | 0x64006400   # (q&HI)|EX
        vals += [f16(lo & 0xffff) - 1032, f16(lo >> 16) - 1032]      # n0,n4 / n2,n6
        vals += [f16(hi & 0xffff) / 16 - 72, f16(hi >> 16) / 16 - 72]  # n1,n5 / n3,n7
    return [v * scale for v in vals]  # per-group scale；抽取序对齐 reshuffle
```

## 5. dequant↔MMA 重叠：双管线饱和

dequant 便宜了，为什么还要精心排指令顺序？因为 dequant 与 MMA 走不同管线，排不好就互相等。matmul lambda 注释逐字（L443）：「We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations」——m 维作内循环，是为了让 dequant 与 matmul 重叠。

机制（1.3 mma.sync 同步语义 recap）：mma.sync 发射后，**只有依赖其结果的指令才等写回**，无关指令可继续发射。下一轮 $j$ 的 dequant 与本轮 $j$ 的 mma 无寄存器依赖，编译器把 dequant 排进 mma 的执行阴影——Tensor Core 在后台算 mma，CUDA Core 同时做下一批 dequant。m 内循环（L456-459）让 `mma(j)` 之后紧跟 `dequant(j+1)`，两条管线交错饱和。README L22 逐字：「We carefully order dequantization and tensor core instructions to ensure that both GPU pipelines are well saturated and do not bottleneck each other」。

指令开销账（本章增量，逐指令计数）。每 8 个权重（一个 int32）每线程：`>>8` 1 条 + dequant(q) 4 条（2 lop3 + 1 hsub2 + 1 hfma2）+ scale0 2 条（hmul2）+ dequant(q>>8) 4 条 + scale1 2 条 = **13 条 CUDA-core 指令 + 2 条 mma**（ $T_M = 16$ 即 `thread_m_blocks = 1`）。摊每权重 1.625 条 CUDA-core + 0.25 条 mma。dequant 全落 lop3（INT 管线）与 hsub2/hfma2/hmul2（FP16 管线），**无 I2F 转换指令**——这是 magic-number 相对朴素「抽字段 + I2F」的指令节省。

这笔开销能不能藏住，看 MAC 预算（inference 3.3 §2 recap，不重推）：现代 GPU FLOP:byte ≈ 100–200，每个 4-bit 权重（0.5 B）「配得起」25–50 次 Tensor Core MAC（README L13-14 逐字「less than 25-50 (tensor core) multiply-accumulates per 4-bit quantized weight」）；而每个权重恰好参与 $B$ 次 MAC（每 token 一次）。 $B = 16$ 时每权重 16 MAC，落在 25–50 预算内——dequant 的 1.625 条指令开销可被 16 次 MAC 的 Tensor Core 时间盖住。 $B$ 再大（>25–50）MAC 预算耗尽，dequant 藏不住，加速塌——这就是 README 近理想 4× 只到 batch 16–32 的指令级根据（理论上界 76 是纯带宽账，未计 dequant 指令成本）。Marlin 论文把近理想 4× 推到 16–32、64–128 仍显著加速，集成 vLLM 端到端至多 2.8×（[Frantar et al., 2024](../references.md)，[A] 摘要口径，recap inference 3.3）。

> **坑**：Marlin 权重必须离线 reshuffle（`marlin.Layer.pack`），直接喂原始 GPTQ 张量不工作；且 tiling 与 K-split 按 token 数自适应选择，故 Marlin 非 batch-invariant（vLLM batch-invariant 模式绕开它改走固定顺序 matmul，inference 3.3 §2 坑注 recap，不重展开）。

## 6. W8A8/FP8：DeepGEMM fine-grained scaling 走读

W4A16 的加速上限卡在 batch 几十，大 batch 想同时省带宽和省算力怎么办？把激活也量化（W8A8），Tensor Core 直接按 FP8 算。5.2 §10 已走读 DeepGEMM 两级累加的概念与 promotion 段（L329-345），本节 recap 其骨架、增量是 **scale 读取路径与谓词选择**、**SM100 硬件 block-scaling 路径**、**DeepJIT 编译路径**。

两级累加 recap（5.2 §10，不重推）：fine-grained scale 沿 K 每 128 元素一变（A 侧 1×128、B 侧 128×128），无法折进事后全局 scale，必须分段累加——每段在 Tensor Core 无 scale 累加，段末乘 $`\mathrm{scale}_a \times \mathrm{scale}_b`$ 提升到 CUDA Core 的 FP32 主累加器，TC 累加器清零重起。代码骨架（`sm90_fp8_gemm_1d2d.cuh`）：注释「Accumulation for WGMMA or CUDA promotion」（L251），两级累加器 `float accum[WGMMA::kNumAccum], final_accum[...]`（L254），`BLOCK_K == 128` 硬断言（L56 逐字「Only support per-128-channel FP8 scaling」），FP8 wgmma m64nNk32（1.3 recap）每 128 K 深 4 条。

**scale 读取与谓词选择**（本章增量）。scale 经 shared 读入：B scale `scale_b_0/1 = ptx::ld_shared(smem_sfb + ...)`（L286-289），A scale `scale_a_0/1 = ptx::ld_shared(smem_sfa[stage_idx] + ...)`（L301-302）；注释逐字（L300）「all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results」——scale 读必须在 warpgroup_arrive 前，防下一调度块污染。promotion 段（L329-345）：`scale_0_0 = scale_a_0 * scale_b_0`（L331），`shifted_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0]`（L341-344）。**谓词 `predicate = ... or i < num_former_iters`（L340）按 fragment 选 scale**——输出 tile 跨两个 B scale 行时，不同 fragment 用不同 B scale（scale_0_0 对 scale_0_1）；注释逐字（L330）「making it as predicates is very important for performance, comparing to two loops」——用谓词而非两个循环，是性能关键。这把 6.1 §6 路径②（软件 mainloop 分段提升）落到了指令级。

**SM100 硬件 block-scaling 路径**（本章增量，6.1 §6 路径③）。README 逐字（L67-68）：「SM90 requires scaling factors in FP32 format.」/「SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single `torch.int`.」——SM90 scale 软件乘（路径②），SM100 scale 硬件消费（路径③）。sm100 kernel 签名含 `tensor_map_sfa`/`tensor_map_sfb`（`sm100_fp8_fp4_gemm_1d1d.cuh` L44-45），SFA/SFB 经 TMA 载入、tcgen05 block-scaling MMA 原生消费。UE8M0 是 8-bit 无符号纯指数（6.1 §4 recap），乘 scale = 指数加法、无尾数舍入；每 scale 1 字节，为 SM90 FP32 scale（4 字节）的 1/4。SF 布局变换表（`docs/scaling-factor-format.md` L89-91）：SM90 float32 转 MN-major、TMA-aligned、**不打包**；SM100 沿 MN broadcast 到 gran_mn=1 再**打包成 UE8M0 int32**、MN-major。SF 值须为精确 2 的幂（§1.2，device 断言 `(value & 0x807fffffu) == 0`）——纯指数约束的硬件兑现。gran_k：SM100 支持 32 或 128、SM90 仅 128（L29）；MXF4 断言 gran_k = 32、K-major（L84-85）。FP4 路径分化（6.1 §5 坑注 recap）：MXF4 packed UMMA_K = 64（4 倍）、MXF8F6F4 unpacked UMMA_K = 32（2 倍），`UMMA_K = kIsMXF4 ? 64 : 32`（L68）。

**DeepJIT 编译路径**（本章增量）。README 逐字（L3）：「All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation.」接口（`csrc/runtime/jit.hpp`）：`deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit`（L12），`init_jit` 设 library_root/include 与 `"cutlass-" + CUTLASS_VERSION`（L14-22），nvcc_flags 加 `--diag-suppress` 与 `-Wno-deprecated-declarations`（L26-29）。磁盘缓存默认 `$HOME/.dj`、`DG_JIT_CACHE_DIR` 可配（README L172）。为什么 serving 要 JIT：M 维（= batch）每步都变、prefill 的 M 千差万别，提前编译枚举不完 shape，JIT 按 shape 特化、编译成本被缓存摊掉（inference 3.3 §3 recap）。三种编译策略对照：Marlin `CALL_IF` 编译期枚举有用组合（§3）、CUTLASS manifest 编译期实例化（5.2 §9 recap）、DeepGEMM 运行时 JIT 按 shape 特化——前者实例化成本前置、运行期零编译，后者免安装期 nvcc、形状特化延迟到运行时。

## 7. grouped GEMM variable-M：GPU 侧调度增量

MoE 每个 expert 收到的 token 数 $M_j$ 不同，grouped GEMM 怎么在 GPU 侧消费这个 variable-M？inference 3.3 §4 已交付 contiguous/masked 两种 layout 与 padding 账（命题 2：每段非空且 $\le M_{\mathrm{blk}}$ 时浪费 $w = 1 - \mu/M_{\mathrm{blk}}$， $\mu = 32$ 给 75%），本节 recap 其语义、增量是**调度器怎么把 blockIdx 映到 (expert, m_block) 并跳过无效块**。

contiguous layout（训练 fwd / prefill，CPU 知 $M_j$）：token 按 expert 排序拼接 + offsets，每段对齐 GEMM M block size（README L78 `get_mk_alignment_for_contiguous_layout()`，默认 128，recap inference 3.3）。调度器（`scheduler/gemm.cuh`）：`grouped_layout[m_block_idx * BLOCK_M]` 给出该 M block 所属 expert 的 offset（L152），每 group 的 `num_m_blocks = ceil_div(grouped_layout[group], BLOCK_M)`（L195），blockIdx 经 cumsum 扫描映到 (group, m_block, n_block)（L190-204）。有效性判据 `is_computation_valid`：contiguous 用 `grouped_layout[...] >= 0`（L300，负值标记 padding 行）。

masked layout（decode + CUDA Graph，CPU 不知 $M_j$）：预分配静态容量 buffer + GPU 侧 mask 计数，kernel 只算有效部分（README L84-86）。调度器有效性判据（L301-302）：`m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx]`——**GPU 侧每 expert 计数**决定哪些 M block 有效；无效块只等 barrier、不算（`sm90_fp8_gemm_1d2d.cuh` L272 `is_computation_valid` 门控，else 分支 L349-354 只 wait/arrive）。这就是 inference 3.3「形状静态、内容动态」的调度器实现：buffer 形状 $[E/e, C, K]$ 不变可被 CUDA Graph 捕获（3.3 recap），kernel 读 GPU 计数跳无效块。典型输入 = DeepEP low-latency kernel 输出（README L86 逐字）。

> **口径注**：masked layout **不消除**块粒度浪费（有效行不足一个 tile 仍按整块算，inference 3.3 §4 tile 粒度推论 recap），消除的是 CPU 同步与动态 shape；padding 浪费账归 inference 3.3 命题 2，本章不重推。

## 8. Tensor Core 路径决策与边界

量化 kernel 的 Tensor Core 路径分两条，选型落在格式 × 硬件 × batch：

| 路径 | 格式 | 硬件 | Tensor Core 指令 | scale 施加 | 代表 |
| --- | --- | --- | --- | --- | --- |
| 原生低精度 MMA | FP8/FP4 | Hopper FP8 / Blackwell FP4 | wgmma m64nNk32 / tcgen05 MXF4 | 软件 promotion（SM90）/ 硬件 block-scaling（SM100） | DeepGEMM |
| dequant→FP16 MMA | INT4 存储 | Ampere/Ada/Hopper | mma.sync m16n8k16（FP16） | mainloop 内 hmul2（grouped）/ epilogue（per-column） | Marlin |

原生路径省 dequant 指令、吃低精度峰值（FP8 2 倍 / FP4 4 倍，6.1 §5）；dequant 路径算力仍按 FP16 峰值计，省的是权重读带宽（W4A16 weight-only），利好 memory-bound decode。INT4 在 Hopper/Blackwell 无原生 MMA（6.1 §5），故 W4A16 必走 dequant 路径——这是 Marlin lop3 magic-number 存在的硬件根据。

**边界声明**（代码 scope 片段级）：本章交付 dequant 融合带宽账（命题 1）、mixed-input tiling 强度（命题 2）、Marlin lop3/magic-number 指令级 dequant（命题 3）、双管线重叠指令账、DeepGEMM scale 读取/谓词/SM100/JIT 片段、grouped GEMM 调度增量。GEMM 五层实体与 collective mainloop 全量走读归 5.2；FA FP8 mainloop 归 6.3；量化算法（GPTQ/AWQ/SmoothQuant）归 inference 3.2；grouped GEMM padding 账与算子语义归 inference 3.3；serving 调度与引擎编排归 inference。格式位布局与逐格式吞吐归 6.1。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 融合对物化流量节省：物化 $`\mathrm{bytes}_{\mathrm{weight}}+4`$ 对融合 $`\mathrm{bytes}_{\mathrm{weight}}`$，比 8.76；物化比 bf16 慢 2.26× | 命题 1 | §2、练习 2 |
| 融合对 bf16 加速 3.88（inference 3.3 式 (1) recap）；临界 batch ≈76，实测近理想只到 16–32 | §2（inference 3.3 命题 1 recap） | §2、练习 3 |
| mixed-input tile 强度：A 常驻 L2 口径 $`2T_M/\mathrm{bytes}_{\mathrm{weight}}`$ = 62.06，A 走 HBM 41.8；比 bf16 3.88 对 2.94 | 命题 2 | §3、练习 2 |
| Marlin tile：THREADS 256/STAGES 4/96KB；batch≤16 用 16×128×128；B evict_first、A/C 常驻 L2 | §3 走读（L705-707/L755-761/L75-76） | §3 |
| lop3 LUT 0xea = (a&b)|c；magic-number EX=1024/SUB=1032/MUL=1/16/ADD=−72；LO/HI 两路径均给 n−8 | 命题 3 | §4、练习 1 |
| Marlin 用 lop3 非 bfe，第二 nibble 用 >>8；zero-point −8 融进 SUB/ADD；无 I2F 指令 | §4（L136-137/L447/L138） | §4、练习 1 |
| dequant 指令账：每 8 权重 13 CUDA-core + 2 mma；每权重 1.625 + 0.25；B=16 落 25–50 MAC 预算内 | §5 计数 | §5、练习 3 |
| m 内循环促 dequant↔MMA 重叠（L443）；mma.sync 无关指令可发射进执行阴影 | §5（1.3 同步语义 recap） | §5 |
| DeepGEMM 两级累加 per-128 promotion（5.2 §10 recap）；scale 读先于 warpgroup_arrive（L300）、谓词选 scale（L340） | §6 | §6 |
| SM90 FP32 scale 软件乘 / SM100 packed UE8M0 硬件消费（README L67-68）；MXF4 UMMA_K 64 / MXF8F6F4 32 | §6（6.1 §6 路径②③） | §6 |
| DeepJIT 运行时编译 + 磁盘缓存 $HOME/.dj；对照 Marlin CALL_IF 编译期枚举 / CUTLASS manifest | §6（jit.hpp L12-32） | §6 |
| grouped GEMM：contiguous offsets 映 blockIdx（L152）/ masked GPU 计数跳无效块（L301-302）；不消除块粒度浪费 | §7（inference 3.3 §4 recap） | §7 |
| Tensor Core 两路径：原生低精度 MMA（FP8/FP4）vs dequant→FP16 MMA（INT4 存储）；INT4 在 Hopper 无原生 MMA | §8 表（6.1 §5） | §8 |

## 练习

1.（计算 · [10]）magic-number 核对（命题 3）：(a) 用 FP16 位模式解码 `0x6400`/`0x6408`/`0x2c00`/`0xd480`；(b) 取 $n = 5$ 算 LO 路径 $(1024+n)-1032$ 与 HI 路径 $(1024+16n)/16-72$；(c) 核对 lop3 LUT `(0xf0&0xcc)|0xaa` 的值与真值表。校验：(a) 1024 / 1032 / 0.0625 / −72（§4）；(b) 两路径均 −3；(c) `0xea = (a&b)|c`（8 组合逐位）。
2.（推导 · [20]）补全命题 1 与命题 2：(a) 证命题 1（物化三遍对融合一遍流量），代入 $g=128$ 算比值与对 bf16 倍数；(b) 证命题 2（mixed-input tile 强度两口径），代入 $T_M=16/T_N=128$ 算 A-L2 与 A-HBM 强度及对 bf16 比；(c) 说明为什么小 $T_M$ 下 A 常驻 L2 是带宽账成立的条件。校验：(a) 4.515625/0.515625 = 8.76、2.26×（§2）；(b) 62.06/41.8、3.88/2.94（§3）；(c) A 字节占 tile 流量 33%（§3）。
3.（计算 · [30]）dequant 指令账与 MAC 预算：(a) 逐指令计数每 8 权重的 CUDA-core 与 mma 条数（§4 走读），摊每权重；(b) 算 $B=16$ 时每权重 MAC 数，对照 README 25–50 预算判断 dequant 能否藏住；(c) 代入 inference 3.3 命题 1 算 $B_{\mathrm{crit}}$，再算 $B=128$ 时 INT4 侧 compute-bound 翻转后的加速（bf16 带宽时间 10.0 μs 对 INT4 计算时间）。校验：(a) 13 + 2、1.625 + 0.25（§5）；(b) 16 MAC 落预算内（§5）；(c) $B_{\mathrm{crit}}\approx76$、 $B=128$ 加速 ≈2.31（inference 3.3 §2 实例）。
4.（代码 · [30]）教学用 Python dequant 参照实现（非优化实现）：(a) 先预测 magic-number 路径与整数直解路径（nibble − 8 再乘 scale）是否逐位精确相等，再写代码验证；(b) 实现 §4 的 `dequant_i4`（lop3 掩码 + OR magic number + FP16 减/乘加），对全部 16 nibble × 若干 int32 打包与整数路径对照；(c) 加 $g=128$ per-group scale，说明为什么无需 I2F 指令。校验：fp16 小整数运算两路径精确相等（容差 0，命题 3 保证 $1024+n$ 与 $1024+16n$ 落 FP16 精确整数区）；与 §4 命题 3 的 $n-8$ 一致。

## 参考

- [Frantar et al., 2024](../references.md)：MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models，arXiv:2408.11743，[A]（key: marlin）+ IST-DASLab/marlin，版本锚 commit 1f25790bdd49（master，2024-09-04，2026-11-25 fresh tarball 核实）——marlin/marlin_cuda_kernel.cu（mma.sync m16n8k16 L98、ldmatrix m8n8.x4 L111、lop3.b32 L118-126、dequant LO/HI/EX magic-number L131-152、scale __hmul2 L155-159、matmul m-inner-loop L443、>>8 L447、split-K shared reduce L463-465、cp.async.cg L61/evict_first L75-76、THREADS/STAGES/SHARED_MEM L705-707、tile 选择 L755-761、CALL_IF L801）；README（4x batch 16-32 L7-8、25-50 MAC L13-14、双管线饱和 L22、reshuffle L23、3.87x L41-43、CC≥8.0 L65）
- [DeepSeek-AI, DeepGEMM](../references.md)：deepseek-ai/DeepGEMM repo（key: deepgemm），教学参照 + 官方实现，版本锚 commit 5081f8513af5（main，2026-10-20，2026-11-25）——sm90_fp8_gemm_1d2d.cuh（BLOCK_K==128 L56、accum/final_accum L254、scale_b L286-289、scale_a L300-302、wgmma L313、Promote with scales L329-345、谓词 L340、is_computation_valid 门控 L272/L349-354）；sm100_fp8_fp4_gemm_1d1d.cuh（tensor_map_sfa/sfb L44-45、MXF4 注释 L51-52、UMMA_K L68、gran_k L84-85/L94-95）；csrc/runtime/jit.hpp（LazyInit L12、init_jit L14-22、nvcc_flags L26-29）；scheduler/gemm.cuh（contiguous offset L152、num_m_blocks L195、is_computation_valid L300-302）；README（DeepJIT L3、SM90 FP32/SM100 UE8M0 L67-68、contiguous L78、masked L84-86、DG_JIT_CACHE_DIR L172）；docs/scaling-factor-format.md（gran_k L29、SF 2 的幂断言 L33、变换表 L89-91）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——lop3.b32（三输入查找表位运算）、prmt.b32（字节置换）、bfe（位域提取）；mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32（1.3 recap 源）；cp.async.cg.shared.global.L2::cache_hint + createpolicy.fractional.L2::evict_first（4.4 recap 源）
