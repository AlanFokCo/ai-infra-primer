# 1.3 Tensor Core 与 MMA 微架构（Tensor Core and MMA Microarchitecture）

> 一条指令怎么做到 4096 个 FLOP，Tensor Core 的 16 倍吞吐为什么不需要 16 倍的寄存器端口？本章交付 MMA 指令几何（mma.sync m16n8k16 / wgmma m64nNk16 / tcgen05）、操作数来源与 FP32 累加器、三代数据流、格式支持矩阵，以及 MMA 吞吐推导（指令几何 FLOPs ÷ 每周期吞吐 = 稳态发射间隔）与操作数复用带宽账 | 前置：1.1（式 (1)(2) 峰值反推、 $c_{\mathrm{BF16}} = 4096$、逐精度 clock）、1.2（命题 1 单发射、§3 寄存器带宽账 512 B/warp FMA、式 (3) occupancy 上界、RF 256 KB）；下游 6.1（格式位布局账 recap 本章指令几何）、5.2（GEMM 走读 MMA atom）、4.2（ldmatrix/swizzle）、4.4（TMA/cp.async）、1.4（wgmma 延迟实测与异步流水） | 锚点：PTX ISA MMA/wgmma/tcgen05 节（[NVIDIA, PTX ISA](../references.md)，[C] 官方规格披露·一手权威）、Hopper whitepaper V1.04（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、Blackwell Technical Brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、H100/Blackwell datasheet（[NVIDIA, datasheet](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）、CUTLASS MMA atom（[NVIDIA, CUTLASS](../references.md)，教学参照 + 官方实现）——官方规格 2026-11-25 一手核实

## 1. 一条指令 4096 个 FLOP：本章口径与两层纪律

1.1 §3 留了一个待核对的口径：一条 mma.sync m16n8k16 含 $16 \times 8 \times 16 \times 2 = 4096$ 个 FLOP，按每 Tensor Core 每周期 1024 个 BF16 FLOP 折 4 个周期——这个「4 周期」是稳态吞吐还是单条延迟？Tensor Core 每周期 1024 个 FLOP 又是怎么来的，为什么 CUDA Core 一个 lane 每周期只有 2 个？本章回答三个问题：MMA 指令的几何与操作数路径长什么样？从指令几何怎么推出稳态吞吐与芯片级峰值闭环？Tensor Core 相对 CUDA Core 的吞吐优势，机制上靠什么撑住？

**本章口径**（章脊锁定）：主口径 **H100 SXM，BF16 dense，TC-boost 1830 MHz**（沿 1.1， $n_{\mathrm{SM}} = 132$、 $c_{\mathrm{BF16}} = 4096$ FLOPs/cycle/SM = 4 Tensor Core × 1024）；B200 第 5 代作对照（HGX B200 SKU，BF16 dense 2250 TFLOPS，沿 1.1 §4）。指令几何一律用字面数 $m \times n \times k$（spec §4.3）；`m64nNk16` 里的 N 是 PTX 指令字面占位（ $N \in \lbrace 8, 16, \ldots, 256 \rbrace$，8 的倍数），**不是参数量 $N$**——本章凡 N 均指指令字面维。累加器精度口径：FP32 accumulate（指令语义层）。

**两层纪律**（1.3 ↔ 6.1，spec §4.2）：本章只交付**指令几何层**——指令形状、操作数来源、累加器位置、吞吐推导、格式「支持与否」。格式的**位布局账**（sign/exponent/mantissa 表、动态范围、量化误差、逐格式吞吐倍数、MX 族与 NVFP4 的 scale 粒度）归 6.1，届时 recap 本章几何为起点。所以本章格式矩阵只标「哪代支持哪个格式」，不拆位布局；FP32 累加只标「指令语义如此」，不推「为什么必须 FP32」的精度账（归 6.1）。代码 scope = 机制片段级（PTX 指令例证），CUTLASS MMA atom 的模块全量走读归 5.2。

Tensor Core 代际与指令族（代际映射沿 T1 一手核实；指令族 ↔ 架构按 PTX target ISA 注，V100/A100 列标反推）：

| 代际 | 代表 SKU / 架构 | 指令族（PTX） | 关键增量 | 出处 |
| --- | --- | --- | --- | --- |
| 第 1 代 | V100 / Volta（sm_70） | mma（warp 级，m8n8k4） | FP16 乘加阵列首次落地 | PTX target sm_70（反推） |
| 第 2 代 | Turing（sm_75） | mma（m8n8k16 等） | INT8/INT4 整数路径 | PTX target sm_75（反推） |
| 第 3 代 | A100 / Ampere（sm_80） | mma.sync（m16n8k*） | TF32/BF16/FP64 TC + cp.async | PTX target sm_80（反推） |
| 第 4 代 | H100 / Hopper（sm_90a） | wgmma.mma_async（warp-group） | 异步 warp-group + TMA + FP8 | [NVIDIA, Hopper whitepaper](../references.md) + PTX，[C] |
| 第 5 代 | B200 / Blackwell（sm_100a） | tcgen05.mma（+ TMEM） | Tensor Memory + FP4/FP6 + 2-SM 协作 | [NVIDIA, Blackwell brief](../references.md) + PTX，[C] |

**口径注**：V100/A100 两列的格式与指令族按 PTX mma 家族 target architecture（sm_70/sm_75/sm_80）反推，非逐代 whitepaper 一手（A100/V100 whitepaper 不在本波引用白名单）；Hopper/Blackwell 两列是官方规格披露（[C]）。第 2 代 Turing 非数据中心主线，本表沿 spec 四代口径（V100/A100/H100/B200）列出、Turing 仅作代际编号衔接。B200 双 die 封装与 NV-HBI 见 1.1 §5（**NV-HBI 是 die↔die，NVLink-C2C 是 GB200 Grace↔Blackwell，勿混**——沿 1.1 坑注）。

## 2. mma.sync m16n8k16：warp 级同步指令与 fragment 布局

一条 mma.sync 指令的操作数从哪来、占多少寄存器？PTX ISA 逐字给出指令语法（[NVIDIA, PTX ISA](../references.md)，[C]，2026-11-25 核实）：

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  d, a, b, c;
```

语义： $D = A \times B + C$，其中 A 是 $16 \times 16$、B 是 $16 \times 8$、C/D 是 $16 \times 8$；`.row.col` 指 A 行主序、B 列主序；四个类型后缀依次为 D/A/B/C 的元素类型——A、B 为 BF16，C、D 为 FP32（**FP32 累加**）。f16 版本同形（`.f32.f16.f16.f32`）。`.sync` 表示 warp 内 32 线程同步执行、`.aligned` 表示 warp 内全线程参与；指令是**同步**的——warp 发射后须等结果写回目的寄存器才能继续依赖它的下一条指令（依赖追踪与 stall 归 1.4）。

所谓 fragment（矩阵片段），是指一条 MMA 指令的矩阵操作数按固定布局摊到 warp 内 32 个线程的寄存器上、每线程只持有自己那份。PTX ISA 矩阵 fragment 布局给出 m16n8k16（A/B 为 f16/bf16、C/D 为 f32）的每线程寄存器占用：

| 操作数 | 形状 | 元素类型 | 每线程寄存器 | 每 warp 字节 | 摊到 32 线程 |
| --- | --- | --- | --- | --- | --- |
| A | $16 \times 16$ | BF16（2 B） | 4 个 32-bit | 512 B | 256 元素 = 每线程 8 个 |
| B | $16 \times 8$ | BF16（2 B） | 2 个 32-bit | 256 B | 128 元素 = 每线程 4 个 |
| C | $16 \times 8$ | FP32（4 B） | 4 个 32-bit | 512 B | 128 元素 = 每线程 4 个 |
| D | $16 \times 8$ | FP32（4 B） | 4 个 32-bit | 512 B | 128 元素 = 每线程 4 个 |

正例：A 的 256 个 BF16 元素 = 512 B，摊到 32 线程 = 每线程 16 B = 4 个 32-bit 寄存器（每寄存器装 2 个 BF16），与 PTX fragment 布局逐字一致。反例：若误以为每线程持有整个 $16 \times 16$ 的 A（256 元素 × 2 B = 512 B/线程），寄存器占用会高估 32 倍——fragment 的本质是**矩阵摊到 warp**，不是每线程复制全矩阵。

数据流（mma.sync 路径，A/B 须先在寄存器）：

```
mma.sync（Ampere/Ada，warp 级同步）操作数路径
  HBM ──cp.async/TMA──► shared memory
                           │ ldmatrix（warp 级矩阵加载，4.2）
                           ▼
        register fragment：A 4 regs / B 2 regs / C 4 regs（每线程）
                           │
                           ▼
        Tensor Core（每 processing block 1 个，1.2 Fig 7）
                           │  D = A×B + C（FP32 累加）
                           ▼
        register accumulator：D 4 regs（每线程，FP32）
                           │  下一条 mma.sync 把 D 当 C 读回（k 方向累加）
                           └──► 循环 k 步 → epilogue 写回
```

ldmatrix 把 shared 里的矩阵按 fragment 布局一次性搬进寄存器（swizzle 避 bank conflict 归 4.2）；cp.async/TMA 把 HBM 数据异步搬进 shared（归 4.4）。所以 mma.sync 的完整链路是「HBM → shared → register fragment → Tensor Core → register accumulator」，A/B 必须经寄存器中转——这个中转的带宽代价在 §4 算账。累加器 D 留在寄存器里，k 方向下一步把它当 C 读回，形成寄存器内累加循环。

## 3. MMA 吞吐推导：指令几何 ÷ 每周期吞吐

4096 个 FLOP 的一条指令，Tensor Core 要算几个周期？推导路线分三步：先写一条指令的 FLOPs（几何计数），再除以每周期吞吐得稳态发射间隔（量纲对账），最后用实测延迟与芯片级峰值两路闭环。关键一步是分清**稳态发射间隔**（吞吐口径）与**单条延迟**（latency 口径）——两者差一个流水线填充，混用即把峰值算错一倍。

一条 MMA 指令做 $m \times n$ 个输出元素、每个元素沿 k 维做 k 次乘加（每次乘加 2 FLOPs），故指令 FLOPs：

$$
F_{\mathrm{instr}} = 2 \cdot m \cdot n \cdot k \qquad (1)
$$

**符号**： $F_{\mathrm{instr}}$ = 一条 MMA 指令的 FLOPs； $m, n, k$ = 指令几何字面维（输出行/列/归约维）。全数字代入：m16n8k16 得 $2 \times 16 \times 8 \times 16 = 4096$ FLOPs；wgmma m64n128k16 得 $2 \times 64 \times 128 \times 16 = 262144$ FLOPs；m64n256k16 得 $524288$ FLOPs。

**命题 1**（MMA 稳态吞吐）。设执行单元每周期吞吐为 $c$（FLOPs/cycle），则一条几何为 $m \times n \times k$ 的 MMA 指令在稳态下的发射间隔为

$$
t_{\mathrm{issue}} = \frac{F_{\mathrm{instr}}}{c} = \frac{2 m n k}{c} \qquad (2)
$$

**证明**（量纲对账 + 结构恒等）： $[F_{\mathrm{instr}}] = \mathrm{FLOPs}$， $[c] = \mathrm{FLOPs}/\mathrm{cycle}$，相除得 cycle——即「产出这么多 FLOP 需要多少周期」。稳态下执行单元每周期产出 $c$ 个 FLOP，一条指令的 $F_{\mathrm{instr}}$ 个 FLOP 恰占用 $F_{\mathrm{instr}}/c$ 个周期的单元时间，故相邻两条指令的发射间隔（吞吐倒数）等于此值。这是结构恒等式而非拟合：吞吐本来就定义为每周期产出，发射间隔是它的倒数按指令 FLOPs 缩放。∎

**口径注**： $c$ 取**执行该指令的单元集合**的每周期吞吐。mma.sync 由单个 warp 发射、用 1 个 Tensor Core， $c = c_{\mathrm{TC}} = 1024$（1.1 反推）；wgmma 由 warp-group（4 warp）发射、协同用全 SM 的 4 个 Tensor Core， $c = c_{\mathrm{SM}} = 4096$。式 (2) 给的是**稳态发射间隔**（连续独立指令的吞吐），不是单条延迟——单条延迟含流水线填充，一般更长（实测见下）。

全数字代入（H100 SXM BF16 主口径）：

- mma.sync m16n8k16： $t_{\mathrm{issue}} = 4096/1024 = 4$ cycles（每 Tensor Core）。4 个 processing block 各 1 个 warp 每 4 周期发一条，SM 合计 $4 \times 4096/4 = 4096$ FLOPs/cycle，与 1.1 反推 $c_{\mathrm{BF16}} = 4096$ 闭环。
- wgmma m64n128k16： $t_{\mathrm{issue}} = 262144/4096 = 64$ cycles（每 SM，warp-group 用全 4 TC）。

实测核对（[Luo et al., 2024](../references.md)，arXiv:2402.13499，H800 微基准 Table IV，[B] **实测口径**）：wgmma m64n128k16 dense 单条延迟 **128 cycles**、sparse SS **144 cycles**。注意 128 是**延迟**（发射到完成），64 是**稳态发射间隔**（反推）——两者差 2× 正是流水线填充/排空的量级：异步语义（§5）让后续指令在前一条完成前就发射，稳态完成率 = 发射率 = 1/64，而单条从发射到写回要走 128 周期。具体流水线深度属闭源细节，本书不推定，只引实测延迟与反推间隔两个口径并标其差。

> **坑**：把实测延迟当稳态间隔。触发条件：拿 wgmma 128 cycles 延迟直接当发射间隔代式 (2) 反推吞吐；现象： $c_{\mathrm{SM}} = 262144/128 = 2048$ FLOPs/cycle，芯片级 BF16 算成 $132 \times 2048 \times 1.83 \times 10^{9} = 494.7$ TFLOPS，较官方 989.4 低估一半；规避：延迟（latency）与吞吐（throughput）是两个口径，稳态吞吐用几何反推（式 (2)）、延迟用实测 [B]，异步流水把两者解耦（§5）。

sparse 一笔（实测口径）：2:4 结构化稀疏让有效 FLOPs 翻倍（k 维压缩一半），但实测延迟 144 对 dense 128（+12.5%，metadata 路径开销），净加速 $2 \times 128/144 = 1.78$ 倍而非理论 2 倍。这是「稀疏峰值 = dense × 2」的 datasheet 口径（1.1 §6）与实测可用加速之间的差——datasheet 印的是理论几何翻倍，实测要扣 metadata 开销。

Tensor Core vs CUDA Core 的 FLOPS 比（recap 1.1 §3，不重推）：每周期吞吐比 $c_{\mathrm{BF16}}/c_{\mathrm{FP32}} = 4096/256 = 16$ 倍；芯片级再叠 clock 比 $1830/1980 = 0.924$，合 $16 \times 0.924 = 14.8$ 倍（ $989.4/66.9 = 14.8$）。1.1 交付了这个比值的芯片级反推，本章 §4 回答它的**机制来源**：不是「lane 多 16 倍」这么简单，而是操作数复用让 16 倍 FLOPs 不需要 16 倍寄存器带宽。

## 4. 操作数复用账：16 倍 FLOPs 为什么不需要 16 倍寄存器端口

先回答一个更基本的问题：Tensor Core 每周期 1024 个 FLOP，是 CUDA Core 一个 lane（2 个 FLOP）的 512 倍，寄存器堆喂得动吗？1.2 §3 算过寄存器 operand 带宽需求上界 2048 B/cycle/SM。若 Tensor Core 真要 16 倍 FLOPs 配 16 倍 operand 带宽，寄存器堆早就崩了。答案藏在 MMA 指令内部的**操作数复用**：A 的每个元素被 n 列复用、B 的每个元素被 m 行复用，一次寄存器读喂多次乘加。

推导路线：先数一条 mma.sync 的寄存器读写字节（fragment 布局），除以指令 FLOPs 得每 FLOP 字节，再与标量 FMA 对比，最后折成每周期 operand 需求与 1.2 上界对账。关键一步是复用计数——A/B 读一次、在阵列内用多次。

一条 mma.sync m16n8k16（BF16）的寄存器流量：读 A 512 B + B 256 B + C 512 B = 1280 B，写 D 512 B，合计 1792 B（§2 fragment 表）。每 FLOP 寄存器字节：

$$
\left(\frac{\mathrm{bytes}}{\mathrm{FLOP}}\right)_{\mathrm{MMA}} = \frac{1792}{4096} = 0.4375 \ \mathrm{B/FLOP} \qquad (3)
$$

对比标量 warp FMA（1.2 式 (1)：512 B / 64 FLOPs）： $512/64 = 8$ B/FLOP。比值 $8/0.4375 = 18.3$——MMA 每 FLOP 的寄存器带宽需求只有标量 FMA 的约 1/18。机制：标量 FMA 每 2 个 FLOP 要读 3 个 operand（a、b、c 各一次），operand 不复用；MMA 把 A 的一行广播给 n 列、B 的一列广播给 m 行，1792 B 的寄存器读喂了 4096 个 FLOP。

**命题 2**（MMA 流的寄存器 operand 需求不超过满 FMA 流）。在 H100 单发射口径下，每 processing block 持续发射 mma.sync m16n8k16（每 4 周期一条）的寄存器 operand 需求为 448 B/cycle，低于持续发射标量 FMA（每周期一条）的 512 B/cycle。**证明**（计数）：MMA 流每 4 周期读写 1792 B，折 $1792/4 = 448$ B/cycle；FMA 流每周期读写 512 B（1.2 式 (1)）。 $448 \lt 512$，故 MMA 流需求更低。全 SM 4 个 block：MMA 流 $4 \times 448 = 1792$ B/cycle，低于 1.2 式 (2) 的满 FMA 上界 2048 B/cycle。∎

这条命题解释了 §3 的 16 倍比值为什么物理上可行：Tensor Core 的 16 倍 FLOPs 不仅不需要 16 倍寄存器端口，反而比满 FMA 流**更省** operand 带宽（448 对 512 B/cycle/block）。乘加阵列把 operand 读摊销到阵列内多次复用，寄存器堆端口预算（1.2 §3）因此撑得住。这是「Tensor Core 为什么快」的机制答案的一半；另一半是阵列本身的并行乘加单元数（闭源，不推定）。

wgmma 把复用推到极致：B（乃至 A）直接从 shared 经描述符喂 Tensor Core，**跳过寄存器中转**。wgmma m64n128k16（BF16，A/B 均 shared）的 shared 流量：A $64 \times 16 \times 2 = 2048$ B + B $16 \times 128 \times 2 = 4096$ B = 6144 B/指令，稳态 64 周期折 $6144/64 = 96$ B/cycle/SM。对比 shared 峰值 128 B/cycle/SM（1.2 §4，32 banks × 4 B），占 75%——**wgmma 的 shared operand 流恰好落在 shared 带宽预算内，留 25% 余量**。寄存器侧则只剩累加器读写（A/B 不进寄存器），每 FLOP 寄存器字节进一步降到约 0.25 B/FLOP（C 读 + D 写 = $2 \times 64 \times 128 \times 4 = 65536$ B / 262144 FLOPs）。这解释了 Hopper 为什么把 B 操作数路径从寄存器改到 shared 描述符：省寄存器、省 ldmatrix 指令、把 operand 带宽压力从寄存器堆转给 shared。

## 5. wgmma：warp-group 异步语义

mma.sync 是同步的——warp 发射后 stall 等结果（1.4 的 execution dependency）。Hopper 的 wgmma 改成异步，这改变了什么？

所谓 warp-group，是指 4 个连续 warp（128 线程）组成的协作单位，wgmma 由整个 warp-group 协同发射、协同持有累加器。PTX ISA 逐字（[NVIDIA, PTX ISA](../references.md)，[C]）：`wgmma.mma_async.m64nNk16`（ $N \in \lbrace 8, 16, \ldots, 256 \rbrace$）；TF32 为 `m64nNk8`，FP8/FP6/FP4 族为 `m64nNk32`（kind::f8f6f4）。操作数来源：**A 可来自寄存器或 shared 描述符，B 只能来自 shared 描述符**；累加器 D 在寄存器（FP32），摊到 warp-group 的 128 线程。

异步语义三件套（PTX wgmma 节，[C]）：

- `wgmma.mma_async` 发射后**立即返回**，不等结果——Tensor Core 在后台算，warp-group 可继续发射后续指令（地址计算、TMA、甚至 softmax 的 CUDA Core 工作）。
- `wgmma.commit_group` 把已发射的 wgmma 打包成一个 group；`wgmma.wait_group N` 等到至多 N 个 group 未完成——这是异步完成的同步点，替代了同步指令的寄存器 scoreboard（1.4）。
- 读 shared 操作数前须保证写入对 async proxy 可见（`fence.proxy.async`；TMA 写入本身走 async proxy，机制归 4.4）。

异步的收益是**跨管线重叠**：Tensor Core 算 MMA 的同时，CUDA Core 算 softmax、TMA 搬下一块数据——三条管线并行。这是 FlashAttention-3 warp specialization / pingpong 的硬件基础（指令级走读归 6.3，本章只立机制）。同步 mma.sync 做不到这点：warp 被 MMA 占住，CUDA Core 工作只能串行排在后面。

累加器寄存器压力（连接 1.2 occupancy）。wgmma m64nNk16 的 FP32 累加器 D 共 $64 \times N$ 个元素，摊到 128 线程：

$$
R_{\mathrm{acc}} = \frac{64 \times N}{128} = \frac{N}{2} \ \text{regs/thread} \qquad (4)
$$

**符号**： $R_{\mathrm{acc}}$ = 累加器占用的每线程寄存器数（仅 D，不含 A 操作数与其他变量）。代入： $N = 128$ 得 64 regs/thread； $N = 256$ 得 128 regs/thread——后者已占每线程上限 255（1.2 Table 4）的一半。

**命题 3**（wgmma 累加器寄存器压力 → occupancy 上界）。设一个 thread block 恰含一个 warp-group（128 线程， $w_b = 4$），累加器是寄存器占用的主项（理想化下界 $R_{\mathrm{th}} = R_{\mathrm{acc}} = N/2$， $S_{\mathrm{sh}} = 0$），则 1.2 式 (3) 的 register 约束给出 occupancy 上界随 N 反比下降。**证明留作练习 3**（计数 + 代入 1.2 式 (3)）。全数字落点： $N = 128$（ $R_{\mathrm{th}} = 64$）→ register 项 $\lfloor 65536/(64 \times 128) \rfloor = 8$ blocks， $W_{\mathrm{active}} = 32$， $\mathcal{O} = 0.5$； $N = 256$（ $R_{\mathrm{th}} = 128$）→ register 项 4 blocks， $W_{\mathrm{active}} = 16$， $\mathcal{O} = 0.25$。N 翻倍，occupancy 上界减半——与 1.2 案例 C1→C2 的 register-binding 模式同构。

这条压力正是 Blackwell 把累加器搬出寄存器的动机（§6）：大 N 的 wgmma 累加器吃掉半个寄存器堆，逼低 occupancy；若累加器有专属存储，寄存器就能腾出来给更多 warp 或更高 ILP。

> **提点**：异步 MMA 是「机制与策略分离」的又一例——硬件提供异步发射 + commit/wait 同步点（机制），具体重叠多少 CUDA Core 工作、几个 group 在飞由 kernel 调度（策略，FA3 的 warp specialization 归 6.3）。删掉这条类比，上文的异步语义与式 (4) 依然完整。

## 6. tcgen05 与 Tensor Memory：累加器搬出寄存器

Blackwell 第 5 代 Tensor Core 的指令族是 tcgen05（PTX ISA 逐字：`tcgen05.mma` / `.sp` / `.ws`，[NVIDIA, PTX ISA](../references.md)，[C]）。最大的结构变化是引入 **Tensor Memory（TMEM）**——一块专供 Tensor Core 累加器使用的片上存储。PTX ISA 逐字（sm_100a/f）：「512 columns and 128 rows per CTA, with each cell being 32-bits」，即

$$
512 \times 128 \times 4 \ \mathrm{B} = 262144 \ \mathrm{B} = 256 \ \mathrm{KB/CTA} \qquad (5)
$$

256 KB——恰等于一个 SM 的整个寄存器堆容量（1.2 §3，RF 256 KB）。Blackwell 等于给 MMA 累加器另配了一个与寄存器堆等大的专属存储层。累加器 D 驻 TMEM（不再占寄存器），A/B 经 shared 描述符喂入，epilogue 用 `tcgen05.ld` 把 TMEM 结果搬回寄存器再写回。寄存器压力（命题 3）由此解除：大 N 不再逼低 occupancy。

TMEM 布局与数据流（Blackwell 路径）：

```
tcgen05（Blackwell，sm_100a）操作数路径
  shared memory ──描述符──► A / B
                              │
                              ▼
              Tensor Core（第 5 代）
                              │  D = A×B + C（FP32 累加）
                              ▼
        Tensor Memory（TMEM）：512 列 × 128 行 × 32-bit = 256 KB/CTA
                              │  tcgen05.ld（epilogue）
                              ▼
        register ──► 写回 HBM
  对比 wgmma：累加器从 register 搬到 TMEM，寄存器腾出给 occupancy/ILP
```

**2-SM 协作**（PTX `.cta_group::1` / `::2`，[C]）：`cta_group::2` 让两个相邻 SM 上的 CTA 协同执行一条 MMA——两个 SM 的 Tensor Core 一起算、各持一半累加器在自己的 TMEM，有效 M 维翻倍。这建立在 Hopper thread block cluster / DSMEM 的物理相邻 SM 分组之上（1.1 §2 提点、3.1 cluster 语义）。精确形状清单与单线程发射语义等细节 ⟨verify W3 PTX/CUTLASS sm100 atom⟩（T1 已核实 cta_group::1/::2 存在与 TMEM 几何，更细的 operand 选项与形状表留 W3 走读销项）。

block scaling 一笔：`tcgen05.mma ... kind::mxf4nvf4` 支持 NVFP4 的块缩放（per-16 UE4M3 scale + global FP32），这是 Blackwell FP4 吞吐翻倍的指令基础。**NVFP4 非 OCP MX v1.0 格式**（MX 族 per-32 E8M0 scale）——位布局与 scale 粒度账归 6.1（recap inference 3.2），本章只标指令族支持。FP4/FP6 的代际吞吐倍数（FP8 2×、FP4 4× vs BF16）同样归 6.1 逐格式账。

## 7. 格式支持矩阵：哪代支持哪个格式

哪个格式从哪一代起有硬件支持？格式支持矩阵如下（✓ = 官方规格表列出或 PTX 指令族支持；△ = PTX 支持但官方表未列独立峰值；— = 不支持/未列。Hopper/Blackwell 列取官方表，V100/A100 列按 PTX target arch 反推）：

| 格式 | V100（1st） | A100（3rd） | H100（4th） | B200（5th） | 出处 |
| --- | --- | --- | --- | --- | --- |
| FP16 | ✓ | ✓ | ✓ | ✓ | PTX sm_70+ / wp Table 3 / brief Table 1 |
| BF16 | — | ✓ | ✓ | ✓ | PTX sm_80 / wp / brief |
| TF32 | — | ✓ | ✓ | ✓ | PTX sm_80 / wp 494.7 / brief |
| FP8（E4M3/E5M2） | — | — | ✓ | ✓ | wgmma f8f6f4 / wp 1978.9 / brief |
| FP6 | — | — | △ | ✓ | wgmma f8f6f4 含 e2m3/e3m2 / brief Table 1 |
| FP4（E2M1） | — | — | △ | ✓ | wgmma f8f6f4 含 e2m1 / brief + datasheet 9000 |
| INT8 | — | ✓ | ✓ | ✓ | PTX sm_75/80 / wp 1979 TOPS / brief |
| INT4 | — | ✓ | 未列 | 未列 | PTX sm_75/80 s4 / wp、brief 表未列 |
| FP64（TC） | — | ✓ | ✓ | ✓ | PTX sm_80 f64 / wp 66.9 / brief |

**口径注**（三层）：其一，H100 列取 whitepaper Table 3 逐精度峰值行（FP64 TC 66.9 / TF32 494.7 / BF16-FP16 989.4 / FP8 1978.9 / INT8 1979 TOPS，[C]）；B200 列取 brief Table 1 数据类型（FP64/FP32/TF32/FP16/BF16/FP8/INT8/FP6/FP4，[C]）+ datasheet FP4 9000 dense。其二，FP6/FP4 的 H100 △：wgmma `kind::f8f6f4` 名字即含 FP8/FP6/FP4 三族（PTX 逐字），但 Hopper 官方规格表只列 FP8 独立峰值、FP6/FP4 无独立加速吞吐披露；Blackwell 才把 FP4/FP6 列为独立格式并给峰值。其三，INT4 在 Hopper/Blackwell 官方表「未列」是披露口径陈述（表里没有），非硬件绝对断言；V100/A100 列按 PTX target arch 反推（非逐代 whitepaper 一手）。

**两层纪律回扣**：本章矩阵只回答「支持与否」（指令几何层）。每个格式的 sign/exponent/mantissa 位布局、动态范围、量化误差、逐格式吞吐倍数（FP8 = 2× BF16、FP4 = 4× BF16）、MX 族与 NVFP4 的 scale 粒度，全部归 6.1 格式账——届时 recap 本章矩阵为硬件支持起点，增量是位布局与精度账。定义章早于使用章：6.1 的格式 × 硬件交叉表以本章为指令侧来源。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| MMA 指令 FLOPs = 2mnk；m16n8k16 = 4096、m64n128k16 = 262144、m64n256k16 = 524288 | 式 (1) | §3、练习 1 |
| 稳态发射间隔 = F_instr / c；mma.sync 4 cycles（c_TC 1024）、wgmma m64n128k16 64 cycles（c_SM 4096） | 式 (2)，命题 1 | §3、练习 1 |
| 延迟 ≠ 吞吐：wgmma 实测延迟 128 cycles [B] vs 反推稳态 64 cycles；混用低估峰值一半（494.7 对 989.4） | §3 实测核对 + 坑注 | §3、练习 4 |
| 2:4 sparse 净加速 1.78×（144 对 128 cycles，+12.5% metadata 开销），非理论 2× | §3 sparse 段 | §3、练习 4 |
| MMA 每 FLOP 寄存器字节 0.4375 vs 标量 FMA 8（1/18.3）；操作数复用是 16× 吞吐的机制来源 | 式 (3)，命题 2 | §4、练习 2 |
| MMA 流寄存器 operand 需求 448 B/cycle/block < 满 FMA 512；SM 1792 < 2048 上界（1.2） | 命题 2 | §4、练习 2 |
| wgmma shared operand 流 96 B/cycle = shared 峰值 128 的 75%；B 走 shared 描述符省寄存器 | §4 wgmma 段 | §4、练习 4 |
| wgmma 异步 warp-group（4 warp/128 线程）+ commit/wait；A register-or-shared、B shared-only、D register FP32 | §5 | §5、练习 3 |
| wgmma 累加器 N/2 regs/thread；N=256 → 128 regs → occupancy 上界 0.25（N=128 → 0.5） | 式 (4)，命题 3 | §5、练习 3 |
| tcgen05 TMEM 512×128×32-bit = 256 KB/CTA（= RF 容量）；累加器搬出寄存器；cta_group::2 双 SM 协作 | 式 (5) | §6 |
| 格式矩阵：FP16 四代、BF16/TF32/FP64 三代起、FP8 四代起、FP4/FP6 五代独立峰值；位布局账归 6.1 | §7 矩阵 | §7 |

## 练习

1.（计算 · [10]）H100 SXM BF16 主口径：(a) 用式 (1) 算 mma.sync m16n8k16 与 wgmma m64n256k16 的指令 FLOPs；(b) 用式 (2) 算 mma.sync 的稳态发射间隔（ $c_{\mathrm{TC}} = 1024$）与 wgmma m64n128k16 的稳态发射间隔（ $c_{\mathrm{SM}} = 4096$）；(c) 由 mma.sync 间隔反推 SM 级每周期 FLOPs（4 block × 4096/间隔），核对 1.1 的 $c_{\mathrm{BF16}} = 4096$。校验：(a) 4096 / 524288 FLOPs，与 1.1 §3 指令口径段一致；(b) 4 / 64 cycles；(c) 4096 FLOPs/cycle/SM，与 1.1 式 (2) 反推值一致（容差 0.1%）。
2.（推导 · [20]）操作数复用账：(a) 由 §2 fragment 布局（A 4 / B 2 / C 4 / D 4 regs，每 reg 4 B × 32 线程）推一条 mma.sync m16n8k16 的寄存器读写总字节，再除式 (1) 的 FLOPs 得式 (3)；(b) 与标量 warp FMA（1.2 式 (1)：512 B / 64 FLOPs）对比，求每 FLOP 字节比；(c) 由命题 2 算 MMA 流与满 FMA 流的每 block 每周期 operand 需求，说明为什么 16× FLOPs 不需要 16× 寄存器端口。校验：(a) 1792 B、0.4375 B/FLOP；(b) 8 B/FLOP、比值 ≈18.3；(c) 448 \lt 512 B/cycle，SM 1792 < 2048（1.2 式 (2) 上界）。
3.（推导 · [20]）补全命题 3 的证明（wgmma 累加器 → occupancy）：(a) 计数 wgmma m64nNk16 的 FP32 累加器 D 共 $64 N$ 元素、摊到 warp-group 128 线程，得式 (4) $R_{\mathrm{acc}} = N/2$；(b) 代入 $N = 128 / 256$ 得 64 / 128 regs/thread；(c) 设每 block 一个 warp-group（128 线程、 $w_b = 4$、 $S_{\mathrm{sh}} = 0$、 $R_{\mathrm{th}} = R_{\mathrm{acc}}$），逐项代入 1.2 式 (3) 求 $B_{\max}$、 $W_{\mathrm{active}}$、 $\mathcal{O}$，验证 N 翻倍 occupancy 上界减半。校验：(b) 64 / 128；(c) $N = 128$ → $B_{\max} = 8$、 $\mathcal{O} = 0.5$， $N = 256$ → $B_{\max} = 4$、 $\mathcal{O} = 0.25$，与 1.2 案例 C1/C2 register-binding 模式同构。
4.（计算 · [30]）实测核对与带宽综合账：(a) wgmma m64n128k16（A/B 均 shared，BF16）的 shared 流量 6144 B / 64 cycles = 96 B/cycle，占 shared 峰值 128 B/cycle（1.2 §4）百分之几？(b) sparse SS 实测 144 cycles 对 dense 128 cycles，有效 FLOPs 2×，求净加速；(c) 若误把延迟 128 cycles 当稳态间隔，反推 $c_{\mathrm{SM}}$ 与芯片级 BF16 峰值各是多少、较官方低估百分之几？先预测 (c) 的方向再算。校验：(a) 75%；(b) $2 \times 128/144 \approx 1.78$；(c) $c_{\mathrm{SM}} = 2048$、494.7 TFLOPS、低估 50%（与 §3 坑注一致）。

## 参考

- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 逐字 + fragment 布局、wgmma.mma_async m64nNk16（N∈8..256）/ m64nNk8 tf32 / m64nNk32 f8f6f4 + commit_group/wait_group/fence.proxy.async、tcgen05.mma/.sp/.ws + Tensor Memory 512 columns×128 rows×32-bit = 256 KB/CTA + cta_group::1/::2 + kind::mxf4nvf4
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25 dam-cdn PDF 一手——Table 3 逐精度峰值（FP64 TC 66.9 / TF32 494.7 / BF16-FP16 989.4 / FP8 1978.9 / INT8 1979 TOPS）、TC-boost 1830 MHz、4 TC/SM、Fig 7 每 block 1 TC(4th)
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——第 5 代 Tensor Core FP4/FP6 + OCP MX、Table 1 数据类型（FP64/FP32/TF32/FP16/BF16/FP8/INT8/FP6/FP4）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100/Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——dense = sparse 之半脚注、HGX B200 BF16 2250 / FP8 4500 / FP4 9000 dense（SKU 口径沿 1.1 §4）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV：wgmma m64n128k16 dense 128 cycles / sparse SS 144 cycles（实测口径，§3 延迟核对）
- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，2026-11-24——MMA atom include/cute/arch/mma*.hpp（CuTe MMA_Atom 封装 PTX 指令；当前版本架构 ⟨verify W3⟩，模块全量走读归 5.2，本章只作机制片段级指针）