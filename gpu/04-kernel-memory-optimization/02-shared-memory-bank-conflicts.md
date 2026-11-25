# 4.2 Shared memory 与 bank conflict（Shared Memory and Bank Conflicts）

> shared 延迟 29 cycles、比 HBM 快 16.5 倍，为什么一个 transpose kernel 的 shared 访问能慢过 HBM？——32 路 bank conflict 把 128 B/cycle 峰值压到 4 B/cycle，@1.83 GHz 合 7.3 GB/s，低于 HBM 每 SM 份额 25.4 GB/s。本章交付 bank 结构与冲突判定（命题 1： $k$ 路冲突串行 $k$ 周期，broadcast 同址例外）、stride 的 gcd 定律与 padding 消除（命题 2： $k = \mathrm{gcd}(s, 32)$，+1 列 padding 使列访问跨 bank 无冲突）、swizzle 异或映射（命题 3：双射且对合，保 16 B 对齐）、ldmatrix 与 128 B swizzle 的 Tensor Core 路径、教学 transpose 的 naive 对 padded conflict 账、shared vs register 权衡 | 前置：2.3（32 banks × 4 B 结构、stride-32 最坏例 7.3 GB/s、配置三旋钮、放置原则）、2.2（延迟表实测口径、全程 load-to-use 对服务周期两口径）、4.1（global sector 账与 stride 饱和律，两套粒度对照）、1.3（MMA operand 路径）；下游 4.3（tiling 的 shared 预算）、4.4（TMA swizzle 模式与 cp.async）、5.1（shuffle 免 shared）、5.2（CUTLASS swizzle layout 走读）、6.3（FA tile 布局） | 锚点：CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、CUTLASS swizzle.hpp（[NVIDIA, CUTLASS](../references.md)，教学参照 + 官方实现，版本锚 4.9.0）、Nsight Compute（[NVIDIA, Nsight Compute](../references.md)，[C]）、CUDA Programming Guide（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 7.3 GB/s 对 25.4 GB/s：片上存储的最坏反例

shared 是片上 SRAM，延迟 29.0 cycles（实测 [B]，recap 2.2 延迟表），比 HBM 全 miss 478.8 cycles 快 16.5 倍；峰值带宽 128 B/cycle/SM（32 banks × 4 B，recap 2.3 §3）。但 2.3 §3 的最坏例把这两个优势同时击穿：stride-32 列访问让全 warp 落同一 bank，32 路冲突串行，有效带宽 4 B/cycle，@1.83 GHz 合 7.3 GB/s——**低于 HBM 每 SM 份额 25.4 GB/s**（3352/132，2.4 口径）。把数据搬进 shared 不等于变快，访问模式决定拿到峰值的几成。本章回答三个问题：bank conflict 怎么判定，串行化因子怎么算？padding 与 swizzle 各怎么消除冲突，代价各是什么？transpose 这类「行列换向」访问的 conflict 账怎么算到周期？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（沿 2.3/4.1）；bank 几何自 CC 5.x 起不变（BP 逐字，§2），结论跨代同形。running example 为 32×32 fp32 tile 的 transpose（§5，教学配置 block 32×32 = 1024 threads）与 BP 官方 C = A·A^T 教学例（§3，V100 实测锚）。shared 峰值 128 B/cycle 的 GB/s 换算沿 2.3 §7 的 1.98 GHz（253 GB/s）；服务周期与最坏例的时间换算取 1.83 GHz（沿 2.3 §3 口径）——两个 clock 各归其口径，不混用（2.3 章脊声明 recap）。延迟 29.0 cycles 为 H800 微基准实测 [B]（[Luo et al., 2024](../references.md)，arXiv:2402.13499），非 H100 官方规格。V100 实测带宽（12.8 / 140.2 / 199.4 GB/s）是 BP 教学例时代口径，机制跨代、绝对值不代入 H100。

符号声明： $k$ = bank 冲突重数（recap 2.3，服务周期数）； $u_b$ = bank $b$ 内被请求的 distinct 32-bit 字数（同址 broadcast 计 1，§2）； $s$ = stride（以 32-bit 字计，沿 4.1 口径；4.1 的 $n_{\mathrm{sec}}$ sector 数本章不占用）； $w$ = shared tile 宽度（以字计，本章局部；裸字母 T 专归序列长、本章不占用，tile 维不用裸 T）； $p$ = 每行 padding 字数（§3）； $a$ = shared 字节地址； $\mathrm{bank}(a)$ = 地址 $a$ 的 bank 号； $\mathrm{off}$ = shared 内字节偏移（§4 swizzle）； $S_{\mathrm{sh}}$ = shared bytes/block（recap 2.3）。bytes 不占字母；B/KB 单位豁免（片上 KB = 1024 B，沿 1.2）。

**边界声明**：本章交付 shared 侧的冲突判定与消除。global 侧 sector 账归 4.1（两套粒度对照见 §2 末）；tiling 的复用与预算账归 4.3；TMA/cp.async 的异步搬运与 descriptor 账归 4.4；CUTLASS swizzle layout 的模块走读归 5.2（本章只到机制片段级）；occupancy 的 shared 项精确计算归 3.4。代码 scope：机制片段级（§3 padding 声明、§5 transpose），不做模块走读。

## 2. bank 结构与冲突判定：k 路冲突串行 k 周期

32 个线程同时访问 shared，硬件怎么仲裁？bank 结构 recap 2.3 §3（不重推）：32 个 bank、连续 32-bit 字映射到连续 bank、每 bank 每周期 32 bit（BP §10.2.3.1 逐字「each bank has a bandwidth of 32 bits every clock cycle, and successive 32-bit words are assigned to successive banks. The warp size is 32 threads and the number of banks is also 32」，[NVIDIA, Best Practices](../references.md)，[C]）。地址到 bank 的映射：

$$
\mathrm{bank}(a) = \lfloor a/4 \rfloor \bmod 32 \qquad (1)
$$

**符号**： $a$ = shared 字节地址； $\lfloor a/4 \rfloor$ = 32-bit 字号；mod 为取余。全数字代入：字 0–31 落 bank 0–31，字 32 回 bank 0——每 128 B 一个 bank 循环。无冲突时 32 bank 并行、一拍服务整 warp： $32 \times 4 = 128$ B/cycle（2.3 峰值库存数）。

```
shared 字地址 → bank 映射（每 bank 4 B/cycle）
字:    w0  w1  w2 ... w31 | w32 w33 ... w63 | w64 ...
bank:   0   1   2 ...  31 |   0   1 ...  31 |   0 ...
       └──── 一拍 32 字 = 128 B（无冲突峰值）────┘
冲突：同一 bank 的不同字必须分拍取 → k 个 distinct 字 = k 拍
例外：同一 bank 的同一字 → broadcast 一拍送全部请求 lane
```

冲突判定的官方语义（BP 逐字）：「if multiple addresses of a memory request map to the same memory bank, the accesses are serialized. The hardware splits a memory request that has bank conflicts into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of separate memory requests.」例外逐字：「The one exception here is when multiple threads in a warp address the same shared memory location, resulting in a broadcast. In this case, multiple broadcasts from different banks are coalesced into a single multicast」——同址读 broadcast 免冲突，多 bank 各自的 broadcast 合成一次 multicast；同址写由其中一个线程写入、哪个线程写 undefined（recap 2.3 §3，PG 口径）。

**命题 1**（冲突判定与串行化因子）。warp 内每 lane 请求一个 32-bit 字（允许同址），设 $u_b$ 为 bank $b$ 内被请求的 distinct 字数（同址计 1），则服务周期数

$$
k = \max_{0 \le b \le 31} u_b \qquad (2)
$$

全 warp 每 lane 4 B 有用时有效带宽 $128/k$ B/cycle。**证明**（路线：下界鸽笼 + 上界构造，两步）：下界——同一 bank 的 $u_b$ 个不同字必须分属不同 conflict-free 请求（每请求每 bank 至多 1 字），故请求数至少 $\max_b u_b$；上界——把每个 bank 的第 $j$ 个字放进第 $j$ 个请求，各请求每 bank 至多 1 字、conflict-free，共 $\max_b u_b$ 个请求足够；硬件按 BP 逐字拆成「as many separate conflict-free requests as necessary」，每请求一拍，故 $k = \max_b u_b$。broadcast 同址字计 1，不增 $u_b$。∎

正例与反例表（fp32 warp，峰值 128 B/cycle；正例 stride-1，边界反例「同 bank 即冲突」——同 bank 同址是 broadcast 不是冲突）：

| 访问模式 | bank 分布 | $k$ | 有效带宽 |
| --- | --- | --- | --- |
| stride-1（连续） | 32 bank 各 1 字 | 1 | 128 B/cycle |
| stride-2（隔字） | 16 bank 各 2 字 | 2 | 64 B/cycle |
| stride-16 | 2 bank 各 16 字 | 16 | 8 B/cycle |
| stride-32（32 宽列访问） | 全落 1 个 bank | 32 | 4 B/cycle |
| 全 lane 同一字 | 同 bank 同址 | 1 | broadcast 一拍 |
| 16 lane 字 0 + 16 lane 字 32 | bank 0 两个不同字 | 2 | 64 B/cycle |

末行是「broadcast 免冲突」的边界反例：两组 lane 各自同址，但两组地址不同（字 0 与字 32 同落 bank 0）， $u_0 = 2$，照样 2 拍——**免冲突的条件是同址，不是同 bank 内有人同址**。stride-32 行 recap 2.3 §3 最坏例：4 B/cycle @1.83 GHz = 7.3 GB/s，低于 HBM 份额 25.4 GB/s。

64-bit 访问的口径：每 lane 8 B 拆成两个 32-bit 字按同规则判，带宽账用实际有用字节。double 连续访问：lane $i$ 读字 $2i, 2i+1$，每 bank 2 个 distinct 字， $k = 2$——但 256 B 有用字节本来就需要至少 2 拍（128 B/cycle 峰值），2 拍搬 256 B 仍是 128 B/cycle，**这不是冲突损失**；double stride-2（字节 stride 16）：lane $i$ 读字 $4i, 4i+1$，16 个 bank 各 4 字， $k = 4$，256 B 搬 4 拍 = 64 B/cycle，相对 2 拍下界折半。**口径注**： $k$ 是服务周期（吞吐口径），冲突损失要看 $k$ 相对「有用字节 ÷ 128 B/cycle（峰值带宽）」这一下界（单位：周期）多出几倍，不是见 $k$ 大于 1 就判冲突； $k$ 与 29-cycle load-to-use 延迟是两笔账（2.2 口径注 recap），服务周期算吞吐、延迟算隐藏并发（1.4），勿混。

shared 与 global 的两套粒度对照（4.1 §5 末预告的兑现）：BP 逐字「Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses by a warp in shared memory.」——shared 没有 sector 粒度、没有 32 B 对齐要求，乱序与不对齐本身不罚，**只罚 bank 冲突**；global 则按 32 B sector 计流量（4.1 式 (1)）。同一个 stride-16：global 侧饱和于 $\eta = 1/8$（32 sector 地板），shared 侧 $k = 16$（16 路冲突，8 B/cycle @1.83 GHz = 14.6 GB/s）——两条定律各自的粒度各自算，勿互搬直觉。

实测侧读法（[NVIDIA, Nsight Compute](../references.md) ProfilingGuide §2.11.1，[C]）：shared wavefronts 逐字「Number of wavefronts required to service the requested shared memory data. Wavefronts are serialized and processed on different cycles.」——wavefront 即命题 1 的 conflict-free 子请求，wavefronts 对 requests 的比值就是平均 $k$；bank conflict 计数器 `l1tex__data_bank_conflicts_pipe_lsu.sum`（recap 2.3 §3）。excessive wavefronts（超出 ideal 的盈余）直接指示冲突损失，drill-down 归 11.2。

## 3. stride 的 gcd 定律与 padding 消除

一列 padding 值多少，为什么能消掉全部冲突？先把 stride 访问的冲突重数算成闭式。推导路线：lane $i$ 的字地址是等差数列 $si + c$，bank 号是它对 32 取余——余数序列的像集大小与每个像点的重数由 $\mathrm{gcd}(s, 32)$ 决定；关键一步是「 $i$ 与 $i + 32/g$ 同余」（ $g = \mathrm{gcd}(s, 32)$ ）给出碰撞周期。

**命题 2**（stride 的 gcd 定律）。全 warp 32 lane、lane $i$ 请求字 $si + c$（ $s \ge 1$ 整数、 $c$ 常数），冲突重数

$$
k = \mathrm{gcd}(s, 32) \qquad (3)
$$

**证明**（要点）：设 $g = \mathrm{gcd}(s, 32)$。bank 序列 $si + c$ 除 32 的余数序列像集大小为 $32/g$（ $si$ 除 32 的余数取遍 $g$ 的倍数平移），每个像点恰被 $g$ 个 lane 命中——因 $s(i + 32/g) - si = 32s/g$ 是 32 的倍数（ $g \mid s$），且 $32/g$ 是最小正周期。故像集内每个 bank 恰有 $g$ 个两两不同的字， $k = g$。完整证明（含像集大小与重数的逐条论证）留作练习 3。∎

代入核对： $s = 1$ 给 $k = 1$； $s = 2$ 给 2； $s = 16$ 给 16； $s = 32$ 给 32——与 §2 表逐行一致。注意与 4.1 的差异：global stride 利用率在 $s \ge 8$ 饱和（sector 地板），shared 的 $k$ 随 $\mathrm{gcd}(s, 32)$ 走、 $s = 32$ 时最坏 32——**shared 没有饱和地板，只有数论结构**。

**推论**（+1 列 padding 消除列访问冲突）。 $w = 32$ 的 tile 列访问 stride 为 $w$；每行 pad $p = 1$ 字后 stride 变 $w + 1 = 33$， $\mathrm{gcd}(33, 32) = 1$， $k = 1$——列访问跨 bank 无冲突。**证明**（一步）：33 除 32 余 1，lane $i$ 的 bank 号即 $i + c$ 除 32 的余数，取遍 32 个 bank，等价 unit stride。∎ BP 官方逐字（§10.2.3.3）：「The simple remedy is to pad the shared memory array so that it has an extra column」，「This padding eliminates the conflicts entirely, because now the stride between threads is w+1 banks (i.e., 33 for current devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a unit stride.」官方代码同形：

```cpp
__shared__ float transposedTile[TILE_DIM][TILE_DIM+1];   // BP 官方 padding 声明（TILE_DIM = 32）
```

边界反例（padding 不是越多越好）： $p = 2$ 给 stride 34， $\mathrm{gcd}(34, 32) = 2$，仍 2 路冲突——**padding 有效的条件是 $\mathrm{gcd}(w+p, 32) = 1$**， $w = 32$ 时即 $p$ 为奇数， $p = 1$ 是最省容量的选择。一般 $w$ 同理（练习 3(b)）。

官方实测锚（BP §10.2.3.3，C = A·A^T 教学例，V100 时代口径）：无优化 12.8 GB/s（global 侧 stride 访问，4.1 饱和律的受害者）→ shared tile 140.2 GB/s（global 合并了，但 shared 列写冲突）→ padding 后 199.4 GB/s。两级收益 10.95 倍与 1.42 倍。注意官方例的冲突在**列写**侧：global 读要 coalesced，被迫按列写 shared tile，stride $w = 32$ 全 warp 同 bank（BP 逐字「this results in a stride between threads of w banks - every thread of the warp hits the same bank」）；§5 的教学 transpose 冲突在**列读**侧——同一笔 bank 账，读写两侧都适用命题 2。

padding 的两笔代价：容量 +3.125%（32×33×4 = 4224 B 对 32×32×4 = 4096 B，每行多 1 字）；**对齐破坏**——行距变 132 B，不是 16 的倍数，行首的 16 B 对齐逐行轮换（132 除 16 余 4），float4 与 ldmatrix 的向量化路径失效（4.1 §4 自然对齐是正确性边界）。标量 fp32 访问用 padding 最省；向量化与 Tensor Core 路径要保对齐，得用下一节的 swizzle。

> **坑**：给 float4/ldmatrix 路径的 tile 加 +1 padding。触发条件：tile 行距 33 字（132 B）上跑 16 B 向量访问；现象：行首 16 B 对齐被破坏，自然对齐违反是未定义行为（4.1 §4 PTX 逐字），不是「慢一点」；规避：标量路径用 padding，向量化/Tensor Core 路径用 swizzle（§4，行距保持 128 B 倍数、只置换行内布局）。

## 4. swizzle：异或映射的双射与 Tensor Core 路径

padding 破坏 16 B 对齐，有没有不破坏对齐的消冲突法？有——swizzle：不改行距、只置换行内字节的位置。所谓 swizzle，是指对 shared 地址偏移做可逆置换、把逻辑上相邻的数据散到不同 bank 的布局变换。CUTLASS 的实现逐字（[NVIDIA, CUTLASS](../references.md) `include/cute/swizzle.hpp`，教学参照 + 官方实现，版本锚 4.9.0）：

> 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY

模板 `Swizzle<BBits, MBase, SShift>`：取偏移的高位段 YY（BBits 位），异或进低位段 ZZ（MBase 起 BBits 位）；apply 逐字 `return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY`。代码级保证：`static_assert(abs(num_shft) >= num_bits)`——移位距离不小于掩码位数，**YY 与 ZZ 位段不相交**，这是双射性的结构前提。

教学实例 `Swizzle<3, 4, 3>`（B=3、M=4、S=3，行距 128 B 的 tile）：取字节偏移的 bit 7–9（行号低 3 位），异或进 bit 4–6（128 B 行内的 16 B chunk 号）：

$$
\mathrm{off}_{\mathrm{sw}} = \mathrm{off} \oplus \left( \left\lfloor \mathrm{off} / 2^{7} \right\rfloor \bmod 2^{3} \right) \cdot 2^{4} \qquad (4)
$$

**符号**： $\mathrm{off}$ = 原字节偏移； $`\mathrm{off}_{\mathrm{sw}}`$ = swizzle 后偏移； $\oplus$ 按位异或。机制一句话：行号决定该行数据落在行内哪个 16 B chunk——第 $r$ 行的 16 B 数据被放到 chunk $r$，8 行散到 8 个 chunk、32 个 bank。

**命题 3**（swizzle 双射与对合）。式 (4) 的映射是地址空间上的双射，且是对合（施加两次还原）；并保持 128 B line 归属与 16 B chunk 内偏移不变。**证明**（三步）：异或掩码只由 bit 7–9 决定，异或只修改 bit 4–6，两段不相交——第二次施加时 bit 7–9 未变、掩码相同， $\mathrm{off} \oplus m \oplus m = \mathrm{off}$，对合成立；对合必双射（逆映射是自身）；bit 7 及以上不变故 128 B line 归属不变，bit 0–3 不变故 16 B chunk 内偏移不变（16 B 对齐保持）。∎ 教学验证：0–4095 全偏移穷举，双射、对合、line 归属与 16 B 对齐四条全部成立。一般 `Swizzle<B, M, S>` 在位段不相交条件（官方 static_assert）下同证（练习 3(c)）。

swizzle 对读侧透明：写入用式 (4) 置换，读取用同一公式再置换一次即还原（对合），逻辑坐标不变——**swizzle 是布局变换不是数据变换**，正确性由双射保证，收益由 bank 散布保证。

Tensor Core 路径的官方接口：ldmatrix（[NVIDIA, PTX ISA](../references.md) §9.7.16.5.15，[C]）逐字「Collectively load one or more matrices across all threads in a warp from the location indicated by the address operand p, from .shared state space into destination register r.」——warp 集体从 shared 搬矩阵进 register，喂 mma 指令（1.3 的 operand 路径）。几何：`.m8n8` 为 8×8 16-bit 矩阵，`.num` = `.x1/.x2/.x4`，`.trans` 可选转置；地址逐字「The eight addresses required for each matrix are provided by eight threads」，「Each address corresponds to the start of a matrix row」，且「Consecutive instances of row need not be stored contiguously in memory.」——**行不必连续存放**，这句是 swizzle 布局的官方许可。

全数字 conflict 账（教学口径：8×8 bf16 矩阵，行 16 B，shared 行距 128 B）：无 swizzle 时行 $r$ 的字偏移 $32r + c$（ $c = 0, \ldots, 3$），bank 号恒等于 $c$——8 行全落 bank 0–3，每 bank 8 个 distinct 字， $k = 8$（命题 1）；`Swizzle<3,4,3>` 后行 $r$ 落 chunk $r$，8 行 × 4 字覆盖 32 个 bank 各 1 字， $k = 1$。**8 倍服务周期差**，且行距仍 128 B、16 B 对齐完好——padding 做不到这点（§3 坑注）。ldmatrix 的 shared 访问按同一 bank 规则判定（教学口径：式 (2) 对 ldmatrix 提供的行地址同样适用）。

硬件侧的 swizzle 支持：TMA 的 tensor-map 描述符带 swizzle 模式（PTX §5.5.7 Swizzling Modes，[C]）——32 B / 64 B / 96 B / 128 B 多档，128 B 档含 16-byte atomicity 等子模式（保持 16 B 数据块完整）；TMA 搬运时由硬件直接按 swizzle 布局写 shared，软件免手工置换（descriptor 与搬运账归 4.4）。CUTLASS 的 swizzle layout 与 MMA atom 的模块级走读归 5.2，MMA 指令几何归 1.3（本章只立 bank 侧动机与机制）。

## 5. 教学 transpose：naive 对 padded 的 conflict 账

32×32 的 transpose 在 shared 侧亏多少？教学 kernel（机制片段级；naive 版把 `[32][33]` 改回 `[32][32]`）：

```cpp
// 教学片段，非优化实现：32x32 tile transpose（padded 版）
__global__ void transpose(const float* in, float* out, int width) {
    __shared__ float tile[32][33];              // +1 列 padding（命题 2 推论）
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];   // global 行读 → shared 行写
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;          // 输出块坐标互换
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * width + x] = tile[threadIdx.x][threadIdx.y];  // shared 列读 → global 行写
}
```

教学配置：block 32×32 = 1024 threads（32 warps，每线程 1 元素），矩阵宽 width 为 32 的倍数（教学假设 1024）；1024 线程 block 的 occupancy 为 2 blocks/SM（2048 线程上限，3.4），此配置为账目清晰、非最优 launch 配置。warp 由同 threadIdx.y 的 32 个线程构成（threadIdx.x = 0..31）。

conflict 账逐相位（命题 1/2，字地址按行距展开）：

- **行写相位**：warp 写 tile[ty][tx]，字地址 ty·32 + tx（naive）或 ty·33 + tx（padded），对 tx 连续， $k = 1$——两版都无冲突。
- **列读相位**：warp 读 tile[tx][ty]，字地址 tx·32 + ty（naive，stride 32）或 tx·33 + ty（padded，stride 33）。naive： $k = \mathrm{gcd}(32, 32) = 32$，全 warp 落 bank ty；padded： $k = \mathrm{gcd}(33, 32) = 1$，bank 号 (tx + ty) 除 32 的余数取遍 32 bank。
- **每 tile 合计**（32 warps）：naive $32 \times (1 + 32) = 1056$ 服务周期；padded $32 \times (1 + 1) = 64$ 周期，**16.5 倍**。@1.83 GHz 合 577.0 ns 对 34.97 ns。列读相位的有效带宽：naive 4 B/cycle（7.3 GB/s）对 padded 128 B/cycle。

**口径注**：1056/64 是 shared 服务周期（吞吐口径，bank 峰值 128 B/cycle），不是 29-cycle load-to-use 延迟——延迟账另算（2.2 口径注 recap），两口径勿混。

global 侧两版都 coalesced：行读 in[y·width + x] 与行写 out[y·width + x] 的 warp 内 x 连续，各 4 sector（4.1 式 (1)）。对照不用 shared 的直接 global transpose：列写 stride = width 字（1024），字节 stride 4096 远超 32，每 warp 32 sector、 $\eta = 1/8$（4.1 饱和律）。所以 shared tile 的收益结构是：**把 global 侧 8 倍流量膨胀换成两侧 coalesced，代价是 shared 往返加（naive 版的）bank 冲突；padding 把冲突代价也消掉**。这与 BP 官方 C = A·A^T 例的三级结构同形（§3：12.8 → 140.2 → 199.4 GB/s，V100 时代口径）：第一级收益来自 global 合并，第二级来自 bank 冲突消除，方向与本节账一致。

> **提点**：transpose 是「两套粒度各付各的账」的最小完整例——global 侧按 sector 数流量（4.1），shared 侧按 bank 数周期（本章），优化的每一步都只是把某一侧的账从坏模式换到好模式。删掉这条类比，上面的逐相位账与命题链依然完整。

## 6. shared vs register：放置权衡的冲突维度

数据该放 shared 还是 register？放置原则 recap 2.3 §2（不重推）：线程私有热数据放 register，block 共享数据放 shared；账面根据是「副本数 × 容量」对「一份容量 + 访问延迟」——32 线程各需同一份 128 B 片段，register 要 32 份副本共 4 KB，shared 一份 128 B 加每次访问 29 cycles。本节补冲突维度与第三条路：

- **register**：operand 读出在执行管线内（无 load 延迟口径，2.2），无用户可见的 bank 冲突（register bank/operand collector 是硬件内部机制，2.3 教学模型）；代价是容量按线程切死（每线程上限 255），多占一个 register 就少驻留一个 block（3.4 register 悬崖）。
- **shared**：一份数据服务全 block，容量 228 KB/SM 远大于单线程 register 预算；代价是 29 cycles 延迟加本章的冲突风险——**冲突不是 shared 的固有属性，是访问模式的属性**，命题 2/3 给了消除工具。
- **warp shuffle**：warp 内 register 直换，不经 shared、无冲突、不占 shared 容量（3.1 §6 预告，实现归 5.1）——warp 私有交换的第三条路，reduction/scan 的 warp 级阶段用它免 shared 往返。

现代实例（recap 1.3，不重推）：Hopper wgmma 的 B 操作数从 shared 经描述符直喂 Tensor Core、跳过 register 中转——shared 成为 MMA 的 operand 源，swizzle 布局（§4）是这条路径无冲突的前提；register 侧省下的容量换 occupancy 或 ILP（3.4 权衡）。

决策表（冲突维度并入 2.3 §7 放置链）：

| 数据形态 | 放置 | 冲突/容量账 |
| --- | --- | --- |
| 线程私有热数据 | register | 无 load 延迟；占 $R_{\mathrm{th}}$ 预算（3.4） |
| warp 内交换 | shuffle | 免 shared、免冲突（5.1） |
| block 共享、标量访问 | shared + padding | $k = 1$（命题 2 推论），容量 +3.125% |
| block 共享、向量化/MMA | shared + swizzle | $k = 1$（命题 3），保 16 B 对齐 |
| block 共享、容量敏感 | shared，权衡 occupancy | $S_{\mathrm{sh}}$ 换驻留 block 数（3.4/4.3） |

shared 容量配置三旋钮（静态 48 KB 上限、动态 opt-in、carveout hint）recap 2.3 §3，occupancy 的 shared 项精确计算（128 B 粒度 + 1 KB 预留）归 3.4——本章的 padding/swizzle 改变的是 $S_{\mathrm{sh}}$ 与访问模式，预算约束本身不在本章重推。

## 7. 边界：本章不做什么

冲突账算完，剩下的归谁？本章交付 bank 映射（式 (1)）、冲突判定（命题 1 + 式 (2)）、gcd 定律与 padding（命题 2 + 式 (3)）、swizzle 双射（命题 3 + 式 (4)）、ldmatrix 与 TMA swizzle 的机制接口、transpose 逐相位账与放置权衡冲突维度；以下各有专章。tiling 的复用因子与 tile 预算归 4.3；TMA descriptor/cp.async 流水归 4.4；shuffle 归约实现归 5.1；CUTLASS swizzle layout 与 GEMM 模块走读归 5.2；FA 的 shared tile 布局归 6.3；occupancy 精确计算归 3.4；profiling drill-down 归 11.2。代码 scope：机制片段级（§3 padding 声明、§5 transpose），不做模块走读。

本章交付清单（下游 recap 接口）：式 (1) bank 映射 + 命题 1 判定（§2）、命题 2 gcd 定律 + padding 推论（§3）、命题 3 swizzle 双射 + 式 (4)（§4）、transpose 1056 对 64 周期账（§5）、放置决策表（§6）。4.3 recap 命题 2/3 作 tile 布局的无冲突约束；4.4 recap §4 TMA swizzle 模式作 descriptor 账的起点；5.2 recap 式 (4) 与 `Swizzle<B,M,S>` 作 CuTe layout 走读的机制侧；6.3 recap §5 账法作 FA tile 布局的冲突判定。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 32 banks × 4 B，bank = (addr/4) mod 32；无冲突峰值 128 B/cycle（CC 5.x+ 几何不变） | 式 (1) + BP 逐字 | §2、练习 1 |
| $k$ 路冲突串行 $k$ 周期： $k = \max_b u_b$（每 bank distinct 字数）；有效 128/k B/cycle | 式 (2)，命题 1 | §2、练习 1/2 |
| broadcast 同址免冲突（multicast 合成）；同 bank 不同址仍冲突（16 lane 字 0 + 16 lane 字 32 → 2 拍） | §2 BP 逐字 + 反例 | §2、练习 1 |
| 64-bit 连续访问 $k = 2$ 但 256 B/2 拍 = 峰值，非冲突损失；冲突看 $k$ 相对字节下界 | §2 口径注 | §2、练习 2 |
| shared 只罚 bank 冲突，无 sector/对齐罚；stride-16 global 饱和 1/8 对 shared 16 路冲突——两套粒度两条定律 | §2 BP 逐字 + 4.1 对照 | §2 |
| stride 访问 $k = \mathrm{gcd}(s, 32)$；+1 padding 使 33 ≡ 1（mod 32）列访问无冲突； $p = 2$ 反例仍 2 路 | 式 (3)，命题 2 + 推论 | §3、练习 2/3 |
| padding 代价：容量 +3.125%（4096→4224 B）+ 行距 132 B 破坏 16 B 对齐（向量化路径失效） | §3 两笔代价 | §3、练习 2 |
| BP 官方三级锚（V100）：12.8 → 140.2 → 199.4 GB/s（global 合并 10.95× + padding 1.42×） | §3 BP 逐字 | §3、练习 4 |
| swizzle 异或映射双射且对合，保 128 B line 与 16 B 对齐；`Swizzle<3,4,3>` 行号 XOR 进 chunk 号 | 式 (4)，命题 3 | §4、练习 3 |
| ldmatrix 8×8 b16 行距 128 B：无 swizzle $k = 8$，swizzle 后 $k = 1$；行不必连续存放（PTX 逐字）；TMA 32/64/96/128 B 硬件 swizzle | §4 PTX 逐字 + 账 | §4 |
| transpose 每 tile：naive 1056 服务周期对 padded 64（16.5×）；global 两侧 coalesced，直接 global 列写 $\eta = 1/8$ | §5 逐相位账 | §5、练习 4 |
| 放置冲突维度：register 无冲突但容量切死；shared 冲突可消除；warp 内交换走 shuffle 免 shared | §6 决策表 | §6 |

## 练习

1.（计算 · [10]）bank 代入（fp32 warp，峰值 128 B/cycle）：(a) stride-1、stride-2、stride-16、stride-32 的 $k$ 与有效带宽；(b) 全 lane 读同一字走什么机制、几拍；(c) stride-32 的有效带宽 @1.83 GHz 折多少 GB/s，与 HBM 每 SM 份额 25.4 GB/s 比谁大？先预测 (c) 的方向再算。校验：(a) 1/2/16/32 路，128/64/8/4 B/cycle，与 §2 表一致（stride-1/2/32 行 recap 2.3 §3）；(b) broadcast 一拍；(c) 7.3 GB/s 低于 25.4，与 2.3 §3 一致（公开数值锚）。
2.（计算 · [20]）padding 与 64-bit 账：(a) 32×33 tile 列读（ty = 0）前 8 个 lane 的 bank 号序列；(b) padding $p = 2$（stride 34）的 $k$；(c) 32×32 fp32 tile padding 后的容量与增量百分比；(d) double 连续访问的 $k$ 与有效带宽，判它是不是冲突损失。校验：(a) bank 0–7 各一（ $33i$ 除 32 余 $i$）；(b) 2（ $\mathrm{gcd}(34, 32) = 2$，命题 2）；(c) 4224 B、+3.125%（§3）；(d) $k = 2$、128 B/cycle，非冲突损失（256 B 下界 2 拍，§2 口径注）。
3.（推导 · [30]）补全命题 2 与命题 3 的证明：(a) 证 bank 序列 $si + c$ 除 32 的余数序列像集大小 $32/g$、每像点重数 $g$（ $g = \mathrm{gcd}(s, 32)$），得 $k = g$；(b) 推出 padding 有效条件 $\mathrm{gcd}(w+p, 32) = 1$，构造 $w = 32$、 $p = 2$ 的反例并算 $k$；(c) 证一般 `Swizzle<B,M,S>` 在位段不相交条件下是对合（掩码只依赖不变位段），从而双射；(d) 说明 padding 破坏 float4 对齐而 swizzle 不破坏的位级原因。校验：(b) $p = 2$ 给 $k = 2$，与练习 2(b) 一致；(c) 施加两次还原（式 (4) 穷举验证，§4）；(d) padding 改行距（bit 4 以上全变）、swizzle 只动 bit 4–6（§3/§4）。
4.（计算 · [30]）transpose 综合账（§5 教学配置）：(a) naive 与 padded 版每 tile 的 shared 服务周期与比值；(b) 列读相位两版的有效带宽（B/cycle）；(c) 不用 shared 的直接 global transpose（width = 1024 fp32）列写的每 warp sector 数与 $\eta$；(d) BP 官方 V100 锚 140.2 → 199.4 GB/s 的比值，与 (a) 的方向是否一致，为什么绝对值不能直接对账。先预测 (a) 的比值数量级再算。校验：(a) 1056 对 64、16.5×（§5）；(b) 4 对 128 B/cycle；(c) 32 sector、 $\eta = 1/8$（4.1 式 (2) 饱和律）；(d) 1.42×，方向一致（同为 padding 消冲突），绝对值不可比——V100 时代口径且官方例含 global 侧账（§3/§5 口径注）。

## 参考

- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——§10.2.3.1 Shared Memory and Memory Banks（CC 5.x+ each bank 32 bits/cycle + successive words → successive banks + warp 32 = banks 32 逐字 + conflict 串行化 factor = separate requests 数逐字 + broadcast/multicast 逐字 + 「Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses」逐字）；§10.2.3.3 C = AAT 教学例（列写冲突逐字 + 「pad ... an extra column」+ 「33 for current devices ... equivalent to a unit stride」逐字 + transposedTile[TILE_DIM][TILE_DIM+1] 代码 + V100 12.8/140.2/199.4 GB/s 三级实测，时代口径）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——§9.7.16.5.15 ldmatrix（「Collectively load one or more matrices across all threads in a warp ... into destination register r」逐字 + .m8n8/.x1/.x2/.x4/.trans + 八地址由八线程提供、每地址对应行首、「Consecutive instances of row need not be stored contiguously in memory」逐字）；§5.5.7 Swizzling Modes（32/64/96/128 B 模式 + 128 B 档 16-byte atomicity 子模式，TMA tensor-map swizzle）
- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，2026-11-25——include/cute/swizzle.hpp（版本锚 4.9.0；「0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY」逐字 + `Swizzle<BBits, MBase, SShift>` + apply「offset ^ shiftr(offset & yyy_msk{}, msk_sft{})」逐字 + `static_assert(abs(num_shft) >= num_bits)` 位段不相交保证）；模块级 layout 走读归 5.2
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——ProfilingGuide §2.11.1 Shared Memory table（shared wavefronts「Wavefronts are serialized and processed on different cycles」逐字）；l1tex__data_bank_conflicts_pipe_lsu.sum 计数器（recap 2.3 §3）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——shared on-chip 高带宽低延迟定性 + 同址写 undefined 语义（recap 2.3 §3 口径）；32 banks 结构（compute-capabilities 表，recap 2.3）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——shared 29.0 cycles 延迟实测口径（H800 微基准 Table IV，非 H100 官方规格，沿 2.2/2.3 声明）
