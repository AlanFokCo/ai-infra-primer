# 6.1 低精度格式与 Tensor Core（Low-Precision Formats and Tensor Core）

> FP8 峰值是 BF16 的 2 倍、FP4 再翻一倍——这两个 2 倍从哪来，位宽减半在算力侧买到了什么、又在精度侧付出了什么？本章 recap 1.3 MMA 指令几何与格式支持矩阵（不重推），交付格式账：FP16/BF16/TF32/FP8 E4M3/E5M2/FP4 E2M1/INT8/INT4 的位布局与动态范围/ULP 闭式（命题 1）、FP32 累加的精度下界（命题 2）、MX 族与 NVFP4 的 scale 编码与粒度开销账、逐格式 MMA 吞吐的 datasheet 对账（命题 3），以及 scale 的三条硬件消费路径（6.2 dequant 融合回链） | 前置：1.3（MMA 指令几何式 (1)(2)、格式支持矩阵、tcgen05 block scaling）、1.1（SKU 与 dense/sparse 口径）、2.4（roofline 与带宽库存）；recap 源：training 1.3 §1（浮点值公式与 FP8 位布局表）、inference 3.2 §2（MX/NVFP4 格式事实与每元素字节账）；下游 6.2（dequant 融合 kernel）、6.3（FA FP8 路径） | 锚点：PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C] 官方规格披露·一手权威）、OCP MX v1.0 规范（[OCP, 2023](../references.md)，[C]）、H100/Blackwell datasheet（[NVIDIA, datasheet](../references.md)，[C]）、Hopper whitepaper V1.04（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、Blackwell Technical Brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、CUTLASS 浮点编码（[NVIDIA, CUTLASS](../references.md)，教学参照 + 官方实现）、DeepGEMM sm100 MMA 路径（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照 + 官方实现）——官方规格与代码锚 2026-11-25 一手核实

## 1. 两个 2 倍从哪来：本章口径与两层纪律

inference 3.2 把权重从 bf16 压到 4 bit，字节账给出近 4 倍的带宽节省——那是**访存侧**的收益。算力侧的账是另一笔：H100 的 FP8 dense 峰值 1978.9 TFLOPS 恰是 BF16 989.4 的 2 倍，B200 的 FP4 9000 又是 BF16 2250 的 4 倍（[NVIDIA, datasheet](../references.md)，[C]）。位宽减半为什么能换算力翻倍？精度侧又付出了什么——FP4 只有 16 个可表值，凭什么敢拿来算 GEMM？本章回答三个问题：位布局怎么读出动态范围与精度（§2-3）？逐格式吞吐怎么与 datasheet 对账（§5）？scale 的粒度开销怎么算、由谁消费（§4/§6）？

**本章口径**（章脊锁定）：主口径 **H100 SXM**（沿 1.1/1.3），对照 **HGX B200**（SKU 口径沿 1.1 §4：HGX B200 180 GB/7.7 TB/s，勿与 GB200 186 GB/8 TB/s 混）；吞吐一律 **dense**（sparse 表值 = dense × 2，1.1 §6 recap）。格式常数取 training 1.3 §1 表 + OCP MX v1.0（[OCP, 2023](../references.md)，[C]）+ PTX ISA 格式节（[NVIDIA, PTX ISA](../references.md)，[C]）；字节账脊柱沿 inference 3.2（单个线性层权重 $4096 \times 4096$，bf16 = 32 MiB）。片上 KB = 1024 B，带宽 GB/s 按 $10^{9}$ B/s。

**两层纪律**（1.3 ↔ 6.1，spec §4.2）：1.3 交付指令几何层——MMA 指令形状、操作数路径、稳态吞吐推导、格式「支持与否」矩阵；本章 recap 其几何与矩阵为起点，增量是**格式账**：位布局闭式、范围/ULP、MX scale 编码、逐格式吞吐倍数对账、scale 消费路径。量化算法（GPTQ/AWQ/SmoothQuant 怎么改量化映射）归 inference 3.2，本章只 recap 结论不重推；dequant 的 kernel 实现（Marlin lop3、DeepGEMM promotion）归 6.2。

**符号声明**：本章 $E$ = 指数位宽、 $M$ = 尾数位宽、 $b$ = 指数偏置、 $e$ = 指数字段整数值、 $s$ = 符号位（沿 training 1.3 式 (1) 记号，**本章 $M$ 是尾数位宽，非消息字节亦非 GEMM 维**）； $K$ = 累加深度（GEMM 归约维，章内局部）； $u$ = 单位舍入误差（unit roundoff）； $q$ = 量化位宽、 $g$ = 量化组大小（沿 inference 3.2）；bytes 不占字母。指令几何一律字面数（如 wgmma m64n128k16），族名 m64nNk16 里的 N 是 PTX 指令字面占位，非参数量。

## 2. 位布局账：三种特殊值约定与范围闭式

一个 E4M3 数为什么最大只能表到 448，而不是把 NaN 码位当有限数算出的 480？答案藏在特殊值约定里。先 recap 值公式（training 1.3 式 (1)，不重推）：正规数 $x = (-1)^{s} \cdot 2^{e-b} \cdot (1+m)$，尾数 $m \in [0,1)$ 占 $M$ 位；指数字段全零时为次正规数，值 $(-1)^{s} \cdot 2^{1-b} \cdot m$，最小正值 $2^{1-b-M}$。指数位买**动态范围**（binade 跨度），尾数位买**相对精度**（1 附近间隔 $2^{-M}$）——这句定性结论 training 1.3 已交付，本章把它落成可逐格式代入的闭式。

推导路线：先按特殊值约定定出指数字段的最大可用值，再乘上该指数下的最大尾数，得最大有限数；最小正规/次正规与 ULP 由值公式直接读出。关键一步是**特殊值约定决定指数字段顶端让出多少码位**——三种约定给三个不同的最大值闭式。

**命题 1**（范围/ULP 闭式）。设指数字段 $E$ 位、尾数 $M$ 位、偏置 $b$，则最小正规数 $2^{1-b}$、最小次正规数 $2^{1-b-M}$、1 附近间隔（ULP） $2^{-M}$、就近舍入最大相对误差 $2^{-M-1}$；最大有限数按特殊值约定分三式：

$$
x_{\max} = \begin{cases} (2 - 2^{-M}) \cdot 2^{2^{E} - 2 - b} & \text{IEEE（inf + NaN，顶端指数让出）} \\ (2 - 2^{-(M-1)}) \cdot 2^{2^{E} - 1 - b} & \text{canonical-NaN（无 inf，顶端尾数让出）} \\ (2 - 2^{-M}) \cdot 2^{2^{E} - 1 - b} & \text{NONE（无特殊值，全码位有限）} \end{cases} \qquad (1)
$$

**证明**（计数，三步）。其一，IEEE 约定指数字段全 1（ $2^{E}-1$）保留给 inf/NaN，最大正规指数 $2^{E}-2$，尾数全 1 可用，得 $(2-2^{-M}) \cdot 2^{2^{E}-2-b}$。其二，canonical-NaN 约定（E4M3/UE4M3）不表 inf，指数字段可取到全 1，但「全 1 指数 + 全 1 尾数」这一个码位保留为 NaN，故顶端指数下尾数最大为 $1.1\ldots10$（ $M-1$ 个 1），即 $2 - 2^{-(M-1)}$。其三，NONE 约定（E2M1）无任何特殊值，全 1 指数 + 全 1 尾数也是有限数，尾数全 1 可用。最小正规数取指数 1、尾数 0；次正规最小取指数 0、尾数最低位 1；ULP 与相对误差由尾数 LSB 权重 $2^{-M}$ 直接得。∎

正例（E4M3，canonical-NaN）： $E=4$、 $M=3$、 $b=7$， $x_{\max} = (2 - 2^{-2}) \cdot 2^{15-7} = 1.75 \times 256 = 448$——与 training 1.3 §1 表逐字一致。边界反例：若误用 NONE 闭式（无特殊值、全 1 指数 + 全 1 尾数当有限数）算 E4M3，得 $1.875 \times 256 = 480$，把一个 NaN 码位当成了有限数；若误用 IEEE 闭式（顶端指数让出给 inf/NaN）则得 $1.875 \times 128 = 240$，少算一个 binade——E4M3 实际走 canonical-NaN 式（448），两个误用方向都错。全格式代入（Python 独立重算，E4M3 另用 256 码位穷举核对最大值 448）：

| 格式 | 总位宽（s/e/m） | bias | 最大有限数 | 最小正规 | 最小次正规 | ULP@1 | 特殊值约定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 16（1/5/10） | 15 | 65504 | $2^{-14}$ | $2^{-24}$ | $2^{-10}$ | IEEE |
| BF16 | 16（1/8/7） | 127 | $3.39 \times 10^{38}$ | $2^{-126}$ | $2^{-133}$ | $2^{-7}$ | IEEE |
| TF32 | 19 有效（1/8/10） | 127 | $3.40 \times 10^{38}$ | $2^{-126}$ | （同 FP32 量级） | $2^{-10}$ | IEEE |
| FP8 E4M3 | 8（1/4/3） | 7 | 448 | $2^{-6}$ | $2^{-9}$ | $2^{-3}$ | canonical-NaN |
| FP8 E5M2 | 8（1/5/2） | 15 | 57344 | $2^{-14}$ | $2^{-16}$ | $2^{-2}$ | IEEE |
| FP4 E2M1 | 4（1/2/1） | 1 | 6 | $2^{0} = 1$ | $2^{-1} = 0.5$ | $2^{-1} = 0.5$ | NONE |

**符号**：s/e/m = 符号/指数/尾数位宽；ULP@1 = 1 附近相邻可表数间隔；TF32 是 32-bit 容器里的 19-bit 计算模式（低 13 位尾数在 Tensor Core 入口截断），非独立存储格式。位布局一眼对照：

```
FP16  [s:1][eeeee:5][mmmmmmmmmm:10]   bias 15   范围≈40 binade
BF16  [s:1][eeeeeeee:8][mmmmmmm:7]    bias 127  范围≈261 binade
TF32  [s:1][eeeeeeee:8][mmmmmmmmmm:10]（32-bit 容器，低 13 位截断）
E4M3  [s:1][eeee:4][mmm:3]            bias 7    无 inf，单一 NaN
E5M2  [s:1][eeeee:5][mm:2]            bias 15   inf/NaN 齐全
E2M1  [s:1][ee:2][m:1]                bias 1    无特殊值，16 码全有限
```

两个读数。**其一，BF16 与 FP16 同为 16 bit，分配相反**：BF16 拿 3 个尾数位换指数位，范围从约 40 binade 扩到约 261 binade（与 FP32 同量级），代价是 ULP 从 $2^{-10}$ 粗到 $2^{-7}$——这就是 bf16 训练免 loss scaling 的格式根据（training 1.3 §4 论证 recap，不重推）。**其二，E2M1 只有 16 个码位**，正数集 $\lbrace 0, 0.5, 1, 1.5, 2, 3, 4, 6 \rbrace$（穷举 16 码核对），ULP@1 = 0.5、最大相对舍入误差 $2^{-2}$（25%）——比 E4M3 的 6.25% 粗 4 倍。FP4 敢用于 GEMM，靠的不是格式本身精度，而是细粒度 scale 把每个元素的相对误差压住（§4）+ 权重对扰动的容忍度（量化算法侧归 inference 3.2）。

> **坑**：拿 NONE 闭式套 E4M3/UE4M3。触发条件：用 NONE 式 $(2-2^{-M}) \cdot 2^{2^{E}-1-b}$（全 1 指数 + 全 1 尾数当有限数）算 E4M3 最大值；现象：得 480 而非 448，把 canonical-NaN 码位当成了有限数（另一误用方向：套 IEEE 式 $(2-2^{-M}) \cdot 2^{2^{E}-2-b}$ 得 240，少一个 binade）；规避：E4M3/UE4M3 走 canonical-NaN 式（顶端尾数让出一个码位），E2M1 走 NONE 式，逐格式核对特殊值约定（CUTLASS `exmy_base.h` 的 `NanInfEncoding` 枚举逐字区分 IEEE_754 / CANONICAL_ONLY / NONE，[NVIDIA, CUTLASS](../references.md)，教学参照）。

INT8/INT4 不走浮点账，走**均匀格点**账。对称量化格点 $[-2^{q-1}, 2^{q-1}-1]$（INT8 $[-128,127]$、INT4 $[-8,7]$，负侧多一格），步长 $\Delta = \mathrm{absmax}/2^{q-1}$（AWQ 约定，inference 3.2 式 (1) recap）。关键差异：**浮点误差是相对的**（同一 binade 内相对误差恒为 $2^{-M-1}$，尺度不变），**整数误差是绝对的**（格点间距 $\Delta$ 固定，小元素相对误差大）。所以 INT 量化必须靠细粒度 scale 把 outlier 限制在小组内、保住组内小元素的有效级数——outlier 把 per-tensor max 抬到 100 倍时正常通道只剩约 2.56 个有效级（inference 3.2 §5 SmoothQuant 账 recap，不重推）。浮点格式（E4M3）天然按 binade 分配精度，对 outlier 的容忍度高于同位宽 INT，这是 FP8 逐步取代 INT8 成为训练/推理主路的格式侧根据之一。

## 3. 累加精度账：为什么必须 FP32 累加

1.3 标了「FP32 累加是指令语义」，把「为什么必须」留给了本章。低精度输入（E4M3 只有 3 位尾数）配 FP32 累加，是不是浪费？先回答一个更基本的问题： $K$ 个乘积沿归约维累加，累加器精度不够会怎样。

推导路线：单步舍入误差界 → 同号最坏情形线性叠加 → 与输入量化误差比大小。关键一步是分清**输入量化误差**（数据进格式时一次性付出）与**累加舍入误差**（每加一步付一次、随 $K$ 累积）——后者必须压到前者之下，累加才不是瓶颈。

设累加器有效尾数 $p$ 位（含隐含位：FP16 $p=11$、BF16 $p=8$、FP32 $p=24$），单位舍入 $u = 2^{-p}$。就近舍入下单步加法误差 $\le u \cdot |\text{部分和}|$。

**命题 2**（累加误差最坏界）。 $K$ 个同号项顺序累加、最坏情形（每步舍入同向、部分和单调增到终值 $S$）下，总相对误差界为 $(K-1) \cdot u$。**证明**（叠加，两步）：其一，第 $i$ 步部分和 $\le S$（同号），该步绝对误差 $\le u \cdot S$；其二， $K-1$ 步最坏同向叠加，总绝对误差 $\le (K-1) u S$，除以 $S$ 得相对界。∎（口径：最坏情形上界，非紧界；随机异号情形误差按约 $\sqrt{K} \cdot u$ 随机游走增长，量级更小，但最坏界给出「不可用」的判据已足够。）

全数字代入（ $K = 4096$，脊柱层归约深度）：FP16 累加 $(K-1) u = 4095 \times 2^{-11} \approx 2.0$——界超过 100%，意味着最坏情形下结果可以完全失真；BF16 累加 $\approx 16$，更不可用；FP32 累加 $4095 \times 2^{-24} \approx 2.44 \times 10^{-4}$（0.024%）。对照输入量化误差：BF16 输入 $2^{-8} = 3.9 \times 10^{-3}$、E4M3 输入 $2^{-4} = 6.25 \times 10^{-2}$。**FP32 累加误差只有 BF16 输入误差的约 1/16、E4M3 输入误差的约 1/256**——累加这一步被输入量化误差完全盖过，不再是精度瓶颈。这就是「低精度输入 + FP32 累加」的定量根据：累加器精度买到的是「误差不在归约维累积放大」，而非更高的输入精度。

边界反例：FP16 累加并非处处禁用—— $K$ 很小（如 $K=16$ 的短归约）时界 $15 \times 2^{-11} \approx 0.007$（0.7%），某些容忍度高的 epilogue 可用；但 GEMM 主循环 $K = 4096$ 量级必须 FP32。另有一层硬件事实：Tensor Core 内部累加链的有效精度低于名义 FP32（V3 实测约 14 bit，故每 128 个元素把部分积提升到 CUDA Core 的 FP32 主累加器，training 1.3 §6 recap 不重推）——这条「两级累加」的 kernel 实现（DeepGEMM promotion）归 5.2 §10 与 6.2，本章只立精度账。

## 4. MX 族与 NVFP4：scale 编码与粒度开销账

FP4 只有 16 个值、最大相对误差 25%，怎么压住量化误差？答案是块缩放（block scaling）：每若干个元素共享一个 scale，把局部动态范围收进格式窗口。inference 3.2 §2 已交付格式事实与每元素字节账，本节 recap 其结论、增量是 **scale 编码本身的结构**与**粒度-精度权衡的定量账**。

格式事实 recap（inference 3.2 §2，不重推）：**OCP MX v1.0**（[OCP, 2023](../references.md)，[C]）规定 MXFP8/MXFP6/MXFP4/MXINT8 一律块大小 32 + E8M0 scale；MXFP4 元素为 E2M1。**NVFP4 非 OCP MX v1.0 格式**——E2M1 元素 + 每 16 元素一个 UE4M3 block scale + tensor 级 FP32 global scale，归 NVIDIA 官方文档与 PTX ISA（[NVIDIA, PTX ISA](../references.md)，[C]，`tcgen05.mma ... kind::mxf4nvf4`，`.e2m1` 元素 `.ue4m3` scale、`scale_vec::4X`、block16）。两种 scale 编码的结构：

- **E8M0**（MX 族）：8-bit 无符号**纯指数**，无符号位无尾数，值 $2^{e-127}$（bias 127）， $e \in [0, 254]$，0xFF 保留为 NaN；范围 $2^{-127}$ 到 $2^{127}$。scale 恒为 2 的整数幂——乘 scale 是精确的指数加减，不引入尾数舍入（inference 3.3 ue8m0 同理）。
- **UE4M3**（NVFP4）：**7-bit 无符号**浮点（4 指数 + 3 尾数，bias 7），装进 8-bit 容器、MSB 补零；有限值集与 E4M3 正半轴相同，最大 448、最小次正规 $2^{-9}$。CUTLASS 编码定义逐字核对：`UE4M3` = `FpBitRepresentation<uint8_t, 8, 4, 3, CANONICAL_ONLY, false>`（[NVIDIA, CUTLASS](../references.md)，教学参照，`exmy_base.h`）——末位 `false` 即无符号位，NaN 约定与 E4M3 同（canonical）。

> **坑**：把 UE4M3 读成 E4M3。触发条件：跨文档混读 NVFP4 scale 类型；现象：UE4M3 是 7-bit 无符号（8-bit 容器 MSB 补零），E4M3 是 8-bit 有符号元素格式，两者位宽与语义都不同；后果：scale 字节账与元素类型混淆，每元素字节虚高。规避：格式三元组（元素类型、块大小、scale 类型）分开读，NVFP4 = E2M1 元素 / block16 / UE4M3 scale（inference 3.2 §2 同坑）。

粒度开销账 recap inference 3.2 式 (3)（每元素字节 = (块数据 + scale + zero-point)/块元素数，不重推），代入脊柱：

| 格式 | 元素 | 块大小 | scale | 每元素字节 | 对 bf16 压缩 | scale/data 开销 |
| --- | --- | --- | --- | --- | --- | --- |
| MXFP8 | E4M3（1 B） | 32 | E8M0（1 B） | 1.03125 | 1.94× | 3.125% |
| MXFP4 | E2M1（0.5 B） | 32 | E8M0（1 B） | 0.53125 | 3.76× | 6.25% |
| NVFP4 | E2M1（0.5 B） | 16 | UE4M3（1 B）+ FP32 global | 0.5625 | 3.56× | 12.5% |
| INT4-g128 | INT4（0.5 B） | 128 | FP16（2 B） | 0.515625 | 3.88× | 3.125% |

表内每元素字节：MXFP8 $(32 \times 1 + 1)/32 = 1.03125$；MXFP4 $(32 \times 0.5 + 1)/32 = 0.53125$；NVFP4 $(16 \times 0.5 + 1)/16 = 0.5625$（global scale 每 tensor 4 字节，长 tensor 摊薄可略）；INT4-g128 $0.5 + 2/128 = 0.515625$。块布局一眼对照：

```
MXFP4 块（32 元素）：[E8M0 scale 1B][E2M1 ×32 = 16B]  → 17B/32 = 0.53125 B/元素
NVFP4 块（16 元素）：[UE4M3 scale 1B][E2M1 ×16 = 8B]  → 9B/16 = 0.5625 B/元素
                     + tensor 级 FP32 global scale 4B（摊薄）
```

**粒度-精度权衡的定量账**（本章增量）。E8M0 只能表 2 的幂，相邻 scale 比值 2（100% 台阶）；UE4M3 带 3 位尾数，同一 binade 内相邻 scale 比值 $1 + 1/8 = 1.125$（12.5% 台阶），细 8 倍。量化须把 scale 向下取到可表值以避免元素溢出，于是 absmax 落在两个可表 scale 之间时动态范围被浪费：E8M0 最坏利用率刚过 50%（absmax 略低于 $2s$），UE4M3 最坏刚过 $8/9$（约 88.9%）。全数字实例（E2M1 元素最大 $V = 6$，组 absmax $a = 6.1$）：需 $s \ge a/V = 1.0167$；E8M0 取 $s = 2$，利用率 $6.1/(2 \times 6) = 0.508$（50.8%）；UE4M3 取 $s = 1.125$，利用率 $6.1/(1.125 \times 6) = 0.904$（90.4%）。**NVFP4 用 +5.9% 字节（ $0.5625/0.53125 = 1.0588$）买到 8 倍细的 scale 台阶 + 2 倍细的块**——这是它比 MXFP4 压缩比略低（3.56× 对 3.76×）却精度更稳的账。两级 scale 分工 recap inference 3.2：UE4M3 范围 $[2^{-9}, 448]$ 单级罩不住整 tensor 动态范围，FP32 global scale 把各块平移进 UE4M3 窗口，block scale 管局部极差。

## 5. 逐格式 MMA 吞吐：datasheet 对账

哪个格式从哪代起有硬件支持、峰值多少？支持矩阵 recap 1.3 §7（不重推），本节增量是**逐格式吞吐列**与**指令几何对账**。交叉表（✓ = 官方表列出或 PTX 指令族支持；△ = PTX 支持但官方表未列独立峰值；— = 不支持/未列；V100/A100 列按 PTX target arch 反推，Hopper/Blackwell 取官方表）：

| 格式 | V100 | A100 | H100 | B200 | H100 dense | B200 dense | 对 BF16 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP64（TC） | — | ✓ | ✓ | ✓ | 66.9 | 37 | 1/14.8 |
| TF32 | — | ✓ | ✓ | ✓ | 494.7 | 1100 | ≈1/2 |
| FP16 | ✓ | ✓ | ✓ | ✓ | 989.4 | 2250 | 1（基准） |
| BF16 | — | ✓ | ✓ | ✓ | 989.4 | 2250 | 1（基准） |
| FP8 E4M3/E5M2 | — | — | ✓ | ✓ | 1978.9 | 4500 | 2 |
| INT8 | — | ✓ | ✓ | ✓ | 1979 TOPS | 4500 TOPS | 2 |
| FP6 | — | — | △ | ✓ | 未列 | 未列 | （同 FP8 路径） |
| FP4 E2M1 | — | — | △ | ✓ | 未列 | 9000 | 4 |
| INT4 | — | ✓ | 未列 | 未列 | — | — | （W4A16 存储格式） |

**口径注**（四层）：其一，H100 列取 whitepaper Table 3 逐精度峰值（[NVIDIA, Hopper whitepaper](../references.md)，[C]），B200 列取 datasheet dense（sparse 表值之半，[NVIDIA, datasheet](../references.md)，[C]）；H100 wp 989.4 与 datasheet 989.5 是舍入路径差（datasheet = sparse 1979 ÷ 2），1.1 §6 已注。其二，B200 TF32 1100/2250 = 0.489 ≈ 1/2（sparse 2.2 PF ÷ 2 的舍入），FP64 37 对 BF16 2250 约 1/61。其三，FP6/FP4 的 H100 △：wgmma `kind::f8f6f4` 名字含 FP8/FP6/FP4 三族（PTX 逐字），但 Hopper 官方表只列 FP8 独立峰值；Blackwell 才把 FP4/FP6 列为独立格式。INT4 在 Hopper/Blackwell「未列」是披露口径陈述，非硬件绝对断言。其四，**FP16/BF16 拆两行**：V100（sm_70）有 FP16 Tensor Core（mma.m8n8k4）但**无 BF16**（BF16 mma 需 sm_80），合并行会掩盖这一差异；**B200 INT8 峰值已披露**——datasheet「Individual Blackwell GPU Specifications」HGX B200 列 INT8 Tensor Core 9 POPS（sparse）÷ 2 = 4500 TOPS dense，对 BF16 比 2，「峰值未列」口径仅适用 INT4。

**命题 3**（吞吐比 = 指令 k 几何比，对账口径）。同 $m$、 $n$ 几何、稳态发射间隔不变时，两格式的 MMA 吞吐比等于其指令 $k$ 维之比。**证明**（recap 1.3 式 (1)(2) 代入，两步）：其一，指令 FLOPs $F_{\mathrm{instr}} = 2mnk$（1.3 式 (1)），稳态发射间隔 $t_{\mathrm{issue}} = F_{\mathrm{instr}}/c$（1.3 式 (2)， $c$ 为执行单元每周期吞吐）；其二， $t_{\mathrm{issue}}$ 相同则吞吐比 $= c$ 比 $= F_{\mathrm{instr}}$ 比 $= k$ 比。∎（口径：这是**一致性对账**——datasheet [C] 是吞吐的权威来源，指令几何解释其机制，非独立推导硅片吞吐。）

全数字对账。H100：wgmma BF16 m64n128k16 的 $F_{\mathrm{instr}} = 2 \times 64 \times 128 \times 16 = 262144$， $c_{\mathrm{SM}} = 4096$ FLOPs/cycle（1.3 recap）， $t_{\mathrm{issue}} = 64$ cycles；FP8 同形 m64n128k32（`kind::f8f6f4`，k 翻倍） $F_{\mathrm{instr}} = 524288$， $c_{\mathrm{SM}} = 8192$（2 倍）， $t_{\mathrm{issue}}$ 仍 64 cycles——datasheet 比 $1978.9/989.4 = 2.0001$ 与 k 比 $32/16 = 2$ 闭环。B200：tcgen05 的 MXF4 专用路径 UMMA_K = 64、MXF8F6F4 路径 UMMA_K = 32（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照，`sm100_fp8_fp4_gemm_1d1d.cuh` L68 `UMMA_K = kIsMXF4 ? 64 : 32`），BF16 k16——k 比 $64/16 = 4$ 对 datasheet $9000/2250 = 4$。机制一句话：**操作数位宽减半，固定宽度的乘加 datapath 每周期处理 2 倍元素**，指令 k 几何随之翻倍，发射间隔不变即吞吐翻倍。

> **坑**：以为 FP4 元素在任何路径都跑 4 倍。触发条件：把 FP4 操作数喂进 MXF8F6F4 unpacked 路径；现象：该路径 UMMA_K = 32（与 FP8 同），FP4 只跑 2 倍而非 4 倍；规避：4 倍须走 packed MXF4/NVFP4 专用 MMA（UMMA_K = 64）。DeepGEMM 注释逐字：「an FP4xFP4 pair uses the dedicated MXF4 MMA (packed 4-bit operands, K-major only, UMMA_K=64); all other combinations use the MXF8F6F4 MMA (unpacked operands, UMMA_K=32)」（`sm100_fp8_fp4_gemm_1d1d.cuh` L51-52），且 MXF4 断言 BLOCK_K = 256、K-major、gran_k = 32（L77/L84-85）。FP6 走 unpacked 路径，无独立峰值（同 FP8 速率）。

INT4 的硬件路径要单独钉一句：PTX 在 sm_75/sm_80 族列出原生 INT4 MMA（1.3 矩阵），但 Hopper/Blackwell 官方表未列独立 INT4 峰值——**INT4 在 Hopper/Blackwell 主要是存储格式而非计算格式**。W4A16 部署把 INT4 权重 dequant 成 FP16 再走 FP16 MMA（Marlin 路线，6.2 走读），省的是权重读带宽，算力仍按 FP16 峰值计。这与 FP8/FP4 的「原生低精度 MMA」是两条不同的硬件路径，§6 汇总。

## 6. scale 消费路径与格式选择：6.2 回链

scale 算出来之后由谁乘回去？三条硬件消费路径，对应三种 kernel 结构：

| 路径 | scale 粒度 | 施加位置 | 代表 | 本章/下游 |
| --- | --- | --- | --- | --- |
| ① epilogue 全局 scale | per-tensor / per-channel | GEMM 收尾一次性乘 | W8A8 INT8（SmoothQuant） | inference 3.2 §5 recap |
| ② 软件 mainloop 分段提升 | fine-grained（沿 K 每 128） | 每段乘 scale 提升到 FP32 主累加器 | DeepGEMM SM90 FP8 | 5.2 §10 / 6.2 走读 |
| ③ 硬件 block-scaling MMA | per-32 / per-16 块 | Tensor Core 指令原生消费 SFA/SFB | Blackwell tcgen05 MX/NVFP4 | 本节 + 6.2 |

路径①最便宜（scale 不进 mainloop），但粒度粗、outlier 容忍度低；路径②把细粒度 scale 折进累加结构——scale 沿 K 每 128 元素一变，无法折进一个事后全局 scale，必须分段累加、段末乘 $`\mathrm{scale}_a \times \mathrm{scale}_b`$ 提升（5.2 §10 已走读 DeepGEMM promotion L329-345，6.2 补 scale 读取与谓词选择的指令级增量）；路径③把 scale 喂进指令本身——DeepGEMM sm100 kernel 签名含 `tensor_map_sfa`/`tensor_map_sfb`（`sm100_fp8_fp4_gemm_1d1d.cuh` L44-45），SFA/SFB 经 TMA 载入、tcgen05 block-scaling MMA 原生消费，软件免手工乘 scale。**NVFP4/MXFP4 的块 scale 走路径③，这是 Blackwell FP4 吞吐翻倍的指令基础**（1.3 §6 已标 `kind::mxf4nvf4`）；而 INT4 在 Hopper 无原生 MMA，dequant 必须软件做进 mainloop（路径②的变体，6.2 Marlin lop3 走读）。

格式选择决策表（硬件路径增量；量化算法怎么选归 inference 3.2，不重推）：

| 场景 | 格式 | 硬件路径 | 依据 |
| --- | --- | --- | --- |
| decode weight-only（小 batch） | INT4 / NVFP4 存储 | Hopper：dequant→FP16 MMA（6.2）；Blackwell：FP4 原生 block-scaling | §5 INT4 段 + §6 路径③ |
| prefill / 大 batch dense | FP8 E4M3 | 原生 FP8 MMA（2 倍峰值）+ fine-grained scale | §5 命题 3 |
| 训练主路 | BF16 / FP8 blockwise | BF16 免 scaling；FP8 两级累加 | training 1.3 recap |
| MoE routed expert | FP4（MXFP4/NVFP4） | Blackwell 原生 block-scaling（4 倍） | §4 + §5 |
| KV cache | FP8 / FP4 | 量化格式账归 inference 2.3 | 指针 |

收口回链 6.2：本章交付了格式账（位布局/范围/ULP/累加精度/scale 编码/吞吐对账/消费路径），6.2 在此基础上走读 dequant 融合 kernel 的指令级实现——INT4 怎么用 lop3 + magic-number 在寄存器里即时还原成 FP16（Marlin）、fine-grained scale 怎么读进 mainloop 并分段提升（DeepGEMM）、dequant 的带宽节省与指令开销怎么算账。格式选择最终落到「精度 × 硬件 × batch 形态」三元组，算法侧与算子语义侧的完整决策表见 inference 3.2 §8 与 3.3 §8。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 范围/ULP 闭式按特殊值约定分三式；E4M3 canonical-NaN 最大 448（非 NONE 式 480、IEEE 式 240）、E2M1 NONE 最大 6 | 命题 1、式 (1) | §2、练习 1 |
| BF16 与 FP16 同 16 bit 分配相反：范围 ≈261 对 ≈40 binade，ULP $2^{-7}$ 对 $2^{-10}$；bf16 免 scaling 根据 | §2 表 + 读数 | §2、training 1.3 §4 |
| FP32 累加误差 $(K-1)u$：K=4096 时 2.44e-4，为 BF16 输入误差 1/16、E4M3 的 1/256；FP16 累加界 ≈2.0 不可用 | 命题 2 | §3、练习 2 |
| E2M1 正数集 {0,0.5,1,1.5,2,3,4,6}，最大相对舍入误差 25%；FP4 靠细粒度 scale + 权重容忍度 | §2 穷举 | §2、§4 |
| MXFP8/MXFP4/NVFP4 每元素 1.03125/0.53125/0.5625 B，压缩 1.94/3.76/3.56×；NVFP4 +5.9% 字节买 8 倍细 scale 台阶 | §4 表（inference 3.2 式 (3) recap） | §4、练习 3 |
| E8M0 纯指数（bias 127，乘 = 指数加）；UE4M3 = 7-bit 无符号 E4M3（8-bit 容器 MSB 补零），非 E4M3 | §4 编码 | §4、练习 3 |
| E8M0 scale 台阶 100%、UE4M3 12.5%；absmax 6.1 利用率 50.8% 对 90.4% | §4 权衡账 | §4、练习 3 |
| 吞吐比 = k 几何比（对账口径）：H100 FP8/BF16 = 2 ↔ k32/k16；B200 FP4/BF16 = 4 ↔ UMMA_K 64/16 | 命题 3 | §5、练习 4 |
| FP4 走 MXF8F6F4 unpacked 路径只 2 倍（UMMA_K 32），4 倍须 packed MXF4/NVFP4（UMMA_K 64） | §5 坑注（DeepGEMM L51-52/L68） | §5、练习 4 |
| INT4 在 Hopper/Blackwell 是存储格式非计算格式；W4A16 dequant→FP16 MMA（6.2） | §5 INT4 段 | §5、6.2 |
| scale 三消费路径：epilogue 全局 / 软件 mainloop 提升 / 硬件 block-scaling MMA；NVFP4 走路径③ | §6 表 | §6、6.2 |

## 练习

1.（计算 · [10]）范围闭式代入（式 (1)）：(a) 用 canonical-NaN 式算 E4M3 的最大有限数、最小次正规数、ULP@1；(b) 穷举 E2M1 的 16 个码位写出正数集与最大值；(c) 用 IEEE 式算 FP16 最大值。校验：(a) 448 / $2^{-9}$ / $2^{-3}$，与 training 1.3 §1 表一致；(b) $\lbrace 0,0.5,1,1.5,2,3,4,6 \rbrace$、最大 6；(c) 65504 = $(2-2^{-10}) \cdot 2^{15}$。
2.（推导 · [20]）补全命题 2 的证明（累加误差界）：(a) 写出单步就近舍入误差界 $\le u \cdot |\text{部分和}|$；(b) 同号最坏情形叠加得 $(K-1)u$，说明为何这是上界而非紧界；(c) 代入 $K=4096$ 算 FP16/FP32 累加界，再与 BF16/E4M3 输入量化误差比大小，验证「FP32 累加不是瓶颈」。校验：(c) FP16 ≈2.0、FP32 ≈2.44e-4；比值 16×/256×（§3）；随机异号 $\sqrt{K}u$ 口径与最坏界的差异见 §3 口径注。
3.（计算 · [20]）scale 粒度账：(a) 按 inference 3.2 式 (3) 算 MXFP8/MXFP4/NVFP4 每元素字节与对 bf16 压缩比；(b) 组 absmax $a=6.1$、E2M1 元素最大 $V=6$，分别算 E8M0 与 UE4M3 向下取 scale 后的动态范围利用率；(c) 算 NVFP4 对 MXFP4 的字节增量百分比，说明买到了什么。校验：(a) 1.03125/0.53125/0.5625、1.94/3.76/3.56×，与 inference 3.2 §2 表一致；(b) 50.8% / 90.4%（§4）；(c) +5.9%、8 倍细 scale 台阶 + 2 倍细块。
4.（计算 · [30]）吞吐对账（命题 3）：(a) 算 H100 FP8/BF16、TF32/BF16 与 B200 FP8/BF16、FP4/BF16 的 datasheet 比；(b) 算 wgmma FP8 m64n128k32 的 $F_{\mathrm{instr}}$ 与 $t_{\mathrm{issue}}$（ $c_{\mathrm{SM}} = 8192$），核对与 BF16 m64n128k16 同发射间隔；(c) 先预测「FP4 操作数走 MXF8F6F4 unpacked 路径的吞吐倍数」再按 UMMA_K 说明；(d) 核对 B200 FP4/FP8 比。校验：(a) 2 / 0.5 / 2 / 4（§5 表）；(b) 524288 / 64 cycles（§5）；(c) 2 倍（UMMA_K 32，§5 坑注）；(d) 9000/4500 = 2。

## 参考

- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——wgmma m64nNk16（bf16/f16）/ m64nNk32（kind::f8f6f4）/ m64nNk8（tf32）；tcgen05.mma kind::mxf4nvf4（.e2m1 元素 .ue4m3 scale、scale_vec::4X、block16）；低精度格式节（E4M3/E5M2/E2M1/E8M0/UE4M3）
- [OCP, 2023](../references.md)：OCP Microscaling Formats (MX) Specification v1.0（key: ocp-mx），[C] 官方规格披露·一手权威，2026-11-24——MXFP8/MXFP6/MXFP4/MXINT8 块 32 + E8M0 scale、MXFP4 元素 E2M1；NVFP4 非 OCP MX v1.0 格式
- [NVIDIA, datasheet](../references.md)：NVIDIA H100/Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM BF16 989.5 / FP8 1979 / TF32 494.5 dense（sparse 之半）；HGX B200 BF16 2250 / FP8 4500 / FP4 9000 / TF32 1100 / FP64 37 / INT8 4500 TOPS（9 POPS sparse ÷ 2）dense；SKU 口径沿 1.1 §4
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——Table 3 逐精度峰值（FP64 TC 66.9 / TF32 494.7 / BF16-FP16 989.4 / FP8 1978.9 / INT8 1979 TOPS）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——第 5 代 Tensor Core FP4/FP6 + OCP MX、Table 1 数据类型
- [NVIDIA, CUTLASS](../references.md)：NVIDIA/cutlass repo（key: cutlass），教学参照 + 官方实现，2026-11-25——浮点编码 include/cutlass/exmy_base.h（NanInfEncoding 枚举 IEEE_754/CANONICAL_ONLY/NONE；UE4M3 = FpBitRepresentation<uint8_t,8,4,3,CANONICAL_ONLY,false> 无符号、E2M1 NONE、UE8M0 纯指数；exponent_bias = 2^(E-1)-1）
- [DeepSeek-AI, DeepGEMM](../references.md)：deepseek-ai/DeepGEMM repo（key: deepgemm），教学参照 + 官方实现，版本锚 commit 5081f8513af5（2026-11-25）——sm100_fp8_fp4_gemm_1d1d.cuh（MXF4 vs MXF8F6F4 注释 L51-52、UMMA_K = kIsMXF4?64:32 L68、BLOCK_K 256/128 L77、K-major + gran_k 32 L84-85、tensor_map_sfa/sfb L44-45）；格式位布局与逐格式吞吐本章增量，两级累加走读归 5.2 §10 / 6.2
