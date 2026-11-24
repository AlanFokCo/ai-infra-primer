# 1.1 芯片组织与世代（Chip Organization and Generations）

> 一块 GPU 的峰值算力由哪三个物理因子乘出来，世代升级各动了哪个因子？本章从 GPC/TPC/SM 三级分组走到 die/chiplet/CoWoS 封装，交付芯片级 FLOPS 反推式与 Hopper→Blackwell 世代对比表（SKU 口径锁定） | 前置：models 1.2（FLOPs 账）、inference 1.3（datasheet 参数表——本章接其微架构来源，规格事实同表不重列）；下游 1.2（SM 内部）、2.1（HBM 栈几何逐项）、2.4（roofline 微架构来源与测量） | 锚点：NVIDIA Hopper whitepaper V1.04（[NVIDIA, Hopper whitepaper](../references.md)，[C] 官方规格披露·一手权威）、Blackwell Technical Brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、H100/H200/HGX/Blackwell datasheet（[NVIDIA, datasheet](../references.md)，[C]）、CUDA Programming Guide（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、Hopper/Blackwell Tuning Guide（[NVIDIA, Hopper tuning guide](../references.md)，[C]）、Hopper blog（[NVIDIA, Hopper blog](../references.md)，[C] 开放技术博客）——全部 2026-11-25 一手核实（dam-cdn 官方 PDF）

## 1. 峰值算力是乘出来的：本章口径与三个因子

同一张 H100 datasheet 上，BF16 Tensor Core dense 989.5 TFLOPS 与 FP32 67 TFLOPS 差约 15 倍。这 15 倍不是「优化程度」的差，而是三个物理因子的乘积差：SM 数、每 SM 每周期该精度类的 FLOPs、该精度类的 boost 频率。本章回答三个问题：这三个因子各从官方规格表的哪一栏读？从规格反推微架构参数的式子怎么写？Hopper→Blackwell 的世代升级动了哪个因子？

**本章口径**（章脊锁定；本章为世代对比章，多 SKU 口径在此一次声明，表外不再换）：

| 口径 | SKU 与几何 | 精度与单位 | 来源 |
| --- | --- | --- | --- |
| 主口径 | H100 SXM5 80 GB：132 SM / 66 TPC，BF16 dense 989.5 TFLOPS，HBM3 3.35 TB/s，TDP ≤700 W | dense（官方 sparse 表值之半）；BF16 为主精度，FP32/FP8 对照 | [NVIDIA, Hopper whitepaper](../references.md) Table 3 + [NVIDIA, datasheet](../references.md)，[C] |
| 对比口径 | H100 PCIe（114 SM）、H200 SXM（141 GB / 4.8 TB/s）、HGX B200（180 GB / 7.7 TB/s）、GB200 每 GPU（186 GB / 8 TB/s） | 同 dense；GB = $10^{9}$ B、TB/s = $10^{12}$ B/s | 同上四份 datasheet，[C] |
| 反推口径 | B200 SM 数 148、每 SM 每周期 FLOPs、boost clock | 官方未披露项标「反推」+ ⟨verify W3/W6⟩ | §3 / §4 自洽性核对 |

单位纪律：显存 GB 按 $10^{9}$ B、带宽 TB/s 按 $10^{12}$ B/s（datasheet 十进制口径）；片上存储 KB/K 按 1024（Programming Guide 逐字注「KB and K units … correspond to 1024 bytes (KiB) and 1024」），GiB 按 $2^{30}$ B，混用处逐一注明。符号： $n_{\mathrm{SM}}$ = SM 数； $F_{\mathrm{peak}}$ = 峰值 FLOPS（dense）； $f_{\mathrm{prec}}$ = 该精度类 boost 频率； $c_{\mathrm{prec}}$ = 每 SM 每周期该精度类 FLOPs（本章反推量）； $\mathrm{BW}$ = 带宽。 $F_{\mathrm{peak}}$ 是 roofline 的分子（models 2.4 §6 入口，gpu 2.4 recap），本章交付它的微架构来源，不重推 roofline；带宽的微架构来源（HBM 栈几何逐项）归 2.1，本章只取芯片级结果。

## 2. GPC/TPC/SM：从 die 到执行单元的三级分组

一颗 GH100 die 上有多少个 SM，出货的 H100 就有多少个吗？不是。full die 144 SM，H100 SXM5 出货 132 SM——差的 12 个去哪了，本节用三级分组与 harvest 账回答。

所谓 SM（streaming multiprocessor，流多处理器），是指 GPU 内可独立调度与执行 warp 的最小完整单元：它自带 warp scheduler、register file、shared memory 与执行管线，一个 thread block 整体驻留在一个 SM 上、不可跨 SM 拆分（内部拆解归 1.2，block 调度语义归 3.1）。SM 之上还有两级物理分组：所谓 TPC（Texture Processing Cluster），是指 2 个 SM 加共享纹理资源的分组；所谓 GPC（Graphics Processing Cluster），是指一组 TPC 构成的更大物理分组（GH100 每 GPC 含 9 TPC）。这两级是**物理分组与 harvest 粒度**，不是编程模型层级——CUDA 的 threadIdx/blockIdx 索引里不出现 GPC/TPC，程序员能调度的最小驻留单位是 SM 上的 block。

GH100 full die 的分组是 8 GPC × 9 TPC × 2 SM = 144 SM（whitepaper Fig 6，[NVIDIA, Hopper whitepaper](../references.md)，[C]）。出货 SKU 按 TPC 粒度屏蔽（harvest）：H100 SXM5 = 66 TPC / 132 SM，H100 PCIe = 57 TPC / 114 SM（Table 3）。harvest 账：144 − 132 = 12 SM = 6 TPC 被屏蔽，132/144 = 91.7% 的 SM 可用。正例：66 TPC × 2 SM/TPC = 132 SM，与 Table 3 逐字一致。反例：把 full die 的 144 当 H100 SXM5 的 SM 数代入峰值式，BF16 峰值会高估 144/132 = 9.1%（1079 对 989.5 TFLOPS）——harvest 是规格表读数第一个要核对的口径。

> **提点**：harvest 的机制原因是良率。die 面积越大，单颗 die 上落进致命缺陷的概率越高；把整颗 die 报废不如屏蔽坏掉的 TPC、让剩下的 SM 照常出货。所以同一颗 GH100 设计会分出 132 SM（SXM5）与 114 SM（PCIe）两档——不是两个芯片设计，是同一设计的两档可用 SM 数。这解释了为什么 §3 的比例律里 $c_{\mathrm{prec}}$ 能当公共因子对消：两档 SKU 的每 SM 微架构完全相同，差的只是 SM 数与 clock。

芯片级框图（whitepaper Fig 6 重绘，[C]；L2 两分区与 partitioned crossbar 为 Fig 6 与 prose 披露）：

```
GH100 full die（8 GPC × 9 TPC × 2 SM = 144 SM）
┌────────────────────────────────────────────────────────────────────┐
│  GPC 0       GPC 1       GPC 2       GPC 3     ← 每 GPC 9 TPC      │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    每 TPC 2 SM        │
│ │ TPC×9  │ │ TPC×9  │ │ TPC×9  │ │ TPC×9  │    = 18 SM/GPC        │
│ │ SM SM… │ │ SM SM… │ │ SM SM… │ │ SM SM… │                      │
│ └────────┘ └────────┘ └────────┘ └────────┘                        │
│  GPC 4       GPC 5       GPC 6       GPC 7                          │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                        │
│ │ TPC×9  │ │ TPC×9  │ │ TPC×9  │ │ TPC×9  │                        │
│ └────────┘ └────────┘ └────────┘ └────────┘                        │
│ ══════════════════ 分区 crossbar ══════════════════                 │
│ ┌──────────────────┐      ┌──────────────────┐                     │
│ │   L2 分区 0      │◄────►│   L2 分区 1      │  合计 50 MB         │
│ └──────────────────┘      └──────────────────┘                     │
│  MC×6（每控制器 512-bit）      MC×6     ← full：12 控制器 / 6 HBM3 栈│
│  NVLink ×18 links             PCIe Gen5 x16                        │
└────────────────────────────────────────────────────────────────────┘
H100 SXM5 出货：屏蔽 6 TPC（12 SM）→ 66 TPC / 132 SM；5 HBM3 栈 / 10 控制器
H100 PCIe 出货：57 TPC / 114 SM
```

每 SM 的执行单元数（Table 3，[C]）：FP32 CUDA Core 128、FP64 单元 64、INT32 单元 64、第 4 代 Tensor Core 4 个——全 GPU 528 个 Tensor Core = 132 × 4。这张单元数表是 §3 反推每 SM 每周期 FLOPs 的物理依据：FP32 的 128 个 lane 每周期各做 1 条 FMA（2 FLOPs），直接给出每 SM 每周期 256 FP32 FLOPs；Tensor Core 的每周期吞吐则须从官方峰值反推（§3）。

> **提点**：GPC/TPC 对 AI kernel 并非完全不可见。Hopper 引入 thread block cluster：一组 block 驻留在物理相邻的 SM 上，可经 DSMEM（distributed shared memory）互访对方的 shared memory（[NVIDIA, Hopper tuning guide](../references.md)，[C]）。cluster 的编程模型语义归 3.1、异步拷贝与 DSMEM 的 kernel 用法归 4.4，本章只记「物理分组在 Hopper 起被部分暴露给编程模型」这一笔。

## 3. 芯片级 FLOPS 反推：三个因子的乘积

H100 的 BF16 峰值 989.5 TFLOPS 是怎么从 132 个 SM 乘出来的？推导路线分三步：先写乘积式并做量纲对账，再按精度类从官方峰值反推每 SM 每周期 FLOPs，最后用逐精度 boost clock 对账。关键一步是第二步的口径——**频率依精度类而异**，Tensor Core 精度类与 FP32/FP64 类用不同的 boost clock，混用即高估。

**命题 1**（峰值乘积式）。芯片级 dense 峰值算力等于 SM 数、每 SM 每周期该精度类 FLOPs、该精度类 boost 频率三者之积：

$$
F_{\mathrm{peak}} = n_{\mathrm{SM}} \cdot c_{\mathrm{prec}} \cdot f_{\mathrm{prec}} \qquad (1)
$$

**符号**： $F_{\mathrm{peak}}$ = dense 峰值 FLOPS； $n_{\mathrm{SM}}$ = 出货 SM 数； $c_{\mathrm{prec}}$ = 每 SM 每周期该精度类 FLOPs； $f_{\mathrm{prec}}$ = 该精度类 boost 频率（Hz）。**证明**（量纲对账）： $[\mathrm{SM}] \times [\mathrm{FLOPs}/(\mathrm{SM} \cdot \mathrm{cycle})] \times [\mathrm{cycle}/\mathrm{s}] = \mathrm{FLOPs}/\mathrm{s}$，三项量纲恰好消成峰值算力单位。这是结构恒等式而非拟合：峰值算力本来就定义为「全部 SM 满负荷、每周期产出 $c_{\mathrm{prec}}$ 个 FLOP、每秒 $f_{\mathrm{prec}}$ 个周期」的乘积，官方 datasheet 的峰值数只是把这三个因子乘好印出来。∎

式 (1) 反过来读就是反推式——官方披露 $F_{\mathrm{peak}}$ 与 $n_{\mathrm{SM}}$、 $f_{\mathrm{prec}}$，每 SM 每周期吞吐是解出来的：

$$
c_{\mathrm{prec}} = \frac{F_{\mathrm{peak}}}{n_{\mathrm{SM}} \cdot f_{\mathrm{prec}}} \qquad (2)
$$

全数字代入（H100 SXM5 主口径）。BF16 Tensor Core：whitepaper Table 3 披露 dense 989.4 TFLOPS、TC-boost clock 1830 MHz（逐字「GPU Boost Clock for FP8, FP16, BF16, TF32 Tensor Core Ops」），代入式 (2) 得 $c_{\mathrm{BF16}} = 989.4 \times 10^{12}/(132 \times 1.83 \times 10^{9}) = 4096$ FLOPs/cycle/SM。按每 SM 4 个 Tensor Core 分摊，每 TC 每周期 1024 BF16 FLOPs（反推）。等价指令口径：一条 mma.sync m16n8k16 含 $16 \times 8 \times 16 \times 2 = 4096$ FLOPs（[NVIDIA, PTX ISA](../references.md)，[C]），按每 TC 每周期 1024 FLOPs 折 4 个周期——指令级吞吐与延迟的实测核对归 1.3 / 1.4，本章只交付芯片级反推。FP32：Table 3 披露 66.9 TFLOPS、FP32/FP64-boost clock 1980 MHz（逐字「for FP64 Tensor Core Ops, FP32 and FP64 non-Tensor Core Ops」）， $c_{\mathrm{FP32}} = 66.9 \times 10^{12}/(132 \times 1.98 \times 10^{9}) = 256$ FLOPs/cycle/SM = 128 lane × 2 FLOPs/FMA，与 §2 的 128 个 FP32 单元逐字对上。

逐精度反推表（H100 SXM5，132 SM；峰值取 whitepaper V1.04 Table 3 / datasheet dense 值，clock 按精度类取对应 boost 栏，[NVIDIA, Hopper whitepaper](../references.md) + [NVIDIA, datasheet](../references.md)，[C]，2026-11-25 核实）：

| 精度类 | dense 峰值 | boost clock | $c_{\mathrm{prec}}$（FLOPs/cycle/SM，反推） | 物理依据 |
| --- | --- | --- | --- | --- |
| FP32（CUDA Core） | 66.9 TFLOPS | 1980 MHz | 256 | 128 FP32 lane × 2（FMA） |
| FP64（CUDA Core） | 33.5 TFLOPS | 1980 MHz | 128 | 64 FP64 单元 × 2 |
| FP64 Tensor Core | 66.9 TFLOPS | 1980 MHz | 256 | FP64 TC = 2× FP64 CUDA Core |
| TF32 Tensor Core | 494.7 TFLOPS | 1830 MHz | 2048 | 4 TC × 512 |
| BF16 / FP16 Tensor Core | 989.4 TFLOPS | 1830 MHz | 4096 | 4 TC × 1024 |
| FP8 Tensor Core | 1978.9 TFLOPS | 1830 MHz | 8192 | 4 TC × 2048 |
| INT8 Tensor Core | 1979 TOPS | 1830 MHz | 8192（OPS） | 同 FP8 几何 |

读这张表的方式：精度每降一档， $c_{\mathrm{prec}}$ 翻倍（TF32 → BF16 → FP8 是 2048 → 4096 → 8192），因为同样的乘加阵列在更窄的操作数上每周期能塞进更多 lane；而 FP32 与 FP64 走 CUDA Core 而非 Tensor Core， $c$ 只有 256 / 128——这就是 §1 那个「15 倍差」的来源： $4096/256 = 16$ 倍的每周期吞吐差，再叠加 clock 差 $1.83/1.98 = 0.924$，合 $16 \times 0.924 = 14.8$ 倍，与 $989.4/66.9 = 14.8$ 逐字对上。datasheet 印 989.5 而 whitepaper 印 989.4，是舍入漂移： $132 \times 4096 \times 1.83 \times 10^{9} = 989.46$ TFLOPS，两份官方文档各取一位小数（§6 版本漂移注）。

反推的认识论地位须分清：表里 $n_{\mathrm{SM}}$、 $F_{\mathrm{peak}}$、 $f_{\mathrm{prec}}$ 三列是官方**披露值**（[C] 一手权威）， $c_{\mathrm{prec}}$ 列是**反推值**——官方没有逐字印「每 SM 每周期 4096 BF16 FLOPs」，这个数是用式 (2) 从三个披露值解出来的。反推之所以可信，靠的是闭环自洽：解出的 $c_{\mathrm{FP32}} = 256$ 恰好等于 §2 披露的 128 个 FP32 lane × 2 FLOPs/FMA，两条独立路径（峰值反推 vs 单元计数）给出同一个数； $c_{\mathrm{BF16}} = 4096$ 按 4 个 Tensor Core 分摊得每 TC 1024，再与 mma.sync m16n8k16 的 4096 FLOPs 几何折成 4 周期，量级与指令级口径相容（实测核对归 1.3）。若反推值与披露的单元数对不上，说明口径取错了（最常见是 clock 混精度类，§3 坑注）。这就是「从规格反推微架构参数」的方法：**披露值定标尺，反推值填微架构，单元计数做交叉验证**。

> **坑**：用 FP32-boost clock 算 Tensor Core 峰值。触发条件：把 Table 3 的 1980 MHz 当成「H100 的频率」统一代入所有精度；现象：BF16 算成 $132 \times 4096 \times 1.98 \times 10^{9} = 1070$ TFLOPS，较官方 989.5 高估 8.2%，整条 roofline 与 MFU 账同步虚高；规避：Tensor Core 精度类（FP8/FP16/BF16/TF32）一律取 1830 MHz，FP32/FP64 类取 1980 MHz——频率是**逐精度类**披露的，不是芯片单一常数。

harvest 与 clock 双降的实例（H100 PCIe，对比口径）：Table 3 披露 PCIe 114 SM、TC-boost 1620 MHz，代入式 (1) 得 BF16 dense $114 \times 4096 \times 1.62 \times 10^{9} = 756.4$ TFLOPS（反推，PCIe 峰值官方表值同口径）。同代际同精度下 $c_{\mathrm{prec}}$ 不变，于是有比例律：

**命题 2**（同代际峰值比例律）。同代际、同精度类的两个 SKU，峰值算力之比等于 SM 数之比乘 boost clock 之比： $F_{\mathrm{peak}}^{(a)}/F_{\mathrm{peak}}^{(b)} = (n_{\mathrm{SM}}^{(a)}/n_{\mathrm{SM}}^{(b)}) \cdot (f^{(a)}/f^{(b)})$。**证明**（要点）：式 (1) 两式相除， $c_{\mathrm{prec}}$ 作为同代际同精度的公共因子对消；完整量纲核对与 PCIe 代入留作练习 2。代入 PCIe/SXM5： $(114/132) \times (1620/1830) = 0.8636 \times 0.8852 = 0.7645$，即 PCIe BF16 峰值是 SXM5 的 76.5%——harvest 砍掉 13.6% 的 SM，clock 再砍 11.5%，两刀相乘剩 76.5%。

## 4. 世代对比：Hopper→Blackwell 动了哪几个因子

H200 与 H100 的 BF16 峰值同为 989.5 TFLOPS，为什么 H200 的 decode 更快？因为世代升级动的因子不同：H200 动的是带宽（式 (1) 之外的分母），Blackwell 动的是 $c_{\mathrm{prec}}$ 与 $n_{\mathrm{SM}}$（分子）。本节把四个 SKU 按因子并排。

世代对比表（dense 口径；H100/H200 取 SXM 列，B200 取 HGX 列，GB200 为每 GPU 值；[NVIDIA, datasheet](../references.md) + [NVIDIA, Hopper whitepaper](../references.md) + [NVIDIA, Blackwell brief](../references.md)，[C]，2026-11-25 核实）：

| 因子 | H100 SXM5 | H200 SXM | HGX B200 | GB200（每 GPU） |
| --- | --- | --- | --- | --- |
| 架构 / 工艺 | Hopper / TSMC 4N | Hopper / TSMC 4N | Blackwell / TSMC 4NP | Blackwell / TSMC 4NP |
| die 组织 | 单 die 814 mm² / 80B 晶体管 | 单 die（同 GH100） | 双 reticle-limit die / 208B 晶体管 | 双 die（同 B200） |
| $n_{\mathrm{SM}}$ | 132 | 132 | 148（反推，官方未披露 ⟨verify W3/W6⟩） | 148（同左，反推） |
| BF16 dense | 989.5 TFLOPS | 989.5 TFLOPS | 2250 TFLOPS | 2500 TFLOPS |
| FP8 dense | 1979 TFLOPS | 1979 TFLOPS | 4500 TFLOPS | 5000 TFLOPS |
| FP4 dense | 无硬件支持 | 无硬件支持 | 9000 TFLOPS | 10000 TFLOPS |
| FP32 | 67 TFLOPS | 67 TFLOPS | 75 TFLOPS | 80 TFLOPS |
| HBM 代际 / 容量 | HBM3 / 80 GB | HBM3e / 141 GB | HBM3e / 180 GB | HBM3e / 186 GB |
| $\mathrm{BW}$ | 3.35 TB/s | 4.8 TB/s | 7.7 TB/s | 8 TB/s |
| NVLink（双向口径） | NVLink4 900 GB/s | NVLink4 900 GB/s | NVLink5 1.8 TB/s | NVLink5 1.8 TB/s |
| TDP | ≤700 W configurable | ≤700 W（NVL ≤600 W） | ≤1000 W | ≤1200 W |
| Tensor Core 代际 | 第 4 代 | 第 4 代 | 第 5 代（FP4/FP6 + OCP MX） | 第 5 代 |

**SKU 口径注**（违反即张冠李戴，沿 inference 1.3 同表纪律）：B200 列指 **HGX B200**（180 GB / 7.7 TB/s），与 **GB200**（Grace 配对、186 GB / 8 TB/s、频率更高、BF16 dense 2500 TFLOPS）是不同 SKU，勿混算；Blackwell brief 的「Up to 192 GB」是 **raw 栈容量**（8 栈 × 24 GB），非出货 SKU usable——HGX B200 usable 180 GB、GB200 186 GB，三个数三个口径（raw vs usable 的实测注归 2.1）。H100 NVL（94 GB / 3.9 TB/s / 350–400 W）与 H200 NVL（≤600 W）是各自变体，勿与 SXM 列混。NVLink 带宽一律注方向口径：H100 官方 900 GB/s 为**双向**（inference 1.3 datasheet 同表），单向 450 GB/s；逐 link 几何（whitepaper Fig 6 披露 18 links）与代际精确值归 7.1。Hopper 无 FP4 硬件支持，FP4 行只对 Blackwell 起成立。

逐因子读表。**H200 对 H100 是纯带宽升级**：同 GH100 die、同算力（132 SM 口径沿 H100 SXM5，逐精度峰值与 H100 相同）、同 $c_{\mathrm{prec}}$，式 (1) 三个因子一个没动，动的是 HBM——5 栈 80 GB / 3.35 TB/s 换成 6 栈 141 GB / 4.8 TB/s（栈几何逐项推导归 2.1）。所以 H200 的 compute-bound 上限不变、memory-bound 上限随带宽线性抬升；roofline 分子不变分母变的账归 2.4（recap models 2.4 §6），本章不重推。**Blackwell 对 Hopper 是分子升级**：第 5 代 Tensor Core 把每 SM 每周期 BF16 吞吐从 4096 翻到 8192（反推，下段核对），新增 FP4/FP6 与 OCP microscaling 格式支持（[NVIDIA, Blackwell brief](../references.md)，[C]），HBM3e 8 栈把带宽推到 7.7–8 TB/s，NVLink5 把双向带宽从 900 GB/s 推到 1.8 TB/s。

B200 SM 数的反推自洽性（**官方未披露 SM 数与 clock**，本段是 spec 口径反推、非官方事实，⟨verify W3/W6⟩ 留实测销项）。HGX B200 datasheet 披露 FP32 75 TFLOPS、BF16 dense 2250 TFLOPS；CC 10.0 每 SM 仍 128 FP32 lane（[NVIDIA, CUDA Programming Guide](../references.md) Table 33/34 + [NVIDIA, Blackwell tuning guide](../references.md)，[C]）。设 $n_{\mathrm{SM}} = 148$：FP32 反推 clock $= 75 \times 10^{12}/(148 \times 256) = 1.98$ GHz；BF16 反推 $c = 8192$ 时 clock $= 2250 \times 10^{12}/(148 \times 8192) = 1.856$ GHz。两精度类的 clock 比 $1.98/1.856 = 1.067$，与 Hopper 的 $1980/1830 = 1.082$ 同模式（FP32 类 clock 高于 TC 类）； $c_{\mathrm{BF16}} = 8192 = 2 \times 4096$ 恰为第 5 代 TC 翻倍。FP4 同法反推 $c = 9000 \times 10^{12}/(148 \times 1.856 \times 10^{9}) = 32768$ FLOPs/cycle/SM $= 4 \times c_{\mathrm{BF16}}$，与「FP4 = 2× FP8 = 4× BF16」的代际格式账一致（格式位布局归 6.1）。但这组数**不唯一**：式 (1) 一个方程含 $n_{\mathrm{SM}}$、 $c$、 $f$ 三个未知数，148 只是与官方峰值自洽的一组解——若取 208 SM，BF16 clock 须降到约 1.40 GHz 才凑出 2250 TFLOPS，不是典型 boost 值。故 148 标「反推自洽、官方未披露」，W3 以 sm_100 deviceQuery / CUTLASS、W6 以 nvidia-smi / MIG 实测销项。

能效账（每瓦 BF16 dense，对比口径）：H100 SXM5 $989.5/700 = 1.41$ TFLOPS/W，HGX B200 $2250/1000 = 2.25$ TFLOPS/W，GB200 $2500/1200 = 2.08$ TFLOPS/W。Blackwell 每瓦算力较 Hopper 提升约 59%（2.25/1.41），但 GB200 的每瓦算力低于 HGX B200（2.08 对 2.25）——更高的 clock 与 TDP 换来的峰值增量，边际能效是递减的。这笔账只比峰值/TDP，不含实测利用率（测量归 2.4 / 11.2）。

## 5. die/chiplet/CoWoS：为什么 B200 是双 die

单 die 能无限做大吗？不能——光刻的 reticle（曝光场）尺寸给单 die 面积设了物理上限。Blackwell brief 逐字：208B 晶体管、TSMC 4NP、「two GPU dies … largest die possible within … reticle size … connected and unified with a single 10 terabyte-per-second (TB/s) chip-to-chip NVIDIA High-Bandwidth Interface (NV-HBI), providing one fully coherent chip」（[NVIDIA, Blackwell brief](../references.md)，[C]）。所以 B200 的 chiplet 方案是：两个各做到 reticle 上限的 die，经 NV-HBI 10 TB/s 拼成**逻辑上单一致**的芯片。

封装剖面（CoWoS 2.5D，H100 SXM5 示意重绘；die 面积/工艺/栈数取 whitepaper Table 3，[C]）：

```
CoWoS（Chip-on-Wafer-on-Substrate）2.5D 封装剖面
     ┌──────────┐ ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
     │ GPU die  │ │HBM3 ││HBM3 ││HBM3 ││HBM3 ││HBM3 │ ← 5 栈
     │ 814 mm²  │ │栈 0 ││栈 1 ││栈 2 ││栈 3 ││栈 4 │   每栈 1024-bit
     │ TSMC 4N  │ └─────┘└─────┘└─────┘└─────┘└─────┘
     └──────────┘
  ══════════════ 硅中介层（silicon interposer）══════════════
     5120-bit 内存接口走线（毫米级）+ NVLink / PCIe IO
  ────────────────── 封装基板（substrate）──────────────────

B200 双 die（Blackwell brief 重绘）
  ┌───────────┐    NV-HBI 10 TB/s    ┌───────────┐
  │  die 0    │◄════════════════════►│  die 1    │ ← 两 reticle-limit die
  │  (4NP)    │    单一致芯片         │  (4NP)    │   合计 208B 晶体管
  └───────────┘                      └───────────┘
  + 8 HBM3e 栈（raw 8×24 GB = 192 GB；HGX usable 180 GB）
```

所谓 CoWoS（Chip-on-Wafer-on-Substrate），是指把 GPU die 与 HBM 栈先键合到硅中介层（silicon interposer）、再整体装到封装基板上的 2.5D 封装。它解决的机制问题是**宽总线的物理走线**：H100 的内存接口宽 5120 bit（= 5 栈 × 1024 bit/栈，Table 3），若走封装基板的毫米级走线，5120 根线的扇出与信号完整性不可行；硅中介层的布线密度高一个量级，把 GPU die 到 HBM 栈的距离压到毫米级，才撑得起 3.35 TB/s（逐项带宽推导归 2.1）。H100 SXM5 用 5 栈 / 10 个 512-bit 内存控制器，full GH100 支持 6 栈 / 12 控制器——H200 正是把第 6 栈补齐到 141 GB / 4.8 TB/s（§4）。

NV-HBI 与 HBM 的带宽对比（口径注：brief 披露 NV-HBI 10 TB/s 未注方向口径，本书按原值引、不做方向推定）：10 TB/s 的 die-to-die 互连高于 HGX B200 的 HBM 7.7 TB/s，两者同量级。设计意图是让跨 die 访问不先于本地 HBM 成为带宽瓶颈——若 NV-HBI 显著低于 HBM，双 die 就会退化成「两颗经窄桥相连的 GPU」而非单一致芯片。跨 die 延迟与一致性的实测归 2.2 / 7.1，本章只引披露的带宽数。

> **坑**：把 NV-HBI 与 NVLink-C2C 当同一条互连。触发条件：见「B200 双 die 经 NVLink-C2C 连接」的转述；现象：die-to-die 与 CPU-to-GPU 两条互连混为一谈，带宽账与拓扑账全错；规避：**NV-HBI**（10 TB/s）是 B200 封装内两个 GPU die 之间的互连（brief 逐字），**NVLink-C2C** 是 GB200 superchip 里 Grace CPU 与 Blackwell GPU 之间的互连——GB200 superchip = 1 Grace CPU + 2 Blackwell GPU（[NVIDIA, Blackwell brief](../references.md) + [NVIDIA, datasheet](../references.md)，[C]），CPU↔GPU 走 NVLink-C2C、GPU 内部双 die 走 NV-HBI、GPU↔GPU 走 NVLink5，三条互连三个层级。NVLink-C2C 的逐 link 带宽与 NVLink 域拓扑归 7.1，服务器形态归 11.1。

chiplet 的账为什么停在双 die 而不是四 die？本书只引已披露信息：brief 披露的是双 die + NV-HBI 单一致芯片，更多 die 的互连带宽与一致性开销属闭源设计细节，不做推测。可判定的一笔是：双 die 方案让 208B 晶体管在 reticle 上限内落地，而一致性上如同一颗芯片（brief 逐字「one fully coherent chip」）；die 边界对 block 调度是否可见属闭源细节，本书不推定——block 整体驻留一个 SM 的语义归 3.1，跨 die 访存延迟归 2.2。

## 6. 官方规格表读法：四个口径陷阱与版本漂移

同一颗 H100，为什么 blog 写 3 TB/s、whitepaper 写 3352 GB/s、datasheet 写 3.35 TB/s？三个数不是矛盾，是版本与口径。读官方规格表须先锁定四件事：dense 还是 sparse、哪个精度类的 clock、哪个 SKU 列、raw 还是 usable。

| 陷阱 | 触发条件 | 后果 | 规避 |
| --- | --- | --- | --- |
| sparse 当 dense | datasheet 表值带 * 号未读脚注 | 峰值高估 2× | 脚注逐字「Specifications in sparse. Dense is one-half of the sparse spec shown.」——dense = 表值之半，本书一律 dense |
| clock 混精度类 | 用 1980 MHz 算 BF16 | 高估 8.2%（§3 坑注） | TC 类 1830 / FP32-FP64 类 1980，逐精度类取栏 |
| SKU 混列 | HGX B200 与 GB200 与 raw 192 GB 混算 | 容量差 6–12 GB、带宽差 0.3 TB/s | §4 SKU 口径注；对账以 inference 1.3 同表（HGX B200）为准 |
| raw 当 usable | brief「Up to 192 GB」当出货容量 | 容量高估 12 GB（192 对 180） | raw = 8×24 GB 栈容量；usable 取 datasheet SKU 列（2.1 实测注） |

版本漂移注（[NVIDIA, Hopper blog](../references.md) vs [NVIDIA, Hopper whitepaper](../references.md)，[C]，2026-11-25 核实）：Hopper blog（早期披露）写「over 3 TB/sec」且注明「Memory Data Rate Not finalized」；whitepaper V1.04 Table 3 是 final 值 3352 GB/s（封面逐字「Includes final GPU / memory clocks and final TFLOPS performance specs. V1.04」）；datasheet 印 3.35 TB/s。三者是早期→final 的漂移，取 final。whitepaper 内部也有一处未更新：HBM prose 仍留「80 GB (five stacks) … over 3 TB/sec」，与同文档 Table 3 的 3352 GB/s 不一致——**以 Table 3 final 值为准**，prose 是残留。BF16 峰值 989.4（whitepaper）对 989.5（datasheet）是舍入漂移（真值 989.46，§3），不是两个规格。Programming Guide 在 CUDA 13.x 拆子页，compute capability 表移到 05-appendices/compute-capabilities.html（Table 33/34/35），旧单页锚失效——引用须用子页 URL（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）。

读表纪律一句话：**官方规格披露是一手权威（[C] 官方规格披露·一手权威），但「官方」内部有版本与口径分层**——取 final 版、取 dense、取对应精度类 clock、取出货 SKU 列；媒体转述只作线索，不作事实来源（spec §2 闭源与官方规格处理条款）。本章全部数值按此纪律取自 2026-11-25 一手核实的 dam-cdn 官方 PDF。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 峰值算力 = SM 数 × 每 SM 每周期 FLOPs × 该精度类 boost clock（结构恒等式，量纲对账证明） | 式 (1)，命题 1 | §3、练习 1 |
| 每 SM 每周期吞吐由官方峰值反推：H100 BF16 4096 / FP32 256 / FP8 8192 FLOPs/cycle/SM | 式 (2) + 逐精度表 | §3、练习 1 |
| 频率依精度类而异：TC 类 1830 MHz、FP32/FP64 类 1980 MHz；混用高估 8.2% | §3 坑注 | §3、§6、练习 1 |
| GH100 full 144 SM（8 GPC×9 TPC×2 SM）；H100 SXM5 harvest 到 132 SM / 66 TPC，PCIe 114 SM | §2 框图 + Table 3 | §2、练习 2 |
| 同代际同精度峰值比 = SM 数比 × clock 比；PCIe/SXM5 = 0.765（756 对 989.5 TFLOPS） | 命题 2 | §3、练习 2 |
| H200 对 H100 纯带宽升级（式 (1) 三因子不动）；Blackwell 动 $c$（BF16 8192 = 2×4096）与格式（FP4） | §4 对比表 | §4、练习 3 |
| B200 SM 数 148 为反推自洽、官方未披露（一方程三未知数，不唯一）⟨verify W3/W6⟩ | §4 反推段 | §4、练习 4 |
| SKU 三口径勿混：HGX B200 180 GB/7.7 vs GB200 186 GB/8 vs raw 192 GB；NVLink 注方向口径 | §4 SKU 口径注 | §4、§6、练习 3 |
| B200 双 reticle-limit die 经 NV-HBI 10 TB/s 成单一致芯片；NV-HBI（die↔die）≠ NVLink-C2C（CPU↔GPU） | §5 框图 + 坑注 | §5 |
| CoWoS 硅中介层撑 5120-bit 宽总线毫米级走线；H100 5 栈/10 控制器，H200 补第 6 栈 | §5 剖面图 | §5、2.1 |
| 规格表读法四陷阱：sparse/dense、clock 精度类、SKU 列、raw/usable；版本漂移取 final | §6 表 | §6 |

## 练习

1.（计算 · [10]）H100 SXM5 主口径：(a) 用式 (1) 与 §2 单元数表算 FP32 峰值（132 SM × 128 lane × 2 FLOPs/FMA × 1.98 GHz），对照 datasheet 67 TFLOPS；(b) 若误用 1980 MHz 算 BF16 dense（ $c = 4096$），得多少 TFLOPS、较官方 989.5 高估百分之几？校验：(a) 与 §3 逐精度表 FP32 行 66.9 TFLOPS 一致（容差 0.2%）；(b) 与 §3 坑注的 8.2% 一致。
2.（推导 · [20]）(a) 补全命题 2 的证明：由式 (1) 写出两 SKU 峰值之比，说明 $c_{\mathrm{prec}}$ 对消的条件（同代际同精度类），并做量纲核对；(b) 代入 H100 PCIe（114 SM / 1620 MHz）与 SXM5（132 SM / 1830 MHz）算比例，再反推 PCIe BF16 dense 峰值；(c) 若 harvest 只砍 SM 不降 clock（114 SM / 1830 MHz），峰值是 SXM5 的百分之几？先预测 (c) 与 (b) 哪个更高再算。校验：(b) 与 §3 的 0.7645 / 756.4 TFLOPS 一致（容差 0.1%）；(c) 应等于 114/132 = 86.4%（clock 因子为 1 的特例）。
3.（计算 · [20]）SKU 与能效账（§4 对比表）：(a) 算 HGX B200 与 GB200 的容量差、带宽差、BF16 dense 差各百分之几；(b) 算 H100 SXM5 / HGX B200 / GB200 的每瓦 BF16 dense（TFLOPS/W），排出次序；(c) 解释为什么 GB200 峰值更高但每瓦算力低于 HGX B200。校验：(a)(b) 与 §4 能效账段逐数一致（1.41 / 2.25 / 2.08 TFLOPS/W）；(b) 次序应为 HGX B200 > GB200 > H100。
4.（计算 · [30]）B200 SM 数反推自洽性（§4 反推段，官方未披露项）：(a) 设 $n_{\mathrm{SM}} = 148$，由 HGX B200 FP32 75 TFLOPS 与 CC 10.0 每 SM 256 FP32 FLOPs/cycle 反推 clock；(b) 由 BF16 dense 2250 TFLOPS 与 $c = 8192$ 反推 TC-boost clock，核对两 clock 比是否与 Hopper 1980/1830 同模式；(c) 由 FP4 dense 9000 TFLOPS 反推 $c_{\mathrm{FP4}}$，验证 = 4× BF16；(d) 若改设 $n_{\mathrm{SM}} = 208$，BF16 clock 须为多少才凑出 2250 TFLOPS？据此说明为什么 148 仍只是「反推自洽」而非官方事实、须留 ⟨verify W3/W6⟩。校验：(a) ≈1.98 GHz、(b) ≈1.856 GHz、(c) 32768 FLOPs/cycle/SM 与 §4 逐数一致；(d) ≈1.40 GHz（不典型 boost 值）；式 (1) 一个方程含三个未知数，解不唯一。

## 参考

- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25 dam-cdn PDF 一手核实——Table 3（132/114 SM、66/57 TPC、128 FP32/64 FP64/64 INT32/4 TC per SM、逐精度 boost clock 1830/1980 MHz、BF16 dense 989.4、5120-bit HBM3 5 栈 3352 GB/s、L2 50 MB、814 mm²/80B/TSMC 4N）、Table 4（CC 9.0）、Fig 6（GH100 full 144 SM + L2 两分区 + 6 栈/12 控制器 + 18 NVLink links）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（含 GB300 更新版，key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——208B 晶体管/TSMC 4NP、双 reticle-limit die + NV-HBI 10 TB/s 单一致芯片、第 5 代 Tensor Core FP4/FP6 + OCP MX、18 NVLink5 links 1.8 TB/s、B200 raw Up to 192 GB / 7.7 TB/s / 1000 W、GB200 8 TB/s；B200 SM 数未披露（⟨verify W3/W6⟩）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100/H200/HGX H100-H200/Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25 四份 PDF 一手核实——dense = sparse 表值之半脚注、H100 SXM 989.5/3.35 TB/s/≤700 W、H200 SXM 141 GB/4.8 TB/s（NVL ≤600 W）、HGX B200 2250/4500/9000 dense/180 GB/7.7 TB/s/≤1000 W、GB200 2500/186 GB/8 TB/s/≤1200 W、H100 NVL 94 GB/3.9 TB/s 勿混
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——compute-capabilities 子页 Table 33/34/35（CC 9.0/10.0 资源库存）、KB=1024 逐字注；CUDA 13.x 拆子页 URL 漂移注
- [NVIDIA, Hopper tuning guide](../references.md)：NVIDIA Hopper Tuning Guide（key: hopper-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——thread block cluster / DSMEM / TMA（§2 提点）
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——CC 10.0 每 SM 128 FP32 lane 与资源库存（§4 反推依据）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——mma.sync m16n8k16 几何（§3 指令口径，详拆归 1.3）
- [NVIDIA, Hopper blog](../references.md)：NVIDIA Hopper Architecture In-Depth blog（key: hopper-blog），[C] 开放技术博客，2026-11-25——早期披露 3 TB/s + Memory Data Rate Not finalized（§6 版本漂移注，whitepaper 互证源）
