# 2.4 内存带宽账与 roofline（Bandwidth Accounting and the Roofline）

> ridge ≈295 FLOP/byte 的两个数——989.5 TFLOPS 与 3.35 TB/s——各从哪条硬件乘法链来？实测点落在 roofline 下方，差的那一段是什么损失？本章 recap models 2.4 §6 的强度/roofline 入口（式 (10)-(12)，翻译符号、不重推），交付三块增量：层级带宽-延迟-容量全表（register→shared/L1/L2→HBM→NVLink/PCIe，per-SM 归一化账）、HBM 带宽与峰值算力的微架构来源对账（recap 2.1 栈几何 + 1.1/1.3 峰值反推，对 inference 1.3 datasheet 同表）、测量方法（nvbandwidth / Nsight Compute，theoretical vs achieved 利用率）与 bound 判定精确化（compute / memory / latency-bound 三区，命题 1）+ 实测 roofline 绘制（命题 2） | 前置：models 2.4 §6（式 (10)-(12) + 入口 prose，recap 源）、inference 1.3（datasheet 参数表与 ridge 表）、2.1（HBM 式 (1) 栈几何）、2.2（延迟表与 sector 口径）、1.2（register/shared 带宽账与 160:10:1 比值）、1.4（Little's law 式 (1)-(5) 与 208 sector 并发）；下游 3.4（occupancy → latency-bound 判定输入）、4.3（tiling 提强度、点上移）、11.2（profiling 全流程与实测 roofline drill-down）、7.2（P2P 带宽矩阵测量） | 锚点：NVIDIA datasheet（[NVIDIA, datasheet](../references.md)，[C] 官方规格披露·一手权威）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、nvbandwidth README（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现，2026-11-25 核实）、Nsight Compute ProfilingGuide §2.9 roofline / SpeedOfLight（[NVIDIA, Nsight Compute](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）、NVLink/PCIe 规格（[NVIDIA, NVLink/NVSwitch](../references.md) / [PCI-SIG, PCIe](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 295 这个数从哪来：三个问题与本章口径

H100 SXM 的 BF16 dense 峰值 989.5 TFLOPS、HBM 带宽 3.35 TB/s，两者相除得 ridge ≈295 FLOP/byte——decode 的强度（1–5 量级）离它差约两个数量级，prefill 的强度（上千）远在它右侧。这个分界数是 models 2.4 §6 交付的入口、inference 1.3 接上 datasheet 的参数；但两个问题还没回答：**295 的分子分母各从哪条硬件乘法链来**（不是营销取整，是 SM 数 × 每周期吞吐 × clock 与栈数 × 通道 × pin rate × 位宽）？**实测 kernel 为什么常常连 roofline 上限都跑不到**——差的那一段是带宽利用率损失、算力利用率损失，还是并发不足的 latency-bound？本章回答三个问题：层级各层的带宽-延迟-容量怎么归一化对比？bound 判定怎么从两区精确化为三区？achieved 带宽与利用率怎么测、实测点怎么画？

**本章口径**（章脊锁定；本章为多 SKU 对比章，多 SKU 口径在此一次声明，表外不再换）：

| 口径 | SKU 与参数（dense） | 来源 |
| --- | --- | --- |
| 主口径 | H100 SXM：BF16 989.5 TFLOPS / 3.35 TB/s / 132 SM / 80 GB | [NVIDIA, datasheet](../references.md) + [NVIDIA, Hopper whitepaper](../references.md)，[C] |
| 对比口径 | H200 SXM：989.5 TFLOPS / 4.8 TB/s / 141 GB；HGX B200：2250 TFLOPS / 7.7 TB/s / 180 GB | inference 1.3 §2 同表（[NVIDIA, datasheet](../references.md)，[C]） |
| 对照注 | GB200 每 GPU：2500 TFLOPS / 8 TB/s / 186 GB——仅对照注，不入表混算 | [NVIDIA, datasheet](../references.md)，[C]；SKU 纪律沿 inference 1.3 |

精度口径：roofline 主精度 **BF16 dense**（serving GEMM 不依赖结构化稀疏，dense 才是可达算力口径，沿 inference 1.3），FP8/FP4 作对比行；官方 sparse 表值之半为 dense（脚注逐字 recap inference 1.3）。单位口径：显存 GB = $10^{9}$ B、带宽 TB/s = $10^{12}$ B/s（datasheet 十进制）；片上存储 KB = 1024 B（Programming Guide 逐字注，recap 1.2）；L2 50 MB 沿 whitepaper 印法（2.2）。**实测口径声明**（沿 1.4/2.2）：精确延迟 cycles（shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8）来自 H800 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B]），非 H100 官方规格；周期 ↔ 时间换算取 TC-boost 1830 MHz，register/shared 带宽账沿用 1.2 的 FP32-boost 1980 MHz（逐精度类 clock，1.1 §3）。本章「实测点」数值凡未引官方披露/已发表基准者，一律标**教学假设口径**——本机无 NVIDIA GPU，测量项交付的是方法 + 官方披露口径（spec §6.2 裁定），不是新跑 benchmark。

符号声明（spec §4.3）： $I$ = 算术强度（FLOP/byte）； $R$ = ridge point； $F_{\mathrm{peak}}$ = 峰值算力（FLOP/s）； $\mathrm{BW}$ = 带宽（byte/s）。**翻译注**（recap 硬规则）：inference 1.3 的 $\beta$（带宽）在本章译作 $\mathrm{BW}$—— $\beta$ 在 gpu 通信章（8.x）专归 α-β 模型的每字节传输时间，不挪用；models 2.4 的 $d$（hidden 宽）译 $d_{\mathrm{model}}$、 $b$（每元素字节）译 $\mathrm{bytes}$（不占字母）、 $t$（已缓存长度）译 $T$（沿 inference 1.3 翻译先例）。 $B$ = batch、 $T$ = 序列长、 $N$ = 参数量、 $L$ = 层数、 $H_{kv}$ = KV 头数、 $d_{\mathrm{head}}$ = head 维（recap models/inference 沿用）。 $n_{\mathrm{SM}}$ = SM 数； $C$ = 提供的在飞并发（sector 口径）； $L_{\mathrm{req}}$ = 所需并发（1.4 口径）； $u$ = 利用率（ $u_B$ 带宽、 $u_F$ 算力）； $F_{\mathrm{ach}}$、$`\mathrm{BW}_{\mathrm{ach}}`$、 $I_{\mathrm{ach}}$ = 实测算力/带宽/强度。混场声明： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread（3.4 recap 出现时同章区分）。

**边界声明**： $I$/吞吐上限/ridge 的推导归 models 2.4 §6（本章 recap + 翻译，**不重推**）；datasheet 参数表归 inference 1.3（本章引同表对账，增量 = 微架构来源与测量——inference W1 errata 裁定的分工）；batch 三方权衡/排队论/SLO 操作点归 inference 1.3；profiling 全流程（Nsight Systems timeline → kernel drill-down）归 11.2；NVLink 拓扑与 P2P transport 归 7.1/7.3；coalescing/tiling/fusion 的 kernel 手段归 4.1/4.3/5.4。代码 scope：机制片段级（§6 测量命令片段）。

## 2. models 2.4 §6 recap：强度与 roofline 入口（不重推）

先回答一个更基本的问题：roofline 为什么是「min of 两条线」？因为 kernel 的吞吐同时被两个独立资源封顶——算力单元每周期能算的 FLOPs 有上限，内存系统每秒能搬的字节有上限；每 byte 对应 $I$ 个 FLOP，所以带宽侧折算成算力就是 $I \times \mathrm{BW}$，两者取小。这三条关系是 models 2.4 §6 的入口 prose（无编号）：算术强度 $I$ = FLOPs ÷ 访存字节数；吞吐上限 = $\min(F_{\mathrm{peak}},\ I \times \mathrm{BW})$；分界 ridge point = 峰值算力 ÷ 带宽。本节只 recap 与翻译，推导见 models 2.4 §6，本章不复述。

两阶段强度——以下三式依次为 models 式 (10)、式 (11)、式 (12) 的符号翻译（按 §1 翻译注译写；**已交付公式，不重推**，推导见 models 2.4 §6）：

$$
I_{\mathrm{prefill}} \approx T_p
$$

$$
I_{\mathrm{decode}}(T) = \frac{2N + 4 L T d_{\mathrm{model}}}{2N + 2 L H_{kv} d_{\mathrm{head}} T \cdot \mathrm{bytes}}
$$

$$
I_{\mathrm{decode}}(T, B) = \frac{B \left(2N + 4 L T d_{\mathrm{model}}\right)}{2N + B \cdot 2 L H_{kv} d_{\mathrm{head}} T \cdot \mathrm{bytes}}
$$

recap 结论（数值均为 models 2.4 §6 已交付口径，锚点家族 Llama-3.1-8B）：

- **prefill**： $T_p$ 个 token 一次前向，权重只读一遍， $I \approx T_p$；prompt 上千 token 时 $I$ 在 $10^{3}$ 量级，远在 ridge 右侧——compute-bound。
- **decode**：短上下文端 $T \to 0$ 时 $I \to 1$ FLOP/byte；长上下文端 8B 满窗（ $T$ = 131072） $I \approx 2.5$——cache 把强度从 1 抬到 2.5，离 ridge 仍差近两个数量级，**memory-bound 在整个窗口内成立**。
- **batch 的作用与边界**： $B$ 小时权重项主导分母， $I \approx B$（「凑 batch」的适用区）； $B$ 大到 cache 项主导后强度饱和于 $(2N + 4 L T d_{\mathrm{model}})/(2 L H_{kv} d_{\mathrm{head}} T \cdot \mathrm{bytes})$，8B 满窗上界约 4.9——长上下文里 batch 对强度的提升被 cache 读取封顶。

这三条结论给的是**上限的形状**：强度在 ridge 哪一侧、上限取哪条线。但入口没回答三个硬件问题—— $F_{\mathrm{peak}}$ 与 $\mathrm{BW}$ 两个数从哪来（§3）、层级各层的带宽怎么对比（§4）、上限跑不满时差在哪、怎么测（§5–§7）。以下三节是本章增量。

## 3. ridge 的微架构来源：分子分母两条乘法链对账

ridge 的分子分母是营销取整还是硬件乘积？推导路线分三步：先 recap 分母的栈几何乘法链（2.1 式 (1)），再 recap 分子的 SM 乘法链（1.1 式 (2) / 1.3 §3），最后代入 ridge 并与 inference 1.3 datasheet 同表逐行对账。关键一步是**两个数都能从硬件库存乘出来**——ridge 不是拟合参数，是两条独立乘法链的商。

ridge 定义式（models 2.4 §6 prose / inference 1.3 式 (1) recap，本章编号供代入引用）：

$$
R = \frac{F_{\mathrm{peak}}}{\mathrm{BW}} \qquad (1)
$$

**分母来源**（recap 2.1 §3 式 (1)(2)，不重推）：HBM 带宽 = 栈数 × 每栈通道数 × pin rate × 每通道位宽 ÷ 8，H100 SXM5 逐项代入（全部披露值，[NVIDIA, Hopper whitepaper](../references.md) Table 3，[C]）：

$$
\mathrm{BW} = \frac{n_{\mathrm{stack}} \cdot n_{\mathrm{ch}} \cdot f_{\mathrm{pin}} \cdot w_{\mathrm{ch}}}{8} = \frac{5 \times 16 \times 5.238 \times 10^{9} \times 64}{8} = 3352.32 \ \mathrm{GB/s} \qquad (2)
$$

**符号**： $n_{\mathrm{stack}} = 5$（wp 逐字「5 HBM3 stacks」）； $n_{\mathrm{ch}} = 16$、 $w_{\mathrm{ch}} = 64$ bit（Micron 互证）； $f_{\mathrm{pin}} = 5.238$ Gbps（2619 MHz DDR × 2）。对账：whitepaper Table 3 印 3352 GB/sec、datasheet 印 3.35 TB/s——推导闭环。**口径守卫**（recap 2.1 命题 1）：1024 bit/stack 是 $n_{\mathrm{ch}} w_{\mathrm{ch}}$ 的积，代入含 $n_{\mathrm{ch}}$ 因子的式会高估 16 倍（53.6 TB/s 的荒谬值）；本章一律代 64 bit/channel。

**分子来源**（recap 1.1 §3 式 (2) / 1.3 §3，不重推）：BF16 dense 峰值 = SM 数 × 每 SM 每周期 BF16 FLOPs × TC-boost clock：

$$
F_{\mathrm{peak}} = n_{\mathrm{SM}} \cdot c_{\mathrm{BF16}} \cdot f_{\mathrm{TC}} = 132 \times 4096 \times 1.83 \times 10^{9} = 989.43 \ \mathrm{TFLOPS} \qquad (3)
$$

**符号**： $n_{\mathrm{SM}} = 132$； $c_{\mathrm{BF16}} = 4096$ FLOPs/cycle/SM = 4 Tensor Core × 1024（1.3 §3 指令几何闭环：mma.sync m16n8k16 = 4096 FLOPs、每 TC 每周期 1024）； $f_{\mathrm{TC}} = 1830$ MHz（wp 逐字 TC 精度类 boost）。对账：whitepaper 印 989.4、datasheet 印 989.5——**舍入路径漂移**（recap 1.1 §3：dense 真值 989.43，wp 直接舍入得 989.4；datasheet 以 sparse 口径印刷，1978.86 先舍入到 1979 再除 2 得 989.5），不是两个规格。本章 ridge 代入取 datasheet 989.5（与 inference 1.3 同表同口径）。

ridge 代入对账表（式 (1)；**与 inference 1.3 §2 ridge 表逐行一致**，同 datasheet dense 口径；单位对账：TFLOPS ÷ TB/s = $10^{12}$ FLOP/s ÷ $10^{12}$ byte/s = FLOP/byte，量纲恰好消去）：

| ridge（FLOP/byte） | H100 SXM | H200 SXM | HGX B200 |
| --- | --- | --- | --- |
| BF16 | $989.5/3.35 \approx 295$ | $989.5/4.8 \approx 206$ | $2250/7.7 \approx 292$ |
| FP8 | $1979/3.35 \approx 591$ | $1979/4.8 \approx 412$ | $4500/7.7 \approx 584$ |
| FP4 | —（Hopper 无 FP4） | — | $9000/7.7 \approx 1169$ |

这张表的读法沿 inference 1.3（recap，不重推）：H200 对 H100 是纯带宽升级（分子不动、分母 3.35→4.8，ridge 295→206）；HGX B200 算力带宽近乎同比放大（BF16 ×2.27、带宽 ×2.3），ridge 几乎不动（295→292）；精度越低 ridge 越高（FP8 ≈ 2× BF16、FP4 ≈ 4×，分子翻倍分母不动）。**本章增量不是 ridge 数值本身，是分子分母各有一条可核对的微架构乘法链**：3.35 不是任意数，是 5 栈 × 16 通道 × 5.238 Gbps × 64 bit ÷ 8；989.5 不是任意数，是 132 SM × 4096 FLOP/cycle × 1.83 GHz 的舍入。换 SKU 时两条链各自重代（H200/B200 的 $f_{\mathrm{pin}}$ 为反推值，recap 2.1 §3 SKU 表），ridge 随之确定。

**SKU 口径注**（recap 2.1 §6）：HGX B200 180 GB / 7.7 TB/s、GB200 186 GB / 8 TB/s、raw 192 GB（8 × 24 GB 栈容量）是三个数三个口径；本章对账以 inference 1.3 同表（HGX B200）为准，GB200 仅作对照注。decode 强度饱和上界（8B 满窗约 4.9）远低于表中任何 ridge（最低 H200 BF16 约 206）——**decode memory-bound 的结论跨 SKU 成立**，换卡对 decode 的收益几乎全来自带宽项（inference 1.3 结论 recap）。

## 4. 层级带宽-延迟-容量全表：归一化才能比

register 比 HBM 快 160 倍——这个数怎么比出来的？直接拿 register 的 4055 GB/s 对 HBM 的 3352 GB/s 会得到「只快 1.2 倍」的错误结论。关键在于**作用域不同**：register/shared/L1 是每 SM 私有（各 132 份），L2/HBM 是全芯片共享（一份）；比较之前必须归一化到同一口径。本节先给全表，再推归一化账。

层级全表（H100 SXM 主口径；容量官方 [C]，延迟实测 [B]，带宽分披露/反推/未披露三口径）：

| 层 | 作用域 | 容量 | 带宽 | 延迟（实测 [B]） | 出处 |
| --- | --- | --- | --- | --- | --- |
| register file | 每 SM 私有 | 256 KB/SM（芯片 33 MB） | 需求上界 2048 B/cycle/SM ≈ 4055 GB/s/SM @1.98 GHz；芯片聚合 ≈535 TB/s | operand 读出在执行管线内（非 load 延迟口径） | PG Table 34 [C] + 1.2 式 (2) 反推 |
| shared | 每 SM（block 共享） | 228 KB/SM 最大档 | 128 B/cycle/SM = 253 GB/s @1.98 GHz（32 banks × 4 B 无冲突峰值）；芯片聚合 ≈33.4 TB/s | 29.0 cycles | PG Table 34 [C] + Luo [B] |
| L1 | 每 SM 私有 | 256 KB 统一 − shared carveout（28–256 KB） | 官方未披露每周期带宽（与 shared 同一 SRAM；测量见 §6/§7） | 40.7 cycles | PG Table 35 [C] + Luo [B] |
| L2 | 全芯片共享 | 50 MB（两分区各 25 MB） | 官方未披露（Nsight 层级 roofline 给实测 ceiling，§7） | 263.0 cycles | wp Table 3 [C] + Luo [B] |
| HBM | 全芯片共享 | 80 GB | 3352 GB/s（式 (2)）；每 SM 份额 25.4 GB/s | 478.8 cycles（L2 miss） | wp + datasheet [C] + Luo [B] |
| NVLink4（片外） | GPU ↔ GPU | 对端 HBM（HGX 8 卡域 8 × 80 GB） | 900 GB/s 双向 = 450 GB/s 单向（18 links × 25 GB/s/link 单向） | 未披露（nvbandwidth latency 测试，7.2） | [NVIDIA, NVLink/NVSwitch](../references.md) [C] + datasheet |
| PCIe Gen5 x16（片外） | GPU ↔ host | host DRAM（平台口径） | 64 GB/s 单向 / 128 GB/s 双向（理论口径） | 未披露（测量归 7.2/11.2） | [PCI-SIG, PCIe](../references.md) [C] + datasheet 双向印法 |

（NVLink/PCIe 是互连不是存储，容量列写其可达的对端存储；B200 NVLink5 双向 1.8 TB/s 作世代对照注，[NVIDIA, Blackwell brief](../references.md) 逐字「18 fifth-generation NVLink links to provide 1.8 [TB/s]」，不入 H100 主表。延迟列的全程 load-to-use 口径与非层间增量注 recap 2.2 §2。）

归一化账推导路线分三步：先把各层按作用域分成私有/共享两类，再各给一个归一化式，最后代入求比值。关键一步是**两类层的归一化方向相反**——私有层乘 $n_{\mathrm{SM}}$ 得芯片聚合，共享层除 $n_{\mathrm{SM}}$ 得每 SM 份额。

共享层（L2/HBM）的每 SM 份额：

$$
\mathrm{BW}_{\mathrm{SM}} = \frac{\mathrm{BW}}{n_{\mathrm{SM}}} \qquad (4)
$$

私有层（register/shared/L1）的芯片聚合：

$$
\mathrm{BW}_{\mathrm{chip}} = n_{\mathrm{SM}} \cdot \mathrm{BW}_{\mathrm{layer}} \qquad (5)
$$

**符号**：$`\mathrm{BW}_{\mathrm{SM}}`$ = 共享层摊到每 SM 的带宽份额； $\mathrm{BW}$ = 共享层芯片级带宽；$`\mathrm{BW}_{\mathrm{layer}}`$ = 私有层每 SM 带宽；$`\mathrm{BW}_{\mathrm{chip}}`$ = 私有层芯片聚合； $n_{\mathrm{SM}} = 132$。量纲对账：式 (4) (B/s) ÷ (SM) = B/s/SM；式 (5) (SM) × (B/s/SM) = B/s——两式方向相反，混用即差 $n_{\mathrm{SM}}$ 倍。

全数字代入（register/shared 带宽为 1.2 式 (2) 需求上界与 bank 几何口径，@1.98 GHz）：

- HBM 每 SM 份额： $3352/132 = 25.4$ GB/s（式 (4)）。
- register 芯片聚合： $132 \times 4055$ GB/s $= 535$ TB/s（式 (5)）；shared 芯片聚合： $132 \times 253$ GB/s $= 33.4$ TB/s。
- **片上比值**（per-SM 同口径）：register : shared : HBM 份额 = $4055 : 253 : 25.4 \approx 160 : 10 : 1$（recap 1.2 §3，本表把它放进完整层级）。
- **片外比值**（芯片级同口径）：HBM : NVLink 单向 : PCIe 单向 = $3352 : 450 : 64 \approx 52 : 7 : 1$；HBM 对 NVLink 双向 = $3352/900 = 3.72$。

> **坑**：per-SM 口径与芯片口径混比。触发条件：拿 register 的 4055 GB/s/SM 直接对 HBM 芯片级 3352 GB/s；现象：得出「register 只比 HBM 快 1.2 倍」，tiling 收益被低估两个量级；规避：私有层先乘 132（或共享层先除 132）再比——正确比值 160:1（式 (4)(5) 量纲对账）。同理 NVLink 900 GB/s 是双向口径，与 HBM 3352（单向数据带宽）比须先统一到单向 450（spec §4.3 方向口径）。

读表的方式沿 2.2 §4 的容量-延迟权衡：每往下一层容量大一个量级以上（256 KB → 228 KB → 50 MB → 80 GB），延迟逐级抬升（管线内 → 29 → 40.7 → 263 → 478.8 cycles）；片外再往下，NVLink/PCIe 的带宽比 HBM 低 7–52 倍，延迟更高（未披露，量级上 P2P 访问经互连往返）。没有「又快又大」的层，所以 kernel 优化的主线是把工作集沿金字塔往上搬（tiling 搬进 shared/register，4.3；persisting 留在 L2，2.2 §5），通信优化的主线是把流量留在高带宽域内（NVLink 域内 all-reduce，8.x）。L1/L2 的每周期带宽官方未披露，本表不推定——测量方法见 §6，Nsight 层级 roofline 的 L1/L2/DRAM ceiling 见 §7。

带宽层级还决定「喂满各层需要多少在飞数据」——带宽延迟积（recap 1.4 §4，不重推）。HBM 口径：芯片级 $3352 \times 10^{9} \times 261.6 \times 10^{-9} \approx 877$ KB 在飞，每 SM ≈6.6 KB（208 sectors）；shared 口径：服务率 128 B/cycle = 4 sectors/cycle、延迟 29 cycles，所需在飞 $4 \times 29 = 116$ sectors = 29 条 warp load（1.4 练习 2c）。对比这两笔：shared 的服务率是 HBM 每 SM 份额的 9.2 倍（ $4/0.434$），延迟却只有其 $1/16.5$（ $29/478.8$），所以所需在飞并发（116 对 208 sectors）反而略低。这意味着：把工作集搬进 shared 之后，瓶颈从「藏 HBM 延迟」换成「shared bank conflict 与指令发射」（4.2），并发预算要按新层的服务率与延迟重算——本表就是重算的输入，tiling 的收益账（4.3）两笔都要记。

## 5. bound 判定精确化：compute / memory / latency 三区

models 2.4 给的是两区（ $I$ 对 $R$ 的左右），但实测 kernel 还有第三种处境：**既没到算力峰值、也没到带宽峰值**——强度在 ridge 左侧、带宽却跑不满。这一区从哪来？从并发不足来（1.4）：带宽是服务率上限，实际吞吐 = 在飞并发 ÷ 延迟，并发不够时服务率空闲。本节把两区精确化为三区。

recap 1.4（不重推）：Little's law $L = \lambda \cdot t_{\mathrm{lat}}$，藏住延迟所需并发 $L_{\mathrm{req}} = \lambda_{\max} \cdot t_{\mathrm{lat}}$；HBM 口径每 SM 服务率 $\lambda_{\mathrm{SM}} = 25.4 \times 10^{9}/(32 \times 1.83 \times 10^{9}) = 0.434$ sectors/cycle，所需并发：

$$
L_{\mathrm{req}} = \lambda_{\mathrm{SM}} \cdot t_{\mathrm{HBM}} = 0.434 \times 478.8 \approx 208 \ \text{sectors} \qquad (6)
$$

即每 SM 约 208 个 sector（≈6.6 KB）在飞 = 52 条 coalesced warp load（ $s_{\mathrm{warp}} = 4$）；ILP 1 时需 52 个 active warp，$`\mathcal{O}_{\min} = 52/64 = 0.81`$（1.4 式 (5)）。这是 latency-bound 与 memory-bound 的分界输入。

**命题 1**（latency-bound 上界与三区判定）。设 $I \lt R$（ridge 左侧）、瓶颈为 HBM，提供的在飞并发为 $C$（sector 口径），所需并发为 $L_{\mathrm{req}}$（式 (6)）。稳态纯访存流口径下，达成吞吐为

$$
F_{\mathrm{ach}} = I \cdot \mathrm{BW} \cdot \min\left(1, \frac{C}{L_{\mathrm{req}}}\right) \qquad (7)
$$

特别地， $C \lt L_{\mathrm{req}}$ 时 $F_{\mathrm{ach}} \lt I \cdot \mathrm{BW}$——点落在 roofline 斜线**下方**，称为 latency-bound。合并 ridge 右侧平台区（ $I \ge R$ 且算力管线并发充足 → compute-bound，上限 $F_{\mathrm{peak}}$）与斜线区（ $I \lt R$ 且 $C \ge L_{\mathrm{req}}$ → memory-bound，上限 $I \cdot \mathrm{BW}$），得三区判定。**证明**（要点）：recap 1.4 命题 1——并发不足时瓶颈资源利用率 $= C/L_{\mathrm{req}}$，故达成带宽 $`\mathrm{BW}_{\mathrm{ach}} = \mathrm{BW} \cdot \min(1, C/L_{\mathrm{req}})`$；每 byte 对应 $I$ 个 FLOP，两边乘 $I$ 得式 (7)。完整证明（含 $I \ge R$ 但算力管线并发不足的 compute 侧 latency-bound 口径、非纯访存流的边界）留作练习 3。∎

全数字实例（ $I = 2.5$，models 8B 满窗口径； $L_{\mathrm{req}} = 208$ sectors）：

| 提供并发 $C$ | 利用率 $C/L_{\mathrm{req}}$ | $`\mathrm{BW}_{\mathrm{ach}}`$ | $F_{\mathrm{ach}}$（式 (7)） | roofline 上限 $I \cdot \mathrm{BW}$ |
| --- | --- | --- | --- | --- |
| 208 sectors（52 warp load，ILP 1） | 1.0 | 3352 GB/s | 8.38 TFLOPS | 8.38 TFLOPS |
| 104 sectors（并发减半） | 0.5 | 1676 GB/s | 4.19 TFLOPS | 8.38 TFLOPS |
| 64 sectors（ $\mathcal{O} = 0.25$、16 warp、ILP 1） | 0.308 | 1032 GB/s | 2.58 TFLOPS | 8.38 TFLOPS |

第三行就是低 occupancy 的定量代价： $\mathcal{O} = 0.25$ 配 ILP 1 只提供 $16 \times 4 = 64$ sectors 在飞，带宽利用率 30.8%，吞吐 2.58 TFLOPS——离斜线上限差 3.25 倍。补救沿 1.4 式 (4)：ILP 提到 4（ $16 \times 4 \times 4 = 256 \ge 208$）即可藏住，或提 occupancy 到 0.81（52 warp）。**latency-bound 不是第三条 roof，是 roof 之下的区域**——roofline 上限本身没达到，这与 memory-bound（达到斜线、被带宽封顶）是两种不同的损失。

三区判定流程（实测诊断的入口，完整 drill-down 归 11.2）：

```
实测 (I_ach, F_ach) + occupancy / 在飞并发
        │
        ├─ I_ach ≥ R？ ──是──► compute-bound 区
        │                        看 u_F：指令 mix / MMA 利用率（5.2 / 6.1）
        │否
        ▼
        ├─ C ≥ L_req？ ──是──► memory-bound 区
        │                        看 u_B：coalescing / 流量 / 融合（4.1 / 4.3 / 5.4）
        │否
        ▼
        └─ latency-bound 区 ──► 提 occupancy / ILP（3.4 / 1.4）
                                 或 tiling 降延迟（4.3）
```

**口径注**：命题 1 取 HBM 瓶颈、纯访存流口径。compute-bound kernel 同样可能 latency-bound——算力管线也需要在飞并发（1.4 §4 FMA 热身：单 scheduler 4 条 warp FMA 在飞、全 SM 16 条），并发不足时点落在平台下方；此时 $L_{\mathrm{req}}$ 换算力管线口径，判定结构不变。Nsight 侧的对应签名：`long_scoreboard` 主导且 `not_selected` 接近零 = 延迟没藏住（recap 1.4 §7），与式 (7) 的 $C \lt L_{\mathrm{req}}$ 是同一件事的两个投影。

## 6. 测量方法：nvbandwidth 与 Nsight Compute

理论 3352 GB/s，kernel 实际拿到多少？两个工具、两个口径：**nvbandwidth 测拷贝天花板**（硬件库存口径——顺序流、无计算干扰，测的是「这块硬件最多能搬多快」）；**Nsight Compute 测 kernel 达成值**（负载口径——真实访存模式下「这个 kernel 实际搬了多快、算了多少」）。两者不可混用：拷贝天花板是 achieved 的上界参照，不是 kernel 的实测值。

nvbandwidth（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现；README 2026-11-25 经 GitHub API 一手核实）：

- **两类拷贝引擎**：CE（copy engine）走 memcpy API，SM 走 kernel 拷贝（README 逐字「CE copies use memcpy APIs. SM copies use kernels.」）。CE 不占 SM、SM 拷贝占算力——测 HBM 天花板两种都跑，差异本身是诊断信息。
- **测试名**（README 逐字）：`host_to_device_memcpy_ce` / `device_to_host_memcpy_ce`（PCIe H2D/D2H）；`device_to_device_memcpy_read_sm` / `device_to_device_memcpy_write_ce`（HBM 与 P2P 带宽矩阵，GPU×GPU 逐对）；`device_to_device_bidirectional_memcpy_*`（双向）；`device_to_device_latency_sm` / `device_to_device_latency_tma`（延迟）。
- **测量流程**（README 逐字口径）：spin kernel + CUDA events——先 enqueue 自旋 kernel 占住设备，待全部测量操作入队后释放 flag，把 enqueue 开销排除在传输计时外；默认 3 次采样取中位数（`--testSamples` / `--useMean` 可调）。SM 拷贝把 copy size 截断到「512 threads/block × SM 数」的整数倍（README 公式），保证各 SM 均匀分摊。
- **延迟测试**：pointer chasing（指针追逐链表）而非顺序访问——强制随机模式防 prefetch、防 cache 命中，TLB 保持 warm（README 逐字口径）；与 2.2 延迟表的微基准 [B] 同口径族。

命令片段（教学参照，测试名以 README 为准）：

```bash
./nvbandwidth -t device_to_device_memcpy_read_sm   # HBM/P2P SM 拷贝带宽矩阵
./nvbandwidth -t host_to_device_memcpy_ce          # PCIe H2D（copy engine）
./nvbandwidth -t device_to_device_latency_sm       # P2P 延迟（pointer chasing）
```

两个口径注（对账前必读）：

> **坑**：拿拷贝带宽直接对 HBM 峰值、或拿双向测试值当双向总和。触发条件一：D2D 同卡拷贝报 $X$ GB/s，直接算 $X/3352$ 当带宽利用率；现象：利用率低估一半——拷贝 payload 为 $M$ 字节时 HBM 实际搬读 $M$ + 写 $M$ = $2M$（源与目的都在 HBM），对账峰值须用 $2X/3352$（账口径：memcpy 的读+写双流量；工具输出口径以其 README 为准）。触发条件二：把 bidirectional 测试报值当双向聚合；现象：NVLink 对账差 2 倍——README 逐字「The reported bidirectional GB/s represents per-direction bandwidth under concurrent opposite-direction copy, not an aggregated sum.」，报值是**并发反向流量下的每方向带宽**；NVLink 900 GB/s 是双向总和口径，对账须每方向 ×2（spec §4.3 方向口径）。规避：对账前先问「报值是 payload 还是总流量、单向还是双向总和」。

Nsight Compute（[NVIDIA, Nsight Compute](../references.md)，[C]，ProfilingGuide 2026-11-25 一手核实）：

- **SpeedOfLight Throughput**：逐单元报「achieved percentage of utilization with respect to the theoretical maximum」（逐字）——compute（SM）与 memory（DRAM）两个百分比，即 $u_F$ 与 $u_B$ 的直接读数；metric 后缀 `.pct_of_peak_sustained_elapsed` 定义为「% of peak sustained rate achieved during unit elapsed cycles」（逐字）。
- **MemoryWorkloadAnalysis**：Mem Busy（单元忙碌率）/ 带宽利用率 / Mem Pipes Busy（访存指令发射率）三个口径分开读——单元忙不等于带宽满，带宽满不等于发射忙。
- **达成值计算**：kernel 时长取 `gpu__time_duration`；DRAM 字节计数器（读+写）除以时长得 $`\mathrm{BW}_{\mathrm{ach}}`$；SASS 操作计数器（FADD/FMUL 计 1、FFMA 计 2 FLOPs）除以时长得 $F_{\mathrm{ach}}$。具体计数器名以 Metrics Reference 为准，本书不逐字枚举。实测强度：

$$
I_{\mathrm{ach}} = \frac{\mathrm{FLOPs}_{\mathrm{ach}}}{\mathrm{bytes}_{\mathrm{ach}}}, \qquad \mathrm{BW}_{\mathrm{ach}} = \frac{\mathrm{bytes}_{\mathrm{ach}}}{t_{\mathrm{kernel}}} \qquad (8)
$$

$$
u_B = \frac{\mathrm{BW}_{\mathrm{ach}}}{\mathrm{BW}}, \qquad u_F = \frac{F_{\mathrm{ach}}}{F_{\mathrm{peak}}} \qquad (9)
$$

**符号**：$`\mathrm{FLOPs}_{\mathrm{ach}}`$ / $`\mathrm{bytes}_{\mathrm{ach}}`$ = kernel 实测总 FLOPs / 总 DRAM 字节； $t_{\mathrm{kernel}}$ = kernel 时长； $u_B$ / $u_F$ = 带宽/算力利用率（theoretical 分母取式 (2)(3) 的披露值）。**口径注**： $I_{\mathrm{ach}}$ 的分母是 DRAM 字节（HBM 口径）——若工作集被 L2 吸收，DRAM 字节小于 kernel 总访存字节， $I_{\mathrm{ach}}$ 会高于「算法强度」；层级 roofline（§7）正是为区分这一层而设。

全数字实例（**教学假设口径**，非实测报告）：某 decode kernel 实测 DRAM 带宽 2.85 TB/s、算力 7.125 TFLOPS。代入式 (9)： $u_B = 2.85/3.35 = 0.851$（85.1%）， $u_F = 7.125/989.5 = 0.0072$（0.72%）；式 (8)： $I_{\mathrm{ach}} = 7.125/2.85 = 2.5$——与 models decode 强度口径（≈2.5）自洽。读法：带宽用了 85%、算力只用了 0.72%——典型 memory-bound；剩下 15% 带宽损失的去向（coalescing 浪费、刷新、调度间隙）由 MemoryWorkloadAnalysis 与 memory chart 分解（4.1/11.2）。

测量顺序与时长口径两条纪律。顺序：先跑 nvbandwidth 拿本机拷贝天花板（硬件库存口径，确认设备健康、对账口径正确），再跑 Nsight Compute 拿 kernel 达成值，用式 (9) 算利用率、按 §5 判区，最后逐层 drill-down——这个顺序保证「利用率」的分母可靠，否则 85% 可能是对着错误的天花板算的。时长：Nsight Compute 的 `gpu__time_duration` 在 launch 前后插 semaphore 时间戳、测的是单 kernel 时长；Nsight Systems/CUPTI 用不同方法测 workload 时长（ProfilingGuide 逐字「use different methods to measure the duration of a workload」，[NVIDIA, Nsight Compute](../references.md)，[C]）——前者不含 kernel 间隙，后者含。所以 kernel 级 $u_F$ 与端到端利用率是两个口径，差值是 launch/同步/CPU 间隙；报利用率先声明用的是哪个时长分母，跨工具对账先统一口径（11.2 timeline 分析）。

## 7. 实测 roofline 绘制：点落 roofline 下方 = 利用率损失

实测点为什么必在 roof 之下、差的那一段怎么读？推导路线分三步：先定实测点坐标（式 (8) 的 $I_{\mathrm{ach}}$ 与 $F_{\mathrm{ach}}$），再证点必在 roof 之下（命题 2），最后按点的位置读损失类型。关键一步是**点到 roof 的垂直距离就是利用率损失**——Nsight ProfilingGuide §2.9 逐字：「The distance from the achieved value to the respective roofline boundary … represents the opportunity for performance improvement.」（[NVIDIA, Nsight Compute](../references.md)，[C]）。

**命题 2**（实测点上界）。任何 kernel 的实测点满足

$$
F_{\mathrm{ach}} \le \min\left(F_{\mathrm{peak}},\ I_{\mathrm{ach}} \cdot \mathrm{BW}\right) \qquad (10)
$$

**证明**：两个独立上界。其一，算力单元发射率封顶： $F_{\mathrm{ach}} \le F_{\mathrm{peak}}$。其二，内存系统服务率封顶： $t_{\mathrm{kernel}}$ 内搬的字节 $`\mathrm{bytes}_{\mathrm{ach}} \le \mathrm{BW} \cdot t_{\mathrm{kernel}}`$，故 $`F_{\mathrm{ach}} = I_{\mathrm{ach}} \cdot \mathrm{bytes}_{\mathrm{ach}}/t_{\mathrm{kernel}} \le I_{\mathrm{ach}} \cdot \mathrm{BW}`$。两界取 min 得式 (10)。∎

利用率损失定义（式 (10) 的相对缺口）：

$$
\mathrm{loss} = 1 - \frac{F_{\mathrm{ach}}}{\min\left(F_{\mathrm{peak}},\ I_{\mathrm{ach}} \cdot \mathrm{BW}\right)} \qquad (11)
$$

**符号**： $\mathrm{loss}$ = 利用率损失（0 = 贴 roof，1 = 零吞吐）。正例：贴斜线的 memory-bound kernel（ $u_B = 1$ ，即 100%）loss = 0——它已被带宽封顶，再优化指令 mix 无收益；反例：loss 大不等于「优化空间大」——若 $I_{\mathrm{ach}}$ 本身被算法钉死（decode ≈2.5），斜线区的 loss 只能靠降字节（量化/融合）或提 $u_B$ 收窄，不能靠提算力收窄。

实测 roofline 示意（H100 SXM BF16 主口径；log-log 坐标，示意非按数值比例；三点为**教学假设口径**）：

```
 F_ach (TFLOPS, log)
 ▲
 │                 ┌────────────────────────── F_peak = 989.5（compute 平台）
 │                /│
 │               / │
 │              /  │  斜率 = BW = 3.35 TB/s（memory 斜线）
 │             /   │
 │            /    │             ● A (I=400, F=700)
 │           /     │               compute-bound，u_F = 70.7%
 │          /      │
 │         /       │
 │        /        │
 │       /         │
 │      /          │
 │     ● B (I=2.5, F=7.125) memory-bound，u_B = 85.1%
 │    / ○ C (I=2.5, F=4.19) latency-bound，u = 50%（斜线下方）
 │   /  │
 │  /   │
 │ /    │
 │/     │
 └──────┼─────────────────────────────────────► I_ach (FLOP/byte, log)
        R ≈ 295
```

三点的读法（式 (11) 代入）：

- **A**（ $I_{\mathrm{ach}} = 400$， $F_{\mathrm{ach}} = 700$ TFLOPS）：roof = $\min(989.5, 400 \times 3.35) = 989.5$， $u_F = 700/989.5 = 0.707$（70.7%），loss 29.3%——compute-bound 区的损失在算力侧：指令 mix 非 MMA 主导、MMA 发射间隙、occupancy 不足（5.2/6.1/3.4）。
- **B**（ $I_{\mathrm{ach}} = 2.5$， $F_{\mathrm{ach}} = 7.125$ TFLOPS）：roof = $2.5 \times 3.35 = 8.375$， $u_B = 7.125/8.375 = 0.851$（85.1%），loss 14.9%——memory-bound 区的损失在带宽侧：coalescing 浪费的 sector（4.1）、非有用字节流量、调度间隙。
- **C**（ $I_{\mathrm{ach}} = 2.5$， $F_{\mathrm{ach}} = 4.19$ TFLOPS）：同强度但并发减半（§5 表第二行）， $u = 4.19/8.375 = 0.5$（50%），loss 50%——latency-bound 区的损失在并发侧：提 occupancy/ILP（3.4/1.4）把点从 C 拉回 B 方向，再谈带宽利用率。

$u_F$ 与 MFU 的关系须钉一句。MFU（Model FLOPs Utilization，模型 FLOPs 利用率）= 实际 FLOPs ÷ 峰值 FLOPs，是 $u_F$ 同定义的模型级口径（1.2 §5 把「算力利用率」指向本章即此）；kernel 级 $u_F$ 是它的分解项。端到端 MFU 不等于任何单个 kernel 的 $u_F$——它是各 kernel $u_F$ 的时间加权平均，再被非 kernel 时间（launch 间隙、同步、CPU 侧调度）打折：一个 decode step 由几十个小 kernel 串成，即便每个 kernel 的 $u_B$ 都接近 1，kernel 间隙仍会吃掉端到端利用率。所以实测 roofline 点回答「这个 kernel 跑得多满」，MFU 回答「这一步/这次训练跑得多满」——前者是后者的诊断分解，逐 kernel 分解与间隙归因的全流程归 11.2。

**层级 roofline**（Nsight §2.9 逐字口径）：「Hierarchical roofline overlays can separate L1, L2, and DRAM bandwidth ceilings. If a kernel sits at the DRAM roof, cache efficiency is often limiting throughput. Improving reuse can move the achieved point toward the L2 or L1 ceiling without changing arithmetic intensity.」——同一 kernel 对 L1/L2/DRAM 各有一条斜线（各层带宽不同，§4 表）；点贴 DRAM roof 说明缓存效率是瓶颈，tiling（4.3）提高复用把点移向 L2/L1 的更高 ceiling，**算法强度不变、达成吞吐上移**。这是「tiling 为什么有效」的 roofline 几何解释：不是把点右移（提 $I$），是把点所在的斜线换高（换带宽层）。Tensor Core kernel 须用 tensor roofline section，并交叉核对 profiler 报的 SM frequency 与 SASS 操作计数（Nsight §2.9 逐字口径），否则 FLOP 计数与指令路径对不上。

serving 侧落点一笔（recap inference 1.3 式 (2) + 利用率修正，教学假设口径）：8B bf16 权重 $W = 16.06$ GB（inference 1.3 §1 口径）， $B = 1$ 忽略 KV 的理想步时 $16.06 \times 10^{9}/(3.35 \times 10^{12}) = 4.79$ ms；若 $u_B = 0.85$（85%），步时 $4.79/0.85 = 5.64$ ms——**带宽利用率直接乘在 serving 延迟上**。这不是新推导，是把式 (9) 的 $u_B$ 代回 inference 1.3 已交付的步时式；实测校准与操作点选择归 inference 9.1 / 本章 §6 方法。

本章交付清单（下游 recap 接口）：models 入口 recap + 翻译（§2）、ridge 分子分母微架构乘法链对账（§3 式 (1)-(3)）、层级全表 + 归一化账（§4 式 (4)(5)）、三区判定与 latency-bound 上界（§5 式 (6)(7) + 命题 1）、测量方法与利用率（§6 式 (8)(9)）、实测点上界与损失（§7 式 (10)(11) + 命题 2）。3.4 recap 式 (6) 作所需 occupancy 的输入；4.3 recap §7 层级 roofline 作 tiling 收益的几何解释；11.2 recap §5 判定流程与 §6 工具口径作 profiling 全流程起点；7.2 recap §6 nvbandwidth P2P 测试作带宽矩阵测量。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| $I$/吞吐上限/ridge 为 models 2.4 §6 已交付入口（式 (10)-(12) recap + 翻译，不重推）；prefill $I \approx T_p$、decode $I \approx 1\text{–}2.5$、batch 饱和上界 8B 满窗 ≈4.9 | §2 recap 块 | §2、models 2.4 §6 |
| ridge = $F_{\mathrm{peak}}/\mathrm{BW}$；H100 BF16 ≈295 / H200 ≈206 / HGX B200 ≈292，FP8 ≈591/412/584、B200 FP4 ≈1169——与 inference 1.3 同表逐行一致 | 式 (1) + 对账表 | §3、练习 1 |
| 分母微架构来源：5 栈 × 16 ch × 5.238 Gbps × 64 bit ÷ 8 = 3352.32 GB/s ≈ datasheet 3.35 TB/s（1024 代入含 $n_{\mathrm{ch}}$ 式高估 16×） | 式 (2)，recap 2.1 | §3、练习 1 |
| 分子微架构来源：132 SM × 4096 FLOP/cycle × 1.83 GHz = 989.43 TFLOPS；wp 989.4 / datasheet 989.5 为舍入路径漂移 | 式 (3)，recap 1.1/1.3 | §3 |
| 层级全表：register 256 KB/管线内/4055 GB/s/SM；shared 228 KB/29.0/253 GB/s；L1 40.7/未披露；L2 50 MB/263.0/未披露；HBM 80 GB/478.8/3352 GB/s；NVLink4 900 双向/450 单向；PCIe Gen5 64 单向/128 双向 | §4 表 | §4、练习 2 |
| 归一化：共享层份额 = 芯片 ÷ 132（HBM 25.4 GB/s/SM）；私有层聚合 = 每 SM × 132（register 535 TB/s）；混比差 132× | 式 (4)(5) + 坑注 | §4、练习 2 |
| 片上比值 register:shared:HBM 份额 ≈ 160:10:1；片外 HBM:NVLink 单向:PCIe 单向 ≈ 52:7:1 | §4 比值段 | §4、练习 2 |
| 三区判定： $I \ge R$ compute-bound； $I \lt R$ 且 $C \ge L_{\mathrm{req}}$ memory-bound； $C \lt L_{\mathrm{req}}$ latency-bound（ $L_{\mathrm{req}}$ ≈208 sectors = 52 warp load） | 式 (6)(7)，命题 1 | §5、练习 3 |
| latency-bound 上界 $F_{\mathrm{ach}} = I \cdot \mathrm{BW} \cdot \min(1, C/L_{\mathrm{req}})$；C = 104 → 4.19 TFLOPS（u 0.5）；O 0.25 ILP 1 → 2.58 TFLOPS（u 0.308） | 式 (7) 实例表 | §5、练习 3 |
| nvbandwidth：CE/SM 两类、memcpy/latency 测试族、spin kernel + events 排除 enqueue、3 采样中位数；D2D 拷贝对账峰值须 ×2（读+写）；双向报值是每方向非总和 | §6（README 逐字） | §6、练习 4 |
| Nsight：SpeedOfLight = achieved % of theoretical； $I_{\mathrm{ach}}$ = FLOPs/bytes、 $u_B$ = BW_ach/BW；2.85 TB/s → u_B 85.1%、u_F 0.72%（教学假设） | 式 (8)(9) | §6、练习 4 |
| 实测点必在 roof 下： $F_{\mathrm{ach}} \le \min(F_{\mathrm{peak}}, I_{\mathrm{ach}} \cdot \mathrm{BW})$；loss = 1 − F_ach/roof；点到 roof 距离 = 改进机会（Nsight 逐字） | 式 (10)(11)，命题 2 | §7、练习 4 |
| 层级 roofline：L1/L2/DRAM 三 ceiling；贴 DRAM roof → 缓存效率瓶颈；tiling 把点移向更高 ceiling、强度不变 | §7（Nsight §2.9 逐字） | §7、4.3 |
| 带宽利用率直乘 serving 延迟：8B 权重 16.06 GB，理想步时 4.79 ms，u_B 85% → 5.64 ms（教学假设） | §7 落点段 | §7、inference 1.3 |

## 练习

1.（计算 · [10]）ridge 代入与来源对账（式 (1)-(3)）：(a) 用 datasheet dense 值算 H100 BF16 与 H200 BF16 的 ridge；(b) 算 HGX B200 FP8 ridge；(c) 用式 (2) 重算 H100 HBM 带宽（GB/s），核对 datasheet 3.35 TB/s。校验：(a) ≈295 / ≈206，(b) ≈584，与 inference 1.3 §2 ridge 表一致（容差 1%）；(c) 3352.32 GB/s，与 2.1 式 (2) 逐数一致。
2.（计算 · [20]）层级归一化账（式 (4)(5)，H100 主口径）：(a) HBM 每 SM 份额；(b) register 芯片聚合带宽（4055 GB/s/SM × 132）与 shared 芯片聚合（253 GB/s/SM × 132）；(c) HBM : NVLink 单向 : PCIe 单向比值；(d) 若误把 register 4055 GB/s/SM 与 HBM 芯片级 3352 GB/s 直接比，得「快几倍」，与正确比值 160 差几倍？先预测 (d) 的方向再算。校验：(a) 25.4 GB/s；(b) ≈535 TB/s / ≈33.4 TB/s；(c) ≈52:7:1；(d) 4055/3352 ≈ 1.2，正确值 160——差约 132 倍（ $n_{\mathrm{SM}}$ 口径错误，与 §4 坑注一致）。
3.（推导 · [30]）补全命题 1 的证明（三区判定与 latency-bound 上界）：(a) recap 1.4 命题 1，写出利用率 $= C/L_{\mathrm{req}}$ 的推导（Little's law 重排：吞吐 = 并发/延迟）；(b) 由达成带宽乘 $I$ 推式 (7)，并分 $C \ge L_{\mathrm{req}}$ 与 $C \lt L_{\mathrm{req}}$ 两案讨论；(c) 讨论 $I \ge R$ 但算力管线并发不足的情形——此时 $L_{\mathrm{req}}$ 取什么口径（提示：1.4 §4 FMA 热身的全 SM 16 条 warp FMA），点落在哪条 roof 下方；(d) 数值验证 §5 表三行（8.38 / 4.19 / 2.58 TFLOPS）。校验：(b) 两案分别退化为 memory-bound 上限与 latency-bound 上界，方向不可反；(d) 与 §5 表逐数一致（容差 1%）；(c) 与 §5 口径注一致。
4.（计算 · [30]）实测点综合账（**教学假设口径**；式 (8)-(11)）：某 kernel 时长 2 ms，实测 DRAM 字节 $5.7 \times 10^{9}$ B、FLOPs $1.425 \times 10^{10}$：(a) 求 $`\mathrm{BW}_{\mathrm{ach}}`$、 $F_{\mathrm{ach}}$、 $I_{\mathrm{ach}}$；(b) 求 roof、 $u_B$ 与 loss，判定属于三区中哪一区；(c) 若并发减半（ $C = 104$ sectors），点移到哪里、loss 变多少；(d) 该 kernel 的 $u_B$ 若用 nvbandwidth D2D 同卡拷贝对账，拷贝报值应为多少 GB/s（读+写双流量口径）？先预测 (b) 的区再算。校验：(a) 2.85 TB/s / 7.125 TFLOPS / 2.5，与 §6 实例自洽；(b) roof 8.375 TFLOPS、 $u_B$ 85.1%、loss 14.9%，memory-bound；(c) 4.19 TFLOPS、loss 50%（§5 表第二行）；(d) 拷贝 payload 口径 = $2.85/2 = 1.425$ TB/s = 1425 GB/s（§6 坑注 ×2 关系的逆用）。

## 参考

- [NVIDIA, datasheet](../references.md)：NVIDIA H100 / H200 / HGX H100-H200 / Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM BF16 1979*→989.5 dense / 3.35 TB/s / 80 GB、H200 4.8 TB/s / 141 GB、HGX B200 2250 / 4500 / 9000 dense / 7.7 TB/s / 180 GB / PCIe Gen5 128 GB/s（双向印法）、GB200 2500 / 8 TB/s / 186 GB 对照；sparse 脚注「Dense is one-half of the sparse spec shown.」（§1/§3，与 inference 1.3 同表）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——Table 3（5 HBM3 stacks / 2619 MHz DDR / 3352 GB/sec / 132 SM / TC-boost 1830 MHz / BF16 dense 989.4 / L2 50 MB / RF 256 KB）、舍入路径漂移注（989.43 → wp 989.4 / datasheet 989.5，recap 1.1 §3）
- [NVIDIA, nvbandwidth](../references.md)：NVIDIA/nvbandwidth repo（key: nvbandwidth），教学参照 + 官方实现，README 2026-11-25 经 GitHub API 一手核实——CE/SM 两类拷贝逐字、测试名（host_to_device_memcpy_ce / device_to_device_memcpy_read_sm / write_ce / bidirectional / latency_sm / latency_tma）、spin kernel + CUDA events 测量流程、3 采样中位数、SM 拷贝 512 threads/block 截断公式、pointer chasing 延迟口径、双向报值「per-direction … not an aggregated sum」逐字（§6）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，ProfilingGuide 2026-11-25 一手核实——§2.9 Roofline Charts（Work/Memory Traffic 定义、achieved value、点到 roof 距离 = 改进机会逐字、层级 L1/L2/DRAM ceiling 逐字、tensor roofline + SM frequency/SASS 交叉核对）、SpeedOfLight「achieved percentage … theoretical maximum」逐字、.pct_of_peak_sustained_elapsed 后缀定义、MemoryWorkloadAnalysis 三口径、gpu__time_duration（§6/§7）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV 延迟 cycles：shared 29.0 / L1 40.7 / L2 263.0 / global 478.8（§4 表延迟列实测口径）
- [NVIDIA, NVLink/NVSwitch](../references.md)：NVIDIA NVLink / NVSwitch 产品页（key: nvlink-nvswitch），[C] 官方规格披露·一手权威，2026-11-24——H100 18 links × 25 GB/s 单向 = 450 单向 = 900 双向（NVLink4）；B200 NVLink5 1.8 TB/s 双向（brief 逐字互证；精确代际值 W4 复核）（§4 表）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露·一手权威，2026-11-24——Gen5 x16 = 64 GB/s 单向（spec §4.3 数值口径；URL 403 bot 拦截，精确出处 W4 复核，双向 128 GB/s 经 datasheet 印法互证）（§4 表）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——KB = 1024 逐字注（§1 单位口径）、Table 34/35 片上容量（§4 表，recap 1.2/2.3）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——「18 fifth-generation NVLink links to provide 1.8 [TB/s]」逐字（§4 世代对照注）
