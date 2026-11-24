# 2.1 HBM 与显存子系统（HBM and the Memory Subsystem）

> 3.35 TB/s 这个数从哪来？本章从栈几何逐项推导 HBM 带宽—— $\mathrm{BW} = n_{\mathrm{stack}} \cdot n_{\mathrm{ch}} \cdot f_{\mathrm{pin}} \cdot w_{\mathrm{ch}} / 8$（ $w_{\mathrm{ch}}$ = 每通道 64 bit；1024 bit/stack 是 $n_{\mathrm{ch}} w_{\mathrm{ch}}$ 的积，代入含 $n_{\mathrm{ch}}$ 因子的式会 16× 高估）——交付栈结构（die stack / TSV / channel / pseudo-channel）、ECC 开销账、内存控制器地址映射（channel interleaving / partition camping）与容量账（raw / usable / nvidia-smi 三口径） | 前置：1.1（芯片框图与 CoWoS、5120-bit 接口、世代对比表）、models 1.2（内存账——本章交付硬件库存）；下游 2.2（缓存层级）、2.4（带宽账与 roofline——recap 本章式 (1) 作微架构来源并对账 datasheet）、4.1（访存合并）、11.3（ECC 计数与 XID 故障） | 锚点：NVIDIA Hopper whitepaper V1.04 Table 3 / Fig 6（[NVIDIA, Hopper whitepaper](../references.md)，[C] 官方规格披露·一手权威）、H100/H200/HGX/Blackwell datasheet（[NVIDIA, datasheet](../references.md)，[C]）、JEDEC HBM3 JESD238A + Micron/Samsung/SK hynix 成员官方互证（[JEDEC, HBM3](../references.md)，[C]）、Blackwell Technical Brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、Blackwell Tuning Guide（[NVIDIA, Blackwell tuning guide](../references.md)，[C]）、B200 功耗微基准（[Zand et al., 2026](../references.md)，arXiv:2610.41919，[B] 实测口径）——官方规格 2026-11-25 一手核实（dam-cdn PDF）

## 1. 3.35 TB/s 从哪来：本章口径

H100 SXM 的 datasheet 印显存带宽 3.35 TB/s，whitepaper Table 3 印 3352 GB/sec——这个数不是测出来的，也不是营销取整，是四个物理因子的乘积：5 栈 × 16 通道 × 5.238 Gbps × 64 bit ÷ 8。本章回答三个问题：带宽公式的四个因子各对应栈几何的什么，为什么每通道位宽必须代 64 而不是 1024？ECC 的开销在哪，吃不吃带宽？地址映射怎么决定标称带宽能不能用满，raw 与 usable 容量差多少？

**本章口径**（章脊锁定；本章为跨 SKU 几何推导章，多 SKU 口径在此一次声明，表外不再换）：

| 口径 | SKU 与几何 | 来源 |
| --- | --- | --- |
| 主口径 | H100 SXM5：5 HBM3 栈 × 16 通道 × 64 bit，2619 MHz DDR，80 GB，3352 GB/s | [NVIDIA, Hopper whitepaper](../references.md) Table 3 逐字 + [NVIDIA, datasheet](../references.md)，[C] |
| 对比口径 | H200 SXM（141 GB / 4.8 TB/s）、HGX B200（180 GB / 7.7 TB/s）、GB200 每 GPU（186 GB / 8 TB/s） | 四份 datasheet，[C] |
| 反推口径 | H200/B200 的 $f_{\mathrm{pin}}$ 与栈数：官方未逐项披露，由带宽/容量反推，逐处标「反推」 | §3 / §6 自洽核对 |

单位纪律：显存 GB 按 $10^{9}$ B、带宽 TB/s 按 $10^{12}$ B/s（datasheet 十进制口径）；GiB 按 $2^{30}$ B（nvidia-smi 二进制口径，§6）；片上存储 KB = 1024 B（Programming Guide 逐字注，recap 1.2）；L2 的 50 MB 沿 whitepaper 印法（归 2.2）。符号（spec §4.3）： $n_{\mathrm{stack}}$ = 栈数； $n_{\mathrm{ch}}$ = 每栈通道数； $f_{\mathrm{pin}}$ = pin rate（每 pin 每秒传输次数）； $w_{\mathrm{ch}}$ = **每通道**位宽（HBM3 = 64 bit）； $\mathrm{BW}$ = 带宽。bytes 不占字母，B/KB/MB/GB/TB 是单位符号豁免。

**边界声明**：本章交付 HBM 带宽的**微架构来源**（栈几何逐项推导）、ECC 开销账、内存控制器地址映射机制与容量库存。带宽的测量方法（nvbandwidth）与层级带宽-延迟-容量全表、roofline 归 2.4——2.4 将 recap 本章式 (1) 并与 inference 1.3 datasheet 参数表对账（inference W1 errata 裁定的分工：参数表归 inference 1.3，微架构来源与测量归 gpu 2.4）；访存模式侧的 coalescing 判定归 4.1；缓存层级归 2.2；生产侧 ECC 计数与 XID 故障处理归 11.3。本章无代码走读（纯硬件机制章）。

## 2. HBM 栈结构：TSV、channel 与 1024-bit 常量

先回答一个更基本的问题：为什么 HBM 能做到 3.35 TB/s，而消费卡的 GDDR 做不到？因为 HBM 用**总线宽度**换带宽——每栈 1024 bit、5 栈并联 5120 bit；GDDR 走的是窄总线、高 pin rate 路线（每芯片接口几十 bit 量级），HBM 走宽总线、中 pin rate 路线。宽度从栈结构来：DRAM die 垂直堆叠，TSV（through-silicon via，硅通孔）穿透 die 做垂直互连，让上千根线在毫米级距离内走通——这正是 CoWoS 硅中介层封装的物理前提（recap 1.1 §5：5120 根线若走封装基板的毫米级走线，扇出与信号完整性不可行）。

所谓 HBM（High Bandwidth Memory，高带宽内存），是指把多颗 DRAM die 经 TSV 垂直互连堆叠、经底部 logic die 引出每栈 1024-bit 宽接口的堆叠显存。栈结构（侧视 + 顶视；几何经成员官方互证，[JEDEC, HBM3](../references.md)，[C]）：

```
HBM3 栈（侧视：8-high DRAM die + base logic die）
┌───────────────────────────────┐
│  DRAM die 7   ─┐              │
│  DRAM die 6    │ TSV 垂直互连 │
│  …             │（穿透各 die）│
│  DRAM die 0   ─┘              │
├───────────────────────────────┤
│  base logic die（PHY/测试）   │
└───────────────┬───────────────┘
                │ microbump → 硅中介层（CoWoS，1.1 §5）
          GPU die（MC ×10，§5）

顶视（每栈）：1024-bit = 16 channel × 64 bit
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ ch0 │ ch1 │ ch2 │ ch3 │ ch4 │ ch5 │ ch6 │ ch7 │ → MC 0（512-bit）
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ ch8 │ ch9 │ch10 │ch11 │ch12 │ch13 │ch14 │ch15 │ → MC 1（512-bit）
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
每 channel 再分 2 个 pseudo-channel（各 32 bit），全栈 32 个
```

三个结构事实，各给出处：

- **die 堆叠高度**：8-high / 12-high（Samsung「12H, 8H」、SK hynix「8,12Hi」逐字）；供 H200 的 24 GB 栈是 8-high（Micron 逐字「24GB 8-high HBM3E is shipping with NVIDIA H200」）。每 die 容量 = 栈容量 ÷ 高度：24 GB ÷ 8 = 3 GB/die（24 Gb）；H100 的 16 GB 栈按 8-high 口径反推为 8 × 2 GB/die（16 Gb）。
- **channel 划分**：HBM3 每栈 16 个独立 channel、每 channel 64 bit，合计 1024 个 IO（Micron 逐字「HBM3E offers 16 independent channels and 32 pseudo channels」+「1024 IO pins」）。上一代 HBM2e 同为 1024 bit/stack，但划分为 8 channel × 128 bit（HBM2/2e 标准口径；本波一手核实聚焦 HBM3/3e）——HBM3 把通道数翻倍、每通道宽度减半，通道粒度更细，独立并行的访问流更多。full GH100 两种栈都支持（whitepaper 逐字「6 HBM3 or HBM2e stacks」）。
- **pseudo-channel**：每个 64-bit channel 再分 2 个可独立调度的 pseudo-channel（各 32 bit），全栈 32 个（Micron 32 pseudo channels 逐字）。pseudo-channel 让同一物理 channel 内两路访问并行推进，提高小粒度随机访问的并行度——不必等整个 channel 服务完一路再服务下一路。

**口径注**：HBM3e 不是独立的 JEDEC 标准编号，是 vendor speed-bin——Samsung HBM3E 9.2 Gbps、SK hynix HBM3E 9.6 Gbps、Micron 大于 9.2 Gbps，都是 HBM3 家族的速度档位（[JEDEC, HBM3](../references.md)，[C]）。JEDEC JESD238A 是 HBM3 标准编号，原文在本波核实中 403 bot 拦截，几何经成员官方产品页互证，精确 URL 留 W4（同 PCI-SIG 通道）。

世代对比表（vendor 披露口径；每栈带宽 = pin rate × 1024/8，式 (3) 的 $n_{\mathrm{stack}} = 1$ 特例）：

| 世代 | channel 划分 | 每栈总线 | pin rate（vendor） | 每栈带宽 | 栈容量 |
| --- | --- | --- | --- | --- | --- |
| HBM2e | 8 ch × 128 bit | 1024 bit | —（上一代，本波不展开） | — | — |
| HBM3 | 16 ch × 64 bit | 1024 bit | 6.4 Gbps（Samsung） | 819 GB/s | 16 / 24 GB |
| HBM3E | 16 ch × 64 bit | 1024 bit | 9.2（Samsung）/ 9.6（SK hynix）Gbps | 1180 / 1229 GB/s | 24–36 GB |

全数字互证（第一个实例）：Samsung HBM3 标称 6.4 Gbps/pin，每栈带宽 $6.4 \times 10^{9} \times 1024/8 = 819.2$ GB/s，与 Samsung 官方「up to 819 GB/s」逐字对上。每栈口径的公式在这里第一次闭环；§3 把它乘上栈数，就是芯片级带宽。

## 3. 带宽来源推导：四个因子逐项乘

为什么 $w_{\mathrm{ch}}$ 必须代 64 而不是 1024？推导路线分四步：先写单通道带宽（pin rate × 位宽 ÷ 8），再乘通道数与栈数得式 (1)，然后量纲对账锁定每个因子的口径，最后代入 H100 披露值逐项核对官方 3352 GB/s。关键一步是第三步——**1024 bit/stack 已含通道数因子**，代入含 $n_{\mathrm{ch}}$ 的式会把通道数算两遍，高估 16 倍。

单通道每次传输搬 $w_{\mathrm{ch}}$ 个 bit（每 pin 每传输 1 bit， $w_{\mathrm{ch}}$ 根 pin 并行），pin rate 为每秒 $f_{\mathrm{pin}}$ 次传输，故单通道带宽 $f_{\mathrm{pin}} w_{\mathrm{ch}}/8$ B/s；乘每栈 $n_{\mathrm{ch}}$ 个通道、 $n_{\mathrm{stack}}$ 个栈：

$$
\mathrm{BW} = \frac{n_{\mathrm{stack}} \cdot n_{\mathrm{ch}} \cdot f_{\mathrm{pin}} \cdot w_{\mathrm{ch}}}{8} \qquad (1)
$$

**符号**： $\mathrm{BW}$ = 芯片级显存带宽（B/s）； $n_{\mathrm{stack}}$ = 栈数； $n_{\mathrm{ch}}$ = 每栈通道数； $f_{\mathrm{pin}}$ = pin rate（transfers/s/pin）； $w_{\mathrm{ch}}$ = 每通道位宽（bit/channel）；8 = bit → byte。量纲对账（逐项核，单位一致硬要求）：

| 因子 | 含义 | 量纲 |
| --- | --- | --- |
| $n_{\mathrm{stack}}$ | 栈数 | stacks |
| $n_{\mathrm{ch}}$ | 每栈通道数 | channels/stack |
| $f_{\mathrm{pin}}$ | pin rate | transfers/(s·pin) |
| $w_{\mathrm{ch}}$ | 每通道位宽 | bit/channel（= pins/channel） |
| 四因子积 ÷ 8 | 带宽 | 约化后 transfers/s × 1 bit/transfer = bit/s；÷ 8 = B/s |

H100 SXM5 逐项代入（全部披露值，[NVIDIA, Hopper whitepaper](../references.md) Table 3，[C]）： $n_{\mathrm{stack}} = 5$（逐字「5 HBM3 stacks」）； $n_{\mathrm{ch}} = 16$、 $w_{\mathrm{ch}} = 64$ bit（Micron 互证）； $f_{\mathrm{pin}}$ 由「Memory Data Rate 2619 MHz DDR」换算——DDR 双沿采样， $2619 \times 2 = 5238$ MT/s = 5.238 Gbps/pin。代入式 (1)：

$$
\mathrm{BW} = \frac{5 \times 16 \times 5.238 \times 10^{9} \times 64}{8} = 3352.32 \ \mathrm{GB/s} \qquad (2)
$$

对账两个官方数：whitepaper Table 3「Memory Bandwidth 3352 GB/sec」（舍入）、datasheet 3.35 TB/s——推导闭环。顺带核总线宽： $5 \times 16 \times 64 = 5120$ bit，与 Table 3「Memory Interface 5120-bit HBM3」逐字一致；这就是 1.1 §5 CoWoS 那 5120-bit 接口的来源。

等价每栈口径：因为 $1024 = n_{\mathrm{ch}} \times w_{\mathrm{ch}}$，式 (1) 可合并通道因子重写为

$$
\mathrm{BW} = \frac{n_{\mathrm{stack}} \cdot f_{\mathrm{pin}} \cdot 1024}{8} \qquad (3)
$$

代入 $5 \times 5.238 \times 10^{9} \times 1024/8 = 3352.32$ GB/s，与式 (1) 同值。两式是同一乘积的两种结合方式，不是两个公式。

**命题 1**（口径等价与误代倍数）。记 $W_{\mathrm{stack}} = n_{\mathrm{ch}} w_{\mathrm{ch}}$（每栈总线宽）。则 (a) 式 (1) 与式 (3) 恒等；(b) 在式 (1) 中把 $w_{\mathrm{ch}}$ 误代为 $W_{\mathrm{stack}}$，结果是真值的 $n_{\mathrm{ch}}$ 倍。**证明**：(a) 乘法结合律， $n_{\mathrm{stack}} n_{\mathrm{ch}} f_{\mathrm{pin}} w_{\mathrm{ch}} = n_{\mathrm{stack}} f_{\mathrm{pin}} (n_{\mathrm{ch}} w_{\mathrm{ch}})$；(b) 误代式为 $n_{\mathrm{stack}} n_{\mathrm{ch}} f_{\mathrm{pin}} W_{\mathrm{stack}}/8$，与真值之比 $= W_{\mathrm{stack}}/w_{\mathrm{ch}} = n_{\mathrm{ch}}$，HBM3 的 $n_{\mathrm{ch}} = 16$。∎

全数字误代实例： $5 \times 16 \times 5.238 \times 10^{9} \times 1024/8 = 53637$ GB/s ≈ 53.6 TB/s——真值 3352 GB/s 的 16 倍，比 NV-HBI die-to-die 的 10 TB/s（1.1 §5）还高 5 倍，显然荒谬。这个荒谬是自检信号：**推导出的带宽若比官方值大整数倍，先查总线宽口径是否重复计了通道数**。

> **坑**：把 1024 bit/stack 代入含 $n_{\mathrm{ch}}$ 因子的式。触发条件：见到「HBM 1024-bit」就直接代 $w_{\mathrm{ch}}$；现象：带宽高估 16×（H100 算成 53.6 TB/s），下游 roofline 拐点、memory-bound 判定、带宽利用率全部跟着失真；规避： $w_{\mathrm{ch}} = 64$ bit/channel 是每通道口径，1024 = $n_{\mathrm{ch}} w_{\mathrm{ch}}$ 是每栈口径——式 (1) 代 64、式 (3) 代 1024，两式不混代（命题 1）。

SKU 对比表（H200/B200 的 $f_{\mathrm{pin}}$ 为**反推**——官方只披露 H100 的 2619 MHz DDR，其余 SKU 不逐字给 per-pin rate）：

| SKU | $n_{\mathrm{stack}}$ | $f_{\mathrm{pin}}$ | 式 (1) 推导 | 官方口径 |
| --- | --- | --- | --- | --- |
| H100 SXM5 | 5（披露） | 5.238 Gbps（披露，2619 DDR × 2） | 3352.32 GB/s | wp 3352 GB/sec / datasheet 3.35 TB/s |
| H200 SXM | 6（配置反推：full GH100 6 栈，141 ≈ 6 × 24） | 6.25 Gbps（反推） | 4800 GB/s | datasheet 4.8 TB/s |
| HGX B200 | 8（反推：raw 192 = 8 × 24） | ≈7.52 Gbps（反推） | 7700 GB/s | datasheet 7.7 TB/s |
| GB200 每 GPU | 8（同上） | ≈7.81 Gbps（反推） | 8000 GB/s | datasheet 8 TB/s |

反推方法：式 (1) 重排为 $f_{\mathrm{pin}} = 8 \cdot \mathrm{BW}/(n_{\mathrm{stack}} \cdot n_{\mathrm{ch}} \cdot w_{\mathrm{ch}})$。代入 H200： $8 \times 4.8 \times 10^{12}/(6 \times 16 \times 64) = 6.25$ Gbps；GB200： $8 \times 8 \times 10^{12}/(8 \times 1024) = 7.8125$ Gbps；HGX B200 同法得 7.52 Gbps。反推值均低于 vendor 最大 speed-bin（9.2 / 9.6 Gbps）——出货 SKU 的工作点由 NVIDIA 设定，不跑 vendor 最高 pin rate；HGX B200 与 GB200 的带宽差（7.7 对 8 TB/s）对应两个 SKU 档位（TDP 1000 对 1200 W，recap 1.1 §4），具体功耗-频率取舍属闭源，本书不推定。

**SKU 口径注**：H100 NVL（94 GB / 3.9 TB/s）是另一变体，勿混入 SXM 列；HGX B200 180 GB / 7.7 TB/s、GB200 186 GB / 8 TB/s、raw 192 GB 是三个数三个口径（§6）；2.4 对账以 inference 1.3 同表（HGX B200）为准。

vendor speed-bin 互证（式 (3) 取 $n_{\mathrm{stack}} = 1$）：Samsung HBM3E 9.2 Gbps → $9.2 \times 10^{9} \times 1024/8 = 1177.6$ GB/s ≈ 官方 1180 GB/s；SK hynix 9.6 Gbps → 1228.8 GB/s ≈ 官方 1.23 TB/s。两家独立 vendor 用同一公式闭环——式 (1) 不是 NVIDIA 专属口径，是 HBM 家族的通用账。

## 4. ECC：校验位从哪来，开销去哪

ECC 要花多少校验位，吃不吃带宽？Hopper whitepaper 的「ECC Memory Resiliency」节（p.38）对后一问有逐字答案：H100 采用 Sideband ECC，校验位不占数据总线——官方 3352 GB/s 不被 ECC 打折。本节先列三条逐字披露，再用 Hamming 界给校验位开销的教学实例化，最后对比 Sideband 与 Inline 两种放置口径的带宽账。

**披露事实**（[NVIDIA, Hopper whitepaper](../references.md) ECC Memory Resiliency 节，[C] 官方规格披露·一手权威，逐字）：

- **SECDED 码族**：「The H100 HBM3/2e memory subsystems support Single-Error Correcting Double-Error Detecting (SECDED) Error Correction Code (ECC) to protect data.」
- **Sideband 放置**：「H100 supports “Sideband ECC” for its HBM3/2e memories, where a small memory region, separate from the main HBM memory, is used for ECC bits (which is in contrast to “Inline ECC”, where a portion of main memory is carved out to store ECC bits).」——校验位存于独立于主 HBM 的一小块存储区域；对照口径 Inline ECC 则从主存中划出一部分存校验位。
- **片上 SRAM 清单**：「Other key memory structures in H100 are also protected by SECDED ECC including the L2 cache and the L1 caches and register files inside all the SMs.」——L2、每 SM 的 L1 与 register file 都在 SECDED 保护之列。

**口径注**：逐字披露的是 SECDED 码族与 Sideband 放置；具体码参数（数据字宽、校验位编码）未逐字披露。下面的 Hamming 界账取 $d = 64$ 作**教学实例化**，给开销的量级，不是对 H100 码参数的断言。

所谓 SECDED（single-error correction, double-error detection），是指能纠正任意单 bit 错、检出任意双 bit 错的纠错码——正是 wp 逐字点名的码族。推导路线：校验位必须给每个单 bit 错误位置一个不同的 syndrome（校验子）——对 syndrome 计数得 Hamming 界；代入 64-bit 数据字得最少校验位；SECDED 在其上加一位总偶校验。

**命题 2**（SECDED 校验位下界）。纠正 $d$ bit 数据字的任意单 bit 错（含校验位自身出错），校验位数 $r$ 须满足 $2^{r} \ge d + r + 1$；SECDED 构造在 SEC 码上加一位总偶校验位，共 $r+1$ 位。**证明**（要点）： $r$ 个校验位形成 $2^{r}$ 个 syndrome，须区分「无错」与 $d+r$ 个 bit 位置各自出错，共 $d+r+1$ 个状态，故得不等式；双 bit 错的总偶校验不变而 SEC syndrome 非零，两者合取可与单 bit 错区分。完整计数论证与 $d = 64$ 代入留作练习 3。

代入 $d = 64$（教学实例化）： $r = 6$ 时 $2^{6} = 64 \ge 71$ 不成立； $r = 7$ 时 $128 \ge 72$ 成立——最少 7 位校验，SECDED 加 1 位总偶校验共 8 位。存储开销 $8/64$ = 12.5%：每 64 个数据 bit 要多存 8 个校验 bit。

带宽开销去哪，对比两种放置口径（口径注：「开销」此处指校验位相对数据位的占比，不是对官方带宽的修正）：

- **Sideband ECC（H100 采用，wp 逐字）**：校验位存于独立存储区域，数据总线宽度不受影响——**数据带宽不变**，这正是官方 3352 GB/s 无需 ECC 折扣的原因；代价是额外的存储区域与走线。
- **Inline ECC（wp 对照口径）**：从主存划出一部分存校验位，校验位与数据共用同一存储与总线——按教学实例化（64 数据 bit + 8 校验 bit = 72 bit 码字），总线每搬 72 bit 只有 64 bit 是数据，有效数据带宽 = 标称 × $64/72 \approx 0.889$（88.9%），即 11.1% 的总线搬运量花在检查位上。

所以开场问题的答案：存储开销是码族固有的（教学实例化 12.5%）——Sideband 省的是总线开销，不是存储开销，独立区域仍要存校验位；带宽开销取决于放置口径，H100 的 Sideband 放置让数据总线不搬校验位，数据带宽开销为零。片上 SRAM 的单 bit 翻转（软错误）由 SECDED 纠正，避免静默数据损坏；wp 同页还披露 Memory Row Remapping（每 bank 预留 spare row，坏行在 boot 时以预留好行替换）。生产侧的 ECC 计数监控与 XID 故障分类归 11.3。

## 5. 内存控制器与地址映射：interleaving 与 partition camping

标称带宽 3352 GB/s 是 80 个通道的聚合——单个 kernel 凭什么能用满 80 个通道？答案在地址映射：内存控制器把物理地址打散到各通道，访问模式决定打散得均不均匀。本节先给 MC 库存，再给 interleaving 机制，最后算 partition camping 的最坏账。

MC 库存（[NVIDIA, Hopper whitepaper](../references.md) Table 3 / Fig 6，[C]）：H100 SXM5 有 **10 个 512-bit 内存控制器**，每栈 2 个（512 bit = 8 channel × 64 bit）；full GH100 为 12 个 / 6 栈。MC 负责地址解码（物理地址 → 栈/通道/bank/行/列）、访问调度与刷新；调度细节属闭源，本书只给机制层。DRAM 的 row buffer 机制（教材口径）：读须先把行 activate 进 row buffer，同行连续访问（row hit）直接取；row miss 须 precharge 再 activate，多付一段延迟。所以顺序流 row hit 率高、随机流低——这是实测带宽低于标称的机制原因之一（延迟之外的另一笔）。

channel interleaving：地址映射把连续地址按细粒度打散到不同通道，让顺序流自动铺满全部 80 个通道并行——这是式 (1) 的聚合带宽可用的映射前提。interleave 粒度与 hash 函数属闭源未披露，下面的账用**简化线性模型**（教学口径）：设 interleave 粒度 $g$ 字节，通道号取「地址 ÷ $g$ 的商除以 80 的余数」。

所谓 partition camping，是指访问模式集中映射到少数通道/分区，使该子集饱和而其余空闲的现象。简化模型下的最坏情形是全部访问落同一通道：

**命题 3**（camping 带宽上界）。简化线性 interleaving 模型下，若全部访问落在同一通道，有效带宽至多为单通道带宽 $f_{\mathrm{pin}} w_{\mathrm{ch}}/8$，占峰值的 $1/(n_{\mathrm{stack}} n_{\mathrm{ch}})$。**证明**：单通道服务率上限即 $f_{\mathrm{pin}} w_{\mathrm{ch}}/8$（式 (1) 的单通道项）；全部请求在该通道排队，完成率不超过其服务率；与峰值之比 $= 1/(n_{\mathrm{stack}} n_{\mathrm{ch}})$。∎

全数字实例：取 $g = 256$ B（假设值），stride = $80 \times 256 = 20480$ B = 20 KB——访问序列每笔的通道号相同，全落一个通道。单通道带宽 $5.238 \times 10^{9} \times 64/8 = 41.9$ GB/s，只有峰值 3352 GB/s 的 $1/80$ = 1.25%。这就是「标称 3.35 TB/s、实测几十 GB/s」的定量代价。

两句边界：其一，实际地址 hash 属闭源，细粒度 interleaving 加 hash 正是为缓解 camping——简化模型给的是最坏上界，不是断言 H100 在 stride 20 KB 时必然 camping；其二，camping 也发生在 L2 分区层（H100 两分区，最坏容量与带宽份额减半），归 2.2；访问模式侧的 coalescing 判定（哪些访问合并成哪些 sector）归 4.1，本节只交付映射侧机制。

## 6. 容量账：raw、usable、nvidia-smi 三口径

80 GB 里有多少能用？H100 的 raw 与 usable 相等（5 × 16 = 80），但 H200/B200 不是——brief 印「Up to 192 GB」，datasheet 印 180 GB，nvidia-smi 报 179 GiB。三个数三个口径，混用是容量账最常见的错。

容量账表（[NVIDIA, datasheet](../references.md) + [NVIDIA, Blackwell brief](../references.md)，[C]；实测列 [Zand et al., 2026](../references.md)，arXiv:2610.41919，[B]）：

| SKU | 栈几何（raw） | datasheet usable | 差额 | nvidia-smi 实测 |
| --- | --- | --- | --- | --- |
| H100 SXM5 | 5 × 16 GB = 80 GB | 80 GB | 0 | — |
| H200 SXM | 6 × 24 GB = 144 GB | 141 GB | 3 GB | — |
| HGX B200 | 8 × 24 GB = 192 GB | 180 GB | 12 GB | 179 GiB ≈ 192.2 GB |
| GB200 每 GPU | 8 × 24 GB = 192 GB | 186 GB | 6 GB | — |

栈数与 die 密度的反推链：H200 的 141 ≈ 6 × 24，且 full GH100 本就是 6 栈配置（whitepaper）→ 6 栈；B200 的 raw 192 = 8 × 24（brief「Up to 192 GB」= 8 栈 × 24 GB）→ 8 栈。每 die 密度：24 GB 8-high → 3 GB/die（24 Gb）；H100 的 16 GB 栈按 8-high 反推 2 GB/die（16 Gb）。**raw 是栈几何乘积，usable 是出货 SKU 口径**，差额为出货预留容量（具体用途未逐字披露，本书不断言）。

单位换算（GiB ↔ GB）：nvidia-smi 按二进制 GiB 报， $179 \times 2^{30} = 1.922 \times 10^{11}$ B ≈ 192.2 GB（十进制）——与 raw 192 对上，而不是 datasheet usable 180（[Zand et al., 2026](../references.md)，arXiv:2610.41919，[B] 实测口径；同文 B300 报 269 GiB ≈ 288.8 GB，同样 ≈ raw 288 对 usable 270）。所以 nvidia-smi 报值口径接近 raw 栈容量，datasheet usable 是 SKU 标称——做容量账先问拿的是哪个口径，跨章对账用同表同口径（2.4 以 inference 1.3 同表 HGX B200 180 GB 为准）。容量装得下什么模型（权重/KV/激活的账）归 models 1.2 / 2.4，本章只交付硬件库存与口径。

本章交付清单（下游 recap 接口）：式 (1)(3) 带宽来源 + 命题 1 口径守卫（§3）、ECC SECDED 账（§4）、MC 映射与 camping 上界（§5）、容量三口径表（§6）。2.4 recap 式 (1) 作 HBM 带宽微架构来源、对账 inference 1.3 datasheet 表（增量 = 层级总表 + 测量方法 + roofline）；2.2 recap MC/通道库存作 cache miss 路径的终点；4.1 recap interleaving/sector 机制作 coalescing 判定的硬件侧；11.3 recap ECC 保护作故障分类。1.1 §4 世代对比表的 HBM 行数值以本章 SKU 表为来源。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| HBM 带宽 = 栈数 × 每栈通道数 × pin rate × 每通道位宽 ÷ 8；H100 = 5 × 16 × 5.238 Gbps × 64/8 = 3352 GB/s | 式 (1)(2) | §3、练习 1 |
| 1024 bit/stack = $n_{\mathrm{ch}} w_{\mathrm{ch}}$；等价每栈口径式 (3)；1024 代入含 $n_{\mathrm{ch}}$ 式高估 16×（53.6 TB/s） | 命题 1 + 坑注 | §3、练习 2 |
| H100 $f_{\mathrm{pin}}$ = 2619 MHz DDR × 2 = 5.238 Gbps（披露）；H200 6.25 / HGX B200 7.52 / GB200 7.81 Gbps 均反推 | §3 SKU 表 | §3、练习 2 |
| HBM3 每栈 16 ch × 64 bit + 32 pseudo-channel；HBM2e 为 8 ch × 128 bit；HBM3e 是 vendor speed-bin 非独立 JEDEC 编号 | §2 结构表 | §2 |
| vendor 每栈带宽互证：6.4 Gbps → 819.2 / 9.2 → 1177.6 / 9.6 → 1228.8 GB/s，同式闭环 | §2 / §3 实例 | §2、练习 1 |
| H100 HBM3/2e 采用 SECDED + Sideband ECC（wp p.38 逐字）：校验位独立区域，数据带宽不受影响（3352 不打折）；L2/L1/register file 同受 SECDED | §4 披露事实 | §4 |
| SECDED 校验位下界 $2^{r} \ge d+r+1$；教学实例化 d=64 → r=7+1=8 位，存储开销 12.5%；Inline 对照口径 64/72 ≈ 88.9% | 命题 2 | §4、练习 3 |
| H100 SXM5 10 个 512-bit MC（每栈 2 个）；interleaving 让顺序流铺满 80 通道 | §5 库存 | §5 |
| camping 最坏：全落单通道 → 41.9 GB/s = 峰值 1/80（简化线性模型，教学口径） | 命题 3 | §5、练习 4 |
| 容量三口径：raw 栈几何（192 = 8 × 24）≠ usable（HGX 180 / GB200 186）≠ nvidia-smi 179 GiB ≈ 192.2 GB | §6 容量表 | §6、练习 4 |
| H200 = 6 栈 144 raw / 141 usable；H100 raw = usable = 80 GB；差额为出货预留（用途未披露） | §6 | §6 |

## 练习

1.（计算 · [10]）H100 SXM5 主口径带宽重算：(a) 由「2619 MHz DDR」换算 $f_{\mathrm{pin}}$（Gbps）；(b) 代入式 (1) 求 $\mathrm{BW}$（GB/s）；(c) 用等价每栈口径式 (3) 重算，核对两式同值。校验：(b)(c) 与 whitepaper Table 3 的 3352 GB/sec、datasheet 3.35 TB/s 一致（容差 0.1%）；总线宽 5 × 16 × 64 与 Table 3「5120-bit」逐字一致。
2.（计算 · [20]）误代守卫与 SKU 反推：(a) 把 1024 代入式 (1) 的 $w_{\mathrm{ch}}$，算出多少 GB/s、是真值的几倍？(b) 由 H200 的 4.8 TB/s 与 6 栈反推 $f_{\mathrm{pin}}$；(c) 同法反推 HGX B200（7.7 TB/s）与 GB200（8 TB/s）的 $f_{\mathrm{pin}}$，并与 vendor 最大 speed-bin 9.2 / 9.6 Gbps 比较。先预测 (a) 的倍数再算。校验：(a) 16×、≈53.6 TB/s，与 §3 坑注一致；(b) 6.25 Gbps；(c) ≈7.52 / 7.81 Gbps，均低于 vendor 最大 bin，与 §3 SKU 表一致。
3.（推导 · [30]）补全命题 2 的证明（SECDED 校验位下界）：(a) 计数论证—— $r$ 个校验位的 $2^{r}$ 个 syndrome 须区分「无错」+ $d+r$ 个单 bit 错位置，得 $2^{r} \ge d+r+1$；(b) 代入 $d = 64$，逐试 $r = 6, 7$ 求最小 $r$；(c) 说明 SECDED 加一位总偶校验后共 8 位，存储开销 12.5%；(d) 算 Inline ECC 对照口径的有效数据带宽占比（教学实例化 d=64），并说明 H100 实际的 Sideband 口径为什么没有这项折扣。校验：(b) $r=6$ 时 $64 \ge 71$ 不成立、 $r=7$ 时 $128 \ge 72$ 成立；(d) $64/72 \approx 0.889$（88.9%），Sideband 口径数据带宽不变，与 §4 账一致。
4.（计算 · [30]）camping 与容量综合账：(a) 简化模型（ $g = 256$ B）下 stride 20 KB 的访问全落单通道，求有效带宽与占峰值比例；(b) 把 nvidia-smi 的 179 GiB 换算成 GB（十进制），判断它接近 raw 192 还是 usable 180；(c) 算 H200 与 HGX B200 的 raw-usable 差额。校验：(a) 41.9 GB/s = 1/80，与 §5 命题 3 实例一致；(b) ≈192.2 GB ≈ raw（arXiv:2610.41919 实测口径）；(c) 3 GB 与 12 GB，与 §6 容量表一致。

## 参考

- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25 dam-cdn PDF 一手核实——Table 3（5 HBM3 stacks / 5120-bit HBM3 / Memory Data Rate 2619 MHz DDR / Memory Bandwidth 3352 GB/sec / 80 GB / 10 个 512-bit MC）、Fig 6（full GH100 6 栈 / 12 控制器）、「6 HBM3 or HBM2e stacks」逐字、ECC Memory Resiliency 节（p.38）：SECDED 码族逐字 + Sideband ECC vs Inline ECC 逐字 + L2/L1/register file SECDED 清单 + Memory Row Remapping
- [NVIDIA, datasheet](../references.md)：NVIDIA H100 / H200 / HGX H100-H200 / Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM 80 GB / 3.35 TB/s、H200 SXM 141 GB / 4.8 TB/s、HGX B200 180 GB / 7.7 TB/s、GB200 186 GB / 8 TB/s、H100 NVL 94 GB / 3.9 TB/s 勿混
- [JEDEC, HBM3](../references.md)：JEDEC HBM3 规范 JESD238A + 成员官方互证（key: jedec-hbm3），[C] 官方规格披露·一手权威，2026-11-25——Micron「16 independent channels and 32 pseudo channels」+「1024 IO pins」+「24GB 8-high HBM3E is shipping with NVIDIA H200」、Samsung HBM3 6.4 Gbps → 819 GB/s / HBM3E 9.2 Gbps → 1180 GB/s / 12H 8H、SK hynix HBM3E 9.6 Gbps → 1.23 TB/s；HBM3e = vendor speed-bin 非独立 JEDEC 编号；JEDEC 原文 403 bot 拦截，精确 URL 留 W4
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——B200「Up to 192 GB HBM3e」= raw 栈容量（8 × 24 GB），非出货 usable
- [NVIDIA, Blackwell tuning guide](../references.md)：NVIDIA Blackwell Tuning Guide（key: blackwell-tuning-guide），[C] 官方规格披露·一手权威，2026-11-25——B200 HBM3/HBM3e up to 180 GB（usable 口径互证）
- [Zand et al., 2026](../references.md)：Matching Workloads to Blackwell: Power-Resolved Benchmarking of NVIDIA B200 and B300 across HPC and AI Workloads（key: b200-power-bench），arXiv:2610.41919，[B] 有 paper 无官方代码，2026-11-25——nvidia-smi HBM3e 179 GiB / 269 GiB ≈ raw 192 / 288 GB（十进制）vs datasheet usable 180 / 270 GB（§6 三口径实测注）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——KB = 1024 逐字注（§1 单位纪律）
