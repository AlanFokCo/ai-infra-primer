# 7.1 NVLink / NVSwitch 架构（NVLink / NVSwitch Architecture）

> 头条 900 GB/s 是双向口径——18 links × 25 GB/s 单向怎么乘出 450 再翻倍到 900？NVLink、NVSwitch、NVLink-C2C、NV-HBI 四条互连各连什么对象，带宽为什么差一个量级？本章交付五块：NVLink 五代几何与带宽乘法链（1.0 P100 → 5.0 B200，方向口径硬规则）、NVSwitch 非阻塞 crossbar 与端口账（Hopper 4 chips × 3.2 TB/s 双向 / Blackwell NVLink Switch 9 trays × 144 ports）、NVLink 域聚合带宽恒等式（7.2 / 14.4 / 130 TB/s，命题 1）、三层互连分层（NV-HBI die-to-die / NVLink-C2C CPU-GPU / NVLink GPU-GPU）、NVLS 网内聚合流量账与 vs PCIe 7×/14× 对账 | 前置：1.1（世代表与 NV-HBI/C2C 区分坑注）、2.4（层级带宽表 NVLink/PCIe 行与方向口径）；下游 7.2（P2P 带宽账与节点拓扑）、7.3（GPUDirect P2P）、8.4/10.2（NVLS 走读与 IB SHARP 对照）、11.1（服务器形态） | 锚点：Pascal / Volta / Ampere / Hopper / Blackwell 五份 whitepaper（[NVIDIA, Pascal whitepaper](../references.md)、[NVIDIA, Volta whitepaper](../references.md)、[NVIDIA, Ampere whitepaper](../references.md)、[NVIDIA, Hopper whitepaper](../references.md)、[NVIDIA, Blackwell brief](../references.md)，[C] 官方规格披露·一手权威）、NVLink/NVSwitch blog（[NVIDIA, NVLink/NVSwitch blog](../references.md)，[C] 开放技术博客）、GB200 blog（[NVIDIA, GB200 blog](../references.md)，[C] 开放技术博客）、NVLink/NVSwitch 产品页（[NVIDIA, NVLink/NVSwitch](../references.md)，[C]）、NVLink-C2C 产品页（[NVIDIA, NVLink-C2C](../references.md)，[C]）、GB200 NVL72 规格（[NVIDIA, GB200 NVL72](../references.md)，[C]）、datasheet（[NVIDIA, datasheet](../references.md)，[C]）、Hopper blog（[NVIDIA, Hopper blog](../references.md)，[C]）、NCCL docs（[NVIDIA, NCCL docs](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 三个问题与本章口径

H100 SXM 的 NVLink 头条带宽是 900 GB/s，可 whitepaper 的乘法链是 18 links × 25 GB/s——乘积是 450，不是 900，差的那个因子 2 从哪来？B200 把头条推到 1.8 TB/s，NVSwitch 从「每节点 4 颗芯片」变成「每机架 9 个 switch tray」，Grace CPU 与 GPU 之间又有一条 900 GB/s 的 NVLink-C2C——这些名字里都带 NVLink，它们是同一条互连吗？本章回答三个问题：每代带宽怎么从 link 几何乘出来、方向口径怎么锁？NVSwitch 凭什么把「每对分链路」变成「任意对满带宽」？NVLink / NVSwitch / NVLink-C2C / NV-HBI 各连什么对象、各在哪一层？

**本章口径**（章脊锁定；本章为世代对比章，多 SKU 口径在此一次声明，表外不再换）：主口径 **H100 SXM**（NVLink4，18 links × 25 GB/s 单向）与 **HGX H100 8-GPU 域**（4 颗第三代 NVSwitch）；对比口径 P100 / V100 / A100（NVLink 1.0/2.0/3.0）、B200（NVLink5）、GB200 NVL72（72-GPU 域）。**方向口径硬规则**（spec §4.3）：官方头条（口径：头条 = 官方产品页/datasheet 印刷的总带宽数）900 GB/s（H100）/ 1.8 TB/s（B200）均为**双向总和**；per-link 25 / 50 GB/s 均为**单向**；per-GPU 单向 = 头条 ÷ 2（450 / 900 GB/s）。本章所有带宽表与推导先注方向、再代数字；凡未注方向的披露值（NV-HBI 10 TB/s）按原值引、不做方向推定。

符号声明（spec §4.3）： $n_{\mathrm{link}}$ = 每 GPU link 数； $n_{\mathrm{lane}}$ = 每 link 每方向差分对数； $r_{\mathrm{lane}}$ = 每 lane 信令速率（Gbps）； $n_{\mathrm{port}}$ = 每 NVSwitch 芯片端口数； $n_{\mathrm{GPU}}$ = 域内 GPU 数； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s）；下标 uni / bidir 标单向 / 双向。 $n$ 专归 rank 数（7.2/8.x 章首声明），本章 GPU 数一律 $n_{\mathrm{GPU}}$； $N$ 专归参数量，本章不占用。单位口径：GB/s = $10^{9}$ B/s、TB/s = $10^{12}$ B/s（十进制，datasheet 口径）；Tbit/s ÷ 8 = TB/s（bit 转 byte）。

**边界声明**：P2P 带宽/延迟账与 nvbandwidth 测量归 7.2；GPUDirect P2P API 与 PCIe ACS/IOMMU 限制归 7.3；NVLS 代码走读归 8.4，IB SHARP 与 fabric 工程归 10.2；服务器形态与供电散热归 11.1。本章只交付互连硬件几何与带宽账。代码 scope：机制片段级（无代码走读）。

## 2. NVLink 五代：带宽是乘出来的

为什么 V100 → A100 → H100 三代 per-link 带宽停在 25 GB/s 不动，per-GPU 带宽却从 300 涨到 600 再到 900 GB/s？所谓 NVLink，是 NVIDIA 的专有 GPU 串行互连：每条 link 含两个方向各一组独立差分对（全双工），带宽由乘法链决定。乘法链有三个因子——link 数 × 每 link lane 数 × 每 lane 信令速率，三代走了不同的路：A100 起 lane 数减半、信令翻倍，两者相消维持 per-link 25 GB/s；per-GPU 的涨幅全来自 link 数（6 → 12 → 18）。推导路线分三步：先给 per-link 信令几何式（式 (1)），再给 per-GPU 聚合式（式 (2)），最后五代逐项代入并与 whitepaper 逐字对账。关键一步是**信令速率是 raw 口径、有效带宽是官方披露值**，两者之间隔着编码开销（V100 的 25.78 → 25 即此）。

per-link 单向带宽的信令几何（raw 口径）：

$$
\mathrm{BW}_{\mathrm{link}}^{\mathrm{raw}} = \frac{n_{\mathrm{lane}} \cdot r_{\mathrm{lane}}}{8} \qquad (1)
$$

**符号**： $n_{\mathrm{lane}}$ = 每 link 每方向差分对（lane）数； $r_{\mathrm{lane}}$ = 每 lane 信令速率（Gbps）；除以 8 是 bit → byte。全数字代入（H100，Hopper wp OSFP 节「8-channels of 100G PAM4 signaling」口径）： $2 \times 100 / 8 = 25$ GB/s——与官方 per-link 有效值 25 GB/s 恰好相等；B200： $2 \times 200 / 8 = 50$ GB/s，亦与披露值相等。V100 是唯一有落差的代： $8 \times 25.78 / 8 = 25.78$ GB/s raw，官方有效值 25 GB/s（Volta wp 逐字「Each link now provides 25 Gigabytes/second in each direction」），编码开销约 3%。PAM4 是四电平脉冲幅度调制（每符号 2 bit），100 / 200 Gbps 已是 bit 速率口径，不再乘 2。

per-GPU 聚合与方向口径（本章核心乘法链）：

$$
\mathrm{BW}_{\mathrm{NVLink}} = n_{\mathrm{link}} \cdot \mathrm{BW}_{\mathrm{link}}, \qquad \mathrm{BW}_{\mathrm{bidir}} = 2 \cdot \mathrm{BW}_{\mathrm{NVLink}} \qquad (2)
$$

**符号**： $`\mathrm{BW}_{\mathrm{NVLink}}`$ = 每 GPU NVLink **单向**带宽（spec §4.3 符号）； $`\mathrm{BW}_{\mathrm{link}}`$ = per-link **单向有效**带宽（式 (1) 的披露值，非 raw）； $`\mathrm{BW}_{\mathrm{bidir}}`$ = 每 GPU **双向总和**，即官方头条口径。因子 2 的来源：每条 link 的两个方向各有一条独立差分对组，可同时满载——双向总和 = 单向 × 2，不是营销取整。全数字代入（H100）： $18 \times 25 = 450$ GB/s 单向， $450 \times 2 = 900$ GB/s 双向——Hopper wp p.47 逐字「H100 includes 18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth」，头条 900 即双向总和。B200： $18 \times 50 = 900$ GB/s 单向， $900 \times 2 = 1800$ GB/s = 1.8 TB/s 双向——Blackwell brief p.10 逐字「1.8 TB/sec total bandwidth, 900 GB/sec in each direction」，「900 GB/sec in each direction」正是单向 900 的逐字披露。

五代全表（全部一手 whitepaper，2026-11-25 核实；per-link / per-GPU 列均为单向，头条列为双向）：

| 代际 | GPU（出货年） | links/GPU | lanes/方向/link | 信令/lane | per-link 单向 | per-GPU 单向 | per-GPU 双向（头条） | 一手出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVLink 1.0 | P100（2016） | 4 | 8 | 20 Gbps | 20 GB/s | 80 GB/s | 160 GB/s | Pascal wp Fig 25 |
| NVLink 2.0 | V100（2017） | 6 | 8 | 25.78 Gbps | 25 GB/s | 150 GB/s | 300 GB/s | Volta wp |
| NVLink 3.0 | A100（2020） | 12 | 4 | 50 Gbps | 25 GB/s | 300 GB/s | 600 GB/s | Ampere wp |
| NVLink 4.0 | H100（2022） | 18 | 2 | 100 Gbps PAM4 | 25 GB/s | 450 GB/s | 900 GB/s | Hopper wp p.47 |
| NVLink 5.0 | B200（2024） | 18 | 2 | 200 Gbps PAM4 | 50 GB/s | 900 GB/s | 1.8 TB/s | Blackwell brief p.10 |

逐代读表。**信令速率近似逐代翻倍**（20 → 25.78 → 50 → 100 → 200 Gbps/lane），**lane 数从 A100 起减半**（8 → 8 → 4 → 2 → 2），两者相消使 per-link 单向在 V100/A100/H100 三代维持 25 GB/s，B200 信令再翻倍而 lane 数不减，per-link 跳到 50 GB/s。**link 数是 per-GPU 涨幅的主因子**：4 → 6 → 12 → 18 → 18。两个官方比值可直接对账：NVLink4 vs NVLink3 = 900/600 = 1.5×（Hopper wp 逐字「1.5x the communications bandwidth」）；NVLink5 vs NVLink4 = 1800/900 = 2×（brief 逐字「doubles the performance」）。P100 的 160 GB/s 双向对 PCIe Gen3 x16（32 GB/s 双向）是 5×（Pascal wp 逐字「5x the bandwidth of PCIe Gen3 x16」）——vs PCIe 的完整对账见 §7。

> **坑**：把头条 900 GB/s 当单向带宽代入。触发条件：拿 datasheet 头条值直接作 α-β 模型的带宽项、或代 $`n_{\mathrm{link}} \cdot \mathrm{BW}_{\mathrm{link}}`$ 时把 25 当双向；现象：通信时间低估 2 倍（带宽高估 2 倍），7.2 的 P2P 账与 8.x 的集合时间账全错；规避：头条 ÷ 2 = 450 GB/s 单向，per-link 25 GB/s 已是单向（spec §4.3 方向口径；recap 2.4 §4 坑注）。

口径注（出货年）：NVLink 1.0 于 2016 年随 P100 出货（Volta wp 逐字「first introduced in 2016 with the Tesla P100」）；GB200 blog 称「introduced in 2014」是宣布口径——本书一律取出货年，两口径不混。

## 3. NVSwitch：从「每对分链路」到「任意对满带宽」

没有 NVSwitch 的 8-GPU 直连全互连，每 GPU 的 900 GB/s 双向要均分给 7 个 peer，每对只剩 900/7 ≈ 128 GB/s；有 NVSwitch，任意单对可达 900 GB/s——7 倍差（blog Table 1 口径，逐笔账在 7.2 §3）。交换芯片凭什么做到？答案在 crossbar 的非阻塞性：链路不再按 peer 切死，而是全部汇入交换矩阵，任意输入端口到任意输出端口可线速接通。

所谓 NVSwitch，是指带 NVLink 端口的 crossbar 交换芯片： $n_{\mathrm{port}}$ 个端口经内部交叉矩阵互连，任意一组互不冲突的输入-输出端口配对（每端口至多出现一次）可同时线速传输（非阻塞，non-blocking）。正例：HGX H100 的 8 GPU 任意排列通信（每 GPU 总注入 900 GB/s 双向不变），NVSwitch 逐对线速接通。边界反例：若 crossbar 内部存在共享链路（阻塞式），某些端口配对会争用内部带宽、峰值随通信对数下降——非阻塞定义恰好排除这种情形。blog 逐字：「every NVIDIA Hopper GPU in a server can communicate at 900 GB/s with any other NVIDIA Hopper GPU simultaneously. The peak rate does not depend on the number of GPUs that are communicating. That is, the NVSwitch is non-blocking.」（[NVIDIA, NVLink/NVSwitch blog](../references.md)，[C]）。口径注：「任意对 900 GB/s」指**单对峰值**可达每 GPU 满注入带宽，且每 GPU 总注入不随通信对数缩水；不是「每一对同时各跑 900」——那要求每 GPU 注入 7 × 900，超出物理带宽（7.2 命题 1 给精确上界）。（口径：注入带宽 = 该 GPU 向互连发送与接收的带宽之和，即双向总和；下文「注入」同此。）

Hopper 第三代 NVSwitch 的披露几何：每 chip **64 个 NVLink4 端口**（Hopper wp 逐字「64 ports of fourth-generation NVLink links」）；每个 HGX H100/H200 8-GPU 系统 **4 颗**第三代 NVSwitch（blog 逐字「four third-generation NVSwitch chips」）；每 chip 双向总带宽 **25.6 Tbit/s**（blog 逐字）。端口账推导路线：每端口就是一条 NVLink4 的交换侧端点，双向 = 2 × 25 GB/s，乘端口数即得芯片双向容量——关键一步是**端口带宽与 GPU 侧 per-link 同口径**（同一条 link 的两端）。

$$
\mathrm{BW}_{\mathrm{chip,bidir}} = n_{\mathrm{port}} \cdot 2 \cdot \mathrm{BW}_{\mathrm{link}} = 64 \times 2 \times 25 = 3200 \ \mathrm{GB/s} = 3.2 \ \mathrm{TB/s} \qquad (3)
$$

**符号**： $n_{\mathrm{port}} = 64$（每 chip 端口数）； $`\mathrm{BW}_{\mathrm{link}} = 25`$ GB/s 单向（NVLink4）。对账：blog 25.6 Tbit/s ÷ 8 = 3.2 TB/s，与式 (3) 逐数闭环。口径注：Hopper wp 印「Total switch throughput increases to 13.6 Tbits/sec」是 **switch throughput 口径**（含信令开销），blog 25.6 Tbit/s 是**双向有效带宽口径**（= 3.2 TB/s）——本书用 blog 口径并注 wp 原文，两数不混（13.6 Tbit/s 含编码开销的单向信令口径，与 3.2 TB/s 有效双向不是同一笔账）。

系统级端口账（聚合口径，可判定）：4 chips × 3.2 TB/s = **12.8 TB/s 双向**交换侧端口容量；GPU 侧 8 × 18 = **144 条 link**，每条 link 占一个交换侧端口，故 144 端口被占用（4 × 64 = 256 总端口，平均每 chip 36 个）。每 GPU 的 18 links 在 4 颗 switch 间的逐颗分布属闭源细节、官方未披露，本书只做聚合账；任意对可经交换矩阵线速接通是 blog 的非阻塞披露口径（7.2 §2 拓扑图），逐颗分布本书不推定。

Blackwell 的交换芯片官方名是 **NVLink Switch**（NVLink 产品页规格表作 NVLink 5 Switch），即第四代 NVSwitch 芯片——「NVSwitch 4.0」不是官方命名（spec 大纲订正注）。NVL72 的披露几何以 **tray** 为单位：每 switch tray **144 个 NVLink 端口 @ 100 GB/s 双向**（GB200 blog 逐字「Each NVLink switch tray delivers 144 NVLink ports at 100 GB so the nine switches fully connect each of the 18 NVLink ports on every one of the 72 Blackwell GPUs」），9 trays。端口账：每 tray 144 × 100 GB/s = 14.4 TB/s 双向；9 × 144 = **1296 端口 = 72 GPU × 18 links**，恰好全连接；聚合 9 × 14.4 = 129.6 ≈ **130 TB/s**（brief Table 2 / datasheet / 产品页一致）。每 tray 含几颗芯片的拆分本书来源未逐字核到一手，只用 tray 级数值推导，不做 per-chip 推定。

```
GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 │     │     │     │     │     │     │     │
 │18   │18   │18   │18   │18   │18   │18   │18    ← 每 GPU 18 links（25 GB/s 单向/link）
 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
┌────────────────────────────────────────────────┐
│ NVSwitch0   NVSwitch1   NVSwitch2   NVSwitch3  │ ← 4 颗第三代 NVSwitch
│ (64 ports)  (64 ports)  (64 ports)  (64 ports) │   每 chip 3.2 TB/s 双向
│ 非阻塞 crossbar：任意输入端口 ↔ 任意输出端口线速 │
└────────────────────────────────────────────────┘
8×18 = 144 links → 占 144 交换端口（总 256，平均 36/chip）
单对路径：GPU_A → NVSwitch_i crossbar → GPU_B（单对峰值 900 GB/s 双向）
```

历史注（口径，非叙事）：Hopper wp 宣布过 256-GPU / 32 节点的 NVLink Switch System，但逐字「NVLink Switch System technology is not available with H100 systems now」——宣布未出货，勿当 H100 规格；GB200 NVL72 才是首个出货的多节点 NVLink 域产品（§4）。

## 4. NVLink 域与聚合带宽恒等式

「域聚合带宽 7.2 TB/s」这个数到底在数什么——是「任意两卡同时跑 7.2 TB/s」吗？不是。它数的是域内全部 GPU 注入带宽之和；任意单对仍受每 GPU 注入上限约束。先把「域」定义钉死，再给恒等式与证明。

所谓 NVLink 域（NVLink domain），是指经 NVLink/NVSwitch 可直接互访 HBM 的一组 GPU（P2P 语义，API 与地址映射归 7.3），域内通信不经 PCIe/NIC。正例：HGX H100 的 8 GPU 经 4 颗 NVSwitch 全互连，是一个 8-GPU 域；GB200 NVL72 的 72 GPU 跨 18 个 compute tray 经 9 个 switch tray 全互连，是一个 72-GPU 域（跨 tray 仍在域内）。边界反例：两台 HGX 服务器各 8 GPU，跨机流量经 NIC 走 RDMA——即使同机架也不是同一个 NVLink 域（机间 fabric 归 9.x/10.x）。域内「全互连」指逻辑全连接（任意对可直通），物理接线是星型（每 GPU 的 links 全汇 NVSwitch）+ crossbar，不是每对专用链路的直连全互连（§3 的 128 vs 900 GB/s 差即由此）。

域形态与聚合带宽表（聚合 = 双向口径，推导见命题 1）：

| 域 | GPU 数 | per-GPU 双向 | 域聚合（双向） | 交换侧 | 出处 |
| --- | --- | --- | --- | --- | --- |
| HGX H100/H200 8-GPU | 8 | 900 GB/s | 7.2 TB/s | 4 颗第三代 NVSwitch | 产品页 + blog |
| HGX B200 8-GPU | 8 | 1.8 TB/s | 14.4 TB/s | NVLink Switch | Blackwell datasheet + brief |
| GB200 NVL72 | 72 | 1.8 TB/s | 130 TB/s（129.6） | 9 switch trays × 144 ports | brief Table 2 + datasheet + blog |
| NVLink5 最大域 | 576 | 1.8 TB/s | 大于 1 PB/s（1036.8 TB/s） | 多机架 NVLink Switch System | brief + blog |

**命题 1**（域聚合带宽恒等式）。设域内 $n_{\mathrm{GPU}}$ 个 GPU 各带 $n_{\mathrm{link}}$ 条 NVLink，per-link 单向有效带宽 $`\mathrm{BW}_{\mathrm{link}}`$，则域聚合带宽（双向口径，定义为各 GPU 双向注入带宽之和）满足

$$
\mathrm{BW}_{\mathrm{domain}} = n_{\mathrm{GPU}} \cdot \mathrm{BW}_{\mathrm{bidir}} = 2 \thinspace n_{\mathrm{GPU}} \thinspace n_{\mathrm{link}} \thinspace \mathrm{BW}_{\mathrm{link}} \qquad (4)
$$

**证明**（路线：per-GPU 式 (2) → 对 $n_{\mathrm{GPU}}$ 求和 → 数链路端点验证无重无漏，三步）：其一，每 GPU 双向注入 = $`2 n_{\mathrm{link}} \mathrm{BW}_{\mathrm{link}}`$（式 (2)，每 link 两方向独立满载）。其二，域聚合按定义是各 GPU 注入之和，同构 GPU 直接乘 $n_{\mathrm{GPU}}$ 得式 (4)。其三，计数验证：HGX/NVL72 接线中每条物理 link 恰有一个 GPU 端点（另一端是 NVSwitch 端口），故按 GPU 求和时每条 link 恰好计一次，无重无漏；交换侧端口容量（式 (3) × chip 数）是另一笔账，约束的是交换矩阵而非 GPU 注入。∎

全数字代入（式 (4)，与官方披露逐行对账）：HGX H100： $2 \times 8 \times 18 \times 25 = 7200$ GB/s = **7.2 TB/s**——NVLink 产品页 NVLink4 Switch 行「Total Aggregate Bandwidth 7.2 TB/s」逐字吻合；HGX B200： $8 \times 1.8 = 14.4$ TB/s——Blackwell datasheet「Total NVLink Bandwidth 14.4 TB/s」吻合；NVL72： $72 \times 1.8 = 129.6 \approx 130$ TB/s——brief Table 2「130 TB/s Bidirectional」吻合（129.6 到 130 是官方取整）；576 GPU 最大域： $576 \times 1.8 = 1036.8$ TB/s ≈ 1.04 PB/s——brief/blog「scales up to 576 GPUs」域聚合 >1 PB/s 吻合。

> **提点**：聚合带宽与「任意对同时满速」是两笔账。聚合 7.2 TB/s 把每个字节在发送端计一次、接收端计一次（双向口径）；它不承诺任意 pair 同时各跑满——每 GPU 注入上限 900 GB/s 双向仍在，单对峰值与总注入的精确关系是 7.2 命题 1。跨域对分带宽（bisection）又是另一口径，归 10.1。

GB200 NVL72 的构成（brief Table 2 + datasheet + partition guide + blog 四源互证）：**72 Blackwell GPU + 36 Grace CPU**；18 个 compute tray，每 tray 4 GPU + 2 Grace；9 个 NVLink switch tray；液冷机架，铜缆 cartridge 互连。GB200 superchip = 1 Grace + 2 Blackwell，CPU↔GPU 经 NVLink-C2C（§5），GPU↔GPU 经 NVLink5 入域。NVL36 = 36 GPU 单 rack；NVL72 = 72 GPU 单 rack（18 dual nodes）或双 rack（18 single nodes）。供电散热与机架功率密度归 11.1，本章只交付互连几何。

## 5. 三层互连：NV-HBI、NVLink-C2C、NVLink

B200 一颗芯片身上有三条「chip-to-chip」互连，带宽分别是 10 TB/s、900 GB/s、1.8 TB/s——差一个量级，连接对象完全不同。把它们混为一谈，带宽账与拓扑账会同时错（recap 1.1 §5 坑注）。本节按封装层级从内到外分层：die 间 → CPU-GPU 间 → GPU-GPU 间。

| 互连 | 连接对象 | 带宽 | 一致性语义 | 一手出处 |
| --- | --- | --- | --- | --- |
| NV-HBI | B200 封装内两个 reticle-limit die | 10 TB/s（方向未披露，按原值引） | 单一致芯片（one fully coherent chip） | Blackwell brief p.7-8，recap 1.1 |
| NVLink-C2C | Grace CPU ↔ GPU（GH200：Grace+H100；GB200：Grace+2 Blackwell） | 900 GB/s 双向 | 一致统一内存 | Hopper blog（GH）/ GB200 blog（GB）逐字 |
| NVLink5 | GPU ↔ GPU（域内） | 1.8 TB/s 双向 | P2P 直访（7.3） | Blackwell brief |
| PCIe Gen5 x16（对照） | GPU ↔ host CPU | 128 GB/s 双向 | 非一致 DMA | Hopper wp p.49 |

所谓 NV-HBI（NVIDIA High-Bandwidth Interface），是指 B200 封装内两个 GPU die 之间的 die-to-die 互连：brief 逐字「The two dies are connected and unified with a single 10 terabyte-per-second (TB/s) chip-to-chip NVIDIA High-Bandwidth Interface (NV-HBI), providing one fully coherent chip」。10 TB/s 高于 HGX B200 的 HBM 7.7 TB/s，设计意图是让跨 die 访问不先于本地 HBM 成为瓶颈（recap 1.1 §5 对比账）；方向口径 brief 未注，本书按原值引、不做方向推定。

所谓 NVLink-C2C（chip-to-chip），是指 Grace CPU 与 GPU 之间的 CPU-GPU 一致互连。两个产品的披露值都是 900 GB/s：Grace-Hopper（GH200）——Hopper blog 逐字「delivering 900 GB/s of total bandwidth, 7x faster than PCIe Gen5」；Grace-Blackwell（GB200）——GB200 blog 逐字「NVLink-Chip-to-Chip (C2C) interface that delivers 900 GB/s of bidirectional bandwidth. With NVLink-C2C, applications have coherent access to a unified memory space」。机制补充（NVLink-C2C 产品页）：支持 Arm AMBA CHI / CXL 协议；先进封装形态（PCB/MCM/interposer/wafer-level）；对 PCIe Gen6 PHY 有 6× 能效、3.5× 面积效率。一致统一内存的含义：CPU 与 GPU 共享地址空间、硬件维护一致性，无需显式拷贝——对比 PCIe 的非一致 DMA 语义（统一内存的 page fault 迁移机制 recap 3.2，Grace-Hopper 硬件一致免 fault）。GB200 每 Grace 连 2 Blackwell 的 per-GPU 带宽拆分来源未逐字披露，本书只引接口级 900 GB/s 双向，不做拆分推定。

```
┌──────────────────────── B200 封装 ────────────────────────┐
│  ┌─────────┐   NV-HBI 10 TB/s    ┌─────────┐              │
│  │  die 0  │◄───────────────────►│  die 1  │ ← die-to-die │
│  └─────────┘  （方向未披露）      └─────────┘   单一致芯片  │
└──────────────────────────────────────────────────────────┘
       ▲                              ▲
       │ NVLink-C2C 900 GB/s 双向      │ NVLink5 1.8 TB/s 双向
       │ （CPU↔GPU，一致统一内存）      │ （GPU↔GPU，入 NVLink 域）
┌──────┴─────┐                 ┌──────┴─────┐
│ Grace CPU  │                 │ 另一颗 B200 │
└────────────┘                 └────────────┘
  GB200 superchip = 1 Grace + 2 B200（C2C）；GPU↔GPU 走 NVLink5
```

> **坑**：把 NV-HBI、NVLink-C2C、NVLink 当同一条互连。触发条件：见「B200 双 die 经 NVLink-C2C 连接」或「C2C 10 TB/s」式转述；现象：die-to-die 与 CPU-to-GPU 与 GPU-to-GPU 三层混为一谈，带宽差一个量级、拓扑账全错；规避：三层三对象——NV-HBI 10 TB/s 是封装内 die↔die（brief 逐字），NVLink-C2C 900 GB/s 双向是 CPU↔GPU（Hopper/GB200 blog 逐字），NVLink5 1.8 TB/s 双向是 GPU↔GPU（brief 逐字）。recap 1.1 §5 坑注，本章扩为三层全表。

## 6. NVLS：把归约搬进交换芯片

ring all-reduce 的数据要在 GPU 之间走 $2(n-1)$ 跳，每跳 SM 做 load/reduce/store——归约能不能不出交换芯片就完成？NVLS 就是这个方向的硬件答案：NVSwitch 内做 multicast 与归约，集合通信从「GPU 算 + 链路搬」变成「交换芯片算 + 链路搬一次」。

所谓 NVLS（NVLink SHARP），是指 NVSwitch 的网内聚合能力：multicast（一对多复制）+ in-network reduction（网内归约），把集合操作卸载到交换芯片硬件。SHARP 全称 Scalable Hierarchical Aggregation and Reduction Protocol，IB 交换机版本归 10.2；NVLS 是 NVLink 域版本，两者层级不同、机制同族。Hopper wp 逐字：「hardware acceleration of collective operations with multicast and NVIDIA SHARP in-network reductions. Accelerated collectives include write broadcast (all_gather), reduce_scatter, and broadcast atomics. In-fabric multicast and reductions provide up to 2x throughput gain while significantly reducing latency for small block size collectives」，并「significantly reduces the load on SMs」。可用性：第三代 NVSwitch（NVLink4）+ Hopper 及以后 GPU（NCCL docs NCCL_NVLS_ENABLE 节逐字「NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures」，[NVIDIA, NCCL docs](../references.md)，[C]）。Blackwell NVLink Switch 新增 SHARP FP8 支持，brief 逐字「4X bandwidth efficiency」。

流量账（**教学推导口径**，机制级；精确协议流程与 kernel 走读归 8.4）：ring all-reduce 每 rank 发送 $2(n-1)/n \cdot M$（recap training 4.1 式 (2)， $d \to n$ 翻译，不重推；此处 $n$ = rank 数、每 GPU 一个 rank， $M$ = 消息总字节——域 GPU 数本章其余处记 $n_{\mathrm{GPU}}$）；NVLS 的机制级图景是每 GPU 把自有 $M$ 字节写入 multicast 组（发送 $M$），NVSwitch 在交换侧完成归约、把全和 $M$ 字节回送每 GPU（接收 $M$）——每 rank 发送 $M$，对比 ring 的 $2(n-1)/n \cdot M$，比值 $2(n-1)/n$： $n = 8$ 时 1.75， $n \to \infty$ 时趋于 2。这正是 wp「up to 2x throughput gain」的流量侧来源（口径：「up to」是上界披露，非保证值；增益以 GPU 注入带宽为瓶颈口径，交换侧 multicast 复制流量由非阻塞端口容量承担，§3 式 (3)）。延迟侧：ring 走 $2(n-1)$ 跳，NVLS 发送-归约-回送约 2 跳，小消息延迟显著下降（wp 逐字）。SM 侧：归约计算从 SM 移进交换芯片，通信 kernel 的 SM 占用下降（NCCL 协议与 kernel 走读归 8.4）。

全数字实例（ $n = 8$， $M = 1$ GB，单向有效带宽取 370.8 GB/s——7.2 §3 的 NCCL 模型口径，教学推导）：ring 每 rank 发送 $2 \times 7/8 \times 1 = 1.75$ GB，带宽项 $1.75 \times 10^{9} / (370.8 \times 10^{9}) \approx 4.72$ ms；NVLS 每 rank 发送 1 GB，带宽项 $10^{9} / (370.8 \times 10^{9}) \approx 2.70$ ms——比值 1.75，与 $2(n-1)/n$ 闭合。NVLS 算法选择与 NVLS_TREE 多节点变体归 8.4，NVLS vs IB SHARP 与 CollNet 集成归 10.2。

## 7. vs PCIe：7× 与 14× 怎么算

Hopper wp 逐字「7x the bandwidth of PCIe Gen 5」，Blackwell brief 逐字「over 14X the bandwidth of PCIe Gen5」——这两个比值是同方向口径相除的结果，不是头条对头条的随意比。先钉 PCIe 的数，再代比值。

PCIe Gen5 x16 = **128 GB/s 双向 = 64 GB/s 单向**（Hopper wp p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」；H100/Blackwell datasheet Interconnect 行「PCIe Gen5: 128GB/s」互证；PCI-SIG 规范 URL 403 残留，数值口径经 NVIDIA 一手闭合，[PCI-SIG, PCIe](../references.md)，[C]）。同方向口径相除：

$$
\frac{\mathrm{BW}_{\mathrm{NVLink4,bidir}}}{\mathrm{BW}_{\mathrm{PCIe5,bidir}}} = \frac{900}{128} \approx 7.03, \qquad \frac{\mathrm{BW}_{\mathrm{NVLink5,bidir}}}{\mathrm{BW}_{\mathrm{PCIe5,bidir}}} = \frac{1800}{128} \approx 14.06 \qquad (5)
$$

单向口径比值相同：450/64 ≈ 7.03、900/64 ≈ 14.06——同向相除时因子 2 消去，所以「7×/14×」对单向、双向口径都成立；但跨方向混比（900 双向 ÷ 64 单向）会再差 2 倍，不是官方比值的口径。全数字对账：7.03 ≈ wp「7x」，14.06 ≈ brief「over 14X」，闭合。更早的代际：NVLink1 对 PCIe Gen3 x16（32 GB/s 双向）= 160/32 = 5×（Pascal wp 逐字）；NVLink-C2C 对 PCIe Gen5 = 900/128 ≈ 7×（Hopper blog 逐字「7x faster than PCIe Gen5」）。

带宽差之外还有功能分工：PCIe 是 GPU↔host CPU 的通道，NIC/NVMe 也挂在 PCIe 上（9.2 GDR 的瓶颈账即 PCIe vs NIC 线速）；NVLink 是 GPU↔GPU 域内互连；NVLink-C2C 是 CPU↔GPU 一致互连。三者不是替代关系而是分层协作——PCIe 拓扑与 NUMA affinity 归 7.2 §5，P2P 语义归 7.3。延迟对比官方未披露精确值，测量方法归 7.2 §3/§6。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 方向口径硬规则：头条 900 GB/s（H100）/1.8 TB/s（B200）= 双向总和；per-link 25/50 GB/s = 单向；per-GPU 单向 = 头条 ÷ 2 | 式 (2)，spec §4.3 | §1、§2、练习 2 |
| per-link 信令几何 = lane 数 × 信令速率 ÷ 8（raw 口径）；H100 2×100/8 = 25、B200 2×200/8 = 50；V100 25.78 raw → 25 有效（编码开销约 3%） | 式 (1) | §2、练习 1 |
| 五代乘法链：信令 20→25.78→50→100→200 Gbps/lane，lane 8→8→4→2→2，link 4→6→12→18→18；NVLink4 = 1.5× NVLink3、NVLink5 = 2× NVLink4（wp/brief 逐字） | §2 五代表 | §2、练习 1 |
| NVSwitch 非阻塞 crossbar：Hopper 第三代 64 ports/chip，HGX 8-GPU = 4 chips，每 chip 3.2 TB/s 双向（25.6 Tbit/s ÷ 8）；wp 13.6 Tbit/s 是 throughput 口径勿混 | 式 (3) | §3、练习 3 |
| Blackwell 官方名 NVLink Switch（NVLink 5 Switch，第四代 NVSwitch 芯片），「NVSwitch 4.0」非官方；NVL72 9 trays × 144 ports @ 100 GB/s 双向 = 1296 = 72×18，聚合 130 TB/s | §3 端口账 | §3、练习 3 |
| 域聚合恒等式：BW_domain = n_GPU × per-GPU 双向 = 2 n_GPU n_link BW_link；7.2 / 14.4 / 129.6≈130 TB/s 与产品页/datasheet/brief 逐行吻合；576 GPU 域 >1 PB/s | 式 (4)，命题 1 | §4、练习 3 |
| 三层互连勿混：NV-HBI 10 TB/s（die↔die，方向未披露）/ NVLink-C2C 900 GB/s 双向（CPU↔GPU 一致）/ NVLink5 1.8 TB/s 双向（GPU↔GPU）；C2C 支持 CHI/CXL，6× 能效 vs PCIe Gen6 PHY | §5 三层表 + 坑注 | §5、练习 4 |
| NVLS = NVSwitch 网内 multicast + 归约；每 rank 发送 M vs ring 2(n-1)/n·M，比值 →2 即「up to 2x」流量侧来源；可用 = 第三代 NVSwitch + Hopper+；Blackwell SHARP FP8 4× | §6 流量账（教学推导） | §6、8.4/10.2 |
| vs PCIe Gen5 x16（64 单向/128 双向）：NVLink4 = 7×、NVLink5 = 14×（同向相除）；NVLink1 vs Gen3 = 5×；C2C = 7× | 式 (5) | §7、练习 4 |
| Hopper 256-GPU NVLink Switch System 宣布未出货（wp 逐字 not available now）；GB200 NVL72 首个出货多节点 NVLink 域 | §3 历史注 | §3、§4 |

## 练习

1.（计算 · [10]）五代乘法链代入（式 (1)(2)）：(a) V100 每 GPU 单向与双向带宽；(b) A100 每 GPU 单向与双向带宽；(c) 用式 (1) 算 A100 per-link raw 带宽（4 lanes × 50 Gbps），核对官方有效值。校验：(a) 150 / 300 GB/s，与 Volta wp「300 Gigabytes/second」逐字一致；(b) 300 / 600 GB/s，与 Ampere wp「600 GB/sec total bandwidth」一致；(c) 25 GB/s，raw = 有效（无编码落差）。
2.（计算 · [20]）方向口径与域聚合（式 (2)(4)）：(a) H100 18 × 25 得单向多少、双向多少；(b) HGX H100 / HGX B200 / NVL72 域聚合各多少 TB/s；(c) 若把头条 900 GB/s 误当单向代入式 (4)，HGX H100 域聚合会高估几倍、得多少 TB/s？先预测 (c) 的方向再算。校验：(a) 450 单向 / 900 双向；(b) 7.2 / 14.4 / 129.6≈130 TB/s，与产品页「Total Aggregate Bandwidth 7.2 TB/s」、Blackwell datasheet 14.4、brief Table 2 130 一致；(c) 高估 2 倍得 14.4 TB/s——恰为 HGX B200 的真值，方向口径错会把 H100 算成 B200（§2 坑注）。
3.（推导 · [30]）证明命题 1（式 (4) 域聚合恒等式）并做 NVSwitch 端口账：(a) 写出三步证明（per-GPU 式 (2) → 求和 → 每条 link 恰一个 GPU 端点的计数论证），说明为什么交换侧端口容量是另一笔账；(b) 用式 (3) 算第三代 NVSwitch 每 chip 双向带宽，核对 blog 25.6 Tbit/s；(c) 算 HGX H100 系统交换侧总端口容量与 GPU 侧 link 总数、占用端口数与平均每 chip 占用数。校验：(b) 3.2 TB/s = 25.6 Tbit/s ÷ 8，与 blog 逐字一致；(c) 12.8 TB/s 双向容量、144 links、144 端口占用（总 256、平均 36/chip），与 §3 聚合账逐数一致；(a) 计数论证须覆盖「无重无漏」两侧。
4.（计算 · [30]）三层互连排序与 PCIe 比值（式 (5)）：(a) 把 NV-HBI / NVLink-C2C / NVLink5 / PCIe Gen5 x16 按带宽从高到低排序（注方向口径，NV-HBI 按原值）；(b) 算 NVLink4 / NVLink5 对 PCIe Gen5 的双向比值，核对 wp「7x」/ brief「over 14X」；(c) GB200 superchip 内 Grace↔Blackwell 走哪条互连、Blackwell↔Blackwell 走哪条、B200 封装内双 die 走哪条？(d) NVLink-C2C 对 PCIe Gen5 的比值与 Hopper blog 逐字是否一致？校验：(a) 10 TB/s（NV-HBI，方向未披露）> 1.8 TB/s（NVLink5 双向）> 900 GB/s（C2C 双向）> 128 GB/s（PCIe 双向）；(b) 7.03 / 14.06；(c) C2C / NVLink5 / NV-HBI（§5 坑注）；(d) 900/128 ≈ 7.03 ≈ blog「7x faster than PCIe Gen5」。

## 参考

- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink4 节 p.47 逐字（18 links × 25 GB/s 单向 = 900 GB/s total、1.5× A100、7× PCIe Gen5）、第三代 NVSwitch 64 ports / 13.6 Tbits/sec throughput 口径、NVLink SHARP 逐字（write broadcast/reduce_scatter/broadcast atomics、up to 2x）、NVLink Switch System 256 GPU「not available with H100 systems now」、PCIe Gen5 节 p.49 逐字（128 total / 64 per-direction）、OSFP 100G PAM4（§2/§3/§6/§7）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink5 节 p.10 逐字（2 lanes × 50 GB/s 单向、18 links、1.8 TB/sec total / 900 GB/sec in each direction、2× Hopper、14× PCIe Gen5、576 GPUs、SHARP FP8 4X bandwidth efficiency）、NV-HBI p.7-8 逐字（10 TB/s die-to-die 单一致芯片）、Table 2 GB200 NVL72（72 GPU/36 Grace/18 compute trays/9 switch trays/130 TB/s 双向）、HGX B200 Aggregate NVLink 14.4 TB/s（§2/§3/§4/§5/§6/§7）
- [NVIDIA, Pascal whitepaper](../references.md)：NVIDIA Tesla P100 / GP100 Pascal Architecture Whitepaper（key: pascal-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink 1.0 Fig 25 逐字（4 links × 8 lanes × 20 Gbps、Total 80 GBps per direction、160 GB/s 双向、5× PCIe Gen3 x16）（§2/§7）
- [NVIDIA, Volta whitepaper](../references.md)：NVIDIA Tesla V100 / GV100 Volta Architecture Whitepaper（key: volta-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink 2.0 逐字（信令 20→25 Gbps、每 link 25 GB/s each direction、4→6 links、300 GB/s）、NVLink 首发 2016 P100 产品口径（§2）
- [NVIDIA, Ampere whitepaper](../references.md)：NVIDIA A100 / GA100 Ampere Architecture Whitepaper（key: ampere-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——NVLink 3.0 逐字（50 Gbit/sec per signal pair、4 lanes/方向、25 GB/s each direction、12 links、600 GB/s total、V100 25.78 Gbps 互证）（§2）
- [NVIDIA, NVLink/NVSwitch blog](../references.md)：NVIDIA NVLink and NVSwitch Supercharge Large Language Model Inference blog（key: nvlink-nvswitch-blog），[C] 开放技术博客、非闭源，2026-11-25——HGX H100/H200 8-GPU = four third-generation NVSwitch chips、每 chip 25.6 Tbit/s 双向、非阻塞逐字、P2P 对照 Table 1（900/7 ≈ 128 GB/s）（§3/§4）
- [NVIDIA, GB200 blog](../references.md)：NVIDIA GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference blog（key: gb200-blog），[C] 开放技术博客、非闭源，2026-11-25——NVLink switch tray 144 ports @ 100 GB 逐字、9 switches 全连接 72×18、NVLink-C2C 900 GB/s bidirectional 逐字、576 GPUs >1 PB/s、NVL72 构成、NVLink 2014 宣布口径注（§3/§4/§5）
- [NVIDIA, NVLink/NVSwitch](../references.md)：NVIDIA NVLink / NVLink Switch 产品页（key: nvlink-nvswitch），[C] 官方规格披露·一手权威，2026-11-25——NVLink4/5 规格列（900/1800 GB/s/GPU、18 links）、NVLink4 Switch 域 8 聚合 7.2 TB/s、NVLink5 Switch 域 8 或 72 聚合 130 TB/s；nvswitch/ 独立页 404 并入 nvlink/；本波只用 4/5 列（Rubin 列非 scope）（§2/§4）
- [NVIDIA, NVLink-C2C](../references.md)：NVIDIA NVLink-C2C 产品页（key: nvlink-c2c），[C] 官方规格披露·一手权威，2026-11-25——chip-to-chip 一致互连、Arm AMBA CHI / CXL、先进封装形态、6× 能效 / 3.5× 面积效率 vs PCIe Gen6 PHY；页面无带宽数值，900 GB/s 锚 Hopper blog / GB200 blog（§5）
- [NVIDIA, GB200 NVL72](../references.md)：NVIDIA GB200 NVL72 规格（key: gb200-nvl72），[C] 官方规格披露·一手权威，2026-11-25——72 GPU + 36 Grace、18 compute trays（4 GPU+2 Grace/tray）+ 9 switch trays、130 TB/s 双向、液冷铜缆；partition guide v1.2 互证（§4）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100 / H200 / HGX H100-H200 / Blackwell 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——Interconnect 行（H100 SXM NVLink 900 GB/s + PCIe Gen5 128 GB/s；Blackwell 全系 NVLink 1.8 TB/s）、Total NVLink Bandwidth 行（NVL72 130 / NVL4 7.2 / HGX B200 14.4 TB/s）、HGX H100/H200 NVSwitch 第三代（4-GPU N/A）（§4/§7）
- [NVIDIA, Hopper blog](../references.md)：NVIDIA Hopper Architecture In-Depth blog（key: hopper-blog），[C] 开放技术博客、非闭源，2026-11-25——NVLink-C2C Grace-Hopper 逐字「900 GB/s of total bandwidth, 7x faster than PCIe Gen5」（§5/§7）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2026-11-25（版本 2.32.3，滞后 repo 2.32.7 注）——NCCL_NVLS_ENABLE 节逐字「NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures」（§6）
- [PCI-SIG, PCIe](../references.md)：PCI-SIG PCIe 规范（key: pcie-spec），[C] 官方规格披露·一手权威，2026-11-25——Gen5 x16 = 64 GB/s 单向 / 128 GB/s 双向；URL 403 bot 拦截残留，数值口径经 Hopper wp p.49 逐字 + datasheet 互证闭合（§7）
