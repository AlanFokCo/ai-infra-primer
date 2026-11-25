# 11.1 GPU 服务器硬件形态（Server Hardware Forms and MIG）

> 一台 DGX H100 铭牌 10.2 kW，一个 GB200 NVL72 机架约 120 kW——功率差 12 倍而 GPU 数只差 9 倍，多出的功率去哪、怎么散？一颗 H100 又怎么物理切成 7 个互相隔离的「小 GPU」，每个分到多少 SM、显存、带宽与 L2？本章从 HGX 基板 / DGX 整机 / NVL72 机架三种形态走到 MIG 硬件分区与 vGPU/SR-IOV 虚拟化，交付服务器选型账（功率 / 散热 / NVLink 域 / PCIe 带宽）与 TCO 电力成分 | 前置：1.1（SKU 几何与 B200 SM 反推）、7.1/7.2（NVLink 域聚合与 PCIe/NUMA 账，recap 不重推）、2.1（raw vs usable）；下游 11.2（profiling）、11.3（故障与可靠性）、cluster（K8S device plugin 与调度，纯文字指针） | 锚点：MIG User Guide（[NVIDIA, MIG User Guide](../references.md)，[C] 官方规格披露·一手权威）、vGPU 17.0 User Guide（[NVIDIA, vGPU docs](../references.md)，[C]）、HGX/DGX 规格（[NVIDIA, HGX/DGX](../references.md)，[C]）、GB200 NVL72 规格（[NVIDIA, GB200 NVL72](../references.md)，[C]）、DGX GB 机架指南（[NVIDIA, DGX GB200 guide](../references.md)，[C]）、datasheet（[NVIDIA, datasheet](../references.md)，[C]）、Blackwell brief（[NVIDIA, Blackwell brief](../references.md)，[C]）、Inside Blackwell Ultra blog（[NVIDIA, Inside Blackwell blog](../references.md)，[C] 开放技术博客）、B200 微基准（[Jarmusch et al., Blackwell microbench](../references.md)，[B]，arXiv:2512.02189）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、nvidia-smi docs（[NVIDIA, nvidia-smi](../references.md)，[C]）——官方规格 2026-11-26 一手核实

## 1. 三个问题与本章口径

一台 DGX H100 的最大功率是 10.2 kW，一个 GB200 NVL72 机架约 120 kW——差约 12 倍，而 GPU 数只差 9 倍（8 对 72）。多出的功率去了哪里，120 kW 的机架为什么必须液冷？一颗 H100 有 132 个 SM，MIG 把它切成最多 7 个实例时，每个实例分到多少 SM、显存、带宽与 L2，切分比例为什么是 1/7 与 1/8 两套？本章回答三个问题：HGX / DGX / NVL72 三种形态的硬件边界在哪，供电散热的账怎么算？MIG 的分区资源账（切片几何、profile 表、实例上界）怎么推？vGPU / SR-IOV 怎么把硬件分区交付给虚拟机，与 cluster 平台编排的边界在哪？

**本章口径**（章脊锁定；本章为服务器形态对比章，多 SKU 口径在此一次声明，表外不再换）：

| 口径 | SKU 与几何 | 功率口径 | 来源 |
| --- | --- | --- | --- |
| 主口径 | DGX H100：8×H100 SXM5 80 GB（132 SM / 3.35 TB/s / L2 50 MB）+ 2×Xeon 8480C + 4×第三代 NVSwitch | ≤700 W/GPU（TDP）；整机 10.2 kW max / 8U | [NVIDIA, HGX/DGX](../references.md) + [NVIDIA, datasheet](../references.md) + [NVIDIA, Hopper whitepaper](../references.md)，[C] |
| 对比口径 | DGX B200：8×B200 180 GB（148 SM，[B] 口径 / 7.7 TB/s）+ 2×Xeon 8570 + 2×第五代 NVLink switch | ≤1000 W/GPU；整机 14.3 kW max / 10U | 同上三份，[C]；SM 数 [B] 见 §6 |
| 对比口径 | GB200 NVL72：72×GB200 每 GPU 186 GB / 8 TB/s + 36 Grace，18 compute tray + 9 switch tray | ≤1200 W/GPU（configurable）；机架 ≈120 kW | [NVIDIA, GB200 NVL72](../references.md) + [NVIDIA, DGX GB200 guide](../references.md) + [NVIDIA, datasheet](../references.md)，[C] |
| MIG 几何对比 | A100-40GB / H100-80GB / B200-180GB profile 全表；GB200-186GB 对比行；A30-24GB（4 切片例外） | —（MIG 不改变 TDP，只切分资源） | [NVIDIA, MIG User Guide](../references.md)，[C] |

单位与读数纪律：GB 按 $10^{9}$ B、TB/s 按 $10^{12}$ B/s（datasheet 十进制口径）；MIG profile 名里的 gb 是十进制 GB 的**舍入命名**（B200 每 memory slice 180/8 = 22.5 GB，profile 名印 23gb，§6）；功率一律 TDP / 铭牌 max 口径（口径：max = 供电散热设计上限，实际平均功率低于此，利用率账归 inference 9.1 §2）；NVLink 带宽方向口径沿 7.1 硬规则（头条 = 双向总和，per-link = 单向）。B200 SM 数 148 是 [B] 微基准口径——官方 datasheet 不印 SM 数（§6 口径注）。

符号声明（spec §4.3）： $n_{\mathrm{SM}}$ = 出货 SM 数； $n_{\mathrm{GPU}}$ = GPU 数（沿 7.1；裸 $n$ 专归 rank 数，本章不占用）； $s_{\mathrm{SM}}$ 、 $s_{\mathrm{mem}}$ = 一个 GI 包含的 SM 切片与 memory 切片数（本章局部）； $f_{\mathrm{SM}}$ 、 $f_{\mathrm{mem}}$ = 对应资源分数； $P_{\mathrm{rack}}$ = 机架（或机箱）铭牌最大功率； $P_{\mathrm{IT}}$ = IT 负载功率（TCO 账取铭牌 max）； $n_{\mathrm{shelf}}$ 、 $n_{\mathrm{PSU}}$ 、 $P_{\mathrm{PSU}}$ = power shelf 数、每 shelf PSU 数、单 PSU 容量； $\mathrm{PUE}$ = 电源使用效率（Power Usage Effectiveness）； $c_{\mathrm{elec}}$ = 电价（USD/kWh）； $h_{\mathrm{year}}$ = 年小时数； $C_{\mathrm{elec}}$ = 年电费（USD）； $\zeta_{\mathrm{air}}$ 、 $\zeta_{\mathrm{water}}$ = 空气 / 水的体积热容（J/(m³·K)）； $\Delta\theta$ = 冷却介质温升（K）； $Q_{\mathrm{air}}$ = 空气体积流量（m³/s）。 $N$ （参数量）、 $B$ （batch）、 $T$ （序列长）、 $M$ （消息字节）、 $\alpha/\beta$ （通信）、 $I/R$ （roofline）本章均不占用。

**边界声明**：NVLink 五代几何与带宽乘法链、域聚合恒等式 recap 7.1，P2P 每对上界与 NUMA min 账 recap 7.2，均不重推导；GPUDirect P2P API 与 ACS/IOMMU 限制归 7.3，GDR 归 9.2，机间 fabric 归 10.x；**K8S device plugin、GPU 调度策略与多机集群编排归 cluster 维度**（边界：MIG/vGPU/SR-IOV 的硬件分区与虚拟化机制归本章，平台编排归 cluster），本章只给纯文字指针、不展开；profiling 归 11.2，XID/ECC 故障归 11.3；完整 TCO 与 USD/M tokens 成本模型归 inference 9.1 §7（指引）。代码 scope：机制片段级（nvidia-smi 命令片段，无 repo 走读）。测量口径：本机无 NVIDIA GPU，本章数值全部为官方披露（[C]/[B] 标注）或显式标注的教学假设，不新跑 benchmark（spec §6.2 口径）。

## 2. 三种形态：基板、整机、机架

HGX H100 8-GPU baseboard 卖的到底是什么？不是一台能开机的服务器，而是一块焊好 8 颗 SXM GPU 与 NVSwitch 的 GPU 基板。所谓 HGX，是指 NVIDIA 的 GPU 基板（baseboard）产品：8 颗 SXM GPU 与 NVSwitch 在板上互连（HGX H100/B200 口径），交付给 OEM（Original Equipment Manufacturer，原始设备制造商）厂商作整机基底；所谓 DGX，是指 NVIDIA 自有品牌整机：基板 + CPU + 主机内存 + NIC + 存储 + PSU + 机箱，一台交付。正例：DGX H100 = HGX H100 基板 + 2×Xeon 8480C + 2 TB DIMM + 8×CX7（ConnectX-7）NIC + 6×3.3 kW PSU（Power Supply Unit，电源单元），8U 机箱。边界反例：只买 HGX 基板得不到可开机系统——基板上没有 CPU 与主机内存，host 侧（PCIe 拓扑、NIC 数量、机箱散热）是 OEM 的设计自由度，所以同是「HGX H100 8-GPU」，不同 OEM 整机的 PCIe 拓扑与 NIC 配置可以不同（§5）。

DGX 整机规格表（user guide 口径，[NVIDIA, HGX/DGX](../references.md)，[C]，2026-11-26 核实）：

| 项 | DGX H100 | DGX B200 |
| --- | --- | --- |
| GPU | 8×H100 SXM（640 GB HBM3） | 8×B200（1440 GB HBM3e） |
| CPU | 2×Intel Xeon 8480C（56 core，PCIe Gen5） | 2×Intel Xeon 8570（56 core，PCIe Gen5） |
| NVSwitch | 4×第三代（NVLink4，GPU-to-GPU 900 GB/s 双向） | 2×第五代 NVLink switch（聚合 14.4 TB/s 双向） |
| NIC | 8×CX7 单口 400G + 2×CX7 双口以太网 | 8×CX7 单口 400G + 2×BF3 双口 |
| 主机内存 | 2 TB（32 DIMM） | 2 TB（可扩 4 TB） |
| 机箱 | 8U / 130.45 kg max | 10U / 142.4 kg max |
| 供电 | 10.2 kW max / 6×3.3 kW PSU 4+2 冗余 / 200-240 VAC | 14.3 kW max / 6×3.3 kW PSU 5+1 冗余 |
| 散热 | 风冷，前进后出 1105 CFM（立方英尺/分钟）@80% fan PWM | 风冷，1550 CFM |

逐行读表。显存 640 = 8×80、1440 = 8×180 GB——B200 列是 **HGX B200 usable 180 GB** 口径，不是 raw 栈容量 192 GB（8×24 GB，recap 2.1；三口径勿混沿 1.1 §4 SKU 注）。NVSwitch 从 4 颗变 2 颗不是缩水：NVLink5 每 chip 72 端口 × 100 GB/s 双向 = 7.2 TB/s，2 颗 = 14.4 TB/s，恰好等于 8 GPU × 1.8 TB/s 的注入聚合（recap 7.1 式 (4)，不重推）；GPU 侧 8×18 = 144 条 link 与交换侧 2×72 = 144 端口一一对应（§3 端口账）。Hopper 代 4 颗 × 64 端口 = 256 端口、占用 144（recap 7.1 §3）。NIC 行是 rail-optimized 的硬件基底：8 张 400G NIC 对 8 颗 GPU，每 GPU 一张同 rail NIC（拓扑账归 10.1，GDR 瓶颈账归 9.2）。

GB200 NVL72 把形态从机箱推到机架（[NVIDIA, DGX GB200 guide](../references.md)，[C]，hardware 页逐字）：18×1RU compute tray（每 tray 2 Grace CPU + 4 Blackwell GPU）+ 9×1RU NVLink switch tray + 2×TOR（Top-of-Rack）管理交换机 + 8 个 power shelf（电源架），NVLink 经无源铜缆 cartridge 背板互连。72 GPU + 36 Grace 构成一个 72-GPU NVLink 域（域聚合 130 TB/s 双向 = 72×1.8 TB/s，recap 7.1 §4，不重推）。GB200 superchip = 1 Grace + 2 Blackwell，CPU↔GPU 走 NVLink-C2C 900 GB/s 双向一致互连（recap 7.1 §5）——CPU-GPU 数据路径不再经 PCIe（§5）。compute tray 对外：4×CX7 单口 400G OSFP + 2×BF3（BlueField-3 DPU）双口 400G（GB200 口径；GB300 换 4×CX8 800G），4×3.84 TB E1.S NVMe 软件 RAID0。

```
形态层级：基板 → 整机 → 机架超节点
┌────────────────────────────────────────────────────────────┐
│ HGX 基板（交付 OEM）                                         │
│   8×SXM GPU + NVSwitch（H100: 4 颗 / B200: 2 颗）            │
│   无 CPU / 主机内存——host 侧是 OEM 设计自由度                 │
└────────────────────────────────────────────────────────────┘
       ▼ 加 CPU / 内存 / NIC / 存储 / PSU / 机箱
┌────────────────────────────────────────────────────────────┐
│ DGX 整机（NVIDIA 自有品牌）                                   │
│   DGX H100: 8U / 10.2 kW max / 风冷 1105 CFM                │
│   DGX B200: 10U / 14.3 kW max                               │
│   NVLink 域 = 8 GPU（机箱内）                                 │
└────────────────────────────────────────────────────────────┘
       ▼ NVLink5 switch tray 把域扩到机架级
┌────────────────────────────────────────────────────────────┐
│ GB200 NVL72 机架（≈120 kW，液冷）                             │
│   18 compute trays（2 Grace + 4 GPU/tray）= 72 GPU + 36 Grace│
│   9 switch trays（2×72-port chip/tray）+ 8 power shelves     │
│   NVLink 域 = 72 GPU（跨 tray 仍在域内，铜缆背板）             │
└────────────────────────────────────────────────────────────┘
```

## 3. NVLink 域：8 卡机箱与 72 卡机架的硬件边界

8 卡机箱的 NVLink 域聚合带宽是 7.2 TB/s（H100）/ 14.4 TB/s（B200），NVL72 是 130 TB/s——三个数都是 7.1 式 (4) 域聚合恒等式（各 GPU 双向注入之和）的直接代入，本章 recap 不重推。形态选择改变的是**域的边界**：所谓 NVLink 域（recap 7.1 §4 定义），是指经 NVLink/NVSwitch 可直接互访 HBM 的一组 GPU，域内通信不经 PCIe/NIC；跨机箱的两颗 GPU 即使同机架也不是同一个域，流量下 NIC 走 RDMA（机间 fabric 归 9.x/10.x）。

| 形态 | 域内 GPU 数 | per-GPU 双向 | 域聚合（双向） | 交换侧 |
| --- | --- | --- | --- | --- |
| DGX H100（HGX H100） | 8 | 900 GB/s | 7.2 TB/s | 4×第三代 NVSwitch（64 port/chip） |
| DGX B200（HGX B200） | 8 | 1.8 TB/s | 14.4 TB/s | 2×第五代 NVLink switch（72 port/chip） |
| GB200 NVL72 | 72 | 1.8 TB/s | 130 TB/s（129.6） | 9 switch trays × 2 chip × 72 port |
| 跨机箱（对照） | 每机箱 8 | — | 节点出口 8×50 = 400 GB/s 单向 | 8×CX7 400G NIC（recap 7.2 §7） |

NVL72 switch tray 的端口账（[NVIDIA, DGX GB200 guide](../references.md)，[C]，逐字「2x NVLink NVSwitches, each of which has 72 NVLink ports」）：每 chip 72 端口 × 100 GB/s 双向 = 7.2 TB/s，即 guide 披露的 57.6 Tbps full duplex ÷ 8；每 tray 2 chip = 14.4 TB/s，9 trays = 129.6 ≈ 130 TB/s，与 brief Table 2 / datasheet 逐数闭合（recap 7.1 §3）。互证：DGX B200 user guide 逐字「2 x 5th generation NVLink switches that provide 14.4 TB/s aggregate bandwidth」= 2×7.2 TB/s，单 chip 7.2 TB/s 口径两处闭合。

> **坑**：把 57.6 Tbps 当整个 switch tray 的带宽。触发条件：guide prose「57.6 Tbps of full duplex bandwidth … in a 1U design」被读成 tray 级总数；现象：57.6 Tbps ÷ 8 = 7.2 TB/s 只是**单 chip** 72 端口的双向容量，当 tray 级用会把域聚合算成 9×7.2 = 64.8 TB/s，低估一半；规避：tray = 2 chip = 115.2 Tbps = 14.4 TB/s，对账 brief/datasheet 的 130 TB/s 聚合（9×14.4 = 129.6）再取数。

域边界决定并行放置的硬件上限（放置规则的账 recap 7.2 §7，不重推）：域内每 GPU 单向 450 GB/s（H100）对 NIC 线速 50 GB/s 是 9 倍（B200 是 18 倍），所以高频集合（TP 的逐层 all-reduce）要留在域内。TP 度不超过 8 时单机箱 8 卡域够用；TP/EP 超过 8 时要么上 NVL72（72-GPU 域，all-reduce 仍走 NVLink 不下 NIC），要么跨机 RDMA 承受一个量级的带宽落差（9.x/10.x）。并行策略本身的账归 training 5.x / inference 3.3，本章只交付硬件边界。两个 SKU 边界注（recap 7.2 §2）：4-GPU HGX H100/H200 无 NVSwitch（datasheet NVSwitch 行 4-GPU = N/A），GPU 间直连 P2P 按均分账；H100 PCIe 版 NVLink 600 GB/s 双向（bridge）勿混 SXM 版 900。

## 4. 供电与散热：从 TDP 到机架的三笔账

8 颗 H100 的 TDP 乘积是 8×700 = 5.6 kW，可 DGX H100 铭牌是 10.2 kW——多出的 4.6 kW 去了哪里？答案是 GPU 之外的整机部件：CPU、主机内存、NIC、NVSwitch、PSU 转换损耗与风扇。供电散热要分三笔账算：GPU-only TDP 账、整机/机架铭牌账、散热介质账。

TDP 表（[NVIDIA, datasheet](../references.md)，[C]，2026-11-26 重取；TDP = Thermal Design Power，热设计功耗）：H100 SXM ≤700 W；HGX B200 ≤1000 W；GB200 ≤1200 W（configurable）；Blackwell Ultra 一档 GB300 ≤1400 W / HGX B300 1100 W（Ultra datasheet，本章只列不展开）。

GPU-only 对铭牌账（铭牌 = 整机/机架最大功率，[NVIDIA, HGX/DGX](../references.md) + [NVIDIA, DGX GB200 guide](../references.md)，[C]）：

| 形态 | GPU-only TDP 乘积 | 铭牌 max | GPU 占比 | 每 GPU 系统级功率 | 每 U 功率 |
| --- | --- | --- | --- | --- | --- |
| DGX H100 | 8×700 = 5.6 kW | 10.2 kW | 54.9% | 1.275 kW | 1.275 kW/U（8U） |
| DGX B200 | 8×1000 = 8 kW | 14.3 kW | 55.9% | 1.7875 kW | 1.43 kW/U（10U） |
| GB200 NVL72 | 72×1200 = 86.4 kW | ≈120 kW（机架级） | 72% | 1.667 kW | —（机架口径） |

口径注（两个口径勿混）：120 kW 是 guide 逐字「The rack power consumption is approximately 120kW」的**机架级约值**，含 36 Grace、18 颗交换芯片、PSU 损耗与配电；86.4 kW 是 72×1200 W 的 **GPU-only** 乘积——两个口径差 33.6 kW，不是矛盾。NVL72 一个机架的功率约等于 11.8 台 DGX H100（120/10.2），但 GPU 数是 9 倍：机架形态把供电散热基础设施摊到更多 GPU 上，每 GPU 系统级功率 1.667 kW 反而低于 DGX B200 的 1.7875 kW。

供电冗余是可判定的计数账。NVL72 用 8 个 power shelf，每 shelf 6×5.5 kW 风冷 PSU（guide 逐字「six air-cooled 5.5kW PSUs in eight power shelves that provide N+N redundancy and the required input power of 33kW per power shelf」）；AC 经 power shelf 转 nominal 50-51V DC，再经 busbar（母线）配电到机架部件。

**命题 1**（N+N 供电冗余不等式）。N+N 冗余下可用功率等于装机容量之半，且须不小于机架最大功率：

$$
P_{\mathrm{usable}} = \frac{1}{2} \thinspace n_{\mathrm{shelf}} \thinspace n_{\mathrm{PSU}} \thinspace P_{\mathrm{PSU}} \ge P_{\mathrm{rack}} \qquad (1)
$$

**符号**： $n_{\mathrm{shelf}}$ = power shelf 数； $n_{\mathrm{PSU}}$ = 每 shelf PSU 数； $P_{\mathrm{PSU}}$ = 单 PSU 容量； $P_{\mathrm{rack}}$ = 机架最大功率。**证明**（路线：N+N 定义 → 计数 → 代入，三步）：其一，N+N 指两组各 N 个模块互为备份，任一组须独立承担全负载，故可用功率 = 装机总量 ÷ 2；其二，装机总量 = shelf 数 × 每 shelf PSU 数 × 单 PSU 容量；其三，代入 NVL72： $0.5 \times 8 \times 6 \times 5.5 = 132$ kW ≥ 120 kW，成立；每 shelf 输入 6×5.5 = 33 kW 与 guide 逐字闭合。∎

DGX 的 4+2 / 5+1 是同一笔账的直接计数（非对半）：DGX H100 承重 4×3.3 = 13.2 kW ≥ 10.2 kW；DGX B200 承重 5×3.3 = 16.5 kW ≥ 14.3 kW。一个容易漏读的约束：DGX H100 user guide 逐字「The system only boots if at least three PSUs are operational」——3×3.3 = 9.9 kW 低于 10.2 kW 满载，所以「≥3 PSU 可 boot」是降级启动阈值，不是满载阈值，满载须 4 颗承重。

散热介质账：风冷还是液冷，由冷却介质的体积热容决定。空气体积热容 $\zeta_{\mathrm{air}} \approx 1.2 \times 10^{3}$ J/(m³·K)（密度约 1.2 kg/m³ × 比热约 $1.0 \times 10^{3}$ J/(kg·K)，教学常数），水 $\zeta_{\mathrm{water}} \approx 4.18 \times 10^{6}$ J/(m³·K)——相差约 3483 倍。带走铭牌功率 $P_{\mathrm{rack}}$ 、温升 $\Delta\theta$ 所需体积流量：

$$
Q_{\mathrm{air}} = \frac{P_{\mathrm{rack}}}{\zeta_{\mathrm{air}} \cdot \Delta\theta} \qquad (2)
$$

**符号**： $Q_{\mathrm{air}}$ = 空气体积流量（m³/s）； $\zeta_{\mathrm{air}}$ = 空气体积热容； $\Delta\theta$ = 进出风温升（K；教学假设取 15 K）。全数字代入：DGX H100 10.2 kW 需 $10200/(1200 \times 15) = 0.567$ m³/s ≈ 1200 CFM——与 user guide 披露的 1105 CFM @80% fan PWM（≈0.52 m³/s）同量级，反推对应温升约 16.3 K（教学常数口径），官方风量与物理账闭合；DGX B200 披露 1550 CFM（≈0.73 m³/s），14.3 kW 反推温升同样约 16.3 K——两代风冷机箱是同一温升设计点。NVL72 120 kW 同温升需 $120000/(1200 \times 15) = 6.67$ m³/s ≈ 14100 CFM，是 DGX H100 的 12.8 倍——100 kW 级机架靠风冷在工程量上不可行，所以液冷是物理必然而非营销选项。换水介质： $120000/(4.18 \times 10^{6} \times 15) = 1.91 \times 10^{-3}$ m³/s ≈ 1.9 L/s，流量需求降约 3500 倍。guide 披露的落地形态：compute tray 的 CPU/GPU 经 cold plate（冷板）+ manifold（歧管）液冷，网络与存储部件仍风冷，配 leak detection（漏液检测）。

> **坑**：拿 GPU-only TDP 乘积做供电散热设计。触发条件：把 8×700 = 5.6 kW 当 DGX H100 的供电需求、或 72×1200 = 86.4 kW 当 NVL72 机架需求；现象：分别低估 45%（对 10.2 kW）与 28%（对 120 kW），PSU 容量、线缆与冷量全部配错；规避：设计一律取整机/机架铭牌口径（10.2 / 14.3 / ≈120 kW），GPU-only 乘积只用于算 GPU 占比。

## 5. PCIe 拓扑与 CPU-GPU affinity

Gen5 x16 理论 64 GB/s 单向，可跨 socket 的 H2D（host-to-device，host 内存到 GPU 显存）拷贝可能只拿到 40 GB/s（教学假设例，recap 7.2 §5）——瓶颈不在 PCIe 本身，在路径上多出的那一跳。本节把 7.2 的 min 账放到整机拓扑上：GPU 挂在哪个 socket 的 root complex（PCIe 树的主机侧根节点，衔接 CPU/内存控制器与 PCIe 设备，recap 7.2 §4）下，决定 H2D 与 GDR 的路径长度。

DGX H100 的 2 颗 Xeon 8480C 构成两个 NUMA node（Non-Uniform Memory Access，非一致内存访问，recap 7.2 §5）：每 socket 挂本地 DRAM，8 颗 GPU 与 8 张 NIC 分挂两侧。逐 PCIe switch 的扇出官方未披露，下图为**教学示意**（非官方接线）：

```
DGX H100 PCIe/NUMA 拓扑（教学示意：每 socket 4 GPU + 4 NIC）
┌──────────────────────────┐  inter-socket link  ┌──────────────────────────┐
│ CPU0 (Xeon 8480C)        │◄───────────────────►│ CPU1 (Xeon 8480C)        │
│ NUMA node 0 + 本地 DRAM  │                     │ NUMA node 1 + 本地 DRAM  │
│ root complex             │                     │ root complex             │
└────────────┬─────────────┘                     └────────────┬─────────────┘
             │ PCIe Gen5                                      │ PCIe Gen5
     ┌───────┴────────┐                               ┌───────┴────────┐
     │ PCIe switch    │                               │ PCIe switch    │
     └─┬───┬───┬───┬──┘                               └─┬───┬───┬───┬──┘
       │   │   │   │                                   │   │   │   │
     GPU0 GPU1 GPU2 GPU3                             GPU4 GPU5 GPU6 GPU7
     NIC0 NIC1 NIC2 NIC3                             NIC4 NIC5 NIC6 NIC7
     （CX7 400G，与 GPU rail 对齐）                    （CX7 400G）
GPU↔GPU 走 NVLink/NVSwitch（§3），不走 PCIe 树；
H2D 路径：host DRAM → 内存控制器 → root complex → PCIe x16 → GPU
```

affinity 三条规则（机制 recap 7.2 §5，本节交付整机侧读法）。其一，host buffer 绑 GPU 本地 NUMA node：H2D 带宽取路径各段最小值 $`\min(\mathrm{BW}_{\mathrm{PCIe}}, \mathrm{BW}_{\mathrm{inter\text{-}socket}})`$（recap 7.2 式 (4)，不重推），跨 socket 教学假设例 $\min(64, 40) = 40$ GB/s，只剩本地的 62.5%；绑定用 `numactl --membind`，对比测量用 nvbandwidth（方法归 7.2 §6）。其二，GPU 与 NIC 同 PCIe switch/root complex 时 GDR 路径最优（瓶颈账 $`\min(\mathrm{BW}_{\mathrm{NIC}}, \mathrm{BW}_{\mathrm{PCIe}})`$ 归 9.2）：CX7 单口 400G = 50 GB/s 低于 Gen5 x16 的 64 GB/s，NIC 线速限；rail 对齐把每 GPU 配一张同侧 NIC（10.1）。其三，拓扑查询用 `nvidia-smi topo -m`（[NVIDIA, nvidia-smi](../references.md)，[C]）：输出矩阵逐对标注 GPU 之间与 GPU-NIC 之间的路径类型（NVLink / 同 PCIe switch / 经 root complex / 跨 socket），是核对 affinity 的第一手工具。PCIe switch 做 lane 扇出，同 switch 下 GPU/NIC 可不经 root complex 直通；P2P 受 ACS/IOMMU 限制的细节归 7.3。

GB200 的增量：Grace↔Blackwell 走 NVLink-C2C 900 GB/s 双向一致互连（recap 7.1 §5），对 PCIe Gen5 的 128 GB/s 双向是 7 倍（recap 7.1 式 (5)）——CPU-GPU 统一内存一致访问，H2D 不再受 PCIe 与 NUMA 约束；PCIe 仍是 NIC/NVMe 的通道（每 compute tray 4×CX7 + 2×BF3）。所以「PCIe 带宽」这条选型约束在 x86 + HGX 形态上是硬账，在 GB200 形态上退到 NIC/存储侧。

## 6. MIG 硬件分区：切片几何与资源账

一颗 H100 怎么物理切成 7 个互相隔离的「小 GPU」，每个小 GPU 的带宽为什么不会被邻居抢走？所谓 MIG（Multi-Instance GPU，多实例 GPU），是指 Ampere 起（compute capability 8.0+）GPU 的硬件分区机制：把 SM、显存控制器、L2、引擎按固定切片物理划分，实例间性能与错误隔离（[NVIDIA, MIG User Guide](../references.md)，[C]，2026-11-04 更新版一手）。支持 SKU：A100/H100/H200/B200/GB200/B300/GB300 最多 7 实例；**A30 = 4 例外**（4 切片设计），RTX PRO 6000 Blackwell 同为 4（supported-gpus 表）。

切片的几何由两个分数钉死（concepts 页逐字）。所谓显存切片（GPU memory slice），是指 GPU 显存资源的最小等分单元，**含对应的内存控制器与 cache**：「roughly one eighth of the total GPU memory resources, including both capacity and bandwidth」——每片 1/8，容量与带宽一起切。所谓 SM 切片（GPU SM slice），是指 SM 的最小等分单元：「roughly one seventh of the total number of SMs available in the GPU when configured in MIG mode」——每片 1/7。口径注：roughly = 分数口径；132/7 = 18.86 不是整数，官方按分数披露、未逐字给出每片整数 SM 数，本书按分数记账、不推定整数分配。L2 随 **memory slice** 方向切分（每片 1/8），不是随 SM slice——具体 profile 的 L2 列以官方表为准（下文坑注）。

分区两级：所谓 GPU 实例（GPU Instance，GI），是指 GPU 切片加引擎（DMA/NVDEC 等）的组合，guide 逐字「A GPU instance provides memory QoS」——GI 是内存保护与带宽 QoS（Quality of Service，服务质量）的边界；所谓计算实例（Compute Instance，CI），是指 GI 内 SM 切片的子集，「The CIs share memory and engines」——CI 之间共享所属 GI 的显存与引擎，但各有专属 SM。正例：A100-40GB 的 1g.5gb = 1 memory slice（5 GB）+ 1 SM slice；4g.20gb = 4+4，其 CI 可再分为 2c.4g.20gb / 3c.4g.20gb（分别占 2 或 3 个 SM slice，共享 20 GB）。边界反例：MPS（CUDA Multi-Process Service）不是物理分区——SM 按百分比逻辑共享、调度硬件仍共享、无错误隔离（concepts 页 Table 3），把 MPS 的「48 客户端」当 MIG 的隔离实例用，一个客户端的错误会波及全体。

| 机制 | 分区类型 | 最大分区数 | SM 性能隔离 | 内存保护 | 带宽 QoS | 错误隔离 | 重配时机 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| streams | 单进程逻辑 | 无限 | 无 | 无 | 无 | 无 | 动态 |
| MPS | 逻辑 | 48 | 百分比（非分区） | 有 | 无 | 无 | 进程启动 |
| MIG | 物理 | 7 | 有 | 有 | 有 | 有 | 空闲时 |

（Table 3 逐字口径；MIG 跨分区互操作为 limited IPC，即受限的进程间通信。）

**命题 2**（MIG 实例数上界与空闲账）。每 GI 至少占 1 个 SM 切片与 1 个 memory 切片，切片互不重叠，故同时存在的 GI 数不超过两类切片数的较小者：7 SM 切片、8 memory 切片的 GPU 上界为 $\min(7, 8) = 7$；A30 两类各 4 切片，上界 $\min(4,4) = 4$。且 7×1g 组合用满 SM（7/7）而显存只用 7/8——第 8 个 memory slice 必然空闲。**证明**（路线：计数上界 → 构造达界 → 空闲账，三步）：其一，每 GI 至少消耗 1 SM 切片 + 1 memory 切片，切片物理互斥，故 GI 数 ≤ min(两类切片数)；其二，官方 profile 表 1g 行「Number of Instances = 7」给出达界构造（7 个 1g GI 各占 1+1 切片，placement 存在）；其三，7×1g 用 7 个 memory slice，剩 8−7 = 1 片 = 1/8 显存空闲（H100 为 10 GB），SM 侧 7/7 用满。∎ 推论（分区守恒）：任意合法组合的 SM 分数之和 ≤ 1、显存分数之和 ≤ 1——切片互斥 + 总数计数直接得；这也是「placement 不足报 Insufficient Resources」的算术来源。

切片分区示意（官方 concepts 图重绘，[NVIDIA, MIG User Guide](../references.md)，[C]；H100 7×1g.10gb placement）：

```
显存侧：8 个 memory slice，每片 1/8（容量 10 GB + 带宽 1/8 + MC + L2 1/8）
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ mem0 │ mem1 │ mem2 │ mem3 │ mem4 │ mem5 │ mem6 │ mem7 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
SM 侧：7 个 SM slice，每片 1/7（roughly ≈18.9 SM）
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ SM sl0 │ SM sl1 │ SM sl2 │ SM sl3 │ SM sl4 │ SM sl5 │ SM sl6 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┘
GI_k = mem_k + SM sl_k（k = 0..6）→ 7 实例：SM 7/7 用满、显存 7/8
mem7 = 1/8 显存（10 GB）空闲——SM 切片耗尽，第 8 个 GI 建不出（命题 2）
```

GI 的资源分数账（本章核心公式）：

$$
f_{\mathrm{SM}} = \frac{s_{\mathrm{SM}}}{7}, \qquad f_{\mathrm{mem}} = \frac{s_{\mathrm{mem}}}{8} \qquad (3)
$$

**符号**： $s_{\mathrm{SM}}$ 、 $s_{\mathrm{mem}}$ = GI 包含的 SM / memory 切片数； $f_{\mathrm{SM}}$ = SM 分数（乘 $n_{\mathrm{SM}}$ 得 SM 数）； $f_{\mathrm{mem}}$ = 显存容量、显存带宽与 L2 的通用分数（L2 逐 profile 以官方表为准）。全数字代入（H100 主口径，132 SM / 80 GB / 3.35 TB/s / L2 50 MB，[NVIDIA, Hopper whitepaper](../references.md) + [NVIDIA, datasheet](../references.md)，[C]）：1g.10gb 得 SM 132/7 ≈ 18.9（roughly 口径）、容量 80/8 = 10 GB、带宽 3350/8 = 418.75 GB/s、L2 50/8 = 6.25 MB——带宽随容量同比例切，这就是 memory QoS 的物理来源：邻居抢不走你的 1/8 带宽。3g.40gb 得 SM 3/7×132 ≈ 56.6、容量 40 GB、带宽 1675 GB/s、L2 25 MB。B200 对比口径（148 SM [B] / 180 GB / 7.7 TB/s）：1g.23gb 得 SM 148/7 ≈ 21.1、容量 180/8 = 22.5 GB（profile 名 23gb 是舍入命名）、带宽 7700/8 = 962.5 GB/s。

官方 profile 表（[NVIDIA, MIG User Guide](../references.md)，[C]，supported-mig-profiles 逐字；Memory/SM/L2 为分数，CE = copy engine 数，Instances = 该 profile 最大实例数）：

H100 80GB（Table 13，PCIe/SXM5 同表）：

| Profile | Memory | SM | L2 | CE | Instances |
| --- | --- | --- | --- | --- | --- |
| 1g.10gb | 1/8 | 1/7 | 1/8 | 1 | 7 |
| 1g.10gb+me | 1/8 | 1/7 | 1/8 | 1 | 1（media 扩展） |
| 1g.20gb | 2/8 | 1/7 | 1/8 | 1 | 4 |
| 2g.20gb | 2/8 | 2/7 | 2/8 | 2 | 3 |
| 3g.40gb | 4/8 | 3/7 | 4/8 | 3 | 2 |
| 4g.40gb | 4/8 | 4/7 | 4/8 | 4 | 1 |
| 7g.80gb | Full | 7/7 | Full | 8 | 1 |

表注：H100 1g.20gb 官方 Memory 列印 1/4，即 2/8——本表统一 /8 分母显式化切片数，数值与官方逐字等价。

B200 180GB（Table 8）：

| Profile | Memory | SM | L2 | CE | Instances |
| --- | --- | --- | --- | --- | --- |
| 1g.23gb | 1/8 | 1/7 | 1/8 | 2 | 7 |
| 1g.23gb+me | 1/8 | 1/7 | 1/8 | 2 | 1（media 扩展） |
| 1g.45gb | 2/8 | 1/7 | 2/8 | 2 | 4 |
| 2g.45gb | 2/8 | 2/7 | 2/8 | 3 | 3 |
| 3g.90gb | 4/8 | 3/7 | 4/8 | 6 | 2 |
| 4g.90gb | 4/8 | 4/7 | 4/8 | 8 | 1 |
| 7g.180gb | Full | 7/7 | Full | 16 | 1 |

A100-40GB 同为 1/8 显存 + 1/7 SM 切片（1g.5gb = 5 GB，7 实例；7g.40gb 整卡；80GB 版名按容量比例改 1g.10gb 等）；GB200-186GB 对比行 1g.23gb（186/8 = 23.25 → 名 23）/ 7g.186gb（CE 16）；A30-24GB 为 1/4 切片（1g.6gb = 24/4，最多 4 实例）。媒体引擎随代际变：H100 整卡 7 NVDEC/7 JPEG/1 OFA，B200 整卡仅 1 NVDEC/1 JPEG/1 OFA（NVDEC = 视频解码、JPEG = 图像解码、OFA = 光流加速器）——视频解码密集型负载选 MIG SKU 时须核媒体引擎列。

非对称 profile 的账：1g.20gb（H100）/ 1g.45gb（B200）拿 2 个 memory slice 但只拿 1 个 SM slice——用算力换容量，适合显存大而算力小的负载。4×1g.20gb 用满显存 8/8（80 GB），SM 只用 4/7，剩 3/7（≈56.6 SM 当量）空闲；对照 4g.40gb + 3g.40gb 双实例组合：SM 4/7+3/7 = 7/7、显存 4/8+4/8 = 8/8，两类资源同时用满（R510 之前该 4+3 组合受限制，新驱动已放开，concepts 页注）。

> **坑**：按 memory slice 数反推 L2。触发条件：见 H100 1g.20gb 有 2 个 memory slice，按「L2 随 memory slice 每片 1/8」算 L2 = 2/8×50 = 12.5 MB；现象：与官方 Table 13 逐字 L2 = 1/8（6.25 MB）不符——H100 的 1g.20gb 是例外行，而 B200 对应 1g.45gb 印 L2 = 2/8，两代表不一致；规避：「L2 随 memory slice 切分」说的是切分**方向**（memory 侧而非 SM 侧），具体数值一律以官方 profile 表 L2 列为准，不做跨代外推。

B200 SM 数的口径注（[B]，勿当 datasheet 数字）：148 SM = 2 die × 74 enabled，物理 2×80 = 160。一手为微基准论文逐字「feature 148 SMs spread across eight GPCs, provide four L2 cache partitions (double those in Hopper), and include eight HBM3e memory stacks」（[Jarmusch et al., Blackwell microbench](../references.md)，[B]，arXiv:2512.02189）；官方 Inside Blackwell Ultra blog 只披露 full config「up to 160 SMs」且「Available SM count and HBM capacity varies by SKU」（[NVIDIA, Inside Blackwell blog](../references.md)，[C] 开放技术博客）——B200 是 harvest SKU（harvest 机制 recap 1.1 §2），**官方 B200 datasheet/brief 不印 SM 数**，所以 148 标 [B] 微基准口径而非 [C] 官方规格。互证：1.1 的反推自洽（FP32 75 TF = 148×256×1.98 GHz）与该值闭合。残留注：论文自述 SM 数对 H200 为 1.09×，与 148/132 = 1.12 不自洽（论文自注该分解未直接实测），引 148 时以逐字 SM 计数为准、不引其比例分解。

隔离与部署。MIG 对 CUDA 编程模型大体透明（guide 逐字「the CUDA programming model remains unchanged」），实例交付为独立设备；同 GPU 上 MIG 实例间 P2P 自驱动 R570+ 支持（版本注）。部署命令（教学片段）：

```bash
nvidia-smi -i 0 -mig 1          # 启用 MIG mode
nvidia-smi mig -lgip            # 列 GI profiles
nvidia-smi mig -lgipp           # 列合法 placements
nvidia-smi mig -cgi 1g.10gb -C  # 建 GI 并随即建 CI
```

> **坑**：把 Hopper+ 的 MIG 启用语义当 Ampere 用。触发条件：按 Ampere 经验以为 enable 必须 GPU reset、且 MIG mode 跨 reboot 持久；现象：Hopper+ 逐字「enabling MIG mode no longer requires a GPU reset to take effect」，且 MIG mode 不跨 reboot 持久（无 InfoROM status bit，driver reload 即 disable）——Ampere 则需 reset 且 InfoROM 持久，两代语义相反；规避：部署脚本按代际分支；另注意**已创建的 GI/CI 在所有代际都不跨 reboot 持久**，重启后须重建；monitoring agent（DCGM/nvsm）占 GPU 会阻塞 Ampere 的 reset，须先停。placement 有物理位置约束，碎片化时 `-lgipp` 列不出合法位置、创建报 Insufficient Resources（命题 2 推论的工程表现）。

## 7. vGPU 与 SR-IOV：把分区交付给虚拟机

MIG 把 GPU 物理切开，vGPU 负责把切好的（或未切的）GPU 交付给虚拟机——中间靠什么硬件机制？所谓 vGPU（virtual GPU，虚拟 GPU），是指让多个 VM 同时直访一颗物理 GPU 的虚拟化方案：NVIDIA Virtual GPU Manager 跑在 hypervisor（虚拟机监控器）上，guest 驱动直访 GPU fast path、管理接口走半虚拟化（[NVIDIA, vGPU docs](../references.md)，[C]，vGPU 17.0 User Guide 一手）；每 vGPU 固定 framebuffer，创建时从物理显存划出、独占至销毁，总分配不得超过物理显存。

两种 backing 模式，隔离级别完全不同。time-sliced vGPU：物理 GPU **未分区**，guide 逐字「All time-sliced vGPUs resident on a GPU share access to the GPU's engines」——所有 vGPU 共享引擎，进程序列化调度、运行时独占，帧率限制器（frame-rate limiter，FRL）限 B 系列 45 FPS、Q/A 系列 60 FPS。MIG-backed vGPU：每 GI 一个 vGPU/VM，继承 MIG 的物理隔离；vGPU 17.0 逐字「GPUs that support the MIG feature are supported only on NVIDIA AI Enterprise」——MIG GPU 的 vGPU 仅 AI Enterprise 支持（许可口径注）。

| 模式 | 物理分区 | 引擎 | 隔离 | 适用 |
| --- | --- | --- | --- | --- |
| time-sliced vGPU | 无 | 共享、序列化调度 | 逻辑（无错误隔离） | 图形/轻负载多 VM |
| MIG-backed vGPU | GI 物理分区 | 每 GI 专属 | SM/内存/带宽/错误 | 多租户 AI 推理 |

SR-IOV 是 PCIe 层的硬件虚拟化机制。所谓 SR-IOV（Single Root I/O Virtualization，单根 IO 虚拟化），是指一个物理 PCIe 设备暴露一个 PF（Physical Function，物理功能，完整 PCIe 功能）与多个 VF（Virtual Function，虚拟功能，轻量 PCIe 功能），VF 经 IOMMU（IO Memory Management Unit，IO 内存管理单元）直通分配给 VM。NVIDIA 数据中心 GPU 自 Ampere 起支持；BIOS 须启 VT-d/IOMMU + SR-IOV + ARI（Alternative Routing ID Interpretation）。启用 VF 用官方脚本（guide 逐字「Use only the custom script sriov-manage … Do not try to enable the virtual function for the GPU by any other means」）：

```bash
/usr/lib/nvidia/sriov-manage -e 0000:41:00.0   # 启 VF（禁手动 sysfs）
ls -l /sys/bus/pci/devices/0000:41:00.0/ | /usr/bin/grep virtfn
# virtfn0 -> ../0000:41:00.4（VF 的 BDF，Bus/Device/Function 地址）
```

每 VF 仅能创建一个 mdev（mediated device）设备（guide 逐字「Only one mdev device file can be created on a virtual function」）；mdev 经 VFIO（Virtual Function IO，用户态设备直通框架）直通给 VM（legacy mdev VFIO vs Ubuntu 24.04+ 的 vendor-specific VFIO）；MIG-backed 的 mdev 不跨 reboot 持久（GI 消失即失效）。正例：启 SR-IOV 后 virtfn0 指向 0000:41:00.4（guide 示例 BDF），一个 VF 一个 mdev 一个 vGPU；边界反例：手动写 sysfs 建 VF 不被支持——VF 生命周期必须由 sriov-manage 管理，否则 vGPU Manager 状态不一致。

**边界指针（不展开）**：MIG 实例与 vGPU 是硬件侧的分区与虚拟化机制（本章交付）；把它们暴露为集群可调度资源——K8S device plugin、GPU 调度策略、多租户配额——以及多机集群编排，归 cluster 维度。本章边界止于「硬件怎么切、hypervisor 怎么接」，平台侧一律纯文字指引。

## 8. 服务器选型账与 TCO 电力成分

一台 DGX H100 全年满载的电费是多少美元？先给选型约束清单，再给电力账。选型是工程权衡而非可判定命题，四个约束各有数值锚：

| 约束 | 数值锚 | 形态后果 |
| --- | --- | --- |
| NVLink 域大小 | 域内 450 GB/s 单向/GPU vs NIC 50 GB/s（9×，recap 7.2 §7） | TP/EP 高频集合度数 ≤8 → 8 卡机箱；更大 → NVL72（72 卡域）或跨机 RDMA |
| 功率与散热 | 机箱 10.2–14.3 kW 风冷 vs 机架 ≈120 kW 液冷（§4） | 机房冷量/配电/承重（130.45–142.4 kg/台）是硬约束；100 kW 级机架须液冷改造 |
| PCIe 与 NUMA | Gen5 x16 64 GB/s 单向；跨 socket 折扣（§5） | H2D 密集负载选 x86 形态须核 NUMA 绑定；GB200 C2C 900 GB/s 免此账 |
| 容量与分区 | 单卡 80→180 GB；MIG 1/8 容量粒度（§6） | 多租户小负载用 MIG 换隔离；大模型整卡或 NVL72 域 |

TCO（Total Cost of Ownership，总拥有成本）的硬件电力成分（口径：只计电费；GPU 购置/折旧、网络、存储、运维不在本章，完整成本模型与 USD/M tokens 账归 inference 9.1 §7，指引不重推）：

$$
C_{\mathrm{elec}} = P_{\mathrm{IT}} \cdot \mathrm{PUE} \cdot h_{\mathrm{year}} \cdot c_{\mathrm{elec}} \qquad (4)
$$

**符号**： $P_{\mathrm{IT}}$ = IT 负载功率（kW，此处取铭牌 max = 上界口径）； $\mathrm{PUE}$ = 电源使用效率（口径：设施总功率 ÷ IT 功率，制冷/配电损耗计入分子）； $h_{\mathrm{year}}$ = 年小时数（8760 h）； $c_{\mathrm{elec}}$ = 电价（USD/kWh）。量纲核对：[kW]×[h]×[USD/kWh] = USD。全数字代入（**教学假设**：PUE 风冷 1.3 / 液冷 1.15，电价 0.1 USD/kWh，全年满载上界；实际平均功率与利用率低于此，归 inference 9.1 §2）：

| 形态 | $P_{\mathrm{IT}}$ | PUE | 年电费 | 每 GPU 年电费 |
| --- | --- | --- | --- | --- |
| DGX H100（8 GPU） | 10.2 kW | 1.3 | 11616 USD | 1452 USD |
| DGX B200（8 GPU） | 14.3 kW | 1.3 | 16285 USD | 2036 USD |
| GB200 NVL72（72 GPU） | 120 kW | 1.15 | 120888 USD | 1679 USD |

读表：每 GPU 电费 B200（2036）高于 NVL72（1679）高于 H100（1452）——液冷的低 PUE 部分抵消了高 TDP：折设施侧每 GPU 功率，NVL72 为 1.667×1.15 = 1.917 kW，低于 DGX B200 的 1.7875×1.3 = 2.324 kW。与租金量级对账（inference 9.1 §7 的教学假设租金 2 USD/GPU-hour，该章口径）：年租金 2×8760 = 17520 USD/GPU，电费占其 8.3%（H100）、9.6%（NVL72）、11.6%（B200）——电力成分约为租金的十分之一量级，是 TCO 硬件侧不可忽略但非主导的一项。电价与 PUE 是地域/设施变量，本表为教学假设非披露值，实际选型代入当地电价与实测 PUE。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 三形态边界：HGX = GPU 基板（无 CPU/主机内存，OEM 整机基底）；DGX = 自有品牌整机；NVL72 = 机架超节点（72 GPU + 36 Grace，18 compute + 9 switch trays） | §2 规格表 + 形态图 | §2 |
| NVLink 域聚合 recap：8 卡 7.2/14.4 TB/s、NVL72 130 TB/s（7.1 式 (4) 不重推）；域边界决定 TP/EP 放置上限（域内 9×/18× NIC） | §3 域表 | §3、7.1/7.2 |
| switch tray 端口账：2 chip × 72 port × 100 GB/s 双向 = 14.4 TB/s/tray；57.6 Tbps 是单 chip 口径（7.2 TB/s），当 tray 级用低估一半 | §3 端口账 + 坑注 | §3、练习 1 |
| GPU-only ≠ 铭牌：5.6/10.2 = 54.9%、8/14.3 = 55.9%、86.4/120 = 72%；120 kW 是机架级约值（含 Grace/switch/损耗） | §4 三笔账表 | §4、练习 1 |
| N+N 可用功率 = 装机之半 ≥ 机架负载：NVL72 132 ≥ 120 kW；DGX 4×3.3 = 13.2 ≥ 10.2、5×3.3 = 16.5 ≥ 14.3；≥3 PSU boot 是降级阈值非满载 | 式 (1)，命题 1 | §4、练习 1 |
| 液冷是物理必然： $\zeta_{\mathrm{water}}/\zeta_{\mathrm{air}}$ ≈ 3483；120 kW @15 K 需风 6.67 m³/s（≈14100 CFM）vs 水 1.9 L/s；DGX H100 官方 1105 CFM 与物理账闭合 | 式 (2) | §4、练习 4 |
| PCIe affinity：H2D = min(PCIe, inter-socket)（recap 7.2 式 (4)）；GPU-NIC 同 switch GDR 最优；nvidia-smi topo -m 核对；GB200 C2C 900 GB/s 免 PCIe H2D 账 | §5 拓扑图 + 三规则 | §5、7.2/9.2 |
| MIG 切片几何：memory slice 1/8（容量+带宽+MC+cache）、SM slice 1/7（roughly）；L2 随 memory slice 方向，逐 profile 以官方表为准（H100 1g.20gb L2 = 1/8 例外） | 式 (3) + profile 表 + 坑注 | §6、练习 2 |
| 实例上界 = min(SM 切片, memory 切片) = 7（A30 = 4）；7×1g 用满 SM、空闲 1/8 显存（10 GB）；分区守恒：分数和 ≤ 1 | 命题 2 + 推论 | §6、练习 3 |
| B200 SM = 148（2 die × 74，物理 160）为 [B] 微基准口径（arXiv:2512.02189），官方 datasheet 不印；blog full config up to 160 varies by SKU | §6 口径注 | §6、1.1 |
| MIG 部署代际漂移：Hopper+ enable 免 reset 且不持久 vs Ampere reset + InfoROM 持久；GI/CI 全代际不跨 reboot；P2P R570+ | §6 坑注 + 命令 | §6 |
| vGPU 两模式：time-sliced（未分区、共享引擎、FRL 45/60 FPS）vs MIG-backed（每 GI 一 VM，仅 AI Enterprise）；SR-IOV：PF/VF + IOMMU 直通，每 VF 一 mdev，sriov-manage 唯一入口 | §7 对比表 | §7 |
| 边界：硬件分区/虚拟化归 gpu 本章；K8S device plugin/调度/多机编排归 cluster（纯文字指针） | §7 边界段 | §1、§7 |
| TCO 电力成分 = 功率 × PUE × 8760 h × 电价；每 GPU 年电费 1452/2036/1679 USD（教学假设），约租金（2 USD/GPU-hour 口径）的 8–12% | 式 (4) | §8、练习 4、inference 9.1 §7 |

## 练习

1.（计算 · [10]）功率与端口账代入（§3/§4）：(a) DGX H100 与 NVL72 的 GPU-only TDP 乘积各占铭牌百分之几；(b) DGX H100/B200 每 U 功率密度；(c) NVL72 switch tray 级双向带宽（2 chip × 72 port × 100 GB/s），并说明 57.6 Tbps 是哪个口径；(d) 验证 N+N：8 shelf × 6 × 5.5 kW 之半是否覆盖 120 kW。校验：(a) 5.6/10.2 = 54.9%、86.4/120 = 72%（guide「approximately 120kW」口径）；(b) 1.275 / 1.43 kW/U；(c) 14.4 TB/s/tray，57.6 Tbps ÷ 8 = 7.2 TB/s 为单 chip，与 DGX B200 UG「2 switches … 14.4 TB/s aggregate」互证；(d) 132 ≥ 120，式 (1)。
2.（计算 · [20]）MIG 分数账（式 (3)）：(a) H100 1g.10gb 与 3g.40gb 的 SM 数（分数口径）、容量、带宽、L2 各多少；(b) B200 1g.23gb 同四项（SM 数用 148 [B] 口径），并解释 profile 名为何是 23gb 而非 22.5gb；(c) H100 1g.20gb 的 L2 是多少——按 memory slice 数反推会错成多少？校验：(a) 与官方 Table 13 分数列逐项一致（1/8、1/7、1/8；4/8、3/7、4/8），带宽 418.75 / 1675 GB/s 为分数 × datasheet 3.35 TB/s，L2 基数 50 MB（Hopper wp）；(b) 180/8 = 22.5 → 名舍入 23，带宽 962.5 GB/s；(c) 官方 L2 = 1/8 = 6.25 MB，反推 2/8 = 12.5 MB 错（§6 坑注）。
3.（推导 · [20]）证明命题 2（实例上界与空闲账）：(a) 写出计数上界论证（每 GI ≥1 SM 切片 + ≥1 memory 切片、切片互斥）；(b) 给出达界构造并说明为什么 7×1g 后第 8 个实例建不出来；(c) 算 7×1g.10gb 的 SM/显存/L2 已用与空闲分数（H100 口径，空闲显存折多少 GB）；(d) A30 为什么上界是 4；(e) 4×1g.20gb 组合哪类资源先用满、SM 空闲几分之几？校验：(b) 官方 profile 表 1g 行 Number of Instances = 7；(c) SM 7/7、显存 7/8（空闲 10 GB）、L2 7/8；(d) min(4,4) = 4（A30 Table 16 1g.6gb Instances = 4）；(e) 显存 8/8 用满、SM 空闲 3/7，与 1g.20gb Instances = 4 一致。
4.（计算 · [30]）散热与 TCO 组合账（式 (2)(4)，教学假设口径）：(a) DGX H100 10.2 kW、 $\Delta\theta$ = 15 K 所需风量（m³/s 与 CFM），对照官方 1105 CFM @80% fan PWM 反推对应温升；(b) NVL72 120 kW 同温升的风量与水量（L/s），两者比值与体积热容比的关系；(c) 三形态年电费与每 GPU 年电费（PUE 风冷 1.3 / 液冷 1.15、电价 0.1 USD/kWh、8760 h 满载上界）；(d) 每 GPU 电费占 inference 9.1 教学租金 2 USD/GPU-hour 年化值的百分之几？先预测 (c) 中每 GPU 电费最高的形态再算。校验：(a) ≈0.567 m³/s ≈ 1200 CFM，与 1105 CFM 同量级（反推 ≈16.3 K）；(b) 6.67 m³/s vs 1.91 L/s，比值 ≈3483 = $\zeta_{\mathrm{water}}/\zeta_{\mathrm{air}}$；(c) 11616 / 16285 / 120888 USD/年（容差 0.1%），每 GPU 最高为 DGX B200（2036 USD）；(d) ≈8.3%（H100）/ 11.6%（B200）/ 9.6%（NVL72）。

## 参考

- [NVIDIA, MIG User Guide](../references.md)：NVIDIA Multi-Instance GPU User Guide（key: mig-user-guide），[C] 官方规格披露·一手权威，2026-11-26 一手核实（2026-11-04 更新版）——concepts 页切片定义逐字（memory slice 1/8 含容量+带宽+MC+cache、SM slice 1/7 roughly、GI memory QoS、CI 共享 memory/engine）、Table 3 streams/MPS/MIG 对比、supported-mig-profiles 全表（A100 Table 2/15、H100 Table 13、B200 Table 8、GB200 Table 7、A30 Table 16）、supported-gpus（CC 8.0+、A30/RTX PRO 6000 = 4）、getting-started（Hopper+ 免 reset 不持久 vs Ampere reset + InfoROM、GI/CI 不跨 reboot、-mig 1/-lgip/-lgipp/-cgi -C）、P2P R570+、R510 placement 限制注（§1/§6）
- [NVIDIA, vGPU docs](../references.md)：NVIDIA vGPU 17.0 User Guide（key: vgpu-docs），[C] 官方规格披露·一手权威，2026-11-26——架构（Virtual GPU Manager + guest fast path + 固定 framebuffer）、time-sliced 逐字（未分区、共享 engines、序列化调度）、FRL（B 45 FPS / Q-A 60 FPS）、MIG-backed 仅 AI Enterprise 逐字、SR-IOV（VT-d/IOMMU + ARI、sriov-manage 唯一入口逐字、virtfn、每 VF 一 mdev 逐字、mdev VFIO vs vendor-specific VFIO）（§7）
- [NVIDIA, HGX/DGX](../references.md)：NVIDIA HGX / DGX 服务器规格（key: hgx-dgx），[C] 官方规格披露·一手权威，2026-11-26——HGX H100/H200 datasheet 8-GPU baseboard 表（NVSwitch 第三代、4-GPU N/A、Total Aggregate 7.2 TB/s）；DGX H100/H200 UG（8×H100 640 GB、2×Xeon 8480C、4×NVSwitch、8×CX7 单口 + 2×CX7 双口、2 TB DIMM、8U/130.45 kg/10.2 kW max/6×3.3 kW 4+2/200-240 VAC/≥3 PSU boot/1105 CFM @80% fan PWM）；DGX B200 UG（8×B200 1440 GB、2×Xeon 8570、2×第五代 NVLink switch 14.4 TB/s aggregate 逐字、10U/142.4 kg/14.3 kW/5+1/1550 CFM）（§2/§3/§4）
- [NVIDIA, GB200 NVL72](../references.md)：NVIDIA GB200 NVL72 规格（key: gb200-nvl72），[C] 官方规格披露·一手权威，2026-11-26——72 GPU + 36 Grace、18 compute trays（4 GPU + 2 Grace/tray）+ 9 switch trays、130 TB/s 双向、液冷铜缆 cartridge；供电散热（≈120 kW、8 power shelves、busbar、液冷，与 dgx-gb200-guide 互证）；NVLink switch tray 2×72-port chip / 57.6 Tbps full duplex（§2/§3/§4）
- [NVIDIA, DGX GB200 guide](../references.md)：NVIDIA DGX GB Rack Scale Systems User Guide hardware 页（key: dgx-gb200-guide），[C] 官方规格披露·一手权威，2026-11-26——rack configuration 逐字（18×1RU compute + 9×1RU switch + 2 TOR + power shelves + 铜缆背板）、compute tray（4×CX7 400G OSFP + 2×BF3、4×3.84 TB E1.S RAID0）、switch tray「2x NVLink NVSwitches, each of which has 72 NVLink ports」+ 57.6 Tbps full duplex 逐字、power shelf「approximately 120kW … six air-cooled 5.5kW PSUs in eight power shelves … N+N … 33kW per power shelf」逐字、busbar nominal 50-51V DC、cold plate + manifold 液冷（network/storage 风冷）+ leak detection（§2/§3/§4）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100 / H200 / HGX / Blackwell / Blackwell Ultra 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-26 重取——TDP（H100 SXM ≤700 W / HGX B200 ≤1000 W / GB200 configurable ≤1200 W / GB300 ≤1400 W / HGX B300 1100 W）、MIG 7、HBM 容量/带宽（80 GB/3.35 TB/s、180 GB/7.7 TB/s、186 GB/8 TB/s）、PCIe Gen5 128 GB/s、NVLink 头条（900 GB/s / 1.8 TB/s 双向）；B200 SM 数不印（§6 [B] 口径注）（§1/§2/§4/§6）
- [NVIDIA, Blackwell brief](../references.md)：NVIDIA Blackwell Architecture Technical Brief（key: blackwell-whitepaper），[C] 官方规格披露·一手权威，2026-11-26——Table 2 GB200 NVL72（72 GPU/36 Grace/18 compute trays/9 switch trays/130 TB/s 双向）、NVLink5 1.8 TB/s、raw Up to 192 GB（usable 180/186 GB 勿混，recap 2.1）、双 die NV-HBI（§2/§3）
- [NVIDIA, Inside Blackwell blog](../references.md)：NVIDIA Inside Blackwell Ultra blog（key: inside-blackwell-blog），[C] 开放技术博客、非闭源，2026-11-26——full config「up to 160 SMs」/8 GPC/640 TC 逐字、「Available SM count and HBM capacity varies by SKU」逐字（B200 harvest SKU 148 的 full config 锚）（§6）
- [Jarmusch et al., Blackwell microbench](../references.md)：Microbenchmarking NVIDIA's Blackwell Architecture（key: blackwell-b200-microbench），[B] 有 paper 无官方代码，arXiv:2512.02189（v3），2026-11-26 一手核实——逐字「148 SMs spread across eight GPCs, … four L2 cache partitions (double those in Hopper) … eight HBM3e memory stacks」；口径注：官方 B200 datasheet/brief 未印 SM 数，148 为微基准口径（Chips and Cheese 实测 74/die×2 互证）；残留：论文「1.09×」比例分解与 148/132 = 1.12 不自洽（自注未实测）（§6）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——132 SM（Table 3）、L2 50 MB、PCIe Gen5 p.49 逐字「128 GB/sec total bandwidth (64 GB/sec in each direction)」（§5/§6）
- [NVIDIA, nvidia-smi](../references.md)：NVIDIA nvidia-smi docs（key: nvidia-smi-docs），[C] 官方规格披露·一手权威，2026-11-26——topo -m（PCIe/NVLink 拓扑矩阵，NUMA/affinity 核对工具）（§5）
