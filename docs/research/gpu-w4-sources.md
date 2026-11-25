# gpu W4 research pass — P7（NVLink/NVSwitch）+ P8（NCCL 集合通信）一手来源核实

Task 1：P7（7.1-7.3）+ P8（8.1-8.4）硬件规格核实 + **NCCL fresh-tarball 代码锚清单**（writer 走读用）+ `gpu/references.md` P7/P8 扩充 + W3 残留修。8.4 为 **CODE-WALKTHROUGH DENSE**（NCCL 模块全量走读），代码锚以 **fresh tarball file:line** 为准，版本锚 + 核实日期 2026-11-25。writer 引用白名单 = `gpu/references.md` P7/P8 条目 + 跨 Part 共享锚点（P1 hopper-whitepaper / blackwell-whitepaper / nvidia-datasheet / hopper-blog / cuda-programming-guide，P2 nvbandwidth，P3 cuda-samples）。

## 0. 摘要

- **NVLink 代际 1.0→5.0 全部一手 whitepaper 核实**（Pascal / Volta / Ampere / Hopper / Blackwell 五份 PDF 本轮全取得）：H100 **18 links × 25 GB/s 单向 = 450 GB/s 单向 = 900 GB/s 双向**（Hopper whitepaper 逐字）；B200 **18 links × 50 GB/s 单向 = 900 GB/s 单向 = 1.8 TB/s 双向**（Blackwell brief 逐字「1.8 TB/sec total bandwidth, 900 GB/sec in each direction」）。**方向口径**：官方头条 900 GB/s / 1.8 TB/s 均为**双向**；per-link 25/50 GB/s 为**单向**。
- **NVSwitch**：H100 = 第三代 NVSwitch，**64 NVLink4 ports/chip**（Hopper wp）+ **HGX H100/H200 8-GPU = 4 NVSwitch chips** + **每 chip 双向 25.6 Tbit/s = 3.2 TB/s**（NVLink/NVSwitch blog 逐字）；Blackwell = **NVLink Switch（NVLink 5 Switch）**，spec「NVSwitch 4.0」非官方命名（订正注）；NVL72 = **9 switch trays × 144 NVLink ports @ 100 GB/s**（GB200 blog 逐字）= 1296 ports = 72 GPU × 18 links，聚合 **130 TB/s**。
- **NVLink 域**：HGX 8-GPU（H100/H200/B200）；GB200 NVL72 = **72 Blackwell GPU + 36 Grace CPU**，18 compute trays（4 GPU + 2 Grace/tray）+ 9 switch trays，液冷机架，铜缆 cartridge（brief Table 2 + datasheet + partition guide + blog 四源互证）；NVLink5 可扩展 **576 GPU 域 >1 PB/s**（brief/blog）。
- **NVLink-C2C vs NV-HBI**：C2C = Grace↔GPU **CPU-GPU** 一致互连 **900 GB/s 双向**（Grace-Hopper：Hopper blog 逐字；Grace-Blackwell：GB200 blog 逐字）；NV-HBI = B200 **双 die 间** die-to-die **10 TB/s** 单一致芯片（brief 逐字，W1 已锚本轮再核）。**两者勿混**。
- **GPUDirect P2P**：CUDA PG 13.5 **P2P 节再重组 → 03-advanced/multi-gpu-systems.html §3.4.2**（旧 peer-to-peer-memory-access-and-transfers.html 404）；**§3.4.2.5 IOMMU/PCI ACS 逐字**（Linux bare-metal IOMMU 开启不支持 PCIe P2P；ACS 把 P2P 流量重定向经 CPU root complex → bisection 带宽降）。GDS **cuFile API v1.20**（cuFileHandleRegister/Read/Write/DriverOpen/BufRegister）核实。**cuda-samples p2pBandwidthLatencyTest 已移除** → simpleP2P（漂移订正）。
- **NCCL 版本锚 2.32.7-1**（commit **351f33fd087e0c4d0cfb99f1c61dc13da7996c28**，master，2026-11-24）fresh tarball（4.7 MB）全量代码锚（§4）。**结构大漂移**：`src/collectives/device/` → `src/device/`；`src/graph/tuning.cc` → `src/tuning/`（per-algo cost model）；`src/enqueue.cc` → `src/enqueue/`；`devcomm.h` → `device.h` + `plugin/nccl_tuner.h`；新增 **NCCL_ALGO_PAT=6 / NVLS_TREE=5**、symmetric kernels、GIN、RAS、CE collectives。
- **经典**：Patarasuk-Yuan 2009（JPDC 69(2):117-124）/ Rabenseifner 2004（LNCS 3036 pp 1-9）Crossref 再核；**新增 Sanders-Speck-Träff 2009**（Parallel Computing 35(12):581-594，double binary tree 学术出处，NCCL 2.4 blog ref [6]，Crossref 核实）。
- **新增 7 条**：pascal-whitepaper / volta-whitepaper / ampere-whitepaper / nvlink-nvswitch-blog / gb200-blog / nvlink-c2c（P7 [C]）+ sanders-two-tree（P8 [B]）。**W3 残留修**：cutlass load L312→**L310**（refs + research 双修，contents API at 5c179edeb3e2 再核）；nvidia-datasheet 补 **INT8**（H100 SXM 3958*→**1979 dense TOPS** / HGX B200 **9 POPS sparse→4500 TOPS dense** / GB200 10 POPS→5000 dense）。
- **残留**：PCI-SIG 精确出处 403（数值口径经 Hopper wp + datasheet 互证闭合）；NVLink Switch 每 tray 2 chips / 每 chip 72 ports / 7.2 TB/s 逐字未核（tray 级 144 ports @ 100 GB/s 一手足够）；B200 SM 数（W1 残留 W6 不变）；JEDEC 403（W1 残留不变）。

## 1. 核实通道与网络 fallback（本轮实际使用）

- **codeload.github.com tarball 主通道**：NCCL commit tarball `https://codeload.github.com/NVIDIA/nccl/tar.gz/351f33fd087e0c4d0cfb99f1c61dc13da7996c28` 200（4724874 B）解压于独立空目录 `/tmp/gpu-w4-nvidia_nccl/`。GitHub API repo/commits 200（版本锚）。
- **images.nvidia.com 本轮可达**（W1 曾挂起）：Volta whitepaper 3851922 B / Pascal whitepaper 3225605 B / Ampere whitepaper 7979890 B 全 200 一手。
- **dam-cdn.nvd.orangelogic.com 稳定 200**（沿 W1）：Hopper wp V1.04（7880278 B）/ Blackwell brief（1268171 B）/ H100 datasheet（411990 B）/ H200 datasheet（642800 B）/ HGX H100-H200 datasheet（312368 B）/ Blackwell datasheet（1156321 B）重取；另取得 Blackwell Ultra datasheet / Line Card / L4 datasheet（GPU resources 落地页 AssetLink，非本波锚点）。
- **docs.nvidia.com 稳定 200**：CUDA PG 13.5 multi-gpu-systems / GDS cuFile API v1.20 / GPUDirect RDMA / NCCL User Guide（2.32.3）/ MNNVL partition guide v1.2 PDF（3151375 B）。
- **developer.nvidia.com blog 200**：NVLink/NVSwitch blog（2024-08-12）/ GB200 NVL72 blog（2024-03）/ NCCL 2.4 blog（2019-02-04）/ Hopper blog（重取）。
- **Crossref API 200**：Patarasuk-Yuan / Rabenseifner / Sanders 三 DOI 元数据（标题/venue/卷期页/作者/日期）。
- **404/403/redirect**：`nvidia.com/en-us/data-center/nvswitch/` **404**（并入 nvlink/ 页）；`grace-hopper/`、`gh200/` 产品页 **404**（产品页重组，C2C 数值改锚 Hopper blog + GB200 blog）；`resources.nvidia.com/en-us-grace-hopper*` **geo-redirect → nvidia.cn 首页**（slug 不存在）；`pcisig.com/specifications` **403**（bot 拦截，沿 W0）；CUDA PG 旧 P2P 子页 **404**（13.5 重组 → multi-gpu-systems.html）；Blackwell platform blog **404**（未用）。
- **nvidia.com 页面 .md 端点**：`nvlink.md` 200（markdown 版，**Rubin 数值与 HTML 漂移**：md 3.6 TB/s / 260 TB/s vs HTML 3 TB/s / 216 TB/s preliminary——NVLink 4/5 列两版一致，本波只用 4/5 列）。
- **cuda-samples master 树 API**（recursive，truncated=false）：p2pBandwidthLatencyTest 不存在；simpleP2P.cu contents API base64（cudaDeviceCanAccessPeer L99 / cudaDeviceEnablePeerAccess L130 / cudaMemcpyPeer L173）。
- **CUTLASS contents API at 5c179edeb3e2**：sm90_mma_tma_gmma_ss_warpspecialized.hpp `load(` 实在 **L310**（W3 残留 L312 订正依据）。
- 工具：`/usr/bin/grep -n`（file:line）；`pdftotext -layout`；`python3 -I`（JSON/base64/HTML 解析，隔离）；下载缓存 `/tmp/gpu-w4-research/`。

## 2. NVLink / NVSwitch 规格核实（方向口径逐项）

### 2.1 NVLink 代际表（全部一手 whitepaper，2026-11-25）

| 代际 | GPU（年份） | links/GPU | lanes/方向/link | 信令/lane | per-link 单向 | per-GPU 单向 | per-GPU 双向（官方头条） | 一手出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVLink 1.0 | P100 / GP100（2016） | 4 | 8 | 20 Gbps | 20 GB/s | 80 GB/s | **160 GB/s** | Pascal wp Fig 25：4×「x8, bidirectional, 20 GBps」+「Total 80GBps per direction」；「160 Gigabytes/second of bidirectional bandwidth—5x the bandwidth of PCIe Gen3 x16」 |
| NVLink 2.0 | V100 / GV100（2017） | 6 | 8 | 25.78 Gbps | 25 GB/s | 150 GB/s | **300 GB/s** | Volta wp：「increases the signaling rate from 20 to 25 Gigabits/second. Each link now provides 25 Gigabytes/second in each direction. The number of links... increased from four to six pushing the supported GPU NVLink bandwidth to 300 Gigabytes/second」 |
| NVLink 3.0 | A100 / GA100（2020） | 12 | 4 | 50 Gbps | 25 GB/s | 300 GB/s | **600 GB/s** | Ampere wp：「data rate of 50 Gbit/sec per signal pair, nearly doubling the 25.78 Gbits/sec rate in V100. Each link uses 4 differential signal pairs (4 lanes) in each direction compared to 8... in Volta... 25 GB/second bandwidth in each direction... twelve in A100, versus 6 in V100, yielding 600 GB/sec total bandwidth versus 300」 |
| NVLink 4.0 | H100 / GH100（2022） | 18 | 2 | 100 Gbps PAM4 | 25 GB/s | **450 GB/s** | **900 GB/s** | Hopper wp V1.04 p.47：「fourth-generation NVLink uses only two high-speed differential pairs in each direction to form a single link, also delivering 25 GB/sec effective bandwidth in each direction. H100 includes 18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth, while A100 includes 12 third-generation NVLink links to provide 600 GB/sec total bandwidth」；OSFP 节「8-channels of 100G PAM4 signaling」 |
| NVLink 5.0 | B200 / GB200（2024） | 18 | 2 | 200 Gbps PAM4 | **50 GB/s** | **900 GB/s** | **1.8 TB/s** | Blackwell brief p.10：「also uses two high-speed differential pairs in each direction... as in the Hopper GPU, NVIDIA Blackwell architecture doubles the effective bandwidth per link to 50 GB/sec in each direction. Blackwell and Blackwell Ultra GPUs include 18 fifth-generation NVLink links to provide 1.8 TB/sec total bandwidth, 900 GB/sec in each direction」 |
| （NVLink 6） | （Vera Rubin） | 36 | — | — | — | — | 3.6 TB/s（md）/ 3 TB/s（HTML preliminary） | NVLink 产品页——**非本波 scope，勿混代际**；HTML 与 .md 端点数值漂移（§7） |

- **方向口径硬规则（spec §4.3）**：官方头条 900 GB/s（H100）/ 1.8 TB/s（B200）= **双向总和**；per-link 25/50 GB/s = **单向**；per-GPU 单向 = 头条 ÷ 2（450/900 GB/s）。推导 $n_{\mathrm{link}} \times \mathrm{BW}_{\mathrm{link}}$ 时须声明方向口径：18 × 25 = 450 GB/s **单向**，×2 = 900 GB/s 双向。
- **代际带宽演进机制**：每 link 单向带宽 20→25→25→25→50 GB/s；lanes/方向/link 8→8→4→2→2（信令速率 20→25.78→50→100→200 Gbps/lane 翻倍驱动，lane 数减半维持每 link 带宽）；links/GPU 4→6→12→18→18。**NVLink4 vs 3 = 1.5×**（Hopper wp 逐字「1.5x the communications bandwidth」）；**NVLink5 vs 4 = 2×**（brief「doubles the performance」）。
- **vs PCIe**：NVLink4 = **7× PCIe Gen5**（Hopper wp 逐字「7x the bandwidth of PCIe Gen 5」，900/128≈7）；NVLink5 = **14× PCIe Gen5**（brief「over 14X the bandwidth of PCIe Gen5」，1800/128≈14）。
- NVLink 1.0 首发年份注：Volta wp「first introduced in 2016 with the Tesla P100」（产品出货）；GB200 blog 称「introduced in 2014」（GTC 宣布）——writer 用 2016（P100 出货）并注宣布 2014。

### 2.2 NVSwitch crossbar

**Hopper（第三代 NVSwitch）**：
- Hopper wp p.47-48 逐字：「Each new third-generation NVSwitch inside a node provides **64 ports of fourth-generation NVLink links**... Total switch throughput increases to **13.6 Tbits/sec** from 7.2 Tbits/sec in the prior generation」。
- NVLink/NVSwitch blog（2024-08-12）逐字：「Every NVIDIA HGX H100 and NVIDIA HGX H200 system with eight GPUs features **four third-generation NVSwitch chips**. The total bidirectional bandwidth of each NVSwitch chip is a staggering **25.6 terabits per second**.」→ **每 chip 3.2 TB/s 双向 = 64 ports × 50 GB/s 双向/port**（25 GB/s 单向 × 2）。
- **口径注**：wp「13.6 Tbits/sec」为 switch throughput 口径（含信令开销的单向口径，64×2 lanes×100G PAM4×1.0625≈13.6T）；blog「25.6 Tbit/s」为**双向有效带宽**口径（=3.2 TB/s）。writer 用 blog 3.2 TB/s 双向并注 wp 原文，勿混。
- **非阻塞 crossbar**（blog 逐字）：「With the NVSwitch, every NVIDIA Hopper GPU in a server can communicate at 900 GB/s with any other NVIDIA Hopper GPU simultaneously. The peak rate does not depend on the number of GPUs that are communicating. That is, the NVSwitch is non-blocking.」
- **P2P 对照账**（blog Table 1，7.2 用）：无 NVSwitch 点对点拓扑，每 GPU 900 GB/s 拆 7 条专用连接 → **每对 128 GB/s**（900/7≈128.6）；2-GPU 通信 128 vs 900 GB/s、8-GPU 7×128 vs 900 GB/s。Llama3.1-70B TP=2 实测 NVSwitch 1.3-1.5× 吞吐（H200，blog Table 2）。
- **NVLink SHARP（NVLS）**（Hopper wp 逐字）：「hardware acceleration of collective operations with multicast and NVIDIA SHARP in-network reductions. Accelerated collectives include **write broadcast (all_gather), reduce_scatter, and broadcast atomics**. In-fabric multicast and reductions provide **up to 2x throughput gain** while significantly reducing latency for small block size collectives... significantly reduces the load on SMs」。NCCL docs NCCL_NVLS_ENABLE 节互证：「NVLink SHARP is available in **third-generation NVSwitch systems (NVLink4) with Hopper and later** GPU architectures」。
- HGX H100/H200 datasheet NVSwitch 行：8-GPU = Third generation；**4-GPU = N/A**（无 NVSwitch，P2P 连接）。

**Blackwell（NVLink Switch / NVLink 5 Switch）**：
- **命名订正**：spec 7.1「B200 NVSwitch 4.0」非官方命名。官方名 = **NVLink Switch**（brief/blog）/ **NVLink 5 Switch**（NVLink 产品页规格表），即第四代 NVSwitch 芯片（Hopper 第三代 → Blackwell 第四代）。writer 用「NVLink Switch（NVLink 5 Switch，第四代 NVSwitch）」并注命名。
- GB200 blog 逐字：「Each NVLink switch tray delivers **144 NVLink ports at 100 GB** so the nine switches fully connect each of the 18 NVLink ports on every one of the 72 Blackwell GPUs.」→ **每 tray 144 ports × 100 GB/s 双向 = 14.4 TB/s 双向**；9 trays × 144 = 1296 ports = 72 GPU × 18 links ✓；聚合 9 × 14.4 = 129.6 ≈ **130 TB/s**（brief/datasheet/产品页一致）。
- brief Table 2（GB200 NVL72）：**NVLink Switch Trays 9**；NVLink Bandwidth **130 TB/s Bidirectional**。partition guide v1.2（DU-12143-001）：NVL72 = 18 compute trays + 9 switch trays（Fig 2）；NVL36×2 = 每 chassis 18 compute + 9 switch。
- brief：「NVLink Switch... delivers **4X bandwidth efficiency** with new NVIDIA SHARP **FP8** support」；NVLink5「scales up to **576 GPUs**」（blog：576 GPU 域 >1 PB/s = 576×1.8≈1037 TB/s）。
- **残留**：每 tray 2 NVLink Switch chips（NVL72 共 18 chips）/ 每 chip 72 ports / 每 chip 7.2 TB/s 双向——本轮未逐字核到一手（blog 以 tray 为「switch」单位）。**writer 用 tray 级一手数值（144 ports @ 100 GB/s）推导**；per-chip 拆分如引用须注「派生/待核」。
- **Hopper 时代 NVLink Switch System（历史注）**：Hopper wp 宣布 256 GPU / 32 节点 NVLink Switch System（2:1 tapered fat tree，57.6 TB/s all-to-all），但逐字「NVLink Switch System technology is **not available with H100 systems now**」——宣布未出货；**GB200 NVL72 才是首个出货的多节点 NVLink 域产品**。writer 7.1 勿把 256-GPU Hopper 域当出货规格。

### 2.3 NVLink 域与聚合带宽账（7.1 关键推导一手）

| 域 | GPU 数 | per-GPU 双向 | 域聚合（= n_GPU × per-GPU 双向） | 出处 |
| --- | --- | --- | --- | --- |
| HGX H100/H200 8-GPU | 8 | 900 GB/s | **7.2 TB/s** | NVLink 产品页 NVLink4 Switch「Total Aggregate Bandwidth 7.2 TB/s」✓ |
| HGX B200 8-GPU | 8 | 1.8 TB/s | **14.4 TB/s** | Blackwell datasheet「Total NVLink Bandwidth 14.4 TB/s（HGX B200）」+ brief HGX B200 表「Aggregate NVLink 14.4 TB/s」 |
| GB200 NVL72 | 72 | 1.8 TB/s | **130 TB/s**（72×1.8=129.6） | brief Table 2 + datasheet + 产品页 + blog 四源一致 |
| GB200 NVL4 | 4 | 1.8 TB/s | 7.2 TB/s | Blackwell datasheet（NVL4 列） |
| （NVLink5 最大域） | 576 | 1.8 TB/s | >1 PB/s | brief/blog（多机架 NVLink Switch System，非单 rack 出货默认） |

- GB200 NVL72 构成（brief Table 2 + blog + partition guide）：**72 Blackwell GPU + 36 Grace CPU**；18 compute trays ×（4 Blackwell + 2 Grace）；9 NVLink switch trays；液冷机架；铜缆 cartridge 互连；GB200 superchip = 2 Blackwell + 1 Grace 经 **NVLink-C2C**。NVL36 = 36 GPU 单 rack（9 dual-GB200 compute nodes）；NVL72 = 72 GPU 单 rack（18 dual nodes）或双 rack（18 single nodes）。
- 全互连 vs switch：8-GPU HGX 经 4 NVSwitch 非阻塞全互连（任意对 900 GB/s 同时）；无 switch 点对点则每对带宽 = 总带宽 ÷ (n-1)（blog 128 GB/s 例）。**NVLink 域内 ring 映射 → 8.2/8.4；域 vs 机间带宽数量级差 → P9/P10**。

### 2.4 NVLink-C2C（CPU-GPU）vs NV-HBI（die-to-die）

| 互连 | 连接对象 | 带宽 | 一致性 | 一手出处 |
| --- | --- | --- | --- | --- |
| **NVLink-C2C** | Grace CPU ↔ Hopper GPU（GH200） | **900 GB/s total（双向）**，7× PCIe Gen5 | 一致统一内存 | Hopper blog 逐字：「H100 is paired to the NVIDIA Grace CPU with the ultra-fast NVIDIA chip-to-chip interconnect, delivering 900 GB/s of total bandwidth, 7x faster than PCIe Gen5」 |
| **NVLink-C2C** | Grace CPU ↔ 2 Blackwell GPU（GB200） | **900 GB/s bidirectional** | 「coherent access to a unified memory space」 | GB200 blog 逐字：「NVLink-Chip-to-Chip (C2C) interface that delivers 900 GB/s of bidirectional bandwidth. With NVLink-C2C, applications have coherent access to a unified memory space」 |
| **NV-HBI** | B200 双 reticle-limit die 间 | **10 TB/s** chip-to-chip | 「one fully coherent chip」 | Blackwell brief p.7-8 逐字：「The two dies are connected and unified with a single 10 terabyte-per-second (TB/s) chip-to-chip NVIDIA High-Bandwidth Interface (NV-HBI), providing one fully coherent chip」 |

- C2C 产品页（nvlink-c2c/ 200）机制补充：chip-to-chip 一致互连，支持 **Arm AMBA CHI / CXL** 协议；先进封装（PCB/MCM/interposer/wafer-level）；**6× 能效 / 3.5× 面积效率 vs PCIe Gen6 PHY**；NVLink Fusion 半定制开放。**页面无带宽数值**——900 GB/s 锚 Hopper blog（GH）/ GB200 blog（GB）。
- **勿混**：NV-HBI 10 TB/s 是**封装内 die-to-die**（B200 双 die 合一），NVLink-C2C 900 GB/s 是**CPU-GPU 芯片间**，NVLink 1.8 TB/s 是 **GPU-GPU**。三者层级/数量级不同（7.1 须分层讲清）。

### 2.5 PCIe（7.2 对照账）

- **PCIe Gen5 x16 = 128 GB/s 双向 = 64 GB/s 单向**——Hopper wp p.49 逐字：「H100 incorporates a PCI Express Gen 5 x16 lane interface, providing **128 GB/sec total bandwidth (64 GB/sec in each direction)** compared to 64 GB/sec total bandwidth (32GB/sec in each direction) in Gen 4 PCIe included in A100」。H100/Blackwell datasheet Interconnect 行「PCIe Gen5: 128GB/s」互证。
- **PCI-SIG 原文 403 残留不变**（W0→W4，bot 拦截）；数值口径已经 NVIDIA 官方 wp + datasheet 一手闭合，pcie-spec 条目保留 403 注 + 互证注。
- H100 PCIe 版 NVLink = 600 GB/s（datasheet，bridge 连接）勿混 SXM 900 GB/s；H100 NVL 94GB×2=188GB（datasheet）沿 W1 口径。

## 3. GPUDirect P2P / GDS 核实（7.3）

### 3.1 CUDA Programming Guide 13.5 P2P 节（URL 漂移销项）

**旧 `03-advanced/peer-to-peer-memory-access-and-transfers.html` 404 → 现行 `03-advanced/multi-gpu-systems.html`**（200，Last updated 2026-10-31）。节结构：

| 节 | 内容（逐字要点） |
| --- | --- |
| §3.4.2.1 Peer-to-Peer Memory Transfers | cudaMemcpyPeer / cudaMemcpyPeerAsync / cudaMemcpy3DPeer / cudaMemcpy3DPeerAsync；UVA 下 cudaMemcpyDefault/DeviceToDevice 可跨设备；**「On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections」** |
| §3.4.2.2 Peer-to-Peer Memory Access | **cudaDeviceCanAccessPeer() 判可行 + cudaDeviceEnablePeerAccess() 启用**；统一虚拟地址空间「the same pointer can be used to address memory on both devices」；fine-grained load/store/atomic 直访 peer HBM |
| §3.4.2.3 Peer-to-Peer Memory Consistency | P2P 访问一致性语义 |
| §3.4.2.4 Multi-Device Managed Memory | 多设备 managed memory |
| **§3.4.2.5 Host IOMMU Hardware, PCI Access Control Services, and VMs** | **「CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer. However, CUDA and the display driver do support IOMMU via virtual machine pass through. The IOMMU must be disabled when running Linux on a [bare-metal system]」**；**「PCI Access Control Services (ACS) can be enabled on systems that support IOMMU. The PCI ACS feature redirects all PCI point-to-point traffic through the CPU root complex, which can cause significant performance loss due to the reduction in overall bisection bandwidth」** |
| §3.4.2.6 Compute Fabric Transport | CUDA 13.x 新增 fabric transport（peer allocation 经 VA 映射的替代路径） |

- **7.3 PCIe P2P 限制一手齐备**：ACS 重定向经 root complex（bisection 带宽降）+ IOMMU bare-metal 不支持（须 disable 或 VM passthrough）+ 同 root complex 要求（GDR docs 互证）。
- GPUDirect RDMA docs（200）互证：「the two devices must share the same upstream PCI Express root complex」；IOMMU「incompatible with IOMMUs performing any form of translation other than 1:1, hence they must be disabled or configured for pass-through」；BAR1（「BAR1 size is set to 16GB or larger」large BAR）；nvidia-peermem 注册 GPU 到 IB 子系统（9.2 回链）。

### 3.2 GPUDirect 家族与 GDS

- **GPUDirect 家族页**（developer.nvidia.com/gpudirect 200）：「GPUDirect is a family of technologies, part of Magnum IO... **GPUDirect Storage, GPUDirect RDMA, GPUDirect Peer to Peer (P2P) and GPUDirect Video**」；GDS「direct data path between local or remote storage, such as NVMe or NVMe over Fabric (NVMe-oF), and GPU memory. It avoids extra copies through a bounce buffer in the CPU's memory... DMA engine near the NIC or storage」；GDR「eliminates the system CPUs and the required buffer copies」。
- **GDS cuFile API Reference v1.20**（docs.nvidia.com/gpudirect-storage/api-reference-guide/ 200）：**cuFileHandleRegister（§3.2.1）/ cuFileRead / cuFileWrite / cuFileDriverOpen / cuFileBufRegister** 逐字核实；cuFileRead/Write 不保证 null stream 排序语义。GDS 主站 200。⟨verify cuFile⟩ 销项。
- **cuda-samples P2P 例漂移订正**：**p2pBandwidthLatencyTest 已从 master 移除**（recursive 树 API 全搜无 p2p bandwidth sample）；现行 = **cpp/0_Introduction/simpleP2P**（simpleP2P.cu：cudaDeviceCanAccessPeer L99 / cudaDeviceEnablePeerAccess L130 / cudaMemcpyPeer 带宽打印 L173）+ cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P + python/4_DistributedComputing/simpleP2P。**P2P 带宽矩阵实测改锚 nvbandwidth**（W1 已锚，7.2）。

## 4. NCCL 代码锚清单（fresh tarball，版本 2.32.7-1）

> **版本锚**：NVIDIA/nccl master commit **351f33fd087e0c4d0cfb99f1c61dc13da7996c28**（2026-11-24，「Merge v2.32.7-1 release into master」）；`makefiles/version.mk` L9-12 `NCCL_MAJOR 2 / MINOR 32 / PATCH 7`。tarball 4724874 B，解压 `/tmp/gpu-w4-nvidia_nccl/nccl-351f33fd.../`。核实日期 2026-11-25。**writer 走读引用一律带版本锚**。

### 4.0 结构漂移（spec 旧布局 → 2.32 现行，writer 必注）

| spec/旧布局 | 2.32.7 现行 | 说明 |
| --- | --- | --- |
| `src/collectives/device/` | **`src/device/`** | kernel + primitives（all_reduce.h / prims_ll/ll128/simple.h / common.h / op128.h） |
| `src/graph/tuning.cc` | **`src/tuning/`** | tuning.cc + cost_model.cc + tuning_general.cc + **per-algo model**（ring.cc / tree.cc / nvls.cc / collnet.cc / pat.cc）+ platforms/（含 rubin.h）+ pipeline_model/ + sym_model/ |
| `src/enqueue.cc` | **`src/enqueue/`** | enqueue.cc + task_prep/{task_classify,task_pretuning,task_posttuning}.cc + task_sched/{legacy,ce,pat?}_sched.cc |
| `src/include/devcomm.h` | **`src/include/device.h`** + **`src/include/plugin/nccl_tuner.h`** | algo/proto 枚举移 nccl_tuner.h（tuner plugin v7 ABI）；src/devcomm/ 为版本化 device ABI（devcomm_v22902...v23204） |
| 算法 ring/tree/NVLS/CollNet | **+ NCCL_ALGO_PAT=6（2.23+）/ NVLS_TREE=5（2.18+）** | PAT = Parallel Aggregated Tree（AllGather/ReduceScatter）；NVLS_TREE = 多节点 NVLS |
| transport P2P/SHM/NET | + coll_net / **nvls 独立**（不在 ncclTransports 表）+ multicast/mc_arena/nvls_ub/rma_socket/net_ib/net_nd/net_socket | ncclTransports[] 仅 4 项（p2p/shm/net/collNet） |
| — | 新增 **symmetric kernels**（src/device/symmetric/ + sym_kernels.cc + NCCL_WIN_ENABLE）/ **GIN**（src/gin/，GPU-Initiated Networking，DeepEP V2.5 后端）/ **RAS**（src/ras/ 可靠性）/ **CE collectives**（ce_coll.cc copy-engine）/ **MNNVL**（mnnvl.cc）/ src/rma/ / src/scheduler/ | 2.28+ 对称内存窗口 API；8.4 走读 scope 以 ring/tree/NVLS/CollNet + 三协议为主，新特性注存在即可 |

### 4.1 初始化流程（ncclCommInitRank，8.4）

| file:line | 机制 |
| --- | --- |
| `src/init.cc` L3071 | **ncclCommInitRank(newcomm, nranks, commId, myrank)** 公共 API 入口 |
| `src/init.cc` L2975 / L2229 | ncclCommInitRankDev → **ncclCommInitRankFunc**（async job，L2132 ncclCommInitRankAsyncJob） |
| `src/init.cc` L2314 | **bootstrapInit**（bootstrap allgather 建立 rank 间带外连接） |
| `src/init.cc` L2320 / L1219 | **initTransportsRank**（初始化主体） |
| `src/init.cc` L1447 / L1451 | **ncclTopoGetSystem**（拓扑检测：NVML/PCI/NVLink → XML → topo system） |
| `src/init.cc` L1453 | **ncclTopoComputePaths**（GPU/NIC/CPU 间路径类型与带宽） |
| `src/init.cc` L1488 / L1496 / L1513 / L1515 / L1525 | **ncclTopoCompute** 逐算法图搜索：ringGraph L1488 / treeGraph L1496 / collNetChainGraph L1513 / collNetDirectGraph L1515 / **nvlsGraph L1525** |
| `src/init.cc` L1957 | ncclTransportP2pSetup（transport 连接建立） |
| `src/init.cc` L2076 / L714 | **devCommSetup**（host→device ncclDevComm + channels 拷贝，L847 cudaMemcpyAsync） |

### 4.2 拓扑检测与图构建（src/graph/，8.4 channel/ring/tree）

| file:line | 机制 |
| --- | --- |
| `src/graph/topo.cc` L1994 / L1072 | ncclTopoGetSystem / ncclTopoGetSystemFromXml（XML→topo） |
| `src/graph/topo.cc` L944 / L956 / L970 | nvlBw=ncclTopoNVLinkBw(compCap)；**LINK_NVL 边构建**（count×nvlBw 分摊到 remote devs） |
| `src/graph/topo.h` L23-26 / L317-323 | **SM80_NVLINK_BW 20.0 / SM90_NVLINK_BW 20.6 / SM100_NVLINK_BW 40.1**（GB/s per-link 单向，**NCCL 有效带宽模型值**，低于 raw 25/50——协议/编码开销折扣）；ncclTopoNVLinkBw 按 cudaCompCap 分派 |
| `src/graph/paths.cc` L104 / L243 / L341 / L264 | **PATH_PHB**（经 CPU root complex）/ **PATH_PXN**（经中间 GPU 到 NIC）/ p2pLevel=**PATH_NVL** 判 P2P 可行 / levelsOldToNew（PATH_LOC/PIX/PXB/PHB/SYS） |
| `src/graph/search.cc` L331 / L357 / L1245 | **ncclTopoSearchRec / ncclTopoSearchRecGpu**（递归图搜索，带宽/延迟约束）/ **ncclTopoCompute**（每算法图搜索入口，cache 一次计算） |
| `src/graph/rings.cc` L29 | **ncclBuildRings**（prev/next 邻接 → ring 数组；loopback 校验「ring does not loop back to start」+ 全 rank 覆盖校验） |
| `src/graph/trees.cc` L32 / L69-88 / L90 | **ncclGetBtree**（binomial tree pattern）/ **double binary tree 注释逐字**：「Build a double binary tree. Take the previous tree for the first tree. For the second tree, we use a **mirror tree (if nranks is even)**... or **shift it by one rank (if nranks is odd)**」（含 12/13-rank ASCII 双树图）/ **ncclGetDtree**（双树构建：第一棵 btree + 第二棵 mirror/shift） |
| `src/graph/connect.cc` L22 / L395 | ncclTopoPreset / **ncclTopoPostset**（图 → channel 连接，ring/tree 落 channel） |
| `src/include/graph.h` L173-187 | **struct ncclTopoGraph**：pattern / minChannels / **nChannels** / **bwIntra / bwInter / latencyInter** / typeIntra（PATH_NVL/PATH_PCI）——8.2/8.3 带宽账的代码载体 |

### 4.3 算法/协议选择与 cost model（src/tuning/ + src/enqueue/，8.3/8.4）

| file:line | 机制 |
| --- | --- |
| `src/include/plugin/nccl_tuner.h` L28-34 | **NCCL_ALGO_TREE 0 / RING 1 / COLLNET_DIRECT 2 / COLLNET_CHAIN 3 / NVLS 4 / NVLS_TREE 5 / PAT 6** |
| `src/include/plugin/nccl_tuner.h` L38-42 / L46-48 | **NCCL_PROTO_LL 0 / LL128 1 / SIMPLE 2**，NCCL_NUM_PROTOCOLS 3；NCCL_HW_NVLINK 0 / PCI 1 / NET 2 |
| `src/tuning/tuning.cc` L42 / L153 / L180 / L347 | ncclTuningInit（cost model + tuner plugin 加载）/ **ncclTuningSelectBestTuning（最小 timeUs 选择）** / ncclTuningCompute / ncclTuningComputeTuning→SimModel |
| `src/tuning/cost_model.cc` L125 / L331 / L342 / L514 | **isLL128Enabled**（compCap + NVLink 门控，LL128 仅 NVLink 平台）/ PreInit / Init / **ncclTuningCostModelSimModel**（per-algo/proto timeUs 计算） |
| `src/tuning/cost_model.cc` L365-379 | **NCCL_PROTO / NCCL_ALGO env 解析**（parseList，^ 排除语法） |
| `src/tuning/tuning_general.cc` L58-73 | **ncclTuningGetTime：`time = lat × latCount × precision_ratio + nBytes/(1000×bw) × precision_ratio`**（lat μs / bw GB/s → time μs；**α-β 模型的代码形态**，8.2/8.3 对账锚）；latCount：ring=numPipeOps，tree=DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS)；**FP8 ring nRanks>8 → precision_ratio=1024**（深环 FP8 归约精度损失降级，注释「Relegate fp8 reduction trees of sufficient depth that they incur precision loss to be least preferred」） |
| `src/tuning/ring.cc` L21 / L93-141 | RingModelInit（perChMaxRingLL128Bw 表）/ **RingModelSim**：lat/bw 查 generalLatencies/generalBandwidths 表 + **ringSimplePlateau 修正**（多节点 AllReduce 或 Blackwell 单节点 NVLink AllReduce/AllGather/ReduceScatter，plateauFactor 1.4（≥sm80）/1.9（<sm80），per-rank/channel bytes 与 step 数调制）→ ncclTuningGetTime |
| `src/tuning/tree.cc` L20-58 / L63-83 | **TreeModelInit（仅 AllReduce，其余 func hard disable）**：bw=min(bwInter,bwIntra)（≥sm100）；busBw=bw×nChannels；AllReduce **busBw×0.92**；LL **×1/3.8**；LL128 **×120/128**（=0.9375 协议效率）；**generalBandwidths=busBw×0.5**（allreduce 2M 流量 → algo 带宽 = bus 带宽之半）；**latency += 2×((ppn-1)×intraLat + log2i(nNodes)×interLat)**（**log2 树深延迟项代码实化**，8.3 O(log n) 锚）；TreeModelSim：**treeCorrectionFactor[proto][logSize]** 尺寸修正（logSize=log2i(nBytes>>6)，0..22） |
| `src/enqueue/task_prep/task_pretuning.cc` L245-257 | fillCollTuningInput（tuning 输入装配） |
| `src/enqueue/task_prep/task_posttuning.cc` L547-548 | **task->algorithm = tuningOut.algo；task->protocol = tuningOut.proto**（选择结果落 task） |
| `src/enqueue/enqueue.cc` L535-554 / L660 | NVLS/NVLS_TREE/PAT/CollNet 分支 + **LL trafficBytes×4**（LL 8B data+8B flag → 线上流量 2×，代码按 4× 计含双 flag 开销口径）/ scheduleCollTasksToPlan |

### 4.4 Transport 层（src/transport/，8.4）

| file:line | 机制 |
| --- | --- |
| `src/transport.cc` L15-20 | **ncclTransports[NTRANSPORTS] = {&p2pTransport, &shmTransport, &netTransport, &collNetTransport}**（**NVLS 不在此表**，独立 nvls.cc/multicast.cc 路径） |
| `src/transport/p2p.cc` L130 / L358 / L331 / L403 / L410-416 | **p2pCanConnect**（拓扑判 P2P 可行）/ **p2pMap**（peer 内存映射）/ **cudaIpcOpenMemHandle(cudaIpcMemLazyEnablePeerAccess)**（跨进程 IPC）/ **p2pSendSetup** / **useRead**（NVLink read 语义 vs write，p2pGetInfo 决定；useMemcpy 时 read=0） |
| `src/transport/p2p.cc` L697 / L740 / L763 | p2pSendProxySetup / p2pRecvProxySetup / p2pSendProxyConnect（P2P proxy 辅助，数据路径 GPU 直访不经 proxy 拷贝） |
| `src/transport/shm.cc` L61 / L88 / L121 / L245 / L269 | **shmCanConnect / shmSendSetup / shmRecvSetup**（host 共享内存 transport，PCIe 同机无 P2P 时）/ shmSend/RecvProxySetup（**SHM 经 proxy 线程拷贝**） |
| `src/transport/net.cc` L296 / L356 / L753 / L794 | **sendSetup / recvSetup**（NET transport，ncclProxyCallBlocking 到 proxy）/ sendProxySetup / recvProxySetup（net plugin 属性协商） |
| `src/transport/net.cc` **L1327 / L1496** | **sendProxyProgress / recvProxyProgress**（**proxy 线程内的网络进度状态机**：isend/irecv/test 轮询，GPU-NIC 协调核心） |
| `src/transport/net.cc` L325 / L337-340 / L1023 | GDRDMA 日志（useGdr）/ **GDRCOPY_SYNC_ENABLE（默认 1）/ GDRCOPY_FLUSH_ENABLE（默认 0）**（gdrcopy 同步字 + PCIe read flush）/ nv_peermem GDR 路径 |
| `src/transport/nvls.cc` L23 / L246 / **L270** / L283 / L359 / L494 | nvlsCanConnect / ncclNvlsInit / **cuMulticastCreate 探测 multicast 支持**（「NVLS multicast support is %savailable」）/ ncclNvlsTreeConnect / ncclNvlsBufferSetup / **ncclNvlsSetup**（NVLS multicast 组建立，NVSwitch SHARP 路径） |
| `src/transport/coll_net.cc` L143 / L169 / L191 / L416 / L1862 | canConnect / sendSetup / recvSetup / **sharedBuffersInit**（CollNet 共享缓冲，SHARP 集成）/ collNetTransport struct |

### 4.5 Proxy 线程（src/proxy.cc，8.4）

| file:line | 机制 |
| --- | --- |
| `src/proxy.cc` L481 | ncclProxyPost（GPU kernel → proxy ops 池提交） |
| `src/proxy.cc` L801 | **progressOps**（遍历活跃 proxy args，调各 transport proxyProgress） |
| `src/proxy.cc` **L956** | **ncclProxyProgress(void* proxyState_)——proxy 线程入口**（每 comm 一线程，循环 progressOps；NET/SHM 数据路径的 CPU 侧代理） |

### 4.6 Kernel 与协议（src/device/，8.4）

| file:line | 机制 |
| --- | --- |
| `src/device/all_reduce.h` **L14 / L86 / L146** | **runRing / runTreeUpDown / runTreeSplit**（AllReduce 三算法 kernel 骨架，template<T,RedOp,Proto>） |
| `src/device/all_reduce.h` L232-242 / L765-786 | ProtoSimple dispatch（ring / treeUpDown / treeSplit）；**LL/LL128 dispatch（ring + treeSplit，无 treeUpDown）** |
| `src/device/all_reduce.h` **L390-442** | **RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE>**：nvls->up multicast 指针 + scatter（FanAsymmetric<0,NCCL_MAX_NVLS_ARITY>）/ gather / reduce-broadcast through NVLS（**NVSwitch 内归约 kernel 侧**） |
| `src/include/nccl_device/impl/ll__types.h` **L14-26** | **union ncclLLFifoLine：{data1 u32, flag1 u32, data2 u32, flag2 u32}，static_assert sizeof==16**（L25）；注释逐字「**Flags follow data so that a partial network receive cannot expose a flag before its data. This assumes contiguous network writes or eight-byte write atomicity**」→ **LL：每 16B line = 8B data + 8B flag，带宽效率 50%**；NCCL_LL_LINES_PER_THREAD 8（L28） |
| `src/device/prims_ll.h` L103-113 | LL recv：**ld.volatile.v4 读 data1/flag1/data2/flag2，spin 直到 flag1==flag && flag2==flag**（flag 即同步，无独立 fence；LL 低延迟机制） |
| `src/include/nccl_device/impl/ll128__types.h` **L13-15 / L17-20** | **NCCL_LL128_LINESIZE 128 / LINEELEMS 16（128/8）/ DATAELEMS 15** → **LL128：每 128B line = 120B data + 8B flag，带宽效率 120/128 = 93.75%**；MAX_NTHREADS 640 / ELEMS_PER_THREAD 120 / SHMEM_ELEMS_PER_THREAD 8 |
| `src/device/prims_ll128.h` L10 / L13 / L24 | **NCCL_LL128_FLAGTHREAD = LINEELEMS-1**（每 line 最后一个 thread 持 flag）/ Primitives<ProtoLL128> 特化 / flagThread（**LL128 需 NVLink 128B 原子性保证**，cost_model isLL128Enabled 门控） |
| `src/device/prims_simple.h` L20 / L104 / L179 | **Primitives<ProtoSimple<SlicePerChunk,StepPerSlice,Unroll,MultimemSrcs,MultimemDsts>>**（Simple 协议，无 flag，**proxy + memory fence 同步**，大消息满带宽）/ waitPeer / genericOp（send/recv/recvReduceSend/recvReduceCopySend 统一数据路径） |
| `src/include/device.h` L80 / L192 / **L304-310** | **MAXCHANNELS 64**（channel 并行上限）/ NCCL_MAX_NVLS_ARITY 32 / **ncclProtoGrainSize：LL 16B / LL128 warp-grain / SIMPLE 512B**（协议粒度） |
| `src/device/common.cu` L23 / `src/device/common.h` L303 / L420 / L29-31 | **ncclDevKernel_Generic**（__global__ 入口，args4K）/ **RunWorkBatch**（work 批处理模板）/ **ncclKernelMain**（SpecializedFnId dispatch）/ ncclDevFuncTable（func 指针表） |
| `src/device/generate.py` L16 / L41-42 | all_protos=["LL","LL128","SIMPLE"]；**per（coll × redop × type × algo × proto）实例化生成**（kernel 爆炸式特化，编译期 dispatch） |
| `src/nccl.h.in` L602 / L618 / L561 / L589 | ncclAllReduce / ncclReduceScatter / ncclReduce / ncclBroadcast 公共 API（8.1 接口表） |

## 5. NCCL docs / blog / 经典核实

### 5.1 NCCL User Guide（docs.nvidia.com/deeplearning/nccl/user-guide/docs/，200，**版本 2.32.3**，滞后 repo 2.32.7 注）

- **NCCL_ALGO**（env.html 逐字）：Ring / Tree（2.5+）、Collnet（2.5-2.13）、CollnetChain / CollnetDirect（2.14+）、**NVLS（2.17+）、NVLSTree（2.18+）、PAT（2.23+）**；「^」排除语法；2.24+ 分号分隔 per-function 语法；「NVLS and NVLSTree enable NVLink SHARP offload」。
- **NCCL_PROTO**（逐字）：LL / LL128 / Simple；「**enabling LL128 on platforms that don't support it can lead to data corruption**」；默认 LL,LL128,Simple（支持 LL128 平台）否则 LL,Simple。
- **NCCL_NVLS_ENABLE**（2.17+，逐字）：「NVLink SHARP is available in **third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures**, allowing collectives such as ncclAllReduce to be offloaded to the NVSwitch domain」。
- NCCL_TOPO_FILE / NCCL_TOPO_DUMP_FILE / NCCL_P2P_LEVEL / NCCL_P2P_DISABLE / NCCL_SHM_DISABLE / NCCL_BUFFSIZE / NCCL_NTHREADS / NCCL_MAX_NCHANNELS / NCCL_MIN_NCHANNELS / NCCL_NVLS_ENABLE / NCCL_MNNVL_ENABLE / NCCL_RAS_ENABLE 均在 env.html ✓（10.2 调优回链）。

### 5.2 NCCL 2.4 double binary tree blog（200，2019-02-04，Sylvain Jeaugey）

- 逐字：「Double binary trees rely on the fact that **half or less ranks in a binary tree are nodes and half (or more) ranks are leaves**. Therefore, we can build a second tree using leaves as nodes and vice-versa... There might be one rank which is a leaf on both trees but **no rank is a node on both trees**.」
- 逐字：「If we use each of the two trees to process half of the data, **each rank will at most receive half of the data twice and send half of the data twice, which is as optimal as rings in terms of data sent/received**.」→ **double binary tree 带宽最优（~2M 流量）一手**。
- 逐字：「NCCL automatically switches back to rings when that pattern results in greater bandwidth」→ ring/tree 实测交叉（8.3 β_eff 差）。
- Summit 24,576 GPUs 实测：延迟最高 **180× 改善** vs ring；跨 L3 IB switch 带宽略降（routing 交互）。
- **ref [6] = Peter Sanders, Jochen Speck, Jesper Larsson Träff (2009)「Two-tree algorithms for full bandwidth broadcast, reduction and scan」**→ 学术出处新增条目（§5.3）。

### 5.3 经典（Crossref API 200，2026-11-25，无 arXiv [B]）

| 文献 | 出处 | DOI | Crossref 核实 |
| --- | --- | --- | --- |
| Patarasuk & Yuan 2009 | Bandwidth optimal all-reduce algorithms for clusters of workstations，**JPDC 69(2):117-124**，2009-02 | 10.1016/j.jpdc.2008.09.002 | 标题/venue/卷期页/作者（Patarasuk, Yuan）✓（W0 已核，本轮再核 + 页码） |
| Rabenseifner 2004 | Optimization of Collective Reduction Operations，**ICCS 2004，Springer LNCS 3036，pp 1-9** | 10.1007/978-3-540-24685-5_1 | 标题/venue/页码/作者 ✓（W0 已核，本轮再核 + 页码） |
| **Sanders, Speck, Träff 2009（新增）** | Two-tree algorithms for full bandwidth broadcast, reduction and scan，**Parallel Computing 35(12):581-594**，2009-12 | 10.1016/j.parco.2009.09.001 | 标题/venue/卷期页/作者 ✓；前驱版 EuroPVM/MPI 2007「Full Bandwidth Broadcast, Reduction and Scan with Only Two Trees」LNCS 4757 pp 17-26（DOI 10.1007/978-3-540-75416-9_10） |

- 8.2 ring 带宽最优性下界证明（spec 强制重推导）：Patarasuk-Yuan 为 ring all-reduce 带宽最优经典锚；Rabenseifner 为 recursive halving/doubling 锚；**本章下界证明按 spec 命题-证明骨架独立重推，经典只作出处支撑**。
- 8.3 double binary tree：NCCL blog（工程披露）+ **Sanders 2009（学术出处）**+ tree.cc log2i 延迟项代码锚。

## 6. ⟨verify⟩ 销项 / 残留

| 项 | 状态 | 依据 |
| --- | --- | --- |
| NVLink 代际带宽（spec 7.1 ⟨verify 代际带宽〉） | **销** | 1.0-5.0 五份 whitepaper 一手（§2.1 表） |
| H100 18 links × 25 GB/s 单向 = 450 单向 = 900 双向（spec 7.1 ⟨verify datasheet〉） | **销** | Hopper wp p.47 逐字 + H100 datasheet NVLink 900GB/s + NVLink 产品页表 |
| B200 1.8 TB/s 双向（spec 7.1 ⟨verify〉） | **销** | brief p.10 逐字「1.8 TB/sec total bandwidth, 900 GB/sec in each direction」+ datasheet + 产品页 |
| NVSwitch crossbar H100 4 NVSwitch（spec 7.1） | **销** | NVLink/NVSwitch blog 逐字「four third-generation NVSwitch chips」+ 64 ports/chip（wp）+ 25.6 Tbit/s 双向/chip（blog） |
| B200 NVSwitch 4.0（spec 7.1 ⟨verify〉） | **销 + 命名订正** | 官方名 NVLink Switch / NVLink 5 Switch（第四代 NVSwitch chip）；NVL72 9 trays × 144 ports @ 100 GB/s（blog 逐字）；**per-chip 72 ports / 7.2 TB/s / 2 chips per tray 逐字未核 → 残留**（tray 级一手足够推导） |
| GB200 NVL72 精确规格（spec 7.1 ⟨verify〉） | **销** | 72 GPU / 36 Grace / 18 compute trays / 9 switch trays / 130 TB/s 双向 / 液冷（brief Table 2 + datasheet + partition guide + blog） |
| NVLink-C2C Grace-Hopper/Blackwell（spec 7.1） | **销** | GH 900 GB/s（Hopper blog 逐字）/ GB 900 GB/s bidi（GB200 blog 逐字）/ CHI/CXL + 6× 能效（C2C 页） |
| NV-HBI 10 TB/s die-to-die（spec 7.1） | **销**（W1 已锚再核） | brief p.7-8 逐字 |
| NVSwitch 产品页 URL（spec 7.1 Sources ⟨verify URL〉） | **销 + 漂移** | nvswitch/ **404** → 并入 nvlink/（「NVLink & NVLink Switch」，200） |
| GDS cuFile API（spec 7.3 ⟨verify〉） | **销** | cuFile API Reference v1.20：HandleRegister/Read/Write/DriverOpen/BufRegister |
| CUDA PG P2P 节（spec 7.3） | **销 + 漂移** | 13.5 → 03-advanced/multi-gpu-systems.html §3.4.2（旧子页 404）；ACS/IOMMU §3.4.2.5 逐字 |
| PCIe Gen5 x16 = 64 GB/s 单向（spec 7.2） | **销（数值口径）** | Hopper wp p.49 逐字 128 total / 64 per-direction + datasheet；**PCI-SIG 精确出处 403 残留不变** |
| NCCL 当前版本 2.x（spec 8.4 ⟨verify〉） | **销** | 2.32.7-1，commit 351f33fd（version.mk + release merge） |
| NCCL double binary tree blog URL（spec 8.3 ⟨verify URL〉） | **销** | 200 + 内容逐字（§5.2） |
| Patarasuk-Yuan / Rabenseifner DOI（spec 8.2） | **销** | Crossref 再核（§5.3） |
| cuda-samples P2P 例（spec 7.3） | **销 + 漂移** | p2pBandwidthLatencyTest 移除 → simpleP2P（§3.2） |
| PCI-SIG 精确出处（W0 残留 W4） | **残留** | pcisig.com 403 bot 拦截；数值口径经 NVIDIA wp/datasheet 闭合，规范 URL 留后续 |
| NVLink Switch per-chip 拆分（2 chips/tray、72 ports、7.2 TB/s） | **残留** | tray 级 144 ports @ 100 GB/s 一手；per-chip 逐字未核（writer 用 tray 级或注派生） |
| B200 SM 数（W1 残留 W3/W6） | **残留不变** | brief/datasheet 仍无 SM 数 |
| JEDEC HBM3 精确 URL（W1 残留） | **残留不变** | 403，与本波无关 |
| rdma-core+9.3 / deepep+10.4 citedBy（W3 残留 W5） | **残留不变** | 留 W5 |

## 7. 版本漂移注（writer 必注）

1. **NCCL 结构大漂移（2.32 vs spec 旧布局）**：`src/collectives/device/` → `src/device/`；`src/graph/tuning.cc` → `src/tuning/`（per-algo cost model）；`src/enqueue.cc` → `src/enqueue/`；`devcomm.h` → `device.h` + `plugin/nccl_tuner.h`。**writer 8.4 引代码一律用 §4 现行路径 + 版本锚 351f33fd**；spec 大纲中的旧路径名须翻译。新增 PAT/NVLS_TREE 算法、symmetric kernels/GIN/RAS/CE 注存在即可（走读 scope 以 ring/tree/NVLS/CollNet + LL/LL128/Simple 为主）。
2. **NVSwitch 命名**：Hopper = 第三代 NVSwitch（NVLink 4 Switch）；Blackwell = **NVLink Switch / NVLink 5 Switch**（第四代 NVSwitch chip）。spec「NVSwitch 4.0」非官方命名，writer 用官方名 + 代际注。NVSwitch 独立产品页 404，规格并入 NVLink 页。
3. **NVLink 产品页 Rubin 主导 + HTML/.md 漂移**：现页主推 NVLink 6（Vera Rubin）；HTML 3 TB/s / 216 TB/s（preliminary）vs .md 3.6 TB/s / 260 TB/s。**本波只用 NVLink 4/5 列（两版一致）**；Rubin 数值勿入正文（非 scope，且 preliminary）。
4. **NVLink Switch System 256-GPU（Hopper）宣布未出货**：Hopper wp 逐字「not available with H100 systems now」；首个出货多节点 NVLink 域 = GB200 NVL72。writer 7.1 历史注勿混。
5. **NVLink 带宽模型值 vs raw**：NCCL topo.h SM90_NVLINK_BW **20.6** / SM100 **40.1** GB/s（per-link 单向有效模型值）vs 硬件 raw 25/50 GB/s——8.4 走读注「NCCL 内部带宽模型含协议/编码折扣」，勿与 7.1 硬件规格混。
6. **NVSwitch  throughput 口径**：Hopper wp「13.6 Tbits/sec」（switch throughput 口径）vs blog「25.6 Tbit/s 双向」（=3.2 TB/s 有效双向）。writer 用 blog 口径并注 wp 原文。
7. **CUDA PG 13.5 P2P 子页重组**：旧 peer-to-peer-memory-access-and-transfers.html 404 → **03-advanced/multi-gpu-systems.html §3.4.2**（W2 已注 13.x 子页漂移，本轮 P2P 节再重组）。
8. **cuda-samples p2pBandwidthLatencyTest 移除**：P2P 教学例改 simpleP2P；带宽矩阵实测改 nvbandwidth（W1 已锚）。
9. **NCCL docs 版本滞后**：docs 2.32.3 vs repo 2.32.7-1——env/算法语义一致，版本注。
10. **Grace Hopper 产品页 404 / resources geo-redirect**：C2C 900 GB/s 数值锚 Hopper blog（GH）/ GB200 blog（GB），不锚已 404 产品页。
11. **NVLink 1.0 年份**：2016（P100 出货，Volta wp）vs 2014（GTC 宣布，GB200 blog）——writer 用 2016 并注。
12. **CUTLASS load L310（W3 残留订正）**：sm90_mma_tma_gmma_ss_warpspecialized.hpp `load(` 在 **L310**（contents API at 5c179edeb3e2 再核），W3 refs/research 误标 L312 已双修；章 5.2 L310 原本正确。

## 8. references 写入 / 被引回写（spec §5 Sources）

- **新增 7 条**：P7 [C] × 6——pascal-whitepaper / volta-whitepaper / ampere-whitepaper（NVLink 1.0/2.0/3.0 一手）+ nvlink-nvswitch-blog（4 NVSwitch + 25.6 Tbit/s + 非阻塞 + P2P 对照，被引 7.1, 7.2）+ gb200-blog（C2C 900 GB/s + switch tray 144 ports + NVL72 构成，被引 7.1）+ nvlink-c2c（C2C 机制页，被引 7.1）；P8 [B] × 1——sanders-two-tree（double binary tree 学术出处，被引 8.3）。
- **更新条目（核实日期 2026-11-25）**：P7 nvlink-nvswitch（代际/域规格表 + nvswitch 404 漂移）/ gb200-nvl72（72/36/18/9/130 TB/s + partition guide URL）/ hgx-dgx（HGX datasheet NVSwitch 行 + B200 14.4 TB/s）/ pcie-spec（403 残留 + wp 互证）/ gpudirect-rdma-docs（家族页 + IOMMU/root complex/BAR1/peermem）/ gds-docs（cuFile v1.20 API URL）；P8 nccl（2.32.7 版本锚 + 结构漂移 + 代码锚摘要）/ nccl-docs（2.32.3 + ALGO/PROTO/NVLS_ENABLE 逐字）/ nccl-tree-blog（内容逐字 + Sanders ref）/ patarasuk-yuan（+页码 117-124）/ rabenseifner（+页码 1-9）；跨 Part：hopper-whitepaper（NVLink4/NVSwitch3/PCIe Gen5 节，被引 += 7.1, 7.2）/ blackwell-whitepaper（NVLink5/NV-HBI/NVL72 Table 2，被引 += 7.1, 7.2）/ nvidia-datasheet（**INT8 补全** + NVLink/PCIe interconnect 行）/ hopper-blog（C2C 900 GB/s，被引 += 7.1）/ cuda-programming-guide（multi-gpu-systems §3.4.2 P2P + ACS/IOMMU）/ cuda-samples（simpleP2P 路径 + p2pBandwidthLatencyTest 移除注）/ **cutlass（load L312→L310 W3 残留修）**。
- **W3 残留修**：cutlass entry load L312→L310（本文件 + gpu-w3-sources.md §3.1/§7 双修）；nvidia-datasheet INT8（H100 SXM 3958*→1979 dense TOPS / HGX H100 8-GPU 32 POPS*→16 dense / HGX B200 9 POPS sparse→4500 TOPS dense / GB200 10 POPS→5000 dense；Blackwell datasheet 脚注 2「Dense is one-half of the sparse spec shown」）。
- **被引回写（spec §5 Sources）**：7.1 = nvlink-nvswitch / pascal-whitepaper / volta-whitepaper / ampere-whitepaper / hopper-whitepaper / blackwell-whitepaper / nvidia-datasheet / nvlink-nvswitch-blog / gb200-blog / nvlink-c2c / gb200-nvl72 / hopper-blog；7.2 = hgx-dgx / nvbandwidth / pcie-spec / nvidia-datasheet / nvlink-nvswitch-blog / hopper-whitepaper / blackwell-whitepaper；7.3 = cuda-programming-guide / gpudirect-rdma-docs / gds-docs / cuda-samples（均已含）；8.1 = nccl-docs / nccl；8.2 = patarasuk-yuan / rabenseifner / nccl；8.3 = nccl-tree-blog / rabenseifner / nccl / nccl-docs / sanders-two-tree / nccl-tests；8.4 = nccl / nccl-docs / nccl-tree-blog。跨 Part 共享锚点被引跨组累加（hopper-whitepaper/blackwell-whitepaper/hopper-blog += 7.1/7.2）。
- **未重核条目维持原核实日期**：nvbandwidth（W1 2026-11-25）/ nccl-tests（W0 2026-11-24，8.3/10.2/11.2 被引维持）。
- parse-check：11 groups / **76 entries**（69 + 7）。

## 9. writer 白名单提示（T2-T5）

- **7.1 NVLink/NVSwitch 架构**：代际表 §2.1（五份 whitepaper [C]，方向口径硬规则）；NVSwitch §2.2（Hopper 4 chips/64 ports/3.2 TB/s 双向 blog 逐字 + NVLink SHARP wp 逐字；Blackwell NVLink Switch 9 trays × 144 ports @ 100 GB/s blog 逐字）；域聚合账 §2.3（7.2/14.4/130 TB/s）；C2C vs NV-HBI §2.4（900 GB/s vs 10 TB/s 分层）；vs PCIe 7×/14×（wp 逐字）。**关键推导**：域聚合 = n_GPU × per-GPU 双向；全互连 vs switch（非阻塞 vs 每对 128 GB/s）。
- **7.2 节点内拓扑与带宽账**：HGX 8-GPU 4 NVSwitch（blog + datasheet）；P2P 对照账 blog Table 1（128 vs 900 GB/s）；PCIe Gen5 x16 64 GB/s 单向（Hopper wp 逐字）；nvbandwidth 实测方法（W1 锚）；NUMA/PHB/PXN 路径（NCCL paths.cc 对照，8.4 回链）。training 4.1「NVLink/IB 有效带宽硬件侧」errata 重判到本章。
- **7.3 GPUDirect P2P**：CUDA PG multi-gpu-systems §3.4.2（cudaDeviceCanAccessPeer/EnablePeerAccess/cudaMemcpyPeer + 8 peer connections 上限 + UVA）；**ACS/IOMMU §3.4.2.5 逐字**；GDR docs root complex/BAR1/peermem；GDS cuFile v1.20（HandleRegister/Read/Write）；GPUDirect 家族页（P2P/RDMA/Storage/Video）；cuda-samples simpleP2P；bounce buffer 省 H2D+D2H 两跳账。**RDMA → 9.2 指针**。
- **8.1 集合原语与消息量账**：NCCL API（nccl.h.in L602/618/561/589）；NCCL docs 原语语义；training 4.1 式 (2)(3) recap（d→n 翻译，不重推）；原语→硬件映射（NVLink 域内 vs 跨机 RDMA，transport 表 §4.4）；α-β 章首声明（非 inference 接受率）。
- **8.2 Ring 算法【重】**：training 4.1 ring 消息量 recap（归纳证明已交付不重推）；**带宽最优性下界证明按 spec 命题-证明骨架独立重推**（chunk 归约树 + 广播树两类不相交传输 ≥ 2(n-1)M/n per chunk，单阶段界 (n-1)/n·M 单独归属）；Patarasuk-Yuan [B]（JPDC 69(2):117-124）出处支撑；Rabenseifner [B] recursive halving 对照；NCCL rings.cc ncclBuildRings L29 + ring.cc cost model（α-β 代码形态 ncclTuningGetTime）；chunking/pipelining 摊薄 + 多 ring 并行（MAXCHANNELS 64）。
- **8.3 Tree / 双二叉树**：NCCL 2.4 blog 逐字（双树互补 + ~2M 带宽最优 + 180× 延迟 + 自动切回 ring）；**Sanders 2009 [B] 学术出处**；trees.cc ncclGetDtree L90（mirror even/shift odd）；**tree.cc log2i(nNodes) 延迟项 + busBw×0.5 + treeCorrectionFactor**（理想 α-β 带宽项相同无交叉 recap training 4.1，真实交叉来自 β_eff 差——禁从纯 α-β 凭空解出 M*）；Rabenseifner 2^log n 步对照。
- **8.4 NCCL 实现走读【模块全量】**：**版本锚 2.32.7-1 / 351f33fd + §4 全量 file:line**；结构漂移注（§7.1）；init 流程（ncclCommInitRank L3071 → bootstrap → topo → graph search → transport → devComm）；channel/ring/tree 构建（graph/）；transport P2P/SHM/NET + proxy（sendProxyProgress L1327 / ncclProxyProgress L956）；协议 LL（16B line 50%）/ LL128（128B line 93.75%）/ Simple（512B grain，proxy+fence）；算法选择（ncclTuningSelectBestTuning 最小 timeUs + NCCL_ALGO/PROTO env）；NVLS（cuMulticastCreate L270 + all_reduce.h L390 NVSwitch 内归约）；CollNet（SHARP）。**与 10.2 两层纪律：本章代码机制，10.2 fabric 工程**。
