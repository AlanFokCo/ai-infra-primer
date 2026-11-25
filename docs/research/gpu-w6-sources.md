# gpu W6 research pass — P11（系统集成与生产）一手来源核实

Task 1：P11（11.1 服务器形态 MIG / 11.2 Profiling 调优 / 11.3 故障可靠性）一手来源核实 + **B200 SM 数 ⟨verify⟩ 销项**（W1 残留）+ `gpu/references.md` P11 扩充（6 条精炼 + 4 条新增）+ 跨 Part 被引回写。本波为最后一内容波（3 章，断言翻 {41,11}），W7 全书 sweep 收官。writer 引用白名单 = `gpu/references.md` P11 条目 + 跨 Part 共享锚点（P1 hopper-whitepaper / blackwell-whitepaper / nvidia-datasheet / nsight-compute，P2 nvbandwidth，P7 hgx-dgx / gb200-nvl72，P8 nccl-tests）。

## 0. 摘要

- **B200 SM 数销项（W1→W3→W6 残留）**：**148 SMs**（2 die × 74 enabled；物理 2×80=160；8 GPC / 4 L2 分区 / 8 HBM3e 栈）。一手 = **[B] arXiv:2512.02189v3**（Jarmusch & Chandrasekaran，U. Delaware，2025-12-01 / 2026-03-02）逐字「148 SMs spread across eight GPCs... four L2 cache partitions... eight HBM3e memory stacks」；**Chips and Cheese 2025-12-15 实测**（8×B200 Verda 实例，媒体线索/互证非事实源）「Each B200 die physically contains 80 SMs... B200 enables 74 SMs per die, giving 148 SMs across the GPU」；**官方 Inside Blackwell Ultra blog [C]** 披露 full config「up to 160 SMs... eight GPCs... 640 Tensor Cores」+「Available SM count... varies by SKU」（B200 为 harvest SKU）。**官方 B200 datasheet/brief 仍不印 SM 数**（本轮重取确认）——148 为微基准论文 + 实测口径，非 datasheet 数字；1.1 反推自洽（FP32 75TF = 148×256×1.98GHz）获外部确认。**残留**：官方未直接披露 B200 SM 数（writer 引 2512.02189 [B] 注口径）；论文内部「SM count 1.09×」vs H200 与 148/132=1.12 不自洽（论文自注 decomposition 未实测）；Cornell CAC 页不可达（000）。
- **MIG 几何全表一手**（MIG User Guide，2026-11-04 更新）：memory slice = 1/8 显存（容量+带宽+MC+cache）、SM slice = 1/7 SM；GI = slices + engines（memory QoS）、CI = GI 内 SM 子集（共享 memory/engine）；**最多 7 实例**（A100/H100/H200/B200/GB200/B300/GB300），**A30 = 4**；A100/H100/H200/B200/GB200/B300/GB300 profile 表逐字（§2）；streams/MPS/MIG 对比表（物理分区/7/SM 隔离/BW QoS/错误隔离）；**Hopper+ enable 无需 reset 且不持久（无 InfoROM bit）vs Ampere 需 reset 且持久**；P2P between MIG R570+；命令 -mig 1 / -lgip / -lgipp / -cgi -C。
- **XID 语义全核**（XID Errors Release 620，2026-11-11 + 官方 Xid-Catalog.xlsx 五 worksheet）：31/43/48/63/64/74/79/94/95/45/13/119/120/143/149/154 逐码（§3）；**版本漂移硬注：XID 74 NVLINK_ERROR 不适用 B-x00/GB-x00/Rubin（Blackwell NVLink5 → XID 149 NETIR + 144-150 decode）**；SXid 仅 Hopper 及更早；经典归因表（archive：HW/Driver/UserApp/SysMem/Bus/Thermal/FB 七列）+ 新 catalog Tenant In-Operation/Investigatory Action + Guidance Classes（WORKFLOW_XID_31/13/DBE/NVLINK_ERR/NVLINK5_ERR）。
- **ECC / row remapping 一手**（GPU Memory Error Management，2026-11-11 更新）：error containment（Ampere+，UE 限受影响 app）/ dynamic page offlining / **row remapping（每 bank 硬件 spare rows，512 remaps vs legacy 64 retirements，GPU reset 生效，终身持久）** / **RAS Repair（Blackwell，HBM channel + L2 slice spare 替换，同 bank 2 次 remap 后第 3 次 UE 触发，XID 160，reboot 生效）** / RMA policy（8 UE rows/bank、已 remap row 再 remap、512 total；SRAM parity >4 / SECDED >2 UCE/bank）；**Hopper wp p.38 SECDED + Sideband ECC（HBM）+ L2/L1/RF SECDED + boot-time row remap 逐字重取**（recap 锚 2.1 §4，非 brief 误写 1.2）；nvidia-smi ECC volatile/aggregate + SRAM parity/SEC-DED 分类 + sources histogram（L2/SM/microcontroller/PCIE）+ repair flags + recovery actions。
- **服务器形态供电散热一手**：HGX H100/H200 datasheet（Jul24）8-GPU baseboard 规格表；**DGX H100/H200 = 8U / 130.45kg / 10.2kW max / 6×3.3kW PSU 4+2**；**DGX B200 = 10U / 142.4kg / 14.3kW max / 5+1 / 2×NVLink5 switch 14.4TB/s**；**GB200 NVL72 机架 ≈120kW**（DGX GB200 guide：8 power shelves ×6×5.5kW PSU N+N / 33kW/shelf / busbar 50-51V DC / 液冷 cold plate + leak detection）；**NVLink switch tray = 2×72-port switch / 57.6 Tbps full duplex**（销 W4「per-chip 72 ports/2 chips per tray 逐字未核」残留）；Blackwell datasheet GB200 1200W / HGX B200 1000W / GB300 1400W / B300 1100W + RAS engine + TEE-I/O。
- **Profiling 一手**：Nsight Systems **v2026.6**（NVTX range 投影 GPU timeline / cudaProfilerStart/Stop / CUDA API trace / CPU IP sampling / --gpu-metrics-devices/frequency/set / profiling overhead 注）；Nsight Compute ProfilingGuide **GPU Speed Of Light + Hierarchical Roofline Chart（Double/Half/Single）**（11.2 roofline 实测回链 2.4）；nvbandwidth / nccl-tests / DCGM repo 200 重核；**DCGM diag r1-r4 测试成员表逐字**（r1 software / r2 +pcie/memory/memory_bandwidth / r3 +diagnostic/targeted_stress/targeted_power/nvbandwidth/nccl_tests / r4 +memtest/pulse）+ field IDs（ECC 310-313 / SRAM 345-346 / row remap 393-396 / NVLink CRC 400-404 / replay 420-429 / recovery 430-435）。
- **新增 4 条**：P11 ecc-memory-mgmt / inside-blackwell-blog / dgx-gb200-guide（[C]）+ blackwell-b200-microbench（[B]）。**跨 Part 回写**：hgx-dgx / gb200-nvl72 扩供电散热 + 销 ⟨verify W6⟩；nvidia-datasheet += Ultra datasheet；blackwell-whitepaper SM 残留注记销 + 11.1；hopper-whitepaper += ECC p.38 + 11.3；nsight-compute += SOL roofline；nvbandwidth / nccl-tests 日期重核。
- **残留**：B200 SM 官方 datasheet 未印（[B] 口径注）；JEDEC 403 / PCI-SIG 403（W1/W4 不变，P11 用已核数值）；Al-Fares PDF（W5 不变，P11 不涉）；Cornell CAC B200 页不可达（000，非关键）。

## 1. 核实通道与网络 fallback（本轮实际使用）

- **docs.nvidia.com 稳定 200**：MIG User Guide（landing + concepts/supported-gpus/supported-mig-profiles/virtualization/getting-started 子页，**2026-11-04 更新，多页重组**）；XID Errors（landing + analyzing-xid-catalog/working-with-xid-errors + **pdf/XID_Errors.pdf 287178 B Release 620** + **_downloads/Xid-Catalog.xlsx 388889 B 五 worksheet** + archive/index.html 经典归因表）；GPU Memory Error Management（a100-gpu-mem-error-mgmt landing + overview/row-remapping/error-containment/dynamic-page-offlining/hbm-channel-repair[title RAS Repair]/rma-policy-thresholds-for-row-remapping[title RMA Policy]/sram-uncorrectable-errors 子页，**2026-11-11 更新**）；nvidia-smi docs（235840 B）；Nsight Systems UserGuide（1314852 B v2026.6）；Nsight Compute NsightCompute + ProfilingGuide（W2 锚重核 200）；DCGM user-guide（feature-overview/dcgm-diagnostics）+ dcgm-api/dcgm-api-field-ids（543229 B）；vGPU 17.0 grid-vgpu-user-guide（3389990 B）；DGX H100/H200 user-guide introduction-to-dgxh100；DGX B200 user-guide introduction-to-dgxb200；**DGX GB200 user-guide hardware.html**（docs.nvidia.com/dgx/dgxgb200-user-guide/，title「DGX GB Rack Scale Systems User Guide」，覆盖 GB200/GB300）。
- **dam-cdn.nvd.orangelogic.com 官方 DAM CDN PDF 一手**（沿 W1/W4）：Hopper wp 705n6ur5（7880278 B，ECC p.38 重取）；Blackwell brief gl2l4l48（1268171 B，GB300 更新版）；Blackwell datasheet 01dyp27s（1156321 B，Oct25）；HGX H100/H200 datasheet 44wa8u63（312368 B，Jul24）；**Blackwell Ultra datasheet 1k0p832e（438824 B，GB300 1400W/B300 1100W）**。
- **developer.nvidia.com blog 200**：**Inside NVIDIA Blackwell Ultra**（inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era，160 SMs full config + RAS + TEE-I/O + GB300 NVL72 power shelf）。
- **GitHub API 200**（用量 ~5）：NVIDIA/nvbandwidth（+ readme raw）/ NVIDIA/nccl-tests / NVIDIA/DCGM 重核 200。
- **arXiv export API 200**：2512.02189（ID↔标题↔作者 Jarmusch/Chandrasekaran，v3 2026-03-02）；**arxiv.org/html/2512.02189v3 200**（271328 B）。
- **chipsandcheese.com curl 200**（WebFetch 域名拦截 → curl + UA fallback）：B200 实测文（nvidias-b200-keeping-the-cuda-juggernaut，2025-12-15）+ GB202 文（blackwell-nvidias-massive-gpu，2025-06-29，**GB202 188 SM 非 B200，勿混**）。
- **nvidia.com 产品页 200**：gb200-nvl72/（液冷定性 + 25× same power，无 rack kW 数字）；resources.nvidia.com/en-us-blackwell-architecture（AssetLink 入口）。
- **404/000**：docs.nvidia.com/deploy/row-remapping/index.html **404**（实在 a100-gpu-mem-error-mgmt/row-remapping.html）；images.nvidia.com dgx-h100-datasheet.pdf **404**（DGX 规格经 user-guide 一手）；docs.nvidia.com/dgx/dgxgb200-user-guide/introduction-to-dgxgb200.html **404**（实在 hardware.html）；resources.nvidia.com/en-us-dgx-systems/dgx-h100-datasheet + en-us-gb200-nvl72-datasheet **通用 JS shell 页**（无 AssetLink，同 byte-size 724748）；**cvw.cac.cornell.edu 000 连接失败**（WebFetch 亦拦截，B200 SM 非关键互证源）；techpowerup.com **bot 拦截页**（媒体线索，不作事实源）。
- 工具：`/usr/bin/grep -n`；`pdftotext -layout`；`python3 -I`（HTML 去标签 / xlsx zipfile+XML 解析 / GitHub JSON）；下载缓存 `/tmp/gpu-w6-research/`。

## 2. MIG 几何与 profile 表（11.1，[C] MIG User Guide 2026-11-04）

### 2.1 概念逐字（concepts.html）

- **GPU memory slice**：「the smallest fraction of the GPU's memory, including the corresponding memory controllers and cache... roughly **one eighth** of the total GPU memory resources, including both capacity and bandwidth」。
- **GPU SM slice**：「the smallest fraction of the SMs... roughly **one seventh** of the total number of SMs available in the GPU when configured in MIG mode」。
- **GPU Instance (GI)**：「a combination of GPU slices and GPU engines (DMAs, NVDECs, and so on)... provides **memory QoS**... Each GPU memory slice gets 1/8 of the total GPU memory resources and each GPU SM slice gets 1/7 of the total number of SMs」。
- **Compute Instance (CI)**：「a subset of the parent GPU instance's SM slices and other GPU engines... The CIs **share memory and engines**」（每 CI 专属 SM）。
- A100-40GB 示例：8×5GB memory slices + 7 SM slices；1g.5gb = 1 memory + 1 compute slice；4g.20gb = 4+4；CI 可再分（2c.4g.20gb 等）。
- **Table 3 CUDA Concurrency Mechanisms**：Streams（单进程/逻辑/无限/无隔离）vs MPS（逻辑/48/SM 百分比非分区/无错误隔离）vs **MIG（物理/7/SM 性能隔离/内存保护/BW QoS/错误隔离/limited IPC/when idle 重配）**。

### 2.2 GPU Instance Profile 表（supported-mig-profiles.html，逐字）

> 列：Profile / Fraction of Memory / Fraction of SMs / L2 Cache / Copy Engines / Number of Instances。NVDEC/JPEG/OFA 见各表。**L2 随 memory slice 切分（1/8 per slice）**。

**A100-SXM4-40GB（Table 15；80GB 名按内存比例 1g.10gb/2g.20gb/3g.40gb/4g.40gb/7g.80gb）**：

| Profile | Memory | SM | L2 | CE | Instances |
| --- | --- | --- | --- | --- | --- |
| 1g.5gb | 1/8 | 1/7 | 1/8 | 1 | 7 |
| 1g.5gb+me | 1/8 | 1/7 | 1/8 | 1 | 1（media ext） |
| 1g.10gb | 2/8 | 1/7 | 2/8 | 1 | 4 |
| 2g.10gb | 2/8 | 2/7 | 2/8 | 2 | 3 |
| 3g.20gb | 4/8 | 3/7 | 4/8 | 3 | 2 |
| 4g.20gb | 4/8 | 4/7 | 4/8 | 4 | 1 |
| 7g.40gb | Full | 7/7 | Full | 7 | 1 |

**H100 80GB（Table 13，PCIe/SXM5）**：1g.10gb（1/8 mem，1/7 SM，L2 1/8，CE 1，7）/ 1g.10gb+me / **1g.20gb（1/4 mem，1/7 SM，L2 1/8，CE 1，4）**（非对称 profile）/ 2g.20gb（2/8，2/7，L2 2/8，CE 2，3）/ 3g.40gb（4/8，3/7，L2 4/8，CE 3，2）/ 4g.40gb（4/8，4/7，L2 4/8，CE 4，1）/ 7g.80gb（Full，7/7，Full L2，**CE 8**，1；7 NVDEC/7 JPEG/1 OFA）。H100 94GB（1g.12gb/2g.24gb/3g.47gb/4g.47gb/7g.94gb）/ 96GB on GH200（3g.48gb/4g.48gb/7g.96gb）同构。

**H200 141GB（Table 14）**：1g.18gb / 1g.35gb / 2g.35gb / 3g.71gb / 4g.71gb / 7g.141gb（CE 1/1/2/3/4/8）。

**B200 180GB（Table 8）**：1g.23gb（1/8，1/7，L2 1/8，**CE 2**，7）/ 1g.23gb+me / 1g.45gb（2/8，1/7，CE 2，4）/ 2g.45gb（2/8，2/7，**CE 3**，3）/ 3g.90gb（4/8，3/7，CE 6，2）/ 4g.90gb（4/8，4/7，CE 8，1）/ 7g.180gb（Full，Full，**CE 16**，1；**1 NVDEC/1 JPEG/1 OFA**——B200 media engine 少于 H100）。

**GB200 186GB（Table 7）**：1g.23gb / 1g.47gb / 2g.47gb / 3g.93gb / 4g.93gb / 7g.186gb（CE 2/2/4/6/8/16；NVDEC 1/1/2/3/4/7，OFA 0/0/0/0/1/2）。

**B300 288GB（Table 6）**：1g.34gb / 1g.67gb / 2g.67gb / 3g.135gb / 4g.135gb / 7g.269gb（CE 2/2/4/6/8/16）。**GB300 252GB（Table 5）**：1g.31gb / 1g.63gb / 2g.63gb / 3g.126gb / 4g.126gb / 7g.251gb。

**A30 24GB（Table 16）= 4 slices（1/4 mem，1/4 SM），max 4 实例**（1g.6gb/2g.12gb/4g.24gb）——「最多 7」不适用 A30。

### 2.3 支持 GPU / 部署 / 虚拟化

- **supported-gpus.html Table 1**：MIG 自 Ampere（CC≥8.0）；B200/GB200 = **GB100 microarch，CC 10.0**，180/186GB，max 7；B300/GB300 = GB110，CC 10.3，288/252GB，7；H100/H200 = GH100 CC 9.0，7；A100 = GA100 CC 8.0，40/80GB，7；A30 = GA100 CC 8.0，24GB，**4**；RTX PRO 6000 Blackwell = GB202 CC 12.0，4。
- **P2P between MIG instances on same GPU：driver R570+**（profile diagrams P2P 列 = nvidia-smi per-profile flag）。
- **getting-started-with-mig.html**：`nvidia-smi -i <GPU> -mig 1` 启用；**Hopper+ 启用无需 GPU reset，MIG mode 不跨 reboot 持久（无 InfoROM status bit），driver reload 即 disable**；**Ampere 需 reset，跨 reboot 持久（InfoROM bit）**；monitoring agent（DCGM/nvsm）占 GPU 阻 reset 须先 stop；`nvidia-smi mig -lgip`（列 GI profiles）/ `-lgipp`（列 placements）/ `-cgi <profiles> -C`（建 GI+CI）；placement 不足报 Insufficient Resources。
- **virtualization.html**：MIG + GPU pass-through（bare-metal 同 workflow）；MIG + vGPU（多 vGPU/VM 并行单 MIG GPU，保 vGPU 隔离；Host MIG Provisioning 见 AI Enterprise docs）；technical brief TB-10226-001_v01（MIG+vCS GPU Partitioning，Nov 2020，A100 时代）。

### 2.4 vGPU / SR-IOV 硬件虚拟化（11.1，[C] vGPU 17.0 User Guide）

- **架构**（§2.1）：NVIDIA Virtual GPU Manager 跑 hypervisor；guest driver 直访 GPU fast path + paravirtualized 管理接口；每 vGPU 固定 framebuffer（创建时从物理 FB 划出，独占至销毁）。
- **time-sliced vGPU**（§2.1.1.1）：物理 GPU 未分区；所有 time-sliced vGPU **共享 engines**（3D/decode/encode）；进程序列化调度，运行时独占 engines；FRL（B-series 45FPS / Q/A 60FPS）。
- **MIG-backed vGPU**：MIG-capable GPU + **NVIDIA AI Enterprise**（vGPU 17.0 note：MIG GPU 仅 AI Enterprise 支持）；每 GI 一个 vGPU/VM。
- **SR-IOV**（§2.2/2.10.2）：Ampere+ 物理 GPU 支持 SR-IOV；BIOS 须启 **VT-d/IOMMU + SR-IOV + ARI**；`/usr/lib/nvidia/sriov-manage -e <domain:bus:slot.func>` 启 VF（**禁手动 sysfs**）；`ls /sys/bus/pci/devices/.../ | grep virtfn`（virtfn0→0000:41:00.4 等）；**每 VF 仅一个 mdev device**；mdev VFIO（legacy）vs vendor-specific VFIO（Ubuntu 24.04+）；MIG-backed vGPU mdev 不跨 reboot 持久（GI 消失）。
- **A100 whitepaper SR-IOV**（W4 已锚 ampere-whitepaper）互证硬件代际。

## 3. XID 错误码语义（11.3，[C] XID Errors Release 620 + Xid-Catalog.xlsx）

### 3.1 关键码逐字（XIDs worksheet：Mnemonic / Applies A100/H-x00/B-x00/GB-x00/Rubin / In-Op / Investigatory / Trigger / Desc）

| XID | Mnemonic | 语义（Desc） | 适用 | In-Op / Investigatory | 归因要点（Trigger） |
| --- | --- | --- | --- | --- | --- |
| 13 | ROBUST_CHANNEL_GR_EXCEPTION / GR_ERROR_SW_NOTIFY | Graphics Engine Exception | 全 YES | RESTART_APP / WORKFLOW_XID_13 | 一般 user app fault（越界/非法指令/非法寄存器）；罕见 HW/系统软件 bug；cuda-gdb / Compute Sanitizer memcheck |
| 31 | ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT | **GPU memory page fault** | 全 YES | RESTART_APP / WORKFLOW_XID_31 | MMU 报 fault（非法地址访问）；通常 app-level bug，亦可 driver/HW bug；重复同 GPU → RUN_DCGMEUD，重复异 GPU/单次 → CHECK_APP/CUDA |
| 43 | ROBUST_CHANNEL_RESETCHANNEL_VERIF_ERROR | **GPU stopped processing** | 全 YES | IGNORE / REPORT_ISSUE | user app 触发 software induced fault 须终止；**GPU 保持健康**；多数非 driver bug 乃 user app error |
| 45 | ROBUST_CHANNEL_PREEMPTIVE_REMOVAL | Preemptive cleanup, due to previous errors | 全 YES | WORKFLOW_XID_45 | app abort（Ctrl-C/sigkill/GPU reset）驱动 teardown；多 CUDA app 撞 DBE 常见；多非 bug |
| 48 | ROBUST_CHANNEL_GPU_ECC_DBE | **Double Bit ECC Error** | 全 YES | WORKFLOW_XID_DBE | GPU 检测 uncorrectable error，回报 user app；持续/复发 CONTACT_SUPPORT；SRAM DBE 查 sram_threshold_exceeded flag |
| 63 | INFOROM_DRAM_RETIREMENT_EVENT | **GPU memory remapping event** | 全 YES | RESET_GPU_DEFERRED | ECC 处理记录；Ampere+ row remapper activity / 更早 dynamic page retirement；**给定 ECC 事件仅见 63/64/177 之一** |
| 64 | INFOROM_DRAM_RETIREMENT_FAILURE | **GPU memory remapping failure** | 全 YES | RESET_GPU / CONTACT_SUPPORT | remap 失败记录 |
| 74 | NVLINK_ERROR | NVLink Error（Desc 列标 Unused，Trigger 存） | **A100 YES / H-x00 YES / B-x00 NO / GB-x00 NO / Rubin NO** | WORKFLOW_NVLINK_ERR / REPORT_ISSUE | GPU↔GPU/NVSwitch NVLink 连接问题；可能 link HW failure 或远端 device 问题（远端 GPU fail 可致本端 XID 74）；nvidia-smi nvlink 查；**Blackwell 不适用 → 149** |
| 79 | ROBUST_CHANNEL_GPU_HAS_FALLEN_OFF_THE_BUS | **GPU has fallen off the bus** | 全 YES | CONTACT_SUPPORT | driver 经 PCIe 访 GPU 不可达；常 PCIe link HW failure 致 link down；查 system/kernel PCI event log；亦可 GPU HW fail/driver issue |
| 94 | ROBUST_CHANNEL_CONTAINED_ERROR | Contained memory error | 全 YES | RESTART_APP / IGNORE | Ampere+ error containment；**限单 app**，须重启该 app，余 app 不受影响；方便时 reset GPU；伴 XID 45 |
| 95 | ROBUST_CHANNEL_UNCONTAINED_ERROR | Uncontained memory error | 全 YES | RESET_GPU / IGNORE | **影响多 app**，affected GPU 须 reset 方可重启 app；伴 XID 45 |
| 119 | GSP_RPC_TIMEOUT | GSP RPC Timeout | 全 YES | RESET_GPU / REPORT_ISSUE | GSP core 错误/RPC 超时；**连续 3 次 RPC timeout → XID 154** |
| 120 | GSP_ERROR | GSP Error | 全 YES | RESET_GPU / INVESTIGATE_SW | GSP core 代码错误 |
| 143 | GPU_INIT_ERROR | GPU Initialization Error | A100 NO / 余 YES | RESET_GPU / CONTACT_SUPPORT | Hopper+ |
| 149 | NVLINK_NETIR_ERROR | NVLINK: NETIR Error | **A100 NO / H-x00 NO / B-x00 YES / GB-x00 YES / Rubin YES** | WORKFLOW_NVLINK5_ERR | **Blackwell NVLink5 错误**（144-150 decode worksheet） |
| 154 | GPU_RECOVERY_ACTION_CHANGED | GPU Recovery Action Changed | 全 YES | XID_154 | 伴随他 XID 总结 recovery action：「None / Drain P2P / Drain and Reset / GPU Reset Required / Node Reboot Required」 |

### 3.2 经典归因表（archive/index.html，七列 HW/Driver/UserApp/SysMem/Bus/Thermal/FB）

- 31 = HW+Driver+UserApp；43 = Driver+UserApp；48 = **HW only**；63 = HW+Driver+FB Corruption；64 = HW+Driver；74 = HW+Driver+Bus；79 = HW+Driver+SysMem+Bus+Thermal；94/95 = HW+Driver+FB。
- 新 catalog（Release 620）改用 **Tenant In-Operation Action / Tenant Investigatory Action** + Guidance Classes（RESET_GPU/RESTART_APP/RESTART_BM/IGNORE/CONTACT_SUPPORT/REPORT_ISSUE/WORKFLOW_XID_*/WORKFLOW_NVLINK_ERR/NVLINK5_ERR/XID_154）。
- **SXid（NVSwitch）仅 Hopper 及更早**（Blackwell NVL72 另机制，Fabric Manager User Guide）。
- **版本漂移硬注**：spec 11.3「常见 XID 31/43/48/63/74/79」中 **74 在 Blackwell（B-x00/GB-x00）不适用**，NVLink5 错误走 149（+144-150 decode）；writer 须注代际，勿把 74 当 Blackwell NVLink 错误码。

## 4. ECC / row remapping / RAS（11.3，[C] GPU Memory Error Management 2026-11-11 + Hopper wp p.38）

### 4.1 GPU Memory Error Management（a100-gpu-mem-error-mgmt，名 A100 实覆盖 Ampere/Hopper/Blackwell）

- **Overview**：Ampere/Hopper/Blackwell 内存错误恢复特性 = error-containment / dynamic page offlining / row-remapping / **RAS Repair for GPU Memory**；「Uncorrectable error to correctable error coverage improved by 10%」；ECC correction of SBEs。
- **Error Containment**（Ampere+）：UE ECC 影响**限遭遇错误的 app**，余 workload 精度/性能不受影响、可续跑、可启新 workload；**多数 UE 无需 GPU reset**（vs Volta 影响全部 workload）；罕见 uncontained 仍影响全部。
- **Dynamic Page Offlining**（Ampere+）：driver 标 UE 所在 page 不可用，当前/新 workload 不再分配该 page；多数 UE 无需 reset。
- **Row Remapping**（Ampere+，逐字）：「a hardware mechanism... Every bank in DRAM is equipped with **spare rows in hardware**... replaces degrading memory cells with spare ones... **does not leave software visible holes** in the address space... requires a **GPU reset** to take effect and will remain **persistent throughout the life of the GPU**」。**Table 2**：legacy page retirement max **64** retirements vs row-remapping max **512** remappings；CE remap 可被 UE remap 替换（bank reserved rows 耗尽时）；pending 生效需 GPU reset。
- **RAS Repair / GPU Memory Repair**（select Blackwell，hbm-channel-repair.html title「RAS Repair」）：swap **spare DRAM channels 或 L2 cache slices**；「After **two row re-mappings in the same bank**, the next occurrence of an uncorrectable ECC error in that bank will attempt to trigger repair if a spare is available」；**XID 160** = channel/L2 slice 成功标记 repair；**需 reboot 生效**。
- **RMA Policy**（rma-policy-thresholds-for-row-remapping.html title「RMA Policy」）：row-remap failure flag 触发 = ① bank 已 **8 UE rows** remapped 后再 attempt；② 已 remap row 再 attempt（可 <8 total）；③ **512 total** UE remappings 后。flag 经 in-band（NVML/nvidia-smi）+ out-of-band（SMBPBI）。**SRAM Threshold Exceeded** = parity-protected SRAM **>4 UCE Unique Count/bank** 或 SECDED-protected SRAM **>2 UCE Unique Count/bank**（Field Diagnostic 判 RMA）。

### 4.2 Hopper whitepaper p.38 逐字（重取，recap 锚 2.1 §4）

- **ECC Memory Resiliency**：「The H100 HBM3/2e memory subsystems support **Single-Error Correcting Double-Error Detecting (SECDED)** ECC... H100 supports "**Sideband ECC**"... a small memory region, **separate from the main HBM memory**, is used for ECC bits（vs "Inline ECC" 从主存划出）... Other key memory structures... also protected by SECDED ECC including the **L2 cache and the L1 caches and register files inside all the SMs**」。
- **Memory Row Remapping**：「invalidate memory rows that have memory cells that generated ECC errors and **replace the rows at boot time** with reserved known-good rows... A number of memory rows in each HBM3/HBM2e memory bank are set aside as **spare rows**」。
- **recap 口径注**：brief Task 1 写「recap 1.2 Hopper ECC」——**ECC SECDED 账实在 2.1 §4**（02-memory-hierarchy/01-hbm-and-memory-subsystem.md L124-147），1.2 无 ECC 节；writer 11.3 ECC recap 指 **2.1 §4**（Sideband 带宽不变 + SECDED 校验位下界命题 2），非 1.2。

### 4.3 nvidia-smi ECC / 健康查询（[C] nvidia-smi docs）

- **ECC Errors**：SBE 自动纠正无 data corruption；DBE 检测不纠正；**volatile**（自 driver load；Linux 无 persistence 时每次 compute app 重置）vs **aggregate**（终身持久，Inforom）；位置 breakdown Device Memory/Register File/L1/L2/Texture/Total；Turing+ SRAM Correctable/Uncorrectable + DRAM Correctable/Uncorrectable；**Ampere+ SRAM 分 parity vs SEC-DED + per-unit histogram（SRAM L2/SM/Microcontroller[PMU/GSP]/PCIE/Other）+ threshold exceeded flag**。
- **Repair flags**：Channel Repair Pending / TPC Repair Pending / Unrepairable Memory；「If one of the repair flags is pending, check the GPU Recovery action」。
- **Page Retirement**：DBE 或多 SBE 同 page → retire（driver 隐藏，分配不可达）；Double Bit ECC / Single Bit ECC / Pending 计数。
- **Recovery actions**（对齐 XID 154）：None / Drain P2P / **Drain and Reset**（contained HBM UCE，page offlining 降容量，**仅 UCE row remap 断言**，drain 后 reset 恢复全容量）/ GPU Reset / Bus Reset（nvidia-smi -r bus）/ System Reboot。
- **--query-remapped-rows**（csv）；**nvidia-smi nvlink -e**（v560+ 含 NVLink5 error counters）；**nvidia-smi topo -m**（PCIe/NVLink 拓扑，11.1 NUMA/affinity 工具）。

### 4.4 DCGM 监控 field IDs（[C] dcgm-api-field-ids + 教学参照 repo）

- ECC：**310** SBE_VOL_TOTAL / **311** DBE_VOL_TOTAL / **312** SBE_AGG_TOTAL / **313** DBE_AGG_TOTAL（monotonically increasing）/ **345** DBE_AGG_SRM / **346** SRAM_EXCEEDED（Ampere+，bool）。
- Row remap：**392** PAGE_RETIRED_PENDING / **393** ROW_REMAP_UNCORRECTABLE_TOTAL / **394** ROW_REMAP_CORRECTABLE_TOTAL / **395** ROW_REMAP_FAILED / **396** ROW_REMAP_PENDING。
- NVLink 链路错误计数器：**400-404** CRC_FLIT_ERROR_L0-L4_TOTAL / **420-429** REPLAY_ERROR（L0-L5 + TOTAL）/ **430-435** RECOVERY_ERROR（L0-L5）。
- 温度/功率：**140** MEMORY_TEMP_CELSIUS / **156** TOTAL_ENERGY_CONSUMPTION（mJ since driver reload）；GPU_TEMP / POWER_USAGE 同表。
- **DCGM diag run levels**（dcgm-diagnostics.html 逐字表）：**r1 Short <2.5s = software**；**r2 Medium（4GPU<2.5min/8GPU<10.5min）= +pcie（PCIe+NVLink）/memory/memory_bandwidth**；**r3 Long（4GPU<10min/8GPU<35min）= +diagnostic/targeted_stress/targeted_power/nvbandwidth/nccl_tests**；**r4 Extra Long（4GPU<45min/8GPU<2.25h）= +memtest/pulse（Input EDPp）**；高级含低级；`dcgmi diag -r "pcie,diagnostic"` 指名测试；`-r 3 -p diagnostic.test_duration=300.0` 参数。**nvbandwidth/nccl_tests 内嵌为 DCGM diag r3 plugin**（11.2/11.3 交叉）。

## 5. 服务器形态 / 供电散热（11.1，[C] datasheet + user guide）

### 5.1 HGX / DGX 规格

- **HGX H100/H200 datasheet**（44wa PDF，Jul24，3361709）：HGX H100 8-GPU = 8×H100 SXM / FP8 32PF* / INT8 32POPS* / BF16 16PF* / TF32 8PF* / FP32 540TF / FP64 270TF / FP64TC 540TF / 640GB HBM3 / NVLink4 / **NVSwitch 第三代（4-GPU N/A）** / GPU-to-GPU 900GB/s / Total Aggregate 7.2TB/s。（「GPU Aggregate Bandwidth 13/27 GB/s」行系 datasheet 单位 typo，实 TB/s，以 Total Aggregate 行为准。）
- **DGX H100/H200 User Guide**（introduction-to-dgxh100.html）：8×H100（640GB）/H200（1128GB）；2×Intel Xeon 8480C（56 core，PCIe Gen5）；4×NVSwitch（NVLink4 900GB/s）；8×CX7 单口 400G + 2×CX7 双口；2TB DIMM；**8U / 130.45kg max / 10.2kW max / 6×3.3kW PSU 4+2 冗余 / 200-240VAC**；≥3 PSU 方可 boot。
- **DGX B200 User Guide**（introduction-to-dgxb200.html）：8×B200（1440GB）；2×Xeon 8570（56 core）；**2×第五代 NVLink switch，14.4TB/s aggregate**；8×CX7 单口 400G + 2×BF3 双口；2TB（可扩 4TB）；**10U / 142.4kg max / 14.3kW max / 6×3.3kW PSU 5+1**。
- **Blackwell datasheet**（01dyp27s PDF，Oct25，4204213）：GB200 NVL72 per-GPU FP4 20PF sparse（dense 10）/ FP8 10（5）/ INT8 10POPS（5）/ BF16 5（2.5）/ TF32 2.5（1.25）/ **FP32 80TF** / FP64 40TF / 186GB 8TB/s / **MIG 7** / 7 NVDEC+7 nvJPEG / **TDP configurable up to 1200W** / NVLink5 1.8TB/s + PCIe Gen5 128GB/s；HGX B200 per-GPU FP4 18（9）/ FP8 9（4.5）/ BF16 4.5（2.25）/ **FP32 75TF** / FP64 37 / 180GB 7.7TB/s / MIG 7 / **TDP up to 1000W**；脚注「Specifications in sparse | dense. Dense is one-half of the sparse spec shown.」；**RAS engine**（AI-powered predictive-management 监控数千数据点预测 downtime）+ **TEE-I/O**（首个 TEE-I/O capable GPU + inline NVLink protection）。
- **Blackwell Ultra datasheet**（1k0p832e PDF）：GB300 NVL72 per-GPU **TDP up to 1400W** / HGX B300 **1100W**；279GB HBM3e（usable 口径，raw 288，MIG 7g.269gb，三数勿混）。
- **Inside Blackwell Ultra blog**（[C]）：Table 2 chip 对比 Hopper 80B/1die/700W vs Blackwell 208B/2die/NVFP4 10|20PF/192GB/8TB/s/**1200W** vs Blackwell Ultra 208B/2die/15|20PF/288GB/8TB/s/**1400W**；**160 SMs（up to）/8 GPC/640 TC（4/SM）/128 CUDA core/SM/TMEM 256KB**；PCIe Gen6 x16 256GB/s 双向（Ultra）；MIG 例（2×140GB/4×70GB/7×34GB）；GB300 NVL72 1.1EF dense FP4 + power shelf + power smoothing（energy storage/burn）。

### 5.2 GB200 NVL72 机架供电散热（DGX GB200 guide hardware.html，逐字）

- **Rack Configuration**：18×1RU compute trays（2 Grace+4 Blackwell）+ 9×1RU NVLink switch trays + 2×TOR 管理 switch + power shelves；NVLink passive copper cable cartridge backplane。
- **Compute tray**：4×CX7 单口 400G OSFP（GB200）/ 4×CX8 800G（GB300）+ 2×BF3 双口 400G；4×3.84TB E1.S NVMe RAID0 + 1.92TB M.2 boot；BMC+HMC OOB。
- **NVLink switch tray**：「**2x NVLink switches, each of which has 72 NVLink ports**」；「**57.6 terabits per second (Tbps) of full duplex bandwidth**... in a 1U design」；NVOS 2.0+ / FM+SM 控制面跑 tray / CPU ERoT。**销 W4 残留**（per-chip 72 ports / 2 chips per tray / 7.2TB/s = 57.6Tbps÷8）。
- **Power Shelves**（逐字）：「The rack power consumption is **approximately 120kW**. The power shelf uses **six air-cooled 5.5kW PSUs in eight power shelves** that provide **N+N redundancy** and the required input power of **33kW per power shelf**」；busbar 配电，AC→**nominal 50-51V DC**。
- **液冷**：compute tray CPU/GPU cold plate 经 manifold 液冷；network/storage 风冷；**leak detection**。
- **机架功率密度账**（11.1 推导）：NVL72 ≈120kW/rack（72 GPU）vs DGX H100 10.2kW/8U（8 GPU）vs DGX B200 14.3kW/10U（8 GPU）；GPU TDP 账 H100 8×700=5.6kW / B200 8×1000=8kW / GB200 72×1200=86.4kW（+Grace+switch+损耗→120kW）。

## 6. Profiling 与调优（11.2，[C] Nsight docs + 教学参照 repo）

- **Nsight Systems v2026.6**（UserGuide）：系统级 timeline；**NVTX range 投影 GPU timeline**（「The Timeline View also projects these ranges onto the GPU timeline」）；cudaProfilerStart/Stop（cuda_profiler_api.h）限定 profiling 区间；CUDA API trace（默认跳过非关键 API，--cuda-trace-all-apis 全量）；CPU IP sampling（--sampling-trigger，lbr backtrace）；**--gpu-metrics-devices / --gpu-metrics-frequency / --gpu-metrics-set**（GPU metrics 采样）；CUDA graph trace granularity（node/graph/host-only）；**profiling overhead 注**（backtrace/all-apis/graph node trace「may cause significant runtime overhead」，flush interval 权衡 memory/overhead）→ 11.2「profiling 开销与采样偏差」一手。
- **Nsight Compute ProfilingGuide**（W2 锚重核 200）：**GPU Speed Of Light Throughput + SpeedOfLight_Hierarchical{Double,Half,Single}RooflineChart**（「High-level overview of the utilization for compute and memory resources... presented as a roofline chart」）→ 11.2 实测 roofline（achieved vs theoretical）回链 2.4；W2 已锚 Warp Stall Reasons（§2.4.7）/ Memory Workload Analysis（Sectors/Req）/ Occupancy Calculator（§4.13）→ 11.2 stall reason 回链 1.4 / occupancy 回链 3.4。
- **nvbandwidth**（repo 200 + README 重核）：CE/SM 两类拷贝带宽实测（W1 锚：spin kernel+CUDA events/3 采样中位数/双向 per-direction）→ 11.2 通信基准回链 2.4/7.2。
- **nccl-tests**（repo 200，W5 锚）：busbw 修正因子 AllReduce 2(n-1)/n / RS/AG/A2A (n-1)/n / Bcast/Reduce 1 → 11.2 通信基准回链 8.3/10.2。
- **DCGM**（repo 200 + feature-overview + diagnostics）：diag r1-r4（§4.4）+ Field Value Watches（GPU metrics）+ Active Health Checks（GPU subsystems）+ Policy Notification → 11.2 监控 + 11.3 健康检查。
- **bottleneck 决策树**（11.2 推导，recap 2.4）：roofline 位置（compute/memory-bound）+ NCU stall reason → 优化方向（coalescing 4.1 / bank conflict 4.2 / tiling 4.3 / occupancy 3.4 / fusion 5.4 / 精度 6.1）；「实测」= 讲方法 + 引官方/已发表数值（spec §6.2，非新跑）。

## 7. ⟨verify⟩ 销项 / 残留台账

| 项 | 状态 | 一手 / 口径 |
| --- | --- | --- |
| B200 SM 数（spec 1.1 ⟨verify datasheet⟩，W1→W3→W6 残留） | **销（148，[B] 口径）** | arXiv:2512.02189v3 逐字 148 SMs/8 GPC/4 L2/8 HBM3e + CC 实测 74/die×2 + 官方 Ultra blog full 160「varies by SKU」+ 1.1 反推自洽；**官方 B200 datasheet/brief 未印 SM 数**（残留注） |
| MIG URL / 几何 / profile / 最多 7（spec 11.1 ⟨verify URL⟩） | **销** | MIG User Guide 2026-11-04 子页全表；A30=4 例外 |
| vGPU / SR-IOV（spec 11.1 ⟨verify⟩） | **销** | vGPU 17.0 UG §2.1/2.10.2 + TB-10226 brief |
| XID 31/43/48/63/74/79 语义（spec 11.3 ⟨verify docs⟩） | **销** | Release 620 catalog xlsx + archive 归因表；**74 Blackwell 不适用→149 漂移注** |
| ECC / row remapping（spec 11.3 ⟨verify⟩） | **销** | GPU Memory Error Management 2026-11-11 + Hopper wp p.38 重取 |
| Nsight Systems/Compute URL（spec 11.2 ⟨verify URL⟩） | **销** | nsys v2026.6 + ncu ProfilingGuide SOL roofline |
| DGX H100 供电散热精确数值（hgx-dgx ⟨verify W6〉） | **销** | DGX H100/H200 UG 10.2kW/8U/130.45kg + DGX B200 14.3kW/10U/142.4kg |
| GB200 NVL72 供电散热（gb200-nvl72「11.1 留 W6」） | **销** | DGX GB200 guide ≈120kW/8 power shelves/busbar/液冷 |
| NVLink switch per-chip 72 ports/2 chips per tray（W4 残留） | **销** | DGX GB200 guide hardware.html 逐字 2×72 ports/57.6Tbps |
| H100 700W / B200 1000W datasheet（spec 11.1 ⟨verify datasheet〉） | **销（W1 已锚本轮重取）** | nvidia-datasheet H100 SXM ≤700W / HGX B200 ≤1000W / GB200 ≤1200W / GB300 ≤1400W |
| JEDEC HBM3 精确 URL（W1） | 残留（403） | P11 不涉，成员官方互证沿 W1 |
| PCI-SIG 精确 URL（W4） | 残留（403） | P11 用 W4 已核 Gen5 128GB/s 数值 |
| Al-Fares PDF（W5） | 残留 | P11 不涉 |
| Cornell CAC B200 SM 页 | 残留（000 不可达） | 非关键，[B] 论文 + CC 实测足够 |

## 8. writer 引用白名单与口径守卫

- **11.1**：mig-user-guide / vgpu-docs / hgx-dgx / gb200-nvl72 / dgx-gb200-guide / nvidia-datasheet / blackwell-whitepaper / inside-blackwell-blog / blackwell-b200-microbench（B200 SM 148 [B] 口径注）；recap 7.1/7.2（NVLink 域）/ 2.1（SKU raw-vs-usable）；K8S device plugin/调度 → cluster 指针（新 seam，不展开）；inference 9.1 成本账指引。
- **11.2**：nsight-systems / nsight-compute / nvbandwidth / nccl-tests / dcgm；recap 2.4（roofline I/ridge）/ 1.4（stall reason）/ 3.4（occupancy）/ 4.1-4.3/5.4/6.1（调优决策树）；「实测」讲方法 + 引官方/已发表数值（spec §6.2）。
- **11.3**：xid-docs / ecc-memory-mgmt / nvidia-smi-docs / dcgm / hopper-whitepaper（ECC p.38 recap 2.1 §4）/ inside-blackwell-blog（RAS）；recap inference 8.5 弹性 / training 7.4 checkpoint 续训（故障分类 training 7.3）；集群级故障编排 → cluster 指针。
- **口径守卫**：① B200 SM 148 标 [B] 微基准口径（官方 datasheet 未印），勿写「官方 datasheet 披露 148」；② XID 74 Blackwell 不适用（→149），SXid 仅 Hopper-；③ MIG「最多 7」A30=4 例外，L2 随 memory slice（1/8）非 SM slice；④ row remap 512 vs legacy 64，GPU reset 生效（Hopper+ MIG enable 无需 reset 勿混）；⑤ ECC recap 锚 **2.1 §4 非 1.2**；⑥ SKU 功率 H100 700 / HGX B200 1000 / GB200 1200 / GB300 1400 / B300 1100 勿混；GB200 FP32 80TF vs HGX B200 75TF；⑦ NVLink 方向头条双向（沿 W4）；⑧ 机架 120kW 是 DGX GB200 guide 口径（≈，含 Grace+switch+损耗），非 72×1200=86.4kW GPU-only。
- **W6 新词登记（writing pass 随波）**：GPU instance / compute instance / memory slice / SM slice（11.1）、row remapping / SECDED（2.1 首现，11.3 深化）、SR-IOV PF/VF（11.1）、NVTX（11.2）、DCGM diag / field ID（11.2/11.3）、XID / SXid（11.3）、RAS Repair / channel repair（11.3）、power shelf / busbar（11.1）。glossary 已种 MIG/vGPU-SR-IOV/Nsight/XID/ECC（W0，首现 11.1/11.2/11.3）。
