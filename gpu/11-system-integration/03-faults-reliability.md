# 11.3 故障与可靠性（Faults and Reliability）

> 凌晨训练作业中断，kernel log 里一行 XID 48——该重启进程、reset 这张卡，还是走 RMA 换卡？故障处置为什么必须先分类再动作？本章把 GPU 硬件故障还原成可判定的诊断账：ECC 错误从 SECDED recap（2.1 §4）走到生产计数与 row remapping（512 对 legacy 64、GPU reset 生效、RMA 阈值 8 UE rows/bank）；XID 错误码逐码分类（31/43/48/63/64/74/79，**74 在 Blackwell 不适用 → 149 NETIR**）；thermal throttle 吞吐上界账（HW slowdown 核心频率减半 → compute-bound 吞吐上界减半）；NVLink/IB 链路错误计数器（CRC flit / replay / recovery）与降级带宽账（18 link 断 1 → 94.4%）；健康检查分层（DCGM diag r1-r4 / nvidia-smi / DCGM 字段监控）与预防性维护三趋势 | 前置：2.1 §4（ECC SECDED/Sideband，recap 源）、1.1 §3（峰值频率账 recap）、2.4（roofline recap）、7.1（NVLink 18 link × 25 GB/s 单向）、9.3 §6-7（CRC/retry/链路降级/straggler recap）、11.1（供电散热与 PCIe 拓扑）、11.2（DCGM 监控与通信基准）；跨维度 recap：inference 8.5（弹性容错）、training 7.4（checkpoint 续训）、training 7.3（故障分类）——均 recap + 指引不重推 | 锚点：NVIDIA XID Errors Release 620（[NVIDIA, XID docs](../references.md)，[C] 官方规格披露·一手权威）、GPU Memory Error Management（[NVIDIA, GPU Memory Error Management](../references.md)，[C]）、nvidia-smi docs（[NVIDIA, nvidia-smi docs](../references.md)，[C]）、DCGM（[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现）、Hopper whitepaper p.38（[NVIDIA, Hopper whitepaper](../references.md)，[C]）、Inside Blackwell Ultra blog（[NVIDIA, Inside Blackwell Ultra blog](../references.md)，[C] 开放技术博客）——官方文档 2026-11-26 一手核实

## 1. 三个问题与本章口径

同一张卡报 XID 48，处置可能是「重启遭遇错误的 app、方便时再 reset」，也可能是「立即 reset」，还可能是「Field Diagnostic 判定 RMA 换卡」——差别不在错误码本身，而在错误落在哪个子系统、影响是否被 contain、remap 资源还剩多少。本章回答三个问题：GPU 硬件故障怎么分类与定位（ECC 计数、XID 语义、链路计数器，§3-6）？故障对频率/吞吐/带宽的定量代价怎么算（§5-6）？健康检查与预防性维护怎么把「事后救火」变成「事前换卡」（§7-8）？系统层弹性容错（inference 8.5 / training 7.4）已交付的部分本章只 recap + 指引，不重推。

**本章口径**（章脊锁定；事实密集章，硬件 SKU/世代为脊，多 SKU 口径在此一次声明）：

| 口径 | SKU 与世代 | 来源 |
| --- | --- | --- |
| 主口径 | H100 SXM（Hopper）：XID 74 适用、row remap 上限 512、NVLink 18 link × 25 GB/s 单向、FP32 66.9 TFLOPS @ 1980 MHz | XID catalog + GPU Memory Error Management + Hopper wp Table 3 / p.47（recap 1.1 §3 / 7.1） |
| 对比口径 | Blackwell（B200/GB200）：XID 74 不适用 → 149 NETIR、RAS Repair（XID 160）、NVLink5 计数器 | XID catalog + GPU Memory Error Management + Inside Blackwell Ultra blog |
| IB 对照口径 | ConnectX-7 NDR 400G 单口（50 GB/s 单向）；CRC/retry/链路降级带宽后果 recap 9.3 §6 | 9.3（不重推） |

数值算例中未披露分量（flit 错误概率、throttle 时间占比）逐处标**教学假设口径**；结构结论（分类、阈值、上界）才是交付物。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**）； $M$ = 被传输消息总字节（§6 recap 算例）； $\mathrm{BW}$ = 带宽（GB/s， $10^{9}$ B/s）； $n_{\mathrm{link}}$ = link 数； $I$ = 算术强度、 $F_{\mathrm{peak}}$ = 峰值 FLOPS、 $n_{\mathrm{SM}}$ = SM 数、 $c_{\mathrm{prec}}$ / $f_{\mathrm{prec}}$ = 每 SM 每周期 FLOPs / 精度类 boost 频率（recap 1.1 式 (2)，不重推）。局部符号： $p$ = 每 flit 出错概率（§6 教学模型）； $\phi$ = throttle 时间占比（§5）； $k$ = 失效 link 数（§6）； $X$ = 无 throttle 吞吐（§5）。 $N$ 专归参数量、 $B$ 专归 batch、 $T$ 专归序列长，本章不占用； $\alpha/\beta$ 通信模型见 9.3，本章不用。

**边界声明**：本章交付 **GPU 硬件故障诊断增量**——ECC 生产计数与 remap/RMA 阈值、XID 分类与处置、链路错误计数器、健康检查与预防性维护。ECC 码族账（SECDED Hamming 界、Sideband 带宽不变）recap 2.1 §4 不重推；弹性容错系统设计 recap inference 8.5（serving 侧冗余 quota 与 rerouting）/ training 7.4（checkpoint 与续训）/ training 7.3（故障四分类与 SDC），不重推；IB CRC/retry/链路降级带宽后果 recap 9.3 §6 不重推。**集群级故障编排（节点 drain、备机池管理、作业重调度、K8S device plugin 健康上报）归 cluster 维度**——本章只交付单卡/子系统级的诊断与处置动作，编排层不展开（新 seam）。代码 scope = 机制片段级（nvidia-smi/dcgmi 命令例，非模块走读）。测量口径沿全书：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. 系统层已经兜住什么：弹性容错 recap

先回答一个更基本的问题：单卡故障在变成作业级损失之前，系统层已经垫了几层缓冲？三层——训练侧 checkpoint 把「故障丢失的工作」压到半个存档间隔量级，serving 侧冗余 quota 把「故障缺口」压到 ms 级 rerouting，故障分类学把「硬件问题」从数据/优化器/数值问题里分出来。这三层都已交付，本章只 recap 结论 + 指引：

- **training 7.4（checkpoint 与续训）**：Llama-3 405B 的 54 天窗口 466 次中断 → MTBF 约 2.8 h（training 7.4 式 (2) recap）；最优存档间隔正比于「单次停顿成本 × MTBF」的平方根（式 (4) recap，推导归 training 7.4 §6）；异步 staging 把 GPU 可见停顿压到一次 D2H 拷贝；续训免重新 warmup 的判据是优化器状态「热」。
- **inference 8.5（弹性与容错）**：冗余 quota 利用率上界 $u_{\max} = 0.8$ 保证可吸收实例故障比例下界 $f_{\max} \ge 0.2$（命题 4 recap，符号沿 inference 8.5）；容错靠请求级 rerouting（ms 级缺口）而非缩短重启（分钟级，命题 5 recap）；大 EP 单通信域里单 GPU 故障让整实例不可用，重启时按当前健康 GPU 重定 EP size（doc L265 口径 recap）。
- **training 7.3（故障分类）**：四类触发通道（数据/优化器/数值/硬件）；herd 419 次意外中断里 GPU 类占 58.7%（口径注：该数是原表 % 列之和，表文差异的核对见 training 7.3 §2.4）；SDC（silent data corruption，静默数据损坏）是唯一健康检查检不出的通道，须间接统计判据 + 定点硬件测试。

系统层容错有一个共同前提：**故障已被检出并定位到具体卡与子系统**。rerouting 要知道哪张卡坏了才能摘请求；checkpoint 恢复要知道换哪张卡才能重启作业；SDC 判据报警后要知道对哪个 rank 做定点测试。「检出 → 定位 → 处置」链条的硬件侧，就是本章增量：

```
故障事件（硬件侧）
  │
  ├─ 检出面：XID（kernel log）/ ECC 计数 / 链路计数器 / 温度功率（§3-7）
  ├─ 定位：哪个子系统？app bug 还是 HW？（XID 分类 §4 + 归因表）
  ├─ 处置：RESTART_APP / RESET_GPU（deferred）/ Drain and Reset / RMA（§3-7）
  │
  ▼
系统层恢复（recap，不重推）
  ├─ serving：rerouting 到健康实例（inference 8.5 §6）
  ├─ training：隔离坏卡 → 换卡 → checkpoint 续训（training 7.4 §5-6）
  ▼
集群级编排（节点 drain / 备机池 / 作业重调度）→ cluster 维度指针
```

> **提点**：机制与策略的分工在容错上同样成立——硬件侧交付「哪张卡、哪个子系统、什么动作」（机制，本章），系统层决定「何时迁请求、何时重启、何时续训」（策略，inference 8.5 / training 7.4），集群层编排节点级动作（cluster 维度）。三层各消费下一层的结论，而不是各读一遍原始日志。

## 3. ECC 错误：从 SECDED recap 到生产计数

2.1 §4 已经回答了「ECC 花多少校验位、吃不吃带宽」：H100 HBM3/2e 用 SECDED（single-error correction, double-error detection，单错纠正双错检测）码族 + Sideband 放置——校验位存独立存储区域，数据带宽 3352 GB/s 不打折；L2/L1/register file 同受 SECDED 保护；wp p.38 同页披露 boot-time row remapping（每 bank 预留 spare row）。Hamming 界与教学实例化 12.5% 存储开销归 2.1 §4 命题 2，本章不重推。本节增量是生产侧三问：错误怎么分类计数？坏了的行怎么修？修到什么程度该换卡？

### 3.1 CE 与 UE：可纠正与不可纠正的分界

所谓 CE（correctable error，可纠正错误），是指 SECDED 能纠正的单 bit 错（SBE，single-bit error）——数据被硬件修正、无 corruption，计数累加但作业无感；所谓 UE（uncorrectable error，不可纠正错误），是指超出纠错能力的错误（DBE，double-bit error 为其典型）——**检得出、纠不了**，数据不可信，驱动报 XID 48（Desc 逐字「Double Bit ECC Error」，[NVIDIA, XID docs](../references.md)，[C]）。分界的生产意义：CE 是趋势信号（预防性维护的输入，§8），UE 是事件信号（处置流程的触发，§4）。

计数口径（[NVIDIA, nvidia-smi docs](../references.md)，[C]）：**volatile** 计数自 driver load 起算（Linux 无 persistence mode 时每次 compute app 重置），**aggregate** 计数终身持久（存 Inforom）。位置 breakdown 分 Device Memory（HBM）/ Register File / L1 / L2 / Texture / Total；Turing+ 分 SRAM 与 DRAM 两路 Correctable/Uncorrectable；Ampere+ SRAM 再分 parity 与 SEC-DED 两种保护，并给 per-unit histogram（L2 / SM / Microcontroller[PMU/GSP] / PCIE / Other）与 threshold exceeded flag。查 SRAM UE 时先看 histogram 定位单元，再查 threshold flag——XID 48 的 catalog 指引（Trigger 列转述）：错误报在 SRAM 侧（excludes framebuffer）时查 sram_threshold_exceeded flag，置位则 CONTACT_SUPPORT。

### 3.2 Error containment 与 page offlining：UE 的影响半径

Ampere 之前（Volta），一个 UE 影响卡上全部 workload；Ampere+ 引入 error containment——所谓 error containment（错误限制），是指把 UE 的影响限制在遭遇错误的 app 内：官方逐字「the impact will be limited to the applications that encounter the error」，其余 workload 精度与性能不受影响、可续跑、可启新 workload，**多数 UE 无需 GPU reset**（[NVIDIA, GPU Memory Error Management](../references.md)，[C]）。对应 XID 94（contained，限单 app，RESTART_APP）与 XID 95（uncontained，影响多 app，须 RESET_GPU）；罕见 uncontained 情形仍存在（官方 note 逐字）。dynamic page offlining 是配套动作：driver 把 UE 所在 page 标为不可用，当前与新 workload 不再分配该 page——可用容量可见地少一页（§7.3 的容量侧后果）。

### 3.3 Row remapping：硬件换行，不留软件空洞

所谓 row remapping（行重映射），是指用 DRAM bank 内硬件预留的 spare row 替换劣化行的机制——官方逐字「Every bank in DRAM is equipped with spare rows in hardware」；与 page retirement 的本质差别是「does not leave software visible holes in the address space」：修复发生在硬件层，地址空间不洞、容量不减（[NVIDIA, GPU Memory Error Management](../references.md)，[C]）。三个硬口径：

- **容量**：Table 2 逐字——legacy page retirement 上限 **64** retirements，row remapping 上限 **512** remappings，8 倍（ $512/64 = 8$ ）。
- **生效与持久**：「requires a GPU reset to take effect」——pending 的 remap 要等 GPU reset 才生效；生效后终身持久（跨 reboot 不丢）。XID 63（remapping event）的处置因此是 RESET_GPU_DEFERRED（可延迟到方便时 reset），不是立即停机。
- **替换策略**：CE 触发的 remap 可被 UE remap 替换（bank reserved rows 耗尽时）——spare row 优先保 UE。

Blackwell 再加一层 **RAS Repair**（select Blackwell 产品）：spare 粒度从 row 升级到 **DRAM channel 或 L2 slice**。触发条件逐字：「After two row re-mappings in the same bank, the next occurrence of an uncorrectable ECC error in that bank will attempt to trigger repair if a spare is available」——同 bank 第 3 次 UE 尝试整 channel（或 L2 slice）替换；成功标记报 **XID 160**，**需 reboot 生效**（比 row remap 的 GPU reset 更重一级）。

### 3.4 RMA 阈值：什么时候该换卡

RMA（Return Merchandise Authorization，返厂授权）由 NVIDIA Field Diagnostic 工具判定——DCGM diag 不做 RMA 判定（§7.1 边界）。row-remap failure flag 的触发条件是三条**析取**（[NVIDIA, GPU Memory Error Management](../references.md) RMA Policy 页，[C]）：

**命题 1**（RMA 触发是析取，8 行不是必要条件）。row-remap failure flag 在以下任一事件触发：① 某 bank 已 remap 8 个 UE row 后再 attempt；② 对已 remap 的 row 再 attempt（可在总 remap 数少于 8 时发生）；③ UE remap 总数达 512 后再 attempt。特别地，「8 UE rows/bank」是充分路径而非必要条件。**证明**（要点）：官方条文以「Any of the following events will trigger」列出三条件，析取语义；条件②逐字注明「can occur with less than eight total remaps to the same bank」。边界反例：某 row 第 1 次 remap 后再次 UE（总 remap 数 1），条件②已触发 flag——不必等满 8 行。∎

SRAM 侧另有 **SRAM Threshold Exceeded** flag：parity 保护 SRAM 每 address bank 超过 **4** 个 UCE Unique Count，或 SECDED 保护 SRAM 每 bank 超过 **2** 个（逐字「More than 4 / More than 2」）。flag 经 in-band（NVML/nvidia-smi）与 out-of-band（SMBPBI）两路可读。全数字实例：某 bank 累计 8 个 UE row remap 后第 9 次 UE attempt → 条件①触发 → Field Diagnostic 验证 → RMA 评估；Blackwell 上同 bank 第 3 次 UE 先走 RAS Repair（§3.3），repair 失败才继续 remap 直至 flag。overview 另披露「Uncorrectable error to correctable error coverage improved by 10%」——Ampere+ 相对前代把更多错误落在可纠正侧（口径：覆盖改善的官方披露值，具体码参数未披露）。

## 4. XID 错误分类：读驱动的错误报告

作业挂了之后，第一现场看哪里？kernel log 里驱动报的错误码怎么读？所谓 XID，是指 NVIDIA 内核驱动向 kernel log 报告的错误事件编号——Linux 下 grep「NVRM: Xid」即可提取（dmesg / /var/log/syslog）；官方口径「The meaning of each message is consistent across driver versions」（[NVIDIA, XID docs](../references.md)，[C] Release 620，2026-11-26 核实）。XID 是故障诊断的第一入口：它把「作业挂了」翻译成「哪个子系统、疑似什么原因、该做什么动作」。

### 4.1 关键码表

关键码语义与处置（Desc 与 In-Operation 列逐字，经典归因取 archive 七列表；13/45/149/154 的归因列不在 archive 表展开，标 —）：

| XID | 语义（Desc 逐字） | 经典归因（archive） | 处置（In-Operation） |
| --- | --- | --- | --- |
| 13 | Graphics Engine Exception | — | RESTART_APP |
| 31 | GPU memory page fault | HW+Driver+UserApp | RESTART_APP |
| 43 | GPU stopped processing | Driver+UserApp | IGNORE |
| 45 | Preemptive cleanup, due to previous errors | — | WORKFLOW_XID_45 |
| 48 | Double Bit ECC Error | HW | WORKFLOW_XID_DBE |
| 63 | GPU memory remapping event | HW+Driver+FB | RESET_GPU_DEFERRED |
| 64 | GPU memory remapping failure | HW+Driver | RESET_GPU |
| 74 | NVLink Error（Desc 列标 Unused，语义见 Trigger） | HW+Driver+Bus | WORKFLOW_NVLINK_ERR |
| 79 | GPU has fallen off the bus | HW+Driver+SysMem+Bus+Thermal | CONTACT_SUPPORT |
| 94 | Contained memory error | HW+Driver+FB | RESTART_APP |
| 95 | Uncontained memory error | HW+Driver+FB | RESET_GPU |
| 149 | NVLINK: NETIR Error（Blackwell NVLink5） | — | WORKFLOW_NVLINK5_ERR |
| 154 | GPU Recovery Action Changed | — | 按 154 报告的动作 |

### 4.2 三个关键判别

**判别一：XID 不等于硬件故障。** 31（page fault）是 MMU 报非法地址访问——逐字「Typically these are application-level bugs, but can also be driver bugs or hardware bugs」；处置先查 app（cuda-gdb / Compute Sanitizer memcheck），**重复出现在同一张 GPU** 才升级 RUN_DCGMEUD（catalog WORKFLOW_XID_31 指引）。43 更明确：user app 触发 software induced fault 须终止，但「The GPU remains in a healthy state」逐字——GPU 健康，处置 IGNORE/REPORT_ISSUE，多数是 user app error 而非 driver bug。48 才是经典归因表里 **HW only** 的码。把 31/43 当硬件故障去换卡，是最常见的误处置。

**判别二：63/64 是 remap 记录，不是新故障。** 63 = remapping event（Ampere+ row remapper 活动 / 更早 GPU 的 page retirement），64 = remapping failure；官方逐字「Only Xid 63, 64, or 177 should be seen for a given ECC event」——给定 ECC 事件只见三者之一（177 本章不展开）。63 的处置是 RESET_GPU_DEFERRED：remap 已记录、pending 生效，安排 reset 即可（§3.3）；64 才需要立即 RESET_GPU / CONTACT_SUPPORT。

**判别三：74 可能是远端故障的「同情错误」。** 74 报 GPU 到另一 GPU 或 NVSwitch 的 NVLink 连接问题，但逐字注明「if a GPU fails, another GPU connected to it over NVLink may report an Xid 74 simply because the link went down as a result」——远端 GPU 挂掉，本端也会报 74。所以 74 的诊断要先用 nvidia-smi nvlink 查链路状态与对端，再定界是链路本身还是远端设备（WORKFLOW_NVLINK_ERR 指引按消息寄存器位分类）。

### 4.3 代际漂移：74 在 Blackwell 不适用

**版本漂移硬注**：XID 74 的适用列是 A100 YES / H-x00 YES / **B-x00 NO / GB-x00 NO / Rubin NO**——Blackwell 的 NVLink5 错误走 **XID 149（NVLINK_NETIR_ERROR）**，须按官方「XID 144-150 Decode」表解码消息字段（intrInfo / errorStatus 等）再定处置（WORKFLOW_NVLINK5_ERR）；149 同样可能是对端设备问题（INVESTIGATE_PEER_DEVICE 指引：查 NVLink 另一端的物理 peer）。把 74 当 Blackwell 的 NVLink 错误码是代际错配。同理，所谓 SXid（Switch Xid），是指 NVSwitch 驱动经同构机制报告的交换侧错误——**仅 Hopper 及更早**；Blackwell NVL72 的交换侧错误另走 Fabric Manager 机制（Fabric Manager User Guide），勿把 SXid 表套到 NVL72。

### 4.4 处置汇总：XID 154 与 recovery action

新 catalog（Release 620）把处置分成 Tenant In-Operation Action（租户当下动作）与 Investigatory Action（调查动作）两列，配 Guidance Classes（RESET_GPU / RESTART_APP / IGNORE / CONTACT_SUPPORT / REPORT_ISSUE / WORKFLOW_* 等）。**XID 154（GPU_RECOVERY_ACTION_CHANGED）**伴随其他 XID 出现，汇总恢复动作，逐字取值五档：「None」「Drain P2P」「Drain and Reset」「GPU Reset Required」「Node Reboot Required」——例如日志串「Xid 154 GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)」。诊断时先读 154 定动作级别，再回查触发它的具体 XID。

```
dmesg 出现 NVRM: Xid
  ├─ 13/31/43 ──► app 级：重启 app + memcheck 查代码
  │                └─ 31 重复同一 GPU ──► dcgmi diag（EUD）排 HW
  ├─ 48/94 ────► contained UE：重启受影响 app，方便时 reset
  │                └─ SRAM DBE ──► 查 sram_threshold_exceeded flag
  ├─ 63 ───────► remap 记录：安排 reset（deferred）使 pending 生效
  ├─ 64/95 ────► remap 失败 / uncontained：立即 reset
  ├─ 74（Hopper 及更早）/ 149（Blackwell）──► 查 nvlink 计数器与对端
  ├─ 79 ───────► PCIe 不可达：查 kernel PCI log，CONTACT_SUPPORT
  └─ 154 ──────► 读 recovery action 五档，定处置级别
```

> **坑**：把任意 XID 都当硬件故障。触发条件：见 XID 即报修/换卡；现象：31/43 多数是 app bug（43 官方逐字 GPU 健康），换卡不解决问题还浪费备机；规避：先按 §4.1 归因列分类，app 级码先查代码，HW 级码（48/64/79）再走硬件流程。

> **坑**：XID 74 套 Blackwell。触发条件：在 B200/GB200 上按 74 诊断 NVLink 错误；现象：74 适用列 B-x00/GB-x00 为 NO，Blackwell NVLink5 错误报 149（144-150 decode），按 74 查会漏诊；规避：先确认世代再选码表（§4.3）；SXid 仅 Hopper 及更早，NVL72 走 Fabric Manager 机制。

## 5. GPU 故障模式：掉总线、thermal throttle 与 power brake

除了 ECC 与链路错误，单卡还有三类故障模式：PCIe 不可达（fallen off bus）、温度降频（thermal throttle）、功率制动（power brake）。三者的检出面与处置各不相同。

### 5.1 Fallen off the bus：PCIe 不可达

XID 79 的机制逐字：「the GPU driver attempts to access the GPU over its PCI Express connection and finds that the GPU is not accessible」——驱动经 PCIe 访问 GPU 发现不可达。常见原因是 PCIe 链路硬件故障导致 link down；诊断先查 system event log 与 kernel PCI event log，再区分是链路/插槽问题还是 GPU 本体故障（官方 trigger 注明亦可 GPU HW fail 或 driver issue）。处置 CONTACT_SUPPORT——掉总线不是重启 app 能解决的，通常要节点级干预（recap 11.1 PCIe 拓扑：root complex / switch 路径上的任何一环都可能是故障点）。

### 5.2 Thermal throttle：两级降频与吞吐上界

温度降频分 SW/HW 两级（[NVIDIA, nvidia-smi docs](../references.md)，[C] Clocks Event Reasons 节逐字）：**SW Thermal Slowdown**——「SW Thermal capping algorithm is reducing clocks below requested clocks because GPU temperature is higher than Max Operating Temp」，软件算法在温度超过 Max Operating Temp 时降频；**HW Thermal Slowdown**——「HW Thermal Slowdowns are reducing the core clocks by a factor of 2 or more due to temperature being too high」，硬件保护直接把核心频率砍半或更多。温度阈值三档（nvidia-smi 温度字段）：GPU Max Operating Temp（SW 降频触发）→ GPU Slowdown Temp（HW 降频触发）→ GPU Shutdown Temp（关机保护）。

降频的吞吐代价是多少？先给结论：**compute-bound kernel 的吞吐上界随核心频率线性缩放，HW slowdown 减半频率即减半上界；memory-bound kernel 不受影响**。这是 roofline 结构的直接推论，不是经验值。

**命题 2**（thermal slowdown 的吞吐上界）。设 kernel 处于 compute-bound 区（roofline 水平分支，recap 2.4），吞吐上界为 $F_{\mathrm{peak}}$；HW Thermal Slowdown 使核心频率降至一半或更低，则吞吐上界降至一半或更低。memory-bound kernel（roofline 斜分支）的吞吐上界由 HBM 带宽决定，核心频率降半不改变该上界。**证明**（路线：两步——先把峰值写成频率的乘积式，再分 roofline 两支讨论）：1.1 式 (2)（recap，不重推）重排为乘积形 $F_{\mathrm{peak}} = n_{\mathrm{SM}} \cdot c_{\mathrm{prec}} \cdot f_{\mathrm{prec}}$ ， $n_{\mathrm{SM}}$ 与 $c_{\mathrm{prec}}$ 是硬件常量，故 $F_{\mathrm{peak}} \propto f_{\mathrm{prec}}$ ，频率减半则峰值减半。compute-bound 时吞吐上界取 $\min(F_{\mathrm{peak}}, I \cdot \mathrm{BW})$ 的前者（recap 2.4），随频率线性降；memory-bound 时上界取后者 $I \cdot \mathrm{BW}$ ，HBM 带宽由栈几何决定（2.1 式 (1) recap），与核心频率无关，故不变。∎ 口径注：结论假设降频后 kernel 仍处原 roofline 区——若降频深到把工作点推过 ridge，须用 min 式重判；实际降频档位闭源，「减半」取官方「factor of 2 or more」的边界值。

全数字实例（主口径 H100 SXM，recap 1.1 §3 表）：FP32 峰值 66.9 TFLOPS @ 1980 MHz；HW slowdown 边界值（恰减半）下频率 990 MHz，FP32 峰值上界 33.45 TFLOPS——一个原本跑到 80% 峰值（53.5 TFLOPS）的 compute-bound kernel，吞吐上界掉到 26.8 TFLOPS。而一个 memory-bound kernel 的吞吐上界仍是 $I \cdot \mathrm{BW}$ （HBM 3352 GB/s 口径，recap 2.1），不随核心频率变——这就是「降频伤算力不伤带宽」的定量形式。口径注：实例取 FP32 精度类（1980 MHz 域）；各精度类 boost 频率不同（1.1 §3），Tensor Core 类峰值按同域频率比例缩放估算。

throttle 的监控面是 **Clocks Event Reasons Counters**（μs 累计）：sw_power_cap / sw_thermal_slowdown / hw_thermal_slowdown / hw_power_brake_slowdown / sync_boost 各一路（nvidia-smi --query-gpu=clocks_event_reasons_counters.* 可查）。设 throttle 时间占比 $\phi$ （教学模型：throttle 期间恰减半、kernel compute-bound），有效吞吐上界

$$
X_{\mathrm{eff}} \le X \left(1 - \frac{\phi}{2}\right) \qquad (1)
$$

**符号**： $X$ = 无 throttle 吞吐； $\phi$ = throttle 时间占比（0 到 1）。代入： $\phi = 0.10$ （10% 时间在 HW slowdown）→ 上界为无 throttle 的 95%； $\phi = 0.50$ → 75%。读法：偶发 throttle 代价小，**持续占比才是冷却系统问题的信号**（§8 预防性维护）。

### 5.3 Power brake 与 power cap：外部制动不是 GPU 故障

功率侧两个事件要分开（nvidia-smi docs 逐字）：**SW Power Cap**——「SW Power Scaling algorithm is reducing the clocks below requested clocks because the GPU is consuming too much power」，这是正常功率管理（power limit 可用 nvidia-smi --power-limit 调），不是故障；**HW Power Brake**——「External Power Brake Assertion is triggered (e.g. by the system power supply)」，外部电源制动断言，触发源在系统供电侧（PSU/机架功率事件），不是 GPU 本体故障。诊断 HW Power Brake 要查供电链路——recap 11.1：DGX H100 6×3.3 kW PSU 4+2 冗余、GB200 NVL72 8 power shelves N+N，PSU 故障或机架功率事件都可能拉 brake。**Sync Boost** 是第三种降频：组内所有 GPU 被最慢一张能达到的频率限制（逐字「All GPUs will be limited by the frequency which can be achieved by the slowest GPU」）——查本卡 throttle reason 不够，要查组内其他卡。

> **坑**：把 SW Power Cap 当故障、把 HW Power Brake 当 GPU 坏。触发条件：见降频即报修 GPU；现象：SW Power Cap 是功率管理常态（设了 power limit 就会触发），HW Power Brake 的根因在供电侧——换 GPU 不解决问题；规避：先读 clock event reason 分清 SW/HW 与触发源，HW brake 查 PSU/机架功率（11.1），SW cap 查 power limit 设置。

## 6. NVLink/IB 链路错误计数器：CRC、replay 与 recovery

链路误码怎么在变成 XID 74/149 之前被看见？靠三族错误计数器。IB 侧的机制与带宽后果已在 9.3 §6 交付（链路层 CRC 检错重传、RoCE 误码落 RC 传输层秒级超时、链路降级 4 lane 降 1 lane 带宽 ÷4、降级检测靠 CRC error / link down 计数器），本章 recap 不重推；增量是 NVLink 侧的计数器语义、replay 开销账与 link 级降级带宽账。

### 6.1 三族计数器：错误严重度的阶梯

NVLink 错误计数器按严重度分三族（DCGM field IDs，[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现，dcgm-api-field-ids 页 2026-11-26 核实）：

| 族 | DCGM 字段（代表 ID） | 语义 | 严重度 |
| --- | --- | --- | --- |
| CRC flit error | 400 系列 per-link，total 409 | flit CRC 校验失败（物理层误码） | 轻：replay 消化 |
| replay error | 420 系列，total 429 | 链路层重传事件 | 中：误码率趋势信号 |
| recovery error | 430 系列，total 439 | 链路恢复/重训练事件 | 重：链路短暂不可用 |

所谓 flit（flow control unit），是指链路层传输与检错的最小单位——CRC 按 flit 计（字段名 CRC_FLIT_ERROR 逐字）。所谓 replay，是指链路层对出错 flit 的重传（与 IB link-level retry 同构：检错 → 链路层重发，传输层不感知，recap 9.3 §6）；所谓 recovery，是指链路进入恢复流程（重训练/重新建链），期间链路不可用——比 replay 严重一个层级（口径：replay 消化单 flit 误码，recovery 是链路级事件）。nvidia-smi nvlink -e 可查错误计数器（v550+ 含 NVLink5 计数器，[NVIDIA, nvidia-smi docs](../references.md)，[C]）；IB 侧同构：端口错误计数器（CRC error / link down 事件）驱动降级检测，诊断流程 recap 9.3 §6。

### 6.2 Replay 开销账：低误码率的带宽代价可忽略

误码率 $p$ 的链路，有效带宽损失多少？先给结论：**几何重传模型下 goodput 等于线速乘 $(1-p)$， $p$ 小时损失约 $p$**——CRC 错误的稳态带宽代价微不足道，真正的威胁是误码率趋势与 recovery 事件。

**命题 3**（replay 几何模型的有效带宽）。教学模型：每 flit 独立出错概率 $p$ ，出错即重传直至成功，每次传输占用链路时间相同，则每 flit 期望传输次数为 $1/(1-p)$ ，有效数据带宽

$$
\mathrm{BW}_{\mathrm{eff}} = \mathrm{BW} \thinspace (1 - p) \qquad (2)
$$

**证明**（路线：几何分布期望 → goodput = 线速 ÷ 期望传输次数，两步）。设 $K$ = 一个 flit 的传输次数， $P(K = j) = p^{j-1}(1-p)$ （前 $j-1$ 次错、第 $j$ 次对，独立性假设在此进入）；期望 $E[K] = \sum_{j=1}^{\infty} j \thinspace p^{j-1}(1-p) = (1-p) \cdot \frac{1}{(1-p)^{2}} = \frac{1}{1-p}$ （用 $\sum_{j \ge 1} j x^{j-1} = 1/(1-x)^{2}$ ）。线速 $\mathrm{BW}$ 下每 flit 平均占 $E[K]$ 次传输时间，goodput 等于 $\mathrm{BW}/E[K] = \mathrm{BW}(1-p)$ 。∎ 口径注：独立误码、无限重传是教学假设——实际 replay 缓冲深度与重传上限闭源，超限升级为 recovery 事件（这正是三族计数器的严重度阶梯）。

全数字实例： $p = 10^{-6}$ （百万分之一）→ 带宽损失 0.0001%，可忽略； $p = 0.01$ → 损失 1%； $p = 0.1$ → 损失 10%。所以「偶发 CRC 错误」不必惊慌——replay 以约 $p$ 的带宽代价消化它；要警惕的是两件事：其一，**错误率趋势**（CRC/replay 计数持续上涨 = 物理层劣化前兆，线缆/连接器问题，§8）；其二，**recovery 事件**（链路重训练期间通信中断，集合通信 max 语义下全局停顿，升级即 XID 74/149）。

### 6.3 Link 降级带宽账：NVLink 断一条 link

NVLink 降级按 link 计（IB 按 lane 计，recap 9.3）。主口径 H100：18 link × 25 GB/s 单向（recap 7.1，wp p.47 逐字），断 $k$ 条 link 后剩余单向带宽

$$
\mathrm{BW}_{\mathrm{remain}} = (n_{\mathrm{link}} - k) \cdot \mathrm{BW}_{\mathrm{link}} \qquad (3)
$$

**符号**： $n_{\mathrm{link}}$ = link 总数（H100 = 18）； $`\mathrm{BW}_{\mathrm{link}}`$ = 每 link 单向带宽（25 GB/s）； $k$ = 失效 link 数。代入： $k = 1$ → $17 \times 25 = 425$ GB/s 单向，为满配 450 GB/s 的 $17/18 \approx 94.4\text{％}$ ； $k = 2$ → 400 GB/s（88.9%）。双向口径同比例（850 对 900 GB/s）。

单条 link 降级只损失约 5.6% 带宽，为什么还要紧急处理？因为集合通信取 max（recap 9.3 §7 straggler 账，不重推）：ring 里经过降级 link 的那个 rank 变慢，整个集合被它拖住——9.3 的 IB 算例（ $n = 16$ 、 $M = 1$ GB，一条 rail 50 降 25 GB/s，集合完成 37.5 变 75 ms 翻倍）在 NVLink 域同构成立，只是降级粒度从 rail 换成 link。诊断动作：nvidia-smi nvlink -e 定位错误计数上涨的 link → 查对端（74 的同情错误语义，§4.2）→ 隔离/换缆（WORKFLOW_NVLINK_ERR 的 INVESTIGATE_LINK_SI_AND_CABLES 指引：cable / connector / transceiver / seating / thermal 逐项排查）。

> **坑**：把偶发 CRC 错误当链路故障，或把「无 XID」当链路健康。触发条件：见 CRC 计数非零即换缆，或见无 XID 即放行；现象：前者过度维护（replay 消化单 flit 误码，带宽代价约 $p$ 可忽略，命题 3），后者漏诊（计数器可以在 XID 之前持续上涨）；规避：看趋势与 recovery 事件，不看单点计数；错误率持续上涨或 recovery 非零才升级处置。

## 7. 健康检查：DCGM diag、nvidia-smi 与监控面

作业启动前怎么主动验证「这张卡能用」？运行中怎么持续盯住它？两个工具族：DCGM diag 做主动诊断（分层级），nvidia-smi 与 DCGM 字段做被动监控（计数器与事件）。

### 7.1 DCGM diag：四个 run level

DCGM（Data Center GPU Manager）diag 按深度分四级，高级含低级（[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现，dcgm-diagnostics 页逐字表，2026-11-26 核实）：

| Level | 时长（官方表值） | 测试成员 |
| --- | --- | --- |
| r1 Short | 低于 2.5 s | software |
| r2 Medium | 4 GPU 低于 2.5 min / 8 GPU 低于 10.5 min | + pcie（PCIe+NVLink）/ memory / memory_bandwidth |
| r3 Long | 4 GPU 低于 10 min / 8 GPU 低于 35 min | + diagnostic / targeted_stress / targeted_power / nvbandwidth / nccl_tests |
| r4 Extra Long | 4 GPU 低于 45 min / 8 GPU 低于 2.25 h | + memtest / pulse（Input EDPp） |

命令：dcgmi diag -r 3 跑 r3；dcgmi diag -r "pcie,diagnostic" 指名测试；-p diagnostic.test_duration=300.0 调参。nvbandwidth 与 nccl_tests 内嵌为 r3 plugin——通信基准（11.2 的测量工具）在这里复用为健康检查项。用法分层：r1/r2 做**作业前快速门**（8 GPU 节点 r2 低于 10.5 min，r4 低于 2.25 h = 135 min，约 13 倍成本差）；r3 做**故障定位**（XID 31 重复同一 GPU → catalog 指引 RUN_DCGMEUD）；r4 做深度体检/周期性维护。边界（官方逐字）：DCGM diag 不提供全面硬件诊断、不主动修复问题、不替代 Field Diagnostic、不服务 RMA 流程——RMA 判定归 Field Diagnostic（§3.4）。

### 7.2 nvidia-smi 与 DCGM 字段：监控面清单

被动监控的查询面（[NVIDIA, nvidia-smi docs](../references.md)，[C]；DCGM 字段 ID 经 dcgm-api-field-ids 页逐项核实，[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现）：

| 监控面 | nvidia-smi | DCGM 字段（ID） |
| --- | --- | --- |
| ECC 计数 | -q（volatile/aggregate/SRAM histogram/threshold flag） | SBE/DBE vol/agg total（310-313）；SRAM DBE agg（345）；SRAM_EXCEEDED（346） |
| Row remap | --query-remapped-rows（csv） | PAGE_RETIRED_PENDING（392）；ROW_REMAP_UNCORRECTABLE_TOTAL（393）/ CORRECTABLE_TOTAL（394）；FAILED（395）；PENDING（396） |
| NVLink 错误 | nvlink -e（v550+ 含 NVLink5） | CRC flit 400 系列 / replay 420 系列 / recovery 430 系列（§6.1） |
| 温度/功率 | -q 温度功率段 | MEMORY_TEMP（140）/ GPU_TEMP（150）/ TOTAL_ENERGY_CONSUMPTION（156，mJ） |
| Throttle | --query-gpu=clocks_event_reasons_counters.*（μs） | 对应 clock event 字段 |
| PCIe | -q | PCIE_REPLAY_TOTAL（202） |

DCGM 在此之上提供三层自动化（feature-overview 页）：**Field Value Watches**（字段值越限触发，如 DBE 计数大于 0）、**Active Health Checks**（GPU 子系统主动检查）、**Policy Notification**（越限通知策略动作）——监控到动作的闭环不需要人盯日志。

### 7.3 Repair flags 与 recovery actions

nvidia-smi 还报 repair flags（Channel Repair Pending / TPC Repair Pending / Unrepairable Memory）与 GPU Recovery Action 字段（nvidia-smi docs 取值列表：None / Reset / Reboot / Drain P2P / Drain and Reset / Recover IMEX Domain / Bus Reset / System Reboot；与 XID 154 五档语义对应，§4.4）。其中 **Drain and Reset** 的语义值得单独钉住（nvidia-smi docs 逐字口径）：contained HBM UCE 后 GPU 临时以降容量运行（部分 frame buffer 被 offlining），「Asserted only for UCE row remaps」——仅 UCE row remap 断言（CE remap 不算）；「After all existing work have drained, reset the GPU to regain its full capacity」——drain 完现有工作后 reset 恢复全容量。这就是「63 → 安排 reset」处置链的容量侧理由：不 reset，pending remap 不生效、offlined page 不归还。

```
健康检查四层（被动 → 主动 → 带外）
┌─────────────────────────────────────────────────┐
│ 持续监控（in-band）：DCGM 字段 / nvidia-smi      │
│   ECC 计数 · remap · NVLink CRC/replay/recovery  │
│   温度功率 · throttle 计数器                     │
├─────────────────────────────────────────────────┤
│ 事件驱动：XID（kernel log）→ 分类处置（§4）      │
│   DCGM watch / policy → 通知                    │
├─────────────────────────────────────────────────┤
│ 主动诊断：dcgmi diag r1-r4（作业前门/定位/体检） │
├─────────────────────────────────────────────────┤
│ 带外（out-of-band）：BMC / SMBPBI flag           │
│   RMA 判定：Field Diagnostic（§3.4）             │
└─────────────────────────────────────────────────┘
```

## 8. 故障影响与预防性维护

定位完成后，处置动作怎么接回作业恢复？平时盯什么信号能在故障前换卡？

### 8.1 处置-恢复衔接

故障对训练/serving 的影响路径已 recap（§2）：训练侧单 GPU 故障可能重启整个作业（training 7.4 recap herd 口径），checkpoint 续训把丢失工作压到半个存档间隔量级；serving 侧大 EP 单卡故障整实例不可用（inference 8.5 recap），rerouting + 冗余 quota 把请求缺口压到 ms 级。GPU 诊断增量是给这条恢复链提供**精确的处置输入**：

| 诊断结论 | 处置动作 | 接回系统层 |
| --- | --- | --- |
| XID 31/43（app 级） | 重启 app，查代码 | 无需硬件动作，作业/实例原地恢复 |
| XID 48+94（contained UE） | 重启受影响 app + 方便时 reset | serving rerouting 摘请求；training 从 checkpoint 续训 |
| XID 63（remap pending） | 安排 reset 使 remap 生效 | drain 后 reset，容量恢复（§7.3） |
| XID 64/95 | 立即 reset | reset 失败升级节点级 |
| XID 79（掉总线） | 节点级排查（PCIe/kernel log） | 节点隔离，作业换节点续训 |
| RMA flag（§3.4） | Field Diagnostic → 换卡 | 备机替换（cluster 编排指针） |

straggler 是另一类「没坏但变慢」的故障：thermal throttle 持续占比高（§5.2）、链路降级（§6.3）都让单卡变慢，同步训练里拖慢整个作业（training 7.4 recap herd 的 straggler 侦测口径）——诊断面是 throttle 计数器与链路计数器，不是 XID。SDC 则相反：硬件状态正常、输出错，健康检查检不出，须 training 7.3 的间接统计判据 + 定点硬件测试（本章的 diag r3/r4 是定点测试的工具）。

### 8.2 预防性维护：三个趋势信号

事后处置不如事前换卡。三个趋势信号对应三个子系统：

- **ECC 趋势**：aggregate 计数单调增（Inforom 终身），CE remap 数量上涨 = bank 劣化前兆；UE remap 逼近阈值（8/bank、512 total）时提前安排换卡，不等 failure flag。Blackwell 的 **RAS engine** 把这件事产品化——官方逐字「AI-powered reliability system monitoring thousands of parameters to predict failures, optimize maintenance schedules, and maximize system uptime in large-scale deployments」（[NVIDIA, Inside Blackwell Ultra blog](../references.md)，[C] 开放技术博客）。
- **温度趋势**：throttle 计数器时间占比 $\phi$ 持续上涨 = 散热劣化（风扇/冷板/气流），查冷却而非查 GPU（recap 11.1 液冷 cold plate + leak detection）；HW Power Brake 频发查供电（§5.3）。
- **链路质量**：CRC/replay 错误率趋势上涨 + recovery 事件 = 物理层劣化（线缆/连接器/transceiver/seating/thermal，WORKFLOW_NVLINK_ERR 指引逐项），在升级成 XID 74/149 之前换缆。

集群级故障编排——节点 drain 调度、备机池管理、作业重调度、K8S device plugin 健康上报——归 cluster 维度（新 seam，本章不展开）；本章交付的边界是单卡/子系统级的诊断结论与处置动作，编排层消费这些结论。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 系统层容错 recap：checkpoint MTBF 约 2.8 h / 冗余 quota 可吸收故障下界 0.2 / rerouting ms 缺口；GPU 增量 = 检出-定位-处置 | §2 recap 链 | §2、§8.1 |
| CE 纠正无 corruption（趋势信号）；UE 检出不可纠（XID 48，事件信号）；volatile/aggregate 两口径 | §3.1 | §3.1、练习 1 |
| Containment（Ampere+）UE 限单 app（94），uncontained 须 reset（95）；多数 UE 无需 reset | §3.2 逐字 | §3.2 |
| Row remap 512 = 8× legacy 64；GPU reset 生效、终身持久、无软件空洞；CE remap 可被 UE 替换 | §3.3、Table 2 逐字 | §3.3、练习 1 |
| RAS Repair（Blackwell）：同 bank 2 次 remap 后第 3 次 UE 触发 channel/L2 slice 替换，XID 160，reboot 生效 | §3.3 逐字 | §3.3 |
| RMA 三条件析取：8 UE rows/bank 非必需（条件②已 remap row 再 UE 可少于 8 触发）；SRAM parity 超过 4 / SECDED 超过 2 | 命题 1 | §3.4、练习 1 |
| XID 分类：31/43 多 app bug（43 GPU 健康）；48 HW only；63 deferred reset / 64 立即；74 可远端同情错误 | §4.1-4.2 | §4、练习 4 |
| 代际漂移：74 不适用 Blackwell → 149 NETIR（144-150 decode）；SXid 仅 Hopper 及更早 | §4.3 | §4.3 |
| XID 154 汇总 recovery action 五档（None / Drain P2P / Drain and Reset / GPU Reset Required / Node Reboot Required） | §4.4 逐字 | §4.4、§7.3 |
| HW Thermal Slowdown 核心频率减半或更多 → compute-bound 吞吐上界减半；memory-bound 不变；占比 φ 的上界式 | 命题 2、式 (1) | §5.2、练习 2 |
| SW Power Cap 常态功率管理；HW Power Brake 外部供电事件；Sync Boost 组内最慢卡限制 | §5.3 逐字 | §5.3 |
| Replay 几何模型 goodput = BW(1-p)：p 百万分之一损失可忽略；威胁在趋势与 recovery 事件 | 命题 3、式 (2) | §6.2、练习 3 |
| NVLink 断 k link：剩余 (18-k)×25 GB/s 单向；k=1 → 425（94.4%）；集合 max 语义放大为 straggler | 式 (3) | §6.3、练习 4 |
| DCGM diag r1-r4（低于 2.5 s 至 8 GPU 低于 2.25 h）；不替代 Field Diagnostic / 不服务 RMA；监控字段 310-313 / 392-396 / 400-439 族 | §7.1-7.2 | §7、练习 4 |
| 预防性维护三趋势：ECC aggregate / throttle 占比 / 链路错误率；RAS engine 预测性维护（Blackwell） | §8.2 | §8.2 |

## 练习

1.（计算 · [10]）ECC 计数与 remap 容量账：(a) row remap 上限是 legacy page retirement 的几倍（Table 2 数值）？(b) 某 bank 已 remap 8 个 UE row，第 9 次 UE attempt 触发哪条 RMA 条件？(c) 判断：「总 remap 数不到 8 就不可能触发 failure flag」对不对，用命题 1 的条件②说明。校验：(a) 与 GPU Memory Error Management Table 2 的 512/64 一致；(b) 与 RMA Policy 页条件①逐字一致；(c) 错——条件②可在少于 8 时触发（命题 1 边界反例）。

2.（计算 · [20]）thermal throttle 吞吐账（命题 2 + 式 (1)）：(a) H100 FP32 峰值 66.9 TFLOPS @ 1980 MHz（1.1 §3 表），HW Thermal Slowdown 边界值（恰减半）下峰值上界多少？(b) 原跑 80% 峰值的 compute-bound kernel 吞吐上界掉到多少？(c) $\phi = 0.10$ 与 0.50 时式 (1) 的有效吞吐上界各为无 throttle 的百分之几？(d) 同 kernel 若是 memory-bound，(a) 的答案变不变？先预测 (d) 再说明。校验：(a) 33.45 TFLOPS，与 nvidia-smi docs「factor of 2 or more」边界值一致；(c) 95% / 75%；(d) 不变——HBM 带宽与核心频率无关（命题 2，2.1 式 (1) recap）。

3.（推导 · [30]）证明命题 3（replay 几何模型）并应用：(a) 写出传输次数 $K$ 的分布 $P(K = j) = p^{j-1}(1-p)$ ，说明独立性假设在哪一步进入；(b) 用 $\sum_{j \ge 1} j x^{j-1} = 1/(1-x)^{2}$ 求 $E[K] = 1/(1-p)$ ；(c) 推出 goodput 等于 BW(1-p)（式 (2)）；(d) 代入 $p = 10^{-6}$ 与 0.01 算带宽损失百分比；(e) 解释为什么 recovery 事件比同等数量的 CRC 错误更危险。校验：(b) $p$ 趋于 0 时 $E[K]$ 趋于 1（无错即一次传完）；(d) 0.0001% / 1%，与 §6.2 实例一致；(e) 与 §6.1 严重度阶梯（链路不可用 vs 单 flit 重传）一致。

4.（计算 · [30]）链路降级与处置综合账：(a) H100 NVLink 18 link × 25 GB/s 单向，断 1 条与 2 条后剩余带宽各多少 GB/s、占满配百分之几（式 (3)）？先预测断 1 条的损失量级再算。(b) recap 9.3 §7 算例（ $n = 16$ 、 $M = 1$ GB，一条 rail 50 降 25 GB/s）：集合完成时间从多少 ms 变到多少 ms，为什么不是「只慢那条流」？(c) 某卡日志序列 XID 48 → 94 → 63，按 §4/§7 写出处置链（谁重启、何时 reset、reset 后容量怎么变）；(d) 8 GPU 节点作业前门选 r2 还是 r4，用 §7.1 时长表算成本比。校验：(a) 425/400 GB/s、94.4%/88.9%，与 §6.3 一致；(b) 37.5 变 75 ms，max 语义（9.3 §7 recap）；(c) 与 §4.4 / §7.3 Drain and Reset 口径一致；(d) r2 低于 10.5 min 对 r4 低于 2.25 h（约 13 倍），与 DCGM diag 逐字表一致。

## 参考

- [NVIDIA, XID docs](../references.md)：NVIDIA XID Errors（key: xid-docs），[C] 官方规格披露·一手权威，Release 620（2026-11-11）+ Xid-Catalog.xlsx + archive 经典归因表，2026-11-26 一手核实——31/43/48/63/64/74/79/94/95/45/13/149/154 逐码语义与适用列；74 Blackwell 不适用 → 149 NETIR（144-150 decode）；SXid 仅 Hopper 及更早；Guidance Classes 与 WORKFLOW_XID_31/DBE/NVLINK_ERR/NVLINK5_ERR；154 recovery action 五档逐字
- [NVIDIA, GPU Memory Error Management](../references.md)：NVIDIA GPU Memory Error Management（key: ecc-memory-mgmt），[C] 官方规格披露·一手权威，2026-11-11 更新，2026-11-26 一手核实——error containment / dynamic page offlining / row remapping（Table 2：512 对 64、GPU reset 生效、终身持久、CE 可被 UE 替换）/ RAS Repair（同 bank 2 次 remap 后第 3 次 UE 触发、XID 160、reboot 生效）/ RMA Policy（8 UE rows/bank、已 remap row 再 attempt、512 total；SRAM parity 超过 4 / SECDED 超过 2）；overview「UCE to CE coverage improved by 10%」
- [NVIDIA, nvidia-smi docs](../references.md)：NVIDIA nvidia-smi docs（key: nvidia-smi-docs），[C] 官方规格披露·一手权威，2026-11-26 一手核实——ECC volatile/aggregate + SRAM parity/SEC-DED + histogram + threshold flag；Clocks Event Reasons（SW/HW Thermal Slowdown「factor of 2 or more」逐字、SW Power Cap、HW Power Brake、Sync Boost）与 Counters（μs）；温度三档（Max Operating / Slowdown / Shutdown Temp）；--query-remapped-rows / nvlink -e（v550+ NVLink5）/ -r / -r bus；repair flags 与 GPU Recovery Action（Drain and Reset「Asserted only for UCE row remaps」逐字）
- [NVIDIA, DCGM](../references.md)：NVIDIA/DCGM repo + docs（key: dcgm），教学参照 + 官方实现，2026-11-26 一手核实——diag r1-r4 时长与测试成员逐字表；dcgmi diag -r / -p 命令；field IDs（ECC 310-313 / SRAM 345-346 / row remap 392-396 / NVLink CRC 400 系列 · replay 420 系列 · recovery 430 系列 / 温度 140 · 150 / 能量 156 / PCIe replay 202）经 dcgm-api-field-ids 页逐项核实；Field Value Watches / Active Health Checks / Policy Notification；diag 边界（不替代 Field Diagnostic / 不服务 RMA）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威——p.38 ECC Memory Resiliency（SECDED + Sideband + L2/L1/RF 清单 + boot-time row remap，recap 锚 2.1 §4）；Table 3 FP32 66.9 TFLOPS / 1980 MHz（§5.2 实例，recap 1.1 §3）；p.47 NVLink 18 link × 25 GB/s 单向（§6.3，recap 7.1）
- [NVIDIA, Inside Blackwell Ultra blog](../references.md)：NVIDIA Inside Blackwell Ultra blog（key: inside-blackwell-blog），[C] 开放技术博客、非闭源，2026-11-26 一手核实——Advanced RAS engine 逐字（AI-powered reliability system monitoring thousands of parameters to predict failures...）；TEE-I/O + inline NVLink protection（§8.2 预防性维护产品化口径）
- 跨维度指引（纯文字）：inference 8.5（弹性伸缩与容错：冗余 quota / rerouting / EP failover，recap 源）、training 7.4（checkpoint 与续训：五分量 / MTBF / 最优间隔，recap 源）、training 7.3（故障四分类与 SDC 统计，recap 源）；gpu 2.1 §4（ECC SECDED/Sideband 码族账）、1.1 §3（峰值频率账）、2.4（roofline）、7.1（NVLink link 几何）、9.3 §6-7（IB CRC/retry/链路降级/straggler）、11.1（供电散热与 PCIe 拓扑）、11.2（DCGM 监控与通信基准）；集群级故障编排 → cluster 维度
