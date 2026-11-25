# gpu W6 场景图 QUALITY 评审档案（gpu-p11，13 图）

- 评审日期：2026-11-26；评审角色：fresh design-eye evaluator（只读，未改模块；未参与实现）
- 对象：commit **c7030b9** 的 `site/scripts/lib/figures/gpu-p11.mjs`（13 图，P11 系统集成）
- 规约：`site/scripts/lib/figures/QUALITY.md`（五维各 ≥4 过线、截图证据）+ `STYLE.md`；先例 `docs/reviews/gpu-w5-fig-quality.md`、`gpu-w3-fig-quality.md`（R1 sub 骑边框判 3 的口径）
- 证据：本轮重跑 `cd site && node scripts/fig-check.mjs gpu-p11` → **13/13 渲染**；逐张 Read `site/qa/figs-gpu-p11/fig-0..12.png`（3.5s 截图时刻）；嫌疑区 sips crop 放大（/tmp/figqa-w6-crops/f6-ilp2.png、f7-clu2.png、f3-corr2.png、f1-isl.png）；reduced 终态独立自跑 `/tmp/figqa-w6-review-reduced.mjs`（口径同 fig-reduced-diff：max 通道差 >16 记变色，t3000 vs t4500）→ **13/13 diffpx=0**，reduced 上下文 `.dia-pulse` computed display=none（31 脉冲）✓；census/锚点/defs 用 `/usr/bin/grep` 逐章核验
- 判定：**11 pass / 2 redraw**（redraw = fig-6 achievedRoofline 清晰 3、fig-7 faultChain 清晰 3；均为局部重画）
- 打分维度（任务口径）：① 内容咬合 ② 视觉清晰（含 QUALITY.md 可读性/层级/细节）③ 一致性（palette/字体/spacing/helper 先例 + 页面级 PULSE_GRAD_ID）④ 动效/reduced ⑤ 可达/稳健（锚点唯一/census/aria/无孤儿）

## 0. census 与全局纪律核验

- 注册表：11.1 = 3 replacesBlock；11.2 = 4 replacesBlock；11.3 = 3 replacesBlock + 3 afterHeading → **10 + 3 = 13** ✓；13/13 显式 `dim:'gpu'` ✓
- 逐章无语言围栏块（grep `^``` ` 配对核验）：11.1 b0(L45)/b1(L129)/b2(L172)（L231/L253 两 ```bash 跳过）；11.2 b0(L29)/b1(L95)/b2(L130)/b3(L207)（L45 ```cpp、L59/L260 ```bash 跳过）；11.3 b0(L33)/b1(L122)/b2(L251) → **10 块全替换、每块恰一次、5 围栏跳过** ✓ 无漏块无孤儿
- afterHeading 唯一命中：11.3 `^## 3\. `→L50、`^### 5\.2 `→L147、`^## 6\. `→L171 各 1 hit ✓
- 脉冲渐变：import `PULSE_GRAD_ID`（fig-pulse-grad），pathPulse 全引用页面级 id；grep `<defs` = 0、`sig-` = 0（无 per-scene defs、无入场/无限 CSS 动画）✓
- helper 先例：cap 提权 wrapper / arrowHead 与 gpu-p09/p10 逐字同构 ✓；**pathPulse 增 `calcMode="spline" keySplines="0.42 0 0.58 1"`**——与 p09/p10 的线性手动脉冲不同，但与 kit edge 脉冲缓动逐字一致（figure-kit.mjs L78-82），属向 kit 口径收敛的改进，不判一致性罚（§4 注记）
- 截图时刻语义：全模块最大 delay 1.2s（fig-7 纵链末跳）< 3.5s ✓；截图光点分布与相位推算一致
- reduced 终态：独立自跑 13/13 diffpx=0、`.dia-pulse` display=none ✓（实现者报告同结论，本轮复核成立）
- 实现者 6 项 layout fix 复核**全部成立**：fig-2 SM 侧题注移卡下 y220（不压配对线）；fig-3 zone 标签 y70 / launch y78 / CPU 条 y84 三层分离；fig-6 A 标签 x600 居 ridge 虚线 x589.6 右侧；fig-7 zone 标签 y122 与 chip y130 分离；fig-8 seal 标签 y450 与首行 y462 分离；fig-10 三 zone 标签与首行分离。**但另有 2 处未列入的碰撞残留**（§2）

## 1. 逐图打分（13 图）

| fig | 函数（锚） | 咬合 | 清晰 | 一致 | 动效 | 稳健 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | serverForms（11.1 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 1 | pcieNuma（11.1 b1） | 5 | 4 | 5 | 4 | 5 | pass |
| 2 | migSlices（11.1 b2） | 5 | 5 | 5 | 4 | 5 | pass |
| 3 | nsysTimeline（11.2 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 4 | bottleneckFlow（11.2 b1） | 5 | 5 | 5 | 4 | 5 | pass |
| 5 | tuningTree（11.2 b2） | 5 | 5 | 5 | 4 | 5 | pass |
| 6 | achievedRoofline（11.2 b3） | 5 | **3** | 5 | 4 | 5 | **REDRAW** |
| 7 | faultChain（11.3 b0） | 5 | **3** | 5 | 4 | 5 | **REDRAW** |
| 8 | xidTree（11.3 b1） | 5 | 5 | 5 | 4 | 5 | pass |
| 9 | healthLayers（11.3 b2） | 5 | 5 | 5 | 4 | 5 | pass |
| 10 | eccRemap（11.3 §3） | 5 | 5 | 5 | 4 | 5 | pass |
| 11 | thermalThrottle（11.3 §5.2） | 5 | 5 | 5 | 4 | 5 | pass |
| 12 | linkDegradation（11.3 §6） | 5 | 5 | 5 | 4 | 5 | pass |

### fig-0 serverForms — 5/5/5/4/5
- 咬合 5：三形态卡逐行对 §2 表（8×SXM+NVSwitch H100 4 颗/B200 2 颗、无 CPU/主机内存、OEM 自由度；2×Xeon+2TB+8×CX7+6×3.3kW、8U/10.2、1105 CFM@80%、10U/14.3、1550 CFM；18 tray=72GPU+36Grace、9 switch tray 2×72-port、8 shelf、C2C 900、cold plate+manifold、leak detection）；域聚合 7.2/14.4/130(129.6) 对 §3 表；54.9/55.9/72%、1.275/1.7875/1.667、3483×、6.67 m³/s≈14100 CFM vs 1.9 L/s、N+N 132≥120 / 13.2≥10.2 / 16.5≥14.3、≥3 PSU 降级阈值、400 GB/s 单向出口——逐数对 §4/§3。
- 清晰 5：三卡横排递进 + 双强箭头脉冲；账条 soft/soft/seal 值标签随条尾不溢（130 TB/s 标签 x576 起 < 720）；底部五行注记字级阶梯清楚。动效 4：递进双脉冲 0/0.4 接力。

### fig-1 pcieNuma — 5/4/5/4/5
- 咬合 5：教学示意声明在题注；每 socket 4GPU+4NIC、8480C、inter-socket 40 GB/s 教学假设、min(64,40)=40=62.5%、numactl/nvbandwidth、GDR min(NIC50,PCIe64)=50、topo -m、GPU↔GPU 走 NVLink、GB200 C2C 900=7×128 双向、PCIe 退 NIC/NVMe（4×CX7+2×BF3）——逐条对 §5。
- 清晰 4：H2D 本地路径三段强线+脉冲方向正确（DRAM→CPU→switch→GPU0）；**inter-socket link / 40 GB/s 两标签横穿 48px 域隙两侧虚线 zone 边**（crop f1-isl 实证：虚线自字隙后穿过，深字浅线可读）——隙内置标固有、注记级（§4），沿 W4 threading 判 4 先例。
- 动效 4：三跳阶梯 0/0.3/0.6 + inter-socket 0.2。稳健 5：aria 与图同口径。

### fig-2 migSlices — 5/5/5/4/5
- 咬合 5：mem 1/8×8（10GB+带宽+MC+L2）/ SM 1/7×7（132/7=18.86 分数记账）；GI_k 配对、mem7 空闲 10GB（命题 2）、min(7,8)=7、A30=4、分区守恒 ≤1；GI/CI 两级 + MPS 反例；式 3 四行 18.9/10/418.75/6.25（**3350/8 为 11.1 口径**，与 11.2 的 3352 乘法链区分正确）；L2 方向 seal 注 + H100 1g.20gb 1/8 例外 vs B200 1g.45gb 2/8 + 反推错 12.5 MB；B200 1g.23gb 148/7≈21.1 **[B]**、22.5→名 23gb、962.5——逐数对 §6 含坑注。
- 清晰 5：配对曲线居卡隙、mem7 虚线短 stub + 「无 SM 切片可配」居右不压 k=6 曲线（截图实证）；SM 题注移卡下后三层分离；双 zone + seal 注标签与首行留隙。动效 4：单配对脉冲点题。

### fig-3 nsysTimeline — 5/5/5/4/5
- 咬合 5：四行 trace（CPU launch 刻度/NVTX 投影/kernel 间隙/copy CE）；18>12、间隙 6 μs；三段账 3960/5934=329×18+12/0.667、**gap 1974=33.3% 占墙钟**（非 wall/busy）、graph ≈3965/1.50× kernel 没动；signature 三件 + 归因三类 + drill-down 顺序——逐数对 §2 式 (1)(2)(3)。
- 清晰 5：间隙底衬 24px 对 kernel 48px = 6:12 比例自证；correlation 虚线居标签左不穿字（crop f3-corr2 实证）；账条 seal+warm 分段、gap 注居中 warm 段下方；zone 标签/launch/CPU 条三层分离（fix 复核成立）。动效 4：CPU 行单脉冲连轴转语义。

### fig-4 bottleneckFlow — 5/5/5/4/5
- 咬合 5：duty→I_ach≥R→C≥L_req(208) 三区 + 系统级分支；NCU 输入 sub（I_ach=FLOPs/bytes、u_B/u_F←SOL、C←O×ILP×s_warp）逐字对 §4；stall 交叉验证非替代 + occupancy 已高仍 long_scoreboard 查 coalescing；命题 1 四因子 1901(56.7%)→2846(84.9%)、1.18× vs 1.50×——对 §4 式 (4)。
- 清晰 5：左判定链纵脉冲阶梯 0/0.2/0.4/0.6、右三区卡 + 是/否标各居边侧不压曲线；肘形否分支入 latency 卡左缘；四因子账条值标签随条尾。动效 4：七脉冲阶梯覆盖全判定路径。

### fig-5 tuningTree — 5/5/5/4/5
- 咬合 5：四泳道十手段与 §5 ASCII 逐行同构（含章节号 3.3/5.4/4.1/4.2/4.3/6.1/6.2/3.4/1.4/4.4/5.2/6.1）；命题 2 加速=流量比、strideCopy 1024→128 MiB/320.33→40.04 μs/8×、u_B 前后 100%、「u_B 到顶≠优化到顶」——对 §5 式 (5) 与对账表。
- 清晰 5：泳道标签与行留 44px、sig→tech 箭头列对齐 x215-245；running-example 脉冲居 coalescing 路。动效 4：单高亮脉冲点题（沿 W5 fatTree 先例）。

### fig-6 achievedRoofline — 5/3/5/4/5 → REDRAW
- 咬合 5：F_peak 989.5 / BW 3.35 / R≈295；A(400,700) u_F 70.7%、B(2.5,7.125) u_B 85.1%、**D(2.5,2.58) loss 69.2%（点标 D 非 C）**、E(0.083,0.238)；D→B 垂直 ILP 1→4 / C 64→256；三轨迹 + 点标口径注 + MFU 0.667×0.551%=0.368% + 两类 kernel 75.8/24.2% 贴斜线——逐数对 §6/§4。
- 清晰 3：**roof 斜线穿「C 64→256 sectors」标签字身**（crop f6-ilp2 实证：斜线在 x≈246-257/y255-262 划过「64→25」数字中部），并 graze「ILP 1→4」下缘（x≈273-285/y241-248）——定量注记被自图 roof 线划线，读数受扰；其余轴/刻度/四点/A 标签（ridge 右移 fix 成立）/底部四行干净。
- 一致 5、动效 4（D→B 单脉冲 0.3 垂直上移语义）、稳健 5（aria 含 D 非 C 口径）。重画要求见 §2-1（局部移两行注记即可）。

### fig-7 faultChain — 5/3/5/4/5 → REDRAW
- 咬合 5：检出四面/定位/处置四动作/系统层 serving+training/集群 pointer 与 §2 ASCII 逐行同构；三层各消费下一层结论 + 前提三件（rerouting/checkpoint/SDC 定点）对 §2 提点。
- 清晰 3：**clu 卡 h32 带 sub → sub 基线 = y+h/2+16 = 卡底边框（kit 公式），「节点 drain / 备机池 / 作业重调度」骑边框、descender 出框**（crop f7-clu2 实证）；serv/train h34 sub 余量仅 1px 贴边——与 W3 R1（gpu-p05 fig-0，h32 sub 骑边框判细节 3 → redraw，fix e423b50 h32→44）同类同判。纵链五跳脉冲穿卡隙正确、zone 标签与 chip 分离（fix 复核成立）。
- 一致 5、动效 4（0/0.3/0.6/0.9/1.2 阶梯）、稳健 5。重画要求见 §2-2（卡高局部修）。

### fig-8 xidTree — 5/5/5/4/5
- 咬合 5：七支 13/31/43、48/94、63、64/95、74/149、79、154 与 §4.4 ASCII 逐行同构；31 重复→EUD、63/64/177 三者之一、64 才立即、74 同情错误、154 五档逐字、**74 Blackwell→149 NETIR（144-150 decode）+ SXid 仅 Hopper/NVL72 走 FM** seal 硬注——对 §4.1-4.4。
- 清晰 5：主干+七支对齐、main/sub 双行行距 42px 不互压；seal 标签与首行分离（fix 成立）；全文字 < 684。动效 4：首支单脉冲点题。

### fig-9 healthLayers — 5/5/5/4/5
- 咬合 5：四层（in-band 监控/事件驱动/diag r1-r4/带外）与 §7.3 ASCII 同构；PCIe replay 202、watch/health/policy 三能力、r3 内嵌 nvbandwidth/nccl_tests、命令三例、r1<2.5s/r2 8GPU<10.5min/r3<35min/r4<2.25h、13× 成本差、RMA 归 Field Diagnostic 边界逐字——对 §7.1-7.2。
- 清晰 5：四 band 标签与行留 40px、band 间 12px 隙纵脉冲阶梯 0/0.3/0.6；底三行注记不出框。动效 4。

### fig-10 eccRemap — 5/5/5/4/5
- 咬合 5：CE 趋势/UE 事件(XID48)、volatile/aggregate+breakdown、containment 94/95 多数无需 reset、page offlining 少一页、**remap 512=8×64** + GPU reset 生效/终身/无空洞 + CE 可被 UE 替换、RAS Repair 同 bank 2 次后第 3 次 UE→channel/L2 slice→**XID 160 reboot**、**RMA 三条件析取（8/bank 充分非必要）**+ SRAM parity>4/SECDED>2——逐条对 §3.1-3.4 命题 1。
- 清晰 5：512/64 双账条值标签随条尾；三 zone 标签与首行分离（fix 成立）；CE/UE 双分支脉冲 0/0.3。动效 4。

### fig-11 thermalThrottle — 5/5/5/4/5
- 咬合 5：三档温度 SW/HW 触发、**1980→990 MHz、66.9→33.45 TFLOPS（÷2 边界值）**、80% 峰值 53.5→26.8、memory-bound I·BW(3352) 不变、φ 式 X(1−φ/2)、0.10→95%/0.50→75%、持续占比=冷却信号、ridge 重判口径注——逐数对 §5.2 命题 2/式 (1)。
- 清晰 5：左三档卡链 + 右四账条 + 底 φ 双条三层；值标签随条尾不溢（33.45 标签 x554 起 < 720）。动效 4：**纯静态账图无动效需求，reduced 天然终态**（沿 W3 fig-1 warpShuffle 静态判 4 先例）。
- 注记：监控面计数器名为缩写（sw_thermal/hw_thermal/sw_power_cap/hw_power_brake vs 章全名 *_slowdown），语义同、不判罚（§4）。

### fig-12 linkDegradation — 5/5/5/4/5
- 咬合 5：三族 400/409、420/429、430/439 严重度阶梯 + flit/replay/recovery 语义；命题 3 BW(1−p) 三档 0.0001%/1%/10%；**18 link 断 1→425/450=94.4%**、k=2 400(88.9%)、双向 850/900；max 语义 + 9.3 算例 37.5→75 ms 同构；诊断链 + 双坑——逐数对 §6.1-6.3。
- 清晰 5：18 格 link strip 单红格 + 「断 1」标居格下；三族卡升级箭头脉冲 0/0.4；左右双栏账不交叠。动效 4。

## 2. redraw findings（任一维 <4）

1. **gpu-p11 fig-6 achievedRoofline（11.2 b3）清晰=3** —— roof 斜线穿定量注记字身。
   - `site/scripts/lib/figures/gpu-p11.mjs` **L468-469**：`cap(305, 248, 'ILP 1→4', …anchor:'end')` / `cap(305, 262, 'C 64→256 sectors', …anchor:'end')`。斜线（L446 path）在 y255-262 带位于 x246-257、y241-248 带位于 x273-285，正穿两标签（C 标签 span 233-305、ILP span ≈270-305）。
   - 修：两行下移至 **y276 / y290**（anchor end 305 保留）：该两带斜线位于 x≤230，与标签 span 233-305 分离；D 标签二行 x325/y276 与 ILP 新位 x≤305 不交；E 标签 y338 在下不碰；D 点底 y266.5 在新 ILP 带顶 y269 之上。勿左移 x245 方案（会撞 L451-452「斜线 = I × BW / BW = 3.35 TB/s」x150-222）。
   - 其余维 ≥4：四点/轴/刻度/轨迹/配色/脉冲保留，局部移注记即可。复审通道：重跑 `node scripts/fig-check.mjs gpu-p11` + fig-6 crop 复核斜线不穿字 + reduced 单帧。
2. **gpu-p11 fig-7 faultChain（11.3 b0）清晰=3** —— clu 卡 sub 骑卡底边框（W3 R1 同类）。
   - `site/scripts/lib/figures/gpu-p11.mjs` **L520**：`node({ id:'clu', x:230, y:400, w:260, h:32, …sub:'节点 drain / 备机池 / 作业重调度' })` → sub 基线 432 = 卡底 432。修 **h32→40**（底 440、sub 436 留 4px；L530 cap y452 隙 12px；scene 486 不变）。
   - 同修 **L516-517** serv/train **h34→40**（sub 378 对底 382 留 4px；zone y312 h72 底 384 仍容卡底 382；L524 纵链 M 360 384 V 400 不变）；loc h36 余量 2px 登记注记（§4）。
   - 其余维 ≥4：纵链/zone/脉冲/配色保留。复审通道：fig-check + fig-7 crop 复核 sub 离框。

## 3. 内容咬合专项核验（任务清单）

- **gap/wall 33.3%**：fig-3 L294/300「gap 1974 μs = 33.3%」对 02 章 L77（1974/5934）✓ 占墙钟非 busy；fig-tag/aria 同口径 ✓。
- **roofline 点标 D 非 C**：fig-6 L476「D (2.5, 2.58) latency-bound」+ L481 口径注（2.4 §7 标 C 本章改 D）对 02 章 L231 ✓ 坐标/loss 不变 ✓。
- **B200 SM 148 [B]**：fig-2 L244「SM 148/7 ≈ 21.1 [B]」对 01 章 L227（datasheet 不印、微基准口径）✓。
- **MIG L2 _asymmetry**：fig-2 L242-243 H100 1g.20gb L2=1/8(6.25) 例外 vs B200 1g.45gb 2/8 + 反推错 12.5 MB，对 01 章 L225 坑注逐字 ✓；L2 随 memory **方向** seal 标题 ✓。
- **XID 74 Blackwell→149**：fig-8 L543 题注 + L554 行 + L573 seal（B-x00/GB-x00/Rubin=NO → 149 NETIR、144-150 decode）+ L574 SXid/FM，对 03 章 L116 ✓。
- **ECC 512=8×64**：fig-10 L631/655-659 双账条 + reset 生效/终身，对 03 章 L68-69 ✓；RMA 析取 L670-671 对 L78 ✓。
- **thermal 66.9→33.45**：fig-11 L707-711 对 03 章 L155 ✓；memory-bound 不变 L714 ✓；φ 95%/75% L719-724 对 L163 ✓。
- **link 94.4%**：fig-12 L776「17×25 = 425 = 满配 450 的 94.4%」对 03 章 L209 ✓；k=2 88.9%、37.5→75 ms L781 对 L211 ✓。
- **PCIe min(64,40)=40**：fig-1 L172「min(64, 40) = 40 GB/s —— 只剩本地 62.5%」对 01 章 L148 ✓。
- 其余抽查：域聚合 7.2/14.4/130(129.6)（fig-0 L87 对 01 章 L76）、N+N 132≥120（L98 对 L109）、3483/6.67/1.9（L97 对 L119）、四因子 1901/2846（fig-4 L364/367 对 02 章 L122）、L_req 208（fig-4 L328 对 02 章 L93）、MFU 0.368%（fig-6 L482 对 02 章 L203）、strideCopy 8×（fig-5 L414 对 02 章 L178）、RAS XID160 reboot（fig-10 L666 对 03 章 L72）、diag r1-r4/13×（fig-9 L616-617 对 03 章 L225-230）——逐字对章 ✓ **无发明数字**。
- 口径区分正确：fig-2 带宽 3350/8=418.75（11.1 口径）vs fig-11/12 的 3352（11.2/11.3 乘法链口径），两章各自闭合 ✓。

## 4. 注记级瑕疵（不判罚、登记备查）

- fig-1 L131-132：inter-socket 两标签横穿域隙虚线 zone 边（48px 隙置 60-76px 标固有；深字浅线可读，沿 W4 threading 判 4 先例）。
- fig-7 loc 卡 h36 sub 余量 2px（不骑框）；serv/train 1px 贴边随 §2-2 同修。
- fig-11 L726：计数器名缩写（sw_thermal/hw_thermal/sw_power_cap/hw_power_brake vs 章 §5.2 全名 *_slowdown/_brake_slowdown），语义同；建议随 polish 轮补全名。
- pathPulse 增 spline 缓动：与 p09/p10 逐字版不同但向 kit edge 口径收敛（改进）；后续波次建议回灌 p09/p10 统一。
- 死变量：L108 `UID='g11b'`、L317 `UID='g11e'`、L627 `UID='g11j'` 声明未用（无 kit edge uid 需求），无害。
- fig-6 底注 elementwise u_B 85.1% 承自 02 章 L201 表（0.238/0.278≈85.6% 为章侧舍入），图忠实镜像不判图罚。

## 5. 汇总

- **13 图：11 pass / 2 redraw**。redraw = fig-6 achievedRoofline（清晰 3，斜线穿 C/ILP 注记）、fig-7 faultChain（清晰 3，clu sub 骑卡底边框，W3 R1 同类）。
- 最低分维：3（fig-6/fig-7 清晰）；pass 图最低为 4（fig-1 清晰 4 域隙标 threading；全图动效 4 沿 W3/W4/W5 已审 spline/paced/静态口径）。
- 全局纪律：census 10+3+5 skip ✓、锚点唯一 ✓、dim gpu 13/13 ✓、页面级 PULSE_GRAD_ID ✓、无 defs/sig- ✓、delay ≤1.2s ✓、reduced 13/13 diffpx=0（独立自跑）✓、无孤儿 ✓、实现者 6 项 fix 复核成立 ✓。
- 复审通道：§2 两处局部修（fig-6 移 L468-469 两行 y276/290；fig-7 L516-517/L520 卡高 40）后重跑 `node scripts/fig-check.mjs gpu-p11` + fig-6/fig-7 crop 复核 + reduced 单帧即可闭环；其余 11 图无需动。

## 6. redraw closure addendum（2026-11-26，scoped re-check of commit 5bfc2d0）

- 范围：仅复核 fig-6 achievedRoofline 与 fig-7 faultChain（commit **5bfc2d0**，1 file / +5−5，只改 fig-6 L468-469 两行 cap y 与 fig-7 L516-517/L520 三卡 h）；其余 11 图维持 §1 原判，未重审（diff scope 证实未触及）。
- 证据：重跑 `node scripts/fig-check.mjs gpu-p11` → 13/13；Read fig-6/fig-7 全图 + sips crop（/tmp/figqa-w6-crops/f6-close.png、f7-close.png）；独立 reduced 自跑（/tmp/figqa-w6-review-reduced.mjs）13/13 diffpx=0、`.dia-pulse` display=none（31 脉冲）。

### Fig-6 Verdict: ADDRESSED
- L468-469 两 cap 下移 y276/y290（anchor end 305 不变）。几何：斜线在 y269-276 带位于 x≈220-230、y283-290 带位于 x≈193-204，均左于标签 span（ILP ≈270-305 / C 233-305）→ 无字身相交；crop 实证斜线自两注记左上掠过、留白楔区干净。
- 邻接复核：D 点底 y266.5 在新 ILP 带顶 y269 之上；D 标签二行 x325/y276 与 ILP（x≤305）x 向分离；E 标签 y338 在下不碰；左「斜线 = I × BW / BW = 3.35 TB/s」（x150-222, y246/260）与新 C 位（x233-305, y290）双向分离。四点/轴/刻度/A 标签/底部四行与 pass 期渲染逐元素一致。
- 复评分（任务五维）：咬合 5 / 清晰 5 / 一致 5 / 动效 4 / 稳健 5 → 最低 4。

### Fig-7 Verdict: ADDRESSED
- clu h32→40：sub 基线 436 对卡底 440 留 4px，crop 实证「节点 drain / 备机池 / 作业重调度」全入框、边框在字下；serv/train h34→40：sub 378 对底 382 留 4px，两 sub 离边；zone 底 384 仍容卡底 382，纵链 M 360 384 V 400 与底注 y452（隙 12px）不变。
- 其余纵链/zone 标签/chip/脉冲与 pass 期一致；loc h36 余量 2px 维持 §4 注记（不骑框）。
- 复评分（任务五维）：咬合 5 / 清晰 5 / 一致 5 / 动效 4 / 稳健 5 → 最低 4。

### Gate: PASS（13/13 各维 ≥4）
- §1 fig-6/fig-7 原打分（清晰 3）与 §2 redraw findings 保留为重画前原判（历史记录）；重画后两处闭环，11 pass + 2 redraw → **13/13 QUALITY pass**。
- 注记级备查（§4）不变：fig-1 域隙标 threading、fig-7 loc 2px 余量、fig-11 计数器缩写、pathPulse spline 回灌建议、死 UID 变量。
