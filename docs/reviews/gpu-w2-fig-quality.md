# gpu W2 场景图 QUALITY 评审档案（gpu-p03 / gpu-p04，18 图）

- 评审方式：`cd site && node scripts/fig-check.mjs gpu-p03 && node scripts/fig-check.mjs gpu-p04` 渲染后逐张 Read 实拍 PNG（`site/qa/figs-gpu-p03/fig-0..9.png`、`site/qa/figs-gpu-p04/fig-0..7.png`）；p04 fig-5/fig-6 的脉冲落点另用 sips 局部裁剪复核（`/tmp/figcrop/f5-rows.png`、`f6-rows.png`）。源码目测仅作佐证，打分以截图为准。
- 规：`site/scripts/lib/figures/QUALITY.md` 五维（动效自然度 / 场景可读性 / 视觉层级 / 细节完成度 / 与正文咬合），1–5 分，任一维 <4 打回重画。
- 内容 fidelity 对照：`gpu/03-cuda-programming-model/*.md`、`gpu/04-kernel-memory-optimization/*.md` 原文逐数核对（见末节）。
- 结论：**17 pass / 1 redraw**（redraw = gpu-p04 fig-5 doubleBuffer，细节完成度 3）。

## 0. census 与全局纪律核查

### census（14 replacesBlock + 4 afterHeading = 18，与注册表逐条核对通过）

| 章 | 无语言围栏块（行号，0 起序号） | 注册图 | 命中 |
|---|---|---|---|
| 3.1 | 24–35 (#0)、89–95 (#1)、142–151 (#2) | threadHierarchy / simtDivergence / blockScheduling | b0/b1/b2 ✓ |
| 3.2 | 28–36 (#0)、172–177 (#1=PTX atomic 伪代码，不画，归算法面板) | addressBacking b0 ✓；fenceScope afterHeading `/^## 2\. /` → 唯一命中 line 46 ✓ |
| 3.3 | 25–32 (#0)、92–103 (#1)、151–156 (#2) | streamQueue / overlapTimeline / graphAmortize | b0/b1/b2 ✓ |
| 3.4 | 54–66 (#0)、148–155 (#1) | occupancyFlow / ncuUtilization | b0/b1 ✓ |
| 4.1 | 40–54 (#0) | coalescingSectors b0 ✓；strideSaturation afterHeading `/^## 3\. /` → 唯一 line 60 ✓ |
| 4.2 | 25–32 (#0) | bankConflict b0 ✓；paddingSwizzle afterHeading `/^## 4\. /` → 唯一 line 91 ✓ |
| 4.3 | 37–52 (#0)、141–148 (#1) | gemmTiling / doubleBuffer | b0/b1 ✓ |
| 4.4 | 95–102 (#0) | multiStagePipeline b0 ✓；tmaMbarrier afterHeading `/^## 3\. /` → 唯一 line 46 ✓ |

合计 replacesBlock = 9(p03) + 5(p04) = 14；afterHeading = 1 + 3 = 4。四个 afterHeading 正则均精确命中一行（grep '^## ' 核对）。

### 脉冲渐变 / reduced-motion / 截图时刻

- 页面级 PULSE_GRAD_ID：两模块均 `import { PULSE_GRAD_ID }`（= `fig-pulse-grad`，figure-kit.mjs:13），pathPulse 与 kit edge 脉冲一律 `url(#fig-pulse-grad)`；模块内 grep `radialGradient|<defs|linearGradient` 为空 → 无 per-scene id 错配 ✓。
- reduced-motion 静态终态：figure.css:201–203 `.dia-pulse{display:none}`（SMIL 光点整点隐藏）；figure.css:264–277 `.fig.in .sig-cell/.sig-bar/…{opacity:1;animation:none;transform:none}` → sig-bar 满宽、sig-cell 全亮的静态终态 ✓（本批只用 dia-pulse / sig-cell / sig-bar，全部覆盖）。
- 截图时刻语义：pulse delay 最大 2.0s（p04 fig-4 eid g）< 3.5s；sig-cell --i ≤ 1.072、sig-bar --i ≤ 0.6，sig-grow 0.7s + 0.4s + --i·0.2s ≤ 1.22s → 截图时条已填满、格已亮 ✓。

## 1. gpu-p03 逐图打分

### fig-0 threadHierarchy（3.1 b0）— 5/5/4/5/5 PASS
- 动效 5：层级链 pulse delay 0/0.3/0.6 + 驻留链 0.9/1.3/1.7 两级cascade，截图时四段光点均在边中段（fig-0 链上红点）。
- 可读 5：grid→block→warp→thread 左→右单链，「含 ×65536/×8 warp/×32」压在对应边上方；物理驻留分区在下，逻辑/物理一眼两本账。
- 层级 4：主链与驻留总数 270336（attn 14px）双焦点略竞争；底部 4 行 10px 注记块偏密（fig-0 y316–370）。
- 细节 5：节点 rx12、箭头显式三角、SM0 卡内 8 block 两行对齐；末行 baseline 370 < viewBox 388，无溢框。
- 咬合 5：65536×256=16777216、1056/8448/270336、62 波、257→9 warps、25637 双射全部对原文 line 25/33/37/55/155。

### fig-1 simtDivergence（3.1 b1）— 5/5/4/5/5 PASS
- 动效 5：lane sig-cell --i 0.1+j·0.012 / 0.7+j·0.012 逐 lane 亮起 + 两槽 pathPulse 0/0.8 + 案例 sig-bar 0–0.5，节奏分层。
- 可读 5：左两发射槽（16 热格 + 16 虚线 mask 格 / 镜像）串行语义自明；右 D1–D5 槽数条 + η 列对照。
- 层级 4：底部 5 行注记（式 3 bold + 4 行 10px）墨量接近主场景，注记层略抢（fig-1 y262–336）。
- 细节 5：η 列 anchor=end 684 对齐；D4/D4′ 条 96px 等长兑现「同槽数」；zone 宽 330 内 lane 行止于 357.6 无溢。
- 咬合 5：D1/D2 均 1/2、D3=1、D4 0.1875 / D4′ 0.8125 差 4.33×、D5 1/4、p=0.99 → 1.275 槽 η≈0.784 且标注「聚合口径，非 E[η]」与原文 line 122–128 逐字同口径。

### fig-2 blockScheduling（3.1 b2）— 5/5/4/4/5 PASS
- 动效 5：grid→sch→zone 三段 pulse 0/0.5 + 波次条 pathPulse 0.6，截图光点在满波段内（fig-2 条上红点）。
- 可读 5：调度链 + SM0/SM1/⋯/SM131 驻留排 + 600px 满波条 vs 6px 尾条，尾波 6.1% 一眼可见。
- 层级 4：波次账 bold 10.5 与红尾注记构成第二焦点，底部 4 行注记密（y304–358）。
- 细节 4：右侧第三行 cap「PG 代表版本均无 GigaThread 一词…」止于 ≈715/720，右边距仅 ~5px，与他行留白不齐（fig-2 右上）。
- 咬合 5：62.06→62 满波 + 尾 64、6.1%、grid-stride 63 次迭代、524288÷8448=62.06 两口径、GigaThread Fermi 时代注均对原文 line 138/155/71。

### fig-3 addressBacking（3.2 b0）— 5/4/4/4/5 PASS
- 动效 5：六映射边 pulse 0/0.4/0.8（实线前三）+ 后三虚线无脉冲，动静区分逻辑主路径与旁路。
- 可读 4：local→HBM 实线与 shared→SRAM 实线存在一次横穿交叉（fig-3 中部 ≈x340,y165）；交叉干净但属可避免的顺序反转交叉。
- 层级 4：左六卡等重、右四 backing 等重，主消息（local 在 HBM）靠 attn 染色 + 标题承担，场景内无更重主链。
- 细节 4：「spill /（2.3）」cap 骑在左 zone 虚线右边框上（x292 跨 x286 边框）；HBM 左端口叠 4 枚同位箭头三角（视觉单枚，可接受但属堆叠）。
- 咬合 5：256 KB RF / 统一 256 KB SRAM / 29.0 / 478.8 / 16.5× / 64 KB=16384 fp32 / 512 KB / spill 三类 / ld.global.nc 边界均对原文 line 17–21/38/40/236。

### fig-4 fenceScope（3.2 afterHeading §2）— 5/5/4/4/5 PASS
- 动效 5：消息传递链 4 段 pulse delay 0/0.4/0.8/1.2，截图四红点各在边中（fig-4 下链）。
- 可读 5：scope 五级宽度递减漏斗 + fence 三级 ≡ 映射 + 底部因果链三面板，ordering/同步边靠 sub 标注区分。
- 层级 4：三面板等重、底部 3 行注记含红字价格线，注记层墨量偏大（y356–392）。
- 细节 4：footer「morally strong 三条件…」行止于 ≈710/720，右边距 ~10px 偏紧（fig-4 倒数第二行）。
- 咬合 5：membar.cta/gl/sys ≡ fence.sc.cta/gpu/sys、均 seq_cst、pre-SM90 cluster 放宽 device、270336 对 256 差 1056×、263.0≈143.7 ns 对 40.7 快 6.46× 但错、morally strong 三条件均对原文 line 62–78/144。

### fig-5 streamQueue（3.3 b0）— 5/5/4/5/5 PASS
- 动效 5：stream2→q0 pulse 0.3 + q0→SM 1.0 + q0→CE 1.4 错峰；stream9 共队列边虚线静止表「额外映射」。
- 可读 5：三 stream 扇入 queue 0 + stream 9 虚线汇入同端口，假依赖一图自明；SM/CE 双出口分叉干净。
- 层级 4：底部 5 行注记（含 PG 逐字行）密，主场景与注记墨量接近（y282–352）。
- 细节 5：⋮ 省略符居中对齐；「共 queue」红注落在两 zone 间空白；无溢框、无叠线。
- 咬合 5：默认 8 条 / 1–32 / CC 8.0+ MAX_COPY_CONNECTIONS / false dependence 逐字 / NULL stream 隐式同步 / stream 内有序间无序均对原文 line 23–34/221。

### fig-6 overlapTimeline（3.3 b1）— 5/5/5/4/5 PASS
- 动效 5：三资源行 pathPulse 0/0.6/1.2 各沿本行流动，截图三点分别在 H2D/SM/D2H 行（fig-6 下面板）。
- 可读 5：串行 vs 流水双面板按真实时间比例（500px/3205.79 与 500px/2235.73），三行资源 + 双时间轴，k 细条藏于 H2D 背后一眼可读。
- 层级 5：zone 标题带公式、节省行 bold、红坑注收尾，三层字级清晰且留白足。
- 细节 4：流水面板轴刻度 baseline 310 贴 zone 底 312（descender 触边框）；截图时刻脉冲瞬遮 c7/d3 格标签（运动态可接受）。
- 咬合 5：3205.79/2235.73/400.724/262.144/7.508/131.072/2097.152/1048.576/60.062/970.06/30.3%/1.434/1.529/65%/3145.73/60.06/1.9%/16×/每 chunk 2097152 元素 16/8/24 MiB 全对原文 line 111–137。

### fig-7 graphAmortize（3.3 b2）— 5/5/5/5/5 PASS
- 动效 5：两阶段边 pulse 0/0.5 + 三档 sig-bar 0/0.2/0.4 左→右生长，截图条已满。
- 可读 5：definition→instantiation→execution 单链 + 每阶段成本 cap 挂边下方 + 三档暴露开销条，摊销故事一眼完整。
- 层级 5：inst 节点 attn + 红 t_inst cap 为唯一重焦点；sync 6.7 条最重、graph 0.5 近黑最短，对比即结论。
- 细节 5：条标签 end@240 对齐、值随条尾 +8px；500000 元素/512 threads 注记行止于 ≈675 无溢。
- 咬合 5：400/3.4/9.6/3.8/2.9/6.7/0.9/0.5/1000 次/0.02/330×5=1.65/1.645/33%/cudaGraphUpload/静态三要求均对原文 line 186–196/145。

### fig-8 occupancyFlow（3.4 b0）— 5/5/4/5/5 PASS
- 动效 5：use/inv→gran 0/0.3 →mn 0.7 →wa 1.1 →oc 1.5 计算流 cascade，截图各段光点在位。
- 可读 5：双输入汇粒度修正→四约束 min→W_active→O 纵向主链 + API 虚线侧挂「权威值」，方向无歧义。
- 层级 4：mn 与 oc 双 attn 焦点 + 底部 4 行注记 + 两侧 7.5px 小注，注记面偏大（y306–410）。
- 细节 5：api 三行 cap 居中 622 止于 ≈674；wa→oc 20px 短边脉冲+箭头填满属瞬态；无溢框。
- 咬合 5：S_alloc=⌈S_sh/128⌉×128+1024、min(32,8,⌊65536/8192⌋=8,∞)=8、768→2 blocks 75%/32→32 blocks 50%/100 KB→2、CC 7.0 37 regs×320 手算 5 对官方 4、b_act≤B_max 方向均对原文 line 35/74–82/108。

### fig-9 ncuUtilization（3.4 b1）— 5/5/5/5/5 PASS
- 动效 5：四条 allocated 段 sig-bar --i 0/0.2/0.4/0.6 依次生长，截图已满宽。
- 可读 5：register 条近满 + warm 零头 vs 其余三条大段 soft unallocated，「谁零头满谁绑定」不用读正文。
- 层级 5：ink allocated 最重、三态图例两行收尾、读法/锯齿/废 Excel 三行注记分层清晰。
- 细节 5：值列 end@684 对齐（49152/65536 等）、条总宽 440 四行一致、图例色块与条色同源。
- 咬合 5：K2 256 threads/R_th 96/S_sh 32 KB → min(32,8,2,6)=2、W=16、O=0.25、49152/67584 分配数、NCU replaces 逐字均对原文 line 108/144–149。

## 2. gpu-p04 逐图打分

### fig-0 coalescingSectors（4.1 b0）— 5/5/5/4/5 PASS
- 动效 5：四行 sector 条 pathPulse 0/0.5/1.0/1.5 错峰扫过，截图各点在本行条上。
- 可读 5：s=1 全黑 / s=2 黑赭相间 / s=16 每 sector 独黑一字 / misaligned 首尾赭残段，四种模式有用 vs 搬废一眼分。
- 层级 5：红 bold 跨度反例行（63≠32）为唯一重注记；右列 4 sec/η 判定对齐成账列。
- 细节 4：截图时刻脉冲瞬遮 s=1 sector 1 首字与 misaligned sector 1 中字（中心线走点的_house 代价，运动态可接受）。
- 咬合 5：**方向核对通过**——stride-16 distinct=32、⌈1988/32⌉=63 高估；misaligned 5 sector、⌈128/32⌉=4 低估；η=1/8、1024 B、0.8、BP「stride of 2 → 50%」逐字均对原文 line 37–38/52–58/86/127。

### fig-1 strideSaturation（4.1 afterHeading §3）— 5/5/5/4/5 PASS
- 动效 5：五档 sig-bar --i 0–0.6 生长 + 右侧机制条 pathPulse 0.8，截图地板两档等长已兑现饱和。
- 可读 5：左 η·BW 递减条封在 419 地板 + 右「每 lane 独占 sector」机制条，饱和因果两面板互证。
- 层级 5：s=8/16 seal 红条 + 红地板标签为焦点；s=1 soft 最长条退为背景对照。
- 细节 4：32-sector 地板虚线参考线画在条填充**之上**（代码序：bars 后 line），s=1/2/4 条面被虚线横穿，读感似删除线；参考线宜垫底。
- 咬合 5：3352/1676/838/419/419、209.5 低估 2×、η=1/min(s,32/bytes)、fp64 s=4 饱和、40.04 对 320.33 μs 全对原文 line 80–88/157–160。

### fig-2 bankConflict（4.2 b0）— 5/5/5/4/5 PASS
- 动效 5：32 字映射条 pathPulse 0 扫过 + 四案例卡静止对照，动静分工正确（映射是机制、案例是状态）。
- 可读 5：w0–w31→bank 0–31 条 + 四案例卡（stride-1/stride-32/broadcast/字0+字32）各挂 B/cycle，k 语义自明。
- 层级 5：stride-32 卡 attn + 红 4 B/cycle + 红 verdict 行三级递进聚焦最坏反例。
- 细节 4：截图时刻脉冲瞬遮字 29 格内数字（中心线走点_house 代价）。
- 咬合 5：k=max u_b、同址 broadcast 例外、字0+字32=16+16 lane→k=2→64 B/cycle、7.3<25.4、double k=2 非冲突损失、stride-16 两套粒度 14.6 GB/s 均对原文 line 42–59。

### fig-3 paddingSwizzle（4.2 afterHeading §4）— 5/5/5/5/5 PASS
- 动效 5：transpose 双条 sig-bar 0/0.2 生长（1056 对 64 周期对比即动画结论）；矩阵为状态图静止合规。
- 可读 5：左 padding 错行矩阵热列跨 bank（2/3/4/5）vs 右 swizzle 对角热格散 chunk，两条消冲突路视觉对立清晰。
- 层级 5：naive seal 长条 316.8px vs padded ink 19.2px，16.5× 不用读字。
- 细节 5：左矩阵行偏移 r·4 可视化 +1 列、bank 数 (r·33+c)%32 逐格对；「← 列访问」落在错行空白区不压格；无溢框。
- 咬合 5：gcd(33,32)=1、p=2 反例 gcd(34,32)=2、4096→4224/+3.125%/132 B 破 16 B 对齐、Swizzle<3,4,3> 式 4、ldmatrix k=8→1、1056/64 周期 577.0/34.97 ns、BP 12.8→140.2→199.4（10.95×+1.42×）均对原文 line 75–89/113/141/188。

### fig-4 gemmTiling（4.3 b0）— 5/5/5/4/5 PASS
- 动效 5：HBM→A/B→reg→MMA→C→写回 7 段 pulse 0–2.0 cascade，截图光点铺满主链各段。
- 可读 5：shared zone 双 tile「HBM 读一次」+ 复用 cap（T_N/T_M=128 消费）+ C 驻留末步写回，复用故事完整。
- 层级 5：naive 2.5px 微条 vs tiled 320px seal 条，128× 流量差即主结论；MMA/C attn 聚焦计算端。
- 细节 4：tiled 值注记「I=64 · 2.147 GB · 640.7 μs · 128×」止于 ≈697/720，右边距 ~23px 偏紧；mma→ct 长斜线干净但为全图唯一长线。
- 咬合 5：128×32/32×128/16 KB/8 A+8 B/64 fp32/128 步/I=T_tile/bytes/0.5→64/274.88 GB 82.00 ms→2.147 GB 640.7 μs/ridge 295、T≥590/33.55 MB 1.56%/tile 扫描 8.590/4.295/1.074 均对原文 line 68–79/87/119/240/256–257。

### fig-5 doubleBuffer（4.3 b1）— 5/4/4/3/5 **REDRAW**
- 动效 5：两行 pathPulse delay 0/0.4 错峰，运动本身平滑、reduced-motion 终态合规。
- 可读 4：同步/双缓冲两时序条重叠关系清楚，但**计算行脉冲跑道 y222 恰在计算条底 216 与装载条顶 228 的正中间**（等距 6px），光点归属哪一行一眼不定（裁剪图 f5-rows 上排红点）。
- 层级 4：三时间卡 + 双时序 + 4 行注记信息密度高，τ_db bold 行与红坑注双焦点略竞争。
- 细节 3：**脉冲跑道系统性脱离条中心线**——计算条中心 206 而 pulse path y222（条底下方 6px 的行间空白）、装载条中心 238 而 path y254（条底下方 6px、注记行上方空白）；截图两红点均浮在空白带（f5-rows 裁剪实证），与全批中心线走点先例（p03 fig-1/2/6、p04 fig-0/1/2）不一致，属对齐缺陷而非瞬态遮挡。
- 咬合 5：139.9/645.2/261.6/906.8/1046.7/13.4%/766.9/1.154/上界 2/6.6 KB 带宽延迟积/T_K=64 1831.8/1552.0=1.180/13→6 blocks register 绑定无损均对原文 line 133–175。
- **重画要求**：pathPulse 两条 path 的 y 改回条中心线（206 / 238），或为跑道加视觉归属（如贴条底边 ≤2px）；消除行间等距浮点。其余四维不动。

### fig-6 multiStagePipeline（4.4 b0）— 5/5/4/4/4 PASS
- 动效 5：装载行 pathPulse 0.3 + 三档对比 sig-bar 0/0.2/0.4，截图稳态槽与对比条语义完整。
- 可读 5：计算 4 槽 + 装载 tile 2 起错排、「恒有 2 份在飞」由 182.7px 重叠条直接可见；序章 cap 交代填充。
- 层级 4：式 1/式 2 两行 mono bold + 三档条 + 3 行推论注记，公式层墨量偏大（y194–212、326–362）。
- 细节 4：装载行 pulse path y150 在条底 144 下方 6px 空白带（f6-rows 裁剪红点），最近元素为上方装载条、归属可推断但非中心线，与 fig-5 同源轻症。
- 咬合 4：645.2/453.4/906.8/1046.7/1.622/1290.4/S_min=3/48 KB shared 项 4 blocks/96 KB/1.87/279.8≥261.6 均对原文 line 78–104；**唯「填充段占 128 步的约 1/128」为图侧自估**（原文只有「序章装 tile 0、tile 1」无分数），已用「约/可略」对冲、量级正确（≪1%），记 fidelity 观察不扣至 <4。
- 备注：装载 tile k+2 条 119px 为 viewBox 截断 + 「…」标注，合规。

### fig-7 tmaMbarrier（4.4 afterHeading §3）— 5/5/5/4/5 PASS
- 动效 5：tensor-map→elected pulse 0 + 记账链 expect_tx→搬运→mbarrier→消费者 0.4/0.8/1.2 cascade，截图四点各在边中。
- 可读 5：左 cp.async 1024 条软件账 vs 右 TMA descriptor 硬件账对立分区 + 底部字节记账链，「1024 对 1」标题即结论。
- 层级 5：elected/mbarrier attn 聚焦硬件侧；红「地址生成开销线性涨」与红「先 wait 再 barrier」两坑注分层收尾。
- 细节 4：末行 CUTLASS PipelineTmaAsync 注记止于 ≈702/720，右边距 ~18px 偏紧（fig-7 底行）。
- 咬合 5：256×4=1024 条/16 B 粒度/commit_group·wait_group N/128 B tensor-map/1D–5D/elected/expect_tx 16384=8192+8192/complete_tx 扣减/try_wait 双条件/tx-count 字节计数 wp 逐字/sleep 对 spin/2×128 B 占 1.2×10⁻⁷/32/64/96/128 B swizzle 均对原文 line 23/50–65/151/180。

## 3. redraw findings（任一维 <4）

1. **gpu-p04 fig-5 doubleBuffer（4.3 b1）细节完成度 3**：脉冲跑道 y222/y254 脱离条中心线 206/238，计算行光点等距浮于两行之间、装载行光点浮于条下空白；截图实证见 `/tmp/figcrop/f5-rows.png`（源 `site/qa/figs-gpu-p04/fig-5.png`）。修复：两条 pathPulse 的 path y 改 206/238（或贴条底 ≤2px），重跑 fig-check 复审。连带可读 4 亦由此引起，修复后预期回 5。
   - 同源轻症（不构成 redraw，建议顺带统一）：gpu-p04 fig-6 装载行 pathPulse y150 → 建议 134（条中心）。

## 4. 内容 fidelity 汇总

- 3.1：16777216/65536×256/132 SM/1056/8448/270336/62 波/尾 64/6.1%/63 迭代/25637/257→9 warps/D1–D5/0.1875/0.8125/4.33×/1.275/0.784（聚合口径注保留）/2^d —— 全对。
- 3.2：256 KB RF/统一 256 KB/29.0/478.8/16.5×/16384 fp32/512 KB/spill 三类/ld.global.nc/membar≡fence.sc 三级/均 seq_cst/pre-SM90 cluster 放宽/1056×/263.0≈143.7 ns/40.7/6.46×/morally strong 三条件 —— 全对。
- 3.3：默认 8/1–32/CC 8.0+ COPY_CONNECTIONS/3205.79/2235.73/262.144/7.508/131.072/2097.152/1048.576/60.062/970.06/30.3%/1.434/1.529/65%/3145.73/16×/400/3.4/9.6/3.8/2.9/6.7/0.9/0.5/1000/0.02/330×5=1.65/1.645/33%/500000 元素 512 threads —— 全对。
- 3.4：128 B+1 KB 粒度/warp 取整/min 四约束/8 blocks O=1.0/768→75%/32→50%/100 KB→2/CC 7.0 手算 5 对官方 4/K2 24576/33792/49152/67584/16/64/2/32/O=0.25/NCU replaces 逐字 —— 全对。
- 4.1：**distinct=32 对 ceil=63 高估、5 对 4 低估方向正确**；1988/1024 B/η=1/8/0.8/419 地板/3352/1676/838/209.5/40.04/320.33/BP stride-2 50% 逐字 —— 全对。
- 4.2：k=max u_b/broadcast 例外/字0+字32 k=2 64 B/cycle/7.3<25.4/gcd 定律/p=2 反例/4224/+3.125%/132 B/Swizzle<3,4,3>/ldmatrix 8→1/1056 对 64/577.0/34.97/16.5×/12.8→140.2→199.4/10.95×/1.42× —— 全对。
- 4.3：16 KB/8 A+8 B/64 fp32/128 步/I=T_tile/bytes/0.5→64/274.88/82.00/2.147/640.7/128×/295/590/33.55/1.56%/8.590/4.295/1.074/139.9/645.2/261.6/906.8/1046.7/13.4%/766.9/1.154/1.180/13→6 —— 全对。
- 4.4：645.2/453.4/1.622/1290.4/S_min=3/48 KB→4 blocks/96 KB/1.87/279.8/1024 条/16384=8192×2/128 B/1.2×10⁻⁷/elected/try_wait/sleep 对 spin/No-Op —— 全对；唯一图侧自估 = fig-6「填充段约 1/128」（已对冲，观察项）。

## 5. 总结

- **18 图：17 pass / 1 redraw**。
- redraw：gpu-p04 fig-5 doubleBuffer（细节 3；脉冲跑道脱中心线、行间等距浮点）。
- 最低分维分布：细节完成度 4 共 9 图（紧边距 / 瞬态遮标签 / 参考线压条 / 跑道轻症），可读性 4 共 2 图（p03 fig-3 单交叉、p04 fig-5 浮点归属），咬合 4 共 1 图（p04 fig-6 自估分数）。
- 全局纪律：census 14+4 ✓、afterHeading 四锚唯一 ✓、页面级 PULSE_GRAD_ID ✓、reduced-motion 静态终态 ✓、delay/--i 上限 ✓。
