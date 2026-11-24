# gpu W1 图品质量评审（QUALITY 五维打分档案）

评审对象：`site/scripts/lib/figures/gpu-p01.mjs`（10 张）+ `gpu-p02.mjs`（8 张），共 18 张。
量规：`site/scripts/lib/figures/QUALITY.md` 五维 1–5 分制，全维 ≥4 过线；创作规范 `STYLE.md`。
评审方法：全部实拍证据 = `site/qa/figs-gpu-p01/fig-0..9.png` + `site/qa/figs-gpu-p02/fig-0..7.png`
（fig-check.mjs 渲染：真实 tokens.css + figure.css + 页面级 PULSE_GRAD_DEFS、`.fig.in` 常亮、dsf=2、t=3.5s 稳态帧；
PNG 时间戳 04:31 晚于模块 04:25/04:30，为最新渲染）。逐张 Read 截图 + 源码坐标核对 + 章节原文对账。
校准先例：`docs/reviews/figure-quality-baseline.md` / `figure-quality-r2.md`（静态账本动效=4 不扣；
方向缺位靠 pulse 补=读 4；viewBox 出框/截断=重画档，分区虚线 gutter 溢出=记注档）。
站点无深色模式（tokens/figure/base.css 无 prefers-color-scheme/data-theme），对比度按纸面+角色浅染卡评判。

维度序：动 = 动效自然度 / 读 = 场景可读性 / 层 = 视觉层级 / 节 = 细节完成度 / 合 = 与正文咬合。

## 一、逐图评分（18 张）

### gpu-p01（01-gpu-microarchitecture）

| # | 图（锚点） | 动 | 读 | 层 | 节 | 合 | 总评 |
|---|---|---|---|---|---|---|---|
| p01-0 | 1.1 #block0 GPC/TPC/SM 三级分组 | 5 | 4 | 5 | 5 | 5 | **4** |
| p01-1 | 1.1 #block1 CoWoS 与 B200 双 die | 5 | 5 | 5 | 5 | 5 | **5** |
| p01-2 | 1.2 #block0 SM 内部框图 | 5 | 5 | 5 | 4 | 5 | **4** |
| p01-3 | 1.2 @##5 occupancy 四约束取 min | 5 | 5 | 5 | 5 | 5 | **5** |
| p01-4 | 1.3 #block1 mma.sync 数据流 | 5 | 5 | 5 | 5 | 5 | **5** |
| p01-5 | 1.3 #block2 tcgen05 与 TMEM | 5 | 5 | 5 | 5 | 5 | **5** |
| p01-6 | 1.3 @##4 操作数复用账 | 4 | 5 | 5 | 5 | 5 | **4** |
| p01-7 | 1.4 #block0 流水线时序 | 5 | 5 | 5 | 4 | 5 | **4** |
| p01-8 | 1.4 #block1 warp 调度与 scoreboard | 5 | 5 | 5 | 4 | 5 | **4** |
| p01-9 | 1.4 @##4 Little's law | 5 | 5 | 5 | 5 | 5 | **5** |

- **p01-0 chipHierarchy**（fig-0.png）
  - 动 5：链边 pulse delay 0/0.3/0.6 + L2 边 0.9，kit spline ease-in-out（keySplines 0.42 0 0.58 1）；3.5s 帧四光点均在飞、相位错开。
  - 读 4：die→GPC→TPC→SM 主链左→右带显式箭头一眼可读；**L2 分区 0↔1 互连无双向箭头**（原 ASCII ◄────►；forward 边 marker 被 CSS 关、return 边 arrow:false 且与 forward 同线段重叠），仅靠单向 pulse 示意，方向语义弱于原文。
  - 层 5：主链标题 12.5 + SM 卡 attn 朱染最重；die 虚线区次级；harvest 条 seal 段点题；留白充足。
  - 节 5：GPC 2×4 网格等距 154/60；harvest 条标签中心含 bars 3px gap 精确（289.6/551.8）；crossbar 注记压线上方 4px 干净；无出框。
  - 合 5：答 §2「差的 12 个去哪了」——144−132=12 SM=6 TPC、91.7%、1079 对 989.5、PCIe 57/114、MC×6×2 侧、NVLink×18 逐字。
- **p01-1 cowosPackage**（fig-1.png）
  - 动 5：die/栈垂落 pulse 0..0.6 + 总线横脉冲 0.6 + NV-HBI 1.2，spline 配速；3.5s 帧垂落点与总线点均在飞。
  - 读 5：剖面自上而下 die/栈→中介层→基板三层线清楚；B200 双 die 双向显式箭头（die0.right 左指 / die1.left 右指）+「← 5 栈」侧注正确。
  - 层 5：NV-HBI 10 TB/s 朱砂 mono 650 最重；双 die 区虚线次级；坑注朱砂收尾。
  - 节 5：5 栈等距 90 pitch；中介层/基板双横线夹注居中；三行底注 18px 行距；无溢框。
  - 合 5：5120-bit 毫米级走线、208B、raw 192/usable 180、10>7.7 TB/s、三互连三层级——§5 逐字。
- **p01-2 smInternals**（fig-2.png）
  - 动 5：4 条 scheduler→dispatch 垂边 pulse delay k×0.3，3.5s 帧四点分处四相位（呼吸错拍）；箭头三角在 dispatch.top。
  - 读 5：2×2 processing block + 右列单元清单 + SM 级共享行，不读正文即 Fig 7 结构；单发射语义由 ×1 卡直接给出。
  - 层 5：scheduler attn 染最重、block 卡 op 浅染次级、共享行三色角色、底注 mono 三级字阶。
  - 节 4：**「SM 级共享」cap（52,316）字顶与 processing block 2 卡底边（y=310）相贴约 2px**（12px 间隙塞 9px 字）；其余对齐精确。
  - 合 5：RF 16384×32-bit=64 KB、32 FP32/16 INT32/16 FP64/1 TC/8 LD-ST、228 KB 档、33 MB、128→66.9、双发射 133.8 反例——§2/§3 逐字。
- **p01-3 occupancyMin**（fig-3.png）
  - 动 5：四约束汇流 pulse 0/0.25/0.5/0.75 + min→W 1.1 + W→O 1.5；3.5s 帧汇流四点相位各异。
  - 读 5：左四卡→min→×8→/64 的收敛-再推导流向无交叉；register 项 strong 边 + 朱注「最紧→绑定」点题。
  - 层 5：min=4 卡 attn + title 14 最重；register 卡 attn 染呼应；右列 proj 次级。
  - 节 5：左列 62px 等距；四枚箭头三角同点叠绘呈单箭头干净；∞/⌊⌋ mono 副行不溢卡。
  - 合 5：C1 走读 min(32,8,4,∞)=4、O=0.5、灵敏度 64→32 翻倍、C5 261120>65536（3.98×）、命题 3 粒度——§5 逐字。
- **p01-4 mmaSyncPath**（fig-4.png）
  - 动 5：链 pulse 0/0.35/0.7 + D 边 1.1 + k 回环 1.5 + epilogue 2.1（全 <3.5）；回环光点 3.5s 在横段。
  - 读 5：HBM→shared→fragment→TC 横链 + 回环「D 当 C 读回」L 形上行箭头指回 fragment.bottom，方向闭环一眼可读。
  - 层 5：TC attn 最重；A/B/C 矩阵 ink/warm 点阵次级；底注 mono 三级。
  - 节 5：矩阵 16×16 / 16×8 / 16×8 形状与原文一致（kit n=行 m=列）；回环注记居矩阵右侧空档不压格；cp.async/TMA、ldmatrix 边注在边上方。
  - 合 5：A 4 regs/B 2/C 4、4096 FLOPs、t_issue 4 cycles、4 block 闭环 c_BF16=4096、fragment 32× 误认——§2/§3 逐字。
- **p01-5 tcgen05Path**（fig-5.png）
  - 动 5：四边 pulse 0/0.5/1.0/1.5 spline；3.5s 帧 S 形流全段有点。
  - 读 5：shared→TC→TMEM→register→写回 HBM 的 S 形流向箭头齐全；「描述符」「D 驻 TMEM」边注在正确侧。
  - 层 5：TMEM 卡最宽 + cache 染 = 新闻点；TC attn 次重；写回 plain 最轻。
  - 节 5：三列 x=36/250/484 对齐；512 列×128 行×32-bit 副行不溢；底注四行 18px 行距。
  - 合 5：262144 B=256 KB/CTA=RF 容量、cta_group::2 有效 M 翻倍、wgmma N=256 吃 128 regs、B shared-only——§6 逐字。
- **p01-6 operandReuse**（fig-6.png）
  - 动 4：纯静态账本图（构成条 + vs 卡 + 每周期条），无动效原语——声明性内容静止合宜（基线先例 #9/#29 同口径）；reduced-motion 终态=自身。
  - 读 5：读 A/B/C + 写 D 构成条→1792 B→MMA vs FMA→448<512 自上而下账链清楚。
  - 层 5：写 D seal 段 + 两行朱砂结论最重；MMA 卡 attn；vs 弱字衔接。
  - 节 5：四段标签中心含 3px gap 精确（128.8/247/365.2/521.8）；每周期条 448:512 比例正确（0.62 同尺）；左标签 anchor:end 对齐 180。
  - 合 5：0.4375 对 8、18.3、448<512、1792<2048、96=128 的 75%——§4 命题 2 逐字。
- **p01-7 pipelineTiming**（fig-7.png）
  - 动 5：三 lane 横脉冲 delay 0.2/0.6/1.0 spline，3.5s 帧三点分处不同 stage 列（错拍呼吸）；依赖/独立链 span 为静态 pill 合宜。
  - 读 5：周期列 1–7 + scheduler 行 + 三 lane 逐拍右移 + 写回@4/5/6 朱注；两条 timeline 的 1/5 对 1/2/3/4 span 对照不读正文即懂。
  - 层 5：E4 attn 染 + 写回朱注 = 延迟点题；scheduler 行 op 浅染；底注 mono。
  - 节 4：**y=236 与 y=248 两行 cap 行距仅 12px**（全站 house 18px），视觉偏挤但不重叠；其余列对齐精确（X(c)=150+(c−1)×66）。
  - 合 5：~4 cycles 延迟/每周期 1 条、依赖链塌 1/4、Little 热身 L=4、shared 29→L2 263→HBM 479——§2/§4 逐字（ASCII block0 同构）。
- **p01-8 warpScheduler**（fig-8.png）
  - 动 5：warp→zone→eligible→dispatch→ex 链 pulse 0/0.5/1.0/1.4 spline；3.5s 帧全链有点。
  - 读 5：resident warps 区→三条件区→右列发射链 + 下方 stall 四分类虚线分支，流向与分支不交叉。
  - 层 5：执行管线 attn 最重、eligible norm 绿、stall 四卡角色染（long_scoreboard attn / not_selected norm）编码语义。
  - 节 4：**类名 cap y=318 与 prose cap y=330 行距 12px** 偏挤（同 p01-7）；zone 标签/⋯16 slots 注在区内不压卡。
  - 合 5：eligible 三条件、16 slots/scheduler（64/4 反推）、stall 四分类映射 Nsight §2.4.7、not_selected 健康签名、wait_group——§3/§7 逐字。
- **p01-9 littlesLaw**（fig-9.png）
  - 动 5：λ×t→L pulse 0/0.4 + 折行 ÷4→52→O pulse 0.9/1.4/1.8 spline；3.5s 帧 S 流全段有点。
  - 读 5：上行乘积链（× glyph）+ 下行除法链（左指箭头）方向无歧义；ILP 1/2/4 条长比例 0.81/0.41/0.20 正确。
  - 层 5：L_SM 与 O 两 attn 卡最重；ILP1 seal 条点题；坑注朱砂收尾。
  - 节 5：三列 x=36/330/554 上下行对齐；条值标签随条尾 +8px；底注 18px 行距无溢框。
  - 合 5：0.434=25.4÷32÷1.83、478.8=261.6 ns、208≈6.6 KB、52 warp、O 0.81、479 warp 高估 9.2×、877 KB、116 sectors——§4/§5 逐字。

### gpu-p02（02-memory-hierarchy）

| # | 图（锚点） | 动 | 读 | 层 | 节 | 合 | 总评 |
|---|---|---|---|---|---|---|---|
| p02-0 | 2.1 #block0 HBM 栈几何→带宽 | 5 | 5 | 5 | 4 | 5 | **4** |
| p02-1 | 2.2 #block0 内存层级金字塔 | 5 | 5 | 5 | 5 | 5 | **5** |
| p02-2 | 2.2 @##2 AMAT 凸组合 | 4 | 5 | 5 | 5 | 5 | **4** |
| p02-3 | 2.2 #block1 line/sector 粒度 | 4 | 5 | 5 | 5 | 5 | **4** |
| p02-4 | 2.3 #block0 register operand 供给 | 5 | 5 | 5 | 4 | 5 | **4** |
| p02-5 | 2.3 @##6 spill 双代价账 | 5 | 5 | 5 | 5 | 5 | **5** |
| p02-6 | 2.4 #block0 三区判定流程 | 5 | 5 | 5 | 5 | 5 | **5** |
| p02-7 | 2.4 #block1 实测 roofline | 5 | 4 | 5 | 4 | 5 | **4** |

- **p02-0 hbmStack**（fig-0.png）
  - 动 5：TSV 垂脉冲 0.2 + 四因子链 0/0.3/0.6/0.9 spline；3.5s 帧 TSV 点近 base die、链点四相位。
  - 读 5：侧视 8-high+base die / 顶视 16 ch→2 MC / 底部四因子乘积链三段各自一眼可读；×/÷8 glyph 在边间隙正确侧。
  - 层 5：3352 结果卡 attn + title 15 最重；因子卡 proj 次级；几何区虚线再次。
  - 节 4：**侧视 zone 标签（mono 10.5 + letter-spacing 0.04em）右端越自身虚线界约 14px 入空 gutter**（无碰撞）；**ch7/ch15 卡右缘与顶视 zone 右界重合（0 padding）**；「⋮」塞在 4px die 间隙略挤。三者均 spacing 记注档（基线 #38/#42 同口径），非 viewBox 出框。
  - 合 5：5×16×5.238×64÷8=3352、2619 DDR×2、5120-bit 逐字、16× 误代 53.6 TB/s、Samsung 819.2、H200 6.25→4.8、MC×10 每栈 2——§2/§3 逐字。
- **p02-1 cacheHierarchy**（fig-1.png）
  - 动 5：L1 miss / L2 miss 两曲脉冲 0.4/1.2 spline；3.5s 帧一点近 L1 左缘、一点近 HBM 左上角（语义=下行在飞）。
  - 读 5：五层金字塔宽度递增 + 左侧 miss 下行箭头 + 右侧私有/共享括注，不读正文即容量-延迟-作用域三账。
  - 层 5：HBM 最宽底基层、register 顶窄层；cache 染统一中间三层；底注 mono 比值点题。
  - 节 5：五层中心对齐 x=300、60px 等距；miss 箭头落 L2/HBM 左上角贴边干净；括注线 x=572 与 HBM 右缘 12px；底注 18px 行距。
  - 合 5：256/228/256/50 MB/80 GB、29.0/40.7/263.0/478.8、4055:253:25.4≈160:10:1、535 TB/s、×132/÷132 归一化——§2/§4 逐字。
- **p02-2 amat**（fig-2.png）
  - 动 4：纯静态凸组合账本（三贡献条 + 灵敏度三条），声明性静止合宜；reduced 终态=自身。
  - 读 5：weight 列→公式注→贡献条→值列四栏自上而下；t_avg 求和行收口；灵敏度单调下降条对照清楚。
  - 层 5：全 miss seal 条最重（主矛盾）；t_avg mono 650 收口行；灵敏度末条 seal 点题。
  - 节 5：条起点统一 x=180、贡献×2.2 与灵敏度×1.6 各自同尺；weight/值标签基线对齐条中心；无溢框。
  - 合 5：0.5×40.7+0.25×263+0.25×478.8=205.8≈112.5 ns、106.7/73.7、降 64%、10% 到 HBM——§2 式 (1) 逐字。
- **p02-3 lineSector**（fig-3.png）
  - 动 4：纯静态粒度对照账（sector 卡 + 两模式条 + 有效带宽条），静止合宜；reduced 终态=自身。
  - 读 5：line=4 sector 卡→coalesced 满条 32 线程刻度 vs 散乱 8 pill+⋯（每 pill 内 seal 4B）→419=3352/8 条对照，浪费语义编码直接。
  - 层 5：散乱 seal 标签 + seal 有效条最重；coalesced ink 条中性重；坑注朱砂。
  - 节 5：sector 卡 150 等距、线程刻度 5.4 等距、pill 22 等距；有效带宽条同尺 0.12（402 对 50.3）；解释注起于 seal 条尾 +10px 不压条。
  - 合 5：128=4×32、η=1 对 12.5%、1024 B 流量、419 GB/s、4× 高估坑、sector 服务率解释 1.4 式 (2)——§3 逐字。
- **p02-4 regOperand**（fig-4.png）
  - 动 5：三冲突源汇流 pulse 0.3/0.6/0.9 + collector→dispatch 1.6 + dispatch→ex 2.0 spline；3.5s 帧汇流三点在飞。
  - 读 5：bank 库存区→无冲突/冲突两例对照（norm 绿 vs attn 朱）→collector→dispatch→ex；冲突例才有汇流边，对照语义清楚。
  - 层 5：冲突行 attn + 朱注「3 路冲突 3 拍」最重；collector op 宽卡次级；bank 库存 data 染最轻。
  - 节 4：**三汇流贝塞尔 + 箭头 + 两脉冲点挤在 bank 行底(236)→collector 顶(252) 的 16px 带内**，3.5s 帧两点近重合于汇流结（fan-in 固有拥挤，瞬态可分离）；其余对齐精确。
  - 合 5：教学模型假设 4 bank 披露口径、r1r2r3 无冲突一拍 / r4r8r12 同 bank 0 三拍、2048 B/cycle 上界、wgmma B 减压——§2 逐字。
- **p02-5 spillCost**（fig-5.png）
  - 动 5：st.local 链 pulse 0/0.4 + ld.local 回环虚线脉冲 0.9 spline；3.5s 帧回环点在中段、上行箭头指回 local.bottom。
  - 读 5：register→local→HBM 出程 + 虚线读回程闭环；带宽账 seal 条 vs 延迟账 MLP4/1 双条，两口径独立一眼可读。
  - 层 5：1.29 ms seal 条最重（带宽主代价）；MLP 条 warm 次级；坑注朱砂收尾。
  - 节 5：条同尺 220/ms（283.8/57.6/231 比例正确）；左标签 anchor:end 240、条起 250、值随尾 +8；回环虚线在节点下方 18px 不压卡。
  - 合 5：32.768 MB、1.29 ms、0.262/1.05 ms、MLP 1→4 降 4×、160× 落差、512 KB 上限、L1 40.7 常命中——§6 命题 1 逐字。
- **p02-6 boundFlow**（fig-6.png）
  - 动 5：inp→d1→d2 垂链 0/0.5 + 两「是」支 0.3/0.9 + 「否」底回环 1.3 spline；3.5s 帧全分支有点。
  - 读 5：决策树自上而下、是/否标签贴分支正确侧；底回环 L 形上行箭头入 latency-bound 底边，无交叉线。
  - 层 5：latency-bound attn 朱染 = 头条（非第三条 roof）；compute norm / memory cache 角色染；决策卡 op。
  - 节 5：左列 x=36 / 右列 x=360 对齐；三区卡等宽 324；底注式 (7) mono + Nsight 签名 18px 行距。
  - 合 5：R≈295、L_req≈208 sectors、式 (7) min(1,C/L_req)、u_F/u_B 指针（5.2/6.1、4.1/4.3/5.4）、long_scoreboard 签名——§5 逐字（ASCII block0 同构）。
- **p02-7 rooflineMeasured**（fig-7.png）
  - 动 5：C→B 单脉冲 1.0 spline（提并发拉回语义）；3.5s 帧点在 C/B 之间；坐标/roof 静态合宜。
  - 读 4：A/B/C 三点 + loss 虚线 + ridge 虚线清楚；**「memory-bound 斜线区」标签置于斜线上方（不可达侧）**——区域名应在斜线下方，现与「斜率=BW」线注同侧偏上，侧别欠准（基线 #12 标签压曲线同档）。
  - 层 5：roof strong 折线最重；seal 实心点 A/B vs 空心 C 编码 bound/latency 差异；loss/利用率 mono 注记三级。
  - 节 4：**C 的「u = 50%」注记基线 328 距 x 轴 330 仅 2px** 贴线；8.375/700/989.5 y tick 与点序自洽；其余对齐干净。
  - 合 5：A(400,700) u_F 70.7% loss 29.3%、B(2.5,7.125) u_B 85.1%、C(2.5,4.19) u 50%、R≈295、989.5/3.35/8.375、层级 roofline tiling 移 ceiling——§7 逐字；「示意非按数值比例」免责声明在标题 ✓。

## 二、重画发现（任一维 <4）

**无。18/18 全维 ≥4，0 张打回。**

残留观察（4 分证据，非阻塞，供后续波次顺手修）：
1. p01-0 读 4：L2 分区互连补双向显式箭头（原 ASCII ◄────►；现 return 边与 forward 重叠且无箭头）。
2. p01-2 节 4：「SM 级共享」cap 下移 4–6px 脱离 block2 底边。
3. p01-7 / p01-8 节 4：y236/248、y318/330 两处 cap 行距 12px → 18px house 行距。
4. p02-0 节 4：侧视 zone 标签缩短或 zone 加宽消 14px 越界；顶视 ch 列左移 4px 留 padding；「⋮」移除（8 die 全绘无需省略号）或加间隙。
5. p02-4 节 4：bank 行底→collector 顶间隙 16px → 28px，散开汇流结。
6. p02-7 读 4 + 节 4：「memory-bound 斜线区」标签移斜线下方；C 的 u=50% 注记上移 6px 离轴。
7. 静态账本三张（p01-6、p02-2、p02-3）动 4 为合宜静止（基线先例），不需加动效。

## 三、census 覆盖核验

裸围栏块全量（build-content.mjs 同口径：无语言围栏按序计 block，带语言围栏跳过）= **14 块**；
注册 replacesBlock = **13**，唯一未画 = 1.3 #block0（@29，PTX 语法伪代码 → 管线纸面算法面板，STYLE 铁律「伪代码不画」✓）。

| 章 | block@行 | 图 | 覆盖 |
|---|---|---|---|
| 1.1 | 0@31 / 1@135 | p01-0 / p01-1 | ✓ §2 die 框图 / §5 CoWoS 剖面 |
| 1.2 | 0@37 | p01-2 | ✓ §2 SM 框图 |
| 1.3 | 0@29 / 1@48 / 2@161 | —（伪代码）/ p01-4 / p01-5 | ✓ |
| 1.4 | 0@21 / 1@54 | p01-7 / p01-8 | ✓ §2 时序 / §3 调度 |
| 2.1 | 0@27 | p02-0 | ✓ §2 栈侧视+顶视 |
| 2.2 | 0@19 / 1@79 | p02-1 / p02-3 | ✓ §2 层级路径 / §3 line/sector |
| 2.3 | 0@23 | p02-4 | ✓ §2 operand 供给教学模型 |
| 2.4 | 0@166 / 1@248 | p02-6 / p02-7 | ✓ §5 三区流程 / §7 实测 roofline |

afterHeading = **5**，锚点各命中唯一一行（grep '^## ' 核验）：
1.2 `/^## 5\. /`@140（occupancy C1 走读签名）；1.3 `/^## 4\. /`@101（复用账签名）；1.4 `/^## 4\. /`@86（Little's law 签名）；
2.2 `/^## 2\. /`@15（AMAT 签名）；2.3 `/^## 6\. /`@103（spill 双代价签名）。
13 + 5 = 18 = 模块注册数 ✓；签名场景（C1/复用账/Little/AMAT/spill）与章脊问题一一对应 ✓。

## 四、内容 fidelity 抽查（数字/形状对原文）

全 18 图注记数字逐条对章节原文核验通过（见逐图「合」证据）。重点复核：
- 1.1：144/132/114、66/57 TPC、91.7%、9.1%（1079 对 989.5）、50 MB=2×25、814 mm²/4N/208B/4NP、NV-HBI 10 TB/s、raw 192/usable 180 ✓。
- 1.2：65536=4×16384、256 KB/SM、33 MB、228/227/256 三口径、128 results/cycle、66.9/133.8、C1 min(32,8,4,∞)=4、C5 261120（3.98×）✓。
- 1.3：fragment A4/B2/C4/D4 regs、512/256/512/512 B、1792/4096=0.4375、8 B/FLOP、18.3、448<512、1792<2048、96=75%、TMEM 512×128×4=262144 B、cta_group::2、N=256→128 regs ✓；矩阵形状 A[16,16]/B[16,8]/C/D[16,8] 与 kit n=行 m=列一致 ✓。
- 1.4：~4 cycles、写回@4/5/6、依赖链 1/4、29/263/479、0.434/478.8/208/6.6 KB/52/0.81/0.41/0.20、479 warp 9.2×、877 KB、116 sectors、16 slots/scheduler ✓。
- 2.1：5×16×5.238×64÷8=3352.32、2619 DDR×2、5120-bit、16×/53.6 TB/s、819.2、6.25→4.8、8-high+base die、16 ch→MC0/1、32 pseudo-channel ✓。
- 2.2：29.0/40.7/263.0/478.8、4055:253:25.4≈160:10:1、535 TB/s、20.35+65.75+119.7=205.8≈112.5 ns、106.7/73.7/64%/10%、128=4×32、η 1/12.5%、419=3352/8、4× 坑 ✓。
- 2.3：假设 4 bank 教学口径披露、r1r2r3/r4r8r12、3 拍、2048 上界、32.768 MB、1.29/0.262/1.05 ms、160×、512 KB、40.7 ✓。
- 2.4：295/208/式 (7)/Nsight 签名、A/B/C 三点坐标与 u/loss（70.7/29.3、85.1/14.9、50/50）、8.375=2.5×3.35、989.5/3.35、层级 roofline ✓；roofline 标「示意非按数值比例」与原文同口径 ✓。
未发现数字漂移、形状错配或口径混代（per-SM/芯片、dense/sparse、raw/usable 均按章脊口径）。

## 五、机制专项核验

- **脉冲渐变页级 id**：两模块 pathPulse 与 kit edge 均引用 `PULSE_GRAD_ID='fig-pulse-grad'`；无 per-scene defs；
  `Chapter.astro:27` 注入 PULSE_GRAD_DEFS、fig-check 预览页同注入 ✓（grep 核验）。
- **easing**：kit pulse `calcMode=spline keySplines="0.42 0 0.58 1"` 单程 ease-in-out、0→1 回绕保方向语义 ✓；无匀速硬切/机械脉冲。
- **reduced-motion 静态终态**：figure.css `@media (prefers-reduced-motion: reduce){ .dia-pulse{display:none} }`（r2 修复后口径）；
  gpu 两模块全部光点均包 `.dia-pulse`（含手写 pathPulse），无裸 SMIL、无 sig-* 无限循环类 → 降级态=完整静态图 ✓。
- **截图时刻语义**：全图 pulse delay ≤2.1s < 3.5s，3.5s 帧所有光点已入飞且相位错开（逐图截图核验）✓。
- **无入场动画**：两模块未用 sig-step/sig-bar 等入场原语，静态元素不加类 ✓（STYLE 铁律）。
- **API 签名**：未改 figure-kit.mjs（本轮只读评审）✓。

## 六、总结

**18 张过线（全维 ≥4）/ 0 张重画。** 最低分维度分布：读 4 ×2（p01-0 L2 双向箭头缺位、p02-7 区域标签侧别），
节 4 ×6（p01-2/p01-7/p01-8/p02-0/p02-4/p02-7 间距贴边类记注），动 4 ×3（静态账本合宜静止）。
census 13 replacesBlock + 5 afterHeading 全覆盖（14 裸块中唯一伪代码块按铁律跳过）；内容 fidelity 全通过；
页级 PULSE_GRAD_ID 与 reduced-motion 静态终态合规。残留观察 7 条列第二节，供 controller 酌情派顺手修，不阻塞 W1 图 QA gate。
