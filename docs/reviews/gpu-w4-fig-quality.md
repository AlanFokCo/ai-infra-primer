# gpu W4 场景图 QUALITY 评审档案（gpu-p07 / gpu-p08，20 图）

- 评审日期：2026-11-25；评审角色：fresh design-eye evaluator（只读，未改模块）
- 规约：`site/scripts/lib/figures/QUALITY.md`（五维各 ≥4 过线，截图证据）+ `STYLE.md`
- 证据：`cd site && node scripts/fig-check.mjs gpu-p07`（7 张）/ `gpu-p08`（13 张），3.5s 截图时刻逐张 Read `site/qa/figs-gpu-p07/fig-*.png`、`site/qa/figs-gpu-p08/fig-*.png`；局部放大 sips crop（/tmp/figqa-crops/p07f1-zonelabel.png、p07f2-gapcap.png、p08f6-mirrortop.png）；reduced 终态两帧像素差自跑（/tmp/figqa-reduced.mjs，口径同 fig-reduced-diff.mjs）+ reduced 帧目视（/tmp/figqa-crops/reduced-p07-fig0.png、reduced-p08-fig6.png）
- 判定：**19 pass / 1 redraw**（redraw = gpu-p08 fig-6 doubleBinaryTree，细节完成度=3）
- 打分顺序：动效自然度 / 场景可读性 / 视觉层级 / 细节完成度 / 与正文咬合

## 0. census 与全局纪律核验

- 注册表 census：gpu-p07 = 5 replacesBlock + 2 afterHeading；gpu-p08 = 10 replacesBlock + 3 afterHeading → **15 replacesBlock + 5 afterHeading = 20** ✓（脚本逐条核对注册表与章节围栏）
- 逐章无语言围栏块核对（开闭配对、有语言围栏不计）：7.1 b0(L69)/b1(L129) 全替换；7.2 b0(L19) 替换（bash 围栏不计）；7.3 b0(L54)/b1(L74) 替换、**b2(L147) cuFile 调用序列为伪代码跳过**（1 pseudocode skip ✓，管线纸面算法面板接管）；8.1 b0(L89)；8.2 b0(L37)/b1(L104)/b2(L119)/b3(L138)；8.3 b0(L43)；8.4 b0(L21)/b1(L92)/b2(L112)/b3(L195) 全替换（8.4 两个 ```c 围栏不计）→ 除伪代码外无漏块 ✓
- afterHeading 锚点唯一命中：7.1 `^## 2\. `→L15、`^## 4\. `→L85；8.1 `^## 3\. `→L48；8.3 `^## 4\. `→L55；8.4 `^## 4\. `→L62 ✓（各 1 hit）
- 脉冲渐变：两模块均 import `PULSE_GRAD_ID`（='fig-pulse-grad'），kit edge pulse 与模块 pathPulse 全引用页面级固定 id；grep 两模块无 `<defs>`/radialGradient/per-scene id；`PULSE_GRAD_DEFS` 由 `src/layouts/Chapter.astro` L27 注入、fig-check.mjs 同注入 ✓ 无错配（截图光点均为朱砂渐变实心点）
- reduced-motion 静态终态：`figure.css` L201-203 reduce 态 `.dia-pulse{display:none}`；Chapter.astro L119-148 reduce 分支移除 .flow 并加 .in。两模块只用 dia-pulse（无 sig-*、无入场动画）。自跑两帧差（t3000 vs t4500，reduce 上下文）：**20/20 diffpx=0**（仅 p08 fig-7 =2px 抗锯齿噪声，远低于 200px 阈值）；computed `.dia-pulse` display=none ✓；reduced 帧目视为完整静态图（节点/连线/条/注记全在）✓
- 截图时刻语义：p07 pulse delay ≤1.2s、p08 ≤2.7s（proxyGdr 状态机 2.4/2.7），dur ≤1.8s → 3.5s 时全部入循环稳态，截图光点分布与相位推算一致 ✓
- 动效口径：kit edge pulse 为 calcMode=spline ease-in-out（端点慢中段快）；pathPulse 为 paced 匀速——与 gpu-p05/p06、inference-part09 已审先例同helper同口径，W3 档案按 4 分接受，本波沿同一口径（不另判罚）

## 1. gpu-p07（7 图）

### fig-0 nvlinkGenerations（7.1 afterHeading §2）— 4/5/5/5/5 → pass
- 动效 4：乘法链两脉冲 spline + 0/0.4s 阶梯 delay，3.5s 光点一在链中段一在 ×2 卡左端口（截图），节奏有呼吸；五代表静态无动效需求。
- 可读性 5：左→右三卡乘法链（25 单向 → 450 单向 → 900 双向）不读正文即懂方向口径；表头 per-link / per-GPU 单向 / 双向头条三列标签各归其位。
- 层级 5：链最重（角色染三色）→ 五代表条居中（H100 seal / B200 warm 点睛）→ 底部三行注记字级递减，留白充足。
- 细节 5：条宽按 uni×0.28 线性、值标签随条尾右移不溢框；右列双向头条 end-anchor 对齐 700；全文字在 viewBox 720×404 内。
- 咬合 5：2×100÷8=25、×18=450、×2=900（式 1/2）；五代 80/160、150/300、300/600、450/900、900/1.8TB 与 §2 表逐格一致；1.5×/2×/5× 对账、V100 25.78→25 约 3%、坑注 ÷2 全对。

### fig-1 nvswitchCrossbar（7.1 b0）— 4/5/5/4/5 → pass
- 动效 4：单对路径三段脉冲（入 1.4s/横 1.6s/出 1.4s，delay 0/0.5/1.0）接力流动，截图三点分居入曲线/横干线/出曲线顶端，方向语义完整。
- 可读性 5：8 GPU 汇接 4 switch + 高亮 GPU0→GPU7 单对路径，一眼懂「全汇交换、单对满带宽」；128 vs 900 对照条在下独立成账。
- 层级 5：高亮 strong 路径最重、普通汇接短线灰弱、zone 虚线分区次之；底部端口账/对照条/反例注三层字级。
- 细节 4：GPU0 入曲线在 zone 标签行穿过「第三代」与「NVSwitch」之间的词隙（crop p07f1-zonelabel 实证：线宽隙内、箭头尖 graze N 左缘约 2px），不压字但穿标签行属紧 threading；其余对齐/圆角/对比干净。
- 咬合 5：144 links 占 144 端口（256 总、平均 36/chip）、4×3.2=12.8 TB/s、900/7≈128 对 900 = 7×、总注入相同与 6300 反例全对 §3/7.2 §3。

### fig-2 nvlinkDomain（7.1 afterHeading §4）— 4/5/5/4/5 → pass
- 动效 4：恒等式链两脉冲 0/0.4s + NVL72 铜缆三脉冲 0.3/0.7/1.1s 阶梯，截图光点在链端口与缆道上，呼吸感成立。
- 可读性 5：n_GPU × per-GPU 双向 = BW_domain 恒等式链 + 四域表 + 18 trays↔9 trays 机架构图，三层各自一眼可读。
- 层级 5：恒等式 attn 终点卡最重、域聚合列 seal 加粗、机架构图淡染退居证据层；注记两行收尾。
- 细节 4：间隙注记「铜缆 cartridge · 恰好全连接」横跨度盖过左右 zone 虚线竖边（crop p07f2-gapcap：文字行与分区线结构相交，3.5s 时 dash 相位落在字隙、未压笔画）；tray 网格 18×(4+2)/9 对齐整齐。
- 咬合 5：式 4 恒等式、2×8×18×25=7200、7.2/14.4/129.6≈130/1036.8 TB/s 四域与出处列、1296=72×18=9×144、72 Blackwell+36 Grace、液冷铜缆全对 §4。

### fig-3 threeLayerInterconnect（7.1 b1）— 4/5/5/5/5 → pass
- 动效 4：NV-HBI 正向脉冲 0.2s + NVLink5 脉冲 1.2s + C2C 脉冲 0.8s，双向边以双箭头静态表达、脉冲只走正向，语义不混（截图双箭头+单光点）。
- 可读性 5：封装 zone 内 die↔die、 zone 外 Grace↔die0、die1↔另一颗 B200 三层三对象各连各的；PCIe 虚线对照一眼可辨。
- 层级 5：三条互连标签 seal/mono 点睛、PCIe plain 灰卡退为对照；底部排序/机制/坑三行字级阶梯。
- 细节 5：双向箭头两端各入卡边、C2C 竖直双箭头端点精确落卡边中点；标签堆（NVLink5/1.8 TB/s/双向）在线上方不压线；全文字出框检查通过。
- 咬合 5：NV-HBI 10 TB/s 方向未披露按原值引、C2C 900 双向一致统一内存、NVLink5 1.8 双向、PCIe 128 双向=1/14、HBM 7.7 对比、CHI/CXL 6× 能效、坑注三层勿混全对 §5。

### fig-4 hgxTopologyRing（7.2 b0）— 4/5/5/4/5 → pass
- 动效 4：ring 一跳三段脉冲（入 1.2/横 1.0/出 1.2，delay 0/0.4/0.8）+ PCIe 三脉冲 0.2/0.5/0.8 阶梯，截图光点完整覆盖两条链。
- 可读性 5：ring 一跳高亮 GPU5→crossbar→GPU6 与灰汇接线分离，「ring 一跳」seal 标签居两竖线之间；PCIe H2D 四卡链独立成段。
- 层级 5：拓扑主图最重、PCIe 侧路径次级标题领起、底部三行口径注记收尾；crossbar 长卡横贯统摄。
- 细节 4：ring 一跳竖线穿 crossbar 卡填充、箭头终止于卡内标题右侧空白（作者注释声明的设计），横线脉冲 y=196 在标题字带下 3px 掠过——可读但线穿卡面略糊卡边界；其余对齐干净。
- 咬合 5：每跳 raw 450 / 模型 370.8=18×20.6、η=82.4%、SM100 40.1→721.8、min(64,40)=40=62.5%（教学假设标注）、4-GPU 无 switch / PCIe 版 600≠900、9×/18× 全对 §2-§7。

### fig-5 peerAddressMapping（7.3 b0）— 4/5/5/5/5 → pass
- 动效 4：解引用竖脉冲 0.2s + 映射横脉冲 0.6s + load/store L 形脉冲 1.2s 阶梯，截图三点各居其段，启用后路径语义完整。
- 可读性 5：启用前 ✗ 虚线断链 vs 启用后双 zone 映射+事务路径，对照一眼成立；「映射/peer access」标签在线上/下各归其侧。
- 层级 5：映射 strong 边最重、kernel attn 染、HBM cache 染；底部四行工程事实字级递减、scope 行 seal 收尾。
- 细节 5：L 形事务线跨 zone 边进入 GPU1、箭头精确入 HBM 底边；✗ 居断线上方；注记四行均在 viewBox 720×372 内。
- 咬合 5：UVA 同一指针两卡通用、三步 API（CanAccessPeer→EnablePeerAccess 单向双向各一次→用）、全局生效乘性开销→VMM、8-peer→n≤9、thread_scope_system 与单 GPU atomic 限制全对 §3。

### fig-6 p2pVsBounce（7.3 b1）— 4/5/5/4/5 → pass
- 动效 4：bounce 两跳脉冲 0/0.5s + P2P 单跳 strong 脉冲 1.0s，截图串行两光点 vs 单跳一光点，跳数差由动效直接表达。
- 可读性 5：上下两路径同端起终点对照（GPU A→GPU B），host 中转卡有无即路径差；时间账三条横条收尾定量。
- 层级 5：P2P strong 边 + seal 标签最重、bounce 灰弱；段标题 seal 加粗领起；增益两层行 seal 收束。
- 细节 4：「D2H · PCIe 跳 1」cap 中心 202、宽约 83 → 左缘 160.5 越 GPU A 卡右边 166 约 5px，首字母 graze 卡边框（截图可见 D 贴卡缘）；H2D cap 与时间账条均干净。
- 咬合 5：host 2M vs 0、串行谐波均值/流水 min、31.25/15.625/2.22 ms、14.06×/7.03×、路径 7××跳数 2×、PCIe P2P 只拿 2×、API 报值≠必然 P2P 坑全对 §4。

## 2. gpu-p08（13 图）

### fig-0 primitiveVolume（8.1 afterHeading §3）— 4/5/5/5/5 → pass
- 动效 4：纯静态双 zone 账表（条+公式+值），无动效需求、reduced 终态天然（两帧差 0）。
- 可读性 5：均衡四 vs root 四双分区、每行 名→公式→条→值 四列节奏统一，不读正文即懂「与 n 解耦 vs root 热点」。
- 层级 5：AR seal 条最长最重、broadcast/reduce warm 次之、gather/scatter soft 有界弱染；zone 标签点题、底部三行注记收尾。
- 细节 5：条宽 v×160 线性、值标签随条尾；a2a 行「0.875 GB · 每 peer M/n = 0.125」长标签止于 zone 右缘内；八行基线 26px 等距。
- 咬合 5：八原语系数与 n=8/M=1GB 数值（1.75/0.875/0.125/7 GB）逐格对 §3 表；命题 1 自环 125MB、ncclAlltoAll count 每 peer、AR=RS+AG ZeRO 同构全对。

### fig-1 ringVsA2a（8.1 b0）— 4/4/5/4/4 → pass
- 动效 4：ring 四边脉冲 0/0.4/0.8/1.2s 轮转 + a2a 五脉冲阶梯，截图环上四点轮转、完全图光点分布，节奏成立。
- 可读性 4：ring 侧方向闭环正确；a2a 侧只画 6 条有向边（每对单箭头：r0→r1/r2、r1→r2/r3、r2→r3、r3→r0），出度不齐（r2/r3 各 1），与「每 rank 向 n-1 peer 各发一份」的双向全连接有差距——无向 K4 形状+底部「每 rank 收发 (n-1)/n·M」注记补偿，但方向cue不完整。
- 层级 5：双 zone 对称、a2a attn 染 vs ring data 染区分语义；底部语义差/排列两行注记字级清晰。
- 细节 4：K4 两对角虚线中心交叉为方形布局固有（虚线区别于实边、不算打架）；对角脉冲穿交叉点可读；节点/箭头对齐干净。
- 咬合 4：ring 2(n-1)=6 步均匀负载、a2a 单跳每 peer 0.125、全 bisection+incast→10.3、归约 vs 搬运语义差全对 §4；唯 a2a 有向边子集化简未逐向兑现章节 ASCII 的 12 向（见可读性）。

### fig-2 ringTwoPhase（8.2 b0）— 4/5/5/5/4 → pass
- 动效 4：逻辑环四脉冲 0/0.35/0.7/1.05s 轮转（截图四点分居四边），两阶段靠时间轴+终态矩阵静态表达，分工合理。
- 可读性 5：左环→右上 6 步时间轴（warm 3 步+seal 3 步）→右下 SR/AG 终态矩阵，三段递进一眼懂两阶段复合。
- 层级 5：时间轴 seal 半段与 AG 全 hot 矩阵最重、SR 对角 hot 点睛；底部总量/单阶段/归纳三行字级阶梯、硬规则行 seal。
- 细节 5：矩阵行列标签 r0-r3/c0-c3 对齐、hot/warm 染区分部分和/全和；时间轴刻度 1-6 居半格；全文字在 720×382 内。
- 咬合 4：2(n-1)/n·M 式 1、单阶段 0.875 与 2× 硬规则、归纳 recap 不重推全对；**SR 终态取主对角（rank i 持 c_i）而章节 §2 括注例为「r1 持 c0」、本模块 fig-5 流水格（章节 b3 原表）推得 rank i 持 c_{i+1}**——旋转约定三处不一，不变式「每 rank 恰持一个全和块」仍成立，判注记级瑕疵。

### fig-3 lowerBoundTrees（8.2 b1）— 4/5/5/5/5 → pass
- 动效 4：归约树三脉冲 0/0.3/0.6 + 广播树三脉冲 0.9/1.2/1.5 + 相加链三脉冲 1.6/1.9/2.2，九脉冲阶梯覆盖证明全路线（截图光点分布）。
- 可读性 5：左「部分和流入 r_c」右「全和流出 r_c」镜像对照 + 中部不相交 seal 断言 + 下部四卡相加链，证明骨架不读正文可复述。
- 层级 5：相加链终点 2(n-1)M attn 卡最重、运算符 +/=/×n 灰弱居隙；zone 标签携带信息状态分类；底部五行注记阶梯。
- 细节 5：运算符居卡隙正中、脉冲端点落卡左端口不压字；「n chunk 求和」小注居 ×n 下方；全文字在 720×424 内。
- 咬合 5：归约树/广播树各 (n-1)M/n、不相交相加每 chunk 2(n-1)M/n、×n 得 2(n-1)M 式 3、14 GB/1.75 GB 达界、单阶段单独归属、弱界差 1.75× 全对 §3。

### fig-4 ringPhysical（8.2 b2）— 4/5/4/5/5 → pass
- 动效 4：八边环脉冲 delay k×0.2s 轮转 + 物理路径两脉冲 0.3/0.8s，截图环上八点轮转方向一致（顺时针），逻辑→物理映射由虚线桥接。
- 可读性 5：八边形逻辑环（0 attn 点睛）与 GPU_i→NVSwitch→GPU_i+1 物理三卡并置，「环邻居是逻辑的」一眼成立。
- 层级 4：左环主、右路径次；右 zone 仅 130 高、其下注记后留白偏大（右列 200-300 带空），双列重心略偏左。
- 细节 5：anchor() 边界锚点使八边箭头精确落卡边、无穿卡；虚线桥跨 zone 隙干净；底部两行代码锚注记对齐。
- 咬合 5：每跳 raw 450/模型 370.8、非阻塞任意排列、ncclBuildRings rings.cc L29 + loopback L51 + 全覆盖两校验、直连拓扑边界（下界拓扑无关/带宽由拓扑供给）全对 §4。

### fig-5 ringPipeline（8.2 b3）— 4/5/5/4/5 → pass
- 动效 4：四 rank 链路横线各一脉冲 delay i×0.3s，截图四点错位分布=「n 条链路同时忙」的直接动效表达；多 ring 条静态。
- 可读性 5：步×rank 网格（发 cX 卡）+ 行下链路光点 + 填充期/稳态括线，流水机制不读正文即懂。
- 层级 5：网格主、稳态注记次、多 ring 三条账（seal k=8 点睛）收尾；字级三层清晰。
- 细节 4：k=8 条底 y=400 与口径注记基线 410 的字顶 ≈400.5 仅 ~0.5px  clearance（截图两行可辨但极紧）；网格卡 150×30 与行线 36 偏移对齐整齐。
- 咬合 5：4×3 发送格与章节 b3 原表逐格一致（r0: c0/c3/c2 …）；2(n-1)α 摊薄 2.8e-14 s/B≈β 1/96；k=1/4/8 → 4.72/1.18/0.59 ms、延迟恒 0.028、MAXCHANNELS 64 与饱和口径全对 §5。

### fig-6 doubleBinaryTree（8.3 b0）— 4/4/5/3/5 → **REDRAW**
- 动效 4：两树水平根边+部分竖边脉冲阶梯（0→1.2s），截图光点沿树下行分布，叶子边静态留白合理。
- 可读性 4：双树镜像+朱/蓝/纸面三色互补一眼可验（偶内部左朱右纸、奇内部左纸右蓝）；**但 mirror 根边 11→3 的箭头落在 3 卡左缘且指向左（朝卡外）**，与脉冲右→左流向矛盾，该边方向 cue 错置。
- 层级 5：双 zone 对称、内部节点染 vs 叶子纸面、底部缺陷/双树/数值三行注记阶梯、双树结论 seal 加粗收束。
- 细节 3：**hz() 以 (11,3) 反序调用（helper 假设 a 在左 b 在右）→ 路径 M 614 84 H 402 横穿 3 与 11 两张卡内部并压住卡名「3」「11」字带**（crop p08f6-mirrortop 实证：灰线穿两卡中线、箭头在 3 左缘朝外）；另左树叶子「1」卡 x=32 越 zone 左缘 36 约 4px。reduced 静态帧同样可见穿卡线（缺陷不随动效消失）。
- 咬合 5：12-rank 双树结构与 trees.cc L69-88 注释 ASCII 逐边一致（0-8/0→4/4→2,6/8→10/2→1,3/6→5,7/10→9,11；mirror 3-11/11→7/3→1/7→5,9/1→0,2/5→4,6/9→8,10）；3D+D=2M、差 2M/n=0.25 GB、mirror 偶/shift 奇 L90 全对 §3。
- 重画要求：mirror 根边改为「11.left → 3.right」的反向水平通道（箭头入 3 右缘朝左），线只走卡隙 434..582；顺带把叶子 1 的 x 收进 zone（≥40）。其余构图/配色/注记保留。

### fig-7 ringTreeCrossover（8.3 afterHeading §4）— 4/5/5/5/5 → pass
- 动效 4：纯静态坐标图（两直线+交叉点+虚线引导），无动效需求、终态天然（两帧差 2px 噪声）。
- 可读性 5：ring seal 低截距缓斜率 vs tree ink 高截距陡斜率、交叉点朱点+双虚线引导到轴，「小 tree 赢大 ring 赢」分区标签各居空白侧，一眼可读。
- 层级 5：交叉点与 M* 标签 seal 加粗为焦点、线端 ring/tree 标签次之、轴刻度弱；底部公式/前提/三临界三行阶梯。
- 细节 5：刻度 0-4 MB / 0-80 μs 对齐、截距标签 28/12 μs 贴线起端、M* 标签居刻度行下不与轴名撞；全文字在 720×438 内。
- 咬合 5：M*=n(n-1-log₂n)α/((n-1)Δβ) 式 2、n=8/α=2/Δβ=4.995e-12→1.83 MB、T≈36.6 μs、Δβ>0 前提与除零坑、三临界 1.83/5.93/667.4 勿混、斜率 1/370.8 vs 1/130 教学假设标注全对 §4。

### fig-8 ncclInitFlow（8.4 b0）— 4/5/5/4/5 → pass
- 动效 4：API 链两脉冲 0/0.3 + 入 zone 竖脉冲 0.6 + 五阶段四脉冲 0.5→1.4 阶梯，截图光点沿全链分布，异步 job→主体分段清晰。
- 可读性 5：上链（API→Dev/Func→bootstrap）+ zone 内五阶段横链（拓扑→路径→图搜索→transport→devComm），file:line 小注居各卡下，代码走读图一眼可索。
- 层级 5：zone 统摄主体、五卡角色染区分阶段、图搜索 attn 点睛；底部模型值/路径梯子两行注记收尾。
- 细节 4：阶段卡隙仅 8px 而脉冲 r4.2（直径 8.4）→ 光点在隙内时 graze 两侧卡边（截图点半骑卡左缘），不压字；卡内 title/sub 字级 10.5/7.5 均在卡宽内。
- 咬合 5：L3071/L2975/L2229/L2314/L1219 链、topo.cc L1994/paths.cc/search.cc L1245/L1957/L2076·L847、SM90 20.6/SM100 40.1 两层账、ringGraph max 32/treeGraph 取 ring、PATH 梯子全对 §2。

### fig-9 transportArchitecture（8.4 afterHeading §4）— 4/5/5/4/5 → pass
- 动效 4：src→canConnect 脉冲 + 四件套扇出四脉冲 0.3→1.2 阶梯，截图光点分居扇出曲线（两点近源为相位使然），选择机制动效成立。
- 可读性 5：canConnect 扇出四卡 + NVLS seal zone 独立虚线路径（「不在 ncclTransports[]」标签点题），四件套与 bypass 一眼分离。
- 层级 5：四件套角色染卡右列主、NVLS seal zone 左下对照、延迟三条账底部收尾（NVLink seal 最重）。
- 细节 4：NET 条底 y=382 与口径注记字顶 ≈382.5 近贴（截图可辨）；NVLS 虚线连接器末端小 hook 卷曲（352 200→344 196→340 204）略赘但箭头落卡顶正确。
- 咬合 5：p2pMap L358/cudaIpcOpenMemHandle/useRead、shm ACS/IOMMU proxy 拷、net GDR/sendProxyProgress L1327、collNet sharedBuffersInit L416→10.2、NVLS cuMulticastCreate L270、延迟 3.4/5.7/14.0 μs（1.7×/4.1×）与校准常数 vs α=2 口径分离全对 §4。

### fig-10 proxyGdr（8.4 b1）— 4/5/5/5/5 → pass
- 动效 4：GDR 三脉冲 0/0.5/1.0 + 非 GDR 四脉冲 1.3→2.2 + 状态机两脉冲 2.4/2.7，全模块最晚 delay 2.7s 仍 <3.5s，截图三段光点齐备、控制面虚线静态分工正确。
- 可读性 5：GDR 数据面直链（kernel→HBM→NIC→fabric）+ proxy 只提交/轮询虚线上挂 vs 非 GDR 多 host 缓冲一跳，对照一眼成立；三计数器链收尾。
- 层级 5：两路径段标题 seal 领起、数据面实边最重、控制面虚线弱；状态机三色卡 + NCCL_STEPS/maxDepth 右注平衡。
- 细节 5：「proxy 拷」cap 居 34px 卡隙内不越卡、「posted」/「isend / test（done++）」各居虚线侧；RDMA READ hop 标签与 NIC sub 重复为有意强调；全文字在 720×404 内。
- 咬合 5：GDR 控制面零拷贝 vs 非 GDR 数据面 D2H、posted/transmitted/done 三计数器 8 槽轮转、maxDepth 防覆写、sendProxyProgress L1327、shm 同数据面拷贝全对 §5。

### fig-11 protocolLines（8.4 b2）— 4/5/5/5/5 → pass
- 动效 4：纯静态 line 布局+效率条账，无动效需求、终态天然（两帧差 0）。
- 可读性 5：LL 四槽 data/flag 交替、LL128 长 data+尾 flag、Simple 全数据无 flag 三行同构对照，η 50%/93.75%/→1 右列点睛，flag 即同步一眼可读。
- 层级 5：flag seal 槽最重（白字反白）、data warm/简灰次之；带宽项三条（Simple seal）与 C*/坑两行收尾阶梯清晰。
- 细节 5：flag 槽白字 7-8px 对比足、LL128 flag 30px 槽容 17px 字居中；Simple 内注记居 rect 中部不溢；条值标签随条尾右移整齐。
- 咬合 5：16B=4×4B/128B=15×8B+8B/512B grain、readLL 自旋双 flag 无 fence、LL128 128B 原子性仅 NVLink（门控 intra≤PATH_NVB）、9.44/5.03/4.72 ms、C*=667.4 KB→M*≈381.4 KB、PCIe 强开数据损坏坑全对 §6。

### fig-12 deviceDispatch（8.4 b3）— 4/5/5/4/5 → pass
- 动效 4：入口链两脉冲 0/0.4 + 扇出四脉冲 0.8→1.7 阶梯，截图光点沿扇出曲线分布，编译期特化→运行期查表的分发机制动效成立。
- 可读性 5：三卡入口链→zone 内四骨架卡（RING/TREE Simple/TREE LL/NVLS）扇出，per(algo×proto) 标签点题，dispatch 结构一眼可索。
- 层级 5：入口链 proj/op/data 染、四骨架角色染区分、底部函数表/70-30/CUTLASS 三行注记阶梯。
- 细节 4：扇出四线共享 d3 底心原点成束后分散（固有扇出、无杂乱交叉）；NVLS 卡 sub「scatter/gather/reduce-bcast L390」约 140px 贴 148 卡宽极限（截图未溢但余量 <8px）。
- 咬合 5：common.cu L23/common.h L420/RunWorkBatch L303、runRing L14/runTreeSplit L146/LL 无 treeUpDown/NVLS L390、70% 归约/30% 广播注释逐字、ncclDevFuncTable 五元索引、CUTLASS manifest 同构、channel↔CTA blockIdx.x 全对 §9。

## 3. redraw findings（任一维 <4）

1. **gpu-p08 fig-6 doubleBinaryTree（8.3 b0）细节完成度=3**：`hz(T2[11], T2[3], 0.3)` 反序调用左→右 helper，mirror 根边路径 `M 614 84 H 402` 横穿 rank 3 与 rank 11 两卡内部、压卡名字带，且箭头落 3 卡左缘朝外（方向 cue 与 11→3 流向矛盾）。crop 与 reduced 静态帧双实证。修法：反向水平通道（11.left→3.right，线走卡隙 434..582，箭头入 3 右缘）；顺带叶子「1」卡 x 32→≥40 收进 zone。其余维 ≥4，构图/配色/咬合保留，局部重画即可。

## 4. 内容咬合专项核验（任务清单）

- **NVLink 方向口径**：p07 fig-0 链 25 per-link 单向→450 per-GPU 单向→900 双向头条 + 五代双列 + 坑注 ÷2 ✓；fig-1 单对 900 双向 vs 直连 128 ✓；fig-4 / p08 fig-4 每跳单向 raw 450 / 模型 370.8 ✓；p07 fig-6 P2P 450 单向 ✓。无头条当单向代入处。
- **ring 总量 vs 单阶段**：p08 fig-0（AR 1.75 vs AG/RS/a2a 0.875）、fig-2（式 1 总量 + 单阶段 0.875「2 倍勿混」seal 行）、fig-3（总流量 2(n-1)M + 单阶段 (n-1)M 单独归属 + 2× 坑）、fig-5（带宽项 2(n-1)/n·Mβ）四处口径一致 ✓ 无 2× 混用。
- **double-tree**：p08 fig-6 结构与 trees.cc 12-rank mirror ASCII 逐边一致、偶/奇内部互补无重叠、3D+D=2M、差 2M/n=0.25 GB ✓（渲染缺陷另列 redraw，不影响内容咬合 5）。
- **M***：p08 fig-7 交点 1.83 MB/36.6 μs、式 2 闭式、Δβ=4.995e-12、Δβ>0 前提与理想模型除零坑、三临界（1.83/5.93/667.4）勿混 ✓；斜率 tree 1/130 GB/s 明标教学假设 ✓。
- 其余抽查：p07 fig-2 域聚合四值与出处列、fig-3 三层排序与 NV-HBI 方向未披露按原值引、p08 fig-8/9/10/11/12 的 file:line 锚（L3071/L1219/L358/L1327/L416/L270/L14/L146/L390 等）与章节逐字一致 ✓。
- 注记级瑕疵（不判罚、登记备查）：p08 fig-2 SR 终态旋转约定与章节括注/fig-5 流水格不一（不变式仍成立）；p08 fig-1 a2a 有向边子集化简；p07 fig-1 入曲线穿 zone 标签词隙；p07 fig-2 间隙注记跨 zone 虚线边；p07 fig-6 D2H cap graze 卡缘；p08 fig-5/fig-9 条底与注记字顶 ~0.5px 紧 clearance。

## 5. 汇总

- **20 图：19 pass / 1 redraw**。redraw = gpu-p08 fig-6（细节 3）。
- 最低分维：gpu-p08 fig-6 细节 3（唯一 <4）；pass 图最低为 gpu-p08 fig-1 可读性 4 / 咬合 4（a2a 方向子集），其余 pass 图各维 ≥4、多数 5。
- 全局纪律：census 15+5+1 skip ✓、锚点唯一 ✓、页面级 PULSE_GRAD_ID ✓、reduced 终态 20/20 静态 ✓、3.5s 语义完整 ✓。
- 复审通道：fig-6 局部重画后重跑 `node scripts/fig-check.mjs gpu-p08` + crop 复核 mirror 根边卡隙与箭头入边，即可闭环。

## 6. redraw closure addendum（2026-11-25，T7 closeout）

- **gpu-p08 fig-6 doubleBinaryTree redraw 闭环**：commit 6b14438 将 mirror 根边改为 11.left→3.right 反向水平通道（线只走卡隙 434..582、箭头入 3 右缘朝左），叶子「1」卡 x 收进 zone。本轮（T7）重跑 `node scripts/fig-check.mjs gpu-p08` 全渲染 + fig-6 截图目视复核：mirror 根边不穿卡、方向 cue 与 11→3 流向一致、叶子 1 在 zone 内 → **redraw 闭环，20/20 QUALITY pass**（19 pass + 1 redraw → 全过）。
- 本档案 §2 fig-6 打分（细节 3）与 §3 redraw findings 保留为重画前原判（历史记录）；重画后细节 ≥4，其余维维持。
- **T7 伴随图文同步修（gpu-p07 fig-6 p2pVsBounce）**：章 7.3 S2 术语修「谐波均值」→「半谐波均值（并联电阻式）」（式 (1) 与全部数值不动），图标签「串行端到端 = 谐波均值」同步改「半谐波均值」；坑注 cap「被 IOMMU/ACS 挡掉仍经 host」同步改「未启用 / IOMMU 挡经 host（~32 GB/s），ACS 开绕行 root complex」（与章 7.3 §4 坑注 P7-05 拆分一致）。重跑 `node scripts/fig-check.mjs gpu-p07` 全渲染 + fig-6 目视复核：两行标签均在框内不溢、数值咬合不变（32/64/450 → 31.25/15.625/2.22 ms、14.06×/7.03×）。
- 本档案 §1 fig-6 咬合行「串行谐波均值」为修前原文，保留为历史记录；修后章与图口径均为半谐波均值。
