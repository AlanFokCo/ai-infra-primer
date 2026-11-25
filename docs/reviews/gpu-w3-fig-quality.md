# gpu W3 场景图 QUALITY 评审档案（gpu-p05 / gpu-p06，22 图）

- 评审日期：2026-11-25；评审角色：fresh design-eye evaluator（只读，未改模块）
- 规约：`site/scripts/lib/figures/QUALITY.md`（五维各 ≥4 过线，截图证据）+ `STYLE.md`
- 证据：`cd site && node scripts/fig-check.mjs gpu-p05`（10 张）/ `gpu-p06`（12 张），3.5s 截图时刻逐张 Read `site/qa/figs-gpu-p05/fig-*.png`、`site/qa/figs-gpu-p06/fig-*.png`；局部放大用 sips crop（/tmp/figcrop/p05-fig0-right.png、p05-fig4-cap.png）
- 判定：**20 pass / 2 redraw**（redraw 均为 gpu-p05，细节完成度=3）

## 0. census 与全局纪律核验

- 注册表 census：gpu-p05 = 7 replacesBlock + 3 afterHeading；gpu-p06 = 7 replacesBlock + 5 afterHeading → **14 replacesBlock + 8 afterHeading = 22** ✓
- 逐章无语言围栏块核对（grep '^```$' 开闭配对）：5.1 b0/b1/b2 全替换；5.2 b1/b3 替换、**b0 嵌套循环与 b2 producer-consumer 执行体为伪代码跳过**；5.3 b0 替换、**b1 ln_fwd 五段伪代码跳过**；5.4 b0 替换；6.1 b0/b1、6.2 b0/b1/b2、6.3 b0、6.4 b0 替换 → **3 pseudocode skip** ✓（管线纸面算法面板接管）
- afterHeading 锚点逐章 grep '^## ' 唯一命中：5.2 `^## 2\. `→L17；5.3 `^## 3\. `→L91；5.4 `^## 2\. `→L15；6.1 `^## 5\. `→L101；6.3 `^## 3\. `→L43、`^## 5\. `→L132；6.4 `^## 4\. `→L102、`^## 6\. `→L118 ✓
- 脉冲渐变：两模块均 import `PULSE_GRAD_ID`（='fig-pulse-grad'），kit edge pulse 与模块 pathPulse 全引用页面级固定 id；`PULSE_GRAD_DEFS` 由 `src/layouts/Chapter.astro` L27 注入、fig-check.mjs L32 同注入；**无 per-scene defs/id 错配** ✓
- reduced-motion 静态终态：`src/styles/figure.css` L201 `.dia-pulse{display:none}`（reduce 态）+ L264 sig-* 复位块；Chapter.astro L119-148 入视口 unpause / 出视口与 reduce 态 `pauseAnimations()`。两模块只用 dia-pulse（无 sig-*），降级态=完整静态图 ✓
- 截图时刻语义：所有 pulse delay ≤2.0s、dur ≤2.6s，3.5s 时各脉冲已入循环稳态 ✓（但端点 dwell 造成两处压字，见 redraw）

## 1. gpu-p05（10 图）

### fig-0 reductionTree（5.1 b0）— 4/5/4/3/5 → **REDRAW**
- 动效自然度 4：叶→层0→根与层级链脉冲 spline ease-in-out + 0.12/0.15/0.4s 阶梯 delay，节奏有呼吸（截图左树光点分布）。
- 场景可读性 5：左 8 元素二叉树 + 右 register→warp→block→grid 四卡链，不读正文即懂「树达界 + 层级拼接」（截图双 zone）。
- 视觉层级 4：双 zone 均衡、root attn 染色点睛、底部四行注记字级阶梯清晰；右列卡缝偏挤。
- 细节完成度 3：**右列 4 张 h=32 层级卡 sub 基线 = y+h/2+16 = 卡底边框**（kit 公式），4 条 sub 标签骑在边框线上（crop 实证：register/warp/block/grid 卡 sub 均被底线穿descender）；卡隙仅 6px，pulse r4.2 端点 dwell，3.5s 光点压住 block 卡 sub「warp0 shuffle 3 步」中部、graze register 卡 sub（crop 红点位置）。
- 与正文咬合 5：W=7/D=3、depth 81=63+5+3+10、67.11 MB/20.02 μs、partial 4 KB 0.0061%、W=16777215/D=24、Brent 64+24=88、I=0.25 离 ridge 1180× 全对 §2。

### fig-1 warpShuffle（5.1 b1）— 4/5/4/4/5 → pass
- 动效 4：纯静态 lane 态表 + 流量账条，无动效需求，reduced-motion 天然终态。
- 可读性 5：down 形 4 行×8 lane 状态表 off=4/2/1 逐行收敛、lane0 朱砂热格收尾；butterfly 全 Σ 行对照一眼分形（截图表区）。
- 层级 4：表主体 > butterfly 对照 > 64B/4084B 账条 > 注记，三层清楚。
- 细节 4：列对齐、mono 数字、热格描边 crisp；zone 标签长行贴边但不出框。
- 咬合 5：offset 4,2,1 / 32 lane 16,8,4,2,1、「will not wrap around」、64 B 对 4084 B 省 63.8×、conflict 0、barrier 1 对 8、redux.sync 仅 u32/s32 全对 §3。

### fig-2 blellochScan（5.1 b2）— 4/4/4/3/5 → **REDRAW**
- 动效 4：up-sweep 三层脉冲阶梯 delay（0.1/0.12/0.15s）有节奏；但端点 dwell 落点不佳（见细节）。
- 可读性 4：树 + down-sweep 结果 strip 结构清楚；**3.5s 截图二层的「10」「26」节点数字被脉冲光点部分遮字**（截图树二层左/右节点读作 1●/●6），关键中间和瞬时不可读。
- 层级 4：root 36 attn 点睛、strip 朱砂结果带、注记阶梯清楚。
- 细节 3：**光点压节点数字**（10/26）+ 11/15 底边框与叶「3」上落点（截图）；短边 16–20px 配 22px 高节点卡，端点 dwell 使 r4.2 光点落入卡内字区。
- 咬合 5：x=[1..8]、up 3/7/11/15→10/26→36、置根 0、y=[0,1,3,6,10,15,21,28]、28+8=36、W=2(n−1)/D=2log2 n、33554430/depth 48、12× work、134.22 MB/40.04 μs 全对 §5。

### fig-3 cutlassHierarchy（5.2 b1）— 4/5/5/4/4 → pass
- 动效 4：左链 4 脉冲 + 右流 3 脉冲阶梯 delay，箭头三角补方向（截图灰三角）。
- 可读性 5：五层实体自顶向下 + 右 HBM→L2→shared→TC→register 数据流，双栏互不干扰。
- 层级 5：左链标题重、file:line 副行弱；右流 attn TC 点睛；底部闭合账注记第三层；留白充足。
- 细节 4：h=52 卡 sub 余量 10px 干净；光点 graze 卡缝边框但不压字；右列边标 cp.async/TMA、描述符、D=A×B+C 对齐右缘。
- 咬合 4：五层 file:line（L142/L289/L398、L65/L269、L72、L211/L460/L99、L2842）、32768 B/÷11.49≈2852、49152 B/96 B/cycle、128 regs、8 条 wgmma/2097152 FLOPs、512 cycles=279.8 ns 全对 §3-§6；**微瑕：HBM→shared 边标「cp.async/TMA」，SM90 走读配置 load 仅 TMA（§5），cp.async 属 SM80 族泛标**。

### fig-4 gemmMainloop（5.2 b3）— 4/4/4/4/5 → pass
- 动效 4：producer→consumer 交接脉冲 0.3/0.7/1.1s 阶梯 + shared 环 pathPulse 2.6s 循环（截图环右下光点）。
- 可读性 4：三泳道 + stage 列头 k..k+3、consumer 右错 30px 表滞后，一眼懂交接；微瑕：交接边 0 永久穿过「acq→2 条 op · expect_tx 32768」注记（crop 实证灰线过 expect_tx 字区）。
- 层级 4：producer norm / consumer attn 角色分色、shared 缓冲纸色第三层、注记阶梯清楚。
- 细节 4：泳道对齐、环虚线圆角折线 crisp；除上述线穿注记外无溢框。
- 咬合 5：Stages=4/T_K=64、warp0 elected、2 条 op/expect_tx 32768、wait→gemm→commit→wait<1>→release、L547 先于 L551、τ 1290.4/279.8、S_min=3、cluster 2×1 省 25%（1.610 GB）全对 §4/§5。

### fig-5 l2Wavefront（5.2 afterHeading §2）— 4/4/4/4/5 → pass
- 动效 4：静态对比图（网格 + 因子卡 + 强度条），无动效需求、终态天然。
- 可读性 4：左 6×6 波前网格 + 右两形因子卡 + ridge 竖线，bound 翻转一眼懂；微瑕：对角热格语义靠注记「A tile 行内共享 · B tile 列内共享」补（热格本身易误读为「仅对角 block 活跃」）。
- 层级 4：方阵波前卡 attn 染色 = 论点强调、raster 卡素色对照、seal 强度条越 ridge 线点睛。
- 细节 4：网格/条/ridge 线对齐 crisp，条宽 64:735 比例正确，注记三行不出框。
- 咬合 5：7.31/467.7、11.49/735.3/186.9 MB、2b/(r+c)、65.8 μs < 138.9 μs、24.1 MB < 50 MB、sm90_tile_scheduler.hpp、坑注全对 §2。

### fig-6 softmaxVariants（5.3 b0）— 4/5/5/5/5 → pass
- 动效 4：online 区 m→ℓ→O 两脉冲 0.3/0.8s 阶梯，20px 卡隙落点干净（截图光点在隙内）。
- 可读性 5：三变体行「卡 + IO strip + 比例条 + 右对齐 MB/μs」平行结构，读几遍决定快慢一眼成立。
- 层级 5：行驻留 attn 朱砂 = 达下界论点、条长 268:201:134 比例即论证、online 区第二层、注记第三层。
- 细节 5：h=48 卡 sub 余量 8px、条与右对齐数值留隙充足、zone 标签与 m/ℓ/O mono 标题全不出框、无重叠。
- 咬合 5：268.44/80.08、201.33/60.06、134.22/40.04、16 KB ≪ 227 KB、256 线程×16、shared 64 B、SFU 4.34 μs/9.2×、exp2 L79/L86 全对 §2（O 更新式与 6.3 式 (2) 同构，跨章一致）。

### fig-7 welfordChan（5.3 afterHeading §3）— 4/5/4/4/5 → pass
- 动效 4：A/B→合并 两脉冲 0.3/0.6s 阶梯；8px 隙端点 dwell graze 卡底边框但不压字（截图红点在 sub 下方缝内）。
- 可读性 5：左「朴素 −8.0 朱砂警示 vs Welford 1.25 ✓ 绿」对照 + 右增量式/A/B/合并树，灾难消去一眼懂。
- 层级 4：naive attn 警示强调修辞成立、合并 attn 收点、注记四行阶梯。
- 细节 4：A/B 卡 h=36 sub 余量 2px 偏紧但不出框；双 zone 均衡、合并式注记逐位核对行清晰。
- 咬合 5：−8.0/1.25/ulp 8/1.0005×10⁸、式 (4) 三更新、A/B c=4 μ=2.5/6.5 M₂=5、合并 42=5+5+16×2、warp_chan_upd_dynamic L672-701、per-warp two-pass 基例、坑注全对 §3。

### fig-8 fusionStrategies（5.4 b0）— 4/5/4/4/5 → pass
- 动效 4：三列卡→zone 脉冲 0/0.35/0.7s 阶梯，12px 隙落点 graze zone 顶边框、不压字（截图三红点）。
- 可读性 5：三列平行分类卡 + 驻留层粗体 + 代表/注记列表，三类各融什么一眼扫完。
- 层级 4：header 卡角色分色重、zone 体中、note faint 弱，三层字级清楚。
- 细节 4：三列 pitch 220 对齐、zone 内 14px 左垫统一、col3 长公式行不出框。
- 咬合 5：①register/②行驻留/③累加器、residual+ReLU/SiLU+mul/TE/CUTLASS ThreadEpilogueOp、D=act(α·acc+β·C+bias)、FA Θ(T²)→O(T²d²_head/M_SRAM)、Marlin 回链全对 §3。

### fig-9 fusionSaving（5.4 afterHeading §2）— 4/5/4/4/5 → pass
- 动效 4：未融合链 3 脉冲 + 融合首读/尾写 2 脉冲阶梯 delay，60px 宽隙落点干净（截图光点在隙内/卡左缘外）。
- 可读性 5：上「4 op 各一读一写 + ↓写↑读/HBM 往返」对下「单 kernel 长卡首读尾写」，省哪笔一眼成立。
- 层级 4：融合 seal 长卡 = 论点、未融合素色链对照、decode/prefill/预算注记阶梯。
- 细节 4：op 卡等距、标签上下夹边对齐、长卡标题居中不出框；光点 graze op2 左边框瞬时。
- 咬合 5：8×33.55=268.44、67.11、201.33/60.06、decode 3.15/0.94 与 15 μs=16×、R₀32→8/R_th96→2、spill 反噬、fan-out 全对 §2/§4。

## 2. gpu-p06（12 图）

### fig-0 formatBitLayout（6.1 b0）— 4/4/4/5/5 → pass
- 动效 4：静态位布局表，无动效需求、终态天然。
- 可读性 4：六格式 s/e/m 比例条 + bias/最大有限/ULP 三列一眼对照；微瑕：sign 深色段无图例/标签（靠 e/m 内标推断），跨行条宽各自归一（TF32 19bit 与 FP16 16bit 同宽）需读内标计数。
- 层级 4：表行均质（表格体裁）、列头 + 底部三行读数注记阶梯清楚。
- 细节 5：六行条/列对齐 crisp、e/m 内标居中、mono 数字列右齐、无溢框无重叠。
- 咬合 5：FP16 1/5/10 bias15 65504 2^-10 … E2M1 1/2/1 bias1 6 2^-1 全对 §2 表；448 对 480 坑、E2M1 正数集/25%=4×6.25%、≈40→≈261 binade 全对。

### fig-1 mxBlockScale（6.1 b1）— 4/5/4/5/5 → pass
- 动效 4：静态块布局 + 利用率条，终态天然。
- 可读性 5：MXFP4 320px 对 NVFP4 160px 数据段 = 32:16 元素比例自证；1B scale 盒 + 利用率 50.8/90.4 条权衡一眼懂。
- 层级 4：seal 90.4% 条 = NVFP4 精度论点强调、布局行上下文、注记三行阶梯。
- 细节 5：scale 盒/数据条/利用率条三组左缘对齐、1B 顶标与 E8M0/UE4M3 侧标齐、注记随条尾留隙。
- 咬合 5：17B/32=0.53125/3.76×、9B/16=0.5625/3.56×、+4B global 摊薄、100%/12.5% 台阶、a=6.1/V=6、+5.9% 买 8× 细台阶 + 2× 细块、四格式每元素字节全对 §4。

### fig-2 throughputKRatio（6.1 afterHeading §5）— 4/5/4/5/5 → pass
- 动效 4：静态几何比图，终态天然。
- 可读性 5：k16:k32=1:2 与 k16:k32:k64=1:2:4 条长即吞吐比，t_issue 64 cyc 双行 + 大号 2× 点睛，命题 3 一眼成立。
- 层级 4：FP8 seal 卡 + 2× 朱砂大字 = 论点、B200 三行次级、对账/机制/坑注记三层。
- 细节 5：条左缘 x220 统一、k 标居中条上、倍数标随条尾、2× 垂直居中两 t_issue 行之间、无重叠。
- 咬合 5：F=262144/524288、c_SM=4096/8192、t_issue 64、1978.9/989.4=2.0001、9000/2250=4↔UMMA_K 64/16、unpacked 只 2× 坑全对 §5。

### fig-3 materializeVsFused（6.2 b0）— 4/5/4/4/5 → pass
- 动效 4：物化链 3 + 融合链 2 脉冲阶梯 delay；端点 dwell 光点 graze INT4/f2 卡边框但不压字（截图）。
- 可读性 5：上三遍链（读0.515625/写2B/读2B 边标 + 箭头）对下一遍融合长卡，dequant 必须融合一眼成立。
- 层级 4：融合 attn 长卡 + seal 8.65 MB 条 = 论点、物化素色链对照、账条三行比例即论证。
- 细节 4：边标夹边上方不穿线、账条 75.7:33.55:8.65 比例正确、标签随条尾；光点 graze 边框瞬时。
- 咬合 5：4.515625/0.515625=8.76、2.26×、75.7/22.6、33.55/10.0、8.65/2.58、3.88=2/0.515625、临界 76/实测 16–32 全对 §2。

### fig-4 marlinMainloop（6.2 b1）— 4/4/4/4/5 → pass
- 动效 4：7 脉冲 0→1.4s 阶梯覆盖 HBM→shared→register→TC 全链，节奏连贯。
- 可读性 4：四列头 + 三源卡→sh_a/b/s→register fragments 大卡→TC，cache policy 分化（cg / cg+evict_first / 新组时）副标即论点；微瑕：sh_b 左端口光点与箭头三角重叠、s2→reg.bottom 曲线贴 reg 卡底边框走。
- 层级 4：register fragments 大卡 = 重心（粗体标题 + 三 mono 行 + seal 注记）、TC attn 收点、列头/注记阶梯。
- 细节 4：大卡内 5 行文本间距均匀不出框；除端口光点/箭头重叠外对齐 crisp。
- 咬合 5：STAGES=4/T_M16/T_N128/T_K128、cp.async.cg L61/evict_first L75-76/新组时 L405、ldmatrix m8n8.x4、lop3、hmul2、m16n8k16、62.06/41.8 对 3.88/2.94、thread_block_reduce L466 全对 §3。

### fig-5 lop3Dequant（6.2 b2）— 4/5/4/4/5 → pass
- 动效 4：LO/HI 双路各 3 脉冲阶梯 delay，指令流水节奏成立。
- 可读性 5：n7..n0 nibble strip + 抽取序注记 + 双路四节点链 + 常数四卡，magic-number 配方一眼可复算。
- 层级 4：n−8 seal 终点卡 = 结果强调、双路均质对照、常数卡素色第三层。
- 细节 4：LO/HI 路标左下对齐、常数卡等距；微瑕：LO 末/HI 中两处光点与左端口箭头三角重叠（截图红点吞灰三角）。
- 咬合 5：0xea=(a&b)|c、0x6400/0x6408/0x2c00/0xd480=1024/1032/1/16/−72、>>8 非 bfe、−8 融进 SUB/ADD、13+2 条/1.625+0.25、25–50 MAC/L443 全对 §4/§5。

### fig-6 fa2Mainloop（6.3 b0）— 4/4/4/4/5 → pass
- 动效 4：9 脉冲 0→2.0s 阶梯 + V 底部 L 形 pathPulse，环形 mainloop 节奏完整。
- 可读性 4：③④ 编号主链 + rescale 反馈左箭头 + V 虚线转置通路，两 GEMM 夹 softmax 一眼懂；微瑕：S 卡 bottom 端口复用（K 入与 acc_s 出同端口，junction 三线汇聚）、epilogue sub 基线被 aep 端点光点 graze。
- 层级 4：③/⑤ attn、④ norm 主链重、shared cache 卡次级、rescale seal 小标点睛、注记三行阶梯。
- 细节 4：V 虚线折线圆角 + 转置注记夹线上方不穿字；除端口汇聚/光点 graze 外无溢框。
- 咬合 5：grid 32×1×32=1024、逆序末块唯一 mask/masking 2 迭代、gemm_rs P 寄存器源 C→A 零移动、warp 4 沿 M 各 16 行/quad 2 shfl、sum 延后/倒数乘/exp2 FFMA+MUFU.EX2、sVt 同址视图全对 §4（m16n8k16 行=lane/4 布局在章 §4，图 quad 描述同构）。

### fig-7 faTileBudget（6.3 afterHeading §3）— 4/5/4/4/5 → pass
- 动效 4：shared/register 两预算→min 节点双脉冲（register 侧 strong 加粗），0.3/0.6s 阶梯。
- 可读性 5：28% warm 松条 vs 99% seal 紧条 + min(780,129.7)=128 收点，register 绑定一眼成立。
- 层级 4：seal 紧条 + seal min 卡 = 论点、warm 松条对照、式 4/5/6 mono 注记第二层、底部三行第三层。
- 细节 4：轨道框 + 填充条预算隐喻 crisp；微瑕：shared 侧脉冲起点光点落 zone 底虚线边框、min 右端口光点与箭头三角重叠。
- 咬合 5：(128+128)·128·2=64 KB/28%、FA3 160 KB/n_stage2、192+32+28=252/255 余 3、780/129.7、192 档 364 超限 43%、2 blocks/O=0.125、FA3 160 regs/WG + setmaxnreg 24/240 全对 §3。

### fig-8 fa3Pingpong（6.3 afterHeading §5）— 4/5/4/5/5 → pass
- 动效 4：静态相位矩阵 + 比值条（pingpong 机制靠同列对照表达），终态天然。
- 可读性 5：WG1/WG2 同列 TC/SFU 互换矩阵 + 「同列=同时刻」注记，token 环交替一眼懂；SFU:TC=1:2 条长自证 50%。
- 层级 4：TC seal = 瓶颈强调、SFU norm 绿对照、WG0 load 泳道次级、66.7% < 74.8% seal 注记点睛。
- 细节 5：三泳道四列 pitch 128 对齐、lane 卡 h=26 标题余量 8px、条标随条尾、居中注记两行不出框、无重叠。
- 咬合 5：16384 exp2÷16/cycle=559.6 ns、TC 1119 ns、50%、66.7% < 74.8%（740/989.5）、warp_scheduler_barrier_sync/arrive、IntraWGOverlap fwd_step wg_wait=-1、EVICT_LAST 对 EVICT_FIRST 全对 §5。

### fig-9 mlaSeesaw（6.4 b0，依 fed4ca4 修正图）— 4/4/4/4/5 → pass
- 动效 4：K→p、m→p、p→O、跨 WG 六脉冲 0→1.5s 阶梯，seesaw 交错节奏成立。
- 可读性 4：WG0/WG1 双 zone + 跨 WG 更新 zone 三层结构清楚；微瑕：k1→d0 实线与 m→c0 虚线在 zone 间空白处交叉一次（可追踪但属交叉线）、V 同源三行侧标靠右缘。
- 层级 4：跨 WG zone 标签点题「seesaw 核心」、m seal 卡共享点睛、K cache 卡次级、底部三行注记阶梯。
- 细节 4：双 zone 对称、跨 WG p0→O_R / p1→O_L 箭头 + ·V0R/·V1L 注记夹边下方；微瑕：三处光点与左端口箭头重叠、p0/p1 顶边框光点 graze。
- 咬合 5：**WG0 p0=q·K0ᵀ / WG1 p1=q·K1ᵀ ✓、K0/K1 各 9 份 TMA（576/64=9）✓、V0/V1=K 同源 latent 512 列视图（8×64）不另装载 ✓、跨 WG O_R+=p0·V0R / O_L+=p1·V1L ✓、rescale(s0) 对 rescale(s0·s1) ✓、32768=R_file/2 与 16384 竖切 ✓、EVICT_FIRST 对照 ✓** 全对 §2 修正口径；微瑕：「deep-dive 逐字」引句为缩略（原文 "…equivalent to FlashAttention's online softmax algorithm"）。

### fig-10 pagedGather（6.4 afterHeading §4）— 4/5/4/4/5 → pass
- 动效 4：三列×两级 6 脉冲 0→1.0s 阶梯，翻译链流动连贯。
- 可读性 5：逻辑块 0/1/2 → bt[0]=0/bt[1]=2/bt[2]=1 → 物理页 0/2/1 三列链，散落翻译一眼懂；slot 公式顶置。
- 层级 4：bt op 卡中枢、物理页 cache 卡收点、页内连续副标 + 底部三行注记阶梯。
- 细节 4：三列 pitch 220 对齐、箭头三角补方向；微瑕：bt[0] 顶端口光点与箭头重叠、bt[1]/bt[2] 底边框光点 graze。
- 咬合 5：slot(t)=bt[⌊t/b⌋]·b+r(t)、b=64/H_kv8/d128/bf16 → 128 KiB/16 KiB、块表 256 B 驻 L2、TMA per-page 对 cp.async gather/PipelineAsync、FP8 headdim>192 禁 IntraWGOverlap（tile_size.h L51）全对 §4。

### fig-11 splitKvMerge（6.4 afterHeading §6）— 4/5/4/4/5 → pass
- 动效 4：4 partial→2 merge→final 六脉冲 0→1.0s 阶梯，归约树节奏成立。
- 可读性 5：两两 merge ⊕ 树 + final seal 卡（O=Σw·O/Σw、LSE=log2(Σw)+m），base-2 加权归约一眼懂。
- 层级 4：final attn 收点重、merge op 中枢、partial data 卡次级、四行注记阶梯。
- 细节 4：树边贝塞尔汇聚不交叉、final 顶箭头三角 crisp；微瑕：merge 卡顶/底边框光点 graze（汇聚 junction 红点）。
- 咬合 5：w_i=2^{LSE_i−m}、ptx_exp2/log2、vec FMA、clamp −inf + d_sum 守卫免 NaN、n_new=Σ⌈T_i/(b·c)⌉/n_new·H_kv≤G_max、c=5→13×8=104≤112/c=4→128→S=13、cascade η=2.76% 全对 §6。

## 3. Redraw findings（任一维 <4）

- **R1 · gpu-p05 fig-0 reductionTree · 细节完成度=3**：右列层级卡 h=32 使 kit sub 基线（y+h/2+16）恰等卡底边框，4 条 sub 骑线；卡隙 6px + pulse 端点 dwell 使 3.5s 光点压 block 卡 sub 字区。建议模块内修：层级卡 h 32→40、行距 38→46（zone h 同步），或该链改 arrow-only 去 pulse；不改 kit API。
- **R2 · gpu-p05 fig-2 blellochScan · 细节完成度=3**：3.5s 脉冲光点遮 up-sweep 二层节点数字「10」「26」（另 graze 11/15 底边框与叶 3）；短边 16–20px + 22px 节点卡 + 端点 dwell 所致。建议模块内修：树层间距各 +8–10px（viewBox h 同步），或短边去 pulse 仅保留层间长边脉冲。

## 4. 非阻塞观察（不影响过线，供下轮打磨）

1. gpu-p05 fig-4：交接边 0 永久穿过「acq→2 条 op · expect_tx 32768」注记字区（crop 实证）；建议注记左移或边改 port。
2. gpu-p05 fig-3：HBM→shared 边标「cp.async/TMA」宜改「TMA」（SM90 走读配置仅 TMA，§5）。
3. gpu-p06 fig-6：S 卡 bottom 端口 K 入/acc_s 出复用致 junction 汇聚；epilogue sub 基线被端点光点 graze。
4. gpu-p06 fig-9：k1→d0 与 m→c0 馈线在 zone 间空白交叉一次；「deep-dive 逐字」引句缩略宜补全或改「大意」。
5. gpu-p06 fig-0：sign 深色段缺图例；跨行条宽各自归一宜加「条宽=本格式总位宽」注。
6. gpu-p05 fig-5：波前网格对角热格语义靠注记补，宜改行/列带染色表共享。
7. 多处 pulse 端点 dwell 光点与左端口箭头三角重叠（p06 fig-4/5/9/10/11）：可考虑端口外偏 2–3px 起 pulse。

## 5. 汇总

- **22 图：20 pass / 2 redraw**（redraw：gpu-p05 fig-0、fig-2，均细节完成度=3）
- 五维最低分分布：细节完成度 3×2（R1/R2）；其余各维全 ≥4；可读性/咬合多数 5
- census ✓（14 replacesBlock + 8 afterHeading + 3 pseudocode skip）；锚点唯一命中 ✓；PULSE_GRAD_ID 页面级 ✓；reduced-motion 静态终态 ✓
- 内容 fidelity：22 图数字/shape/机制措辞逐章核对全对（重点：MLA seesaw WG0 p0=Q·K0ᵀ/WG1 p1=Q·K1ᵀ + K 9 TMA/V 8×64 同源视图 + 跨 WG O_R+=p0·V0R/O_L+=p1·V1L + rescale(s0)/(s0·s1)；m16n8k16 几何与 quad/lane-4 行布局同构；FA tile 预算 64/227 对 252/255→T_M=128；FA2 tile 流水 ③④⑤ + rescale 反馈 + sVt 视图）；仅 2 处措辞微瑕（§4.2/4.4）不涉数字。

## 6. Redraw closure addendum（2026-11-25，T7/T8 闭环）

- **R1/R2 已修复，22/22 过线**：commit **e423b50**（fix(gpu): W3 gpu-p05 fig-0/2 卡高/层距/短边 pulse 落点）按 §3 建议模块内修——fig-0 reductionTree 层级卡 h 32→44（sub 基线 y+38 距卡底 y+44 留 6px 不骑边框）、层距 38→56（卡隙 12px）、卡间短竖边去 pulse 改箭头 + 右侧侧 bus pathPulse、zone 高 182→246；fig-2 blellochScan up-sweep 内边改 top→bottom 走向并去 pulse（3.5s 光点不再遮 10/26/36）、leaf→up0 脉冲 dur2.4/delay2.3 使光点落叶-节点间隙。fig-check gpu-p05=10 / gpu-p06=12 全渲染，Read fig-0/fig-2 复核 sub 离框、光点不遮字、无溢出/重叠。**判定翻为 22 pass / 0 redraw**。
- **T7 内容 fidelity 随文订正（不改 layout，仅措辞/数字与正文 S2/S3 修复对齐）**：gpu-p05 fig-0 底注「离 ridge 295 差约 1180 倍」→「离 fp32 ridge ≈20 差约 80 倍（295 是 BF16 TC ridge）」（随 5.1 S2-1 ridge 口径）；gpu-p05 fig-1 warpShuffle 注改「表列 shuffle 源值语义…v += shfl 下高位 lane 实持翻倍旧值，仅 lane0 结果有效」（随 5.1 S3-1）；gpu-p06 fig-0 formatBitLayout「误用 IEEE 闭式得 480」→「误用 NONE 闭式得 480…IEEE 式给 240」（随 6.1 E4M3 三约定订正）；gpu-p04 fig（4.4）「完成被计算盖住」→「完成被 (S−1)τ_S 盖住（带宽主导）」（W2 残留措辞）。
- 非阻塞观察（§4）维持延迟站点 polish 轮，不入本波 commit gate。
