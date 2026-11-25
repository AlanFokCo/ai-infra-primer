# gpu W5 场景图 QUALITY 评审档案（gpu-p09 / gpu-p10，18 图）

- 评审日期：2026-11-25；评审角色：fresh design-eye evaluator（只读，未改模块；未参与实现）
- 对象：commit **aa84fcf** 的 `site/scripts/lib/figures/gpu-p09.mjs`（9 图）/ `gpu-p10.mjs`（9 图）
- 规约：`site/scripts/lib/figures/QUALITY.md`（五维各 ≥4 过线、截图证据）+ `STYLE.md`；先例 `docs/reviews/gpu-w4-fig-quality.md`
- 证据：本轮重跑 `cd site && node scripts/fig-check.mjs gpu-p09` / `gpu-p10`（18/18 渲染），逐张 Read `site/qa/figs-gpu-p09/fig-0..8.png`、`site/qa/figs-gpu-p10/fig-0..8.png`（3.5s 截图时刻）；reduced 终态独立自跑 `/tmp/figqa-w5-review-reduced.mjs`（口径同 fig-reduced-diff：max 通道差 >16 记变色，t3000 vs t4500）→ **18/18 diffpx=0**，reduced 上下文 `.dia-pulse` computed display=none ✓；census/锚点/defs 用 `/usr/bin/grep` 逐章核验
- 判定：**17 pass / 1 redraw**（redraw = gpu-p09 fig-2 qpStateMachine，内容咬合=2）
- 打分维度（任务口径）：① 内容咬合 ② 视觉清晰（含 QUALITY.md 可读性/层级/细节）③ 一致性（palette/字体/spacing/helper 先例 + 页面级 PULSE_GRAD_ID）④ 动效/reduced ⑤ 可达/稳健（锚点唯一/census/aria/无孤儿）。与 W4 档案五维（动效/可读/层级/细节/咬合）的交叉映射：①=咬合；②=min(可读,细节) 且层级证据并入；④=动效自然度+reduced 终态；③⑤ 为 W4 §0 全局纪律的逐图化。

## 0. census 与全局纪律核验

- 注册表：p09 = 7 replacesBlock + 2 afterHeading；p10 = 6 replacesBlock + 3 afterHeading → **13 + 5 = 18** ✓；18/18 显式 `dim:'gpu'` ✓
- 逐章无语言围栏块（grep `^``` ` 配对核验）：9.1 b0(L29)/b1(L116)（L66 ```c 伪代码跳过）；9.2 b0(L27)；9.3 b0(L23)/b1(L61)/b2(L81)/b3(L140)；10.1 b0(L25)/b1(L92)；10.2 b0(L45)；10.3 b0(L69)（L57 ```c 跳过）；10.4 b0(L125)/b1(L183)（L47/L58 两 ```c 跳过）→ **13 块全替换、每块恰一次、4 伪代码跳过** ✓ 无漏块无孤儿
- afterHeading 唯一命中：9.1 `^## 3\. `、9.2 `^## 4\. `、10.2 `^## 4\. `、10.3 `^## 5\. ` / `^## 7\. ` 各 1 hit ✓
- 脉冲渐变：两模块 import `PULSE_GRAD_ID`（fig-pulse-grad），pathPulse 与 kit edge 全引用页面级 id；grep 两模块 `<defs` = 0、`sig-` = 0（无 per-scene defs、无入场动画）✓；cap 提权 wrapper / pathPulse / arrowHead 三 helper 与 gpu-p07/p08 逐字同构 ✓ 一致性先例成立
- 截图时刻语义：p09 最大 delay 1.8s、p10 最大 1.4s（i*0.3+0.5, i=3），均 < 3.5s ✓ 截图光点分布与相位推算一致
- reduced 终态：独立自跑 18/18 diffpx=0（实现者报告同结论，本轮复核成立）✓
- **关键时序事实**：章订正 commit 23b4fb7（9.1 §5 DeepEP per-peer QP 池 O(n²) + UD STATE 订正）是图 commit aa84fcf 的**父提交**——图提交时订正已在章内，但 fig-2 仍承订正前旧文（见 §3）。并行内容评审 docs/reviews/gpu-w5-review-p9.md 的 S2-1/S3-1 即该订正来源；其结论与本轮图侧发现一致。

## 1. gpu-p09（9 图）

| fig | 函数（锚） | 咬合 | 清晰 | 一致 | 动效 | 稳健 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | tcpVsRdma（9.1 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 1 | verbsFourObjects（9.1 §3） | 5 | 5 | 5 | 4 | 5 | pass |
| 2 | qpStateMachine（9.1 b1） | **2** | 4 | 5 | 4 | 4 | **REDRAW** |
| 3 | gdrVsBounce（9.2 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 4 | gdrMinRegime（9.2 §4） | 5 | 5 | 5 | 4 | 5 | pass |
| 5 | congestionChain（9.3 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 6 | pfcPause（9.3 b1） | 5 | 5 | 5 | 4 | 5 | pass |
| 7 | dcqcnLoop（9.3 b2） | 5 | 5 | 5 | 4 | 5 | pass |
| 8 | ibtaCC（9.3 b3） | 5 | 5 | 5 | 4 | 5 | pass |

### fig-0 tcpVsRdma — 5/5/5/4/5
- 咬合 5：Δ_TCP=2t_copy+2t_ctx+t_stack=3.82 μs vs Δ_RDMA=0.2 μs（19.1×、省 3.62）、1 GB 拷贝项 200 ms 主导、CPU offload≠完全不参与坑注，逐数对 §2/命题 1。
- 清晰 5：TCP 灰 zone / RDMA seal zone 上下对照，copy1/协议栈/DMA 标签各居边上，接收侧对称行补全；账条值标签随条尾不溢。
- 一致 5：角色染/zone/seal 收束行与 p07/p08 同口径。动效 4：三跳阶梯 0/0.4/0.8 + RDMA 1.0/1.4 接力，截图光点分居各段。稳健 5：aria-label 数值与图一致。

### fig-1 verbsFourObjects — 5/5/5/4/5
- 咬合 5：verbs.h L2692/L3246/L2962/L3662/L3040 五锚逐字；rkey 带外交换、未注册保护错误、与 8.4 proxy 分层同构全对 §3。
- 清晰 5：慢/快双 zone，MR→WR 水平 lkey 通道、QP→NIC 下弯通道、CQ→CQE 底部通道互不交叉（实现期 fix#1 复核成立）；隙内标签不 graze 卡缘。
- 动效 4：doorbell 竖脉冲 0.2 + NIC→CQE 0.6，控制面虚线静态分工正确。

### fig-2 qpStateMachine — 2/4/5/4/4 → REDRAW
- 咬合 2：**三处与订正后章文矛盾**（详见 §3）：① 「QP 池 O(n)／建连成本压到 O(n)」+ fig-tag + aria-label 承订正前旧论，章 §5 现文明确「不是把连接数压到 O(n)」——per-peer 池、全网 24·n(n−1) 仍 O(n²)（n=1024 ≈2.51×10⁷）；对照条把池画成比全互连小 ~43×，方向反了（实为 24× 更多 QP）。② 「UD：仅改 STATE」为订正前旧文（现文：INIT 设 pkey/port/qkey、RTR 仅 STATE、RTS 设 SQ_PSN）。③ 每态属性列整体左移一态：RESET 下挂「设 port/pkey/access flags」（章归 INIT）、INIT 下挂远端 QPN/PSN/GID/LID/PATH_MTU（章归 RTR）、RTR 下挂本地 PSN/TIMEOUT/RETRY_CNT/RNR_RETRY（章归 RTS）——与卡内 sub（INIT=本地属性）自相矛盾，截图实证。
- 清晰 4：构图/对齐/账条本身干净可读，但属性错挂使不读正文的读者学反状态语义。一致 5、动效 4（状态链 0/0.4/0.8 + 开销链 1.2/1.6 阶梯）。稳健 4：aria-label 与 fig-tag 同步携带错误标度论断（机器可读面同污染）。
- 重画要求见 §3；其余构图/配色/状态链保留，局部重画即可。

### fig-3 gdrVsBounce — 5/5/5/4/5
- 咬合 5：28.07 半谐波 (1/64+1/50)⁻¹（1.78×）、19.51 三跳（2.56×）、流水 min(64,50)=50 持平、host 4M→0、staging 15.625/31.25 ms、gdrcopy 0.10 vs cudaMemcpy 6–7 μs，逐数对 §6 式 (3)(4)(5)。
- 清晰 5：bounce 三跳 vs GDR 单跳同端起终点对照；hop 标签移边上、GDR 双标签分置边上下（fix#2 复核成立）；三条账条色阶 soft/warm/seal 分明。

### fig-4 gdrMinRegime — 5/5/5/4/5（本波关键 regime 图）
- 咬合 5：式 1 四行表逐行（50/64→50 NIC 限；双口 2×NDR200 不翻倍；100/64→64 PCIe 限；100/128→100 NIC 限 + Gen6 代际注）；cut-through 不另计跳 / store-and-forward 才半谐波；HBM 3350/50=67×；「400G 判 PCIe 瓶颈」坑 seal 行——与 §4 表/裁定逐字一致。
- 清晰 5：min 三卡链 + 表 + vs HBM 条三层；67× 值标签移条上方（fix#3 复核成立）不压条尾。

### fig-5 congestionChain — 5/5/5/4/5
- 咬合 5：三分支 α_eff=α₀+t_queue / HoL-victim / β_eff=1/BW_eff → 带宽项 2(n-1)/n·M·β_eff（8.2 式 2）→ max；straggler n=16 每 rank 1.875 GB、37.5→75 ms 翻倍；两层兜底 μs/ms 分工，全对 §2/§7。
- 清晰 5：汇聚箭头入卡顶、α/β 注记上移后与汇聚起点分离（fix#4 复核成立）；光点沿三分支+汇聚阶梯分布。

### fig-6 pfcPause — 5/5/5/4/5
- 咬合 5：2:1 incast A=100/C=50、Q_th=512 KB→10.24 μs、headroom ≥ C·t_react+MTU=104 KB（t_react=2 μs）、storm=无让路或成环、三代价与逐类结构性缺陷，全对 §3 命题 1/2。
- 清晰 5：pause 回传 L 形通道入 src 底边、入口分支竖线、victim 虚线挂回传通道各归其位；注记行不与竖线相交。

### fig-7 dcqcnLoop — 5/5/5/4/5
- 咬合 5：RP/CP/NP + Kmin/Kmax/Pmax、T_CNP=50 μs、T_w=55 μs、F=5、无 slow start、α 译 w 注；走读 50→25→12.5 后 fast recovery 向 RT=25：18.75→21.875→23.4375→24.21875→24.609；命题 3 闭式代入 RC₀=25/RT=50 k=5→49.21875（1.56%、k≥6）——**两条走读口径分 zone 标注不混** ✓ 全对 §4。
- 清晰 5：CP 两行注记避开 x126/x594 CNP 竖线（fix#5 复核成立）；CNP 回传强线入 RP 底边。

### fig-8 ibtaCC — 5/5/5/4/5
- 咬合 5：①②③ 三阶段、FECN 随包向前 / BECN piggyback ACK·CN 包、CCT/CCTI/CCM、per-port per-VL 0-15、t_tx=81.92 ns、IPD=t_tx→25 GB/s（占空比条 pkt/IPD 交替示意），全对 §5 式 (8)。
- 清晰 5：时域/频域对照行收尾；双 zone 账目字级阶梯清晰。

## 2. gpu-p10（9 图）

| fig | 函数（锚） | 咬合 | 清晰 | 一致 | 动效 | 稳健 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | fatTreeClos（10.1 b0） | 5 | 4 | 5 | 4 | 5 | pass |
| 1 | railOptimized（10.1 b1） | 5 | 5 | 5 | 4 | 5 | pass |
| 2 | hierRing（10.2 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 3 | sharpAggregation（10.2 §4） | 5 | 4 | 5 | 4 | 5 | pass |
| 4 | deepepNormal（10.3 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 5 | deepepLLHook（10.3 §5） | 5 | 4 | 5 | 4 | 5 | pass |
| 6 | a2aIncast（10.3 §7） | 5 | 5 | 5 | 4 | 5 | pass |
| 7 | ringAttentionOverlap（10.4 b0） | 5 | 5 | 5 | 4 | 5 | pass |
| 8 | ringPhysicalMapping（10.4 b1） | 5 | 5 | 5 | 4 | 5 | pass |

### fig-0 fatTreeClos — 5/4/5/4/5
- 咬合 5：k²/2：144→10368（XDR 逐字闭合）/64→2048；三级 k³/4=65536；pattern k^n/2^(n−1)；全二分 iff k_up≥k_down（König）；1:1 可重排 vs 严格 2k_down−1；3:1 48/16→3072(+50%) 二分 ÷3，全对 §3。
- 清晰 4：二部链接只画 5 条子集（leaf_1→三 spine 全、leaf_2/leaf_k 各一条）+「…」省略，靠「每台 spine 接全部 leaf／每 leaf 上行接互异 spine」注记补偿；leaf_1→spine_{k/2} 长 curve 与 leaf_2→spine_2 curve 有一处固有交叉、spine 底端口强/弱线共点收敛——拓扑固有、不压字不穿卡，判注记级（沿 W4 a2a 子集先例）。
- 动效 4：单高亮上行脉冲点题，其余静态灰弱。

### fig-1 railOptimized — 5/5/5/4/5
- 咬合 5：同号 1 跳 / 跨 rail 3 跳 / PXN 域内+1 跳仍 1 跳；式 6/7：8×50=400、vs 3600=9×（单向注入）、37.5→4.6875 ms 恰好 ÷8（n=16 节点粒度环）；三层一致（接线/PCIe/channel），全对 §4/§7。
- 清晰 5：4 rail 横排 GPU—NIC—leaf—NIC—GPU，node zone 双翼、leaf 居中；脉冲 i×0.3 阶梯截图四点错位=四 rail 并行直接表达。

### fig-2 hierRing — 5/5/5/4/5
- 咬合 5：三相（NVLink RS → 8 rail IB AR(M/8) → NVLink AG）；每 NIC 2(k−1)/k·M/8=218.75 MB vs 平环 1.96875 GB=9.0×；T_hier=4.72+4.375+0.224=9.32 ms vs 39.375=4.2×；口径注 NCCL 多 channel 平环 ≈0.246 GB 同量级、k=1 退化，全对 §3 命题 1。
- 清晰 5：三 zone 相位递进 + 强箭头跨隙；流量/时间双账条色阶分明。

### fig-3 sharpAggregation — 5/4/5/4/5
- 咬合 5：上传每 rank 自有 M 一次/交换机内归约/虚线广播 M；n=64 出口 20 vs 39.375 ms（1.97，→2 上界披露非保证）；万卡 286.7 ms vs 392 μs=731×；NVLS 域内对照 + NVLS_TREE×CollNet；SAT 16 KB/OST，全对 §4。
- 清晰 4：r0 顶上传脉冲点与广播虚线箭头共点、r1 顶虚/实线近接——双向树边共端口固有，可辨但略挤；286.7 ms 灰字移条上方（fix#6 复核成立）对比足。
- 动效 4：上传六脉冲 0→0.8 阶梯沿树上行，广播虚线静态分工正确。

### fig-4 deepepNormal — 5/5/5/4/5
- 咬合 5：v1.2.1 锚；translate_dst_rdma_rank normal=节点号/LL=GPU；WarpRole 五角色 L370-376、sender warps 7 L1008、「RDMA consumers and NVL producers」L690-691；去重 242.2 vs 109.0 MB=2.22×、D_card/D_node=7.6/3.6=2.11×；README EP8 153/158、EP64 51/50 趋线速；V2.5 hybrid 后继注，全对 §4。
- 清晰 5：两阶段语义注记改全宽两行置 zone 下方（fix#7 复核成立）不出框不穿 zone 边；WarpRole 五卡 646px 行宽入框、7.5px mono 标题各卡内不溢。

### fig-5 deepepLLHook — 5/4/5/4/5
- 咬合 5：SEND 只 issue / hook() 触发 RECV、接收侧零 SM（V1 范围注）；IBGDA L63/L339/L495 + 预注册 24 QP + #173 节点内 NVLink（「纯 RDMA」指跨机）；ΔT=α_NVL+mβ_NVL：2.19/8.25 μs；**V2.5 硬注独立 seal zone**：V1 全移除/EPBuffer hybrid-direct/Gin GDAKI/NVSHMEM 不再依赖/EP 占 SM（zero-SM 不支持）/IBGDA 仅注释/QP 解析 direct min(sms,9)·hybrid sms×16+1——与 §6 对照表逐行一致 ✓
- 清晰 4：时序轴尾线伸出 RECV 条右 ~60px 略赘（时间延续读法可接受）；其余三相条/刻度/注记干净。
- 动效 4：后台搬段单脉冲 0.4s，SEND/RECV 静态条分工正确。

### fig-6 a2aIncast — 5/5/5/4/5
- 咬合 5：均衡入口 (n−1)·C/(n−1)=C vs 偏斜 σC、超额 (σ−1)C、拥塞 iff σ>1 且持续；σ=1.09/2/4 → 54.5/100/200、4.5/50/150、填 64 MB 14.2/1.28/0.427 ms；EPLB moe_lb 1.090→91.7% + ECN/PFC/AR 两手；ring 无 incast 对照；DCQCN 16:1/32:1，全对 §7 命题 3。
- 清晰 5：双 zone 三 sender 汇聚对照，σ 表四列 + 填队列条色阶（soft→seal）点睛。

### fig-7 ringAttentionOverlap — 5/5/5/4/5
- 咬合 5：comm 234 宽被 compute 240 宽覆盖（0.977）；式 4/5 阈值 8000/1079 token 与比值 0.977/0.066、T_blk=4096→1.95 暴露；式 6 t_ring 87.96=16×5.498 vs 串行 168.49 省 47.8%；WAR 竞态 + group 死锁 + 全 mask 步不能跳，全对 §4。
- 清晰 5：comm 脉冲移卡下 y162 通道（fix#8 复核成立）不压标题字带；依赖虚线走 16px 卡隙入 compute1 左缘；总时间条 ×2.6 scale 后值标签不压条。
- 注：阈值表头「T_blk=8192 读法」与 n=8 行比值 0.066（实对应 T_blk=16384）的口径松驰**逐字承自章 10.4 §6 表**——图忠实镜像章表，不判图罚，登记章侧备查（§5）。

### fig-8 ringPhysicalMapping — 5/5/5/4/5
- 咬合 5：朴素 r7→r8/r15→r0 跨 rail 经 spine vs rail 对齐反转 node 1（r7→r15 rail 7、r8→r0 rail 0）同 rail 1 跳；步拍=最慢边 5.369 ms、域内 14 跳只决 slack；阈值 1079/8000/6250 与 0.066/0.977/0.763；raw 9× vs 模型 7.4×（8000/1079≈7.4）；causal zigzag 充分条件 16000，全对 §6。
- 清晰 5：rail 7 标签居 40px 卡隙不 graze 双卡缘（fix#9 复核成立）；回环通道注记居横线上方 6px；双 zone 虚/强对照一眼成立。

## 3. redraw findings（任一维 <4）

1. **gpu-p09 fig-2 qpStateMachine（9.1 b1）内容咬合=2** —— 图承订正前旧章文，与父提交 23b4fb7 订正后的 §5 直接矛盾（实现者报告 §3 仍把「O(n²)→O(n)」列为已核验，属误核验）。三处必修：
   - `site/scripts/lib/figures/gpu-p09.mjs` **L183-189**（+ **L193** aria-label、**L560** fig-tag）：「DeepEP 预建固定 QP 池复用…承载所有 token 路由」「QP 池 O(n)」「24 QP/rank —— 建连成本压到 O(n)」→ 改为 per-peer 池口径：`num_qps_per_rank=24` 是**每对端** 24 条（NVSHMEM_IBGDA_NUM_RC_PER_PE，buffer.py L101；按 dst_expert_local_idx 选 QP），全网 24·n(n−1) **仍 O(n²)**（n=1024 ≈2.51×10⁷），目的 per-expert 流量 spreading 非省连接；对照条若保留须反向（池=24× 全互连单 QP 口径）或改注记行「长驻池摊销建连、免逐消息建连」；fig-tag/aria 同步删「压到 O(n)」。
   - **L169**：「UD：仅改 STATE（无连接，不需对端元数据）」→「UD：INIT 设 pkey/port/qkey、RTR 仅 STATE、RTS 设 SQ_PSN（无对端元数据）」。
   - **L161-168**：每态属性列右移一态对齐章 §5 表：INIT=设 port/pkey/access flags；RTR=设远端 QPN/PSN/GID/LID/PATH_MTU；RTS=设本地 PSN/TIMEOUT/RETRY_CNT/RNR_RETRY/MAX_QP_RD_ATOMIC；RESET 列改「创建态」或删（卡 sub 已有）。
   - 其余维 ≥4：状态链/开销链构图、配色、脉冲保留，局部重画即可。复审通道：重跑 `node scripts/fig-check.mjs gpu-p09` + fig-2 截图复核三处 + reduced 单帧。

## 4. 内容咬合专项核验（任务清单）

- **GDR min-regime**：p09 fig-4 四行逐字（CX7 单口/双口 NIC 线速限、CX8+Gen5 PCIe 限、CX8+Gen6 回 NIC 限）+ 67× HBM + cut-through/store-and-forward 区分 + 坑 seal ✓；p09 fig-3 串行/端到端/流水三口径分离 ✓。无「400G=PCIe 瓶颈」误判。
- **NVLink bidir vs uni**：p10 fig-1 3600=8×450 单向注入、9× ✓；fig-2/fig-7/fig-8 370.8 模型单向 ✓；~160 GB/s 为 H800 README 口径（章标注）✓。两模块无双向头条当单向代入处。
- **ring 总量 vs 单阶段**：p10 fig-2 每 NIC 2(k−1)/k·M/8 vs 平环 2(n−1)/n·M（9.0×）+ T_hier 三相加和 ✓；p09 fig-5 带宽项 2(n-1)/n·M·β_eff + straggler max ✓；p10 fig-3 SHARP 每 rank 发送 M（→2 上界披露）与 ring 2(n−1)/n·M 对照 ✓。无 2× 混用。
- **DCQCN 速率走读**：p09 fig-7 首砍减半 50→25→12.5、fast recovery 五步向 RT=25、命题 3 代入 RC₀=25/RT=50 另标（49.21875/1.56%/k≥6）✓ 两口径不混；T_CNP/T_w/F/无 slow start/α 译 w 全对。
- **fat-tree 10368**：p10 fig-0 k²/2 逐字闭合 + 2048/65536/pattern + König + 可重排 vs 严格 + 3:1 权衡 ✓。
- **DeepEP V1 vs V2.5**：p10 fig-4 v1.2.1 锚 + hybrid 后继注 ✓；fig-5 V2.5 硬注 seal zone 六项逐行对 §6 ✓（V1 hook≠当前接口明示）。**唯 p09 fig-2 QP 池标度论断为订正前残留 → redraw**（§3）。
- **incast σC**：p10 fig-6 均衡 C / 偏斜 σC / iff σ>1 持续 / 三 σ 填队列数 / EPLB+ECN/PFC/AR 两手 ✓。
- **Ring overlap**：p10 fig-7 0.977 覆盖 + 阈值三行 + 87.96/168.49/47.8% + WAR/group/全 mask ✓；fig-8 rail 对齐反转 + 步拍最慢边 + 7.4× vs raw 9× + causal 16000 ✓。
- 其余抽查：p09 fig-1 verbs.h 五锚、fig-6 headroom 104 KB、fig-8 81.92 ns/25 GB/s；p10 fig-4 WarpRole/去重双比值/README 表、fig-5 IBGDA 三锚/QP 解析式 —— 逐字对章 ✓。

## 5. 注记级瑕疵（不判罚、登记备查）

- p10 fig-0：二部链接子集 + 固有交叉/共点收敛（注记补偿，沿 W4 先例）。
- p10 fig-3：r0/r1 顶上传脉冲点与广播箭头共点/近接（双向树边固有）。
- p10 fig-5：时序轴尾线伸出 RECV 条 ~60px。
- p10 fig-7/fig-8：阈值表头「T_blk=8192 读法」与 n=8 行 0.066（对应 T_blk=16384）口径松驰——**章 10.4 §6 表自身如此**，图逐字镜像；建议章侧表头改「章脊读法（n=8 行按 T_blk=16384）」，图随章同步，本轮不判图罚。
- p09 fig-2 之外无压字/穿卡/出框；实现期 9 项 layout fix 本轮截图复核全部成立。

## 6. 汇总

- **18 图：17 pass / 1 redraw**。redraw = gpu-p09 fig-2（咬合 2）。
- 最低分维：gpu-p09 fig-2 咬合 2（唯一 <4）；pass 图最低为 4（p10 fig-0/fig-3/fig-5 清晰 4；全图动效 4 沿 W3/W4 已审 spline/paced 口径）。
- 全局纪律：census 13+5+4 skip ✓、锚点唯一 ✓、dim gpu 18/18 ✓、页面级 PULSE_GRAD_ID ✓、helper 先例同构 ✓、reduced 终态 18/18 diffpx=0（独立自跑）✓、3.5s 语义完整 ✓、无孤儿图 ✓。
- 复审通道：fig-2 按 §3 三处局部重画后重跑 `node scripts/fig-check.mjs gpu-p09` + 截图/reduced 复核即可闭环；其余 17 图无需动。

## 7. redraw closure addendum（2026-11-25，scoped re-check of commit 1ea8f76）

- 范围：仅复核 gpu-p09 fig-2 qpStateMachine（commit 1ea8f76，1 file / +29−24，只改 fig-2 函数体 + 注册标题）；其余 17 图维持 §1/§2 原判，未重审。
- 证据：重跑 `node scripts/fig-check.mjs gpu-p09` → 9/9；Read `site/qa/figs-gpu-p09/fig-2.png` 目视；独立 reduced 自跑（/tmp/figqa-w5-review-reduced.mjs）p09 9/9 diffpx=0（含 fig-2）、`.dia-pulse` display=none；逐条对章 9.1 §5 L124/L126 + 练习 2(d) L176 + ASCII b1。

### Fig-2 Verdict
1. **QP 标度 per-peer O(n²) — ADDRESSED**：gpu-p09.mjs L186-196 订正 seal 行「24 是每对端 QP 池（NVSHMEM_IBGDA_NUM_RC_PER_PE），非全网总数」+「24 = 本地 expert 数、按 dst_expert_local_idx 选 QP … per-expert spreading」+ 坑行「把 24 读成全网总数 / O(n)」；对照条反向修正：全互连 1/对端 1.05×10⁶ 短灰条 vs per-peer 池 24/对端 长 seal 条（25.1×17.5 px ≈ 24×）、值标签「24·n(n−1) ≈ 2.51×10⁷ · 仍 O(n²)」置条上方不压条（截图实证）；fig-tag L565 与 aria-label L198 同步改「仍 O(n²)」。与章 L126/练习 2(d) 逐数逐义一致。
2. **UD 状态属性 — ADDRESSED**：L171-172「UD 无连接：不需对端元数据」+「UD 迁移表（man）：INIT 设 pkey_index/port/qkey、RTR 仅 STATE、RTS 设 SQ_PSN」，与章 L124 订正后逐字一致；旧「仅改 STATE」已删。
3. **每态属性列对齐 — ADDRESSED**：L162-170 列右移归位：RESET=不设属性（创建态）、INIT=设 port/pkey/access flags（本地）、RTR=设远端 QPN/PSN/GID/LID/PATH_MTU、RTS=设本地 PSN/TIMEOUT/RETRY_CNT/RNR_RETRY/MAX_QP_RD_ATOMIC（三行，y122/136/150，与下行 y170 留 20px）；与卡内 sub（INIT=本地属性）自洽、与章 §5 每态表/ASCII b1 归属一致（截图实证）。
- 布局复核：RTS 三行列、双 UD 行、开销账下移（y212/226）、池条值标签与上行值标签 x 向分离（≈300 vs ≈457 起）均无压字/穿卡/出框；viewBox 720×428 内全文字 ✓；脉冲阶梯保留（0/0.4/0.8 + 1.2/1.6）。
- fig-2 复评分（任务五维）：咬合 5 / 清晰 5 / 一致 5 / 动效 4 / 稳健 5 → 最低 4（动效沿 W3/W4 已审口径）。

### Gate: PASS（18/18 各维 ≥4）
- §1 fig-2 原打分（咬合 2）与 §3 redraw findings 保留为重画前原判（历史记录）；重画后三处闭环，17 pass + 1 redraw → **18/18 QUALITY pass**。
- 注记级备查（§5）不变：p10 fig-7/fig-8 阈值表头口径松驰承自章 10.4 §6，建议章侧修，图随章同步。
