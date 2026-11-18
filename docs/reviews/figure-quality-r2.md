# 图品质量规 R2 评审：training 50 张评分 + models 修复回归复验

量规：`site/scripts/lib/figures/QUALITY.md` 五维 1–5 分制，全维 ≥4 过线。本轮只评审、不改图代码；新增评审工具 `site/scripts/fig-review-training.mjs`（与 fig-review.mjs 同口径，过滤 dim='training'，内联 reduced 两帧像素差与降级 DOM 核验）。

## 评审方法与证据链（全部实拍，无源码目测）

1. **training 注册表枚举**：`figures/training-part01..09.mjs` 共 **50 张**（全部 `dim:'training'` + `replacesBlock`）；逐键核对章内裸围栏块序号，**50/50 挂载目标存在**。
2. **逐张渲染实拍**：真实 `tokens.css + figure.css + PULSE_GRAD_DEFS`、`.fig.in` 常亮，Playwright（channel: chrome, dsf=2）逐张四帧：正常 t=0.6s / t=6s，reduced-motion t=3s / t=4.5s → `site/qa/fig-review-training/tNNN-*.png`（50×4 = 200 帧 + index.html/index.json/reduced-diff.json）。疑似缺陷另以 dsf=4 局部放大复查（`t032-zoom-core.png`、`t036-zoom-right.png`）。
3. **动效度量**：正常模式早晚帧像素差证运动存在（45/50 有运动；5 张 0px 为调度网格/括号格/掩码表等声明性内容，入场后静止合宜，同基线口径不扣分）；reduced 两帧像素差测降级（阈值 >200px 判仍在动）；渲染态 `getBBox` 逐 text 对 viewBox 越界检查。
4. **models 回归**：`node scripts/fig-review.mjs` 全量重渲 136 张（修复后工作树）→ `qa/fig-review/`；`node scripts/fig-reduced-diff.mjs` 重测降级与 pulse 配速探针；基线数值取 `qa/fig-review-r1/*-baseline.json` 与基线报告记录值。基线截图目录现状说明：`qa/fig-review/` 在修复落地后已被修复方重跑一次（基线 JSON 以 `-baseline` 后缀保留在 `qa/fig-review-r1/`），本轮将其整体改名为 `qa/fig-review-r1/` 留存，R2 证据为评审人独立重跑的全新渲染。

---

## Part 1：training 50 张评分表

维度序：动效自然度 / 场景可读性 / 视觉层级 / 细节完成度 / 正文咬合。总评 = min（过线口径）；证据列 = 实拍截图（稳态帧 `tNNN-norm-t6000.png`，均在 `site/qa/fig-review-training/`）+ 度量。

**系统性度量结论（先行）**：
- reduced-motion 降级：50/50 合格（>200px 仍动 = 0 张；残留 #10 @3px、#27 @157px 均低于阈值且两帧目视无差）；DOM 核验 `sig-dashflow` computed animation-name = none、`.dia-pulse` display = none（training 无 sig-rotor）。
- 脉冲配速：kit `edge(pulse)` 与各模块 `pathPulse` 均为 `calcMode="spline" keySplines="0.42 0 0.58 1"` 缓动（探针实测 t=10% 进度 0.026、 t=50% 进度 0.546，缓入缓出生效），循环有呼吸感、单向不破坏数据流方向。
- 文本出界：渲染态 getBBox 命中 1 张（#36 区标签 6px，放大复查 glyph 完整渲染、记注不扣分）。
- 纪律：9/9 模块脉冲渐变引用页面级 `PULSE_GRAD_ID`，无 per-scene defs 错配。

| # | 图（slug # block） | 标题 | 动 | 读 | 层 | 节 | 合 | 总评 | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 01-autograd…/01-computation-graph-and-autograd #0 | 计算图双联：前向 × 反向 adjoint | 4 | 5 | 5 | 5 | 5 | **4** | t000；前向/反向双区、L→种子边收口，vjp 公式贴右缘 |
| 1 | 01-autograd…/02-operator-backward #0 | matmul 反向 = 2× 前向 | 4 | 5 | 4 | 5 | 5 | **4** | t001；反向右→左流向合 adjoint 语义，三处 2mnk 徽标对账 |
| 2 | 01-autograd…/02-operator-backward #1 | fused / chunked CE | 4 | 5 | 5 | 4 | 5 | **4** | t002；片上 buffer 虚线区清楚；H 节点 sub 宽于卡（未出 viewBox） |
| 3 | 01-autograd…/03-mixed-precision #0 | fp16 混合精度环路 | 4 | 5 | 5 | 5 | 5 | **4** | t003；fp32/fp16 双区+回环，S 窗口式 (4) 与 GradScaler 口径在 footer |
| 4 | 01-autograd…/04-gradient-checkpointing #0 | 分段 checkpoint | 4 | 5 | 5 | 5 | 5 | **4** | t004；边界驻留条 + 重算段虚线牵引，M(k) 式 2/3 footer |
| 5 | 01-autograd…/05-gradient-accumulation #0 | no_sync 压 all-reduce | 4 | 5 | 5 | 5 | 5 | **4** | t005；naive/no_sync 双行对照，hook 关 ghost 芯片，账 209→26 GiB |
| 6 | 02-optimizers/01-sgd-and-momentum #0 | momentum 两种写法 | 4 | 5 | 4 | 4 | 5 | **4** | t006；换算桥 m=(1−β)·v 居中；节点 sub 宽出卡缘 |
| 7 | 02-optimizers/01-sgd-and-momentum #1 | 二次碗 zigzag vs heavy ball | 4 | 5 | 5 | 5 | 5 | **4** | t007；轨迹为式 (1)(17) 最优调参真实模拟，右侧速率账与步数账 |
| 8 | 02-optimizers/02-adam-and-adamw #0 | Adam 双通道 + λθ 支路 | 4 | 5 | 4 | 4 | 5 | **4** | t008；AdamW chip sub 宽出卡缘；torch 分岔一行 footer 点题 |
| 9 | 02-optimizers/03-lr-schedules #0 | WSD vs cosine | 4 | 5 | 4 | 4 | 5 | **4** | t009；「cosine（T = S）」标签压灰曲线（基线 #12 同类口径记 4） |
| 10 | 02-optimizers/04-gradient-clipping-and-stability #0 | 稳定原语 placement 七站 | 4 | 5 | 5 | 5 | 5 | **4** | t010；七站截面+监控总线双联；reduced 残留 3px 忽略 |
| 11 | 02-optimizers/05-muon-and-spectral #0 | Newton-Schulz 5 步 | 4 | 5 | 4 | 4 | 5 | **4** | t011；「×T=5·bf16」压区界、σ 轴「1」标签贴虚线框 |
| 12 | 03-loss-functions/01-ce-family #0 | CE 账流水线 | 4 | 5 | 5 | 4 | 5 | **4** | t012；纵链 + 右侧两笔 V 维账；mask chip sub 宽于卡 |
| 13 | 03-loss-functions/02-preference-and-contrastive-objectives #0 | DPO 数据流 | 4 | 5 | 4 | 4 | 5 | **4** | t013；fused 区标签与 π_θ→logp_θ 入边相交（轻） |
| 14 | 04-data-parallelism/01-dp-and-ddp #0 | ring all-reduce | 4 | 5 | 5 | 5 | 5 | **4** | t014；环 + scatter-reduce/all-gather 双网格，星型基线对照账 |
| 15 | 04-data-parallelism/01-dp-and-ddp #1 | DDP bucket overlap | 4 | 5 | 5 | 5 | 5 | **4** | t015；GPU/bucket/通信三泳道，B0 首桶缩小注记精确 |
| 16 | 04-data-parallelism/02-zero #0 | DP vs ZeRO-1 通信恒等 | 4 | 5 | 5 | 5 | 5 | **4** | t016；双区同构对照，2N = N+N 恒等 footer 含论文 §7.2.1 原句 |
| 17 | 04-data-parallelism/02-zero #1 | ZeRO-3 逐层节奏 | 4 | 5 | 5 | 5 | 5 | **4** | t017；正反双链 + 常驻显存条 + RS 红条归 owner |
| 18 | 04-data-parallelism/03-fsdp #0 | FSDP 单 unit 生命周期 | 4 | 5 | 5 | 5 | 5 | **4** | t018；泳道 + 状态点 + 式 2 峰值账；入场后静止（声明性时序，合宜） |
| 19 | 04-data-parallelism/04-memory-and-comms-accounting #0 | DP/ZeRO/FSDP 决策树 | 4 | 4 | 4 | 5 | 5 | **4** | t019；树密但判定分级正确，405B 生产实例注记在位 |
| 20 | 05-model-parallelism/01-tensor-parallelism #0 | 列切 / 行切 | 4 | 5 | 5 | 5 | 5 | **4** | t020；双区矩阵象形 + 前向/反向通信结论条 |
| 21 | 05-model-parallelism/01-tensor-parallelism #1 | 一层 TP 全貌 | 4 | 5 | 4 | 4 | 5 | **4** | t021；残差长弧跨全图，弧顶近题注行（轻） |
| 22 | 05-model-parallelism/02-sequence-and-context-parallelism #0 | TP-only vs TP+SP | 4 | 5 | 5 | 5 | 5 | **4** | t022；逐字节恒等 footer + Table 2 激活账 |
| 23 | 05-model-parallelism/02-sequence-and-context-parallelism #1 | Ring Attention 轮转网格 | 4 | 5 | 5 | 5 | 5 | **4** | t023；4×4 网格 ✓/✗ 编码清楚；声明性静止合宜 |
| 24 | 05-model-parallelism/03-pipeline-parallelism #0 | GPipe | 4 | 5 | 5 | 5 | 5 | **4** | t024；调度网格 + 气泡注记 + makespan/气泡式；声明性静止合宜 |
| 25 | 05-model-parallelism/03-pipeline-parallelism #1 | 1F1B | 4 | 5 | 4 | 4 | 5 | **4** | t025；「B1 级联逐段错开一格」注记占掉刻度 9/11 |
| 26 | 05-model-parallelism/03-pipeline-parallelism #2 | interleaved 蛇形 | 4 | 4 | 4 | 5 | 5 | **4** | t026；c8→c5 回绕边语义需读 footer 方明 |
| 27 | 05-model-parallelism/03-pipeline-parallelism #3 | Zero Bubble B/W | 4 | 5 | 4 | 5 | 5 | **4** | t027；轮廓双联（梯形 vs 平行四边形填尾）清楚；reduced 残留 157px 亚阈值、两帧目视无差 |
| 28 | 05-model-parallelism/04-expert-parallelism #0 | EP 两次 all-to-all | 4 | 5 | 5 | 5 | 5 | **4** | t028；dispatch/combine 双区 + 批量不等条，尾延迟注记点题 |
| 29 | 05-model-parallelism/05-combining-strategies #0 | rank 编号几何 | 4 | 5 | 4 | 5 | 5 | **4** | t029；stride 括号格密但色/长编码有效；声明性静止合宜 |
| 30 | 06-megatron-and-stacks/01-megatron-architecture #1 | TransformerLayer TP 走读 | 4 | 5 | 4 | 4 | 5 | **4** | t030；双柱 + g①/g② 收口；列内 sub 略出卡缘 |
| 31 | 06-megatron-and-stacks/02-megatron-runtime #0 | 16 rank 三族组排布 | 4 | 5 | 5 | 5 | 5 | **4** | t031；TP 括号 / PP 弧 / DP 虚线各一位，order 可重排注记在位 |
| 32 | 06-megatron-and-stacks/03-modern-training-stacks #0 | Megatron 家族三层 | 4 | 4 | 4 | 3 | 5 | **3** | t032 + t032-zoom-core：**Core→Bridge 边斜穿中央节点「Megatron-Core」标题文本**（线自节点左缘入、标题字中段穿过、右上缘出往 Bridge） |
| 33 | 06-megatron-and-stacks/03-modern-training-stacks #1 | 训练栈 ↔ 推理栈成环 | 4 | 5 | 4 | 5 | 5 | **4** | t033；双向回环 + checkpoint 曲线；区标签与边注相邻未撞 |
| 34 | 07-pretraining/01-data-pipeline #0 | 数据管线五站漏斗 | 4 | 5 | 5 | 5 | 5 | **4** | t034；漏斗条随站递减 + FineWeb/GLM-4.5 实账 |
| 35 | 07-pretraining/02-recipes #0 | 四家 LR 曲线 | 4 | 5 | 4 | 4 | 5 | **4** | t035；「常数尾 500B」标签压红线尾（轻） |
| 36 | 07-pretraining/04-checkpointing-and-resumption #0 | 分布式 checkpoint 网格解耦 | 4 | 5 | 5 | 4 | 5 | **4** | t036 + t036-zoom-right：区标签度量出 viewBox 6px，glyph 完整渲染（基线 #38 同类口径，记注不扣分） |
| 37 | 07-pretraining/04-checkpointing-and-resumption #1 | async save 两段式 | 4 | 5 | 5 | 5 | 5 | **4** | t037；GPU/staging/fabric 三泳道，两个 future 注记精确 |
| 38 | 08-rl-math/01-policy-gradient #0 | LLM 即 MDP | 4 | 5 | 5 | 5 | 5 | **4** | t038；状态链 + 末端奖励落盒，五元组逐项落地 footer |
| 39 | 08-rl-math/02-actor-critic-and-gae #0 | actor-critic 联合 loss | 4 | 5 | 4 | 5 | 5 | **4** | t039；δ→Â→双 loss 分流，R̂ 边注与 L_V sub 互证 |
| 40 | 08-rl-math/03-ppo #0 | PPO clip 分段图像 | 4 | 5 | 5 | 5 | 5 | **4** | t040；A=±2 双面版镜像 + 梯度区注记，平台截收益语义到位 |
| 41 | 08-rl-math/03-ppo #1 | PPO 一轮 | 4 | 5 | 5 | 5 | 5 | **4** | t041；①②③ 链 + θ_old←θ 回环，冻结纪律 footer |
| 42 | 08-rl-math/04-grpo-and-variants #0 | PPO vs GRPO 组件对照 | 4 | 5 | 5 | 5 | 5 | **4** | t042；r_1..r_G 小条 + ghost 删除位，显存/KL/动机三行 footer |
| 43 | 08-rl-math/05-llm-rl-systems #0 | RL iteration 五步 | 4 | 5 | 5 | 5 | 5 | **4** | t043；五步链 + 回环，OpenRLHF 80% profile 注记点题 |
| 44 | 08-rl-math/05-llm-rl-systems #1 | single-controller 星型编排 | 4 | 5 | 5 | 5 | 5 | **4** | t044；四组模型星型 + ①–⑤ 编号边，常驻账式 1 footer |
| 45 | 09-post-training/01-sft #1 | 多轮 mask 两取法 | 4 | 5 | 5 | 5 | 5 | **4** | t045；token 段行 + 双 mask 行对照；声明性静止合宜 |
| 46 | 09-post-training/02-rlhf-pipeline #0 | RLHF 三段链 | 4 | 5 | 4 | 5 | 5 | **4** | t046；三阶段区 + actor←π_SFT / critic←RM 继承虚线 |
| 47 | 09-post-training/03-dpo-family #0 | DPO 推导链 | 4 | 5 | 5 | 5 | 5 | **4** | t047；五步恒等链 + β·log Z(x) 划去项注记 |
| 48 | 09-post-training/04-rejection-sampling-and-rft #0 | RFT 一轮 | 4 | 5 | 5 | 5 | 5 | **4** | t048；推理栈/训练栈双区 + 回灌链 + 漂移回环 |
| 49 | 09-post-training/05-rlvr-and-reasoning #0 | R1 配方 | 4 | 5 | 5 | 5 | 5 | **4** | t049；四阶段主线 + R1-Zero/蒸馏双支线 + ckpt 标记 |

**training 结论：过线 49/50。** 维度均分：动效 4.00 / 可读 4.94 / 层级 4.64 / 细节 4.72 / 咬合 5.00（逐张目视：每图均引本章式号/§号/实账，fig-tag 全部点题；50/50 replacesBlock 挂载目标核验存在）。

### 未过线清单（training）

| # | 图 | 五维 | 病灶证据 |
|---|---|---|---|
| 32 | 06-megatron-and-stacks/03-modern-training-stacks #0（Megatron 家族：Core / NeMo / Bridge） | 4/4/4/**3**/5 | `t032-zoom-core.png`（dsf=4 局部放大）：Core→Bridge 边斜穿中央节点内部，直穿「Megatron-Core」标题文本中段——自节点左缘（非端口）入、右上缘出。修复方向：边改从 Core 节点 right/top 端口出、绕行节点外侧入 Bridge 左/下缘 |

---

## Part 2：models 修复回归复验

复验对象 = 基线报告档 1–3 清单：出框/贴界 5 张（#15/#88/#19/#27/#38）+ reduced-motion 降级失守 25 张 + 残留观察 2 张（#30/#57）。修复组：A = `figure.css` 降级复位提权（`.fig.in .sig-*`，0-3-0）+ 裸 SMIL `sig-flow` 降级隐藏；B = kit `pulse` spline 缓动 + 五张重画。本轮实拍复验（`qa/fig-review/fNNN-*.png` 全新渲染 + reduced-diff.json + DOM 核验）。

### 档 1–2：出框/贴界 5 张（重画复验）

| # | 图 | 基线五维 | R2 五维 | 复验证据 |
|---|---|---|---|---|
| 15 | 02-transformer-anatomy/02-tokenization-and-embeddings #0 | 4/4/4/**2**/5 | 4/4/4/**5**/5 | f015-norm-t6000：页脚拆两行均完整在框（「CJK 单字最坏 3 token（§3）」全句）；embedding 节点加宽 112→120、sub 留白正常；渲染态 getBBox 越界 0 命中 |
| 88 | 07-diffusion…/02-ddpm @2.3 噪声日程 | 3/4/4/**2**/5 | 4/4/4/**5**/5 | f088-norm-t6000：viewBox 340→356，页脚两行完整（「…（§2.3 的账）」全句）；图例移至曲线族右上空白、不再压端点；reduced 353px→0 |
| 19 | 02-transformer-anatomy/04-causality-and-kv-cache #1 | 4/4/4/**3**/5 | 4/4/4/**5**/5 | f019-norm-t6000：concat+W^O 上移 y 352→340、底缘 384/400 留白；V→o_t 改顶边直落、o_t→concat 端口对端口，回绕消除 |
| 27 | 03-objectives…/03-decoder-only-and-icl @§6(b) | 3/4/4/**3**/5 | 4/4/4/**5**/5 | f027-norm-t6000：注记 ①②③ x=28 起、② 式 7 收尾在框内；reduced 2932px→0 |
| 38 | 03-objectives…/02-encoder-architectures #1（ModernBERT 层栈） | 4/4/4/**4**/5 | 4/4/4/**5**/5 | f038-norm-t6000：题注缩短后度量 0 命中（基线 722px 出界 2px 的记注项消除） |

### 档 3：reduced-motion 降级失守 25 张（系统性修复复验）

修复：`figure.css` 复位选择器提权至 `.fig.in .sig-*` 且裸 SMIL `sig-flow` 降级 `display:none`。复验三口径：

1. **像素差**（reduced t3000 vs t4500，同基线阈值 200px）：25/25 全部降至 ≤4px——#1 6137→4、#59 6363→0、#6 4244→0、#97 3766→0、#4 3525→0、#27 2932→0、#56 2450→0、#8 2124→0、#28 1291→0、#98 1285→0、#33 1013→0、#93 1010→0、#92 940→0、#5 875→0、#89 860→0、#96 839→0、#95 656→0、#54 653→0、#7 610→0、#58 595→0、#94 553→0、#26 534→3、#91 408→0、#12 359→0、#88 353→0。残留观察 #30 213→0、#57 221→0。
2. **DOM 级**（reduced 上下文 computed style）：`sig-rotor` animation-name = **none**、`sig-dashflow` = **none**、`sig-flow` display = **none**、`.dia-pulse` display = **none**（`matchMedia` reduce 已生效）。
3. **抽查实拍**：f001-reduced-t4500（RoPE 三转子）三转子静止于同一角度，降级终态正确。

25 张动效维均由 3 → **4**。

### kit pulse 配速缓动（全局提升项）复验

`figure-kit.mjs` 的 `edge(pulse)` 与各处 `pathPulse` 均加 `calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1"`。探针实测（4s 直线通道）：t=10% 处位移进度 **0.026**（线性应 0.10，缓入成立）、t=50% 处 **0.546**（中点≈0.5 成立）——单程 ease-in-out、循环有呼吸感，有向数据流语义不破。基线「paced 匀速脉冲」评注项闭环。

### 回归结论

- **档 1–3 共 28 张未过线 → 全部过线**；残留观察 2 张归零。
- 渲染态文本越界全量复查：136 张仅剩 #54 一条命中，为基线已判定的旋转文本度量误报（f054-norm-t6000 目视右侧竖排「反向：联合去掩码（式 2）」完整在框内），不扣分。
- **models 维度：108/136 → 136/136 过线。**
