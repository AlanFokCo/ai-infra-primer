# 8.5 弹性伸缩与容错（Elasticity and Fault Tolerance）

> 流量摆动让在线容量追不上需求，硬件故障让在线容量瞬间出现缺口——伸缩与容错为什么是同一笔冗余账？prefill 按 non-cached token 数定机数、decode 按 running request 数调整，grace 1 min 把未完成请求连同 KV 一次性转发，quota 利用率高于 0.8 禁 scale-in，大 EP 单 GPU 故障让整实例不可用、却靠请求级 rerouting 而非缩短重启来保 SLO。本章把这些官方披露规则还原成可算的触发账、迁移账与冗余账 | 前置：1.1（goodput 与 SLO attainment）、1.2（prefill FLOPs 与 decode 带宽账）、1.3（Little 定律与排队）、2.1 / 2.4（KV 容量与抢占重算账）、4.2（Llumnix 迁移机制与成本账）、7.4（Wide-EP、EPLB 与 release 边界）；8.2（xPyD 配比）与 9.1（容量规划）只 recap 指引；training 7.3（SDC）纯文字指引 | 锚点：DeepInfer doc（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露，L251 / L253 / L257 / L261 / L265 / L119，核实 2026-11-23）、Llumnix（[Sun et al., 2024](../references.md)，arXiv:2406.03243，[A]，llumnix-project/llumnix-ray v0）、Dynamo planner（[NVIDIA, ai-dynamo/dynamo](../references.md)，教学参照）、MemServe（[Hu et al., 2024](../references.md)，arXiv:2406.17565，[B]，无官方 repo）

## 1. 伸缩与容错是同一笔账

先回答一个更基本的问题：一个 serving 集群的在线容量，为什么永远不等于它「刚好需要」的容量？因为需求随流量摆动，而容量随硬件故障跳变——两者都让「在线容量 ≠ 需求」成为常态而非例外。弹性伸缩（elasticity）调容量去追需求，容错（fault tolerance）在容量出现缺口时保住请求的 SLO。本章的论点是：这两件事共用同一笔冗余账——多备的容量既是伸缩的缓冲，也是故障的落点。

**本章口径**（R2-2 章脊锁定）：数值算例沿用 1.3 / 4.2 的 8B/H100 章脊，以便与 4.2 的迁移成本账直接对账——Llama-3.1-8B 几何（ $L = 32$ 、 $\kappa L = 128$ KiB/token、matmul 口径 $N = 7.50 \times 10^{9}$ ），H100 SXM（ $F_{\mathrm{eff}} = 494.75$ TFLOPS = 50% × 989.5、每 GPU 可供 KV 的 HBM 预算取 60 GiB 教学值），单机 8 GPU；工作负载 $\lambda = 1000$ req/s、 $T_{\mathrm{in}} = 4096$ 、 $T_{\mathrm{out}} = 512$ 、prefix 命中率 $h = 0.5$ 、TPOT SLO 上限 $\tau = 20$ ms（本章 $\tau$ = TPOT SLO，与 1.4 采样温度 $\tau$ 不同章不同义； $\lambda$ = 到达率，非 GAE 参数）。DeepInfer 生产规则与生产数字（EP64、64P+64D、grace 1 min、quota 0.8、400 Gbps/8 GPU、重启分钟级）按「官方披露」+ doc 行号单独标注，**不与 8B 教学算例混算**；换口径处一句话声明。bytes 不占数学字母，一律文字「字节」或 $\mathrm{bytes}$ ；实例计数用 $n_P / n_D$ （8.2 约定），禁裸 P / D 当计数。边界：本章讲 serving 侧弹性与容错的系统设计；RDMA / NVLink 硬件与网络拓扑归 gpu 7.1 / 9.1 / 10.1，训练侧弹性编排归 training 维度，KV 传输带宽账详 8.3。

```text
需求与容量的两条曲线（伸缩追需求，冗余吸收故障）

  容量
   ^          ____ 在线容量 n_online（含冗余）
   |     ____/    \____        <- scale-out 追流量峰
   |  __/   冗余余量  \___     <- scale-in 被 u>0.8 挡住
   | /  ·····需求 n_demand····\····
   |/                          \__
   +---------------------------------> 时间
        故障缺口 f·n_online
        |<-- rerouting 落点 -->|  请求迁到健康实例（ms 级）
        ready 机池（unoccupied）= scale-out 库存，免冷启动分钟级
```

所谓弹性伸缩，是指平台按负载信号增减在线实例数；所谓容错，是指单实例失效时请求仍能在 SLO 内继续；所谓请求级 rerouting，是指把故障实例上的请求改派或迁移到健康实例续跑（机制见 §6）。三者的交点是冗余 quota（§5）：没有冗余，伸缩只能追着故障跑；有了冗余，故障缺口由 rerouting 填、重启在后台慢慢做（§6）。

## 2. 伸缩触发账：non-cached tokens 与 running requests

DeepInfer 的 autoscaling 用什么信号定机数？doc L251 逐字（官方披露，2026-11-23 核实）：「Prefill determines the number of machines it needs from the number of non-cached tokens; decode adjusts based on the number of running requests. Adjustments are rate-limited and bounded above and below.」两相用不同信号——prefill 看 non-cached token 数（算力需求），decode 看 running request 数（并发需求）。本节把这条规则还原成两个可算的触发账。

**prefill 侧：non-cached token 率定机数。** 路线（3 步）：先把到达率折成「需要真正计算的 token 率」（扣掉 prefix cache 命中），再除以单机 prefill 算力容量，最后向上取整。关键一步是命中部分不耗 prefill 算力——这是用 non-cached 而非 total tokens 的全部理由。设到达率 $\lambda$ 、输入长 $T_{\mathrm{in}}$ 、命中率 $h$ ，则 non-cached token 到达率

$$
R_{\mathrm{nc}} = \lambda \thinspace (1 - h) \thinspace T_{\mathrm{in}} \qquad (1)
$$

**符号**： $R_{\mathrm{nc}}$ = 每秒需 prefill 计算的 token 数； $h$ = prefix 命中率（口径：被 cache 复用、无需重算的输入 token 占比，2.2）。单机 prefill 容量由算力定：每 GPU 每秒可算 $F_{\mathrm{eff}} / (2 N_{\mathrm{act}})$ 个 non-cached token（prefill 每 token 约 $2 N_{\mathrm{act}}$ FLOPs，1.2 recap）， $g_P$ 个 GPU 组成一台 prefill 机，则

$$
n_P = \left\lceil \frac{R_{\mathrm{nc}}}{C_P} \right\rceil, \qquad C_P = \frac{F_{\mathrm{eff}}}{2 N_{\mathrm{act}}} \thinspace g_P \qquad (2)
$$

**符号**： $C_P$ = 单机 prefill non-cached token 容量（token/s）； $N_{\mathrm{act}}$ = 激活参数量（MoE 用激活、dense 用 $N$ ，1.2）； $g_P$ = 单机 prefill GPU 数。代入章脊（dense $N_{\mathrm{act}} = N$ 、 $g_P = 8$ ）：每 GPU 容量 $494.75 \times 10^{12} / (2 \times 7.50 \times 10^{9}) \approx 32{,}983$ token/s， $C_P \approx 263{,}867$ token/s； $R_{\mathrm{nc}} = 1000 \times 0.5 \times 4096 = 2{,}048{,}000$ token/s， $n_P = \lceil 7.761 \rceil = 8$ 台。

**命题 1**（命中线性消减 prefill 需求）：在式 (2) 下 prefill 需求机数（取整前） $\propto (1 - h)$ ， $n_P$ 是其向上取整；decode 需求机数不直接含 $h$ 。**证明**（要点）：式 (1) 中 $R_{\mathrm{nc}}$ 对 $(1-h)$ 线性，式 (2) 的 $C_P$ 与 $h$ 无关，故取整前的需求 $R_{\mathrm{nc}} / C_P$ 随 $(1-h)$ 线性缩放（ $n_P$ 再取整，引入至多 1 台的舍入）；decode 侧触发量是 running request 数（式 (3)），由 $\lambda$ 与输出时长定，与输入是否命中无关。∎ 边界反例： $h = 1$ （全命中）时 $R_{\mathrm{nc}} = 0$ 、 $n_P = 0$ ，但 decode 仍须 $n_D > 0$ ——命中省的是 prefill 算力，不省 decode 并发；这正是「prefill 看 token、decode 看请求」两信号分离的结构性原因。命中率如何换成成本节省见 9.1（recap，不重推）。

**decode 侧：running request 数定机数。** 路线（3 步）：用 Little 定律把到达率折成稳态并发请求数，再除以单机 decode 并发容量，向上取整。关键一步是 Little 定律把「每秒来多少」接通到「同时在跑多少」。设输出长 $T_{\mathrm{out}}$ 、每 token 时延 $\tau$ ，请求在 decode 的逗留时间约 $T_{\mathrm{out}} \tau$ （忽略排队），则稳态 running request 数

$$
C = \lambda \thinspace T_{\mathrm{out}} \thinspace \tau \qquad (3)
$$

**符号**： $C$ = 全集群稳态并发 decode 请求数（口径：正在逐 token 生成的请求总数，即 doc 的 running requests；Little 定律 $C = \lambda W_{\mathrm{soj}}$ ，1.3 recap）。单机 decode 并发容量 $C_D$ 由 KV HBM 预算定：每请求平均占 $\bar{T} \kappa L$ 字节（ $\bar{T} = T_{\mathrm{in}} + T_{\mathrm{out}}/2$ 为平均上下文长），每 GPU 预算 $M_{\mathrm{KV}}$ 字节、 $g_D$ 个 GPU 一台机，则

$$
n_D = \left\lceil \frac{C}{C_D} \right\rceil, \qquad C_D = \frac{M_{\mathrm{KV}}}{\bar{T} \thinspace \kappa L} \thinspace g_D \qquad (4)
$$

**符号**： $C_D$ = 单机 decode 并发容量（请求数）； $M_{\mathrm{KV}}$ = 每 GPU 可供 KV 的 HBM 字节； $\kappa$ = 每 token 每层 KV 字节（1.2）； $g_D$ = 单机 decode GPU 数。代入章脊： $C = 1000 \times 512 \times 0.020 = 10{,}240$ 请求； $\bar{T} = 4352$ 、每请求 KV $= 4352 \times 128$ KiB $\approx 0.531$ GiB，每 GPU 并发 $60 / 0.531 \approx 112.9$ 、 $C_D \approx 903.5$ ， $n_D = \lceil 11.33 \rceil = 12$ 台。配比 $n_P : n_D = 8 : 12 = 1 : 1.5$ ——这是容量闸主导 sizing 的教学负载配比，**不是** DeepInfer 生产配比；生产 ~1:1 是流量经验值、最小高效单元 64P+64D（doc L91 官方披露），最优配比随负载与硬件变化（8.2 xPyD 推导 recap，不重推）。

> **提点**：decode 并发还有第二道闸——HBM 带宽。式 (4) 只计 KV 容量约束；当并发高到每步 KV 读字节逼近带宽上限，TPOT 先于容量撞 SLO（1.2 / 1.3 roofline recap）。生产 sizing 取两道闸的较小者，本章算例取容量闸（带宽账归 9.1）。

**rate-limit 与上下界。** doc L251 的「rate-limited and bounded above and below」是防抖与防越界两道护栏。设第 $k$ 个调整周期的目标机数 $n^{\mathrm{target}}$ （由式 (2) 或 (4) 算出），实际下发

$$
n^{(k+1)} = \mathrm{clip}\negthinspace\left(n^{\mathrm{target}},\ n^{(k)} - \Delta_{\mathrm{down}},\ n^{(k)} + \Delta_{\mathrm{up}}\right), \qquad n_{\min} \le n^{(k+1)} \le n_{\max} \qquad (5)
$$

**符号**： $\Delta_{\mathrm{up}} / \Delta_{\mathrm{down}}$ = 每周期允许增 / 减的机数上限（rate-limit）； $n_{\min} / n_{\max}$ = 机数上下界； $\mathrm{clip}(x, a, b)$ = 把 $x$ 夹到 $[a, b]$ 。代入：当前 $n_P = 8$ 、目标跳到 16、 $\Delta_{\mathrm{up}} = 2$ ，则需 $\lceil (16-8)/2 \rceil = 4$ 个周期才到位——流量尖峰被摊平，避免一次性冷启动大批机器（冷启动分钟级，§6）。上下界则保证低谷不低于最小可用单元（DeepInfer 最小高效单元 64P+64D，doc L91）、高峰不超过 quota 上限。rate-limit 的代价是响应滞后：尖峰前几个周期容量不足，靠冗余 quota（§5）兜底。

## 3. SLO-aware 伸缩：负载信号 vs 延迟信号

负载信号能直接保证 SLO 吗？§2 的触发量是 token 数与请求数，不是延迟；另一条路线直接用 SLO 信号（延迟）驱动伸缩——Dynamo 的 SLA planner 是公开对照（[NVIDIA, ai-dynamo/dynamo](../references.md)，教学参照，planner-design.md）。它双模并用：throughput-based 用流量预测 + 引擎性能模型算出保 TTFT/ITL SLA 所需的副本下界；load-based 用 ForwardPassMetrics（FPM）事件流做在线回归、按实测延迟反应式修正副本数。两相各有独立性能模型，worker inventory 记 ready/expected 计数与在途 scale。

两条路线的差别不在「要不要 SLO」，而在「用什么信号闭环」。负载信号（DeepInfer）直接测需求、测量成本低且 robust，但它不直接保证延迟——负载到延迟之间隔着排队与干扰，须靠冗余余量吸收（§5 命题 4）。SLO 信号（Dynamo）直接测目标量（TTFT/ITL），闭环对准目标，但依赖性能模型与预测，模型误差会直接进伸缩决策。对照如下：

| 轴 | DeepInfer autoscaling（doc L251） | Dynamo SLA planner（planner-design.md） |
| --- | --- | --- |
| 触发信号 | 负载量：prefill non-cached tokens / decode running reqs | 延迟量：TTFT / ITL SLA + 流量预测 |
| 闭环方式 | 反应式，rate-limit + 上下界 | throughput 设副本下界 + load 在线回归反应式修正 |
| SLO 保证路径 | 间接：冗余 quota 余量吸收负载到延迟的缺口 | 直接：性能模型把 SLA 翻译成副本数 |
| 主要风险 | 余量不足时尖峰撞 SLO | 性能模型 / 预测误差进决策 |
| 证据 | 官方披露（教学参照） | 官方 repo 文档（教学参照） |

**命题 2**（负载信号 + 余量间接保 SLO）：若在线容量始终不低于需求的 $1/u_{\max}$ 倍（ $u_{\max} < 1$ 为利用率上界），则负载信号伸缩可在不直接测延迟的前提下留出吸收排队与干扰的余量。**证明**（要点）：利用率 $u$ （需求 / 容量） $\le u_{\max}$ 意味着容量至少是需求的 $1/u_{\max}$ 倍，spare 比例至少 $1 - u_{\max}$ ；由 1.3 排队账，逗留时间随 $u \to 1$ 发散，把 $u$ 压在 $u_{\max} < 1$ 即把排队等待压在有限值，余量 $1 - u_{\max}$ 同时吸收流量摆动与故障缺口（§5）。∎ 口径注：这是「余量换 SLO」的结构论证，不给出 $u_{\max}$ 到具体 TTFT 分位的定量映射（那要性能模型，正是 Dynamo 路线补的环节）； $u_{\max} = 0.8$ 的生产取值见 §5。

## 4. grace period 与请求迁移

下线一台还挂着未完成请求的 decode 机，怎么既不丢请求、又不让整机为几条长尾请求空等？直接杀进程会丢请求，等它们自然结束又可能很久。DeepInfer 的解法是 grace period。doc L253 逐字（官方披露）：「To address the resource waste that long-tail requests cause in large-EP deployments, a decode engine (CE) enters a grace period (1 min) after receiving a shutdown notice; any requests still unfinished, along with their KV cache, are forwarded in one shot to other CEs to continue decoding.」所谓 grace period，是指 decode 引擎（CE）收到 shutdown 通知后的一段缓冲窗口（1 min），窗口内让短请求自然收尾、窗口末把仍未完成的请求连同 KV 一次性（in one shot）转发到其他 CE 续 decode。

动机是大 EP 的长尾资源浪费：EP64 的 decode 实例是一个整体（§6），批里大部分请求已结束、只剩几条长尾请求时，整台 64 GPU 实例仍被这几条请求占着。grace 窗口先放短请求走，再把长尾几条迁走、释放整机。迁移为什么可行？因为转发成本是 ms 级。

**grace 迁移成本账。** 路线（3 步）：先算一次性转发的暴露 downtime（全量剩余 KV 传输），再对照重算与 Llumnix 多阶段迁移，最后说明 1 min 窗口为何宽裕。关键一步是「一次性」意味着传输不被 decode 掩盖、整段暴露。设请求剩余上下文长 $T_{\mathrm{rem}}$ （已生成前缀的 KV 须全部搬走），实例间有效带宽 $\beta_{\mathrm{net}}$ ，则一次性转发时间

$$
t_{\mathrm{fwd}} = \frac{\kappa L \thinspace T_{\mathrm{rem}}}{\beta_{\mathrm{net}}} \qquad (6)
$$

**符号**： $t_{\mathrm{fwd}}$ = 一次性转发暴露 downtime（源端停发末 token 到目的端续发首 token 的缺口主项）； $\kappa L$ = 每 token 全层 KV 字节； $\beta_{\mathrm{net}}$ = 实例间有效带宽（口径：单向，目的端收齐即续跑）。代入章脊（ $T_{\mathrm{rem}} = 4096$ 、KV $= 0.5$ GiB、 $\beta_{\mathrm{net}} = 50$ GB/s 即 400 Gbps）： $t_{\mathrm{fwd}} = 0.5369 / 50$ s $\approx 10.7$ ms。对照同 $T_{\mathrm{rem}}$ 的重算 $\approx 133$ ms（线性项 $2 N T_{\mathrm{rem}} / F_{\mathrm{eff}} \approx 124$ ms + attention 二次项 $\approx 8.9$ ms，4.2 口径 recap）：转发比重算便宜约 12×（口径：便宜指暴露时间之比），且不丢已生成 token。

**命题 3**（一次性 vs 多阶段的暴露 downtime）：一次性 grace 迁移的暴露 downtime 随剩余序列长线性增长（式 (6)）；Llumnix 多阶段迁移只暴露末次 iteration 的 KV 拷贝、近似与序列长无关。**证明**（要点）：一次性转发在源端停算后整段搬 KV，暴露时间 = 全量 KV / 带宽 $\propto T_{\mathrm{rem}}$ ；多阶段迁移（4.2 §4 recap）前几阶段边 decode 边拷、批量段被持续 decode 掩盖，末阶段 drain 出 batch 只拷末块，暴露时间 $\approx$ 末次 iteration KV / 带宽 = 常数。∎ 代入对照（4.2 式 (4) 章脊）：多阶段批量段 10.7 ms 被掩盖、尾段 $\le 16$ 块 $\times 64 \times 128$ KiB / 50 GB/s $\approx 2.7$ ms 暴露，合计 13.4 ms 但只暴露 2.7 ms；一次性则整段 10.7 ms 全暴露。

> **坑**：把 DeepInfer 一次性 grace 迁移与 Llumnix 多阶段 live migration 当同一机制。触发条件：见「迁移」就套 4.2 的「near-zero downtime」；现象：一次性转发整段 KV 暴露（ $\propto T_{\mathrm{rem}}$ ），多阶段只暴露尾段（常数 20-30 ms，论文 §6.2）——两者暴露 downtime 量纲不同；规避：grace 迁移是 scale-in 的计划性动作（非延迟热路径），1 min 窗口吸收整段暴露绰绰有余，故 DeepInfer 选简单的一次性而非多阶段；Llumnix 多阶段服务于在线负载均衡 / 碎片整理（延迟敏感），才值得付多阶段协调复杂度。

1 min 窗口为何宽裕？式 (6) 的转发是 ms 级，而长尾请求自然结束是 s 级：剩余输出 $T_{\mathrm{out}}^{\mathrm{rem}}$ 个 token 按 $\tau = 20$ ms 自然结束需 $T_{\mathrm{out}}^{\mathrm{rem}} \times 20$ ms，要超过 60 s grace 须 $T_{\mathrm{out}}^{\mathrm{rem}} > 3000$ token。所以 grace 窗口的主要作用不是「等转发」，而是「放短请求自然走、少迁几条」；窗口末仍剩的长尾请求，转发成本（ms）远小于其自然结束时间（s），迁走即释放整机。

**Llumnix 的伸缩编排（recap 4.2 + 指引）。** 4.2 已走读迁移机制与成本账（多阶段、gloo 后端、migration_policy），本章只补伸缩编排层。Llumnix 用 virtual usage 统一四场景（load balancing / de-fragmentation / prioritization / drain-saturate for auto-scaling，论文 §4.4）：所谓 virtual usage，是指把请求对实例资源的占用折算成一个可加减的统一度量，其「空闲度」freeness $F$ 可负（负值表示该实例应收缩）。auto-scaling 即 drain-saturate 场景——把低 freeness 实例的请求迁出（drain）以释放整机、或把请求迁入（saturate）以填满实例；论文 §6.5 给 freeness 阈值 [10, 60]、同激进度下至多省 16% 成本 / 同 P99 prefill 延迟目标下至多省 36%。全局调度器（instance-oriented，按 load 配对 src/dst）+ 每实例 llumlet（local scheduler + migration coordinator）的两层架构见论文 §4.3；代码锚 llumnix-ray v0 `llumnix/llumlet/migration_coordinator.py` MigrationCoordinator L228 / migrate_out L251、`scaler.py` Scaler L84、`manager.py`（global scheduler）。迁移传输用 Gloo Send/Recv 而非 NCCL（论文 §5：NCCL 并发调用 unsafe，迁移须与推理并行）——机制与成本账详 4.2，不重推。

> **坑**：Llumnix 版本与后端漂移。触发条件：按 v1 或 NCCL 后端写 v0 机制；现象：机制锚 v0 = llumnix-project/llumnix-ray（论文实现，末次更新 2026-03），v1 重构另立 llumnix-project/llumnix（cloud-native）；v0 迁移后端默认 gloo（`RayColMigrationBackend` L220，pygloo bf16→glooFloat16 L212-215），NCCL 并发 unsafe 非默认；规避：本章机制与代码锚一律 v0，v1 只作版本注。

## 5. 冗余 quota：ready 机池与 0.8 禁 scale-in

伸缩为什么要故意多备容量？doc L257 逐字（官方披露）：「Deployments keep a redundant quota to absorb the impact of traffic swings, in two forms: ready machines held unoccupied in the pool as scale-out inventory, and always keeping more instances online than the computed demand — concretely, scale-in is forbidden whenever quota utilization is above 0.8.」所谓冗余 quota，是指为吸收流量摆动而多备的容量，两种形态：其一，ready 机池——空占在池里的机器作 scale-out 库存；其二，在线实例数始终多于计算需求——具体化为 quota 利用率高于 0.8 时禁止 scale-in。

**冗余 quota 权衡账。** 路线（3 步）：先定义利用率与余量，再推余量如何同时吸收流量摆动与故障缺口，最后给冗余的成本。关键一步是「利用率上界」与「可吸收故障比例下界」是同一个量的两面。设需求机数 $n_{\mathrm{demand}}$ （式 (2)/(4) 算出）、在线机数 $n_{\mathrm{online}}$ ，quota 利用率

$$
u = \frac{n_{\mathrm{demand}}}{n_{\mathrm{online}}}, \qquad u > u_{\max} = 0.8 \Rightarrow \text{scale-in forbidden} \qquad (7)
$$

**符号**： $u$ = quota 利用率（口径：计算需求占在线容量的比例，非 GPU 占用率；quota 账按机数计，一台 prefill/decode 机为一个计数单位，大 EP decode 实例内部的 GPU 组成与 failover 见 §6，机数与 GPU 数两层粒度不混）； $u_{\max} = 0.8$ = 禁 scale-in 阈值（doc L257）。在线容量 $n_{\mathrm{online}} = n_{\mathrm{demand}} / u \ge n_{\mathrm{demand}} / 0.8 = 1.25 \thinspace n_{\mathrm{demand}}$ ，即至少多备 25%（相对需求）的余量。这笔余量买两件事：流量上摆时不用等冷启动（ready 机池即时顶上），实例故障时有落点接住 rerouting（§6）。

**命题 4**（利用率上界 = 容错余量下界）：设集群可吸收的实例故障比例上界为 $f_{\max}$ （故障后仍能服务需求的最大故障比例）。若在线容量利用率 $u \le u_{\max}$ ，则 $f_{\max} = 1 - u \ge 1 - u_{\max}$ ——冗余 quota 保证**至少**可吸收 $1 - u_{\max}$ 比例的实例故障，最坏情形 $u = u_{\max}$ 时取等； $u_{\max} = 0.8$ 时 $f_{\max} \ge 0.2$ 。**证明留作练习 3**（路线提示：故障后剩余容量 $(1-f) n_{\mathrm{online}} \ge n_{\mathrm{demand}}$ 解出 $f \le 1 - u$ ，即 $f_{\max} = 1 - u$ ；再由 $u \le u_{\max}$ 得 $1 - u \ge 1 - u_{\max}$ ）。代入： $n_{\mathrm{online}} = 100$ 、 $n_{\mathrm{demand}} = 80$ 、 $u = 0.8$ （最坏情形）， $f_{\max} = 0.2$ ；故障 $f = 0.2$ （20 台）后剩 80 台 = 需求仍可服务，故障 21 台（ $f = 0.21 > f_{\max}$ ）后剩 79 < 80 出现缺口。若冗余更多（ $u = 0.5$ 、需求 50 台）， $f_{\max} = 0.5 > 0.2$ ，可吸收 $f = 0.3$ 的故障——可吸收比例随冗余增大，故 0.8 阈值保证的是**下界** 0.2 而非上界。所以 0.8 阈值把「至少 20% 在线容量是 spare」变成硬约束——这 20% 既是伸缩缓冲，也是故障落点，正是 §1「同一笔账」的定量兑现。

冗余的成本是实打实的：多备的 $n_{\mathrm{online}} - n_{\mathrm{demand}}$ 台机器闲置或低载（上例 20 台）。权衡的另一侧是收益——免冷启动延迟与免 SLO 违约。ready 机池把 scale-out 延迟从「冷启动分钟级」（§6 重启分钟级同源）压到「就绪机即时上线」；在线余量把故障缺口从「请求 stall 等重启」压到「rerouting ms 级」（§6 命题 5）。冗余不是浪费，是把分钟级的不可用换成 ms 级的迁移 + 一笔闲置成本。成本账（USD/M tokens 与冗余定价）归 9.1，本章只记结构。

> **提点**：「scale-in 禁于 $u > 0.8$」是一条把容错余量写进伸缩策略的护栏——它不直接谈故障，却用利用率上界保证了故障落点。策略（何时缩）与机制（故障怎么接）在此分离：缩容策略留余量，rerouting 机制用余量。

## 6. EP failover 与容错哲学

大 EP 部署的容错难在哪？doc L265 逐字（官方披露）：「In a large-EP deployment, all GPUs of an instance form a single communication domain, so a single failed GPU makes the whole instance unusable. On restart, in order to keep the instance's remaining healthy GPUs in service, the platform re-determines the EP size from all currently healthy GPUs, and the instance takes the number of GPUs it actually receives as its EP size.」所谓 EP failover，是指大 EP 实例在 GPU 故障后的恢复策略：单通信域里一张 GPU 故障就让整实例不可用（all-to-all 须全员参与，7.4 / training 5.4 recap），重启时平台从当前健康 GPU 重定 EP size，实例以实收 GPU 数为 EP。

**EP 重定 size 的账。** 实例从 $e$ 张 GPU 缩到 $e'$ 张（ $e' < e$ ），每卡 MoE 权重流量 $\propto E/e$ （1.2 / 7.4 命题 recap），故

$$
\frac{w(e')}{w(e)} = \frac{e}{e'} \qquad (8)
$$

**符号**： $w(e)$ = EP 度 $e$ 下的每卡 MoE 权重流量； $e$ = 原 EP 度； $e'$ = failover 后实收 EP 度； $E$ = routed expert 总数（与 $e$ 严格区分大小写）。代入 V4.1-Flash（ $E = 384$ ，config [W]）：EP64 每卡 $384/64 = 6$ 个 expert，缩到 EP56 每卡 $384/56 \approx 6.857$ 个，每卡流量 $\times 64/56 \approx 1.143$ （+14.3%）。代价是每卡负载升、整实例聚合吞吐降（卡少了）；收益是实例**仍可用**——健康 GPU 继续服务，而非整机停摆等齐 64 张。EP 重定 size 须重排 expert（EPLB recap 7.4，不重推），这是重启分钟级的一部分。口径注：式 (8) 是每卡权重流量的结构性比例（ $\propto E/e$ ），非实测吞吐比；吞吐还受通信 / 同步与 EP 拐点影响（7.4 EP>32 递减 recap）。

**引擎侧 elastic EP（recap 6.3 + 指针）。** 上面的 EP failover 是 DeepInfer 平台层口径（doc L265 披露、release 外，靠重启重定 size）；开源引擎另把 EP 伸缩做成引擎侧运行时状态机、不靠重启：SGLang `elastic_ep/elastic_ep.py` 的 ScaleCohort L30（一起扩缩的 rank 组）/ ElasticEPStateManager L132 / update_dp_attention_for_elastic_ep L77（扩缩后同步 attention DP 宽度）+ expert 权重备份（新 rank 从备份取 expert、不重读 checkpoint），模块走读见 6.3 §3；vLLM 对照件 `distributed/elastic_ep/`（elastic_execute / elastic_state / standby_state，5.3 注）。两层分工：引擎侧状态机管「EP 组运行时怎么扩缩与同步」（6.3），平台层 failover 管「故障后实例收几张 GPU、何时重启」（本章 doc 口径）；本章增量是触发账与容错哲学，不重走读引擎侧状态机。

**容错哲学：不缩短重启，靠冗余 + rerouting。** doc L261 逐字（官方披露）：「Restarts are on the order of minutes. Fault tolerance comes from PE (prefill engine) / CE quota redundancy plus request-level rerouting, rather than from shortening the restart time of a single engine.」重启是分钟级；容错来自 PE/CE quota 冗余加请求级 rerouting，而非缩短单引擎重启时间。为什么不走「把重启做快」这条路？算一笔量纲账就清楚。

**命题 5**（rerouting 缺口与重启时间解耦）：设单引擎重启时间 $T_{\mathrm{restart}}$ （分钟级）、请求级 rerouting 缺口 $t_{\mathrm{reroute}}$ （迁移 / 重派发，ms 级），则故障请求的 SLO 缺口由 $t_{\mathrm{reroute}}$ 界定、与 $T_{\mathrm{restart}}$ 无关，前提是冗余 quota 提供健康落点。**证明**（要点）：故障发生后，请求经 rerouting 迁到健康实例（grace 迁移式 (6) 或重派发），缺口 $\approx t_{\mathrm{fwd}}$ ms 级；重启在后台恢复故障实例容量、耗时 $T_{\mathrm{restart}}$ ，但请求已在健康实例续跑、不等重启；冗余 quota（§5 命题 4 的 $f_{\max} \ge 1 - u_{\max}$ 余量）保证健康落点存在。故 SLO 缺口 $\le t_{\mathrm{reroute}} \ll T_{\mathrm{restart}}$ 。∎ 代入章脊： $T_{\mathrm{restart}} \approx 60$ s（分钟级）、 $\tau = 20$ ms，若靠重启则每请求 stall $60 / 0.020 = 3000\times$ TPOT SLO；rerouting 缺口 $\approx 10.7$ ms $< 20$ ms SLO。把重启从 60 s 压到 30 s 仍 stall $1500\times$ SLO——缩短重启这条路在量纲上就走不通，冗余 + rerouting 才把缺口压进 SLO。

> **提点**：这是「机制与策略分离」在容错上的体现——重启是容量恢复机制（慢、后台），rerouting 是请求保活机制（快、前台），冗余 quota 是两者之间的缓冲。容错设计的目标不是让慢机制变快，而是让快机制不必等慢机制。

## 7. 故障分类、检测与训推共调度

故障怎么检出？容错的前提是检出故障，而故障按可检性分三类，检测手段各不同：

| 类别 | 表现 | 检测手段 | 处置 |
| --- | --- | --- | --- |
| crash（崩溃） | GPU 掉卡 / 进程退出 / NCCL 通信域断裂 | 健康检查 / heartbeat 失联 | 标记实例 down → rerouting（§6） |
| hang（挂起） | NCCL 集合通信卡死 / 死锁，进程活着但不产出 | watchdog 超时（步时无进展） | 超时判故障 → rerouting |
| SDC（静默数据损坏） | 硬件状态正常、输出错，无日志预警 | 健康检查不可检；间接统计判据 | 一句指引 → training 7.3 |

健康检查的公开实现锚：Llumnix 用 Ray actor liveness + heartbeat 检故障实例、故障 instance 的请求 abort 后由 Ray 重启 actor（论文 §5）；MemServe 的 cluster manager 用 heartbeat 检故障、广播 cluster info、释放故障 instance 占的 memory block 防泄漏（§4.4，[B] 讲论文不推测代码）；Dynamo 用 worker register/deregister + discovery 做 readiness（runtime xPyD，worker 运行时加入 / 移除，8.4 recap）。三者共同点：检测面是「实例是否还在正常产出」，检出后把请求从故障实例摘走——这正是 §6 rerouting 的触发端。SDC（silent data corruption，静默数据损坏）是唯一健康检查检不出的通道：计算结果错而硬件状态正常，须走间接统计判据 + 定点硬件测试，其故障统计口径与检出方法归 training 7.3（纯文字指引，本章不展开）。

**训推集群共调度。** doc L119 逐字（官方披露）：「Training and inference clusters are managed together, and the resource allocation between them can be adjusted dynamically.」训练与推理集群统一管理，两者间的资源分配可随流量动态调整。从 serving 侧资源视角看：推理需求有昼夜摆动，低谷时推理把 GPU 让给训练、高峰时收回；§5 的 ready 机池与在线余量正是让这个动态分配可行的缓冲——推理缩容释放的机器进训练池，推理扩容从 ready 池即时取机。边界：训练侧的弹性编排（作业调度、checkpoint、弹性训练恢复）归 training 维度，本章只讲 serving 侧的资源分配视角，不越训练侧编排。

> **坑**：把「训推共调度」读成推理引擎内部调度训练任务。触发条件：见「资源动态分配」就以为单引擎混跑训推；现象：doc L119 讲的是**集群间**资源配额随流量调整（平台层），不是引擎内训推混部（那是 RL rollout 的 colocate，归 9.3 / training 8.5）；规避：本章只记集群间配额视角，引擎内训推切换归 9.3。

## 总结

| 论断 | 证据 / 回指 |
| --- | --- |
| 伸缩与容错共用冗余账：多备容量既是伸缩缓冲也是故障落点 | §1、§5 命题 4 |
| prefill 机数由 non-cached token 率定（需求 $\propto (1-h)$ ，取整前），decode 机数由 running request 数定（Little 定律） | 式 (1)-(4)、命题 1、doc L251 |
| rate-limit + 上下界防抖防越界，代价是响应滞后、靠冗余兜底 | 式 (5)、§2 |
| 负载信号（DeepInfer）间接保 SLO 靠余量；SLO 信号（Dynamo）直接闭环靠性能模型 | §3 对照表、命题 2 |
| grace 1 min 一次性转发剩余 KV，暴露 downtime $\propto T_{\mathrm{rem}}$ ；多阶段只暴露尾段常数 | 式 (6)、命题 3、doc L253、4.2 recap |
| quota 利用率 $u \le 0.8$ 保证可吸收故障比例下界 $f_{\max} \ge 0.2$ ；ready 机池免冷启动 | 式 (7)、命题 4、doc L257 |
| EP 单通信域单 GPU 故障整实例不可用；重定 EP size 每卡流量 $\times e/e'$ ，保可用而非保效率；引擎侧 elastic EP 状态机 recap 6.3 | 式 (8)、doc L265、7.4 / 6.3 recap |
| 容错靠冗余 + 请求级 rerouting（ms 缺口）而非缩短重启（分钟级，量纲上走不通） | 命题 5、doc L261 |
| 故障三分（crash / hang / SDC），健康检查检前两类、SDC 归 training 7.3；训推共调度是集群间配额视角 | §7、doc L119 |

## 练习

1.（计算 · [10]）伸缩触发账代入：沿用章脊（ $\lambda = 1000$ req/s、 $T_{\mathrm{in}} = 4096$ 、 $T_{\mathrm{out}} = 512$ 、 $h = 0.5$ 、 $\tau = 20$ ms、 $C_P \approx 263{,}867$ token/s、 $C_D \approx 903.5$ ）。(a) 由式 (1)(2) 算 $n_P$ ；(b) 由式 (3)(4) 算 running request 数 $C$ 与 $n_D$ ；(c) 命中率升到 $h = 0.8$ ， $n_P$ 变多少、 $n_D$ 变不变？先写出 (c) 的预测再算。校验：(a) $n_P = 8$ ；(b) $C = 10{,}240$ 、 $n_D = 12$ ；(c) non-cached 率降为 0.4×（命题 1，取整前）， $n_D$ 不含 $h$ 故不变。

2.（计算 · [20]）grace 迁移成本对账：章脊 $T_{\mathrm{rem}} = 4096$ 、 $\beta_{\mathrm{net}} = 50$ GB/s。(a) 由式 (6) 算一次性转发 $t_{\mathrm{fwd}}$ ；(b) 算同 $T_{\mathrm{rem}}$ 重算时间（线性 + attention 二次项）；(c) 对照 4.2 式 (4) 多阶段迁移的暴露尾段，先预测「一次性 vs 多阶段 vs 重算」三者暴露 downtime 的大小序再计算。校验：(a) $\approx 10.7$ ms；(b) $\approx 133$ ms（4.2 口径）；(c) 多阶段尾段 2.7 ms < 一次性 10.7 ms < 重算 133 ms，与命题 3 一致。

3.（推导 · [30]）证命题 4（利用率上界 = 容错余量下界）：(a) 由「故障后剩余容量 $(1-f) n_{\mathrm{online}} \ge n_{\mathrm{demand}}$ 」与式 (7) 解出可吸收上界 $f_{\max} = 1 - u$ ，再由 $u \le u_{\max}$ 证 $f_{\max} \ge 1 - u_{\max}$ （ $u = u_{\max}$ 取等）；(b) 代入 $u_{\max} = 0.8$ 、 $n_{\mathrm{online}} = 100$ 、 $n_{\mathrm{demand}} = 80$ （ $u = 0.8$ 最坏情形），验证 $f = 0.2 = f_{\max}$ 是边界（剩 80 = 需求）、 $f = 0.21$ 出现缺口；再取 $u = 0.5$ （需求 50）验证 $f_{\max} = 0.5 > 0.2$ ；(c) 若把阈值放宽到 $u_{\max} = 0.9$ ，容错余量下界与闲置成本各如何变？校验：(a) 与命题 4 陈述一致（ $f_{\max} = 1-u \ge 1-u_{\max}$ ，方向为下界）；(b) $f = 0.2$ 剩 80、 $f = 0.21$ 剩 79 < 80、 $u = 0.5$ 时 $f_{\max} = 0.5$ ；(c) 下界降到 10%、闲置成本降——与 §5 权衡账同向。

4.（论证 · [40]）设计一个 SLO-aware 伸缩 + 容错策略并论证其 SLO 缺口界：(a) 对比 DeepInfer 负载信号（doc L251）与 Dynamo SLA planner（throughput + load 双模），给出在「性能模型不可靠」与「流量尖峰频繁」两种情形下各选哪条路线及理由；(b) 把命题 5 的 rerouting 缺口界扩展到「故障比例 $f$ 超过可吸收上界 $f_{\max} = 1 - u$ 」的情形——此时缺口由什么界定？给出 $f$ 、 $f_{\max}$ 、 $t_{\mathrm{reroute}}$ 、 $T_{\mathrm{restart}}$ 的关系式；(c) 论证 ready 机池规模如何同时影响 scale-out 延迟与 (b) 的缺口，并说明训推共调度（doc L119）下 ready 池被训练借走时的风险。校验：(a) 与 §3 对照表两风险轴一致；(b) 与命题 4/5 前提（冗余存在）一致， $f > f_{\max} = 1 - u$ 时落点不足、缺口须含等待重启项；(c) 与 §5 ready 池免冷启动 + §7 共调度缓冲一致。

## 参考

- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——autoscaling 触发信号与 rate-limit 上下界（L251）、grace 1 min 一次性转发（L253）、冗余 quota 与 0.8 禁 scale-in（L257）、重启分钟级与 rerouting 哲学（L261）、EP failover 重定 size（L265）、训推共调度（L119）、~1:1 配比与 64P+64D 最小单元（L91）、跨 DC 400 Gbps/8 GPU（L97）
- [Sun et al., 2024](../references.md)：Llumnix: Dynamic Scheduling for Large Language Model Serving，arXiv:2406.03243，[A]（key: llumnix；github.com/llumnix-project/llumnix-ray = v0 论文实现，v1 另立 llumnix-project/llumnix）——multi-stage migration 与 downtime（§4.2 / §6.2）、virtual usage 与 freeness 四场景（§4.4）、global scheduler + llumlet（§4.3）、Gloo 非 NCCL（§5）、auto-scaling freeness [10,60] 与 16%/36% 成本节省（§6.5）；机制与成本账 recap 4.2；代码锚 migration_coordinator.py L228 / scaler.py L84（2026-11-23 核实）
- [NVIDIA, ai-dynamo/dynamo](../references.md)：key: dynamo，教学参照（核实 2026-11-23）——SLA planner 双模（throughput-based 流量预测 + 性能模型设副本下界 / load-based FPM 在线回归反应式修正，planner-design.md）、worker inventory ready/expected、runtime xPyD worker 加入 / 移除（8.4 recap）
- [Hu et al., 2024](../references.md)：MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool，arXiv:2406.17565，[B]（key: memserve，无官方 repo，讲论文不推测代码）——§4.4 cluster manager heartbeat 故障检测 + 释放故障 instance memory block
- [SGLang, sgl-project/sglang repo](../references.md)：key: sglang-repo，[A]（W3 同日 tarball，2026-11-23 复核）——引擎侧 elastic EP 指针（recap 6.3 §3）：`srt/elastic_ep/elastic_ep.py` ScaleCohort L30 / ElasticEPStateManager L132 / update_dp_attention_for_elastic_ep L77 + expert_backup_manager.py
- [vLLM, vllm-project/vllm repo](../references.md)：key: vllm-repo，[A]（W3 同日 tarball，2026-11-23 复核）——引擎侧 elastic EP 对照件 `distributed/elastic_ep/`（elastic_execute / elastic_state / standby_state，5.3 注）
- 跨维度指引（纯文字）：inference 1.1（goodput / SLO attainment）、1.2（prefill FLOPs / decode 带宽账）、1.3（Little 定律 / 排队）、2.1 / 2.4（KV 容量 / 抢占重算）、4.2（Llumnix 迁移机制与成本账）、7.4（Wide-EP / EPLB / release 边界）、8.2（xPyD 配比）、8.3（KV 传输带宽账）、8.4（PD 系统对照 / Dynamo runtime xPyD）、9.1（容量与成本账）、9.3（RL rollout colocate）；training 7.3（SDC 故障统计与检出）、training 8.5（训推混部训练侧编排）
