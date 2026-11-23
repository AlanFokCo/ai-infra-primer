# 7.4 大规模 EP 部署与负载均衡（Large-Scale EP Deployment and Load Balancing）

> 同一个全局 batch，EP 从 8 扩到 64，每卡 MoE 权重流量降 7.4×——为什么 prefill 偏偏用最小 EP/DP、decode 用最大 EP/DP？四个负载均衡器（Prefill / Decode / VE / EPLB）各均衡什么量、各用什么信息？本章兑现 1.2 的 Wide-EP 伏笔，走读 release 内的 EPLB 算法，并把 doc 披露的三 DP LB 架构、Engram ENP 与 deepseek-recipe 对回生产口径 | 前置：1.2（MoE 每 expert batch 命题 2、每卡流量命题 3、Wide-EP $\propto 1/e$ 账）、1.3（scale-up domain 与硬件参数表）、3.3（Mega MoE 与 EP 拐点命题 3、DeepEP 语义）、4.3（compute-quota 二次求解）、2.2 / 2.4（prefix caching 与专用存储集群）、1.1（离线 benchmark ≠ 在线 goodput）、7.1（release 边界）、7.3（kernel 栈与 DeepEP lb 原语）；training 5.4（EP 负载不均衡 $\max_j R_j$）与 models 4.5（MoE 路由）只 recap 不重推 | 锚点：DeepInfer doc + repo（[DeepSeek-AI, inference_deployment_qa.md](../references.md)、[DeepSeek-AI, DeepInfer](../references.md)，教学参照 + 官方披露；release 内 `model_executor/eplb/` 走读）、EPLB（[DeepSeek-AI, EPLB](../references.md)，教学参照 + 官方实现）、DeepEP（[DeepSeek-AI, DeepEP](../references.md)，教学参照）、deepseek-recipe（[DeepSeek-AI, deepseek-recipe](../references.md)，教学参照）、DualPath（[Wu et al., 2026](../references.md)，arXiv:2602.21548，[B]）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）

## 1. Wide-EP 生产口径与章脊

prefill 用最小 EP/DP、decode 用最大 EP/DP——同一个模型，为什么两个阶段的并行度取向相反？DeepInfer 官方部署 QA 给的生产取值（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）：prefill EP2–8 / DP2–8，「use the smallest EP/DP that HBM allows」（doc L47）；decode EP64–128 / DP64–128（Wide-EP），「use the largest EP/DP that the available GPUs allow」（doc L49）；VE 单 GPU 一般够、推荐更便宜的 GPU（L51）；TP 与 CP 暂不用（L53）。attention 侧 #DP = #EP（doc L33，沿 V3/R1 推理系统口径），四个负载均衡器 + KVCache Storage 配套（L33）。所谓 Wide-EP，是指大 $e$ 的 EP 部署（decode 阶段 EP64–128 量级，1.2 §6 定义 recap）。

prefill 取最小的 rationale（doc L57 逐字，官方披露）：「Unlike decode, prefill already has a sufficiently large batch size per expert. Smaller DP/EP groups improve both data-parallel and expert-parallel load balance and reduce NVLink traffic.」拆成三笔：其一，prefill 每 expert batch 已够大（长 prompt 一次过， $B_{\mathrm{exp}} = Bk/E$ 天然高，1.2 命题 2），不需要靠扩 EP 凑强度；其二，小 DP/EP 组改善均衡（组内 rank 少、负载粒度粗的相对偏斜小，§3 命题 2）；其三，减 NVLink 流量（dispatch/combine 跨卡通信随组规模涨）。decode 取最大则相反：每步每序列只 1 token，每 expert batch 小，必须靠 Wide-EP 减每卡权重流量 + 耦合扩 DP 凑全局 batch 提强度（§2）。EP within (super)node 增 batch 与 GEMM 效率（doc L29）。

下图是 doc 架构图（architecture_overview.png）的 ASCII redraw，把四个 LB 与 EPLB、存储的位置关系登记（出处 doc L137-161 + 架构图；**release 边界**：Entrypoint / 三 DP LB / kvcache-store 编排不在 repo，只讲 doc 披露架构、不推测代码；EPLB 内置 P/D 在 release 可走读，§4）：

```
Request → Recipe（多协议 → 统一对话，§9）→ Entrypoint（CHECK 角色 hash-chain 前缀匹配，release 外）
   ├─ 图像 → VE LB（machine 随机 + fewest patches，§7）→ VE（单 GPU）
   ├─ prefill → Prefill LB（4 长度档 + compute-quota pull，§5）→ Prefill 引擎（EP2–8 / DP2–8）
   └─ decode → Decode LB（seq count + total KV length 双维 dispatch，§6）→ Decode 引擎（EP64–128 Wide-EP）
        每引擎内置 EPLB（冗余副本 + 统计窗 + 重分配 + symm_mem shuffle，§3/§4，release 内）
   KVCache Storage（3FS kvcache-store 专用集群，任何机命中任何前缀，release 外；2.4）
```

**本章口径**：模型几何取 V4.1-Flash config（[W]： $E$ = 384 routed + 1 shared、 $k$ = 6、 $d_{\mathrm{model}}$ = 5120、moe_intermediate_size 2304、expert MXFP4）；并行度记 $e$（EP）/ $d$（DP）， $B$ 为全局 batch（ $B = d\cdot B_{\mathrm{local}}$， $B_{\mathrm{local}}$ 单 DP 组本地 batch，1.2 §5 记号 recap）；带宽换算沿用 1.3 参数表的 B200 HBM 口径 7.7 TB/s（官方规格 [C]，参数表出处见 1.3，本章不另引 datasheet）。生产数字标「官方披露」+ doc 行号 + 核实日期 2026-11-23。边界：本章讲 serving 侧 EP 部署与负载均衡算法；RDMA / NVLink 硬件与网络拓扑归 cluster 维度，serving 并行组合（TP/PP/CP）归 8.1，弹性容错详展归 8.5。V4.1-Flash 模型侧（Engram / DSpark draft / CED）形式化归 models 维度（V4.1 待补），本章只引 config 字段做 serving 账。

## 2. Wide-EP 账兑现：从 $\propto 1/e$ 到生产数字

每卡 MoE 权重流量 $\propto E/e$、每卡 compute $\propto Bk/e$，所以每步时间 $\propto 1/e$ 直到通信主导——这是 1.2 埋的伏笔，本节用 V4.1 生产配置兑现。recap 两条已证命题（不重推）：1.2 命题 3（每卡流量 $`M_{\mathrm{rank}} = (E/e)\cdot W_{\mathrm{exp}}\cdot\mathrm{bytes}_{\mathrm{param}}\cdot L_{\mathrm{moe}}`$，对 $e$ 反比、每卡强度与 $e$ 无关、总流量不变）；3.3 命题 3（步时 $\tau(e) = \max(A/e, F) + g(e)$，拐点 $e^{\ast} = A/F$，过拐点后通信/同步主导）。所谓兑现，是把符号换成 V4.1 config 与官方 EPLB 配置的具体数。

先算单 expert 权重字节（serving 账，config 字段 [W]）。SwiGLU 三矩阵（gate / up / down）： $W_{\mathrm{exp}} = 3\times 5120\times 2304 = 35{,}389{,}440$ 参数；routed expert MXFP4（per-32 ue8m0，每元素 0.53125 字节，7.3 §4 格式账）→ 每 expert 每层 $35{,}389{,}440\times 0.53125 = 18{,}800{,}640$ B ≈ 18.8 MB。再代入官方 EPLB 冗余配置（§3）：EP8 prefill 每 GPU 4 冗余 → 物理 expert 总数 384 + 32 = 416、每卡 416/8 = 52 个；EP64 decode 每 GPU 1 冗余 → 384 + 64 = 448、每卡 448/64 = 7 个（doc L131「7 experts per GPU after rounding, excluding the shared expert」逐字，384/64 + 1 = 7 ✓）。每卡每 MoE 层权重流量与读时间下界（全激活口径，B200 7.7 TB/s）：

| 配置 | 物理 expert/卡 | 每卡每层权重流量 | 读时间下界 | 每 expert batch（ $B_{\mathrm{local}}$ = 256） |
| --- | --- | --- | --- | --- |
| EP8 DP8（prefill 冗余 4/卡） | 52 | 977.6 MB | 127.0 μs | 32 |
| EP64 DP64（decode 冗余 1/卡） | 7 | 131.6 MB | 17.1 μs | 256 |

流量比 52/7 ≈ 7.43×，读时间同比例缩——这就是 doc L61「Wide-EP substantially reduces per-step memory traffic in the MoE expert part and, with good EPLB support, effectively raises TPS」（官方披露）的账面。两个限定词对应命题前提：「good EPLB support」= 均衡路由（否则热卡流量与计算双超均值、步时被尾卡拖住，§3）；「per-step memory traffic」= 每卡口径（总流量不变）。每 expert batch 列用 #DP = #EP 耦合口径： $B = d\cdot B_{\mathrm{local}}$， $B_{\mathrm{exp}} = Bk/E$——EP64DP64 时 $B = 64\times 256 = 16384$、 $B_{\mathrm{exp}} = 16384\times 6/384 = 256$；EP8DP8 时 $B = 2048$、 $B_{\mathrm{exp}} = 32$。每卡收到的分配对数 $Bk/e = B_{\mathrm{local}}\cdot k = 1536$ 两配置相同（耦合扩张下每卡计算量不变，3.3 命题 3 的 $F$ 常数前提）——Wide-EP 减的是每卡权重流量，不是每卡计算；提强度靠耦合扩 DP 增全局 $B$（1.2 §5 reconciling 注、doc L295「Increasing EP raises arithmetic intensity」）。

官方 benchmark 图（doc L63 wide_ep_decode_pareto.png，一手读取）：EP64DP64「384+64 experts」vs EP8DP8「384+32 experts」，context 131,072，点标 = decode_num_requests/rank（256…8）。读图近似（**图读近似**，非逐字数值）：约 100 tok/s/user 处 EP64 ≈ 26.2k vs EP8 ≈ 24.8k tok/s/GPU（256 req/rank 点）；约 180–200 tok/s/user 处 EP64 ≈ 22.8k（128 点）vs EP8 ≈ 12.7k（64 点）≈ 1.8×——EP64 曲线全程压制 EP8。

> **坑**：把图副标题的 τ=3.25 与 4.4 几何级数混算。触发条件：见 wide_ep 图「nextn=3 fixed | per-token acceptance 75% → τ=3.25」就代入 4.4 式 (5)；现象：τ=3.25 是该图自有口径（逐位置独立接受近似，1 + 3×0.75 = 3.25），而 4.4 式 (5) 的前缀接受几何级数 $E[N_{\mathrm{tok}}] = (1-\alpha^{\gamma+1})/(1-\alpha)$ 在 $\alpha$ = 0.75、 $\gamma$ = 3 时 = 2.734——两数不同源（图是 benchmark 假设口径，run_sweep.sh decode 模式位置参数 nextn / draft_acceptance 即此假设）；规避：引图照抄图口径 + 一句「图为 benchmark 假设口径」，勿与 4.4 公式混算，也别与 3.3 步时 $\tau(e)$ 混记号。

EP > 32 递减（doc L293-297，官方披露，3.3 命题 3 的生产口径 recap）：「Increasing EP raises arithmetic intensity to avoid being memory-bound. Once memory bandwidth is no longer the bottleneck, a larger EP size introduces additional communication and synchronization overhead.」（L295）+「Different jobs sharing NVLink within a supernode can interfere with one another and slow down. Using the whole supernode for a single EP group avoids that interference.」（L297）。两笔代价：通信/同步随 $e$ 涨（ $g(e)$），超节点内多 job 共享 NVLink 互相干扰——整超节点作单 EP 组把干扰内部化。Mega MoE 把 $g(e)$ 重叠进计算、拐点外推（7.3 §5），但推不到无穷。另一笔 Wide-EP 代价是容错：大 EP 单通信域，单 GPU 故障 → 整实例不可用，重启从健康 GPU 重定 EP size（doc L249-265 官方披露，弹性/容错详展归 8.5，本章只记指针）。

## 3. EPLB 机制与生产参数

EPLB 把热 expert 复制多份摊到不同卡——复制谁、复制几份、多久重排一次，生产参数是什么？机制（doc L125 逐字，官方披露，2026-11-23 核实）：每 GPU 除负责的 expert 外另持若干冗余 expert 承载热 expert 副本；收集 last N steps 每 expert 负载，every M steps 按统计重分配 expert 到 GPU；expert 权重经 NVLink `symm_mem` 在组内 shuffle。生产参数（doc L127 / L131，官方披露）：

| 阶段 | 冗余/GPU | 统计窗 | 重分配周期 | 实测 moe_lb（slowest rank / mean） |
| --- | --- | --- | --- | --- |
| prefill | 4（rounded up 至 ep_size 倍数） | 100 步 | 1000 步 | 1.074（EP4DP4）/ 1.045（EP2DP2） |
| decode | 1（EP64 → 7 expert/GPU，excl shared） | 1000 步 | 10000 步 | 1.090 |

moe_lb 口径：最慢 rank 负载 / 平均 rank 负载（doc 逐字「slowest rank / mean」），是 EPLB **已生效后**的残差偏斜，不是裸偏斜。读数：若 MoE 部分步时 ∝ 最慢卡负载，则相对完全均衡（各卡 = 均值，perfectly balanced）的吞吐比 = 1/moe_lb——decode 1.090 → 91.7%（损失 8.26%），prefill 1.074 → 93.1%、1.045 → 95.7%（python 复核）。注意这与 §7 的「~80% of perfectly balanced offline」是两笔账：moe_lb 只量 MoE 部分 EP rank 间偏斜，80% 量端到端在线 vs 离线（含排队、DP 不均、SLO 操作点保守化等），勿混。

冗余副本为什么能降偏斜？先给理想化骨架。设 logical expert $j$ 窗口负载 $\ell_j$、副本数 $c_j$，理想均摊下每副本负载 $\ell_j/c_j$，记 $M = \max_j \ell_j/c_j$。

**命题 1**（冗余副本一步贪心）。向任意 expert 加一个副本不增 $M$；向每副本负载最大的 expert $j^{\ast}$（ $\ell_{j^{\ast}}/c_{j^{\ast}} = M$）加副本，使复制后的 $M$ 最小（一步贪心最优）。

**证明**（路线：逐副本负载单调 → 分最大者与非最大者两情形，三步）。其一，加副本使 $\ell_j/(c_j+1) \le \ell_j/c_j$，其余 expert 不变，故新最大值不超过 $M$——任意复制都不增。其二，若 $j^{\ast}$ 是唯一最大者，复制它后最大值变为「次大值与 $\ell_{j^{\ast}}/(c_{j^{\ast}}+1)$ 的较大者」，不超过 $M$。其三，复制任何非最大者 $j$ 时， $\ell_{j^{\ast}}/c_{j^{\ast}} = M$ 这项不动，故新最大值仍为 $M$——所以复制最大者使复制后的最大值最小。∎ 正例： $\ell = [8, 4]$、 $c = [1, 1]$，每副本 $[8, 4]$、 $M$ = 8；复制最大者（j=0）→ $[4, 4]$、 $M$ = 4；复制 j=1 → $[8, 2]$、 $M$ = 8——复制最大者严格更优。边界反例：「复制谁都一样」不成立（上例 j=1 不降 max）；「复制越多越均衡」也有界——副本总数受每卡物理槽位与显存约束（§4 rounding），且 $\ell_j$ 随路由漂移，静态最优不保持。

口径注：命题 1 是单步、理想均摊骨架。release 内 `policy.py` 的实际目标更细——冗余分配贪心最小化「加一个副本后**逐 step** 最大每副本负载之和」（窗口向量 [num_steps, layers, experts]，非单步标量），放置贪心 `_batch_greedy_indices` L12 另最小化 sum_step max_rank load（§4）。命题 1 给的是该贪心的单调性骨架，不是代码目标函数的逐字复述。

负载粒度为什么随 EP 变粗？第二条理想化命题。

**命题 2**（负载粒度）。设各 expert 负载独立同分布（均值 $\mu$、方差 $\sigma^2$），每卡持 $m = E/e$ 个 expert（无冗余理想化），则卡负载的变异系数对 $e$ 单调增：

$$
\mathrm{CV} = \frac{\sigma}{\mu\sqrt{m}} = \frac{\sigma}{\mu}\sqrt{\frac{e}{E}} \qquad (1)
$$

**符号**： $\mathrm{CV}$ = 卡负载变异系数（标准差 / 均值）； $m = E/e$ = 每卡 expert 数； $\sigma/\mu$ = 单 expert 负载的变异系数（理想化常数）。消歧：此 $\mu,\sigma$ 为命题 2 理想化负载分布的均值 / 标准差（局部记号），非 1.3 排队论的服务率 $\mu$。

**证明**（独立和的方差，两步）：卡负载 = $m$ 个独立 expert 负载之和，均值 $m\mu$、方差 $m\sigma^2$、标准差 $\sigma\sqrt{m}$；代入定义 $\mathrm{CV} = \sigma\sqrt{m}/(m\mu) = \sigma/(\mu\sqrt{m})$，再代入 $m = E/e$ 得式 (1)，对 $e$ 增。∎ 代入（ $\sigma/\mu$ = 1 示意）：EP2 CV ≈ 0.072、EP4 ≈ 0.102、EP64 ≈ 0.408——EP 越大、每卡 expert 越少、相对偏斜越大。方向与官方 moe_lb 一致：1.045（EP2）< 1.074（EP4）< 1.090（EP64），随 $e$ 单调升。口径注：iid 是理想化（热 expert 相关、EPLB 已改变分布），命题 2 给方向不给定量拟合；moe_lb 是 max/mean 非 CV，两者不可直接等同。这解释了 prefill 取小 EP（均衡好，doc L57）与 decode 大 EP 必须配 EPLB（粒度粗、偏斜大）的取向。

training 5.4 §4 负载不均衡 recap（不重推）：一层 MoE 墙上时间由最慢卡决定，偏斜度 $\max_j R_j/\bar{R}$ 乘性进层时长（木桶命题）；模型侧读数 router $\max_i f_i$ 与分布熵、系统侧读数 $\max_j R_j$ 是同一偏斜两端。V3 aux-loss-free 偏置控制器（步长 $\gamma = 10^{-3}$、seq 级 aux $\alpha = 10^{-4}$，training 5.4 记号 recap）把均衡写进路由控制回路；推理侧的系统兜底就是 EPLB 冗余副本——训练靠路由控制、推理靠容量复制。指针收窄（spec §2 errata 通道）：training 5.4 §4 与 models 4.5 §7 原「EPLB 系统实现归 cluster」句，spec §2 已裁定重判——serving 侧 EPLB（冗余副本/统计窗/重分配/moe_lb）归本章（8.1 recap），硬件/网络拓扑归 cluster；指针句收窄留 W4 closeout errata 执行（截至本章核实，training 5.4 §4 / models 4.5 §7 原文尚未改）。

## 4. EPLB 两套算法走读：repo 计划算法 ≠ release 引擎算法

deepseek-ai/EPLB repo 的 `eplb.py` 与 DeepInfer release 内 `model_executor/eplb/` 是同一件事吗？不是——两套算法须区分（研究笔记语义纠正 3，写作陷阱）。前者是 V3 时代的**计划算法**（只算复制 + 放置计划，统计窗/重分配周期/权重 shuffle 不在 repo），后者是 release 内的**引擎实现**（per-step 窗口向量 + 贪心 + symm_mem shuffle + moe_lb 统计）。SGLang `eplb_algorithms/deepseek.py` 镜像的是 EPLB repo 算法（W3 锚，6.3）。三列对照：

| 维度 | EPLB repo（eplb.py 164L） | DeepInfer release（model_executor/eplb/） | DeepEP 原语层 |
| --- | --- | --- | --- |
| 定位 | 复制 + 放置计划（负载预测出 repo 范围） | 统计窗 + 重分配 + shuffle + moe_lb 全引擎侧 | 冗余权重/梯度通信原语 |
| 冗余分配 | replicate_experts L44：迭代复制 weight/logcnt 最大者 | policy.py rebalance_experts_global L73：最小化逐 step 最大每副本负载之和 | — |
| 放置 | balanced_packing L5：降序贪心装最轻非满包 | _batch_greedy_indices L12：最小化 sum_step max_rank load | — |
| policy | rebalance_experts_hierarchical L74（组→节点→节点内复制→GPU）/ rebalance_experts L131（num_groups%num_nodes==0 走 hierarchical 否则 global） | 全局贪心（per-step 窗口） | — |
| 阶段口径 | hierarchical 用于 prefill 小 EP / global 用于 decode 大 EP（README） | 引擎按 is_prompt 分阶段 | — |
| shuffle | 不在 repo | shuffle.py intranode symm_mem L150 / internode NCCL P2P L65 | lb_prefetch_weights / lb_reduce_grads |

> **坑**：把 EPLB repo 算法当 DeepInfer release 算法读。触发条件：见「EPLB」就引 eplb.py 的 hierarchical/global 当引擎实现；现象：release 内是 per-step 窗口向量贪心（policy.py L73），目标函数与 eplb.py 的 weight/logcnt 迭代复制不同，照 repo 算法对引擎代码会落空；规避：计划算法锚 EPLB repo、引擎实现锚 DeepInfer eplb/、SGLang 镜像锚 repo 算法，三处分开标注（研究笔记语义纠正 3）。

release 内引擎走读（file:line，2026-11-23 fresh tarball 核实）。统计与重分配周期在 `eplb/__init__.py` **ExpertLoadBalancer L42**：统计窗 ring `_expert_count_window`（形状 [window_size, num_moe_layers, num_local_physical_experts]，L114 区）；首次 rearrange 提前到 1/4 周期（L121-123 注释逐字「首次 rearrange 提前到 1/4 周期后，尽早纠正冷启动的不均衡」，`initial_rearrangement_delay = max(1, interval//4)`）；moe_lb 统计 `_update_load_balance_stats` L167 每 `MOE_LB_STAT_INTERVAL` = 128 步（environ.py L125）经 `dist.all_gather_into_tensor` L188 收集各 rank 每步 token 数，算 max/mean（L190-193 存 mean_sum 与 max_sum）。窗与周期由 environ 配：`EXPERT_COUNT_WINDOW_SIZE` 默认 100（L122）、`EXPERT_REARRANGEMENT_STEP_INTERVAL`（L123）、`OVERRIDE_NUM_EXTRA_EXPERTS`（L121）——doc 的 prefill 100 步窗 / decode 1000 步窗即此族参数。

冗余数 rounding（`resolve_num_extra_experts` L21）：`padding = -(num_routed_experts + min_num_extra_experts) % ep_size`，返回 min_extra + padding——取不小于下限、且使物理 expert 总数被 ep_size 整除的冗余数。代入 decode（384 routed、min_extra 64、ep 64）：padding = −(384+64) % 64 = 0，总 448、每卡 7 ✓；prefill EP8（min_extra 32）：−416 % 8 = 0，总 416、每卡 52 ✓（python 复核，§2 表）。

权重 shuffle（`shuffle.py`）两路。intranode（`expert_model_parallel_shuffle_inplace_intranode` L150）：经 symm_mem 镜像直接读对端权重——`__init__.py` L206-220 建对称 buffer（`symm_mem.empty` + `symm_mem.rendezvous` + `handle.get_buffer` 取各 rank 镜像），shuffle 时逐层刷新复用（镜像只容一层权重）；卡内 origin→current 拷贝从自己镜像读避免覆盖，远端缺失 expert 从配对 sender 镜像拷（同一 expert 多槽只读一次），收尾同 expert 其余槽从已填首槽复制。NPU 无 symm_mem（`__init__.py` L199 TODO 逐字「NPU doesn't support symm_mem yet」，`_intranode_shuffle` 须 `is_within_nvl_domain` 且 `PLATFORM.is_gpu`）——Ascend 走 internode 路。internode（`expert_model_parallel_shuffle_inplace` L65）：NCCL batched P2P，按对端 rank 差 mod `REARRANGE_P2P_GROUPS`（environ L118 默认 1）分桶逐桶批量收发（L113-114）。通信原语层落点：DeepEP `EPBuffer.lb_prefetch_weights`（计算前经 NVLink 推原 expert 权重 + 量化 scale 到冗余槽）/ `lb_reduce_grads`（backward 冗余 expert FP32 梯度归约回原 expert）（7.3 §6 已走读接口，README L330-380）——引擎侧 shuffle 与 DeepEP 原语是同一冗余副本机制的两层（引擎管计划与本地拷贝，DeepEP 管跨域权重/梯度通信）。

## 5. Prefill LB：四长度档与 compute-quota pull

Prefill LB 均衡什么、为什么用 pull 不用 dispatch？先声明 release 边界：三 DP LB（prefill/decode/VE）与 Entrypoint CHECK、kvcache-store 编排**不在 repo**（全 repo grep 无 compute_quota / quadratic 代码，研究笔记语义纠正 7）——本节只讲 doc 披露架构，不推测代码。

架构四件（doc L137-143，官方披露，2026-11-23 核实）。其一，分档（L137）：请求按总上下文长分 4 档 <16K / 16–64K / 64–256K / ≥256K；每档引擎数按上轮 compute-quota 需求份额动态重分，各档独立 per-batch quota cap。分档让同档序列的已算前缀长 $c$ 接近，章 4.3 的 all-reduce MIN 保守性损失变小（4.3 §4 已述）。其二，跨 DP 组 pull（L139）：引擎按剩余 compute quota 拉工作；任何引擎可命中任何前缀（专用存储集群，2.2 / 2.4），所以**无 prefix-affinity 路由**——这是 pull 可行的前提（若有亲和，请求只能去持前缀的引擎，pull 退化为亲和路由）。其三，DP 组内（L141）：请求到当前 quota load 最低的 rank。其四，单 GPU（L143）：FIFO 出队，chunk size 不固定而从预算解出。

compute-quota 求解 recap（4.3 §4 式 (7)(8)，不重推）：每 prefill 步给时间预算 `compute_quota_limit`（ns），拟合二次成本模型 $\mathrm{cost}(q, c)$（ $q$ = chunk token 数、 $c$ = 已算前缀长，沿 doc 原记号、与 CP 度 $c$ 区分），解二次取最大 $q$ 不超 32,768-token 预算且恰耗尽剩余时间，预算再经 DP 组 all-reduce MIN 缩减。4.3 已给二次形来源（参数 GEMM 线性 + 前缀 attention 双线性 + 自 attention 二次）与正根求解，本章只登记生产口径。

> **坑**：compute-quota 两口径混写（研究笔记语义纠正 4）。触发条件：把 DualPath paper 与 DeepInfer doc 的求解方式当同一个；现象：DualPath paper §6.2（arXiv:2602.21548，[B]）= FIFO 加请求至 compute quota（attention 层预测时长上限）、超限**对 bsz 二分搜索**切 chunked prefill，(cached, bsz) 对 + profiling 拟合层时长；DeepInfer doc L143 生产口径 = 拟合二次 $\mathrm{cost}(q,c)$ **解二次**取最大 $q$ ≤ 32,768 + all-reduce MIN——两口径求解方式不同（二分 vs 解二次）；规避：两口径都给 + 漂移注（paper 2026-02 → doc v0.2 2026-11 生产细化；doc 引 DualPath 但实现口径不同），生产参数（32,768 / all-reduce MIN / 4 档）只以 doc 为准（paper 无这些数）。

pull 的调度语义（doc L159，官方披露）：prefill 引擎用当前步剩余时间决定何时拉新请求——剩余时间刚好够备批时拉一批，fetch/准备与当前步计算重叠，步结束时批已就绪，最小化 TTFT 同时保持引擎满利用。这是「控输出（步时）」而非「控输入（token 数）」的参数化（4.3 §4 末段 recap）：quota 是每步消耗品，解出的 $q$ 让时间成本贴预算走。pull vs dispatch 的结构性 rationale 见 §6 命题 3。

## 6. Decode LB：双维均衡与 dispatch

Decode LB 为什么必须用全局视图 dispatch，而 prefill 可以本地 pull？doc L147（官方披露，2026-11-23 核实）：Decode LB 在所有 decode 实例的所有 GPU 间均匀路由，原则是同时均衡两维——request count（决定每步 batch size）与 total KV cache length（决定 attention 部分成本）；启发式把短请求发往低并发但大 total KV 的 GPU、长请求发往小 total KV 的 GPU；「See DualPath's intra-group scheduling algorithm for details」。

两维对应 decode 步时的两笔成本（1.2 decode 账 recap）：attention 成本 ∝ total KV length（KV 读 $\kappa L\sum T$），FFN/MoE 成本 ∝ request count（每步 batch）。只均衡一维会偏：只均衡 seq count 则长序列堆在一卡、attention 拖尾；只均衡 KV length 则短请求堆在一卡、batch 维偏斜。数值例：GPU A 32 seqs / total KV 2M tokens，GPU B 64 seqs / total KV 0.5M——短请求（+1 seq、小 ΔKV）→ A（低并发 32、大 KV 2M），长请求（大 ΔKV）→ B（小 KV 0.5M），两维各趋均值。

dispatch 制（doc L161，官方披露）：请求一经调度留在原 decode 引擎至 finish（无迁移）；随请求量增长负载应在所有 decode 引擎间均匀增；Decode LB 收集各引擎 seq count + total KV length、算均值、路由使各引擎趋近均值——决策依赖全局视图，故由 Decode LB dispatch（而非引擎 pull）。术语注：doc 称「intra-group」、DualPath paper 称「intra-engine」（§6.2 Intra-Engine Scheduling），同指 DP/EP 组内调度（研究笔记 ⟨verify⟩ 销项）。DualPath（[Wu et al., 2026](../references.md)，arXiv:2602.21548，[B]，无官方 repo）§6.2 是此机制源头：DP attention 各 GPU attention 时长不齐 → 同步气泡，须使各卡 attention 层时长相近以最小化等待；paper 摘要数字 offline 至多 1.87× / online 平均 1.96× 不违 SLO、NIC 负载不均衡 1.53→1.18、attention Max/Avg 低至 1.06（paper 口径，非 DeepInfer 生产数）。

**命题 3**（pull vs dispatch 的信息局部性）。若工作单元可分割、可延迟领取且无状态迁移成本（prefill chunk），则本地剩余预算信号足以驱动负载均衡（pull）；若工作单元一经调度即绑定引擎并累积状态（decode 序列至 finish），则均衡须在放置时用全局视图（dispatch）。

**证明**（要点，结构性论证）。prefill 侧：请求可留共享队列、任何引擎可命中任何前缀（专用存储集群消除亲和，2.2 / 2.4），引擎按本地剩余 quota 拉取、fetch 与当前步计算重叠（doc L159）——无需全局协调，本地信号（剩余时间）充分。decode 侧：请求绑定后迁移须付 KV 传输成本（live migration 账 recap 4.2 Llumnix / 8.5，非免费），放置是一次性决策；要使各引擎 seq count 与 total KV length 趋近均值，须在放置时收集全局状态（doc L161）——全局视图必要。∎ 边界反例：若 decode 允许免费迁移则 dispatch 非必需（但 KV 迁移非免费）；若 prefill 有 prefix-affinity 则 pull 退化为亲和路由（DeepInfer 用专用存储集群消除亲和，命题前提成立）。口径注：命题 3 是设计 rationale 的结构论证，不是定量最优性定理；它解释「为什么两阶段用不同信息结构」，不给出路由算法的逐字实现（三 DP LB 在 release 外）。

## 7. VE LB 与在线/离线均衡口径

VE LB 怎么在不跑 encoder 的前提下估负载？doc L151（官方披露，2026-11-23 核实）：图像按 machine 粒度路由——prefill 机随机选一台「每 GPU 一个 vision encoder」的机器，机内把图像交给当前处理 patch 最少的 GPU；patch 数直接由尺寸算（宽 / patch_size × 高 / patch_size，doc 原式）。代入 V4.1 vision_config patch_size = 14（[W]）：448×448 图 → $(448/14)^2 = 32^2 = 1024$ patches；896×448 → $64\times 32 = 2048$ patches（python 复核）。fewest-patches 用尺寸直算负载、无需先跑 encoder——这是 VE LB 区别于 decode LB（须收集运行时 KV 状态）的轻量处。VE 单 GPU 部署、推荐便宜 GPU（doc L51），多模态 serving 详展归 9.4（image patch 账 + downsample + encoder cache 复用），本章只登记 LB 口径。

在线 vs 离线均衡（doc L155，官方披露）：「we reach roughly 80% of a perfectly balanced offline benchmark」——在线生产吞吐约为完全均衡离线基准的 80%。recap 1.1 §3：离线 benchmark（固定 batch、无到达过程、无 SLO）测产能上界，在线 goodput 受排队、负载不均与 attainment 约束；拿离线数字直接承诺在线产能会系统性高估约两成（容量规划折扣归 9.1）。口径分离（再强调）：80% 是端到端在线/离线比，moe_lb（§3）是 MoE 部分 EP rank 间残差偏斜——前者含后者但远不止后者，两数不可互推。四个 LB 合起来把在线推向离线，但到达动态与 SLO 操作点保守化留下 ~20% 缺口。

## 8. Engram ENP：vocab 维分片与 one-sided RDMA read

Engram 的 embedding 表大到单卡放不下，怎么分片又不让取数成为步时瓶颈？官方 doc L83-87（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）：Engram 用独立并行组 ENP（Engram parallelism），沿 vocabulary 维分片；prefill 用 ENP8，decode 用 ENP size = DP/EP size（L83）。GPU 侧 vocabulary 存 local host memory，每 rank 只持 1/ENP shard、该内存注册 RDMA read；forward 开始时一个 kernel 发 **one-sided RDMA read**，NIC 在后台把数据从 host memory 搬进 GPU memory；scales 足够小留 GPU memory、经 NVLink 传（L85）。NPU 侧 CPU offload 尚不支持，全部 Engram 数据留 NPU memory（L87）。

config 字段（[W]，只列不推导）：engram_layer_ids [1, 14]、engram_num_embeddings [384006168, 384016682]、engram_vocab_size 16,000,000、engram_max_ngram_size 4、engram_n_heads 8、engram_head_dim 256、engram_compressed_vocab_size 99092、engram_pad_token_id 2。分片算术（serving 账）：ENP8 时每 rank 持 engram_num_embeddings[0]/8 = 384,006,168/8 = 48,000,771 条（python 复核，整除）；decode ENP = DP/EP（如 64）时按 1/ENP 口径分片（doc 口径，非整除时由引擎处理，本章不虚构每 rank 精确条数）。模型侧 Engram（n-gram 记忆机制、compressed tokenizer 数学）归 models 维度（V4.1 待补），本章只讲 serving 分片与取数。

机制三笔账。其一，为什么 host memory + RDMA 而非全留 HBM：embedding 表条数亿级（384M），全留 HBM 挤占 KV cache 与权重；host memory 容量大、RDMA 注册后 NIC 直取，one-sided read 不占对端 CPU（发起方单方读，语义归 cluster）。其二，为什么 scales 留 GPU 经 NVLink：scales 小（相对 embedding 数据），留 GPU 避免每次取数都走 host→GPU，NVLink 域内传 scale 延迟低。其三，为什么 forward 开始即发 read：取数与计算重叠（NIC 后台搬），kernel 发起后不阻塞——与 2.4 GDR swap-in（RDMA NIC 直写 HBM 不占 SM）同思路，但 Engram 是 host→GPU 读、KV swap-in 是存储→GPU，方向与源不同。NPU 无 offload 的代码互证：environ.py L154 `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE = EnvVar(lambda v: _bool(v) or envs.DEEPINFER_ACCELERATOR == "ascend")`——Ascend 强制全留 NPU memory，与 doc L87 一致。

库层与引擎侧落点（映射级）。库层：DeepEP `EngramBuffer`（experimental，RDMA，GPU/CPU 多层存储 + 多层 fetch，`fetch(indices)` 每层返回 completion hook，7.3 §6 已走读接口，README L382+）——one-sided RDMA read 的库层落点。引擎侧：`model_executor/models/v4_modules/engram.py` CompressedTokenizer L35 / NgramEmbedding L69 / Engram L107 / EngramRuntime L164 + `tilelang_ops/gather_engram_tokens_zos.py`（ZOS 路径取 Engram token）——引擎管分片放置与取数调度，DeepEP 管 RDMA 通信。RDMA 硬件层归 cluster 维度。

## 9. deepseek-recipe：多协议统一与流式解析

请求从 Chat Completions / Responses / Messages 三种 API 进来，怎么共用一套模板与编码？deepseek-recipe（[DeepSeek-AI, deepseek-recipe](../references.md)，教学参照，repo + doc L354-372，2026-11-23 tarball 核实）是 Rust 实现的多协议适配层。Rust workspace crates：deepseek-recipe-{core, encoding, image, macros, python} + server-rs / server-py 示例；core/src/{conversation, messages, prompt, special_tokens, tools}.rs + encoding/src/v4/{dsv4, dsv41}.rs。四件机制（doc L354-372 官方口径逐字对应）：

其一，多协议 → 统一对话（L354）：把 Chat Completions / Responses / Messages 请求转成单一 Conversation 格式，共用一套模型模板与编码逻辑生成 prompt 与 token 序列；模型输出解析成 events（body text / reasoning content / tool calls）再转各协议流式响应；请求转换、prompt 编码、输出解析全 Rust 实现、Python bindings 易用（L356）。

其二，special-token 注入防护（L360-362，双向）。输入侧：建 prompt 时记录显式插入的 special token 及位置（`special_ranges` 机制，docs/tokenizer.md L65-76），编码时 special range 直映射 token ID、其间普通文本按 base vocabulary 编码；用户输入 / tool 结果 / tool-call 参数里出现的 `<｜User｜>`、`<｜image｜>` 等拼写按普通 token 序列编码——文本无法注入控制 token 影响模型行为。输出侧：按 token ID 识别 end-of-reasoning marker `</think>` 与 DSML tag 的 `｜DSML｜` special token；模型用普通 token 拼出的同样文本被正确保留为字面内容、不获得触发 end-of-reasoning 或 tool call 的能力。所谓注入防护，是指「控制 token 只能由系统显式插入、不能由用户文本伪装」的双向隔离。

其三，分段并行 BPE（L364）：建 prompt 时 special token 位置与边界已知，BPE 的普通文本段（special token 之间）可独立编码——跳过普通文本内的 special token 匹配与切分、各段并行编码（`tokenizers` crate 的 `TOKENIZERS_PARALLELISM`）、按原序拼接；降低超长上下文请求的 tokenization 时间、改善请求延迟。

其四，流式状态机 + 增量 KMP + DSML tool args（L368-372）。后端按 token 粒度返 chunk，单个 UTF-8 字符 / DSML tag / tool 参数可能跨多 chunk；流式解析须状态机保留进行中状态、及时返回已确定内容以降延迟。跨 chunk 解析：遇不完整 UTF-8 字符先缓冲待后续 token 补全再交解析器；解析器在 reasoning / body / tool-argument 三相位间维持状态（`output_parser.rs` State L376），delimiter marker 用增量 KMP 匹配（`kmp_table` L706 / KmpTable L704 / 调用 L680-691）——chunk 在 marker 中途结束时保留匹配进度、下一 chunk 续认，已确定内容随解析随发、无需重扫前文。tool 参数转换：tool 名解析出即可返回，随后按 DSML 参数边界直接生成 JSON 碎片——字符串值逐段 JSON-escape，数组/对象等非字符串值按原始 JSON 文本发出，无需构造完整参数对象、无重复解析与序列化；长参数也能边生成边返碎片，缩短首参数碎片等待。DSML open/close marker 定义在 L391-393。max_tokens 默认（README）：非 thinking 8192 / thinking 65536 / thinking + max effort 131072。流式输出工程 recap 1.4（incremental detokenization、UTF-8 跨 chunk），本节是 DeepSeek 生产形态的 repo 兑现。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| prefill 最小 EP/DP（HBM 允许）、decode 最大 EP/DP（GPU 允许）；#DP = #EP | doc L47/L49/L57/L33 | §1、§2 |
| Wide-EP 每卡流量 $\propto E/e$、每步 $\propto 1/e$ 至通信主导；EP8 52/卡 vs EP64 7/卡，流量比 7.43× | 1.2 命题 3 / 3.3 命题 3 recap；18.8 MB/expert | §2、练习 2 |
| 每 expert batch $Bk/E$；耦合扩 DP 提强度（每卡计算 $B_{\mathrm{local}}k$ 不变） | EP64 256 vs EP8 32 | §2、1.2 §5 |
| wide_ep 图 τ=3.25 自有口径（1+3×0.75）≠ 4.4 几何级数 2.734 | 图读近似 + 口径注 | §2 坑注 |
| EP>32 递减：通信/同步主导 + 超节点多 job NVLink 干扰；整超节点单 EP 组 | doc L293-297 | §2、3.3 |
| EPLB 冗余副本 + 统计窗 + 重分配 + symm_mem shuffle；prefill 4/100/1000、decode 1/1000/10000 | moe_lb 1.074/1.045/1.090 | §3、练习 1 |
| 冗余副本 argmax 一步贪心最优；负载粒度 CV $\propto\sqrt{e/E}$ | 命题 1 / 命题 2、式 (1) | §3、练习 3 |
| EPLB repo 计划算法 ≠ release 引擎算法（per-step 窗口贪心）；SGLang 镜像 repo | 三列对照表 | §4、练习 4 |
| Prefill LB 4 档 + compute-quota pull + 无 prefix-affinity；二次求解 recap 4.3 | doc L137-143；DualPath 二分 vs doc 解二次 | §5 |
| Decode LB seq count + total KV length 双维 dispatch；pull vs dispatch 信息局部性 | 命题 3；DualPath [B] §6.2 | §6 |
| VE LB machine 随机 + fewest patches $(w/14)(h/14)$；在线 ~80% 离线 | 1024/2048 patches；doc L151/L155 | §7 |
| Engram ENP vocab 分片 + host one-sided RDMA read + scales NVLink；prefill ENP8 / decode ENP=DP/EP | 48,000,771 条/rank；NPU 无 offload | §8 |
| deepseek-recipe 多协议统一 + 注入防护双向 + 分段并行 BPE + 流式 KMP/DSML | doc L354-372；max_tokens 8192/65536/131072 | §9 |

## 练习

1.（计算 · [10]）EPLB 冗余 rounding（`resolve_num_extra_experts` L21 公式）：(a) decode EP64、min_extra 64，求 padding、物理 expert 总数与每卡数；(b) prefill EP8、min_extra 32 同求；(c) prefill EP4、min_extra 16 同求。校验：(a) 每卡 7、(b) 每卡 52，与 doc L131 / L127 及 §2 表一致；(c) 每卡 100；公式 padding = −(384+min_extra) % ep_size。
2.（计算 · [20]）Wide-EP 流量账（V4.1 config [W]，B200 7.7 TB/s 沿用 1.3）：(a) 单 expert SwiGLU 三矩阵参数量与 MXFP4 字节（每元素 0.53125）；(b) EP8（52/卡）与 EP64（7/卡）每卡每 MoE 层权重流量与读时间下界，验证比 ≈ 7.43×；(c) #DP=#EP、 $B_{\mathrm{local}}$ = 256 时两配置的每 expert batch 与每卡分配对数，验证每卡计算不变；(d) 先预测 wide_ep 图 τ=3.25 与 4.4 式 (5)（ $\alpha$=0.75、 $\gamma$=3）是否相等再各算。校验：(a) 18.8 MB；(b) 977.6 / 131.6 MB、127.0 / 17.1 μs；(c) 256 vs 32、每卡 1536；(d) 3.25 ≠ 2.734（图口径 vs 前缀几何级数）。
3.（推导 · [30]）命题 1 闭环（冗余副本贪心）：(a) 补全三步证明（逐副本负载单调 → argmax 情形 → 非 argmax 情形）；(b) 给边界反例： $\ell=[8,4]$、 $c=[1,1]$ 时复制非 argmax 为什么不降 max；(c) 说明 release policy.py L73 的目标（逐 step 最大每副本负载之和）与命题 1 单步骨架的口径差。校验：(a) 与 §3 证明一致；(b) 复制 j=1 后 max 仍 8；(c) 对回 §4 走读（窗口向量 [num_steps, layers, experts]）。
4.（代码 · [30]）ExpertLoadBalancer 走读（`eplb/__init__.py`，2026-11-23 tarball）：(a) 统计窗 ring 的形状与默认窗长（environ `EXPERT_COUNT_WINDOW_SIZE`）；(b) 首次 rearrange 为什么提前到 1/4 周期（L121-123 注释），求 interval=1000 时的首次 rearrange 步；(c) moe_lb 怎么经 all_gather 算（L167-193，`MOE_LB_STAT_INTERVAL`=128），指出 max/mean 各对应哪行；(d) intranode symm_mem shuffle 与 NPU 的关系（L199 TODO）。校验：file:line 锚对研究笔记；(b) delay=interval//4=250、首次 rearrange 在第 250 步（计数器初值 interval−delay=750，每步 +1 至 interval=1000 触发，即 250 步后——「提前到 1/4 周期」）；(c) all_gather L188、mean/max L190-193；(d) NPU 无 symm_mem 走 internode P2P。

## 参考

- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——Wide-EP 取值与 rationale（L29/L47-53/L57/L61）、wide_ep 图（L63）、EPLB 机制与参数（L125/L127/L131）、Prefill LB（L137-143）、Decode LB（L147/L159/L161）、VE LB（L151）、~80% offline（L155）、Engram ENP（L83-87）、EP>32（L293-297）、弹性容错（L249-265）、Recipe（L354-372）
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——release 内 `model_executor/eplb/`（ExpertLoadBalancer L42 / policy.py L73 / shuffle.py L150/L65 / resolve_num_extra_experts L21）、environ.py（EXPERT_COUNT_WINDOW_SIZE 100 / MOE_LB_STAT_INTERVAL 128 / REARRANGE_P2P_GROUPS / ENGRAM_FORCE_GPU_STORAGE L154）、v4_modules/engram.py、tilelang_ops/gather_engram_tokens_zos；main@a679bb86 fresh tarball 2026-11-23；DP LB/Entrypoint/kvcache-store 不在 repo
- [DeepSeek-AI, EPLB](../references.md)：key: eplb，教学参照 + 官方实现——eplb.py 164L 四函数（balanced_packing L5 / replicate_experts L44 / rebalance_experts_hierarchical L74 / rebalance_experts L131）；hierarchical prefill 小 EP / global decode 大 EP；与 DeepInfer release eplb/ 算法不同；2026-11-23 tarball 全文核
- [DeepSeek-AI, DeepEP](../references.md)：key: deepep，教学参照 + 官方实现——EPBuffer.lb_prefetch_weights / lb_reduce_grads（L330-380）= EPLB shuffle 原语；EngramBuffer（L382+）= ENP 库层；2026-11-23 tarball 核实
- [DeepSeek-AI, deepseek-recipe](../references.md)：key: deepseek-recipe，教学参照 + 官方实现——Rust crates + special_ranges（docs/tokenizer.md L65-76）+ output_parser.rs（State L376 / kmp_table L706 / DSML marker L391-393）+ max_tokens 默认；2026-11-23 tarball 核实
- [Wu et al., 2026](../references.md)：DualPath，arXiv:2602.21548，[B]（key: dualpath，无官方 repo）——§6.2 Intra-Engine Scheduling = doc L147 compute-quota 源头（FIFO + bsz 二分，与 doc 解二次两口径）；摘要 1.87×/1.96×、NIC 1.53→1.18、attention Max/Avg 1.06；2026-11-23 abs+HTML 核实
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：key: v41-flash-config，[W]——E 384/k 6/d_model 5120/moe_intermediate_size 2304/expert fp4；engram_* 字段族；vision patch_size 14；模型侧形式化归 models（V4.1 待补）
