# 3.2 偏好与对比目标的接口（Interfaces of Preference and Contrastive Objectives）

> 本章做一件事：把后训练全部目标的接口形态定义清楚。偏好数据 $(x, y_w, y_l)$ 配 Bradley-Terry 模型 $P(y_w \succ y_l \mid x) = \sigma(r_w - r_l)$，从随机效用公理完整推导；reward 的平移不可识别性，即 $r$ 与 $r + c(x)$ 的等价类——9.3 DPO 重参数化的伏笔；对比目标接口（InfoNCE 一句 recap，链 models 6.2）；在线 / 离线目标的分界定义（P8 / P9 判据表的定义源）；最后是后训练目标一览表与 TRL DPOTrainer 的输入格式走读 | 前置：3.1（CE 训练侧口径、logsumexp 稳定形式）、models 1.1（NLL / CE）、models 6.2（InfoNCE 推导）| 锚点：TRL `dpo_trainer.py`（huggingface/trl repo，教学参照 + 官方实现）、Bradley & Terry 1952 [B]（无 arXiv）、Christiano et al. [B]、DPO [A]

## 1. 接口问题：后训练目标只用三种数据形态

SFT → RM → PPO 的三段链与 DPO 家族加起来，数据接口只有三种形态。先固定记号：prompt $x$、response $y$、策略 $\pi_\theta(y \mid x)$、参考策略 $\pi_{\mathrm{ref}}$；优势、组大小等 RL 记号到 8.x 再引入。三种接口形态：

1. **示范** $(x, y^{\ast})$：给定 prompt 的标准回答——SFT 的输入，目标就是 3.1 的 CE；
2. **pairwise 偏好** $(x, y_w, y_l)$：同一 prompt 下一对回答、标注员判定 $y_w$ 优于 $y_l$——RM 与 DPO 家族的输入（§2）；
3. **带奖励的采样** $(x, \lbrace y_i, r_i \rbrace)$：当前策略采出的回答与其标量奖励——PPO / GRPO 的输入（§5）。

为什么先把接口定下来再谈目标？因为接口决定目标函数能看见什么。示范接口只携带「正确回答是什么」；偏好接口只携带「同一 prompt 下谁更好」；采样接口携带「当前策略的行为分布加逐样本标量」。一个目标能表达什么偏好结构、能识别什么参数（§3 的不可识别性）、要不要在训练循环里养一个采样引擎（§5 的分界），都在接口选定时决定——目标函数的数学形态是接口的下游。

本章的任务是把定义固定下来：每个接口的概率模型是什么、目标函数的期望取自谁。各目标的完整推导、实现走读与失效模式归各自的定义章（§6 一览表给指针）。本章只有一处例外：Bradley-Terry 模型是全部偏好目标的公共件，它的推导就放在这里（§2）；它带出的不可识别性结论（§3），是 9.3 整条 DPO 推导链的伏笔。

## 2. Bradley-Terry：从随机效用到 $\sigma(r_w - r_l)$

问题：标注员给的不是分数而是二选一——怎么把「 $y_w$ 优于 $y_l$ 」接到一个实值 reward 函数 $r(x, y)$ 上？回答是成对比较文献里的标准模型（[Bradley & Terry, 1952](../references.md)，Biometrika 39(3/4):324-345，无 arXiv，[B]），从随机效用（random utility）出发三步走完。

**第一步，效用模型**。假设每个候选回答的「被选中倾向」由一个带噪声的潜变量驱动：

$$
U_i = r(x, y_i) + \varepsilon_i, \qquad \varepsilon_i \stackrel{\mathrm{iid}}{\sim} \mathrm{Gumbel}(0, 1) \qquad (1)
$$

**符号**： $r(x, y)$：确定性的偏好强度（reward）； $\varepsilon_i$：逐候选独立的 Gumbel 噪声，密度 $f(t) = e^{-t} \exp(-e^{-t})$，CDF $F(t) = \exp(-e^{-t})$。标注结果是 $y_w$ 胜出，即事件 $U_w \gt U_l$。

**第二步，Gumbel 差是 logistic**。两个 iid Gumbel 之差的 CDF 有闭式。对 $u \in \mathbb{R}$：

$$
\Pr(\varepsilon_l - \varepsilon_w \le u) = \int_{-\infty}^{\infty} F(t + u) \thinspace f(t) \thinspace \mathrm{d} t = \int_{0}^{\infty} e^{-(1 + e^{-u}) \thinspace s} \thinspace \mathrm{d} s = \frac{1}{1 + e^{-u}} = \sigma(u) \qquad (2)
$$

**符号**： $\sigma(u) = 1 / (1 + e^{-u})$：logistic 函数。第一步是卷积（固定 $\varepsilon_w = t$ 求条件概率再积分）；第二步代入 $s = e^{-t}$（ $\mathrm{d} s = -e^{-t} \mathrm{d} t$，积分限翻转），被积函数合成 $e^{-s \cdot (1 + e^{-u})}$；第三步是指数分布的积分。三步合起来：

$$
P(y_w \succ y_l \mid x) = \Pr(U_w \gt U_l) = \Pr(\varepsilon_l - \varepsilon_w \lt r_w - r_l) = \sigma\big( r(x, y_w) - r(x, y_l) \big) \qquad (3)
$$

这就是 Bradley-Terry 模型：成对偏好概率 = sigmoid 作用于 reward 差。

代入数字校准直觉。 $\Delta = r_w - r_l$ = 0.693 时 $P$ = 2/3：0.69 个 nat 的 reward 差对应 2:1 的胜率。 $\Delta$ = 2.2 时 $P$ ≈ 0.9。反向读：reward 的单位是 log-odds，差 1 nat 等于 odds（ $P / (1 - P)$）乘 $e$。这个量纲校准后面会反复用到：RM 输出的「分差」、DPO 里 $\beta$ 的语义（把策略对数比折算回 reward 的汇率），都是同一杆秤。

**第三条路（outline）**：同一个式子可以从公理而非噪声模型推出。Luce 选择公理要求：任意候选集 $S$ 上的选择概率满足「两候选的比值与集合里其余成员无关」—— $P_S(y) / P_S(y') = w(y) / w(y')$， $w$ 为正值函数。归一化给出 $P_S(y) = w(y) / \sum_{y' \in S} w(y')$；取 $w = e^{r}$、 $S = \lbrace y_w, y_l \rbrace$，即退化回式 (3)。两条路的分工不同：Gumbel 路径给出生成机制（标注员是带噪效用最大化者），Luce 路径给出公理化刻画（选择一致性），它们在同一个 sigmoid 上会合。这条 outline 的意义在 P9：当偏好数据违反 Luce 公理时（位置偏置、标注噪声异质、候选集构成本身改变两两比值），式 (3) 的适配度就有明确的公理边界可查——RM 目标失配时，先回这里查假设。

RM 训练用的正是逐对形式。reward model $r_\theta$ 在偏好数据集 $D$ 上最小化负对数似然：

$$
L_{\mathrm{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \right] \qquad (4)
$$

出处为 RLHF 的原始流程（[Christiano et al., 2017](../references.md)，arXiv:1706.03741，[B]）；RM 的完整训练流程与失效模式归 9.2。式 (4) 的梯度值得写一行。记 $\Delta = r_\theta(x, y_w) - r_\theta(x, y_l)$：

$$
\frac{\partial L}{\partial r_\theta(x, y_w)} = \sigma(\Delta) - 1 = -P(\text{模型当前判反}), \qquad \frac{\partial L}{\partial r_\theta(x, y_l)} = 1 - \sigma(\Delta) \qquad (5)
$$

**符号**：逐样本对 $r_w$ 的下压力恰是「模型此刻把这对判反的概率」——已经判对的样本（ $\Delta \gg 0$）梯度趋零，判反的样本满力拉动。难样本加权不是额外设计，是 sigmoid 损失的导数形态自带的（对照 3.1 §4 末 focal loss 的手工调制：那里要把这个性质显式乘进去，这里它是内生的）。

式 (4) 还有一条结构注记：它是 pairwise 接口上能写出的最简判别目标。把式 (4) 与 3.1 式 (1) 并排看。CE 是「词表 $V$ 类的分类」，BT 是「每对候选上的二分类」；两者的 logits 都只是 reward（或 logit）的差，归一化都靠 softmax / sigmoid 家族，梯度同为「预测 − 目标」形态（1.2 式 (5)）。后训练在数学上没有任何新算子，全部复用 P1-P3 的机制；新的是接口（什么进 loss）与采样语义（§5）。这也是本章只讲这两件事的原因。

统计性质记一句。式 (4) 是 BT 模型下的极大似然：模型正确设定时，它是 reward 差的一致估计。loss 曲线的地板由偏好噪声（标注不一致率）决定，不为零——与 3.1 §1 的 CE 地板同型（数据本身的不可约随机性）。标注一致率的估计方法（重复标注、标注员间一致性）归 9.2 的数据节。

## 3. reward 的平移不可识别性

式 (3) 只含 $r$ 的差。对任意只依赖 prompt 的函数 $c(x)$：

$$
\sigma\big( (r_w + c(x)) - (r_l + c(x)) \big) = \sigma(r_w - r_l) \qquad (6)
$$

也就是说， $r$ 与 $r + c(x)$ 产生**完全相同**的偏好分布。pairwise 数据对 reward 的识别只能到等价类 $[r] = \lbrace r + c(x) : c \text{ 任意} \rbrace$——逐 prompt 的加性常数是自由规范（gauge）。这不是病态边角，而是 BT 模型的结构性质：偏好数据只编码「同一 prompt 内谁比谁好」，任何关于「绝对多好」的信息都不在数据里。

一个两 prompt 算例把这件事说透。设真实 reward 为 $r$：prompt $x_1$ 下 $r(x_1, a) = 2$、 $r(x_1, b) = 1$；prompt $x_2$ 下 $r(x_2, c) = 5$、 $r(x_2, d) = 4$。取 $c(x_1) = 10$、 $c(x_2) = -3$，则 $r' = r + c(x)$ 在两条 prompt 下给出逐位相同的偏好概率（ $\sigma(1)$ ≈ 0.731）。任何规模的 pairwise 数据都无法区分 $r$ 与 $r'$；但 $r'$ 之下，「 $x_1$ 的回答普遍好于 $x_2$ 的」这类跨 prompt 断言被整个翻转。数据里没有的东西，再多的数据也学不出来。

三个后果。其一，**RM 分值无跨 prompt 语义**。 $r(x_1, y_1) \gt r(x_2, y_2)$ 这类跨 prompt 比较的真假，不被任何 pairwise 数据约束。用 RM 分值的绝对值做阈值过滤（例如「只留 r > 3 的样本」），假设了数据中本没有的可比性；合法用法只有同 prompt 内的排序（9.2 的 reward hacking 分析从这里入口）。其二，**gauge 恰好是 DPO 闭式解里的 $\log Z(x)$**。KL 约束目标的最优策略满足 $\pi^{\ast}(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x) \exp(r / \beta)$，反解出 $r = \beta \log(\pi^{\ast} / \pi_{\mathrm{ref}}) + \beta \log Z(x)$，其中 $\log Z(x)$ 正是逐 prompt 平移项。代入式 (3) 后它逐字消去——reward 由此可重参数化为策略对数比，而不需要物化配分函数（[Rafailov et al., 2023](../references.md)，arXiv:2305.18290，[A]；完整推导链归 9.3）。本节只登记伏笔：DPO 能成立，数学上就是式 (6) 的不可识别性。其三，**工程语义**。想给 RM 分值加绝对含义（校准曲线、跨 prompt 阈值），必须引入 pairwise 之外的信息源。数据形态决定什么可识别，先查形态再设计指标。

## 4. 对比目标接口

InfoNCE 的完整推导（batch 相似度矩阵上的双向 CE、温度参数化、uniformity 分析）归 models 6.2，此处只登记接口形态：

$$
L_{\mathrm{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(s_{ii})}{\sum_{j=1}^{B} \exp(s_{ij})} \qquad (7)
$$

**符号**： $s_{ij}$：anchor $i$ 与候选 $j$ 的相似度 logit（CLIP 口径为归一化内积除温度，models 6.2）； $B$：batch 大小。接口三元组是（anchor, positive, negatives），而负样本供给就是 batch 本身——「类别数」等于 batch 大小、随 batch 逐批漂移（models 6.2 §2 口径）。

训练侧工程只有一句：负样本数即 batch 大小，跨卡训练时负样本池靠 all-gather 扩充（通信原语归 4.1；CLIP 的大 batch 口径归 models 6.2）。与 §2 的结构对照值得记一笔：BT 是两个候选上的对比，InfoNCE 是 $B$ 个候选上的对比——同一棵 softmax 树上的两片叶子。梯度同为 $p - e_y$ 形态（1.2 式 (5)、3.1 §1），数值纪律（logsumexp 稳定、fp32 保护）也整套沿用 3.1。后训练对对比目标的消费在 P9 主要不是直接作为策略目标，而是作为数据构造与评估工具（检索负样本、构造偏好对）——这里把接口登记清楚即可。

## 5. 在线 / 离线：目标对采样分布的依赖

判据只有一条：看目标期望号下的采样分布是谁。

- **on-policy（在线）**：目标的期望取自当前策略 $\pi_\theta$（或与参数同步更新的 rollout 策略 $\pi_{\theta_{\mathrm{old}}}$）。参数每更新一次，旧样本的分布假设就失效，必须重采。
- **off-policy**：数据来自旧策略或他策略，期望号下是 $\pi_{\theta_{\mathrm{old}}}$ 而非 $\pi_\theta$——需要重要性比校正或接受偏差（P8 展开）。
- **offline（离线）**：目标只依赖静态数据集 $D$； $\pi_\theta$ 可以出现在被求值的函数内部（如 $\log \pi_\theta(y \mid x)$ 对固定的 $y$ 求值），但没有任何一步从 $\pi_\theta$ 采样。

注意，判据落在「采样」而非「求值」上。offline 目标照样把 $\pi_\theta$ 当作函数反复查询——DPO 每步都要算 $\log \pi_\theta(y_w \mid x)$——它免除的只是「从 $\pi_\theta$ 生成新样本」这一个动作。这一个动作的存废决定系统形态：on-policy 目标必须配 rollout 基础设施（推理引擎、权重同步、staleness 管理，账归 8.5），offline 目标一台纯训练栈就够。

三个实例把定义坐实。SFT： $`\mathbb{E}_{(x, y^{\ast}) \sim D}[-\log \pi_\theta(y^{\ast} \mid x)]`$， $\pi_\theta$ 只在被求值位置——offline。DPO：期望取自静态偏好集 $D$， $\pi_\theta$ 出现在 sigmoid 内部（§6 的走读）——offline。它不需要 rollout 引擎，这是它对 PPO 的工程折价所在。PPO：surrogate 的期望取自刚采的 rollout 批 $\pi_{\theta_{\mathrm{old}}}$， $\theta_{\mathrm{old}}$ 与 $\theta$ 只差若干次更新，陈旧性由 clip 管理——on-policy 口径（8.3 展开）；GRPO 的同 prompt 组采样同属此格（8.4）。

一个边界情形值得登记：拒绝采样微调（RFT）每轮先用当前模型采样、筛分、再 SFT。单轮内部是 offline——数据采完即冻结，训练不再采样；跨轮看，数据分布随策略漂移，又带 on-policy 的味。按本章判据它落 offline：每一轮的目标函数里，没有以 $\pi_\theta$ 为采样分布的期望。「数据来自谁」与「目标期望取自谁」是两件事，分界定义只管后者（RFT 的完整形态归 9.4）。这条分界的全部用途在 P8 / P9：样本效率、staleness 校正、要不要为 rollout 单独养一个推理引擎，都先由「期望取自谁」决定（判据表归 9.4）。

## 6. 后训练目标一览与 TRL 接口走读

把全部目标摆上同一张表（本章只立接口；定义章指针见末列）：

| 目标 | 数据接口 | 采样分布 | on / off | 定义章 |
| --- | --- | --- | --- | --- |
| SFT-CE | $(x, y^{\ast})$ 示范 | 静态语料 | offline | 9.1（CE 机制归 3.1） |
| RM-BT | $(x, y_w, y_l)$ | 静态偏好对 | offline | 9.2（BT 模型归本章式 (3)(4)） |
| PPO | $(x, y, r)$ + 每轮 rollout | $\pi_{\theta_{\mathrm{old}}}$ | on-policy | 8.3 |
| GRPO | $(x, \lbrace y_i, r_i \rbrace_{i=1}^{G})$ | 当前策略组采样 | on-policy | 8.4 |
| DPO | $(x, y_w, y_l)$ + $\pi_{\mathrm{ref}}$ | 静态偏好对 | offline | 9.3 |
| KTO | $(x, y, \pm)$ 单边标签 | 静态 | offline | 9.3 |
| SimPO | $(x, y_w, y_l)$，无 ref | 静态偏好对 | offline | 9.3 |

读法。离线一列的四个目标共享 §2 的偏好接口或其退化形态（KTO 退化为单边）；在线两格共享 §5 的 rollout 口径。 $\pi_{\mathrm{ref}}$ 的出现（DPO 行）不改变 offline 判定：它是被求值的固定函数，不是采样源。

表的用法：拿到一个新目标，先归格——数据形态落进三种接口的哪一种、采样分布是谁。两个答案定了，它的实现需求（要不要 rollout 引擎、要不要 ref 模型、batch 结构怎么拼）就跟着定了。P8 / P9 的每一章都是这张表一行的展开：RL 侧（8.3 / 8.4）把「采样」行的数学补齐（策略梯度、clip、组基线），后训练侧（9.1-9.5）把「静态数据」行的工程补齐（数据配比、失效模式、配方）。

接口的实现形态以 TRL 为准走读一遍（huggingface/trl repo，教学参照 + 官方实现；`trl/trainer/dpo_trainer.py`，2026-11-17 核实）：

```
(x, y_w, y_l)  ──tokenize──►  {prompt_ids, chosen_ids, rejected_ids}
                                 │  DataCollatorForPreference
                                 ▼
            batch = [chosen 半批; rejected 半批]（input_ids = prompt + response 拼接）
            completion_mask：prompt 段 0 / response 段 1
                                 │  policy π_θ 与 ref π_ref 各一次前向
                                 ▼
            fused LM head（add_fused_lm_head 补丁）：分 tile 过输出头、按 label 直接产出逐 token logp，logits 不物化
                                 │  按 completion_mask 对 response 段求和 → 序列级 logp
                                 ▼
            logratio = logp_policy − logp_ref（chosen / rejected 各一）
                                 ▼
            L = −log σ(β · (logratio_w − logratio_l))    ← 式 (3) 代入 DPO 重参数化，推导归 9.3
```

代码级事实四条（均核到行）：

1. **数据格式**：dataset 三键 `"prompt"` / `"chosen"` / `"rejected"`（plain text 或 conversational 两种形态，docstring 口径）；tokenize 后样本三键 `"prompt_ids"` / `"chosen_ids"` / `"rejected_ids"`。
2. **拼批**：`DataCollatorForPreference` 输出的 `input_ids` 前半为 chosen、后半为 rejected（docstring 原文："The first half of the batch corresponds to the `"chosen_ids"` and the second half to the `"rejected_ids"`"）。`completion_mask` 逐 token 标出 response 段——3.1 §1 的 mask 语义在这里复用，prompt 段不计入序列级 logp。另有可选的 `ref_chosen_logps` / `ref_rejected_logps` 预计算键：开 `precompute_ref_log_probs` 后，ref 模型只在训练前跑一遍，训练期不驻留。
3. **损失**：`_compute_loss` 内 `chosen_logratios = chosen_logps - ref_chosen_logps`。默认 `loss_type="sigmoid"` 时 `loss = -F.logsigmoid(beta * delta_score)`；`delta_score = chosen_scores - rejected_scores`，默认 `f_divergence_type="reverse_kl"` 分支下 scores 即 logratios，`delta_score = chosen_logratios - rejected_logratios`。这正是式 (3) 的 BT 形式代入 DPO 重参数化后的样子（9.3 给出从 §3 式 (6) 到这行的完整链）。
4. **数值纪律**：逐 token logp 走 fused LM head。训练器初始化时经 `add_fused_lm_head`（`trl/trainer/utils.py`）给模型打补丁，前向变为 `model(..., labels=labels, fused_lm_head=True).log_probs`：分 tile 过输出头、只产出 label 非 −100 位置的 log-prob，返回 `FusedCausalLMOutput`，全程不物化 $(B, T, V)$ logits（该函数 docstring 口径，TRL main 2026-11-17 核实；2026-10 前为 `selective_log_softmax` 的 gather 方案，库内演化注意版本）。此形态与 3.1 §3 的 FLCE「CE 前移进投影」同型：3.1 的 logsumexp 纪律在后训练栈里以 fused 形式再现。

走读之外补两条机制注记。其一，chosen 与 rejected 拼进同一个 batch 不是收纳癖。两条序列共享 prompt 前缀，对半拼接让一次前向同时产出两支的 logp，kernel 利用率与显存占用都按一个 batch 计；`ref_logps.chunk(2, dim=0)` 的注释（"batch is [chosen, rejected]"）就是这个布局的断言。其二，序列级 logp 是对 completion 段**求和**而非求平均（源码 `sum(dim=1)` 口径）。长回答的总 logp 天然更低，DPO 目标因此对长度有隐式偏置；把求和换成平均正是 SimPO 的平均 logp 隐式 reward 的出发点之一（归 9.3 的家族对照）。接口层的归约选择（sum vs mean）在这里又一次改变目标语义，与 3.1 §1 的 mean 归约注记同源。

显存与吞吐的接口账记一句：pairwise 接口让每个样本吃两次前向（chosen + rejected），ref 再计两次（或预计算摊掉）。batch 里 $B$ 个偏好对对应 $2B$ 条序列的前向，这是偏好目标相对 SFT 的固有折价；RL 目标（PPO / GRPO）的四模型驻留账更大，归 8.5，两笔账独立。

## 总结

| 主题 | 公式 | 要点 |
| --- | --- | --- |
| Bradley-Terry | 式 (1)-(3) | Gumbel 差 = logistic ⇒ $P(y_w \succ y_l \mid x) = \sigma(r_w - r_l)$；Luce 公理 outline 与噪声模型同式会合 [B] |
| RM 目标 | 式 (4)(5) | 逐样本梯度力 = 模型当前判反概率（难样本加权内生）；Christiano et al. 出处 [B] |
| 平移不可识别 | 式 (6) | 等价类 $[r] = \lbrace r + c(x) \rbrace$； $\log Z(x)$ 即 gauge，DPO 重参数化的数学支点（9.3）[A] |
| InfoNCE 接口 | 式 (7) | （anchor, positive, batch 内负样本）；类别数 = batch 大小；推导归 models 6.2 |
| on / off 分界 | 期望号下的采样分布 | on-policy = 期望取自 $\pi_\theta$；offline = $\pi_\theta$ 只在被求值函数里；判据表归 9.4 |
| TRL DPO 接口 | 三键 + 对半拼批 | `prompt`/`chosen`/`rejected`；loss = $-\log \sigma(\beta \cdot \Delta\mathrm{logratio})$；fused LM head 与 3.1 FLCE 同型 |

## 练习

1.（推导）补全式 (2)：写出代入 $s = e^{-t}$ 的完整两步（含积分限翻转）；再证对称形式 $P(y_l \succ y_w \mid x) = 1 - P(y_w \succ y_l \mid x)$，说明这条对称性为什么必须先验成立。
2.（推导）平移不可识别的逆命题：设 $r$ 与 $r'$ 在全部 $(x, y_w, y_l)$ 上给出相同的 BT 概率，证明 $r'(x, y) - r(x, y)$ 只依赖 $x$（提示：固定 $x$，取三个候选两两配对，消去公共差）。
3.（计算）取 $\Delta$ = 2.0 / 0.5 / 0.0，各算式 (4) 的逐样本损失与式 (5) 的梯度；据此解释 RM 训练为什么不需要显式的难样本挖掘模块。
4.（代码）按 TRL 接口手搓最小 DPO step：构造两个偏好对，按 `DataCollatorForPreference` 的语义拼 `input_ids` 与 `completion_mask`，用一个小 LM 分别前向 policy 与冻结的 ref，gather 序列级 logp 并算出与 $-\log \sigma(\beta \cdot \Delta)$ 一致的 loss（教学用，非优化实现）；对照 TRL 的 batch 布局断言你的拼接顺序。

## 参考

- [Bradley & Terry, 1952](../references.md)：Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons，Biometrika 39(3/4):324-345（无 arXiv），[B]（key: bradley-terry）——式 (3) 的原始出处
- [Christiano et al., 2017](../references.md)：Deep Reinforcement Learning from Human Preferences，arXiv:1706.03741，[B]（key: christiano）——式 (4) 的 RM 训练目标出处
- [Rafailov et al., 2023](../references.md)：Direct Preference Optimization: Your Language Model is Secretly a Reward Model，arXiv:2305.18290，[A]（key: dpo）——重参数化与 $\log Z(x)$ 消去（完整推导归 9.3）
- [HuggingFace, huggingface/trl repo](../references.md)：教学参照 + 官方实现（key: trl-repo）——`trl/trainer/dpo_trainer.py`（数据三键、`DataCollatorForPreference`、sigmoid 损失）、`trl/trainer/utils.py`（`add_fused_lm_head`，fused LM head 补丁）
