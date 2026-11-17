# 8.3 PPO 完整推导（Proximal Policy Optimization: A Full Derivation）

> 本章把 PPO 拆到底：on-policy 的数据浪费账 → importance sampling 与 surrogate objective 的一阶近似性质 → TRPO 的单调改进界与 KL 硬约束（共轭梯度求解只给结论与账）→ clip 目标的逐情形推导与悲观下界性质 → 多 epoch 复用的陈旧偏差分析 → 完整算法流程与实现细节清单（逐条配失效模式）→ CleanRL `ppo.py` 逐行走读 | 前置：8.1（策略梯度定理、baseline 无偏性）、8.2（GAE、actor-critic 联合 loss）；LLM 侧的 RLHF 全链归 9.2、系统实现归 8.5 | 锚点：PPO 原文（[Schulman et al., 2017](../references.md)，arXiv:1707.06347，[B]）、TRPO（[Schulman et al., 2015](../references.md)，arXiv:1502.05477，[B]）、GAE（[Schulman et al., 2016](../references.md)，arXiv:1506.02438，[B]）、CleanRL `cleanrl/ppo.py`（教学参照，主走读）、OpenRLHF（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]，前瞻引用）

## 1. on-policy 的代价：一批数据只换一次梯度

8.1 的策略梯度 $`\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace A_t]`$ 里，期望号下的分布是**当前策略自己**——梯度估计所需的样本必须由被优化的那个分布产生。参数每更新一步，旧数据的分布假设就失效一步：一批 rollout 支撑一次梯度更新即作废，样本复用率约等于 1。这不是实现不够聪明，是 on-policy 算法的定义性约束。

代价账在 LLM 场景更难看：一条轨迹是一次完整的长序列生成（RLHF 里是整段 response），rollout 成本远高于一步梯度；监督学习里同一批数据过几十个 epoch 是常态，RL 里却是「采一批、用一次」。量级直觉：一条 1024 token 的 response 的生成成本约等于它在训练侧的一次前向加一次反向的三分之一（ $6N$ FLOPs/token 训练口径对 $2N$ FLOPs/token 推理口径，models 1.2 的账），但生成必须串行逐 token 走完，训练侧的矩阵乘可以并行摊满——同一 token 数的 rollout 墙钟时间往往是更新段的数倍到数十倍。8.5 会把 rollout/train 的引擎拆分与权重同步做成正式账，本章只需要这个方向感：样本贵、梯度相对便宜，多花几次梯度换样本是划算的买卖。8.2 的 GAE 解决的是 advantage 估计的偏差-方差权衡，没有解决数据来源问题。后者要回答的问题是：能否用旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采的数据，估计新策略 $\pi_\theta$ 的改进方向？答案分两层：importance sampling 给出修正工具（§2），TRPO 给出「修正多远之内可信」的界（§3），PPO 把界换成一阶的廉价截断（§4-5）。

## 2. Importance sampling 与 surrogate objective

策略性能有一个精确恒等式作起点。记折扣状态访问分布 $\rho_\pi(s) = \sum_{t=0}^{\infty} \gamma^{t} \Pr(s_t = s \mid \pi)$，TRPO 论文（[Schulman et al., 2015](../references.md)，arXiv:1502.05477，[B]，其 §2 式 (2)）给出：

$$
\eta(\tilde{\pi}) = \eta(\pi) + \sum_{s} \rho_{\tilde{\pi}}(s) \sum_{a} \tilde{\pi}(a|s) \thinspace A_\pi(s, a) \qquad (1)
$$

**符号**： $\eta(\pi)$：策略 $\pi$ 的期望折扣回报； $A_\pi(s, a)$：旧策略的优势函数（8.2）。新策略的性能 = 旧性能 + 一项按**新策略自己走出来的状态分布**加权的期望优势。该恒等式蕴含策略迭代的经典结论（逐状态期望优势非负则 $\eta$ 不降），但不直接给算法： $\rho_{\tilde{\pi}}$ 对 $\tilde{\pi}$ 的依赖无法采样求解。

局部近似把右端的 $\rho_{\tilde{\pi}}$ 换成 $\rho_\pi$：

$$
L_\pi(\tilde{\pi}) = \eta(\pi) + \sum_{s} \rho_\pi(s) \sum_{a} \tilde{\pi}(a|s) \thinspace A_\pi(s, a) \qquad (2)
$$

代价是忽略状态分布漂移；收益是**一阶匹配**： $L_\pi(\pi) = \eta(\pi)$ 且 $\nabla_\theta L|_{\theta_{\mathrm{old}}} = \nabla_\theta \eta|_{\theta_{\mathrm{old}}}$（Kakade-Langford 的经典结论，TRPO 引为其式 (4)）。在旧策略的无穷小邻域内，最大化 $L$ 就是最大化 $\eta$——surrogate objective（替代目标）的全部合法性来自这一阶匹配，离开邻域后的误差由 §3 的界控制。

采样形式用 importance sampling 把动作期望搬到旧策略下。由 $`\sum_a \pi_\theta(a|s) A(s,a) = \mathbb{E}_{a \sim \pi_{\mathrm{old}}}[\thinspace \pi_\theta(a|s)/\pi_{\theta_{\mathrm{old}}}(a|s) \cdot A(s,a) \thinspace]`$，写成轨迹形式：

$$
L_{\theta_{\mathrm{old}}}(\theta) = \eta(\theta_{\mathrm{old}}) + \mathbb{E}_{\tau \sim \pi_{\theta_{\mathrm{old}}}} \left[ \sum_{t} \gamma^{t} \thinspace \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)} \thinspace \hat{A}_t \right] \qquad (3)
$$

**符号**： $`\hat{A}_t`$ 为 rollout 后用 GAE 一次性算好并冻结的优势估计（8.2）。IS 在此处只纠**动作**分布——比率 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\mathrm{old}}}(a_t|s_t)$ 把「动作采自谁」修正掉；**状态**分布仍是 $\rho_{\mathrm{old}}$，式 (2) 的近似原样保留。这就是 IS 在 PPO/TRPO 语境里的精确语义：动作失配有硬修正，状态失配靠「新旧策略足够近」的假设兜底。

IS 自身的失效模式也要记账，而且两条都能写成式子。**方差账**：ratio $r = \pi_\theta / \pi_{\mathrm{old}}$ 在旧策略下恒有 $`\mathbb{E}_{\mathrm{old}}[r] = 1`$；其二阶矩

$$
\mathbb{E}_{\mathrm{old}}[r^{2}] = \sum_{a} \frac{\pi_\theta(a|s)^{2}}{\pi_{\mathrm{old}}(a|s)} = 1 + D_{\chi^{2}}(\pi_\theta \Vert \pi_{\mathrm{old}}) \qquad (4)
$$

即 $`\mathrm{Var}_{\mathrm{old}}(r) = D_{\chi^{2}}`$（卡方散度）。f-散度在近端二阶等价（8.4 §3 还会用到这条）：策略移动小时 $D_{\chi^{2}} \approx D_{\mathrm{KL}}$，但方向相反地，它随策略偏离**加速**增长——KL 还差得不多时卡方已经很大，IS 权重的方差先于「目标近似失效」爆炸。序列级情形更糟：整条轨迹的 ratio 是逐 token 连乘 $\prod_t r_t$，各步小偏差相乘累积，方差随序列长度指数级增长——8.4 §7 的 GSPO 正是从这条账出发。**偏差账**：式 (3) 对 $L$ 无偏，但 $L$ 对 $\eta$ 的一阶近似只在邻域内成立——ratio 跑远时，「被无偏估计的那个目标」本身已经不是真实目标。所以问题从来不是「要不要限制新旧策略距离」，而是「用什么机制限制」。

**样本效率账的量化**（论文实验口径）：MuJoCo 设置下每轮 rollout $T = 2048$ 步、总预算 1M 环境步，即全程约 488 次策略更新、每次更新消费 2048 条新鲜转移；对比同一实现下每批数据复用 $K = 10$ 个 epoch × 32 个 minibatch = 320 次梯度步。on-policy 基线（每批只过一遍、不重复 epoch）在同预算下只能走 1/10 的参数量程——「数据复用率」与「每步可信度」的交换就是 §3-4 两部机制的定价对象。

## 3. TRPO：单调改进界与 KL 硬约束

TRPO 先把「邻域内可信」做成显式界。其 Theorem 1：记 $\alpha = D_{\mathrm{TV}}^{\max}(\pi_{\mathrm{old}}, \pi_{\mathrm{new}}) = \max_s D_{\mathrm{TV}}(\pi_{\mathrm{old}}(\cdot|s) \Vert \pi_{\mathrm{new}}(\cdot|s))$，则

$$
\eta(\pi_{\mathrm{new}}) \ge L_{\pi_{\mathrm{old}}}(\pi_{\mathrm{new}}) - \frac{4 \epsilon \gamma}{(1-\gamma)^{2}} \thinspace \alpha^{2}, \qquad \epsilon = \max_{s,a} |A_\pi(s,a)| \qquad (5)
$$

再用 Pinsker 型不等式 $D_{\mathrm{TV}}^{2} \le D_{\mathrm{KL}}$ 把 TV 换成 KL： $\eta \ge L - C \cdot D_{\mathrm{KL}}^{\max}$， $C = 4\epsilon\gamma/(1-\gamma)^{2}$。minorize-maximize 读法：每轮最大化右端的悲观下界， $\eta$ 单调不降——这是 TRPO「trust region」名字的出处。

但这个界不能直接当优化器用： $C$ 的尺度是灾难。 $\gamma = 0.99$ 时 $(1-\gamma)^{-2} = 10^{4}$，再乘最大优势模长——按式 (5) 的惩罚系数走，步长被压到不可用（练习 3 算这个数）。TRPO 实际算法的两步替换（其 §4，式 (11)(12) 处）：惩罚改硬约束、逐状态最大 KL 改状态平均 KL：

$$
\max_\theta \thinspace L_{\theta_{\mathrm{old}}}(\theta) \qquad \text{s.t.} \qquad \bar{D}_{\mathrm{KL}}(\theta_{\mathrm{old}}, \theta) \le \delta \qquad (6)
$$

**符号**： $`\bar{D}_{\mathrm{KL}}`$ 为 $`\mathbb{E}_{s \sim \rho_{\mathrm{old}}}[D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}(\cdot|s) \Vert \pi_\theta(\cdot|s))]`$； $\delta$ 为信任域半径，论文全部实验取 $\delta = 0.01$。注意此处的 $\delta$ 是 KL 阈值，与 GAE 的 TD 残差 $\delta_t$（8.2）是两个符号，本书沿用各自文献的记号并在出现时注明。

TRPO 的求解按约定只给结论与账。先补采样侧一句：论文给两种 rollout 方案——single path（沿当前策略跑整条轨迹，逐状态-动作对估 $`\hat{A}`$，不要求环境可重置）与 vine（在 rollout 状态集里挑若干状态各分叉出多条 rollout 以降低优势估计方差，要求模拟器可任意重置状态）；后续实现全部走 single path；vine 在论文实验中作为对照变体与 single path 并列报告（其 Table 1/2），之后未获沿用。求解侧：目标对 $\theta$ 一阶展开、KL 二阶展开（Hessian 即 Fisher 信息矩阵 $F$，论文用逐状态解析求积的 FIM，而非梯度外积的经验 FIM），得到约束二次规划，解是自然梯度方向： $\theta \leftarrow \theta_{\mathrm{old}} + \sqrt{2\delta / (g^{\top} F^{-1} g)} \cdot F^{-1} g$。 $F^{-1} g$ 不显式求逆，用共轭梯度（conjugate gradient, CG）解线性系统 $Fx = g$——每次 CG 迭代只需一次 Fisher-向量积（成本与一次梯度回传同量级，通过自动微分的二阶算子实现），论文取 10 次迭代（其附录 C：更多迭代不再加速策略改进，且 FIM 可在 10% 子采样数据上计算）；最后沿该方向回溯线搜索，收缩步长直到 KL 约束满足且 surrogate 确实改善。账：一次 TRPO 更新 ≈ 十余次 HVP + 线搜索的数次前向，比一次 Adam 步贵一个数量级以上；换来的是式 (5)-(6) 保证的近似单调性。PPO 的问题由此成形：能不能不碰 FIM、纯一阶，拿到近似的信任域效果？

## 4. PPO clip 目标：逐情形推导

PPO（[Schulman et al., 2017](../references.md)，arXiv:1707.06347，[B]）从不加约束的 surrogate 出发： $`L^{\mathrm{CPI}}(\theta) = \hat{\mathbb{E}}_t[r_t(\theta) \thinspace \hat{A}_t]`$（论文沿用 conservative policy iteration 的记号）。它的失效很直白： $`\hat{A}_t \gt 0`$ 的样本上，最大化目标会一路推高 $r_t$——没有任何机制阻止 ratio 发散，式 (2) 的一阶近似随之失效，论文原文是 "an excessively large policy update"（"destructively large" 一词在 §2 评述朴素策略梯度处）。

clip 版目标把逐样本 ratio 截断后取悲观众。先登记 LLM 场景的映射：状态 $s_t$ 是 prompt 加已生成前缀，动作 $a_t$ 是下一个 token， $r_t(\theta)$ 是逐 token 的 $\pi_\theta(o_t|q, o_{\lt t}) / \pi_{\theta_{\mathrm{old}}}(o_t|q, o_{\lt t})$——本节全部推导在该映射下逐字成立：

$$
L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \thinspace \hat{A}_t, \hspace{5mu} \mathrm{clip}\left( r_t(\theta), \thinspace 1 - \epsilon, \thinspace 1 + \epsilon \right) \hat{A}_t \right) \right] \qquad (7)
$$

**符号**： $\mathrm{clip}(x, lo, hi) = \min(\max(x, lo), hi)$； $\epsilon$：截断半径，论文默认 0.2； $`\hat{\mathbb{E}}_t`$ 为 rollout 批数据上的经验平均。

### 4.1 逐情形分析

min 的哪个分支起作用，按 $`\hat{A}_t`$ 的符号分两种情形（对应论文 Figure 1 的左右两图，此处以文字加数轴重绘）：

**情形一， $`\hat{A}_t \gt 0`$**（该动作优于基线，方向是提高其概率）： $r_t \le 1+\epsilon$ 时截断分支 $`(1+\epsilon)\hat{A}_t \ge r_t \hat{A}_t`$，min 取未截断项，目标照常上升、梯度照常（ $`\nabla_\theta (r_t \hat{A}_t) = \hat{A}_t \thinspace r_t \thinspace \nabla_\theta \log \pi_\theta(a_t|s_t)`$ ）； $r_t \gt 1+\epsilon$ 时 min 取 $`(1+\epsilon)\hat{A}_t`$，目标被钉平、梯度为零。合写：

$$
L_t = \hat{A}_t \cdot \min\left( r_t, \thinspace 1 + \epsilon \right), \qquad \hat{A}_t \gt 0 \qquad (8)
$$

向上修订有天花板：概率已抬高 $\epsilon$ 以上的样本不再产生任何收益。

**情形二， $`\hat{A}_t \lt 0`$**（该动作劣于基线，方向是压低其概率）：注意负 advantage 下 min 的选择方向反转。 $r_t \ge 1-\epsilon$ 时目标照常； $r_t \lt 1-\epsilon$ 时 $`\min(r_t \hat{A}_t, \thinspace (1-\epsilon)\hat{A}_t) = (1-\epsilon)\hat{A}_t`$（负数乘更小的正系数反而更大，被 min 排除），钉平、梯度为零。合写：

$$
L_t = \hat{A}_t \cdot \max\left( r_t, \thinspace 1 - \epsilon \right), \qquad \hat{A}_t \lt 0 \qquad (9)
$$

向下修订有地板：概率已压低 $\epsilon$ 以上的样本同样出局。两情形合起来，clip 的几何图像是一条「界内线性、界外平台」的分段曲线（论文 Figure 1 的 ASCII 重绘，取 $`\hat{A}_t = 2`$、 $\epsilon = 0.2$ 为例）：

```
A>0（A=2, ε=0.2）：L = 2·min(r, 1.2)      A<0（A=−2, ε=0.2）：L = −2·max(r, 0.8)

 2.4 ┤··················●━━━━━━━━━━━━        0 ┤────┬─────┬─────┬─────┬───
     │                ╱                          │    │     │     │     │
 2.0 ┤            ●                          −1.6 ┤━━━━●     │     │
     │         ╱                                    │     ╲    │     │
 1.0 ┤     ●                                     −2.0 ┤       ●     │
     │  ╱                                           │         ╲   │
   0 ●───┴────┴─────┴─────┴───                   −3.0 ┤───────────●
     0  0.5 0.8 1.0 1.2  r                          0  0.5 0.8 1.0 1.5  r
     ◄——— 有梯度 ———►◄ 梯度=0 ►                    ◄ 梯度=0 ►◄——— 有梯度 ———►
```

数值走一遍。左图： $r_t$ 从 0.5 爬到 1.5，目标值依次 1.0 / 1.6 / 2.0 / 2.4 / 2.4 / 2.4（ $r$ = 0.5 / 0.8 / 1.0 / 1.2 / 1.4 / 1.5）——界内等斜率，一过 1.2 立即钉平；梯度 $\nabla_\theta L_t = 2 r_t \nabla \log \pi_\theta$ 在 $r_t \le 1.2$ 上成立，之外恒零。右图： $r_t$ 从 1.5 降到 0.5，目标值依次 -3.0 / -2.0 / -1.6 / -1.6 / -1.6（ $r$ = 1.5 / 1.0 / 0.8 / 0.6 / 0.5）——压低坏动作的概率让目标从 -3.0 升到 -1.6，但一过 0.8 目标钉平：继续压到低不再产生收益。

clip 的几何含义至此可以一句话说透：平台段恰好截掉「继续沿当前方向走远」的全部收益，而线性段的斜率（ $`\pm \hat{A}_t`$ ）把「界内走多远」的定价权留给梯度本身。

反向方向永不截断： $`\hat{A}_t \gt 0`$ 时把 $r_t$ 往回推（撤销提高）始终有梯度， $`\hat{A}_t \lt 0`$ 时把 $r_t$ 往回推（撤销压低）也始终有梯度。clip 只拦截「继续走远」，不拦截「回退」——这是它与 KL 惩罚的本质区别：后者对双向偏离对称收费，前者只截单向。

两个性质写成命题备查：

**悲观下界（pessimistic bound）**。逐点有 $L^{\mathrm{CLIP}}(\theta) \le L^{\mathrm{CPI}}(\theta)$：min 恒取两者较小，clip 版是原 surrogate 的下界——最大化它至多低估、绝不高估这一步的收益，与 TRPO「最大化下界」同构，只是下界用逐样本截断而非全局 KL 构造。验证： $`\hat{A}_t \gt 0`$ 时仅 $r_t \gt 1+\epsilon$ 段严格更小、 $r_t \le 1+\epsilon$ 段取等； $`\hat{A}_t \lt 0`$ 时仅 $r_t \lt 1-\epsilon$ 段严格更小（练习 1）。

**梯度开关**。平台段梯度恒零 ⇒ 越界样本对参数更新零贡献。一批数据中触发截断的比例（clip fraction）因此是 PPO 的一阶健康指标：过高说明 lr、epoch 数或数据陈旧度过激，有效 batch 在悄悄缩水（§8 的 `clipfracs` 就是它的实现）。与 clipfrac 互补的读数是近似 KL：clipfrac 量「多少样本被关掉」，approx_kl 量「还活着的样本平均走了多远」——两者同高是 lr 过激，clipfrac 高而 KL 低则是典型的「少数样本飞出、多数未动」，该调小的是每 epoch 的步数而非步长。

还要登记一条 clip 不管的事：它不按 KL 硬约束，多 epoch 更新后新旧策略的 KL 可以漂出 $\epsilon$ 对应的范围——实现层用 KL early-stop 兜底（§6 第 4 条）。

### 4.2 多 epoch 复用的陈旧偏差：importance ratio 偏差分析

PPO 相对 TRPO 省下的算力花在数据复用上：同一批 rollout 跑 $K$ 个 epoch 的 minibatch 更新。复用的代价可以精确刻画（本书推导）。记 rollout 策略为 $\pi_{\mathrm{old}}$，第 $k$ 个 epoch 的参数为 $\theta_k$：

- $k = 1$ 时 $r_t \approx 1$，截断基本不触发， $`\nabla L^{\mathrm{CLIP}} \approx \hat{\mathbb{E}}[\hat{A}_t \nabla \log \pi_\theta]`$ ——近似 on-policy 策略梯度，无偏性口径同 8.1；
- $k$ 增大时 $\pi_{\theta_k}$ 偏离 $\pi_{\mathrm{old}}$：数据分布对当前参数而言是陈旧的， $r_t$ 系统性偏离 1，更多样本撞上平台段、梯度被关掉—— $L^{\mathrm{CLIP}}$ 相对 $L^{\mathrm{CPI}}$ 的下压，正是目标函数对陈旧数据的主动折价。

一个两 epoch 的微型算例把折价过程具体化。设某个样本 $`\hat{A}_t = 1`$、 $\epsilon = 0.2$，第一个 epoch 把它的 ratio 从 1.0 推到 1.2（界内，全程有梯度）；第二个 epoch 开头 ratio 已是 1.2，再往上立即钉平——该样本的剩余「复用价值」为零，它在后续 epoch 里唯一的贡献是消耗 minibatch 位置。推过头的样本则更糟：lr 过大时第一个 epoch 就把 ratio 推到 1.5，该样本此后九个 epoch 全部沉默；沉默样本占比正是 clipfrac 曲线在 $K$ 增大时的爬升段。这一观察给出 epoch 数的机制性上限：当 clipfrac 接近「多数样本已被截断」时，继续加 epoch 只是在冻结的数据上做随机扰动。

所以 clip 同时干两件事：对「单步走太远」截断（信任域语义），对「数据变陈旧」折价（off-policy 语义）。要强调的是它不是无偏校正——式 (7) 的期望即使在 $k=1$ 处也只恢复到策略梯度的一阶近似，clip 更刻意放弃了 IS 的精确性，换方差控制与近似单调；真正做 off-policy 无偏校正的另一条路（staleness 校正、行为策略重放）归 8.5。

与 KL 方案的论文级对照也在这里。PPO 论文同时给了 PPO-Penalty 一支： $`L^{\mathrm{KLPEN}}(\theta) = \hat{\mathbb{E}}_t[r_t \hat{A}_t - \beta \thinspace D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}(\cdot|s_t) \Vert \pi_\theta(\cdot|s_t))]`$， $\beta$ 自适应——每轮更新后测 $`d = \hat{\mathbb{E}}_t[D_{\mathrm{KL}}]`$， $d \lt d_{\mathrm{targ}}/1.5$ 则 $\beta$ 减半， $d \gt 1.5 \thinspace d_{\mathrm{targ}}$ 则 $\beta$ 加倍（其 §4， $d_{\mathrm{targ}}$ 为目标 KL）。其 §6.1（Table 1）在 7 个 MuJoCo 任务、1M 步的口径下对比：clip 变体总体优于固定 $\beta$ 、自适应 $\beta$ 与无约束变体——一阶截断打败显式散度惩罚，这是 PPO-Clip 成为默认形态的直接证据来源。

## 5. 完整 PPO 算法

actor-critic 形态的总目标（论文式 (9)，本章编号 (10)）：

$$
L_t^{\mathrm{CLIP+VF+S}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\mathrm{CLIP}}(\theta) - c_1 \thinspace L_t^{\mathrm{VF}}(\theta) + c_2 \thinspace S[\pi_\theta](s_t) \right] \qquad (10)
$$

**符号**： $L_t^{\mathrm{VF}} = (V_\theta(s_t) - V_t^{\mathrm{targ}})^{2}$：价值回归项； $S[\pi_\theta](s_t)$：策略熵奖励； $c_1, c_2$：系数。论文 MuJoCo 口径用分离的 policy/value 网络、不用熵奖励——其 §6.1 明言 "coefficient $c_1$ is irrelevant"、 $c_2$ 无涉； $c_1 = 1$、 $c_2 = 0.01$ 属 Atari 口径（其附录 Table 5）。价值目标取 $\lambda$ -return 形式 $`V_t^{\mathrm{targ}} = \hat{A}_t + V_{\mathrm{old}}(s_t)`$（实现层标准做法，与 8.2 的 GAE 递推配套）。

一轮迭代的完整流程（论文 Algorithm 1 的展开，加上实现层的标准配套；论文 Algorithm 1 用 $N$ 记并行 actor 数，本章按全书约定把 $N$ 专留参数量、并行环境数记作 $N_{\mathrm{env}}$）：

```
┌─ rollout（π_old 冻结）────────────────────────────────────────────┐
│ N_env 个并行环境 × T 步                                                │
│ 存 (s_t, a_t, r_t, log π_old(a_t|s_t), V_old(s_t), done)           │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
        GAE（8.2）：δ_t = r_t + γ·V_old(s_{t+1}) − V_old(s_t)
        逆序递推 Â_t = Σ_l (γλ)^l δ_{t+l}；V_targ = Â + V_old
        （Â、log π_old、V_old 在整个更新阶段冻结）
                                ▼
┌─ 更新（K 个 epoch，每 epoch 打乱后切 minibatch）─────────────────┐
│ 每个 minibatch：                                                 │
│   r_t = exp(log π_θ − log π_old)                                  │
│   L_CLIP = mean( min(r·Â, clip(r, 1±ε)·Â) )  （Â 做 minibatch 归一）│
│   L_VF（可带 value clip）+ 熵奖励 → 合成式 (10)，Adam 反传          │
│   诊断：approx_kl、clipfrac                                       │
│   approx_kl > target_kl ⇒ 提前终止本轮 epoch（可选）               │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
                     θ_old ← θ，回到下一轮 rollout
```

论文 MuJoCo 口径的配平数字（其附录 Table 3；OpenAI Baselines 的 MuJoCo 默认配置同口径）：单环境、 $T = 2048$ 、 $K = 10$ 、minibatch $M = 64$ 、 $\gamma = 0.99$ 、 $\lambda = 0.95$ 、Adam lr $3 \times 10^{-4}$ 、 $\epsilon$ 在 §6.1 消融（见 §7）、总计 1M 环境步，policy/value 分离网络、无熵奖励；Atari 口径（其附录 Table 5）：8 个并行环境、 $T = 128$ 、 $K = 3$ 、 $M = 32 \times 8 = 256$ 、 $\epsilon = 0.1$ 与 lr $2.5 \times 10^{-4}$ 均乘线性退火因子、 $c_1 = 1$ 、 $c_2 = 0.01$（Baselines 的 Atari 默认把 epoch 数改为 4，与论文 Table 5 不同——此句为本书注记）。一批 rollout 被 $K$ 个 epoch 复用——样本效率的全部来源在此，代价即 §4.2 的陈旧偏差； $K$ 与 $M$ 的选取实质上是在「数据复用率」与「有效梯度新鲜度」之间配平。

三个循环不变量值得点名，它们把 §3-4 的机制钉在流程里。其一，rollout 段的全部张量（ $\log \pi_{\mathrm{old}}$、 $V_{\mathrm{old}}$、 $`\hat{A}_t`$ ）在进入更新段之前冻结成常数张量——更新段的任何梯度都不许流回它们，否则 ratio 的分母会跟着 $`\theta`$ 动、式 (7) 的截断语义直接失效（实现上是 `detach` / `torch.no_grad` 的边界，§8 可见）。其二，minibatch 必须**遍历切分**而非随机重采： $N_{\mathrm{env}} \cdot T$ 个样本每 epoch 各被消费恰好一次，随机有放回采样会让部分样本被重复消费、部分缺席，在 $K$ 个 epoch 上把陈旧偏差进一步放大（代码事实：CleanRL 用 `np.random.shuffle` 打乱索引后顺序切分；随机有放回取数是已知错误形态——此句为本书注记）。其三，early-stop 的判断量用的是新旧策略的近似 KL（§8 的 k3 估计器），不是 ratio 本身——KL 是分布级距离，对「多数样本没动、少数样本飞出去」的形态比 clipfrac 更敏感。

## 6. 实现细节清单（逐条配失效模式）

论文式 (7)(9) 到能复现论文曲线的实现之间有一层「不在论文里」的细节，CleanRL 文档把它的核心部分整理为 13 条实现细节并逐条对齐了 OpenAI Baselines 的曲线（[Huang et al., CleanRL repo](../references.md)，教学参照）。挑对训练动力学影响最直接的六条，每条附失效模式：

1. **advantage 归一化**： $\hat{A} \leftarrow (\hat{A} - \mathrm{mean})/(\mathrm{std} + 10^{-8})$，**minibatch 级**而非全 batch 级，CleanRL 默认开。失效模式：minibatch 太小（几十个样本）时均值与方差本身的估计噪声直接注入梯度方向；极端情形一个 minibatch 内 advantage 全同号，归一化把噪声放大成信号。LLM-RL 的批往往只有几十条 prompt 乘组内样本，这条比在 Atari 上更敏感——8.4 §2 的组内归一化是它的近亲，适用同一套警惕。一个可参考的量级：minibatch 内 advantage 的样本均值本身的标准差约为 $`\sigma_A / \sqrt{M}`$， $M = 64$ 时相对漂移已达 $0.125 \sigma_A$——归一化在降噪的同时也在注入一个 $1/\sqrt{M}$ 量级的随机缩放。
2. **value clipping**： $L^{\mathrm{VF}} = \max\left[ (V_\theta - V^{\mathrm{targ}})^{2}, \thinspace (\mathrm{clip}(V_\theta, \thinspace V_{\mathrm{old}} \pm \epsilon) - V^{\mathrm{targ}})^{2} \right]$——与 clip 目标同构的价值截断，Baselines 与 CleanRL 默认开。**注意它不在论文式 (9) 里**，是实现层的发明；后续复现研究对其收益结论不一（此句为本书注记，无单一出处）。失效模式：reward 尺度漂移时 $V$ 被锁在 $V_{\mathrm{old}}$ 附近追不上目标， $\hat{A} = V^{\mathrm{targ}} - V$ 出现系统性偏移。
3. **熵奖励系数 $c_2$**：MuJoCo 取 0、Atari 取 0.01。失效模式：过大则策略停在近均匀分布、exploit 不足；过小则熵过早塌缩、策略钉死在次优确定性动作上。LLM-RL 里同一现象（entropy 塌缩）的机制分析与对策见 8.4 §6 的 clip-higher。
4. **KL early-stop**：approx_kl 超过 target_kl 时提前结束本轮 epoch（CleanRL 中 `target_kl` 默认关闭）。失效模式：阈值太紧，有效 epoch 数掉到 1 以下，rollout 数据被浪费、退回接近单更新 on-policy；太松则 §4.2 的陈旧偏差累积，末尾几个 epoch 在噪声里更新。
5. **价值损失系数与网络结构**：论文 $c_1 = 1$ ；Baselines/CleanRL 取 0.5 且 policy 与 value 用分离的 MLP（不共享主干）。失效模式：共享主干时两个目标的梯度互相干扰，critic 欠训会拖着 policy 的特征表示走； $c_1$ 过大则 critic 过拟合当前 return 分布，policy 目标被相对稀释。
6. **Adam 的 $\epsilon$ 与 lr 退火**：Baselines 用 Adam $\epsilon = 10^{-5}$（区别于 PyTorch 默认 $10^{-8}$ ），lr 随训练进度线性退火。失效模式：套库默认值会改变早期有效步长（机制同 2.2 §4 的 $\epsilon$ 两 regime）；lr 不退火则末期的更新噪声不收敛，ratio 持续撞界。

环境侧另有标配两条：reward/observation 归一化（Atari 为 reward clipping）与全局梯度裁剪（norm 0.5）——失效模式分别是价值目标非平稳，以及偶发大梯度把界内样本的 ratio 一次推出截断区。

## 7. 超参敏感性：论文的数据口径

PPO 论文自身的敏感性证据分三组，按原样登记（[Schulman et al., 2017](../references.md)，[B]）：

- **surrogate 变体对比**（其 §6.1 Table 1）：7 个 MuJoCo 任务 × 3 种子、每任务 1M 步，clip（ $\epsilon$ = 0.1 / 0.2 / 0.3）归一得分 0.76 / 0.82 / 0.70，自适应 KL（ $d_{\mathrm{targ}}$ = 0.003 / 0.01 / 0.03）0.68 / 0.74 / 0.71，固定 KL（ $\beta$ = 0.3 / 1 / 3 / 10）0.62 / 0.71 / 0.72 / 0.69，无约束无截断 -0.39——clip 全面最优，无约束最差（HalfCheetah 上比随机策略还差）。这是「clip 作为默认形态」的出处，不是社区约定俗成。
- ** $\epsilon$ 取值**：论文 §6.1 的 Table 1 就是三点消融—— $\epsilon \in \lbrace 0.1, 0.2, 0.3 \rbrace$ 在 7 个 MuJoCo 任务 × 3 种子（21 次运行归一平均）上得 0.76 / 0.82 / 0.70，均远胜无截断的 -0.39，正文明示这些值 "which we search over"；0.2 为三点中最优。更细的网格与 LLM 域的消融论文未做（此句为本书注记）。
- **学习率与总步数的交互**：论文 MuJoCo 实验 1M 环境步内约 488 次更新（ $10^{6} / 2048$ ），每次更新 320 个梯度步——lr $3 \times 10^{-4}$ 是在「每批 320 步」的复用强度下标定的；把 $K$ 或 minibatch 数改掉而不动 lr，等价于改有效步长，这与 2.3 的「batch 变更必须联动 lr」是同一纪律的 RL 版本。
- ** $K$ / $T$ / $M$ 配平**：论文 MuJoCo 口径 $K = 10$ 意味着每批数据复用 10 遍； $K$ 翻倍的代价非线性——§4.2 的陈旧偏差随 epoch 累积，论文未给敏感性曲线。社区复现收敛后的口径更保守（CleanRL 默认 $K = 4$ ），配 target_kl 或 clipfrac 监控使用。

还有一条跨章纪律：PPO 的全部稳定性结论都建立在「critic 与 reward 可信」的前提上。LLM 场景里 reward 来自另一个模型（RM），过优化 RM 的代理奖励会触发 reward hacking——目标侧的分析归 9.2；当 reward 退化成规则验证器（数学答案对错）时 critic 的训练信号同时退化，这正是 8.4 去 critic 的动机。本章的式 (7)-(10) 在两个场景里逐字成立，变的只是 $\hat{A}$ 的来源与可信度。

LLM 场景的定位一句：PPO 是 InstructGPT 式 RLHF 的原始优化器，「KL 进 reward、四模型驻留」的全链形态归 9.2；工程栈（rollout/train 引擎分离、权重同步）归 8.5。锚点纪律注记：TRL 已于 2026-02 移除 `ppo_trainer.py`（2026-11-17 经 GitHub API 核实 `trl/trainer/` 目录已无此文件），PPO 的实现走读锚因此落在 CleanRL 与 OpenRLHF。

## 8. CleanRL `ppo.py` 逐行走读

以 vwxyzjn/cleanrl 仓库的 `cleanrl/ppo.py` 为准（教学参照；下列行级事实核实于 2026-11-17）。默认配置：并行环境数 $N_{\mathrm{env}} = 4$ 、 $T = 128$ 步、 $K = 4$ epoch、4 个 minibatch、 $\epsilon = 0.2$ 、 $\gamma = 0.99$ 、 $\lambda = 0.95$ 、lr $2.5 \times 10^{-4}$ 退火、 $c_1 = 0.5$ 、 $c_2 = 0.01$ 、梯度裁剪 0.5。

**rollout 段**。六个预分配缓冲 `obs / actions / logprobs / rewards / dones / values`，形状 $[T, N_{\mathrm{env}}]$；每步在无梯度下 `agent.get_action_and_value` 采动作并记 $V$。 $\pi_{\mathrm{old}}$ 的 log prob 与 $V_{\mathrm{old}}$ 在此冻结——它们正是式 (7) 的 ratio 分母与 value clip 的参照点。

**GAE 段**，与 8.2 的推导逐行对应（源码行）：

```python
delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
```

逆序递推把 $`\hat{A}_{t+1}`$ 折进 $`\hat{A}_t`$ ；`returns = advantages + values` 即 $\lambda$ -return 价值目标。

**更新段**。展平成 batch（大小 $N_{\mathrm{env}} \cdot T$ ），每 epoch 重洗索引、切 minibatch；ratio 写成 `logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()`——log 域相减再取指数，比直接除概率数值稳。诊断三件套逐行对应 §4 的分析：`old_approx_kl = (-logratio).mean()` 与 `approx_kl = ((ratio - 1) - logratio).mean()` 是新旧策略 KL 的两个蒙特卡洛估计（前者 k1、后者 k3，谱系与性质是 8.4 §3 的主题，此处只登记它们充当 early-stop 与监控的输入）；`clipfracs` 记录 $|r_t - 1| \gt \epsilon$ 的样本占比，即 §4.1「梯度被关掉的样本比例」的实现——口径注记：clipfracs 是双向计数（ $r$ 高出上界与低于下界都计入），而梯度真正被关掉的样本还要按 $`\hat{A}_t`$ 的符号取单向子集，clipfracs 因此是关断量的上界口径。

clip 目标的代码形态值得单独看一行：loss 取负后 min 变 max——

```python
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

与式 (7) 逐项相同（最小化 `max(-·)` ≡ 最大化 `min(·)`）。value clip（对应 §6 第 2 条）：`v_clipped = b_values + clamp(newvalue - b_values, ±clip_coef)`，再取未截断与截断两个平方误差的逐点最大、乘 0.5。合成损失 `loss = pg_loss - ent_coef * entropy + v_loss * vf_coef` 即式 (10)。early-stop 一行：`if args.target_kl is not None and approx_kl > args.target_kl: break`——该 `break` 在 minibatch 循环体内，跳出的只是本 epoch 的剩余 minibatch，已算完的更新照常生效；外层 epoch 循环继续进入下一轮，KL 仍超阈时会再次触发。

教学骨架（教学用，非优化实现）把 §5 的流程写成单机可读形式：

```python
# 教学用，非优化实现：PPO 更新核心（GAE 逆推 + clip 目标 + value clip）
advantages = torch.zeros_like(rewards); gae = 0.0
for t in reversed(range(T)):
    nonterm = 1.0 - dones[t + 1]
    delta = rewards[t] + gamma * values[t + 1] * nonterm - values[t]
    gae = delta + gamma * lam * nonterm * gae
    advantages[t] = gae
returns = advantages + values                     # lambda-return 价值目标
for epoch in range(K):
    for mb in shuffled_minibatches(N_env * T):
        ratio = (logp_new[mb] - logp_old[mb]).exp()
        adv = (advantages[mb] - advantages[mb].mean()) / (advantages[mb].std() + 1e-8)
        pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
        v_clip = v_old[mb] + (v_new[mb] - v_old[mb]).clamp(-eps, eps)
        v_loss = 0.5 * torch.max((v_new[mb] - returns[mb])**2,
                                 (v_clip - returns[mb])**2).mean()
        loss = pg - ent_coef * entropy[mb].mean() + vf_coef * v_loss
        # Adam 反传；approx_kl > target_kl 则 break
```

**buffer 边界与 bootstrap**。两段代码之间的接口有一个容易写错的语义点：rollout 末状态的 $V(s_{T})$ 要用于 GAE 递推的 bootstrap（`next_value` 参与最后一个 $\delta$ ），且 `nextnonterminal` 把 done 处的自举切断——episode 终止时不从下一状态取值。Atari 类环境还要区分 termination（真终止）与 truncation（步数上限截断）：时限感知（time-limit-aware）的做法在截断处保留自举，否则长 episode 的 return 被系统性低估。CleanRL `ppo.py` 里 `next_done = np.logical_or(terminations, truncations)` 把两者合并，GAE 自举在两处都切断——这是一个已知的简化处理，严格口径应只在真终止处切断；该语义边界的完整讨论归 8.2 的 GAE 实现注记。

**退火与优化器**。`anneal_lr=True` 时 lr 按 `1 - update / num_updates` 线性衰减到 0（`optimizer.param_groups` 逐 update 改写）；Adam 的构建在 `agent.parameters()` 全体参数上、 $\epsilon = 10^{-5}$——§6 第 6 条的两个细节在此落地。

与真实实现的差别：真实栈处理连续动作分布（`Normal` 的对数概率按维求和）、LSTM 状态重放、多 GPU 与向量环境的边界（truncation 与 termination 分开处理 bootstrap）等分支；骨架只保留语义主线。

工业栈对照一句：OpenRLHF 的 PPO 损失在 `openrlhf/models/loss.py` 的 `PolicyLoss`（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]；路径核实于 2026-11-17）——其 clip 半径已解耦为 `clip_eps_low` / `clip_eps_high`，即 8.4 §6 DAPO clip-higher 的形态在工业栈里已是内置项；四模型驻留与 rollout/train 分离的系统走读归 8.5。教学参照的另一个入口是 CleanRL 文档站给出的基准对照：`ppo.py` 在经典控制三任务上的平均回报与 OpenAI Baselines 同量级（其文档页表格口径：CartPole-v1 490.04 ± 6.12 对 497.54 ± 4.02 等）——「单文件实现足以复现参考结果」是把它选作主走读对象的理由。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| on-policy 代价 | 一批 rollout 只支撑一次无偏梯度步；LLM 场景 rollout 成本远高于梯度 | §1 |
| surrogate 一阶匹配 | $L_\pi$ 换 $\rho_{\tilde{\pi}}$ 为 $\rho_\pi$ 后 $L=\eta$、梯度相等（在 $\theta_{\mathrm{old}}$ 处）；IS 只纠动作分布 | 式 (2)(3)，arXiv:1502.05477 |
| TRPO 界 | $\eta \ge L - C D_{\mathrm{KL}}^{\max}$， $C = 4\epsilon\gamma/(1-\gamma)^2$ 尺度灾难 → 硬约束 $`\bar{D}_{\mathrm{KL}} \le \delta`$（0.01）；CG + 线搜索求解 | 式 (5)(6)，arXiv:1502.05477 |
| PPO clip | 悲观下界 + 梯度开关； $`\hat{A} \gt 0`$ 截上、 $`\hat{A} \lt 0`$ 截下，反向永不截 | 式 (7)-(9)，arXiv:1707.06347 |
| 陈旧偏差 | 多 epoch 复用使 $r_t$ 偏离 1，clip 对陈旧数据主动折价；非无偏校正 | §4.2（本书推导） |
| 完整算法 | 式 (10) + rollout → GAE → $K$ epoch minibatch；MuJoCo 口径 $T$=2048、 $K$=10、 $M$=64、 $\epsilon$=0.2 | 式 (10) 与 §5 |
| 实现细节 | adv 归一化（minibatch 级）、value clip（非论文）、熵奖励、KL early-stop、 $c_1$ 与分离网络、Adam $\epsilon$=1e-5 + lr 退火 | §6，CleanRL 13 条 |
| CleanRL 走读 | ratio log 域、双 KL 诊断（k1/k3）、clipfrac、`max(-·)` 写法、early-stop 跳本 epoch 剩余 minibatch | §8，`cleanrl/ppo.py` |

## 练习

1.（推导）clip 目标的悲观性：证明对任意 $r \gt 0$ 与 $\hat{A} \neq 0$， $\min(r\hat{A}, \thinspace \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\hat{A}) \le r\hat{A}$，并分 $\hat{A}$ 正负给出等号成立的 $r$ 区间；由此验证式 (8)(9) 的分段写法与「反向永不截断」的论断。
2.（推导）IS 方差：设单步动作上 $D_{\mathrm{KL}}(\pi_{\mathrm{old}} \Vert \pi_\theta) = \kappa$ 有界，写出 $`\mathbb{E}_{\mathrm{old}}[r_t^2]`$ 的表达式并说明它随 $\kappa$ 增长；进一步说明序列级 ratio $\prod_{t=1}^{T} r_t$ 的方差为什么随 $T$ 爆炸（与 8.4 §7 的 GSPO 动机衔接）。
3.（计算）TRPO 界尺度： $\gamma = 0.99$、 $\epsilon = \max_{s,a}|A| = 1$，算 $C = 4\epsilon\gamma/(1-\gamma)^2$ ；取 $D_{\mathrm{KL}}^{\max} = 0.01$，惩罚项多大、相对 $L$ 的典型量级意味着什么？由此解释论文为什么放弃惩罚式 (5) 改约束式 (6)。
4.（代码）跑 CleanRL `ppo.py`（CartPole-v1， `--no-cuda`），把 `--update-epochs` 从 4 调到 12，读 `losses/clipfrac` 与 `losses/approx_kl` 两条曲线的变化，用 §4.2 的陈旧偏差框架解释；再把 `--norm-adv` 关掉对比收敛差异。

## 参考

- [Schulman et al., 2017](../references.md)：Proximal Policy Optimization Algorithms，arXiv:1707.06347，[B]（key: ppo）——clip 目标式 (7)（论文式 (7)）、总目标式 (10)（论文式 (9)）、自适应 $\beta$ 规则（论文 §4）与本章 §6 实验口径
- [Schulman et al., 2015](../references.md)：Trust Region Policy Optimization，arXiv:1502.05477，[B]（key: trpo）——恒等式 (1)、surrogate 式 (2)(3)、Theorem 1（式 (5)）、硬约束式 (6) 与 CG + 线搜索、 $\delta$=0.01
- [Schulman et al., 2016](../references.md)：High-Dimensional Continuous Control Using Generalized Advantage Estimation，arXiv:1506.02438，[B]（key: gae）—— $`\hat{A}_t`$ 估计（完整推导归 8.2）
- [Huang et al., vwxyzjn/cleanrl repo](../references.md)（key: cleanrl，教学参照）——`cleanrl/ppo.py` 走读对象、13 条实现细节与默认配置、Baselines 曲线对齐
- [Hu et al., 2024](../references.md)：OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework，arXiv:2405.11143，[A]（key: openrlhf）——`openrlhf/models/loss.py` `PolicyLoss` 的解耦 clip；系统走读归 8.5
