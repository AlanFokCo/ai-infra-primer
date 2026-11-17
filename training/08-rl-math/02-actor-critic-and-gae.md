# 8.2 Actor-Critic 与 GAE（Actor-Critic and Generalized Advantage Estimation）

> 8.1 的最优 baseline 推导落在一个不可查表的对象上：价值函数 V。本章把它学成网络——critic 的 TD(0) 训练与 Bellman 残差、一步残差 δ 恰好是 advantage 的一步无偏估计的引理、GAE 的完整推导（k 步 advantage 的指数加权 = TD 残差的 (γλ) 折扣和、偏差-方差权衡的定量式、GAE 论文的 γ-just 框架）、actor-critic 联合 loss 与 CleanRL `ppo.py` 的文件级走读，最后落在 LLM 场景：末端奖励下 value 函数的退化形态与 token 级 credit assignment 的困难——这是 8.4 GRPO 去 critic 的伏笔 | 前置：8.1（策略梯度定理、baseline 无偏性与最优 baseline）、3.2 §5（on-policy 口径）、2.4（梯度裁剪——实现走读处用到） | 锚点：GAE 论文（[Schulman et al., 2016](../references.md)，arXiv:1506.02438，[B]）、CleanRL（vwxyzjn/cleanrl repo，`cleanrl/ppo.py`，教学参照）、Sutton & Barto 教材 [B]

## 1. 从 baseline 到 critic：8.1 留下的问题

8.1 式 (23) 给出最优 baseline 的闭式： $b^{\ast}(s)$ 是以 score 范数平方加权的 $Q$ 平均，等权近似下退化为 $`V^{\pi}(s) = \mathbb{E}_a[Q(s, a)]`$ 。无论哪一档，都需要一个「给定状态、估计未来回报」的函数——它在 LLM 场景（状态 = prompt + 前缀，空间随长度指数膨胀）不可能查表，只能函数逼近。actor-critic 的结构由此注定：**actor** 是策略 $\pi_\theta$ ，**critic** 是学出来的价值网络 $V_\varphi$ ，两者在同一个 rollout 数据流上联合训练。

本章的问题链： $V$ 怎么从样本学（§2-3）→ 学到的 $V$ 怎么变成低方差的 advantage 估计（§4-5，GAE）→ 两个网络的 loss 怎么拼、生产实现长什么样（§6）→ LLM 的末端奖励结构让这套机器退化成什么样（§7）。

## 2. V、Q 与 Bellman 方程

沿 8.1 的记号（ $s_t, a_t, r_t$ ， $\gamma$ ，episodic 视界 $T$ ）。三个价值对象的定义：

$$
V^{\pi,\gamma}(s) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{l=0}^{\infty} \gamma^{l} \thinspace r_{t+l} \mid s_t = s \right], \qquad Q^{\pi,\gamma}(s, a) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{l=0}^{\infty} \gamma^{l} \thinspace r_{t+l} \mid s_t = s, \thinspace a_t = a \right] \qquad (1)
$$

$$
A^{\pi,\gamma}(s, a) = Q^{\pi,\gamma}(s, a) - V^{\pi,\gamma}(s) \qquad (2)
$$

**符号**： $V^{\pi,\gamma}$ ：状态价值——从 $s$ 出发沿 $\pi$ 的期望折扣回报； $Q^{\pi,\gamma}$ ：动作价值——在 $s$ 先执行 $a$ 再沿 $\pi$ ； $A^{\pi,\gamma}$ ：advantage（优势函数），动作相对策略默认行为的好坏。advantage 有一条贯穿本章的性质：

$$
\mathbb{E}_{a \sim \pi(\cdot|s)}\left[ A^{\pi,\gamma}(s, a) \right] = \mathbb{E}_{a \sim \pi}\left[ Q^{\pi,\gamma}(s, a) \right] - V^{\pi,\gamma}(s) = V^{\pi,\gamma}(s) - V^{\pi,\gamma}(s) = 0 \qquad (3)
$$

Bellman 方程把价值写成一步自洽条件（[Sutton & Barto, 2018](../references.md)，[B]，§3.5-3.6；策略评估口径）：

$$
V^{\pi,\gamma}(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}\left[ r(s, a) + \gamma \thinspace \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[ V^{\pi,\gamma}(s') \right] \right], \qquad Q^{\pi,\gamma}(s, a) = r(s, a) + \gamma \thinspace \mathbb{E}_{s'}\left[ V^{\pi,\gamma}(s') \right] \qquad (4)
$$

**符号**： $r(s, a) = \mathbb{E}[r_t \mid s_t = s, a_t = a]$ ：期望即时奖励。式 (4) 的读法是递归定义：价值 = 一步奖励 + 折扣后的下一步价值。LLM 场景下 $P$ 是确定性拼接（8.1 §2.2），外层对 $s'$ 的期望退化为单点取值——但 $r$ 只在末端非零的退化结构让学习的难度原样保留（§7）。

## 3. Critic 学习：TD(0) 目标与回归 loss

学 $V_\varphi \approx V^{\pi,\gamma}$ 的监督信号有两极。Monte Carlo 目标：用整条轨迹的实测回报 $G_t$ 回归 $V_\varphi(s_t)$ ——无偏（ $G_t$ 的期望就是 $V^{\pi}$ ），但方差是全轨迹随机性的总和。TD(0) 目标：用 Bellman 方程右端的单步采样

$$
y_t = r_t + \gamma \thinspace V_\varphi(s_{t+1}) \qquad (5)
$$

**符号**： $y_t$ ：TD target——一次真实奖励加 critic 自己的下一步估值（bootstrap）。critic 的回归 loss ：

$$
L_V(\varphi) = \frac{1}{2} \thinspace \mathbb{E}_t\left[ \left( V_\varphi(s_t) - y_t \right)^{2} \right] \qquad (6)
$$

**符号**：期望对 on-policy rollout 缓冲里的步取。三条机制注记。其一，**半梯度（semi-gradient）口径**： $y_t$ 依赖 $\varphi$ 但求梯度时按常数处理（stop-gradient）——对式 (6) 求导得到的是 $-(V_\varphi(s_t) - y_t) \thinspace \nabla_\varphi V_\varphi(s_t)$ ，不是「Bellman 残差平方」的真梯度；真梯度版本（residual gradient）另有算法且性质不同，教材对这条分界有专门讨论（[Sutton & Barto, 2018](../references.md)，[B]，§9.3、§11.5）。其二，**bootstrap 的偏差-方差位置**：TD target 的方差只含一步奖励与一步转移的随机性，代价是把 $V_\varphi$ 当前的误差直接喂回目标——critic 没收敛时监督信号本身有偏。其三，**on-policy 移动靶**： $V^{\pi,\gamma}$ 随 actor 的每次更新而变，critic 追的是一个移动目标；PPO 系实现对两者的相对步长有明确配比（§6 的 loss 加权与 epoch 结构）。

TD 残差（Bellman residual）是后面一切的地基：

$$
\delta_t^{V} = r_t + \gamma \thinspace V(s_{t+1}) - V(s_t) \qquad (7)
$$

**符号**： $\delta_t^{V}$ ：以 $V$ 为近似价值函数时第 $t$ 步的 TD 残差。它是「这一步实际拿到的（奖励 + 未来估值）」与「站在 $s_t$ 时的估值」之差——critic 的惊讶度。

## 4. 一步残差就是 advantage：δ 的无偏性引理

**引理**（GAE 论文的出发点，[Schulman et al., 2016](../references.md)，arXiv:1506.02438，[B]）：当 $V = V^{\pi,\gamma}$ 精确成立时，

$$
\mathbb{E}_{s_{t+1}}\left[ \delta_t^{V} \mid s_t, \thinspace a_t \right] = A^{\pi,\gamma}(s_t, a_t) \qquad (8)
$$

**证明**：对式 (7) 取条件期望，前两项由式 (4) 的 $Q$ 形式给出 $\mathbb{E}[r_t + \gamma V^{\pi,\gamma}(s_{t+1}) \mid s_t, a_t] = Q^{\pi,\gamma}(s_t, a_t)$ ，第三项 $V^{\pi,\gamma}(s_t)$ 在条件号下是常数；由式 (2) 即得。 $\square$

**推论**：再对 $a_t \sim \pi$ 取期望，由式 (3) 得 $\mathbb{E}[\delta_t^{V} \mid s_t] = 0$ ——精确价值下，残差在状态层面也是零均值的。

式 (8) 把 8.1 的 baseline 机器接到了可计算对象上： $\delta_t^{V}$ 是 advantage 的**一步无偏估计**（V 精确时）。对照 Monte Carlo 的 $G_t - b$ ： $\delta$ 只含一步奖励与一步转移的随机性，方差小一个量级维度；代价是 $V \neq V^{\pi}$ 时它有偏——偏差随 $V$ 的误差线性进入。把「一步」推广成「 $k$ 步」再对 $k$ 做指数混合，就是 GAE。

**数值对照**（本书构造）：两步 MDP、 $\gamma = 1$ 、奖励只在末端。 $s_0$ 处动作 $a \in \lbrace L, R \rbrace$ 各以概率 0.5 被采、确定转移到 $s_1(a)$ ；末端 $R \sim \mathrm{Bernoulli}(p_a)$ ， $p_L = 0.8$ 、 $p_R = 0.2$ 。精确价值： $V(s_1(L)) = 0.8$ 、 $V(s_1(R)) = 0.2$ 、 $V(s_0) = 0.5$ 。对 $a_0 = L$ 这一步的 credit：Monte Carlo 给 $R - 0.5 \in \lbrace 0.5, -0.5 \rbrace$ ，二阶矩 $\mathbb{E}[(R-0.5)^2 \mid L] = 0.25$ ——八成把握的好结局仍有两成样本（ R = 0 ，概率 0.2）给出负权重；TD 残差给 $\delta_0 = 0.8 - 0.5 = 0.3$ ，逐样本恒等、方差为零——末端采样的 Bernoulli 噪声被整个吸收进第 1 步的残差（ $\delta_1 = R - 0.8$ ，零均值、不再触碰 $a_0$ 的 credit）。bootstrap 把「下游噪声」与「本步 credit」分离的机制，在这个两格世界里就是全部故事；GAE 的 $\lambda$ 是把这个分离连续调粗的旋钮。

## 5. GAE：k 步 advantage 的指数加权

### 5.1 k 步 advantage 与它的伸缩形式

把式 (7) 的单步残差累加 $k$ 步，即 **k 步 advantage 估计**（GAE 论文式 (12)-(14) 的构造，[Schulman et al., 2016](../references.md)，[B]）：

$$
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^{l} \thinspace \delta_{t+l}^{V} = \sum_{l=0}^{k-1} \gamma^{l} \thinspace r_{t+l} + \gamma^{k} \thinspace V(s_{t+k}) - V(s_t) \qquad (9)
$$

**符号**：右端是伸缩（telescoping）结果——相邻两项的 $\gamma^{l+1} V(s_{t+l+1})$ 与 $-\gamma^{l} \cdot \gamma V(s_{t+l+1})$ 逐对相消，只剩首尾的 $-V(s_t)$ 与 $\gamma^{k} V(s_{t+k})$ 。读法： $k$ 步真实奖励 + $k$ 步后的 bootstrap − 当前估值。 $k = 1$ 即式 (7)； $k \to \infty$ 时（论文式 (15)）

$$
\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^{l} \thinspace r_{t+l} - V(s_t) \qquad (10)
$$

退化为「实测回报减 baseline」——Monte Carlo 形态回到台上。

**偏差分析**（本书推导，与论文「bias generally becomes smaller as $k \to \infty$ 」的论断同口径）。对式 (9) 取条件期望并与真 advantage 比对：真值 $Q^{\pi,\gamma}(s_t, a_t) = \mathbb{E}[\sum_{l \lt k} \gamma^{l} r_{t+l} + \gamma^{k} V^{\pi,\gamma}(s_{t+k}) \mid s_t, a_t]$ ，于是

$$
\mathbb{E}\left[ \hat{A}_t^{(k)} \mid s_t, a_t \right] - A^{\pi,\gamma}(s_t, a_t) = \gamma^{k} \thinspace \mathbb{E}\left[ (V - V^{\pi,\gamma})(s_{t+k}) \mid s_t, a_t \right] + \left( V^{\pi,\gamma}(s_t) - V(s_t) \right) \qquad (11)
$$

**符号**：右端两项分工明确。第二项只依赖 $s_t$ ——由 8.1 式 (20) 的 baseline 引理，它在策略梯度里期望为零，**不影响 PG 的偏差**；第一项是 $k$ 步后的价值误差、被 $\gamma^{k}$ 折扣——这才是 PG 感受到的偏差，量级被 $\gamma^{k} \thinspace \Vert V - V^{\pi,\gamma} \Vert_\infty$ 控制。 $k$ 越大偏差越小、方差越接近 Monte Carlo； $k = 1$ 偏差最大（全靠 bootstrap）、方差最小。 $k$ 是偏差-方差权衡的旋钮，GAE 把这个旋钮换成连续参数 $\lambda$ 。

### 5.2 GAE 定义与系数收集

**GAE（generalized advantage estimation）**：对全部 $k$ 步估计做几何加权平均（权重 $(1-\lambda)\lambda^{k-1}$ ，和为 1），再把 $\delta$ 的系数逐项收集（论文式 (16)，[Schulman et al., 2016](../references.md)，[B]）：

$$
\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \thinspace \hat{A}_t^{(k)} = \sum_{l=0}^{\infty} (\gamma\lambda)^{l} \thinspace \delta_{t+l}^{V} \qquad (12)
$$

**推导**（系数收集）： $\delta_{t+l}$ 在 $`\hat{A}_t^{(k)}`$ 中出现当且仅当 $k \gt l$ ，系数为 $\gamma^{l}$ ；加权后其总系数为

$$
(1-\lambda) \sum_{k=l+1}^{\infty} \lambda^{k-1} \gamma^{l} = (1-\lambda) \thinspace \gamma^{l} \thinspace \frac{\lambda^{l}}{1-\lambda} = (\gamma\lambda)^{l} \qquad (13)
$$

代回即得式 (12) 右端。 $\square$

两个端点（论文式 (17)(18)）： $\lambda = 0$ 时 $`\hat{A}_t = \delta_t^{V}`$ ——纯 bootstrap； $\lambda = 1$ 时回到式 (10) 的 Monte Carlo 形态。式 (12) 还有一个实现层形态：它是**反向递推** $\Lambda_t = \delta_t + \gamma\lambda \thinspace \Lambda_{t+1}$ 的展开——按 $t$ 从后往前扫一遍 buffer 即得全部 $\hat{A}$ ， $O(T)$ 一次通过（§6.2 的 CleanRL 代码就是这个递推）。

**偏差-方差权衡的定量式**。把式 (11) 代入式 (12) 的混合：对 $`\hat{A}_t^{\mathrm{GAE}}`$ 的 PG 相关偏差有（本书推导）

$$
\left\lvert \thinspace \mathbb{E}\left[ \hat{A}_t^{\mathrm{GAE}} \mid s_t, a_t \right] - A^{\pi,\gamma}(s_t, a_t) \thinspace \right\rvert_{\text{PG 相关部分}} \le \frac{\gamma(1-\lambda)}{1-\gamma\lambda} \thinspace \Vert V - V^{\pi,\gamma} \Vert_\infty \qquad (14)
$$

**符号**：左端已剔除只依赖 $s_t$ 的 baseline 型项（式 (11) 第二项，对每个 $k$ 相同，混合后不变）；右端因子由 $\sum_{k \ge 1} (1-\lambda)\lambda^{k-1}\gamma^{k} = \gamma(1-\lambda)/(1-\gamma\lambda)$ 给出。这个因子对 $\lambda$ 单调减： $\lambda = 0$ 时为 $\gamma$ （一步 bootstrap 的最大偏差）， $\lambda \to 1$ 时趋于 0 。两点读法：其一，sup 范数界是悲观口径——它假设价值误差在 $k$ 步后的状态上仍与当前动作同向，实践中 $\mathbb{E}[(V - V^{\pi,\gamma})(s_{t+k}) \mid s_t, a_t]$ 随混合向误差的平稳均值收缩，同 $\lambda$ 下的真实偏差通常远小于此界；其二，即便按悲观界，因子的形状也说明「 $\lambda$ 微调偏差」的正确理解是把它推向 1 的过程， $\lambda$ = 0.95 处界仍约 0.83（即 0.84 $\gamma$ ）—— $V$ 的精度始终是前提， $\lambda$ 不是免检牌。另一面是方差： $(\gamma\lambda)^{l}$ 的权重让残差求和的有效窗长 $\sim 1/(1-\gamma\lambda)$ ， $\lambda$ 越大、参与求和的随机项越多，方差越接近 Monte Carlo。代入默认配置给这个窗长一个数： $\gamma$ = 0.99、 $\lambda$ = 0.95 时 $\gamma\lambda$ = 0.9405、有效窗长约 17 步； $\lambda$ 降到 0.9 时约 9 步； $\lambda$ = 0 时退到 1 步。换言之中等 $\lambda$ 已经把信用分配的时间尺度压到一个 episode 的零头——这是它相对 Monte Carlo（窗长 = 剩余全程）省方差的来源，也是对 $\gamma^{k} \Vert V - V^{\pi} \Vert$ 这条偏差链愿意多付的代价。论文对两个参数的原始口径值得逐字登记（[Schulman et al., 2016](../references.md)，[B]，§3）：「 $\gamma \lt 1$ 无论价值函数准不准都给策略梯度引入偏差； $\lambda \lt 1$ 只在价值函数不准时引入偏差」——前半句对应 8.1 §4.3 注记二的折扣约定，后半句正是式 (14) 的定性版。论文在其连续控制任务上的经验是最优 $\lambda$ 明显低于最优 $\gamma$ （同节，其任务域口径；LLM 场景的实现口径随 8.3 / 8.4 展开）。

**精确 $V$ 下的一个强结论**（本书推导）：当 $V = V^{\pi,\gamma}$ 精确成立，对任意 $\lambda$ 有 $`\mathbb{E}[\hat{A}_t^{\mathrm{GAE}} \mid s_t, a_t] = A^{\pi,\gamma}(s_t, a_t)`$ —— $l \ge 1$ 的残差项经「先对 $a_{t+l} \sim \pi$ 取期望」内层为零（式 (8) 推论与式 (3)），塔性质收尾。即价值精确时 GAE 对任意 $\lambda$ 条件无偏， $\lambda$ 此时是纯粹的方差旋钮。

**$\gamma$-just 框架**（论文 §2 的分析语言，Definition 1 与 Proposition 1）：论文把折扣策略梯度的实践估计器写成 $g^{\gamma} = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace A^{\pi,\gamma}(s_t, a_t)]$ （状态的 $\gamma^{t}$ 权重按惯例省略——8.1 §4.3 注记二），并称 $`\hat{A}_t`$ 为 **$\gamma$-just** ，若 $`\mathbb{E}[\hat{A}_t \thinspace \nabla \log \pi_t] = \mathbb{E}[A^{\pi,\gamma} \thinspace \nabla \log \pi_t]`$ 对每个 $t$ 成立——此时用 $\hat{A}$ 拼出的估计器期望恰为 $g^{\gamma}$ 。其 Proposition 1 给出充分条件： $`\hat{A}_t = Q_t - b_t`$ ，其中 $Q_t$ 是 $Q^{\pi,\gamma}(s_t, a_t)$ 的无偏估计、 $b_t$ 只依赖 $a_t$ 之前的历史——结构正是「无偏未来项减 baseline」。GAE 在 $V$ 精确时落进这个框架（上面的条件无偏结论更强）； $V$ 不准时它是有偏但低方差的折中，论文的策略梯度估计写作 $`g^{\gamma} \approx \mathbb{E}[\sum_t \nabla \log \pi_t \thinspace \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}]`$ 。论文另注：该构造与 TD( $\lambda$ )（[Sutton & Barto](../references.md)，[B]）同型——TD( $\lambda$ ) 估计的是价值函数，GAE 估计的是 advantage。

## 6. Actor-Critic 联合 loss 与 CleanRL 走读

### 6.1 三项联合 loss

actor 与 critic 在同一批 rollout 上联合训练，目标三项拼装（最大化口径写成 loss 即取负）：

$$
L(\theta, \varphi) = -\mathbb{E}_t\left[ \log \pi_\theta(a_t|s_t) \thinspace \hat{A}_t \right] + c_v \thinspace \mathbb{E}_t\left[ \left( V_\varphi(s_t) - \hat{R}_t \right)^{2} \right] - c_e \thinspace \mathbb{E}_t\left[ H\left( \pi_\theta(\cdot|s_t) \right) \right] \qquad (15)
$$

**符号**：第一项是 8.1 式 (26) 的伪损失（ $\hat{A}$ 常数化）； $`\hat{R}_t = \hat{A}_t + V_\varphi(s_t)`$ （stop-gradient）是 critic 的回归目标—— $\lambda$ -return 形态（ $\lambda = 1$ 时即实测回报）； $H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \log \pi(a|s)$ 是策略熵； $c_v, c_e$ 为权重。熵奖励（entropy bonus）的机制位置：8.1 §3.1 已见高置信区间 score 幅度收缩、策略越确定梯度信号越弱——熵项直接惩罚这种收缩，维持探索所需的随机性；权重过小则熵塌缩（entropy 塌缩的工业对策归 8.4 的 DAPO 段），过大则策略噪声盖过 reward 信号。联合 loss 的数据流：

```
rollout（当前 π_θ 采 N_traj 条轨迹，缓冲 states / actions / rewards / dones）
        │
        ├─ critic 前向：V_φ(s_t) 逐帧估值
        ▼
  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)  ──反向递推──►  Â_t = Σ_l (γλ)^l δ_{t+l}
        │                                            │
        │        critic 目标 R̂_t = Â_t + V(s_t) ◄────┘
        ▼                                            ▼
  L_V = (V_φ(s_t) − R̂_t)²            L_pg = −log π_θ(a_t|s_t)·Â_t − c_e·H
        └────────────── 一次 backward 同更 θ 与 φ ──────────────┘
```

### 6.2 CleanRL `ppo.py` 文件级走读

CleanRL 是单文件教学实现仓库（vwxyzjn/cleanrl repo，教学参照；下述代码事实核实于 2026-11-17，文件 `cleanrl/ppo.py`）。它是 PPO 实现——clip 目标本身归 8.3，本章只看它的 actor-critic 骨架与 GAE 计算。

**网络结构**（`class Agent(nn.Module)`）：critic 与 actor 是两个独立 MLP——`self.critic` 以 `Linear(64, 1)` 收尾输出标量估值，`self.actor` 以 `Linear(64, n_act)` 收尾输出 logits；`get_action_and_value` 一次前向返回 `(action, log_prob, entropy, value)` ，熵来自 `Categorical(logits=logits).entropy()` 。两网不共享 trunk，但共用**一个** optimizer、一次 backward、一组梯度裁剪（ `nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)` ——2.4 的 global norm 机制，默认阈值 0.5）。

**GAE 计算块**（rollout 结束后、 `with torch.no_grad():` 下，逐行即式 (7)(12) 的反向递推）：

```python
next_value = agent.get_value(next_obs).reshape(1, -1)
advantages = torch.zeros_like(rewards).to(device)
lastgaelam = 0
for t in reversed(range(args.num_steps)):
    if t == args.num_steps - 1:
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
    else:
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
returns = advantages + values
```

逐行对回公式。 `delta` 行即式 (7)， `lastgaelam` 行即递推 $\Lambda_t = \delta_t + \gamma\lambda \cdot m_{t+1} \cdot \Lambda_{t+1}$ （ $m$ 为终止掩码）——展开恰是式 (12) 在 episode 边界截断的版本： $t$ 处动作导致终止时 `dones[t+1]` = 1，下一步的 bootstrap 值与残差链同被清零。 `dones` 数组的口径值得点破： rollout 循环里 `dones[step]` 存的是与 `obs[step]` 同时到达的终止标志（上一步是否终结了 episode），所以递推里给 $V(s_{t+1})$ 配 mask 用的是 `dones[t + 1]` ——下标错位一位是存储口径而非笔误。最后一行 `returns = advantages + values` 即 $`\hat{R}_t = \hat{A}_t + V(s_t)`$ ——critic 的 $\lambda$ -return 回归目标，正是式 (15) 第二项的目标构造。默认超参（同文件 Args）： $\gamma$ = 0.99、 $\lambda$ = 0.95、 `num_envs` = 4、 `num_steps` = 128。

**更新阶段**的 loss 行与式 (15) 逐项对应（minibatch 内）： `norm_adv` 开时先把 advantage 标准化为 `(adv − mean) / (std + 1e-8)` ； `pg_loss = max(−A·ratio, −A·clamp(ratio))` ——clip 的逐情形分析归 8.3，此处只需注意 `ratio = 1` 处其梯度恰为 $-\hat{A} \thinspace \nabla \log \pi$ 、即式 (15) 第一项； `v_loss = 0.5 * ((newvalue − b_returns) ** 2).mean()` 即第二项； `entropy_loss = entropy.mean()` ；总损失 `loss = pg_loss − args.ent_coef * entropy_loss + v_loss * args.vf_coef` （默认 `ent_coef` = 0.01、 `vf_coef` = 0.5）——式 (15) 的符号约定逐字对上。

教学骨架（教学用，非优化实现）把递推写成自含函数，口径与上面代码一致（ `dones[t]` = $s_t$ 是否处于 episode 边界之后）：

```python
# 教学用，非优化实现：GAE(γ,λ) 反向递推 + λ-return 目标（与 CleanRL ppo.py 的 lastgaelam 循环同构）
# dones[t] 为与 s_t 同时到达的终止标志（上一步是否终结 episode）；next_done/next_value 为缓冲外的一步
def compute_gae(rewards, values, next_value, dones, next_done, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done
            nv = next_value
        else:
            nonterminal = 1.0 - dones[t + 1]
            nv = values[t + 1]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]      # 式 (7)
        last = delta + gamma * lam * nonterminal * last                # 式 (12) 的反向递推
        adv[t] = last
    return adv, adv + values                                           # (Â_t, R̂_t)
```

## 7. LLM 场景：末端奖励下的退化形态

把 GAE 机器装进 8.1 §2.2 的退化 MDP（ $\gamma = 1$ ， $r_t = 0$ 对 $t \lt T-1$ ，末端 $r_{T-1} = R(x, y)$ ），残差逐段显形：

$$
\delta_t^{V} = V(s_{t+1}) - V(s_t) \quad (t \lt T-1), \qquad \delta_{T-1}^{V} = R(x, y) - V(s_{T-1}) \qquad (16)
$$

中间步骤的残差纯由 critic 自己的相邻估值差构成——环境在末端之前不提供任何信号。代入式 (12)（ $\gamma = 1$ ）：

$$
\hat{A}_t^{\mathrm{GAE}(1, \lambda)} = \sum_{l=0}^{T-2-t} \lambda^{l} \thinspace \left( V(s_{t+l+1}) - V(s_{t+l}) \right) + \lambda^{T-1-t} \thinspace \left( R - V(s_{T-1}) \right) \qquad (17)
$$

**符号**：右端第一项是路径上 $V$ 的逐段差的 $\lambda$ 折扣和，第二项是末端真相的 $\lambda$ 衰减。 $\lambda = 1$ 时第一项伸缩相消、全式塌成

$$
\hat{A}_t^{\mathrm{GAE}(1, 1)} = R(x, y) - V(s_t) \qquad (18)
$$

即 8.1 §7.2 的结论在此兑现： $\gamma = 1$ 加末端奖励下，token 级 advantage 的全部 per-token 差异都由 $V(s_t)$ 沿路径的取值承载——reward 本身对每个 token 一视同仁。 $V$ 的准确含义是 $V^{\pi}(s_t) = \mathbb{E}[R \mid x, y_{\le t}]$ ：给定前缀的结局期望——一个「过程分」形态的对象，但它的训练信号只有末端标量。

困难在三层叠加。其一，**信号稀疏的转移而非消失**：critic 要从「整段回答一个好/坏」学出「哪个前缀好/坏」，监督信号是同一个 $R$ 在全部前缀上的复用——token 级 credit assignment 被从 actor 侧转移到 critic 侧，没有消失。critic 的学习动力学会按信息量排队展开：先学到的是 prompt 级均值（ $\mathbb{E}[R \mid x]$ ，同一条轨迹内所有前缀共享的最粗划分），然后才是前缀之间有分辨力的分裂——而 per-token advantage 的用处恰恰只在这第二层之后才开始兑现。其二，**目标沿序列强相关**： $`\hat{R}_t = \hat{A}_t + V(s_t)`$ 在同一轨迹内共享同一个末端 $R$ ，critic 每步回归的有效样本数接近序列数而非 token 数；长 response 下 critic 的收敛速度被这个相关性直接拖慢。其三，**critic 本身是第二个要驻留显存、要训练的网络**——系统账（四模型驻留）归 8.5。

这三条合起来就是 8.4 的动机源头：GRPO 的选择是干脆去掉 critic，用同 prompt 下 $G$ 个采样的组均值作 baseline——在式 (18) 的形态里，这相当于用「组内经验均值」替代「学出来的前缀期望」，用统计量换掉一个网络；组基线的偏差-方差分析（含 $(G-1)/G$ 因子）归 8.4。工程上另一件常做的事是把 reward 整形为非退化形态——逐 token 的 KL 惩罚加末端 $R$ ——让中间步骤的 $r_t$ 不再恒零，8.4 与 9.2 展开。

## 总结

| 主题 | 公式 | 要点 |
| --- | --- | --- |
| 价值三件套 | 式 (1)(2) | $A = Q - V$； $`\mathbb{E}_{a \sim \pi}[A] = 0`$（式 (3)） |
| Bellman 方程 | 式 (4) | 价值 = 一步奖励 + 折扣后估值；LLM 下 $P$ 确定但 $r$ 末端退化 |
| Critic 学习 | 式 (5)(6) | TD(0) bootstrap 目标 + 回归 loss；半梯度口径（target stop-grad）；on-policy 移动靶 |
| $\delta$ 无偏引理 | 式 (7)(8) | $V$ 精确时 $\mathbb{E}[\delta_t \mid s_t, a_t] = A^{\pi,\gamma}(s_t, a_t)$；GAE 的出发点 [B] |
| k 步 advantage | 式 (9)(10) | 伸缩形式；PG 相关偏差被 $\gamma^{k} \Vert V - V^{\pi} \Vert_\infty$ 控制（式 (11)） |
| GAE | 式 (12)(13) | k 步估计的几何混合 = $\delta$ 的 $(\gamma\lambda)^{l}$ 折扣和；反向递推 $O(T)$ [B] |
| 偏差-方差 | 式 (14) | 偏差因子 $\gamma(1-\lambda)/(1-\gamma\lambda)$ 随 $\lambda$ 减；论文口径： $\lambda \lt 1$ 只在 $V$ 不准时引入偏差 |
| 联合 loss | 式 (15) | policy 伪损失 + value 回归（目标 $`\hat{R}_t = \hat{A}_t + V(s_t)`$）+ entropy bonus |
| CleanRL | §6.2 | `cleanrl/ppo.py`：双 MLP 共享 optimizer；GAE 反向递推逐行对回式 (7)(12)； $\gamma$=0.99、 $\lambda$=0.95、 $c_e$=0.01、 $c_v$=0.5（教学参照） |
| LLM 退化 | 式 (16)-(18) | $\gamma$=1 + 末端奖励 ⇒ $\lambda$=1 时 $`\hat{A}_t = R - V(s_t)`$；credit 困难转移给 critic ⇒ 8.4 去 critic 动机 |

## 练习

1.（推导）从式 (7) 出发证明式 (9) 的伸缩形式（写出 $\delta_t + \gamma \delta_{t+1} + \gamma^{2} \delta_{t+2}$ 的展开与相消）；再由式 (9) 证式 (11) 的偏差分解，并说明为什么 $(V^{\pi} - V)(s_t)$ 项不进入策略梯度的偏差。

2.（推导）证明「精确 $V$ 下 GAE 对任意 $\lambda$ 条件无偏」：即 $V = V^{\pi,\gamma}$ 时 $`\mathbb{E}[\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} \mid s_t, a_t] = A^{\pi,\gamma}(s_t, a_t)`$（提示：对 $l \ge 1$ 的项，先对 $a_{t+l} \sim \pi(\cdot|s_{t+l})$ 取内层期望，用式 (3)）。这个结论与论文「 $\lambda \lt 1$ 只在价值函数不准时引入偏差」如何对上？

3.（计算）取 $T = 4$ 、 $(r_0, r_1, r_2, r_3) = (0, 0, 0, 1)$ 、 $(V(s_0), \ldots, V(s_3)) = (0.3, 0.4, 0.2, 0.6)$ 、 $V(s_4) = 0$ （终止）：(a) 手算 $\gamma = 1$ 、 $\lambda = 1$ 与 $\lambda = 0.5$ 两组下的全部 $`\hat{A}_t`$ ，验证 $\lambda = 1$ 组满足式 (18)；(b) 计算 $`\hat{R}_t = \hat{A}_t + V(s_t)`$ 并说明 $\lambda = 0.5$ 组中 critic 目标的构成；(c) 把 $V$ 换成真值 $V^{\pi}(s_t) = \mathbb{E}[R \mid s_t]$（设 $\mathbb{E}[R] = 0.5$ 且前缀无信息，即全部 $V^{\pi}(s_t) = 0.5$ ），重算两组的 $`\mathbb{E}[\hat{A}_t]`$ 并对照练习 2 的结论。

4.（代码）用 §6.2 的教学骨架实现 `compute_gae`，再用 $\sum_l (\gamma\lambda)^{l} \delta_{t+l}$ 的直接求和版对拍（随机 $T = 50$ 、含两个 episode 边界的合成数据，断言逐元素相等）；然后把 $\lambda$ 从 0 扫到 1，在「 $V$ 带固定比例噪声」的合成设定下画出 $`\hat{A}_0`$ 的偏差-方差曲线（5000 次独立重复估计），标出式 (14) 预言的偏差因子形状。

## 参考

- [Schulman et al., 2016](../references.md)：High-Dimensional Continuous Control Using Generalized Advantage Estimation，arXiv:1506.02438，[B]（key: gae）—— $\delta_t^{V}$ 定义（式 (7)）、k 步估计与指数加权（式 (9)(12)，论文式 (12)-(18)）、 $\gamma$-just 框架（其 Definition 1 / Proposition 1）、「 $\gamma \lt 1$ 恒有偏、 $\lambda \lt 1$ 只在 $V$ 不准时有偏」与最优 $\lambda$ 低于最优 $\gamma$ 的经验口径（其 §3）
- [Sutton & Barto, 2018](../references.md)：Reinforcement Learning: An Introduction（第 2 版），MIT Press，[B]（key: sutton-barto）——Bellman 方程（§3.5-3.6）、TD(0) 与半梯度口径（§6.1、§9.3、§11.5）、TD( $\lambda$ )（第 12 章，GAE 的同型对照）
- [Huang et al., CleanRL repo](../references.md)：vwxyzjn/cleanrl（key: cleanrl，教学参照）——`cleanrl/ppo.py`：`Agent`（双 MLP）、GAE 反向递推块、 `returns = advantages + values`、联合 loss 与默认超参（ $\gamma$=0.99、 $\lambda$=0.95、 `ent_coef`=0.01、 `vf_coef`=0.5、 `max_grad_norm`=0.5）；代码事实核实于 2026-11-17
