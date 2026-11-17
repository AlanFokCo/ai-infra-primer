# 8.1 MDP、策略梯度定理与 REINFORCE（MDP, the Policy Gradient Theorem, and REINFORCE）

> 本章从「reward 不可微」这个硬约束出发，建立 LLM 强化学习的全部基础件：MDP 形式化与 LLM 场景的 token-level 退化结构（state = prompt + 已生成前缀、action = 下一 token、reward 集中在末端）、log-derivative trick 逐步展开的策略梯度定理（轨迹形式与状态分布形式两种）、reward-to-go 因果引理、baseline 无偏性证明与最优 baseline 的方差最小化推导、REINFORCE 算法，以及长序列下的方差分析（噪声地板随 T 线性增长、末端奖励下 reward-to-go 失效、baseline 成为唯一杠杆），最后落在样本效率问题——on-policy 约束与 rollout 成本是 8.3 importance sampling 与 PPO 的动机 | 前置：2.2（AdamW）、3.2 §5（在线 / 离线分界定义）、models 1.1（语言建模的链式分解）、1.1（vjp 与 autograd 机制） | 锚点：Williams 1992 [B]（REINFORCE 原始出处，无 arXiv）、Sutton et al. 1999 [B]（函数逼近策略梯度定理，无 arXiv）、Sutton & Barto 教材第 13 章 [B]、Christiano et al. 2017 [B]（RLHF 语境一句）

## 1. 动机：reward 不可微，只有样本可用

后训练阶段的 reward——reward model 的标量打分，或 verifier 的对错判定——对策略参数 $\theta$ 没有梯度通路。SFT 的 CE 能对 $\theta$ 求导，因为 loss 是模型输出的可微函数，梯度沿计算图 vjp 回传（1.1）；RL 的 reward 只依赖采样得到的回答 $y$， $\theta$ 影响 reward 的唯一路径是改变 $y$ 的采样分布。问题的形式化是

$$
\max_\theta \thinspace \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[ R(x, y) \right]
$$

期望号下的分布依赖 $\theta$ ，被期望的函数 $R$ 只能前向求值——RM 是另一个网络，verifier 是符号判定，两者都不提供 $\partial R / \partial \theta$ 。这类「分布含参数、函数只求值」的优化问题有通用解法：score function estimator（对数导数技巧），它就是策略梯度的全部内容。

## 2. MDP 形式化与 LLM 的退化结构

### 2.1 MDP 五元组与轨迹分布

马尔可夫决策过程（Markov decision process, MDP）记为 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$ 加初始状态分布 $p_0$ ：

- $\mathcal{S}$ ：状态空间； $\mathcal{A}$ ：动作空间；
- $P(s'|s, a)$ ：转移核——在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率；
- $r_t = r(s_t, a_t, s_{t+1})$ ：第 $t$ 步动作后收到的标量奖励。本书约定 $r_t$ 跟随 $(s_t, a_t)$ ，与 GAE 论文同口径（8.2 沿用），与 Sutton-Barto 的 $r_{t+1}$ 记号差一个下标平移；
- $\gamma \in [0, 1]$ ：折扣因子（discount factor）。

策略（policy） $\pi_\theta(a|s)$ 是参数化的动作分布。一条轨迹（trajectory） $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots, s_T)$ 的联合分布分解为

$$
p_\theta(\tau) = p_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \thinspace P(s_{t+1}|s_t, a_t) \qquad (1)
$$

**符号**： $T$ ：回合长度（有限视界，episodic 设定）；乘积的三类因子分别是初始分布、策略、环境转移。从第 $t$ 步出发的折扣回报（reward-to-go）记为

$$
G_t = \sum_{l=0}^{T-1-t} \gamma^{l} \thinspace r_{t+l} \qquad (2)
$$

**符号**： $G_t$ ：从 $t$ 起的折扣回报和； $\gamma = 1$ 时退化为剩余奖励的简单求和。整条轨迹的回报 $R(\tau) = G_0$ 。

### 2.2 LLM 映射：全部随机性都在策略里

把 LLM 生成装进 MDP 框架，五元组逐项落地：

| MDP 件 | LLM 对应 |
| --- | --- |
| 状态 $s_t$ | prompt $x$ 拼接已生成前缀 $y_{\le t}$ |
| 动作 $a_t$ | 下一个 token $y_{t+1}$ ；动作空间 = 词表 $\mathcal{V}$ ，量级 $10^{5}$ |
| 转移 $P$ | 确定性拼接： $s_{t+1} = (s_t, a_t)$ ，概率为 1 |
| 奖励 $r_t$ | $t \lt T-1$ 时为 0；末端 $r_{T-1} = R(x, y)$ |
| 终止 | EOS token 或长度上限 $T_{\max}$ |
| $\gamma$ | 惯例取 1（有限视界、无折扣） |

对应的状态链：

```
s_0 = x ──a_0=y_1──► s_1=(x,y_1) ──a_1=y_2──► s_2=(x,y_1,y_2) ──► ⋯ ──► s_T=(x,y)
  │         r_0=0           r_1=0                                  r_{T-1}=R(x,y)
  └─ π_θ(·|s_t) = LM 在位置 t 的 softmax 输出        ↑ reward model 或 verifier（不可微黑盒）
```

这个映射的结构注记有三条。其一，**转移完全已知且确定**：环境没有未知动力学，轨迹分布式 (1) 里的 $P$ 与 $p_0$ 都是已知的（拼接规则与 prompt 分布），MDP 的随机性全部由 $\pi_\theta$ 自己携带——「探索」在这里就是采样温度下的 token 抽样。其二，**奖励集中在末端**：中间步骤没有任何即时奖励信号，token-level MDP 是退化结构；§7 会看到这让因果引理变成空操作。其三，**同一个对象的两种粒度**：sequence-level 视角把整个回答 $y$ 当作一个动作（contextual bandit，prompt 为 context），token-level 视角把每个 token 当作一步。两者严格等价——语言建模的链式分解

$$
\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t|x, y_{\lt t})
$$

（models 1.1 的 LM 定义本身）让整条序列的 score 恰好是逐 token score 之和；§7.1 会证明，不带 per-token baseline 时两种粒度的梯度估计器逐字相同，差别只在 credit assignment 的粒度工具上。

reward 的来源一句带过：RLHF 路线的 $R$ 来自人类偏好数据训出的 reward model（[Christiano et al., 2017](../references.md)，arXiv:1706.03741，[B]；RM 训练归 9.2），RLVR 路线来自规则验证器（数学等价判定、代码单测，归 9.5）。本章只需要一条性质： $R(x, y)$ 是只能前向求值的标量黑盒。

**终止的形式化**一句：EOS 之后状态进入吸收态（absorbing state，转移回自身、奖励恒零），有限回合由此嵌入无穷视界记号， $\gamma = 1$ 下各级数仍因 $T$ 有界而收敛。长度上限 $T_{\max}$ 造成的截断样本要单独注意：轨迹在生成完成前被强行终止，其 $R$ 仍是良定义的（RM 对截断文本照样打分），但样本不再代表「策略想生成的完整回答」——截断样本进不进 batch、以什么权重进，是 8.4 才处理的工程决策（DAPO 的 overlong 处理），本章只需记住截断 ≠ 正常终止。

## 3. 目标函数与 log-derivative trick

Episodic 未折扣目标（ $\gamma = 1$ ）写成轨迹期望：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \right] = \sum_{\tau} p_\theta(\tau) \thinspace R(\tau) \qquad (3)
$$

**符号**：求和遍历全部可能轨迹（LLM 场景下轨迹空间有限但天文数字大—— $T$ 步、每步 $|\mathcal{V}|$ 个分支——显式求和不可行，只能采样）。 $R(\tau)$ 的值不依赖 $\theta$ ：它只依赖轨迹内容； $\theta$ 只通过采样分布进入目标。逐步展开求导。

**第一步，梯度穿过求和**：

$$
\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta p_\theta(\tau) \thinspace R(\tau) \qquad (4)
$$

求和与梯度的交换在有限轨迹集下逐字成立；连续或无限视界情形需要支配收敛类的正则条件，本书按成立处理（教材同口径，[Sutton & Barto, 2018](../references.md)，[B]，§13.1-13.2）。

**第二步，log-derivative trick**。对任意正的概率值， $\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \thinspace \nabla_\theta \log p_\theta(\tau)$ ——链式法则作用于 $\log p$ 再乘回去。代入式 (4)：

$$
\nabla_\theta J = \sum_{\tau} p_\theta(\tau) \thinspace R(\tau) \thinspace \nabla_\theta \log p_\theta(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \thinspace \nabla_\theta \log p_\theta(\tau) \right] \qquad (5)
$$

「对分布求导」变成「在分布下求期望」——采样估计的大门由此打开：不需要枚举轨迹，从 $\pi_\theta$ 采一批、算加权 score 的样本均值即可。这一个代数恒等式是全章的支点，它的适用条件是 $p_\theta(\tau) \gt 0$ 在 $R(\tau) \neq 0$ 处成立（绝对连续；softmax 策略对所有有限序列输出正概率，自动满足）。

**第三步，展开** $\log p_\theta(\tau)$ 。对式 (1) 取对数：

$$
\log p_\theta(\tau) = \log p_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1}|s_t, a_t) \qquad (6)
$$

三类因子中只有策略项依赖 $\theta$ ——初始分布与转移核的 log 项被 $\nabla_\theta$ 清零：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \qquad (7)
$$

式 (7) 是策略梯度全部工程含义的源头：**环境的转移概率与初始状态分布不需要知道，也不需要可微**。对 LLM 这是字面成立—— $P$ 是确定性拼接、 $p_0$ 是 prompt 分布，两者本就与 $\theta$ 无关。式 (7) 右端的单步项 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 称为 score function（得分函数），它有一个贯穿全章的性质：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[ \nabla_\theta \log \pi_\theta(a|s) \right] = \sum_a \pi_\theta(a|s) \thinspace \nabla_\theta \log \pi_\theta(a|s) = \sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0 \qquad (8)
$$

**符号**：第一步期望展开、第二步 $\pi \cdot \nabla \log \pi = \nabla \pi$ 、第三步梯度与求和交换、末步概率归一。score 在当前策略下零均值——§5 的 baseline 合法性与 §6 的方差分解都从这一条长出来。

三步合起来：

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \qquad (9)
$$

LLM 读法： $R(\tau) = R(x, y)$ ，内层和式是对 response 全部 token 的 score 求和。注意式 (9) 里每个 token 的 score 都乘**同一个**整条轨迹的回报——「哪个 token 把回答带好了」的信息完全没有被区分，这是 credit assignment 问题在本章的形态，也是方差问题的根源（§6）。

### 3.1 softmax 策略的 score 闭式

LLM 的策略是词表上的 categorical 分布： $`\pi_\theta(a|s) = \mathrm{softmax}(z)_a`$ ， $z$ 为模型输出的 logits。score 在 logit 空间有闭式——对 $\log \pi_\theta(a|s) = z_a - \log \sum_k e^{z_k}$ 逐坐标求导：

$$
\frac{\partial \log \pi_\theta(a|s)}{\partial z_j} = \delta_{ja} - \pi_\theta(j|s)
$$

**符号**： $\delta_{ja}$ ：Kronecker delta（ $j = a$ 时为 1，否则 0）。右端即「one-hot 减概率向量」——与 1.2 的 softmax-CE 融合反向是同一个 Jacobian（ $p - \delta_y$ 形态）：策略梯度的 score 与监督 CE 的 logit 梯度共享同一条反向通路，区别只在 CE 里目标 token 由数据给定、这里动作由策略自己采样，且外面多乘一个 credit 权重。式 (8) 的零均值在闭式下可直接验算： $`\mathbb{E}_a[\nabla_z \log \pi] = \sum_a \pi_a (e_a - p) = p - p = 0`$（ $e_a$ 为 one-hot 基向量）。

参数梯度由链式法则经模型内部回传（1.1 的 vjp 机制），实现层只需 gather 所采 token 的 $\log \pi$ 再 backward（§6 的伪损失写法），logit 闭式由 autograd 自动处理。但显式形态解释了一个动力学事实：某 token 被押到 $p_a \to 1$ 时， $e_a - p$ 的各分量趋零——高置信区间的 score 幅度自然收缩，策略越确定、梯度信号越弱；这是 entropy 管理与探索问题的第一道数学影子（8.2 的 entropy bonus、8.4 的 entropy 塌缩动机都在这条性质上做文章）。

## 4. 策略梯度定理：因果引理、Q 形式、状态分布

式 (9) 是轨迹形式的梯度。策略梯度定理（policy gradient theorem）把它改写成「逐步 score × 该步动作的价值」的形式。

### 4.1 因果引理：动作影响不了过去的奖励

**引理**：对 $t' \lt t$ ，有 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace r_{t'}] = 0$ 。

**证明**：对 $t$ 步之前的历史 $h_t = (s_0, a_0, \ldots, s_t)$ 取条件期望。 $t' \lt t$ 时 $r_{t'}$ 由 $h_t$ 完全决定，在条件号下是常数；内层期望由式 (8) 为零：

$$
\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace r_{t'} \right] = \mathbb{E}\left[ r_{t'} \thinspace \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \mid h_t \right] \right] = \mathbb{E}\left[ r_{t'} \cdot 0 \right] = 0 \qquad (10)
$$

$\square$

把式 (9) 中的 $R(\tau) = \sum_{t'} r_{t'}$ 对每个 $t$ 拆成「过去」（ $t' \lt t$ ）与「未来」（ $t' \ge t$ ）两段，过去项的贡献被引理清零，每个 score 只需乘它**之后**的奖励：

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace G_t \right] \qquad (11)
$$

这一步去掉的是估计器里「与本步动作无关的噪声源」：过去的奖励乘上零均值 score，期望为零、只贡献方差（§6 把它算成账）。

**数值化这条引理的方差账**（本书构造）：设第 0 步奖励 $r_0$ 为 $\pm 1$ 等概率（由环境噪声决定、与后续动作无关），第 1 步 score $g_1$ 为 $\pm 1$ 等概率且与 $r_0$ 独立。式 (9) 形式的估计器含 $r_0 \thinspace g_1$ 项：期望 $\mathbb{E}[r_0 g_1] = 0$ 、二阶矩 $\mathbb{E}[r_0^{2} g_1^{2}] = 1$ ——每个「过去奖励 × 未来 score」项贡献一份纯方差， $K$ 个无关过去奖励就是 $K$ 份。若环境噪声的幅度是真实 credit 信号的 100 倍，未剔除的过去项把估计器标准差放大约 100 倍（方差 $10^{4}$ 倍）、信噪比掉两个数量级——因果引理不是美学化简，是方差的数量级管理。

### 4.2 Q 形式：塔性质一步

$G_t$ 在给定 $(s_t, a_t)$ 下的条件期望是动作价值函数（action-value function）：

$$
Q^{\pi}(s, a) = \mathbb{E}_{\tau \sim \pi}\left[ G_t \mid s_t = s, \thinspace a_t = a \right] \qquad (12)
$$

**符号**： $Q^{\pi}$ ：在 $s$ 执行 $a$ 后沿 $\pi$ 继续的期望回报。对式 (11) 用塔性质（先对 $(s_t, a_t)$ 条件化，再对外层取期望），马尔可夫性保证 $G_t$ 的将来只经 $(s_t, a_t)$ 进入：

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace Q^{\pi}(s_t, a_t) \right] \qquad (13)
$$

即梯度 = 策略诱导轨迹分布下「score × 动作价值」的逐步期望之和。Sutton et al. 1999 对函数逼近策略给出了同一形式的证明，并指出它对任意可微策略参数化成立（[Sutton et al., 1999](../references.md)，NeurIPS 1999，无 arXiv，[B]）。 $Q^{\pi}$ 不可查表时——LLM 场景永远如此——它只是理论对象：实践中用采样回报（§5）或学习到的近似（8.2 的 critic）代替。

### 4.3 状态分布与 discounted 形式

改看折扣目标 $`J_\gamma(\theta) = \mathbb{E}_{s_0 \sim p_0}[V^{\pi_\theta}(s_0)]`$ ，其中状态价值函数

$$
V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{l=0}^{\infty} \gamma^{l} \thinspace r_{t+l} \mid s_t = s \right] \qquad (14)
$$

**符号**： $V^{\pi}$ ：从 $s$ 出发沿 $\pi$ 的期望折扣回报；这里视界取无穷（episodic 情形用吸收态处理，结论同形）。这次不走轨迹 trick，而从 Bellman 方程 $V^{\pi}(s) = \sum_a \pi_\theta(a|s) \thinspace Q^{\pi,\gamma}(s, a)$ 两端对 $\theta$ 求导（Sutton-Barto §13.2 的路线，[Sutton & Barto, 2018](../references.md)，[B]）：

$$
\nabla_\theta V^{\pi}(s) = \sum_a \left[ \nabla_\theta \pi_\theta(a|s) \thinspace Q^{\pi,\gamma}(s, a) + \pi_\theta(a|s) \thinspace \nabla_\theta Q^{\pi,\gamma}(s, a) \right] \qquad (15)
$$

**符号**： $Q^{\pi,\gamma}(s, a) = \mathbb{E}[r_t + \gamma V^{\pi}(s_{t+1}) \mid s_t = s, a_t = a]$ ：折扣动作价值。 $r$ 不依赖 $\theta$ ，所以

$$
\nabla_\theta Q^{\pi,\gamma}(s, a) = \gamma \sum_{s'} P(s'|s, a) \thinspace \nabla_\theta V^{\pi}(s') \qquad (16)
$$

代回式 (15)，右端再次含有 $\nabla_\theta V^{\pi}$ ——这是一个不动点方程。记 $\Pr(s \to x, k, \pi)$ 为策略 $\pi$ 下从 $s$ 恰好 $k$ 步转移到 $x$ 的概率，逐层展开该递归（每次展开把本状态的 score×Q 项留下、把 $\nabla V$ 项以概率 $\gamma \Pr(\cdot)$ 推向下一步）：

$$
\nabla_\theta V^{\pi}(s) = \sum_{x \in \mathcal{S}} \sum_{k=0}^{\infty} \gamma^{k} \thinspace \Pr(s \to x, k, \pi) \thinspace \sum_a \nabla_\theta \pi_\theta(a|x) \thinspace Q^{\pi,\gamma}(x, a) \qquad (17)
$$

展开的几何级数在 $\gamma \lt 1$ 时收敛（episodic 有限视界下亦然——乘积测度有界）。对 $s_0 \sim p_0$ 取期望，定义**折扣状态访问测度**（discounted state visitation measure）

$$
d^{\gamma}(x) = \sum_{t=0}^{\infty} \gamma^{t} \thinspace \Pr(s_t = x) \qquad (18)
$$

（概率对 $p_0$ 与 $\pi$ 的转移取），式 (17) 收拢成策略梯度定理的折扣-状态分布形式：

$$
\nabla_\theta J_\gamma = \sum_{t=0}^{\infty} \gamma^{t} \thinspace \mathbb{E}_{s_t, a_t}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace Q^{\pi,\gamma}(s_t, a_t) \right] = \sum_x d^{\gamma}(x) \sum_a \nabla_\theta \pi_\theta(a|x) \thinspace Q^{\pi,\gamma}(x, a) \qquad (19)
$$

三条注记把式 (19) 读全。第一，它是**恒等式**、无比例常数；把 $d^{\gamma}$ 归一化为 $\mu = d^{\gamma} / Z$（持续任务下 $Z = \sum_x d^{\gamma}(x) = 1/(1-\gamma)$ ，episodic 下 $Z$ 为期望回合长度），即得教材中常见的比例形式 $\nabla J \propto \sum_s \mu(s) \sum_a \nabla \pi \thinspace q_{\pi}$ ——比例系数被吸收进学习率（[Sutton & Barto, 2018](../references.md)，[B]，§13.2）。第二， $\gamma^{t}$ 的权重落在**状态**上：折扣目标的梯度更看重早期状态的策略改进；实践中一个广泛使用的变体把 $\gamma$ 只折进回报、丢掉状态的 $\gamma^{t}$ 权重（估计器写成 $\sum_t \nabla \log \pi_t \thinspace \Psi_t$ 、 $\Psi_t$ 内部含 $\gamma$ ）——8.2 会看到 GAE 论文正是采用这个约定并明确标注它相对无折扣目标已引入偏差。第三，LLM 场景 $\gamma = 1$ 时式 (19) 退化回式 (13) 的未折扣形态， $d^{\gamma}$ 的发散问题随有限 $T$ 消失。

## 5. Baseline：无偏性证明与最优 baseline

### 5.1 无偏性：减任意状态函数都不动期望

式 (13) 中的 $Q^{\pi}$ 可以减去任意只依赖状态的函数 $b(s)$ （baseline），期望不变。对固定的 $s$ ：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left[ \nabla_\theta \log \pi_\theta(a|s) \thinspace b(s) \right] = b(s) \thinspace \sum_a \nabla_\theta \pi_\theta(a|s) = b(s) \thinspace \nabla_\theta 1 = 0 \qquad (20)
$$

对外层状态分布再取期望仍为零。条件只有一条： $b$ 不依赖当前动作 $a$ ——它可以依赖 $s$ 与 $s$ 之前的历史（练习 2 证推广形式并给反例）。于是对任意 baseline：

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace \left( Q^{\pi}(s_t, a_t) - b(s_t) \right) \right] \qquad (21)
$$

期望不动、方差变——baseline 的全部作用是降方差。Sutton et al. 1999 在函数逼近设定下证明了同一结论（[Sutton et al., 1999](../references.md)，[B]）；选哪一族 $b$ 是设计自由，§5.2 给出「最优」的精确含义。

### 5.2 最优 baseline：方差最小化推导

哪个 $b(s)$ 把方差压得最狠？逐状态最小化估计器的二阶矩。固定 $s$ ，记 $g(a) = \nabla_\theta \log \pi_\theta(a|s)$ （向量），估计器 $X(a) = g(a) \thinspace (Q(s, a) - b)$ 。由式 (20)，均值 $\mathbb{E}[X] = \mathbb{E}[g \thinspace Q]$ 与 $b$ 无关，最小化方差等价于最小化

$$
\mathbb{E}_a\left[ \Vert X \Vert^2 \right] = \mathbb{E}_a\left[ \Vert g(a) \Vert^2 \thinspace \left( Q(s, a) - b \right)^2 \right] \qquad (22)
$$

**符号**： $\Vert \cdot \Vert$ ：参数空间欧氏范数；期望对 $a \sim \pi_\theta(\cdot|s)$ 。右端是 $b$ 的二次型，逐坐标求导置零：

$$
\frac{\partial}{\partial b} \thinspace \mathbb{E}_a\left[ \Vert g \Vert^2 (Q - b)^2 \right] = -2 \thinspace \mathbb{E}_a\left[ \Vert g \Vert^2 (Q - b) \right] = 0 \quad \Longrightarrow \quad b^{\ast}(s) = \frac{\mathbb{E}_a\left[ \Vert g(a) \Vert^2 \thinspace Q(s, a) \right]}{\mathbb{E}_a\left[ \Vert g(a) \Vert^2 \right]} \qquad (23)
$$

**最优 baseline = 以 score 范数平方为权的 Q 加权平均**。当 $\Vert g \Vert^2$ 与 $Q$ 近似无关（各动作 score 幅度相近）时， $`b^{\ast}(s) \approx \mathbb{E}_a[Q(s, a)] = V^{\pi}(s)`$ ——价值函数是近似最优 baseline，这是 8.2 专门学一个 $V$ 网络的动机源头；当 score 幅度在动作间不均时 $b^{\ast} \neq V$ 是系统性差异而非噪声（下面算例）。最优 baseline 的方差最小化推导以 Greensmith et al. 2004 为标准出处（[Greensmith et al., 2004](../references.md)，JMLR 5:1471-1530，无 arXiv，[B]）；Sutton-Barto §13.4 给出无偏性与定性讨论。

**算例**（数值构造，本书）：单状态、三动作， $\pi = (0.5, 0.25, 0.25)$ ， $Q = (3, 1, 0)$ ，取标量 score $g = (1, 0, -2)$ ——满足 $\sum_i \pi_i g_i = 0.5 - 0 - 0.5 = 0$ ，即式 (8) 的离散形态。 $V^{\pi} = 0.5 \times 3 + 0.25 \times 1 = 1.75$ 。逐个 baseline 算估计器 $X = g_i (Q_i - b)$ 的方差（ $\mathbb{E}[X] = \mathbb{E}[gQ] = 1.5$ ，与 $b$ 无关，可作自检）：

- $b = 0$ ： $\mathbb{E}[X^2] = 0.5 \times 9 + 0 + 0 = 4.5$ ， $\mathrm{Var} = 4.5 - 1.5^2 = 2.25$ ；
- $b^{\ast} = \mathbb{E}[g^2 Q] / \mathbb{E}[g^2] = (0.5 \times 1 \times 3 + 0 + 0) / (0.5 + 0 + 0.25 \times 4) = 1.5 / 1.5 = 1$ ： $\mathbb{E}[X^2] = 0.5 \times 4 + 0 + 0.25 \times 4 \times 1 = 3$ ， $\mathrm{Var} = 3 - 2.25 = 0.75$ ；
- $b = V^{\pi} = 1.75$ ： $\mathbb{E}[X^2] = 0.5 \times 1.25^2 + 0 + 0.25 \times 4 \times 1.75^2 \approx 3.84$ ， $\mathrm{Var} \approx 3.84 - 2.25 = 1.59$ 。

$b^{\ast}$ 把方差从 2.25 压到 0.75（3 倍）， $V$ 只压到 1.59——此例中 score 幅度最大的动作（ $|g| = 2$ 的 $a_3$ ）恰好 $Q$ 最低，范数加权与等权平均发生分歧。工程上没人逐状态解式 (23)：要么用常数 baseline（回报的移动平均），要么学 $V$ 作近似——后者就是 actor-critic（8.2）。

### 5.3 baseline 谱系：从常数到价值网络

式 (21) 留下的设计空间，按「状态依赖的粒度」排队：

- **常数 baseline**： $b$ = batch 均值回报。§7 的方差分解里它恰好杀掉 $\mu_R^{2}$ 项；零学习成本，每步白拿。
- **历史依赖 baseline**： $b$ 可依赖直到 $s_t$ 为止的全部历史（练习 2 证其合法性）——§7.2 会看到，这允许 per-token 的 $b(s_t)$ ，是 LLM 退化结构下 credit assignment 的唯一入口。
- **价值网络 baseline**： $b(s) = V_\varphi(s)$ ，用回归学出的前缀条件期望——8.2 的全部内容。
- **同 prompt 组均值**：同一 prompt 下 $G$ 个采样回报的均值作 $b$ ——8.4 GRPO 的选择，免学价值网络。

排序背后的账：baseline 越贴近 $\mathbb{E}[R \mid \text{可获得信息}]$ ，残差方差越小；但越复杂的 baseline 自己要付学习成本与偏差风险。§7 的玩具模型里 baseline 的收益由 $\mu_R^{2} / \sigma_R^{2}$ 的比值决定——奖励越「确定可分」（ $p$ 远离 0.5），常数 baseline 吃掉越多；残差方差大时（难任务、训练早期），学习型 baseline 的增量收益才显著。

## 6. REINFORCE：从随机单元到轨迹梯度

REINFORCE 的原始形态来自 Williams 1992（[Williams, 1992](../references.md)，Machine Learning 8:229-256，无 arXiv，[B]）：网络含随机输出单元，单元 $i$ 依概率 $g_i(\xi_i, w)$ 采样输出（ $\xi_i$ 为注入的随机性），环境回以标量强化信号 $r$ ，按

$$
\Delta w_{ij} = \alpha_{ij} \thinspace (r - b_{ij}) \thinspace e_{ij}, \qquad e_{ij} = \frac{\partial \ln g_i}{\partial w_{ij}} \qquad (24)
$$

更新。**符号**： $\alpha_{ij}$ ：学习率； $b_{ij}$ ：不依赖即时输出的 baseline； $e_{ij}$ ：characteristic eligibility（资格），即该参数对输出对数概率的导数。论文的核心结果是 $\mathbb{E}[\Delta w]$ 与 $\nabla_w \mathbb{E}[r]$ 同向（baseline 不依赖单元即时输出时无偏）——式 (20)(21) 的原始版本。把「随机单元」换成整条轨迹、把 $r$ 换成 reward-to-go 减 baseline，就是现代教科书形式（Sutton-Barto §13.3-13.4 的 REINFORCE 与 REINFORCE with baseline）：

$$
\hat{g} = \frac{1}{N_{\mathrm{traj}}} \sum_{i=1}^{N_{\mathrm{traj}}} \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \thinspace \left( G_t^{(i)} - b \right), \qquad \theta \leftarrow \theta + \eta \thinspace \hat{g} \qquad (25)
$$

**符号**： $N_{\mathrm{traj}}$ ：批内轨迹数（本书约定 $N$ 专归参数量，轨迹数不占这个字母）； $T_i$ ：第 $i$ 条轨迹长度； $\hat{g}$ ：式 (11) 的 Monte Carlo 样本均值，由大数律随 $N_{\mathrm{traj}}$ 收敛到真梯度，方差按 $1/N_{\mathrm{traj}}$ 缩（§7）。算法框：

```
loop:
  用当前 π_θ 采 N_traj 条轨迹             ← on-policy：期望号下是当前策略
  每条轨迹逐步算 G_t（γ=1 时为剩余奖励和；LLM 末端奖励下恒为 R）
  按式 (25) 拼梯度估计，交给 optimizer（AdamW，2.2）
```

实现注记：与监督训练共用一套 autograd 栈的写法是构造**伪损失**

$$
L(\theta) = -\frac{1}{N_{\mathrm{traj}}} \sum_{i=1}^{N_{\mathrm{traj}}} \sum_{t=0}^{T_i-1} \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \thinspace \hat{A}_t^{(i)}, \qquad \hat{A}_t^{(i)} = G_t^{(i)} - b \thinspace \text{（detach，常数化）} \qquad (26)
$$

$\nabla L = -\hat{g}$ ： $\log \pi$ 的梯度由 autograd 产出（1.1 的 vjp 机制）， $\hat{A}$ 只作逐样本权重。「把 credit 当权重、把 log 概率当损失」是全部 RL 训练栈的共用模式，8.2-8.4 的目标函数都写成这个形态。

一个退化情形把「信号从哪来」钉死：若所有采样的回报恒为常数 $R \equiv c$ ，式 (25) 的期望为 $c \sum_t \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)] = 0$（式 (8) 逐分量作用）——没有任何学习信号。驱动策略移动的不是 reward 的绝对水平，而是 reward 在采样分布下的**变差**；这也是 baseline 不动期望的镜像读法——减掉常数不动信号，只动噪声。

教学代码（教学用，非优化实现；省 padding / EOS mask、并行采样与 KL 正则，只保留式 (26) 的语义骨架）：

```python
# 教学用，非优化实现：REINFORCE（γ=1，常数 baseline = batch 均值回报）
# model: 小 LM，forward(ids) -> logits [B, T, V]；reward_fn(prompt, ids) -> 标量黑盒
def reinforce_step(model, opt, prompts, reward_fn, max_len=64):
    ids = sample_batch(model, prompts, max_len)          # 从当前 π_θ 自回归采样，不回传梯度
    logits = model(ids[:, :-1])                          # [B, T, V]，一次前向取全部 logit
    logp = torch.log_softmax(logits, dim=-1)
    logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # 逐 token log π(a_t|s_t)
    R = torch.tensor([reward_fn(p, y) for p, y in zip(prompts, ids)], dtype=logp.dtype)
    A = (R - R.mean()).unsqueeze(-1)                     # 常数 baseline：式 (21) 的最简特例
    loss = -(logp * A).sum(-1).mean()                    # 式 (26)：sequence 内求和、batch 内平均
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), R.mean().item()
```

它与生产实现的差距逐条对应后文：采一批用一次（样本效率，§8）；baseline 只是 batch 均值而非前缀条件的 $V$（8.2）；对单步更新幅度没有任何信任域约束（8.3 的 clip）；采样引擎与训练引擎未拆分（8.5）。

## 7. 方差分析：长序列、退化奖励与粒度

### 7.1 单样本噪声地板：随 $T$ 线性增长

REINFORCE 的方差从哪来、被什么压低，用一个显式假设的玩具模型算成账（本书分析；假设逐条列出，不假装它预测真实方差）。设：

- on-policy 采样 ⇒ 每个 token 的 score 零均值（式 (8)），且跨步不相关： $\mathbb{E}[g_t] = 0$ ， $\mathbb{E}[g_t g_{t'}^{\top}] = \sigma^{2} I \cdot [t = t']$ ；
- 末端奖励 $R$ 与 score 的噪声分量独立， $\mathbb{E}[R] = \mu_R$ ， $\mathrm{Var}(R) = \sigma_R^{2}$ 。

取式 (9) 的单样本估计器 $\hat{g} = R \sum_{t=1}^{T} g_t$ 。第二条假设抹掉了信号（真实梯度来自 $R$ 与 $g$ 的相关性），留下的是纯噪声地板：

$$
\mathbb{E}[\hat{g} \thinspace \hat{g}^{\top}] = \mathbb{E}[R^{2}] \thinspace \mathbb{E}\left[ \Big( \sum_t g_t \Big) \Big( \sum_t g_t \Big)^{\top} \right] = (\sigma_R^{2} + \mu_R^{2}) \thinspace T \thinspace \sigma^{2} I
\quad \Longrightarrow \quad
\mathrm{Var}(\hat{g}) = T \thinspace \sigma^{2} \thinspace (\sigma_R^{2} + \mu_R^{2}) \qquad (27)
$$

两个读法。其一，**噪声地板随序列长度线性涨**： $T$ 个零均值 score 各携一份方差，共享同一个 $R$ 把它们同相放大。长链推理（ $T$ 上千）的 REINFORCE 单样本估计天然更吵，这是结构性结论而非实现问题。其二， $\mu_R^{2}$ 与 $\sigma_R^{2}$ 分家：常数 baseline $b = \mu_R$ 恰好杀掉 $\mu_R^{2}$ 项，使 $\mathrm{Var} = T \sigma^{2} \sigma_R^{2}$ 。代入二值奖励的数字（答对率 $p = 0.9$ ： $\mu_R^{2} = 0.81$ 、 $\sigma_R^{2} = 0.09$ ）：加 baseline 前后噪声系数 0.90 → 0.09，一个数量级。剩下的 $\sigma_R^{2}$ 项只有两个去处——状态依赖 baseline（把「从 $s_t$ 就能预言的那部分 $R$ 」减掉，即 8.2 的 $V(s_t)$ ）与加大 batch（ $N_{\mathrm{traj}}$ 条独立轨迹平均，方差整除 $N_{\mathrm{traj}}$ ）。

### 7.2 末端奖励的退化：reward-to-go 是空操作

LLM 的奖励结构（ $r_t = 0$ 对 $t \lt T-1$ ）下，reward-to-go 退化为常数：

$$
G_t = r_t + r_{t+1} + \cdots + r_{T-1} = R(x, y) \qquad \text{对所有 } t \text{ 成立} \qquad (28)
$$

§4.1 的因果引理在这个场景是空操作——没有「过去的奖励」可供剔除。两个直接推论：

1. **token-level 与 sequence-level 估计器逐字相同**。由 §2.2 的链式分解， $\sum_t \nabla \log \pi_\theta(y_t|s_t) \cdot R = \nabla \log \pi_\theta(y|x) \cdot R$ ——把整个回答当一个动作的 bandit REINFORCE 与逐 token 的全回报 REINFORCE 是同一个估计器。
2. **per-token credit 的唯一工具是 per-token baseline**。把每个 token 的权重从 $(R - b)$ 换成 $(R - b(s_t))$ ，式 (27) 中的 $\mu_R$ 就被换成「给定前缀 $s_t$ 后 $R$ 的残差」——前缀对结果的解释力越强，残差方差越小。常数 baseline（免费）→ 前缀条件 baseline $V(s_t)$ （要学一个网络，8.2）→ 同 prompt 组统计 baseline（8.4 的 GRPO），LLM-RL 的方差缩减沿这个杠杆排序演进。

per-token baseline 的收益有一个干净的量化（本书分析）。取最优的前缀条件 baseline $b(s_t) = \mathbb{E}[R \mid s_t]$ ，每个 score 项的权重 $w_t = R - \mathbb{E}[R \mid s_t]$ 满足

$$
\mathbb{E}[w_t^{2}] = \mathbb{E}\left[ \mathrm{Var}(R \mid s_t) \right] = \mathrm{Var}(R) - \mathrm{Var}\left( \mathbb{E}[R \mid s_t] \right) \qquad (29)
$$

**符号**：末步是全方差公式（law of total variance）。读法：per-token baseline 杀掉的恰是「 $R$ 中被前缀解释掉的方差份额」——某 prompt 下「开头两个 token 就基本注定结局」时 $\mathrm{Var}(\mathbb{E}[R|s_t])$ 大、收益大；结果完全由末端偶然决定时收益趋零。把式 (29) 代回式 (27) 的结构：噪声地板从 $T \sigma^{2} (\sigma_R^{2} + \mu_R^{2})$ 降到 $T \sigma^{2} \thinspace \mathbb{E}[\mathrm{Var}(R|s_t)]$ 量级（忽略跨步相关）——这就是 8.2 值得为 $V$ 单独养一个网络的定量理由。

### 7.3 batch 与采样成本的账

$N_{\mathrm{traj}}$ 条独立轨迹的样本均值把式 (27) 整除 $N_{\mathrm{traj}}$ ；要把单步梯度噪声压到固定水平，所需 $N_{\mathrm{traj}}$ 随 $T$ 线性涨。另一边，LLM rollout 的成本也是每条约 $T$ 次解码步（自回归串行）。噪声地板与采样成本两头都挂 $T$ ——长链推理 RL 的吞吐账由此而来；系统侧（rollout 引擎、权重同步）归 8.5。

## 8. 样本效率与 on-policy 约束

式 (25) 的期望号下是当前策略 $\pi_\theta$ ——按 3.2 §5 的分界定义，REINFORCE 是 on-policy：参数每更新一次，旧样本的分布假设即失效。三个直接后果：

1. **数据一次性**：每个梯度步消耗的轨迹必须采自当前（或近似当前）策略，采一批只能用一次；对照离线目标（SFT / DPO）一份静态数据反复过 epoch，样本成本高出一个量级维度。
2. **rollout 常驻训练循环**：每个 step 之前先付 $N_{\mathrm{traj}} \times T$ 次解码，训练循环里必须养一个采样引擎，这是 LLM-RL 系统形态与纯训练栈分家的根源（8.5）。量级账可以先记一笔（口径 recap：models 1.2 的每 token 算力账——前向约 $2N$ FLOPs、训练前向加反向约 $6N$ FLOPs， $N$ 为参数量）： $N_{\mathrm{traj}}$ 条长度 $T$ 的 rollout 约付 $2N \cdot N_{\mathrm{traj}} \thinspace T$ FLOPs 的前向（decode 还有 KV cache 与引擎常数，系统侧归 8.5），紧随的一次训练前向反向约 $6N \cdot N_{\mathrm{traj}} \thinspace T$ ——rollout 与训练的 FLOPs 比约 1:3，采样不是免费的副产品，而是与训练同量级的常驻开销；梯度方差要求的 $N_{\mathrm{traj}}$ 越大（§7.3），这个比值越向 rollout 倾斜。
3. **复用的偏差与校正**：拿旧策略 $\pi_{\theta_{\mathrm{old}}}$ 的样本直接更新，期望号下分布错误、估计有偏；校正工具是重要性比 $\pi_\theta / \pi_{\theta_{\mathrm{old}}}$ ，配套的是限制单步移动幅度的信任域思路——这两条正是 8.3 PPO 的出发点。

RLHF 语境一句：把这套梯度估计机器嵌进后训练完整链路（人类偏好数据 → reward model → 策略优化）的流程来自 Christiano et al. 2017（[Christiano et al., 2017](../references.md)，arXiv:1706.03741，[B]），工程链归 9.2；本章交付的是它的梯度内核。

## 总结

| 主题 | 公式 | 要点 |
| --- | --- | --- |
| LLM 的 MDP 映射 | §2.2 表与状态链图 | $P$ 确定、 $r$ 末端、 $\gamma = 1$；全部随机性在策略；sequence-level ≡ token-level（链式分解） |
| log-derivative trick | 式 (5)(7) | $\nabla p = p \nabla \log p$；环境与初始分布被 $\nabla_\theta$ 清零——model-free 的来源 |
| score 零均值 | 式 (8) | baseline 合法性与方差分解的共同前提 |
| 因果引理 | 式 (10)(11) | 动作不影响过去奖励；过去项只贡献方差 |
| PG 定理（episodic） | 式 (13) | $`\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_t \thinspace Q^{\pi}_t]`$ [B]（Sutton et al. 1999） |
| PG 定理（discounted） | 式 (17)-(19) | 递归展开 ⇒ $d^{\gamma}$ 加权；无比例常数；教材比例形式吸收 $1/(1-\gamma)$ [B]（Sutton-Barto §13.2） |
| baseline 无偏 | 式 (20)(21) | 减任意 $b(s)$ 不动期望，只动方差 |
| 最优 baseline | 式 (23) | score 范数加权的 Q 平均； $V^{\pi}$ 是等权特例 ⇒ 8.2 学 $V$ 的动机 |
| REINFORCE | 式 (24)(25)(26) | Williams 1992 原始形式 [B]；伪损失 = credit 加权 log π |
| 方差 | 式 (27)(28) | 噪声地板 $T \sigma^{2}(\sigma_R^{2} + \mu_R^{2})$；末端奖励下 reward-to-go 空转，baseline 是唯一杠杆 |
| 样本效率 | §8 | on-policy ⇒ 数据一次性 + rollout 常驻 ⇒ 8.3 重要性比与 clip 的动机 |

## 练习

1.（推导）补全式 (15)→(17) 的不动点展开：把式 (16) 代入式 (15) 后对 $\nabla V^{\pi}$ 递归迭代两次，归纳出 $\sum_k \gamma^{k} \Pr(s \to x, k, \pi)$ 的一般项；说明 $\gamma \lt 1$ 与有限视界两种情形下级数收敛的理由。

2.（推导）baseline 引理的推广与边界：(a) 证明 $b$ 可依赖 $(s_0, a_0, \ldots, s_t)$ （直到 $s_t$ 为止的历史）仍有 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \thinspace b] = 0$ ；(b) 取 §5.2 算例的三动作设置，令 $b$ 依赖动作、取 $b(s, a) = Q(s, a)$ ，验证估计器恒为零向量而 $\mathbb{E}[gQ] \neq 0$ ——动作依赖的 baseline 把信号一起减掉了。

3.（计算）§5.2 算例改参数： $\pi = (0.6, 0.2, 0.2)$ 、 $Q = (3, 1, 0)$ 、 $g = (1, -1, -2)$ 。(a) 验证 $\sum_i \pi_i g_i = 0$ ；(b) 算 $V^{\pi}$ 、 $b^{\ast}$ 与三个 baseline（ $0$ 、 $b^{\ast}$ 、 $V^{\pi}$ ）下的估计器方差；(c) 若 score 改为 $g = (1, 1, -4)$ （先验证零均值条件）， $b^{\ast}$ 与 $V^{\pi}$ 的差距拉大还是缩小？解释机制。

4.（代码）在 §6 教学代码上把常数 baseline 换成指数移动平均 baseline（ $b \leftarrow 0.9 b + 0.1 \bar{R}$ ），构造合成二值奖励（回答含特定子串记 1 否则 0，可调难度 $p$ ），对单步梯度估计 $\hat{g}$ 做 200 次独立重复、比较两种 baseline 的样本方差；把结果对照式 (27) 的 $\mu_R^{2} / \sigma_R^{2}$ 分解，讨论 $p$ 接近 0.5 与接近 1 时 baseline 收益的变化方向。

## 参考

- [Williams, 1992](../references.md)：Simple statistical gradient-following algorithms for connectionist reinforcement learning，Machine Learning 8:229-256（无 arXiv），[B]（key: williams1992）——REINFORCE 原始出处，式 (24) 与 baseline 无偏性的原始证明
- [Sutton et al., 1999](../references.md)：Policy Gradient Methods for Reinforcement Learning with Function Approximation，NeurIPS 1999（无 arXiv），[B]（key: sutton1999）——函数逼近设定下的策略梯度定理（式 (13)）与 baseline 合法性
- [Sutton & Barto, 2018](../references.md)：Reinforcement Learning: An Introduction（第 2 版），MIT Press，[B]（key: sutton-barto）——§13.2 状态分布形式的推导路线（式 (15)-(19)）、§13.3-13.4 REINFORCE 与 baseline 无偏性（式 (20)(21) 口径）
- [Greensmith et al., 2004](../references.md)：Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning，JMLR 5:1471-1530（无 arXiv），[B]（key: greensmith2004）——式 (23) 最优 baseline 方差最小化的标准出处
- [Christiano et al., 2017](../references.md)：Deep Reinforcement Learning from Human Preferences，arXiv:1706.03741，[B]（key: christiano）——RLHF 链路口径（偏好 → RM → 策略优化），工程展开归 9.2
