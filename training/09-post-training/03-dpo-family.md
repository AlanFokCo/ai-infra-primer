# 9.3 DPO 家族：直接偏好优化（The DPO Family: Direct Preference Optimization）

> 本章讲一族「不训显式 reward model、直接在偏好数据上优化策略」的方法。主线是 DPO 的完整推导链：KL 约束 RLHF 目标有闭式最优解；把最优解反解成 reward 的表达式，代入 Bradley-Terry 模型后配分函数 $Z(x)$ 自行消去——这一步正是 3.2 §3 平移不可识别伏笔的兑现；由此得到直接优化策略的 DPO 目标。随后分析 DPO 与 PPO 何时等价、何时偏离，再逐家拆四个修正：IPO（ ΨPO 框架与过拟合分析）、KTO（Kahneman-Tversky 价值函数、单边数据）、SimPO（reference-free 平均 log prob 隐式 reward + margin）、ORPO（SFT + odds ratio 单阶段），最后给家族对照总表与 TRL 实现走读 | 前置：3.2（偏好接口、Bradley-Terry 模型、平移不可识别性、在线/离线分界定义）、9.2（RM 训练与 PPO 管线）、8.3（PPO 目标与 clip）、3.1（CE 的训练侧口径） | 锚点：DPO（[Rafailov et al., 2023](../references.md)，arXiv:2305.18290，[A]，论文 + eric-mitchell/direct-preference-optimization 官方 repo）、IPO（[Azar et al., 2023](../references.md)，arXiv:2310.12036，[B]）、KTO（[Ethayarajh et al., 2024](../references.md)，arXiv:2402.01306，[A]）、SimPO（[Meng et al., 2024](../references.md)，arXiv:2405.14734，[A]）、ORPO（[Hong et al., 2024](../references.md)，arXiv:2403.07691，[A]）、Bradley-Terry 1952 [B]、huggingface/trl（教学参照 + 官方实现，main 分支 2026-11-18 核实）

## 1. 问题：能不能把 RLHF 的两个模型合成一个

先回答一个更基本的问题：「训 RM + 用 RL 优化 RM」这两步，能不能折叠成一个直接作用在策略上的监督式目标？如果最终要的只是「策略对偏好的拟合」，折叠的动机就写在 9.2 的 RLHF 管线的三笔明确成本里。第一，训一个独立的 reward model，多一份训练循环、多一份显存。第二，跑 PPO，需要 rollout 引擎、四个模型同时驻留，账归 8.5。第三，两个模型各自引入误差：RM 有分布外校准误差，RL 有采样方差。DPO 的出发问题正是这个折叠是否可行。

答案是肯定的，而且折叠的每一步都是严格的代数恒等变形，没有任何一步是近似。链条全貌（本章 §2-§4 的顺序）：

```
KL 约束 RLHF 目标（式 1）
  max_π E_{x~D, y~π}[r(x,y)] − β·KL(π‖π_ref)
        │  §2：逐 prompt 变分，改写为 KL(π‖π*) − log Z(x)，Gibbs 不等式取等
        ▼
闭式最优解（式 3）
  π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)
        │  §3：两边取 log 解出 r
        ▼
reward 重参数化（式 4）
  r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
        │  代入 Bradley-Terry 偏好模型：β·log Z(x) 逐 prompt 相消
        ▼
偏好概率只含策略（式 5）
        │  对参数化策略 π_θ 做极大似然
        ▼
DPO 目标（式 6）
  L = −E_D log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)))
```

走完这条链，RM 与 RL 同时消失：偏好数据直接约束策略，训练循环退化为一个带冻结参考模型的监督式分类损失。下面几节逐环推一遍这条链，§5 讨论它在哪些环节会偏离理想情形，§6-§9 讲四个后继工作各自修哪一环。

本章口径一句话：超参与实证数字一律随各家论文原文引用，不做跨论文换算。尤其各家的 β 同名异物、量纲不同——DPO 与 SimPO 的 β 差一个数量级——跨方法搬数值是无效操作，换算理由在 SimPO 节给出。

## 2. KL 约束目标的闭式最优解

这个目标的最优策略有闭式解，而且两步就能推出来：先把目标逐 prompt 改写成「KL 散度减一个常数」，再用 Gibbs 不等式直接读出取等条件——关键一步是补进一个只依赖 prompt 的配分函数。起点是 RLHF 的策略优化目标（DPO 论文式 (3)，与 9.2 的 PPO 阶段同一式；[Rafailov et al., 2023](../references.md)，[A]）：

$$
J(\pi) = \mathbb{E}_{x \sim D,\hspace{4mu} y \sim \pi(\cdot \mid x)}\left[ r(x, y) \right] - \beta \thinspace \mathbb{E}_{x \sim D}\left[ D_{\mathrm{KL}}\left( \pi(\cdot \mid x) \Vert \pi_{\mathrm{ref}}(\cdot \mid x) \right) \right] \qquad (1)
$$

**符号**： $x$：prompt，取自静态分布 $D$； $y$：完整回答（序列级动作，8.1 的口径）； $r(x, y)$：reward（先假设已知，§3 把它消掉）； $\pi_{\mathrm{ref}}$：冻结的参考策略（通常取 SFT 模型）； $\beta > 0$：KL 正则强度； $\pi$：被优化的策略，此步不对它做任何参数化假设（非参数策略类，逐 prompt 独立）。

这里有个关键的结构性观察：目标里没有跨 prompt 的耦合。 $J(\pi)$ 是逐 prompt 项的期望， $\pi(\cdot \mid x)$ 对每个 $x$ 都是独立的概率分布，所以可以固定 $x$、逐 prompt 求极值，最后把结果对 $x$ 平均回去。固定 $x$，展开 KL，再除以 $-\beta$（把 max 变成 min）：

$$
\begin{aligned}
\min_{\pi(\cdot \mid x)}\hspace{4mu} & \mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[ \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) \right] \\
=\hspace{4mu} & \min_{\pi(\cdot \mid x)}\hspace{4mu} \mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[ \log \frac{\pi(y \mid x)}{\dfrac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)} - \log Z(x) \right]
\end{aligned} \qquad (2)
$$

**符号**：第一行是定义展开（ $`D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}}) = \mathbb{E}_{y \sim \pi}[\log(\pi / \pi_{\mathrm{ref}})]`$ ）；第二行引入配分函数

$$
Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

这一步把分母补成了一个合法分布： $\dfrac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp(r(x, y) / \beta)$ 非负、对 $y$ 求和为 1（ $Z(x)$ 正是归一化常数）。补它要付出 $\log Z(x)$ 一项；但 $Z(x)$ 只依赖 $x$、 $\pi_{\mathrm{ref}}$ 与 $r$，不含 $\pi$，对变分问题来说它是常数。于是第二行括号里的第一项是一个标准 KL 散度：

$$
\min_{\pi(\cdot \mid x)}\hspace{4mu} \left[ D_{\mathrm{KL}}\left( \pi(\cdot \mid x) \Vert \pi^{\ast}(\cdot \mid x) \right) - \log Z(x) \right], \qquad \pi^{\ast}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \qquad (3)
$$

**命题**（闭式最优解）：固定 x 后，上述变分问题的最小值在 $\pi(\cdot \mid x) = \pi^{\ast}(\cdot \mid x)$ 处取得；逐 prompt 成立即全局成立。**证明**：括号里的 log Z(x) 项不含 π，对变分问题是常数；剩下的 KL 散度非负，当且仅当两分布相等时取零（Gibbs 不等式）——所以最小值恰在两分布相等处取得。∎ 这就是闭式最优解（DPO 论文式 (4) 与附录 A.1 的完整链）。

代一组全数字走一遍闭式解的形状：设某 prompt 下只有两条候选回答，π_ref 给它们各 1/2 的概率，r/β 分别为 ln 3 与 0。未归一权重为 1/2 × 3 = 3/2 与 1/2 × 1 = 1/2，配分函数 Z(x) = 2，最优策略把概率改成 3/4 与 1/4——reward 高出 β·ln 3 的那条，概率从 1/2 抬到 3/4。

这个结果有两个读法。其一， $\beta$ 的语义就此固定： $\beta \to 0$ 时 $\pi^{\ast}$ 退化为在 $\pi_{\mathrm{ref}}$ 支撑集上的 $\arg\max_y r$（全部质量压到 reward 最高点）； $\beta \to \infty$ 时 $\exp(r / \beta) \to 1$、 $\pi^{\ast} \to \pi_{\mathrm{ref}}$。其二，最优值本身可以算出来：代回式 (3)，逐 prompt 的最优目标值是 $\beta \log Z(x)$。所谓指数倾斜（exponential tilting），是指给参考分布逐点乘上 reward 的指数因子再归一化；它对原目标的增益，恰好是倾斜分布的对数配分函数乘 $\beta$。这个量在 §3 会以「平移自由度」的身份重新出现。

这个闭式解不是 DPO 的发明——它是 KL 正则化强化学习目标的标准最优形式（DPO 论文引用 Peng et al. 2019 与 Peters & Schaal 2007 等先例）。DPO 论文的贡献在下一步：把这个「含未知 $r$ 与不可算 $Z$」的解反解成 $r$ 的表达式，再把它消灭掉。

## 3. 重参数化：配分函数在 Bradley-Terry 里自行消去

下一步只做一件事：从闭式解里把 reward 反解出来，再看着配分函数在偏好差分里自行消去。式 (3) 两边取 log、移项：

$$
r(x, y) = \beta \log \frac{\pi^{\ast}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x) \qquad (4)
$$

**符号**： $\pi^{\ast}$：reward $r$ 对应的 KL 约束最优策略； $Z(x)$：式 (3) 的配分函数。式 (4) 说：真实 reward 可以分解为「最优策略相对参考策略的对数比」加一个**只依赖 $x$** 的项。

沿用上文的全数字例核账：两条回答的对数比项分别为 ln(3/4 ÷ 1/2) = ln 3 − ln 2 与 ln(1/4 ÷ 1/2) = −ln 2，各加上 β·log Z(x) = β·ln 2 后，恰好还原出 r 的两条值 β·ln 3 与 0——只依赖 prompt 的那一项就此显形。

现在把式 (4) 代入 3.2 式 (3) 的 Bradley-Terry 偏好模型 $p^{\ast}(y_1 \succ y_2 \mid x) = \sigma(r^{\ast}(x, y_1) - r^{\ast}(x, y_2))$。偏好概率只吃 reward 差：

$$
r(x, y_1) - r(x, y_2) = \beta \log \frac{\pi^{\ast}(y_1 \mid x)}{\pi_{\mathrm{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^{\ast}(y_2 \mid x)}{\pi_{\mathrm{ref}}(y_2 \mid x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{=\thinspace 0}
$$

$Z(x)$ 项逐字相消，得（DPO 论文式 (6)）：

$$
p^{\ast}(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left( \beta \log \dfrac{\pi^{\ast}(y_2 \mid x)}{\pi_{\mathrm{ref}}(y_2 \mid x)} - \beta \log \dfrac{\pi^{\ast}(y_1 \mid x)}{\pi_{\mathrm{ref}}(y_1 \mid x)} \right)} \qquad (5)
$$

这一步相消不是巧合，而是 3.2 §3 平移不可识别性的闭环兑现。Bradley-Terry 模型只识别 reward 的等价类 $[r] = \lbrace r + c(x) \rbrace$（ $c$ 是任意逐 prompt 函数），而 $\beta \log Z(x)$ 恰是一个逐 prompt 函数，落在不可识别的 gauge 自由度里（口径：gauge 自由度 = 不改变偏好概率的逐 prompt 加性项），代入差分后必然消失。正例：给同一 prompt 下所有回答的 reward 同加一个常数，偏好概率不变，加前加后同属一个等价类。边界反例：若加的「常数」随回答不同而不同（c 依赖 y 而不只依赖 x），偏好概率就会改变——它不在等价类里，而是换了一个 reward。换个方向说：DPO 能用「策略对数比」这个不含 $Z$ 的量完整表达偏好模型，恰恰因为偏好数据本来就看不见 $Z$。

> **提点**：这也是 DPO 全文标题的读法：语言模型（的对数比）就是一个隐式 reward model——隐式到连配分函数都不用估。

还剩一个合法性问题：重参数化 $r_\theta(x, y) = \beta \log(\pi_\theta(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))$ 会不会限制可表达的 reward 类？**命题**（表达力无损；DPO 论文 Theorem 1）：不会——每个等价类都有一个代表元具有重参数化形式。**证明**（构造投影）：对任意 reward $r$，投影 $f(r) = r - \beta \log Z(x)$（减去它自己的配分函数项）与 $r$ 同属一个等价类，而 $f(r)$ 恰能写成 $\beta \log(\pi_r / \pi_{\mathrm{ref}})$ 的形式——投影不改变偏好，形式却已合规。∎ 再配合 Lemma 1（同类 reward 诱导同一偏好分布）与 Lemma 2（同类 reward 在 KL 约束目标下诱导同一最优策略），「优化隐式 reward」与「优化真 reward」在目标层面是同一个问题。

最后一步，对参数化策略做极大似然。式 (5) 把偏好概率写成了最优策略的函数；把 $\pi^{\ast}$ 换成待训的 $\pi_\theta$，对静态偏好数据集 $D = \lbrace (x^{(i)}, y_w^{(i)}, y_l^{(i)}) \rbrace_{i=1}^{N}$ 最大化偏好似然，即得 DPO 目标（DPO 论文式 (7)）：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\hspace{4mu} \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right] \qquad (6)
$$

**符号**： $y_w$ / $y_l$：偏好对中的 chosen / rejected 回答； $\sigma$：logistic sigmoid； $\pi_{\mathrm{ref}}$ 冻结、不回传梯度。目标里出现的隐式 reward 记作 $`\hat{r}_\theta(x, y) = \beta \log(\pi_\theta(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))`$。整条链没有一步近似：闭式解是恒等变形，取对数是恒等变形，代入 BT 时 $Z$ 相消也是恒等变形，只有最后的 MLE 是统计原则。近似的账全在下一层：数据是不是静态的、BT 假设成不成立，见 §5。

代一组全数字单样本把损失算到底：取 β = 1，设 chosen 支的策略/参考概率比为 4/5、rejected 支为 1/2。sigmoid 输入 = ln(4/5) − ln(1/2) = ln(8/5)，σ 值恰为 8/13，单样本损失 = ln(13/8) nat——两支的对数比同号相消后只剩 ln(8/5) 的 margin，损失自然比随机水平（ln 2）低。

流程口径（DPO 论文 §4 的 outline）分两步。第一步采数据：对每个 prompt 采 $y_1, y_2 \sim \pi_{\mathrm{ref}}(\cdot \mid x)$，人工标注成对偏好，构成 $D$。第二步优化：对给定的 $\pi_{\mathrm{ref}}$ 与 $\beta$ 最小化式 (6)。参考模型的取法有讲究：偏好数据若来自 SFT 模型，就取 $\pi_{\mathrm{ref}} = \pi_{\mathrm{SFT}}$；如果没有现成的 SFT 模型，论文建议先在 chosen 回答上做最大似然得到一个 $\pi_{\mathrm{ref}}$，缓解「真参考分布不可得」带来的分布偏移。

## 4. DPO 梯度的结构：一个自动加权项

DPO 的梯度靠什么区分「排错的样本」与「排对的样本」？答案是一个自动加权项——本章后四节的所有修正都从它出发。式 (6) 对 $\theta$ 的梯度有显式结构（DPO 论文式 (8)）。令 $u = \beta(h_w - h_l)$， $h_y = \log \pi_\theta(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x)$，则 $`\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}[\sigma(-u) \cdot \nabla_\theta u]`$，展开：

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \thinspace \mathbb{E}_{(x, y_w, y_l) \sim D}\left[ \underbrace{\sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right)}_{\text{排序越错权重越大}} \cdot \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right] \qquad (7)
$$

**符号**：括号里的两项是对称的推力：第一项升 chosen 的似然，第二项降 rejected 的似然。外面的 sigmoid 因子是逐样本权重：隐式 reward 把 rejected 排得比 chosen 越高（即排序越错），该样本权重越大、趋近 1；排序已正确的样本权重趋近 0。全数字例：排序恰好打平时权重 = 1/2；chosen 的隐式 reward 已高出 ln 9（排序正确）时权重降到 1/10，反过来落后 ln 9（排序错误）时权重升到 9/10——权重随错误程度单调上升。这个权重项有实证地位：DPO 论文报告，去掉它（朴素地对两支各做符号相反的似然梯度）会使模型退化（其附录 Table 3）。

> **提点**：加权不是装饰，是目标结构的承重部分。

式 (7) 同时暴露了 DPO 的两个结构性偏置，后面四家修正都从这里出发：

- **无界性**： $-\log \sigma(u)$ 关于 $u$ 严格递减、以 0 为渐近线——损失永远奖励更大的隐式 reward 差，不存在「拟合完成」的驻点；确定性偏好（同一对 $(y_w, y_l)$ 永远同向标注）下，总体目标的最优解是 margin 推到无穷。IPO 的 §4.2 诊断（§6）正是对这一条的展开。
- **长度耦合**： $\log \pi_\theta(y \mid x)$ 是逐 token log 概率的**求和**（TRL 与官方 repo 的实现口径都是 `sum(dim=1)`，见 §10），长回答的总 logp 天然更低——隐式 reward 随长度单调下偏。SimPO 的平均 logp 修正在 §8。

## 5. DPO 与 PPO：等价的边界

DPO 与 PPO 在什么条件下是同一个解、又在哪些环节分道扬镳？先立可判定的部分。**命题**（目标层等价）：当 Bradley-Terry 模型精确成立、RM 是偏好数据的 MLE 解、且两边都取到全局最优时，RLHF 目标（式 (1)）与 DPO 目标（式 (6)）的最优策略重合（[Rafailov et al., 2023](../references.md)，[A]，§4；IPO 论文 Proposition 4 把这一对应推广到更一般的形式，[B]）。**证明**（要点）：三步——BT 精确使偏好分布只由 reward 等价类决定，MLE 使训出的 RM 落在真 reward 的类里，两边取到全局最优时最优策略都是同一个闭式解。但等价链的每一环都带着前提。实践中两者的偏离可以分四笔账来列。

**数据分布：offline 对 on-policy**。DPO 的期望取自静态偏好集 $D$（按 3.2 §5 的判据是 offline 目标）， $D$ 里的 $(y_w, y_l)$ 来自采样时刻的旧策略（通常 $\pi_{\mathrm{SFT}}$）。一旦 $\pi_\theta$ 在训练中移出 $D$ 的支撑集，数据对新行为区域就零约束——不重采就没有纠偏信号。PPO 相反：每轮从当前策略重采 rollout、用 RM 现场打分，监督分布随策略走（on-policy，账与 staleness 管理归 8.3/8.5）。折中形态是迭代式 DPO：每轮用当前最优模型重采偏好对再训。Llama-3 的 DPO 就只用「上一轮产出的最新偏好批次」训练（[Grattafiori et al., 2024](../references.md)，[A]，§4.1.4）——本质是把 offline 目标的数据源周期性拉回当前策略附近，向 on-policy 挪半步。

**reward 过拟合的两副面孔**。RLHF 路线里 reward hacking 是显式的：RM 是代理，策略把它优化过度就偏离真偏好（9.2 的 Goodhart 分析）。DPO 路线里不存在可被 hack 的独立 RM，但式 (6) 的隐式 reward 差无界（§4）。IPO 的诊断（§6）是：确定性偏好下 DPO 会把 logit 差推向无穷，策略退化成 $\pi(y_l) = 0$ 的极端形式，KL 约束在目标函数里被 sigmoid 的饱和吞掉。同一份风险，RLHF 里是「RM 被刷分」，DPO 里是「似然差被无限推高」。IPO 还有一个反向观察：实践中 RLHF 的 RM 通常是**欠拟合**的（有限数据、加正则、early stop），欠拟合的 RM 把 reward 差钳在有限范围内，反而充当了策略的正则化器；DPO 跳过 RM，把这份隐式正则一并跳过了（[Azar et al., 2023](../references.md)，[B]，§4.2）。

**KL 约束的作用位置**。PPO 式 RLHF 把 KL 逐 token 折进 reward，作为逐 token 惩罚项贯穿每个 rollout 步（8.3、9.2）。DPO 的 $\beta$ 只在 sigmoid 里出现一次，约束的全部效力都走「梯度随排序正确而衰减」这条间接通道——偏好一旦排对，把 $\pi_\theta$ 拉回 $\pi_{\mathrm{ref}}$ 的力也随之消失。这不是 bug：式 (6) 在目标层与式 (1) 同解。但它意味着 DPO 对 $\beta$ 的响应形态与 PPO 的 KL 系数不同——PPO 的 KL 是持续的拉回，DPO 的 $\beta$ 是 margin 的缩放常数。

**实证对照**。DPO 论文的可控实验（IMDb 情感转向，真 reward 可取）显示：DPO 的 reward-KL 前沿严格优于 PPO，包括能用真 reward 的 PPO-GT 变体；TL;DR 摘要任务上，DPO 在 temperature 0 下对参考摘要的胜率约 61%，PPO 约 57%（同 SFT 起点，[Rafailov et al., 2023](../references.md)，[A]，§6）。工业侧的选型记录是 Llama-3：他们试训过 PPO，结论是 DPO「在大模型上所需算力更少、表现更好，尤其 IFEval 类指令遵循基准」（[Grattafiori et al., 2024](../references.md)，[A]，§4.1.4）；配置为 lr $10^{-5}$、 $\beta = 0.1$，外加两条稳定化改造——屏蔽格式化 token（防止对比目标在公共特殊 token 上自相矛盾），以及 chosen 序列上加系数 0.2 的 NLL 正则（防止 chosen 似然绝对值下滑）。组件账的另一面：DPO 只需策略加冻结参考两份权重，PPO-RLHF 需四份（actor/critic/RM/ref）——8.5 的显存账。

> **提点**：等价是目标层的，工程不是。

## 6. IPO：把 logit 换回恒等映射，治无界性

DPO 的损失为什么没有「拟合完成」的驻点？IPO（[Azar et al., 2023](../references.md)，arXiv:2310.12036，[B]）分两步回答：先给一个把 RLHF 与 DPO 统一进去的通用框架，再指出框架里哪个部件导致过拟合，换掉它。

**ΨPO 框架**。设 $\Psi: [0,1] \to \mathbb{R}$ 为任意非减映射， $\mu$ 为行为策略（采样偏好对所用的分布）， $\tau > 0$ 为正则系数（地位同 DPO 的 $\beta$），定义

$$
\max_{\pi}\hspace{4mu} \mathbb{E}_{x \sim \rho,\hspace{4mu} y \sim \pi(\cdot \mid x),\hspace{4mu} y' \sim \mu(\cdot \mid x)}\left[ \Psi\left( p^{\ast}(y \succ y' \mid x) \right) \right] - \tau \thinspace D_{\mathrm{KL}}\left( \pi \Vert \pi_{\mathrm{ref}} \right) \qquad (8)
$$

**符号**： $p^{\ast}(y \succ y' \mid x)$：真实偏好概率（对人类标注噪声取期望后的量）； $\rho$：prompt 分布。DPO/RLHF 是 $\Psi(q) = \log(q/(1-q))$（logit 映射）的特例：BT 假设下 $\Psi(p^{\ast}(y \succ y')) = r(y) - r(y')$，期望号内退化为 reward 差，式 (8) 回到式 (1)（IPO 论文 Proposition 1）。

**过拟合诊断（其 §4.2）**。logit 映射在 $q \to 1$ 时发散到 $+\infty$。取确定性偏好的极端情形： $p^{\ast}(y \succ y') = 1$，BT 模型要表达它就必须让 $r(y) - r(y') \to +\infty$；代入闭式解式 (3)，得 $\pi^{\ast}(y') / \pi^{\ast}(y) = 0$。也就是说，无论 $\tau$ 多大， $y'$ 的概率都被压成零，KL 正则形同虚设。有限数据让问题更尖锐：真实偏好哪怕是 0.8，少量样本的估计也完全可能是 $\hat{p} = 1$，经验最优同样把 $y'$ 清零。根因有两层： $\Psi$ 无界，加上没有显式 RM 提供欠拟合正则（§5 已述）。

**IPO = $\Psi$ 取恒等**。路线五步：恒等映射代入变分问题、解出最优对数比的有界靶值、把「达到靶值」改写成最小二乘、换成样本示性函数、对称合并降方差——关键一步是「平方损失把无界的分类目标换成有界回归」。有界的 $\Psi$ 让 KL 项在确定性偏好下仍然有效；最自然的选择是 $\Psi(q) = q$，目标变成「直接最大化对 $\mu$ 的总偏好」 $`\max_\pi p^{\ast}_{\rho}(\pi \succ \mu) - \tau D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}})`$。与 §2 同样的变分给出最优解 $\pi^{\ast}(y) \propto \pi_{\mathrm{ref}}(y) \exp(\tau^{-1} g(y))$、 $`g(y) = \mathbb{E}_{y' \sim \mu}[p^{\ast}(y \succ y')]`$；取对数比后，最优策略满足

$$
h^{\ast}(y, y') = \log \frac{\pi^{\ast}(y) \thinspace \pi_{\mathrm{ref}}(y')}{\pi^{\ast}(y') \thinspace \pi_{\mathrm{ref}}(y)} = \tau^{-1} \left( p^{\ast}(y \succ \mu) - p^{\ast}(y' \succ \mu) \right) \qquad (9)
$$

即最优策略的对数似然比差应当等于一个**有界常数**（总偏好差的 $\tau^{-1}$ 倍），而非像 DPO 那样被逼向无穷。IPO 的做法是把式 (9) 视作关于 $h_\pi$ 的求根问题，改写成最小二乘： $`L(\pi) = \mathbb{E}_{y, y' \sim \mu}[(h_\pi(y, y') - \tau^{-1}(p^{\ast}(y \succ \mu) - p^{\ast}(y' \succ \mu)))^{2}]`$。**命题**（靶值唯一可达；其 Theorem 2）：该损失在 $\pi^{\ast}$ 处取零，且（在支撑集匹配的条件下）全局/局部最优唯一。**证明**：原论文，本书不重复。总体目标里的 $p^{\ast}$ 不可观测，换成样本示性函数 $I(y, y') \sim \mathrm{Bernoulli}(p^{\ast}(y \succ y'))$ 后，两式仅差一个与 $\pi$ 无关的常数（其 Proposition 3；证明用到 $h_\pi$ 对 $(y, y')$ 的可加分解与 $y, y'$ 独立同分布这两条对称性）。再利用每对数据同时贡献 $(y_w, y_l, 1)$ 与 $(y_l, y_w, 0)$ 两项的对称性合并降方差，得经验损失（其式 (17)）：

$$
\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}_{(x, y_w, y_l) \sim D}\left[ \left( h_\pi(x, y_w, y_l) - \frac{1}{2\tau} \right)^{2} \right], \qquad h_\pi(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_\theta(y_l \mid x)} - \log \frac{\pi_{\mathrm{ref}}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \qquad (10)
$$

全数字例：取 τ = 1/2，靶值 = 1/(2τ) = 1。某偏好对的对数似然比差为 0（策略与参考尚无差异）时，该样本损失 = (0 − 1) 的平方 = 1；对数似然比差升到 2 时损失同样是 1——偏离靶值无论方向都受罚，与 DPO「差越大越好」的单调形态正好相反。

读法：IPO 不再问「chosen 是否比 rejected 更可能被偏好」（sigmoid 分类），而是直接把对数似然比差**回归**到固定靶值 $1/(2\tau)$。 $\tau$ 的方向从式 (8) 读出： $\tau$ 乘在 KL 项上， $\tau$ 大意味着强正则，策略被压在 $\pi_{\mathrm{ref}}$ 附近，要求的两支似然比差就小——式 (9) 右端的 $\tau^{-1}$ 因子与式 (10) 的靶值 $1/(2\tau)$ 与此自洽（强正则 ⇒ 小靶值）。与 DPO 的对照很直接：DPO 把差推向 sigmoid 的渐近区（无界），IPO 把差固定在 $1/(2\tau)$（有界驻点），过拟合确定性偏好的通道就此关闭。

> **坑**：照 IPO 论文文本写序列级求和的平方损失，损失会随长度缩放、与论文报告的数字对不上。TRL main 的 `dpo_trainer.py` IPO 分支对 $h_\pi$ 按 completion 长度做 token 平均再平方（源码注释：此归一化「论文未写明、已与作者确认、论文报告数字对应该归一化形式」）——这是论文文本之外的实现层事实，引用时以代码注释口径为准。

## 7. KTO：单边数据与 Kahneman-Tversky 价值函数

手里只有「这个回答好/坏」的单边标签、没有成对偏好，还能做偏好优化吗？KTO（[Ethayarajh et al., 2024](../references.md)，arXiv:2402.01306，[A]）换的正是数据形态：成对偏好 $(x, y_w, y_l)$ 换成单边标签 $(x, y, \pm)$——「这个回答好/坏」，不要求成对。动机账是数据成本：成对标注要同一 prompt 的两个样本配对比较，单边标签的生产形态宽得多（点赞/点踩、采纳/拒绝、对话是否继续）。

理论基础是前景理论（prospect theory）的价值函数。Tversky-Kahneman 的经典形式（跨人类被试的实验拟合中位数 $\alpha = 0.88$、 $\lambda = 2.25$）：

$$
v(z) = \begin{cases} (z - z_0)^{\alpha} & z \ge z_0 \\ -\lambda (z_0 - z)^{\alpha} & z \lt z_0 \end{cases} \qquad (11)
$$

**符号**： $z_0$：参考点（reference point）——人对「得与失」的感知都相对于参考点而非绝对量； $\alpha < 1$：收益区凹（边际效用递减）； $\lambda > 1$：损失区更陡（损失厌恶）。KTO 论文先把现有偏好目标统一成 HALO（human-aware loss）框架——价值函数形如 $`v(r_\theta(x, y) - \mathbb{E}_{Q}[r_\theta(x, y')])`$、 $v$ 非减且在收益区凹——并证明 DPO 与 PPO-Clip 都是 HALO（其 Theorem 3.5）——这是两个正例；边界反例是收益区取凸形的价值函数（如平方），它不满足凹性要求，被 HALO 排除在外。然后它把式 (11) 实例化到 LLM 对齐：幂函数换成数值上更稳的 sigmoid（同为收益凹、损失凸），参考点取「策略整体的漂移量」：

$$
\mathcal{L}_{\mathrm{KTO}}(\pi_\theta, \pi_{\mathrm{ref}}) = \mathbb{E}_{(x, y) \sim D}\left[ \lambda_y - v_{\mathrm{KTO}}(x, y) \right] \qquad (12)
$$

$$
v_{\mathrm{KTO}}(x, y) = \begin{cases} \lambda_D \thinspace \sigma\left( \beta \left( r_\theta(x, y) - z_0 \right) \right) & y \sim y_{\mathrm{desirable}} \\ \lambda_U \thinspace \sigma\left( \beta \left( z_0 - r_\theta(x, y) \right) \right) & y \sim y_{\mathrm{undesirable}} \end{cases}, \qquad r_\theta(x, y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \qquad (13)
$$

**符号**： $z_0 = D_{\mathrm{KL}}(\pi_\theta(\cdot \mid x) \Vert \pi_{\mathrm{ref}}(\cdot \mid x))$：参考点，「平均水平的回答比参考模型好多少」； $\lambda_D, \lambda_U$：得/失两侧的权重（损失厌恶的载体）； $\lambda_y$ 按 $y$ 的极性取 $\lambda_D$ 或 $\lambda_U$，只起保持损失非负的作用。结构的读法：desirable 样本要把 $r_\theta$ 推到参考点之上，undesirable 样本要压到参考点之下。与 DPO 的差分不同，这里是逐样本对参考点的单边拉扯，不需要同 prompt 内的配对。全数字例（β = 1）：某 desirable 样本高出参考点 ln 3，价值 = λ_D·σ(ln 3) = 3λ_D/4；某 undesirable 样本低于参考点 ln 3，价值 = 3λ_U/4——离参考点同样的距离，两侧各乘各的权重。

$z_0$ 的估计是工程要点。定义式要对全策略分布求 KL（即从 $\pi_\theta$ 采样，慢且不可微）；实践取有偏替代：batch 内把 completion 错位配对（ $x_i$ 配 $y_{i+1}$ 的循环移位），以错配对的 log 比均值估计

$$
\hat{z}_0 = \max\left( 0,\hspace{4mu} \frac{1}{m} \sum_{i=1}^{m} \log \frac{\pi_\theta(y_j \mid x_i)}{\pi_{\mathrm{ref}}(y_j \mid x_i)} \right), \qquad j = (i + 1) \bmod m \qquad (14)
$$

且不回传梯度（原文： $z_0$ 只控制损失饱和位置）；clamp 在 0 使估计带正偏但方差更低。用错配对而不用本配对的理由：数据里的 $(x, y)$ 是被刻意挑选的好/坏样本，log 比幅度不代表策略的典型行为。TRL main 的 `kto_trainer.py` 逐项对应： `kl = (KL_logps - ref_KL_logps).mean().detach()` 后 `.clamp(min=0)`，chosen 支 `1 - sigmoid(beta * (logratios - kl))`、rejected 支 `1 - sigmoid(beta * (kl - logratios))`，权重乘 `desirable_weight` / `undesirable_weight`（教学参照，2026-11-18 核实）。

> **坑**：microbatch 取 1 时 KTO 的参考点无从估计—— $z_0$ 靠 batch 内错位配对构造，至少需要两条样本才能凑出错位对。这是一个容易忽略的 batch 约束。

超参口径（其 §4.2，逐条原文）分三条：

- **学习率**：KTO 对学习率比 DPO 更敏感，推荐起点 5e-6（DPO 默认 5e-7 的 2-10 倍），因为参考点调整后的 reward 幅度更小。
- ** $\beta$ 分两档**：已过 SFT 的较大模型取 $[0.01, 0.10]$；未经 SFT 直接对齐的较小模型取 $[0.10, 1.00]$。
- **类别不平衡**：用权重配平，经验区间 $\lambda_D n_D / (\lambda_U n_U) \in [1, 3/2]$（得侧灵敏度略占优）， $n_D, n_U$ 为两类样本数。

实证位置：GPT-4 裁判的胜率口径下，KTO 在各模型规模上不劣于 DPO；对 Llama 系列，KTO 单用（不经过 SFT）即匹配 SFT+DPO 两阶段的成绩、且好于 DPO 单用（其 Figure 3 口径，误差棒为 90% 二项置信区间）；「每 prompt 只留一个 $y$」的去配对消融（one-y-per-x）验证了单边数据的可行性（[Ethayarajh et al., 2024](../references.md)，[A]，Figure 3 与 §4）。

## 8. SimPO：reference-free 的平均 logp 与 margin

reference 模型能不能整个省掉？SimPO（[Meng et al., 2024](../references.md)，arXiv:2405.14734，[A]）保留成对偏好数据与 sigmoid 形态，换掉的是隐式 reward 的定义。论文对 DPO 隐式 reward 提了两条动机批评（其 §2.2，逐字口径）。第一，训练时需要一个 reference 模型 $\pi_{\mathrm{ref}}$，带来额外的显存与计算成本。第二，也是更根本的一条：**训练时优化的 reward 与生成时的决策指标不匹配**。DPO 的 $r = \beta \log(\pi_\theta / \pi_{\mathrm{ref}})$ 含参考比，而解码端（beam search、多项选择排序）没有 reference 模型参与，用的是平均 log 似然。后果可以直接测：reward 排序 $r(x, y_w) > r(x, y_l)$ 成立并不蕴含似然排序 $p_\theta(y_w \mid x) > p_\theta(y_l \mid x)$ 成立——论文实测，DPO 训完的模型在训练集三元组上只有约 50% 满足两排序一致（其 Figure 4(b)）；一项同期工作也发现，现有模型经 DPO 后在平均 log 似然口径下的排序准确率接近随机。

直接的替代——token log 概率**求和**——有 length bias：序列越长，总 logp 越低。把求和值当 reward 优化时，若 $y_w$ 比 $y_l$ 长，模型就被迫人为抬高长序列的概率来凑出差分，退化风险上升。解法是把生成本身使用的指标直接当 reward，即长度归一化的平均 logp：

$$
r_{\mathrm{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{\lt t}) \qquad (15)
$$

**符号**： $|y|$：回答的 token 数； $\beta$：reward 差的缩放常数。reference 模型消失——不再需要 $Z(x)$、 $\pi_{\mathrm{ref}}$，显存与计算各省一份。

第二个部件是目标 margin。把 Bradley-Terry 目标改为要求 reward 差**超过**一个正阈值 $\gamma$（BT 文献中称 home advantage）：

$$
p(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) - \gamma \right) \qquad (16)
$$

代入式 (15) 得 SimPO 目标（其式 (6)）：

$$
\mathcal{L}_{\mathrm{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[ \log \sigma\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right] \qquad (17)
$$

全数字例：chosen 平均 logp = −3/2（长 4 token，sum = −6），rejected 平均 logp = −3/2（长 8 token，sum = −12），取 β = 2、γ = 1。SimPO 口径的 sigmoid 输入 = 2×(−3/2) − 2×(−3/2) − 1 = −1，损失 = ln(1 + e)；同一对样本放进 sum 口径（去掉 γ），输入 ∝ (−6) − (−12) = 6——平均口径下打不平的账（margin 吃掉全部差值），sum 口径却把更短的 chosen 直接奖成正 margin。

两个设计各自的实证账（其 §4.2-4.3）。**长度归一化**：带 LN 的 SimPO 在所有偏好对上都学到正 reward 差，与长度差无关；去掉 LN 后，reward 与长度呈强正 Spearman 相关，且对「 $y_w$ 短于 $y_l$ 」的对子学到负差分——长度被当成 reward 的代理榨取（length exploitation）。**margin**： $\gamma$ 增大使留出集上的 reward 排序准确率单调上升，但 AlpacaEval 2 胜率先升后降。其 Figure 3(b)(c) 给出分布证据（原文 §4.3 正文误作 Figure 2(b)(c)，实为 Figure 3 的 γ 扫描图）：增大 $\gamma$ 会压扁 reward 分布并拉低 chosen 序列的平均 log 似然，过头即退化；原文给出的解释停在假设层（reward 分布拟合与似然校准之间的权衡），本书不加深。推荐区间 $\beta \in [2.0, 2.5]$、 $\gamma \in [0.5, 1.5]$（逐设定最优值在其附录 Table 8：Llama-3-Instruct 设定为 $\beta = 2.5$、 $\gamma = 1.4$）。

> **坑**：把 SimPO 的 $\beta$ 与 DPO 的 $\beta$ 直接互搬——两家差一个数量级（0.1 对 2.5）：平均 logp 的逐 token 尺度约 $-1$ 至 $-3$，乘 $\beta = 2.5$ 后的 reward 差与 DPO 的 sum-logp 比差同量级。换 reward 单位就要先换算超参，跨方法搬 $\beta$ 拿到的正则强度整个错位。

实证位置：AlpacaEval 2 上最多超 DPO 6.4 分、Arena-Hard 上最多超 7.5 分；Gemma-2-9B-it 上的最优配置拿到 AlpacaEval 2 长度控制胜率 72.4%、Arena-Hard 59.1%（其摘要与 §4.1 口径），且响应长度无显著膨胀。

## 9. ORPO：把偏好惩罚折进 SFT 的 odds ratio

只对 chosen 做最大似然的 SFT，给 rejected 回答留了什么约束？答案是「无任何约束」——ORPO（[Hong et al., 2024](../references.md)，arXiv:2403.07691，[A]）的起点正是这个观察。其 §3 的实证（OPT-350M 在 HH-RLHF 上微调，Figure 3）：尽管监督只打在 chosen 上，chosen 与 rejected 两支的 log 概率**同时上升**，rejected 有时甚至高于 chosen——域内适配的代价是 rejected 风格被一并抬升；而后置的偏好对齐阶段（DPO 等）才是纠偏环节。ORPO 的主张：纠偏不需要独立阶段——在 SFT 的 NLL 上加一个拉开两支差距的小惩罚即可，单阶段、无 reference 模型。

惩罚的构造用 odds ratio。对序列级「概率」 $P_\theta(y \mid x)$（论文式 (3) 的口径：其 log 取逐 token 平均 log 似然，即几何平均意义上的序列得分）定义 odds：

$$
\mathrm{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}, \qquad \mathcal{L}_{\mathrm{OR}} = -\log \sigma\left( \log \frac{\mathrm{odds}_\theta(y_w \mid x)}{\mathrm{odds}_\theta(y_l \mid x)} \right) \qquad (18)
$$

**符号**：odds 是「生成 $y$ 相对不生成 $y$ 的倍数」；log odds ratio 比较两支的这个倍数。总目标（其式 (6)）：

$$
\mathcal{L}_{\mathrm{ORPO}} = \mathbb{E}_{(x, y_w, y_l) \sim D}\left[ \mathcal{L}_{\mathrm{SFT}} + \lambda \cdot \mathcal{L}_{\mathrm{OR}} \right] \qquad (19)
$$

$`\mathcal{L}_{\mathrm{SFT}}`$ 只作用于 chosen 支（标准 NLL）， $`\mathcal{L}_{\mathrm{OR}}`$ 负责把两支分开——「SFT 管域内适配，odds 项负责压低 rejected 风格」的分工（其 §4.2 口径）。全数字例：设 chosen 的序列得分 P_w = 1/2、rejected P_l = 1/4，则 odds 分别为 1 与 1/3，log odds 比 = ln 3，odds 项损失 = ln(4/3)；梯度里 rejected 支的力度因子 1/(1 − P_l) = 4/3——P_l 越接近 1，压它的力度越大。

为什么用 odds 而不直接用 log 概率比（DPO 式）？梯度分解给出答案（其 §4.3 与附录 A；符号按本书重排）：记 $`u = \log(\mathrm{odds}_w / \mathrm{odds}_l)`$，则

$$
\nabla_\theta \mathcal{L}_{\mathrm{OR}} = -\underbrace{\frac{1}{1 + \mathrm{odds}_w / \mathrm{odds}_l}}_{\delta(d)} \cdot \underbrace{\left( \frac{\nabla_\theta \log P_\theta(y_w \mid x)}{1 - P_\theta(y_w \mid x)} - \frac{\nabla_\theta \log P_\theta(y_l \mid x)}{1 - P_\theta(y_l \mid x)} \right)}_{h(d)} \qquad (20)
$$

（链式法则： $-\log \sigma(u)$ 的导数贡献 $\sigma(-u) = 1/(1 + e^{u})$； $\log \mathrm{odds} = \log P - \log(1 - P)$ 的导数是 $\nabla \log P / (1 - P)$。论文式 (8)-(10) 写的是幅度形式 $`\nabla \mathcal{L}_{\mathrm{OR}} = \delta(d) \cdot h(d)`$，梯度下降方向为 $-\delta(d) \cdot h(d)$，本书保留显式负号。）结构上看： $\delta(d)$ 与 DPO 式 (7) 的 sigmoid 权重同型（两支差距越大、权重越小）； $h(d)$ 的分母 $1 - P_\theta$ 是关键差异—— $P_l \to 1$ 时 $1/(1 - P_l) \to \infty$，即 rejected 回答当前概率越高，压它的梯度力度越大； $P_l \to 0$ 时收敛到纯 log 比的力度。odds 形式把「当前犯错的程度」放大进梯度，而纯 log 比的梯度对两支的当前水平不敏感。这是论文对「odds 是 sensible choice」的机制论证（其 §4.3 与附录 A 的对照实验支撑）。

实证位置：只在 UltraFeedback 上单阶段训练，Phi-2（2.7B）/Llama-2（7B）/Mistral（7B）超过若干 7B-13B 的既有对齐模型——AlpacaEval 2.0 最高 12.20%、IFEval 66.19%、MT-Bench 7.32（其摘要口径）。实现注记：TRL 的 ORPO 原在 `trl/trainer/orpo_trainer.py`，main 分支已于 2026-02 移入 `trl/experimental/orpo/orpo_trainer.py`（commit 「Remove deprecated ORPO after moved to experimental」，2026-11-18 核实）；TRL 实现取 `average_log_prob=True`（与论文式 (3) 的平均口径一致）， `log_odds = (chosen_logps - rejected_logps) - (log1mexp(chosen_logps) - log1mexp(rejected_logps))`（ `log1mexp` 为 $\log(1 - e^{x})$ 的稳定计算），总损失为 `nll_loss - beta * logsigmoid(log_odds).mean()`——式 (19) 的直接对应。

> **坑**：读 TRL 配置时，ORPO 的 $\lambda$ 在 TRL 里命名为 `beta`，与 DPO 的 $\beta$ 同名异物——照 DPO 的经验值填 ORPO 的 `beta`，正则强度整个错位。

## 10. 家族对照总表与实现走读

五家摆在一起，四个口径问题一次对照：各家目标形态、reference 需求、数据形态、锚定机制差在哪（「长度处理」指目标对回答长度的隐式偏置控制；「锚定机制」指防止策略无限漂移的部件）：

| 方法 | 目标形态 | reference 需求 | 数据形态 | 长度处理 | 锚定机制 | 实证位置（各论文口径） |
| --- | --- | --- | --- | --- | --- | --- |
| DPO 式 (6) | $-\log\sigma(\beta\Delta_w - \beta\Delta_l)$ | 需要 $\pi_{\mathrm{ref}}$（冻结） | 成对偏好 | 无（sum logp，隐式长度偏置） | $\beta$ 缩放 + sigmoid 饱和 | TL;DR 胜率 61% vs PPO 57% [A] |
| IPO 式 (10) | $(h_\pi - 1/(2\tau))^{2}$ 平方回归 | 需要 $\pi_{\mathrm{ref}}$ | 成对偏好 | 论文无；TRL 实现加 token 平均 | 有界靶值 $1/(2\tau)$ | bandit 例上治 DPO 过拟合 [B] |
| KTO 式 (12)(13) | 价值函数 $\sigma(\beta(r - z_0))$ | 需要 $\pi_{\mathrm{ref}}$（含 $z_0$） | 单边 $\pm$ 标签 | 无 | 参考点 $z_0$（batch 内 KL 估计） | Llama 上 KTO 单用匹配 SFT+DPO [A] |
| SimPO 式 (17) | $`-\log\sigma(\beta \cdot \overline{\log p}_w - \beta \cdot \overline{\log p}_l - \gamma)`$ | 不需要 | 成对偏好 | 平均 logp 显式归一 | margin $\gamma$（无 KL 锚） | AlpacaEval 2 +6.4 / Arena-Hard +7.5 vs DPO [A] |
| ORPO 式 (19) | NLL $+ \lambda \cdot (-\log\sigma(\log \mathrm{OR}))$ | 不需要 | 成对（chosen 兼作 SFT 目标） | 几何平均（平均 log 似然口径） | SFT 项本身 | UltraFeedback 单阶段 7B 超 13B 口径 [A] |

读表要点。其一，reference 需求与数据形态是两个独立的轴：KTO 要 reference 但不要配对，SimPO 要配对但不要 reference，ORPO 两个都不要但要求 chosen 支可直接做 SFT。其二，「锚定」一列的分歧是设计哲学的分歧：DPO/IPO/KTO 保留 $\pi_{\mathrm{ref}}$ 作为显式锚（闭式解的血统），SimPO/ORPO 放弃显式锚、改用 margin 或 SFT 项来防漂移——放弃 KL 锚换来的自由度由「不过度漂移」的实证承诺背书，而非目标层保证。其三，长度偏置是全家族的公共战场：DPO 的 sum logp 隐式偏置（§4）、IPO 论文外 TRL 补的 token 平均（§6）、SimPO 的平均 logp 显式归一（§8）、ORPO 的几何平均口径（§9）——同一问题的四种计价方式。

TRL main 分支的对照走读（教学参照 + 官方实现，2026-11-18 核实）：

- **DPO 主链**在 `trl/trainer/dpo_trainer.py` 的 `_compute_loss`：`ref_logps = ref_per_token_logps.sum(dim=1)`（token 求和，§4 长度耦合的实现层证据）；`chosen_logratios = chosen_logps - ref_chosen_logps`；`f_divergence_type == "reverse_kl"` 下 `delta_score = chosen_scores - rejected_scores`；`loss_type == "sigmoid"` 时 `per_sequence_loss = -F.logsigmoid(self.beta * delta_score)`——式 (6) 的逐行落地。
- **IPO 分支**：`(ipo_delta - 1/(2*self.beta))**2`（式 (10)，含 §6 所述 token 平均）。
- **KTO** 在 `trl/trainer/kto_trainer.py`（§7 已走读）；**ORPO** 移入 `trl/experimental/orpo/orpo_trainer.py`（§9 已走读）。

DPO 官方复现 repo（eric-mitchell/direct-preference-optimization， `trainers.py` 的 `dpo_loss`，[A]）更贴论文原文： `pi_logratios = policy_chosen_logps - policy_rejected_logps`、 `logits = pi_logratios - ref_logratios`（注释即「also known as $h_\pi^{y_w, y_l}$」）、`losses = -F.logsigmoid(beta * logits)`；IPO 分支 `(logits - 1/(2*beta))**2` 直接标注「Eq. 17」。另有 `reference_free=True` 开关（ref 对数比置零——隐式均匀参考，是消融件而非论文主张）。输入格式侧（chosen/rejected 同 batch 对半拼接的前向复用）3.2 §6 已走读，此处不重复。

## 总结

| 环节 | 公式 | 要点 |
| --- | --- | --- |
| KL 约束目标 | 式 (1) | 逐 prompt 无耦合 ⇒ 可逐 prompt 变分 |
| 闭式最优解 | 式 (3) | $\pi^{\ast} = Z(x)^{-1} \pi_{\mathrm{ref}} \exp(r/\beta)$；最优值 $\beta \log Z(x)$；Gibbs 取等 |
| reward 重参数化 | 式 (4) | $r = \beta\log(\pi^{\ast}/\pi_{\mathrm{ref}}) + \beta\log Z(x)$； $Z$ 即 3.2 的 gauge |
| BT 代入 | 式 (5) | $Z(x)$ 逐 prompt 相消 = 平移不可识别性兑现；Theorem 1：等价类无表达损失 |
| DPO 目标 | 式 (6) | offline、双模型、隐式 reward $`\hat{r}_\theta = \beta\log(\pi_\theta/\pi_{\mathrm{ref}})`$ |
| DPO 梯度 | 式 (7) | 排序错误加权 $`\sigma(\hat{r}_l - \hat{r}_w)`$；无界性与长度耦合两个结构性偏置 |
| vs PPO | §5 | 目标层等价（BT 精确 + MLE + 全局最优）；数据分布 / 正则通道 / KL 作用位置三层偏离；Llama-3 选 DPO（算力少、IFEval 好，lr 1e-5、 $\beta$=0.1 + 两条稳定化）[A] |
| IPO | 式 (8)(9)(10) | ΨPO 框架；logit 无界 ⇒ 确定性偏好下 KL 失效；平方回归到靶值 $1/(2\tau)$ [B] |
| KTO | 式 (11)-(14) | KT 价值函数（ $\alpha$=0.88、 $\lambda$=2.25 经典值）；单边数据； $z_0$ 用错配对估计、不回传 [A] |
| SimPO | 式 (15)-(17) | 平均 logp 消 reference 且对齐生成指标；margin $\gamma$； $\beta \in [2.0,2.5]$、 $\gamma \in [0.5,1.5]$ [A] |
| ORPO | 式 (18)-(20) | odds ratio 惩罚折进 SFT； $1/(1-P)$ 把当前犯错程度放大进梯度；TRL 已移入 experimental [A] |

## 练习

1.（推导 · [20]）IPO 的对称合并：从采样损失 $`\mathbb{E}_{y, y' \sim \mu}[(h_\pi(y, y') - \tau^{-1} I(y, y'))^{2}]`$ 出发，利用每对 $(y_w, y_l)$ 同时贡献 $(y_w, y_l, 1)$ 与 $(y_l, y_w, 0)$ 两项，证明经验损失等于 $`\mathbb{E}_D[(h_\pi(y_w, y_l) - \tau^{-1})^{2} + h_\pi(y_l, y_w)^{2}]/2`$，再配方合并为式 (10)（差一个与 $\pi$ 无关的常数）；指出该常数是什么。

2.（计算 · [20]）长度偏置算账：设 chosen 与 rejected 的逐 token 平均 logp 同为 $-1.5$，chosen 长 50 token、rejected 长 200 token。先写出你对 (a)(b) 两口径 sigmoid 输入符号的预测，再动手算。(a) 算 DPO 口径（sum logp、 $\beta = 0.1$、 $\pi_{\mathrm{ref}}$ 与 $\pi_\theta$ 暂相等）下式 (6) 的 sigmoid 输入与损失值；(b) 换成 SimPO 口径（平均 logp、 $\beta = 2.5$、 $\gamma = 0.5$）重算；(c) 若训练把两支的平均 logp 都推升 0.1，两口径的损失各变多少——定量说明 sum 口径下「拉长 rejected」也是一条降低损失的通路。校验：(a)(b) 两口径的 sigmoid 输入符号应相反；(c) 中 sum 口径的损失变化方向应与「拉长 rejected 能降损」的判断一致。

3.（推导 · [30]）把 §2 的逐 prompt 变分补全到逐字严格：从式 (1) 出发，(a) 论证 $J(\pi)$ 对 $x$ 的可分解性；(b) 完成式 (2) 的配分函数补齐步；(c) 用 Gibbs 不等式给出取等条件；(d) 代回求出逐 prompt 最优值 $\beta \log Z(x)$。(e) 设 $r$ 乘常数 $c > 0$、 $\beta$ 同乘以 $c$，证明 $\pi^{\ast}$ 不变——reward 的尺度自由度被 $\beta$ 吸收，这解释了各论文 $\beta$ 口径不可互搬（警示：若误把 $\beta$ 除以 $c$， $r/\beta$ 会多出 $c^{2}$ 因子、 $\pi^{\ast}$ 改变——尺度共变只允许同乘）。

4.（代码 · [30]，教学用）用 TRL main 的 `dpo_trainer.py` 与 `experimental/orpo/orpo_trainer.py` 各跑一个 toy 偏好集（或读源码后手算），核对前先写下预测，再核对：(a) DPO 的 `delta_score` 在 `reverse_kl` 下恰为式 (6) 的 sigmoid 输入；(b) ORPO 的 `log1mexp` 写法与式 (18) 的 $\log(1 - P)$ 在数值上的差异（取 $P$ 接近 1 的 logp 触发朴素写法的下溢）；(c) IPO 分支的 token 平均相对论文式 (17) 在变长数据上差多少。校验：(b) 中朴素写法在 logp 极接近 0（即 P 接近 1）时下溢为 −inf，`log1mexp` 仍给出有限值。

## 参考

- [Rafailov et al., 2023](../references.md)：Direct Preference Optimization: Your Language Model is Secretly a Reward Model，arXiv:2305.18290，[A]（key: dpo）——式 (1)(3)(4)(5)(6)(7) 全链、Theorem 1 与 Lemmas、IMDb/TL;DR 实证；官方 repo eric-mitchell/direct-preference-optimization `trainers.py`
- [Azar et al., 2023](../references.md)：A General Theoretical Paradigm to Understand Learning from Human Preferences，arXiv:2310.12036，[B]（key: ipo）—— ΨPO 框架式 (8)、过拟合分析（§4.2）、IPO 求根-最小二乘链与式 (17)、Theorem 2 唯一性
- [Ethayarajh et al., 2024](../references.md)：KTO: Model Alignment as Prospect Theoretic Optimization，arXiv:2402.01306，[A]（key: kto）——KT 价值函数式 (11)、HALO 框架、KTO 损失式 (12)(13)、 $`\hat{z}_0`$ 估计式 (14)、超参口径（lr 5e-6、 $\beta$ 两档、 $\lambda_D n_D / \lambda_U n_U \in [1, 3/2]$）
- [Meng et al., 2024](../references.md)：SimPO: Simple Preference Optimization with a Reference-Free Reward，arXiv:2405.14734，[A]（key: simpo）——两动机（reference 成本、生成指标不匹配，~50% 排序一致率）、平均 logp reward 式 (15)、margin 式 (16)、目标式 (17)、LN 防 length exploitation（§4.2）、推荐超参区间
- [Hong et al., 2024](../references.md)：ORPO: Monolithic Preference Optimization without Reference Model，arXiv:2403.07691，[A]（key: orpo）——odds ratio 式 (18)、总目标式 (19)、梯度分解式 (20)、UltraFeedback 单阶段实证
- [Bradley & Terry, 1952](../references.md)：Rank Analysis of Incomplete Block Designs，[B]（key: bradley-terry，无 arXiv）——偏好模型出处（3.2 已推导）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——DPO 选型与配置（§4.1.4：lr $10^{-5}$、 $\beta$=0.1、格式化 token 屏蔽、NLL 正则 0.2、只用最新偏好批次）
- [HuggingFace, huggingface/trl repo](../references.md)（key: trl-repo，教学参照 + 官方实现）——`trl/trainer/dpo_trainer.py`（`_compute_loss`、IPO 分支）、`trl/trainer/kto_trainer.py`、`trl/experimental/orpo/orpo_trainer.py`（main 分支 2026-11-18 核实；ORPO 于 2026-02 移入 experimental）
