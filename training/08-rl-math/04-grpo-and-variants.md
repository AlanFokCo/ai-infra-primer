# 8.4 GRPO 与 LLM-RL 变体（GRPO and LLM-RL Variants）

> 本章从 PPO 目标出发推导 GRPO 及其 2025 年变体族：去 critic 的动机与组内归一化 advantage → 组基线的 $(G-1)/G$ 偏差完整推导与 leave-one-out 无偏版 → KL 项的两个位置与 k1/k2/k3 估计器谱系 → token 级 vs 序列级 importance ratio 的粒度差异 → Dr.GRPO 的两处归一化偏置分析 → DAPO 工业四件套 → GSPO 序列级 ratio → 变体对照总表收口 | 前置：8.3（PPO clip 目标、陈旧偏差分析）、8.1（baseline 无偏性定理）；R1 训练配方归 9.5、系统实现归 8.5 | 锚点：DeepSeekMath（[Shao et al., 2024](../references.md)，arXiv:2402.03300，[A]，GRPO 出处）、DeepSeek-R1（[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]）、Understanding R1-Zero（[Liu et al., 2025](../references.md)，arXiv:2503.20783，[A]，Dr.GRPO）、DAPO（[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]）、GSPO（[Zheng et al., 2025](../references.md)，arXiv:2507.18071，[A]）、Schulman KL 博客（教学参照）、TRL `grpo_trainer.py`（教学参照）

## 1. 从 PPO 到 GRPO：去掉 critic

8.2 结尾留下的困境是 GRPO 的直接动机：LLM 场景的 reward 只在序列末端给出（outcome reward），critic 却要逐 token 输出 $V(s_t)$——监督信号是一个标量、学习目标是整段前缀的函数，critic 在长 CoT 上很难训准；同时 critic 通常与 actor 同规模（DeepSeekMath 原文口径：another model of comparable size），显存与算力直接翻倍（四模型驻留的完整账归 8.5）。

GRPO（Group Relative Policy Optimization，[Shao et al., 2024](../references.md)，arXiv:2402.03300，[A]，其 §4.1）的回答：baseline 不需要学出来——对同一个 prompt 采一组回答，组内 reward 的均值就是 baseline 的蒙特卡洛估计。形式化：对每个问题 $q$，旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采 $G$ 条回答 $\lbrace o_i \rbrace_{i=1}^{G}$，奖励模型（或规则验证器）打分得 $\mathbf{r} = \lbrace r_1, \ldots, r_G \rbrace$，组内归一化给出 advantage：

$$
\hat{A}_{i,t} = \widetilde{r}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} \qquad (1)
$$

**符号**： $G$：组大小（全书约定： $G$ 专归 GRPO 组大小，不作并行度数）； $`\hat{A}_{i,t}`$ 对回答 $i$ 的**所有** token $t$ 取同值 $`\widetilde{r}_i`$——outcome supervision 下整条序列共享一个 advantage，token 级 credit assignment 被完全摊平。这是去 critic 的代价，也是它的简化：没有逐 token 的价值函数要训，也就没有价值误差沿序列传播。

目标函数（其式 (3)，本书换章内编号）：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\hspace{4mu} \lbrace o_i \rbrace_{i=1}^{G} \sim \pi_{\theta_{\mathrm{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\lbrace \min\left( r_{i,t}(\theta) \thinspace \hat{A}_{i,t},\hspace{4mu} \mathrm{clip}\left( r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right) - \beta \thinspace D_{\mathrm{KL}}\left( \pi_\theta \Vert \pi_{\mathrm{ref}} \right) \right\rbrace \right] \qquad (2)
$$

**符号**： $r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q, o_{i,\lt t}) / \pi_{\theta_{\mathrm{old}}}(o_{i,t}|q, o_{i,\lt t})$：逐 token importance ratio； $\varepsilon$：clip 半径； $\beta$：KL 系数； $\pi_{\mathrm{ref}}$：参考策略（通常取 SFT 模型，迭代式 GRPO 中每轮刷新为当前策略）。结构上与 8.3 式 (10) 逐件对应：min-clip 项原样保留，差别只有三处。

先把两个算法的组件结构并排画出来（组件 = 需要驻留或前向的模型/估计器）：

```
PPO-RLHF（四组件）                    GRPO（三组件）
┌──────────┐  rollout   ┌────────┐   ┌──────────┐  同 prompt 采 G 条  ┌────────┐
│ π_old    │ ─────────► │ reward │   │ π_old    │ ─────────────────► │ reward │
│ (actor)  │            │ model  │   │ (actor)  │                    │ model/ │
└──────────┘            └───┬────┘   └──────────┘                    │ 规则   │
┌──────────┐  V(s_t)   ┌────┴───┐   ┌──────────┐                    └───┬────┘
│ critic   │ ────────► │  GAE   │   │ π_ref    │ ──k3 KL──────────┐     │ r_1..r_G
│ V_ψ      │           │  Â_t   │   │ (冻结)   │                  ▼     ▼
└──────────┘           └───┬────┘   └──────────┘            Â_i=(r_i−mean)/std
┌──────────┐  KL 进 reward│                              无 critic、无 GAE
│ π_ref    │ ────────────┘
└──────────┘
```

差异一，baseline 的来源： $V_\psi(s_t)$ 换成组均值。critic 网络、价值损失、GAE 递推整组删除——这就是为什么 GRPO 的 memory 收益是「约一个 actor 的量级」而不是一个百分比（8.5 的账）。

差异二，KL 的位置。PPO 式 RLHF 把 KL 逐 token 折进奖励（DeepSeekMath 式 (2)，InstructGPT 口径）：

$$
r_t = r_\varphi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{\lt t})}{\pi_{\mathrm{ref}}(o_t|q, o_{\lt t})} \qquad (3)
$$

**符号**： $r_\varphi$：reward model。GRPO 改为把 KL 作为显式项挂在损失里（式 (2) 末项），原文理由是「避免把 $`\hat{A}_{i,t}`$ 的计算复杂化」——进 reward 的 KL 会参与 GAE/组基线的全部下游计算，进 loss 的 KL 与 advantage 解耦、只起锚定作用。两个位置的估计器选择在 §3 展开。

差异三，外层多一层组平均 $\frac{1}{G} \sum_i$ 与序列内均值 $\frac{1}{|o_i|} \sum_t$——后者正是 Dr.GRPO 与 DAPO 先后要动的归一化（§5、§6）。

原文为组均值 baseline 还给了一条动机性论证（其 §4.1.1）：reward model 本身就训在「同一问题的回答两两比较」的数据上，组内相对比较与 RM 的比较本性对齐——绝对打分有校准漂移，相对比较在同一 prompt 内自校准。这条论证同时也预告了组方法的适用边界：reward 必须能在同题多答之间可比，开放式生成（无验证器、RM 校准差）里组均值基线的噪声会直接进 advantage。

实证配置（其 §4.2，DeepSeekMath-RL 7B）： $G = 64$、 $\beta = 0.04$、策略 lr $10^{-6}$、最大生成长度 1024、训练批 1024、每个 exploration 阶段只更新一次（ $\mu = 1$ ）。迭代结构（其 Algorithm 1）：Iterative GRPO 把训练包成「外层刷新 $\pi_{\mathrm{ref}} \leftarrow \pi_\theta$、内层多步更新」的嵌套循环——KL 锚点随策略前进，锚定的是「当前的自己」而非「出发时的 SFT」，reward model 也随之用新采样数据续训（混入 10% 历史 replay）。这与一次后训练里锚定固定 SFT 模型的做法（9.2 的 RLHF 口径）是两种正则语义：前者防单步漂移、后者防累计漂移。

教学实现对照：TRL 的 `trl/trainer/grpo_trainer.py`（教学参照，2026-11-17 核实路径）把 advantage 写成 `advantages = rewards - mean_grouped_rewards` 再按 `scale_rewards` 配置除以 `std_rewards + 1e-4`——式 (1) 的直接落地，分母上的 $10^{-4}$ 是防全同组的数值护栏。

## 2. 组基线的偏差-方差分析

本节的问题是组基线的统计学性质：它既是方差缩减装置（baseline 的本职），又是偏差的潜在来源（含己均值与 std 归一化）。先把结论摆出来：含己均值的期望梯度是 $\nabla J$ 的 $(G-1)/G$ 倍——严格有偏但只是常数缩放；std 归一化引入的是逐问题的权重扭曲——真偏置；leave-one-out 版则精确无偏。三个结论分别推导如下。

式 (1) 的分子里， $\mathrm{mean}(\mathbf{r})$ 包含了 $r_i$ 自己——8.1 的 baseline 无偏性定理要求 baseline 与当前动作条件独立，组均值违反了这一条。偏差多大，可以精确算出来（本节为本书推导）。

**设定**（序列级简化）：固定 $q$，i.i.d. 采 $G$ 条 $o_1, \ldots, o_G \sim \pi_\theta(\cdot|q)$，reward $R_i = R(o_i)$；看新鲜 on-policy 点（内层迭代起步、ratio 为 1）处的策略梯度估计量：

$$
\hat{g} = \frac{1}{G} \sum_{i=1}^{G} \left( R_i - \overline{R} \right) \nabla_\theta \log \pi_\theta(o_i|q), \qquad \overline{R} = \frac{1}{G} \sum_{j=1}^{G} R_j \qquad (4)
$$

取期望。第一项直接是策略梯度： $`\mathbb{E}[\frac{1}{G}\sum_i R_i \nabla_\theta \log \pi_\theta(o_i)] = \nabla J_q`$。第二项把 $`\overline{R}`$ 按 $j = i$ 与 $j \neq i$ 拆开：

$$
\mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \overline{R} \thinspace \nabla_\theta \log \pi_\theta(o_i) \right] = \frac{1}{G^{2}} \sum_{i=1}^{G} \left( \underbrace{\mathbb{E}\left[ R_i \nabla_\theta \log \pi_\theta(o_i) \right]}_{\nabla J_q} + \sum_{j \neq i} \mathbb{E}[R_j] \cdot \underbrace{\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(o_i) \right]}_{= 0} \right) = \frac{1}{G} \nabla J_q \qquad (5)
$$

$j \neq i$ 项里 $o_j$ 与 $o_i$ 独立，score 函数的零均值性 $`\mathbb{E}_{o \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(o)] = 0`$ 把交叉项全部消掉； $j = i$ 项每个贡献一个 $\nabla J_q$。合并两项：

$$
\mathbb{E}[\hat{g}] = \nabla J_q - \frac{1}{G} \nabla J_q = \frac{G-1}{G} \thinspace \nabla J_q \qquad (6)
$$

**结论一：全组均值 baseline 严格有偏**，但偏差是一个与 $q$、 $\theta$ 无关的常数因子 $(G-1)/G$——方向不偏、幅度统一缩小， $G = 64$ 时约 1.6%。这类偏差被学习率直接吸收，实践中无害；它的价值在概念边界上：「baseline 无偏」要求的是 baseline 与动作**条件独立**，含己均值恰好踩线。

**leave-one-out 版精确无偏**： $b_i = \frac{1}{G-1} \sum_{j \neq i} R_j$ 与 $o_i$ 独立，于是

$$
\mathbb{E}\left[ b_i \thinspace \nabla_\theta \log \pi_\theta(o_i) \right] = \mathbb{E}[b_i] \cdot \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(o_i) \right] = 0 \qquad (7)
$$

$`E[\hat{g}_{\mathrm{LOO}}] = \nabla J_q`$ 精确成立（ $b_i$ 只依赖 $q$ 与其他样本，满足 8.1 的合法形态）。GRPO 选含己均值换实现上的单行简洁；RLOO（REINFORCE Leave-One-Out）一族算法选的正是式 (7) 的形态。

一个 $G = 4$ 的算例把式 (6) 走实。设二值 reward（对 1 错 0），当前策略对该题的通过率为 $p$。真梯度（log-导数形式）是「正确回答被提概率、错误回答被压概率」的净效应；含己均值版在 $G = 4$ 下把这股力统一缩到 $3/4$——每个样本的贡献里，有 $1/G$ 的份额被自己进入基线的那部分抵消掉了。leave-one-out 版逐样本用其余三条的均值作基线： $R_i = 1$ 时基线是其余样本的正确率， $R_i = 0$ 时同理，期望恰好抵消，式 (7) 的零恒等逐样本可见。

再记方差侧。组均值是 8.1 最优常数 baseline（ $\mathbb{E}[R|q]$ ）的样本估计，其自身的采样方差约为 $`\sigma_R^{2}/G`$—— $G$ 同时是方差旋钮与偏差旋钮： $G$ 越大，baseline 越稳、 $(G-1)/G$ 越接近 1； $G$ 太小时 baseline 噪声与 reward 噪声同阶，组归一化反而注入方差。DeepSeekMath 取 $G = 64$（其 §4.2）、DAPO 取 $G = 16$（其 §4.1），都是在这个旋钮上的取值。

**结论二：std 归一化是逐问题权重扭曲**。式 (1) 的分母使问题 $q$ 的整组梯度被乘上 $`1/\mathrm{std}_q`$：reward 接近全同（几乎全部对或几乎全部错、std 小而非零）的题被放大权重，难度适中（std 大）的题被稀释。这不是常数因子——逐问题变化的权重学习率吸收不掉。恰好全对/全错的组则是另一个极端：分子恒零、无梯度（数值上靠分母加小常数护栏），这是 §6 DAPO dynamic sampling 要剔除的对象。

## 3. KL 项的位置与 k1/k2/k3 估计器

式 (2) 与式 (3) 给出 KL 的两个落位，后果不同：进 reward 时 KL 参与 advantage 计算（组基线把 KL 项一并归一化，任务信号与锚定信号混在一个 $`\hat{A}`$ 里）；进 loss 时 advantage 是纯任务信号，KL 只负责「别把策略拉离 $\pi_{\mathrm{ref}}$ 太远」。DeepSeek-R1 沿用 GRPO 的写法（[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]，其 §2.2.1）：ratio 与 KL 都写成序列级简写形式，估计器与 DeepSeekMath 相同。

两个位置的实际后果可以各记一笔。进 reward 时，KL 惩罚被折进每个 token 的即时奖励，优势估计（GAE 或组基线）把「任务做得好不好」与「离参考有多远」学进同一个 $`\hat{A}`$—— $\beta$ 稍大，advantage 的符号就可能被 KL 项翻转；其好处是惩罚沿序列逐 token 塑形，位置感强。进 loss 时 advantage 是纯任务信号，KL 只出现在目标末端、梯度直接作用在 log 概率比上——位置感弱但语义干净，且估计器可以自由换成低方差形式（下文的 k3）。DeepSeekMath 的表述是后者「avoiding complicating the calculation of $`\hat{A}_{i,t}`$」（其 §4.1 原文口径）。

KL 期望本身也要估计：样本是逐 token 采的， $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})$ 没有闭式解。Schulman 2020 年的博客（[Schulman, 2020](../references.md)，教学参照）给出三个估计器的完整谱系，记 $x \sim q$、 $r = p(x)/q(x)$，估计 $`D_{\mathrm{KL}}(q \Vert p) = \mathbb{E}_q[-\log r]`$：

- **k1**： $`\hat{k}_1 = -\log r`$ ——定义即无偏，但逐样本值约半数为负而 KL 恒非负，方差高；
- **k2**： $`\hat{k}_2 = \frac{1}{2}(\log r)^{2}`$ ——恒正、方差低，但有偏；偏差小的机制是其期望为 f-散度（ $f = (\log \cdot)^{2}/2$ ），而一切可微 f-散度在 $p \approx q$ 处二阶展开同为 $\frac{f''(1)}{2} \theta^{\top} F \theta$（ $F$ 为 Fisher 信息矩阵），k2 与 KL 的 $f''(1)$ 同为 1——近端时偏差是二阶小量；
- **k3**： $`\hat{k}_3 = (r - 1) - \log r`$ ——对 k1 加控制变量 $r - 1$（零期望： $`\mathbb{E}_q[r] = \int q \cdot p/q = 1`$ ），与 k1 负相关部分相消； $\log x \le x - 1$ 保证恒正。无偏性一行证完：

$$
\mathbb{E}_{x \sim q}\left[ \hat{k}_3 \right] = \mathbb{E}_q\left[ \frac{p(x)}{q(x)} \right] - 1 - \mathbb{E}_q\left[ \log \frac{p(x)}{q(x)} \right] = 1 - 1 + D_{\mathrm{KL}}(q \Vert p) \qquad (8)
$$

博客的数值对照（ $q = \mathcal{N}(0,1)$、 $p = \mathcal{N}(\mu, 1)$ ）：真 KL = 0.005 时 k1 的标准差达真值的 20 倍、k3 仅 1.42 倍、k2 偏差仅 0.2%；真 KL = 0.5 时 k2 偏差涨到 50%、k3 仍无偏且标准差低于 k2。k3 严格占优——DeepSeekMath 式 (4) 选用的正是它：

$$
\widehat{D}_{\mathrm{KL}}\left( \pi_\theta \Vert \pi_{\mathrm{ref}} \right)_{i,t} = \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,\lt t})}{\pi_\theta(o_{i,t}|q, o_{i,\lt t})} - \log \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,\lt t})}{\pi_\theta(o_{i,t}|q, o_{i,\lt t})} - 1 \qquad (9)
$$

方向注意：样本来自 $\pi_{\mathrm{old}}$（内层首轮即 $\pi_\theta$ ），式 (9) 里的比率是 $\pi_{\mathrm{ref}} / \pi_\theta$——对应前向 KL $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})$，「当前策略背离参考多远」。实现注记：8.3 §8 的 CleanRL 诊断位用的是同一对估计器（k1 作 `old_approx_kl`、k3 作 `approx_kl`）——k1/k3 在 PPO 里做监控，在 GRPO 里进损失。

## 4. ratio 的粒度：token 级 vs 序列级

式 (2) 的 $r_{i,t}$ 是逐 token 的：每个生成位置一个 importance weight。两个事实把这个粒度选择推成问题。

其一，8.3 §2 的方差账在序列上相乘：整条回答的似然比是 $\prod_t r_{i,t}$，每步的小偏差连乘累积，长 CoT（数千 token）下序列级失配可以极端，而 clip 是按 token 截的——截断决策与被校正的对象不在同一粒度。

其二，IS 的成立条件是**多样本平均**： $`\mathbb{E}_{\mathrm{beh}}[w \cdot f]`$ 靠多个样本上的权重平均逼近目标期望；token 级 ratio 在每个位置上只有一个样本（ $o_{i,t}$ 是 $\pi_{\mathrm{old}}(\cdot|q, o_{i,\lt t})$ 的一次采样），weight 起不到分布校正作用，退化成高方差噪声源，且噪声沿序列累积、被 clip 的开关效应进一步放大。这是 GSPO 论文对 GRPO 目标「ill-posed」的指控原文（[Zheng et al., 2025](../references.md)，arXiv:2507.18071，[A]，其 §3），§7 展开其修法。

一个算术例子把粒度的差异坐实。设每 token 的新旧概率比只在 0.95 与 1.05 之间交替（温和漂移），100 token 的回答的序列连乘比为 $`(1.05 \times 0.95)^{50} = 0.9975^{50} \approx 0.882`$——单 token 最大偏离 5%，序列级已漂移 12%；1000 token 时约为 $0.9975^{500} \approx 0.29$。也就是说：逐 token 的「小更新」在序列粒度上被连乘成「大漂移」，而 clip 发生在 token 粒度——长 CoT 场景里，负责校正的机制和需要校正的对象之间差着序列长度个数量级（练习 4 把这条算完）。

## 5. Dr.GRPO：两处归一化的偏置分析

Understanding R1-Zero（[Liu et al., 2025](../references.md)，arXiv:2503.20783，[A]）把式 (2) 拆出两处逐样本权重扭曲（其 §3.1 与 Figure 5）：

**response 级长度偏置**（来源： $\frac{1}{|o_i|}$ 归一化）。同组内每个 token 的梯度贡献被回答长度除： $`\hat{A}_{i,t} \gt 0`$（答对）时短回答的逐 token 梯度更大，正确回答被推向变短； $`\hat{A}_{i,t} \lt 0`$（答错）时长回答的惩罚被摊薄，错误回答被推向变长。论文据此重审 R1-Zero 式训练的标志性现象：「错误回答越来越长」至少部分是这个归一化的优化伪影，不能全记给「长 CoT 涌现」。

**question 级难度偏置**（来源：std 归一化）。即 §2 结论二： $`1/\mathrm{std}_q`$ 让接近全对/全错的题获得爆炸权重。原文的对照观察：advantage normalization 在经典 RL 里跨整个 batch 做，GRPO 把它下沉到 question 粒度才引入这个扭曲。

两处扭曲可以合并成一行读法：GRPO 的有效 advantage（其 Figure 5 的记号）是 $`a_{i,t} = \frac{1}{|o_i|} \cdot \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}`$——「无偏版本」 $`\widetilde{A}_{i,t} = r_i - \mathrm{mean}(\mathbf{r})`$ 被两个逐样本权重 $1/|o_i|$ 与 $1/\mathrm{std}(\mathbf{r})$ 重新加权；偏置的全部内容就是这两个权重随回答长度与题目难度的变化。

**修正（Dr.GRPO = "GRPO Done Right"，其 §3.2）**：删掉这两项——advantage 只减组均值不除 std（ $`\widetilde{A}_{i,t} = r_i - \mathrm{mean}(\mathbf{r})`$ ），损失聚合把逐序列均值换成全局常数（实现上把 `masked_mean` 的除数从 `mask.sum()` 换成 `MAX_TOKENS` 一类的固定值，其论文的代码对照）。修正后的目标形式上回到「蒙特卡洛 return + 组均值 baseline 的 PPO」。一个口径对齐：其附录 A 的无偏性论证走 RLOO 等价链—— $`\frac{G}{G-1} \cdot \widetilde{A}_{i,t} = \hat{A}_{i,t}^{\mathrm{RLOO}}`$（展开约简后即 leave-one-out 形式），常数因子 $G/(G-1)$ 可被学习率吸收而不改 RL 动力学（原文口径）。这与 §2 的推导是同一件事的两个侧面：式 (6) 的 $(G-1)/G$ 因子与这里的 $G/(G-1)$ 互为倒数，leave-one-out 版（式 (7)）即 RLOO、精确无偏。

实证口径（其 §3.2）：Dr.GRPO 阻止错误回答长度随训练持续膨胀、token 效率提升；极简配方（Qwen2.5-Math-7B 基座 + MATH level 3-5 题 + Qwen-Math 模板）在 AIME 2024 上达 43.3%，27 小时 @ 8×A100。同一篇论文还点名了一个实现层事实：主流开源 PPO 实现把损失按 response 长度归一（per-response：OpenRLHF 口径；per-batch：TRL/veRL 口径），同样的长度偏置在 GRPO 出现之前就存在于这些 PPO 实现里——「公式无偏、实现有偏」。

## 6. DAPO：工业四件套

DAPO（[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]）是在 veRL 上构建并完全开源的大规模 RL 系统：Qwen2.5-32B 基座 → AIME 2024 得 50 分，超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分且只用 50% 训练步数（其摘要与 §1 口径）。算法层是 GRPO 加四件工业修补。其完整目标（其式 (10)）在 GRPO 骨架上同时改了 clip 半径、聚合方式与采样约束，迭代结构沿用「采样 → 组内 advantage → 内层 $\mu$ 次更新」的 GRPO 循环（其 Algorithm 1）。逐条如下（均对照原文 §3 核实；其目标式里 advantage 仍用式 (1) 的 std 归一化版，未采纳 §5 的修正）：

**(1) Clip-Higher（解耦截断）**。动机是 entropy 塌缩：朴素 PPO/GRPO 下策略熵随训练快速下降、组内样本趋同（其 Figure 2(b) 口径）。机制在 clip 的不对称算术上： $`\hat{A} \gt 0`$ 时上界把概率提升钉在 $\pi_{\mathrm{old}} \cdot (1+\varepsilon)$——原文算例： $\pi_{\mathrm{old}} = 0.01$ 的「探索」token 最多提到 0.012，而 $\pi_{\mathrm{old}} = 0.9$ 的「利用」token 可提到 1.08；其统计还显示被上截断的 token 平均概率低于 0.2——上界 clip 约束的恰是低概率探索。修法是上下界解耦（其式 (10) 的高亮项），实验取 $\varepsilon_{\mathrm{low}} = 0.2$、 $\varepsilon_{\mathrm{high}} = 0.28$（其 §4.1）；下界不加宽的理由：放宽 $\varepsilon_{\mathrm{low}}$ 会把低概率 token 压向 0、采样空间塌缩（原文口径）。失效模式对照：上下界同步加宽（如 $\varepsilon = 0.3$）会让负 advantage 样本被过度压低——低概率 token 直接趋零、回答多样性反而下降；只留默认对称 0.2 则回到熵塌缩路径。clip-higher 的全部内容就是承认「提高好 token」与「压低坏 token」是两种不同的操作，该用两个半径。

**(2) Dynamic Sampling（动态采样）**。全对/全错的组 advantage 恒零（§2），无梯度贡献；随训练推进全对组占比持续上升（其 Figure 3(b)），有效 prompt 数缩水、批梯度方差上升。修法：过采样加过滤，目标批只保留 $0 \lt |\lbrace o_i : o_i \text{ 与答案等价} \rbrace| \lt G$ 的 prompt，持续采样直到填满（其式 (11) 的约束条件）。效率注记：同步 rollout 下生成时间本由长尾样本决定，过滤不必拖慢吞吐（原文口径）。失效模式对照：不过滤时，后期训练的批梯度被越来越多的零 advantage 组稀释——批内有效样本比例下降，等价于 lr 被随机下调且方差上升；过滤的代价是采样量变动态（难题通过率低的阶段要过采样更多），预算核算需按「有效 prompt 数」而非「采样 prompt 数」记。

**(3) Token-level policy gradient loss（token 级损失聚合）**。把式 (2) 的「先按序列取均值、再按组取均值」改成按全批 token 总数归一（其式 (12)）：

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\hspace{3mu} \lbrace o_i \rbrace_{i=1}^{G} \sim \pi_{\theta_{\mathrm{old}}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\hspace{4mu} \mathrm{clip}\left( r_{i,t}(\theta), 1-\varepsilon_{\mathrm{low}}, 1+\varepsilon_{\mathrm{high}} \right) \hat{A}_{i,t} \right) \right] \qquad (10)
$$

长序列内每个 token 的权重不再被 $\frac{1}{|o_i|}$ 稀释。动机（其 §3.3）：长 CoT 场景下，sample 级聚合让长样本里的高质量推理模式学不动、低质量模式（乱码、重复用词）罚不动——后者还会反向推高熵与长度。与 §5 对照：DAPO 把聚合改成「越长贡献越大」的 token 级，Dr.GRPO 改成常数归一——两者都反对 per-response 均值，方向不同（§8 表）。失效模式对照：sample 级聚合下，一条 8k token 的乱码回答与一条 500 token 的正常回答对损失的贡献相同——乱码 token 的单条惩罚被稀释 16 倍；token 级聚合把惩罚强度恢复为逐 token 恒定，「重复刷屏」不再划算。

**(4) Overlong 处理**。默认给截断样本惩罚性 reward 会把「推理过程正确但超长」的样本误判为负例、注入奖励噪声。两招：overlong filtering（截断样本的损失直接 mask）与 soft overlong punishment——长度进入缓冲区间后线性递增惩罚（其式 (13)）：

$$
R_{\mathrm{length}}(y) = \begin{cases} 0, & |y| \le L_{\max} - L_{\mathrm{cache}} \\ \dfrac{(L_{\max} - L_{\mathrm{cache}}) - |y|}{L_{\mathrm{cache}}}, & L_{\max} - L_{\mathrm{cache}} \lt |y| \le L_{\max} \\ -1, & L_{\max} \lt |y| \end{cases} \qquad (11)
$$

**符号**： $L_{\max}$：最大生成长度； $L_{\mathrm{cache}}$：缓冲区间宽度。惩罚加在规则正确性 reward 之上，把「别超长」做成连续信号而不是二值截断。配套配置（其 §4.1 与 §4.2）：prompt 批 512、 $G = 16$、minibatch 512（每个 rollout step 16 次梯度更新）、AdamW lr $10^{-6}$ 恒定加 20 步线性 warmup；软惩罚取 $L_{\max} - L_{\mathrm{cache}} = 16384$、 $L_{\mathrm{cache}} = 4096$（即 $L_{\max} = 20480$）；AIME 评估报 avg@32（temperature 1.0、top-p 0.7）。消融口径：同一 Qwen2.5-32B 基座上 vanilla GRPO 只到 30 分，四件套叠加到 50 分（其 Table 1 与 §4.2）。失效模式对照：硬截断加负 reward 时，模型学到的是「接近上限就恐慌」而非「压缩推理」——长难题的推理过程被惩罚信号污染（原文的 reward noise 口径）；软惩罚把压力集中在缓冲区， $L_{\mathrm{cache}}$ 宽度是「早预警」与「省预算」之间的旋钮。

## 7. GSPO：把 ratio 提到序列级

GSPO（[Zheng et al., 2025](../references.md)，arXiv:2507.18071，[A]）把 §4 的指控落成算法。先登记该论文的实证动机：token 级 IS weight 的噪声在长序列上累积并被 clip 放大，论文报告了由此导致的**不可逆** model collapse（MoE 模型尤其敏感；回退 checkpoint、重调超参均无法恢复——其 §3 口径）。原则陈述值得照引：优化目标的单位应与奖励的单位匹配——reward 发给整条序列，off-policy 校正就该在序列级做。

序列级 ratio 定义为长度归一化的几何平均（其式 (7)；原文记 prompt 为 $x$、回答为 $y$，本章统一为 $q$、 $o$）：

$$
s_i(\theta) = \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)} \right)^{\frac{1}{|o_i|}} = \exp\left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_\theta(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q, o_{i,\lt t})} \right) \qquad (12)
$$

目标是同形的 min-clip（其式 (5)）： $`\mathcal{J}_{\mathrm{GSPO}} = \mathbb{E}[ \frac{1}{G} \sum_i \min( s_i(\theta) \hat{A}_i,\ \mathrm{clip}(s_i(\theta), 1 \pm \varepsilon) \hat{A}_i ) ]`$——clip 施加在整条回答上：过 off-policy 的回答整条退出梯度估计，而不是逐 token 截断。长度归一化（ $\frac{1}{|o_i|}$ 次幂）把不同长度回答的 ratio 压回统一数值范围，clip 半径才有共同语义（原文口径：否则少数 token 的似然变化就能让序列 ratio 剧烈波动）。也正因 ratio 定义不同，GSPO 与 GRPO 的 clip 半径差一个量级：该论文实验取左界 $3 \times 10^{-4}$、右界 $4 \times 10^{-4}$，对照组 GRPO 调参后为 0.2 / 0.27（其 §5.1）。

梯度对照把粒度差异写到式子上（其 §4.2，略去 clip）：

$$
\nabla_\theta \mathcal{J}_{\mathrm{GSPO}} = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} s_i(\theta) \thinspace \hat{A}_i \cdot \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t}|q, o_{i,\lt t}) \right] \qquad (13)
$$

$$
\nabla_\theta \mathcal{J}_{\mathrm{GRPO}} = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \hat{A}_i \cdot \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \frac{\pi_\theta(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q, o_{i,\lt t})} \thinspace \nabla_\theta \log \pi_\theta(o_{i,t}|q, o_{i,\lt t}) \right] \qquad (14)
$$

差别只在 token 梯度的权重：GRPO 给每个 token 乘各自的逐 token ratio（ $`\hat{A}_i \gt 0`$ 时在 $(0, 1+\varepsilon]$、 $`\hat{A}_i \lt 0`$ 时在 $[1-\varepsilon, \infty)$ 内不等），GSPO 给全序列同一个序列级权重 $s_i(\theta)$——逐 token 权重的不相等被整个移除。clip 半径差一个量级的机制可以算出来：对策略的同一处扰动，逐 token ratio 的连乘把各位置扰动**相加在指数上**（ $\prod_t r_t = \exp(\sum_t \log r_t)$ ，扰动按 $O(T)$ 累积），而 $s_i$ 的几何平均把同一扰动摊回 $O(1)$（指数上除以 $|o_i|$ ）——所以 GRPO 的 $\varepsilon$ 必须容忍「 $T$ 步扰动累进」的序列漂移，GSPO 的 $\varepsilon$ 只需约束「平均每 token 漂移」。0.2 对 3e-4 的数量级差（约 600 倍）与典型 CoT 长度（数百到数千 token）同阶——这不是调参巧合，是 ratio 定义换单位后的必然换算（练习 4）。

实证口径（其 §5.1）：Qwen3-30B-A3B-Base 冷启动模型上 GSPO 训练全程稳定、同等算力下训练精度与基准表现优于 GRPO；MoE 的 RL 训练不再需要 GRPO 所依赖的 Routing Replay 策略；GSPO 已用于 Qwen3 系列模型的 RL 训练。MoE 场景的失效链条在原文里有完整记录：GRPO 的逐 token ratio 在专家路由的稀疏激活下波动更大，clip 的开关效应与路由抖动互相放大，最终触发 collapse；序列级 ratio 把路由层面的逐 token 抖动平均掉，Routing Replay（固定专家分配的重放技巧）因此成为多余。

变体 GSPO-token（其 §4.3）服务需要逐 token 差异化 advantage 的场景（如多轮 RL）： $`s_{i,t}(\theta) = \mathrm{sg}[s_i(\theta)] \cdot \pi_\theta(o_{i,t}|\cdot) / \mathrm{sg}[\pi_\theta(o_{i,t}|\cdot)]`$（ $\mathrm{sg}$ 为 stop-gradient）——数值上恒等于 $s_i(\theta)$、梯度形态恢复逐 token 分解；同 response 内 advantage 取同值时与 GSPO 在目标、clip 条件与梯度上严格一致（原文结论）。实现含义：同一套逐 token 损失管线只需替换 ratio 的构造即可在两种粒度间切换，工程改动量集中在 ratio 的 detach 边界上。

## 8. 变体对照总表

| 算法 | baseline 来源 | ratio 粒度 | clip 形式 | 长度处理 | KL 位置与估计 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| PPO（LLM 形态） | 学习的 critic + GAE | token | 对称 $\varepsilon$（0.2） | per-response 均值（实现层惯例） | 进 reward（式 (3)），逐 token log-ratio | arXiv:1707.06347 [B] |
| GRPO | 组内均值 + std 归一 | token | 对称 $\varepsilon$ | per-response 均值 $1/\|o_i\|$ | 进 loss，k3； $\beta$=0.04（DeepSeekMath 口径） | arXiv:2402.03300 [A] |
| Dr.GRPO | 组内均值（去 std） | token | 对称 $\varepsilon$ | 常数归一（MAX_TOKENS 口径） | 同 GRPO | arXiv:2503.20783 [A] |
| DAPO | 组内均值 + std 归一 | token | 解耦 $`\varepsilon_{\mathrm{low}}`$ / $`\varepsilon_{\mathrm{high}}`$（0.2/0.28） | 全批 token 归一 $1/\sum_i \|o_i\|$ | 移除 KL（其 §2.3） | arXiv:2503.14476 [A] |
| GSPO | 组内均值 + std 归一 | sequence（几何平均） | 序列级 $\varepsilon$（3e-4/4e-4 口径） | ratio 自带长度归一 | 原文略去 KL（for brevity） | arXiv:2507.18071 [A] |
| GSPO-token | 组内均值 + std 归一 | token（sg 恒等序列级权重） | token 级 | 同 GSPO | 原文略去 KL（for brevity） | arXiv:2507.18071 [A] |

表注：「per-response 均值」一行指损失聚合方式，Dr.GRPO §3.1 对开源 PPO 实现的批判同样适用；KL 列的「同 GRPO」指沿 DeepSeekMath 的 loss 内 k3 形式；DAPO 的移除是显式设计决定（长 CoT 下策略本就大幅偏离初始模型，其 §2.3 口径），GSPO 的略去是行文简化而非设计主张（其 §2 口径）；各开源实现另有取舍（8.5）。

按行读这张表可以得到三条选型逻辑。其一，**baseline 来源**决定组件账：critic 一列从「学习」换成「组内统计」省掉的是一个与 actor 同规模的模型及其训练循环（8.5 的显存账），换来的是式 (6) 的 $(G-1)/G$ 常数偏差与 §2 的方差旋钮。其二，**ratio 粒度**决定校正的可信范围：token 级细但方差沿长度累积，序列级稳但把整条的取舍压成一个开关——GSPO-token 的 sg 技巧说明两者可以在同一实现里互换外壳。其三，**长度处理**是 2025 年变体的共同战场：Dr.GRPO 的常数归一与 DAPO 的 token 级聚合同为「反对 per-response 均值」的两种方向，GSPO 则把长度吸收进 ratio 定义本身；目前没有任何一家给出跨任务的一致性消融，选型只能按失效模式对号入座——错误回答膨胀选 Dr.GRPO 口径，熵塌缩与全对组稀释选 DAPO，长序列 MoE 训练不稳选 GSPO（各论文的实证场景口径，本书不给出超出来源的排序）。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| GRPO 目标 | 去 critic：组内 $(r_i - \mathrm{mean})/\mathrm{std}$ 作 advantage；KL 进 loss 用 k3 | 式 (1)(2)(9)，arXiv:2402.03300 |
| 组基线偏差 | 含己均值 ⇒ $E[\hat{g}] = \frac{G-1}{G}\nabla J$（常数因子，lr 可吸收）；leave-one-out 精确无偏；std 归一化 = 逐问题难度权重 | 式 (4)-(7)，本书推导 |
| k1/k2/k3 | k3 = $(r-1) - \log r$：无偏 + 恒正 + 低方差（控制变量 + Bregman 几何） | 式 (8)，Schulman 2020 博客 |
| Dr.GRPO | 删 $1/\|o_i\|$ 与 std 两处归一化；长度偏置（答错变长）与难度偏置的实证修正 | arXiv:2503.20783 §3 |
| DAPO | clip-higher（0.2/0.28）、dynamic sampling、token 级聚合、overlong 软惩罚 | 式 (10)(11)，arXiv:2503.14476 §3 |
| GSPO | 序列级几何平均 ratio；token 等权 vs GRPO 逐 token 加权；clip 半径差一个量级；MoE 免 Routing Replay | 式 (12)-(14)，arXiv:2507.18071 |

## 练习

1.（推导）含己均值的缩放形式：把式 (4) 的估计量改为 $\hat{g}' = \frac{1}{G-1} \sum_i (R_i - \overline{R}) \nabla_\theta \log \pi_\theta(o_i)$，求 $\mathbb{E}[\hat{g}']$；再设 $G = 2$，把含己均值版展开成「成对偏好差」的形式并说明它与 3.2 的 Bradley-Terry 差分结构的相似性。
2.（计算）k3 的方差优势： $q = \mathcal{N}(0,1)$、 $p = \mathcal{N}(0.1, 1)$，用 $10^{6}$ 个样本蒙特卡洛估计 k1/k2/k3 的均值与标准差，对照 Schulman 博客的表（std/真值：k1 ≈ 20、k3 ≈ 1.42）；再把均值差拉到 1 重算，解释 k2 的偏差为何随真 KL 增大而上升（提示：f-散度的二阶等价只在近端成立）。
3.（推导）DAPO clip-higher 的算术： $`\hat{A} \gt 0`$、 $\varepsilon_{\mathrm{low}} = 0.2$、 $\varepsilon_{\mathrm{high}} = 0.28$，分别计算 $\pi_{\mathrm{old}}$ = 0.01 / 0.5 / 0.9 三个 token 的概率提升上限；对称 $\varepsilon = 0.2$ 下各是多少？由此定量解释「上界 clip 主要约束探索」。
4.（计算）GSPO 的长度效应：两条回答逐 token ratio 都在 0.95 与 1.05 间交替，一条长 4 token、一条长 100 token——分别算序列连乘 ratio $\prod_t r_t$ 与式 (12) 的 $s_i(\theta)$，说明长度归一化如何把 clip 半径的语义从 $O(\varepsilon \cdot T)$ 压回 $O(\varepsilon)$。

## 参考

- [Shao et al., 2024](../references.md)：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models，arXiv:2402.03300，[A]（key: deepseekmath）——GRPO 目标式 (2)、advantage 式 (1)、KL 进 loss 与 k3 估计器式 (9)、RL 配置（ $G$=64、 $\beta$=0.04）
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning，arXiv:2501.12948，[A]（key: deepseek-r1）——GRPO 的序列级简写形式（其 §2.2.1）
- [Liu et al., 2025](../references.md)：Understanding R1-Zero-Like Training: A Critical Perspective，arXiv:2503.20783，[A]（key: dr-grpo）——std/长度归一化偏置分析、Dr.GRPO 修正、极简配方 AIME 2024 43.3%（27h @ 8×A100）、开源 PPO 实现的长度归一化批判
- [Yu et al., 2025](../references.md)：DAPO: An Open-Source LLM Reinforcement Learning System at Scale，arXiv:2503.14476，[A]（key: dapo）——clip-higher（ $\varepsilon_{\mathrm{low}}$=0.2 / $\varepsilon_{\mathrm{high}}$=0.28）、dynamic sampling、token 级聚合式 (10)、overlong 软惩罚式 (11)、AIME 2024 50 分口径
- [Zheng et al., 2025](../references.md)：Group Sequence Policy Optimization，arXiv:2507.18071，[A]（key: gspo）——序列级 ratio 式 (12)、梯度对照式 (13)(14)、clip 半径 3e-4/4e-4、GSPO-token、MoE 免 Routing Replay
- [Schulman, 2020](../references.md)：Approximating KL Divergence（教学参照，key: schulman-kl）——k1/k2/k3 估计器谱系与数值对照表
- [HuggingFace, huggingface/trl repo](../references.md)（key: trl-repo，教学参照）——`trl/trainer/grpo_trainer.py` 的 advantage 计算与 `scale_rewards` 配置（2026-11-17 核实）
