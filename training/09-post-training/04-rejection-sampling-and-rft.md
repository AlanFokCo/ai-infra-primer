# 9.4 拒绝采样与 RFT：采样-筛选-回灌（Rejection Sampling and RFT）

> 本章讲「当前模型采 $n$ 个 → 验证器或 RM 打分 → 选 best → SFT 回灌」这条循环。先给 RFT 的流程定义，再把筛选后的数据写成截断-再归一化的分布（本章的关键形式化），软加权版本正好对接 9.3 的闭式解。随后划清 RFT 与 RL 的分界（无梯度回传、只改数据分布），讲 STaR 的 self-improvement 循环——原文诊断的问题是「解不出新题 ⇒ 无训练信号 ⇒ 停滞」，修正是 rationalization——以及 Llama-3 的 RFT 工程口径。最后从 verifiable 任务过渡到 9.5 的 RLVR，并给出在线 / 离线分界判据表（3.2 §5 定义的兑现） | 前置：9.1（SFT）、9.2（RM 与 RLHF 管线）、9.3（DPO 与闭式解式 (3)）、3.2（在线 / 离线分界定义）、8.3-8.4（PPO / GRPO） | 锚点：STaR（[Zelikman et al., 2022](../references.md)，arXiv:2203.14465，[B]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、Christiano et al.（[B]，偏好 RL 源头）、huggingface/trl（教学参照 + 官方实现，main 分支 2026-11-18 核实）

## 1. RFT 的流程定义

拒绝采样微调（rejection sampling fine-tuning, RFT）的单轮流程分四步：

1. **采样**：对每个 prompt $x$，从当前策略 $\pi_\theta(\cdot \mid x)$ 独立采 $n$ 条回答 $y_1, \ldots, y_n$；
2. **打分**：验证器（verifier，规则判对错）或 reward model 给每条回答打标量分 $s_i$；
3. **筛选**：按阈值或 top-$`k`$ 保留子集 $D^{+}$（极端情形 best-of-$`n`$，只留 $\arg\max_i s_i$——Llama-3 的 RS 取法，见 §5）；
4. **回灌**：把 $D^{+}$ 当示范数据跑标准 SFT（9.1 的 CE 目标），得到下一轮策略。

```
                ┌──────────────── 一轮 RFT ────────────────┐
                │                                          │
  π_θ(t)        │  采 n 条          打分           筛选      │  SFT 回灌
 ──────────────►│ y_1..y_n ~π_θ  ──► s_i=V(x,y_i) ──► D⁺ ──┼──► π_θ(t+1)
   (推理栈)      │                 (规则 / RM)    top-k 阈值 │   (训练栈)
                └──────────────────────────────────────────┘
        数据在采样后冻结；训练阶段不再访问 π_θ 的采样分布
```

两个结构性事实先从这张图里读出。第一，训练阶段（第 4 步）的输入是**冻结数据集** $D^{+}$，与 9.1 的 SFT 没有任何形式差别——RFT 的全部特异性都在数据生产侧。第二，跨轮看， $D^{+}$ 的分布随策略漂移（第 $t+1$ 轮采自 $\pi_\theta^{(t+1)}$），所以 RFT 是「离线目标 + 滚动数据源」的复合体；这条性质在 §3 与 §7 分别展开。TRL 稳定分支没有专用 RFT trainer（main 分支 trainer 目录的主要 trainer 族：dpo / grpo / kto / sft / reward / rloo / distillation，2026-11-18 核实）——RFT 由「推理栈采样 + 筛选脚本 + `sft_trainer.py`」拼装而成，这正是它「数据侧算法」身份的工程证据。

## 2. 形式化：截断-再归一化的采样分布

把第 1-3 步合并看成一个分布变换。固定 $x$，记保留事件 $A(x) = \lbrace y : s(x, y) \ge \tau(x) \rbrace$（top-$`k`$ 口径下 $\tau(x)$ 为第 $k$ 大得分的随机阈值），则筛选后的数据分布是当前策略的**截断再归一化**：

$$
\hat{\pi}(y \mid x) = \frac{\pi_\theta(y \mid x) \thinspace \mathbb{1}\left[ y \in A(x) \right]}{\Pr_{y' \sim \pi_\theta(\cdot \mid x)}\left[ y' \in A(x) \right]} = \frac{\pi_\theta(y \mid x) \thinspace \mathbb{1}\left[ y \in A(x) \right]}{p_A(x)} \qquad (1)
$$

**符号**： $\mathbb{1}[\cdot]$：示性函数； $p_A(x)$：保留质量（分母是截断分布的归一化常数）。best-of-$`n`$ 是 $k = 1$ 的特例，即 $\tau(x)$ 取 $n$ 个样本得分的最大值——阈值本身是随机的，但条件化在已采样本上之后形式不变。

第 4 步的 SFT 是对 $\hat{\pi}$ 做极大似然：

$$
\pi^{+} = \arg\max_{\pi}\hspace{4mu} \mathbb{E}_{x \sim D}\hspace{4mu} \mathbb{E}_{y \sim \hat{\pi}(\cdot \mid x)}\left[ \log \pi(y \mid x) \right] = \arg\min_{\pi}\hspace{4mu} \mathbb{E}_{x \sim D}\left[ D_{\mathrm{KL}}\left( \hat{\pi}(\cdot \mid x) \Vert \pi(\cdot \mid x) \right) \right] \qquad (2)
$$

（第二等号： $\hat{\pi}$ 固定， $`-\mathbb{E}_{\hat{\pi}}[\log \pi] = D_{\mathrm{KL}}(\hat{\pi} \Vert \pi) + H(\hat{\pi})`$，熵项与 $\pi$ 无关。）**所以一轮 RFT = 把新策略交叉熵投影到当前策略的截断分布上**。这个视角立刻给出三条可检验的推论：

- **支撑集只缩不涨**： $\hat{\pi}$ 的支撑 ⊆ $\pi_\theta$ 的支撑——RFT 不可能发明当前策略采不出来的行为，它只重排已有行为的概率质量。「探索」被整个外包给第 1 步的 $n$ 次采样。
- **截断比加权更硬**：式 (1) 对被丢弃区域直接归零。若把硬示性换成软权重 $w(x, y) \propto \exp(s(x, y) / \alpha)$（按得分指数倾斜保留概率），截断分布变为

$$
\widetilde{\pi}(y \mid x) \propto \pi_\theta(y \mid x) \exp\left( s(x, y) / \alpha \right) \qquad (3)
$$

这正是 9.3 式 (3) 的闭式最优解形态——以当前策略自身为 $\pi_{\mathrm{ref}}$、以 $s$ 为 reward、以 $\alpha$ 为 $\beta$ 的 KL 约束目标最优解。RFT 一族（含其软加权变体，文献中的 reward-weighted regression 形态）与 KL 约束 RL 目标由此对接：硬截断是 $\alpha \to 0^{+}$ 侧的极限近似（只留峰区），软加权是有限温度的显式形式。9.3 的推导链在这里被第二次消费——同一个指数倾斜结构，一处用来消掉 RM（DPO），一处用来解释 RFT 的数据倾斜。
- **$p_A(x)$ 是难度的仪表盘**：保留质量 $p_A(x)$ 低，意味着当前策略在 $x$ 上几乎采不出合格回答；对这类 $x$，截断分布集中在极少数偶然成功的样本上，SFT 信号由侥幸样本主导。这是 §4 停滞问题的形式化入口。

## 3. 与 RL 的分界：梯度回传的有无

RFT 与 RL 优化的是同一类东西（提高高分回答的概率），分界不在目标而在机制。三条判据逐条对账。

**梯度路径**。RL（PPO / GRPO，8.3 / 8.4）把标量得分 $s_i$ 折成 advantage，乘 $\nabla_\theta \log \pi_\theta(y_t \mid \cdot)$ 回传——**低分样本也贡献梯度**（负 advantage 往下压）。RFT 的第 4 步是纯 SFT： $-\log \pi_\theta(y^{+} \mid x)$ 只作用于保留样本，被丢弃的 $n-1$ 条回答在目标函数里根本不存在，它们的信息在筛选那一步就被示性函数抹掉了。STaR 原文把这个关系说得更具体：把「生成 rationale 再作答」看成潜变量模型 $p_M(y \mid x) = \sum_r p(r \mid x) \thinspace p(y \mid x, r)$，以答案正确性为示性 reward，则总期望奖励与策略梯度为

$$
J = \sum_{i}\hspace{4mu} \mathbb{E}_{(\hat{r}_i, \hat{y}_i) \sim p_M(\cdot \mid x_i)}\left[ \mathbb{1}\left( \hat{y}_i = y_i \right) \right], \qquad \nabla J = \sum_{i}\hspace{4mu} \mathbb{E}_{(\hat{r}_i, \hat{y}_i) \sim p_M}\left[ \mathbb{1}\left( \hat{y}_i = y_i \right) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i \mid x_i) \right] \qquad (4)
$$

**符号**： $p_M$：策略的 rationale-answer 联合分布； $y_i$：标准答案。式 (4) 的示性函数把答错样本的梯度整个丢弃——「这正是 STaR 的筛选步」（原文口径：the indicator function discards the gradient for all sampled rationales that do not lead to the correct answer: this is the filtering process）。换言之，RFT/STaR 是 REINFORCE 式目标的一个**截断采样近似**：用贪心解码降方差（代价是探索有偏），对同一批数据走多个梯度步（同若干 policy gradient 算法的做法）。原文明确定位为「近似」，不主张等价。

**数据分布的角色**。RFT 改变的是**数据分布**：把 SFT 的输入分布从人工语料换成当前策略的截断尾部，模型参数的全部更新仍由 CE 驱动。RL 改变的是**目标函数**：在采样分布上加 advantage 加权。前者把「学得对」外包给筛选器，后者把「学得对」编码进梯度符号。

**部署形态**。RFT 的训练栈与 SFT 完全同构（单模型、无 rollout 引擎、无参考模型），采样阶段可以是任意推理服务——8.5 的四模型驻留、权重同步、staleness 管理的整本账都不存在。代价是 §2 的支撑集上限与 §4 的停滞。

## 4. STaR 循环：停滞问题与 rationalization

STaR（[Zelikman et al., 2022](../references.md)，arXiv:2203.14465，[B]）是 RFT 在推理任务上的原型。它的数据集只有问题与答案 $D = \lbrace (x_i, y_i) \rbrace$，没有中间推理。做法是：用少量带 rationale 的 few-shot 例子（约 10 个， $P \ll D$ ） prompting 当前模型，对每道题生成 rationale + 答案；只保留答案正确的（rationale, answer）对；用保留集微调，再采下一轮。两个原文设计决定值得逐字登记。其一，**每轮从原始预训练模型重新微调**，而非在上一轮模型上续训——原文理由是避免过拟合（"once we collect a new dataset, we train from the original pre-trained model instead of continually training one model to avoid overfitting"）。其二，循环终止条件是性能 plateau。

**停滞的诊断（原文 §3.2 口径，逐字引用其逻辑）**：「模型只在它答对的样本上训练，所以当模型无法解出训练集中的新题时，改进就终止了——根本原因是算法从失败样本里拿不到任何训练信号。」（"improvement ends when the model fails to solve new problems in the training set. This is fundamentally due to the fact that the algorithm cannot obtain any training signal from failed examples."）用 §2 的语言复述：对难度超过当前策略的 $x$， $p_A(x) \approx 0$，截断分布为空，式 (2) 对这些 $x$ 零贡献——self-improvement 的信号供给被当前能力上限钳住了。

**rationalization 修正**：对解不出的题，把标准答案作为提示（hint）喂给模型，让它反向生成通往该答案的 rationale，再把**去掉 hint 的**（问题, rationale, 答案）加入训练集（原文 Figure 2 的 hint 模板与「as if the model had come up with the rationale without the hint」的拼接口径）。机制读法：hint 把搜索问题从「在整个 rationale 空间里找一条通往正确答案的链」降为「解释一条已知存在的链」——保留质量从 $p_A(x) \approx 0$ 被人为拉高，信号供给恢复。代价同样可以定位：rationalization 样本的 rationale 质量没有正确性背书（hint 保证结论对、不保证推理对），它是用信号量换信号质。

实证锚点（原文摘要口径）：CommonsenseQA 上 STaR（GPT-J 6B 基座）达 72.5%，比直接答案微调（direct fine-tuning，60.0%）高 12.5 个百分点，比 few-shot 基线高 35.9 个百分点，与 30 倍大的微调 GPT-3（73.0%）持平。

## 5. Llama-3 的 RFT 用法

Llama-3 把 RFT 做成后训练的标准组件之一（[Grattafiori et al., 2024](../references.md)，[A]，§4.1-4.2 口径；该论文的后训练整体结构是 SFT → RM → DPO 的六轮迭代，RS 产出的数据进 SFT 混合集）。逐条工程口径：

- **采样配置**：对人工标注收集的每个 prompt，从「最新 chat 策略」采 $K$ 条输出， $K$ 典型取 10-30；采样源通常是上一轮后训练的最佳 checkpoint，或某一能力维度的最佳 checkpoint——采样源的选择本身就是配方的一部分，不是默认当前策略。
- **筛选器**：用 reward model 选最佳候选（原文标注 consistent with Bai et al. 2022）——Llama-3 的 RS 是 RM 打分路线，不是规则验证器路线。
- **系统提示引导**：后期轮次在 RS 采样时注入 system prompt，把回答的语气、风格、格式导向各能力的期望形态（代码场景有专门例子：system prompt 引导出带注释、变量名更好的代码）。
- **推理栈**：RS 的吞吐用 PagedAttention 支撑（预定义最大输出长度以消除 swap-out、同 prompt 的 $K$ 条输出共享 KV cache 页），合计吞吐提升超过 2 倍。
- **质检**：RS 产出的数据仍有质量问题（论文举例代码块含 bug），且 RS 数据的判定比合成代码数据更难（原文：RS 回答通常混杂自然语言与代码，代码未必可执行）。因此对 RS 数据用 model-as-judge 过滤：早期 Llama 3 版本按代码正确性与代码风格双判据打 0/1 分，只保留满分 2 的样本。执行信号过滤归合成代码数据产线。这条过滤还有一个记录在案的反噬：满分标准过度剔除了难题 prompt 的样本，下游基准回退；对策是把最难一档编码数据的回答改写到能通过 judge 为止。质量与难度的平衡是筛出来的，不是过滤阈值一劳永逸定的。「采样-筛选」的筛选不止 RM 一层。
- **位置**：RS 数据与合成数据、少量人工精调数据共同构成 SFT 混合；每轮后训练重新收集偏好标注与 SFT 数据、从最新模型采样合成数据（§4.1.6 与 §4.2.2 口径）。

与 STaR 的对照读法：STaR 的筛选器是规则（答案对/错），Llama-3 的筛选器是 RM；STaR 每轮从基座重训，Llama-3 的轮次间是能力滚动累积的六轮 pipeline。两者共享同一个截断-回灌骨架，差异全在筛选器与迭代结构上。

## 6. 向 verifiable 任务的过渡

把 §2 的筛选器换成「数学答案判等对 / 代码过单测」这类规则验证器，截断分布式 (1) 的保留事件就变成 $A(x) = \lbrace y : \mathrm{verify}(y) = 1 \rbrace$——reward 是二值的、无 RM 可 hack、信号与真值同源。STaR 已经是这个形态（示性 reward）。沿着这条轴再往前走一步：不再丢弃失败样本，把二值 reward 通过策略梯度回传给全部采样（式 (4) 的完整形态而非其截断近似），就是 RLVR——9.5 的主题。RFT 与 RLVR 的分界由此清晰：同一筛选器，一个只准它决定数据分布（offline 回灌），一个让它进梯度（on-policy 回传）。

## 7. 在线 / 离线判据表

3.2 §5 的定义（判据落在「目标的期望从谁采样」而非「数据来自谁」）在本章兑现为策略训练方法的归格表：

| 方法 | 数据来源 | 目标期望取自 | 训练中是否采样 | 判据格 | 定义章 |
| --- | --- | --- | --- | --- | --- |
| SFT | 人工 / 蒸馏静态集 | 静态集 $D$ | 否 | offline | 9.1 |
| DPO | 静态偏好对 | 静态集 $D$ | 否 | offline | 9.3 |
| KTO / SimPO / ORPO | 静态（单边或成对） | 静态集 $D$ | 否 | offline | 9.3 |
| RFT（单轮） | 当前策略采样后冻结的筛选集 | 筛选集 $D^{+}$（目标内无从 $\pi_\theta$ 采样的期望） | 训练前采，训练中不采 | offline（跨轮数据分布随策略漂移） | 9.4 |
| 迭代 DPO（Llama-3 式） | 每轮用最新模型重采偏好对 | 当轮静态集 | 轮间采，轮内不采 | offline（轮内） | 9.3 |
| RLHF-PPO | 每轮 rollout 批 | $\pi_{\theta_{\mathrm{old}}}$ | 训练中持续采 | on-policy | 8.3 / 9.2 |
| GRPO | 当前策略同 prompt 组采样 | $\pi_{\theta_{\mathrm{old}}}$ | 训练中持续采 | on-policy | 8.4 |
| RLVR（GRPO 承载） | 当前策略采样 + 规则打分 | $\pi_{\theta_{\mathrm{old}}}$ | 训练中持续采 | on-policy | 9.5 |

三条判读规则。其一，**「数据来自当前策略」不改变 offline 判定**：RFT 与迭代 DPO 的数据是热采的，但目标函数的期望号下是冻结数据集——按 3.2 §5 的判据落 offline；它们的 on-policy 性体现在**轮间**的数据刷新节奏，不在目标内。其二，**系统形态由判据格决定**：offline 格的方法一台纯训练栈即可运行（DPO 对 PPO 的工程折价来源，3.2 §5 已立）；on-policy 格必须养 rollout 基础设施（推理引擎、权重同步、staleness 管理，账归 8.5）。其三，**选择是分布匹配问题**：数据静态可得（存量偏好集）选 offline 族；奖励可程序化验证且要压榨失败样本的梯度信号，选 on-policy 族（9.5）；采样便宜而验证贵（RM 打分）时 RFT 的截断回灌是中间档——它买了「数据分布随策略刷新」的一部分好处，没付 rollout-训练耦合的全部成本。

## 总结

| 环节 | 公式 / 口径 | 要点 |
| --- | --- | --- |
| RFT 流程 | §1 四步 | 特异性全在数据生产侧；TRL 无专用 trainer，采样栈 + 筛选 + `sft_trainer.py` 拼装 |
| 截断分布 | 式 (1) | $\hat{\pi} \propto \pi_\theta \cdot \mathbb{1}[y \in A]$；支撑集只缩不涨； $p_A(x)$ 是难度仪表盘 |
| SFT 回灌 | 式 (2) | 一轮 RFT = 向当前策略截断分布的交叉熵投影 |
| 软加权对接 | 式 (3) | $\exp(s/\alpha)$ 倾斜 = 9.3 式 (3) 闭式解以自身为 $\pi_{\mathrm{ref}}$；指数倾斜结构的第二次消费 |
| 与 RL 分界 | 式 (4) | STaR 原文：示性 reward 的策略梯度丢弃答错样本梯度 = 筛选步；RFT 是其截断采样近似 [B] |
| STaR 停滞 | §4 原文口径 | 解不出新题 ⇒ $p_A \approx 0$ ⇒ 无训练信号；rationalization 用 hint 反向生成 rationale、入集时去掉 hint；每轮从基座重训 |
| STaR 实证 | CommonsenseQA 72.5% vs direct 60.0% vs 30× GPT-3 的 73.0% | [B]，摘要口径 |
| Llama-3 RS | $K$=10-30、RM 选 best、system prompt 引导、PagedAttention >2× 吞吐、model-as-judge 质检（其 §4.3.1） | [A]，§4.2.2；进 SFT 混合，六轮迭代 |
| 判据表 | §7 | 判据 = 目标期望从谁采样；RFT / DPO / SFT 落 offline，PPO / GRPO / RLVR 落 on-policy |

## 练习

1.（推导）补全式 (2) 的第二等号（写出 $`-\mathbb{E}_{\hat{\pi}}[\log \pi]`$ 到 $D_{\mathrm{KL}}(\hat{\pi} \Vert \pi)$ 的恒等变形），并证明：若 SFT 的函数类包含 $\hat{\pi}$ 且优化达到全局最优，则一轮 RFT 后 $\pi^{+}(\cdot \mid x) = \hat{\pi}(\cdot \mid x)$。由此说明 RFT 单轮的「最优结果」就是把截断分布整个搬进策略——探索边界由第 1 步的采样完全决定。

2.（计算）设当前策略对某题的单样本正确率为 $p$。(a) 求 best-of-$`n`$ 至少采到一条正确答案的概率 $1 - (1-p)^{n}$，并算 $p = 0.1$、 $n = 10$ 的值；(b) 求保留质量 $p_A(x)$（阈值取「答案正确」）与 $n$、 $p$ 的关系；(c) 设训练完美吸收筛选集（练习 1 的结论），下一轮该题的单样本正确率变为多少——分「题目只此一道」与「题目分布有迁移」两种情形讨论，由此解释 STaR 停滞的形式化机制（ $p_A \to 0$ 处的信号枯竭）。

3.（推导）软加权版本的对接：验证式 (3) 恰为 9.3 式 (3) 以 $\pi_{\mathrm{ref}} = \pi_\theta$、 $\beta = \alpha$ 的特例；再求该倾斜分布与原策略的 KL 散度 $D_{\mathrm{KL}}(\widetilde{\pi} \Vert \pi_\theta)$ 的表达式（用 $\alpha$、 $s$ 的矩表示到二阶），说明 $\alpha$ 同时控制倾斜强度与偏离幅度。

4.（分析）把 9.3 家族总表与 §7 判据表交叉：给出「无 reference + offline」与「有 reference + on-policy」两个格子里各有哪些方法；再用「判据格 → 系统形态 → 失效模式」三段式，为「数学竞赛题库（有标准答案）」与「开放域对话偏好（人工标注存量集）」各选一个方法并给出理由。

## 参考

- [Zelikman et al., 2022](../references.md)：STaR: Bootstrapping Reasoning with Reasoning，arXiv:2203.14465，[B]（key: star）——循环结构、停滞诊断与 rationalization 动机（§3.2）、潜变量模型与式 (4) 的策略梯度视角、每轮从基座重训、CommonsenseQA 数字
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——RS 工程口径（ $K$=10-30、RM 选 best、system prompt 引导、PagedAttention 2×、质检过滤）、六轮迭代结构（§4.1.6、§4.2.2）
- [Christiano et al., 2017](../references.md)：Deep Reinforcement Learning from Human Preferences，arXiv:1706.03741，[B]（key: christiano）——偏好 RL 源头（RM 打分路线的上游）
- [HuggingFace, huggingface/trl repo](../references.md)（key: trl-repo，教学参照 + 官方实现）——main 分支 trainer 目录的主要 trainer 族清单（无专用 RFT trainer，`sft_trainer.py` 为回灌组件，2026-11-18 核实）
