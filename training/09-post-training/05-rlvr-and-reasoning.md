# 9.5 RLVR 与推理模型的训练（RLVR and the Training of Reasoning Models）

> 本章回答一个问题：不训练 reward model，RL 的奖励信号从哪里来？答案是可验证奖励（RLVR）——用规则程序直接判对错，同时划清它的防 hacking 边界。围绕它拆三条已公开配方：DeepSeek-R1 的四阶段训练链（含 R1-Zero 对照支线，逐阶段的依赖与目的）、Kimi K1.5 的 online mirror descent 变体、长度惩罚与 long2short、Qwen3 的 thinking 模式训练；最后收束到长 CoT 训练动力学（response 长度增长的两种归因、entropy 塌缩）与 test-time scaling 的训练侧含义、蒸馏账 | 前置：8.4（GRPO/Dr.GRPO/DAPO 的目标与失效模式，本章直接复用其式 (1)(2) 口径）、9.1（SFT 工程语义）、9.4（拒绝采样的截断分布视角）、3.2（在线/离线分界）；推理侧成本指针 models 4.8 §5 | 锚点：DeepSeek-R1（[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]）、Kimi K1.5（[Kimi Team, 2025](../references.md)，arXiv:2501.12599，[A]）、Qwen3（[Qwen Team, 2025](../references.md)，arXiv:2505.09388，[A]）、Understanding R1-Zero / Dr.GRPO（[Liu et al., 2025](../references.md)，arXiv:2503.20783，[A]）、DAPO（[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]）；实现锚点：verl `verl/utils/reward_score/math_verify.py`（[A]，HybridFlow 官方栈，2026-11-18 核实）、sail-sg/understand-r1-zero `train_zero_math.py`（Dr.GRPO 官方代码，2026-11-18 核实）

本章口径一句：章内数字全部是所引论文的原始报告口径，不做跨论文换算；模型规模随配方走（R1 链的基座是 DeepSeek-V3-Base，Kimi 那条线未公开参数量，Qwen3 的标尺落在其旗舰 MoE 上）。各节数字当各家的实验记录读，不当同一杆秤上的比较。

## 1. RLVR：reward 不学习，直接验证

RL 的奖励一定要用一个训练出来的 reward model 给吗？不一定。所谓可验证奖励 RL（Reinforcement Learning with Verifiable Rewards, RLVR），是指 reward 不由学习出来的模型打分、而由一个确定性验证程序直接判定的 RL 形态——9.2 的 RLHF 全链里那个 reward model 可以整组拿掉。只要任务的答案可以被程序机械判定（数学题有唯一答案、代码有测试用例），reward 就能这样给出。对题目 $x$ 与模型回答 $o$：

$$
r(x, o) = \begin{cases} 1, & \mathrm{verify}(o, y^{\ast}) = \mathrm{pass} \\ 0, & \text{否则} \end{cases} \qquad (1)
$$

**符号**： $y^{\ast}$：数据集自带的参考答案； $\mathrm{verify}$：规则验证器（rule-based verifier），一段人工写的判定代码，不含学习成分。

两个主力实例。数学题：提取回答中 `\boxed{}` 内的表达式，与 $y^{\ast}$ 做符号等价判定—— $a^{2} - 4$ 与 $(a+2)(a-2)$ 必须判同，这个等价判定本身是验证器的核心难度。代码题：把模型补全放进沙箱跑预定义单测，全过为 1。R1-Zero 的 reward 系统就是这两类规则（[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]，其 §2.2.2）：accuracy rewards（数学等价判定、LeetCode 编译器按测试用例反馈）与 format rewards（强制思考过程落在 think 标签对内）。

算到底的一个实例：数学卷上模型输出 `\boxed{42}`、参考答案 42——提取后判同，reward 得 1；把 42 改成 43，得 0。这是正例。再给一个边界反例：开放式写作没有可判定的标准答案，verify 无从定义——这类任务不在 RLVR 的覆盖域内，reward 只能退回学习路线（RM）。

为什么不干脆训练一个 RM？R1 论文给的是一手理由（其 §2.2.2）：神经 reward model 在大规模 RL 中会被 hack；要修就得重训 RM，既吃额外算力又把管线复杂化。R1-Zero 因此不用任何形式的神经 RM，outcome 和 process 都不用。

这背后的机制值得说一句。9.2 的 reward hacking 有一个共同通道：reward 来自一个学习出来的代理。RM 是人类偏好的有损压缩，策略对 RM 过优化时，分数上涨而真实偏好原地不动。式 (1) 的验证器不是学习出来的：数学等价和单测通过是题目自带的语义事实，代理与目标之间的缝隙被结构性消除，这条 hacking 通道整个不存在。

但「防 hacking」不等于「防一切」。规则验证器至少留三个残余攻击面。

其一，验证器假阳性。答案可猜的题（选择题、判断题）会让「推理过程全错、最终答案蒙对」也拿到 reward 1。K1.5 称之为 false positive verification，对策做在数据侧：直接排除选择题、判断题、证明题；再加一道猜测过滤——让模型不写思维链（Chain-of-Thought, CoT）直接猜答案， $N = 8$ 次尝试内猜对就判定该题易被 hack、从 prompt 集剔除（[Kimi Team, 2025](../references.md)，arXiv:2501.12599，[A]，其 §2.1）。

其二，outcome 验证不看过程。错误推理撞上正确答案同样得 1，噪声以正 reward 的形式进入梯度。

其三，format reward 这类形式约束本身仍是可被模式化利用的弱信号。

所以准确的说法是：RLVR 移走的是「代理奖励被过优化」这一类失效，不是全部失效。也因此，prompt 集的可验证性筛选是 reward 定义的一部分，不是数据工程的附属品——K1.5 给 prompt 集定的三条性质是：多样覆盖、难度均衡、可被准确评估。

实现层长什么样？走读 verl 的规则验证器（`verl/utils/reward_score/math_verify.py`，函数 `compute_score(model_output, ground_truth, timeout_score=0, timeout=30.0)`，2026-11-18 核实）。它把参考答案包成 `\boxed{...}`，调 HuggingFace Math-Verify 库的 `parse`（gold 用 `LatexExtractionConfig`，pred 用 `ExprExtractionConfig + LatexExtractionConfig` 双通道），再做 `verify(gold, pred)` 等价判定；任意 gold-pred 对判同即得 1.0。

工程护栏见下方坑注。Dr.GRPO 官方 repo 是同构实现（sail-sg/understand-r1-zero，`train_zero_math.py` 的 `MATHOracle.get_reward` 调 `math_grader.boxed_reward_fn`，多进程打分、异常记 0）。

> **坑**：规则验证器挂在 rollout 打分的关键路径上——一条让符号判定陷入死循环的畸形回答，会拖停整个 batch 的打分。规避办法是子进程化加超时护栏：verl 把整个判定放进 4 worker 的 `ProcessPoolExecutor` 子进程，30 秒超时返回 0，超时按判错计。

## 2. R1-Zero：基座上直接 GRPO，长 CoT 涌现

R1-Zero 的配方只有一行：在 DeepSeek-V3-Base（基座，无 SFT）上直接跑 GRPO，reward 用 §1 的两类规则（[DeepSeek-AI, 2025](../references.md)，其 §2.2；GRPO 的目标与组基线见 8.4 式 (1)(2)，[Shao et al., 2024](../references.md) 出处）。

训练模板只做结构约束：先输出推理过程、再输出最终答案，用 think/answer 标签包裹。论文明确说不加任何内容偏置——不强制反思、不指定解题策略。目的恰恰是干净地观察 RL 之下模型的自然演化（其 §2.2.3 口径）。

这样训出来会发生什么？三个观测，全部有图有数。

第一，能力曲线。AIME 2024 pass@1 从训练初的 15.6% 单调爬到 71.0%（其 Figure 2）；majority voting（cons@64）到 86.7%（其 Table 2）。

第二，长度曲线。训练集上的平均 response 长度随训练持续增长，论文的表述是 thinking time 从数百 token 涨到数千 token（其 Figure 3）。模型自发学会用更多 test-time 计算换正确率——这就是 test-time scaling 被训练内化的直接证据，§7 展开。

第三，行为涌现。反思（中途 revisit 并重新评估先前步骤）与换路探索自发出现；论文给出「aha moment」实例：模型在中途插入 "Wait, wait. Wait." 并推翻重来（其 Table 3）。但这个证据的强度要先打个折——§6 会重审它，基座本身已带自反关键词。

R1-Zero 的缺陷同样由论文点名（其 §2.2.4 末段）：可读性差、语言混杂（language mixing，即单条 CoT 里混用多种语言；「缺 markdown 结构、回答不适合阅读」的具体描述见其 §2.3.1 的 Readability 条）。

R1 配方的动机按原文口径是两条：修这两个缺陷，并进一步增强推理性能（其摘要："To address these issues and further enhance reasoning performance"）。缺陷侧的本质是「产物不可用」：AIME 再高，混语且无结构的 CoT 交不到用户手里。下一节的多阶段链，逐阶段对着这两条来设计。

## 3. DeepSeek-R1：四阶段配方与依赖图

R1 的训练是四阶段链（[DeepSeek-AI, 2025](../references.md)，其 §2.3 明确称 four stages）。每个阶段的输入依赖与目的各不相同。先把依赖图画出来，再逐阶段拆：

```
                DeepSeek-V3-Base（基座）
                 │                     │
   支线（对照实验）│                     │ 主线（R1 配方）
                 ▼                     ▼
   ┌──────────────────────┐   阶段 1  冷启动 SFT
   │ R1-Zero：直接 GRPO     │          数据：数千条长 CoT
   │ 规则 reward（式 (1)）  │─输出整理─┐  来源：① few-shot 长 CoT 示例提示
   │ AIME 15.6% → 71.0%    │ +人工后处理├─② 直接提示生成带反思/验证的回答
   │ 长度随训练增长         │          │  ③ R1-Zero 输出的可读化收集
   │ 缺陷：可读性差、        │          │  格式：|<推理过程>|<summary> 两段式
   │       语言混杂         │          │  目的：压住 RL 早期不稳 + 可读性先验
   └──────────────────────┘          ▼
                        阶段 2  推理导向 RL（GRPO，流程同 R1-Zero）
                                 reward = 规则正确性 + 语言一致性（式 (2)）
                                 任务域：数学/代码/科学/逻辑（可验证）
                                 终点：推理任务上收敛
                                 │ checkpoint（数据发生器）
                                 ▼
                        阶段 3  拒绝采样 + SFT（回到 V3-Base 重训）
                                 推理数据约 600k：阶段 2 ckpt 采样、
                                   规则验证（部分用 V3 作生成式 judge）；
                                   滤除混杂语 / 长段落 / 代码块
                                 非推理数据约 200k：复用 V3 SFT 管线
                                 合计约 800k，SFT 两个 epoch
                                 │ checkpoint
                                 ▼
                        阶段 4  全场景 RL
                                 推理数据：规则 reward（同阶段 2）
                                 通用数据：RM（V3 偏好管线）
                                 helpfulness 只评 summary；
                                 harmlessness 评整段（CoT + summary）
                                 │
                                 ▼
                     DeepSeek-R1（AIME 79.8% / MATH-500 97.3%）
                                 │ 800k 数据复用
                                 ▼
                     蒸馏支线：小模型 SFT（无 RL，§7）
```

**阶段 1，冷启动 SFT**（其 §2.3.1）。规模是「数千条」长 CoT，量级刻意小——论文的理由是防止基座直接进 RL 时的早期不稳定阶段。数据有四个来源：以长 CoT 为示例的 few-shot 提示；直接提示模型生成带反思与验证的详细回答；把 R1-Zero 的输出按可读格式收集；人工标注后处理。格式定成「推理过程 + 结尾 summary」两段式，创建时就过滤掉不适合阅读的回答。

注意依赖图上那支跨线箭头：R1-Zero 的输出是冷启动数据的来源之一。支线实验不是摆设，是主线的数据上游。论文还给了冷启动的第二个收益：按人类先验设计的格式起点，让最终性能优于 R1-Zero（其原话：迭代训练是更好的方式）。

**阶段 2，推理导向 RL**（其 §2.3.2）。在冷启动 checkpoint 上跑与 R1-Zero 相同的大规模 RL（GRPO），任务限在「定义良好、有明确解」的推理密集域。语言混杂在这一阶段被正面处理：论文观察到 CoT 在多语 prompt 下频繁混用语言，于是引入语言一致性奖励（language consistency reward）：

$$
r_{\mathrm{lang}}(o) = \frac{\text{CoT 中目标语言的词数}}{\text{CoT 的总词数}}, \qquad r_{\mathrm{final}}(o) = r_{\mathrm{acc}}(o) + r_{\mathrm{lang}}(o) \qquad (2)
$$

**符号**： $r_{\mathrm{acc}}$：式 (1) 的规则正确性奖励。

代入一组数：一条 CoT 共 100 词、其中 80 词是目标语言，语言一致性项得 80/100；该题答对时正确性项记 1，最终 reward 就是两者相加。题目答错时语言分照加、正确性项为 0——所以该奖励只修饰形态，不掩盖对错。

两个要点都按原文口径。一是归属：这是 R1 阶段 2 的组件，R1-Zero 没有它——Zero 的缺陷正是它要修的东西；其他家的配方里也没有同名组件，K1.5 全文无此机制，勿张冠李戴。二是代价：论文明确报告消融显示该对齐带来轻微的性能下降，接受代价的理由是与人类可读性偏好对齐。组合方式是把两个 reward 直接相加，不加权重系数。

**阶段 3，拒绝采样 + SFT**（其 §2.3.3）。阶段 2 收敛后，其 checkpoint 换了一个角色：数据发生器。对推理 prompt 做拒绝采样，每题采多条、只留正确的。数据域也从「仅规则可验证」扩宽：一部分题由生成式 judge 把关——把 ground truth 与模型预测一起喂给 DeepSeek-V3 判定。同时过滤混杂语言、长段落与代码块的 CoT。最终得约 600k 推理样本。

非推理侧约 200k：写作、factual QA、自我认知、翻译，直接复用 V3 的 SFT 管线数据；部分任务由 V3 生成潜在 CoT，简单问候类不带 CoT。合计约 800k。

一个容易读错的依赖细节：这一步的 SFT **从 DeepSeek-V3-Base 重新起训**（原文 "fine-tune DeepSeek-V3-Base for two epochs"），不是在 RL checkpoint 上续训。阶段 2 的 checkpoint 只以数据形式进入主线。拒绝采样本身的截断分布形式化归 9.4；本章只登记它在配方里的位置——RL 的产物被离线过滤成 SFT 数据，on-policy 信号转成静态数据集。

**阶段 4，全场景 RL**（其 §2.3.4）。目标是对齐人类偏好：helpfulness、harmlessness 与推理能力同时维护。推理数据沿用规则 reward；通用数据改用 RM，沿用 V3 的偏好对分布与管线。两处评审口径值得抄下来：helpfulness 只评最终 summary，避免偏好信号干扰推理过程本身；harmlessness 评整段输出（CoT + summary），因为风险可能藏在推理里。

逐阶段的依赖逻辑合起来读：阶段 1 解「RL 早期不稳、可读性先验难以从二值 reward 长出」；阶段 2 解「推理能力的主体增益、语言混杂」；阶段 3 解「RL 只覆盖可验证域，通用能力要从数据侧补回」，并重起 SFT 把两个域揉进同一个 checkpoint；阶段 4 解「偏好对齐」。每一步的产物形态不同——策略 checkpoint、数据、策略。依赖图上真正流动的不是参数，而是能力载体的转换：参数 → 数据 → 参数 → 参数。（「阶段 3 为何回基座重训」论文未给显式理由；上文的动机读法是本书分析，非原文口径。）

## 4. Kimi K1.5：mirror descent 变体与长度经济

K1.5 先把 reward 写成二值判定 $r(x, y, y^{\ast}) \in \lbrace 0, 1 \rbrace$（[Kimi Team, 2025](../references.md)，arXiv:2501.12599，[A]，其 §2.3.1）：可验证题用规则，自由形式答案用 judge RM。训练目标是最大化期望 reward。优化算法不是 PPO/GRPO 的 clip 家族，而是 online policy mirror descent 的一个变体（其 §2.3.2）：第 $i$ 轮以当前模型 $\pi_{\theta_i}$ 为参考策略，解一个 KL 正则问题。路线一句话：先解出这个目标的闭式最优，再把闭式解改写成可采样的平方代理损失，最后对代理取梯度——关键一步是把归一化常数用组内采样近似掉

$$
\max_{\theta}\ \mathbb{E}_{(x, y^{\ast}) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta}\left[ r(x, y, y^{\ast}) \right] - \tau \thinspace \mathrm{KL}\left( \pi_\theta(x) \Vert \pi_{\theta_i}(x) \right) \right] \qquad (3)
$$

**符号**： $z$：思维链（中间步骤序列）； $y$：最终答案； $\tau \gt 0$：正则强度。式 (3) 有闭式解（练习 3 推导）：

$$
\pi^{\ast}(y, z \mid x) = \frac{1}{Z} \thinspace \pi_{\theta_i}(y, z \mid x) \thinspace \exp\left( r(x, y, y^{\ast}) \big/ \tau \right), \qquad Z = \sum_{y', z'} \pi_{\theta_i}(y', z' \mid x) \thinspace \exp\left( r(x, y', y^{\ast}) \big/ \tau \right) \qquad (4)
$$

**命题**（KL 正则目标的闭式解）：上面的 KL 正则目标，其最优策略存在且唯一，形式为「参考策略按 reward 指数倾斜、再归一化」的分布——即上方这个式子。**证明**（路线；完整展开留作练习 3）：把目标看成逐 prompt 上的分布优化，归一化约束以 Lagrange 乘子进入，逐点变分即得指数倾斜；归一化常数 Z 正是乘子的兑现。数值直觉：组内四条回答恰有一条答对时，这一条拿到的相对权重是其余三条的 e 的 1/τ 次幂倍——τ 越小，概率质量向正确答案集中得越狠。这个形态与 9.3 的 DPO 闭式解同族——KL 正则 RL 的最优解都是指数倾斜形态。但 $\pi^{\ast}$ 无法直接采样，K1.5 把它转成对数比约束 $r - \tau \log Z = \tau \log(\pi^{\ast} / \pi_{\theta_i})$，再用平方残差做代理损失：

$$
L(\theta) = \mathbb{E}_{(x, y^{\ast}) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_{\theta_i}} \left[ \left( r(x, y, y^{\ast}) - \tau \log Z - \tau \log \frac{\pi_\theta(y, z \mid x)}{\pi_{\theta_i}(y, z \mid x)} \right)^{2} \right] \right] \qquad (5)
$$

其中 $\tau \log Z$ 用采样近似 $\tau \log \frac{1}{k} \sum_{j} \exp(r_j / \tau)$。实践中发现更省事的替代同样有效：直接拿采样 reward 的经验均值 $\overline{r}$（原文依据： $\tau \to \infty$ 时 $\tau \log Z$ 趋于 $\pi_{\theta_i}$ 下的期望 reward）。对式 (5) 取梯度，每题从参考策略采 $k$ 条：

$$
\frac{1}{k} \sum_{j=1}^{k} \left( \nabla_\theta \log \pi_\theta(y_j, z_j \mid x) \thinspace \left( r(x, y_j, y^{\ast}) - \overline{r} \right) - \frac{\tau}{2} \thinspace \nabla_\theta \left( \log \frac{\pi_\theta(y_j, z_j \mid x)}{\pi_{\theta_i}(y_j, z_j \mid x)} \right)^{2} \right) \qquad (6)
$$

代入玩具数读一遍第一项：组内回答 4 条、reward 依次 1、1、0、0，经验均值为 1/2——前两条的 log 概率梯度按 +1/2 加权，后两条按 −1/2 压低；第二项则把当前策略与参考策略的 log 比向 0 拉回。

式 (6) 的第一项正是「组均值作 baseline 的策略梯度」，与 GRPO 的组基线（8.4 式 (1)，去掉 std 归一）同族。与 GRPO 有两处差异：样本采自参考策略 $\pi_{\theta_i}$ 而非严格 on-policy，迭代内允许 off-policy 复用；正则从「ratio clip」换成「log-ratio 的 $\ell_2$ 惩罚」。

另外两条实现口径。一，不用 value network——论文的理由是长 CoT 需要试错式探索：value 函数的 credit assignment 会惩罚「走错但能救回来」的中间步骤，而这种步骤正是训练反思能力所需的素材。二，每轮换参考策略时重置优化器，因为各轮的优化问题不同。

> **提点**：「每轮重置优化器」在监督训练里几乎没有对应物。它等效于承认每轮的优化问题都不同——动量里攒下的历史曲率信息不但无用，还会把上一轮参考策略的残留带进新一轮。

长度在 K1.5 里是被显式管理的变量。论文观测到 RL 训练中 response 长度持续上涨（overthinking），于是把长度做成奖励项（其 §2.3.3）：组内 $k$ 条回答，记 $\mathrm{len}(i)$ 为第 $i$ 条长度， $\mathrm{min\_len}$ / $\mathrm{max\_len}$ 为组内极值（全相等则全组长度奖励为零）：

$$
\mathrm{len\_reward}(i) = \begin{cases} \lambda, & r(x, y_i, y^{\ast}) = 1 \\ \min(0, \lambda), & r(x, y_i, y^{\ast}) = 0 \end{cases}, \qquad \lambda = 0.5 - \frac{\mathrm{len}(i) - \mathrm{min\_len}}{\mathrm{max\_len} - \mathrm{min\_len}} \qquad (7)
$$

语义读一遍：正确的回答在组内按长度线性退火，最短拿 $+0.5$、最长拿 $-0.5$；错误的只罚不奖（ $\lambda \gt 0$ 时取 0）。「错了还写长」被双重惩罚。一组具体数（组内长度 200、400、800、1600，对、对、错、错）：首条 λ 为 1/2、末条 λ 为 −1/2，中间两条与错误两条的最终奖励留作练习 2 实算。该奖励加权加到原 reward 上。论文还报告训练初期上长度惩罚会拖慢学习，处置方式很直接：先跑无惩罚的标准阶段，再切到恒定惩罚（warmup 式调度）。

配套还有三件采样与系统件（其 §2.3.4 与 §2.6.2）。第一，curriculum sampling：按难度标签先易后难。第二，prioritized sampling：按 $1 - s_i$ 反比于历史通过率采样， $s_i$ 是逐题成功率——与 8.4 §6 的 DAPO dynamic sampling 同为「把算力花在非饱和题上」，只是实现位置不同：K1.5 动采样权重，DAPO 动组过滤。第三，partial rollout：每条 rollout 设固定 token 预算，超预算的轨迹截断、尾部进 replay buffer 下一轮续算；只有当前轮段是 on-policy，段可从 loss 中剔除；异步 rollout worker 避免长轨迹独占资源；重复检测提前终止。

**命题**（全对全错组零梯度）：组内 reward 全同（全对或全错）时，每条回答减去组均值后恰为零，该组对批梯度无贡献；std 归一只是逐组公共因子（实现上加小常数防除零），不改变零贡献（KL 正则项不在此列，它照常起锚定作用）。**证明**（路线；数值展开留作练习 1）：advantage 的构造是组内 reward 减组均值，全同则逐条为零；单题通过率 p 之下，组内既有对又有错的概率等于 1 减去两个纯态的概率——这个数值形态，以及它在训练后期对批梯度的稀释，练习 1 展开。

long2short（其 §2.4）解决一个反向问题：长 CoT 模型强，但推理贵。目标是把长 CoT 模型的「思考先验」转移给短模型，四条路线并列：model merging（长模型与短模型权重合并）、shortest rejection sampling（同题采样取最短正确回答做 SFT）、DPO（以长/短回答构对）、long2short RL。最后一种的做法：选性能与 token 效率平衡最好的模型为基座，单独跑第二阶段 RL——加式 (7) 的长度惩罚，并把最大 rollout 长度砍小，超长即使正确也罚。论文图 7 的对照显示 long2short RL 的 token 效率最高。数字量级：k1.5-short w/ rl 在 AIME 2024 得 60.8（8 次平均），平均只用 3272 token。长模型本体的标尺：AIME 77.5、MATH 500 96.2、Codeforces 94 百分位、MathVista 74.9。

还有一处与 R1 的对照必须点出来：K1.5 在数学 reward 上用了学习出来的判定器。value-head RM 收集约 800k 数据、人工抽查准确率 84.4%；CoT RM（先生成逐步判定再输出 JSON 结论）同样约 800k、准确率 98.5%；RL 采用后者（其 §2.3.5）。这与 §1 的「规则验证器防 hacking」不矛盾：R1 的数学题把答案限定在 `\boxed{}` 格式里，规则可判；K1.5 要覆盖自由形式答案，等价判定本身需要模型参与。代价是把「judge 的准确率」变成 reward 质量的上限——98.5% 对 84.4% 的差距，就是这条上限的实测值。

## 5. Qwen3：thinking 模式的训练侧

Qwen3 的后训练同为四阶段（[Qwen Team, 2025](../references.md)，arXiv:2505.09388，[A]，其 §4.1-4.4），骨架与 R1 平行，但目标多一个：把 thinking 与 non-thinking 两种模式训进同一个模型。

**阶段 1，Long-CoT Cold Start**。数据集覆盖数学、代码、逻辑、STEM，每题带已验证的参考答案或代码测试用例。过滤分两道。query 侧：用 Qwen2.5-72B-Instruct 剔除不可验证题（多子问题、通用文本生成），再剔除「不用 CoT 就能答对」的题——防浅层猜测混进推理数据；并做领域均衡标注。response 侧：用 QwQ-32B 每题采 $N$ 个候选，QwQ 持续失败的题人工核查，再滤掉六类——最终答案错、大量重复、猜测无推理、thinking 与 summary 不一致、语言混杂或风格跳变、疑似与验证集过近。

注意一个分工差异：语言混杂在这里是数据过滤条件，不是奖励项。与 R1 阶段 2 的奖励级处理（式 (2)）相比，这是同一问题的两种解法。该阶段的明确目标是「灌入推理模式而不追求即时性能」：把样本数与训练步数都压到最小，给后续 RL 留出空间（原文口径：potential 不被过早限死）。

**阶段 2，Reasoning RL**。进入这一阶段的 query-verifier 对要满足四条标准：未在冷启动用过；对冷启动模型可学习；尽可能难；覆盖足够宽的子领域。最终 3995 对，用 GRPO 更新。工程口径三条：大 batch、每题 rollout 数拉高、off-policy 训练提样本效率（具体数值未披露）；通过控制模型 entropy（使其稳定上升或保持平稳）平衡探索与利用；单次 RL 运行全程无需人工调超参。标尺数字：Qwen3-235B-A22B 的 AIME'24 在 170 个 RL step 内从 70.1 升到 85.1。

**阶段 3，Thinking Mode Fusion**。在 Reasoning RL 模型上做持续 SFT，把 non-thinking 能力融进来。数据构造分两侧。thinking 侧：用阶段 2 模型对阶段 1 query 做拒绝采样生成，保证融合不损伤阶段 2 的推理能力。non-thinking 侧：覆盖代码、数学、指令遵循、多语、创作、问答、角色扮演，用自动 checklist 评质量，低资源语言特意上调翻译任务占比。

chat template 引入 /think 与 /no_think 开关（写在 user query 或 system message）；non-thinking 样本保留空的 think 块，以维持格式一致；默认 thinking 模式；多轮对话随机插多个开关，以最后一个为准。

thinking budget 的机制顺此长出：思维长度到用户阈值时人工截断、插入 stop-thinking 指令，模型基于已有推理直接给答案。论文强调该能力未经显式训练，是模式融合的副产物。评估侧配置（其评估设置口径）：thinking 模式采样 temperature 0.6、top-p 0.95、top-k 20。

**阶段 4，General RL**。覆盖 20 余个任务，各自定制打分标准；目标能力五类：指令遵循、格式遵循（含 /think 开关响应与 think 标签使用）、偏好对齐、agent 能力（rollout 中允许与真实环境多轮交互调工具）、专项场景（如 RAG 抗幻觉）。reward 分三类：规则型（指令遵循、格式这类高精判定，防 hacking）；带参考答案的模型判定（Qwen2.5-72B-Instruct 对照参考打分，避免纯规则的假阴性）；无参考的偏好 RM（覆盖开放查询）。

小模型不走这条链，走 strong-to-weak distillation（其 §4.5），分两相。off-policy 相：把 teacher 的 /think 与 /no_think 双模式输出直接蒸给小模型打底。on-policy 相：让小模型自采，对齐 teacher（Qwen3-32B 或 235B-A22B）的 logits，最小化 KL。与 R1 的纯 SFT 蒸馏（§7）对照，Qwen3 多了一步 student-on-policy 的 logits 级对齐。

把两家骨架并排看。R1：冷启动 → 推理 RL → 拒采 SFT（回基座重训）→ 全场景 RL。Qwen3：长 CoT 冷启动 → Reasoning RL → 模式融合 SFT（不回基座）→ General RL。共同的配方骨架是「SFT 灌格式先验 → RL 攻可验证域 → SFT 融合通用域 → RL 对齐偏好」；差别在阶段 3 的续训起点与阶段 4 的 reward 结构——R1 通用域用 RM、推理域用规则，Qwen3 三类 reward 混编。

## 6. 长 CoT 训练动力学：长度增长与 entropy 塌缩

R1 的 Figure 3 留下的表面事实是：response 长度随训练单调增长。这个增长该记在谁头上？Dr.GRPO 论文（[Liu et al., 2025](../references.md)，arXiv:2503.20783，[A]）把它拆成两种归因，各自给了证据。

归因一，能力侧。长度增长与 reward 增长同步——他们自己的对照实验里，GRPO 与 Dr.GRPO 都呈现「reward 涨则长度涨」（其 Figure 6 Plots 1-2）；R1-Zero 相对基座在全部回答类别上长度都增加，含正确回答（其 Table 5；错误回答平均更长，论文的解释是错误更多来自难题、难题本就需要更长推理）。这部分增长是策略在学「用更多 token 换正确率」，是真实的 test-time 计算内化。

归因二，优化伪影。GRPO 的逐 response 长度归一（8.4 §5 的 $1/|o_i|$ 项）让错误回答被摊薄惩罚、推向变长。表现是：GRPO 在 reward 改善放缓后长度仍持续上涨；Dr.GRPO 去掉该偏置后，长度不再疯长、错误回答的长度回落（其 Figure 6 Plot 2/4）。

结论口径要精确：「RL 涌现长 CoT」这个说法里混着真能力增长与归一化伪影两种成分。拿长度曲线当能力证据之前，先除伪影。

entropy 塌缩是长 CoT RL 的第二条失效线：朴素 PPO/GRPO 下策略熵随训练快速下降、组内样本趋同，探索被截断（DAPO 其 Figure 2(b) 的观测，[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]）。DAPO 的 clip-higher 正是针对它的修法：解耦上下界 $\varepsilon_{\mathrm{low}} = 0.2$ / $\varepsilon_{\mathrm{high}} = 0.28$，上界放宽让低概率探索 token 能涨；机制推导见 8.4 §6。此处只登记它在配方层的对偶——Qwen3 阶段 2 从调度侧控制 entropy 稳定上升或持平（§5），DAPO 从 clip 几何侧动手。同一失效模式，两种干预位置。

「aha moment」的证据强度也被同一篇论文重审（其 §2.3）。他们自己部署 DeepSeek-V3-Base-685B，用 R1 模板在 500 道 MATH 题上采样，基座回答中即可采到自反关键词（"Aha"、"wait"、"verify the problem"；原文口径为 "a decent amount"，未给占比数字）。Qwen2.5 基座在无任何模板时回答率 100%、为被测各基座最高；Qwen2.5-Math 无模板的平均成绩较 4-shot 提示提升约 60%（其 §2.2 与 Table 1）。

怎么读这组证据？RL 放大了预训练已有的模式，「从无到有的涌现」被降级为「从有到显的放大」。这不削弱 RL 的功劳——15.6% → 71.0% 的能力曲线是硬的——但把叙事从「创造」修正为「选择压力」（本书读法）。

## 7. 推理预算的内化与蒸馏

「花多少 token 想」是推理期的超参，还是训练出来的策略性质？本章的回答是后者：RL 把推理预算从推理期超参变成了策略的内生变量。§6 归因一说明策略会自己拉长预算；K1.5 的长度惩罚与 long2short（§4）是把预算往回压的对偶操作；Qwen3 的 thinking budget（§5）把预算控制权暴露给用户。架构侧为这条链付出的账——KV cache 随链长线性增长、串行 decode 步数放大、训推算力配比重写——归 models 4.8 §5。本章只指出一点：训练与推理的接口在「长度」这一个标量上。

蒸馏是推理能力向下传导的便宜通道（口径：「便宜」指达到同等 AIME 分数所需的训练算力更低；数据由 R1 主流程阶段 3 副产，不另计费）。R1 的配方（其 §2.4）：把阶段 3 的约 800k 数据直接拿去 SFT 开源小模型——Qwen2.5-Math-1.5B/7B、Qwen2.5-14B/32B、Llama-3.1-8B、Llama-3.3-70B-Instruct——只做 SFT、不加 RL。论文明说加 RL 还能涨，但留给社区；本次只为证明蒸馏本身有效。

结果：R1-Distill-Qwen-1.5B 在 AIME 28.9%、MATH 83.9%；R1-Distill-Qwen-7B 的 AIME 55.5% 超过 QwQ-32B-Preview；R1-Distill-Qwen-32B 的 AIME 72.6%、MATH-500 94.3%、LiveCodeBench 57.2%。

关键对照实验（其 §4.1）：在同一 Qwen-32B-Base 基座上换成直接跑大规模 RL（数学/代码/STEM 数据，超 10K step，不消费那 800k），只得 AIME 47.0%。蒸馏 72.6% 对自训 RL 47.0%，同规模下差 25.6 个点。

论文的两条结论照抄其义：其一，蒸馏强模型到弱模型效果好且便宜（上文 72.6% 对 47.0% 即其例证）；其二，小模型自训大规模 RL 算力需求高、且未必追平蒸馏，突破智能上限仍需更强基座与更大规模 RL。工程归属：蒸馏的 SFT 侧语义（loss masking、epoch、LR 量级）归 9.1；拒绝采样数据的生成侧归 9.4；Qwen3 的 on-policy logits 蒸馏变体见 §5。

## 总结

| 组件 / 现象 | 机制与公式 | 出处与证据 |
| --- | --- | --- |
| RLVR reward | 式 (1)：规则验证器二值判定；无学习代理 ⇒ RM 过优化通道结构性消除；残余攻击面：可猜题假阳性、过程不检、形式信号可被利用 | R1 §2.2.2 [A]；K1.5 §2.1（N=8 猜测过滤）[A] |
| R1-Zero | 基座直接 GRPO；AIME 15.6%→71.0%（cons@64 86.7%）；长度单调增长；aha moment 实例；缺陷=可读性差+语言混杂 | R1 §2.2，Fig 2/3、Table 2/3 [A] |
| R1 四阶段 | 冷启动 SFT（数千条）→ 推理 RL（式 (2) 语言一致性奖励在此引入、轻微性能代价）→ 拒采 SFT（约 600k+200k、回基座重训 2 epoch）→ 全场景 RL（推理规则 + 通用 RM；helpfulness 只评 summary） | R1 §2.3 [A] |
| K1.5 优化器 | online mirror descent 变体：式 (3) KL 正则 → 式 (4) 指数倾斜闭式解 → 式 (5) 平方代理 → 式 (6) 组均值 baseline + log-ratio $\ell_2$；无 value net；每轮重置优化器 | K1.5 §2.3.2 [A] |
| K1.5 长度经济 | 式 (7) 组内线性长度惩罚（warmup 后恒定）；long2short 四法，RL 版砍最大 rollout；short w/ rl 60.8 AIME @ 3272 token | K1.5 §2.3.3/§2.4 [A] |
| Qwen3 thinking | 四阶段；阶段 2 仅 3995 对 query-verifier、entropy 稳态控制、AIME'24 70.1→85.1（170 step）；模式融合 SFT + /think 开关；thinking budget 为融合副产物（非显式训练）；General RL 三类 reward | Qwen3 §4 [A] |
| 长度动力学 | 增长 = 能力内化（与 reward 同步、全类别增长）+ 归一化伪影（ $1/\|o_i\|$ 项，reward 改善放缓仍涨）；先用 Dr.GRPO 口径除伪影再当证据 | Dr.GRPO Fig 6/Table 5 [A] |
| entropy 塌缩 | DAPO Fig 2(b) 观测 → clip-higher（8.4 §6 机制）；Qwen3 调度侧 entropy 控制为对偶干预 | DAPO [A]、Qwen3 §4.2 [A] |
| 蒸馏 | R1：800k SFT-only；Distill-32B 72.6% vs 同基座自训 RL 47.0%（AIME）；Qwen3：off-policy + on-policy logits KL | R1 §2.4/§4.1 [A]；Qwen3 §4.5 [A] |

## 练习

1.（计算 · [20]）GRPO 组内有效题比例：二值 reward、单题通过率 $p$、组大小 $G$，advantage 非零要求组内既有对也有错，证明有效组概率为 $1 - p^{G} - (1-p)^{G}$；代入 $G = 16$、 $p = 0.9$ 与 $p = 0.99$ 各算一遍（约 0.815 与 0.148），由此解释训练后期「全对组占比上升 ⇒ 批梯度被零 advantage 稀释」的机制（DAPO dynamic sampling 的动机，8.4 §6）。校验：G 取 2 时公式应退化为 2·p·(1−p)，直接展开即可对照。
2.（计算 · [20]）式 (7) 的数值形态：组内 4 条回答长度 200 / 400 / 800 / 1600，reward 1 / 1 / 0 / 0。算各条的 $\lambda$ 与最终 $\mathrm{len\_reward}$；再问：若 800 那条改判为对（reward 变 1），组内激励结构怎么变？结合 §4 的 warmup 调度说明为什么训练初期不宜上该惩罚。校验：λ 对长度单调递减，且最短条恒为 1/2、最长条恒为 −1/2——先把四个端点值对上再谈中间两条。
3.（推导 · [30]）从式 (3) 推出式 (4)：把 $\pi(y, z \mid x)$ 当作逐 $x$ 的分布做变分，加归一化约束 $\sum_{y, z} \pi(y, z \mid x) = 1$ 的 Lagrange 乘子，证明最优解为指数倾斜形式 $\pi^{\ast} \propto \pi_{\theta_i} \exp(r / \tau)$；再验证 $\tau \to \infty$ 时 $`\tau \log Z \to \mathbb{E}_{\pi_{\theta_i}}[r]`$（提示： $\exp(r/\tau) \approx 1 + r/\tau$ ），说明 K1.5 用经验均值 $\overline{r}$ 近似 $\tau \log Z$ 的合理性边界（ $\tau$ 小、reward 方差大时该近似往哪边偏）。校验：把解代回 KL 正则目标，最优值应恰为对数配分常数（τ 乘 log Z）；代不出这个值，说明归一化丢了。
4.（代码 · [30]）写一个 50 行内的玩具验证器：正则提取回答中 `\boxed{...}` 内容、字符串归一化（去空格、大小写折叠）后与参考答案比对，接进一个手写的小组 GRPO 打分循环。构造两个 false positive 用例（(a) 回答正文出现 "the answer is 42" 但推理全错；(b) 选择题蒙对），对照 §1 的残余攻击面与 K1.5 §2.1 的数据侧过滤，说明你的玩具验证器需要补哪几件才能进训练回路（符号等价、子进程超时、可猜题过滤）。校验：(a)(b) 两例在修补后的验证器上必须判 0，而「`\boxed{42}` 对参考答案 42」必须判 1；三个用例全过才算修好。

## 参考

- [DeepSeek-AI, 2025](../references.md)：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning，arXiv:2501.12948，[A]（key: deepseek-r1）——R1-Zero 配方与三类观测（§2.2）、R1 四阶段（§2.3，语言一致性奖励在 §2.3.2）、蒸馏配方与蒸馏 vs 自训 RL 对照（§2.4、§4.1）、PRM/MCTS 失败记录（§4.2）
- [Kimi Team, 2025](../references.md)：Kimi k1.5: Scaling Reinforcement Learning with LLMs，arXiv:2501.12599，[A]（key: kimi-k1.5）——RL prompt 集三性质与猜测过滤（§2.1）、online mirror descent 变体式 (3)-(6)（§2.3.2）、长度惩罚式 (7)（§2.3.3）、采样策略（§2.3.4）、long2short（§2.4）、CoT RM 98.5% 对 84.4%（§2.3.5）、partial rollout（§2.6.2）
- [Qwen Team, 2025](../references.md)：Qwen3 Technical Report，arXiv:2505.09388，[A]（key: qwen3）——四阶段后训练（§4.1-4.4：冷启动过滤、3995 对 + entropy 控制、模式融合与 thinking budget、General RL 三类 reward）、strong-to-weak 蒸馏（§4.5）
- [Liu et al., 2025](../references.md)：Understanding R1-Zero-Like Training: A Critical Perspective，arXiv:2503.20783，[A]（key: dr-grpo）——长度增长的双归因（Fig 6、Table 5）、基座自反关键词重审（§2.3）；官方代码 sail-sg/understand-r1-zero `train_zero_math.py`（2026-11-18 核实）
- [Yu et al., 2025](../references.md)：DAPO: An Open-Source LLM Reinforcement Learning System at Scale，arXiv:2503.14476，[A]（key: dapo）——entropy 塌缩观测（Fig 2(b)）与 clip-higher 动机（机制归 8.4 §6）
- [Shao et al., 2024](../references.md)：DeepSeekMath（GRPO 出处），arXiv:2402.03300，[A]（key: deepseekmath）——组基线目标（推导归 8.4）
- [Sheng et al., 2024](../references.md)：HybridFlow / veRL，arXiv:2409.19256，[A]（key: verl）——规则验证器实现锚点 `verl/utils/reward_score/math_verify.py` `compute_score`（2026-11-18 核实）
