# training W6 批 2 审查报告 — 9.4 / 9.5 逐章 + 全波 5 章跨章 sweep

日期：2026-11-18。evaluator：独立 fresh-context（未继承写作上下文；WebFetch 被环境封锁，全部一手来源经 curl 拉取 ar5iv 全文 HTML + GitHub API 本地核对）。审查对象：`training/09-post-training/04-rejection-sampling-and-rft.md`（9.4）、`05-rlvr-and-reasoning.md`（9.5）逐字；`01-sft.md`（9.1）、`02-rlhf-pipeline.md`（9.2）、`03-dpo-family.md`（9.3）跨章扫读 + 抽查。

立场声明：假设内容有错，逐项回源。下文 verification log 是审查存在的证据。

## 结论先行

**不放行**（一轮小修后复审即可）。S1 × 1（9.4 式 (2) 旁注恒等式符号错误，一字符修复）、S2 × 3（9.5 两处 § 号错挂、9.4 一处 Llama-3 质检归属混排、9.4 正文残留「存疑待补源」过程标记）、S3 × 6。两项强制重推导已独立完成（截断-再归一化形式与判据表；R1 四阶段依赖图），R1 的 language consistency reward 归属核实为 R1 阶段 2 独有、K1.5 全文无此机制——章内归属正确。

## Findings 表

| # | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S1** | 9.4 §2，式 (2) 后括注 | 「−E_π̂[log π] = D_KL(π̂‖π) − H(π̂)」 | 交叉熵分解为 H(π̂,π) = H(π̂) + D_KL(π̂‖π)：D_KL = E_π̂[log π̂] − E_π̂[log π] = −H(π̂) − E_π̂[log π]，移项得 −E_π̂[log π] = D_KL(π̂‖π) **+** H(π̂)。现文为 −H(π̂)。反例验证：取 π̂=π，左端 = H(π) > 0，现文右端 = −H(π) < 0，恒等式为假 | 「− H(π̂)」改「+ H(π̂)」。负载结论（argmax E_π̂[log π] ⇔ argmin D_KL，因 H(π̂) 与 π 无关）不受影响；练习 1 要求学生补全此变形，现文会与学生正确推导矛盾，须同步无歧义 |
| F2 | **S2** | 9.5 §6 第三段 + 9.5 参考节 dr-grpo 条 | 「『aha moment』的证据强度也被同一篇论文重审（其 §4）」 | arXiv:2503.20783 原文该内容在 **§2.3**（节题即 "Aha Moment Already Appears in Base Models Including DeepSeek-V3-Base"，含自部署 685B、500 MATH 题、"Aha"/"wait"/"verify the problem" 关键词）；该论文 §4 是 "Closing Remarks"，无此内容 | 两处（§6 正文与参考节）「其 §4」均改「其 §2.3」 |
| F3 | **S2** | 9.4 §5 质检条 | 「RS 产出的数据仍有质量问题（论文举例代码 bug），用执行信号与 model-as-judge 过滤后再进 SFT 集」 | arXiv:2407.21783 §4.3.1 "Filtering training data with execution and model-as-judge signals"：RS 数据的判定难点原文明说 "Detecting these issues in our rejection-sampled data is not as straightforward as it is for our synthetic code data"，对 RS 数据用的是 **model-as-judge**（code correctness + code style 二判据、满分 2 保留）；execution 过滤归 synthetic code data 产线 | 改为「RS 数据用 model-as-judge 过滤（代码正确性 + 风格双判据）；执行信号过滤归合成代码数据」——「筛选不止 RM 一层」的论点保留 |
| F4 | **S2** | 9.4 §4 末段 | 「多样性塌缩作为外延担忧在文献中存在……（存疑待 research pass 补源）」 | spec 9.4 的合规路径是「research pass 补出处后与 STaR 原文区分标注」——出处未补，过程标记直接进入正文。STaR 原文登记的失效形态确为停滞（§3.2 逐字核对），多样性塌缩非其内容，章内区分本身正确；但正文残留生产 TODO 属协议缺口 | 二选一：research pass 补上 self-training 多样性塌缩出处后保留外延句；或删外延句、只留已核实的停滞诊断 |
| F5 | S3 | 9.4 §1 + 参考节 trl-repo 条 | TRL main trainer 目录枚举「dpo / grpo / kto / sft / reward / rloo / distillation」 | GitHub contents API 实测（2026-11-18）`trl/trainer/` 下 `*_trainer.py` 共 9 个：另含 `async_grpo_trainer.py` 与 `base_trainer.py`。核心断言「无专用 RFT trainer」成立 | 枚举前加「主要 trainer 族」或补全两项 |
| F6 | S3 | 9.5 §2 末段 | 「其 §2.2.4 末段：可读性差（回答不适合阅读、缺 markdown 结构）与语言混杂」 | §2.2.4 末段（"Drawback of DeepSeek-R1-Zero"）只点名 "poor readability, and language mixing"；「缺 markdown 结构」（"lack markdown formatting"）出自 §2.3.1 的 Readability 条 | 括注改为两节分挂或删「缺 markdown 结构」细节 |
| F7 | S3 | 9.5 §7 | 「同样的 800k 数据换成在 Qwen-32B-Base 上直接跑大规模 RL」 | R1 §4.1 的对照实验不消费那 800k 数据——RL 用自有数学/代码/STEM 数据训 >10K step；现文易被读作「800k 数据进 RL」。对照的真实控制变量是同一 32B 基座 | 改「同一 Qwen-32B-Base 基座上换成直接跑大规模 RL（数学/代码/STEM，超 10K step）」 |
| F8 | S3 | 9.5 §6 归因二 | 「GRPO 在 reward 已经平台化后长度仍持续上涨」 | 原文："GRPO tends to continually generate longer responses even when the reward improvement **slows down** (Plot 2)"——变慢，未断言平台化 | 「平台化」改「改善放缓」 |
| F9 | S3 | training/references.md | 「被引章节」未回写：christiano / trl-repo 缺 9.4；qwen3 / dapo / deepseekmath / verl 缺 9.5 | 两 writer 报告均已披露此待办（简报禁止 writer 改 references.md，归 controller/research pass） | controller 波末统一回写 |
| F10 | S3 | 9.4 §7 判据表 | 自称「全方法归格表」但无 RM-BT 行（3.2 §5 总表有 RM-BT offline 行） | 3.2 §5 一览表含 RM-BT；9.4 表只覆盖策略训练方法 | 可选：补 RM-BT 行或改表题为「策略训练方法归格表」 |

## 强制独立重推导（中间过程）

### 重推导 1：9.4 截断-再归一化（式 (1)）与判据表逻辑完备性

固定 $x$，采样 $y \sim \pi_\theta(\cdot \mid x)$，保留 iff $y \in A(x)$。保留样本的条件分布：$\Pr[y \mid y \in A] = \pi_\theta(y \mid x)\mathbb{1}[y \in A] / \Pr_{y' \sim \pi_\theta}[y' \in A] = \pi_\theta \mathbb{1}[y \in A]/p_A(x)$ —— 式 (1) 成立。best-of-$n$ 的阈值 $\tau(x)$ = 已采 $n$ 条中第 $k$ 大得分，是随机的；章内已显式声明「阈值本身是随机的，但条件化在已采样本上之后形式不变」——逐实现（per-batch）看是截断，边际化 $\tau$ 后是截断族的混合，不再是单一截断；但式 (2) 的 SFT 目标对数据分布是线性的（$\mathbb{E}_{y \sim D^{+}}[\log \pi]$），对 $\tau$ 取期望后 argmin 形式不变；「支撑集只缩不涨」对任意截断混合亦成立。**形式化合法**。

式 (2) 第二等号：**发现符号错误（F1，S1）**——正确恒等式为 $-\mathbb{E}_{\hat{\pi}}[\log\pi] = D_{\mathrm{KL}}(\hat{\pi}\Vert\pi) + H(\hat{\pi})$。

判据表逐行对 3.2 §5 定义（判据 = 目标的期望从谁采样，非数据来自谁）：SFT/DPO/KTO/SimPO/ORPO 静态集 → offline ✓；RFT 单轮（训练前采样、目标内无 $\pi_\theta$ 采样期望）→ offline，跨轮漂移单列 ✓（与 3.2 §5 的 RFT 边界情形登记逐字一致）；迭代 DPO 轮内 offline ✓（本书推演，写作方已申报，与判据自洽，**接受**）；PPO/GRPO/RLVR 目标期望取自 $\pi_{\theta_{\mathrm{old}}}$ → on-policy ✓。8 行全部归格正确；完备性上仅缺 RM-BT 行（F10，S3）。

### 重推导 2：9.5 R1 四阶段配方依赖图（对 arXiv:2501.12948 §2.3 逐阶段）

- 支线：R1-Zero = DeepSeek-V3-Base 直接 GRPO（§2.2.1 确认 GRPO 且引 Shao et al. 2024）→ 输出整理后喂主线冷启动（§2.3.1 "gathering DeepSeek-R1-Zero outputs in a readable format"）——跨线依赖属实 ✓
- 阶段 1 冷启动 SFT：输入 V3-Base + 「thousands of」长 CoT（四来源逐字对上：few-shot 长 CoT 示例 / 直接提示生成带反思验证的回答 / R1-Zero 输出可读化 / 人工后处理）；格式 |special_token|⟨reasoning_process⟩|special_token|⟨summary⟩，创建时滤除不宜阅读回答；目的 "to prevent the early unstable cold start phase" + Readability + Potential（"we observe better performance against DeepSeek-R1-Zero"）✓
- 阶段 2 推理 RL：冷启动 ckpt 上跑与 R1-Zero 相同的大规模 RL；域 = coding/math/science/logic；**language consistency reward 在此引入**（"we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT"），与 accuracy reward "directly summing"；消融 "a slight degradation in the model's performance"——章内式 (2) 与叙述逐字对上 ✓；收敛于推理任务止 ✓
- 阶段 3 拒采 SFT：阶段 2 ckpt 作数据发生器；部分数据用 DeepSeek-V3 作 generative judge（"feeding the ground-truth and model predictions into DeepSeek-V3 for judgment"）；滤除 mixed languages / long paragraphs / code blocks；约 600k 推理 + 约 200k 非推理（writing/factual QA/self-cognition/translation，复用 V3 管线，简单问候不带 CoT）= 约 800k；**"We fine-tune DeepSeek-V3-Base for two epochs"**——回基座重训逐字属实 ✓
- 阶段 4 全场景 RL：推理用规则 reward；通用用 RM（沿用 V3 偏好分布与管线）；helpfulness 只评 final summary、harmlessness 评整段（CoT + summary）✓
- 产物形态流「参数 → 数据 → 参数 → 参数」与依赖图一致；阶段 3 动机论文未给理由、章内已标「本书分析，非原文口径」 ✓
- 蒸馏支线：800k 直接 SFT 六个开源小模型、只做 SFT 无 RL（"leaving the exploration of the RL stage to the broader research community"）✓；Distill-32B 72.6/94.3/57.2、Distill-1.5B 28.9/83.9、Distill-7B 55.5 超 QwQ-32B-Preview、§4.1 对照 R1-Zero-Qwen-32B 47.0（>10K step）——全部逐数对上 ✓

## Verification log（抽查项 × 来源 × 结论）

来源获取方式：curl 拉取 ar5iv 全文（2501.12948 / 2203.14465 / 2407.21783 / 2501.12599 / 2505.09388 / 2503.20783 / 2503.14476）+ GitHub API（3 个代码锚点）。结论列：✓ = 逐字/逐数对上。

| 章 | 抽查项 | 来源（原文引句/数据） | 结论 |
| --- | --- | --- | --- |
| 9.4 | STaR 停滞诊断引句 | 2203.14465 §3.2："improvement ends when the model fails to solve new problems in the training set. This is fundamentally due to the fact that the algorithm cannot obtain any training signal from failed examples." | ✓ 逐字 |
| 9.4 | STaR 每轮从基座重训 | §3.1："once we collect a new dataset, we train from the original pre-trained model instead of continually training one model to avoid overfitting" | ✓ 逐字 |
| 9.4 | STaR 终止条件 plateau | §3.1："We keep repeating this process until the performance plateaus" | ✓ |
| 9.4 | STaR 式 (4) 潜变量 PG | 原文式 (1)(2) alttext：$p_M(y\mid x)=\sum_r p(r\mid x)p(y\mid x,r)$；$J=\sum_i \mathbb{E}[\mathbbm{1}(\hat{y}_i=y_i)]$；$\nabla J=\sum_i \mathbb{E}[\mathbbm{1}\cdot\nabla\log p_M(\hat{y}_i,\hat{r}_i\mid x_i)]$ | ✓ 逐式 |
| 9.4 | "this is the filtering process" + 两近似 | §3.1："the indicator function discards the gradient... this is the filtering process in STaR (Line 5). Thus, STaR approximates... (1) greedily decoding... to reduce variance (at the cost of potentially biased exploration), (2) taking multiple gradient steps on the same batch (similar to some policy gradient algorithms)" | ✓（含「定位为近似」口径） |
| 9.4 | rationalization 机制 + 去 hint 引句 | §3.2："we do not include the hint in its corresponding prompt, as if the model had come up with the rationale without the hint"；Figure 2 hint 模板 | ✓ 逐字 |
| 9.4 | few-shot 例子约 10 个 | §3.1："$P \ll D$ (e.g. $P=10$)" | ✓ |
| 9.4 | CommonsenseQA 72.5 / 60.0 / 36.6 / 73.0（30×） | Table 1 与摘要："performs comparably to a fine-tuned model that is 30× larger (72.5% vs. 73.0%)"；72.5−36.6=35.9 | ✓ 逐数 |
| 9.4 | 多样性塌缩归属边界 | STaR 全文无 diversity collapse 实证；章内标为外延担忧、无单一出处 | ✓ 归属正确；残留过程标记见 F4 |
| 9.4 | Llama-3 RS：K=10-30、最新 chat 策略、RM 选 best、Bai 2022 | §4.2.2："we sample K (typically between 10 and 30) outputs from the latest chat model policy (usually the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint for a particular capability) and use our reward model to select the best candidate, consistent with Bai et al. 2022" | ✓ 逐字 |
| 9.4 | system prompt 引导（代码例） | §4.2.2："In later rounds... we introduce system prompts to steer RS responses to conform with desirable tone, style, or formatting"；§4.3.1 代码例（注释、变量名） | ✓ |
| 9.4 | PagedAttention >2× | §4.2.2：预定义最大输出长度消 swap-out + 同 prompt 共享 KV cache 页，"a throughput improvement of over 2×"（alttext 核实） | ✓ |
| 9.4 | RS 质检 | model-as-judge 属实；**execution 归 synthetic code data** | F3（S2） |
| 9.4 | 六轮迭代、每轮重收偏好/SFT 数据 | §4.1.6："we apply the above methods in six rounds. In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models" | ✓ |
| 9.4 | TRL 无专用 RFT trainer | GitHub API `trl/trainer/`：dpo/grpo/kto/sft/reward/rloo/distillation + async_grpo + base | ✓ 结论成立；枚举缺 2 项（F5） |
| 9.4 | 式 (1)(3)(4)、判据表 | 重推导 1 | ✓；式 (2) 括注 F1（S1） |
| 9.5 | RLVR 式 (1) 与两类规则 | R1 §2.2.2：accuracy rewards（math boxed 格式判定、LeetCode 编译器按测试用例）+ format rewards（think 标签对） | ✓ |
| 9.5 | 不用神经 RM 的理由 | R1 §2.2.2："the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline"（outcome/process 均不用） | ✓ 逐字 |
| 9.5 | K1.5 残余攻击面（false positive、排除选择/判断/证明、N=8 猜测过滤） | 2501.12599 §2.1："false positive verification—where the model reaches the correct answer through an incorrect reasoning process"；"we exclude... multiple-choice, true/false, and proof-based questions"；"guess potential answers without any CoT... within N=8 attempts... removed" | ✓ 逐字 |
| 9.5 | K1.5 三条性质 | §2.1：Diverse Coverage / Balanced Difficulty / Accurate Evaluability | ✓ |
| 9.5 | R1-Zero：15.6→71.0（Fig 2）、cons@64 86.7（Table 2）、Fig 3 训练集长度增长、Table 3 aha（"Wait, wait. Wait."）、§2.2.3 模板无内容偏置、§2.2.4 末段缺陷 | 全部逐字/逐数对上（Figure 2/3 caption、Table 2/3、§2.2.3 "avoiding any content-specific biases"、§2.2.4 末 "poor readability, and language mixing"） | ✓（F6 小疵） |
| 9.5 | R1 四阶段全链 | 重推导 2 | ✓ 全部 |
| 9.5 | **language consistency reward 归属** | R1 §2.3.2 引入、定义、直接相加、消融轻微性能下降——逐字对上；K1.5 全文 grep "language consistency" = 0 命中 | ✓ **归 R1，非 K1.5**；章内归属纪律正确 |
| 9.5 | K1.5 式 (3)-(6)（mirror descent 目标/闭式解/对数比约束/平方代理/采样近似 τlogZ/经验均值依据 τ→∞/式 (6) 梯度） | 2501.12599 §2.3.2 alttext 逐式比对（含 $\frac{\tau}{2}$ 系数、(r−r̄) 组均值 baseline、"sampled from π_{θ_i} rather than on-policy, and an l2-regularization"） | ✓ 逐式 |
| 9.5 | K1.5 不用 value net 的理由（惩罚「走错但救得回」的中间步骤）、每轮重置优化器 | §2.3.2："selecting z'_{t+1} would be penalized... exploring z'_{t+1} is extremely valuable for training the model to generate long CoT"；"we also reset the optimizer at the start of each iteration" | ✓ |
| 9.5 | K1.5 长度惩罚式 (7)（λ 公式、min(0,λ)、全相等全零、加权相加、warmup 调度） | §2.3.3 alttext 与文字逐字对上 | ✓ |
| 9.5 | K1.5 curriculum/prioritized（1−s_i）/partial rollout | §2.3.4 / §2.6.2 逐句对上（固定 token 预算、replay buffer 续算、当前段 on-policy、段可剔出 loss、异步 worker、重复检测） | ✓ |
| 9.5 | long2short 四法、第二阶段配置、60.8@3272 token（8 次平均）、Fig 7 token 效率最高；长模型 77.5/96.2/94th/74.9 | §2.4、§3.4、摘要 逐字/逐数对上 | ✓ |
| 9.5 | K1.5 CoT RM vs Classic RM（98.5 / 84.4，各约 800k，RL 用 CoT RM，JSON 结论） | §2.3.5："value-head based reward model and collected approximately 800k data points"；"about 800k CoT-labeled examples... step-by-step reasoning process before providing a final correctness judgment in JSON format"；"Classic RM achieved an accuracy of approximately 84.4, while the Chain-of-Thought RM reached 98.5 accuracy. In the RL training process, we adopted the Chain-of-Thought RM" | ✓（% 号惯例读法，写作方已申报，**接受**） |
| 9.5 | Qwen3 四阶段（冷启动双过滤六类、minimize 样本与步数；3995 对 + GRPO + entropy 稳态 + 70.1→85.1/170 step；模式融合 /think /no_think/空 think 块/默认 thinking/末 flag 生效/thinking budget 副产物；General RL 20+ 任务五能力三类 reward） | 2505.09388 §4.1-4.4 逐句对上 | ✓ |
| 9.5 | Qwen3 蒸馏两相（off-policy 双模式打底、on-policy logits KL，teacher 32B/235B） | §4.5 逐句对上 | ✓ |
| 9.5 | Qwen3 thinking 评估采样 0.6/0.95/20 | 评估设置段："thinking mode... temperature of 0.6, a top-p value of 0.95, and a top-k value of 20" | ✓ |
| 9.5 | Dr.GRPO 长度双归因（Fig 6 Plots 1-2 同步增长、Plot 2/4 平台期与回落；Table 5 全类别含正确回答增长、错误更长因难题） | 2503.20783："their response length increases along with training reward (Plots 1 & 2)"；"continually generate longer responses even when the reward improvement slows down (Plot 2)"；"Dr. GRPO prevents the response length from growing wildly (Plot 2)"；Table 5 caption 与分析段 | ✓（F8 小疵） |
| 9.5 | Dr.GRPO 基座重审（685B 自部署、500 MATH、自反关键词、Qwen2.5 无模板强推理） | §2.3（非 §4 → F2）："hosting DeepSeek-V3-Base-685B ourselves and investigating its responses of 500 MATH questions"；"keywords such as 'Aha', 'wait', and 'verify the problem'" | ✓ 内容属实，§ 号错挂 |
| 9.5 | DAPO entropy 塌缩 Fig 2(b)、clip-higher 0.2/0.28 | 2503.14476 §3.1："the entropy of the policy decreases quickly as training progresses (Figure 2(b))... nearly identical"；"ε_low to 0.2 and ε_high to 0.28" | ✓ |
| 9.5 | verl `math_verify.py`（compute_score 签名、boxed 包装、双通道 parse、verify、4 worker、30s→0） | GitHub API 取 main 源文件逐行比对：`def compute_score(model_output, ground_truth, timeout_score=0, timeout=30.0)`、`ProcessPoolExecutor(max_workers=4)`、`LatexExtractionConfig`/`ExprExtractionConfig` 双通道 | ✓ 逐行 |
| 9.5 | understand-r1-zero `train_zero_math.py`（MATHOracle.get_reward → math_grader.boxed_reward_fn、多进程、超时记 0） | GitHub API：`from understand_r1_zero.math_grader import (... boxed_reward_fn)`；`mp_pool.apply_async` + `except TimeoutError: rewards.append(0.0)` | ✓ 逐行 |
| sweep | 3.2 §5 ↔ 9.4 §7 判据兑现 | 3.2 §5 定义（判据落在采样而非求值；「数据来自谁」与「目标期望取自谁」分离；RFT 边界情形登记 + 「判据表归 9.4」）与 9.4 §7 逐行一致 | ✓ |
| sweep | 8.4 ↔ 9.5 回链（式 (1)(2) GRPO、§5 1/\|o_i\|、§6 clip-higher 0.2/0.28 与 dynamic sampling、DAPO 承载） | 8.4 文本逐项在位 | ✓ |
| sweep | 9.1 ↔ 9.5（蒸馏 SFT 语义：loss masking/epoch/LR）；9.2 ↔ 9.5（RLVR 防 hacking 预告兑现）；9.4 ↔ 9.5（§6 过渡 → 9.5 式 (1)；9.5 §3 截断分布归 9.4 回链） | 各章文本 | ✓ |
| sweep | 9.1 抽查：Llama-3 Table 7（52.66/14.89/21.19、846.1=535.7+310.4） | llama3 Table 7 原文数据 | ✓ 逐数 |
| sweep | 术语一致性（on/off-policy/offline、RFT、CoT 首现「思维链（Chain-of-Thought, CoT）」、截断分布、组大小 G(GRPO)/k(K1.5 源文记号，局部已定义)） | 全 5 章 | ✓ |
| sweep | AI 味禁令词 + 扩展词表（10 词）+ 含糊量词 | 全 5 章 grep：禁令词零命中；「显著」等全部带出处或数值 | ✓ |

**覆盖率声明**：9.4 的数值/引句/工程口径断言 12/12 回源（STaR 7 项、Llama-3 5 项），TRL 目录 1/1；9.5 的五家论文断言全量回源（R1 四阶段逐阶段 + R1-Zero 6 项 + 蒸馏 5 组数；K1.5 式 (3)-(7) 逐式 + 9 项口径；Qwen3 四阶段逐条；Dr.GRPO 4 项；DAPO 2 项），代码锚点 3/3 GitHub API 实测；9.1-9.3 为 sweep 抽查（跨章回链 5 组 + Table 7 一组 + 文风全扫），非逐字重审（批 1 evaluator 职责）。

## 写作方存疑项裁定

task-32：① 9.4 多样性塌缩归属——处理方向合规（STaR 原文确无此实证），但正文残留过程标记 → F4（S2）。② 9.3 TRL IPO token 平均——实现层事实已注明代码注释口径，**接受**。③ 9.3 ORPO 梯度显式负号——独立链式法则复核：$d(-\log\sigma(u))/du = -1/(1+e^u)$、$d\log\mathrm{odds} = \nabla\log P/(1-P)$，章内式 (20) 正确，**接受**。④ 9.4 §7 迭代 DPO 落 offline——与 3.2 §5 判据自洽，**接受**。⑤ 9.3 式 §2 的 β·log Z(x) 回代——标准两步代数，**接受**。

task-33：① K1.5 84.4/98.5 无 % 号——accuracy 按百分数读是标准惯例，**接受**。② Qwen3 采样参数只标「评估设置口径」——已核实数值位于评估设置段，标法诚实，**接受**。③ R1 阶段 3 回基座重训动机标「本书分析」——原文确无理由，**接受**。④ thinking budget 指令改写未照引——改写忠实（原文字符串含换行），**接受**。⑤「涌现=选择压力」标「本书读法」——**接受**。

## 机械检查

- `bash scripts/lint-chapter.sh`：9.4、9.5 各 **LINT OK**；全 5 章（9.1-9.5）**LINT OK × 5**（exit 0，2026-11-18 本机重跑）。
- 模板：两章均有 metadata 块 / 总结表 / 练习 4 题 / 参考链 `../references.md`；9.4 有 RFT 循环 ASCII 图 + 判据表，9.5 有 R1 四阶段依赖图；字数 3557 / 5654 CJK（≥3500 达标）。
- arXiv ID 全部存在于 references.md（lint 检查项 4 通过）；证据等级标注齐全。

## 处置建议

1. 修 F1（一字符：−H → +H）、F2（两处 §4 → §2.3）、F3（质检归属改写）、F4（补源或删句）——S1/S2 必修；
2. 9.4 修复后由 evaluator 复审该章（S1 协议）；F5-F10 累计入下波清零（F9 已在 controller 待办）；
3. 复审重点只需重跑 lint + 核对 F1/F3 改动行。

## 修复复审记录（2026-11-18 复审轮）

复审方式：重读两章当前全文（非 diff），逐条对照初审 findings；修复触及处回原文复核；lint 重跑。

| # | 判定 | 当前文本证据 |
| --- | --- | --- |
| F1（S1） | **ADDRESSED** | 9.4 §2 式 (2) 括注现为「$-\mathbb{E}_{\hat{\pi}}[\log \pi] = D_{\mathrm{KL}}(\hat{\pi} \Vert \pi) + H(\hat{\pi})$」——符号已正（自验：π̂=π 时两端同为 H(π)）。练习 1 原文只要求「写出 −E[log π] 到 D_KL 的恒等变形」、不含 H 的符号陈述，与修正后正文无歧义一致；「熵项与 π 无关 ⇒ argmax⇔argmin」链完整 |
| F2（S2） | **ADDRESSED** | 9.5 §6「重审（其 §2.3）」、参考节 dr-grpo 条「基座自反关键词重审（§2.3）」——两处均已改；grep「其 §4」残留命中均为 Qwen3 §4.1-4.4 / §4.5 与 R1 §4.1 的合法引用 |
| F3（S2） | **NOT ADDRESSED（残留一处）** | 9.4 §5 质检条已正确改写（「对 RS 数据用 model-as-judge 过滤……执行信号过滤归合成代码数据产线」，新增的双判据满分 2、剔除难题样本回退与改写对策均与 2407.21783 §4.3.1 原文对上，无新错误）；但**总结表「Llama-3 RS」行仍写「执行+judge 质检」**，与修正后正文自相矛盾——同一 S2 的残留实例。修复：该单元格改「model-as-judge 质检（§4.3.1）」（该行出处「§4.2.2」可同时补「质检 §4.3.1」） |
| F4（S2） | **ADDRESSED** | 9.4 §4 的外延担忧段（含「存疑待 research pass 补源」）已整段删除；grep「多样性塌缩/存疑待」零命中；§4 收束于实证锚点段，无悬空引用；metadata 与总结表无残留 |
| F5（S3） | **ADDRESSED** | 9.4 §1 与参考节均已加「主要 trainer 族」限定语 |
| F6（S3） | **ADDRESSED** | 9.5 §2 现为「可读性差与语言混杂……（「缺 markdown 结构、回答不适合阅读」的具体描述见其 §2.3.1 的 Readability 条）」——细节重挂 §2.3.1 |
| F7（S3） | **ADDRESSED** | 9.5 §7 现为「在同一 Qwen-32B-Base 基座上换成直接跑大规模 RL（数学/代码/STEM 数据，超 10K step，不消费那 800k）」——歧义消除且显式排除数据复用 |
| F8（S3） | **ADDRESSED** | 9.5 §6「reward 改善放缓后长度仍持续上涨」；总结表同步改「reward 改善放缓仍涨」 |
| F9（S3） | **ADDRESSED** | references.md 已回写：christiano/trl-repo 含 9.4；qwen3/dapo/deepseekmath/verl 含 9.5（controller 完成） |
| F10（S3） | **ADDRESSED** | 9.4 §7 表题改「策略训练方法的归格表」 |

新问题检查（只查修复触及处）：F3 正文改写的三处新增细节（双判据满分 2、难题样本剔除回退、改写最难档对策）已对 2407.21783 §4.3.1 原文逐句复核无误；其余修复均为删改/换词，未引入新断言。lint 重跑：9.4、9.5 各 LINT OK（exit 0，2026-11-18）。

**复审结论：暂不放行**——S1 已销、S2 余 F3 一处总结表单元格残留。该修复为单格机械改写（「执行+judge 质检」→「model-as-judge 质检（§4.3.1）」），改完即放行，无需全章复审，复核该单元格即可。
