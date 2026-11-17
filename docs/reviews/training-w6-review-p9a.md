# training W6 批 1（9.1–9.3）对抗审查报告

日期：2026-11-18。审查人：独立 evaluator（fresh context，与写作 agent 无共享上下文）。立场：假设有错，逐条回一手来源。
审查对象：`training/09-post-training/01-sft.md`、`02-rlhf-pipeline.md`、`03-dpo-family.md`。
依据：STYLE_GUIDE.md、docs/reviews/2026-11-15-style-audit/rubric.md、training/references.md、training spec §5（9.1–9.3 大纲）。

**结论：不放行。** S1×0、S2×6、S3×9。S2 全部可局部修复（节号指针、量表口径、练习题干、TRL 走读两处），无重写级问题；事实主干（全部公式、数值、引语、arXiv ID、repo 路径）经一手来源逐条核实无误。

---

## 1. Findings 表

### S2（必须先修）

| # | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 9.3 练习 1(e) | 「设 $r$ 乘常数 $c>0$、$\beta$ 同除以 $c$，证明 $\pi^{\ast}$ 不变——reward 的尺度自由度被 $\beta$ 吸收」 | 题干错误。$\pi^{\ast} \propto \pi_{\mathrm{ref}}\exp(r/\beta)$；$r' = cr$、$\beta' = \beta/c$ 时 $r'/\beta' = c^2 r/\beta$，$\pi^{\ast}$ 改变（除非 $c=1$）。保持不变的变换是 $r \to cr$、$\beta \to c\beta$（同乘），DPO 原文式 (4) 直接可见（arXiv:2305.18290，ar5iv 核对） | 「$\beta$ 同除以 $c$」改为「$\beta$ 同乘以 $c$」；或改为证伪式提问（问 c² 从哪来） |
| S2-2 | 9.1 §6（TRL 走读段首） | 「数据集按字段三分流：`text` 走纯语言建模；`prompt`/`completion` 分栏走 `_collate_prompt_completion`；`messages` 走 `_collate_language_modeling`」及后文「TRL 的两个 collator 都带 `add_special_tokens=False` 注释与开关」 | TRL main（2026-11-18，GitHub API 拉取）：`_collate_prompt_completion` 与 `_collate_language_modeling` 只存在于 `DataCollatorForVisionLanguageModeling`（sft_trainer.py L380/L431，视觉路径）；text-only 数据在 `_prepare_dataset`/`tokenize_fn`（L1376-1441）预处理期生成 `completion_mask`（`[0]*prompt_len + [1]*...`）与 `assistant_masks`，labels 在准备期折入，`DataCollatorForLanguageModeling`（L89）只对齐 padding（其 docstring 明言 "the SFTTrainer takes care of this during dataset preparation"）。`add_special_tokens=False` 注释只出现在 VLM collator 的 3 个调用点 | 把三分流走读明确标注为 VLM collator（`_collate_*`）+ text-only 路径在 `_prepare_dataset`/`tokenize_fn` 期完成同等掩码语义；「两个 collator 都带」改为「VLM collator 的三处调用点带」 |
| S2-3 | 9.1 §2 | 「`assistant_only_loss` 开启时若模板的 `{% generation %}` 标记把 assistant 的终止 token 划到生成区之外，`SFTTrainer` 初始化即报错并提示改模板」 | TRL main sft_trainer.py L1022-1030：该检查（`is_chat_template_stop_token_trained`）触发的是 `logger.warning`（"the model may not learn to stop… or edit the chat template"），训练继续；硬错误（ValueError）只在 `get_training_chat_template` 无法补 generation 标记时抛出（trl/chat_template_utils.py L1509） | 「初始化即报错」改为「初始化即警告（训练继续，风险在推理期暴露）；模板完全缺标记且无法自动修补时才报 ValueError」 |
| S2-4 | 9.2 §2、§3.1、参考列表 | InstructGPT 的 K=4–9 标注协议、RM 目标式 (1)、脚注 3（打散过拟合）出处标「§3.3」 | InstructGPT 实际结构（ar5iv 全文核对）：§3.2 Dataset、§3.3 Tasks、§3.4 Human data collection、§3.5 Models。K=4–9、式 (1)、脚注 3 全在 §3.5 "Models" 的 "Reward modeling (RM)" 段；§3.3 是任务来源，无 RM 内容 | 三处「§3.3」改「§3.5」；参考列表同步 |
| S2-5 | 9.2 §2 | RM 精度 72.4±0.4% / 69.6±0.9%（五折交叉验证）出处标「其 App. C.2」 | 该实验报告于正文 §4.1 末段与 App. E.2「Reward model generalization across sets of labelers」（"inter- and intra-group validation accuracies … 72.4 ± 0.4%, and 69.6 ± 0.9%"，五折 CV 原文）；App. C.2 只含 RM 训练超参（E.2 原文自己引用 C.2 仅为超参复用） | 「App. C.2」改「§4.1 与 App. E.2」 |
| S2-6 | 9.1 §3 | LIMA 质量消融「同来源未过滤的 2000 条比过滤版显著差（0.5 分制差 0.5）」 | LIMA §5 + App. D：消融评估用 ChatGPT 打 **1–6 分 Likert**，原文为 "a significant 0.5 point difference"（Figure 5/6 语境）——是 6 分制上差 0.5，不是「0.5 分制」 | 「0.5 分制差 0.5」改「6 分制上显著差 0.5 分」 |

### S3（记录，下波清零）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S3-1 | 9.1 §5、9.2 §5.1 | 「KL 系数加大 100 倍也修不回（App. E.11）」 | 该结论在 App. E.6（Figure 34："even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions [persist]"）与 §4.2；E.11 只含 pretraining ratio 消融与 8×/2× 时间账 | 指针改「App. E.6 Figure 34 / §4.2」 |
| S3-2 | 9.2 §2 | 「训练组对 held-out 标注员 77.3 ± 1.3%」 | 原文（§3.4）："training labelers agree with each-other 72.6±1.5%…, while for held-out labelers this number is 77.3±1.3%"——77.3 是 held-out 组**内部**一致率，非训练组对 held-out 组的跨组一致率 | 改为「held-out 标注员组内部 77.3」 |
| S3-3 | 9.2 §2 | 「一次排序任务产出全部 C(K,2) 个成对判断，平局丢弃」 | 论文只说标注侧 "Ties are encouraged in cases where two outputs seem to be of similar quality"（App. B.4）；RM 训练对 tie 的处置未写 | 删「平局丢弃」或标注为推定 |
| S3-4 | 9.2 §5.4 | 「维护 actor 的指数滑动平均副本，报告与评估用 EMA 权重」 | 原文（App. C.4）只有 "Exponential moving averages of the weights are applied, with a decay rate of 0.992"——未指明 actor/critic，亦无「评估用 EMA」之说；且伪码把 EMA 列在 critic 行、正文归 actor，自相不一致 | 删「报告与评估用 EMA 权重」；伪码与正文的归属对齐（按原文保守写「policy 侧」或并注明原文未指明） |
| S3-5 | 9.2 §3.4 | 「（其 §3.2.2，原话见 §4.1）」 | reward hacking 定义原话在 Llama-2 §3.2.2 ✓；但本章 §4 无 §4.1 小节，「原话见 §4.1」为畸形指针（应指本章 §4 开头） | 改「原话见本章 §4 开头」 |
| S3-6 | 9.2 §1 图注 | 「数据量递减 13k → 33k → 31k」 | 13k → 33k 是递增（InstructGPT §3.2/Table 6）；只有 33k → 31k 递减 | 改「数据量 13k → 33k → 31k（SFT→RM 扩、RM→PPO 微缩）」或删「递减」 |
| S3-7 | 9.2 §2 | 「比 InstructGPT 的 33k prompt 大两个数量级」 | 100 万 / 3.3 万 ≈ 30 倍 ≈ 1.5 个数量级（且一为二元比较数、一为 prompt 数，口径已不同） | 改「大约 30 倍（pair 对 prompt 口径）」 |
| S3-8 | 9.3 §8 | 「其 Figure 2(b)(c) 的分布证据」（γ-margin 分析） | SimPO §4.3 正文确写 "Figure 2(b) and Figure 2(c)"，但该图的 caption 与内容（γ 扫描）实为 **Figure 3**(b)(c)——论文自身笔误，章节照抄了笔误；Figure 2(b)(c) 是 LN 的 Spearman 图 | 改指 Figure 3(b)(c) 并加注「原文 §4.3 正文误作 Figure 2(b)(c)」 |
| S3-9 | 9.2 §5.2 | 「另有 loss 侧注入点（`ppo_loss` 的 `use_kl_loss`）」 | veRL main（2026-11-18）：`use_kl_loss` 是 actor config 选项（`verl/workers/config/actor.py` L168，默认 False，`kl_loss_coef=0.001`、`kl_loss_type="low_var_kl"`）；loss 侧已重构为 `compute_policy_loss_*` 函数族，无 `ppo_loss` 函数 | 改「actor config 的 `use_kl_loss`（`verl/workers/config/actor.py`）」 |

**S1：0 项。** 全部公式（含推导重章两章的全部编号式）、数值、arXiv ID、repo 路径、引语经一手来源核实无一虚构或张冠李戴。

---

## 2. 强制独立重推导记录（逐步中间过程）

### 2.1（9.2）RM 目标与逐 completion 梯度

- BT 模型 $P(y_w \succ y_l \mid x) = \sigma(r_w - r_l)$，NLL $-\log\sigma(r_w - r_l)$；InstructGPT 式 (1) 的聚合 $-\frac{1}{\binom{K}{2}}\mathbb{E}_D[\log\sigma(r_w - r_l)]$ 与论文逐字对上（ar5iv:2203.02155 §3.5）。
- 式 (2) 独立重推：对胜出对 $(i \succ j)$，$\partial[-\log\sigma(r_i - r_j)]/\partial r_i = -\sigma(-(r_i - r_j)) = \sigma(r_i - r_j) - 1$；对落败对 $(j \succ i)$，$\partial[-\log\sigma(r_j - r_i)]/\partial r_i = +\sigma(r_i - r_j)$。汇总乘 $1/\binom{K}{2}$——与章节式 (2) 完全一致。模长 $= \sigma(\text{反向差}) = $ 模型判反概率 ✓。
- margin 变体：$-\log\sigma(\Delta r - m)$，$\partial/\partial\Delta r = \sigma(m - \Delta r)$；$\Delta r = 0$ 处模长 $\sigma(m(r))$；$0 < \Delta r < m$ 区间梯度仍 $\gt \sigma(0) = 0.5$ 量级——「判对未达 margin 仍满力」成立 ✓（练习 2 可解）。
- 练习 1 数值验算（$K=4$、$r=(2,1,0.9,0)$）：$\partial L/\partial r = (-0.106, -0.079, +0.073, +0.113)$，$y_3$ 收双向最大合力（毛力 0.436）——与题目预期（争议样本力平衡）自洽 ✓。

### 2.2（9.2）Goodhart 形式化 outline

式 (4) 是恒等式：$\widehat{G} - G = (\mathbb{E}_\pi[r^\ast + \varepsilon] - \mathbb{E}_{\mathrm{ref}}[r^\ast + \varepsilon]) - (\mathbb{E}_\pi[r^\ast] - \mathbb{E}_{\mathrm{ref}}[r^\ast]) = \mathbb{E}_\pi[\varepsilon] - \mathbb{E}_{\mathrm{ref}}[\varepsilon]$ ✓。机制论证（优化器是 $\varepsilon$ 的正向选择器、$d \to 0$ 时 gap 归零、驼峰形态只给 outline 不拟合）均为本书标注的推导 outline，无断言越界 ✓。

### 2.3（9.3）DPO 闭式解全链（逐步）

1. 逐 prompt 分解：$J(\pi)$ 无跨 prompt 耦合 ⇒ 固定 $x$ 逐点变分 ✓。
2. 变 max 为 min：乘 $-1/\beta$ 得 $\min_{\pi(\cdot|x)} \mathbb{E}_\pi[\log(\pi/\pi_{\mathrm{ref}}) - r/\beta]$ ✓。
3. 配分函数补齐：$\log\frac{\pi}{(1/Z)\pi_{\mathrm{ref}}e^{r/\beta}} - \log Z = \log\frac{\pi}{\pi_{\mathrm{ref}}} - \frac{r}{\beta}$（$Z(x) = \sum_y \pi_{\mathrm{ref}} e^{r/\beta}$ 为归一化常数，对 $\pi$ 是常数）✓ 与式 (2) 一致。
4. Gibbs 取等：$\min[\mathrm{KL}(\pi\Vert\pi^\ast) - \log Z]$ 在 $\pi = \pi^\ast$ 处取得 ⇒ 式 (3) ✓。
5. 最优值回代：$\pi = \pi^\ast$ 时逐 prompt 目标值 $= -\beta(-\log Z) = \beta\log Z(x)$ ✓（写作方标注的补步，代数正确）。
6. 重参数化：对式 (3) 取 log 移项得 $r = \beta\log(\pi^\ast/\pi_{\mathrm{ref}}) + \beta\log Z(x)$ ✓（式 (4)）。
7. BT 代入：$r_1 - r_2$ 中 $\beta\log Z(x)$ 逐字相消 ✓；$\sigma(v) = 1/(1+e^{-v})$ 展开得式 (5)，与 DPO 原文式 (6) 逐项一致 ✓。
8. MLE 得式 (6) = 原文式 (7) ✓；梯度式 (7) = 原文式 (8)：$\nabla L = -\mathbb{E}[\sigma(-u)\nabla u]$，$u = \beta(h_w - h_l)$，展开系数 $\beta\sigma(\hat r_l - \hat r_w)$ ✓。
9. 等价类论证（Theorem 1：$f(r) = r - \beta\log Z$ 与 $r$ 同类且具重参数化形式）与原文 proof sketch 一致 ✓。

### 2.4（9.3）IPO 修正链

- $\Psi = \mathrm{id}$ 时 ΨPO 目标化为 $\max_\pi \mathbb{E}_\pi[g(y)] - \tau\mathrm{KL}$（$g(y) = p^\ast(y \succ \mu)$），同 §2 变分得 $\pi^\ast \propto \pi_{\mathrm{ref}}\exp(g/\tau)$；对数比差 $h^\ast(y,y') = \tau^{-1}(g(y) - g(y'))$ ✓（式 (9) = 原文式 (12)）。
- 求根→最小二乘式 (13)（Theorem 2 唯一性需 $\mathrm{Supp}(\mu) = \mathrm{Supp}(\pi_{\mathrm{ref}})$ ✓）。
- 采样损失式 (16)（Prop 3：与式 (13) 差一个与 $\pi$ 无关常数，证明用 $h_\pi$ 可加分解 + $y,y'$ i.i.d. ✓）。
- 对称合并：每对贡献 $(h - \tau^{-1})^2 + h^2$，除以 2；配方 $(h-a)^2 + h^2 = 2(h - a/2)^2 + a^2/2$（$a = \tau^{-1}$）⇒ 经验损失 $(h_\pi - \frac{1}{2\tau})^2$ + 常数 ✓（式 (10) = 原文式 (17)，ar5iv:2310.12036 逐字核对）。
- $\tau$ 方向自洽性（强正则 ⇒ 小靶值）✓。
- 另：ORPO 式 (20) 独立重推——$-\log\sigma(u)$ 导数贡献 $-\sigma(-u) = -\frac{1}{1 + \mathrm{odds}_w/\mathrm{odds}_l}$，$\partial\log\mathrm{odds}/\partial\theta = \nabla\log P/(1-P)$ ⇒ $\nabla L_{OR} = -\delta(d)h(d)$；原文式 (8) 写 $\nabla L_{OR} = \delta(d)h(d)$（无显式负号）实为幅度形式/原文符号省略，章节保留显式负号并注明——处理正确 ✓（写作方存疑项 3 裁定：维持原写法）。

---

## 3. Verification log（抽查项 × 来源 × 结论）

覆盖率口径：三章全部编号公式（9.1 式 (1)-(3)、9.2 式 (1)-(7)、9.3 式 (1)-(20)）100% 核对；全部 arXiv ID（15 个）100% 经 arXiv API 复核；正文中带具体数字的实证断言约 58 条，抽查 46 条（79%），全部回一手原文；代码行级断言 31 条全部经 GitHub API 拉 main 源文件核对（100%）；STYLE_GUIDE 机械项 + rubric A/B 词表全量机扫。

| # | 抽查项 | 来源（一手） | 结论 |
| --- | --- | --- | --- |
| 1 | 15 个 arXiv ID↔标题 | export.arxiv.org API | 全对（2203.02155 InstructGPT、2305.11206 LIMA、2305.18290 DPO、2310.12036 IPO、2402.01306 KTO、2405.14734 SimPO、2403.07691 ORPO、2407.21783 herd、2307.09288 Llama-2、1706.03741 Christiano、1707.06347 PPO、2405.11143 OpenRLHF、2409.19256 HybridFlow、2503.14476 DAPO、2501.12948 R1）✓ |
| 2 | InstructGPT 13k/33k/31k、K=4–9、脚注 3、72.6/77.3 一致率、72.4/69.6 精度、75% 筛选门槛（App B.1）、RM 初始化数据集清单、1 epoch/lr 9e-6/batch 64/2304 对上限、6B RM 服务全档、175B RM 不稳定、unembedding→scalar 头 | ar5iv:2203.02155 全文 | 数值全对 ✓；节号指针三处错（S2-4、S2-5） |
| 3 | SFT 超参（5.03e-6/9.65e-6、16 epoch、cosine→10%、无 warmup、dropout 0.2、batch 8/32、RM score 选型）、SFT-init 10% 预训练混入（App C.3）、γ=27.8、8×数据、时间翻倍（E.11）、β=0.02（Stiennon 口径）、β 扫描 0.01–0.02（E.7/Fig.36）、8.05e-6 发散阈（E.9）、512 prompts/8×64/单 inner epoch/EMA 0.992/clip 0.2/T=1/≤1k token/GAE 不打折（C.4）、value lr 9e-6/5e-6、85±3%/71±4%、毒性 −25%（respectful）与「更有毒」（Fig.39）、TruthfulQA「small but significant、default 行为」、Winogender/CrowS-Pairs 无改善、hedging 归因（§4.3）、40 名标注员 Upwork/ScaleAI（§3.4）、§5.2 对齐边界、KL×100 修不回（E.6/Fig.34）、512k episodes（E.6/Fig.35）、alignment tax 术语 | 同上 | 全对 ✓；两处指针错（S3-1）、EMA 归属越界（S3-4） |
| 4 | LIMA：750/250、43%/58%/65%、Alpaca 52k 对照、16 倍数量平台、质量 0.5 分差、多样性 StackExchange>wikiHow、多轮 30 条 → 45.2%→76.1%、15/42→1/46、6/10 三轮内脱轨、15 epoch、lr 1e-5→1e-6 线性、无 warmup、wd 0.1、batch 32、50 条留出集 5–10 epoch 手挑、「perplexity 不相关」、Superficial Alignment Hypothesis 原文句 | ar5iv:2305.11206 全文 | 全对 ✓；量表口径错（S2-6） |
| 5 | Llama-3 herd：SFT lr 1e-5、8.5K–9K 步（§4.1.3）、Table 7（52.66/14.89/21.19、846.1=535.7+310.4、4.7 轮、context+final response 表注）、margin 删除引语逐字（§4.1.2）、拼接+乱序近似（§4.1.2）、edited 三元组（§4.2.1）、「significantly better or better」过滤（§4.2.1）、DPO lr 1e-5/β=0.1/格式化 token 屏蔽/NLL 0.2（§4.1.4）、「less compute…better on IFEval」逐字（§4.1.4）、最新批数据（§4.1.4/§4.2.1）、模型平均（§4.1.5）、六轮（§4.1.6）、405B 预训练峰值 8e-5（§3.4.1）、2.7M 合成代码（§4.3.1） | ar5iv:2407.21783 全文 | 全对 ✓ |
| 6 | Llama-2：SFT lr 2e-5/batch 64/wd 0.1/seq 4096/cosine（§3.1）、margin 式 (2)+Table 27+A.3.3「regresses performance on similar samples」逐字、四档强度、>100 万二元比较（§3.2.1）、RM 1 epoch/lr 5e-6(70B)/1e-5（§3.2.2）、reward hacking 定义引语逐字（§3.2.2）、开源数据混合动机（§3.2.2）、70B 预训练峰值 1.5e-4（Table 1） | ar5iv:2307.09288 全文 | 全对 ✓ |
| 7 | DPO：式 (3)(4)(5)(6)(7)(8) 逐式、A.1 链、「Following prior work」先例（Peng 2019/Peters & Schaal 2007 在引文表）、Theorem 1/Lemma 1/2、附录 Table 3 退化消融、π_ref=π_SFT/chosen-MLE 替代、IMDb 前沿「strictly dominates PPO…even PPO-GT」逐字、TL;DR 61% vs 57%（temperature 0） | ar5iv:2305.18290 全文 | 全对 ✓ |
| 8 | IPO：ΨPO 式 (6)、Prop 1（logit 特例）、§4.2 过拟合诊断（欠拟合 RM 正则化器论）、式 (8) 总偏好、式 (12) 求根、式 (13) 最小二乘、Prop 3（可加分解+i.i.d. 对称）、式 (17) 靶值 1/(2τ)、Theorem 2 支撑集条件、Prop 4（App B） | ar5iv:2310.12036 全文 | 全对 ✓ |
| 9 | KTO：KT 式 (4)（α=0.88/λ=2.25 中位数）、sigmoid 替换、HALO 与 Theorem 3.5、式 (8)（λ_y 非负脚注）、ẑ_0 错配对+clamp(0)+不回传（§4.1）、microbatch≥2、lr 5e-6（DPO 5e-7 的 2–10 倍）、β 两档 [0.01,0.10]/[0.10,1.00]、λ_D n_D/(λ_U n_U)∈[1,3/2]、Figure 3（KTO 单用匹配 SFT+DPO）、one-y-per-x 消融 | ar5iv:2402.01306 全文 | 全对 ✓ |
| 10 | SimPO：§2.2 两动机逐字、~50% 训练集排序一致（Fig.4(b)）、同期工作随机准确率转述、式 (4)(5)(6)、home advantage 脚注、LN 消融（正对差/强 Spearman/短 chosen 负差分）、γ 倒 U 与分布证据、β∈[2.0,2.5]/γ∈[0.5,1.5]、Table 8（Llama-3-Instruct 2.5/1.4）、摘要 +6.4/+7.5/72.4%/59.1% | ar5iv:2405.14734 全文 | 全对 ✓；图号照抄了原文笔误（S3-8） |
| 11 | ORPO：§3 OPT-350M/HH-RLHF 双升引语逐字、式 (3) 平均 log 似然口径、式 (4) odds、式 (6)(7)、式 (8)-(10) 幅度形式、§4.3「sensible choice」论证、摘要 12.20%/66.19%/7.32 | ar5iv:2403.07691 全文 | 全对 ✓ |
| 12 | PPO 论文：自适应 β「d < d_targ/1.5 ⇒ β/2；d > 1.5 d_targ ⇒ 2β」（§4，PDF 抽取核对）、「KL penalty performed worse than the clipped surrogate objective」逐字 | arxiv.org/pdf/1707.06347（PDF→文本） | 全对 ✓ |
| 13 | TRL main（2026-11-18，GitHub API）：sft_config.py（packing_strategy 默认 bfd/bfd_split/wrapped 三档、max_length 1024、truncation_mode 仅 keep_start）、sft_trainer.py（`_collate_*` 归属、completion_only_loss 自动推断 L1033-1036、assistant_only_loss 与 {% generation %}、is_chat_template_stop_token_trained L1022 为 warning、vision 禁 packing L792、`add_special_tokens=False` 三处注释、seq_lengths→padding-free docstring、fused_lm_head 存在）、dpo_trainer.py（_compute_loss L1366、ref sum(dim=1) L1351/1422、chosen_logratios L1431、reverse_kl→delta_score L1434/1464、sigmoid L1469、IPO 分支 L1474-1485 含「作者确认」注释）、kto_trainer.py（kl 计算 L1507-1508、两支 1-sigmoid L1517/1533、desirable/undesirable_weight）、experimental/orpo/orpo_trainer.py（log1mexp L87、log_odds L689-691、losses=beta*ratio L694、loss=nll−losses.mean() L844、average_log_prob=True L801）、trl/trainer/orpo_trainer.py main 404 + 迁移 commit d705ac4d（2026-02-10「Remove deprecated ORPO after moved to experimental #5050」） | api.github.com（raw + commits search） | 除 S2-2/S2-3/S3-9 外全对 ✓；ORPO 迁移与 9.3 引用相容（§9/§10/参考列表均指向 experimental 路径）✓ |
| 14 | OpenRLHF main：loss.py PairWiseLoss margin 开关（L301-313）、PolicyLoss clip_eps_low/high（L123-135/201）、model.py score=nn.Linear(hidden,1,bias=False)（L186/254）、eos_indices fliplr-argmax+gather（L214/235）、rm_trainer.py concatenated_forward（L321）与 pair acc（L175） | api.github.com | 全对 ✓ |
| 15 | veRL main：trainer_utils.py apply_kl_penalty（token_level_rewards = token_level_scores − beta·kld L57、beta=kl_ctrl.value、kl_ctrl.update）、core_algos.py AdaptiveKLController（clip(KL/target−1, ±0.2)、mult=1+err·n_steps/horizon L172-174）= 式 (7) 逐符号 | api.github.com | 全对 ✓；use_kl_loss 表述略偏（S3-9） |
| 16 | eric-mitchell/direct-preference-optimization trainers.py：pi_logratios/ref_logratios、「also known as h」注释、IPO「Eq. 17」、(logits−1/(2β))²、reference_free 开关与均匀参考注释 | api.github.com | 全对 ✓ |
| 17 | chat template 实例：Qwen2.5-7B-Instruct tokenizer_config.json（`<|im_start|>role\n…<|im_end|>\n`、generation prompt `<|im_start|>assistant\n`、eos=`<|im_end|>`、无 generation 标记）、Meta-Llama-3-8B-Instruct（经 NousResearch 同名镜像：`<|start_header_id|>role<|end_header_id|>\n\n`+`<|eot_id|>`、BOS 模板注入、add_generation_prompt 追加不闭合 assistant 头、instruct eos=`<|eot_id|>`） | hf-mirror.com（HF 同源镜像，raw JSON） | 全对 ✓ |
| 18 | 跨章引用接口：models 8.3 式 (1)（loss masking）、models 9.4 §6 式 (10)（LoRA，视频控制章）、training 3.2（BT 式 (3)(4)、平移不可识别 §3、在线/离线 §5、TRL 走读 §6）、7.1 §6 式 (7)(8)（η_pad/η_pack） | 本地文件核对 | 全部存在且口径一致 ✓ |
| 19 | 练习可解性：9.2 练习 1 数值手算（见 §2.1）；9.3 练习 3 两口径账（DPO u=0→loss ln2≈0.693；SimPO −0.5→≈0.974；双升 0.1 时 DPO 恶化至 ≈1.70、SimPO 不变——「拉长 rejected 降损」通路成立） | 独立重算 | 自洽 ✓；9.3 练习 1(e) 题干错（S2-1） |

## 4. 教学审查

- 9.3 推导链无断步：变分→配分补齐→Gibbs→重参数化→BT 代入→MLE 每步有形变依据；「$Z(x)$ 相消 = 平移不可识别兑现」的回接与 3.2 §3 文本核对一致（gauge 伏笔确在 3.2 §3 登记）✓。IPO 的 ΨPO→恒等映射→求根→最小二乘→对称合并链条完整，符号 $\Psi/\mu/\tau/\rho/g$ 均有定义 ✓。
- 9.2 式 (1)→式 (2) 的 sigmoid 求导未逐步展开，但属一步微分且有「3.2 式 (5) 的聚合形式」回指，可接受；三个推论的论证链完整 ✓。
- 前置依赖核对：3.1/3.2/7.1/8.2/8.3/8.4/8.5/models 8.3/models 9.4 均存在且被引小节核对一致（log 项 #18）；9.4/9.5 为前瞻指针，符合 W6 并行写作约定 ✓。
- 无循环解释、无未定义符号进入公式（逐式符号表齐全）✓。
- 唯一断步级问题：9.3 练习 1(e) 题干错误（S2-1）。

## 5. 规范审查（STYLE_GUIDE + rubric）

- lint：`LINT OK` × 3（2026-11-18 本机重跑）。
- 模板：三章均有 8 行内 `>` metadata、`## 总结`（表格）、`## 练习`（各 4 题，推导/计算/代码混合）、`## 参考`（链 ../references.md 含 key）✓；ASCII 图：9.1 模板实例块+多轮 masking 布局、9.2 三通路全链图、9.3 推导链图 ✓（≥1 张/章达标）。
- 篇幅（CJK 口径）：9.1 = 4187（≥3500 ✓）、9.2 = 5808（≥5500 ✓）、9.3 = 6156（≥5500 ✓）。
- AI 味：lint 词表 + rubric A1 扩展词表（值得注意的是/综上所述/总而言之/由此可见/不难发现/众所周知/让我们/本节将/正如前面/如前文所述/其实/我们不妨/随着…快速发展/强大的/革命性/优雅/极致/高效地/简洁地）全量 grep 零命中 ✓；无历史叙事段；各节以事实/公式/问题开场 ✓；数字均有出处（除 S3-7 的数量级口径）。
- 公式纪律（机扫，含 `$`...`$` 保护形优先剥离后重扫）：无禁用反斜杠转义、无行内矩阵环境、无 `$` 内侧空格、无开口/闭口 flanking 违规、无 `_` 接标点强调冲突、无表格内裸 `|` 公式 ✓。
- 证据等级：全部锚点标注 [A]/[B]/教学参照，与 references.md 一致（TRL 教学参照+官方实现口径与 spec §4.1 裁决一致）✓。
- 术语无漂移（response/prompt/掩码/轮次/reference 全书一致）✓。

## 6. 写作方存疑项的独立裁定

| 存疑项（出处） | 裁定 |
| --- | --- |
| task-31 #1（9.1 §5 纯 SFT replay 证据弱） | 接受现状。InstructGPT App C.3 的 10% 混入确按「helpful for PPO training」记录（已核原文），章节已标「本书注记」并与 RL 侧证据分级区分——处理合规，无需修改 |
| task-31 #2（9.1 §3 全轮 vs 末轮无公开消融） | 接受现状。Llama-3 Table 7 表注（context+final response）与 TRL `assistant_only_loss` 两条出处均核实；并列不裁决的写法合规 |
| task-31 #3（9.2 §3.1 的 1/C(K,2) 归一化语义推定） | 接受现状。式 (1) 照录原文 ✓，「逐 prompt 等权」已显式标注「本书注记（由单 batch 元素实现口径推定）」 |
| task-31 #4（β=0.02 沿 Stiennon 口径未引 arXiv） | 接受现状。原文确为 "in the same way as Stiennon et al., 2020"（已核）；Stiennon 不在 references.md，不引 ID 正确 |
| task-31 #5（Goodhart 式 (4) 本书推导） | 接受现状。重推证实为恒等式（§2.2），已标「本书推导 outline」，未冒充实证 |
| task-31 #6（policy lr 无单一数字） | 接受现状。E.9 只有扫描区间与发散阈（8.05e-6 已核），按原样呈现正确 |
| task-32 #2（TRL IPO token 平均为代码注释口径） | 接受现状。TRL main 源码注释逐字核实（"not explicitly discussed in the IPO paper; we confirmed this choice with the IPO authors"），章节注明「实现层事实、以代码注释口径为准」处理正确 |
| task-32 #3（ORPO 梯度符号） | 维持章节写法。独立重推（§2.4）：正确梯度为 $-\delta(d)h(d)$；原文式 (8) 无显式负号属幅度形式，章节保留负号并注明差异——正确且更严谨 |
| task-32 #5（9.3 §2 最优值 β·log Z(x) 为补步） | 接受现状。补步代数正确（§2.3 第 5 步），无额外假设 |
| ORPO 移入 experimental 与 9.3 引用的相容性（controller 加查） | 相容 ✓。main 分支 `trl/trainer/orpo_trainer.py` 404、`trl/experimental/orpo/orpo_trainer.py` 200，迁移 commit d705ac4d（2026-02-10）与章节「2026-02 移入」一致；9.3 §9/§10/参考列表全部指向 experimental 路径 |

## 7. 修复闭环建议

S2×6 必须先修：S2-4/S2-5/S2-6 为指针与口径的单行修改；S2-1 改一个词（「除以」→「乘以」）；S2-3 把「报错」改「警告」并补一句硬错误条件；S2-2 需要把 §6 走读的归属改准（VLM collator vs 预处理期 `tokenize_fn`），建议顺带保留现有机制描述（它对 VLM 通路逐行正确）。无 S1，修复后按协议不需要章级复审；S3×9 记入下波清零。
