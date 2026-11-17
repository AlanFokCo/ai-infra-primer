# 9.2 RLHF 全链：Reward Model → PPO（The RLHF Pipeline: Reward Modeling to PPO）

> 本章把 RLHF 三段链拆到公式与配方两级：偏好数据的标注协议与一致性上限（72.6% 的标注员间一致率是整条链的精度地板）；reward model 的完整训练目标推导（全 pair 枚举、margin 变体、逐 completion 梯度语义）与其结构 / 初始化 / epoch 配方；reward hacking 的 Goodhart 形式化 outline（代理奖励与真实偏好的 gap 随 KL 预算增长的机制）与三个实证形态；PPO 阶段全链路（InstructGPT 目标式、逐 token KL 进 reward、β 自适应两方案、流程伪码与四模型显存回链）；实证收益与局限（帮助性 / 无害性拆分、现代变体对照）| 前置：3.2（Bradley-Terry 模型式 (3)(4)、平移不可识别性、在线/离线分界）、8.2（GAE）、8.3（PPO clip、自适应 β 出处）、8.4 §3（k3 KL 估计器）、8.5（四模型显存式 (1)、KL 注入点、权重同步） | 锚点：InstructGPT（[Ouyang et al., 2022](../references.md)，arXiv:2203.02155，[B]——有 paper、代码未开放）、Llama-2（[Touvron et al., 2023](../references.md)，arXiv:2307.09288，[A]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、Christiano et al.（[Christiano et al., 2017](../references.md)，arXiv:1706.03741，[B]）、OpenRLHF（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]）、veRL（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]）

## 1. 三通路全景

RLHF 的原始流程是三段流水线，每段消费上一段的产物（InstructGPT Figure 2 的数据流，[Ouyang et al., 2022](../references.md)，[B]，§2）：

```
阶段一 SFT                  阶段二 RM                    阶段三 PPO
13k prompts                33k prompts                  31k prompts
标注员撰写+API              每 prompt K=4~9 条采样回答      只含 prompt，无人工标签
标注员写示范回答             标注员排序 → C(K,2) 偏好对      rollout：π_θ 对 prompt 采样
      │                          │                          │
      ▼                          ▼                          ▼
  π_SFT ─────────────────►  RM r_θ：6B                 actor π_θ（init = π_SFT）
 （兼作阶段三的             去 unembedding              打分：r_θ(x,y) − β·log(π_θ/π_ref)
   初始化与 π_ref）          换 scalar head               更新：PPO clip + γ·预训练项
                                                        critic 6B（init = RM）
─────────────────────────────────────────────────────────────────
数据量 13k → 33k → 31k（SFT→RM 扩、RM→PPO 微缩）；监督形态：示范 → 排序 → 无标注（只靠采样）
```

三段的分工一句话各钉一个：SFT 把「指令→回答」的格式教会基座（9.1）；RM 把人类的成对偏好蒸馏成一个标量函数；PPO 用这个标量函数当奖励、以 SFT 模型为锚做在线优化。按 3.2 §5 的分界判据归格：SFT 与 RM 都是 offline（静态数据集），PPO 是 on-policy（期望取自滚动更新的 rollout 策略）——整条链的系统复杂度集中在第三段（四模型驻留与引擎拆分的账归 8.5，本章回链不重复）。

两笔规模账先立住。其一，数据量：13k / 33k / 31k 是 InstructGPT 的三段 prompt 数（其 §3.2 与 Table 6），对照同模型预训练的数千亿 token，后训练三段的全部人工数据装不满一块硬盘——后训练改的是分布形状，不是知识存量（LIMA 的 Superficial Alignment Hypothesis 给了同一读法的 SFT 侧证据，9.1 §3）。其二，模型规模：InstructGPT 全程只用一个 6B RM 服务 1.3B 到 175B 全部 policy 档（其 App. C：175B RM 训练不稳定、且算力代价高）；critic 同为 6B、从 RM 初始化——「打分与估值用小模型、被优化对象随便大」是这条链的原始形态。

## 2. 偏好数据：标注协议与一致性上限

**标注协议**。为摊薄标注成本，标注员一次看到同一 prompt 的 $K$ 条回答并排排序， $K$ 取 4 到 9（[Ouyang et al., 2022](../references.md)，[B]，§3.5 的 Reward modeling 段）：一次排序任务产出至多 $\binom{K}{2}$ 个成对判断（平局是否进入训练，归 §3.3 的 RM 侧口径）。这个协议的副产品是 pair 之间强相关——同一批回答两两组合，标注误差在同一次任务内共享。Llama-2 的协议多一档信息：标注按四档偏好强度（significantly better 等）记录而非纯二值，这一档后来直接进了 margin 损失（§3.2）；规模上，Llama-2 累积的偏好数据超过 100 万条二元比较（其 §3.2.1 的 Meta reward modeling data 口径），约为 InstructGPT 33k prompt 的 30 倍（比较条数对 prompt 数的口径）——偏好数据的军备竞赛在后训练里的位置，与预训练语料的扩张同构。Llama-3 在此基础上再加一列：部分 prompt 的标注员可直接编辑 chosen 回答得到第三条「edited response」，排序成为 edited ≻ chosen ≻ rejected 的三元组（其 §4.1.2 口径）——标注从「判优劣」扩展到「顺手修答案」，同一次人工接触产出更密的监督。采样侧另一条纪律：被排序的回答来自当前最新模型（Llama-3 每轮用上一轮最强 checkpoint 采样、只把「chosen 明显更好或更好」的对留下、丢弃相近对——其 §4.2.1，[A]），偏好数据由此跟随策略分布滚动更新。

**一致性是地板**。InstructGPT 报告标注员间一致率：训练组内部 72.6 ± 1.5%，held-out 标注员组内部 77.3 ± 1.3%（其 §3.4）。这两个数定义了整条链的精度天花板：RM 学的是这批标注员的集体判断，逐对预测精度不可能长期超过标注员彼此的一致率——实测 RM 在留出数据上的精度为 72.4 ± 0.4%（训练组内的留出对）与 69.6 ± 0.9%（held-out 标注员组，五折交叉验证口径；正文 §4.1 末段与 App. E.2），两个数贴着一致率下沿。读法（本书分析）：RM 精度与一致率之间的差不是模型欠训的证据，而是偏好数据本身噪声的度量——式 (1) 的 loss 地板由这个噪声决定（3.2 §2 的不可约随机性口径），继续堆 RM 容量先撞上的是标注噪声而非模型表达力。标注协议的全部改进杠杆因此在「协议与筛选」一侧：单人重复标注估噪声、多人标注取多数、按一致性筛标注员——InstructGPT 的标注员筛选带 75% 一致率软门槛（其 App. B）。

一句口径外推：用模型替代人做偏好标注（RLAIF、Constitutional AI 一脉）改变的是标注成本与一致率的权衡——模型标注一致率可以做高，但「与谁一致」换成了另一个模型；该方向的完整展开超出本章，此处只登记接口位置：偏好数据的来源从人换成模型，式 (1) 逐字不动。

## 3. Reward Model 训练：目标、结构与配方

### 3.1 完整训练目标：全 pair 枚举

Bradley-Terry 模型 $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ 与其 NLL 目标已在 3.2 §2 从随机效用公理完整推导（式 (1)-(4)），此处只接结论。InstructGPT 的实际 RM 目标在其上多做一步聚合——一个 prompt 的全部 $\binom{K}{2}$ 个偏好对作为**单个 batch 元素**参与训练（其式 (1)）：

$$
\mathrm{loss}(\theta) = -\frac{1}{\binom{K}{2}} \thinspace \mathbb{E}_{(x,\thinspace y_w,\thinspace y_l) \sim D} \left[ \log \sigma\big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \right] \qquad (1)
$$

**符号**： $r_\theta$：RM 的标量输出； $(y_w, y_l)$：同一 prompt 下的胜/负回答对；期望取自偏好数据集 $D$ ，其中每个 prompt 的 $\binom{K}{2}$ 个对整体作为一个样本。动机论文写得很白：若把 $\binom{K}{2}$ 个对打散混进数据集，每条 completion 会参与 $K - 1$ 次独立梯度更新、一个 epoch 内被重复消费，RM 单 epoch 即过拟合；全 pair 作为一个 batch 元素则每条 completion 只需一次前向、不再过拟合，validation accuracy 与 log loss 都显著改善（[Ouyang et al., 2022](../references.md)，[B]，§3.5 及其脚注 3）。归一化 $1/\binom{K}{2}$ 让每个 prompt 等权——若按 pair 总量计， $K$ 大的 prompt 权重随对数平方涨（本书注记：式 (1) 照录原文形式，逐 prompt 内部的求和结构由「单 batch 元素」的实现口径推定）。

逐 completion 的梯度值得展开一层（3.2 式 (5) 的聚合形式）。对 prompt 内第 $i$ 条回答的得分 $r_i = r_\theta(x, y_i)$ ，式 (1) 的梯度为

$$
\frac{\partial \mathrm{loss}}{\partial r_i} = \frac{1}{\binom{K}{2}} \left[ \sum_{j:\thinspace i \succ j} \big( \sigma(r_i - r_j) - 1 \big) + \sum_{j:\thinspace j \succ i} \sigma(r_i - r_j) \right] \qquad (2)
$$

**符号**：第一个和跑遍「 $i$ 胜出」的对，第二个跑遍「 $i$ 落败」的对。逐项读：胜出对的贡献 $\sigma(r_i - r_j) - 1 = -\sigma(r_j - r_i)$ 是负的（梯度下降把 $r_i$ 往上推），落败对的贡献 $\sigma(r_i - r_j)$ 是正的（把 $r_i$ 压下）；每项的模长恰是模型当前把该对**判反的概率**。三个直接推论：(i) 已被拉开的对梯度趋零，RM 的训练信号自动集中在当前分不开的难对上——难样本加权是 sigmoid 损失的导数形态自带的（3.2 §2 同结论）；(ii) 胜负各半的中间档回答（争议样本）收正负两股满力，其平衡位置由两股概率权重决定而非计数——争议大的回答得分被推向「恰好让两股力抵消」的中间值；(iii) 每条回答参与 $K - 1$ 个对，其更新是多股力的向量和——单对标注噪声被聚合稀释，这正是「同 prompt 全 pair 枚举」相对「随机抽对」的方差收益。

### 3.2 margin 变体与它的消失

Llama-2 在式 (1) 上叠一项 margin（其式 (2)）：

$$
\mathrm{loss}_{\mathrm{margin}} = -\log \sigma\big( r_\theta(x, y_w) - r_\theta(x, y_l) - m(r) \big) \qquad (3)
$$

**符号**： $m(r) \ge 0$：偏好强度分档的离散函数——「明显更好」的对给大 margin、「相近」的对给小 margin（[Touvron et al., 2023](../references.md)，[A]，§3.2.1 与其 Table 27）。梯度语义随之改档：无 margin 时，判对（ $\Delta r \gt 0$ ）即进入梯度衰减区；有 margin 时，分界被推到 $\Delta r = m(r)$ ——模型必须把「明显更好」的对拉开到 $m(r)$ 以上才算达标。消融结论是双向的：margin 提升可分对上的精度，但在相近对上回退（其 App. A.3.3）。这个变体的一生值得按出处记一笔：Llama-2 引入（其偏好数据恰好带四档强度标注，margin 有数据源）；Llama-3 把它删了——原文口径是数据规模上来之后收益递减（"we remove the margin term in the loss, as we observe diminishing improvements after data scaling"，[Grattafiori et al., 2024](../references.md)，[A]，§4.1.2）。OpenRLHF 的实现把 margin 留作开关：`openrlhf/models/loss.py` 的 `PairWiseLoss.forward` 以 `margin` 为可选参数，存在时算 $-\mathrm{logsigmoid}(r_c - r_r - m)$（2026-11-18 核实）。

Llama-3 对 RM 训练循环还做了第二处工程改动，值得与式 (1) 对照着记：把同一 prompt 的多条回答（含 edited 第三条）**拼接进同一行序列**训练、回答随机乱序，替代「每对一行」的标准形态——原文口径是对标准做法的近似，但消融显示训练效率提升而精度无损（其 §4.1.2）。机制读法（本书分析）：共享 prompt 前缀只算一次前向，与 InstructGPT「全 pair 作单 batch 元素、每条 completion 一次前向」省的是同一笔账；差别在拼接形态把 $\binom{K}{2}$ 个独立前向进一步压成一次，KV 复用跨回答共享。

### 3.3 结构、初始化与配方

**结构**：RM = 同架构 LM 去掉 unembedding、换一个标量投影头。InstructGPT 原文口径：「unembedding layer 被替换为输出标量的投影层」（其 App. C）；OpenRLHF 的落法（`openrlhf/models/model.py`，2026-11-18 核实）：`nn.Linear(hidden_size, 1, bias=False)` 挂为 `score` 属性，读分位置在每条序列最后一个非 pad token——`eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)` 再 `values.gather(dim=1, index=eos_indices)`。读分位置在末 token 不是惯例而是语义必需：causal 掩码下只有末位置的表示见过全部回答内容，任何中间位置的「分」都没读完全文（本书注记；BT 模型的 $r(x, y)$ 是整段回答的函数）。chosen 与 rejected 拼进同一 batch 一次前向（`concatenated_forward`——与 3.2 §6 的 TRL 对半拼批同构），pair 精度监控为 `(chosen_reward > reject_reward)` 的均值（`openrlhf/trainer/rm_trainer.py`）。

**初始化与配方**（InstructGPT App. C 逐项）：RM 从一个在公开 NLP 数据集（ARC / BoolQ / CoQA / DROP / MultiNLI / OpenBookQA / QuAC / RACE / Winogrande）上微调过的 6B GPT-3 出发——作者注明这 mainly 是历史原因，从 GPT-3 或 SFT 模型初始化效果相近；训 1 个 epoch（多 epoch 迅速过拟合、validation loss 明显恶化），lr $9 \times 10^{-6}$ 、cosine 降至 10%、batch 64 个 prompt（每 prompt 至多 $\binom{K}{2}$ 对——平局在 RM 训练侧丢弃，App. C.2 口径；单 batch 至多 2304 对）；lr 与 schedule 不敏感（50% 变动无感），epoch 数敏感。Llama-2 同口径：RM 训 1 epoch，70B 档最大 lr $5 \times 10^{-6}$ 、其余 $1 \times 10^{-5}$（[Touvron et al., 2023](../references.md)，[A]，§3.2.2）。规模的取法前面已立：单一 6B RM 服务全部 policy 档，175B RM「训练不稳定、不宜作 RL 的 value function 初值」（其 App. C）——RM 是评分器不是生成器，容量让位给稳定性。

### 3.4 校准与分布外泛化

式 (1) 只约束排序，不约束分值刻度（3.2 §3 的平移不可识别性： $r$ 与 $r + c(x)$ 等价）。要给 RM 输出更多语义，得查校准性（calibration）：BT 模型断言 $\sigma(\Delta r)$ 等于经验胜率，把留出对按预测概率分桶、对照桶内实际胜率即可检验——完全校准的 RM 里 $\sigma(\Delta r) = 0.9$ 的桶应含约九成真实胜率（本书分析口径；InstructGPT 未发布校准曲线，此处只登记检验方法，不断言结果）。

分布外（OOD）泛化是 RM 的结构性弱点：训练分布由采样时的策略决定，PPO 一旦把策略推离，RM 就要在从未见过的回答形态上打分——那里正是 §4 的 hacking 发生处。工程答案是把 RM 数据做成跟随策略的活分布：Llama-3 六轮后训练每轮重收偏好标注，DPO 训练只用「最新若干批、与当前 policy 分布贴合」的数据（其 §4.1.4 与 §4.2.1 原文口径）；Llama-2 的另一条是把开源偏好数据留在混合里——作者明确记录动机为改善泛化、防 reward hacking（其 §3.2.2，原话见本章 §4 开头）。两条合起来的语义：RM 不是训一次封存的资产，是与策略同步滚动的耗材。

## 4. Reward hacking：Goodhart 的形式化 outline

Llama-2 论文给了 reward hacking 的工作定义：「利用 RM 的弱点人为抬高得分、实际表现反而更差」（"taking advantage of some weaknesses of our reward, and so artificially inflating the score despite performing less well"，[Touvron et al., 2023](../references.md)，[A]，§3.2.2）。把这个现象写成式子只需一组记号（本书推导 outline）。设真实偏好强度为 $r^{\ast}(x, y)$ （理想标注总体下的 BT log-odds），RM 学到的是 $\hat{r} = r^{\ast} + \varepsilon$ ，误差项 $\varepsilon(x, y)$ 在 RM 训练分布上被数据钉住（期望小），分布之外不受任何约束。PPO 阶段的优化目标是代理奖励（式 (5) 展开），策略相对 $\pi_{\mathrm{ref}}$ 移动 KL 距离 $d$ 后，代理收益与真实收益的差为

$$
\widehat{G}(\pi) - G(\pi) = \mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[ \varepsilon(x, y) \right] - \mathbb{E}_{y \sim \pi_{\mathrm{ref}}(\cdot \mid x)}\left[ \varepsilon(x, y) \right] \qquad (4)
$$

**符号**： $`\widehat{G}(\pi) = \mathbb{E}_\pi[\hat{r}] - \mathbb{E}_{\mathrm{ref}}[\hat{r}]`$ ：代理收益； $G(\pi)$：同式的 $r^{\ast}$ 版。式 (4) 是恒等式，机制全在两项的行为差上： $`\mathbb{E}_{\mathrm{ref}}[\varepsilon]`$ 被 RM 训练数据锚在小值（ $\pi_{\mathrm{ref}}$ 的样本就来自 RM 见过的分布）； $`\mathbb{E}_{\pi}[\varepsilon]`$ 则随 $\pi$ 离开数据流形而脱锚——而**最大化代理奖励的过程本身就是 $\varepsilon$ 的正向选择器**：优化器在 KL 球内专挑 $\hat{r}$ 高的区域，其中一部分是 $r^{\ast}$ 真的高，另一部分是 $\varepsilon$ 恰好为正。优化压力越大（KL 预算越宽、β 越小、步数越多），被选中区域里 $\varepsilon \gt 0$ 的占比越高——Goodhart 定律在这条链上的精确形态：「指标成为目标后便不再是好指标」= 式 (4) 右端随优化压力单调变坏。两个边界读法：(i) $d \to 0$ 时 gap 归零——KL 锚的全部意义就是把策略按在 RM 可信的流形附近，这是式 (5) 里 $\beta$ 项的存在理由；(ii) gap 的增长不代表 $G$ 立即下降——真实收益先升后平的驼峰形态由此而来（代理收益继续涨、真实收益持平再回落），定量的 hump 曲线拟合超出本书证据范围，此处只给机制 outline（本书注记）。

三个实证形态各配出处。**过度对冲（hedging）**：InstructGPT 记录了完整的反馈环——标注指南要求奖励「认知谦逊」，标注员倾向给打太极的回答高分，RM 学到这个相关，最终模型对直问直答的问题也给多解搪塞（[Ouyang et al., 2022](../references.md)，[B]，§4.3：原文把该行为部分归因于标注指令经 RM 的放大路径）。这是式 (4) 的人因版本： $\varepsilon$ 不是随机噪声，而是标注协议系统性注入的偏置，优化器忠实地把它放大。**长度通胀**：RM 训练数据里「更长」与「更好」存在相关时，长度就进 $\varepsilon$ ——工程侧的直接证据是 RL 栈需要专门的长度对策：DAPO 的 overlong 处理与 dynamic sampling 正是为长度失控设计的组件（[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]；算法侧归 8.4 §6）；长 CoT 场景下 response 长度随训练持续增长有 RLVR 语境的直接记录（[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]；9.5 展开）。**缓解手段的位置**：式 (4) 指出三个作用点——压 $d$（KL 项 β，§5.3）、缩小 $\varepsilon$（RM 数据随策略滚动重采，§3.4；多源混合，Llama-2 口径）、换掉可被操纵的奖励本身（规则验证器——RLVR 的防 hacking 性质归 9.5）。现代栈把第一条做成监控：watch 的指标是代理奖励与 KL 的联合曲线，不是奖励单线。

## 5. PPO 阶段全链路

### 5.1 InstructGPT 目标式

PPO 阶段的完整优化目标（InstructGPT 式 (2)，照录原式换本书符号）：

$$
\mathrm{objective}(\phi) = \mathbb{E}_{(x,\thinspace y) \sim D_{\pi^{\mathrm{RL}}_{\phi}}} \left[ r_\theta(x, y) - \beta \log \frac{\pi^{\mathrm{RL}}_{\phi}(y \mid x)}{\pi^{\mathrm{SFT}}(y \mid x)} \right] + \gamma \thinspace \mathbb{E}_{x \sim D_{\mathrm{pretrain}}} \left[ \log \pi^{\mathrm{RL}}_{\phi}(x) \right] \qquad (5)
$$

**符号**： $`\pi^{\mathrm{RL}}_{\phi}`$：被优化的策略； $r_\theta$：冻结的 RM； $`\pi^{\mathrm{SFT}}`$：SFT 模型，兼作 KL 锚（即全书约定的 $\pi_{\mathrm{ref}}$ ）； $D_{\mathrm{pretrain}}$：预训练数据分布； $\beta$：KL 奖励系数，原文取 0.02（KL 的计算方式原文注明沿用 Stiennon et al. 2020，控制器细节未展开）； $\gamma$：预训练混合系数—— $\gamma = 0$ 是纯 PPO， $\gamma = 27.8$ 是论文默认的 PPO-ptx。期望号下的分布是第一项的命门： $(x, y)$ 采自**当前策略自己**——on-policy（3.2 §5 判据），每轮更新后必须重采，rollout 引擎由此成为基础设施（账归 8.5）。

三项各管一件事。RM 项是目标本体；KL 项把策略按在 $\pi_{\mathrm{SFT}}$ 附近——除了 §4 的防 hacking 语义，它同时保住文本的基座连贯性（RM 只在「像人话的回答」分布上训过）；预训练项直接对抗 forgetting（9.1 §5 的 replay 账在 RL 侧的对应物：原文记录 SQuADv2 / DROP 回退由该项修复，而把 β 加大 100 倍修不回——其 §4.2 与 App. E.6 的 Figure 34）。 $\gamma = 27.8$ 配 8 倍于 RL episode 的预训练数据量（App. C.4），代价是训练时间约翻倍（App. E.11）。

### 5.2 逐 token 落地：RM 分在末端，KL 逐 token

式 (5) 是序列级目标；PPO 的 advantage（8.2 的 GAE）需要逐 token 奖励。实现层的标准分解（veRL / OpenRLHF 同形态）：RM 分只加在回答的最后一个 token 上，KL 惩罚逐 token 扣除：

$$
\tilde{r}_t = -\beta \thinspace \widehat{D}_{\mathrm{KL},t} \quad (t \lt T), \qquad \tilde{r}_T = r_\theta(x, y) - \beta \thinspace \widehat{D}_{\mathrm{KL},T} \qquad (6)
$$

**符号**： $T$：回答长度； $`\widehat{D}_{\mathrm{KL},t}`$：第 $t$ 个 token 处当前策略对参考策略的 KL 估计——无闭式解，用低方差无偏的 k3 估计器 $\exp(\log \pi_{\mathrm{ref}} - \log \pi_\theta) - (\log \pi_{\mathrm{ref}} - \log \pi_\theta) - 1$（完整推导与 k1/k2 对照归 8.4 §3，出处 [Schulman, 2020](../references.md)，教学参照）。逐 token 求和后 $`\sum_t \tilde{r}_t`$ 恰好回到式 (5) 第一项的序列形态（KL 逐 token 分解的恒等性，8.4 §3 同口径）。分解的工程语义值得单独一句：RM 分集中在末端一个 token 上，KL 惩罚摊在每一步——advantage 经 GAE 回溯后，「哪个 token 为最终得分负责」的 credit assignment 问题原样浮现（8.2 §5 已分析这个退化结构），逐 token 的 KL 项恰好给每个位置一个「别跑远」的即时信号，部分代偿了末端奖励的稀疏。工程落点（8.5 §8 已走读、此处回链）：veRL 在 `verl/trainer/ppo/trainer_utils.py` 的 `apply_kl_penalty` 里做 `token_level_rewards = token_level_scores − β · kld`，β 来自 `kl_ctrl`；另有 loss 侧注入点：actor config 的 `use_kl_loss` 开关（`verl/workers/config/actor.py`，默认 False，`kl_loss_coef` 默认 0.001、`kl_loss_type` 默认 `"low_var_kl"`，2026-11-18 核实）。GRPO 一脉把 KL 整个搬进 loss（DeepSeekMath 口径），PPO 一脉留在 reward——两路的梯度语义差异归 8.4 §3，本章只用结论。

### 5.3 β 自适应：两套方案

β 的取值本身有直接的消融证据：InstructGPT 在带预训练混合的 PPO 上扫 KL 系数，人评 Likert 分在 β = 0 与 β = 2 两端都显著变差，最优区间在 0.01–0.02（[Ouyang et al., 2022](../references.md)，[B]，App. E.7 的 Figure 36 口径）——β = 0 是「撤掉锚」、β 过大是「锚死」，最优值卡在中间一个窄带，这正是它被选作 0.02 的出处。

但固定的 β 无法贯穿训练：策略离锚渐远时同一 β 的约束力在变化。两套有出处的自适应方案。**PPO 论文版**：每轮更新后量 $`\hat{d} = \hat{\mathbb{E}}_t[D_{\mathrm{KL}}]`$ ， $\hat{d} \lt d_{\mathrm{targ}} / 1.5$ 则 β 减半、 $\hat{d} \gt 1.5 \thinspace d_{\mathrm{targ}}$ 则 β 加倍（[Schulman et al., 2017](../references.md)，arXiv:1707.06347，[B]，§4——8.3 §4.2 已登记，该论文自己的消融里 clip 变体总分手压过自适应 β 变体，故 LLM 栈里它更多是监控兜底而非主力）。**veRL 版**（`verl/trainer/ppo/core_algos.py` 的 `AdaptiveKLController`，2026-11-18 核实）：比例控制

$$
\beta \leftarrow \beta \cdot \left( 1 + \frac{n_{\mathrm{steps}}}{H} \cdot \mathrm{clip}\left( \frac{\widehat{\mathrm{KL}}}{d_{\mathrm{targ}}} - 1, \thinspace -0.2, \thinspace 0.2 \right) \right) \qquad (7)
$$

**符号**： $`\widehat{\mathrm{KL}}`$：当前测得的 KL； $d_{\mathrm{targ}}$：目标 KL； $H$：horizon 常数；clip 把单步调整率钳在 ±20%。与 PPO 版的差别是平滑：离目标越远调得越狠但封顶，避免倍率式跳变。实现注记：KL 的估计照样走 §5.2 的 k3 一族；OpenRLHF 的 PPO 损失在 `openrlhf/models/loss.py` 的 `PolicyLoss`（clip 半径解耦为 `clip_eps_low` / `clip_eps_high`，8.3 §8 已登记）。

### 5.4 流程伪码与四模型账

一轮 PPO 迭代的完整伪码（InstructGPT App. C.4 超参嵌在注释里；框架实现参照 veRL / OpenRLHF）：

```
冻结: RM r_θ、critic V_ψ 的初始化(ψ ← θ_RM)、ref π_ref(= π_SFT)
loop 每一轮迭代:
    ① rollout：π_θ_old 对 512 个 prompt 采样（T=1 温度，回答 ≤1k token）
    ② 打分：RM 前向取末端分 r(x,y)；ref 前向取逐 token logp
             → 按式 (6) 合成逐 token 奖励 r̃_t
    ③ advantage：GAE（γ_discount = 1，不打折）→ Â_t 冻结
    ④ 更新：batch 512 拆 8 个 minibatch（64）、单 inner epoch
             actor：PPO clip（ε = 0.2）+ γ·预训练项（γ = 27.8）
             critic：价值回归；权重 EMA 0.992（原文未指明挂载侧）；常数 lr、前 10 轮 warmup
    ⑤ 权重同步：训练引擎 → rollout 引擎（账归 8.5 §4）
```

四模型同驻的显存账已归 8.5 式 (1)：actor 训练态 $16 N_a$ + critic 训练态 $16 N_c$ + ref 推理态 $2 N_a$ + RM 推理态 $2 N_r$ + rollout 副本 $2 N_a$ + KV cache——InstructGPT 的实例是「actor 1.3B-175B 变、RM 与 critic 钉死 6B」，式中 $N_c$ 、 $N_r$ 固定、 $N_a$ 缩放；GRPO 去 critic 的省法与现代放置策略同样归 8.5，此处不重复。

伪码里三个超参的出处与读法（均 InstructGPT App. C.4 口径）。其一，**单 inner epoch**：512 条 rollout 拆 8 个 minibatch 各过一遍即弃——对照 8.3 的 MuJoCo 口径 $K = 10$ 个 epoch，LLM-RL 把数据复用压到最低，陈旧偏差（8.3 §4.2）几乎为零，代价是样本效率回到 on-policy 下限；rollout 占大头的成本结构（8.5 §1 的 80% profile）让这个取舍成立。其二，**权重 EMA 0.992**：原文只记「对权重施加指数滑动平均、衰减率 0.992」，未指明挂在 policy 还是 value 一侧、也未写评估是否使用 EMA 权重（本书注记）；机制上与 2.1 momentum 同族的方差压制，有效记忆窗约 $1/(1-0.992) = 125$ 步。其三，**GAE 不打折**（ $\gamma = 1$ ）：回答 ≤1k token 的 episodic 任务里折扣没有位置，且 RM 分本身就是全程目标的度量；value function 的 lr 单独定（9e-6 / 5e-6），policy 的 lr 扫出来的结论是「不带预训练混合时 $8.05 \times 10^{-6}$ 以上全部发散、带混合后容忍度显著变宽」（App. E.9 口径）——式 (5) 的 γ 项顺带也是稳定器。

### 5.5 与 SFT 段的对照

把三段并排读，形态差异一眼见底：SFT 与 RM 是静态数据上的监督训练（9.1 与 §3 的配方纪律：小 lr、防过拟合、评估器选型），PPO 是带自采样回路的在线优化（8.3 的陈旧偏差、8.5 的系统账、本节的目标与控制器）——同一套「预训练 CE 的变体」在离线两段是凸性尚可的判别问题，进在线段后变成「目标函数自己在动」的控制问题。Llama-3 的路线选择是这个差异的直接后果：其 post-training 明确记录「探索过 PPO 等在线算法，但 DPO 在大规模模型上算力需求更低、且指令遵循基准（IFEval）上表现更好」（[Grattafiori et al., 2024](../references.md)，[A]，§4.1.4）——把在线段换成离线目标（DPO，9.3）正是绕开本节全部系统复杂度的路线；拒绝采样 + SFT 的迭代（9.4）是另一条把「在线」降级为「离线多轮」的路线。三条路线共享同一个上游：§2 的偏好数据与 §3 的 RM——数据层的事实在哪条路线里都成立。

## 6. 实证收益与局限

**收益**（InstructGPT 原文口径，[B]）：标注员盲评下 1.3B 的 InstructGPT 输出优于 175B GPT-3——对齐把模型规模的账打出一个数量级以上的折扣；175B InstructGPT 对 175B GPT-3 胜率 85 ± 3%、对 few-shot GPT-3 胜率 71 ± 4%；毒性上「被指示保持尊重」时比 GPT-3 少约 25% 有毒输出（RealToxicityPrompts，Perspective API 口径）；TruthfulQA 上有小但显著的真实性提升（默认行为，无需专门提示）。

**局限**（同出处逐项）：偏见基准（Winogender、CrowS-Pairs）无显著改善；「被指示保持尊重」的毒性优势在不加该提示时消失，且被显式要求生成有毒内容时比 GPT-3 更有毒； §4 的 hedging 与「顺着错误前提走」两类残余失败模式直接进案例分析。**对齐税**（alignment tax）同样出自这篇论文：不带预训练混合的 PPO 在 SQuADv2 / DROP 等公开 NLP 基准上回退，γ 项（式 (5)）把它拉回但「不能完全消除回退」（其 §5.4 口径）；把 RL 训练从 256k episodes 加倍到 512k，回退随训练进程继续加深（App. E.6 的 Figure 35 口径）——偏好收益与通用能力之间存在真实的交换面，预训练项是缓冲不是消除。对齐对象的边界论文自己划清：偏好来自约 40 名标注员（Upwork / Scale AI 渠道、以英语为主），「对齐到特定人群的偏好」不等于对齐到任何更广义的「人类价值」（其 §5.2 原话口径）。

**帮助性与无害性的拆分**是下一阶段的标准做法：两个目标存在此消彼长（trade-off 现象经 [Touvron et al., 2023](../references.md)，[A]，§3.2.2 转引确认），单一 RM 难以兼顾，Llama-2 因此分训 Helpfulness RM 与 Safety RM 两个头。这条拆分的语义注记一句：两个 RM 的分歧本身就是「人类偏好不可一维化」的证据——式 (1) 的标量假设在大尺度上开始崩，多目标 RLHF 的入口在这里。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 三通路 | SFT 13k → RM 33k → PPO 31k prompts；RM/critic 钉 6B 服务全部 policy 档 | §1；arXiv:2203.02155 §3.2 |
| 标注一致性 | 训练组 72.6±1.5%、held-out 77.3±1.3%；RM 精度 72.4% / 69.6% 贴一致率下沿——一致率是地板 | §2，App. C.2 |
| RM 目标 | 全 $\binom{K}{2}$ 对作单 batch 元素、 $1/\binom{K}{2}$ 归一（式 (1)）；逐 completion 梯度式 (2)：模长 = 判反概率、争议样本收双向满力；1 epoch 过拟合纪律 | arXiv:2203.02155 式 (1) 与脚注 3 |
| margin | 式 (3)：分界推到 $\Delta r = m(r)$ ；Llama-2 引入（四档强度标注）、Llama-3 删除（数据规模化后收益递减） | arXiv:2307.09288 §3.2.2；arXiv:2407.21783 §4.1.2 |
| RM 结构 | 去 unembedding 换 scalar 头；末非 pad token 读分（OpenRLHF `eos_indices` gather）；6B 单一 RM | §3.3 |
| Goodhart outline | 式 (4)：gap = $`\mathbb{E}_\pi[\varepsilon] - \mathbb{E}_{\mathrm{ref}}[\varepsilon]`$ ，优化器是 $\varepsilon$ 的正向选择器；KL 锚 = 把策略按在 RM 可信流形附近 | §4（本书推导） |
| PPO 目标 | 式 (5)：RM 分 − β·KL + γ·预训练项；β=0.02、γ=27.8、8 倍预训练数据、时间翻倍 | arXiv:2203.02155 式 (2)、App. C.4 / E.11 |
| 逐 token 落地 | 式 (6)：末端 RM 分 + 逐 token k3 KL；veRL `apply_kl_penalty` | 8.4 §3；8.5 §8 |
| β 自适应 | PPO 论文 double/halve（1.5× 阈）；veRL 比例控制式 (7)（±20% 钳） | arXiv:1707.06347 §4；veRL `core_algos.py` |
| 实证 | 1.3B > 175B GPT-3；85±3% / 71±4% 胜率；毒性 −25%（respectful）；偏见基准无改善；hedging 失败模式 | §6 |
| 现代对照 | Llama-3：DPO 算力更低、IFEval 更好（§4.1.4 原文口径）；在线段 → 离线目标的路线分岔 | 9.3 / 9.4 预告 |

## 练习

1.（推导）式 (2) 的逐项验证：对 $K = 4$ 、真实排序 $y_1 \succ y_2 \succ y_3 \succ y_4$ ，设当前得分 $r = (2, 1, 0.9, 0)$ 。(a) 枚举全部 6 个对并算出式 (2) 的四个分量（保留两位小数）；(b) 指出哪条回答收双向最大拉力，解释「争议样本被推向力平衡点」的语义；(c) 若把 $r_3$ 改为 1.1（与 $r_2$ 交叉），梯度分量怎么变？

2.（推导）margin 的梯度分析：对式 (3)，证明判对但未达 margin（ $0 \lt \Delta r \lt m(r)$ ）的对仍产生满力梯度的条件，并算出 $\Delta r = 0$ 处梯度模长与 $m(r)$ 的关系；据此解释 Llama-2 App. A.3.3 的消融结论「margin 提升可分对精度、回退相近对」的必然性。

3.（计算）RM batch 的对数账：batch 64 prompt、 $K$ 均匀取 4 到 9。(a) 算单 batch 偏好对数的期望与上限（InstructGPT 的 2304 上限怎么来的）；(b) 若改为打散训练、每对一次前向，同 batch 的前向次数从 $64K$ 涨到多少；(c) 按 6B RM、bf16、 $2N$ FLOPs/token 前向口径，序列长 1k，估两种形态单 batch 的前向 FLOPs 比。

4.（代码）教学实现 `PairWiseLoss`（教学用，非优化实现）：手写 $-\log \sigma(r_w - r_l)$ 与带 margin 变体，与 `torch.nn.functional.logsigmoid` 对拍；再实现 OpenRLHF 式末端读分（`attention_mask` 翻转 argmax + gather），在含右 pad 的合成 batch 上验证读分位置正确；最后构造 $\Delta r$ 网格画出 loss 与梯度模长曲线，标出 $m(r) = 1$ 时分界移动的位置。

## 参考

- [Ouyang et al., 2022](../references.md)：Training language models to follow instructions with human feedback，arXiv:2203.02155，[B]（key: instructgpt）——三段流程与数据规模（§3.2 / Table 6）、K=4~9 标注协议与 RM 目标式 (1)、全 pair 单 batch 元素（§3.5 的 Reward modeling 段及脚注 3）、标注一致率（§3.4）、RM 精度与五折交叉（§4.1 末段与 App. E.2）、RM 结构与 1 epoch 配方（§3.5、App. C.2）、PPO 目标式 (2)（§3.5）与 β=0.02 / γ=27.8 / 8 倍预训练数据（App. C.4 / E.11）、KL 加大百倍修不回（§4.2 与 App. E.6 Figure 34）、β 消融 0.01–0.02 最优区间（App. E.7）、policy lr 扫描与发散阈（App. E.9）、hedging 失败模式（§4.3）、标注员构成与对齐边界（§5.2）、对齐税未完全消除（§5.4）
- [Touvron et al., 2023](../references.md)：Llama 2，arXiv:2307.09288，[A]（key: llama2）——margin 变体式 (3) 与消融（§3.2.1-3.2.2 / App. A.3.3）、Helpfulness / Safety 双 RM、reward hacking 工作定义、开源数据混合防 hacking、RM 1 epoch 与 lr（§3.2.2）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——移除 margin（§4.1.2）、六轮迭代与最新批数据（§4.1.4 / §4.2.1）、弃 PPO 取 DPO 的原文口径（§4.1.4）
- [Christiano et al., 2017](../references.md)：Deep Reinforcement Learning from Human Preferences，arXiv:1706.03741，[B]（key: christiano）——RM 训练目标的原始出处（3.2 §2 同引）
- [Schulman et al., 2017](../references.md)：Proximal Policy Optimization Algorithms，arXiv:1707.06347，[B]（key: ppo）——自适应 β 的 double/halve 方案（§4；clip 推导归 8.3）
- [Schulman, 2020](../references.md)：Approximating KL Divergence（教学参照，key: schulman-kl）——k3 估计器出处（推导归 8.4 §3）
- [Hu et al., 2024](../references.md)：OpenRLHF，arXiv:2405.11143，[A]（key: openrlhf）——`openrlhf/models/loss.py`（`PairWiseLoss` margin 开关、`PolicyLoss`）、`openrlhf/models/model.py`（`score` 头与 `eos_indices` 读分）、`openrlhf/trainer/rm_trainer.py`（`concatenated_forward`、pair 精度监控）；2026-11-18 核实
- [Sheng et al., 2024](../references.md)：HybridFlow（veRL），arXiv:2409.19256，[A]（key: verl）——`verl/trainer/ppo/core_algos.py`（`AdaptiveKLController` 式 (7)）、`verl/trainer/ppo/trainer_utils.py`（`apply_kl_penalty`）、`verl/workers/config/actor.py`（`use_kl_loss` / `kl_loss_coef` / `kl_loss_type`）；2026-11-18 核实
- [Yu et al., 2025](../references.md)：DAPO，arXiv:2503.14476，[A]（key: dapo）——overlong 处理作为长度通胀的工程证据（算法侧归 8.4）
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-R1，arXiv:2501.12948，[A]（key: deepseek-r1）——RL 训练中 response 长度增长的记录（展开归 9.5）
