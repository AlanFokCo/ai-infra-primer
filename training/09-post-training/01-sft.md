# 9.1 SFT：指令微调的流程与数据形态（SFT: Pipeline and Data Forms of Instruction Tuning）

> SFT 是后训练的第一段，目标函数却没有新东西：还是预训练那个逐 token 交叉熵，唯一的变化是 loss 只计 response token。本章讲「同一个 CE」之下的工程语义——chat template 怎么把消息列表写成 token 序列（Llama-3 与 Qwen2.5 实例对照）、多轮对话的 masking 策略各自意味着什么梯度、LIMA 的质量实证、SFT 配方（LR、epoch、packing）、catastrophic forgetting 的 replay 对策，最后走读 TRL `SFTTrainer` 的实现 | 前置：3.1（CE 训练侧口径）、3.2（后训练目标接口表）、7.1 §6（packing 效率账）、models 8.3 式 (1)（loss masking 定义，纯文字指引） | 锚点：TRL `trl/trainer/sft_trainer.py`（huggingface/trl repo，教学参照 + 官方实现）、Llama-3 herd post-training 节（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、LIMA（[Zhou et al., 2023](../references.md)，arXiv:2305.11206，[B]）、Llama-2（[Touvron et al., 2023](../references.md)，arXiv:2307.09288，[A]）、InstructGPT（[Ouyang et al., 2022](../references.md)，arXiv:2203.02155，[B]）

## 1. SFT 目标：同一个 CE，换的是掩码

SFT（supervised fine-tuning，指令微调）的训练目标就是预训练的 next-token CE，原样复用。唯一的结构变化在损失掩码（loss masking）：loss 只落在回答段上。对拼接后的序列 $x_1, \ldots, x_S$（system / user / assistant 各轮展平成一条），定义

$$
L_{\mathrm{SFT}}(\theta) = -\frac{1}{\sum_s m_s} \sum_{s=1}^{S} m_s \thinspace \log \pi_\theta\negthinspace\left( x_s \mid x_{\lt s} \right), \qquad m_s \in \lbrace 0, 1 \rbrace \qquad (1)
$$

**符号**： $m_s = 1$ 当且仅当位置 $s$ 属于 response（assistant 回答段）； $\sum_s m_s$ 为归一化的 response token 数。这个公式的完整定义、两类被屏蔽位置（prompt 与视觉 token）的机制理由、shift-by-one 对齐、「被 mask 位置不省 forward」的反向语义，都已在 models 8.3 §1（式 (1) 及同节代码走读）交付，本书不重复推导；CE 的数值纪律（logsumexp 稳定、fused 实现）归 3.1。本章回答的是工程一侧的三个问题： $m_s$ 由谁生成（chat template 与 collator，§2 与 §6）；多轮对话里 $m_s$ 的两种取法各自意味着什么梯度（§3）；监督密度 $\eta = \sum_s m_s / S$ 如何进入配方选择（§4）。

先固定一个口径：式 (1) 的归一化按 response token 数取平均（TRL 的默认口径，§6）。若改成按序列求和，长回答的逐 token 梯度就被摊薄——归约位置会改变目标语义，与 3.1 §1 的 mean 归约注记同族。

## 2. Chat template：把消息列表写成 token 序列

SFT 数据集的原始形态不是字符串，而是消息列表——`[{"role": "system"/"user"/"assistant", "content": ...}, ...]`。chat template 是挂在 tokenizer 上的一段 Jinja 模板，也是 HF `apply_chat_template` 的输入。它只做一件事：把消息列表确定性地展平成 token 序列——逐消息包上角色定界符、在序列开头注入 BOS、按需追加 generation prompt 后缀。两家主流实例的实际产物（模板原文分别取自 Meta-Llama-3-8B-Instruct 与 Qwen2.5-7B-Instruct 的 `tokenizer_config.json`，2026-11-18 核实）：

```
Llama-3 系（一轮 user + generation prompt）：
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n          ← add_generation_prompt 追加

Qwen2.5 系（ChatML 格式）：
<|im_start|>system\n{system content}<|im_end|>\n
<|im_start|>user\n{content}<|im_end|>\n
<|im_start|>assistant\n                                  ← add_generation_prompt 追加
```

结构对照值得逐点看。两家的共同骨架是「角色头 + 内容 + 轮终止符」：Llama-3 用 `<|start_header_id|>role<|end_header_id|>` 包角色名、`<|eot_id|>` 结束一轮；Qwen 用 `<|im_start|>role` 起头、`<|im_end|>` 收尾。差异在两处。其一，**轮终止符与文档终止符分离**：Llama-3 对话里的结束符是 `<|eot_id|>` 而非预训练的 `<|end_of_text|>`，Qwen 的 EOS 直接就是 `<|im_end|>`。预训练词表里的「文档结束」与「该轮到对方说话」被拆成两个 token，后训练只用后者。其二，**generation prompt 是不闭合的角色头**：推理时 `add_generation_prompt=True` 在序列尾部追加 `<|start_header_id|>assistant<|end_header_id|>\n\n`（Llama-3）或 `<|im_start|>assistant\n`（Qwen），让模型的第一个采样 token 落在「assistant 内容」的分布上。训练与推理走同一个模板函数：训练时模板输出后接真实回答与终止符，推理时模板输出就是采样起点。

模板机制带出两条训练纪律。第一条是 **EOS 进 loss**：assistant 轮末尾的 `<|eot_id|>` / `<|im_end|>` 必须被 $m_s = 1$ 覆盖，否则模型从未学过「说完就停」，推理时只能靠 max_tokens 截断。TRL 对此有一条显式检查链（`trl/trainer/sft_trainer.py`，2026-11-18 核实）：`assistant_only_loss` 开启时先经 `get_training_chat_template` 自动换成带 `{% generation %}` 标记的训练模板；若换用后终止 token 仍落在生成区之外，初始化只发 `logger.warning`——训练照常继续，warning 原文点明该问题通常到推理期才暴露（模型学不会停）；模板完全缺标记且自动修补无效（assistant mask 全零）时，才在数据准备期抛 RuntimeError。第二条是 **BOS 单份**：模板自身已注入 `<|begin_of_text|>`，tokenizer 侧再注入就成双份——TRL 的 VLM collator 在三处 tokenizer/processor 调用点都带 `add_special_tokens=False` 并附注释（同文件，§6）。

## 3. 数据形态与多轮 masking 的梯度语义

SFT 消费三种数据形态：单轮指令-回答对 $(x, y^{\ast})$；多轮对话 $[u_1, a_1, u_2, a_2, \ldots, u_K, a_K]$（ $u$ 为 user 轮、 $a$ 为 assistant 轮）；prompt-completion 分栏（与第一种同构，只是字段拆分）。多轮形态引出本章第一个推导点： $m_s$ 揭开哪些 assistant 轮。

**两种策略**。全轮监督（all-assistant）把每一轮 assistant 内容都揭开， $m_s = 1 \Leftrightarrow s \in a_1 \cup \cdots \cup a_K$；末轮监督（last-turn）只揭开 $a_K$ 。两种取法的序列布局：

```
多轮样本 [u1 a1 u2 a2 u3 a3]，m_s 的两种取法：
token 段:  [sys] [u1] [a1] [u2] [a2] [u3] [a3]
全轮监督:    0     0    1    0    1    0    1     ← η 高，一条序列喂 3 轮监督
末轮监督:  0     0    0    0    0    0    1     ← η 低，一条序列只喂末轮
两策略的 prompt 段恒为 0；forward 成本相同，差别全在 loss 覆盖
```

**梯度语义**。全轮监督下目标展开为

$$
L_{\mathrm{all}} = -\sum_{k=1}^{K} \thinspace \sum_{s \in a_k} \log \pi_\theta\negthinspace\left( x_s \mid u_1, a_1, \ldots, u_k, a_{k, \lt s} \right) \qquad (2)
$$

**符号**：内层求和的每个 token 以「前 $k-1$ 轮完整对话 + 本轮已生成前缀」为条件。这个条件分布恰是多轮推理时的真实场景：第 $k$ 轮生成时，上下文里的 $a_1, \ldots, a_{k-1}$ 就是模型自己在前几轮的产出，全轮监督让训练分布与推理分布同构。末轮监督把监督面收窄到单一决策点：模型只在「给定完整对话历史续写最后回答」这一个条件分布上被训练。

两种策略的取舍按三条账走。其一，**监督密度**：同一条序列的 forward 成本恒定，全轮监督的 $\eta$ 随轮数近似线性放大——轮数多的对话数据用全轮，同样的 GPU 小时换更多 loss token。其二，**数据质量的暴露面**：全轮监督把早期 assistant 轮也变成监督目标，数据里任何一轮的质量缺陷都会进 loss；末轮监督只对最后一条回答的质量敏感。数据管线若只能保证末轮质量（例如末轮由最新模型生成、前面轮次来自旧模型），末轮是更安全的取法。其三，**两家的实证取法恰好分占两端**：Llama-3 的 SFT 数据结构是「context（除最后一轮外的全部对话轮）+ final response」（herd 论文 Table 7 表注口径，[A]），即末轮监督形态；TRL 的 `assistant_only_loss` 走全轮（模板对每段 assistant 内容都打 `{% generation %}` 标记，§6）。没有公开的跨家消融裁决两者。可观测的共识是「prompt 段恒不监督」——对 $u_k$ 计 loss 等于训练模型续写用户提问，方向与部署时的条件化相反（models 8.3 §1.1 同口径）。

多轮能力的另一个实证锚点在 LIMA 的对照实验里。只用 1000 条单轮样本训出的模型，在 10 段实测对话中有 6 段三轮内脱轨；向训练集追加 30 条多轮对话样本（合计 1030 条）后，人工评为 excellent 的回答比例从 45.2% 升到 76.1%，失败率从 42 轮里 15 次降到 46 轮里 1 次（[Zhou et al., 2023](../references.md)，arXiv:2305.11206，[B]，§6）。多轮数据的需求可以小到两位数——这个量级结论比策略选择本身更值得记住。

**质量压过数量的实证口径**。LIMA 的主实验把「少而精」推到极限：LLaMa 65B 只在 1000 条人工策展样本上 SFT（750 条取自 Stack Exchange / wikiHow 高质量问答、250 条手写），无任何 RL。人工偏好评估中，它对 GPT-4 有 43% 的情形持平或更优，对 Bard 58%、对经 RLHF 训练的 DaVinci003 65%；同基座在 Alpaca 的 52,000 条数据上复现的对照模型，偏好率反而更低（[Zhou et al., 2023](../references.md)，[B]，§4-5）。作者据此提出 Superficial Alignment Hypothesis：模型的知识与能力几乎全部在预训练获得，对齐学习的只是与用户交互的形式或格式。若假设成立，SFT 的数据问题就不是「喂多少知识」而是「示范什么形式」，质量与多样性的权重自然压过数量。三个消融分别隔离三个轴（均在其 §5，7B 规模）：数量轴上训练集指数翻倍、偏好评估几乎不动（16 倍数据增量无显著收益）；质量轴上同来源未过滤的 2000 条比过滤版差 0.5 分（6 分制 Likert——消融序列用 ChatGPT 按 1-6 分打分，其 §5 与 App. D 口径）；多样性轴上 prompt 异质的 Stack Exchange 优于 prompt 同质的 wikiHow（原文文字口径 "significantly higher performance"，其 §5 的 Figure 5，无点值）。结论原文值得照录其形：alignment 的 scaling 不是数量的函数，而是「保持高回答质量前提下的 prompt 多样性」的函数。

数量一极的工业对照同样有出处：InstructGPT 的 SFT 集约 13k 训练 prompt（API 收集 + 标注员撰写，[Ouyang et al., 2022](../references.md)，arXiv:2203.02155，[B]，§3.2）；Llama-3 的 SFT 混合里通用英语占 52.66%、代码 14.89%、推理与工具 21.19%（Table 7，按样本数），且相当部分是模型自产的合成数据（仅代码一项就超过 2.7M 条——其 §4.3.1；拒绝采样产线归 9.4）。两条路线不矛盾：LIMA 回答的是「打通指令格式的下限在哪」，工业配方回答的是「全能力面的覆盖面要多大」——下限由质量决定，覆盖面由数量补齐。

## 4. 配方：LR、epoch 与 packing 的取舍

SFT 超参的逐家出处（全部一手口径；预训练峰值 lr 的出处归 7.2 的对照表）：

| 出处 | SFT lr | epoch | batch | schedule 与其他 |
| --- | --- | --- | --- | --- |
| InstructGPT 175B [B] | $5.03 \times 10^{-6}$ | 16 | 8 | cosine 降至 10%、无 warmup、residual dropout 0.2；1.3B/6B 为 $9.65 \times 10^{-6}$ 、batch 32（App. C） |
| Llama-2 [A] | $2 \times 10^{-5}$ | 未披露 | 64 | cosine、wd 0.1、序列长 4096（§3.1） |
| Llama-3 405B [A] | $10^{-5}$ | 未按 epoch 报 | — | 8.5K–9K steps，跨轮次与数据混合通用（§4.1.3） |
| LIMA 65B [B] | $10^{-5} \to 10^{-6}$ | 15 | 32 | 线性衰减、无 warmup、AdamW wd 0.1（§3） |

**LR 量级**。SFT 的 lr 远低于同模型预训练峰值，这是跨家一致的方向；但比例必须逐家算，不存在普适规则。Llama-3 405B 的 SFT $10^{-5}$ 对预训练峰值 $8 \times 10^{-5}$ 约为 1/8（[Grattafiori et al., 2024](../references.md)，[A]，§3.4.1 与 §4.1.3 对照）；Llama-2 的 SFT $2 \times 10^{-5}$ 对 70B 档预训练峰值 $1.5 \times 10^{-4}$ 约为 1/7.5（[Touvron et al., 2023](../references.md)，[A]，§2.2 与 §3.1）。机制上怎么读（本书分析）：预训练的 lr 峰值是为「从随机初始化出发、梯度尺度跨数量级漂移」设定的（2.3 的 warmup 与噪声尺度论证）；SFT 的起点已在 loss 曲面低处，数据量小 3 个以上数量级，梯度估计的 $\Sigma/B$ 分母里没有足够的 batch 摊薄噪声。小 lr 同时回应两笔账：「步长别毁掉已学结构」与「小数据梯度噪声大」。把 1/8 当默认起点可行，当规则就越界了：预训练峰值本身随规模变化（7.2 §1 的表），比值自然跟着变。

**epoch 的过拟合账**。SFT 的 epoch 数远高于「一遍数据」的直觉；四家的取法要连同选 checkpoint 的判据一起看才有意义。InstructGPT 的 SFT 基线训 16 epoch，但论文明确「按 RM score 而非 validation loss 选模型」——作者发现 RM 打分比验证 loss 更能预测人类偏好（[Ouyang et al., 2022](../references.md)，[B]，App. C）。LIMA 训 15 epoch，checkpoint 在第 5 到第 10 epoch 之间用 50 条留出集手工挑，理由同样是「perplexity 与生成质量不相关」（[Zhou et al., 2023](../references.md)，[B]，§3）。两条出处给出同一个教训：SFT 阶段的过拟合不能用训练 loss 检测——loss 继续降、生成质量先升后降，评估器必须是任务级的（人评 / RM 分 / benchmark）。对照组在 reward model 一侧：RM 训 1 个 epoch 之后 validation loss 即明显恶化（InstructGPT App. C 与 Llama-2 §3.2.2 同口径），判别式目标的过拟合反而能在 loss 上直接读出。生成式与判别式目标的过拟合可观测性不对称——排进监控面板时要用两套判据。

**packing vs padding**。SFT 语料的回答普遍偏短（Llama-3 SFT 数据平均每条 846 token，Table 7），padding 到定长 $T$ 的浪费比预训练更甚——7.1 §6 的效率账原样平移：不 pack 的有效 token 占比 $\eta_{\mathrm{pad}} = \mathbb{E}[L] / T$ ，顺序拼接 pack 约 $1 - \mathbb{E}[L]/(2T)$ 。SFT 侧的增量只有两条口径注记。(i) packing 把多条独立样本拼进同一序列后，跨样本 attention 的处置与预训练同族——块对角 mask + position_ids 重置的 FA varlen 路线（7.1 §6），TRL 的 BFD packing 产出 `seq_lengths` 键并自动切到 padding-free 展平形态（§6）。(ii) SFT 多了一重「截断切断 response」的敏感性：样本尾部是监督目标本体，`max_length` 截断直接砍 loss 覆盖。TRL 默认 `truncation_mode="keep_start"` 保住回答端（`trl/trainer/sft_config.py`）——预训练里无关痛痒的左/右截断选择，在 SFT 里有目标语义。

## 5. Catastrophic forgetting 与 replay 的账

目标分布从预训练混合换成指令数据后，旧分布上的似然不再有任何项值守。catastrophic forgetting 的机制一句话就说完：梯度只朝新分布的高似然区走，参数在旧任务上的位置随之漂移。真正值得拆的是对策的账。

**replay（混入预训练数据）的格式**。把预训练分布上的 CE 以系数 $\gamma_{\mathrm{pt}}$ 加回目标：

$$
L = L_{\mathrm{SFT}} + \gamma_{\mathrm{pt}} \thinspace \mathbb{E}_{x \sim D_{\mathrm{pretrain}}}\left[ -\log \pi_\theta(x) \right] \qquad (3)
$$

**符号**： $D_{\mathrm{pretrain}}$：预训练数据分布。InstructGPT 给了这条思路最完整的实证链，分布在两个位置。第一处是 RLHF 的 SFT 初始化阶段就混入 10% 预训练数据（原文口径「对后续 PPO 训练有帮助」，App. C.3）。第二处在 RL 阶段：同一思想放大为 PPO-ptx 的 $\gamma = 27.8$ 预训练项——该项把 SQuADv2 / DROP 等公开 NLP 基准上的回退拉回来，而单纯把 KL 系数加大 100 倍修不回（[Ouyang et al., 2022](../references.md)，[B]，§4.2 与 App. E.6 的 Figure 34；9.2 §5 展开完整目标式）。代价是直接的：预训练数据按 8 倍于 RL episode 的量混入（App. C.4）后，训练时间约为不带混合的两倍（App. E.11）——replay 的对价是吞吐，不是显存。

两条读法边界需要标注。其一，式 (3) 在**纯 SFT 阶段**的公开消融证据等级低于 RL 阶段：InstructGPT 的 10% 混入是按「对 PPO 有益」记录的，SFT 单独的 forgetting 曲线没有同等级出处（本书注记）。其二，LIMA 的 Superficial Alignment Hypothesis 提供了对照视角：若能力本就在预训练、SFT 只学形式，则「遗忘」更准确的描述是形式覆盖了能力——replay 保住的是旧分布的似然，任务能力的保持是间接效果（本书读法；假设本身出处 [Zhou et al., 2023](../references.md)，[B]，§2）。工程侧还有一招来自 Llama-3：对 RM / SFT / DPO 各阶段用不同数据版本或超参训出的多个模型做权重平均（model averaging，其 §4.1.5）。它不是 replay，但对「单一配方的漂移」起同一方向的缓冲作用。

## 6. TRL SFTTrainer 走读

TRL 的 `SFTTrainer`（huggingface/trl repo，`trl/trainer/sft_trainer.py` 与 `sft_config.py`，教学参照 + 官方实现；行级事实 2026-11-18 于 main 核实）对纯文本与多模态数据走两条不同的掩码生产线，先把这条分工说清楚再分头走读。

**文本数据的掩码在数据准备期一次性生成**。`_prepare_dataset` 里的 `tokenize_fn`（同文件 L1290 / L1376 附近）按字段三分流：`text` 字段是纯语言建模（不 mask，退化为继续预训练形态）；`prompt` / `completion` 分栏样本分别 tokenize prompt 与 prompt+completion，以 `common_prefix_length` 定分界、生成 `completion_mask`（prompt 段 0、completion 段 1）；`messages` 对话样本调 `apply_chat_template(..., return_assistant_tokens_mask=...)` 取 `assistant_masks`。labels 的 −100 折叠也在准备期完成——运行时 collator `DataCollatorForLanguageModeling`（同文件 L89）的 docstring 把分工写明：「不应计入 loss 的 token 应已在 labels 里置 −100，SFTTrainer 在 dataset preparation 处理」，collator 本体只对齐 padding（padding-free / BFD packing 时改出 `position_ids` 与 `cu_seq_lens`）。

**多模态数据的掩码在 collator 期逐 batch 生成**。`DataCollatorForVisionLanguageModeling`（同文件 L270）下的 `_collate_prompt_completion`（L431）做两侧不对称 padding 后拼接（prompt 左 pad、completion 右 pad、`torch.cat` 成 `input_ids`、`completion_mask` 随行生成），`_collate_language_modeling`（L380）走 `apply_chat_template` 取 `assistant_masks`。式 (1) 的 $m_s$ 在两条生产线上都是布尔索引置 −100，只是生成的位置不同。

还有三个跨路径的机制细节。其一，`completion_only_loss` 的默认值不是固定的：数据集同时含 `prompt` 与 `completion` 键时自动开启（源码注释口径「default not used in practice; SFTTrainer always passes the relevant value」）。其二，`assistant_masks` 依赖模板里的 `{% generation %}...{% endgeneration %}` 标记圈出每段 assistant 内容——全轮监督（§3）由此实现；标记缺失时的警告/报错链归 §2。其三，文本准备期有一个值得知道的边界处理：分界不按「prompt 长度」而按 `common_prefix_length`。若模板把 prompt 单独渲染与接续渲染得不一致（或边界处 token 合并），prompt 尾部会被留在训练序列之外、并发一次性警告（`tokenize_fn` 内 `logger.warning_once`）——这是 chat template 与分词不复合时的标准处置。

**吞吐开关**：`packing=True` 默认走 `"bfd"`（best-fit decreasing：样本按长度降序、贪心填入剩余空间最小的箱子，溢出截断；另有 `"bfd_split"` 与 `"wrapped"` 两档——`sft_config.py` 字段 docstring），产出 `seq_lengths` 键后自动进入 padding-free 形态——batch 内样本展平为单条长序列、以 position ids 与 FA varlen 边界分隔（7.1 §6 的块对角处置同族），padding 开销归零。`max_length` 默认 1024、`truncation_mode` 固定 `"keep_start"`（§4 的截断口径）。

教学骨架（教学用，非优化实现）把两条通路的 labels 语义合并成最小实现：

```python
# 教学用，非优化实现：response-only labels 的构造（对应式 (1) 的 m_s）
def build_sft_labels(input_ids, attention_mask, response_mask, ignore_index=-100):
    labels = input_ids.clone()
    labels[attention_mask == 0] = ignore_index    # pad 不监督（两通路共有）
    labels[response_mask == 0] = ignore_index     # 非 response 不监督（SFT 增量）
    return labels
# TRL 的对应物：prompt-completion 通路的 response_mask = completion_mask；
# conversational 通路的 response_mask = assistant_masks（{% generation %} 标记产出）
```

与真实实现的差别：VLM collator 另处理左 pad 冲洗（`flush_left`）、多模态张量、vision 数据集禁 packing 等分支；骨架只保留掩码语义。PEFT 的位置在此补一句：LoRA 把权重增量参数化为低秩乘积、冻结基座（公式与「起点即原模型」的零初始化性质在 models 9.4 §6 式 (10) 交付，该式位于视频编辑章内，纯文字指引、不重复推导）。训练侧的全部差异只在配置层：optimizer 状态账随可训参数数缩（2.2 的 $(4+2c)N$ 口径直接改 $N$ ）、lr 惯例上探、式 (1)(3) 的目标形态一字不改。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| SFT 目标 | 预训练同形 CE + response-only 掩码（式 (1)）；定义归 models 8.3 式 (1)，本章增量在工程语义 | 式 (1) |
| chat template | 角色头 + 内容 + 轮终止符的确定性展平；generation prompt = 不闭合 assistant 头；EOS 进 loss 与 BOS 单份两条纪律 | §2，TRL 检查项 |
| 多轮 masking | 全轮监督：训练分布与多轮推理同构、 $\eta$ 高、暴露全部轮次质量；末轮监督：Llama-3 的 context + final response 形态（Table 7） | 式 (2) |
| 质量 > 数量 | LIMA：1000 条对 GPT-4 持平或更优 43%；16 倍数量无显著收益、质量 0.5 分差、多样性显著；Superficial Alignment Hypothesis | arXiv:2305.11206 [B] |
| LR 惯例 | 远低于预训练峰值，逐家：Llama-3 ≈1/8、Llama-2 ≈1/7.5、InstructGPT 175B $5.03 \times 10^{-6}$ ；比例无普适规则 | §4 表 |
| epoch 账 | SFT 15–16 epoch 存在（LIMA / InstructGPT）但 checkpoint 按任务级评估选（RM score / 人评）；loss 不可作过拟合判据；RM 侧 1 epoch 即过拟合 | arXiv:2203.02155 App. C |
| packing | BFD + padding-free；SFT 特有敏感性：截断切断监督目标，`keep_start` 保回答端 | §4；7.1 §6 |
| forgetting | replay 式 (3)；InstructGPT SFT-init 混 10% 预训练；RL 侧 $\gamma$=27.8 归 9.2；纯 SFT 消融证据弱（标注） | 式 (3) |

## 练习

1.（推导）式 (2) 的梯度语义：设三轮对话 $[u_1, a_1, u_2, a_2, u_3, a_3]$ 且训练数据中的 $a_1, a_2$ 由旧策略 $\pi_{\mathrm{old}}$ 生成。写出全轮监督与末轮监督各自的目标期望（注意条件分布中的 $a_1, a_2$ 来自谁），并论证：当 $\pi_\theta$ 偏离 $\pi_{\mathrm{old}}$ 后，全轮监督在第 2、3 轮上引入的条件失配方向是什么（提示：推理时前缀来自 $\pi_\theta$ 自身）。为什么这个失配不属于 8.x 的 off-policy 问题（3.2 §5 的分界判据）？

2.（计算）监督密度账：Llama-3 SFT 数据平均 846.1 token / 条（context 535.7 + response 310.4，Table 7 口径），训练序列长 $T$ = 4096。(a) 不 pack 的 $\eta_{\mathrm{pad}}$ 与 response 监督密度各是多少？(b) 顺序 pack 后（7.1 式 (8)）每序列有效 token 数与监督 token 数各升到多少？(c) 若把监督策略从末轮改为全轮、平均轮数 4.7，监督 token 数近似怎么变？说明哪个量不变、为什么。

3.（计算）replay 成本：按 InstructGPT 的口径，预训练数据以 8 倍于 RL episode 的量混入后训练时间约翻倍（App. E.11）。设某 SFT 配方以 $\gamma_{\mathrm{pt}} = 0.1$ 混入预训练 token、要求总目标中预训练项贡献的梯度范数份额不超过 10%，在「两项 loss 量级相近」的假设下，混入的预训练 token 占比上限是多少？把结果与 InstructGPT SFT-init 的 10% 混入对照，讨论系数的两种定法（loss 加权 vs 数据配比）何时等价。

4.（代码）按 §6 的骨架实现最小 SFT collator（教学用，非优化实现）：输入两条 prompt-completion 样本，完成左 pad prompt / 右 pad completion / 拼接 / labels 构造；用一个小 LM 前向，验证 (a) prompt 位置 logits 不进 loss，(b) 交换 padding 侧后 loss 不变，(c) 把 response 末尾的 EOS token 从 mask 中剔除后，模型在若干步过拟合后不再生成 EOS——对照 §2 的纪律一。

## 参考

- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——SFT lr $10^{-5}$ 与 8.5K–9K 步（§4.1.3）、SFT 数据结构 context + final response 与数据构成（Table 7）、六轮迭代（§4.1.6）、模型平均（§4.1.5）、405B 预训练峰值 $8 \times 10^{-5}$（§3.4.1）
- [Zhou et al., 2023](../references.md)：LIMA: Less Is More for Alignment，arXiv:2305.11206，[B]（key: lima）——1000 条配方与 15 epoch 细节（§3）、偏好评估 43% / 58% / 65%（§4-5）、数量/质量/多样性三消融（§5）、多轮 30 条对照（§6）、Superficial Alignment Hypothesis（§2）
- [Touvron et al., 2023](../references.md)：Llama 2，arXiv:2307.09288，[A]（key: llama2）——SFT 配方 lr $2 \times 10^{-5}$ / batch 64 / seq 4096（§3.1）、RM 1 epoch 过拟合口径（§3.2.2）
- [Ouyang et al., 2022](../references.md)：Training language models to follow instructions with human feedback，arXiv:2203.02155，[B]（key: instructgpt）——SFT 13k prompt（§3.2）、SFT 16 epoch 与 RM score 选型（§3.5、App. C.1）、SFT-init 10% 预训练混入（App. C.3）、PPO-ptx $\gamma$=27.8 与 8 倍混入（App. C.4 / E.11）、KL 加大百倍修不回（§4.2 与 App. E.6 Figure 34）
- [HuggingFace, huggingface/trl repo](../references.md)：教学参照 + 官方实现（key: trl-repo）——`trl/trainer/sft_trainer.py`（两个 collator、`completion_only_loss` 自动推断、`assistant_only_loss` 与 `{% generation %}` 检查）、`trl/trainer/sft_config.py`（packing 默认 `bfd`、`truncation_mode="keep_start"`）；2026-11-18 核实
