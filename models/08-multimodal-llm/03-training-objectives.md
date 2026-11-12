# 8.3 VLM 训练目标与数据形式（Training Objectives and Data Formats）

> 本章回答「VLM 的训练信号长什么样」：损失掩码（loss masking）——next-token loss 只在 response token 上计，prompt 与视觉 token 全部屏蔽（公式 + 代码走读）；多阶段配方——stage 1 对齐只训 connector、stage 2 指令微调解冻 LLM、stage 3 偏好对齐与幻觉专项数据构造；交错图文的数据形式——`<image>` 占位符展开机制与多图位置编码排布；幻觉的形式化与机制假设——language prior 压过视觉证据的 KL 分析，数据侧与解码侧缓解（VCD 对比解码公式）；评测方法论——benchmark 类型学、多选题局限、claimed vs 真实能力 | 前置：1.1（LM 目标与 NLL）、2.4（causal mask 与 KV cache）、8.1（组件链、input-level 拼接式 (3)、冻结策略 §5）、8.2（M-RoPE 三元组式 (14)、Q-Former 两阶段 §3.4） | 锚点：LLaVA-OneVision（训练配方全开放）[A]（LLaVA-VL/LLaVA-NeXT）、Qwen2.5-VL [A]、Molmo（数据全开放）[A]（allenai/Molmo）、haotian-liu/LLaVA（loss masking 走读）[A]、VCD [A]

## 1. loss masking：loss 只落在 response 上

### 1.1 公式与两类被屏蔽的位置

8.1 式 (1) 把 VLM 形式化为条件 LM；训练它就是最大化训练样本中回答部分的似然。设一条样本拼接后的完整序列为 $x_1, \ldots, x_S$（system + user prompt + 视觉段 + assistant 回答，视觉段占其中 $M$ 个位置），定义掩码：

$$
m_s = \begin{cases} 1, & s \in \mathrm{response} \\ 0, & \mathrm{otherwise} \end{cases}, \qquad L = -\sum_{s=1}^{S} m_s \log p\!\left( x_s \mid x_{<s} \right) \qquad (1)
$$

**符号**：$S$：拼接后总长度（文本 + 视觉段）；response 指 assistant 回答段；$x_{<s}$ 里已经混着视觉段——input-level 接入下（8.1 式 (3)）它是序列的一部分，条件化不需要任何特殊记号。式 (1) 与纯文本 SFT 的 loss 形式完全一致：VLM 的训练目标没有引入任何新 loss，全部的新东西在「$m_s$ 怎么取」。

被 mask 掉的两类位置各有机制理由。**视觉 token**：$v$ 是连续向量行（8.1 式 (2)），不在文本词表上，输出头的 softmax 根本给不出它的标签——没有 ground truth 可计；它们在序列里的角色是条件（建 KV、供后续 attend），不是预测目标。**prompt token**：建模对象是 $p(\text{response} \mid \text{prompt},\, x_{img})$，prompt 是给定的条件；对 prompt 计 loss 等于让模型学「生成用户的问题」——方向与推理时的条件化相反，还把有限的监督容量花在复述输入上。一个特例：stage 1 的 caption 预训练里，回答就是整条输出（图 → caption），response 覆盖 caption 全部 token；多轮对话样本里 response 是各 assistant 轮的并集，$m_s$ 逐段揭开。

```
一条单轮样本的序列布局与 m_s：
[system] [user 前半] [v_1 … v_M] [user 后半] [assistant 回答]
   0         0           0           0           1 … 1      ← m_s
   └──────── 条件段：建 KV、供 attend，不计 loss ────────┘  └ response ┘
```

### 1.2 代码走读

实现只有一句话：把 labels 的非 response 位置填成 PyTorch CrossEntropyLoss 的 ignore_index（约定 $-100$，该位置不参与 loss 与梯度）。教学版（非优化实现；对照 LLaVA 官方 repo 训练侧的 labels 构造，[llava-repo](../references.md)，[A]）：

```python
def build_labels(input_ids, response_spans, ignore_index=-100):
    # input_ids: [B, S]，<image> 占位符已在序列中占位（§3.1 展开前的形态）
    labels = torch.full_like(input_ids, ignore_index)   # 默认全屏蔽
    for b, spans in enumerate(response_spans):
        for lo, hi in spans:                            # 每段 assistant 回答
            labels[b, lo:hi] = input_ids[b, lo:hi]      # 只在 response 段揭开
    return labels
```

三个实现细节。(a) shift-by-one 对齐：位置 $s$ 的 logits 预测 token $s+1$（HF causal LM 在 forward 内部完成移位），`build_labels` 返回与 `input_ids` 等长的张量，移位不在此处做。(b) `<image>` 占位符与展开后的视觉段在 labels 里恒为 $-100$，padding 同理。(c) **被 mask 的位置不省 forward**：它们照常过全部 $L$ 层、建 KV cache、被 response attend——mask 只作用于 loss 一项。这条对反向传播同样成立：视觉段的表示参与了 response 的预测，梯度沿 attention 路径回传到 connector（以及被解冻的视觉塔与 LLM）——冻结双塔的 stage 1 里，connector 收到的正是这股「透过 mask 的梯度」。

### 1.3 有效 token 账

训练 FLOPs 按全序列 $S$ 计，监督信号只按 $\sum_s m_s$ 计，两者之比 $\eta = \sum_s m_s / S$ 是这条样本的监督密度。算一笔典型账：AnyRes 4 切图 $M = 2880$（8.1 §4 口径）+ system/prompt 500 token + 回答 150 token，$\eta \approx 150 / 3530 \approx 4\%$。$\eta$ 低不等于浪费——视觉段与 prompt 的 forward 是条件化的必要成本；但它解释了数据工程的两个共同动作：把多条 QA 打进一条序列（packing，摊薄条件段的重复成本），以及把回答写长写密（dense caption，§2.2 的 Molmo 路线——同样的视觉 forward 成本，换取更多 loss token 且逐对象接地）。$\eta$ 也是读各家数据配方的一把尺：同样的 token 总量，监督密度可以差出一个量级。

## 2. 多阶段配方：对齐 → 指令 → 偏好

当代开放 VLM 的训练不是一次 loss 而是一条流水线：同一个式 (1)，换数据、换冻结开关、跑若干阶段。各家 report 的结构交集是三段（LLaVA 系、BLIP-2、Qwen2.5-VL、Molmo 的共同骨架；BLIP-2 的两阶段已在 8.2 §3.4 从 connector 视角拆过，本节从配方视角看全貌）：

```
stage 1 对齐             stage 2 指令微调                stage 3 偏好对齐
冻 ViT、冻 LLM           解冻 LLM（+connector）          参数面同 stage 2
只训 connector     ──►   多任务指令混合（含纯文本）──►     chosen/rejected 偏好对
caption 数据             模板与 special token 定型        幻觉专项（§2.3）
```

### 2.1 stage 1：对齐预训练——只训 connector

冻结 ViT 与 LLM，只训 $\phi$；数据是 web 规模 image-caption 对（LLaVA-1.5 口径 558K 条，8.2 §2 已锁；工业配方大几个量级）。这一阶段学什么，8.1 §5 已给过论证：几十万级图文对就能把 $v$ 对齐进 LLM 的 embedding 空间 ⇒ connector 学的是**空间对齐**（把视觉塔已备好的语义特征变换进 LLM 可读的坐标系），不是内容补全。8.2 §1 留下的「自由度 (iii) 表征翻译由训练目标完成」，兑现处就是这里：式 (1) 的梯度只经过 $v$ 回传，$\phi$ 被迫把视觉特征摆进 LLM 词表近旁「可续写」的区域——对齐是 loss 几何的推论，不是额外设计的模块。

为什么先冻双塔：caption 数据噪声大（alt-text 与图像内容错位是常态），一上来就解冻 LLM，低质文本分布会直接污染语言能力；冻结时全部对齐误差由千万级参数的 connector 吸收，双塔的预训练资产零风险。stage 1 的产物会看图、会写 caption，但指令遵循为零——它没见过指令格式。

### 2.2 stage 2：指令微调——解冻 LLM

解冻 LLM（与 connector 一起；ViT 多数家仍冻，Qwen 系例外——vision tower 参与训练，8.1 §5 冻结矩阵）。数据换成多任务指令混合：VQA、OCR、文档、grounding、图表、多轮对话，外加一份纯文本指令数据（LLaVA-1.5 的指令混合总量 665K 条、含纯文本份额，引原表）。

两个数据教训是各家 report 的公约数。(a) **纯文本数据必须保留**：多模态混合里不放文本指令，LLM 原有的指令遵循与长文能力会退化——冻结派与解冻派都承认的遗忘风险，只是用不同方式付费（前者付融合深度，后者付文本数据配额）。(b) **全部任务统一灌进同一对话模板**：special token（`<image>`、轮次标记、角色标记）的用法在这一阶段定型——§3.3 将看到，这套模板从此就是模型的分布内格式。

两个配方变体值得单列。**渐进迁移**（LLaVA-OneVision，[Li et al., 2024](../references.md)，arXiv:2408.03326，[A]）：stage 2 内部再分单图 → 多图/视频，后阶段在前阶段 checkpoint 上续训，论文 ablation 支撑「视频能力可以从图像迁移获得」的取径（8.4 展开）。**数据全开放**（Molmo，[Deitke et al., 2024](../references.md)，arXiv:2409.17146，[A]）：PixMo 的 caption 不是 web 爬的 alt-text，而是标注者口述图像内容再转写的 dense caption——单条覆盖的对象、属性、空间关系密度远超 web caption，直接对应 §1.3 的 $\eta$ 账；另有把对象坐标写进回答文本的 pointing 数据，把位置监督变成普通的 next-token 目标。数据与配方全部随权重开放（[allenai/Molmo](../references.md)，[A]），这也是本章把它列为锚点的原因：配方类断言第一次可以逐条对照原始数据复算。

### 2.3 stage 3：偏好对齐与幻觉专项数据

偏好对齐的算法本体（reward model、DPO/RLHF 族）归 training 维度，本书不展开；VLM 侧独有的增量是**偏好数据的构造对象**。文本侧的偏好标注没有 ground truth 锚点，标注质量靠人味；VLM 有锚——图像本身就是：对同一图采样模型回答，用检测器或人工核对断言的对象存在性，把「图像支持的回答」与「含幻觉对象的回答」配成 chosen/rejected 对。负样本还可以半自动构造：往 faithful caption 里注入图中不存在的对象或错误属性，即得 rejected。当代开放旗舰的配方均含这一阶段（Qwen2.5-VL 的 tech report 口径，[Bai et al., 2025](../references.md)，arXiv:2502.13923，[A]）。§4 将看到，这一阶段治理的正是「language prior 压过视觉证据」的机制后果——偏好信号把「说图里没有的东西」的隐式奖励压成负的。

### 2.4 各家配方对照

| 系统 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- |
| LLaVA-1.5 [A] | 558K caption；冻双塔、只训 MLP | 665K 指令混合；解冻 LLM | 无 |
| LLaVA-OneVision [A] | 沿用前序 checkpoint | 单图 → 多图/视频渐进 SFT | （论文范围外） |
| Qwen2.5-VL [A] | 多阶段预训练；ViT 参与训练 | 多任务 SFT | 偏好对齐（tech report 口径） |
| Molmo [A] | caption 预训练（PixMo dense caption） | SFT（PixMo 混合） | （引论文原表） |

读法：三段的分工是「先对齐空间（stage 1）、再学任务格式（stage 2）、最后校准行为偏好（stage 3）」——冻结开关逐段放开、数据从粗到精、可训参数量级逐段放大。阶段划得越细，对数据质量的容忍度越高：噪声最大的 web caption 只接触 connector，最精的偏好数据作用于全参数。一段式端到端配方不是不存在，而是把三笔风险（噪声污染、能力遗忘、偏好未校准）合并支付。

## 3. 交错图文的数据形式

### 3.1 `<image>` 占位符展开

数据集的原始样本是「文本 + 图像文件指针」，图像不进 tokenizer。交错图文（interleaved image-text）样本的机制链条三步：(i) 文本里的 `<image>` 被编成 special token id，在 ids 序列里占一个位置；(ii) collate 时按 8.1 式 (3) 把该位置展开成 $M$ 个视觉 embedding 行——实现两路：id 层把占位符重复 $M$ 次、embedding 查表后整段替换为 $v$（替换必须在查表之后：占位符 id 查出的行是废值），或者干脆不把占位符编进 ids、按段拼接 embedding；(iii) labels 同步——占位符与视觉段恒为 $-100$（§1.2 细节 (b)）。

展开后序列长度从 $S$ 变为 $S + M - 1$：attention、位置编码、loss mask 全部作用在展开后的序列上。对训练框架而言，图像段就是一段「不在词表里但形状合法的 embedding 行」（8.1 §3.1）——交错图文数据形式的全部工程就在「占位、展开、屏蔽」三件事上，没有第四种机关。

### 3.2 多图的位置编码排布

多图样本 = 多个 `<image>` 占位符按出现顺序逐段展开。位置编码用 8.2 式 (14) 的 M-RoPE 三元组：每图内部 $(h, w)$ 网格从各自段基址起、互不干扰；文本计数跳过视觉段占掉的 $\max(T, H_g, W_g)$ 区间（8.2 §9.4 伪码的 `pos` 推进规则）；图与图的先后由段基址 $t_0$ 的递增表达。一条推论直接可用：**图像段的插入不改变文本段的相对结构**（8.2 §9.4 性质 (a)：RoPE 的 score 平移不变 ⇒ 文本内部任意两 token 的差不变）——这就是为什么交错图文训练不需要为「图插在句中」发明新的位置方案：每图领走一段三元组地址，文本的 1D 结构原样保留。

### 3.3 训练/推理格式一致性

对话模板、special token、占位符位置、图像在轮次内的默认位置，合起来构成模型的分布内格式。训练时图恒在 user 轮开头的模型，推理时把图插到句中，位置编码与 attention 统计即刻出训练分布——不是能力缺失，是格式失配。部署纪律随之很硬：推理 prompt 模板必须与训练模板逐 token 一致（含空格与换行），这是 VLM 工程事故的高发点。交错图文预训练的价值正在于把「图在任意位置」变成分布内行为：Flamingo 的网页图文交错数据（图像出现在文本序列的任意位置，[Alayrac et al., 2022](../references.md)，arXiv:2204.14198，[B]）支撑了 few-shot 图文交错 prompt 的形态——序列里先给若干「图 + 说明」示例、最后给待答的图，模型按同样的交错规律续写。开放配方（LLaVA-OneVision 的多图/视频阶段、Qwen2.5-VL 的多图数据）把同一能力收进指令微调的数据配比里。

## 4. 幻觉：形式化、机制假设与缓解

### 4.1 定义与操作化

物体幻觉（object hallucination）：生成文本断言了图像不支持的内容——不存在的对象、错误的属性/计数/关系、编造的文字内容。形式化：把回答 $y$ 解析成原子断言集合 $A(y)$（「图中有一只猫」「猫是黑的」各算一条），记图像证据支持的断言集为 $S(x_{img})$（由人工或 oracle 判定），幻觉率：

$$
\rho = \frac{\left| A(y) \setminus S(x_{img}) \right|}{\left| A(y) \right|} \qquad (2)
$$

操作化两条路线：**判别探针**（问「图中有 X 吗」，统计 yes/no 准确率——可控、可自动评分，但只测存在性这一类断言）与**开放生成**（自由 caption 后标注断言真假——贴近部署形态、标注贵）。VCD 论文的评测协议兼用两者（[Leng et al., 2023](../references.md)，arXiv:2311.16922，[A]）。

### 4.2 机制假设：language prior 压过视觉证据

为什么 VLM 会「不看图说话」？把模型的输出分布与同一模型闭眼（不给图）的分布对比，定义逐位置的视觉接地强度：

$$
G_t = D_{KL}\left( p(\cdot \mid y_{<t},\, x_{img}) \;\Vert\; p(\cdot \mid y_{<t}) \right) \qquad (3)
$$

**符号**：右端是同一模型在视觉条件缺失下的输出分布。$G_t \approx 0$ 意味着第 $t$ 步的生成与闭眼无异——该位置的词是语言先验选的。压低 $G_t$ 的三个机制性因素（机制假设级表述：各论文的探针与消融方向一致，但「VLM 内部何时何地丢失视觉证据」没有闭合理论，本节不断言超出论文的结论）：

(a) **监督的不对称**：式 (1) 只监督文本 token——caption 的文本先验（「街景常配车」「餐桌常配餐具」这类对象共现统计）是降低训练 loss 的合法 shortcut，视觉证据只是特征之一，先验在式 (1) 下不受任何惩罚；
(b) **条件占比与稀释**：视觉段 $M$ 相对全序列小，decode 到长回答后段时文本前缀主导 attention（8.1 §4 的 token 账在生成侧的再现）；经验签名与此一致——VCD 论文报告回答越长、生成位置越靠后，幻觉断言占比越高（引论文口径）；
(c) **数据的先验泄漏**：web caption 本身不完全接地（alt-text 与图像内容错位），先验从 stage 1 起就被写进 connector 与 LLM。

式 (3) 同时给出一个探针：逐位置算 $G_t$、找生成路径上 $G_t$ 塌陷的区间，就是「闭眼续写」的高发位。注意方向：$G_t \approx 0$ 推出该位置不接地，是幻觉的高发条件；反过来 $G_t$ 大并不保证 faithful（图像可以改变分布、但改向错误方向）——精确表述与反例构造留作练习 3。

### 4.3 缓解：数据侧与解码侧

**数据侧**（治本）：(i) dense caption——把「图里有什么」写全，Molmo/PixMo 的口述转写路线（§2.2）直接压先验泄漏；(ii) 负样本监督——「图中没有 X」类样本教模型拒绝不存在的对象，偏好对构造见 §2.3；(iii) 位置/计数监督——pointing 数据把对象坐标写进回答，断言必须落到像素才算对。

**解码侧**（事后，不碰权重）：视觉对比解码（visual contrastive decoding，VCD，[Leng et al., 2023](../references.md)，arXiv:2311.16922，[A]；官方代码 DAMO-NLP-SG/VCD）。推理时并行构造一个视觉条件受损的分布，把先验成分从主分布里减掉：

$$
\tilde p(y_t) = \mathrm{softmax}\left( (1 + \alpha)\, \ell_{img}(y_t) - \alpha\, \ell_{noise}(y_t) \right) \qquad (4)
$$

**符号**：$\ell_{img}$：原条件下输出头的 logit（论文记号口径）；$\ell_{noise}$：图像经扩散前向加噪、视觉内容被摧毁后的条件 logit；$\alpha > 0$：对比强度。机制读法：受损条件下视觉证据被抽掉、剩下的主要是语言先验与共现偏差，减法恰把「不看图也会说的成分」从主分布里压掉——对照式 (3)，VCD 是推理时放大 $G_t$ 的操作。论文同式还带一条可信度截断：只在原分布高置信的候选 token 集上施加对比、集外置零——否则减法会把主分布里概率近零的 token 的 logit 差放大成奇异峰（推导留作练习 2）。代价账：每步 decode 两次前向（原条件 + 受损条件），推理成本翻倍；$\alpha$ 逐任务调。

定位一句话：VCD 是事后补丁，治本是数据与训练侧；它的存在同时构成对 §4.2 机制假设的操控验证——若幻觉与语言先验无关，减先验就不该有效，而它有效（引论文主表）。

## 5. 评测方法论

### 5.1 benchmark 类型学

各家 tech report 评测表的结构公约数是四类（Qwen2.5-VL、LLaVA-OneVision、Molmo 的原表均按此组织；分数引原表，本章不转载）：

| 类型 | 测什么 | 典型形式 |
| --- | --- | --- |
| 感知 | 对象/属性/计数/空间关系 | VQA、存在性探针 |
| OCR/文档 | 文字、表格、图表、文档结构 | OCR 问答、文档解析 |
| 推理/知识 | 多步推理、数学、学科知识 | 多选题为主 |
| 幻觉率 | 断言接地程度 | yes/no 探针 + 开放 caption 标注（§4.1） |

四类的分工对应本书的组织逻辑：感知与 OCR 直接受 8.2 的分辨率/connector 选择影响（小字账的下游）；推理/知识更多反映 LLM 本体的能力；幻觉率独立成类，因为它测的是 §4 的机制而非一般能力——一个感知分高的模型可以完全不妨碍它幻觉率也高（看得清与说得实是两件事）。

### 5.2 多选题的三个坑

多选题是评测表的主力形式，也是系统性偏差的主要来源。

(a) **选项偏置**：模型对选项标记与顺序有位置先验，打乱选项顺序分数会动——同一模型同一题、不同排布不同分，分差里有多少是能力、多少是偏好，需要协议层面控制（循环排布取平均是常见做法）。

(b) **猜测基线压缩信号区间**：$k$ 选 1 的随机基线是 $1/k$（4 选 1 即 25%），真实信号被压在 $[1/k,\, 1]$ 的区间里——两个模型差 5 个点，在 75 个点的有效区间上读与在 100 个点刻度上读，观感完全不同；报分差前先把基线减掉。

(c) **判别-生成错位与盲测泄漏**：多选测「从选项里挑」，部署形态是开放生成；且相当比例的题目可以不看图、靠文本先验答对（选项本身泄漏信息、常识可解）——若同一 benchmark 上纯文本盲测基线显著高于随机，说明题目本身可被语言先验破解，视觉增益要从分数里单独减去这项。这条与 §4.2 是同一枚硬币的两面：先验压过证据既是幻觉的机制，也是虚高分的机制。

### 5.3 claimed vs 真实能力

tech report 的分数是「在作者的评测协议下」的数字：prompt 模板、few-shot 设置、答案抽取规则（正则还是模型判分）每一项都能移动若干点；web 规模训练数据与公开测试集的重叠不可完全排除（基准污染）。读表纪律三条：**看协议**（模板与抽取规则是否公开可复现）、**看基线**（随机与盲测各是多少）、**看开放生成证据**（多选高分不保证自由回答 faithful——§4.1 两条操作化路线给出的模型排名可以不一致，幻觉专项探针的意义正在于此）。claimed 能力与真实能力的差，就是协议、基线、形态三项未对齐部分的和；对工程选型的含义很直接——自己的任务分布上跑一个小规模开放生成评测，胜过搬运任何一张总表。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| loss masking | 式 (1) | loss 只落在 response；视觉 token 无 ground truth、prompt 是条件非目标；labels 置 $-100$，被 mask 位置不省 forward、梯度照常回传 |
| 有效 token 账 | $\eta = \sum_s m_s / S$ | AnyRes 图 + 短回答下 $\eta$ 可低至约 4%；packing 与 dense caption 是提 $\eta$ 的两条数据工程路线 |
| 多阶段配方 | §2.4 表 | stage 1 对齐只训 connector（冻双塔）→ stage 2 指令解冻 LLM（纯文本数据必留）→ stage 3 偏好对齐（幻觉专项 chosen/rejected；算法归 training 维度） |
| 交错图文 | §3 | `<image>` 占位 → 展开成 $M$ 行 → labels 恒 $-100$；多图 M-RoPE 段基址递增、文本相对结构不变；训练/推理模板须逐 token 一致 |
| 幻觉形式化 | 式 (2)(3) | 断言集减证据集；$G_t$ = 视觉接地强度，$G_t \approx 0$ 即闭眼续写；三机制：监督不对称、条件稀释、先验泄漏 |
| VCD | 式 (4) | $(1+\alpha)\,\ell_{img} - \alpha\,\ell_{noise}$ + 候选集截断；每步两次前向换幻觉率下降 |
| 评测方法论 | §5 | 四类 benchmark；多选题三坑（选项偏置、$1/k$ 基线、判别-生成错位）；读表三看（协议、基线、开放生成证据） |

## 练习

1.（计算）一条样本：system 100 token + user prompt 200 token + AnyRes 4 切图（$M = 2880$）+ assistant 回答 150 token。(a) 算 $\eta$。(b) 把 4 条同构 QA pack 进一条序列（各带各的图，共享一份 system），序列总长与 $\eta$ 各是多少？(c) 解释为什么 $\eta$ 不影响 forward FLOPs、只影响监督密度；被 mask 的视觉 token 在反向传播里真的没有梯度吗（提示：它们对 response 的 loss 有贡献路径）？
2.（推导）VCD 式 (4)。(a) 证明当 $\ell_{noise} = \ell_{img}$（视觉条件无贡献）时 $\tilde p = \mathrm{softmax}(\ell_{img})$。(b) 设 token $y^{*}$ 满足 $\ell_{img}(y^{*}) \ll 0$ 且 $\ell_{noise}(y^{*}) < \ell_{img}(y^{*})$（主分布几乎不要它、受损分布更排斥它）：把式 (4) 的分子项改写为 $\ell_{img} + \alpha(\ell_{img} - \ell_{noise})$，分析它随 $\alpha$ 的行为，说明候选集截断防的是哪种失效模式。(c) 讨论 $\alpha \to \infty$ 时 $\tilde p$ 集中到什么样的 token 上。
3.（推导）式 (3) 的 $G_t$。(a) 用 KL 散度的性质说明 $G_t \ge 0$、$G_t = 0$ 当且仅当两分布相等。(b) 构造一个 $G_t$ 大但回答仍幻觉的例子（提示：图像可以改变分布、但改向错误方向），据此说明「$G_t$ 小是幻觉的高发条件而非必要条件」。(c) 讨论用 $G_t$ 做幻觉探针的漏报与误报来源。
4.（代码）教学实现 VCD 解码一步（非优化实现）：输入 `logits_img`、`logits_noise`（均 $\mathbb{R}^{|\mathcal{V}|}$）、$\alpha$、截断阈值 $\beta \in (0, 1)$；先按 $p_{img} \ge \beta \cdot \max p_{img}$ 取候选集，再在候选集上计算式 (4)、集外置零，输出归一化分布。写清与贪心采样的接口。

## 参考

- [Li et al., 2024](../references.md)：LLaVA-OneVision: Easy Visual Task Transfer，arXiv:2408.03326，[A]（key: llava-onevision）——§2.2 渐进迁移配方（单图 → 多图/视频）、§2.4 配方对照
- [Bai et al., 2025](../references.md)：Qwen2.5-VL Technical Report，arXiv:2502.13923，[A]（key: qwen2.5-vl）——§2.3 偏好对齐阶段、§2.4 多阶段配方口径
- [Deitke et al., 2024](../references.md)：Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models，arXiv:2409.17146，[A]（key: molmo）——§2.2 dense caption 与 pointing 数据、§4.3 数据侧缓解
- [Leng et al., 2023](../references.md)：Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding，arXiv:2311.16922，[A]（key: vcd）——§4.1 评测协议、§4.2 幻觉随回答长度升高、§4.3 VCD 式 (4) 与候选集截断
- [Liu et al., 2023](../references.md)：Visual Instruction Tuning，arXiv:2304.08485，[A]（key: llava）——§2 两段配方（caption 对齐 + 指令微调）出处
- [Liu et al., 2023](../references.md)：Improved Baselines with Visual Instruction Tuning，arXiv:2310.03744，[A]（key: llava-1.5）——§2.1/§2.2 的 558K/665K 数据口径
- [Alayrac et al., 2022](../references.md)：Flamingo: a Visual Language Model for Few-Shot Learning，arXiv:2204.14198，[B]（key: flamingo）——§3.3 交错图文预训练数据与 few-shot 形态
- [haotian-liu/LLaVA](../references.md)：[A]（key: llava-repo）——§1.2 labels 构造走读锚点
- [LLaVA-VL/LLaVA-NeXT](../references.md)：[A]（key: llava-next-repo）——LLaVA-OneVision 配方代码锚点
- [allenai/Molmo](../references.md)：[A]（key: molmo-repo）——PixMo 数据开放锚点
