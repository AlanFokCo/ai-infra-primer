# 3.3 Decoder-only 与 In-Context Learning（Decoder-Only and In-Context Learning）

> 本章回答两个问题：为什么生成式语言模型收敛到 decoder-only 形态，以及为什么把示例写进 prompt 就能在不更新参数的情况下改变模型行为。前一半把「自回归作为通用接口」拆成三条机制论证（任务空间、增量计算经济学、训练-推理一致性），后一半建立 in-context learning 的形式化与三层理论解释（隐式梯度下降 / induction heads / 任务识别），以 Pythia 的 154 checkpoints 实证训练相变收尾 | 前置：1.1（自回归分解、CE、teacher forcing、祖先采样）、1.3（emergence 的测量学分析）、2.4（causal mask、KV cache 推导与账、prefill / decode）、3.1（objective 与 mask 的统一框架）、3.2（encoder 形态生成边界的完整证明）| 锚点：Pythia（16 模型 × 154 checkpoints）[A]、TransformerLens [A]、GPT-2 [W]；GPT-3 [B]、von Oswald [B]、Akyürek [B]、Dai [B]、Min [B]、Olsson（induction heads，transformer-circuits.pub）[B]

## 1. 论证一（任务空间）：任何条件建模任务都可写成序列续写

一个判别式任务的数学内容是条件分布 $p(y \mid c)$ —— $c$ 含任务规范与输入， $y$ 是输出。把两者序列化成单条 token 序列 $z = c \oplus y$（ $\oplus$ 为拼接），链式法则（1.1 式 (3)）给出：

$$
p(y \mid c) = \prod_{t=1}^{T_y} p\left(z_{|c|+t} \mid z_{<|c|+t}\right) \qquad (1)
$$

**符号**：$c$ ：任务规范与输入的序列化（instruction、demonstration、query 输入拼成的总前缀）；$|c|$ ：前缀 token 数；$y = (z_{|c|+1}, \ldots, z_{|c|+T_y})$ ：输出段。右式每个因子都是 CLM 前向在对应位置上的 next-token 条件分布，不是什么新对象。

式 (1) 的读法：条件任务是 CLM 联合分布在一个位置区间上的切片。对照 1.1 的 $(x', M, \ell)$ 三元组（objective 侧在 3.1 展开），它对应「 $x' = z$ 原样、 $M$ 取 causal 下三角、 $\ell$ 只在 $y$ 区段为 1」——纯 CLM 的一个 loss mask 取值。因此 CLM 训练天然覆盖所有「前缀 → 后继」形态的条件任务：teacher forcing 对语料中每条序列的每个前缀都计 loss（1.1 §4），等价于对语料可识别的全部条件结构做混合 MLE。GPT-2 的报告把这个观察立为方法论（[Radford et al., 2019](../references.md)，[W]）：自然语料本身含有翻译对照、问答对、原文与摘要等海量隐式任务实例，足够通用的 LM 目标就是一种无监督的多任务学习——GPT-2 的 zero-shot 任务评测（不给示例、只在 prompt 里写任务格式）即据此设计。GPT-3 再推进一步，把任务规范本身也搬进前缀 $c$ ，得到第 4 节的 in-context learning（ICL，上下文学习）。

与 encoder 形态对照，任务接口的位置不同：

| | encoder + 任务头（3.2） | decoder-only |
| --- | --- | --- |
| 任务接口 | 参数：每任务一个 head， $p(y \mid h_{\mathrm{CLS}})$ | 数据：prompt 文本，式 (1) |
| 适配新任务 | 梯度回路（fine-tune，权重改变） | 写 prompt（零梯度，权重不变） |
| 条件结构 | head 参数化 | 序列拼接 |

「接口是数据而不是参数」是 ICL 存在的前提，它的账由后两条论证结，能力边界由第 4–8 节结。序列化本身不损失信息——链式法则对任意顺序成立——但有一个实现级约束：序列化必须把输入放在输出之前，否则式 (1) 的每个条件因子里混入未知量，自回归采样无从起步；模板 $\tau$ 在演示与 query 间保持一致，则是让模型把前缀识别为「同一任务的实例列」的最低结构要求（第 7 节会看到格式本身是演示信息构成的大头之一）。序列化不保证模型学会了利用这种结构，那是实证问题。

## 2. 论证二（增量计算经济学）：KV cache 只适配 causal

2.4 已把账算完，此处只收银。可生成的判定是可见集条件（2.4 式 (4)）：$V(i) \subseteq \{1, \ldots, i\}$ 对每个位置成立；causal mask 是满足该条件的最大可见集——每个位置看见全部过去，一个 token 都不浪费。prefix-LM 是它的折中（前缀块内双向、续写区因果，2.4 §2 的对照表），MLM 的全 0 mask 处处违反（不能自回归生成的完整证明，含「多轮迭代采样不能绕过」的论证，在 3.2）。

经济学含义分两层。其一，可缓存性与因果性是同一性质的两面（2.4 §4.2）：双向模型追加一个 token 会改变所有位置的 stream，下游全部重算，cache 无物可存；朴素逐 token 生成的总账是 $\approx NT^{2} + \frac{2}{3}LT^{3}d$（2.4 式 (5)），而 cache decode 是 $2NT + 2LT^{2}d$（2.4 式 (8)），attention 项差约 $T/3$ 倍，满窗 $T$ = 131072 时约 $4.4 \times 10^{4}$ 倍。生成式服务的可行性挂在这条不等式上。其二，cache 的内存形态（2.4 式 (9)：每 token 每层 $2 H_{kv} d_{\mathrm{head}} b$ 字节、随 $T$ 线性增长）使「长 prompt + 短输出」型任务的成本可预测、可定价——ICL 恰好是这个形状： $K$ 个 demonstration 的 prefill 一次付清（compute-bound 的大 matmul，2.4 §6），同一会话的后续 query 复用同一份 cache。

把三条论证连起来看，ICL 还是一次成本移位：任务适配的代价从「训练侧的一次性梯度回路」（fine-tune：额外数据、额外优化、产出新权重）搬到「推理侧的每请求 prefill」。两笔账的结构性差异（一次性训练成本对每请求前缀成本）在练习 2 展开；哪个划算取决于请求量与任务数量，但「存在推理侧选项」本身改变了适配的工程形态。

这条论证同时给出 ICL 的物理载体：参数不动时，前向过程中唯一的可写存储是 KV cache 与当前 token 的 residual stream。第 4 节形式化里「无参数更新」的全部信息量，就压在这两份随序列线性增长的激活里——第 5、6 节的两层机制解释，都是在这个载体上发生的。

## 3. 论证三（训练-推理一致性）：一个前向，两处结账

2.4 §7 的命题：causal mask 下，teacher forcing 并行前向第 $t$ 位的 logits 与串行 decode 第 $t$ 步的 logits 是同一数学对象。推论：训练与推理共用同一参数化、同一前向、同一条件分布族——训练 loss 的每一项都是一次推理操作的精确对应物。两处对照把这条性质的位置摆清。encoder 侧：MLM 的训练操作（双向上下文下填 mask 位）与任何可行的生成操作都不对应（3.2 给出证明），训练出来的条件分布族与生成需要的条件分布族不相交。fine-tune 范式侧：任务适配要开第二个梯度回路、产出新权重，部署的前向与被训练的前向之间隔着一次参数迁移。decoder-only + ICL 的适配在推理侧完成，「服务的那个前向就是被训练的那个前向」逐字成立。

一致性的边界也要写清：1.1 §4 登记的暴露偏差（exposure bias，训练时条件于真实前缀、推理时条件于模型自采前缀的分布失配）不在此处解决，归 training 维度；本节保证的是架构层的一致性，不是分布层。

三条论证合起来给出 decoder-only 的位置：式 (1) 保证任务空间不缺，2.4 的账保证服务经济可行，一致性保证训练信号不打折地落在部署形态上。剩下的是能力问题：模型在前向中「用演示做了什么」——这是 ICL 形式化与三层理论解释要回答的。

## 4. ICL 形式化：demonstration 作为条件，参数不动

GPT-3 的定义性设置（[Brown et al., 2020](../references.md)，arXiv:2005.14165，[B]）：冻结模型 $\theta$ ，把 $K$ 个演示样例与 query 经模板 $\tau$ 序列化：

$$
s_K = \tau(x_1, y_1) \oplus \tau(x_2, y_2) \oplus \cdots \oplus \tau(x_K, y_K) \oplus \tau(x_q) \qquad (2)
$$

**符号**：$(x_i, y_i)$ ：第 $i$ 个 demonstration 的输入-标签对；$\tau$ ：把一对样本渲染成文本的模板（含字段名与分隔符）；$x_q$ ：查询输入；$s_K$ ：完整 prompt。预测 $\hat{y} \sim p_\theta(\cdot \mid s_K)$（argmax 或采样，1.1 §7-8 的口径）。 $K = 0$ 为 zero-shot（ $\tau(x_q)$ 前只有任务描述）、 $K = 1$ 为 one-shot、 $K \ge 2$ 为 few-shot。

```
s_K 的布局（K = 2，情感分类例）：
┌────────────────────────┬────────────────────────┬──────────────────────┐
│ τ(x_1, y_1)            │ τ(x_2, y_2)            │ τ(x_q)               │
│ "输入: 节奏拖沓 标签: 负" │ "输入: 摄影极佳 标签: 正" │ "输入: 值回票价 标签:" │
└────────────────────────┴────────────────────────┴──────────┬───────────┘
        prefill 一次付清（cache 常驻会话）                       模型续写位
```

三条性质把这个对象与微调严格分开：

1. **无梯度**。整个过程中不存在 loss、不存在 $\nabla_\theta$ ， $\theta$ 逐 bit 不变。「学习」若发生，只能发生在前向的激活里——由第 2 节，具体地址是 $s_K$ 的 KV cache（ $2 L H_{kv} d_{\mathrm{head}} |s_K| b$ 字节，2.4 式 (9) 代入 $T = |s_K|$ 、 $B = 1$ ）与 decode 位置的 residual stream。
2. **信息通道只有 token**。模型关于任务的全部可用信息就是 $s_K$ 的 token 序列；没有独立的任务 embedding 输入端口，没有旁路。
3. **经验规律**（GPT-3 报告）：多数任务上 few-shot 分数高于 one-shot、one-shot 高于 zero-shot，且差距随模型规模扩大——在 1.3 §7 的语境里，这正是「prompting 策略型」涌现能力报告的原始形态，其 metric 形状问题同样适用（第 8 节回到这一点）。

平凡化警告：next-token CE（1.1 式 (4)）没有任何一项直接优化「从演示学映射」——训练目标里不存在演示结构， $s_K$ 只是恰好长这样的前缀。因此「为什么有效」需要机制层回答。以下三层解释各给一个侧面：(a)(b) 在机制层（权重能装下什么算法、真实电路长什么样），(c) 在功能层（演示的信息构成是什么）。

## 5. 解释层 (a)：linear attention ≈ 对演示做一步梯度下降

取演示任务为线性回归：$y_i = W x_i + \varepsilon$ ，演示集 $\{(x_i, y_i)\}_{i=1}^{K}$ ，query 为 $x_q$ 。若「模型的前向」等价于「先对演示集训练一个线性模型、再用它预测」，一步梯度下降是最小候选算法。从 $W_0 = 0$ 出发，平方损失 $\mathcal{L}(W) = \frac{1}{2} \sum_i \lVert W x_i - y_i \rVert^{2}$ 的一步 GD（ $\nabla_W \mathcal{L} = \sum_i (W x_i - y_i) x_i^{\top}$ ）：

$$
W_1 = W_0 - \eta \sum_{i=1}^{K} \left( W_0 x_i - y_i \right) x_i^{\top} = \eta \sum_{i=1}^{K} y_i x_i^{\top}, \qquad \hat{y} = W_1 x_q = \eta \sum_{i=1}^{K} y_i \left( x_i^{\top} x_q \right) \qquad (3)
$$

**符号**：$\eta$ ：学习率；$x_i \in \mathbb{R}^{d_x}$ 、 $y_i \in \mathbb{R}^{d_y}$ 。

von Oswald et al.（[2022](../references.md)，arXiv:2212.07677，[B]）的构造性结果（定理 sketch）：去掉 softmax 的 linear attention 层 $o_t = \sum_s v_s \langle k_s, q_t \rangle$ 可以逐项实现式 (3)。构造：token 嵌入取堆叠向量 $z_i = (x_i;\, y_i) \in \mathbb{R}^{d_x + d_y}$（query 位置 $z_q = (x_q;\, 0)$ ），三个投影取分块选择矩阵 $W_K z_i = x_i$ 、 $W_Q z_q = x_q$ 、 $W_V z_i = \eta\, y_i$ ，代入 query 行：

$$
o_q = \sum_{i=1}^{K} \left( \eta\, y_i \right) \left\langle x_i, x_q \right\rangle = \eta \sum_{i=1}^{K} y_i \left( x_i^{\top} x_q \right) = W_1 x_q \qquad (4)
$$

**符号**：$o_q$ ：query 位置的 attention 输出；$\langle x_i, x_q \rangle = x_i^{\top} x_q$ ；其余同式 (3)。

与式 (3) 逐项相等。demo 行的输出在构造里是 don't-care（预测只读 query 行）。 $W_0 \neq 0$ 的一般情形把 value 换成残差 $\eta \left( y_i - W_0 x_i \right)$ 并保留一条 $W_0 x_q$ 直连项即可——von Oswald 的完整构造用分块矩阵把残差项也装进嵌入，多层堆叠对应多步 GD。

适用边界必须钉死，否则这层解释会被过度引用：等价只对**线性核**（无 softmax）精确成立——softmax 归一化给每个演示项的权重引入全序列耦合，逐项相等断裂（断裂位置与断裂项的形式是练习 1）；权重是**构造**出来的，定理不保证训练必然找到它；演示任务被设为线性回归，非线性任务对应的是逐层 GD 或其他算法的推广命题。

互补方向是识别而非构造。Akyürek et al.（arXiv:2211.15661，[B]）训练真实 transformer 做 in-context 线性回归，然后把前向输出与候选算法逐点对照——闭式 ridge（岭回归）解

$$
\widehat{W} = \left( X^{\top} X + \lambda I \right)^{-1} X^{\top} y \qquad (5)
$$

**符号**：$X$ ：演示输入堆成的矩阵；$y$ ：演示标签向量；$\lambda$ ：ridge 正则系数。

以及不同步数、不同学习率的显式 GD，并用探针从各层隐状态恢复中间估计量。结果：训练后的 transformer 的预测与这些标准估计量高度贴合，隐状态中含有逐层精炼的估计——前向过程的行为与一个显式优化算法的运行轨迹对得上。Dai et al.（arXiv:2212.10559，[B]）并列引用：meta-optimizer 视角把预训练 SGD 视为外循环（更新 $\theta$ ）、ICL 视为内循环（attention 对演示的聚合充当对「工作记忆」的元更新），并给出 ICL 与显式 fine-tune 的行为对照实验。三篇的证据强度分级：von Oswald 是构造性定理（在假设内精确成立），Akyürek 是统计识别（在测量精度内成立），Dai 是类比加行为实验——同向但不同级，引用时不可混为一谈。

## 6. 解释层 (b)：induction heads 的两层回路

任务原型：序列 $[\mathrm{A}][\mathrm{B}] \cdots [\mathrm{A}]$ 之后该接 $[\mathrm{B}]$ ——「上次 A 后面跟的是 B」。Olsson et al.（[2022](../references.md)，transformer-circuits.pub，[B]）在两层 attention-only 玩具模型里给出这个行为的最小电路实现，分工是两个 head：

```
序列:      …  [A]_i  [B]_{i+1}  ………  [A]_t  (下一位置)
                 ▲                     │
第 1 层          │ previous-token head │
   x^(1)_{i+1} 含「我的前驱是 A」 ◄───────┘  A^{(1)}_{j+1, j} ≈ 1：拷贝前驱表征
                 │
第 2 层          │ induction head（在位置 t 工作）
   query(t)：「我是 A」 ──prefix match──► key(i+1)：「我的前驱是 A」 命中
   value(i+1) = B 的表征 ──copy──► 写入位置 t 的 stream ──► W_U 抬升 B 的 logit
```

第 1 层 **previous-token head**（前驱 token 头）：attention pattern 学成前一位偏移，把前驱位置的 token 表征拷进当前位置的 stream：

$$
x_{j+1}^{(1)} = x_{j+1}^{(0)} + \sum_{s} A^{(1)}_{j+1,\, s}\, W_{OV}^{(1)} E_{x_s} \approx x_{j+1}^{(0)} + W_{OV}^{(1)} E_{x_j} \qquad (6)
$$

**符号**：$E_{x_j} \in \mathbb{R}^{d}$ ：token $x_j$ 的嵌入（residual stream 初值）；$W_{OV}^{(1)}$ ：第 1 层的 OV 矩阵；近似号用 $A^{(1)}_{j+1,\, j} \approx 1$ 。效果：位置 $i+1$ 的 stream 同时编码「我是 B」与「我的前驱是 A」。

第 2 层 **induction head**（归纳头）：当前位置 $t$（第二个 A）的 query 去匹配「前驱是 A 的位置」的 key——这一步叫 prefix matching（前缀匹配）；命中位置 $i+1$ 后，OV 侧把该位置的 token 表征（B）搬到 $t$ ——这一步叫 copying（复制），经 unembedding $W_U$ 抬升 B 的 logit。把式 (6) 代入第 2 层 score，展开组合双线性型：

$$
s^{(2)}_{t,\, i+1} = \frac{1}{\sqrt{d}} \left\langle W_Q^{(2)} x_t^{(1)},\, W_K^{(2)} x_{i+1}^{(1)} \right\rangle = \frac{1}{\sqrt{d}} \left( E_{x_t} \right)^{\top} \underbrace{\left( W_Q^{(2)} \right)^{\top} W_K^{(2)} W_{OV}^{(1)}}_{\text{token-match}} E_{x_i} + \cdots \qquad (7)
$$

**符号**：$\cdots$ 为不经过第 1 层路径的展开项（直连嵌入项与各位置的自作用项），不携带 $x_i$ 与 $x_t$ 的匹配信号。当 $x_i = x_t$（都是 A）时该双线性型取大值——prefix matching 的代数形态就是一个三层矩阵乘积构成的「token 相等」检测器，跨两个 head 组合而成，单看任何一层都不存在。

这套机制不是叙事，是可测量的。测量协议（TransformerLens，[Nanda & Bloom](../references.md)，[A]）：构造随机 token 序列重复两遍 $[r_1, \ldots, r_n][r_1, \ldots, r_n]$ ，第二遍中位置 $n + j$（token $r_j$ 再次出现）的正确 induction 目标是第一遍的 $j+1$ 位；**induction score** 定义为 head 的 attention pattern 落在这些目标位置上的平均质量（pattern 在一条偏移对角线上的 mass）。prefix-matching score 与 copying score 分别单独测两侧：QK 双线性型（式 (7)）对 token 相等的响应强度，与 OV 输出经 $W_U$ 后对相应 token logit 的抬升方向。GPT-2 small（[Radford et al., 2019](../references.md)，[W]）是这类测量的标准对象之一——4.7 的 IOI 完整电路分析在同一模型上进行。

与 (a) 的分工：(a) 是容量命题（权重层面装得下什么算法），(b) 是机制命题（训练出的真实电路长什么样）。前者说「可以装下 GD」，后者说「实际成形的是这个两层回路」。

## 7. 解释层 (c)：任务识别视角

Min et al.（arXiv:2202.12837，[B]）的 ablation 直接拆演示的信息构成。核心操作：把 demonstration 的 label 替换为从 label 空间随机采样的 label——input 分布、label 空间、模板格式全部保留，唯一被破坏的是 input↔label 映射。结果（论文在 12 个模型、分类与多选任务族上报告）：性能只小幅下降，且远好于无演示基线。逐项剥离的结论：

| 被破坏的成分 | 性能影响 |
| --- | --- |
| input↔label 映射（random label） | 小幅下降 |
| input 分布（换成分布外句子，映射保持） | 明显下降 |
| 模板格式 / 演示整体移除 | 下降最多 |

含义：在这些任务上，演示的主要贡献是（i）label 空间（合法答案的集合）、（ii）input 分布（领域与文本样式）、（iii）格式——三者帮助模型从预训练学到的混合任务分布中识别当前任务（task recognition），而不是把一条新映射学进前向。

三层解释的适用域由此划开，互不冲突：(c) 的论断限于「映射已被预训练覆盖」的任务族（常见 NLU 分类 / 多选）——演示里的映射本身是新的时候（合成映射、打乱标签的重映射任务），random label 会致命中，(a) 的 GD 侧刻画接管；(b) 刻画复制类结构的电路，与映射学习正交。把任何一层当成 ICL 的完整理论都是过度外推。

## 8. Pythia 实证：训练相变

Olsson et al. 的第二组结果是动力学而非结构：沿训练步数看，few-shot ICL 分数的提升集中在一个窄窗口内（相变状），且该窗口与 induction head 的 induction score 起跳窗口重合；窗口位置在不同规模模型之间按已见 token 数刻度大致对齐。这把「ICL 从哪来」从功能描述推进到一个机制事件：某个电路在训练中成形，功能随之上线——解释层 (b) 由此接上训练轴。

在公共底座上重做这类分析的标准仪器是 Pythia（[Biderman et al., 2023](../references.md)，arXiv:2304.01373，[A]）：16 个模型（70M–12B）、每模型 154 个公开 checkpoints、数据与数据顺序完全一致（1.3 §7 已把它作为 emergence 测量学的对照底座引入）。154 个训练轴采样点正是分辨「平滑提升」与「相变跳变」所需的密度。两轴对比值得写清：1.3 §7 说明规模轴上报告的涌现大多是 metric artifact；训练轴相变的证据结构更强——分数跳变有 induction score 的同步起跳作机制侧旁证，不只是 metric 形状。两条轴的判别标准不同，结论不可互相挪用。

## 9. 诚实标注：ICL 理论是开放领域

- (a) 的逐项等价只在无 softmax、构造初始化、线性演示任务下精确成立；向 softmax / 深层 / 非线性任务的推广靠识别实验与类比（Akyürek、Dai），证据强度逐级递减。
- (b) 解释了复制类 ICL 的电路；语义映射类任务是否由 induction-head 族电路承载，没有同等级的证据；「induction head 是全部 ICL 的必要条件」这一强度的因果断言只在小模型消融上得到过支持。
- (c) 的「演示不教映射」结论随任务族翻转（第 7 节末）。三层解释是互补刻画，不是互斥终论。
- 测量学警告贯穿全场：few-shot 曲线的跳变仍部分受 metric 形状支配（1.3 §7）；ICL 的能力结论应先过 metric 审查，再过机制审查。

三层共同锁死的只有一件事（第 2 节）：参数不动时，信息只能经 token 序列进入、只能住在激活里。任何合格的 ICL 理论都必须把「学习」完整压进 KV cache 与 residual stream——这是后续章节继续推进的共同地基（4.7 的电路工具箱、training 维度的涌现讨论）。

## 总结

| 内容 | 公式 / 出处 | 要点 |
| --- | --- | --- |
| 任务空间论证 | 式 (1)：$p(y \mid c) = \prod_t p(z_{|c|+t} \mid z_{<|c|+t})$ | 条件任务 = CLM 联合分布的切片；任务接口是数据不是参数 |
| 增量计算经济学 | 2.4 式 (4)(5)(8)(9) | 可缓存 ⟺ causal；ICL 的物理载体 = KV cache + residual stream |
| 训练-推理一致性 | 2.4 §7 | 并行前向 ≡ 串行 decode 同一数学对象；exposure bias 归 training |
| ICL 形式化 | 式 (2)：$s_K = \tau(x_1, y_1) \oplus \cdots \oplus \tau(x_q)$ | 无梯度、信息通道只有 token；zero / one / few-shot |
| 隐式 GD | 式 (3)(4)；ridge 式 (5) | von Oswald 构造（精确、线性核）/ Akyürek 识别 / Dai 类比——三级证据 |
| induction heads | 式 (6) previous-token；式 (7) 组合双线性型 | prefix matching + copying；induction score 可测（随机重复序列协议） |
| 任务识别 | Min et al. ablation 表 | 映射 ≠ 演示信息的大头；结论随任务新颖度翻转 |
| 训练相变 | Pythia 154 checkpoints [A] | few-shot 分数与 induction score 同步起跳；训练轴证据强于规模轴 |
| 开放领域 | 第 9 节 | 三层互补不互斥；共同约束 = 信息只住在激活里 |

## 练习

1.（推导）补全第 5 节 $W_0 \neq 0$ 的构造：给出嵌入 $z_i$ 与 $W_K, W_Q, W_V$ 的分块矩阵形式，使一层 linear attention 的 query 行输出 $\left( W_0 - \eta \nabla_W \mathcal{L} \right) x_q$ 。再把 $o_t = \sum_s v_s \langle k_s, q_t \rangle$ 换回 softmax attention，指出式 (4) 的逐项相等在哪一步断裂、断裂项的形式是什么。
2.（计算）ICL 的服务账：Llama-3.1-8B 口径（ $L = 32$ 、 $H_{kv} = 8$ 、 $d_{\mathrm{head}} = 128$ 、bf16），每条 demonstration 64 token、query 32 token， $K \in \{0, 1, 4, 16, 64\}$ ：(a) 用 2.4 式 (9) 算各 $K$ 下 prompt 的 KV cache 字节数与每多一条演示的边际字节；(b) 用 2.4 式 (8) 估算各 $K$ 的 prefill FLOPs；(c) 对照「把任务 fine-tune 进权重」的方案，指出两笔账的结构性差异（一次性训练成本对每请求 prefill 成本），说明各适用于什么请求量与任务数量的组合。
3.（代码）用 TransformerLens 加载 GPT-2 small：取 50 个随机 token 重复两遍组成序列，对每个 head 计算 induction score（attention pattern 在「偏移重复起点 +1」目标位上的平均 mass），列出得分最高的 5 个 head；对其中最强者再测 prefix-matching score 与 copying score，验证式 (6)(7) 预测的 QK / OV 分工。（教学练习，测量口径以 TransformerLens 文档为准。）
4.（分析）若 ablate 第 1 层的 previous-token head（ $W_{OV}^{(1)} \to 0$ ），式 (7) 的匹配项剩余什么？induction score 预期如何变化？设计一个检验把「prefix matching 依赖第 1 层」与「第 2 层独立形成匹配能力」分开：可用的仪器是 Pythia 某模型相变窗口前后的相邻 checkpoints，或 TransformerLens 的 ablation 接口——写明你观测哪个量、两种假设下它各应呈现什么形态。

## 参考

- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）——ICL 定义、zero / one / few-shot 设置与规模规律
- [Radford et al., 2019](../references.md)：Language Models are Unsupervised Multitask Learners，[W]（key: gpt2）——LM 目标即隐式多任务的论点；GPT-2 small 为第 6 节测量对象
- [von Oswald et al., 2022](../references.md)：Transformers Learn In-Context by Gradient Descent，arXiv:2212.07677，[B]（key: von-oswald）——linear attention 实现一步 GD 的构造性定理
- [Akyürek et al., 2022](../references.md)：What Learning Algorithm Is In-Context Learning? Investigations with Linear Models，arXiv:2211.15661，[B]（key: akyurek）——学习算法识别与隐状态探针
- [Dai et al., 2022](../references.md)：Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers，arXiv:2212.10559，[B]（key: dai-icl）——meta-optimizer 视角与 fine-tune 行为对照
- [Min et al., 2022](../references.md)：Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?，arXiv:2202.12837，[B]（key: min-icl）——random-label 等 ablation 与演示信息构成
- [Olsson et al., 2022](../references.md)：In-context Learning and Induction Heads（transformer-circuits.pub，无 arXiv），[B]（key: induction-heads）——两层回路、induction score、训练相变
- [Biderman et al., 2023](../references.md)：Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling，arXiv:2304.01373，[A]（key: pythia）——16 模型 × 154 checkpoints 的训练轴底座
- [Nanda & Bloom](../references.md)：TransformerLens，[A]（key: transformerlens）——induction / prefix-matching / copying score 的测量工具
