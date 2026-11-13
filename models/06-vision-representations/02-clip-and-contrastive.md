# 6.2 CLIP 与对比学习（CLIP and Contrastive Learning）

> 本章解决「图与文两种模态如何共用一个 embedding 空间」这一个问题：InfoNCE 双向交叉熵的完整推导（batch $B$ 对 → 相似度矩阵 $S = VT^{\top}/\tau$ → 行/列两个 $B$ 类分类）、温度 $\tau$ 的梯度机制与可学参数化（ $\tau = e^{-t}$、init 0.07、clamp 上限 100）、batch size 的 Monte Carlo 解读与 uniformity 渐近分解、zero-shot 分类 = 检索的公式与 prompt ensembling、alignment / uniformity 几何诊断指标；CLIP embedding 空间的两个下游角色（P8 的 VLM 视觉塔、P7 的 T2I 文本编码器）登记为伏笔 | 前置：1.1（CE/NLL 式 (4)(5)(6)、采样温度式 (9)）、6.1（式 (12)(13)(15)：logit 约定、InfoNCE 行方向登记、softmax-CE 梯度、SigLIP 对照）、2.3（内积 score 惯例） | 锚点：CLIP [A]、Wang & Isola（alignment/uniformity）[A]、openai/CLIP [A]、mlfoundations/open_clip [A]

## 1. 问题定义：把两种模态钉进同一个单位球面

分类范式把视觉表征钉死在固定类别表上：ImageNet 的 1000 类是一个封闭集合，分类头就是一张 $K \times d$ 的冻结版输出头（1.1 §2 的 unembedding 结构），训好之后对表外概念没有任何表达——加一个类别就要动输出头并重训。CLIP（[Radford et al., 2021](../references.md)，arXiv:2103.00020，[A]）换了问题提法：不学「图 → 类别 id」的映射，学一个**共享表征空间**，让图和文在这个空间里的几何关系（余弦相似度）直接编码「配不配」。类别表从权重里拿出来，变成推理期用文本即时生成的可检索库（§5）——封闭集合随之打开。这个提法同时统一了一批任务：分类、以文搜图、以图搜文、图搜图，全是同一空间里的几何查询，不再各自需要一颗任务头。

监督来源的转换是这一提法的另一半。分类标注是逐样本的人工动作，类别表多长、标注成本就随类数线性长；网页图文对的监督来自共现本身——alt text、图说、上下文段落——获取成本是爬取与清洗，不是逐条标注，规模上限由网页存量决定。代价是标签噪声：共现不等于语义匹配（一张产品图旁边的文字可能是配送说明），这个噪声在 §2.2 的 CE 地板与 §3 的温度自校准里都会回来。

形式化。数据是 $B$ 个图文对 $\lbrace(x_i, y_i)\rbrace_{i=1}^{B}$，采样自网页共现分布 $p_{\mathrm{data}}$（CLIP 的规模：400M 对，数据工程见 §7）。两个 encoder：视觉塔 $f$（6.1 的 ViT 系），文本塔 $g$（一个带 causal mask 的 Transformer，取 [EOS] 位置的激活做读出——2.4 的 mask 约定，读出槽位逻辑同 6.1 §2 的 [CLS]）。两塔输出各过一个投影矩阵后单位化（6.1 §7 的全书约定）：

$$
v_i = \frac{\tilde v_i}{\lVert \tilde v_i \rVert_2}, \qquad t_j = \frac{\tilde t_j}{\lVert \tilde t_j \rVert_2}, \qquad \tilde v_i = W_v\thinspace f(x_i), \quad \tilde t_j = W_t\thinspace g(y_j) \qquad (1)
$$

**符号**： $v_i, t_j \in \mathbb{R}^{d}$：图、文的单位化 embedding； $W_v, W_t$：模态各自的投影矩阵，把两塔各自的输出宽度映到共享维 $d$（CLIP ViT-L/14 档 $d$ = 768，[openai/CLIP repo](../references.md)，[A]）。投影层存在的结构原因：两塔宽度本来不同， $W_v, W_t$ 把「塔的宽度」与「共享空间的维数」两个设计变量解耦，塔可以按各自模态的账选型。单位化把全部几何压到单位球面 $S^{d-1}$ 上： $\langle v_i, t_j \rangle = \cos\theta_{ij} \in [-1, 1]$，相似度就是球面夹角，模长信息被显式丢弃——「多长」不编码语义，「朝哪」才编码。

训练目标要同时做到两件事：正样本对（positive pair） $(v_i, t_i)$ 拉近（夹角趋于 0），不配对的 $(v_i, t_{j \neq i})$ 推开。难点全在「推开」一侧——拿哪些负样本（negative samples）、按什么权重推、推多远。InfoNCE 用一张 batch 相似度矩阵加两个 softmax 一次性回答了这三个问题，下面逐步推。

## 2. InfoNCE：batch 相似度矩阵上的两个交叉熵（完整推导）

### 2.1 相似度矩阵：batch 结构即负样本供给

batch 内 $B$ 对样本给出 $B$ 个图 embedding 与 $B$ 个文 embedding，堆成 $V = [v_1; \ldots; v_B] \in \mathbb{R}^{B \times d}$、 $T = [t_1; \ldots; t_B] \in \mathbb{R}^{B \times d}$。全部两两相似度一次矩阵乘算完，除以温度 $\tau$ 得 logit 矩阵：

$$
S = \frac{1}{\tau}\thinspace V T^{\top} \in \mathbb{R}^{B \times B}, \qquad S_{ij} = \frac{\langle v_i, t_j \rangle}{\tau} \qquad (2)
$$

**符号**： $S$：相似度 logit 矩阵——即 6.1 式 (12) 的 CLIP 版（SigLIP 记号的 $\gamma = 1/\tau$、 $b = 0$）； $\tau > 0$：温度，§3 专拆。

$S$ 的对角线 $S_{ii}$ 是正样本对的 score，同行、同列的其余 $B - 1$ 个元素天然充当负样本——不需要额外的负样本采样管线，**batch 结构本身就是负样本供给**。这就是对比学习（contrastive learning）视角：每个样本的学习信号来自「与正确配对比、与 batch 内其余 $B - 1$ 个比」的对比。一次训练步的数据流：

```
图像 x_1..x_B ─► 视觉塔 f ─► W_v ─► 单位化 ─► V [B, d] ─┐
                                                          ├─► S = V Tᵀ / τ   [B, B]
文本 y_1..y_B ─► 文本塔 g ─► W_t ─► 单位化 ─► T [B, d] ─┘         │
       S 的对角 S_ii = 正样本对（分类标签）；                          ▼
       同行/列其余 B−1 个 = 负样本            ┌ 行 softmax + CE（image→text，标签 = i）
                                             └ 列 softmax + CE（text→image，标签 = j）
                                                              ▼
                                              L = ½ ( L_i2t + L_t2i )
```

### 2.2 行方向与列方向：两个 $B$ 类分类问题

把 $S$ 的第 $i$ 行读成一个分类问题：「给定图 $x_i$，batch 里哪条文本是它的描述？」—— $B$ 个候选、正确标签是 $i$。softmax + CE 是 1.1 已建好的工具（式 (2)(5)，NLL 与 CE 的同一性见 1.1 §4），逐行套用：

$$
L_{i2t} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{B} \exp(S_{ij})} = \frac{1}{B} \sum_{i=1}^{B} \mathrm{CE}\left( \delta_i,\hspace{5mu} \mathrm{softmax}(S_{i\cdot}) \right) \qquad (3)
$$

**符号**： $S_{i\cdot}$： $S$ 的第 $i$ 行； $\delta_i$：标签 $i$ 的 one-hot 分布。式 (3) 即 6.1 式 (13) 登记的 InfoNCE 行方向。列方向对称——「给定文本 $y_j$，哪张图是它的配图」：

$$
L_{t2i} = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{\exp(S_{jj})}{\sum_{i=1}^{B} \exp(S_{ij})} \qquad (4)
$$

总损失取两个方向的平均：

$$
L = \frac{1}{2} \left( L_{i2t} + L_{t2i} \right) \qquad (5)
$$

为什么要两个方向：行方向只保证「每张图学会认自己的文本」，列方向覆盖「每条文本学会认自己的图」——丢掉式 (4) 时，文本侧的点从不作为 anchor 被拉近正样本，检索的两个方向（以文搜图、以图搜文）会一边倒（练习 1 从梯度上看这个不对称）。两个方向共享同一个 $S$，算力上是一次 $B \times B$ 矩阵乘喂两个方向的 softmax，没有第二份相似度账。

用 1.1 式 (6) 的分解读每一行： $\mathrm{CE} = H(p) + \mathrm{KL}(p \Vert q_\theta)$。真实匹配是确定的一对一时地板 $H(p) = 0$；但网页图文对的共现是弱标签（一张图可以配多条合理文本），数据本身带正熵，loss 降不到 0 的那部分就是数据噪声的度量——这把 §7 的数据质量问题直接接进了 loss 的数值解读。另一个基准数是随机基线：乱猜时行 CE = $\log B$； $B$ = 32768（CLIP 训练 batch，论文 §3）时基线为 $\log 32768 \approx 10.40$ nats，loss 的每一分下降都是真实对齐信号（练习 2）。

与 1.1 的语言建模 CE 做一次结构对照，能看清对比学习的独特处。LM 的分类问题是「类别 = 固定词表 $V$、类别向量 = 输出头 $W$ 的行」，词表与 unembedding 在训练全程固定可查；InfoNCE 的分类问题是「类别 = batch 内 $B$ 个候选、类别向量 = 另一个塔此刻的输出」——**标签空间本身随参数一起漂移**。这带来两个直接后果：(a) 类别含义跨步不一致， $t_j$ 每步都在动，模型追的是一个移动靶，这正是表征塌缩（全部向量挤到一起，式 (6) 梯度趋零也能压低 loss）必须通过 uniformity 项（§4/§6）从结构上防住的原因；(b) 候选集合随 batch 换，同一锚点在不同 batch 里面对不同负样本，学习目标自带课程式的随机性。另外，in-batch 负样本有个固有的噪声项：batch 里可能混进语义等价的对（两张猫图各配猫图说，互为「负样本」却被强行推开）—— $B$ 越大假负样本越多，但 §4 的估计收益增长更快，工程上天平倒向大 batch。

一个 $B$ = 3 的算例把式 (2)-(5) 走一遍。设三对样本的余弦相似度矩阵 $C$（ $c_{ij} = \langle v_i, t_j \rangle$）为

$$
C = \begin{bmatrix} 0.3 & 0.1 & -0.2 \\ 0.0 & 0.4 & 0.2 \\ -0.1 & 0.1 & 0.5 \end{bmatrix}, \qquad S = \gamma\thinspace C \approx \begin{bmatrix} 4.29 & 1.43 & -2.86 \\ 0 & 5.71 & 2.86 \\ -1.43 & 1.43 & 7.14 \end{bmatrix} \qquad (\tau = 0.07,\hspace{5mu} \gamma \approx 14.3)
$$

逐行 softmax（ $\exp$ 后归一化）：第一行 $\exp$ 值为 $(72.7,\ 4.17,\ 0.057)$， $p_{11} = 72.7 / 77.0 \approx 0.945$，逐样本 loss $L_1 \approx 0.057$；第二行 $(1,\ 303,\ 17.4)$， $p_{22} \approx 0.943$， $L_2 \approx 0.059$；第三行 $(0.24,\ 4.17,\ 1266)$， $p_{33} \approx 0.997$， $L_3 \approx 0.004$。行方向 $L_{i2t} \approx 0.040$，对照随机基线 $\log 3 \approx 1.099$。列方向同法（对每列归一化）得 $L_{t2i} \approx 0.019$，总损失 $L \approx 0.029$。两个观察：其一， $\gamma$ 把 $[-1, 1]$ 的余弦差放大 14 倍后，正样本只需 0.2-0.3 的余弦优势就能拿到 0.94 以上的行内概率——温度的放大作用是 loss 能快速低于基线的原因；其二，行 loss 与列 loss 不相等（0.040 vs 0.019），因为行 softmax 与列 softmax 是同一个 $S$ 上两种不同方向的归一化——双向平均不是冗余，是两个检索方向的各自优化。

### 2.3 梯度：吸引项与排斥项的显式分解

softmax + CE 的梯度是 1.1 §4 登记的「预测减真值」。对式 (3) 逐元素求导（即 6.1 式 (15)）：

$$
\frac{\partial L_{i2t}}{\partial S_{ij}} = \frac{1}{B} \left( p_{ij} - \delta_{ij} \right), \qquad p_{ij} = \frac{\exp(S_{ij})}{\sum_{j'=1}^{B} \exp(S_{ij'})} \qquad (6)
$$

**符号**： $p_{ij}$：第 $i$ 行 softmax 后、文本 $j$ 被选中为配图的概率； $\delta_{ij}$：Kronecker delta。列方向同形，记 $`p^{c}_{ij} = \exp(S_{ij}) / \sum_{i'} \exp(S_{i'j})`$（对行下标归一化）；合并式 (5) 的 $1/2$：

$$
\frac{\partial L}{\partial S_{ij}} = \frac{1}{2B} \left( p_{ij} + p^{c}_{ij} - 2\thinspace\delta_{ij} \right) \qquad (7)
$$

更有信息量的是对 embedding 本身的梯度。链式过 $S_{ij} = \langle v_i, t_j \rangle / \tau$（ $\partial S_{ij} / \partial v_i = t_j / \tau$），行方向：

$$
\frac{\partial L_{i2t}}{\partial v_i} = \sum_{j=1}^{B} \frac{\partial L_{i2t}}{\partial S_{ij}} \cdot \frac{t_j}{\tau} = \frac{1}{B\tau} \sum_{j=1}^{B} \left( p_{ij} - \delta_{ij} \right) t_j = \frac{1}{B\tau} \sum_{j \neq i} p_{ij} \left( t_j - t_i \right) \qquad (8)
$$

最后一步用 $\sum_j p_{ij} = 1$ 把 $-t_i$ 拆进求和（练习 1 补全）。取负号（梯度下降方向）读机制： $v_i$ 被**拉向**正样本 $t_i$（合力系数 $1 - p_{ii}$），同时被**推开**每个负样本 $t_j$、推力权重恰是 softmax 概率 $p_{ij}$。「对比」二字落在这个式子里：softmax 自动给「当前最容易混淆的负样本」（ $S_{ij}$ 最大的那些）分配最大的排斥力——hard negative mining 不是额外模块，而是 softmax 归一化的导数属性。 $p_{ii} \to 1$ 时整行梯度归零：已学会的 anchor 自动让出学习信号。

数值实现不照公式裸写：softmax 走 log-sum-exp（减行最大值，1.1 §2 的数值技巧），两个方向各一次 `cross_entropy` 调用。教学实现（与 open_clip 官方 `src/open_clip/loss.py` 的 `ClipLoss.forward` 对应，[open_clip](../references.md)，[A]）：

```python
import torch
import torch.nn.functional as F

# 教学用，非优化实现。对应 open_clip/src/open_clip/loss.py 的 ClipLoss。
def clip_infonce(v, t, log_logit_scale):
    # v, t: [B, d]，已单位化（式 (1)）；log_logit_scale 即 t = log γ，γ = 1/τ
    logit_scale = log_logit_scale.exp().clamp(max=100.0)   # 论文 §3 的 clamp
    S = logit_scale * (v @ t.T)                            # [B, B]，式 (2)
    labels = torch.arange(v.shape[0], device=v.device)     # 行/列标签都是对角
    loss_i2t = F.cross_entropy(S, labels)                  # 式 (3)，内部 log-sum-exp
    loss_t2i = F.cross_entropy(S.T, labels)                # 式 (4)
    return (loss_i2t + loss_t2i) / 2                       # 式 (5)
```

与官方实现的差异点只有一个：open_clip 先对两塔 embedding 做跨卡 all-gather 拼出全局 batch 再算 $S$；loss 形状与上面完全一致。all-gather 的必要性正是 6.1 §7 讲的跨样本归一化耦合——softmax 分母要看到全 batch，这份通信成本就是后来 SigLIP 改 sigmoid 的动机（对照见 6.1 §7，本章不重复推导）。

顺手记 loss 本体的算力账： $S$ 的矩阵乘是 $2B^{2}d$ FLOPs， $B$ = 32768、 $d$ = 768 时约 $1.6 \times 10^{12}$ FLOPs（1.6 TFLOPs）——相对双塔对 32768 张图与 32768 条文本的前向（各自以 PFLOPs 计）是零头。CLIP 训练的成本大头永远在两塔前向与 all-gather 通信，不在对比 loss 本身；这与 6.1 §7 对 SigLIP「省的是同步不是 FLOPs」的判词互相印证。

## 3. 温度 $\tau$：锐度、梯度尺度与可学参数化

式 (2) 里 $\tau$ 只做一件事：把 $\cos\theta \in [-1, 1]$ 的有界相似度缩放成 logit。但 softmax 对尺度极敏感，这个标量是训练稳定的第一变量。先把两个极限钉死（形式与 1.1 式 (9) 的采样温度相同，但角色不同：这里的 $\tau$ 在 loss 内部、是可学参数，不是推理期的分布整形旋钮）：

$$
\tau \to 0^{+}:\quad p_{i\cdot} \to \text{one-hot}\left( \mathrm{arg\thinspace max}_j\hspace{5mu} S_{ij} \right); \qquad \tau \to \infty:\quad p_{ij} \to \frac{1}{B} \qquad (9)
$$

代回式 (8) 读两端。 $\tau \to \infty$： $p_{ij} \approx 1/B$ 均匀分摊，推力摊平到全部负样本，且式 (8) 前还乘 $1/\tau$——梯度幅度趋于 0，目标被均匀化压死，学不动。 $\tau \to 0^{+}$：概率质量全压到当前最难的负样本上（除非正样本已是 argmax），排斥信号极度尖锐再乘 $1/\tau$ 放大——hard negative mining 的极限形态，梯度方差与 $\exp$ 的数值压力同时爆炸。可学 $\tau$ 让模型自己在这两端之间选工作点。

CLIP 的参数化（论文 §2/§3，[Radford et al., 2021](../references.md)，[A]）：不直接学 $\tau$，学 $t = \log \gamma = -\log \tau$，对数参数化保证 $\gamma = e^{t} > 0$；初始化 $\tau = 0.07$（即 $\gamma \approx 14.3$、 $t \approx 2.66$）；训练中把 logit scale clamp 在 $\gamma \le 100$（即 $\tau \ge 0.01$）。clamp 的机制：cos 有界使 logit 幅度被钉死在 $\pm \gamma$，上限 100 给 $\exp$ 的数值范围封顶（配合 log-sum-exp 才不出溢出区），并防止训练早期 $\tau$ 缩得过小导致式 (8) 的 $1/\tau$ 把梯度打爆。代入数值感受一下尺度： $\gamma = 14.3$ 时 logit 落在 $[-14.3, 14.3]$， $\gamma = 100$ 时落在 $\pm 100$——同样的余弦差，锐度差 7 倍。列方向对 $t$ 的梯度同形（把行 softmax 换成列 softmax），总梯度是两个方向的平均：温度是全局共享标量，它汇总的是全部 $2B$ 个分类问题里正样本的平均优势，不是逐样本可调。

$\tau$ 自身的梯度有一个干净恒等式。行方向单样本 loss 为 $L_i = -S_{ii} + \log \sum_j \exp(S_{ij})$，写 $S_{ij} = \gamma\thinspace c_{ij}$（ $c_{ij} = \langle v_i, t_j \rangle$），对 $t = \log \gamma$ 求导（ $\partial S_{ij} / \partial t = S_{ij}$ 的链式）：

$$
\frac{\partial L_i}{\partial t} = \sum_{j=1}^{B} \left( p_{ij} - \delta_{ij} \right) S_{ij} = \gamma \left( \sum_{j=1}^{B} p_{ij}\thinspace c_{ij} - c_{ii} \right) \qquad (10)
$$

右端括号内第一项是 softmax 加权的行平均相似度、第二项是正样本相似度； $\gamma > 0$，梯度的符号完全由括号决定。读法：当正样本相似度 $c_{ii}$ 超过该行加权平均时， $\partial L_i / \partial t < 0$，梯度下降增大 $t$（ $\gamma$ 增大、 $\tau$ 减小、分布调尖）——正样本已占优，加锐直接压低 loss；反之 $c_{ii}$ 低于平均时 $\tau$ 被调大，这个错配样本的区分被软化，不对噪声对硬推。可学温度因此是一个自动校准器：把「正样本相对当前负样本池的平均优势」翻译成 softmax 锐度的全局调整量；clamp 100 是这个校准器的硬边界。

两个推论登记。(a) **温度是模型的一部分，不是采样旋钮**：§5 的 zero-shot 公式 (14) 直接复用训练收敛的 $\tau$，因为检索分辨率和训练时的 softmax 锐度是同一个量——推理期改 $\tau$ 等于改变「多大余弦差算显著」的刻度，与训练中学出的 embedding 几何脱节。(b) **无 clamp 的失效形态**：若不对 $\gamma$ 封顶，训练早期 embedding 近随机（ $c_{ij}$ 全在 0 附近），式 (10) 的梯度幅度虽小，但部分实现里 $\exp(\gamma c)$ 在 $\gamma$ 失控增长时先撞 fp16/fp32 的数值墙；clamp 100 把最大 logit 幅度锁在 $\pm 100$，配合 log-sum-exp 的减最大值（1.1 §2），数值上始终在安全区。

## 4. batch size：负样本池的 Monte Carlo 解读

§2.1 已确立 batch 内其余 $B - 1$ 个样本是全部负样本，batch size 因此不是吞吐旋钮，而是**目标函数的构成参数**。把行 loss 的分母展开：

$$
L_i = -S_{ii} + \log \left( \exp(S_{ii}) + \sum_{j \neq i} \exp(S_{ij}) \right) \qquad (11)
$$

求和项改写为 $\sum_{j \neq i} \exp(S_{ij}) = (B - 1) \cdot \frac{1}{B-1} \sum_{j \neq i} \exp\left( \langle v_i, t_j \rangle / \tau \right)$——括号里的平均正是 $`\mathbb{E}_{t' \sim p_{\mathrm{data}}}\left[ \exp\left( \langle v_i, t' \rangle / \tau \right) \right]`$ 的 $B - 1$ 样本 Monte Carlo 估计。代入得

$$
L_i \approx -S_{ii} + \log(B - 1) + \log \mathbb{E}_{t'}\left[ \exp\left( \langle v_i, t' \rangle / \tau \right) \right] \qquad (12)
$$

$\log(B - 1)$ 对参数是常数、不产梯度；有梯度的是第三项——把每个 anchor 对随机负样本的平均亲和压下去。 $B$ 的作用全部落在估计质量上：Monte Carlo 相对误差随 $1/\sqrt{B}$ 收缩， $B$ = 32768 时约 $`0.6\%`$， $B$ = 256 时约 $`6\%`$——小 batch 下分母估计的噪声直接灌进梯度。同时 $B$ 决定任务本身的难度刻度：随机基线 $\log B$ 随 batch 对数增长，大 batch 下「认出正样本」是更难的分类题，解出来需要的对齐信号更强。

Monte Carlo 视角之外还有一个覆盖度视角：负样本池同时是 $p_{\mathrm{data}}$ 的采样近似—— $B$ 小则每个 anchor 只见过多模态分布的一小角，「推开随机负样本」的目标被错估成「推开这一小角」； $B$ 大则负样本池对文本流形的覆盖更全，推开的方向才逼近真实的分布几何。

这一渐近结构有严格形式（[Wang & Isola, 2020](../references.md)，arXiv:2005.10242，[A]，Theorem 1 的记号翻译）：负样本数 $M \to \infty$ 时，

$$
L_{\mathrm{row}} - \log M \hspace{5mu}\to\hspace{5mu} -\frac{1}{\tau}\thinspace \mathbb{E}_{(v, t) \sim p_{\mathrm{pos}}}\left[ \langle v, t \rangle \right] + \mathbb{E}_{v}\left[ \log \mathbb{E}_{t'}\left[ \exp\left( \langle v, t' \rangle / \tau \right) \right] \right] \qquad (13)
$$

**符号**： $p_{\mathrm{pos}}$：正样本对分布。第一项是 alignment（正样本对内积越大越好），第二项是逐 anchor 的 log-partition——压它等价于让 anchor 与随机样本尽量不像，即球面上的 uniformity。该文同时证明：球面均匀分布是这类 pairwise 能量泛函的最小化者，uniformity 项的地板就是「均匀」。「大 batch 提升 uniformity」的机制链条至此完整： $B$ 大 → 式 (13) 第二项的 Monte Carlo 估计更准 → uniformity 目标被真正优化而不是被估计噪声稀释。CLIP 的 $B$ = 32768（论文 §3）正是这个账的工程落点；代价同样直接—— $B$ 个样本的 embedding 必须全部可见才能算分母（6.1 §7 的归一化耦合），32768 对的双塔前向加跨卡 all-gather 构成 CLIP 训练成本的大头。

工程实现里「有效 $B$」的口径要说清：数据并行下每张卡只算本地 micro-batch 的 embedding，不做 all-gather 时分母里的负样本池就缩成本地大小——负样本数由**全局 gather 后的 batch** 决定，与单卡 micro-batch、梯度累积步数无关。open_clip 的 `ClipLoss` 正是先 gather 再算 $S$（[open_clip](../references.md)，[A]），这与 §2.3 代码的教学版（单机单 batch）只差这一步通信。

## 5. zero-shot 分类 = 以文搜图的检索

训练好的双塔不需要任何微调就能当分类器，机制是把分类改写成检索。给定类别集合 $\lbrace1, \ldots, K\rbrace$，每个类名 $y$ 套进 prompt 模板（如 `a photo of a {label}`）过文本塔得 $t_{\mathrm{prompt}(y)}$；图像 $x$ 的类别后验就是一行 InfoNCE softmax——候选集从 batch 换成类别表：

$$
p(y \mid x) = \frac{\exp\left( \langle v_x,\hspace{5mu} t_{\mathrm{prompt}(y)} \rangle / \tau \right)}{\sum_{y'=1}^{K} \exp\left( \langle v_x,\hspace{5mu} t_{\mathrm{prompt}(y')} \rangle / \tau \right)} \qquad (14)
$$

**符号**： $v_x$：待分类图像的单位化 embedding； $\tau$：训练出的温度，直接复用。式 (14) 与式 (3) 是同一个对象：在 $K$ 个候选文本里检索最配的一张图，argmax 即预测类别。零样本（zero-shot）的「零」指的是零次梯度更新——分类头由文本塔现场生成，权重一个不动。

把评测协议说死，避免与另一条常见路线混淆：zero-shot 是式 (14) 的纯检索，不碰下游标签；linear probe 是冻结双塔、在 embedding 上额外训一个线性分类头，要用下游训练集。CLIP 论文两条都报（摘要口径：400M 对训练后，zero-shot 在 ImageNet 上匹配了原始 ResNet-50 的准确率，不用一张 ImageNet 标签；并在 30 多个数据集上以同一协议评测，[Radford et al., 2021](../references.md)，[A]）——前者度量「空间几何天然对不对」，后者度量「特征线性可分性好不好」，是两件事。计算账也顺手记： $K$ 条类别文本各过一次文本塔（可离线缓存），推理期分类 = 一次矩阵乘 $[n, d] \times [d, K]$。

prompt 本身成为可工程化的自由度。**prompt ensembling**：同一类别用 $M$ 个模板（`a photo of a {label}`、`a sketch of a {label}` 等），embedding 取平均后重新单位化：

$$
\bar t_y = \frac{\tilde t_y}{\lVert \tilde t_y \rVert_2}, \qquad \tilde t_y = \frac{1}{M} \sum_{m=1}^{M} g\left( \mathrm{prompt}_m(y) \right) \qquad (15)
$$

论文在 ImageNet 上集成了 80 个模板（论文 §2.4 口径，[A]）。机制解读：平均消掉模板特有的措辞方差——同一类别的不同措辞（`a photo of a dog` 与 `a sketch of a dog`）在球面上各有散布，均值向量留下类别语义的主方向；代价是类别数 × 模板数倍的文本塔前向，但只需离线跑一次。单模板与集成之间的精度差是 prompt 敏感性的直接度量：裸类名（`dog`）脱离训练分布里「图说」的语境，套一个 `a photo of a {}` 就把查询拉回到 caption 语域——prompt 工程在这里不是玄学，是把推理查询对齐到训练文本分布 $p_{\mathrm{data}}$ 的分布匹配动作。

「类别文本当可检索库」的构造同时给出开放词表：推理期增删类别 = 增删几条编码好的文本，不碰任何权重——§1 开头「封闭类别表」的问题在这里闭环。同一个 $S$ 的构造也支持同模态检索（图搜图用 $\langle v_i, v_j \rangle$、文搜文用 $\langle t_i, t_j \rangle$），因为两个模态已被钉进同一个单位球面，余弦几何在模态内部同样成立。

## 6. 表征几何诊断：alignment 与 uniformity

训练 loss 是 surrogate，表征空间本身的质量用两个直接可测的几何指标诊断（[Wang & Isola, 2020](../references.md)，arXiv:2005.10242，[A]，指标定义按该文 §3 改写为本章双模态记号）：

$$
L_{\mathrm{align}} = \mathbb{E}_{(v, t) \sim p_{\mathrm{pos}}}\left[ \lVert v - t \rVert_2^{2} \right] = 2 - 2\thinspace \mathbb{E}\left[ \cos\theta \right] \qquad (16)
$$

**符号**：第二个等号用单位化恒等式 $\lVert v - t \rVert_2^{2} = 2 - 2\langle v, t \rangle$。alignment 低 = 正样本对贴合。测量口径：留出集的正样本对上取经验平均。

$$
L_{\mathrm{uniform}} = \log \mathbb{E}_{\thinspace v,\thinspace v' \stackrel{iid}{\sim} p_{\mathrm{data}}}\left[ \exp\left( -2 \lVert v - v' \rVert_2^{2} \right) \right] \qquad (17)
$$

**符号**： $v, v'$：从 embedding 边际分布独立采两点（双模态下对两池分别测、或合并测，口径需在报告时写清）；核取 Gaussian potential，系数 2 为该文主档。uniformity 低 = 点在球面上铺得开；§4 已登记：球面均匀分布是这类 pairwise 能量的最小化者，式 (17) 的地板即「均匀」。单位化后 $\exp(-2\lVert v - v' \rVert^2) = \exp(4\cos\theta - 4)$，所以均匀性完全由随机两点夹角的余弦分布决定（练习 4 做化简与 toy 计算）。

两个指标各管一种失败模式：alignment 偏高 = 图文没对上（正样本对 cos 低，检索错配）；uniformity 偏高 = 表征塌缩进球面一小块——全体向量挤成细锥、随机两点 cos 全在 0.9 以上，余弦失去区分度，检索分辨率丧失（各向异性）。式 (13) 的渐近分解说明 InfoNCE 同时在压这两项；§3 的 $\tau$ 控制两项被优化时的锐度分辨率，§4 的 $B$ 控制 uniformity 项的估计精度——几何诊断把超参选择翻译成可观测量的调参。

测量协议上有三条纪律。(a) 两指标都在留出集上测——训练集上的 alignment 会被过拟合美化；(b) uniformity 的期望用样本平均做 Monte Carlo，样本对数要上万，否则估计方差盖过模型间差异（与 §4 同一个 $1/\sqrt{B}$ 账）；(c) 双模态场景的 uniformity 有两池可选（图池、文池）与合并池，三者数值不可混报——对齐是跨模态性质，均匀性是逐池性质。训练动态上两指标通常一起走，但超参失调时会分叉： $\tau$ 过大（梯度趋于 0 的极端）两指标都停滞； $B$ 过小则 alignment 照降而 uniformity 被噪声拖着走——诊断表的价值正在把「loss 在降但检索不行」拆到具体一项上。落地流程一句话：取任一 CLIP 系权重（open_clip 提供全家系，[open_clip](../references.md)，[A]），在留出图文对上跑一次双塔、单位化，式 (16) 用正样本对均值、式 (17) 用随机配对的样本平均各估一次，两张数各对应诊断表一行。

## 7. 下游角色与数据工程（伏笔登记）

CLIP 训出的共享空间在两个后续 Part 直接复用，本章只登记接口与角色边界。**VLM 视觉塔**：CLIP/SigLIP 系视觉塔连同其单位球面表征被接入 LLM 作为视觉前端（当代默认是 6.1 §8 的 SigLIP 2 族）——因为 embedding 空间已经与文本语义对齐过，跨模态投影层只需学一个近乎保距的浅映射就能让 LLM「读懂」视觉 token，这是图文对比预训练对多模态 LLM 的核心馈赠，P8 展开。**T2I 文本编码器**：diffusion 系文生图模型直接冻结 CLIP 文本塔作条件编码器，prompt 的语义几何就是 CLIP 空间的几何，条件注入的质量上限部分由本章的 alignment/uniformity 决定，P7 展开。两个角色共享同一条性质：下游消费的是**空间几何**（相对位置与夹角），不是单个向量——这正是对比目标训出来的东西。

数据工程一句话：400M 图文对取自网页共现，构造侧用一份 50 万个检索 query 的列表做配额采样（每个 query 至多 2 万条）加清洗来控制噪声与类目均衡（论文 §2.1），监督信号本质是「共现即弱标签」——配比、去重与清洗配方属于 training 维度，本章只把网页共现分布当作式 (11)-(13) 期望下的 $p_{\mathrm{data}}$。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 共享空间 | 式 (1)：双塔 + 投影 + 单位化 | 几何 = 单位球面夹角；CLIP ViT-L/14 档 $d$ = 768 |
| InfoNCE | 式 (2)-(5) | $S = VT^{\top}/\tau$；行/列两个 $B$ 类 CE 取平均；对角为正、同行列其余 $B-1$ 个为负样本；随机基线 $\log B$ |
| 梯度分解 | 式 (6)-(8) | 拉正样本（力 ∝ $1 - p_{ii}$）+ 推负样本（力 ∝ $p_{ij}$）；hard negative mining 是 softmax 的导数属性 |
| 温度 $\tau$ | 式 (9)(10) | $\tau \to 0$ 最难负样本独大、 $\tau \to \infty$ 梯度归零；可学 $t = \log(1/\tau)$，init 0.07、 $\gamma \le 100$ clamp；式 (10) 锐度自校准 |
| batch size | 式 (11)-(13) | 分母 = $B-1$ 样本 Monte Carlo 估计，误差 ∝ $1/\sqrt{B}$；渐近分解 = alignment + uniformity；CLIP $B$ = 32768 |
| zero-shot | 式 (14)(15) | 分类 = 以文搜图检索；ensembling = 模板 embedding 平均再单位化（ImageNet 80 个模板）；摘要口径匹配 ResNet-50 |
| 几何诊断 | 式 (16)(17) | alignment 低 = 对得齐；uniformity 低 = 铺得开；塌缩与错配各对应一项 |
| 下游伏笔 | §7 | VLM 视觉塔 → P8；T2I 文本编码器 → P7；数据工程 → training 维度 |

## 练习

1.（推导）(a) 写出列方向 $`p^{c}_{ij}`$ 的定义，由式 (5) 证明式 (7)；(b) 补全式 (8)：明确指出 $\sum_j p_{ij} = 1$ 用在哪一步，并把「吸引 + 排斥」改写成 $\frac{1}{B\tau}\left[ (1 - p_{ii})\thinspace t_i - \sum_{j \neq i} p_{ij}\thinspace t_j \right]$ 验证等价；(c) 若丢掉列方向只训式 (3)， $t_j$ 从哪些项获得梯度、缺了哪些项？由此说明单方向训练对以文搜图的影响。
2.（计算）(a) $B$ = 32768 时随机基线行 CE 是多少 nats？训练末期某行 CE 降到 2.0，该行正样本概率与随机基线各是多少、差多少倍？(b) 某行 $c_{ii}$ = 0.30、 $\sum_j p_{ij} c_{ij}$ = 0.18、 $\gamma$ = 14.3，用式 (10) 算 $\partial L_i / \partial t$，说明这一步会把 $\tau$ 调大还是调小、为何合理。(c) $\tau$ 固定、 $B$ 从 1024 升到 32768，式 (12) 中 Monte Carlo 相对误差约缩多少倍？
3.（代码）用 §2.3 的教学实现：(a) 构造 $B$ = 64、 $d$ = 32 的随机单位化 $V, T$（对角人为加 0.3 余弦偏置），跑 200 步 Adam 只更新 $V, T$，验证对角余弦均值上升、行 CE 从约 $\log 64$ 单调下降；(b) 把温度设为可学（对数参数化 + clamp 100），打印前 50 步的变化方向并用式 (10) 解释符号；(c) 手写一行梯度（式 (6)）与 autograd 输出对拍到 $10^{-6}$。
4.（推导 + 计算）(a) 用单位化恒等式把式 (16) 化成 $2 - 2\thinspace\mathbb{E}[\cos\theta]$、式 (17) 化成 $-4 + \log \mathbb{E}\left[ e^{4\cos\theta} \right]$；(b) 构造两个 toy 分布：A（正样本对 cos = 0.9、随机对 cos = 0）与 B（正样本对 cos = 0.9、随机对 cos = 0.85），期望用 delta 近似各算两指标，指出 B 属于 §6 的哪种失败模式、对应哪个指标失灵。

## 参考

- [Radford et al., 2021](../references.md)：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020，[A]（key: clip）——式 (2)-(5) 双向 InfoNCE、可学温度（init 0.07、对数参数化、clamp 100）、400M 对与 30+ 数据集、zero-shot 匹配 ResNet-50（摘要）、80 模板 ensembling（§2.4）、batch 32768（§3）、[EOS] 读出与文本塔结构（§2.1）
- [Wang & Isola, 2020](../references.md)：Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere，arXiv:2005.10242，[A]（key: alignment-uniformity）——式 (13) 渐近分解（Theorem 1）、式 (16)(17) 指标定义（§3）、球面均匀分布最小化 pairwise 能量
- [openai/CLIP](../references.md)：[A]（key: clip-repo）——官方模型定义与 zero-shot 协议；CLIP ViT-L/14 共享维 $d$ = 768
- [mlfoundations/open_clip](../references.md)：[A]（key: open-clip）——`src/open_clip/loss.py` 的 `ClipLoss.forward`，§2.3 代码走读锚点（跨卡 all-gather 拼全局 batch）
