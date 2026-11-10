# 2.2 Tokenization 与 Embedding（Tokenization and Embeddings）

> 本章拆开数据进模型的第一道关口与参数表里第一个、最后一个矩阵：BPE / WordPiece / Unigram 三套 tokenizer 训练算法（含手算例与 EM outline）、vocab size 的 T × V × d 三方账、embedding 查表的等价性与稀疏梯度、weight tying 的梯度耦合、softmax 瓶颈的完整推导，以及 tokenizer-free 的 BLT 对照 | 前置：1.1（token / vocab / bits per byte / softmax 瓶颈引入）、2.1（2Vd 参数口径）| 锚点：Qwen3-8B tokenizer 文件 [A]、Llama-3.1-8B config [A]、GPT-2 [W]；Sennrich et al.（BPE）[A]、Kudo（Unigram）[A]、Wu et al.（WordPiece）[B]、Press & Wolf（tying）[B]、Yang et al.（softmax bottleneck）[A]、Pagnoni et al.（BLT）[A]

## 1. tokenizer 在数据链路中的位置

Qwen3-8B 的 config.json 写着 vocab_size = 151936，而同一 repo 里的 vocab.json 只有 151643 个条目。差值的去向：26 个 added special token（id 151643–151668，含 `<|endoftext|>` 、`<|im_start|>` 、`<|im_end|>` 、`<think>` 、`</think>` ），其余为向上对齐的 padding（[Qwen3-8B config](../references.md)，[A]）。config 的 $V$ 决定 embedding 矩阵的行数，vocab.json 决定文本能被切成什么——「词表」在这两个文件里不是同一个对象，本章两个含义都会用到。

从字符串到向量的一段链路：

```
原始字符串   "newest"
    │  UTF-8 编码
    ▼
byte 序列    [110, 101, 119, 101, 115, 116]     ← byte-level 起点：256 个基元
    │  tokenizer 切分（§2–§5：merge 表或概率模型）
    ▼
token 序列   ["new", "est"]  →  id 序列          ← 1.1 样本空间的元素
    │  查表 e_t = E_{id,:}（§8，gather 无乘加）
    ▼
embedding 向量序列  [T, d]  →  residual stream 起点（2.1 式 (1)）
```

tokenizer 的训练是一次性的、先于 LM 训练的压缩问题求解；一旦冻结，训练与推理全程不再改变——这个决定的代价与收益就是 §7 的三方账。本章产出两个后续章节直接复用的结论：§7 的 vocab 账（P5 逐家对账时引用）与 §9 的 tied / untied 梯度分析（2.8 复用）。

## 2. BPE：频率驱动的 merge

BPE（byte-pair encoding）用于 NLP 分词的形式来自 Sennrich et al.（[Sennrich et al., 2016](../references.md)，arXiv:1508.07909，[A]）。输入是按词（或预切分单元）统计好频次的语料；初始切分把每个单元拆成基元符号（字符或 byte），然后循环 merge：统计相邻对频次，把最高频的一对焊成新符号加入词表，并重写语料。每轮的形式化：

$$
n(a, b) = \sum_{w} n(w) \cdot \#_{w}(a, b), \qquad (a^{*}, b^{*}) = \operatorname{arg\,max}_{(a, b)} \; n(a, b) \qquad (1)
$$

**符号**：$n(w)$ ：词 $w$ 的出现次数；$\#_w(a, b)$ ：$w$ 的当前切分中相邻对 $(a, b)$ 的出现次数。词表初始为基元集，每轮 +1，直到达到预算 $V$ 。每条 merge 的学习顺序编号即它的优先级（rank），编码时要靠它（见下）。

**手算例**。语料取四个词：low ×5、lower ×2、newest ×6、widest ×3（词频为演示而设）。初始基元 10 个（l, o, w, e, r, n, s, t, i, d），语料总长 $5 \times 3 + 2 \times 5 + 6 \times 6 + 3 \times 6 = 79$ 个符号。前四轮 merge：

| 轮 | merge | 加权频次 | merge 后语料总长 |
| --- | --- | --- | --- |
| 0 | — | — | 79 |
| 1 | e+s → es | 9 | 70 |
| 2 | es+t → est | 9 | 61 |
| 3 | l+o → lo | 7 | 54 |
| 4 | lo+w → low | 7 | 47 |

第 1 轮 (e,s) 与 (s,t) 同为 9，取谁由实现的 tie-breaking 决定；第 2 轮立刻把 est 合成——**merge 可以组合，token 因此带层级结构**。四轮下来语料长度砍掉约 40%。

编码新文本不需要重新统计：把新词拆成基元，**按 rank 从小到大依次套用 merge 表**（每条 merge 能套就套的贪心）。用上面四条 merge 编码没见过的 lowest：l-o-w-e-s-t →（rank 1）l-o-w-es-t →（rank 2）l-o-w-est →（rank 3）lo-w-est →（rank 4）low-est。未见词被分解为见过的子词——subword 方案对未见词的泛化靠的就是这个组合性。

教学用实现（非优化实现）：

```python
from collections import Counter

def train_bpe(corpus, num_merges):
    """教学用，非优化实现。corpus: {词: 频次}；返回 merge 表（按 rank 排序）。"""
    splits = {w: list(w) for w in corpus}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w, cnt in corpus.items():
            s = splits[w]
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += cnt          # 式 (1) 的 n(a, b)
        if not pairs:
            break
        merges.append(max(pairs, key=pairs.get))        # argmax 频次
        a, b = merges[-1]
        for w, s in splits.items():                     # 重写语料
            out, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out
    return merges, splits

def encode(word, merges):
    """按 rank 顺序贪心套 merge（教学用，非优化实现）。"""
    s = list(word)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == (a, b):
                out.append(a + b); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s
```

**压缩视角**。定义语料的符号总长 $L_{\mathrm{corpus}} = \sum_w n(w) \, |\mathrm{split}(w)|$ 。一次加权频次为 $c$ 的 merge 把 $c$ 处「两个符号」换成「一个符号」，总长恰好减 $c$ ——上表最后一列就是这样算出来的。BPE 的 argmax 频次规则因此就是对语料符号数的贪心压缩：tokenizer 训练是压缩问题，与 1.1 §5「LM 即压缩」同构——那边压的是概率模型下的码长，这边压的是码本下的符号数。两者在 bits per byte 处合流（§6）。

## 3. byte-level BPE：OOV 在机制上的消除

基元取 256 个 byte 值而不是 Unicode 字符，**未登录词（out-of-vocabulary, OOV）** 在机制上就不存在：任何字符串经 UTF-8 编码都是 byte 序列，最坏情况退化到每 byte 一个 token，编码函数对任意输入有定义。两个真实词表的分解印证这个结构：

- GPT-2：$V$ = 50257 = 256（byte 基元）+ 50000（merge 数）+ 1（特殊 token `<|endoftext|>` ）（[Radford et al., 2019](../references.md)，[W]）。
- Qwen3-8B：151936 = 151643（vocab.json，基元与 merge 产物）+ 26（added special token）+ padding（[Qwen3-8B config](../references.md)，[A]）。

代价写在非拉丁文字上：一个 CJK 字符占 3 个 UTF-8 byte，merge 表没覆盖它时一个汉字就是 3 个 token——词表的多语言覆盖不足会直接抬高该语言的 fertility（§6）。这是大词表路线（§7 的 128K / 151K / 262K）的主要动机之一。

## 4. WordPiece：似然驱动的 merge

WordPiece（[Wu et al., 2016](../references.md)，arXiv:1609.08144，[B]；出自 GNMT 论文，无独立开源实现，常见实现为社区重写）与 BPE 共享自底向上的 merge 骨架，差别在选对规则：不看原始频次，看**合并给语料似然带来的增益**。把当前切分当作 unigram 语言模型 $p(u) = n(u) / N$（$N$ 为符号总数），merge $(a, b)$ 的得分取

$$
\mathrm{score}(a, b) = \frac{p(ab)}{p(a) \, p(b)} = \frac{n(a, b) \cdot N}{n(a) \, n(b)} \qquad (2)
$$

**符号**：$p(ab)$ ：合并后新符号的经验概率；分母是 $a$ 、 $b$ 独立假设下的共现概率。式 (2) 是 WordPiece 选对准则的社区标准重述口径——GNMT 原文只描述「合并使训练语料似然增益最大」的准则，未给显式公式。

逐轮选 score 最大的对。读法：式 (2) 是共现对独立性的偏离度——频次高但各自更高频的对（如 e+s）被压下去，「黏性强」的对优先合并。编码端是另一套规矩：从串首开始的贪心最长匹配，子词间的续写关系由实现侧的标记表达——这与 BPE 编码的 rank 序贪心不同，两套规矩不通用（§12 常见误解）。

## 5. Unigram LM：把切分建成概率模型

Unigram 方案（[Kudo, 2018](../references.md)，arXiv:1804.10959，[A]；SentencePiece 为其官方实现）换方向：不构造 merge 规则，直接给「切分」一个概率模型，从一个远大于目标的种子词表出发自顶向下剪枝。

模型假设：词表 $\mathcal{V}$ 上每个 token 有概率 $p(v)$ ，$\sum_{v \in \mathcal{V}} p(v) = 1$ ；字符串 $x$ 的合法切分全体记为 $S(x)$（切法数的上限为 $2^{n-1}$ ：每对相邻位置切或不切，词表约束再筛掉一批）；切分 $z = (z_1, \ldots, z_k)$ 的概率取独立乘积 $p(z) = \prod_{i=1}^{k} p(z_i)$ 。字符串的边缘似然是全部切分之和，训练目标为

$$
\max_{\theta} \; \sum_{x \in D} \log \sum_{z \in S(x)} p(z; \theta) \qquad (3)
$$

**符号**：$\theta = \{ p(v) \}_{v \in \mathcal{V}}$ ：词表上的 unigram 分布；$D$ ：语料；内层求和把指数多个切分边缘化。

双重求和直接优化不可行，走 EM：

**E 步**：在每个 $x$ 上求切分的后验 $q(z \mid x) = p(z) / p(x)$ 。利用格结构，边缘似然用 forward 递推求得：记 $\alpha_i$ 为前缀 $x_{1:i}$ 的边缘似然，

$$
\alpha_0 = 1, \qquad \alpha_i = \sum_{j < i, \; x_{j+1:i} \in \mathcal{V}} \alpha_j \, p\big(x_{j+1:i}\big) \qquad (4)
$$

**符号**：$\alpha_i$ ：前缀 $x_{1:i}$ 全部合法切分的概率和；求和只跑遍「$x_{j+1:i}$ 在词表中」的 $j$ ，递推复杂度 $O(n \cdot L_{\max})$ ，$L_{\max}$ 为最长 token 长度。配一个对称的 backward 递推 $\beta_j$ ，token $v$ 在后验下的期望计数 $c(v)$ 由 $\alpha$ 、 $\beta$ 在格上拼出（对语料求和）。

**M 步**：期望计数归一化即闭式更新 $p(v) = c(v) / \sum_{v'} c(v')$ （对带约束 $\sum_v p(v) = 1$ 的辅助函数求极值，推导留作练习 2）。

**剪枝**：每轮 EM 后按「从词表删掉该 token 导致的对数似然损失」排序，砍掉损失最小的一批（保留比例是实现超参），迭代至目标词表大小 $V$ 。

编码（推理）时不再边缘化，取最可能切分：

$$
\hat{z} = \operatorname{arg\,max}_{z \in S(x)} \; p(z) \qquad (5)
$$

格上的 Viterbi 解码与式 (4) 同形、把求和换成 max：$\delta_i = \max_{j} \; \delta_j \, p(x_{j+1:i})$ 。

式 (3) 的后验还附赠一个 BPE 没有的东西：**切分的随机性**。训练 LM 时按 $q(z \mid x)$ 采样切分而不是恒用 Viterbi，同一句话以不同粒度反复出现——这就是该论文标题所指的 subword regularization。

三算法对照：

| | BPE | WordPiece | Unigram |
| --- | --- | --- | --- |
| 构词表方向 | 自底向上 merge | 自底向上 merge | 自顶向下剪枝 |
| 选对 / 留词准则 | 频次 argmax，式 (1) | 似然增益 argmax，式 (2) | 边缘似然 EM + 剪枝，式 (3) |
| 编码 | 按 rank 贪心 | 最长匹配贪心 | Viterbi 最优，式 (5) |
| 官方实现 | 无单一定点（HF tokenizers 等社区实现） | 无独立开源实现 | google/sentencepiece |

## 6. 质量指标：fertility 与 compression ratio

同一语料、两个 tokenizer，比较只需要两个比值：

$$
\text{fertility} = \frac{\#\text{tokens}}{\#\text{words}}, \qquad r_{\mathrm{comp}} = \frac{n_{\mathrm{byte}}}{T} \qquad (6)
$$

**符号**：fertility：平均每词 token 数，应按语言分别统计；$r_{\mathrm{comp}}$ ：平均每 token 字节数（压缩率）；$T$ ：语料 token 总数；$n_{\mathrm{byte}}$ ：语料字节总数。

fertility 低 = 切得粗 = 同一文本的 token 序列更短。但 per-token 指标跨 tokenizer 不可比（1.1 §6 的两个陷阱：token 数一变，per-token CE 的平均项数就变，PPL 刻度跟着变）。公平的模型间比较口径是 bits per byte：总 NLL 折 bits 除以字节数，tokenizer 从公式里消掉。工程侧 $T$ 本身就是成本：固定字节预算下 $T$ 决定训练序列预算与推理的串行 decode 步数（2.4），$r_{\mathrm{comp}}$ 每提高一分，全链路的 token 成本同比例下降。

## 7. vocab size 的三方权衡账

五个真实词表锚点：

| 模型 | $V$ | tokenizer | 出处 |
| --- | --- | --- | --- |
| GPT-2 | 50257 | byte-level BPE | [Radford et al., 2019](../references.md)，[W] |
| Llama-3.1 | 128256 | byte-level BPE | [Llama-3.1-8B config](../references.md)，[A] |
| Qwen3 | 151936 | byte-level BPE | [Qwen3-8B config](../references.md)，[A] |
| Gemma 2 | 256128 | SentencePiece | 论文口径：[Gemma Team, 2024](../references.md)（arXiv:2408.00118，[A]）；HF 仓库 gated |
| Gemma 3 | 262144 | SentencePiece | 论文口径：[Gemma Team, 2025](../references.md)（arXiv:2503.19786，[A]）；HF 仓库 gated |

跨代对照的补充锚点在 Llama 3 论文里：其 128K 词表 tokenizer 的压缩率为 3.94 chars/token，论文同处给出前一代 Llama 2 tokenizer 的对照值 3.17 chars/token（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）——换词表后每 token 平均多吞约 24% 的字符。Llama-2 词表的通行数值 32000 出自其 HF config（[W]：有 config 与权重、无独立论文条目，故不进对照表）。

$V$ 同时挤压三个方向，这就是 T × V × d 账：

1. **$V \uparrow \Rightarrow T \downarrow$（序列账）**。merge 越多、压缩越高，同字节文本的 token 数越少：attention 的 $O(T^{2} d)$ 项（1.2 第 6 节）、推理的串行 decode 步数（2.4）都随 $T$ 省。这是把 $V$ 往大推的力。
2. **$V \uparrow \Rightarrow$ 参数与 softmax 账 $\uparrow$**。untied 时输入 embedding 加输出头共 $2Vd$ 参数（2.1 式 (4)）：Llama-3.1-8B 为 $2 \times 128256 \times 4096 = 1.05 \times 10^{9}$ ，占全部 8.03B 参数的 13.1%；Qwen3-8B 为 $1.245 \times 10^{9}$ 。输出头的每 token 前向 FLOPs 同为 $2Vd$（Llama-3.1-8B 约 $1.05 \times 10^{9}$ FLOPs，对照层部分 $2N \approx 1.4 \times 10^{10}$ FLOPs，约占 7%），词表上的 softmax 归一化与采样每位置再花 $O(V)$ ，相对输出头 matmul 的 $O(Vd)$ 是低阶项。
3. **$V \uparrow \Rightarrow$ 学习难度 $\uparrow$**。固定数据预算下词表越大长尾越长，低频 token 的 embedding 行被更新的次数越少（§8 的稀疏梯度）；分类类别数变多，uniform 基线 $\ln V$ 抬高（1.1 §6）。这是把 $V$ 往小压的力。

50257 / 128256 / 151936 / 256128 / 262144 这五档不是「越大越好」的爬坡，而是三笔账在不同数据配比（多语言与代码占比）下的不同最优解；P5 各章对具体模型的 vocab 选择对账时复用本表。

## 8. embedding 层：one-hot 等价、稀疏梯度与初始化

embedding 层是整数下标到向量的映射。记 one-hot 行向量 $o_t = \mathrm{onehot}(x_t) \in \{0, 1\}^{V}$ ，则

$$
e_t = o_t \, E = E_{x_t, :}, \qquad E \in \mathbb{R}^{V \times d} \qquad (7)
$$

**符号**：$E$ ：输入 embedding 矩阵（2.1 式 (1) 的同一对象）；$e_t \in \mathbb{R}^{d}$ ：token $x_t$ 的向量；批量后 shape 为 $[B, T, d]$ ，与 2.1 式 (1) 的 $x^{(0)}$ 对齐。

one-hot 乘矩阵等于按行取——实现上是 gather（索引读），没有乘加：1.2 的 FLOPs 账把 embedding 排除在外，原因就是这个等价性，它不是 matmul 而是内存寻址。反向传播同样稀疏：

$$
\frac{\partial L}{\partial E} = \sum_{t} o_t^{\top} g_t^{\top}, \qquad g_t = \frac{\partial L}{\partial e_t} \qquad (8)
$$

**符号**：$g_t$ ：backbone 回传给 embedding 向量的上游梯度；$o_t^{\top} g_t^{\top}$ 是 $[V, 1] \times [1, d]$ 的外积，只在第 $x_t$ 行非零。

每一步只有 batch 中出现过的 token 行拿到非零梯度。两个直接后果：低频 token 的行长期停在初始化附近（cold rows）；$V$ 越大长尾越冷——§7 第三方账在梯度侧的根据就是式 (8)。

教学验证（非优化实现）：

```python
import torch
import torch.nn.functional as F

V, d = 8, 4
E = torch.randn(V, d)
ids = torch.tensor([2, 5, 2])
# one-hot 乘矩阵 == 按行 gather（式 (7)）
assert torch.equal(F.one_hot(ids, V).float() @ E, F.embedding(ids, E))
```

**初始化尺度的 0.02 惯例**。GPT-2 的 initializer_range = 0.02（[Radford et al., 2019](../references.md)，[W]，config 字段实测），Qwen3-8B 的 config 取同一数值（[Qwen3-8B config](../references.md)，[A]）。依据是一笔方差账：行向量按 $E_v \sim \mathcal{N}(0, \sigma^{2} I)$ 初始化时 $\mathbb{E}\|E_v\|^{2} = d \sigma^{2}$ ；输出头同样初始化时，若 final RMSNorm 后的 $h$ 分量方差约为 1，则 logit 方差 $\mathrm{Var}(z_v) = d \sigma^{2}$ ，即

$$
\mathrm{std}(z_v) = \sigma \sqrt{d} \qquad (9)
$$

**符号**：$\sigma$ ：初始化标准差；$z_v$ ：token $v$ 的 logit。

$d$ = 4096、$\sigma$ = 0.02 时 $\mathrm{std}(z_v) \approx 1.28$ ：初始 logits 是 $O(1)$ 小量，初始 softmax 接近均匀，初始 loss $\approx \ln V$（Qwen3-8B：$\ln 151936 \approx 11.9$）。对照 Xavier 尺度 $1 / \sqrt{d} \approx 0.0156$（$d$ = 4096），0.02 同量级。式 (9) 也给出惯例的失效方向：固定 $\sigma$ 而 $d$ 翻倍，初始 logit 方差跟着翻倍——0.02 不是普适常数，是 $d$ 在数百到数千区间的经验值。

## 9. weight tying：一份参数干两份活

**权重绑定（weight tying）**：输出头与输入 embedding 共享同一矩阵，$W_U = E$ ，logits 由 $z_t = E h_t$ 给出，参数账从 $2Vd$ 降为 $Vd$（[Press & Wolf, 2017](../references.md)，arXiv:1608.05859，[B]）。GPT-2 是 tying 收益的标准算例：层参数 $12Ld^{2} = 12 \times 12 \times 768^{2} \approx 84.9 \times 10^{6}$ ，tied 词表矩阵 $Vd = 50257 \times 768 \approx 38.6 \times 10^{6}$ ，合计约 123.5M，与通行的 124M 口径一致（[Radford et al., 2019](../references.md)，[W]）——若 untie，这个量级的模型要多出约 31% 的参数。而 Llama-3.1-8B、Qwen3-8B、OLMo-2-7B 的 tie_word_embeddings 全为 false（各 config 条目，[A]）：8B 量级上 $2Vd$ 只占 13%–15%，买得起了。

省参数之外，tying 改变的是梯度结构。记每位置 loss 对 logits 的梯度 $r_t = q_t - \delta_{x_{t+1}} \in \mathbb{R}^{V}$（1.1 §4 的 softmax-CE 复合导数），则共享矩阵 $E$ 收到两路梯度：

$$
\frac{\partial L}{\partial E} = \underbrace{\sum_{t} r_t \, h_t^{\top}}_{\text{output role}} + \underbrace{\sum_{t} o_t^{\top} g_t^{\top}}_{\text{input role}} \qquad (10)
$$

**符号**：$h_t$ ：位置 $t$ 的 final 隐状态（$d$ 维）；$r_t$ ：softmax 概率减真值；$o_t$ 、 $g_t$ 同式 (7)(8)。两路都是 $[V, d]$ 。

逐行看耦合：第 $v$ 行的更新是 $\sum_t r_{t,v} \, h_t + \sum_{t : x_t = v} g_t$ 。第一路**每个位置更新每一行**（非目标行也有 $r_{t,v} = q_{t,v} > 0$ ），把行推向「正确分类器方向」；第二路只在 $v$ 出现在输入时更新该行，把行推向「有用的输入特征方向」。两个角色对同一向量提出不同需求：作为分类器，$E_v$ 要与「应当预测 $v$ 」的隐状态方向对齐；作为输入特征，$E_v$ 要编码「$v$ 出现了」对后续预测有用的信息——高频功能词作为输入信息量低、作为目标出现频繁，两个方向一般不重合，tied 行被迫折中。untie 就是让两份几何各自收敛。与 §10 的衔接：tied 时输入映射与输出映射共用同一矩阵的 rank 预算，untie 把联合参数预算翻倍（单矩阵的 rank 上限仍为 $d$ ）——大模型倾向 untie 的容量侧理由就在这里。2.8 组装整机时复用本节的 tied / untied 判定口径。

工程注记：共用一份矩阵时输入侧与输出侧的合理动态范围不同，老配方靠显式缩放调和——Transformer 原文在 tied 配方下把 embedding 乘 $\sqrt{d}$ 再进 stream（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.4）；现代配方直接 untie，缩放问题随之一并消失。

## 10. softmax 瓶颈：输出头 rank 上限的完整推导

1.1 §9 引入了 softmax 瓶颈并给出结论 $\mathrm{rank}(\log P) \le d + 2$ ；此处把推导补全，再讨论它与 tying 的关系。取 $N$ 个不同上下文，隐状态堆成 $H \in \mathbb{R}^{N \times d}$ ，logits 矩阵

$$
Z = H W_U^{\top} + e_N \, b^{\top} \in \mathbb{R}^{N \times V} \qquad (11)
$$

**符号**：$e_N \in \mathbb{R}^{N}$ ：行方向全 1 向量；$W_U \in \mathbb{R}^{V \times d}$ 、 $b \in \mathbb{R}^{V}$ ：输出头权重与偏置。$e_N b^{\top}$ 是 $[N, 1] \times [1, V]$ 的外积，维度相容。

逐步推。第一步，$\mathrm{rank}(H W_U^{\top}) \le \min(\mathrm{rank}(H), \mathrm{rank}(W_U)) \le d$（乘积的秩不超过各因子的秩）；$e_N b^{\top}$ 的每一行都是 $b$ ，秩为 1；矩阵和的秩不超过秩的和，故 $\mathrm{rank}(Z) \le d + 1$ 。

第二步，log-softmax 逐行进行：$\log q_{c, v} = z_{c, v} - u_c$ ，其中 $u_c = \log \sum_{v'} \exp(z_{c, v'})$ 只依赖行。每行减去一个行常数，等价于减一个 rank-1 矩阵 $u \, 1_V^{\top}$ ：

$$
\log P = Z - u \, 1_V^{\top} \quad \Longrightarrow \quad \mathrm{rank}(\log P) \le d + 2 \qquad (12)
$$

**符号**：$\log P \in \mathbb{R}^{N \times V}$ ：$N$ 个上下文上的模型对数概率矩阵；$u \in \mathbb{R}^{N}$ ：逐行归一化常数；$1_V \in \mathbb{R}^{V}$ ：词表方向全 1 向量，$u \, 1_V^{\top}$ 是 $[N, 1] \times [1, V]$ 的外积。无 bias 时第一步收紧为 $d$ ，结果为 $d + 1$ 。

瓶颈的含义：真实语言在 $N$ 个上下文上的对数概率矩阵没有理由低秩——同一前缀之后的合理续写由语义、语法、语用等多组近似独立的因素决定，行与行之间接近满秩；原论文在上下文独立性假设下证明了真实矩阵满秩（[Yang et al., 2018](../references.md)，arXiv:1711.03953，[A]）。只要真实 rank 超过 $d + 2$ ，单 softmax 头就有拟合不了的残差，加深主干、加数据都无效——瓶颈在 $z = W_U h$ 这个线性结构上。代入锚点：$d$ = 4096、$V$ = 151936，rank 上限 4098 不足满秩的 3%。

原论文给的解法是 **Mixture of Softmaxes（MoS）**：$K$ 个打分头的凸组合替代单 softmax，

$$
q(v \mid c) = \sum_{k=1}^{K} \pi_{c, k} \; \frac{\exp\big(h_c^{\top} w_{k, v}\big)}{\sum_{v'} \exp\big(h_c^{\top} w_{k, v'}\big)}, \qquad \pi_c = \mathrm{softmax}(W_{\pi} h_c) \qquad (13)
$$

**符号**：$w_{k, v}$ ：第 $k$ 个打分头里 token $v$ 的向量；$\pi_{c, k}$ ：上下文决定的混合权重。log 无法穿过外层的混合求和，$\log P$ 不再是低秩矩阵的行变换，可达 rank 随 $K$ 放大（详细界见原论文；官方实现 zihangdai/mos）。

MoS 的原始实验对象是 RNN 语言模型；本书的现代锚点（Llama-3.1、Qwen3、OLMo-2）仍用单 softmax 头，bottleneck 残差在 $d$ = 4096 下被默认接受而非解决——这是一个被绕开的限制，不是被消除的限制。与 §9 的接口：tied 时同一个 $E$ 既当式 (11) 的 $W_U$ 又当输入映射，rank 预算一份两用；MoS 与 untie 是两个正交的松手方向。

## 11. tokenizer-free 对照：BLT 的 byte 与动态 patch

§2–§5 的 tokenizer 都在训练前把「字符串 → 离散单元」冻结成静态码本。BLT（Byte Latent Transformer，[Pagnoni et al., 2024](../references.md)，arXiv:2412.09871，[A]；官方 repo facebookresearch/blt）把这个映射改成模型内部的动态决策：

```
bytes   b1 b2 b3 b4 b5 b6 b7 b8 ...
          │  小型 byte LM 逐位计算下一 byte 的熵 H(b_{i+1} | b_{<=i})
          ▼  熵超阈值（或相对增益大）处切开
patch   [b1 b2 b3] [b4 b5] [b6 b7 b8] ...    ← 长度随可预测性变化
          │  轻量 byte encoder 把 patch 压成向量
          ▼
    大 latent transformer（在 patch 序列上计算）
          ▼
    轻量 byte decoder 逐 byte 展开输出
```

机制要点：patch 边界由熵驱动——可预测区段（常见词、固定模板）合并为长 patch、省计算；高不确定区段（专名、随机串）切成短 patch、把算力花在那里。固定 FLOP 预算下「算力按不确定性分配」是静态码本做不到的事。原论文做了 FLOP 对齐的 scaling 对照（至 8B 参数 / 4T bytes），报告在固定推理 FLOPs 下追平或超过 BPE tokenizer 的 Llama-3 式基线。代价侧同样明确：byte 序列长度约为 token 序列的 4 倍量级，byte 侧只能用轻量模块，主算力集中在 patch 级 transformer；同时 $2Vd$ 的词表矩阵整个消失（§7 第二笔账），换来 byte encoder / decoder 的小额固定开销。BLT 目前是研究对照而非主流配方，但它标出了 tokenizer 范式的边界：码本的静态性是工程选择，不是数学必需。

## 12. 真实 tokenizer 走读：Qwen3-8B 与 Llama-3.1

**Qwen3-8B**（repo 文件实测，[Qwen3-8B config](../references.md)，[A]）：tokenizer_class = Qwen2Tokenizer，即 byte-level BPE，与 Qwen2 / 2.5 同一 tokenizer 家族。repo 内三件：vocab.json（token → id，151643 条）、merges.txt（merge 按 rank 排列）、tokenizer.json（HF tokenizers 的编译产物，加载走这条）。added_tokens_decoder 含 26 个特殊 token，占 id 151643–151668（`<|endoftext|>` 、`<|im_start|>` 、`<|im_end|>` 、`<think>` 、`</think>` 等）；config vocab_size 151936 = 151643 + 26 再向上对齐。bos_token_id = 151643，eos_token_id = 151645（聊天模板以 `<|im_end|>` 收尾）。两个长度字段不要混：config 的 max_position_embeddings = 40960 是模型原生窗口，tokenizer_config 的 model_max_length = 131072 是 tokenizer 侧上限（1.1 §1 已核对）。

**Llama-3.1**：config 侧 vocab_size = 128256（[Llama-3.1-8B config](../references.md)，[A]）；Llama 3 论文记录其 tokenizer 为 BPE 家族、128K 词表，并给出压缩率对照：3.94 chars/token 对前一代 Llama 2 tokenizer 的 3.17 chars/token（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）。与 Qwen3 对照着读：同属 byte-level BPE 路线，$V$ 差约 1.18 倍；§7 的三方账（多语言覆盖、$2Vd$ 、长尾冷启动）就是解读这两家取值的框架。

动手核对入口：HF 侧 `AutoTokenizer.from_pretrained` 之后，`tokenizer.get_vocab()` 给词表，BPE 系的 merge 表看 merges.txt 或 `tokenizer.backend_tokenizer`；Unigram 系（SentencePiece 产物）的 token 概率存在 `.model` 文件里。

常见误解三条：

1. **config 的 vocab_size ≠ vocab.json 条目数**。前者含 added special token 与 padding 对齐（151643 vs 151936），embedding 矩阵行数按 config 的算。
2. **max_position_embeddings ≠ tokenizer 的 model_max_length**。前者是模型的原生位置窗口，后者是 tokenizer 侧的服务上限，两个字段分属两个文件（1.1 §1）。
3. **BPE 编码不是最长匹配**。BPE 按 merge 的 rank 序贪心，最长匹配是 WordPiece 的规矩；两套都自称「贪心」但顺序定义不同，混用会切出不同的 token 序列。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| BPE | 频次 argmax merge，式 (1) | 贪心压缩：merge 频次 = 语料缩短量；编码按 rank 贪心 |
| byte-level | 基元 = 256 byte | 无 OOV；GPT-2 50257 = 256 + 50000 + 1；CJK 单字最坏 3 token |
| WordPiece | score = $p(ab) / (p(a) \, p(b))$，式 (2) | 似然增益驱动；最长匹配编码 |
| Unigram | $\max_\theta \sum_x \log \sum_{z \in S(x)} p(z)$，式 (3)–(5) | EM + 剪枝；Viterbi 编码；切分采样 = subword regularization |
| 质量指标 | fertility、$r_{\mathrm{comp}}$，式 (6) | 跨 tokenizer 只比 bits per byte（1.1 §6） |
| vocab 三方账 | $T$ ↓ vs $2Vd$ ↑ vs 长尾冷启动 | 50257 / 128256 / 151936 / 256128 / 262144 对照；Llama-3.1-8B 的 $2Vd$ = 1.05B（13.1%） |
| embedding | $e_t = o_t E$，式 (7)(8) | one-hot 乘 ≡ gather；行稀疏梯度 → cold rows |
| 初始化 | $\sigma$ = 0.02；$\mathrm{std}(z_v) = \sigma \sqrt{d}$，式 (9) | GPT-2 / Qwen3 同值；初始 loss ≈ $\ln V$ |
| weight tying | $W_U = E$；两路梯度耦合，式 (10) | GPT-2 tied（124M 算例）；现代 8B 锚点全 untied |
| softmax 瓶颈 | $\mathrm{rank}(\log P) \le d + 2$，式 (11)(12) | MoS 抬 rank，式 (13)；tying 与其共用 rank 预算 |
| BLT | 熵驱动动态 patch | tokenizer-free 对照；8B 参数 / 4T bytes 的 FLOP 对齐 scaling |

## 练习

1.（计算）对语料 {banana ×4, bandana ×2, ananas ×3} 手算 BPE 的前三轮 merge（字符基元，词频加权）：列出每轮的相邻对频次表、选中的对与语料总长。再用学到的 merge 编码未见词 bananas，给出切分结果。（提示：第一轮 (a,n) 以 18 次居首。）
2.（推导）Unigram EM 的 M 步：写出辅助函数 $Q(\theta, \theta') = \sum_x \mathbb{E}_{q(z \mid x; \theta')}[\log p(z; \theta)]$ ，在约束 $\sum_v p(v) = 1$ 下用 Lagrange 乘子法推出闭式更新 $p(v) = c(v) / \sum_{v'} c(v')$ 。
3.（计算）Qwen3-8B（$V$ = 151936，$d$ = 4096，untied）：(a) 计算 $2Vd$ 参数；层参数按 2.1 的方法以 $11.5 d^{2} \times 36$ 估算（GQA 使 attention 为 $2.5 d^{2}$ 、SwiGLU 且 $d_{\mathrm{ff}} = 3d$ 使 FFN 为 $9 d^{2}$ ），总参数约多少、词表矩阵占几成？(b) 若改为 tied，省多少参数？(c) 输出头每 token 前向 FLOPs（$2Vd$ ）占「层部分 $2N$ + 输出头」的几成？
4.（推导 + 论证）(a) 从式 (10) 写出第 $v$ 行梯度的完整表达式，并指出两路梯度各自为零的条件；(b) 无 bias 且 tied（$W_U = E$ ）时，说明「输入映射与输出映射共享同一 rank ≤ $d$ 预算」的确切含义；(c) 解释为什么 MoS 的 $\log P$ 不受式 (12) 约束（提示：$\log$ 无法穿过混合求和）。

## 参考

- [Sennrich et al., 2016](../references.md)：Neural Machine Translation of Rare Words with Subword Units，arXiv:1508.07909，[A]（key: bpe）——BPE 训练算法与压缩视角出处
- [Kudo, 2018](../references.md)：Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates，arXiv:1804.10959，[A]（key: unigram）——Unigram 目标、EM 与剪枝出处
- [Wu et al., 2016](../references.md)：Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation，arXiv:1609.08144，[B]（key: wordpiece）——WordPiece 似然准则出处
- [Press & Wolf, 2017](../references.md)：Using the Output Embedding to Improve Language Models，arXiv:1608.05859，[B]（key: weight-tying）
- [Yang et al., 2018](../references.md)：Breaking the Softmax Bottleneck: A High-Rank RNN Language Model，arXiv:1711.03953，[A]（key: softmax-bottleneck）
- [Pagnoni et al., 2024](../references.md)：Byte Latent Transformer: Patches Scale Better Than Tokens，arXiv:2412.09871，[A]（key: blt）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——128K 词表与 3.17 → 3.94 chars/token 压缩对照的出处
- [Gemma Team, 2024](../references.md)：Gemma 2: Improving Open Language Models at a Practical Size，arXiv:2408.00118，[A]（key: gemma2）——256128 词表的论文口径
- [Gemma Team, 2025](../references.md)：Gemma 3 Technical Report，arXiv:2503.19786，[A]（key: gemma3）——262144 词表的论文口径
- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——tied 配方下 embedding 乘 $\sqrt{d}$ 的出处（§3.4）
- [Radford et al., 2019](../references.md)：Language Models are Unsupervised Multitask Learners + HF 权重，[W]（key: gpt2）——50257 分解与 initializer_range = 0.02
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [Qwen, Qwen3-8B config.json 与 tokenizer 文件](../references.md)：[A]（key: qwen3-8b-config）
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——tie_word_embeddings = false 的对照锚点
