# 3.1 Objective 与 Mask 的统一形式化（A Unified Formalization of Objectives and Masks）

> 本章把 1.1 留下的 $(x', M, \ell)$ 接口补全为完整的 objective 设计空间：objective = 腐蚀方案 × attention mask × loss mask，五种目标（CLM / MLM / prefix-LM 与 GLM blank infilling / T5 span corruption / UL2）逐个落成公式与 mask 实例，「能否自回归生成」收进一条可判定条件，perplexity 跨 objective 的比较论证为无意义；三个官方 masking 实现对照走读 | 前置：1.1（$(x', M, \ell)$ 三元组、CE 与 PPL）、2.4（mask 加性形式、可生成判定式） | 锚点：google-research/bert（`create_pretraining_data.py`）[A]、THUDM/GLM [A]、T5 官方 repo google-research/text-to-text-transfer-transformer [A]、ModernBERT [A]；UL2（无官方开源实现，公式按论文拆）[B]

## 1. 统一框架：腐蚀 × mask × loss mask

1.1 式 (8) 把训练目标写成 $L(\theta) = - \sum_t \ell_t \log q_\theta(x_t | x'; M)$ ，但把随机性的来源留在了记号之外： $x'$ 从哪来、 $\ell$ 由谁定。补上这一环，objective 的设计空间就完整参数化了——一个 objective 由三个对象完全确定：

- **腐蚀方案（corruption scheme）** $C$ ：随机映射 $x' = C(x; \omega)$ ， $\omega$ 为全部随机决策（选中哪些位置、替换成什么、如何重排拼接）；
- **attention mask** $M(\omega) \in \{0, -\infty\}^{T \times T}$ ：可见性（2.4 §1 的加性语义），允许依赖同一次抽样；
- **loss mask** $\ell(\omega) \in \{0, 1\}^{T}$ ：哪些位置计入损失，由 $C$ 的抽样结果决定。

训练目标是对数据与腐蚀随机性的双期望：

$$
L(\theta) = \mathbb{E}_{x \sim p_{\mathrm{data}},\; \omega \sim C}\left[ - \sum_{t} \ell_t(\omega) \log q_\theta\big(x_t \mid x'(\omega);\, M(\omega)\big) \right] \qquad (1)
$$

**符号**： $x \sim p_{\mathrm{data}}$ ：原始语料序列； $\omega$ ：腐蚀方案的内部随机性； $x'(\omega)$ ：本次抽样的实际输入； $\ell_t(\omega)$ ：loss mask，1 = 该位置计入损失； $M(\omega)$ ：attention mask；求和遍历输入序列位置。

encoder-decoder 形态（T5 / UL2）是同一对象的二元组写法：腐蚀序列 $x'$ 进 encoder（ $M_{\mathrm{enc}} \equiv 0$ ），target 序列 $y$ 在 decoder 侧因果生成，损失 $- \sum_j \log q_\theta(y_j | y_{<j}, x')$ ——把它看成「loss mask 落在拼接序列后半段」即与式 (1) 同构；GLM 正是用单序列拼接实现同一件事（§4），两种写法只是 $M$ 的布局差异。

1.1 §7 已给出 CLM / MLM / prefix-LM 三行取值，本章把行数补到五，并给每行公式、mask 实例与官方实现。同一条 8 token 序列在五种取值下的形态（图内简写 [M] = [MASK]、[s] = span 起始符、<X>/<Y>/<Z> = sentinel；为图示把 span 画得偏大，BERT 与 T5 的实际腐蚀率均为 15%，见 §3 / §5）：

```
原序列 x :  the  cat  sat  on  the  mat  and  slept        (T = 8)

CLM    x' = 原样                                           M: 下三角
       ℓ  = 1   1   1   1   1   1   1   1

MLM    x' = the [M] sat  on  the  mat  and [M]             M ≡ 0
       ℓ  = 0   1   0   0   0   0   0   1       (仅选中位；每位独立掷 80/10/10)

prefix x' = 原样               (前缀长 4)                   M: 前缀块双向 + 后缀下三角
       ℓ  = 0   0   0   0   1   1   1   1

GLM    x' = [the [M] on the mat [M] slept] ‖ [s] cat sat [s] and
             └──── Part A，双向 ────┘   └── Part B（乱序 span）──┘
       ℓ  = 只落 Part B 的内容 token（cat sat and）

T5     enc x' = the <X> on the mat <Y> slept               M_enc ≡ 0
       dec y  = <X> cat sat <Y> and <Z>                    M_dec 因果 + cross-attn
       ℓ  = 只落 dec 序列
```

五种取值的逐行判定与汇总表在 §7。

## 2. CLM：零腐蚀的参照行

CLM 是框架的零点： $C$ 取恒等（ $x' = x$ ）， $M$ 取下三角（2.4 式 (3)）， $\ell$ 全 1，式 (1) 退化为 1.1 式 (4)：

$$
L_{\mathrm{CLM}} = - \frac{1}{T} \sum_{t=1}^{T} \log q_\theta(x_t \mid x_{<t}) \qquad (2)
$$

**符号**： $T$ ：序列长度； $q_\theta(x_t \mid x_{<t})$ ：以真实前缀为条件的下一 token 分布（1.1 式 (1)–(3) 的记号）。

GPT 家族（[Radford et al., 2019](../references.md)，[W]）的全部性质已由该取值固定：一条序列一次前向产出 $T$ 个分类问题（1.1 §4 的 teacher forcing），生成语义是精确的祖先采样（1.1 §8）。它在下文的作用是对照列——每个非 CLM 目标都按「改了 $(C, M, \ell)$ 的哪几项、换来什么、付出什么」来定位。

## 3. MLM：15% 选中与 80/10/10

### 3.1 定义

BERT 的 masked LM（[Devlin et al., 2018](../references.md)，arXiv:1810.04805，[A]）：逐序列、逐位置以概率 $r$ = 0.15 独立选中，得选中集 $S$ （ $\mathbb{E}|S| = 0.15\,T$ ）；选中位置经三支腐蚀通道，未选中位原样进入 $x'$ ：

$$
x'_t = \begin{cases} \text{[MASK]}, & \text{w.p. } 0.8 \\ u, \quad u \sim \mathrm{Unif}(V), & \text{w.p. } 0.1 \\ x_t, & \text{w.p. } 0.1 \end{cases} \qquad t \in S \qquad (3)
$$

（w.p. = with probability，以概率。） $M \equiv 0$ （双向全可见）， $\ell_t = 1 \Longleftrightarrow t \in S$ ：

$$
L_{\mathrm{MLM}} = - \frac{1}{|S|} \sum_{t \in S} \log q_\theta(x_t \mid x') \qquad (4)
$$

**符号**： $S$ ：被选中（被问）位置集； $u$ ：词表上均匀抽出的替换 token； $x'$ ：腐蚀后的输入序列。BERT 另有 NSP 目标，其公式与失败机制归 3.2；本章只拆 MLM。

### 3.2 三支混合的损失分解与地板

式 (3) 的三支通道把期望损失拆成三个不同估计问题的混合：

$$
\mathbb{E}\, L_{\mathrm{MLM}} = 0.8\, \bar{L}_{\mathrm{mask}} + 0.1\, \bar{L}_{\mathrm{rand}} + 0.1\, \bar{L}_{\mathrm{keep}} \qquad (5)
$$

**符号**： $\bar{L}_b$ ：分支 $b$ 下选中位置的期望 NLL（对数据与分支内随机性取期望）。

各分支的 Bayes 地板（最优模型也压不下去的残差）：

- **mask 支**： [MASK] 不携带 $x_t$ 的信息，最优预测是上下文条件分布，地板 $H(x_t \mid \mathrm{ctx})$ ；
- **rand 支**： $u$ 与 $x_t$ 独立，同样只能靠上下文，地板同为 $H(x_t \mid \mathrm{ctx})$ ；
- **keep 支**： $x'_t = x_t$ 即答案本身，地板 0。

混合地板 = $0.9\, H(x_t \mid \mathrm{ctx})$ ——这是**分支可知口径的下界**： [MASK] 支的分支身份可从输入确知，而 keep / rand 两支从显示 token 上不可区分（§3.3 的后验分析），真实地板略高于 $0.9\, H$ ；作为下界它已足够支撑后面的比较论证。结论：一个已达最优的模型，其 MLM 损失读数天然比「纯 mask」目标低约 10%。这个偏移与模型质量无关，是比较任何 MLM 系损失数字之前要先剥掉的常数（§8 回收）。

### 3.3 80/10/10 的分布 mismatch：定量形式

为什么不是 100% [MASK]，论文给出的理由是 [MASK] 在下游 fine-tuning 中从不出现（[Devlin et al., 2018](../references.md)，[A]）。写成数：预训练输入流中 [MASK] 的占位率是 $r \times 0.8 = 12\%$ ，下游是 0——输入分布漂移。80/10/10 用两支「真实 token」通道把查询接口铺回真实 token 上，代价同样可以精确算出。在选中且显示真实 token $v$ 的位置（占全部位置的 $r \times 0.2 = 3\%$ ），最优预测给 $v$ 本身放置的概率质量为

$$
P(y = v \mid x'_t = v,\; t \in S) = \frac{0.1\, p(v) \cdot 1 + 0.1\, V^{-1} p(v)}{0.1\, p(v) + 0.1\, V^{-1}} = \frac{p(v)\, (1 + V^{-1})}{p(v) + V^{-1}} \qquad (6)
$$

**推导**：产生「显示 $v$ 」的分支有两条——keep 支要求原 token 就是 $v$ （联合概率 $0.1\, p(v)$ ，此时 $y = v$ 必中）；rand 支以 $1/V$ 采中 $v$ （联合 $0.1 / V$ ，此时目标服从数据分布， $y = v$ 的概率为 $p(v)$ ）。分子是两条支路上 $y = v$ 的联合概率之和，分母是显示 $v$ 的总概率； $p(v)$ 为上下文边际化后的 token 边际频率。

代入 BERT 的 WordPiece 词表 $V$ = 30522（[Google, BERT 官方 repo](../references.md)，[A]）， $V^{-1} \approx 3.3 \times 10^{-5}$ ：频率为均匀水平 10 倍的 token 上 copy 质量已达 $10/11 \approx 0.91$ ，高频 token 趋于 1——模型在真实 token 位置学到的条件分布以「照抄」为主导；频率低于 $1/V$ 的极端长尾上 rand 支后验反超，模型把显示 token 当噪声、退回上下文条件分布。也就是说，keep / rand 两支把 3% 位置的监督信号污染成 copy 任务与「对显示 token 保持不信任」——这是消除 [MASK] 输入漂移的交换物，不是免费的。

### 3.4 信号密度

$\mathbb{E}|S| / T = 0.15$ ：同一条序列一次前向，MLM 只在 15% 的位置上产出监督，CLM 是 100%，监督密度差约 6.7 倍。ELECTRA 的 RTD 目标正是从这一格出发（判别器逐位置二分类、每个位置都有信号），公式与样本效率论证在 3.2 展开。

### 3.5 官方实现走读：create_pretraining_data.py

google-research/bert 的预训练数据脚本 `create_pretraining_data.py`（[Google, BERT 官方 repo](../references.md)，[A]）中，函数 `create_masked_lm_predictions` 的执行序：

1. 候选下标 = 全部非特殊 token 位置（ [CLS] / [SEP] 除外），洗牌；
2. `num_to_predict = min(max_predictions_per_seq, max(1, round(len(tokens) * masked_lm_prob)))` ——15% 在此落地， `max(1, ·)` 给短序列保底；
3. 按洗牌序取前 `num_to_predict` 个候选位，每位独立掷一次： $< 0.8$ → 替换为 `[MASK]` ； $< 0.9$ → 词表均匀随机词；否则保留原 token；
4. 整词掩码（whole-word masking）变体把候选按 WordPiece 词分组（ `##` 续片归入同组）：同组同选（整组进出选中集），但组内逐 token 独立掷 80/10/10——同一词的各续片可以落进不同分支。

教学用复刻（非优化实现；与官方的差异：官方用 `covered_indexes` 集合支持整词分组、超出配额即中断，逐位 80/10/10 的分布语义相同）：

```python
import random

def create_masked_lm_predictions(tokens, masked_lm_prob=0.15,
                                 max_predictions_per_seq=None,
                                 vocab_words=None, rng=None):
    """教学用，非优化实现：复刻 google-research/bert create_pretraining_data.py
    同名函数的语义（15% 选中、80/10/10 替换）。"""
    rng = rng or random.Random()
    cand = [i for i, tok in enumerate(tokens) if tok not in ("[CLS]", "[SEP]")]
    rng.shuffle(cand)
    num_to_predict = min(max_predictions_per_seq or len(tokens),
                         max(1, round(len(tokens) * masked_lm_prob)))
    out, loss_pos = list(tokens), []
    for i in cand[:num_to_predict]:
        p = rng.random()
        if p < 0.8:
            out[i] = "[MASK]"
        elif p < 0.9:
            out[i] = rng.choice(vocab_words)
        # 余下 10%：保留原 token（监督目标仍是原 token）
        loss_pos.append(i)
    return out, sorted(loss_pos)
```

## 4. prefix-LM 与 GLM：blank infilling + 2D 位置编码

### 4.1 prefix-LM：一块 2×2 的 mask

前缀长 $p$ 时 $M$ 是块矩阵：前缀区 $p \times p$ 全 0、后缀区下三角、跨区只有「后缀看前缀」（矩阵实例见 1.1 §7）。损失：

$$
L_{\mathrm{prefix}} = - \frac{1}{T - p} \sum_{t = p+1}^{T} \log q_\theta(x_t \mid x_{\le p},\, x_{p+1 : t-1}) \qquad (7)
$$

**符号**： $p$ ：前缀长度；损失只落后缀。

它坐在 CLM（ $p$ = 0）与 MLM（ $p$ = $T$ ）之间：前缀是零代价的双向上下文，后缀保持可生成性（2.4 §2 的判定在续写区成立）。这个块状取值本身没有新机制；GLM 把它一般化成「任意多条空白的自回归填充」，多出来的两块机制——span 化与二维位置——是本节的主角。

### 4.2 GLM 的腐蚀方案与目标函数

GLM（[Du et al., 2021](../references.md)，arXiv:2103.10360，[A]）的空白填充（blank infilling）：对原序列反复采 span——长度采自 Poisson 分布（原文取 $\lambda$ = 3）、起点均匀——直到覆盖 15% 的 token，得 span 集 $\{s_1, \ldots, s_m\}$ 。每条 span 在输入中塌缩成单个 [MASK] token，得腐蚀序列 $x^{c}$ （长 $T - \sum_i |s_i| + m$ ）。模型输入 = $x^{c}$ （Part A）后接乱序拼接的 span（Part B：每条 span 前置一个起始符 [s]，排列 $\pi$ 在 $\{1, \ldots, m\}$ 上均匀采样）。目标函数：

$$
L_{\mathrm{GLM}} = - \mathbb{E}_{\pi}\left[ \sum_{i=1}^{m} \sum_{j=1}^{|s_{\pi(i)}|} \log q_\theta\big(s_{\pi(i),\, j} \mid x^{c},\, s_{\pi(<i)},\, s_{\pi(i),\, <j}\big) \right] \qquad (8)
$$

**符号**： $s_{i,\, j}$ ：第 $i$ 条 span 的第 $j$ 个 token； $s_{\pi(<i)}$ ：排列 $\pi$ 下先于它的全部 span； $s_{\pi(i),\, <j}$ ：本 span 内前 $j - 1$ 个 token；损失按 span 总 token 数归一。

结构读法（对应 $M(\omega)$ 的分块）：Part A 内部双向（A-A 块全 0）；Part B 的每个位置能看见全部 Part A、排列序上先于本 span 的其他 span、以及本 span 内自己的左侧（span 内因果），未来 span 不可见； $\ell$ 只落 Part B。对 $\pi$ 取期望使「先填哪个空」不构成偏置；mask 随每次抽样变化，是 $M(\omega)$ 依赖 $\omega$ 的实例。原文在同一机制上还定义了更长跨度的变体（整句级 span；文档级单 span，长度取原文的 50%–100%），把填空连续推向长文生成（同篇，[A]）；该文档级长 blank 变体在 GLM-130B 中记为 [gMASK] token（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]）。

### 4.3 2D 位置编码：两维寻址的必要性

blank infilling 对位置系统提出了 CLM / MLM 都没有的需求：预测 $s_{i,\, j}$ 时模型既要知道「在填哪个空」，又要知道「空内第几个」。GLM 的答案：每个 token 带两个位置 id $(p^{(1)}_t,\, p^{(2)}_t)$ ：

$$
p^{(1)}_t = \begin{cases} \mathrm{idx}(t), & t \in A \\ k_i, & t \in s_i \cup \{[s]_i\} \end{cases} \qquad (9)
$$

$$
p^{(2)}_t = \begin{cases} 0, & t \in A \cup \{[s]\} \\ j, & t = s_{i,\, j} \end{cases} \qquad (10)
$$

**符号**： $A$ ：Part A 的位置集； $\mathrm{idx}(t)$ ：Part A 内的下标； $k_i$ ：span $s_i$ 塌缩出的 [MASK] 在 $x^{c}$ 中的位置； $s_{i,\, j}$ ：第 $i$ 条 span 的第 $j$ 个 token（式 (8) 的记号）； $[s]_i$ ：第 $i$ 条 span 的起始符。读法：Part A 的成员以自身在 $x^{c}$ 中的下标为 $p^{(1)}$ 、 $p^{(2)}$ 取 0；Part B 的成员一律以所填空白的地址为 $p^{(1)}$ 、以空内偏移为 $p^{(2)}$ （起始符偏移 0）。

注入为 stream 加性（2.5 §1 的注入点分类）：两张可学绝对位置表（2.5 §2.3 的形态），与 token embedding 三者相加：

$$
e_t = E_{x_t} + P^{(1)}_{p^{(1)}_t} + P^{(2)}_{p^{(2)}_t}, \qquad P^{(1)} \in \mathbb{R}^{T_{\max} \times d},\quad P^{(2)} \in \mathbb{R}^{L_{\max} \times d} \qquad (11)
$$

**符号**： $E$ ：token embedding 表； $P^{(1)}, P^{(2)}$ ：两张 learned absolute 位置表； $L_{\max}$ ：span 内最大偏移； $d$ ：stream 宽度。

逐维看必要性：缺 $p^{(2)}$ ，同一 span 的所有 token 共享一个地址，空内顺序无从表达，退化为无序填空；缺 $p^{(1)}$ ，模型知道「第 $j$ 个」却不知道在填哪个空，多条空白的内容无法与各自的地址对齐。Part A 的 $p^{(2)}$ = 0 还兼任类型位（「我是上下文，不是填充」）。生成时因此自然落定：在 [MASK] 位置（ $p^{(1)}$ = $k_i$ ）起逐 token 生成，第 $j$ 个 token 带 $(k_i,\, j)$ ，直到该 span 的结束符。一个完整实例（0-indexed）：

```
原序列 :  x1  x2  x3  x4  x5  x6        spans: [x2 x3]（起点 1）、[x5]（起点 4）
Part A :  x1 [M]  x4 [M]  x6            位置 k = 0..4；两个 [M] 在 k = 1 与 3
Part B :                 [s]  x2  x3  [s]  x5     （取 π = 恒等）

token  :  x1  [M]  x4  [M]  x6  |  [s]  x2  x3  [s]  x5
p1     :   0   1   2   3    4   |   1   1   1   3   3
p2     :   0   0   0   0    0   |   0   1   2   0   1
ℓ      :   0   0   0   0    0   |   0   1   1   0   1
```

### 4.4 prefix-LM 作为特例

取 $m$ = 1、span = 后缀 $(p{+}1 .. T)$ 、 $\pi$ 为恒等：Part A = 前缀（双向），Part B = 后缀（因果， $p^{(2)}$ 从 1 起递增即后缀内相对位置），式 (8) 逐符号退化为式 (7)。即 GLM = prefix-LM × 多空白 × 乱序期望，2D 位置编码是这个一般化引入的配件——这是「从 objective 反向决定架构件」的实例，GLM 谱系的后续形态在 5.4 展开。

### 4.5 官方实现走读：THUDM/GLM

GLM 的官方实现在 THUDM/GLM（[THUDM, GLM 官方 repo](../references.md)，[A]；zai-org/GLM 不是 2021 论文代码，引用一律以 THUDM 为准）。数据管线按 §4.2 采 span、构造 Part A / Part B 与二维位置 id；attention mask 按 §4.2 的分块规则从 Part 归属与 span 归属派生。教学用复刻（非优化实现）：

```python
def build_glm_example(tokens, spans, rng, MASK="[MASK]", SOP="[s]"):
    """教学用，非优化实现：GLM blank infilling 的数据构造（对应 THUDM/GLM
    预训练数据管线的语义）。spans: [(start, length)]，互不重叠、按原序给出。"""
    x_c, blank_at, cur = [], [], 0            # Part A：每条 span 塌缩为单个 [MASK]
    for a, l in spans:
        x_c += tokens[cur:a]
        blank_at.append(len(x_c)); x_c.append(MASK)
        cur = a + l
    x_c += tokens[cur:]

    order = list(range(len(spans)))
    rng.shuffle(order)                        # π：span 的均匀随机排列
    ids  = list(x_c)
    pos1 = list(range(len(x_c)))              # Part A：p1 = 自身位置，p2 = 0
    pos2 = [0] * len(ids)
    loss = [0] * len(ids)
    for i in order:                           # Part B：乱序拼接
        a, l = spans[i]; k = blank_at[i]
        for j, tok in enumerate([SOP] + tokens[a:a + l]):
            ids.append(tok)
            pos1.append(k)                    # p1 = 该空白 [MASK] 的地址
            pos2.append(j)                    # p2 = span 内偏移（起始符为 0）
            loss.append(1 if j >= 1 else 0)   # ℓ 只落 span 内容
    return ids, pos1, pos2, loss
```

与官方实现的差异：官方管线还处理长文档变体（文档级长 blank，GLM-130B 起记为 [gMASK] token）、多任务混合与批次内 padding；二维位置 id 与分块 mask 的构造语义相同。

## 5. span corruption：T5 的 sentinel 方案

### 5.1 sentinel 替换与 target 构造

T5（[Raffel et al., 2019](../references.md)，arXiv:1910.10683，[A]）的片段腐蚀（span corruption）把「被掩内容」从输入里物理移除：选中若干连续 span，每条在 encoder 输入中替换为一个哨兵 token（sentinel token：词表中专留的 extra_id 位——论文口径词表 32000；发布的 checkpoint / config 口径为 32000 + 100 个 sentinel 保留位 = 32100），被移除的内容按序移入 decoder 的 target，同样以 sentinel 分隔、并以一个收尾 sentinel 结束：

```
原序列    :  the  cat  sat  on  the  mat  and  slept
enc 输入  :  the  <X>  on  the  mat  <Y>  slept       ← [cat sat] → <X>，[and] → <Y>
dec target:  <X>  cat  sat  <Y>  and  <Z>             ← <Z> = 收尾 sentinel
```

$$
L_{\mathrm{T5}} = - \frac{1}{|y|} \sum_{j=1}^{|y|} \log q_\theta(y_j \mid y_{<j},\, x') \qquad (12)
$$

**符号**： $x'$ ：sentinel 替换后的 encoder 输入（ $M_{\mathrm{enc}} \equiv 0$ ）； $y$ ：target 序列，decoder 对其因果、并 cross-attend encoder 输出（enc-dec 结构本身在 3.2 展开）。sentinel 使 target 成为「哪个空 → 填什么」的自界定记录；与 GLM 的单序列拼接（§4）是同一数学对象的两种布局。

### 5.2 span 长度分布

官方实现（T5 repo google-research/text-to-text-transfer-transformer 的 `t5/data/preprocessors.py`，函数 `span_corruption`，默认 `noise_density=0.15`、`mean_noise_span_length=3.0`；论文 [Raffel et al., 2019](../references.md)，[A]）先把两个计数定死：

$$
m = \mathrm{clip}(\mathrm{round}(r\, T),\, 1,\, T - 1), \qquad n_s = \max\big(1,\, \mathrm{round}(m / \mu)\big) \qquad (13)
$$

**符号**： $m$ ：腐蚀 token 总数； $n_s$ ：span 条数； $r$ = 0.15：破坏率； $\mu$ = 3：目标 span 均长。

然后把 $m$ 个腐蚀 token 与 $T - m$ 个保留 token 各自均匀随机切分成 $n_s$ 段（uniform random composition：把 $n$ 个 token 切成 $n_s$ 段，即在 $n - 1$ 个可切点中均匀无放回选 $n_s - 1$ 个），交替拼接。span 长度因此不是逐位独立抽签，而是定总量的随机分割。单条 span 长度 $\ell$ 的边缘分布： $m$ 切成 $n_s$ 段的均匀切分共 $\binom{m - 1}{n_s - 1}$ 种， $\ell = k$ 占

$$
P(\ell = k) = \frac{\binom{m - k - 1}{n_s - 2}}{\binom{m - 1}{n_s - 1}}, \qquad \mathbb{E}\, \ell = \frac{m}{n_s} \approx \mu \qquad (14)
$$

（均值由 $n_s$ 段的对称性直接读出。）大 $m$ 极限看相邻项之比： $P(\ell = k) / P(\ell = k - 1) = (m - k - n_s + 2) / (m - k) \to 1 - n_s / m = 1 - 1 / \mu$ ——即 $\ell$ 渐近几何分布 Geometric( $1 / \mu$ )，「平均 span 长 3」的实际分布形态就是它。

代入 $T$ = 512 的账： $m$ = 77、 $n_s$ = 26；enc 输入长 = $512 - 77 + 26$ = 461，target 长 = $77 + 26 + 1$ = 104。两个读数：(i) target 只有原序列的约 20%，decoder 侧计算量同比例压缩——原文选择 span 方案的账面理由之一；(ii) loss 支撑 104 个位置，对照 CLM 同序列的 512——这是 §8 分母差异的实例。

### 5.3 代码走读

教学用复刻（非优化实现；与官方的差异：官方以位图构造噪声段、sentinel 从词表尾部倒序分配 id，并对超长序列做长度预算；切分与替换语义相同）：

```python
import numpy as np

def random_composition(n, k, rng):
    """n 个 token 均匀随机切成 k 段：在 n-1 个可切点中无放回选 k-1 个。"""
    if k == 1:
        return [n]
    cuts = np.sort(rng.choice(n - 1, size=k - 1, replace=False)) + 1
    return np.diff(np.concatenate([[0], cuts, [n]])).tolist()

def span_corruption(tokens, noise_density=0.15, mean_noise_span_length=3.0, rng=None):
    """教学用，非优化实现：对应 t5/data/preprocessors.py 的 span_corruption /
    random_spans_noise_mask 语义。"""
    rng = rng or np.random.default_rng()
    T = len(tokens)
    m = min(max(int(round(noise_density * T)), 1), T - 1)
    n_spans = max(int(round(m / mean_noise_span_length)), 1)
    noise_lens = random_composition(m, n_spans, rng)
    keep_lens  = random_composition(T - m, n_spans, rng)
    inputs, targets, pos = [], [], 0
    for i in range(n_spans):
        inputs.extend(tokens[pos:pos + keep_lens[i]]); pos += keep_lens[i]
        s = f"<extra_id_{i}>"
        inputs.append(s)
        targets.append(s)
        targets.extend(tokens[pos:pos + noise_lens[i]]); pos += noise_lens[i]
    inputs.extend(tokens[pos:])
    targets.append(f"<extra_id_{n_spans}>")           # 收尾 sentinel
    return inputs, targets
```

## 6. UL2：mixture-of-denoisers

UL2（[Tay et al., 2022](../references.md)，arXiv:2205.05131，[B]——无官方开源实现，本节公式按论文拆，勿以社区复现为准）把 §1 的设计空间做成可采样的混合分布：每个去噪器（denoiser） $k$ 由参数 $(\mu_k, r_k, n_k)$ （span 均长、破坏率、span 数）与顺序模式定义，训练目标是对 denoiser 的加权和：

$$
L_{\mathrm{UL2}} = \sum_{k} w_k\; \mathbb{E}_{x,\, \omega \sim C_k}\left[ - \sum_{j} \log q_\theta(y_j \mid y_{<j},\, x';\, m_k) \right] \qquad (15)
$$

**符号**： $C_k$ ：第 $k$ 个 denoiser 的腐蚀分布； $m_k$ ：模式切换符（mode-switch token）； $w_k$ ：混合权重。

三个模式族： **R-denoiser**（regular）= T5 式短 span（ $\mu \approx 3$ 、 $r \approx 15\%$ ）； **X-denoiser**（extreme）= 长 span 或高破坏率，span 均长抬到数十 token、破坏率抬到约 50% 的量级，逼模型做长程再生； **S-denoiser**（sequential）= 不按 span 而按切点，保留前缀、target 为后缀全文——即 prefix-LM / CLM 型目标。机制关键是 mode-switch token（ [R] / [S] / [X] ）拼在输入之前：「当前是哪个 denoiser」被写成上下文的一部分，同一组参数在不同模式下被显式条件化，而不是隐式承担任务间干扰。UL2-20B 的预训练用了 7 个 denoiser 的混合（同篇）。对 §1 框架的意义：CLM 型与 MLM 型目标可以共居一个模型，代价只是输入侧多一个模式位。

## 7. objective ↔ 能力：可生成性的数学判定与总表

「能否自回归生成」不是架构印象，是一条可判定条件。设 objective 训出的条件分布族为 $\{q_t(\cdot \mid c_t)\}$ （ $c_t$ 为被问位置的条件集），它支持自回归生成当且仅当存在生成顺序 $\pi$ ，使第 $\tau$ 步的条件集满足

$$
c_{\pi(\tau)} \subseteq G_{\tau} \cup I, \qquad G_{\tau} = \{\pi(1), \ldots, \pi(\tau - 1)\} \qquad (16)
$$

**符号**： $G_\tau$ ：已生成位置集合； $I$ ：生成前完整已知的输入。这是 2.4 式 (4)（ $V(i) \subseteq \{1, \ldots, i\}$ ）在 objective 层面的重述。

逐族判定：

- **CLM**： $c_t = x_{<t} = G_t$ ，式 (16) 取等。
- **MLM**： $c_t = x'$ 为整条腐蚀序列，含未来位置的真值（腐蚀只动 15% 的位置——其中 12% 替换为 [MASK]，其余位置真值全露），任何只含过去的 $G_\tau$ 都不满足式 (16)。逐位留一重掩的迭代采样也绕不过：第 $\tau$ 步实际计算的是 $q(x_\tau \mid x_{<\tau},\, \text{[MASK]}_{> \tau})$ ，与训练条件（固定 12% [MASK] 率、其余为真值）不是同一条件分布，且 [MASK] 率随迭代进度持续漂移。更根本地，伪似然（pseudo-likelihood）式连乘 $\prod_t q(x_t \mid x_{\setminus t})$ 不满足链式法则对条件集的嵌套要求（1.1 式 (3)：第 $t$ 个因子只能以前 $t - 1$ 个已生成为条件），它不是任何联合分布的分解——MLM 产出的是一组相容性无保证的局部条件，不是一个可采样的联合。「encoder 形态为何不能直接生成」的完整证明在 3.2。
- **prefix-LM**： $I$ = 前缀，续写区满足式 (16)。
- **GLM**： $I$ = Part A（ $x^{c}$ 生成前已知），span 逐条生成、span 内因果，条件集 = $x^{c} \cup$ 已生成 span $\cup$ 本 span 前缀，满足式 (16)。
- **T5 / UL2**： $I$ = encoder 输入，decoder 因果，满足式 (16)。

总表（ $\mathbb{E}|\ell|$ 列按 $T$ = 512、15% 腐蚀口径；「能」= 支持自回归生成）：

| objective | $x'$ | $M$ | $\ell$ 支撑 | $\mathbb{E}\vert\ell\vert$ | 可 AR 生成 | 代表 |
| --- | --- | --- | --- | --- | --- | --- |
| CLM | $x$ 原样 | 下三角 | 全位置 | 512 | 能 | GPT 家族 |
| MLM | 15% 位经 80/10/10 | 全 0 | 选中位 | 77 | 否 | BERT、ModernBERT |
| prefix-LM | $x$ 原样 | 前缀双向 + 后缀因果 | 后缀 | $T - p$ | 后缀能 | GLM 特例形态 |
| GLM infilling | $x^{c}$ + 乱序 span | A 双向 + B span 内因果 | span token | 77 | 能（逐 span） | GLM |
| T5 span corruption | sentinel 序列 | enc 全 0 + dec 因果 | target | 104 | 能（对 $y \mid x'$ ） | T5 |
| UL2 | 随 denoiser 模式 | 随模式 | 随模式 | 随模式 | 随模式 | UL2 |

表内「代表」列的古今两格：MLM 行里 ModernBERT（[Warner et al., 2024](../references.md)，arXiv:2412.13663，[A]）把同一目标带进现代配方（RoPE、长上下文、local/global attention 交替），其架构走读在 3.2；GLM 行的谱系延续在 5.4。这张表是后续章节的接口：3.2 展开 MLM 行的 encoder 形态与不可生成证明，3.3 展开 CLM 行的 in-context 性质，4.x 的效率讨论全部在这个 $(C, M, \ell)$ 空间内定位。

## 8. Perplexity 跨 objective 不可比

MLM 系论文不报 perplexity、CLM 的 PPL 也搬不到 MLM 上——这不是报告惯例，是三个独立的数学障碍。1.1 §6 已处理跨词表与跨 tokenizer 两个陷阱，本节加第三个。

**障碍一：支撑集不同。** CLM 的 per-token CE 平均在全 $T$ 个位置；MLM 只在 $|S| \approx 0.15\,T$ 个选中位；T5 在 target 的 $m + n_s + 1$ 个位置（ $T$ = 512 例：104）。三个平均对三个不同的位置集、三类不同的被问方式取值——CLM 问每一个位置（含高频功能词位），MLM 把全部监督压在腐蚀器选中的 15% 上。

**障碍二：条件不同 ⇒ 地板不同。** 对 mask / rand 支（ $x'$ 不含 $x_t$ 自身的信息），涉及三个条件熵，其间两条不等式是定理：

$$
H(x_t \mid x_{\setminus t}) \;\le\; H(x_t \mid x_{<t}), \qquad H(x_t \mid x_{\setminus t}) \;\le\; H(x_t \mid x') \;\le\; H(x_t) \qquad (17)
$$

**符号**： $H(\cdot \mid \cdot)$ ：条件熵； $x_{\setminus t}$ ：除位置 $t$ 之外的完整真值序列； $x'$ ：腐蚀后输入； $g$ ：腐蚀映射（含其内部随机性 $\omega$ ）。

第一条： $x_{\setminus t}$ 比 $x_{<t}$ 多出未来位置，条件化更多变量不增熵——全量双向地板不高于 CLM 地板。第二条：左半是数据处理不等式的条件形式（ $x' = g(x_{\setminus t}, \omega)$ 是腐蚀函数， $H(x_t \mid g(x_{\setminus t}, \omega)) \ge H(x_t \mid x_{\setminus t})$ ），右半是条件化不增熵。MLM 的地板落在中间区间；腐蚀只动 15% 的位置， $x'$ 保留了包括未来在内的 85% 真值，其地板一般远低于 CLM 的 $H(x_t \mid x_{<t})$ ——两个目标的地板之差是任务难度差，与模型质量无关；keep 支再把 MLM 侧地板压到约 0.9 倍（§3.2 的分支可知下界口径）。两边没有可对齐的零点。

**障碍三：pseudo-PPL 不是 perplexity。** MLM 评估实践里的「perplexity」通常是伪困惑度（pseudo-perplexity）：逐位置留一重掩、其余位置给真值，

$$
\text{pseudo-PPL} = \exp\left( - \frac{1}{T} \sum_{t=1}^{T} \log q_\theta(x_t \mid x_{\setminus t}) \right) \qquad (18)
$$

**符号**： $q_\theta(x_t \mid x_{\setminus t})$ ：留一重掩评估下的位置 $t$ 条件分布——位置 $t$ 单独换 [MASK]、其余位置给真值； $P(x_{1:T})$ ：序列的联合概率。

其连乘不满足链式嵌套（同 §7），不等于 $\exp\big( - T^{-1} \log P(x_{1:T}) \big)$ ——对任何联合分布都不成立（条件独立的退化情形除外）。它不具备 1.1 §6 的「等效均匀分支因子」语义，与 CLM 的 PPL 不是同一泛函的两个取值。

可比口径因此只剩三条：同 objective、同 tokenizer、同测试集内部比 PPL；跨 objective 比下游任务指标；需要绝对标度时用 bits per byte（1.1 §6）——而 bits/byte 要求全序列似然，MLM 族给不出，这是不可比性的另一种表述。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 统一框架 | 式 (1)： $C \times M(\omega) \times \ell(\omega)$ | objective = 腐蚀 × mask × loss mask；enc-dec 为其二元组写法 |
| CLM | 式 (2) | 零腐蚀参照行；监督密度 100% |
| MLM | 式 (3)(4)：15% 选中 + 80/10/10 | 混合地板 0.9·H（式 (5)，分支可知口径的下界）；copy 后验式 (6)；[MASK] 输入率 12% vs 下游 0；监督密度 15% |
| GLM | 式 (8)–(11)：blank infilling + 2D 位置 | span 塌缩成单 [MASK] + 乱序拼接； $(p^{(1)}, p^{(2)})$ = 空白地址 × 空内偏移；prefix-LM 为其特例 |
| T5 span corruption | 式 (12)–(14)：sentinel 方案 | 定总量均匀切分； $\ell$ 渐近 Geometric(1/μ)（μ = 3、r = 0.15）； $T$ = 512 时 target 仅 104 位 |
| UL2 | 式 (15)：mixture-of-denoisers | R / S / X 三族 + mode-switch token；[B] 无官方实现 |
| 可生成判定 | 式 (16) | $c_{\pi(\tau)} \subseteq G_\tau \cup I$ ；MLM 违反（pseudo-likelihood 非联合） |
| PPL 不可比 | 障碍一/二/三 | 支撑集 / 地板 / pseudo-PPL 非似然；跨 objective 只能比下游或 bits per byte |

## 练习

1.（推导）完成式 (6) 的推导，并证明两条极限行为： $p(v) \gg V^{-1}$ 时 copy 质量 → 1， $p(v) \ll V^{-1}$ 时 → $p(v)$ 。代入 $V$ = 30522，对 $p(v) \in \{10^{-2}, 10^{-4}, 10^{-6}\}$ 各算一个后验值。再证式 (5) 的混合地板 = $0.9\, H(x_t \mid \mathrm{ctx})$ 。
2.（计算） $T$ = 2048、 $r$ = 0.15、 $\mu$ = 3 的 T5 账： $m$ 、 $n_s$ 、enc 输入长、target 长各是多少；用几何近似算 $P(\ell = 1)$ 与 $P(\ell \ge 10)$ ；把 target 的 loss 位置数与 CLM 同序列对照，给出监督密度之比。
3.（推导）取 $x$ = [x1 … x6]、spans = {[x2 x3], [x5]}、 $\pi$ = 恒等，写出 GLM 的完整输入序列、每个位置的 $(p^{(1)}, p^{(2)})$ 与 $\ell$ ，并写出 $M$ 的分块结构（哪些块全 0、哪些块下三角、哪些块全 $-\infty$ ）。再令 span 为单条后缀 [x5 x6]，验证式 (8) 逐符号退化为式 (7)。
4.（代码）实现 §3.5 与 §5.3 的两个教学函数，在 $10^{6}$ token 的玩具语料上验证：(a) BERT 输出中 [MASK] 占比 → 12%、被选中位占比 → 15%、keep 位占比 → 1.5%；(b) T5 的经验 span 均长 → 3，经验长度分布与 Geometric(1/3) 的直方图对照。

## 参考

- [Devlin et al., 2018](../references.md)：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，arXiv:1810.04805，[A]（key: bert）——MLM 目标、15% 与 80/10/10 的动机
- [Google, BERT 官方 repo](../references.md)：[A]（key: bert-repo）——`create_pretraining_data.py` / `create_masked_lm_predictions`
- [Du et al., 2021](../references.md)：GLM: General Language Model Pretraining with Autoregressive Blank Infilling，arXiv:2103.10360，[A]（key: glm）——blank infilling、2D 位置编码、Poisson span 长度
- [THUDM, GLM 官方 repo](../references.md)：[A]（key: glm-repo）——infilling 实现（zai-org/GLM 非 2021 论文代码）
- [Zeng et al., 2022](../references.md)：GLM-130B: An Open Bilingual Pre-trained Model，arXiv:2210.02414，[A]（key: glm-130b）——[gMASK] token 记法的出处
- [Raffel et al., 2019](../references.md)：Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer，arXiv:1910.10683，[A]（key: t5）——span corruption；实现侧 google-research/text-to-text-transfer-transformer 的 `t5/data/preprocessors.py::span_corruption`
- [Tay et al., 2022](../references.md)：UL2: Unifying Language Learning Paradigms，arXiv:2205.05131，[B]（key: ul2）——mixture-of-denoisers；无官方开源实现，公式按论文拆
- [Warner et al., 2024](../references.md)：Smarter, Better, Faster, Longer（ModernBERT），arXiv:2412.13663，[A]（key: modernbert）——MLM 目标的现代延续
- [Radford et al., 2019](../references.md)：Language Models are Unsupervised Multitask Learners，[W]（key: gpt2）——CLM 参照系
