# 1.1 语言建模的数学形式化（The Mathematical Formulation of Language Modeling）

> 本章把「语言模型」压成一个可计算的数学对象：token 序列上的概率分布、它的分解方式、训练目标与评价指标，以及采样生成与输出头表示能力的两个边界 | 前置：线性代数（矩阵乘法、rank）、概率论（条件分布、期望、KL 散度） | 锚点：Llama-3.1-8B config [A]、Qwen3-8B config [A]、GPT-2 技术报告 [W]；Kaplan et al. [B]、GPT-3 [B]、softmax bottleneck [A]

## 1. 样本空间：token 序列

Qwen3-8B 的 config.json 写着 vocab_size = 151936（[Qwen3-8B config](../references.md)，[A]）。这个整数是本章全部数学的基本单元数：模型处理的「文字」不是字符，而是这个集合里的下标。

**词表（vocabulary）** 是有限集合 $V = \lbrace0, 1, ..., V-1\rbrace$ ，其大小也记为 $V$ （集合与基数复用同一符号，按上下文区分）。**token** 是词表中的一个元素，工程上就是一个整数 id。**分词器（tokenizer）** 是从底层字符串到 token 序列的确定映射 $`\mathrm{tok}: \Sigma^{*} \to V^{T}`$ ，其中 $\Sigma$ 是字节或字符集，$`\Sigma^{*}`$ 是其上全部有限字符串，输出长度 $T$ 随输入文本变化。Qwen3-8B 的 $\mathrm{tok}$ 是 byte-level BPE（Qwen2Tokenizer 家族）：vocab.json 含 151643 个 merge 后的基础 token，added_tokens_decoder 再加 26 个特殊 token（id 151643–151668，含 `<|endoftext|>` 、`<|im_start|>` 、`<|im_end|>`），config 中的 151936 = 151643 + 26 再向上 padding 对齐（[Qwen3-8B config](../references.md)，[A]）。tokenizer 算法细节（BPE / Unigram / WordPiece）在 2.2 展开，本章只需要 $\mathrm{tok}$ 存在且确定。

三个真实词表大小，本章反复用作算例：

| 模型 | $V$ | 上下文窗口 $T_{\max}$ | 证据 |
| --- | --- | --- | --- |
| GPT-2 small | 50257 | 1024 | [Radford et al., 2019](../references.md)，[W] |
| Llama-3.1-8B | 128256 | 131072 | [Llama-3.1-8B config](../references.md)，[A] |
| Qwen3-8B | 151936 | 40960 | [Qwen3-8B config](../references.md)，[A] |

**符号**： $T_{\max}$ ：模型的上下文窗口（context window），即单次可处理的最大序列长度。

两个边界 token 需要显式登记，后面的公式要用：GPT-2 用同一个 id 50256（`<|endoftext|>`）充当序列开始与结束符；Qwen3-8B 的 bos_token_id = 151643（`<|endoftext|>`）、eos_token_id = 151645（`<|im_end|>`）。全书统一约定 $x_0$ = bos token：它是人为添加的第 0 个 token，作用是给「第一个 token 的分布」一个条件对象。 $x_0$ 只在 $t$ = 1 的边界条件里出现，不计入序列本身—— $x_{\lt t}$ 的定义（式 (3)）从 $x_1$ 起算。

**样本空间** $X$ 是所有长度 $T$ 满足 $1 \le T \le T_{\max}$ 的 token 序列的并集： $X$ 的每个元素是某个 $V^{T}$ 里的序列。一个语言模型就是 $X$ 上的一个概率分布 $P$ 。

**符号**： $X$ ：样本空间； $V^{T}$ ：长度为 $T$ 的 token 序列全体。

序列进入模型前还有一步：token id 本身只是下标，不含任何几何结构；模型第一层把它查表换成 $\mathbb{R}^{d}$ 里的 embedding 向量（embedding 矩阵与第 2 节输出头 $W$ 的关系、查表的梯度性质见 2.2）。本章在 id 层面建概率论，不涉及向量语义。

上下文窗口有两个字段容易混淆：Qwen3-8B 的 config.json 里 max_position_embeddings = 40960（原生训练窗口），而 tokenizer_config.json 里 model_max_length = 131072（YaRN 扩展后的服务上限）。两个数来自不同字段、含义不同，引用时别混（扩展机制见 4.6）。

## 2. 模型定义：从上下文到概率单纯形

一个 decoder-only 语言模型是一个带参函数

$$
f_\theta: V^{t-1} \to \Delta^{V-1}, \qquad \Delta^{V-1} = \lbrace q \in \mathbb{R}^{V} : q_i \ge 0, \sum_{i=1}^{V} q_i = 1 \rbrace \qquad (1)
$$

**符号**： $\theta$ ：全部可学习参数； $t$ ：当前位置； $\Delta^{V-1}$ ： $V$ 个类别上的概率单纯形（probability simplex），即所有分量非负、和为 1 的 $V$ 维向量的集合； $q$ ：单纯形中一点，即词表上的一个分布。

式 (1) 的信息量全在值域上：无论前面见过什么上下文，模型的输出永远是「下一个 token 取词表中每个值的概率」。工程实现分两步：主干网络把上下文压成一个向量，输出头（output head，也叫 unembedding）把它映射成分布：

$$
h = \mathrm{backbone}_\theta(x_{\lt t}) \in \mathbb{R}^{d}, \qquad z = W h + b \in \mathbb{R}^{V}, \qquad q_\theta(\cdot | x_{\lt t}) = \mathrm{softmax}(z), \qquad q_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)} \qquad (2)
$$

**符号**： $h$ ：上下文末尾位置的隐状态（hidden state），批量后 shape 为 $[B, T, d]$ ； $d$ ：隐藏维（config 的 hidden_size）； $z$ ：logits，未归一化的打分，shape $[B, T, V]$ ； $W \in \mathbb{R}^{V \times d}$ 、 $b \in \mathbb{R}^{V}$ ：输出头权重与偏置； $q_\theta(\cdot | x_{\lt t})$ ：给定上下文时下一 token 的条件分布。

锚点数值：Llama-3.1-8B 与 Qwen3-8B 都是 $d$ = 4096，且 tie_word_embeddings = false，即 $W$ 与输入 embedding 矩阵是相互独立的两份参数（[Llama-3.1-8B config](../references.md)、[Qwen3-8B config](../references.md)，均 [A]）。单算输出头：Llama-3.1-8B 的 $W$ 有 $128256 \times 4096 = 525336576$ 个参数（约 5.3 亿），Qwen3-8B 为 $151936 \times 4096 = 622329856$（约 6.2 亿）——输出头自己就是一个 0.5B 量级的矩阵，这就是第 9 节 rank 瓶颈的舞台。weight tying 的取舍见 2.2。

softmax 的一个性质后面会用到： $\exp(z_i) > 0$ 对任何有限 logits 成立，所以 $q_\theta$ 给词表每个 token 的概率都严格大于 0——模型的支持集（support）恒为整个 $V$ 。代价是数值上必须防溢出（实现里先减 $\max_i z_i$ 再取 $\exp$ ，等价性留作思考，2.3 与官方实现对照时再落实）；收益是 $q_\theta(v) = 0$ 导致的 $\log 0$ 在训练里不会出现。

## 3. 链式法则到自回归分解（完整推导）

式 (1) 只定义了「单步条件分布」， $X$ 上的完整分布 $P(x_{1:T})$ 还没着落。直接列表参数化需要 $V^{T} - 1$ 个自由参数——GPT-2 的规模下是 $50257^{1024}$ ，连写都写不下。概率论的链式法则（chain rule）给出精确分解，一步不跳地推：

条件概率的定义式为

$$
P(A, B) = P(A | B) \cdot P(B)
$$

**符号**： $A, B$ ：两个随机事件； $P(A, B)$ ：联合概率； $P(A | B)$ ： $B$ 发生时 $A$ 的条件概率。

取 $A = \lbrace x_T\rbrace$ 、 $B = \lbrace x_{1:T-1}\rbrace$ ，套定义式：

$$
P(x_{1:T}) = P(x_T | x_{1:T-1}) \cdot P(x_{1:T-1})
$$

右端第二个因子仍是联合分布，对它再套一次定义式（取 $A = \lbrace x_{T-1}\rbrace$ 、 $B = \lbrace x_{1:T-2}\rbrace$ ）：

$$
P(x_{1:T-1}) = P(x_{T-1} | x_{1:T-2}) \cdot P(x_{1:T-2})
$$

代回去：

$$
P(x_{1:T}) = P(x_T | x_{1:T-1}) \cdot P(x_{T-1} | x_{1:T-2}) \cdot P(x_{1:T-2})
$$

每套一次定义式，右端的联合分布长度减 1，直到剩下 $P(x_1)$ 。先看 $T$ = 3 的完整展开，确认模式：

$$
P(x_1, x_2, x_3) = P(x_3 | x_1, x_2) \cdot P(x_1, x_2) = P(x_3 | x_1, x_2) \cdot P(x_2 | x_1) \cdot P(x_1)
$$

对一般 $T$ 用归纳法。**归纳基**： $T$ = 1 时序列为单 token， $P(x_1)$ 写成条件形式 $P(x_1 | x_0)$ ——这是 bos 约定唯一进入公式的地方（ $x_{<1}$ 本身为空序列）。**归纳假设**：对长度 $k$ 成立，即 $P(x_{1:k}) = \prod_{t=1}^{k} P(x_t | x_{\lt t})$ 。**归纳步骤**：对长度 $k+1$ ，

$$
P(x_{1:k+1}) = P(x_{k+1} | x_{1:k}) \cdot P(x_{1:k}) = P(x_{k+1} | x_{1:k}) \cdot \prod_{t=1}^{k} P(x_t | x_{\lt t}) = \prod_{t=1}^{k+1} P(x_t | x_{\lt t})
$$

**注**：上面几行中间推导无新符号—— $A, B$ 沿用定义式注记， $k$ 为归纳变量（序列长度）， $x_{1:k}$ 表示前 $k$ 个 token 的联合。第一步是条件概率定义式（取 $A = \lbrace x_{k+1}\rbrace$ 、 $B = \lbrace x_{1:k}\rbrace$ ），第二步代入归纳假设，第三步把因子并入连乘。归纳成立，得**自回归分解（autoregressive factorization）**。严格性注记：定义式 $P(A, B) = P(A | B) P(B)$ 在 $P(B) = 0$ 时条件概率无定义；对语言数据这不是问题——实际语料里每个出现的前缀都有正概率，而模型侧由 softmax 的严格正性（第 2 节）保证 $q_\theta$ 处处有定义，归纳链条每一步都合法：

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{\lt t}), \qquad x_{\lt t} = (x_1, ..., x_{t-1}) \qquad (3)
$$

**符号**： $x_{1:T}$ ：完整序列； $x_{\lt t}$ ：位置 $t$ 之前的全部 token，即 $(x_1, ..., x_{t-1})$ ，共 $t-1$ 个元素，与式 (1) 的定义域 $V^{t-1}$ 一致。注意 $x_{\lt t}$ 不含 $x_0$ ：位置 $t$ 的完整条件是 $(x_0, x_{\lt t})$ ，因 $x_0$ 恒为 bos，本书统一简记为 $q_\theta(\cdot | x_{\lt t})$ 。

三个性质决定它为什么是所有 LLM 的工作方式：

1. **精确，无近似**。式 (3) 是恒等式，任何联合分布都满足，不是模型假设。
2. **顺序任意**。链式法则对位置的任意排列 $\pi$ 都成立（练习 1 证明）。选从左到右的因果序只有一个原因：生成时未来位置尚不存在，只有因果序的每个因子 $P(x_t | x_{\lt t})$ 在采样时刻是可用的。
3. **把存不下的表变成共享函数**。 $V^{T}$ 个表项被换成 $T$ 个条件分布，而每个条件分布都由同一个 $f_\theta$ 供给（式 (1)(2)）——指数爆炸被参数共享压掉。

## 4. 训练目标：NLL 与 cross-entropy 的同一性

训练数据 $D$ 是来自未知真实分布 $p_{\mathrm{data}}$ 的样本。极大似然估计（MLE）等价于最小化负对数似然（negative log-likelihood, NLL）。把式 (3) 代入并除以 token 数，得到 per-token 训练目标：

$$
\mathrm{NLL}(\theta) = - \frac{1}{T} \sum_{t=1}^{T} \log q_\theta(x_t | x_{\lt t}) \qquad (4)
$$

**符号**： $\mathrm{NLL}(\theta)$ ：平均每 token 负对数似然； $q_\theta(x_t | x_{\lt t})$ ：模型在真实前缀下给真实下一 token 的概率；对数默认取 e 为底（单位 nats），以 2 为底时单位 bits，换算见第 5 节。

交叉熵（cross-entropy）定义为

$$
\mathrm{CE}(p, q) = - \sum_{v=1}^{V} p(v) \log q(v) \qquad (5)
$$

**符号**： $p$ ：目标分布； $q$ ：模型分布；求和遍历词表。

NLL 与 CE 是同一个东西，一步展开即可见：训练目标里 $p$ 是真实 token $x_t$ 的 one-hot 分布 $\delta_{x_t}$ （在 $v = x_t$ 处为 1，其余为 0），代入式 (5)：

$$
\mathrm{CE}(\delta_{x_t}, q_\theta) = - \sum_{v=1}^{V} \delta_{x_t}(v) \log q_\theta(v | x_{\lt t}) = - \log q_\theta(x_t | x_{\lt t})
$$

求和里只有 $v = x_t$ 一项非零。对 $t$ 取平均即式 (4)。所以「训 NLL」「训 cross-entropy」「训 MLE」在 LM 语境下是同一句话。另外记一个反向传播的入口事实：softmax 与 CE 复合后，式 (4) 对 logits $z_i$ 的偏导数有闭式 $q_i - \delta_{x_t}(i)$ ，即「预测概率减真值」，逐位置的分类误差信号直接落在 logits 上。

计算层面靠 **教师强制（teacher forcing）**：训练时喂给模型的条件 $x_{\lt t}$ 一律是数据里的真实 token，不是模型自己采样的结果。这样 $T$ 个位置的条件分布可以在一次前向里并行算出，loss 再求平均：

```
位置 t     :   1      2      3      4      5
输入       : <bos>    w1     w2     w3     w4     ← 真实 token（teacher forcing）
              |      |      |      |      |
              +------+------+------+------+------ 一次前向，5 个位置并行
              v      v      v      v      v
输出       :  q1     q2     q3     q4     q5     ← 每个 q_t 是 Δ^{V-1} 中一点
监督目标   :   w1     w2     w3     w4   <eos>    ← 输入整体右移一位对齐
逐位置 loss: -log q1(w1)  ...        -log q5(<eos>)   →  取平均得式 (4)
```

代价是训练与生成的分布不一致：生成时条件来自模型自己的输出，错了会滚雪球。这个落差叫 **暴露偏差（exposure bias）**，缓解方法属于 training 维度，本章只登记定义。

teacher forcing 的另一面是数据效率：一条长 $T$ 的序列经右移对齐后给出 $T$ 个分类问题（每个位置一个，类别数 $V$ ），一次前向全部打分。对照生成阶段：每产出一个 token 需要一次前向，长 $T$ 的输出要串行 $T$ 次。训练并行、生成串行的不对称，是 KV cache（2.4）与整个 inference 维度存在的根本原因。

教学用参考实现（非优化实现）：

```python
import torch

def nll_per_token(logits, targets):
    """教学用，非优化实现。logits: [B, T, V]；targets: [B, T]，即输入右移一位后的真实 token。"""
    log_q = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))  # log-softmax，[B, T, V]
    nll = -log_q.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)     # 取真实 token 的 log q，[B, T]
    return nll.mean()
```

## 5. 信息论视角：CE = H(p) + KL(p‖q)

把 CE 按 KL 散度展开，一行看清训练到底在优化什么。KL 散度定义并展开：

$$
\mathrm{KL}(p \Vert q_\theta) = \sum_{v=1}^{V} p(v) \log \frac{p(v)}{q_\theta(v)} = \sum_{v=1}^{V} p(v) \log p(v) - \sum_{v=1}^{V} p(v) \log q_\theta(v) = - H(p) + \mathrm{CE}(p, q_\theta)
$$

第一步把商的对数拆成两项，第二步识别出两个定义： $H(p) = - \sum_v p(v) \log p(v)$ 是 $p$ 的熵（entropy），第二项正是式 (5)。移项得

$$
\mathrm{CE}(p, q_\theta) = H(p) + \mathrm{KL}(p \Vert q_\theta) \qquad (6)
$$

**符号**： $H(p)$ ：真实分布的熵； $\mathrm{KL}(p \Vert q_\theta)$ ： $p$ 对 $q_\theta$ 的 KL 散度，非负，当且仅当两分布相等时为 0。

式 (6) 读出三条工程结论：

1. **最小化 CE 等价于最小化 KL**。 $H(p)$ 不含 $\theta$ ，优化 CE 就是在缩小模型与数据分布的 KL 距离。
2. **loss 有数据决定的地板**。 $\mathrm{CE} \ge H(p)$ ，等号仅在 $q_\theta = p$ 时成立。loss 降不到 0，残差是语言本身的不可压缩性。
3. **超出部分就是浪费的码长**。 $\mathrm{CE} - H(p) = \mathrm{KL}$ 是用 $q_\theta$ 给来自 $p$ 的数据编码时、每 token 比最优码多花的比特数——LM 训练等价于学一个算术编码器，这是「LM 即压缩」的精确版本。

KL 散度不对称，方向不是细节：MLE 优化的是 $\mathrm{KL}(p_{\mathrm{data}} \Vert q_\theta)$ ——期望在 $p_{\mathrm{data}}$ 下取， $q_\theta$ 必须给数据出现的每个 mode 分配足够概率，否则 $\log q_\theta \to -\infty$ 的惩罚极大，行为是「全覆盖」（mean-seeking）。反过来 $\mathrm{KL}(q_\theta \Vert p_{\mathrm{data}})$ 期望在模型分布下取，允许 $q_\theta$ 收缩到 $p_{\mathrm{data}}$ 的少数 mode 上（mode-seeking），是部分蒸馏与生成后训练场景的对象。本章式 (6) 固定用前一个方向，因为训练 loss 的期望就是对数据分布取的。

单位换算：CE 以 nats 计时，**bits/token** = $\mathrm{CE} / \ln 2$ 。两个模型在同一测试集上差 0.1 nats/token，就是每 token 差约 0.144 bits。

## 6. Perplexity 与两个陷阱

困惑度（perplexity）只是 CE 的指数化：

$$
\mathrm{PPL} = \exp(\mathrm{CE}_{\mathrm{nats}}) = 2^{\mathrm{CE}_{\mathrm{bits}}} \qquad (7)
$$

**符号**： $\mathrm{PPL}$ ：perplexity；两种底数的 CE 按上式互换。直觉读法：等效均匀分支因子——模型每步相当于在一个 $\mathrm{PPL}$ 个选项的均匀分布里猜。

基本性质： $\mathrm{PPL} \ge 1$ ，且由式 (6) 知实际地板更高： $\mathrm{PPL} \ge \exp(H(p))$ ，等号要求模型与数据分布完全一致。随机猜测（ $q$ 为均匀分布）时 $\mathrm{CE} = \log V$ ，故 **uniform 基线 PPL = $V$** 。代入三个锚点：GPT-2 的乱猜基线是 50257，Llama-3.1-8B 是 128256，Qwen3-8B 是 151936。同一个「乱猜」在不同词表下 PPL 差 3 倍——陷阱一已经露面。

**陷阱一：PPL 依赖词表大小。** PPL 的量纲是「词表里的选项数」， $V$ 一变，基线和数值刻度一起变。跨模型比较 PPL，前提必须是同 tokenizer、同 $V$ 、同测试集。

**陷阱二：跨 tokenizer 不可比，即使 $V$ 相同。** 同一字符串经不同 tokenizer 得到的 token 数不同，per-token CE 是对不同数量的项取平均。构造算例（数字为演示而设）：字符串 $s$ ，tokenizer A 切成 10 个 token，tokenizer B 切成 20 个；两个模型对 $s$ 的总 NLL 都是 20 nats。则 A 的 per-token CE = 2.0、PPL = $\exp(2.0)$ ，约 7.39；B 的 per-token CE = 1.0、PPL = $\exp(1.0)$ ，约 2.72。两个模型对字符串的建模完全一样好，per-token PPL 却差 2.7 倍——B 只是摊到了更多 token 上。正确做法是换总量纲：bits per byte（总 NLL 折 bits 后除以原文字节数），tokenizer 从中消掉：

$$
\text{bits per byte} = \frac{1}{n_{\mathrm{byte}}} \sum_{t=1}^{T} \frac{- \log q_\theta(x_t | x_{\lt t})}{\ln 2}
$$

**符号**： $n_{\mathrm{byte}}$ ：原文的字节数；分子是该序列的总 NLL（nats），除以 $\ln 2$ 折成 bits。这个量只依赖字符串与模型，与 tokenizer 无关，是跨模型比较的可用口径。

这也是为什么 scaling law 论文把 CE（而不是 PPL）画在纵轴上、并且全程固定 tokenizer 与测试集：[Kaplan et al., 2020](../references.md)（arXiv:2001.08361，[B]）与 [Brown et al., 2020](../references.md)（arXiv:2005.14165，[B]）的曲线只在该固定口径内有绝对意义，跨论文对数字前先核对口径（1.3 展开）。

## 7. CLM / MLM / prefix-LM：attention mask + loss mask 的统一形式

三类训练目标不需要三套定义。统一成三个对象：**attention mask** $M \in \lbrace0, -\infty\rbrace^{T \times T}$ 决定位置 $i$ 能看见谁（进入 attention 的 softmax，细节在 2.3），**loss mask** $\ell \in \lbrace0, 1\rbrace^{T}$ 决定哪些位置计入损失，**输入序列** $x'$ 是实际送入模型的 token 序列（可能被腐蚀）：

$$
\alpha_{ij} = \frac{\exp(s_{ij} + M_{ij})}{\sum_{l=1}^{T} \exp(s_{il} + M_{il})}, \qquad L(\theta) = - \sum_{t=1}^{T} \ell_t \log q_\theta(x_t | x'; M) \qquad (8)
$$

**符号**： $s_{ij}$ ：未加 mask 的 attention 打分（2.3 定义）； $\alpha_{ij}$ ：mask 后的 attention 权重， $M_{ij} = -\infty$ 时 $\exp$ 项为 0，位置 $j$ 对 $i$ 不可见； $\ell_t$ ：loss mask，1 表示该位置计损失； $x'$ ：经 mask 方案处理后的输入序列； $q_\theta(x_t | x'; M)$ ：在输入 $x'$ 与可见性 $M$ 下，位置 $t$ 处真实 token $x_t$ 的条件概率。

式 (8) 的条件是整条输入 $x'$ 而不是前缀 $x_{\lt t}$ ：可见性语义由 $M$ 承担，腐蚀语义由 $x'$ 承担。CLM 时 $x' = x$ 且 $M$ 取下三角，位置 $t$ 恰好只能看到 $x_{\lt t}$ ，式 (8) 退化为本章前半的 $q_\theta(x_t | x_{\lt t})$ ；MLM 与 prefix-LM 的双向可见性也收进同一记号。三种 objective 只是 $(x', M, \ell)$ 的三个取值：

- **CLM**（causal LM，GPT 家族）： $x' = x$ ， $M_{ij} = 0$ 当 $j \le i$ ，否则 $-\infty$ （下三角可见）， $\ell_t = 1$ 对所有 $t$ 。即本章式 (3)(4)。
- **MLM**（masked LM，BERT 家族）： $x'$ 为腐蚀序列——一个子集的位置被替换为特殊 mask token； $M$ 全 0（双向全可见）， $\ell_t = 1$ 仅落在被替换的位置上。
- **prefix-LM**： $x' = x$ ，前缀区双向可见、续写区因果可见， $\ell_t = 1$ 仅落在续写区。

$T$ = 4 的两个 $M$ 实例（左 CLM，右 prefix 长为 2 的 prefix-LM）：

$$
M_{\mathrm{CLM}} = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix} \qquad M_{\mathrm{prefix}} = \begin{bmatrix} 0 & 0 & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

**符号**：行 $i$ 是被影响的位置，列 $j$ 是被看的位置；0 = 可见， $-\infty$ = 屏蔽。

这套 $(x', M, \ell)$ 记号是本书后续章节的接口：2.4 用它定义因果性与 KV cache 的依赖结构，3.1 在此基础上把 span corruption、UL2 等更多目标收进同一框架，3.2 则证明 $M$ 全 0 的架构为什么不能直接做自回归生成。此处只建记号，不展开。

## 8. 采样生成：temperature 与 top-p

训练好的 $f_\theta$ 生成文本就是把式 (3) 反过来用：从 $x_0$ = bos 出发，逐位置从 $q_\theta(\cdot | x_{\lt t})$ 中采样 $x_t$ ，拼回上下文继续，直到采出 eos 或达到 $T_{\max}$ 。由链式法则，这样得到的序列恰好是模型分布的精确样本，即祖先采样（ancestral sampling）。两个最常用的分布整形手段：

**温度（temperature）**：对 logits 做缩放再 softmax，

$$
q_i(\tau) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{V} \exp(z_j / \tau)} \qquad (9)
$$

**符号**： $\tau > 0$ ：温度。 $\tau = 1$ 即原始分布； $\tau \to 0^{+}$ 时，记 $`i^{*}`$ 为最大 logit 的下标，对任意 $`z_i < z_{i^{*}}`$ 有比值 $`q_i / q_{i^{*}} = \exp((z_i - z_{i^{*}}) / \tau) \to 0`$ ，分布退化为 argmax 的 one-hot； $\tau \to \infty$ 时所有比值趋于 1，分布趋于均匀。

**top-p（nucleus sampling）**：把概率降序排为 $p_{(1)} \ge p_{(2)} \ge ...$ ，取满足累积质量阈值的最小前缀集合 $S = \lbrace1, ..., k\rbrace$ 使 $\sum_{i=1}^{k} p_{(i)} \ge p$ ，集合外清零并重归一：

$$
q'_i = \begin{cases} p_{(i)} / Z, & i \in S \\ 0, & \text{otherwise} \end{cases}, \qquad Z = \sum_{j \in S} p_{(j)} \qquad (10)
$$

**符号**： $p \in (0, 1]$ ：累积质量阈值； $Z$ ：保留集合的概率总质量。温度与 top-p 可叠加：先按式 (9) 整形，再按式 (10) 截断。

两个极端各自的失效模式说明为什么需要整形：argmax（ $\tau \to 0$ 的极限）逐步取 mode，而逐位置 mode 的串联不保证整句高概率——局部最优的贪心链，实际行为是退化成重复循环； $\tau = 1$ 的纯采样保留了分布尾部，长尾里低质量 token 积少成多。top-p 的机制正好对症：集合大小随分布形状自适应——分布尖锐时 $S$ 只有一两个元素，分布平坦时 $S$ 放大，尾部噪声被按质量而不是按个数切掉。生成阶段的系统工程——KV cache、批处理、投机采样——属于 [inference 维度](../../inference/README.md)，本章只保证采样侧的分布语义清楚。

## 9. Softmax 瓶颈（softmax bottleneck）：输出头的 rank 上限

式 (2) 藏着一个与数据无关的表示上限。把 $N$ 个不同上下文的隐状态堆成 $H \in \mathbb{R}^{N \times d}$ ，对应的 logits 矩阵为

$$
Z = H W^{\top} + e_N b^{\top} \in \mathbb{R}^{N \times V}
$$

**符号**： $e_N \in \mathbb{R}^{N}$ ：行方向的全 1 列向量； $b \in \mathbb{R}^{V}$ ：词表维偏置， $e_N b^{\top} \in \mathbb{R}^{N \times V}$ 是 rank 1（每一行都是同一个 $b$ ）； $H W^{\top}$ 的 rank 不超过 $d$ 。

log-softmax 是逐行操作： $\log q_{c,v} = z_{c,v} - \log \sum_{v'} \exp(z_{c,v'})$（ $v'$ 为词表上的求和哑指标， $c$ 为上下文下标），行 $c$ 减去的常数记为 $u_c$ ，于是

$$
\log P = Z - u 1_V^{\top}, \qquad \mathrm{rank}(\log P) \le d + 2 \qquad (11)
$$

**符号**： $\log P \in \mathbb{R}^{N \times V}$ ：模型在 $N$ 个上下文上的对数概率矩阵； $u \in \mathbb{R}^{N}$ ：逐行归一化常数； $1_V \in \mathbb{R}^{V}$ ：词表维的全 1 列向量， $u 1_V^{\top} \in \mathbb{R}^{N \times V}$ 是 rank 1（第 $c$ 行是常向量 $u_c \cdot 1_V^{\top}$ ）。不含 bias 时上界收紧为 $d + 1$ 。

结论：无论 softmax 怎么训练，模型能表达的对数概率矩阵 rank 以 $d$ 为上界，与 $V$ 无关。真实语言的条件分布矩阵没有理由落在这个低秩子空间里：同一上下文后面接哪些 token，由语义、语法、语用多组近似独立的因素决定，互斥但各自合理的续写集合很大，对应的 $\log P$ 矩阵行与行之间的线性关系接近满秩。只要真实矩阵的 rank 超过 $d + 2$ ，就存在模型永远拟合不了的残差——增加数据、加深主干都无效，因为瓶颈在输出头的线性结构上，这就是 **softmax 瓶颈（softmax bottleneck）**（[Yang et al., 2018](../references.md)，arXiv:1711.03953，[A]）。代入锚点： $d$ = 4096、 $V$ = 151936（Qwen3-8B），rank 上限 4098 对满秩 151936，不到词表大小的 3%。原论文的解法是 Mixture of Softmaxes（MoS）：用多个低秩头的凸组合把 rank 抬上去，公式拆解连同它与 weight tying 的关系放在 2.2。

## 总结

| 概念 | 记号 / 公式 | 要点 |
| --- | --- | --- |
| 样本空间 | $X$ = 全体长度 $\le T_{\max}$ 的 token 序列 | $V$ = 词表大小（50257 / 128256 / 151936） |
| 模型 | $f_\theta: V^{t-1} \to \Delta^{V-1}$ | 输出恒为下一 token 的分布，式 (1)(2) |
| AR 分解 | $P(x_{1:T}) = \prod_t P(x_t \vert x_{\lt t})$ | 链式法则恒等式；顺序任意，因果序为生成服务，式 (3) |
| 训练目标 | $\mathrm{NLL}$ = 逐位置 CE 的平均 | NLL = CE = MLE，式 (4)(5) |
| 信息论分解 | $\mathrm{CE} = H(p) + \mathrm{KL}(p \Vert q_\theta)$ | 地板是数据熵；超出的 bits 即浪费码长，式 (6) |
| perplexity | $\mathrm{PPL} = \exp(\mathrm{CE})$ | uniform 基线 = $V$ ；跨词表、跨 tokenizer 不可比，式 (7) |
| objective 统一 | $(x', M, \ell)$ ：输入 + attention mask + loss mask | CLM / MLM / prefix-LM 只是三种取值，式 (8) |
| 采样整形 | 温度式 (9)、top-p 式 (10) | $\tau \to 0$ 为 argmax；top-p 动态截断尾部 |
| softmax 瓶颈 | $\mathrm{rank}(\log P) \le d + 2$ | 上限由 $d$ 定，与 $V$ 无关；MoS 见 2.2，式 (11) |

## 练习

1.（推导）证明式 (3) 的顺序任意性：对位置集合 $\lbrace1, ..., T\rbrace$ 的任意排列 $\pi$ ，用条件概率定义式归纳证明 $P(x_{1:T}) = \prod_{t=1}^{T} P(x_{\pi(t)} | x_{\pi(1)}, ..., x_{\pi(t-1)})$ 。再从生成过程的因果性出发，说明为什么实际系统只取恒等排列。
2.（计算）某模型在固定测试集上 per-token CE = 3.2 nats。(a) 求 PPL；(b) 折成 bits/token；(c) 设 $V$ = 50257，uniform 基线的 CE 与 PPL 各是多少，该模型相对基线每 token 省多少 bits；(d) 换一个 tokenizer 后同一测试集 token 数翻倍、总 NLL 不变，新的 per-token PPL 是多少？这个数字变化说明了什么？
3.（推导）完成式 (11) 的细节：设 $H \in \mathbb{R}^{N \times d}$ 、 $W \in \mathbb{R}^{V \times d}$ 、 $b \in \mathbb{R}^{V}$ ，写出 $u$ 的逐分量定义，并证明 $\mathrm{rank}(\log P) \le d + 2$ ；说明无 bias 时为何收紧为 $d + 1$ 。代入 Qwen3-8B 的 $d$ = 4096、 $V$ = 151936，计算上界占满秩的比例。
4.（代码）教学练习，不使用 torch.nn.functional.cross_entropy：(a) 给定 logits（ $[B, T, V]$ ）与 targets（ $[B, T]$ ），手写 per-token NLL，与官方函数对拍验证；(b) 写单步 nucleus 采样：排序、累积求和、按阈值 $p$ 截断、重归一、multinomial 采样，打印各中间张量 shape。

## 参考

- [Kaplan et al., 2020](../references.md)：Scaling Laws for Neural Language Models，arXiv:2001.08361，[B]（key: kaplan）
- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）
- [Radford et al., 2019](../references.md)：Language Models are Unsupervised Multitask Learners，OpenAI 技术报告 + HF 权重，[W]（key: gpt2）
- [Yang et al., 2018](../references.md)：Breaking the Softmax Bottleneck: A High-Rank RNN Language Model，arXiv:1711.03953，[A]（key: softmax-bottleneck）
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [Qwen, Qwen3-8B config.json 与 tokenizer 文件](../references.md)：[A]（key: qwen3-8b-config）
