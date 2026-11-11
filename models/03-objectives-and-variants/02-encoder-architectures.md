# 3.2 Encoder-only 与 Encoder-Decoder（Encoder-only and Encoder-Decoder Architectures）

> 本章回答两个问题：双向 attention 堆出的 encoder 形态能做什么、不能做什么（不能自回归生成——给完整证明）；把「读」与「写」拆成两个栈的 encoder-decoder 形态靠什么结构接起来（cross-attention），以及这个分工在什么任务上占优 | 前置：1.1（$(x', M, \ell)$ 三元组、链式分解）、2.2（weight tying）、2.3（attention 公式）、2.4（mask 数学、KV cache 账）、3.1（objective 与 corruption 方案） | 锚点：ModernBERT-base/large config [A]、ELECTRA 官方 repo [A]、openai/whisper repo [A]；BERT [A]、RoBERTa [A]、ALBERT [A]、ELECTRA [A]、T5 [A]、Whisper [A]

## 1. Encoder-only：M ≡ 0 的形态定义

BERT 的全部 12 层里没有一处 mask：attention mask 恒为全零矩阵（[Devlin et al., 2018](../references.md)，arXiv:1810.04805，[A]）。用 2.4 §1 的记号写出来：

$$
M_{ij} = 0 \quad \forall i, j \qquad \Longleftrightarrow \qquad V(i) = \{1, \ldots, T\} \qquad (1)
$$

**符号**：$M \in \{0, -\infty\}^{T \times T}$：attention mask，加在 softmax 前的 score 上（2.4 式 (1)）；$V(i)$：位置 $i$ 的可见集。式 (1) 是 2.4 §2 对照表里 MLM 那一行的架构侧读法。

每个位置看见全序列，attention 是唯一跨位置混合信息的算子（2.1 §1），其余组件（RMSNorm/LayerNorm、FFN、残差加）与 decoder 栈逐字同构——**encoder-only 与 decoder-only 的架构差异只在 $M$ 这一个对象上**。差异的后果却很大：这个形态的产品不是「下一 token 的分布」，而是整序列表征矩阵

$$
H^{(L)} = \mathrm{Enc}_\theta(x') \in \mathbb{R}^{T \times d} \qquad (2)
$$

**符号**：$x'$：实际输入序列（可能经腐蚀，3.1）；$L$：层数；$d$：隐藏维。每一行 $h_t^{(L)}$ 都是全序列的函数——这个依赖方向是 §2 的全部麻烦的来源，也是 §3 全部接口的资产。

输入格式上还有两个沿用至今的约定：序列前置特殊 token [CLS]、句对之间与末尾插 [SEP]（BERT 的输入模板，词表为 30522 的 WordPiece）。这两个特殊 token 不携带词汇内容，其作用分别是序列级读出位（§3）与句对边界标记（§4）。

## 2. 能力边界：无法自回归生成的证明

2.4 §2 已给出可生成性的判定门槛 $V(i) \subseteq \{1, \ldots, i\}$（2.4 式 (4)），并指出 MLM 的全零矩阵处处违反它。本节把「为什么违反了就无法生成」补成一个完整命题，包括最常见的绕过方案为什么无效。

**命题**（$M \equiv 0$ 不可自回归生成）。设网络所有层满足式 (1)，输入长度 $T$、输出 logits $z \in \mathbb{R}^{T \times V}$。则：(i) $z_t$ 定义的条件分布以未来 token 为条件，不是 1.1 式 (3) 链式分解的任何因子，且在生成第 $t$ 步无法实例化；(ii) 不存在合法的增量计算——追加 token 使全部历史表征失效；(iii) 给未来位置填占位符不能修复 (i)；(iv) 迭代重采样是另一个数学对象，不构成反例。

**(i) 无可用因子。** 自回归采样第 $t$ 步需要 $P(x_t | x_{<t})$。而 $M \equiv 0$ 时，第一层 attention 的第 $t$ 行为

$$
o_t = \sum_{j=1}^{T} \alpha_{tj} v_j, \qquad \alpha_{tj} = \frac{\exp(s_{tj})}{\sum_{l=1}^{T} \exp(s_{tl})} > 0 \quad \forall j \qquad (3)
$$

**符号**：$s_{tj}$：未加 mask 的 score（2.3 式 (4)）；$v_j$：位置 $j$ 的 value 向量；$\alpha_{tj}$：attention 权重。score 有限时 $\exp$ 严格为正，所以 $o_t$ 以正权重混入每一个未来位置的 value，经层堆叠后 $z_t = f(x_1, \ldots, x_T)$ 依赖全序列。生成第 $t$ 步时 $x_{>t}$ 尚不存在，这个条件分布连输入都凑不齐。要让 $z_t$ 不依赖未来，唯一途径是 $\alpha_{tj} = 0$（$j > t$），即 $M$ 回到下三角——「双向可见」与「可自回归生成」在 mask 对象上恰好互斥。

**(ii) 无增量计算。** 2.4 §4.2 的缓存证明在第一步归纳就断裂：$o_s$ 依赖 $x_{>s}$，追加第 $T+1$ 个 token 会经由式 (3) 改变所有历史位置的 stream——cache 里的 $k_s, v_s$ 全部过期。不是「没存」，是「存了也错」。唯一忠实的执行方式是每步对全序列重跑前向，$T$ 步总账为 2.4 式 (5) 的 $\approx NT^{2} + \frac{2}{3}LT^{3}d$，且没有任何 cache 版本可退化到——2.4 §3 的 prefill/decode 两阶段对这个形态无从谈起。

**(iii) 占位符修补无效。** 把未来位置填成常量（mask token 或随机 token）再前向，得到的是 $q(x_t | x_{<t}, \tilde{x}_{>t})$，条件是「真实前缀 + 人造后缀」。由 (i) 的严格正依赖性，改变后缀取值就改变 $z_t$，所以该分布是占位符取值的函数，不是数据的条件分布。更糟一层：训练时腐蚀序列 $x'$ 的占位分布（BERT 口径 15% 位置被选中，3.1）与生成时的占位分布（位置 $t$ 之后全占位）不匹配——条件本身就落在训练分布之外。两层都修不好。

**(iv) 迭代精修是另一对象。** 常见反驳：「先填全 [MASK]，反复重预测，不也能出文本？」能出，但产物不是自回归样本。该过程是以整条序列为状态的马氏链：每步重采样一个（或全部）位置，转移核由 $q(x_t | x_{\setminus t})$ 给出。它没有链式分解，没有任何定理保证这组单点条件族与数据联合分布相容、或迭代收敛到它——MLM 的 loss 只定义了腐蚀条件下的条件分布，从未定义联合分布。4.8 的扩散式文本模型正是把这条链系统化的另一条生成范式（训练目标与账全部不同），它不再声称实现 $\prod_t P(x_t | x_{<t})$。

**推论（反向读出）。** 满足 $V(i) \subseteq \{1, \ldots, i\}$ 的最大可见集是 $\{1, \ldots, i\}$ 本身——causal mask 是「可生成」硬约束下信息量最大的 mask。decoder-only 形态（3.3）是这条约束的饱和解，本节与 2.4 §2 合起来把它变成了定理而非经验。

## 3. 输出接口：读出而不是生成

encoder 的产出是式 (2) 的矩阵 $H^{(L)}$，下游任务从它读出。三类标准接口：

**[CLS] 汇聚（序列级）**。取 [CLS] 位置的最终层向量，过分类头：

$$
p = \mathrm{softmax}\left(W_c \, \tanh(W_p \, h_{[\mathrm{CLS}]}^{(L)})\right), \qquad W_c \in \mathbb{R}^{C \times d}, \ W_p \in \mathbb{R}^{d \times d} \qquad (4)
$$

**符号**：$C$：类别数；$W_p$：pooler 变换（BERT 结构）。机制上 [CLS] 是一个学出来的 pooling 位：它无词汇内容先验，表征全部来自逐层双向聚合——每层 attention 都允许它从任意位置取证。BERT 预训练用同一个头做 NSP（§4），fine-tune 时换成任务头。

**mean pooling（句向量）**。对有效 token 位取均值：

$$
\bar{h} = \frac{1}{|\mathcal{I}|} \sum_{t \in \mathcal{I}} h_t^{(L)} \qquad (5)
$$

**符号**：$\mathcal{I}$：有效 token 下标集（剔除 padding 与特殊 token）。检索/句向量的默认读出；与 [CLS] 的差别是没有专属训练信号，表示质量全由 MLM 间接塑形。

**token-level head（逐位置）**。每个位置独立分类：

$$
p_t = \mathrm{softmax}(W h_t^{(L)}), \qquad W \in \mathbb{R}^{C \times d} \qquad (6)
$$

序列标注（$C$ 为标签集）与抽取式问答（两个向量 $w_s, w_e$ 分别给出 start/end 位置的打分 $p_t^{s} \propto \exp(w_s^{\top} h_t)$）都是这个形态。注意 MLM 预训练头本身就是式 (6) 的特例：$z_t = h_t^{(L)} E^{\top} + b$，$C = V$ 且与输入 embedding 权重绑定（weight tying，2.2）。

三类接口的共同点值得记住：它们全部是「读出」——没有一个定义了序列上的生成分布。§2 的能力边界在接口层就看得见。

## 4. NSP：公式与失败机制

BERT 的第二个预训练目标是下一句预测（next sentence prediction, NSP）：输入句对 $(A, B)$，50% 的样本里 $B$ 是 $A$ 的真实下文（IsNext），50% 从语料随机抽取（NotNext），在 [CLS] 表示上做二分类：

$$
L_{\mathrm{NSP}} = - \log P(c | h_{[\mathrm{CLS}]}), \qquad c \in \{\text{IsNext}, \text{NotNext}\} \qquad (7)
$$

**符号**：$P(c | \cdot)$：式 (4) 的二分类特例（$C$ = 2）。BERT 总 loss 为 $L_{\mathrm{MLM}} + L_{\mathrm{NSP}}$ 等权相加。

失败机制是任务混淆：IsNext/NotNext 的分类面上纠缠着两个信号——主题相关性（topic）与连贯性（coherence）。随机负样本几乎必然主题跳变，于是「主题是否匹配」单独就够把分类准确率推高：模型走低阻路径，$h_{[\mathrm{CLS}]}$ 学成主题检测器，连贯性信号几乎不进梯度。代理任务过易，目标信号被替代信号吸干。

证据是消融而不是论证：RoBERTa（[Liu et al., 2019](../references.md)，arXiv:1907.11692，[A]）把输入格式与 NSP 分四档对照——SEGMENT-PAIR 与 SENTENCE-PAIR 保留 NSP，FULL-SENTENCES（可跨文档边界的长序列块）与 DOC-SENTENCES（单文档内的完整句子块，不跨文档）去掉 NSP——论文结论口径为去掉 NSP 后下游整体持平或略升；逐指标看并非单调，例如 SENTENCE-PAIR 换到 DOC-SENTENCES 时 SQuAD1.1 从 90.4 降到 88.7，账面提升主要来自输入格式的更换而非 NSP 的移除本身。结论：NSP 的净贡献为零或负，BERT 预训练的收益全部在 MLM 一侧，RoBERTa 因此只训 MLM。

机制修正的反面构造来自 ALBERT：把负样本换成「同一文档中连续两句交换顺序」（sentence-order prediction, SOP）——主题一致性由构造保证，分类面上只剩连贯性信号（[Lan et al., 2019](../references.md)，arXiv:1909.11942，[A]）。SOP 的存在本身就是对「任务混淆」诊断的确认：把易信号从负样本里剔除，任务才迫使模型学目标信号。在 3.1 的 $(x', M, \ell)$ 框架里 NSP 是第四个对象——它不动三元组中的任何一个，是挂在读出接口上的辅助 loss；RoBERTa 的消融说明辅助 loss 不是免费的，设计错了会污染表示。

## 5. ALBERT：参数效率的两个杠杆

ALBERT（[Lan et al., 2019](../references.md)，arXiv:1909.11942，[A]）不动 mask、不动 objective 类型（MLM + SOP），动的是参数布局。两个杠杆都可以用 BERT-base 字段（$V$ = 30522、$d$ = 768、$L$ = 12、$d_{\mathrm{ff}}$ = 3072 = $4d$）把账算到底；以下数字是按字段直算的演算例。

**杠杆一：embedding 分解（factorized embedding）。** BERT 把词表直接映到隐藏维：$E \in \mathbb{R}^{V \times d}$，查表 $h_t^{(0)} = E[x_t]$。ALBERT 插入瓶颈维 $e \ll d$：

$$
E = E_1 E_2, \qquad E_1 \in \mathbb{R}^{V \times e}, \quad E_2 \in \mathbb{R}^{e \times d} \qquad (8)
$$

**符号**：$E$：完整 embedding 矩阵；$E_1$：词表到瓶颈维的投影；$E_2$：瓶颈维到隐藏维的投影；$e$：瓶颈维（$e \ll d$）；查表变为 $h_t^{(0)} = (E_1[x_t]) E_2$。

参数量从 $Vd$ 变 $Ve + ed$。代入 $e$ = 128：$Vd \approx 23.4 \times 10^{6}$，$Ve + ed \approx 3.91 \times 10^{6} + 0.098 \times 10^{6} \approx 4.0 \times 10^{6}$——单项削减约 83%。机制理由：one-hot 查表是「身份 $\to$ 词义」的一次映射，中间没有上下文混合，rank 需求远低于 $d$（对照 1.1 §9 的 rank 视角）；分解把「上下文无关的词义空间」（$e$ 维）与「上下文混合的表示空间」（$d$ 维）解耦，各定各的容量。

**杠杆二：跨层参数共享（cross-layer parameter sharing）。** $L$ 层共享同一份参数，同一算子迭代 $L$ 次：

$$
h^{(\ell+1)} = f_\theta(h^{(\ell)}), \qquad \ell = 0, \ldots, L-1 \qquad (9)
$$

其中 $\theta$ 跨层共用（所有层同一份参数）。

层参数从 $L \cdot n_{\mathrm{layer}}$ 降到 $n_{\mathrm{layer}}$。代入字段：每层 attention 四矩阵 $4d^{2}$ 加 FFN $2 d \, d_{\mathrm{ff}}$ 共 $12 d^{2} \approx 7.1 \times 10^{6}$（忽略 LN 与 bias），12 层不共享为 $84.9 \times 10^{6}$、共享后 $7.1 \times 10^{6}$。两杠杆叠加，同构型总参数从约 $1.1 \times 10^{8}$（BERT-base 公开口径 110M）压到约 $1.2 \times 10^{7}$ 量级。

代价写在数学里：迭代同一算子 $L$ 次是不动点式的受限函数族——第 3 层与第 9 层被迫是同一个变换，逐层特化（不同深度分工）被牺牲。ALBERT 的取舍是把省下的参数预算换更大的 $d$：论文的结论方向是参数效率（同下游性能下参数少约一个量级），不是同参数下的表达力提升。这个机制换个方向看也熟悉——共享参数沿深度迭代，与 RNN 沿时间共享参数是同一种参数复用，只是复用轴换了。

## 6. ELECTRA：把监督铺满全序列

MLM 的监督密度是结构性短板：BERT 口径下只有 15% 的位置计 loss（选中与替换方案见 3.1），一条序列 85% 位置的表示参与前向却不直接受训。ELECTRA（[Clark et al., 2020](../references.md)，arXiv:2003.10555，[A]；官方 repo [google-research/electra](../references.md)，[A]）把任务换成逐位置二分类，让每个位置都出信号。

结构是两级：一个小型 generator $G$ 按 MLM 训练（15% 腐蚀），在它的输出分布上采样替换 token 得到腐蚀序列 $x^{c}$；判别器 $D$（全尺寸 encoder）逐位置判断「该位置是否被替换过」——替换 token 检测（replaced token detection, RTD）：

$$
L_{G} = - \sum_{t \in \mathcal{M}} \log q_G(x_t | \tilde{x}), \qquad L_{D} = - \sum_{t=1}^{T} \left[ y_t \log D(h_t) + (1 - y_t) \log (1 - D(h_t)) \right] \qquad (10)
$$

**符号**：$\mathcal{M}$：被腐蚀的位置集；$\tilde{x}$：腐蚀输入；$q_G$：generator 的 MLM 分布；$x^{c}_t \sim q_G(\cdot | \tilde{x})$（$t \in \mathcal{M}$）；$y_t$：替换指示，$x^{c}_t \neq x_t$ 时为 1、否则为 0；$D(h_t) = \mathrm{sigmoid}(w^{\top} h_t)$：判别器输出「被替换」的概率。总 loss 为 $L_G + \lambda L_D$，论文取 $\lambda$ = 50——最终交付的模型是 $D$，训练信号以它为主，$G$ 只是供负例的脚手架。

与 GAN 的界线要划清：式 (10) 里的采样 $x^{c}_t \sim q_G$ 不可微，$L_D$ 的梯度到不了 $G$；$G$ 只被自己的 MLM loss 训练，两个网络各训各的——这不是 min-max 对抗，而是「用一个会进步的噪声源给判别器供负例」。收益恰好来自 $G$ 会进步：训练早期负例离谱（$D$ 轻松学会粗边界），随 $G$ 变好负例越来越微妙，$D$ 的决策边界被迫贴近真实数据分布的细结构——负例难度自适应，且不需要对抗训练的任何稳定性技巧。$G$ 刻意做小（ELECTRA-base 口径 $G$ 宽 256、$D$ 宽 768）并与 $D$ 共享 token embedding：$G$ 太准会让 $y$ 几乎全 0、$D$ 没信号，太弱则负例过易学不到细表示。

样本效率的账在一行里：$L_D$ 的支撑集是全部 $T$ 个位置，$L_{\mathrm{MLM}}$ 是 $0.15 T$——同等 token 吞吐下，直接受训的位置数多 $1 / 0.15 \approx 6.7$ 倍。单看一个位置，二分类的监督（约 1 bit 级）远弱于 $V$ 类分类；密度的 6.7 倍把总量扳回。论文的同算力对照（固定 FLOPs 比下游分）显示 ELECTRA 一致高于同规模 MLM 系——监督铺满全序列是该收益的机制解释。

## 7. Encoder-decoder：cross-attention 与条件生成

把 §2 的边界与 §3 的接口能力合并，得到分工形态：双向栈只管「读」（输入完整已知，$M \equiv 0$ 合法），因果栈只管「写」（保住 2.4 式 (4) 的可生成性），中间用 cross-attention 接通。生成语义是条件链式分解：

$$
P(y_{1:T} | x_{1:S}) = \prod_{t=1}^{T} P(y_t | y_{<t}, \, x_{1:S}) \qquad (11)
$$

**符号**：$x_{1:S}$：完整已知的输入；$y_{1:T}$：待生成的输出。链式法则作用在 $y$ 上，$x$ 全程作条件变量、不进乘积。§2 的障碍被绕开的方式很直接：双向性只施加于「已经存在的对象」。

机制：encoder 栈把 $x_{1:S}$ 编成 $H^{\mathrm{enc}} \in \mathbb{R}^{S \times d}$（式 (1) 的全可见，一次前向）；decoder 每层含两个 attention 子层——causal self-attention（对已生前缀，2.4 的 KV cache 全套适用）与 cross-attention：

$$
q_t = u_t W^{Q}, \qquad K^{\mathrm{enc}} = H^{\mathrm{enc}} W^{K}, \qquad V^{\mathrm{enc}} = H^{\mathrm{enc}} W^{V} \qquad (12)
$$

$$
o_t = \sum_{s=1}^{S} \frac{\exp\left(\langle q_t, k_s^{\mathrm{enc}} \rangle / \sqrt{d_{\mathrm{head}}}\right)}{\sum_{j=1}^{S} \exp\left(\langle q_t, k_j^{\mathrm{enc}} \rangle / \sqrt{d_{\mathrm{head}}}\right)} \, v_s^{\mathrm{enc}} \qquad (13)
$$

**符号**：$u_t \in \mathbb{R}^{d}$：decoder 层在位置 $t$ 的归一化后 stream（query 侧，来自已生前缀）；$K^{\mathrm{enc}}, V^{\mathrm{enc}} \in \mathbb{R}^{S \times d_{\mathrm{head}}}$（逐 head）：encoder 输出的投影，key/value 侧；求和遍历全部 $S$ 个输入位置。

```
enc-dec 数据流（decode 第 t 步，逐层，省略 B/H 维）
x_{1:S} ── encoder (M≡0, 一次前向) ──► H^enc [S, d]
                                          │  W^K, W^V（每层一份参数）
                                          ▼
                              K^enc, V^enc [S, d_head] ────┐  一次计算，
                                                           │  decode 全程复用
y_{<t} ── causal self-attn（KV cache，2.4）──► u_t ──W^Q──► q_t
                                                           │
                o_t = softmax(q_t K^encᵀ/√d) V^enc  ◄──────┘  每步读同一份
```

结构要点三条：

1. **K/V 来自 encoder，一次计算、全程复用。** $K^{\mathrm{enc}}, V^{\mathrm{enc}}$ 是定长 $S$ 的张量，prefill 时算一次，decode 每步读同一份——这是一种 cache，但性质与 self-attention 的 KV cache 相反：**它不随 $t$ 增长**。仿 2.4 式 (9) 的账：cross 部分每序列 $2 L_d H d_{\mathrm{head}} S b$ bytes 是常数；self 部分 $2 L_d H d_{\mathrm{head}} t b$ 随生成步线性涨。代 T5-base 字段（$L_d$ = 12、$H$ = 12、$d_{\mathrm{head}}$ = 64、bf16、$S$ = 512）：cross cache $\approx 1.8 \times 10^{7}$ bytes（约 18 MiB）恒定；self cache 在 $t$ = 512 时同量级，此后线性超过它。
2. **cross-attention 无 mask。** $x$ 整个是给定条件，可见集恒为 $\{1, \ldots, S\}$；本章唯一活着的 mask 是 decoder self-attention 的下三角。与 2.4 §2 对照：enc-dec 的可见性拼合恰是 prefix-LM 的块状结构——encoder 侧全可见对应 prefix 块、decoder 侧下三角对应续写区。二者是同一可见性结构的两种参数化：prefix-LM 用一个栈一份参数兼任两区，enc-dec 给 prefix 区独立的参数与深度。
3. **逐层交叉。** decoder 每层有独立的 $W^{K}, W^{V}$ 作用于同一个 $H^{\mathrm{enc}}$——每层以自己的 query 视角重读输入，浅层与深层的读法可以分化（表面对齐与语义对齐分层）。

T5（[Raffel et al., 2019](../references.md)，arXiv:1910.10683，[A]）是这个形态的标准参照：enc/dec 对称各 12 层（base 口径，总参约 2.2 亿），翻译、摘要、分类、回归全部统一成 text-to-text 的条件生成；它的预训练 objective（span corruption）在 3.1。

## 8. 参数分配与 token 效率的账

同宽度同深度的对照可以把账算干净。记一层 self-attention + FFN 的参数为 $12 d^{2}$（attention $4d^{2}$ + FFN $8d^{2}$，$d_{\mathrm{ff}} = 4d$，忽略 bias/LN），decoder 层多一组 cross-attention 故为 $16 d^{2}$。12+12 的 enc-dec 总层参数 $= 12 \times 12d^{2} + 12 \times 16d^{2} = 336 d^{2}$；对齐同参数的 decoder-only 为 28 层。

**生成侧 FLOPs。** 每生成一个 token 只过 decoder 栈：参数项 $2 \times 192 d^{2}$（1.2 式 (2) 的每 token $2N$ 口径），对照 28 层 decoder-only 的 $2 \times 336 d^{2}$——decode 参数项约为 $0.57$ 倍。直觉「减半」被 cross-attention 子层吃掉一块：若只看 self-attention 骨架（$12 \times 12d^{2}$ 对 $24 \times 12d^{2}$）才精确减半。encoder 的一次性编码成本（$2 \times 144 d^{2} \times S$ 量级）摊到整条输出上，输出越长越可忽略。

**理解侧的表征密度。** causal mask 下位置 $i$ 的表征只暴露于 $i$ 个 token，整序列平均暴露 $S/2$；双向栈每位置恒见全部 $S$ 个——同为 12 层，encoder 每参数的上下文暴露量约为 decoder-only 读同一输入时的 2 倍。这是「理解任务的参数效率双向栈更高」的粗口径来源：参数相同，每 token 表征的信息进账翻倍。

两笔账合读：任务的输入与输出可分离时，enc-dec 把「读」的预算放在全暴露的双向栈、「写」的预算放在串行成本打 6 折的因果栈，两头都比单一混合栈划算。反方向同样成立：输入与输出同源同链（开放续写、对话）时，decoder-only 让 prompt 与生成共享同一份前缀计算（KV cache 直接续写），分离反而只剩开销——那条线的展开在 3.3。

## 9. 何时 enc-dec 占优：固定输入任务与 Whisper 活例

判据一句话：**输入在生成开始前完整已知、且与输出不同链**。

- 机器翻译：源句完整固定，双向编码合法；输出走因果链。
- 摘要：文档固定，输出短、decode 步少，编码质量主导质量上限——双向栈的信息密度直接兑现。
- 语音识别（ASR）：输入是声学特征帧序列、输出是文本——「读」与「写」连模态都不同，分工无法合并进一个栈。

Whisper（[Radford et al., 2022](../references.md)，arXiv:2212.04356，[A]；官方 repo [openai/whisper](../references.md)，[A]）是活例：encoder 双向吃 mel 频谱帧（定长 30 秒窗），decoder 因果地吐文本 token 与任务控制 token（语言标记、转写/翻译、时间戳）。cross-attention 在这里承担显式的模态桥接：每个文本位置的 query 向全部声学帧的 $K^{\mathrm{enc}}, V^{\mathrm{enc}}$ 取证——式 (12)(13) 一字不改，只是 $H^{\mathrm{enc}}$ 的每一行来自声学帧而非文本 token。前端、多任务格式与 CTC 路线的对照走读在 10.2；本章只取结构例证：ASR 把「输入固定、输出变化」推到极致（输入连续、输出离散、长度对齐事先未知），enc-dec 仍是标准答案——这是该形态没有随 decoder-only 浪潮消失的机制原因。

## 10. ModernBERT：encoder 的现代化形态

BERT 的架构字段停在 2018 年的零件上（可学绝对位置 embedding、每层全 global attention、padding 批处理）。ModernBERT（[Warner et al., 2024](../references.md)，arXiv:2412.13663，[A]）保持 encoder-only 形态与 MLM objective 不变，把零件全部换成 Part 2 的现代件。以下字段取自 config（[ModernBERT-base/large config 与官方 repo](../references.md)，[A]，hf-mirror 实测 2026-11-11）：base 为 22 层、hidden 768、12 heads、intermediate 1152、vocab 50368、ctx 8192、`tie_word_embeddings=true`、`attention_bias=false`；large 为 28 层、hidden 1024、16 heads、intermediate 2624。三个机制逐一看。

**(1) RoPE 双 base。** config 里 `position_embedding_type="absolute"` 是残留字段——实际位置编码是 RoPE（`rope_theta` 字段为证），读 config 勿被误导。且 global 层与 local 层配不同 base：`global_rope_theta=160000`、`local_rope_theta=10000`。按 2.5 的波长账：base 越大、低频分量的波长越长、远程相对位置的相位差越缓——要服务 8192 全程依赖的 global 层配大 base，总窗宽 128（每侧 64）的 local 层用小 base 保高频分辨。同一模型内位置编码按层职责分配，这是 2.5 的零件在 encoder 侧的精细化用法。

**(2) local/global 交替。** `global_attn_every_n_layers=3`：每 3 层一个 full attention（式 (1) 全可见），其余层为滑窗 local（`local_attention=128`，口径为总窗宽 128 token——论文原文 "a 128 token, local sliding window attention"；HF `modeling_modernbert.py` 中 `sliding_window = local_attention // 2`，即每位置见左右各 64）。BERT 时代每层都付 $O(T^{2})$ 的 global 账；ctx 拉到 8192 后 attention 项 $2LT^{2}d$（1.2 式 (6)）不再可忽略。交替结构是 sparse attention 谱系（4.2）在 encoder 侧的实例——SWA 堆叠感受野 $L \cdot w$ 的推导同样适用，global 层周期性把局部信息混回全局；与 decoder 侧 Gemma 3 的 5:1 local:global（4.2）同族，比例不同。

**(3) 反填充（unpadding）。** 变长序列批处理的旧方案是 pad 到 batch 最大长度、attention mask 屏蔽 padding 位——padding 位置照样吃 FLOPs。unpadding 把 batch 内全部序列首尾拼接成单条长序列（shape $[\sum_i T_i, d]$），用边界标记（cu_seqlens / position ids 复位）让 attention 只在各序列内部发生（FlashAttention 的 varlen 接口，4.1）。attention 项从 $B \cdot T_{\max}^{2}$ 降到 $\sum_i T_i^{2}$，序列长度方差大的批（检索、分类流量）收益直接；逐 token 的参数项不变。对 encoder 的主战场（嵌入服务、rerank——吞吐敏感、长度参差），它与交替 attention 的组合是实测提速的主要来源。

```
ModernBERT-base 结构一瞥（22 层，global 周期 3，示意）
┌─────────────────────────────────────────────────┐
│ layer 0   global attn（M≡0，RoPE θ=1.6×10⁵）    │ ← 全序列混合
│ layer 1   local  attn（总窗 128，RoPE θ=10⁴）   │
│ layer 2   local  attn（总窗 128，RoPE θ=10⁴）   │
│ layer 3   global attn                            │   …周期重复
│  ...                                              │
│ layer 21  global attn                            │
└─────────────────────────────────────────────────┘
输入：batch 内序列 unpad 拼接 [ΣᵢTᵢ, 768]，边界由 cu_seqlens 给出
输出：同形表征 → MLM head（与输入 embedding 权重绑定，vocab 50368）
```

走完一遍收一句：形态没变——$M \equiv 0$、MLM、读出式接口；变的是每个零件。encoder 的「现代化」是把 Part 2 的组件库整体移植，这反向说明 Part 2 的逐件分析是形态无关的。

## 总结

| 对象 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| encoder-only | $M \equiv 0$、$V(i) = \{1..T\}$，式 (1)(2) | 产品是整个 $H^{(L)}$，不是生成分布 |
| 不可自回归生成 | §2 命题 (i)–(iv) | 无链式因子、无增量计算、占位符无效、迭代精修是另一对象；causal mask 是可生成约束下的饱和解 |
| 输出接口 | [CLS] 式 (4)、mean pooling 式 (5)、token head 式 (6) | 全是读出；MLM 头本身是式 (6) 特例（weight tying） |
| NSP | 式 (7)，$[CLS]$ 二分类 | 任务混淆：主题信号吸干连贯性；RoBERTa 消融：净贡献 ≤ 0；ALBERT 的 SOP 是机制修正 |
| ALBERT | 式 (8) $Vd \to Ve + ed$（23.4M→4.0M）；式 (9) 层共享（84.9M→7.1M） | 110M → ~12M 量级；代价 = 逐层特化被牺牲 |
| ELECTRA | 式 (10)，$\lambda$ = 50 | 监督密度 ×6.7；非对抗（采样隔断梯度），G 训完即弃 |
| cross-attention | 式 (11)–(13) | K/V 来自 encoder、一次计算全程复用：定长 cache；无 mask |
| 参数/token 账 | decoder 层 $16d^{2}$ vs self-only 层 $12d^{2}$ | decode 参数项 ≈ 0.57×；理解侧暴露量 ×2 |
| 占优场景 | 翻译 / 摘要 / ASR（Whisper，10.2 详读） | 输入完整已知、与输出不同链 |
| ModernBERT | RoPE 双 $\theta$（$1.6 \times 10^{5}$ / $10^{4}$）、3:1 local（总窗宽 128，每侧 64）/global、unpadding | 形态不变零件全换；8192 ctx；config 的 absolute 字段是残留 |

## 练习

1.（推导）补全 §2 命题 (i) 的细节：对单层 $M \equiv 0$ 的 attention，写出 $z_t$ 对 $x_{t+1}$ 的依赖链（经 $v_{t+1}$ 与经 $s_{t,t+1}$ 两条路径），证明 $z_t$ 与 $x_{t+1}$ 无关当且仅当 $\alpha_{t,t+1} = 0$；进而说明「把未来位置填常量」为何仍得不到 $P(x_t | x_{<t})$——常量占位依然是条件的一部分。
2.（计算）ALBERT 参数账：取 BERT-base 字段 $V$ = 30522、$d$ = 768、$L$ = 12、$d_{\mathrm{ff}}$ = 3072、$e$ = 128，忽略 LN 与 bias。(a) 分别算 embedding $Vd$ 与 $Ve + ed$；(b) 层参数 12 层不共享 vs 共享（attention $4d^{2}$ + FFN $2 d \, d_{\mathrm{ff}}$）；(c) 总量比值。再把 $d$ 升到 1024、$e$ 不变重算 (a)(c)，说明 embedding 分解的削减率随 $d$ 增大如何变化。
3.（计算）cross-attention cache 账：T5-base 口径 $L_d$ = 12、$H$ = 12、$d_{\mathrm{head}}$ = 64、bf16。(a) $S$ = 512 时 cross K/V 每序列多少字节；(b) self-attention cache 在 $t$ = 512 与 $t$ = 4096 各多少；(c) $t \to \infty$ 时 cross 占总 cache 的比例极限是什么；(d) 长文档摘要（$S$ = 8192、$T$ = 1024）下 cross 与 self 哪份大？
4.（概念）ELECTRA 的 $\lambda$ 与 $G$ 的尺寸：$L_G$ 的支撑集是 15% 位置、$L_D$ 是 100%——若 $\lambda$ = 1，总梯度被哪侧主导？论文取 $\lambda$ = 50 且最终交付 $D$，从「谁是学生、谁是脚手架」解释这个数量级的方向。再讨论：$G$ 训练到接近完美（替换几乎不可检测）时 $L_D$ 会发生什么，由此说明 $G$ 刻意做小的机制理由。

## 参考

- [Devlin et al., 2018](../references.md)：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，arXiv:1810.04805，[A]（key: bert）——$M \equiv 0$ 形态、MLM + NSP 原始定义、[CLS]/[SEP] 输入模板
- [Liu et al., 2019](../references.md)：RoBERTa: A Robustly Optimized BERT Pretraining Approach，arXiv:1907.11692，[A]（key: roberta）——NSP 移除消融的出处
- [Lan et al., 2019](../references.md)：ALBERT: A Lite BERT for Self-supervised Learning of Language Representations，arXiv:1909.11942，[A]（key: albert）——embedding 分解、跨层参数共享、SOP
- [Clark et al., 2020](../references.md)：ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators，arXiv:2003.10555，[A]（key: electra）——RTD 目标、$\lambda$ = 50、同算力样本效率对照
- [Google, ELECTRA 官方 repo](../references.md)：[A]（key: electra-repo）——RTD generator/discriminator 实现
- [Raffel et al., 2019](../references.md)：Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer，arXiv:1910.10683，[A]（key: t5）——enc-dec 标准参照、text-to-text 统一
- [Radford et al., 2022](../references.md)：Robust Speech Recognition via Large-Scale Weak Supervision，arXiv:2212.04356，[A]（key: whisper）——ASR 的 enc-dec 活例（10.2 详读）
- [Warner et al., 2024](../references.md)：Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder（ModernBERT），arXiv:2412.13663，[A]（key: modernbert）
- [Answer.AI, ModernBERT-base/large config 与官方 repo](../references.md)：[A]（key: modernbert-config）——config 字段实测口径（2026-11-11）
