# 4.7 机制可解释性基础（Mechanistic Interpretability Primer）

> 本章把「读懂一台训练好的 Transformer」落成四件可计算的东西：以 residual stream 纯加法结构为地基的 logit lens（中间层读出）、superposition 的叠加几何（ $n > d$ 时近似正交编码的干扰账）与其换基工具 SAE（字典学习 $h \approx Dz$）、circuit 方法论（induction head 回链与 IOI 全流程）、activation patching 的因果语义（do 算子与恢复率），以方法工具箱对照表与结论可信度分级收尾 | 前置：2.1（residual stream 式 (1)(2)）、2.7（norm 语义）、3.3 §6–§8（induction heads、Pythia 训练相变） | 锚点：TransformerLens [A]、SAELens [A] + Gemma Scope SAEs（arXiv:2408.05147）[A]、Pythia（arXiv:2304.01373）[A]；IOI（arXiv:2211.00593）[A]、Toy Models of Superposition [B]、induction heads [B]

## 1. 地基：residual stream 是 read-modify-write 总线

2.1 式 (2) 把整机输出写成纯加和 $x^{(L)} = x^{(0)} + \sum_{\ell} a_{\ell} + \sum_{\ell} f_{\ell}$ 。本章全部方法都从这条加法结构里长出来。先把它改写成 read-modify-write 的通用循环：每个组件（一个 attention head、一个 MLP）从 stream 读入（经 norm 与各自投影），把输出**加回** stream；没有任何组件能覆盖或擦除别人的写入，对冲（写入一个反向量）是唯一可用的「删除」。

```
logit lens（§2）：       ▼          ▼                    ▼
                      W_U x⁽¹⁾    W_U x⁽²⁾    …    W_U · RMSNorm(x⁽ᴸ⁾) = logits
                        │          │                    │
x⁽⁰⁾ ═► ═► ═► ═► x⁽¹⁾ ═► ═► x⁽²⁾ ═► ═► … ═► ═► x⁽ᴸ⁾ ═► ═► logits
                      ▲          ▲                    ▲
                  写入 a₁     写入 f₁             写入 a_L, f_L   （只加不覆盖）
                  └─ ablation = 从和中删去一项；patching = 把一项换成它在别的输入下的取值（§6）
```

从这条结构直接读出三个可操作推论（2.1 §2 已登记过第三条）：

1. 任意中间态 $x^{(\ell)}$ 乘上同一个 $W_U$ 仍是一个合法的词表读出 → logit lens（§2）；
2. 从和中删去某项 $a_\ell$ 即「关掉」该组件 → ablation；
3. 把某项替换成它在另一个输入下的取值即一次受控干预 → activation patching（§6）。

记号约定沿用 2.1： $x^{(\ell)} \in \mathbb{R}^{B \times T \times d}$ ；本章多在单位置讨论，记 $h \in \mathbb{R}^{d}$ 为某位置某层的 stream 向量（文献常写作 $h_\ell$ ）。组件粒度取「一个 head 的输出（ $W_O$ 之后、残差加之前）或一个 MLP 的输出」，这是干预的最小单位。

## 2. Logit lens：中间层的词表投影

Logit lens 的操作只有一步：把中间层 stream 直接乘输出头。第 $\ell$ 层处的读出

$$
\tilde{p}_{\ell} = \mathrm{softmax}\left( W_U\thinspace x^{(\ell)} \right) \qquad (1)
$$

**符号**： $\tilde{p}_{\ell} \in \Delta^{V-1}$ ：第 $\ell$ 层读出的词表分布； $W_U \in \mathbb{R}^{V \times d}$ ：输出头（untied；tied 时即 $E$ ，2.2）； $x^{(\ell)}$ ：第 $\ell$ 层后的 stream（2.1 式 (1)）。

出处注记：该方法没有正式论文（源头是 nostalgebraist 2020 年的博客文章），本章按社区通行定义使用、不列引用条目；它作为测量工具的标准化由 TransformerLens 一类库完成（[Nanda & Bloom](../references.md)，[A]）。

两个细节决定读数的可信度：

- **final norm 的取舍**。2.1 式 (1) 里 $W_U$ 之前还有一道 final RMSNorm。中间层读出带不带它是两种约定：不带 norm 忠实于「原样 stream」；带上（ $\mathrm{softmax}(W_U\thinspace \mathrm{RMSNorm}_f(x^{(\ell)}))$ ）校准了 stream 逐层增长的尺度，但 RMSNorm 的增益向量是对终态学出的，套在中间层上是近似。读数的绝对值差一层校准，排序结论通常稳健。
- **$\tilde{p}_{\ell}$ 的语义**。它不是「第 $\ell$ 层时模型的预测」——模型没有中途做预测的机会——而是「若后续层全部变成恒等映射，此刻的输出将是什么」。

用途三件：(i) 逐层追踪目标 token 的 logit / rank 曲线，定位「预测成形」的层段；(ii) 定位特定信息（如 IOI 中被复制的名字，§5）首次可被 $W_U$ 读出的位置，为 patching 圈定候选层；(iii) 对照终层读出，检测「已写入但被后续层对冲掉」的信号——纯加法结构里对冲是唯一的删除，必然留下痕迹。

局限三条，每条对应后文一个方法：

1. **坐标系局限**。式 (1) 只依赖 $x^{(\ell)}$ 落在 $W_U$ 行空间里的分量；以与全部 $W_U$ 行近正交的方向存储的信息完全不可见——superposition（§3）保证这种方向大量存在。修正路线是给每层学一个仿射翻译器再投影、校准层间分布漂移（文献称 tuned lens），本章不展开。
2. **分布局限**。中间层 stream 不在 $W_U$ 的训练输入分布上，极端 logit 是外推读出，不能当概率解释。
3. **因果局限**。读得到 ≠ 在用：某层读数与最终输出相关，不证明该层计算导致了输出。从相关到因果是 §6 的 do 算子干的事。

## 3. Superposition： $n > d$ 的叠加几何

neuron 不可读的机制原因：模型要编码的特征数 $n$ 远大于 stream 维数 $d$ ，而 $d$ 维空间里的正交方向只有 $d$ 个。Toy Models of Superposition（[Elhage et al., 2022](../references.md)，transformer-circuits.pub，[B]）把这个处境做成可解析的玩具模型：真实特征向量 $x \in \mathbb{R}^{n}$ （ $x_i$ 为特征 $i$ 的强度，稀疏激活），模型只有 $d < n$ 维可用，编码-重建共用一组方向：

$$
h = Wx, \qquad \hat{x} = \mathrm{ReLU}\left( W^{\top} h + b \right), \qquad W \in \mathbb{R}^{d \times n} \qquad (2)
$$

训练目标按特征重要性加权：

$$
\mathcal{L} = \sum_{i=1}^{n} I_i\thinspace \mathbb{E}\left[ \left( x_i - \hat{x}_i \right)^{2} \right] \qquad (3)
$$

**符号**： $W_{:i} \in \mathbb{R}^{d}$ ：特征 $i$ 的存储方向（ $W$ 的第 $i$ 列）； $b \in \mathbb{R}^{n}$ ； $I_i$ ：特征 $i$ 的重要性权重。编码与解码共用 $W$ （跨特征耦合全在 $W^{\top}W$ 里）是玩具设定，真实模型没有这条约束，但叠加几何不变。

逐特征展开重建（先忽略 ReLU 与 $b$ ，看信号结构）：

$$
\hat{x}_i = \underbrace{\lVert W_{:i} \rVert^{2}\thinspace x_i}_{\text{self}} + \underbrace{\sum_{j \neq i} \langle W_{:i}, W_{:j} \rangle\thinspace x_j}_{\text{interference}} \qquad (4)
$$

两项各有内容。self 项： $\lVert W_{:i} \rVert < 1$ 即「该特征以缩短的向量存储」，自身重建被压缩——superposition 里范数是特征分到的带宽。interference 项：方向两两不正交造成的交叉读出，是叠加的代价。

几何侧的容量账： $d$ 维单位球上两个随机方向的内积近似 $\mathcal{N}(0, 1/d)$ ； $n$ 个随机单位向量的最大成对内积按 $\sim \sqrt{4 \ln n / d}$ 增长（对 $\sim n^{2}$ 个成对事件取 max 的集中估计，常数依赖尾界口径）。固定可容忍干扰 $\epsilon$ ，可容纳的特征数 $n \sim \exp(\epsilon^{2} d / 4)$ 量级—— $n$ 对 $d$ 是指数的，「近似正交」远比「正交」能装。

但叠加真正成立靠的是稀疏。设各特征以概率 $S$ 独立激活、激活时二阶矩 $\mathbb{E}[x^{2}]$ ，式 (4) 干扰项的方差

$$
\mathrm{Var}\left[ \sum_{j \neq i} \langle W_{:i}, W_{:j} \rangle\thinspace x_j \right] \approx S\thinspace \mathbb{E}\left[ x^{2} \right] \sum_{j \neq i} \langle W_{:i}, W_{:j} \rangle^{2} \sim \frac{(n-1)\thinspace S}{d}\thinspace \mathbb{E}\left[ x^{2} \right] \qquad (5)
$$

**符号**： $S$ ：单特征激活概率；第二步用随机方向的 $\mathbb{E}[\langle W_{:i}, W_{:j} \rangle^{2}] = 1/d$ 。判据： $nS \ll d$ 时干扰淹没不了信号——同时在线的特征数 $\sim nS$ 才是有效负载，稀疏把负载压回 $d$ 的量级以下。玩具模型里确能扫出随稀疏度从 monosemantic（一方向一特征、正交存储）到 superposition 的相变 [B，玩具设定]。

**polysemanticity 是式 (4) 的用户界面后果**：neuron $k$ 的激活 $h_k = \sum_i W_{ki} x_i$ 是多个特征的混合坐标；特征一旦以叠加存储，单个 neuron 必然对多个无关概念族同时响应。结论不是「neuron 有时碰巧多义」，而是「neuron 基与特征基系统性不对齐」。要单义坐标就得换基——这是 §4 的 SAE 要干的。同一条几何也解释了 §2 的局限 1：与 $W_U$ 行空间近正交的叠加分量，logit lens 天然读不到。

## 4. SAE：把 stream 换成稀疏特征坐标

若特征以叠加方向存储，找回它们的直接办法是字典学习：学一个过完备字典，把 stream 向量分解成稀疏组合。稀疏自编码器（sparse autoencoder，SAE）：

$$
z = \mathrm{ReLU}\left( W_{\mathrm{enc}} h + b_{\mathrm{enc}} \right), \qquad \hat{h} = Dz + b_{\mathrm{dec}}, \qquad D \in \mathbb{R}^{d \times n},\quad n \gg d \qquad (6)
$$

$$
\mathcal{L}_{\mathrm{SAE}} = \left\lVert h - \hat{h} \right\rVert_{2}^{2} + \lambda \left\lVert z \right\rVert_{1} \qquad (7)
$$

**符号**： $z \in \mathbb{R}^{n}$ ：稀疏特征激活， $z_i$ 读作「特征 $i$ 在此位置的强度」； $D$ ：字典，列 $D_{:i}$ 归一化为单位范数（防止用列范数膨胀来逃逸 L1 惩罚），希望对齐式 (2) 的 $W_{:i}$ ； $n$ 常见为 $d$ 的数十到数百倍。式 (7) 两项分工：重构项保真、L1 项压稀疏；沿 $\lambda$ 扫出一条 Pareto 前沿，评估时点按 $L_0 = \lVert z \rVert_0$ （每位置平均激活特征数）归档。

```
h ∈ R^d ─► encoder（ReLU / TopK / 门控）─► z ∈ R^n（稀疏：L0 ≪ n）
  ▲                                            │
  └────  decoder D（列 = 特征方向，单位范数）◄───┘
         loss = ‖h − ĥ‖² + λ‖z‖₁（L1 族，式 (7)）
```

L1 的已知副作用：惩罚正比于幅度，把全部非零 $z_i$ 系统性压小（shrinkage），重构幅度随之偏低（练习 3 推导标量版）。两个变体族针对这一点：

- **TopK（k-SAE）**：去掉 L1，硬约束稀疏度—— $z = \operatorname{TopK}(W_{\mathrm{enc}} h + b_{\mathrm{enc}})$ ，每位置只保留最大的 $k$ 个分量、其余置零。 $L_0 \equiv k$ ，超参 $\lambda$ 消失，幅度不被压缩；代价是稀疏度对每个位置一视同仁。
- **门控（gating）/ JumpReLU**：把「是否激活」与「激活多强」拆成两条路径，

$$
z = \operatorname{H}\left( W_g h + b_g \right) \odot \mathrm{ReLU}\left( W_m h + b_m \right) \qquad (8)
$$

**符号**： $\operatorname{H}$ ：逐分量单位阶跃（ $\operatorname{H}(u) = 1$ 当 $u > 0$ ，否则 $0$ ），梯度经 straight-through 估计器回传；稀疏惩罚只加在门路径上，magnitude 路径（ $W_m$ ）不受 L1 压缩。JumpReLU 是单投影特化：共用 $a = W_{\mathrm{enc}} h + b_{\mathrm{enc}}$ 、逐特征学阈值 $\theta$ ， $z = a \odot \operatorname{H}(a - \theta)$ 。Gemma Scope 采用的就是 JumpReLU 形态（[Lieberum et al., 2024](../references.md)，arXiv:2408.05147，[A]）。

实证锚点。Gemma Scope [A]：在 Gemma 2（9B 与 2B）全部层、逐层三个挂载点（residual stream、attention 输出、MLP 输出）训练 JumpReLU SAE，宽度档从 16K 到 1M 特征、多档稀疏度，全套开放；配套训练与分析库 SAELens（[Bloom et al.](../references.md)，[A]，decoderesearch/SAELens）。其 monosemanticity 实证：训出的 SAE 特征中「对单一概念族选择性响应」的比例远高于以 neuron 为单位的基线——单义坐标是学出来的，不是模型的原生坐标。

三条局限随工具一起交付：

1. **特征分裂**（feature splitting）：加宽 $n$ 时，粗特征被拆成更细粒度的多个——「特征」的粒度本身是 $n$ 与 $\lambda$ 的函数，不是模型的内在量。
2. **重构 ≠ 计算**：字典只被要求重构 $h$ ，可能学到重构捷径；SAE 特征是否就是下游计算实际读取的方向，要靠 §6 的干预实验另证。
3. **覆盖率未知**：重构误差非零，部分行为相关方向可以不被字典张成——「特征清单完备」目前没有判据。

## 5. Circuits：从组件到算法

circuit 的定义：实现某个行为子功能的组件子图——节点是 head / MLP，边是「前者的写入被后者读出」。3.3 §6 的 induction head 就是最小实例：两个 head 跨两层，靠组合双线性型（3.3 式 (7)）完成 prefix matching + copying，单看任一层该功能都不存在——circuit 的意义恰在这种跨组件组合。

方法论全流程的范本是 IOI（indirect object identification）circuit（[Wang et al., 2022](../references.md)，arXiv:2211.00593，[A]）。任务模板：「When Mary and John went to the store, John gave a drink to ___」→ 应填 Mary（IO = 间接宾语；S = John，句中重复出现的那个名字）。对象模型 GPT-2 small（12 层 × 12 head = 144 个 head；[Radford et al., 2019](../references.md)，[W]）。论文的结论：26 个 head 归入 7 类功能角色，主干链路三类：

```
duplicate token heads（早层）：在重复名 John 的第二次出现处标记「此 token 出现过」
        │ 写入 stream
        ▼
S-inhibition heads（中层）：在 END 位置写入抑制信号，使下游对重复名 S 的 attend 质量下降
        │
        ▼
name mover heads（晚层，L9H6 / L9H9 / L10H0 档）：从 IO（Mary）位置拷贝名字表征到 END，
        经 W_U 直接抬升 Mary 的 logit（logit lens 逐层可见这条抬升）
```

外加 previous token heads 与 induction heads 供流位置与重复信息（3.3 §6 的回路在这里复用为原料），以及两组对照角色：backup name movers（主 name mover 被 knock out 后补上、行为部分恢复）与 negative name movers（压低 IO logit 的对冲组件）。

这个案例的教学价值不在 26 这个数字，在方法论的四个动作——每个都是 §6 工具的实例：

1. **定指标**：logit difference $\mathrm{LD} = \mathrm{logit}(\mathrm{IO}) - \mathrm{logit}(\mathrm{S})$ ，把行为压成一个标量；
2. **造腐蚀**：把三个名字换成随机名（破坏关键变量、保留模板），建立 clean / corrupted 双跑对照；
3. **逐组件 patching**：定位哪些组件的激活携带行为相关信号（恢复率公式即 §6 式 (10)）；
4. **完备性压力测试**：knockout 主组件、观察是否有 backup 接管。backup name movers 的存在直接说明「组件必要性」结论对 ablation 协议敏感：置零某 head 后行为仍在，不等于该 head 无功能。

工具归属注记：IOI 的实验在 Easy-Transformer（TransformerLens 的前身）上完成，现行复现与走读用 TransformerLens [A]；沿训练轴测量 circuit 成形（induction head 相变，回链 3.3 §8）的仪器底座是 Pythia 的 154 个公开 checkpoints（[Biderman et al., 2023](../references.md)，arXiv:2304.01373，[A]）。

## 6. Activation patching：从相关到因果

前面所有观测工具（logit lens、probing、SAE 重构）给出的都是相关证据。patching 把观测换成干预。记号： $f$ 为模型， $h_i(x)$ 为组件 $i$ 在输入 $x$ 下的激活，do 算子表示「前向中把 $h_i$ 强制设为 $v$ ，其余照常」：

$$
f\left( x;\thinspace \mathrm{do}\left( h_i \leftarrow v \right) \right) \qquad (9)
$$

这不是近似记号而是精确的执行定义：residual stream 的加法结构保证组件输出是可独立替换的对象（§1 推论 2/3），干预只沿 $h_i$ 的下游传播。这套协议（因果追踪，causal tracing）的方法论定型见 IOI 论文（[Wang et al., 2022](../references.md)，[A]）。

标准协议（corrupt → patch-clean）：clean 输入 $x_c$ 与 corrupted 输入 $x_r$ 只差关键变量；行为标量指标 $m$ （IOI 用 LD）。组件 $i$ 的恢复率

$$
\mathrm{rec}(i) = \frac{ m\left( f\left( x_r;\thinspace \mathrm{do}\left( h_i \leftarrow h_i(x_c) \right) \right) \right) - m\left( f(x_r) \right) }{ m\left( f(x_c) \right) - m\left( f(x_r) \right) } \qquad (10)
$$

**符号**：分子：corrupted 跑里只修复组件 $i$ 后恢复的指标量；分母：全恢复量，归一化使 $\mathrm{rec}$ 大致落在 $[0, 1]$ 。读法： $\mathrm{rec}(i) \approx 1$ → 组件 $i$ 单独携带该行为的充分信息； $\approx 0$ → 单独无关（不排除参与组合）。方向反过来——在 clean 跑里做 $\mathrm{do}(h_i \leftarrow 0)$ （zero ablation）或 $\mathrm{do}(h_i \leftarrow \bar{h}_i)$ （均值 ablation， $\bar{h}_i$ 取腐蚀分布上的均值）——测的是必要性。两个方向都要测：IOI 的 backup 教训（§5）正是「必要性为假 + 充分性为真」的组合实例。

两个细化：

- **path patching**：把干预单位从节点换成边。sender $s$ 的输出只沿「到 receiver $r$ 输入」的那条路径替换为 clean 值、其余路径保持 corrupted，测的是 $s \to r$ 这条边是否在传行为相关信号——circuit 的边结构就是这样钉出来的；代价是实验数随节点对数平方增长。
- **probing 的对照定位**：在冻结激活上训线性探针 $\hat{y} = w^{\top} h$ ，测「该属性在激活里是否线性可解码」。探针给出的是相关证据的下限（线性探针都读得出，则信息必在）；但可解码 ≠ 被下游使用——探针自己能读出模型从不读取的方向（superposition 保证这种方向存在）。工具链的分工因此是：probing 圈假设，patching 定因果。

## 7. 方法工具箱对照表

§2–§6 的工具压进一张表：

| 方法 | 操作 | 测什么 | 证据类型 | 粒度 | 回答不了 |
| --- | --- | --- | --- | --- | --- |
| logit lens（§2） | $W_U$ 投影中间层，式 (1) | 输出坐标系下的中间读数 | 相关 | 层 × 位置 | 非 $W_U$ 方向的信息；读数是否被用 |
| probing | 冻结激活上训线性探针 | 属性的线性可解码性 | 相关（下限） | 方向 | 模型计算是否使用该信息 |
| zero / mean ablation | $\mathrm{do}(h_i \leftarrow 0 / \bar{h}_i)$ | 组件必要性 | 因果 | 组件 | 冗余接管（backup）掩盖真功能 |
| activation patching | $\mathrm{do}(h_i \leftarrow h_i(x_c))$ ，式 (10) | 组件对行为的充分贡献 | 因果 | 组件 × 位置 | 组件间的连接结构 |
| path patching | 逐边替换 | 组件间的因果边 | 因果 | 边 | 成本随节点对平方增长；间接通路 |
| SAE（§4） | 字典学习，式 (7)(8) | stream 的稀疏特征坐标 | 相关（因果地位需 patching 补） | 特征 | 学出的特征是否=模型在用的特征 |

使用顺序的惯例：lens / probing 提假设 → patching 定组件 → path patching 钉边 → SAE 换坐标系做特征级读数；一篇 circuit 论文就是这四者按序的组合（§5 的 IOI 即实例）。

## 8. 诚实边界：结论可信度分级

本章出现过的结论按当前证据强度分三档（分级依据：复现广度、方法学依赖度、外推距离）：

| 档位 | 结论 | 依据与注脚 |
| --- | --- | --- |
| 高 | induction head 机制与其训练相变；IOI circuit 主干在 GPT-2 small 上成立 | 多团队复现 + 机制闭环（组合双线性型可解析验证，3.3 §6；Pythia 154 checkpoints 底座 [A]）；IOI 自带注脚：backup heads 说明「完备性」主张要打折 |
| 中 | superposition 是真实模型的编码策略；SAE 特征的单义性统计 | 玩具模型内严格成立（[B]），真实模型中是 linear representation hypothesis 假设 + 间接证据；Gemma Scope 的可解释性统计依赖评分口径（[A]，自述局限） |
| 低 / 开放 | 小模型 circuit 结论向 frontier 模型的迁移；SAE 特征清单的完备性；「一个特征 = 一个概念」的读法 | 无正面判据；已知反例模式：feature splitting、重构吸收（§4） |

Gemma Scope 与 transformer-circuits.pub 系列的自述局限可以压成一句：SAE 的目标函数只保证重构与稀疏，不保证与模型内部计算对齐——任何由 SAE 特征出发的因果断言，都必须再过一次 §6 的干预实验才算数。这也是本章工具链的排序原则：观测工具负责提出可检验的假设，因果工具负责宣判。

## 总结

| 内容 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| read-modify-write | 2.1 式 (2)： $x^{(L)} = x^{(0)} + \sum a_\ell + \sum f_\ell$ | 纯加法 → 可读、可删、可换（§1 三推论），全章地基 |
| logit lens | 式 (1)： $\tilde{p}_\ell = \mathrm{softmax}(W_U x^{(\ell)})$ | 输出坐标系读中间层；坐标系 / 分布 / 因果三局限；无正式出处（博客 2020） |
| superposition | 式 (4)(5)： $\hat{x}_i = \lVert W_{:i} \rVert^{2} x_i + \sum_{j \neq i} \langle W_{:i}, W_{:j} \rangle x_j$ ；干扰方差 $\sim (n-1)S/d$ | 随机方向容量 $n \sim \exp(\epsilon^{2} d / 4)$ ；稀疏判据 $nS \ll d$ ；polysemanticity = 换基问题 |
| SAE | 式 (7)： $\lVert h - Dz \rVert^{2} + \lambda \lVert z \rVert_1$ ；TopK；门控式 (8) | L1 shrinkage → TopK / JumpReLU；按 $L_0$ 归档；Gemma Scope 逐层三挂载点开放 |
| circuits | IOI：26 heads / 7 类（GPT-2 small，144 heads） | 定指标 → 造腐蚀 → patching → knockout 完备性；backup name mover 教训 |
| activation patching | 式 (9)(10)：do 算子与恢复率 | 充分性（patch-clean）与必要性（ablation）都要测；path patching 钉边；probing 只给相关下限 |
| 可信度分级 | §8 三档表 | 观测提假设、因果做宣判；SAE 因果断言须过 patching |

## 练习

1.（推导）从 2.1 式 (2) 出发，证明「把第 $k$ 层 MLP 置零」与「把第 $k$ 层 MLP 输出替换为它在输入 $x'$ 下的取值」都对应前向计算图上的良定义操作，并写出两者在加和形式下各自的编辑方式。若把 Pre-LN 换成 Post-LN（ $x \leftarrow \mathrm{LN}(x + f(x))$ ），这两个操作还良定义吗？（提示：对照 2.1 §2 末的 Post-LN 注记，纯加法展开是先决条件。）
2.（计算）superposition 的两条独立约束：取 $d = 4096$ 、 $\epsilon = 0.1$ ，用 $\sqrt{4 \ln n / d} \le \epsilon$ 估算随机方向可容纳特征数 $n$ 的量级；再取稀疏度 $S = 0.01$ ，用式 (5) 的判据 $nS \ll d$ 给出第二个上界。哪个先咬紧？说明「几何容量」与「稀疏负载」为什么必须分开算。
3.（推导）L1 的 shrinkage：对标量问题 $\min_z\thinspace (h - z)^{2} + \lambda |z|$ （ $h > 0$ ），证明解为软阈值 $`z^{*} = \max(h - \lambda/2,\ 0)`$ 。由此说明式 (7) 为何系统性低估特征幅度，以及 TopK 与门控变体各自在哪个环节取消这个偏差。
4.（代码 + 分析）用 TransformerLens 对 GPT-2 small 跑 IOI 协议（教学练习，测量口径以 TransformerLens 文档为准）：构造 20 条 Mary/John 模板与对应的随机名腐蚀版，定义 $\mathrm{LD} = \mathrm{logit}(\mathrm{IO}) - \mathrm{logit}(\mathrm{S})$ ，对第 9、10 层全部 head 做 activation patching，按式 (10) 的恢复率排序列出前 5 个 head，与论文报告的 name mover 位置（L9H6 / L9H9 / L10H0）对照；再对恢复率最高者做 zero ablation，报告 LD 下降幅度，并用 backup name mover 解释「ablation 后行为仍在」的可能原因。

## 参考

- [Olsson et al., 2022](../references.md)：In-context Learning and Induction Heads（transformer-circuits.pub，无 arXiv），[B]（key: induction-heads）——最小 circuit 与训练相变，回链 3.3 §6/§8
- [Elhage et al., 2022](../references.md)：Toy Models of Superposition（transformer-circuits.pub，无 arXiv），[B]（key: toy-models）——式 (2)–(5) 的玩具模型、稀疏度-叠加相图
- [Wang et al., 2022](../references.md)：Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small，arXiv:2211.00593，[A]（key: ioi）——IOI circuit（26 heads / 7 类）与 patching 方法论全流程
- [Lieberum et al., 2024](../references.md)：Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2，arXiv:2408.05147，[A]（key: gemma-scope）——JumpReLU SAE 套件、monosemanticity 实证与自述局限
- [Bloom et al.](../references.md)：SAELens（decoderesearch/SAELens），[A]（key: saelens）——SAE 训练与分析库
- [Nanda & Bloom](../references.md)：TransformerLens，[A]（key: transformerlens）——logit lens / patching 测量工具锚点
- [Biderman et al., 2023](../references.md)：Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling，arXiv:2304.01373，[A]（key: pythia）——训练轴 circuit 测量的 154-checkpoint 底座
- [Radford et al., 2019](../references.md)：GPT-2（IOI 的对象模型），[W]（key: gpt2）
