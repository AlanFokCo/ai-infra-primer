# 2.7 Normalization 与 Residual（Normalization and the Residual Stream）

> 本章回答一组相互咬合的问题：LayerNorm 的公式与两个不变性；为什么 BatchNorm 在序列建模里数学上就不成立（三条原因）；RMSNorm 去掉 re-centering 丢了什么；推导主线是 Pre-LN vs Post-LN 的梯度分析——LN Jacobian 谱范数引理（ $\lVert J_{\mathrm{LN}}(x) \rVert = O(\sqrt{d}/\lVert x \rVert)$）→ Xiong et al. Theorem 1 精确表述（初始化时最后一层 FFN 参数梯度：Post-LN $O(d\sqrt{\ln d})$ 与深度 $L$ 无关、Pre-LN $O(d\sqrt{\ln d / L})$ 随深度衰减）→ warmup 的数学必然性；残差侧给出梯度流恒等式 $\partial L / \partial x_l = (\partial L / \partial x_L)(I + \partial \sum f_i / \partial x_l)$ 的 identity 项推导；组件侧收尾于 QK-Norm（attention logits 的结构性封顶）、DeepNorm 缩放系数、logit soft-capping（仅 Gemma 2：attn 50.0 / final 30.0，Gemma 3 已弃用）与 zero-centered 增益 | 前置：2.1（residual stream 式 (1)(2)）、2.3（ $\sqrt{d_k}$ 链与 softmax 饱和 §3.4）、2.6（FFN 结构） | 锚点：Gemma 2 [A]（soft-capping 出处）、Gemma 3 [A]（QK-norm、pre+post RMSNorm）、Llama-3.1-8B config [A]（RMSNorm + Pre-LN 标准配方）、OLMo-2-7B [A]（q/k norm 代码走读）；Ba et al. [B]、Zhang & Sennrich [B]、Xiong et al. [B]、Wang et al. [A]、Henry et al. [B]

## 1. LayerNorm：公式与两个不变性

先登记「为什么必须有归一化」这个问题本身。2.1 式 (2) 说 stream 是 $2L+1$ 个向量之和：前向看，没有归一化时 stream 的二阶矩随层数漂移（§5.2 的随机游走口径给出 $\Theta(\sqrt{L})$ 增长），logits 的尺度跟着漂；反向看，本章 §5 的引理会证明归一化 Jacobian 的增益反比于输入范数——归一化器是 stream 上的自动增益控制（automatic gain control）：前向钉住输出二阶矩（ $\sqrt{d}$），反向按输入范数自动缩放梯度。本章全部结果都挂在这两条上。

层归一化（LayerNorm）对单个位置的特征向量做归一化（[Ba et al., 2016](../references.md)，arXiv:1607.06450，[B]）。取 $x \in \mathbb{R}^{d}$（residual stream 在某一位置上的切片，2.1 式 (1) 的逐位置口径）：

$$
\mathrm{LN}(x) = \gamma \odot \hat{x} + \beta, \qquad \hat{x} = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \qquad \sigma^{2} = \frac{1}{d} \sum_{i=1}^{d} \left( x_i - \mu \right)^{2} \qquad (1)
$$

**符号**： $\mu, \sigma^{2}$：沿特征维 $d$ 的均值与方差（实现中加 $\epsilon$ 防除零）； $\gamma, \beta \in \mathbb{R}^{d}$：逐特征可学增益（gain）与偏置； $\odot$：逐分量乘； $\hat{x}$：归一化核，满足 $`\sum_i \hat{x}_i = 0`$、 $\lVert \hat{x} \rVert_{2} = \sqrt{d}$。

LN 对上游表达施加两个精确的不变性（invariance），这是它对优化几何起塑形作用的结构来源。

**不变性一（输入的 scale/shift）**。对任意标量 $\alpha > 0$ 与 $c \in \mathbb{R}$，把 $x$ 换成 $x' = \alpha x + c \cdot 1_d$（ $1_d \in \mathbb{R}^{d}$：特征维全 1 列向量，沿用 1.1 的 $1_V$ 记法）：

$$
\mu' = \alpha \mu + c, \qquad \sigma' = \alpha \sigma, \qquad \hat{x}' = \frac{\alpha x + c \cdot 1_d - \alpha \mu - c}{\alpha \sigma} = \hat{x} \qquad (2)
$$

归一化核原样不动，LN 输出完全不变：整体缩放与整体平移被商掉。

**不变性二（上游权重矩阵的重参数化）**。设 LN 的输入来自线性层 $h = W x$（ $W \in \mathbb{R}^{d \times n}$）。两个参数方向不改变函数值：

$$
W' = \delta W \ (\delta > 0): \quad h' = \delta h; \qquad W'' = W + 1_d a^{\top}: \quad h'' = h + \left( a^{\top} x \right) 1_d \qquad (3)
$$

第一式归入式 (2) 的缩放半条，LN 不变；第二式里 $a \in \mathbb{R}^{n}$ 任意， $a^{\top} x$ 是标量， $h$ 被加上一个沿 $1_d$ 方向的标量倍平移，归入式 (2) 的平移半条，LN 同样不变。合起来：「整个矩阵的径向缩放」与「 $1_d a^{\top}$ 平移」这两个参数方向被 LN 商掉—— $\hat{h}$ 沿这两个方向分段常数，相应梯度分量恒为零（ $\gamma, \beta$ 之后的通路除外）。参数空间中不产生函数变化的方向被自动剔除，学习发生在商空间上；被商掉的尺度自由度由可学的 $\gamma$ 显式还回——还回多少交给数据决定，这正是 $\gamma$ 的存在理由。

一个直接推论：LN 之后的线性层，其权重范数的径向增长不改变函数——训练中 $\lVert W \rVert$ 的漂移是「自由」的，函数只感知 $W$ 的方向。这把权重范数与有效步长解耦：对 LN 保护的层，权重衰减（weight decay）的作用对象从函数本身变成有效学习率——范数越大，同一 $\eta$ 下函数方向的相对变化越小。偏置 $\beta$ 的角色对称：它是有意打破平移商的那只手，把被减均值吃掉的逐特征直流分量以可学形式还回。

## 2. 为什么序列建模不用 BatchNorm

批归一化（BatchNorm）的统计轴是 batch × 位置：对每个特征通道 $i$，

$$
\mu_{i}^{\mathrm{BN}} = \frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} x_{b,t,i}, \qquad \left( \sigma_{i}^{\mathrm{BN}} \right)^{2} = \frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} \left( x_{b,t,i} - \mu_{i}^{\mathrm{BN}} \right)^{2} \qquad (4)
$$

在序列建模里它有三条数学层面的硬伤，逐条列出。

**原因一：变长序列使统计量被 padding 污染。**式 (4) 的求和跑遍 batch 内全部 $(b, t)$ 位置；真实 batch 中各序列长度不一，短序列的尾部是 padding。每个通道的 $\mu_i, \sigma_i$ 因此依赖该 batch 的长度分布：同一条序列放进不同的 batch 组合，归一化结果不同。统计量成了 batch 组成的函数，而不是数据的函数。

**原因二：训练 / 推理统计漂移。**BN 的推理模式以训练期滑动平均的 $\hat{\mu}, \hat{\sigma}$ 替代实时统计。语言模型 decode 时 $B = 1$、 $T$ 逐 token 增长：推理时的真实统计与训练期（定长 $T_{\mathrm{train}}$、大 batch）攒下的滑动平均系统性不匹配，且偏差随生成长度漂移——归一化所用的参数在推理时不对应任何可估计的总体。

**原因三：跨样本、跨位置耦合破坏逐位置独立性。**式 (4) 让位置 $(b, t)$ 的输出依赖同 batch 其他样本与其他位置的取值，直接违反 2.1 §1 的架构二分（除 attention 外一切算子逐位置独立）。后果是实打实的：同一前缀的表示随 batch 同伴而变，训练信号混入 batch 组成的噪声；teacher forcing 下位置 $t$ 的表示还经由共享统计量间接依赖其他位置——逐位置独立性这个让 KV cache（2.4）与 FFN 逐 token 分析（2.6）成立的前提被拆掉。

LN 把统计轴换成特征维 $d$： $\mu, \sigma$ 只在单位置的 $d$ 个分量内计算，与 $B$、 $T$、batch 组成、序列长度全部无关，训练与推理同一公式——三条硬伤同时消失。对照表：

| 维度 | BatchNorm（式 (4)） | LayerNorm（式 (1)） |
| --- | --- | --- |
| 统计轴 | batch × 位置，逐通道 | 特征维 $d$，逐位置 |
| 逐位置独立（2.1 §1 二分） | 破坏：位置间经共享统计耦合 | 保持 |
| 训练 / 推理公式 | 不同（推理用滑动平均） | 相同 |
| 变长序列 | 统计量被 padding 污染 | 无关 |
| batch 组成变化 | 同一样本表示随之改变 | 无感 |

这是「序列建模默认 LN 族」的完整理由，与实现便利性无关。反过来读也成立：BN 在卷积网络上成立的条件，恰是序列建模缺的两条——输入定长（统计量不被 padding 污染）、样本 iid 且 batch 大（滑动平均是稳定总体的良好估计）。三条硬伤不是 BN 的实现缺陷，是它的统计轴假设与序列数据的结构性错配。

## 3. RMSNorm：去掉 re-centering 丢了什么

RMSNorm 把 LN 的两步统计砍成一步（[Zhang & Sennrich, 2019](../references.md)，arXiv:1910.07467，[B]）：

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}, \qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^{2} + \epsilon} \qquad (5)
$$

**符号**： $\epsilon > 0$：数值稳定项（Qwen3-8B 与 OLMo-2-7B 的 rms_norm_eps 均为 $10^{-6}$，各 config 条目 [A]）。归一化后 $\lVert x / \mathrm{RMS}(x) \rVert_{2} = \sqrt{d}$，与 LN 相同。

两者的代数距离可以写成一行。由方差恒等式 $\sigma^{2} = \frac{1}{d}\sum_i x_i^{2} - \mu^{2} = \mathrm{RMS}(x)^{2} - \mu^{2}$（忽略 $\epsilon$），式 (1) 的核可改写为

$$
\hat{x} = \frac{x - \mu}{\sqrt{\mathrm{RMS}(x)^{2} - \mu^{2}}} \qquad (5\text{a})
$$

即 LN 的核 = RMSNorm 的核前面多一步减均值、且分母用修正后的 RMS；RMSNorm 对应 $\mu = 0$ 的特例。stream 的 DC 分量小时（ $|\mu| \ll \mathrm{RMS}(x)$）两者数值接近，差异随 $|\mu| / \mathrm{RMS}(x)$ 增长。

与式 (1) 对照，丢掉的是减均值这一步（re-centering）。数学上丢掉的东西可以精确点名：

1. **加性常量不变性消失**。式 (2) 的平移半条不再成立： $\mathrm{RMS}(x + c \cdot 1_d) \neq \mathrm{RMS}(x)$，沿 $1_d$ 方向的分量（全部特征共享的 DC 分量）被保留并参与归一化。
2. **$1_d$ 方向不再被投影清零**。LN 的 $\hat{x}$ 恒活在 $d-1$ 维零均值子空间；RMSNorm 的输出不在该子空间内，stream 的 DC 分量随训练自由漂移。

为什么丢掉这两样仍够用——RMSNorm 论文的核心论证可以概括为 re-scaling only（[B]）：LN 起实质作用的部分是**尺度控制**而非中心化。§5 的 Jacobian 引理给出机制：梯度行为的旋钮是「归一化输出范数被钉在 $\sqrt{d}$」这一点，RMSNorm 完整保留；DC 分量只是一个固定方向上的内容，下游线性投影可以学着忽略它（把 $W$ 沿 $1_d$ 方向的响应调到近零即可），不需要归一化器代劳。实证上 RMSNorm 与原 LN 训练动态相当、在 Transformer 上无质量损失（[B]），同时省掉一次对 $d$ 的均值归约：LN 需要均值与方差两笔相互依赖的归约（或一遍 Welford 双累积），RMSNorm 只需平方和一笔——FLOPs 上相对 matmul 项可忽略，收益在访存趟数与 kernel 融合（少一个同步点，练习 3 把这笔账算出来）。Llama-3.1（RMSNorm + Pre-LN，[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、OLMo-2（[A]）、Qwen3（[A]）当前全部默认 RMSNorm。

**zero-centered 增益变体**。Gemma 2/3 把增益参数化为「相对恒等映射的偏差」（[Gemma Team, 2024](../references.md)，arXiv:2408.00118，[A]；Qwen3-Next 同式，其 hybrid 结构归 4.4 / 5.3 展开）：

$$
\mathrm{RMSNorm}_{\mathrm{zc}}(x) = \left( 1_d + \omega \right) \odot \frac{x}{\mathrm{RMS}(x)}, \qquad \omega_{\mathrm{init}} = 0 \qquad (6)
$$

与式 (5) 是同一函数类的重参数化（ $\gamma = 1_d + \omega$），但优化面不同。动机两条：(a) 参数本身即偏差，初始化精确落在恒等映射上，优化器在 0 附近对称工作，而不是围绕 1.0 做带符号偏置的更新；(b) 低精度存储友好：bf16 在 1.0 附近的舍入分辨率约 $2^{-8}$，增益存成 $\gamma$ 时该舍入直接就是有效增益的误差；存成 $\omega$（典型量级远小于 1）把有效精度集中在偏差上。两条均为机制论证，非论文断言。

## 4. Residual：梯度流的恒等项

2.1 式 (2) 把 Pre-LN 整机展开成纯加法。本节把它对反向传播的含义推完。把第 $l$ 个子层之后到第 $L$ 层的全部子层输出合记为 $\sum_{i=l+1}^{L} f_i$（下标 $i$ 跑遍子层，attention 与 FFN 交替；每个 $f_i$ 以其前的 stream 为输入），则

$$
x_L = x_l + \sum_{i=l+1}^{L} f_i \qquad (7)
$$

对 $x_l$ 求导。链式法则： $\partial L / \partial x_l = (\partial L / \partial x_L) \cdot (\partial x_L / \partial x_l)$；对式 (7) 求 Jacobian，注意每个 $f_i$ 通过其输入依赖 $x_l$：

$$
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l+1}^{L} f_i \right) \qquad (8)
$$

**符号**： $I$： $d \times d$ 单位阵； $\partial \sum f_i / \partial x_l$： $L - l$ 个子层复合 Jacobian 之和。

式 (8) 是 residual 连接的全部魔法：上游梯度 $\partial L / \partial x_L$ 经 $I$ 项**原样**到达第 $l$ 层，子层 Jacobian 的贡献叠加在旁。「梯度不消失」的准确含义不是梯度不变小，而是**没有任何一层能把梯度乘成零**：即使每个子层的 Jacobian 都很小（病态初始化、饱和非线性），恒等通道不经过任何参数， $\partial L / \partial x_l$ 的恒等分量与深度无关。反向信号被拆成「直达项 + 修正项」，层数只改变修正项的个数，不动直达项。

初始化时修正项的量级可以估：每个子层 Jacobian 的谱范数受 init scale 控制（2.2 §8 的 0.02 惯例），修正项是若干子层局部 Jacobian 乘积之和，量级远小于 1。训练初期 $\partial L / \partial x_l \approx \partial L / \partial x_L$ 几乎逐字成立——第 1 层与第 $L$ 层拿到几乎相同的上游信号；训练中修正项长大后，式 (8) 变成「直达 + 逐层修正」的混合。这正是 §5.3 末「Pre-LN 各层梯度大致均匀」的另一侧面。式 (8) 还是 2.1 §2 可加展开（式 (2)）的反向版本：前向的「只加不覆盖」让中间态可被直接读出（logit lens，4.7 的前提），反向的恒等项让梯度可被直接送达——同一条恒等通路的两头。

式 (8) 同时划清了恒等通路的边界：它只对「纯加法展开成立」的结构成立。Post-LN（ $x \leftarrow \mathrm{LN}(x + f(x))$）在每层加法之后插一次归一化， $x_L$ 无法写成 $x_l$ 加若干项，恒等通道被逐层的 LN Jacobian 打断——这正是下一节两种摆放梯度行为分叉的结构根源。

## 5. Pre-LN vs Post-LN：Xiong et al. 梯度分析（本章主线）

两种摆放方式（逐位置、单个子层、省略 $\gamma, \beta$）：

$$
\text{Post-LN:} \quad x_{l+1} = \mathrm{LN}\left( x_l + f_l(x_l) \right); \qquad \text{Pre-LN:} \quad x_{l+1} = x_l + f_l\left( \mathrm{LN}(x_l) \right) \qquad (9)
$$

结构图与梯度量级剖面（初始化时）：

```
Post-LN                              Pre-LN
x_l ──┬─────────────►(+)──► LN ──► x_{l+1}      x_l ──┬───────────────────►(+)──► x_{l+1}
      └──► f_l ──────►┘                              └──► LN ──► f_l ──────►┘
      identity 被 LN 逐层重标定                        identity 直达（式 (8) 成立）

参数梯度量级（初始化；Xiong et al. Theorem 1 + extended analysis）：
Post-LN:  输出端 ████████████░░░░ 输入端     近输出端大、向输入端衰减；末层上界 O(d√ln d)，与 L 无关
Pre-LN:   输出端 ████████████████ 输入端     各层大致均匀；末层上界 O(d√(ln d / L))，随 L 衰减
```

一个常见误传先纠正：「Post-LN 浅层梯度爆炸」与原文**方向相反**——正式定理只覆盖最后一层 FFN，其他层的结论（近输出端大、向输入端衰减）来自 supplementary 的扩展分析。精确表述如下。

### 5.1 引理：LN 的 Jacobian 谱范数

对式 (1) 的归一化核 $\hat{x} = (x - \mu)/\sigma$ 求 Jacobian。记 $P = I - \frac{1}{d} 1_d 1_d^{\top}$（向零均值子空间的投影矩阵， $\lVert P \rVert = 1$）。由商的求导法则，并用 $\partial \mu / \partial x = 1_d / d$、 $\partial \sigma / \partial x = (x - \mu)/(d\sigma)$：

$$
J_{\mathrm{LN}}(x) = \frac{\partial \hat{x}}{\partial x} = \frac{1}{\sigma} \left( P - \frac{1}{d} \hat{x} \hat{x}^{\top} \right) \qquad (10)
$$

读结构。 $\hat{x}$ 本身在零均值子空间里（ $P\hat{x} = \hat{x}$）且 $\lVert \hat{x} \rVert = \sqrt{d}$，故 $\hat{x}\hat{x}^{\top}/d$ 是沿 $\hat{x}$ 方向的单位投影。于是 $J_{\mathrm{LN}}$ 在 $1_d$ 与 $\hat{x}$ 两个方向上特征值为 0（rank $d - 2$），在与两者正交的 $d - 2$ 维子空间上恰为 $(1/\sigma) I$。谱范数：

$$
\lVert J_{\mathrm{LN}}(x) \rVert_{2} = \frac{1}{\sigma} = \frac{\sqrt{d}}{\lVert x - \mu \rVert} = O\left( \frac{\sqrt{d}}{\lVert x \rVert} \right) \qquad (11)
$$

末步在「均值相对范数可忽略」的常见情形下成立。**LN 的反向增益反比于输入范数**——stream 越粗，归一化把梯度压得越狠。这个反比就是两种摆放梯度分叉的全部机关。

把 RMSNorm 的 Jacobian 并排写： $J_{\mathrm{RMS}}(x) = \mathrm{RMS}(x)^{-1} \left( I - \bar{x}\bar{x}^{\top}/d \right)$（ $\bar{x} = x / \mathrm{RMS}(x)$）。它只杀 $\bar{x}$ 一个方向（rank $d-1$）；LN 多杀 $1_d$ 方向（rank $d-2$）——§3「丢了什么」在谱语言下的版本：被减均值吃掉的那一格正是 DC 方向。谱范数结论两者一致： $\lVert J \rVert = O(\sqrt{d}/\lVert x \rVert)$，故 §5.3 的骨架对 RMSNorm 逐字成立。

### 5.2 两种摆放的 stream 范数行为

- **Post-LN**：每层输出被 LN 重标定， $\lVert x_l \rVert \approx \sqrt{d}$（不计 $\gamma$），与层号 $l$ 无关。最后一层 FFN 处 LN 的输入范数为 $\Theta(\sqrt{d})$ 量级，不随 $L$ 收缩。
- **Pre-LN**：stream 只加不除（2.1 式 (2)）。初始化时各子层输出近似零均值、互不相关，随机游走口径：

$$
\left\lVert x^{(L)} \right\rVert^{2} \approx \left\lVert x^{(0)} \right\rVert^{2} + \sum_{i=1}^{2L} \left\lVert f_i \right\rVert^{2} = \Theta\left( L d \right) \qquad (12)
$$

即 $\lVert x^{(L)} \rVert = \Theta(\sqrt{Ld})$：Pre-LN 最终 LN 的输入范数随深度按 $\sqrt{L}$ 增长（ $2L$ 为子层总数，每层 attention + FFN 各一）。式 (12) 顺带解释了 final norm 为什么必须存在：没有它，末层 stream 范数 $\Theta(\sqrt{Ld})$ 直接乘进 logits，输出尺度随深度漂移；final RMSNorm 把输出头看到的范数钉回 $\sqrt{d}$。

两条行为合起来就是 §5.3 的分叉：Post-LN 的 LN 输入范数被锁在 $\Theta(\sqrt{d})$，引理 (11) 的增益因子为 $O(1)$；Pre-LN 的最终 LN 输入随 $\sqrt{L}$ 变粗，增益因子为 $O(1/\sqrt{L})$。同一引理、两个工作点。

### 5.3 Theorem 1（精确表述）与推导骨架

[Xiong et al., 2020](../references.md)（arXiv:2002.04745，[B]）§3 Theorem 1：假设末层相关量的范数平方——Post-LN 末层 LN 输入（残差和） $`\lVert x^{\mathrm{post}}_{L,i} \rVert^{2}`$ 与 Pre-LN 末层 LN 输入 $`\lVert x^{\mathrm{pre}}_{L+1,i} \rVert^{2}`$——对所有**序列位置** $i$ 均 $(\varepsilon, \delta)$-bounded（ $\varepsilon$、 $\delta = \delta(\varepsilon)$ 为小量；下标 $L$ 表明假设只涉末层量， $i$ 是位置指标不是特征维）。这里 $(\varepsilon, \delta)$-bounded 是标量随机变量 $Z = \lVert x \rVert^{2}$ 的集中度条件： $\Pr\left( (Z - \mathbb{E}Z) / \mathbb{E}Z \le \varepsilon \right) \ge 1 - \delta$；分量近似高斯时 $\delta$ 为 $\exp\left( -d\varepsilon^{2}/8 \right)$ 量级——维数越高，范数平方的集中度越强。则以至少 $0.99 - \delta - \varepsilon/(0.9 + \varepsilon)$ 的概率，**初始化时最后一层 FFN 参数** $W^{2,L}$ 的梯度满足

$$
\text{Post-LN:} \quad \left\lVert \frac{\partial \tilde{L}}{\partial W^{2,L}} \right\rVert_{F} \le O\left( d \sqrt{\ln d} \right); \qquad \text{Pre-LN:} \quad \left\lVert \frac{\partial \tilde{L}}{\partial W^{2,L}} \right\rVert_{F} \le O\left( d \sqrt{\frac{\ln d}{L}} \right) \qquad (13)
$$

**符号**： $\tilde{L}$：损失函数； $\lVert \cdot \rVert_F$：Frobenius 范数； $W^{2,L}$：最后一层 FFN 的第二个（输出侧）矩阵。Post-LN 的上界与 $L$ 无关；Pre-LN 的上界随 $L$ 衰减。

推导骨架（三步；完整链见原文 §3 与附录）：

1. **末层 FFN 梯度的外积分解**。 $\partial \tilde{L} / \partial W^{2,L}$ 是外积：上游行向量 $\delta$（经最终 LN 回传到 FFN 输出端）与 FFN 第一层激活 $h$ 的外积，故 $\lVert \partial \tilde{L} / \partial W^{2,L} \rVert_F = \lVert \delta \rVert \cdot \lVert h \rVert$。
2. **引理 (11) 进入 $\delta$，Chernoff 界进入 $h$**。从 logits 回传到末层 FFN 输出端的链是 $g = \partial \tilde{L} / \partial z$ → 过输出头 $W_U$ → 过最终 LN： $\delta = J_{\mathrm{LN}}(x_{\mathrm{in}})^{\top} W_U^{\top} g$。原文假设下损失侧导数按 $O(1)$ 口径取——CE 的 $g = p - y$ 既非高斯也非零均值，随机性不从这一路来； $\lVert W_U^{\top} g \rVert$ 为 $O(\sqrt{d})$ 量级。 $\sqrt{\ln d}$ 因子来自外积的另一腿：FFN 隐层的 ReLU 预激活逐单元近似独立，对 $d$ 个单元取极大的 Chernoff 界给出逐单元 $O(\sqrt{\ln d})$ 的高概率上界，于是 $\lVert h \rVert = O(\sqrt{d \ln d})$；再与 $\lVert \delta \rVert$ 相乘、F-norm 维数累计给出 $d$ 因子。
3. **代入 §5.2 的范数**。Post-LN： $\lVert x_{\mathrm{in}} \rVert = \Theta(\sqrt{d})$， $\sqrt{d}/\lVert x_{\mathrm{in}} \rVert = O(1)$，上界 $O(d\sqrt{\ln d})$ 与 $L$ 无关；Pre-LN： $\lVert x_{\mathrm{in}} \rVert = \Theta(\sqrt{Ld})$，因子变成 $O(1/\sqrt{L})$，上界 $O(d\sqrt{\ln d / L})$——深度每翻四倍，末层梯度上界减半。

量级对照： $d = 4096$ 时 $\sqrt{\ln d} \approx 2.9$，Pre-LN 相对 Post-LN 的上界比是 $1/\sqrt{L}$—— $L = 32$ 时约 $5.7$ 倍， $L = 128$ 时约 $11$ 倍：深度越深，两种摆放在初始化处的末层梯度差越多，这正是 Post-LN 在深模型上先撑不住的定量形态。

为什么定理恰好落在「最后一层 FFN」：这是梯度链上**恰好只过一次最终 LN Jacobian** 的位置—— $W^{2,L}$ 的梯度经式 (13) 骨架的三步完全由引理 (11) 与 §5.2 的范数决定，可以写成干净的上界。往浅层走一层，链上就多出中间各层的 Jacobian 乘积与残差分支的混合项，不再能被单一范数因子吸收——这正是其余各层结论只能给扩展分析、给不了同形式定理的原因。

定理只覆盖最后一层 FFN。其余各层的行为在原文 supplementary 的 extended analysis（非正式推导）：**Post-LN 的参数梯度靠近输出端大、向输入端逐层衰减**——机制是 Post-LN 的反向在每一层都要过一次 $J_{\mathrm{LN}}$，而式 (10) 在 $\hat{x}$ 方向上特征值为零、其余方向带 $1/\sigma$ 缩放，上游信号每过一层被重投影削弱一次，抵达浅层时已衰减；Pre-LN 因式 (8) 的恒等项直达各层、全程只在末端过一次 LN，各层梯度量级大致均匀。方向再强调一次：衰减朝向输入端，不是「浅层爆炸」。

### 5.4 warmup 的数学必然性

式 (13) 把 Post-LN 初始化时的训练困难从传闻变成算术。两层后果：

1. **绝对量级**。末层 FFN 梯度上界 $O(d\sqrt{\ln d})$ 不含任何 $1/\sqrt{L}$ 的自动衰减； $d = 4096$ 时这是个大数。训练第一步若直接吃全量学习率 $\eta$，输出侧若干层收到远超健康尺度的参数更新。
2. **逐层不均**。梯度从输出端到输入端跨若干数量级，单一全局 $\eta$ 无法同时适配：迁就输出端则浅层几乎不学；迁就浅层则输出端第一步就把 LN 的输入统计（ $\mu, \sigma$）打飞，LN 输入分布剧变又反过来改写所有下游梯度——正反馈式发散。

学习率预热（warmup：前 $w$ 步把 $\eta$ 从 0 线性升至目标值）是对这两条的直接回应：把最初几百步的有效更新压到 LN 统计能适应的尺度内，前向分布稳定后再放开。换个角度说，初始化时逐层安全的学习率本该是 $\eta_l \propto 1 / \lVert \partial \tilde{L} / \partial \theta_l \rVert$——逐层不同、输出端最小；warmup 用一个时变全局 $\eta(t)$ 近似这个逐层条件，因为梯度范数的层间比例随训练推进趋于稳定，几百步之后单一 $\eta$ 重新变得可行。其必要性由式 (13) 的 $L$ 无关上界与逐层不均共同支撑——末层上界是定理、逐层剖面是扩展分析，合起来是结构性结论而非调参玄学：Post-LN 深堆叠不配 warmup 在初始化处即发散（Xiong et al. 实验：深 Post-LN 无 warmup 发散、配 warmup 可训、Pre-LN 无 warmup 直接可训，[B]）。Pre-LN 的 $1/\sqrt{L}$ 自带「越深越保守」的初始化梯度尺度，恒等通路又把各层梯度拉匀，warmup 由此可省——Llama-3.1、Qwen3 的 RMSNorm + Pre-LN 继承的正是这个结论。OLMo 2 走的是输出侧变体：reordered norm， $x \leftarrow x + \mathrm{RMSNorm}(f(x))$——norm 只作用于分支输出、不碰 stream（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，Table 3 与 §3.3.2），恒等通路与式 (8) 对它保留，§5.3 的 Pre-LN 侧结论相应成立。式 (13) 与「近输出端大、向输入端衰减」的剖面是后续章节的公共引用点：P5 各家配方对照、4.4 hybrid 架构的 norm 摆放讨论都回到这里。

## 6. DeepNorm：把 Post-LN 修到 1000 层

Post-LN 的另一条出路不是预热，而是改残差本身。DeepNorm（[Wang et al., 2022](../references.md)，arXiv:2203.00555，[A]）给 Post-LN 加两个常数：

$$
x_{l+1} = \mathrm{LN}\left( \alpha \thinspace x_l + f_l(x_l) \right) \qquad (14)
$$

残差分支乘 $\alpha$；各子层输出侧投影（attention 的 $W^{O}$、FFN 的 $W_2$ 等）的初始化再乘 $\beta$。常数（原文 Figure 2； $L$ 为层数，encoder-only 与 decoder-only 同式）： $\alpha = (2L)^{1/4}$、 $\beta = (8L)^{-1/4}$。encoder-decoder 形态的 decoder 侧另取 $(3M)^{1/4}$ / $(12M)^{-1/4}$（ $M$ 为 decoder 层数）——本书主线是 decoder-only，只用前者。代入 $L = 1000$ 走一遍量级： $\alpha = 2000^{1/4} \approx 6.7$、 $\beta = 8000^{-1/4} \approx 0.106$——残差主干放大 6.7 倍、子层初始输出压到约 1/9，LN 输入由放大后的主干主导。

推导骨架（完整链见原文 §3 的更新界分析，eq (16)(17) 一带）。记 $v, w$ 为子层内 value / 输出投影的初始化缩放因子（同取 $\beta$）。论文对一步 SGD 后模型输出的变化量推出上界，要求更新幅度为 $\Theta(\eta)$、与深度 $L$ 无关，得到条件

$$
\frac{2L\left( v^{2} + w^{2} \right)}{\alpha^{2}} = 1 \qquad (14\text{a})
$$

代入 $v = w = \beta$： $\alpha^{2} = 4L\beta^{2}$，即 $\alpha/\beta = 2\sqrt{L} = \Theta(\sqrt{L})$——残差主干的增益要随深度的平方根压过子层缩放的乘积。绝对尺度再由初始化范数匹配钉死成 $\alpha = (2L)^{1/4}$、 $\beta = (8L)^{-1/4}$；乘积 $\alpha\beta = (1/4)^{1/4} = 1/\sqrt{2}$ 是这组解的**后果**，不是独立的设计条件。效果：Post-LN 的逐层不均与末层大梯度被 $\alpha$（抬高主干、压低子层相对幅度）与 $\beta$（缩小子层初始输出）联手压平，模型更新幅度与深度解耦，无 warmup 稳定训练到 1000 层（[A]）。读法：DeepNorm 把 §5.4 的 warmup 需求从训练日程改写成了架构常数。

两条路线的分工值得记清：Pre-LN 用恒等通路换梯度均匀，代价是 stream 范数随 $\sqrt{L}$ 增长、需要 final norm 收尾，且子层读到的始终是未重标定的 stream；DeepNorm 保留 Post-LN「每层输出过 LN」的逐层重标定，用 $(\alpha, \beta)$ 把更新压平，代价是两个随深度取的常数与更深的调参耦合。现代开源 LLM 主流选了 Pre-LN 或其输出侧变体（§9 表），DeepNorm 的价值在证明 1000 层量级上两条路都走得通。

## 7. QK-Norm：把归一化搬进 attention 内部

2.3 §3.4 留了一个口子： $\sqrt{d_k}$ 缩放只保证**初始化时** $\mathrm{Var}(s) = 1$；训练中 $q, k$ 的范数没有任何机制封顶， $\langle q, k \rangle$ 幅度随训练漂移，logit 持续长大则 softmax 重新滑进饱和区、2.3 式 (9) 的梯度再被饱和亏空 $\varepsilon$ 压制。QK-Norm（[Henry et al., 2020](../references.md)，arXiv:2010.04245，[B]）的原始形式是把点积整个换成余弦相似度：逐 head 对 $q, k$ 做 $\ell_2$ 归一化，并用一个可学标量 $g$ **替换** $1/\sqrt{d_k}$ 缩放：

$$
s_{ts} = g \cdot \left\langle \frac{q_t}{\lVert q_t \rVert_{2}}, \frac{k_s}{\lVert k_s \rVert_{2}} \right\rangle \qquad (15)
$$

**符号**： $g$：可学标量增益，初始化值依赖训练语料的序列长度（取词数的 97.5 分位数，原文 eq (3)，Table 5 消融该分位数选择）。原文口径下 logit 的硬上界就是 $|g|$ 本身（余弦有界， $|\langle \hat{q}, \hat{k} \rangle| \le 1$）——原文消融中掉分最多的一项正是去掉 $g$：没有它 logits 被压在 $[-1, 1]$，softmax 分不开谁该被注意。

现代变体（Gemma 3、OLMo 2）把「 $\ell_2$ 归一化 + $g$」换成 RMSNorm，并**保留** $1/\sqrt{d_k}$ 缩放（OLMo 2 的 `q_norm` / `k_norm`，2.3 §8 代码走读，[A]）：

$$
s_{ts} = \frac{\left\langle \mathrm{RMSNorm}(q_t), \mathrm{RMSNorm}(k_s) \right\rangle}{\sqrt{d_k}} \qquad (15\text{a})
$$

归一化粒度有两种：Gemma 3 与 Henry et al. 原文同为**逐 head**（在每 head 的 $d_k$ 维上做——§1 的特征维在这里换成 head 维）；OLMo 2 的 `q_norm` / `k_norm` 实测为**全投影维**变体——RMSNorm 作用于 `q_proj` 输出的 $H d_k = 4096$ 维、在 head 拆分之前（2.3 §8 走读）。摆放位置注记：RoPE 是块对角旋转、正交变换（2.5），保持范数——归一化放在 RoPE 之前还是之后，下面的封顶结论不变；OLMo 2 的实现把它放在投影之后、RoPE 之前。

现代变体的封顶效果是一行 Cauchy-Schwarz：RMSNorm（无增益）后 $\lVert q_h \rVert = \lVert k_h \rVert = \sqrt{d_k}$，故

$$
|s| \le \frac{\lVert q \rVert \thinspace \lVert k \rVert}{\sqrt{d_k}} = \sqrt{d_k} \qquad (16)
$$

式 (16) 的硬封顶是现代变体的**逐 head 口径**（Gemma 3 的粒度；原文形式的上界是 $|g|$ 本身）： $d_k = 128$ 时 logit 幅度被钉死在 11.3 以内。全维变体（OLMo 2）下，拆分后每 head 范数只在平均意义上为 $\sqrt{d_k}$（ $\sum_h \lVert q_h \rVert^{2} = d$），逐 head 最坏界松到 $H\sqrt{d_k}$——封顶仍成立，但按典型值而非逐 head 硬界理解。可学增益会把上界乘上 $\gamma^{2}$，但**增长本身被关进了参数里**：logit 失控从「激活统计的自由漂移」变成「增益向量的学习问题」，尺度小、可监控。

失控的速度可以量化：2.3 式 (7) 的口径下 logit 标准差正比于 $\lVert q \rVert \thinspace \lVert k \rVert / \sqrt{d_k}$， $q, k$ 范数同乘 $g$ 则 logit 幅度乘 $g^{2}$。初始化时 std $\approx 1$； $g^{2} \approx \sqrt{d_k} = 11.3$、即 $g \approx 3.4$ 时 std 回到 11 量级——即 2.3 §3.2 未缩放情形（ $\sigma_s \approx 11$、行内最大差距 $\Delta \approx 40$ nat、 $p_{\max} \approx 0.8\text{–}0.94$）的饱和区，梯度再被饱和亏空压制。式 (16) 把这个自由度从激活统计中拿走，饱和回归的唯一通道只剩增益 $\gamma$ 的学习。

这就是 attention logits 爆炸的结构性解法，与 2.3 的初始化缩放互补： $\sqrt{d_k}$ 管出生，QK-Norm 管终生。采用记录：Henry et al. 原文（[B]）；ViT-22B 为同路线的早期大规模实例；OLMo 2 的全维变体（[A]，见上）；**Gemma 3 以 QK-norm 取代 Gemma 2 的 soft-capping**（原文表述为直接替代，[Gemma Team, 2025](../references.md)，arXiv:2503.19786，[A]），且在每个子层前后各放一次 RMSNorm（pre + post RMSNorm）——即 $`x + \mathrm{RMSNorm}_{\mathrm{post}}(\mathrm{Attn}(\mathrm{RMSNorm}_{\mathrm{pre}}(x)))`$ 的双层夹法。归属边界再划一次：Gemma 2 没有 QK-norm，它的 logit 稳定器只有 §8 的 soft-capping。

## 8. logit soft-capping：Gemma 2 独占的软钳制

soft-capping 是另一条封顶路线，且**只属于 Gemma 2**——Gemma 2 的 logit 稳定器只有它（论文全文无 QK-norm，[Gemma Team, 2024](../references.md)，arXiv:2408.00118，[A]）：对 logit 套一层缩放的 tanh，

$$
z' = c \cdot \tanh\left( \frac{z}{c} \right) \qquad (17)
$$

Gemma 2 的两处取值：attention logits $c = 50.0$，最终输出 logits $c = 30.0$。性质两行说清： $|z'| < c$ 硬有界（tanh 值域）；导数 $\mathrm{d}z'/\mathrm{d}z = 1 - \tanh^{2}(z/c) \in (0, 1]$ 处处非零——对比硬截断 $\mathrm{clip}(z, -c, c)$ 在界外导数为 0、上游梯度被砍死，soft-capping 把「越界」变成「梯度渐弱但不归零」； $|z| \ll c$ 时 $z' \approx z$，只在尾部介入。梯度衰减的节奏代入数字： $|z| = c$ 处衰减到 $1 - \tanh^{2}(1) \approx 0.42$， $|z| = 2c$ 处约 $0.07$， $|z| = 3c$ 处约 $0.01$——界外梯度不是断供，是按 tanh 平方的速率收紧。

两处 $c$ 的分工不同：attention logits 的 50.0 管 2.3 §3 的饱和问题（score 被压在 $[-50, 50]$，softmax 不退化成 one-hot）；final logits 的 30.0 管输出分布——词表 logits 封在 $(-30, 30)$ 内，top-1 与词表底部的 logit 差至多为 60 nat，等价于给单 token 概率的可达范围设了上界，输出头无法把分布推到数值极端。归属务必准确：50.0 / 30.0 两个数字出自 Gemma 2 论文；**Gemma 3 已弃用 soft-capping**（以 §7 的 QK-norm 替代），不要把这两个常数写到 Gemma 3 头上。

## 9. 各家配方对照

| 模型 | norm | 摆放 | logit 稳定器 | 证据 |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | RMSNorm | Pre-LN + final RMSNorm | 无（仅 $\sqrt{d_k}$ 缩放） | [A] |
| OLMo-2-7B | RMSNorm（eps $10^{-6}$） | reordered： $x + \mathrm{RMSNorm}(f(x))$，norm 在子层输出 | QK-norm（`q_norm` / `k_norm`，全维变体） | [A] |
| Gemma 2 | zero-centered RMSNorm | pre + post | soft-capping（50.0 / 30.0，无 QK-norm） | [A] |
| Gemma 3 | zero-centered RMSNorm | pre + post | QK-norm（取代 soft-capping） | [A] |
| GPT-2（活化石对照） | LayerNorm | Pre-LN + 末尾 final LN（GPT-2 §2.2） | 无 | [W] |

GPT-2 一行是老配方的活化石：LayerNorm（非 RMSNorm）+ Pre-LN + 末尾 final LN——GPT-2 §2.2 的原文表述即「LN 移到每个子块的输入端、末尾再加一层 LN」（[Radford et al., 2019](../references.md)，[W]；layer_norm_epsilon = $10^{-5}$，config 口径），结构上正是本章式 (9) 的 Pre-LN 行加 final norm 收尾，在结构上绕开了 Post-LN 路线的 warmup 困境（§5.4）。它与现代配方的差异在 norm 种类（LayerNorm vs RMSNorm）而非摆放；nanoGPT 的教学配方与其逐项对照见 2.8。Qwen3-Next 的 zero-centered norm 与其 hybrid 结构在 4.4 / 5.3 展开。实现锚点：transformers repo 的 `LlamaRMSNorm`（`src/transformers/models/llama/modeling_llama.py`，[HuggingFace, modeling_llama.py](../references.md)，[A]）是式 (5) 的逐行对应。教学版（教学用，非优化实现）与官方实现的对应关系：

```python
import torch

def rmsnorm(x, weight, eps=1e-6):             # x: [B, T, d]；教学用，非优化实现
    var = x.pow(2).mean(-1, keepdim=True)     # 式 (5) 的 RMS(x)^2（沿特征维归约）
    return weight * x * torch.rsqrt(var + eps)  # rsqrt = 1/sqrt；weight 即 gamma（zero-centered 变体存 1+omega）
```

官方 `LlamaRMSNorm.forward` 与这三行的唯一实质差异是精度处理：先升 fp32 做归约与除法再降回输入 dtype。`Olmo2Attention` 的 `q_norm` / `k_norm`（`modeling_olmo2.py`，[A]）是式 (15a) 的生产形态。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| LayerNorm | 式 (1) | 特征维 $\mu, \sigma$；输入 scale/shift 不变性（式 (2)）与权重 $\delta W$ / $1_d a^{\top}$ 不变性（式 (3)）——两个参数方向被商掉， $\gamma$ 还回尺度 |
| 不用 BatchNorm | 式 (4) | 三硬伤：变长 padding 污染统计、训练/推理统计漂移、跨样本跨位置耦合破坏逐位置独立 |
| RMSNorm | 式 (5) | 丢 re-centering（DC 方向不清零、加性常量不变性消失）；re-scaling 是有效成分；省一次均值归约 |
| zero-centered | $(1_d+\omega) \odot x/\mathrm{RMS}(x)$，式 (6) | $\omega_{\mathrm{init}} = 0$：恒等起点 + 低精度友好；Gemma、Qwen3-Next |
| 残差恒等项 | 式 (8) | 恒等通道无参数、与深度无关；Post-LN 的 LN 打断纯加法展开 |
| LN Jacobian 引理 | $J = \frac{1}{\sigma}(P - \hat{x}\hat{x}^{\top}/d)$，式 (10)(11) | rank $d-2$； $\lVert J \rVert = O(\sqrt{d}/\lVert x \rVert)$：反向增益反比输入范数 |
| Xiong Theorem 1 | 式 (13) | Post-LN $O(d\sqrt{\ln d})$ 与 $L$ 无关 / Pre-LN $O(d\sqrt{\ln d / L})$；扩展分析：Post-LN 近输出大、向输入衰减（非浅层爆炸）⟹ warmup 数学必然 |
| DeepNorm | $\alpha = (2L)^{1/4}$、 $\beta = (8L)^{-1/4}$（encoder-only / decoder-only 同式），式 (14)(14a) | 更新界条件 $2L(v^{2}+w^{2})/\alpha^{2}=1$ 出 $\alpha/\beta = \Theta(\sqrt{L})$； $\alpha\beta = 1/\sqrt{2}$ 是解的后果；warmup 写成架构常数 |
| QK-Norm | 原文 $s = g\langle q/\lVert q \rVert, k/\lVert k \rVert \rangle$（逐 head ℓ2 + 可学 $g$，式 (15)）；现代变体 RMSNorm + $1/\sqrt{d_k}$，式 (15a)(16) | 上界：原文 $\|g\|$ 本身；变体 $\sqrt{d_k}$（逐 head 口径，OLMo 2 全维变体最坏 $H\sqrt{d_k}$）；Gemma 3 取代 Gemma 2 的 soft-capping |
| soft-capping | $z' = c\tanh(z/c)$，attn 50.0 / final 30.0，式 (17) | 仅 Gemma 2；导数处处非零 vs clip 界外零梯度；Gemma 3 已弃用 |

## 练习

1.（推导）补全式 (10)：从 $\hat{x} = (x-\mu)/\sigma$ 出发，依次写出 $\partial\mu/\partial x$、 $\partial\sigma/\partial x$、 $\partial\hat{x}/\partial x$ 三步；验证 $J_{\mathrm{LN}} \thinspace 1_d = 0$ 与 $J_{\mathrm{LN}} \thinspace \hat{x} = 0$，据此说明 $J_{\mathrm{LN}}$ 的秩恰为 $d-2$（ $d \ge 3$）；再证在与 $\lbrace1_d, \hat{x}\rbrace$ 正交的子空间上 $J_{\mathrm{LN}}$ 的作用恰为 $(1/\sigma) I$。
2.（推导）把式 (8) 推广到含归一化的情形：(a) 对 Pre-LN 写出 $\partial L / \partial x_l$ 的完整表达式（提示：恒等项仍在，但子层贡献项里每个 $f_i$ 的 Jacobian 左乘 $J_{\mathrm{LN}}(x_i)$）；(b) 对 Post-LN 说明为什么不存在含 $I$ 的类似分解（提示： $x_L$ 不能写成 $x_l$ 加若干项）；(c) 由 (a)(b) 解释 extended analysis「Pre-LN 各层梯度大致均匀、Post-LN 向输入端衰减」。
3.（计算）RMSNorm vs LN 的账：对 $d = 4096$，数一遍两变体各自需要几遍对 $d$ 的归约（reduction）与几次逐元素乘加；取 $L = 32$（每层 2 个 norm + 1 个 final norm）、 $T = 8192$、 $B = 1$，估 LN 相对 RMSNorm 多花的 FLOPs，并与一层 SwiGLU FFN 的 $6\thinspace d\thinspace d_{\mathrm{ff}} T$（ $d_{\mathrm{ff}} = 14336$）对比求百分比；据此说明这笔省的收益主要在访存趟数与 kernel 融合而非 FLOPs。
4.（推导 + 论证）soft-capping 与饱和：设 logit $z$ 的上游梯度为 $g$。写出经式 (17) 后的梯度 $g'$；取 $c = 50.0$，求使 $|g'/g| < 0.01$ 的 $|z|$；据此给出「梯度不归零但渐弱」的精确含义，并解释为什么 $c$ 不能取得远小于训练期 logit 的典型幅度（提示：2.3 §3 的饱和分析会以另一种形式回来）。

## 参考

- [Ba et al., 2016](../references.md)：Layer Normalization，arXiv:1607.06450，[B]（key: layernorm）——式 (1) 与不变性
- [Zhang & Sennrich, 2019](../references.md)：Root Mean Square Layer Normalization，arXiv:1910.07467，[B]（key: rmsnorm）——式 (5) 与 re-scaling only 论证
- [Xiong et al., 2020](../references.md)：On Layer Normalization in the Transformer Architecture，arXiv:2002.04745，[B]（key: pre-ln）——§5 Theorem 1（式 (13)）、extended analysis 与 warmup 实验
- [Wang et al., 2022](../references.md)：DeepNet: Scaling Transformers to 1,000 Layers，arXiv:2203.00555，[A]（key: deepnorm）——式 (14) 与缩放常数
- [Henry et al., 2020](../references.md)：Query-Key Normalization for Transformers，arXiv:2010.04245，[B]（key: qk-norm）——式 (15)
- [Gemma Team, 2024](../references.md)：Gemma 2: Improving Open Language Models at a Practical Size，arXiv:2408.00118，[A]（key: gemma2）——soft-capping 50.0 / 30.0、zero-centered RMSNorm
- [Gemma Team, 2025](../references.md)：Gemma 3 Technical Report，arXiv:2503.19786，[A]（key: gemma3）——QK-norm 取代 soft-capping、pre+post RMSNorm
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——RMSNorm + Pre-LN 标准配方
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——rms_norm_eps = $10^{-6}$
- [Qwen, Qwen3-8B config.json](../references.md)：[A]（key: qwen3-8b-config）——rms_norm_eps = $10^{-6}$
- [Radford et al., 2019](../references.md)：[W]（key: gpt2）——Pre-LN + final LN + LayerNorm 老配方对照（§2.2 原文表述）
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——`LlamaRMSNorm` 实现对照
- [HuggingFace, transformers repo OLMo 2 modeling 代码](../references.md)：[A]（key: olmo2-modeling）——`q_norm` / `k_norm` 走读
