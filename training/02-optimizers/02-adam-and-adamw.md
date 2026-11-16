# 2.2 Adam → AdamW 完整推导（Adam and AdamW: A Full Derivation）

> 本章把 LLM 预训练事实上的标准优化器拆到底：逐坐标自适应的动机、两阶矩估计与 bias correction 的几何级数推导、ε 的位置与两个 regime、Reddi 反例与 AMSGrad 修正、L2 ≠ weight decay 的完整证明与 AdamW 解耦式、LLM 超参惯例对照表（逐家出处）与 param groups 的 no-wd 名单及参数账，最后逐行走读 `torch.optim.AdamW` 源码并与教学实现对拍 | 前置：2.1（SGD 与 momentum、梯度噪声 $\Sigma/B$ 口径）、1.3（fp32 master weights）；models 1.2 式 (8)（优化器状态账）、models 2.7（LN Jacobian 引理，用于 §8）| 锚点：`torch.optim.AdamW`（pytorch/pytorch repo，`torch/optim/adamw.py` 与 `torch/optim/adam.py`，教学参照 + 官方实现）、DeepSeek-V3 技术报告 [A]、OLMo 2 [A]、Llama-2 [A]、Llama-3 herd [A]、GPT-3 [B]；Adam 原文 [B]、AdamW [A]、AMSGrad [B]

## 1. 动机：为什么每个坐标要独立步长

SGD（2.1）用同一个全局学习率 $\eta$ 更新全部坐标。这在梯度各向同性时没有问题，但真实网络的梯度是高度各向异性的：embedding 的大部分行在一个 batch 里梯度为零、命中行梯度很大；不同层、同一矩阵不同坐标之间的梯度尺度可以差多个数量级。单一 $\eta$ 被迫迁就梯度最大的坐标（否则该坐标发散），其余坐标的步长被同比例压小——2.1 二次碗分析中条件数 $\kappa$ 对收敛速率的压制，在逐坐标视角下就是这个现象。

各向异性在 LLM 里有三个可点名的来源：embedding 行的命中是稀疏的（一个 batch 只触到词表的一小部分）；深度方向上梯度尺度逐层变化（Pre-LN 下浅层信号被恒等通路放大，models 2.7 §5）；输出头的 logit 尺度随训练漂移。逐坐标自适应（per-coordinate adaptivity）的回答：为每个坐标维护其梯度尺度的在线估计 $v_i$，让步长除以该尺度：

$$
\Delta\theta_i = -\eta \thinspace \frac{g_i}{\sqrt{v_i} + \epsilon} \qquad (1)
$$

**符号**： $\theta_i$：参数第 $i$ 个坐标； $g_i$：该坐标的 batch 梯度； $v_i$：坐标梯度二阶矩的在线估计； $\epsilon > 0$：数值下限项（§4）。梯度大的坐标步长自动缩小，梯度小的坐标相对放大。式 (1) 还有一个关键不变性：把某坐标的梯度放大 $c$ 倍， $v_i$ 同步涨 $c^{2}$ 倍，步长不变——逐坐标梯度缩放不变性（per-coordinate scaling invariance），「各坐标梯度尺度不同」由此不再是优化的障碍。剩下三个问题： $v$ 怎么估（EMA 系数）、分子用什么（momentum）、估计偏差怎么修（bias correction）。这就是 Adam 的全部结构。

## 2. Adam 的完整形式

Adam（[Kingma & Ba, 2015](../references.md)，arXiv:1412.6980，[B]）在 §1 的骨架上填入两个指数移动平均（exponential moving average, EMA）。每步执行（全部运算逐元素）：

$$
\begin{aligned}
g_t &= \nabla_\theta f_t(\theta_{t-1}) \\
m_t &= \beta_1 \thinspace m_{t-1} + (1 - \beta_1)\thinspace g_t \\
v_t &= \beta_2 \thinspace v_{t-1} + (1 - \beta_2)\thinspace g_t^{2} \\
\hat{m}_t &= m_t \big/ (1 - \beta_1^{t}), \qquad \hat{v}_t = v_t \big/ (1 - \beta_2^{t}) \\
\theta_t &= \theta_{t-1} - \eta \thinspace \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned} \qquad (2)
$$

**符号**： $\theta_t \in \mathbb{R}^{n}$：第 $t$ 步参数； $g_t$：第 $t$ 步 batch 梯度； $m_t, v_t \in \mathbb{R}^{n}$：梯度一阶矩与二阶原始矩的 EMA，零初始化 $m_0 = v_0 = 0$； $\beta_1, \beta_2 \in [0, 1)$：EMA 衰减系数； $\eta$：学习率； $`\hat{m}_t`$、 $`\hat{v}_t`$：偏差校正（bias correction）后的估计，推导见 §3。 $m_t$ 即 2.1 momentum 的 EMA 写法，有效记忆窗量级 $1/(1-\beta_1)$（ $\beta_1 = 0.9$ 对应约 10 步）； $v_t$ 估计 $\mathbb{E}[g^{2}]$——注意是二阶**原始矩**而非方差 $\mathrm{Var}(g) = \mathbb{E}[g^{2}] - (\mathbb{E}g)^{2}$，这个差别在 §6 的 L2 污染分析里会起作用。除以 $`\sqrt{\hat{v}_t}`$ 即式 (1) 的逐坐标归一化； $\hat{m} / \sqrt{\hat{v}}$ 的方向由梯度均值的符号主导、量级被压到 $O(1)$， $\eta$ 由此获得直接语义：每坐标每步位移的典型量级（严格上界见 §3 末）。

一个两坐标算例把式 (2) 的行为走一遍。坐标 A：梯度恒为 $g_A = 100$；坐标 B：梯度在 $\pm 1$ 之间交替、均值为 0。 $t = 1$ 时 bias correction 给出 $`\hat{m}_1 = g_1`$、 $`\hat{v}_1 = g_1^{2}`$ 精确成立，两个坐标的首步都是 $\eta \cdot g_1 / (|g_1| + \epsilon) \approx \eta \cdot \mathrm{sign}(g_1)$——与尺度无关的满步长。走到平稳态：A 坐标 $\hat{m} \to 100$、 $\hat{v} \to 10^{4}$，步长 $\to \eta \cdot 100/100 = \eta$（保持满步长）；B 坐标 $\hat{m} \to 0$、 $\hat{v} \to 1$，步长 $\to 0$。SGD 对这两个坐标只能给同一个 $\eta$：照顾 B 的噪声则 A 爬行，照顾 A 的尺度则 B 在噪声里随机游走；Adam 把两个坐标自动分开。

更新方向的另一个读法是逐坐标信噪比。设某坐标的梯度平稳、均值为 $\mu$、标准差为 $\sigma$： $\hat{m} \to \mu$、 $\hat{v} \to \mu^{2} + \sigma^{2}$，于是更新方向收敛到 $\mu / \sqrt{\mu^{2} + \sigma^{2}}$——信噪比（SNR）高（ $|\mu| \gg \sigma$ ）的坐标走出接近 $\pm 1$ 的满步长，噪声主导（ $|\mu| \ll \sigma$ ）的坐标步长被自动压到 $|\mu|/\sigma$ 量级。Adam 把「该坐标有没有稳定方向」编码进了步长本身，这与 2.1 的梯度噪声账（ $\Sigma/B$ ）是同一枚硬币的逐坐标版本；batch 大小如何穿过这个比值是 2.3 §5 的主题。

一步 AdamW 的数据流（wd 支路即 §6 的主题）：

```
g_t ──► m ← β1·m + (1−β1)·g ──► m̂ = m/(1−β1^t) ──┐
 │                                                  ├─► θ ← θ·(1−ηλ) − η·m̂/(√v̂ + ε)
 └──► v ← β2·v + (1−β2)·g⊙g ──► v̂ = v/(1−β2^t) ──┘        ↑            ↑
   L2 写法：g ← g + λθ（λθ 进 m、v 两通道，被 √v̂ 逐坐标缩放）   │            └ 梯度项：逐坐标归一
   AdamW 写法：θ ← θ·(1−ηλ)（λθ 不经过 m / v / √v̂）─────────┘   wd 项：与梯度历史无关
```

超参默认值：原文对被测问题的推荐是 $\beta_1 = 0.9$、 $\beta_2 = 0.999$、 $\epsilon = 10^{-8}$、 $\eta = 10^{-3}$；LLM 预训练的惯例偏移（ $\beta_2 = 0.95$ 等）与逐家出处见 §7。

## 3. Bias correction：除以 $1 - \beta^{t}$ 的来源

式 (2) 里两个除数 $1 - \beta^{t}$ 不是工程补丁，是一行几何级数的结论。问题在零初始化： $m_0 = 0$ 意味着 $m_t$ 的早期值系统性地偏向 0，是 $\mathbb{E}[g]$ 的有偏估计。把 $m_t$ 的递推展开：

$$
m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \thinspace g_i \qquad (3)
$$

**符号**：和式是全部历史梯度的指数加权和，第 $i$ 步权重为 $(1-\beta_1)\beta_1^{t-i}$。先对权重本身求和：

$$
(1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = (1 - \beta_1) \cdot \frac{1 - \beta_1^{t}}{1 - \beta_1} = 1 - \beta_1^{t}
$$

权重和不为 1，缺的 $\beta_1^{t}$ 正是「零点初始化」占掉的份额。为什么不换一个初始化绕开这件事： $m_0$、 $v_0$ 的非零先验在初始化时拿不到（梯度尺度事先未知），零初始化是唯一不引入额外超参的选择——bias correction 就是把「零初始化 + 显式除法」固定为标准做法的那一步，PyTorch 的实现（§9 的 `zeros_like`）与它逐项对应。

取期望，假设梯度均值平稳（ $\mathbb{E}[g_i] = \mathbb{E}[g]$，Adam 原文的假设）：

$$
\mathbb{E}[m_t] = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \thinspace \mathbb{E}[g_i] = (1 - \beta_1^{t}) \thinspace \mathbb{E}[g] \qquad (4)
$$

第一步是期望线性性，第二步代入平稳假设与几何级数。即 $\mathbb{E}[m_t]$ 低估 $\mathbb{E}[g]$ 恰好一个因子 $1 - \beta_1^{t}$；除以它即得无偏估计 $`\hat{m}_t`$。对 $v_t$ 逐字同理： $\mathbb{E}[v_t] = (1 - \beta_2^{t}) \thinspace \mathbb{E}[g^{2}]$， $`\hat{v}_t = v_t / (1 - \beta_2^{t})`$。 $t \to \infty$ 时校正因子趋于 1，校正只在早期有实质影响。量化「早期」： $\beta_1 = 0.9$ 时 $1 - \beta_1^{t}$ 在 $t$ = 1 / 10 / 50 处分别为 0.10 / 0.65 / 0.995——一阶矩的校正约 50 步内消化完； $\beta_2 = 0.999$ 时同三点为 $10^{-3}$ / 0.010 / 0.049——二阶矩的低估要 $t \approx 1000$（ $1 - 0.999^{1000} \approx 0.632$ ）才过六成， $\beta_2 = 0.95$ 时 $t$ = 50 即达 0.923。两个 $\beta$ 的校正时标差两个数量级，式 (2) 里两个除数缺一不可的原因在此。非平稳情形（ $\mathbb{E}[g_i]$ 随 $i$ 漂移）下式 (4) 多一个残差项——原文写作 $\mathbb{E}[m_t] = \mathbb{E}[g_t](1 - \beta_1^{t}) + \zeta$， $\zeta$ 由漂移量控制、漂移缓慢时为小量（练习 1）。

校正缺失时第一步偏多少，可以直接算。 $t = 1$： $m_1 = (1-\beta_1)\thinspace g_1$， $v_1 = (1-\beta_2)\thinspace g_1^{2}$，未校正的更新量为

$$
\eta \thinspace \frac{(1-\beta_1)\thinspace g_1}{\sqrt{(1-\beta_2)\thinspace g_1^{2}} + \epsilon} \approx \eta \cdot \frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \cdot \mathrm{sign}(g_1) \qquad (v_1 \gg \epsilon^{2} \text{ 时})
$$

代入 $\beta_1 = 0.9$、 $\beta_2 = 0.999$：系数 $0.1 / \sqrt{0.001} \approx 3.16$——首步比校正后的 $\eta \cdot \mathrm{sign}(g_1)$ 大 3.16 倍； $\beta_2$ 换成 LLM 惯例 0.95：系数 $0.1 / \sqrt{0.05} \approx 0.447$——首步反而偏小一半以上。bias correction 的作用是把首步钉死在 $\eta \cdot g_1 / (|g_1| + \epsilon) \approx \eta \cdot \mathrm{sign}(g_1)$：一个与 $\beta_1$、 $\beta_2$ 无关的确定值。两个观察：(i) 校正在最初几步承担了一部分隐式 warmup 的功能（ $\beta_2 = 0.999$ 时把首步压低 3.16 倍），但它只修偏**均值**、不修方差——早期 $\hat{v}$ 的方差问题才是显式 warmup 的对象（2.3 §2）；(ii) 「每步位移被 $\eta$ 封顶」只在平稳态渐近成立——梯度平稳时 $\hat{m} \to \mu$、 $\hat{v} \to \mu^{2} + \sigma^{2}$，Jensen 不等式给出 $\hat{m}^{2} \le \hat{v}$（即 §2 的 SNR 口径）；非平稳的瞬时步可以超过 $\eta$，因为 $\hat{m}$ 与 $\hat{v}$ 用两组不同权重（ $\beta_1 \neq \beta_2$ ），Cauchy–Schwarz 不适用——算例： $\beta_1 = 0.9$、 $\beta_2 = 0.999$，前 90 步梯度为 0、第 91–100 步为 1，第 100 步得 $\hat{m} \approx 0.65$、 $\hat{v} \approx 0.10$、位移 $\approx 2.0\eta$。原文（§2.1）自己的口径就是近似有界：最稀疏情形（ $(1-\beta_1) \gt \sqrt{1-\beta_2}$ 时）上界为 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$——默认 $\beta_1$=0.9、 $\beta_2$=0.999 下恰为上文算过的 3.16 $\eta$；非稀疏情形约 $\eta$ 量级。 $\eta$ 的语义是「典型步长量级」而非严格上界。

## 4. $\epsilon$：位置与两个 regime

$\epsilon$ 在式 (2) 中的位置是 $\sqrt{\hat{v}}$ 之后、除法之前（原论文与 PyTorch 实现同口径）。它把每个坐标的步长夹在两个 regime 之间：

- **自适应 regime**（ $`\sqrt{\hat{v}_i} \gg \epsilon`$ ）：步长 $`\approx \eta \cdot \hat{m}_i / \sqrt{\hat{v}_i}`$，逐坐标归一化起作用， §1 的缩放不变性成立。
- **线性 regime**（ $`\sqrt{\hat{v}_i} \ll \epsilon`$ ）：分母 $\approx \epsilon$，步长 $`\approx (\eta / \epsilon) \cdot \hat{m}_i`$——退回梯度线性的 SGD 型更新，有效学习率被钳在 $\eta / \epsilon$，且 $\hat{v}$ 的估计噪声不再放大步长； $\hat{v} \to 0$ 的除零也在这里被兜住。

两个 regime 的分界在 $\hat{v} \approx \epsilon^{2}$： $\epsilon = 10^{-8}$ 时分界 $\hat{v} = 10^{-16}$， $\epsilon = 10^{-5}$ 时分界 $\hat{v} = 10^{-10}$。稀疏坐标让 $\epsilon$ 从数值补丁变成 regime 开关：embedding 中未被本 batch 命中的行梯度为零，其 $v$ 按 $\beta_2$ 逐步衰减、 $\hat{v}$ 指数缩水， $\sqrt{\hat{v}}$ 可以落到 $\epsilon$ 附近——这些行在两次命中之间处在线性 regime，命中时被 $\sqrt{\hat{v}}$ 压制。坐标「多久被命中一次」与 $\beta_2$ 的遗忘速度共同决定它停在哪个 regime。

这个常数对训练早期动力学有可测的影响：OLMo 2 把 AdamW 的 $\epsilon$ 从 $10^{-5}$ 降到 $10^{-8}$ 后，早期更新变大、梯度范数更快收敛到更低的水平且此后一直更低，论文将其列为训练稳定性干预之一（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§3.4.1）；对照口径：Llama-2 用 $\epsilon = 10^{-5}$（§7 表）。机制读法：训练早期 $\hat{v}$ 普遍偏小， $\epsilon = 10^{-5}$ 会把大批坐标压在线性 regime、步长被钳小约一个量级； $\epsilon$ 越小，越多坐标尽早进入自适应 regime。

## 5. Adam 可以不收敛：Reddi 反例与 AMSGrad

Reddi et al. 证明 Adam 原始论文的收敛证明有误，并给出显式反例（[Reddi et al., 2019](../references.md)，arXiv:1904.09237，[B]）。框架是在线凸优化：每步对手选定一个凸损失 $f_t$，算法出 $x_t$，评价指标是 regret $R_T = \sum_{t=1}^{T} f_t(x_t) - \min_{x \in \mathcal{F}} \sum_{t=1}^{T} f_t(x)$（与最优固定点的差距）；「收敛」的判据是平均 regret 趋零，即 $R_T / T \to 0$。反例的可行域为 $\mathcal{F} = [-1, 1]$，损失函数序列为

$$
f_t(x) = \begin{cases} C \thinspace x, & \text{当 } t \text{ 除 } 3 \text{ 余 } 1 \\ -x, & \text{其余步} \end{cases}, \qquad C > 2 \qquad (5)
$$

**符号**： $C$：大梯度常数。每三步出现一次大梯度 $C$，其余两步梯度为 $-1$。时间平均损失 $\bar{f}(x) = (C-2)\thinspace x / 3$ 关于 $x$ 递增（ $C > 2$ ； $C = 2$ 时 $\bar{f} \equiv 0$、所有点并列最优， $C \gt 2$ 是让 $x = -1$ 成为唯一最优的临界），最优定点 $x^{\ast} = -1$。

取 $\beta_1 = 0$（去掉 momentum 只留自适应项）、 $\beta_2 = 1/(1+C^{2})$、恒定步长 $\alpha$，走稳态三步循环（渐近估计，忽略 bias correction），以 $C = 3$（即 $\beta_2 = 0.1$ ）为例：

- 大梯度步： $g = +3$， $v \approx (1-\beta_2) \times 9 = 8.1$，位移 $\approx -\alpha \times 3 / \sqrt{8.1} \approx -1.05\alpha$；
- 小梯度步一： $g = -1$， $v = 0.1 \times 8.1 + 0.9 \times 1 = 1.71$，位移 $\approx +\alpha / \sqrt{1.71} \approx +0.76\alpha$；
- 小梯度步二： $v = 0.1 \times 1.71 + 0.9 \approx 1.07$，位移 $\approx +0.97\alpha$。

净漂移约 $+0.68\alpha$ / 循环——Adam 稳定爬向 $x = +1$。此时 $\bar{f}(+1) - \bar{f}(-1) = 2(C-2)/3 > 0$：算法钉在可行域内**最坏**的点， $R_T / T$ 收敛到正常数而非零（原文 Theorem 1；同一问题上 SGD 与 AdaGrad 正常收敛）。机制诊断：罕见的大梯度被自己贡献的 $v$ 除掉（ $\sqrt{v} \approx C$ ）——量级信息恰好被逐坐标归一化抹掉，只留下符号；而小梯度步里 $v$ 回落、有效步长 $\alpha/\sqrt{\hat{v}}$ 回升。论文把病根定位到其收敛分析所需的量： $\Gamma_t = \sqrt{V_t}/\alpha_t - \sqrt{V_{t-1}}/\alpha_{t-1}$（ $V_t = \mathrm{diag}(v_t)$ ），逐坐标有效学习率不增等价于 $\Gamma_t \succeq 0$；Adam/RMSprop 的 $v$ 可减， $\Gamma_t$ 可以不定号，原 Adam 证明误设了它半正定。原文进一步证明：对任意满足 $\beta_1 \lt \sqrt{\beta_2}$ 的常数 $\beta_1, \beta_2$，都存在使 Adam 平均 regret 非零的在线问题（Theorem 2）。随机版（Theorem 3）的构造把三步循环换成独立同分布抽样：以概率 $p = (1+\delta)/(C+1)$ 出大梯度 $C$（ $\delta$ 为小正数）、以概率 $1-p$ 出 $-1$，则期望损失 $\mathbb{E}[f(x)] = \delta \cdot x$ 的最优点仍是 $x = -1$，而 Adam 在该分布下收敛到错误一侧（详细分析见原文）。这组构造还解释了一个工程直觉： $\beta_2$ 越大、 $v$ 的记忆越长，偶发大梯度对分母的贡献越持久——原文以此说明实践中大 $\beta_2$ 为何更稳；但 Theorem 2 的要点是**任何**满足 $\beta_1 \lt \sqrt{\beta_2}$ 的常数组合都治不好它，只能换算法。

修正只需要一行：维护历史 $\hat{v}$ 的逐坐标最大值

$$
\hat{v}_t^{\max} = \max\left(\hat{v}_{t-1}^{\max},\ \hat{v}_t\right), \qquad \theta_t = \theta_{t-1} - \alpha_t \thinspace \frac{\hat{m}_t}{\sqrt{\hat{v}_t^{\max}}} \qquad (6)
$$

分母只增不减 ⇒ 逐坐标有效步长单调不增 ⇒ $\Gamma_t \succeq 0$ 恢复，平均 regret 以 $O(1/\sqrt{T})$ 收敛（原文 Theorem 4 及推论）。这就是 AMSGrad。代回反例： $\hat{v}^{\max}$ 被钉在 8.1，两个小梯度步各只有 $+\alpha/\sqrt{8.1} \approx +0.35\alpha$，净漂移 $-1.05\alpha + 0.70\alpha \lt 0$，回归 $x^{\ast}$ 方向。式 (6) 与 learning rate schedule 的关系也值得记一句： $\alpha_t$ 单调衰减时 $\Gamma_t \succeq 0$ 更容易成立——AMSGrad 的本质是把「有效步长不增」从 schedule 层的近似性质挪成算法层的逐坐标保证。PyTorch 的 Adam/AdamW 以 `amsgrad=True` 开关实现式 (6)（§9 走读），但 LLM 预训练实践中无人打开： $\beta_2$ 接近 1、schedule 全程衰减，「有效步长不增」在实践轨迹上近似成立，反例是刻意构造的对抗形态——此条为本书评注，无单一出处。

## 6. L2 ≠ weight decay：AdamW 的解耦证明

AdamW（[Loshchilov & Hutter, 2019](../references.md)，arXiv:1711.05101，[A]）的出发点是一个被长期掩盖的事实：「L2 正则」与「weight decay」只在朴素 SGD 下是同一件事的两种写法。

**SGD 情形：精确等价（原文 Prop 1）**。对损失加 $\frac{\lambda^{\prime}}{2} \lVert \theta \rVert^{2}$ 的 L2 项，梯度多出 $\lambda^{\prime} \theta$，朴素 SGD 更新为

$$
\theta_t = \theta_{t-1} - \alpha \thinspace \left( \nabla f_t + \lambda^{\prime} \thinspace \theta_{t-1} \right) = (1 - \alpha \lambda^{\prime}) \thinspace \theta_{t-1} - \alpha \thinspace \nabla f_t \qquad (7)
$$

**符号**： $\alpha$：学习率； $\lambda^{\prime}$：L2 系数。右端正是「weight decay 率 $\lambda = \alpha \lambda^{\prime}$ 的解耦衰减」 $\theta_t = (1-\lambda)\theta_{t-1} - \alpha \nabla f_t$。两个写法在 SGD 下逐步相同，换算关系 $\lambda^{\prime} = \lambda / \alpha$——「L2 就是 weight decay」这句话的唯一合法语境，也是深度学习库里长期混用的根源。

**Adam 情形：等价不成立（原文 Prop 2 的内容）**。把 L2 加进 Adam：梯度变为 $`\tilde{g}_t = \nabla f_t + \lambda^{\prime} \theta_{t-1}`$，两个矩都建立在 $\tilde{g}$ 上：

$$
\tilde{m}_t = \beta_1 \thinspace \tilde{m}_{t-1} + (1 - \beta_1)\thinspace (g_t + \lambda^{\prime} \theta_{t-1}), \qquad \tilde{v}_t = \beta_2 \thinspace \tilde{v}_{t-1} + (1 - \beta_2)\thinspace (g_t + \lambda^{\prime} \theta_{t-1})^{2} \qquad (8)
$$

更新 $`-\eta \thinspace \hat{\tilde{m}}_t / (\sqrt{\hat{\tilde{v}}_t} + \epsilon)`$。扭曲分三层拆出：

1. **衰减力被逐坐标缩放**： $\lambda^{\prime} \theta_i$ 的步长贡献是 $`-\eta \cdot (\lambda^{\prime} \theta_i \text{ 的 EMA}) / (\sqrt{\hat{\tilde{v}}_i} + \epsilon)`$——历史梯度大的坐标 $`\hat{\tilde{v}}_i`$ 大、衰减被除小。同一个 $\lambda^{\prime}$ 对不同坐标产生不同的相对衰减率。
2. **衰减力自带滞后**： $\lambda^{\prime}\theta$ 进的是 $m$ 通道，EMA 把衰减力变成 $\theta$ 的指数加权历史均值—— $\theta$ 快速移动时，作用在当前参数上的是过去若干步的衰减方向， $\beta_1 = 0.9$ 对应约 10 步的延迟。
3. **正则信号污染二阶矩**：式 (8) 右端展开含交叉项 $2\lambda^{\prime} \theta \odot g + \lambda^{\prime 2} \theta^{2}$——分母估计不再只反映数据梯度的尺度； $\theta$ 大的坐标分母被进一步抬高，衰减再被抑制（反向自增强）。
4. **不存在等价的 L2 系数**：解耦衰减把每步的 $\theta$ 乘同一因子 $(1 - \eta\lambda)$，与梯度历史无关；L2 路径的衰减项 $-\alpha\lambda^{\prime}\theta$ 必须穿过逐坐标预条件子 $`M_t = (\mathrm{diag}(\sqrt{\hat{\tilde{v}}_t}) + \epsilon)^{-1}`$。两条轨迹逐步相同的必要条件是 $\alpha M_t (\lambda^{\prime}\theta) = \lambda \theta$ 对所有 $t$ 成立，即 $M_t = (\lambda / (\alpha\lambda^{\prime})) \cdot I$——与「自适应预条件子不是单位阵的标量倍」矛盾。这正是原文 Prop 2 的形式化内容（练习 2 补全）。

代入数字看第 1 层的量级。取 $\eta = 3 \times 10^{-4}$、 $\lambda^{\prime} = 0.1$，两个坐标的历史梯度尺度分别为 $`\hat{v}_1 = 1`$（大梯度坐标）与 $`\hat{v}_2 = 10^{-6}`$（小梯度坐标）， $\theta_1 = \theta_2 = 1$。L2 路径下每步的相对衰减率为 $`\eta \lambda^{\prime} / (\sqrt{\hat{v}_i} + \epsilon)`$：坐标 1 得 $3 \times 10^{-5}$，坐标 2 得 $3 \times 10^{-5} / 10^{-3} = 3 \times 10^{-2}$——同一个正则系数，小梯度坐标的相对衰减强度是大梯度坐标的 1000 倍；换 AdamW，两个坐标同为 $\eta\lambda = 3 \times 10^{-5}$。「L2 对自适应优化器不是均匀正则」不是修辞，是这个除法。

扭曲清单还有一条与 schedule 的交互：L2 的衰减力随 $\theta$ 的幅度走（ $\lambda^{\prime}\theta$ 是 $\theta$ 的线性函数），而式 (9) 的解耦衰减同样正比于 $\theta$——两者的差异不在「是否正比于 $\theta$」，而在「是否被 $\sqrt{\hat{v}}$ 除」与「是否进 EMA」。把这三条合起来，Adam+L2 里 $\lambda^{\prime}$ 的实际语义是「按坐标梯度历史加权、带滞后、还抬高自适应分母的衰减」，这正是它与 $\eta$ 在调参网格上纠缠不清的原因；AdamW 把 $\lambda$ 还原成单一的相对衰减率。

**固定预条件子下的等价形式（原文 Prop 3）**。若 $M = \mathrm{diag}(s)^{-1}$ 是**固定**对角阵，「decay 率 $\lambda$」与「对尺度加权范数 $\frac{\lambda}{2\alpha} \lVert \theta \odot \sqrt{s} \rVert^{2}$ 的 L2」逐步等价——验证只需两行：该正则项的梯度是 $(\lambda/\alpha) \thinspace s \odot \theta$，穿过 $M$ 得 $-\alpha \cdot \mathrm{diag}(s)^{-1} \cdot (\lambda/\alpha) \thinspace s \odot \theta = -\lambda\theta$，与解耦衰减一致。读法：weight decay 隐含的惩罚按 $s_i$（历史梯度尺度）加权——大梯度坐标被惩罚得更重；反过来说，均匀惩罚的普通 L2 对大梯度坐标正则**相对不足**。这也是原文的机制解读：with L2，历史梯度大的权重被正则的相对量更小；decoupled weight decay 对所有权重施加同一相对衰减率 $\lambda$。

**AdamW：把衰减移出梯度**。原文 Algorithm 2 第 12 行（ $\eta_t$ 为 schedule 乘子、 $\alpha$ 为基础学习率）：

$$
\theta_t = \theta_{t-1} - \eta_t \left( \alpha \thinspace \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \thinspace \theta_{t-1} \right) \qquad (9)
$$

衰减项不经过 $m$、 $v$、 $\sqrt{\hat{v}}$，直接按固定相对率收缩 $\theta$。把 schedule 乘子折进学习率（ $\eta_t \cdot \alpha$ 即当步 $`\mathrm{lr}_t`$ ），就是 PyTorch 的形式： $\theta \leftarrow \theta\thinspace(1 - \mathrm{lr} \cdot \lambda) - \mathrm{lr} \cdot \hat{m}/(\sqrt{\hat{v}} + \epsilon)$。一处单位换算：两边的衰减因子分别为 $\eta_t \lambda$ 与 $`\mathrm{lr}_t \cdot \mathrm{wd}`$，由 $`\mathrm{lr}_t = \eta_t \thinspace \alpha`$ 折合即得 torch 的 `weight_decay` 对应原文的 $\lambda / \alpha$。这个「衰减跟随 schedule」的语义有直接的实践证据：Llama-3 的 scaling-law 实验设定每步 weight decay 为当步学习率的 0.1 倍（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.2.1）——正是式 (9) 中 $\eta_t \lambda$ 的结构（该论文未披露 $\beta$ 与 $\epsilon$，纪律注记见 §7）。原文附录另给出 normalized weight decay 提议： $\lambda = \lambda_{\mathrm{norm}} \sqrt{b/(BT)}$（ $b$ = batch 大小、 $B$ = 数据集大小、 $T$ = epoch 数），把衰减对「总步数随 batch 改变」的隐式依赖显式化——这是附录级的提议而非主结果，列出备查。

式 (9) 还有一个账本读法：整个训练里参数的累计衰减因子约为 $\prod_t (1 - \eta_t \lambda) \approx \exp(-\lambda \sum_t \eta_t)$——总衰减量由 $\lambda$ 乘 **lr 曲线下的面积**决定，而不是由单独的步数或峰值决定。schedule 拉长、峰值提高都等比例放大正则强度；normalized weight decay 想要归一化的正是这个面积对 batch/epoch 口径的依赖。这个读法同时解释了为什么 wd 的调整在换 schedule 时必须重调： $\lambda$ 不变而 $\sum_t \eta_t$ 变了，等效正则量就变了（本书分析）。

实证结论一句话：AdamW 在 CIFAR-10 / ImageNet-32×32 上以更少的调参把 Adam 的泛化差距弥合到 SGD-momentum 水平，且 $\eta$ 与 $\lambda$ 的最优值近似解耦（原文 §4-5，[A]）。LLM 预训练侧的采用情况见 §7。

## 7. 超参结构：默认值与 LLM 惯例对照

$\beta_1, \beta_2, \epsilon, \lambda$ 四个数的逐家出处（均为一手口径；未披露写未披露，不把二手转述当事实）：

| 来源 | $\beta_1$ | $\beta_2$ | $\epsilon$ | weight decay | 梯度裁剪 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| Adam 原文默认 [B] | 0.9 | 0.999 | $10^{-8}$ | — | — | lr 默认 $10^{-3}$ |
| `torch.optim.AdamW` 默认（教学参照） | 0.9 | 0.999 | $10^{-8}$ | 0.01 | — | `torch/optim/adamw.py` 签名 |
| GPT-3 [B] | 0.9 | 0.95 | $10^{-8}$ | 0.1 | 1.0 | [Brown et al., 2020](../references.md)，arXiv:2005.14165，App. B；论文写 "Adam"，未说明 wd 的解耦语义 |
| Llama-2 [A] | 0.9 | 0.95 | $10^{-5}$ | 0.1 | 1.0 | [Touvron et al., 2023](../references.md)，arXiv:2307.09288，§2 |
| DeepSeek-V3 [A] | 0.9 | 0.95 | 未披露 | 0.1 | 1.0 | [DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，§4.2 |
| OLMo 2 7B [A] | 0.9 | 0.95 | $10^{-8}$ | 0.1（embedding 除外，§8） | 1.0 | OLMo-core 训练 config 口径（`src/scripts/train/OLMo2/OLMo2-7B.py`）； $\epsilon$ 变更的动机见 §4 |
| Llama-3 [A] | 未披露 | 未披露 | 未披露 | 0.1×lr（scaling-law 实验口径，§6） | — | herd 论文只披露 AdamW 与 lr 日程；optimizer 超参未披露，不可引为 $\beta_2 = 0.95$ 的出处 |

结构上值得记录的三条规律。第一， $\beta_2$ 从原文默认 0.999 到 LLM 惯例 0.95 是 GPT-3 起的偏移（表中前三个 LLM 出处全部为 0.95）。 $v_t$ 的 EMA 有效记忆窗量级为 $1/(1-\beta_2)$：0.999 对应约 1000 步，0.95 对应约 20 步——短窗让二阶矩估计更快跟上梯度尺度的漂移（warmup、batch ramp、数据配比切换都会改变梯度统计），代价是窗口内样本更少、估计方差更大。需要说明：各家报告均未给出选 0.95 的消融依据，本段是本书的机制读法，不是出处事实。第二， $\beta_1 = 0.9$ 在所有出处中无一偏移——一阶矩的窗口（约 10 步）没有人动，被动过的总是二阶矩与 $\epsilon$。第三，矩估计的存储精度可以压：DeepSeek-V3 把 AdamW 的一、二阶矩以 BF16 存储（master 权重与梯度仍 fp32），「无可观测的性能下降」（[DeepSeek-AI, 2024](../references.md)，[A]，§3.3.3）——代入 models 1.2 式 (8) 的优化器状态账，矩估计字节系数 $c$ 从 4 降到 2， $(4+2c)N$ 从 $12N$ 降到 $8N$ bytes；671B 模型这一项省 $4N$ bytes 量级（约 2.7 TB，SI 口径）。优化器状态的切分（每张卡只存一片）归 4.2 的 ZeRO 账。

表外还有一项高度一致：梯度裁剪阈值 1.0 出现在全部四个 LLM 出处里——裁剪的公式与阈值选择归 2.4 专章。torch 的默认值（ $\beta_2$ = 0.999、wd = 0.01）与 LLM 惯例全部错开，用库默认值起 LLM 预训练是一个已知的错误起点。本节这张表只覆盖 optimizer 本体；lr / bsz / warmup / decay / wd / $\beta_2$ / clip 的全字段真实配方对照表归 7.2。

## 8. Param groups：no-wd 名单与参数账

式 (9) 的 $\lambda$ 不需要对所有参数一致。PyTorch 的机制是 param groups：优化器按参数子集分组，每组可覆盖自己的 `lr` / `weight_decay` 等超参（`torch/optim/optimizer.py` 的 `param_groups` 机制）。通行的工程惯例是按张量类型分两组：

- **decay 组**：参与矩阵乘的权重（各线性层的二维权重矩阵）；
- **no-decay 组**：norm 增益、偏置（bias）、以及（多家的）embedding。

norm 增益与偏置不进 decay 的理由有两层：它们的参数占比在小数点后多个量级（下文的账），正则化收益可忽略；归一化增益还存在联合尺度冗余——RMSNorm 增益 $\gamma$ 放大 $c$ 倍、下游线性层权重缩小 $c$ 倍，函数不变，decay $\gamma$ 主要推动的是这个冗余方向的参数化而非函数本身（本书分析）。embedding 不进 decay 有一个记录在案的失败案例：OLMo 2 发现对 embedding 施加标准 weight decay 会把 embedding 范数压得过头，而 LN 对输入 $x$ 的 Jacobian 正比于 $1/\lVert x \rVert$（models 2.7 §5.1 的引理口径），过小的 embedding 会在早期层制造大梯度；OLMo 2 的方案是直接对 embedding 关 wd，之后 embedding 范数稳定在健康区间（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§3.4.2；配置实证：OLMo-core `OLMo2-7B.py` 中 `OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))`）。

代码层面的形态（教学用，非优化实现；生产栈的等价机制是配置级的 group override，如上文 OLMo-core 一行）：

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    (decay if p.dim() >= 2 and "embed" not in name else no_decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```

参数账（Llama-3.1-8B 数值，取自 models 1.2 的逐项账）：embedding $V \cdot d = 128256 \times 4096 \approx 5.25 \times 10^{8}$，占总参数 $8.03 \times 10^{9}$ 的 6.5%——embedding 是否进 decay 是可观的正则化决策；norm 参数 $(2L+1) \cdot d = 65 \times 4096 \approx 2.7 \times 10^{5}$，占 0.003%；bias 在现代 LLM 里普遍不存在（Llama-3.1 与 OLMo 2 的线性层均无 bias），no-wd 名单里的 bias 一项主要对应含 bias 的老架构。结论：no-wd 名单动的几乎不是参数占比，而是这些张量的训练动力学。

## 9. `torch.optim.AdamW` 源码逐行走读

以 torch 2.8 为准（pytorch/pytorch repo，`torch/optim/adamw.py` 与 `torch/optim/adam.py`，教学参照 + 官方实现；下述行级事实已对照源码核实）。

**入口分流（`torch/optim/adamw.py`）**。`class AdamW(Adam)` 只做一件事：以 `decoupled_weight_decay=True` 调 `Adam.__init__`；`__setstate__` 在载入旧 checkpoint 时强制该旗标为 True（向后兼容）。函数式入口 `adamw(...)` 同样转调 `adam(...)` 并置同一旗标——AdamW 的全部计算与 Adam 共享一份实现，分岔只在 weight decay 一行。由此也得到库层面的对照：`torch.optim.Adam(weight_decay=λ)`（默认 `decoupled_weight_decay=False`）走的是式 (8) 的 L2 路径，与 `torch.optim.AdamW` 的差异恰好是 §6 证明的那一处——两个类的名字之差就是 Prop 2 的内容之差。

入口文件里还嵌着一份可对照的文本：`AdamW.__doc__` 用 LaTeX 渲染了完整算法块，其中衰减行 $\theta_t \leftarrow \theta_{t-1} - \gamma\lambda\theta_{t-1}$ 写在矩更新之前——与 `_single_tensor_adam` 里 `param.mul_` 的位置一致；amsgrad 分支在 docstring 里是「先取 max 再除以 $1-\beta_2^{t}$」。

**状态初始化（`adam.py` `_init_group`）**。惰性初始化：`state["step"]` 置 0（非 capturable/fused 时放在 CPU——源码注释说明这是为省 CUDA kernel launch）；`exp_avg` / `exp_avg_sq` 为 `torch.zeros_like(p)`（零初始化 ⇒ §3 的 bias correction 必要性在实现层面可见）；`amsgrad=True` 时多一个 `max_exp_avg_sq`。

**主循环（`adam.py` `_single_tensor_adam`）**，逐参数执行，与式 (2)(9) 逐行对应：

```python
step_t += 1                                              # t 从 1 起计
if weight_decay != 0:
    if decoupled_weight_decay:
        param.mul_(1 - lr * weight_decay)                # AdamW：theta <- theta(1 - lr*wd)，式 (9)
    else:
        grad = grad.add(param, alpha=weight_decay)       # Adam+L2：g <- g + wd*theta，进式 (8)
exp_avg.lerp_(grad, 1 - beta1)                           # m <- beta1*m + (1-beta1)*g
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)  # v <- beta2*v + (1-beta2)*g*g
bias_correction1 = 1 - beta1**step                       # 式 (4) 的分母
bias_correction2 = 1 - beta2**step
step_size = lr / bias_correction1
denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)   # sqrt(v_hat) + eps
param.addcdiv_(exp_avg, denom, value=-step_size)         # theta <- theta - lr*m_hat/denom
```

逐维对照三点：(i) wd 分岔的两行就是 §6 的证明对象——`grad.add(param)` 走式 (8)（污染 $m, v$），`param.mul_` 走式 (9)（不经过梯度通道）；(ii) 衰减在矩更新**之前**执行，用的都是 $\theta_{t-1}$，与式 (9) 数学相同；(iii) `denom` 把 $\sqrt{\hat{v}}$ 写成 $\sqrt{v} / \sqrt{1-\beta_2^{t}}$ 再 `add_(eps)`，配合 `step_size = lr/(1-\beta_1^{t})` 与 `value=-step_size`，代数恒等于 $-\mathrm{lr}\cdot\hat{m}/(\sqrt{\hat{v}}+\epsilon)$。amsgrad 分支在最后两行之前插入 `torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])` 并改用 max 版做 `denom`——式 (6) 的实现就是这两行。一个与原文的细节差异：torch 对 max 值同样施加 bias correction（分母为 $\sqrt{v^{\max}} / \sqrt{1-\beta_2^{t}}$），原文 Algorithm 2 不校正；除以正标量与逐坐标 max 可交换，单调性不受影响。capturable 路径把 `step_size` 折进 `denom`（源码注释：把标量运算折进分母以避免一次额外的参数级读写），是等价改写而非数学差异。

**变体一句**：`_multi_tensor_adam`（`foreach=True`）用 `torch._foreach_*` 族把逐元素运算按参数列表合并、摊薄 kernel launch；`_fused_adam`（`fused=True`）把整个 step 合成单个 CUDA kernel；两者数学与 `_single_tensor_adam` 逐行相同，差别全在调度与访存。

教学重写版（教学用，非优化实现）与官方实现对拍：

```python
import torch

def teach_adamw_step(p, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    m.mul_(beta1).add_(g, alpha=1 - beta1)                 # 式 (2)：一阶矩 EMA
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)          # 式 (2)：二阶矩 EMA
    bc1, bc2 = 1 - beta1**t, 1 - beta2**t                  # 式 (4)：bias correction 因子
    p.mul_(1 - lr * wd)                                    # 式 (9)：解耦 weight decay
    p.addcdiv_(m, (v.sqrt() / bc2**0.5).add_(eps), value=-lr / bc1)

p1 = torch.nn.Parameter(torch.randn(30, 20))
p2 = torch.nn.Parameter(torch.randn(30, 20)); p2.data.copy_(p1.data)
m, v = torch.zeros_like(p1), torch.zeros_like(p1)
opt = torch.optim.AdamW([p2], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
for t in range(1, 26):
    g = torch.randn(30, 20) * (10 ** (t % 3))              # 跨量级的合成梯度
    p1.grad = p2.grad = g.clone()
    opt.step()
    teach_adamw_step(p1.data, g, m, v, t, 3e-4, 0.9, 0.95, 1e-8, 0.1)
assert torch.allclose(p1.data, p2.data, atol=1e-6, rtol=1e-5)
# 容差口径：fp32 下逐元素运算顺序不同（实测 25 步最大差 1.2e-7）；float64 下差在 1e-16 量级
```

## 总结

| 机制 | 公式 | 要点 |
| --- | --- | --- |
| 逐坐标自适应 | $\Delta\theta_i = -\eta g_i/(\sqrt{v_i}+\epsilon)$，式 (1) | 步长除以坐标梯度尺度；逐坐标缩放不变 |
| Adam | 式 (2) | $m$ 为 momentum 的 EMA 写法； $v$ 估二阶原始矩（非方差） |
| bias correction | 式 (3)(4) | 零初始化 ⇒ 权重和 $1-\beta^{t}$；不修则首步偏 3.16 倍（ $\beta_2$=0.999） |
| $\epsilon$ | 分界 $\hat{v} \approx \epsilon^{2}$ | 自适应 regime vs 线性 regime；OLMo 2： $10^{-5} \to 10^{-8}$ [A] |
| Adam 不收敛 | 反例式 (5)，AMSGrad 式 (6) | 大梯度被自己的 $v$ 除掉； $\Gamma_t \succeq 0$ ⇔ $\hat{v}^{\max}$ 修正 |
| L2 ≠ weight decay | SGD 等价式 (7)；Adam 扭曲式 (8) | L2 的 $\lambda\theta$ 被 $\sqrt{\hat{v}}$ 逐坐标除小且污染二阶矩 |
| AdamW | 式 (9) | $\theta \leftarrow \theta(1-\eta\lambda) - \eta\hat{m}/(\sqrt{\hat{v}}+\epsilon)$；torch 实现为 `param.mul_(1-lr*wd)` |
| LLM 惯例 | §7 表 | $\beta_1$=0.9、 $\beta_2$=0.95、wd=0.1、clip 1.0； $\epsilon$ 分 $10^{-8}$ 与 $10^{-5}$ 两派；Llama-3 未披露 optimizer 超参 |
| param groups | §8 | norm/bias/embedding 不进 decay；embedding 的账为 6.5%（8B 实例） |

## 练习

1.（推导）bias correction 的非平稳残差：设 $\mathbb{E}[g_i] = \bar{g} + \delta_i$（ $\bar{g}$ 为常数， $\delta_i$ 为漂移），证明 $\mathbb{E}[m_t] = (1-\beta_1^{t})\thinspace\bar{g} + (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}\delta_i$，并给出残差绝对值的上界 $\max_i |\delta_i|$；对 $v_t$ 写出对应等式，说明为什么原文称 $\beta$ 接近 1 时残差为小量（提示：漂移项的权重和同样不超过 1）。
2.（推导）补全 §6 Prop 2 的形式化：设优化器更新为 $\theta_t = \theta_{t-1} - \alpha M_t \nabla f_t(\theta_{t-1})$， $M_t$ 为对角正定阵且 $M_t \neq kI$。证明：不存在 L2 系数 $\lambda^{\prime}$，使「对 $f + \frac{\lambda^{\prime}}{2}\lVert\theta\rVert^{2}$ 跑该优化器」与「对 $f$ 跑该优化器外加每步 $(1-\lambda)\theta$ 衰减」的轨迹相同（按文中的必要条件路线；注意需要排除 $\theta_{t-1} = 0$ 的退化步）。
3.（计算）取 Reddi 反例 $C = 10$（ $\beta_2 = 1/101$ ）：(a) 手算稳态三步循环的 $v$ 与位移（精确到两位小数），验证净漂移方向指向 $x = +1$；(b) 换 AMSGrad 重算，验证净漂移反向；(c) $\beta_2$ 改为 0.999（不随 $C$ 特调）时循环形态怎么变，为什么原文说「大 $\beta_2$ 有缓解作用但不根治」。
4.（代码）运行 §9 的对拍脚本（fp32 与 float64 各一遍），再完成两个实验：(a) 把参数拆成 decay / no-decay 两个 param group（模拟 norm/embedding 名单），验证 torch 与教学版仍逐步一致；(b) 把教学版中 `p.mul_(1 - lr * wd)` 移到 `addcdiv_` 之后执行，数值差多少？说明两种顺序在数学上各自对应式 (9) 的哪种写法。

## 参考

- [Kingma & Ba, 2015](../references.md)：Adam: A Method for Stochastic Optimization，arXiv:1412.6980，[B]（key: adam）——式 (2) 与 bias correction 出处，默认超参出处
- [Loshchilov & Hutter, 2019](../references.md)：Decoupled Weight Decay Regularization，arXiv:1711.05101，[A]（key: adamw）——Prop 1-3 与 Algorithm 2（式 (7)(8)(9)）、normalized weight decay
- [Reddi et al., 2019](../references.md)：On the Convergence of Adam and Beyond，arXiv:1904.09237，[B]（key: amsgrad）——反例式 (5)、 $\Gamma_t$ 条件与 AMSGrad 式 (6)
- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）—— $\beta_2$=0.95 惯例的出处（App. B）
- [Touvron et al., 2023](../references.md)：Llama 2: Open Foundation and Fine-Tuned Chat Models，arXiv:2307.09288，[A]（key: llama2）—— $\epsilon = 10^{-5}$ 对照口径（§2）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——AdamW 超参（§4.2）与 BF16 矩估计（§3.3.3）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）—— $\epsilon$ 消融（§3.4.1）与 embedding no-wd（§3.4.2）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——wd = 0.1×lr 的 schedule 耦合口径（§3.2.1）；optimizer 超参未披露
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照 + 官方实现（key: pytorch-repo）——`torch/optim/adamw.py`、`torch/optim/adam.py`（`_single_tensor_adam` / `_multi_tensor_adam` / `_fused_adam`）、`torch/optim/optimizer.py`（param groups）
