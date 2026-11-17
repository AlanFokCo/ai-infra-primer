# 2.4 梯度裁剪与稳定性原语（Gradient Clipping and Stability Primitives）

> 本章讲训练稳定性的第一道保险丝：global norm clipping 的公式与方向保持推导、阈值 1.0 惯例的可查证出处链、clip 与 Adam 二阶矩的交互机制，以及 loss spike 的四类机制预告与全部稳定原语的 placement 总图 | 前置：2.2（Adam/AdamW 的 $m$、 $v$ 二态与解耦式）、1.5（梯度累加与 clip 的顺序语义）、models 2.3 §4（softmax 数值溢出账）| 锚点：`torch.nn.utils.clip_grad_norm_`（pytorch repo，教学参照+官方实现）、GLM-130B [A]、ST-MoE [B]、GPT-3 [B]、DeepSeek-V3 [A]、OLMo 2 [A]

## 1. Global norm clipping：定义与实现

GPT-3 的训练配置原文有一句话：「we clip the global norm of the gradient at 1.0」（[Brown et al., 2020](../references.md)，arXiv:2005.14165，[B]，附录 B）。这句话里有三个关键词：global（全体参数拼成一个向量）、norm（按范数而非按值）、1.0（阈值惯例，§3 展开）。先把公式写死。

设模型的全部可训练参数为张量集合 $\lbrace W_p \rbrace_{p \in \mathcal{P}}$，某一步各参数的梯度为 $g_p = \partial \mathcal{L} / \partial W_p$。global norm clipping 先把所有梯度张量拼成一个向量求 L2 范数，再按同一系数缩放：

$$
\lVert g \rVert_2 = \sqrt{\sum_{p \in \mathcal{P}} \lVert g_p \rVert_F^{2}}, \qquad c = \min\left(1, \hspace{5mu} \frac{\tau}{\lVert g \rVert_2 + \varepsilon}\right), \qquad g_p \leftarrow c \thinspace g_p \qquad (1)
$$

**符号**： $\mathcal{P}$：参与裁剪的参数集合（生产上通常是全部可训练参数）； $\lVert g_p \rVert_F$：第 $p$ 个梯度张量的 Frobenius 范数（全部元素平方和开根号，张量形状被抹平）； $\tau > 0$：裁剪阈值（PyTorch 接口里的 `max_norm`）； $c$：缩放系数； $\varepsilon = 10^{-6}$：防除零的小量（实现口径，见下文代码走读）。

三个设计点都写在式 (1) 里。第一，范数是全局的：分子分母都是全体梯度拼成的向量的范数，不是逐层各裁各的——逐层裁剪与全局裁剪的差别在 §5 的 GLM-130B 案例里会变成真问题。第二，缩放是等比的：全体 $g_p$ 乘同一个标量 $c$，参数组之间的相对比例不变。第三， $c = \min(1, \cdot)$：范数小于阈值时 $c = 1$ 不动，只缩不放——clip 是保险丝，不是归一化。

实现走读。PyTorch 官方实现在 pytorch repo 的 `torch/nn/utils/clip_grad.py`（[PyTorch](../references.md)，教学参照+官方实现），拆成两个函数，与式 (1) 逐步对应：

- `_get_total_norm`：先对每个梯度张量算范数（`torch._foreach_norm`），再对这些范数求一次同阶范数（norm-of-norms）。p = 2 时 $\sqrt{\sum_p \lVert g_p \rVert_2^2}$ 与「拼接后求范数」逐字等价，但省掉了真正的拼接。可选 `error_if_nonfinite=True`：总范数为 NaN/Inf 时直接抛错——把「梯度炸了」从静默缩放变成显式异常，生产上配合 skip-step 逻辑使用（7.3 的干预手册）。
- `_clip_grads_with_norm_`：`clip_coef = max_norm / (total_norm + 1e-6)`，然后 `clip_coef_clamped = torch.clamp(clip_coef, max=1.0)`，最后逐张量 `mul_`。注意实现里没有 `if clip_coef < 1` 分支——源码注释说明原因：分支会引入一次 CPU↔GPU 同步，clamp 写法把判断留在 GPU 上。代价是「不裁」时也逐张量乘了一次 1.0。
- `clip_grad_norm_` 返回裁剪前的 `total_norm`。这个返回值是免费的监控指标：每步记录它，就得到 §3 分位数选阈值和 §5 spike 预警的原始数据流。

教学版对照（教学用，非优化实现——直译式 (1)，含一次 `.item()` 同步，无 foreach 融合）：

```python
import torch

def global_norm_clip_(params, tau, eps=1e-6):
    """教学用，非优化实现。式 (1) 的直译：拼范数 → 算系数 → 等比缩放。"""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum(torch.linalg.vector_norm(g) ** 2 for g in grads))
    c = min(1.0, tau / (total.item() + eps))   # .item() 是 GPU→CPU 同步点；官方实现用 clamp 消除
    for g in grads:
        g.mul_(c)
    return total                                # 与官方一致：返回裁剪前范数供监控

# 等价性断言（CPU / fp32 口径）
import torch.nn as nn
m1, m2 = nn.Linear(64, 64), nn.Linear(64, 64)
m2.load_state_dict(m1.state_dict())
x = torch.randn(8, 64)
for m in (m1, m2):
    m(x).square().mean().backward()
t1 = global_norm_clip_(list(m1.parameters()), tau=1.0)
t2 = torch.nn.utils.clip_grad_norm_(m2.parameters(), max_norm=1.0)
assert torch.allclose(t1, t2, atol=1e-6)
assert all(torch.allclose(a.grad, b.grad, atol=1e-6)
           for a, b in zip(m1.parameters(), m2.parameters()))
```

一个顺序语义注记（与 1.5 衔接）：clip 必须作用于**累加完成后的全 batch 梯度**——gradient accumulation 的 $m$ 个 micro-batch 各自的小梯度都可能低于阈值，累加后未必；逐 micro-batch 裁剪会改变累加结果的统计含义。同理，分布式下 clip 在梯度 all-reduce 之后、optimizer step 之前（4.1 的 DDP 时序）。

## 2. 关键推导：范数裁剪保持方向，逐值裁剪不保持

式 (1) 中 $g' = c \thinspace g$、 $c \ge 0$ 为标量。两条等价的精确表述把「方向保持」钉死：

**(a) 单位方向不变**。 $c > 0$ 时 $g' / \lVert g' \rVert_2 = c \thinspace g / (c \lVert g \rVert_2) = g / \lVert g \rVert_2$；对任意固定方向 $u$， $\langle g', u \rangle = c \thinspace \langle g, u \rangle$ 同步缩放。任意两个坐标的比值 $`g'_i / g'_j = g_i / g_j`$ 也不变——坐标间的相对结构完整保留。

**(b) 式 (1) 是欧氏投影**。裁剪后梯度恰为「半径 $\tau$ 的球内离 $g$ 最近的点」：

$$
\min_{h : \lVert h \rVert_2 \le \tau} \lVert h - g \rVert_2^{2} \qquad (2)
$$

**符号**： $h$：裁剪后的梯度向量（全体参数拼接口径，下同）。

证明一行：展开 $\lVert h - g \rVert_2^2 = \lVert h \rVert_2^2 - 2 \langle h, g \rangle + \lVert g \rVert_2^2$，固定 $\lVert h \rVert_2$ 时需最大化 $\langle h, g \rangle$；由 Cauchy-Schwarz， $\langle h, g \rangle \le \lVert h \rVert_2 \lVert g \rVert_2$，等号当且仅当 $h \parallel g$。于是 $h = s \thinspace g / \lVert g \rVert_2$（ $s = \lVert h \rVert_2$），目标变成 $\min_{0 \le s \le \tau} (s - \lVert g \rVert_2)^2$，解 $s = \min(\lVert g \rVert_2, \tau)$——即式 (1)。「在可信半径内尽量不动原梯度」这一准则的唯一解就是等比缩放。

对照物是逐值裁剪（per-value clipping，PyTorch 的 `clip_grad_value_`）： $h_i = \mathrm{sign}(g_i) \min(|g_i|, \tau)$，逐坐标截断。取二维例 $g = (100, 1)$、 $\tau = 1$：逐值裁剪得 $h = (1, 1)$，方向角从 $\arctan(1/100) \approx 0.57^{\circ}$ 被扭到 $45^{\circ}$——偏差约 44.4°。范数裁剪得 $h = g / \lVert g \rVert \approx (1.0, 0.01)$，方向偏差精确为 0。高维下逐值裁剪把方向推向符号向量 $\mathrm{sign}(g)$（所在象限的对角线）：所有大分量被压平到同一刻度，「哪个坐标需要更大的步」这一相对幅度信息被抹掉。而逐坐标的相对幅度恰恰是下游一阶方法的输入——momentum 在 $1/(1-\beta)$ 步窗口内累积方向（2.1），Adam 的 $m$ 同样（2.2）；方向被扭曲的梯度会在 EMA 里驻留数十步，不是一步的代价。

逐值裁剪不是投影到欧氏球而是投影到 $\ell_\infty$ 盒子；两种投影的几何差异就是方向保持性的全部来源。本书此后「梯度裁剪」默认指式 (1) 的范数口径。

「方向」在高维下的可操作读法值得登记一句：相邻步梯度的余弦相似度、梯度与 momentum buffer 的夹角，都是方向统计的常规监控量。范数裁剪后这些量与裁剪前完全一致；逐值裁剪会把它们系统性推向符号向量的方向——监控曲线上的异常平移往往就是这种几何扭曲的指纹。

## 3. 阈值选择：1.0 惯例的出处链与分位数监控

$\tau = 1.0$ 是 LLM 训练的统治性默认值。本书可查证（references.md 已收录）的出处链：

| 模型 | 口径原文 | 出处 |
| --- | --- | --- |
| GPT-3 | 「we clip the global norm of the gradient at 1.0」（附录 B） | [Brown et al., 2020](../references.md)，arXiv:2005.14165，[B] |
| GLM-130B | 「clip gradients using a clipping value of 1.0」 | [Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]，§2.2 |
| DeepSeek-V3 | 「The gradient clipping norm is set to 1.0.」 | [DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A] |
| OLMo 2 | 超参表 Gradient Clipping = 1.0 | [Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A] |

要理解 1.0 是不是「任意」的，先看清 $\tau$ 的语义：它是梯度范数的信任半径——超出部分被压回边界。 $\tau$ 相对典型范数的位置决定 clip 的行为模式： $\tau$ 远大于日常范数时 clip 常年不触发，只是保险丝； $\tau$ 小于中位范数时 clip 每步触发，优化器实际跑的是「定幅度方向跟随」而非裸梯度——两种模式都可行，但超参语义完全不同（§4 展开）。1.0 惯例属于哪一种，取决于具体模型的典型梯度范数量级，而这个量级随初始化、精度与阶段漂移——所以严肃的工程做法不是背默认值，是监控。

分位数监控法（工程实践，无单一出处，标注为本书口径）：`clip_grad_norm_` 每步白送一个裁剪前的 total_norm；维护它的 running 分布，把 $\tau$ 设在历史范数的高分位（如 P99 量级），并跟踪**触发率**（被裁步数占比）作为健康指标：长期触发率接近 0 说明阈值形同虚设（可接受——保险丝本就该常年不动作）；触发率持续接近 1 说明真实梯度尺度系统性超过阈值，该调的不是阈值而是 LR 或 warmup（2.3）。触发率的突然跳升是比 loss 更灵敏的先行指标——下一段的 GLM-130B 记录给了这个判断一份实证。

GLM-130B 的训练笔记记录了两条关键观察（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]，§3）：其一，「a training collapse usually lags behind a 'spike' in gradient norm by a few training steps」——训练崩溃通常滞后于梯度范数突刺数步，grad norm spike 是 collapse 的先行指标而非同时现象；其二，spike「通常由 embedding 层的异常梯度引起」，早期训练中 embedding 层梯度范数「often several magnitude larger」于其他层。这两条合起来就是监控体系的设计依据（指标体系详拆归 7.3）：total_norm 逐步入日志，分位数定阈值，触发率与 spike 形态做预警。

典型范数轨迹的形态随模型、初始化与阶段漂移，不存在跨模型通用的「正常区间」——这正是监控必须按本模型历史自定阈值的原因，也是 1.0 惯例只能当起点、不能当结论的原因。一个可复用的经验结构是：warmup 结束前后范数分布最不稳（LR 在爬坡，2.3），进入稳定期后分布收窄，spike 在收窄后的分布上最显眼——把分位数计算的起点设在 warmup 结束之后，能避免早期宽幅波动把阈值顶高。

## 4. clip 与 Adam 的交互：一致裁剪假设与两种 regime

AdamW 的更新（2.2 式推导的结论形式）： $\Delta\theta = -\eta \thinspace \hat{m} / (\sqrt{\hat{v}} + \epsilon) - \eta \lambda \theta$（[Loshchilov & Hutter, 2017](../references.md)，arXiv:1711.05101，[A]）。clip 作用在 optimizer 之前，把 Adam 看到的梯度序列从 $g_t$ 换成 $`g'_t = c_t \thinspace g_t`$。分析这个交互只需要盯住 $c_t$ 的时间结构。

**常数缩放免疫**。若 $c_t \equiv c$（假想情形：每步同倍缩放），则 $m_t \to c \thinspace m_t$、 $v_t \to c^2 \thinspace v_t$，逐坐标比值 $\hat{m} / \sqrt{\hat{v}}$ 精确不变——Adam 对梯度的全局常数缩放免疫。这就是「Adam 自带尺度不变性」的精确版本，也是「clip 只是偶发缩放时几乎不伤 Adam」的底层原因。

**变系数情形**。 $c_t$ 随步变化时，免疫消失： $\hat{m} \approx \sum_t w_t \thinspace c_t \thinspace g_t$ 而 $\hat{v} \approx \sum_t w_t \thinspace c_t^2 \thinspace g_t^2$（ $w_t$ 为 EMA 权重），被裁步（ $c_t \ll 1$）在分子里的贡献被压得比分母更狠——方向统计系统性地向未裁步倾斜。这不是病态，是 clip 的代价形式：被压掉的那一步信息确实丢了。

按触发频率分两种 regime：

- **罕见触发（保险丝 regime）**。clip 只拦离群 spike。此时它保护的对象主要是 $v$：设典型梯度元素量级下 $\hat{v}$ 稳定在 $v_0$，一个元素幅度为典型值 $k$ 倍的 spike 会把该坐标 $v$ 瞬间抬高约 $(1-\beta_2) \thinspace k^2 v_0$。代入 LLM 惯例 $\beta_2 = 0.95$（出处见 2.2 的超参表）与 $k = 100$： $v$ 跳升约 $0.05 \times 10^{4} = 500$ 倍， $\sqrt{\hat{v}}$ 跳升约 22 倍，随后以时间常数 $1/(1-\beta_2) = 20$ 步指数回落—— spike 之后约 20 步里更新幅度被持续压缩（「宿醉」），同时 spike 方向在 $m$ 里驻留约 $1/(1-\beta_1) = 10$ 步。clip 在源头把 spike 截到 $\tau$，两笔账都不发生。
- **一致触发（归一化 regime）**。每步都裁时，Adam 的统计量相对「裁剪后梯度」自洽——分析 clip+Adam 组合时常见的**一致裁剪假设**（consistent clipping）即指此： $c_t = \tau / \lVert g_t \rVert$ 恒成立，组合行为接近对方向归一后的自适应方法。此 regime 下 clip 不再是保险丝而是算法本体的一部分，阈值 $\tau$ 与 LR 共同决定有效更新幅度，调参语义改变。

诚实的边界：以上是机制分析而非收敛定理；两种 regime 之间的中间态（触发率几十个百分点）没有干净的等价刻画，工程上用 §3 的触发率监控避开它。另有一个结构性事实值得登记：clip 的缩放系数 $c_t$ 是全参数共享的标量，不改变步内的逐坐标相对结构，因此它与 Adam 的逐坐标自适应是正交的两种「尺度处理」——一个管步间，一个管坐标间。

## 5. Loss spike：机制分类学预告与 GLM-130B 案例

GLM-130B 的技术报告为训练稳定性专辟一节（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]，§3），是本书稳定性案例的主锚点。要点：混合精度方案为 FP16 前向反向 + FP32 优化器态与主权重（Apex O2 口径）；训练初期即遭遇频繁 loss spike，且「随训练推进越来越频繁」；spike 形态分两种——部分无明确原因、自行恢复，部分以梯度范数骤升为前兆、数步后 loss spike 甚至 NaN。最终定位到 embedding 层异常梯度，修复手段是 embedding gradient shrink（EGS）：把 embedding 矩阵做一次前向恒等、反向缩放的改写，

$$
E \leftarrow \alpha \thinspace E + (1 - \alpha) \thinspace \mathrm{detach}(E) \qquad (3)
$$

**符号**： $E$：word embedding 矩阵； $\mathrm{detach}(\cdot)$：切断梯度（前向取值不变）； $\alpha \in (0, 1]$：收缩因子。

前向值不变（ $\alpha E + (1-\alpha) E = E$），反向时梯度经链式法则乘 $\alpha$——EGS 把这一路的梯度精确缩到 $\alpha$ 倍。GLM-130B 报告 $\alpha = 0.1$「wipes out most spikes」，且开销可忽略；最终完整训练只遭遇 3 次晚期发散，进一步收缩 embedding 梯度仍然有效。EGS 与式 (1) 的对比正好说明全局 clip 的盲区：global norm 被最大的贡献者主导，embedding 一路异常时，全局裁剪要么阈值被它抬高（其他层失去保护）、要么全模型陪它缩放（正常层的更新幅度被无辜压缩）；EGS 是针对单一路径的定点处理。架构侧的配套选择（DeepNorm 型 Post-LN 使主支数值有界）归 models 2.7 的 LN 变体讨论，此处只登记事实：GLM-130B 的稳定性是数值精度、架构选择、梯度端处理三层手段叠出来的，没有单一银弹。

把 GLM-130B 的散点记录整理成机制分类学（每一类的详拆、监控指标与干预手册归 7.3，本节只立框架）：

| 类别 | 机制 | 本章/本书触点 |
| --- | --- | --- |
| 数据触发 | 坏 batch（异常样本集中出现）把单步梯度推出分布外 | clip 兜底（式 (1)）；skip batch 归 7.3 |
| 优化器触发 | 二阶矩失配 / $\hat{v}$ 崩塌；Muon 的 attention logit 漂移 | §4 的宿醉账；2.5 §7 MuonClip |
| 数值触发 | FP16 溢出（attention score 超动态范围）；softmax 溢出账 | models 2.3 §4（fp16 在 logit ≈ 11 溢出）；1.3 混合精度 |
| 硬件触发 | 静默数据损坏（SDC）与 GPU 故障 | 7.3 / 7.4（Llama-3 中断统计口径） |

## 6. 稳定原语 placement 总图

一个训练 step 的数据流上，本书覆盖的稳定原语各有一个确定的作用点：

```
数据 batch    前向（模型内）        logits      loss        反向梯度        optimizer 更新      参数
   │              │                 │             │             │                │              │
   ▼              ▼                 ▼             ▼             ▼                ▼              ▼
 过滤/配比    QK-norm           z-loss       warmup     混合精度防护       grad clip       MuonClip 的
 (→7.1)      soft-capping     (→3.1)        (→2.3)      (→1.3)            式 (1) 本章     QK-clip (→2.5)
             (models 2.7)                                  EGS 式 (3)     weight decay
                                             │                              (→2.2 / 2.5)
                                             └── LR 端：warmup 与 schedule 改 LR 标量序列、不改数据流，贯穿全程
监控总线：grad norm（本章 §3，clip 返回值白送）· update ratio · logit max · attention entropy（指标体系归 7.3）
```

读法：每一列是数据流的一个截面，列内的原语只作用于该截面——QK-norm / soft-capping 改的是模型前向的数学形式（模型侧公式走读归 models 2.7，training 侧不重复）；z-loss 改的是 loss 对 logit 的惩罚项（归 3.1，router 侧谱系链 models 4.5 式 (12)、输出侧链 models 8.5 式 (7)，出处 [Zoph et al., 2022](../references.md)，arXiv:2202.08906，[B]）；grad clip 与 EGS 改的是进入 optimizer 之前的梯度；MuonClip 的 QK-clip 改的是 optimizer 更新之后的权重（2.5 §7）；warmup 与 schedule 改的是 LR 这个标量序列（2.3）。串扰最少的设计原则写在图上：每个问题在它的原生截面处理，跨截面的「代偿」（如用 clip 去压 logit 爆炸）能工作但代价是 §4 分析的统计扭曲。这张图的另一个用法是反着读：遇到一类新的不稳定现象时，先定位它发生在哪个截面，再检查该截面有没有对应的原语——没有，就说明该现象需要新机制，7.3 的干预手册正是按这个索引组织的。

## 总结

| 项 | 公式 / 口径 | 要点 |
| --- | --- | --- |
| global norm clip | 式 (1)： $c = \min(1, \tau / (\lVert g \rVert_2 + \varepsilon))$ 全体等比缩放 | 全局拼范数；只缩不放；作用于累加与 all-reduce 之后 |
| 方向保持 | $g' = c \thinspace g$；欧氏投影式 (2) | 逐值裁剪把 $(100,1)$ 扭 44.4°，范数裁剪偏差为 0 |
| 阈值 | 1.0 惯例：GPT-3 [B] / GLM-130B [A] / DeepSeek-V3 [A] / OLMo 2 [A] | 分位数监控 + 触发率为本书工程口径 |
| clip × Adam | $c_t$ 恒定则 $\hat{m}/\sqrt{\hat{v}}$ 免疫；spike 的 $v$ 宿醉约 $1/(1-\beta_2) = 20$ 步 | 罕见触发 = 保险丝；一致触发 = 归一化 regime（启发式，非定理） |
| GLM-130B 案例 | EGS 式 (3)， $\alpha = 0.1$ | grad norm spike 先行于 collapse 数步；embedding 梯度异常大几个数量级 |
| placement | §6 总图 | 每个原语守住自己的截面；spike 四类机制详拆归 7.3 |

## 练习

1.（推导）补全式 (2) 的完整论证：设 $g \neq 0$，证明 $\min_{\lVert h \rVert_2 \le \tau} \lVert h - g \rVert_2$ 的解唯一且为 $\min(1, \tau / \lVert g \rVert_2) \thinspace g$（Cauchy-Schwarz 等号条件给出唯一性）。再证：把约束换成 $\ell_\infty$ 盒子 $\lVert h \rVert_\infty \le \tau$ 后，解变为逐值裁剪 $h_i = \mathrm{sign}(g_i) \min(|g_i|, \tau)$——由此说明逐值裁剪同样是投影、只是投影的几何不同。
2.（计算）二维推广： $g = (M, 1)$、 $\tau = 1$。写出逐值裁剪后的方向角偏差 $\Delta\theta(M)$，求 $M \to \infty$ 的极限；再给范数裁剪在同一 $M$ 扫描下的偏差（应为恒 0）。由此说明「方向扭曲随梯度病态程度无界增长」。
3.（代码）跑通 §1 教学版与 `torch.nn.utils.clip_grad_norm_` 的等价断言；然后构造合成梯度序列：前 100 步元素 iid $\mathcal{N}(0, 1)$，第 101 步全体乘 100（模拟 spike），分别在「裁 / 不裁」两路下驱动 Adam（ $\beta_1 = 0.9$、 $\beta_2 = 0.95$），画出 $\sqrt{\hat{v}}$ 的轨迹，验证宿醉窗口约为 $1/(1-\beta_2) = 20$ 步量级，并解释为什么被裁一路的 $v$ 曲线无凸起。
4.（分析）设 spike 出现频率为每 $K$ 步一次、幅度倍数 $k$，在一致裁剪假设与保险丝 regime 的分界上估算：当 $K$ 小到什么程度时， $\hat{v}$ 的统计再也无法「忘记」上一个 spike（提示：比较 $K$ 与 $1/(1-\beta_2)$）？由此给触发率监控一个量化的报警线。

## 参考

- [PyTorch, pytorch repo](../references.md)：教学参照（key: pytorch-repo）——`torch/nn/utils/clip_grad.py` 中 `clip_grad_norm_` / `_get_total_norm` / `_clip_grads_with_norm_` 为 §1 走读对象
- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners（GPT-3），arXiv:2005.14165，[B]（key: gpt3）——global norm 1.0 惯例的可查证出处
- [Zeng et al., 2022](../references.md)：GLM-130B: An Open Bilingual Pre-trained Model，arXiv:2210.02414，[A]（key: glm-130b）——§2.2 裁剪值 1.0；§3 spike 记录、grad norm 先行指标与 EGS
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——梯度裁剪范数 1.0
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——超参表 Gradient Clipping 1.0
- [Loshchilov & Hutter, 2017](../references.md)：Decoupled Weight Decay Regularization，arXiv:1711.05101，[A]（key: adamw）——§4 的 AdamW 更新形式
- [Zoph et al., 2022](../references.md)：ST-MoE: Designing Stable and Transferable Sparse Expert Models，arXiv:2202.08906，[B]（key: st-moe）——z-loss 谱系出处，placement 图的 3.1 指针
