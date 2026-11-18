# 3.1 CE 家族与数值稳定实现（The Cross-Entropy Family and Its Numerically Stable Implementation）

> 交叉熵在论文里只有一行公式，放进生产训练却要多出六层工程。本章逐层展开：训练侧的 per-token 平均与 mask 口径；logsumexp 的数值稳定推导（与 models 2.3 §4 的 max 减法互为对偶）；fused / chunked CE 的显存全账与 torch / Liger / Megatron vocab 并行三处接口；label smoothing 的目标改写与有限最优 logit；z-loss 的梯度推导，以及 z-loss / QK-norm / soft-capping 的 injection point 对照表；辅助 loss 进总账的工程接口 | 前置：1.2 式 (5)（softmax-CE 融合反向）、1.2 §10（fused CE 机制）、1.3 §2（autocast 精度策略）、1.5 式 (4)（变长归一化陷阱）、models 1.1（NLL / CE 同一性）、models 8.3 式 (1)（loss mask）、models 2.3 §4（softmax 数值溢出账）| 锚点：linkedin/Liger-Kernel [A]、`torch.nn.CrossEntropyLoss`（pytorch repo，教学参照 + 官方实现）、Megatron-LM vocab 并行 CE [A]、ST-MoE [B]、Chameleon [B]、label smoothing 原文 [B]、focal loss 原文 [B]

## 1. 训练侧 CE 口径：per-token 平均与 mask

训练框架里的交叉熵长什么样？答案可以先从 PyTorch 的文档读出两条硬规定（pytorch/pytorch repo `torch/nn/modules/loss.py`，教学参照 + 官方实现，2026-11-17 核实）。第一，输入是**未归一化的 logits**——文档原文口径：不要求逐类为正、不要求和为 1。第二，`reduction='mean'` 时按**未屏蔽目标数**归一，分母是「未屏蔽位置的权重和」。模型侧的定义链在 models 1.1 式 (4)-(6)（NLL = CE = MLE、CE = H + KL），本章只固定训练侧口径。

一条样本位置 $t$ 上的逐 token CE，写成 logits 的显式函数：

$$
\ell_t = -z_{t, y_t} + \log \sum_{v=1}^{V} \exp(z_{t, v}) \qquad (1)
$$

**符号**： $z_t \in \mathbb{R}^{V}$：位置 $t$ 的输出 logits（LM 头 matmul 的输出）； $y_t$：目标 token id； $V$：词表大小。第二项即 logsumexp，其数值稳定形式是 §2 的主题。式 (1) 对 $z_t$ 的梯度有闭式 $p - e_y$（1.2 式 (5) 已推导），本章不重复。

整批损失是带 mask 的平均：

$$
L = \frac{1}{N_{\mathrm{tok}}} \sum_{b, t} m_{b, t} \thinspace \ell_{b, t}, \qquad N_{\mathrm{tok}} = \sum_{b, t} m_{b, t}, \qquad m_{b, t} \in \lbrace 0, 1 \rbrace \qquad (2)
$$

**符号**： $m_{b, t}$：loss mask——1 计入、0 屏蔽。mask 的语义（SFT 里 prompt 不计、VLM 里视觉 token 不计；labels 置 $-100$ 走 `ignore_index` 约定）在模型侧有定义链，见 models 8.3 式 (1)。这里只登记训练侧三条口径：

1. **屏蔽的是监督，不是计算**。被 mask 的位置照常前向、照常产生 logits，只是 $\ell$ 不进式 (2)。mask 省不下任何前向 FLOPs（models 8.3 §1 同口径）。
2. **mean 归约的分母随 mask 浮动**。变长 response 下，各 micro-batch 的 $N_{\mathrm{tok}}$ 不相等。这时「各自 mean 再等权平均」不等于全局 token 平均，修正权重是 $n_i / \sum_j n_j$。推导与反例归 1.5 式 (4)，这里不重复。
3. **loss 有地板**。 $\mathrm{CE} = H(p) + \mathrm{KL}(p \Vert q_\theta) \ge H(p)$（models 1.1 式 (6)）——横轴再加 step，纵轴也降不到 0。跨数据配比比较 loss 绝对值时，先核对地板。梯度一侧还有一个口径：mean 归约的 $1 / N_{\mathrm{tok}}$ 因子会进全部梯度（1.2 §4 的归约语义注记）。它与 gradient accumulation 的 $1/m$ 是相乘关系（1.5 式 (3)），实现里两处各放一次、只放一次。

数据流与 shape（单卡口径；分块与并行版本归 §3）：

```
H [B, T, d] ── LM 头 matmul（2BTdV FLOPs）──► logits z [B, T, V]   ← 2BTV bytes（bf16，§3 的账）
                                                  │  行内 logsumexp（式 (3)）
                                                  ▼
                                          逐 token ℓ [B, T]
                                                  │  ⊙ mask m [B, T]
                                                  ▼
                                          L 标量（式 (2)）
```

## 2. logsumexp 数值稳定：max 减法在 log 域

式 (1) 按字面直接实现有一道硬上界： $\exp(z)$ 会溢出。fp32 与 bf16 下 $z \approx 88$ 到顶（ $e^{88} \approx 1.7 \times 10^{38}$，fp32 上限约 $3.4 \times 10^{38}$），fp16 更早， $z \approx 11$ 即溢出（上限 65504）——完整溢出账在 models 2.3 §4。训练侧的对策和 attention softmax 用的是同一个技巧：把最大值从指数里提出来。

$$
\log \sum_{v=1}^{V} e^{z_v} = \log \left( e^{m} \sum_{v=1}^{V} e^{z_v - m} \right) = m + \log \sum_{v=1}^{V} e^{z_v - m}, \qquad m = \max_v z_v \qquad (3)
$$

**符号**： $m$：行内最大 logit。第一步提取公因子 $e^{m}$，第二步 log 拆积。右端逐项受控：平移后最大值恰为 0，被加项落在 $(0, 1]$，和式夹在 $[1, V]$，于是

$$
0 \le \log \sum_v e^{z_v - m} \le \log V
$$

这样一来，logsumexp 被锁在 $[m, \hspace{4mu} m + \log V]$ 的确定区间内，永不溢出。反方向的下溢无害： $z_v - m$ 很负的项 exp 后下溢为 0，丢掉的只是一个本就趋零的概率贡献（models 2.3 §4 同口径）。

这个技巧和 models 2.3 §4 是一对对偶，值得说破。那边的 max 减法稳住的是 attention 前向的 softmax：在概率域里逐行归一化。这里的 logsumexp 移项稳住的是损失本体：在 log 域里操作，不需求出显式概率。两处共用同一条不变性——softmax 对 logit 整体平移不变，其微分形式就是 softmax Jacobian 行和为 0（models 2.3 §3.3）。式 (3) 对 $z_j$ 求导也直接回到 softmax： $\partial \log \sum_v e^{z_v} / \partial z_j = e^{z_j} / Z = p_j$。它与式 (1) 的第一项合成，就是 1.2 式 (5) 的 $p - e_y$。稳定前向与融合反向在同一条恒等式上合拢，全程不需要物化概率分布 $p$。

实现口径两条。其一，PyTorch 的 `log_softmax` docstring 写明：与「先 softmax 再取 log」数学等价，但分两步做更慢且数值不稳定（"mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable"，`torch/nn/functional.py`，2026-11-17 核实）。融合形式（式 (3)）因此是生产默认。其二，混合精度下这一段的 dtype 不是 bf16。autocast 的算子 cast 策略把 softmax / log_softmax / CE 类保持在 fp32（1.3 §2 口径，`torch/amp/autocast_mode.py`）。所以 §3 的显存账里，log_softmax 的 save 按 4 字节计，不是 2 字节。

## 3. fused / chunked CE：全账与接口

机制已在 1.2 §10 交付：分块投影 → 块内 logsumexp → 逐块 loss 与 $p - e_y$ 梯度，logits 永不全量物化。本节给全账与接口。先看基线数字： $V$ = 128256、 $T$ = 4096 时，单序列 bf16 logits 为 $128256 \times 4096 \times 2$ bytes ≈ 0.98 GiB。

显存全账如下（朴素路径三笔 vs FLCE 增量峰值，单位 GiB； $M_c$ = 2048 为 Liger 典型块长口径，1.2 §10）：

| 项 | 字节公式 | $B$=1, $T$=4096 | $B$=8, $T$=4096 |
| --- | --- | --- | --- |
| bf16 logits | $2BTV$ | 0.98 | 7.83 |
| fp32 log_softmax save（autocast 口径） | $4BTV$ | 1.96 | 15.66 |
| logits 梯度 buffer | $2BTV$ | 0.98 | 7.83 |
| 朴素合计 | $8BTV$ | 3.91 | 31.31 |
| FLCE 增量峰值（块 logits，bf16） | $2 M_c V$ | 0.49 | 0.49 |

读法。save 项按 fp32 计（§2 末的 autocast 口径）。FLCE 一行是与 $B$ 无关的增量峰值：分块把峰值从 $O(BTV)$ 压到 $O(M_c V)$，块长就是显存旋钮（1.2 §10 的推导）。 $B$=8 时省下的约 31 GiB，相当于一张 H100 80GB 显存的近四成。注意 1.2 §10 给过「最省两笔」口径（每序列约 2.9 GiB、 $B$=8 约 24 GiB）；本表把梯度 buffer 一并入账，所以数字更大。

FLOPs 账。LM 头投影前向一次， $2BTdV$；反向两处同形 matmul（ $dH$ 与 $dW$ ），再 $4BTdV$。FLCE 不省任何 matmul：默认路径把反向折进前向的分块循环（官方 docstring 口径：CE 是末端算子、上游梯度恒为 1，梯度可在 forward 里算完）；backward 重算路径多付一次投影，反向/前向从 ≈2 变 ≈3（1.2 §11 总表）。省的是显存与访存，不是 FLOPs。Liger 论文报告的平均收益：对照 HuggingFace 实现，吞吐 +20%、显存 −60%（[Hsu et al., 2024](../references.md)，arXiv:2410.10989，[A]）。还有一条结构性事实：logits 是损失的直接输入，不是某层的中间值，gradient checkpointing 的重算逻辑对它不适用（1.2 §10）。fused CE 是这类显存的解法本身——这是它单独立账的根据。

生产接口三处（路径均 2026-11-17 核实）：

1. **PyTorch**：`torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)`（`torch/nn/modules/loss.py`，教学参照 + 官方实现）。target 有两种形态：类别 id（long；文档注明此形态等价于 LogSoftmax + NLLLoss 的组合），或类别概率分布（float，文档列举的用途即 blended labels 与 label smoothing）。§4 的 label smoothing 走的就是概率 target 这条路径。
2. **Liger**：`LigerFusedLinearCrossEntropyLoss`（`src/liger_kernel/transformers/fused_linear_cross_entropy.py`；kernel 本体在 `src/liger_kernel/ops/fused_linear_cross_entropy.py` 的 `LigerFusedLinearCrossEntropyFunction`）。注意它的 forward 签名：`(lin_weight, _input, target)`——吃隐状态与输出头权重，**不吃 logits**。接口形状的改变本身就是机制：「CE 前移进投影」，这正是 logits 永不物化的接口表达。
3. **Megatron-LM（TP 下的 vocab 并行 CE，5.1 前瞻）**：`megatron/core/ops/loss/cross_entropy.py` 的 `VocabParallelCrossEntropy`（旧路径 `megatron/core/tensor_parallel/cross_entropy.py` 现为 deprecation shim）。 $V$ 维按 TP rank 切分后，每卡只持有 $[B, T, V/t]$。前向用三次 $[B, T]$ 量级的 all-reduce 完成全局 logsumexp：行内 max（MAX 归约）→ 局部 $\sum \exp$ 求和（SUM）→ 目标 logit 聚合（SUM，非持有 rank 置 0），最后逐位置 loss = logsumexp − predicted logit。源码里对应三段分工：`calculate_logits_max` / `calculate_predicted_logits` / `calculate_cross_entropy_loss`。通信量与 $V$ 无关：每 token 只有三个 fp32 标量。对照「all-gather 全量 $[B, T, V]$ logits 再算 CE」的方案，通信量差一个 $V$ 量级。logsumexp 的平移结构（式 (3)）正是它能被拆成「先约 max、再约和」两次归约的原因。

教学实现不重复造：1.2 §10 的 `ChunkedLinearCE` 就是本节接口的 Python 形态，练习 4 会在其上加 z-loss。

三处接口放在一张账上看，分工就清楚了。torch 接口定义**语义基准**：归约、mask、label_smoothing 的标准语义都以它的 docstring 为准。Liger 接口解决**单卡显存**（ $O(BTV)$ → $O(M_c V)$）。Megatron 接口解决**词表切分**（ $V$ 维随 TP 分片，通信停在 $[B, T]$ 量级）。三者不互斥：生产栈里的常见组合是 TP 内层走 vocab 并行 CE，外层仍装不下时再套 chunked 实现。还有一个容易被忽略的事实：数值口径是三层叠加的。式 (3) 的 max 减法、autocast 的 fp32 兜底、Megatron 源码里 `vocab_parallel_logits.float()` 的显式升精度，各自独立地把同一段计算固定在 fp32。训练侧 CE 的数值稳定不是某一层给的，是三层冗余叠出来的。

## 4. label smoothing：目标分布的改写

PyTorch 的 `label_smoothing` 参数 docstring 把机制写成一句话：target 变成「原 ground truth 与均匀分布的混合」，并指向原始出处（[Szegedy et al., 2015](../references.md)，arXiv:1512.00567，[B]）。写出公式：

$$
q'_v = (1 - \varepsilon) \thinspace \delta_{v, y} + \frac{\varepsilon}{V}, \qquad L_{\mathrm{ls}} = - \sum_v q'_v \log p_v = (1 - \varepsilon) \thinspace H(\delta_y, p) + \varepsilon \thinspace H(u, p) \qquad (4)
$$

**符号**： $\varepsilon \in [0, 1]$：平滑强度； $u$：均匀分布 $u_v = 1/V$； $\delta_{v,y}$：Kronecker delta。右端是原文的等价写法：LS = 原 CE 与「对均匀分布的 CE」的凸组合。

梯度形态几乎不变。softmax-CE 对任意目标分布 $q$ 的 logit 梯度恒为 $p - q$（1.2 式 (5) 的 one-hot 是它的特例）：

$$
\frac{\partial L_{\mathrm{ls}}}{\partial z_j} = p_j - q'_j \qquad (5)
$$

差别全在目标值里。逐项看式 (5)。目标位的梯度是 $p_y - (1 - \varepsilon + \varepsilon/V)$，它在 $p_y = 1 - \varepsilon + \varepsilon/V$ 处变号。对比朴素 CE：目标位梯度 $p_y - 1$ 只在 $p_y = 1$ 处才归零。LS 把推压的终点从无穷远收到了有限置信度。最优解可以直接算： $\mathrm{CE}(q', p)$ 在 $p = q'$ 处取最小（KL 非负），对应 logit gap

$$
z_y - z_j = \log \frac{1 - \varepsilon + \varepsilon / V}{\varepsilon / V} \approx \log \frac{V \thinspace (1 - \varepsilon)}{\varepsilon} \qquad (6)
$$

代入 $V$ = 128256、 $\varepsilon$ = 0.1：gap ≈ 14.0 nats。这是一个有限常数，不再是朴素 CE 下无界增长的 $\log(p_y / p_j)$。原文的机制表述（防止最大 logit 远大于其余）由此有了定量版本：LS 是对 logit 幅度的内建制动，制动点由 $\varepsilon$ 解析给出。

两个训练侧后果。其一，**loss 地板被抬高**： $L_{\mathrm{ls}} \ge H(q')$。代入 $V$ = 128256、 $\varepsilon$ = 0.1，算得 $H(q') \approx 1.50$ nats——开了 LS 的曲线与没开的不能直接比绝对值。其二，**现状口径**：本书锚定的公开 LLM 预训练配方（GPT-3 / Llama-2 / Llama-3 / DeepSeek-V3 / OLMo 2，逐家出处见 2.2 §7 表）在训练目标与超参披露中均无 label smoothing 字样。这是「披露中缺席」类型的证据，不等同于「声明弃用」。机制读法（LS 防过自信的收益在 token 级海量监督下被摊薄）属本书分析，无单一出处。

家族位置记一句。focal loss（[Lin et al., 2017](../references.md)，arXiv:1708.02002，[B]）给逐样本 CE 乘上调制因子 $(1 - p_t)^{\gamma}$（原文主口径 $\gamma$ = 2），把易分类样本的梯度压掉，出处场景是 dense 目标检测的前景-背景极端不均衡。它与 LS 恰好是 CE 家族的两个改造方向：LS 改**目标分布**，focal 改**逐样本梯度权重**。LLM 预训练里每个 token 的监督都计入梯度，不存在「海量易负样本淹没信号」的结构——focal 针对的问题形态在语言建模里不存在，本书不作展开。

## 5. z-loss：梯度推导与 injection point 对照

先 recap 形式： $L_z = (1/M) \sum_t (\log Z_t)^{2}$，逐位置惩罚 log-partition。router 侧定义链 models 4.5 式 (12)（ST-MoE），输出侧链 models 8.5 式 (7)（Chameleon， $\mu$ = $10^{-5}$）。谱系以 ST-MoE 原文为准：router z-loss 是 Mesh TensorFlow 代码库中 final-softmax z-loss 的改造（[Zoph et al., 2022](../references.md)，arXiv:2202.08906，[B]，其 §3 引言的归因句与 §3.3 的本体定义）。本章的增量是梯度推导与对照表。

逐位置项对 logit 的梯度，链式两步，不跳步： $\partial (\log Z)^{2} / \partial z_j = 2 \log Z \cdot (1/Z) \cdot \partial Z / \partial z_j$，而 $\partial Z / \partial z_j = e^{z_j}$，于是

$$
\frac{\partial L_{z, t}}{\partial z_{t, j}} = 2 \thinspace \log Z_t \cdot p_{t, j} \qquad (7)
$$

**符号**： $p_t = \mathrm{softmax}(z_t)$； $Z_t = \sum_v e^{z_{t, v}}$。式 (7) 读作：按概率质量成比例地下压每个 logit，力度正比于当前的 $\log Z$。

把 CE 与 z-loss 的梯度分别在平移方向上做内积，两者的分工立刻显形。记全 1 向量 $\mathbf{1}$ 为 logit 平移方向。CE 梯度 $g_{\mathrm{CE}} = p - e_y$ 满足 $\langle g_{\mathrm{CE}}, \mathbf{1} \rangle = \sum_j p_j - 1 = 0$：CE 对平移方向完全盲，这正是 softmax 平移不变性的微分形式。z-loss 梯度 $g_z = 2 \log Z \cdot p$ 沿平移方向的分量是 $\langle g_z, \mathbf{1} \rangle = 2 \log Z$。结论：词表 softmax 的 $V$ 个自由度里，CE 管 $V - 1$ 个相对方向，z-loss 管剩下的平移方向，各管一段。CE 对 $\log Z$ 还有一股力，梯度下降下可以算出来： $\mathrm{d} \log Z / \mathrm{d} t = p^{\top} (-\eta \thinspace g_{\mathrm{CE}}) = \eta \thinspace (p_y - \lVert p \rVert^{2})$。等号是逐字代数；力的符号有条件。目标位为 argmax 时 $\sum_j p_j^{2} \le p_y \sum_j p_j = p_y$，力非负（支撑非退化时严格为正）。训练中后期，模型在多数位置上已把正确类放上 argmax，CE 持续把 $\log Z$ 往上抬，z-loss 正是这个上抬通道的配重。训练早期正确类未必是 argmax，该力可以反向——算例： $p = (0.4, 0.6)$、目标为前者， $0.4 - 0.52 \lt 0$。这不动摇 z-loss 的职责：它的不动点 $\log Z = 0$ 与 CE 的方向无关，平移自由度始终由它锁定。

z-loss 单独作用时，动力学有闭式解。设梯度流 $\dot z = -2 \lambda_z \log Z \cdot p$，则

$$
\frac{\mathrm{d} \log Z}{\mathrm{d} t} = \sum_j \frac{\partial \log Z}{\partial z_j} \thinspace \dot z_j = \sum_j p_j \cdot \left( -2 \lambda_z \log Z \cdot p_j \right) = -2 \lambda_z \lVert p \rVert^{2} \log Z \qquad (8)
$$

**符号**： $\lambda_z$：z-loss 系数。 $\log Z$ 以速率 $2 \lambda_z \lVert p \rVert^{2}$ 指数回归 0，不动点是 $\log Z = 0$（即 $Z$ = 1）。 $\log Z$ 变号时力的方向随之反转，两侧都指向不动点——这是一个自稳定回路。速率因子 $\lVert p \rVert^{2} \in [1/V, 1]$：分布越尖锐，水平回归越快。垂直于平移方向的残余分量 $2 \log Z \cdot (p - \mathbf{1} / V)$ 把高概率 logit 压得比低概率的更狠，会轻微压平分布——这是固定平移自由度的附带代价。实测看 ST-MoE 的消融：无质量损失。baseline 6 次跑 4 次稳定；update clipping 3/3 稳定但质量受损（验证集负对数困惑度 -4.206 vs baseline 的 -1.755）；router z-loss 3/3 稳定且质量无损（-1.741，其 Table 4，[B]）。同一论文还记录了对 attention logits 加 z-loss 同样改善稳定性（其 footnote 口径）。

量级账能解释系数为什么都那么小。初始化附近 logits 在零上下， $\log Z \approx \log V$： $V$ = 128256 时约 11.8。式 (7) 的逐坐标幅度可达 $2 \times 11.8 \approx 23.5$（ $p_j \approx 1$ 的行），比 CE 梯度（逐坐标 $\le 1$）大一个量级有余。要让 z-loss 只当「平移方向的弱弹簧」而不喧宾夺主，系数就得压在 $10^{-3}$ 到 $10^{-5}$。两个真实取值：ST-MoE 取 $c_z$ = 0.001（超参扫描定；原文 Appendix B 的扫描显示 $\ge 10^{-2}$ 会把 z-loss 项本身压到近零）；Chameleon 输出侧取 $\mu$ = $10^{-5}$（models 8.5 式 (7) 口径）。另有一条精度协同：ST-MoE 的机制动机是 bf16 舍入误差随数值幅度增长，z-loss 压低 logit 幅度，也就压住了指数函数的输入误差；论文同时明确所有被取指数的张量转 fp32（[B]，其 §3.4 aside 口径）。惩罚加在 loss 上，收益落在数值精度上。

三个同类手段的 injection point 对照如下（模型侧公式全部纯文字指引，不重复推导）：

| 手段 | 作用点 | 约束形式 | 有界性 | 梯度行为 | 采用记录 |
| --- | --- | --- | --- | --- | --- |
| z-loss | loss 端：router logits 或输出 logits 上的加性惩罚 | $(\log Z)^{2}$ 二次罚 | 软约束（无硬界） | $2 \log Z \cdot p$， $\log Z = 0$ 处归零 | ST-MoE router 侧（式 (7) 谱系，[B]）；Chameleon 输出侧 $\mu$ = $10^{-5}$ [B] |
| QK-norm | 前向内部： $q, k$ 归一化后做点积 | 结构性归一（非 loss） | attention logit 硬上界（变体口径 $\sqrt{d_k}$） | 无额外 loss 项；梯度经归一化 Jacobian 走 | OLMo 2、Gemma 3（models 2.7 §7） |
| soft-capping | 前向内部：logits 过 $c \tanh(z / c)$ | 逐值软钳（非 loss） | $\vert z' \vert \lt c$ 硬有界 | $1 - \tanh^{2}$ 渐弱、不归零 | 仅 Gemma 2（attn 50.0 / final 30.0）；Gemma 3 已弃用（models 2.7 §8） |

读法。z-loss 在**目标函数**里加一项；QK-norm 与 soft-capping 改**前向函数**本身。前者不动模型的前向语义，推理时扔掉即可；后两者是架构的一部分，推理时必须在场。作用截面不同，失效模式也不同：z-loss 抑制不了前向内部的数值爆炸（它只在 loss 端反馈），soft-capping 管不住被钳位置之外的漂移。全局面的 placement 图见 2.4 §6。

## 6. 辅助 loss 的工程接口：加权与梯度耦合

总损失是加性组合：

$$
L_{\mathrm{tot}} = L_{\mathrm{CE}} + \sum_k \lambda_k \thinspace L_k \qquad (9)
$$

**符号**： $L_k$：各辅助项（MoE 负载均衡、z-loss、MTP 等）； $\lambda_k$：权重系数。ST-MoE 的实例即 $L_{\mathrm{tot}} = L_{\mathrm{CE}} + c_B L_B + c_z L_Z$（[Zoph et al., 2022](../references.md)，[B]，其式 (6) 口径）。各辅助项的模型侧定义不在本章：MoE aux loss 链 models 4.5 式 (5)(6)，MTP 目标链 models 4.5 对应节。autograd 层面，式 (9) 的梯度就是各项梯度之和（1.1 的线性性）。所以「加权」真正调的是**梯度量级配比**，不是 loss 数值配比。一个量级参照：z-loss 原始梯度幅度比 CE 大约 20 倍（§5 的量级账）， $c_z$ = 0.001 的实际含义是把它的梯度压到 CE 的约 2% 水平。调 $\lambda$ 的正确顺序：先看两项的梯度范数量级，再谈 loss 值配比。

工程接口三条。其一，**归一化语义与 CE 同坑**：辅助项的 mean 归约在变长 micro-batch 下，踩的是与 1.5 式 (4) 相同的口径陷阱。MoE aux loss 还额外依赖 batch 统计量（ $f_i$ 与 $P_i$ 按 batch 内 token 统计，models 4.5 式 (5)），micro-batch 切分与 DP 副本划分都会改变它的统计底数——换并行配置时对账先查这类「依赖 batch 统计的 loss」。其二，**统计量共享**：z-loss 要的 $\log Z$ 与 CE 的 logsumexp 是同一个标量。fused 实现里同一趟行内归约可顺带产出它（1.2 §10 的块结构上加一个标量累加器），不必为 z-loss 重扫一遍 $[M, V]$。教学形态如下（教学用，非优化实现）：

```python
import torch

def ce_with_zloss(logits, y, c_z):
    """教学用，非优化实现。式 (1)(3) 的稳定 CE 加式 (7) 的 z-loss，共享同一趟 logsumexp。"""
    m = logits.max(dim=-1, keepdim=True).values
    logZ = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))  # 式 (3)
    ce = (logZ - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)).mean()   # 式 (1) + 全员 mean（无 mask）
    return ce + c_z * logZ.square().mean()                                # 式 (9) 两项
```

其三，**日志分账**：生产监控把 $L_{\mathrm{CE}}$ 与各 $\lambda_k L_k$ 分行记录。总 loss 的下降里混着系数可调的项，跨配置比较时只有分账后的 CE 可比（与 1.5 §3 的日志口径同源）。

## 总结

| 主题 | 公式 / 结论 | 要点 |
| --- | --- | --- |
| 训练侧 CE 口径 | 式 (1)(2) | per-token 平均；mask 屏蔽监督不屏蔽计算；mean 按未屏蔽数归一；变长陷阱归 1.5 式 (4) |
| logsumexp 稳定 | 式 (3)，值域 $[m, \hspace{4mu} m + \log V]$ | max 减法的 log 域对偶（models 2.3 §4）；梯度回 $p - e_y$ |
| fused CE 账 | $8BTV$ 朴素 vs $2 M_c V$ 分块 | $B$=8、 $T$=4096：31.3 GiB → 0.49 GiB；FLOPs 不省；Liger 平均吞吐 +20% / 显存 −60% [A] |
| 生产接口 | torch / Liger / Megatron | Liger 接口不吃 logits；TP 下 3 次 $[B, T]$ all-reduce 完成 logsumexp |
| label smoothing | 式 (4)(5)(6) | 梯度 $p - q'$；最优 logit gap ≈ $\log(V(1-\varepsilon)/\varepsilon)$ ≈ 14 nats；地板 $H(q') \approx 1.50$ nats；公开 LLM 配方披露中缺席 |
| focal loss | $(1 - p_t)^{\gamma}$ 调制 | 改逐样本梯度权重（对照 LS 改目标分布）；LLM 预训练无对应结构 [B] |
| z-loss | 式 (7)(8) | 梯度 $2 \log Z \cdot p$；CE 盲于平移方向、z-loss 补位； $\log Z$ 指数回归 0；ST-MoE $c_z$ = 0.001 / Chameleon $\mu$ = $10^{-5}$ [B] |
| 辅助 loss 接口 | 式 (9) | 加权 = 调梯度量级配比；batch 统计类 loss 对并行切分敏感； $\log Z$ 与 logsumexp 共享一趟归约 |

## 练习

1.（推导）补全式 (3) 的证明并加两个界：(a) 证明 $\log \sum_v e^{z_v} \in [m, \hspace{4mu} m + \log V]$；(b) 设所有非最大项满足 $z_v - m \le -20$，给出只保留最大项时 logsumexp 的绝对误差上界（以 $V$ 表出），由此说明「下溢无害」的定量含义；(c) 对式 (3) 求 $z_j$ 的导数，验证回到 softmax——max 减法不改变梯度。
2.（计算）label smoothing 的账，取 $V$ = 128256：(a) $\varepsilon$ = 0.05 / 0.1 / 0.2 三档各算最优 logit gap（式 (6)）与地板 $H(q')$；(b) 解释为什么同一验证集上 $\varepsilon$ = 0.1 的曲线不能与 $\varepsilon$ = 0 的直接比 loss 绝对值，给出可比的换算口径（提示：减地板，或回到 KL）。
3.（推导）z-loss 动力学：(a) 证明式 (8) 的不动点只有 $\log Z = 0$ 且两侧力均指向它；(b) 把 CE 对 $\log Z$ 的作用力 $\eta \thinspace (p_y - \lVert p \rVert^{2})$（符号条件见 §5：目标位为 argmax 时非负）与 z-loss 的回归力放进同一个梯度下降步，推出力平衡水平 $\log Z^{\ast} = (p_y - \lVert p \rVert^{2}) / (2 \thinspace c_z \thinspace \lVert p \rVert^{2})$，并代入 $p_y$ = 0.9、 $\lVert p \rVert^{2}$ = 0.85、 $c_z$ = 0.001 估算其量级——该组参数下目标位是否 argmax、力是否为正，先判后代。
4.（代码）在 1.2 §10 的 `ChunkedLinearCE` 上加 z-loss：前向在块内 logsumexp 处顺手累加 $\sum (\log Z)^{2}$，反向按式 (7) 把 $2 c_z \log Z \cdot p$ 并进 $p - e_y$（教学用，非优化实现）。用 `torch.autograd.gradcheck`（fp64）验证，并与「物化 logits 的朴素实现 + 显式 z-loss」对拍 loss 与梯度。

## 参考

- [Zoph et al., 2022](../references.md)：ST-MoE: Designing Stable and Transferable Sparse Expert Models，arXiv:2202.08906，[B]（key: st-moe）——z-loss 公式（原文式 (5)）、Mesh TensorFlow 谱系、 $c_z$ = 0.001 与总 loss（原文式 (6)）、Table 4 稳定性消融
- [Chameleon Team, 2024](../references.md)：Chameleon: Mixed-Modal Early-Fusion Foundation Models，arXiv:2405.09818，[B]（key: chameleon）——输出侧 z-loss $\mu$ = $10^{-5}$（models 8.5 式 (7) 的转引口径）
- [Szegedy et al., 2015](../references.md)：Rethinking the Inception Architecture for Computer Vision，arXiv:1512.00567，[B]（key: label-smoothing）——式 (4) 的目标改写与混合分解
- [Lin et al., 2017](../references.md)：Focal Loss for Dense Object Detection，arXiv:1708.02002，[B]（key: focal-loss）—— $(1 - p_t)^{\gamma}$ 调制因子与类别不均衡场景
- [Hsu et al., 2024](../references.md)：Liger Kernel: Efficient Triton Kernels for LLM Training，arXiv:2410.10989，[A]（key: liger-kernel）——FLCE 机制与吞吐 +20% / 显存 −60% 收益口径；repo `src/liger_kernel/transformers/fused_linear_cross_entropy.py`、`src/liger_kernel/ops/fused_linear_cross_entropy.py`
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照 + 官方实现（key: pytorch-repo）——`torch/nn/modules/loss.py`（CrossEntropyLoss 的 reduction / ignore_index / label_smoothing 口径）、`torch/nn/functional.py`（log_softmax 数值注记）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——repo `megatron/core/ops/loss/cross_entropy.py`（`VocabParallelCrossEntropy` 的三次 $[B, T]$ all-reduce）
