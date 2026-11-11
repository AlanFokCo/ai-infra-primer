# 4.8 前沿方向（Frontier Outlook）

> 本章回收 4.1–4.7 的账与结构，盘点六个仍在移动的方向：吸收态 diffusion LLM（forward 逐 token 腐蚀、reverse 参数化、ELBO 塌缩为加权 CE；与 AR 的 decode 账公式级对比；LLaDA / SEDD）、MoD 与 early-exit 的深度稀疏、differential attention 的双 softmax 消噪、TTT 与 looped-depth recurrence、test-time scaling 与 agentic 负载对四堵墙（4.6）的压力反转；每项标注成熟度；native multimodal 归 P8.5 | 前置：1.1（AR 分解与 NLL）、1.3（训推全生命周期账）、2.4（KV cache 账）、3.1（mask 与 objective）、4.1–4.7 | 锚点：LLaDA（arXiv:2502.09992，ML-GSAI/LLaDA）[A]、SEDD（arXiv:2310.16834，louaaron/Score-Entropy-Discrete-Diffusion）[A]、Differential Transformer（arXiv:2410.05258，代码在 microsoft/unilm）[A]、MoD（arXiv:2404.02258）[B]、TTT（arXiv:2407.04620）[B]、DeepSeek-V3.2（arXiv:2512.02556）[A]、Kimi K3（arXiv:2607.24653）[A]

## 1. Diffusion LLM：把链式法则换成逐步去掩码

LLaDA 把掩码扩散（masked diffusion）推到 8B 参数、从头训练、并与同规模自回归（AR）基线同量级对照（[Nie et al., 2025](../references.md)，arXiv:2502.09992，[A]）——文本扩散第一次有了「同量级 AR 对照」意义上的证据，而不再是玩具规模的存在性声明。本节给出它的完整形式化，然后与 AR 做公式级对账。

**Forward：逐 token 独立腐蚀。** 词表外加一个吸收态（absorbing state）[M]；调度 $\alpha_t$ 从 $\alpha_0 = 1$ 单调降到 $\alpha_1 = 0$。每个位置独立地以概率 $1 - \alpha_t$ 被替换为 [M]：

$$
q\left(x_t^{i} \mid x_0^{i}\right) = \mathrm{Cat}\big(\alpha_t\, x_0^{i} + (1-\alpha_t)\, e_M\big) \qquad (1)
$$

**符号**：$x_0^{i}$：第 $i$ 个位置的干净 token（one-hot）；$e_M$：[M] 的 one-hot；$\mathrm{Cat}$：one-hot 凸组合即分类分布。吸收态的含义：一旦变 [M] 便永远停留，$t = 1$ 时全序列抹除。全序列联合分布为逐位置乘积。

**Reverse：参数化与 carry-over。** 条件于干净 token 的后验，对 masked 位置是两次调度的凸组合；未掩码位置原样停留（carry-over）：

$$
q\left(x_s^{i} \mid x_t^{i} = M,\ x_0^{i}\right) = \mathrm{Cat}\Big(\frac{\alpha_s - \alpha_t}{1-\alpha_t}\, x_0^{i} + \frac{1-\alpha_s}{1-\alpha_t}\, e_M\Big), \qquad s < t \qquad (2)
$$

**符号**：首项系数为「揭开」概率，第二项为「保持掩码」概率；$s = 0$ 时揭开概率为 1，$s \to t$ 时退化为保持掩码。reverse 的参数化是一台双向 Transformer $f_\theta(x_t)$：对全部 [M] 位置联合预测干净 token 分布 $\tilde{p}_\theta(x_0^{i} \mid x_t)$，代入式 (2) 替换未知的 $x_0^{i}$。注意与 3.1 的分工：这里用双向表征做**生成**，靠的是迭代细化而非因果分解。

**ELBO → 逐步 CE。** 离散扩散的 ELBO 逐步写出后，每一步的贡献恰好落在该步揭开位置的 CE 上，权重由调度给出：线性调度 $\alpha_t = 1 - t$ 下整串变分塌缩为

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1)}\; \frac{1}{t} \sum_{i:\ x_t^{i} = M} \mathrm{CE}\big(x_0^{i},\ \tilde{p}_\theta(\cdot \mid x_t)\big) \qquad (3)
$$

**符号**：$t$：掩码率（线性调度下 $1 - \alpha_t = t$）；求和只覆盖被掩位置；$1/t$ 为调度权重（推导为练习 1）。训练循环就三行：均匀采样掩码率 $t$、按 $t$ 独立掩码、只对被掩位置算 CE 并除以 $t$。与 3.1 的 MLM 对照：BERT 把掩码率固定在 15%，式 (3) 把掩码率本身变成被积变量——diffusion LLM 因此与 encoder 家族共享大部分训练基础设施。SEDD（[Lou et al., 2023](../references.md)，arXiv:2310.16834，[A]）走更一般的连续时间马尔可夫链参数化：网络直接学具体分数（concrete score）$s_\theta(x)_y \approx q_t(y)/q_t(x)$（扰动后边际的比值），用分数熵（score entropy，Bregman 散度族的 proper scoring rule，极小点唯一落在真实比值）训练，ELBO 由逐步比值的积分给出；吸收态只是其 rate matrix 的一个特例。

```
前向（式 (1)），t: 0 → 1，逐位置独立腐蚀：
t=0.0    天  气  很  好  我  们  出  去  玩
t=0.3    天 [M]  很  好 [M]  们  出 [M]  玩
t=0.7   [M] [M]  很 [M] [M]  们 [M] [M]  玩
t=1.0   [M] [M] [M] [M] [M] [M] [M] [M] [M]
反向（式 (2)(3)）：f_θ 对全部 [M] 联合预测 → 按调度提交一个子集 → t 减小
同一步行内提交的位置按条件独立处理 ⟺ 对精确联合后验的近似：
并行度本身就是质量-成本旋钮
```

**与 AR 的公式级对账**（三笔）：

| 项 | AR（1.1 / 2.4） | masked diffusion |
| --- | --- | --- |
| 似然 | $p(x) = \prod_i p(x_i \mid x_{<i})$，NLL 精确可算 | 式 (3) 是 ELBO 上界，perplexity 只能报 bound |
| decode | $T$ 步串行前向 + KV cache 逐步复用（2.4 账） | $N$ 步（$N \le T$ 可调）全序列双向前向，**无 KV cache**：步与步之间全部隐状态被双向耦合改写，上一步的 K/V 对下一步无效，每步成本 ≈ 一次全长 prefill |
| 并行度 | 每前向产 1 token（4.5 的 MTP / speculative 放宽到不足 2） | 每前向可提交任意多位置；提交子集越小越接近精确联合，$N = T$ 时退化为逐位置采样 |

质量差距的证据现状：LLaDA 论文报告 8B 量级与 LLaMA-3-8B 级 AR 基线在标准基准上互有胜负、整体接近，并在 reversal 类任务上给出明显优势（论文 claim）；SEDD 在 GPT-2 量级对照中报告吸收态变体的 zero-shot perplexity 优于同规模 GPT-2（论文 claim）。边界同样明确：10B+ 量级、与前沿 AR 同数据预算的扩散对照尚无公开证据；$N$ 步全序列前向、无 cache 复用的 serving 账（练习 2 给出 attention 项的 $2N$ 倍账）未在工业负载下验证。**成熟度：研究原型 → 早期工业验证（LLaDA 系）**。

## 2. 深度稀疏：MoD 与 early-exit

MoD（混合深度，mixture-of-depths；[Raposo et al., 2024](../references.md)，arXiv:2404.02258，[B]）把 4.5 的路由思想从「选哪个 FFN」搬到「这个 token 要不要过这个 block」：每层一台标量 router $r_i^{l} = w^{\top} x_i^{l}$，top-$k$ 选中者过 attention + FFN，其余走残差直通：

$$
x_i^{l+1} = \begin{cases} x_i^{l} + f^{l}\big(\tilde{x}^{l}\big)_i, & i \in \mathrm{TopK}\big(r^{l}, k\big) \\ x_i^{l}, & \text{otherwise} \end{cases} \qquad (4)
$$

**符号**：$f^{l}$：第 $l$ 个 block（attention + FFN）；$k$：静态容量（论文用序列长的固定分数，如 1/8）；$\tilde{x}^{l}$：选中 token 的子序列。容量写死 ⇒ 负载均衡由构造保证，无 aux loss——与 4.5 的 expert choice 同族（选择方在 block 侧而非 token 侧）。论文报告 iso-FLOPs 训练曲线下移，部分配置下同质量每前向 FLOPs 省至一半量级（论文 claim）。代价两笔：router 的 top-$k$ 依赖整条序列，AR 推理需要辅助预测头来近似；选中 token 的 attention 只在选中集合内闭合，逐 token 的深度不再均匀。**成熟度：研究原型**（无官方代码）。early-exit 是同族更激进的版本：判定器不问「过不过这一层」而问「是否就此读出」，可视为式 (4) 的容量随 token 自适应的极限形态；两者共同的工程难点是逐 token 可变深度与 batch 规整执行的冲突——序列内每个位置的层数不同，kernel 与流水线的处理归 inference 与 cluster 维度。

## 3. Differential Attention：两张 softmax 图相减

differential attention（[Ye et al., 2024](../references.md)，arXiv:2410.05258，[A]；实现代码在 microsoft/unilm 的 Diff-Transformer 目录）瞄准的是 4.6 熵稀释与 4.2 attention sink 的共同病灶：softmax 归一化强迫注意力质量总和恒为 1，无关上下文上必然沉积共模质量。处方是把每个 head 的 Q、K 对半切分，各算一张 softmax 图再相减：

$$
\mathrm{DiffAttn}(X) = \Big(\mathrm{softmax}\big(\tfrac{Q_1 K_1^{\top}}{\sqrt{d_h}}\big) - \lambda\, \mathrm{softmax}\big(\tfrac{Q_2 K_2^{\top}}{\sqrt{d_h}}\big)\Big)\, V \qquad (5)
$$

**符号**：$Q_1, K_1$ 与 $Q_2, K_2$：逐头对半切分（各 $d_h / 2$ 维，两张半维图合计开销与一张全维图持平，等 FLOPs 等参数）；$\lambda$：逐头可学标量（论文用指数差参数化与分层初始化）；$V$：两半共享。机制的读法：第一张图学「信号 + 噪声」、第二张图专学共模噪声，相减消噪——与差分放大器的共模抑制同构。论文在 $10^{9}$ 参数量级对照报告 iso-算力下 LM loss 更低、长上下文检索与 ICL 稳定性改善（论文 claim）。与 4.2 / 4.6 的关系是结构级的：sink 与熵稀释在那里被诊断、在这里被造进了算子。**成熟度：研究原型**。

## 4. 状态即模型：TTT 与 looped-depth recurrence

TTT（测试时训练，test-time training；[Sun et al., 2024](../references.md)，arXiv:2407.04620，[B]）把 RNN 的隐状态直接换成一个小模型的权重，递推本身就是梯度下降：

$$
W_i = W_{i-1} - \eta\, \nabla_W\, \mathcal{L}\big(W_{i-1};\ x_i\big), \qquad z_i = W_i x_i \qquad (6)
$$

**符号**：$W_i$：第 $i$ 步的状态（一个小模型的全部权重）；$\eta$：内环学习率；$\mathcal{L}$：自监督重构损失（如 $\lVert W x_i - x_i \rVert^{2}$）；$z_i$：读出。TTT-Linear 的 $W$ 是矩阵，与 4.3 的 DeltaNet「对当前联想做一步在线 GD」严格同族；TTT-MLP 把状态换成两层 MLP，状态容量不再受 4.3 矩阵状态 rank 界的约束（该上界针对线性读出陈述）。训练侧用 mini-batch 内循环恢复并行度，复杂度与 linear attention 同阶；论文对照止于 1.3B 参数量级。**成熟度：研究原型**。

looped-depth recurrence 是相邻坐标上的另一个旋钮：同一 block 的权重沿深度循环调用 $r$ 次，参数不变、有效深度乘 $r$。它与 MoD 对偶——一个沿 token 维稀疏深度、一个沿深度维共享参数——也与 §5 的推理链同构：都是用串行步数换模型规模。本条无锚点论文，仅作设计空间坐标登记。**成熟度：研究原型**。

## 5. 负载形态反转：test-time scaling 与 agentic

前四节是「新算子」；本节是两个已经在 serving 账上兑现的需求侧变化，方向相同：单次请求的 token 生成量与驻留长度上移。

**测试时扩展（test-time scaling）的架构含义**。长推理链把单次请求的生成从 $10^{2}$–$10^{3}$ token 上移到 $10^{3}$–$10^{5}$：链全程驻留 context，4.6 的四堵墙从长尾需求变为默认约束——KV cache 随链长线性增长（2.4 账），直接抬高 MLA / DSA / KDA（4.1 / 4.2 / 4.3）的受益面；串行 decode 步数随链长放大，MTP 类摊薄（4.5）从加速器变成标配；1.3 的全生命周期账中推理侧占比上移，训练-推理算力配比被重写。工业佐证：DeepSeek-V3.2 把 top-2048 稀疏检索（4.2）带上 128K serving（arXiv:2512.02556，[A]），Kimi K3 以 KDA 主干加 1M 上下文上架（arXiv:2607.24653，[A]）——稀疏层与线性层的工业化正是推理链经济学的直接产物。**成熟度：工业现实**。

**Agentic 化的架构含义**。多轮工具调用在机制上等于：每轮把工具返回 append 进 context，会话单调变长且前缀高度共享。两个直接后果：(a) 前缀复用成为 serving 一级优化（prefix caching 的调度与命中归 inference 维度），架构侧对应的是长窗与有效上下文（4.6 的 claimed vs effective 口径）成为默认指标而非宣传数字；(b) 会话状态即 KV cache 的持久化、分叉与回滚——长会话的边际成本由固定状态层（4.4 的 hybrid 账）与稀疏检索层（4.2）决定，链式生成的串行延迟由 MTP（4.5）与 §1 的并行解码对冲。**成熟度：工业现实**。

## 6. 边界与去向

原生多模态（native multimodal：tokenizer-free 视觉输入、混合模态的 mask 结构、统一 token 空间）整体归 P8.5，本章只登记一个事实：Kimi K3 已把 native vision 接上 KDA 主干（4.4，arXiv:2607.24653，[A]）。至此 P4 的账闭合：2.x 的每个组件在 4.1–4.7 各有一个被账驱动的替代族，本章六个方向回答「下一笔账在哪里」。P5 转入案例研究，按固定模板逐模型精读 config 谱系。

## 总结

| 方向 | 机制一句话 | 关键公式 | 证据锚点 | 成熟度 |
| --- | --- | --- | --- | --- |
| diffusion LLM | 逐 token mask 腐蚀 + 联合去掩码；ELBO 塌缩为 $1/t$ 加权 CE | 式 (1)(2)(3) | LLaDA（8B）[A]、SEDD [A] | 研究原型 → 早期工业验证 |
| MoD / early-exit | 逐 token 决定过不过 block；静态容量 top-$k$ | 式 (4) | MoD [B] | 研究原型 |
| differential attention | 两张半维 softmax 图相减消共模噪声 | 式 (5) | arXiv:2410.05258 [A] | 研究原型 |
| TTT | 隐状态 = 小模型权重，递推即 test-time GD | 式 (6) | TTT [B] | 研究原型 |
| looped-depth recurrence | 同 block 沿深度循环，参数 ↔ 深度解耦 | —（无锚点） | 概念登记 | 研究原型 |
| test-time scaling | 链长 → context / cache / 串行延迟三重压力 | 复用 4.6 四堵墙 | DeepSeek-V3.2 [A]、Kimi K3 [A] | 工业现实 |
| agentic | 多轮工具调用 = 长上下文 + KV 状态管理 | 前缀共享（归 inference） | 同上 | 工业现实 |
| native multimodal | 统一 token 空间、混合模态 mask | 归 P8.5 | Kimi K3 [A] | 工业验证中 |

## 练习

1.（推导）从式 (1)(2) 出发，对线性调度 $\alpha_t = 1 - t$ 推导式 (3)：(a) 写出逐步 ELBO 中 $t \to s$ 一步的贡献，说明权重 $-\alpha_t' / (1 - \alpha_t)$ 的来源；(b) 说明未掩码位置为什么不贡献 loss（carry-over 项恒为零）；(c) 指出 $1/t$ 权重在 $t \to 0$ 时发散，为什么实践中无害（提示：$t$ 小时被掩位置数的期望也小）。
2.（计算）decode 的 attention FLOPs 对账。设 $L$ 层、head 总维 $d$、生成长度 $T$、diffusion 步数 $N$。(a) 证明 AR decode 全程的 scores + AV 合计约 $2LT^{2}d$（第 $t$ 步 $4td$，逐层）；(b) 证明 masked diffusion 每步（全序列双向、无 cache 复用）为 $4LT^{2}d$、全程 $4NLT^{2}d$，比值为 $2N$；(c) 取 $T = 8192$、$N = 64$ 算出倍数 128×，并定性说明这笔多付的 FLOPs 在什么 batch 与延迟预算下可能被并行提交抵消。
3.（代码）教学实现 masked diffusion 训练步（教学用，非优化实现）：给定 batch $x_0 \in \{1,\ldots,V\}^{B \times T}$ 与模型 $f_\theta$，(a) 采样 $t \sim U(\epsilon, 1)$ 并按式 (1) 逐位置独立掩码；(b) 前向得 logits $[B, T, V+1]$，按式 (3) 计算加权 CE（只在被掩位置、除 $t$）；(c) 数值断言：被掩位置数的 batch 期望为 $tT$，且掩码索引集合与 loss mask 一致。

## 参考

- [Nie et al., 2025](../references.md)：Large Language Diffusion Models（LLaDA），arXiv:2502.09992，[A]（key: llada）——8B 吸收态 diffusion LLM；式 (1)–(3)、AR 对照与质量证据出处
- [Lou et al., 2023](../references.md)：Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution（SEDD），arXiv:2310.16834，[A]（key: sedd）——concrete score 与 score entropy 参数化
- [Ye et al., 2024](../references.md)：Differential Transformer，arXiv:2410.05258，[A]（key: differential-transformer）——式 (5) 双 softmax 消噪；代码在 microsoft/unilm
- [Raposo et al., 2024](../references.md)：Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models，arXiv:2404.02258，[B]（key: mod）——式 (4) 逐 token 深度路由与 iso-FLOPs 证据
- [Sun et al., 2024](../references.md)：Learning to (Learn at Test Time): RNNs with Expressive Hidden States，arXiv:2407.04620，[B]（key: ttt）——式 (6) TTT 层
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-V3.2，arXiv:2512.02556，[A]（key: deepseek-v3.2）——DSA 上架 128K，test-time scaling 的工业佐证
- [Kimi Team, 2026](../references.md)：Kimi K3: Open Frontier Intelligence，arXiv:2607.24653，[A]（key: kimi-k3）——KDA + 1M ctx + native vision，agentic 与 P8.5 边界的佐证
