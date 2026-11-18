# 1.2 训练算子的反向推导（Backward Derivations of Training Operators）

> 本章把 1.1 的 vjp 框架落到 LLM 训练的每个算子上，逐个推出反向闭式并对照实现。推导主线是 matmul 反向（ $\partial L / \partial A = G B^{\top}$ 与 $\partial L / \partial B = A^{\top} G$ 的逐元素推导）、softmax 与 softmax-CE 融合反向（ $p - e_y$ 的全链化简）、RMSNorm / LayerNorm 反向、GELU / SiLU / SwiGLU 反向；随后是三个机制型算子：embedding 反向 = scatter-add、dropout 反向的 mask 复用与 RNG 语义、FlashAttention 反向的重算策略与 IO 账；最后是 fused / chunked cross-entropy 的显存账与 Liger 机制，以一张算子 vjp 总表（前向公式 / 反向公式 / save 什么 / FLOPs 比）汇总 | 前置：1.1（vjp 链式执行、save_for_backward 语义）、models 2.3（softmax Jacobian，其式 (8)）、models 2.7（LN Jacobian 引理）、models 4.1（FlashAttention 前向）| 锚点：FlashAttention 论文（[A]）、linkedin/Liger-Kernel（[A]）、PyTorch ATen `tools/autograd/derivatives.yaml`（教学参照）

## 1. 记号与总纲

全章记号先固定。算子输出 $y$ 的上游梯度记 $G := \partial L / \partial y$（与 models 1.2 §4 一致）；损失 $L$ 是标量。shape 沿用 models 2.1 的 $[B, T, H, D]$ 记法；本章频繁把 $[B, T, \ldots]$ 折成 $[M, \ldots]$，其中 $M = B \cdot T$ 是 token 行数——逐位置算子对每行独立作用，共享参数的梯度对行求和（§2 推论 1）。

每个算子的推导走同一条三步套路：把前向写成逐元素形式；选定上游 $G$；对单个输入变量收拢链式求和、认出矩阵形。读完前两个算子（§2、§3），这个套路会固化为固定流程，后面每个算子只是它的再实例化。

每条推导写完后，用同一份四件套自查：(a) 形状对账——梯度必与参数同形（§2 演示）；(b) 标量特例手算——取一维输入把通式退化到口算可验；(c) 极端输入存活检查—— $p_y \to 0$、 $x \to 0$ 这类边界代入公式看是否发散（§4 的 $1 / p_y$ 就是这么被揪出来的）；(d) gradcheck 数值证伪（§4、§10 的代码模板）。

读每个算子时只问四件事：**前向公式、反向 vjp、反向要 save 什么、反向/前向 FLOPs 比**。save 清单决定激活账（1.1 §6.2 的内存语义在此逐项兑现），FLOPs 比决定 1.1 §5「反向 ≈ 2× 前向」的成立范围。验证纪律一句话：本章每条闭式都能被 `torch.autograd.gradcheck` 数值证伪；教学代码在 §4 与 §10 给出模板，凡是与官方实现并列处，差异点写到文件与函数级。

## 2. matmul 反向：本章一切推导的母式

设 $Y = AB$， $A \in \mathbb{R}^{m \times k}$， $B \in \mathbb{R}^{k \times n}$， $Y_{ij} = \sum_{s=1}^{k} A_{is} B_{sj}$。给定 $G = \partial L / \partial Y \in \mathbb{R}^{m \times n}$，用逐元素链式法则推 $A$ 的梯度。 $A_{is}$ 只出现在 $Y$ 的第 $i$ 行里，故对 $L$ 的影响途经 $Y_{i1}, \ldots, Y_{in}$ 全部 $n$ 个元素：

$$
\frac{\partial L}{\partial A_{is}} = \sum_{j=1}^{n} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial A_{is}} = \sum_{j=1}^{n} G_{ij} \thinspace B_{sj} \qquad (1)
$$

**符号**： $s \in \lbrace 1, \ldots, k \rbrace$：收缩维下标；第二个等号用 $\partial Y_{ij} / \partial A_{is} = B_{sj}$（ $Y_{ij}$ 对 $A_{is}$ 是线性的）。式 (1) 右端正是矩阵乘积 $G B^{\top}$ 的 $(i, s)$ 元。对 $B_{sj}$ 同理（它出现在第 $j$ 列的全部 $m$ 个元素里），对 $i$ 求和给出 $A^{\top} G$ 的 $(s, j)$ 元。合起来：

$$
\frac{\partial L}{\partial A} = G B^{\top}, \qquad \frac{\partial L}{\partial B} = A^{\top} G \qquad (2)
$$

shape 走读与账：

```
前向:  A [m, k] @ B [k, n] = Y [m, n]              FLOPs = 2mnk
反向:  dA = G [m, n] @ Bᵀ [n, k]  ──► [m, k]       FLOPs = 2mnk
       dB = Aᵀ [k, m] @ G [m, n]  ──► [k, n]       FLOPs = 2mnk
       合计 4mnk = 2 × 前向
```

**FLOPs 比 = 2 的精确出处**就在这里：两个 vjp 各是一次同形 matmul，各 $2mnk$（models 1.2 式 (1) 的 $2mnk$ 口径），合计恰为前向两倍——1.1 §5 的一般论证在这里落地。代入一组真实尺度。Llama-3.1-8B 的 FFN down 投影（ $m$ = 4096 token、 $k$ = 14336、 $n$ = 4096 ）：前向 $2 \times 4096 \times 14336 \times 4096 \approx 4.8 \times 10^{11}$ FLOPs，反向两笔同形 matmul 合计约 $9.6 \times 10^{11}$。逐层逐算子都是这个 1:2。

每条反向推导写完，先做断步检查： $\partial L / \partial A$ 必须与 $A$ 同形—— $G B^{\top}$ 是 $[m, n] \times [n, k] = [m, k]$，对； $\partial L / \partial B$ 是 $[k, m] \times [m, n] = [k, n]$，对。形状对不上的反向公式必错；形状对得上的，再交给 gradcheck（§4）。

这里有两条直接推论。**推论 1（共享权重跨 token 求和）**：逐位置线性层 $y_t = x_t W$ 对 $M$ 个 token 共享 $W$。把 token 堆成 $X \in \mathbb{R}^{M \times k}$、上游梯度堆成 $G \in \mathbb{R}^{M \times n}$，则 $\partial L / \partial W = X^{\top} G = \sum_t x_t^{\top} G_t$——与 models 1.2 式 (3) 的 $\sum_b x_b^{\top} G_b$ 逐项一致。而输入梯度 $\partial L / \partial x_t = G_t W^{\top}$ 逐行独立。「权重梯度跨 token 求和、输入梯度逐 token 独立」是同一条式 (2) 的两半，也是梯度累积（1.5）与数据并行梯度平均（4.1）的合法性的算子级根据。**推论 2（batched matmul）**： $Y^{(b)} = A^{(b)} B^{(b)}$ 的式 (2) 逐 batch 成立——attention 的 $QK^{\top}$ 与 $PV$（§9）就靠它。

实现对照：ATen 的 `mm` 在 `tools/autograd/derivatives.yaml`（pytorch/pytorch repo，教学参照）里的条目正是这两个 vjp——`self: mm_mat1_backward(grad, mat2, ...)` 与 `mat2: mm_mat2_backward(grad, self, ...)`，即「grad 乘另一输入的转置」各一次，外加 stride / layout / 共轭的边界处理；带 batch 与广播的 `matmul` 走 `matmul_backward`，语义同式 (2)，只在广播维上多做一次归约。手写公式与生产实现的差别在工程边界，不在数学。硬件侧补一句：两个 vjp 与前向同为 GEMM，同一 kernel 家族、同量级的算术强度——「反向不慢于 2×」在带宽侧也没有隐藏代价。这正是 6ND 敢于把反向按 2× 计入的底层理由（kernel 层归 gpu 维度）。

5.1 前置：式 (2) 直接决定张量并行（TP）的通信位置。 $Y = XA$ 按列切 $A = [A_1 \mid A_2]$（输出分片到两卡）时，输入梯度 $\partial L / \partial X = G A^{\top} = \sum_p G_p A_p^{\top}$ 天然是跨分片的部分和——所以列切 ⇒ 反向一次 all-reduce。按行切（输入分片）时 $Y = \sum_p X_p A_p$ 前向即跨分片求和，反向 $\partial L / \partial X_p = G A_p^{\top}$ 无通信——所以行切 ⇒ 前向一次 all-reduce。Megatron 的 f / g 恒等算子（5.1）就是这两条规则的封装。

## 3. softmax 反向

$p = \mathrm{softmax}(z)$ 逐行作用（沿词表维 $V$ 或序列维 $T$）。其 Jacobian $J = \mathrm{diag}(p) - p p^{\top}$ 已在 models 2.3 式 (8) 完整推出（含饱和分析），此处只登记 vjp 形式与成本。给定行上游梯度 $g = \partial L / \partial p$，行向量左乘 $J$：

$$
\frac{\partial L}{\partial z} = g \thinspace J = g \odot p - \langle g, p \rangle \thinspace p = p \odot \bigl( g - \langle p, g \rangle \bigr) \qquad (3)
$$

**符号**： $\langle \cdot, \cdot \rangle$：行内内积；第一步用 $J$ 对称（models 2.3 §3.3）；逐分量形即 $\partial L / \partial z_j = p_j ( g_j - \langle p, g \rangle )$。

成本结构值得单看：前向每行约 $3V$（exp、求和、除各一遍），反向一次内积、一次数乘、两次逐元素组合约 $4V$——常数倍同阶。若物化 $J$： $V$ = 128256 时单行 Jacobian 有 $V^{2} \approx 1.6 \times 10^{10}$ 个元素，bf16 下约 31 GiB——一个 token 的 softmax Jacobian 就超过一张消费级显卡的显存。vjp 形式把 Jacobian 的存在性整个消掉：它在推导里出现，在内存里从不出现。save 什么？式 (3) 只需 $p$ 与 $g$，故存输出 $p$（每行 $V$ 个元素）即可；重算派可以只存 $z$、反向时重算 $p$（1.4 的选择题）。

与训练动力学的一条连线：式 (3) 就是 models 2.3 §3.4「饱和 ⇒ 梯度消失」结论的 vjp 形态——饱和行（ $p$ 逼近 one-hot）的梯度被 $p_j$ 本身压死。训练侧的含义是 attention 与输出头的 logit 尺度都必须受控；QK-norm 的动机在 models 2.7，输出侧的 z-loss 归 3.1。

## 4. softmax-CE 融合反向： $p - e_y$【关键推导】

训练损失的逐 token 形式： $L_t = -\log p_{t, y_t}$，其中 $p_t = \mathrm{softmax}(z_t)$、 $z_t \in \mathbb{R}^{V}$ 是 logit、 $y_t$ 是目标 token id。把对数展开， $L_t$ 是 logit 的显式函数：

$$
L_t = -z_{t, y_t} + \log \sum_{v=1}^{V} \exp(z_{t, v}) \qquad (4)
$$

**符号**：第二项即 logsumexp（其数值稳定的 max 减法口径见 models 2.3 §4；分块实现归 §10）。对 $z_{t, j}$ 求导：第一项仅在 $j = y_t$ 时有贡献 $-1$，第二项按 log 与和的链式各求一次：

$$
\frac{\partial L_t}{\partial z_{t, j}} = -\delta_{j y_t} + \frac{\exp(z_{t, j})}{\sum_v \exp(z_{t, v})} = p_{t, j} - \delta_{j y_t} \qquad (5)
$$

**符号**： $\delta_{j y_t}$：Kronecker delta。向量形： $\partial L_t / \partial z_t = p_t - e_{y_t}$， $e_{y}$ 为第 $y$ 个 one-hot 向量。全链就一步：softmax 的概率减目标的 one-hot。

这条化简值得单列一节：它是「分步反向会数值爆炸」的经典案例，也是融合算子存在的数学理由。假如按计算图分两步走——先经 $\partial L / \partial p_i = -\delta_{i y} / p_y$（对数的导数），再乘 softmax Jacobian 式 (3)。代数上这是恒等的：代入 $g_i = -\delta_{iy} / p_y$，内积 $\langle p, g \rangle = -1$，于是 $p_j (g_j - \langle p, g \rangle) = p_j (-\delta_{jy} / p_y + 1) = p_j - \delta_{jy}$。但数值上完全是另一回事。中间量 $-1 / p_y$ 在 $p_y \to 0$ 时发散：bf16 的最大可表示值约 $3.4 \times 10^{38}$， $p_y$ 低于约 $3 \times 10^{-39}$ 时 $-1 / p_y$ 直接溢出为 $-\infty$。再往下的次正规区（最小正规数约 $1.2 \times 10^{-38}$、最小次正规数约 $9.2 \times 10^{-41}$）里， $p_y$ 本身的尾数逐位丢失，梯度精度同步劣化。而融合式 $p - e_y$ 的每个分量恒在 $[-1, 1]$ 里，奇点根本不存在。生产实现（`torch.nn.functional.cross_entropy`，pytorch/pytorch repo `torch/nn/functional.py`，教学参照）前向走 logsumexp 稳定形；eager 反向是 log_softmax 与 nll_loss 两个 ATen 算子反向的复合，其代数效果恒等于 $p - e_y$—— $1 / p_y$ 这个中间量在任意一条实现路径上都不出现。

式 (5) 还直接给出 CE 梯度的两个 regime。预测自信且正确（ $p_y \to 1$ ）时，整行梯度 $\to 0$——已学会的 token 不再产生信号。预测错误（ $p_y \to 0$ ）时，目标分量 $\approx -1$、其余分量 $\approx p_j$——梯度分量恒有界， $\lvert \partial L / \partial z_j \rvert \le 2$。CE 经这个算子永不爆炸；训练中观测到的 logit 失控另有出处（z-loss 的动机，3.1）。

归约语义是隐藏的口径坑：训练 loss 通常是逐 token 平均 $\bar{L} = (1/M) \sum_t L_t$（mean 归约），梯度因此带 $1/M$ 因子；sum 归约则没有。这个因子在梯度累积（1.5）与 fused CE 的分块实现（§10）里都必须显式处理，处理错一位就是 $M$ 倍的梯度尺度错误。

全链 shape 走读一遍汇总： $z$ 从 $[M, V]$ 进，softmax 保持 $[M, V]$，逐行取目标分量后得 $[M]$，mean 归约成标量；反向从标量的种子 1 出发，按式 (5) 直接落回 $[M, V]$ 的 $p - e_y$——中间概率 $p$ 在融合实现里从不单独成账（它与前向的 $z$ 复用同一份内存语义）。

教学代码（教学用，非优化实现）：

```python
import torch

class FusedSoftmaxCE(torch.autograd.Function):
    """教学用，非优化实现。softmax + 交叉熵的融合前向/反向（式 (4)(5)）。"""

    @staticmethod
    def forward(ctx, z, y):
        m = z.max(dim=-1, keepdim=True).values      # 行最大值（max 减法，models 2.3 §4）
        logZ = m.squeeze(-1) + torch.log(torch.exp(z - m).sum(dim=-1))
        loss = (logZ - z[torch.arange(z.size(0)), y]).mean()
        ctx.save_for_backward(z, y)                  # 存 logits；生产实现分块重算（§10）
        ctx.M = z.size(0)
        return loss

    @staticmethod
    def backward(ctx, g):
        z, y = ctx.saved_tensors
        p = torch.softmax(z, dim=-1)
        p[torch.arange(ctx.M), y] -= 1.0             # p − e_y，式 (5)
        return g * p / ctx.M, None                   # mean 归约的 1/M；y 无梯度

torch.manual_seed(0)
z = torch.randn(8, 17, dtype=torch.float64, requires_grad=True)
y = torch.randint(0, 17, (8,))
torch.autograd.gradcheck(lambda t: FusedSoftmaxCE.apply(t, y), (z,))   # 数值证伪
```

## 5. RMSNorm 与 LayerNorm 反向【关键推导】

### 5.1 RMSNorm 全链推导

前向（逐 token 行 $x \in \mathbb{R}^{D}$ ）： $r = \sqrt{(1/D) \sum_j x_j^{2} + \varepsilon}$， $y_i = \gamma_i \thinspace x_i / r$。给定上游 $g = \partial L / \partial y$，记 $u = g \odot \gamma$（把增益吸收进上游）。先求两个中间导数。 $\partial r / \partial x_i$ 由 $r^{2} = (1/D) \sum_j x_j^{2} + \varepsilon$ 两边微分得到： $2 r \thinspace \partial r = (2 x_i / D) \thinspace \partial x_i$，即 $\partial r / \partial x_i = x_i / (D r)$。于是

$$
\frac{\partial y_j}{\partial x_i} = \frac{\gamma_j}{r} \thinspace \delta_{ij} - \frac{\gamma_j \thinspace x_j}{r^{2}} \cdot \frac{\partial r}{\partial x_i} = \frac{\gamma_j}{r} \thinspace \delta_{ij} - \frac{\gamma_j \thinspace x_j \thinspace x_i}{D \thinspace r^{3}} \qquad (6)
$$

**符号**： $\delta_{ij}$：Kronecker delta；第一项是分子的直接依赖，第二项是分母 $r$ 的全局耦合——RMSNorm 的每对输入输出都经 $r$ 相连，这正是其 Jacobian 非对角的原因。对 $j$ 求和收拢：

$$
\frac{\partial L}{\partial x_i} = \sum_j u_j \thinspace \frac{\partial (x_j / r)}{\partial x_i} = \frac{u_i}{r} - \frac{x_i}{D \thinspace r^{3}} \sum_j u_j \thinspace x_j \qquad (7)
$$

记 $\hat{y} = x / r$（归一化后、乘 $\gamma$ 前的中间量），把 $`x_i = r \thinspace \hat{y}_i`$ 与 $\sum_j u_j x_j = r \thinspace \langle u, \hat{y} \rangle$ 代入式 (7) 的第二项，即得向量形：

$$
\frac{\partial L}{\partial x} = \frac{1}{r} \left( u - \frac{\hat{y} \thinspace \langle u, \hat{y} \rangle}{D} \right), \qquad \frac{\partial L}{\partial \gamma} = \sum_{\mathrm{tokens}} g \odot \hat{y} \qquad (8)
$$

**符号**： $`\langle u, \hat{y} \rangle = \sum_j u_j \hat{y}_j`$：行内内积； $\gamma$ 的梯度按 §2 推论 1 跨全部 token 行求和。逐分量验证： $`(1/r) \thinspace (u_i - \hat{y}_i \thinspace \langle u, \hat{y} \rangle / D) = u_i / r - x_i \thinspace \langle u, x \rangle / (D r^{3})`$，与式 (7) 逐项一致。

结构读法：式 (8) 把上游梯度在归一化方向 $\hat{y}$ 上的投影（按均值 $\langle u, \hat{y} \rangle / D$ ）抠掉，再除 $r$。前向「除以 $r$ 使输出能量与 $\lVert x \rVert$ 解耦」；对偶地，反向把梯度与 $\hat{y}$ 方向解耦。两个边界注记。其一， $\varepsilon$ 不显含于式 (7)(8)——它只通过 $r$ 生效： $\lVert x \rVert \to 0$ 时 $r \to \sqrt{\varepsilon}$，梯度尺度被 $\varepsilon$ 兜底，这是它的唯一机制角色。其二，RMSNorm 是逐行算子：内积 $\langle u, \hat{y} \rangle$ 只在行内收缩，行间无耦合。反向里唯一的跨 token 耦合是 $\gamma$ 的梯度求和——那是参数共享使然（§2 推论 1），不是算子结构。成本方面：前向每行约 $4D$（平方和、除 $r$、乘 $\gamma$ 各一遍），反向约 $5D$（内积两遍、组合两遍、除 $r$ 一遍）——同为 $O(D)$ 的常数倍，淹没在 matmul 项里。save 什么？式 (8) 需要 $\hat{y}$ 与 $r$，实现常存 $x$ 与 $r$（inv_rms）、反向时重算 $\hat{y} = x / r$；逐 token 的 save 量是一个 $D$ 维向量加一个标量。

### 5.2 LayerNorm：多一项中心化

LayerNorm 前向多一步减均值： $\hat{x} = (x - \mu) / \sigma$， $y = \gamma \odot \hat{x} + \beta$，其中 $\mu = (1/D) \sum_i x_i$、 $\sigma^{2} = (1/D) \sum_i (x_i - \mu)^{2}$。其 Jacobian 引理 $J = (I - \mathbf{1} \mathbf{1}^{\top} / D - \hat{x} \hat{x}^{\top} / D) / \sigma$ 已在 models 2.7 推出（纯文字指引，不重复推导），vjp 形式为：

$$
\frac{\partial L}{\partial x} = \frac{1}{\sigma} \left( u - \bar{u} \thinspace \mathbf{1} - \frac{\hat{x} \thinspace \langle u, \hat{x} \rangle}{D} \right), \qquad u = g \odot \gamma, \quad \bar{u} = \frac{1}{D} \sum_i u_i \qquad (9)
$$

**符号**： $\mathbf{1}$：全 1 向量； $\bar{u}$： $u$ 的均值。与式 (8) 逐项对照：RMSNorm 少的正是中心化项 $\bar{u} \thinspace \mathbf{1}$（其 Jacobian 少 $\mathbf{1} \mathbf{1}^{\top} / D$ 这一项）——不减均值，反向就不用抠均值方向的分量。 $\beta$ 的梯度是 $\sum_{\mathrm{tokens}} g$， $\gamma$ 的梯度同式 (8)、把 $\hat{y}$ 换成 $\hat{x}$。save 清单的取舍也多一分：式 (9) 需要 $\hat{x}$ 与 $\sigma$，实现可以存 $x$、 $\mu$、 $\sigma$（反向重算 $\hat{x}$ ），也可以直接存 $\hat{x}$ 与 $\sigma$。前者多一份 $\mu$ 的每行标量，后者省一次重算——差一个每 token 标量的量级，归 1.4 的逐算子权衡。式 (9) 的完整推导留作练习 1。

## 6. GELU / SiLU / SwiGLU 反向

逐元素激活的反向是同一形态 $g \odot \varphi'(x)$，差别只在导数闭式。GELU 的 erf 参数化： $\varphi(x) = x \thinspace \Phi(x)$， $\Phi$ 为标准正态 CDF，乘积法则：

$$
\varphi'(x) = \Phi(x) + x \thinspace \phi(x), \qquad \phi(x) = \frac{1}{\sqrt{2 \pi}} \exp\left( -\frac{x^{2}}{2} \right) \qquad (10)
$$

**符号**： $\phi$：标准正态密度。实现有参数化分叉：`torch.nn.functional.gelu` 默认 erf 精确形，`approximate='tanh'` 是 tanh 逼近形（GPT-2 与 Megatron 的传统口径）。两形不是「同一函数的两种导数写法」，而是两个不同函数、各有其导数；函数值差在 $O(10^{-3})$ 量级，混用等于换算子。以哪个为准，看 config 的激活字段与实现代码（`torch/nn/functional.py`，教学参照）。

SiLU： $\mathrm{SiLU}(x) = x \thinspace \sigma(x)$， $\sigma$ 为 sigmoid。导数的形态值得先看一眼再推： $`\mathrm{SiLU}'(0) = 0.5`$， $x \to +\infty$ 时趋于 1， $x \to -\infty$ 时按 $\sigma(x)$ 趋于 0。换句话说，反向经 SiLU 就是把上游梯度乘上一个 0 到 1 之间、随 $x$ 单调的门——负半轴的信号被按比例压掉，但不像 ReLU 那样归零。乘积法则加 $\sigma' = \sigma (1 - \sigma)$：

$$
\mathrm{SiLU}'(x) = \sigma(x) + x \thinspace \sigma(x) \left( 1 - \sigma(x) \right) = \sigma(x) \left( 1 + x \thinspace (1 - \sigma(x)) \right) \qquad (11)
$$

SwiGLU FFN（Llama 口径：门投影 $W_3$、上投影 $W_1$、下投影 $W_2$； $x$ 为 $[M, d]$ ）： $a = x W_3$， $b = x W_1$， $h = \mathrm{SiLU}(a) \odot b$， $y = h W_2$。反向把式 (2) 与式 (11) 组链：给定 $G = \partial L / \partial y$，

$$
\frac{\partial L}{\partial h} = G W_2^{\top}; \qquad \frac{\partial L}{\partial a} = \mathrm{SiLU}'(a) \odot b \odot \frac{\partial L}{\partial h}; \qquad \frac{\partial L}{\partial b} = \mathrm{SiLU}(a) \odot \frac{\partial L}{\partial h} \qquad (12)
$$

**符号**： $\partial L / \partial a$、 $\partial L / \partial b \in \mathbb{R}^{M \times d_{\mathrm{ff}}}$。再接 matmul 反向：权重梯度 $\partial L / \partial W_3 = x^{\top} (\partial L / \partial a)$ 等三次外积，输入梯度 $\partial L / \partial x = (\partial L / \partial a) W_3^{\top} + (\partial L / \partial b) W_1^{\top}$（fan-out 求和，1.1 式 (2)）。FLOPs 账：前向 3 次 matmul 共 $6 M d \thinspace d_{\mathrm{ff}}$，反向 6 次同形 matmul 共 $12 M d \thinspace d_{\mathrm{ff}}$——严整的 2×；门的逐元素部分 $O(M d_{\mathrm{ff}})$，低一个 $d$ 因子。save 清单： $a$、 $b$（式 (12) 需要）、 $x$（权重梯度需要）、 $h$（ $W_2$ 的权重梯度需要）——每 token 约 $3 d_{\mathrm{ff}} + d$ 个元素。SwiGLU 因此是激活账的大头（models 1.2 账二的 $3 d_{\mathrm{ff}}$ 项即出于此），选择性重算（1.4）的第一刀通常砍在这里。

## 7. embedding 反向 = scatter-add

前向是查表： $`y_t = W[\mathrm{id}_t]`$， $W \in \mathbb{R}^{V \times D}$，无乘加。给定逐 token 上游 $g_t = \partial L / \partial y_t \in \mathbb{R}^{D}$，反向把每行梯度累加回词表对应行：

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{M} e_{\mathrm{id}_t} \thinspace g_t^{\top} \quad \Longleftrightarrow \quad \left( \frac{\partial L}{\partial W} \right)_{v, :} = \sum_{t:\hspace{5mu} \mathrm{id}_t = v} g_t \qquad (13)
$$

**符号**： $`e_{\mathrm{id}_t} \in \mathbb{R}^{V}`$：one-hot；右式即「第 $v$ 行收到全部取词 $v$ 的 token 的梯度之和」——scatter-add（`index_add` 语义）。

两个结构性后果。其一，梯度行稀疏：一个 step 触及的行数 $\le \min(M, V)$； $M$ = 32768、 $V$ = 128256 时即使词表全覆盖也只有约 26% 的行非零，实际 batch 的 distinct token 数更低（高频词在一个 batch 内复现，只合并成一行求和）。`nn.Embedding` 的 `sparse=True` 选项据此把权重梯度以稀疏张量返回（配稀疏感知优化器使用，`torch/nn/modules/sparse.py`，教学参照；优化器侧归 Part 2），dense 反向仍是训练默认。其二，账的形态：FLOPs 上前向 gather 与反向 scatter-add 都约等于零——embedding 是「纯搬运」算子，它的成本在内存带宽与参数字节数（models 1.2 的口径），不在 FLOPs。save 什么？只需整数下标列（每 token 一个 4–8 字节的 id），不必存输出——反向按式 (13) 只需要「哪个 token 查了哪一行」。

一个组合注记：embedding 与输出头共享权重（weight tying）时，同一矩阵的梯度是式 (13)（查表侧）与式 (2)（matmul 侧）两路之和——1.1 式 (2) 求和规则的又一个实例；Llama-3.1-8B 不共享（tie_word_embeddings = false，models 1.2 §9 的口径），两路各自成账。同形参数的两种算子命运也值得对照：embedding 矩阵 $[V, d]$ 查表、FLOPs 为零；输出头矩阵同形，却是每 token 全价 $2Vd$ 的 matmul（models 1.2 式 (2) 输出头项的出处）。参数字节账相同，FLOPs 账天壤之别——原因就是查表与乘加的算子差异。

## 8. dropout 反向：mask 复用与 RNG 语义

inverted dropout 的前向（训练态）： $y = m \odot x / (1 - q)$， $m_i \sim \mathrm{Bernoulli}(1 - q)$ 逐元素独立；eval 态是恒等（期望已被训练态的 $1 / (1 - q)$ 校准——也因此 eval 忘关 dropout 不是「少正则」而是直接换了前向函数）。给定上游 $g$：

$$
\frac{\partial L}{\partial x} = g \odot m / (1 - q) \qquad (14)
$$

关键在「同一张 $m$ 」：反向必须复用前向采样的 mask。重抽一张等于换了一个算子——梯度不再对应当次前向实际计算的函数；期望上仍无偏，但逐样本的正确性定义失效，梯度检查器的数值对照会直接失败。复用有两条实现路。(a) 存 mask：1 字节 / 元素（bool），是 bf16 tensor 的一半，save 清单上 dropout 因此只花半价。(b) 存 RNG 状态、反向重抽同一张 mask：PyTorch 的设备 RNG 是计数器型（CUDA 上为 Philox），保存「进入前向时的状态」再重放，即可逐位复现整段随机流。`torch.utils.checkpoint` 重算前向时靠的正是这个保存 / 恢复（`torch/utils/checkpoint.py`，教学参照；机制归 1.4），FlashAttention 反向对 attention 内 dropout 的处理同此（§9 的论文口径：不存 $O(N^{2})$ mask，存 RNG 状态重抽）。FLOPs 比 ≈ 1：反向与前向同为一次逐元素乘。

## 9. FlashAttention 反向的重算策略

前向数学（分块 tiling + online softmax 逐块归一化累加）已在 models 4.1 推出，本节只做训练侧三件事：反向算什么、存什么、账是多少。记号约定（仅本节）：沿用论文记号， $N$ = 序列长（即 shape 记法的 $T$ ）、 $d$ = head 维、 $M_{\mathrm{sram}}$ = SRAM 容量；与本章其余各节的 $M$（token 行数）不相混。

**反向算什么**。attention 的反向在数学上是五条闭式（§2 推论 2 与式 (3) 的直接组合；[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]，附录 B 同此）： $\mathrm{d}V = P^{\top} \thinspace \mathrm{d}O$、 $\mathrm{d}P = \mathrm{d}O \thinspace V^{\top}$、 $\mathrm{d}S_{ij} = P_{ij} ( \mathrm{d}P_{ij} - \sum_l P_{il} \thinspace \mathrm{d}P_{il} )$（式 (3) 的逐行形）、 $\mathrm{d}Q = \mathrm{d}S \thinspace K$、 $\mathrm{d}K = \mathrm{d}S^{\top} \thinspace Q$（ $1 / \sqrt{d}$ 缩放随同 $S$ 进出）。标准实现存下前向的 $P \in \mathbb{R}^{N \times N}$ 供反向查用—— $O(N^{2})$ 内存加 $O(N^{2})$ 的 HBM 读写。

**存什么**。FlashAttention 反向不存 $S$、 $P$：存 $Q, K, V, O$（各 $[N, d]$ ）与逐行的 softmax 归一化统计（行最大值 $m_i$ 与归一化和 $\ell_i$，或等价的 logsumexp，合计 $O(N)$ ），反向时按块两重循环（论文附录 B 的 Algorithm 4）从 $Q, K$ 重算 $S$ 与 $P$。一个化简让「存 $O$ 就够」严格成立：softmax 反向需要的行和量 $D_i := \sum_j P_{ij} \thinspace \mathrm{d}P_{ij}$ 可以不经 $P$ 算出——

$$
D_i = \sum_{j=1}^{N} P_{ij} \thinspace \mathrm{d}P_{ij} = \sum_{j=1}^{N} P_{ij} \thinspace \langle v_j, \mathrm{d}O_i \rangle = \left\langle \sum_{j=1}^{N} P_{ij} \thinspace v_j, \hspace{5mu} \mathrm{d}O_i \right\rangle = \langle O_i, \mathrm{d}O_i \rangle \qquad (15)
$$

**符号**： $\mathrm{d}P_{ij} = \langle v_j, \mathrm{d}O_i \rangle$（由 $O = PV$ 经式 (2) ）；第三步交换求和与内积；最后一步代回 $O$ 的定义。于是 $D$ 由保存的 $O$ 与上游 $\mathrm{d}O$ 直接给出， $P$ 只在重算它的块内短暂存在。数学一致性：重算的 $S_{ij} = \langle q_i, k_j \rangle / \sqrt{d}$ 与前向的是同一批数——**重算不是近似**，反向梯度与「存下 $P$ 的方案」在数学上逐位一致（浮点累加顺序可不同，容差口径同 models 2.3 §8 的注记）。与前向的结构对偶：前向按块扫 $K, V$、在线累积 $O$ 与行统计；反向按块两重循环，每块从 HBM 读入 $Q, K, V, O, \mathrm{d}O$ 的对应块与行统计，在 SRAM 里重算 $S, P$ 后，就地算出该块对 $\mathrm{d}Q, \mathrm{d}K, \mathrm{d}V$ 的贡献并累加——每个块只进 HBM 一次，IO 账的 $N^{2} d^{2} / M_{\mathrm{sram}}$ 就是这么来的。attention 内 dropout 按 §8 路 (b)：不存 $O(N^{2})$ 的 mask，存前向的 RNG 状态、反向重抽（论文附录 B 口径，只多 $O(N)$ 额外内存）。

**账**。IO 复杂度（论文 Theorem 2 的前向界与 Theorem 5 的反向界，同型）：标准 attention 需要 $\Theta(Nd + N^{2})$ 次 HBM 访问，FlashAttention 需要 $\Theta(N^{2} d^{2} / M_{\mathrm{sram}})$ 次；在 $d$ 取 64–128、 $M_{\mathrm{sram}}$ 约 100 KB 的典型区间， $d^{2} \ll M_{\mathrm{sram}}$，访问量少一个量级级，论文并给出「任何精确 attention 算法无法在全部 SRAM 尺度上渐近优于它」的下界。内存：标准 $O(N^{2})$（存 $S$、 $P$ 两张），FlashAttention 在输入输出本体之外只多 $O(N)$（归一化统计）。代入一组数看内存差： $N$ = 4096、bf16 时，单头单序列单层的 $P$ 是 $4096^{2} \times 2$ bytes = 32 MiB；32 头 32 层、 $S$ 与 $P$ 两张共存就是 64 GiB（与 models 2.3 §7 的 scores+probs 表同口径）。FA 存下的行统计 $\ell, m$ 是 $2 \times 4096 \times 4$ bytes = 32 KiB——同一位置，三个数量级的差。这正是 models 2.3 §7 那张 $O(T^{2})$ 内存表被消掉的方式。FLOPs：前向 2 次 matmul（ $QK^{\top}$、 $PV$ ），反向 5 次——重算 $QK^{\top}$ 一次，加上五条闭式里的 4 次（ $\mathrm{d}V$、 $\mathrm{d}P$、 $\mathrm{d}Q$、 $\mathrm{d}K$ 各一次）；反向/前向 = 2.5，比 §2 的 2 倍律多出的 0.5 就是重算的标价。论文的实测口径是「即使 FLOPs 更多，重算因 HBM 访问减少反而加速反向」——FLOPs 一分没省，省的是内存与访存，与 models 1.2 §6 对 FlashAttention 的口径一致。块调度与 SRAM 驻留的 kernel 级实现归 gpu 维度。

## 10. fused / chunked cross-entropy：不物化 $[B, T, V]$ 的 logits

**账**。输出头把 $[M, d]$ 隐状态投到 $[M, V]$ logits： $V$ = 128256、 $T$ = 4096、 $B$ = 1 时 bf16 logits = $128256 \times 4096 \times 2$ bytes ≈ 0.98 GiB——单序列、单步中间量。反向还要加两笔。一笔是 CE 的 save：以 torch 的默认组合 log_softmax + nll_loss 为例，反向要 save log_softmax 的输出 $[M, V]$，autocast 下它是 fp32，约 1.96 GiB。另一笔是 logits 的梯度 buffer，与 logits 同形——Liger 论文把「logits 与其梯度的双重物化」列为动机，其 CE kernel 以梯度就地覆写 logits 来避免双物化。即便按最省的两笔算，每序列约 2.9 GiB； $B$ = 8 时仅 logits 相关张量约 24 GiB——一张 H100 80GB 的近三分之一，只为算一个标量 loss。

**机制**。fused linear cross entropy（FLCE；[Hsu et al., 2024](../references.md)，arXiv:2410.10989 + linkedin/Liger-Kernel repo，[A]）把 $M = B \cdot T$ 个 token 分块（块长 $M_c$ ），逐块在 kernel 内完成「投影 → CE → 梯度」全链，logits 永不全量物化：

```
H [M, d]（反向要用的隐状态，本就要存）
 │  按行分块，块长 M_c
 ▼
H_c [M_c, d] ──片上投影──► Z_c = H_c Wᵀ [M_c, V]（临时 buffer，不落 HBM）
                            │  行内 logsumexp（V 维流式求 max 与 sum）
                            ▼
                    partial loss（标量，跨块累加）
                            │  反向：式 (5) 的 p − e_y 就地覆写 Z_c
              ┌─────────────┴─────────────┐
              ▼                           ▼
   dH_c = (p − e_y) W  [M_c, d]   dW += H_cᵀ (p − e_y)  [d, V]（跨块累加，§2 推论 1）
```

数学形式（论文式 (17) 还原到本章记号）： $z = W^{\top} h$， $\nabla_h L = W \thinspace \nabla_z L$， $\nabla_W L = \sum_h h \thinspace (\nabla_z L)^{\top}$——逐块求和正是 §2 推论 1 的权重求和规则。两点口径注意。其一，与 FlashAttention 的 online softmax 不同：CE 的归一化封闭在每行内部（行内 logsumexp），块间没有跨块重归一化，只有标量 loss 与梯度累加器的累加。其二，mean 归约的 $1 / M$ 必须落在块粒度上：Liger 按 $M_c / M$ 缩放块内梯度（论文 Remark 口径）——缩放错一位，梯度就差一个块数倍，与 §4 的归约坑同源。

分块把峰值显存从 $O(MV)$ 压到 $O(M_c V)$：块长 $M_c$ 的 bf16 logits 块是 $2 M_c V$ bytes， $M_c$ = 2048、 $V$ = 128256 时约 0.49 GiB，与块长无关的完整 logits（ $M$ = 32768 时 7.8 GiB）相比小一个数量级以上。块长按显存预算反解（早期版本实现注释的口径：令块 logits 与 $M \times d$ 同级，例 $M$ = 16384、 $V$ = 32000、 $d$ = 4096 ⇒ 块长 2048；当前 main 已改写为按显存预算常数求块长，思想相同）。块越小越省显存、越难喂饱 GPU——这个权衡是 kernel 调参面。

把这笔账放回全模型：同配置（ $B$ = 8、 $T$ = 4096）下 models 1.2 账二的层激活合计约 256 GiB，logits 链的约 24 GiB 占其中约 9%。比例不起眼，但它有三条特殊性质：每 step 必付、与层数 $L$ 无关、随 $V$ 线性。更关键的是它**不能被 gradient checkpointing 省掉**——logits 不是某层的中间值，而是损失的直接输入，重算它必须重算输出头 matmul（ $2MdV$ 不再廉价）。这正是 fused CE 不可被 1.4 的手段替代、必须单独立算的根据。

**实现层两个路径**（`src/liger_kernel/ops/fused_linear_cross_entropy.py`，函数 `LigerFusedLinearCrossEntropyFunction`，repo 已核实）。默认路径利用「CE 是末端算子、上游梯度恒为 1」，把上图的反向三步直接在 forward 里做完——官方 docstring 原文：「Since Cross Entropy Loss is the last layer, we can compute the gradient at the forward pass」——从而连 $H$ 都不必为反向保存，backward 只剩乘标量。`reduction='none'` 等需要逐 token 上游梯度的路径，则在 backward 里重入同一分块函数重算。两种形态是同一原则：**logits 永不全量物化，梯度在分块粒度上就地产生**。论文报告的平均收益：吞吐 +20%、显存 −60%（对照 HuggingFace 实现的论文摘要口径）。谱系上，重算换显存的思想源头是 gradient checkpointing（[Chen et al., 2016](../references.md)，arXiv:1604.06174，[B]）——§9 的 FA 反向与本节的 chunked CE 都是它在算子内部的内化；层粒度的版本归 1.4。

教学代码（教学用，非优化实现）——在 §4 的融合 CE 上加一层分块，反向逐块重算 logits：

```python
import torch

class ChunkedLinearCE(torch.autograd.Function):
    """教学用，非优化实现。不物化 [M, V] logits：分块前向 + 反向逐块重算。"""

    @staticmethod
    def forward(ctx, H, W, y, chunk):
        M = H.size(0)                        # W: [V, d]，输出头
        ctx.save_for_backward(H, W, y)       # 只存 H、W、目标；logits 不存
        ctx.chunk = chunk
        loss = H.new_zeros((), dtype=torch.float32)
        for s in range(0, M, chunk):
            zc = H[s:s + chunk] @ W.T        # 块内投影：logits 块 [chunk, V]
            mc = zc.max(dim=-1, keepdim=True).values
            logZ = mc.squeeze(-1) + torch.log(torch.exp(zc - mc).sum(dim=-1))
            loss = loss + (logZ - zc[torch.arange(zc.size(0)), y[s:s + chunk]]).sum()
        return loss / M                      # mean 归约

    @staticmethod
    def backward(ctx, g):
        H, W, y = ctx.saved_tensors
        M, dH, dW = H.size(0), torch.zeros_like(H), torch.zeros_like(W)
        for s in range(0, M, ctx.chunk):
            zc = H[s:s + chunk] @ W.t()      # 反向重算 logits 块
            p = torch.softmax(zc, dim=-1)
            p[torch.arange(p.size(0)), y[s:s + chunk]] -= 1.0   # p − e_y
            p *= g / M                       # mean 归约的 1/M
            dH[s:s + chunk] = p @ W
            dW += p.T @ H[s:s + chunk]       # 跨块累加（§2 推论 1）
        return dH, dW, None, None
```

`gradcheck`（fp64）与 `F.cross_entropy(H @ W.T, y)` 的数值对照留作练习 4。

## 11. 算子 vjp 总表

全章结果一页汇总。FLOPs 比 = 反向/前向（matmul 口径 $2mnk$；「常数倍」指与前向同为逐元素 $O(\cdot)$、常数在 2–6 之间）。

| 算子 | 前向 | 反向 vjp | save 什么 | 反向/前向 FLOPs |
| --- | --- | --- | --- | --- |
| matmul $Y = AB$ | $2mnk$ | $G B^{\top}$、 $A^{\top} G$（式 (2)） | $A$、 $B$ | 2 |
| softmax（逐行） | $O(V)$/行 | $p \odot (g - \langle p, g \rangle)$（式 (3)） | $p$（或 $z$ 重算） | 常数倍 |
| softmax-CE 融合 | $O(V)$/行 | $p - e_y$（式 (5)） | $z$；分块实现反向重算 $z$ | 常数倍 |
| RMSNorm | $\approx 4D$/行 | $(u - \hat{y} \langle u, \hat{y} \rangle / D) / r$（式 (8)） | $x$、 $r$ | 常数倍 |
| LayerNorm | $\approx 5D$/行 | 式 (9)（多中心化项） | $x$（或 $\hat{x}$ ）、 $\mu$、 $\sigma$ | 常数倍 |
| GELU / SiLU | $O(D)$ | $g \odot \varphi'(x)$（式 (10)(11)） | $x$ | 常数倍 |
| SwiGLU FFN | $6 M d \thinspace d_{\mathrm{ff}}$ | 式 (12) + 6 次同形 matmul | $a$、 $b$、 $x$、 $h$ | 2 |
| embedding | ≈0（查表） | scatter-add（式 (13)） | 整数下标 | ≈0（带宽账） |
| dropout | $O(n)$ | $g \odot m / (1 - q)$（式 (14)） | mask（1 B/元素）或 RNG 状态 | ≈1 |
| 残差加 | $O(n)$ | $g$ 原样回传两路（1.1 式 (2)） | 无 | ≈1 |
| attention（FlashAttention） | 2 次 matmul | 4 次反向 matmul + 1 次重算（式 (15) 与五条闭式） | $Q$、 $K$、 $V$、 $O$ + 行统计 $O(N)$ | 2.5 |
| LM 头 + CE（FLCE） | $2 M d V$ | 分块 $p - e_y$ + 两次投影 matmul | $H$（默认路径连 $H$ 都不另存） | ≈2（反向重算路径 ≈3） |

读法：「save 什么」列是激活账的逐项来源（1.4 的压缩对象）；「FLOPs 比」列说明 6ND 的「反向 = 2× 前向」对 matmul 主导部分精确成立，唯二例外是重算型算子（2.5 与 ≈3 两行）——它们用 FLOPs 换掉的正是「save 什么」列里最大的两项（ $T \times T$ 的 $P$ 与 $[M, V]$ 的 logits）。

用 save 列可以把 models 1.2 账二的逐层激活表重新推一遍（8B 口径、每 token）。RMSNorm1 存 $x$ 与 $r$ → $d$ 个元素；QKV 投影是 matmul，两个输入都要存 → 层输入 $x$ 一份 $d$（与上行的 $x$ 是同一份，去重后只计一次）加 QKV 输出本身供后续消费 $(H_q + 2 H_{kv}) \thinspace d_{\mathrm{head}}$；attention 本体在 FA 口径下存 $Q, K, V, O$ 与行统计（否则是 $H_q \cdot T / 2$ 个概率元素，即那张表里的 $16T$ 项）；out 投影存其输入 $d$；RMSNorm2 同第一个；SwiGLU 存 $a, b, x$ 加 $h$ → $3 d_{\mathrm{ff}} + d$。逐项加总（去重后）正是 models 1.2 的每层每 token $4d + (H_q + 2 H_{kv}) \thinspace d_{\mathrm{head}} + 3 d_{\mathrm{ff}}$ 定长项结构——激活账不是经验清单，是这张 vjp 总表 save 列的投影。

## 总结

| 推导 | 结果 | 用途 |
| --- | --- | --- |
| matmul 反向 | $\partial L / \partial A = G B^{\top}$， $\partial L / \partial B = A^{\top} G$（式 (1)(2)） | 6ND 反向 $4N$ 的精确依据；TP 通信位置的母式（5.1） |
| softmax vjp | $p \odot (g - \langle p, g \rangle)$（式 (3)） | 永不物化 $V \times V$ Jacobian（ $V$ = 128256 时单行约 31 GiB） |
| softmax-CE 融合 | $p - e_y$（式 (4)(5)） | 消除 $1 / p_y$ 数值奇点；融合算子存在的数学理由 |
| RMSNorm 反向 | 式 (6)–(8) | 抠 $\hat{y}$ 方向投影；LayerNorm 多中心化项（式 (9)，models 2.7 引理） |
| GELU / SiLU / SwiGLU | 式 (10)–(12) | 逐元素 vjp；SwiGLU 反向 = 6 次 matmul，save $3 d_{\mathrm{ff}} + d$ / token |
| embedding | scatter-add（式 (13)） | 行稀疏梯度；`sparse=True` 一句 |
| dropout | $g \odot m / (1 - q)$（式 (14)） | mask 复用；RNG 状态重抽（1.4 与 FA 反向共用） |
| FlashAttention 反向 | 式 (15) + 五条闭式 | 存 $O$ + 行统计、不存 $P$；IO $\Theta(N^{2} d^{2} / M_{\mathrm{sram}})$；反向 2.5× |
| fused / chunked CE | 分块 $p - e_y$ + 就地梯度 | logits 永不全量物化；Liger 平均吞吐 +20%、显存 −60%（[A]） |

## 练习

1.（推导）完成式 (9) 的 LayerNorm 反向：从 $\hat{x} = (x - \mu) / \sigma$ 出发，把 $\mu$ 与 $\sigma$ 对 $x$ 的依赖展开，验证其 vjp 比 RMSNorm 的式 (8) 恰好多一项 $-\bar{u} \thinspace \mathbf{1} / \sigma$；再说明这一项对应「全体输入同时加常数不改变输出」的不变性（对照 models 2.3 §3.3 对 softmax Jacobian 行和为 0 的讨论）。

2.（推导）用 §2 推论 2 证明 TP 的两条通信规则：列切线性层（ $A$ 按列分片到 $t$ 卡）的输入梯度是对各卡部分和的 all-reduce；行切线性层（ $A$ 按行分片、输入相应分片）的前向输出是 all-reduce 而反向无通信。由此写出每个线性层在两种切法下各自前向 + 反向的通信次数（5.1 的前置）。

3.（计算） $V$ = 128256、 $d$ = 4096：(a) $B \times T$ = 8×4096 时，朴素路径（bf16 logits 物化 + fp32 log_softmax save）的 logits 相关显存是多少 GiB？(b) FLCE 路径块长 2048、块 logits 以 fp32 临时物化时，峰值是多少？(c) 把 $T$ 拉到 32768（ $B$ 不变），两路径各是多少？由此说明为什么长序列训练里 fused CE 从可选变成必需。

4.（代码）跑通 §10 的 `ChunkedLinearCE`：用 `torch.autograd.gradcheck`（fp64）验证其反向；与 `F.cross_entropy(H @ W.T, y)` 对 loss 与 `H.grad`、`W.grad` 到 `atol=1e-10`；在 GPU 上以 $V$ = 128256、 $M$ = 32768 跑两版，用 `torch.cuda.max_memory_allocated` 记录峰值显存并对照练习 3 (a)(b) 的估算。

## 参考

- [Dao et al., 2022](../references.md)：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，arXiv:2205.14135，[A]（key: flashattention）——Theorem 2 的 IO 复杂度（ $\Theta(Nd + N^{2})$ vs $\Theta(N^{2} d^{2} / M_{\mathrm{sram}})$ ）、附录 B 的反向算法（存 $Q, K, V, O$ 与归一化统计、按块重算 $S$ / $P$、 $D_i = \langle O_i, \mathrm{d}O_i \rangle$、RNG 状态代 mask）
- [Hsu et al., 2024](../references.md)：Liger Kernel: Efficient Triton Kernels for LLM Training，arXiv:2410.10989，[A]（key: liger-kernel）——FLCE 机制（其式 (17)）与收益口径；repo `src/liger_kernel/ops/fused_linear_cross_entropy.py`
- [Chen et al., 2016](../references.md)：Training Deep Nets with Sublinear Memory Cost，arXiv:1604.06174，[B]（key: gradient-checkpointing）——重计算换显存的思想源头，层粒度版本归 1.4
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照（key: pytorch-repo）——`tools/autograd/derivatives.yaml`（`mm` / `matmul` 的 vjp 条目）、`torch/nn/functional.py`（cross_entropy 与 gelu 的两参数化）、`torch/nn/modules/sparse.py`（Embedding 的 sparse 选项）
