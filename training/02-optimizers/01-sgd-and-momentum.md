# 2.1 SGD 与 Momentum 完整推导（SGD and Momentum: The Complete Derivations）

> 本章回答一个问题：随机梯度下降为什么这样工作，能不能一步一步推出来。先从 mini-batch 梯度的统计性质入手——无偏性，以及噪声按 $\Sigma / B$ 缩放——推出最优步长与 batch 的关系，看清 linear scaling 的适用区。再证明 momentum 两种写法（速度形式与 EMA 形式）的等价，说清 Nesterov 的外推点差异。然后在二次碗上算出精确收敛速率：GD 每步收缩 $(\kappa-1)/(\kappa+1)$ ，heavy ball 收缩 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ 。最后把 momentum 与梯度噪声的相互作用算成一笔账 | 前置：1.5（等效 batch 的 tokens 口径与噪声缩放的前置引用）、models 1.1（loss 记号）、models 1.2（账的口径） | 锚点：`torch.optim.SGD`（pytorch/pytorch repo `torch/optim/sgd.py`，教学参照 + 官方实现）；[Polyak, 1964](../references.md)（无 arXiv，[B]）；[Sutskever et al., 2013](../references.md)（ICML，无 arXiv，[B]）；[McCandlish et al., 2018](../references.md)，arXiv:1812.06162，[B]；[Goyal et al., 2017](../references.md)，arXiv:1706.02677，[B]

## 1. 全批 GD 到 mini-batch SGD：无偏性与噪声缩放

训练目标是数据分布上的期望损失 $`\mathcal{L}(\theta) = \mathbb{E}_{x \sim \rho}\left[\ell_x(\theta)\right]`$ ，其梯度记 $g(\theta) = \nabla \mathcal{L}(\theta)$ 。全批梯度下降（gradient descent, GD）每步用真梯度：

$$
\theta_{t+1} = \theta_t - \eta \thinspace g(\theta_t) \qquad (1)
$$

**符号**： $\theta \in \mathbb{R}^{N}$ ：参数； $\eta$ ：步长（learning rate）； $t$ ：step 下标； $\mathcal{L}$ 取期望口径，有限数据集时替换为经验平均，本章结论不变。

真梯度算不起：一次 $g$ 要扫完全部数据。mini-batch SGD 用 $B$ 个 iid 样本的均值代替：

$$
\hat g_B(\theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell_{x_i}(\theta), \qquad x_1, \ldots, x_B \stackrel{\mathrm{iid}}{\sim} \rho \qquad (2)
$$

这个替代有两条统计性质，后面所有的账都由它们决定（[McCandlish et al., 2018](../references.md)，arXiv:1812.06162，[B]，§2 式 (2.2) 同式）。**无偏性**：期望的线性性逐项给出 $\mathbb{E}[\nabla \ell_{x_i}] = g$ ，故 $\mathbb{E}[\hat g_B] = g$ ：mini-batch 估计的系统误差为零，全部误差在方差里。**协方差按 $1/B$ 缩放**：

$$
\mathrm{Cov}(\hat g_B) = \frac{1}{B} \Sigma(\theta), \qquad \Sigma(\theta) \equiv \mathbb{E}_{x \sim \rho}\left[(\nabla \ell_x - g)(\nabla \ell_x - g)^{\top}\right] \qquad (3)
$$

**符号**： $\Sigma(\theta) \in \mathbb{R}^{N \times N}$ ：单样本梯度的协方差矩阵。推导一行： $B$ 个 iid 零均值项求平均，交叉项期望为零，方差除以 $B$ 。逐坐标看：每个参数的梯度噪声方差 ÷ $B$ 、标准差 ÷ $\sqrt B$ ——「batch 扩四倍、噪声减半」这个说法的精确形式就是式 (3)。

信号与噪声在同一个二阶矩里相加：

$$
\mathbb{E}\left\lVert \hat g_B \right\rVert^{2} = \lVert g \rVert^{2} + \frac{1}{B} \mathrm{tr}\thinspace \Sigma \qquad (4)
$$

把 $\mathbb{E}\lVert \hat g - g \rVert^{2} = \mathrm{tr}\thinspace \mathrm{Cov}(\hat g)$ 展开即得。两项的比值定义梯度噪声尺度（gradient noise scale），本章取简化形式 $`\mathcal{B}_{\mathrm{simple}} = \mathrm{tr}\thinspace \Sigma / \lVert g \rVert^{2}`$ 。McCandlish 原文另定义带 Hessian 的 $`\mathcal{B}_{\mathrm{noise}} = \mathrm{tr}(H\Sigma) / (g^{\top} H g)`$ ，两者差一个小常数倍（其式 (2.8)/(2.9)）；本章的 L-光滑口径对应的正是 simple 版，完整区分归 2.3。 $`B \ll \mathcal{B}_{\mathrm{simple}}`$ 时噪声项主导，估计的模长主要成分是噪声； $`B \gg \mathcal{B}_{\mathrm{simple}}`$ 时 $\mathbb{E}\lVert \hat g \rVert^{2} \approx \lVert g \rVert^{2}$ ，再加 batch 几乎不改变估计。McCandlish et al. 的实证结论：这个纯统计量在数量级精度上预测「最大有用 batch」（critical batch size），且随训练漂移——loss 下降、 $\lVert g \rVert$ 缩小，噪声尺度上升。噪声尺度的估计器、critical batch 的定义与测量归 2.3；本章只用式 (3)(4) 两个恒等式。

## 2. 步长与噪声的联合账：最优步长的两个 regime

式 (3)(4) 还只是静态性质；要把它们接到训练进程上，需要损失的几何。取局部二次模型（§5 的完整分析对象）： $`\mathcal{L}(\theta) = \frac{1}{2} (\theta - \theta^{\ast})^{\top} H (\theta - \theta^{\ast})`$ ， $H \succ 0$ 为 Hessian，最大特征值 $L$ 即光滑常数。对精确二次做一步 $\theta \leftarrow \theta - \eta \hat g$ ：

$$
\mathcal{L}(\theta - \eta \hat g) = \mathcal{L}(\theta) - \eta \langle g, \hat g \rangle + \frac{\eta^{2}}{2} \hat g^{\top} H \hat g \le \mathcal{L}(\theta) - \eta \langle g, \hat g \rangle + \frac{\eta^{2} L}{2} \lVert \hat g \rVert^{2} \qquad (5)
$$

取期望： $\mathbb{E}\langle g, \hat g \rangle = \lVert g \rVert^{2}$ （无偏），代入式 (4)：

$$
\mathbb{E}\left[\mathcal{L}(\theta - \eta \hat g)\right] \le \mathcal{L}(\theta) - \eta \lVert g \rVert^{2} + \frac{\eta^{2} L}{2} \left(\lVert g \rVert^{2} + \frac{\mathrm{tr}\thinspace \Sigma}{B}\right) \qquad (6)
$$

右端关于 $\eta$ 是开口向上的二次式，顶点即最优步长：

$$
\eta^{\ast}(B) = \frac{\lVert g \rVert^{2}}{L \left(\lVert g \rVert^{2} + \mathrm{tr}\thinspace \Sigma / B\right)} = \frac{1}{L} \cdot \frac{1}{1 + \mathcal{B}_{\mathrm{simple}} / B} \qquad (7)
$$

**符号**： $`\eta^{\ast}(B)`$ ：该 batch 下最大化单步期望进展的步长。式 (7) 把整个 batch-步长关系压缩成一条曲线，两个 regime 直接读出：

- **小 batch 区**（ $`B \ll \mathcal{B}_{\mathrm{simple}}`$ ）： $`\eta^{\ast} \approx \dfrac{B}{L \thinspace \mathcal{B}_{\mathrm{simple}}}`$ ——最优步长**正比于** batch。[Goyal et al., 2017](../references.md)（arXiv:1706.02677，[B]）的 linear scaling rule「batch 乘 $k$ 、步长乘 $k$ 、其余超参不动」，正是这个 regime 的经验形式；该文用它在 ImageNet 上把 batch 推到 8192 且精度不掉。式 (7) 给出这条经验的适用区与比例系数：它是噪声主导区的近似，不是普适规律。
- **大 batch 区**（ $`B \gg \mathcal{B}_{\mathrm{simple}}`$ ）： $`\eta^{\ast} \to 1/L`$ 饱和。代回式 (6)，最大单步期望进展 $\to \lVert g \rVert^{2} / (2L)$ 同样饱和：batch 再大也换不来更多每步进展，拐点量级就是 $`\mathcal{B}_{\mathrm{simple}}`$ 。这是 critical batch size 的一阶定义；精确口径（对训练全程积分、随训练漂移）归 2.3。

这条曲线同时解释了为什么「batch 大一倍、step 数减半」只在噪声区近似成立：饱和区里加 batch 几乎不减少所需步数，只是白烧数据与算力。

## 3. Momentum：速度形式与 EMA 形式的等价

病态曲率下 GD 的困难有精确的定量刻画（§5：收缩因子被条件数 $\kappa$ 卡住）。momentum（动量）的对策是对梯度历史做指数加权。文献与代码里流行两种写法。

速度形式（PyTorch 与 Sutskever 的约定）：

$$
v_{t+1} = \beta \thinspace v_t + \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta \thinspace v_{t+1}, \qquad v_0 = 0 \qquad (8)
$$

EMA 形式（指数移动平均，exponential moving average）：

$$
m_{t+1} = \beta \thinspace m_t + (1-\beta) \thinspace \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta' \thinspace m_{t+1}, \qquad m_0 = 0 \qquad (9)
$$

**符号**： $v_t$ ：速度（velocity，未归一的梯度加权和）； $m_t$ ：梯度的 EMA（权重和为 1 量级）； $\beta \in [0, 1)$ ：动量系数； $\hat g_t$ ：第 $t$ 步的 mini-batch 梯度。

**等价证明（变量代换）**。令 $m_t = (1-\beta) \thinspace v_t$ 。式 (8) 第一式两边乘 $(1-\beta)$ ，得 $(1-\beta) v_{t+1} = \beta (1-\beta) v_t + (1-\beta) \hat g_t$ ，即 $m_{t+1} = \beta m_t + (1-\beta) \hat g_t$ ——正是式 (9) 第一式。初值同为 0，归纳成立。更新侧：式 (8) 第二式 $= \theta_t - \dfrac{\eta}{1-\beta} m_{t+1}$ ，与式 (9) 第二式恒等当且仅当

$$
\eta' = \frac{\eta}{1-\beta} \qquad (10)
$$

故两形式是同一算法的两种参数化，换算只隔一个因子 $1-\beta$ 。读文献或改代码时先问一句： $\eta$ 乘在归一化之前还是之后？混用两形式的超参，复现就悄悄偏了。

把式 (8) 展开看结构：

$$
v_{t+1} = \sum_{s=1}^{t+1} \beta^{t+1-s} \hat g_s, \qquad \sum_{s=1}^{t+1} \beta^{t+1-s} = \frac{1 - \beta^{t+1}}{1-\beta} \to \frac{1}{1-\beta} \qquad (11)
$$

三个直接读数：

- **有效步长放大**：历史梯度同号的方向上，更新幅度 ≈ $\eta \cdot \frac{1}{1-\beta} \cdot \lVert \hat g \rVert$ ——momentum 把有效步长放大 $1/(1-\beta)$ 倍（ $\beta = 0.9$ 为 10 倍）。换 $\beta$ 必须联动 $\eta$ ，否则步长偷偷变了一个数量级。
- **平均视野**：最近 $k$ 步的权重占比 $1 - \beta^{k}$ ；覆盖 90% 权重需 $k \ge \ln 0.1 / \ln \beta$ ： $\beta$ = 0.9 / 0.95 / 0.99 对应约 22 / 45 / 229 步。 $\beta$ 的含义就是记忆长度。
- **符号交替分量自相抵消**：噪声与锯齿分量方向来回变，加权和被压——§6 给出精确方差账。

```
ĝ_t ──► v ← β·v + ĝ （速度累加器）──► θ ← θ − η·v          式 (8)
              ║  m = (1−β)·v，η′ = η/(1−β)（式 (10)）
              ▼
ĝ_t ──► m ← β·m + (1−β)·ĝ （EMA 平滑器）──► θ ← θ − η′·m   式 (9)
```

EMA 形式还有一个开局细节： $m_0 = 0$ 意味着前若干步的 $m_t$ 系统性偏小—— $t$ 步后权重和只有 $1 - \beta^{t}$ ，前期估计带着 $1 - \beta^{t}$ 的衰减偏差。SGD-momentum 实践通常不处理（PyTorch 首步直接以 $\hat g_1$ 初始化 buffer，见下）；Adam 把这个偏差显式除掉（bias correction，2.2 推导）。两处对照着看，就知道后者在修正什么。

**第三种约定：步长在速度内部**。[Sutskever et al., 2013](../references.md)（ICML 2013，PMLR 28(3):1139-1147，无 arXiv，[B]）原文的 classical momentum（CM）：

$$
v_{t+1} = \mu \thinspace v_t - \varepsilon \thinspace \hat g_t, \qquad \theta_{t+1} = \theta_t + v_{t+1} \qquad (12)
$$

代换 $u_t = -v_t / \varepsilon$ 立即化为式 (8)（ $\eta = \varepsilon$ ）：定常步长下，两约定等价。**但步长随时间变化时不再等价**。式 (12) 的历史项里固化的是旧步长；切换到新步长，需把历史乘 $\eta_{t+1} / \eta_t$ 。[Goyal et al., 2017](../references.md)（[B]，§3 式 (9)–(10)）把这个修正称为 momentum correction，并指出 warmup 期步长快速上升时缺了它会不稳定（历史项相对新梯度过小）。PyTorch 的式 (8) 约定把步长放在速度外，天然免修正——这是 warmup 普及后更稳的写法。

**PyTorch 实现核对**（pytorch/pytorch repo `torch/optim/sgd.py`，教学参照 + 官方实现；`_single_tensor_sgd` 逐行）。核心两行对应式 (8)：`buf.mul_(momentum).add_(grad, alpha=1-dampening)` 是第一式，`param.add_(grad, alpha=-lr)` 是第二式。还有三个附带事实。第一，dampening $\tau$ 把新梯度乘 $(1-\tau)$ ，等价于有效步长乘 $(1-\tau)$ ，默认 0。第二，首步 `buf` 直接克隆首个梯度而非从零加，恰对应 $v_0 = 0$ 下 $v_1 = \hat g_1$ 。第三，`weight_decay` 参数把 $\lambda \theta$ **加进梯度**再进 buffer——L2 与 momentum 耦合进同一 EMA；2.2 会证明这在自适应优化器下不等价于 weight decay，正是 AdamW 的动机。`foreach` / `fused` 分支是同一数学的算子融合变体，逐行语义相同。momentum buffer 本身是一份与参数同形的常驻内存——优化器状态内存账从这里开始，2.2 会再加两份。

## 4. Nesterov：先外推，再求梯

式 (8) 在 $\theta_t$ 处求梯度；Nesterov 的版本在「动量外推点」处求（NAG，[Sutskever et al., 2013](../references.md) 其式 (3)–(4)，[B]）：

$$
v_{t+1} = \mu \thinspace v_t - \varepsilon \thinspace \nabla \mathcal{L}(\theta_t + \mu \thinspace v_t), \qquad \theta_{t+1} = \theta_t + v_{t+1} \qquad (13)
$$

与 CM 的唯一差别是梯度的自变量： $\theta_t + \mu v_t$ 。先走到动量即将带你去的位置，再在那里看梯度。机制含义分两种情形。若动量方向继续对，外推点梯度与当前点渐近一致，两者无差。若即将冲过头——高曲率方向、碗壁反弹前夜——外推点的梯度已经反向，更新被提前刹车。Nesterov 比 CM 多一次「向前看」的修正；在 $\kappa$ 大的问题上，这种修正正对着 §5 的病根。

这个差别可以写成精确代数。令 $\tilde\theta_t = \theta_t + \mu v_t$ （外推点），则 $\theta_t = \tilde\theta_t - \mu v_t$ ，代入式 (13) 第二式：

$$
\tilde\theta_{t+1} = \theta_{t+1} + \mu \thinspace v_{t+1} = \tilde\theta_t - \mu \thinspace v_t + (1+\mu) \thinspace v_{t+1}
$$

再代入 $v_{t+1} = \mu v_t - \varepsilon \nabla \mathcal{L}(\tilde\theta_t)$ （注意梯度自变量恰是 $\tilde\theta_t$ ）：

$$
\tilde\theta_{t+1} = \tilde\theta_t + \mu^{2} v_t - (1+\mu) \thinspace \varepsilon \thinspace \nabla \mathcal{L}(\tilde\theta_t) \qquad (14)
$$

对照 CM 在同一点的更新 $\tilde\theta_{t+1} = \tilde\theta_t + \mu v_t - \varepsilon \nabla\mathcal{L}(\tilde\theta_t)$ ：NAG 把动量项系数从 $\mu$ 压到 $\mu^{2}$ ，把当前梯度项系数从 1 抬到 $1+\mu$ 。弱化了历史，强化了即时修正。PyTorch `nesterov=True` 实现的就是这个重排：先 `buf.mul_(momentum).add_(grad)` ，再令有效梯度 $= \hat g_t + \beta \cdot \mathrm{buf}$ ，展开为 $(1+\beta) \thinspace \hat g_t + \beta^{2} \thinspace v_{t-1}$ ，与式 (14) 逐项对应（PyTorch 参数序列对应式 (13) 的外推点序列，速度的符号约定差一个负号）。该文件 docstring 明言实现与 Sutskever 约定的这层关系。

实践上，NAG 与 CM 的差距是二阶小量，多数报告里两者互换不改变结论。Nesterov 名字的分量在理论侧：凸优化中把速率推进到 $\sqrt\kappa$ 量级的加速法族。§5 的 heavy ball 在二次碗上已达同阶；Nesterov 加速把同类速率推广到一般强凸函数。本书不展开该族。

## 5. 二次碗上的收敛分析： $\kappa$ 对 $\sqrt\kappa$

局部最小值附近，任何二阶可导损失都近似二次碗：

$$
\mathcal{L}(\theta) = \frac{1}{2} \theta^{\top} H \theta, \qquad H \succ 0, \qquad 0 \lt \mu = \lambda_1 \le \cdots \le \lambda_n = L, \qquad \kappa = \frac{L}{\mu} \qquad (15)
$$

**符号**：最优点平移到原点； $\mu, L$ ： $H$ 的最小、最大特征值； $\kappa$ ：条件数（condition number），碗的「扁」的程度。

**GD 的速率**。 $\theta_{t+1} = (I - \eta H) \thinspace \theta_t$ 。按 $H$ 的特征基分解，第 $i$ 个坐标独立收缩： $e_{t+1}^{(i)} = (1 - \eta \lambda_i) \thinspace e_t^{(i)}$ 。最坏方向决定整体速率，即 $\rho_{\mathrm{GD}}(\eta) = \max_i |1 - \eta \lambda_i| = \max(|1 - \eta\mu|, |1 - \eta L|)$ —— $|1 - \eta\lambda|$ 关于 $\lambda$ 的极值必在端点。端点平衡 $1 - \eta\mu = -(1 - \eta L)$ 给出最优定步长与收缩因子：

$$
\eta^{\ast} = \frac{2}{L + \mu}, \qquad \rho_{\mathrm{GD}} = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1} \qquad (16)
$$

大 $\kappa$ 下 $\rho_{\mathrm{GD}} \approx 1 - 2/\kappa$ ：每步只消掉最慢方向误差的一丁点。机制读法：步长被最陡方向压住（ $\eta \le 2/L$ 才不发散），最平方向的收缩系数 $1 - \eta\mu \approx 1 - 2/\kappa$ 几乎不动。这就是「峡谷里的 GD」：快方向来回弹跳（zigzag），慢方向前进极慢。

**Heavy ball 的速率**（[Polyak, 1964](../references.md)，USSR Computational Mathematics and Mathematical Physics 4(5):1-17，无 arXiv，[B]）。在式 (1) 上加历史项（即式 (8) 的非随机版重排）：

$$
\theta_{t+1} = \theta_t - \eta \thinspace H \theta_t + \beta \thinspace (\theta_t - \theta_{t-1}) \qquad (17)
$$

按特征基分解后每个坐标是二阶线性递推 $e_{t+1} = (1 + \beta - \eta\lambda) \thinspace e_t - \beta \thinspace e_{t-1}$ ，写成状态向量形式：

$$
\begin{bmatrix} e_{t+1} \\ e_t \end{bmatrix} = M \begin{bmatrix} e_t \\ e_{t-1} \end{bmatrix}, \qquad M = \begin{bmatrix} 1 + \beta - \eta\lambda & -\beta \\ 1 & 0 \end{bmatrix}
$$

**符号**： $M \in \mathbb{R}^{2 \times 2}$ ：该特征方向的单步转移矩阵。特征方程

$$
r^{2} - (1 + \beta - \eta\lambda) \thinspace r + \beta = 0 \qquad (18)
$$

该坐标的收缩速率 = $M$ 的谱半径 $\max(|r_1|, |r_2|)$ 。关键观察：当判别式 $(1 + \beta - \eta\lambda)^{2} - 4\beta \lt 0$ 时，两根共轭复，模长 $\sqrt{r_1 r_2} = \sqrt\beta$ （根之积 = $\beta$ ）。也就是说，复根区内速率与 $\lambda$ **无关**，统一为 $\sqrt\beta$ 。最优调参的思路随之明确：把整段 $[\mu, L]$ 罩进复根区，并让两个端点恰好压在边界上：

$$
1 + \beta - \eta\mu = 2\sqrt\beta, \qquad 1 + \beta - \eta L = -2\sqrt\beta \qquad (19)
$$

两式相加得 $\eta = 2(1+\beta)/(L+\mu)$ ；相减得 $\eta(L-\mu) = 4\sqrt\beta$ 。消去 $\eta$ 得 $\dfrac{\kappa-1}{\kappa+1} = \dfrac{2\sqrt\beta}{1+\beta}$ ，解出

$$
\sqrt{\beta^{\ast}} = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}, \qquad \eta^{\ast} = \frac{4}{\left(\sqrt L + \sqrt\mu\right)^{2}}, \qquad \rho_{\mathrm{HB}} = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \qquad (20)
$$

大 $\kappa$ 下 $\rho_{\mathrm{HB}} \approx 1 - 2/\sqrt\kappa$ 。对照式 (16)：达到给定精度所需步数从 $\Theta(\kappa \ln(1/\varepsilon))$ 降到 $\Theta(\sqrt\kappa \ln(1/\varepsilon))$ ——momentum 在二次碗上把条件数开方。数值感（步数 = $\ln 10^{6} / (-\ln \rho)$ ，误差压到 $10^{-6}$ ）：

| $\kappa$ | $\rho_{\mathrm{GD}}$ | $\rho_{\mathrm{HB}}$ | GD 步数 | HB 步数 |
| --- | --- | --- | --- | --- |
| $10^{2}$ | 0.9802 | 0.8182 | 691 | 69 |
| $10^{4}$ | 0.99980 | 0.98020 | 69078 | 691 |

第二行里 HB 的 $\rho$ 恰等于第一行 GD 的 $\rho$ ——同一个 $\sqrt\kappa$ 关系： $\kappa = 10^{4}$ 的 momentum 问题与 $\kappa = 100$ 的裸 GD 问题一样难。

```
二次碗截面（H 的两个特征方向）
  快方向（曲率 L） ↑    GD：η ≤ 2/L 被快方向压住，在碗壁间 zigzag
                   │        慢方向每步只缩 ≈ 1 − 2/κ
  ─────────────────┼───────────────────→ 慢方向（曲率 μ）
                   │    heavy ball：历史速度在快方向符号交替、自相抵消，
                   ↓        在慢方向同号累加；最优调参后所有方向统一
                            以 (√κ−1)/(√κ+1) 收缩
```

三条边界说明。其一，式 (20) 是**渐近最坏方向**口径：heavy ball 的误差曲线振荡下行、不单调，谱半径只控制包络。其二，分析依赖精确二次与定常 $\eta, \beta$ ；随机梯度情形归 §6。其三，深度网络非凸，二次碗只是局部模型。[Sutskever et al., 2013](../references.md)（[B]）的实证贡献正在于展示这个局部模型的适用范围：配好初始化与递增的 $\mu$ schedule 后，其定性预测在深网与长程 RNN 上成立，一阶动量法训出了此前只有 Hessian-Free 二阶方法能到的解。

## 6. Momentum × 梯度噪声：平均降噪与噪声累积的对账

噪声进入 momentum 的账有两面，先算「累积」侧。设梯度噪声逐 step iid、协方差 $\Sigma / B$ （式 (3)），速度缓冲是历史噪声的几何加权和（式 (11)），平稳协方差：

$$
\mathrm{Cov}(v_{\infty}) = \frac{\Sigma}{B} \sum_{k=0}^{\infty} \beta^{2k} = \frac{\Sigma}{B \left(1 - \beta^{2}\right)} \qquad (21)
$$

缓冲区的噪声被**放大** $1/(1-\beta^{2})$ 倍： $\beta = 0.9$ 为 5.26 倍、0.99 为 50 倍。只看这一面，momentum 是噪声的蓄水池。

但这只是存量。流量账要按有效步长来对：每步注入参数的噪声方差为 $\eta^{2} \thinspace \mathrm{tr}\thinspace \mathrm{Cov}(v_{\infty})$ ，换成有效步长 $\eta_{\mathrm{eff}} = \eta/(1-\beta)$ （式 (10)(11) 的口径）表示：

$$
\eta^{2} \cdot \frac{\mathrm{tr}\thinspace \Sigma}{B (1-\beta^{2})} = \eta_{\mathrm{eff}}^{2} \cdot \frac{(1-\beta)^{2}}{1-\beta^{2}} \cdot \frac{\mathrm{tr}\thinspace \Sigma}{B} = \eta_{\mathrm{eff}}^{2} \cdot \frac{1-\beta}{1+\beta} \cdot \frac{\mathrm{tr}\thinspace \Sigma}{B} \qquad (22)
$$

与裸 SGD 在同一 $\eta_{\mathrm{eff}}$ 下的注入 $\eta_{\mathrm{eff}}^{2} \thinspace \mathrm{tr}\thinspace \Sigma / B$ 相比，momentum 的噪声注入只剩 $(1-\beta)/(1+\beta)$ ： $\beta$ = 0.9 / 0.95 / 0.99 对应 1/19 / 1/39 / 1/199。直觉一句话：权重和（ $\approx 1/(1-\beta)$ ）以平方进入有效步长，权重平方和（ $1/(1-\beta^{2})$ ）进入噪声；前者长得更快，平均视野越长，信噪比越高。两面对账完毕：「累积」放大的是缓冲区的存量，「降噪」压缩的是单位有效步长的流量。

代价在偏差侧：EMA 里的梯度信息有平均滞后，几何权重的平均年龄是 $\sum_k k \beta^{k} / \sum_k \beta^{k} = \beta/(1-\beta)$ 步（ $\beta = 0.9$ 为 9 步、0.95 为 19 步）。方向快速转弯的阶段——warmup 结束、schedule 切换点、loss landscape 的鞍部出口——滞后直接拖慢响应。至此 $\beta$ 的全部含义可以绑在一行：视野 $1/(1-\beta)$ 步、噪声注入 $(1-\beta)/(1+\beta)$ 、滞后 $\beta/(1-\beta)$ 步，三者是同一个旋钮。

与 2.3 的接口：McCandlish et al. 的稳态「温度」口径（SGD 稳态分布的尺度 ~ $\eta/B$ ）在 momentum 下把 $\eta$ 换成 $\eta/(1-\beta)$ ——与本节有效步长账是同一事实的两种说法；温度、noise scale 与 critical batch 的完整联动归 2.3。

## 7. 实践惯例与边界： $\beta$ 的读数与 LLM 为什么不是 SGD-momentum

$\beta = 0.9$ 是深度学习的事实默认值。PyTorch 官方 docstring 的示例写作 `SGD(..., lr=0.1, momentum=0.9)` ；[Goyal et al., 2017](../references.md)（[B]）的 8192-batch ImageNet 配方用 Nesterov $m = 0.9$ 、weight decay $10^{-4}$ 、 $\eta = 0.1 \cdot B/256$ 。更大 $\beta$ 的出处是更难优化的问题。[Sutskever et al., 2013](../references.md)（[B]）对深 autoencoder 用递增 schedule $\mu_t = \min\left(1 - 2^{-1-\log_2(\lfloor t/250 \rfloor + 1)}, \mu_{\max}\right)$ ， $\mu_{\max}$ 取自 $\lbrace 0.999, 0.995, 0.99, 0.9, 0 \rbrace$ ；长程 RNN 实验用前 1000 步 $\mu = 0.9$ ，随后切 $\mu_0 \in \lbrace 0, 0.9, 0.98, 0.995 \rbrace$ 。规律：问题越病态——曲率 $\kappa$ 越大、时间依赖越长——视野 $1/(1-\beta)$ 要越长。schedule 递增的原因也直观：训练初期方向尚未对准，长记忆只会固化错误方向。

LLM 预训练的主流却不是 SGD-momentum：各家公开配方的 optimizer 列几乎一律 Adam/AdamW（全字段对照表归 7.2，数学推导归 2.2）。机制层面的动机是什么？LLM 各参数块的梯度幅度跨数量级。embedding 行的更新稀疏而猛烈，dense 矩阵的更新稠密而平缓，logit 尺度随训练漂移。单一标量步长配 momentum 只能按全局几何折中；逐坐标自适应——每坐标独立有效步长——是对这个结构的直接回应，这就是 2.2 的起点。另一条回应路线是按矩阵结构的谱方法（2.5）。口径提醒：LLM 文献里成对出现的「0.9 / 0.95」是 Adam 的 $(\beta_1, \beta_2)$ 两阶矩系数（归 2.2 的超参表），与本章 momentum 的 $\beta$ 不是同一对象；尽管 $\beta_1$ 的机制正是式 (9) 的 EMA，其「0.9」的读数（视野约 10 步）与本章一致。SGD-momentum 留下的基准不会过时：式 (16)(20) 的二次碗速率是全部加速分析的参照系，linear scaling 与 critical batch 的账（2.3）以本章的噪声模型为底层。

## 总结

| 对象 | 公式 | 要点 |
| --- | --- | --- |
| mini-batch 估计 | $\mathbb{E}[\hat g_B] = g$ 、 $\mathrm{Cov} = \Sigma/B$ | 式 (2)(3)；无偏、噪声 ÷ $\sqrt B$ |
| 二阶矩分解 | $\mathbb{E}\lVert\hat g\rVert^{2} = \lVert g\rVert^{2} + \mathrm{tr}\thinspace\Sigma/B$ | 式 (4)；噪声尺度 $`\mathcal{B}_{\mathrm{simple}} = \mathrm{tr}\thinspace\Sigma/\lVert g\rVert^{2}`$ |
| 最优步长 | $`\eta^{\ast} = L^{-1} / \left(1 + \mathcal{B}_{\mathrm{simple}}/B\right)`$ | 式 (7)；小 $B$ 区 $\eta \propto B$（linear scaling 适用区）、大 $B$ 区饱和 $\to 1/L$ |
| momentum 等价 | 式 (8)↔(9)， $\eta' = \eta/(1-\beta)$ | 式 (10)；有效步长 × $1/(1-\beta)$ 、视野 $1/(1-\beta)$ 步 |
| 速度内含步长的约定 | 式 (12)；步长变化需乘 $\eta_{t+1}/\eta_t$ | momentum correction（Goyal §3）；PyTorch 约定免修正 |
| Nesterov | 外推点求梯，式 (13)(14) | 动量项 $\mu \to \mu^{2}$ 、梯度项 $1 \to 1+\mu$ ；PyTorch `nesterov=True` 为同式重排 |
| GD 二次碗 | $\rho = (\kappa-1)/(\kappa+1)$ | 式 (16)；步数 $\Theta(\kappa \ln(1/\varepsilon))$ |
| heavy ball 二次碗 | $\rho = (\sqrt\kappa-1)/(\sqrt\kappa+1)$ | 式 (20)；Polyak 1964；步数 $\Theta(\sqrt\kappa \ln(1/\varepsilon))$ |
| 噪声 × momentum | 缓冲放大 $1/(1-\beta^{2})$ ；等有效步长注入 $(1-\beta)/(1+\beta)$ | 式 (21)(22)； $\beta = 0.9$ ⇒ 注入 1/19；滞后 $\beta/(1-\beta)$ 步 |

## 练习

1.（推导）不借用光滑性放缩，对精确二次 $\mathcal{L} = \frac{1}{2}\theta^{\top}H\theta$ 直接展开 $\mathbb{E}\left[\hat g^{\top} H \hat g\right]$ （提示： $\hat g = H\theta + \xi$ ， $\mathbb{E}[\xi] = 0$ 、 $\mathrm{Cov}(\xi) = \Sigma/B$ ），给出 $`\eta^{\ast}(B)`$ 的精确式，并说明何时退回式 (7) 的上界版。
2.（推导）从式 (18) 出发：(a) 验证复根区 $|r| = \sqrt\beta$ 与 $\lambda$ 无关；(b) 补全式 (19) 到式 (20) 的代数；(c) 证明当某方向落在实根区（判别式 $> 0$ ）时其收缩慢于 $\sqrt\beta$ ——最优调参必须把 $[\mu, L]$ 全部罩进复根区。
3.（计算） $\beta = 0.95$ ：(a) 平均视野 $1/(1-\beta)$ 与覆盖 90% 权重所需步数；(b) 缓冲区噪声放大倍数 $1/(1-\beta^{2})$ ；(c) 等有效步长下相对裸 SGD 的噪声注入比 $(1-\beta)/(1+\beta)$ ；(d) 信息平均滞后 $\beta/(1-\beta)$ 。四个数写在一行，说明调 $\beta$ 时 $\eta$ 与 warmup 步数各应如何联动。
4.（代码）(a) numpy 复现： $H$ 取对角阵、特征值在 $[\mu, L]$ 上对数等距， $\kappa = 10^{4}$ ，按式 (16) 与式 (20) 调参分别跑 GD 与 heavy ball，把 $\log \mathcal{L}$ 曲线的斜率对回两个 $\rho$ 。(b) PyTorch 侧：同一损失上验证 `SGD(momentum=β, lr=η)` 与手写 EMA 形式（式 (9)， $\eta' = \eta/(1-\beta)$ ）逐 step 参数一致；再验证 `nesterov=True` 与式 (13) 的手写实现在外推点序列上逐步一致。

## 参考

- [Polyak, 1964](../references.md)：Some Methods of Speeding up the Convergence of Iteration Methods，USSR Computational Mathematics and Mathematical Physics 4(5):1-17，无 arXiv，[B]（key: polyak1964）——heavy ball 与二次碗最优速率（式 (17)–(20)）
- [Sutskever et al., 2013](../references.md)：On the Importance of Initialization and Momentum in Deep Learning，ICML 2013（PMLR 28(3):1139-1147），无 arXiv，[B]（key: sutskever2013）——CM/NAG 形式（式 (12)(13)）、 $\mu$ 递增 schedule、深网与 RNN 实证
- [McCandlish et al., 2018](../references.md)：An Empirical Model of Large-Batch Training，arXiv:1812.06162，[B]（key: critical-batch）——式 (3) 同式（其式 (2.2)）、梯度噪声尺度与温度口径；critical batch 归 2.3
- [Goyal et al., 2017](../references.md)：Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour，arXiv:1706.02677，[B]（key: linear-scaling）——linear scaling rule、momentum correction（其 §3 式 (9)–(10)）、Nesterov $m = 0.9$ 配方
- [PyTorch, pytorch/pytorch repo](../references.md)（key: pytorch-repo，教学参照）——`torch/optim/sgd.py`：`SGD.step` → `sgd` → `_single_tensor_sgd` 逐行对照；docstring 的 Sutskever 约定差异注记
