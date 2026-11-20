# 1.1 计算图与 autograd 的执行机制（Computation Graphs and the Execution Mechanics of Autograd）

> 本章回答一个问题：`loss.backward()` 这一行背后发生了什么。先给训练 step 的三步形式化，把计算图定义成「算子节点 × tensor 边」的 DAG；再把链式法则落成执行形式——adjoint 从损失端逐节点左乘、每步只算 vjp、永不物化 Jacobian。然后算两笔账：前向模式 vs 反向模式谁划算（标量损失 ⇒ 只能反向），反向 FLOPs 为什么 ≈ 2× 前向（闭合 models 1.2 的 6ND 口径边界）。最后走读 PyTorch autograd 的实现：grad_fn 链、AccumulateGrad、save_for_backward 的内存语义，以及 retain_graph / no_grad / inference_mode / detach 的精确语义 | 前置：models 1.1（语言建模形式化）、models 1.2（ $2mnk$ 计数口径、 $2N$ / $6ND$ 的账、激活账二）| 锚点：PyTorch autograd（pytorch/pytorch repo，`torch/autograd/` 与 `torch/csrc/autograd/`，教学参照）；Baydin et al.（[B]，AD 综述）；Li et al.（[A]，DDP 论文的引擎与 hook 机制背景）

## 1. 训练 step 的三步形式化

一个训练 step 是三件事，对应三段性质完全不同的程序：

$$
\text{前向：}\quad L_t = \ell\bigl(f_{\theta_t}(x^{(t)}), y^{(t)}\bigr); \qquad \text{反向：}\quad g_t = \nabla_\theta L_t; \qquad \text{更新：}\quad \theta_{t+1} = \theta_t - \eta_t \thinspace u_t(g_t) \qquad (1)
$$

**符号**： $\theta_t$：第 $t$ 步的参数（全部可训练参数拼成的向量）； $x^{(t)}, y^{(t)}$：第 $t$ 步的数据批次； $\ell$：损失函数（语言建模的交叉熵口径见 models 1.1，训练侧实现归 3.1）； $g_t$：参数梯度； $\eta_t$：学习率； $u_t(\cdot)$：优化器的更新映射（SGD 下是恒等，AdamW 下是矩估计归一化加解耦衰减，推导归 2.1 / 2.2）。

三步里只有第二步不是「把已写好的程序再跑一遍」。前向是你显式写出的程序；反向不是人写的，是框架从计算图自动生成的另一段程序。本章回答两个问题：这段自动生成的程序长什么样（§2–§4），它由什么机制执行（§6）。先立一个成本锚点：按 models 1.2 的 8B 口径，每 token 前向约 $1.5 \times 10^{10}$ FLOPs，反向约为其两倍。这个「两倍」不是经验值，是结构结论——依据在 §5，逐算子的证明在 1.2。

本章口径一句话：参数量与算力账用 8B 模型（层参数口径为主，含输出头处随文标注）；token 数分两处——模式账用 4096，激活账用 32768，均随文标注；精度默认 bf16。

## 2. 计算图：算子作节点、tensor 作边的 DAG

取最小非平凡例子 $y = (x W_1) W_2 + b$、 $L = \mathrm{mean}(y^{\odot 2})$（ $\odot$ 为逐元素乘），它的前向数据流与反向 adjoint 流：

```
前向（节点 = 算子，边 = tensor，标注 shape）          反向（adjoint 流，式 (4) 的逐节点 vjp——vjp 在 §3 定义）

x [B,T,d1] ─┐                                           
            ▼                                           
          mm(W1) ──► h1 [B,T,d2] ──┐               dL/dh1 = dL/dh2 · W2ᵀ
                     W1 [d1,d2] ───┘(参数边)             dL/dW1 = xᵀ · dL/dh1
                                   ▼                    
                                 mm(W2) ──► h2 [B,T,d3] ──┐      dL/dx  = dL/dh1 · W1ᵀ
                                  W2 [d2,d3] ──┘          ▼
                                          b [d3] ──► add ──► y [B,T,d3] ──► mean(·²) ──► L（标量）
                                                                                        ▲
                                                        种子：dL/dL = 1（标量损失的默认 adjoint）
```

把这张图形式化：计算图（computation graph）是有向无环图（directed acyclic graph, DAG）。节点是算子实例（一次 `mm`、一次 `add`），边是 tensor（算子的输入输出）。无环是训练图的结构性事实：decoder-only LLM 的前向是无环数据流，梯度沿边反向流动时遇不到环。无环还有一个直接的工程含义：DAG 必有拓扑序，「反向按逆拓扑序逐节点执行」才有定义——§6.1 的引擎调度就建立在这上面。反例顺便定边界：带环的数据流（比如按时间展开的循环结构）拓扑序不存在，反向调度无从谈起——无环因此必须写进定义。

DAG 不是树。残差连接 $y = x + h(x)$ 让 $x$ 同时被两个算子消费（fan-out），损失沿多条路径依赖同一个 tensor。梯度侧的对应规则是「沿全部路径求和」——这是多元链式法则的直接推论：若 $x$ 被 $f_a$ 与 $f_b$ 消费，则

$$
\bar{x} = \bar{x}^{(a)} + \bar{x}^{(b)} \qquad (2)
$$

**符号**：所谓伴随量（adjoint），是指标量损失对某中间 tensor $x$ 的梯度，记 $\bar{x} := \partial L / \partial x$；上标 $(a), (b)$ 区分两条路径各自回传的分量。代入整数看一眼：两路各回传 3 与 4，该 tensor 收到的梯度是 7。式 (2) 是「梯度累加」语义的数学原型：PyTorch 的 `.grad` 是 `+=` 累加、gradient accumulation（1.5）把同一参数的多份梯度相加，都源自这一求和规则。这条规则的完整证明留作练习 4。

## 3. 链式法则的执行形式：逐节点 vjp，永不物化 Jacobian

链式法则给出的是一个矩阵连乘恒等式，但恒等式不告诉你从哪头开始乘——顺序不同，成本差多少？路线：先写出连乘形式，再用结合律比较两种求值顺序，关键一步是「向量乘矩阵与矩阵乘矩阵不同价」。把前向写成复合链 $x_k = f_k(x_{k-1})$， $k = 1, \ldots, K$， $x_k \in \mathbb{R}^{n_k}$，损失 $L = \ell(x_K) \in \mathbb{R}$（参数含在 $x_0$ 里）。链式法则给出梯度的 Jacobian 连乘积：

$$
\frac{\partial L}{\partial x_0} = \nabla \ell \cdot J_K \cdot J_{K-1} \cdots J_1, \qquad J_k := \frac{\partial f_k}{\partial x_{k-1}} \in \mathbb{R}^{n_k \times n_{k-1}} \qquad (3)
$$

**符号**： $\nabla \ell \in \mathbb{R}^{1 \times n_K}$：损失对末层激活的梯度（行向量）； $J_k$：第 $k$ 个算子的雅可比矩阵（Jacobian matrix）。

矩阵乘法有结合律，所以从哪头开始乘是自由选择的；但两种顺序的成本天差地别。先看从右往左。它要先算两个 $n \times n$ 矩阵之积（ $O(n^{3})$ ），逐层压出全链 Jacobian，最后再乘向量；而且每个 $J_k$ 得先物化出来，占 $n_k \times n_{k-1}$ 个元素。物化的代价有多大？代入现实尺度： $d$ = 4096 的逐位置算子作用在 $M$ = 4096 个 token 上，每个 token 对应一个 $4096 \times 4096$ 的 Jacobian，bf16 下共 $M d^{2} \times 2$ bytes = 128 GiB。这只是一层、一个 step、一份 Jacobian；32 层的 8B 模型连一层的零头都放不下。FLOPs 同样不成立：从右往左乘全链，每 token 约 $K n^{3}$， $K$ = 32、 $n$ = 4096 时约 $2.2 \times 10^{12}$——是同 token 前向（ $2N \approx 1.4 \times 10^{10}$ ）的 150 倍以上。

从左往右才是反向传播。定义逐节点 adjoint $`\bar{x}_k`$ $:= \partial L / \partial x_k \in \mathbb{R}^{1 \times n_k}$，递推：

$$
\bar{x}_K = \nabla \ell, \qquad \bar{x}_{k-1} = \bar{x}_k \thinspace J_k, \quad k = K, \ldots, 1 \qquad (4)
$$

每一步算的是「行向量 × Jacobian」，这个乘积叫向量-雅可比积（vector-Jacobian product, vjp）。关键事实是：对每个具体算子， $`\bar{x}_k \thinspace J_k`$ 都有不经过 $J_k$ 的闭式。看两个例子。逐元素算子 $f(x) = \varphi(x)$ 的 Jacobian 是对角阵 $J = \mathrm{diag}(\varphi'(x))$，它的 vjp 是 $`\bar{x} \odot \varphi'(x)`$ ——只做 $O(n)$ 次乘加，那个 $n \times n$ 的对角矩阵从头到尾没有出现。matmul 的 vjp 是另一次同形 matmul（逐元素推导在 1.2 §2）。**反向传播就是式 (4) 从损失端向左逐节点执行：每步只调该算子的 vjp，Jacobian 永不物化。** 遇到 fan-out 节点，按式 (2) 把多条回传路径的 adjoint 相加，再继续左传。

成本也随之可控。每个算子的 vjp 与其前向同阶（差一个小常数），所以整条反向链的总成本是 $O(\sum_k \mathrm{cost}(f_k))$——前向的小常数倍。「反向与前向同阶」的一般论证就在这里；精确常数（matmul 网络为 2）在 §5 与 1.2 给出。

这套「把程序变换成逐节点 vjp 链」的视角，是自动微分（automatic differentiation, AD）文献的标准框架（[Baydin et al., 2018](../references.md)，arXiv:1502.05767，[B]）。它和两种替代方案的区分值得记住。符号微分对表达式树整体求导，表达式随深度指数膨胀。数值微分用差商逼近，既有截断误差，又要每个输入方向各算一次前向。AD 在算子粒度套用链式法则：精度精确到浮点，成本是前向的常数倍。这是它能支撑全部现代训练的根据。

## 4. 前向模式 vs 反向模式：输入维与输出维的账

式 (3) 还有第三种乘法顺序：从输入端向右传播切线（tangent）。递推写成 $`\dot{x}_k = J_k \thinspace \dot{x}_{k-1}`$，叫 Jacobian-向量积（jvp）：从 $`\dot{x}_0`$（输入的扰动方向）出发逐节点右传，终点 $`\dot{x}_K`$ 就是 $L$ 沿 $`\dot{x}_0`$ 方向的方向导数。这是前向模式（forward mode）；式 (4) 那样从损失端左传的是反向模式（reverse mode）。下文「一巡」的口径：沿图完整执行一遍对应模式。两种模式的能力不对称：

- 一巡前向模式（选一个 $`\dot{x}_0`$ ）产出标量输出沿**一个输入方向**的导数；
- 一巡反向模式（选一个输出分量的种子 adjoint）产出**该输出对全部输入**的梯度。

账很好算。对 $f: \mathbb{R}^{n} \to \mathbb{R}^{m}$，求完整 Jacobian 需要 $n$ 次前向模式（逐输入方向喂单位切线），或者 $m$ 次反向模式（逐输出分量喂单位 adjoint）；每巡的成本都是前向求值的小常数倍（[Baydin et al., 2018](../references.md)，[B]）。所以谁划算由 $n$ 与 $m$ 的相对大小决定： $n \ll m$ 用前向， $m \ll n$ 用反向。

LLM 把这个比值推到了极端。这里 $n$ = 参数量：8B 模型 $N \approx 7 \times 10^{9}$（models 1.2 §9 的层参数口径）。而 $m$ = 1——损失是标量。前向模式求一次全梯度要约 $7 \times 10^{9}$ 次前向当量；反向模式只要一巡。折成时间更直白：batch 4096 token 的一次前向约 $6 \times 10^{13}$ FLOPs，乘上 $7 \times 10^{9}$ 巡，按 $10^{15}$ FLOPS 的有效吞吐（口径：可持续的实际吞吐，非峰值）折合约 14 年；反向模式一巡约 0.1 秒。九个数量级的差距没有工程补救空间——LLM 训练的梯度计算只能用反向模式。这不是框架选择，是结构决定。练习 2 把这笔账的分步算术走一遍。

反向模式赢下的梯度不是免费的（口径：「免费」指不另耗 FLOPs），代价写在内存上。式 (4) 第 $k$ 步的 vjp 要用前向的中间值——matmul 的 vjp 就需要前向输入 $x_{k-1}$ 。于是前向的每个中间 tensor 都必须活到反向路过它为止。这就是 models 1.2 账二（激活内存）的源头；逐算子「存什么」的清单在 1.2 §11，「少存、用时重算」的压缩手段在 1.4。前向模式不需要存中间值：切线与前向同行，内存 $O(1)$——但它算不起。PyTorch 也暴露了前向模式接口（`torch.autograd.forward_ad`，教学参照），在 LLM 规模下它只出现在 Hessian-向量积这类研究场景里。

> **提点**：内存与方向的互换是 AD 里最经典的权衡——反向模式用内存换「一次拿全梯度」，前向模式用时间换「不存中间值」。后文所有的激活压缩手段，都在这条权衡的反向模式一侧做文章。

## 5. 反向 FLOPs ≈ 2× 前向的依据

models 1.2 §4 用过这个比例（其式 (4)：反向 $4N$ = 两倍前向 $2N$ ），本节给出它的一般依据，逐元素推导在 1.2 §2。

**命题**（反向 ≈ 2× 前向）：matmul 主导的网络里，反向传播的 FLOPs 约为前向的两倍——matmul 项上精确成立，逐元素与归一化项同阶、被淹没；结构结论，不是经验值。

**证明**（要点）：路线是先证 matmul 特例，再看一般算子的结构，最后看非 matmul 项为什么不翻盘。先看 matmul 特例。 $Y = AB$ 的前向是 $2mnk$ FLOPs（models 1.2 式 (1) 的口径）。反向要为两个输入各算一份 vjp： $\partial L / \partial A = G B^{\top}$ 与 $\partial L / \partial B = A^{\top} G$（ $G = \partial L / \partial Y$ ）。这两个各是一次同形 matmul，各 $2mnk$，合计 $4mnk$——恰为前向两倍，特例成立。再看一般算子 $y = f(x, W)$：反向要为每个可微输入各算一份 vjp——输入侧那份回传上游，权重侧那份累加进 `.grad`。matmul 类算子的两份 vjp 都与前向同价；逐元素算子（GELU、dropout）前向反向同为 $O(n)$ ；归一化类（RMSNorm）反向是前向的小常数倍（多两次内积）。后两类都被 matmul 项淹没，所以命题成立。

代入 8B 口径的绝对数：每 token 前向 $2N \approx 1.4 \times 10^{10}$ FLOPs（层参数口径；含输出头约 $1.5 \times 10^{10}$，即 §1 与 §4 用的口径），反向 $4N \approx 2.8 \times 10^{10}$ FLOPs。6ND 的系数 6 就是 2 + 4 这样拼出来的（models 1.2 §5 的口径）。

这笔账顺便闭合了 models 1.2 的口径边界。 $C = 6ND$ = （前向 $2N$ + 反向 $4N$ ）× $D$，其适用条件声明「忽略 attention 二次项」。二次项服从同一比例： $QK^{\top}$ 与 $PV$ 两个 batched matmul，反向各是两次同形 matmul，反向 = $2 \times 2LT^{2}d$ 。所以 6ND「不含二次项」不是近似误差，只是把一笔同样满足 1:2 结构的账单另列了——需要时按 $3 \times 2LT^{2}d$（前向 + 反向）加回即可。有一个例外值得预告：重算型实现（FlashAttention）的反向要多付一次打分矩阵的重算，反向 = 2.5× 前向（推导在 1.2 §9）。「反向恒为 2 倍」在重计算算子处失效，这正是 1.4 重计算账的基础。

## 6. PyTorch autograd 机制走读

前五节的数学，在 PyTorch 里具体长什么样？本节落到实现来回答。走读对象是 pytorch/pytorch repo（教学参照）；引擎与 hook 的机制级描述另见 [Li et al., 2020](../references.md)（arXiv:2006.15704，[A]，§3）。

### 6.1 grad_fn 链、leaf 与 AccumulateGrad

先说记录规则。前向执行时，一个算子调用只要有一个输入 `requires_grad=True`，就会实例化一个反向节点（C++ 的 `Node`，Python 侧暴露为输出 tensor 的 `.grad_fn` ）；节点的 `next_functions` 字段存指向各输入反向节点的边。图没有显式邻接表——拓扑就靠这条 `grad_fn` 指针链隐式维护，而且每次前向都重建（§7）。 `requires_grad` 沿算子传播的规则只有一句话：输出的记录开关 = 任一输入的记录开关。全部输入都不要求梯度时，这次调用不产生节点，输出也是「不要梯度」的普通 tensor。`no_grad` 与 `detach` 是这个传播规则的两个出口（§6.3）。

叶子张量（leaf tensor）是你直接创建、不是算子产物的 tensor，它的 `.grad_fn` 为 None。 `requires_grad=True` 的叶子在图里对应一个 `AccumulateGrad` 节点（`torch/csrc/autograd/functions/accumulate_grad.cpp`，教学参照）：反向到达叶子时，adjoint 按 `+=` 累加进 `.grad` 。注意是累加不是覆盖——所以每个 step 开始前要显式 `zero_grad()` 清零。这正是式 (2) 求和规则在参数侧的落点：同一参数被多处消费时，各路径的梯度在这里汇合。

再看执行。 `loss.backward()` 由引擎（`torch/csrc/autograd/engine.cpp`，教学参照）建一次执行任务，从损失节点出发按依赖逆序调度：一个节点的全部下游 adjoint 到齐（fan-in 求和完成），它就进入就绪队列（ready queue），按设备分 worker 线程执行；它的反向一算完，就触发其上游节点就绪。种子语义也值得记住：标量损失的默认初始 adjoint 是 1；对非标量输出调 `backward` 必须显式传 `gradient=v`——语义就是以 $v$ 为初始 adjoint 做一次 vjp，这是式 (4) 的 API 形态。

这套「梯度就绪即触发」的接口，正是分布式的挂接点。DDP 在构造期给每个参数的 gradient accumulator 装 post-hook：某个参数的梯度一就绪，就触发所属 bucket 的计数；整桶到齐，就异步发 all-reduce。通信由此与反向重叠（[Li et al., 2020](../references.md)，[A]，§3 的伪代码口径；DDP 的机制与通信账归 4.1）。

### 6.2 ctx.save_for_backward：存什么 = 激活账的源头

自定义算子的标准写法长这样（`torch/autograd/function.py` 的 `Function`，教学参照）：`forward(ctx, ...)` 里用 `ctx.save_for_backward(t)` 把反向要用的 tensor 挂上节点，`backward(ctx, G)` 里从 `ctx.saved_tensors` 取回。被 save 的 tensor 引用计数加一，活到该次反向执行完。所以**逐算子的 save 清单就是激活内存账单**：matmul 要 save 两个输入（两个 vjp 各要一个），softmax 只需 save 输出 $p$，embedding 只需 save 整数下标。1.2 §11 的总表会逐算子列出「save 什么、反向/前向 FLOPs 比」；1.4 的 gradient checkpointing 就是把「少 save、用时重算」系统化。量级实例： $M$ = 32768 token、 $k$ = 4096 的一个线性层，save 的输入是 $M k \times 2$ bytes = 256 MiB。一个 step 里每个线性层都存这样一份，乘上层数就是激活账的体量。

配套的正确性保险丝是版本计数（version counter）：被 save 的 tensor 若在反向前被原地修改（in-place），反向时报「modified by an inplace operation」类 RuntimeError——防止反向读到被污染的中间值。这也是 §6.3 里 `inference_mode` 关掉计数后更快的对照面。

> **坑**：对 autograd save 过的 tensor 做原地修改（in-place）——反向在真正用到它时才报「modified by an inplace operation」，错误现场离肇事点很远，排查费劲。规避：要改值先 clone；写自定义算子时把输入当只读。

### 6.3 retain_graph / no_grad / inference_mode / detach 的精确语义

四个 API 常被混用，语义各不相同：

- 默认情况下 `backward()` 执行完就释放本次保存量（save 的引用归还）；对同一图再次 `backward` 会报「Trying to backward through the graph a second time」。 `retain_graph=True` 让保存量活下来，允许同一图多次反向——代价是激活内存一直驻留到最后一次反向，且 `.grad` 照式 (2) 继续累加。量级上：8B 模型、 $B \cdot T$ = 32768 的激活驻留在百 GiB 级（models 1.2 账二的口径），两次反向期间它一直占着。 `retain_graph` 就是用显存驻留时长换「不重跑前向」的开关。
- `torch.no_grad()`：上下文内不记录——产出 tensor 的 `requires_grad=False`、无 `grad_fn`，图不生长。评估与统计收集的标准姿势。
- `torch.inference_mode()`：`no_grad` 的严格超集，额外关闭版本计数与 view 跟踪，更快也更省；代价是其中创建的 inference tensor 不能再进入记录的图（把它喂给需要反向的计算会报错），离开上下文后也不能被原地修改。部署侧默认值。
- `detach()`：返回与原 tensor 共享存储的新 tensor， `requires_grad=False`、无 `grad_fn` ——梯度流在此截断。注意存储共享意味着版本计数也共享：对 detach 出的 tensor 原地修改，原 tensor 的 autograd 安全检查照样能侦测到；它不是复制，写它会改原 tensor。

### 6.4 最小代码对照

把本节的机制压成一小段教学代码，跑一遍就能对上号：

```python
import torch

class MyLinear(torch.autograd.Function):
    """教学用，非优化实现。y = x @ W + b 的手写 autograd 节点。"""

    @staticmethod
    def forward(ctx, x, W, b):
        ctx.save_for_backward(x, W)   # 两个 vjp 分别要用 x 与 W（§5）
        return x @ W + b

    @staticmethod
    def backward(ctx, G):             # G = ∂L/∂y，shape 与 y 相同
        x, W = ctx.saved_tensors
        dx = G @ W.T                  # 输入侧 vjp：G Bᵀ 的实例
        dW = x.T @ G                  # 权重侧 vjp：对 batch 维的求和内含于 xᵀG
        db = G.sum(dim=0)             # bias 沿 batch 维累加（式 (2) 的实例）
        return dx, dW, db             # 按 forward 的参数顺序逐个返回梯度

x = torch.randn(4, 3, requires_grad=True)   # leaf：grad_fn 为 None
W = torch.randn(3, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)
y = MyLinear.apply(x, W, b)
assert y.grad_fn is not None          # 非 leaf：记录了 MyLinearBackward 节点
print(y.grad_fn.next_functions)       # 指回 x / W / b 的 AccumulateGrad 节点的三条边
loss = y.square().mean()
loss.backward()                       # 标量损失，种子 adjoint = 1
assert x.grad.shape == x.shape and W.grad.shape == W.shape
```

数值验证的模式（`gradcheck`、与 `torch.nn.Linear` 对照）在 1.2 §10 的教学代码里完整给出。跑这段代码能直接观察到两条机制事实。第一， `y.grad_fn` 是 `MyLinearBackward` 节点，它的 `next_functions` 是三条形如 `(AccumulateGrad 对象, 0)` 的边——`0` 是上游节点输出的槽位号。第二， `loss.backward()` 之后只有三个 leaf 的 `.grad` 非 None； `y` 与 `loss` 这些中间量的梯度默认不驻留（`retain_grad()` 可以改变这一点）。这正是式 (4) 的执行图景：adjoint 只在节点间流动，到达叶子才沉淀为 `.grad`。

## 7. 动态图的工程含义与图捕获

PyTorch 的图是 define-by-run（动态图）：每行前向代码执行时才实例化节点，图随每次前向重建。这带来两条工程含义。其一，Python 控制流（`if` / `for` / 数据依赖的分支）天然可微——图只是当次执行轨迹的记录，没走到的分支不在图里。其二，调试粒度是逐行 Python：任何中间 tensor 都可以直接打印、下断点，报错能指到具体的算子调用。代价也有两笔：逐算子的 Python 派发与图元数据开销（口径：「开销」此处指墙钟时间，显存另算），外加编译器看不到跨算子全貌、无从融合。这笔开销在 LLM 训练里不算大：单算子派发是微秒级，一个 $d$ = 4096 的 matmul 是毫秒级。但模型变小或控制流变密时（RL rollout 的小步循环、agent 场景的条件分支），派发占比会升到两位数百分比——动态图的便利与开销，必须放在一起计价。把图静态化再做优化（`torch.compile` 的图捕获与 kernel 融合、CUDA graph 重放），消除的正是这两笔开销；其机制归 gpu 维度（kernel 与图捕获），training 侧只在训练栈配置面遇到（6.3）。编译栈对反向的处理同样在演进：`torch.compile` 的 AOTAutograd 在前向追踪时同步切出反向图，Compiled Autograd 再把反向执行本身纳入整图优化——§6 的逐节点机制在编译路径上变成可被全局重排与融合的对象。数学内容不变，变的是调度粒度。

## 总结

| 概念 | 公式 / 机制 | 要点 |
| --- | --- | --- |
| 训练 step | 式 (1)：前向 → 反向 → 更新 | 只有反向是自动生成的程序 |
| 计算图 | 节点 = 算子、边 = tensor 的 DAG | fan-out 处 adjoint 求和（式 (2)），`.grad` 累加的原型 |
| 链式法则执行 | 式 (3)(4)： $`\bar{x}_{k-1} = \bar{x}_k \thinspace J_k`$ | 逐节点 vjp、永不物化 Jacobian；物化一层（ $d$ = 4096、 $M$ = 4096）即 128 GiB |
| 两种模式 | $n$ 次 jvp vs $m$ 次 vjp | LLM： $n \approx 7 \times 10^{9}$、 $m$ = 1 ⇒ 只能反向模式（练习 2） |
| 成本结构 | 反向 ≈ 2× 前向（matmul： $G B^{\top}$ + $A^{\top} G$ ） | 6ND 的反向 $4N$ 由此而来；重算型算子 2.5×（1.2 §9） |
| 内存权衡 | 反向模式存全部中间值 | 激活账源头（models 1.2 账二）；压缩归 1.4 |
| autograd 实现 | grad_fn 链 / AccumulateGrad / save_for_backward / 版本计数 | 引擎按依赖逆序调度；DDP hook 挂 accumulator（4.1） |
| 图模式四 API | retain_graph / no_grad / inference_mode / detach | 语义差见 §6.3 |

## 练习

1.（推导 · [20]）设 $x_k = f_k(x_{k-1})$， $k = 1, \ldots, K$，全部 $n_k = n$。(a) 证明「物化式 (3) 的全链 Jacobian」（从右往左逐层乘）需要 $O(K n^{3})$ FLOPs 与 $O(n^{2})$ 内存，而式 (4) 的 vjp 链只需 $O(K n^{2})$ FLOPs 与 $O(n)$ 的额外内存（前向激活不计）。(b) 当 $f_k$ 全部是逐元素算子时，两种方案各是多少？由此说明 vjp 的优势只在 Jacobian 稠密（matmul 类）时显著。

2.（计算 · [20]）用「每巡成本 = 前向求值」口径估算 LLM 全梯度的模式差：8B 模型（ $n$ = $7 \times 10^{9}$ ）、一个 step 4096 token、每 token 前向 $1.5 \times 10^{10}$ FLOPs。(a) 一次前向多少 FLOPs？(b) 前向模式求全梯度（ $7 \times 10^{9}$ 巡）共多少 FLOPs？按 $10^{15}$ FLOPS 有效吞吐折合多少天？(c) 反向模式一巡（≈2× 前向）折合约多少秒？（有效吞吐的量级口径归 gpu 维度，此处只取数量级。）先写出你对 (c) 的数量级预测再计算。校验：(b) 与 (c) 的比值应等于巡数的一半——(c) 那一巡本身约为两倍前向。

3.（代码 · [30]）用 §6.4 的 `MyLinear` 与 `torch.nn.Linear` 对同一组随机输入比对前向输出与 `x.grad / W.grad / b.grad`（fp64，`atol=1e-10` ）；再用 `torch.autograd.gradcheck` 验证 `MyLinear` 的反向（注意 gradcheck 要求 float64 输入）。最后从 `loss.grad_fn` 出发沿 `next_functions` 走三层，逐个打印节点类型，指出 AccumulateGrad 出现在什么位置。校验：三份梯度与 `torch.nn.Linear` 逐一对上（题面容差即锚）。

4.（推导 + 代码 · [30]）fan-out 求和的实验：构造 $y = h(x) + h(x)$（同一算子消费 $x$ 两次）。先按式 (2) 写出 $\bar{x}$ 的两路径求和；再在 PyTorch 里验证 `x.grad` 等于两份 vjp 之和。追加一问：以 `retain_graph=True` 连续两次 `backward()` 后 `x.grad` 是单次的几倍？先预测倍数再跑，用 §6.1 的累加语义解释。校验：两次 backward 后 `x.grad` 恰为单次的 2 倍。

## 参考

- [Baydin et al., 2018](../references.md)：Automatic Differentiation in Machine Learning: a Survey，arXiv:1502.05767，[B]（key: baydin-ad-survey）——前向/反向模式的分类、vjp/jvp 术语与「每巡成本 = 前向小常数倍」的成本框架
- [Li et al., 2020](../references.md)：PyTorch Distributed: Experiences on Accelerating Data Parallel Training，arXiv:2006.15704，[A]（key: pytorch-ddp）——§3：autograd 引擎调度、gradient accumulator post-hook 与 bucket 化 all-reduce 的机制（4.1 的背景）
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照（key: pytorch-repo）——`torch/autograd/function.py`（`Function` 与 `ctx.save_for_backward`）、`torch/csrc/autograd/engine.cpp`（反向调度引擎）、`torch/csrc/autograd/functions/accumulate_grad.cpp`（AccumulateGrad）
