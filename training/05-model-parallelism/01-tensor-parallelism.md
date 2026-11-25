# 5.1 TP：列切、行切与 all-reduce 位置（Tensor Parallelism: Column/Row Sharding and Where the All-Reduce Sits）

> 本章回答一个问题：一层 transformer 的参数沿宽度劈到多张卡上，通信落在什么位置、付多少字节。内容分四步。第一， $Y = XA$ 的列切与行切两条通信规则（1.2 matmul 反向母式的直接应用）。第二，Megatron 的 f/g 共轭恒等算子——f 前向恒等、反向 all-reduce，g 恰好相反——本章推导它们，用 autograd Function 写出，并对照官方实现。第三，MLP 块与 attention 块的切法：逐位置算子为什么免通信、attention 为什么能按 head 切，各有一个论证。第四，通信账：每层 2 次前向 all-reduce 的位置图，消息量 $2bsh$ bytes，前向每层 $4bsh$、反向对称翻倍，全模型逐层叠加。最后讨论 TP 度数的两类上限（整除约束；通信-计算比随 $t$ 线性涨的 NVLink 边界论证），以及 GQA / MLA 下的变化与 DeepSeek-V3 不用 TP 的论文口径 | 前置：1.2（matmul 反向，式 (2)）、1.4（激活逐算子账——TP 后冗余的结构）、4.1（ring all-reduce 通信量，式 (2)）、4.4（DP 家族决策树的边界）；models 2.3（attention 公式）、models 4.1（MLA 结构） | 锚点：Megatron-LM（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]；NVIDIA/Megatron-LM repo `megatron/core/tensor_parallel/layers.py` 与 `mappings.py`，路径与类名核实于 2026-11-16）、Megatron SP 激活账（[Korthikanti et al., 2022](../references.md)，[A]）、Llama-3 herd（[Grattafiori et al., 2024](../references.md)，[A]，405B 几何与 TP8 配置）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，[A]）；PyTorch autograd Function 机制（pytorch/pytorch repo，教学参照）

## 1. 切分对象：DP 家族之外的那一维

什么时候 DP 家族不够用？4.4 的决策树走到尽头时会撞两种墙。第一，单层参数本身就超过一张卡：三态切分再狠，也只是把状态摊到更多卡上，单算子的前向仍要在一张卡上算出来。第二，激活随 $B \cdot T$ 涨，涨到通信换显存也兜不住（1.4 的账）。张量并行的回答是换一根切分轴。所谓张量并行（TP），是指把同一层的参数沿宽度维切到多张卡、层与层的连接处用集合通信把结果拼回去的切分方式：DP 沿 batch 维切数据，TP 沿**宽度维**切参数。同一层的权重矩阵被劈成 $t$ 份，每卡只持一份分片、只算自己那一份的乘加。

**符号约定**（本章及 P5 全 Part，与 spec §4.3 及 1.4 / 4.x 一致）： $t$：TP 度数； $b$：每卡 micro-batch 序列数； $s$：序列长度； $h$：hidden 宽度； $a$：attention head 数； $L$：层数； $N$：参数量；DP 度数恒记 $d$。激活默认 bf16（每元素 2 bytes，1.3 的配置口径）。本章口径：数字实例只用两组固定标尺——405B 训练配置与 8B 几何，各自在首次使用处给出出处；符号口径全章不换（即本条的符号约定）。

先把 TP 与已见过的切法摆进同一张对照表，机制差异就清楚了：

| 方案 | 切什么 | 每卡参数 / 优化器态 | 通信发生在 | 每 micro-batch 通信触发次数 |
| --- | --- | --- | --- | --- |
| DP / ZeRO-1/2 | 数据维（三态复制或部分切片） | $16N$ / $4N + 12N/d$ / $2N + 14N/d$ | step 边界的梯度归约 | 常数次（4.2） |
| ZeRO-3 / FSDP | 三态全切片 | $16N/d$ | 每层参数取回（前向 + 反向） | $2L$ 次 all-gather + $L$ 次 reduce-scatter |
| TP | 层内宽度维 | $16N/t$（优化器态随分片自然除 $t$） | 层内的激活归约 | 每层 4 次 all-reduce（§6） |

TP 与 ZeRO-3 给出的数字相同——每卡只留 $1/\text{度数}$ 的参数——但通信的落点完全不同。ZeRO-3 搬的是**参数**：每层前向反向各做一次 all-gather，用完即弃。TP 搬的是**激活**：归约发生在层内，参数分片全程不动。这个差别决定了两者的度数经济学。ZeRO-3 的通信量正比于参数量 $N$、与度数无关，可以开到 $d = 1024$；TP 的通信消息量正比于激活 $bsh$、发生在每一层的计算中间，度数一大就顶到计算前面（推导见 §6-7）。TP 因此是「节点内的并行」：度数上限由节点内互联带宽决定。

还有一层动机只在训练侧成立。TP 的每卡 $16N/t$ 里，**优化器状态与梯度不需要任何额外机制就除了 $t$**：每卡的参数分片是物理上不同的张量，梯度天然只落在本分片上，AdamW 的 $m, v$ 也只按分片实例化。ZeRO-3 做不到这一点——参数全量只是「暂态取回」，梯度还要 reduce-scatter 归并到 owner。TP 用代码侵入（改模型定义）换来了这层干净。FLOPs 侧也有一句要立：TP 不引入任何冗余计算，每卡算的是同一层的不同乘加分片，集群总 FLOPs 与单卡恒等。TP 的全部开销都在通信一侧（口径：「开销」指通信字节与其暴露时间——FLOPs 已在上一句核过是零增量）；这与 PP 的气泡（5.3）是两个不同的代价科目。

## 2. 列切与行切：两条通信规则

TP 的全部规则从一个 matmul 出来。设 $Y = XA$， $X \in \mathbb{R}^{m \times k}$， $A \in \mathbb{R}^{k \times n}$（token 行堆叠口径， $m = b \cdot s$，即 1.2 §1 的折记号）。把 $A$ 分给 $t$ 张卡只有两种切法；通信落在什么位置，由 1.2 式 (2) 的 matmul 反向直接决定。

**列切（column parallel）**： $A$ 按列分块 $A = [A_1 \mid A_2 \mid \cdots \mid A_t]$，每卡持 $A_p \in \mathbb{R}^{k \times (n/t)}$，各算

$$
Y_p = X \thinspace A_p, \qquad Y = [Y_1 \mid Y_2 \mid \cdots \mid Y_t] \qquad (1)
$$

**符号**： $Y_p \in \mathbb{R}^{m \times (n/t)}$：rank $p$ 的输出分片；完整输出 $Y$ 是各分片沿列的拼接。前向无通信。若下游消费者（如逐元素激活）不需要全量 $Y$，拼接可以永不物化——§4 正是这个情形。反向由 1.2 式 (2) 给出：设 $G = \partial L / \partial Y$，其列分片 $G_p$ 即 rank $p$ 本地输出的上游梯度。 $G_p$ 本地可得—— $Y_p$ 的下游在分片内闭合时，它已经在本地。于是

$$
\frac{\partial L}{\partial A_p} = X^{\top} G_p \hspace{5mu} \text{（本地）}; \qquad \frac{\partial L}{\partial X} = G A^{\top} = \sum_{p=1}^{t} G_p \thinspace A_p^{\top} \qquad (2)
$$

**符号**：权重梯度逐卡本地；输入梯度是 $t$ 个部分和 $\sum_p G_p A_p^{\top}$。每卡能算出自己那一项，但全量需要跨卡求和——这就是**反向一次 all-reduce**（消息量 $= X$ 的梯度尺寸 $2mk$ bytes）。

**行切（row parallel）**： $A$ 按行分块 $A^{\top} = [A_1^{\top} \mid \cdots \mid A_t^{\top}]$（ $A_p \in \mathbb{R}^{(k/t) \times n}$ ），输入 $X$ 相应沿收缩维切列 $X = [X_1 \mid \cdots \mid X_t]$（ $X_p \in \mathbb{R}^{m \times (k/t)}$ ），则

$$
Y = XA = \sum_{p=1}^{t} X_p \thinspace A_p \qquad (3)
$$

每卡算出的 $X_p A_p$ 是一个 $[m, n]$ 的部分和，完整输出是跨卡求和——**前向一次 all-reduce**。反向给定 $G = \partial L / \partial Y$。注意 all-reduce 之后各卡持有的 $G$ 相同（归约语义保证）：

$$
\frac{\partial L}{\partial X_p} = G \thinspace A_p^{\top} \hspace{5mu} \text{（本地）}; \qquad \frac{\partial L}{\partial A_p} = X_p^{\top} \thinspace G \hspace{5mu} \text{（本地）} \qquad (4)
$$

两个梯度都只用本地分片与全量 $G$，无通信。对照表：

| 切法 | 分片 | 前向 | 反向 | 通信落点 |
| --- | --- | --- | --- | --- |
| 列切 | $A_p$： $[k, n/t]$ | $Y_p = X A_p$，无通信，输出天然分片 | $\partial L/\partial X = \sum_p G_p A_p^{\top}$ | 反向 1 次 all-reduce |
| 行切 | $A_p$： $[k/t, n]$， $X_p$ 同步切 | $Y = \sum_p X_p A_p$：部分和求和 | $\partial L/\partial X_p$、 $\partial L/\partial A_p$ 本地 | 前向 1 次 all-reduce |

$t = 2$ 的 shape 走读把两条规则走成直觉（AR = all-reduce）：

```
列切：A [k, n] ─► A0 [k, n/2] │ A1 [k, n/2]；X [m, k] 两卡全量复制
  rank0: Y0 = X A0  [m, n/2]        rank1: Y1 = X A1  [m, n/2]
  前向：无通信，Y = [Y0 | Y1] 只在逻辑上存在（下游逐位置 ⇒ 永不拼接）
  反向：dX = G0·A0ᵀ + G1·A1ᵀ —— 两项各在本地，求和 = 1 次 AR（[m, k] 消息）
行切：A [k, n] ─► A0/A1 [k/2, n]；X 沿收缩维同步切 X0/X1 [m, k/2]
  rank0: X0·A0（部分和 [m, n]）     rank1: X1·A1（部分和 [m, n]）
  前向：Y = X0A0 + X1A1 = 1 次 AR（[m, n] 消息）
  反向：dX0 = G·A0ᵀ、dA0 = X0ᵀ·G —— 全本地，无通信
```

走读里有一个反复出现的模式：无论哪种切法， $X$（列切的输入）或 $G$（行切的下游梯度）总有一方要在各卡复制。原因是结构性的——TP 只切参数与由参数定义的中间维，层的「边界张量」（块输入 $[m, k]$ 与块输出 $[m, n]$ ）始终是完整的。这条观察会在 §4 末与 §9 两次兑现为显存账上的冗余项。

还有一点要立住：行切的前向通信与列切的反向通信，传的都**不是**参数或权重梯度，而是**激活或其梯度**—— $X_p A_p$ 部分和、 $\partial L / \partial X$ 部分和，尺寸都是 $[m, n]$ 或 $[m, k]$ 的逐 token 张量。这就是 §1 表里「TP 通信搬激活」的出处。Megatron 的 MLP / attention 切法只有一招：把一个列切层与一个行切层首尾相接，让列切的输出分片恰好是行切的输入分片，中间永不拼接（§4-5 展开）。两次通信一次在反向（列切）、一次在前向（行切）；下一节把它们封装成一对可复用的算子。

## 3. f / g 共轭恒等算子【关键推导】

上表的两个通信落点位置固定、语义互补，Megatron 把它们封成一对算子（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]，§3；论文原话称 f 与 g 互为 conjugate）：

$$
f: \quad \text{前向 } f(x) = x \text{（恒等）}, \qquad \text{反向 } \frac{\partial L}{\partial x} = \mathrm{AllReduce}\left( \frac{\partial L}{\partial f} \right) \qquad (5)
$$

$$
g: \quad \text{前向 } g(x) = \mathrm{AllReduce}(x), \qquad \text{反向 } \frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \text{（恒等）} \qquad (6)
$$

**符号**： $\mathrm{AllReduce}$：TP 组内逐元素求和（4.1 §2 的集合通信；求和语义，DP 侧的「平均」归 DDP 管，见本节末注记）。f 贴在列切层的**输入**前，g 贴在行切层的**输出**后。两个方向各验证一遍这对定义确实是 §2 的通信规则、且前向语义与单卡恒等。

把这对定义立成一个可判定的论断。**命题**（f/g 恒等）：接到「列切 + 逐位置 + 行切」链上的 f/g，前向与单卡逐项相等，反向与「单卡反向、再按分片切走」逐项一致。**证明路线**：前向验数值、反向沿链式法则验通信点；关键一步是列切输入梯度的部分和恰好落在 f 的 backward 上。分两个方向各验一遍。

**前向等价**。f 前向是恒等，所以列切层看到的是原封不动的 $X$；g 前向是 all-reduce，所以行切层的部分和被求和成完整输出。拼起来看， $g(\sum_p \varphi(f(X)\thinspace A_p)\thinspace B_p)$ 一类的复合在前向与单卡 $Y = \varphi(XA)\thinspace B$ 逐项相等（ $\varphi$ 为逐位置算子的情形见 §4）。也就是说，f/g 不改变前向函数，只改变中间量的存放方式。

**反向等价**。沿链式法则逐节点走一遍。g 的反向是恒等，所以行切层拿到的 $G = \partial L / \partial Y$ 未经改动，式 (4) 的两个本地梯度成立。列切层的输入梯度按式 (2) 是部分和 $\sum_p G_p A_p^{\top}$；f 的反向恰好对上游梯度做一次 all-reduce，输出的正是完整 $\partial L / \partial X$。整条链上没有第二个通信点：行切的反向免通信（式 (4)），列切的反向通信全部由 f 一处完成。反向结果与「单卡反向、再把梯度按分片切走」逐项一致——是恒等，不是近似。

autograd Function 的直接写法（教学用，非优化实现；自定义 Function 的 forward / backward 语义归 1.1）：

```python
import torch
import torch.distributed as dist

class CopyToTPRegion(torch.autograd.Function):
    """教学用，非优化实现。f：前向恒等 / 反向 all-reduce（式 (5)）。"""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x                       # 前向恒等；save 语义：无需 save 任何量

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.contiguous()
        dist.all_reduce(grad_in, group=ctx.group)   # 反向：输入梯度的部分和归约
        return grad_in, None

class ReduceFromTPRegion(torch.autograd.Function):
    """教学用，非优化实现。g：前向 all-reduce / 反向恒等（式 (6)）。"""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.contiguous()
        dist.all_reduce(out, group=ctx.group)       # 前向：部分和求和成完整输出
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None          # 反向恒等：归约后的 G 各卡一致（式 (4) 的前提）
```

官方实现的逐行对照在 NVIDIA/Megatron-LM repo `megatron/core/tensor_parallel/mappings.py`。 `_CopyToModelParallelRegion` 的 forward 返回输入本身、backward 调 `_reduce` 做 `torch.distributed.all_reduce`——这就是式 (5)。 `_ReduceFromModelParallelRegion` 的 forward 调 `_reduce`、backward 原样回传——这就是式 (6)。两者与上面的教学版逐行同构（文件与类名核实于 2026-11-16；`layers.py` 侧的调用点走读归 6.1）。对外的公开包装是 `copy_to_tensor_model_parallel_region` / `reduce_from_tensor_model_parallel_region` 两个函数。同文件还有对称的另一对：`_ScatterToModelParallelRegion`（前向沿末维切、反向 all-gather）与 `_GatherFromModelParallelRegion`（前向 all-gather、反向沿末维切）。它们是把「恒等 / all-reduce」换成「切 / 拼」得到的变体，用于 embedding 与可选的输出拼接路径；5.2 会看到它们沿序列维的 SP 版本。

三处实现注记。其一，归约语义是**求和**。TP 组内各卡算的是同一批数据的不同分片，部分和该加不该平均；「除以度数」是 DDP 梯度平均（4.1 §3.2）的事，与 f/g 无关，两套归约语义不可混淆。其二，f 的前向恒等意味着反向的 all-reduce 必须对**全部** TP rank 同步发起。各卡反向到达 f 的时刻由本地计算决定，所以 f/g 把 TP 组的反向锁成同步步进；任何一卡的 straggler 都会直接顶到 all-reduce 的延迟项上（4.1 式 (3) 的 $2(t-1)\alpha$ 项）。这是 TP 组要小、要同构的另一条理由。其三，**RNG 同步注记**：dropout 等随机算子的随机性在 TP 组内必须分区管理。TP 区段内各卡分片上的 dropout 用各卡独立的随机流（分片内容本就不同）；复制区段（g 之后、残差与 norm 一带）的 dropout 则必须跨 TP 组共享同一张 mask，否则各卡前向不再一致、g 的归约语义被破坏。Megatron 的对应设计是把随机状态分成「TP 组内共享」与「按分片独立」两组管理（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]，附录 B.2 口径）。

## 4. MLP 块的 TP：列切接行切，中间不拼接

先看 GPT 式 MLP 块： $Y = \varphi(X W_1)\thinspace W_2$， $W_1 \in \mathbb{R}^{h \times d_{\mathrm{ff}}}$， $W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times h}$， $\varphi$ 为逐位置激活（GeLU 或 SwiGLU 一族）。切法（Megatron 论文 §3.1）是两刀错开： $W_1$ 列切（每卡持 $[h, d_{\mathrm{ff}}/t]$ ）， $W_2$ 行切（每卡持 $[d_{\mathrm{ff}}/t, h]$ ）。中间维 $d_{\mathrm{ff}}$ 是两个切口的公共轴——对前者是拼接轴，对后者是收缩轴：

$$
X \hspace{5mu}\xrightarrow{\hspace{5mu} f \hspace{5mu}}\hspace{5mu} X \hspace{5mu}\xrightarrow{\hspace{5mu} W_1^{(p)} \hspace{5mu}}\hspace{5mu} H_p = X W_1^{(p)} \hspace{5mu}\xrightarrow{\hspace{5mu} \varphi \hspace{5mu}}\hspace{5mu} \varphi(H_p) \hspace{5mu}\xrightarrow{\hspace{5mu} W_2^{(p)} \hspace{5mu}}\hspace{5mu} \varphi(H_p)\thinspace W_2^{(p)} \hspace{5mu}\xrightarrow{\hspace{5mu} g \hspace{5mu}}\hspace{5mu} Y \qquad (7)
$$

**符号**： $W_1^{(p)} \in \mathbb{R}^{h \times (d_{\mathrm{ff}}/t)}$、 $W_2^{(p)} \in \mathbb{R}^{(d_{\mathrm{ff}}/t) \times h}$：rank $p$ 的分片； $H_p \in \mathbb{R}^{m \times (d_{\mathrm{ff}}/t)}$：中间激活分片。为什么拼接可以跳过？靠的是 $\varphi$ 的逐位置性。 $\varphi$ 逐元素作用于 $H = [H_1 \mid \cdots \mid H_t]$ 的每个分量，而列分片之间互不重叠，所以 $\varphi(H)$ 的第 $p$ 段 $= \varphi(H_p)$——**逐位置算子在列切下免通信**，而且逐字精确，不是近似。这同时是一条限制：列切后面接的只能是逐位置算子；换成任何跨分量耦合的算子（比如对 $d_{\mathrm{ff}}$ 维求归一化的 softmax），免通信立即失效。这个约束在 5.2 会从反方向回来：RMSNorm 沿 $h$ 耦合，所以不能留在 TP 区段内。

反向沿同一条链走回去，每个节点的通信属性已在 §2-3 确定：g 反向恒等； $W_2$ 行切反向免通信（式 (4)）； $\varphi$ 逐元素反向本地； $W_1$ 列切的输入梯度是部分和，由 f 的反向 all-reduce 归并。全块通信量：前向 1 次 all-reduce（g）、反向 1 次（f），各搬一个 $[m, h]$ 消息。参数账：每卡 $(h \cdot d_{\mathrm{ff}} + d_{\mathrm{ff}} \cdot h)/t = 2h \thinspace d_{\mathrm{ff}}/t$， $d_{\mathrm{ff}} = 4h$ 时即 $8h^{2}/t$——连同梯度与 AdamW 两阶矩，优化器三态同步除 $t$。代入 8B 几何（hidden 4096、度数 8）：MLP 两块每卡约 1700 万个参数，恰是未切时的八分之一。SwiGLU 三矩阵版本（1.2 §6）同理： $W_1, W_3$ 同取列切（门与值两支各持分片， $\mathrm{SiLU}(a_p) \odot b_p$ 仍逐位置）， $W_2$ 行切，通信次数不变。

> **坑**：把列切层的输入当成「已切分」是常见误读。触发处是 $H_p$ 这步 matmul——它的输入 $X$ 在每卡都是**全量**，因为 f 是恒等、不是切分；后果是把激活账算小。列切省的是参数与输出分片的显存；输入激活 $X$ 在 TP 组内是复制的。这份复制正是 1.4 激活账里「TP 后仍冗余的 10 项」的一部分来源，也是 5.2 的 SP 要消掉的东西。

## 5. attention 块的 TP：按 head 切

attention 块的两个线性层落在同一条规则上，中间的 attention core 靠另一条独立性免通信。记号用 MHA 口径： $h = a \cdot d_{\mathrm{head}}$， $d_{\mathrm{head}}$ 为 head 维。head $j$ 的输出

$$
O_j = \mathrm{softmax}\left( \frac{Q_j K_j^{\top}}{\sqrt{d_{\mathrm{head}}}} \right) V_j \qquad (8)
$$

**符号**： $Q_j, K_j, V_j \in \mathbb{R}^{m \times d_{\mathrm{head}}}$：head $j$ 的投影结果。softmax 的归一化沿 key 位置维收缩，封闭在单个 head 内。所以 head $j$ 的输出只依赖 $Q_j, K_j, V_j$，不触碰任何其他 head 的中间量。这意味着「按 head 分组到各卡」对 attention core 是精确切分：每卡持 $a/t$ 个 head 的 $Q, K, V$，本地跑自己的 FlashAttention（机制归 models 4.1，kernel 归 gpu 6.3），全程无通信。

两端的线性层按 §2 落位。QKV 投影 $W_{QKV} \in \mathbb{R}^{h \times 3h}$ 沿输出维**列切**，切口对齐 head 边界：每卡 $[h, 3h/t]$，恰好 $a/t$ 个 head 的 Q/K/V，前面贴 f。out 投影 $W_O \in \mathbb{R}^{h \times h}$ 沿输入维（head 拼接轴）**行切**：每卡 $[h/t, h]$，输入恰是本卡 $a/t$ 个 head 的输出拼接段，后面贴 g。整块的通信与 MLP 块逐项同构：前向 1 次 all-reduce（g）、反向 1 次（f），消息同为 $[m, h]$。参数账： $W_{QKV}$ 与 $W_O$ 各 $h^{2}$ 量级（MHA），每卡合计约 $4h^{2}/t$；与 MLP 的 $8h^{2}/t$ 相加，单层参数每卡约 $12h^{2}/t$。沿用上节的 8B 几何：attention 两块每卡再添约 800 万个参数，单层合计约 2500 万个。

还有一个拼接对偶值得点破。attention core 的输出按 head 分段躺在各卡，恰好就是 out proj 行切所需的输入分片——head 拼接轴同时是 $W_O$ 的收缩轴。「列切输出分片 = 行切输入分片」这一对齐，与 §4 的 MLP 完全相同。

> **提点**：工程上 $W_Q, W_K, W_V$ 三个矩阵常合并成一个 $[h, 3h]$ 的 QKV 权重再做列切，切口仍按 head 对齐；切出来的每卡分片是三个子块而非一整段。这不改变任何账，只是实现层的内存布局选择。

这条切法把 TP 度数的第一条硬约束推出来了：head 是切分的最小单元， $t$ 必须整除 $a$（GQA 的情形见 §8）。RoPE 之类的逐位置、逐 head 内变换不影响结论——它们作用在各 head 自己的 $Q_j, K_j$ 上。跨位置耦合的只有 softmax 的 key 维收缩，而 key 维在单 head 内是完整的。这就是 attention core 免通信、而沿 $h$ 收缩的 RMSNorm 必须留在 TP 区段之外的原因（与 §4 末是同一个论证）。

## 6. 每层通信位置图与通信账【关键推导】

把两个块拼成一层，f/g 的位置图（每卡视角，f = 前向恒等 / 反向 AR，g = 前向 AR / 反向恒等，AR = all-reduce）：

```
        复制区段（各卡全量：残差、RMSNorm、dropout —— 5.2 的 SP 对象）
            │
 x [b,s,h] ─┼─ f ─► QKV 列切 [h, 3h/t] ─► 逐 head attention ─► out proj 行切 [h/t, h] ─► g ─► (+残差)
            │      ▲ 反向 AR                    免通信（按 head）            ▲ 前向 AR
            ├─ f ─► W1 列切 [h, d_ff/t] ─► φ 逐位置（免通信）─► W2 行切 [dff/t, h] ─► g ─► (+残差)
            │
     前向通信：g × 2；反向通信：f × 2 —— 每层 4 次 all-reduce，消息同为 [b, s, h]
```

**通信账**的路线：先定单次消息量，再数每层次数，最后乘 ring 系数与层数。逐项立起来。被归约的张量是完整 hidden 激活 $[b, s, h]$，bf16 下消息量

$$
M_{\mathrm{msg}} = 2 \thinspace b \thinspace s \thinspace h \quad \text{bytes} \qquad (9)
$$

**符号**： $2$：bf16 字节数。每层前向 2 次 all-reduce（attention 块的 g 加 MLP 块的 g）、反向 2 次（两处的 f）。于是一个 micro-batch 的消息量口径是：前向每层 $4bsh$ bytes，反向对称再 $4bsh$，单层单 micro-batch 合计 $8bsh$ bytes； $L$ 层逐层叠加得全模型 $8bshL$（外加 embedding 的一次，见 §9）。换算成每卡收发字节数，乘 4.1 式 (2) 的 ring 系数 $2(t-1)/t$（发送口径，接收同量）：

$$
C_{\mathrm{rank}} = 8 \thinspace b \thinspace s \thinspace h \thinspace L \cdot 2\thinspace \frac{t-1}{t} \quad \xrightarrow{\hspace{5mu} t \to \infty \hspace{5mu}} \quad 16 \thinspace b \thinspace s \thinspace h \thinspace L \quad \text{bytes / micro-batch} \qquad (10)
$$

两个读数。其一，每卡通信量**与 $t$ 几乎无关**——系数从 $t = 2$ 的 1 饱和到 2。TP 加卡不省通信，省的是每卡计算。其二，消息的全部体量由激活 $bsh$ 与层数决定，与参数量 $N$ 无直接关系。这与 DP 侧「通信 $\approx 4N$ 与 $d$ 无关」（4.4 §4）是同构的另一极。

代入真实标尺算一遍。Llama-3 405B 的几何： $L = 126$、 $h = 16384$、 $a = 128$（herd 论文 Table 3）；训练配置 TP8 × CP1 × PP16 × DP64（herd 论文 Table 4，[Grattafiori et al., 2024](../references.md)，[A]）。取单卡视角的一个 micro-batch（ $b = 1$、 $s = 8192$ ）： $M_{\mathrm{msg}} = 2 \times 8192 \times 16384 \approx 268$ MB。PP16 下每卡约 8 层，式 (10) 给出每 micro-batch 每卡 TP 通信 $\approx 1.88$ GB/层 $\times 8 \approx 15$ GB。再看时间轴。该 micro-batch 的计算量是 $6 \times (405 \times 10^{9} / 128) \times 8192 \approx 1.56 \times 10^{14}$ FLOPs，按 herd Table 4 的 430 TFLOPS/GPU 折算约 360 ms。通信的带宽下界：节点内 NVLink 域约 900 GB/s（H100 公开规格），算出约 17 ms——占计算的 5% 以下，落在可藏区间。同一份字节搬到跨节点 IB（每卡 50 GB/s，4.4 §4 的 OLMo 2 集群口径）要约 300 ms，与计算同量级，再也藏不住。**TP ≤ 8、留在节点内 NVLink 域内**，就是这个除法的结果，不是经验习惯。

为什么这个比值随 $t$ 线性恶化？每卡每层的计算量随分片反比缩，为 $C_1 / t$（ $C_1$ 为单卡单层 FLOPs）；而每卡每层通信字节数 $16 bsh (t-1)/t$ 饱和到常数 $16bsh$——

$$
\frac{\text{通信}}{\text{计算}} = \frac{16 \thinspace b \thinspace s \thinspace h \thinspace (t-1)/t}{C_1 / t} = \frac{16 \thinspace b \thinspace s \thinspace h}{C_1} \thinspace (t - 1) \qquad (11)
$$

通信-计算比 $\propto (t-1)$：度数翻一倍，通信相对计算的权重就近似翻一倍；再叠加 4.1 式 (3) 的延迟项 $2(t-1)\alpha$ 同步线性涨。这就是 TP 的扩展曲线总是在 $t = 8$（一个 NVLink 节点的卡数）附近停下的原因——再往上，通信先吃掉 NVLink 的余量，然后撞上 IB 的量级差。

与 ZeRO-3 的对照把账本合上。两者都把每卡三态降到 $1/\text{度数}$：ZeRO-3 付 $3N$ 元素/step 的参数与梯度通信（4.2 式 (3)），TP 付式 (10) 的激活通信。差别在时序上——TP 的归约卡在两段 GEMM 之间的关键路径上，难 overlap；ZeRO-3 的通信在层边界，可与计算重叠。这笔交换决定了 TP 的正确用法：小度数、节点内、吃最快的互联。跨家族的摆位组合归 5.5。

**与 gradient accumulation 的交互：TP 通信不摊薄**。1.5 / 4.1 的 `no_sync` 能把 step 边界的梯度归约摊薄到 $1/m$，但同样的招在 TP 这里不成立——4.2 §4.4 的注记恰好成立在反面。式 (10) 的激活 all-reduce 发生在**每个 micro-batch 的每一层中间**， $m$ 个 micro-batch 恒付 $m$ 份，没有任何机制能把它们推迟到累加边界合并。这是「层内通信」与「step 边界通信」的结构性差别，与 ZeRO-3 参数取回不摊薄（4.2 §4.4）同构。字节数降不下来，工程能动的就只剩暴露比例：Megatron 把反向 f 的 all-reduce 挂成异步、与同层权重梯度 GEMM 重叠（`layers.py` 的 `LinearWithGradAccumulationAndAsyncCommunication`，走读归 6.1），前向 g 的归约则按输出分块与 GEMM 流水化。这些手段不改变式 (10) 的字节账，只决定其中多少落在关键路径上。

## 7. 度数上限（一）：整除约束

通信给了 $t$ 的软上限，整除给了硬约束清单：

| 被切对象 | 约束 | 出处 |
| --- | --- | --- |
| QKV / out proj | $t$ 整除 head 数 $a$（GQA 见 §8） | §5 |
| MLP $W_1$ / $W_2$ | $t$ 整除 $d_{\mathrm{ff}}$ | §4 |
| embedding / 输出头 | $t$ 整除词表 $V$（否则 padding 补齐） | §9 |

代实例看一遍。Llama-3.1-8B： $a = 32$、GQA 8 KV head（herd 论文口径）—— $t \in \lbrace 1, 2, 4, 8 \rbrace$ 时 Q 与 KV 两侧都整除， $t = 16$ 起 KV 侧破缺（§8 展开）。405B： $a = 128$， $t = 8$ 很富裕。GPT-3： $a = 96$，支持到 $t = 32$。这些约束全是结构性事实，没有调参空间。

Megatron 论文的规模实测给了这套约束下的可行性锚点：8.3B 模型、8 路 TP × 64 路 DP、512 张 V100，全程 15.1 PFLOPS、76% 扩展效率（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]，摘要口径；512 卡 = 32 台 DGX-2H，8 路 TP 组收在单台 16 卡的 NVSwitch 全互连域内，每机装两个 TP 组）。论文的设计论述同时明确：model parallel 组收在节点内，跨节点交给 data parallel。也就是说，§6 的账在 2019 年的硬件上（节点内 NVLink 对节点间 IB 的量级差）已经成立。

## 8. GQA / MLA 下的 TP

**GQA：KV head 数成为新约束**。GQA 把 KV head 数压到 $a_{kv} \lt a$（ $g = a / a_{kv}$ 个 Q head 共享一组 KV，结构推导归 models 4.1）。TP 按 head 切分时，整除约束从 $a$ 一处收紧到两侧：Q 侧切 $a/t$，KV 侧切 $a_{kv}/t$。当 $a_{kv} \lt t$（比如 8 KV head 对 $t = 16$ ），KV 侧无 head 可切，工程上只剩两条路。第一条是**复制**：每个 rank 持完整的 KV 投影权重与 KV 激活，参数与激活账的对应项不再除 $t$；Q 侧照常分片，attention core 在每卡用自己的 Q head 与共享 KV 算。第二条是**通信**：需要时把 KV all-gather 进各卡。主流实现取前者——Megatron 的 GQA 支持按「每 rank 至少一个 KV head」处理，不足则复制；后者把 KV 通信引进 attention core 内部，破掉 §5 的免通信结构。复制的代价有界：KV 投影参数占比 $2 \thinspace a_{kv} \thinspace d_{\mathrm{head}} \thinspace h / (4h^{2}) = a_{kv} / (2a)$，8B 几何下为 1/8，复制 16 份也只把 attention 块的参数账抬回约 1/8 的权重——但度数的整齐性没了。激活侧同理：每卡要为全部 $a_{kv}$ 个 KV head 存 $K, V$ 激活（而非 $a_{kv}/t$ 个），TP 对 attention core 激活的除法在 KV 侧失效、只在 Q 侧成立。这不该意外：GQA 本来就是为压 KV 侧而生的结构（models 4.1），TP 在它身上能再切的东西，按设计就已经不多了。

**MLA：低秩结构让「按 head 切」的覆盖面进一步缩小**。MLA（结构推导与维度表归 models 4.1 / models 5.2，此处只取结构事实）没有逐 head 的独立 K/V：全部 head 共享一个低秩 latent（V3 config 口径 $d_c = 512$，远小于 $h = 7168$ ），外加 decoupled RoPE 分量；各 head 的 K/V 经共享 latent 的 up-projection（或吸收进 Q 侧权重）重建。对 TP 的直接后果是：latent 投影（ $h \to d_c$ 一带）被全部 head 使用，必须在每个 rank 复制——它不是「某个 head 的参数」，无处可切。可按 head 切的只剩 Q 侧与 per-head 的吸收矩阵。所以 attention 块内 TP 能除 $t$ 的参数占比比 MHA/GQA 更低，而被复制的 latent 激活还随 $s$ 线性涨。

**DeepSeek-V3 不用 TP**。论文的训练并行配置是 16 路 PP、64 路 EP、ZeRO-1 DP，明确不用 TP，原文措辞 "without using costly tensor parallelism"（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，训练框架一节的口径）。论文把这一选择放在显存工程的语境里：MLA 把 KV 侧显存压下去，选择性重算压激活（1.4 的机制），EP 把专家参数（V3 的参数大头在 671B 中的 expert 侧）切到 64 路。非专家参数与激活在单卡预算内已可安置，TP 的逐层激活 all-reduce（式 (10) 的量级）就没有可换的对象了。以上是论文口径的转述。再进一步，把上两段（MLA 低秩结构对 TP 收益的压缩）与论文口径接起来读：「V3 不用 TP」不是孤立的工程口味，而是三条的合流——MLA 的 attention 侧可切份额小、参数大头已被 EP 切走、通信成本按式 (11) 随 $t$ 线性涨。这段合流读法是本书补充分析，非论文原文论断。

## 9. 显存账与遗留的冗余

TP 的三态账很干净：参数、梯度、AdamW 两阶矩全部按物理分片除 $t$，每卡 $16N/t$ bytes（参考配置口径，models 1.2 式 (8) 的 $16N$ 分解）。比 ZeRO-3 还省一步：没有暂态取回、没有梯度归并，分片是常量。

激活账不干净，这正是 5.2 的入口。TP 区段内（QKV / attention / out proj、MLP 两线性加激活函数）的中间激活按分片除 $t$。但区段之间的**复制区段**不一样：残差流、RMSNorm、dropout 的输入输出是完整 $[b, s, h]$ 张量，在每卡各存一份。1.4 的逐算子账把这件事量化成了系数（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]）：无并行口径每层 $sbh(34 + 5as/h)$ bytes，TP 后成为 $sbh(10 + 24/t + 5as/(ht))$。拆开看： $24/t$ 与 $5as/(ht)$ 两项被 TP 切走； $10$ 那一份——两个 norm 的输入 $4sbh$、三处 dropout 掩码与残差侧的逐位置张量——冗余地留在每张卡上。消掉这 $10$ 的办法是把复制区段沿序列维切给各卡，这就是 sequence parallelism；5.2 的通信恒等推导会证明，这次搬家通信量一分不涨。

输入输出两端各一句。embedding 查表层按词表维切： `VocabParallelEmbedding` 让各卡持 $V/t$ 行，命中区间外的 token 本地查零、输出置零，再做一次 all-reduce 求和出完整 embedding——掩码置零加全归约的组合与式 (3) 的行切同构（repo `megatron/core/tensor_parallel/layers.py`，核实于 2026-11-16）。这一步前向多 1 次 $[b, s, h]$ 的 all-reduce，反向走它的转置。输出头与 embedding tied 时，logit 沿词表维列切、各卡只持 $V/t$ 列，CE 在词表分片上分布式完成（与 1.2 §10 的 chunked CE 同族：logits 永不全量物化）。词表整除进不了硬约束清单时用 padding 凑齐，开销一行可算。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 两条切分规则 | 列切：输出拼接、反向 all-reduce；行切：前向 all-reduce、反向免通信 | 式 (1)–(4)；1.2 式 (2) |
| f / g 共轭算子 | f = 前向恒等 / 反向 AR，g = 前向 AR / 反向恒等；前向函数与单卡恒等、反向梯度逐项一致 | 式 (5)(6)；`mappings.py` 的 `_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion` |
| MLP 块 | $W_1$ 列切 + 逐位置 $\varphi$ 免通信 + $W_2$ 行切；参数 $8h^{2}/t$ /卡 | 式 (7)；论文 §3.1 |
| attention 块 | 按 head 切（head 独立性免通信），QKV 列切 + out proj 行切；约 $4h^{2}/t$ /卡 | 式 (8)；论文 §3.2 |
| 通信账 | 消息量 $M_{\mathrm{msg}} = 2bsh$ bytes；每层前向 2 次 + 反向 2 次 all-reduce，单层 $8bsh$、全模型 $8bshL$；每卡再乘 $2(t-1)/t$ | 式 (9)(10)；4.1 式 (2) |
| NVLink 边界 | 通信-计算比 $\propto (t-1)$；405B 实例节点内约 17 ms 对 360 ms 计算、跨节点约 300 ms 不可用 ⇒ TP ≤ 8 节点内 | 式 (11)；herd Table 3/4 |
| 整除约束 | $t$ 整除 $a$、 $d_{\mathrm{ff}}$、（padding 后） $V$；8.3B @ 512 V100 实测 15.1 PFLOPS / 76%（摘要口径） | §7 |
| GQA / MLA | $a_{kv} \lt t$ 时 KV 复制；MLA latent 投影无处可切；V3 不用 TP 的论文口径是显存工程，MLA 收益分析为本书补充 | §8 |
| 显存 | 三态 $16N/t$；激活块内除 $t$、复制区段冗余 10 项（ $sbh(10 + 24/t + 5as/(ht))$ ）→ SP 归 5.2 | 1.4；arXiv:2205.05198 |

## 练习

1.（推导 · [20]）从 1.2 式 (2) 出发完成 §2 的反向：对列切证明 $\partial L/\partial X = \sum_p G_p A_p^{\top}$ 的每一项只依赖本卡分片，对行切证明式 (4) 的两个梯度无通信；再证明 MLP 块（式 (7) 链）每 micro-batch 的 TP 通信恰为前向 1 次 + 反向 1 次 all-reduce，不能更少（提示：输入梯度部分和的归约不可免，g 的前向归约不可免）。本题即本章两条切分规则与 f/g 恒等命题的证明闭环。
2.（计算 · [20]）取 $h = 8192$、 $s = 8192$、 $b = 1$、 $L = 80$、无 PP：(a) 按式 (10) 算 $t = 8$ 时每 micro-batch 每卡 TP 通信字节数（GB）与 900 GB/s 下的时间下界；(b) 同模型单卡计算时间按 $6 \cdot (N/t) \cdot b s$ FLOPs、有效吞吐 400 TFLOPS 折算，求通信-计算比并与式 (11) 对照；(c) 把 $t$ 改为 16 重算两数，说明为什么这一步必须跨节点时整档不成立。先按通信-计算比公式预测 (c) 比 (b) 大几倍，再动手算。校验：(a) 的每卡字节数应与正文 405B 实例同阶（每 micro-batch 每卡十几 GB）。
3.（推导 · [30]）逐位置算子免通信的边界：设列切后接的是对 $h$ 维做归一化的 LayerNorm（而非逐元素的 $\varphi$ ），证明此时各卡无法本地完成归一化，给出所需通信算子（沿 $h$ 的归约统计量）与消息量级；由此解释 Megatron 为什么把 norm 放在 TP 区段之外的复制区段执行（对照 §9 的冗余 10 项与 5.2 的动机）。
4.（代码 · [30]）教学用，非优化实现：用 §3 的 `CopyToTPRegion` / `ReduceFromTPRegion` 与两个手写分片线性层拼一个 $t = 2$ 的 TP MLP（单进程可用 group=None 退化验证语义，双进程 gloo 验证通信），与同参数单卡 MLP 对 loss、输入梯度与权重梯度做数值对照；再把列切层后接一个 LayerNorm，观察数值分叉，对照练习 3 的结论。先写下预测——接上 LayerNorm 后哪一项先分叉、loss 是否仍一致——再跑。校验：fp64 对拍， `torch.allclose` 容差 1e-10。

## 参考

- [Shoeybi et al., 2019](../references.md)：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，arXiv:1909.08053，[A]（key: megatron-lm）——列切/行切规则与 f/g 共轭算子（§3）、MLP 与 attention 块切法（§3.1-3.2）、8.3B @ 512 V100 的 15.1 PFLOPS / 76% 扩展效率（摘要）、TP 收在节点内的设计论述；实现对照 NVIDIA/Megatron-LM repo `megatron/core/tensor_parallel/mappings.py`（`_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion` / scatter-gather 对称对）与 `layers.py`（`ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding`），路径与类名核实于 2026-11-16
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——PP16 × EP64 × ZeRO-1、不用 TP（"without using costly tensor parallelism"）的训练框架口径；MLA 结构链 models 4.1 / 5.2
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——405B 几何（Table 3）与 TP8 × CP1 × PP16 × DP64 配置、430 TFLOPS/GPU（Table 4）；GQA 8 KV head 口径
- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——激活逐算子账与 TP 后冗余结构（ $10 + 24/t + 5as/(ht)$ ），SP 推导归 5.2
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照（key: pytorch-repo）——`torch.autograd.Function` 的 forward/backward 语义（1.1）与 `torch.distributed.all_reduce`
