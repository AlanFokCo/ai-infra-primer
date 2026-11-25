# 4.1 DP → DDP：数据并行的数学与实现（Data Parallelism and DistributedDataParallel）

> 数据并行只做两件事： $d$ 个模型副本各自对不同的数据算梯度，再把梯度跨副本平均。本章先回答这个平均在数学上等于什么——答案是大 batch 梯度，与 1.5 的累加是同一条线性叠加；再回答它在系统上花多少通信——ring all-reduce 的账，每 rank $2(d-1)/d \cdot M$ bytes。随后走读 PyTorch DDP 的实现：bucket 化、反向与通信的 overlap、`no_sync` ；中途插一段同步与异步 SGD 的分界。最后落到一个事实上：DDP 不省显存，每个 rank 仍复制全量参数、梯度与优化器状态，这正是 4.2 ZeRO 的出发点 | 前置：1.1（autograd hook 与 `AccumulateGrad`）、1.5（累加等价与 `no_sync` 的 API 语义）、models 1.2（ $16N$ 三态账） | 锚点：PyTorch DDP 论文（[Li et al., 2020](../references.md)，arXiv:2006.15704，[A]）与实现 pytorch/pytorch repo `torch/nn/parallel/distributed.py`（教学参照 + 官方实现）；ZeRO 论文（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]，动机）；Megatron-LM（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]，DP/MP 分界对照）

本章口径：模型一律 7B bf16（梯度缓冲约 13 GiB），d 为 DP 度数；惟 DDP 实测段用论文的 ResNet-152 / 60M 参数口径，随文标注、不外推。

## 1. DP 的数学：平均梯度与大 batch 的统一

数据并行（data parallelism, DP）的规则只有一条。 $d$ 个进程（rank）各持一份完整模型副本，各自对不同的数据分片做前向反向，然后把梯度跨副本平均，各自在本地跑同一个 optimizer 更新。设第 $r$ 个 rank 分到 $b$ 条序列（分片 $S_r$ ）。这个平均梯度与大 batch 的梯度是什么关系？答案是一行线性代数：

$$
g_r = \frac{1}{b} \sum_{j \in S_r} \nabla \ell_j, \qquad \bar{g} = \frac{1}{d} \sum_{r=1}^{d} g_r = \nabla \left( \frac{1}{d \thinspace b} \sum_{j=1}^{d \thinspace b} \ell_j \right) = \nabla \bar\ell \qquad (1)
$$

**符号**： $d$ ：DP 度数（副本数，spec §4.3 约定）； $b$ ：每 rank 的 micro-batch 序列数； $\ell_j$ ：第 $j$ 条序列的标量 loss； $g_r$ ：rank $r$ 的本地平均梯度； $\bar{g}$ ：跨 rank 平均梯度。

**命题**（平均梯度等价）：各副本的平均梯度与大 batch 梯度逐分量相等。**证明**（三步）：第一步是分片内平均的定义，第二步把 $d \cdot b$ 条序列的和按分片分组，第三步用梯度的线性性——没有近似。代入整数看一眼：d 取 2、每 rank 各 2 条序列，逐样本梯度依次为 1、3 与 5、7；两 rank 的本地平均分别是 2 与 6，跨 rank 平均后各拿到 4，与 4 条序列一次大 batch 的结果（(1+3+5+7)/4 = 4）同值。

式 (1) 和 1.5 的累加等价是什么关系？同一个线性叠加，只是拆在两个轴上：accumulation 沿时间轴把一个 batch 拆成 $m$ 份顺序累加，DP 沿空间轴拆成 $d$ 份并行。两者相乘的合成形式就是 1.5 式 (6) 的 $g = (1/d) \sum_r (1/m) \sum_i \nabla \hat\ell_{r,i}$ 。1.5 的三条等价前提也逐条继承到每个分片：第一，等权，各分片序列数相等；第二，逐样本独立，不能有 BatchNorm 这类跨样本统计量；第三，token 口径，变长序列要按有效 token 加权，不能用等权的 mean-of-means。等效 batch 的口径沿用 1.5 式 (5)： $B_{\mathrm{tok}} = b \cdot m \cdot d \cdot T$ ，DP 往里乘的因子是 $d$ 。梯度噪声协方差按 $\Sigma / B_{\mathrm{tok}}$ 缩（推导归 2.1），LR 与 schedule 的联动归 2.3。本章只负责「平均」这一步的机制与账。

为什么归约目标必须是平均（或求和）？因为式 (1) 的等价依赖归约运算与大 batch 目标的语义匹配：大 batch 目标是样本平均，归约就必须是求和——平均只是求和乘 $1/d$ ，可以在归约前、中、后任一位置折入。换成任何非线性归约（max、median 之类），得到的量就不再对应任何 batch 的梯度。这条约束也划清了 DP 的边界：DP 复现的是「更大的 batch」，不是「更多的模型」。想改模型的梯度结构（比如 MoE 的 per-expert 梯度），得靠模型并行的切分（P5）。

同步与异步的分界在这里说清。**同步 DP**：各 rank 在同一参数版本上工作，同一时刻拿到同一份平均梯度。数学上，它与单机大 batch 逐步相同（浮点求和顺序不同带来的位级差异除外）——2.x 的全部优化器公式不用为 DP 改一个符号。**异步 SGD**（parameter server 路线）是另一套语义：各 worker 在陈旧参数版本上算梯度、随时提交，server 随到随更新。worker $i$ 算出的梯度对应的是若干步之前的参数；这个 staleness 使更新规则不再是式 (1) 的平均，收敛分析要另立假设。LLM 预训练的全部主流栈（DDP、ZeRO、FSDP、Megatron）都选同步语义——通信贵，但优化轨迹可复现、可与单机 recipe 直接对照。本章与 4.2 / 4.3 只讨论同步 DP；异步路线的账本书不展开。

## 2. all-reduce 的语义与通信账

### 2.1 语义与不合格基线

梯度平均在系统层是一个集合通信（collective）：**all-reduce**。它的定义： $d$ 个 rank 各持一份 $M$ 字节的向量（梯度缓冲，bf16 下 $M = 2N$ bytes， $N$ 为参数量）；操作结束后，每个 rank 都拿到逐元素归约的结果。最小例：d 取 2、两份梯度为 [1] 与 [3]，SUM 语义的 all-reduce 之后两边各得 [4]。DDP 的语义是先除 $d$ 再求和，即平均（§3.2 展开）。衡量一个 all-reduce 实现看两个量：每 rank 发送的字节数 $C$ （接收对称同量），以及消息步数（决定延迟项）。

先看一个不合格的对照基线，星型归约：所有 rank 把梯度发给 rank 0，rank 0 归约后再广播回去。问题在 rank 0 的链路：一步要承载 $2(d-1) \cdot M$ bytes，随 $d$ 线性膨胀。代入数字：7B 模型的 bf16 梯度 $M \approx 13$ GiB， $d = 512$ 时 rank 0 每步要收发各约 6.5 TiB。任何可用方案都必须让单 rank 的账与 $d$ 解耦。

### 2.2 ring all-reduce：通信量推导

ring 怎么把星型的账降下来？路线：把 all-reduce 拆成 scatter-reduce 与 all-gather 两个阶段，各用一条归纳不变式算出每 rank 的发送量；关键一步是切块——每步每 rank 只发 d 分之一块。ring all-reduce 把向量切成 $d$ 块、每块 $M/d$ bytes，分两个阶段，各走 $d - 1$ 步。以 $d = 4$ 为例（ $c_0, \ldots, c_3$ 为四个块）：

```
环结构（箭头 = 发送方向）：

        r0 ──→ r1
        ↑        │
        │        ▼
        r3 ←── r2

scatter-reduce（d−1 = 3 步，每步每 rank 发 M/d）：
  step1  r0→r1 发 c0 部分和   r1→r2 发 c1   r2→r3 发 c2   r3→r0 发 c3
         各 rank 把收到的块加进自己持有的同名块部分和
  step2  继续沿环转发上一步收到的部分和（c0 现在到 r2 …）
  step3 后：r1 持 c0 全和   r2 持 c1 全和   r3 持 c2 全和   r0 持 c3 全和

all-gather（3 步，每步每 rank 发 M/d）：各 rank 把自己的全和块沿环逐步转发
  3 步后：每个 rank 集齐 c0…c3 四个全和块 = 归约完成的完整向量
```

**命题**（scatter-reduce 不变式）：第 $s$ 步（ $s = 1, \ldots, d-1$ ）里，rank $r$ 把手上的部分和块发给 $r+1$ ，同时从 $r-1$ 收一块，加进本地同名块的部分和；第 $s$ 步结束时，每个块的部分和恰好累加了 $s+1$ 个 rank 的贡献。**证明**（对 s 归纳，三段）：基例，s 取 0 时每块只有本地一份贡献，空虚成立；保持，设第 s 步后成立，第 s+1 步每块的部分和被转发到下一 rank 并加上该 rank 的本地块，贡献数从 s+1 变为 s+2；终止， $s = d-1$ 时每块集齐全部 $d$ 份贡献，且每 rank 恰好持有一个全归约块。逐步展开留作练习 3(b)。这一阶段每 rank 累计发送 $(d-1) \cdot M/d$ bytes。

**all-gather 阶段**是对称操作：每 rank 把自己的全归约块沿环转发， $d-1$ 步后人人集齐。每 rank 再发 $(d-1) \cdot M/d$ bytes。

合计每 rank 发送（接收同量）：

$$
C_{\mathrm{rank}} = 2 \thinspace \frac{d-1}{d} \cdot M \hspace{5mu} \approx \hspace{5mu} 2M \qquad (2)
$$

**符号**： $C_{\mathrm{rank}}$ ：一次 all-reduce 每 rank 的发送字节数； $M$ ：被归约向量的总字节数。系数 $2(d-1)/d$ 从 $d = 2$ 时的 1 单调逼近 2，可以说**带宽账与 $d$ 基本无关**——这是 ring 能撑到万卡规模的核心性质（对照星型基线的 $2(d-1)M$ ）。用两个端点自检这个公式。 $d = 2$ 时式 (2) 给出 $M$ ，正是两 rank 互换全量向量的下界。 $d \to \infty$ 时收敛到 $2M$ ，即「每 rank 交出自己的一份、收回全体的一份」的不可约下界。

时间模型把延迟项与带宽项分开：

$$
T_{\mathrm{ring}} = 2(d-1)\thinspace \alpha + 2 \thinspace \frac{d-1}{d} \cdot M \cdot \beta \qquad (3)
$$

**符号**： $\alpha$ ：每跳消息的固定延迟； $\beta$ ：每字节传输时间（有效带宽的倒数）。这个模型有两个 regime。小 $M$ 时，延迟项 $2(d-1)\alpha$ 主导，随 $d$ 线性涨；大 $M$ 时，带宽项主导，与 $d$ 无关。LLM 的梯度同步牢固处于第二个 regime：7B bf16 梯度 $M \approx 13$ GiB，一次 all-reduce 每 rank 发送约 $2M \approx 26$ GiB（与 1.5 §4 的账同一数字），耗时 $\approx 2M \cdot \beta$ 。所以注意力全在有效带宽上。节点内 NVLink 与跨节点 IB 的有效带宽差一个量级（硬件侧归 gpu：节点内 → 7.2，机间 → 9.1 / 9.2），这条量级差决定了 DP 组在拓扑上怎么摆（组合规则归 5.5）。

**树形 all-reduce 值得对照一句**。recursive halving/doubling 把消息步数从 $2(d-1)$ 压到 $2 \lceil \log_2 d \rceil$ ，每 rank 字节数仍是约 $2M$ 量级。大消息下两者的带宽项相同，树形赢的只是延迟项。DDP 论文在背景节同时引用了 ring-based 与 tree-based 两种 all-reduce 实现（[Li et al., 2020](../references.md)，arXiv:2006.15704，[A]，§2）。NCCL 按消息大小与拓扑在变体间选择；选择逻辑归 gpu 8.3 / 8.4 / 10.2，本书只用式 (2)(3) 的账。

## 3. DDP 实现走读：bucket 化、hook 与 overlap

DDP 把 §1-2 的语义包成一个 `nn.Module` 薄壳（pytorch/pytorch repo，`torch/nn/parallel/distributed.py`，教学参照 + 官方实现；[Li et al., 2020](../references.md)，arXiv:2006.15704，[A]；下述机制经论文与源码双重核实）。下面分构造期与反向期两段走读。

### 3.1 构造期：按参数注册的逆序分桶

DDP 的两条构造假设直接写在 docstring 的警告里。第一，所有参数必须在构造前注册完毕，之后不增删。第二，各进程的参数注册顺序必须一致。第二条关乎正确性。all-reduce 的桶内容必须跨 rank 逐字节对齐；而 PyTorch 是动态图，各 rank 的梯度就绪顺序本可以不同。论文 Fig. 3(a) 给了一个错位实例：rank 1 按 $g_1, g_2, g_3, g_4$ 就绪，rank 2 按 $g_1, g_3, g_4, g_2$ 就绪。若各按本地顺序发桶，两边桶的内容就错位，归约结果错误乃至崩溃。所以 bucket 划分必须在构造期静态确定、各 rank 一致。

所谓桶（bucket），是指拼进同一段连续缓冲、共用一次 all-reduce 的参数梯度组。桶怎么划？规则是把参数按 `model.parameters()` 的**逆序**装入桶。源码注释给了理由：参数的注册顺序通常与前向调用顺序一致，逆序就近似了反向传播中梯度的产出顺序。论文自己的口径更坦白：这不是完美解，是工程开销（口径：每条消息的启动与延迟成本，非带宽项）最小的近似。默认每桶上限 `bucket_cap_mb = 25` MiB。逆序后的第一桶（也就是反向中最先就绪的一桶）会单独缩小：源码注释写明目的是 "reduce latency for early-computed gradients"，默认值约 1 MB。这样反向刚开始就有一小桶可发，通信尽早启动，与后续计算重叠。

桶大小的权衡有实测支撑（论文 Fig. 2）。实验用 60M 个 fp32 参数，对比「逐个参数各发一次 all-reduce」与「聚合成大张量再发」：NCCL 与 Gloo 的总耗时都随单张量变大而下降，因为小张量下每条消息的固定开销占主导。两个库的饱和点不同——Gloo 在每张量约 50 万参数处饱和，NVLink 上的 NCCL 到 20M 参数仍未见饱和。但桶也不能大到整模一个：那就退化回「反向全算完才开始通信」，overlap 归零。25 MiB 是这个权衡的出厂值；论文明确建议按应用实测调。

### 3.2 反向期：梯度就绪即通信

hook 机制（论文 §3.2.3）分三步看。第一，DDP 给每个参数的 gradient accumulator（1.1 的 `AccumulateGrad` 节点）装同一个 post-hook；hook 在该参数梯度累加完成时触发，并检查所属桶。第二，桶内全部参数的 hook 都触发后，由最后一个 hook 发起该桶的**异步** all-reduce，随即返回，反向继续算下一层。反向按层逆序推进，桶就按注册逆序依次就绪，通信与剩余的反向计算并行。第三，反向末尾有一个 finalize 步：等所有异步 all-reduce 完成、把归约值写回梯度缓冲，之后才轮到 `clip → step → zero_grad`（顺序语义同 1.5 §3）。

```
时间 →     │←──────────── 反向计算（层序 L4 → L1）────────────→│
GPU 计算:  L4 bwd    L3 bwd    L2 bwd    L1 bwd
              └┐        └┐        └┐        └┐
bucket 就绪:   B0        B1        B2        B3     （B0 含最后注册的参数，最先算完）
通信流:        └─ AR(B0) ─┴─ AR(B1) ─┴─ AR(B2) ─┴─ AR(B3) ─┐
                                                           ▼
                                              finalize（等全部完成、写回 .grad）
```

**平均语义**：DDP 计算的是跨 rank 平均梯度，写回 `param.grad`（论文原话：DDP 总是计算所有梯度的平均并写回 `.grad` 字段）。「除以 world size」由 C++ reducer 在归约时折入：`torch/csrc/distributed/c10d/reducer.cpp` 里 `div_factor_` 取 process group size，bucket 视图执行 `.div_(div_factor_)`。于是 optimizer 看到的梯度与单卡大 batch 完全同义，式 (1) 在实现层面逐字兑现。这里有一个易混的同名参数：`divide_by_initial_world_size`（默认 True）是 `DDP.join()` context manager（uneven-inputs 训练）的参数——只在有 rank 数据耗尽、先行退出的 join 路径上，决定按初始 world size 还是按剩余 rank 数做除法，与常规路径的平均语义无关。

> **坑**：某轮迭代只走模型子图（条件分支、RL rollout 里的部分网络等）时，缺席参数的 hook 永不触发——现象是其所在桶永远等不到「最后一个 hook」，all-reduce 挂起（论文 Fig. 3(b)）。规避：DDP 在前向结束时从输出张量出发遍历 autograd 图，标记本轮实际参与的参数，以「参与集合的 hook 全触发」作为桶就绪的充分条件；跨 rank 的全局未参与参数，再用一次小 bitmap 的 all-reduce 对齐（bitmap 远小于梯度，一个桶装下）。

### 3.3 overlap 的账与边界

overlap 什么时候成立？判据可以直接写出来。设桶 $k$ 有 $B_k$ 字节、有效带宽对应 $\beta$ ，该桶通信耗时约 $2(d-1)/d \cdot B_k \cdot \beta$ 。若它不超过「该桶就绪时刻到反向结束」的剩余计算时间，则通信被完全藏住（口径：通信与计算重叠、不占关键路径），单步开销与单卡无异。论文 Fig. 2(c)(d) 给了一个量级对照：ResNet-152 的 GPU 反向约 250 ms，与同体量梯度在 NCCL on NVLink 上的 all-reduce 耗时同量级。重叠空间成立，但并不富余——这正是桶参数值得调的原因。两个暴露通信的边界情形：其一，模型小、层算得快、网络慢，反向时间轴太短，step 时间被 $2M\beta$ 决定；其二，大 $d$ 下单桶的 ring 延迟项 $2(d-1)\alpha$ 变长，靠多桶流水摊薄。判据的数值演算见练习 2。

### 3.4 `no_sync`：accumulation 下的通信除以 $m$

`no_sync` 的语义账已在 1.5 §4 立好：每个 optimizer step 只付一轮 all-reduce，通信总量除以累加步数 $m$ 。这里补实现侧的三行。第一，`no_sync` 上下文管理器做的事只是在进出时翻转 `require_backward_grad_sync` 旗标（`distributed.py` 中的实现：进入置 False，退出恢复原值）。第二，这个旗标在**前向**读出——forward 里以 `torch.is_grad_enabled() and self.require_backward_grad_sync` 决定是否走 reducer 路径，所以 docstring 警告「前向必须包在 context 内，否则梯度仍会被同步」。第三，context 内所有 DDP hook 被禁用（论文 §3.2.4 原话），反向退化为纯本地梯度累加；退出 context 后的第一次前向反向恢复正常同步，把 $m$ 步累加的梯度一轮归约完。1.5 §4 的数值例（ $m = 8$ 时 naive 方案每步每 rank 约 209 GiB、`no_sync` 约 26 GiB）就是这个开关的账。

## 4. DDP 的显存冗余：从 DP 到 ZeRO 的动机

DDP 对通信做了一切优化，对显存什么都没做：每个 rank 仍持有全量参数、全量梯度、全量优化器状态。这笔账是 bf16 工作权重 $2N$ + bf16 梯度 $2N$ + fp32 master $4N$ + fp32 两阶矩 $8N$ = $16N$ bytes。它的分解与对账表已在 models 1.2 式 (8) 交付，此处只登记结论，不重复推导；激活账独立，归 1.4。ZeRO 论文给了标尺例：7.5B 参数、Adam 混合精度（其记号 $K = 12$ ）时，模型状态合计 120 GB，超过当时任何单卡（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]，Table 1）。

DP 只摊薄了**计算**：每 rank 算 $1/d$ 的数据、付 $1/d$ 的 FLOPs。三态显存一字节没省—— $d$ 个 rank 的集群拿着 $d \times 16N$ 的总显存，只装下一份 $16N$ 的模型状态。模型一旦越过单卡 $16N$ 的线，纯 DP 就到头了。传统的下一步是模型并行：按张量或按层把模型本身切开（Megatron 路线，[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]；推导归 5.1 / 5.3），代价是完全不同的通信模式与代码侵入。ZeRO 换了一个角度看这件事： $16N$ 里真正属于「这份数据」的只有梯度的一份副本，参数与优化器状态的跨 rank 复制是纯冗余。把三态按 rank 切片、用时再取回，DP 的通信效率可以原样保留。这句话就是 4.2 的全部内容。

> **提点**：Megatron-LM 维护自己的 DP 层（NVIDIA/Megatron-LM repo `megatron/core/distributed`），把梯度存进连续缓冲做 overlap；其 distributed optimizer 是 ZeRO-1/2 的同族思想（优化器状态按 DP 域切片），走读归 6.2。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| DP 平均等价 | $d$ 副本梯度平均 = $d \cdot b$ 大 batch 梯度；与 accumulation 是同一线性叠加的两个轴 | 式 (1)；1.5 式 (2)(3)(6) |
| 同步 vs 异步 | 同步 DP 与单机大 batch 逐步同义；异步 SGD 引入 staleness、更新规则不同 | §1 |
| ring all-reduce | scatter-reduce + all-gather 各 $d-1$ 步；每 rank $2(d-1)/d \cdot M$ bytes；带宽项与 $d$ 无关 | 式 (2)(3) |
| 树形对照 | 消息步数 $2 \lceil \log_2 d \rceil$ 、字节数同量级；变体选择归 gpu 8.3 / 8.4 / 10.2 | §2.2 |
| DDP 分桶 | 参数注册逆序入桶；默认 25 MiB、首桶约 1 MB；桶内 hook 齐套即发异步 all-reduce | arXiv:2006.15704 §3.2；`distributed.py` |
| overlap | 判据：桶通信 $\le$ 该桶就绪后的剩余反向时间；逆序注册为对齐梯度就绪顺序 | §3.2-3.3 |
| 平均语义 | 除 world size 后求和、写回 `.grad`；optimizer 与单卡同语义 | §3.2 |
| `no_sync` | 旗标在前向读出、前向必须进 context；通信 ÷ $m$ 的账归 1.5 | §3.4；1.5 §4 |
| 显存冗余 | DDP 不省显存：每 rank $16N$ bytes，7.5B 即 120 GB ⇒ ZeRO 动机 | models 1.2 式 (8)；arXiv:1910.02054 Table 1 |

## 练习

1.（计算 · [20]）overlap 判据演算：某模型反向总时长 300 ms，梯度 $M = 26$ GiB，均分 8 桶，有效带宽 200 GB/s。(a) 每桶 all-reduce 耗时多少？(b) 设梯度在反向时间轴上均匀就绪（第 $k$ 桶在反向进程的 $k/8$ 处就绪），哪些桶能被剩余计算完全藏住、哪些暴露？暴露多少毫秒？(c) 改成 32 桶重算，结论怎么变、代价是什么？（忽略延迟项 $\alpha$ 。）校验：(a) 的八桶耗时之和与全量一次归约的耗时同量级。

2.（计算 · [20]） $N = 7 \times 10^{9}$ 、bf16 梯度、 $d \in \lbrace 8, 64, 512 \rbrace$ ：(a) 按式 (2) 算每 rank 每 step 的 all-reduce 发送字节数；(b) 有效带宽 100 GB/s 时通信时间下限各多少毫秒？(c) 设单卡有效算力 $10^{15}$ FLOPS、训练每 token 约 $6N$ FLOPs，每 rank 每 step 至少要吃进多少 token 才能把这次通信藏进计算？对照 1.5 式 (5) 说明大 $d$ 为什么总是与大 batch 绑定。校验：(a) 三档的系数都落在 1 到 2 之间。

3.（推导 · [30]）星型归约 vs ring： $d$ 个 rank、向量 $M$ 字节。(a) 写出星型方案 root 链路的收发字节数，证明它随 $d$ 线性增长；(b) 补全 §2.2 的归纳：把「scatter-reduce 第 $s$ 步结束时每块部分和累加了 $s+1$ 份贡献」写成关于 $s$ 的不变式并证明之，由此推出每 rank 恰发送 $(d-1) \cdot M/d$ ；(c) $d = 2^{k}$ 时写出 recursive halving/doubling 的消息步数与每 rank 字节数，与式 (2)(3) 逐项对照。校验：(c) 在两个 rank 时退化为互换全量向量的下界。

4.（代码 · [30]）教学用，非优化实现：纯 Python 在单机模拟 $d = 4$ 的 ring all-reduce——用列表模拟 4 个逻辑 rank，逐步执行 §2.2 的 scatter-reduce 与 all-gather，断言 (i) 结束后每 rank 持有全和向量；(ii) 每 rank 累计发送字节数恰为 $2(d-1)/d \cdot M$ ；(iii) 把归约换成平均（先除 $d$ ）后结果与直接平均一致。校验：断言 (ii) 正是正文推导的每 rank 发送量公式。

## 参考

- [Li et al., 2020](../references.md)：PyTorch Distributed: Experiences on Accelerating Data Parallel Training，arXiv:2006.15704，[A]（key: pytorch-ddp）——bucket 化动机与实测（§3.2.2 / Fig. 2）、逆序注册与 hook 机制（§3.2.3 / Fig. 3）、平均语义、未参与参数处理、`no_sync`（§3.2.4）
- [Rajbhandari et al., 2020](../references.md)：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，arXiv:1910.02054，[A]（key: zero）—— $16N$ 三态账（ $K = 12$ 口径）、Table 1 的 120 GB 标尺例、DP 冗余观察（4.2 的动机）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，arXiv:1909.08053，[A]（key: megatron-lm）——DP/MP 分界对照；Megatron DP 层与 distributed optimizer 走读归 6.2
- [PyTorch, pytorch/pytorch repo](../references.md)（key: pytorch-repo，教学参照 + 官方实现）——`torch/nn/parallel/distributed.py`：逆序分桶与 `bucket_cap_mb = 25` MiB 默认、首桶缩小优化（约 1 MB）、`no_sync` 旗标语义与 docstring 警告；`torch/csrc/distributed/c10d/reducer.cpp`：`div_factor_` 平均语义的实现位置（核实于 2026-11-17）
