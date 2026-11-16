# 1.4 激活内存与 Gradient Checkpointing（Activation Memory and Gradient Checkpointing）

> 本章把激活显存从「一笔随 $B \cdot T$ 涨的账」精确化到逐算子字节账（Megatron 的 $sbh(34 + 5as/h)$ 公式逐项拆解），再给出重计算换内存的完整数学： $L$ 层分 $k$ 段、内存 $M(k) = a(k + L/k)$ 、最优 $k = \sqrt{L}$ 、代价一次额外前向（+33% step 时间）；full vs selective recompute 的对照账（GPT-3 省 70% 激活只付 2.7% FLOPs）；`torch.utils.checkpoint` 的机制走读（RNG 态保存/恢复、dropout 掩码一致性、`use_reentrant` 两模式语义差异）与策略决策表 | 前置：1.1（vjp 链式执行、`ctx.save_for_backward` 语义）、1.2（算子 vjp 总表——每个算子反向需要前向留什么）、models 1.2（激活账初版： $BTL(65536 + 16T) \times 2$ bytes） | 锚点：`torch/utils/checkpoint.py`（教学参照 + 官方实现）、[Korthikanti et al., 2022] [A]；[Chen et al., 2016] [B]

## 1. 激活账精确化：反向时哪些张量必须还在

反向传播不是免费的方向场：每个算子的 vjp 都要消费前向留下的某些张量（1.1 的 `ctx.save_for_backward` 语义；逐算子的「前向公式 / 反向公式 / 需 save 的量」总表在 1.2）。最典型的是 matmul： $Y = XW$ 的权重梯度 $\partial \mathcal{L} / \partial W = X^{\top} G$ 需要前向输入 $X$ ——前向算完不存 $X$ ，反向就算不出权重梯度。激活账的全部内容就是回答：一张 $L$ 层网络的前向，每个算子必须为反向留下什么。

Megatron 论文把这笔账算到了字节粒度（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]，§4.1）。记号： $s$ = 序列长度， $b$ = micro-batch 序列数， $h$ = hidden 宽度， $a$ = attention head 数；激活以 2 字节（fp16/bf16）计、dropout 掩码以 1 字节计。单个 transformer 层（GPT 式：attention 块 + MLP 块 + 两个 LayerNorm，MLP 为 $h \to 4h$ 的 GeLU 结构）逐项清点——每行同时给出「为什么要存」：

| 算子 | 存下的张量 | 字节数 | 反向用途 |
| --- | --- | --- | --- |
| QKV 投影 | 共享输入 | $2sbh$ | 三个 matmul 的权重梯度 |
| $QK^{\top}$ | $Q$ 、 $K$ | $4sbh$ | matmul 双向梯度 |
| softmax | 输出概率 $P$ | $2as^{2}b$ | softmax 反向的 Jacobian-vjp |
| softmax dropout | 掩码 | $as^{2}b$ | 掩码逐元素乘回梯度 |
| $PV$ | dropout 输出、 $V$ | $2as^{2}b + 2sbh$ | matmul 双向梯度 |
| attention 线性投影 | 输入 | $2sbh$ | 权重梯度 |
| attention dropout | 掩码 | $sbh$ | 掩码乘回 |
| MLP 两个线性 | 各自输入 | $2sbh + 8sbh$ | 权重梯度 |
| GeLU | 输入 | $8sbh$ | 逐元素导数 $\mathrm{GeLU}'(x)$ |
| MLP dropout | 掩码 | $sbh$ | 掩码乘回 |
| 两个 LayerNorm | 各自输入 | $4sbh$ | norm 反向（1.2） |

合计：attention 块 $11sbh + 5as^{2}b$ 、MLP 块 $19sbh$ 、norm $4sbh$ ，全层

$$
\mathrm{Mem}_{\mathrm{layer}} = sbh\left( 34 + 5\frac{as}{h} \right) \text{ bytes} \qquad (1)
$$

**符号**：定长项 34 来自全部随 $s$ 线性的张量； $5as/h$ 项来自 attention 核心（ $QK^{\top}$ 、softmax、softmax dropout、 $PV$ 四处共存的 $T^{2}$ 级张量）。式 (1) 是论文 Equation 1（无并行口径）；论文的 Equation 5 再把 pipeline、embedding、输出头等修正项补齐，并注明那些项相对式 (1) 可忽略（22B 模型上小于 0.01%）。

读式 (1) 的方式与 models 1.2 的激活账相同：定长项随 $B \cdot T$ 线性， $5as^{2}b$ 项随 $T^{2}$ 二次——models 1.2 的「 $65536 + 16T$ 」结构与这里「 $34 + 5as/h$ 」是同一件事的两种实现口径（SwiGLU 三矩阵对 GeLU 双矩阵、causal 平均对全量 $T^{2}$ 、存的张量清单不同），结构项不变。代入 GPT-3 175B（ $a = 96$ 、 $s = 2048$ 、 $h = 12288$ ）： $5as/h = 80$ ，定长项 34—— $T^{2}$ 张量占全层激活的 70%。这个比例是第 3 节 selective recompute 的全部动机。

存与不存的判据逐行可对回 1.2 的 vjp 总表，两条规则覆盖全表。其一，matmul 类算子的两个梯度各要一个操作数： $Y = XW$ 的 $\partial \mathcal{L} / \partial W = X^{\top} G$ 要 $X$ ， $\partial \mathcal{L} / \partial X = G W^{\top}$ 要 $W$（ $W$ 是参数、常驻显存，不进激活账）——所以表中每个 matmul 行存的净是「输入」。其二，逐元素算子的反向是「输出梯度 = 导函数值 × 上游梯度」：导函数若能用输出闭式表达就存输出（softmax 反向 $p \odot (g - \langle p, g \rangle)$ 只用输出 $P$ ，推导见 1.2），必须由输入求值就存输入（ $\mathrm{GeLU}'$ 无输出的闭式表达，存输入 $8sbh$ ）。dropout 两行特殊：掩码是 1 字节布尔，而它的「存」在重计算实现里直接转化为 RNG 态问题——第 4 节会看到 checkpoint 实现通过保存/恢复 RNG 态让掩码被逐位重建，等效于存掩码。

代入一组现代训练口径的数字（ $h = 4096$ 、 $a = 32$ 、 $s = 4096$ 、 $b = 1$ ，近 Llama-3.1-8B 的几何）：每 token 每层 $34h + 5as \approx 7.9 \times 10^{5}$ 字节 ≈ 776 KB，32 层累计约 24 MB/token，一个 $B \cdot T = 32768$ token 的 step 全存约 800 GB——80 GB 级单卡无望，「重计算与并行切分至少启用其一」由此从经验变成算术。models 1.2 对同规模模型给出约 256 GiB 的账，与本段相差约 3 倍，差异来自所存张量清单的精简程度与 causal 平均口径；两账同阶同结构：定长项 $\propto BTLh$ ， $T^{2}$ 项 $\propto BLaT^{2}$ 。

## 2. 重计算换内存： $\sqrt{L}$ 分段最优

Chen et al. 的方案把问题压成一行优化（[Chen et al., 2016](../references.md)，arXiv:1604.06174，[B]，§4.3）。设 $L$ 层链式网络、每层激活 $a$ 字节（均匀假设；非均匀情形的贪心分配见论文 Alg. 3），把 $L$ 层切成 $k$ 段，只存 $k$ 个段边界激活（checkpoint）；反向进入第 $i$ 段时，从段入口的 checkpoint 重新前向该段、即时重建段内 $(L/k)$ 层的激活供 vjp 消费，用完即释放。峰值内存两笔相加：

$$
M(k) = \underbrace{k \cdot a}_{\text{checkpoints}} + \underbrace{\frac{L}{k} \cdot a}_{\text{segment recompute peak}} \qquad (2)
$$

**符号**：第一笔全程驻留；第二笔只在反向经过该段时存在（逐段重建、逐段释放，峰值即单段）。对 $k$ 求导： $M'(k) = a\thinspace (1 - L / k^{2}) = 0$ 得

$$
k^{\ast} = \sqrt{L}, \qquad M(k^{\ast}) = 2a\sqrt{L} \qquad (3)
$$

激活内存从 $O(L)$ 降到 $O(\sqrt{L})$ 。代价：每个 checkpoint 段在前向跑一次、反向前再重跑一次——一次额外前向。按前向 : 反向 = 1 : 2 的 FLOPs 比（models 1.2 式 (4)），step 总开销从 3 份涨到 4 份：+33%，即 $C$ 从 $6ND$ 涨到 $8ND$ 量级（models 1.2 练习 4 的口径）。Megatron 的实测与此一致：full recomputation 的执行时间开销 30–40%（[Korthikanti et al., 2022](../references.md)，[A]，§5）。Chen et al. 的实证：1000 层 ResNet 的激活内存从 48 GB 压到 7 GB；论文另给递归版本的极端情形—— $O(\log L)$ 内存、 $O(L \log L)$ 额外计算，本书不展开。

两个工程注记。其一， $k$ 无须整除 $L$ ，式 (2) 对任意分段成立；实践中段界常与 pipeline stage 边界对齐（5.3），可选的 $k$ 是离散集合，取离 $\sqrt{L}$ 最近的可行值即可——式 (2) 在最优点附近平坦： $k$ 偏到 $2\sqrt{L}$ 时 $M$ 只涨到 $2.5 \thinspace a\sqrt{L}$ ，即 $1.25 \times M(k^{\ast})$ 。其二，本节的分段粒度是整层，段内激活一律不驻留；把粒度拆到层内算子、让「贵的留下、便宜的重算」，就是第 3 节 selective 的内容。

时序图（ $k$ 段，□ 为驻留的 checkpoint，重算段以 ── 标出）：

```
前向   x0 ──seg1──► x1 ──seg2──► x2 ── ... ──segk──► xk ──► loss
       □            □            □                  □
       存           存           存                 存（共 k 份，全程驻留）

反向   loss ─► 段k：取 x_{k-1} ──重算段k（激活即用即弃）──► vjp ─► ...
            ─► 段2：取 x1 ──重算段2──► vjp ─► 段1 同法
峰值 = k 份 checkpoint + 1 段重算激活 = 式 (2)
```

## 3. full vs selective：Megatron 的选择账

式 (3) 的段取到最细（每段一层）就是 full activation recomputation：只存层输入，每层 $2sbh$ 字节、全模型 $2sbhL$ ，激活从式 (1) 的 $(34 + 5as/h)$ 倍压到 2 倍；代价即第 2 节的一次额外前向，30–40% 时间开销。介于「全存」与「full」之间的粗粒度折中——只 checkpoint 一部分层、其余层全存——在 PP 下经常失效：层已按 stage 切薄，可调的层数所剩无几（MT-NLG 每设备只有 3 层，论文 §5）。Megatron 论文的观察是这笔交易可以拆细：**不是所有激活都同样难算**（[Korthikanti et al., 2022](../references.md)，[A]，§5）。式 (1) 里 $5as^{2}b$ 那五项—— $QK^{\top}$ 、softmax、softmax dropout、 $PV$ ——张量巨大但每元素 FLOPs 极低（softmax、dropout 是逐元素算子；两个 matmul 的收缩维只有 head 维 $h/a$ ）；其余 34 项对应的投影与 MLP matmul 才是算力大头。selective activation recomputation 据此只重算 attention 核心、保留其余全部激活：内存占大头的没了，重算的 FLOPs 只是零头。

论文的两个实例（§5）：GPT-3 175B（ $5as/h = 80$ ）省 70% 激活内存、重算仅引入 2.7% FLOPs 开销；MT-NLG 530B（ $5as/h = 64$ ）省 65%、1.6%。这两个 FLOPs 数可以独立复核。重算内容是 $QK^{\top}$ 与 $PV$ 两个 matmul（softmax / dropout 的逐元素开销相对低一个 head 维因子）：每序列每层 $4 s^{2} h$ FLOPs——朴素实现先算满 $s \times s$ 矩阵再 mask，causal 不减这笔账（models 1.2 式 (6) 的减半是 causal 感知实现的口径）；对照层前向 matmul 总量 $24 s h^{2}$（QKV 投影 $6sh^{2}$ 、输出投影 $2sh^{2}$ 、MLP $16sh^{2}$ ）：

$$
\frac{\text{attn core recompute}}{\text{layer fwd matmul}} = \frac{4 s^{2} h}{24 s h^{2}} = \frac{s}{6 h} \qquad (4)
$$

**符号**：分子为重算一遍 attention 核心的稠密 FLOPs；分母为层前向的投影与 MLP matmul（不含 attention 本体 $4 s^{2} h$ ）。代入：GPT-3 得 $2048 / (6 \times 12288) \approx 0.028$ 即 2.8%，MT-NLG 得 $2048 / (6 \times 20480) \approx 0.017$ 即 1.7%——与论文的 2.7% / 1.6% 差 0.1 个百分点，差额来自分母口径：论文（Appendix A）的分母另含 attention 本体与 logits 项。式 (4) 同时说明收益的结构： $s$ 越长、 $h$ 越窄，selective 越划算。单组启用条件即

$$
5\frac{as}{h} > 34 \qquad (5)
$$

**符号**：不等式成立时 $T^{2}$ 部分占全层激活过半，selective 才「存得少于一半」。逐层时间实测（论文 Table 4，22B 模型单层前向 + 反向基线 7.7 + 11.9 ms）：full recompute 合计开销 39%（高于 33% 的纯 FLOPs 预期——基线反向本身做了 all-reduce 与权重梯度计算的 overlap 优化，分母变小使相对开销上浮，论文 §6.2 注记），selective 7%，selective 叠加 sequence parallelism 后 4%；530B 与 1T 规模上 selective + SP 一侧开销降到 2%（论文 Figure 8 的合并口径），full 仍为 36%。端到端吞吐较 full recompute 提升 29.0–32.1%（论文 Table 5）。

不同并行与重算组合下的每层激活账（论文 Table 2， $t$ 为 TP 度数）：

| 配置 | 每层激活字节数 |
| --- | --- |
| 无并行 | $sbh\thinspace (34 + 5as/h)$ |
| TP | $sbh\thinspace (10 + 24/t + 5as/(ht))$ |
| TP + sequence parallel | $sbh\thinspace (34/t + 5as/(ht))$ |
| TP + selective | $sbh\thinspace (10 + 24/t)$ |
| TP + SP + selective | $sbh\thinspace (34/t)$ |
| full recompute | $2sbh$ |

一行边界注记：式 (1) 的 $5as^{2}b$ 项以物化 softmax 概率矩阵 $P$ 为前提；FlashAttention（models 4.1）不物化 $P$ 、只留 online softmax 的统计量，把该项的大头直接消掉。现代训练栈里两者通常同时启用——FlashAttention 管 $T^{2}$ 项、checkpoint 管定长项——selective 的收益对象随 FlashAttention 的普及而收缩，这是本书对论文口径之外形势的注记。

## 4. `torch.utils.checkpoint` 机制走读

PyTorch 的官方实现在 pytorch/pytorch repo 的 `torch/utils/checkpoint.py`（[PyTorch, pytorch/pytorch repo](../references.md)，教学参照 + 官方实现；以下函数与类名均出自该文件）。入口 `checkpoint(function, *args, use_reentrant=...)` 提供两套实现，语义差异不小，官方推荐 `use_reentrant=False` 。

**重入版（`use_reentrant=True` ，类 `CheckpointFunction`）**——一个自定义 `torch.autograd.Function` ：

- 前向：`torch.no_grad()` 下运行 `function` ——不建 autograd 图，段内中间激活随算随弃；`ctx.save_for_backward` 只存输入张量。`preserve_rng_state`（默认 True）时 stash 前向 RNG 态：CPU 侧 `torch.get_rng_state()` ，设备侧 `get_device_states(*args)` 逐设备存；autocast 状态（`_get_autocast_kwargs` ）同样记录。
- 反向：`torch.random.fork_rng` 上下文里把 RNG 态恢复到前向时刻（`torch.set_rng_state(ctx.fwd_cpu_state)` + `set_device_states` ），`torch.enable_grad()` 下整段重跑 `function` ——这次建图；然后对重算输出调 `torch.autograd.backward` ，把输入梯度取回并沿图继续上传。
- 限制（源码注释与文档口径）：与 `torch.autograd.grad` 的 `inputs` 参数不兼容；至少一个输入与一个输出 `requires_grad=True` ，否则段内无梯度；段内含 `detach()` 或嵌套结构（list/dict 里的张量）不参与 autograd——这些在非重入版都支持。

**非重入版（`use_reentrant=False` ，`_checkpoint_without_reentrant_generator`）**——不包 Function，改用 `torch.autograd.graph.saved_tensors_hooks` 拦截「保存」动作：

- 前向在 `_checkpoint_hook` 下运行：`pack_hook` 被任何算子要存张量时触发，不存张量本体，只返回一个空占位 `_Holder` ，同时记录 metadata（shape / dtype / device，供 `determinism_check` 比对）；输入经 `_NoopSaveInputs`（一个只存输入的自定义 Function）挂进图（版本注记：此类名核实于 PyTorch ≤ 2.10 的源码，本章写作时本机 2.8.0 逐行核对；更新的 release 已把该逻辑重构进 `_CheckpointFrame`，机制不变——版本漂移的一般性风险见 training/README 的锚定纪律）。
- 反向首次 unpack 某个占位时：经 `input_saver.grad_fn` 取回输入，恢复 RNG 与 autocast 态，在 `_recomputation_hook` 下重跑 `function` ；重算过程中的 `pack_hook` 按序把每个被 save 的张量写回对应 holder，反向图随即拿到真实张量。默认开启 early stop：重算出全部所需张量即抛 `_StopRecomputationError` 终止重算（`set_checkpoint_early_stop(False)` 可关）；重入版无此机制，永远整段重算——文档明示这是两版差异之一。
- `determinism_check="default"` 比对重算张量与前向 metadata 的 shape/dtype/device；若前向与重算不一致（如全局变量漂移），重入版会静默产出错误梯度，非重入版至少能报 shape 级不符。

RNG 态的保存/恢复不是细节而是正确性条件：段内有 dropout 时，反向 vjp 要乘回前向那张掩码（1.2 的 dropout 反向）；重算若用不同的 RNG 态，掩码就不同，梯度静默地错。两版都靠「stash 前向态 + `fork_rng` 恢复 + 用完还原外围态」保证重算与前向逐位一致。`preserve_rng_state=False` 只在段内无随机性时才安全。

两个补充。其一，同文件还有便利函数 `checkpoint_sequential(functions, segments, input)` ：把 Sequential 式层序列切成 `segments` 段逐段 checkpoint，是式 (2) 分段的直接 API 化。其二，两版的能力差逐条来自源码 docstring 的对照清单：

| 能力 | 重入版（`use_reentrant=True`） | 非重入版（`False`，推荐） |
| --- | --- | --- |
| 前向是否建 autograd 图 | 不建（`no_grad` 下运行） | 建图，保存动作被 hook 拦截为占位 |
| 重算范围 | 永远整段重跑 | early stop：所需张量齐即停 |
| `torch.autograd.grad` 带 `inputs` | 不支持（反向报错） | 支持 |
| kwargs、嵌套结构、段内 detach | 不支持 | 支持 |
| `determinism_check` | 不可用 | 默认比对重算张量的 shape / dtype / device |

新代码一律 `use_reentrant=False` ——官方文档明确推荐，重入版存在的意义是历史兼容；`determinism_check="none"` 可关掉 metadata 比对换一点开销，默认开启时它是「前向与重算不一致」这类静默错误的第一道哨兵。

教学骨架（教学用，非优化实现）：

```python
# 教学用，非优化实现：逐 block checkpoint 的标准用法
from torch.utils.checkpoint import checkpoint

x = embed(tokens)                      # 段外激活照常驻留
for block in blocks:                   # 每个 transformer block 一个 checkpoint 段
    x = checkpoint(block, x, use_reentrant=False)   # 只存段输入 x，段内激活反向时重算
loss = loss_fn(head(x), targets)
loss.backward()                        # 反向逐段：取输入 → 恢复 RNG → 重算 → vjp
```

## 5. 与 TP / PP 的交互与策略决策表

**TP 交互**。TP 把 attention 与 MLP 块内的激活按度数 $t$ 切分，但 LayerNorm、dropout 等逐位置算子的激活在各 rank 冗余——式 (1) 拆成论文 Equation 2 的 $10 + 24/t + 5as/(ht)$ 正是这种结构的账： $24/t$ 项是块内被切分的部分， $10$ 项是冗余不动的部分（消掉它是 sequence parallelism 的事，归 5.2）。代入 GPT-3 几何与 $t = 8$ ：TP 后每层 $10 + 3 + 10 = 23$ 个 $sbh$ 单位，再叠 selective 压到 13——无并行口径 114 的约九分之一。checkpoint 段与 TP 通信算子正交：段内该 all-reduce 照常 all-reduce，重算时原样重放同样的通信；TP 组内各 rank 必须重算出同一份激活，RNG 态因此按 DP rank 区分种子、TP 组内共享（实现细节归 5.1 / 6.2）。

**PP 交互**。PP 按层分 stage，每个 stage 只持有自己那几层：checkpoint 策略逐 stage 独立决定，互不影响。真正与激活峰值耦合的是流水线调度：GPipe 式调度下同一时刻有 $m$ 份 micro-batch 的激活在飞，1F1B 压到至多 $p$ 份（ $p$ 为 stage 数；气泡与在飞份数的推导归 5.3）。stage 峰值 = 单份激活 × 在飞份数——本章式 (1)(2) 决定「一份多大」，schedule 决定「同时几份」，两轴正交相乘。还有一个粒度后果：一个 stage 内层数少时（MT-NLG 每设备只有 3 层，论文 §5 原话），「只 checkpoint 其中几层」这种粗粒度折中直接不可用，选择塌缩成 full 或 selective 二选一——这是 selective 存在的另一个理由。

**决策表**。给定单卡显存 $M_{\mathrm{budget}}$ 与可接受的 step 时间膨胀上限，选择顺序：

| 情形 | 策略 | 代价 |
| --- | --- | --- |
| 激活放得下 | 不 checkpoint | 0 |
| 差一点（ $T^{2}$ 项顶格） | selective（式 (5) 成立时） | 2–7% 时间（论文实测区间） |
| 差一个数量级 | full（每层一段） | 30–40% 时间 |
| 还不够 | full + 缩小 micro-batch（梯度累加补回批量，归 1.5） | 时间 + 通信账另算 |
| 算力有余量、想换批量 | 反向用省下的显存放大 micro-batch | 重算开销换吞吐 |

把表读一遍的具体情形：设权重三态与通信 buffer 已由并行方案安置（激活账与三态账合成统一账归 4.4），单卡剩 20 GiB 给激活；模型取 $h = 4096$ 、 $a = 32$ 、 $s = 4096$ 、 $L = 32$ 、 $B \cdot T = 8192$ token——此几何下 $5as/h = 160$ ，式 (1) 的系数为 $34 + 160 = 194$ ，全存激活约 194 GiB（794624 字节 × 32 层 × 8192 token），缺口近 10 倍。逐档换算：只上 selective 剩 $34 / 194$ 即 34 GiB，仍超预算；TP 8 路加 selective 为 $194 \times (10 + 3) / 194 = 13$ GiB，已满足 20 GiB 预算、只付 2–7% 时间；再叠 SP 压到 $34 / (8 \times 194)$ 即 4.25 GiB，属进一步压缩（SP 的通信与时间账归 5.2）；full checkpoint 压到 $194 \times 2 / 194 = 2$ GiB，但付 30–40% 时间——算力余量不够时这笔交易不成立。决策的读法：先算「激活缺口倍数」，再从表内自上而下取第一个满足预算的策略。

判断式 (5) 的两端代入即知： $a = 32$ 、 $s = 4096$ 、 $h = 4096$ 时 $5as/h = 160 > 34$ ，selective 划算；短序列小模型 $5as/h$ 只有个位数时，selective 无肉可吃，要么不 checkpoint 要么直接 full。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 逐算子激活账 | 式 (1)： $sbh(34 + 5as/h)$ | 每项对应 1.2 vjp 表的「需 save 的量」； $T^{2}$ 项占 GPT-3 层激活 70% |
| $\sqrt{L}$ 最优 | 式 (2)(3)： $M(k) = a(k + L/k)$ ， $k^{\ast} = \sqrt{L}$ | 内存 $O(\sqrt{L})$ 、代价一次额外前向（+33%，实测 30–40%） |
| full vs selective | 式 (4)：重算占比 $s/6h$ ；式 (5)：启用条件 $5as/h > 34$ | GPT-3 省 70% 付 2.7%；full 压到 $2sbhL$ 付 30–40% |
| torch 实现 | `torch/utils/checkpoint.py` | RNG 态 stash/恢复保 dropout 一致；非重入版 = saved_tensors_hooks 占位 + early stop |
| 并行交互 | 论文 Table 2（ $t$ 度 TP 五行） | PP 逐 stage 独立；「一份多大」×「同时几份」正交 |

## 练习

1.（推导）把式 (2) 推广到非均匀层：第 $\ell$ 层激活 $a_\ell$ 任意，分段点集合 $S$ 的内存为 $\sum_{i \in S} a_i + \max_{\mathrm{seg}} \sum_{\ell \in \mathrm{seg}} a_\ell$ ；说明 $k^{\ast} = \sqrt{L}$ 在均匀情形由均值-几何均值不等式一步得到，并举一个非均匀例子说明贪心预算分配（Chen et al. Alg. 3）的必要性。
2.（计算）代入验证论文数字：GPT-3 175B（ $a = 96$ 、 $s = 2048$ 、 $h = 12288$ ）算 $5as/h = 80$ 与 selective 的省内存比例 $80 / (34 + 80) \approx 70$% ；再用 Llama-3.1-8B 训练口径（ $a = 32$ 、 $h = 4096$ 、 $s = 4096$ 、 $L = 32$ ）分别算全存、full、TP 8 路 + selective 三种配置下每 token 的激活字节数。
3.（计算） $L = 61$ 层的模型（DeepSeek-V3 层数）：均匀假设下全存需 $61a$ ，式 (3) 最优分段需 $2\sqrt{61} \thinspace a \approx 15.6 \thinspace a$ ；若段数必须为整数且整除层数，枚举可行 $k$ 并给出实际最优与理论最优的差距。
4.（代码）教学用：手写一个 30 行的 mini-checkpoint（存输入 + `torch.no_grad` 前向 + backward 钩子内重算），与 `torch.utils.checkpoint(..., use_reentrant=False)` 在同一含 dropout 的网络上对拍梯度；再置 `preserve_rng_state=False` 观察梯度偏差，验证第 4 节的 RNG 论证。

## 参考

- [Chen et al., 2016](../references.md)：Training Deep Nets with Sublinear Memory Cost，arXiv:1604.06174，[B]（key: gradient-checkpointing）——式 (2)(3) 的原始方案、1000 层 ResNet 48 GB → 7 GB、 $O(\log L)$ 递归变体
- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——式 (1) 逐字节账（其 Eq. 1）、Table 2 六行配置、selective 的 70% / 2.7% 与 Table 4 时间实测
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照 + 官方实现（key: pytorch-repo）——`torch/utils/checkpoint.py`：`checkpoint` / `CheckpointFunction` / `_checkpoint_without_reentrant_generator` / `_checkpoint_hook` / `_recomputation_hook` / `_NoopSaveInputs`（≤ 2.10 类名，新版本已重构）/ `set_checkpoint_early_stop`
