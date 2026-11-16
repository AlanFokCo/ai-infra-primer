# 1.5 Gradient Accumulation 与 Micro-batch（Gradient Accumulation and Micro-batching）

> 本章把「拆 batch」算到底：大 batch 拆成 $m$ 个 micro-batch 依次前向反向、梯度原地累加、最后统一更新——数学等价的精确条件（归一化位置与变长序列的权重陷阱）、等效 batch 的 tokens 口径与 LR/schedule 的耦合、梯度裁剪与优化器状态的顺序语义、以及数据并行下 `no_sync` 把 all-reduce 轮次除以 $m$ 的通信账 | 前置：1.1（反向时 `.grad` 的原地累加语义）、1.4（激活内存账——拆 micro-batch 的动机）、models 1.2（ $6ND$ FLOPs 口径） | 锚点：PyTorch DDP `no_sync`（[Li et al., 2020](../references.md)，arXiv:2006.15704，[A]，§3.2.4；实现 pytorch/pytorch repo `torch/nn/parallel/distributed.py`，教学参照）；梯度噪声背景 [McCandlish et al., 2018](../references.md)，arXiv:1812.06162，[B]

## 1. 累加等价性：micro-batch 平均的平均 = 大 batch 的平均

显存装不下想用的 batch 时，标准做法是把 batch 切成 $m$ 份逐个前向反向。为什么会「想用」放不下的 batch：数据并行的度数 $d$ 直接乘进等效 batch（§4），集群越大、每步摊到的序列越多；同时梯度噪声随 batch 按 $1/B$ 收缩（2.1 式 (3)），大步长需要大 batch 支撑。激活峰值内存近似正比于一次前向的序列数（1.4），单卡放不下的那部分就由累加补齐。设目标 batch 含 $B$ 条序列，第 $j$ 条的 loss 为 $\ell_j$（序列内 token 已平均；token 级口径见式 (4)），大 batch 目标是

$$
\bar\ell = \frac{1}{B} \sum_{j=1}^{B} \ell_j \qquad (1)
$$

**符号**： $B$ ：batch 中序列数； $\ell_j$ ：第 $j$ 条序列的标量 loss； $\bar\ell$ ：大 batch 目标。

把 batch 切成 $m$ 个 micro-batch $S_1, \ldots, S_m$ ，各含 $b = B/m$ 条序列，定义 micro-batch 平均 loss ：

$$
\hat\ell_i = \frac{1}{b} \sum_{j \in S_i} \ell_j, \qquad \bar\ell = \frac{1}{m} \sum_{i=1}^{m} \hat\ell_i \qquad (2)
$$

**符号**： $\hat\ell_i$ ：第 $i$ 个 micro-batch 内的平均 loss。第二行就是「先各自平均再累加」与「全量一次平均」的等价——加法的结合律，没有近似。

梯度侧同理。对 $\hat\ell_i / m$ 调 `backward()`，autograd 把 $\nabla (\hat\ell_i / m)$ 原地加进每个参数的 `.grad`（1.1 的 `AccumulateGrad` 语义：`.grad` 不清零就持续累加）。 $m$ 个 micro-batch 走完后

$$
\mathrm{grad} = \sum_{i=1}^{m} \nabla \frac{\hat\ell_i}{m} = \frac{1}{m} \sum_{i=1}^{m} \nabla \hat\ell_i = \nabla \bar\ell \qquad (3)
$$

第二步用梯度的线性性。即：累加方案拿到的参数梯度与大 batch 一次反向完全相同。这条性质还有一个常用方向：它说明 gradient accumulation 与 gradient checkpointing（1.4）是两种不同的显存手段——checkpointing 用一次额外前向换激活不落地（FLOPs 涨到约 $8ND$ ），累加不动 FLOPs 但每步算得慢（ $m$ 次小前向反向的固定开销）；两者可以叠加，先靠累加把单步 batch 抬到目标，再靠 checkpointing 把每个 micro-batch 的激活峰值压进显存。

$1/m$ 这个因子的位置有三种等价放法——(a) 每个 micro-batch 的 loss 先除 $m$ 再 `backward()`（本章代码的写法）；(b) 累加 $m$ 份原始梯度、更新前统一除 $m$ ；(c) 更新时把 LR 临时除 $m$ 。只能选一处放。漏放的效果 = 梯度放大 $m$ 倍 ≈ LR 放大 $m$ 倍： $m = 16$ 的设定下等于无声地把 LR 抬 16 倍，训练轻则 loss 曲线整体偏漂、重则直接发散——这是累加实现里最常见的 bug。(c) 会把 LR 的语义搅乱（schedule、warmup、裁剪阈值都要跟着换算），只作等价性说明，不要落地。

等价性成立依赖三个前提，逐条核对：

1. **等权前提**：式 (2) 的第二个等号要求各 micro-batch 权重相等，即 $|S_i|$ 全等。LLM 预训练用定长 packed 序列（每条恰好 $T$ 个 token），天然满足；dataloader 末尾凑不齐整批的余数 micro-batch 是例外，要么丢弃要么按式 (4) 加权。
2. **逐样本独立前提**：式 (3) 要求一条序列的前向不依赖同 batch 的其他序列。LayerNorm / RMSNorm 是逐 token 算子（models 2.7），满足；BatchNorm 这类跨样本统计量让 micro-batch 的前向依赖切分方式（均值方差在当前 micro-batch 内估计），等价性直接失效——LLM 不用 BatchNorm，此坑在 CV 侧。
3. **token 口径前提**：若 loss 在 micro-batch 内按「有效 token 平均」（padding mask 后的 mean 归约），而各 micro-batch 的有效 token 数 $n_i$ 不等，则等权的 mean-of-means 不再等于全局 token 平均：

$$
\bar\ell_{\mathrm{tok}} = \frac{\sum_{i=1}^{m} n_i \thinspace \hat\ell_i}{\sum_{i=1}^{m} n_i} \neq \frac{1}{m} \sum_{i=1}^{m} \hat\ell_i \qquad (4)
$$

**符号**： $n_i$ ：micro-batch $i$ 的有效 token 数； $\bar\ell_{\mathrm{tok}}$ ：按 token 加权的全局平均。修正：第 $i$ 个 micro-batch 的 loss 乘 $n_i / \sum_i n_i$ 再反向，而不是 $1/m$ 。定长 packed 数据下 $n_i$ 全等，式 (4) 退回式 (2)——练习 1 构造反例。

审查一份真实训练代码是否踩坑，按三条前提倒查即可：`criterion` 的 reduction 是 mean 还是 sum、每个 micro-batch 的 loss 乘的是 $1/m$ 还是 $1$ 、micro-batch 之间 token 数是否恒定、norm 层是否逐 token。四个答案都对，式 (3) 就在你的代码里成立；任何一个不对，先修口径再谈调参。

**FLOPs 与内存账**。总计算量与切分无关：每 token 前向加反向约 $6N$ FLOPs（models 1.2 式 (5)），总 token 数相同则总 FLOPs 相同，累加在纸面上是免费的。实际代价在两处：激活峰值内存从 $\propto B$ 降到 $\propto b$ （1.4 的激活账 ÷ $m$ ，这正是动机）换来每 micro-batch 一轮的 kernel 启动、框架调度与同步开销—— $b$ 太小时单次前向反向喂不饱 GPU，累加从「白拿的显存」退化成「拿吞吐换显存」， $b$ 的选择下限由硬件利用率定、上限由显存定。累加自身在显存上近乎免费：`.grad` 始终只有一份参数大小的缓冲， $m$ 不放大它。浮点层面还有一句实话：浮点加法不满足结合律， $m$ 步累加与一次大 batch 反向的求和顺序不同，结果是「数学恒等、位级不等」——fp32 梯度缓冲下差异远低于梯度噪声本身，bf16 场景放宽容差即可（§5 的自检代码有注）。

**归一化位置选哪处的工程差别**也值得记一笔。(a) 的坏处是 loss 日志被缩小 $m$ 倍（§3 的日志口径靠求和恢复）；(b) 要在更新前插一次全参数缩放，多一遍读写；(c) 已否决。生产训练栈多数走 (a) 的变体——把 $1/m$ 折进 criterion 或梯度累加器的缩放里——原因正是它把「每个 micro-batch 对总梯度的贡献」这个语义写在了反向发生的地方。

## 2. 与 LR / schedule 的耦合：batch 一律换算成 tokens

「batch size」在各家配方报告里有三种口径：序列数、样本数、tokens 数——彼此之间差 $T$ 与 $m \cdot d$ 的因子，跨报告直接比数字没有意义。本书口径（7.2 的配方表统一按此换算）：**比较配方时 batch 一律用每步 tokens 数**：

$$
B_{\mathrm{tok}} = b \cdot m \cdot d \cdot T \qquad (5)
$$

**符号**： $b$ ：每设备 micro-batch 序列数； $m$ ：累加步数； $d$ ：数据并行度数（副本数；单卡 $d = 1$ ，DP 语义见 §4 与 4.1）； $T$ ：序列长； $B_{\mathrm{tok}}$ ：一次 optimizer step 消耗的 token 数，即等效 batch（effective batch）的 tokens 口径。算例： $b = 8$ 、 $m = 16$ 、 $d = 256$ 、 $T = 4096$ 时 $B_{\mathrm{tok}} = 2^{27} \approx 1.3 \times 10^{8}$ tokens/step；同一份配方换到 $d = 512$ 的集群、 $m$ 减到 8， $B_{\mathrm{tok}}$ 不变——式 (5) 的正确用法就是固定分子端、调配三个乘数。

累加改变的是**更新频率**而非单次估计的质量（质量侧见 §3）。固定 token 预算 $D$ 时 optimizer step 数 = $D / B_{\mathrm{tok}}$ ： $m$ 扩大 8 倍，步数缩 8 倍。两个直接后果：

- **schedule 的步数轴**：warmup、cosine / linear decay 都按 optimizer step 索引（2.3）。累加倍数变了而 schedule 的步数参数不动，等价于把 warmup 与 decay 在 token 轴上拉长 $m$ 倍——一个 2000 步 warmup 在 $m = 1$ 与 $m = 8$ 下覆盖的 token 数差 8 倍。改 $m$ 必须同步改 schedule 步数，或干脆把 schedule 定义在 token 轴上。
- **LR 与 batch 联动**：等效 batch 变大后单次梯度估计更准，能承受的步长更大——小噪声区近似 $\eta \propto B$ 的 linear scaling rule 与其失效边界（critical batch size）归 2.1 §2 与 2.3 推导。本章只立规矩：调 $m$ 就是调 $B_{\mathrm{tok}}$ ，LR 与 schedule 都要联动；反过来，复现一份只报 tokens 口径的配方时，把 $B_{\mathrm{tok}}$ 按自己的 $d$ 与显存拆成 $b \cdot m$ 是自由的，优化轨迹在式 (3) 的意义上不变。

## 3. 顺序与状态陷阱：裁剪、噪声、优化器内部计数

式 (3) 的等价不等于「随便拆都对」。三个陷阱按踩中频率排：

**梯度裁剪必须作用于累加之后**。`clip_grad_norm_` 读的是 `.grad` 的当前值，正确顺序： $m$ 个 micro-batch 全部累加完（DP 下含一次 all-reduce，§4）→ clip → `step()` → `zero_grad()`。逐 micro-batch 裁剪不是任何意义上的近似——裁剪是按范数的非线性缩放：两个真实方向相反的 micro-batch 梯度 $g_1 = 0.9$ 、 $g_2 = -0.9$ （一维示意），全和为 0、不该动；逐段以阈值 0.5 裁剪得 $0.5 - 0.5 = 0$ ，此例侥幸相同；换成 $g_1 = 2$ 、 $g_2 = -0.1$ ：全和 $1.9$ 、按比例缩到阈值 1.0，方向不变；逐段裁得 $1.0 - 0.1 = 0.9$ ——方向与幅度都错了。一般地 $\mathrm{clip}\left(\sum_i g_i\right) \neq \sum_i \mathrm{clip}(g_i)$ ，逐段裁剪系统性扭曲更新。范数裁剪的公式与阈值选择归 2.4，本章只立顺序语义。fp16 loss scaling 场景同理：unscale 也必须发生在累加完成之后、clip 之前（scaling 机制归 1.3）。

**噪声口径：accumulation 不加噪也不降噪**。式 (3) 是精确等式：累加后的单次更新梯度与真大 batch 的估计是同一个量，协方差同为 $\Sigma / B_{\mathrm{tok}}$ （ $\Sigma$ = 单样本梯度协方差，推导归 2.1 式 (3)；梯度噪声尺度的背景见 [McCandlish et al., 2018](../references.md)，arXiv:1812.06162，[B]）。accumulation 不改变单次估计的方差，改变的是单位 token 的更新次数；噪声与步长的联合账全部归 2.1 / 2.3，本章结论只有一句：把 $m$ 调大在优化侧等价于把 batch 调大，没有任何额外的统计量代价或收益。

**优化器的步数计数不感知累加**。Adam 的 bias correction 因子 $1 - \beta^{t}$ 、二阶矩 EMA、warmup 状态全部按 `step()` 调用次数推进（2.2）。 $m$ 从 1 改到 8，优化器看到的「训练进度」按 step 计不变、按 token 计缩 8 倍——任何按 step 标定的量（bias correction、schedule、日志横轴、checkpoint 间隔）都不与 token 轴自动对齐，换算出错时曲线看起来「没问题」但配方已经变了。

**`.grad` 的生命周期**。每个 optimizer step 结束后 `zero_grad(set_to_none=True)` 一次：置 None 而非写零，下一次 `backward()` 重新分配——语义与清零等价，省一次全参数写。顺序颠倒（先 zero 再 clip / step）是第二常见的累加 bug，症状是参数完全不动或裁了个空。

**日志语义随之改变**。逐 micro-batch 打 loss 打出来的是 $m$ 个不同子集的平均，方差比整批 loss 大约 $m$ 倍（子集小 $m$ 倍、式 (4) 同权时），曲线显得毛糙；对账口径是累加期间把各 micro-batch 的 loss（已乘 $1/m$ ）求和、每个 optimizer step 记一次——这个数才是与「大 batch 训练曲线」可比的量。吞吐口径同理：tokens/s 按实际消费的 token 计（含累加步），与 optimizer step 数解耦。评估侧不需要累加：无反向就无激活驻留（`no_grad` / `inference_mode` 语义归 1.1），eval batch 可以拉到显存上限，与训练侧的 $b$ 无关。

## 4. 与数据并行的组合：`no_sync` 把通信除以 $m$

数据并行（DP）： $d$ 个副本各算各的梯度、再跨副本求平均——机制（bucket 化、反向与通信 overlap、hook 注册时机）的完整走读归 4.1，本章只取 API 语义。PyTorch DDP 的同步语义从实现读（pytorch/pytorch repo `torch/nn/parallel/distributed.py`，教学参照）：梯度就绪回调里执行「梯度除以 world size、再 all-reduce（SUM）」，各 rank 拿到的是跨 rank **平均**梯度。于是 DP + 累加的组合在数学上仍是同一个平均：

$$
g = \frac{1}{d} \sum_{r=1}^{d} g_r, \qquad g_r = \frac{1}{m} \sum_{i=1}^{m} \nabla \hat\ell_{r,i} \qquad (6)
$$

**符号**： $g_r$ ：rank $r$ 本地累加 $m$ 步后的梯度； $g$ ：一次同步后各 rank 的共同梯度。代入式 (2)： $g = \nabla \bar\ell$ ， $\bar\ell$ 跑遍全部 $d \cdot m \cdot b$ 条序列——等效 batch 再乘 $d$ ，与式 (5) 一致。这也说明 DP 与累加在数学上是同一个操作的两个旋钮：都往等效 batch 里乘因子，区别只在通信与显存的位置。

问题在通信轮次。naive 实现里每个 micro-batch 的反向都触发一轮梯度 all-reduce，一次 optimizer step 付 $m$ 轮。注意这不产生正确性 bug——梯度数值照样对——是纯性能浪费：每轮同步的都是「还没累加完的部分和」，等其返回的值还要被后续累加覆盖。DDP 的 `no_sync` 上下文管理器为此而设（[Li et al., 2020](../references.md)，arXiv:2006.15704，[A]，§3.2.4；语义经论文与源码 docstring 双重核实）：进入 context 时把 `require_backward_grad_sync` 置 False，梯度就绪回调直接返回、不发起通信；**前向也必须包在 context 内**（同步与否在前向时锁定，docstring 警告原话）；退出后第一个反向把累加好的梯度一次同步完。每个 optimizer step 恰好一轮：

```
时间 →     micro 1        micro 2        micro 3        micro 4
naive:   fwd + bwd      fwd + bwd      fwd + bwd      fwd + bwd
             │ AR           │ AR           │ AR           │ AR     ← 每 step 4 轮 all-reduce
no_sync: fwd + bwd      fwd + bwd      fwd + bwd      fwd + bwd
           (hook 关)      (hook 关)      (hook 关)         │ AR     ← 每 step 1 轮
                                                           ▼
                                             clip → step → zero_grad
```

通信账：设梯度张量字节数为 $M$ （bf16 梯度下 $M = 2N$ bytes），ring all-reduce 每 rank 每轮通信量 $\approx 2M$ （系数 $2(d-1)/d$ 的推导归 4.1）。固定 token 预算下 naive 的通信总量 $\propto m$ 轮 / step、`no_sync` 为 1 轮 / step——**通信总量除以 $m$** 。数值感： $N = 7 \times 10^{9}$ 、bf16 梯度 $M \approx 13$ GiB， $m = 8$ 时 naive 每步每 rank 约 $209$ GiB、`no_sync` 约 $26$ GiB，等效于把累加期间的同步开销整个抹掉。两个边界注记：其一，bucket 化 overlap（反向算到一半就开始通信，归 4.1）只发生在最后一次反向；前 $m - 1$ 次反向没有任何通信，也谈不上 overlap——no_sync 省的是通信本身，不只是暴露的通信。其二，单卡 $d = 1$ 时 `no_sync` 无对象可言，单机累加写普通循环（§5 第一段）即可。

与异步方案的对照值得立一句：累加 + `no_sync` 保持严格的同步语义——各 rank 在同一参数版本上工作、同一时刻提交同一份平均梯度，没有 staleness；这与「各 worker 异步推梯度」的路线（异步 SGD）在语义上完全不同，后者用陈旧梯度换通信吞吐，收敛行为不再是式 (6) 的平均。同步与异步的账归 4.1 展开，本章只需记住：累加是同步框架内的省通信手段，不改变优化轨迹的数学定义。

## 5. 实现走读：手动循环与 DDP `no_sync`

单机手动累加（教学用，非优化实现）：

```python
# 教学用，非优化实现。等价性：m 步累加 ≡ 大 batch 一次反向（式 (3)）
m = 16  # 累加步数；等效 batch = b * m（单卡 d = 1）
for x, y in micro_batches:                      # 每个 micro-batch: x [b, T]
    loss = criterion(model(x), y) / m           # 归一化位置 (a)：先除 m 再 backward
    loss.backward()                             # .grad 原地累加（1.1 AccumulateGrad）
clip_grad_norm_(model.parameters(), max_norm=1.0)  # 累加完之后才裁剪（§3）
optimizer.step()
optimizer.zero_grad(set_to_none=True)           # 每 optimizer step 清一次
```

DDP 版（教学用，非优化实现；DDP 内部机制归 4.1）：

```python
# 教学用，非优化实现。语义与论文 §3.2.4 的示例一致
for i, (x, y) in enumerate(micro_batches):
    if i < m - 1:
        with ddp.no_sync():                     # 前向必须包在 context 内（docstring 警告）
            loss = criterion(ddp(x), y) / m
            loss.backward()                     # hook 关闭：不 all-reduce，纯累加
    else:
        loss = criterion(ddp(x), y) / m         # 最后一个 micro-batch：反向时一轮 all-reduce
        loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)  # all-reduce 之后、step 之前
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

等价性自检（单机可跑，教学用，非优化实现）：

```python
# 教学用，非优化实现：断言累加梯度与大 batch 梯度一致（式 (3) 的数值验证）
import copy
import torch
torch.manual_seed(0)
model_acc = MyModel()
model_big = copy.deepcopy(model_acc)
X, Y = make_batch(B=32, T=128)                   # 造数据
loss_big = criterion(model_big(X), Y)            # 一次反向（32 条整批）
loss_big.backward()
m = 8
for xs, ys in zip(X.chunk(m), Y.chunk(m)):       # chunk 等长 ⇒ 式 (2) 等权前提满足
    (criterion(model_acc(xs), ys) / m).backward()
for p_big, p_acc in zip(model_big.parameters(), model_acc.parameters()):
    assert torch.allclose(p_big.grad, p_acc.grad, atol=1e-6, rtol=1e-5)
# fp32 下应通过；bf16 需放宽 atol——浮点加法顺序不同，位级本就不等（§1）
```

三段代码的次序就是全章的次序：除 $m$ 在反向之前，clip 在累加完成之后，zero_grad 在 step 之后；DP 版再多一条——前向与反向一起进 `no_sync` 的 context。次序全部来自前面的公式与顺序语义，没有一条是框架癖好。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 累加等价 | $m$ 份 micro-batch 平均的平均 = 大 batch 平均；梯度侧由线性性成立 | 式 (1)–(3) |
| 等权前提 | 定长 packed 序列精确成立；变长需按有效 token 数加权 | 式 (4) |
| 归一化位置 | $1/m$ 只放一处（loss 内 / 更新前；勿放 LR） | §1 |
| tokens 口径 | 等效 batch $B_{\mathrm{tok}} = b \cdot m \cdot d \cdot T$ ；schedule 按 step 索引，改 $m$ 必联动 | 式 (5)；2.3、7.2 |
| 噪声 | 单次估计协方差 = $\Sigma / B_{\mathrm{tok}}$ ，与真大 batch 相同；变的是更新频率 | 2.1 式 (3) |
| 裁剪顺序 | clip 在累加（DP 下含 all-reduce）完成之后；unscale 同位置 | §3；2.4、1.3 |
| DDP 组合 | `no_sync`：context 内关闭同步 hook，每 optimizer step 一轮 all-reduce，通信 ÷ $m$ | 式 (6)；arXiv:2006.15704 §3.2.4 |

## 练习

1.（推导）设 $m$ 个 micro-batch 的有效 token 数 $n_1, \ldots, n_m$ 不全等，micro-batch $i$ 的 loss 为其内部有效 token 平均 $\hat\ell_i$ 。(a) 证明：取权重 $w_i = n_i / \sum_j n_j$ 时 $\sum_i w_i \thinspace \nabla \hat\ell_i$ 等于全体有效 token 上的平均梯度。(b) 给一组 $n_i$ 与 $\hat\ell_i$ 的具体数值，使等权 mean-of-means 与全局 token 平均相差 20% 以上。(c) 说明定长 packed 序列为何自动免除该问题。
2.（计算） $N = 7 \times 10^{9}$ 参数、bf16 梯度、 $d = 512$ 卡、 $m = 8$ 。(a) 按 ring all-reduce 每轮每 rank $2(d-1)/d \cdot M$ 字节（推导归 4.1），算 naive 与 `no_sync` 两种方案每个 optimizer step 的每 rank 通信量。(b) 设单轮 all-reduce 耗时 200 ms、每个 micro-batch 前向加反向 1 s，两种方案通信占 step 时间的比例各是多少？(c) 同一模型改走 $m = 1$ 并把 $b$ 扩大 8 倍（假设显存装得下）：通信量与 (a) 何异？
3.（计算）某配方报告「global batch size 1024 序列、序列长 8192」，训练代码为 $b = 2$ 、 $d = 64$ 。(a) 求 $m$ 与 $B_{\mathrm{tok}}$ 。(b) 若把 $m$ 减半、其余不动，token 预算不变时 optimizer step 数如何变？cosine schedule 的总步数参数应如何改才能保持 token 轴上的 decay 形状？
4.（代码）跑通 §5 的等价性自检；然后把 criterion 换成带 padding mask 的 token 平均 loss，人为让各 micro-batch 有效 token 数不等（如末段补 pad），分别用等权累加与练习 1 (a) 的加权累加，用 `torch.allclose` 判定哪个能对上整批 token 平均口径的梯度。

## 参考

- [Li et al., 2020](../references.md)：PyTorch Distributed: Experiences on Accelerating Data Parallel Training，arXiv:2006.15704，[A]（key: pytorch-ddp）——§3.2.4 gradient accumulation 与 `no_sync` 的出处
- [McCandlish et al., 2018](../references.md)：An Empirical Model of Large-Batch Training，arXiv:1812.06162，[B]（key: critical-batch）——梯度噪声 $\Sigma / B$ 背景，2.1 / 2.3 展开
- [PyTorch, pytorch/pytorch repo](../references.md)（key: pytorch-repo，教学参照）——`torch/nn/parallel/distributed.py`：`no_sync` 上下文管理器（`require_backward_grad_sync` 门控）与梯度回调的「除 world size 后 all-reduce」平均语义
