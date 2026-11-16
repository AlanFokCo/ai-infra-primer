# 4.2 ZeRO：优化器状态、梯度、参数的逐阶切分（ZeRO: Stage-wise Partitioning of Optimizer States, Gradients, and Parameters）

> 4.1 结尾的账是本章的起点：DDP 每 rank 复制全量模型状态 $16N$ bytes，卡越多浪费越大。ZeRO 把三类状态按 DP 域逐阶切片——优化器状态（ZeRO-1）→ 梯度（ZeRO-2）→ 参数（ZeRO-3）——本章逐阶推导各自省什么、付什么：ZeRO-1/2 与 DP 通信量严格相等的论证、ZeRO-3 参数取回时序与通信 ×1.5 的推导、ZeRO-Offload / Infinity 把状态搬去 CPU/NVMe 的账（PCIe 下界与聚合带宽）、ZeRO++ 把跨节点通信 $3M$ 压到 $0.75M$ 的三件组件，最后走读 DeepSpeed 官方实现（`deepspeed/runtime/zero/` 的文件级结构）并给各阶 × 模型规模 × 卡数的配置决策表 | 前置：4.1（ring all-reduce 账、DDP 冗余）、1.3（混合精度三件套）、1.4（激活账——本章不碰）、models 1.2（ $16N$ 三态分解，式 (8)） | 锚点：ZeRO 论文（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]）、ZeRO-Offload（[Ren et al., 2021](../references.md)，arXiv:2101.06840，[A]）、ZeRO-Infinity（[Rajbhandari et al., 2021](../references.md)，arXiv:2104.07857，[A]）、ZeRO++（[Wang et al., 2023](../references.md)，arXiv:2306.10209，[A]）、deepspeedai/DeepSpeed repo（[A]，机制级走读）

## 1. 冗余在哪：DP 的三态全量复制

bf16 混合精度 + AdamW 下，一个 rank 要常驻的模型状态是四笔：bf16 工作参数 $2N$ 、bf16 梯度 $2N$ 、fp32 master 参数 $4N$ 、fp32 一阶/二阶矩估计 $8N$ ，合计 $16N$ bytes——分解与逐项对账表已在 models 1.2 式 (8) 交付，本章只登记结论、不重复推导；激活是另一笔独立的账（随 $B \cdot T$ 线性），归 1.4。ZeRO 论文的记号是 $\Psi$ （参数量）与 $K$ （优化器状态字节乘数）：Adam 混合精度下 $K = 12$ （fp32 master 4 + 矩估计 4 + 4），模型状态总量写成 $(4 + K)\Psi = 16\Psi$ bytes，与本书记号的 $16N$ 是同一件事（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]，§3；论文假设 fp16 + fp32 master，bf16 字节数相同）。

DP（4.1）对这 $16N$ 一字节不省： $d$ 个 rank 各自复制全量，集群总显存 $d \times 16N$ 只装一份模型状态。ZeRO 的切入角度是：冗余不在「数据并行」这个概念里，而在状态的**存放方式**——三类状态的可切性与切分代价各不相同，于是切成三阶（论文记号 $P_{os}$ / $P_g$ / $P_p$ ，累进开启）：

| 方案 | 每 rank 模型状态 | 通信量 vs DP | 渐近（ $d$ 大） |
| --- | --- | --- | --- |
| DP 基线 | $16N$ | $2N$ 元素（4.1 式 (2)） | $16N$ |
| ZeRO-1（ $P_{os}$ ） | $4N + 12N/d$ | 相同 | $\approx 4N$ |
| ZeRO-2（+ $P_g$ ） | $2N + 14N/d$ | 相同 | $\approx 2N$ |
| ZeRO-3（+ $P_p$ ） | $16N/d$ | ×1.5 上限 | $16N/d$ 线性 |

各行推导在下面四节。论文 Table 1 的标尺数字（ $K = 12$ 、32 GB V100 口径）：7.5B 模型在 $d = 64$ 下 DP 需 120 GB，ZeRO-1 降到 31.4 GB、ZeRO-2 16.6 GB、ZeRO-3 1.88 GB；1T 模型在 $d = 1024$ 下 ZeRO-3 每 rank 15.6 GB——万亿参数在千卡上可训练。本章习题 2 复核这些数字。

ZeRO 论文还有 ZeRO-R 一支：激活切片（ $P_a$ ）、临时缓冲定容（ $P_b$ ）、显存碎片治理（ $M_D$ ）。激活侧主线在本书归 1.4（gradient checkpointing 与 selective recompute）， $P_a$ 相对模型并行的通信量分析在论文 §8，本章不展开。

## 2. ZeRO-1（ $P_{os}$ ）：切优化器状态

**机制**（论文 §5.1）：把优化器状态（fp32 master、 $m$ 、 $v$ ）均分成 $d$ 份，rank $r$ 只存第 $r$ 份，也只负责更新对应的 $1/d$ 参数分区。一个 step：

1. 各 rank 本地前向反向，得到全量本地梯度（参数仍全量持有，前向反向不需要别人的状态）；
2. 梯度归约：rank $r$ 只需要**自己分区**的平均梯度——梯度通信从 all-reduce 换成 reduce-scatter，各 rank 收到自己分区对应的平均梯度块；
3. 各 rank 在自己的 fp32 master 分片上跑 optimizer 更新；
4. 更新后的参数分区必须全员可见：对参数做一次 all-gather，各 rank 拼回全量新参数，进入下一 step。

**显存账**（论文 §5.1，换本书记号）：

$$
\mathrm{Mem}_1 = \underbrace{2N}_{\text{bf16 参数}} + \underbrace{2N}_{\text{梯度}} + \underbrace{12N \big/ d}_{\text{优化器状态分片}} = 4N + \frac{12N}{d} \qquad (1)
$$

**符号**： $12N$ = fp32 master $4N$ + 两阶矩 $8N$ （ $K = 12$ 口径）； $d$ ：DP 度数。 $d$ 大时 $`\mathrm{Mem}_1 \approx 4N`$ ，相对 DP 的 $16N$ 省 4 倍（论文 Figure 1 / §5.1 的 4x）。代数字：7.5B @ $d = 64$ ： $4 \times 7.5 + 12 \times 7.5 / 64 = 30 + 1.4 = 31.4$ GB，与论文 Table 1 逐项一致。

**通信量论证（本节的关键推导）**。基线 DP 的梯度 all-reduce 在带宽最优实现（ring）下本身就是「reduce-scatter + all-gather」两段（4.1 §2.2），每 rank 合计移动 $2(d-1)/d \cdot N \approx 2N$ 个元素。ZeRO-1 的第 2 步恰好就是一次 reduce-scatter（每 rank 约 $N$ 个元素），第 4 步恰好就是一次 all-gather（同样约 $N$ 个元素）——**两个集合操作与基线逐一同构，每 rank 通信量严格相等**；区别只在第二阶段搬运的内容从「归约后的梯度」换成「更新后的参数」（同为 bf16、同 $N$ 个元素，字节数也相同）。这就是论文 §7.2.1 的结论： $P_{os}$ 与 $P_{os+g}$ 「same communication volume as DP」。DP 用通信搬梯度、ZeRO-1 用同样的通信搬参数——通信预算一文未涨，换来的是优化器状态显存 $\div d$ 。

```
       前向反向      梯度归约             本地更新          参数同步
DP:     本地 ─────► all-reduce ────────► 全量参数更新 ────► （无）
                    (RS + AG，2N 元素)
ZeRO-1: 本地 ─────► reduce-scatter ────► 1/d 分区更新 ───► all-gather 更新后参数
                    (N 元素)                                (N 元素)
        通信量：两者每 rank 各 2N 元素，严格相等
```

实现注记一句：第 2 步若不用 reduce-scatter、改用「梯度全量 all-reduce 后各取自己那片」，每个 step 反而多付一次 $N$ 元素——all-reduce 本身是 RS+AG 两段共 $2N$ ，而参数 all-gather 仍不可免（各 rank 只更新了自己的 fp32 分片，下一 step 前向必须拿到全量新参数），合计 $3N$ 对 RS 变体的 $2N$ ，贵 1.5 倍。这正是 DeepSpeed stage 1/2 默认走 reduce-scatter 的原因（`deepspeed/runtime/zero/stage_1_and_2.py` 构造参数 `reduce_scatter=True` 为默认，§6）。省不省梯度显存是 ZeRO-2 的事，与这里的通信选择无关。

## 3. ZeRO-2（+ $P_g$ ）：再切梯度

ZeRO-1 里每个 rank 仍要实例化全量梯度缓冲（ $2N$ ）才能喂给归约。ZeRO-2 把梯度也按分区管理（论文 §5.2）：既然 rank $r$ 最终只需要第 $r$ 分区的归约梯度，**非己分区的梯度块在归约完成后立即释放**——反向每算出一批梯度，按 bucket 聚成块、reduce-scatter 到 owner rank，其余 rank 的这部分缓冲随即作废，不再需要等到整个反向结束。梯度显存从 $2N$ 降到 $2N/d$ 加一个 bucket 粒度的工作缓冲：

$$
\mathrm{Mem}_2 = 2N + \frac{2N}{d} + \frac{12N}{d} = 2N + \frac{14N}{d} \qquad (2)
$$

$d$ 大时 $\approx 2N$ ，相对 DP 省 8 倍（论文 §5.2 的 8x）；7.5B @ $d = 64$ ： $15 + 14 \times 7.5/64 = 15 + 1.64 = 16.6$ GB（Table 1 一致）。通信量与 ZeRO-1 逐项相同——reduce-scatter 照发、参数 all-gather 照发；变的只是非己梯度块的**驻留时间**（从整个 step 缩到 bucket 粒度）。论文把这一步的桶化与 AMP 优化器的梯度桶化对照：同为 bucket 粒度 overlap 通信与计算，ZeRO-2 更进一步，在分区边界上做的是归约（reduce-scatter）而非 all-reduce，归约完即释放，overlap 与省显存一举两得。一个口径注记：梯度以 bf16 还是 fp32 参与归约是另一个轴（DeepSpeed 的 `communication_data_type` 可调），本章的账按 bf16 梯度 $2N$ 口径，与 1.3 的混合精度配置一致。

ZeRO-1 与 ZeRO-2 合起来回答了「为什么前两阶通信免费」：DP 的一次 all-reduce 本来就可以拆成 reduce-scatter + all-gather 两次各 $N$ 元素的通信；ZeRO 做的只是让这两段通信分别落在「梯度归约」与「参数同步」两个本来就需要的事件上，并把第二段的内容换成更新后的参数。通信模式一字未改，显存已经从 $16N$ 降到 $\approx 2N$ 。

## 4. ZeRO-3（+ $P_p$ ）：再切参数

### 4.1 机制与取回时序

ZeRO-3 把参数本体也切掉：rank $r$ 只**常驻**自己的 $2N/d$ bf16 参数分片（以及对应的 fp32 master 分片与矩估计分片）。前向算到某一层时，该层全量参数不在本地——先把它从各 owner 手里聚齐（all-gather），用完立即释放、只留自己的分片；反向时再聚齐一次（逆序），算完该层梯度后，对梯度做 reduce-scatter 归并到各 owner。一个 step 的取回时序（论文 §5.3 与 §7.2.2 的口径：「把参数 all-gather 重新调度、摊平到整个前向里，用完即弃；反向再聚一次」）：

```
层序(前向→):  L1        L2        L3        L4
参数:       AG(L1)    AG(L2)    AG(L3)    AG(L4)      ← 算到谁聚谁（可预取下一层）
              │fwd      │fwd      │fwd      │fwd
             释放      释放      释放      释放        ← 用完即弃，回到 1/d 常驻
层序(反向←):  L4        L3        L2        L1
参数:       AG(L4)    AG(L3)    AG(L2)    AG(L1)      ← 反向再聚一次（逆序）
              │bwd      │bwd      │bwd      │bwd
梯度:        RS(L4)    RS(L3)    RS(L2)    RS(L1)     ← 梯度归并到各 owner
常驻显存:   ── 自己的 2N/d 参数分片 + 12N/d 优化器分片 + 2N/d 梯度分片
              + 当前层全量参数（取回期间）+ 预取缓冲 ──
```

AG = 参数 all-gather，RS = 梯度 reduce-scatter。论文 §5.3 的用词是 broadcast（owner 广播自己的分片），§7.2.2 把它等价重述为「摊在整个前向里的参数 all-gather」；DeepSpeed 的实现走 all-gather（§6.2）。两种表述的通信量同价：每参数恰好被传送一次、每 rank 收到全量，即每 rank 接收 $N$ 个元素。

反向为什么必须再见一次参数：1.2 的 matmul 反向给出答案——层 $Y = XW$ 的反向需要 $\partial L / \partial X = G W^{\top}$ 与 $\partial L / \partial W = X^{\top} G$ 两个同形 matmul，输入梯度的计算离不开全量 $W$ 。所以反向不是「顺手用到」参数，而是每层梯度计算的硬输入；前向用完即弃的参数，反向只能重新 gather。这就是时序图上第二次 AG 的来源，也是 ×1.5 里那 0.5 的构成。

**峰值显存注记**： $16N/d$ 是常驻口径；任一时刻还要加上当前计算层的全量参数与预取中的下一层——这部分**不随 $d$ 缩**。模型存在特大单层时这项会露头，配套的工程手段是层内再切（ZeRO-Infinity 的 memory-centric tiling，§5.2）或按参数粒度的分片管理（FSDP 的 flat-param 语义，归 4.3）。峰值公式的完整推导归 4.3 与 4.4 的总账。

### 4.2 通信量推导：×1.5 从哪来

逐项数一个 step 内每 rank 的通信量（以 bf16 元素计，参数与梯度同为 $N$ 个元素）：

1. **前向参数 all-gather**：全模型的每个参数在前向中恰好被 gather 一次 → 约 $N$ 个元素；
2. **反向参数 all-gather**：反向再需要一遍全量参数（逆序）→ 再 $N$ 个元素；
3. **梯度 reduce-scatter**：与 ZeRO-1/2 相同 → $N$ 个元素；
4. **更新后参数同步：零**。各 rank 在自己的 fp32 master 分片上更新、写回自己的 bf16 分片即可——下一个 step 的前向反正要重新 gather，不需要把更新后的参数当场推给任何人（对照 ZeRO-1 第 4 步的那次 all-gather，这里省掉了）。

合计 $3N$ 个元素，对照 DP / ZeRO-1 / ZeRO-2 的 $2N$ ：

$$
\frac{C_{\mathrm{ZeRO\text{-}3}}}{C_{\mathrm{DP}}} = \frac{3N}{2N} = 1.5 \qquad (3)
$$

这就是论文 §7.2.2 的「maximum of 1.5× communication」。两个方向把这个数读全。其一，1.5 是**上限**口径（论文原话 "maximum of"）：参数 gather 摊进前向反向、与计算重叠（算当前层时预取下一层，§6.2 的实现正是这个结构），暴露在关键路径上的通信增量可以小于账面。其二，这 0.5 倍通信换来的是渐近 $16N/d$ 的**线性**显存下降——DP 度数从此变成显存杠杆：卡越多每卡越省，与 DP「卡越多显存越浪费」恰好相反。7.5B @ $d = 64$ 的模型状态从 ZeRO-2 的 16.6 GB 降到 1.88 GB，同一模型在 $d = 1024$ 时是 0.12 GB（论文 Table 1）。

### 4.3 为什么 ZeRO-3 只多付一次 gather

把 ZeRO-3 与 ZeRO-1 摆在一起对照，1.5 的结构就透明了：ZeRO-1 的参数 all-gather 发生在 optimizer step 之后、下个前向之前，一次把 $N$ 个元素推齐并让每 rank **常驻**全量；ZeRO-3 不付这一次，代价是前向与反向各付一次按需 gather——账面从 $2N$ 变 $3N$ ，多出来的 $N$ 就是「反向需要再看一遍参数」的价格。而前向那次 gather 与 ZeRO-1 的「更新后 all-gather」在语义上是同一件事（让每个 rank 拿到非己参数的可读副本），区别只在时机（用时取 vs 更新后推）与驻留（用完即弃 vs 常驻）。这个对照说明 ZeRO 三阶不是三个算法，而是同一条「状态随用随取」光谱上的三个档：切得越狠，常驻越少、临时通信越多——到 ZeRO-3，常驻压到 $16N/d$ ，通信停在 1.5 倍封顶。

### 4.4 与 gradient accumulation 的交互：参数通信恒不摊薄

1.5 / 4.1 的 `no_sync` 把 DP 的梯度同步从每 micro-batch 一轮压到每 optimizer step 一轮，通信总量除以 $m$ 。这个省法依赖一个前提：全量梯度缓冲就在本地， $m$ 次反向的累加是纯本地操作（1.1 的 `.grad` 原地累加语义）。分片之后，这个前提变成显存与通信的交换面——梯度侧与参数侧分开看：

- **梯度归约：÷ $m$ 可行，代价是 $2N$ 全量梯度缓冲**。FSDP 论文把累加分成两个变体（[Zhao et al., 2023](../references.md)，arXiv:2304.11277，[A]，§3.3.4）：with-communication 变体照常跨 rank 归约、各 rank 保存分片梯度；without-communication 变体不跨 rank 归约、各 rank 保存**未分片**的全量梯度，通信推迟到累加边界一次付清——后者「用显存换通信」。映射到 ZeRO 各阶：ZeRO-1 的全量梯度缓冲本来就在其显存口径（ $4N + 12N/d$ 含 $2N$ 梯度）内，累加照旧本地进行、归约与参数 all-gather 推迟到边界，÷ $m$ 摊薄与 DP + `no_sync` 同构；ZeRO-2/3 的**保梯度显存配置**（DeepSpeed stage3 形态）走的是 with-communication 路线——归约随每个 micro-batch 发生、在 owner 的分片上原地累加（stage3 实现：分区梯度在 `__param_id_to_grad_partition` 里跨 micro-batch 累加，边界处收尾 `finalize_gradient_accumulation_boundary`），每 micro-batch 付一次梯度 RS（ $N$ 元素）、每 step $mN$ ；要 ÷ $m$ 就得回到全量缓冲形态，把 ZeRO-2/3 省下的梯度显存整个让回。
- **参数取回：任何配置下都不摊薄**。ZeRO-3 的前向 AG + 反向 AG 是每个 micro-batch 的硬需求（数据不同、参数都要用，§4.1 时序）， $m$ 个 micro-batch 恒付 $2mN$ 元素，与梯度侧选哪种缓冲形态无关。ZeRO-1/2 参数常驻，无此项。

合起来的结论：DP + `no_sync` 每 step 通信 $2N$ 元素；ZeRO-3 在保梯度显存配置（DeepSpeed stage3 形态）下每 step $3mN$ （每个 micro-batch 各付 $3N$ ），等效 batch 相同时是 DP 的 $1.5m$ 倍；就算梯度侧改走 without-communication 变体，参数取回的 $2mN$ 也纹丝不动。式 (3) 的 1.5 是**每个 micro-batch** 的口径；做大 batch 训练时这条换算必须带上。工程对策不是少吃通信，而是把单 step 的计算拉长（大 $b$ 、小 $m$ ）让通信占比下降，或上 ZeRO++ 把单次通信本身砍小（§5.3）。

## 5. ZeRO-Offload 与 ZeRO-Infinity：把状态搬出 GPU

### 5.1 ZeRO-Offload：唯一最优卸载策略与 PCIe 下界

切分再狠， $d$ 张卡的总显存还是 GPU 显存。ZeRO-Offload（[Ren et al., 2021](../references.md)，arXiv:2101.06840，[A]）把账做到 CPU 内存上，其推导框架值得完整展开。把一步训练画成数据流图（论文 §3.1）：圆形节点 = 五类模型状态（fp16 参数、fp16 梯度、fp32 master、fp32 $m$ 、fp32 $v$ ），方形节点 = 三类计算（前向、反向、参数更新），边权 = 该边一个 step 流过的字节数（fp16 源 $2N$ bytes、fp32 源 $4N$ bytes；论文记参数量为 $M$ 、边权写 $2M$ / $4M$ ，本章统一用 $N$ ）。一种 offload 策略 = 该图在 GPU/CPU 间的二划分，跨切边的边权之和就是每 step 的 CPU↔GPU 通信量。三条第一性约束依次收紧（论文 §3.2-3.4）：

1. **CPU 算力约束**：前向反向的计算复杂度是 $O(N \cdot B)$ （ $B$ 为等效 batch），参数更新是 $O(N)$ ——CPU 吞吐差 GPU 多个量级，只有 $O(N)$ 的计算可以下放。前向反向合并成 FWD-BWD 超节点，钉在 GPU 侧。
2. **通信下界**：合并后图中每个节点都在环上，任何二划分至少切两条边、每条边权至少 $2N$ → **任何模型状态 offload 策略的 CPU↔GPU 通信量下界是 $4N$ bytes/step**（论文 §3.3；本书注记：部分状态卸载可低于此界，此处按整状态卸载口径）。进一步，fp32 状态必须与其生产者/消费者同侧——否则必切到一条 $4N$ 边、总量至少 $6N$ ——于是 fp32 三态与参数更新合并成 Update 超节点。
3. **显存最大化**：fp16 参数与 FWD-BWD 之间的边权是 $4N$ （前向读 + 反向读各 $2N$ ），把参数放到 CPU 会把通信推到 $6N$ → fp16 参数钉在 GPU；剩余的 fp16 梯度与 Update 超节点全放 CPU。

三条约束依次收紧后，合法的划分只剩四种（论文 Table 1， $M$ 为论文的参数量记号、即本书 $N$ ；显存为 GPU 侧口径）：

| FWD-BWD | p16 | g16 | Update 超节点 | GPU 显存 | 相对基线 |
| --- | --- | --- | --- | --- | --- |
| GPU | GPU | GPU | GPU | $16N$ | 1x（基线） |
| GPU | GPU | CPU | GPU | $14N$ | 1.14x |
| GPU | GPU | GPU | CPU | $4N$ | 4x |
| GPU | GPU | CPU | CPU | $4N$ | 8x |

收敛到**唯一最优策略**（论文 §3.5）：fp16 参数与前向反向在 GPU；fp16 梯度、全部 fp32 状态、参数更新在 CPU。GPU 常驻只剩 fp16 参数（ $2N$ ）加一块有界的梯度暂存——梯度逐块产出即分组传往 CPU、与剩余反向重叠（论文 §4.1）。口径注记：上表最后一行的「 $4N$ 」与「 8x 」在算术上不一致（ $16N / 4N = 4$ ）；论文 §4.1 的机制描述（梯度产出即传、GPU 只留小块暂存）与 8x（即 $\approx 2N$ ）自洽，本书按机制口径理解，表格照录原文。通信量踩在下界上：梯度 $2N$ 下行、更新后的参数 $2N$ 上行（CPU 侧更新 fp32 master 后 cast 回 fp16 传回），合计恰为 $4N$ 。CPU 侧的执行是重写的 Adam：SIMD 向量指令 + 循环展开 + OpenMP 多线程，论文口径比 PyTorch 优化器快 6 倍（§5.1）；CPU 更新在小 batch 下追不上 GPU 时，用**一步延迟参数更新**（DPU，论文 §5.2）：前若干步正常，进入 DPU 后第 $t$ 步 CPU 用第 $t-1$ 步的梯度更新、GPU 同时跑第 $t$ 步的前向反向，把 CPU 时间藏进 GPU 计算，论文报告端到端吞吐至多再提 1.5 倍且不改收敛。

瓶颈就是 PCIe 这一条窄管：论文测试平台（DGX-2）的 GPU↔CPU 双向带宽 32 GB/s（其 Table 2）；ZeRO-Infinity 论文给了更细的分解——CPU 内存带宽约 100 GB/s、NVMe 约 25 GB/s 都不慢，慢在每块 GPU 到 CPU/NVMe 之间可用 PCIe 带宽只有约 10-12 GB/s（[Rajbhandari et al., 2021](../references.md)，arXiv:2104.07857，[A]，§4 及其脚注）。每 step $4N$ bytes 的搬运量除以这条带宽就是 offload 的时间地板：10B 模型为 $4 \times 10^{10}$ bytes ≈ 1.25 s（32 GB/s 口径）——只有单 step 计算时间足够长（batch 足够大），这笔搬运才藏得住（练习 4 算这个下界）。论文实测口径：单卡 V100 训 10B 模型 40 TFLOPS（对照 PyTorch 单卡不 offload 的最大模型——贡献清单口径 1.2B、摘要口径 1.4B——的 30 TFLOPS），单 GPU 可训规模超 13B（约 10 倍），与 Megatron 式模型并行叠加后单 DGX-2 节点超 70B（4 倍于单独 MP），多卡到 128 GPU 近似线性扩展（摘要与 §6 口径）；多卡形态是 ZeRO-Offload 与 ZeRO 数据并行组合——梯度先跨卡归约、各卡只下传自己分区的梯度（其 §4.2）。

### 5.2 ZeRO-Infinity：NVMe 与聚合带宽

ZeRO-Infinity（[Rajbhandari et al., 2021](../references.md)，arXiv:2104.07857，[A]）把 offload 再推一层到 NVMe，并正面回答上节的单链路瓶颈。五个组件（论文 §1 清单）：**infinity offload engine**（同时利用 GPU/CPU/NVMe 三层内存与 GPU/CPU 两侧算力的异构卸载）、**memory-centric tiling**（特大单算子沿维切小来算，免去模型并行）、**bandwidth-centric partitioning**、**overlap-centric design**、**ease-inspired implementation**（免模型代码重构）。核心机制是 bandwidth-centric partitioning：单块 GPU 面对 CPU/NVMe 只有约 10-12 GB/s 的 PCIe，但 $d$ 块 GPU 各有自己的一条——把卸载状态的读路径组织成「各卡从慢介质并行取自己那份、再经 GPU 间链路交换」，让**跨设备聚合带宽**（而非单链路带宽）供给每块卡；配合全链路重叠（GPU-GPU 通信、CPU-GPU、NVMe-CPU 全部与计算 overlap）。实测口径：32 个 DGX-2 节点（512 块 V100）上训 32 万亿参数模型、吞吐超 25 PFLOPS；论文 Table 3 的前瞻论证：加速器算力再涨 10 倍，512 卡集群每卡也只需约 30 GB/s 的慢内存带宽即可保持高效——聚合法把「慢内存」从瓶颈变成容量池。

### 5.3 ZeRO++：把通信本身砍小

ZeRO-3 的 $3N$ 还能再压。ZeRO++（[Wang et al., 2023](../references.md)，arXiv:2306.10209，[A]）的三件组件各自对应 ZeRO-3 的一次集合通信（以下 $M$ 为参数量、数字为跨节点通信量口径，论文原文）：

1. **qwZ（量化权重通信）**：前向参数 all-gather 之前把权重从 FP16（2 字节）块量化到 INT8（1 字节），通信量 $M \to 0.5M$ ；量化按参数子块独立做（block-based，保训练精度），配套自研量化 CUDA kernel。
2. **hpZ（分层权重分区）**：反向的参数 all-gather 改为只在节点内做——每台机器内常驻一份完整权重副本（拿显存换通信），跨节点通信 $M \to 0$ 。
3. **qgZ（量化梯度通信）**：梯度归约换成 all-to-all 式的新范式——梯度按块量化到 INT4 传输、**归约之前恢复全精度**（直接在低精度上做归约会累积放大误差，这是不能简单量化 reduce-scatter 的原因）；并利用节点内带宽远高于节点间的事实，先节点内归约再跨节点，跨节点量 $M \to 0.25M$ 。

合计跨节点通信从 $3M$ 压到 $0.75M$ （ $0.5 + 0 + 0.25$ ），即相对 ZeRO-3 的 4 倍跨节点压缩（论文摘要/§1 口径）。适用画像即论文 §1.1 点名的两个场景：跨节点链路低带宽的集群（云环境里跨节点常是 100 Gbps 以下的以太网，论文口径）与每卡小 batch（等效 batch 撞收敛上限、卡数又多的情形——与 §4.4 的累加账同源）；节点内带宽富余被 hpZ 的副本显存买来用。

## 6. DeepSpeed 实现走读

ZeRO 的官方实现是 DeepSpeed（[DeepSpeed, deepspeedai/DeepSpeed repo](../references.md)，[A]；组织已由 microsoft/DeepSpeed 更名为 deepspeedai/DeepSpeed，旧名 301 重定向；下列路径与类名核实于 2026-11-17）。配置面先立住：`zero_optimization.stage` 取 1/2/3（`deepspeed/runtime/zero/config.py` 的 `ZeroStageEnum`：`1 = optimizer_states`、`2 = gradients`、`3 = weights`——枚举名即「这一阶新切什么」）；`offload_optimizer` 与 `offload_param` 各挂 `{device: cpu|nvme}` ，是 Offload/Infinity 的落地开关；stage 3 的取回粒度由 `stage3_prefetch_bucket_size`（默认 $5 \times 10^{7}$ 元素）与 `stage3_param_persistence_threshold`（默认 $10^{5}$ ）控制。

**Stage 1/2**（`deepspeed/runtime/zero/stage_1_and_2.py`，类 `DeepSpeedZeroOptimizer`）：梯度按 ipg bucket 聚合，`average_tensor` 默认走 reduce-scatter 分支（构造参数 `reduce_scatter=True`）把每桶梯度归约到 owner rank；owner 在自己的 fp32 分片上更新，再把更新后的参数分片 all-gather 回全员——§2-3 的算法逐项落地。

**Stage 3** 分三个文件看，与 §4 的时序一一对上：

- **`stage3.py`**（类 `DeepSpeedZeroOptimizer_Stage3`）：构造期 `_create_fp16_partitions_with_defragmentation` 把每个参数组摊平成一条连续 fp16 缓冲、按 rank 切成连续区间（每 rank 拿一段），配套 `_create_fp32_partitions` 建 fp32 主分片；反向期 `create_reduce_and_remove_grad_hooks` 装梯度 hook，梯度进 ipg bucket（`__add_grad_to_ipg_bucket`，在独立 reduce 流上拷入连续缓冲），桶满即 `__avg_scatter_contiguous_grads` 发 reduce-scatter、`partition_grads` 把归约结果落到各 rank 的分区缓冲并释放非己部分——§4 时序图里 RS(L4)…RS(L1) 的实现体。`step()` 按子组循环：`_prepare_sub_group`（offload 时把该组优化器状态换回）→ unscale/clip → `_optimizer_step`（offload 时走 `DeepSpeedCPUAdam`，且有一条 `_cpu_adam_step_and_copy_back` 快路径：CPU kernel 更新 fp32 主分片的同时直接写出 bf16 分片拷回 GPU）→ `_reassign_or_swap_out_partitioned_parameters`（更新后的 bf16 分片写回常驻位）→ `_release_sub_group`。
- **`parameter_offload.py`**（类 `DeepSpeedZeRoOffload`）：给每个 submodule 注册前向 pre-hook 与 post-hook——`pre_sub_module_forward_function` 调 `fetch_sub_module(sub_module, forward=True)` 取回该模块参数，`post_sub_module_forward_function` 调 `release_sub_module` 用后释放；反向的 pre/post 通过在反向图中插入 autograd.Function shim（`_run_before_backward_function` / `_run_after_backward_function`）触发；根模块的 pre-hook 每个 step 调 `reset_step` 重置调度器。
- **`partitioned_param_coordinator.py`**（类 `PartitionedParameterCoordinator`）：取回调度的核心，其 `fetch_sub_module` 的 docstring 把策略写成三步——先发当前模块参数的取回、再发后续若干模块的预取、最后阻塞等当前模块就绪。`__all_gather_params_` 对状态为 `NOT_AVAILABLE` 的参数在独立 CUDA 流上发异步 `all_gather_coalesced`，状态转 `INFLIGHT` 、句柄进 `InflightParamRegistry`，使用前阻塞等待转 `AVAILABLE`；`release_sub_module` 把不再被活跃模块引用的参数释放回分片态。预取的正确性靠 **trace**：第一轮迭代逐模块记录执行顺序（`record_module`），之后凭完整 trace（`is_complete_trace`）决定预取谁、释放谁；控制流变化会使 trace 失效并重录。NVMe 卸载的分片另有 `__prefetch_nvme_param_partitions` 提前搬。

教学骨架（教学用，非优化实现）把 §4.1 的时序写成单机可读的伪代码：

```python
# 教学用，非优化实现：ZeRO-3 一个 step 的取回-使用-释放循环（逻辑模拟 d 个 rank）
for layer in layers:                            # 前向
    w = all_gather(shard_of(layer))             # AG(L)：聚齐该层全量参数
    x = layer.forward(x, w)
    free(w)                                     # 用后释放，回到 2N/d 常驻
for layer in reversed(layers):                  # 反向
    w = all_gather(shard_of(layer))             # 再 AG 一次（逆序）
    gx, gw = layer.backward(g_out, w)
    free(w)
    my_gw = reduce_scatter(gw)[rank]            # RS：梯度归并到自己的分片
    update_shard(fp32_shard_of(layer), my_gw)   # 本地更新 1/d 的 fp32 分片
```

与真实实现的差别：真实栈把 AG/RS 全部异步化、按 trace 预取、bucket 聚合、独立流上与计算重叠，并处理 MoE 参数组、持久参数（`ds_persist`）、量化权重（ZeRO++ 路径）等分支；骨架只保留语义骨架。

## 7. 配置决策：各阶 × 模型规模 × 卡数

论文 Table 1 的推算数字先行（ $K = 12$ 、每 rank 模型状态 GB、32 GB V100 口径，粗体系本书标注的「装得下」）：

| 模型 | $d$ | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- | --- |
| 7.5B | 64 | 120 | **31.4** | **16.6** | **1.88** |
| 128B | 64 | 2048 | 536 | 284 | **32** |
| 1T | 1024 | 16000 | 4011 | 2013 | **15.6** |

决策规则按「哪堵墙先撞」排序：

1. ** $16N$ 连同激活装得下** → 纯 DDP 最省（无取回调度开销）；但优化器状态占 $12N/16N$ 是大头，ZeRO-1/2 以相同通信量换 4-8 倍显存下降，DeepSpeed 系栈里几乎总该开。
2. **撞模型状态显存墙** → 逐阶加深：ZeRO-1（省约 4x）→ ZeRO-2（约 8x）→ ZeRO-3（约 $d$ 倍，通信付到 1.5 倍上限）。大规模下 ZeRO-3 是默认形态—— $d$ 越大每卡越省；PyTorch 侧的同族实现是 FSDP（归 4.3）。
3. **卡少模型大** → Offload / Infinity：GPU 只留 fp16 参数与有界暂存，CPU/NVMe 扛状态与更新；代价是每 step $4N$ bytes 的 PCIe 搬运下界，只有大 batch（单 step 计算足够长）才藏得住（§5.1 的账，练习 4）。
4. **跨节点带宽先撞** → ZeRO++ 三件套：权重量化通信、节点内副本、量化梯度 all-to-all，跨节点 $3M \to 0.75M$ 。
5. **激活墙不在此表**：激活账与 checkpointing 归 1.4，与 ZeRO 各阶正交叠加；与模型并行（TP/PP，P5）的组合总账归 5.5。Megatron 栈内的对应物（distributed optimizer，ZeRO-1/2 同族）走读归 6.2。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 三态冗余 | DP 每 rank 复制 $16N$ bytes（ $K = 12$ ）；冗余在状态存放，不在 DP 概念 | models 1.2 式 (8)；arXiv:1910.02054 §3 |
| ZeRO-1 | 切优化器状态： $4N + 12N/d$ ；通信与 DP 严格相等（梯度 RS + 参数 AG = $2N$ 元素） | 式 (1)；论文 §5.1 / §7.2 |
| ZeRO-2 | 再切梯度： $2N + 14N/d$ ；通信不变；省在非己梯度块的生命周期 | 式 (2)；论文 §5.2 |
| ZeRO-3 机制 | 再切参数：常驻 $16N/d$ + 当前层全量 + 预取；前向 AG → 用后释放 → 反向再 AG → 梯度 RS | §4.1 时序图；论文 §5.3 |
| ZeRO-3 通信 | $3N$ 元素 = DP 的 1.5 倍上限（每 micro-batch 口径）；更新后无参数同步；预取摊进计算 | 式 (3)；论文 §7.2.2 |
| 累加交互 | 梯度归约 ÷ $m$ 需回 $2N$ 全量缓冲（FSDP §3.3.4 无通信变体）；参数取回 $2mN$ 恒不摊薄 | §4.4 |
| Offload | 数据流图三约束 → 唯一最优：fp16 参数 + 前向反向在 GPU，其余全 CPU；通信下界 $4N$ bytes/step；CPU Adam 6x；DPU 1.5x | arXiv:2101.06840 §3-5 |
| Infinity | NVMe 第三层；bandwidth-centric partitioning 以聚合带宽补单链路 PCIe（单卡 10-12 GB/s）；512×V100 训 32T 参数、超 25 PFLOPS | arXiv:2104.07857 |
| ZeRO++ | qwZ（INT8 权重）/ hpZ（节点内副本）/ qgZ（INT4 梯度 + 归约前恢复）：跨节点 $3M \to 0.75M$ | arXiv:2306.10209 |
| DeepSpeed | stage 枚举名 = 新切什么；stage3.py 分区与梯度 RS；parameter_offload.py 模块 hook；coordinator trace + 异步 AG + 预取 | §6，deepspeedai/DeepSpeed |

## 练习

1.（推导）ZeRO-1 通信不变性的形式化：设参数均分为 $d$ 个连续分区，按 4.1 式 (2) 的 ring 口径逐项展开，证明「梯度 reduce-scatter + 更新后参数 all-gather」与「梯度 all-reduce」在每 rank 通信字节数上严格相等；并说明第二段的内容从梯度换成参数为什么不改字节数（注意两者 dtype 的口径）。再答：ZeRO-2 相对 ZeRO-1 为什么不可能再降通信量？

2.（计算） $N = 70 \times 10^{9}$ 、bf16、Adam 混合精度（ $K = 12$ ）、 $d = 64$ ：(a) 算 DP / ZeRO-1 / ZeRO-2 / ZeRO-3 的每 rank 模型状态字节数（GB，SI 口径）；(b) 对照 80 GB 显存、激活与缓冲预留 20 GB，判定各阶可行性；(c) 同模型改 $d = 8$ 重算，从哪一阶开始撞墙？(d) 用论文 Table 1 的 7.5B @ $d = 64$ 三个数字（31.4 / 16.6 / 1.88 GB）反向验证式 (1)(2) 与 $16N/d$ 。

3.（推导）ZeRO-3 的峰值显存修正：设模型最大单层参数为 $N_{\max}$ 、预取深度 1 层，论证每 rank 峰值模型状态显存约为 $16N/d + 4N_{\max}$ bytes（当前层 + 预取层的全量 bf16 参数），代入 $N = 128 \times 10^{9}$ 、 $d = 64$ 、 $N_{\max} = 3 \times 10^{8}$ 给出数值；说明为什么论文 Table 1 的 $16N/d$ 口径在大 $d$ 下仍大致正确，以及什么样的模型结构会让修正项露头。

4.（计算）Offload 的 batch 下界：单 GPU、GPU↔CPU 双向带宽 32 GB/s，模型 $N = 10^{10}$ 。(a) 每 step $4N$ bytes 搬运的下限时间是多少？(b) 设该卡有效算力 $10^{14}$ FLOPS、训练每 token 约 $6N$ FLOPs，让搬运时间不超过 step 计算时间的最小每卡 batch tokens 数是多少？(c) 把这个数放进 1.5 式 (5) 的等效 batch 口径，说明 offload 为什么总与大 batch / gradient accumulation 绑定。

## 参考

- [Rajbhandari et al., 2020](../references.md)：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，arXiv:1910.02054，[A]（key: zero）——三阶段机制与显存公式（§5）、 $K$ / $\Psi$ 记号（§3）、通信量分析（§7： $P_{os}$ / $P_{os+g}$ 与 DP 同量、 $P_{os+g+p}$ 上限 1.5×）、Table 1 标尺数字
- [Ren et al., 2021](../references.md)：ZeRO-Offload: Democratizing Billion-Scale Model Training，arXiv:2101.06840，[A]（key: zero-offload）——数据流图与 $4N$ 通信下界（§3）、唯一最优策略（§3.5）、CPU Adam 与 DPU（§5）、13B 单卡 / 40 TFLOPS / 70B 单节点（摘要与 §6）
- [Rajbhandari et al., 2021](../references.md)：ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning，arXiv:2104.07857，[A]（key: zero-infinity）——五组件（§1）、单链路 10-12 GB/s 与 CPU/NVMe 带宽口径（§4）、32T 参数 @ 512×V100 与 25+ PFLOPS（§1 实测口径）
- [Wang et al., 2023](../references.md)：ZeRO++: Extremely Efficient Collective Communication for Giant Model Training，arXiv:2306.10209，[A]（key: zero-plusplus）——qwZ / hpZ / qgZ 三组件与跨节点 $3M \to 0.75M$
- [DeepSpeed, deepspeedai/DeepSpeed repo](../references.md)（key: deepspeed-repo，[A]）——`deepspeed/runtime/zero/stage_1_and_2.py`（stage 1/2 与 reduce-scatter 默认）、`stage3.py`（`DeepSpeedZeroOptimizer_Stage3`：分区、ipg bucket、`step()` 子组循环）、`parameter_offload.py`（`DeepSpeedZeRoOffload`：模块 hook）、`partitioned_param_coordinator.py`（`PartitionedParameterCoordinator`：trace、预取、`InflightParamRegistry`）、`config.py`（`ZeroStageEnum` 与 offload 配置）；路径与类名核实于 2026-11-17
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——MP 对照；distributed optimizer 走读归 6.2
- [PyTorch, pytorch/pytorch repo](../references.md)（key: pytorch-repo，教学参照）——DDP 基线对照（4.1）
- [Zhao et al., 2023](../references.md)：PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel，arXiv:2304.11277，[A]（key: fsdp）——§3.3.4 gradient accumulation 的 with/without-communication 两变体（§4.4 引其口径）
