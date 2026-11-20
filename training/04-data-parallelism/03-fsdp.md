# 4.3 FSDP / FSDP2：PyTorch 原生的全分片（FSDP and FSDP2: PyTorch-Native Full Sharding）

> 本章讲 PyTorch 原生的全分片方案 FSDP 与 FSDP2。先对齐语义：FSDP 就是 ZeRO-3 的 PyTorch 实现，差异全在工程。然后拆三个机制——FSDP1 把一组参数揉成一个 FlatParameter（padding 上界 $F-1$）的做法与代价；每个 step 里参数何时取回、何时释放（峰值显存公式由此推出）；FSDP2 改成逐参数 DTensor 分片的动机与收益。最后看混合精度、CPU offload、与 DDP 的兼容面，并走读 `torch/distributed/fsdp/_fully_shard/` 与 TorchTitan 的并行接线 | 前置：4.1（DDP 的 all-reduce 与 bucket overlap）、4.2（ZeRO 三阶切分与通信量推导）、1.3（混合精度三件套）、1.5（梯度累加）；models 1.2 式 (8)（ $16N$ 三态账） | 锚点：[Zhao et al., 2023]（FSDP 论文，[A]，覆盖 FSDP1 语义）、pytorch/pytorch repo 的 FSDP2 实现（`torch/distributed/fsdp/_fully_shard/`，教学参照 + 官方实现——FSDP2 无独立论文，口径以官方实现与设计文档为准）、pytorch/torchtitan repo（教学参照 + 官方实现）

## 1. 语义对齐：FSDP = ZeRO-3 的 PyTorch 实现

FSDP 与 ZeRO-3 是什么关系？FSDP 论文（[Zhao et al., 2023](../references.md)，arXiv:2304.11277，[A]）§3.2.1 自己给出了对照口径：full sharding 策略在 ring 算法下的通信量是 DDP 的 1.5 倍。4.2 从 ZeRO-3 的取回时序推出的是同一个 ×1.5：前向 all-gather 一次、反向再 all-gather 一次、反向结束前 reduce-scatter 一次，共三次 $M$ 级通信，对应 DDP 的两次。两套系统实现同一算法：参数、梯度、optimizer state 三态都分片；前向用完即释放；反向重新取回再归约。差异全部在工程侧——分片粒度、通信调度、内存管理与 API 形态。本章讲的就是这些工程差异。本章口径：数字实例统一取 7B 模型、bf16 参数与梯度；用到其他口径时随文声明。

语义对齐可以精确到配置项：FSDP1 用 `ShardingStrategy` 枚举表达切分深度；FSDP2 把它折叠成 `fully_shard` 的一个参数 `reshard_after_forward`；DeepSpeed 侧的对应物是 ZeRO 的阶。映射表如下（出自 TorchTitan 仓库的 FSDP 迁移文档 `torchtitan/distributed/FSDP.md`，[PyTorch, pytorch/torchtitan repo](../references.md)，教学参照 + 官方实现）：

| FSDP1（`sharding_strategy`） | FSDP2（`fully_shard` 参数） | DeepSpeed 对应 | 三态口径 |
| --- | --- | --- | --- |
| `NO_SHARD` | 不调用 `fully_shard` | ZeRO 关闭 | 全量复制，等价 DDP |
| `SHARD_GRAD_OP` | 1D mesh + `reshard_after_forward=False` | ZeRO-2 | 分片梯度与 optimizer 态；参数前向 all-gather 后**不释放** |
| `FULL_SHARD` | 1D mesh + `reshard_after_forward=True` | ZeRO-3 | 三态全分片；参数用后释放、反向重新取回 |
| `HYBRID_SHARD`（2D mesh） | 2D mesh + `reshard_after_forward=True` | MiCS（迁移文档的对应口径；本书 references 未收其条目） | 组内分片 + 组间复制 |
| —（FSDP1 无此档） | `reshard_after_forward=<int>`（如 8） | ZeRO++ 的 hpZ（[Wang et al., 2023](../references.md)，arXiv:2306.10209，[A]） | 前向后把参数重分片到较小的子组，反向 all-gather 只在该子组内进行 |

表中第二行有个要点。 `reshard_after_forward=False` 时，参数在前向结束后保持全量驻留，反向就不必再 all-gather 一次。通信退回 $2\Psi$，与 DDP 同量级；代价是参数显存一点不省。Llama-3 405B 的 FSDP 用的正是这一档：分片 optimizer 态与梯度，但参数不在前向后重分片——原文「model shards we do not reshard after forward computation to avoid an extra all-gather communication during backward passes」（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.3.2）。分片深度换通信量，这条权衡从论文配置表一路延伸到生产配置。

分片范围用一个变量统一表达：sharding factor $F$，即分片组的大小。 $F = 1$ 时退化为全复制——梯度只做 all-reduce，就是 DDP。 $F$ 等于全局度数时是全分片。介于两者之间叫 hybrid sharding：参数在大小为 $F$ 的组内分片，在 $W/F$ 个组之间复制。梯度归约也分两步走：先在分片组内 reduce-scatter，再在复制组间 all-reduce 分片后的梯度。

**命题**（两阶段归约等价）：先分片组内 reduce-scatter、再复制组间 all-reduce 分片梯度，与一次全局归约的结果相同。**证明**（要点）：全局求和按分片组做加法分解，结合律直接给出；通信路径的差别不改写数值。

把分片组收进节点内之后，跨机流量就有一笔明确的账。设模型大小 $M$（元素数口径）、全局度数 $W$、分片组大小 $G$，论文给出每 GPU 的跨机流量为

$$
2 M \thinspace \frac{W - 1}{G \thinspace W} \qquad (1)
$$

**符号**： $W$：全局并行度数（本书其余章记 $d$，此处保留论文记号以便对照）； $G$：分片组大小（同前述 $F$）； $M$：参数量。对照项：全复制的跨机流量是 $2M(W-1)/W$，全分片是 $3M(W-1)/W$。全分片的总流量更高，但 hybrid 形态的跨机流量反而更低，约为全复制的 $1/G$——因为 all-gather 与 reduce-scatter 都留在节点内走 NVLink，跨机只剩复制组的 all-reduce（量级 $2M/G$）。这就是「分片组贴节点边界」这条放置规则的算术依据（论文 §3.2.2）。代入一个数：分片组 8 卡（一个节点）、全局 512 卡时，跨机流量降到全复制的约 1/8；总流量不变，只是换了位置。本书全书约定 DP 度数为 $d$；式 (1) 在其余章节出现时，读作 $W \to d$、 $G \to d_s$。

## 2. FlatParameter：FSDP1 的分片粒度

FSDP1 的通信单元不是单个 `nn.Parameter`，而是一个 FSDP unit（一组子模块）的全部参数揉成的一个一维张量。揉法分三步：各参数先 flatten，再 concat，右端补零到能被分片度数整除；然后切成等长的 $F$ 块，每 rank 持有一块。这就是论文 §3.2 的 flatten-concat-chunk 算法；实现在 `torch/distributed/fsdp/_flat_param.py` 的 `FlatParameter` 与 `FlatParamHandle`（路径核实于 2026-11-17）。padding 的上界是 $F-1$ 个元素，可忽略。梯度继承同一套 unsharded/sharded 形状，原参数的存储由 FlatParameter 持有。

为什么不干脆让每个参数自己成为一个通信单元？论文 §3 的实测给了硬数字：固定总通信量约 $2^{30}$ 个 fp32 元素，改变单次 all-gather 的大小——单次量跌破 33M 元素后，总通信时间开始快速上升。原因在集合通信的启动与延迟项（口径：指每次集合通信的固定成本，与消息字节数无关的部分）：消息太小，固定开销摊不薄。把一层的十几个参数合成一个 FlatParameter，集合通信次数从 $O(\text{参数个数})$ 降到 $O(\text{unit 数})$，每次通信的消息就足够大。

> **提点**：「flat」的理由不在数学，在 NCCL 的粒度-效率曲线——消息越大，启动与延迟的固定成本摊得越薄。删掉这条，分片语义一个字不变，变的只有吞吐。

粒度还是显存旋钮，方向相反：unit 切得越细，unshard 暂态越小（§3），但集合通信次数越多、吞吐越差。论文称之为 FSDP 的 memory-throughput trade-off；控制手段就是用户怎么把子模块划成 unit。FSDP1 的旋钮是 `auto_wrap_policy`，例如按模块类型把每个 transformer block 包成一个 unit，实现见 `torch/distributed/fsdp/_wrap_utils.py` 与 `wrap.py`。FSDP2 删掉了这个包装层（它不是 wrapper，见 §5）：「逐 block 调一次 `fully_shard`」就是 auto_wrap 的全部语义，策略退回为用户代码里的一个 for 循环。

FlatParameter 的代价留到 §5 结算，这里先记两条。第一，单个参数失去独立身份：冻结、精度转换这类逐参数操作要穿过分片边界才能表达，组合性差。第二，state dict 逻辑随之膨胀——FSDP2 迁移文档对 FSDP1 的原话是，state dict 逻辑「thousands of lines」且需额外通信。

## 3. unshard 时序与峰值显存（关键推导）

单卡峰值显存出现在一个 step 的哪一刻？答案藏在每个 FSDP unit 的参数生命周期里。路线：先把一个 unit 的参数生命周期逐拍排开（时序图），再把峰值时刻显存里的全部占用按「常驻 + 暂态」两笔加总——关键一步是认出峰值出现在反向途中。所谓常驻，是指在整个 step 内不变的占用；所谓暂态，是指只在某个区间存在、用完即释放的占用。FULL_SHARD 下，每个 FSDP unit 的参数在一个 step 里经历同一条生命周期。把 4.2 的 ZeRO-3 取回时序搬到「逐 unit」粒度，unit $\ell$ 的状态机如下：

```
                 常驻（全 step）：分片参数 2N/d + 分片梯度 2N/d + optimizer 态 12N/d
                 ────────────────────────────────────────────────────────────
前向   unit ℓ−1 计算中 │ 预取线程：all-gather(ℓ) 发出
       unit ℓ    │ all-gather(ℓ) 完成 → 全量参数就绪 → 前向计算 → 释放全量参数（reshard）
反向   unit ℓ+1  │ …反向推进中…（ℓ 的梯度尚未算）
       unit ℓ    │ 再 all-gather(ℓ) → vjp 计算 → 全量梯度短暂存在 →
                 │ reduce-scatter(ℓ)：梯度归约并分片 → 释放全量参数与全量梯度
```

峰值显存由这条时序唯一确定。先定符号：参数量 $N$；DP 度数 $d$；每个 unit 的参数量 $P_u$（逐 block 包装时 $P_u \approx N/L$， $L$ 为层数）；前向允许在飞的预取 unit 数 $n_{\mathrm{pf}}$（prefetch 深度）。参数与梯度按 bf16（2 字节）计：

$$
M_{\mathrm{peak}} = \underbrace{\frac{2N + 2N + 12N}{d}}_{\text{三态分片常驻}} + \underbrace{2 P_u \thinspace (2 + n_{\mathrm{pf}})}_{\text{unshard 暂态}} + M_{\mathrm{act}} + M_{\mathrm{buf}} \qquad (2)
$$

**符号**：第一项是 models 1.2 式 (8) 的 $16N$ 三态账除以分片度数：fp32 master $4N$、fp32 矩估计 $8N$、bf16 参数 $2N$、bf16 梯度 $2N$。梯度分片只在反向存活，但峰值出现在反向途中，所以计入常驻项。暂态项含三个因子：当前 unit 的全量参数 $2P_u$；当前 unit 在 reduce-scatter 之前的全量梯度 $2P_u$；预取中的下一 unit 参数 $2 n_{\mathrm{pf}} P_u$。 $M_{\mathrm{act}}$ 是激活账，全部内容在 1.4：全存时每层 $sbh(34 + 5as/h)$，full checkpoint 压到每层 $2sbh$。 $M_{\mathrm{buf}}$ 是通信与框架 buffer（all-gather 输出缓冲、NCCL workspace、allocator 碎片），量级由实现决定，不进公式。

代入一组数字看各笔的比例：7B 模型， $d = 8$，逐 block 包装（ $L = 32$）， $n_{\mathrm{pf}} = 1$。常驻项 $16N/d = 14 \times 10^{9}$ bytes ≈ 13.0 GiB。暂态项： $P_u = N/L = 2.2 \times 10^{8}$，代入得 $2 P_u (2 + 1) = 1.3 \times 10^{9}$ bytes ≈ 1.2 GiB，只有常驻的 9%——「全分片 ≈ 显存除以 $d$」的近似在逐 block 粒度下成立。另一个极端是把整个模型包成一个 unit（ $P_u = N$ ）：暂态变成 $2N \times 3 = 4.2 \times 10^{10}$ bytes ≈ 39.1 GiB，比常驻还大。粒度旋钮的两端就是这样，auto_wrap 必须存在的原因也在这里。

公式 (2) 还直接解释两条生产口径。其一，`reshard_after_forward=False`（§1 的 ZeRO-2 档）：常驻项里的参数从 $2N/d$ 换回 $2N$，全量参数全程驻留，同时删去反向那次 all-gather。Llama-3 的选择就是这一档。其二，FSDP1 的内存管理围绕暂态项设计：限制在飞 unsharded 块的数量，必要时挂起 CPU 线程、等 GPU 追平（论文 §3.4；API 上的 `limit_all_gathers`）。本质是把式 (2) 的 $n_{\mathrm{pf}}$ 限制在显存预算内。FSDP2 换了一套内存管理（§5），这个旋钮随之取消。

CPU offload 是式 (2) 的另一个变体：分片参数常驻 CPU 内存，unshard 时经 PCIe 搬上 GPU。FSDP1 的开关是 `CPUOffload(offload_params=True)`，FSDP2 是 `offload_policy=CPUOffloadPolicy(...)`。代价用带宽就能估：每 step 要把参数全量搬上 GPU 两次（前向一次、反向一次），共约 $4N$ bytes。按 PCIe 5.0 x16 单向约 63 GB/s 的公开规格，7B 模型约 $28 \times 10^{9}$ bytes，下界约 0.45 s/step。与计算流水重叠之后的净开销取决于实现——这里给的是本书的带宽下界账。offload 族的完整账（含 ZeRO-Offload / Infinity 的梯度与 optimizer 态卸载）归 4.2，本节只立 FSDP 侧的接口位置。

## 4. prefetch 与 overlap：单 NCCL 流上的排序问题

分片把通信插进了计算的关键路径。overlap 要做的只有一件事：让通信在计算仍在跑的时候发出去。FSDP 论文（§3.3）记录了 eager 模式下的两个障碍和解法。

**反向的排序障碍**。默认实现里，反向走到 unit $\ell$ 时先发 reduce-scatter( $\ell$ )，再发 all-gather( $\ell-1$ )。两个集合通信排在同一条 NCCL stream 上：reduce-scatter 不完成，all-gather 就不能开始；而后续梯度计算又在等这个 all-gather。两次通信就这样串行暴露在关键路径上。FSDP 的 backward prefetching 把顺序倒过来：先发 all-gather( $\ell-1$ )，再发 reduce-scatter( $\ell$ )——取下一单元的参数与归约当前单元的梯度在流上重叠。剩下的问题只有一个：eager 模式下，系统事先不知道「下一个」是哪个 unit。FSDP 的答案很直接：每个 iteration 现记。前向执行顺序的逆序就是反向顺序；当 iter 记录、当 iter 使用。记录逐 iteration 刷新，所以控制流跨 iteration 变化的模型也兼容。

**前向的发行速度障碍**。eager 模式的 host 开销不小，CPU 逐个 unit 发 all-gather 的速度可能赶不上 GPU 消费的速度。对静态图，FSDP 允许沿用上一 iteration 的前向顺序，提前显式预取下一个 all-gather（forward prefetching）。

FSDP2 在这两处都收紧了口径（迁移文档）。反向 prefetch 不再是选项：`BACKWARD_PRE` 被固定为唯一行为，因为 `BACKWARD_POST` 在嵌套模块情形会错误预取（文档注记）。前向 prefetch 在迁移文档成文时标注为「under discussion」，以当前实现为准。

> **提点**：把「正确但只有专家会开的开关」变成「唯一路径」，是 FSDP2 整体设计的缩影——API 表面积缩小，正确性不再依赖用户记得开哪个开关。

## 5. FSDP2：per-param DTensor 的重写

FSDP2 为什么要把 FlatParameter 拆掉重写？官方说明在 pytorch/torchtitan repo 的 `torchtitan/distributed/FSDP.md`（教学参照 + 官方实现；FSDP2 无独立论文），动机落在 FlatParameter 上：参数揉成一团之后，逐参数行为（冻结、精度转换）难以表达，组合性差；state dict 逻辑膨胀且需要额外通信。重写后的语义变化逐条如下：

- **分片表示**：每个 `nn.Parameter` 独立成为一个 DTensor（distributed tensor），沿 dim-0 分片，不再有 FlatParameter。DTensor 的意思是：张量逻辑上完整，物理上按维度切到多卡。`model.named_parameters()` 返回的名字不变，optimizer 与梯度裁剪看到的都是 DTensor——`nn.utils.clip_grad_norm_` 直接可用；FSDP1 则必须用专门的 `FSDP.clip_grad_norm_()` 才能算分片范数。
- **state dict 免通信**：`model.state_dict()` 直接返回分片的 DTensor 字典，不触发集合通信（FSDP1 默认 full state dict、需 all-gather）；checkpoint 的存取与并行度解耦因此落在 DTensor 层（分布式 checkpoint 归 7.4）。
- **内存管理**：新系统不依赖 `recordStream`，也不需要 CPU 同步；峰值显存更低，而且是确定的。`limit_all_gathers` 与 CPU 挂起机制整体取消。验证口径：迁移文档记录了 TorchTitan 上 Llama-7B / 8×H100 的对照——FSDP2 峰值显存低 7%，MFU 更高，loss 曲线重合（torchtitan PR #165）。
- **API 形态**：`fully_shard(module)` 不是 wrapper。它通过 `@contract` 装饰器给 `module` 动态换类：生成一个同时继承 `FSDPModule` 与原类的新类，改写模块的 `__class__`。模块层级与全限定名不变；状态对象经 `fully_shard.state(module)` 取回。
- **初始化流程**：先在 meta device 上建模型（不占显存），再逐 block 调 `fully_shard`，然后 `model.to_empty(device="cuda")` 分配真实存储，最后跑用户初始化函数。分片先于物化，FSDP1 的 `param_init_fn` 配套机制随之取消。
- **混合精度**：`mp_policy=MixedPrecisionPolicy(param_dtype=..., reduce_dtype=..., output_dtype=..., cast_forward_inputs=...)`。dtype 的流向是：分片参数以原 dtype 存储；all-gather 与计算用 `param_dtype`；梯度 reduce-scatter 用 `reduce_dtype`——bf16 省通信，fp32 保归约精度，这个选择直接改写 4.4 通信账里梯度项的字节系数。相对 FSDP1 的 `mixed_precision`，删去了 `buffer_dtype`，新增 `output_dtype`。
- **梯度累加**：`no_sync` 语义改名为 `set_requires_gradient_sync`。FSDP 论文 §3.3.4 把累积分成两个变体。with communication：每个 micro-batch 照常归约，各 rank 保存分片梯度。without communication：通信推迟到累加边界，各 rank 在循环内累加的是**未分片**梯度，显存为此多驻留一份全量梯度缓冲（bf16 即 $2N$ bytes）。通信账因此是两句话。第一句：梯度归约项从每 micro-batch 一次摊到每个 optimizer step 一次，即 ÷ $m$，代价就是这份缓冲。第二句：参数取回（前向、反向各一次 all-gather）逐 micro-batch 发生，任何配置下都不摊薄。hybrid（HSDP）形态下，复制组的 all-reduce 用 `set_requires_all_reduce` 单独控制。

「更细粒度」的收益具体是两条。第一，per-param 分片让取回、释放、dtype 转换、冻结都以单参数为粒度表达；unshard 的调度与显存管理不再需要照顾一个揉合了多个参数的大块。第二，通信侧按通信组把同组参数复制进连续缓冲，合并成一次大集合通信再发行——FlatParameter 当年靠「揉大」拿到的通信粒度，用这种方式补了回来（实现见 §7 的 `_fsdp_collectives.py`）。

## 6. 与 DDP 的 API 兼容面

换成 FSDP 之后，原来 DDP 的用法还成立吗？FSDP1 的 `FullyShardedDataParallel` 与 DDP（4.1）同为 nn.Module wrapper：包一层就完成并行化，构造参数换成 sharding 策略、auto wrap、mixed precision 等。FSDP2 连 wrapper 都不是，`fully_shard` 只是模块级标注。但「模型代码零改动、外层加并行」的 DDP 体验仍然保留，靠两点：逐 block 调用不改动模块源码；根模块再调一次 `fully_shard`，兜住 block 之外的参数（embedding、输出头、norm——凡是没有被内层 `fully_shard` 认领的参数，都归根模块的通信组）。

两处不兼容必须知道。第一，FSDP1 的 FlatParameter 会把参数改名（`flat_param`）：不经过 `state_dict_type` 转换，checkpoint 与非 FSDP 代码互不相认。FSDP2 不改名，state dict 直接是分片形态，这一类问题随之消失；代价是下游（推理侧加载、部分微调工具）必须认识 DTensor，或者用 `full_tensor()` 收拢。第二，分片语义对 optimizer 有硬约束：optimizer 必须在 `fully_shard` 之后构造，因为它拿到的是分片后的参数/DTensor。先建 optimizer 再分片，optimizer state 会按全量参数分配，分片失效。这条顺序约束在两个版本里都存在——它是「分片改写了参数身份」的直接后果。

> **坑**：先构造 optimizer、后调 `fully_shard`——optimizer state 按全量参数分配，分片静默失效，不显式报错。规避：全部 `fully_shard` 调用完成之后再建 optimizer。

## 7. 代码走读：`_fully_shard/` 与 TorchTitan 接线

官方实现的代码长什么样？FSDP2 的实现集中在 pytorch/pytorch repo 的 `torch/distributed/fsdp/_fully_shard/`（[PyTorch, pytorch/pytorch repo](../references.md)，教学参照 + 官方实现；以下路径与符号名核实于 2026-11-17，经 GitHub contents API）：

- `_fully_shard.py`：`fully_shard` 入口，签名含 `mesh`、`reshard_after_forward: bool | int | None`、`shard_placement_fn`、`mp_policy`、`offload_policy`、`ignored_params`；另有 `FSDPModule`（动态换类的注入侧）与 `UnshardHandle`（手动 unshard 的句柄）。docstring 原文即本章 §3 的时序：初始化分片 → forward pre-hook all-gather → forward hook 释放（若 reshard）→ backward hook 再 all-gather、用后释放参数并 reduce-scatter 梯度。
- `_fsdp_param.py`：`FSDPParam`，单个参数的分片视图，持有分片 DTensor 与 unsharded 形状的元数据。`alloc_storage` / `free_storage` 是 §5 说的「确定性显存管理」的落点：unsharded 存储显式分配、显式释放，不依赖 `recordStream` 的隐式生命周期。
- `_fsdp_param_group.py`：`FSDPParamGroup`，一次 `fully_shard` 调用认领的参数集合及其通信上下文（`FSDPCommContext`）。all-gather 与 reduce-scatter 的调度以 group 为单位——就是 §3 时序里的「unit」。
- `_fsdp_collectives.py`：all-gather / reduce-scatter 的实现族，默认路径与 symmetric-memory 路径并列。`all_gather_copy_in_*` 负责把组内参数复制进连续缓冲、合成一次大消息再发行。

TorchTitan（[PyTorch, pytorch/torchtitan repo](../references.md)，教学参照 + 官方实现）给出生产配置的最小样本。入口是 `torchtitan/distributed/fsdp.py` 的 `apply_fsdp_to_decoder(model, dp_mesh, param_dtype, reduce_dtype, ...)`：对全部 decoder 模型（llama3 / qwen3 / deepseek_v3 / gpt_oss 共用）逐 transformer block 调 `fully_shard`，最后再调一次兜住根模块。`reshard_after_forward_policy` 取 `default` / `always` / `never` 三档，对应 §1 的策略表；`cpu_offload` 开关接 `CPUOffloadPolicy`。并行度数与 mesh 的声明在 `torchtitan/config/parallelism.py`。多维 mesh（DP×TP×CP×PP）的组装归 5.5 与 6.3，本章只看 DP 维。

教学骨架（教学用，非优化实现），对应 §3 时序与 §5 的 API：

```python
# 教学用，非优化实现：FSDP2 最小接线（2 卡起步；torchrun 启动）
import torch, torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

torch.distributed.init_process_group("nccl")
mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (torch.distributed.get_world_size(),))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

with torch.device("meta"):
    model = MyTransformer()                       # meta 上构图，不占显存
for block in model.blocks:                        # 逐 block 一个通信组（auto_wrap 的 FSDP2 形态）
    fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mesh=mesh, mp_policy=mp)       # 根模块兜住 embedding / 输出头
model.to_empty(device="cuda")                     # 分片之后才有真实存储
model.init_weights()                              # 每个 rank 只见自己的 dim-0 分片

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)   # 必须在 fully_shard 之后构造
for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()                               # 时序：AG → fwd → free → AG → bwd → RS
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # 直接作用于 DTensor
    opt.step(); opt.zero_grad(set_to_none=True)
```

## 总结

| 主题 | 公式 / 口径 | 要点 |
| --- | --- | --- |
| 语义对齐 | FSDP full shard = ZeRO-3，通信 ×1.5（两实现同源） | 策略映射表：`reshard_after_forward` 三档 ↔ FSDP1 枚举 ↔ ZeRO 阶（第 1 节） |
| FlatParameter | flatten-concat-chunk，padding $\le F-1$ | 合并通信应对 33M 元素效率悬崖；粒度 = 显存-吞吐旋钮（第 2 节） |
| 峰值显存 | 式 (2)： $16N/d + 2P_u(2+n_{\mathrm{pf}}) + M_{\mathrm{act}} + M_{\mathrm{buf}}$ | 常驻 ÷ $d$，暂态由 unit 粒度与 prefetch 深度决定 |
| prefetch | 反向：先 AG( $\ell-1$ ) 后 RS( $\ell$ )（第 4 节） | 单 NCCL 流的排序障碍；顺序 = 前向序的逆，逐 iteration 现记；FSDP2 固定 BACKWARD_PRE |
| FSDP2 | per-param DTensor（dim-0）（第 5 节） | 无 FlatParameter；state dict 免通信；无 `recordStream`、无 CPU 同步；7B/8×H100 峰值 −7%（迁移文档口径） |
| 混合精度 | `MixedPrecisionPolicy` 四字段（第 5 节） | 存原 dtype、算 `param_dtype`、归约 `reduce_dtype`；后者改梯度通信字节系数 |
| 兼容面 | optimizer 须在分片后构造（第 6 节） | FSDP2 非 wrapper（`@contract` 换类）；`clip_grad_norm_` 直接吃 DTensor |

## 练习

1.（计算 · [10]）70B 模型、 $d = 64$、逐 block 包装（ $L = 80$）、 $n_{\mathrm{pf}} = 1$：按式 (2) 计算常驻与暂态两项（GiB），并给出 H100 80 GiB 上 $M_{\mathrm{act}} + M_{\mathrm{buf}}$ 的余量；若把包装粒度改为每 4 层一个 unit，暂态项变为多少。校验：常驻项与下一章三态静态表的 70B 行、d=64 格一致。
2.（推导 · [20]）补全式 (2) 的暂态项：按 §3 时序逐拍列出反向经过 unit $\ell$ 时显存里的全部参数/梯度张量（含相邻 unit 的预取），验证系数 $(2 + n_{\mathrm{pf}})$；再说明若 `reshard_after_forward=False`，式 (2) 的哪几项如何变化，由此解释 Llama-3 配置的显存-通信取舍（其 DP 组内参数全量驻留）。
3.（推导 · [30]）hybrid sharding 的跨机流量：从「分片组内完成 all-gather 与 reduce-scatter、复制组间只对分片梯度做 all-reduce」出发，推式 (1)（论文口径 $2M(W-1)/(GW)$ ），并验证 $G = 1$ 与 $G = W$ 两个端点分别回到全复制与「跨机流量趋零」的极限；说明为什么端点 $G = W$ 与全分片的总流量 $3M(W-1)/W$ 不矛盾（提示：式 (1) 计量的是跨机部分）。
4.（代码 · [30]）教学用：先按正文峰值显存公式写下两档峰值差的预测值；再把 §7 的骨架改成 `reshard_after_forward=False`，在 2 卡上跑同一个 step，用 `torch.cuda.max_memory_allocated` 对照两档的峰值差，验证式 (2) 中常驻项参数从 $2N/d$ 变 $2N$ 的预测；再把 `reduce_dtype` 改为 `torch.float32`，观察梯度通信字节数的翻倍（用 `torch.profiler` 的通信算子记录）。校验：fp32 归约档的梯度通信字节数恰为 bf16 档的 2 倍；两档峰值差约等于参数全量与分片之差。

## 参考

- [Zhao et al., 2023](../references.md)：PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel，arXiv:2304.11277，[A]（key: fsdp）——FSDP1 语义：sharding factor 与三档策略、×1.5 通信口径、式 (1) 的 hybrid 跨机流量、FlatParameter 与 33M 元素悬崖、backward/forward prefetch 机制、内存管理（§3.2-3.4）
- [Rajbhandari et al., 2020](../references.md)：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，arXiv:1910.02054，[A]（key: zero）——ZeRO-3 取回时序与通信量推导（归 4.2 展开，本章 §1 对齐用）
- [Wang et al., 2023](../references.md)：ZeRO++，arXiv:2306.10209，[A]（key: zero-plusplus）——hpZ 与 `reshard_after_forward=<int>` 的对应
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——生产口径：分片 optimizer 态与梯度、不前向后重分片参数（§3.3.2）
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照 + 官方实现（key: pytorch-repo）——`torch/distributed/fsdp/_fully_shard/`（`fully_shard` / `FSDPModule` / `FSDPParam` / `FSDPParamGroup` / collectives 族）、`torch/distributed/fsdp/fully_sharded_data_parallel.py` 与 `_flat_param.py`（FSDP1）；路径核实于 2026-11-17
- [PyTorch, pytorch/torchtitan repo](../references.md)：教学参照 + 官方实现（key: torchtitan-repo）——`torchtitan/distributed/fsdp.py`（`apply_fsdp_to_decoder`）、`torchtitan/config/parallelism.py`、`torchtitan/distributed/FSDP.md`（FSDP1→FSDP2 迁移文档：重写动机、API 映射表、7B 验证口径）
