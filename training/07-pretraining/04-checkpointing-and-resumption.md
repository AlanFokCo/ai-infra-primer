# 7.4 Checkpoint 与续训（Checkpointing and Resumption）

> 大规模训练怎么存档、怎么读档？本章依次回答：一份能精确续训的 checkpoint 必须装下哪五个分量（参数 / 优化器态 / 三路 RNG / 数据游标 / schedule 相位）；它的字节账怎么算（ $14N$ 口径，与 herd 论文「每 GPU 1MB-4GB」对账）；保存与加载的并行度如何解耦（resharding：torch.distributed.checkpoint 与 Megatron dist-ckpt）；异步写盘如何把 GPU 停顿压到 staging 的一次拷贝；干净恢复为什么不需要重新 warmup；最后用 Llama-3 的 466 次中断统计推出 checkpoint 最优间隔 | 前置：2.2（AdamW 状态与 bias correction）、2.3（warmup 的双重论证）、4.2（ $16N$ 三态账）、4.3（FSDP 分片语义）、6.2 §5（Megatron dist-ckpt 机制）、7.3 §2.4（中断分类统计）| 锚点：`torch.distributed.checkpoint`（pytorch/pytorch repo `torch/distributed/checkpoint/`，教学参照+官方实现，无独立论文）、Megatron-LM dist-ckpt [A]（arXiv:1909.08053 repo）、Llama-3 herd [A]（arXiv:2407.21783 §3.3.1 / §3.3.4）、FSDP [A]（arXiv:2304.11277）

## 1. checkpoint 里有什么：精确可恢复性的五个分量

本章口径：字节账与停顿账统一取 Llama-3 405B、16384 卡一档（bf16 混合精度 + AdamW）；中断统计取自同一窗口的公开记录。

从 checkpoint 续训，怎样才算「接上了」？精确可恢复（exact resumption）的语义是：续训后的轨迹与「从未中断」的反事实轨迹逐步一致。注意这个一致是**状态级**的——CUDA kernel 的归约顺序等因素使逐位（bitwise）一致在非确定实现下本就不保证。工程上追求的是状态精确、数值近似一致（本书口径注记）。

要让状态精确，checkpoint 必须装下五个分量，缺一不可：

| 分量 | 内容 | 缺失的后果 |
| --- | --- | --- |
| 模型参数 | bf16 工作参数（+ fp32 master，若与工作参数分开存） | 不可恢复 |
| 优化器状态 | fp32 master、一阶矩 $m$、二阶矩 $v$、**step 计数器 $t$** | $m$ / $v$ 丢失 ⇒ 回冷启动，7.3 §5.3 的重新 warmup 不可避免； $t$ 丢失 ⇒ bias correction 因子 $1 - \beta^{t}$ 算错（2.2 式 (4)），前若干步更新系统性偏差 |
| RNG 状态 | 三处：python `random`、torch CPU generator、**每卡各一个** CUDA generator | dropout mask 与数据 shuffle 序列改变，精确复现断裂（1.4 的 checkpoint RNG 语义同源） |
| dataloader 游标 | 全局已消费样本数（consumed samples 口径） | 数据被重复消费或跳过，exactly-once 断裂 |
| schedule 相位 | 全局 step、LR schedule 相位、batch ramp 进度 | LR 从错的相位继续（等价于一次无意的 LR 跳变，7.3 §3 的 LR 监控行会立刻报警） |

RNG 为什么要单列三处？原因在实现层。`torch.get_rng_state()` 只拿 CPU generator；CUDA 侧每张卡有独立 generator（`torch.cuda.get_rng_state_all()` 返回逐卡列表）；python `random`（以及用了它的话还有 numpy）是第三条独立序列。哪路随机性挂在哪条序列上随实现而变——CUDA 上的 dropout 用 CUDA generator，数据 shuffle 常用 CPU / python 侧序列——审计自己的栈是本书口径的纪律。共同点是：少存任何一路，续训后的随机性就从「同一序列的继续」变成「另一条序列」。

> **坑**：只存 CPU 的 generator、漏掉逐卡 CUDA 状态——续训后 dropout mask 换成另一条序列，轨迹从恢复点悄悄分叉且 loss 照常下降。规避：三路 RNG 全存（上表第 3 行）。

教学形态的最小集合（教学用，非优化实现）：

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),          # 含 m / v 与 step 计数器
    "rng_python": random.getstate(),
    "rng_torch_cpu": torch.get_rng_state(),
    "rng_torch_cuda": torch.cuda.get_rng_state_all(),   # 逐卡列表
    "consumed_samples": sampler.consumed,                # 全局样本数
    "step": global_step,                                 # schedule 相位
}
```

dataloader 游标的正确粒度是**全局样本序号**，不是 per-rank 偏移。原因：数据顺序在 Megatron 式栈里是（seed, consumed_samples）的确定函数，续训时按同一 seed 重建 sampler、按游标跳过即可。这样游标在并行度变化（ §3 的 resharding）后仍然有效；存 per-rank 偏移则在换 DP 度数后无法拼回全局顺序（Megatron 的 `consumed_samples` 机制，回链 6.2）。

> **坑**：只记各 rank 的消费偏移——换并行度后续训，偏移拼不回全局顺序，数据被重复消费或静默跳过。规避：游标一律存全局已消费样本数。

## 2. 字节账： $14N$ 、分片与写盘时间

一份续训级 checkpoint 有多大？内容按 4.2 的账走（bf16 混合精度 + AdamW，models 1.2 式 (8) 的分解）：

$$
B_{\mathrm{ckpt}} = \underbrace{2N}_{\text{bf16 参数}} + \underbrace{4N}_{\text{fp32 master}} + \underbrace{4N + 4N}_{m, \hspace{5mu} v} = 14N \thinspace \text{bytes} \qquad (1)
$$

**符号**： $N$：参数量。

代入 405B 算一笔： $14 \times 405 \times 10^{9} \approx 5.7 \times 10^{12}$ bytes（SI 口径约 5.7 TB）。分片保存时（FSDP / ZeRO 家族每 rank 只持自己那片，4.3 / 4.2），16384 卡下每卡写 $5.7 \times 10^{12} / 16384 \approx 3.5 \times 10^{8}$ bytes ≈ 0.35 GB；8192 卡下约 0.69 GB。这组数落在 herd 论文「每 GPU 的模型状态 1MB 到 4GB」的口径带内（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.3.1）；该带同时覆盖 8B 到 405B 的模型族与不同分片度，本书的对账数值与其相容。

三个变体注记。第一，只服务推理 / 评测的发布 checkpoint 不含优化器态， $2N$（bf16）或 $4N$（fp32）即可。第二，fp32 master 与工作参数同时在场时，存 master 即精确推出工作参数（1.3 式 (2) 的 $`w_t = \mathrm{round}_{16}(\theta_t)`$：对同一 fp32 值重舍入是确定的幂等映射）；反向不可精确恢复——bf16 工作参数的低位信息在 master 里才有。生产栈通常都存。第三，DeepSeek-V3 把两阶矩以 BF16 存储（2.2 §7 已核，其 §3.3.3 口径），该变体下 $B_{\mathrm{ckpt}} = 2N + 4N + 2N + 2N = 10N$ 。

写盘要花多少时间？先看存储侧的口径：herd 的训练存储是 Tectonic 文件系统上的专用 fabric——240 PB 容量、7500 台 SSD 服务器、可持续吞吐 2 TB/s、峰值 7 TB/s（§3.3.1，[A]）。5.7 TB 全量按 2 TB/s 可持续吞吐写，下界约 2.8 s。

再对照 step 时间。Table 4 的 16384 卡配置（TP8×CP1×PP16×DP128）下，每 batch 16M token、每卡 400 TFLOPS（MFU $41\text{％}$ ）。每 step 计算量 $6 \times 405 \times 10^{9} \times 16 \times 10^{6} \approx 3.9 \times 10^{19}$ FLOPs；集群有效算力 $16384 \times 400 \times 10^{12} \approx 6.6 \times 10^{18}$ FLOPS/s；一个 step 约 5.9 s。

两笔账合在一起：**一次阻塞式全量写盘 ≈ 半个 step**。这就是「写盘 stall」的账面量级，也是 herd 把 checkpoint 写入描述为「highly bursty」、把目标写成「minimize GPU pause time」的原因（§3.3.1 原文口径）。1.6 万卡在同一时刻一起写，还伴随数据中心级功率波动（万卡同时空转等 checkpoint 落盘，瞬时功率摆动以十 MW 计，§3.3.4 记录）。

## 3. 分布式 checkpoint：保存与加载的并行度解耦

换一批卡续训，旧存档还能读吗？分布式训练给 checkpoint 加了一个单机不存在的问题：**保存时的并行网格与加载时的网格未必相同**。8B 模型在 64 卡上按 ZeRO-3 分片存下的参数分片，换 32 卡续训时分片边界整体移动；TP/PP 度数变化时分片维度本身改变。逐 rank「各存各的、各读各的」的朴素格式在这个场景下直接失效——这就是 resharding 问题。

工程答案是把「逻辑张量」与「物理分片」用元数据分开：每个分片以（key + 全局 shape + 全局 offset + 本地块）登记；保存时各 rank 只写自己那片；加载时按**目标**网格重新规划每个 rank 读哪些分片。两个一手实现：

- **`torch.distributed.checkpoint`（DCP）**（pytorch repo `torch/distributed/checkpoint/`，教学参照+官方实现）：`save` 以 SPMD 方式工作——正确处理 `ShardedTensor` 与 DTensor，每个 rank 只保存自己的本地分片（官方文档口径）。planner 层把各 rank 的局部写计划（SavePlan / WriteItem）合成全局 metadata；加载侧按新网格生成读计划（LoadPlan / ReadItem）。保存 / 加载的并行度由此解耦：加载时的 world size 可以与保存时不同，读计划按目标网格重新声明每个 rank 需要哪些全局区间，存储层按 offset 切片读出。`state_dict` 里嵌了 `Stateful` 对象（模型、优化器）时，保存走各自的 `state_dict()` 钩子。
- **Megatron dist-ckpt**（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]，repo 的 `megatron/core/dist_checkpointing/`）：同一思想在 Megatron 栈内的形态——`ShardedTensor` 元数据登记、保存 / 加载网格解耦、distributed optimizer 的分片区间映射随 checkpoint 携带。机制级走读已在 6.2 §5 交付，本章只登记它作为 resharding 参照的位置。

FSDP 家族（[Zhao et al., 2023](../references.md)，arXiv:2304.11277，[A]；语义走读归 4.3）的全分片模型状态使「每 rank 存本地分片」成为唯一自然的保存形态——分片 checkpoint 不是可选优化，而是分片训练的直接后果；DCP 对 `ShardedTensor` 的一等支持正是顺着这个语义长出来的。元数据与数据分离还附带两个工程能力。第一，只读元数据即可检视 checkpoint 内容清单，不搬任何张量字节（Megatron dist-ckpt 有对应的只读元数据接口，6.2 §5）。第二，按 key 子集加载即得部分恢复——只取模型参数、跳过优化器态，就是「评测加载」与「换优化器续训」两个场景的同一机制（后者接 §5 的重 warmup 判别）。

```
保存网格（DP=4，每 rank 1/4）          逻辑全量张量（元数据：全局 shape + offset）
rank0 [t0] ┐                       ┌──────────────────────────────┐
rank1 [t1] ├─► 各 rank 写本地片 ──►│  key: "layer3.w"             │
rank2 [t2] │                       │  global_shape = [4096, 4096] │
rank3 [t3] ┘                       │  shard0: offset [0:1024, :]  │──┐
                                   │  shard1: offset [1024:2048,:]│  │
                                   └──────────────────────────────┘  │
加载网格（DP=2，每 rank 读全局的一半，跨旧分片边界拼装） ◄─────────────┘
rank0' ← shard0 + shard1        rank1' ← shard2 + shard3
```

## 4. 写盘 stall 与异步 checkpoint

回到 §2 的账：阻塞写一次全量 ≈ 半个 step（405B / 16384 卡 / 2 TB/s 口径）。异步 checkpoint 的目标是把 GPU 可见的停顿从「整次写盘」压到一次 staging（暂存：state_dict 先拷进主机内存缓冲，再由后台落盘）。DCP 的官方机制（`torch/distributed/checkpoint/state_dict_saver.py` 的 `async_save`，文档口径，教学参照+官方实现）分两段：

```
GPU 显存                    主机内存（staging）           持久存储（fabric）
─────────────          ─────────────────────      ─────────────────────
训练 step 继续          staging：state_dict          upload：后台线程/进程
     │                  逐张量 D2H 拷贝                写文件 / 对象存储
     ▼                        │                            │
 async_save() 调用 ──►  D2H（阻塞段，~秒级以下） ──►  staging_completion
     │                        │                       （future：此后 GPU
     ▼                        ▼                        侧张量可安全复写）
 训练不等写盘 ────────────────────────────────►  upload_completion
                                                    （future：落盘完成）
```

文档级事实有三条。第一，`async_save` 先把 state_dict de-stage 到 staging 存储（默认 CPU 内存），再在独立线程里执行保存（`async_checkpointer_type` 可选 THREAD / PROCESS；staging 实现可经 `AsyncStager` 接口替换，见 `torch/distributed/checkpoint/staging.py`）。第二，返回值 `AsyncSaveResponse` 带两个 future：`staging_completion`（本地拷贝完成）与 `upload_completion`（checkpoint 落盘完成）。第三，该特性文档标注为 experimental。

正确性约束跟着两阶段结构走：**staging 完成前，训练侧不能原地改写被保存的张量**。生产栈把它落在两种模式之一：「optimizer step 完成后发起 async_save、下一步前向开始前等 `staging_completion`」，或「保存快照对象本身」。GPU 暴露的停顿只剩 D2H 拷贝一段：每卡约 0.35 GB（ §2）的 staging 拷贝，相对约 6 s 的 step 是小数点后两个量级的事；写盘本身（2.8 s 下界）全部藏进后台。

herd 论文的口径止于需求侧：bursty writes 的账（每 GPU 1MB-4GB、存储可持续 2 TB/s）与「minimize GPU pause time」的目标，未披露其异步机制细节（本书注记）。机制侧的公开参照即上文 DCP 文档。两者拼起来是完整的设计空间：需求是「GPU 停顿最小化 + 频率提高」，机制是「staging 解耦 + 后台写盘」。

## 5. 续训语义：干净恢复为什么不需要重新 warmup

故障恢复的标准动作是「从最近 checkpoint 原样恢复，schedule 从断点相位继续」——而不是重跑 warmup。为什么？论证拆开是两半。

**优化器侧**：warmup 的存在理由（2.3 的双重论证）是冷启动时 $\hat{v}$ 的估计方差大、bias correction 未消化——两者都是**早期步数**的现象。bias correction 因子 $1 - \beta_2^{t}$ 在 $\beta_2 = 0.95$、 $t = 1000$ 处已与 1 相差 $e^{-51}$ 量级；RAdam 口径的方差自由度 $\rho_t$ 在大 $t$ 处趋于 $2/(1-\beta_2) - 1 = 39$，远离发散区。续训时 $m$、 $v$、 $t$ 全部从 checkpoint 恢复（ §1），优化器状态是「热」的。warmup 针对的病在恢复点根本不存在，重跑 warmup 反而人为引入一段 LR 凹陷。

**对照情形**：若优化器状态被丢弃（换优化器、换精度格式、checkpoint 只存了参数）， $m = v = 0$ 的冷启动条件重新成立，此时 7.3 §5.3 的重新 warmup 才登场。一句话判别：**warmup 冷却的是优化器状态，不是参数**；状态热则免 warmup，状态丢则重 warmup（本书论证，基于 2.2 / 2.3 的已交付结论）。

**数据侧**：exactly-once 的恢复靠 §1 的全局游标——按（seed, consumed_samples）重建 sampler 并跳过已消费段，任何样本不被重复也不被跳过。DP 度数变化时全局游标语义不变（ §3 的解耦），per-rank 偏移则失效。这条与 resharding 的组合，是「换卡数续训」能成立的关键一环。

**schedule 侧**：LR schedule 是纯函数 $\eta = f(t)$（2.3 的各族形式），batch ramp 同样是步数的分段函数（7.2 的 ramp 配方）。所以 schedule 分量的全部内容就是一个整数 $t$ 外加 ramp 相位。存对 $t$，LR 相位自动正确；存错 $t$，故障形态是 LR 在恢复点跳变——7.3 §3 的 LR 监控行是这类 bug 的检出面。

## 6. 容错：MTBF 账与最优 checkpoint 间隔

Llama-3 405B 的 54 天窗口给了一组可算账的中断统计（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.3.4；逐类分解表归 7.3 §2.4）：466 次作业中断（47 次计划内 + 419 次意外）。平均中断间隔（mean time between failures，MTBF）：

$$
\mathrm{MTBF} \approx \frac{54 \times 86400}{466} \approx 1.0 \times 10^{4} \thinspace \text{s} \approx 2.8 \thinspace \text{小时} \qquad (2)
$$

同步训练下，单个 GPU 故障就可能要重启整个作业（§3.3.4 原文）。论文记录的系统性对策是：压缩作业启动与 checkpoint 时间；用 NCCL flight recorder（集合通信元数据与 stack trace 进 ring buffer、watchdog 超时自动 dump）做快速定界；用慢节点（straggler）侦测工具抓「仍在线但变慢」的卡——单个 straggler 可以拖慢数千块 GPU，对策是按进程组优先排查嫌疑最大的通信（同节口径）；故障自动处理。整个窗口显著人工干预仅 3 次，有效训练时间保持在 $90\text{％}$ 以上。

健康检查与自动替换的闭环形态，从这组记录里可以直接读出：检测（flight recorder / watchdog / straggler 工具）→ 定位到具体 rank 与节点 → 隔离替换 → 从最近 checkpoint 按 §5 的语义恢复。闭环里每一环的时延都直接进 $R$。

checkpoint 间隔多久一次最划算？先把结论摆出来。**命题**（最优存档间隔）：存档开销与故障损失之和最小时，间隔正比于「单次停顿成本 × 平均故障间隔」的平方根；重启成本只抬总损失，不进最优条件。**证明路线**：写出单位时间浪费率的两项账，对间隔求导，二阶导确认极小。设每 $T$ 秒计算时间存一次档、单次 GPU 停顿成本 $\delta$ 秒、重启成本 $R$ 秒；故障间隔均值即式 (2) 的 $\mathrm{MTBF}$ 。均匀故障位置假设下，一次故障的平均丢失工作为 $T/2$ 。单位时间浪费率

$$
C(T) = \underbrace{\frac{\delta}{T}}_{\text{存档开销}} + \underbrace{\frac{T / 2 + R}{\mathrm{MTBF}}}_{\text{故障损失}} \qquad (3)
$$

对 $T$ 求导： $-\delta / T^{2} + 1 / (2 \thinspace \mathrm{MTBF}) = 0$，二阶导 $2\delta / T^{3} \gt 0$ 确认为极小：

$$
T^{\ast} = \sqrt{2 \thinspace \delta \cdot \mathrm{MTBF}} \qquad (4)
$$

重启成本 $R$ 不进最优间隔（它在式 (3) 里是可加常数 $\mathrm{MTBF}^{-1} R$ ），只抬总损失。这个一阶模型是 checkpoint 文献的经典结论形式，此处为本书自含推导。

代入两组数字看机制价值（MTBF 取式 (2) 的 $10^{4}$ s）：

| 模式 | $\delta$ | $T^{\ast}$ | 最优点的浪费率 $\sqrt{2\delta / \mathrm{MTBF}}$ |
| --- | --- | --- | --- |
| 阻塞写（ §2 下界 2.8 s） | 2.8 s | $\approx 237$ s（约 4 分钟） | $\approx 2.4\text{％}$ |
| 异步（staging 暴露 0.3 s，本书估计口径） | 0.3 s | $\approx 77$ s | $\approx 0.8\text{％}$ |

读法： $\delta$ 越小，最优间隔**越短**、浪费率越低——存档便宜了（口径：单次存档造成的 GPU 停顿变短），就值得更频繁地存。这正是 herd 「minimize GPU pause time + increase checkpoint frequency」（§3.3.1 原文）的定量形态：异步机制把 $\delta$ 压低一个量级，换来更高频率与更低故障损失。工程注记两条。第一，实际间隔不会顶到式 (4) 的分钟级下限——元数据开销、存储回收、评测节奏等式外因素把实践值推到分钟到小时档（本书工程口径）。第二，故障恢复的总损失还有 $R$（重启 + 重载 + 重建连接）一项；herd 把「压缩启动时间」与「压缩 checkpoint 时间」并列为提效手段（§3.3.4），对应的就是压 $R$。

弹性训练（elastic training）一句：PyTorch 的 elastic 配套（`torch.distributed.elastic` / torchrun 的 rendezvous 机制，pytorch repo，教学参照）支持 worker 增删不终止整个作业；它与 §3 的并行度无关格式组合才是完整闭环——成员变化后按新网格加载、全局游标不动，续训语义逐字成立。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| checkpoint 五分量 | 参数 / 优化器态（含 step 计数）/ 三路 RNG / 全局数据游标 / schedule 相位 | §1 表；pytorch-repo（教学参照） |
| 字节账 | 式 (1)： $B_{\mathrm{ckpt}} = 14N$ bytes（bf16 参数 + fp32 master + fp32 $m,v$ ）；405B ≈ 5.7 TB，16384 卡每卡 ≈ 0.35 GB，落在 herd 的 1MB-4GB 带内 | models 1.2 式 (8)；herd §3.3.1 [A] |
| 写盘时间 | 2 TB/s 可持续口径下 405B 全量 ≈ 2.8 s ≈ 半个 step（Table 4 配置 step ≈ 5.9 s） | herd §3.3.1 / Table 4 [A] |
| resharding | 逻辑张量 + 全局 offset 元数据，保存 / 加载网格解耦；DCP 的 SPMD save 与 planner；Megatron dist-ckpt 归 6.2 §5 | pytorch repo DCP 文档；arXiv:1909.08053 repo |
| 异步 checkpoint | staging（默认 CPU）+ 后台线程写盘；`AsyncSaveResponse` 双 future；staging 完成前禁改被存张量 | DCP `async_save` 文档口径（experimental）；练习 1（c） |
| 续训语义 | 优化器状态热 ⇒ 免重新 warmup；状态丢 ⇒ 重 warmup；数据游标全局化保 exactly-once | §5（本书论证，基于 2.2 / 2.3） |
| MTBF 与最优间隔 | 式 (2)：466 次 / 54 天 ⇒ MTBF ≈ 2.8 h；式 (4)： $T^{\ast} = \sqrt{2\delta \cdot \mathrm{MTBF}}$ ；异步压 $\delta$ ⇒ 更密更省 | herd §3.3.4 [A]；式 (3)(4) 本书推导 |

## 练习

1.（计算 · [20]）字节与停顿账：某 MoE 模型总参数 $N = 1 \times 10^{12}$、bf16 + AdamW（fp32 master 与矩估计），DP 分片 4096 卡。(a) 按式 (1) 算全量 checkpoint 字节数与每卡分片；(b) 存储可持续吞吐 2 TB/s 时阻塞写的下界时间；(c) 若每 step 2.0 s，求阻塞写相对 step 的占比，并说明异步 staging 把哪一段从 GPU 关键路径上移走。校验：全量字节数应为参数量的 14 倍（byte 口径）。

2.（推导 · [30]）式 (4) 的两个推广：(a) 存档成本改为 $\delta_0 + \delta_1$（ $\delta_0$ 为 GPU 停顿、 $\delta_1$ 为不占 GPU 的后台成本），说明只有 $\delta_0$ 进最优间隔；(b) 故障间隔不是指数分布而是「恰好每 $M$ 秒一次」的周期故障，重推最优 $T$ 应满足的离散条件，并说明均匀位置假设在哪一步进入了式 (3)。

3.（计算 · [30]）MTBF 的规模律：设单卡年均故障率使单卡 MTBF 为 $M_1$ ，集群 $d$ 卡、故障独立，作业级 MTBF $\approx M_1 / d$ 。(a) 用式 (2) 的 2.8 h 与 $d$ = 16384 反推 $M_1$ 的年化口径；(b) 外推到 $d$ = 131072（8 倍）时作业级 MTBF 与式 (4) 下的最优间隔（ $\delta$ = 0.3 s）各变为多少；(c) 讨论这个外推在多大程度上可信（提示： §3.3.4 的故障构成里哪些随 $d$ 线性、哪些不是）。校验：（b）的作业级间隔应恰好缩到八分之一。

4.（代码 · [30]）教学实现续训一致性检查（教学用，非优化实现）：对一个两层 MLP 跑 30 步 AdamW，第 10 步末按 §1 的五分量存 checkpoint；先写出你对（b）的预测——轨迹偏离从哪一步开始——再动手。分支 A 继续跑到 30 步，分支 B 在第 10 步读档后跑到 30 步；(a) CPU 固定 seed + `torch.use_deterministic_algorithms(True)` 下比较两分支终参的逐位一致性；(b) 故意不恢复 CUDA / python RNG 分量，比较 loss 轨迹的偏离步点；(c) 故意把 optimizer 的 step 计数器清零，观察 bias correction 因子错配带来的前若干步更新偏差（回链 2.2 式 (4)）。校验：（a）两分支终参逐位一致（容差为零）。

## 参考

- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照+官方实现（key: pytorch-repo）——`torch/distributed/checkpoint/state_dict_saver.py`（`save` 的 SPMD 语义、`async_save` 两阶段与 `AsyncSaveResponse` 双 future、`AsyncCheckpointerType`）、`torch/distributed/checkpoint/staging.py`（`AsyncStager`）；RNG API（`torch.get_rng_state` / `torch.cuda.get_rng_state_all`）；`torch.distributed.elastic` 一句；文档口径核实于 2026-11-17
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——repo `megatron/core/dist_checkpointing/` 的 `ShardedTensor` 元数据与保存 / 加载网格解耦（机制走读归 6.2 §5）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——§3.3.1 存储口径（240 PB / 7500 服务器 / 2 TB/s 可持续 / 7 TB/s 峰值、每 GPU 1MB-4GB、bursty writes、minimize GPU pause time）、§3.3.4 中断统计（466 / 47 / 419、3 次人工干预、 $90\text{％}$ + 有效训练时间、NCCL flight recorder、功率波动）、Table 4（16384 卡配置与 16M tokens/batch、400 TFLOPS/GPU、MFU $41\text{％}$ ）
- [Zhao et al., 2023](../references.md)：PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel，arXiv:2304.11277，[A]（key: fsdp）——全分片模型状态语义（分片 checkpoint 的动机，走读归 4.3）
