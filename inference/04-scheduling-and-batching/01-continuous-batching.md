# 4.1 Continuous Batching 与迭代级调度（Continuous Batching and Iteration-Level Scheduling）

> batch 里明明还有请求没做完，GPU 为什么在给已完成的请求空转，新请求却只能排在 batch 外面？本章推导 static batching 的三种浪费账与利用率上界，给出 Orca 迭代级调度与 selective batching 的机制和正确性命题，把吞吐提升算成结构结论，并将每步调度循环对到 vLLM V1 与 SGLang 的 scheduler 代码 | 前置：inference 1.1（TTFT / TPOT 指标）、1.2（prefill / decode 账、算术强度与带宽下界步时）、2.1（paged 块管理、slot、watermark）、models 2.4（KV cache 容量与前缀不可变命题） | 锚点：Orca（[Yu et al., 2022](../references.md)，USENIX OSDI'22，无 arXiv，[B]）；vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）；SGLang（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]）；机制片段 `vllm/v1/engine/core.py` + `v1/core/sched/` 与 SGLang `managers/scheduler.py` + `schedule_batch.py`（2026-11-23 main 核实；模块全量走读见 5.2 / 6.2）

## 1. static batching 的三种浪费：利用率账

8 条请求同时进 batch，输出长度是 7 条 64 token 加 1 条 512 token。第 64 步之后，7 个 slot 的请求已经完成，但 batch 不能散——GPU 继续按 8 条的满批跑剩下的 448 步，其中 7/8 的算力花在已完成的请求上。这不是病态构造，而是 static batching 的常态。

所谓 static batching（静态批处理），是指以请求为粒度的 run-to-completion 调度：batch 一旦组成就连续执行，直到批内**所有**请求完成；期间既不允许新请求加入，也不允许已完成请求离开。三种浪费都由这条「同进同出」约束产生：

- **早 finish 空转**：短请求完成后，它的 slot 仍被占用并参与每步前向，直到最长请求结束。
- **head-of-line blocking（队头阻塞）**：新到达的请求必须等当前 batch 整体完成才能进批；最坏等待 = 当前 batch 的剩余步数。
- **padding 浪费**：同批请求 prompt 长度不齐时，prefill 按最长 prompt 补齐（pad）成等长 tensor，补出来的 token 是纯计算浪费。

先量化空转。设 batch 内 $B$ 条请求的输出长度为 $T_{o,i}$（本章 $B$ = batch 内请求数； $T_o$ = 输出 token 数，prompt 长记 $T_p$），每步每 slot 至多产出 1 个有用 token，则 slot 利用率：

$$
\eta = \frac{\sum_{i=1}^{B} T_{o,i}}{B \cdot \max_i T_{o,i}} \qquad (1)
$$

**符号**： $\eta$ = slot 利用率（口径：有用 token 产出占「slot × 步数」总容量的比例，不是 1.3 的排队利用率 $\rho$）； $T_{o,i}$ = 第 $i$ 条请求的输出长度；分母 = batch 占用的总 slot-步数。

**命题 1**（static 利用率上界）： $\eta \le 1$，等号成立当且仅当所有 $T_{o,i}$ 相等。**证明**：分子是 $B$ 个非负项之和，每项 $T_{o,i} \le \max_j T_{o,j}$，故分子 $\le B \max_j T_{o,j}$ = 分母；等号要求每项都取到最大值，即所有输出长度相等。证毕。正例：8 条全 64 token， $\eta$ = 1。边界反例：只要有一条输出长度不同， $\eta$ 严格小于 1——输出长度分布越长尾，空转越重。

代入开场 workload： $\sum T_{o,i}$ = 7 × 64 + 512 = 960， $B \max T_{o,i}$ = 8 × 512 = 4096， $\eta$ = 960 / 4096 ≈ 23.4%。近八成的 slot-步容量被空转吃掉。

padding 是另一笔独立的账。设同批 prompt 长度为 $T_{p,i}$，prefill 补齐到 $\max_i T_{p,i}$，浪费的 token 数与相对浪费：

$$
\mathrm{waste}_{\mathrm{pad}} = B \cdot \max_i T_{p,i} - \sum_{i=1}^{B} T_{p,i}, \qquad \frac{\mathrm{waste}_{\mathrm{pad}}}{B \cdot \max_i T_{p,i}} \qquad (2)
$$

**符号**： $T_{p,i}$ = 第 $i$ 条请求的 prompt 长度；浪费按 token 数计，FLOPs 浪费 ≈ $`2N \cdot \mathrm{waste}_{\mathrm{pad}}`$（1.2 式 (1) 的 prefill 线性项口径，attention 二次项另计）。代入教学 workload：7 条 prompt 512 token 加 1 条 2048 token，补齐后算 8 × 2048 = 16384 token，有用仅 5632，浪费 10752 token（65.6%）；按 8B 口径 $N = 7.50 \times 10^{9}$ 折 FLOPs ≈ $1.61 \times 10^{14}$，以 50% 峰值算力（494.75 TFLOPS，2.4 教学假设口径）折合约 0.33 s——一次 prefill 的大半算力花在 pad 上。

队头阻塞的账最直白：新请求最坏要等当前 batch 跑完全部剩余步数。开场 workload 里，batch 刚起步时到达的请求要等 512 步；按 §4 式 (3) 的步时口径（5.13 ms/步）折合约 2.6 s 的纯排队，且这段等待与它自身的 prompt 长短无关。

三种浪费的时间线对照如下（ASCII 重绘自 Orca Fig.1 的示意结构，[Yu et al., 2022](../references.md)，[B]；x 轴 = iteration，R1–R4 为四条请求，■ = 有用计算，□ = 空转，↓ = 到达）：

```
static batching（run-to-completion）
        iter 1    2    3    4    5    6    7    8
R1(2tok) ■    ■    □    □    □    □    □    □
R2(4tok) ■    ■    ■    ■    □    □    □    □
R3(3tok) ■    ■    ■    □    □    □    □    □
R4(8tok) ■    ■    ■    ■    ■    ■    ■    ■
R5 ↓iter3 到达 ────────────── 等待整批结束 ──────────→ iter9 才进批

continuous batching（iteration-level）
        iter 1    2    3    4    5    6    7    8
R1(2tok) ■    ■    离开
R2(4tok) ■    ■    ■    ■    离开
R3(3tok) ■    ■    ■    离开
R4(8tok) ■    ■    ■    ■    ■    ■    ■    ■
R5 ↓iter3 到达      ■    ■    ■    ■    ■    ■   ← 立即补位
```

**本章数值口径**（全章例题复用，与 1.2 / 2.1 章脊一致）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、bf16）， $\kappa L$ = 128 KiB/token； $N = 7.50 \times 10^{9}$（matmul 口径），权重字节 $W = 1.61 \times 10^{10}$ B；硬件 H100 SXM（bf16 dense 峰值 989.5 TFLOPS、HBM 3.35 TB/s，[NVIDIA datasheet](../references.md)，[C] 官方规格披露，完整表见 1.3）。workload 用本节教学构造（8 条、输出 7 × 64 + 512），换口径处逐节声明。

## 2. Orca：迭代级调度

能不能把调度粒度从「一条请求的一生」缩小到「一次前向」？Orca 的答案是 iteration-level scheduling（迭代级调度），摘要原文：「schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch」（[Yu et al., 2022](../references.md)，[B]，USENIX OSDI'22，无 arXiv）。所谓迭代级调度，是指调度器每次只委托执行引擎跑**一个** iteration，拿到结果后再决定下一个 iteration 的 batch 组成。

循环只有三步（Orca §3 口径）：(1) 从 request pool 选出本步要跑的请求集合；(2) 委托引擎对该集合执行一个 iteration；(3) 接收结果——完成的请求立即返回客户端并释放资源，新到达的请求在下一个 iteration 边界就有机会进批。§1 的三种浪费由此逐个消解：完成即离开（空转消失）、到达即可进批（队头阻塞消失）、batch 组成逐步重选（padding 不再被「同进同出」锁死——但变长 prompt 的合批仍需 §3 的 selective batching 才可行）。

调度自由度变大后，公平性需要重新定义。Orca 给出 iteration-level FCFS（first-come-first-served，先到先服务）性质（§4.2 原文定义）：对任意两条请求，若 $x_i$ 比 $x_j$ 早到，则 $x_i$ 已执行的 iteration 数不少于 $x_j$。注意这不保证早到的先返回——晚到的短请求可能先完成（§1 时间线里 R5 若只有 2 token，会在 R4 之前返回），FCFS 约束的是**获得服务的顺序**而非完成顺序。

Orca 的准入算法（Algorithm 1）有两个旋钮：max batch size（max_bs，批内请求数上限）与 KV slot 总数（n_slots）。关键细节是 slot 预留：initiation 阶段（从未跑过）的请求按 `max_tokens` 属性**一次性预留**全部未来 slot，预留不下就不准入（论文 §4.2，lines 23-26）。这避开了一个死锁：若不留预留，池满时调度器可能发不出任何 iteration——所有在批请求都差一个 token 的 KV 空间，而谁也无法先完成。

> **坑**：预留式准入把 2.1 式 (2) 的 reservation 浪费请了回来——按 `max_tokens` = 2048 预留而实际只生成 64 token 的请求，占着 2048 slot 不放。触发条件 = 输出长度远小于 max_tokens 的长尾 workload；后果 = 并发 batch 被预留压小、吞吐受损。Orca 用预留换无死锁，vLLM 用 paged 按需分配 + 抢占换低浪费（2.1 命题 1：每请求浪费 ≤ 1 块；抢占机制见 4.2）——两条路线的取舍是「预留保安全」对「按需 + 抢占保利用率」。

> **提点**：迭代级调度与操作系统的时间片轮转同构——进程（请求）不在 CPU（GPU）上跑完一生，而是每个时间片（iteration）重新竞争。差别在状态体积：进程上下文是 KB 级寄存器，请求上下文是 GB 级 KV cache，所以「换出」的代价账（4.2）才是 serving 调度的核心难题，而不是调度算法本身。

## 3. selective batching：哪些算子能合批

迭代级调度让调度器可以任选请求集合，但执行引擎未必能把任选的集合算成一个 batch。Orca 列出三种不能直接合批的情形（§3，C2 挑战）：(1) 两条都在 initiation 阶段但 prompt 长度不同——输入 tensor 的长度维不等；(2) 两条都在 increment 阶段但 token index 不同——attention 的 KV tensor 形状不同；(3) 两条处于不同阶段——一个吃整段 prompt、一个吃单 token。batching 的前提是「每条请求的执行由相同算子构成、每个算子吃相同形状的输入」，三种情形各破坏一条。

破局点是算子分类。非 attention 算子（QKV / FFN 投影、LayerNorm 等）是 position-wise 的：输出第 $t$ 行只依赖输入第 $t$ 行，与 batch 里有谁、各请求多长无关。把各请求的 token 沿长度维摊平（flatten）成一个 $[\sum_i T_i,\ d_{\mathrm{model}}]$ 的大 tensor，这些算子就能一次算完。attention 则逐序列私有——KV 与 causal mask 按序列定义，必须拆开各算各的。这就是 selective batching（选择性批处理），Orca §1 原文：「splits the batch and processes each request individually for the Attention operation while applying batching to other operations」；§4 又称 token-wise (instead of request-wise) batching。

执行流是 Split → 逐序列 Attention → Merge（Orca Fig.5 口径：7 个 token 的批，非 attention 算子输入 $[7, H]$，attention 前插 Split、后插 Merge）：

```
请求 A(2 tok) ─┐                    ┌─ Attn(A: q[2], KV_A) ─┐
请求 B(3 tok) ─┼─ Flatten [7,4096] ─┤  Attn(B: q[3], KV_B)  ├─ Merge [7,4096] ─→ FFN/LN…
请求 C(2 tok) ─┘   QKV/FFN/LN 合批  └─ Attn(C: q[2], KV_C) ─┘
                   （权重只读一遍）     （逐序列，KV 私有）
```

**命题 2**（selective batching 等价性）：设算子 $F$ 是 position-wise 的（输出每行只依赖输入对应行），则对输入行的任意划分，分块计算再拼接的结果与整批计算逐行相等；attention 不是 position-wise 的，跨序列合并计算会改变输出。**证明**（要点）：position-wise 情形由行独立性直接得——每行的计算不读其他行，划分不改变任何行的输入；attention 情形，合并后 token 的可见集扩到另一序列的 KV，softmax 归一化域改变，输出随之改变（前缀不可变命题保证逐序列计算本身正确，models 2.4）。完整证明留作练习 2。正例：LayerNorm 逐行归一化、GEMM 逐行乘同一权重矩阵，均满足行独立。边界反例：把两条序列的 token 合进同一个 attention 而不加隔离 mask，序列 A 的 token 会 attend 到序列 B 的 KV——输出与独立计算不同，且泄漏了另一请求的内容。

合批的收益落在权重读上。代入教学实例：3 条请求各 2 / 3 / 2 个 token，QKV 投影权重 $[4096, 6144]$（q 32 × 128 = 4096 维、kv 2 × 8 × 128 = 2048 维，8B 口径）。合批 GEMM $[7, 4096] \times [4096, 6144]$：FLOPs = 2 × 7 × 4096 × 6144 ≈ $3.52 \times 10^{8}$，权重只读一遍 = 4096 × 6144 × 2 ≈ $5.03 \times 10^{7}$ B，算术强度 $I$ = 2 × 7 / 2 = 7 FLOP/byte（bf16 口径，1.2 式 (4) 的 $I \approx B$ 在 token 维的同形结论）。逐请求算则是 3 次 GEMV，权重读 3 遍、强度只有 2–3 FLOP/byte——离 ridge point（H100 bf16 ≈ 295，1.3 式 (1)）同样远，但访存流量差 3 倍。selective batching 不改变 FLOPs，改变的是权重流量与 kernel launch 次数。

## 4. continuous batching 机制与吞吐账

迭代级调度 + paged 按需分配，到底换来多少吞吐？先给结论：饱和到达、纯 decode 口径下，增益是 static 利用率的倒数 $1/\eta$——这是结构结论，不是拟合值。所谓 continuous batching（连续批处理），是指 batch 组成每个 iteration 更新、请求随时加入与离开的 serving 形态（术语对应：Orca 的 iteration-level scheduling 是机制源头，continuous batching 是引擎文档的通用名）。它比 Orca 原始形态多一件武器：slot 不再按 max_tokens 预留，而是 paged 按需分配（2.1），准入失败的请求排队或被抢占，而不是把池占死。

机制三件套：

- **slot 管理**：slot = 一个 token 的 KV 存储位（2.1 口径）。paged 池按需给块，每请求浪费 ≤ 1 块；请求完成即释放块回池。
- **running / waiting 队列**：running = 已准入、持有 KV 块的请求；waiting = 到达但尚未准入的请求。每步调度先保 running（decode 不断流），再用剩余预算准入 waiting。
- **每步循环**：schedule（定组成、分块）→ execute（一次前向 + 采样）→ update（追加 token、判完成、释放 / 回收）。

```
            ┌────────────────────────────────────────┐
            │              每步循环                    │
 new req ──→│ waiting ──准入──→ running ──完成──→ 返回  │
            │   ↑    (token预算+块可用)  │              │
            │   └────── 抢占(4.2) ──────┘              │
            │ schedule → execute(1 iteration) → update │
            └────────────────────────────────────────┘
```

准入受两个预算约束：**token 预算**（一步最多算多少 token，vLLM `max_num_batched_tokens`，`config/scheduler.py` L47，2026-11-23 核实）与**并发上限**（批内最多多少请求，`max_num_seqs`）；此外还有块池水位（watermark 只约束 waiting / preempted 准入，2.1 §4，触发与抢占账见 4.2）。prefill 请求一步吃掉的 token 数 = prompt 长（或 chunk 长，4.3），decode 请求每步吃 1 个——两类请求在同一个 token 预算下竞争，这是 §7 stall 问题的根源。

吞吐账现在可以算了。步时用 1.2 的带宽下界口径 recap（memory-bound decode，1.2 式 (4) 分母 ÷ 带宽；不重复推导）：

$$
t_{\mathrm{step}}(B, \bar{T}) \approx \frac{W + B \thinspace \kappa L \thinspace \bar{T}}{\beta_{\mathrm{HBM}}} \qquad (3)
$$

**符号**： $W$ = 权重字节（全批共享、每步读一遍）； $\kappa L$ = 每 token 全层 KV 字节； $\bar{T}$ = 批内平均序列长（prompt + 已生成）； $\beta_{\mathrm{HBM}}$ = 显存带宽。口径：带宽下界，忽略 attention 计算与 kernel 开销；「等步时近似」指 $t_{\mathrm{step}}$ 不随 batch 组成变化—— $\bar{T}$ 漂移时该近似失效（下方坑注）。代入章脊： $B$ = 8、 $\bar{T}$ = 1024（教学口径：平均在批序列长）， $t_{\mathrm{step}}$ = (1.61 × 10^10 + 8 × 131072 × 1024) B ÷ (3.35 × 10^12 B/s) ≈ 5.13 ms。

**命题 3**（continuous vs static 吞吐账）：在饱和到达（waiting 队列不空）、等步时近似与**纯 decode 口径**（批内 $B$ 条请求全在 decode，忽略新准入请求的 prefill 步；prefill 混入时增益低于 $1/\eta$）下，产出同样 $\sum_i T_{o,i}$ 个有用 token，static batching 每批需 $\max_i T_{o,i}$ 步，continuous batching 需 $\sum_i T_{o,i} / B$ 步，吞吐增益为 $1/\eta$。**证明**（路线：两边各数「步数 × 每步有用 token」，关键一步是饱和到达下 continuous 每步产出 $B$ 个有用 token）：static 批组成锁定，跑 $\max T_{o,i}$ 步、产出 $\sum T_{o,i}$ 个有用 token；continuous 每步都有 $B$ 条未完成请求在跑（完成即补位），每步产出 $B$ 个有用 token，故 $\sum T_{o,i} / B$ 步完成。步时相同，时间比 = 步数比 = $B \max T_{o,i} / \sum T_{o,i} = 1/\eta$（式 (1)）。证毕。

$$
\frac{\mathrm{throughput}_{\mathrm{continuous}}}{\mathrm{throughput}_{\mathrm{static}}} = \frac{1}{\eta} = \frac{B \cdot \max_i T_{o,i}}{\sum_i T_{o,i}} \qquad (4)
$$

**符号**：同式 (1)；口径：同一 workload、同一 $B$、饱和到达、等步时近似、纯 decode（两边同略新请求 prefill 步，口径对齐；prefill 混入时增益低于 $1/\eta$）下的结构结论，不是实测拟合值。代入 §1 workload：static 512 步 × 5.13 ms ≈ 2.62 s，吞吐 960 / 2.62 ≈ 366 tokens/s；continuous 120 步 × 5.13 ms ≈ 0.62 s，吞吐 ≈ 1561 tokens/s；增益 4.27 = 1 / 0.234。输出长度越长尾（ $\eta$ 越低），增益越大——这解释了为什么 agent / 推理类长输出 workload 上 continuous batching 的收益比短问答更可观。

实测校准：Orca 在 GPT-3 175B 上报告同延迟水平下 36.9× 吞吐提升，对照 NVIDIA FasterTransformer（[Yu et al., 2022](../references.md)，[B]，摘要口径）。口径注：36.9× 是**系统级**对比——baseline 的浪费不止空转（还有 padding、队头阻塞、预留显存压低并发），Orca 侧还含分布式执行引擎与 selective batching 的贡献；式 (4) 只计空转一项，不能直接对账 36.9×，但方向一致：baseline 浪费越多，迭代级调度的增益越大。

> **坑**：等步时近似在序列长方差大时失效——式 (3) 的 KV 读项正比于 $\bar{T}$，长请求进出 batch 会逐步改变步时，式 (4) 的增益随之偏差。触发条件 = 长短请求混跑（如 128K 长文与短问答同池）；后果 = 按式 (4) 估的吞吐偏乐观或偏悲观。规避 = 精确账按逐步模拟（步时随组成更新），容量规划口径见 1.3 排队论与 9.1。

## 5. vLLM V1 调度循环走读（机制例证级）

三步循环在代码里长什么样？vLLM V1 的 `EngineCore.step()`（`vllm/v1/engine/core.py` L714，2026-11-23 main 核实）就是 schedule → execute → update 三段（机制例证摘录，模块全量走读见 5.1 / 5.2）：

```python
# vllm/v1/engine/core.py L714-744（摘录，教学用省略错误处理）
def step(self):
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule(...)      # L724 定组成
    future = self.model_executor.execute_model(          # L727 一次前向
        scheduler_output, non_block=True)
    model_output = self._wait_for_output(future)
    engine_core_outputs = self.scheduler.update_from_output(  # L738 更新
        scheduler_output, model_output)
    return engine_core_outputs, ...
```

`Scheduler.schedule()`（`vllm/v1/core/sched/scheduler.py` L702）的核心是一个统一 token 视图，源码注释逐字：「There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec.」调度器不给请求贴 prefill / decode 标签，只盯一个差值：`num_tokens_with_spec − num_computed_tokens` = 还差多少 token 没算。每步给各请求分配 token 让这个差值归零——prefill 是差值 = prompt 长的特例，decode 是差值 = 1 的特例，chunked prefill（差值分多步还，4.3）与投机解码（差值含 draft token，4.4）都是同一循环的特例。

循环体内两段顺序是硬语义：

1. **先 running**（L778 起）：逐请求算 `num_new_tokens`，被 token 预算与 `long_prefill_token_threshold` 钳位（L829-833）；然后 `allocate_slots` 分块。分配失败进入抢占循环——选 victim、释放块、重试（L898-969），成本账与 victim 选择见 4.2。
2. **后 waiting**（L1040 起）：仅当本步**没有发生抢占**时才准入新请求（L1041 条件 `if not preempted_reqs`）；准入前查 prefix cache 命中（2.2），命中部分不占计算预算；watermark 门只在这里生效（2.1 §4）。

`update_from_output` 收尾：追加采样 token、判完成（EOS / max_tokens）、释放完成请求的块——释放的块下一步就能准入 waiting 请求，「完成即补位」由此闭合。async 变体 `step_with_batch_queue`（L759）把 schedule 与 execute 流水化（batch queue 里同时有多个在途步），语义不变、时序重叠，架构账见 5.1。

机制 ↔ 代码对照：

| 机制概念（§4） | vLLM V1 落点（2026-11-23 main） |
| --- | --- |
| 每步循环 schedule → execute → update | `engine/core.py` `step()` L714 |
| 统一 token 视图（无 phase 标签） | `core/sched/scheduler.py` `schedule()` L702 注释 |
| running 优先、token 预算钳位 | 同上 L778 / L829-833 |
| 准入条件：无抢占 + 预算 + 块 + watermark | 同上 L1040-1041、`kv_cache_manager.py` L529-536 |
| 完成即释放、块回池 | `update_from_output` + `block_pool.py`（2.1） |
| 抢占（分配失败触发） | 同上 L898-969 / `_preempt_request` L1831（4.2） |

## 6. SGLang 调度循环走读（机制例证级）

SGLang 的循环骨架相同，batch 组织不同：prefill 批与 decode 批是两个对象，在步边界合并。`event_loop_normal`（`python/sglang/srt/managers/scheduler.py` L2231，2026-11-23 main 核实）：

```python
# sglang/srt/managers/scheduler.py L2231-2262（摘录）
def event_loop_normal(self):
    while True:
        self.ingest_requests()                    # 收新请求进 waiting
        plan = self.get_next_batch_to_run(        # L4386 定本步 batch
            running_batch=self.running_batch, last_batch=self.last_batch)
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        if batch:
            result = self.run_batch(batch)        # L5455 一次前向
            self.process_batch_result(batch, result)  # 更新/判完成
        self.last_batch = batch
```

`get_next_batch_to_run`（L4386）做三件事：(1) 把上一步的 prefill（extend）批合并进 `running_batch`——prefill 完成的请求从此转入 decode，未完成的 chunked 请求（`chunked_req`）除外（4.3）；(2) 过滤已完成请求；(3) 二选一：尝试组新 prefill 批（`get_new_batch_prefill`，L4553，从 waiting 按 `SchedulePolicy.calc_priority` 排序准入——策略全集见 4.2），否则更新 decode 批（`update_running_batch`，L5262）。decode 批每步先查 `check_decode_mem`（`schedule_batch.py` L3007）：下一步的 KV 增量装不下就触发 `retract_decode`（L3030，SGLang 的抢占形态，账见 4.2）。

数据结构在 `schedule_batch.py`：`Req`（L442）持单请求状态（input / output token、prefix 匹配节点），`ScheduleBatch`（L2009）持批状态与 `forward_mode`（EXTEND = prefill 形态、DECODE = 逐 token 形态）。与 vLLM 的统一 token 视图对照：SGLang 在批级区分两种 forward mode，vLLM 在请求级只记 computed / total 差值——两种组织都能表达 chunked prefill 与混合批（4.3），差别在状态机的粒度。

`event_loop_overlap`（L2266）是重叠变体：CPU 调度与 GPU 前向重叠执行，本步 launch 后才处理上一步结果（`result_queue` 缓存）。机制细节与 bubble 消除账归 6.2，本章只登记一句：重叠改的是循环时序，不改「每步重选 batch」的迭代级语义。

| 循环阶段 | vLLM V1 | SGLang |
| --- | --- | --- |
| 收请求 | frontend → EngineCore 队列（5.1） | `ingest_requests` |
| 定组成 | `schedule()`：running 先、waiting 后 | `get_next_batch_to_run`：merge + prefill 批优先 |
| 执行 | `execute_model` | `run_batch` |
| 更新 | `update_from_output` | `process_batch_result` |
| 抢占 | 分配失败 → preempt（4.2） | `check_decode_mem` 失败 → retract（4.2） |

## 7. 边界：迭代级调度不解决什么

batch 组成每步更新，调度就没有问题了吗？三个问题仍在，各有专章。其一，**长 prefill 独占一步**：迭代级调度只在步边界换组成，不能打断正在执行的前向——一条 32K prompt 的 prefill 会让同批 decode 请求的 TPOT 尖刺，解法是把 prefill 切成 chunk 与 decode 混批（chunked prefill，4.3）。其二，**每步 CPU 调度开销**：schedule 与 update 是 CPU 工作，步时越短占比越高，解法是 overlap / async 调度（6.2 / 5.1）与算子层流水（3.4 NanoFlow）。其三，**prefill 与 decode 的资源竞争是结构性的**：同一张卡上两类负载的 SLO 互相牵制，迭代级调度只能缓解不能消除，彻底解法是把两阶段拆到不同实例（PD 分离，8.2）。投机解码的 verify 步是多 token 的 prefill 形态，其调度含义见 4.4 / 4.5。

## 总结

| 论断 | 回指 |
| --- | --- |
| static batching 三浪费：早 finish 空转、队头阻塞、padding；利用率 $\eta = \sum T_{o,i} / (B \max T_{o,i}) \le 1$，等号当且仅当输出等长 | 式 (1)(2)、命题 1、§1 |
| 迭代级调度 = 每 iteration 重选 batch；iteration-level FCFS 约束服务顺序而非完成顺序 | §2、练习 4 |
| Orca 准入按 max_tokens 预留 slot 防死锁，代价是 reservation 浪费回归；paged 按需 + 抢占是另一条路线 | §2、2.1 命题 1 |
| selective batching：position-wise 算子摊平合批、attention 逐序列；等价性由行独立性保证，跨序列 attention 改变输出 | 命题 2、§3、练习 2 |
| 合批不减 FLOPs，减权重流量：7 token 合批强度 7 FLOP/byte，逐请求 2–3 | §3、1.2 式 (4) |
| 饱和到达 + 等步时近似 + 纯 decode 口径下吞吐增益 = $1/\eta$（结构结论）；教学 workload 4.27×；Orca 实测 36.9× 是系统级口径 | 式 (3)(4)、命题 3、§4 |
| vLLM V1：统一 token 视图（无 phase 标签），running 先 waiting 后，无抢占才准入 | §5、练习 4 |
| SGLang：prefill / decode 两批对象步边界合并；overlap 变体改时序不改迭代级语义 | §6 |
| 迭代级调度不解决长 prefill stall（4.3）、CPU 调度开销（6.2）、P/D 结构竞争（8.2） | §7 |

## 练习

1.（计算 · [10]）4 条请求同时到达，输出长度分别为 64 / 128 / 256 / 512 token：(a) 按式 (1) 算 static batching 的 $\eta$；(b) 按式 (4) 算饱和到达下 continuous batching 的吞吐增益；(c) 若 4 条输出都改为 256， $\eta$ 与增益各是多少？校验：(c) 应落在命题 1 的等号条件上（ $\eta$ = 1、增益 = 1）；(a) 的分子 = 960。
2.（证明 · [20]）补全命题 2 的完整证明：(a) 对 position-wise 算子，证明任意行划分下「分块计算再拼接 = 整批计算」（从行独立性定义出发，逐行论证）；(b) 构造一个跨序列合并 attention 改变输出的最小反例（两条序列、各 1 个 query token 即可），指明 softmax 归一化域如何变化；(c) 说明 $B$ = 1 时 selective batching 退化为逐请求执行。校验：(b) 的反例应违反前缀不可变命题的可见集约束（models 2.4）；(c) 与式 (4) 在 $B$ = 1 时增益 = 1 一致。
3.（计算 · [30]）预留式 vs paged 准入账（章脊口径，KV 池 48 GiB，与 2.1 §1 同池）：请求 max_tokens = 2048、实际平均长度 500 token。(a) Orca 式按 max_tokens 预留，池装得下多少条并发？(b) paged 按需（ $b$ = 64），装得下多少条？(c) 用 1.2 的 $I \approx B$ recap，估两种准入下权重 GEMM 的算术强度比，并说明 decode 吞吐的方向。校验：(a)(b) 应分别与 2.1 §1 的 192 / 768 一致（该节用 $b$ = 16，本练 $b$ = 64 时 $\lceil 500/64 \rceil \times 64$ = 512 slot，结论不变）；(c) 强度比 = 并发比。
4.（代码 · [30]）读 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`（main 2026-11-23）：先写出预测——(a) running 请求的 `num_new_tokens` 由哪两个字段之差决定？(b) waiting 队头请求 prompt 8192 token、本步 token 预算剩 4096，这一步与下一步各给它算多少 token？(c) 为什么这套设计让 chunked prefill 不需要独立的调度路径？再对照 L702 注释与 L824-833 的钳位逻辑验证预测。校验：源码 `num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens` 与 `min(num_new_tokens, token_budget, ...)`；注释「There's no "decoding phase" nor "prefill phase" in the scheduler」。

## 参考

- [Yu et al., 2022](../references.md)：Orca: A Distributed Serving System for Transformer-Based Generative Models，USENIX OSDI'22，无 arXiv（PDF usenix.org/system/files/osdi22-yu.pdf），[B]（key: orca）——iteration-level scheduling 摘要与 §3、selective batching §1 / §4 / Fig.5、C2 三情形 §3、FCFS 性质与 Algorithm 1 §4.2、36.9× 摘要口径
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——paged 按需分配与抢占背景（§4.5）
- [Zheng et al., 2023](../references.md)：SGLang: Efficient Execution of Structured Language Model Programs，arXiv:2312.07104，[A]（key: sglang）——RadixAttention 与 cache-aware 调度背景（§3）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——机制片段 `v1/engine/core.py` `step()` L714 / `step_with_batch_queue` L759、`v1/core/sched/scheduler.py` `schedule()` L702 / L778 / L829-833 / L1040-1041、`config/scheduler.py` L47（均 2026-11-23 main 核实）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——机制片段 `managers/scheduler.py` `event_loop_normal` L2231 / `event_loop_overlap` L2266 / `get_next_batch_to_run` L4386 / `run_batch` L5455、`managers/schedule_batch.py` `Req` L442 / `ScheduleBatch` L2009 / `check_decode_mem` L3007（均 2026-11-23 main 核实）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——bf16 dense 989.5 TFLOPS、HBM 3.35 TB/s（步时与 padding 折算）
- 跨维度指引（纯文字）：models 2.4（KV cache 容量、前缀不可变命题）；inference 1.2（式 (1) prefill FLOPs、式 (4) 强度与带宽下界步时）、1.3（datasheet 表、ridge、排队论）、2.1（slot / 块池 / watermark / reservation 浪费）、2.2（prefix cache 命中）、4.2（抢占与策略）、4.3（chunked prefill）、5.1 / 5.2（vLLM 架构与模块全量走读）、6.2（overlap 调度）、8.2（PD 分离）、9.1（容量规划）
