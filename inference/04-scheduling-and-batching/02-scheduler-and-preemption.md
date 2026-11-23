# 4.2 调度器架构与抢占（Scheduler Architecture and Preemption）

> KV cache 池装满的那一刻，调度器该牺牲谁、用什么代价把它请出去、又凭什么决定下一个进批的是谁？本章给出三队列架构与策略空间，推导 recompute vs swap 的恢复成本账与临界带宽，拆解 FastServe 的 skip-join MLFQ 与 Llumnix 的在线迁移，并把 SGLang 九种调度策略与 vLLM 的 priority 队列对到源码 | 前置：inference 4.1（迭代级调度与每步循环）、2.1（块池、watermark、抢占语义）、2.4（PCIe 带宽口径与取回 vs 重算账）、1.2（prefill FLOPs 账）、1.1（TTFT / TBT 指标） | 锚点：FastServe（[Wu et al., 2023](../references.md)，arXiv:2305.05920，[A]，NSDI'26 发表 + LLMServe/FastServe 官方 artifact）；Llumnix（[Sun et al., 2024](../references.md)，arXiv:2406.03243，[A]，llumnix-project/llumnix-ray）；vLLM / SGLang scheduler（[A]，机制片段 2026-11-23 main 核实，模块全量走读见 5.2 / 6.2）

## 1. 调度器架构：三队列与策略空间

4.1 的循环里，请求在 waiting 与 running 之间流转；但 KV 池满时 running 请求也装不下下一步的增量，必须有第三条路。经典设计（vLLM 论文 §4.5 口径，[Kwon et al., 2023](../references.md)，[A]）是三队列：

```
                 准入(预算+块+watermark)
   waiting ─────────────────────────→ running
      ↑                                  │
      │  recompute 恢复(重排队)            │ 块分配失败 → 抢占
      │                                  ↓
      └────────────── swapped ←──── 换出 KV 到 host
         swap-in 恢复(换回即续跑)
```

- **waiting**：已到达、未准入（或抢占后重排队）的请求，不占 KV 块。
- **running**：在批请求，持有 KV 块，每步消费 token 预算。
- **swapped**：被抢占且 KV 换出到 host 内存的请求，块已释放、状态保留，swap-in 后从断点续跑。

所谓抢占（preemption），是指调度器为腾出 KV 块而把 running 请求移出批的操作；恢复有两条路——**recompute**（释放 KV，恢复时重算）与 **swap**（KV 换出到 host，恢复时换回）。两条路的成本账是 §2 的主题。版本事实先钉住：**vLLM V1 当前 main 没有 swapped 队列**——`_preempt_request` 直接释放块、把 `num_computed_tokens` 清零、请求插回 waiting 队头（`v1/core/sched/scheduler.py` L1831-1877；2026-11-23 核实该文件无 swapped 状态），即 V1 抢占只走 recompute 路；三队列是设计空间，不是每个引擎的现役实现。SGLang 的抢占叫 retract：`retract_decode` 释放设备 KV，`release_req` 可选留 host 备份（`offload_kv`），有备份从备份恢复、无备份且不可重建则 abort（`schedule_batch.py` L3030-3124）。

策略空间有两个正交维度：**准入顺序**（谁先进批）与 **victim 选择**（谁先被抢占）。vLLM 的 `SchedulingPolicy` 只有两档：FCFS / PRIORITY（`v1/core/sched/request_queue.py` L13-17）。FCFS 用 deque（先到先服务）；PRIORITY 用 heap，**priority 值小者优先、同值早到优先**（L131-144 docstring）。victim 选择随策略切换：FCFS 抢占 `running[-1]`（最晚到达者）；PRIORITY 抢占 `max(running, key=(priority, arrival_time))`（优先级最低且最晚到者）——`scheduler.py` L917-923。SGLang 的 retract 顺序另有一套：按输出长度升序 retract（最短输出先牺牲），同输出长度按输入长度降序（`schedule_batch.py` L3137-3164），且至少保留一条请求（L3043-3045）。

公平与饥饿是策略的代价面。FCFS 公平但队头阻塞（4.1 §1）；严格 priority 下低优先级请求可能永远排不上——饥饿；cache-aware 贪心（命中长的先跑）同样会饿死短前缀请求，SGLang 论文明说「greedy cache-aware scheduling ... can lead to starvation」（[Zheng et al., 2023](../references.md)，[A]，§3）。解法谱系：aging（等待越久优先级越高，§5 HRRN）、显式提升（§3 FastServe 周期性提升饥饿请求）、虚拟到达时间 boost（§5 UNIBOOST）。

> **提点**：三队列是机制，FCFS / priority / MLFQ 是策略——机制回答「状态放哪、怎么搬」，策略回答「谁先谁后」。引擎把两者分离：vLLM 换策略只换 `RequestQueue` 实现，块池与抢占机制不动；这与 4.1 selective batching 的「调度器选谁、引擎决定怎么算」是同一条设计原则的两层。

## 2. 抢占成本账：recompute vs swap

被抢占的请求恢复时要付多少代价？路线：两条恢复路各写一个成本式——recompute 是「一次 prefill 的 FLOPs ÷ 有效算力」，swap 是「往返 KV 字节 ÷ 有效拷贝带宽」；令两式相等解出临界带宽，再看它随序列长的单调性。关键一步是认出 **recompute 是一次 prefill 而不是 $T$ 次 decode**：vLLM 论文 §4.5 原文口径，已生成的 token 与原 prompt 拼成新 prompt，全部位置的 KV 在一个 prompt-phase iteration 里生成（[Kwon et al., 2023](../references.md)，[A]）。

**本章数值口径**（复用 1.2 / 2.1 / 2.4 章脊）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、bf16）， $\kappa L$ = 128 KiB/token； $N = 7.50 \times 10^{9}$（matmul 口径）；H100 SXM bf16 dense 峰值 989.5 TFLOPS（[NVIDIA datasheet](../references.md)，[C] 官方规格披露）， $F_{\mathrm{eff}}$ = 50% × 峰值 = 494.75 TFLOPS（2.4 教学假设口径）；PCIe Gen5 x16 单向 ~64 GB/s（2.4 口径）。§3 换论文 OPT-175B / OPT-2.7B 口径，原因：复现论文原数；§4 迁移算例回到章脊。

设被抢占请求当前序列长 $T$（prompt + 已生成）。recompute 成本按 1.2 prefill 账（线性项 + causal attention 二次项，2.4 同口径）：

$$
t_{\mathrm{recompute}} = \frac{2NT + 2LT^{2} d_{\mathrm{model}}}{F_{\mathrm{eff}}} \qquad (1)
$$

**符号**： $N$ = matmul 口径参数量； $L$ = 层数； $d_{\mathrm{model}}$ = hidden 宽； $F_{\mathrm{eff}}$ = 有效算力（本章取 50% × 989.5 TFLOPS = 494.75 TFLOPS，2.4 教学假设口径，实测按引擎 profile 代入）；attention 项已含 causal 减半。swap 成本是往返两份流量（换出 D2H + 换回 H2D）：

$$
t_{\mathrm{swap}} = \frac{2 \kappa L T}{\beta_{\mathrm{pcie}}} \qquad (2)
$$

**符号**： $\kappa L$ = 每 token 全层 KV 字节（章脊 128 KiB）； $\beta_{\mathrm{pcie}}$ = PCIe 单向**有效**拷贝带宽（理论值 Gen5 x16 ~64 GB/s，2.4 口径；有效值可能远低于理论，下方坑注）；因子 2 = 往返。口径：两式都忽略固定延迟项（kernel launch、拷贝栈建立），小 $T$ 时须加回。代入章脊 $T$ = 4096：KV = 4096 × 128 KiB = 512 MiB，往返 1.07 × 10^9 B，按 64 GB/s 得 $t_{\mathrm{swap}}$ ≈ 16.8 ms；recompute FLOPs ≈ 6.58 × 10^13，按 494.75 TFLOPS 得 $t_{\mathrm{recompute}}$ ≈ 133 ms——理想拷贝下 swap 快约 7.9×。

**命题 1**（恢复临界带宽）：swap 快于 recompute 当且仅当 $\beta_{\mathrm{pcie}} \gt \beta^{\ast}$，其中

$$
\beta^{\ast} = \frac{2 \kappa L T \thinspace F_{\mathrm{eff}}}{2NT + 2LT^{2} d_{\mathrm{model}}} = \frac{\kappa L \thinspace F_{\mathrm{eff}}}{N + L T d_{\mathrm{model}}} \qquad (3)
$$

且 $\beta^{\ast}$ 随 $T$ 严格单调递减。**证明**（路线：交叉相乘解不等式，再看分母单调性）：由式 (1)(2)， $t_{\mathrm{swap}} \lt t_{\mathrm{recompute}}$ 等价于 $2\kappa LT / \beta_{\mathrm{pcie}} \lt (2NT + 2LT^{2}d_{\mathrm{model}})/F_{\mathrm{eff}}$，交叉相乘（各量皆正）解出 $\beta_{\mathrm{pcie}} \gt 2\kappa LT F_{\mathrm{eff}} / (2NT + 2LT^{2}d_{\mathrm{model}})$，约去 $2T$ 得式 (3) 右形；右形分母 $N + LTd_{\mathrm{model}}$ 随 $T$ 严格增、分子常数，故 $\beta^{\ast}$ 严格减。证毕。正例： $T$ = 4096 时 $\beta^{\ast}$ ≈ 8.07 GB/s，Gen5 理论 64 GB/s 远高于它，理想拷贝下 swap 快。边界： $T \to 0$ 时 $\beta^{\ast} \to \kappa L F_{\mathrm{eff}} / N$ ≈ 8.64 GB/s（上确界）；固定延迟项主导的极小 $T$ 区间两式都失效，须加回延迟项再比。

代入章脊（8B @ H100， $\kappa L$ = 128 KiB/token）两档：

| $T$ | KV 字节 | $t_{\mathrm{swap}}$（64 GB/s 往返） | recompute FLOPs | $t_{\mathrm{recompute}}$ | 比值 | $\beta^{\ast}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 512 MiB | 16.8 ms | $6.58 \times 10^{13}$ | 133 ms | 7.9× | 8.07 GB/s |
| 32768 | 4 GiB | 134 ms | $7.73 \times 10^{14}$ | 1.56 s | 11.6× | 5.50 GB/s |

长上下文更偏向 swap：recompute 含 $T^{2}$ 项而 swap 只线性于 $T$，所以 $\beta^{\ast}$ 递减（命题 1）。 $T$ = 4096 时 attention 项只占 recompute FLOPs 的 6.7%，线性项主导； $T$ = 32768 时二次项升到 36%，差距拉开。与 2.4 §2 的账区分口径：2.4 是**单向取回**（prefix cache 命中场景，临界 ~GB/s 级），本章是**往返 swap**（抢占恢复场景），同 $T$ 同 $N$ 下临界带宽差因子 2，场景不可互换。

> **坑**：式 (2) 的 $\beta_{\mathrm{pcie}}$ 是有效带宽，不是理论带宽——paged 小块的换出把同样字节数拆成更多次小拷贝（ $b$ = 16 的拷贝次数是 $b$ = 64 的 4 倍），PCIe 效率远低于大块连续拷贝（2.1 §3：小 $b$ 下 swap 开销大；2.4：copy engine 细粒度块拷贝性能差）。触发条件 = 小 block size + 按块 swap；后果 = 实测 swap 时间远超式 (2) 理想值。vLLM 论文 §7.3 实测（OPT-13B/175B、A100）：小 $b$ 下 swap 开销过大，recompute 开销不高于 swap 延迟的 20%（论文 workload 内）——与本章理想账方向相反，原因正是有效带宽塌陷。这解释了 V1 默认只走 recompute：理想账给方向，实测有效带宽定取舍。

理想账之外还有两笔修正。其一，recompute 的实际成本可被 prefix cache 压低：被抢占请求释放的带 hash 块仍是驱逐候选而非立即失效（2.1 §4），恢复时命中前缀就免算那段——式 (1) 是上界。其二，抢占的吞吐代价要按频率摊：一次 $T$ = 4096 的 recompute 多花 133 ms GPU 时间，约等于 26 个整批 decode 步（4.1 式 (3) 的 5.13 ms/步）；抢占越频繁，稀释越重。

**watermark 触发链**（recap 2.1 §4，不重复推导）：watermark 是准入预留——`watermark_blocks = int(watermark × num_blocks)`（`kv_cache_manager.py` L200），只加在 WAITING / PREEMPTED 请求的准入需求上（L529-536），当前 main 默认 0.0（`config/scheduler.py` L232）。它不是抢占触发器，而是**抢占频率的调节器**：准入时多留空闲块，running 请求下一步增量就有缓冲，分配失败更少。真正的触发链在 vLLM 是：running 请求 `allocate_slots` 返回 None → 进入抢占循环选 victim、释放块、重试（`scheduler.py` L898-969）；在 SGLang 是：`check_decode_mem` 判下一步增量装不下 → `retract_decode`（`scheduler.py` L5275-5297）。抢占与准入不同步发生：vLLM 本步有抢占就不准入新请求（L1041），避免「刚腾出的块立刻被新请求吃掉、原请求回不来」的振荡。

## 3. FastServe：skip-join MLFQ 与 proactive swap

输出长度未知的请求流，怎么调度才能逼近「短作业优先」？SRPT（shortest remaining processing time，最短剩余时间优先）需要知道剩余工作量，LLM 的输出长度恰恰不可预知。FastServe 利用的是半信息无关（semi information-agnostic）设定：**输出长度未知，但每次 iteration 的时间可预测**——prefill 时间随输入长度增长，decode 迭代时间近似常数（论文 §4.1，Fig.5 OPT-2.7B / A100 实测口径），轻量 profiling 即可提前取得（[Wu et al., 2023](../references.md)，[A]）。

经典 MLFQ（multi-level feedback queue，多级反馈队列）是 LAS（least attained service，最少已获服务优先）的工程化近似： $n$ 个优先级队列 $Q_1, \ldots, Q_n$，quantum（时间片）逐级递增；新作业进最高优先级 $Q_1$，用完 quantum 未完成就降级。短作业在高优先级队列里就跑完，长作业逐级下沉——不需要作业大小知识就逼近 SRPT。但直接搬到 LLM 有一个两难：prefill（第一次 iteration）时间可能超过 $Q_1$ 的 quantum，此时抢占则中间激活作废重算，不抢占则队头阻塞回潮（论文 §4.1 的 strawman（稻草人方案）分析）。

skip-join（跳级入队）的解法：**按预测的 prefill 时间 $t_{\mathrm{init}}$ 选入队层级**——跳过所有 quantum 小于 $t_{\mathrm{init}}$ 的高优先级队列，直接加入满足 $q_i \ge t_{\mathrm{init}}$ 的最高优先级队列 $Q_i$（ $q_i$ = $Q_i$ 的 quantum 时间预算；Algorithm 1： $p_{\mathrm{job}} = \min i\ \mathrm{s.t.}\ q_i \ge t_{\mathrm{init}}$）；降级按下一 iteration 时间选目标队列，quantum 逐级翻倍、 $Q_1$ quantum = 最小迭代时间（论文 §4.1 实现细节）。

**命题 2**（skip-join 的 prefill 完整性）：按 skip-join 规则入队的请求，其第一次 iteration 不会因 quantum 耗尽被抢占；抢占粒度 = 单个 output token。**证明**：入队条件保证 $q_i \ge t_{\mathrm{init}}$，而第一次 iteration 耗时恰为 $t_{\mathrm{init}}$，故在 quantum 内完成；此后每次 iteration 只产一个 token，调度器在 iteration 边界决策，所以抢占只能发生在 token 边界。证毕。边界反例：profiling 低估 $t_{\mathrm{init}}$（实际 prefill 更慢，如长 prompt 触发 attention 二次项超估）时， $q_i \ge t_{\mathrm{init}}$ 的前提失效，prefill 仍可能被腰斩——预测精度是命题的前提，不是结论。

饥饿防治：skip-join 与降级会让长输入长输出请求沉底，调度器周期性把饥饿请求提升回 $Q_1$；饥饿阈值按 SLO 调，论文默认 300 ms（论文记号 $\alpha$，本章避免与接受率 $\alpha$ 冲突，以文字表述）。教学数值（quantum 口径： $q_1$ = 8 步、逐级翻倍）：输出 8 token 的短作业在 $Q_1$ 内跑完；输出 512 token 的长作业经 8 + 16 + 32 + 64 + 128 + 256 = 504 步后沉到 $Q_7$。此时到达的短作业 skip-join 进 $Q_1$，最多等一个 $Q_1$ quantum（8 步）即开跑——FCFS 下它要等长作业 512 步跑完。

MLFQ 的代价是 KV 压力：在途作业 = running + 各队列 pending，多于 run-to-completion（只保 running）。论文 Fig.8 口径：OPT-2.7B、最大输出长度 20 的合成 workload 下，skip-join MLFQ 的峰值 KV 占用可达 FCFS 的 7×。两个 strawman 都不行：defer 新请求 → MLFQ 退化成 FCFS；kill + recompute 低优先级 → 浪费算力，且与饥饿提升组合可能死锁（被 kill 的请求提升后反过来 kill 当前请求，循环互杀）。FastServe 的答案是 **proactive swapping（主动换页）**：低优先级队列的 KV 主动换出到 host，即将调度的作业提前换入，换入传输与 running 作业的计算重叠——swap 开销移出关键路径。量级（论文 §4.2 口径，OPT-175B 16 × A100）：单作业 KV ≈ 2.3 GB，decode 单步 ~60 ms，PCIe 4.0 × 16 全带宽 swap ~36 ms——reactive 串行会把步时拉长六成，proactive 重叠后藏进计算。

KV 字节按本维度符号翻译（论文记号 $4lh(s+t)$，fp16 MHA： $h$ = hidden 宽 = $H_{kv} d_{\mathrm{head}}$，4 = K、V 两份 × 2 字节； $s, t$ = 输入 / 输出长，译作 $T_p, T_o$）： $`\mathrm{Mem}_{\mathrm{KV}} = \kappa L (T_p + T_o)`$， $\kappa L = 4lh$。代入 OPT-175B（ $l$ = 96、 $h$ = 12288）： $\kappa L$ = 4 × 96 × 12288 ≈ $4.72 \times 10^{6}$ B/token = 4.5 MiB； $T_p$ = 512、 $T_o$ = 1 时单作业 513 × 4.72 × 10^6 ≈ $2.42 \times 10^{9}$ B ≈ 2.25 GiB（论文报 2.3 GB，舍入口径）。实现规模：2.9K 行 Python（frontend + scheduler）+ 8.1K 行 C++/CUDA（执行引擎 SwiftTransformer，Ray actor GPU workers），custom CUDA kernel 支持 Orca 迭代级调度与 vLLM PagedAttention（论文 §Implementation 逐字口径）。端到端：对照 vLLM，同平均 / 尾延迟约束下吞吐至多 31.4× / 17.9×（摘要口径）。

```
Q1 (q=8)   [短作业■] ──完成即返回
Q2 (q=16)  [中作业■■]
Q3 (q=32)  [长作业■■■■] ──quantum 耗尽──→ 降级
 ...
Q7         [超长作业] ←─ 周期性提升(防饥饿, 阈值按 SLO, 默认 300ms)
新到达: profile t_init ──skip──→ 跳过 q < t_init 的队列, 直接入 Q_i
pending KV ──proactive swap-out──→ host; 即将调度者提前 swap-in(与计算重叠)
```

引用口径注：arXiv 版标题「Fast Distributed Inference Serving for Large Language Models」；NSDI'26 发表版标题「FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference」，作者逐字一致，同一工作（2026-11-23 USENIX 页核实）。官方 artifact = LLMServe/FastServe（AE README 自认 NSDI'26 paper artifact），等级 [A]。

## 4. Llumnix：跨实例在线迁移

单实例调度再精，也修不了实例之间的负载失衡与 KV 碎片——请求该去哪个实例，是比「实例内谁先跑」更高一层的问题。Llumnix 把请求及其 KV 状态在实例间**在线迁移**（live migration），类比操作系统跨 CPU 核的上下文切换（论文摘要自喻），用运行时重调度统一处理负载均衡、隔离、碎片整理与优先级 / SLO 区分（[Sun et al., 2024](../references.md)，[A]，摘要）。

机制锚 v0 实现（llumnix-project/llumnix-ray，论文实现；版本注：2026-03 v1 重构另立 llumnix-project/llumnix（cloud-native），v0 留作本地部署 / 调度实验，repo README 口径；原 AlibabaPAI/llumnix 已 301 更名）。全局调度器分 dispatch（新请求派发）与 migration（在途请求迁移）两路；迁移策略 Balanced（负载差收敛才迁，`migration_policy.py` L54-74）与 Defrag（高低负载配对整理碎片，L76-87），负载度量可选 remaining_steps 或 kv_blocks_ratio（`docs/Arguments.md`）。迁移本身是**多阶段**的：默认至多 3 阶段（`MIGRATION_MAX_STAGES` = 3），剩余块数少于 16 时做最后阶段（`MIGRATION_LAST_STAGE_MAX_BLOCKS` = 16，`config/default.py` L225-227 注释逐字）——前几阶段边 decode 边拷 KV，最后阶段短暂停拷完剩余小块；传输后端 gloo / nccl / rayrpc / grpc / kvtransfer（ipc / rdma，Arguments.md）。

迁移成本账（路线：总成本 = 批量传输时间 + 最后阶段停拷时间，关键一步是批量部分被持续 decode 掩盖，暴露的只有尾段）：

$$
t_{\mathrm{mig}} \approx \frac{\kappa L T}{\beta_{\mathrm{net}}} + t_{\mathrm{last}}, \qquad t_{\mathrm{last}} \le \frac{16 \thinspace b \thinspace \kappa L}{\beta_{\mathrm{net}}} \qquad (4)
$$

**符号**： $\beta_{\mathrm{net}}$ = 实例间有效带宽； $b$ = block size； $t_{\mathrm{last}}$ = 最后阶段停拷（downtime）；16 = 最后阶段块数上限（v0 默认）。口径：单向传输（目的端收齐即续跑，源端副本随后释放），与 §2 往返 swap 差因子 2；批量阶段与 decode 重叠，暴露时间以尾段为主（README「near-zero overhead」主张的机制基础）。代入章脊： $T$ = 4096、KV = 512 MiB、400 Gbps = 50 GB/s，批量段 10.7 ms；最后阶段 ≤ 16 块 × $b$ = 64 × 128 KiB = 128 MiB，停拷 ≈ 2.7 ms，合计 $t_{\mathrm{mig}}$ ≈ 13.4 ms。对照同 $T$ 的 recompute 133 ms：迁移便宜约 10×（口径：全量传输 + 停拷 13.4 ms；若批量段完全被 decode 掩盖、只暴露尾段 2.7 ms，比值约 49×），且请求不丢已生成 token——这就是「迁移优于抢占重算」的定量依据（跨实例场景）。实测：尾延迟改善一个数量级、高优先级请求至多加速 1.5×、同尾延迟下至多省 36% 成本（摘要口径）；README 补充口径：对照按队列长度负载均衡，TTFT 均值 / P99 至多 4.6× / 9.1×、P99 TBT 改善 15%（16 × Qwen2.5-7B A10 + 16 × Llama2-13B A800 评测）。弹性伸缩与迁移的编排账归 8.5，KV 传输硬件与带宽账归 8.3。

## 5. SGLang schedule_policy：策略全集

waiting 队列按什么顺序准入？SGLang 把答案集中在 `managers/schedule_policy.py` 的 `SchedulePolicy.calc_priority`（L279）。版本锚定（2026-11-23 main 核实）：早期文档常列的五项（LPM / DFS_WEIGHT / FCFS / LOF / RANDOM）仍在，但已是子集——当前共九种，分两族（L188-204）：

| 族 | 策略 | 排序键（源码语义） | 目标 |
| --- | --- | --- | --- |
| CacheAware | LPM | 前缀命中长度降序（L803-813） | 命中率最大化 |
| CacheAware | DFS_WEIGHT | radix 树 DFS 权重序（L858-863） | 命中率最优序（论文口径） |
| CacheAware | HRRN | 响应比降序，token-based aging（L822-855） | 命中率 + 防饥饿 |
| CacheAware | SHORTEST_PREFILL_FIRST | 未缓存 prefill 工作量升序 + aging（L685+） | 平均 TTFT |
| CacheAware | UNIBOOST | 虚拟到达时间升序（L950-976） | 尾延迟感知准入 |
| CacheAgnostic | FCFS | 到达序（可叠 priority，L888-897） | 公平基线 |
| CacheAgnostic | LOF | max_new_tokens 降序（L866-880） | 长输出先跑 |
| CacheAgnostic | RANDOM | 随机 shuffle（L883-885） | 实验基线 |
| CacheAgnostic | ROUTING_KEY | running 批中 routing key 频次降序（L979-1005） | 同 key 亲和 |

CacheAware 族先对 waiting 请求做前缀匹配（`_compute_prefix_matches`），再按命中信息排序；CacheAgnostic 族不依赖匹配结果。两个公共机制值得单独记：其一，**temporary deprioritized**——与某 leader 请求共享未缓存前缀的请求被暂时压到队尾，等 leader 先把前缀算进缓存再跟进，避免重复 prefill（LPM / SPF / UNIBOOST 共用）；其二，**bounded LPM**——waiting 队列超过阈值时只在 `BOUNDED_LPM_WINDOW_SIZE` = 64 的窗口内做匹配（L181），控制每步匹配开销，窗口外按 FCFS 刷新。

**LPM**（longest prefix match）：命中越长越先跑，prefill 节省直接最大化（命中率 → 节省账见 2.2）。代价是 §1 引过的饥饿论断——短前缀请求可能长期排不上。**DFS_WEIGHT**：按 radix 树的 DFS 权重序准入；论文口径下，cache 容量 ≥ 最大请求长度时 DFS 序达到最优命中率（每条树边的 KV 只算一次，下界 $C \ge \sum_e |e|$ 取等），且 longest-shared-prefix-first 序与 DFS 序等价（[Zheng et al., 2023](../references.md)，[A]，§3 / 附录 A.3；机制 recap 2.2，不重复证明）。

**HRRN**（highest response ratio next）把经典响应比调度翻译成 token 时钟。设 $U$ = 请求未缓存的 prefill token 数（uncached 工作量）， $\mathrm{waited}$ = 入队以来引擎已处理的 prefill token 数（`processed_tokens − arrival_processed_tokens`，调用方维护单调计数器）：

$$
\mathrm{ratio} = 1 + \frac{\mathrm{waited}}{U} \qquad (5)
$$

**符号**： $U$ = uncached 长度（token）； $\mathrm{waited}$ = 等待期间流过的 prefill token 数；源码按 $-\mathrm{waited}/U$ 排序（常数 1 不影响序，L843-855）。代入： $U$ = 1024、 $\mathrm{waited}$ = 10000 token 时 ratio = 1 + 9.77 = 10.77；新到达请求 $\mathrm{waited}$ = 0，ratio = 1。docstring 给出等价性：吞吐恒定时 $\mathrm{waited}/U$ = 等待秒数 ÷ 估计 prefill 时间，即经典 HRRN 的「1 + 等待时间 ÷ 服务时间」。token 时钟的好处是不依赖墙钟——多 rank 下处理 token 数同步推进，排序天然一致。

**命题 3**（HRRN 序性质）：(i) $U$ 相同的两条请求，早到者响应比高、先被调度（FCFS 特例）；(ii) $U \gt 0$ 的请求响应比随 $\mathrm{waited}$ 严格单调增；(iii) 新到达请求 $\mathrm{waited}$ = 0，响应比增量为全队列最小（ $U$ = 0 的请求无 prefill 工作、立即放行，不参与排序，L848-850）。**证明**：(i) 同 $U$ 时比值大小由 $\mathrm{waited}$ 定，早到者 $\mathrm{waited}$ 大；(ii) $U$ 固定正数， $\mathrm{waited}$ 每步不减且引擎处理时严格增；(iii) 新到达 $\mathrm{waited}$ = 0，比值 = 1 为下界。证毕。防饥饿推论：由 (ii)(iii)，新到达者从响应比 1 起步，不能**立即**插到响应比已大于 1 的等待请求前面。 $U$ 更小的新请求 aging 更快（响应比增速 $1/U$ 更大），等待足够久后可以反超，但每条反超者被服务后即离队；稳定负载（到达率有限）下各请求的等待有界，不发生永久饥饿。这正是对 LPM 饥饿论断的机制性回答。数值（`processed_tokens` = 100000）：A（ $U$ = 1024，入队快照 90000）比值增量 9.77；B（ $U$ = 4096，快照 90000）2.44；C（ $U$ = 1024，快照 99000）0.98；新到 D 为 0——序 A > B > C > D，A / C 同 $U$ 保持 FCFS 相对序（性质 (i)）。

**SHORTEST_PREFILL_FIRST**：按 uncached prefill 工作量升序（短作业优先的 prefill 版），配 aging factor 与负载 scaling 的类边界防饥饿；短作业先完成降平均等待，是 SJF（shortest job first，最短作业优先）直觉在 TTFT 上的应用（排队论入口见 1.3）。**UNIBOOST** 走另一条路：不改排序键的语义，而是给短工作量请求一个「虚拟提前到达」的 boost。设 $g$ = boost 衰减率（源码 `uniboost_gamma`，默认 10.0 /s，L220）、 $u$ = 每 token 时间系数（源码 `SGLANG_UNIBOOST_ALPHA_US_PER_TOKEN`，默认 64.0 μs/token，`environ.py` L727）——**符号翻译注**：源码参数名 gamma / alpha 与本维度 $\gamma$（draft 长度）/ $\alpha$（接受率）冲突，按 spec §4.3 硬规则译作 $g, u$：

$$
\Phi = \mathrm{arrival} - b(u \cdot U), \qquad b(w) = \frac{\ln\left(1 / (1 - e^{-g w})\right)}{g} \qquad (6)
$$

**符号**： $\Phi$ = 虚拟到达时间（越小越先准入）； $\mathrm{arrival}$ = 实际到达时刻（秒）； $U$ = uncached 工作量（token）； $b(w)$ = boost 量（秒），随 $w$ 增大衰减到 0。长作业 $b \to 0$，保持 FCFS 原位、永不饿死；短作业获得提前量，单 token 工作量时约 0.74 s（默认参数）。代入默认参数： $U$ = 100 时 $b$ ≈ 0.278 s， $U$ = 1000 时 0.075 s， $U$ = 10000 时 $1.7 \times 10^{-4}$ s——万 token 长请求几乎不被 boost，百 token 短请求虚拟提前约 0.28 s。

> **坑**： $b(0)$ 发散（ $\ln(1/0)$）——恢复的 chunk 可能已无剩余工作量。源码以 $\max(U, 1)$ 兜底（L942）；自行实现同类 boost 时若漏掉这个钳位，零工作量请求会算出 inf 并污染排序。

CacheAgnostic 族四种：**FCFS** 到达序，开 `enable_priority_scheduling` 后按 (priority × sign, 入队时间) 排序（L888-897，sign 由 `schedule_low_priority_values_first` 定）；**LOF**（longest output first）按 `max_new_tokens` 降序——长输出请求先开跑，其长 decode 与其他请求重叠，makespan（总完成时间）直觉（注意用的是上界 max_new_tokens 而非实际输出长）；**RANDOM** 随机 shuffle，作公平性 / 消融基线；**ROUTING_KEY** 按 running 批中 routing key 的出现频次排序——key 已在批内的请求优先、频次高者优先（L983-1005），实现同 key 请求的批内亲和；routing_key 是 API 传入的请求级字段（`io_struct.py` L361），与 LoRA 的 lora_id 字段相互独立。

## 6. encoder cache 调度与策略决策表

多模态请求的 prefill 还有一道前置门：vision encoder 的输出 embedding 必须先就位。vLLM 把 encoder 输出独立于 KV 缓存管理（`v1/core/encoder_cache_manager.py`，2026-11-23 核实）：容量以 embedding 数计（`num_free_slots`，L79），`allocate` / `free` 逐请求登记（L211 / L300），容量不足时驱逐旧条目腾位（L203-208）；调度器侧 `_try_schedule_encoder_inputs` 在 `encoder_compute_budget`（`max_num_encoder_input_tokens`）内安排 encoder 计算，预算耗尽的请求本步跳过（调度器的 `num_new_tokens` = 0 跳过分支，循环走读见 4.1 §5）。抢占连带释放 encoder cache（`scheduler.py` L1850）——恢复时 encoder 输出也要重算或重取。教学数值（口径：教学构造，非引擎默认）：cache 容量 4096 embedding、单图 576 embedding，则至多缓存 7 张图的输出（4096 / 576 = 7.1），第 8 张触发驱逐。多模态 serving 的完整账（VE 分离、encoder 算力配比）归 9.4，本章只登记调度接口。

策略决策表（workload × SLO × 策略；机制锚点均见本章对应节）：

| workload 特征 | SLO 目标 | 推荐策略 / 机制 | 风险与对冲 |
| --- | --- | --- | --- |
| 高前缀复用（多轮 / agent / RAG） | 吞吐 + TTFT | LPM / DFS_WEIGHT（§5） | 饥饿 → HRRN 或 periodic 提升 |
| 长短 prompt 混跑 | 平均 TTFT | SHORTEST_PREFILL_FIRST（§5） | 长请求等待 → aging factor |
| 尾延迟敏感（在线交互） | P99 TTFT / TBT | UNIBOOST（§5）；FastServe MLFQ（§3） | 抢占 KV 压力 → proactive swap |
| 多租户分级 | 高优先级 SLO | vLLM PRIORITY / SGLang priority（§1） | 低优先级饥饿 → aging / 提升 |
| 输出长尾、离线批处理 | 纯吞吐 | FCFS + 大 batch，少抢占（§1） | 队头阻塞（离线可接受） |
| 实例间负载失衡 / 碎片 | 尾延迟 + 成本 | Llumnix 迁移（§4） | 迁移带宽 → 多阶段 + 尾段停拷 |
| 同 routing-key 亲和（用户指定 key） | 批内切换开销 | ROUTING_KEY（§5） | 稀有 key 请求靠后 → 必要时换 FCFS |
| 多模态 | TTFT | encoder cache 预算门（§6） | 容量驱逐 → 9.4 配比账 |

收束一句：调度策略 = 准入顺序 + victim 选择 + 恢复路径三个决策的组合；三队列、块池、迁移通道是共享机制。换策略不动机制，换机制（如 V0 swap → V1 recompute）则重算 §2 的成本账——这正是本章把账推导放在策略清单之前的原因。

## 总结

| 论断 | 回指 |
| --- | --- |
| 三队列（waiting / running / swapped）是设计空间；vLLM V1 现役无 swapped，抢占 = recompute；SGLang retract 可选 host 备份 | §1 |
| victim 选择：vLLM FCFS 抢最晚到、PRIORITY 抢最低优先级；SGLang retract 最短输出先 | §1 |
| recompute = 一次 prefill（非 $T$ 次 decode）；swap = 往返 KV ÷ 有效带宽 | 式 (1)(2)、§2 |
| swap 更快当且仅当 $\beta_{\mathrm{pcie}} \gt \beta^{\ast} = \kappa L F_{\mathrm{eff}} / (N + LTd_{\mathrm{model}})$； $\beta^{\ast}$ 随 $T$ 递减（8.64 → 5.50 GB/s） | 命题 1、式 (3)、练习 2 |
| 理想账用理论带宽、实测用有效带宽：小 $b$ swap 效率塌陷，论文 §7.3 recompute ≤ 20% swap 延迟，V1 默认 recompute | §2 坑注 |
| watermark 是准入预留（抢占频率调节器），不是抢占触发器；触发 = 分配失败 / decode 内存检查失败 | §2、2.1 §4 |
| skip-join 按 $t_{\mathrm{init}}$ 选入队层级，保证 prefill 不被腰斩；抢占粒度 = 单 token；饥饿靠周期提升（默认阈值 300 ms） | 命题 2、§3 |
| MLFQ 峰值 KV 可达 FCFS 7×（论文 Fig.8 口径）→ proactive swap 重叠传输与计算 | §3 |
| 迁移成本 = 单向 KV ÷ 网络带宽 + 尾段停拷（≤16 块）；8B 口径 10.7 + 2.7 = 13.4 ms，比 recompute 便宜 ~10×（全量口径；只暴露尾段则 ~49×） | 式 (4)、§4 |
| SGLang 九策略 = CacheAware 5 + CacheAgnostic 4（旧五项为子集，2026-11-23 锚定）；HRRN 响应比 token 时钟防饥饿 | §5、命题 3、式 (5) |
| UNIBOOST 虚拟到达 boost 随工作量衰减到 0，长请求保持 FCFS 位；源码 gamma / alpha 译作 $g, u$ | 式 (6)、§5 |
| encoder cache 独立容量门，抢占连带释放；多模态全账归 9.4 | §6 |

## 练习

1.（计算 · [10]）章脊口径（ $\kappa L$ = 128 KiB/token、 $F_{\mathrm{eff}}$ = 494.75 TFLOPS、PCIe Gen5 单向 64 GB/s）， $T$ = 4096 的请求被抢占：(a) KV 字节数；(b) 按式 (2) 算往返 swap 时间；(c) 按式 (1) 算 recompute 时间（attention 项不可略，先估占比再算）；(d) 比值与「哪条路快」的结论。校验：(b)/(c) 应分别 ≈ 16.8 ms / 133 ms，与 §2 表一致；结论方向与命题 1 在 $\beta^{\ast}$ ≈ 8.07 GB/s 处的判定一致。
2.（推导 · [20]）从头证明命题 1：(a) 由式 (1)(2) 解出临界带宽 $\beta^{\ast}$ 并化简到式 (3) 右形；(b) 证明 $\beta^{\ast}$ 随 $T$ 严格单调递减；(c) 求 $T \to 0$ 极限并代入章脊数值；(d) 说明为什么「长上下文更偏向 swap」是结构结论而非拟合。校验：(c) 极限 = $\kappa L F_{\mathrm{eff}} / N$ ≈ 8.6 GB/s； $T$ = 32768 时 $\beta^{\ast}$ ≈ 5.50 GB/s，与 §2 表一致。
3.（计算 · [30]）四条 waiting 请求：A（ $U$ = 1024，入队快照 90000，墙钟到达 0 s）、B（ $U$ = 4096，快照 90000，到达 0 s）、C（ $U$ = 1024，快照 99000，到达 9 s）、D（ $U$ = 2048，快照 100000，到达 10 s），当前 `processed_tokens` = 100000。(a) 按式 (5) 算 HRRN 响应比增量并排序；(b) 按式 (6)（ $g$ = 10 /s、 $u$ = 64 μs/token）算 UNIBOOST 的 $\Phi$ 并排序；(c) 指出哪对请求的相对序验证了命题 3(i)，并解释 D 在两种策略下都排最后的原因。校验：(a) 序 A > B > C > D（增量 9.77 / 2.44 / 0.98 / 0）；(b) $\Phi_A$ ≈ −0.073 s、 $\Phi_B$ ≈ −0.008 s，序不变；A / C 同 $U$ 保持 FCFS 相对序。
4.（代码 · [30]）读 `vllm/v1/core/sched/scheduler.py` 的抢占段（L898-969）与 `_preempt_request`（L1831-1877，main 2026-11-23）：先写出预测——(a) FCFS 策略下 victim 是哪条请求？(b) PRIORITY 策略下 victim 是哪条（给出排序键）？(c) 抢占后该请求的 `num_computed_tokens`、状态、队列位置各变成什么？(d) 为什么 V1 不需要 swapped 队列？再对照源码验证，并说明恢复时 prefix cache 命中如何改变式 (1) 的实际成本。校验：源码 `preempted_req = self.running[-1]` / `max(self.running, key=lambda r: (r.priority, r.arrival_time))` / `request.num_computed_tokens = 0` / `self.waiting.prepend_request(request)`；(d) 与 §2「recompute 是上界」段一致。

## 参考

- [Wu et al., 2023](../references.md)：Fast Distributed Inference Serving for Large Language Models，arXiv:2305.05920，[A]（key: fastserve；NSDI'26 发表版标题「FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference」，usenix.org/conference/nsdi26/presentation/wu-bingyang；官方 artifact github.com/LLMServe/FastServe，2026-11-23 核实，等级 [A]）——skip-join MLFQ §4.1 / Algorithm 1、proactive swapping §4.2（2.3 GB / 60 ms / 36 ms / 7× 口径）、实现规模 §Implementation、31.4× / 17.9× 摘要
- [Sun et al., 2024](../references.md)：Llumnix: Dynamic Scheduling for Large Language Model Serving，arXiv:2406.03243，[A]（key: llumnix；github.com/llumnix-project/llumnix-ray = v0 论文实现，v1 另立 llumnix-project/llumnix）——live migration 与重调度摘要、尾延迟 / 1.5× / 36% 摘要口径；v0 机制片段 `global_scheduler/migration_policy.py` L54-87、`config/default.py` L225-227、`docs/Arguments.md`（2026-11-23 核实）
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——三队列与抢占 §4.5、recompute 一次 prefill 口径 §4.5、recompute vs swap 实测 §7.3
- [Zheng et al., 2023](../references.md)：SGLang: Efficient Execution of Structured Language Model Programs，arXiv:2312.07104，[A]（key: sglang）——cache-aware 调度饥饿论断 §3、DFS 序最优命中率与等价性 §3 / 附录 A.3
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——机制片段 `v1/core/sched/scheduler.py` L898-969 / L917-923 / L1041 / L1831-1877、`v1/core/sched/request_queue.py` L13-17 / L131-144、`v1/core/kv_cache_manager.py` L200 / L529-536、`config/scheduler.py` L232、`v1/core/encoder_cache_manager.py` L79 / L203-208 / L211 / L300（均 2026-11-23 main 核实）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——机制片段 `managers/schedule_policy.py` L181 / L188-204 / L279 / L803-1005、`managers/schedule_batch.py` L3007 / L3030-3124 / L3137-3164、`managers/scheduler.py` L5275-5297、`environ.py` L727（均 2026-11-23 main 核实；策略全集版本漂移注见 §5）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——bf16 dense 989.5 TFLOPS、PCIe Gen5（带宽口径 recap 2.4）
- 跨维度指引（纯文字）：inference 1.2（prefill FLOPs 账）、1.3（排队论 / SJF 直觉）、2.1（块池 / watermark / 抢占语义）、2.2（prefix cache 命中与 radix 树）、2.4（PCIe 口径、取回 vs 重算单向账）、4.1（迭代级调度与步时）、4.3（chunked prefill）、5.2 / 6.2（模块全量走读）、8.3（KV 传输带宽账）、8.5（弹性伸缩与迁移编排）、9.4（多模态 serving）