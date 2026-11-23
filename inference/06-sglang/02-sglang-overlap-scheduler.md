# 6.2 Overlap Scheduler 与零开销调度（Overlap Scheduler and Zero-Overhead Scheduling）

> GPU 前向只要几毫秒，CPU 调度凭什么能吃掉一半的步时，又凭什么能被完全藏进 GPU 时间里？本章走读 SGLang 的 event_loop_overlap 与 FutureMap stash/publish/resolve 机制，完整推导 bubble 消除账与 future-token resolve 正确性命题，并把 cache-aware 调度、chunked prefill 准入、DP attention controller 对回 4.x 的公式；batch_overlap/ 目录的 TBO/SBO 是另一机制，专节区分 | 前置：4.1（迭代级调度循环与步时账）、4.2（调度策略与抢占）、4.3（chunked prefill 与 stall 账）、2.2（基数树与命中账）、3.4（CUDA Graph 与 NanoFlow）、6.1（srt 架构与两级池） | 锚点：SGLang [Zheng et al., 2023] [A]（arXiv:2312.07104 + sgl-project/sglang）；sglang-repo [A]（`srt/managers/{scheduler,overlap_utils,schedule_policy,schedule_batch,prefill_delayer,shortest_prefill_load,data_parallel_controller}.py` + `srt/batch_overlap/`，2026-11-23 main fresh tarball file:line 核实）；SGLang v0.4 blog [C]（开放技术博客、非闭源）；NanoFlow [Zhu et al., 2024] [A]（arXiv:2408.12757）；DeepInfer doc（教学参照 + 官方实现，官方披露口径）

## 1. CPU 调度开销：GPU idle bubble 从何而来

一个 decode 步的 GPU 时间可以短到 5 ms 量级（4.1 章脊口径：8B @ H100、 $B$ = 8、 $\bar{T}$ = 1024，带宽下界步时 ≈ 5.13 ms）。CPU 侧每步要做的事却一件不少：收请求（ZMQ recv + msgspec 反序列化）、前缀匹配（基数树行走，2.2 命题 1 的 $O(T)$）、准入与 KV 分配（PrefillAdder）、metadata tensor 构建、采样参数准备，外加上一批的后处理（追加 token、判完成、释放 KV、发输出）。SGLang v0.4 blog 的判断逐字：「An unoptimized inference engine can spend as much as half of its time on CPU overhead」（[SGLang v0.4 blog](../references.md)，[C] 开放技术博客、非闭源，2024-12-04，2026-11-23 全文核实）——batch scheduling / memory allocation / prefix matching 三项是开销主体。

把一步的 CPU 工作拆细（锚点均 scheduler.py，2026-11-23 核实）：ingest_requests（L2369，ZMQ recv + process_input_requests 分发进 waiting）；get_next_batch_to_run（L4386，merge 上一批 + filter 已完成 + get_new_batch_prefill 准入或 update_running_batch）；run_batch 的 CPU 段（采样参数 gather、StepPenalties 构建、metadata tensor 准备，L5488-5501）；process_batch_result（L5957，按 forward_mode 分流 decode / prefill 处理器：追加 output_ids、判 EOS / max_tokens、释放 KV、经 ZMQ 发 BatchTokenIDOutput）。blog 点名要隐藏的 radix cache 操作——match_prefix / insert / evict（6.1 §4）——全在这条 CPU 路径上执行，队列长时 $O(T)$ 匹配乘以请求数（§6 的 bounded LPM 正是给这份开销封顶）。

normal 循环（`event_loop_normal`，scheduler.py L2231，4.1 §6 已走读骨架）里这些工作与 GPU 执行是串行的：schedule 完才 launch forward，forward + sample 完才 post-process，post-process 完才进下一步 schedule。把每步时间拆成四段：

$$
T_{\mathrm{normal}} = c_s + t_f + t_s + c_p \qquad (1)
$$

**符号**： $c_s$ = schedule 的 CPU 时间（ingest + 匹配 + 准入 + 分配 + metadata）； $c_p$ = post-process 的 CPU 时间（结果处理 + 输出发送）； $t_f$ = GPU forward 时间； $t_s$ = GPU sample 时间。口径： $c_s, c_p$ 是本章局部成本记号（沿 4.4 $c_d / c_v$ 先例，与 CP 度 $c$ 区分）；迭代序号本章用 $j$（blog 引文的 forward(N) 保留原记号， $N$ 在全书专归参数量，spec §4.3）。GPU 每步空闲 $c_s + c_p$——schedule 与 post-process 期间 GPU 无 kernel 可跑，这就是 bubble。

**本章数值口径**（沿用 4.1 章脊，全章例题复用）：8B @ H100 SXM， $B$ = 8、 $\bar{T}$ = 1024， $t_{\mathrm{gpu}} := t_f + t_s$ ≈ 5.13 ms（4.1 式 (3) 带宽下界；sample 是 $[B, \mathrm{vocab}]$ 上的小 kernel， $t_s \ll t_f$，教学近似 $t_{\mathrm{gpu}} \approx t_f$，下同）。CPU 开销取教学假设 $c_{\mathrm{cpu}} := c_s + c_p$ = 1 ms（声明：非实测，量级参照 blog「至多一半」口径的下沿），则 $T_{\mathrm{normal}}$ = 6.13 ms，GPU 利用率 5.13 / 6.13 ≈ 83.7%；blog 的极端口径 $c_{\mathrm{cpu}} = t_{\mathrm{gpu}}$ 时利用率跌到 50%。小模型 + 大 TP 下 $t_{\mathrm{gpu}}$ 更短、CPU 占比更高——blog 逐字：「The speedup is most significant on small models and large tensor parallelism sizes.」

## 2. event_loop_overlap：循环走读

把 CPU 工作藏进 GPU 时间，循环结构要怎么改？答案是**结果处理延迟一拍**：本步 launch forward 之后不立即处理本步结果，而是处理上一步的结果，本步结果入队等下一拍。`event_loop_overlap`（scheduler.py L2266，2026-11-23 核实）教学摘录（省略 pause / idle / invariant 分支与 HiCache 细节，主干语义不变）：

```python
# sglang/srt/managers/scheduler.py event_loop_overlap L2266（教学摘录）
self.result_queue = deque()          # (batch.copy(), batch_result) 对
def pop_and_process():
    tmp_batch, tmp_result = self.result_queue.popleft()
    self.process_batch_result(tmp_batch, tmp_result)   # 处理第 j−1 批

while True:
    self.ingest_requests()                             # 收新请求
    early = self.should_process_last_batch_early(self.last_batch)  # L2338
    if early:
        pop_and_process()                              # 提前处理 j−1
    plan = self.get_next_batch_to_run(...)             # L4386 调度第 j 批
    batch = plan.batch_to_run
    if batch:
        batch_result = self.run_batch(batch)           # L5455 launch forward(j)，非阻塞
        self._apply_war_barrier()                      # L2207 WAR 栅栏（§5）
        self.result_queue.append((batch.copy(), batch_result))
    if self.last_batch and not early:
        pop_and_process()                              # GPU 跑 j 时 CPU 处理 j−1
    if self.is_generation:
        self.launch_batch_sample_if_needed(batch_result, batch)  # L5928
    self.last_batch = batch
```

三个设计点逐个说清。

**其一，result_queue 与 batch.copy()**。入队的是批对象的**快照**：copy()（schedule_batch.py L3545）只保留 process_batch_result 用到的字段，reqs 列表切片作防御性快照——源码注释口径：filter_batch / merge_batch 历史上原地改列表，如今 rebind 新列表，切片保留以保证快照永不别名原批。不 copy 则后处理读到的批状态可能已被下一步的 merge / filter 改写（6.1 §3）。队列深度稳态为 1：第 $j$ 拍入队第 $j$ 批、弹出第 $j-1$ 批；提前处理路径（其三）在调度前弹出，本拍末的 not early 条件（L2322）保证不重复弹出——两条路径互斥，深度不变。后处理需要的 D2H 结果拷贝早在上一拍就已发出（_copy_result_to_cpu，L5900：copy_stream 等 forward_stream 后异步拷贝，copy_done event 收尾）；pop_and_process 先 copy_done.synchronize()（batch_result_processor.py L1157）再读 host 结果，等待至多是批 $j-1$ 的 GPU 残余。GPU-bound 稳态（ $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$）下该等待 ≈ $t_{\mathrm{gpu}} - c_{\mathrm{cpu}}$（章脊算例 ≈ 4.13 ms）——CPU 被上一拍的 sync 钉住，launch 间隔被拉长到 $t_{\mathrm{gpu}}$，**这正是把 CPU pacing 到 GPU 节奏的机制**；只有 CPU-bound（ $c_{\mathrm{cpu}} > t_{\mathrm{gpu}}$）时等待才趋近零（GPU 早已跑完）。零开销指 **GPU 零 idle，不是 CPU 零等待**：等待方向是 CPU 等 GPU，只要条件成立，GPU 从不等 CPU。

**其二，sample 的延迟 launch**。run_batch 内 forward 与 sample 默认一起 launch，但 grammar（结构化输出）请求例外：sample 的 mask 依赖上一批已确认 token 推进后的 FSM 状态。tp_worker.py L747-767 的条件（overlap + 非投机 + 有 grammar，或开 SGLANG_ENABLE_DELAY_SAMPLE 且非 prefill-only 批）把 sample 包成 delay_sample_func 延迟 closure；launch_batch_sample_if_needed（L5928）在 pop_and_process **之后**才 launch 它——源码注释逐字（L2329）：「It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.」CPU 顺序保证 mask 正确，GPU 顺序保证 sample 排在 forward 之后（同流）。

**其三，提前处理闸门**。should_process_last_batch_early（L2338）在两种情形把第 $j-1$ 批的处理提到调度第 $j$ 批**之前**：(a) SGLANG_DISABLE_OVERLAP_AFTER_PREFILL 开启且上一批是 prefill——prefill 结果早处理早流式，保 TTFT（DP 下 is_extend_in_batch 跨 rank 同步，各 rank 同决策，无死锁）；(b) last_batch.grammar_needs_sync()——下一批的 grammar bitmask 需要本批的 FSM advance。投机 v2 另有 _advance_pending_grammar（L2359）：在 worker 内 generate_token_bitmask 之前把队列里未处理 decode 结果的 FSM 推进掉，CPU advance 与 target verify forward 重叠（grammar barrier）。

时序对照（ASCII，默认非 grammar 路径，sample 随 forward 一起 launch）：

```
拍:        j−1                  j                    j+1
CPU:  sched(j−1)→launch fwd(j−1)→post(j−2)→sched(j)→launch fwd(j)→post(j−1)→sched(j+1)→launch fwd(j+1)→post(j)
GPU:            [forward(j−1) + sample(j−1)]   [forward(j) + sample(j)]   [forward(j+1) + sample(j+1)]
                └──── 零 bubble 条件：launch fwd(j+1) 早于 forward(j)+sample(j) 结束 ────┘
```

normal 循环的 GPU 在每两段 forward 之间空 $c_s + c_p$；overlap 循环里这段空隙被 post(j−1) + sched(j+1) 填满——只要填得下（§4 的账）。4.1 §6 已登记一句「重叠改的是循环时序，不改每步重选 batch 的迭代级语义」，本章兑现机制细节。

## 3. FutureMap：stash / publish / resolve 的当前语义

CPU 调度第 $j+1$ 批时，第 $j$ 批的 sample 还没跑完——下一批的输入 token 从哪里来？blog 时代的答案是 future tokens placeholder（「resolving dependencies by creating future tokens and carefully scheduling CUDA events and synchronization」，v0.4 blog 逐字）：给未产出的 token 一个占位符，调度先用占位符组批，执行时再解析。当前 main 的实现已演进为 **always-on pool-indexed relay**——FutureMap（`managers/overlap_utils.py` L309，824 行文件的核心类），docstring 逐字：「Always-on pool-indexed relay for cross-iter values. Forward writes via publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.」

> **坑**：照 blog 的 placeholder 负索引设计走读当前代码。触发条件：以 v0.4 blog 为唯一依据预测 FutureMap 接口；现象：当前实现没有占位符对象，decode 批的 input_ids 直接是 None，token 由 GPU 侧 gather 解析。规避：概念口径引 blog，代码口径锚当前 stash/publish/resolve（版本日期 2026-11-23）；命题证明一律用当前语义。

**缓冲区**：output_tokens_buf 与 new_seq_lens_buf，均为长度 req_pool_size 的 int64 GPU tensor，**按 req_pool_idx 索引**（请求槽位号，6.1 §5 的 req_to_token 行号）。slot 0 镜像 KV padding 行——CUDA Graph padded batch 的假行 req_pool_idx = 0，gather / scatter 落在 slot 0 上无害（与 ReqToTokenPool 第 0 行、KV 池 slot 0 同一设计，三处同构）。_DEBUG_ASSERT（SGLANG_IS_IN_CI）下两 buf 以 −1 poison 初始化，gather 时融合 kernel 断言非负并写回 −1（_assert_nonneg_and_invalidate，L92）——consume-once 调试不变式：每行必须先写后读，抓「gather 无中间 stash」类 bug；生产关闭（§5 坑注）。

三个操作的语义（锚点均 overlap_utils.py，2026-11-23 核实）：

| 操作 | 锚点 | 写 / 读 | 内容 |
| --- | --- | --- | --- |
| stash | L780（_stash L784） | 写 output_tokens_buf[indices] | 已确认 token（RelayPayload.bonus_tokens）+ 投机 extras（topk_p / topk_index / hidden_states / draft_probs）；ngram 走 accept_tokens / accept_lens |
| publish | L749 | 写 new_seq_lens_buf[indices] | 新序列长；投机 v2 另 record publish_ready event（门控私有 D2H 流） |
| resolve_forward_inputs | L127（顶层函数） | 读 output_tokens_buf | prefill：pinned CPU staging H2D；decode：buf[req_pool_indices] gather；MIXED：cat(prefill H2D, buf[mix_running_indices]) |
| resolve_seq_lens_cpu | L687 | 读 new_seq_lens_buf | 投机 v2 的 seq_lens（accept 数 CPU 不可知）；私有 fwd_prepare_d2h_stream 拷入 pinned 镜像，不同步 schedule_stream |

写读两端的代码本体（overlap_utils.py，教学摘录，省略 dtype 转换与调试分支）：

```python
# sglang/srt/managers/overlap_utils.py（main 2026-11-23，摘录）
# stash（L780 → _stash L800）：已确认 token 落 buf，GPU scatter
self.output_tokens_buf[indices] = payload.bonus_tokens
# resolve_forward_inputs（L127，decode 分支 L165-166）：forward 入口物化 input_ids
elif batch.input_ids is None and future_map.spec_algo.is_none():
    batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
```

两行就是 token relay 的全部写读端：写端一次 scatter（payload 是 sample 输出的 GPU tensor，无 host 往返），读端一次 gather（索引 = 批的 req_pool_indices）。调度路径全程不接触 token 值——「调度只消费数据无关量」在代码层面就是这两行之间没有任何 .cpu() / item()（后处理仍会在 CPU 读已确认 token 更新 output_ids，那是结果消费，不是调度依赖）。MIXED 分支多一次拼接：prefill 部分从 pinned CPU staging 做 H2D（prefill_input_ids_cpu），decode 部分从 buf[mix_running_indices] gather，cat 成 input_ids（L141-160）；prefill 的 token 是 CPU 已知的 prompt，走 staging 而非 relay，两条来源在 forward 入口汇合。

接线全链（scheduler.py，2026-11-23 核实）：init_overlap（L1837）创建 future_map = spec_algorithm.create_future_map（L1854），注释逐字（L1840）：「FutureMap is always-on: input_ids relay used in both modes」——**normal 循环也走 input_ids relay**，不是 overlap 专属。run_batch（L5455）内：resolve_seq_lens_cpu（L5511，stream_ordered 变体 L5520）→ forward_stream.wait_stream(schedule_stream)（L5516）→ resolve_forward_inputs（L5525）→ forward_batch_generation（L5553，非阻塞）→ 非投机路径 publish(future_indices, batch.seq_lens + 1)（L5558）→ batch.input_ids = None（L5594，注释「Next-iter input_ids relayed via future_map」）。投机 v2 的 publish 经 on_publish callback 在 worker 内 verify 与 draft_extend 之间触发（L5537），让调度准备与 draft_extend 重叠。MIXED 批的 running 部分同样 input_ids = None（schedule_batch.py L2855 注释逐字：「Decode tokens of the running portion live in future_map.output_tokens_buf」），mix_running_indices 记录其 req_pool_idx。

机制的核心洞察是**数据依赖分类**：

| 量 | 数据依赖性 | 调度时谁知道 | 解析方式 |
| --- | --- | --- | --- |
| 非投机 decode 的 seq_lens | 无关（每步恰 +1） | CPU 可提前算 | batch.seq_lens + 1，CPU metadata |
| 输入 token 值 | 有关（sample 输出） | 只有 GPU 知道 | output_tokens_buf relay + forward 入口 gather |
| 投机 accept 长度 | 有关（拒绝采样结果） | 只有 GPU 知道 | new_seq_lens_buf + publish_ready event |
| 批组成 / KV 分配 | 无关（调度决策） | CPU | get_next_batch_to_run 正常算 |

CPU 调度只消费数据无关量，数据有关量一律 GPU 流上解析——这就是「调度提前一拍」在数据层面的可行性条件。relay / defer_relay（L420 / L409）是投机 top-k-1 批的 in-place 复用优化（上一轮 draft 输出原地消费则免 scatter），stash_bonus_tokens（L634）只写 token 行（staging 转换不带 draft extras），resolve_mixed_spec_tails（L642）在 publish 栅栏后重绑 MIXED 批投机尾行的长度与 out_cache_loc——三者一句注，主干不变。

## 4. bubble 消除账：完全隐藏的条件与加速比

CPU 时间被完全隐藏的条件是什么，加速比上限又是多少？路线：把循环看成 CPU / GPU 两级流水，写出稳态每拍时间的 max 式，再与 normal 串行时间作比——关键一步是「下一批 forward 的 launch 时刻必须早于本批 GPU 执行结束」。

**命题 1**（bubble 消除条件与加速比）：稳态下 overlap 循环每拍时间为

$$
T_{\mathrm{overlap}} = \max(c_{\mathrm{cpu}},\ t_{\mathrm{gpu}}) \qquad (2)
$$

**符号**： $c_{\mathrm{cpu}} = c_s + c_p$（每拍 CPU 工作总量）； $t_{\mathrm{gpu}} = t_f + t_s$（每拍 GPU 执行总量）；口径：稳态（队列深度恒 1）、launch 链非阻塞、D2H 一拍松弛成立（§2）。CPU 时间完全隐藏当且仅当 $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$；相对 normal 循环的加速比为

$$
F = \frac{T_{\mathrm{normal}}}{T_{\mathrm{overlap}}} = \frac{c_{\mathrm{cpu}} + t_{\mathrm{gpu}}}{\max(c_{\mathrm{cpu}},\ t_{\mathrm{gpu}})} \qquad (3)
$$

且 $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$ 时 $F = 1 + c_{\mathrm{cpu}} / t_{\mathrm{gpu}} \le 2$。

**证明**（路线：对相邻 launch 间隔做流水分析——「fwd(j+1) 的 launch 时刻不晚于 fwd(j)+sample(j) 的 GPU 结束时刻」等价于 kernel 队列不空转；三段式）。基例（流水填充）：第 1 拍无上一批结果可处理，CPU 只做 sched(1) + launch，GPU 跑 forward(1)+sample(1)；第 2 拍起 post(1) 与 sched(2) 交错，进入「launch 一批、处理上一批」的稳态节奏，填充期时间与 normal 同量级，稳态论断从第 2 拍起算。保持：设 $S_j$ = GPU 开跑批 $j$ 的时刻、 $L_j$ = launch(j) 时刻（launch 非阻塞， $L_j \le S_j$）。CPU 每拍工作量 $c_{\mathrm{cpu}}$，但 pop_and_process(j−1) 必须 sync copy_done，其就绪时刻 ≈ $S_j$（GPU 跑完批 $j-1$ 加拷贝，§2）——故下一次 launch 时刻 $L_{j+1} = \max(L_j, S_j) + c_{\mathrm{cpu}}$：CPU-bound（ $L_j \ge S_j$）时按 CPU 节奏连续 launch；GPU-bound（ $L_j < S_j$）时 CPU 被 sync 钉在 $S_j$，sync 等待 $t_{\mathrm{gpu}} - c_{\mathrm{cpu}}$ 就是 pacing 步，launch 间隔 = $t_{\mathrm{gpu}}$。不变式 $L_j \le S_j$ 的保持：若 $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$，则 $L_{j+1} = S_j + c_{\mathrm{cpu}} \le S_j + t_{\mathrm{gpu}} = S_{j+1}$，kernel 队列不空，GPU 零 idle，每拍 = $t_{\mathrm{gpu}}$；若 $c_{\mathrm{cpu}} > t_{\mathrm{gpu}}$，launch 间隔 = $c_{\mathrm{cpu}}$，GPU 跑完等 launch（ $S_{j+1} = L_{j+1}$，不变式取等），每拍 = $c_{\mathrm{cpu}}$、空闲 $c_{\mathrm{cpu}} - t_{\mathrm{gpu}}$。两种情形合并即式 (2)。终止：稳态吞吐 = $1 / T_{\mathrm{overlap}}$，normal 吞吐 = $1 / T_{\mathrm{normal}}$（式 (1)），作比得式 (3)； $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$ 时代入 max = $t_{\mathrm{gpu}}$ 得 $F = 1 + c_{\mathrm{cpu}} / t_{\mathrm{gpu}}$，上界 2 在 $c_{\mathrm{cpu}} = t_{\mathrm{gpu}}$ 取到。证毕。

正例： $c_{\mathrm{cpu}}$ = 1 ms、 $t_{\mathrm{gpu}}$ = 5.13 ms，条件成立， $F$ = 6.13 / 5.13 ≈ 1.195。边界反例： $c_{\mathrm{cpu}}$ = 8 ms > $t_{\mathrm{gpu}}$，GPU 每拍仍空 8 − 5.13 = 2.87 ms， $F$ = 13.13 / 8 ≈ 1.64 < 2——overlap 不是无条件 2×，CPU 超重时瓶颈回到 CPU，式 (3) 退化为 $(c_{\mathrm{cpu}} + t_{\mathrm{gpu}}) / c_{\mathrm{cpu}}$，随 $c_{\mathrm{cpu}}$ 增大趋近 1。上界 2 是结构结论而非拟合：完全隐藏时 $T_{\mathrm{normal}} = c_{\mathrm{cpu}} + t_{\mathrm{gpu}} \le 2 t_{\mathrm{gpu}} = 2 T_{\mathrm{overlap}}$，串行时间至多是重叠时间的两倍，等号在 CPU 与 GPU 等长时取到——收益的天花板由「被藏起来的工作占比」决定，与硬件无关。

全数字代入（章脊口径，按声明精度 $t_{\mathrm{gpu}}$ = 5.13 ms 计算；精确值 5.1265 ms，4.1 式 (3)，舍入差 ≤ 0.1%；python 复核 2026-11-23）：

| $c_{\mathrm{cpu}}$ (ms) | $T_{\mathrm{normal}}$ (ms) | $T_{\mathrm{overlap}}$ (ms) | $F$ | normal GPU 利用率 | 完全隐藏 |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 5.63 | 5.13 | 1.097 | 91.1% | 是 |
| 1 | 6.13 | 5.13 | 1.195 | 83.7% | 是 |
| 3 | 8.13 | 5.13 | 1.585 | 63.1% | 是 |
| 5.13 | 10.26 | 5.13 | 2.000 | 50.0% | 是（临界） |
| 8 | 13.13 | 8.00 | 1.641 | 39.1% | 否 |

实测校准（blog 口径，[C]）：v0.4 的 zero-overhead scheduler 相对 v0.3 取得 **1.1× 吞吐**、相对其他 SOTA baseline 1.3×（headline 数，workload 依赖）——落在表内 $c_{\mathrm{cpu}}$ ≈ 0.5 ms 一行附近，说明该 benchmark 上 CPU 开销本就不大，重叠吃掉的是剩余部分。Nsight 验证逐字：5 个连续 decode batch「no single idle time on the GPU」（Triton attention backend 口径；FlashInfer backend 当时有小 gap，blog 注明待下个 release 解决）——命题 1 的「GPU 无 idle」是 profile 可证伪的论断，不是修辞。消融开关 `--disable-overlap`（blog 复现命令：Llama-3.2-3B-Instruct + --disable-overlap）退回 normal 循环。式中的量也有实测入口，不止教学假设：scheduler 内置 stage 级计时（scheduler_stage_method 装饰器标记 get_next_batch_to_run / process_batch_result 两 stage，metrics_reporter.start_scheduler_time_accounting 在循环入口启动计时，L2204）， $c_s$ 与 $c_p$ 可从 metrics 读出；GPU 侧 $t_f + t_s$ 用 Nsight kernel 时间线对照——blog 的 Nsight 图正是命题 1 的测量方法：GPU 时间线无缝即条件成立。

两个边界讨论。prefill 批 $t_f$ 大（compute-bound，1.2）， $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$ 容易成立，重叠收益稳；decode 小批（ $B$ = 1） $t_{\mathrm{gpu}}$ 短，CPU 开销暴露——这解释 blog「小模型 + 大 TP 收益最大」： $t_{\mathrm{gpu}}$ 被模型小与通信摊薄压短，normal 循环的 CPU 占比升高，重叠的相对收益 $c_{\mathrm{cpu}} / t_{\mathrm{gpu}}$ 变大。另一边界：grammar 路径的 sample 延迟 launch 把 CPU 顺序约束加回循环（§2），should_process_last_batch_early 的提前处理会局部恢复串行——零开销是**默认路径**的论断，约束解码与 TTFT 敏感 prefill 是声明过的例外。

## 5. future-token resolve 正确性：命题与证明

输入 token 在 CPU 调度时还不存在，gather 出来的值凭什么恰好等于串行执行该有的值？路线：对迭代归纳一个缓冲区不变式，关键一步是 **CUDA 同流 kernel 按提交序执行**——stash 与 gather 都在 forward_stream 上提交，流序就是正确性栅栏，不需要 host 同步。

**命题 2**（resolve 正确性，非投机 decode 路径）：设请求 $r$ 的 req_pool_idx = $i \ne 0$，第 $j$ 批为含 $r$ 的 decode 批。若每次 forward 入口的 resolve_forward_inputs 之前， $r$ 的上一拍产出已经 stash，则第 $j$ 批 gather 到的 input_ids 等于第 $j-1$ 批 sample 产出的 token（或调度器 stash 的最后确认 token），与串行执行逐 token 一致。

**证明**（路线：不变式 + 流序；三段式）。不变式 $I(j)$：gather(j) 在 forward_stream 上执行时，output_tokens_buf[i] 恰为 $r$ 的最后确认 token。基例：prefill 后的首个 decode 拍——prefill 的 forward_batch_generation 内 sample 产出首 token，_relay_forward_payload（L5835）以 RelayPayload(bonus_tokens=next_token_ids) 调 stash（L5860 经 relay），scatter 在 forward_stream 上排在 sample kernel 之后；HiSparse staging 路径则由调度器 stash CPU 已确认的 r.output_ids[-1]（L4349）——注意该 stash 提交在 **schedule_stream** 上，由 run_batch 的 forward_stream.wait_stream(schedule_stream)（L5516）定序到 gather 之前，是跨流序（保持步的同流序不覆盖此基例，由 wait_stream 补上）。两条基例都满足 $I$。保持：设 $I(j-1)$ 成立。第 $j-1$ 拍的 sample kernel 与 stash scatter 在 forward_stream 上按提交序执行（sample 先、stash 后，run_batch 的 forward_stream_ctx 内）；第 $j$ 拍的 gather 在 run_batch 内 forward_stream.wait_stream(schedule_stream)（L5516）之后提交到**同一条** forward_stream——提交序 stash(j−1) 先于 gather(j)，故执行序亦然，gather(j) 读到 stash(j−1) 写入的值，即第 $j-1$ 批 sample 的产出。期间 schedule_stream 上的 metadata 写入（第 $j$ 批的 seq_lens / out_cache_loc 准备）被 wait_stream 挡在 gather 之前完成，不污染 token 值；反向的 WAR 危害由 _apply_war_barrier 挡住（见下）。故 $I(j)$ 成立。终止：任意 $j$ 拍 gather 值正确，forward(j) 的输入序列与串行执行一致；由前缀不可变命题（models 2.4），KV 与输出随之逐 token 一致。证毕。

用单请求 $r$（req_pool_idx = 5）把连续两拍的 relay 写具体：第 $j-1$ 拍，forward(j−1) 读 input_ids = buf[5] = $t_{j-2}$，sample 产出 $t_{j-1}$，stash 写 buf[5] = $t_{j-1}$；第 $j$ 拍，CPU 调度批 $j$（seq_lens = 上拍长度 + 1，不读 token 值），forward 入口 gather buf[5] = $t_{j-1}$ 作输入——CPU 从未见过的值直接从 GPU 传到 GPU。若第 $j-1$ 拍是 prefill，stash 来源是 prefill sample 的首 token（基例）；若 $r$ 来自 HiSparse staging，stash 来源是 CPU 已确认的 output_ids[-1]（L4349）——两种来源都是「最后确认 token」，不变式的措辞对两者同时成立。

命题的四个正确性要件，缺一不可：

| 要件 | 代码落点 | 挡住的危害 |
| --- | --- | --- |
| 流序（同流提交序 + wait_stream） | run_batch L5516；stash / gather 同在 forward_stream | gather 读到上上拍的旧 token（跨拍竞态） |
| WAR barrier | _apply_war_barrier L2207：schedule_stream.wait_event(shared_read_done_event)，无 event 则 wait_stream(forward_stream) | 第 $j+1$ 拍的 schedule_stream 写入（释放 / 重分配 KV 行、req_to_token 更新）越过第 $j$ 批 forward 的共享读，forward 读到被改写的索引 |
| slot 0 padding 镜像 | FutureMap slot 0 ↔ ReqToTokenPool 行 0 ↔ KV 池 slot 0（6.1 §5） | CUDA Graph padded 假行的 gather / scatter 污染真实请求 |
| grammar CPU 顺序 | launch_batch_sample_if_needed 在 pop_and_process 后（L2329）；should_process_last_batch_early L2338 / _advance_pending_grammar L2359 | sample(j) 的 mask 用未推进的 FSM 状态，输出违反约束 |

WAR barrier 值得单独走读（L2207，摘录）：

```python
# sglang/srt/managers/scheduler.py _apply_war_barrier（main 2026-11-23，摘录）
runner = self.model_worker.last_shared_read_runner
ev = runner.shared_read_done_event
runner.shared_read_done_event = None
if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
    self.schedule_stream.wait_event(ev)      # 细粒度：只等共享读完成
else:
    self.schedule_stream.wait_stream(self.forward_stream)   # 粗粒度兜底
```

危害场景具体化：统一内存池下，post-process(j−1) 释放已完成请求的 KV 行、sched(j+1) 把这些行分配给新请求并写 req_to_token——这些是 schedule_stream 上的写；而 forward(j) 正在读同一批池行（WAR，write-after-read）。无 barrier 则写可能先于读完成，forward(j) 读到新请求的索引，输出静默错误。barrier 把「读完成」event 插进 schedule_stream，写被挡在读后。注意它 fence 的是**共享缓冲区的读写序**，不是 token relay 本身——token 正确性由同流序保证（命题 2），两者分工不同。

边界反例三个：(a) 若 stash 与 gather 分属两条流且无 event 栅栏，提交序不再约束执行序，gather 可能读到旧值——命题 2 的保持步失效；(b) 若去掉 WAR barrier，构造「第 $j$ 批 forward 未读完、第 $j-1$ 批已完成请求的 KV 行被重分配」的时序即得错误输出；(c) 若 grammar 请求的 sample 不等 pop_and_process，mask 基于落后一拍的 FSM 状态，可能放行非法 token。

> **坑**：把 −1 poison 初始化当成生产环境的哨兵机制。触发条件：见 _DEBUG_ASSERT 分支就推断「生产靠 −1 检测未初始化」；现象：该分支仅 SGLANG_IS_IN_CI 开启，生产 buf 是 torch.empty 未初始化，正确性完全靠流序 + 调用协议（先 stash 后 gather），不靠运行时检测。规避：引用 consume-once 断言时注明 CI-only 调试不变式（版本日期 2026-11-23）。

投机 v2 的扩展（证明要点，不单列命题）：accept 长度数据有关，CPU 调度时不可知，故 seq_lens 走 publish → new_seq_lens_buf → resolve_seq_lens_cpu 的 relay；publish 在 worker 内 record publish_ready event，resolve 侧私有 fwd_prepare_d2h_stream wait_event 后拷入 pinned 镜像再 synchronize——**只同步私有 D2H 流，不同步 schedule_stream**（L734-735 注释逐字：「don't sync the schedule stream; gate a private stream on the publish event」），调度主流不被 D2H 拖住。MIXED 批的投机尾行长度在 publish 栅栏后 late-bind（resolve_mixed_spec_tails L642）。正确性骨架与命题 2 同：event 栅栏替代同流序，承担跨流定序；逐行对照留读 resolve_seq_lens_cpu 源码（L687-747）。

## 6. cache-aware 调度与 continuous batching 实现

waiting 队列的准入顺序如何既保命中率又不重复 prefill？4.2 §5 已交付策略全集（九策略两族：CacheAware 的 LPM / DFS_WEIGHT / HRRN / SHORTEST_PREFILL_FIRST / UNIBOOST，CacheAgnostic 的 FCFS / LOF / RANDOM / ROUTING_KEY）与排序键公式（式 (5)(6)），本节走读引擎接线与两个公共机制的实现，不重复推导。

接线：get_new_batch_prefill（scheduler.py L4553）经 SchedulePolicy.calc_priority 排序 waiting 队列后逐请求过 PrefillAdder。bounded LPM（schedule_policy.py L181，BOUNDED_LPM_WINDOW_SIZE = 64）：waiting 队列超阈值时只在队首 64 个窗口内做前缀匹配，窗口外按 FCFS 刷新——注释逐字：「Bound matching work and cross-batch overtaking without a new server argument」，限的是每步匹配开销（2.2 命题 1 的 $O(T)$ 乘以队列长）与跨批越序。

**in-batch prefix dedup**（schedule_policy.py L114-127）：两个阈值环境变量，默认均 32 token。对缓存命中 < IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD（32）的请求，检查它与 waiting 队列内其他请求的共享前缀；若队列内命中 > IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD（32），该请求被暂时压到队尾——等 leader 请求先把公共前缀算进缓存，跟进者直接命中。机制账：两条请求共享 800 token 未缓存前缀，不 dedup 则各算一遍；dedup 后第二条命中 800 token，省 $2 \times 7.5 \times 10^{9} \times 800$ = 12 TFLOP（章脊口径，线性项；attention 项按 2.2 式 (2) 另计）。这与 4.2 §5 的 temporary deprioritized 是同一机制的两个阈值面。

**PrefillAdder**（schedule_policy.py L1077）是准入的预算账本，字段 ↔ 4.3 公式对照：

| 字段 | 语义 | 公式对应 |
| --- | --- | --- |
| rem_input_tokens | 本步 prefill token 预算（减去混批 decode token 数 num_mixed_decode_tokens） | 4.3 式 (2) 的 token 预算项 |
| rem_chunk_tokens | chunk token 帽（chunked_prefill_size 派生） | 4.3 $c_{\mathrm{chunk}}$ |
| new_token_ratio | 为 running 请求预留未来 decode token 的比例 | 准入的 KV headroom（下段） |
| memory_budget | KV 池可用 + 可驱逐回收的总预算 | 2.1 块池水位 / 4.2 抢占触发 |
| chunked_req_limit / dllm_config | 续 chunk 请求限额 / diffusion 特化 | 4.3 / 9.4 指针 |

add_one_req（L2103）逐请求定价：_select_prefill_admission 给出 (prefix_len, extend_len, is_chunked)——extend_len 超 rem_chunk_tokens 即截断成 chunk（4.3 的混合批由此而来）；AddReqResult 四值（CONTINUE / NO_TOKEN / OTHER / UNSCHEDULABLE，L1012）区分「预算尽」「装不下」「永不可调度」（后者直接 abort 不排队）。

new_token_ratio 是动态估计器（scheduler_components/new_token_ratio_tracker.py NewTokenRatioTracker，2026-11-23 核实）：init = SGLANG_INIT_NEW_TOKEN_RATIO（默认 0.7）× schedule_conservativeness，每步 decay 向 min = init × 0.14 衰减（600 步 decay 窗），retract 发生后按实测重估 (已解码 + 20 × 请求数) / (max_new_tokens 总和 + 1)。语义：预留 = max_new_tokens × ratio 个未来 token 的 KV headroom——悲观起步防准入过满触发 retract，随观测放松。代入章脊：max_new_tokens = 512，init 0.7 预留 ≈ 358 token ≈ 44.8 MiB/请求；衰减到 0.098 后预留 ≈ 50 token ≈ 6.3 MiB——同一请求的预留差 7×，这是「准入保守度」的运行时旋钮（retract 成本账见 4.2 §2）。

continuous batching 的数据结构 recap 4.1 §6 一句：Req（schedule_batch.py L442）持单请求状态，ScheduleBatch（L2009）持批状态与 forward_mode，prefill / decode 两批对象步边界合并。Req 的字段分三组：prompt 状态（origin_input_ids）、输出状态（output_ids）、KV 状态（kv.req_pool_idx / prefix_len / last_node）——第三组正是接 6.1 §4-§5 基数树与两级池的把手，overlap 的 relay 索引（req_pool_idx）也取自这里。本节增量是 NextBatchPlan（L3699）——get_next_batch_to_run 返回 (batch_to_run, running_batch) 对，overlap 循环据此更新 running_batch（§2 摘录）；update_running_batch（L5262）内 check_decode_mem 失败触发 retract_decode（SGLang 的抢占形态，4.2），new_token_ratio 随之重估（上段）。

## 7. chunked prefill 准入：prefill_delayer 与 shortest_prefill_load

chunk size 定了（4.3），准入时机与顺序谁定？4.3 已交付 stall 账与 chunk size 权衡（式 (2)(6)），本节走读当前 main 的两个准入装置——它们回答的是 4.3 未展开的两个问题：KV 紧张时 prefill 该不该等（delayer），长短请求混跑时谁先（load-scaled shortest-first）。

**PrefillDelayer**（managers/prefill_delayer.py L76，2026-11-23 核实）：prefill 准入延迟器，机制是 **batch size 效率 / GPU 利用率**，不是 KV 容量。延迟触发两个条件（L275-280）：slot 条件 = max_running_requests − global_running_batch_max < global_max_prefill_bs_max——剩余 decode slot 装不下最近 prefill 批高水位，此时放 prefill 进批，源码注释逐字（L281-282）：「the first merge_batch causes the decoding to fail to reach the maximum batch size」；queue 条件（opt-in，显式设 queue_min_ratio 才启用）= waiting 队列 < min(running × ratio, prefill 容量)（L252-271）——攒不够大 prefill 批就延迟，针对 decode 逐条完成、把 prefill 打碎成小批的 workload。**低水位是反向的安全阀**：token_usage < token_usage_low_watermark（池快**空**、GPU 利用不足）时强制放行、跳过延迟（L177-181，注释逐字 L230-231：「Safety valve: low KV usage means GPU is underutilized, skip delay」）——不是延迟触发。延迟上限 max_delay_passes 与 max_delay_ms（未设时 5000 ms 本地安全上限，源码注释：「a local safety cap, not a semantic default」）。RecentPrefillBatchSizeTracker（L26）跟踪最近 16 次非空 prefill 尝试的最大 batch size 高水位——decode-only / idle pass 不老化水位（docstring 逐字），该高水位正是 slot 条件里的 global_max_prefill_bs_max。PrefillDelayerSinglePassExecutor（L384）每 pass 与准入协商（negotiate_should_allow_prefill，跨 rank gather 全局状态）。机制账：延迟 prefill 让 decode 凑满批 / prefill 攒大批，用 TTFT 换批效率——与 4.2 §2 的 retract 账（KV 容量压力）是不同触发源，勿混。

**shortest_prefill_load.py**（模块 docstring 逐字口径，2026-11-23 核实）：「Load-scaled shortest-prefill-first」——类边界与 prefill token 预算随 backlog 缩放。backlog $B$ = waiting 请求未缓存 prompt token + 续 chunk 剩余（per DP attention rank）。记号消歧：此 $B$ 沿代码 docstring 原记号 = backlog（waiting 队列工作量），非 spec §4.3 的批内 $B$； $C$ 即 4.3 $c_{\mathrm{chunk}}$ 的 per-rank 值。设 chunk size $C$、halving 参数 $K$（chunks 单位），默认派生 decay_start = $2K$ chunks、budget min_ratio = 1/8、doubling = $K$ chunks（ShortestPrefillLoadScaling.resolve，msgspec Struct）：

$$
\mathrm{boundary}(B) = C \cdot 2^{-\max(0,\ B - D_{\mathrm{start}}) / H} \qquad (4)
$$

**符号**： $H = K \cdot C$（halving_tokens，每 $H$ token backlog 边界减半）； $D_{\mathrm{start}} = 2K \cdot C$（decay 起点）。边界以下的请求 shortest-first 且可并入续 chunk，其余按到达序、永不并入——过载时边界趋零，序趋近 FCFS（防饥饿的类边界设计，对照 4.2 §5 SHORTEST_PREFILL_FIRST 的 aging）。

$$
\mathrm{budget}(B) = \min\left(C,\ r_{\min} \cdot C \cdot 2^{B / D}\right) \qquad (5)
$$

**符号**： $D = K \cdot C$（budget_doubling_tokens）； $r_{\min}$ = 1/8； $B \ge B_{\mathrm{full}} = D \log_2 (1 / r_{\min})$ 后不再 cap（预算达到 $C$）。语义：backlog 轻时每批 prefill 预算小（保 decode 步时），backlog 重时指数放大到 $C$（吞吐优先）——4.3 §6「chunk size 是运行时变量」原则的负载自适应版。

全数字代入（ $C$ = 2048、 $K$ = 4，则 $H = D$ = 8192、 $D_{\mathrm{start}}$ = 16384、 $B_{\mathrm{full}}$ = 24576，python 复核 2026-11-23）：

| backlog $B$ (token) | boundary (token) | budget (token) |
| --- | --- | --- |
| 8192 | 2048 | 512 |
| 16384 | 2048 | 1024 |
| 24576 | 1024 | 2048 = $C$（达帽，不再 cap） |
| 32768 | 512 | 无 cap |

读表：backlog 从 8192 涨到 32768（4×），边界从 2048 缩到 512（短作业优先的范围收窄），预算从 512 放到无 cap——轻载保延迟、重载保吞吐，两端行为都由同一组公式给出。dynamic_chunk_sizer.py（DynamicChunkSizer L37 / ChunkSizePredictor L267，2026-11-23 核实）是 chunk size 本身的运行时求解器，与本题的准入预算正交（4.3 §6 指针）；配置经 runtime_context get_schedule()（L1351）读取，chunked_prefill_size 迁出 server_args 的漂移注 4.3 §5 已给，不重复。

## 8. DP attention controller：请求级负载均衡

MLA 模型单 KV head，TP8 会把 KV cache 复制 8 份（blog 逐字：「it will lead to duplicated KV cache and unwanted memory usage」）——attention 改用 DP、MoE 前 all-gather / 后 redistribute，是 v0.4 的解法（blog benchmark：DeepSeek-Coder-V2-Instruct-FP8，8×H100 80GB，random-input 1 / output 512、10000 prompts，**1.9× decode 吞吐** vs v0.3，[C]）。通信账与并行策略归 8.1，本节走读请求分发的控制器实现。

DataParallelController（managers/data_parallel_controller.py L159，1003 行文件）把请求 dispatch 到多个 DP worker（每 worker 独立 Scheduler + 独立基数树）。负载均衡方法 LoadBalanceMethod（L106，当前四值，2026-11-23 核实）：

| 方法 | 语义 | 适用 |
| --- | --- | --- |
| ROUND_ROBIN | 轮转 | 无状态基线 |
| FOLLOW_BOOTSTRAP_ROOM | 跟随 PD bootstrap room（prefill 侧定） | PD 分离（6.3 指针） |
| TOTAL_REQUESTS | 选 running + waiting 请求数最小的 rank | 请求数均衡 |
| TOTAL_TOKENS | 选 total_tokens 最小（同值比请求数） | token 工作量均衡 |

DPBudget（L123）维护各 rank 负载快照：update_budget 从 shm 快照更新、按 timestamp 跳过 stale 读；dispatch 对 TOTAL_REQUESTS 取 argmin、TOTAL_TOKENS 取 (tokens, requests) 字典序 argmin，分发后把目标 rank 负载 +1（启发式，防同一快照内连续 dispatch 全压一个 rank）。注意这是**请求级** LB（controller 分发），与 vLLM 的 DPCoordinator 同层（5.1 走读）；worker 内部的 cache-aware 排序（§6）是第二层。blog 的 cache-aware load balancer（router 维护 worker 真实 radix tree 的近似树、lazy 更新、Rust 实现 sglang-router，up to 1.9× 吞吐 / 3.8× 命中率）是第三层（实例间路由），归 8.4 / 9.2 对照，本章不展开。

每个 DP rank 独立处理 prefill / decode / idle 批（blog 口径）：无序列可跑的 rank 发 ForwardMode.IDLE 批（6.1 §3）陪跑 MoE 前后的 collective，保证各 rank 同步进入 all-gather / redistribute——idle 不是空转浪费，是 DP 组同步的占位。layers/dp_attention.py（DpPaddingMode L153 / set_dp_buffer_len L354）管 buffer 几何，scheduler_components/dp_attn.py 管调度侧协商，一句注。DP attention 与 overlap 的交互也要登记：每个 rank 的 scheduler 独立跑 event_loop_overlap，require_mlp_sync 时经 maybe_prepare_mlp_sync_batch（scheduler.py L4487 / L4506）协商同步批，idle 批保证无序列 rank 也进入 collective——重叠改的是各 rank 内部的 CPU/GPU 时序，不破坏 DP 组的步调一致；投机 + DP 下 prefill 与 decode 批不混批的协调分支（L4477-4488）是声明过的例外。

> **坑**：把 `--enable-dp-attention` 当当前配置字段。触发条件：照 v0.4 blog 命令或旧教程写 server_args 锚点；现象：当前 main enable_dp_attention **不再是字段**——server_args.py L345 / L355 的 resolved_dict 里它是派生报告值（attn_dp_size > 1），供 /server_info 客户端兼容。规避：配置锚 arg_groups/parallel_hook.py + runtime_context get_parallel()（L1314）；派生公式 attn_tp_size = tp_size // attn_dp_size // attn_cp_size（derive_attn_tp_size，runtime_context.py L73），tp_size = world size；blog flag 作历史口径注（版本日期 2026-11-23）。

## 9. batch_overlap/ ≠ overlap scheduler：TBO 与 SBO

`batch_overlap/` 目录名字里有 overlap，它就是本章的 overlap scheduler 吗？**不是**——目录名易误导，两者是不同层次的机制，必须区分：本章主线是 **CPU ↔ GPU 重叠**（event_loop_overlap + FutureMap，调度层）；batch_overlap/ 是 **GPU 上 compute ↔ comm 重叠**（算子层，MoE 场景），两个机制、两个层次、两套代码。

**TBO（two-batch overlap，`batch_overlap/two_batch_overlap.py`）**：把一个 batch 切两半（compute_split_seq_index L80），两半 ping-pong 执行——一半的 MoE dispatch / combine 通信与另一半的计算重叠。锚点：TboCudaGraphRunnerPlugin L338（CUDA Graph 兼容）/ TboDPAttentionPreparer L395（DP attention 下的切分准备）/ TboForwardBatchPreparer L503（批切分与 metadata）/ MaybeTboDeepEPDispatcher L1161（DeepEP dispatcher 的 TBO 包装）；执行经 operations.py 的 execute_operations / execute_overlapped_operations 双入口。**SBO（single-batch overlap，`single_batch_overlap.py`）**：单批内部的重叠——SboFlags（L28）开关 combine ↔ down-gemm 双流重叠（CombineOverlapArgs L63 / DownGemmOverlapArgs L75，num_sms / signal / event 参数化），不切批。

四机制对照（vLLM DBO 与 NanoFlow 为对照基线，各归其章）：

| 机制 | 重叠对象 | 粒度 | 锚点 / 出处 |
| --- | --- | --- | --- |
| overlap scheduler（本章） | CPU 调度 ↔ GPU 执行 | 迭代级 | scheduler.py event_loop_overlap L2266 + overlap_utils.py FutureMap L309 |
| TBO | compute ↔ comm（MoE dispatch/combine） | 批切两半 | batch_overlap/two_batch_overlap.py（本节） |
| SBO | combine ↔ down-gemm 等 | 单批内双流 | batch_overlap/single_batch_overlap.py SboFlags L28 |
| vLLM DBO（ubatching） | compute ↔ comm | 批切两半 + 两 worker 线程 | 5.3 走读（docs/design/dbo.md 同族口径） |
| NanoFlow | compute / memory / network | 设备内 nano-batch | 3.4 §7 式 (4)（[Zhu et al., 2024](../references.md)，arXiv:2408.12757，[A]） |

TBO / SBO 与 vLLM DBO 同族（batch 切分 + 通信计算重叠），通信账与 EP 部署归 8.1；本章登记语义边界，不展开算子细节。

## 10. batch_invariant_ops 与同源对照：SGLang / NanoFlow → DeepInfer

overlap 调度与确定性推理有没有交互？有，而且落在 chunk 切分上。batch_invariant_ops/batch_invariant_ops.py 的 kernel 机制（固定 reduction 顺序、禁 atomicAdd 非确定累加）3.4 §6 已交付，recap 一句 + 指引，不重复。本章增量是调度侧耦合：PrefillAdder 的 exact_chunk_fill 开关（schedule_policy.py L1125-1131）在 deterministic 模式下关闭——源码注释逐字：「deterministic mode keeps chunk ends on pages, which some batch-invariant kernels depend on」。chunk 末端对齐 page 是 batch-invariant kernel 的前提，调度器为此放弃精确填满 chunk 的自由度：确定性不是 kernel 一家的事，调度预算也要让路。

同源对照收束本章。DeepInfer 的 async scheduling 官方 doc 逐字（[DeepInfer doc](../references.md)，教学参照 + 官方实现，官方披露，doc L334，2026-11-23 核实）：「Following existing implementations (SGLang, NanoFlow), the pipeline is overlapped by scheduling asynchronously between GPU and CPU: while the GPU runs forward(N) → sample(N), the CPU concurrently runs schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1), leaving the GPU with no idle time throughout.」——lineage 自认 SGLang 与 NanoFlow，机制与本章式 (2) 同构（7.1 详展回链）。NanoFlow 一侧，v0.4 blog 逐字自认：「This idea is simple and has been proposed in NanoFlow」；NanoFlow 的设备内 nano-batch 流水（3.4 §7 式 (4)， $F_{\max} \le 3$）是算子层版本，本章的迭代级 CPU/GPU 重叠是调度层版本，两者叠加不冲突。vLLM V1 的 async scheduling（5.2 走读）目标相同、依赖解析不同：官方设计文档口径「while the GPU executes step N, the CPU schedules step N+1, using placeholders for the tokens that step N will produce」（[vLLM repo](../references.md)，docs/design/arch_overview.md）——placeholder 在 CPU 侧占位、worker 侧解析；SGLang 当前实现是 pool-indexed relay、GPU 流上 gather（§3）。

| 系统 | 重叠目标 | 依赖解析 | 锚点 |
| --- | --- | --- | --- |
| SGLang | CPU schedule/post ↔ GPU forward/sample | FutureMap stash/publish/resolve（GPU 流序） | 本章 §2-§5 |
| vLLM V1 | 同 | placeholder + worker 侧状态缓存 | 5.2（AsyncScheduler） |
| DeepInfer | 同 | 仿 SGLang / NanoFlow（官方自认） | 7.1（doc L334） |
| NanoFlow | compute/memory/network（设备内） | nano-batch 流水 | 3.4 §7 |

## 总结

| 论断 | 回指 |
| --- | --- |
| normal 循环 GPU 每步 idle $c_s + c_p$；blog 口径未优化引擎至多一半时间在 CPU overhead | 式 (1)、§1 |
| overlap = 结果处理延迟一拍 + sample 延迟 launch（grammar）+ 提前处理闸门（TTFT / grammar sync） | §2、event_loop_overlap L2266 |
| FutureMap = always-on pool-indexed relay（req_pool_idx 索引、slot 0 padding 镜像）；blog placeholder 是概念口径，当前是 stash/publish/resolve | §3、overlap_utils.py L309、漂移注 |
| CPU 调度只消费数据无关量；token 值与 accept 长度数据有关，GPU 流上解析 | §3 分类表 |
| bubble 消除条件 $c_{\mathrm{cpu}} \le t_{\mathrm{gpu}}$； $F = (c_{\mathrm{cpu}} + t_{\mathrm{gpu}}) / \max(c_{\mathrm{cpu}}, t_{\mathrm{gpu}})$，完全隐藏时 $F = 1 + c_{\mathrm{cpu}} / t_{\mathrm{gpu}} \le 2$ | 命题 1、式 (2)(3)、练习 1 |
| 章脊代入 $c_{\mathrm{cpu}}$ = 1 ms → $F$ ≈ 1.195；blog 实测 1.1×（vs v0.3）/ 1.3×（vs SOTA）；Nsight 5 batch 零 idle（Triton 口径） | §4 表、练习 1 |
| resolve 正确性 = 同流提交序归纳；四要件：流序 / WAR barrier / slot 0 / grammar CPU 顺序 | 命题 2、§5、练习 2 |
| WAR barrier fence 共享池读写序（schedule_stream 写挡在 forward 读后），与 token relay 分工不同 | §5、_apply_war_barrier L2207 |
| in-batch dedup 双阈值 32/32；bounded LPM 窗口 64；PrefillAdder 预算 ↔ 4.3；new_token_ratio 0.7 → 0.098 动态预留 | §6、4.2 §5 / 4.3 指针 |
| PrefillDelayer slot/queue 条件延迟 prefill 换批效率（低水位是 force-allow 安全阀，非触发）；shortest_prefill_load 边界 / 预算随 backlog 指数缩放，过载趋 FCFS | §7、式 (4)(5)、练习 4 |
| DP controller 四值 LB + DPBudget shm 快照；enable_dp_attention 已派生（attn_dp_size > 1），blog flag 历史口径 | §8、漂移注 |
| batch_overlap/ = TBO/SBO（GPU compute↔comm），非 overlap scheduler（CPU↔GPU）；与 vLLM DBO 同族 | §9 对照表 |
| DeepInfer async scheduling 自认 SGLang / NanoFlow 同源；deterministic 模式令 chunk 末端对齐 page | §10、doc L334、7.1 回链 |

## 练习

1.（计算 · [10]）章脊口径 $t_{\mathrm{gpu}}$ = 5.13 ms：(a) $c_{\mathrm{cpu}}$ = 0.5 / 3 / 8 ms 时按式 (3) 算 $F$ 与 normal GPU 利用率；(b) $c_{\mathrm{cpu}}$ 为何值时 $F$ 取最大，最大多少？(c) 若模型换小使 $t_{\mathrm{gpu}}$ 减半到 2.56 ms、 $c_{\mathrm{cpu}}$ 不变（1 ms）， $F$ 变为多少，方向与 blog「小模型收益大」是否一致？校验：(a) 与 §4 表对应行一致（python 复核值 @5.13 ms：1.097 / 1.585 / 1.641）；(b) 用命题 1 的临界条件；(c) $F = 1 + c_{\mathrm{cpu}} / t_{\mathrm{gpu}}$ 随 $t_{\mathrm{gpu}}$ 减小而增大。
2.（证明 · [20]）补全命题 2 的保持步并构造 WAR 反例：(a) 写出 stash(j−1) 与 gather(j) 之间 forward_stream 上的完整提交序（含 wait_stream 位置），论证执行序；(b) 构造一个去掉 _apply_war_barrier 后的时序反例：第 $j$ 批 forward 仍在读 KV 池行时，post-process(j−1) + sched(j+1) 在 schedule_stream 上释放并重分配同一行，指出 forward(j) 读到什么、输出为何错；(c) 说明 slot 0 镜像为何让 CUDA Graph padded 批的 gather 无害（req_pool_idx = 0 的读写落在哪）。校验：(b) 的反例必须违反「schedule_stream 写挡在 forward 共享读后」的 barrier 语义（L2207 注释）；(c) 与 6.1 §5 的 ReqToTokenPool 行 0 / KV 池 slot 0 三处同构一致。
3.（代码 · [30]）读 scheduler.py 的 event_loop_overlap（L2266）/ should_process_last_batch_early（L2338）/ launch_batch_sample_if_needed（L5928）与 tp_worker.py L747-767（main 2026-11-23），先写预测再对照：(a) 非 grammar decode 批的 sample 在哪一拍、哪个函数内 launch？grammar 批的 sample 又在何时 launch？(b) SGLANG_DISABLE_OVERLAP_AFTER_PREFILL 开启后 prefill 批的结果处理时序如何变，为什么保 TTFT？(c) result_queue 里存 batch.copy() 而非 batch 本身，若去掉 copy 会被下一步的哪个操作改写？校验：(a) 对照 tp_worker delay_sample_func 条件与 L2329 注释；(b) 对照 L2346-2354 的 DP 同步注释；(c) 对照 get_next_batch_to_run 的 merge_batch / filter_batch（L4440-4463）。
4.（计算 · [30]）shortest_prefill_load 账（式 (4)(5)， $C$ = 2048、 $K$ = 4）：(a) 算 $H$、 $D_{\mathrm{start}}$、 $B_{\mathrm{full}}$；(b) backlog $B$ = 12288 / 20480 / 28672 token 时的 boundary 与 budget（budget 达 $C$ 后写「无 cap」）；(c) 若 $K$ 改 8， $B$ = 24576 时 boundary 变为多少，过载趋 FCFS 的速度变快还是变慢？校验：(b) 与 §7 表同公式（python 复核： $B$ = 16384 → budget 1024、 $B$ = 24576 → 达帽）；(c) $K$ 翻倍使 $H$ 与 $D_{\mathrm{start}}$ 同翻倍， $B$ = 24576 尚未过 decay 起点（boundary = $C$ = 2048），边界收缩更慢。

## 参考

- [SGLang v0.4 blog](../references.md)：SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs，LMSYS，2024-12-04，[C]（key: sglang-v04-blog，开放技术博客、非闭源，2026-11-23 全文核实）——CPU overhead 至多一半、NanoFlow lineage 逐字、future tokens 概念口径、Nsight 5 batch 零 idle（Triton / FlashInfer gap 注）、1.1× / 1.3×、--disable-overlap、DP attention 1.9×（DeepSeek-Coder-V2-Instruct-FP8，8×H100）、cache-aware LB 1.9× / 3.8×
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——`srt/managers/scheduler.py`（event_loop_normal L2231 / event_loop_overlap L2266 / _apply_war_barrier L2207 / should_process_last_batch_early L2338 / _advance_pending_grammar L2359 / init_overlap L1837 / get_next_batch_to_run L4386 / get_new_batch_prefill L4553 / run_batch L5455 / _relay_forward_payload L5835 / launch_batch_sample_if_needed L5928 / process_batch_result L5957）、`overlap_utils.py`（FutureMap L309 / resolve_forward_inputs L127 / relay L420 / stash L780 / publish L749 / resolve_seq_lens_cpu L687）、`schedule_policy.py`（BOUNDED_LPM_WINDOW_SIZE L181 / in-batch 阈值 L114-127 / PrefillAdder L1077 / add_one_req L2103）、`schedule_batch.py`（Req L442 / ScheduleBatch L2009 / L2855 MIXED 注释 / NextBatchPlan L3699）、`prefill_delayer.py`（L26 / L76 / L177-181 force-allow / L230-231 安全阀注释 / L252-280 slot+queue 触发 / L384）、`shortest_prefill_load.py`、`data_parallel_controller.py`（L106 / L123 / L159）、`scheduler_components/{new_token_ratio_tracker,ipc_channels,batch_result_processor}.py`（copy_done.synchronize L1157）、`tp_worker.py` L624 / L747-767、`batch_overlap/{two_batch_overlap,single_batch_overlap,operations}.py`、`batch_invariant_ops/`、`server_args.py` L345 / L355、`runtime_context.py` L73 / L1314 / L1351（均 2026-11-23 main fresh tarball file:line 核实）
- [Zheng et al., 2023](../references.md)：SGLang: Efficient Execution of Structured Language Model Programs，arXiv:2312.07104，[A]（key: sglang）——RadixAttention 与 cache-aware 调度背景（§3）
- [Zhu et al., 2024](../references.md)：NanoFlow: Towards Optimal Large Language Model Serving Throughput，arXiv:2408.12757，[A]（key: nanoflow）——设备内 nano-batch 流水（3.4 §7 recap），v0.4 blog 自认 lineage
- [DeepInfer doc](../references.md)：doc/inference_deployment_qa.md，教学参照 + 官方实现（key: deepinfer-doc，官方披露口径注）——L334 async scheduling 逐字（Following existing implementations (SGLang, NanoFlow)...），7.1 详展
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——docs/design/arch_overview.md Async Scheduling 节逐字（placeholder 口径），5.2 走读对照
- 跨维度指引（纯文字）：inference 1.2（prefill / decode 账）、2.2（基数树匹配复杂度与命中节省）、3.4 §6 / §7（batch-invariant kernels、NanoFlow 式 (4)）、4.1 §6（调度循环机制级与式 (3) 步时）、4.2（策略全集与 retract 成本）、4.3（stall 账与 chunk size 权衡）、5.2 / 5.3（vLLM async scheduling 与 DBO）、6.1（srt 架构、两级池与 slot 0）、6.3（PD bootstrap room）、7.1（DeepInfer 架构）、8.1（DP attention 通信账与 EP）、8.4 / 9.2（cache-aware LB 与引擎对照）
