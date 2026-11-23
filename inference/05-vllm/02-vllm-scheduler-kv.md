# 5.2 vLLM 调度器与 KV 管理器走读（vLLM Scheduler and KV Manager Walkthrough）

> 4.1 的 token 预算、4.2 的抢占账、4.3 的 chunk 钳位、2.1 的块语义，在 vLLM V1 里各是哪几行代码？一个 full + SWA 混合模型的 prefix 命中为什么要跑「不动点」循环？本章对 `v1/core/sched/` 与 `v1/core/` 的 KV 四文件做模块全量走读，把 P1-P4 的公式逐行对到代码，并证明 hybrid 命中协调的收敛性 | 前置：inference 2.1（块表 / 碎片上界 / COW / watermark，机制级）、2.2（hash 链与命中节省）、4.1（迭代级调度）、4.2（抢占成本账）、4.3（chunked prefill）、5.1（V1 架构与数据流） | 锚点：vLLM [Kwon et al., 2023] [A]（arXiv:2309.06180）；代码 `vllm/v1/core/sched/` + `v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager,kv_cache_utils,kv_cache_planning,kv_cache_config_builder,encoder_cache_manager}.py` + `v1/{kv_cache_interface,kv_cache_layout}.py`（2026-11-23 main tarball file:line 核实）；官方 `docs/design/{arch_overview,hybrid_kv_cache_manager,prefix_caching}.md`

## 1. 两层走读定位与模块地图

机制章讲过的块语义，走读章还要再推一遍吗？不。2.1 已经交付块管理的**机制**：块表地址翻译、每请求浪费 ≤1 块的碎片上界（2.1 命题 1）、COW 正确性（2.1 命题 2）、`ref_cnt` 与驱逐候选、watermark 准入预留，并给了 `block_pool.py` 的 `touch` / `free_blocks` / `get_new_blocks` 机制摘录与协调器 LCM 摘录。本章不重推这些——recap 一句：**分页把 KV 切成定长块、按块表 gather、按引用计数共享与驱逐**（机制与证明见 2.1，hash 链见 2.2）；本章做的是同一批文件的**模块全量走读**：完整调度循环、队列实现、抢占完整流程、KV 三层架构、spec 数据类族、hybrid 协调算法、容量规划链。两层分工是 spec §4.2 的硬纪律：2.1 机制级 ↔ 5.2 代码级，重访片段一律 recap + 指引。

模块地图（2026-11-23 main 核实，行号 = 类/函数定义行）：

| 子系统 | 文件 | 关键类 / 函数 |
| --- | --- | --- |
| 调度器 | `core/sched/scheduler.py`（3993 行） | Scheduler L96 / schedule() L702 / _preempt_request L1831 / update_from_output L2334 |
| async 调度 | `core/sched/async_scheduler.py` | AsyncScheduler L10 / _update_after_schedule L18 |
| 请求队列 | `core/sched/request_queue.py` | SchedulingPolicy L13 / FCFSRequestQueue L75 / PriorityRequestQueue L131 |
| 调度输出 | `core/sched/output.py` | SchedulerOutput L241 / NewRequestData L41 / CachedRequestData L139 |
| 块池 | `core/block_pool.py`（936 行） | BlockPool L107 / get_new_blocks L681 / touch L774 / free_blocks L798 |
| 块元数据 + 自由队列 + hash | `core/kv_cache_utils.py` | KVCacheBlock L160 / FreeKVCacheBlockQueue L248 / hash_block_tokens L697 |
| KV 管理接口 | `core/kv_cache_manager.py`（935 行） | KVCacheManager L130 / get_computed_blocks L278 / allocate_slots L391 |
| 协调器 | `core/kv_cache_coordinator.py`（904 行） | Unitary L496 / Hybrid L565 / find_longest_cache_hit L704 |
| 逐类型管理器 | `core/single_type_kv_cache_manager.py`（3377 行） | FullAttentionManager L874 / SlidingWindowManager L1068 / MambaManager L1850 |
| KV spec 数据类 | `v1/kv_cache_interface.py`（963 行，**v1/ 根**） | KVCacheSpec L189 / AttentionSpec L456 / MLAAttentionSpec L525 / KVCacheSpecKind L150 |
| 张量布局 | `v1/kv_cache_layout.py`（58 行，**v1/ 根**） | KVCacheLayout L15（stride 排列枚举） |
| 容量规划 | `core/kv_cache_planning.py`（1817 行） | get_kv_cache_configs L1456 / check_enough_kv_cache_memory L195 |
| 规划 builder | `core/kv_cache_config_builder.py`（102 行） | KVCacheConfigBuilder L19 / get_kv_cache_config_builder L86 |
| encoder cache | `core/encoder_cache_manager.py`（450 行） | EncoderCacheManager L18 / can_allocate L148 |

> **坑**：三个高频锚点错误。其一，`kv_cache_interface.py` / `kv_cache_layout.py` 在 `vllm/v1/` **根目录**，不在 `core/`（W1 已钉、W3 再确认）；`kv_cache_layout.py` 的 `KVCacheLayout` 是张量 stride 排列枚举（LBHNC / LBNHC / …，逻辑形状恒为 [L, B, H, N, C]），**不是块管理器**，别与 manager 类族混。其二，**`FullAttentionSpec` / `SlidingWindowSpec` 类不存在**——spec 数据类是 `KVCacheSpec` / `AttentionSpec` / `MLAAttentionSpec` / `MambaSpec` + `KVCacheSpecKind` 枚举，full / SWA 的块几何差异由 **manager 类**（FullAttentionManager / SlidingWindowManager）承载（§6）；照旧类名写锚点会落空。其三，`v1/watermarking/` 是生成文本水印（detector / watermarker），**不是** KV watermark——后者是 `scheduler_config.watermark`（config/scheduler.py L232），两词同形异义。

**本章数值口径**（章脊锁定，沿用 2.1 / 4.2 / 4.3）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、bf16）， $\kappa$ = 4096 字节/token/层、 $\kappa L$ = 128 KiB/token； $N = 7.50 \times 10^{9}$（matmul 口径）； $F_{\mathrm{eff}}$ = 494.75 TFLOPS（50% × H100 bf16 dense 989.5，[NVIDIA datasheet](../references.md)，[C] 官方规格披露，2.4 教学假设）；block size $b$ = 64（当前默认，config/cache.py L185）；KV 池 48 GiB（2.1 教学口径）；token 预算 max_num_batched_tokens = 2048（**教学预算**，取 SchedulerConfig dataclass fallback，config/scheduler.py L40，docstring 逐字「mainly for convenience when testing」）。hybrid 例换 Gemma-3 / 教学 full+SWA 层分布，随文声明。

> **口径注**：2048 不是引擎运行默认。实际默认由 `EngineArgs.create_engine_config` 按硬件 × usage context 分档（engine/arg_utils.py L3002-3023，2026-11-23 核实）：H100/H200（≥70 GiB 非 A100）在线 OPENAI_API_SERVER = **8192**、离线 LLM_CLASS = **16384**；≥160 GiB（B200/B300）双 16384；其余档 8192/2048。本章算例统一用教学预算 2048（已声明），换实际默认只改预算数值、钳位链与公式不变。

## 2. schedule()：统一 token 视图与预算钳位

一步调度到底怎么把「prompt 还没算完」和「decode 只差一个 token」装进同一个循环？答案在 `schedule()`（scheduler.py L702）开头的注释，逐字：「There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec.」（4.1 §5 已引此注释，recap 一句：调度器不贴 phase 标签，只盯差值；此处走读完整钳位链）。每请求本步要算的 token 数（L824-828）：

```python
# vllm/v1/core/sched/scheduler.py L824-833（机制摘录）
num_new_tokens = (request.num_tokens_with_spec
                  + request.num_output_placeholders
                  - request.num_computed_tokens)
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold          # 长 prefill 帽
num_new_tokens = min(num_new_tokens, token_budget,
                     input_budget - draft_slots)            # 预算钳位
```

`num_tokens_with_spec` = prompt + 已生成 + draft token 总数，`num_computed_tokens` = 已算前缀长，差值就是「还差多少没算」——prefill 是差值 = prompt 长的特例，decode 是差值 = 1 的特例，chunked prefill 是差值分多步还，投机解码是差值含 draft（4.4 / 4.5）。`num_output_placeholders` 是 async scheduling 的在途 token 计数（§8）。钳位链有三道：`long_prefill_token_threshold`（长 prefill 帽，防一条长 prompt 饿死他人）、`token_budget`（= max_num_scheduled_tokens，L728）、`input_budget - draft_slots`（max_num_batched_tokens 扣 draft 槽）。随后还有 max_model_len 钳位（L837-842）与 Mamba 块对齐（L845）。

这正是 4.3 §5 教学伪码的官方实现。差异要明（教学重写 vs 官方）：4.3 的伪码只写 `q = min(remaining_prompt, budget)`，官方多了 threshold 帽、draft_slots 扣减、encoder 预算（`_try_schedule_encoder_inputs` L2111 会再缩 num_new_tokens）、Mamba 对齐与 lookahead 预留。`long_prefill_token_threshold` 还有自适应分支（L771-776）：开启 adaptive 时帽取 `max(threshold, input_budget // num_eligible_reqs)`——保证任何请求不被切到低于「预算 ÷ 请求数」的公平份额（config/scheduler.py L104 / L112）。

**代码 ↔ 4.3 公式对照**：4.3 式 (2) 的 iteration 时间模型里，chunk token 数 $c_{\mathrm{chunk}}$ 是 GEMM 相与 attention 相的自变量；官方代码里 $c_{\mathrm{chunk}}$ 就是钳位后的 `num_new_tokens`，上界由 token_budget 给（4.3 §3 的 $c_{\mathrm{hi}}$ 是 SLO 反推的理论上界，引擎用固定 token 预算近似它）。代入章脊：预算 2048、批内 64 条 decode（各差 1 token）、一条 8192 prompt 的新请求——running 段先扣 64，waiting 段给新请求 `min(8192, 2048−64)` = **1984** token（与 4.3 §5 伪码运行结果一致），余下 8192−1984 = 6208 在后续步继续；共 $\lceil 8192/1984 \rceil$ = **5** 步（1984×4 + 256）。token 预算固定工作量、步时随前缀浮动——这是 4.3 §5 「token 预算 vs 时间预算」对照的 vLLM 侧。

running 段之后是 waiting 段（L1041），准入条件第一条就是 `if not preempted_reqs`——**本步发生过抢占就不准入新请求**（4.1 §5 / 4.2 §2 已钉，防「刚腾出的块立刻被新请求吃掉」振荡）。准入前查 prefix cache 命中（L1132-1149，`_get_local_prefix_cache_hit` → `get_computed_blocks`），命中部分计入 `num_computed_tokens`、不占计算预算；新请求本步 token 数 = `request.num_tokens - num_computed_tokens`（L1348）。准入成功则 `running.append` + 状态置 RUNNING + `num_computed_tokens` = 命中长（L1596-1624）。waiting 队列还可按缓存前缀重排（`_reorder_waiting_by_cached_prefix` L2843，cache-aware 窗口开启时）——这是 vLLM 侧的 LPM 近似（4.2 §5 SGLang 九策略对照，vLLM 只有 FCFS / PRIORITY 两档 + 可选前缀重排）。

调度收尾构造 `SchedulerOutput`（L1758）：new 请求打包 `NewRequestData`（全量）、cached 请求打包 `CachedRequestData`（diff）（5.1 §5 命题 1 的二分协议），再调 `_update_after_schedule`（L1815 → L1879）**乐观推进**计数器：`request.num_computed_tokens += num_scheduled_token`。注释逐字解释为什么先推进：「Advance the number of computed tokens here allowing us to schedule the prefill request again immediately in the next scheduling step」——chunked prefill 的下一 chunk 不必等本步输出回来就能排，若后续有 spec token 被拒再在 `update_from_output` 里回调。这是 async 流水的计数基础（§8）。

## 3. 请求队列：deque 与 heap 的 prepend 语义

FCFS 与 PRIORITY 两档策略，落到数据结构是哪两个队列类？答案是 deque 与 heap（request_queue.py）。`SchedulingPolicy`（L13）只有 FCFS / PRIORITY 两值；`create_request_queue`（L200）按策略建队。`FCFSRequestQueue`（L75）继承 `deque`：`add_request` = append（队尾进）、`pop_request` = popleft（队头出）、`prepend_request` = appendleft（插队头）。`PriorityRequestQueue`（L131）是 heap：请求按 `(priority, arrival_time)` 排序，**priority 值小者优先、同值早到优先**（docstring 逐字）；关键差异在 `prepend_request`（L159）——heap 没有「插队头」概念，它直接 `add_request` 按序重插，docstring 逐字「In a priority queue, there is no concept of prepending to the front」。

这个差异有调度后果：抢占恢复时 `_preempt_request` 调 `waiting.prepend_request`（§4），FCFS 下被抢请求回**队头**（下一步优先恢复），PRIORITY 下回**按优先级排序的位置**（低优先级者可能继续排后面）。4.2 §1 已钉 victim 选择（FCFS 抢最晚到、PRIORITY 抢最低优先级），此处补恢复侧的队列语义——抢占与恢复共用同一个 `prepend_request` 接口，但两档策略的落点不同。`RequestQueue` 抽象基类（L20）还规定 `peek_request` / `remove_request` / `prepend_requests` 等接口，换策略只换队列实现，块池与抢占机制不动（4.2 §1 提点「机制与策略分离」的代码体现）。config 默认 `policy = "fcfs"`（config/scheduler.py L195）。

## 4. 抢占走读：recompute-only 与 stale output

KV 池装满、running 请求下一步增量分不到块时，调度器牺牲谁、怎么牺牲？`schedule()` 的 running 段里，`allocate_slots` 返回 None 就进抢占重试循环（L898-969）：选 victim → 释放 → 重试分配，直到装下或无可抢者。victim 选择（L917-923，4.2 §1 已引）：PRIORITY 取 `max(running, key=(priority, arrival_time))`，FCFS 取 `running[-1]`。牺牲动作本体是 `_preempt_request`（L1831-1877，机制摘录）：

```python
# vllm/v1/core/sched/scheduler.py L1849-1876（机制摘录）
self._free_request_blocks(request)          # 释放 KV 块回池
self.encoder_cache_manager.free(request)    # 连带释放 encoder cache
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0             # 进度清零 → 恢复时重算
request.spec_token_ids = []
request.num_stale_output_tokens = request.num_in_flight_tokens  # async 在途输出标 stale
request.num_output_placeholders = 0
request.num_preemptions += 1
self.waiting.prepend_request(request)       # 回 waiting（FCFS 队头 / PRIORITY 按序）
```

**V1 抢占 = recompute-only，无 swap 路径**（4.2 §1 已钉版本事实，此处给官方文档逐字佐证，arch_overview.md Preemption 节）：「When re-admitted, it is recomputed rather than swapped back from CPU memory. Prefix caching or a KV connector can restore part of its progress immediately.」释放的块带 hash 仍是驱逐候选而非立即失效（2.1 §4），所以恢复时 `get_computed_blocks` 可能命中前缀、免算那段——recompute 成本是上界。

**代码 ↔ 4.2 / 2.2 公式对照**：4.2 式 (1) 给 recompute 成本 $t_{\mathrm{recompute}} = (2NT + 2LT^{2}d_{\mathrm{model}})/F_{\mathrm{eff}}$（一次 prefill，非 $T$ 次 decode）；2.2 式 (2) 给命中 $\delta$ 前缀的节省 $\Delta\mathrm{FLOPs} = 2N\delta + 2L d_{\mathrm{model}}\delta^{2}$。两者相减就是被抢请求恢复的实际成本。代入章脊 $T$ = 4096、命中 $\delta$ = 3072（75%）：全量 recompute FLOPs = $6.58 \times 10^{13}$（4.2 §2 表，按 $F_{\mathrm{eff}}$ 折 133 ms），命中节省 FLOPs = $4.86 \times 10^{13}$（2.2 §4；**口径差须声明**：2.2 原文按峰值 989.5 TFLOPS 理想折算省约 49 ms，本章按 $F_{\mathrm{eff}}$ = 494.75 TFLOPS 折算 98.1 ms，两口径差 2×），实际 ≈ $1.73 \times 10^{13}$ FLOPs ≈ **34.9 ms**（本章 $F_{\mathrm{eff}}$ 口径，133 − 98.1）——prefix cache 把抢占恢复成本压到约 1/4。这就是官方文档「restore part of its progress immediately」的定量含义（公式本体见 4.2 / 2.2，此处只做代码场景代入，不重推）。

async scheduling 下抢占还有一个微妙处：被抢请求可能还有在途步的输出没回来。`_preempt_request` 把这些输出标 `num_stale_output_tokens`（L1867），注释逐字：「Its tokens are still delivered on return (dropping them would perturb spec-decode acceptance) but must not mutate the reset counters」——stale 输出照常交付（丢掉会扰动投机接受率），但不再改动已清零的计数器，`update_from_output`（L2406-2410）逐步 drain 这份 stale 份额。只有 `drop_stale_output` 模式（如 `reset_prefix_cache` 同步抢占恢复、或 KV connector 待交接）才真丢（L1864）。这是「抢占不丢 token」与「计数器一致」两个目标的折中，正确性依赖 in-flight 计数的精确 drain。

**watermark 与准入门**（recap 2.1 §4 / 4.2 §2，补代码细节）：watermark 是准入预留，不是抢占触发器。`KVCacheManager` 构造时 `watermark_blocks = int(watermark × num_blocks)`（kv_cache_manager.py L200），`allocate_slots` 只对 WAITING / PREEMPTED 请求加 watermark（L529-536），且多一个条件 `has_scheduled_reqs`——本步已有请求被调度时才加（避免空步过度保守）。分配判据（L595-599）：`required = num_blocks_to_allocate + watermark_blocks`，`available = free_blocks - reserved_blocks`，`required > available` 返回 None。当前默认 watermark = 0.0（config/scheduler.py L232）。代入章脊：池 6144 块、watermark = 0.01 → 61 块 = 488 MiB 准入预留（2.1 §4 同值）。另有一道独立的准入门 `full_sequence_must_fit`（L542-559，由 `scheduler_reserve_full_isl` 默认 True 触发，config/scheduler.py L226-230）：准入前检查**全输入序列**装得下，而非只查首 chunk——防 chunked prefill 过度准入导致 KV 抖动（thrashing）。检查长度 = `min(request.num_tokens, max_model_len)`（L544），`num_tokens` = prompt + **已生成** token（request.py L338-339，`len(_all_token_ids)`）：新请求即全 prompt 长、恢复请求含已生成部分，**不含未来生成**（docstring 逐字「full input sequence length」）。真正的抢占触发链仍是 running 请求 `allocate_slots` 返回 None（4.2 §2），watermark 只调节其频率。

## 5. KV 管理器三层架构与块池实现

KV 管理为什么分三层？因为「调度器接口」「跨组协调」「逐类型块逻辑」是三件事。`KVCacheManager`（kv_cache_manager.py L130）是调度器看到的唯一接口；它持一个 `KVCacheCoordinator`（协调器）与一个共享 `BlockPool`；协调器持一组 `SingleTypeKVCacheManager`（逐类型管理器），每组一个。工厂 `get_kv_cache_coordinator`（kv_cache_coordinator.py L850）按配置选协调器：prefix caching 关 → `KVCacheCoordinatorNoPrefixCache`（L449）；单组 → `UnitaryKVCacheCoordinator`（L496）；多组 → `HybridKVCacheCoordinator`（L565）。

```
Scheduler ──► KVCacheManager (L130, 接口)
                 │  get_computed_blocks / allocate_slots / free / cache_blocks
                 ▼
              KVCacheCoordinator (L84, 协调)
                 │  Unitary(单组) / Hybrid(多组不动点) / NoPrefixCache
                 ├──► BlockPool (L107, 全组共享一个块池)
                 │      free_block_queue + cached_block_hash_to_block + null_block
                 └──► SingleTypeKVCacheManager × 组数
                        FullAttentionManager / SlidingWindowManager / MambaManager …
```

协调器基类（L84）钉两条结构约束：调度粒度 `scheduler_block_size` = 各组块大小的公倍数（LCM 断言 L106，2.1 §6 已引摘录，recap：保证 chunk 与命中边界同时落在每组块边界上）；全组共享**一个** BlockPool（L112），各组块大小可不同但 page size 统一（§7）。`single_type_managers` 元组逐组创建（L121）。

**BlockPool 全量走读**（2.1 §4 给了 touch/free/get_new_blocks 机制摘录，此处补自由队列实现与缓存登记）。`BlockPool`（L107）持三件：`blocks`（全部物理块列表）、`free_block_queue`（自由队列）、`cached_block_hash_to_block`（hash → 块映射，`BlockHashToBlockMap` L38）。初始化取 0 号块作 `null_block`（L164，2.1 §4 已钉：SWA 窗外 / Mamba 跳过块的占位，ref_cnt 不维护）。三个操作：

- `get_new_blocks`（L681）：`free_block_queue.popleft_n(num_blocks)` 取块，逐块 `_maybe_evict_cached_block`（L751，清 hash 登记）再置 `ref_cnt = 1`。取块即覆盖内容，缓存登记必须先清（2.1 §4 坑注）。
- `touch`（L774）：命中复活，`ref_cnt == 0` 先从自由队列 `remove` 再 +1（2.1 §4 摘录）。
- `free_blocks`（L798）：`ref_cnt` −1，归零按有无 hash 分流——无 hash 块 `prepend_n` 队头（LIFO 立即复用），有 hash 块 `append_n` 队尾（FIFO / LRU 驱逐）。源码注释逐字：「LIFO reuse of non-cached blocks for better GPU locality」与「FIFO reuse of cached blocks for LRU eviction behavior」（2.1 练习 4 已引）。

自由队列 `FreeKVCacheBlockQueue`（kv_cache_utils.py L248）是 blog 「zero-overhead prefix caching」的数据结构核心。它不用 Python `deque`，而是**双向链表**，docstring 逐字：「support removing a block in the middle of the queue in O(1) time... this class does not allocate any Python objects when manipulating the linked list. Instead, this class manipulates the prev_free_block and next_free_block attributes of the given blocks.」——`touch` 复活要把块从队列中部摘除，deque 做这是 $O(n)$，链表靠块自带的 `prev_free_block` / `next_free_block` 指针（KVCacheBlock L180-181）做 $O(1)$，且全程不新建 Python 对象（减 GC 压力）。`popleft`（L301）/ `remove`（L381）/ `append`（L404）都是指针改写；fake head / tail 哨兵（L287-288）省边界分支。队列序：LRU 在前，同请求释放的块按「hash 链尾在前」排（docstring：more hash tokens 的块先驱逐，保前缀命中链尽量长）。这正是 blog 「常数时间驱逐 + 最小化 Python 对象创建 → 0% 命中吞吐损失 <1%」的代码落点（[vLLM V1 blog](../references.md)，[C]）。

块写满登记 hash：`cache_full_blocks`（block_pool.py L203）为满块算 hash 插入映射。hash 函数 `hash_block_tokens`（kv_cache_utils.py L697）与链式递推 $h_i = H(h_{i-1}, \tau_i, e_i)$ 已在 2.2 §2 摘录并推导（recap + 指引，不重推）；此处补隔离键生成 `generate_block_hash_extra_keys`（L627）：extra_keys = LoRA adapter id + 多模态特征 hash + cache_salt + prompt_embeds digest（2.2 §2 坑注「extra_keys 漏配是正确性风险」的生成侧）。请求级 hash 链由 `request_block_hasher`（L938）逐块算。命中查找 `get_computed_blocks`（kv_cache_manager.py L278）经协调器 `find_longest_cache_hit` 逐组查，有一个细节值得钉（L308-313 注释）：全命中时必须重算最后一个 token 以取 logits，故 `max_cache_hit_length` = prompt_length − 1，又因 `allocate_slots` 要求 num_computed_tokens 块对齐，可能触发整块重算而非单 token——这是命中节省的一个小块级损失（2.2 §2 尾部损失账的同源细节）。

`allocate_slots`（L391）是分配主入口，docstring 给五段布局（comp / new_comp / ext_comp / new / lookahead）与三阶段：先释放跳过块（`remove_skipped_blocks` L575，SWA 窗外）并查自由块够不够（不够返回 None）→ 处理前缀命中块（`allocate_new_computed_blocks` L607）→ 分配新块（`allocate_new_blocks` L616）→ 登记缓存（`cache_blocks` L637，只 cache 已确定 token，draft 可能被拒不 cache）。COW 拷贝对经 `take_kv_cache_block_copies`（L867）取出、随 `SchedulerOutput.kv_cache_block_copies`（output.py L318）交 worker 执行（2.1 §5 的 `_apply_cow` / `take_pending_cow_copies` 机制，single_type_kv_cache_manager.py L585 / L553，recap + 指引）。`free`（L641）**逆序**释放块——尾块先驱逐，保前缀链（docstring 逐字「free the blocks in reverse order so that the tail blocks are evicted first」）。

## 6. spec 数据类族与 hybrid 命中协调

full / SWA / MLA 层的块几何不同，代码怎么表达？答案分两半：**spec 数据类**描述「一层 KV 长什么样」，**manager 类**描述「这类层怎么分配与命中」。两者经 `KVCacheGroup` 绑定。

spec 数据类（kv_cache_interface.py，v1/ 根）：`KVCacheSpec`（L189）是基类，持 block_size 与 `page_size_bytes` / `num_states` / `tokens_per_state` 接口；`KVCacheSpecKind`（L150）枚举九种——FULL_ATTENTION / MLA_ATTENTION / SLIDING_WINDOW / SLIDING_WINDOW_MLA / MAMBA / CHUNKED_LOCAL_ATTENTION / SINK_FULL_ATTENTION / ENCODER_ONLY_ATTENTION / CROSS_ATTENTION（+ UNKNOWN）。`AttentionSpec`（L456）持 num_kv_heads / head_size / head_size_v / dtype / kv_quant_mode / tokens_per_state，页字节公式（L499-514）：

$$
\mathrm{page} = H_{kv} \cdot n_s \cdot (d_K + d_V) \cdot \mathrm{bytes} \qquad (1)
$$

**符号**：page = 一层一个块的字节数（代码 `page_size_bytes`）； $H_{kv}$ = KV head 数（代码 num_kv_heads，或 packing 后的 head slot 数；§4.3 统一记号）； $n_s$ = 块内 state 数（代码 num_states = block_size / tokens_per_state，dense attention tokens_per_state = 1 故 $n_s = b$）； $d_K$ / $d_V$ = K / V 每 head 维（代码 head_size / head_size_v）；bytes = 每元素字节。代码里 state_content_size_bytes = $(d_K + d_V) \times \mathrm{bytes}$（L499-503），page = num_heads × num_states × state_content（L506-514）。式 (1) 就是 2.1 式 (1) 的 $\kappa$ 乘块大小：每 token 每层 $\kappa = H_{kv} (d_K + d_V) \mathrm{bytes}$，page = $b\kappa$。代入章脊 GQA（ $H_{kv}$ = 8、 $d_K = d_V$ = 128、bf16）：state_content = (128+128)×2 = 512 B，page = 8 × 64 × 512 = 262144 B = **256 KiB**/层/块； $\kappa$ = 8×512 = 4096 B（2.1 口径一致）。`MLAAttentionSpec`（L525）令 head_size_v = 0（单 latent、无独立 V，即 $d_V = 0$）；代入 MLA 几何（ $d_{\mathrm{latent}}$ = 512、 $d_{\mathrm{rope}}$ = 64，1.2 / 2.1 口径，缓存维 $d_K$ = 576）： $H_{kv}$ = 1、state_content = 576×2 = 1152 B = $\kappa_{\mathrm{MLA}}$，page = 64×1152 = **72 KiB**/层/块——MLA 块几何由 latent 维定，协调器视其为又一个组，无特殊分支（2.1 §6）。`MambaSpec`（L550）的 page 来自 state shapes（与 block_size 无关），tokens_per_state = −1（整块一个 state）。

manager 类族（single_type_kv_cache_manager.py）：基类 `SingleTypeKVCacheManager`（L42）定义 `remove_skipped_blocks`（L817，按 `get_num_skipped_tokens` 释放窗外块、换 null_block）与 `max_admission_blocks`（L99，每请求准入块帽）等接口；子类按 attention 类型实现——`FullAttentionManager`（L874）、`SlidingWindowManager`（L1068）、`ChunkedLocalAttentionManager`（L1679）、`MambaManager`（L1850）、`CrossAttentionManager`（L2737）、`SinkFullAttentionManager`（L2809），另有 HiSparse 三 manager（Source L2821 / Hot L3018 / Resident L3135，DSA/QSA local KV offload，语义归 5.3）。`_MANAGER_KINDS`（L3314 映射表，L3335 查表）把 manager 类映到 KVCacheSpecKind。`KVCacheGroup`（kv_cache_interface.py L682）= 共享 manager policy + block_size 的一组层（`from_layers` L706 断言同组同 policy 同块大小），组内层共享一张块表；`KVCacheLayerConfig`（L657）= manager_config + spec，`ManagerConfig`（L169）= manager_cls + frozen Params（可 hash、按值比较，故同 policy 层可合组）。

两类 manager 的命中算法方向相反，这是 hybrid 协调的根基：

- **FullAttentionManager.find_longest_cache_hit**（L878）：**从左到右**扫，第一个 miss 即停——链式 hash 保证「前块 miss 则后块必 miss」（源码注释「A missing block implies every later block misses too (chained hashes)」）。命中 = 最长连续满块前缀。边界注：另有 fine-grained Phase 2——当 alignment_tokens < block_size 时，探入首个非满块内部的 hash 边界（从高到低）以命中子块前缀（L931-955；上文机制摘录略此分支）。
- **SlidingWindowManager.find_longest_cache_hit**（L1150）：**从右到左**扫，找最右的连续 cached 块段，长度 ≥ `sliding_window_contiguous_blocks` = $\lceil (w-1)/b \rceil$ 即停（L1180），段右边界即命中前缀终点，段左的块填 null_block（窗外已释放）。道理：SWA 只需最近 $w$ 个 token 的 KV 在缓存，前缀更早的块可丢（2.1 §6 / 官方 hybrid_kv_cache_manager.md Prefix caching Case 1）。边界注：左缘短前缀 fallback——连续段不足需求块数时仍计命中（取该短段，L1223-1235），不要求凑满窗口。

SWA 的驻留块帽由 `max_admission_blocks`（L1099）给： $\lceil (w - 1 + r + f)/b \rceil + 1$（再 cap 到 max_model_len），其中 $r$ = extra_retained_tokens（多模块投机延迟释放）、 $f$ = max_in_flight_tokens（async 在途）。这是 2.1 式 (9) $\lceil w/b \rceil + 1$ 的代码精确形——多出 $r$、 $f$ 两个修正项；稳态 $r = f$ = 0 时， $w$ = 128、 $b$ = 64 给 $\lceil 127/64 \rceil + 1$ = 3 块，与 2.1 式 (9) 一致（recap，不重推）。`get_num_skipped_tokens`（L1310）= 当前窗口起点之前的 token 数，驱动 `remove_skipped_blocks` 逐 chunk 释放窗外块。

**HybridKVCacheCoordinator 的不动点命中**（L704）。多组模型（如 full + SWA）的命中前缀必须**同时**被每组接受：full 组命中 3072，但 SWA 组在 3072 处的窗口块可能已被驱逐。协调器用迭代不动点算法 reconcile（docstring 逐字「Each attention type either accepts the current candidate length or reduces it. If any type reduces the length, restart checks over all types. This converges because length monotonically decreases and is bounded below by 0.」）：

```python
# vllm/v1/core/kv_cache_coordinator.py L743-785（机制摘录，教学用省略组内细节）
hit_length = max_cache_hit_length
while True:
    curr_hit_length = hit_length
    for manager_config, block_size, group_ids in self.attention_groups:
        if issubclass(manager_cls, FullAttentionManager) and cached_blocks is not None:
            continue                       # full 向下封闭：只查一次，后续只截断
        hit_blocks, curr_hit_length = manager_cls.find_longest_cache_hit(
            block_hashes, max_length=curr_hit_length, ...)
    if curr_hit_length >= hit_length:      # 无组再缩 → 不动点
        break
    hit_length = curr_hit_length
    if is_simple_hybrid:                   # 恰 full + X 两组：一轮即够
        break
# 收尾：full 组块截断到 hit_length（向下封闭）
```

**命题 1**（hybrid 命中不动点收敛与一致性）：`find_longest_cache_hit` 的循环必终止；终止时返回的 `hit_length` 是被所有 attention 组同时接受的前缀长，且各组返回的块都覆盖该前缀。

**证明**（路线：终止性用单调有界整数列，一致性用退出条件 + full 向下封闭，两步）。终止性：每轮要么 `curr_hit_length >= hit_length` 退出，要么 `hit_length` 严格减小（某组把候选缩到更小）。对齐前提要分清：初值 $\ell_0$ = max_cache_hit_length = num_tokens − 1（request.py `get_max_cache_hit_length`，末 token 必重算取 logits），**一般不对齐**（如 prompt 4096 → $\ell_0$ = 4095， $a$ = 64）；对齐性只对**各 manager 返回值**成立（`hit_length -= hit_length % alignment_tokens`，single_type_kv_cache_manager.py L954）。故首轮缩量可 $\lt a$（但 $\ge 1$，整数严格减），其后各轮候选已对齐、每轮缩 $\ge a$；非负整数列至多 $\lfloor \ell_0/a \rfloor + 2$ 轮终止（ $\ell_0$ = 初始候选长， $a$ = 对齐粒度；+2 容纳首轮不足 $a$ 的缩量）。一致性：退出那轮没有组把候选缩到 `hit_length` 以下，即每组各自的命中长 ≥ `hit_length`；full attention 组向下封闭（命中前缀的任意子前缀仍命中，链式 hash 保证），收尾把 full 组块截断到 `hit_length`（L789-799）后仍覆盖该前缀；非 full 组在最后一轮已接受 `hit_length`。故各组块都覆盖 `hit_length` 前缀，返回一致。证毕。

下面两例按**通用不动点算法**口径走读（多组逐轮缩长）；full+SWA 恰为 simple hybrid，代码实际走 `is_simple_hybrid` 一轮 break + 收尾截断路径（见段末），轮数口径以此为准。正例（教学 full + SWA， $b$ = 64、 $w$ = 128，需连续 2 块）：请求 $T$ = 4096（64 块），full 组块 0-47 cached（命中 3072）、48+ miss；SWA 组若块 46、47 cached，从右扫即接受 3072，一轮收敛。边界反例（缩长）：SWA 块 46、47 已驱逐、44、45 cached——从右扫 47 miss、46 miss、45 hit、44 hit（连续 2）→ SWA 命中 = 46×64 = **2944** < 3072，通用算法触发第二轮：full 组截断到 2944（块 0-45 ⊆ cached 0-47，仍命中），SWA 重查 2944 接受，收敛于 2944（两轮、缩 128 token = 2 块；simple hybrid 代码路径则一轮 break 后直接截断 full 组到 2944）。`is_simple_hybrid`（恰 full + X 两组、full 在前）时一轮即 break——full 先查给上界、X 缩长后 full 只需截断（向下封闭），不必重查，这是官方文档「full + X」算法（hybrid_kv_cache_manager.md Case 2）的代码优化。

> **坑**：照官方 hybrid_kv_cache_manager.md 写「HybridKVCacheCoordinator 只支持恰两组」。触发条件：引该 doc Implementation 节「Handles exactly two KV cache groups」；现象：doc 自注基于旧 commit 458e74、「things may change」，当前 main 的不动点循环遍历 `attention_groups` 多组、`verify_and_split_kv_cache_groups`（L665）按 manager policy 分组，支持多于两种 attention 类型。规避：算法口径以代码 L704 不动点为准，doc 只作 full+X 两组的特例说明；引用带版本注。

## 7. KV 容量规划：从 profile 到 num_blocks

引擎启动时怎么知道该切多少个块？一条规划链把显存测量变成块数（kv_cache_planning.py + kv_cache_config_builder.py）。worker 侧先 `profile_run`（model_runner.py L1100）跑 dummy 前向测峰值激活，`Worker.determine_available_memory`（gpu_worker.py L724）算出可用于 KV 的字节；core 侧 `get_kv_cache_configs`（kv_cache_planning.py L1456）汇总各 worker 的可用显存与层 spec，产出每 worker 的 `KVCacheConfig`。核心一行（L1566）：

```python
# vllm/v1/core/kv_cache_planning.py L1563-1568（机制摘录）
bytes_per_block = _pool_bytes_per_block_from_builder(builder, vllm_config, projected_groups)
num_blocks = available_memory_one_worker // bytes_per_block
```

`bytes_per_block` = 一个块号在所有 device 组、所有层上占的总字节（= 各组 page_size 之和，组 page = 组内层 page 之和，KVCacheGroup.page_size_bytes L742）。这就是 2.1 式 (1) 容量账的离散化：

$$
n_{\mathrm{blk}} = \left\lfloor \frac{M_{\mathrm{avail}}}{b \sum_{\ell} \kappa_{\ell}} \right\rfloor \qquad (2)
$$

**符号**： $n_{\mathrm{blk}}$ = 块数（代码 num_blocks）； $M_{\mathrm{avail}}$ = profile 后可用 KV 显存字节； $b$ = block size； $\kappa_\ell$ = 第 $\ell$ 层每 token KV 字节（式 (1) 的 state_content × $H_{kv}$）；分母 = 一个块全层字节。代入章脊（8B、 $b$ = 64、32 层均 GQA $\kappa$ = 4096 B）：分母 = 64 × 32 × 4096 = 8 MiB， $M_{\mathrm{avail}}$ = 48 GiB → num_blocks = **6144**（2.1 §4 同池），token 容量 = 6144 × 64 = 393216 ≈ 384K。max_model_len = 131072 的单请求需 $\lceil 131072/64 \rceil$ = 2048 块，满窗并发 = $\lfloor 6144/2048 \rfloor$ = **3** 条——这是 9.1 容量规划的引擎侧入口（`get_max_concurrency_for_kv_cache_config` kv_cache_utils.py L1051 逐组求每请求块数再除池容量）。

规划链还有四件配套：`check_enough_kv_cache_memory`（L195）检查可用显存至少装得下一条 max_model_len 请求，不够报错并给 hint；`estimate_max_model_len`（L141）二分搜索装得下的最大模型长；`_auto_fit_max_model_len`（L1282）在用户设 max_model_len = −1 时自动定长；多 worker 取 **min num_blocks**（L1578）并按比例缩 tensor（`shrink_kv_cache_configs` L1738），保证集中式调度器的块号在所有 worker 一致（PP 各 stage 层数不同、可用显存不同，取最紧者）。hybrid 模型的 page size 统一由 `unify_hybrid_kv_cache_specs`（L834）/ `get_uniform_page_size`（L324）做：单池只容一种 page size，不同 $\kappa$ 的层靠调 block_size 或 pad 对齐（官方 hybrid_kv_cache_manager.md Allocation 五 case：toy / 规则比例 / 非规则比例 / 不同 kv_hidden_size（Mamba）/ KV sharing）。Gemma-3-27b 例（doc Case 3）：52 SWA + 10 full，按最小组大小 min(52,10) = 10 层/组 → 1 full 组 + 6 SWA 组（末组 2 层 + 8 padding 层），共 7 组。builder 接口 `KVCacheConfigBuilder`（kv_cache_config_builder.py L19）把「层怎么分组」（get_kv_cache_groups L37）与「组怎么落成 tensor」（get_kv_cache_config_from_groups L55）交给平台 hook（get_kv_cache_config_builder L86），默认 `DefaultKVCacheConfigBuilder`（planning L1677）。

## 8. encoder cache 与 async 调度

多模态请求的 vision embedding 缓存谁管？`EncoderCacheManager`（encoder_cache_manager.py L18），独立于 KV 块池，容量以 **embedding 数**计（cache_size / num_free_slots，L78-79）。它按 mm_hash（多模态输入特征 hash）跨请求共享 embedding：`check_and_update_cache`（L111）查命中并登记引用；`can_allocate`（L148）查容量，不够时从 `freeable`（无请求引用的旧条目，OrderedDict FIFO）**分配时驱逐**（L203-208，最旧优先）；`allocate`（L211）扣 num_free_slots。调度器侧 `_try_schedule_encoder_inputs`（scheduler.py L2111）在 `encoder_compute_budget` 内安排 encoder 计算，预算或缓存不足的请求本步 num_new_tokens 被缩到该 encoder 输入之前（docstring：adjusts num_new_tokens to schedule only the decoder tokens up to just before the unschedulable encoder input）。抢占连带释放 encoder cache（`_preempt_request` L1850）。预算由 `compute_mm_encoder_budget`（L329）按模型与 scheduler 配置算。这是 blog 第 7 点「encoder cache 使 MLLM 可 chunked prefill」的代码落点——V0 图像与文本必须同步处理（vision embedding 步后即弃），V1 缓存 embedding 让文本可分 chunk 跨步（[vLLM V1 blog](../references.md)，[C]）。多模态 serving 全账（VE 分离、encoder 算力配比）归 9.4，本章只登记调度接口（4.2 §6 已给容量门机制，recap + 指引）。

最后回到 §1 的 CPU 开销税：调度循环本身是纯 CPU，步时越短占比越高，vLLM 怎么藏？`AsyncScheduler`（async_scheduler.py L10）继承 Scheduler，调度侧 override 是 `_update_after_schedule`（L18；另 override `__init__` 设 PP decode_stagger = pipeline_parallel_size，L11-16，PP decode microbatch 节奏归 8.1）：调度完不等输出，立刻给每个非 prefill 请求加 `num_output_placeholders += num_sampled_tokens_per_step + cur_num_spec_tokens`（L45-47），并把 spec_token_ids 设为 `[-1] × n` 占位（L51）。官方 arch_overview.md Async Scheduling 节逐字：「while the GPU executes step N, the CPU schedules step N+1, using placeholders for the tokens that step N will produce.」placeholder 是**计数占位**（告诉调度器「这一步将产出几个 token」），真实 token id 由 worker 产出、`update_from_output` 回填——与 SGLang FutureMap 中继真实 token 值的形态不同（6.2 详展）。配合 `step_with_batch_queue`（core.py L759，5.1 §3）的 batch queue，CPU schedule(N+1) 与 GPU forward(N) 重叠。

**代码 ↔ CPU 开销对照**（承接 5.1 式 (1)，完整 bubble 消除账与正确性证明归 6.2）：同步循环步时 = $t_{\mathrm{cpu}} + t_{\mathrm{gpu}}$（串行），async 把 schedule(N+1) 藏进 forward(N)，完全隐藏条件 = CPU 段 ≤ GPU 段。代入 5.1 教学口径 $t_{\mathrm{cpu}}$ = 2 ms、 $t_{\mathrm{gpu}}$ = 5 ms：同步 7 ms/步，async 理想 5 ms/步，**1.4×**——这是 5.1 式 (1) GPU 占比 71% → 100% 的步时读数。vLLM 的 placeholder 计数 + 乐观推进（§2 `_update_after_schedule`）+ stale output drain（§4）是这套重叠的正确性要件；SGLang 侧用 FutureMap stash/publish/resolve + WAR barrier + stream order 达成同一目标，两者的 bubble 账与 future-token resolve 正确性证明在 6.2 强制重推导，此处只对 vLLM 代码形态、不重推。DeepInfer 同源（官方 doc：Following SGLang, NanoFlow，7.1 回链）。

## 总结

| 论断 | 回指 |
| --- | --- |
| 两层纪律：2.1 机制级（块表 / 碎片上界 / COW / watermark）recap + 指引，本章模块全量走读；interface/layout 在 v1/ 根、FullAttentionSpec/SlidingWindowSpec 不存在、watermarking ≠ KV watermark | §1、坑注 |
| schedule() 统一 token 视图：num_new_tokens = num_tokens_with_spec + placeholders − num_computed；三道钳位（threshold / token_budget / input_budget−draft_slots）= 4.3 chunk 上界的代码形 | §2、4.1 §5 / 4.3 §5 |
| 预算 2048、64 decode + 8192 prompt → chunk 1984、5 步；本步有抢占就不准入（L1041）；_update_after_schedule 乐观推进计数器 | §2 |
| FCFS = deque（prepend 插队头）/ PRIORITY = heap（prepend 按序重插）；抢占恢复落点随策略异 | §3、4.2 §1 |
| V1 抢占 = recompute-only（num_computed_tokens=0 + 回 waiting，无 swap）；恢复成本 = 4.2 式 (1) − 2.2 式 (2)，T=4096/δ=3072 例 133→34.9 ms；async stale output 交付但不改计数器 | §4、命题场景 |
| watermark 只约束 WAITING/PREEMPTED 且 has_scheduled_reqs；默认 0.0；full_sequence_must_fit 防 chunked 过度准入；触发链 = allocate_slots 返回 None | §4、2.1 §4 / 4.2 §2 |
| KV 三层：KVCacheManager（接口）→ Coordinator（Unitary/Hybrid/NoPrefixCache）→ SingleTypeManager × 组；全组共享一个 BlockPool | §5 |
| FreeKVCacheBlockQueue = 零分配双向链表，O(1) 中部摘除 → blog zero-overhead prefix caching（0% 命中 <1%）；free 逆序保前缀链 | §5、2.1 §4 |
| page = num_heads × num_states × (head_size+head_size_v) × bytes = bκ；GQA 256 KiB / MLA 72 KiB 每层每块；spec 数据类 + manager 类族经 KVCacheGroup 绑定 | 式 (1)、§6 |
| full 命中左→右（链 hash 一 miss 全 miss）/ SWA 右→左（连续 ⌈(w−1)/b⌉ 块）；SWA 驻留帽 = 2.1 式 (9) 加 in-flight/extra 修正 | §6、2.1 §6 |
| hybrid 命中不动点：单调减整数列必终止，退出时各组接受同一前缀（full 向下封闭截断）；full 3072 / SWA 缩 2944 两轮例 | 命题 1、§6 |
| num_blocks = ⌊M_avail / (b Σκ)⌋；48 GiB / 8 MiB = 6144 块、满窗并发 3；多 worker 取 min；profile_run → determine_available_memory → get_kv_cache_configs | 式 (2)、§7、9.1 |
| EncoderCacheManager 按 embedding 数计容量、mm_hash 跨请求共享、分配时 FIFO 驱逐；使 MLLM 可 chunked prefill | §8、9.4 |
| AsyncScheduler placeholder 计数 + 乐观推进 + batch queue 重叠 CPU/GPU；2+5=7→5 ms（1.4×）；完整 bubble 账与 resolve 证明归 6.2 | §8、5.1 式 (1) |

## 练习

1.（计算 · [10]）token 预算分配（章脊：预算 2048、 $b$ = 64）：(a) 批内 32 条 decode + 一条 4096 prompt 新请求，本步给新请求多少 token？(b) 该 prompt 共需几步算完（每步 decode 数不变）？(c) 若 long_prefill_token_threshold = 1024，(a) 的答案变成多少？校验：(a) 2048−32 = 2016；(b) $\lceil 4096/2016 \rceil$ = 3 步（2016+2016+64）；(c) min(2016, 1024) = 1024，与 §2 钳位链一致。
2.（计算 · [20]）容量规划账（式 (2)，章脊 8B、 $b$ = 64、 $M_{\mathrm{avail}}$ = 48 GiB）：(a) 每块全层字节与 num_blocks？(b) max_model_len = 32768 时单请求块数与满窗并发？(c) 若换 MLA 几何（ $\kappa$ = 1152 B/token/层、40 层），num_blocks 变成多少？先预测 (c) 相对 (a) 的方向再算。校验：(a) 8 MiB、6144；(b) 512 块、12 条；(c) 分母 = 64×40×1152 ≈ 2.81 MiB，num_blocks ≈ 17476，比 (a) 大（MLA κ 小 → 同显存切更多块），与 1.2 MLA 带宽账方向一致。
3.（代码 · [30]）读 `vllm/v1/core/sched/scheduler.py` 的抢占循环（L898-969）与 `_preempt_request`（L1831-1877，main 2026-11-23）：先写出预测——(a) FCFS 下 victim 是哪条、释放后 num_computed_tokens / status / 队列位置各变成什么？(b) 被抢请求恢复时为什么可能比全量 recompute 便宜？(c) async 下被抢请求的在途输出为什么仍交付而不丢？再对照源码验证，并指出 watermark 在哪一步生效、是不是抢占触发器。校验：(a) running[-1]、num_computed_tokens=0、PREEMPTED、waiting 队头（L1854/1853/1876）；(b) 释放块带 hash 仍是驱逐候选，get_computed_blocks 命中前缀免算（§4，4.2 式 (1) 是上界）；(c) L1857-1867 注释「dropping them would perturb spec-decode acceptance」；watermark 在 allocate_slots L529-536 准入侧，触发器是 allocate_slots 返回 None。
4.（证明 · [30]）命题 1 的终止界与一致性：(a) 证明不动点循环至多 $\lfloor \ell_0/a \rfloor + 2$ 轮终止（ $\ell_0$ 初始候选长、 $a$ 对齐粒度），指出每轮 hit_length 的变化性质（提示：初值 num_tokens − 1 一般不对齐，首轮缩量可 $\lt a$）；(b) 证明退出时各组返回块都覆盖 hit_length 前缀，指明 full attention 的向下封闭性在哪一步用到；(c) 构造一个三轮才收敛的 full + SWA + chunked-local 三组例（给各组 cached 块分布）；(d) 说明 is_simple_hybrid 为什么能把 full+X 两组压到一轮。校验：(a) 初值一般不对齐、首轮减 ≥ 1、后续每轮减 ≥ $a$ 或退出（对齐只对 manager 返回值成立，L954）；(b) 退出条件 curr ≥ hit 保证无组再缩 + L789-799 full 截断；(c) 每轮至少一组缩长、第三轮无组缩；(d) full 先查给上界、X 缩长后 full 向下封闭只需截断（§6 边界反例的两轮在 full+X 下压成一轮）。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——分页 / 抢占 / prefix caching 论文锚点
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——`v1/core/sched/{scheduler,async_scheduler,request_queue,output,interface}.py`、`v1/core/{block_pool,kv_cache_utils,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager,kv_cache_planning,kv_cache_config_builder,encoder_cache_manager}.py`、`v1/{kv_cache_interface,kv_cache_layout}.py`、`v1/worker/gpu/model_runner.py`、`v1/worker/gpu_worker.py`、`config/{scheduler,cache}.py`、`docs/design/{arch_overview,hybrid_kv_cache_manager,prefix_caching}.md`（均 2026-11-23 main tarball file:line 核实）
- [vLLM V1 blog](../references.md)：vLLM V1: A Major Upgrade to vLLM's Core Architecture，vLLM Team，2025-01-27，[C]（key: vllm-v1-blog，开放技术博客、非闭源）——zero-overhead prefix caching（常数时间驱逐 + 减 Python 对象、0% 命中 <1%）、encoder cache 使 MLLM 可 chunked prefill、Persistent Batch
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露）——bf16 dense 989.5 TFLOPS（ $F_{\mathrm{eff}}$ 口径）
- 跨维度指引（纯文字）：inference 1.2（κ 几何表 / MLA 带宽账）、2.1（块表 / 碎片上界 式 (4) / COW 命题 2 / watermark / SWA 驻留 式 (9)，机制级）、2.2（hash 链 式 (1) / 命中节省 式 (2)）、4.1（迭代级调度 / 统一 token 视图 / 步时 式 (3)）、4.2（抢占成本 式 (1)(2)(3) / 策略 / watermark 触发链）、4.3（chunk 钳位 / iteration 模型 式 (2) / token vs 时间预算）、5.1（V1 架构 / busy loop / diff 协议 命题 1 / CPU 开销 式 (1)）、5.3（HiSparse 语义 / MessageQueue / 分布式）、5.4（结构化输出 / 投机 / 多模态）、6.2（overlap bubble 消除账 + future-token resolve 正确性证明，强制重推导）、7.1（DeepInfer async 同源）、9.1（容量规划）、9.4（多模态 serving 全账）