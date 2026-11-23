# W3 P6 审查报告 — SGLang 走读（6.1 / 6.2【重】/ 6.3）

日期：2026-11-23。evaluator：fresh-context 对抗审查（不继承写作上下文）。
对象：`inference/06-sglang/{01-sglang-architecture,02-sglang-overlap-scheduler,03-sglang-distributed-pd}.md`。
依据：evaluator-brief（W1）+ W3 dispatch；spec §5 P6 / §4.1-4.3 / §6.3；STYLE_GUIDE §3/§4/§7/§8/§9/§11；R2 style sheet 12 条 + 习题分级；rubric A/B/C；`inference/references.md`；`docs/research/inference-w3-sources.md`。

## 0. 方法与独立性

- **一手代码**：evaluator 自取 codeload fresh tarball（sgl-project/sglang main，2026-11-23 10:53，47,805,118 B，解压 /tmp/iw3-eval-p6/fresh/sglang-main）。与 research pass 同日 tarball（/tmp/iw3-evidence/sglang-x，09:11）对 scheduler.py / overlap_utils.py / radix_cache.py / base/conn.py / prefill.py / decode.py 逐文件 md5 比对**全部 SAME**——研究快照为真当日 main，锚点核查在 evaluator 自取副本上执行。
- vLLM 对照锚点与 docs/design/arch_overview.md 引文核于 research pass 同日 vllm tarball（/tmp/iw3-evidence/vllm-x，09:10，53,534,089 B，codeload 一手Artifact）。
- blog 引文：lmsys.org v0.4 blog **evaluator 重新抓取**（/tmp/iw3-eval-p6/web/sglang-v04-blog.html，64,211 B，title 与关键句命中）+ research 存档 txt 逐句对照；论文引文核于 W1 arXiv 2312.07104v2 全文存档；DeepInfer doc L334 核于 W1 同日 tarball 全文。
- 数值：全部例题 python3 独立重算（见 §4）；lint 三章重跑；CJK 独立计数；禁令词 + rubric A 扩展词表 + 语气词全扫。

## 1. Findings 表

**S1 = 0；S2 = 3；S3 = 5；另有 out-of-batch errata 1 条（3.4 指针）。**

| # | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S2-1 | S2 | 6.2 §7（L216）+ 总结表 L306 | PrefillDelayer 触发语义写反 + 机制账错：章写「触发条件 = KV token usage 低于 token_usage_low_watermark（池快满）或可选 queue_min_ratio」，机制账讲「池快满放 prefill 挤占 decode KV headroom → retract thrashing，延迟 prefill 换 retract 成本」 | prefill_delayer.py L177-181：`local_token_watermark_force_allow = local_prefillable and (token_usage < x)`；L230-241 注释逐字「Safety valve: low KV usage means GPU is underutilized, skip delay」——**低水位是强制放行（跳过延迟）条件，不是延迟触发**；「usage 低」= 池快空，章括注「池快满」自相矛盾。实际延迟触发（L275-280）：slot_condition = `max_running_requests − global_running_batch_max < global_max_prefill_bs_max`（注释 L281-282：首批 merge 会让 decode 达不到最大 batch size）+ opt-in queue_condition = waiting 队列 < min(running×ratio, max_prefill_bs)（L252-271，攒大 prefill 批、防 decode 逐条完成把 prefill 打碎）。机制 = batch size 效率 / GPU 利用率，不是 KV/retract | 重写触发描述：低水位 = force-allow 安全阀；延迟触发 = slot 条件（running 逼近 max_running_requests 相对最近 prefill 高水位）+ 可选 queue 条件（waiting 不足 queue_min）；机制账改「延迟 prefill 让 decode 凑满批 / prefill 攒大批，TTFT 换批效率」；总结表行同步。研究笔记 ⟨verify⟩ 销项「KV token usage 低水位触发」措辞误导，以代码为准（笔记更正登记） |
| S2-2 | S2 | 6.2 §2 设计点其一（L52）+ 命题 1 证明保持步（L130） | 「等 CPU 走到 pop_and_process 时 GPU 早已跑完第 j−1 批——一拍的松弛把同步等待压到近零」在 GPU-bound 稳态（章自己的算例 c_cpu=1 < t_gpu=5.13）不成立；证明保持步「间隔总 CPU 时长 = c_p + c_s = c_cpu」默认 D2H 已就绪，缺同步等待 pacing 步 | process_batch_result_decode（batch_result_processor.py L1151）在 L1157 `result.copy_done.synchronize()`；copy_done 在 copy_stream（wait forward_stream，scheduler.py L5914）完成后才就绪 = GPU 跑完批 j−1 之后。GPU-bound 稳态：launch(j) 发生在 GPU 完成 j−1 之前约 t_gpu − c_cpu（CPU 被上一拍 sync pacing），CPU 到 pop_and_process(j−1) 时 GPU 尚未跑完 j−1，**sync 等待 ≈ t_gpu − c_cpu ≈ 4.13 ms**，正是把 CPU 钉在 GPU 节奏上的机制；「等待近零」仅在 CPU-bound（c_cpu ≥ t_gpu）成立。命题 1 的式 (2)(3) 与条件本身正确（独立重推导见 §3），但证明需补：launch(j+1) = max(launch(j), S_j + ε) + c_cpu（S_j = GPU 开跑批 j），不变式 launch(j) ≤ S_j，故 c_cpu ≤ t_gpu 时 launch(j+1) ≤ S_j + t_gpu = S_{j+1}；CPU-bound 时 launch 间隔 = c_cpu > t_gpu → GPU 每拍空 c_cpu − t_gpu | §2 改「D2H 上一拍已发出，sync 等待至多是批 j−1 的 GPU 残余；GPU-bound 稳态该等待 ≈ t_gpu − c_cpu，把 CPU pacing 到 GPU——零开销指 GPU 零 idle，不是 CPU 零等待」；证明保持步补 max/pacing 一句（结论不变）。顺带在 §5 基例 2 点明 HiSparse staging stash（L4349）提交在 schedule_stream，由 run_batch L5516 wait_stream 定序到 gather 之前（现证明默认同流序，该基例是跨流序） |
| S2-3 | S2 | 6.1 §4（L161）+ 练习 4(b) 校验锚（L278） | free_segment 语义误读：章写「新请求与缓存重复的那段 KV 槽位立即释放，树里只留一份」；练习 4(b) 校验锚「归还量 = R2 与缓存重复的前缀长度」（R2 命中 3 → 暗示答案 3） | radix_cache.py L509-512：`free_segment(kv_indices[req.kv.cache_protected_len : result.prefix_len])`。cache_protected_len = 匹配时树已拥有的前缀（mem_cache/common.py L231 `req.kv.protect_prefix(match_result.cache_protected_len)`；schedule_batch.py L282-284 注释「The request's own KV is [cache_protected_len, kv_allocated_len) … Tree cache owns [0, here) (matched or inserted)」；common.py L420-421「The protected prefix is not this req's to free」）。匹配前缀在 req_to_token 行里是**树自有索引的借用**，不是请求新算的重复 KV，释放它们会腐蚀树。该 free 段非空仅当请求自算 KV 与**执行期间新入树**内容重复（prefix_len > protected，如并发同前缀竞态——恰 6.2 §6 in-batch dedup 场景）。练习场景 R2：protected = prefix_len = 3 → free 段为空，**正确答案 0** | §4 改「free_segment 释放的是『请求自算 KV 与执行期间新入树内容重复』的段 [cache_protected_len, prefix_len)；匹配前缀是树自有索引（match 时 protect_prefix），不在释放之列」；练习 4(b) 校验锚改「归还量 = max(0, prefix_len − cache_protected_len)；本题 R2 两者相等（均 3），归还 0——命中段是借用树索引，无重复可还」 |
| S3-1 | S3 | 6.2 §4 表（L138-141）+ 练习 1 校验锚（L313） | 临界行 T_normal = 10.25 应为 10.26（5.13 + 5.13）；F = 1.098（c=0.5 行）与声明精度 5.13 不符（5.63/5.13 = 1.0975 → 1.097），1.098 对应未舍入 t_gpu ≈ 5.1265（4.1 式 (3) 精确值 17,173,741,824/3.35e12 = 5.1265 ms） | python 重算：c=0.5/1/3/8 各行 F = 1.097/1.195/1.585/1.641（@5.13）；1.098/1.195/1.585/1.641（@5.1265） | 表与校验锚统一精度：要么声明 t_gpu 取未舍入 5.1265（10.26 仍须改），要么改 1.097/10.26 |
| S3-2 | S3 | 6.3 §1（L19）+ 命题 1 边界反例（L29） | 断言清单缺 c=1 情形的整除保证：章只引 cp gate（attn_cp_size>1 时 L137-141）与 MoE gate；命题 1「断言保证 t = d·c·a」与反例「t=8、d=3 …直接拒绝启动」在 c=1 时依据的是未引用的另一处断言 | parallel_hook.py L377 `if attn_dp_enabled_of(...)`（= attn_dp_size>1 or scale joiner，runtime_context.py L199）分支内 **L389 `assert cfg.tp_size % cfg.attn_dp_size == 0`**——t=8、d=3、c=1 由 L389 拒绝（cp gate 不触发）；elastic 路径另有 L650-659 divisor 校验。反例结论本身为真，但章引断言集覆盖不到它 | §1 补一句：attn_dp_size>1 时 parallel_hook.py L389 断言 tp_size % attn_dp_size == 0（c=1 情形的整除保证）；命题 1 反例点名 L389 |
| S3-3 | S3 | 6.2 §2 其三（L56）、§5 要件表（L165）、参考（L321） | `_advance_pending_grammar` 锚 L2361 落在 docstring 内，def 实在 L2359 | scheduler.py L2359 `def _advance_pending_grammar(self):`（L2360-2365 docstring）；研究笔记同误，章继承 | 三处 L2361 → L2359 |
| S3-4 | S3 | 6.3 §7（L173） | 「模块 docstring 逐字『Checkpoint-engine integration for SGLang…』」未指明文件；checkpoint_engine/__init__.py 的 docstring 是另一段（「Checkpoint engine module for SGLang…」），引文实在 checkpoint_engine_worker.py L15-16 | checkpoint_engine_worker.py L15-16 逐字命中；__init__.py L1-5 不同文 | 引文锚点名 checkpoint_engine_worker.py L15-16 |
| S3-5 | S3 | 6.2 §7（L218-230）；6.3 §1（L19） | 符号局部声明与全书约定的张力：6.2 §7 用 B = backlog token 数（waiting 队列，非 batch）、C = chunk size，而 §4.3 定 B 专归 batch、4.3/6.3 用 c_chunk（6.2 §6 自己也把 rem_chunk_tokens 对到「4.3 c_chunk」）；6.3 用 t = tp_size（world size）而 §4.3 t = TP 度 | 两处均有章内局部声明且沿代码 docstring 原记号（shortest_prefill_load.py L4「The backlog B…」），属可辩护的代码忠实记号，但跨章不一致 | 建议 6.2 §7 加一句消歧（「此 B 为代码 docstring 原记号 = backlog，非批内 B；C 即 4.3 c_chunk 的 per-rank 值」）或改用 B_wait/c_chunk；6.3 t 的语义迁移已显式声明，可保留 |
| ERR-1 | errata（out-of-batch） | inference/03-inference-operators/04-fusion-and-graph-capture.md L7、L99 | 3.4 两处指针「引擎编译模块全量走读归 5.3 / 6.2」「SGLang 归 6.2」与 spec 6.3 大纲（compilation 在 6.3）及已交付 6.3 §8 不一致 | spec §5 6.3 内容行含「compilation（torch compile、CUDA Graph ↔ 3.4）」；6.2 大纲无 compilation；6.3 §8 已交付引擎调用级走读 | W2 章走 errata 通道：L7「5.3 / 6.2」→「5.3 / 6.3」、L99「SGLang 归 6.2」→「SGLang 归 6.3」（writer D concern ④ 确认成立） |

## 2. Writer concerns 裁定（独立核查，不照单全收）

| concern | 裁定 | 证据 |
| --- | --- | --- |
| ① PREBUILT「never enters model forward」vs 笔记「PREBUILT批解码」 | **章正确**。PREBUILT 是 metadata 态：首 token 已在 prefill 侧采样、id/logprobs 经 aux 传递，decode 首次真实 forward 是第二 token 的 DECODE 步 | decode_schedule_batch_mixin.py L24 prepare_for_prebuilt、L32 注释逐字「PREBUILT never enters a model forward」；utils.py MetadataBuffers L688-690 注释逐字 + output_ids (size,16) int32；run_batch L5468-5474 prebuilt 分支（内层 idle 批为 DP collective 陪跑，不违反 PREBUILT 本身不进 forward）。笔记措辞易误读，章口径成立 |
| ② process_prefill_chunk 位置 | **章正确**。在 get_next_disagg_prefill_batch_to_run 内调用（call L839，def L1661），非 process_batch_result 直调；章 §4 描述与代码一致 | prefill.py L818 def get_next_disagg_prefill_batch_to_run、L839-843 调 process_prefill_chunk（带 prefill_chunk_processed 防重入）、L856 event_loop 顺序 ingest → move_bootstrapped → get_next… → run_batch → process_batch_result → process_disagg_prefill_transfer_queue 全对 |
| ③ cp/MoE 断言 gate | **章正确**（attn_cp_size>1 / moe_dp_size>1 gate 逐字核实）；补 L389 见 S3-2 | parallel_hook.py L135-161；8.1 建议同口径成立 |
| ④ weight_updater 全路径 | **章正确** | managers/scheduler_components/weight_updater.py L131 / L164 / L179 / L182 / L210 / L215 全命中 |
| ⑤ 3.4「SGLang编译归6.2」指针 | **不一致确认**，章从 spec（6.3 §8 交付 compilation）正确；3.4 留 errata（ERR-1） | 3.4 L7/L99 原文；spec §5 6.3 大纲 |
| ⑥ c_cpu=1ms 教学假设 vs blog 1.1× | **口径分离合格**：§1 显式「声明：非实测，量级参照 blog『至多一半』口径的下沿」；§4 blog 1.1×/1.3× 挂 [C] 实测校准，两不混 | 6.2 L19、L144；blog L14 逐字 |
| ⑦ jump-forward inkling2 stub 两口径 | **章正确**：论文 compressed FSM 机制 + 当前 XGrammar 默认 mask 路线两口径分开，坑注到位 | inkling2_grammar.py L229 try_jump_forward return None、L232-233 jump_and_retokenize raise RuntimeError；xgrammar_backend.py 无 jump 符号（grep 确认） |
| W3-C 自纠：_split_node 不调 _update_leaf_status | **自纠正确**。_split_node（L684-707）无 _update_leaf_status 调用；new_node.lock_ref = child.lock_ref（L691）继承；child 的 lock_ref/children 不变；命题 1 证明按源码成立（详 §3.3） | radix_cache.py L684-707 全文核 |

## 3. 强制重推导（6.2，独立重推）

### 3.1 bubble 消除账 —— 账 PASS，书面证明有 S2-2 缺口

独立建模：CPU 每拍工作 = ingest + sched(j) + post(j−1) + launch，总量 c_cpu = c_s + c_p；GPU 每拍 = forward(j)+sample(j) = t_gpu。launch 非阻塞、D2H 上一拍发出（copy_stream wait forward_stream，scheduler.py L5914）、result_queue 稳态深度 1。设 S_j = GPU 开跑批 j 时刻，L_j = launch(j) 时刻。CPU 在 pop_and_process(j−1) 处 sync copy_done（batch_result_processor.py L1157），就绪时刻 = S_j + ε（GPU 跑完 j−1 + 拷贝）。故 L_{j+1} = max(L_j + δ, S_j + ε) + c_cpu ≈ max(L_j, S_j) + c_cpu。

- 不变式 L_j ≤ S_j（launch 先于 GPU 开跑）：基例由流水填充建立；保持 L_{j+1} = max(L_j, S_j) + c_cpu = S_j + c_cpu ≤ S_j + t_gpu = S_{j+1}（当 c_cpu ≤ t_gpu）。GPU kernel 队列不空 → 零 idle，每拍 = t_gpu。
- c_cpu > t_gpu：GPU 跑完等 launch，每拍 = c_cpu，GPU 每拍空 c_cpu − t_gpu。
- 合并 T_overlap = max(c_cpu, t_gpu)（式 (2) ✓）；T_normal = c_cpu + t_gpu（式 (1) ✓）；F = (c_cpu+t_gpu)/max（式 (3) ✓）；完全隐藏时 F = 1 + c_cpu/t_gpu ≤ 2，等号 c_cpu = t_gpu ✓。上界 2 是结构结论 ✓。

**结论**：公式、条件、数值表全部正确。但章写证明的保持步把 launch 间隔无条件写成 c_cpu 并断言「D2H 已就绪」——GPU-bound 稳态下间隔实为 t_gpu（sync 等待 pacing），§2「同步等待压到近零 / GPU 早已跑完 j−1」在该 regime 为假（章自己的算例即 GPU-bound）。结论可救、证明需补 max/pacing 步 → **S2-2**。

### 3.2 future-token resolve 正确性（命题 2）—— PASS

按当前 FutureMap stash/publish/resolve 语义独立核验（非 blog placeholder）：

1. **同流提交序**：gather 在 run_batch 的 `with self.forward_stream_ctx:` 内（L5515）、forward_stream.wait_stream(schedule_stream)（L5516）之后调 resolve_forward_inputs（L5525 → overlap_utils.py L127，decode 分支 L165-166 `batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]`）。stash 端：非投机路径 _relay_forward_payload（L5578 调用，def L5835）同在 forward_stream_ctx 内、forward_batch_generation（L5553，含 sample launch）之后经 relay（L5860）→ _stash scatter（L800）；grammar 延迟 sample 路径 launch_batch_sample_if_needed（L5928）亦在 forward_stream_ctx 内（L5936）先 delay_sample_func() 再 _relay_forward_payload（L5941）——两路径 stash 均在 forward_stream 上排在 sample kernel 之后。beat j−1 的 Python 提交先于 beat j → 提交序 stash(j−1) ≺ gather(j) → 同流执行序亦然。gather(j) 读 stash(j−1) 写入值 ✓。
2. **基例**：prefill 后首 decode 拍——prefill sample 首 token 经 RelayPayload(bonus_tokens=next_token_ids)（L5850）relay stash ✓；HiSparse staging——_build_hisparse_decode_batch L4349-4351 stash CPU 已确认 output_ids[-1]（该 stash 在 schedule_stream 上，由 L5516 wait_stream 定序到 gather 前——章证明未点明跨流序，见 S2-2 附带修正）。
3. **WAR barrier**：_apply_war_barrier L2207-2218 逐字核（shared_read_done_event 细粒度 / wait_stream 粗粒度兜底 / SGLANG_FORCE_COARSE_WAR_BARRIER），fence 的是 schedule_stream 写（KV 释放/重分配、req_to_token 更新）对 forward 共享读的 WAR，与 token relay 分工不同 ✓ 章描述准确。
4. **slot 0 镜像**：FutureMap L324-325 注释 + ReqToTokenPool 行 0（memory_pool.py L417-419 注释逐字）+ KV 池 slot 0（allocator/token.py L43-45 注释逐字）三处同构 ✓。
5. **poison CI-only**：_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()（L74，默认 False）；生产 torch.empty（L345-350）；_assert_nonneg_and_invalidate L92 fused assert+scatter −1 ✓ 章坑注准确。
6. **grammar CPU 顺序**：launch_batch_sample_if_needed 在 pop_and_process 之后（L2329 注释逐字、L2331）；should_process_last_batch_early L2338（TTFT env + DP is_extend_in_batch 同步 L2346-2354 / grammar_needs_sync L2357）；_advance_pending_grammar def L2359（章锚 L2361 → S3-3）✓。
7. **投机 v2 扩展**：publish L749 写 new_seq_lens_buf + spec_algo.is_some() 时 record publish_ready（L769-773）；resolve_seq_lens_cpu L687 私有 fwd_prepare_d2h_stream wait_event → pinned 拷贝 → synchronize（L736-739），L734-735 注释逐字「don't sync the schedule stream; gate a private stream on the publish event」✓；resolve_mixed_spec_tails L642 publish 栅栏后 late-bind ✓。

命题 2 陈述（条件式）与证明（不变式 + 流序归纳 + 前缀不可变收束）在当前语义下**成立**。四要件表与三个边界反例均与代码对应。

### 3.3 其余命题复核

- **6.1 命题 1（锁保护）PASS**：_update_leaf_status L797-810 入集条件 = 非 evicted ∧ lock_ref==0 ∧ 所有孩子 evicted ✓；inc_lock_ref L589-602 沿路径 +1、0→1 转 evictable→protected、逐节点调 _update_leaf_status ✓；dec 对称 ✓；_split_node L684-707 继承 lock_ref、不调 _update_leaf_status、child 状态不变（writer 自纠正确）✓；_insert_helper 新叶 L762-763 对父与新节点调 ✓；evict L559-587 只从 evictable_leaves 快照堆弹、父入堆显式查 lock_ref==0（L580）✓；单线程调度下 evict 执行期间无锁变更，快照论证成立。正例/边界反例（只锁叶不锁路径 → 祖先悬空）合理。
- **6.3 命题 1（rank 双射）PASS（补 S3-2）**：混合进制分解唯一性论证正确；derive_attention_ranks L79 返回 (tp_rank % a, tp_rank // (a·c))——章「源码返回式正是第一、第三两项」准确；t = d·c·a 的保证在 c>1 走 L140、c=1 走 L389（章未引，S3-2）；反例 t=8 d=3 拒绝启动为真（L389）。
- **6.3 命题 2（chunk 恰好覆盖）PASS**：s_0 = decode_prefix_len（prefill.py L546）；每次发送后 start_send_idx = end_idx（L2084）；非末 chunk clamp_kv_send_end（L148，min(end_idx, cache_protected_len) 只向下夹）+ 页对齐推迟（L1799 `end_idx = end_idx - end_idx % page_size`）；末 chunk end_idx = min(extend_end, len(origin_input_ids)) 不对齐（L1766-1767 / L1791-1793）；早发路径 maybe_send_cached_prefix_chunk（L1712）同样走 send_kv_chunk 并更新 start_send_idx，telescoping 不破。总字节 κL(T−T_hit) 与 chunk 数无关 ✓；b=1 退化平凡 ✓；边界反例（不满页早发 → 传未算槽位或重发）成立。

## 4. 数值独立重算（python3，全部命中除 S3-1）

- 6.1：κL=131072 B=128 KiB ✓；48 GiB/131072=393216 ✓；2048×4096×4=32 MiB、0.0651% ✓；131072 列=1 GiB ✓；2Nδ=15.3 TFLOP ✓；jump 5×5.13=25.65≈25.7 ms ✓；API spec 3060/1002=3.05 ✓（练习 3：8210/2002≈4.1，方向锚成立）。
- 6.2：F 表 1.097*/1.195/1.585/2.000/1.641（*1.098 为未舍入 t_gpu 口径，S3-1）；利用率 91.1/83.7/63.1/50.0/39.1% ✓；临界行 T_normal 10.26≠10.25（S3-1）；t_gpu 减半 F=1.390 方向 ✓；new_token_ratio 358.4 token/44.8 MiB → 0.098/50 token/6.3 MiB、7× ✓（env 默认 0.7/0.14/600/20 核 environ.py L708-711）；shortest_prefill_load H=D=8192、D_start=16384、B_full=24576 ✓，表 512/1024/2048/无cap 与 boundary 2048/2048/1024/512 ✓，练习 4 值 724/1448/无cap、K=8 boundary 2048 ✓；dedup 12 TFLOP ✓。
- 6.3：κL=70272 B=68.625 KiB ✓；274.5 MiB/seq ✓；17.16 vs 2.14 GiB ✓；4×549 MiB=2.14 GiB ✓；24576 token=1.61 GiB ✓；34.5 ms@50 / 276 ms@6.25 ✓；chunk 11.5 ms ✓；aux 占比 ~6.3e-6 ✓；31232 描述符、72 KiB/页/层 ✓；0.875 MiB all-reduce ✓；练习 1 rank13=(dp3,cp0,tp1) 回代 13 ✓、ep4×moe_dp2=8≤16 过 L153 但 ≠16 拒于 L159 ✓。

## 5. 代码锚点核查（verification log）

**覆盖率**：三章全部 file:line 锚点逐一核对（约 190 个），研究笔记锚点全量复核；下列为逐项记录（路径相对 python/sglang/，fresh tarball 2026-11-23）。未列出的随文锚点（如 §3 数据流内各行号）均在同批 grep/sed 中命中。

### 6.1（全部命中）
| 锚点 | 源 | 结论 |
| --- | --- | --- |
| lang/api.py gen L75 / select L236 / image L228 / video L232 / gen_int L142 / gen_string L185；choices 分支 L102（return L103）；regex 校验分支 L111（compile L113） | api.py sed | ✓ |
| interpreter.py fork L367 / wrapper L885；tracer.py L108；base_backend.py fork_program L38；fork 摘录（SglCommitLazy/sync/variables/text_/fork_start_text_pos，官方另复制 messages_/images_/cur_role 章已注差异） | sed L367-398 | ✓ 摘录忠实 |
| engine.py Engine L266 / update_weights_from_ipc L1790 / release L1899 / resume L1905 | grep | ✓ |
| tokenizer_manager.py TokenizerManager L608 / ReqState L281 / _tokenize_one_request L1275 | grep | ✓ |
| scheduler.py Scheduler L513 + 五 mixin（disagg decode/prefill/PP/dllm/mlx overlap）L513-519；7322 行 | sed/wc | ✓ |
| detokenizer_manager.py L80 / event_loop L164；tp_worker.py BaseTpWorker L83 / TpModelWorker L283 / forward_batch_generation L624 | grep | ✓ |
| ipc_channels.py SchedulerIpcChannels L17；PULL scheduler_input L36 / PUSH tokenizer L48 / PUSH detokenizer L59 | sed | ✓（raw→SenderWrapper 包装，摘录语义不变） |
| io_struct.py 3043 行；GenerateReqInput L220 / TokenizedGenerateReqInput L1205 / BatchTokenizedGenerateReqInput L1345 / BatchTokenIDOutput L1832 / BatchStrOutput L1942 / BatchEmbeddingOutput L2039 / FlushCacheReqInput L2068 / SessionParams L146；server_args PortArgs L826 | grep/wc | ✓ |
| model_runner.py ModelRunner L332（ModelRunnerOutput L306）；forward_batch_info.py ForwardBatch L442；forward_mode.py ForwardMode L20 **恰八值**（EXTEND/DECODE/MIXED/IDLE/TARGET_VERIFY/DRAFT_EXTEND_V2/PREBUILT/DLLM_EXTEND） | grep/sed | ✓ |
| radix_cache.py 839 行；RadixKey L60（token_ids/extra_key/cache_salt/is_bigram/limit）/ page_aligned L148 / match L179→match_at L183 gallop+二分 C 级切片 / child_key；TreeNode L251 / evicted L275=value is None；RadixCache L282；match_prefix L369 docstring「intentionally kept disjoint」逐字 / _match_prefix_helper L658 摘录忠实；insert L433 / _insert_cache L455 / free_segment L509-512；evict L559（get_priority/evictable_leaves/free_segment start_pos=0/_delete_leaf/父入堆 L580/kv_events.record_remove）；inc L589 / dec L604 / evictable_size L628；_split_node L684（继承 lock_ref L691、value clone L693/696、**无 _update_leaf_status**）；_insert_helper L712（新叶 L762-763）；_delete_leaf L787；_update_leaf_status L797 三合取 | 全文分段读 | ✓（free 段语义见 S2-3） |
| memory_pool.py 6701 行；ReqToTokenPool L382 docstring 逐字 / req_to_token (size+1, max_context_len) int32 L422-424 / 行 0 注释逐字 L417-419 / free_slots 尾部弹出 O(need_size) L453-468 注释逐字 / req_generation 单调 L485；MambaPool L521 / KVCache L1949 / MHATokenToKVPool L2282 / _kv_buffer_shapes L2666（NHD rows=size+page）/ FP4 L3648 / MXFP8 L3874 / MLATokenToKVPool L4875（kv_buffer (rows,1,kv_cache_dim) L4993-4998、kv_cache_dim=512+64）/ DSATokenToKVPool L5467 | sed/grep | ✓ |
| allocator/token.py L28（free_pages arange(1,size+1)、slot 0 注释逐字）；allocator/paged.py L132（(seq_lens−1)%page_size==0 L124、alloc_decode_kernel Triton L325/340） | sed/grep | ✓ |
| unified_radix_cache.py UnifiedRadixCache L224；hybrid_cache/hybrid_cache_controller.py HybridCacheController L325（BaseHiCacheController） | grep | ✓ |
| constrained/ GrammarManager L37 / XGrammarGrammar L78 / XGrammarGrammarBackend L201 / GuidanceBackend L224 / OutlinesGrammarBackend L88 / inkling2 try_jump_forward L229 return None + jump_and_retokenize L232 raise | grep/sed | ✓ |
| lang/backend/openai.py L100 注释 / spec_max_num_tries=3 L104 / _prepare_spec_execution L109 / max_tokens 覆写 L116 / 采样一致断言 L132 / L153 错误提示逐字「@function(num_api_spec_tokens=128)」 | sed | ✓ |
| 论文引文（2312.07104v2 全文）：2.1× 行数（L114）/ 6.4× 摘要（L61）+ baseline Guidance/vLLM/LMQL（L98）/ essay judge running example（L102-113：select related→fork 3→merge→summary+grade→regex JSON）/ 契约句 L170 逐字 / CUDA kernels asynchronously L126 逐字 / Frontend Hint L172 / page=1 token L166 逐字 / LRU 叶优先 L167 逐字 / compressed FSM 四步（L660 characters/strings、L210 one forward pass、L668 retokenize 例与 minor overhead、L670 distorted probabilities 逐字 + [ABCD][+-] 例 + workaround）/ API spec exec L213-215（character description 例逐字、ignoring stop condition、interpreter keeps…matches and reuses）/ ~threefold L281 逐字 | sglang-paper.txt | ✓ 全部逐字命中 |

### 6.2（全部命中，除 S3-3 行号）
| 锚点 | 源 | 结论 |
| --- | --- | --- |
| scheduler.py event_loop_normal L2231 / event_loop_overlap L2266（result_queue deque、pop_and_process、early 闸门、batch.copy() 入队、not-early L2322、L2329 注释逐字、launch_batch_sample_if_needed L2331）；教学摘录与源码主干一致 | sed L2190-2420 | ✓ |
| _apply_war_barrier L2207-2218（细/粗粒度 + FORCE_COARSE env）摘录逐字；start_scheduler_time_accounting L2204；scheduler_stage_method 标 GET_NEXT_BATCH（L4385）/ PROCESS_BATCH_RESULT（L5956）两 stage | sed/grep | ✓ |
| should_process_last_batch_early L2338（SGLANG_DISABLE_OVERLAP_AFTER_PREFILL + DP is_extend_in_batch 同步注释 L2346-2354 / grammar_needs_sync L2357）；_advance_pending_grammar **def L2359**（章锚 L2361 在 docstring 内 → S3-3）；ingest_requests L2369 | sed | ✓（行号 S3-3） |
| init_overlap L1837 / L1840 注释逐字「FutureMap is always-on: input_ids relay used in both modes」/ create_future_map L1854 | sed | ✓ |
| get_next_batch_to_run L4386（merge/filter L4440-4463 / spec+DP 不混批 L4477-4488 注释逐字 / maybe_prepare_mlp_sync_batch L4487+L4506 / update_running_batch 调用 L4496）；_build_hisparse_decode_batch L4321 内 stash L4349-4351（RelayPayload(bonus_tokens=last_tokens) + input_ids=None）；get_new_batch_prefill L4553；update_running_batch L5262（check_decode_mem L5275→retract_decode L5297→ratio 重估）；init_chunked_prefill L1468 | sed/grep | ✓ |
| run_batch L5455：CPU 段 L5488-5501 / resolve_seq_lens_cpu L5511（stream_ordered L5520）/ forward_stream_ctx L5515 + wait_stream L5516 / resolve_forward_inputs L5525 / on_publish L5537 / forward_batch_generation L5553 / publish(seq_lens+1) L5558 / _relay_forward_payload L5578 / _copy_result_to_cpu L5589 / input_ids=None L5594 + L5593 注释逐字 | sed | ✓ |
| _relay_forward_payload L5835（bonus_tokens=next_token_ids L5850、relay L5860）；_copy_result_to_cpu L5900（copy_stream.wait_stream L5914、async_d2h_worker L5926）；launch_batch_sample_if_needed L5928（forward_stream_ctx L5936、delay_sample_func L5938、relay L5941）；process_batch_result L5957（decode/prefill/prebuilt/idle 分流）；copy_done.synchronize 在 batch_result_processor.py L1157 | sed/grep | ✓（S2-2 证据） |
| overlap_utils.py 824 行：FutureMap L309 docstring 逐字 / buf int64 req_pool_size L338-350 / slot 0 注释 L324-325 / _DEBUG_ASSERT=SGLANG_IS_IN_CI L74（生产 empty）/ _assert_nonneg_and_invalidate L92 / resolve_forward_inputs L127（prefill H2D L141-142、MIXED cat L143-160、decode gather L165-166）/ RelayPayload L189 / ConfidenceRelay L226 / FutureRelayBuffers L101 / decide_needs_cpu_seq_lens L26 / defer_relay L409 / relay L420（topk==1 bypass）/ make_staged_spec_input L531 / resolve_confidence_cpu L552 / stash_bonus_tokens L634 / resolve_mixed_spec_tails L642 / resolve_seq_lens_cpu L687（L734-735 注释逐字、私有流 L736-739）/ publish L749（publish_ready 仅 spec L769-773）/ stash L780 / _stash L784（scatter L800） | 全文读 | ✓ |
| schedule_batch.py 3701 行：Req L442（ReqDllmMixin）/ ScheduleBatch L2009 / L2855 MIXED 注释逐字 + input_ids=None + mix_running_indices / copy() L3545 注释逐字（filter/merge rebind、切片防别名）/ NextBatchPlan L3699 / grammar_needs_sync L2279 | grep/sed | ✓ |
| schedule_policy.py 2548 行：in-batch 双阈值 L114-127 默认 32/32 + 语义注释（cache 命中<check 触发检查、队列命中>deprioritize 压尾）/ BOUNDED_LPM_WINDOW_SIZE=64 L181 + 注释逐字 / UNIBOOST_MATCH_WINDOW_SIZE=64 L185 / CacheAwarePolicy L188 五值 / CacheAgnosticPolicy L198 四值 / SchedulePolicy L207 / AddReqResult L1012 四值（UNSCHEDULABLE abort 注释）/ _PrefillAdmission L1022 / PrefillAdder L1077（rem_input_tokens−num_mixed_decode_tokens、rem_chunk_tokens、new_token_ratio、memory_budget L1139、chunked_req_limit/dllm_config）/ exact_chunk_fill L1125-1131 + deterministic 注释逐字 / add_one_req L2103 / _select_prefill_admission L2304 | sed/grep | ✓ |
| new_token_ratio_tracker.py NewTokenRatioTracker L14（init×conservativeness、min=init×factor、decay=(init−min)/steps、retract 公式 (decoded+steps×n)/(Σmax_new+1)）；environ.py 默认 0.7/0.14/600/20 L708-711 | 全文读 | ✓ |
| prefill_delayer.py RecentPrefillBatchSizeTracker L26（窗口 16、decode-only/idle 不老化 docstring 逐字）/ PrefillDelayer L76 / 5000ms 注释逐字 L90-91 / PrefillDelayerSinglePassExecutor L384 / negotiate_should_allow_prefill L429；**触发语义 L177-181 force-allow + L252-280 slot/queue delay** | sed | ✓ 锚点；语义 → S2-1 |
| shortest_prefill_load.py 88 行全文：docstring 逐字（backlog B 定义、类边界、halving、budget、FCFS 收敛）/ resolve 默认 decay_start=2K、min_ratio=1/8、doubling=K / class_boundary L71-74 = 式 (4) / token_budget L76-88 = 式 (5)（B≥B_full→None） | 全文读 | ✓ 式 (4)(5) 与代码恒等 |
| data_parallel_controller.py 1003 行：WorkerPortHandshake L100 / LoadBalanceMethod L106 四值 / DPBudget L123（update_budget shm+timestamp 跳 stale、TOTAL_REQUESTS argmin、TOTAL_TOKENS (tokens,requests) 字典序、+1 启发式）/ DataParallelController L159 | sed | ✓ |
| tp_worker.py delay_sample 条件 L747-767（overlap ∧ ¬spec ∧ (grammar ∨ DELAY_SAMPLE ∧ ¬prefill-only)） | sed | ✓ |
| batch_overlap/：two_batch_overlap.py compute_split_seq_index L80 / TboCudaGraphRunnerPlugin L338 / TboDPAttentionPreparer L395 / TboForwardBatchPreparer L503 / model_forward_stages L953 / MaybeTboDeepEPDispatcher L1161；single_batch_overlap.py SboFlags L28 / CombineOverlapArgs L63 / DownGemmOverlapArgs L75；operations.py execute_operations L29 / execute_overlapped_operations L38 | grep | ✓ **§9 两机制区分正确**（CPU↔GPU vs compute↔comm） |
| server_args.py L345/L355 enable_dp_attention 派生（resolved_dict 注释 + 代码逐字）；runtime_context.py derive_attn_tp_size L73 / get_parallel L1314 / get_schedule L1351；layers/dp_attention.py DpPaddingMode L153 / set_dp_buffer_len L354 / get_global_dp_buffer L370 / deployment_attn_dp_size L61 | sed/grep | ✓ |
| blog 引文（evaluator 重抓 + 存档 txt）：half CPU overhead 逐字 / 1.1× vs v0.3 + 1.3× vs SOTA / small models large TP 逐字 / NanoFlow lineage 逐字 / future tokens + CUDA events 逐字 / Nsight 5 batch 零 idle（Triton、FlashInfer gap 注）逐字 / --disable-overlap Llama-3.2-3B-Instruct / DP attention 1.9×（DeepSeek-Coder-V2-Instruct-FP8、8×H100 80GB、--random-input 1 --random-output 512 --num-prompts 10000 逐字）/ duplicated KV cache 引文逐字 / cache-aware LB 1.9×/3.8× + 近似树 lazy / Rust router / XGrammar 10× | web + txt | ✓ 全部 |
| DeepInfer doc L334 逐字（Following existing implementations (SGLang, NanoFlow)… schedule(N+1)→launch_forward(N+1)→post-process(N−1)→launch_sample(N+1)… no idle time throughout） | DeepInfer-main/doc | ✓ |
| vLLM docs/design/arch_overview.md L173-179 Async Scheduling 节逐字（while the GPU executes step N, the CPU schedules step N+1, using placeholders for the tokens that step N will produce） | vllm tarball | ✓ |

### 6.3（全部命中，除 S3-4 文件归属）
| 锚点 | 源 | 结论 |
| --- | --- | --- |
| arg_groups/fields/parallel.py dp_size L77 / attn_dp_size L85 help 逐字；parallel_hook.py L136/L149 注释逐字「The tp_size is the world size, not the real tensor parallel size」/ cp gate L135-141 / MoE gate L148-161（moe_dp>1 前提、L153 ≤、L159 ep>1 等式、pp==1）/ **L389 attn_dp 整除断言（章未引 → S3-2）**/ L650-659 elastic divisor | sed | ✓ |
| runtime_context.py derive_attn_tp_size L73（return tp_size//attn_dp_size//attn_cp_size）/ derive_attention_ranks L79（docstring (dp,cp,tp) 逐字、返回 tp%a 与 tp//(a·c) 两项）/ attn_dp_enabled_of L191-199 / get_parallel L1314 | sed | ✓ |
| server_args.py resolved_dict L345-355 逐字 | sed | ✓ |
| parallel_state.py GroupCoordinator L271 / get_world_group L2178 / get_tp_group L2258 / get_attn_tp_group L2262 / get_attn_cp_group L2270 / get_dcp_group L2278 / get_moe_dp_group L2287 / get_moe_ep_group L2291 / get_moe_tp_group L2295 / get_pp_group L2311 / initialize_model_parallel L2596 | grep | ✓ |
| device_communicators/ CustomAllreduce L40 / AllReduceChoice L111 / CustomAllReduceV2 L152 / TorchSymmMemCommunicator L34 / MooncakeTransferEngine L117（KVTransferEngine 子类）；pynccl/pymscclpp/quick/ring_simple/dpx_hier/multicast/shm_broadcast 文件均在 | grep/ls | ✓ |
| eplb/ EPLBManager L29（四 getter 协作对象 + >= 断言 + stale data 注释逐字）/ ExpertDistributionRecorder L135 + 五 gatherer L489/L576/L605/L660/L693 / ExpertDistributionMetrics L121 / ExpertLocationMetadata L74 / deepseek.py rebalance_experts_hierarchical L86 / rebalance_experts L171 | sed/grep | ✓ |
| elastic_ep/ ScaleCohort L30（msgspec frozen）/ register L37 / validate L65 / update_dp_attention_for_elastic_ep L77 / ElasticEPState L99 / ElasticEPStateManager L132 / elastic_expanded_world_enabled L374；expert_backup_{manager,client}.py 在 | grep/ls | ✓ |
| disaggregation/base/conn.py **341 行全文读**：StateType L17（SWA/DSA/MAMBA/BLOCK_SCALE 等）/ KVTransferMetric L49 / KVTransferDestination L57 DEVICE/HOST / KVArgs L62（kv_data_ptrs/lens/item_lens/layer_ids/dtype_str/kv_layout 注释逐字 L69/aux/state_types/page_size/kv_head_num/total_kv_head_num/ib_device/gpu_id/pp_rank/prefill_start_layer/prefill_end_layer/host_kv_data_ptrs/**state_dim_per_tensor L94 注释逐字**）/ KVPoll L142 五态值 / BaseKVManager L150（should_verify_kv_checksum L156 TP 宽度语义 docstring、register_to_bootstrap L173）/ BaseKVSender L191（init L202/send L209/poll L240）/ BaseKVReceiver L266（init L289/send_metadata L299 docstring 逐字/poll L313）/ BaseKVBootstrapServer L339 | 全文读 | ✓ |
| common/conn.py 3631 行：PrefillServerInfo L126 / AbortNotification L196 / AbortAck L264 / SendRecord L300 / CommonKVManager L311（request_status L453、update_status L787）/ CommonKVSender L2325 / CommonKVReceiver L2698 / CommonKVBootstrapServer L3244（aiohttp web.Application L3248）；common/bootstrap.py BootstrapRoom L18 / DeferredBootstrap L27 / 模块 docstring 逐字「This table only joins a completed optimistic prefill with decode's control endpoint. It performs no I/O, including while holding its lock.」 | grep/sed | ✓ |
| prefill.py：clamp_kv_send_end L148（min(end_idx, cache_protected_len) 只推迟）/ PrefillBootstrapQueue L270 / start_send_idx=decode_prefix_len L546 / get_next_disagg_prefill_batch_to_run L818（process_prefill_chunk call L839）/ event_loop_normal_disagg_prefill L856 / process_prefill_chunk L1661 / maybe_send_cached_prefix_chunk L1712（WAR 注释逐字 + ev.record(forward_stream) L1749-1751）/ send_kv_chunk L1753（末 chunk checksum 注释逐字、end_idx=min(extend_end,len(origin_input_ids))）/ 页对齐 L1799 / start_send_idx=end_idx L2084 / staging raise L322-325 逐字 / staging fallback cps=chunked_prefill_size or 8192 L330 | sed/grep | ✓ 命题 2 全锚点 |
| decode.py：DecodePreallocQueue L471 / move_preallocated L1653（不变式注释逐字 L1677-1678）/ DecodeTransferQueue L3303 / move_transferred L3610 / publish_pending_metadata L3614（注释逐字 + all_reduce_flags MIN）/ SchedulerDisaggregationDecodeMixin L4061 / event_loop_normal_disagg_decode L4063 / prepare_for_prebuilt 调用 L4458 / process_decode_queue L4541（polling_interval L4569）；decode_schedule_batch_mixin.py prepare_for_prebuilt L24 / 「PREBUILT never enters a model forward」L32 逐字；forward_mode.py PREBUILT L38 | grep/sed | ✓ |
| io_struct.py bootstrap_host/port/room L350-352；disagg_service.py start_disagg_service L17 + 注释逐字 L18；mode.py DisaggregationMode NULL/PREFILL/DECODE + FAKE_BOOTSTRAP_HOST="2.2.2.2" L5 | grep | ✓ |
| utils.py MetadataBuffers L639（注释逐字 L688-689、output_ids (size,16) int32 L690、output_hidden_states (size,hidden_size) L737）/ TransferBackend L1006 五值 | sed | ✓ |
| mooncake/conn.py 4060 行：MooncakeKVManager L364（StagingManagerMixin+CommonKVManager）/ init_engine L519 / register_buffer_to_engine L547 / send_kvcache_staged L814 / _send_kvcache_generic L1342（group_concurrent_contiguous L1374）/ send_kvcache L1675 / send_kvcache_dcp L1845 / send_kvcache_slice L2204 / send_aux L2344 / send_aux_tcp L2383 / Sender L4015 / Receiver L4055 / BootstrapServer L4059 | grep | ✓ |
| nixl/conn.py 3856 行：NixlKVManager L531 / register L2153 / send_kvcache L2505 / dcp L2562 / mixed L2633 / staged L2782 / send_aux L2981 / Sender L3843 / Receiver L3851 | grep | ✓ |
| connector/ BaseConnector L13（URL docstring 逐字）/ weight_iterator L35 / pull_files L41 / BaseKVConnector L75（RedisConnector 基类，权重/文件语义）/ BaseFileConnector L97 / S3Connector L69 / RedisConnector L16 / AzureBlobConnector L66 / RemoteInstanceConnector L17 | sed/grep | ✓ |
| checkpoint_engine/ worker Extension L39 / Impl L95；update.py split_checkpoint_files L79 / split_tensors L92 / req_inference L113 / update_weights L144 / join L182；**引文 docstring 在 worker L15-16（S3-4）**；weight_updater.py Manager L131 / flush L164 / record L179 / disk L182 / init_group L210 / destroy L215 | grep/head | ✓ |
| compilation/ CompilationConfig L20（capture_sizes/compiler eager 默认/split_ops←SPLIT_OPS）/ configure_inductor L50（combo_kernels=True L60 + q_norm/k_norm 注释逐字 L58）/ backend.py CompilerManager L82 / SplitItem L223 / PiecewiseCompileInterpreter L279（torch.fx.Interpreter）/ SGLangBackend L371（compiler=="inductor" L38）/ cuda_piecewise_backend.py ConcreteSizeEntry L33 / CUDAPiecewiseBackend L49 | sed/grep | ✓ |
| dllm/ config.py DllmConfig L8（algorithm/block_size/mask_id/max_running_requests/first_done_first_out_mode）/ algorithm/base.py DllmAlgorithm L30 | sed | ✓ |
| vLLM 对照列：GroupCoordinator L427 / CustomAllreduce L120 / SymmMemCommunicator L29 / EplbState L321（eplb_state.py）/ KVConnectorBase_V1 L182 / MooncakeConnector L798 / MultiConnector L142 | vllm tarball grep | ✓ |

### 跨章 recap / 符号 / 规范 log
| 项 | 对照 | 结论 |
| --- | --- | --- |
| 4.1 式 (3) 5.13 ms（B=8、T̄=1024、带宽下界） | 4.1 L120 原式代入 | ✓ 忠实（精确值 5.1265，S3-1 精度注） |
| 4.1 §6「重叠改时序不改迭代级语义」 | 4.1 L195 | ✓ 逐字 |
| 2.2 命题 1（O(T)）/ 命题 2（节省式）/ 式 (2) 在 §4 / extra_key 坑注 §2 / b/2=32 token≈4 MiB | 2.2 L70/L103/L106/L46/L44 | ✓ |
| 1.2 式 (3) κ 两行 / N=7.50e9 matmul / κL=128 KiB / V3 行 61L 7168 512+64 / 68.6 KiB | 1.2 L48/L19/L60/L16/L236 | ✓ |
| 2.4 式 (3) BW_req→t_send recap 形 / 400 Gbps÷8=6.25 GB/s / §8 版本元组含权重 + 三理由 / §9 HiCache 三层 + HiSparse PD direct-to-host | 2.4 L81/L86/L161/L157/L163-175 | ✓ |
| 3.4 §6 batch-invariant / §7 式 (4) F_max≤3 / NanoFlow lineage | 3.4 L124/L140/L164 | ✓；**3.4 L7/L99 指针 6.2→6.3 errata（ERR-1）** |
| 4.2 §5 九策略两族 + 式 (5)(6) + temporary deprioritized + bounded LPM / §2 retract recompute 账 | 4.2 L116/L132/L139/L149/L31 | ✓ |
| 4.3 式 (2)(6) / §5 chunked_prefill_size 迁出漂移注 / §6「chunk size 是运行时变量」 | 4.3 L84/L126/L210/L239 | ✓ |
| training 5.1 §8 MLA latent 复制 + 7168 / 每层 2 次前向 all-reduce（训练 4 含反向） | training 5.1 L224-230 + metadata | ✓ |
| 2.1 §1 48 GiB / §3 b=64 版本锚 / §4 ref_cnt | 2.1 L17/L137/L148 | ✓ |
| 1.4 §5 grammar mask / §7 增量 detokenization；models 2.4 前缀不可变命题 | 1.4 L132/L168；models 2.4 L112 | ✓ |
| 符号 §4.3：B/b/bytes/κ/d_model/d_latent/d_rope/γ 未误用；c_s/c_p/c_cpu/j 局部声明 ✓；c=CP 度（6.3）✓、c_chunk 局部声明 ✓；N 专归参数量（blog forward(N) 保留原记号 + 注）✓；6.2 §7 B/C 与 6.3 t 张力 → S3-5 | 全章扫 | ✓（S3-5） |
| 两层走读纪律：6.1 §4 recap 2.2 + 实现增量 / 6.2 §6-§7 recap 4.2-4.3 + 接线增量 / 6.3 §8 recap 3.4 + 引擎调用级 / 通信账 goodput 带宽全账归 8.1-8.3 纯文字 / 6.1 §8 vLLM 列 recap 2.1-2.2 + 5.1-5.2 指针 | 各章 | ✓ 不重复展开 |
| 模板：metadata 行 / 总结表回指列 / 练习 2-4 题难度非递减（[10][20][20][30] / [10][20][30][30] / [10][20][30][30]）/ 每章 ≥1 题证命题（6.1 练 2→命题 1、6.2 练 2→命题 2、6.3 练 3→命题 2）/ 校验锚全在 / ASCII 图 ≥1（3/1/3） | 三章 | ✓ |
| R2-1 节首（问句/数值/事实开场逐节扫）/ R2-2 章脊声明（三章均有，6.3 换 T 有原因句）/ R2-3 式后 300 字内数字实例 / R2-4 命题骨架 + 路线句 + 正反例 / R2-7 口径注 / R2-8 判断句定义（所谓 SGLang/DP attention/EPLB/bootstrap/PD 分离）/ R2-10 提点 / R2-11 坑注（2/3/4 处，触发+现象+规避）/ R2-12 回指 | 三章 | ✓ |
| lint（检查项 1-6） | bash scripts/lint-chapter.sh ×3 | LINT OK ×3 |
| 篇幅（正文 CJK，去代码/公式） | python 独立计数 | 6.1=4823≥3500 ✓ / 6.2=6141≥5500 ✓ / 6.3=5002≥3500 ✓ |
| 禁令词（§9 词表 + rubric A1 扩展 + A2 营销词 + 语气词其实/呢/吧/嘛） | python 全扫 | 0 命中 |
| 历史叙事 / 空洞评论段 / 术语漂移 | 人工扫 | 无违规（blog/论文均 citation 化，版本口径非年代叙事） |
| 引用白名单：6.1=sglang/sglang-repo/sglang-v04-blog/xgrammar/outlines/guidance/vllm/nvidia-datasheet；6.2=+nanoflow/deepinfer-doc/vllm-repo；6.3=+mooncake/nixl；hybridflow 未引 | references.md 对照 | ✓ 全在白名单；arXiv ID 全有条目；被引章节字段回写留 closeout |
| 证据等级：[A] 引擎/论文、blog [C] 开放技术博客非闭源、DeepInfer doc 教学参照+官方披露、guidance/nixl 教学参照 | 各章 metadata + 参考 | ✓ |

## 6. 结论

- **S1=0 / S2=3 / S3=5 + errata 1**。三章代码锚点密度与准确率极高（~190 锚点仅 2 处行号/归属微差），引文全部逐字命中，数值全部可复算，两层纪律与白名单合规。
- 三个 S2 均为**代码语义误读**而非锚点缺失：PrefillDelayer 触发反转（S2-1）、D2H 同步 pacing 缺失致证明保持步与 §2 机制句在 GPU-bound regime 失真（S2-2）、free_segment 释放段误读致练习锚错误（S2-3）。修复均为节内改写，不动章节结构。
- **强制重推导**：bubble 账公式/条件/数值 PASS、resolve 命题证明 PASS（当前 stash/publish/resolve 语义）；命题 1 书面证明保持步需补 pacing 步（S2-2）。6.1 命题 1、6.3 命题 1/2 复核 PASS。
- S2 修复后须复审（6.1/6.2 各一处以上）；S3 与 ERR-1 按 §6.4 通道处置（ERR-1 归 W2 errata 通道，3.4 两处 6.2→6.3）。
