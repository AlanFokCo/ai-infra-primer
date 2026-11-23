# Inference W3 修复复审报告（fix round re-review）

- 日期：2026-11-23
- 审查人：fresh-context re-reviewer（不继承写作 / 修复上下文）
- 对象：W3 七章 `inference/05-vllm/{01,02,03,04}-*.md` + `inference/06-sglang/{01,02,03}-*.md`
- 依据：原 findings `docs/reviews/inference-w3-review-p5.md`（S1=0/S2=11/S3=13）+ `inference-w3-review-p6.md`（S1=0/S2=3/S3=5+ERR-1）；修复报告 `.superpowers/sdd/inference-w3/writer-{A,B,C,D}-report.md` fix round 段；spec §4.3；STYLE_GUIDE / R2。
- 代码事实源：**re-reviewer 自取 fresh codeload tarball**（vllm main 53,567,931 B / sglang main 47,805,849 B，2026-11-23，解压 `/tmp/iw3-fixrev/x/{vllm,sglang}-main`），不信任 writer 报告行号。关键文件行数与 research / evaluator 快照逐项一致（scheduler.py 3993 / kv_cache_manager 935 / single_type 3377 / core 3177 / gpu_worker 1956；sglang scheduler 7322 / radix_cache 839 / overlap_utils 824）——同 commit 口径，锚点分歧判笔误而非版本漂移。
- 方法：/usr/bin/grep -F 逐字 + sed 逐行核锚点；数值 python3 -I 独立重算（`/tmp/iw3-fixrev/scripts/nums.py`）；CJK 独立计数（`cjk_count.py`）；lint 7/7 重跑；命题独立重推。

## 0. 结论

**S2 = 14/14 ADDRESSED（P5 11/11 + P6 3/3）；S1 = 0；新增破坏 = 3 条 S3 残留（无新 S1/S2）。** 全部修正锚点对 fresh tarball 精确命中；全部改动证明独立重推 PASS；数值 / 符号 / lint / 篇幅 / 命题编号 / 练习分级全过。3 条 S3 残留均为「修复后正文已正确、但压缩复述行（ASCII / 总结表）未同步」的内部不一致，一行可修，不阻塞 S2 关闭。references 被引回写（S3-9）/ 3.4 指针（ERR-1）/ 1.4 指针（P5 advisory）/ research notes prefill_delayer 措辞属 out-of-batch 通道，确认仍 pending，不计本 batch 破坏。

## 1. S2 逐条裁定（14 条）

| # | 章·位置（修后） | 裁定 | 一手证据（fresh tarball / 重推） |
| --- | --- | --- | --- |
| P5-S2-1 | 5.2 §1 L30 + 口径注 L32 | **ADDRESSED** | 2048 改标「教学预算（SchedulerConfig dataclass fallback，config/scheduler.py L40）」；tarball L40 `DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048` ✓，docstring L49-52「mainly for convenience when testing…set in EngineArgs.create_engine_config」逐字 ✓。口径注 arg_utils.py L3002-3023：≥160 GiB 双 16384（L3002-3005）/ H100·H200（≥70 GiB 非 A100 非 GB10）LLM_CLASS=16384·OPENAI_API_SERVER=8192（L3016-3019）✓。章内算例仍用 2048（已声明）。 |
| P5-S2-2 | 5.2 §4 L88 | **ADDRESSED** | 「含未来生成」删，改「全输入序列」；检查长度 min(request.num_tokens, max_model_len)（kv_cache_manager.py L544 ✓）；num_tokens=prompt+已生成（request.py L338-339 `return len(self._all_token_ids)` ✓）；config/scheduler.py L226-230 docstring「full input sequence length…rather than only checking the first chunk」逐字 ✓；「不含未来生成」明确。 |
| P5-S2-3 | 5.2 §4 L84 | **ADDRESSED** | 口径差声明：2.2 原文峰值 989.5→≈49 ms vs 本章 F_eff=494.75→98.1 ms，差 2×。2.2 §4 L113 原文「48.6 TFLOP…989.5 TFLOPS 理想折算省约 49 ms」核 ✓。重算：save=4.8554e13 → @989.5=49.07 ms / @494.75=98.14 ms；full=6.5838e13→133.07 ms；actual=34.94 ms；133−98.1=34.9 ✓；989.5/494.75=2.0 ✓。 |
| P5-S2-4 | 5.2 §6 命题 1 证明 L163 + 练习 4a L221 | **ADDRESSED** | 界 ⌊ℓ₀/a⌋+1→**+2**；对齐前提补全：ℓ₀=num_tokens−1 一般不对齐（4096→4095）、对齐只对 manager 返回值成立（single_type **L954** `hit_length -= hit_length % alignment_tokens`）、首轮缩 ≥1 可 <a、后续每轮 ≥a。练习 4a 校验锚同步。**重推 PASS**（见 §2.1）。writer L954 实测正确（evaluator 原引 L957 实为 `del computed[num_blocks:]`，±3 行笔误，writer 自纠成立）。 |
| P5-S2-5 | 5.4 §2 继承链表 L37 + 命题 1 statement L41 + 证明 L43 + 练习 4a L339 | **ADDRESSED** | MTP override 补全 5 个：load_draft_model L15 + on_prefill_begin L42 / on_prefill_end L48 / on_multi_step_decode_begin L55 / on_multi_step_decode_end L61；share_mtp_topk_indices 改标类属性 L13（load_draft_model 内 L34 赋值）。tarball mtp/speculator.py（63 行）def 行 15/42/48/55/61 + L13 类属性 + L34 赋值 + hook 体 set_skip_topk(L46/59/63)/compact_topk_indices(L53) 逐行 ✓；eagle（18 行）只 def load_draft_model L13 ✓；两壳均无 propose ✓。**残留**：ASCII L59 + 总结 L316 未同步（新 S3-1/2，见 §3）。 |
| P5-S2-6 | 5.3 §1 执行链表 L22 | **ADDRESSED** | ExecutorWithExternalLauncher L161 移到 uniproc_executor.py 行、标继承 UniProcExecutor。tarball 全 repo grep 唯一命中 `v1/executor/uniproc_executor.py:161:class ExecutorWithExternalLauncher(UniProcExecutor)` ✓。 |
| P5-S2-7 | 5.3 §1 表 L25 + ASCII L45 | **ADDRESSED** | init_worker_distributed_environment L1910→**L1914**（两处）。tarball gpu_worker.py L1914 `def init_worker_distributed_environment(` ✓；L1910 = `def elastic_ep_execute` ✓。 |
| P5-S2-8 | 5.4 §5 L176 + 总结 L327 | **ADDRESSED** | LRUCacheWorkerLoRAManager L245→**L276**（两处）。tarball lora/worker_manager.py L276 `class LRUCacheWorkerLoRAManager(WorkerLoRAManager)` ✓。 |
| P5-S2-9 | 5.4 §2 命题 1 边界反例 L43 | **ADDRESSED** | _ngram_scan_kernel L18→**L21**。tarball ngram/speculator.py L18 空行 / L20 `@triton.jit` / L21 `def _ngram_scan_kernel(` ✓。 |
| P5-S2-10 | 5.3 §5 block size 段 L172 + 总结 L378 | **ADDRESSED** | cpu_mla 16 改标 supports_block_size 硬约束（L71-72 `return block_size is None or block_size == 16`）+ 平台层强制（docstring L25-26「the platform layer forces block_size=16 for MLA models」），明确「无 get_preferred_block_size、非偏好覆盖」；sparse_swa get_preferred_block_size L166-167=256 保留为偏好覆盖。tarball 逐项 ✓（cpu_mla grep get_preferred_block_size 零命中）。 |
| P5-S2-11 | 5.4 §5 L138 公式 / L140 翻译注 / snippet L185-193 / ASCII L204-206 / §1 L20 | **ADDRESSED** | LoRA 因子 B/A→**W_up/W_down**（ΔW=W_up·W_down、y=Wx+W_up(W_down·x)、shrink=W_down·x、expand=W_up(W_down·x)）；符号翻译注声明论文 B/A 按 §4.3 译出（B 专归 batch、A 已归 API server）。全章扫：$B$ 仅表 batch（L20/49/102/162/336）+ 翻译注引论文原记号；$A$ 仅在消歧注（L20/140）；snippet 形参 W_down/W_up ✓。**§4.3 合规**（见 §4）。 |
| P6-S2-1 | 6.2 §7 L216 + 总结 L306 | **ADDRESSED** | PrefillDelayer 触发语义重写：低水位=**force-allow 安全阀**（L177-181 `local_token_watermark_force_allow = local_prefillable and (token_usage < x)`；注释 L230-231「Safety valve: low KV usage means GPU is underutilized, skip delay」逐字 ✓；池快**空**非快满）；延迟触发=slot 条件（L275-278 `max_running_requests − global_running_batch_max < global_max_prefill_bs_max`，注释 L281-282「the first merge_batch causes the decoding to fail to reach the maximum batch size」逐字 ✓）+ opt-in queue 条件（L252-271，_queue_trigger_enabled=(queue_min_ratio is not None) L95 ✓）；机制账改「TTFT 换批效率 / GPU 利用率」，与 4.2 retract（KV 容量）显式区分；tracker 高水位=slot 右端项 ✓。总结行同步。 |
| P6-S2-2 | 6.2 §2 设计点其一 L52 + 命题 1 证明保持步 L130 + §5 基例 2 L154 | **ADDRESSED** | 补 copy_done.synchronize()（batch_result_processor.py def L1151 / sync **L1157** ✓）；GPU-bound 稳态 sync 等待≈t_gpu−c_cpu≈4.13 ms 是 **pacing 机制**（CPU 等 GPU）；「等待近零」限定 CPU-bound；「零开销=GPU 零 idle，非 CPU 零等待」。证明保持步改 L_{j+1}=max(L_j,S_j)+c_cpu + 不变式 L_j≤S_j；§5 基例 2 点明 HiSparse stash（L4349）在 schedule_stream、由 wait_stream（L5516）跨流定序。**重推 PASS**（见 §2.2）。锚点 L5914 copy_stream.wait_stream / L5516 forward_stream.wait_stream / L4349 stash 逐行 ✓。 |
| P6-S2-3 | 6.1 §4 insert 段 L161 + 对照表 L192 + 总结 L266 + 练习 4b L279 | **ADDRESSED** | free_segment 语义改正：匹配前缀=树自有索引借用（protect_prefix，common.py L230-231 ✓；schedule_batch.py L282-284 注释「The request's own KV is [cache_protected_len, kv_allocated_len)…Tree cache owns [0, here)」逐字 ✓；common.py L420-421「The protected prefix is not this req's to free」逐字 ✓）；free 段 [protected, prefix_len) 仅释放请求自算 KV 与执行期间新入树内容重复（radix_cache.py L509-512 ✓）；练习 4b 顺序 R2 归还 **0**（protected=prefix_len=3）+ 并发变体归还 **3**（protected=0）。**重推 PASS**（见 §2.3）。 |

## 2. 改动证明独立重推（不信任 writer）

### 2.1 5.2 命题 1 终止界 ⌊ℓ₀/a⌋+2 —— PASS（界正确且紧）

前提：初值 ℓ₀ 一般不对齐；各 manager 返回值对齐到 a（full：single_type L954 `hit_length -= hit_length % a`；SWA：L1228-1233 pop 至 `len*block_size % a == 0`；coordinator L766 对所有 manager 传同一 `alignment_tokens=self._cache_hit_alignment_tokens`，ChunkedLocal docstring 声明返回对齐——前提对全部被查询组成立）。

推导：设 ℓ₀ = q·a + r（q=⌊ℓ₀/a⌋，0≤r<a）。
- r≥1（不对齐）：首轮 ℓ₀→≤q·a（缩 ≥1，可 <a）；其后候选已对齐，每轮缩 ≥a，从 ≤q·a 降至 0 至多 q 轮；再加 1 退出轮（curr≥hit 不缩）。总轮数 ≤ 1+q+1 = **⌊ℓ₀/a⌋+2**。
- r=0（对齐）：首轮即缩 ≥a，至多 q 轮降 0 + 1 退出 = q+1 ≤ 界。
- 紧例 ℓ₀=4095、a=64：q=63，首轮 4095→4032（缩 63<64），4032→…→0 共 63 轮，退出 1 轮 = **65 = 63+2** ✓（python 逐轮 trace 确认）。旧界 +1=64 差 1，新界正确。
- 一致性部分（退出轮无组再缩 + full 向下封闭收尾截断 L789-799）未改，evaluator 已验，维持 PASS。
- 注：ℓ₀「= num_tokens−1」是默认口径简化（代码 get_max_cache_hit_length = min(num_tokens−1, prefix_cache_hit_limit, bound)，request.py L356-360；默认 limit=None 时即 num_tokens−1）。界对任意 ℓ₀ 成立，简化不影响证明。与 evaluator 建议措辞一致，不判。

### 2.2 6.2 命题 1 pacing 保持步 —— PASS（结论不变）

建模：S_j=GPU 开跑批 j、L_j=launch(j)（非阻塞，L_j≤S_j）。copy_done(j−1) 就绪≈S_j+ε（GPU 跑完 j−1+拷贝，copy_stream wait forward_stream L5914）。pop_and_process(j−1) 先 sync（L1157）再做 c_p，随后 sched(j+1) 做 c_s，故 **L_{j+1}=max(L_j, S_j)+c_cpu**（c_cpu=c_s+c_p，ε 吸收）。
- GPU-bound（c_cpu≤t_gpu）：不变式 L_j≤S_j 保持——L_{j+1}=S_j+c_cpu≤S_j+t_gpu=S_{j+1}（kernel 队列不空，GPU 零 idle，每拍 t_gpu）。稳态 L_j=S_j−t_gpu+c_cpu → sync 等待=S_j−L_j=**t_gpu−c_cpu≈4.13 ms**，launch 间隔=t_gpu ✓。
- CPU-bound（c_cpu>t_gpu）：L_{j+1}=L_j+c_cpu>S_j+t_gpu → GPU 跑完等 launch，S_{j+1}=L_{j+1}（不变式取等），每拍 c_cpu、空闲 c_cpu−t_gpu ✓。
- 合并 T_overlap=max(c_cpu,t_gpu)（式 2）；F=(c_cpu+t_gpu)/max（式 3）；完全隐藏 F=1+c_cpu/t_gpu≤2，等号 c_cpu=t_gpu ✓。
- 基例（流水填充）：第 1 拍 L_1≤S_1 平凡；第 2 拍 L_2=S_1+c_cpu≤S_2 ✓。§5 基例 2 HiSparse stash 跨流序由 wait_stream L5516 补 ✓。
- 数值：@5.13 表 F=1.097/1.195/1.585/2.000/1.641、临界 T_normal=10.26、利用率 91.1/83.7/63.1/50.0/39.1% 全重算命中 ✓；精确 t_gpu=5.1265 ms（4.1 式 3）✓。

### 2.3 6.1 free_segment 归还量 —— PASS

free_segment(kv_indices[protected:prefix_len])（radix L509-512）。匹配前缀 [0,protected) 是树自有索引借用（match 时 protect_prefix，common L230-231），不可释放。
- 顺序 R2（R1 先完成插树 [s,a,b,c]）：R2 match 时 protected=3，insert 时 prefix_len=3 → free [3:3]=空 → **归还 0** ✓。
- 并发 R2（树空、R1/R2 都 miss 各算全前缀，R1 先插 [s,a,b]）：R2 match 时 protected=0，insert 时树已有 [s,a,b] → prefix_len=3 → free [0:3]=R2 自算的 s,a,b 重复段 → **归还 3** ✓。
- 校验锚 max(0, prefix_len−protected) 正确；练习 4b 两情形与代码语义闭合 ✓。

### 2.4 5.4 命题 1（MTP 薄壳）—— PASS（枚举补全，结论不变）

mtp 63 行 override 集合 = {load_draft_model, on_prefill_begin, on_prefill_end, on_multi_step_decode_begin, on_multi_step_decode_end}（5 个，tarball 逐行核）；四 hook 体仅 set_skip_topk 切换 + compact_topk_indices 压实（index 共享 / 输入准备）；share_mtp_topk_indices 是类属性非 override。eagle 18 行只 {load_draft_model}。两集合均不含 propose → MRO 落 TargetDependentARSpeculator.propose L103 ✓。ngram 自实现 propose L213 不在链上 ✓。结论（薄壳不改 propose 语义）维持。

### 2.5 6.3 命题 1（c=1 整除分支）—— PASS

t=d·c·a 保证：c>1 由 parallel_hook L140（tp%(attn_dp·attn_cp)==0，gate attn_cp_size>1 L135）；c=1 且 attn_dp>1 由 **L389**（tp%attn_dp==0，gate attn_dp_enabled_of L377 = attn_dp_size>1 or ep_scale_joiner，runtime_context L199）✓。反例 t=8/d=3/c=1：cp gate 不触发，L389 assert 8%3==0 失败 → 拒绝启动 ✓。c=1/d=1 时 a=t 平凡成立（无需断言）。双射证明（混合进制唯一性）未改，维持 PASS。

## 3. 新增破坏（3 条 S3 残留，无新 S1/S2）

| # | severity | 位置 | 问题 | 建议 |
| --- | --- | --- | --- | --- |
| N-S3-1 | S3 | 5.4 §2 ASCII L59 | 「EagleSpeculator / MTPSpeculator（薄壳：**只管 load_draft_model**）」——S2-5 修复后 MTP 实有 5 个 override，此 ASCII 行未同步，与修正后的继承链表 L37 / 命题 1 L41 矛盾 | 改「薄壳：权重加载 + 输入准备 hook（MTP 另 4 hook，见表）」或「只管 load_draft_model 与 index 共享 hook」 |
| N-S3-2 | S3 | 5.4 总结表 L316 | 「命题 1：薄壳**只 override 权重加载**」——压缩复述未同步 S2-5，与修正后命题 1（MTP 另 override 四 hook）矛盾 | 改「薄壳只 override 权重加载与输入准备 hook（不含 propose），propose 语义由第 4 层决定」 |
| N-S3-3 | S3 | 5.3 总结表 L382 | kv_hints 消费链顺序「Request L85 → **ReqContext L109 → connector L172**」与修正后正文 L301（Request → connector L172 → ReqContext L109）相反。代码核实：offloading/scheduler.py `_create_req_context`（L160）读 req.kv_hints 构造 ReqContext（L172 `kv_hints=req.kv_hints`）→ 数据流是 Request→connector→ReqContext，正文正确、总结行顺序陈旧 | 总结行改「Request L85 → connector L172 → ReqContext L109」与正文一致 |

三条均为「正文已修对、压缩复述行漏同步」的内部不一致（R2-12 总结回指 / ASCII 与正文一致性），一行可修，归下一波 S3 清零，不阻塞本 batch S2 关闭。

## 4. 抽查 S3 + 跨章核验（全过）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| 6.3 cp/MoE L389 断言（S3-2） | ✓ | parallel_hook L377 gate / L389 assert tp%attn_dp==0；cp gate L135-141 / MoE L153(≤)/L159(==) 逐行核；命题 1 反例点名 L389（L29）+ 总结 L207 补 L389 |
| 6.3 checkpoint_engine_worker L15-16（S3-4） | ✓ | worker L15-16「Checkpoint-engine integration for SGLang…via IPC for checkpoint-engine compatibility」逐字；__init__.py L1-5 确为另一段（「Checkpoint engine module…」）；章 L173 锚点名 worker |
| 6.2 表 1.097/10.26 精度（S3-1） | ✓ | @5.13 重算 F=1.097/1.195/1.585/2.000/1.641、临界 T_normal=10.26；表头精度注（精确 5.1265，舍入差≤0.1%）；练习 1 锚 1.097/1.585/1.641 同步 |
| _advance_pending_grammar L2359（S3-3） | ✓ | scheduler.py def L2359（L2360-2365 docstring）；6.2 三处（L56/L165/L321）全改 L2359 |
| 5.3 kv_hints 全路径 | ✓ 正文 | distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py L172 存在且 `kv_hints=req.kv_hints`；request.py L85 / kv_offload/base.py L109 ✓；正文 L301 三跳顺序正确（总结行残留见 N-S3-3） |
| 5.3 命题 2 修复（S3-4） | ✓ | 外借 gloss 改「_unpin_pages 释放 resident 块回池复用」（coordinator.py L395-407 docstring「Release the pages' resident blocks to the pool」/ L402 num_unpinnable=num_resident−ACTIVE_TAIL_PAGES）；types.py L6-8 注释逐字；「留一页余量」明标本书解释 |
| 5.3 命题 3 修复（S3-5） | ✓ | 前提补「步间不重叠（同步步骤语义）」L319；「稳态步时」改「单步 makespan」+ 口径注；前提失效三方向（补步间重叠→(C+M)/max(C,M)，归 5.2 §8）L325；总结 L383 / 练习 3a/c L391 同步 |
| 5.2 S3-1/2/3/7/12 | ✓ | AsyncScheduler「调度侧 override _update_after_schedule（另 __init__ 设 decode_stagger，L11-16）」L193（tarball async_scheduler L11-16 ✓）；trace 首句标通用算法口径 + simple hybrid 一轮 L165；full fine-grained Phase 2（L931-955 ✓）/ SWA 左缘 fallback（L1223-1235 ✓）边界注；H→H_kv 全章无裸 $H$ 残留；5.1 命题 1 S_full 补「文本口径；mm 另含 mm_features/prompt_embeds 非 Θ(T_p) 项」L158 |
| 5.4 S3-8/10/11 | ✓ | A→n_adp / A_hot→n_hot（9 处 n_adp，无 A_hot 残留）；xgrammar「config 默认 auto（structured_outputs.py **L22** `backend="auto"`，docstring「With "auto", we will make opinionated choices」）→事实默认 xgrammar」L87/92/298/348（writer L22 实测正确，evaluator 原引 L21 偏 1 行）；update_from_output L739→**L738**（core.py L738 调用起始行 ✓，与 5.1 L112 统一） |
| 6.2 S3-5 记号消歧 | ✓ | L218「此 B 沿代码 docstring 原记号=backlog，非批内 B；C 即 4.3 c_chunk 的 per-rank 值」；shortest_prefill_load docstring「The backlog B is the uncached prompt tokens…per DP attention rank」逐字核 ✓ |
| 符号 §4.3 全 7 章 | ✓ | B 专归 batch（5.4 LoRA 译 W_up/W_down，B/A 仅翻译注引论文原记号）；b=block size；bytes 不占字母（无 b=bytes）；H_kv（5.2 无裸 H）；f_hot 不占 ρ（5.3 L291 消歧注，ρ 仅 5.1/5.3 消歧注出现）；A=API server（5.1/5.3）vs n_adp（5.4）；6.2 B=backlog / 6.3 t=world 均章内声明（evaluator 已判可保留）；α/γ/κ/d_model/d_latent/d_rope 全 batch 一致 |
| lint 7/7 | ✓ | `lint-chapter.sh` 7×LINT OK；`lint-math-boundary.py` 7×exit 0（flanking/反转义/禁用宏/表格竖线/\lt 全过） |
| 篇幅（独立 CJK 计数） | ✓ | 保守口径（去 metadata/参考/块引用/代码/公式）：5.1=3682 / 5.2=4883 / 5.3=6273 / 5.4=5114 / 6.1=4543 / 6.2=5833 / 6.3=4577，均≥3500；6.2【重】5833≥5500 ✓；全量口径 4310/5403/7158/6041/5009/6351/5029 与 writer 报告一致 |
| 命题编号每章重启 | ✓ | 5.1=1 / 5.2=1 / 5.3=1-3 / 5.4=1-4 / 6.1=1 / 6.2=1-2 / 6.3=1-2 |
| 练习分级 | ✓ | 7 章难度非递减、≥1 题 [10]/[20]、无 [40]、每计算/代码题带校验锚、每章≥1 题证命题（5.1 练 4 / 5.2 练 4 / 5.3 练 2+3 / 5.4 练 2+3 / 6.1 练 2 / 6.2 练 2 / 6.3 练 3） |
| 无陈旧值 | ✓ | 无「2048 当前默认」（改教学预算）；98.1 仅 5.2 L84 带口径声明（无他章误引）；无 B-A LoRA（仅翻译注）；无「末层」；无 L1910/L245(LRU)/L18(ngram)/L2361/1.098/10.25/换出到 host/ρ_hot 残留 |
| 跨章 recap 忠实 | ✓ | 2.4 §1 L19 PCIe 64 GB/s 换算注存在（5.3 L293/391、5.4 L20/102「2.4 §1 口径」指针有效，S3-13）；3.4 §7 式 (4) τ_c=6/τ_m=4/τ_n=2→12/6=2×（5.3 命题 3 对照忠实）；2.2 §4 L113 峰值 49 ms（5.2 口径差声明忠实） |

## 5. Verification log（抽查项 × 来源 × 结论）

**方法**：re-reviewer 自取 codeload fresh tarball（不占 API 限额）：vllm main（53,567,931 B）/ sglang main（47,805,849 B），2026-11-23，`/tmp/iw3-fixrev/x/`。锚点用 /usr/bin/grep -F + sed 逐行核；数值 python3 -I 独立重算；CJK 独立计数；lint 重跑。

### 5.1 修正锚点逐项（fresh tarball，全精确命中）

| 锚点 | 源 | 结论 |
| --- | --- | --- |
| config/scheduler.py L40 DEFAULT_MAX_NUM_BATCHED_TOKENS=2048 + docstring L49-52 | sed | ✓ |
| engine/arg_utils.py L3002-3023（≥160GiB 双 16384 / H100 8192·16384 / 其余 8192·2048） | sed/grep | ✓ |
| kv_cache_manager.py L544 full_num_tokens=min(num_tokens,max_model_len) | sed | ✓ |
| request.py L338-339 num_tokens=len(_all_token_ids) / L356-360 get_max_cache_hit_length | sed | ✓ |
| config/scheduler.py L226-230 full input sequence docstring / L232 watermark=0.0 | sed | ✓ |
| single_type L954 hit_length-=hit_length%alignment_tokens（full）/ L1228-1233 SWA pop 对齐 / L931-955 Phase 2 / L1223-1235 左缘 fallback | sed/grep | ✓（writer L954 正确，evaluator L957 笔误） |
| kv_cache_coordinator.py L704 不动点 / L766 alignment_tokens 传参 / L781-785 退出+simple hybrid / L789-799 截断 | sed | ✓ |
| uniproc_executor.py L161 ExecutorWithExternalLauncher(UniProcExecutor)（全 repo 唯一） | grep | ✓ |
| gpu_worker.py L1914 init_worker_distributed_environment / L1910 elastic_ep_execute | sed | ✓ |
| lora/worker_manager.py L276 LRUCacheWorkerLoRAManager | grep | ✓ |
| ngram/speculator.py L20 @triton.jit / L21 def _ngram_scan_kernel | sed | ✓ |
| mla/cpu_mla.py L71-72 supports_block_size / docstring L25-26 / 无 get_preferred；sparse_swa L166-167=256 | sed/grep | ✓ |
| mtp/speculator.py def 15/42/48/55/61 + 类属性 13 + 赋值 34 + hook 体 46/53/59/63；eagle def 13（18 行） | cat -n/grep | ✓ |
| 继承链 BaseSpeculator 74 / DraftModelSpeculator 165 / AutoRegressive 24 / TargetDependentAR 23 propose 103 / load_eagle_model 115 / Ngram 151 propose 213 | grep | ✓ |
| config/structured_outputs.py L22 backend="auto" + docstring | sed | ✓ |
| core.py L738 update_from_output 调用起始 | sed | ✓ |
| offloading/scheduler.py L160 _create_req_context / L172 kv_hints=req.kv_hints；request.py L85；kv_offload/base.py L109 | sed/grep | ✓ |
| hisparse coordinator.py L395 _unpin_pages / L402 num_unpinnable；types.py L6-8 ACTIVE_TAIL_PAGES=2 | sed | ✓ |
| async_scheduler.py L10 class / L11-16 __init__ decode_stagger / L18 _update_after_schedule | sed | ✓ |
| prefill_delayer.py L177-181 force-allow / L230-231 安全阀注释 / L275-278 slot / L281-282 注释 / L252-271 queue / L95 opt-in / L26 tracker / L76 PrefillDelayer / L384 SinglePassExecutor / L90-91 5000ms | sed/grep | ✓ |
| batch_result_processor.py def L1151 / sync L1157；scheduler.py L5914 copy_stream.wait / L5516 forward_stream.wait / L4349 stash / L2359 _advance_pending_grammar / L2329 注释 | sed/grep | ✓ |
| radix_cache.py L509-512 free_segment / L369 match_prefix / L684 _split_node（L691 lock_ref 继承、无 _update_leaf_status）/ L589 inc / L797 _update_leaf_status 三合取 / L559 evict | sed/grep | ✓ |
| common.py L230-231 protect_prefix / L420-421 注释；schedule_batch.py L282-284 KV 所有权注释 | sed | ✓ |
| parallel_hook.py L377 gate / L389 assert / L135-141 cp gate / L153/L159 MoE；runtime_context L199 attn_dp_enabled_of | sed/grep | ✓ |
| checkpoint_engine_worker.py L15-16 docstring；__init__.py L1-5 另一段 | sed | ✓ |
| 6.2 event_loop_overlap 2266 / _apply_war_barrier 2207 / should_process_last_batch_early 2338 / launch_batch_sample_if_needed 5928 / FutureMap 309 / resolve_forward_inputs 127；shortest_prefill_load docstring；environ L708-709 0.7/0.14 | grep/sed | ✓ |

### 5.2 数值独立重算（python3 -I，全命中）

S2-3 三 ms 值（49.07/98.14/34.94）+ 2× 口径；S2-4 轮界 65/首轮缩 63（10 组 ℓ₀ 边界 + 4095 逐轮 trace）；6.2 F 表 @5.13 全 5 行 + 临界 10.26 + pacing 4.13 + 精确 5.1265；6.2 §7 shortest_prefill_load H=D=8192/D_start=16384/B_full=24576 + 表 4 行 + 练习 4 值 724/1448/无 cap；6.1 free 0/3；5.4 LoRA 1,409,024/45.09M/90.2MB/0.56%/0.60%/0.78%/0.50% + launch 44800/448→224/2.24ms/14.8% + 驻留 0.72/9.02/90.2GB + sleep 23.9→80/3.347× + bitmask 1MiB/16KiB/16.4μs；5.3 DBO 8ms/1.25×/4/3 + HiSparse 0.78%/45MiB/3.02GB/386.5GB/0.369ms + TP 512/8KiB/385/321μs/2.55%/6.67%/16.7%；6.3 κL=70272/274.5MiB/17.16/2.14GiB/1.61GiB/34.5/276ms/549MiB/11.5ms/rank13 回代 13/aux 6.25e-6。

### 5.3 覆盖率声明

- S2 修正锚点：14 条 finding 涉及的全部新增/改动 file:line 锚点逐项对 fresh tarball 核（上表 ~40 项），全精确命中；writer 自纠 L954（vs evaluator L957）/ structured_outputs L22（vs L21）实测正确。
- 改动证明：5.2 终止界 / 6.2 pacing / 6.1 free_segment / 5.4 MTP 枚举 / 6.3 c=1 分支独立重推全 PASS。
- 抽查 S3 + 跨章：§4 表逐项；符号 §4.3 全 7 章扫；lint 7/7；CJK 独立计数；命题编号/练习分级；陈旧值 grep 清零。
- 未覆盖（声明）：punica/S-LoRA tarball 本轮未重取（5.4 坑注锚点 bgmv 166/sgmv_cutlass 231/sgmv 319/dispatch_bgmv 76/135 沿 evaluator P5 已核口径，fix round 未改这些锚点）；blog/论文引文沿 evaluator P5/P6 一手核口径（fix round 未改引文）；5.3 §9 / 5.4 §8 对照表未来章指针按 spec 大纲意图核。

## 6. Out-of-batch 遗留（确认 pending，不计本 batch 破坏）

| 项 | 通道 | 现状 |
| --- | --- | --- |
| S3-9 references.md 被引回写（nvidia-datasheet +5.1-5.4 / sglang-repo +5.3,5.4 / deepinfer-doc / mooncake / nixl +5.3 / outlines +5.4） | research pass / closeout | 仍 pending（核 nvidia-datasheet 被引=1.2…4.4，无 5.x）；不阻塞章节关闭 |
| ERR-1 3.4 L7/L99「SGLang 编译归 6.2」→6.3 | W2 errata（controller） | 仍「6.2」（sed 核）；writer D 未触碰 3.4，正确 |
| P5 advisory 1.4 §3 sampling 旧路径 v1/sample/ | errata（controller） | W1 章，不在本 batch；5.4 §7 坑注已按新路径回链 |
| research notes ⟨verify⟩ prefill_delayer「KV token usage 低水位触发」措辞 | research pass | notes L22 仍旧措辞；章已按代码改对（P6-S2-1），笔记更正归 research pass |

## 7. 修复与复审要求

- **S2 14/14 关闭**（锚点类已对 fresh tarball 回归，证明类已独立重推）。本 batch 无 S1，不触发 S1 强制复审通道。
- **新增 3 条 S3 残留**（N-S3-1/2/3：5.4 ASCII L59 + 总结 L316 MTP 措辞、5.3 总结 L382 kv_hints 顺序）累计至下一波 S3 清零，一行可修，不阻塞 S2 关闭。
- writer 反驳通道：附证据（源码引句/行号）复核，复核维持才强制修复。
- out-of-batch 遗留（§6）归各自通道，不阻塞 P5/P6 章节关闭。
