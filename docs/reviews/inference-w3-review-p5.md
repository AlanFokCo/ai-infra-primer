# Inference W3 审查报告 — P5 vLLM 走读（5.1 / 5.2 / 5.3 / 5.4）

- 日期：2026-11-23
- 审查人：fresh-context adversarial evaluator（不继承写作上下文）
- 对象：`inference/05-vllm/{01-vllm-architecture,02-vllm-scheduler-kv,03-vllm-distributed,04-vllm-features}.md`
- 协议：`.superpowers/sdd/inference-w1/evaluator-brief.md` 三任务 + W3 dispatch 特化（代码锚点直核 repo、命题重推、两层纪律、writer concerns 裁定）
- 代码事实源：**evaluator 自取 fresh codeload tarball**（vllm main / punica master / S-LoRA main / sglang main，2026-11-23，/tmp/iw3-eval-p5/x/），不信任 writer 报告与 research notes 行号；23 项文件行数声明与 notes 快照逐项一致（同一 commit 口径），行号分歧均判为 notes/writer 笔误而非版本漂移
- 结论：**S1 = 0 / S2 = 11 / S3 = 13**。无致命事实错误；命题主体全部成立（5.2 命题 1 终止界有一处对齐前提缺口，S2-4）；11 条 S2 中 6 条为锚点行号/文件归属错、3 条为语义/口径事实错、1 条证明骨架缺口、1 条符号约定冲突，全部可一段内修复。含 S2 章修复后须复审关闭。

## 1. Findings 表

### S2（必须修复）

| # | 位置 | 断言（章节原文） | 证据（一手） | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 5.2 §1 章脊（L30） | 「token 预算 max_num_batched_tokens = 2048（**当前默认**，config/scheduler.py L40）」 | config/scheduler.py L40 `DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048` 是 dataclass fallback，L49-52 docstring 逐字「The default value here is mainly for convenience when testing. In real usage, this should be set in EngineArgs.create_engine_config」；engine/arg_utils.py L3002-3023：H100/H200（≥70 GiB 非 A100）档 **OPENAI_API_SERVER=8192 / LLM_CLASS=16384**（≥160 GiB 档双 16384，其余档 8192/2048） | 2048 改标「教学预算（SchedulerConfig dataclass fallback，L40）」；补一句引擎实际默认按硬件 × usage context 分档（H100 在线 8192 / 离线 16384，arg_utils.py）。章内算例用 2048 不受影响（教学值已声明） |
| S2-2 | 5.2 §4 watermark 段（L86） | 「full_sequence_must_fit……准入前检查**整条序列（含未来生成）**装得下」 | kv_cache_manager.py L542-547：`full_num_tokens = min(request.num_tokens, self.max_model_len)`；request.py L338-339 `num_tokens = len(_all_token_ids)` = prompt + **已生成**，不含未来 max_tokens；config/scheduler.py L226-230 docstring「full **input** sequence length……rather than only checking the first chunk」 | 删「含未来生成」，改「新请求 = 全 prompt 长、恢复请求 = prompt + 已生成 token；不含未来生成」 |
| S2-3 | 5.2 §4 抢占成本代入（L82） | 「命中节省 = 4.86×10¹³（**2.2 §4，98.1 ms**）」 | 2.2 §4（L113）原文：48.6 TFLOP「按 H100 bf16 dense **989.5** TFLOPS 理想折算省约 **49 ms**」；2.2 全章无 494.75 / 98.1。98.1 ms 是本章 F_eff=494.75 的折算，与 2.2 峰值口径差 2×，未声明 | 括注改「FLOPs 见 2.2 §4；2.2 原文按峰值折算 49 ms，本章按 F_eff=494.75 折算 98.1 ms（口径差 2×）」。133−98.1=34.9 ms 的本章账本身正确 |
| S2-4 | 5.2 §6 命题 1 证明（L161）+ 练习 4(a) | 「hit_length 是非负整数**且对齐到 alignment_tokens**，故至多 ⌊ℓ₀/a⌋+1 轮终止」 | 初始 ℓ₀ = max_cache_hit_length = min(num_tokens−1, …)（request.py L357-361 get_max_cache_hit_length），**一般不对齐**（prompt 4096 → ℓ₀=4095，a=64）；首轮 full manager 返回 4032（对齐），缩量 63 < a。对齐性只对**各 manager 返回值**成立（single_type L957 `hit_length -= hit_length % alignment_tokens`），不对初值成立。最坏轮数 ⌊ℓ₀/a⌋+2 | 界改 ⌊ℓ₀/a⌋+2（或加前提「ℓ₀ 已对齐」并说明初值 num_tokens−1 一般不对齐、首轮缩量可 <a）；练习 4(a) 校验锚「每轮严格减 ≥a」同步改「首轮减 ≥1、后续每轮减 ≥a」。命题的终止性与一致性结论不受影响（已对代码验证成立） |
| S2-5 | 5.4 §2 命题 1 证明第一步 + 继承链表薄壳行 + 练习 4(a) 校验锚 | 「MTP override 集合 = {load_draft_model, set_prefill_rows 绑定, share_mtp_topk_indices}」「只 override 权重加载（MTP 另有输入绑定）」 | mtp/speculator.py（63 行）实际 override **5 个方法**：load_draft_model + `on_prefill_begin` / `on_prefill_end` / `on_multi_step_decode_begin` / `on_multi_step_decode_end`（share_mtp_topk_indices 是类属性，在 load_draft_model 内赋值；四 hook 做 skip_topk 切换与 compact_topk_indices，均为 index 共享/输入准备）。EAGLE 18 行只 override load_draft_model ✓ | 证明第一步与表格补全四 hook（注明全部属输入准备/index 共享、不含 propose，结论不变）；练习 4(a) 锚改「EAGLE 只 override 权重加载；MTP 另 override 四个 prefill/decode 生命周期 hook（index 共享）」 |
| S2-6 | 5.3 §1 执行链表（L24） | ray_executor_v2.py 行「另有 ExecutorWithExternalLauncher L161」 | 该类在 **uniproc_executor.py L161**（`class ExecutorWithExternalLauncher(UniProcExecutor)`），不在 ray_executor_v2.py（全 repo grep 唯一命中）；继承对象是 UniProcExecutor 非 MultiprocExecutor | 移到 uniproc 行或独立行，标 uniproc_executor.py L161（继承 UniProcExecutor）。research notes 该行未标文件，writer 归属错 |
| S2-7 | 5.3 §1 表 + ASCII 图（L25/L45） | 「init_worker_distributed_environment L1910」 | gpu_worker.py 实际 **L1914**；L1910 是 `def elastic_ep_execute`（另一函数） | 行号改 L1914（两处） |
| S2-8 | 5.4 §5 adapter 生命周期（L170） | 「LRUCacheWorkerLoRAManager L245」 | lora/worker_manager.py 实际 **L276**；L245 是另一方法内 `raise RuntimeError(` | 行号改 L276 |
| S2-9 | 5.4 §2 命题 1 边界反例（L43） | 「Triton _ngram_scan_kernel L18」 | ngram/speculator.py：L18 空行，@triton.jit L20，`def _ngram_scan_kernel` **L21**（MIN_N/MAX_N constexpr L30-31 ✓） | 行号改 L21 |
| S2-10 | 5.3 §5 block size 偏好段（L172） | 「backend 可以覆盖：mla/sparse_swa.py 的 get_preferred_block_size 返回 256（L166-167）、**mla/cpu_mla.py 返回 16**」 | sparse_swa.py L166-167 ✓；但 cpu_mla.py **无 get_preferred_block_size**——16 来自 `supports_block_size`（L71-72，`block_size is None or block_size == 16` 硬约束）+ 平台层强制（模块 docstring L25-26「the platform layer forces block_size=16 for MLA models」） | 改「cpu_mla.py 经 supports_block_size 只接受 16（平台层强制，非偏好覆盖）」；数值 16 本身正确（2.1 已交付同值） |
| S2-11 | 5.4 §5（L136 公式段 / L189 差异点） | 「$\Delta W = BA$（$B$ 为 $d_{\mathrm{out}} \times r$、$A$ 为 $r \times d_{\mathrm{in}}$），前向 $y = Wx + B(Ax)$」及「分段 gather 不同 adapter 的 $A$ / $B$」 | spec §4.3 消歧硬规则「**$B$ 专归 batch**」；同章式 (1) $B$ = 批内请求数、§1 已声明 $A$ = adapter 数——$B$ 双重赋值（batch vs LoRA 上投影矩阵）、$A$ 双重赋值（adapter 数 vs 下投影矩阵），且沿用了 LoRA 论文字母未翻译（§4.3 翻译硬规则） | 数学记号译作 $U$（up）/ $D$（down）或 $W_{\mathrm{up}}/W_{\mathrm{down}}$ + 一句口径注「LoRA 论文记号 B/A，按 §4.3 译出避免占 B」；教学代码 snippet 的形参名 A/B 属代码标识符可保留，但正文数学与「$A$/$B$」指称须同步换 |

### S3（累计，下一波复审统一清零）

| # | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| S3-1 | 5.2 §8（L191） | 「AsyncScheduler……**只** override _update_after_schedule」——还 override `__init__`（decode_stagger，PP decode microbatch 节奏，async_scheduler.py L11-16） | 改「调度侧 override 是 _update_after_schedule（另 __init__ 设 PP decode_stagger，归 8.1 PP）」 |
| S3-2 | 5.2 §6 边界反例（L163） | 两轮 trace（「触发第二轮……SWA 重查」）是通用算法口径；本例 full+SWA 恰为 simple hybrid，代码实际一轮 break + 收尾截断（章末已给优化说明 + 练习 4(d) 闭环，但 trace 处未标口径） | trace 首句加「按通用不动点算法走读；本例在代码中走 is_simple_hybrid 一轮路径（见下）」 |
| S3-3 | 5.2 §6 命中算法描述 | FullAttentionManager 描述略 fine-grained Phase 2（alignment_tokens < block_size 时探入首个非满块内部 hash 边界，single_type L931-955）；SWA 描述略左缘短前缀 fallback（连续段 < 需求块数仍计命中，L1224-1230） | 各补一句边界注（机制摘录已声明简化，非错误） |
| S3-4 | 5.3 §7 命题 2（L295） | 「page 外借（**换出到 host**）」——代码语义是 `_unpin_pages` 释放 resident 块回池（coordinator.py L395-407；types.py L6-7 注释「stay pinned so a page written by an in-flight step is never handed out under it」）；eager mirror 下 host 副本已在，unpin = 放弃 GPU 驻留 | 括注改「释放 resident 块回池复用」；命题/证明在已声明的注释语义下成立，教学口径标注诚实 |
| S3-5 | 5.3 §8 命题 3（L319） | 「稳态步时」隐含「步边界不跨步重叠（同步步骤语义）」前提未显式；若与 async 跨步流水组合，上界可放宽至 (C+M)/max(C,M)。4/3 推导本身正确（单步 makespan 最优） | 前提列补一句「步间不重叠；跨步流水属 async scheduling 另账（5.2 §8）」 |
| S3-6 | 5.3 §7 式 (4) | $\rho_{\mathrm{hot}}$ 占用 $\rho$（§4.3 表 $\rho$ = 利用率）；已有下标 + 符号表声明 | 改 $f_{\mathrm{hot}}$ 或仿 5.1 $\eta_{\mathrm{gpu}}$ 加一句消歧注 |
| S3-7 | 5.2 §6 式 (1) | $H$ 表 KV head 数（§4.3 表为 $H_{kv}$）；章内局部声明一致 | 统一 $H_{kv}$（或保留局部声明并在符号表注明与 §4.3 的对应） |
| S3-8 | 5.4 §1 | $A$ = adapter 数与 5.1/5.3 的 $A$ = API server 数跨章同字母异义（均局部声明，不违 §4.3 但 Part 内易混） | 改 $n_{\mathrm{adp}}$（与 S2-11 一并修） |
| S3-9 | references.md 被引回写 | 6 条目缺 W3 章号：nvidia-datasheet（+5.1,5.2,5.3,5.4）、sglang-repo（+5.3,5.4）、deepinfer-doc（+5.3）、mooncake（+5.3）、nixl（+5.3）、outlines（+5.4）。research notes「sglang-repo 被引已含 5.x」声称与条目实际不符 | 按写入规则回写被引章节（research pass 职责，记此备查） |
| S3-10 | 5.4 §3 backend 表 | 「xgrammar（默认 backend）」——config 默认字面值是 "auto"（config/structured_outputs.py L21，docstring「opinionated choices」）；xgrammar 为 auto 解析的事实默认（W3 research 口径，references.md xgrammar 条目同） | 加半句「config 默认 auto，事实默认解析到 xgrammar」 |
| S3-11 | 5.1 §3 摘录注释 vs 5.4 §3 | update_from_output 调用行 5.1 标 L738、5.4 标 L739（调用跨 L738-739，均不算错） | 统一 L738（调用起始行） |
| S3-12 | 5.1 §5 命题 1 证明 | $S_{\mathrm{full}}(T_p)=\Theta(T_p)$ 为纯文本口径——mm_features / prompt_embeds 另加非 $\Theta(T_p)$ 项（仍只发一次，式 (2) 上界不受影响） | 加一句「文本口径；mm 请求另含 mm_features 项，仍一次性发送」 |
| S3-13 | 5.3 §7 / 练习 3(d) | PCIe 单向 64 GB/s 无就地出处指针（5.4 §1 已标「2.4 §1 口径」，2.4 §1 有完整换算注） | 补「2.4 §1 口径」 |

**Advisory（跨波，不计 P5 findings）**：1.4 §3（L55 / L208）仍把 sampling 实现锚 `vllm/v1/sample/`——5.4 §7 坑注已按新路径（gpu/sample/sampler.py L42）处理回链，但 1.4 自身指针句需经 errata 通道收窄（W1 章，不在本 batch 修改范围）。

## 2. 命题重推导与数学核验（独立重推，不信任 writer）

| 命题 | 结论 | 核验要点 |
| --- | --- | --- |
| **5.1 命题 1**（diff 控制面字节上界） | **PASS** | 对 output.py L41-55 / L139-147 逐字段核：NewRequestData 含 prompt_token_ids + prefill_token_ids + block_ids（Θ(T_p)，文本口径见 S3-12）；CachedRequestData = req_ids / new_token_ids（PP-only，注释逐字核）/ new_block_ids / num_computed_tokens / num_output_tokens，无一随已缓存 T 增长 → s_diff=O(1) ✓；两类互斥穷尽 ✓；边界反例「恢复按新请求重发全量」= scheduler.py L1606-1607 注释逐字 ✓。数值例：64×(4096×4+64×4+100)=1.02 MiB、64×32=2 KiB、523×、204/0.4 MiB/s ✓；练习 2 的 4.1 MiB / 4 KiB / 1043× / 815/0.8 MiB/s / 32.6 KiB 全部重算一致 ✓ |
| **5.2 命题 1**（hybrid 不动点收敛与一致性） | **PASS（终止界有一处前提缺口，S2-4）** | 对 kv_cache_coordinator.py L704-800 全读：docstring 逐字 ✓；单调减整数列必终止 ✓；退出轮 curr==hit_length 且各被查询组均接受 ✓；full 组向下封闭（链 hash，single_type L921-927 注释逐字）+ 收尾 cdiv 截断（L789-799）✓；is_simple_hybrid = 恰两组且 full 在前（L738-741 + verify_and_split 排序 L681-684）✓；alignment 对 manager 返回值成立（L957）但对初值 ℓ₀=num_tokens−1 不成立 → 轮界 ⌊ℓ₀/a⌋+1 最坏差 1（S2-4）。正例/边界反例（3072→2944，⌈127/64⌉=2 连续块）重算 ✓ |
| 5.3 命题 1（serving 归约位点减半） | PASS | training 5.1 式 (5)(6) f/g 共轭原文核 ✓；前向只 g 触发、每层 2 位点 ✓；SP 换原语不换位点 ✓；vocab embedding 额外 1 次前向归约 = training 5.1 §9 原文 ✓。式 (2)(3) 全部数值重算：512 KiB / 8 KiB / 32 MiB、1.75×524288/9e11=1.02 μs、385/321 μs、16.7%、2.55%/6.67%、prefill 65.2 μs→4.50 ms→6.8% ✓；τ_hop/1/BW 符号翻译符合 §4.3 ✓ |
| 5.3 命题 2（active tail 不外借） | PASS（教学口径已诚实声明；gloss 见 S3-4） | types.py L8 ACTIVE_TAIL_PAGES=2 + L6-7 注释逐字 ✓；coordinator.py L402 `num_unpinnable = num_resident − 2` ✓；写入 ⊆ 末尾 ⌈(γ+1)/b⌉ ≤ 2 页 → 与外借集合不相交 ✓；大 chunk prefill 边界反例诚实 ✓；「2 而非 1 的账面理由」明确标为推导非官方 ✓ |
| 5.3 命题 3（DBO 4/3 上界） | PASS（隐前提见 S3-5） | 独立重推：半批时序 c1[0,C/2]、m1[C/2,C/2+M/2]、c2[C/2,C]、m2[max(C,C/2+M/2),…] → makespan = max(C,M)+min(C,M)/2，且为该约束下单步最优；(x+y)/(x+y/2) ≤ 4/3，C=M 取等 ✓；C=6/M=4→8 ms/1.25×、C=M=5→7.5 ms/1.333× ✓；对照 3.4 式 (4) NanoFlow 12/6=2×（τ_c=6/τ_m=4/τ_n=2 教学参数原文核）✓；dbo.md 机制口径（DP+EP all-to-all / TP all-reduce-overlap / 两线程 ping-pong / FusedMoEModularKernel yield）逐字核 ✓ |
| 5.4 命题 1（薄壳不改 propose 语义） | 结论 PASS；证明枚举不全（S2-5） | eagle 18 行只 override load_draft_model ✓；mtp 63 行另 override 四 hook（S2-5）；两壳均无 propose → MRO 落到 TargetDependentARSpeculator.propose L103 ✓；ngram 自实现 propose L213 不在链上 ✓ |
| 5.4 命题 2（SGMV launch 与 base 流量账） | PASS | 2·n_site vs 2·A·n_site 计数 ✓；n_site=224、A=100 → 44800×5 μs=224 ms vs 448×5 μs=2.24 ms、14.8% ✓；base 共享论证 recap 1.2 忠实 ✓；A=1 退化边界 ✓；fused 入口（add_lora_w13 L528 / add_lora_fused_moe L464）位点计数口径注诚实 ✓ |
| 5.4 命题 3（extra-key 共享边界） | PASS | generate_block_hash_extra_keys L627-660：lora + mm + cache_salt + prompt_embeds 四源逐字核 ✓；hash 确定性 + 碰撞概率口径已诚实声明 ✓；mm 分支与 blog 第 7 点一致 ✓ |
| 5.4 命题 4（wake_up KV 失效边界） | PASS（条件式，level 细节归 9.3 ✓） | cumem.py L261-300：offload_tags 外一律 discard、释放物理页保留虚拟地址 ✓；Worker.sleep L351 / wake_up L389（tags 选择性恢复）✓；「若释放了 kv_cache 池物理页则旧 KV 不可信」条件式成立 ✓；weights 按 level 可恢复（level 1 offload 到 host）与代码一致 ✓；colocate 账 80−16.1−40=23.9、80/23.9=3.35× ✓ |

**其余数值全量重算**（全部一致）：5.2 章脊 κ=4096 B / κL=128 KiB / page GQA 256 KiB / MLA 72 KiB（d_K=576=512+64，与 1.2/2.1/3.1 几何一致）/ 6144 块 / 384K token / 满窗并发 3 / watermark 61 块 488 MiB / chunk 1984×4+256=8192 共 5 步 / recompute 6.58e13→133 ms、节省 4.86e13、差 1.73e13→34.9 ms（F_eff 口径内自洽）/ SWA 帽 ⌈127/64⌉+1=3 / 练习 1-2 全部锚值；5.3 进程数 6/17/10/25 / HiSparse ρ_hot=0.78% / 47.2 MB(45 MiB) / 6.04 GB / 3.02 GB / 386.5 GB / 22.5 MiB→0.37 ms；5.4 bitmask 1 MiB / 16 KiB / 16.4 μs / LoRA 每层 1,409,024 / 45.09 M / 90.2 MB / 0.56% / 0.60% / 0.78% / 0.50% / 驻留 0.72/9.02/90.2 GB。章脊 recap 值（15.1 ms / 4.81 ms / 66.5 ms / 16.1 GB / N=7.50e9 / 330 kernel×5 μs=1.65 ms）与 1.2 / 3.4 原文逐项一致。

## 3. 两层走读纪律（spec §4.2）

- **5.2 ↔ 2.1/2.2/4.x**：PASS。§1 显式 recap 2.1 命题 1/2 + 机制片段清单并声明「本章不重推」；式 (1)(2) 是代码级增量（page_size_bytes 属性 / num_blocks 离散化）非机制重推；SWA 帽标「2.1 式 (9) 的代码精确形（recap，不重推）」；§4 抢占账明写「公式本体见 4.2 / 2.2，此处只做代码场景代入，不重推」；hash 链 recap 2.2 §2 不重推；COW 经 take_kv_cache_block_copies 接口登记 + 指引 2.1 §5。
- **5.3 ↔ 3.1/3.4/2.4**：PASS。§5 开头显式「3.1 已交付算子算法级……recap 一句 + 指引，不重推其公式与证明；增量只在引擎如何选与调」；max_num_splits 三态 recap 不重列；§6「融合 IO 账、launch 账、piecewise 切分语义与正确性命题 recap 3.4，不重推」；§7「2.4 已交付三层存储账、临界带宽与 SGLang HiSparse 机制」+ 换入带宽「recap 2.4 式 (3) 同形」。§3 f/g 只 recap 共轭定义，命题 1 是 spec 指定的 serving 侧增量（无 backward）非重推 training 账。
- **5.4 ↔ 4.4/4.5/1.4/2.4/training 8.5**：PASS。各节开头均有两层纪律声明；LoRA 账 / bitmask 账 / sleep 接口边界是 spec 指定增量；分布保持证明、E[N_tok]、加速比定理、grammar 原理、三层存储账、RL 训练侧编排全部 recap + 指针。
- 未发现任何 W1/W2 公式重推导。

## 4. Writer concerns 裁定（独立核实，不照单全收）

| 项 | 裁定 |
| --- | --- |
| V0 已移除 / blog alpha 历史化 | **成立**。engine/llm_engine.py = V1 alias（docstring 逐字）✓；envs.py 无 VLLM_USE_V1 ✓；blog 限制清单（log probs / PP / structured / spec / prometheus / LoRA + encoder-decoder / Mamba / embedding + Ampere）经 blog 全文一手核 ✓；5.1 坑注带历史限定 ✓ |
| hisparse = DSA/QSA local KV offload | **成立**。docs/design/hisparse.md（321 行，Status: experimental）+ HiSparseConfig docstring 逐字 ✓；三角色分工 / HMA 分配权 / host_pool_gib per-DP-replica / MLA replicated view / QSA(Qwen4Exp) BLNHC / 两条 startup concurrency bound / MultiConnector 组合全部对 doc 逐字核 ✓；8 文件 ✓ |
| kv_hints = orchestrator 版本化 hint 协议 | **成立**。protocol.py KvHintAction(action_id/action_type/action_version/payload) + KvHintsEnvelope(protocol_version/message_id/actions)，msgspec frozen Struct ✓；消费链 Request.kv_hints L85 → ReqContext L101/kv_hints L109 → offloading/scheduler.py L172 ✓；ROUTER_HINT_CAPABILITIES 在 tiering/kvcr/manager.py ✓ |
| DBO ≠ AsyncScheduler；TBO/SBO ≠ overlap scheduler | **成立**。5.3 §8 显式区分两机制 ✓；dbo.md 口径逐字核 ✓；SGLang batch_overlap/ = TBO/SBO 与 event_loop_overlap + FutureMap 的区分与 research notes 语义纠正 1 一致 ✓ |
| 教学口径标注（归约门阈值 / ShmRingBuffer / DBO 4/3 / ACTIVE_TAIL_PAGES） | **诚实**。§4「不引阈值数值」✓；ShmRingBuffer「机制口径，逐字段实现以源码为准」（单写多读 + per-reader flag + 背压语义与 shm_broadcast.py L280-330 docstring 一致）✓；DBO 4/3 为带显式理想前提的自推命题、未冒充官方 ✓；ACTIVE_TAIL_PAGES 推导明标「按常量与注释语义推导」✓ |
| 教学值声明（V=131072 / r=16 / KV 池 40 GB / τ_lat=5 μs） | **全部已声明**（5.4 §1 / 5.3 §1 章脊段）✓ |
| sleep level 1/2 细节 → 9.3，命题 4 条件式 | **成立**。命题 4 以「若 sleep 释放了 kv_cache tag 池物理页」为前提 ✓；level 逐 level 账显式归 9.3 ✓；与 Worker.sleep / cumem 代码语义一致 ✓ |
| MLA head_size=576（1.2/2.1 几何） | **成立**。5.2 §6 d_K=576=512+64、d_V=0、κ_MLA=1152 B（bf16）与 1.2 / 2.1 / 3.1 一致 ✓；MLAAttentionSpec head_size_v=0 代码核 ✓ |
| _MANAGER_KINDS def L3314 vs lookup L3335 | **成立**。L3314 映射表定义、L3335 get_kv_cache_kind 内查表，两处均精确 ✓（writer 对 research notes 单锚点的细化正确） |
| Punica master / sgmv_expand TODO / vLLM Triton lineage | **成立**。punica master tarball：sgmv L319 内 `raise NotImplementedError("TODO: sgmv_expand")` 逐字 ✓；bgmv L166 / sgmv_cutlass L231 ✓；csrc/sgmv/{sgmv.h,sgmv_cutlass.cu,sgmv_cutlass.cuh} + bgmv/ + sgmv_flashinfer/ ✓；vLLM lora_shrink_op L102 / lora_expand_op L24 双侧实现 ✓；S-LoRA dispatch_bgmv L76/L135 + mem_manager.py ✓ |
| spec_decode 薄壳继承链 | **成立**（MTP override 枚举不全另计 S2-5）。BaseSpeculator L74 → DraftModelSpeculator L165 → AutoRegressiveSpeculator L24 → TargetDependentARSpeculator L23 propose L103 全链精确 ✓；eagle 18 / mtp 63 行 ✓；load_eagle_model L115 ✓；目录扩展清单（dspark/dflash/dflash2/dfly/suffix/standalone_ar/multi_module_mtp/xpress/dspine/gemma4/lilicorr/fly.py）逐项存在 ✓ |
| Sampler 位置 / KV spec 类族 / interface+layout 在 v1/ 根 / watermarking ≠ KV watermark | **全部成立**。sampler.py L42 ✓；v1/sample/ 只剩 ops/ ✓；FullAttentionSpec/SlidingWindowSpec 全 repo 不存在、AttentionSpec L456 + KVCacheSpecKind L150（9 种 + UNKNOWN）+ manager 类族 ✓；两文件在 v1/ 根 ✓；KVCacheLayout = stride 排列枚举（逻辑形 [L,B,H,N,C]）✓；watermark config L232 default 0.0 ✓ |

## 5. 规范审查（STYLE_GUIDE / R2 / rubric）

- **lint**：4/4 `LINT OK`；`lint-math-boundary.py` 4/4 零违规（flanking / 反转义 / 禁用宏 / 表格竖线全过）。
- **篇幅**（正文 CJK，去代码/公式）：5.1 = 4285 / 5.2 = 5037 / 5.3 = 6950 / 5.4 = 5831，均 ≥3500 标准章线 ✓（P5 无推导重章标记）。
- **模板**：metadata 定位|前置|锚点 ✓；总结表（论断 + 回指列，R2-12）✓；练习 4 题/章 ✓；参考列表链接 ../references.md ✓；ASCII 图每章 ≥1（5.1×2 / 5.2×1 / 5.3×4 / 5.4×3）✓。
- **R2 逐条**：R2-1 节首设问/数值/事实 4 章全节通过；R2-2 章脊锁定 + 换口径声明（5.2 hybrid 例 / 5.3 §7 V4.1 / 5.4 §6 KV 池）✓；R2-3 每公式 300 字内全数字实例 ✓；R2-4 命题-证明骨架 10 个命题全带路线句 ✓；R2-5 ≥4 步推导路线句 ✓；R2-6 定义配例 + 边界反例 ✓；R2-7 口径注（η_gpu / 控制面 / 教学序列化 / τ_lat / V 圆整 / goodput 导出量 / 位点计数）✓；R2-8 判断句定义（persistent batch / CPU overhead / 通信组 / 执行链 / HiSparse / KV hint / shrink/expand / sleep mode）✓；R2-9 逻辑关节词 ✓；R2-10/11 提点/坑注块引用 ✓（坑注均含触发条件+现象+规避）；R2-12 总结回指 ✓。
- **习题分级**：格式 `N.（类型 · [难度]）` ✓；难度非递减 [10][20][30][30] ✓；≥1 题 [10]/[20] ✓；无 [40] ✓；每道计算/代码题带校验锚 ✓；每章 ≥1 题以正文命题为目标（5.1 练 4 / 5.2 练 4 / 5.3 练 2+3 / 5.4 练 2+3）✓；代码题「先写出预测」✓。
- **rubric A/B**：禁令词扫描零命中（「即其实例」为合法书面语，非口语「其实」）；无营销形容词、无语气松散词、无历史叙事（V0.6.0→V1 为 spec 5.1 大纲指定的机制谱系，blog 作证据）；无含糊量词；术语统一（prefill/decode/KV cache/block size 保留英文口径与 W1/W2 一致）。
- **命题编号**：每章从命题 1 起 ✓；跨章命题引用均带章号前缀（2.1 命题 2 / 3.4 命题 2 / 1.4 命题 2/5 / 4.4 命题 1）且目标命题存在 ✓。
- **引用白名单**：全部引用 key 在 references.md 存在且等级标注齐全（vllm / vllm-repo / vllm-v1-blog / nvidia-datasheet / flashinfer / sglang-repo / deepinfer-doc / mooncake / nixl / punica / s-lora / xgrammar / guidance / outlines）；arXiv ID lint 全过；被引回写缺 6 条目见 S3-9。nvidia-datasheet 不在 research notes W3 白名单枚举内，但条目系 W1 一手核实、数值为 1.3/2.4 忠实 recap——判引用合法（先收录后引用满足），白名单枚举遗漏记 research pass 备查。
- **跨章符号 §4.3**：B/b/κ/bytes/d_model/d_latent/d_rope/α/γ/t/p/e/d/𝒢 全 batch 一致 ✓；τ_hop/τ_lat 翻译声明 ✓；违规两处：5.4 LoRA 矩阵占 B/A（S2-11）、ρ_hot/H/A 的局部占用（S3-6/7/8）。

## 6. Verification log（抽查项 × 来源 × 结论 × 覆盖率）

**方法**：evaluator 自取 codeload fresh tarball（不占 API 限额）：vllm main（/tmp/iw3-eval-p5/x/vllm）、punica master、S-LoRA main、sglang main，2026-11-23。脚本 /tmp/iw3-eval-p5/scripts/check_anchors.py 对章节全部定义行锚点做 (file, line, symbol) 三元组机械核对；引句/注释/doc 口径用 /usr/bin/grep -F 逐字核对；数值全部独立重算；blog 与 arXiv 摘要一手 curl。

### 6.1 定义行锚点（492 项检查：468 精确命中 / 24 偏差逐项裁定）

| 章 | 检查数（约） | 精确命中 | 偏差裁定 |
| --- | --- | --- | --- |
| 5.1 | 125 | 121 | core.py L739（调用跨 738-739，S3-11）/ L1897（if 行，sleep 在 1898，范围内）/ core_client L127（if 行，raise 在 128）/ ray_executor_v2 L161（**错文件 → S2-6**）/ gpu_worker L1910（**错行 → S2-7**，5.3 亦引） |
| 5.2 | 175 | 166 | 9 项均为所引范围覆盖实际内容或 ±1-2 行构造内偏移（L829/837/1132/898/917/2406/308/595/106/743/789/706/203 逐项读源码裁定为范围语义正确） |
| 5.3 | 110 | 108 | sparse_swa L166（def 行，return 256 在 167，章引 L166-167 ✓）；gpu_worker L1910 同 S2-7 |
| 5.4 | 82 | 73 | ngram L18（**空行 → S2-9**）；worker_manager L245（**错行 → S2-8**）；speculative.py L113/122/123 是 Literal 类型别名非 class（行号精确、成员清单逐项展开核对一致，「枚举」为宽松措辞不究）；余为范围内偏移 |

另：**23/23 文件行数声明精确一致**（core.py 3177 / core_client 2255 / input_processor 692 / model_runner 2681 / scheduler 3993 / block_pool 936 / kv_cache_manager 935 / coordinator 904 / single_type 3377 / kv_cache_interface 963 / kv_cache_layout 58 / planning 1817 / config_builder 102 / encoder_cache 450 / parallel_state 2857 / all2all 1381 / partition_rules 74 / rejection_sampler 483 / eagle 18 / mtp 63 / kv_offload base 861 / loggers 1467 / hisparse.md 321）——与 research notes 同快照，行号分歧判笔误。

### 6.2 逐字引句 / 注释 / 官方 doc（约 50 项，全命中除已列 findings）

| 引句 | 来源 | 结论 |
| --- | --- | --- |
| 「There's no "decoding phase" nor "prefill phase"……num_tokens_with_spec」 | scheduler.py L705-707 | ✓ 逐字 |
| 「A resumed request is re-sent in full, like a new one……」 | scheduler.py L1606-1607 | ✓ 逐字（5.1 命题 1 边界反例） |
| 「Advance the number of computed tokens here allowing us to schedule the prefill request again immediately……」 | scheduler.py L1884-1886 | ✓ 逐字 |
| 「Its tokens are still delivered on return (dropping them would perturb spec-decode acceptance)……」 | scheduler.py L1857-1863 | ✓ 逐字 |
| 「In a priority queue, there is no concept of prepending to the front」 | request_queue.py L161-163 | ✓ 逐字 |
| 「When re-admitted, it is recomputed rather than swapped back from CPU memory. Prefix caching or a KV connector can restore part of its progress immediately.」 | docs/design/arch_overview.md L166-168 | ✓ 逐字 |
| 「while the GPU executes step N, the CPU schedules step N+1, using placeholders……」 | arch_overview.md L177-179 | ✓ 逐字 |
| Process Count Summary 表 + tp4=6 / tp2dp4=17 + 多对多 ZMQ + --api-server-count + VLLM_MEDIA_LOADING_THREAD_COUNT=8 | arch_overview.md L75 / L103-125 + envs.py L81 | ✓ 逐字/逐项 |
| 「LIFO reuse of non-cached blocks for better GPU locality」/「FIFO reuse of cached blocks for LRU eviction behavior」 | block_pool.py L829-831 | ✓ 逐字 |
| FreeKVCacheBlockQueue docstring（O(1) 中部摘除 / 零 Python 对象 / prev,next 指针 / LRU 在前 / more hash tokens 在前） | kv_cache_utils.py L248-266 | ✓ 逐字 |
| 「A missing block implies every later block misses too (chained hashes)」 | single_type_kv_cache_manager.py L921-922 | ✓ 逐字 |
| 「Each attention type either accepts the current candidate length or reduces it……bounded below by 0」 | kv_cache_coordinator.py L710-714 | ✓ 逐字 |
| 「free the blocks in reverse order so that the tail blocks are evicted first」 | kv_cache_manager.py L643-644 | ✓ 逐字 |
| 「Handles exactly two KV cache groups」+ commit 458e74 warning + Gemma-3-27b 52SWA+10full→7 组（末组 2+8 padding） | docs/design/hybrid_kv_cache_manager.md L4 / L225 / L111-121 | ✓ 逐字；代码多组支持（attention_groups 循环 + verify_and_split 无两组断言）→ 5.2 坑注成立 |
| max_cache_hit_length = prompt_length−1 + 整块重算 NOTE | kv_cache_manager.py L308-313 | ✓ 逐字 |
| allocate_slots 五段布局 comp/new_comp/ext_comp/new/lookahead + full_sequence_must_fit docstring | kv_cache_manager.py L426-447 / L441-444 | ✓（「含未来生成」断言错 → S2-2） |
| watermark 只加 WAITING/PREEMPTED + has_scheduled_reqs 注释 | kv_cache_manager.py L529-536 | ✓ 逐字 |
| 「Configuration for HiSparse DSA and QSA KV offloading」/ device_buffer_size 默认 1 top-k + 1 slack / eager_host_mirror=True | config/attention.py L21-37 | ✓ 逐字 |
| 「Neither worker-side object allocates or frees logical blocks. HMA provides……」/ host_pool_gib per-DP-replica / MLA replicated view / QSA Qwen4Exp BLNHC / 两条 concurrency bound / MultiConnector 组合 | docs/design/hisparse.md L3 / L16 / L31-46 / L53 / L61 / L28-29 | ✓ 逐字 |
| ACTIVE_TAIL_PAGES=2 + 「a page written by an in-flight step is never handed out under it」 | v1/hisparse/types.py L6-8；coordinator.py L402；layout.py L521 | ✓ 逐字（gloss S3-4） |
| KvHintAction / KvHintsEnvelope 字段 + frozen msgspec | v1/kv_hints/protocol.py L9-26 | ✓ 逐项 |
| DBO：DP+EP sparse all-to-all / TP all-reduce-overlap / 两线程 ping-pong / FusedMoEModularKernel yield / UBatchRunner+UBatchContext | docs/design/dbo.md L5-11 / L74 | ✓ 逐字 |
| ShmRingBuffer 单写多读 + per-reader flags + 写者等所有读者（背压） | shm_broadcast.py L280-330 docstring | ✓ 机制口径一致 |
| 「Scheduler-side manager for SimpleCPUOffloadConnector」 | v1/simple_kv_offload/manager.py L3 | ✓ 逐字 |
| 「can be extremely large (e.g., several GBs)」+ UVA 默认 + prefill_len≥prompt_len 注释 | gpu/states.py L32-51 | ✓ 逐字 |
| 「fulfilling the batch queue has a higher priority than getting model outputs」/「EngineCore waiting for work.」 | core.py L768-769 / L1855 | ✓ 逐字 |
| CompilationMode NONE=0/STOCK=1/DYNAMO=2/VLLM=3 + CUDAGraphMode 五值 + FULL_AND_PIECEWISE v1 default + piecewise 依赖 splitting_ops | config/compilation.py L38-51 / L165-174 / L744 / L761-767 | ✓ 逐项（与 3.4 W2 口径一致） |
| blog：~5ms / isolated EngineCore 逐字 / {request_id: num_tokens} / 1.7×(without multi-step) / 0% 命中 <1% / Persistent Batch / Llama 3.1 8B+3.3 70B ShareGPT / Qwen2-VL VisionArena / V0.6.0 ZeroMQ / alpha 限制清单 / Ampere | blog.vllm.ai/2025/01/27/v1-alpha-release.html（curl 全文） | ✓ 全部一手命中 |
| Punica 摘要：12× vs SOTA + 2ms/token + 单份 base + scheduler 整合 + 自链 repo | export.arxiv.org API 2310.18547 | ✓ 逐字 |
| S-LoRA 摘要：主存常驻 + Unified Paging（变 rank + 变长 KV）+ 新 TP + 异构 batching kernel + 数千 adapter + up to 4× vs HF PEFT/vLLM naive | export.arxiv.org API 2311.03285 | ✓ 逐字 |
| punica sgmv_expand NotImplementedError / bgmv L166 / sgmv_cutlass L231 / sgmv L319 / csrc 三件 | punica master tarball | ✓ |
| S-LoRA dispatch_bgmv L76/L135 / mem_manager.py / csrc/bgmv/ | S-LoRA main tarball | ✓ |

### 6.3 语义走读核验（命题 substrate，全文读）

scheduler.py schedule() 钳位链（L760-850：adaptive max(threshold, input_budget//num_eligible) ✓ / threshold 帽 ✓ / min 三道 ✓ / max_model_len ✓ / Mamba 对齐 ✓）、抢占循环（L890-970：victim PRIORITY max / FCFS running[-1] ✓、drop_stale_output=requires_kv_delivery ✓）、_preempt_request 全体（L1831-1877 ✓）、waiting 准入（L1041 not preempted_reqs ✓ / L1132-1149 prefix lookup ✓ / L1348 ✓ / L1596-1624 ✓）、乐观推进（L1879-1900 ✓）、stale drain（L2406-2410 ✓）；async_scheduler.py 全文（placeholder 公式 ✓ / [-1] 占位 ✓ / is_prefill_chunk 跳过 ✓ / __init__ override → S3-1）；coordinator 基类（LCM assert ✓ / 共享 BlockPool ✓ / 工厂三分支 ✓）+ Hybrid 全体（simple hybrid ✓ / full skip ✓ / 截断 ✓ / retains_longer_hit 略去属声明简化）；FullAttentionManager / SlidingWindowManager 命中算法全文（方向 ✓ / 连续块数 ✓ / null_block ✓ / 帽公式 cdiv(min(w−1+r+f, max_model_len),b)+1 ✓）；AttentionSpec page 属性链 ✓ / MLAAttentionSpec head_size_v=0 ✓ / MambaSpec tokens_per_state=−1 ✓ / num_states=block_size//tokens_per_state ✓ / KVCacheGroup.from_layers 断言 + page_size_bytes 求和 ✓ / ManagerConfig frozen hashable ✓；_MANAGER_KINDS L3314/L3335 ✓；BlockPool 三件 + null_block + get_new_blocks(evict→ref_cnt=1) + touch + free_blocks 分流 ✓；planning num_blocks 行 + min 缩 ✓ / estimate_max_model_len 二分 docstring ✓ / get_max_concurrency 逐组求和除池 ✓；EncoderCacheManager embedding 计数 + FIFO popitem(last=False) ✓ / _try_schedule_encoder_inputs docstring ✓；EngineCore.step 五段 + busy loop + post_step + idle 阻塞 ✓；LLMEngine.step 四段顺序 ✓；MultiprocExecutor.execute_model=collective_rpc(unique_reply_rank=output_rank) ✓；RequestState 维护顺序 finish→free→add→update→apply_staged_writes ✓；cumem.sleep offload/discard 语义 + Worker.sleep level 分支 + wake_up(tags) ✓；eagle/mtp 薄壳全文 ✓；selector 缓存键 AttentionSelectorConfig ✓ / AttentionCGSupport 存在 ✓ / CudaCommunicator 三路分派 ✓；模块目录清单（spec_decode 扩展 / attention backends 扩展 / weight_transfer 五 engine / kv_offload policies+tiering / connector 后端 / gpu/sample 全件 / multimodal 全件含 video.py / v1/sample 只剩 ops / launchers/api_server / pp_utils+pp_transport / DPSyncState+dispatch_cg_and_sync_dp / MMEncoderModelRunner(GPUModelRunner) / DynamicSDSchedule）逐项 ✓；SGLang 对照锚点 16 项（GroupCoordinator L271 / CustomAllreduce L40 / V2 L152 / TorchSymmMem L34 / DataParallelController L159 / LoadBalanceMethod L106 / EPLBManager L29 / ElasticEPStateManager L132 / CompilationConfig L20 / CUDAPiecewiseBackend L49 / server_args L345+L355 / derive_attn_tp_size L73 / get_parallel L1314 / release+resume L1899/L1905）全 ✓。

### 6.4 跨章 recap 忠实性（W1/W2 原文 grep 对照）

1.2（N=7.50e9/8.03e9 双口径、16.1 GB、15.1/66.5 ms、κ 表）✓；3.4（330×5 μs=1.65 ms、4.81 ms 权重-only、CompilationMode 漂移注、命题 2 指针固定+内容可见、§7 τ_c=6/τ_m=4/τ_n=2→12/6=2×、doc L334 逐字）✓；2.1（命题 1/2、式 (1)(4)(6)(7)(9)、48 GiB/6144/61 块/488 MiB、watermark 0.0 L232、§5 COW / §6 LCM）✓；2.2（式 (1) hash 链、式 (2) 节省、§4 = 48.6 TFLOP/**49 ms 峰值口径** → S2-3）；4.1（§5 统一 token 视图注释逐字、完成即补位）✓；4.2（式 (1) 133 ms、§1 victim、§2 watermark 触发链、§5 九策略、§6 encoder）✓；4.3（式 (2) iteration 模型、§5 伪码 1984、c_hi）✓；3.1（式 (10) M_main=KκL=22.5 MiB、§6 max_num_splits 三态、§5 MLA 512+64 + DeepInfer 不用 TP）✓；1.4（式 (1)-(8) 编号逐一对上、命题 2/5、§5 末句异步化、§3 旧路径 → advisory）✓；2.4（§1 PCIe 64 GB/s 换算注、式 (1)(2)(3)、§2 t_overlap、§4 write-back 可丢+拷贝三条、§9 SGLang HiSparse 模型清单 V3.2/GLM-5.1/Kimi-K3.1/V4/MiniMax M3 + eager backup + direct-to-host + DeepInfer 不 offload）✓；4.4（命题 1、式 (6) E[K]、§6 B(γ+1)）✓；4.5（§5 last/aux_hidden_states、§6 α̂ 口径 + oracle 上界 + γ+1 verify、§2 typical acceptance、§4 n-gram 查表）✓；1.1（式 (1)(2)(3)）✓；training 5.1（式 (5)(6) f/g、2bsh、§9 vocab embedding 前向 +1 归约）✓；training 4.1（式 (2)(3) ring 2(d−1)/d + 2(d−1)α）✓。**5.4 §7「E[K] 口径（被接受/总 draft）」沿用 4.5 §6 既定口径（α̂ 分子口径），跨章一致，不判**（log 备查：E[K] 本体是计数，4.4 式 (6)）。

### 6.5 覆盖率声明

- 定义行锚点：章节出现的全部类/函数/常量/字段行锚点均入检（492 项，重复提及去重）；writer 声称 134/196/134/138（含重复提及与范围锚点），本审查按去重后全量核对。
- 文件行数声明 23/23；逐字引句/注释/doc 口径约 50 项全核；数值算例约 40 组全部独立重算；blog 10 项论断 + arXiv 2 篇摘要一手核；SGLang 对照锚点 16 项抽核全中。
- 未覆盖项（声明）：5.3 §9 对照表 DeepInfer 列（7.x 未写，指针级论断按 W1/W2 已交付口径核）；5.4 §8 决策表「边界」列中指向未来章（9.3/9.4）的论断按 spec 大纲意图核；punica/S-LoRA 论文正文 SGMV 机制细节按 research notes W3 已核口径 + 摘要一手核（未重取论文全文）。

## 7. 修复与复审要求

- S2×11 必须修复（其中 S2-1/2/3/4/5/10/11 为语义/证明/符号级，S2-6/7/8/9 为锚点行号/文件级一行修）；S3×13 累计至下一波复审统一清零（逐条判定修复/接受/驳回）。
- 本 batch 无 S1，不触发 S1 强制复审通道；S2 修复后由 evaluator 复核关闭（锚点类可用同一 check_anchors.py 回归）。
- references.md 被引回写（S3-9）与 1.4 指针 errata（advisory）归 controller/research pass 通道，不阻塞 P5 章节关闭。
- writer 反驳通道：附证据（源码引句/行号）复核，复核维持才强制修复。
