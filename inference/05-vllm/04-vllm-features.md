# 5.4 vLLM 高级特性（vLLM Advanced Features）

> 投机解码、结构化输出、KV offload、多 LoRA、多模态、sleep mode、metrics、sampling——八个特性各挂在一步 decode 的哪一段，各自的账怎么算？本章走读 vLLM 的特性层实现：spec_decode 的薄壳继承链（18 / 63 行的 speculator.py 背后是什么）、grammar bitmask 的异步填充、kv_offload 两模块分工、Punica SGMV 的多 adapter batch 账、sleep / wake_up 的权重更新接口与 KV 失效边界，末尾给特性 × 配置决策表 | 前置：5.1（V1 架构与数据流）、5.2（调度器与 KV 管理器）、5.3（分布式执行与编译）、1.1（SLO 指标定义）、1.4（采样链与结构化输出原理）、2.2（prefix caching）、2.4（KV offload 与分层账）、4.4（投机解码账）、4.5（投机系统机制）；training 8.5（RL rollout 训练侧）只 recap 不重推 | 锚点：vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、vllm-project/vllm main（[vLLM repo](../references.md)，[A]，本章全部 file:line 核实于 2026-11-23 fresh tarball）、V1 官方 blog（[vLLM Team, 2025](../references.md)，[C] 开放技术博客、非闭源）、Punica（[Chen et al., 2023](../references.md)，arXiv:2310.18547，[A]，master 分支）、S-LoRA（[Sheng et al., 2023](../references.md)，arXiv:2311.03285，[A]）、XGrammar（[Dong et al., 2024](../references.md)，arXiv:2411.15100，[A]）、guidance / llguidance（[guidance repo](../references.md)，教学参照）、Outlines（[Willard & Louf, 2023](../references.md)，arXiv:2307.09702，[A]）、SGLang repo（[SGLang repo](../references.md)，[A]，sleep 对照）

## 1. 特性层挂在执行链的哪一段

八个特性各挂在一步 decode 的哪一段？5.1 的数据流是 add_request → schedule → execute_model → sample_tokens → update_from_output，特性不是八条独立流水线，而是挂在这条链的不同位点上——挂错位置就会把 CPU 工作塞进 GPU 关键路径。先给挂载表，后面各节按表走读。

| 特性 | 挂载位点 | 核心执行件（2026-11-23） | 详展 |
| --- | --- | --- | --- |
| 投机解码 | execute_model 内（draft propose + target verify）+ sample 侧 accept | gpu/spec_decode/ 继承链、RejectionSampler L152 | §2 |
| 结构化输出 | schedule 后取 bitmask、sample 前应用、output 后推进状态 | StructuredOutputManager L52 | §3 |
| KV offload | schedule 侧 lookup / prepare、worker 侧 load / store | kv_offload/ + simple_kv_offload/ | §4 |
| LoRA | execute_model 内每个 linear 位点 + 请求级 adapter 绑定 | PunicaWrapperGPU L33 + Triton shrink/expand | §5 |
| sleep / wake_up | 引擎级状态切换（不在单步链内） | CuMemAllocator L62 + Worker.sleep L351 | §6 |
| sampling | sample_tokens（L2279）内 | Sampler L42（gpu/sample/sampler.py） | §7 |
| metrics | 每步 stats 收集 + 周期导出 | PrometheusStatLogger L487 | §7 |
| 多模态 | schedule 侧 encoder 调度 + execute_model 内 encoder 前向 | EncoderCacheManager（5.2）+ EncoderRunner（5.3 §8） | §8 |

**本章口径**（章脊锁定，换口径随文声明）：主口径 = Llama-3.1-8B dense GQA（ $L$ = 32、 $d_{\mathrm{model}}$ = 4096、 $d_{\mathrm{ff}}$ = 14336、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16），硬件 H100 SXM（HBM 3.35 TB/s、80 GB——1.3 datasheet 参数表 recap；PCIe Gen5 单向 ~64 GB/s——2.4 §1 口径；[NVIDIA datasheet](../references.md)，[C] 官方规格披露）；decode $B$ = 64、 $T$ = 4096 步时 15.1 ms、权重 16.1 GB——recap 1.2 / 3.4 算例，不重推。局部记号： $V$ = 词表大小（数值例取 $V$ = 131072 = $2^{17}$ ，教学圆整数口径便于整除，实际值依模型 tokenizer config）； $r$ = LoRA rank（数值例取 16，教学值）； $n_{\mathrm{adp}}$ = batch 内不同 adapter 数、 $n_{\mathrm{hot}}$ = GPU 驻留 adapter 数（不占 $A$ ：5.1 / 5.3 的 $A$ = API server 数；LoRA 因子记号译名见 §5 符号翻译注）； $n_{\mathrm{site}}$ = 挂 LoRA 的 linear 位点数； $\alpha$ = 接受率、 $\gamma$ = draft 长度（spec §4.3）； $b$ = block size；bytes 不占字母。投机例沿用 4.4 / 4.5 章脊（ $\alpha$ = 0.8、 $\gamma$ = 4、 $c = 1/32$ ）；§6 的 KV 池 40 GB 是教学部署值（实际由 5.2 的容量规划定）。

## 2. 投机解码：薄壳与继承链

`eagle/speculator.py` 只有 18 行、`mtp/speculator.py` 只有 63 行——照这两个文件走读会拿到空壳。实现在哪？沿继承链往上追四层。

**两层走读纪律**（spec §4.2）：4.4 已交付接受规则、分布保持证明、 $E[N_{\mathrm{tok}}]$ 与加速比定理，4.5 已交付 EAGLE 三代 / Medusa / MTP / REST / n-gram 的机制与 draft 分类学；本节 recap 一句 + 指引，不重推其公式，增量只在**继承链结构、配置面与辅助件**。

继承链（`vllm/v1/worker/gpu/spec_decode/`，2026-11-23）：

| 层 | 类与锚点 | 职责 |
| --- | --- | --- |
| 1 | speculator.py BaseSpeculator L74（propose L100） | 接口基类：propose 契约 |
| 2 | speculator.py DraftModelSpeculator L165（load_model L300） | 持 draft 模型的基类：权重加载与设备管理 |
| 3 | autoregressive/speculator.py AutoRegressiveSpeculator L24 | 自回归 draft 执行：capture L118 / _run_multi_step_decode L167 / _fused_multi_step_decode L321 |
| 4 | target_dependent_ar/speculator.py TargetDependentARSpeculator L23 | **消费 target 隐藏态的 AR draft**：propose L103 / _run_model L273 / _prefill L339 / _decode L391 / prepare_prefill_inputs L525 / update_draft_inputs L692 |
| 薄壳 | eagle/speculator.py EagleSpeculator L12（18 行） | 只 override load_draft_model → eagle/utils.py load_eagle_model L115 |
| 薄壳 | mtp/speculator.py MTPSpeculator L12（63 行） | load_draft_model L15（同走 eagle/utils.py load_eagle_model L115）+ 四个生命周期 hook：on_prefill_begin L42 / on_prefill_end L48 / on_multi_step_decode_begin L55 / on_multi_step_decode_end L61（做 skip_topk 切换与 compact_topk_indices，均属 index 共享 / 输入准备）；share_mtp_topk_indices 是类属性（L13），在 load_draft_model 内赋值 |

第 4 层还有两个 Triton kernel：_prepare_prefill_inputs_kernel L443 与 _prepare_decode_inputs_kernel L565——draft 输入准备本身是 GPU kernel，不是 CPU 循环（这是 4.5 §1「draft 复用 target 隐藏态、不重跑主干」在实现层的落点：输入准备也要便宜）。

**命题 1**（薄壳不改 propose 语义）。EagleSpeculator 与 MTPSpeculator 都不 override propose：EAGLE 只 override 权重加载（load_draft_model）；MTP 另 override 四个 prefill / decode 生命周期 hook（on_prefill_begin / on_prefill_end / on_multi_step_decode_begin / on_multi_step_decode_end），全部属 index 共享与输入准备。故两条路线的 draft 生成语义由 TargetDependentARSpeculator.propose（L103）决定，差别在 draft 权重来源与输入准备。

**证明**（路线：继承链上的 override 集合 → 动态分派落点 → 语义等价，三步）。第一步，逐类枚举 override：EAGLE（18 行）= {load_draft_model}；MTP（63 行）= {load_draft_model, on_prefill_begin, on_prefill_end, on_multi_step_decode_begin, on_multi_step_decode_end}——四个 hook 的函数体只做 set_skip_topk(True/False) 切换与 compact_topk_indices(last_token_indices) 压实（L42-63），属 index 共享 / 输入准备；share_mtp_topk_indices 是类属性（L13）而非 override，在 load_draft_model 内按 draft 模型能力赋值（L34）。两个集合均不含 propose。第二步，Python 的方法分派沿 MRO 向上找第一个定义——propose 在两个薄壳里都无定义，故落到 TargetDependentARSpeculator.propose（L103）。第三步，propose 的行为由其代码与输入决定：权重加载路径只改变 draft 网络的参数值，四个 hook 只改变 topk 索引的共享与压实状态，两者都不改 propose 的计算结构。∎ 正例：EAGLE 与 MTP 共用同一条 verify / accept 路径（RejectionSampler L152）。边界反例：n-gram 路线不在此链上——NgramGPUSpeculator（ngram/speculator.py L151）自实现 propose（L213），draft 来自 GPU 侧 n-gram 匹配（Triton _ngram_scan_kernel L21，MIN_N / MAX_N 为 constexpr），即 4.5 §4 的查表路线，不消费 target 隐藏态。

> **坑**：只锚 `eagle/speculator.py` / `mtp/speculator.py` 做走读或复核。触发条件：照 W2 之前的目录口径（「三件均在」）读当前 main；现象：拿到 18 / 63 行薄壳，propose / verify 逻辑全部落空；规避：沿 BaseSpeculator → DraftModelSpeculator → AutoRegressiveSpeculator → TargetDependentARSpeculator 链锚定（propose L103），薄壳只作权重加载入口（2026-11-23 核实）。目录已扩展（dspark/ / dflash/ / dflash2/ / dfly/ / suffix/ / standalone_ar/ / multi_module_mtp/ / xpress/ / dspine/ / gemma4/ / lilicorr/ / fly.py），超出 eagle / ngram / mtp 范围的一句注，不展开；dspark/ 是 DSpark 已进 vLLM 的落点，机制与账归 7.2（回链 4.5）。

接受侧与自适应件（同目录，2026-11-23）：rejection_sampler.py RejectionSampler L152（483 行）+ rejection_sampler_utils.py——4.4 命题 1 的拒绝采样 + 残差修正的实现落点（分布保持证明 recap 4.4，不重推）；acceptance_estimator.py OnlineAcceptanceEstimator L345（在线估计接受率）；adaptive_verification.py AdaptiveVerificationManager L227 / AdmittedDraftCounts L214（动态调 verify 长度）；另有 speculator_config.py / draft_logits.py / extract_hidden_states{,_config}.py。4.5 §6 已立「接受率测量与自适应 $\gamma$ 」的机制与 oracle 上界，本节只给实现件。

旧目录 `v1/spec_decode/` 现在是 metrics / dynamic 辅助（W0 口径再确认）：metrics.py SpecDecodingStats L19 / SpecDecodingLogging L64 / SpecDecodingProm L195；dynamic/utils.py 的 validate_and_normalize_dynamic_sd_schedule——DynamicSDSchedule 是 list[tuple[int,int,int]]，**按 batch size 动态调 $\gamma$ **。这条配置的账面动机 recap 4.4 §6 / 4.5 §6：大 batch 下 verify 的 $B(\gamma+1)$ 个 token 吃满预算（ $B$ = 32、 $\gamma$ = 4 → 160 token）， $\gamma$ 的边际收益下降，故按 batch 分档给不同 $\gamma$ （逐档数值依 profile，本章不虚构）。

配置面（`config/speculative.py`，2026-11-23）：SpeculativeMethod L113（ngram / draft_model / dfly / suffix / eagle / eagle3 / extract_hidden_states / mtp 系 / dflash / ngram_gpu / dspark）、RejectionSampleMethod L122（standard / synthetic / block / fly）、DraftSampleMethod L123（greedy / probabilistic）；字段 method L475 / rejection_sample_method L620。三个枚举正交：method 定 draft 来源、draft_sample_method 定 draft 侧采样、rejection_sample_method 定接受规则——4.5 §2 的「分布保持 vs typical acceptance」分水岭在配置面就是第三个枚举的选择。

```
投机一轮的调用链（EAGLE / MTP 共用；算法与账归 4.4 / 4.5）
──────────────────────────────────────────────────────────
target forward（verify，γ+1 个位置）
   │ last_hidden_states / aux_hidden_states（4.5 §5 接口）
   ▼
EagleSpeculator / MTPSpeculator（薄壳：权重加载；MTP 另 4 个输入准备 hook，见 §2 表）
   │ 分派向上
   ▼
TargetDependentARSpeculator.propose L103
   ├─ prepare_prefill_inputs L525 / update_draft_inputs L692
   │    └─ Triton kernel L443 / L565（GPU 侧输入准备）
   ├─ _prefill L339 / _decode L391 / _run_model L273
   └─ AutoRegressiveSpeculator：_run_multi_step_decode L167
        / _fused_multi_step_decode L321 / capture L118（draft 侧图捕获）
   ▼ draft tokens [num_reqs, γ]
RejectionSampler L152 ── 接受前缀 + 残差重采样（4.4 命题 1）
   ▼
OnlineAcceptanceEstimator L345 → AdaptiveVerificationManager L227（调 γ）
   ▼
SpecDecodingStats L19 / SpecDecodingProm L195（→ §7 metrics）
──────────────────────────────────────────────────────────
```

## 3. 结构化输出：bitmask 与异步填充

一个 $V$ = 131072 的词表、64 条并发请求，每步要往 GPU 送多少字节的合法性 mask？答案是 1 MiB——而且这份 mask 必须在采样前就绪。本节走读它的生产与填充时机。

**两层纪律**：1.4 §5 已交付 grammar → 自动机 → token mask 的原理、三系统（Outlines FSM / XGrammar PDA / guidance 程序）对照与命题 5（约束保持）；本节 recap 一句 + 指引，增量只在 bitmask 的工程实现与异步填充。

执行件（`vllm/v1/structured_output/`，2026-11-23）：StructuredOutputManager（`__init__.py` L52）——grammar_init L173（请求级编译自动机）、grammar_bitmask L374（取本步 mask）、accept_tokens L506（按已采 token 推进自动机状态）、trim_spec_tokens L349（投机 token 的 grammar 修剪）、_fill_bitmasks L222 + _async_submit_fill_bitmask L245（mask 填充与异步提交）。backend 接口在 backend_types.py（StructuredOutputBackend / StructuredOutputGrammar）；request.py / utils.py 辅助。

| backend | 锚点 | 机制（1.4 §5 对照） |
| --- | --- | --- |
| xgrammar | backend_xgrammar.py XgrammarBackend L38 / XgrammarGrammar L234（accept_tokens L250 / validate_tokens L272 / rollback L294 / fill_bitmask L299） | CFG / PDA + 自适应 mask 扩展（config 默认 auto，事实默认解析到 xgrammar） |
| guidance | backend_guidance.py（import llguidance / llguidance.hf / llguidance.torch L28-30） | 经 **llguidance**（Rust 移植）集成，非 guidance Python 库 |
| outlines | backend_outlines.py | FSM + 预计算状态 → token 集 index |
| lm-format-enforcer | backend_lm_format_enforcer.py | 另一 backend，一句注不展开 |

backend 的配置面：`config/structured_outputs.py` 的 backend 字段默认字面值是 "auto"（L22，docstring 逐字「With "auto", we will make opinionated choices」）——xgrammar 是 auto 解析后的**事实默认**，不是配置字面默认（口径：引用「默认 backend」时须区分这两层）。

> **坑**：把 vLLM 的 guidance backend 当成 guidance Python 库的集成。触发条件：按库名推断依赖；现象：实际 import 的是 llguidance / llguidance.hf / llguidance.torch（backend_guidance.py L28-30），性能与行为特征属 Rust 实现；规避：引用时写「guidance 系 backend，经 llguidance 集成」（guidance-ai/llguidance repo，2026-11-23 核实）。SGLang 侧同构（llguidance_backend.py，6.1 走读）。

bitmask 的账。每请求每步一份词表长度的位图（合法位 1 / 非法位 0）：

$$
M_{\mathrm{mask}} = \frac{B \thinspace V}{8} \qquad (1)
$$

**符号**： $M_{\mathrm{mask}}$ = 每步 bitmask 字节数； $B$ = 批内请求数； $V$ = 词表大小；除以 8 = 每字节 8 位。代入 $V$ = 131072、 $B$ = 64：每请求 16384 字节 = 16 KiB，全批 1 MiB。这 1 MiB 是每步的 CPU 填充工作量 + CPU→GPU 传输量（PCIe 单向 64 GB/s 下传输项 16.4 μs）；对照 15.1 ms 步时，传输项本身不大，**真正贵的是 CPU 侧逐请求逐 token 的自动机检查与填充**——所以有 _async_submit_fill_bitmask（L245）。

填充时机是代码与机制的关键接缝。EngineCore.step()（`v1/engine/core.py` L714）的序列是：schedule（L724）→ execute_model(non_block=True)（L727）→ get_grammar_bitmask → _wait_for_output → sample_tokens → update_from_output（L738，调用起始行）。bitmask 的取用排在**非阻塞 execute_model 之后**：GPU 在跑前向时，CPU 并行填充 mask；到 sample_tokens 前必须就绪。这就是 1.4 §5 末句「grammar 编译可异步化」的引擎兑现，也是 9.4 结构化输出专用机（SO 专用机异步编译 grammar automaton）的同一条思路在单引擎内的形态。scheduler 侧接口是 get_grammar_bitmask（L2310）与 GrammarOutput（`sched/output.py` L353）。

与投机的交互两处：trim_spec_tokens（L349）在 draft token 送 verify 前按 grammar 修剪（非法 draft 直接剪掉，省 verify 算力）；rollback（XgrammarGrammar.rollback L294）在接受前缀确定后回滚自动机状态到被拒分支之前——4.4 的拒绝采样在 grammar 侧的对应动作。accept_tokens（L506 / L250）则是 1.4 命题 5 的状态转移 $s \leftarrow \delta(s, \text{token})$ ：每步 mask + 每步推进，两者缺一则合法性不变式破产（1.4 命题 5 的边界反例）。

## 4. KV offload：两模块分工

`v1/kv_offload/` 与 `v1/simple_kv_offload/` 是两套还是一套？答案是两套：前者是分层 / 多后端框架，后者是单一 CPU offload connector 的简化实现（W1 口径本轮再确认）。**两层纪律**：2.4 已交付三层存储账、临界带宽（式 (2)）、传输带宽需求（式 (3)）与拷贝路径（copy engine / GDR），本节 recap + 指引，增量只在抽象与执行件的对应关系。

框架侧（`v1/kv_offload/`，2026-11-23）：base.py（861 行）定义 OffloadingManager ABC L329 与一组抽象——Medium L57（介质）、Locality L64（局部性）、TierFilter L84（层过滤）、ReqContext L101（请求上下文，含 kv_hints L109，接 5.3 §7 的编排协议）、LoadStoreSpec L197 / GPULoadStoreSpec L570（载入 / 存出规格）、OffloadingEvent L216（事件）。后端两个：

| 后端 | 锚点 | 执行件 |
| --- | --- | --- |
| cpu/ | manager.py CPUOffloadingManager L55 | gpu_worker.py（Transfer L82 / CopyPlan L144）、swap_blocks_triton.py、host_register.py、shared_offload_region.py、policies/{lru,arc,base,factory}.py |
| tiering/ | manager.py TieringOffloadingManager L277 / CPUPrimaryTierOffloadingManager L120 | async_lookup.py、backpressure.py、fs/ obj/ p2p/（control / data / session / manager.py——nixl 数据面 + zmq 控制面）、kvcr/（KVCR adapter）、example/ |

另有 factory.py / config.py / file_mapper.py / pointer_utils.py 与 fs/worker.py。简化侧（`v1/simple_kv_offload/`）：manager.py 的 docstring 逐字「Scheduler-side manager for SimpleCPUOffloadConnector」+ SimpleCPUOffloadScheduler L143；worker.py / host_buffer.py / copy_backend.py / disk_backend.py / cuda_mem_ops.py / sizing.py / metadata.py / metrics.py。connector 本体不在该目录，而在 `distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` SimpleCPUOffloadConnector L81（实现 KVConnectorBase_V1 + SupportsHMA，接口基线见 5.3 §9）。

代码与公式的对应表（2.4 的账落在哪个抽象上）：

| 2.4 的账 | 执行件 | 机制 |
| --- | --- | --- |
| 式 (1) 命中率加权期望成本 | policies/{lru,arc} + TierFilter L84 | 驱逐策略决定 $h_1 / h_2 / h_3$ 的分配 |
| 式 (2) 临界带宽 $\mathrm{BW}^{\ast}$ | Medium L57 / Locality L64 | 介质与局部性决定取回是否快于重算 |
| 式 (3) 传输带宽需求 | CopyPlan L144 / Transfer L82 | 批量与拷贝计划决定有效带宽 |
| §4 write-back 可丢、读路径严格 | backpressure.py | 写路径限速 / 背压，不挤占读带宽 |
| §2 重叠窗口 $t_{\mathrm{overlap}}$ | async_lookup.py | 查找与预取异步化，藏进排队 / 计算 |
| §4 拷贝路径三条 | swap_blocks_triton.py（SM kernel）/ copy_backend.py / cuda_mem_ops.py | Triton 搬块占 SM；后端抽象可换 DMA / GDR 路径 |

与 5.3 §7 的 HiSparse 分工一句话：kv_offload 是**容量驱动**分层（全量 KV 块都可能换入换出），HiSparse 是**attention 驱动**分层（只换入 top-k 行）——2.4 §9 已立这条对照，两模块在 vLLM 侧并存，经 connector 层组合（5.3 §9 的 MultiConnector）。

## 5. LoRA serving：SGMV 与多 adapter batch

100 个租户各挂一个 rank 16 的 adapter，base model 要存几份？答案是一份——但 naive 实现会让每步的 kernel launch 数涨 100 倍。本节把这两笔账算到数。

机制 recap（LoRA 的模型侧形式化归 models 维度，此处只给 serving 用到的形式）：每个 linear 位点的权重增量是低秩分解 $\Delta W = W_{\mathrm{up}} W_{\mathrm{down}}$ （ $W_{\mathrm{up}}$ 为 $d_{\mathrm{out}} \times r$ 、 $W_{\mathrm{down}}$ 为 $r \times d_{\mathrm{in}}$ ），前向 $y = Wx + W_{\mathrm{up}}(W_{\mathrm{down}} x)$ 。所谓 shrink，是指 $W_{\mathrm{down}} x$ 这一步（ $d_{\mathrm{in}} \to r$ 降维）；所谓 expand，是指 $W_{\mathrm{up}}(W_{\mathrm{down}} x)$ 这一步（ $r \to d_{\mathrm{out}}$ 升维）。

> **符号翻译注**（spec §4.3 硬规则）：LoRA 论文记号 $\Delta W = BA$ 的两个因子在本维度译作 $W_{\mathrm{up}}$ （原 B）/ $W_{\mathrm{down}}$ （原 A）—— $B$ 专归 batch（本章式 (1) 的批内请求数）、 $A$ 在 5.1 / 5.3 已归 API server 数，两个字母都不占用。adapter 计数同理用 $n_{\mathrm{adp}}$ （批内不同 adapter 数）与 $n_{\mathrm{hot}}$ （GPU 驻留 adapter 数），不占 $A$ 。

两步的参数量与 FLOPs 都只与 $r$ 线性相关，与 base 权重的 $d_{\mathrm{in}} d_{\mathrm{out}}$ 相比是小量：

$$
P_{\mathrm{site}} = r (d_{\mathrm{in}} + d_{\mathrm{out}}), \qquad \frac{C_{\mathrm{LoRA}}}{C_{\mathrm{base}}} = \frac{r (d_{\mathrm{in}} + d_{\mathrm{out}})}{d_{\mathrm{in}} d_{\mathrm{out}}} = r \left( \frac{1}{d_{\mathrm{in}}} + \frac{1}{d_{\mathrm{out}}} \right) \qquad (2)
$$

**符号**： $P_{\mathrm{site}}$ = 单个 linear 位点的 adapter 参数量（ $W_{\mathrm{down}}$ 与 $W_{\mathrm{up}}$ 之和）； $C_{\mathrm{LoRA}} / C_{\mathrm{base}}$ = 该位点 LoRA 增量 FLOPs 与 base GEMM FLOPs 之比（每 token 口径，系数 2 在分子分母约去）； $r$ = rank。代入 $r$ = 16： $d_{\mathrm{in}} = d_{\mathrm{out}}$ = 4096 的位点相对开销 $= 16 \times 2/4096$ = 0.78%； $4096 \times 14336$ 的位点 $= 16 \times (1/4096 + 1/14336)$ = 0.50%。全模型口径（8B 章脊，7 个 linear 位点 / 层：q / k / v / o + gate / up / down）：每层 $P$ = 1,409,024，32 层合计 **45.09 M 参数**，bf16 合 90.2 MB——占 base 权重 16.1 GB 的 0.56%；每 token LoRA FLOPs 90.2 M，对 base $2N$ = 15 GFLOPs 是 0.60%。**算力与权重流量上 LoRA 都是小量，贵的是 kernel 数**。

Punica（[Chen et al., 2023](../references.md)，arXiv:2310.18547，[A]）的答案是 SGMV（Segmented Gather Matrix-Vector multiplication）kernel：把批内不同 adapter 的 LoRA addon 计算合进一个 kernel 分段处理，GPU 只留一份 base model；摘要口径 **12× 吞吐 vs SOTA、每 token 仅加 2 ms 延迟**，scheduler 整合多租户 workload。对照 BGMV（逐 adapter 的 batch=1 GEMV）：后者每个 adapter 各发一次 kernel。launch 账 recap 3.4 式 (2)（ $n_{\mathrm{ker}} \cdot t_{\mathrm{launch}}$ ，教学值 5 μs）：

$$
n_{\mathrm{launch}}^{\mathrm{naive}} = 2 \thinspace n_{\mathrm{adp}} \thinspace n_{\mathrm{site}}, \qquad n_{\mathrm{launch}}^{\mathrm{SGMV}} = 2 \thinspace n_{\mathrm{site}} \qquad (3)
$$

**符号**： $n_{\mathrm{site}}$ = 挂 LoRA 的 linear 位点数（8B 章脊 7 × 32 = 224）；系数 2 = shrink + expand 两个 kernel； $n_{\mathrm{adp}}$ = 批内不同 adapter 数。代入 $n_{\mathrm{adp}}$ = 100：naive 44800 次 launch × 5 μs = **224 ms**，SGMV 448 次 × 5 μs = **2.24 ms**——差 100 倍（恰为 $n_{\mathrm{adp}}$ ）。而 2.24 ms 本身已占 15.1 ms 步时的 14.8%：多 LoRA 的 launch 开销不是零头，须进 CUDA Graph（5.3 §6 的 piecewise）或进一步融合。位点计数口径注：本章按逐矩阵计（q / k / v / o / gate / up / down 各一位点）；vLLM 实现另有 fused 入口（add_lora_w13 L528 对应 gate / up 双支、add_lora_fused_moe L464 对应 MoE 位点），融合后位点数更少、launch 账同比例下降——逐矩阵口径给的是上界计数。

> **提点**：多租户共享 base 的收益在权重流量与显存（一份 base 服务 $n_{\mathrm{adp}}$ 个 adapter），代价在 kernel 数与调度复杂度。判断一个 LoRA 部署贵不贵，先数 launch 再算字节——式 (2) 的 0.6% 与式 (3) 的 14.8% 说的是同一件事的两面。

**命题 2**（多 adapter batch 的 launch 与 base 流量账）。设批内 $n_{\mathrm{adp}}$ 个 adapter、 $n_{\mathrm{site}}$ 个位点。SGMV 路线的每步 LoRA kernel 数为 $2 n_{\mathrm{site}}$ ，与 $n_{\mathrm{adp}}$ 无关；base 权重每步仍只读一份（不随 $n_{\mathrm{adp}}$ 复制）。naive 逐 adapter 路线的 kernel 数为 $2 n_{\mathrm{adp}} n_{\mathrm{site}}$ 。

**证明**（路线：kernel 计数按「位点 × adapter」两维展开 → SGMV 的分段语义消掉 adapter 维 → base 流量按共享论证，三步）。第一步，每个位点要算 shrink 与 expand 两个 kernel；naive 路线按 adapter 分别发，故计数为 $2 n_{\mathrm{adp}} n_{\mathrm{site}}$ 。第二步，SGMV 的语义是把同一位点上不同 adapter 的 token 按 adapter index 分段 gather 进一个 kernel（论文 SGMV 口径），adapter 维成为 kernel 内的分段索引而非 launch 维，计数降为 $2 n_{\mathrm{site}}$ 。第三步，base 权重 $W$ 对所有 adapter 相同，batch 内全部 token 共享同一次 GEMM（1.2 的 decode 权重读账：权重读一遍、FLOPs 随 $B$ 涨），故 base 流量与 $n_{\mathrm{adp}}$ 无关；只有 adapter 权重按驻留集合读（驻留容量账见下文式 (4)）。∎ 边界： $n_{\mathrm{adp}}$ = 1 时两式相等（SGMV 退化为单 adapter，分段开销白费）；adapter 权重若超出 GPU 驻留容量则须换入，流量账多一项（S-LoRA 的主存常驻 + 按需取）。

S-LoRA（[Sheng et al., 2023](../references.md)，arXiv:2311.03285，[A]）解决的正是驻留容量：所有 adapter 常驻主存、按需取 GPU；**Unified Paging** 用统一内存池管不同 rank 的 adapter 权重与变长 KV cache（减碎片）；另有新 TP 策略与异构 batching 的定制 CUDA kernel；摘要口径单 GPU 数千 adapter、**up to 4× 吞吐 vs HuggingFace PEFT / vLLM naive LoRA**。驻留容量账：

$$
M_{\mathrm{adapter}} = n_{\mathrm{hot}} \thinspace P_{\mathrm{LoRA}} \cdot \mathrm{bytes} \qquad (4)
$$

**符号**： $M_{\mathrm{adapter}}$ = GPU 驻留 adapter 权重字节数； $n_{\mathrm{hot}}$ = 驻留 adapter 数； $P_{\mathrm{LoRA}}$ = 单 adapter 全模型参数量（8B 章脊 45.09 M）； $\mathrm{bytes}$ = 每元素字节数（bf16 = 2）。代入： $n_{\mathrm{hot}}$ = 8 → 0.72 GB；100 → 9.02 GB；1000 → 90.2 GB——**超过单卡 HBM 80 GB**，这就是「数千 adapter 必须常驻主存、按需取」的容量理由。驻留池与 KV 池互斥（2.1 容量账）：多留一个 adapter 就少一块 KV。

> **坑**：把 Punica repo 当生产可用实现引用。触发条件：按论文锚点直接取 repo 代码；现象：默认分支是 **master**（main 404），且 master 快照的 `sgmv_expand` 是 NotImplementedError（只有 `sgmv_shrink` 已实现）——repo 长期未维护；规避：机制按论文 SGMV 讲，生产实现锚 vLLM 自有 Triton kernel（下方 lineage 注）。路径：csrc/sgmv/{sgmv.h,sgmv_cutlass.cu,sgmv_cutlass.cuh} + csrc/bgmv/ + csrc/sgmv_flashinfer/ + src/punica/ops/__init__.py（bgmv L166 / sgmv_cutlass L231 / sgmv L319），2026-11-23 tarball 核实。S-LoRA repo（main）：slora/common/mem_manager.py（unified paging）+ gqa_mem_manager.py / int8kv_mem_manager.py + slora/csrc/{bgmv/,lora_ops.cc（dispatch_bgmv L76/L135）} + slora/server/ + slora/mprophet/。

vLLM 的生产实现是 Punica lineage 的 Triton 化（2026-11-23）：`lora/punica_wrapper/punica_gpu.py` PunicaWrapperGPU L33——add_lora_linear L210 / add_lora_logits L275 / add_lora_fused_moe L464 / add_lora_w13 L528 / add_lora_w2 L664（MoE 与 SwiGLU 双支都有专用入口）；基类 punica_base.py PunicaWrapperABC L21 / PunicaWrapperBase L120，另有 punica_selector.py / punica_cpu.py / punica_xpu.py（平台分支）。kernel 在 `lora/ops/triton_ops/`：lora_shrink_op.py（_lora_shrink_kernel L102）、lora_expand_op.py（_lora_expand_kernel L24）、fused_moe_lora_op.py 与 fp8 变体——**shrink 与 expand 两侧都实现**（与 Punica master 快照的 shrink-only 对照）。

adapter 生命周期：worker_manager.py WorkerLoRAManager L27 / LRUCacheWorkerLoRAManager L276（worker 侧热加载 + LRU）；model_manager.py LoRAModelManager L76 / AdapterLRUCache L65 / LRUCacheLoRAModelManager L1447；lora_model.py / lora_weights.py / resolver.py / peft_helper.py / request.py。runner 侧经 LoRAModelRunnerMixin（`v1/worker/lora_model_runner_mixin.py`，GPUModelRunner 继承它，5.3 §1）。引擎级 API 在 EngineCore：add_lora L1233 / remove_lora L1236 / list_loras L1239 / **pin_lora L1242**（钉住不被 LRU 驱逐——热加载的优先级旋钮）。

**命题 3**（prefix cache 共享的 extra-key 边界）。vLLM 的 block hash 由 token ids 与 extra keys 共同决定，extra keys 含 mm / lora / prompt_embeds（`v1/core/kv_cache_utils.py` generate_block_hash_extra_keys L627）。故挂不同 adapter 的两条请求，即使 prompt token 序列逐字相同，也不共享 prefix cache 块。

**证明**（路线：hash 输入集合 → 输入不同则 hash 不同 → 命中判定失败，三步）。第一步，2.2 的 hash 链机制：块 hash 是「前缀 token ids + extra keys」的哈希（hash_block_tokens L697，5.2 已锚）。第二步，extra keys 含 lora 标识，故 adapter 不同 → hash 输入不同 → hash 值不同（哈希函数的确定性：同输入同输出，不同输入以高概率不同输出；此处按引擎语义取「不同 extra keys 产生不同块标识」）。第三步，命中判定按 hash 查 BlockHashToBlockMap（5.2），标识不同即 miss——两块物理上即使内容相同也不复用。∎ 正例：同一 adapter 的两条请求共享前缀（extra keys 相同）。边界反例：若误以为「prompt 相同就能共享」，多 LoRA 部署的命中率估算会系统性偏高——2.2 的命中率账须**按 adapter 分组**再算。mm 输入同理（image hash 进 extra keys，V1 blog 第 7 点）：同 prompt 不同图片不共享。

教学片段（教学用，非优化实现）——单位点的 shrink / expand 两步：

```python
def lora_forward(x, W, W_down, W_up):
    """教学用，非优化实现：单 adapter、单 token 批。
    x: [d_in]，W: [d_out, d_in]，W_down: [r, d_in]，W_up: [d_out, r]
    （形参名沿正文译名：论文记号 A/B → W_down/W_up，见 §5 符号翻译注）
    """
    base = W @ x                 # base GEMV：全 batch 共享一次（命题 2）
    shrink = W_down @ x          # [r]，d_in → r
    expand = W_up @ shrink       # [d_out]，r → d_out
    return base + expand
```

与官方实现的差异三点。其一，官方 _lora_shrink_kernel / _lora_expand_kernel 是 Triton kernel，按批内 token 的 adapter index **分段 gather** 不同 adapter 的 $W_{\mathrm{down}}$ / $W_{\mathrm{up}}$ （SGMV 语义，命题 2 的第二步），教学版只处理单 adapter。其二，官方有 fused_moe_lora 与 fp8 变体（MoE 位点与低精度），教学版是 dense bf16。其三，官方的 base GEMM 与 LoRA addon 在不同 kernel 里（base 走量化 GEMM 路径，3.2 / 3.3），教学版写成一个表达式只为展示两步结构。

```
多 LoRA batch 的 SGMV 分段（n_adp=3 个 adapter、一个 linear 位点）
──────────────────────────────────────────────────────────
batch tokens:  t1 t2 t3 t4 t5 t6 t7 t8
adapter idx:    0  0  1  1  1  2  2  2
                └──┬──┘└────┬────┘└────┬────┘
shrink kernel（一次 launch）：按段 gather W_down[0] / W_down[1] / W_down[2]
   段 0: [t1,t2] @ W_down[0]   段 1: [t3,t4,t5] @ W_down[1]   段 2: [t6,t7,t8] @ W_down[2]
expand kernel（一次 launch）：同分段 gather W_up[0] / W_up[1] / W_up[2]
base GEMM：全部 8 个 token 共享 W（只读一份，命题 2）
naive 对照：3 个 adapter × 2 kernel = 6 次 launch（SGMV 为 2 次）
──────────────────────────────────────────────────────────
```

## 6. sleep mode / wake_up：RL rollout 的权重更新接口

colocate 部署（训练与推理共卡）下，训练器要显存时推理引擎怎么办？vLLM 的答案是 sleep：不销毁进程、释放 GPU 物理显存，训练完再 wake_up。所谓 sleep mode，是指引擎在不退出进程、保留虚拟地址布局的前提下释放 GPU 物理显存、按 level 处置权重的低占用状态；wake_up 是其逆操作（按 tags 恢复）。

**两层纪律**：training 8.5 已交付 RL rollout 的训练侧编排（四模型显存、rollout / train 引擎拆分、权重同步训练侧），本节 recap 一句 + 指引，增量只在引擎侧接口与正确性边界；colocate vs disagg 的权衡与每 iteration 同步成本账归 9.3。

实现件三层（2026-11-23）：

| 层 | 锚点 | 职责 |
| --- | --- | --- |
| 显存分配器 | device_allocator/cumem.py CuMemAllocator L62（sleep L261 / wake_up L364 / use_memory_pool L394） | CUDA virtual memory management；**tag 分池 weights / kv_cache** |
| worker | gpu_worker.py Worker.sleep L351（level 1/2）/ wake_up L389（tags）/ sleep_mode_backend L305 | 单卡侧执行 sleep / wake |
| 引擎 / API | EngineCore.sleep L1103 / wake_up L1156 / release_kv_cache_memory L1187；LLM.sleep L853 / wake_up L897 / collective_rpc L587 | 引擎级状态切换与对外 API |

机制要点：CuMemAllocator 用 CUDA 虚拟内存管理把显存按 tag 分池（weights 与 kv_cache 各一池），sleep / wake_up 以 tag 为粒度操作——释放物理页而保留虚拟地址，wake_up 时重新映射。这解释了为什么 wake_up 能按 tags 选择性恢复（只恢复 weights 池、KV 池另行重建）。权重更新侧另有 EngineCore.set_weight_version L1264 / compute_weight_checksums L1230（版本与校验和），以及 `distributed/weight_transfer/` 的五类 engine（nccl / m2n / modelexpress / ipc / sharded_rdt，训练器 → 推理引擎的权重传输后端族，5.3 §4 已登记）——逐 engine 的带宽账与 rollout 编排归 9.3。

**命题 4**（wake_up 后的 KV 失效边界）。若 sleep 释放了 kv_cache tag 池的物理页，则 wake_up 后该池的旧 KV 内容不可信：任何依赖旧 KV 的请求必须重算或经 KV connector 取回，不能沿用旧 block table 直接 decode。

**证明**（路线：物理页释放 → 内容不再驻留 → 重映射后内容未定义，三步）。第一步，sleep 释放 kv_cache 池的物理页（tag 粒度操作，CuMemAllocator.sleep L261），原 KV 字节不再驻留 GPU。第二步，wake_up 重新分配 / 映射物理页（L364），新页内容不由引擎保证等于旧 KV——虚拟地址相同不蕴含内容相同（3.4 命题 2 的「指针固定 + 内容可见」前提里，内容可见依赖 buffer 未被释放，此处前提已破）。第三步，旧 block table（2.1 式 (3)）仍指向那些虚拟地址，直接 decode 会读到未定义内容——故必须重建 KV（重算 prefill）或从 connector 取回（§4）。∎ 边界：weights 池若按 level 处置为保留到 host，则 wake_up 可恢复原权重（内容由 host 副本保证）；KV 池无此路径，除非经 §4 的 offload / connector 事先存出。level 1 / 2 的逐 level 显存账与权重处置细节归 9.3，本章只交付「KV 必失效、权重按 level 可恢复」的接口边界。

> **坑**：wake_up 后沿用旧 KV 与旧 block table 继续 decode。触发条件：RL rollout 循环里 sleep → 更新权重 → wake_up 后未重建 KV；现象：读到未定义显存内容，输出静默损坏（不报错）；规避：wake_up 后清空 / 重建 KV 池，或经 connector 取回；权重侧用 set_weight_version（L1264）/ compute_weight_checksums（L1230）校验版本一致。另一处易混：`v1/watermarking/` 是生成文本水印，不是 KV watermark（KV 水位在 scheduler_config.watermark，5.2）——两词同形异义。

显存交还的账（colocate 口径，教学部署值）：单卡 80 GB，推理侧权重 16.1 GB + KV 池 40 GB = 56.1 GB，训练器可用 23.9 GB；sleep 释放后训练器可用升到 80 GB 量级（**3.35×**）。这就是 colocate 的账面收益：同一张卡在训练与推理之间分时复用，代价是每次切换的 sleep / wake 时间与 KV 重建（9.3 的同步成本账）。

```
colocate RL rollout 一轮（引擎侧接口）
──────────────────────────────────────────────────────────
rollout 阶段：引擎正常 serving（权重 + KV 占 56.1 GB）
   │ LLM.sleep L853 → EngineCore.sleep L1103 → Worker.sleep L351
   ▼ CuMemAllocator.sleep L261：释放物理页（weights / kv_cache 按 tag）
训练阶段：训练器占卡（可用显存 23.9 → 80 GB 量级）
   │ 权重更新：weight_transfer/ engines（nccl / ipc / ...）
   ▼ LLM.wake_up L897 → EngineCore.wake_up L1156 → Worker.wake_up L389(tags)
恢复阶段：weights 池按 level 恢复；KV 池重建（命题 4）
   │ set_weight_version L1264 / compute_weight_checksums L1230
   ▼ 下一轮 rollout
SGLang 对照：release_memory_occupation / resume_memory_occupation（6.3）
──────────────────────────────────────────────────────────
```

## 7. sampling 与 metrics：两个横切面

Sampler 在哪个文件？不在 `v1/sample/sampler.py`——该目录当前只剩 ops/，Sampler 与 processor 链已迁到 worker 侧，照旧路径锚定会落空。

sampling 侧（2026-11-23）：**Sampler 在 `v1/worker/gpu/sample/sampler.py` L42**（apply_sampling_params L223 / apply_staged_writes L112）；processor 链各件在同目录 gpu/sample/：greedy_argmax / gumbel / min_p / penalties / logit_bias / bad_words / thinking_budget / watermark / prompt_logprob / logprob / batch_shard / states / output / trace_replay，加 logits_processor/{interface,loader}.py；截断算子在 `v1/sample/ops/`：topk_topp_sampler.py（apply_top_k_top_p L198 / flashinfer_sample L335 / apply_top_k_top_p_pytorch L228）与 topk_topp_triton.py。

> **坑**：按 `v1/sample/sampler.py` 锚定 Sampler。触发条件：照旧路径口径（1.4 §3 errata 前措辞「sampling 实现 vllm/v1/sample/」）读当前 main；现象：v1/sample/ 只剩 ops/，Sampler 与 processor 链已迁 gpu/sample/；规避：Sampler 锚 `v1/worker/gpu/sample/sampler.py` L42、processor 锚 gpu/sample/ 各件、截断算子锚 v1/sample/ops/（2026-11-23 核实，1.4 回链按新路径）。

代码与 1.4 公式的对应（recap 一句 + 指引，不重推）：

| 1.4 的公式 / 命题 | 实现落点 |
| --- | --- |
| 式 (1) 温度缩放 | Sampler.apply_sampling_params L223 |
| 式 (2) top-p / 式 (4) min-p 截断 | ops/topk_topp_sampler.py（apply_top_k_top_p L198）+ gpu/sample/min_p.py |
| 式 (3) 截断再归一化 | 同上（截断后归一） |
| 式 (5)(6) penalty 三式 | gpu/sample/penalties.py |
| 式 (7) log_softmax 稳定形 | gpu/sample/logprob.py / prompt_logprob.py |
| 式 (8) Gumbel-max | gpu/sample/gumbel.py |
| 命题 2（penalty 不可交换 → 顺序是语义） | logits_processor/{interface,loader}.py 的固定链序 |
| §6 batch invariance | 3.4 §6 的确定性模式（kernel 层），采样侧 greedy_argmax.py |

截断有两条实现路径：flashinfer_sample（L335，走 FlashInfer 的 fused sampling kernel）与 apply_top_k_top_p_pytorch（L228，PyTorch 参考路径）——kernel 路线与参考路线并存，后者是对账基准（3.3 的 sampling kernel 账）。execute_model（L1865）与 sample_tokens（L2279）分离（5.3 §1）使采样链可以晚一步执行，这是 async scheduling 的接口前提。

metrics 侧（`v1/metrics/`，2026-11-23）：loggers.py（1467 行）StatLoggerBase L55 / LoggingStatLogger L113 / **PrometheusStatLogger L487** / StatLoggerManager L1372；stats.py SchedulerStats L224 / RequestStateStats L264 / PrefixCacheStats L121 / FinishedRequestStats L293 / RequestSpecDecodeMetrics L355；辅助 prometheus.py（multiprocess REGISTRY）/ reader.py / perf.py / buckets.py / cache_hit_source.py / ray_wrappers.py；官方 docs/design/metrics.md + docs/usage/metrics.md。

指标与 1.1 定义的对应（**口径**：Prometheus 导出的是原始量，goodput 是导出量——须按 SLO attainment 目标外部计算，引擎不直接给）：

| 1.1 的量 | 代码落点 | 口径注 |
| --- | --- | --- |
| TTFT / TPOT（式 (1)）/ E2E（式 (2)） | FinishedRequestStats L293 + RequestStateStats L264 | 逐请求时间戳统计，百分位由 buckets.py 分桶 |
| 吞吐（tokens/s、requests/s） | SchedulerStats L224 | 每步调度量累计 |
| goodput $\mathcal{G}$ （式 (3)） | 无直接指标 | 须按 attainment 目标外部计算（per-GPU 归一 + SLO 约束） |
| prefix 命中率 | PrefixCacheStats L121 + cache_hit_source.py | 2.2 命中率账的数据源 |
| 接受率 $\alpha$ / 接受长度 | RequestSpecDecodeMetrics L355 + SpecDecodingStats L19 | 4.4 的 $E[K]$ 口径（被接受 draft token 数 / 总 draft 数），勿与 $E[N_{\mathrm{tok}}]$ 混 |

多进程部署（DP / TP 多 worker）下各进程的指标经 multiprocess REGISTRY 汇聚（prometheus.py）——否则每个 worker 各暴露一份、抓取端看到的是分片而非全局。

## 8. 多模态与特性 × 配置决策表

多模态请求的 image 在哪一步变成 embedding？vLLM 侧三处执行件（2026-11-23）：`multimodal/` 的 processing/（processor.py 等）+ cache/{base,lru,shm}.py + hasher.py + registry.py + encoder_budget.py + image.py / audio.py / video.py；encoder 前向在 gpu/mm/（EncoderRunner L36，5.3 §8 已锚）；encoder cache 在 EncoderCacheManager（5.2：check_and_update_cache L111 / compute_mm_encoder_budget L329）。V1 blog 第 7 点的三件（预处理独立非阻塞进程 + 预处理 cache、image hash 进 prefix caching、encoder cache 使 MLLM 可 chunked prefill）分别落在 processing/cache、generate_block_hash_extra_keys（L627，命题 3）、EncoderCacheManager。image patch 账、VE 分离与负载均衡、encoder cache 命中节省归 9.4 / 7.4——本章只登记落点。

特性 × 配置决策表（配置字段与锚点均 2026-11-23 核实；「边界」列给已核实的依赖或代价，不虚构互斥）：

| 特性 | 配置入口 | 边界 / 代价 | 回指 |
| --- | --- | --- | --- |
| 投机解码 | speculative_config：method L475（SpeculativeMethod L113）× rejection_sample_method L620 × draft_sample_method | verify 吃 $B(\gamma+1)$ token 预算；接受规则决定分布保持与否 | §2、4.4 / 4.5 |
| 结构化输出 | structured_output backend（config 默认 auto → 事实默认 xgrammar / guidance 经 llguidance / outlines / lm_format_enforcer） | 每步 $BV/8$ 字节 mask + CPU 填充；与投机共用需 trim / rollback | §3、1.4 |
| KV offload | kv_transfer_config：kv_connector × kv_role（OffloadingConnector / SimpleCPUOffloadConnector 等） | 取回值不值按 2.4 式 (2) 临界带宽；写路径背压 | §4、2.4 |
| HiSparse | HiSparseConfig（config/attention.py L21）：device_buffer_size / eager_host_mirror + host_pool_gib | 限 DSA/QSA 系模型 + backend supports_hisparse；驻留比 2K/T | 5.3 §7 |
| LoRA | lora_config + EngineCore add_lora L1233 / pin_lora L1242 | 驻留容量式 (4) 与 KV 池互斥；prefix cache 按 adapter 分区（命题 3）；launch 账式 (3) | §5 |
| sleep / wake_up | sleep level（1/2）+ wake_up tags | KV 必失效（命题 4）；切换时间 + KV 重建成本 | §6、9.3 |
| 编译 / 图捕获 | CompilationMode（VLLM_COMPILE=3）× CUDAGraphMode | piecewise 依赖 splitting_ops；图内存与 KV 容量互斥 | 5.3 §6、3.4 |
| sampling | top-k / top-p / min-p / penalties 参数 | 顺序是语义（1.4 命题 2）；确定性模式有吞吐代价 | §7、1.4 / 3.4 |
| metrics | PrometheusStatLogger L487 + multiprocess REGISTRY | goodput 须外部按 attainment 算 | §7、1.1 |
| 多模态 | 预处理 cache + encoder cache（encoder_budget） | image hash 进 block hash extra keys（命题 3 的 mm 分支） | §8、9.4 |

三条选型论断（可判定）：其一，**特性代价的主项各不相同**——投机的代价在 verify 算力（ $B(\gamma+1)$ ）、结构化输出在 CPU mask 填充、LoRA 在 kernel 数（式 (3)）、offload 在取回带宽（2.4 式 (3)）、sleep 在切换与 KV 重建；按主项而非「特性开关」做容量规划。其二，**两个特性会改变 prefix cache 的共享边界**（LoRA 与 mm，命题 3），命中率账须按 extra keys 分组。其三，**编译与图捕获是其他特性的公共底座**：LoRA 的 448 次 launch、投机的 draft 侧 capture（AutoRegressiveSpeculator.capture L118）、自定义归约的 register_graph_buffers（5.3 §4）都要经 5.3 §6 的链条才能把 CPU 开销抹掉。

## 总结

| 论断 | 公式 / 数值 / 锚点 | 回指 |
| --- | --- | --- |
| 八特性挂在 step 链的不同位点；bitmask 取用排在非阻塞 execute_model 之后（CPU 填充与 GPU forward 重叠） | §1 表、core.py step L714 序列 | §1、§3 |
| eagle / mtp speculator.py 是 18 / 63 行薄壳；实现在 TargetDependentARSpeculator.propose L103（四层继承链） | §2 表 + 坑注 | §2、练习 4 |
| 命题 1：薄壳只 override 权重加载与输入准备 hook（EAGLE 仅权重加载、MTP 另 4 hook，均不含 propose），propose 语义由第 4 层决定；ngram 自实现 propose L213 不在此链 | 命题 1 | §2 |
| 接受侧 RejectionSampler L152；自适应 OnlineAcceptanceEstimator L345 / AdaptiveVerificationManager L227；旧 v1/spec_decode/ = metrics + 按 batch size 动态 γ | §2 | §2、4.4 §6 |
| 配置三枚举正交：SpeculativeMethod L113 × RejectionSampleMethod L122 × DraftSampleMethod L123 | §2 | §2、4.5 §2 |
| bitmask 每步 $BV/8$ 字节：V=131072、B=64 → 1 MiB（每请求 16 KiB） | 式 (1) | §3、练习 1 |
| guidance backend 经 llguidance（Rust）集成；trim_spec_tokens L349 / rollback L294 是与投机的两个交互点 | §3 表 + 坑注 | §3、1.4 |
| kv_offload = 分层多后端框架（OffloadingManager L329 / CPUOffloadingManager L55 / TieringOffloadingManager L277）；simple_kv_offload = 单 CPU connector（SimpleCPUOffloadScheduler L143 + SimpleCPUOffloadConnector L81） | §4 两表 | §4、2.4 |
| LoRA 单位点参数 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ 、相对 FLOPs 0.5–0.78%（r=16）；8B 全模型 45.09 M 参数 / 90.2 MB / base 的 0.56% | 式 (2) | §5、练习 1 |
| 命题 2：SGMV 每步 $2n_{\mathrm{site}}$ 次 launch（与 $n_{\mathrm{adp}}$ 无关），naive $2n_{\mathrm{adp}}n_{\mathrm{site}}$ ； $n_{\mathrm{adp}}$ = 100 → 2.24 ms vs 224 ms；base 权重只读一份 | 式 (3)、命题 2 | §5、练习 2 |
| Punica 12× 吞吐 / +2 ms（摘要）；repo master 分支 sgmv_expand 未实现；vLLM 生产 kernel = 自有 Triton shrink + expand | §5 坑注 | §5 |
| S-LoRA Unified Paging + 主存常驻； $n_{\mathrm{hot}}$ = 1000 → 90.2 GB 超单卡 HBM | 式 (4) | §5 |
| 命题 3：extra keys 含 lora / mm → 不同 adapter 或图片不共享 prefix cache 块；命中率须按 adapter 分组 | 命题 3、generate_block_hash_extra_keys L627 | §5、§8 |
| adapter 热加载 LRU（AdapterLRUCache L65 / LRUCacheWorkerLoRAManager L276）+ pin_lora L1242 | §5 | §5 |
| 命题 4：sleep 释放 kv_cache 池物理页 → wake_up 后旧 KV 不可信，须重建或经 connector 取回 | 命题 4 + 坑注 | §6、练习 3 |
| sleep 三层：CuMemAllocator L62（tag 分池）/ Worker.sleep L351 / EngineCore.sleep L1103 + LLM.sleep L853；colocate 显存交还 23.9 → 80 GB（3.35×） | §6 表 + 图 | §6、9.3 |
| Sampler 在 gpu/sample/sampler.py L42（v1/sample/ 只剩 ops/）；1.4 各式逐件落点 | §7 表 + 坑注 | §7、练习 4 |
| metrics：PrometheusStatLogger L487 + stats 各类；goodput 是导出量须外部按 attainment 算；α 用 E[K] 口径 | §7 表 | §7、1.1 |
| 特性代价主项各异（verify 算力 / CPU mask / kernel 数 / 取回带宽 / 切换成本）；编译与图捕获是公共底座 | §8 表 + 三论断 | §8、5.3 §6 |

## 练习

1.（计算 · [10]）bitmask 与 adapter 参数代入：(a) 用式 (1) 算 $V$ = 131072、 $B$ = 64 的每步 mask 字节数与每请求字节数；(b) 用式 (2) 算 $r$ = 16 时 8B 章脊（7 位点 / 层：4 个 4096×4096 + 2 个 4096×14336 + 1 个 14336×4096，32 层）的单 adapter 参数量与 bf16 字节数，并求占 base 权重 16.1 GB 的比例。校验：(a) 1 MiB / 16 KiB；(b) 45.09 M / 90.2 MB / 0.56%，与 §3 / §5 一致。
2.（推导 · [20]）命题 2 证明闭环：(a) 按「位点 × adapter」两维独立推出 naive 与 SGMV 的 launch 计数（式 (3)），指明哪一步用到 SGMV 的分段 gather 语义；(b) 代入 $n_{\mathrm{adp}}$ = 100、 $n_{\mathrm{site}}$ = 224、 $t_{\mathrm{launch}}$ = 5 μs 算两个时间，并求 SGMV 时间占 15.1 ms 步时的比例；(c) 论证 base 权重流量与 $n_{\mathrm{adp}}$ 无关，指出 $n_{\mathrm{adp}}$ = 1 的退化边界。校验：(b) 224 ms vs 2.24 ms、14.8%，与 §5 一致；(c) 应引用 1.2 的「权重读一遍、FLOPs 随 B 涨」。
3.（推导 · [30]）命题 4 与 colocate 账：(a) 独立证明 wake_up 后旧 KV 不可信（三步：物理页释放 → 内容不驻留 → 重映射内容未定义），并指出哪一步用到 3.4 命题 2 的「内容可见」前提；(b) 说明 weights 池与 kv_cache 池的处置差异、wake_up(tags) 为何能选择性恢复；(c) 算单卡 80 GB、权重 16.1 GB、KV 池 40 GB 的 sleep 前后训练器可用显存与倍数。校验：(c) 23.9 → 80 GB、3.35×，与 §6 一致；(a) 结论应蕴含「沿用旧 block table 会静默损坏」。
4.（代码 · [30]）走读继承链与 sampling 位置（vllm main，2026-11-23）：先写出预测——(a) `eagle/speculator.py` 与 `mtp/speculator.py` 各有多少行、override 了哪些方法；(b) propose 的实现在哪个类哪一行、它消费 target 的什么张量；(c) Sampler 在哪个文件、`v1/sample/` 下还剩什么。再对照 `gpu/spec_decode/` 继承链与 `gpu/sample/sampler.py` 验证；并说明 ngram 路线为何不在同一条链上。校验：(a) 18 / 63 行；EAGLE 只 override 权重加载，MTP 另 override 四个 prefill / decode 生命周期 hook（index 共享；share_mtp_topk_indices 是类属性、非 override）；(b) TargetDependentARSpeculator propose L103、last_hidden_states / aux_hidden_states（4.5 §5）；(c) sampler.py L42、v1/sample/ 只剩 ops/。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——vLLM 论文锚点（块管理 / COW 背景归 2.1）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——本章全部 file:line 锚点：gpu/spec_decode/ 继承链（BaseSpeculator L74 / DraftModelSpeculator L165 / AutoRegressiveSpeculator L24 / TargetDependentARSpeculator L23 propose L103 / EagleSpeculator L12 / MTPSpeculator L12 / NgramGPUSpeculator L151 / RejectionSampler L152 / OnlineAcceptanceEstimator L345 / AdaptiveVerificationManager L227）、v1/spec_decode/（SpecDecodingStats L19 / dynamic）、config/speculative.py（L113 / L122 / L123 / L475 / L620）、v1/structured_output/（StructuredOutputManager L52 / XgrammarBackend L38 / XgrammarGrammar L234 / backend_guidance llguidance）、v1/kv_offload/（OffloadingManager L329 / CPUOffloadingManager L55 / TieringOffloadingManager L277）+ v1/simple_kv_offload/（SimpleCPUOffloadScheduler L143）、lora/（PunicaWrapperGPU L33 / triton_ops / WorkerLoRAManager L27 / LoRAModelManager L76 / AdapterLRUCache L65）、device_allocator/cumem.py（CuMemAllocator L62）+ Worker.sleep L351 / wake_up L389 + EngineCore / LLM sleep API、v1/metrics/（PrometheusStatLogger L487 / stats.py）、gpu/sample/sampler.py L42 + v1/sample/ops/、multimodal/；2026-11-23 fresh tarball 核实
- [vLLM Team, 2025](../references.md)：vLLM V1: A Major Upgrade to vLLM's Core Architecture（官方 blog，2025-01-27），[C] 开放技术博客、非闭源（key: vllm-v1-blog）——MLLM first-class 三件（预处理独立进程 + cache、image hash 进 prefix caching、encoder cache 支持 chunked prefill）；alpha 限制清单（无 spec decode / structured output / LoRA / Prometheus）已历史化，当前 main 全有（5.1 漂移注）
- [Chen et al., 2023](../references.md)：Punica: Multi-Tenant LoRA Serving，arXiv:2310.18547，[A]（key: punica，punica-ai/punica **master** 分支）——SGMV kernel、GPU 只留一份 base model、12× 吞吐 + 每 token 加 2 ms（摘要逐字）；repo 现状 sgmv_expand = NotImplementedError（§5 坑注）
- [Sheng et al., 2023](../references.md)：S-LoRA: Serving Thousands of Concurrent LoRA Adapters，arXiv:2311.03285，[A]（key: s-lora，S-LoRA/S-LoRA main）——adapter 主存常驻 + 按需取 GPU、Unified Paging、新 TP 策略 + 异构 batching kernel、数千 adapter / up to 4× 吞吐（摘要逐字）；slora/common/mem_manager.py + csrc/bgmv/ + lora_ops.cc dispatch_bgmv
- [Dong et al., 2024](../references.md)：XGrammar，arXiv:2411.15100，[A]（key: xgrammar）——CFG / PDA + 自适应 mask 扩展（1.4 §5 原理），vLLM backend="auto" 解析后的事实默认 structured output backend（§3）
- [guidance repo](../references.md)：guidance-ai/guidance + guidance-ai/llguidance，教学参照（key: guidance）——vLLM backend_guidance.py 实际经 llguidance（Rust 移植）集成（§3 坑注）
- [Willard & Louf, 2023](../references.md)：Efficient Guided Generation for Large Language Models（Outlines），arXiv:2307.09702，[A]（key: outlines）——FSM + 预计算 index（1.4 §5），vLLM backend_outlines.py（§3）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——sleep 对照：Engine.release_memory_occupation L1899 / resume_memory_occupation L1905（tags 分池），详展 6.3
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 SXM 80 GB / 3.35 TB/s、PCIe Gen5 单向 ~64 GB/s（§3 / §6 换算）
- 跨维度指引（纯文字）：training 8.5（RL rollout 四模型显存与权重同步训练侧，§6 recap）、training 8.4（off-policy 校正）；models 4.5（MTP 目标）、models 维度 LoRA 模型侧形式化；inference 1.1（指标定义式 (1)-(3)）、1.2（权重读账）、1.4（采样链与命题 2 / 5）、2.1（块表与容量账）、2.2（hash 链与命中率）、2.4（三层账与临界带宽）、3.2 / 3.3（量化 GEMM 与 sampling kernel）、3.4（launch 账式 (2) 与命题 2）、4.4（接受规则 / 分布保持 / 加速比）、4.5（draft 路线与 verify 形态）、5.1（数据流）、5.2（KV 管理器 / EncoderCacheManager / hash extra keys）、5.3（执行链 / 编译 / HiSparse / connector 基线）、6.1（SGLang 结构化输出）、6.3（SGLang 权重更新）、7.2（DSpark 正式走读，dspark/ 回链）、9.3（RL rollout serving 全账）、9.4（多模态与 SO 专用机）
