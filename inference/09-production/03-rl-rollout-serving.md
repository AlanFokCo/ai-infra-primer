# 9.3 RL Rollout Serving（RL Rollout Serving）

> 训练侧算清了四模型显存、权重同步下界与 off-policy 校正（training 8.5）之后，推理引擎侧还欠什么？——显存怎么在不杀进程的前提下交还、新权重从哪条通道进引擎、长 CoT 截断后怎么续跑、rollout 的时长与吞吐账怎么算。本章只写引擎侧增量，训练侧一律 recap + 纯文字指引不重推 | 前置：training 8.5（LLM-RL 系统实现）/ training 8.4（GRPO 与变体）纯文字指引；inference 5.4 §6（sleep mode 接口与 KV 失效命题）、6.3 §7（checkpoint_engine）、4.2（抢占成本账）、4.4（投机解码）、2.1（KV 容量账）、1.2（算力 / 带宽账） | 锚点：vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]，repo HEAD 6aeedab553dd）、SGLang（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]，HEAD ac0243723b0d）、veRL / HybridFlow（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]，HEAD 7b6c9cc0e19d）；OpenRLHF 引擎接口经 training 8.5 §7 纯文字指引（本维度无独立条目）；代码锚点 2026-11-24 contents API 核实

## 1. 训练侧 recap：三笔账已交付，引擎侧欠四件事

所谓 RL rollout，是指 RL 训练中 actor 用推理引擎对 prompt 采样 response 的阶段——它是 PPO 时间里的大头（约 80%，7B A100 profile，training 8.5 §1 引 OpenRLHF §3.2），也是本章全部账目的负载对象。训练侧的账已在 training 8.5 / 8.4 交付，本章 recap 一句 + 指引，不重推：

| 训练侧已交付 | 结论 recap | 指引 |
| --- | --- | --- |
| 四模型显存 | PPO 全配置同尺寸 38N + KV；GRPO 去 critic + RLVR 去 RM → 20N | training 8.5 §2 式 (1) |
| 引擎拆分 | 训练引擎（Megatron / FSDP）+ 推理引擎分置；layout 不一致引出 resharding | training 8.5 §3 |
| 权重同步下界 | t_sync ≥ 2N_a/BW；70B：NVLink 600 GB/s ≈ 0.23 s、25 GB/s ≈ 5.6 s | training 8.5 §4 式 (2) |
| 3D-HybridEngine | micro-DP 组内 all-gather：通信 (d_g−1)/(tp)·M、峰值 M/(t_g p_g)、冗余 0 | training 8.5 §4 式 (3)、论文 Table 2 |
| off-policy 校正 | 两来源：staleness（版本错位）+ 引擎数值差；bypass vs decoupled；截断 IS / 拒采样 | training 8.5 §5 |
| 长 CoT | response 长度漂移拉大 KV 与时长预算；max_len 截断切断 EOS | training 8.5 §8 |

引擎侧增量是四件事：**显存切换与权重接收接口**（§2 vLLM sleep/wake、§3 SGLang checkpoint_engine / release-resume）；**colocate vs disagg rollout 的显存 / 通信权衡**（§4）；**partial / async rollout 的截断续跑与 staleness 控制**（§5）；**长 CoT 的 serving 含义与 rollout 吞吐账**（§6 / §7），外加 veRL 与 OpenRLHF 的引擎接口对照（§7）。

**本章口径**（R2-2 章脊锁定）：权重 / 同步账沿用 training 8.5 的 7B / 70B bf16 两档（权重字节 = 2N_a）；KV / 长度 / recompute 账沿用 2.1 / 4.2 章脊 Llama-3.1-8B 几何（L = 32、H_kv = 8、d_head = 128、d_model = 4096、bf16，κL = 128 KiB/token，N = 7.50 × 10⁹ matmul 口径，F_eff = 494.75 TFLOPS，单卡 KV 池 48 GiB）；带宽只用 HybridFlow 测试床两档（节点内 600 GB/s、节点间约 25 GB/s）；其余数值为教学构造值，逐处声明。**符号翻译**（spec §4.3）：training 8.5 式 (2) 的 B = 带宽，本章译作 BW——B 在本维度专归 batch；N_a = actor 参数量（沿用 training 8.5）；G = GRPO 组大小（training 约定）；C = 并发序列数、n / m = 训练侧 / rollout 侧 GPU 数（本章声明，PD 实例计数 n_P / n_D 见 8.2 / 9.1）。

## 2. vLLM sleep/wake：不杀进程的显存切换

colocate 部署下训练器要显存时，推理引擎为什么不能直接杀进程？因为重启要重新加载权重、重建 CUDA context 与通信组——冷启动开销每轮都要付一遍，而 sleep/wake 只动物理页、不动进程。vLLM 的答案是 sleep mode：不退出进程、保留虚拟地址布局，释放 GPU 物理显存，训练完 wake_up 恢复——接口三层（cumem 分配器 / worker / 引擎 API）与命题 4（wake_up 后旧 KV 必失效）已在 5.4 §6 交付，本节 recap 一句 + 指引，增量在 level 语义、路径漂移与权重传输引擎族。

> **坑**：按 W3 行号引用 sleep/wake。触发条件：HEAD 自 W3 已移动（vllm 6aeedab553dd，2026-11-23）；现象：gpu_worker.py 的 Worker.sleep / wake_up 按旧锚 L351 / L389 定位落空。本轮事实（2026-11-24 contents API）：**Worker.sleep L361 / wake_up L402**（Worker 类 L233 → L243）；cumem.py 与 llm.py 行号未漂移——CuMemAllocator L62（sleep L261 / wake_up L364 / use_memory_pool L394）、LLM.sleep L853 / wake_up L897 / collective_rpc L587。规避：file:line 以本轮核实为准，跨波引用先对漂移表。

**level 语义**（Worker.sleep(level)，HEAD 6aeedab553dd 三文件互证，2026-11-24 fresh fetch：gpu_worker.py L361-381 + device_allocator/sleep_mode_backend.py CuMemBackend.suspend L126 + cumem.py sleep L261 / wake_up L364）：

- **level 1：权重 offload 到 CPU，只丢 KV**——discardable = ∅（gpu_worker L368-371：level 1 不取任何丢弃集），CuMemBackend.suspend 把 weights / runtime / cudagraph 三个 tag 池 offload 到 host 备份（cumem sleep L292：tag 在 offload 列表且非 discardable → CPU backup），wake_up 自动从备份拷回原权重（L386-387 cudaMemcpy，**无需重新注入**）；kv_cache 池不在 offload 列表、永不备份（backend suspend docstring 逐字「KV cache and workspace scratch are never restored」），物理页释放后须重建（5.4 命题 4）。
- **level 2：丢弃被改写集、保留非改写集**——discardable = 「权重更新会改写的参数」的 storage 地址集（L338 `_discardable_parameters` docstring 逐字「Storage of the parameters the weight update rewrites」；实现 = 全参数减去 sleep_preserve_parameter_names 声明的 preserved 集）：被改写集直接丢弃、无 host 备份（wake_up 后是空显存，由权重更新通道填回），非改写集（preserved）offload 到 CPU、wake_up 恢复；KV 同样丢弃。RL 全参更新场景里 actor 权重每轮都被改写，可保留集合相应很小；它的价值在部分参数不更新的场景（如 LoRA base 权重）。
- **wake_up(tags)**：weights / kv_cache 是 deferrable tags，按 tags 选择性恢复（其余 tag 总是恢复）；kv_cache 唤醒时重建 kv_transfer group——恢复粒度是 tag 池，不是全有全无。

> **坑**：把 level 1 写成「丢弃权重、wake_up 后须重新注入」。触发条件：按二手摘要写——W6 研究笔记语义纠正 4 初版口径「level 1 discard all」即其一，本轮 fresh fetch 推翻；现象：权重处置写反，rollout 循环被设计成「level 1 sleep → wake_up → 重新广播权重」，每轮多付一次本不需要的同步。事实（三文件互证）：level 1 权重 offload 到 host、wake_up 自动恢复、仅 KV 丢弃——与早期公开文档的 offload-to-CPU 描述一致；level 2 才是丢弃被改写集、保留非改写集。规避：level 语义以 HEAD 代码为准（gpu_worker L368-371 / CuMemBackend.suspend / cumem L292），不按摘要口径。

底层机制 recap 一句：CuMemAllocator 用 CUDA virtual memory management 把显存按 tag 分池（weights / kv_cache 两主池，runtime / cudagraph 亦按 tag），sleep 释放物理页而保留虚拟地址，wake_up 重新映射（5.4 §6，不重推）。两 level 的差别只在权重备份策略：level 1 付全量 host 备份 + wake_up 的 H2D 回拷；level 2 免掉被改写集的备份与回拷——RL colocate 每轮都要改写 actor 权重，备份被改写集是纯浪费，level 2 是自然选择（机制层读法）。

显存交还的账（本章换 7B 口径，原因：与 training 8.5 权重账对齐；5.4 §6 用 16.1 GB 权重教学值，得 3.35×）：单卡 80 GB，7B bf16 权重 2N_a = 14 GB + KV 池 40 GB（教学部署值，沿 5.4 §6）= 54 GB，训练器可用 26 GB；sleep level 1 后（权重 offload 到 host、KV 丢弃）GPU 物理页全释放，训练器可用升到 80 GB 量级（80/26 ≈ 3.08×），代价是 host 侧 14 GB 权重备份与 wake_up 的 H2D 回拷。这就是 colocate 的账面收益，代价是每次切换的 sleep/wake 时间与 KV 重建（§4 的 s 项）。

**权重传输引擎族**：`distributed/weight_transfer/` 当前六引擎——nccl / m2n / modelexpress / ipc / sharded_rdt / sparse_nccl（+ base / clients / factory / packed_tensor），训练器 → 推理引擎的权重传输后端族。目录已扩展：W3 登记五引擎，本轮新增 sparse_nccl_engine（2026-11-24 contents API 核实）。veRL 侧的映射：vllm_rollout 的 release（L204）调 sleep.remote、resume（L195）调 wake_up.remote(tags)；sleep_level 由 config.layered_summon 判——开启取 level 1（L125-129，代码带 memory overflow 警告，与 level 1 全量 host 备份语义自洽），否则取 VLLM_SLEEP_LEVEL 常量。「LoRA adapter 模式 level 1 只释 KV」是 sglang_rollout 的判据（L206，§7），两个 backend 勿混——框架把引擎接口包成 rollout worker 的生命周期动作（§7）。

## 3. SGLang checkpoint_engine / release-resume：另一套答案

SGLang 为什么不做 level 语义？它把显存释放的粒度直接暴露为 tags：release_memory_occupation(tags) 按池释放（kv_cache / weights），resume_memory_occupation(tags) 按池恢复——没有 level 1/2 的权重处置分层，释放哪些池由调用方点名。接口族已在 6.3 §7 走读（connector/ 是权重 / 文件存储不是 PD KV 的语义坑，recap 一句 + 指针），本节增量是漂移后的行号与权重更新面的完整登记。

> **坑**：按 W3 行号引用 SGLang 显存 / 权重接口。触发条件：HEAD 自 W3 移动（sglang ac0243723b0d，2026-11-23）；现象：engine.py 三个入口按旧锚 L1899 / L1905 / L1790 定位偏移。本轮事实（2026-11-24 contents API）：**release_memory_occupation L1902 / resume_memory_occupation L1908 / update_weights_from_ipc L1793**（各 +3）；weight_updater.py / checkpoint_engine 文件行号未漂移。规避：同 §2，file:line 以本轮为准。

权重更新面（`managers/scheduler_components/weight_updater.py` SchedulerWeightUpdaterManager L131，行号本轮再确认）：

| 接口 | 锚点 | 职责 |
| --- | --- | --- |
| update_weights_from_disk | L182 | 从磁盘 checkpoint 加载新权重 |
| init_weights_update_group / destroy_weights_update_group | L210 / L215 | 建立 / 销毁分布式权重更新通信组 |
| update_weights_from_distributed | L232 | 从训练侧通信组广播接收 |
| update_weights_from_tensor | L287 | 从内存 tensor 直接更新 |
| flush_cache_after_weight_update | L164 | 权重更新后清空 KV cache |
| record_weight_version_after_update | L179 | 更新后记录权重版本（staleness 监控的引擎侧落点） |

flush 不是可选优化：权重是 cache 版本元组的分量（2.4 §8 recap），旧权重算出的 KV 对新权重无效——与 vLLM 命题 4（KV 必失效）同一条正确性边界，只是触发时机不同（vLLM 在 sleep 释放物理页时失效，SGLang 在权重更新时主动 flush）。

checkpoint_engine/ 是外部集成通道：`checkpoint_engine/update.py` 是 torchrun CLI（update_weights L144；split_checkpoint_files L79 / split_tensors L92 / req_inference L113 / join L182），`checkpoint_engine_worker.py` SGLangCheckpointEngineWorkerExtension L39 / Impl L95 走 zmq IPC，依赖外部 MoonshotAI/checkpoint-engine——与 6.3 §7 登记的语义一致，行号本轮再确认。

两引擎接口对照（增量在粒度与入口形态）：

| 轴 | vLLM | SGLang |
| --- | --- | --- |
| 显存释放 | Worker.sleep(level)：level 定权重备份策略（1 权重 offload 到 host / 2 丢弃被改写集、保留非改写） | release_memory_occupation(tags)：tags 点名释放池 |
| 显存恢复 | wake_up(tags) 按 tag 恢复 + 重建 kv_transfer group | resume_memory_occupation(tags) |
| 权重更新入口 | collective_rpc + weight_transfer/ 六引擎 | from_disk / from_distributed / from_tensor / from_ipc 四入口 |
| 版本登记 | set_weight_version / compute_weight_checksums（5.4 §6 recap） | record_weight_version_after_update L179 |
| KV 失效边界 | sleep 释放 kv_cache 池 → 旧 KV 必失效（5.4 命题 4） | 权重更新 → flush_cache_after_weight_update L164 |

## 4. colocate vs disagg rollout：显存切换 vs 权重广播

同一批卡既训练又 rollout，和 rollout 独占一批卡，每轮迭代各付什么代价？路线：把每轮时间拆成 rollout / train / 切换 / 同步四项，两种放置分别代入，再比 GPU·时间成本——关键一步是认出 **colocate 把两相串行化、disagg 把两相并行化但付常驻卡与跨集群同步**。所谓 colocate rollout，是指 rollout 与训练交替占用同一批卡（时分复用，靠 §2 / §3 的显存切换错峰）；disagg rollout 是指 rollout 引擎常驻独立卡池，与训练集群分离（权重经广播同步）。

设 T_r = rollout 相时长、T_t = 训练相时长（含打分与 advantage，教学口径）。两种放置的每轮迭代时间：

$$
T_{\mathrm{iter}}^{\mathrm{co}} = T_r + T_t + s \qquad (1)
$$

$$
T_{\mathrm{iter}}^{\mathrm{dis}} = \max(T_r, T_t) + r \qquad (2)
$$

**符号**：s = colocate 切换开销（口径：sleep/wake 物理页操作 + 本地权重同步；新批 KV 冷启动 prefill 计入 T_r，retained 前缀重建场景见 §5）；r = disagg 跨集群权重广播耗时。代入教学构造值（T_r = T_t = 300 s、s = 20 s、r = 5.6 s）：colocate 迭代 620 s、disagg 305.6 s——墙钟差约 2×，GPU·时间账见命题 1。r 的下界即训练侧同步账（training 8.5 §4 式 (2)，本章译文）：

$$
t_{\mathrm{sync}} \ge \frac{2 N_a}{\mathrm{BW}} \qquad (3)
$$

**符号**：N_a = actor 参数量；BW = 同步路径有效带宽（bytes/s；training 8.5 原记号 B，本章译 BW，B 专归 batch）。代入 70B bf16（2N_a = 1.4 × 10¹¹ bytes）：600 GB/s → ≈ 0.23 s，25 GB/s → ≈ 5.6 s——与 training 8.5 §4 表同行同值（recap 对账）。disagg 的 r 落在跨节点链路时以 5.6 s 起步；colocate 的 s 里本地同步走 NVLink 域，亚秒级。

每轮 GPU·时间成本（卡数 × 迭代时长）：colocate 用 n 卡，W_co = n(T_r + T_t + s)；disagg 用 n 训练卡 + m rollout 卡，W_dis = (n + m)(max(T_r, T_t) + r)。

**命题 1**（平衡情形判据）：当 T_r = T_t = T 且 m = n 时，disagg 的每轮 GPU·时间低于 colocate 当且仅当 2r < s。**证明**（路线：两式相减，符号由差项定）：W_co = n(2T + s)，W_dis = 2n(T + r)，相减得

$$
W_{\mathrm{co}} - W_{\mathrm{dis}} = n (s - 2r) \qquad (4)
$$

各量皆正，差的符号由 s − 2r 决定，故判据成立。证毕。代入教学构造值（70B，n = m = 64，T = 300 s，s = 20 s 含切换与 NVLink 同步，r = 5.6 s 跨节点）：W_co = 64 × 620 = 39,680 GPU·s，W_dis = 128 × 305.6 = 39,116.8 GPU·s，disagg 省 563.2 GPU·s（≈ 1.4%）；判据 2r = 11.2 < 20 ✓。边界反例：跨集群慢链路使 r = 15 s 时 2r = 30 > 20，colocate 反超——判据对链路带宽敏感，不是「分卡总是贵」或「共卡总是省」。

读账三条。其一，平衡负载下两模式 GPU·时间差由切换 / 同步开销项决定，不由并行度决定——disagg 迭代墙钟 305.6 s 对 colocate 620 s 快约 2×，但卡数也 2×，GPU·时间几乎打平。其二，真实决策变量是卡数预算（colocate 省卡）与迭代延迟（disagg 快），外加异步潜力（§5：disagg rollout 卡可超前跑，colocate 不能）。其三，负载不平衡时（80% profile 意味着 T_r ≈ 4T_t），同步 disagg 的训练卡在 rollout 相空闲（口径：同步流水线、无异步超前），GPU·时间反而可能更高——这是 async rollout 的动机，不是 disagg 的固有缺陷。

```
colocate（n 卡时分复用）              disagg（n 训练卡 + m rollout 卡）
──────────────────────────────       ──────────────────────────────
时间 →                                时间 →
[rollout T_r][train T_t][s]           训练卡:  [train T_t][等待…]
 同一批卡交替占两相                    rollout卡: [rollout T_r ……]
迭代 = T_r + T_t + s                  迭代 = max(T_r, T_t) + r
s = 切换 + 本地同步                    r = 跨集群权重广播（式 (3) 下界）
```

代码对应物：veRL V1 的 trainer_colocate_async.py（同卡池）与 trainer_separate_async.py（独立 rollout 卡 + hybrid 池）；OpenRLHF 对 70B 以上默认四模型分卡、另留 colocate 开关（training 8.5 §7 指引）。放置与同步带宽是设计输入（training 8.5 §4 提点 recap）：等跑起来才发现同步走慢链路，再改放置的代价就高了。

## 5. partial rollout / async rollout：长 CoT 截断续跑

长 CoT 的一条轨迹要跑几万 token，为什么不能让它一口气跑完再更新权重？因为同步迭代里 rollout 相被最长轨迹拖住（§6 命题 3），训练卡干等。async rollout 的答案是让 rollout 与训练解耦进度：所谓 async rollout，是指 rollout 引擎不等训练完成就持续采样、训练侧按 staleness 预算消费轨迹的流水线形态；partial rollout 是它的轨迹层配套——长轨迹在迭代边界被 abort，保留已生成部分，权重更新后续跑。

**版本事实先钉住**（2026-11-24 contents API，verl 7b6c9cc0e19d）：veRL mainline 异步训练 = **V1 async trainer**（trainer/ppo/v1/ 的 trainer_colocate_async / trainer_separate_async + replay_buffer + TransferQueue，docs/advance/v1_async_trainer.md）。

> **坑**：按 fully_async / one_step_off 文档写 veRL 当前异步形态。触发条件：docs/advance/{fully_async,one_step_off}.md 现为 stub，逐字「This implementation has moved to verl-recipe … pinned to verl v0.9.1 and use its V0 trainer」；现象：把已迁出的 V0 实现当 mainline。规避：mainline 异步以 V1 async trainer 为准；fully_async / one_step_off 去 verl-recipe（pinned v0.9.1）找。

**partial rollout 是 V1 async client 行为，不是配置开关**（语义口径）：abort 时保留 tokens + logprobs → LB 重试续跑 → 续跑用更新后的模型版本 → 保留前缀的 KV 重建后继续 decode。四步里引擎只看见「带长前缀的新请求」——轨迹层的暂停 / 恢复与引擎层的 sleep/wake 是两层接口（口径：前者保 tokens + logprobs 在 replay buffer / LB，后者保虚拟地址释放物理页；勿混）。

```
partial rollout：一条长 CoT 轨迹（V1 async client 行为）
──────────────────────────────────────────────────────────
版本 v_i rollout：prompt → decode … 已生成 T_pre tokens
   │ abort（迭代边界 / 资源切换）
   ▼ 保留 tokens + logprobs（replay buffer / LB，不在引擎显存）
权重更新 v_i → v_{i+1}（§4 同步通道）
   │ LB 重试：保留前缀 + 续跑请求重新入引擎
   ▼ 前缀 KV 重建（一次 prefill，4.2 式 (1) 口径）
版本 v_{i+1} 继续 decode → EOS / 下一次 abort
代价：额外 prefill + 轨迹内策略版本混合（命题 2）
```

重建成本账（recap 4.2 式 (1)，不重推）：保留前缀 T_pre = 8192，8B 章脊口径 recompute FLOPs = 2NT_pre + 2LT_pre²d_model ≈ 1.405 × 10¹⁴，按 F_eff = 494.75 TFLOPS 得 ≈ 284 ms；对照丢弃前缀重新采样——8192 个 token 逐个 decode，按 5 ms/token（4.4 章脊口径）要 ≈ 41 s。**重建比重采便宜约 144×**：prefill 是 compute-bound 一次过，decode 重采是 memory-bound 串行步——这就是「保留 tokens、重建 KV」优于「整条重采」的账面依据。

**命题 2**（轨迹内策略版本混合）：跨 k 次权重更新续跑的轨迹，其前缀 token 采样自旧版本策略、后缀采样自新版本策略，整条轨迹的采样分布不是单一策略。**证明**（构造）：abort-retain 机制保留的前缀由版本 v_i 采样（机制定义），续跑段由 v_{i+k} 采样（LB 重试用更新后权重），两段分布不同（权重更新改变策略），故轨迹是分段混合分布。证毕。边界：abort 与续跑之间若权重未更新（版本不变），退化为单一策略——混合不是必然，取决于是否跨版本。后果归 off-policy 校正：staleness 控制之外还要处理轨迹内版本差，序列级 ratio（training 8.4 §7 GSPO）与截断 IS / 拒采样（training 8.5 §5）是算法侧承接点，本章不重推。

**staleness 控制**（V1 async trainer，docs/advance/v1_async_trainer.md）：max_off_policy_threshold 默认 8 个模型版本——轨迹所用权重落后当前训练版本超过阈值即触发策略，strategy 取 drop 或 wait（丢弃重采 / 等待追平）；配套 trajectory_spans / staleness 监控。阈值是权衡预算：放宽阈值省重采、但 ratio 方差随版本差涨（training 8.5 §5 的 clip 区间分析 recap）。引擎侧的 pause/resume 接口（vLLM sleep/wake、SGLang release/resume）服务于显存时分；partial rollout 的截断续跑发生在轨迹层，两层各司其职。

## 6. 长 CoT 长度爆炸：serving 侧含义

RL 训练中 response 长度会漂移（DeepSeek-R1 记录了平均 response 持续增长，training 8.5 §8 指引），serving 侧先感受到的是 KV 容量。容量账 recap 2.1（不重推）：

$$
\mathrm{Mem}_{\mathrm{KV}} = C \cdot \bar{T} \cdot \kappa L \qquad (5)
$$

**符号**：C = 并发序列数；T̄ = 平均序列长（prompt + 已生成）；κL = 每 token 全层 KV 字节（章脊 128 KiB）。代入单卡 8B 章脊：8 个 prompt、G = 8（GRPO 每 prompt 采 G 条，training 8.4 §1），并发 C = 8 × 8 = 64 条，T̄ = 4096 → 64 × 4096 × 128 KiB = 32 GiB，装进 48 GiB KV 池；长度漂移 T̄ → 8192 → 64 GiB，**超池 33%**——触发抢占。GRPO 的 G 是乘数：组大小翻倍，KV 与生成 token 数同步翻倍（training 8.5 §2「省下的预算被搬到 KV 项」recap）。

抢占的代价 recap 4.2 §2：vLLM V1 只走 recompute（释放 KV、恢复时一次 prefill 重算），T = 8192 的 recompute ≈ 284 ms（§5 同账）；长度越长二次项占比越高（4.2 表：T = 32768 时 attention 项占 36%）。serving 侧配套三条：长度预算 / 截断率 / 有效样本率纳入每轮监控（training 8.5 §8）；max_len 截断切断 EOS、规则验证器判不了对错（算法对策 DAPO overlong / dynamic sampling 归 training 8.4 §6，指引不展开）；partial rollout（§5）把长轨迹拆段，避免单条轨迹独占 KV 到迭代结束。

时长侧的结论更硬：**rollout 时长由 batch 内最长序列决定**，这不是经验观察，是结构下界。

**命题 3**（rollout 时长下界）：run-to-completion 口径下（迭代结束 = 全部序列完成），T_roll ≥ max_i T_i · t_step,min。**证明**（路线：自回归串行依赖 → 每 token 一步 → 单步下界，三步）：序列 i 的 T_i 个 token 必须逐个串行生成（自回归定义，前缀不可变命题归 models 2.4 / inference 2.1 recap）；每步至少 t_step,min——decode 是 memory-bound，单步须读一遍全量权重，这是 1.2 带宽账的下界而非拟合值；迭代须等最慢序列完成，故下界由最长序列定。证毕。代入：max_i T_i = 32768、t_step,min = 5 ms（4.4 章脊 8B @ H100 单步口径）→ T_roll ≥ 163.84 s ≈ 164 s。对照 §7 的平均口径估算 83.9 s——**最长序列到 32768 时，下界已超平均估算**，尾部决定时长。边界：partial rollout（§5）用 abort-续跑把迭代边界与序列完成解耦，下界转而约束单条轨迹的累计墙钟，不再约束单轮迭代——这正是它缓解尾部阻塞的机制。

## 7. 引擎接口对照与 rollout 吞吐账

veRL 与 OpenRLHF 的 rollout 引擎接口差在哪？差在**可换引擎性**：veRL 是三引擎 backend，OpenRLHF 是 vLLM-only（全 repo 无 sglang，2026-11-24 tarball 26494490 grep 核实；OpenRLHF 事实经 training 8.5 §7 纯文字指引，本维度不引其 arXiv 条目）。

veRL 三引擎（workers/rollout/，hybridflow 锚 [Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]）：vllm_rollout/vllm_rollout.py ServerAdapter L49、sglang_rollout/sglang_rollout.py ServerAdapter L128、trtllm_rollout/ 亦在树。权重同步抽象成 checkpoint_engine/ 多后端（nccl / nixl / mooncake / kimi / hccl / delta + base + delta_sync/）。接口动作对照：

| 动作 | veRL vllm_rollout | veRL sglang_rollout | OpenRLHF（vLLM-only） |
| --- | --- | --- | --- |
| 权重更新 | update_weights L210：collective_rpc update_weights_from_ipc + BucketedWeightSender（zmq，bucket_size_mb，use_shm 共享内存） | update_weights L305：tensor buckets CUDA IPC（MultiprocessingSerializer + UpdateWeightsFromTensorReqInput，参考 THUDM/slime） | broadcast_to_vllm L409：NCCL 广播 / ray.util.collective + CUDA IPC _handle_cuda_ipc L439 |
| 显存释放 | release L204 → sleep.remote（sleep_level 由 layered_summon 判，§2） | release L288 → release_memory_occupation(tags=[kv_cache, weights])；sleep_level 1 LoRA 只释 kv_cache（L206 判据） | vllm_engine.py sleep L136 |
| 显存恢复 | resume L195 → wake_up.remote(tags) | resume L276 → resume_memory_occupation(tags) | wake_up L139 |
| 权重接收 | vLLM weight_transfer/ 六引擎（§2） | SGLang 四入口（§3） | update_weight L115 / update_weight_cuda_ipc L121 |

OpenRLHF 侧两个细节（training 8.5 §4/§7 已走读，recap + 指引）：ZeRO-3 下先 GatheredParameters（L472）把分片聚回再广播——直接播分片，rollout 侧拼到的是碎片（training 8.5 §4 坑注）；CUDA IPC 路径 reduce_tensor → IPC handle → all_gather_object → update_weight_cuda_ipc，省一次拷贝。选型论断（接口级，非性能排名）：**veRL 三引擎 = 引擎可换**——SGLang radix 适配多轮 / 共享前缀的 RL workload、TRT-LLM 适配 NVIDIA 栈（9.2 §4）；**OpenRLHF vLLM-only = 单引擎接口深耕**（NCCL + IPC 双路径），换引擎须自建适配层。

**rollout 吞吐账**（spec 关键推导）：路线：先算每轮生成 token 总量（采样并发 × 组大小 × 平均长度），除以引擎聚合吞吐得平均口径时长，再用命题 3 下界校核尾部，最后看投机解码的加速因子。每轮生成 token 数 = n_prompt · G · L̄_out（GRPO 每 prompt 采 G 条），平均口径 rollout 时长：

$$
T_{\mathrm{roll}} \approx \frac{n_{\mathrm{prompt}} \cdot G \cdot \bar{L}_{\mathrm{out}}}{\Theta} \qquad (6)
$$

**符号**：n_prompt = 每轮 prompt 数；G = GRPO 组大小；L̄_out = 平均 response 长度；Θ = rollout 集群聚合 decode 吞吐（tokens/s，口径：教学构造值，实测按引擎 profile 代入）。换集群级口径（原因：单卡装不下全批——4096 序列 × 4096 token × 128 KiB = 2 TiB KV，远超单卡 48 GiB 池）：n_prompt = 512、G = 8、L̄_out = 4096 → 每轮 16,777,216 ≈ 1.678 × 10⁷ token；Θ = 2 × 10⁵ tokens/s → T_roll ≈ 83.9 s。命题 3 校核：最长序列 32768 时下界 164 s > 83.9 s，平均口径低估——长度方差大时以尾部为准（§6）。

投机解码是吞吐乘数：引擎开 MTP / draft-verify 时，有效吞吐乘加速因子 $F = (1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$ （4.4 命题 3 recap， $c = c_d/c_v$ 无量纲，不重推）。代入 4.4 章脊 α = 0.8、γ = 4、c = 0.1：F ≈ 2.40，T_roll ≈ 83.9/2.40 ≈ 35 s。**前提是分布保持**（4.4 §2 命题 recap）：投机解码不改变目标分布，rollout 采样语义才不被破坏——这是 RL 能用投机解码的正确性依据；rollout 返回的 rollout_log_probs 供训练侧 bypass / decoupled 校正消费（training 8.5 §5 指引）。

## 总结

| 论断 | 账 / 证据 | 回指 |
| --- | --- | --- |
| 训练侧三笔账（38N + KV / t_sync ≥ 2N_a/BW / 3D-HybridEngine）recap 不重推；引擎侧增量 = 显存切换 / 权重接收 / 放置 / 续跑 / 吞吐 | training 8.5 §2/§4 式 (1)(2)(3)；本章式 (3) 译文 | §1、练习 1 |
| vLLM sleep level 1 权重 offload 到 CPU（wake_up 自动恢复）、仅 KV 丢弃；level 2 丢弃被改写集、非改写集 offload 到 host；wake_up(tags) 按池恢复 + 重建 kv_transfer group | Worker.sleep L361 / wake_up L402（W3 L351/L389 漂移）；L368-371 discardable + CuMemBackend.suspend L126 + cumem L292/L386-387 三文件互证 | §2、坑注、5.4 §6 |
| SGLang 释放粒度 = tags 点名（kv_cache / weights），无 level 分层；权重更新四入口 + flush KV + 版本登记 | engine.py L1902/L1908/L1793（+3 漂移）；weight_updater L131 族 | §3、坑注、6.3 §7 |
| wake_up 后旧 KV 必失效（vLLM）与权重更新必 flush（SGLang）是同一条正确性边界的两种触发 | 5.4 命题 4；2.4 §8 cache 版本元组 | §2/§3 |
| 平衡情形（T_r = T_t、m = n）disagg GPU·时间更省当且仅当 2r < s；墙钟快 2× 但卡数 2×，GPU·时间几乎打平 | 命题 1、式 (4)；64 卡例省 563.2 GPU·s（1.4%） | §4、练习 2 |
| partial rollout = V1 async client 行为（abort 保留 tokens + logprobs → LB 续跑 → 新版本 + 前缀 KV 重建），非配置开关；fully_async/one_step_off 已迁 verl-recipe pinned v0.9.1 | v1_async_trainer.md；stub 逐字 | §5、坑注 |
| 前缀重建（一次 prefill ≈ 284 ms）比丢弃重采（≈ 41 s）便宜约 144×；代价 = 额外 prefill + 轨迹内策略版本混合 | 4.2 式 (1) 口径；命题 2 | §5、练习 3 |
| staleness 预算 max_off_policy_threshold 默认 8 版本，strategy drop 或 wait；轨迹内版本差归序列级 ratio / 截断 IS | V1 async trainer；training 8.4 §7 / 8.5 §5 指引 | §5 |
| 长度漂移 → KV 超池 → recompute 抢占；rollout 时长下界由最长序列定（结构结论非拟合） | 式 (5)：32 → 64 GiB 超 48 GiB 池；命题 3：32768 × 5 ms = 164 s | §6、练习 3 |
| veRL 三引擎（vllm/sglang/trtllm）= 引擎可换；OpenRLHF vLLM-only = 单引擎深耕（NCCL + IPC 双路径，ZeRO-3 先聚回再播） | ServerAdapter L49/L128；broadcast_to_vllm L409（training 8.5 §7 指引） | §7、练习 4 |
| rollout 吞吐账 = n_prompt·G·L̄_out/Θ；投机解码乘 F（分布保持是前提）；尾部用命题 3 校核 | 式 (6)：83.9 s；F ≈ 2.40 → 35 s | §7、练习 3 |

## 练习

1.（计算 · [10]）同步下界与 level 语义：(a) 用式 (3) 算 7B / 70B bf16 在 600 GB/s 与 25 GB/s 两档的每轮同步下界（四格）；(b) 判别：sleep level 1 与 level 2 各如何处置权重？wake_up 后旧 KV 能否沿用？校验：(a) 与 training 8.5 §4 表同行同值（0.023 / 0.56 / 0.23 / 5.6 s）；(b) 与 §2 level 语义（HEAD 三文件互证）及 5.4 命题 4 一致：权重按 level 或 offload 到 host 或丢弃被改写集，KV 两 level 均必失效。
2.（推导 · [20]）命题 1 推广（m ≠ n）：(a) 平衡情形 T_r = T_t = T 下，从 W_co = n(2T + s)、W_dis = (n + m)(T + r) 推出 disagg 更省的 m 上界（用 T、s、r、n 表示）；(b) 代入 T = 300 s、s = 20 s、r = 5.6 s、n = 64，判断 m = 32 与 m = 64 各是否更省；(c) 给边界反例：r 满足什么条件时 m = n 也救不回 disagg。先写出你对 (b) m = 64 的预测再算。校验：(a) m = n 时退化为式 (4) 判据 2r < s；(c) 与 §4 边界反例（r = 15 s）一致。
3.（计算 · [30]）长 CoT 与 partial rollout 账（8B 章脊，κL = 128 KiB/token，F_eff = 494.75 TFLOPS，t_step,min = 5 ms）：(a) C = 64、T̄ 从 4096 漂到 8192，用式 (5) 算 KV 占用并判断是否超 48 GiB 池；(b) T_pre = 8192 的前缀重建 FLOPs 与时间（4.2 式 (1)），对照丢弃重采 8192 × 5 ms，求比值；(c) 用命题 3 算 max_i T_i = 32768 的 rollout 时长下界，与式 (6) 的 83.9 s 平均估算比较并解释谁决定时长。校验：(a) 32 → 64 GiB（超池 33%）；(b) ≈ 284 ms 对 ≈ 41 s（约 144×）；(c) 164 s > 83.9 s，与 §6 数值一致。
4.（代码 · [30]）引擎接口走读（2026-11-24 contents API 通道）：(a) veRL workers/rollout/ 下三个 backend 目录与 ServerAdapter 行号（vllm L49 / sglang L128）；(b) sglang_rollout release L288 的 tags 参数与 sleep_level 1 LoRA 语义；(c) vllm_rollout update_weights L210 走哪条 RPC + 发送件；(d) 先预测再核对：OpenRLHF ZeRO-3 下若跳过 GatheredParameters 直接广播本 rank 分片，rollout 侧拿到什么？校验：(a)(b)(c) 与 §7 表 file:line 一致；(d) 碎片而非完整模型（training 8.5 §4 坑注）；OpenRLHF 出处经 training 8.5 §7 指引核对，不在本维度引 arXiv 条目。

## 参考

- [Kwon et al., 2023](../references.md)：vLLM，arXiv:2309.06180，[A]（key: vllm-repo，HEAD 6aeedab553dd，2026-11-24 contents API 深核 + fix 轮 fresh fetch 三文件互证）——device_allocator/cumem.py CuMemAllocator L62（sleep L261 / wake_up L364 / use_memory_pool L394；sleep L292 offload→CPU backup / wake_up L386-387 cudaMemcpy 恢复）；v1/worker/gpu_worker.py Worker.sleep L361 / wake_up L402（W3 L351/L389 漂移 +10/+13，Worker 类 L233→L243；L338 _discardable_parameters = 被改写集 / L368-371 level 1 discardable=∅、level 2 取被改写集）；device_allocator/sleep_mode_backend.py CuMemBackend.suspend L126（offload weights/runtime/cudagraph，KV 永不备份）——level 1 权重 offload 到 host 自动恢复、仅 KV 丢弃（研究笔记语义纠正 4 初版口径「level 1 discard all」被本轮推翻）；entrypoints/llm.py sleep L853 / wake_up L897 / collective_rpc L587；EngineCore.sleep/wake（v1/engine/core.py）；distributed/weight_transfer/ 六引擎（nccl / m2n / modelexpress / ipc / sharded_rdt / sparse_nccl，W3 五 → 本轮六）；接口三层与命题 4 recap 5.4 §6
- [Zheng et al., 2023](../references.md)：SGLang，arXiv:2312.07104，[A]（key: sglang-repo，HEAD ac0243723b0d，2026-11-24 contents API 深核）——entrypoints/engine.py release_memory_occupation L1902 / resume_memory_occupation L1908 / update_weights_from_ipc L1793（W3 L1899/L1905/L1790 漂移 +3）；managers/scheduler_components/weight_updater.py SchedulerWeightUpdaterManager L131（update_weights_from_disk L182 / init_weights_update_group L210 / destroy_weights_update_group L215 / update_weights_from_distributed L232 / update_weights_from_tensor L287 / flush_cache_after_weight_update L164 / record_weight_version_after_update L179）；checkpoint_engine/update.py update_weights L144（split_checkpoint_files L79 / split_tensors L92 / req_inference L113 / join L182）；checkpoint_engine_worker.py SGLangCheckpointEngineWorkerExtension L39 / Impl L95（zmq IPC，外部 MoonshotAI/checkpoint-engine）；connector 语义坑 recap 6.3 §7
- [Sheng et al., 2024](../references.md)：HybridFlow（veRL），arXiv:2409.19256，[A]（key: hybridflow，HEAD 7b6c9cc0e19d，2026-11-24 tree + contents 深核）——workers/rollout/{vllm_rollout/vllm_rollout.py ServerAdapter L49 + update_weights L210 / release L204 / resume L195 + sleep_level layered_summon 判据 L125-129；sglang_rollout/sglang_rollout.py ServerAdapter L128 + update_weights L305 / release L288 / resume L276 + sleep_level LoRA 判据 L206（level 1 只释 kv_cache）；trtllm_rollout/}；checkpoint_engine/ 六后端（nccl / nixl / mooncake / kimi / hccl / delta 各一个 checkpoint_engine 实现 + base + delta_sync/）；trainer/ppo/v1/（trainer_colocate_async / trainer_separate_async / replay_buffer）+ docs/advance/v1_async_trainer.md（partial rollout abort-retain-resume + max_off_policy_threshold=8 + strategy drop 或 wait + trajectory_spans/staleness 监控）；fully_async/one_step_off 迁 verl-recipe pinned v0.9.1（stub 逐字）；训练侧 3D-HybridEngine 与实测 recap training 8.5 §4/§6
- OpenRLHF 引擎接口（vLLM-only）：经 training 8.5 §7 纯文字指引锚定（该维度 references 条目）——ppo_actor.py broadcast_to_vllm L409（NCCL / ray.util.collective + _handle_cuda_ipc L439，ZeRO-3 GatheredParameters L472）；vllm_engine.py update_weight L115 / update_weight_cuda_ipc L121 / sleep L136 / wake_up L139；全 repo 无 sglang（tarball 26494490，2026-11-24 grep 核实）。本维度无独立条目，不引其 arXiv ID（lint 白名单纪律）
- 训练侧指引（纯文字，不挂链接）：training 8.5 §2 式 (1)（四模型显存）/ §4 式 (2)(3)（同步下界与 3D-HybridEngine）/ §5（off-policy 两来源与 bypass/decoupled）/ §7（OpenRLHF 对照）/ §8（长 CoT）；training 8.4 §1（GRPO 组大小）/ §6（DAPO）/ §7（GSPO 序列级 ratio）
