# 7.1 DeepInfer 架构走读（DeepInfer Architecture Walkthrough）

> 一个日服务 155 万亿 token 的生产引擎，开源 release 却只含「单引擎」功能——release 边界到底画在哪，缺的每一件对应完整 serving 系统的哪个组件？本章走读 DeepInfer 的 repo 结构、step 循环、双平台抽象与 config 映射，给出 release 边界 ↔ 完整 serving 系统的缺口映射命题，并为 7.2-7.4 立架构对照基线 | 前置：inference 5.1（vLLM V1 进程架构）、6.1 / 6.2（SGLang 架构与 overlap 调度）、4.1（迭代级调度）、2.1（块管理）、1.1（在线 / 离线分界）；models 2.4（KV cache 容量） | 锚点：DeepInfer [教学参照 + 官方实现]（deepseek-ai/DeepInfer main@a679bb86，README + doc/inference_deployment_qa.md v0.2，2026-11-23 fresh tarball file:line 核实）；V4.1-Flash config [W]（hf-mirror 传输核实）；生产数字一律「官方披露」口径 + doc 行号；vLLM [A] / SGLang [A] 为对照基线（5.1 / 6.1 已走读）

## 1. 定位：生产引擎的开源版，三个口径先钉死

单集群日服务 155 万亿 token、每 GPU-hour 38 USD——这是 DeepInfer 承载的生产规模（[DeepInfer doc](../references.md)，官方披露，doc L17，核实 2026-11-23）。先钉口径：38 USD/GPU-hour 是**收入**（off-peak 价），doc 原文用词是 "overall revenue" 并注明 "at the off-peak price"，不是成本。

> **坑**：把 38 USD/GPU-hour 当成本口径。触发条件：代入 9.1 的 USD/M tokens 成本公式；现象：把收入当成本，成本账整体失真——成本侧官方未披露。规避：该数只作生产规模与商业形态的参照；成本模型用 9.1 自己的假设。

所谓 DeepInfer，是指 DeepSeek 生产推理引擎的开源 release（README L3 逐字 "the high-performance inference framework that powers DeepSeek's large-scale production services"），不是独立研究原型；证据等级 = 教学参照 + 官方实现（无 paper，README 与 doc 为官方一手，生产数字不升 [A]、不降 [C]）。定位有三个硬口径，全部来自 README 一手（2026-11-23 核实）：

| 口径 | README 事实 | 含义 |
| --- | --- | --- |
| offline inference release | L3 逐字 "This open-source release provides offline inference with the core features needed for serving at scale" | 只含离线推理（1.1 的分界：无 SLO 约束的批量生成）；`entrypoints/` 只有 `llm.py`，无 API server（§3） |
| V4.1-Flash only | L7 Supported models 仅 DeepSeek-V4.1-Flash | 引擎不支持第二个模型几何，`_assert_supported` 硬门（§5） |
| 双平台 | L35-36 NVIDIA SM100 GPUs / Huawei Ascend 950 NPUs；L48-52 Linux / GCC ≥ 13.0 / Python 3.12 / CUDA 13（NVIDIA）/ CANN 9.2（Ascend）；L61 `DEEPINFER_ACCELERATOR=nvidia` 或 `ascend` | 一套引擎两个硬件后端，`platform.py` 抽象（§3） |

许可证 Apache-2.0（L91）。lineage 同样是一手口径：L95 逐字 "originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM"——vLLM 是主要代码基座，SGLang 与 TensorRT-LLM 是设计启发；async scheduling 另自认仿 SGLang / NanoFlow（doc L334，§4 回链 6.2）。README L11-18 列八项特性，每项的走读落点：

| README 特性（L11-18） | 机制一句 | 走读章 |
| --- | --- | --- |
| Optimized kernels, model and sampler runtime | FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect + custom TileLang ops（L23 链接） | 7.3 |
| Attention DP / balanced CP, MoE EP + EPLB | #DP = #EP（doc L33）；EPLB 引擎侧在 release | 7.4 / 8.1 |
| DSpark and confidence-scheduled verification | 变长 verify；引擎实现名 dflash（§5 坑注） | 7.2 |
| SWA bounded replay | 末 128 token 恒定成本重建 SWA（sliding window attention，滑窗注意力） | 7.2 |
| Asynchronous scheduling (zero-overhead batch scheduler) | 默认开启（L26），ZOS（§4） | 本章 §4，6.2 回链 |
| Batch-invariant inference | `DEEPINFER_BATCH_INVARIANT=1`，NVIDIA only（L27） | 3.4 回链 |
| Deterministic sampling | BATCH_INVARIANT=1 + ENABLE_MTP_VARLEN=0 + 固定 `seq_seed`（L28） | 1.4 回链 |
| Structured outputs | `so_args={"schema_type","schema"}`（L29，examples/_structured_output.py） | 本章 §3，9.4 |

**本章数值口径**（章脊锁定）：模型几何 = V4.1-Flash config [W]（§5 表，只列 serving 字段不推导）；硬件 = SM100（README L83 benchmark 口径：prefill 4 卡 / decode 64 卡，各在一个 scale-up domain 内）；repo 锚 = main@a679bb86（2026-11-03 commit，doc v0.2；与 W1 同日 tarball byte-identical，W1-W3 已交付锚点全部维持）；生产数字 = 官方披露 + doc 行号，核实 2026-11-23。Ascend 950 侧本章只登记双平台抽象，kernel 性能数归 7.3。

## 2. Release 边界：三项不含，与一条实例边界命题

README 明说三项不在 release，但 PE/CE 角色（§3）、StateBlock 落盘（7.2）、EPLB（7.4）的引擎侧代码又确实在 repo 里——边界到底画在哪？README L38-42 逐字列三项（"Required for a complete inference service (not included in this release)"）：其一，A KV cache system；其二，VE/Prefill/Decode disaggregation，括注逐字 "most required engine functionality is already implemented"；其三，Load balancing across data-parallel replicas for both prefill and decode。

> **坑**：把 release 边界的耦合原因归到 README。触发条件：引「与内部 infra 耦合所以未开源」时标 README 出处；现象：README 只列三项清单，原因与架构指针在 doc L21——逐字 "tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced"，官方知乎架构长文 URL 也在 doc L21（正文 403 bot 拦截未读；doc 自述 "This Q&A builds on that blog post"，以 doc 为一手即足）。规避：清单归 README L38-42，原因 + 知乎指针归 doc L21，两处不混。

缩写先行：VE = vision encoder（视觉编码器）；EPLB = expert parallelism load balancer（专家并行负载均衡器，7.4 详展）；PE / CE = prefill / completion engine（§3 精确定义）；StateBlock 是引擎侧 SWA 状态落盘结构（7.2 详展）。

完整 serving 系统长什么样？doc L39 给出架构图（architecture_overview.png），组件口径散见 L33（四个 LB + KVCache Storage）、L35（Recipe）、L43（"A disaggregated E/P/D architecture plus dedicated storage nodes"）。按 doc 披露重绘如下，虚线框出 release 外的编排层：

```
            完整 serving 系统（doc L39 架构图重绘，官方披露）
 请求 ─► Recipe（协议转换/tokenize，另开源 deepseek-recipe）
              │
              ▼
        ┌────────────┐
        │ Entrypoint │ CHECK 角色 hash-chain 前缀匹配 → 命中长度 → 路由
        └─┬───┬───┬──┘ in-memory directory（跨机 SWA 定位）
   ┌──────┘   │   └──────┐        ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
   ▼          ▼          ▼        ┆ release 外：跨实例编排 ┆
 VE LB    Prefill LB  Decode LB   ┆ Entrypoint / 三 DP LB / ┆
   │          │          │        ┆ kvcache-store / 弹性层  ┆
   ▼          ▼          ▼        ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
 ┌────┐   ┌──────┐    ┌──────┐
 │VE×m│   │PE ×n │    │CE ×n │  ← release 内：单引擎实例
 │1GPU│   │EP2-8 │    │EP64- │    engine_type PROMPT/COMPLETION
 └────┘   └──┬───┘    └──┬───┘    StateBlock/EPLB/DSpark/SWA replay
             │ KV 传输    │
             ▼           ▼
        ┌──────────────────────┐
        │ 3FS kvcache-store    │ 存储编排不在 release
        │ （专用存储集群）       │ （3FS 文件系统本身另开源）
        └──────────────────────┘
```

缺口可以写成一条可验证的命题。先定义：所谓引擎实例（engine instance），是指一个独立部署的 DeepInfer 进程组——一个 PE 或 CE 及其内部的 EP/DP 通信域（doc L47-49：prefill EP2-8 / DP2-8，decode EP64-128 / DP64-128）；所谓 release 边界，是指 README L38-42 三项不含所对应的组件集合。

**命题 1**（release 实例边界）：README L38-42 三项不含对应的缺口组件，恰好是 doc 架构中决策范围跨引擎实例的编排层（引擎间路由与负载均衡、跨机存储编排、集群弹性），repo 无实现、仅 doc 披露；决策范围在单个引擎实例内部（含实例内 EP/DP 通信域）的运行时组件，release repo 均有实现。协议层 Recipe 与 kernel 栈、底层文件系统另开源，既非缺口也非「仅 doc」（见本节提点三分）。

**证明**（路线：按决策范围给组件分类，再逐件对 repo 验存在性——关键一步是认出边界是「实例」而非「进程」或「单卡」）。正例（release 内）：C++ Scheduler 定单引擎步组成（csrc/engine/scheduler.hpp）；StateBlock 引擎侧落盘（worker/cache_engine.py L221）；PE/CE 角色钩子（engine/engine_type.py + llm_engine.py init_pd_split L801 / pe_prepare_save_kvcache L818 / pe_do_save_kvcache L821）；EPLB（model_executor/eplb/ ExpertLoadBalancer L42）；DSpark / SWA replay / SO / batch-invariant 全在引擎侧。反例（release 外）：Entrypoint CHECK 匹配与路由（doc L191，跨机 metadata directory）；Prefill LB（doc L137-143，跨 DP 组 pull）；Decode LB（doc L147 / L161，全局视图 dispatch）；VE LB（doc L151，machine 粒度路由）；kvcache-store 编排（doc L165-169 三层表）；autoscaling / grace / EP failover（doc L249-265，集群平台层）。边界案例：EPLB 跨 rank（EP 组内 all_gather 统计 + 权重 shuffle）但不跨实例——EP 组就是实例本身，所以在 release 内；若把边界收紧到「单进程」，EPLB 与 DP attention 都会被错误划出。反向边界：PE/CE 的**角色功能**在 release、**角色间编排**不在——README L41 "most required engine functionality is already implemented" 正是此口径。代码级验证：全 repo grep compute_quota / quadratic 为空（2026-11-23 核实）——prefill LB 的二次成本模型（doc L143）只有 doc 披露、无 repo 实现。证毕。

命题 1 的交付物是缺口映射表（本章关键推导；「在 release」列的代码锚均 2026-11-23 tarball 核实，「仅 doc」列只讲披露架构、不推测代码）：

| 完整系统组件 | doc 行号 | release 状态 | 代码锚 / 披露口径 | 走读章 |
| --- | --- | --- | --- | --- |
| Recipe 协议层 | L35 / L354-372 | 另开源（deepseek-recipe repo） | Rust 多协议 / 注入防护 / 分段并行 BPE | 7.4 |
| Entrypoint（CHECK 匹配 / 路由 / directory） | L191 / L223 | 不在 repo，仅 doc | hash-chain metadata 匹配；命中长度路由前必算、不可隐藏 | 8.4 |
| VE LB | L151 | 不在 repo，仅 doc | machine 随机 + fewest patches | 7.4 |
| Prefill LB | L137-143 | 不在 repo，仅 doc | 4 长度档 + compute-quota pull + 二次成本模型 | 7.4 |
| Decode LB | L147 / L161 | 不在 repo，仅 doc | seq count + total KV length 双维 dispatch | 7.4 |
| EPLB | L125-131 | 在 release | ExpertLoadBalancer L42 + policy.py + shuffle.py | 7.4 |
| VE 引擎侧 | L51 | 在 release | ve/ MultimodalProcessor L29 | 9.4 |
| PE/CE 引擎侧角色 | README L41 | 在 release | engine_type.py + pe_*_save_kvcache L818/821 | 本章 §4 |
| KV 引擎侧（block / state 管理） | L167 | 在 release | csrc/engine/block_manager.hpp + cache_engine.py StateBlock L221 | 7.2 |
| kvcache-store 编排 | L165-169 / L187 | 不在 repo，仅 doc（3FS 文件系统另开源） | 三层存储表 / GC LRU / TTL | 2.4 回链 / 8.4 |
| 弹性平台层 | L249-265 | 不在 repo，仅 doc | autoscaling / grace 1 min / quota 冗余 / EP failover | 8.5 |

README 三项与表行的对应：项一（KV cache system）= Entrypoint directory / fetch 编排 + kvcache-store 两行——引擎侧 block / state 管理在 release，缺的是跨机存储系统；项二（VE/P/D disaggregation）= 三个 LB 行 + 编排，引擎侧角色功能已实现；项三（DP LB）= Prefill LB + Decode LB 两行。缺口的代价有官方定量口径：在线负载均衡达到完全均衡离线 benchmark 的约 80%（doc L155 逐字 "roughly 80% of a perfectly balanced offline benchmark"，官方披露）——这约两成的吞吐差距就是编排层的生产价值（7.4 详展）。

> **提点**：「另开源」与「仅 doc」是两种不同的缺口。Recipe、kernel 栈六 repo、3FS 文件系统都有独立开源 repo，缺的只是 DeepInfer 内的集成编排；Entrypoint 与三个 DP LB 则连独立 repo 都没有，只有 doc 架构披露。读 release 时先分清这一层，才不会把「不在本 repo」误读成「闭源」。

## 3. Repo 结构导览：deepinfer/ 模块树与 C++ 热路径

Python 侧 `from deepinfer._C.engine import Scheduler`（llm_engine.py L19 逐字）——C++ 实现本体在哪？答案不在 `_C/`：`deepinfer/_C/` 只有 4 个 native extension 类型 stub（`__init__.pyi` / `cuda_utils.pyi` / `engine.pyi` / `graph_timer.pyi`，`__init__.pyi` 为三行独立 import：`from . import cuda_utils`、`from . import engine`、`from . import graph_timer`），C++ 源码在根 `csrc/`（33 文件，2026-11-23 核实）。

> **坑**：在 `_C/` 目录找 C++ 实现。触发条件：按「_C 是 native 模块所以源码在此」的直觉走读；现象：只见 .pyi 类型声明，找不到调度器本体。规避：类型 stub 锚 `deepinfer/_C/`，实现锚根 `csrc/`（pybind.cpp 绑定 + engine/ 调度器族）；调度器本体 = csrc/engine/scheduler.hpp（1329 行）。

模块树如下（file:line 锚均 main@a679bb86 核实；只列本章与 P7 走读相关模块）：

```
DeepInfer/
├── csrc/                        # C++ 热路径（33 文件）
│   ├── pybind.cpp               # pybind11 绑定 → deepinfer._C
│   └── engine/
│       ├── scheduler.hpp        # C++ Scheduler（EngineType L39 / ALL-ANY 投票 L299 /
│       │                        #   process_MTPInfo L1196 cumprod，7.2）
│       ├── mtp_varlen.hpp       # MTPVarlen L133（DSpark 总预算选择，7.2）
│       ├── flops_estimator.hpp  # FlopsEstimator L28（FP8 等效 FLOPs / MFU 口径）
│       ├── block_manager.hpp / sequence.hpp / zos.cpp/hpp / ...
│       └── input_metadata/prepare_tensor_input_v4.cpp  # KEEP_TOKENS 约束 L355
├── deepinfer/
│   ├── _C/                      # 4 个 .pyi stub（非 C++ 源码）
│   ├── entrypoints/llm.py       # LLM L23（offline only；generate L90 / _run_engine L168）
│   ├── engine/                  # arg_utils.py EngineArgs L18 / engine_type.py /
│   │                            #   llm_engine.py LLMEngine L130（step L860 / post_process L984）
│   ├── worker/                  # Worker L51 / ModelRunner L53 / cache_engine.py StateBlock L221（7.2）
│   │                            #   cuda_graph_runner / dflash_cuda_graph_runner / decode_perf_cache / zos_status
│   ├── model_executor/
│   │   ├── models/deepseek.py   # DeepseekModel L59 / DeepseekForCausalLM L251
│   │   ├── models/v4_modules/   # attn / moe / engram / index_attn / yoco / hc / compress（7.2 / 7.4）
│   │   ├── models/mtp/dflash.py # DFlash L82 + ConfidenceHead L46（DSpark 引擎实现，7.2）
│   │   ├── layers/              # Sampler L59 + tilelang_ops/（cuda/ + asc/ 双平台）
│   │   ├── eplb/                # ExpertLoadBalancer L42（7.4）
│   │   ├── parallel_utils/      # ParallelProcessGroup L51 / init_parallel_groups L196 / CPManager L9
│   │   └── input_metadata/      # InputMetadataV4 / YocoMetadataMixin L32
│   ├── distributed/shm_broadcast.py  # ShmRingBuffer L33 / MessageQueue L162（vLLM 同名 lineage）
│   ├── ve/                      # MultimodalProcessor L29（vision encoder 引擎侧）
│   ├── npu_ops/                 # deep_select.py topk L21（_torch_topk fallback L7）/ vit_attn.py
│   ├── platform.py              # PlatformEnum L14 / PlatformGPU L55 / PlatformNPU L86 / get_platform L223
│   ├── structure_output/        # XGrammarCompiler L14 / XGrammarUtils L23 / SOMatcherBuilder build L58
│   ├── hugging_face/            # config.py HF→engine 映射（§5）/ loading / weights / tokenizer
│   └── environ.py               # DEEPINFER_* 族（228 行，§5）
├── deep_cluster/                # topology/bind + hardware/{nvidia,ascend}（一句注）
├── vendor/                      # DeepEP / DeepEP-Ascend / DeepGEMM / DeepGEMM-Ascend / DeepSelect / FlashMLA submodule
└── examples/                    # benchmark.py（588 行）/ sweep/run_sweep.sh / yoco/ 等价验证
```

逐模块钉关键事实。**入口**：`LLM`（entrypoints/llm.py L23）是唯一 serving 入口，offline only——`entrypoints/` 目录只有 llm.py，没有 OpenAI 兼容 server（在线协议层 = Recipe + Entrypoint，另开源或 release 外，§2）。其 docstring 与 vLLM 的 `LLM` 类逐段同构（"An LLM for generating texts from given prompts and sampling parameters..."），是 README L95 「primarily on vLLM」的代码级证据。`generate`（L90）收 prompts 与 SamplingParams，`_run_engine`（L168）的循环是 `while True` 内 step/flush 三元 + break（语义：unfinished 时 step()、否则 flush_inflight_reqs() 收尾）——ZOS 在途步的尾部 flush（§4）。

**engine type**：engine_type.py 定义三种引擎角色——NORMAL = 0（prompt + completion 混合）、PROMPT = 1（只做 prefill，即生产口径的 PE，prefill engine）、COMPLETION = 2（只做 decode，即 CE，completion engine）；csrc/engine/scheduler.hpp L39-43 的 C++ enum 与之对齐（python docstring 逐字要求「请和 csrc 中的 enum class EngineType 保持一致」）。`default_zero_overhead_scheduling` property：NORMAL 与 COMPLETION 默认 True，PROMPT 默认 False——README L26 「Enabled by default」的代码级含义（§4）。COMPLETION 没有 prefill 调度通道，无法重算被抢占的请求，启动时 TORCH_CHECK 强制 `stop_preempted_request=true`（scheduler.hpp L101-104）——抢占策略与引擎角色耦合（4.2 的 recompute vs swap 账在此被角色收窄）。PE/CE 角色拆分就是 doc L31 VE/P/D 分离的引擎侧地基：角色在 release，编排不在（命题 1）。

**worker 与执行**：Worker（worker.py L51）的 `execute_model_step`（L89）是统一执行入口，四段语义（§4 走读）；ModelRunner（model_runner.py L53）分 `execute_main_model`（L308）与 `execute_sample`（L337）两阶段，另有 `profile_run`（L444）显存 profiling、`capture_model`（L515）/ `capture_dflash_model`（L532）CUDA Graph 捕获、`profile_decode_perf`（L562）启动时实测 (batch_size, forward_ms) 曲线喂给 MTPVarlen——这条曲线是 DSpark 变长 verify 的硬件吞吐模型来源（7.2 强制重推导的 cost 项）。cache_engine.py 的 StateBlock（L221）与 SWA state cache 是 7.2 的走读对象，本章只登记存在。

**IPC 与分布式**：distributed/shm_broadcast.py 的 ShmRingBuffer（L33）/ MessageQueue（L162）做 executor ↔ worker 共享内存广播——与 vLLM 的 `distributed/device_communicators/shm_broadcast.py` 同名组件同 lineage（5.1 §5），每步高频小消息走共享内存环不走网络。parallel_utils/ 的 ParallelProcessGroup（L51）/ init_parallel_groups（L196）管 EP/DP/CP 进程组，CPManager（cp.py L9）对应 README L12 的 balanced CP（CP 取舍的官方 rationale 归 8.1）。

**双平台**：platform.py 的 PlatformEnum（L14）/ Platform 抽象（L19）/ PlatformGPU（L55）/ PlatformNPU（L86）/ get_platform（L223，读 `DEEPINFER_ACCELERATOR`）把硬件差异收敛到 pin_host_memory、get_distributed_options 等接口；npu_ops/ 是 Ascend 算子补丁（deep_select.py topk L21，缺 native 实现时 _torch_topk fallback L7；vit_attn.py / patch_imports.py）；tilelang_ops/ 分 cuda/ 与 asc/ 双目录。一句注：DeepGEMM-Ascend / DeepEP-Ascend 另立 repo（同 API），FlashMLA / TileKernels / DeepSelect 的 Ascend 支持在主 repo 内（7.3 分列）。

**结构化输出**：structure_output/ 的 XGrammarCompiler（compiler.py L14）编译 grammar，mask.py 的 allocate_vocab_mask（L10）/ XGrammarUtils（L23）产 token mask，SOMatcherBuilder（so_matcher_builder.py build L58）支持 replay_tokens 恢复 matcher 状态——抢占恢复后 grammar 状态不丢。SO 的生产形态（专用机异步编译、请求入引擎前取编译结果）是 doc L344-348 披露，归 9.4；引擎侧 SO 在 release。

范围外一句注（2026-11-23 核实，不展开）：deep_cluster/（硬件拓扑绑定）与 vendor/（六个 kernel submodule）为大纲外新增目录。**benchmark 方法学**：examples/benchmark.py（588 行）默认口径 --decode_num_requests 64 / --context_length 32768 / --prefill_batch_size 32768，_trace.py 的 dump_trace 产 kernel usage trace（doc L281 一手，7.3 详展）；sweep/run_sweep.sh 复现 README L83 的 Pareto 图（4 SM100 prefill / 64 decode，各单 scale-up domain）。MFU 口径值得单独钉：FlopsEstimator（csrc/engine/flops_estimator.hpp L28）按 FP8 等效 FLOPs 计（flops_coef FP4=1 / FP8=2 / BF16=4 / TF32=8），注释逐字排除 draft / padding / replicas——MFU 度量的是 target 模型吞吐，DFlash 加速用的额外计算不计入（L22-27）。读 DeepInfer 的 MFU 数字时先对这个口径，否则与含 draft 的口径混比。

## 4. step() 循环：C++ 调度器、步型投票与 ZOS

每一步的组成由谁决定？DeepInfer 的答案与 vLLM / SGLang 都不同：C++ Scheduler。`_create_scheduler`（llm_engine.py L307）构造 native Scheduler，传入 block_manager / state_manager / FlopsEstimator / DP 进程组（ZOS 开启时用 cpu_data_parallel_group）——调度热路径整体在 csrc/engine/scheduler.hpp（1329 行），Python 侧只做簿记与发射。

`step()`（L860）逐段如下（机制摘录，教学用省略错误处理；行号均 2026-11-23 核实）：

```python
# deepinfer/engine/llm_engine.py L860-959（摘录）
def step(self):
    seq_metadata_list, scheduler_outputs = self.scheduler.schedule(...)  # ① C++ 调度（兼释放 finished 组）
    # ② clone/zero states+blocks（L888 注：开启 yoco 截断时允许抢占导致的结果变化）
    self.post_schedule(scheduler_outputs)                                 # ③ 调度后处理
    # ④ SO 簿记：last prompt chunk 的 so_status 迁移（L910-928）
    if not scheduler_outputs.prompt_run and self.copy_state is not None:
        self.do_copy_state(to_copy_state_seqs)                            # ⑤ decode KV 落盘
    self.multi_modal_processor.fetch()                                    # ⑥ 多模态 fetch
    request_outputs = self._execute_model_async_step(...)                 # ⑦ 非阻塞发射 worker
    return request_outputs                                                # ZOS decode：返回上一步 outputs
```

七段里三个语义要点。其一，⑤ 的 copy_state 是 decode 步的 KV 落盘钩子（should_copy_state L811 / do_copy_state L787）：decode 不做 KV offload、不用 Hi-Sparse（decode 不受 cache 容量约束），但产出的 KV 仍写盘提命中率（doc L183，官方披露；与 2.4 的 offload 叙述区分口径）。其二，⑦ 的 `_execute_model_async_step`（L961）经 `_run_workers("execute_model_step")` 非阻塞发射，ZOS 语义在 worker 侧——Worker.execute_model_step 的 docstring（worker.py L96-102）逐字四段：forward 当前步 → post-process 上一步 → sample 当前步 → post-process 当前步（prompt / 非 ZOS 当场做；ZOS decode 步推迟到下一步、与 forward 重叠）。所以 ZOS decode 时 step() 返回的是**上一步**的输出，尾部在途请求由 `flush_inflight_reqs`（L977） drain——`_run_engine` 的 while True + step/flush 三元 + break（llm.py L176-186）正是为此。其三，输出生命周期：step docstring（L867-870）明示 RequestOutput 由 RequestOutputPool 管理、下一步 step() 自动释放，要保留必须 clone（DeferredRelease L67 收口）——_run_engine L181 逐字 clone。

ZOS（zero-overhead scheduling，零开销调度）是指 CPU 调度与 GPU 执行重叠、GPU 全程无 idle 的调度形态。doc L330 先给同步基线的问题：schedule(N) → forward(N) → sample(N) → post-process(N) 串行时，GPU 在 post-process(N) 与 schedule(N+1) 期间空转；L334 给重叠方案（官方披露，逐字 "Following existing implementations (SGLang, NanoFlow)"）：

```
        ZOS 重叠时序（doc L334 口径）
 GPU:  ── forward(N) ── sample(N) ──┬── forward(N+1) ── sample(N+1) ──
 CPU:  schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)
       └──────────── 全部藏进 GPU forward(N)/sample(N) 时间内 ────────────┘
```

lineage 回链（6.2 已交付，不重推）：SGLang 的 event_loop_overlap + FutureMap 是同一问题的 Python 侧解（6.2 的 bubble 消除账与 future token 正确性证明），其 idea 自认源自 NanoFlow [Zhu et al., 2024] [A]（arXiv:2408.12757，v0.4 blog 逐字）；DeepInfer 在此之上把调度热路径本身搬进 C++——overlap 藏残余 CPU 工作，C++ 压低 CPU 工作本身，两手法叠加。引擎侧的测量闭环在 `post_process`（L984）：prompt 步先 pe_do_save_kvcache，再 _process_model_outputs，最后 `update_mtp_runtime`（L995-1002）用 forward / sample 的 CUDA event elapsed_time 喂 MTPVarlen——DSpark 的硬件吞吐曲线由此在线更新（7.2）。

NORMAL 引擎的步型决策值得单独走读：DP 组内每步必须同型（全组 prefill 或全组 decode），靠投票统一（scheduler.hpp L299-314）。两类尝试点：ALL 点每 `PREFILL_ATTEMPT_INTERVAL` = 70 个 decode step 一次，全员都有 prefill 活才做，防一个 rank 干活其余陪跑；ANY 点每 `PREFILL_ANY_ATTEMPT_INTERVAL` = 100 步一次，任一 rank 有活就一起做，给孤立 rank 兜底（environ.py L181 / L184 默认值）；投票前还有 GPU_CACHE_PREFILL_WATERMARK 水位门。机制原因：attention DP 下各 rank 的步形状必须对齐，否则 all-reduce / all-to-all 同步点上先完成的 rank 空等——这是 #DP = #EP 耦合（doc L33）在调度器的落地（8.1 详展）。PROMPT 引擎则断言每步必 prompt_run（llm_engine.py L947-949，PE 不 decode）；COMPLETION 禁抢占重算（§3）。

## 5. config 与 env：DEEPINFER_* 族与 HF→engine 映射

environ.py 用 228 行定义 59 个 DEEPINFER_* 环境变量（2026-11-23 核实）——这是引擎的全部配置面。按功能分组，代表变量与默认值：

| 组 | 代表变量（默认） | 作用 |
| --- | --- | --- |
| Platform | ACCELERATOR（nvidia） | nvidia / ascend 双平台开关（README L61） |
| Distributed | NCCL_NUM_SMS（32）/ GLOO_TIMEOUT（14400 s） | 通信 SM 预算 / 进程组超时 |
| MoE 与 EP | EXPERT_COUNT_WINDOW_SIZE（100）/ MOE_LB_STAT_INTERVAL（128） | EPLB 统计窗 / moe_lb 上报间隔（7.4） |
| MTP | USE_NEXTN_DECODE（5）/ ENABLE_MTP_VARLEN / MAX_BACKTRACKING_TOKENS / DFLASH_CONFIDENCE_HEAD_TEMPERATURE | 投机上限 / 变长 verify / 回溯长度 / 置信温度（7.2） |
| KV Cache Store | NUM_STATE_SLOTS_RATIO（1.2）/ PREFILL_STORE_EXTRA_SIZE（128） | StateBlock slot 数比例 / 落盘 ring 余量（7.2） |
| Scheduling | PREFILL_ATTEMPT_INTERVAL（70）/ PREFILL_ANY_ATTEMPT_INTERVAL（100）/ ZERO_OVERHEAD_SCHEDULING | ALL / ANY 投票点 / ZOS override（§4） |
| YoCo | YOCO_PREFILL_TRUNCATION_MODE（off，可选 mask / compact）/ YOCO_PREFILL_KEEP_TOKENS | SWA replay 截断模式与保留后缀（7.2） |
| SO | XGRAMMAR_CACHE_LIMIT_BYTES（10 GiB）/ SO_BUILDER_CPUS（4） | grammar 缓存上限 / 编译 CPU 数 |
| CUDA Graph | ENFORCE_EAGER / CUDAGRAPH_BATCH_SIZE_INTERVAL（8）/ DECODE_PERF_CACHE_TTL_SEC（3600） | 禁图 / 捕获间隔 / decode 曲线缓存 TTL |
| Profiling 与多模态 | BATCH_INVARIANT / MULTIMODAL_POOL_SIZE_GB（8） | 批不变（NVIDIA only）/ VE 内存池 |

> **坑**：把 `DEEPINFER_USE_NEXTN_DECODE` 当 bool 开关。触发条件：设 1 以为「开启投机」；现象：它是 int 上限——environ.py L136-137 中文注逐字「最大支持投机几个；默认 5，显式设为 0 可关闭投机解码」，README L24 同口径 "controls the maximum draft length (default: 5)"。设 1 的含义是至多 draft 1 个 token，不是开启。规避：开关用 ENABLE_MTP_VARLEN（变长 verify）与 nextn=0（关投机）两个独立旋钮。

MTP 组里有一条代码级约束值得写成公式。YoCo 截断开启时，保留后缀必须覆盖 SWA 窗加最大回溯：

$$
K_{\mathrm{keep}} \ge W_{\mathrm{swa}} + \delta = 128 + 6 = 134 \qquad (1)
$$

**符号**： $K_{\mathrm{keep}}$ = DEEPINFER_YOCO_PREFILL_KEEP_TOKENS（保留末尾 query token 数）； $W_{\mathrm{swa}}$ = sliding_window（config [W] = 128）； $\delta$ = DEEPINFER_MAX_BACKTRACKING_TOKENS = max(v, nextn + 1)，nextn = USE_NEXTN_DECODE = 5 时为 6（environ.py L139；最大回溯深度，与 7.2 同符号）。约束双重 enforce：csrc prepare_tensor_input_v4.cpp L355-357 与 worker/model_runner.py L108-112。代入 README L25 默认值： $K_{\mathrm{keep}}$ = 140 ≥ 134，余量 6 token——140 是配置保留后缀，覆盖 134 有余量，与 doc 的 134 不矛盾（ring +6 与回溯 6 的两义、replay 机制本体归 7.2）。若把 nextn 调到 7，则 $\delta$ = 8、下界 136，140 仍合法。YOCO 是模型侧架构名（[Sun et al., 2024](../references.md)，arXiv:2405.05254，[A]），env 族沿用其名；形式化归 models 维度（V4.1 待补），7.2 只走读引擎侧 replay。

**V4.1-Flash serving 相关 config recap**（[W]，hf-mirror 2026-11-23 第四次重取全字段一致；huggingface.co 本机不可达、如实留痕）。一句 recap：V4.1-Flash 是 40 层 backbone + 3 层 nextn 的混合注意力 MoE——384 routed + 1 shared expert top-6 路由，MLA 变体单 KV head，叠加滑窗 / 压缩 / indexer 稀疏路径与 Engram n-gram 记忆；模型侧形式化（CED / YOCO 混合注意力、Engram、DSpark draft 结构、compress_ratios、hyper-connections）归 models 维度（V4.1 待补），本章只列 serving 决策相关字段：

| 字段（config.json） | 值 | serving 含义（一句） |
| --- | --- | --- |
| num_hidden_layers / num_nextn_predict_layers | 40 / 3 | backbone 40 层；nextn 3 层作 DSpark draft（7.2） |
| n_routed_experts / n_shared_experts / num_experts_per_tok | 384 / 1 / 6 | EP 划分与 EPLB 冗余基数（7.4：EP64 → 7 experts/GPU） |
| num_key_value_heads / head_dim / qk_rope_head_dim | 1 / 512 / 64 | MLA 变体单 KV head； $\kappa = 512 \cdot \mathrm{bytes}$（head_dim 512 含 rope 64，引擎 kv_lora_rank = 448，读法见 7.2 §9）；TP 会复制 KV → attention DP（8.1） |
| q_lora_rank / o_lora_rank / o_groups | 1280 / 1024 / 8 | MLA 变体低秩几何（模型侧归 models） |
| sliding_window | 128 | SWA 窗 = replay 恒定成本的窗宽（7.2） |
| compress_ratios | 43 项（0,0 + 2×18 + 1×20 + 0,0,0） | 逐层压缩比；引擎读法 _n_prefill_layer = 20（式 (2)） |
| kv_source_layer_ids | [2,8,14,20] | block KV 源层；_source_map 最近源层映射（下） |
| index_topk / index_n_heads / index_head_dim | 512 / 32 / 128 | indexer 稀疏路径（3.1 DeepSelect 回链） |
| engram_layer_ids / engram_vocab_size | [1,14] / 16000000 | Engram n-gram 记忆：ENP vocab 分片 + host RDMA（7.4） |
| dspark_block_size / dspark_target_layer_ids / dspark_markov_rank / dspark_n_routed_experts / dspark_num_experts_per_tok | 5 / [37,38,39] / 256 / 128 / 3 | DSpark draft 几何（7.2；引擎名 dflash，见坑注） |
| hc_mult / hc_sinkhorn_iters | 4 / 20 | hyper-connections 残差流宽度倍数（模型侧背景一句，下） |
| quantization_config | fp8 / weight_block_size [32,32] / scale_fmt ue8m0 / expert_dtype fp4 | 权重量化格式（3.2 回链；routed expert MXFP4） |
| max_position_embeddings / rope_scaling | 1048576 / YaRN ×16（from 65536） | 1M 上下文口径（1.2 长上下文账） |
| vision_config.patch_size | 14 | VE patch 账 (w/14)(h/14)（7.4 VE LB / 9.4） |

hc_* 字段族的模型侧背景一句：hyper-connections 是残差流宽度扩展技术（[Zhu et al., 2024](../references.md)，arXiv:2409.19606，[B]），形式化归 models（V4.1 待补）；引擎侧落点 v4_modules/hc.py / hc_blocks.py / mhc_ops.py + TileKernels 的 mHC Sinkhorn kernel（7.3）。

**HF→engine 映射**（hugging_face/config.py，2026-11-23 核实）把上表字段翻译成引擎配置，五处读法是走读关键：

- `_assert_supported`（L51）：硬门——hidden_act == silu / attention_bias == False / attention_dropout == 0 / num_key_value_heads == 1 / dtype == bfloat16，任一不符 raise ValueError。这就是「V4.1-Flash only」的代码级含义：引擎不接受第二个模型几何。
- `_source_map`（L62）：docstring 逐字 "Map each layer to the nearest source layer that is not later than that layer"——每层映射到不大于它的最近源层；kv_source_layer_ids [2,8,14,20] 经此成为 layer_kv_map（block KV 源层映射的引擎读法；doc 与 config 的表述张力不做模型侧调和，7.2 口径）。
- `_build_attn_module_list`（L146）：断言 len(compress_ratios) == n_layer + nextn，ratio == 0 建 window spec、ratio > 0 建 compress spec，lm head 占位 None，nextn 层建 DFlash spec。
- `_n_prefill_layer`（L158）：取首个 ratio == 1 的下标；`_build_dflash_cfg`（L177）：把 dspark_* 字段族映射为 cfg.dflash_cfg（sliding_window / block_size / target_layer_ids / markov_rank / noise_token / 小 MoE top-3）。
- `kv_lora_rank` 导出（L288-289）：注释逐字 "HF's shared KV vector includes RoPE; the internal rank excludes it."——kv_lora_rank = head_dim − qk_rope_head_dim = 512 − 64 = 448。config **没有** kv_lora_rank 字段，由引擎从 head_dim 导出；加回 rope 64 共 512 元素（V41_FORMAT.quantized_elements = 512，cache_common.py L109）。

> **坑**：把 V3 几何套 V4.1——head_dim 512 当 kv_lora_rank 再加 rope 64 得 576 元素。触发条件：按 V3「latent 512 + rope 64」式算 V4.1 的 κ；现象：重复计 rope——V4.1 的 head_dim 512 **已含** rope，真值 512 元素（448 latent + 64 rope），bf16 κ = 1024 B、MXFP8 528 B、per-16 FP4 288 B。规避：以引擎读法为一手权威——config.py L289（kv_lora_rank = 448）+ V41_FORMAT 512 元素 + FlashMLA 528/288 格式 + doc L223 官方 528 B/token 四方互证（7.2 §9 / 7.3 §4）；576 是 V3 系几何（旧 FlashMLA README head_dim_k 576），不是 V4.1 的合法替代口径。

两条读法写成公式：

$$
\mathrm{len}(\mathrm{ratios}) = L + L_{\mathrm{nextn}} = 40 + 3 = 43, \qquad n_{\mathrm{prefill}} = \min\lbrace i : \mathrm{ratio}_i = 1 \rbrace = 20 \qquad (2)
$$

**符号**： $L$ = num_hidden_layers = 40； $L_{\mathrm{nextn}}$ = num_nextn_predict_layers = 3； $`\mathrm{ratio}_i`$ = compress_ratios 第 i 项； $n_{\mathrm{prefill}}$ = _n_prefill_layer 返回值。代入全字段：compress_ratios = [0, 0, 2×18, 1×20, 0, 0, 0]——下标 0-1 为 0（窗层）、2-19 为 2（18 个压缩层）、20-39 为 1（20 层）、40-42 为 0（3 个 nextn）；首个 ratio == 1 在下标 20，故 backbone 前 20 层为 prefill 侧（encoder）、后 20 层为 YOCO consumer（cache_engine.py L113 以 layer ≥ n_prefill_layer 判 consumer 层）。43 = 40 + 3 的断言与 20 的读法都可用 python 一行复核（练习 1）。

> **坑**：把 dflash 当第二种投机方法。触发条件：同时看到 config 的 dspark_* 与代码的 dflash_cfg / DFlash 类；现象：误读为两套方法，或按 DSpark 名找不到实现。规避：DSpark 是模型 / 技术名（config dspark_* 字段族），DFlash 是引擎实现名——_build_dflash_cfg L177 做 dspark_* → dflash_cfg 映射，draft 模型类 = models/mtp/dflash.py DFlash L82、ConfidenceHead L46，CUDA graph runner = worker/dflash_cuda_graph_runner.py，env = DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE。vLLM 侧 spec_decode 目录同时有 dspark/ 与 dflash/（W3 注），两引擎命名各自独立。7.2 按 DFlash 类名走读。

## 6. 与 vLLM / SGLang 架构对照

三个引擎都有 continuous batching、prefix caching、投机解码与结构化输出，DeepInfer 的架构选择差异在哪？不在机制有无，而在专用化深度与开源完整度。对照表如下（vLLM / SGLang 侧事实 recap 5.1 / 5.2 / 6.1 / 6.2 已交付内容，不重推）：

| 轴 | DeepInfer（本章） | vLLM V1（5.1 / 5.2） | SGLang（6.1 / 6.2） |
| --- | --- | --- | --- |
| 调度热路径 | C++ Scheduler（scheduler.hpp 1329 行）+ ZOS 重叠 | Python Scheduler + AsyncScheduler，EngineCore busy loop | Python event_loop_overlap + FutureMap |
| 步型决策 | NORMAL ALL/ANY 投票（DP 组同型）+ PE/CE 角色拆分 | 统一 token 预算，无 prefill/decode 阶段区分 | ForwardMode EXTEND/DECODE/MIXED + PrefillDelayer |
| KV 引擎侧 | StateBlock（SWA state）+ paged block | BlockPool + block hash 链 | radix tree + 两级内存池 |
| KV 系统完整度 | 三层存储仅 doc 披露，kvcache-store 编排不在 release（命题 1） | kv_offload/ + kv_connector（nixl / mooncake / lmcache / hf3fs）开源 | HiCache 三层 + disaggregation KV connector 开源 |
| IPC | shm MessageQueue（vLLM 同名 lineage） | ZMQ+msgpack（frontend↔core）+ shm 广播（executor↔worker） | ZMQ 三进程链 |
| 投机解码 | DSpark 变长 verify（dflash，7.2） | eagle / ngram / mtp speculator 继承链 | eagle_worker_v2 / dspark_components / ngram |
| 结构化输出 | XGrammarCompiler + SOMatcherBuilder（replay_tokens 恢复） | xgrammar / guidance / outlines 多 backend | XGrammar 默认 + llguidance / outlines |
| 确定性 | BATCH_INVARIANT（NVIDIA only）+ deterministic sampling 三件套 | determinism/batch_invariant.py | batch_invariant_ops/ |
| 硬件面 | SM100 + Ascend 950 双平台（platform.py） | NVIDIA 为主的多 backend | NVIDIA 为主的多 backend |
| 模型面 | V4.1-Flash only（_assert_supported 硬门） | 多模型通用 | 多模型通用 |
| serving 面 | offline release，无 API server | offline LLM + online AsyncLLM / OpenAI server | offline Engine + online HTTP server |
| lineage | README L95：vLLM 主 + SGLang / TRT-LLM 启发；doc L334：async 仿 SGLang / NanoFlow | V1 重构自认受 SGLang / TRT-LLM 等影响（v1 blog） | overlap idea 源自 NanoFlow（v0.4 blog 逐字） |

读表三个论断。其一，调度轴的差异是**CPU 开销税的三种解法**（5.1 式 (1) recap）：vLLM 把 EngineCore 隔离成纯 busy loop、frontend 外置重叠；SGLang 调度与执行同进程、用 overlap + FutureMap 藏 CPU 时间；DeepInfer 把调度热路径搬进 C++ 再叠加 ZOS——单模型几何让调度器可以专用化到 C++，不必背通用模型的 Python 抽象层。其二，KV 轴的差异是**开源完整度**：vLLM / SGLang 的 KV 系统（offload / connector / 分层）全开源，DeepInfer 引擎侧 StateBlock + paged block 在 release、系统侧三层存储编排不在（命题 1）——这是三引擎开源形态的最大缺口，也是「生产引擎开源版」与「通用开源引擎」的本质区别。其三，专用化换深度：_assert_supported 硬门 + 双平台 + offline only，用模型 / 硬件 / serving 面的广度换单模型生产路径的深度（DSpark / SWA replay / EPLB / batch-invariant 全栈协同）。选型论断归 9.2，三引擎 PD 与分布式对照归 8.4，本章只立 P7 侧基线。

## 总结

| 论断 | 回指 |
| --- | --- |
| DeepInfer = DeepSeek 生产引擎开源版：155T tokens/day、38 USD/GPU-hour 是收入非成本（官方披露，doc L17）；offline release / V4.1-Flash only / SM100 + Ascend 950 / Apache-2.0 / vLLM 主 lineage（README L3 / L7 / L35-36 / L91 / L95） | §1 |
| release 边界三项（README L38-42）：KV cache 系统 / VE-P-D disagg 编排 / DP LB；耦合原因 + 知乎指针在 doc L21 不在 README | §2 坑注 |
| 命题 1（实例边界）：实例内组件全在 release（含跨 rank 的 EPLB），跨实例编排仅 doc；全 repo 无 compute_quota / quadratic 代码 | §2、练习 4 |
| 缺口映射 11 行：Entrypoint / 三 DP LB / kvcache-store / 弹性层仅 doc；Recipe / kernel 栈 / 3FS 文件系统另开源；LB 缺口代价 ~20%（doc L155 约 80% 口径） | §2 表 |
| _C/ = 4 个 .pyi stub，C++ 源码在根 csrc/（33 文件）；调度器本体 scheduler.hpp 1329 行 | §3 坑注 |
| LLM L23 offline only 无 API server；engine_type NORMAL / PROMPT(=PE) / COMPLETION(=CE)，ZOS 默认 NORMAL/CE 开、PE 关；CE 禁抢占重算 | §3 |
| step() 七段：C++ schedule → clone/zero → post_schedule → SO 簿记 → copy_state（decode KV 落盘，doc L183）→ mm fetch → 非阻塞发射；ZOS decode 返回上一步输出，flush_inflight_reqs 收尾 | §4 |
| ZOS = GPU forward(N)/sample(N) 与 CPU schedule(N+1)/post-process(N−1) 重叠（doc L334，仿 SGLang / NanoFlow，6.2 回链）；NORMAL 步型 ALL/ANY 投票（70/100 步） | §4 |
| KEEP_TOKENS ≥ 128 + 6 = 134，README 默认 140 余量 6（式 (1)，双重 enforce）；USE_NEXTN_DECODE 是 int 上限非 bool | 式 (1)、§5 坑注 |
| V4.1 config [W]：40L+3 nextn / 384+1 top-6 / MLA 变体 kv_heads 1 head_dim 512（含 rope；512 元素 = 448 latent + 64 rope，κ = 512·bytes）/ sliding_window 128；len(ratios)=43=40+3、_n_prefill_layer=20（式 (2)）；模型侧归 models（V4.1 待补） | §5 表、坑注 |
| DSpark（config dspark_*）↔ DFlash（引擎实现 dflash_cfg / DFlash L82 / ConfidenceHead L46）；_assert_supported 硬门 = V4.1-Flash only 的代码含义 | §5 坑注 |
| 对照三论断：调度 = CPU 税三解法（C++ 热路径 + ZOS）；KV = 开源完整度最大缺口；专用化换深度 | §6 |

## 练习

1.（计算 · [10]）config 读法（式 (2) 口径）：给定 compress_ratios = [0, 0, 2×18, 1×20, 0, 0, 0]（43 项）。(a) _build_attn_module_list 断言的右端 L + nextn 等于多少？(b) _n_prefill_layer 返回什么？(c) encoder 与 YOCO consumer 各几层？先写出预测再用 python 一行 `next(i for i, r in enumerate(ratios) if r == 1)` 验证。校验：config.json 的 num_hidden_layers = 40 / num_nextn_predict_layers = 3 字段；config.py L151 断言与 L161 定义；cache_engine.py L113 的 layer ≥ n_prefill_layer 判据。
2.（计算 · [20]）env 约束账（式 (1) 口径）：(a) nextn = 5 时 $\delta$ 与 KEEP_TOKENS 下界各是多少？(b) README 默认 140 的余量是多少 token？(c) 若 nextn 调到 7，140 是否仍合法？(d) 若显式设 DEEPINFER_MAX_BACKTRACKING_TOKENS = 10（nextn 仍 5），下界变成多少、140 是否仍合法？校验：environ.py L139 的 max(v, nextn + 1) 与 L198-201 约束注；(c)(d) 的合法性以 $128 + \delta \le 140$ 判定。
3.（代码 · [30]）step() 与 ZOS 走读（main@a679bb86）：先写出预测再对照源码——(a) ZOS decode 步里 step() 返回的是当前步还是上一步的 outputs？worker.py 哪几行 docstring 钉死这个语义？(b) PROMPT 引擎为什么在 llm_engine.py L947-949 断言 prompt_run？(c) flush_inflight_reqs（L977）做什么、为什么 _run_engine 的循环需要它？(d) COMPLETION 为什么必须配 stop_preempted_request=true？校验：worker.py L96-102 四段 docstring；llm_engine.py L947-949 / L977-982；scheduler.hpp L101-104 TORCH_CHECK；llm.py L176-186 while True + step/flush 三元 + break。
4.（论证 · [30]）补全命题 1 的验证：(a) 预测全 repo grep compute_quota / quadratic 的结果，并说明它证明哪个组件不在 release；(b) 把 EPLB / StateBlock / PE-CE 钩子 / Entrypoint / Decode LB 五件按决策范围（实例内 / 跨实例）分类，逐件给 repo 锚或 doc 行号；(c) 边界案例：EPLB 跨 rank 为什么仍在 release 内？若把命题边界收紧到「单进程」，哪两件会被错误划出？校验：README L38-42 三项清单 + doc L21 耦合原因；doc L137-161 / L191 / L249-265 披露行号；grep 为空（2026-11-23 核实）；EPLB 锚 model_executor/eplb/__init__.py L42、StateBlock 锚 worker/cache_engine.py L221。

## 参考

- [DeepInfer repo](../references.md)：deepseek-ai/DeepInfer，教学参照 + 官方实现（key: deepinfer-repo）——README L3 / L7 / L11-18 / L23-29 / L35-42 / L48-52 / L61 / L67-69 / L79-83 / L91 / L95 一手口径 + deepinfer/ 模块树与 csrc/ file:line 锚（main@a679bb86，2026-11-23 fresh tarball 核实；与 W1 同日 tarball byte-identical）
- [DeepInfer doc](../references.md)：doc/inference_deployment_qa.md v0.2（372 行），教学参照 + 官方实现（key: deepinfer-doc，官方披露口径注：不升 [A]、不降 [C]，核实 2026-11-23）——L17 生产规模与收入口径 / L21 耦合原因 + 知乎指针 / L33 四 LB + KVCache Storage / L39 架构图 / L47-53 并行口径 / L137-161 三 LB / L165-169 三层存储表 / L183 decode 落盘 / L191 Entrypoint / L249-265 弹性 / L330-334 async scheduling
- [V4.1-Flash config](../references.md)：DeepSeek-V4.1-Flash-1017 config.json，[W]（key: v41-flash-config，hf-mirror 传输核实，huggingface.co 本机不可达）——§5 serving 字段表与引擎读法锚；模型侧形式化归 models 维度（V4.1 待补）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——对照基线（5.1 / 5.2 已走读；shm_broadcast 同名组件 lineage）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——对照基线（6.1 / 6.2 已走读；event_loop_overlap + FutureMap）
- [Zhu et al., 2024](../references.md)：NanoFlow: Towards Optimal Large Language Model Serving Throughput，arXiv:2408.12757，[A]（key: nanoflow）——async scheduling lineage（doc L334 自认仿 SGLang / NanoFlow；6.2 回链）
- [Sun et al., 2024](../references.md)：You Only Cache Once: Decoder-Decoder Architectures for Language Models，arXiv:2405.05254，[A]（key: yoco）——YOCO env 族命名的模型侧源头指针（7.2 详展，不推导）
- [Zhu et al., 2024](../references.md)：Hyper-Connections，arXiv:2409.19606，[B]（key: hyper-connections）——hc_* 字段模型侧背景一句（形式化归 models，V4.1 待补）
- [3FS](../references.md)：deepseek-ai/3FS，教学参照（key: 3fs）——kvcache-store 底层文件系统（另开源；存储编排层不在 release，命题 1）
- [deepseek-recipe](../references.md)：deepseek-ai/deepseek-recipe，教学参照（key: deepseek-recipe）——Recipe 协议层（另开源，7.4 详展）
- 跨维度指引（纯文字）：models 2.4（KV cache 容量）、models 维度 V4.1 谱系待补；inference 1.1（在线 / 离线分界）、1.2（MLA κ 口径）、1.4（采样与确定性）、2.1（块管理）、2.4（分层存储与 offload）、3.1（indexer / DeepSelect）、3.2（量化格式）、3.4（batch-invariant / TileLang）、4.1（迭代级调度）、4.2（抢占策略）、5.1 / 5.2（vLLM 架构与调度 KV）、6.1 / 6.2（SGLang 架构与 overlap）、7.2（DSpark / SWA replay / StateBlock）、7.3（kernel 栈与 benchmark）、7.4（EPLB / 三 LB / Engram ENP / Recipe）、8.1（DP / CP 取舍）、8.4（PD 系统对照）、8.5（弹性容错）、9.2（引擎选型）、9.4（SO 生产形态 / VE）
