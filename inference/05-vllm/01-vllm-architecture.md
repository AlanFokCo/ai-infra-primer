# 5.1 vLLM V1 架构总览（vLLM V1 Architecture Overview）

> GPU 每步前向只要 ~5 ms，CPU 侧的调度、组批、detokenize 凭什么不把它拖成 10 ms？V1 把引擎拆成哪几类进程、一条请求怎样穿过它们、P1-P4 的公式各自落在哪个文件？本章走读 vLLM V1 的进程架构与一次请求的完整数据流，给出 diff 协议的控制面字节上界命题，并立起机制 ↔ 文件的映射总表 | 前置：inference 1.1（请求生命周期与 SLO）、1.2（prefill / decode 账）、2.1（块管理）、4.1（迭代级调度与每步循环）；models 2.4（KV cache 容量） | 锚点：vLLM / PagedAttention [Kwon et al., 2023] [A]（arXiv:2309.06180 + vllm-project/vllm）；vLLM V1 重构 blog [C]（开放技术博客、非闭源，2025-01-27，全文核实）；代码 `vllm/v1/{engine,executor,worker}/` + `vllm/entrypoints/`（2026-11-23 main tarball file:line 核实；调度器与 KV 管理器全量走读见 5.2，分布式与编译见 5.3，高级特性见 5.4）

## 1. 重构动机：一笔 CPU 开销账

Llama-8B 在 H100 上跑一步 decode，GPU 执行时间低至约 5 ms（[vLLM V1 blog](../references.md)，[C] 开放技术博客、非闭源，2025-01-27，原文「execution time on the GPU is as low as ~5ms」）。同一时刻，CPU 要干完一整套活：跑 API server、调度下一步、准备输入 tensor、detokenize 上一步输出、把 token 流式推给客户端。这些活与前向串行时，步时就是两段之和。设 CPU 段 $t_{\mathrm{cpu}}$、GPU 段 $t_{\mathrm{gpu}}$，同步循环里 GPU 真正在算的时间占比为

$$
\eta_{\mathrm{gpu}} = \frac{t_{\mathrm{gpu}}}{t_{\mathrm{cpu}} + t_{\mathrm{gpu}}} \qquad (1)
$$

**符号**： $\eta_{\mathrm{gpu}}$ = GPU 时间占比（口径：一步墙钟里 GPU 前向所占份额，不是 1.3 的排队利用率 $\rho$，也不是 4.1 的 slot 利用率 $\eta$）； $t_{\mathrm{cpu}}$ = 该步纯 CPU 工作（调度 + 组批 + 输出处理）； $t_{\mathrm{gpu}}$ = 前向 + 采样。代入 blog 的 $t_{\mathrm{gpu}}$ = 5 ms，教学取 $t_{\mathrm{cpu}}$ = 2 ms（教学假设，非 blog 披露值）： $\eta_{\mathrm{gpu}}$ = 5/7 ≈ 71%——近三成墙钟 GPU 在等 CPU。模型越小、卡越快， $t_{\mathrm{gpu}}$ 越短，这个占比越难看：CPU 段不随 GPU 变快而缩短，它是固定税。所谓 CPU 开销（CPU overhead），就是指这一步墙钟里花在 GPU 前向之外、由 CPU 串行承担的管理性工作（口径：含 API 处理、tokenize、调度、输入准备、detokenize、流式推送；不含 GPU kernel 本身）。

V1 的答案不是把 CPU 活算得更快，而是把它**移出关键路径**：让 CPU 段与 GPU 段重叠。式 (1) 的分母从「和」变成「max」的条件与完整 bubble 账归 6.2（SGLang overlap scheduler 的强制重推导），vLLM 侧的代码形态见 5.2 §8；本章只钉架构事实——V1 把哪些 CPU 活搬去了哪里。blog 给出的总收益：相同 kernel 下吞吐至多 1.7×（vs V0，without multi-step scheduling 口径；Llama-3.1-8B / Llama-3.3-70B ShareGPT + Qwen2-VL VisionArena 实测），增益主要来自架构而非 kernel。

blog 把 V0 的结构性问题归成三条，V1 逐条对治（[vLLM V1 blog](../references.md)，[C]，2025-01-27 全文核实）：

| V0 问题（blog 口径） | 代价 | V1 对治 | 代码落点 |
| --- | --- | --- | --- |
| scheduler 与 Worker 0 同进程（省广播 IPC） | 架构不对称、单卡/多卡行为分裂 | worker 侧缓存 request state、每步只传 diff，scheduler 与 Worker 0 分进程 | `SchedulerOutput` 的 new/cached 二分（§5） |
| 每步重建输入 tensor 与 metadata | CPU 组批开销随 batch 涨 | Persistent Batch：缓存 tensor 只应用 diff + Numpy 替代 Python 原生操作 | `InputBatch`（§5） |
| prefix caching 低命中时 CPU 开销大、默认关闭 | 高复用 workload 白白重算前缀 | 常数时间驱逐 + 最小化 Python 对象创建，0% 命中吞吐损失 <1%，默认开启 | `FreeKVCacheBlockQueue`（5.2 §5） |

第一条的机制谱系值得钉一句：multiprocessing API server（ZeroMQ IPC，让 API server 与 AsyncLLM 重叠）在 V0.6.0 已落地；V1 把同一手法推进到 AsyncLLM 核心，blog 逐字「creating an isolated EngineCore execution loop that focuses exclusively on the scheduler and model executor」——tokenize、多模态预处理、detokenize、流式推送全部留在 frontend，与核心循环重叠。这不是新发明，是把「CPU 活移出 GPU 关键路径」从 API 层贯彻到引擎层。

> **坑**：把 V1 blog 的 alpha 限制清单当现状。触发条件：照 2025-01-27 blog 写「V1 不支持 PP / spec decode / structured output / LoRA / Prometheus / logprobs」；现象：与当前 main 全面冲突——这些特性现在全有，且 **V0 已从代码库移除**（`vllm/engine/llm_engine.py` 只剩一行别名「LLMEngine is an alias of vllm.v1.engine.llm_engine.LLMEngine」，`envs.py` 无 `VLLM_USE_V1`，2026-11-23 tarball 核实）。规避：blog 只作重构动机与架构口径的来源，特性支持面一律以当前 main 为准；限制清单是 alpha 快照，引用必须带「历史」限定。

**本章数值口径**（章脊锁定，全章例题复用，与 1.2 / 2.1 / 4.1 一致）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16）， $\kappa L$ = 128 KiB/token；硬件 H100 SXM（bf16 dense 峰值 989.5 TFLOPS、HBM 3.35 TB/s，[NVIDIA datasheet](../references.md)，[C] 官方规格披露，完整表见 1.3）。blog 的 ~5 ms / 1.7× / <1% 是 blog benchmark 口径，逐处标注； $t_{\mathrm{cpu}}$ = 2 ms 是教学假设。进程数与 IPC 字节例用本章 §2 / §5 声明的部署口径。

## 2. 进程架构：四类进程与两个入口

一个 `vllm serve -tp=4` 的单节点部署，操作系统里到底跑着几个进程？答案是 6 个，不是 1 个。官方设计文档 `docs/design/arch_overview.md`（2026-11-23 tarball，一手）把 V1 进程分成四类（[vLLM repo](../references.md)，[A]）：

| 进程类型 | 数量 | 职责 | 代码 |
| --- | --- | --- | --- |
| API Server | $A$（默认 = DP 度） | HTTP（OpenAI 兼容）、tokenize、多模态加载、流式回传 | `entrypoints/openai/api_server.py` + `entrypoints/launchers/api_server` |
| Engine Core | 每 DP rank 1 个 | scheduler + KV 管理 + 协调 GPU worker，busy loop | `v1/engine/core.py` |
| GPU Worker | 每 GPU 1 个（= DP × PP × TP） | 载权重、跑前向、管 GPU 显存 | `v1/executor/multiproc_executor.py` + `v1/worker/gpu_worker.py` |
| DP Coordinator | DP > 1 时 1 个 | DP rank 间负载均衡 + MoE 同步步协调 | `v1/engine/coordinator.py` |

进程总数 = API server 数 + EngineCore 数 + GPU worker 数（DP > 1 再 +1）。两个官方算例（arch_overview.md Process Count Summary 表逐字）：`-tp=4` 单节点 = 1 API + 1 core + 4 worker = **6 进程**；`-tp=2 -dp=4`（8 GPU）= 4 API + 4 core + 8 worker + 1 coordinator = **17 进程**。读表的关键：API server 与 EngineCore 之间是**多对多** ZMQ 拓扑——每个 API server 连所有 EngineCore，任何 API server 可把请求路由到任何 core；DP 下 API server 数自动扩到 DP 度（也可 `--api-server-count` 手动设）。CPU 资源要按进程数配，不是按 GPU 数配。

```
                         V1 进程架构（tp=2, dp=4 示意，17 进程）
 ┌─────────────── API Server 进程 ×4 ───────────────┐
 │  HTTP / tokenize / mm 加载 / detokenize / stream  │
 └───────┬───────────────────────────────┬──────────┘
         │ ZMQ + msgpack（多对多）         │
 ┌───────▼────────┐  ┌────────────────┐  ┌▼───────────────┐
 │ EngineCore ×4   │  │ DP Coordinator │  │ （每 core 管    │
 │ scheduler+KV    │◄─┤ 负载均衡/MoE 同步│  │  2 个 worker）  │
 │ busy loop       │  └────────────────┘  └────────────────┘
 └───────┬────────┘
         │ 共享内存 MessageQueue 广播（rpc_broadcast_mq）
 ┌───────▼──────────────────────────────┐
 │ GPU Worker 进程 ×8（每 GPU 一个）       │
 │ Worker → GPUModelRunner → model       │
 └──────────────────────────────────────┘
```

两个入口对应两种 serving 形态（1.1 的在线/离线分界）：

- **离线（offline）**：`LLM` 类（`vllm/entrypoints/llm.py` L63，`generate` L431）。Python 进程内直接调 `llm.generate(prompts, sampling_params)`，不起 HTTP server。底层走 `LLMEngine`（`v1/engine/llm_engine.py` L54，`add_request` L231 / `step` L323），同步拉输出。
- **在线（online）**：`vllm serve` 起 OpenAI 兼容 server（`entrypoints/openai/api_server.py`，chat 路由已模块化到 `chat_completion/api_router.py` `create_chat_completion` L59 + `chat_completion/serving.py` L290）。底层走 `AsyncLLM`（`v1/engine/async_llm.py` L94，`add_request` L411 / `generate` L740 / `encode` L1053），asyncio 异步流式。

`LLMEngine` 与 `AsyncLLM` 是同一引擎核心的两种 frontend 包装，不是两套引擎。frontend 与 EngineCore 之间用哪种 client，由两个开关组合决定（`v1/engine/core_client.py` `EngineCoreClient.make_client` L111）：

| multiprocess_mode | asyncio_mode | client | 场景 |
| --- | --- | --- | --- |
| False | False | `InprocClient`（L392） | EngineCore 在同进程，直调（调试/离线） |
| True | False | `SyncMPClient`（L988） | 多进程同步（离线 `LLMEngine` 默认） |
| True | True | `AsyncMPClient`（L1184） | 多进程异步（在线 `AsyncLLM`） |
| True | True + DP | `DPAsyncMPClient`（L1541）/ `DPLBAsyncMPClient`（L1780） | DP 外部/内部负载均衡 |

`VLLM_ENABLE_V1_MULTIPROCESSING` 默认 True（`envs.py` L175，2026-11-23 核实），所以默认部署里 EngineCore 是独立进程——这正是 §1 「isolated EngineCore」的落地。asyncio 不带多进程（`InprocClient` + asyncio）当前直接 `NotImplementedError`（core_client.py L127）：异步 frontend 必须配独立 EngineCore 进程才有重叠可言。

进程数直接决定 CPU 配额：每个 API server 进程另用多线程做媒体加载（`VLLM_MEDIA_LOADING_THREAD_COUNT` 默认 8，arch_overview.md 口径），EngineCore 与 worker 各占一核做调度与执行。所以 17 进程的部署不是「8 张卡配 8 核」就够——frontend 的 tokenize / detokenize / 媒体加载与 EngineCore 的调度都是独立 CPU 消耗，配少了式 (1) 的 $t_{\mathrm{cpu}}$ 反而涨。CPU 资源 sizing 的完整建议归官方 configuration/optimization.md 与 9.1 容量规划，本章只钉「按进程数而非 GPU 数配 CPU」这条结构事实。

## 3. EngineCore：busy loop 与一步的五段

EngineCore 进程醒着的时候在干什么？一个不停转的 busy loop。`v1/engine/core.py`（3177 行）里有两个类要分清：`EngineCore`（L130）是引擎核心对象（持 scheduler + model_executor），`EngineCoreProc`（L1341）是它的进程包装（加输入/输出队列与 socket 线程）。busy loop 本体（`run_busy_loop` L1814，机制摘录，非全文）：

```python
# vllm/v1/engine/core.py L1814-1825（摘录）
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()      # L1840 收请求，直到有活要干
        self._maybe_publish_request_counts()
        self._process_engine_step()      # L1882 跑一步，输出进 output_queue
        self._maybe_publish_request_counts()
```

`_process_input_queue`（L1840）把 input_queue 里的客户端请求（add / abort / utility）逐条 `_handle_client_request` 消化，没活时阻塞等；`_process_engine_step`（L1882）调 `step_fn` 跑一步、把 `EngineCoreOutputs` 塞进 output_queue。输入与输出各有一条独立 socket 线程搬运：`process_input_sockets`（L2198）从 ZMQ 收请求 deserialize 进 input_queue，`process_output_sockets`（L2305）从 output_queue 取输出 serialize 发 ZMQ——busy loop 本身不碰网络 IO，这是「核心循环只做 scheduler + executor」的进程内体现。

一步的内部是五段（`step()` L714，机制摘录；4.1 §5 已给 schedule → execute → update 三段骨架，此处补全 grammar 与 sample 的分离）：

```python
# vllm/v1/engine/core.py L714-744（摘录，教学用省略错误处理）
def step(self):
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule(...)          # L724 ① 定组成
    future = self.model_executor.execute_model(              # L727 ② 前向（non_block）
        scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(     # L728 ③ grammar mask（CPU）
        scheduler_output)
    model_output = self._wait_for_output(future)             # 等前向
    if model_output is None:
        model_output = self.model_executor.sample_tokens(    # L732 ④ 采样
            grammar_output)
    engine_core_outputs = self.scheduler.update_from_output( # L738 ⑤ 更新
        scheduler_output, model_output)
    return engine_core_outputs, ...
```

五段的顺序是硬语义，关键在 ②③④ 的交错：`execute_model` 以 `non_block=True` 发射前向（返回 Future），CPU 不等 GPU，转头去算 ③ 的 grammar bitmask（结构化输出的 token 掩码，纯 CPU，归 5.4）；等前向回来再 ④ 采样。这样 grammar 的 CPU 时间藏进了 GPU 前向——式 (1) 的 $t_{\mathrm{cpu}}$ 被 overlap 削掉一块。`execute_model` 与 `sample_tokens` 在 worker 侧也是两个独立方法（`GPUModelRunner` L1865 / L2279，§5），这个分离就是 async scheduling 的接口：前向产 hidden states 暂存，采样何时做由调度侧定。

没活时 busy loop 不空转烧 CPU：`_process_input_queue`（L1840）在 `has_work()` 为假时阻塞在 `input_queue.get(block=...)`，先通知 idle 回调、清空 aborts 队列，再睡到下一条请求来（源码「EngineCore waiting for work」debug 日志）。有活但本步没跑前向（如请求在等远端 KV）时，`_process_engine_step`（L1897）`time.sleep(0.001)` 让出 GIL，给后台传输线程进度——这是 KV connector 异步加载（8.4）不被 busy loop 饿死的细节。每步收尾还有 `post_step`（L750）：非 async scheduling 时从 worker 取 draft token ids 更新回 scheduler（投机解码的 draft 回填，5.4）；async 时这步在 worker 侧做，core 不取。

async scheduling 开启时走变体 `step_with_batch_queue`（L759）：batch queue 里同时有多个在途步，调度优先填满队列而非等输出（docstring 逐字「fulfilling the batch queue has a higher priority than getting model outputs」）。语义不变、时序重叠，placeholder 机制与 CPU 开销账见 5.2 §8。DP 下另有 `DPEngineCoreProc`（L2592，`run_busy_loop` L2779），每 DP rank 一个独立 EngineCore + scheduler，跨 rank 协商归 5.3 / 8.1。

> **提点**：busy loop 与操作系统的内核调度循环同构——都是「收事件 → 决策 → 派发 → 收结果」的无限循环。差别在派发对象：OS 派时间片给进程，EngineCore 派 token 预算给请求；OS 的上下文是 KB 级寄存器，EngineCore 的上下文是 GB 级 KV cache，所以「换出」代价（4.2 抢占账）才是 serving 调度的核心难题。

## 4. frontend ↔ EngineCore：异步 IPC 与输出处理

请求从 HTTP 进来到变成 token id，穿过哪些对象？以在线路径为例，`AsyncLLM.add_request`（async_llm.py L411）先把 prompt 交给 `InputProcessor`（`input_processor.py` L41，692 行；旧名 processor.py 不存在）做 tokenize 与多模态处理，产出 `EngineCoreRequest`（`v1/engine/__init__.py` L175）；再经 client 发往 EngineCore。`AsyncMPClient.add_request_async`（core_client.py L1374）把请求 msgpack 序列化、ZMQ 发出。序列化器是 `MsgpackEncoder`（`v1/serial_utils.py` L135）/ `MsgpackDecoder`（L312），传输层 `zmq` + `zmq.asyncio`（core_client.py L22-23）；大 tensor 走 OOB 通道 `TensorIpcSender` / `TensorIpcReceiver`（`v1/engine/tensor_ipc.py` L44 / L106），不挤 msgpack 主路。

输出回程对称：EngineCore 的 `EngineCoreOutputs`（`__init__.py` L343，内含逐请求 `EngineCoreOutput` L275）经 output_queue → `process_output_sockets` → ZMQ → `AsyncMPClient` 的 `outputs_queue`（asyncio.Queue，L1209）→ `OutputProcessor`（`output_processor.py` L554）。`OutputProcessor` 持每请求 `RequestState`（L184），把新 token 交给 `IncrementalDetokenizer`（`detokenizer.py` L76；快路径 `FastIncrementalDetokenizer` L225 / 慢路径 `SlowIncrementalDetokenizer` L310）做增量 detokenize（1.4 的 UTF-8 跨 chunk 问题在此处理），结果推进 `RequestOutputCollector`（L54）——`generate()`（L740）的 async iterator 就从这里逐 token yield 给客户端。

parallel sampling（ $n$ 个输出共享 prompt）在 engine 层 fork：`ParentRequest`（`parallel_sampling.py` L13）把一条父请求拆成 $n$ 条子请求，块共享经前缀 hash + `ref_cnt` + `touch` 实现（2.1 §5 的 COW 语义，recap 一句：共享块不可变、写入至多复制一块；机制与正确性证明见 2.1 命题 2，此处不重推）。DP coordinator（`coordinator.py` `DPCoordinator` L23 / `DPCoordinatorProc` L148）只在 DP > 1 时存在，管 rank 间负载均衡与 MoE 同步步——分布式账归 5.3 / 8.1，本章只登记进程。

## 5. executor → worker：diff 协议与 persistent batch

scheduler 定好一步的组成后，怎样把它告诉 8 个 worker 进程，又不让 IPC 成为新瓶颈？答案是**只发增量**。`Executor`（`v1/executor/abstract.py` L55）是执行器抽象，三个实现：`UniProcExecutor`（uniproc_executor.py L51，单进程）、`MultiprocExecutor`（multiproc_executor.py L150，多进程）、`RayExecutorV2`（ray_executor_v2.py L239，继承 MultiprocExecutor）。`MultiprocExecutor` 用共享内存 `MessageQueue` 做广播（`rpc_broadcast_mq` L203；`MessageQueue` 本体在 `distributed/device_communicators/shm_broadcast.py` L498，环缓冲语义归 5.3），`execute_model`（L431）就是一次 `collective_rpc("execute_model", ...)`，只让 `output_rank` 回结果。worker 侧 `WorkerProc`（multiproc_executor.py L743）收 RPC，调 `Worker.execute_model`（`gpu_worker.py` L1513），再进 `GPUModelRunner.execute_model`（`gpu/model_runner.py` L1865）。

> **坑**：照旧教程锚 `ray_distributed_executor.py`。触发条件：引用 2025 年前的 Ray executor 路径；现象：当前 main 该文件不存在，Ray 执行器是 `ray_executor_v2.py` 的 `RayExecutorV2`（L239，继承 MultiprocExecutor，2026-11-23 核实）。规避：executor 锚点用 abstract / uniproc / multiproc / ray_executor_v2 四名。

增量的载体是 `SchedulerOutput`（`v1/core/sched/output.py` L241）的二分结构，源码注释逐字：「We cache the request's data in each worker process, so that we don't need to re-send it every scheduling step」（new 请求）与「Since the request's data is already cached in the worker processes, we only send the diff to minimize the communication cost」（cached 请求）。两个数据类：

- `NewRequestData`（output.py L41）：首次调度的请求发**全量**——prompt_token_ids、mm_features、sampling_params、block_ids、num_computed_tokens、lora_request 等。
- `CachedRequestData`（output.py L139）：已在 worker 缓存的请求只发**diff**——req_ids、new_block_ids（本步新分配的块）、num_computed_tokens、num_output_tokens；`new_token_ids` 仅 PP 用，非 PP 时为空。

worker 侧对应持 persistent batch。所谓 persistent batch（持久批），是指 worker 进程常驻缓存整批输入 tensor 与请求状态、每步只应用增量而不重建的批组织方式（blog 第 5 点，对治 V0 每步重建）：`InputBatch`（`gpu/input_batch.py` L43）缓存整批的 input_ids / positions / query_start_loc / seq_lens 等 tensor，`InputBuffers`（L23）按 max_num_reqs / max_num_tokens 预分配 GPU 缓冲；每步只把 diff 写进这些缓冲（Numpy 操作），不重建。块表 `BlockTables`（`gpu/block_table.py` L34）与请求状态 `RequestState`（`gpu/states.py` L9）同样常驻。这就是 blog 「Persistent Batch」的代码形态：V0 每步重建，V1 缓存 + diff。

常驻状态的核心是 `RequestState`：它持 `req_id_to_index` / `index_to_req_id` 双向映射与 `free_indices` 空闲槽列表（请求结束归还槽、新请求复用槽，不重新分配），以及 `all_token_ids`（形状 [max_num_reqs, max_model_len] 的 token 矩阵，默认用 UVA（unified virtual addressing，统一虚拟寻址）而非纯 GPU resident 以省显存，源码注释「can be extremely large... several GBs」）、prompt_len / prefill_len / total_len / num_computed_tokens 等计数器。`GPUModelRunner.execute_model`（L1865）每步开头按固定顺序维护这些状态：`finish_requests`（L1286）→ `free_states`（L1302）→ `add_requests`（L1315）→ `update_requests`（L1376），再 `block_tables.apply_staged_writes`（L1888）把本步新块号落进块表——全是增量写，没有整批重建。prompt_len 与 prefill_len 的区分值得钉：后者含抢占恢复时重放的已生成 token，故 prefill_len ≥ prompt_len（states.py L43-51 注释），frequency penalty / prompt logprobs 等特性必须区分两者（1.4 / 5.4）。

这个协议的通信量可以写成命题。先钉口径：控制面（control plane）指调度器发给 worker 的调度指令消息（SchedulerOutput，含请求状态与块号），区别于 KV 数据本身——后者常驻 worker 显存、不随步传输（口径：本章只算控制面字节，KV 与权重流量归 1.2 / 8.1）。

**命题 1**（diff 协议控制面字节上界）：设某步新调度 $n_{\mathrm{new}}$ 条请求（含抢占后恢复者）、已在 worker 缓存 $n_{\mathrm{old}}$ 条，单条请求全量状态字节为 $S_{\mathrm{full}}(T_p)$（ $T_p$ = prompt 长），单条 cached 请求 diff 字节为 $s_{\mathrm{diff}}$，则该步 scheduler → worker 控制面字节满足

$$
\mathrm{bytes}_{\mathrm{step}} \le n_{\mathrm{new}} \thinspace S_{\mathrm{full}}(T_p) + n_{\mathrm{old}} \thinspace s_{\mathrm{diff}} \qquad (2)
$$

且 $s_{\mathrm{diff}}$ 与请求上下文长 $T$ 无关（ $O(1)$）。

**证明**（路线：逐字段数两类数据类的字节，关键一步是认出 cached 请求的 diff 字段都不含随 $T$ 增长的序列本体）。`NewRequestData` 含 prompt_token_ids（ $T_p$ 个 token id）与 block_ids（ $\lceil T_p/b \rceil$ 个块号），故 $S_{\mathrm{full}}(T_p) = \Theta(T_p)$（文本口径；mm 请求另含 mm_features / prompt_embeds 等非 $\Theta(T_p)$ 项，仍只一次性发送，式 (2) 上界不受影响）；它只在请求首次调度时发一次（`scheduled_new_reqs` 列表）。`CachedRequestData` 的字段是 req_ids（定长标识串）、new_block_ids（本步新增块号，decode 每步至多 $\lceil 1/b \rceil = 1$ 个/组）、num_computed_tokens 与 num_output_tokens（两个计数器）——没有一项随已缓存的 $T$ 增长，故 $s_{\mathrm{diff}} = O(1)$。两类请求互斥且穷尽本步调度集合，求和即式 (2)。证毕。

边界反例：被抢占的请求恢复时**按新请求重发全量**——scheduler.py L1606 注释逐字「A resumed request is re-sent in full, like a new one: its blocks were freed and its worker state dropped at preemption」。所以式 (2) 的 $n_{\mathrm{new}}$ 必须把恢复者计入，命题对「恢复 = 免费 diff」的直觉不成立。

代入本章口径看量级（教学序列化口径：token id 按 int32 计 4 字节、req_id 计 8 字节、单条 diff 计 32 字节；msgpack 实际变长编码会更小，此处取上界口径）： $B$ = 64 条 decode 请求、上下文 $T$ = 4096、 $b$ = 64。全量重发协议每步发 64 × (4096 × 4 + 64 × 4 + 100) ≈ 64 × 16.35 KiB ≈ **1.02 MiB**；diff 协议每步只发 64 × 32 = **2 KiB**，比值 ≈ 523×。按 blog 的 5 ms/步折算：全量重发要持续 ~204 MiB/s 的序列化 + 拷贝带宽，diff 只要 ~0.4 MiB/s。前者本身就是一笔可观的 CPU 税（序列化 1 MiB 的 Python 对象远不止 memcpy），后者可忽略——这就是「worker 缓存 state + 只传 diff」把 IPC 移出关键路径的定量含义。式 (2) 是结构上界，不是实测吞吐；实测 IPC 时间归 5.3 的 MessageQueue 走读。

## 6. 一次请求的完整数据流

把前五节串起来：一条在线请求从 HTTP 到第一个 token，依次穿过哪些文件？下表是完整链（file:function:line 均 2026-11-23 main 核实），ASCII 时序图随后。

| 阶段 | 动作 | 落点 |
| --- | --- | --- |
| 1 接入 | HTTP chat/completions → 渲染 prompt | `entrypoints/openai/api_server.py` + `chat_completion/serving.py` L290 |
| 2 frontend | tokenize + mm 处理 → EngineCoreRequest | `AsyncLLM.add_request` L411 → `InputProcessor` L41 |
| 3 IPC 去 | msgpack + ZMQ 发往 EngineCore | `AsyncMPClient.add_request_async` L1374 |
| 4 入队 | socket 线程收 → input_queue | `EngineCoreProc.process_input_sockets` L2198 |
| 5 注册 | busy loop 消化 → scheduler.waiting | `run_busy_loop` L1814 → `EngineCore.add_request` L566 → `Scheduler.add_request` L3112 |
| 6 调度 | 定组成 + 分块（token 预算） | `Scheduler.schedule` L702（5.2 全量走读） |
| 7 执行 | 广播 SchedulerOutput → worker 前向 | `Executor.execute_model` → `Worker.execute_model` L1513 → `GPUModelRunner.execute_model` L1865 |
| 8 采样 | grammar mask + sample | `get_grammar_bitmask` L728 → `GPUModelRunner.sample_tokens` L2279 |
| 9 更新 | 追加 token / 判完成 / 释放块 | `Scheduler.update_from_output` L2334 |
| 10 IPC 回 | EngineCoreOutputs → ZMQ | `process_output_sockets` L2305 |
| 11 frontend | detokenize + 流式 yield | `OutputProcessor` L554 → `IncrementalDetokenizer` L76 → `generate` L740 |

```
HTTP ─► AsyncLLM.add_request ─► InputProcessor ─► EngineCoreRequest
                                                      │ ZMQ+msgpack
        ┌─────────────────── EngineCore 进程 ─────────▼──────────────┐
        │ process_input_sockets ─► input_queue ─► run_busy_loop      │
        │   step(): schedule ─► execute_model(non_block) ─┐          │
        │           get_grammar_bitmask（CPU，与 GPU 重叠）│          │
        │           _wait_for_output ─► sample_tokens ◄───┘          │
        │           update_from_output ─► output_queue               │
        └──────────────────────────┬─────────────────────────────────┘
              共享内存广播 ▼        │ ZMQ+msgpack
        GPU Worker ×N：persistent batch diff 更新 ─► forward ─► sample
                                   │
        OutputProcessor ◄──────────┘ ─► IncrementalDetokenizer ─► yield token
```

离线与在线的差别只在 frontend 与 client：离线 `LLM.generate` → `LLMEngine.add_request`（L231）→ `LLMEngine.step`（L323）同步拉 `engine_core.get_output()`，默认 `SyncMPClient`（或调试用 `InprocClient` 直调）；在线走 `AsyncLLM` + `AsyncMPClient` 的 asyncio 流式。离线 `LLMEngine.step` 的内部顺序是「取输出 → process_outputs → abort 命中 stop string 的请求 → 记 stats」（L323-365），与在线 `OutputProcessor` 同一套输出处理逻辑，只是一个同步拉、一个异步推。EngineCore 内部的 busy loop 与五段 step 两条路径完全一致——这是 V1 「单卡/多卡、离线/在线行为一致」对称架构的体现（blog 第 4 点）。embedding / pooling 类请求走 `AsyncLLM.encode`（L1053）而非 generate，输出是 pooler 向量不是 token 流，但接入与 IPC 链相同。token 的生命周期则统一在 scheduler 的计数器里：`num_computed_tokens` 每步前进 `num_scheduled_tokens`，采样 token 追加进 `output_token_ids`，判完成后释放块回池（4.1 §5 的「完成即补位」，机制 recap 一句，全量走读见 5.2）。

## 7. repo 导览与机制 ↔ 文件映射总表

`vllm/v1/` 的模块树长什么样？下图是 2026-11-23 main 的骨架（只列本章与 P5 走读相关模块，新增模块一句注）：

```
vllm/v1/
├── engine/            # frontend + 引擎核心
│   ├── async_llm.py / llm_engine.py      # AsyncLLM / LLMEngine（两入口）
│   ├── core.py / core_client.py          # EngineCore busy loop / IPC client
│   ├── input_processor.py / output_processor.py / detokenizer.py
│   └── coordinator.py / parallel_sampling.py / tensor_ipc.py
├── core/              # 调度器 + KV 管理（CPU 侧，5.2 全量走读）
│   ├── sched/         #   scheduler / async_scheduler / request_queue / output
│   ├── block_pool.py / kv_cache_manager.py / kv_cache_coordinator.py
│   ├── single_type_kv_cache_manager.py / kv_cache_utils.py
│   └── kv_cache_planning.py / kv_cache_config_builder.py / encoder_cache_manager.py
├── kv_cache_interface.py / kv_cache_layout.py   # KV spec 与布局（在 v1/ 根，非 core/）
├── executor/          # abstract / uniproc / multiproc / ray_executor_v2
├── worker/            # gpu_worker.py + gpu/{model_runner,input_batch,block_table,
│                      #   sample/,spec_decode/,mm/}（Sampler 在 gpu/sample/）
├── attention/         # backend selector + 实现（5.3）
├── sample/            # 只剩 ops/（top-k/top-p kernel；Sampler 不在这里）
├── spec_decode/       # metrics / dynamic 辅助（实现在 worker/gpu/spec_decode/，5.4）
├── structured_output/ # grammar backend（5.4）
├── kv_offload/ + simple_kv_offload/   # KV offload（5.4 / 8.4）
├── metrics/           # Prometheus / 统计（5.4）
└── request.py / outputs.py / serial_utils.py
```

> **坑**：把 `kv_cache_interface.py` / `kv_cache_layout.py` 锚到 `v1/core/`。触发条件：照「KV 管理都在 core/」的直觉写路径；现象：两文件实际在 `vllm/v1/` **根目录**（2026-11-23 核实，W1 已钉、W3 再确认），core/ 下找不到。规避：spec 数据类与布局枚举锚 `v1/kv_cache_interface.py` / `v1/kv_cache_layout.py`；块管理四文件才在 `v1/core/`。另：`v1/sample/` 当前只剩 `ops/`，`Sampler` 在 `v1/worker/gpu/sample/sampler.py` L42（语义纠正，5.4 详展）；`v1/watermarking/` 是生成文本水印，不是 KV watermark（5.2 §4 辨）。

P1-P4 的机制与公式落在哪些文件？下表是本章交付的映射总表（走读章公式增量 = 代码 ↔ 公式对照，承接 P1-P4；各章公式本体不重推，指针到对应走读章）：

| 机制章 | 公式 / 机制 | vLLM V1 落点（2026-11-23 main） | 走读章 |
| --- | --- | --- | --- |
| 1.1 | TTFT / TPOT / goodput 指标、请求生命周期 | `v1/metrics/{stats,loggers}.py`（SchedulerStats L224 / PrometheusStatLogger L487）+ engine 输出链 | 5.4 |
| 1.2 | prefill / decode 算力与带宽账 | 公式层；引擎侧 `profile_run`（model_runner L1100）+ `determine_available_memory`（gpu_worker L724）实测账所约束的可用显存 | 5.2 §7 |
| 1.4 | 采样公式、logits processor 链 | `gpu/sample/sampler.py` Sampler L42 + `gpu/sample/{min_p,penalties,...}.py` + `sample/ops/topk_topp_sampler.py` | 5.4 |
| 2.1 | 块表 / 碎片上界 / COW / watermark | `core/{block_pool,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager}.py` | 5.2 |
| 2.2 | hash 链 $h_i = H(h_{i-1}, \tau_i, e_i)$ | `core/kv_cache_utils.py` `hash_block_tokens` L697 + `get_computed_blocks` L278 | 5.2 |
| 2.4 | 分层存储与 offload | `v1/kv_offload/` + `simple_kv_offload/` + `kv_transfer/kv_connector/v1/` | 5.4 / 8.4 |
| 3.1 | split-KV / cascade / MLA absorbed（算法级） | `v1/attention/backends/` + `mla/`（引擎调用级） | 5.3 |
| 3.4 | 融合 / CUDA Graph / 编译 | `compilation/{piecewise_backend,cuda_graph,backends}.py` + `config/compilation.py` | 5.3 |
| 4.1 | 迭代级调度、统一 token 视图 | `engine/core.py` `step()` L714 + `core/sched/scheduler.py` `schedule()` L702 | 5.2 |
| 4.2 | 抢占成本账、调度策略 | `sched/scheduler.py` `_preempt_request` L1831 + `request_queue.py` | 5.2 |
| 4.3 | chunk 预算与 stall 账 | `schedule()` L829-833 钳位 + `config/scheduler.py` L47 / L104 | 5.2 |
| 4.4 / 4.5 | 接受率 / 加速比 / draft-verify | `gpu/spec_decode/`（TargetDependentARSpeculator 继承链）+ `rejection_sampler.py` | 5.4 |

范围外新模块一句注（2026-11-23 核实，不展开）：`v1/` 下另有 `hisparse/`（DSA/QSA local KV offload，5.3 语义节）、`kv_hints/`（orchestrator KV hint 协议，5.3）、`fault_tolerance/`、`pool/`（late interaction）、`watermarking/`（文本水印）；`distributed/` 下扩 `elastic_ep/` + `weight_transfer/`（8.5 / 9.3 指针）。这些不在 P5 大纲走读范围，登记存在即可。

## 总结

| 论断 | 回指 |
| --- | --- |
| V1 重构动机 = CPU 开销税：8B@H100 GPU 步 ~5 ms，同步循环 GPU 占比 $t_{\mathrm{gpu}}/(t_{\mathrm{cpu}}+t_{\mathrm{gpu}})$；对治 = 把 CPU 活移出关键路径（blog 口径，1.7× 吞吐增益主要来自架构） | 式 (1)、§1 |
| V0 已移除（llm_engine.py = V1 alias、无 VLLM_USE_V1）；blog alpha 限制清单是历史快照 | §1 坑注 |
| 四类进程：API server（默认 DP 个）/ EngineCore（每 DP rank 1）/ GPU worker（每 GPU 1）/ DP coordinator（DP>1 时 1）；tp4=6 进程、tp2dp4=17 进程 | §2 表 |
| 离线 LLM/LLMEngine vs 在线 AsyncLLM/OpenAI server；client 由 multiprocess × asyncio 选（Inproc/SyncMP/AsyncMP/DP） | §2 |
| EngineCore busy loop：input_queue → step 五段（schedule / execute(non_block) / grammar / sample / update）→ output_queue；grammar CPU 藏进 GPU 前向 | §3、4.1 §5 |
| frontend ↔ EngineCore = ZMQ + msgpack；输出经 OutputProcessor + IncrementalDetokenizer 流式 yield | §4 |
| diff 协议： $`\mathrm{bytes}_{\mathrm{step}} \le n_{\mathrm{new}} S_{\mathrm{full}} + n_{\mathrm{old}} s_{\mathrm{diff}}`$， $s_{\mathrm{diff}} = O(1)$；64×4096 例 1.02 MiB → 2 KiB（523×）；抢占恢复按新请求重发全量 | 命题 1、式 (2)、§5 |
| Persistent Batch = InputBatch 缓存 tensor + 每步 diff（Numpy）；execute_model 与 sample_tokens 分离 = async 接口 | §5 |
| 一次请求 11 阶段链（add_request → … → yield）；离线/在线只换 frontend 与 client，EngineCore 内部一致 | §6 |
| 机制 ↔ 文件映射总表（P1-P4 → v1/ 模块）；interface/layout 在 v1/ 根、Sampler 在 gpu/sample/、watermarking ≠ KV watermark | §7、坑注 |

## 练习

1.（计算 · [10]）进程数账（arch_overview.md Process Count Summary 口径）：(a) `-tp=8` 单节点部署有几个进程？(b) `-tp=4 -dp=2`（8 GPU）有几个？(c) `-tp=2 -pp=2 -dp=4`（16 GPU）有几个？先写出每类的数量再求和。校验：(a) 10；(b) 2 API + 2 core + 8 worker + 1 coordinator = 13；(c) 4 API + 4 core + 16 worker + 1 coordinator = 25；GPU worker 数恒等于 DP × PP × TP。
2.（计算 · [20]）diff 协议通信量（命题 1 口径，教学序列化：token id 4 字节、req_id 8 字节、单条 diff 32 字节）： $B$ = 128 条 decode、上下文 $T$ = 8192、 $b$ = 64。(a) 全量重发协议每步控制面字节？(b) diff 协议每步字节？(c) 比值与按 5 ms/步折算的两者带宽？(d) 若其中 1 条被抢占后本步恢复，diff 协议字节变成多少（恢复者按新请求重发全量）？校验：(a) ≈ 4.1 MiB；(b) 4 KiB；(c) 比值 ≈ 1043、带宽 ≈ 815 MiB/s vs 0.8 MiB/s；(d) 比 (b) 多约一条全量（≈ 32.6 KiB），与命题 1 边界反例一致。
3.（代码 · [30]）读 `vllm/v1/engine/core.py` 的 `step()`（L714）与 `run_busy_loop`（L1814，main 2026-11-23）：先写出预测——(a) `execute_model` 为什么用 `non_block=True`，紧接着的 `get_grammar_bitmask` 在 CPU 还是 GPU 上跑？(b) 若把 ③ grammar 与 ④ sample 的顺序对调（先 sample 再算 grammar），结构化输出会出什么问题？(c) busy loop 里网络 IO 在哪个线程，为什么不在 `step()` 里？再对照 L727-732 与 L2198 / L2305 验证。校验：(a) 源码 `execute_model(..., non_block=True)` 后紧跟 `get_grammar_bitmask`（CPU），grammar 时间藏进前向；(b) 采样需要 grammar bitmask 作 logits 掩码，先采样会产出违反 grammar 的 token（5.4 structured output）；(c) IO 在 `process_input_sockets` / `process_output_sockets` 独立线程，busy loop 只做 scheduler + executor。
4.（证明 · [30]）补全命题 1 的字段计数并讨论边界：(a) 逐字段列出 `CachedRequestData`（output.py L139）并论证每项 $O(1)$（与上下文长 $T$ 无关）；(b) 论证 `NewRequestData` 的 $S_{\mathrm{full}}(T_p) = \Theta(T_p)$，指出哪两个字段主导；(c) 说明为什么 PP > 1 时 `new_token_ids` 非空、这是否破坏 $s_{\mathrm{diff}} = O(1)$（提示：PP 每步每请求新增 token 数有界）；(d) 构造一个使命题取等的工作负载（每步全是新请求）。校验：(a) 与源码字段一致（req_ids / new_token_ids / new_block_ids / num_computed_tokens / num_output_tokens）；(b) 主导项 = prompt_token_ids（ $T_p$ 个）+ block_ids（ $\lceil T_p/b \rceil$ 个）；(c) PP 新增 token 数 ≤ 每步调度数，仍 $O(1)$；(d) $n_{\mathrm{old}} = 0$ 时式 (2) 取等。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——PagedAttention 与 V1 引擎的论文锚点
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——`v1/engine/{core,core_client,async_llm,llm_engine,input_processor,output_processor,detokenizer,coordinator,parallel_sampling,tensor_ipc}.py`、`v1/executor/{abstract,uniproc_executor,multiproc_executor,ray_executor_v2}.py`、`v1/worker/{gpu_worker,gpu/model_runner,gpu/input_batch,gpu/block_table,gpu/states}.py`、`v1/core/sched/output.py`、`v1/engine/__init__.py`、`v1/serial_utils.py`、`entrypoints/{llm,openai/api_server}.py`、`config/scheduler.py`、`envs.py`、`docs/design/arch_overview.md`（均 2026-11-23 main tarball file:line 核实）
- [vLLM V1 blog](../references.md)：vLLM V1: A Major Upgrade to vLLM's Core Architecture，vLLM Team，2025-01-27，[C]（key: vllm-v1-blog，开放技术博客、非闭源，全文核实）——CPU overhead 动机（~5 ms）、isolated EngineCore、{request_id: num_tokens} 调度、zero-overhead prefix caching（0% 命中 <1%）、TP 对称架构（worker 缓存 state + diff）、Persistent Batch、1.7× 吞吐、alpha 限制清单（历史快照）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——bf16 dense 989.5 TFLOPS、HBM 3.35 TB/s（章脊硬件口径）
- 跨维度指引（纯文字）：models 2.4（KV cache 容量）；inference 1.1（请求生命周期 / SLO / 在线离线分界）、1.2（prefill / decode 账）、1.3（datasheet 参数表）、1.4（增量 detokenize / 结构化输出）、2.1（块管理 / COW）、2.2（hash 链）、4.1（迭代级调度与每步循环）、4.2（抢占）、5.2（调度器与 KV 管理器全量走读）、5.3（分布式 / attention backend / 编译 / MessageQueue）、5.4（采样 / 投机 / 结构化输出 / metrics / sleep）、6.2（overlap bubble 消除账与正确性证明）、8.1（DP / MoE 通信账）