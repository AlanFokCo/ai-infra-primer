# 5.3 vLLM 分布式执行与编译（vLLM Distributed Execution and Compilation）

> 一步 decode 在 8 张卡上跑：通信落在哪几个函数、消息多大、谁来搬，attention kernel 与 CUDA Graph 又由哪几行配置决定？本章走读 vLLM 的执行链（executor → worker → model runner）、并行配置面与三条归约路径、attention backend 的引擎调用级接口、编译与图捕获模块、hisparse / kv_hints 两个新模块，末尾给三引擎分布式对照基线 | 前置：5.1（V1 架构与数据流）、5.2（调度器与 KV 管理器）、1.2（decode 步时账）、2.1（块表与 block size 权衡）、3.1（decode attention 算子族，算法级）、3.4（融合 / CUDA Graph / 编译，算子级）、2.4（KV offload 与分层账）、4.3（chunked prefill 预算）；training 5.1（TP f/g 与通信账）、training 4.1（ring all-reduce 账）只 recap 不重推 | 锚点：vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、vllm-project/vllm main（[vLLM repo](../references.md)，[A]，本章全部 file:line 核实于 2026-11-23 fresh tarball）、V1 官方 blog（[vLLM Team, 2025](../references.md)，[C] 开放技术博客、非闭源）、FlashInfer（[Ye et al., 2025](../references.md)，arXiv:2501.01005，[A]，3.1 回链）、SGLang repo（[SGLang repo](../references.md)，[A]，对照基线）、DeepInfer doc（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露，对照基线）、Mooncake（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]）与 NIXL（[NIXL repo](../references.md)，教学参照，connector 基线）

## 1. 执行链：一步 decode 经过哪几层进程

tp=4 的单节点部署有 6 个进程，tp=2、dp=4 有 17 个（官方 `docs/design/arch_overview.md` 的 Process Count Summary 表，2026-11-23 核实）。进程数不是部署花絮：它决定每一步的控制消息走哪条通道、谁持有请求状态、以及编译与图捕获在哪个进程里发生。所谓执行链，是指从调度器输出到 GPU kernel 之间的三层执行件——executor（进程间广播与收集）、worker（单卡环境与显存管理）、model runner（单步前向与采样）。

进程计数可以直接写成式子。设 API server 数为 $A$（默认等于 DP 度）、DP 度 $d$、PP 度 $p$、TP 度 $t$：

$$
N_{\mathrm{proc}} = A + d + d \cdot p \cdot t + \mathbf{1}[d > 1] \qquad (1)
$$

**符号**： $N_{\mathrm{proc}}$ = 单节点进程总数（DP 跨节点时按节点分摊，本式取单节点口径）； $A$ = API server 进程数（默认 $A = d$）； $d$ = EngineCore 进程数（每 DP rank 一个）； $d \cdot p \cdot t$ = GPU worker 进程数；末项 = DP coordinator（仅 $d > 1$ 时存在一个）。代入： $d$ = 1、 $p$ = 1、 $t$ = 4 → $1 + 1 + 4 + 0$ = 6； $d$ = 4、 $p$ = 1、 $t$ = 2 → $4 + 4 + 8 + 1$ = 17——与官方表两行逐项一致。

三层执行件的锚点（`vllm/v1/executor/`、`vllm/v1/worker/`，2026-11-23）：

| 层 | 文件与类 | 职责 | 关键方法 |
| --- | --- | --- | --- |
| executor | `executor/abstract.py` Executor L55 | 统一接口：把 SchedulerOutput 广播给 worker、收回输出 | — |
| executor | `uniproc_executor.py` UniProcExecutor L51 | 单卡：同进程直调，无 IPC | 同文件另有 ExecutorWithExternalLauncher L161（继承 UniProcExecutor，外部 launcher 启动） |
| executor | `multiproc_executor.py` MultiprocExecutor L150 | 多卡：共享内存广播 + worker 进程管理 | rpc_broadcast_mq = MessageQueue L203；WorkerProc L743 |
| executor | `ray_executor_v2.py` RayExecutorV2 L239 | 跨节点：继承 MultiprocExecutor | — |
| worker | `worker/gpu_worker.py` Worker L233 | 单卡环境、显存预算、编译预热 | init_worker_distributed_environment L1914 / load_model L637 / determine_available_memory L724 / compile_or_warm_up_model L1068 / execute_model L1513 |
| runner | `worker/gpu/model_runner.py` GPUModelRunner L209 | 单步前向与采样 | execute_model L1865 / sample_tokens L2279 / load_model L407 / profile_run L1100 |

> **坑**：按旧资料找 `ray_distributed_executor.py`——当前 main 不存在该文件，跨节点 executor 是 `ray_executor_v2.py` 的 RayExecutorV2（L239，继承 MultiprocExecutor）。触发条件：照旧 blog / 旧版文档锚定跨节点路径；现象：路径落空、或误以为 Ray 路线与多进程路线是两套独立实现；规避：按新名锚定并记核实日期（2026-11-23）。另有 `cuda_checkpoint_executor.py`（权重 checkpoint 相关，接口面归 9.3），本章不展开。

两条 IPC 通道不要混。frontend ↔ EngineCore 走 ZMQ + msgpack（5.1 走读）；executor ↔ worker 走**共享内存环**——`distributed/device_communicators/shm_broadcast.py` 的 MessageQueue L498 / ShmRingBuffer L280。分两条的理由是消息形态与频率不同：前者承载请求级控制消息与输出流（add_request、逐 token 输出），后者是每步一次的调度结果广播，负载为 persistent batch 的 diff（5.1 的 TP 对称架构：worker 侧缓存 request state、每步只传 diff）。高频小消息走共享内存环，省掉 socket 协议栈与一次拷贝（机制口径）。环缓冲语义（5.1 §5 留此走读；以下是共享内存广播环的机制口径，逐字段实现以源码为准）：ShmRingBuffer（L280）是固定槽位的共享内存环，写者**只写一份**并推进写指针，各 worker 持自己的读指针按序消费（对照逐 worker socket 发送需写 $t$ 份）；槽位复用要求读者跟上写者，否则写者等待——背压由此而来。MessageQueue（L498）在环上封装消息边界与 RPC 语义，executor 的 rpc_broadcast_mq（`multiproc_executor.py` L203）就是该环的一个实例。V1 blog 逐字口径：V0 的 scheduler 与 Worker 0 同进程以省 IPC，代价是单卡与多卡行为不对称；V1 把 scheduler 与 Worker 0 分进程，worker 侧缓存状态使单卡/多卡走同一条路径（[vLLM Team, 2025](../references.md)，[C] 开放技术博客、非闭源）。

execute_model（L1865）与 sample_tokens（L2279）在 runner 里是**两个方法**而非一个——这是 async scheduling 的接口前提：CPU 可以在 GPU 跑前向时调度下一步，采样结果晚一步回填（5.2 的 AsyncScheduler 走读）。本章的编译与图捕获（§6）也挂在这条分离线上：图捕获的对象是前向段，采样链的动态分支留在图外。

```
vLLM 执行链（tp=4、dp=1、pp=1，6 进程）
──────────────────────────────────────────────────────────
API server 进程 ──ZMQ+msgpack──► EngineCore 进程
                                   │ Scheduler.schedule()（5.2）
                                   ▼ SchedulerOutput（每步 diff）
                              Executor（MultiprocExecutor L150）
                                   │ MessageQueue（shm 环，L498）
              ┌────────────┬───────┴──────┬────────────┐
              ▼            ▼              ▼            ▼
          Worker 0      Worker 1      Worker 2     Worker 3   ← GPU worker 进程
          (L233)        每进程：init_worker_distributed_environment L1914
              │                 load_model L637 → determine_available_memory L724
              ▼                 compile_or_warm_up_model L1068
        GPUModelRunner（L209）
          execute_model L1865 ──► forward（TP 组内 all-reduce，§3）
          sample_tokens L2279 ──► 采样链（5.4 §7）
              └──────── 输出回 EngineCore → update_from_output（5.2）
──────────────────────────────────────────────────────────
```

**本章口径**（章脊锁定，换口径随文声明）：主口径 = Llama-3.1-8B dense GQA（ $L$ = 32、 $d_{\mathrm{model}}$ = 4096、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16），硬件 H100 SXM（HBM 3.35 TB/s、BF16 dense 989.5 TFLOPS、NVLink 900 GB/s——1.3 datasheet 参数表 recap，[NVIDIA datasheet](../references.md)，[C] 官方规格披露）；decode $B$ = 64、 $T$ = 4096 步时 15.1 ms、 $B$ = 1 权重-only 下界 4.81 ms、prefill $T$ = 4096 计算下界 66.5 ms——三者 recap 1.2 / 3.4 的算例，不重推。TP 通信例取 $t$ = 8（NVLink 域内）。局部记号： $t, p, e, d$ = TP / PP / EP / DP 度（spec §4.3）； $B$ = batch 内序列数（decode 每序列 1 token）； $\tau_{\mathrm{lat}}$ = 每次归约的固定延迟（教学值 5 μs，含 launch 与组同步）；bytes 不占字母，一律文字「字节」。§7 换 V4.1-Flash 口径（原因：HiSparse 只服务 DSA/QSA 系稀疏 attention 模型），§8 的 DBO 例沿用 3.4 §7 的教学参数。

## 2. 并行配置面：四个度数落到哪些字段

四个并行度在 vLLM 里是四个显式字段，还是一堆派生量？答案是显式字段加一个 auto 解析器——与 SGLang 的派生面正好相反（本节末对照）。配置在 `vllm/config/parallel.py`（2026-11-23）：tensor_parallel_size L253、pipeline_parallel_size L251、data_parallel_size L258（+ data_parallel_size_local L261）、enable_expert_parallel L301；EP 的 all-to-all 后端由 all2all_backend L332 选（默认 "auto"，解析器 _resolve_all2all_backend L128）；负载均衡由 enable_eplb L318 / eplb_config L320 / num_redundant_experts L186 控制。CP 度分两个字段：prefill_context_parallel_size L255 与 decode_context_parallel_size L549——两阶段可用不同 CP 度（serving 侧 CP 的取舍与 LoongServe 对照归 8.1，本章只登记字段）。

通信组在 `vllm/distributed/parallel_state.py`（2857 行）：GroupCoordinator L427 是统一封装，五个原语方法 all_reduce L730 / all_gather L773 / reduce_scatter L804 / gather L845 / broadcast L860；组的取用入口 get_tp_group L1613 / get_pp_group L1665 / get_dp_group L1702 / get_ep_group L1710 / get_dcp_group L1647；初始化 init_distributed_environment L1889 → initialize_model_parallel L2131。所谓通信组（group coordinator），是指一组 rank 加其上的集合通信原语封装——模型层只调「在 TP 组上 all_reduce」，不直接碰 NCCL communicator。

| 并行度 | 配置字段 | 通信组 | 每步原语（decode） | 详展 |
| --- | --- | --- | --- | --- |
| TP $t$ | tensor_parallel_size L253 | get_tp_group L1613 | 每层 2 次 all-reduce（§3） | 8.1 |
| PP $p$ | pipeline_parallel_size L251 | get_pp_group L1665 | 层间中间 tensor 点对点送 | 8.1 |
| EP $e$ | enable_expert_parallel L301 + all2all_backend L332 | get_ep_group L1710 | MoE dispatch / combine all-to-all（§4） | 3.3 / 7.4 |
| DP $d$ | data_parallel_size L258 | get_dp_group L1702 | attention 无跨 rank 通信；MoE 前 all-gather / 后 redistribute | 8.1 |
| CP $c$ | prefill_context_parallel_size L255 / decode_context_parallel_size L549 | get_dcp_group L1647 | 序列维切分的 KV 交换 | 8.1 |

PP 的执行件：`worker/gpu/pp_utils.py` + `distributed/pp_transport/`，worker 侧用 AsyncIntermediateTensors（`gpu_worker.py` L196）承接层间中间结果——异步承接是 decode PP 的 token 级流水前提（bubble 与延迟账归 8.1）。

DP attention 的执行结构是本章的一个重点，因为它改了进程拓扑而不只是改了通信组：**每 DP rank 一个独立 EngineCore + scheduler**（DPEngineCoreProc，`v1/engine/core.py` L2592；DPCoordinator，`v1/engine/coordinator.py` L23），attention 逐 DP rank 独立算自己那批请求，MoE 走 EP 组。跨 rank 的协商件在 `worker/gpu/dp_utils.py` 的 DPSyncState（dispatch_cg_and_sync_dp 跨 DP rank 协商 CUDA graph）。为什么图捕获要跨 rank 协商？因为 MoE 的 all-to-all 是集合通信：全部参与 rank 必须进入配对的同一次调用。若各 rank 按本地 batch 独立选图（不同捕获 size、不同 kernel 序列），集合通信的配对就会错位——所以「这一步走哪张图」必须是组内一致的决定，而不是各 rank 的本地决定。MLA 模型为什么倾向这条路线（单 KV head 切不动 TP、KV 复制问题）recap 一句：通信账与容量账归 8.1，本章只交付执行结构。

> **坑**：把 SGLang 的 `--enable-dp-attention` 当当前字段写。触发条件：照 v0.4 blog 或旧文档锚定 SGLang 并行配置；现象：当前 main 无此字段——它由 attn_dp_size > 1 派生（`server_args.py` L345/L355），且 tp_size = world size、attn_tp_size = tp_size // attn_dp_size // attn_cp_size（`runtime_context.py` derive_attn_tp_size L73），配置经 `arg_groups/parallel_hook.py` + get_parallel()（L1314）读取；规避：SGLang 侧按 runtime_context / arg_groups 新面锚定（6.3 走读），vLLM 侧按 config/parallel.py 显式字段锚定，两引擎配置面不可互译（2026-11-23 核实）。

## 3. TP 通信账：serving 没有 backward

training 5.1 的账是每层 4 次 all-reduce；serving 只跑前向，次数减半——但消息量也缩到 decode 的 $B$ 个 token，延迟项反而成了主项。这一节把差异算到数。

recap training 5.1 的 f/g 共轭（不重推）：f 前向恒等、反向 all-reduce；g 前向 all-reduce、反向恒等。一层 transformer 里 attention 块与 MLP 块各贴一对 f/g，故前向 2 次归约（两处 g）、反向 2 次（两处 f），合计 4 次；消息量 $2bsh$ 字节（其 $b$ = micro-batch 序列数、 $s$ = 序列长、 $h$ = hidden 宽）。

**命题 1**（serving 归约位点减半）。推理前向中，TP 组内每层的集合归约位点为 2 处（attention 块与 MLP 块各一处），是 training 每层 4 次的一半；且这 2 处都在前向图内触发，不存在反向归约。

**证明**（路线：推理图的算子集合 → f/g 的触发条件 → 计数，三步）。第一步，推理只执行前向图，反向图不存在（无 loss、无 autograd 反向遍历）。第二步，f 的归约在其反向、g 的归约在其前向（training 5.1 式 (5)(6) 的共轭定义）——前向图里只有 g 触发，f 是恒等。第三步，每层两个行切位点（attention 的 out 投影、MLP 的 down 投影）各贴一个 g，故每层恰 2 处。∎ 正例：8B、 $L$ = 32 → 每步 64 处归约。边界两条：其一，开 sequence parallelism 时归约原语换成 reduce-scatter + all-gather 对（GroupCoordinator 的 reduce_scatter L804 / all_gather L773），原语数翻倍但**归约位点仍是每层 2 处**——命题计的是位点，不是 kernel 数；其二，vocab parallel embedding 另有一处前向归约（training 5.1 §9），按整模型计须加 1，本命题只计 transformer 层。

消息量与时间。decode 每序列 1 token，training 的 $2bsh$ 翻译到本维度（ $b \to B$ 、 $s \to 1$ 、 $h \to d_{\mathrm{model}}$ ，spec §4.3 翻译硬规则）即：

$$
M_{\mathrm{msg}} = 2 \thinspace B \thinspace d_{\mathrm{model}} \qquad (2)
$$

**符号**： $M_{\mathrm{msg}}$ = 单次 all-reduce 的消息字节数；系数 2 = bf16 每元素字节数（字节量已编码在系数里）； $B$ = batch 内序列数（decode 口径，每序列 1 token）； $d_{\mathrm{model}}$ = hidden 宽。代入 $B$ = 64、 $d_{\mathrm{model}}$ = 4096： $M_{\mathrm{msg}}$ = 524288 字节 = 512 KiB； $B$ = 1 时 8192 字节 = 8 KiB。prefill 口径（ $T$ 个 token 一次前向，换口径原因：展示消息量随 token 数线性涨）： $T$ = 4096 → 32 MiB，是 decode $B$ = 64 的 64 倍。

每步 TP 通信时间 recap training 4.1 式 (2)(3) 的 ring 账（每卡收发字节乘 $2(t-1)/t$ 、延迟项 $2(t-1)$ 跳），符号翻译：training 的 $\alpha$ （每跳延迟）在本维度译作 $\tau_{\mathrm{hop}}$ （ $\alpha$ 专归投机接受率）、 $\beta$ （每字节时间）译作 $1/\mathrm{BW}$ ：

$$
\tau_{\mathrm{TP}} = 2L \left( \tau_{\mathrm{lat}} + \frac{2(t-1)}{t} \cdot \frac{M_{\mathrm{msg}}}{\mathrm{BW}} \right) \qquad (3)
$$

**符号**： $\tau_{\mathrm{TP}}$ = 每步 TP 通信总时间； $2L$ = 每步归约次数（命题 1 的位点数，all-reduce 口径；SP 变体换原语另计）； $\tau_{\mathrm{lat}}$ = 每次归约的固定延迟（含 launch、组同步与跳数；教学值 5 μs，ring 口径下 $\approx 2(t-1)\tau_{\mathrm{hop}}$ ，实测依硬件与实现）； $\mathrm{BW}$ = 链路带宽（NVLink 900 GB/s，datasheet 口径）； $t$ = TP 度。代入 $t$ = 8、 $B$ = 64：带宽项 $= 1.75 \times 524288 / (9 \times 10^{11})$ = 1.02 μs， $\tau_{\mathrm{TP}} = 64 \times (5 + 1.02)$ = 385 μs。 $B$ = 1：带宽项 0.016 μs， $\tau_{\mathrm{TP}}$ = 321 μs。**消息量缩 64 倍，通信时间只降 16.7%**——延迟项与消息量无关，小 batch decode 的 TP 成本几乎全是它。

读数须带口径注：385 μs 对单卡步时 15.1 ms 是 2.55%、321 μs 对 $B$ = 1 的 4.81 ms 是 6.67%，但这两个分母是**单卡对照基线**——TP 后权重读与算力按 $t$ 分片，步时本身下降，通信占比随之上升。度数选择、TPOT 全账与 NVLink 域边界归 8.1（该章强制重推导 TP decode 延迟账），本章只交付「次数减半 + 延迟项主导」这两条结构结论与代码落点。prefill 侧代入：带宽项 $= 1.75 \times 33.55 \times 10^{6} / (9 \times 10^{11})$ = 65.2 μs， $\tau_{\mathrm{TP}} = 64 \times (5 + 65.2)$ = 4.50 ms，对 66.5 ms 单卡 prefill 是 6.8%——**带宽项主导，与 decode 相反**。

> **提点**：通信优化的方向由消息量决定，不由「通信很贵」这句笼统判断决定。decode 小消息压延迟项（§4 的自定义归约路径），prefill 大消息压带宽项（ring 已近带宽最优，能动的是拓扑与重叠）。同一个 TP 组，两个阶段的最优路径不同——这是引擎要准备多条归约路径的账面理由。

## 4. 谁来搬：三条归约路径与 EP all-to-all 族

式 (3) 的延迟项与消息量无关——8 卡 ring 就是 14 跳的固定成本。vLLM 为此准备了三条路径：NCCL、custom allreduce、PyTorch symmetric memory，选择由两个门函数决定。

统一入口是 GroupCoordinator.all_reduce（L730），内部经 CudaCommunicator（`device_communicators/cuda_communicator.py` L35）分派到三个实现：

| 路径 | 类与锚点 | 门函数 | 机制要点 |
| --- | --- | --- | --- |
| NCCL | PyNcclCommunicator，`pynccl.py` L91 | 默认回退 | ring / tree 集合通信，带宽项近最优 |
| custom allreduce | CustomAllreduce，`custom_all_reduce.py` L120 | should_custom_ar L511 | 自定义归约 kernel；custom_all_reduce L561 / custom_all_gather L603 / custom_reduce_scatter L679；MNNVL multimem 支持 |
| symmetric memory | SymmMemCommunicator，`symm_mem.py` L29 | should_use_symm_mem L130 | PyTorch symmetric memory 路径，all_reduce L140 |

两个门函数（should_custom_ar / should_use_symm_mem）判定本次归约走哪条路径，未命中则回退 NCCL（口径：判据含消息大小与组内互联能力，具体阈值随版本与硬件变化，本章只交付分派结构与延迟账，不引阈值数值）。自定义路径的账面动机就是 §3 的结论：把 ring 的多跳延迟项压下来，代价是适用条件受限（组规模、互联拓扑、消息大小）——所以它是门控的快路径，不是无条件替换。

CUDA Graph 兼容是自定义归约的硬约束。CustomAllreduce 提供 capture L476 与 register_graph_buffers L489 两个方法：图捕获时把归约用的 buffer 注册进图，replay 才能按固定指针找到对端地址。

> **坑**：把自定义归约捕进 CUDA Graph 而不注册图 buffer。触发条件：capture 路径未走 register_graph_buffers（L489）；现象：replay 按捕获期记录的指针访问归约 buffer，违反 3.4 命题 2 的「指针固定」前提——输出不随输入变或归约错位；规避：capture / register_graph_buffers 配对使用，或让该归约落在图外（piecewise 切分点，§6）。

EP 侧的通信不是 all-reduce 而是 all-to-all（MoE dispatch / combine，语义与账 recap 3.3，DeepEP 模式与硬件层归 gpu 10.3）。vLLM 把它做成一个 manager 族——`distributed/device_communicators/all2all.py`（1381 行，2026-11-23）：

| manager | 锚点 | 后端 / 路线 |
| --- | --- | --- |
| AgRsAll2AllManager | L57 | all-gather + reduce-scatter 组合路线 |
| DeepEPHTAll2AllManager | L257 | DeepEP high-throughput |
| DeepEPLLAll2AllManager | L316 | DeepEP low-latency |
| DeepEPV2All2AllManager | L1222 | DeepEP V2.5 统一 EPBuffer 接口 |
| NixlEPAll2AllManager | L441 | NIXL 传输 |
| FlashInferNVLinkTwoSidedManager / OneSidedManager | L680 / L804 | NVLink 双侧 / 单侧 |
| MoriAll2AllManager / MoriEPv2All2AllManager | L968 / L1123 | Mori |
| MoonEPAll2AllManager | L1316 | MoonEP |

> **坑**：按 DeepEP V1 的「normal 两阶段 vs low-latency 纯 RDMA hook」语义读当前引擎侧接口。触发条件：照旧文档理解 DeepEPHT / DeepEPLL 两个 manager 的关系；现象：DeepEP 当前 main 是 V2.5 统一 EPBuffer（V1 API 已移除，W2 核实口径，语义锚 tag v1.2.1），vLLM 侧 DeepEPV2All2AllManager（L1222）正是统一接口的引擎落点；规避：讲 V1 语义时锚 tag、讲当前实现时锚 V2 manager，两个口径分列（3.3 已交付漂移注）。

选择由 all2all_backend（默认 "auto"）经 _resolve_all2all_backend（L128）解析——与 attention backend 的 selector（§5）同构：能力与硬件约束下的自动选择，配置面只留一个枚举。EPLB 的执行件在 `distributed/eplb/`：EplbState（`eplb_state.py` L321）/ EplbModelState L77 + rebalance_execute.py / policy/ / migration_scheduler.py / async_worker.py——vLLM 自有 EPLB 实现，与 SGLang `eplb/`、DeepSeek EPLB 的算法对照归 7.4 / 8.1，本章只登记模块与配置字段（enable_eplb L318 / num_redundant_experts L186）。范围外新模块一句注：`distributed/elastic_ep/`（弹性 EP，→ 8.5）、`distributed/weight_transfer/`（nccl / m2n / modelexpress / ipc / sharded_rdt 五类权重传输 engine，→ 9.3）、`v1/fault_tolerance/engine_core_sentinel.py`（容错，→ 8.5），2026-11-23 核实存在，本章不展开。

## 5. attention backends：selector 与接口契约（引擎调用级）

同一个 8B 模型，换一张卡或换一种 KV 精度，attention kernel 就换一个——谁做这个选择、依据是什么？本节走读引擎调用级：selector 逻辑、backend 接口契约、metadata 构建、CUDA Graph 兼容声明。**两层走读纪律**（spec §4.2）：3.1 已交付算子算法级（split-KV 归约正确性、cascade 两级、MLA 吸收形几何、FlashInfer plan/run 与 BSR 格式），本节 recap 一句 + 指引 3.1，不重推其公式与证明；增量只在「引擎如何选与调」。

选择件在 `vllm/v1/attention/selector.py`：get_attn_backend L111（入口）、_cached_get_attn_backend L212（按配置缓存选择结果）、AttentionSelectorConfig L25（选择依据的配置结构）、get_attn_spec_kind L70（取 KV spec 种类——对接 5.2 的 KVCacheSpecKind 枚举）。接口契约在 `backend.py`：AttentionBackend L57 定义一组能力查询与校验方法，AttentionMetadata L402 / CommonAttentionMetadata L410 定义 metadata 基类与跨 backend 公共部分。

| 方法（AttentionBackend L57） | 约束对象 | 对应机制章 |
| --- | --- | --- |
| get_impl_cls / get_builder_cls | 实现类与 metadata 构建器 | 本节下表 |
| supports_head_size | $d_{\mathrm{head}}$ 几何 | 3.1 §5（MLA 512 + 64） |
| supports_kv_cache_dtype | KV 精度（bf16 / fp8 / nvfp4 等） | 2.3 |
| get_preferred_block_size L168 | backend 偏好的 block size $b$ | 2.1 式 (6)(7) |
| supports_batch_invariance L222 | 是否支持批不变模式 | 3.4 §6 |
| supports_hisparse L230 | 是否支持 HiSparse 数据面 | §7 |
| validate_configuration L296 | 组合合法性校验 | 本节 |

这张表就是 selector 的判据来源：选择不是「哪个 kernel 快」的 benchmark 排序，而是**能力约束下的可行性筛选**——head size、KV dtype、block size、图模式、批不变、HiSparse 六项约束先过滤，再在可行集里按平台与偏好定一个。validate_configuration（L296）是组合校验的兜底：单项能力都支持不代表组合合法（例如某 backend 支持 fp8 KV 但不支持 fp8 KV + full graph）。

backend 族（`v1/attention/backends/`，2026-11-23）：

| backend | 锚点 | 定位 |
| --- | --- | --- |
| FlashAttention | flash_attn.py FlashAttentionBackend L396 / FlashAttentionImpl L1154 | FA varlen 接口，3.1 §6 已展示其 max_num_splits 三态 |
| FlashInfer | flashinfer.py FlashInferBackend L666 / FlashInferImpl L2355 | plan/run 两阶段，3.1 §4 算法级 |
| Triton | triton_attn.py TritonAttentionBackend L306 / TritonAttentionImpl L409 | Triton 实现路线 |
| MLA 子目录 | mla/{flashmla,flashmla_sparse,cutlass_mla,triton_mla,flashinfer_mla,flashinfer_mla_sparse,flashattn_mla,sparse_swa,cpu_mla,amx_mla,indexer,sparse_indexer,index_group,tokenspeed_mla}.py | MLA / DSA 几何专用（含 HiSparseMLAIndexGroup） |
| 其他 | flex_attention / cpu_attn / flash_attn_diffkv / triton_flash_attn / rocm_* / mamba*_attn / linear_attn / gdn_attn | 平台与模型族分支 |

注册表在 `backends/registry.py`；目录已大幅扩展（b12x / composite / flashinfer_ts / turboquant / ultraquant / zentorch_sdpa 等，2026-11-23 核实），超出本章范围的一句注，不展开。

block size 偏好是代码与公式的一个直接接缝。2.1 式 (6)(7) 给出碎片与块表开销权衡下的最优 $b^{\ast}$ ；引擎侧默认 64（`config/cache.py` DEFAULT_BLOCK_SIZE L185），但 backend 侧有两种不同的约束形态：`mla/sparse_swa.py` 的 get_preferred_block_size 返回 256（L166-167，**偏好覆盖**）；`mla/cpu_mla.py` 无 get_preferred_block_size，16 来自 supports_block_size 的硬约束（L71-72：`block_size is None or block_size == 16`）加平台层强制（模块 docstring L25-26 逐字「the platform layer forces block_size=16 for MLA models」）——**只接受 16，不是偏好覆盖**。机制解释：式 (7) 的权衡是**内存侧**的，backend 的偏好或硬约束是**kernel 访存粒度侧**的（一次 gather 的连续字节数、tile 形状）——最终 $b$ 取两者的交集，而不是任一单独最优。

metadata 构建走 get_builder_cls 返回的 builder：它把 scheduler 输出（每请求 token 数、block table、KV 写入 slot）转成 backend 专用 metadata。3.1 §6 已展示 FlashAttentionMetadata 的 block_table / scheduler_metadata / max_num_splits 三件与 max_num_splits 的三态语义（0 启发式 / full graph 固定上界 / batch-invariant 取 1），此处 recap + 指引，不重列。CommonAttentionMetadata（L410）承载跨 backend 的公共字段，专用字段由各 backend 的 metadata 类扩展。CUDA Graph 兼容由 AttentionCGSupport 声明（backend 支持哪种图模式），与 §6 的 CUDAGraphMode 配对：声明不支持 full graph 的 backend 只能走 piecewise 或 eager。

教学片段（教学用，非优化实现）——selector 的决策骨架：

```python
def select_backend(cfg, kv_spec_kind, platform):
    """教学用，非优化实现：线性扫描 + 能力过滤。"""
    candidates = registry.list_backends(platform)      # 官方：注册表按平台分派
    for be in candidates:
        if not be.supports_head_size(cfg.head_size):
            continue
        if not be.supports_kv_cache_dtype(cfg.kv_dtype):
            continue
        if cfg.batch_invariant and not be.supports_batch_invariance():
            continue
        if cfg.hisparse and not be.supports_hisparse():
            continue
        be.validate_configuration(cfg)                 # 组合合法性兜底
        return be
    raise ValueError("no feasible attention backend")
```

与官方实现的差异三点：官方 get_attn_backend（L111）经 _cached_get_attn_backend（L212）按 AttentionSelectorConfig 缓存结果（选择每模型一次，不是每步一次）；官方的候选序与平台判定由 registry 与平台层决定，教学版简化为线性扫描；官方的 KV spec 种类经 get_attn_spec_kind（L70）取自 5.2 的 KVCacheSpecKind，教学版当作入参。

```
attention 调用链（引擎调用级；算法级归 3.1）
──────────────────────────────────────────────────────────
config + KV spec kind（5.2）
   │ get_attn_spec_kind L70
   ▼
selector.get_attn_backend L111 ──缓存──► _cached_get_attn_backend L212
   │ 能力过滤：head_size / kv_dtype / block_size / CG / batch_invariant / hisparse
   ▼
AttentionBackend（L57）
   ├─ get_builder_cls ──► builder：SchedulerOutput → AttentionMetadata
   │                        （block_table / seq_lens / slot_mapping → backend 专用字段）
   └─ get_impl_cls ─────► Impl.forward(hidden_states, metadata, kv_cache)
                            │ 3.1 的算子算法在此被调用（split-KV / cascade / absorbed MLA）
                            ▼ (O, LSE)  ← LSE 是标准输出（3.1 §6）
──────────────────────────────────────────────────────────
```

## 6. 编译与图捕获：模块走读

3.4 算过：330 个 kernel × 5 μs = 1.65 ms 的 launch 开销， $B$ = 1 时占步时三分之一。vLLM 用哪几个文件把这笔开销抹掉？本节走读实现件与配置面；融合 IO 账、launch 账、piecewise 切分语义与正确性命题 recap 3.4，不重推。

配置面两个枚举（`config/compilation.py`，与 W2 口径一致、无漂移）：CompilationMode L38——NONE = 0（eager）/ STOCK_TORCH_COMPILE = 1 / DYNAMO_TRACE_ONCE = 2 / **VLLM_COMPILE = 3**（自定义 Inductor backend：caching + piecewise compilation + shape specialization + custom passes）；CUDAGraphMode L165——NONE / PIECEWISE / FULL / FULL_DECODE_ONLY = (FULL, NONE) / FULL_AND_PIECEWISE = (FULL, PIECEWISE)。旧名 CompilationLevel 与「PIECEWISE = 3」在当前 main 不存在（3.4 §3 坑注已交付，此处只回指）。

实现件的走读链（`vllm/compilation/`，2026-11-23）：

| 件 | 锚点 | 职责 |
| --- | --- | --- |
| decorators.py | support_torch_compile L88-123（overload 族） | 装饰模型类，声明可编译入口 |
| backends.py | CompilerManager L133 / split_graph L580 / PiecewiseCompileInterpreter L714 | 编译管理、按规则切图、逐段编译解释器 |
| partition_rules.py | 74 行 | attention 切分规则（切分点集合） |
| piecewise_backend.py | PiecewiseBackend L86 / RangeEntry L80 | 按 shape 区间选编译产物 |
| cuda_graph.py | CUDAGraphWrapper L144 / CUDAGraphEntry L127 | 图捕获与重放封装，每捕获 size 一条目 |
| cudagraph_pool.py / breakable_cudagraph.py / caching.py | — | 图内存池 / 可断图 / 编译缓存 |
| passes/ | fusion / collective_fusion / sequence_parallelism 等 | Inductor 自定义 pass |

链条读法：support_torch_compile 装饰模型 → Dynamo trace 出 FX graph → VLLM_COMPILE backend 接手（CompilerManager）→ split_graph（L580）按 partition_rules.py 的规则在 attention 算子处切段 → 每段经 PiecewiseCompileInterpreter（L714）编译 → 运行时 PiecewiseBackend（L86）按输入 shape 落在哪个 RangeEntry（L80）选对应编译产物 → 段外层由 CUDAGraphWrapper（L144）按 CUDAGraphEntry（L127，每捕获 size 一条）capture / replay。两个索引键不要混：**RangeEntry 按 shape 区间索引编译产物，CUDAGraphEntry 按捕获 size 索引图**——前者决定「跑哪份编译代码」，后者决定「重放哪张图」。

passes/ 里的 collective_fusion 值得单独点名：它把通信算子与相邻计算融合进同一 Inductor pass 的变换范围——这是**编译器级**的通信-计算重叠手段，与 §4 的手写归约路径、§8 的 DBO 批级重叠是三个不同层次（编译器 / kernel / 批调度）。sequence_parallelism pass 则对应 §3 命题 1 边界里的 SP 变体（归约换 reduce-scatter + all-gather 对）。

| 配置组合 | 语义 | 适用 |
| --- | --- | --- |
| VLLM_COMPILE + FULL_AND_PIECEWISE | decode 批走 full graph、prefill 与混合批走 piecewise | V1 默认（3.4 §3 口径） |
| VLLM_COMPILE + FULL_DECODE_ONLY | 只给 decode 批捕 full graph，混合批不捕 | PD 分离的 D 实例省图内存（8.2） |
| STOCK_TORCH_COMPILE + NONE | 标准 torch.compile，无图捕获 | 调试 / 不支持图的平台 |
| NONE + NONE | 全 eager | 基线对照 |

启动代价两笔挂在 Worker.compile_or_warm_up_model（L1068）：每个捕获 size 各捕一张图 × warmup 次数（启动时间），图内存池 + 常驻 buffer 占 HBM（与 KV 容量互斥——5.2 的 kv_cache_planning 容量规划须把图内存计入预留）。piecewise 依赖 piecewise compilation（mode = VLLM_COMPILE 且 splitting_ops 非空，3.4 §4 口径），full graph 有无编译都支持。

```
编译与图捕获流水线（VLLM_COMPILE + FULL_AND_PIECEWISE）
──────────────────────────────────────────────────────────
模型类 ──support_torch_compile（decorators.py L88）──► Dynamo trace
   ▼ FX graph
CompilerManager（backends.py L133）
   │ split_graph L580 ← partition_rules.py（attention 切分规则）
   ▼ 段 1 / 段 2 / ...（attention 算子为切分点）
PiecewiseCompileInterpreter L714 → 逐段 Inductor 编译（passes/：fusion / collective_fusion）
   ▼ 编译产物按 shape 区间归档
PiecewiseBackend L86 ── RangeEntry L80（shape → 产物）
   ▼ 运行时
CUDAGraphWrapper L144 ── CUDAGraphEntry L127（捕获 size → 图）
   replay 段 1 → attention（eager，读当前 T / block table）→ replay 段 2 → ...
──────────────────────────────────────────────────────────
```

## 7. hisparse 与 kv_hints：稀疏 attention 的本地 offload 与编排协议

DSA 模型每步主 attention 只读 512 行 KV（3.1 式 (10) 的 20 MiB），全量 131072 行为什么还要留在 GPU？答案是没必要——这正是 HiSparse 的账面动机。所谓 HiSparse，是指 DSA / QSA 系模型的**本地** KV offload 方案：GPU 只留 per-request hot buffer，全量 KV 常驻 host，按 attention 的选择结果换入（官方 `docs/design/hisparse.md`，321 行，Status: experimental，2026-11-23 核实；配置类 HiSparseConfig docstring 逐字「Configuration for HiSparse DSA and QSA KV offloading」）。**两层纪律**：2.4 已交付三层存储账、临界带宽与 SGLang HiSparse 机制，本节只走读 vLLM 侧实现落点与两个新模块的语义。

三角色分工（doc 口径）：KV cache system 管 GPU 块分配；HiSparseCoordinator 管 host 逻辑块与 residency（驻留状态）；HiSparseConnector + HiSparseWorker + HiSparseRuntime 管 scheduler ↔ worker 的传输与数据面。doc 逐字划清分配权：「Neither worker-side object allocates or frees logical blocks. HMA provides the GPU allocation shared by the resident and hot groups」——worker 侧对象不分配也不释放逻辑块，GPU 分配由 HMA 提供、resident 与 hot 两组共享（HMA 缩写沿 doc 原文，指统一 GPU 分配层）。

文件走读（`vllm/v1/hisparse/`，8 文件，2026-11-23）：

| 文件 | 内容 |
| --- | --- |
| backend.py | HiSparseBackend：device ops ABC（pin_tensor / create_cache_ops / validate_kv_cache_config） |
| runtime.py | resident / host / hot-cache 数据面（模块 docstring「HiSparse resident, host, and hot-cache data plane」）；HiSparseTopKResult |
| coordinator.py | HiSparseCoordinator：消费三个 HiSparse manager |
| types.py | SparseKVPageTransfer / SparseKVRowMirror / SparseKVOffloadCommand / SparseKVResidencyUpdate；ACTIVE_TAIL_PAGES = 2 |
| layout.py / binding.py | KV cache group 布局 / worker 侧 attention 层接线 |
| block_pool.py | SharedEventQueueBlockPool |

manager 三件在 5.2 已锚（`single_type_kv_cache_manager.py`：HiSparseSourceManager L2821 / HiSparseHotManager L3018 / HiSparseResidentManager L3135），spec 数据类为 HiSparsePageSpec（`v1/kv_cache_interface.py` L427）与 HiSparseHotPageSpec L442——recap + 指引 5.2，本节不重列其块几何。

配置两个旋钮（`config/attention.py` HiSparseConfig L21）：device_buffer_size = per-request GPU hot-buffer **行数**，默认 = 1 份 top-k（当前 decode query 要用）+ 1 份 top-k 的 LRU slack（缓存余量）；eager_host_mirror = True（默认）= decode 写出的 KV 行在 forward 期间镜像到 host——与 SGLang HiSparse 的 eager backup 同构（2.4 §9）。部署面经 kv_transfer_config 的 kv_connector = "HiSparseConnector" + host_pool_gib；host_pool_gib 是 **per-DP-replica** 可用 host cache 容量，不是节点预算（doc 口径，多 DP rank 共节点时勿按节点总量代入）。

驻留比与容量账。设 top-k 行数 $K$ 、上下文长 $T$ 、每 token 每层 KV 字节 $\kappa$ 、层数 $L$ ，默认 hot buffer 行数 $= 2K$ ：

$$
f_{\mathrm{hot}} = \frac{2K}{T}, \qquad M_{\mathrm{hot}} = 2K \thinspace \kappa \thinspace L \thinspace B \qquad (4)
$$

**符号**： $f_{\mathrm{hot}}$ = GPU 驻留行数占全量 KV 行数的比例（记 $f$ 不记 $\rho$ ： $\rho$ 按 spec §4.3 专归利用率）； $M_{\mathrm{hot}}$ = 全 batch 的 hot buffer 字节数； $K$ = index_topk； $B$ = batch 内请求数。换 V4.1-Flash 口径（原因：HiSparse 只服务 DSA/QSA 系模型； $K$ = 512、 $L$ = 40 取自 config [W]， $\kappa$ = 1024 字节为 MLA bf16 教学口径，V4.1 共享 KV 向量 512 元素 = 448 latent + 64 rope、见 1.2 §3；生产主 KV 用 per-16 e4m3 低精度、格式账归 2.3 / 2.4）： $T$ = 131072 时 $f_{\mathrm{hot}}$ = 1024/131072 = **0.78%**；单请求 hot buffer = 1024 × 1024 × 40 = 41.9 MB（40 MiB），全量对照 5.37 GB； $B$ = 64 时 hot 合计 2.68 GB，全量对照 343.6 GB——GPU 侧容量需求降两个数量级，这就是「decode 不被容量约束」之外的另一条路（2.4 §9 的 DeepInfer 口径是不 offload，HiSparse 是 offload 但只换入 top-k）。

换入的带宽账 recap 2.4 式 (3) 同形：全 miss 一步要换入 $K \kappa L$ = 512 × 1024 × 40 = 20 MiB，PCIe 单向 64 GB/s（2.4 §1 口径）下 0.33 ms——与 3.1 式 (10) 的主路读取量同值（主路读多少就得换入多少，全 miss 时）。hot buffer 的 LRU slack 与 eager_host_mirror 就是把这笔摊薄的两个机制：前者让跨步重复选中的行不换入，后者让 decode 新写的行立刻有 host 副本、不必等驱逐再写。

**命题 2**（active tail 不外借）。HiSparse 的 page 外借指 `_unpin_pages` 把 resident 块释放回池复用（coordinator.py L395-407；eager_host_mirror 下 host 副本已在，unpin = 放弃 GPU 驻留，不是写回 host），该操作跳过每请求末尾 ACTIVE_TAIL_PAGES = 2 页（L402：num_unpinnable = num_resident − ACTIVE_TAIL_PAGES）。若单步写入的 token 数不超过 $2b$ （ $b$ = block size），则任一 in-flight step 正在写入的 page 在该步结束前不会被释放回池。

**证明**（路线：写入位置集合 → active tail 覆盖 → 外借集合不相交，三步）。第一步，decode 一步每请求写入至多 $\gamma + 1$ 个 token（无投机时 1 个，4.5 §6），写入位置总在序列末尾，故落在末尾 $\lceil (\gamma+1)/b \rceil$ 页内。第二步，前提 $\gamma + 1 \le 2b$ 给出 $\lceil (\gamma+1)/b \rceil \le 2$ ，即写入页集合含于末尾 2 页 = active tail 集合。第三步，外借规则跳过 active tail，故外借集合与写入集合不相交——正在写的页不被抽走。∎ 边界反例：大 chunk prefill 一步写入可达数千 token（4.3 的 $c_{\mathrm{chunk}}$ ），远超 $2b$ （ $b$ = 64 时 128 token），末尾 2 页不足以覆盖——此时保护须由别的机制承担（prefill 路径不走 decode 的 top-k 换入），命题前提明确限定在 decode 步。常量取 2 而非 1 的账面理由：给 LRU 重排与 in-flight 写入留一页余量（口径：本节按 types.py L6-8 的常量与注释逐字「stay pinned so a page written by an in-flight step is never handed out under it」推导，「留一页余量」是本书解释而非官方说明）。

其余 doc 事实四条（只登记，不展开）：MLA 的 TP rank 持 replicated view（KV 在 TP 组内复制，与 §2 的 DP attention 动机同源）；QSA（Qwen4Exp）用 BLNHC layout，indexer keys / raw-key rings / recurrent state 留 GPU 不外借；startup 有两条 concurrency bound（generic 与 HiSparse steady-state 分别估算）；HiSparse 是 local KV connector，与 PD connector 经 MultiConnector 组合（§9）。

kv_hints 是另一个新模块，语义与 offload 无关而是**编排协议**。所谓 KV hint，是指 orchestrator（外层编排器）随请求附带的版本化 KV 动作声明——告诉引擎「这条请求的 KV 该从哪取、放到哪」，而不是由引擎自己猜。`v1/kv_hints/protocol.py`：KvHintAction（action_id / action_type / action_version / payload）+ KvHintsEnvelope（protocol_version / message_id / actions），均为 msgspec frozen Struct（不可变、可序列化）。消费链三跳：Request.kv_hints（`v1/request.py` L85）→ offloading connector 的 scheduler 侧把 req.kv_hints 装进请求上下文（`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` L172）→ kv_offload 的 ReqContext.kv_hints 字段（`v1/kv_offload/base.py` L109，供 offloading manager 消费）；action 的 schema 由实现组件校验（协议只定信封，语义归消费方）；kvcr tier 侧有 ROUTER_HINT_CAPABILITIES（`tiering/kvcr/manager.py`，KVCR 二级存储 adapter）。版本化字段（action_version / protocol_version）是协议演进的正确性要件：消费方按版本判定能否处理，不认的版本拒绝而非误读。

| 轴 | vLLM HiSparse（本节） | SGLang HiSparse（2.4 §9 已交付） |
| --- | --- | --- |
| 同族机制 | DSA 系 decode top-k hot buffer + host 全量 KV | 同 |
| 模型覆盖 | DSA + QSA（Qwen4Exp，BLNHC layout） | DSA 系（doc 列 V3.2 / GLM-5.1 / Kimi-K3.1 / V4 / MiniMax M3） |
| 写回 | eager_host_mirror（forward 期间镜像） | eager backup（异步回写上一 token KV） |
| PD 集成 | local connector，经 MultiConnector 与 PD connector 组合 | PD direct-to-host（prefill GPU 经 RDMA 直写 decode host pool） |
| 编排协议 | kv_hints（版本化 action 信封） | —（2.4 未交付对应模块） |

## 8. 多模态执行与 DBO：两个重叠面

vision encoder 的前向与 LLM 的 decode 步在同一张卡上抢 SM——vLLM 把它们放在哪个执行件里？多模态执行件三处（2026-11-23）：`worker/gpu/mm/encoder_runner.py` EncoderRunner L36（+ encoder_cache.py）、`worker/mm_encoder_model_runner.py` MMEncoderModelRunner L38（**继承 GPUModelRunner**——encoder 走同一套 runner 基础设施，不是独立引擎）、`multimodal/` 的 processing/processor.py + cache/{base,lru,shm}.py + hasher.py + encoder_budget.py。调度侧接口是 scheduler 的 _try_schedule_encoder_inputs（L2111）与 EncoderCacheManager（5.2 已锚：check_and_update_cache L111 / compute_mm_encoder_budget L329）。

encoder 分离部署（vision encoder 独立单 GPU、cheaper GPU 推荐）是 DeepInfer 的官方口径，image patch 账、VE 负载均衡与 encoder cache 复用归 9.4 / 7.4——本章只交付 vLLM 侧执行件落点。一个跨特性的接缝在此登记：image hash 进 prefix caching（V1 blog 第 7 点），block hash 的 extra keys 含 mm / lora / prompt_embeds（`kv_cache_utils.py` generate_block_hash_extra_keys L627）——多模态与多 LoRA 都会改变前缀共享的边界，LoRA 侧的命题与账归 5.4 §5。

DBO（Dual Batch Overlap）是 vLLM 侧的**计算-通信重叠**，与 CPU-GPU 重叠（async scheduling，5.2）是两个机制。官方 `docs/design/dbo.md` 口径：DP + EP 部署下把 MoE 的 sparse all-to-all 与计算重叠；TP 无 EP 时做 all-reduce-overlap；实现是 batch 切两半 + 两 worker 线程 ping-pong，FusedMoEModularKernel 提供 yield points。执行件：`v1/worker/ubatching.py` UBatchContext L21 + `worker/gpu/ubatch_utils.py` UBatchRunner L308。

**命题 3**（DBO 理想加速上界 4/3）。设整批的计算段时间 $C$ 、通信段时间 $M$ ，两者可完全并行、半批等长且成本线性可分（各 $C/2$ 、 $M/2$ ），忽略干扰与同步开销，且**步间不重叠**（同步步骤语义）。则两半 ping-pong 的单步 makespan（口径：本步从启动到两半都完成的墙钟时间）为 $\max(C, M) + \min(C, M)/2$ ，相对串行 $C + M$ 的加速比不超过 $4/3$ ，且在 $C = M$ 时取到。

**证明**（路线：画时序求总长 → 写成 max/min 形 → 比值放缩，三步）。第一步，时序：半批 1 计算占 $[0, C/2]$ 、通信占 $[C/2, C/2 + M/2]$ ；半批 2 的计算在半批 1 通信开始时启动（重叠），占 $[C/2, C]$ ，其通信占 $[C, C + M/2]$ 。总长 $= C + M/2$ （ $C \ge M$ 时）或 $M + C/2$ （ $M > C$ 时），统一写 $\max(C,M) + \min(C,M)/2$ 。第二步，记 $x = \max$ 、 $y = \min$ ，加速比 $= (x+y)/(x + y/2) = 1 + (y/2)/(x + y/2)$ 。第三步， $x \ge y$ 故分母 $x + y/2 \ge 3y/2$ ，比值 $\le 1 + (y/2)/(3y/2) = 4/3$ ； $x = y$ 时取等。∎

代入 3.4 §7 的教学参数（ $C$ = 6 ms、 $M$ = 4 ms）：串行 10 ms，DBO = 6 + 2 = 8 ms，加速 1.25×； $C = M$ = 5 ms 时 7.5 ms，加速 1.333× = 上界。对照 3.4 式 (4) 的 NanoFlow 三资源流水上界（同参数 12/6 = 2×）：**两资源对半切的上界 4/3 低于三资源流水的 2×**——资源轴越多、可重叠的段越多，上界越高；DBO 只重叠计算与通信两轴，且只切两半。

前提失效方向三条（账面折扣）：其一，通信占 SM——DeepEP 的 dispatch / combine 占 SM（3.3 口径），与计算并发时互相干扰， $C/2$ 与 $M/2$ 的线性可分假设破产（3.4 §7 的干扰折扣同构）；其二，半批后 kernel 效率下降——小 batch 的 GEMM 占用率与 grouped GEMM padding 账（3.3 §4）都随 batch 缩小而恶化；其三，**步间重叠**——若与 async scheduling 的跨步流水组合（5.2 §8），本步通信可与下一步计算重叠，上界放宽到 $(C+M)/\max(C,M)$ ，那是另一笔账，本命题只计单步 makespan。所以 4/3 是同步步骤语义下的理想上界，实测须按 profile 折扣。

```
DBO ping-pong（C=6 ms、M=4 ms，半批各 C/2=3、M/2=2）
──────────────────────────────────────────────────────────
串行：   [── compute 6 ──][── comm 4 ──]                10 ms
DBO：
  半批1  [compute 3][comm 2]
  半批2           [compute 3][comm 2]
         0        3        5        6        8   (ms)
  计算轴 ████░░░░████░░░░░░░░  ← 半批2 计算与半批1 通信重叠
  通信轴 ░░░░████░░░░████░░░░
                                          总长 8 ms = max + min/2
──────────────────────────────────────────────────────────
```

与 SGLang 的 TBO / SBO（two-batch / single-batch overlap，`batch_overlap/`）同族——注意 SGLang 的 `batch_overlap/` **不是** overlap scheduler（后者是 event_loop_overlap + FutureMap 的 CPU-GPU 重叠），两机制的区分与 FutureMap 走读归 6.2；vLLM 侧的对应区分是 DBO（本节，计算-通信）vs AsyncScheduler（5.2 §8，CPU-GPU）。DeepInfer 的异步调度承 SGLang / NanoFlow 路线（doc L334 官方披露，3.4 §7 已引逐字），走读归 7.1。

## 9. 三引擎分布式对照基线

三个引擎的分布式差异落在哪几个轴上？答案是配置面、通信路径、编排粒度三处结构差。下表为 P6 / P7 / P8 立基线，各轴详展归对应章，此处只登记结构与锚点（对照纪律：recap + 指针，不重复展开）。

| 轴 | vLLM（本章） | SGLang（6.3 详展） | DeepInfer（7.x 详展） |
| --- | --- | --- | --- |
| 并行配置面 | 显式字段 config/parallel.py（§2） | 派生面 runtime_context / arg_groups，enable_dp_attention 由 attn_dp_size > 1 派生 | DEEPINFER_* 环境变量族（7.1） |
| TP 归约 | GroupCoordinator L427 + CustomAllreduce L120 / SymmMemCommunicator L29 | GroupCoordinator L271 + CustomAllreduce L40 / CustomAllReduceV2 L152 / TorchSymmMemCommunicator L34 | decode 不用 TP（官方披露，3.1 §5 已引；理由与账归 8.1） |
| EP all-to-all | all2all.py manager 族（DeepEP HT/LL/V2、Nixl、FlashInfer NVLink、Mori、MoonEP） | layers/moe/ + DeepEP dispatcher（MaybeTboDeepEPDispatcher） | DeepEP + Mega MoE（3.3 / 7.3） |
| DP attention | DPEngineCoreProc L2592 + DPCoordinator L23 + DPSyncState | DataParallelController L159 + LoadBalanceMethod 四值 | DP/EP 耦合 + Prefill/Decode LB（7.4） |
| EPLB | distributed/eplb/（EplbState L321） | eplb/（EPLBManager L29） | DeepSeek EPLB（7.4） |
| 弹性 EP | distributed/elastic_ep/（→ 8.5） | elastic_ep/（ElasticEPStateManager L132） | release 边界外（7.1） |
| 编译 | CompilationMode 四级 + CUDAGraphMode 五值（§6） | CompilationConfig L20 + CUDAPiecewiseBackend L49（piecewise 单路线） | TileLang / TileKernels 手写路线（3.4 / 7.3） |
| 重叠 | DBO（计算-通信，§8）+ AsyncScheduler（CPU-GPU，5.2） | TBO/SBO（计算-通信）+ event_loop_overlap / FutureMap（CPU-GPU，6.2） | async scheduling（doc L334 官方披露，7.1） |
| PD KV 接口 | KVConnectorBase_V1 + NixlPush/Pull + MooncakeConnector + MultiConnector | disaggregation/base/conn.py 四抽象 + mooncake / nixl backend | 3FS kvcache-store + GDR swap-in（2.4 / 8.4） |
| 权重更新 | sleep / wake_up + weight_transfer/ engines（5.4 §6） | checkpoint_engine + release/resume_memory_occupation（6.3） | → 9.3 |

PD KV 接口的基线细节（为 8.4 立，本章不展开传输账）：`distributed/kv_transfer/kv_connector/v1/base.py` 的 KVConnectorBase_V1 L182 / KVConnectorRole L140（scheduler / worker 角色）/ SupportsHMA L102；NIXL 后端 `nixl/connector.py` NixlBaseConnector L79 / **NixlPullConnector L374 / NixlPushConnector L415**（push / pull 双模式，分件 base_scheduler / base_worker / push_* / pull_*）；Mooncake 后端 `mooncake/mooncake_connector.py` MooncakeConnector L798 / MooncakeConnectorScheduler L1023 / MooncakeConnectorWorker L1377；**MultiConnector L142**（`multi_connector.py`）组合多个 connector——HiSparse 的 local connector 与 PD 的 remote connector 共存就靠它（§7）。另有 OffloadingConnector L56 / SimpleCPUOffloadConnector L81（KV offload 侧，5.4 §4 走读）与 hf3fs / lmcache / mooncake_store / moriio / umbp / flexkv 等后端。传输语义、带宽账与编排对照归 8.3 / 8.4。

对照表的三条论断（可判定，非清单复述）：其一，配置面差异是**显式 vs 派生**——vLLM 把度数当输入字段，SGLang 把度数当 world size 的分解结果，引用两引擎配置时不可互译（§2 坑注）。其二，通信路径差异是**多 manager vs 单 dispatcher**——vLLM 的 all2all.py 把八类后端做成平级 manager 由 auto 解析，SGLang 侧以 DeepEP dispatcher 为中心；这不是「谁支持得多」，而是选择权放在配置面还是运行时的结构差。其三，PD 接口差异是**connector 组合 vs bootstrap 协议**——vLLM 用 MultiConnector 把 local / remote / offload 三类 connector 叠起来，SGLang 用 base/conn.py 的四抽象加 bootstrap 撮合；两者的传输账同归 8.3，编排哲学不同。

## 总结

| 论断 | 公式 / 数值 / 锚点 | 回指 |
| --- | --- | --- |
| 进程数 $N_{\mathrm{proc}} = A + d + dpt + \mathbf{1}[d>1]$ ；tp=4 单节点 6、tp=2 dp=4 为 17 | 式 (1)，官方 arch_overview.md 表 | §1 |
| 执行链三层：Executor L55 → Worker L233 → GPUModelRunner L209；executor↔worker 走 shm MessageQueue L498，frontend↔EngineCore 走 ZMQ | §1 表 | §1、5.1 |
| RayExecutorV2 L239 继承 MultiprocExecutor；旧 ray_distributed_executor.py 不存在 | §1 坑注 | §1 |
| vLLM 并行度是显式字段（config/parallel.py）；SGLang 是派生面（attn_dp_size > 1） | §2 表 + 坑注 | §2、6.3 |
| 命题 1：serving 每层归约位点 2 处（training 4 次的一半），只有 g 触发；SP 换原语不换位点 | 命题 1 | §3、练习 2 |
| decode 消息 $M_{\mathrm{msg}} = 2Bd_{\mathrm{model}}$ ：B=64 → 512 KiB、B=1 → 8 KiB | 式 (2) | §3、练习 1 |
| $\tau_{\mathrm{TP}} = 2L(\tau_{\mathrm{lat}} + 2(t-1)/t \cdot M/\mathrm{BW})$ ：B=64 → 385 μs、B=1 → 321 μs；消息缩 64× 时间只降 16.7% | 式 (3) | §3、练习 2 |
| 三条归约路径：NCCL / CustomAllreduce L120 / SymmMemCommunicator L29，门函数 should_custom_ar L511 / should_use_symm_mem L130 | §4 表 | §4 |
| 自定义归约进图须 capture L476 / register_graph_buffers L489（指针固定前提） | §4 坑注 | §4、3.4 命题 2 |
| EP all-to-all 八类 manager（DeepEPV2All2AllManager L1222 为当前统一接口落点）；all2all_backend "auto" L332 | §4 表 | §4、3.3 |
| attention backend 选择 = 六项能力约束下的可行性筛选（get_attn_backend L111 / AttentionBackend L57）；block size 默认 64，SWA 256 为偏好覆盖、CPU_MLA 16 为 supports_block_size 硬约束 + 平台强制 | §5 两表 | §5、2.1 式 (7)、3.1 |
| 编译链：support_torch_compile L88 → split_graph L580（partition_rules）→ PiecewiseBackend L86 / RangeEntry L80 → CUDAGraphWrapper L144 / CUDAGraphEntry L127 | §6 表 + 图 | §6、3.4 |
| HiSparse = DSA/QSA 本地 offload，三角色分工 + ACTIVE_TAIL_PAGES=2；默认 hot buffer 2K 行，V4.1 口径驻留 0.78%、B=64 热区 2.68 GB vs 全量 343.6 GB | 式 (4)、命题 2 | §7、练习 3 |
| 命题 2：单步写入 ≤ 2b token 时 active tail 覆盖写入页，外借不相交；大 chunk prefill 是边界反例 | 命题 2 | §7 |
| kv_hints = orchestrator 版本化 KV 动作协议（KvHintAction / KvHintsEnvelope），Request L85 → connector L172 → ReqContext L109 | §7 | §7、5.4 §4 |
| 命题 3：DBO 两半 ping-pong 单步 makespan = max + min/2（步间不重叠前提），加速上界 4/3（C=M 取到）；C=6/M=4 → 1.25× | 命题 3 | §8、练习 3 |
| DBO（计算-通信）≠ AsyncScheduler（CPU-GPU）；SGLang 对应区分 TBO/SBO ≠ event_loop_overlap | §8 | §8、6.2 |
| 三引擎对照三论断：显式 vs 派生配置面、多 manager vs 单 dispatcher、connector 组合 vs bootstrap 协议 | §9 表 | §9、6.3 / 7.x / 8.4 |

## 练习

1.（计算 · [10]）进程与消息量代入：(a) 用式 (1) 算 $d$ = 1、 $p$ = 1、 $t$ = 8 与 $d$ = 4、 $p$ = 2、 $t$ = 2 的进程数（ $A = d$ ）；(b) 用式 (2) 算 $B$ = 64 与 $B$ = 1 的 $M_{\mathrm{msg}}$ （ $d_{\mathrm{model}}$ = 4096、bf16），并换算 KiB。校验：(a) 第一组与 §1 的 tp=4 例同构（ $t$ 翻倍则 worker 项翻倍）；(b) 512 KiB / 8 KiB，与 §3 一致。
2.（推导 · [20]）命题 1 与延迟主导：(a) 从 f/g 共轭定义独立证明 serving 每层归约位点为 2，指出哪一步用到「推理无反向图」；(b) 说明 sequence parallelism 为何换原语不换位点；(c) 用式 (3) 算 $B$ = 1 与 $B$ = 64 的 $\tau_{\mathrm{TP}}$ （ $t$ = 8、 $\tau_{\mathrm{lat}}$ = 5 μs、NVLink 900 GB/s），解释消息量缩 64 倍而时间只降 16.7% 的机制。校验：(c) 321 μs / 385 μs，与 §3 一致；(a) 结论应蕴含「embedding 的额外一次归约不在层计数内」。
3.（推导 · [30]）命题 3 证明闭环 + HiSparse 账：(a) 画两半 ping-pong 时序，独立推出单步 makespan $\max(C,M) + \min(C,M)/2$（步间不重叠前提）；(b) 证加速上界 4/3 且 $C = M$ 取等；(c) 给三条前提失效方向（通信占 SM、半批 kernel 效率、步间重叠）并说明各使上界往哪个方向偏；(d) 用式 (4) 算 V4.1 口径（ $K$ = 512、 $T$ = 131072、 $\kappa$ = 1024 字节、 $L$ = 40、 $B$ = 64）的驻留比与热区容量，并算全 miss 一步的换入时间（PCIe 64 GB/s，2.4 §1 口径）。校验：(b) $C$ = 6、 $M$ = 4 → 8 ms / 1.25×；(d) 0.78% / 2.68 GB / 0.33 ms，与 §7 一致。
4.（代码 · [30]）走读 selector 与编译链（vllm main，2026-11-23）：先写出预测——(a) get_attn_backend（L111）的缓存键来自哪个结构、选择每步发生还是每模型发生；(b) split_graph（L580）的切分点由哪个文件决定、切分点算子为什么必须排除在图外；(c) RangeEntry（L80）与 CUDAGraphEntry（L127）各按什么索引、两者能否互相替代。再对照 `v1/attention/selector.py`、`compilation/{backends,piecewise_backend,cuda_graph,partition_rules}.py` 验证。校验：(b) 答案应指向 partition_rules.py（74 行）与 3.4 命题 2 的动态 launch 参数前提；(c) 前者按 shape 区间、后者按捕获 size，与 §6 一致。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——vLLM 论文锚点（块管理与 PagedAttention 背景归 2.1）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——本章全部 file:line 锚点：executor/（Executor L55 / MultiprocExecutor L150 / RayExecutorV2 L239）、worker/（Worker L233 / GPUModelRunner L209 / execute_model L1865 / sample_tokens L2279 / EncoderRunner L36 / MMEncoderModelRunner L38 / UBatchContext L21 / UBatchRunner L308）、distributed/（GroupCoordinator L427 / CustomAllreduce L120 / SymmMemCommunicator L29 / MessageQueue L498 / all2all.py manager 族 / eplb/ / kv_connector/v1/）、config/（parallel.py / compilation.py L38 L165 / attention.py HiSparseConfig L21）、v1/attention/（selector.py L111 / backend.py L57 / backends 族）、compilation/（PiecewiseBackend L86 / split_graph L580 / CUDAGraphWrapper L144）、v1/hisparse/ 与 v1/kv_hints/；docs/design/{arch_overview,hisparse,dbo,cuda_graphs,optimization_levels}.md，2026-11-23 fresh tarball 核实
- [vLLM Team, 2025](../references.md)：vLLM V1: A Major Upgrade to vLLM's Core Architecture（官方 blog，2025-01-27），[C] 开放技术博客、非闭源（key: vllm-v1-blog）——TP 对称架构（worker 侧缓存 request state、每步只传 diff）、persistent batch、MLLM first-class（image hash 进 prefix caching、encoder cache）；alpha 限制清单已历史化（V0 已移除，5.1 漂移注）
- [Ye et al., 2025](../references.md)：FlashInfer，arXiv:2501.01005，[A]（key: flashinfer）——plan/run 与 BSR 格式的算法级源头（3.1 §4 已交付，本章 §5 只走读引擎调用级）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——对照基线：GroupCoordinator L271 / CustomAllreduce L40 / CustomAllReduceV2 L152 / TorchSymmMemCommunicator L34 / DataParallelController L159 / EPLBManager L29 / ElasticEPStateManager L132 / CompilationConfig L20 / CUDAPiecewiseBackend L49 / disaggregation/base/conn.py 四抽象；配置面派生（runtime_context derive_attn_tp_size L73 / get_parallel L1314），2026-11-23 核实（详展 6.3）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——对照基线：decode 不用 TP（3.1 §5 已引 FlashMLA deep-dive 口径）、async scheduling 承 SGLang / NanoFlow（L334，3.4 §7 已引逐字）、VE 分离与 Wide-EP（7.4 / 9.4 详展），2026-11-23 核实
- [Qin et al., 2024](../references.md)：Mooncake，arXiv:2407.00079，[A]（key: mooncake）——MooncakeConnector L798 的传输引擎源头（transfer engine 账归 2.4 / 8.3）
- [NIXL repo](../references.md)：ai-dynamo/nixl，教学参照（key: nixl）——NixlPullConnector L374 / NixlPushConnector L415 与 NixlEPAll2AllManager L441 的传输抽象源头（agent / descriptor 语义归 8.3）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 SXM HBM 3.35 TB/s、BF16 dense 989.5 TFLOPS、NVLink 900 GB/s（§3 带宽换算）
- 跨维度指引（纯文字）：training 5.1（f/g 共轭、每层 4 次归约、消息量 2bsh）、training 4.1（ring 账式 (2)(3)，符号 α→τ_hop、β→1/BW 翻译）、training 5.4（EP all-to-all 训练账）；models 4.1（MLA 低秩与吸收）、models 4.2（DSA indexer）；inference 1.2（步时账）、2.1（块表与 block size 式 (6)(7)）、2.3（KV 量化）、2.4（三层账与临界带宽、SGLang HiSparse）、3.1（算子算法级）、3.3（DeepEP / grouped GEMM）、3.4（融合 / launch 账 / piecewise 命题 2）、4.3（chunked prefill）、4.5（verify 的 γ+1 token）、5.1（架构与数据流）、5.2（调度器 / KV 管理器 / AsyncScheduler / EncoderCacheManager）、5.4（LoRA / sleep / sampling / kv_offload）、6.2（TBO/SBO 与 FutureMap 区分）、6.3（SGLang 分布式与 PD）、7.1 / 7.3 / 7.4（DeepInfer）、8.1（TP decode 延迟账与 DP attention 通信账）、8.3 / 8.4（KV 传输与 PD 系统）、8.5（弹性与容错）、9.3（权重更新）、9.4（多模态）；silicon 归 gpu P1–P6、通信硬件层归 gpu P7–P10（cluster 仅平台编排）
