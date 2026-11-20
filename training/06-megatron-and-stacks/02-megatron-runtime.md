# 6.2 Megatron 运行时：进程组、调度与 Distributed Optimizer（Megatron Runtime: Process Groups, Schedules, and the Distributed Optimizer）

> 本章回答三个问题：world size 怎样变成一棵进程组树、一次 train step 在调度层走哪些代码、Megatron 的 Distributed Optimizer 与 DeepSpeed ZeRO 是什么关系——顺带登记 dist-ckpt 的并行度无关保存（衔接 7.4） | 前置：5.1（TP 通信账）、5.2（SP/CP）、5.3（PP 气泡与 1F1B / 交错 schedule）、5.4（EP）、6.1（层栈并行原语代码）、4.2（ZeRO 逐阶账）、1.4（激活与重计算账）、1.5（梯度累加与 `no_sync`） | 锚点：NVIDIA/Megatron-LM repo（`megatron/core/parallel_state.py`、`megatron/core/pipeline_parallel/schedules.py`、`megatron/core/optimizer/`、`megatron/core/distributed/`、`megatron/core/dist_checkpointing/`——全部路径与函数名 2026-11-17 经 GitHub 核实）[A]、Megatron-LM 论文（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）、Megatron 集群调度论文（[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]）、ZeRO 论文（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]）、selective recompute 论文（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]）

## 1. rank 映射：world size 到进程组树

几千个 rank、五个并行度，哪块 GPU 进哪个通信组？Megatron 在启动时一次算清：把 $W$ 个 rank 排成一张正交并行网格，每种通信原语各属一组。先看 dense 侧。四个并行度——TP/CP/DP/PP，记为 $t, c, d, p$（本书符号约定，P4/P5 同口径）——满足整除关系

$$
W = t \cdot c \cdot d \cdot p \qquad (1)
$$

**符号**： $t$：tensor parallel 度； $c$：context parallel 度； $d$：data parallel 度； $p$：pipeline parallel 度。注意 $d$ 不是独立输入。`initialize_model_parallel`（`megatron/core/parallel_state.py`）先算 `model_size = t · c · p`（再乘 GTP-remat 轴，默认 1）；`world_size % model_size != 0` 就直接报错，否则取 `data_parallel_size = world_size // model_size`。EP 度 $e$ 不进式 (1) 这张 dense 网格——expert 层有另一张网格，下文的双 generator 机制会讲。

代入数字：world = 15360、t = 8、c = 1、p = 12 时，d = 15360 / 96 = 160。本章数字口径先立在这里：进程组示例统一用 world = 16 的网格（引自源码 docstring 的算例另行注明出处），显存账统一 bf16 混合精度、实例取 128 亿参数与 d = 64；练习题的口径随题面另给。

**order 与 stride**。`initialize_model_parallel` 有个 `order` 参数，默认 `"tp-cp-ep-dp-pp"`。语义一句话：order 串从左到右，对应 rank 坐标从低位到高位——**最左的并行维 stride 最小，组内 rank 最相邻；最右的 stride 最大**。默认 order 下，全局 rank 分解为

$$
r = i_{\mathrm{tp}} + t \cdot \left( i_{\mathrm{cp}} + c \cdot \left( i_{\mathrm{ep}} + e \cdot \left( i_{\mathrm{dp}} + d \cdot i_{\mathrm{pp}} \right) \right) \right) \qquad (2)
$$

**符号**： $r \in [0, W)$：全局 rank； $i_{\mathrm{tp}} \in [0, t)$ 、 $i_{\mathrm{cp}} \in [0, c)$ 、 $i_{\mathrm{ep}} \in [0, e)$ 、 $i_{\mathrm{dp}} \in [0, d)$ 、 $i_{\mathrm{pp}} \in [0, p)$：五个并行维坐标。

**命题**（stride 规则）：所谓 stride，是指同一轴上相邻坐标对应的全局 rank 间隔；默认 order 下，任一轴的 stride 等于它左侧各轴维度数之积（前缀积）。**证明**（要点）：生成函数 `generate_masked_orthogonal_rank_groups` 先用 `prefix_product` 求出各轴 stride，再用 `decompose` 按 stride 分解组号与组内序号，最后内积重组——逐步展开即得上式。正例见下文的 16 卡网格；改 order 后连续性失效的反例，留作练习 2。

同一个 order 串被两个 rank generator 复用。dense 侧 generator 拿 $e = 1$（坐标 $i_{\mathrm{ep}}$ 恒为 0，式 (2) 退化为四维嵌套）；expert 侧 generator 拿 $c = 1$。为什么 CP 与 EP 不同时展开？`RankGenerator.__init__` 里有一条硬断言 `ep == 1 or cp == 1`——同一张网格内两者互斥。

代入一组具体度数，把映射看活。取 $W = 16$、 $t = 2$、 $c = 1$、 $e = 1$、 $d = 2$、 $p = 4$，式 (2) 实例化成 $r = i_{\mathrm{tp}} + 2 \thinspace i_{\mathrm{dp}} + 4 \thinspace i_{\mathrm{pp}}$：

```
world = 16, t = 2, c = 1, e = 1, d = 2, p = 4   →   r = i_tp + 2·i_dp + 4·i_pp

              i_pp = 0      i_pp = 1      i_pp = 2      i_pp = 3
 i_dp = 0      0    1        4    5        8    9        12   13
 i_dp = 1      2    3        6    7       10   11        14   15
               └TP 组┘      （TP 组 = 相邻 rank 对，stride 1）

TP 组（stride 1，8 组）: {0,1} {2,3} … {14,15}            ← 通信最频，落节点内
DP 组（stride 2，8 组）: {0,2} {1,3} … {13,15}
PP 组（stride 4，4 组）: {0,4,8,12} {1,5,9,13} {2,6,10,14} {3,7,11,15}  ← stride 最大，跨节点
```

这个默认 order 不是任意约定，而是 5.5「TP 进节点内、PP 跨节点」组合规则的代码落地。原因按通信频率排。TP 每层两次 all-reduce（5.1 的账），频率最高、延迟最敏感，所以放在最小 stride 上；rank 连续分配时——第 $r$ 块 GPU 在第 $\lfloor r / 8 \rfloor$ 个节点—— $t \le 8$ 的 TP 组就整组落在同一节点的 NVLink 域内。PP 相反：它只传 stage 边界激活（5.3），通信量最小，于是放到最大 stride 上跨节点。想重排也行，改 `order` 参数（比如 `"tp-pp-dp"`）即可，进程组树会整体重算（练习 1、2 考这条性质）。

源码自带的两个 docstring 算例，可以把 stride 语义再验算一遍。第一个在 `generate_masked_orthogonal_rank_groups` 的 docstring 里：进制链是 global_rank = tp_rank + dp_rank × tp_size + pp_rank × tp_size × dp_size，配一个度数 2-3-4（ $W = 24$ ）的算例，两者都对应 order `"tp-dp-pp"`——DP 组为 $\lbrace 0, 2, 4 \rbrace$、 $\lbrace 1, 3, 5 \rbrace$、……、 $\lbrace 19, 21, 23 \rbrace$，DP 轴的 stride 恰为 $t = 2$ ，与式 (2) 逐项一致。

> **提点**：同一段 docstring 的 prose 两处把 order 写成 `"tp-pp-dp"`，与进制链和算例的 tp-dp-pp 口径矛盾——源码自身不一致。本章统一按算例与公式的 tp-dp-pp 口径引用，5.5 引用同一 docstring 时同口径。

第二个算例在 `initialize_model_parallel` 的 docstring 里：8 块 GPU、 $t = 4$、 $c = 2$ 时，CP 组为 $\lbrace 0, 4 \rbrace$、 $\lbrace 1, 5 \rbrace$、 $\lbrace 2, 6 \rbrace$、 $\lbrace 3, 7 \rbrace$。CP 轴 stride 为 $t = 4$ ，同一 CP 组的两个 rank 处理同一序列的前后两块。两个算例都是同一条规则的推论：order 最左轴 stride = 1，向右逐轴乘前缀积。

**`RankGenerator` 机制**（行号以 2026-11-17 的 main 分支为准，下同）。`RankGenerator`（`parallel_state.py`，类定义行 522）收下各维度数与 order 串，把 order 切成 token 序列并映射成度数列表。`get_ranks(token)`（行 589）对目标并行维做布尔 mask，调 `generate_masked_orthogonal_rank_groups`（行 301）。这个函数先用前缀积算出各轴 stride，再把「组号」按未选中轴分解、把「组内序号」按选中轴分解（内部函数 `decompose`，即式 (2) 的逆运算），最后内积重组出每个组的 rank 列表。生成的 rank 集合随后逐个 `create_group` 建成 NCCL 进程组。用同一机制建出的组包括：TP 组（token `"tp"`）、CP 组（`"cp"`，另支持 hierarchical CP 子组）、DP 组（`"dp"`）、PP 组（`"pp"`）、model-parallel 组（`"tp-pp"`，即 TP×PP 复合域）、TP×DP 复合组等。

DP 为什么有两组？`_DATA_PARALLEL_GROUP`（`"dp"`）与 `_DATA_PARALLEL_GROUP_WITH_CP`（`"dp-cp"`）分工不同。CP 组内各 rank 处理同一条序列的不同分块、持有相同权重，权重梯度的归约域因此必须把 CP 并进来——`initialize_model_parallel` docstring 的原话是 "we piggyback GPUs of context parallelism on data parallel group for weight gradient all-reduce"。落到用法：数据加载按纯 dp 口径分片（同一 CP 组读同一样本的不同块），梯度归约按 dp-cp 口径走。

还有两条按 PP 组派生的组。一条是 embedding group：每条 PP 组取首、末两个 stage 的 rank（`default_embedding_ranks`，行 615）。tied embedding 的输入表住首 stage、输出 head 共享同一权重住末 stage，两处梯度需要跨首末同步，`schedules.py` 的 `finish_embedding_wgrad_compute` 执行这次归约。position embedding group 同理，只取首 stage。MoE 侧另有一整组 expert 组（EP 组、expert-DP 组等），由 expert generator 生成；代码断言两个 generator 的 PP 组必须逐组相同，保证 dense 与 expert 层的流水线切分一致。

除 NCCL 组外，DP 两口径还各建一份 gloo 镜像（`create_gloo_process_groups`，默认开）：gloo 组供 CPU 侧通信（checkpoint 元数据交换等），NCCL 组供 GPU 集合通信。至此，一次初始化建出的组清单（不含 expert 侧与 hierarchical CP 子组）是：TP、CP、DP、DP-with-CP、TP×DP、PP、model-parallel（TP×PP）、embedding、position embedding。每 rank 在每张组里恰好出现一次；accessor（`get_tensor_model_parallel_group()` 等）返回本 rank 所属的那一个。

最后一条现状注记。`parallel_state.py` 的模块 docstring 声明：这套「全局单网格 + 全局变量存组」正在向显式传组迁移，新机制是 `megatron/core/process_groups_config.py` 的 `ProcessGroupCollection`，新特性必须显式传入进程组。本节描述的全局网格仍是默认训练路径，也是所有 accessor（`get_tensor_model_parallel_group()` 等）的底层事实源。

## 2. 一次 train step 的数据流：`get_forward_backward_func` 走读

一个 train step 在代码里长什么样？入口固定在 `megatron/training/training.py` 的 `train_step`（行 2986），四步成链：先清零梯度缓冲（`model_chunk.zero_grad_buffer()` 与 `optimizer.zero_grad()`），再调 `get_forward_backward_func()`（`megatron/core/pipeline_parallel/schedules.py` 行 53）拿到本配置对应的 schedule 函数，然后跑完全部 micro-batch 的前向反向，最后 `optimizer.step()`。分支开关写在 `get_forward_backward_func` 末尾（行 161-167），共三分支：

- $p = 1$：`forward_backward_no_pipelining`（行 735）——无 PP 形态，逐 micro-batch 前向反向；
- $p > 1$ 且无 virtual stage：`forward_backward_pipelining_without_interleaving`（行 2169）——即 5.3 推导过的非交错 1F1B；
- $p > 1$ 且有 virtual stage：`forward_backward_pipelining_with_interleaving`（行 1013）——交错 1F1B，气泡再除 $v$ 倍的那套 schedule（[Narayanan et al., 2021](../references.md)，arXiv:2104.04473，[A]；算法账归 5.3）。

**前向段**。`forward_step`（schedules.py 行 404）做三件事：从 data_iterator 取一个 micro-batch，跑模型，返回输出张量与 loss 闭包。PP 非首 stage 的输入先经 p2p 收上一 stage 的激活。`get_forward_backward_func` 的 docstring 里有一条形状注记：开 SP 时张量的序列维按原长除以 $t$ 传入——SP 的序列切分在调度入口就体现在形状里（机制归 5.2）。

层内的集合通信——TP 的 f/g 恒等算子、SP 区段的 all-gather / reduce-scatter、CP 的 ring 收发、EP 的 all-to-all——全部发生在各 transformer 层内部，机制与通信账归 6.1 与 5.1-5.4。调度层要管的只有 PP 的跨 stage 传递，它由 `P2PCommunicator`（`megatron/core/pipeline_parallel/p2p_communication.py` 行 148）的复合算子承担。1F1B 稳态里，每个 micro-batch 典型地调一次 `send_forward_recv_backward`：把本 stage 的前向输出发往下一 stage，同时收下一 stage 回传的梯度。warmup / cooldown 段则用 `recv_forward`、`send_forward` 等单方向算子。

交错分支多一层 model chunk：每 rank 持有 $v$ 个不连续的层段， `(microbatch_id, model_chunk_id)` 的交替次序由 `get_schedule_table`（行 983）预生成，warmup 段长度与气泡账随之改变（精确推导归 5.3，此处只登记代码位置）。

**反向段**。`backward_step`（行 551）经 `custom_backward`（行 204）把 output_tensor_grad 喂进 autograd。权重梯度不进 `.grad` 字段，而是累加进一块连续大缓冲。机制在 DDP 包装层：`megatron/core/distributed/distributed_data_parallel.py` 的 `DistributedDataParallel`（行 88）把每个参数的 `main_grad` 注册成梯度缓冲里的 view（view 即共享底层存储的张量视图，不另占显存）。某 bucket 内全部参数的 wgrad 就绪后，反向 hook 触发 `start_grad_sync`；distributed optimizer 开启时，这就是 per-bucket 的 reduce-scatter（`megatron/core/distributed/param_and_grad_buffer.py` 行 662，底层为 `reduce_scatter_tensor`），`overlap_grad_reduce` 开启时它与后续反向计算重叠。

梯度累加的中段怎么办？`no_sync_func`（schedules.py 行 2272 起的 disable / enable 对）把同步整段关掉，最后一个 micro-batch 才放行——1.5 的通信 ÷ $m$ 机制在此落地。schedule 返回后，`finish_grad_sync` 等全部 bucket 的归约落地。

流水线收尾处还有一笔容易漏掉的跨 stage 同步：tied embedding 的 wgrad。首 stage 的输入 embedding 与末 stage 的输出 head 共享同一权重，两处 wgrad 在各自 stage 的反向里先后算出，时机错开，必须合流。schedule 在 cooldown 段经 `finalize_model_grads_func` 路径调 `clear_embedding_activation_buffer`（行 888）与 `finish_embedding_wgrad_compute`（行 907），在 §1 的 embedding group 上把两份 wgrad 归约成一份——PP 组首末 rank 的派生组在这里真正被用到。

一次 step 的通信事件时序（ $p > 1$ 非交错、dist-opt 开、overlap 开）：

```
train_step（p > 1，非交错 1F1B，dist-opt 开，overlap 开）
│
├─ zero_grad_buffer / optimizer.zero_grad
│
├─ get_forward_backward_func ─► forward_backward_pipelining_without_interleaving
│   │
│   │  warmup（前 p−1 个 micro-batch 只前向）：
│   │    recv_forward ─► forward_step ─► send_forward
│   │    层内通信：TP all-reduce ×2/层（5.1）＋ SP 区段 AG/RS（5.2）
│   │            ＋ CP ring（5.2）＋ EP all-to-all（5.4）
│   │
│   │  steady（1F1B 交替，每步一个前向 + 一个反向）：
│   │    前向后：send_forward_recv_backward（p2p 复合算子，P2PCommunicator）
│   │    反向：  custom_backward ─► wgrad 累加进 main_grad buffer
│   │            bucket 就绪 ─► 异步 reduce-scatter（overlap_grad_reduce）
│   │    累加中段：no_sync_func 禁同步（1.5），末个 micro-batch 才 finish
│   │
│   │  cooldown：排空剩余反向
│
├─ finish_grad_sync：全部 bucket 的 RS 收尾
│
└─ optimizer.step()：
     prepare_grads（main_grad 分片 → fp32 主梯度、unscale、NaN 检查）
     → clip_grad_norm → 内层 step（只更新本 rank 的分片）
     → _copy_main_params_to_param_buffer
     → start_param_sync（参数 buffer all-gather；overlap_param_gather 时
       与下一 step 前向重叠，DDP 前向 pre-hook 按 bucket 等待）
```

把时序图里的通信按域归堆。DP 域只有两笔：梯度 reduce-scatter（约 $N$ 元素）与更新后参数 all-gather（约 $N$ 元素），合计约 $2N$ 元素——与 4.1 的 DDP all-reduce 及 4.2 的 ZeRO-1 逐项同量。TP/CP/EP 的集合通信摊在层内，PP 的 p2p 摊在 stage 边界。一次 step 的通信总量叠加表归 5.5，本节只负责把事件序列与代码锚点对齐。

## 3. activation checkpointing：full / selective / uniform 的配置面

Megatron 的重计算策略就是 1.4 两档账的配置化。配置面只有三个字段，都在 `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，行 699 / 710 / 718 核实）：

- `recompute_granularity ∈ {full, selective}`。`full`：每个被标记的重算单元只存输入、弃全部中间激活，反向时整段重算。`selective`：只重算 attention 核心那部分 $O(T^{2})$ 激活（attention score 与 softmax 输出），保留廉价的逐元素激活——即 [Korthikanti et al., 2022](../references.md)（arXiv:2205.05198，[A]）的 selective recompute，1.4 已给全账（重算量 < 10% 换激活大头）。selective 开启时下面两个字段不适用。
- `recompute_method ∈ {uniform, block}` + `recompute_num_layers`（granularity 为 full 时生效）。`uniform`：把本 stage 的层每 `recompute_num_layers` 层切成一个重算单元（单元内层数越省激活、重算越多——1.4 的 $\sqrt{L}$ 权衡在本 stage 层数上重演）；`block`：只对前 `recompute_num_layers` 层整段重算，其余层的激活全量保留。

PP 下每个 stage 只持有自己的层，重计算按 stage 独立配置与执行（1.4 的 PP 交互口径）。

> **坑**：`parallel_state.py` 里的 `gtp_remat` 轴名字也带 "remat"，但它指 weight rematerialization——沿 out_features 维切权重、每次前向反向用异步 all-gather 重新物化权重——不是 activation recomputation。读源码时把两者混淆会错配并行轴语义；该轴 docstring 有显式声明（"remat here is NOT activation recomputation"），以其为准。本节的激活重计算与它无关。

## 4. Distributed Optimizer：Megatron 的 ZeRO 同族实现

4.2 推导的 ZeRO 分阶，在 Megatron 里的对应物叫 `DistributedOptimizer`（`megatron/core/optimizer/distrib_optimizer.py` 行 202；类 docstring 自述 "Optimizer that shards state across data-parallel ranks"），外加一层 DDP 缓冲。官方文档（repo 内 `docs/user-guide/features/dist_optimizer.md`）把它直接声明为 ZeRO 论文分片思想在 DP 域的落地（[Rajbhandari et al., 2020](../references.md)，arXiv:1910.02054，[A]）。一个 step 拆成五段看（文档 Data Flow 节口径 + 代码锚点）：

1. **梯度缓冲与分片映射**。每个 model chunk 的参数摊平进一块连续梯度缓冲（`_ParamAndGradBuffer`，param_and_grad_buffer.py）。缓冲先 padding 到 $d$ 的整数倍，再按连续区间均分，rank $r$ own 第 $r$ 段。`_build_model_gbuf_param_range_map`（distrib_optimizer.py 行 281）为每个参数建四重 Range：gbuf_world（参数在全缓冲的区间）、gbuf_world_in_bucket（bucket 内区间）、gbuf_local（本 rank 区间内的段）、param（参数自身的哪一片归本 rank）。注意**切分不尊重参数边界**：一个参数可以被相邻两个 DP rank 各 own 一段，optimizer 的一切操作都走缓冲 view（该方法 docstring 原文："each created range references a shard (or subset) of the full parameter"）。
2. **反向 + per-bucket reduce-scatter**。wgrad 产出即累加进缓冲，bucket 就绪即异步 RS（§2 已走读）。
3. **RS 完成态**。每个 rank 只有自己区间内的梯度是归约完毕的，区间外的内容作废（官方文档的 16 元素示例原话：rank 0 拿到元素 [0:4]，"remaining 12 elements are garbage"）。
4. **分片更新**。`optimizer.step()`（`megatron/core/optimizer/optimizer.py` 的 `MixedPrecisionOptimizer.step`，行 1076）→ `prepare_grads` 把本 rank 区间的梯度 cast 成 fp32 main grad（`_copy_model_grads_to_main_grads`）→ unscale 与 NaN 检查 → clip → 内层 optimizer 只在本 rank 的 fp32 master 分片上更新——fp32 master 与两阶矩估计都只有 $12N / d$ 量级。
5. **参数回写 + all-gather**。`_copy_main_params_to_param_buffer`（distrib_optimizer.py 行 3305）把更新后的 fp32 分片 cast 回 bf16、写进参数缓冲的本 rank 区间；`start_param_sync`（param_and_grad_buffer.py 行 441）对参数缓冲发 all-gather（`all_gather_into_tensor`）。模块里的参数本来就是缓冲的 view，AG 落地后下一 step 的前向直接可读；`overlap_param_gather` 开启时这次 AG 与下一 step 的前向重叠，DDP 的前向 pre-hook 按 bucket 等待参数就绪。

**显存账**。官方文档给出的每参数字节表（bytes/参数， $d$ 为 DP 度）：

| 配置 | 非分布式 optimizer | DistributedOptimizer |
| --- | --- | --- |
| fp16 参数 + fp16 梯度 | 20 | $4 + 16/d$ |
| bf16 参数 + fp32 梯度 | 18 | $6 + 12/d$ |
| fp32 参数 + fp32 梯度 | 16 | $8 + 8/d$ |

非分布式列 = 参数 + 梯度 + fp32 master + $m$ + $v$ 全量复制（如 18 = 2 + 4 + 4 + 4 + 4）；分布式列只把优化器侧状态除以 $d$ ，参数与梯度缓冲保持全量常驻。换成 bf16 参数 + bf16 梯度口径即 $4N + 12N/d$ ——与 4.2 式 (1) 的 ZeRO-1 账逐项相同。代入数字（128 亿参数、d = 64，bf16 参数 + fp32 梯度行）：非分布式每 rank 为 18 × 128 = 2304 亿字节，dist-opt 降到 (6 + 12/64) × 128 = 792 亿字节。

**通信账**。先立命题再解释。**命题**（通信同量）：dist-opt 每 rank 每 step 的 DP 域通信与 DP all-reduce 及 ZeRO-1 严格相等——梯度 RS（约 $N$ 元素）加参数 AG（约 $N$ 元素），每 rank 约 $2N$ 元素。**证明**（要点）：ring 下 all-reduce 本来就是 RS + AG 两段，dist-opt 把这两段分别落在梯度归约与参数同步两个本就需要的事件上，通信模式一字未改；4.2 §2 的同构论证逐字适用。开关语义写在 `DistributedDataParallelConfig`（`megatron/core/distributed/distributed_data_parallel_config.py`）的字段注释里：`use_distributed_optimizer` 开 = 梯度走 reduce-scatter、参数走 all-gather，关 = 梯度 all-reduce；`bucket_size` 默认 `max(4×10^7, 10^6 × dp_size)` 个参数。分片粒度还有一个旋钮：`num_distributed_optimizer_instances` 把 DP 域再切成若干独立 dist-opt 实例做部分分片（`initialize_model_parallel` 行 668 起，要求 DP 域大小被它整除）。

**与 DeepSpeed ZeRO 的对照**（4.2 回链，逐维度对齐）：

| 维度 | Megatron dist-opt | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | --- | --- | --- | --- |
| 优化器状态（master + $m$ + $v$ ） | ÷ $d$ | ÷ $d$ | ÷ $d$ | ÷ $d$ |
| 梯度归约通信 | per-bucket RS | RS | RS | RS |
| 梯度显存 | 全量缓冲常驻 | 全量 | ÷ $d$（归约后即弃） | ÷ $d$ |
| 参数显存 | 全量常驻 | 全量 | 全量 | ÷ $d$（用时 gather） |
| 参数同步 | step 后一次 AG | step 后一次 AG | step 后一次 AG | 无（下步重新 gather） |
| 每 step DP 域通信 | ≈ $2N$ 元素 | ≈ $2N$ | ≈ $2N$ | ≈ $3N$（1.5× 上限，4.2 式 (3)） |

读表。Megatron dist-opt 的显存口径等于 ZeRO-1，通信模式（per-bucket RS 流水）与 ZeRO-2 同形，参数维完全不动。它没有 ZeRO-2 的梯度显存切分——全量缓冲常驻是 `main_grad` view 机制的代价；也不需要 ZeRO-3 的反向二次参数 gather（4.2 §4.1）——参数根本没有被切走。为什么停在 ZeRO-1 族？Megatron 的显存杠杆在模型并行维：TP/PP/EP 先把参数与激活切掉（5.1-5.5），DP 域只负责摊薄优化器状态。要 ZeRO-3 语义，请走 FSDP 系（4.3）或 DeepSpeed 系（4.2 §4）的栈——此句为本书分析读法。

分片之后，逐 step 的监控统计多一笔归约。梯度范数与零元素计数本是全局量——2.4 的 global norm clipping 定义在全体参数拼成的向量上——而 dist-opt 下每个 rank 只持自己分片的归约梯度。归约域的口径写在 `DistributedOptimizer.get_grad_stats_parallel_group`（distrib_optimizer.py 行 1111）的 docstring 里：dist-opt 下梯度统计在 distributed optimizer 实例的全部 rank 上归约（非分布式时只需在 model-parallel 组上归约）。这笔通信是标量级小量，但它是 clip 阈值语义正确的前提：范数必须在全 DP 域归约，否则每个 rank 拿局部范数裁剪，有效阈值会被抬高 $\sqrt{d}$ 倍量级（2.4 练习可推）。

## 5. dist-ckpt：并行度无关的保存（前瞻 7.4）

Megatron 的 checkpoint 层是 `megatron/core/dist_checkpointing/`（repo README 的模块注释："Distributed checkpoint saving, loading, and resharding"）。核心抽象是 `ShardedTensor`（`dist_checkpointing/mapping.py` 行 64）：每个张量以 key + 全局 shape + 全局 offset + 本地分片的元数据登记。`serialization.py` 的 `load`（行 78）与 `save`（行 364）按这套元数据写读，**保存时的并行网格与加载时的网格解耦**——换并行度续训（resharding）就是按新网格重新规划每个 rank 读哪些分片。

optimizer 侧，`DistributedOptimizer` 声明了 fully-reshardable 格式集合（`checkpoint_fully_reshardable_formats`，行 216：`fully_reshardable` / `fully_sharded_model_space` / `fsdp_dtensor`）。这里有个细节：§4 的分片按缓冲连续区间、可切断参数，optimizer state 的保存因此必须携带区间映射元数据（`sharded_state_dict`，distrib_optimizer.py 行 1774）——dist-ckpt 的元数据层正是为这类分片设计的。读写后端可插拔：`get_default_save_sharded_strategy(backend="torch_dist")`（serialization.py 行 67）给出默认后端；加载前可用 `load_sharded_metadata` 只读元数据检视 checkpoint 内容，不搬数据。checkpoint 的完整内容清单（RNG / dataloader cursor / schedule 步数）、异步写盘账与 MTBF 口径归 7.4，本节只立机制位置。

## 总结

| 主题 | 结论 | 代码锚点 / 出处 |
| --- | --- | --- |
| rank 映射 | order 默认 `"tp-cp-ep-dp-pp"`，最左轴 stride 最小；式 (1)(2) | `parallel_state.py`：`RankGenerator`、`generate_masked_orthogonal_rank_groups` |
| 进程组树 | dense 网格 $W = t c d p$ 与 expert 网格 $W = t_e e p d_e$ 双 generator；EP×CP 互斥断言；DP 分 dp 与 dp-cp 两口径；embedding group = PP 首末 stage（练习 1、2） | `initialize_model_parallel`、`default_embedding_ranks` |
| 调度入口 | 三分支：no-pipelining / 非交错 1F1B / 交错 1F1B（vp） | `schedules.py` `get_forward_backward_func` |
| train step 通信 | DP 域两笔：梯度 per-bucket RS + 参数 AG ≈ $2N$ 元素；PP 走 `P2PCommunicator` 复合算子；累加中段 `no_sync_func`（练习 3(c)） | `distributed_data_parallel.py`、`param_and_grad_buffer.py` |
| 重计算 | `recompute_granularity × recompute_method × recompute_num_layers`；selective = selective recompute 论文 | `transformer_config.py`；arXiv:2205.05198；1.4 回链 |
| Distributed Optimizer | 显存 = ZeRO-1 口径（bf16+fp32 梯度： $6 + 12/d$ bytes/参数）；通信 = ZeRO-2 同形（per-bucket RS）；参数不切、无 ZeRO-3 二次 gather | `distrib_optimizer.py`；repo `docs/user-guide/features/dist_optimizer.md`；4.2 对照 |
| dist-ckpt | `ShardedTensor` 元数据登记，保存/加载网格解耦（resharding） | `dist_checkpointing/mapping.py`、`serialization.py`；7.4 前瞻 |

## 练习

1.（推导 · [20]）order 取默认 `"tp-cp-ep-dp-pp"`， $W = 32$、 $t = 4$、 $c = 2$、 $d = 2$、 $p = 2$（ $e = 1$）：(a) 实例化式 (2) 并写出五个轴各自的 stride；(b) 求 rank 23 的五维坐标（提示：按 stride 从高位到低位逐次取模）；(c) 列出 rank 23 所在的 TP、CP、DP、PP 组的全部成员；(d) 若每节点 8 块 GPU 且 rank 按节点连续分配，rank 23 的哪些组整组落在单节点内？校验：(c) 中每张组的「组大小 × 组数」都等于 32。

2.（推导 · [30]）证明正文 stride 规则命题的推论：默认 order 下 TP 组必为连续 rank 区间 $\lbrace k t, k t + 1, \ldots, k t + t - 1 \rbrace$（ $k \in [0, W/t)$ ）；进一步，若 $t$ 整除每节点 GPU 数 $g$ 且 rank 按节点连续分配，则每个 TP 组整组落在同一节点。改 order 为 `"dp-tp-pp"`（ $W = 16$、 $t = 2$、 $d = 2$、 $p = 4$ ）后该性质是否仍成立？给出反例或证明。校验：改 order 后把全部 TP 组完整写出再判定；证明按基例、保持、终止三段写全。

3.（计算 · [30]）bf16 参数 + fp32 梯度、 $N = 7 \times 10^{9}$、 $d = 64$：(a) 按 §4 官方表算 dist-opt 每 rank 模型状态字节数（GB，SI 口径）；(b) 同条件改用 bf16 梯度重算，并与 4.2 式 (1) 的 ZeRO-1 逐项对照，说明两口径的差异项是什么；(c) 每 step 每 rank 的 DP 域通信字节数（bf16 元素口径），与纯 DDP（4.1）对照。先写出你对 (b) 两口径倍数差的预测再计算。校验：(b) 两口径之差应恰好等于梯度 dtype 的字节差乘以参数量；(c) 应等于「2 × 参数量 × 2 字节」。

4.（分析 · [30]）Megatron dist-opt 不需要 ZeRO-3 的「反向二次参数 gather」（4.2 §4.1 时序图里的第二趟 AG）。说明它省掉这趟 gather 的机制原因，并指出代价是哪一笔显存没有随 $d$ 缩小；在什么模型规模与卡数组合下，这笔代价会迫使使用者离开 dist-opt 转向 ZeRO-3 / FSDP 系栈？（结合 4.2 §7 决策表回答。）校验：临界规模可用正文显存表的三行口径外推复核。

## 参考

- [Shoeybi et al., 2019](../references.md)：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，arXiv:1909.08053，[A]（key: megatron-lm）——NVIDIA/Megatron-LM repo：`megatron/core/parallel_state.py`（`initialize_model_parallel`、`RankGenerator`、`generate_masked_orthogonal_rank_groups`、`default_embedding_ranks`）、`megatron/core/pipeline_parallel/schedules.py`（`get_forward_backward_func`、`forward_backward_pipelining_without_interleaving` 等三分支）、`megatron/core/pipeline_parallel/p2p_communication.py`（`P2PCommunicator`）、`megatron/core/distributed/distributed_data_parallel.py` 与 `param_and_grad_buffer.py`（`start_grad_sync` / `start_param_sync`）、`megatron/core/optimizer/distrib_optimizer.py`（`DistributedOptimizer`、`_build_model_gbuf_param_range_map`）、`megatron/core/optimizer/optimizer.py`（`MixedPrecisionOptimizer.step`）、`megatron/core/transformer/transformer_config.py`（recompute 三字段）、`megatron/core/dist_checkpointing/`（`mapping.py` 的 `ShardedTensor`、`serialization.py` 的 `save`/`load`）、`docs/user-guide/features/dist_optimizer.md`（显存表与 Data Flow）；路径与函数名核实于 2026-11-17
- [Narayanan et al., 2021](../references.md)：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM，arXiv:2104.04473，[A]（key: megatron-cluster）——交错 1F1B schedule 的论文出处（算法账归 5.3）
- [Rajbhandari et al., 2020](../references.md)：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，arXiv:1910.02054，[A]（key: zero）——§4 对照表的 ZeRO-1/2/3 各阶口径（推导归 4.2）
- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——selective recompute 出处（账归 1.4）
