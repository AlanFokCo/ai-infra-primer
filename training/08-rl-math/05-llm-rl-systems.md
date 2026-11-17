# 8.5 LLM-RL 的系统实现（Systems for LLM-RL: Four Models, Two Engines, and Weight Sync）

> 8.1-8.4 交付的是算法：策略梯度、GAE、PPO clip、GRPO 组基线。把其中任何一个跑在 70B 模型上，立刻撞上三笔系统账——四个模型同驻显存的 bytes 账、训练引擎与推理引擎的拆分、每轮迭代把新权重推回 rollout 侧的同步成本。本章逐笔清算，并以 veRL 为主走读（HybridFlow 论文的 hybrid 控制模型与 3D-HybridEngine resharding）、OpenRLHF 为对照 | 前置：8.3（PPO 目标与 clip）、8.4（GRPO / DAPO / GSPO）、4.2（ZeRO 显存账）、6.2（Megatron 运行时）、6.3（训练栈定位） | 锚点：veRL / HybridFlow 论文 + verl-project/verl repo（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]）、OpenRLHF 论文 + OpenRLHF/OpenRLHF repo（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]）、vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、Megatron-LM（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）、FSDP（[Zhao et al., 2023](../references.md)，arXiv:2304.11277，[A]）、TRL（[HuggingFace, huggingface/trl repo](../references.md)，教学参照）

## 1. 一次 RL iteration 的数据流：为什么这是系统问题

先把瓶颈位置钉死：OpenRLHF 论文的 profile（LLaMA2-7B + A100，其 §3.2）显示 PPO 的样本生成阶段占整体训练时间的 80%——自回归解码是 memory-bound 的，吞吐靠放大推理 batch 买，而四个模型共驻同卡的框架（TRL、DeepSpeed-Chat）恰恰没有显存余量放大 batch（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]）。LLM-RL 的系统问题由此而来：它不是「一个模型的训练」，而是「四个模型 + 两套引擎 + 一条每轮必走的权重同步边」。

一次 RL iteration 的完整数据流（PPO 全配置；GRPO 砍掉 critic 一支，8.4）：

```
                ┌──────────────── 一次 RL iteration ────────────────┐
                │                                                    │
 prompt batch ─►│ ① rollout：actor（推理引擎，旧权重 θ_old）采样      │
                │     GRPO 时每 prompt 采 G 条（8.4）                 │
                │ ② 打分：ref 前向取 logp、reward 模型 / 规则验证器    │
                │     给 r_t，逐 token 算 KL(ref ‖ θ_old)             │
                │ ③ advantage 聚合（GAE 归 8.2；组基线归 8.4）         │
                │ ④ 训练引擎多 epoch 更新 actor（与 critic）→ θ        │
                │ ⑤ 权重同步：训练侧 θ → rollout 引擎（每轮一次）      │
                └────────────────────────────────────────────────────┘
```

五个环节里只有 ④ 是常规训练（其并行与显存账归 4.x/5.x/6.x）；①⑤ 涉及的第二套引擎、②③ 的多模型编排、以及它们之间的数据搬运，是 P1-P7 没有的新账。本章结构：§2 算四模型显存，§3 拆引擎，§4 算权重同步，§5 处理同步与数值引入的 off-policy，§6-§7 走读 veRL 与 OpenRLHF，§8 落 KL 与长 CoT 的工程位置。

## 2. 四模型同驻显存账

8.3 的 PPO 需要四个模型同场：actor、reference、reward、critic。逐一给出常驻字节数——训练态模型按 bf16 参数 + bf16 梯度 + fp32 master + fp32 两阶矩的 $16N$ 口径（分解推导归 models 1.2 式 (8) 与 4.2，此处只引用结论），纯推理模型只需 bf16 权重 $2N$ 加推理侧的 KV cache：

| 模型 | 状态性质 | 常驻 bytes |
| --- | --- | --- |
| actor | 训练（参数 + 梯度 + fp32 master + $m, v$） | $16 N_a$ |
| critic | 训练 | $16 N_c$ |
| reference | 推理（仅前向取 logp） | $2 N_a$ |
| reward | 推理（仅前向打分） | $2 N_r$ |
| rollout actor | 推理引擎内副本 + KV cache | $2 N_a$ + KV |

合计（actor 与 ref 同构取 $N_a$）：

$$
\mathrm{Mem}_{\mathrm{PPO}} = \underbrace{16 N_a}_{\text{actor 训练}} + \underbrace{16 N_c}_{\text{critic 训练}} + \underbrace{2 N_a}_{\text{ref}} + \underbrace{2 N_r}_{\text{reward}} + \underbrace{2 N_a}_{\text{rollout 副本}} + \mathrm{KV} \qquad (1)
$$

**符号**： $N_a$：actor 参数量； $N_c$：critic 参数量； $N_r$：reward 模型参数量； $\mathrm{KV}$：rollout 引擎的 KV cache（随并发序列数与长度线性涨，归 inference 维度的账）。四模型同尺寸 $N$ 时式 (1) 简化为 $38N + \mathrm{KV}$ 。代数字：全 7B 配置 $N = 7 \times 10^{9}$ ，模型状态合计 $38 \times 7 = 266$ GB——单卡 80 GB 连一个训练态 actor（ $16 \times 7 = 112$ GB）都装不下，切分不是优化而是前提；全 70B 时总量 $2.66$ TB，对照 HybridFlow 论文测试床的 128 × 80 GB（约 10 TB 总量）可放，但每张卡放谁、何时用谁是放置问题（§3、§6）。

两个标准省钱手段直接改式 (1) 的系数。其一，训练态两模型过 ZeRO-1（4.2）：优化器状态 ÷ $d$ ，每模型从 $16N$ 降到 $4N + 12N/d$ ， $d = 64$ 时 actor 一侧从 $16N_a$ 降到约 $4.2 N_a$ 。其二，GRPO（8.4）去 critic：式 (1) 直接减 $16 N_c$ ；RLVR 场景 reward 换规则验证器（9.5 的口径）再减 $2 N_r$ ——全同尺寸下从 $38N$ 降到 $20N$ ，省 47%。代价不在显存而在 rollout：GRPO 每 prompt 采 $G$ 条，生成 token 数与 KV cache 同乘 $G$ ——式 (1) 省下的预算被搬到 $\mathrm{KV}$ 项与 ① 的时长里。这条交换是 8.4 组基线「免 critic」的系统侧价格。

## 3. rollout / train 引擎拆分

① 为什么不复用训练引擎做采样。自回归解码逐 token 串行、每步读全量权重，是 memory-bandwidth-bound 负载；推理引擎（vLLM）的 PagedAttention、连续批处理、KV cache 分页管理正是为这个负载形态做的——内部机制归 inference 维度，本章只登记接口事实。训练引擎（Megatron 归 6.x、FSDP 归 4.3）的 TP/PP 切分为训练吞吐优化，layer 间通信模式与 decode 完全不匹配；OpenRLHF 的 profile（§1 的 80%）说明生成阶段是主瓶颈，引擎选错直接拖垮整轮。

② 拆分后的形态（本章核心图）：

```
                  single-controller（编排进程：数据流顺序、多模型间数据搬运）
                  │  数据以 TensorDict / DataProto 形态在模型间流转
   ┌──────────────┼───────────────────────┐
   ▼              ▼                       ▼
┌────────────┐ ┌────────────┐     ┌─────────────────────┐
│ actor 训练  │ │ critic 训练 │     │ rollout 引擎组       │
│ Megatron / │ │ （PPO 才有） │     │ vLLM × n 实例        │
│ FSDP       │ │ 同左        │     │ 自身 TP_g 分片        │
│ TP×PP×DP   │ └────────────┘     │ 权重 2N_a + KV cache │
│ 训练态 16N │                      └──────────┬──────────┘
└─────┬──────┘                                 │
      │ ④ 更新 θ      ⑤ 权重同步（§4）          │ ① 用 θ_old 采样
      └────────────────────────────────────────►│
┌────────────┐ ┌────────────┐
│ ref 模型    │ │ reward 模型 │   ② 打分（前向 only）
│ 2N_a 推理   │ │ 2N_r 推理   │     RLVR 时换成规则验证器
└────────────┘ └────────────┘
   ③ advantage 聚合在编排侧完成（GAE 归 8.2；组基线归 8.4）
```

③ 拆分引入两个子问题。其一，**并行布局不一致**：训练侧按 $(t, p, d)$ 分片，rollout 侧按自己的 $(t_g, p_g)$ 分片，同一份权重在两侧的分片方案不同——权重同步因此不是一次简单广播，而是 resharding（§4 的账）。其二，**放置**：四模型与两引擎怎么映射到物理卡上。两种基本模式——**共卡时分复用**（colocate：rollout 与训练交替占同一批卡，空闲期把对方权重 offload 到 CPU）与**分卡专用**（separate：rollout 引擎常驻一批卡，训练常驻另一批）。HybridFlow 支持多种放置、由 Auto device mapping 搜索方案（其论文 Table 1 与 §6 口径），共卡形态下训练期把生成用权重卸载到 CPU（其论文 §7 的实现口径）；OpenRLHF 对 70B 以上模型默认四模型分卡放置、另提供 colocate 开关（其论文 §3.1 与 CLI 口径，§7 展开）。共卡省卡数、分卡省切换，选择由模型尺寸与集群规模决定（练习 4）。

## 4. 权重同步：每轮一次的 resharding 与广播

on-policy 语义要求 ① 用的 $\pi_{\theta_{\mathrm{old}}}$ 尽量接近当前 $\pi_\theta$ ——每轮 ④ 结束后把新权重推回 rollout 引擎。同步的字节数就是 actor 权重本体 $2 N_a$（bf16），于是同步时间的带宽下界为

$$
t_{\mathrm{sync}} \ge \frac{2 N_a}{B} \qquad (2)
$$

**符号**： $B$：同步路径的有效带宽（bytes/s）。式 (2) 只含带宽项：集合通信的延迟与拓扑因子在小消息下占优，GB 级消息的传输时间被带宽项主导，故式 (2) 为下界口径。代数字（带宽取 HybridFlow 测试床口径：节点内 NVLink 600 GB/s、节点间 200 Gb/s 约 25 GB/s，[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]，§8.1）：

| actor | 权重 bytes（bf16） | 600 GB/s 下界 | 25 GB/s 下界 |
| --- | --- | --- | --- |
| 7B | $1.4 \times 10^{10}$ | ≈ 0.023 s | ≈ 0.56 s |
| 70B | $1.4 \times 10^{11}$ | ≈ 0.23 s | ≈ 5.6 s |

读表：同步路径落在 NVLink 域内（rollout 与训练共卡或同节点）时，70B 的每轮同步是亚秒级；跨节点走 IB 时 70B 要 5.6 s 起步——这就是为什么共卡 / 同节点放置与同步路径的带宽是设计输入而不是事后调优项。第三条路径是**共享存储**：训练侧写 checkpoint 到 CPU 内存或磁盘、rollout 侧再加载，走 PCIe / 存储栈（单卡 GPU↔CPU 链路约 10-12 GB/s 的口径归 4.2 §5），70B 一次要十几秒级，只在两引擎不在同一 NCCL 域（跨集群、异构）时才值得。

**3D-HybridEngine：把 resharding 的通信与冗余同时压掉**。HybridFlow 论文的核心系统贡献落在权重转换这一步（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]，§5）。问题：训练布局 $(t, p, d)$ 下每卡持 $M / (tp)$ ，生成布局 $(t_g, p_g)$ 下每卡需 $M / (t_g p_g)$ ， $M$ 为 actor 权重 bytes。朴素做法（论文称 DeepSpeed-Chat 形态）是全 $N_g = t p d$ 卡 all-gather；HybridFlow-V 是在训练 TP×PP 组内 all-gather；HybridFlow 的做法是重排生成分组——生成 TP/PP 组按 $t/t_g$ 、 $p/p_g$ 为间隔取 rank，再把沿生成 TP/PP 维相邻的 rank 顺次编成 **micro-DP 组**，组大小

$$
d_g = \frac{t p}{t_g p_g} \qquad (3)
$$

这样每卡的生成分片恰为本 micro-DP 组内各卡训练分片之并：本地训练分片直接复用，只需在 micro-DP 组内 all-gather 补齐缺的部分。三种方案的转换开销对照（论文 Table 2，逐项照录）：

| 方案 | 转换通信量（每卡） | 峰值参数显存 | 冗余显存 |
| --- | --- | --- | --- |
| DeepSpeed-Chat 形态 | $\frac{t p d - 1}{t p d} M$ | $M$ | $\frac{1}{t p d} M$ |
| HybridFlow-V（朴素分组） | $\frac{t p - 1}{t p} M$ | $M$ | $\frac{1}{t p} M$ |
| HybridFlow（micro-DP 分组） | $\frac{d_g - 1}{t p} M$ | $\frac{1}{t_g p_g} M$ | $0$ |

**符号**： $t, p, d$：训练侧 TP/PP/DP 度（全书约定，4.x 同口径）； $t_g, p_g$：生成侧 TP/PP 度； $d_g$：micro-DP 组大小； $N_g = t p d$：actor 训练与生成共用的 GPU 总数（HybridFlow 论文原记号 $N_a$ ，与本章 §2 的参数量 $N_a$ 无关，此处换字母避歧）。生成侧网格是 $t_g \times p_g \times d_g \times d = N_g$ ——micro-DP 之外还套一层与训练 DP 同度的外层 DP（本书读法）。论文 §5.4 给出第三行通信量的等价写法 $\frac{t p - t_g p_g}{t_g p_g \thinspace t p} M$ ，与 $\frac{d_g - 1}{tp} M$ 按式 (3) 代入即等。

代论文 Figure 8 的 8 卡例：训练 $(t, p, d) = (4, 1, 2)$ 、生成 $(t_g, p_g) = (2, 1)$ ，则 $d_g = 2$ ：

```
8 GPU：训练 (t,p,d)=(4,1,2) → 生成 (t_g,p_g)=(2,1)，d_g=2

训练布局：TP 组 {G1..G4} {G5..G8}          每卡持 M/4
生成分组：TP_g 组 {G1,G3} {G2,G4} ...      每卡需 M/2
         micro-DP 组 {G1,G2} {G3,G4} {G5,G6} {G7,G8}
转换：G1 的生成分片 = G1、G2 训练分片之并
      → 组内一次 all-gather，各卡只收 (d_g−1)/(tp) · M = M/4
通信系数：DS-Chat 0.875M ／ HF-V 0.75M ／ HybridFlow 0.25M
冗余：    前两者部分卡需另存训练权重；HybridFlow 零冗余
```

组内 all-gather 还有一个并行红利：各 micro-DP 组的 all-gather 互不依赖、并发执行（论文 §5.3 口径）。峰值显存从全量 $M$ 降到生成分片大小 $M / (t_g p_g)$ ——「zero memory redundancy」的精确含义是：转换期间不需要为两套布局各留一份权重。

**代码落点**。OpenRLHF 的同步在 `openrlhf/trainer/ray/ppo_actor.py` 的 `broadcast_to_vllm`：逐参数走 `_model_update_group.broadcast(param.data, src=0, ...)` 的 NCCL 广播（或 `ray.util.collective` 后端），ZeRO-3 下先用 `deepspeed.zero.GatheredParameters` 把分片聚回 rank 0 再播；同文件另有 CUDA IPC 路径 `_handle_cuda_ipc`（`reduce_tensor` 产 IPC handle，rollout 侧 `update_weight_cuda_ipc` 重建张量，免去一次拷贝）。veRL 当前 main 分支把同步抽象成 `verl/checkpoint_engine/` 的多后端（`nccl_checkpoint_engine.py` / `nixl_checkpoint_engine.py` / `mooncake_checkpoint_engine.py` 等），rollout 侧 `verl/workers/rollout/vllm_rollout/vllm_rollout.py` 的 `update_weights` 按桶接收（`bucketed_weight_transfer.py` 走 IPC / 共享内存），并提供 `release()` / `resume()`——训练期让 vLLM `sleep` 释放 KV cache 与权重显存、rollout 前 `wake_up` 恢复，这是共卡模式下 KV cache 与训练显存错峰的机制（HybridFlow 论文 §7 的对应口径：训练期把生成权重 offload 到 CPU、生成完把 KV cache 卸回 CPU）。路径均核实于 2026-11-17 的 main 分支；两 repo 都在快速演化（veRL 的 checkpoint_engine 多后端、v1 trainer 均为论文之后的新增），引用以路径与函数名为准。

## 5. off-policy 校正：staleness 与引擎数值差

式 (2) 的账压出一个设计诱惑：降低同步频率、让 rollout 用旧权重多跑几轮，把 $t_{\mathrm{sync}}$ 摊薄。代价是数据分布漂移——rollout 由 $\pi_{\theta_{\mathrm{old}}}$ 采样、更新在 $\pi_\theta$ 上做，修正靠 importance ratio（8.3 的完整推导）：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}, \qquad \hat{r}_t = \min\lbrace r_t, \thinspace c \rbrace \qquad (4)
$$

**符号**： $s_t$：prompt 加已生成前缀； $a_t$：第 $t$ 个生成 token（8.1 的 token-level MDP 约定）； $c$：截断阈值。PPO clip 本身就是 ratio 落在 $1 \pm \epsilon$ 内的容忍区间（8.3）；staleness 一大，ratio 质量被推到 clip 区间外，有效梯度样本减少。截断 IS（ $`\hat{r}_t`$ ）用偏差换方差——ratio 的方差随连乘长度爆炸（8.1 的方差分析），截断把尾部压掉。

staleness 之外还有第二个、更隐蔽的 off-policy 来源：**同一份权重、两套引擎的数值差**。rollout 引擎与训练引擎的 kernel、batching、精度路径不同，同一个 $(s_t, a_t)$ 在两边的 logp 可以有可观差异——即使权重零时差，ratio 也带噪声。这不是理论担忧，veRL 的 `verl/trainer/ppo/rollout_corr_helper.py` 模块 docstring 把三类来源并列写明："Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)"（引擎实现差）、"Model update staleness"（版本错位）、泛化的分布漂移；该模块提供 token 级与序列级 IS 权重、按散度过滤的拒采样（rejection sampling）掩码、以及 KL / PPL / χ² / ESS 等 off-policy 诊断指标，在 `trainer_base.py` 与 actor worker 两侧接入（均 2026-11-17 核实）。

工程上先消能消的部分：veRL 的 `verl/trainer/ppo/v1/trainer_base.py` 里 `_compute_old_log_prob` 提供两种模式（其 docstring 口径）——bypass 模式直接用 rollout 引擎返回的 `rollout_log_probs` 当 $\pi_{\theta_{\mathrm{old}}}$（两策略口径： $\pi_{\mathrm{rollout}}$ 与 $\pi_\theta$ ）；decoupled 模式由训练引擎对 rollout 产出的 token 重算一遍 old logp，作为 PPO 更新的近端锚点（三策略口径： $\pi_{\mathrm{rollout}}$ 、 $\pi_{\mathrm{old}}$ 、 $\pi_\theta$ ），`rollout_log_probs` 仍保留给 rollout 校正用。重算让 PPO ratio 的分子分母出自同一套 kernel——引擎数值差从主更新的 ratio 里消掉，残余的真实采样分布与重算锚点之间的差交给截断 IS / 拒采样处理。序列级 ratio 的粒度选择（token 级 vs 序列级的方差对照）归 8.4 的 GSPO，本节只登记系统侧的接入点。

## 6. veRL 走读：hybrid 控制模型与文件地图

HybridFlow 论文的问题陈述先从控制模型二分（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]，§1-§2）：RLHF 数据流里每个节点是一个分布式 LLM 程序、边是多对多的数据广播。纯 single-controller（传统 RL 框架的形态）用一个中心进程指挥一切，节点内的分布式计算也要中心逐卡派控，dispatch 开销随卡数涨；纯 multi-controller（既有 RLHF 系统的形态）让每个模型自治，节点间数据依赖被嵌套进各自的分布式程序里，表达与改排都不灵活。HybridFlow 的答案是 **hybrid 控制模型**：模型间数据流编排用 single-controller、模型内分布式计算用 multi-controller，配一组分层 API（base class 3DParallelWorker 封装各模型的分布式前后向 / 生成 / 更新；collect / distribute 原语表达模型间数据重分片）把计算与数据依赖解耦（论文 §4）。实现口径（论文 §7）：总计约 1.2 万行 Python，分层 API 约 1800 行，single-controller 建在 Ray 上、用 RPC 编排各模型执行顺序、中间数据走 TensorDict；训练引擎支持 Megatron-LM / FSDP / DeepSpeed，生成用 vLLM（把其中心式 KV cache 管理器换成分布式以贴合多控制器形态）；3D-HybridEngine 约 2400 行。实测（论文 §8，16 机 × 8 A100-80GB）：PPO 吞吐对 DeepSpeed-Chat v0.14.0 / OpenRLHF v0.2.5 / NeMo-Aligner v0.2.0 的加速比为 7B 1.68–8.63 倍、13B 2.70–18.96 倍、34B 2.41–20.57 倍、70B 5.17–17.98 倍；摘要的全算法口径 1.53–20.57 倍。

repo 现状（组织已由 volcengine/verl 更名 verl-project/verl；下列路径与类名 2026-11-17 于 main 分支核实——repo 仍在演化，函数级引用以核实日期为准）：

- **入口**：`verl/trainer/main_ppo.py`——hydra 读配置，`main` 先按算法需求校验（`need_critic` / `need_reference_policy`，GRPO 类配置下 critic worker 直接不建，§2 的账在入口就生效），`run_ppo` 起 Ray、把 `TaskRunnerV1` 作为 remote actor 拉起，其 `run` 里 `get_trainer_cls(config.trainer.v1.trainer_mode)` 选训练器实现。
- **训练器族**：`verl/trainer/ppo/v1/`——`trainer_base.py`（共享主循环：①-⑤ 的编排）、`trainer_sync.py`（同步）、`trainer_colocate_async.py` 与 `trainer_separate_async.py`（共卡 / 分卡两种异步放置，§3 两模式的代码对应物）、`replay_buffer.py`。
- **算法核**：`verl/trainer/ppo/core_algos.py`——`AdvantageEstimator` 枚举一行一个变体（GAE / GRPO / GRPO_VECTORIZED / GRPO_PASSK / GDPO / RLOO / REINFORCE_PLUS_PLUS 及其 baseline 版 / REMAX / GPG / OPO），`AdaptiveKLController` / `FixedKLController` / `get_kl_controller`（KL 系数控制器），`kl_penalty`（§8）。
- **训练引擎层**：`verl/workers/engine/base.py` 的 `BaseEngine` 定义引擎接口（`train_batch` / `infer_batch` / `forward_backward_batch` 等），实现分布在 `verl/workers/engine/fsdp/` 与 `verl/workers/engine/megatron/`——4.3 与 6.x 的栈在这一层被插进 RL 回路。
- **rollout 层**：`verl/workers/rollout/vllm_rollout/`（§4 已走读）；`verl/single_controller/base/`（`worker.py` / `worker_group.py` / `decorator.py`）与 `verl/single_controller/ray/base.py` 是 hybrid 控制模型里 single-controller 半边的机制本体。
- **off-policy 校正**：`verl/trainer/ppo/rollout_corr_helper.py`（§5 已走读）。

## 7. OpenRLHF 对照

OpenRLHF 的设计取舍与 veRL 几乎逐项对偶（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]）。论文（§1、§3.1）的立场：既有框架把四个模型共置同卡（TRL / ColossalChat / DeepSpeed-Chat 靠 ZeRO 挤），70B 以上放不下；OpenRLHF 反过来**默认分卡**——actor / critic / reward / ref 各占一组 GPU，用 Ray 做放置与编排，训练侧用 DeepSpeed（ZeRO），rollout 侧用 vLLM 引擎组（`vllm_num_engines` 与 `vllm_tensor_parallel_size` 独立配置），同时保留合并开关（`--colocate_actor_ref` / `--colocate_critic_reward` / `ref_reward_offload`）供小模型省卡。性能叙事也分置：veRL 攻 resharding 与控制开销，OpenRLHF 攻 rollout 吞吐（§1 的 80% profile 出自其 §3.2）与代码可改性；摘要口径的加速比为 1.22–1.68 倍（对同代框架）。repo 落点（2026-11-17 核实）：入口 `openrlhf/cli/train_ppo_ray.py`；训练逻辑 `openrlhf/trainer/ppo_trainer.py`；Ray 侧 `openrlhf/trainer/ray/` 下 `ppo_actor.py`（含 §4 的 `broadcast_to_vllm`）、`ppo_critic.py`、`vllm_engine.py`（`update_weight` / `update_weight_cuda_ipc` 两个权重接收入口）、`launcher.py`。

对照表收束两栈分歧：

| 维度 | veRL（HybridFlow） | OpenRLHF |
| --- | --- | --- |
| 控制模型 | hybrid：编排单控 + 模型内多控（论文核心贡献） | Ray actor 群 + trainer 编排 |
| 训练引擎 | Megatron / FSDP 可插（`BaseEngine` 抽象） | DeepSpeed（ZeRO） |
| rollout 引擎 | vLLM（KV 管理器分布式化，论文 §7） | vLLM 独立引擎组 |
| 权重同步 | checkpoint_engine 多后端 + 3D-HybridEngine resharding（式 (3) 与论文 Table 2） | NCCL 广播组或 CUDA IPC（`broadcast_to_vllm`） |
| 放置 | 多种放置均可，Auto device mapping 搜索（论文 §6）；共卡时训练期 offload 生成权重（论文 §7） | 70B+ 四模型分卡；小模型开 colocate |
| 论文实测口径 | 1.53–20.57 倍（128×A100，PPO 7B-70B） | 1.22–1.68 倍（摘要口径） |

第三极是 TRL（[HuggingFace, huggingface/trl repo](../references.md)，教学参照）：`trl/trainer/grpo_trainer.py`（2026-11-17 核实存在）是单进程教学形态——无引擎拆分、无多模型放置问题，vLLM 仅作可选采样后端；读通 §2-§5 的账之后，它是上手验证 8.4 算法语义的最小环境，7B 以下的实验足够，规模一上来就回到本节两栈。

## 8. KL 惩罚与长 CoT 的工程位置

**KL 的两个注入点都在代码里实名存在**。k1/k2/k3 估计器的数学归 8.4（出处 [Schulman, 2020](../references.md)，教学参照），此处只登记工程形态。veRL 的估计器实现在 `verl/trainer/ppo/core_algos.py` 的 `kl_penalty`：k1 为 logp 差；k2 为半平方；k3 为 $\exp(\log \pi_{\mathrm{ref}} - \log \pi_\theta) - (\log \pi_{\mathrm{ref}} - \log \pi_\theta) - 1$ 的低方差无偏形式（代码内对中间量做 ±20 / ±10 clamp 防爆）；后缀 `+`（如 `k3+`）触发 straight-through——前向用 k3、反向换 k2 的梯度，其注释给出的理由是 k1/k3 的期望是 KL 但期望梯度不等于 KL 的梯度，k2 的梯度才是对的。注入位置分两路：**reward 侧**逐 token 扣分在 `verl/trainer/ppo/trainer_utils.py` 的 `apply_kl_penalty`——`token_level_rewards = token_level_scores − β · kld`（β 来自 `kl_ctrl`，adaptive 档按目标 KL 与 horizon 自适应，8.3 的实现细节清单同族）；**loss 侧**在 `verl/workers/utils/losses.py` 的 `ppo_loss`——`use_kl_loss` 开启时把 `kl_loss × kl_loss_coef` 加进 policy loss。两路的梯度语义差异（逐 token 惩罚进 advantage vs 直接进目标）归 8.3/8.4 的算法分析，系统侧的要点是：ref 前向与 kld 计算都按逐 token 逐样本走， $2 N_a$ 的 ref 常驻（式 (1)）与一次额外前向是这两个注入点共同的价格。

**长 CoT 把式 (1) 的 KV 项与 ① 的时长一起拉大**。DeepSeek-R1 论文记录其 RL 训练中平均 response 长度持续增长（self-evolution 分析口径，[DeepSeek-AI, 2025](../references.md)，arXiv:2501.12948，[A]；算法侧归 9.5）。系统后果三条：其一，rollout 时长由 batch 内最长序列决定，长度漂移意味着 ① 的时间预算随训练进程单调恶化；其二，KV cache 随长度线性涨，共卡模式下与训练显存的错峰（§4 的 sleep/wake）从可选变成必需；其三，`max_len` 截断切断 EOS——规则验证器对截断样本判不了对错，奖励信号与梯度语义同时受损，overlong 处理与 dynamic sampling 的算法对策归 8.4（DAPO 细节包，[Yu et al., 2025](../references.md)，arXiv:2503.14476，[A]），系统侧的配套是把长度预算、截断率与有效样本率纳入每轮监控，并允许 `max_len` 随训练阶段调整。

## 总结

| 主题 | 结论 | 出处 / 锚点 |
| --- | --- | --- |
| 四模型显存 | PPO 全配置 $\mathrm{Mem} = 16(N_a + N_c) + 4 N_a + 2 N_r + \mathrm{KV}$（式 (1)）；同尺寸 $38N$ ；GRPO 去 critic、RLVR 去 RM 降到 $20N$ | models 1.2 式 (8)；4.2；8.4 |
| 引擎拆分 | 生成阶段约占 PPO 时间 80%（7B A100 profile）；训练引擎（Megatron/FSDP）+ 推理引擎（vLLM）分置，layout 不一致引出 resharding | OpenRLHF §3.2；图归 §3 |
| 权重同步成本 | $t_{\mathrm{sync}} \ge 2 N_a / B$（式 (2)）；70B：NVLink 域 ≈0.23 s，25 GB/s 链路 ≈5.6 s，共享存储十几秒级 | HybridFlow §8.1 带宽口径 |
| 3D-HybridEngine | micro-DP 组内 all-gather：通信 $\frac{d_g - 1}{tp} M$ 、峰值 $M/(t_g p_g)$ 、冗余 0（Table 2） | HybridFlow §5 |
| off-policy | 两来源：版本 staleness + 引擎数值差；decoupled 重算锚点 + 截断 IS / 拒采样 | 式 (4)；`rollout_corr_helper.py` |
| veRL | hybrid 控制模型（编排单控 + 计算多控）；约 1.2 万行；1.53–20.57 倍 | arXiv:2409.19256 |
| OpenRLHF | Ray 分卡放置四模型（70B+）+ vLLM 引擎组；1.22–1.68 倍 | arXiv:2405.11143 |
| KL 工程 | reward 侧 `apply_kl_penalty` 与 loss 侧 `ppo_loss` 双注入点；k1/k2/k3 与 straight-through `k3+` | `core_algos.py`；8.4 |

## 练习

1.（计算）actor / critic / ref / RM 全为 70B、bf16 + AdamW、训练态两模型开 ZeRO-1（ $d = 64$ ）：(a) 按式 (1) 与 $4N + 12N/d$ 口径算集群模型状态总量（KV 不计）；(b) 对照 128 × 80 GB 集群，给出放置可行性与留给激活 / KV 的余量；(c) 改 GRPO + 规则奖励重算，给出节省的 bytes 与相对比例，并说明这笔节省被搬去了哪一项。

2.（计算）权重同步预算：actor 70B（bf16），集群节点内 600 GB/s、节点间 25 GB/s。(a) 用式 (2) 算两条路径的每轮同步下界；(b) 设一轮 RL iteration 的 rollout + train 合计 20 min，两种放置下同步各占多少比例；(c) 若改异步流水线每 $k$ 轮同步一次，写出「同步成本 ÷ $k$ 、staleness 随 $k$ 涨」的权衡式，并结合式 (4) 说明 $k$ 增大时 clip 区间外的样本质量如何变化（8.3 的 ratio 分析回链）。

3.（推导）验证 HybridFlow Table 2 的 8 卡例： $(t, p, d) = (4, 1, 2)$ 、 $(t_g, p_g) = (2, 1)$ 。(a) 推出 $d_g$ 并列出三个方案的通信系数；(b) 证明 micro-DP 构造下每卡的生成分片恰为本组各卡训练分片之并（零冗余的来源）；(c) 若生成侧改 $(t_g, p_g) = (4, 1)$ ， $d_g$ 与通信量怎么变，什么分组条件下 HybridFlow 的通信优势消失？

4.（分析）共卡 vs 分卡的判定：把式 (1) 的模型状态、式 (2) 的同步下界、sleep/wake 的 offload 开销放进同一个框架，论证什么模型规模与卡数组合下共卡时分复用优于分卡专用、什么组合下反转；指出你引入的每个假设的敏感性（提示：offload 走 GPU↔CPU 链路，带宽口径归 4.2 §5）。

## 参考

- [Sheng et al., 2024](../references.md)：HybridFlow: A Flexible and Efficient RLHF Framework（veRL 论文），arXiv:2409.19256，[A]（key: verl）——hybrid 控制模型与分层 API（§4）、3D-HybridEngine 与 Table 2（§5）、实现与实测口径（§7-§8）；repo 走读：verl-project/verl 的 `verl/trainer/main_ppo.py`、`verl/trainer/ppo/v1/`（`trainer_base.py` 的 `_compute_old_log_prob` 双模式、`trainer_sync.py` / `trainer_colocate_async.py` / `trainer_separate_async.py`）、`verl/trainer/ppo/core_algos.py`（`AdvantageEstimator`、`kl_penalty`、KL 控制器）、`verl/trainer/ppo/trainer_utils.py`（`apply_kl_penalty`）、`verl/trainer/ppo/rollout_corr_helper.py`（截断 IS 与拒采样）、`verl/workers/engine/`（`BaseEngine` 与 fsdp / megatron 实现）、`verl/workers/rollout/vllm_rollout/`（`update_weights`、`release` / `resume`）、`verl/checkpoint_engine/`（NCCL / NIXL / Mooncake 多后端）、`verl/single_controller/`；路径核实于 2026-11-17
- [Hu et al., 2024](../references.md)：OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework，arXiv:2405.11143，[A]（key: openrlhf）——分卡放置立场（§3.1）、生成阶段 80% profile（§3.2）、摘要加速比口径；repo 走读：OpenRLHF/OpenRLHF 的 `openrlhf/cli/train_ppo_ray.py`、`openrlhf/trainer/ppo_trainer.py`、`openrlhf/trainer/ray/ppo_actor.py`（`broadcast_to_vllm`、NCCL 广播与 CUDA IPC 双路径）、`openrlhf/trainer/ray/vllm_engine.py`（`update_weight` / `update_weight_cuda_ipc`）；路径核实于 2026-11-17
- [Kwon et al., 2023](../references.md)：vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——rollout 引擎锚点（内部机制归 inference 维度）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——训练引擎锚点之一（运行时归 6.2）
- [Zhao et al., 2023](../references.md)：PyTorch FSDP，arXiv:2304.11277，[A]（key: fsdp）——训练引擎锚点之二（机制归 4.3）
- [Rajbhandari et al., 2020](../references.md)：ZeRO，arXiv:1910.02054，[A]（key: zero）——训练态 $16N$ 与 ZeRO-1 $4N + 12N/d$ 口径（推导归 4.2）
- [Schulman et al., 2017](../references.md)：Proximal Policy Optimization Algorithms，arXiv:1707.06347，[B]（key: ppo）——四模型配置与 clip 容忍区间（推导归 8.3）
- [Shao et al., 2024](../references.md)：DeepSeekMath（GRPO 出处），arXiv:2402.03300，[A]（key: deepseekmath）——去 critic 的算法来源（推导归 8.4）
- [Yu et al., 2025](../references.md)：DAPO，arXiv:2503.14476，[A]（key: dapo）——overlong 处理与 dynamic sampling（算法侧归 8.4）
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-R1，arXiv:2501.12948，[A]（key: deepseek-r1）——RL 训练中 response 长度增长的记录
- [Schulman, 2020](../references.md)：Approximating KL Divergence，教学参照（key: schulman-kl）——k1/k2/k3 估计器出处（推导归 8.4）
- [HuggingFace, huggingface/trl repo](../references.md)：教学参照（key: trl-repo）——`trl/trainer/grpo_trainer.py` 单进程形态对照（2026-11-17 核实）
