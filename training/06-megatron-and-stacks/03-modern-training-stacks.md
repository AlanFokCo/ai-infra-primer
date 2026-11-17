# 6.3 现代训练栈对照与选型（Modern Training Stacks: Comparison and Selection）

> P4-P6 的算法地图落到可下载的栈上是六个名字：Megatron-Core / NeMo / Megatron-Bridge、TorchTitan、DeepSpeed、OLMo-core，外加后训练侧的 TRL / veRL / OpenRLHF。本章逐栈登记覆盖面的代码级事实（repo 现状全部核实于 2026-11-17），给出按规模 × 并行需求 × 团队栈的选型表，并划清训练栈与推理栈的边界 | 前置：4.2（ZeRO 各阶）、4.3（FSDP/FSDP2）、P5（模型并行全家桶）、6.1-6.2（Megatron 层栈与运行时） | 锚点：pytorch/torchtitan repo（[PyTorch, pytorch/torchtitan repo](../references.md)，教学参照 + 官方实现）、Megatron-LM 论文 + repo（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）、deepspeedai/DeepSpeed repo（[DeepSpeed, deepspeedai/DeepSpeed repo](../references.md)，[A]）、allenai/OLMo-core（[AllenAI, allenai/OLMo-core repo](../references.md)，[A]）、NVIDIA NeMo（[NVIDIA, NVIDIA NeMo](../references.md)，教学参照）、huggingface/trl（[HuggingFace, huggingface/trl repo](../references.md)，教学参照 + 官方实现）

## 1. Megatron 家族：Core / NeMo / Bridge 的三层分工

Megatron-LM repo 的 README 把自己拆成两层，原话照录：Megatron Core 是 "a composable library with GPU-optimized building blocks for custom training frameworks"（可组合的库，提供 transformer 构件、TP/PP/DP/EP/CP 并行与混合精度），Megatron-LM 则是 "a reference example that includes Megatron Core plus pre-configured training scripts"（Core 加预配置脚本的参考实现）。6.1/6.2 走读的全部路径（`megatron/core/` 下的并行原语、进程组、调度、optimizer、dist-ckpt）都在 Core 层。

```
HF 生态模型/权重 ◄──双向转换──► Megatron-Bridge ──┐
                                                  ▼
NeMo（框架层：recipe / 数据 / 评测 / 导出） ──► Megatron-Core（库：5D 并行 + dist-opt + dist-ckpt）
                                                  ▲
Megatron-LM（参考实现：Core + 预配置训练脚本） ────┘
```

NeMo 是再上一层的框架封装（references 口径：「Megatron-Core 的上游封装」）。组织现状有一处 2026 年的变动需要登记（核实于 2026-11-17）：`NVIDIA/NeMo` 现 301 重定向到 `NVIDIA-NeMo/Speech`——后者即 2019 年创建的原 NeMo 库（nemo-toolkit 包名与徽章未变），NVIDIA-NeMo 组织下另拆出 RL、Curator、Evaluator、Automodel、Megatron-Bridge 等独立库；references.md 里 nemo 条目的 URL 经重定向仍可达。拆库之后框架层的职能按库名即可读：数据策展（Curator）、评测（Evaluator）、导出部署（Export-Deploy）、后训练 RL（RL）、内容护栏（Guardrails）、HF 生态直连（Automodel），加上原 NeMo 本体（Speech）。对只关心大规模预训练的团队，日常打交道的仍是最下面两层：Megatron-Core 的库接口与 Megatron-LM 的参考脚本。

Megatron-Bridge（`NVIDIA-NeMo/Megatron-Bridge`）的 repo 自述是 "Training library for Megatron-based models with bidirectional Hugging Face conversion capability"：一头接 HF 生态的模型定义与权重格式，一头接 Megatron-Core 的 5D 并行运行时，双向 checkpoint 转换是其核心功能。典型用法：把 HF 上的开放权重转进 Megatron 做大规模继续训练或微调，再转回 HF 格式发布。三层各自的读者不同：Megatron-Core 面向搭自有训练框架的人（库接口），Megatron-LM 参考实现层面向要「能跑起来的默认答案」的人（预配置脚本把 Core 按经典配置装配好，新模型接入通常从复制参考实现开始），NeMo / Bridge 面向要把模型、数据、评测、导出整条链一次拿走的人。

## 2. TorchTitan：PyTorch 原生最小栈

pytorch/torchtitan 的 README 首行标语是 "A PyTorch-native platform for pretraining and post-training generative AI models at scale"（BSD 协议）。它的设计轴与 Megatron 相反：不自建并行原语层，全部组合原生 PyTorch 能力。README 特性清单（2026-11-17 核实）逐项是：

- DP：FSDP2 / HSDP / DDP——FSDP2 即 per-param DTensor 全分片，语义 = ZeRO-3（4.3）；
- TP：含 async TP；EP 亦有；
- PP：直接用 `torch.distributed.pipelining` 的 schedule——1F1B、interleaved 1F1B、GPipe（5.3 的算法族在此以原生形式出现）；
- CP：all-gather KV 与 Ulysses 两种实现（5.2 的两个算法族）；
- 激活：checkpointing 分 full / selective / regional 三档，配 regional `torch.compile` 与 CUDA graphs；
- kernel：FlashAttention-3/4、FlexAttention、varlen attention；
- 其余：BF16 optimizer states、FSDP CPU offload、Grain 数据管线（packing / mixing / 可 checkpoint 的数据状态）、distributed checkpointing（含异步 save）、HF↔DCP 转换脚本。

代码组织上，`torchtitan/` 的主要目录为 `models, components, distributed, config, experiments, rl, observability, ops, quantization, tools`（2026-11-17 核实的完整清单）：README 的组织原则是 "Models declare how they shard"——模型代码只声明切分意图，并行策略以组合方式施加；`rl` 目录的存在说明它已从纯预训练栈扩到后训练。分布式层的文件名就是组合面本身：`torchtitan/distributed/` 按维分文件——`fsdp.py`（FSDP2 / HSDP 装配）、`pipeline_parallel.py`（PP schedule 装配）、`context_parallel.py`（CP）、`activation_remat.py`（三档重计算）、`offloading.py`（CPU offload）、`parallelism_context.py`（导出 `ParallelismContext` 与 `DistributedTopology`，把各维进程组与设备网格收进一个显式上下文对象）。这里有一处与 Megatron 的工程风格对照值得点名：Megatron 的进程组树是全局单例（6.2 §1 的 `parallel_state.py`，其模块 docstring 声明正在向显式传组迁移），TorchTitan 从设计上就是显式上下文对象——分歧不在算法，在并行状态放在哪里。`components/` 层是训练循环的零件盒：`checkpointer`（DCP 封装与异步 save）、`data`（Grain 管线）、`optim`（含 BF16 optimizer states）、`loss.py`、`tokenizer.py`、`validator.py`；顶层以 `train.py` / `trainer.py` / `training_engine.py` 三个文件做入口。

与 Megatron 的覆盖面对照：

| 维度 | Megatron-Core | TorchTitan |
| --- | --- | --- |
| 并行维 | TP/SP/CP/PP/EP/DP 全（6.1-6.2） | TP/CP/PP/EP/DP 全（原生组合） |
| DP 域分片档 | dist-opt = ZeRO-1 族（6.2 §4） | FSDP2 = ZeRO-3 语义（4.3） |
| PP schedule | 非交错 / 交错 1F1B（自研） | `torch.distributed.pipelining`（1F1B / 交错 / GPipe） |
| checkpoint | dist-ckpt，保存/加载网格解耦（6.2 §5） | DCP，含异步 save |
| 证据形态 | 论文 + repo（[A]） | repo 锚点（references 登记为教学参照 + 官方实现） |

表里最结构性的一行是 DP 分片档：Megatron 停在 ZeRO-1 族（参数全量常驻、每 step 通信约 $2N$ 元素），TorchTitan 的 DP 直接是 ZeRO-3 语义（参数用时 gather、通信到 $3N$ 上限、显存随 $d$ 线性缩）。同样的卡数下两者落在 4.2 决策表的不同行——选栈之前先按 4.2 / 5.5 的账算出自己该在哪一行，再谈栈（练习 1 算这个差异）。

阅读路径上两栈也互补：要理解并行机制本身，TorchTitan 的组合式写法让「机制 → 原生 API」的映射直接可见（4.3 的 FSDP2、5.3 的 pipelining schedule 都是原生组件，没有中间封装层）；要理解生产级大栈的全部边角（进程组树、混合精度状态机、分布式 ckpt 的分片映射），Megatron-Core 是绕不开的——6.1（原语）→ 6.2（运行时）的走读顺序就是推荐阅读顺序。

## 3. DeepSpeed：ZeRO 原厂与组合用法的现状

ZeRO 的原厂实现在 deepspeedai/DeepSpeed 的 `deepspeed/runtime/zero/`（4.2 §6 已做文件级走读，此处不重复）。与 Megatron 的组合范式由 Megatron-DeepSpeed 仓承载（TP/PP 走 Megatron、ZeRO 分片走 DeepSpeed）；该仓随组织更名现居 `deepspeedai/Megatron-DeepSpeed`，最近一次提交是 2025-08-14（核实 2026-11-17），对照 DeepSpeed 主仓 2026-11 仍在日更。组合需求降温的原因：Megatron-Core 自带 dist-opt 之后，「Megatron 并行 + ZeRO-1/2 分片」不再需要外挂 DeepSpeed（6.2 §4 的机制）——此句为本书分析读法。DeepSpeed 仍独占一档：ZeRO-Offload / ZeRO-Infinity 的 CPU/NVMe 卸载（4.2 §5 的 $4N$ bytes/step PCIe 下界账）在卡少模型大场景没有替代品。另两项独有能力的出处登记如下：ZeRO++ 的跨节点通信三件套（qwZ / hpZ / qgZ，跨节点 $3M \to 0.75M$ ，4.2 §5.3），以及配置面 `zero_optimization.stage` 的枚举语义（1/2/3 = 这一阶新切优化器状态 / 梯度 / 参数，4.2 §6）。

## 4. OLMo-core：全开放配套栈

allenai/OLMo-core 是 OLMo 2 的训练栈（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]），配套完整性在六个栈里独一档：包内 `data / nn / optim / distributed / train / eval / launch` 全模块开源（包结构核实 2026-11-17），训练配置即发布配方（2.2 §7 引过的 `src/scripts/train/OLMo2/OLMo2-7B.py` 就是这个栈的 config 形态），外加开放数据与逐阶段 checkpoint。包结构里另有几块值得点名：`float8` 与 `mxfp8_config`（低精度训练配置，衔接 1.3）、`kernels` 与 `ops`（自定义算子）、`generate` 与 `eval`——OLMo-core 覆盖从数据到评测生成的完整链路，不只是训练循环。它的定位不是覆盖并行维的广度，而是可复现性的深度：要逐行复现一线配方（7.2 的全字段配方表）或做训练动力学研究（2.4 / 7.3 的稳定性记录），OLMo-core 是唯一数据、代码、配置、权重四者全开的锚点。

## 5. 后训练栈预告：接口定位

P9 的三个栈在这里只立接口定位，数学与系统展开归 8.4 / 8.5 与 P9：

- TRL（huggingface/trl，[教学参照 + 官方实现](../references.md)）：HF 生态的 SFT / DPO / GRPO trainer 库，中小规模与教学场景的标准入口；其偏好数据接口（DPOTrainer 的输入格式）已在 3.2 走读过——TRL 与本 Part 各栈的关系是「训练循环全部交给 HF Trainer，并行能力继承 transformers / accelerate 生态」，大规模 RL 不是它的设计目标；
- veRL（[Sheng et al., 2024](../references.md)，arXiv:2409.19256，[A]）：字节的 RL 训练框架，核心是 hybrid 控制模型（数据流编排用单控制器、模型内计算用多控制器），训练引擎可插 Megatron 或 FSDP——8.5 的主走读对象之一；
- OpenRLHF（[Hu et al., 2024](../references.md)，arXiv:2405.11143，[A]）：Ray + vLLM 组合的 RLHF 栈，actor / critic / reward / ref 四模型按 Ray actor 分布、rollout 交 vLLM——8.5 的对照对象。

三个栈共享的系统问题也就是 8.5 的章节目录：四模型同驻的显存账、训练引擎与 rollout 引擎之间的权重同步成本、rollout 与训练版本错位带来的 off-policy 校正。本章只立接口：这些栈对「训练引擎」的要求恰好是本 Part 各栈的产出形态（一个能 forward/backward/step 并能导出权重的大模型实例）。

三者的共同结构就是 8.5 的三段式，训练栈与推理栈的边界由此划清：

```
┌──────────── 训练栈（本章各栈） ────────────┐   权重同步（每 iteration 广播，8.5 的账）
│  actor 训练引擎：forward/backward/optimizer │ ◄──────────────┐
└─────────────────────────────────────────────┘                │
        ▲ prompts + 经验数据                                    │
        │              ┌───────── 推理栈 ─────────┐             │
        └───────────── │ rollout 引擎（vLLM 等）：  │ ────────────┘
                       │ 批量采样生成；内部机制      │
                       │ （continuous batching /   │
                       │  PagedAttention）归       │
                       │  inference 维度           │
                       └───────────────────────────┘
```

边界判据一句话：训练栈产出与消费 checkpoint，rollout 引擎的内部机制归 inference 维度只链接，训练侧要关心的只是每 iteration 的权重广播成本（8.5 展开）。RL 场景下两段引擎分开选型——训练段用本 Part 各栈、rollout 段用推理引擎——是后训练系统设计的常态组合。

## 6. 选型决策表

按「先算账、后选栈」的纪律，账归 4.4 / 5.5，本表只做账到代码的映射。模型规模 × 并行需求两轴之外还有第三轴——团队栈：已有 NVIDIA 栈运维经验的团队沿用 Megatron 系的学习成本最低；基建全在 PyTorch 原生分布式（FSDP / DTensor / pipelining）上的团队选 TorchTitan 可以少维护一层自有并行原语；要把训练栈本身当研究对象逐行读的，从 OLMo-core 与 TorchTitan 入手——两者代码量都远小于 Megatron-LM 全仓（此句为定性判断，无量化口径）。

| 场景 | 首选栈 | 依据 |
| --- | --- | --- |
| 从零预训练 ≥70B，TP/PP/EP 全家 + 成熟调度 | Megatron-Core（或经 NeMo / Bridge 封装） | 5D 并行 + dist-opt + dist-ckpt 覆盖最全（6.1/6.2） |
| HF 开放权重进 Megatron 体系继续训练 | Megatron-Bridge | 双向转换是其核心功能（§1） |
| PyTorch 原生路线、要小内核可改、ZeRO-3 档分片 | TorchTitan | 原生原语组合；FSDP2 语义（§2） |
| 卡少模型大、需 CPU/NVMe 卸载 | DeepSpeed（ZeRO-Offload / Infinity） | 4.2 §5 的独有档（§3） |
| 全开放复现与训练研究 | OLMo-core | 数据/代码/配置/权重全开（§4） |
| 后训练 SFT/DPO，中小规模 | TRL | §5 接口定位 |
| 后训练大规模 RL（PPO/GRPO） | veRL / OpenRLHF | 8.5 展开 |

本表不决定的两件事：模型结构选型（dense vs MoE、长上下文路线）归 models 维度，数据与配方（lr / bsz / warmup 全字段）归 7.2——栈只承接这两件事定完之后剩下的工程映射。

## 总结

| 栈 | 一句话定位 | 关键事实（核实日 2026-11-17） |
| --- | --- | --- |
| Megatron-Core | 5D 并行 + dist-opt + dist-ckpt 的可组合库 | repo README 双层自述；6.1/6.2 走读 |
| NeMo / Megatron-Bridge | 框架封装层 / HF↔Megatron 双向转换 | `NVIDIA/NeMo` 重定向至 `NVIDIA-NeMo/Speech`；Bridge 在 `NVIDIA-NeMo/Megatron-Bridge` |
| TorchTitan | PyTorch 原生最小栈，模型自声明 sharding | FSDP2 + pipelining schedule + Grain + DCP；`rl` 目录已存在 |
| DeepSpeed | ZeRO 原厂；Offload/Infinity 独占档 | 主仓日更；Megatron-DeepSpeed 组合仓 2025-08 后无新提交 |
| OLMo-core | 四者全开的复现锚点 | 包内 data/nn/optim/train/eval 全模块开源 |
| TRL / veRL / OpenRLHF | 后训练栈，接口定位 | 三段式：训练引擎 + rollout 引擎 + 权重同步（8.5） |

## 练习

1.（计算）64 卡、bf16、 $N = 70 \times 10^{9}$  dense 模型：分别按 (a) Megatron-Core dist-opt（bf16 梯度口径，6.2 §4 表）与 (b) TorchTitan FSDP2（ZeRO-3 语义，4.2 §4 账）算单卡模型状态显存（GB，SI 口径），并各加激活余量 20 GB 后对照 80 GB 卡判定可行性；说明同样卡数下两栈落点差异来自哪个设计选择。

2.（查证）拉取 pytorch/torchtitan 与 NVIDIA/Megatron-LM 的当前 README，把两者声明支持的并行维与 schedule 列成表，与 §2 对照表逐行核对。栈的特性清单以一手 README 为准——本题训练的就是不引用二手综述的习惯。

3.（设计）8 台 8×H100（80 GB）节点，任务是从 HF 开放权重继续训练一个 32B dense 模型：先按 4.2 / 5.5 的账定并行度数（给出两组合法配置），再按 §6 决策表选栈，写出每个候选方案的取舍（转换成本、DP 分片档、checkpoint 路径）。

## 参考

- [PyTorch, pytorch/torchtitan repo](../references.md)（key: torchtitan-repo，教学参照 + 官方实现）——README 特性清单与 `torchtitan/` 包结构（2026-11-17 核实）
- [Shoeybi et al., 2019](../references.md)：Megatron-LM，arXiv:1909.08053，[A]（key: megatron-lm）——repo README 的 Core / LM 双层自述；NeMo 重定向与 Megatron-Bridge 归属（2026-11-17 核实）
- [NVIDIA, NVIDIA NeMo](../references.md)（key: nemo，教学参照）——框架封装层口径
- [DeepSpeed, deepspeedai/DeepSpeed repo](../references.md)（key: deepspeed-repo，[A]）——ZeRO 实现位置（4.2 §6）与 Megatron-DeepSpeed 组合仓现状（2026-11-17 核实）
- [AllenAI, allenai/OLMo-core repo](../references.md)（key: olmo-core，[A]）——包结构与 OLMo 2 训练配置形态（2026-11-17 核实）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——OLMo-core 对应的配方论文
- [HuggingFace, huggingface/trl repo](../references.md)（key: trl-repo，教学参照 + 官方实现）——§5 的 TRL 接口定位
- [Sheng et al., 2024](../references.md)：HybridFlow: A Flexible and Efficient RLHF Framework（veRL 论文），arXiv:2409.19256，[A]（key: verl）——§5 的 veRL 接口定位
- [Hu et al., 2024](../references.md)：OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework，arXiv:2405.11143，[A]（key: openrlhf）——§5 的 OpenRLHF 接口定位
