# ai-infra-primer

一套 hands-on 的 AI infrastructure primer——from tokens to teraflops：从 token 的数学形式化开始，把模型、训练、推理、硬件、集群五层的技术账算到公式级。中文正文、术语保持英文，所有数值事实带出处并标注证据等级。

**阅读站点（推荐）**：<https://alanfokco.github.io/ai-infra-primer/> — 构建期 KaTeX 渲染、全文搜索、章内目录与阅读进度；由 GitHub Actions 自动构建部署（`.github/workflows/pages.yml`，站点源码在 `site/`）。

## 五维度导航

| 维度 | 状态 | 范围 |
| --- | --- | --- |
| [models/](models/README.md) | 已关闭（W1–W7 全部审结） | 生成式模型的数学：文本 LLM 主干、视觉表征、扩散生图、多模态 LLM、视频生成、语音音频（10 Part / 51 章） |
| [training/](training/README.md) | 已关闭（W0–W6 全部审结） | 训练动力学与训练系统：autograd 与算子反向、optimizer 与 LR（AdamW / Muon）、混合精度与 FP8、数据并行（DDP / ZeRO / FSDP）与模型并行（TP / SP / CP / PP / EP）、Megatron-LM、预训练全流程、RL 数学（PPO / GRPO）、后训练全流程（SFT / RLHF / DPO / RLVR）（9 Part / 39 章） |
| [inference/](inference/README.md) | 已关闭（W0–W7 全部审结） | serving 运行时系统：请求生命周期与 SLO、prefill / decode 性能账与 roofline、KV cache 系统（paged / prefix / 量化 / offload / 分层存储）、推理算子与权重量化、调度与 batching、投机解码、三大引擎代码走读（vLLM V1 / SGLang / DeepInfer）、分布式推理与 PD 分离、弹性伸缩与容错、生产全流程（9 Part / 37 章） |
| [gpu/](gpu/README.md) | 已关闭（W0–W7 全部审结） | GPU 硅片与通信全栈：微架构、内存层级与带宽账、CUDA 编程模型、kernel 访存/计算优化、低精度与注意力 kernel、NVLink/NVSwitch、NCCL、RDMA/GPUDirect、机间 fabric、系统集成与生产（11 Part / 41 章） |
| [cluster/](cluster/README.md) | 占位 | K8S/容器/Agent Sandbox：平台编排层 |

维度边界按机制归属划分：模型侧公式在 models，训练动力学在 training，运行时系统在 inference，硅上细节与通信全栈在 gpu，平台编排在 cluster；交叉处只做链接，不重复展开。

## models 阅读路径

10 个 Part 的依赖顺序：P1 Foundations → P2 Transformer Anatomy → P3 Objectives & Variants → P4 Modern Architectures → P5 Case Studies 是文本主干；P6 Vision Representations 接 P2，P7 Diffusion Image Generation 接 P6，P8 Multimodal LLM 接 P7；P9 Video Generation 接 P7，P10 Audio & Speech 接 P6（RVQ codec）。完整目录、依赖图与逐章进度表见 [models/README.md](models/README.md)。

## training 阅读路径

9 个 Part 的依赖顺序：优化线 P1 Autograd 与反向机制 → P2 Optimizer 与 LR → P3 Loss 函数族；并行线 P4 数据并行（DDP / ZeRO / FSDP）→ P5 模型并行（TP / SP / CP / PP / EP）→ P6 Megatron 与训练栈；P7 预训练全流程合流两线；P8 RL 数学详解 → P9 后训练全流程。完整目录、依赖图与逐章进度表见 [training/README.md](training/README.md)。

## inference 阅读路径

9 个 Part 的依赖顺序：机制线 P1 推理原理与性能账 → P2 KV cache 系统 / P3 推理算子 → P4 调度与 batching；引擎走读 P5 vLLM / P6 SGLang / P7 DeepInfer 与 DeepSeek 栈立于机制之上；P8 分布式推理与 PD 分离收三引擎的分布式实现；P9 生产全流程合流。完整目录、依赖图与逐章进度表见 [inference/README.md](inference/README.md)。

## 规范

- [STYLE_GUIDE.md](STYLE_GUIDE.md)：写作与审查的唯一标准——语言、公式、代码、图、锚点规则、引用与证据分级（[A] / [B] / [C] / [W]）
- [models/references.md](models/references.md)、[training/references.md](training/references.md)、[inference/references.md](inference/references.md) 与 [gpu/references.md](gpu/references.md)：各维度文献汇总（按 Part 分组，chapter → sources 映射）

## 项目状态

`models 维度 51 章已关闭（W1–W7）` `training 维度 39 章已关闭（W0–W6）` `inference 维度 37 章已关闭（W0–W7）` `gpu 维度 41 章已关闭（W0–W7）` `cluster 占位` `站点上线（models + training + inference + gpu）`

逐章进度（未开始 / 已写 / 审查中 / 已关闭）见 [models/README.md](models/README.md)、[training/README.md](training/README.md)、[inference/README.md](inference/README.md) 与 [gpu/README.md](gpu/README.md) 的进度表。
