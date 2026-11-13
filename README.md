# ai-infra-primer

一套 hands-on 的 AI infrastructure primer——from tokens to teraflops：从 token 的数学形式化开始，把模型、训练、推理、硬件、集群五层的技术账算到公式级。中文正文、术语保持英文，所有数值事实带出处并标注证据等级。

**阅读站点（推荐）**：<https://alanfokco.github.io/ai-infra-primer/> — 构建期 KaTeX 渲染、全文搜索、章内目录与阅读进度；由 GitHub Actions 自动构建部署（`.github/workflows/pages.yml`，站点源码在 `site/`）。

## 五维度导航

| 维度 | 状态 | 范围 |
| --- | --- | --- |
| [models/](models/README.md) | 进行中（W1） | 生成式模型的数学：文本 LLM 主干、视觉表征、扩散生图、多模态 LLM、视频生成、语音音频（10 Part / 51 章） |
| [training/](training/README.md) | 占位 | 训练算法与工程：optimizer、RLHF（PPO / GRPO）、FP8、分布式训练策略 |
| [inference/](inference/README.md) | 占位 | serving 系统：continuous batching、PagedAttention、speculative decoding 的系统实现 |
| [gpu/](gpu/README.md) | 占位 | GPU 硬件与 kernel：内存层级、roofline、FlashAttention 等算子的实现层 |
| [cluster/](cluster/README.md) | 占位 | 集群与通信：EP、all-to-all、Ring Attention、网络拓扑 |

维度边界按机制归属划分：模型侧公式在 models，训练动力学在 training，运行时系统在 inference，硅上细节在 gpu，跨机通信在 cluster；交叉处只做链接，不重复展开。

## models 阅读路径

10 个 Part 的依赖顺序：P1 Foundations → P2 Transformer Anatomy → P3 Objectives & Variants → P4 Modern Architectures → P5 Case Studies 是文本主干；P6 Vision Representations 接 P2，P7 Diffusion Image Generation 接 P6，P8 Multimodal LLM 接 P7；P9 Video Generation 接 P7，P10 Audio & Speech 接 P6（RVQ codec）。完整目录、依赖图与逐章进度表见 [models/README.md](models/README.md)。

## 规范

- [STYLE_GUIDE.md](STYLE_GUIDE.md)：写作与审查的唯一标准——语言、公式、代码、图、锚点规则、引用与证据分级（[A] / [B] / [C] / [W]）
- [models/references.md](models/references.md)：全维度文献汇总（按 Part 分组，chapter → sources 映射）

## 项目状态

`波次 W1` `models 维度写作中` `training / inference / gpu / cluster 占位`

逐章进度（未开始 / 已写 / 审查中 / 已关闭）见 [models/README.md](models/README.md) 的进度表。
