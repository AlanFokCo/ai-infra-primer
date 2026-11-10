# training 维度：训练算法与工程（Training）

> 状态：未开工（占位）。本文件只声明范围与接口；正式章节设计另起 spec。

## 范围

本维度讲训练算法与训练工程：

- optimizer 及其动力学：AdamW、Muon 等更新规则的公式、稳定性行为与调参账
- RL 后训练：PPO / GRPO 的目标函数、优势估计与实现
- 低精度训练工程：FP8 quantization scheme（per-tensor / blockwise scaling、累加精度、异常值处理）
- 分布式训练策略：数据 / 张量 / 流水并行、优化器状态切分、吞吐与显存的账

边界：训练目标的数学形式（loss 本身）归 models；如何优化目标归这里。划分依据见 [models 维度 spec §2](../docs/superpowers/specs/2026-11-09-models-dimension-design.md)。

## 与 models 的接口

models 侧交叉链接到这里的章节：

- [1.2 参数、计算、内存的账](../models/01-foundations/02-compute-accounting.md)：优化器状态内存账（6N~12N bytes）
- P5 案例章训练侧：[5.2 DeepSeek](../models/05-case-studies/02-deepseek.md)（FP8 训练工程）、[5.3 Qwen](../models/05-case-studies/03-qwen.md)（thinking 模式）、[5.5 Kimi / Moonshot](../models/05-case-studies/05-kimi-moonshot.md)（Muon、K1.5 RL）
