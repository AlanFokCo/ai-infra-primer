# inference 维度：serving 系统（Inference）

> 状态：未开工（占位）。本文件只声明范围与接口；正式章节设计另起 spec。

## 范围

本维度讲模型上线后的 serving 系统：

- continuous batching 与请求调度
- PagedAttention：KV cache 的系统管理（块分配、换出、prefix 复用）
- speculative decoding 的系统实现：draft / verify 流水线、接受率到延迟的换算
- 采样（temperature / top-p）与流式输出的工程实现

边界：KV cache 公式、prefill / decode 算术强度、MTP 的模型侧目标归 models；运行时调度与内存管理归这里。划分依据见 [models 维度 spec §2](../docs/superpowers/specs/2026-11-09-models-dimension-design.md)。

## 与 models 的接口

models 侧交叉链接到这里的章节：

- [2.4 Causality 与 KV Cache](../models/02-transformer-anatomy/04-causality-and-kv-cache.md)：prefill / decode 算术强度分析的系统侧展开
- [4.5 MoE 与 MTP](../models/04-modern-architectures/05-moe.md)：MTP 推理时作为 self-speculative decoding 的系统细节
