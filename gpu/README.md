# gpu 维度：GPU 硬件与 kernel（GPU）

> 状态：未开工（占位）。本文件只声明范围与接口；正式章节设计另起 spec。

## 范围

本维度讲硅上的实现层：

- 硬件模型：HBM / SRAM / register 内存层级、带宽与算力数字、roofline
- kernel 实现：FlashAttention 的 CUDA 实现（tiling、occupancy、register 压力）与 FA2 / FA3 的硬件对齐改进
- 数值格式的硬件路径：bf16 / FP8 tensor core

边界：FlashAttention 的数学（online softmax 正确性）与 IO 复杂度推导归 models 4.1；kernel 与硬件归这里。划分依据见 [models 维度 spec §2](../docs/superpowers/specs/2026-11-09-models-dimension-design.md)。

## 与 models 的接口

models 侧交叉链接到这里的章节：

- [2.4 Causality 与 KV Cache](../models/02-transformer-anatomy/04-causality-and-kv-cache.md)：prefill / decode 算术强度分野（§6）是 roofline 入口；roofline 的逐卡参数与测量方法归本维度
- [4.1 KV 压缩与精确 Attention 加速](../models/04-modern-architectures/01-kv-compression-and-exact-attention.md)：FlashAttention 数学讲完后的 kernel 细节
