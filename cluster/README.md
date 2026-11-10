# cluster 维度：集群与通信（Cluster）

> 状态：未开工（占位）。本文件只声明范围与接口；正式章节设计另起 spec。

## 范围

本维度讲跨机跨卡的通信层：

- EP（expert parallelism）：MoE all-to-all dispatch / combine 的通信实现与拥塞行为
- Ring Attention：序列分块跨设备的通信细节（compute / communication overlap、拓扑映射）
- 集合通信与网络拓扑：all-reduce / all-to-all 的带宽账、NVLink / InfiniBand 层级

边界：MoE 路由公式归 models 4.5、Ring Attention 分块机制归 models 4.6；通信实现归这里。划分依据见 [models 维度 spec §2](../docs/superpowers/specs/2026-11-09-models-dimension-design.md)。

## 与 models 的接口

models 侧交叉链接到这里的章节：

- [4.5 MoE 与 MTP](../models/04-modern-architectures/05-moe.md)：EP 的 all-to-all dispatch 公式级描述之后，通信细节在这里
- [4.6 Long Context](../models/04-modern-architectures/06-long-context.md)：Ring Attention 分块跨设备机制图的通信侧
