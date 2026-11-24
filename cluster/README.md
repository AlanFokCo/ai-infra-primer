# cluster 维度：平台编排（Cluster）

> 状态：未开工（占位）。本文件只声明范围与接口；正式章节设计另起 spec。

## 范围

本维度讲 GPU 集群的平台编排层（2026-11-24 用户裁定重定义；原通信 scope——EP all-to-all、Ring Attention 通信细节、集合通信与网络拓扑、NVLink/IB 层级——全部并入 gpu 维度，见 [gpu/README.md](../gpu/README.md)）：

- K8S GPU 调度：device plugin、调度策略与拓扑感知分配
- 容器运行时：GPU 容器化与运行时栈
- Agent Sandbox 隔离：沙箱机制与资源边界
- 多租户编排：配额、隔离与共享
- 集群作业调度：作业队列、抢占与弹性

边界：硬件机制与通信全栈（节点内 NVLink → NCCL → RDMA → 机间 fabric → 大规模通信工程）归 gpu，平台编排归 cluster。MIG/vGPU/SR-IOV 的**硬件分区机制**归 gpu 11.1，K8S device plugin 与调度策略归这里。划分依据见 [gpu 维度 spec §2](../docs/superpowers/specs/2026-11-24-gpu-dimension-design.md)。

## 与 gpu 的接口

gpu 侧交叉指引到这里（纯文字，不放死链）：

- gpu 11.1 GPU 服务器硬件形态：MIG/vGPU 硬件分区之后，K8S device plugin 与调度策略在这里
- gpu 11.3 故障与可靠性：GPU 硬件故障诊断之后，集群级故障编排在这里
