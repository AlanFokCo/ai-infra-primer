# gpu W0 research pass — references 种入证据记录（2026-11-24）

Task 2：`gpu/references.md` 骨架 + 种入 ≥50 条。本文件记录核实通道、逐源证据、⟨verify⟩ 残留延迟项、网络 fallback、parse-check 与污染检查。种入自 gpu spec 附录「Spec 预核实记录」+ §5 各章 Sources 意图。

## 0. 结果摘要

- **parse-check（pure ESM，controller PREFLIGHT RULING 2）**：`groups 11 entries 59`（≥50 地板过）。
- **per-Part**：P1=9 / P2=2 / P3=1 / P4=0 / P5=9 / P6=10 / P7=6 / P8=6 / P9=6 / P10=4 / P11=6 = 59。
- **证据等级分布**：[C]=27 / 教学参照=16 / [A]=10 / [B]=5 / [W]=1。
- **污染检查全过**：59 数据行均恰 7 字段；无 escaped `\|`；无重复 key；日期均 YYYY-MM-DD（2026-11-24）；等级单值无 [A/B] 混合；citedBy 格式合规；key 均小写 slug；无空字段。
- P4 空表说明：P4（kernel 访存优化）锚点均为跨 Part 共享（Best Practices / PTX ISA / Nsight Compute / CUTLASS 首引 P1，cuda-samples 首引 P3），按「收录首引 Part、被引跨组累加」规则本组无独立首引条目，4.1-4.4 被引在对应条目累加。

## 1. arXiv ID 核实（https export API 批量 200，10/10）

命令：`curl "https://export.arxiv.org/api/query?id_list=2205.14135,2307.08691,2407.08608,2408.11743,2408.12757,2504.17577,2501.01005,2309.06180,2412.19437,2310.01889&max_results=10"` → HTTP 200 / 36630 B。ID↔标题↔第一作者逐一对上：

| arXiv ID | 版本 | 第一作者 | 标题 | key |
| --- | --- | --- | --- | --- |
| 2205.14135 | v2 | Tri Dao | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | flashattention |
| 2307.08691 | v1 | Tri Dao | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | flashattention2 |
| 2407.08608 | v2 | Jay Shah | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | flashattention3 |
| 2408.11743 | v1 | Elias Frantar | MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models | marlin |
| 2408.12757 | v2 | Kan Zhu | NanoFlow: Towards Optimal Large Language Model Serving Throughput | nanoflow |
| 2504.17577 | v2 | Lei Wang | TileLang: A Composable Tiled Programming Model for AI Systems | tilelang |
| 2501.01005 | v2 | Zihao Ye | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | flashinfer |
| 2309.06180 | v1 | Woosuk Kwon | Efficient Memory Management for Large Language Model Serving with PagedAttention | vllm |
| 2412.19437 | v2 | DeepSeek-AI | DeepSeek-V3 Technical Report | deepseekv3 |
| 2310.01889 | v4 | Hao Liu | Ring Attention with Blockwise Transformers for Near-Infinite Context | ring-attention |

- **Ring Attention 2310.01889 ⟨verify W5⟩ 提前销项**：W0 export API 再核 v4 标题作者全中，沿 training/references.md L114 已核实 [A]（2026-11-17）+ spec C-10 裁定 [B]→[A]。
- 网络 note：本轮 https export API 批量可用（沿 inference W5/W6 恢复记录），未触 429，无需 abs/HTML fallback。

## 2. GitHub repo 核实（API，22 repo）

命令：`curl https://api.github.com/repos/<owner>/<repo>`，取 HTTP code + full_name + default_branch。

| repo | code | full_name | branch | note |
| --- | --- | --- | --- | --- |
| NVIDIA/cutlass | 200 | NVIDIA/cutlass | main | |
| NVIDIA/nccl | 200 | NVIDIA/nccl | master | |
| NVIDIA/nvbandwidth | 200 | NVIDIA/nvbandwidth | main | |
| NVIDIA/gdrcopy | 200 | NVIDIA/gdrcopy | master | |
| NVIDIA/nccl-tests | 200 | NVIDIA/nccl-tests | master | |
| NVIDIA/DCGM | 200 | NVIDIA/DCGM | master | |
| NVIDIA/cuda-samples | 200 | NVIDIA/cuda-samples | master | |
| NVIDIA/open-gpu-kernel-modules | 200 | NVIDIA/open-gpu-kernel-modules | main | nvidia-peermem GDR 内核锚 |
| Dao-AILab/flash-attention | 200 | Dao-AILab/flash-attention | main | |
| deepseek-ai/DeepEP | 200 | deepseek-ai/DeepEP | main | |
| deepseek-ai/FlashMLA | 200 | deepseek-ai/FlashMLA | main | |
| deepseek-ai/DeepGEMM | 200 | deepseek-ai/DeepGEMM | main | |
| IST-DASLab/marlin | 200 | IST-DASLab/marlin | master | |
| NVIDIA/cub | 200 | NVIDIA/cub | main | **archived=true**，description 自链 NVIDIA/cccl |
| NVIDIA/cccl | 200 | NVIDIA/cccl | — | CUB 现归属（cub 条目锚此） |
| openai/triton | **301** | → triton-lang/triton | main | 跟随重定向 200，full_name=triton-lang/triton |
| linux-rdma/rdma-core | 200 | linux-rdma/rdma-core | master | |
| NVIDIA/TransformerEngine | 200 | NVIDIA/TransformerEngine | main | |
| pytorch/pytorch | 200 | pytorch/pytorch | main | |
| efeslab/Nanoflow | 200 | efeslab/Nanoflow | Nanoflow-python | |
| tile-ai/tilelang | 200 | tile-ai/tilelang | main | |
| flashinfer-ai/flashinfer | 200 | flashinfer-ai/flashinfer | main | |
| vllm-project/vllm | 200 | vllm-project/vllm | main | |

- **组织迁移注**：openai/triton 301→triton-lang/triton（条目锚新 org + 重定向注）；NVIDIA/cub archived→并入 NVIDIA/cccl（cub 条目锚 cccl + archived 注）。
- 12 repo 200 与 spec 附录预核实记录一致；本轮另核 cub/cccl/triton/rdma-core/TransformerEngine/pytorch/Nanoflow/tilelang/flashinfer/vllm 等待核 repo（spec ⟨verify 各波⟩ 项 W0 提前销项）。
- 网络 note：GitHub core API 本轮可达（沿 inference W1 恢复记录），未触限流，无需 codeload/trees fallback。

## 3. NVIDIA 官方文档 URL 核实（HTTP curl，-L 跟随重定向）

| URL | code | key / note |
| --- | --- | --- |
| docs.nvidia.com/cuda/parallel-thread-execution/ | 200 | ptx-isa |
| docs.nvidia.com/cuda/cuda-c-best-practices-guide/ | 200 | cuda-best-practices |
| docs.nvidia.com/deeplearning/nccl/user-guide/docs/ | 200 | nccl-docs |
| docs.nvidia.com/cuda/gpudirect-rdma/ | 200 | gpudirect-rdma-docs |
| resources.nvidia.com/en-us-blackwell-architecture | 200 | blackwell-whitepaper（landing） |
| resources.nvidia.com/en-us-hopper-architecture | 200 | hopper-whitepaper（landing） |
| nvidia.com/en-us/data-center/h100/ | 200 | nvidia-datasheet |
| nvidia.com/en-us/data-center/h200/ | 200 | nvidia-datasheet |
| resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23 | 200 | nvidia-datasheet（H100/H200 PDF 落地页） |
| dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf | 200 | nvidia-datasheet（Blackwell datasheet PDF） |
| **docs.nvidia.com/cuda/cuda-c-programming-guide/** | **404** | 旧 URL（spec 附录已记 404） |
| **docs.nvidia.com/cuda/cuda-programming-guide/** | **200** | **cuda-programming-guide 现行 URL，⟨verify W1⟩ 销项** |
| docs.nvidia.com/cuda/ | 200 | CUDA docs 根（互证） |
| docs.nvidia.com/datacenter/tesla/mig-user-guide/ | 200 | mig-user-guide（⟨verify W6⟩ 提前销项） |
| docs.nvidia.com/nsight-compute/NsightCompute/index.html | 200 | nsight-compute |
| docs.nvidia.com/nsight-compute/ProfilingGuide/index.html | 200 | nsight-compute（profiling guide） |
| docs.nvidia.com/nsight-systems/UserGuide/index.html | 200 | nsight-systems（⟨verify W6⟩ 提前销项） |
| docs.nvidia.com/deploy/xid-errors/index.html | 200 | xid-docs（⟨verify W6⟩ 提前销项） |
| opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf | 200 | ocp-mx（沿 inference 已核实） |
| docs.nvidia.com/gpudirect-storage/ | 200 | gds-docs |
| docs.nvidia.com/vgpu/ | 200 | vgpu-docs |
| nvidia.com/en-us/data-center/nvlink/ | 200 | nvlink-nvswitch |
| nvidia.com/en-us/data-center/gb200-nvl72/ | 200 | gb200-nvl72 |
| nvidia.com/en-us/data-center/hgx/ | 200 | hgx-dgx |
| developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/ | 200 | nccl-tree-blog（double binary tree） |
| developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | 200 | hopper-blog |
| docs.nvidia.com/cuda/cublas/ | 200 | cublaslt-docs |
| docs.nvidia.com/deploy/nvidia-smi/ | 200 | nvidia-smi-docs |
| docs.nvidia.com/networking/ | 200 | nvidia-networking-docs |
| docs.nvidia.com/networking/display/SHARPv300 | 200 | sharp-docs（⟨verify W5⟩ 提前销项） |
| **infinibandta.org/ibta-specifications-download/** | **404** | 旧 URL |
| **infinibandta.org/specifications/** | **404** | 旧 URL |
| **infinibandta.org/ibta-specification/** | **200** | **ibta-spec 现行 URL（root 页 href 自链），⟨verify W5⟩ 销项** |

## 4. 老经典无 arXiv 核实（Crossref + URL）

Crossref API `https://api.crossref.org/works/<DOI>` 确认标题 venue（bot 拦截时的权威互证通道）：

| key | DOI | Crossref 标题 | venue/年 | URL 可达性 |
| --- | --- | --- | --- | --- |
| blelloch-scan | —（CMU-CS-90-190） | Prefix Sums and Their Applications | CMU tech report 1990 | cs.cmu.edu/~guyb/papers/Ble90.pdf **200** |
| rabenseifner | 10.1007/978-3-540-24685-5_1 | Optimization of Collective Reduction Operations | LNCS（ICCS）2004 | link.springer.com **200** + 页面 title 逐字 |
| patarasuk-yuan | 10.1016/j.jpdc.2008.09.002 | Bandwidth optimal all-reduce algorithms for clusters of workstations | JPDC 2009 | doi.org **200**→linkinghub.elsevier.com/retrieve/pii/S0743731508001767 |
| dcqcn | 10.1145/2785956.2787484 | Congestion Control for Large-Scale RDMA Deployments | SIGCOMM 2015 | ACM DL 403；**conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf 200**（官方 proceedings，163822 B 有效 PDF） |
| al-fares | 10.1145/1402958.1402967 | A scalable, commodity data center network architecture | SIGCOMM 2008 | ACM DL 403 + PDF 镜像 000/404；**Crossref 确认标题 venue**，⟨verify W5 可达 PDF⟩ |

## 5. 模型 config 核实（hf-mirror）

- `https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json`：307→resolve-cache，-L 跟随 **200 / 3311 B**（与 inference W4 记录字节数一致）。
- 顶层 keys：architectures（DeepseekV41ForCausalLM）/ model_type（deepseek_v41）/ text_config / vision_config / quantization_config 等——**MLA 几何字段在 text_config 嵌套**。
- text_config 核实：num_hidden_layers 40 / num_attention_heads 64 / **head_dim 512** / **qk_rope_head_dim 64** / q_lora_rank 1280 / o_lora_rank 1024 / o_groups 8 / **无 kv_lora_rank 字段**（引擎导出 448 latent，512 = 448 + 64 rope，沿 inference W4 errata 口径；勿写 V3 576 = 512 + 64）。
- huggingface.co 本机不可达→hf-mirror 传输核实（沿 inference 已踩坑通道）。

## 6. ⟨verify⟩ 残留延迟项（来源真实、精确 URL/数值留对应波）

| 项 | key | W0 状态 | 留波 |
| --- | --- | --- | --- |
| JEDEC HBM3 精确出处 | jedec-hbm3 | jesd238a/b URL 403 bot 拦截；规范真实（JESD238 HBM3） | W1 |
| Hopper/Blackwell whitepaper + datasheet 精确数值 | hopper-whitepaper / blackwell-whitepaper / nvidia-datasheet | landing 200；Hopper PDF 镜像不可达（沿四路审查记录），单发射以 SM 框图 + BP Table 5 + Hopper blog 互证；B200 精确数值沿 inference 1.3 表 | W1 |
| PCI-SIG 精确出处 | pcie-spec | pcisig.com 403 bot 拦截；规范真实（Gen5 x16 = 64 GB/s 单向） | W4 |
| NVLink 代际带宽 / GB200 NVL72 精确规格 | nvlink-nvswitch / gb200-nvl72 | 产品页 200；精确代际值留 datasheet | W4 |
| Al-Fares fat-tree 可达 PDF | al-fares | ACM DL 403 + 镜像 000/404；Crossref 确认标题 venue | W5 |
| peer-memory 状态 | open-gpu-kernel-modules | 原 Mellanox/peer-memory 404 改锚 open-gpu-kernel-modules（nvidia-peermem）+ gdrcopy；原 repo 是否并入 MLNX_OFED/驱动包待核 | W5 |
| CUTLASS 当前版本架构 / swizzle-pipeline 路径 | cutlass | repo 200；3.x/4.x 架构 + include/cute 路径留 fresh tarball | W3 |
| TransformerEngine norm kernel 路径 | transformerengine | repo 200；具体 kernel 路径留 tarball | W3 |
| NVIDIA networking 精确子页（ConnectX/Quantum/verbs 手册） | nvidia-networking-docs | networking 根 200；精确子页留调研 | W5 |
| NCCL 当前版本 2.x / fresh tarball file:line | nccl | repo 200（master）；版本 + 走读锚留 fresh tarball | W4 |
| DeepEP fresh tarball 双锚 file:line | deepep | repo 200（main）；双锚 tag v1.2.1 + main V2.5 沿 inference 已锚，file:line 留 fresh tarball | W5 |
| FA4 是否存在 / flash-attention fresh tarball | flashattention | repo 200（main）；FA4⟨verify⟩ + file:line 留 tarball | W3 |

## 7. 网络 fallback 通道（本轮实际使用）

- huggingface.co 不可达 → **hf-mirror**（V4.1 config 307→resolve-cache -L 跟随 200）。
- ACM DL / JEDEC / PCI-SIG **403 bot 拦截** → **Crossref API**（标题 venue 权威互证）+ **doi.org** 解析 + **官方 proceedings PDF**（DCQCN conferences.sigcomm.org 200）。
- Hopper whitepaper PDF 镜像不可达 → **官方 landing 200** + Hopper blog / SM 框图 / Best Practices Table 5 互证（沿四路审查 A 路 fallback）。
- GitHub 组织迁移 301 → **-L 跟随重定向**取新 full_name（triton-lang/triton）；archived repo → description 自链新归属（NVIDIA/cub→cccl）。
- 本轮 arXiv https export API / GitHub core API 均可达，未触 429/限流，abs/HTML 与 codeload/trees fallback 未启用（备而不用）。

## 8. 种入纪律

- 跨 Part 共享锚点收录首引 Part、被引跨组累加（CUTLASS P1 被引 1.3/4.2/4.3/4.4/5.2；PTX ISA P1 被引 1.2…6.3；CUDA Guide P1 被引 1.2…7.3；nccl P8 被引 8.1…10.4；deepep P10 被引 9.2/10.3）。
- 等级单值（spec §4.1）：官方规格 [C]「官方规格披露·一手权威」；无 paper 官方 repo 教学参照（官方实现）；paper+code [A]；老经典无 arXiv [B] + 原始出处；模型 config [W]（仅 6.4 kernel 几何例引用）。无 [A/B] 混合档。
- 禁竖线：arXiv ID 与 URL、多 URL 之间用 ` ; ` 分隔；单元格内无 `|` 含 escaped `\|`。
- 被引章节按 spec §5 Sources 意图填写，各波 research/writing pass 回写精炼。
