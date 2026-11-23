# 9.2 其他推理引擎对照与选型（Other Inference Engines: Comparison and Selection）

> vLLM / SGLang / DeepInfer 之外，TensorRT-LLM 与 LMDeploy 各自押注什么？给定模型、硬件、特性需求与团队栈，选型依据是什么？本章是短篇对照章：三主引擎只留定位 recap + 走读指针，PD 分离只 recap 8.4，增量限定为两个引擎的当前形态核实与选型决策表 | 前置：P5（5.1–5.4 vLLM 走读）、P6（6.1–6.3 SGLang 走读）、P7（7.1–7.4 DeepInfer 与 DeepSeek 栈）、8.4（PD 分离系统对照）、4.1（continuous batching）、3.2（权重量化）、3.4（融合、图捕获与编译） | 锚点：TensorRT-LLM（[NVIDIA, TensorRT-LLM](../references.md)，教学参照 + 官方实现，main a41224a3，2026-11-24 contents API 核实）、LMDeploy（[Zhang et al., 2025](../references.md)，arXiv:2508.15601，[A]，EuroSys 2027 + InternLM/lmdeploy main 517cb6d7，2026-11-24 核实）；vLLM / SGLang / DeepInfer 定位 recap 挂各自走读章锚点

## 1. 对照框架：定位一层，选型一层

五个引擎都能跑 LLM serving，为什么不能拿一张功能清单打勾就选型？因为功能清单抹掉了三类真实差异：硬件锁定（NVIDIA 专属还是双平台）、生态绑定（模型族 / 编排栈 / 社区插件）、开源完整度（引擎在不在 release、编排要不要自建）。本章把这三类差异对到具体证据上，再给决策表。

**两层走读纪律**（spec §4.2）：vLLM / SGLang / DeepInfer 的模块全量走读已在 P5 / P6 / P7 交付，本章重访一律 recap + 指针；PD 分离的六系统对照已在 8.4 交付，本章只 recap 一句 + 选型定位，不双展开（C-S3-4）。增量只有两件事：TRT-LLM 与 LMDeploy 的当前形态核实（§2 / §3），以及五引擎选型决策表（§4）。

三主引擎定位 recap（各一句，走读指针随后）：

| 引擎 | 一句话定位 | 走读指针 |
| --- | --- | --- |
| vLLM | 通用生态最广的开源引擎：PagedAttention 分页 KV + continuous batching + connector 插件化 PD | P5（5.1 架构 / 5.2 调度与 KV / 5.3 分布式 / 5.4 特性） |
| SGLang | 结构化程序 + RadixAttention 前缀复用：前端语言与运行时协同设计，PD 引擎内置 | P6（6.1 架构与 radix / 6.2 overlap scheduler / 6.3 分布式与 PD） |
| DeepInfer | DeepSeek 模型族专用生产引擎开源版：SM100 + Ascend 950 双平台，编排层不在 release | P7（7.1 架构 / 7.2 DSpark / 7.3 kernel 栈 / 7.4 大规模 EP）；release 边界见 7.1 命题 1、8.4 §6 |

PD 分离选型 recap 一句：直传路线（vLLM connector / SGLang 内置 / TRT-LLM Python transceiver V2 + NIXL）与存储中转路线（DeepInfer 3FS）的账形态差异、编排三型与六轴对照，全部归 8.4 §8 选型论断表，本章不重复。

**边界声明**（spec §2）：llama.cpp / MLC-LLM 类端侧与 CPU 推理不覆盖；非 LLM serving（扩散生图 / 视频 / 语音的部署）不覆盖；文本扩散 LLM 只作 9.4 展望一句。

**本章口径**：对照章不引入新机制推导；所有性能数字为官方披露或论文口径（逐处标注来源与核实日期，2026-11-24），不作跨引擎实测排名；「× vLLM」类倍率是特定 workload 下的披露值，不是普适结论。

## 2. TensorRT-LLM：当前是 PyTorch-native，编译路线已历史化

TRT-LLM 还是「把模型编译成 TensorRT engine」的路线吗？**不是**——这是本章最大的漂移坑。当前 main = **PyTorch-native**：README L290 逐字「Architected on PyTorch」，模型以 native PyTorch 定义在 `tensorrt_llm/_torch/models/`，执行走 PyExecutor（[NVIDIA, TensorRT-LLM](../references.md)，教学参照 + 官方实现，main a41224a3，2026-11-24 contents API 核实）。

> **坑**：按 spec 大纲字面或 legacy 文档把 TRT-LLM 写成「TRT 编译器路线」。触发条件：读 legacy/architecture/workflow.md 或旧版 convert_checkpoint + trtllm-build 教程；现象：把已移除的两步 engine-build 路线当当前形态。事实（workflow.md 逐字）：「The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.」规避：当前形态写 PyTorch-native + 手写 kernel 混合；「编译路线」只作历史身份（曾走全图 AOT 编译，现移除），新项目入口是 trtllm-serve / LLM Python API。

当前执行结构（docs/source/developer-guide/overview.md，2026-11-24 核实）：PyExecutor(Worker) 每 rank 一个后台循环，循环内四件 = Scheduler / KVCacheManager / ModelEngine / Sampler——与 vLLM EngineCore busy loop（5.1 §3）、SGLang scheduler 事件循环（6.1 §3）同族。两条路线的结构对照：

```
当前 main（PyTorch-native）
────────────────────────────────────────────────
Python LLM API / trtllm-serve
   │  native PyTorch 模型（tensorrt_llm/_torch/models/）
   ▼
PyExecutor (Worker)：每 rank 后台循环
   ┌─────────────────────────────────────────┐
   │ Scheduler → KVCacheManager →            │
   │ ModelEngine → Sampler → （下一 iteration）│
   └─────────────────────────────────────────┘
   IFB = continuous batching；Overlap Scheduler 默认开

legacy 路线（已移除，文档仅留 cross-reference）
────────────────────────────────────────────────
HF checkpoint → convert_checkpoint → trtllm-build
   → TRT engine（全图 AOT 编译产物）→ runtime
```

调度与执行面三个事实（均官方文档口径）：

- **inflight batching（IFB）**：docs/source/features/paged-attention-ifb-scheduler.md 的定义 = continuous / iteration-level batching、packed no-padding、context 序列排 generation 前（口径：IFB 是 continuous batching 的 TRT-LLM 命名，不是独立新调度机制；机制账归 4.1，Orca iteration-level 源头同族）。
- **Overlap Scheduler 默认开**（developer-guide/overview.md）：GPU 第 n+1 步与 CPU 第 n 步重叠，与 SGLang zero-overhead scheduler / vLLM 同族（6.2 recap）；CUDA Graph padding 至多 22% e2e 吞吐（官方披露口径「up to a 22% end-to-end throughput increase」，workload 依赖）。
- **Blackwell 支持**（supported-hardware.md）：B200 / GB200 / B300 / GB300 / DGX Spark，另列 Hopper H100 / H200 / GH200、Ada L20 / L40 / L40S、Ampere A100——NVIDIA 专属硬件矩阵，新代际首发支持是它的选型筹码。

disagg 一句选型定位：当前 = Python transceiver KvCacheTransceiverV2 + backend NIXL + KV cache manager V2，bounce buffer 聚合小 fragment，编排走 Dynamo / trtllm-serve / Triton BLS 三路线——实现细节与漂移坑全部 recap 8.4 §5，本章不展开。

**编译 vs 手写 kernel 路线对照**（spec 关键对照，增量在「趋同」论断）：

| 轴 | vLLM / SGLang | TRT-LLM 当前 | TRT-LLM legacy（已移除） |
| --- | --- | --- | --- |
| 模型定义 | PyTorch（5.1 / 6.1） | native PyTorch（_torch/models/） | HF checkpoint 转换 |
| 执行 | EngineCore 循环 / 事件循环 | PyExecutor 每 rank 循环 | TRT engine runtime |
| 编译 | piecewise torch.compile + 手写 kernel（3.4 / 6.3 §8） | PyTorch-native + 手写 kernel 混合 | 全图 AOT 编译（trtllm-build） |
| batching | continuous（4.1） | IFB = continuous | IFB（同名） |

**命题 1**（执行模型趋同）：TRT-LLM 当前 main 的执行模型与 vLLM / SGLang 同族——每 rank 后台循环 + iteration-level batching + CPU/GPU 重叠调度；「编译 vs 手写 kernel」不再是它与两引擎的执行模型分界。**证明**（要点，逐项对照）：循环结构对位上表第二行（PyExecutor ↔ EngineCore ↔ 事件循环）；batching 对位第四行（IFB 定义逐字 = continuous / iteration-level）；重叠调度对位 Overlap Scheduler 与 6.2 同族。三项全中，故执行模型同族；分界移到生态与硬件矩阵（Dynamo / Triton 编排栈、NVIDIA 专属、Blackwell 首发）。边界：命题限当前 main——legacy 全图 AOT 路线确实是另一种执行模型（编译产物驱动，非 Python 循环），已移除，只作历史对照。证毕。

选型定位一句：**NVIDIA 专属 + PyTorch-native 编译/手写 kernel 混合 + Dynamo / Triton 生态**（§4 表第 4 行）。

## 3. LMDeploy：TurboMind 与混合精度部署

LMDeploy 的论文级贡献是什么？[Zhang et al., 2025](../references.md)（arXiv:2508.15601，[A]，EuroSys 2027；发表状态一手出处 = repo README L25 [2026/08] news，arXiv metadata 无 journal_ref——口径注）给出 TurboMind 的双 hardware-aware 混合精度管线：GEMM 侧 offline weight packing + online acceleration，attention 侧不同 Q/K/V 精度组合；配四技术（hardware-aware weight packing / adaptive head alignment / instruction-level parallelism / KV memory loading pipeline）。评测覆盖 16 LLM × 4 GPU 架构，混合精度负载至多 61% 更低延迟（均值 30%）/ 156% 更高吞吐（均值 58%）——**对照基线是现有混合精度框架，不是 vLLM 通用部署**（论文口径，勿混读）。代入教学值看量级：基线延迟 200 ms 时，至多降 61% → 78 ms、均值降 30% → 140 ms；吞吐 +156% = 基线的 2.56 倍。

**TurboMind 引擎**（docs/en/inference/turbomind.md，2026-11-24 contents API 核实，main 517cb6d7）：基于 NVIDIA FasterTransformer 的 C++ 引擎。核心调度特性 persistent batch：

> **坑**：把 persistent batch 当独立于 continuous batching 的新调度机制。触发条件：按名字直译「持久批」，以为是静态 batch 或新范式；现象：选型表里给它单列一格「新机制」。事实（turbomind.md 逐字）：「You may recognize this feature as 'continuous batching' in other repos」——persistent batch 是 continuous batching 的 LMDeploy 命名：N 个预配置 batch slots，请求有空槽即加入、生成完释放槽复用、cache-hit 历史 token 免重解码、batch 自动增减以缩减计算。规避：机制账一律回 4.1（iteration-level 调度），本节只登记命名差异与 slot 实现形态。

KV cache manager：memory-pool + LRU 的「cache of KV caches」——被驱逐序列转存 token IDs，miss 时 FMHA 重解码恢复（口径：recompute 式抢占恢复，与 vLLM V1 recompute-only 抢占同族，成本账回链 4.2 §2）；cutlass FMHA 支持 Q/K 不等长 + INT8 KV。

**W4A16 AWQ 部署**（docs/en/quantization/w4a16.md）：所谓 W4A16，是指权重 4 bit、激活 16 bit 的 weight-only 量化部署形态（AWQ 算法机制——salient 通道 per-channel scaling 等价变换——recap 3.2，不重推）。TurboMind 支持 AWQ / GPTQ INT4 推理，量化模块只实现 AWQ（`lmdeploy lite auto_awq` 命令）；硬件覆盖 V100（sm70）/ Turing / Ampere / Ada——老代际卡是它的差异化阵地。权重显存账：

$$
\mathrm{Mem}_{\mathrm{W4A16}} \approx N \times 0.5 \ \mathrm{bytes} \qquad (1)
$$

**符号**： $N$ = 参数量；0.5 bytes = INT4 每权重元素字节数（4 bit）；激活侧 A16 不计入权重账。代入 7B： $N = 7 \times 10^{9}$ → 3.5 GB，对照 bf16 的 14 GB 省 4×（理想账；AWQ group scale 开销使实际略高，group 128 口径见 3.2）。V100 32 GB 单卡装 7B W4A16 权重后仍余 28 GB 量级给 KV 与激活——这就是「老卡 + 混合精度」选型的账面依据。

**InternLM 生态**（README 官方披露口径）：internlm2_5-7b-chat 示例 / InternVL2 / InternLM-XComposer2.5 一等支持；internlm2-20b 16+ RPS、1.8× vLLM（特定 workload 披露值，非普适倍率）。生态外延三条：双引擎 TurboMind + PyTorchEngine（纯 Python，Ascend 支持）；MXFP4 支持 NVIDIA V100+（gpt-oss H800 1.5× vLLM，README 披露口径）；DeepSeek PD disagg 经 DLSlime + Mooncake（一句登记，PD 账归 8.4）。

选型定位一句：**InternLM 生态 + 混合精度 W4A16 部署 + FasterTransformer lineage**（§4 表第 5 行）。

## 4. 选型决策表：模型 × 硬件 × 特性 × 团队

五个引擎摆在面前，选哪个？选型不是「哪个引擎最强」，而是「哪条约束先咬合」——硬件锁定、模型族生态、特性需求、团队栈四类约束里，最硬的那条直接划掉多数候选。下表论断式给出五引擎的咬合场景（每行回指证据节；三主引擎行的依据在走读章，本章只 recap 定位）：

| 场景（模型 × 硬件 × 特性 × 团队） | 选 | 论断依据 | 回指 |
| --- | --- | --- | --- |
| 通用开源模型 + 无专属硬件锁定 + 要社区插件 / 特性广度 + 团队栈未定 | vLLM | 通用生态最广：特性层（投机 / 结构化 / LoRA / offload）与 connector 插件化 PD | §1 表、P5（5.4 特性决策表） |
| 结构化程序 / 多调用共享前缀（agent、few-shot fork）+ 前缀复用率敏感 | SGLang | RadixAttention + 前端语言协同设计；PD 引擎内置省一层编排 | §1 表、P6（6.1 / 2.2）、8.4 §3 |
| DeepSeek 模型族（V3 / V4.1）+ SM100 或 Ascend 950 + 大规模 EP + 团队能自建编排 | DeepInfer | 模型族专用 kernel 栈 + 双平台；代价：编排层不在 release | §1 表、P7（7.3 / 7.4）、7.1 命题 1、8.4 §6 |
| NVIDIA 专属栈 + Dynamo / Triton 编排生态 + Blackwell 新代际首发 | TRT-LLM | PyTorch-native + 手写 kernel 混合；B200 / GB200 / B300 / GB300 / DGX Spark；勿按 legacy 编译路线文档 | §2、命题 1、8.4 §5 |
| InternLM 生态 + W4A16 混合精度 + V100 级老卡或 Ascend（PyTorchEngine） | LMDeploy | TurboMind FasterTransformer lineage + AWQ INT4 + 论文级混合精度管线 | §3、式 (1) |
| PD 分离生产选型（直传 vs 存储中转、编排三型） | 按 8.4 六轴表 | 本章不双展开 | 8.4 §8 |

两条边界论断。其一，**硬件轴上 TRT-LLM 与 DeepInfer 是两种不同的锁定**：前者锁 NVIDIA 全代际（Ampere 到 Blackwell），后者锁 DeepSeek 模型族 × 双平台（SM100 + Ascend 950）——「NVIDIA 卡上跑 DeepSeek」两个候选都合法，分界在模型族专用优化与编排自建成本（7.1 / 8.4 §6）。其二，**「× vLLM」倍率全部是 workload 条件披露**（internlm2-20b 1.8×、gpt-oss 1.5×、论文 61%/156% 对混合精度基线），跨 workload 外推无效；生产选型以自有 workload 实测为准（benchmark 方法学与陷阱归 1.1）。

## 总结

| 论断 | 证据 / 账 | 回指 |
| --- | --- | --- |
| TRT-LLM 当前 = PyTorch-native（PyExecutor 每 rank 循环 + native PyTorch 模型）；legacy TensorRT 编译后端已移除 | README L290 逐字 + workflow.md 逐字（removed / cross-reference only） | §2、坑注 |
| 「编译 vs 手写 kernel」不再是 TRT-LLM 与 vLLM / SGLang 的执行模型分界；分界在生态与硬件矩阵 | 命题 1（循环 / batching / 重叠三项对位） | §2、练习 2 |
| IFB = continuous / iteration-level batching 的 TRT-LLM 命名，非新机制 | paged-attention-ifb-scheduler.md 定义 | §2、4.1 |
| TRT-LLM 硬件矩阵 NVIDIA 专属：Blackwell B200 / GB200 / B300 / GB300 / DGX Spark + Hopper / Ada / Ampere | supported-hardware.md | §2 |
| persistent batch = continuous batching 的 LMDeploy 命名（N slots），非新机制 | turbomind.md 逐字 | §3、坑注、练习 1 |
| LMDeploy 论文贡献 = 双 hardware-aware 混合精度管线 + 四技术；至多 61% 降延迟 / 156% 提吞吐，基线是混合精度框架 | arXiv:2508.15601 v2（EuroSys 2027，README L25） | §3 |
| W4A16 权重账 ≈ N × 0.5 bytes：7B → 3.5 GB（bf16 的 1/4），V100 级老卡可部署 | 式 (1)；AWQ 机制 recap 3.2 | §3、练习 3 |
| LMDeploy KV 驱逐 = recompute 式（转 token IDs、miss 重解码），与 vLLM V1 抢占同族 | turbomind.md KV manager | §3、4.2 §2 |
| 选型 = 最硬约束先咬合：硬件锁定 / 模型族生态 / 特性 / 团队栈；「× vLLM」倍率皆 workload 条件披露 | 决策表六行 + 两条边界论断 | §4、练习 1/3 |
| 三主引擎与 PD 系统对照只 recap + 指针，不双展开 | 两层走读纪律（spec §4.2，C-S3-4） | §1、8.4 §8 |

## 练习

1.（论证 · [10]）口径判别四连：(a) 「TRT-LLM 当前走 TRT 全图编译路线」对不对；(b) 「persistent batch 是独立于 continuous batching 的新调度机制」对不对；(c) 「LMDeploy 论文 61% 延迟降低是相对 vLLM 通用部署」对不对；(d) 「IFB 与 4.1 的 iteration-level scheduling 是同族机制」对不对。逐条给判定 + 一句依据。校验：(a)(b)(c) 均错、(d) 对——与 §2 / §3 两处坑注及论文口径注逐字一致。
2.（论证 · [20]）命题 1 证明闭环（执行模型趋同）：(a) 补全三项对位表——PyExecutor 循环 ↔ vLLM ↔ SGLang 各对到哪个文件级结构（走读章指针）；(b) 说明为什么「编译 vs 手写 kernel」不再是当前分界、分界移到了哪两轴；(c) 给边界反例：legacy 路线在什么意义上仍是另一种执行模型，为什么命题必须限定当前 main。校验：(a) 与 §2 对照表及 5.1 §3 / 6.1 §3 / 6.2 一致；(c) 与 workflow.md 逐字声明一致。
3.（代码 · [30]）repo 走读与口径核对（2026-11-24 contents API 通道）：(a) TRT-LLM supported-hardware.md 列出 Blackwell 五个 SKU，workflow.md 找到 legacy 移除声明逐字；(b) LMDeploy turbomind.md 找到 persistent batch 的 continuous batching 逐字对照句，w4a16.md 确认量化命令与硬件代际；(c) 先预测再核对：internlm2-20b「1.8× vLLM」是什么口径——普适结论还是 workload 条件披露？README 里找出处。校验：(a) supported-hardware.md Blackwell 节五 SKU 与 workflow.md「removed … cross-reference only」逐字；(b) `lmdeploy lite auto_awq` + V100(sm70)/Turing/Ampere/Ada；(c) §4 边界论断二（16+ RPS workload 披露值）。

## 参考

- [NVIDIA, TensorRT-LLM](../references.md)：key: tensorrt-llm，教学参照 + 官方实现（main a41224a3，2026-11-24 contents API 深核）——README L290「Architected on PyTorch」；docs/source/developer-guide/overview.md PyExecutor(Worker) 每 rank 循环（Scheduler / KVCacheManager / ModelEngine / Sampler）+ Overlap Scheduler 默认开 + CUDA Graph padding 至多 22%（repo 另有 docs/source/overview.md 产品页与 torch/arch_overview.md，同名易混，内容锚定 developer-guide 版）；tensorrt_llm/_torch/models/ native PyTorch 模型；legacy/architecture/workflow.md legacy TensorRT backend removed 逐字；docs/source/features/paged-attention-ifb-scheduler.md IFB 定义；supported-hardware.md Blackwell B200/GB200/B300/GB300/DGX Spark + Hopper/Ada/Ampere；disagg Python transceiver V2 + NIXL（recap 8.4 §5）
- [Zhang et al., 2025](../references.md)：LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind，arXiv:2508.15601，[A]（key: lmdeploy，EuroSys 2027，v2 2026-05-15；发表状态一手 = README L25 [2026/08] news）——双 hardware-aware 混合精度管线 + 四技术 + 16 LLM × 4 GPU + 至多 61% 降延迟（均值 30%）/ 156% 提吞吐（均值 58%）vs 现有混合精度框架；repo main 517cb6d7：docs/en/inference/turbomind.md（FasterTransformer lineage + persistent batch = continuous batching 逐字 + KV manager LRU recompute + cutlass FMHA INT8 KV）、docs/en/quantization/w4a16.md（AWQ/GPTQ INT4、lmdeploy lite auto_awq、V100/Turing/Ampere/Ada）、InternLM 生态与 MXFP4 / PyTorchEngine / DLSlime+Mooncake 披露
- [Kwon et al., 2023](../references.md)：vLLM，arXiv:2309.06180，[A]（key: vllm-repo）——三主引擎定位 recap 锚点，模块走读归 P5（5.1–5.4），本章不重复展开
- [Zheng et al., 2023](../references.md)：SGLang，arXiv:2312.07104，[A]（key: sglang-repo）——三主引擎定位 recap 锚点，模块走读归 P6（6.1–6.3），本章不重复展开
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——DeepSeek 专用 + SM100/Ascend 950 双平台定位；release 边界（编排层不在）见 7.1 命题 1 与 8.4 §6；走读归 P7
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方披露——DeepInfer 生产形态口径（PD 配比 / 存储中转）recap 归 8.4 §6 与 9.1，本章只定位引用
