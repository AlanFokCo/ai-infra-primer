# 7.3 DeepSeek 推理 Kernel 栈（DeepSeek Inference Kernel Stack）

> P3 把算子的算法级账推完了——每条公式落在哪个 repo、开源版相对内部版改了什么、低精度格式在 kernel 接口上怎么落地？本章把 FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect / Mega MoE 六件一手链接对回 P3 公式，交付映射总表与双平台口径 | 前置：3.1（decode attention 算子族：split-KV、吸收形 MLA、DSA 选择链）、3.2（量化格式字节账）、3.3（量化 GEMM / grouped / Mega MoE / DeepEP）、3.4（融合 / CUDA Graph / TileLang）、1.2（ $\kappa$ 几何与 MoE 流量账）、2.3（KV 格式账）、7.1（DeepInfer 架构与 release 边界）、7.2（DSpark 的 TileLang kernel 实例）；models 4.1 §4（MLA 低秩）与 4.2 §6（DSA indexer）只 recap 不重推 | 锚点：FlashMLA（[DeepSeek-AI, FlashMLA](../references.md)，教学参照 + 官方实现；main + 旧 commit ba89a34 双锚）、DeepGEMM / DeepEP / TileKernels / DeepSelect（[DeepSeek-AI, DeepGEMM](../references.md)、[DeepSeek-AI, DeepEP](../references.md)、[DeepSeek-AI, TileKernels](../references.md)、[DeepSeek-AI, DeepSelect](../references.md)，教学参照 + 官方实现）、TileLang（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）、DeepInfer repo + doc（[DeepSeek-AI, DeepInfer](../references.md)、[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）

## 1. kernel 栈全景与章脊口径

DeepInfer README L23 把优化 kernel 列为六件一手链接：FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect，加 custom TileLang ops（`deepinfer/model_executor/layers/tilelang_ops`）（[DeepSeek-AI, DeepInfer](../references.md)，教学参照，2026-11-23 fresh tarball 核实）。这六件各管 forward 的哪一段？所谓 kernel 栈（kernel stack），是指引擎把模型 forward 拆成算子后、每个算子背后实际执行的优化 kernel 库及其编译底座的集合。下图按 V4.1-Flash 的三条计算路径 + 编译底座 redraw（repo 结构锚 7.1，算子算法锚 P3）：

```
V4.1-Flash forward（model_executor/models/deepseek.py DeepseekForCausalLM L251）
├─ attention 路径
│   ├─ indexer 打分：DeepGEMM fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits
│   ├─ 选择：tilelang_ops/candidate_blocks.py（块预选）→ DeepSelect topk（K=512）
│   └─ 主 attention：FlashMLA sparse prefill / decode（FP8 / FP4 KV）
│       └─ fused norm-RoPE-attn-RoPE-cast（Q_b / Wv 预 permute + fp8_einsum）
├─ MoE 路径
│   ├─ routing：TileKernels top-k / scoring（v4_modules/moe.py TopKGate L40）
│   ├─ dispatch / combine：DeepEP EPBuffer（V2.5）
│   ├─ expert GEMM：DeepGEMM grouped（contiguous / masked）
│   └─ Mega MoE：DeepGEMM mega-kernel（dispatch + linear1/2 + SwiGLU + combine）
├─ 采样路径：Sampler（layers/sampler.py L59）→ DeepSelect topk（fp32）/ fused_sample
└─ 编译底座：TileLang DSL + TileKernels + DeepJIT + tilelang_ops（cuda/ + asc/ 双后端）
硬件：NVIDIA SM100 / SM103 + Ascend 950（vendor/ 六 submodule）
```

两条目录事实先登记（7.1 详展，此处一句）。其一，`vendor/` 含 DeepEP / DeepEP-Ascend / DeepGEMM / DeepGEMM-Ascend / DeepSelect / FlashMLA 六个 submodule——kernel 栈以 submodule 形式钉进引擎树（2026-11-23 tarball 核实）。其二，`deep_cluster/`（topology/bind + hardware/{nvidia,ascend}）是硬件拓扑绑定，属大纲外新增，本章不展开。Ascend 变体的分列口径：DeepGEMM-Ascend / DeepEP-Ascend 另立 repo（2026.09.30 发布、同 API），FlashMLA / TileKernels / DeepSelect 的 Ascend 支持在主 repo 内（双后端运行时选择）——§3 / §5 / §6 / §7 按此分列。

**本章口径**：模型几何取 V4.1-Flash config（[W]： $H_{kv}$=1、共享 KV 向量 512 维 = kv_lora_rank 448 + rope 64（config `head_dim` 512 **含** rope，引擎导出 kv_lora_rank=448，读法见 §3）、index_head_dim 128、index_topk 512、40 层 + 3 nextn、384 routed + 1 shared top-6）；硬件取 SM100（B200）与 Ascend 950 双平台；性能数字一律 README / doc 官方口径，生产数字标「官方披露」+ doc 行号 + 核实日期 2026-11-23。边界声明：本章讲算子的算法、tiling / 分裂策略、IO 与算力账、引擎调用接口；warp / tensor core / SASS 指令级实现归 gpu 维度，RDMA / NVLink 硬件层归 cluster 维度。V4.1-Flash 模型侧形式化（CED / Engram / DSpark draft / compress_ratios）归 models 维度（V4.1 待补），本章只引 config 字段做 serving 账。

## 2. P3 公式 ↔ repo 映射总表

P3 四章交付了算子的算法级账，本章不重推——映射怎么读？维度内两层走读纪律（spec §4.2）：3.x 是算法级（公式推导 + 机制例证片段），7.3 是映射级（每条公式对回 repo 文件 / 函数 + 版本口径 + 双平台落点）。重访 3.x 已展示的片段一律 recap 一句 + 指引，增量限定为「落在哪、版本是什么、开源版差多少」。下表是本章的脊：

| P3 锚（算法级） | recap 一句 | repo 落点（映射级） | 本章增量 |
| --- | --- | --- | --- |
| 3.1 式 (6)(7)、命题 3 | 吸收形 MLA：score 打在 cached latent、V 侧先加权和再上投影 | FlashMLA sparse decode（main，V4.1 FP8/FP4 KV）；Hopper 机制锚 ba89a34 | §3 |
| 3.1 §2、命题 1 | split-KV 把 KV 维变并行维、LSE 归约正确 | FlashMLA splitkv_mla + combine（策略级） | §3 |
| 3.1 式 (9)、§7 选择链 | DeepSelect 期望处理量界 + 块预选两级 | DeepSelect topk + tilelang_ops/candidate_blocks.py + DeepGEMM mqa_logits | §5 / §7 |
| 2.3 式 (4)、3.2 §2 格式表 | 每元素字节 = (数据 + scale) / 块大小 | FlashMLA 528/288 KV + DeepGEMM packed UE8M0 + TileKernels cast/dequant | §4 |
| 3.3 式 (3)(4)、命题 2 | grouped GEMM 只 group M 轴；contiguous vs masked | DeepGEMM m_grouped_fp8_gemm_nt_contiguous / _masked | §5 |
| 3.3 式 (5)、命题 3 | EP 扩张拐点 $\tau(e) = \max(A/e, F) + g(e)$ | DeepGEMM Mega MoE（组件）+ doc L285 / L293-297 | §5（账归 7.4） |
| 3.3 式 (6)、§6 | dispatch 字节 $`Bk\thinspace d_{\mathrm{model}}\cdot\mathrm{bytes}_{\mathrm{act}}`$；V1 两族语义 | DeepEP EPBuffer（V2.5）/ tag v1.2.1（V1 语义） | §6 |
| 3.3 式 (7)、§7 | logits IO 遍数账；DeepSelect 有效带宽 | DeepSelect sampling fp32（vocab ~128K） | §7 |
| 3.4 式 (1) | 融合省中间 tensor IO： $\Delta_{\mathrm{IO}} = 2(n-1)S$ | FlashMLA fused kernel + TileKernels fused SwiGLU+quant | §3 / §7 |
| 3.4 式 (2)(3)、§5 | launch 开销 / CUDA Graph replay；TileLang DSL | DeepJIT + tilelang_ops + masked layout 消费 | §5 / §7 |
| 3.4 §5 | TileLang tile 抽象、调度与数据流解耦 | tilelang [A] + TileKernels + tilelang_ops/cuda/schedule_mtp_varlen.py（7.2 实例） | §7 |

读表三件事。其一，「repo 落点」列给的是接口 / 文件级锚，不是指令级实现——后者归 gpu。其二，版本口径随锚注明：FlashMLA 分 main（V4.1）与 ba89a34（Hopper），DeepEP 分 main（V2.5）与 tag v1.2.1（V1），混读会全部落空（§3 / §6 坑注）。其三，Mega MoE 是 DeepGEMM 的**组件**而非独立 repo（DeepGEMM README 口径，3.3 §5 已辨），表中归在 DeepGEMM 行下。EP > 32 递减账（3.3 命题 3）的生产 instantiation 与 EPLB 配套归 7.4，本章只登记 Mega MoE 接口与 doc 论断。

## 3. FlashMLA：V4.1 sparse / fused 与 Hopper 双锚

2026.09.30 之后 clone FlashMLA main 找 V3 的 bf16 MLA decode kernel，会全部落空——代码已移除。README breaking notice 逐字（2026-11-23 核实）：该 release 移除 Hopper 架构与 V3 / V3.2 / V4.0 支持、改 FP8 / FP4 KV 格式，与前版不兼容；要跑旧模型或旧 KV 格式须切到旧 commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`（[DeepSeek-AI, FlashMLA](../references.md)，教学参照）。所以 FlashMLA 走读必须双锚：当前 main 面向 V4.1（B200 + Ascend 950）的 sparse / fused kernel，Hopper 时代 MLA decode 机制锚 ba89a34 + `docs/20250422-new-kernel-deep-dive.md`（两版 repo 均在，本轮旧 commit tarball 重取核实）。

> **坑**：把 FlashMLA 当前 main 当 Hopper MLA decode 读。触发条件：2026.09.30 之后按旧文档找 V3 bf16 decode kernel；现象：代码已移除，只剩 V4.1 FP8/FP4 sparse 与 fused kernel，接口与 KV 格式全对不上；规避：Hopper 机制锚 ba89a34（3.1 §5 已走读），V4.1 内容锚 main（本节），两处版本分开标注。

Hopper 旧口径 recap（3.1 §5 已完整走读，此处只记几何）：吸收形 score 内维 $d_{\mathrm{latent}} + d_{\mathrm{rope}} = 512 + 64 = 576$（V3 系），K block 64×576、V 即同一份 512 维 latent（deep-dive 逐字「in MLA, K and V are the same with different names」）；paged KV 每 token 576 元素连续存放，FP8 with scale 格式下每 page row 656 B（ba89a34 README 逐字）。吸收形等价与 regime 翻转（ $I_{\mathrm{MLA}} = H_q s_q (d_k + d_v)/d_k$，3.1 式 (8)）归 3.1 / models 4.1，不重推。V4.1 的 head_dim_k 降为 512（= kv_lora_rank 448 + rope 64， $d_{\mathrm{rope}}$ 保持 64），KV 改量化格式（§4）；旧 README 标 head_dim_k 576（V3 系，kv_lora_rank 512 + rope 64）/ 512（V4/V4.1，含 rope）——几何随模型代际变，格式随 release 变，两个维度别混。引擎读法（与 7.2 §9 一致，一手权威）：config **无** kv_lora_rank 字段，`hugging_face/config.py` L289 注释逐字「HF's shared KV vector includes RoPE; the internal rank excludes it.」+ `kv_lora_rank = head_dim − qk_rope_head_dim` = 512 − 64 = 448；`cache_common.py` V41_FORMAT quantized_elements=512、`cache_engine.py` L54 assert kv_lora_rank + rope_head_dim == 512——故 §4 的 528 B = 512 元素（448 latent + 64 rope）+ 16 scale，而非 576 元素（V3 几何误套会重复计 rope）。

当前 main 的三类 kernel（README，2026-11-23 核实）。其一，token-level sparse attention prefill / decode（FP8 / FP4 KV），支撑 DSA：indices 为 [B, s_q, K] int32、值 = 页块号 × page_block_size + 块内偏移（页地址已 bake），kernel 不再消费 block_table（参数仍必传、None 即可），无效位置 −1——3.1 §7 接口口径 recap，选择链（candidate blocks → DeepSelect）归 §7。其二，fused norm-RoPE-attn-RoPE-cast kernel：把 Q-norm（仅 V4 用、V4.1 不用）、Q-RoPE、core attention、O-RoPE（conjugate）、cast-to-FP8 融进单 kernel，省掉这些小 kernel 的独立开销（3.4 式 (1) 融合 IO 账的生产实例），TFlops 持平甚至略高；代价是须预先 permute Q_b 与 Wv 权重，输出经 `deep_gemm.fp8_einsum("bhr,hdr->bhd", ..., recipe=(1, 1, 32))` 消费（README 逐字，recipe 三元组 (1,1,32)）。其三，dense MHA prefill / backward（NVIDIA PR 贡献，SM100）。

性能数字（README 官方口径，B200 + CUDA 13.3，2026-11-23 核实）：

| kernel | prefill | decode | 平台 / 备注 |
| --- | --- | --- | --- |
| fused norm-RoPE-attn-RoPE-cast | 1460 TFlops | 950 TFlops | B200；2026.09.30 decode 再优化 10–15% |
| sparse attention | 1350 TFlops | 1024 TFlops | B200；README 推荐改用 fused |
| dense MHA | 1460 TFlops（fwd） | 1000 TFlops（bwd） | B200，NVIDIA 报 |
| sparse attention | 410 TFlops（95% 峰值） | 360 TFlops（83% 峰值） | Ascend 950；deep-dive docs/20260930 |

两个读数。其一，fused decode 标称 950 低于 sparse decode 1024，但 2026.09.30 release 把 fused decode 再优化 10–15%（README News 逐字），且 fused 省掉前后小 kernel 的独立 launch 与 IO——单看 attention kernel 的 TFlops 会误判，端到端步时还要算融合省下的那几遍小 kernel（3.4 式 (1)(2)）。其二，Ascend 950 的 410 / 360 TFlops 各标 95% / 83% 硬件峰值，反推峰值约 432 TFlops 量级（410/0.95 ≈ 431.6、360/0.83 ≈ 433.7，舍入一致）——这是 sparse attention 在该 NPU 上的有效峰值口径，与 datasheet 的稠密 FP8 峰值不是同一个数（硬件参数表归 1.3，微架构归 gpu）。requirements：NVIDIA SM100 / SM103 + CUDA 13.1+（13.2+ 推荐，提供量化 decode kernel 用的原生 `cvt.rn.bf16x2`，13.1 回退经 FP32 加宽更慢）；Ascend 950 + CANN 9.2.0+。

## 4. 低精度落地：528 / 288 / 68 三格式字节账

V4.1 主 KV 每 token 528 B（MXFP8）或 288 B（per-16 FP4），indexer KV 68 B——这三个数怎么从格式三元组算出来？格式三元组（元素类型、块大小、scale 类型）是 3.2 §2 的读法，本节把 DeepInfer 生产披露的三个格式对回该账。官方 doc 逐字（L271 / L273 / L275，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）：主 KV「per-16, e4m3 — like NVFP4 but without the global scale」；indexer KV「per-32, ue8m0, MXFP4」；MoE routed-expert 权重「per-32, ue8m0, MXFP4」。

字节账按 2.3 式 (4)（ $`\mathrm{bytes}_{\mathrm{elem}} = (k\cdot\mathrm{bytes}_{\mathrm{data}} + \mathrm{bytes}_{\mathrm{scale}})/k`$，FP 格式 zero-point = 0；3.2 式 (3) 是其含 zp 项的推广，recap 不重推）逐格式代入，并与 FlashMLA README、DeepInfer `cache_engine.py` KVCache docstring 三方互证：

| 格式 | 元素 | 块 $k$ | scale | 每 token 字节分解 | 每元素字节 | 对回 |
| --- | --- | --- | --- | --- | --- | --- |
| 主 KV MXFP8 | E4M3（1 B） | 32 | E8M0 ×1 B | 512 数据 + 16 scale = 528 B | 1.03125 | MXFP8（3.2 §2 表） |
| 主 KV per-16 FP4 | E2M1（0.5 B） | 16 | E4M3 ×1 B | 256 数据 + 32 scale = 288 B | 0.5625 | 类 NVFP4 无 global scale |
| indexer KV MXFP4 | E2M1（0.5 B） | 32 | E8M0 ×1 B | 64 数据 + 4 scale = 68 B | 0.53125 | MXFP4（3.2 §2 表） |

三个分解都算到底（python 复核 2026-11-23）。MXFP8：512 维 × 1 B = 512 B 数据，512/32 = 16 个 E8M0 scale × 1 B = 16 B，合 528 B；每元素 528/512 = 1.03125，恰等于 3.2 §2 格式表的 MXFP8 值 (32×1+1)/32。per-16 FP4：512 维 × 0.5 B = 256 B 数据（E2M1 两值一字节、偶索引低 nibble），512/16 = 32 个 E4M3 scale × 1 B = 32 B，合 288 B；每元素 288/512 = 0.5625，恰等于 NVFP4 的 (16×0.5+1)/16——doc 说「like NVFP4 but without the global scale」正是此意：块大小 16 与 E4M3 block scale 同 NVFP4，只去掉 tensor-wide FP32 global scale（NVFP4 格式账归 3.2 §2 / 2.3 §4）。indexer MXFP4：128 维 × 0.5 B = 64 B，128/32 = 4 个 E8M0 scale × 1 B = 4 B，合 68 B；每元素 68/128 = 0.53125 = MXFP4 值。

> **坑**：把「per-16, e4m3」的 e4m3 读成元素类型。触发条件：见 doc L271 就以为主 KV 元素是 FP8 E4M3（1 字节）；现象：该行在「Where is FP4 used?」标题下，元素是 FP4 E2M1（0.5 字节）、e4m3 是 **scale** 类型，误读使每元素字节虚高、 $\kappa$ 账同步错（2.3 §4 / 3.2 §2 同坑）；规避：格式三元组分开读，元素类型对标题上下文 + FlashMLA README 类型表双重核对。

三方互证逐字。FlashMLA README（2026-11-23 核实）：V4.1（head_dim 512）格式由 k_cache 末维（每 token 字节）检测，528 或 288；528 = 前 512 B float8_e4m3 覆盖全 512 维（64 RoPE 维也量化、无 bf16 部分）+ 末 16 B float8_e8m0 scale（每 32 个 e4m3 一个）；288 = 前 256 B 含 512 个 e2m1（2 值/字节、偶索引低 nibble）+ 末 32 B float8_e4m3 scale（每 16 个 e2m1 一个），且 288 格式仅作 extra_k_cache（当 k_cache 为 528 格式时）。DeepInfer `worker/cache_engine.py` KVCache docstring：swa_cache「Each V41 token stores [512B FP8 data | 16B UE8M0 scale]」（528）；ca_cache「V41 stores [data | scale] per token: 288 bytes for per-16 FP4 or 528 for MXFP8」；`_kv_bytes_per_token` L46 按 V41_FORMAT / V41_FP4_FORMAT 分派。口径注：doc L271 列的是主 KV 的 FP4 选项口径，FlashMLA 接口口径 288 仅作 extra_k_cache（k_cache 为 528 时），引擎两种格式都支持——三源各按其口径读，不强行调和成「主 KV 只有一种格式」。

kernel 落点（映射级）。KV 的 cast / dequant 由 TileKernels 的 per-token / per-block / per-channel FP8/FP4 cast/dequant 算子承担（§7）；GEMM 侧 SM100 消费 packed UE8M0 scale（4 个 UE8M0 pack 进一个 `torch.int`，DeepGEMM README，§5）——KV 里逐 scale 1 字节的 E8M0，到 GEMM 接口被 pack 成 int 消费，存储格式与计算格式是两层。routed expert 权重 per-32 ue8m0 MXFP4（doc L275）↔ config `expert_dtype` fp4 / `scale_fmt` ue8m0 / `weight_block_size` [32,32]（[W]）：非 expert 权重走 FP8 + 32×32 块，routed expert 走 FP4，3.2 §7 已对照。格式账的精度代价（量化误差界、GPTQ/AWQ 补偿）归 3.2，本节只交付字节与接口落地。

## 5. DeepGEMM：统一 tensor core 库与 Mega MoE

DeepGEMM 不止 GEMM——README 把 Mega MoE、indexer MQA logits、HyperConnection（HC）都列为组件。所谓统一 tensor core 库，是指把 FP8 / FP4 / BF16 GEMM、fused MoE、MQA scoring、HC 收进单一 CUDA codebase、全部 kernel 经 DeepJIT 运行时编译（安装期无 CUDA 编译）的库（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照，README 2026-11-23 核实）。DeepJIT 按 shape 特化 + 磁盘缓存（默认 `$HOME/.dj`，`DG_JIT_CACHE_DIR` 可配，fallback 前缀 `DJ_JIT_*`）——serving 的 M 维（= batch）每步变、prefill 的 M 千差万别，提前编译枚举不完 shape，JIT 把编译成本摊进缓存（3.3 §3 已走读，recap）。

细粒度 scaling 与 grouped GEMM recap（3.3 §3 / §4，不重推）：SM90（Hopper）要求 scaling factor 为 FP32、rescale 软件侧；SM100（Blackwell）要求 packed UE8M0（4 个 pack 进一个 int），tensor core block-scaling 路径原生消费。grouped GEMM 只 group M 轴（N/K 固定，为 MoE expert 同 shape 定制）：contiguous layout（`m_grouped_fp8_gemm_nt_contiguous`，训练 fwd / prefill，每段对齐 `get_mk_alignment_for_contiguous_layout()`）vs masked layout（`m_grouped_fp8_gemm_nt_masked`，decode + CUDA Graph，CPU 不知每 expert token 数，典型输入 = DeepEP low-latency 输出）。padding 浪费账（3.3 式 (3)、命题 2）与 masked 容量账（式 (4)）归 3.3。

Mega MoE 是融合最彻底的形态，也是 DeepGEMM 的组件（README 口径，非独立 repo）。单个 mega-kernel 融合 EP dispatch + linear1 / linear2（FP8×FP4 或 FP8×FP8）+ SwiGLU + EP combine，NVLink 通信与 tensor core 计算重叠（3.3 §5 命题 3 的 $\tau \approx \max(\tau_{\mathrm{comm}}, \tau_{\mathrm{gemm}})$ 账）。接口四件（README，2026-11-23 核实，完整例见 `tests/test_mega_moe.py`）：

```python
# DeepGEMM Mega MoE 接口序列（README 口径，教学摘录）
buffer = deep_gemm.get_symm_buffer_for_mega_moe(...)   # 分配 symmetric memory（须 PyTorch >= 2.9）
l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)  # 权重布局变换
l1 = (deep_gemm.localize(l1[0]), l1[1])                # locality domain 本地化（issue #462）
l2 = (deep_gemm.localize(l2[0]), l2[1])
deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer)          # 执行 mega-kernel
```

symmetric memory（对称内存：各 rank 以相同布局分配、可被远端直接访问的 buffer）须多进程 launch、PyTorch ≥ 2.9——这是 Mega MoE 把 NVLink 通信融进 kernel 的前提（通信对端 buffer 地址已知，才能在一个 kernel 内重叠搬运与计算）。`localize` 对应 locality domain（#462），把权重变换到本地域布局。官方论断（doc L285 逐字，官方披露，2026-11-23 核实）：「Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE」——限定词「within a scale-up domain」的账（域内带宽高、 $\tau_{\mathrm{comm}}$ 藏得进计算；跨域 RDMA 延迟藏不住）归 3.3 §5。EP > 32 递减（doc L293-297：带宽不再瓶颈后通信 / 同步主导 + 超节点多 job NVLink 干扰、整超节点单 EP 组避免）的生产账与 EPLB 配套归 7.4，本章只登记接口与论断。

组件另两件：indexer MQA logits（`fp8_fp4_mqa_logits` non-paged / `fp8_fp4_paged_mqa_logits` paged，weighted ReLU MQA，DSA 打分侧，3.1 §7 选择链的上游）；HyperConnection（HC，V4.1 hc_* 字段的算子侧，模型侧形式化归 models 维度 V4.1 待补）。DeepGEMM-Ascend 另立 repo（2026.09.30 发布、同 API），与主 repo 的 CUDA 路径分列。warp specialization / TMA / SASS 级优化归 gpu 维度。

## 6. DeepEP：V2.5 统一接口与负载均衡原语

DeepEP 当前 main 是 V2.5，V1 的 normal / low-latency 两族 API 已移除——EPLB 的权重 shuffle 和 Engram 的 RDMA read 各落在哪个原语？V2.5 把 high-throughput 与 low-latency 统一为单 **EPBuffer** 接口（hybrid / direct 两模式保留、NCCL Gin 后端、DeepJIT 编译、SM 与 QP 数解析计算免 auto-tuning），`ElasticBuffer` 拆为 EPBuffer / EngramBuffer / PPBuffer / BucketBuffer（共享 BufferBase 生命周期）+ BufferAllocator（buffer 构造前规划对称 tensor 分配）（[DeepSeek-AI, DeepEP](../references.md)，教学参照，README 2026-11-23 核实）。训练 / prefill 与 decode 的区分移到参数 `do_cpu_sync`：True 给精确输出尺寸 + CPU 侧 expert 计数（接 contiguous grouped GEMM），False 给 GPU 侧接收计数 + 容量分配、expert GEMM 用 `handle.psum_num_recv_tokens_per_expert` 定有效段（接 masked，3.3 §4 / §6 闭环）。

V1 语义锚 tag v1.2.1（V1 最后版本，机制讲解用；当前 main 已移除，3.3 §6 坑注同口径）。normal（high-throughput）：NVLink + RDMA 两阶段 asymmetric-domain forwarding，利用两域带宽不对称做聚合转发，SM 数可控；low-latency：纯 RDMA 点对点 + hook 重叠不占 SM（接收侧），decode 小消息延迟主导。性能表（v1.2.1 README，H800 + CX7 400Gb/s，2026-11-23 再核）：normal intranode EP8 153/158 GB/s、internode EP64 51/50 GB/s；low-latency EP8 77/114 μs、EP256 194/360 μs（dispatch/combine）。这组数是 V1 语义的教学锚，V2.5 接口已变、量级口径沿用需重测。

两个新原语是 7.4 的库层落点（README Expert load balancing 节 L330-380 + Engram 节 L382+，2026-11-23 核实）。其一，`EPBuffer.lb_prefetch_weights(redundant_expert_weights, expert_weights, redundancy_mapping)`：expert 计算前经 NVLink 把原 expert 权重 + 量化 scale 推到冗余槽；`lb_reduce_grads`：backward 把冗余 expert 的 FP32 梯度经 NVLink 归约回原 expert。两者是 NVLink 域内 collective（域内每 rank 都须调用，含无冗余槽的 rank）、异步跑在通信流上返回 EventOverlap、`.wait()` 后才能消费；冗余存储经 `lb_allocation_plan_or_num_bytes` 预留（BufferAllocator 计划或对齐字节数），与 dispatch/combine buffer 分离。README 自述动机来自 MoonEP / UltraEP 的 expert-replication 路线——这是 EPLB 冗余副本权重 shuffle 的通信原语层（7.4 §4 走读引擎侧怎么用）。其二，`EngramBuffer`（experimental，RDMA）：GPU/CPU 多层存储 + 多层 fetch，`fetch(indices)` 取 [num_layers, num_tokens, num_entries_per_token] int32、每层返回一个 completion hook、用该层数据前须调对应 hook——这是 Engram ENP one-sided RDMA read 的库层落点（7.4 §8）。

SM 账（V2.5 README Notes 逐字）：「EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported」——通信 kernel 占 SM、与计算竞争，是 3.3 命题 3 里 $g(e)$ 的一个成分；对照 2.4 的 GDR swap-in（RDMA NIC 直写 HBM、不占 SM）：KV 传输可零 SM，EP dispatch/combine 不行（当前版本口径）。口径分层注：V1 low-latency 的 hook 接收是后台 RDMA 不占 SM（v1.2.1 README「without any SM occupation」，指接收侧），V2.5 Notes 说 dispatch/combine 整体——版本与范围都不同，勿混读（3.3 §6 同注）。DeepEP-Ascend 另立 repo（同 API、Ascend 950 全性能）；Engram 需 NCCL 提供 `ncclGinOptFlagsWarpGet` 的 build。requirements：Hopper+ / CUDA 13.1+ / NCCL 2.32.3+ / PyTorch 2.10+。硬件层归 cluster 维度。

## 7. TileLang / TileKernels / DeepSelect：编译底座与选择算子

TileLang 是 DSL、TileKernels 是算子库、DeepSelect 是 TopK kernel——三者分工是什么？TileLang（[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）是 tile 抽象的 DSL：把调度空间（thread binding、layout、tensorize、pipeline）与数据流解耦，用户写数据流、编译器负责其余优化，多硬件后端（NVIDIA + Ascend）（3.4 §5 已走读，recap）。DSL 实例 recap（7.2 已完整走读算法，此处只看形式）：DeepInfer `tilelang_ops/schedule_mtp_varlen.py` 的 `compute_seqlens_q_by_confidence_rates` L11 → `cuda/schedule_mtp_varlen.py` 用 `@tilelang.jit`（L7）+ `T.prim_func`（L18）+ 两个 `T.Kernel`（L27 / L40）+ `T.alloc_reducer`（L46）写 DSpark 的 cumprod + 全局 top-k + fused remap scatter——声明式数据流、调度交给编译器，这是 3.4 §5 「TileLang 实例」的生产兑现（算法正确性归 7.2，不重推）。

TileKernels（[DeepSeek-AI, TileKernels](../references.md)，教学参照，README 2026-11-23 核实）是这条路线的算子库实证：数十个 TileLang 优化算子，七族——MoE Routing（top-k / scoring）、Quantization（per-token / per-block / per-channel FP8/FP4 cast/dequant + fused SwiGLU+quant）、Engram（gating fused RMSNorm）、mHC（Sinkhorn）、Transform（RoPE）、Rand、Modeling；多数接近硬件算力或带宽极限、已用于内部训推；2026-09-30 起 NVIDIA + Ascend 双后端运行时自动选择。requirements：PyTorch 2.13+ / TileLang 0.1.15+ / CUDA 13.1+ / CANN 9.2.0+（版本号本轮补齐复核）。§4 的 KV cast/dequant、§5 的 SwiGLU+quant 融合在此都有算子落点；Engram / mHC 的模型侧形式化归 models 维度（V4.1 待补），本章只列算子名。

DeepSelect（[DeepSeek-AI, DeepSelect](../references.md)，教学参照，README + deep-dive 2026-11-23 核实）是 DSA lightning indexer 与 sampler 的 TopK kernel，对 `torch.topk` 2–20×。两场景：indexer bf16 / topk ≤ 4096 / 主优化 topk = 512（= V4.1 config index_topk [W]）；sampling fp32 / vocab ~128K（benchmark vocab_size 129280，与 config 同值）。算法（deep-dive 口径，3.1 式 (9) 期望界已推导）：维护当前第 K 大阈值，Scan / Filter / Compact 三步循环，每输入元素恰读一遍（带宽最优），额外空间 $O(K + b_s + b_2)$ 驻留 shared memory。性能度量是**有效内存带宽**（口径：TopK 无浮点运算，FLOP rate 无意义——README 逐字）。工程开关：`sorted_index` 非必要不开（耗性能）、`return_value=False` 跳过值输出更快（README 建议）。

> **坑**：把 DeepSelect 的 fp32 sampling 场景外推到 Ascend。触发条件：在 Ascend 950 上跑 fp32 vocab ~128K 的 sampling TopK；现象：README 逐字「only the CUDA implementation supports torch.float32 ... Ascend supports torch.bfloat16 only」，fp32 路径在 Ascend 不可用；规避：双平台对照时 sampling fp32 仅 CUDA，Ascend 走 bf16（indexer 场景本就 bf16，不受影响）；引擎侧 Ascend topk 经 `npu_ops/deep_select.py` topk L21（_torch_topk fallback L7）。

选择链的引擎落点（3.1 §7 已走读算法，此处登记文件）：`tilelang_ops/candidate_blocks.py` 做块级预选（candidate_block_max 池化 + 块 TopK + 压实 + restore），DeepSelect 做最终 token TopK，DeepGEMM mqa_logits 做打分——三级各管一段，DSA serving 全链算法归 3.1。

## 8. open vs internal 差距与 benchmark 方法学

官方披露 FlashMLA 开源版 decode kernel 慢 5–10%、端到端至多差 5%——这两个数互相一致的条件是什么？doc L289 逐字（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）：「They are broadly the same; the indexer and mega series differ somewhat. FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.」L291：「End-to-end performance is close — at most 5% apart.」口径注：SASS 级归因属官方披露口径，本章不展开指令级实现（spec risk 豁免，silicon 归 gpu）；「broadly the same」指开源版与内部版总体一致，差异集中在 indexer 与 mega 系列（略有不同）及 FlashMLA 缺的那一个手调 SASS 优化。

把两个官方数对成一笔账。路线：decode kernel 慢 $\delta$（本章 $\delta$ = kernel 慢比例，非 7.2 的回溯 $\delta$=6）只影响它在步时里的占比部分，其余不变——关键一步是「e2e 相对差距 = kernel 占比 × kernel 相对慢」。设 decode attention kernel 占端到端步时的比例为 $f$（ $0 \le f \le 1$），开源版该 kernel 时间乘 $(1+\delta)$，其余部分不变：

**命题 1**（e2e 差距的占比折算）。设端到端步时 $T = T_{\mathrm{attn}} + T_{\mathrm{rest}}$， $f = T_{\mathrm{attn}}/T$，开源版仅 attention kernel 变慢 $\delta$（ $T_{\mathrm{rest}}$ 不变），则端到端相对差距恰为 $f\delta$：

$$
\frac{T' - T}{T} = \frac{T_{\mathrm{attn}}(1+\delta) + T_{\mathrm{rest}} - T}{T} = f\delta \qquad (1)
$$

**证明**（代数，一步）： $T' - T = T_{\mathrm{attn}}\delta$，除以 $T$ 得 $f\delta$。∎ 正例： $f = 0.3$、 $\delta = 0.08$ → e2e 差距 2.4%； $f = 0.5$、 $\delta = 0.10$ → 5.0%。边界： $f \to 1$（步时全是 attention）差距 $\to \delta$； $f \to 0$ 差距 $\to 0$。一致性读数：官方给 $\delta \le 0.10$ 且 e2e $\le 0.05$，由式 (1) 反推 $f \le 0.05/0.10 = 0.5$——即 decode attention kernel 占步时不超过一半，两个官方数才自洽。这与 decode 步时的结构一致：MoE expert GEMM、dispatch/combine、采样、norm 等占去另一半以上（1.2 decode 账、3.3 MoE 账）。 $f$ 本身官方未披露，命题 1 是结构读数而非实测拟合；它给的是「两个官方数相容的条件」，不是 $f$ 的测量值。

> **提点**：kernel 级差距与端到端差距是两笔账，别把 5–10% 直接当 e2e 损失。式 (1) 的折算因子 $f$ 由 workload 与模型几何定：长上下文 decode attention 占比高、 $f$ 大，e2e 差距靠近 $\delta$；短上下文 MoE 主导、 $f$ 小，e2e 差距远小于 $\delta$。读官方「e2e ≤5%」时要带上它的 workload 口径。

benchmark 方法学（怎么自己复现与对账）。doc L281 逐字（官方披露）：用 repo 的 `examples/benchmark.py` profile 取 trace，trace 含 kernel usage 信息——`examples/_trace.py` 的 `dump_trace` L54 / `export_profile_trace` L143 是导出落点。benchmark.py 关键参数（2026-11-23 tarball 核实）：`--decode_num_requests` 64 / `--context_length` 32768 / `--prefill_batch_size` 32768 / `--non_cached_tokens` 4096 / `--yoco_clip`（0=off、裸给=140 compact）/ `--sampler_force_topk` 1024 / `--enable_eplb` / `--tracing_path` / `--result_jsonl`。README L83：Pareto 图测于 4 SM100（prefill）与 64（decode），各在一个 scale-up domain 内；复现走 `examples/sweep/run_sweep.sh`（decode / prefill / encoder 三模式扫一维 batch 参数出吞吐-延迟曲线，decode 模式位置参数含 nextn 与 draft_acceptance——7.4 引 wide_ep 图的 benchmark 假设口径即出于此）。

MFU 口径（对账时最容易错的一笔）。`csrc/engine/flops_estimator.hpp` FlopsEstimator L28 按 FP8 等效 FLOPs 计 main-model 工作量，系数（L13-19，含 multiply-add 2 FLOPs 与 FP8 等效权重）：FP4=1 / FP8=2 / BF16=4 / TF32=8（FP32 同 TF32，记账参考非实际 kernel 精度）。换算：FP8 等效 = 原始 FLOPs × 系数/2，即 FP4 折半、FP8 不变、BF16 翻倍、TF32 翻四倍。注释逐字：「exclude draft/padding/replicas」「MFU measures target-model FLOPs throughput; DFlash is extra compute used to accelerate that model, so its FLOPs are excluded」——MFU 只计目标模型，DSpark draft、grouped GEMM padding、EPLB 冗余副本的额外计算全部排除。全数字实例：某步 FP4 expert GEMM 600 TFLOP（原始）+ FP8 dense 200 + BF16 attention 100 → FP8 等效 = 600×0.5 + 200×1 + 100×2 = 700 TFLOP；若把 draft 与 padding 也算进分子，MFU 会虚高（口径错）。benchmark 陷阱 recap 1.1：concurrency ≠ batch、token 计数口径、warmup、prefix cache 命中污染、离线 benchmark ≠ 在线 goodput（DeepInfer「80% of perfectly balanced offline」口径，7.4 §7 详）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| kernel 栈六件一手链接（README L23）+ custom TileLang ops；vendor/ 六 submodule | §1 全景图 | §1、7.1 |
| P3 公式 ↔ repo 映射：3.x 算法级 / 7.3 映射级，recap 不重推 | 映射总表 11 行 | §2 |
| FlashMLA 2026.09.30 breaking：移除 Hopper + V3/V3.2/V4.0、改 KV 格式；Hopper 锚 ba89a34 | 576 / 656 B 旧口径 | §3、3.1 §5 |
| V4.1 KV 528（MXFP8）/ 288（per-16 FP4）B、indexer 68 B；每元素 1.03125 / 0.5625 / 0.53125 | 三格式分解表 | §4、2.3 式 (4)、练习 1 |
| fused kernel 须预 permute Q_b/Wv + fp8_einsum recipe (1,1,32)；B200 fused 1460/950、sparse 1350/1024 | §3 性能表 | §3 |
| Ascend 950 sparse 410/360 TFlops（95%/83% 峰值）；DeepGEMM-Ascend / DeepEP-Ascend 另立 repo | §3 / §5 / §6 | §3、§5、§6 |
| Mega MoE ⊂ DeepGEMM：四 API + symm mem（PyTorch ≥ 2.9）+ localize；scale-up 域内通用解 | doc L285 | §5、3.3 §5 |
| DeepEP V2.5 EPBuffer；lb_prefetch_weights / lb_reduce_grads = EPLB shuffle 原语；EngramBuffer = ENP 库层 | README L330-380 / L382+ | §6、7.4 |
| DeepSelect 2–20× vs torch.topk；fp32 sampling 仅 CUDA、Ascend 仅 bf16 | §7 坑注 | §7、3.1 §7 |
| open vs internal：decode 慢 5–10%、e2e ≤5%；e2e 差距 = $f\delta$，相容条件 $f \le 0.5$ | 命题 1、式 (1) | §8、练习 2 |
| benchmark：benchmark.py trace → kernel usage；MFU FP8 等效、排除 draft/padding/replicas | FlopsEstimator 系数 | §8、练习 4 |

## 练习

1.（计算 · [10]）V4.1 三格式字节账（[W] + FlashMLA README 口径）：(a) 主 KV MXFP8（512 维 E4M3、per-32 E8M0 scale）每 token 字节与每元素字节；(b) 主 KV per-16 FP4（512 维 E2M1、per-16 E4M3 scale）同求；(c) indexer KV MXFP4（128 维 E2M1、per-32 E8M0 scale）同求。校验：(a)(b)(c) 的每 token 字节分别 = 528 / 288 / 68，与 FlashMLA README 逐字及 `cache_engine.py` KVCache docstring 一致；每元素字节对回 3.2 §2 格式表的 MXFP8 / NVFP4 / MXFP4 行。
2.（推导 · [20]）命题 1 闭环（open vs internal 差距）：(a) 补全式 (1) 的代数证明（ $T' - T = T_{\mathrm{attn}}\delta$ 一步）；(b) 代入 $f = 0.3$、 $\delta = 0.08$ 与 $f = 0.5$、 $\delta = 0.10$ 求 e2e 差距；(c) 由官方 $\delta \le 0.10$、e2e $\le 0.05$ 反推 $f$ 的上界，并说明为什么 decode 步时结构支持该上界。校验：(b) 与 §8 正例一致；(c) 上界 = 0.5，与 doc L289 / L291 两官方数自洽。
3.（代码 · [30]）Mega MoE 接口走读（DeepGEMM README + `tests/test_mega_moe.py`）：(a) 列出四件 API 的调用顺序与各自职责（buffer 分配 / 权重变换 / localize / 执行）；(b) 说明为什么须 symmetric memory 多进程 launch 且 PyTorch ≥ 2.9，对照 3.3 §5 的通信-计算重叠账；(c) 指出 masked grouped GEMM（`m_grouped_fp8_gemm_nt_masked`）消费 DeepEP 哪个输出、`do_cpu_sync` 取何值，把 §5 与 §6 的接口闭环。校验：file:line 锚对研究笔记（DeepGEMM README L114-144、DeepEP V2.5 `do_cpu_sync` / `psum_num_recv_tokens_per_expert`）；(c) 与 3.3 §4 两种 layout 一致。
4.（计算 · [30]）FlopsEstimator FP8 等效口径（`csrc/engine/flops_estimator.hpp` L13-19）：(a) 某步原始 FLOPs 为 FP4 expert GEMM 600 TFLOP + FP8 dense 200 + BF16 attention 100，求 FP8 等效总量；(b) 若误把 DSpark draft 的 150 TFLOP（FP8）与 grouped padding 的 80 TFLOP（FP4）算进分子，FP8 等效虚高多少；(c) 说明 MFU 口径为什么排除 draft/padding/replicas（注释逐字）。校验：(a) = 700 TFLOP（系数 FP4=1/FP8=2/BF16=4，等效 = 原始 × 系数/2）；(b) 虚高 = 150×1 + 80×0.5 = 190 TFLOP；(c) 对回 §8 注释引文。

## 参考

- [DeepSeek-AI, FlashMLA](../references.md)：key: flashmla，教学参照 + 官方实现——main（V4.1 sparse/fused，B200 1460/950、1350/1024，Ascend 410/360）+ 旧 commit ba89a34（Hopper 576/656 B）双锚；528/288 KV 格式逐字、indices 页块编码、fused permute + fp8_einsum recipe (1,1,32)；2026-11-23 双 tarball 核实
- [DeepSeek-AI, DeepGEMM](../references.md)：key: deepgemm，教学参照 + 官方实现——统一 tensor core 库 + DeepJIT；SM90 FP32 SF / SM100 packed UE8M0；grouped contiguous/masked；Mega MoE 四 API + symm mem（PyTorch ≥ 2.9）+ localize（#462）；indexer mqa_logits + HC；DeepGEMM-Ascend 另立；2026-11-23 tarball 核实
- [DeepSeek-AI, DeepEP](../references.md)：key: deepep，教学参照 + 官方实现——V2.5 EPBuffer（hybrid/direct、NCCL Gin、do_cpu_sync）；V1 语义锚 tag v1.2.1（性能表）；lb_prefetch_weights / lb_reduce_grads（L330-380）+ EngramBuffer（L382+）；zero-SM RDMA EP 不支持；DeepEP-Ascend 另立；2026-11-23 双 tarball 核实
- [DeepSeek-AI, TileKernels](../references.md)：key: tilekernels，教学参照 + 官方实现——数十 TileLang 算子七族；requirements PyTorch 2.13+ / TileLang 0.1.15+；NVIDIA + Ascend 双后端运行时选择；2026-11-23 tarball 核实
- [DeepSeek-AI, DeepSelect](../references.md)：key: deepselect，教学参照 + 官方实现——DSA indexer + sampler TopK，2–20× vs torch.topk；fp32 仅 CUDA、Ascend 仅 bf16；sorted_index / return_value 开关；有效内存带宽度量；2026-11-23 tarball 核实
- [Wang et al., 2025](../references.md)：TileLang，arXiv:2504.17577，[A]（key: tilelang）——tile 抽象 DSL、调度与数据流解耦、多硬件后端；schedule_mtp_varlen 实例（7.2 走读）
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——README L23 kernel 链接、vendor/ 六 submodule、tilelang_ops（cuda/asc）、cache_engine.py KVCache docstring、flops_estimator.hpp、examples/benchmark.py + _trace.py + run_sweep.sh；main@a679bb86 fresh tarball 2026-11-23
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——低精度三格式（L271/L273/L275）、Mega MoE 通用解（L285）、open vs internal（L289/L291）、EP>32 递减（L293-297）、benchmark trace（L281）
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：key: v41-flash-config，[W]——MLA 几何 / index_topk 512 / index_head_dim 128 / quantization_config（weight_block_size [32,32]、scale_fmt ue8m0、expert_dtype fp4）；模型侧形式化归 models（V4.1 待补）
