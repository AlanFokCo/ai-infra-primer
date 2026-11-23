# 3.1 Decode Attention 算子族（Decode Attention Kernels）

> prefill 的 FlashAttention 靠 batch × head × query 块铺满 GPU，decode 每步 query 长度只有 1——并行度从哪里补回来，补回来的代价是什么？本章沿六节回答：split-KV 把 KV 维变成并行维（reduction 正确性与账）、FA2/FA3 的并行与流水策略、FlashInfer 的 BSR 统一格式与 plan/run 接口（含 cascade 两级正确性）、FlashMLA 的吸收形几何与 DSA 的稀疏选择链 | 前置：models 4.1 §5（FlashAttention 前向数学：online softmax / tiling / IO 账，本章 recap 不重推）、models 4.1 §4（MLA 低秩与 absorption）、models 4.2 §6（DSA lightning indexer）、models 2.4（KV cache 与前缀不可变命题）；inference 1.2（κ 几何表与 decode 带宽账）、2.1（块表式 (3) 与 block size b）、2.2（prefix caching）、1.4（batch invariance 概念） | 锚点：FlashInfer（[Ye et al., 2025](../references.md)，arXiv:2501.01005，[A]）、FA2 / FA3（[Dao, 2023](../references.md)，arXiv:2307.08691，[A]；[Shah et al., 2024](../references.md)，arXiv:2407.08608，[A]）、FlashDecoding（[Dao et al., 2023](../references.md)，CRFM 博客，[C] 开放技术博客、非闭源，无 arXiv）、FlashMLA / DeepSelect（[DeepSeek-AI, FlashMLA](../references.md)、[DeepSeek-AI, DeepSelect](../references.md)，教学参照 + 官方实现；FlashMLA Hopper 走读锚旧 commit ba89a34）、vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A] + vllm-repo）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）

## 1. decode 把 attention 的并行度问题换了方向

FlashAttention 在 prefill 把 batch、head、query 块三个维度铺满 GPU；decode 每步 query 长度只有 1，第三个维度塌缩——batch 1 时整个 attention 只用得到 GPU 的不到 1%（[Dao et al., 2023](../references.md)，CRFM 博客逐字口径，[C] 开放技术博客、非闭源）。并行度从哪里补回来？这是本章全部算子族的共同出发点。

先 recap 三件已交付的数学（models 4.1 §5，本章不重推）：online softmax 把行归一化拆成 running max 与 running 指数和，每来一块换基准重缩放，换基准是恒等变换（models 4.1 式 (15)(16)）；tiling 让 Q/K/V 分块驻留 SRAM，score 矩阵从生到死不碰 HBM；IO 账 $O(T^{2} d_{\mathrm{head}}^{2} / M)$（models 4.1 式 (17)，其 d = head 维，按 spec §4.3 翻译规则译为本章 $d_{\mathrm{head}}$ ）说明长序列瓶颈是访存不是 FLOPs。本章的全部 split / merge 数学都是这条「换基准恒等式」的推广：tile 块级归约归 models（online softmax 本体），段级（split 间）归约是本章增量。

decode 的 shape 走读：q 为 [B, H_q, 1, d_head]，K/V cache 为 [B, H_kv, T, d_head]（GQA 口径；MLA 与 DSA 的几何分别见 §5 / §7）。FA2 的 query 块并行在 q = 1 时只剩一个块，grid（kernel 一次启动的全部 thread block 集合）退化为 B × H_kv ——FlashInfer 的 decode scheduler 里 grid y 维 `gdy = num_kv_heads`（`scheduler.cuh`，2026-11-23 main 核实）正是这个网格模型。代入章脊口径：8B、B = 1 时 grid 只有 8 个 CTA（cooperative thread array，即 thread block），对 A100 的 108 个 SM（博客口径）覆盖率 8/108 ≈ 7.4%。博客的论断逐字：「if the batch size is smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100), the operation will only use a small part of the GPU」；长上下文让情况更糟，因为 KV 容量把 batch 压得更小。所谓占用率（occupancy），本章按「grid 的 CTA 数能否铺满 SM」读（口径：博客的粗口径；warp 级占用率的精确定义与硬件来源归 gpu 维度）。

**本章口径**（章脊锁定，全章例题复用，换口径随文声明）：

| 口径 | 几何 | 硬件与用途 | 来源 |
| --- | --- | --- | --- |
| 8B dense GQA | L = 32、H_q = 32、H_kv = 8、d_head = 128、bf16 KV，κ = 4096 字节/token/层（1.2 式 (3)） | 带宽-时间换算用 H100 SXM（3.35 TB/s、BF16 dense 989.5 TFLOPS）；occupancy 算例用 A100 108 SM（博客口径）——两笔账各自出处，不混算同一台机器 | 1.2 章脊表；[NVIDIA datasheet](../references.md)，[C] 官方规格披露；SM 数 [Dao et al., 2023](../references.md)，[C] |
| V3 MLA | H_q = 128、d_latent = 512、d_rope = 64、61 层 | H800 SXM5（deep-dive doc 口径：3.35 TB/s、throttled 峰值约 865 TFlops）——§5 专用，换口径原因：FlashMLA deep-dive 的原始 benchmark 口径 | [DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]；[DeepSeek-AI, FlashMLA](../references.md)，教学参照 |
| V4.1-Flash | index_topk = 512、index_n_heads = 32、index_head_dim = 128、40 层 | §7 专用；KV 精度用 bf16 教学反事实，生产格式随文注 | [DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]（hf-mirror 传输核实，2026-11-23） |

符号约定：B = batch 内请求数，T = KV 长度，b = block size（token 数，2.1），κ = 每 token 每层 KV 字节数（1.2），H_q = query head 数（本章声明），S = split 数；bytes 不占字母，一律文字「字节」或 $\mathrm{bytes}$ 。跨维度翻译沿 spec §4.3：models 的 d（hidden）译 $d_{\mathrm{model}}$ 、b（每元素字节数）译「字节」（1.2 同口径）。

算子边界声明（spec §2 三层划分）：本章讲算法、tiling 策略、IO 与算力账、引擎调用接口；warp 内寄存器分配、tensor core 指令形状、SASS 调度归 gpu 维度（纯文字指引，不展开）；引擎模块全量走读归 5.3 / 6.x / 7.3；模型侧数学归 models。下文 FA3 的 warp specialization 与 FlashMLA 的 seesaw 都是**调度策略级**描述，不属指令级内容。本章增量六件：§2 split-KV 正确性与账、§3 FA2/FA3 策略、§4 FlashInfer 格式与接口（含 cascade 正确性）、§5 FlashMLA 吸收形几何、§6 paged kernel 接口、§7 DSA 选择链。

## 2. split-KV：把 KV 维变成并行维（FlashDecoding）

8 个 CTA 铺不满 108 个 SM，缺口 100 个——能不能把一条请求的 KV 切开，让上百个 SM 各算一段，再拼出与不切完全相同的结果？能。FlashDecoding 博客给出三步（[Dao et al., 2023](../references.md)，[C]，全文核实）：(1) KV 切 chunk——「step (1) does not involve any GPU operation, as the key/value chunks are views of the full key/value tensors」（口径：「免费」指无数据移动、仅元数据视图，不是零成本）；(2) 各 split 并行跑 FlashAttention，每行每 split 多写一个标量——该段 score 的 log-sum-exp（LSE）；(3) 跨 split 归约，用 LSE 加权各段贡献。online softmax 因此被用在两级：split 内（models 4.1 的块间）与 split 间（本节）。所谓 split-KV，是指把单条请求的 KV 序列沿长度维切成多段、各段独立算 partial attention、再用 LSE 加权归约成完整输出的并行策略。

并行度账先行。设 S 为 split 数，decode attention 的 grid 规模为

$$
n_{\mathrm{CTA}} = B \cdot H_{kv} \cdot S \qquad (1)
$$

**符号**：n_CTA = grid 中 CTA 数；B = batch；H_kv = KV head 数；S = 每请求 KV 的 split 数。式 (1) 取 FlashInfer 的网格模型：每 CTA 处理一个 (请求, KV head, split) 三元组，GQA 组内 H_q/H_kv 个 query head 在 CTA 内共享 KV 读取（`scheduler.cuh` `gdy = num_kv_heads`）。代入章脊口径：B = 1、H_kv = 8、S = 1 时 n_CTA = 8；取 S = 14 得 n_CTA = 112 ≥ 108，恰好铺满 A100。此时每段长 $\lceil 131072 / 14 \rceil$ = 9363 token。S 的上限由最小段长决定：FlashInfer 的二分搜索下界是 $\max(128 / b, 1)$ 页（`scheduler.cuh` L205，即最小段长 128 token），T = 131072 时 S 至多 1024。段长下界是启发式：段越短，归约占比 η ∝ S 越高（式 (4)），段内 tiling 效率也越差。

每段产出什么？设第 i 段的位置集为 $`\mathcal{I}_i`$ ，段内跑完 online softmax 后维护四个量：

$$
m_i = \max_{s \in \mathcal{I}_i} x_s, \quad \ell_i = \sum_{s \in \mathcal{I}_i} e^{x_s - m_i}, \quad O_i = \frac{1}{\ell_i} \sum_{s \in \mathcal{I}_i} e^{x_s - m_i} v_s, \quad \mathrm{LSE}_i = m_i + \ln \ell_i \qquad (2)
$$

**符号**：x_s = query 对位置 s 的 score（含缩放与 mask）；v_s = 位置 s 的 value 向量；m_i = 段内最大 score；ℓ_i = 段内以 m_i 为基准的指数和；O_i = 段内归一化输出（「假设这段就是全部 KV」的 attention 结果）；$`\mathrm{LSE}_i`$ = 段内指数和取对数（零基准），即 $`\ln \sum_{s \in \mathcal{I}_i} e^{x_s}`$ 。段内更新就是 models 4.1 式 (15)(16)，不重推；本节的增量是把 (O_i, LSE_i) 当作可合并的**状态**写出段外。

两段合并的数学（FlashInfer `merge_state` 的实现口径，`cascade.py` L42）：

$$
\begin{aligned}
m &= \max(\mathrm{LSE}_i, \mathrm{LSE}_j), \qquad w_i = \exp(\mathrm{LSE}_i - m), \qquad w_j = \exp(\mathrm{LSE}_j - m) \\
O &= \frac{w_i O_i + w_j O_j}{w_i + w_j}, \qquad \mathrm{LSE} = m + \ln(w_i + w_j)
\end{aligned} \qquad (3)
$$

**符号**：m = 两段 LSE 的较大者（数值稳定基准）；w_i, w_j = 两段在基准 m 下的指数权重（i, j 为段标签，i ≠ j）；O = 合并输出；LSE = 合并后的 log-sum-exp。减去 m 再取 exp 保证指数自变量 ≤ 0，overflow 在构造上不可能（models 4.1 §5.2 同一数值性质）。

全数字实例（沿 models 4.1 §5.2 的 logit 例，配二维 value）：x = (1, 2, 5, −1)，v_1 = (1, 0)、v_2 = (0, 1)、v_3 = (2, 1)、v_4 = (0, −1)，切两段（段 1 = {1, 2}、段 2 = {3, 4}）。段 1：m_1 = 2、ℓ_1 = 1.3679、O_1 = (0.2689, 0.7311)、LSE_1 = 2.3133；段 2：m_2 = 5、ℓ_2 = 1.0025、O_2 = (1.9951, 0.9951)、LSE_2 = 5.0025。合并：m = 5.0025，w_1 = exp(2.3133 − 5.0025) = 0.0679，w_2 = 1，O = (1.8853, 0.9783)，LSE = 5.0682。直接对全行做 softmax：O = (1.8853, 0.9783)、LSE = 5.0682——逐位相同。注意中间量 ℓ 依赖所选基准（合并用 m = 5.0025，直接算用 m = 5），但 O 与 LSE 是基准无关量，两者才是状态的合法分量。

**命题 1**（split-KV reduction 正确性）。对任意两个不相交的位置集 $`\mathcal{I}_i, \mathcal{I}_j`$ （i ≠ j），式 (3) 合并出的状态与直接对 $`\mathcal{I}_i \cup \mathcal{I}_j`$ 计算 attention 的状态逐点相等（O 相等、LSE 相等）；合并算子 ⊕ 满足结合律与交换律，S 段归约可按任意顺序组合。

**证明**（路线：展开 O_i 定义 → 代入合并式 → 指数和合并 → 归一化；关键一步是基准平移在指数里抵消）。记零基准指数和 $`E_i = \exp(\mathrm{LSE}_i)`$ （E_j 同理）。第一步，分子项展开：

$$
w_i O_i = e^{\mathrm{LSE}_i - m} \cdot \frac{\sum_{s \in \mathcal{I}_i} e^{x_s - m_i} v_s}{\ell_i} = e^{-m} \sum_{s \in \mathcal{I}_i} e^{x_s} v_s
$$

其中用了 $`e^{\mathrm{LSE}_i} = \ell_i e^{m_i}`$ 把段内基准 m_i 消掉。第二步，两段分子相加（不相交并集上的和式直接合并）：

$$
w_i O_i + w_j O_j = e^{-m} \sum_{s \in \mathcal{I}_i \cup \mathcal{I}_j} e^{x_s} v_s
$$

第三步，分母同构： $`w_i + w_j = e^{-m} \sum_{s \in \mathcal{I}_i \cup \mathcal{I}_j} e^{x_s} = e^{\mathrm{LSE}_{i \cup j} - m}`$ 。第四步，相除得 $`O = \sum e^{x_s} v_s / \sum e^{x_s}`$ ，即并集上的 softmax 加权和（零基准形，与任意基准形逐点相等）；LSE 侧 $`m + \ln(w_i + w_j) = \ln(e^{\mathrm{LSE}_i} + e^{\mathrm{LSE}_j}) = \mathrm{LSE}_{i \cup j}`$ 。结合律与交换律继承自不相交并集的结合交换律与加法交换律。S 段归纳：基例两段已证；归纳步「前 k 段合并状态再并第 k+1 段」留作练习 2。∎

边界例：空 split 的 ℓ = 0、LSE = −∞，权重 w = exp(−∞ − m) = 0，合并自动退化为另一段——实现以 −∞ 哨兵处理空段，不需要特判分支。正例即上方数值例。命题 1 是 FlashDecoding 第三步、FlashInfer `merge_state` / `merge_states`（`cascade.py` L42 / L170）与 §4 cascade 的共同数学基础；FlashInfer 论文把 (O, LSE) 称为 attention state、把 ⊕ 称为其上的标准归约算子（类比 GEMM 的求和，[Ye et al., 2025](../references.md)，[A]，§2.2）。

> **坑**：LSE 必须 fp32。触发条件：把 $`\mathrm{LSE}_i`$ 存成 bf16（尾数 8 bit）；现象：长 KV 下 LSE 量级为 max score + ln T，bf16 舍入使 exp(LSE_i − m) 权重失真，归约输出随 split 数漂移——batch invariance（1.4 定义、3.4 kernel 账）同时被破坏；规避：partial state 的 LSE 一律 fp32（`merge_state` 显式 `.to(torch.float32)`，`cascade.py` L90-91），partial O 可以 bf16（其误差经权重归一后是二阶小量）。

归约的账。并行度不是免费的：S 段 partial state 要写出再读回。设 partial O 与 LSE 均 fp32（FlashInfer float workspace 口径），每请求每层额外流量与占比为

$$
M_{\mathrm{red}} \approx 2 S H_q (4 d_{\mathrm{head}} + 4), \qquad \eta = \frac{M_{\mathrm{red}}}{T \kappa} \qquad (4)
$$

**符号**：M_red = 归约额外 HBM 流量，单位字节（S 段 partial 写 + 读，最终输出写为高阶小量略去）；系数 4 = fp32 每元素字节数（字节量已编码在系数里，式尾不再乘 bytes 因子）；括号内 +4 = 每 head 一个 fp32 LSE 标量；η = 额外流量占 KV 读的比例；κ 见 1.2 式 (3)。代入章脊口径（S = 14、H_q = 32、d_head = 128、T = 131072、κ = 4096 字节）：M_red ≈ 2 × 14 × 32 × 516 = 451.5 KiB，KV 读 Tκ = 512 MiB，η ≈ 0.086%。FLOPs 侧同理：归约约 S·H_q·d_head 量级，attention 本体 4T·H_q·d_head（1.2 式 (2) 口径），占比 S/(4T) ≈ 0.003%。结论：归约是小头，并行度收益是主项——但收益能兑现多少，取决于 attention 在步时里的份额。

收益的两级读数（博客口径 + 章脊口径）。博客实测：CodeLlama-34B、batch 1、序列 512→64k，attention 本身至多比 FlashAttention 快 50×，e2e decode 至多快 8×；32k 以内 attention 时间近乎恒定（GPU 被铺满，序列变长只是每段变长）。e2e 远小于 attention 加速比是 Amdahl 折扣：decode 步时还有权重 GEMV（1.2 式 (4)）。用章脊口径算这个折扣：8B、B = 1、T = 131072 时 attention KV 读每层 512 MiB，按 H100 3.35 TB/s 折合 160 μs/层、32 层合 5.13 ms；权重读 16.1 GB 折合 4.81 ms——即使 attention 降到零，e2e 上界也只有 (5.13 + 4.81)/4.81 ≈ 2.1×。博客的 8× 是 34B 模型长窗口口径（attention 份额更高），两个数不矛盾，差别全在 attention 占步时的比例。可用版本：FlashAttention ≥ 2.2、xFormers ≥ 0.0.22（博客口径）。

split 决策不由人工指定 S，而按 occupancy 实测反解。FlashInfer 的机制（`include/flashinfer/attention/scheduler.cuh` L138-211，2026-11-23 main 核实，机制摘录）：

```cpp
// flashinfer/include/flashinfer/attention/scheduler.cuh（机制摘录）
max_grid_size = num_blocks_per_sm * num_sm;   // occupancy API 实测，非名义 SM 数
if (batch_size * gdy >= max_grid_size) {      // gdy = num_kv_heads
  split_kv = false;                           // 已铺满，不切
} else {
  // 二分搜索最小 pages/batch，使 new_batch_size * gdy <= max_grid_size
  if (new_batch_size == batch_size && !enable_cuda_graph)
    split_kv = false;                         // 短序列切了也不增并行，不值得
  else
    split_kv = true;                          // CUDA Graph 恒用 partition-kv（静态形状）
}
```

三个结构事实：其一，max_grid_size 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 实测每 SM 可容 block 数再乘 SM 数——并行度上限是 occupancy 事实而非名义 SM 数；其二，二分搜索求的是「满足铺满的最小段长」，段长以页（b token）为单位，与 2.1 的块管理同粒度；其三，CUDA Graph 下恒切（形状静态），非 CUDA Graph 下短序列不切（省归约）。vLLM 的 FA3 backend 把同一决策暴露为 `max_num_splits`（§6 走读）：0 = 启发式（非 CUDA Graph 兼容）、full graph = 固定上界、batch-invariant 模式 = 1（禁分裂）。论文口径还有一条省账：短 KV 请求不切、partial 直写最终输出 buffer，绕过 workspace 与归约 kernel（[Ye et al., 2025](../references.md)，[A]，附录）。

```
split-KV 数据流（B=1、H_kv=8、S=14 示意；8B 口径）
──────────────────────────────────────────────────────
KV 序列 T=131072（逻辑连续，物理按块散落——2.1）
  切 S=14 段（view，无数据移动）
  ┌──────────┬──────────┬─────┬──────────┐
  │ split 1  │ split 2  │ ... │ split 14 │ ← 14×8=112 CTA
  │ 段内 FA  │ 段内 FA  │     │ 段内 FA  │ ← online softmax
  │(O_1,LSE_1)│(O_2,LSE_2)│ ... │(O_14,LSE_14)│ ← partial 写 workspace
  └──────────┴──────────┴─────┴──────────┘
  reduction kernel：⊕ 归约（式 (3)，LSE 加权）
  ──────────────────────────────────────► (O, LSE)
  额外 IO ≈ 451.5 KiB/层，占 KV 读 512 MiB 的 0.086%
```

> **提点**：split-KV 与 online softmax 是同一数学的两级使用——「分块换基准」从片内搬到片间，正确性证明的归纳结构完全同构（models 4.1 §5.2 对块归纳，命题 1 对 split 归纳）。读懂一个就读懂另一个；这也是 §4 cascade 能直接复用式 (3) 的原因。

## 3. FA2 / FA3：同一数学，改并行与流水（算法级）

FA1 只跑到理论峰值的 25–40%，FA2 在 A100 提到 50–73%，FA3 在 H100 把 FP16 推到 740 TFLOPs/s（75% util）——三代改的不是 attention 的数学对象，是并行划分与执行流水。本节只到策略级；指令级归 gpu 维度。

FA2 三技术（[Dao, 2023](../references.md)，arXiv:2307.08691，[A]，摘要口径）：(1) 减少 non-matmul FLOPs——softmax 的重缩放与除法跑在 CUDA core 上，吞吐远低于 tensor core 的 matmul，削减它们的次数直接抬利用率；(2) 单 head 也沿 seq 维跨 thread block 并行，提 occupancy（FA2 论文的 seq 维并行沿 query 维；decode q = 1 时该思路的同构策略是 KV split——那是 FlashDecoding 的独立贡献，FA ≥ 2.2 集成，博客口径，见 §2）；(3) thread block 内 warp 间的 work partitioning，减 shared memory 通信。报告数字：约 2× FA1，A100 50–73% 峰值，训练至 225 TFLOPs/s（72% MFU，摘要口径）。

FA3 三技术（[Shah et al., 2024](../references.md)，arXiv:2407.08608，[A]，摘要口径）：(1) warp-specialization——把 thread block 内的 warp 分成生产者 / 消费者角色，数据搬运（TMA 异步）与计算重叠，这是调度策略描述，TMA / WGMMA 的指令语义归 gpu；(2) block-wise matmul 与 softmax 交错（ping-pong）——两个 warpgroup 交替，一个做 matmul 时另一个做 softmax，把 CUDA core 与 tensor core 同时喂饱；(3) FP8 block quantization + incoherent processing——逐块量化控误差，随机正交变换把 outlier 摊开（算法级；变换的矩阵构造归论文）。报告数字：FA2 在 H100 仅 35% util（异步硬件特性未被同步调度利用），FA3 提升 1.5–2.0×，FP16 至 740 TFLOPs/s（75%）、FP8 近 1.2 PFLOPs/s，FP8 数值误差比基线 FP8 attention 低 2.6×。一致性核对：740 / 989.5 = 74.8% ≈ 75%（H100 BF16 dense 峰值取 [NVIDIA datasheet](../references.md)，[C]）。

| 代际 | 并行 / 流水策略 | 报告数字（出处口径） |
| --- | --- | --- |
| FA1 | batch × head × query 块；同步两趟循环 | 理论峰值 25–40%（FA2 摘要回溯） |
| FA2 | 减 non-matmul FLOPs；seq 维跨 thread block；warp 间 partitioning | ~2× FA1；A100 50–73% 峰值；225 TFLOPs/s（72% MFU） |
| FA3 | warp-specialization 异步重叠；ping-pong 交错；FP8 block quant + incoherent processing | H100 1.5–2.0× FA2；FP16 740 TFLOPs/s（75%）；FP8 近 1.2 PFLOPs/s；误差低 2.6× |

> **提点**：三代保持数学对象不变（精确等价，models 4.1 §5.2）——所以命题 1 的归约正确性与代际无关，变的只是「谁算哪块、何时算」。这是 §2 的账能跨 FA2 / FA3 / FlashInfer / FlashMLA 复用的原因：正确性证一次，策略随便换。

## 4. FlashInfer：BSR 统一格式、JIT 与 plan/run 接口

paged KV（2.1）、ragged 变长批、共享前缀、tree mask——四种 KV 形态要不要四套 kernel？FlashInfer 的答案是一套 block-sparse 模板加 JIT 实例化（[Ye et al., 2025](../references.md)，arXiv:2501.01005，[A]）。

所谓 BSR（block-sparse row），是指把非零元素组织成定长方块、按行压缩存储索引的稀疏格式。FlashInfer 用它统一 KV 表示：Q 与 O 是 ragged tensor（变长请求拼接、无 padding），KV cache 是 BSR——列块大小 = page size b（由 KV 管理算法决定，2.1），行块大小 = query tile 尺寸（编译期决定），任意 (行块, 列块) 组合都支持（论文口径）。页表就是 BSR 的 indices 数组：2.1 式 (3) 的地址翻译在这里变成稀疏矩阵的非零块索引，gather 语义不变。论文 Fig 2 把页表画成行块 4、列块 1 的 BSR；§2 的 split-KV 决策、§5 的 MLA paged KV 都是这个格式的特例。

KV layout 是模板参数：NHD = [num_pages, b, H_kv, d_head]，HND = [num_pages, H_kv, b, d_head]（`flashinfer/utils.py` L57-58，NHD = 0 / HND = 1，2026-11-23 核实）。差别在访存粒度：代入 b = 64、H_kv = 8、d_head = 128、bf16，一页 128 KiB；NHD 下同一 head 相邻 token 的 stride 是 8 × 128 × 2 = 2048 字节（散读），HND 下同一 head 的页内 slice 是 64 × 128 × 2 = 16 KiB 连续。decode 逐 head 读 KV，HND 的合并访存占优；prefill 全 head 一起读，NHD 的 token-major 占优——layout 选择是 workload 依赖的，JIT 各实例化一份。

JIT 编译 attention variants（`flashinfer/jit/core.py` `gen_jit_spec` L760）：variant 规格是一组 CUDA 代码字符串 functor（score 修改、query/key 变换、mask 逻辑），填进 kernel 模板后运行时编译缓存。tile 尺寸编译期定、序列长度运行时定——两者分离是 CUDA Graph 兼容的前提（3.4 的 piecewise 语义同构）。算法底座按硬件分：FA2 算法用于 ≤ sm89，FA3 算法用于 Hopper（论文口径）。投机解码的 tree mask 就是一个 variant：packed custom_mask（`flashinfer/prefill.py` `maybe_packed_custom_mask`）把逐 (query, key) 对的可见性压成位数组——verify 的 γ+1 个 query 共享一份 KV，mask 编码树祖先关系（tree attention 的算法与正确性归 4.4，[Miao et al., 2023](../references.md)，arXiv:2305.09781，[A]）。数值：γ+1 = 8、T = 4096 时 bool mask 32 KiB，packed 后 4 KiB——mask 字节缩 8 倍，kernel 按位判定。

plan/run 两阶段 API（`flashinfer/decode.py` `BatchDecodeWithPagedKVCacheWrapper` L1338，plan L2179 / run L3237，2026-11-23 核实）：plan 在 CPU 侧吃序列长度信息，产出 split 决策、每 CTA 工作队列、partial/final 输出索引映射，异步拷进 GPU workspace；run 启动 persistent kernel 消费 plan。plan 可在同一 generation step 的全部 L 层复用（序列长度相同）——CPU 调度开销被 L 层摊薄。CUDA Graph 下 workspace 按上界预分配、地址固定，split 决策在 capture 时冻结（§2 的「恒切」）。load-balanced scheduler 受 Stream-K 启发但**去掉 atomic 聚合**：相同序列长度输入产生相同归约顺序，输出确定（论文口径）——这是 batch invariance（1.4）在 attention kernel 侧的前提，kernel 账归 3.4。

cascade attention（术语对应：cascade 是 repo API 名——`MultiLevelCascadeAttentionWrapper`，`cascade.py` L226，plan L401 / run L537；论文称 composable formats，全文无 cascade 一词）。机制：共享前缀 + 独立后缀两级分解。论文 Fig 3 口径：前缀子矩阵用大行块（如 3×1，3 个 query 在 shared memory 里共享一次前缀 KV 读取），后缀用 1×1 块（每 query 走 global memory / L2 读自己的后缀）；不移动数据，只计算 indices 与 index pointer。

**命题 2**（cascade 两级正确性）。设 B 条请求共享前缀 P（长 T_p），请求 i 的独立后缀为 U_i，P 与各 U_i 位置集不相交，请求 i 的可见集 V_i = P ∪ U_i；行块大小（query tile 行数）记 r_q。级 1 对每个 query 算前缀状态 (O_P(q_i), LSE_P(q_i))，级 2 算后缀状态 (O_{U_i}, LSE_{U_i})，按式 (3) 合并——输出与直接对 V_i 做 attention 逐点相等；前缀 KV 的物理读取次数从 B 次降为 query tile 数 $`\lceil B/r_q \rceil`$ 。

**证明**（路线：不相交并 → 命题 1 → 前缀不可变保证共享状态合法；IO 按读取计数）。正确性：V_i = P ⊔ U_i 是不相交并，命题 1 直接给出合并状态 = 全集状态。前缀状态对所有请求合法的前提是前缀 KV 内容逐请求相同——这由前缀不可变性保证（k_s, v_s 只依赖 x_{≤s}，models 2.4 命题，recap 一句）；物理共享由 2.1 的 ref_cnt / COW 与 2.2 的 prefix caching 机制落地，kernel 侧只读一份。IO：朴素做法每请求读前缀一遍，流量 B·T_p·κL；cascade 每 query tile 读一遍，流量 $`\lceil B/r_q \rceil \cdot T_p \cdot \kappa L`$ ；后缀 B·T_s·κL 不变。边界：B = 1 或无共享前缀时 cascade 退化为单级加一次多余 merge（纯开销）；后缀之间也有共享（beam search 演化共享，2.1 §5）时两级不够，`MultiLevelCascadeAttentionWrapper` 支持多级递归——每多一级共享就多一次 ⊕。∎

等长后缀口径的 IO 账（行块容纳全部 B 条）：

$$
M_{\mathrm{naive}} = B (T_p + T_s) \kappa L, \qquad M_{\mathrm{cas}} \approx (T_p + B T_s) \kappa L \qquad (5)
$$

**符号**：T_p = 共享前缀长；T_s = 每请求独立后缀长（等长口径）；κL = 每 token 全层 KV 字节（1.2）；M = 每步 attention 的 KV 读流量（全层合计）。代入章脊口径（B = 32、T_p = 2048、T_s = 256、κL = 128 KiB）：M_naive = 32 × 2304 × 128 KiB = 9 GiB，M_cas ≈ 2048 × 128 KiB + 32 × 256 × 128 KiB = 256 MiB + 1 GiB = 1.25 GiB——省 86.1%，前缀流量从 8 GiB 压到 256 MiB。论文 e2e 实测（MLC-Engine、Llama-3.1 8B/70B、ShareGPT、请求率 16）：收益峰值在 n = 4，ITL 降 13.73%（8B）/ 17.42%（70B），TTFT 降 16.41%（8B）/ 22.86%（70B）（[A]，论文 §4.4 parallel-generation 评测口径）。e2e 远小于 86% 的两个机制：attention 只占步时一部分（Amdahl，§2 同账）；ShareGPT 短序列下 attention 份额低，n 再大收益趋于平台（论文口径）。

与 2.2 的分工：prefix caching 省容量与 prefill 计算（跨时间复用），cascade 省同一步内 decode attention 对前缀的重复读取（跨请求带宽共享）——两者正交，可叠加。

```
cascade 两级（B=4 共享前缀 P，行块=4 示意）
──────────────────────────────────────────────
级 1：前缀 P（T_p=2048）KV 读一遍进 shared memory
   q_1 q_2 q_3 q_4 各算 (O_P(q_i), LSE_P(q_i))
   ← 大行块 BSR：4 query 共享一次前缀读取
级 2：各请求独立后缀 U_i（1×1 块 BSR）
   q_i 算 (O_{U_i}, LSE_{U_i})
合并：merge_state（式 (3)）逐 query
   (O_i, LSE_i) = 前缀状态 ⊕ 后缀状态
前缀 HBM 读取：B 次 → 1 次（每 query tile）
```

## 5. FlashMLA：吸收形 MLA decode 的 kernel 几何

64×512 的输出矩阵要占 32,768 个 32-bit 寄存器，而每个 SM 只有 65,536 个——FA3 的 ping-pong（两份输出矩阵交替）在 MLA decode 上直接放不下。FlashMLA 的解法是把输出竖切两半、两个 KV block 做「跷跷板」（seesaw）。本节走读锚旧 commit `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`（README 自链）+ `docs/20250422-new-kernel-deep-dive.md`（仍在当前 repo）：2026.09.30 release 已移除 Hopper 架构与 V3/V3.2/V4.0 支持并改 FP8/FP4 KV 格式，当前 main 面向 V4.1（B200 + Ascend 950）的 sparse / fused kernel，详展归 7.3（[DeepSeek-AI, FlashMLA](../references.md)，教学参照，2026-11-23 核实）。

> **坑**：把 FlashMLA 当前 main 当 Hopper MLA decode 读。触发条件：2026.09.30 之后 clone main 找 V3 的 bf16 MLA decode kernel；现象：代码已移除，只剩 V4.1 FP8/FP4 sparse 与 fused norm-RoPE-attn-RoPE-cast kernel，按旧文档对代码会全部落空；规避：Hopper 机制锚 ba89a34 + deep-dive doc（本节口径），当前 main 的 V4.1 内容归 7.3。

吸收形等价 recap（models 4.1 §4.3-4.4 已完整推导，此处给关键步 + serving 含义；符号翻译：models 的 d_c → d_latent = 512、d_R → d_rope = 64、d_h = nope head 维 128、d_c' = q_lora_rank 1536，V3 口径 [A]）。K 侧把上投影吸进 query：

$$
s_{t,s,h} = \frac{\left\langle \tilde{q}_{t,h},\thinspace c_s^{KV} \right\rangle + \left\langle q_{t,h}^{R},\thinspace k_s^{R} \right\rangle}{\sqrt{d_h + d_{\mathrm{rope}}}}, \qquad \tilde{q}_{t,h} = \tilde{W}_h c_t^{Q}, \quad \tilde{W}_h = (W_h^{UQ})^{\top} W_h^{UK} \qquad (6)
$$

**符号**：c_s^{KV} = 缓存的 512 维 latent；$`\tilde{W}_h`$ = 第 h 头合并算子（1536×512，离线预计算，无逐位置量）；$`\tilde{q}_{t,h}`$ = 吸收后 query（512 维）；q^R / k^R = decoupled RoPE 通道（64 维，不吸收）。V3 代入：score 内维 512 + 64 = 576，较展开形 192 涨 3 倍——吸收的 FLOPs 代价（models 4.1 §4.4 recap）。V 侧把加权和提到上投影之前：

$$
m_{t,h} = \sum_{s} p_{t,s,h} \thinspace c_s^{KV}, \qquad o_t = \sum_{h=1}^{H_q} \widehat{W}_h \thinspace m_{t,h}, \qquad \widehat{W}_h = W_h^{O} W_h^{UV} \qquad (7)
$$

**符号**：m_{t,h} = latent 空间加权和（512 维；沿 models 4.1 记号，与 §2 的 running max m_i 无关）；p = softmax 权重；$`\widehat{W}_h`$ = 吸进输出投影的合并算子。

**命题 3**（吸收形等价）。吸收形（式 (6)(7)）与展开形（物化 k^C = W^{UK} c、v^C = W^{UV} c 后做标准 attention）计算的 o_t 逐点相等。

**证明**（两个关键步，recap models 4.1 式 (11)-(14)）。K 侧：矩阵乘结合律（转置规则）给出

$$
(W_h^{UQ} c_t^{Q})^{\top} (W_h^{UK} c_s^{KV}) = (c_t^{Q})^{\top} (W_h^{UQ})^{\top} W_h^{UK} c_s^{KV} = \langle \tilde{q}_{t,h}, c_s^{KV} \rangle
$$

——score 直接打在 cached latent 上，k^C 从计算图消失。V 侧：线性性给出 $\sum_s p_{t,s,h} W_h^{UV} c_s^{KV} = W_h^{UV} \sum_s p_{t,s,h} c_s^{KV}$ ——先加权和再上投影；输出投影吸收 $W_h^{O} W_h^{UV}$ 。RoPE 部分不吸收（k^R 逐位置旋转、原样缓存）：decoupled 结构是吸收的前提，低秩与 RoPE 的冲突推导归 models 4.1 §4.1-4.2，不重推。∎

serving 含义一（几何与读取）。score 内维从 192 涨到 576（逐对 FLOPs ×3），KV 读从展开形每 token 每层 32832 元素降到 576 元素（57× 削减，models 4.1 §4.3 账 recap）。代价侧（本章账，非 models 4.1 §4.4 原文账）：Q 侧合并矩阵每 head 1536×512 = 786432 元素，是展开形两因子（128×1536 + 128×512 = 262144）的 3 倍，参数字节同比例放大；decode 逐步 query 投影 FLOPs 则是展开形的 4 倍（2×1536×512 对 2×1536×128——展开形 k^C 已缓存，逐步只投 q^C）。score 逐对 FLOPs ×3 才是 models 4.1 §4.4 的代价账（recap）。decode memory-bound 下用 FLOPs 换 57× 读取削减是净赚，prefill / 训练走展开形（models 4.1 §4.4）。kernel 几何：K block 64×576 = 512 latent + 64 rope，V 就是同一份 512 维 latent（deep-dive 逐字：「in MLA, K and V are the same with different names」）；576 = 9 × 64，故一个 K block 拆 9 次 64×64 的 TMA copy，第一份落地即启动第一段 GEMM（fine-grained copy-GEMM pipelining，策略级；TMA 指令语义归 gpu）。paged KV：块表 + cache_seqlens 输入，每 token 576 元素连续存放，kernel 按 64×576 tile 读取——2.1 式 (3) 的地址翻译同形。

serving 含义二（regime 翻转）。吸收形 decode attention 的算术强度（deep-dive doc 公式，翻译：doc 的 d_k = d_latent + d_rope = 576、d_v = d_latent = 512、s_q = 每请求 query token 数，decode 无投机时 = 1）：

$$
I_{\mathrm{MLA}} = \frac{2 H_q s_q T (d_k + d_v)}{2 T d_k} = H_q s_q \cdot \frac{d_k + d_v}{d_k} \qquad (8)
$$

**符号**：分子 = score + 加权和 FLOPs（2 H_q s_q T (d_k + d_v)）；分母 = KV 读字节（bf16、K/V 同源只读一遍，2 T d_k；略去 q 与输出读写，doc 近似，T ≫ H_q s_q）；I_MLA 单位 FLOP/byte。推导一步：FLOPs 与字节各按定义写出、T 对消即得。代入 V3（H_q = 128、s_q = 1）：128 × 1088/576 = 241.8 FLOP/byte。doc 取 (d_k + d_v)/d_k ≈ 2 近似，阈值 H_q s_q ≥ 128（= 865/3.35/2 ≈ 129 的取整，H800 throttled 峰值 865 TFlops、带宽 3.35 TB/s），判 compute-bound；精确读数 241.8 是 throttled ridge 258.2 的 94%——更精确的说法是 compute-bound 边缘，kernel 按 compute-bound 设计（doc 口径）。doc 的部署前提：DeepSeek decode 实例不用 TP（deep-dive 引官方 inference system overview 口径），H_q = 128 全量在单卡；若 TP 切 head，则 H_q 降、强度同降（TP 账归 8.1）。对照 8B GQA decode attention：I = H_q / H_kv = 32/8 = 4（FLOPs 4T H_q d_head ÷ 字节 4T H_kv d_head），memory-bound。机制：MLA 一份 latent 全头共享（H_kv = 1 式的复用），K/V 同源再省一份读取——「宽头 MQA」；吸收的「FLOPs 换字节」交易在 H_q = 128 时把 decode attention 从 memory-bound 推到 ridge 边缘。s_q > 1（投机 verify）强度再乘 s_q——verify 是 prefill 形态（4.4 / 4.5）。

寄存器预算 → seesaw（策略级）。输出矩阵 64×512 fp32 = 32,768 个寄存器，SM 寄存器文件 65,536 个——一份输出占一半，两份（FA3 ping-pong 的前提）放不下，因为其余状态还要占寄存器。解法：输出竖切 O_L / O_R（各 64×256 = 16,384 寄存器），两个 warpgroup 各持一半；每轮取两个 KV block，共享一个 running max m；每个半区更新时乘上「自上次更新以来全部 max 位移的累积因子」——O_R 的更新因子是两个 scale 因子之积（deep-dive 步骤 8），等于 exp(m_old − m_new)，m_old 为该半区上次更新时的基准、m_new 为当前全局 max（译注：deep-dive 的 scale 定义写作 exp(m_new − m)，与重缩放因子互为倒数；重缩放因子须 ≤ 1，本章按数学正确的 exp(m_old − m_new) 方向读）。正确性：这正是 online softmax 不变量（models 4.1 式 (16)）按半区维护——两个半区的贡献最终都落在同一全局 max 基准上，与命题 1 的两 block 合并同构；doc 逐字「mathematically equivalent to FlashAttention's online softmax algorithm」。splitkv_mla 与 combine 两个 kernel 经 programmatic dependent launch 重叠（策略级），tile scheduler 把 request × block 作业均衡分配到 SM——§2 split 决策的 MLA 实例。性能（doc / README 口径）：至多 660 TFlops（H800，throttled 峰值 865 的 76%）、至多 80% tensor core 利用率；memory-bound 设置 3 TB/s，比旧 ping-pong 版慢约 2%（doc 口径，可接受）。

open vs internal 差距（官方披露口径）：DeepInfer 官方披露「FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower」（doc L289，2026-11-23 核实），e2e 至多差 5%——SASS 级归因与详账归 7.3，本章只记口径。

> **提点**：吸收是模型侧设计（models 4.1），serving 后果却是 regime 翻转——同一笔「FLOPs 换字节」交易，H_q 小（GQA 组大小 4）时强度贴地、优化带宽；H_q = 128 时强度贴 ridge、优化算力利用率。模型几何决定 kernel 的优化方向，这是「模型侧账」与「算子侧账」的接缝。

## 6. paged attention kernel 与 vLLM backend 接口

块表在 2.1 是内存管理语义，到了 kernel 侧它是什么？是一个 gather 的 indices 数组——paged attention 就是 block-sparse attention 在稠密可见集下的特例。recap 2.1：slot(t) = bt[⌊t/b⌋]·b + r(t)（2.1 式 (3)），#filled 限定有效范围（尾块垃圾值不读），null block 占位免分支。kernel 视角：逻辑连续不变（causal mask 同形），物理散落经块表翻译；b > 1 让块内多位置并行（2.1 §2 论文口径）。

vLLM v1 的 backend 族（`vllm/v1/attention/backends/`，2026-11-23 main 核实）：`flash_attn.py` / `flashinfer.py` / `flex_attention.py` / `triton_attn.py` / `cpu_attn.py` 等，加 `mla/` 子目录（`flashmla.py` / `flashinfer_mla.py` / `cutlass_mla.py` / `indexer.py` / `sparse_swa.py` 等）——一个 kernel 族 × 硬件 × KV 几何一个 backend，registry 按平台与能力选择。机制例证（`flash_attn.py`，模块全量走读归 5.3）：`build()`（L909）产出 FlashAttentionMetadata（block_table L1063、scheduler_metadata、max_num_splits），`forward`（L1284）把它们传给 FA varlen 接口（L1583 调用，`return_softmax_lse=True`）。max_num_splits 三态（L952-965，机制摘录）：

```python
# vllm/v1/attention/backends/flash_attn.py build()（机制摘录）
max_num_splits = 0  # 0 = FA3 启发式，非 CUDA Graph 兼容
if self.use_full_cuda_graph and ...:
    max_num_splits = self.max_num_splits   # 固定上界，中间 buffer 预分配
if envs.VLLM_BATCH_INVARIANT:
    max_num_splits = 1                     # batch-invariant：禁分裂，确定性归约
```

中间 buffer 形状 [num_splits, num_heads, num_tokens, head_size]（源码注释口径）——固定 split 数才能预分配、才能 capture 进 CUDA Graph；batch-invariant 模式取 1 是干脆不切（归约顺序平凡确定），与「固定但 >1」不冲突：前者要的是顺序确定，后者要的是形状静态，两个约束各管一件事（3.4 详账）。

接口契约表（本章算子族的共同输入输出）：

| kernel 族 | KV 输入 | split / 调度接口 | 输出 |
| --- | --- | --- | --- |
| FA2/FA3 varlen | paged KV + block_table + seqused_k | scheduler_metadata、num_splits | O、LSE |
| FlashInfer plan/run | paged / ragged BSR（NHD/HND） | plan workspace（split 决策 + 工作队列） | O、LSE（attention state） |
| FlashMLA（Hopper，ba89a34） | paged KV + block_table + cache_seqlens | tile scheduler metadata、num_splits | O、LSE |
| DSA sparse（FlashMLA indices） | indices（页地址已 bake） | 同上一行 | O、LSE |

LSE 是标准输出而非可选项：FlashInfer 论文把 (O, LSE) 立为 canonical output、⊕ 立为 canonical reduction；FlashMLA 返回 (out, lse)；vLLM `return_softmax_lse=True`。用途三件：split-KV 归约（§2）、cascade 合并（§4）、跨实例 partial attention 合并（PD 分离 / context parallel，8.x）。引擎不再把 attention 当「返回 O 的函数」，而当「返回可合并状态的状态机」——这是本章算子族在接口层的共性，也是 §2 命题 1 的工程兑现。

## 7. DSA serving：indexer K cache、DeepSelect 与 candidate blocks

V4.1-Flash 的 index_topk = 512 [W]：主 attention 每步只读 512 行 KV，是全量 131072 行的 0.39%。但选出这 512 行本身要对全历史打分——indexer K cache 的读取反而成了带宽主项。稀疏化把瓶颈搬了家，本节把这条选择链的算子与账走读一遍。

recap models 4.2 §6（不重推）：lightning indexer 对全部历史逐 token 打分（ReLU 加权内积、小维 d_I、可 FP8），TopK 选中后 MLA 主路只 gather 选中行；cache 总量不变，变的是每步读哪几行。V3.2 选 K = 2048（论文口径）→ V4.1 index_topk = 512 / index_n_heads = 32 / index_head_dim = 128（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）。模型侧形式化归 models 维度（V4.1 待补）。背景对照：NSA 是块级选择（[Yuan et al., 2025](../references.md)，arXiv:2502.11089，[B]，无官方 repo）；DSA 最终选择是 token 级，但 serving 侧选择链含一个块级预选环节（candidate blocks，下）——预选是加速手段，不改最终选择的粒度。

```
DSA decode 选择链（V4.1 口径，T=131072、K=512）
─────────────────────────────────────────────────────
indexer K cache [T, 128]/层（头间共享；生产 MXFP4）
   │ indexer 打分（DeepGEMM mqa_logits kernel，3.3）
   ▼ scores [T]（bf16）
candidate_block_max：按 b_c=16 池化 → 块分数 [T/16]
   │ 块级 TopK（K_b 块）→ 压实 K_b×16 列
   ▼
DeepSelect token TopK（K=512；式 (9) 期望账）
   │ restore：block_id×16+offset（2.1 式 (3) 同形）
   ▼ indices [B, s_q, 512]（页地址已 bake）
sparse MLA decode（FlashMLA）：只 gather 512 行
   ▼ (O, LSE)
主路读 20 MiB/步；indexer 读 1.25 GiB/步（bf16）→ 瓶颈搬家
```

indexer K cache：每 token 每层 d_I = 128 元素、头间共享一份（models 4.2 §6 口径）；DeepSelect 的 indexer 场景用 bf16，DeepInfer 生产 indexer KV 用 per-32 ue8m0 MXFP4（官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照；格式账归 2.3 / 3.2）。容量小（128 对主路 512），读取却是全量 T——账见式 (10)。打分 kernel（weighted ReLU MQA logits，fp8/fp4）属 DeepGEMM 组件（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照），走读归 3.3。

DeepSelect（[DeepSeek-AI, DeepSelect](../references.md)，教学参照，README + deep-dive 2026-11-23 核实）：DSA indexer 与 sampler 的 TopK kernel，2–20× vs torch.topk；indexer 场景 bf16 / topk ≤ 4096 / 主优化 topk = 512（与 V4.1 config 一致），sampling 场景 fp32 / vocab 约 128K；CUDA + Ascend（Ascend 仅 bf16）。性能度量 = 有效内存带宽（口径：TopK 无浮点运算，FLOP rate 无意义——README 逐字）。算法（deep-dive 口径）：维护当前第 K 大阈值（初始 −∞），三步循环——Scan（按 b_s 块读输入，随机块序）/ Filter（只有 > 阈值的元素进候选 buffer）/ Compact（buffer 达 K + b_2 触发 shared memory 内 radix-select，压回 K 个、更新阈值）。阈值单调不降，后面的块越难通过；随机块序保证对任意输入的期望界。每个输入元素恰读一遍（带宽最优），额外空间 O(K + b_s + b_2) 驻留 shared memory。期望处理量（doc 定理，符号翻译：doc 的 k → K、B → b_s、B_2 → b_2、N → T、L → C_buf = K + b_s + b_2，避免与 batch B、层数 L、block b 冲突）：

$$
E[W_{\mathrm{sel}}] \le \left( 1 + \frac{K}{b_2} \right) (K + b_s + b_2) \thinspace H_m, \qquad m = \lceil T / b_s \rceil, \quad H_m = \sum_{i=1}^{m} \frac{1}{i} \qquad (9)
$$

**符号**：W_sel = 全部 radix-select 调用处理的元素总数（doc 记 W，译名避与 1.2 权重字节 W 冲突）；b_s = 扫描块长；b_2 = 压实触发增量；H_m = 调和数。路线一句（doc 证明骨架，分析设元素两两不同，doc 注明同分不恶化界）：随机块序下第 i 块新 append 的元素在已见元素中秩 ≤ C_buf，故 E[A_i] ≤ C_buf / i，对 i 求和得调和数；W_sel = A + K·R（A 为 append 总数，R 为循环内压实次数，R·b_2 ≤ A）。b_s, b_2 = Θ(K) 时额外计算 O(K log(T/K)) ≪ T。代入章脊口径（T = 131072、K = 512、b_s = b_2 = 1024）：m = 128、H_128 ≈ 5.433，E[W_sel] ≤ 1.5 × 2560 × 5.433 ≈ 20863 个元素 ≈ T 的 16%；输入读一遍 = 256 KiB/行（bf16）。指令级优化（PTX 比较 / 位操作、浮点加法代整数加法等）归 gpu 维度。

candidate blocks（DeepInfer `deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py`，W1 同日 tarball 核实）：两级选择——(1) `candidate_block_max` 把 token 分数按 b_c（8 或 16）池化成块分数（块内有效 token 取 max），最后一个有效块强制 +inf（当前块必选）；(2) 块级 TopK 选 K_b 块；(3) 压实选中块的 token 分数（K_b · b_c 列），第二级 token TopK 取最终 K 个；(4) `restore_candidate_indices_inplace` 把压实索引还原成序列局部索引：block_id × b_c + offset——与 2.1 式 (3) 的块表算术同形。行 padding 到 1024 字节对齐 DeepSelect 的 stride 要求（接口事实，README `get_stride_requirement`）。

**命题 4**（预选精确性条件）。考虑「块 max 池化 → 按块分数选 K_b 块 → 候选内第二级 token TopK」的两级选择（无强制名额）。设 token 分数两两不同（bf16 indexer 分数可能同分；同分时 top-K 集本身依赖 tie-breaking，结论另议），τ 为第 K 大的 token 分数。若 K_b ≥ K，则不丢失任何真实 top-K token（精确）；若 K_b < K，则落选的 top-K token 数恒 ≤ K − K_b，且存在输入使损失恰达 K − K_b。

**证明**（路线：阈值 τ → 含 top-K token 的块分数 ≥ τ → 这类块至多 K 个 → K_b ≥ K 全收；损失上界分两种情形）。分数两两不同，故分数 ≥ τ 的 token 恰 K 个。含 top-K token 的块，其 max ≥ τ；不同块含不同 token，故块分数 ≥ τ 的块至多 K 个。K_b ≥ K 时按块分数排序必全收这至多 K 个块，第二级在容量 K_b · b_c ≥ K 的候选上取 top-K 即精确还原。K_b < K 时分两种情形：若块分数 ≥ τ 的块不足 K_b 个，则含 top-K token 的块全部入选（它们排在一切分数 < τ 的块之前），损失 0；否则入选的 K_b 个块的块分数均 ≥ τ，每块至少含 1 个 top-K token，故至少收进 K_b 个，损失 ≤ K − K_b。上界可达：K = 512 个 top token 分散在 512 个不同块（T ≥ 8192 时可能，每块恰 1 个），K_b = 64 只选 64 块，恰 448 = K − K_b 个真实 top-K token 落选。∎

实现口径注：DeepInfer 把最后一个有效块强制 +inf（当前块必选），占用 K_b 个名额之一——分数互异前提下，精确性条件收紧为 K_b ≥ K + 1（给强制块留一个名额）；K_b = K 时强制块可能挤掉一个真实 top-K 块。边界代入：K_b = 513、b_c = 16 时候选容量 8208，第二级工作集约全量 T = 131072 的 1/16 且精确；K_b 不足则是近似选择，召回风险与 models 4.2 §8 同形（稀疏 / 驱逐的共同软肋）。K_b 取值是部署参数，本章不虚构。

> **坑**：把 block-max 预选当无条件无损。触发条件：预选名额不足（无强制口径 K_b < K；强制当前块口径 K_b ≤ K）且 top-K token 分散；现象：真实高分 token 不在候选块内，第二级 TopK 只能在候选里选次优——输出静默退化，不报错；规避：核对 K_b ≥ K + 1（含强制块的精确口径，分数互异前提），或显式接受近似口径并评估召回（models 4.2 §8）；分数同分时精确性还依赖 tie-breaking。

读取账（V4.1 反事实全层 full attention、bf16 教学口径；实际混合几何归 7.2，换口径原因：本节只对比主路与 indexer 两笔读，不展开层分布）：

$$
M_{\mathrm{main}} = K \kappa L, \qquad M_{\mathrm{idx}} = T d_I \cdot \mathrm{bytes}_I \cdot L \qquad (10)
$$

**符号**：M_main = 主路每步 KV 读字节（选中 K 行）；M_idx = indexer 每步 K cache 读字节（全量 T 行）；d_I = index_head_dim = 128 [W]；$`\mathrm{bytes}_I`$ = indexer 每元素字节数（bf16 = 2）；κ = 512 × 2 = 1024 字节（MLA bf16；V4.1 共享 KV 向量 512 元素 = 448 latent + 64 rope，见 1.2 §3，V3 的 576 不适用）；L = 40。代入 T = 131072：M_main = 512 × 1024 × 40 = 20 MiB（全量对照 5 GiB，0.39%）；M_idx（bf16）= 131072 × 128 × 2 × 40 = 1.25 GiB = 64× M_main。换 MXFP4 indexer（每元素 0.5 + 1/32 = 0.53125 字节，[OCP MX](../references.md)，[C]）：M_idx ≈ 340 MiB，仍 17× M_main，较 bf16 省 3.76×。结论：DSA 把带宽瓶颈从主 KV 搬到 indexer K——indexer 精度 / 格式成为新杠杆（2.3 量化、3.2 格式账）；MLA 吸收形的 57× 读取削减（对展开形，models 4.1）与 DSA 的 256× 稀疏（131072/512）都叠加在这个杠杆之上。

sparse MLA 接口（FlashMLA README，2026-11-23 核实；Hopper kernel 内部锚 ba89a34）：indices 为 [B, s_q, K] int32，值 = 页块号 × page_block_size + 块内偏移——页地址已 bake 进 indices，kernel 不再消费 block_table（参数仍必传，None 即可）；无效位置 −1。对照 §6 的 dense paged 接口：dense 传块表、kernel 做翻译；DSA sparse 传翻译好的 indices、kernel 直接 gather——选择与寻址合进一个 tensor，这是稀疏化在接口层的落点。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| decode q = 1 塌缩 query 并行维，grid 退化为 B × H_kv；batch 1 用 <1% GPU（博客口径） | 式 (1)，8/108 ≈ 7.4% | §1、练习 1 |
| split-KV 三步：view 切段 → 段内 FA + LSE → LSE 加权归约；online softmax 两级使用 | 式 (2)(3) | §2、练习 2 |
| 命题 1：归约与全量 attention 逐点相等（精确非近似）；⊕ 结合交换，S 段任意序 | 命题 1、式 (3) | §2、练习 2 |
| 归约 IO 占比 ≈ 2S H_q (4d_head + 4)/(Tκ)：8B 满窗 S = 14 仅 0.086%；e2e 收益按 attention 份额折扣（8B B = 1 上界 ≈ 2.1×，博客 34B 口径 8×） | 式 (4) | §2、练习 1 |
| FA2 三技术（non-matmul / seq 并行 / warp partitioning）A100 50–73% 峰值；FA3 三技术（warp specialization / ping-pong / FP8 block quant + incoherent）H100 FP16 740 TFLOPs/s（75%）、FP8 近 1.2 PFLOPs/s | §3 表 | §3 |
| BSR 统一 paged / ragged / 共享前缀 / tree mask；列块 = b、行块 = query tile；NHD/HND 是模板参数（decode HND 合并访存占优） | §4 | §4 |
| plan/run 分离编译期 tile 与运行时序列长；plan 跨 L 层复用；CUDA Graph 静态配置 + 确定性归约（无 atomic） | §4 | §4、练习 4 |
| 命题 2：cascade 两级合并 = 全量 attention（不相交并 + 命题 1）；前缀读 B → ceil(B/行块) 次；B = 32 例省 86.1% IO，论文 e2e 峰值 ITL −13.73%（8B，n = 4） | 命题 2、式 (5) | §4、练习 3 |
| 命题 3：吸收形 ≡ 展开形（结合律 + 线性性）；score 内维 576、读取 57× 削减；I_MLA = H_q s_q (d_k + d_v)/d_k ≈ 242（V3），H800 throttled ridge 的 94%——compute-bound 边缘 | 命题 3、式 (6)-(8) | §5 |
| 寄存器预算 32768/65536 → ping-pong 不可行 → O 竖切 + seesaw（半区 online softmax 不变量）；FlashMLA Hopper 锚 ba89a34（2026.09.30 移除） | §5 | §5 |
| 命题 4：分数互异下 block-max 预选 K_b ≥ K 精确（无强制名额），K_b < K 损失恒 ≤ K − K_b 且可达；实现强制当前块占一名额，条件收紧 K_b ≥ K + 1；DSA 把带宽瓶颈搬到 indexer K（bf16 口径 64× 主路，MXFP4 省 3.76×） | 命题 4、式 (9)(10) | §7 |
| LSE 是标准输出：split 归约 / cascade / 跨实例合并共用 ⊕；vLLM max_num_splits 三态（0 启发式 / CG 固定 / batch-invariant = 1） | §6 表 | §6、练习 4 |

## 练习

1.（计算 · [10]）8B GQA 口径（H_kv = 8、H_q = 32、d_head = 128、κ = 4096 字节，A100 108 SM 博客口径）：(a) B = 1 不切时 grid 的 CTA 数与 SM 覆盖率；(b) 铺满 SM 所需 S 与对应段长（T = 131072）；(c) fp32 partial 口径的归约额外 IO 及其占 KV 读的比例（式 (4)）。校验：(a) 8/108 ≈ 7.4%；(c) 比例 < 0.1%，与 §2 一致。
2.（推导 · [20]）补全命题 1 的 S 段归纳步：(a) 设前 k 段合并状态等于其并集的 attention 状态，证再并第 k+1 段仍相等（用式 (3) 与不相交性）；(b) 说明 ⊕ 的结合律为何保证「任意归约顺序结果相同」，并指出浮点非结合在哪一步破坏位级相等（1.4 batch invariance）；(c) 用 §2 数值例验证交换律：段 1 ⊕ 段 2 与段 2 ⊕ 段 1 的 O 应 allclose。校验：(c) 容差 1e-5；(a) 即命题 1 的归纳步。
3.（推导 · [30]）cascade 账与正确性：(a) 证明命题 2 的 IO 论断（前缀读取 B → ceil(B/行块) 次），指明哪一步用到前缀不可变性（models 2.4）；(b) B = 32、T_p = 2048、T_s = 256、κL = 128 KiB，用式 (5) 算 M_naive / M_cas 与节省比例；(c) 解释论文 e2e ITL 收益（13.73% @8B n = 4）为何远小于 IO 节省，给两个机制。校验：(b) 9 GiB → 1.25 GiB、86.1%，与 §4 一致；(c) 应含 Amdahl 折扣与短序列 attention 份额两条。
4.（代码 · [30]）读 FlashInfer `scheduler.cuh` L138-211 与 vLLM `flash_attn.py` L952-965（2026-11-23 main）：先写出预测——split_kv = false 的条件有哪些？CUDA Graph 与 batch-invariant 模式下 max_num_splits 各取什么？再对照源码验证；解释 CUDA Graph 要求固定 split 数、batch-invariant 要求 num_splits = 1，两个约束为何不冲突（固定 ≠ 1）。校验：源码注释「0 means use FA3's heuristics, not CG compatible」与「when using CUDAGraph, we always use partition-kv kernel」；预测应与这两条一致。

## 参考

- [Dao et al., 2023](../references.md)：Flash-Decoding for long-context inference，CRFM Stanford 博客，[C] 开放技术博客、非闭源，无 arXiv（key: flashdecoding）——三步 split-KV、occupancy 论断（A100 108 SM、batch 1 <1%）、8× e2e / 50× attention / 32k 恒定、FA ≥ 2.2 / xFormers ≥ 0.0.22
- [Dao, 2023](../references.md)：FlashAttention-2，arXiv:2307.08691，[A]（key: fa2）——三技术、~2× FA1、A100 50–73% 峰值、225 TFLOPs/s（72% MFU）
- [Shah et al., 2024](../references.md)：FlashAttention-3，arXiv:2407.08608，[A]（key: fa3）——三技术、H100 1.5–2.0×、FP16 740 TFLOPs/s（75%）、FP8 近 1.2 PFLOPs/s、误差低 2.6×
- [Ye et al., 2025](../references.md)：FlashInfer，arXiv:2501.01005，[A]（key: flashinfer）——BSR / composable formats / JIT / load-balanced scheduler / attention state ⊕；repo 路径 `cascade.py`（merge_state L42 / MultiLevelCascadeAttentionWrapper L226）、`scheduler.cuh`（split_kv L138-211）、`decode.py`（L1338 / plan L2179 / run L3237）、`utils.py`（NHD/HND L57-58）、`jit/core.py`（gen_jit_spec L760）、`prefill.py`（maybe_packed_custom_mask），2026-11-23 main 核实
- [DeepSeek-AI, FlashMLA](../references.md)：教学参照 + 官方实现（key: flashmla）——Hopper MLA decode 锚旧 commit ba89a34 + docs/20250422-new-kernel-deep-dive.md（64×576 K block、32768/65536 寄存器、seesaw、9 次 TMA copy、660 TFlops / 3 TB/s）；2026.09.30 移除 Hopper / V3 系，当前 main 归 7.3
- [DeepSeek-AI, DeepSelect](../references.md)：教学参照 + 官方实现（key: deepselect）——TopK 2–20×、bf16 indexer / fp32 sampling、有效带宽度量、scan/filter/compact 与期望界（docs/DeepSelect-deep-dive.md）
- [Kwon et al., 2023](../references.md)：PagedAttention，arXiv:2309.06180，[A]（key: vllm）——paged KV 与块表背景（2.1）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——`v1/attention/backends/{flash_attn,flashinfer,flex_attention,cpu_attn,...}.py` + `mla/`；`flash_attn.py` build L909 / max_num_splits L952-965 / forward L1284 / varlen L1583，2026-11-23 main 核实
- [DeepSeek-AI, DeepInfer repo](../references.md)：教学参照（key: deepinfer-repo）——`deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py`（candidate_block_max / restore_candidate_indices_inplace）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——FlashMLA open vs internal 5%–10%（L289）、indexer KV per-32 ue8m0 MXFP4，2026-11-23 核实
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：[W]（key: v41-flash-config）——index_topk 512 / index_n_heads 32 / index_head_dim 128、40 层（hf-mirror 传输核实，2026-11-23）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——V3 MLA 几何（H_q 128、512 + 64）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 SXM BF16 dense 989.5 TFLOPS、3.35 TB/s
- [Miao et al., 2023](../references.md)：SpecInfer，arXiv:2305.09781，[A]（key: specinfer）——tree attention 算法源头（正确性归 4.4）
- [Yuan et al., 2025](../references.md)：NSA，arXiv:2502.11089，[B] 无官方 repo（key: nsa）——块级稀疏背景对照
- [OCP MX](../references.md)：MX Specification v1.0，[C] 官方格式规范（key: ocp-mx）——MXFP4 每元素 0.5 + 1/32 字节
- [DeepSeek-AI, DeepGEMM](../references.md)：教学参照（key: deepgemm）——indexer MQA logits kernel（走读归 3.3）
- 跨维度指引（纯文字）：models 4.1 §5（online softmax 式 (15)(16)、tiling、IO 式 (17)）、§4.3-4.4（absorption 式 (11)-(14)）；models 4.2 §6（DSA indexer）/ §8（召回风险）；models 2.4（前缀不可变命题）；inference 1.2（κ 式 (3)、decode 账式 (2)(4)）、1.4（batch invariance）、2.1（块表式 (3)）、2.2（prefix caching）、2.3 / 3.2（KV 量化与格式账）、3.3（DeepGEMM）、3.4（融合 / CUDA Graph / batch-invariant kernels）、4.4 / 4.5（投机解码）、5.3（vLLM attention 全量走读）、7.2（V4.1 混合几何）、7.3（FlashMLA 当前 main 与 SASS 差距）、8.x（跨实例 partial attention 合并）；warp / tensor core / SASS 指令级归 gpu 维度
