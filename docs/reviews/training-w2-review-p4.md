# training W2 evaluator 批 2 审查报告 — 4.2/4.3/4.4 逐章 + 全波 6 章 sweep

日期：2026-11-17。审查人：独立 fresh-context evaluator（与写作 agent、批 1 evaluator 无共享上下文）。立场：假设内容有错，直到一手来源证明无误。
对象：training/04-data-parallelism/02-zero.md（推导重章）、03-fsdp.md、04-memory-and-comms-accounting.md 逐字审查；training/03-loss-functions/ 与 04-data-parallelism/ 全 6 章跨章 sweep。
方法：ar5iv 全文逐条对照（1910.02054 / 2101.06840 / 2104.07857 / 2306.10209 / 2304.11277 / 2407.21783 / 2412.19437 / 2501.00656 / 2202.08906 / arXiv abs 2410.10989）；DeepSpeed / PyTorch / TorchTitan / TRL / Megatron-LM / Liger 代码经 GitHub contents API 下载原文核对；全部数值独立重算。

---

## 0. 两项裁定（任务书额外委托）

### 裁定 (a)：Llama-3 herd 论文 Table 4 是否含并行度数 —— **含，完整含**

ar5iv 2407.21783 §3.3.2 原文逐字（Table 4 及其前文）：

> "...we achieve an overall BF16 Model FLOPs Utilization (MFU; Chowdhery et al. 2023) of 38-43% for the configurations shown in Table 4. The slight drop in MFU to 41% on 16K GPUs with DP=128 compared to 43% on 8K GPUs with DP=64 is due to the lower batch size per DP group needed to keep the global tokens per batch constant during training."
>
> Table 4 表头与三行：**GPUs | TP | CP | PP | DP | Seq. Len. | Batch size/DP | Tokens/Batch | TFLOPs/GPU | BF16 MFU** —— 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43%；16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41%；16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38%。表注："Scaling configurations and MFU for each stage of Llama 3 405B pre-training."

**结论**：并行度数（TP/CP/PP/DP 全四维数值）在 herd 论文正文 Table 4 中。spec 审查 B 的判定「度数 ✗ 不在正文（TP8/CP16/PP16 grep 全 0）」是 grep 模式失误（度数在表格里以分列数值出现，不作 "TP8" 连写）——**B 判错了**。4.4 章的 Table 4 转录（含三行全部十列数字）逐字核对无误。

**对 spec 的回改建议**：spec §5.5 大纲行「Llama-3 405B（16384 GPU 集群拓扑——**并行度数不在 herd paper 正文，⟨verify⟩ 标记保留**、只能引集群与中断统计）」应当改写：度数可直接引 Table 4（[A] 级一手），⟨verify⟩ 标记可移除。spec §6.3 及 5.5 依赖该判定的措辞同步松动。

### 裁定 (b)：ZeRO-Offload Table 1 的 4M/8x 算术不一致 —— **原文确实如此，4.2 的处理口径合理**

ar5iv 2101.06840 Table 1 原文逐字（"Memory savings for offload strategies that minimizes communication volume"）：

```
FWD-BWD  p16  g16  Update   Memory  Reduction
gpu      gpu  gpu  gpu      16M     1x (baseline)
gpu      gpu  cpu  gpu      14M     1.14x
gpu      gpu  gpu  cpu      4M      4x
gpu      gpu  cpu  cpu      4M      8x        ← 末行：4M 与 8x 并存
```

末行算术不一致属实（16M/4M = 4x ≠ 8x）。机制侧核对：论文 §3.5（唯一最优策略 = fp16 参数 + 前向反向在 GPU，g16 与 Update 超节点全去 CPU）与 §4.1（"transfer these gradients... in small groups to the CPU memory immediately after they are computed. Therefore, only a small amount of memory is required to temporarily hold the gradients on the GPU"）自洽的 GPU 驻留是 ≈2M（fp16 参数 2M + 有界梯度暂存），对应 8x——表内 4M 应为原文转写之误（疑似从上一行复制）。

**4.2 的处理**（照录原表 + 口径注记指明 4N 与 8x 不一致、按机制口径理解为 ≈2N）正确且诚实，优于悄悄改表或照抄不注。**裁定：writer 上报属实，章节处理批准，无需改动。**

---

## 1. 强制独立重推导（逐步中间过程）

### R1：4.2 ZeRO 各阶通信量与取回时序

约定：每 rank 带宽项发送量，以 bf16 元素计（梯度与参数各 N 元素）；ring/带宽最优集合通信下 reduce-scatter 与 all-gather 各发送 (d−1)/d·N，all-reduce = RS + AG 合计 2(d−1)/d·N。

**ZeRO-1/2 与 DP 严格相等的论证**：
1. DP 基线每 step：一次梯度 all-reduce = 2(d−1)/d·N ≈ 2N（4.1 式 (2)，M = 2N bytes 即 N 元素）。
2. ZeRO-1 每 step：梯度 reduce-scatter（各 rank 收自己分区的归约梯度）发送 (d−1)/d·N；更新后参数 all-gather（bf16，N 元素）发送 (d−1)/d·N。合计 2(d−1)/d·N。
3. 逐项对照：DP 的 RS 段 ↔ ZeRO-1 的梯度 RS（同为 N 元素梯度，内容同）；DP 的 AG 段 ↔ ZeRO-1 的参数 AG（同为 N 元素 bf16 张量——字节数相同，内容从「归约后梯度」换成「更新后参数」）。**严格相等**（不仅是渐近），系数同为 2(d−1)/d。
4. 论文对照：§7.2.1 原文 "So the total communication volume per training step is Ψ+Ψ = 2Ψ, exactly the same as the baseline DP" ✓；§7 引言 "P_os and P_g... no additional communication" ✓。

**ZeRO-3 = ×1.5 的推导**：
1. 前向：每层参数用前 all-gather、用后释放；全模型累计每 rank 发送 (d−1)/d·N ≈ N。
2. 反向：matmul 反向 dX = GW^⊤ 需要全量 W（1.2 的算子账）——参数必须再聚一次（逆序），再 ≈ N。
3. 梯度 reduce-scatter ≈ N。
4. 更新后参数同步 = 0：各 rank 只更新自己的 fp32 master 分片并写回自己的 bf16 分片，下个 step 反正重新 gather——对照 ZeRO-1 的第 4 步，这次 AG 被省掉。
5. 合计 3N；C(ZeRO-3)/C(DP) = 3N/2N = 1.5。论文对照：§7.2.2 原文 "The total volume is therefore 3Ψ which is 1.5x compared to the baseline"；§7 引言 "incurs a maximum of 1.5x communication"（上限口径 ✓，章节标注「上限」正确）。

**取回时序图逐步核对**（4.2 §4.1 ASCII 图）：前向 L1→L4 各层 AG(L)→fwd→释放；反向 L4→L1 各层 AG(L)→bwd→RS(grad)。论文 §7.2.2 原文："reschedule the parameter all-gather by spreading it across the entire forward propagation, and discarding the parameters once they have been used... this all-gather needs to happen once again for the backward propagation in the reverse order... in addition to the communication volume incurred by the reduce-scatter of the gradients" —— 图的每一拍与原文逐步对应 ✓。DeepSpeed 实现侧（stage3.py 的 ipg bucket RS + parameter_offload.py 的模块级 fetch/release hook + coordinator 的预取）与时序图同构 ✓（代码核实见 verification log #12-15）。

### R2：4.4 总账公式与数值表行重算

式 (1) 五项分解 M_gpu = M_param + M_grad + M_opt + M_act(κ) + M_buf：与 4.2 各阶显存式（Mem₁ = 4N+12N/d、Mem₂ = 2N+14N/d、ZeRO-3 = 16N/d）逐项一致，与 4.3 式 (2)（常驻 16N/d + 暂态 2P_u(2+n_pf)）的拼接无冲突 ✓。

**表行重算 1（405B 行，GiB = 2³⁰）**：N = 405×10⁹。16N = 6.48×10¹² B = 6035.0 ✓；Z1@8 = (4+12/8)N = 5.5N = 2.2275×10¹² = 2074.5 ✓；Z2@8 = 3.75N = 1414.4 ✓；Z3@8 = 2N = 754.4 ✓；Z3@64 = 16N/64 = 94.3 ✓；Z3@512 = 11.79 ✓（表值 11.8）。
**表行重算 2（1T 行）**：16N = 1.6×10¹³ = 14901.2 ✓；Z1@8 = 5.5×10¹²... 即 5500×10⁹ B = 5122.3 ✓；Z2@8 = 3750×10⁹ = 3492.5 ✓；Z3@8 = 1862.6 ✓；Z3@64 = 232.8 ✓；Z3@512 = 29.1 ✓；正文另引 Z3@1024 = 15.625×10⁹ = 14.55 ✓（表注 14.6）。
**临界算例重算（70B Z3@8 vs H200 141）**：三态 130.4（= 16×70/8×10⁹/2³⁰ = 130.38 ✓）；暂态 6N/L = 6×70×10⁹/80 = 5.25×10⁹ B = 4.89 ✓（4.9）；激活 full checkpoint 2sbhL = 2×4096×1×8192×80 = 5.37×10⁹ B = 5.00 ✓；合计 140.3 GiB、余量 0.7 GiB ✓。
**通信-时间换算重算**：7B 梯度 28×10⁹ B ✓；d=8 @900 GB/s → 31.1 ms ✓；d=512 每 GPU 7812.5 token、计算 0.82 s ✓；通信 28×10⁹/50×10⁹ = 0.56 s、占比 0.56/0.82 = 68% ✓；ZeRO-3 档 42×10⁹ B → 0.84 s > 0.82 s ✓；hybrid d_s=8 → 2×2N/8 = 3.5×10⁹ B → 70 ms ✓。
**MFU 重算**：Llama-3 隐含峰值 430/0.43 = 1000 ✓；step 时间 = 6×405×10⁹×16×10⁶/(8192×430×10¹²) = 11.0 s ✓；DeepSeek-V3 反推 6×37×10⁹×10¹²/(1.8×10⁵×3600) = 3.43×10¹⁴ FLOPS ✓，对 989 TFLOPS = 34.7% ✓，对 1979 TFLOPS = 17.3% ✓（与 models 1.2 §9 同式同值，交叉一致）。

---

## 2. Findings 表

### S1（事实性致命）：0 条

### S2（实质问题）：4 条

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | training/04-data-parallelism/01-dp-and-ddp.md 式 (2)（第 58 行，display 块） | `2 \thinspace \frac{d-1}{d} \cdot M \;\approx\; 2M` 使用了禁用命令 `\;` | STYLE_GUIDE §3 反转义陷阱（实测）：GitHub 管线把 `\;` 反转义为可见 `;`——该式渲染为「…M ;≈; 2M」。白名单 §3.1：间距只许 `\thinspace`/`\hspace{5mu}`。全 training 树 grep 确认这是唯一残留（批 1 漏网，sweep 捕获） | 改为 `\hspace{5mu}\approx\hspace{5mu}` 或去掉间距 |
| S2-2 | training/04-data-parallelism/02-zero.md §2 末（第 50 行） | 「第 2 步用 reduce-scatter 还是『all-reduce 后各取自己那片』在通信量上同价（ring 口径都是约 2N 元素）」 | 算术不成立：full all-reduce（RS+AG，2N）拿全量归约梯度后各取分片，**参数 all-gather 仍不可免**（各 rank 只更新自己的 fp32 分片，下一 step 前向必须拿到全量新参数）——AR 变体合计 2N+N = 3N/step，比 RS 变体（N+N = 2N）贵 1.5×。代码对照：stage_1_and_2.py `average_tensor` 的 `if not self.reduce_scatter:` 分支走 `gradient_reduction_w_predivide` → `dist.all_reduce(...)`（全文核对）；论文 §7.2.1 的 2Ψ 等式依赖 gradient scatter-reduce（"instead of an all-reduce, ZeRO only requires a scatter-reduce"） | 改为：AR 变体每 step 多付一次 N 元素（3N vs 2N），这正是 stage 1/2 默认 `reduce_scatter=True` 的原因 |
| S2-3 | 4.2 §4.4（第 112-118 行）↔ 4.3 §5（第 87 行） | 跨章矛盾：4.2 称 ZeRO-2/3「除以 m 的省法不存在」、ZeRO-3 累加下为 3mN/step；4.3 称 FSDP2 `set_requires_gradient_sync` 下「1.5 的通信 ÷m 账原样适用」 | FSDP 论文 §3.3.4 明确两变体：without communication 时各 rank 累加**未分片**梯度（显存 +2N）、通信推迟到累加边界——梯度项 ÷m 存在，代价是 2N 梯度缓冲；而参数取回（前向+反向 AG，2N/micro-batch）任何配置下都 ×m 不摊薄。两章各说对一半：4.2 的「必须随每个 micro-batch 归约」只在保梯度显存的配置（DeepSpeed stage3 形态，`__param_id_to_grad_partition` 跨 micro-batch 累加，代码已核）下成立；4.3 的「原样适用」漏掉了参数 AG 不摊薄 | 两章统一口径：梯度归约 ÷m 可行、代价 2N 全量梯度缓冲（FSDP 无通信变体）；参数取回 ×m 恒不摊薄；4.2 的 3mN 结论标注「保梯度显存配置」限定 |
| S2-4 | training/04-data-parallelism/04-memory-and-comms-accounting.md §5（第 122 行） | 中断统计引「§3.3.1」 | 论文实际：466 次中断（47 计划 + 419 意外）、>90% effective training time、54 天快照均在 **§3.3.4 Reliability and Operational Challenges**；§3.3.1 是 Training Infrastructure（数字本身全部核实无误） | §3.3.1 → §3.3.4 |

### S3（minor，累计下波清零）：6 条

| # | 位置 | 断言 | 建议 |
| --- | --- | --- | --- |
| S3-1 | training/references.md | 被引章节回写缺口：llama3 缺 4.3/4.4；zero 缺 4.4；fsdp 缺 4.4；megatron-sp 缺 4.4；llama2 缺 4.4；pytorch-repo 缺 4.1/4.3（逐章正文核对） | 按写入规则回写（W1 先例同档处理） |
| S3-2 | 4.4 §1（第 15 行） | 「DDP 的梯度 bucket（默认按约 25 MB 分桶）」——4.1 用 25 MiB，单位漂移 | 统一 MiB |
| S3-3 | 4.4 §1（第 13 行） | 「矩估计降精度时按 (4+2c)N 口径换系数，1.3 §4」——(4+2c)N 公式在 models 1.2 式 (8)；1.3 §4 只有 DeepSeek-V3 定性实例（矩估计 BF16），无该式 | 指针改「models 1.2 式 (8)，实例见 1.3 §4」 |
| S3-4 | 4.4 §5（第 116 行） | 「43→41→38 分别对应…长上下文阶段 CP 开大后的额外开销」——论文只解释 43→41（batch/DP 减半，§3.3.2 原文已核）；41→38 的 CP 归因是本书分析，未标注 | 标注「本书分析」或软化措辞 |
| S3-5 | 4.2 §5.1（第 127 行） | 「（论文 §3.3，脚注：部分状态卸载可更低…）」——论文脚注原文是「memory savings per communication volume 之比在部分卸载下仍成立」；章节脚注（绝对通信量可低于 4N）数学正确，但措辞易被读成论文原注 | 改为「本书注记：…」 |
| S3-6 | training/01-autograd-and-backprop/04-gradient-checkpointing.md §5（第 159 行） | 前向指针「并行方案（P4 / P5 的内容）」为 Part 级；4.4 已存在，可精确到章（4.4 总账）——1.4↔4.4 激活账回链现只有 4.x→1.4 方向是章级 | 收紧为「总账归 4.4」 |

### Sweep 中判定为「接受/无需修」的存疑项（记录在案）

- 4.3 用 $G$ 表分片组大小：spec §4.3 硬规则禁止的是混场章用 $G$ 表 **DP 度数**；4.3 的 $G$ 是分片组大小、局部定义明确、并给了 $G \to d_s$ 的全书换算注。接受。
- 4.2 §5.1「4 倍于单独 MP」：Offload 论文贡献清单口径为 4x（"4x increase in model size over model parallelism"），摘要口径 4.5x——章节取贡献清单口径，可接受。
- 4.4 §3「余量回到 70+ GiB」：141−65.2 = 75.8，按三态余量读成立。接受。
- GPU 标称容量按 GiB 读（80/141/192）：与工具口径一致（H100 nvidia-smi 报 81920 MiB = 80 GiB 物理容量），章内已显式声明口径。接受。
- 4.3 策略映射表缺 `_HYBRID_SHARD_ZERO2` 行、增 `NO_SHARD` 行：相对迁移文档的编辑取舍，语义完整。接受。

---

## 3. Verification log（抽查项 × 来源 × 结论）

| # | 抽查项 | 来源（一手） | 结论 |
| --- | --- | --- | --- |
| 1 | ZeRO Table 1 全部三行标尺（7.5B@64: 120/31.4/16.6/1.88；128B@64: 2048/536/284/32；1T@1024: 16000/4011/2013/15.6）与 0.12 GB@1024 | ar5iv 1910.02054 Table 1 逐字 | ✓ 全对；粗体=32GB V100 装得下口径与表注一致 |
| 2 | K=12、(4+K)Ψ=16Ψ、4x/8x/Nd 渐近 | 同上 Figure 1 表注、§5.1/§5.2/§5.4 | ✓ 逐字 |
| 3 | ZeRO-1/2 与 DP 同通信量、ZeRO-3 "maximum of 1.5×"、3Ψ 构成、取回时序（spread across forward / once again backward / RS of gradients） | 同上 §7、§7.2.1、§7.2.2 逐字 | ✓ 逐字（含 "broadcast" 用词在 §5.3、all-gather 重述在 §7.2.2 的双口径） |
| 4 | ZeRO-R（P_a/P_b/M_D）与 §8 归属、§1 的 16TB/1024 卡/16GB | 同上 §8、§1 | ✓ 逐字 |
| 5 | Offload：4M 通信下界、6M 次优约束、四格表、唯一最优（§3.5）、梯度逐块即传（§4.1）、多卡 ZeRO 组合（§4.2）、CPU Adam 6x（§5.1）、DPU 至多 1.5x、DGX-2 PCIe 双向 32GBps（Table 2）、40 TFLOPS@10B、PyTorch 1.2B（贡献清单）/1.4B（摘要）@30 TFLOPS、13B 单卡 10x、70B 单节点 4x/4.5x、128 GPU 近线性 | ar5iv 2101.06840 §3-§6 逐字 | ✓ 全部命中（含裁定 b：Table 1 末行 4M/8x 不一致原文属实，章节处理批准） |
| 6 | Infinity：五组件名、32T@512×V100、>25 PFLOPS、单链路 10-12 GB/s（CPU≈100/NVMe≈25 GB/s 对照，脚注 6）、Table 3 前瞻 10x 算力→30 GB/s | ar5iv 2104.07857 §1/§4/§9 | ✓ 逐字 |
| 7 | ZeRO++：qwZ（FP16→INT8 block-based，Dettmers 引，自研 CUDA kernel）、hpZ（节点内完整副本、跨节点 M→0）、qgZ（INT4 块量化传输、归约前恢复全精度、all-to-all 范式）、3M→0.75M = 4x、云环境 <100Gbps 以太网口径 | ar5iv 2306.10209 §1/§2/§5 逐字 | ✓ 逐字（含 "reduction on low-precision values will accumulate and amplify the errors" 机制依据） |
| 8 | FSDP 论文：§3.2.1 的 1.5x 口径；式 (1) 2M(W−1)/(GW) 与两端点 2M(W−1)/W、3M(W−1)/W；flatten-concat-chunk 与 padding ≤ F−1；33M 元素悬崖（Figure 2(b)，2³⁰ fp32 固定总量）；backward prefetch 排序反转 + 逐 iteration 重录前向序；forward prefetch（CPU 发行速度）；§3.4 rate limiter + 挂起 CPU | ar5iv 2304.11277 §3.1-§3.4 逐字 | ✓ 全部命中。注：论文正文「F = W/G」句的 F/G 相对其 Figure 4（16 GPU = 2 分片组 × 8 复制组）自相矛盾（应为 F=G=8）；章节采用与公式自洽的读法（G=分片组大小）——与我的独立推导 2M/G 一致，处理正确 |
| 9 | Llama-3：Table 4 全三行度数/MFU（裁定 a）；FSDP reshard 引语；43→41 的 batch/DP 机制；38-43% 区间；MFU 归 Chowdhery et al. 2023；466/47/419/54 天/>90%；Table 3 的 126 层/16384 | ar5iv 2407.21783 §3.3.2/§3.3.4/Table 3 逐字 | ✓ 全部命中（唯 4.4 把中断统计引到 §3.3.1 → S2-4）；41→38 的 CP 归因论文未明言 → S3-4 |
| 10 | DeepSeek-V3：14.8T tokens、2.664M GPU-hours（预训练）/2.788M（全程）、180K/万亿 token、3.7 天@2048 H800、671B/37B 激活、§3.1 集群、§3.5.1 的 132 SM 划走 20 + Tensor Core 闲置 | ar5iv 2412.19437 摘要/Table 1/§3.1/§3.5.1 逐字 | ✓ 全部命中 |
| 11 | OLMo 2：Table 6 的 1.8/4.6 ×10²³（6ND 近似、Kaplan 口径）；§6.1.1 Jupiter 128 节点/1024 H100/每节点 8×400Gb/s IB；Figure 15 垃圾回收吞吐曲线；全文无 MFU 字样 | ar5iv 2501.00656 逐字 + 全文 grep | ✓ 全部命中（MFU 0 命中证实「不报」） |
| 12 | DeepSpeed：ZeroStageEnum（1=optimizer_states/2=gradients/3=weights）；stage3_prefetch_bucket_size 默认 5e7、stage3_param_persistence_threshold 默认 1e5；offload_param/offload_optimizer + OffloadDeviceEnum(cpu/nvme) | GitHub API deepspeedai/DeepSpeed master: runtime/zero/config.py | ✓ 逐字 |
| 13 | stage_1_and_2.py：DeepSpeedZeroOptimizer、构造默认 reduce_scatter=True、average_tensor 的 RS 分支结构（`if not self.reduce_scatter:` → all_reduce 分支——S2-2 的直接证据） | 同上 stage_1_and_2.py 下载核对 | ✓ |
| 14 | stage3.py：DeepSpeedZeroOptimizer_Stage3、_create_fp16_partitions_with_defragmentation、_create_fp32_partitions、create_reduce_and_remove_grad_hooks、__add_grad_to_ipg_bucket（独立 reduce_and_partition_stream）、__avg_scatter_contiguous_grads、partition_grads、_prepare_sub_group→unscale_and_clip→_optimizer_step（_cpu_adam_step_and_copy_back 快路径、DeepSpeedCPUAdam）→_reassign_or_swap_out_partitioned_parameters→_release_sub_group、__param_id_to_grad_partition、finalize_gradient_accumulation_boundary | 同上 stage3.py 下载核对 | ✓ 全部符号与调用序命中 |
| 15 | parameter_offload.py（DeepSpeedZeRoOffload、pre/post_sub_module_forward_function、fetch/release_sub_module、_run_before/after_backward_function、根模块 pre-hook reset_step）与 partitioned_param_coordinator.py（fetch_sub_module docstring 三步原文、__all_gather_params_、NOT_AVAILABLE/INFLIGHT/AVAILABLE、all_gather_coalesced、InflightParamRegistry、record_module、is_complete_trace、__prefetch_nvme_param_partitions） | 同上两文件下载核对 | ✓ 全部符号命中，docstring 三步逐字 |
| 16 | PyTorch FSDP2：fully_shard 签名（mesh/reshard_after_forward: bool\|int\|None/shard_placement_fn/mp_policy/offload_policy/ignored_params）、FSDPModule、UnshardHandle、@contract(state_cls=FSDPState) 与「fully_shard.state(module)」注释原文、MixedPrecisionPolicy 四字段（param_dtype/reduce_dtype/output_dtype/cast_forward_inputs）、CPUOffloadPolicy、set_requires_gradient_sync/set_requires_all_reduce、FSDPParam、alloc_storage/free_storage、FSDPParamGroup、FSDPCommContext、all_gather_copy_in_* 族 | GitHub API pytorch/pytorch master: torch/distributed/fsdp/_fully_shard/ 五文件 | ✓ 全部命中 |
| 17 | FSDP1：_flat_param.py（FlatParameter:202、FlatParamHandle:481）、fully_sharded_data_parallel.py、_wrap_utils.py、wrap.py | GitHub API 路径 + 类名 | ✓ 200/命中 |
| 18 | TorchTitan：distributed/fsdp.py 的 apply_fsdp_to_decoder、get_fsdp_reshard_after_forward_policy（always/never/default 三档）、CPUOffloadPolicy 接线；config/parallelism.py 存在；FSDP.md 迁移文档：策略映射表（FULL_SHARD↔ZeRO-3、SHARD_GRAD_OP↔ZeRO-2、HYBRID_SHARD↔MiCS、int↔hpZ）、「state dict logic thousands of lines」、no recordStream/无 CPU 同步、PR #165（Llama-7B/8×H100、峰值 −7%、MFU 更高、loss 重合）、BACKWARD_PRE 固定 + BACKWARD_POST 嵌套错误注记、forward_prefetch "under discussion"、meta→fully_shard→to_empty→init_weights 流程、clip_grad_norm_ 直吃 DTensor | GitHub API pytorch/torchtitan 三文件下载核对 | ✓ 全部逐字命中 |
| 19 | ST-MoE（3.1 抽查）：Table 4 消融（baseline 4/6、−1.755；update clipping 3/3、−4.206；router z-loss 3/3、−1.741）、c_z=0.001（扫描定）、总 loss 式 (6)、attention logits z-loss 脚注 | ar5iv 2202.08906 逐字 | ✓ 全部命中 |
| 20 | Liger（3.1 抽查）：摘要「average 20% throughput increase / 60% GPU memory reduction vs HuggingFace」 | arXiv abs 2410.10989 | ✓ 逐字 |
| 21 | TRL（3.2 抽查）：dpo_trainer.py 三键、"first half of the batch" docstring、chosen_logratios 行、logsigmoid 损失路径、序列 logp 的 sum(dim=1)；utils.py selective_log_softmax:535 | GitHub API huggingface/trl 两文件 | ✓ 命中 |
| 22 | Megatron CE 路径（3.1 抽查）：megatron/core/ops/loss/cross_entropy.py 与旧路径 tensor_parallel/cross_entropy.py 均存在；Liger 两路径存在 | GitHub API | ✓ 6/6 200 |
| 23 | 4.1 抽核（sweep）：ring 式 (2)(3)、25 MiB 默认桶 + 首桶缩小（_DEFAULT_FIRST_BUCKET_BYTES、"reduce latency for early-computed gradients" 注释原文）、no_sync 旗标语义、DDP 平均写回 .grad | GitHub API pytorch distributed.py | ✓ 命中（唯式 (2) 的 `\;` → S2-1） |
| 24 | models 1.2 交叉引用：式 (8)（(4+2c)N 分解与 16N 口径）、§9 的 DeepSeek 反推 3.43×10¹⁴ | models/01-foundations/02-compute-accounting.md | ✓ 与 4.2/4.4 引用一致 |
| 25 | 1.4/1.5 回链：1.5 §4↔4.1 §3.4 双向章级指针 ✓；1.5 式 (5)(6) 编号与 4.1/4.2 引用一致 ✓；1.4 的 2–7%/30–40% 与 4.4 引用一致 ✓；1.4→P4 仅 Part 级 → S3-6 | 三章文本交叉读 | 见 S3-6 |
| 26 | GPU 规格：H100 80GB/H200 141GB/B200 192GB、H100 BF16 dense 989 TFLOPS、FP8 dense 1979、H800 同算力低互联、NVLink 900 GB/s、PCIe 5.0 x16 ≈63 GB/s | 厂商公开规格（章节口径声明一致；Llama-3 隐含峰值 430/0.43≈1000 与 989 互证） | ✓ 一致 |
| 27 | 符号约定 sweep：P3/P4 六章 DP 度数统一 $d$；$G$ 未用于 DP 度数；4.3 的 W/G 论文记号附换算注 | spec §4.3 对照 | ✓ |
| 28 | 篇幅（CJK 汉字）：3.1=3808、3.2=3620、4.1=3801、4.2=5827（重章 ≥5500 软目标 ✓）、4.3=3944、4.4=3866 | 本地统计 | ✓ 全部达标 |

**覆盖率声明**：4.2/4.3/4.4 的全部公式（式 (1)(2)(3)、4.3 式 (1)(2)、4.4 式 (1)(2)）逐一独立重推；三章全部数值事实（ZeRO 表 12 格、4.4 静态表 24 格、装得下判定 13 行、通信换算 6 组、MFU 3 家、Offload/Infinity/ZeRO++ 全部 headline 数字）100% 对一手来源或独立重算；三章全部 repo 路径与符号名 100% 经 GitHub API 核对；P3 两章抽查 4 项（ST-MoE 消融、Liger 摘要、TRL 接口、Megatron/Liger 路径）全中；sweep 覆盖符号约定、术语、四组跨章指针、S3 清零。

## 4. 机械检查

- 3 章逐章 lint：02-zero.md / 03-fsdp.md / 04-memory-and-comms-accounting.md → 全部 LINT OK（2026-11-17 本地运行）。
- 全 6 章 lint 汇总：3.1、3.2、4.1、4.2、4.3、4.4 → 6/6 LINT OK。
- 公式白名单补充扫描（lint 不覆盖项）：全 6 章 grep `\,`/`\;`/`\:`/`\!`/`\{`/`\}`/`\|`/`\%`/`\#`/`\operatorname`/`\boxed`/`\color` → 唯一命中为 4.1 式 (2) 的 `\;`（S2-1）。

## 5. 结论

**S1=0，S2=4，S3=6。不放行**（协议：S1/S2 必须修复后复审）。

修复范围：S2-1（4.1，一字级）、S2-4（4.4，一节号级）为 trivial；S2-2（4.2 实现注记改写）、S2-3（4.2 §4.4 与 4.3 §5 的累加-通信口径对齐）需要写作 agent 动笔但改动局部。两项裁定（a）（b）均已闭环：(a) spec §5.5 大纲需回改（度数可引 herd Table 4，⟨verify⟩ 可移除）；(b) 4.2 处理批准、无需改动。S3×6 记入下波清零清单。修复后我复审 4.1/4.2/4.3/4.4 对应四处即可。

---

## 修复复审记录（2026-11-17 复审轮）

复审方式：四章当前全文重读（非 diff），修复触及处的新声明回一手来源复核；全树禁命令重扫；四章 lint 重跑。

### 逐条判定

| Finding | 判定 | 当前文本证据 |
| --- | --- | --- |
| S2-1（4.1 式 (2) `\;`） | **ADDRESSED** | 4.1 式 (2) 现为 `2 \thinspace \frac{d-1}{d} \cdot M \hspace{5mu} \approx \hspace{5mu} 2M`（白名单写法）；全 training 树重扫 `\;`/`\!`/`\{`/`\}`/`\|` 零命中 |
| S2-2（4.2 实现注记「同价」） | **ADDRESSED** | 4.2 §2 末现写：「改用『梯度全量 all-reduce 后各取自己那片』，每个 step 反而多付一次 N 元素——all-reduce 本身是 RS+AG 两段共 2N，而参数 all-gather 仍不可免……合计 3N 对 RS 变体的 2N，贵 1.5 倍。这正是 DeepSpeed stage 1/2 默认走 reduce-scatter 的原因」——算术复核正确，与 stage_1_and_2.py 的 `average_tensor` 双分支结构（`if not self.reduce_scatter:` → `gradient_reduction_w_predivide` → `dist.all_reduce`）一致 |
| S2-3（4.2↔4.3 累加口径冲突） | **ADDRESSED**（重点复审，见下） | 4.2 §4.4 改写为「参数通信恒不摊薄」题 + 两分项；4.3 §5 梯度累加 bullet 同步改写；4.4 §4 修正口径同步。三处口径逐句互洽（见下） |
| S2-4（4.4 中断统计节号） | **ADDRESSED** | 4.4 §5 现引「§3.3.4」——与 herd 论文实际节位（3.3.4 Reliability and Operational Challenges）一致 |
| S3-1（references 回写） | **ADDRESSED**（controller 完成，一项新残留见下） | pytorch-repo 补 4.1/4.3；llama2 补 4.4；llama3 补 4.3/4.4；zero 补 4.4；fsdp 补 4.4；megatron-sp 补 4.4——六处缺口全闭合 |
| S3-2（25 MB→MiB） | **ADDRESSED** | 4.4 §1 现写「默认按 25 MiB 分桶」 |
| S3-3（(4+2c)N 指针） | **ADDRESSED** | 4.4 §1 现写「系数定义见 models 1.2 式 (8)，BF16 矩估计实例见 1.3 §4」 |
| S3-4（CP 归因未标本书分析） | **ADDRESSED** | 4.4 §5 现写「43→41 的归因出自论文（§3.3.2……）；41→38 发生在长上下文阶段（CP 从 1 开到 16、序列从 8K 到 128K）——后一段的 CP 归因是本书分析，论文未逐句解释」（CP 1→16、8K→128K 与 Table 4 逐格一致） |
| S3-5（Offload 脚注归属） | **ADDRESSED** | 4.2 §5.1 现写「（论文 §3.3；本书注记：部分状态卸载可低于此界，此处按整状态卸载口径）」 |
| S3-6（1.4 前向指针 Part 级） | **NOT ADDRESSED**（不阻塞） | 1.4 §5 仍为「P4 / P5 的内容」；修复清单本就未含此项——按 S3 协议累计，转下波清零 |
| 批 1 残留 F-3（4.1 平均语义误挂 `divide_by_initial_world_size`） | **ADDRESSED** | 4.1 §3.2 现写「『除以 world size』由 C++ reducer 在归约时折入（`torch/csrc/distributed/c10d/reducer.cpp`：`div_factor_` 取 process group size、bucket 视图 `.div_(div_factor_)`）」——复核 reducer.cpp 现行 master：`div_factor_ = process_group_->getSize()`（L585）、`bucket_view.div_(div_factor_)`（L436）、`bucket.gradients.div_(div_factor_)`（L502）逐字命中；新增辨伪注「`divide_by_initial_world_size`（默认 True）是 `DDP.join()` 的参数，只在 uneven-inputs join 路径上生效」与 distributed.py L2028 的 `join(divide_by_initial_world_size=True, ...)` 签名及 docstring（"uneven inputs across processes"）一致 |

### S2-3 重点复审：三处口径自洽性

统一口径两侧现为同两句话：①梯度归约 ÷m 可行、代价是一份未分片全量梯度缓冲（2N bytes，FSDP 论文 §3.3.4 without-communication 变体——论文原文 "Without communication, FSDP does not reduce gradients across ranks, and each rank saves the unsharded gradients" 复核一致）；②参数取回（前向+反向 all-gather）逐 micro-batch 发生、任何配置下不摊薄。4.2 §4.4 落到「ZeRO-3 保梯度显存配置 = 3mN/step、就算梯度侧换变体 2mN 纹丝不动」；4.3 §5 落到「梯度归约项 ÷m（代价即这份缓冲）；参数取回……不摊薄」；4.4 §4 修正口径落到「DP 档 4N 整体 ÷m；分片档仅梯度 RS 项 ÷m，参数取回每 micro-batch 共 4N 不摊薄」——三章三处逐句互洽，无残留矛盾。4.2 参考节新增 FSDP §3.3.4 条目 ✓。

### 修复触及处的新问题扫描

- **新残留（S3 级，转下波清零）**：S2-3 的修复使 4.2 新引 FSDP 论文（正文 §4.4 与参考节均在列），但 references.md 的 fsdp 条目被引章节为「4.3, 4.4, 7.4」——缺 4.2。属修复引入的回写遗漏，与 S3-6 一同转入下波 S3 清零。
- 4.4 总结表「累积 ÷m」一格相对改写后的 §4 偏粗（分片档只梯度项 ÷m）——总结格本就作指针用、精确口径在 §4，判定接受，不立 findings。
- 其余触及处（4.1 §3.2、4.2 §2/§4.4/§5.1/参考节、4.3 §5、4.4 §1/§4/§5）逐句复读，未发现新事实或格式问题。

### lint 重跑记录

`bash scripts/lint-chapter.sh` 对 4.1 / 4.2 / 4.3 / 4.4 逐章重跑（2026-11-17）：4/4 LINT OK。

### 复审结论

S2×4 全部 ADDRESSED、批 1 残留 F-3 ADDRESSED、S3 已修 4 项 + controller 回写 1 项；未闭合 2 项均为 S3 级（1.4 指针收紧、fsdp 回写补 4.2），按协议累计不阻塞。**放行 W2 全 6 章。**
