# Training 维度 Spec 对抗审查报告 · Evaluator B（Part 4-6 + §4.5 + §8）

- 日期：2026-11-16
- 审查对象：`docs/superpowers/specs/2026-11-16-training-dimension-design.md` §5 Part 4-6（章节 4.1-6.3）、§4.5 lint 参数化、§8 站点集成
- 立场：fresh context，不信 spec 附录预核实记录；全部一手来源自行核实
- 方法：arXiv API 批量核对 ID↔标题；ar5iv 全文 grep 论文论断；GitHub contents API + raw 内容核对 repo 路径与符号位置；直接读 site/ 与 scripts/ 源码核对现状描述；公式独立重算

## Findings

### S2（必须修复，5 条）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S2-1 | §5 5.1 通信账 | 「每层每向 $2BTd$ bytes」与同句「每层 2 次 all-reduce」自相矛盾 | Megatron 原文确认每层前向 2 次 all-reduce（attention 后 + FFN 后："two all-reduces in the forward path and two in the backward path"），每次消息为 $B \times T \times d$ 元素 = bf16 下 $2BTd$ bytes ⇒ 每向消息量应为 $4BTd$ bytes；按 ring 每 rank 流量口径还要再乘 $2(t-1)/t$ | 大纲钉死口径（消息量 vs 每 rank 流量）并把系数改对：消息量口径每向 $4BTd$ bytes |
| S2-2 | §5 6.1（牵连 5.1 代码对照） | f/g 的 autograd 实现（`_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion`）锚定 `megatron/core/tensor_parallel/layers.py` | GitHub 文件内容实测：两个类定义在 `megatron/core/tensor_parallel/mappings.py`（L205 / L225）；layers.py 里是 ColumnParallelLinear（L1107）、RowParallelLinear（L1531）、VocabParallelEmbedding（L351）——这些归属正确 | 6.1 的 f/g 走读锚点改为 `mappings.py`；layers.py 保留给 Column/RowParallelLinear |
| S2-3 | §5 5.2 | 「causal 负载均衡（zigzag / striped 切分）」以 Ring Attention（2310.01889）为 Sources | 2310.01889 ar5iv 全文实测：causal / zigzag / striped / balanced / imbalance 均 0 命中（仅 "permutation invariance" 环调度 6 处）。该机制正确一手来源是 Striped Attention（arXiv:2405.13119）与 Megatron CP 实现 | 5.2 Sources 补 2405.13119，或以 Megatron-LM CP 代码为锚；不得把 zigzag 归于 Ring Attention 论文 |
| S2-4 | §5 5.1 / 5.5 | DeepSeek-V3 的 TP=1 归因于「MLA 低秩结构 TP 收益分析」 | 2412.19437 全文 "tensor parallelism" 仅 2 处，口径均为「meticulously optimize the memory footprint, making it possible to train DeepSeek-V3 without using costly tensor parallelism」（§3.2 开头 + 摘要）——论文从无 MLA↔TP 收益分析 | 按论文口径改写（显存工程使 TP 不必要），或把 MLA-TP 分析明确标注为本章自行推导而非论文口径 |
| S2-5 | §5 4.4 真实数字表 | 「H100 80GB / 141GB / B200 192GB」 | 141GB 是 H200（HBM3e）；H100 为 80GB（SXM/PCIe）或 94GB（NVL）。装不装得下的判定表直接依赖此数 | 改为 H100 80GB / H200 141GB / B200 192GB |

### S3（下波清零，6 条）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S3-1 | §5 5.3 | zero-bubble 小节只点名 ZB-H1 | 2401.10241：ZB-H1 是「peak memory ≤ 1F1B」的省显存调度（"ensures that the maximum peak memory usage ... doesn't exceed that of 1F1B"）；真正零气泡是 ZB-H2（"it's possible to achieve a zero bubble schedule, which we label as ZB-H2"） | 大纲并列 ZB-H1/ZB-H2 的显存-气泡权衡 |
| S3-2 | §5 4.2 | 内容含 ZeRO++ 但 Sources 未收其出处 | ZeRO++ 正确 arXiv ID = 2306.10209（实测 2306.10207 是无关数学论文，差 2 号） | research pass 收 2306.10209 入 references.md |
| S3-3 | §5 4.3 | FSDP2 锚点标 [A]（「FSDP 论文 + 实现」） | 2304.11277 是 FSDP1（FlatParameter 时代）的论文；FSDP2（per-param DTensor）无论文 | 按 §4.1 口径改为论文 [A] 只覆盖 FSDP1 语义 + FSDP2 实现走「教学参照 + 官方实现」 |
| S3-4 | §5 5.4 | 「Megatron-Core MoE 支持 [A]」 | Megatron 三篇论文（1909.08053 / 2205.05198 / 2104.04473）均不含 MoE/EP；MoE 支持无对应论文 | 降级为「教学参照 + 官方实现」或注明 [A] 仅指 Megatron 本体论文 |
| S3-5 | §5 5.5 | 「OLMo 2（FSDP-only 的选择）」无 ⟨verify⟩ | 2501.00656 ar5iv 全文 FSDP / fully sharded / data parallel 均 0 命中；OLMo-core repo 的 `src/olmo_core/distributed/parallel/` 反而有 data_parallel/tensor_parallel/pipeline_parallel/context_parallel/expert_parallel 全套 | 加 ⟨verify⟩，或改以 OLMo-core 的 OLMo 2 发布配置为一手来源 |
| S3-6 | §8 / §4.5 | links.mjs「只需扩展两处」；STYLE_GUIDE 泛化 | links.mjs 还有第三处 /models/ 硬编码：rewriteTarget 的 bare 章节链接分支（L56 `${base}/models/...`）——纯文字约定下不触发，但维度化漏改会静默错路由；STYLE_GUIDE 头部/适用范围/§10/§11 用法示例已泛化（spec 写作前已改），但 §11 检查项 4 行文（L176）与 §1 术语表行（L12）仍写 `models/references.md` | links.mjs 维度化覆盖三处；§4.5 任务清单补上 STYLE_GUIDE L12/L176 两处残留 |

### 无 S1

Part 4-6 全部 headline 事实声称（ZeRO 16N 与通信 ×1.5、ring all-reduce、f/g 恒等算子、SP 通信恒等、GPipe 气泡、1F1B 峰值激活、ZB 反向拆分、DualPipe、DSv3 PP16/EP64/无 TP、Llama 16K 集群与中断统计、全部 arXiv ID、全部 repo 路径）经一手来源核实无误。

## Verification Log

| # | 抽查项 | 来源 | 结论 |
|---|---|---|---|
| 1 | 17 个 arXiv ID ↔ 标题（1910.02054 / 2101.06840 / 2104.07857 / 2006.15704 / 2304.11277 / 1909.08053 / 2205.05198 / 2310.01889 / 2309.14509 / 1811.06965 / 2104.04473 / 2401.10241 / 2006.16668 / 2412.19437 / 2407.21783 / 2501.00656 / 2507.20534） | arXiv API id_list 批量查询 | 17/17 全对（含 OLMo 2 = "2 OLMo 2 Furious"、K2 = "Kimi K2: Open Agentic Intelligence"） |
| 2 | ZeRO 16N 分解（2N params + 2N grads + 4N master + 8N m,v） | 1910.02054 ar5iv 全文 | ✓ 原文："2Ψ and 2Ψ bytes... fp32 copy of the parameters, momentum and variance, with memory requirements of 4Ψ, 4Ψ, and 4Ψ"（Ψ=参数量，即 16Ψ） |
| 3 | ZeRO-1/2 与 DP 同通信量、ZeRO-3 ×1.5 | 1910.02054 §7 | ✓ "same communication volume as DP"（Pos、Pos+g）；"The total volume is therefore 3Ψ which is 1.5x compared to the baseline"（Pos+g+p，baseline=2Ψ） |
| 4 | ring all-reduce 每 rank 2(G−1)/G·M | 独立推导 + ZeRO 论文佐证（"all-reduce 通信量 = 2×message_size"、all-gather = 1×） | ✓ scatter-reduce (G−1)/G·M + all-gather (G−1)/G·M |
| 5 | Megatron f/g 恒等算子 + 每层 2 次 all-reduce（前向） | 1909.08053 ar5iv（Figure 3 说明 + Code 1） | ✓ "f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity"；"two all-reduces in the forward path and two in the backward path" |
| 6 | 每层每向 2BTd bytes | 独立重算 | ✗ 与「2 次 all-reduce」矛盾 → S2-1 |
| 7 | SP 通信恒等（all-reduce ↔ all-gather+reduce-scatter） | 2205.05198 ar5iv | ✓ "a ring all-reduce is composed of two steps: a reduce-scatter followed by an all-gather. As a result, the communication bandwidth used for tensor parallelism and tensor together with sequence parallelism are the same" |
| 8 | Ring Attention online softmax 跨块正确性 | 2310.01889 ar5iv | ✓ 机制在（blockwise numerator/denominator/max_score rescale + KV 环转 ppermute）；论文用 "permutation invariance" 表述调度自由 |
| 9 | zigzag / striped 负载均衡归属 | 2310.01889 全文 grep | ✗ 0 命中 → S2-3 |
| 10 | GPipe 气泡 (p−1)/(m+p−1) | 1811.06965 ar5iv | ✓ "bubble time is O((K−1)/(M+K−1))"，M≥4K 可忽略 |
| 11 | 1F1B 峰值激活 ≤p 份/stage | 2104.04473 ar5iv | ✓ "limits the number of in-flight microbatches ... to the depth of the pipeline, instead of the number of microbatches in a batch" |
| 12 | ZB 反向拆 input-grad / weight-grad | 2401.10241 ar5iv | ✓ "split the backward computation into two parts, one that computes gradient for the input and another that computes for the parameters"；但零气泡属 ZB-H2 → S3-1 |
| 13 | DSv3 DualPipe / PP16 / EP64×8节点 / ZeRO-1 / 无 TP | 2412.19437 ar5iv §3.2 | ✓ "applies 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism"；DualPipe "bidirectional pipeline scheduling... feeds micro-batches from both ends"；TP=1 理由口径偏差 → S2-4 |
| 14 | DSv3 node-limited routing / DeepEP | 2412.19437 | ✓ M=4 节点限制路由逐字在；DeepEP 论文 0 命中（2025-02 才开源，repo deepseek-ai/DeepEP 实测 200）——章节引用应指 repo 非论文 |
| 15 | Llama-3 TP8×CP16×PP16×DP、16384 GPU | 2407.21783 ar5iv §3.3.2 | 度数 ✗ 不在正文（TP8/CP16/PP16 grep 全 0，仅定性 4D 描述 + Figure 5 图内）；"up to 16K H100"（24K 集群用至多 16K）✓。spec 的 ⟨verify⟩ 标记正确且必须保留；备选来源：torchtitan 405B 配置或 [C] 级二手 |
| 16 | Llama-3 54 天中断统计（7.4 ⟨verify⟩ 项） | 2407.21783 §3.3.4 | ✓ 54-day snapshot 466 次中断（47 planned + 419 unexpected，约 78% 硬件、GPU 占 58.7%、有效训练时间 >90%） |
| 17 | Kimi K2 训练侧并行配置（5.5 ⟨verify⟩） | 2507.20534 ar5iv §2.4.2 | ✓ 已可解决：16-way PP（virtual stages）× 16-way EP × ZeRO-1 DP，节点数需为 32 的倍数 |
| 18 | Megatron repo 路径 ×4 + f/g 符号位置 | GitHub contents API + raw 文件 grep | 路径 4/4 ✓（layers.py / mappings.py / parallel_state.py / pipeline_parallel/schedules.py / optimizer/ 全 200）；f/g 在 mappings.py 非 layers.py → S2-2 |
| 19 | deepspeedai/DeepSpeed stage3.py + 组织更名 | GitHub API | ✓ 200；microsoft/DeepSpeed 301 重定向确认 |
| 20 | torchtitan/distributed、torch _fully_shard、distributed.py | GitHub API | ✓ 全 200 |
| 21 | OLMo-core 存在性与并行支持面 | GitHub API | ✓ repo 存在；含 TP/PP/CP/EP 全套 → S3-5 |
| 22 | Ulysses all-to-all 换维机制 | 2309.14509 ar5iv abstract | ✓ "employs all-to-all ... each GPU receives the full sequence but only for a non-overlapping subset of the attention heads" |
| 23 | DDP bucketing / overlap / no_sync | 2006.15704 ar5iv | ✓ 三者均在（no_sync 含代码示例） |
| 24 | H100 141GB | 独立重算（NVIDIA 规格常识） | ✗ 141GB=H200 → S2-5 |
| 25 | §4.5 lint 现状（检查项 4 硬编码 models/references.md） | scripts/lint-chapter.sh L13 实测 | ✓ 属实；首段参数化与脚本结构兼容（$f 相对路径可取首段）；其余检查项维度无关 |
| 26 | §8 站点现状：PARTS 表 / 51/10/≥100 断言 / 路由 / live:false / QA 命令链 | build-content.mjs L37-48、L192-194；pages/ 结构；index.astro L25；package.json qa | ✓ 全部属实；QA 链与 package.json `qa` script 逐字一致 |
| 27 | links.mjs 现状描述 | links.mjs 源码 | ✓ 大体属实（../references.md→/models/references/、../../<dim>/README.md→GitHub）；漏 L56 第三处 → S3-6 |
| 28 | STYLE_GUIDE 泛化状态 | STYLE_GUIDE.md 头部/L12/L150/L168/L176 | 头部、适用范围、§10、§11 用法示例已泛化；L12 术语表行与 L176 检查项 4 行文残留 models-only → S3-6 |
| 29 | figures/STYLE.md、fig-check.mjs、qa-shots.mjs、check-links.mjs 存在性 | site/scripts 目录 ls | ✓ 全在 |
| 30 | FSDP2 论文归属 | 2304.11277 标题与内容 | FSDP1 时代论文 → S3-3 |

**覆盖率**：Part 4-6 共 12 章的全部关键事实/公式/归因声称、spec 列出的全部 arXiv ID（17 个在审范围内全核）、全部 GitHub 锚点路径（9 个 + f/g 符号级）、§4.5 lint 方案、§8 站点方案全部现状断言。方案可行性（DIMENSIONS 抽象、分维度 references.json、/training/ 路由、pagefind 收录）经源码核对无明显坑，唯 links.mjs 第三处硬编码需纳入（S3-6）。

## 结论

- S1: 0 / S2: 5 / S3: 6
- 结论：**修复 5 条 S2 后放行**（无 S1，方案层与 lint/站点描述全部属实；S2 集中在大纲级公式系数、代码文件锚点、两处文献归属与一处硬件规格）。
