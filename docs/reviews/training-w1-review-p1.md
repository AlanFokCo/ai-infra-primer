# training W1 对抗审查报告 — 批 1：Part 1 五章（1.1–1.5）

日期：2026-11-17。审查人：独立 fresh-context evaluator（与写作 agent 无共享上下文）。立场：假设内容有错，直到一手来源证明无误。

审查对象：`training/01-autograd-and-backprop/01–05`（计算图与 autograd / 算子反向推导【重】/ 混合精度与 FP8 / 激活内存与 gradient checkpointing / gradient accumulation）。

标准：STYLE_GUIDE.md 全条款、`docs/reviews/2026-11-15-style-audit/rubric.md`、training spec §5 Part 1 大纲（覆盖度对照）、`training/references.md`（引用基准）。

**结论：不放行（1 条 S1）。** S1 ×1、S2 ×4、S3 ×9。S1 为 1.4 §5 决策表算例数字错误（结论翻转），必须修复后复审该节；S2 同步修复；S3 记入下波清零。

## 0. 机械检查（lint）

`bash scripts/lint-chapter.sh` 对 5 章逐一执行（仓库根目录，2026-11-17）：

| 章 | 结果 | 退出码 |
| --- | --- | --- |
| 01-computation-graph-and-autograd.md | LINT OK | 0 |
| 02-operator-backward.md | LINT OK | 0 |
| 03-mixed-precision.md | LINT OK | 0 |
| 04-gradient-checkpointing.md | LINT OK | 0 |
| 05-gradient-accumulation.md | LINT OK | 0 |

篇幅（CJK 汉字数，不含标点/公式/代码，按 2026-11-12 controller 口径实测）：1.1 = 3254（标准章下限 3500，**不达标**）；1.2 = 4866（推导重章软目标 5500，低于软目标）；1.3 = 3948；1.4 = 3574；1.5 = 3525。

公式渲染硬规则扫描（开口侧 flanking / 闭口侧 flanking / 反转义陷阱 `\{` `\|` `\,` `\;` `\!` `\%` `\#` / 公式内 `<`+字母 / 行内 `*` 冲突）：五章全部干净，零命中（脚本化扫描 + 人工复核命中行）。

## 1. 强制独立重推导（逐步中间过程）

### 1.1 matmul 反向（1.2 §2 式 (1)(2)）——通过

从零重推。$Y = AB$，$Y_{ij} = \sum_{s=1}^{k} A_{is} B_{sj}$，$G = \partial L / \partial Y \in \mathbb{R}^{m \times n}$。

$A_{is}$ 只出现在 $Y$ 的第 $i$ 行（$Y_{i1}, \ldots, Y_{in}$），故

$$\frac{\partial L}{\partial A_{is}} = \sum_{j=1}^{n} \frac{\partial L}{\partial Y_{ij}} \frac{\partial Y_{ij}}{\partial A_{is}} = \sum_{j=1}^{n} G_{ij} B_{sj} = (G B^{\top})_{is}$$

第二步用 $\partial Y_{ij} / \partial A_{is} = B_{sj}$（$Y_{ij}$ 对 $A_{is}$ 线性）。对 $B_{sj}$ 对称：它出现在第 $j$ 列全部 $m$ 个元素，$\partial L / \partial B_{sj} = \sum_i A_{is} G_{ij} = (A^{\top} G)_{sj}$。与章节式 (1)(2) 逐式一致。

FLOPs 复核：前向 $2mnk$；$GB^{\top}$ 为 $[m,n]\times[n,k]$ = $2mnk$；$A^{\top}G$ 为 $[k,m]\times[m,n]$ = $2kmn$；合计 $4mnk$ = 恰 2× 前向。章节实例（down 投影 $m$=4096、$k$=14336、$n$=4096）：前向 $2 \times 4096 \times 14336 \times 4096 = 4.81 \times 10^{11}$ ✓「约 $4.8 \times 10^{11}$」、反向 $9.62 \times 10^{11}$ ✓「约 $9.6 \times 10^{11}$」。形状断步检查 $[m,n]\times[n,k]=[m,k]$、$[k,m]\times[m,n]=[k,n]$ ✓。

### 1.2 softmax-CE 融合梯度（1.2 §4 式 (4)(5)）——通过

$L_t = -z_{t,y_t} + \log \sum_v \exp(z_{t,v})$。对 $z_{t,j}$ 求导：第一项贡献 $-\delta_{j y_t}$；第二项链式：$\frac{1}{\sum_v e^{z_v}} \cdot e^{z_{t,j}} = p_{t,j}$。合并得 $p_{t,j} - \delta_{j y_t}$ ✓ 与式 (5) 一致。

分步路径的代数验证（章节内嵌推导）：$g_i = \partial L / \partial p_i = -\delta_{iy} / p_y$，则 $\langle p, g \rangle = p_y \cdot (-1/p_y) = -1$，代入 softmax vjp $p_j(g_j - \langle p, g \rangle) = p_j(-\delta_{jy}/p_y + 1) = p_j - \delta_{jy}$ ✓ 恒等成立。

有界性：$|p_j - \delta_{jy}| \le p_j + \delta_{jy} \le 2$ ✓「$\lvert \partial L / \partial z_j \rvert \le 2$」✓。

### 1.3 loss scaling（1.3 §3 式 (3)(4)(5)）——公式通过，单位表述有 S2

- 下溢界：fp16 = 1/5/10、bias 15 → 最小正规数 $2^{1-15} = 2^{-14}$ ✓；最小次正规数 $2^{-14} \times 2^{-10} = 2^{-24}$ ✓。与论文（arXiv:1710.03740 §3.1：「any value whose magnitude is smaller than $2^{-24}$ becomes zero in FP16」）一致 ✓。
- 恒等式：$\nabla_w(S\mathcal{L}) = S \nabla_w \mathcal{L}$（梯度线性性）；$\theta - \eta \cdot (S \nabla \mathcal{L})/S = \theta - \eta \nabla \mathcal{L}$ ✓ 实数算术下恒等。vjp 对 adjoint 线性故 $S$ 逐节点穿透 ✓。
- 窗口：不压死要求 $S g_{\min} \ge 2^{-24}$，不溢出要求 $S g_{\max} < 65504$（论文原文 65,504 ✓）→ 式 (4) ✓。窗口存在条件 $g_{\max}/g_{\min} < 65504 \times 2^{24} = 1.099 \times 10^{12}$ ✓ 数值正确。**但** $1.1 \times 10^{12} = 2^{40.0}$：「40 个」是二倍程（binade）口径，十进制数量级是 12——见 finding S2-2。
- dynamic scaling：GradScaler 源码签名核实 init_scale=$2^{16}$、growth_factor=2.0、backoff_factor=0.5、growth_interval=2000 ✓（`torch/amp/grad_scaler.py` 128–130 行），式 (5) 语义一致 ✓。

### 1.4 √L 分段最优（1.4 §2 式 (2)(3)）——通过

$M(k) = ka + (L/k)a$（$k$ 个 checkpoint 全程驻留 + 单段重算峰值 $(L/k)a$）。一阶条件 $M'(k) = a(1 - L/k^{2}) = 0 \Rightarrow k^{\ast} = \sqrt{L}$；二阶导 $M'' = 2aL/k^{3} > 0$ 确认极小。$M(k^{\ast}) = 2a\sqrt{L}$ ✓。均值-几何均值同证：$ka + La/k \ge 2a\sqrt{L}$ 等号当且仅当 $k = \sqrt{L}$（与练习 1 口径一致 ✓）。

代价：每段反向时重跑一次前向 = 恰一次额外前向；fwd:bwd = 1:2 → step 总开销 3 → 4 份 = +33% ✓；$6ND \to 8ND$ 与 models 1.2 练习 4 口径一致 ✓。

平坦性注记复核：$M(2\sqrt{L}) = a(2\sqrt{L} + \sqrt{L}/2) = 2.5 a\sqrt{L} = 1.25 \times M(k^{\ast})$ ✓。

与 Chen et al. 原文 §4.3 对照：「cost-total = max cost-of-segment$(i)$ + $O(k)$ = $O(n/k) + O(k)$，$k = \sqrt{n}$ 得 $O(2\sqrt{n})$」、「only requires an additional forward pass」、「Since the backward operation is nearly twice as time consuming as the forward one, it only slows down the computation by a small amount」——逐项一致 ✓。

## 2. Findings

### S1（事实性致命，必须修复后复审）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S1-1 | 1.4 §5 决策表算例 | 「TP 8 路加 selective 为 $194 \times 13 / 114 \approx 22$ GiB，还差一点；TP 8 路叠加 SP 与 selective 为 $194 \times 34 / 912 \approx 7.2$ GiB」 | 本例自建基线为 $194$ GiB（$h$=4096、$a$=32、$s$=4096 → $5as/h = 160$，无并行因子 $34+160=194$，同段上文自算）。按论文 Table 2（TP+selective = $10 + 24/t$；TP+SP+selective = $34/t$），$t=8$ 时正确换算为 $194 \times 13/194 = 13$ GiB 与 $194 \times 34/(8 \times 194) = 4.25$ GiB。章节两数误用了 GPT-3 几何的基线 114（$34+80$，同节上一段）与 $912 = 8 \times 114$。同段 full 行「$194 \times 2/194 = 2$ GiB」用的是正确基线，可证 114/912 系笔误。后果：正确数字下 TP+selective（13 GiB）已满足 20 GiB 预算，「还差一点、须再叠 SP」的阶梯叙事结论翻转。 | 两数改为 13 GiB 与 4.25 GiB；相应改写阶梯叙事（TP+selective 已满足预算；再叠 SP 到 4.25 GiB 属进一步压缩/换时间）。 |

### S2（实质问题，必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 1.2 §9 | 「与 models 1.2 §7 的 scores+probs 表同口径」 | scores+probs 内存表（$T$=4096 → scores 1 GiB、scores+probs 2 GiB、×32 层 64 GiB）实际位于 **models 2.3 §7**（`models/02-transformer-anatomy/03-attention-mechanism.md` 228 行表）；models 1.2 §7 是「每层 FLOPs 分解与 shape 走读」表，无 scores+probs 账。64 GiB 数值本身与 models 2.3 §7 一致、正确。 | 指针改为「models 2.3 §7」。 |
| S2-2 | 1.3 §3 式 (4) 符号注 + §4 | 「梯度分布的动态范围只要不超过 40 个数量级，窗口就非空」；「离下溢地板有上百个数量级」 | $65504 \times 2^{24} \approx 1.1 \times 10^{12} = 2^{40.0}$：40 是二倍程（binade）数，十进制数量级为 12。本章 §1 已自定义「二倍程（binade）」为二进制口径术语，这两处用「数量级」属术语漂移 + 数值单位歧义（按十进制读法夸大 3.3 倍）。§4 处 $2^{-20}$ 到 $2^{-126}$ 相距 106 个二倍程 = 32 个十进制数量级，同病。 | 两处改为「40 个二倍程（约 12 个十进制数量级）」、「上百个二倍程」或统一换算。 |
| S2-3 | 1.2 §4 | 「bf16 下 $p_y \lt 10^{-38}$ 即下溢为 0、梯度炸成 $-\infty$」 | bf16 最小正规数 $2^{-126} \approx 1.2 \times 10^{-38}$、最小次正规数 $2^{-133} \approx 9.2 \times 10^{-41}$（本批 1.3 §1 表同口径）：$p_y = 10^{-38}$ 处于 bf16 次正规区而非零；$-1/p_y$ 溢出 bf16 最大值（$3.39 \times 10^{38}$）的阈值是 $p_y \approx 2.9 \times 10^{-39}$；下溢为 0 在 $9.2 \times 10^{-41}$ 以下才发生。即使按 fp32（autocast 下 CE 实际计算精度）口径，$10^{-38}$ 也是次正规而非零。机制结论（分步路径有 $1/p_y$ 奇点、融合式恒有界）不受影响。 | 改为正确口径，如「$p_y$ 低于约 $3 \times 10^{-39}$ 时 $-1/p_y$ 在 bf16 中溢出为 $-\infty$；次正规区尾数逐位丢失」；或改举 fp32 数值。 |
| S2-4 | 1.1 全章 | 篇幅 3254 CJK 字 | STYLE_GUIDE §8：标准章 ≥ 3500 字（CJK 汉字、不含标点/公式/代码）。实测（脚本剥离代码块、行内 code、`$$` 与行内公式后计 CJK 字符）3254，差约 7%。内容完整度无问题，属约定冲突。 | 补 §6 或 §7 的实质内容（如 retain_graph 内存实例、Compiled Autograd 一句现状）约 300 字。 |

### S3（minor，记录入下波清零）

| # | 位置 | 断言 | 证据 / 建议 |
| --- | --- | --- | --- |
| S3-1 | 1.3 §4 | 「bf16 乘积同样精确（7 + 7 = 14 位有效位）」 | 与 §2 fp16「11 + 11 = 22 位」的计数口径不一致（后者含隐含位）。正确口径：bf16 有效位 8 位（1 隐含 + 7 尾数），乘积至多 16 位 ≤ fp32 的 24 位。结论不变，建议统一为含隐含位口径（8 + 8 = 16）。 |
| S3-2 | 1.2 §10 | 「块长按显存预算反解（实现注释的口径：…例 $M$ = 16384、$V$ = 32000、$d$ = 4096 ⇒ 块长 2048）」 | 所引注释存在于 Liger-Kernel 旧版（v0.5.2 该文件 34–37 行：「BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048」✓ 数值核实一致）；当前 main 已改写为 `_CHUNK_MEM_CONST` 预算注释（「approximately BT x H elements regardless of vocabulary size」——同义不同文）。建议标注版本或改为「早期版本注释」。 |
| S3-3 | 1.4 §4 + 参考条 | `_NoopSaveInputs`（一个只存输入的自定义 Function） | 该类名在 PyTorch ≤ v2.10.0 存在（v2.10.0 该文件 grep 命中 2 处）；最新 release v2.15.0 已重构为 `_CheckpointFrame.save_inputs` / `_make_saved_tensor`（main 分支同名消失，机制注释「we use a noop autograd Function to save inputs」仍在 839 行）。机制描述本身正确，仅内部类名漂移。建议加「（v2.11 起重构为 `_CheckpointFrame.save_inputs`）」一句。 |
| S3-4 | training/references.md | pytorch-repo 条目「被引章节」字段 = 「1.1, 1.2, 1.4, 2.1, 2.2, 2.4」 | 实际 1.3（grad_scaler.py / autocast_mode.py）与 1.5（distributed.py）均引用该条目（两章参考节均在列）。按 references.md 写入规则补记「1.3, 1.5」。 |
| S3-5 | 1.2 §9 | 「IO 复杂度（论文 Theorem 2）：标准 $\Theta(Nd + N^{2})$ vs FA $\Theta(N^{2} d^{2} / M_{\mathrm{sram}})$」 | 所引数值正确，但 Theorem 2 是前向陈述；反向通行次数的同型界是论文 Theorem 5（附录 C 证明）。本节语境恰在讲反向，建议写「Theorem 2（前向）与 Theorem 5（反向）同型界」。 |
| S3-6 | 1.4 §3 | 「对照层前向 matmul 总量 $24 s h^{2}$（QKV 投影 $6sh^{2}$、输出投影 $2sh^{2}$、MLP $16sh^{2}$）」上文称「分母为层前向的全部 matmul」 | $24sh^{2}$ 不含 attention 本体 $QK^{\top}$/$PV$ 的 $4s^{2}h$：「全部 matmul」与括号枚举不一致。这正是本章自算 2.8%/1.7% 与论文 2.7%/1.6% 差 0.1pp 的来源（论文 Appendix A 的 model FLOPs 分母含本体项与 logits 项 $v/(12hL)$，精确比式 (8)/(7) 代入 GPT-3 得 2.69%、MT-NLG（$v$=51200 口径）得 1.64%）。建议「全部」删去，并加半句说明与论文数的 0.1pp 差来自分母口径。 |
| S3-7 | 1.4 §3 | 「530B 与 1T 规模上 selective 一侧开销降到 2%，full 仍为 36%」 | 论文 Figure 8 的 2% 是 SP + selective 合并口径（图注：「Present work includes both sequence parallelism and selective activation recomputation」），与同句 22B 行「selective 叠加 sequence parallelism 后 4%」的口径衔接但措辞省略了 SP。建议改为「selective + SP 一侧」。 |
| S3-8 | 1.2 §4 | 「生产实现（`torch.nn.functional.cross_entropy`…）…反向直接以 $p - e_y$ 落 kernel」 | eager 路径实为 `log_softmax` + `nll_loss` 复合（两个 ATen 反向的组合代数效果 = $p - e_y$），不是单一融合 kernel；「$1/p_y$ 中间量不存在」的结论正确（log_softmax 反向从保存的 log-prob 取 exp，无 $1/p_y$）。建议「落 kernel」改「复合反向的代数效果即 $p - e_y$」。 |
| S3-9 | 1.1 §2 | 图注「（adjoint 流，式 (4) 的逐节点 vjp）」 | vjp 的正式定义在 §3（向量-雅可比积命名处），§2 图注先用属前向引用。轻微；可在图注加「（§3 定义）」四字。 |

## 3. Verification log（抽查项 × 来源 × 结论）

### 3.1 arXiv ID ↔ 标题（arXiv API `export.arxiv.org/api/query?id_list=...`，2026-11-17 实测）

| arXiv ID | API 返回标题 | 章节口径 | 结论 |
| --- | --- | --- | --- |
| 1502.05767 | Automatic differentiation in machine learning: a survey | Baydin AD 综述 | ✓ |
| 2006.15704 | PyTorch Distributed: Experiences on Accelerating Data Parallel Training | Li et al. DDP | ✓ |
| 2205.14135 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | ✓ |
| 2410.10989 | Liger Kernel: Efficient Triton Kernels for LLM Training | Hsu et al. | ✓ |
| 1604.06174 | Training Deep Nets with Sublinear Memory Cost | Chen et al. | ✓ |
| 2205.05198 | Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al. | ✓ |
| 1710.03740 | Mixed Precision Training | Micikevicius et al. 2017 | ✓ |
| 2209.05433 | FP8 Formats for Deep Learning | Micikevicius et al. 2022 | ✓ |
| 2412.19437 | DeepSeek-V3 Technical Report | DeepSeek-AI | ✓ |
| 1812.06162 | An Empirical Model of Large-Batch Training | McCandlish et al. | ✓ |

10/10 通过（references.md 的 Part 1–2 相关条目全覆盖）。

### 3.2 GitHub repo 路径（contents API，2026-11-17 实测，全部 HTTP 200）

pytorch/pytorch：`torch/autograd/function.py`、`torch/csrc/autograd/engine.cpp`、`torch/csrc/autograd/functions/accumulate_grad.cpp`、`tools/autograd/derivatives.yaml`、`torch/nn/functional.py`、`torch/nn/modules/sparse.py`、`torch/amp/grad_scaler.py`、`torch/amp/autocast_mode.py`、`torch/utils/checkpoint.py`、`torch/nn/parallel/distributed.py`、`torch/autograd/forward_ad.py`（11/11 ✓）。linkedin/Liger-Kernel：`src/liger_kernel/ops/fused_linear_cross_entropy.py` ✓。

### 3.3 论文论断逐项核对（ar5iv 全文转文本 + 关键词定位，引句均为论文原文）

**FlashAttention（2205.14135，[A]）** — 对照 1.2 §9：
- Theorem 2：「Standard attention (Algorithm 0) requires $\Theta(Nd+N^2)$ HBM accesses, while FlashAttention (Algorithm 1) requires $\Theta(N^2 d^2 M^{-1})$…typical values of $d$ (64-128) and $M$ (around 100KB)」✓（反向同型界为 Theorem 5，见 S3-5）
- 下界：「no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes」✓
- 附录 B：存 $O$ 与 $(m, \ell)$ 重算 $S$、$P$ ✓；「$D_i = do_i^{\top} o_i$」= 章节式 (15) $\langle O_i, \mathrm{d}O_i \rangle$ ✓；dropout：「save the pseudo-random number generator states…only use $O(N)$ extra memory」✓；反向五条闭式与 Algorithm 4 step 20（$\mathrm{d}S_{ij} = P_{ij} \circ (\mathrm{dP}_{ij} - D_i)$）✓
- 「even though FlashAttention has higher FLOP count…(due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime」✓

**Korthikanti et al.（2205.05198，[A]）** — 对照 1.4 §1/§3/§5：
- 逐项账：QKV 共享输入 $2sbh$、$QK^{\top}$ 存 $Q,K$ 共 $4sbh$、softmax 输出 $2as^{2}b$、softmax dropout mask $as^{2}b$、$PV$ 存 dropout 输出+$V$ = $2as^{2}b + 2sbh$、attention 投影 $2sbh$、attention dropout $sbh$、MLP 两线性 $2sbh + 8sbh$、GeLU 输入 $8sbh$、MLP dropout $sbh$、两 LayerNorm 各 $2sbh$——与论文 §4.1 逐字一致 ✓；合计 attention $11sbh + 5as^{2}b$、MLP $19sbh$、LN $4sbh$ ✓；式 (1) $sbh(34 + 5as/h)$ ✓
- Table 2 六行（无并行 / TP $10+24/t+5as/(ht)$ / TP+SP $34/t+5as/(ht)$ / TP+selective $10+24/t$ / TP+SP+selective $34/t$ / full $2sbh$）✓ 逐行一致
- GPT-3（$a$=96、$s$=2048、$h$=12288 → $5as/h$=80）省 70%、MT-NLG（→64）省 65%，FLOPs 开销 2.7% / 1.6% ✓（章节自算 2.8%/1.7%，0.1pp 差源于分母口径，见 S3-6）
- Table 4：22B 基线 7.7 + 11.9 ms、full 39%、selective 7%、selective+SP 4% ✓；「the overhead of 39%…(as opposed to the expected 33%) is due to an optimization in the backward pass where we overlap all-reduce communication with the linear weight's gradient computation」✓ 与章节 §6.2 注记一致
- 530B/1T：「the overhead is just 2%, compared to 36% overhead for full recompute」✓（口径为 SP+selective 合并，见 S3-7）
- Table 5：「between 29.0% and 32.1% improvement in the throughput over performing full recomputation」✓
- Eq 5 修正项「for a model with 22B parameters, these extra terms account for less than 0.01%」✓
- 「when training MT-NLG there are only three layers per device」✓

**Micikevicius 2017（1710.03740，[B]）** — 对照 1.3 §1–3：
- 「half-precision math throughput in recent GPUs is 2× to 8× higher」✓（§1）
- 「approximately 5% of weight gradient values have exponents smaller than −24」（Mandarin，Figure 2(b)）✓；「smaller than $2^{-24}$ becomes zero in FP16」✓
- 「at least 2048 times larger…right-shifted by 11 or more positions」✓；「updating FP16 weights results in 80% relative accuracy loss」✓；「roughly halved」内存 ✓
- §3.2：Multibox SSD 直方图、「much of the FP16 representable range was left unused」✓；「scaling factors ranging from 8 to 32K (many networks did not require a scaling factor)」✓；「below 65,504」上限规则 ✓；「unscaling right after the backward pass but before gradient clipping…no hyper-parameters…have to be adjusted」✓；展望节动态方案「dynamically increased or decreased by inspecting the weight gradients for overflow, skipping weight updates」✓

**FP8 Formats（2209.05433，[B]）** — 对照 1.3 §1/§5：
- Table 1：E4M3 max normal $1.75 \times 2^{8} = 448$、E5M2 $1.75 \times 2^{15} = 57{,}344$、min normal $2^{-6}$ / $2^{-14}$、min subnormal $2^{-9}$ / $2^{-16}$ ✓ 全部一致
- 「Infinities are not represented…only one mantissa bit-pattern for NaNs…from 17 to 18 binades…would be 240」✓
- 「E5M2…follows the IEEE 754 conventions…can be viewed as IEEE half precision with fewer mantissa bits」✓
- 「E4M3 for weight and activation tensors, and E5M2 for gradient tensors…some networks can train with just the E4M3 or the E5M2 type」✓
- 「not a good choice for FP8…resulting in too many skipped updates」✓；per-tensor 饱和 ✓；「unscaling is applied once per dot-product, thus amortized」✓；「within run-to-run variation」✓

**DeepSeek-V3（2412.19437，[A]）** — 对照 1.3 §6 与 1.4 练习 3：
- 全 E4M3：「we adopt the E4M3 format on all tensors for higher precision」✓；hybrid 前作「E4M3…in Fprop and E5M2…in Dgrad and Wgrad」✓
- 粒度：「activations…1x128 tile basis (per token per 128 channels)…weights…128x128 block basis (per 128 input channels per 128 output channels)」✓；附录 B.2 块化激活梯度致发散 ✓
- 累加：「H800…limited to retaining around 14 bits」✓；「$K = 4096$…maximum relative error of nearly 2%」✓；「$N_C = 128$ elements, equivalent to 4 WGMMAs…minimal accumulation interval that can significantly improve precision without introducing substantial overhead」✓；「two WGMMA to persist concurrently…one warpgroup performs the promotion…other…MMA」✓
- 三 GEMM FP8：「Fprop…Dgrad…Wgrad…all executed in FP8…outputs in BF16 or FP32」「theoretically doubles the computational speed compared with the original BF16 method」✓；「FP8 Wgrad GEMM allows activations to be stored in FP8 for use in the backward pass」✓
- 高精度保留清单：「embedding module, the output head, MoE gating modules, normalization operators, and attention operators」✓
- E5M6：「Inputs of the Linear after the attention operator…customized E5M6…converted from an 1x128 quantization tile to an 128x1 tile in the backward pass…all the scaling factors are round scaled, i.e., integral power of 2」✓
- MoE SwiGLU：「cache the inputs…recompute its output in the backward pass…stored in FP8」✓；通信：「quantize the activation before MoE up-projections into FP8…activation gradient before MoE down-projections…combine…retain in BF16」✓
- optimizer 态：「BF16…first and second moments in the AdamW optimizer, without incurring observable performance degradation…master weights…and gradients (used for batch size accumulation)…retained in FP32 to ensure numerical stability」✓
- 验证：「two model scales similar to DeepSeek-V2-Lite and DeepSeek-V2, training for approximately 1 trillion tokens…relative loss error…consistently below 0.25%, well within the acceptable range of training randomness」✓；附录 B.1 细节 16B@1.33T、230B@0.9T ✓（「各约 1T」为合理舍入）
- 「train DeepSeek-V3 on 14.8T high-quality and diverse tokens」✓；「number of Transformer layers to 61」✓

**Chen et al. 2016（1604.06174，[B]）** — 对照 1.2 §10、1.4 §2：
- §4.3：cost $= O(n/k) + O(k)$、$k = \sqrt{n}$ → $O(2\sqrt{n})$、「only requires an additional forward pass」✓；Alg. 3 贪心预算分配 ✓
- 「reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G」✓（原文为「memory cost」总口径，章节写作「激活内存」略窄，量值一致）
- 「memory consumption can be reduced to $O(\log n)$ with as little as $O(n \log n)$ extra cost」✓

**Li et al. 2020（2006.15704，[A]）** — 对照 1.1 §6.1、1.5 §4：
- §3 autograd hook：「DDP loops over all parameters…finds the gradient accumulator…installs the same post-hook…fire post hooks when the corresponding gradient is ready…each bucket keeps a count of pending gradients…decrements the count」✓
- §3.2.4：「In no_sync mode, all DDP hooks are disabled, and the first backward pass out of the context will synchronize the accumulated gradients altogether」「the flag is consumed in the forward function of DDP」✓（前向内锁定语义 ✓）

**Baydin et al. 2018（1502.05767，[B]）** — 对照 1.1 §3/§4：
- 「the full Jacobian can be computed in $n$ evaluations」（前向模式）✓；「matrix-free way of computing Jacobian–vector products」✓；「AD guarantees that the amount of arithmetic goes up by no more than a small constant factor」✓；「accurate evaluation of derivatives at machine precision」✓（对应章节「精度精确到浮点、成本是前向的常数倍」）

**Liger（2410.10989 + repo，[A]）** — 对照 1.2 §10：
- 「on average 20% increase in training throughput and a 60% reduction in GPU memory…compared with HuggingFace implementations」✓（摘要）
- 式 (16) $\nabla_x \mathcal{L} = y - t$ ✓；式 (17) $x = W^{\top}h$、$\nabla_h = W \nabla_x$、$\nabla_W = \sum_h h(\nabla_x)^{\top}$ ✓；Remark：「scale the gradients of the chunked inputs and the projection layer weights with the ratio of chunk size / ($B \times T$)」✓（= 章节「按 $M_c/M$ 缩放」）
- 「in-place replacement of the logit tensor…preventing the double materialization of two large tensors」✓
- repo docstring：「Since Cross Entropy Loss is the last layer, we can compute the gradient at the forward pass. By doing so, we don't have to store the _input and target for the backward pass」✓（逐字一致）；`reduction='none'` 路径 defer 到 backward 重算的源码注释 ✓
- 块长注释（v0.5.2）：「BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048」✓（当前 main 已改写，见 S3-2）

### 3.4 PyTorch 源码核对（GitHub raw / contents API，2026-11-17）

- GradScaler 默认值：`init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000`（grad_scaler.py 128–131 行）✓；文档口径「After that, step skipping should occur rarely (once every few hundred or thousand iterations)」✓
- autocast fp32 清单：`AT_FORALL_FP32` 含 `layer_norm / native_layer_norm / rms_norm / group_norm`；`AT_FORALL_FP32_SET_OPT_DTYPE` 含 `softmax, int` 与 `log_softmax, int`（autocast_mode.h 854–922 行）✓ ——**写作方存疑项 (b) 核实成立**：autocast 下 `log_softmax` 以 fp32 执行，其输出（$[M, V]$）被 save 供反向，fp32 口径成立
- `derivatives.yaml`：`mm` 条目「`self: mm_mat1_backward(grad, mat2, ...)` / `mat2: mm_mat2_backward(grad, self, ...)`」（1239–1241 行）✓；`matmul` 走 `matmul_backward`（384 行）✓
- engine.cpp：依赖计数到零入就绪队列（535–540 行 `--it->second == 0 … heap.push`）✓；「We allocate one thread to handle all CPU operations, and a thread per CUDA device」（365–367 行注释）✓ ——**写作方存疑项 (a) 前半核实成立**
- inference_mode：grad_mode.py docstring「disabling view tracking and version counter bumps…tensors created in this mode cannot be used in computations recorded by autograd」✓；RFC-0011「Inplace modifications on inference tensors outside of inference mode [are disabled]」「The immutable invariant」✓ ——**存疑项 (a) 后半（变异禁令）核实成立**
- checkpoint.py：`CheckpointFunction` 前向 `no_grad` + `save_for_backward` 输入 + RNG stash（`get_rng_state` / `get_device_states` / `_get_autocast_kwargs`）✓；反向 `fork_rng` + `set_rng_state` + `enable_grad` 重跑 ✓；非重入 `_checkpoint_hook` pack 返回 `_Holder` 占位 + metadata（shape/dtype/device）✓；`_StopRecomputationError` early stop（默认 `early_stop=True`）✓；`_DEFAULT_DETERMINISM_MODE = "default"` ✓；docstring「It is recommended that you use `use_reentrant=False`」✓；重入版不兼容 `.grad()`/`inputs` 参数的 RuntimeError ✓；`checkpoint_sequential(functions, segments, input)` ✓；`_NoopSaveInputs` 在 v2.10.0 存在、v2.15.0 已重构（见 S3-3）
- distributed.py：`no_sync` docstring「The forward pass should be included inside the context manager, or else gradients will still be synchronized」✓；`require_backward_grad_sync` 门控（1003/1717–1722 行）✓；除 world size 后求和（1217 行 `delay_grad_buffer.div_(world_size)`、`divide_by_initial_world_size=True` 默认，2028/2066 行）✓

### 3.5 数值复算（独立重算，非正文自述）

| 项 | 章节值 | 复算 | 结论 |
| --- | --- | --- | --- |
| 1.1 §3 物化 Jacobian 内存 | $d$=4096、$M$=4096 → 128 GiB | $4096 \times 4096^{2} \times 2 = 1.374 \times 10^{11}$ B = 127.98 GiB | ✓ |
| 1.2 §2 down 投影前向 | ≈$4.8 \times 10^{11}$ FLOPs | $2 \times 4096 \times 14336 \times 4096 = 4.81 \times 10^{11}$ | ✓ |
| 1.2 §3 softmax Jacobian | $V$=128256 → $V^{2} \approx 1.6 \times 10^{10}$ 元素 ≈ 31 GiB | $1.645 \times 10^{10}$ 元素、×2 B = 30.6 GiB | ✓ |
| 1.2 §10 logits 账 | bf16 0.98 GiB/序列；fp32 save 1.96 GiB；$B$=8 → 24 GiB；块 2048 → 0.49 GiB；$M$=32768 → 7.8 GiB | $128256 \times 4096 \times 2 = 0.979$ GiB；×4 B = 1.957 GiB；$(0.98+1.96) \times 8 = 23.5$；$2048 \times 128256 \times 2 = 0.49$ GiB；$32768 \times 128256 \times 2 = 7.83$ GiB | ✓ |
| 1.2 §7 embedding 行稀疏 | 26% | $32768 / 128256 = 25.5\%$ | ✓ |
| 1.3 式 (4) 窗口宽度 | $65504 \cdot 2^{24} \approx 1.1 \times 10^{12}$ | $= 1.099 \times 10^{12} = 2^{40.0}$ | ✓ 数值；单位表述见 S2-2 |
| 1.3 §6.3 粒度开销 | 权重 0.024%、激活 3.1% | $4/16384 = 0.0244\%$；$4/128 = 3.125\%$ | ✓ |
| 1.4 §1 实例 | 794624 B/token/层 ≈ 776 KB；32 层 24 MB/token；$B{\cdot}T$=32768 → 约 800 GB | $34 \times 4096 + 5 \times 32 \times 4096 = 794624$；$\times 32 = 24.3$ MB；$\times 32768 = 8.33 \times 10^{11}$ B = 776 GiB ≈ 833 GB | ✓（GB/GiB 混读但量级对） |
| 1.4 §3 选择性占比 | GPT-3 $s/(6h)$ ≈ 2.8%、MT-NLG ≈ 1.7% | $2048/73728 = 2.78\%$；$2048/122880 = 1.67\%$ | ✓ 自算值；与论文 2.7/1.6 的 0.1pp 差见 S3-6 |
| 1.4 §5 决策算例 | 22 GiB / 7.2 GiB | **应为 13 GiB / 4.25 GiB**（基线误用 114/912，本例基线 194） | ✗ → S1-1 |
| 1.4 练习 3 | $2\sqrt{61} \approx 15.6$ | $2 \times 7.810 = 15.62$（61 为素数，整除约束下只有 $k$=1, 61） | ✓ |
| 1.5 §2 算例 | $B_{\mathrm{tok}} = 2^{27} \approx 1.3 \times 10^{8}$ | $8 \times 16 \times 256 \times 4096 = 134{,}217{,}728 = 2^{27}$ | ✓ |
| 1.5 §4 通信账 | $M \approx 13$ GiB；naive 209 GiB；no_sync 26 GiB | $2 \times 7 \times 10^{9}$ B = 13.04 GiB；$8 \times 2M = 208.6$；$2M = 26.1$ | ✓ |
| 1.1 练习 2 | (a) $4096 \times 1.5 \times 10^{10}$；(b) $7 \times 10^{9}$ 巡 @ $10^{15}$ FLOPS | $6.14 \times 10^{13}$；$4.3 \times 10^{23}$ FLOPs ≈ 4978 天 | ✓（练习口径自洽） |

### 3.6 跨章引用完整性（对 models 维度）

- models 1.2 式 (1) $2mnk$ ✓、式 (3) $\sum_b x_b^{\top} G_b$ ✓、式 (4) 反向 $4N$ ✓、式 (5) $6ND$/练习 4 的 $8ND$ ✓、式 (6) causal 减半 $2LT^{2}d$ ✓、§8 账二 $65536 + 16T$ 与定长项结构 $4d + (H_q + 2H_{kv})d_{\mathrm{head}} + 3d_{\mathrm{ff}}$ ✓、$B$=8/$T$=4096 → 256 GiB ✓、§9 $N \approx 6.98 \times 10^{9}$ ✓
- models 2.3 式 (8) softmax Jacobian $J = \mathrm{diag}(p) - pp^{\top}$ ✓、§3.3 对称与行和为 0 ✓、§3.4 饱和 ⇒ 梯度消失 ✓、§4 max 减法 ✓、§7 scores+probs 表（64 GiB 行）✓ —— 1.2 §9 指针误写为「models 1.2 §7」→ S2-1
- models 2.7 LN Jacobian 引理 ✓（§5 谱范数引理在案）、QK-norm ✓
- models 4.1 FlashAttention 前向（online softmax / tiling）✓

### 3.7 spec §5 Part 1 大纲覆盖度

1.1–1.5 五章对照 spec 大纲逐条比对：全部要点在案（1.1：三步形式化/DAG/vjp/两种模式账/2× 依据闭合 6ND 边界/autograd 走读四 API/动态图含义；1.2：matmul/softmax/softmax-CE/RMSNorm-LN/GELU-SwiGLU/embedding/dropout/FA 反向/fused-CE/vjp 总表；1.3：格式表/三件套/loss scaling 三式/bf16 论证/E4M3-E5M2 分工与 V3 全 E4M3 对照/blockwise 粒度账/高精度累加/master fp32 论证；1.4：逐算子账/√L 最优/full vs selective/TP-PP 交互/checkpoint.py 两模式/决策表；1.5：累加等价/归一化位置/$B_{\mathrm{tok}}$ 口径/裁剪与计数陷阱/no_sync 通信账/实现走读）。无遗漏项；前瞻引用（4.1、5.1、5.3、2.x、3.1、7.2）均与 spec 依赖图一致，无循环前置。

### 3.8 文风与规范（rubric A/B + STYLE_GUIDE §2/§9）

- 禁令词（lint 词表 + rubric A 扩展表）全文扫描：零命中。「显著提升」一处出现在论文引文转述内（1.3 §6.4，「最小能显著提升精度…」对应原文「significantly improve precision without introducing substantial overhead」），按 rubric 豁免。
- 历史叙事：零命中（无时间线、无人物故事、无「某年某司发布」段落）。
- 空洞评论段：五章总结均为表格 ✓；练习每章 4 题（推导/计算/代码混合）✓；metadata 块（定位|前置|锚点+证据等级）五章齐全 ✓。
- 含糊量词：扫描「一些/很多/大量/较快/略微」零命中。
- 术语一致性：vjp/jvp、adjoint、binade 等首现处均有「中文（English）」或定义；发现一处单位术语漂移（数量级 vs 二倍程 → S2-2）。
- 证据等级：全部引用带 [A]/[B]/教学参照 标注 ✓；arXiv ID 与 references.md 条目互查一致（lint 项 4 通过 + 3.1 复核）。
- 教学代码：五章代码块均标「教学用，非优化实现」✓；§6.4/§4/§10/§5 的代码逐行审读，语义正确（FusedSoftmaxCE 的 max 减法、$p - e_y$ 就地、$1/M$ 归约位置；ChunkedLinearCE 的逐块重算与跨块 $dW$ 累加；MyLinear 的 vjp 顺序；累加循环的除 $m$ → clip → step → zero_grad 次序）✓。

### 3.9 覆盖率声明

- 五章全文逐行通读；全部 14 个编号公式（1.1 式 (1)–(4)、1.2 式 (1)–(15)、1.3 式 (1)–(6)、1.4 式 (1)–(5)、1.5 式 (1)–(6)）逐一独立重推或逐步验算（含强制清单三项的全部中间步骤，见 §1）。
- 全部 10 个 arXiv ID 经 API 核标题（100%）；全部 12 个 GitHub 路径经 contents API（100%）；正文数值事实抽查 16 项（见 3.5，覆盖五章全部量级性论断的代表项）15 项通过、1 项 FAIL（S1-1）。
- 论文论断核对：FlashAttention 8 项、Korthikanti 11 项、Micikevicius 2017 8 项、FP8 Formats 6 项、DeepSeek-V3 12 项、Chen 2016 3 项、Li 2020 2 项、Baydin 4 项、Liger 6 项（论文 + repo 合计）——逐项引原文对照，除 findings 所列外全部一致。
- 未覆盖声明：1.1 §7「派发微秒级 / matmul 毫秒级」为量级论述未逐项实测（不影响结论）；DDP「除 world size」语义以源码默认参数 + 论文 §3.2.4 双重确认，未运行分布式实验。

## 4. 放行结论

**不放行。** S1-1（1.4 §5 决策算例数字错误、结论翻转）必须修复并复审该节；S2-1–S2-4 同步修复；S3 ×9 记录入下波清零。修复量大头在 1.4 一段与 1.3 两处措辞，预计小改即可闭环。其余内容（数学推导、论文口径、源码机制、代码正确性、规范符合度）经逐项核查扎实，尤其写作方自报的两项存疑点（autograd 引擎调度 + inference_mode 变异禁令、torch CE fp32 save 口径）经源码与 RFC 独立核实均成立。

## 修复复审记录（2026-11-17，轮次 2）

复审方式：重读五章当前全文（非 diff）；每条 finding 逐条对照当前文本；S1-1 的换算链逐步重算；修复触及处查新增问题；五章 lint 重跑（2026-11-17）全部 LINT OK；CJK 字数实测 3646 / 4977 / 3969 / 3687 / 3525（1.1 已达标；1.2 推导重章软目标 5500 仍未及，软目标不阻断）。

### S1-1（1.4 §5 决策算例）——ADDRESSED，关闭

当前文本（§5）：「此几何下 $5as/h = 160$，式 (1) 的系数为 $34 + 160 = 194$，全存激活约 194 GiB（794624 字节 × 32 层 × 8192 token），缺口近 10 倍。逐档换算：只上 selective 剩 $34/194$ 即 34 GiB，仍超预算；TP 8 路加 selective 为 $194 \times (10 + 3)/194 = 13$ GiB，已满足 20 GiB 预算、只付 2–7% 时间；再叠 SP 压到 $34/(8 \times 194)$ 即 4.25 GiB，属进一步压缩换时间；full checkpoint 压到 $194 \times 2/194 = 2$ GiB，但付 30–40% 时间」。

逐步重算：基线 $34 + 160 = 194$ 显式立出 ✓；794624 B × 32 × 8192 = 2.0835×10^11 B = 194.0 GiB ✓；selective $34/194$ → 34 GiB ✓（超 20 GiB，「仍超预算」正确）；TP $t=8$ + selective 因子 $10 + 24/8 = 13$ → $194 \times 13/194 = 13$ GiB ✓（论文 Table 2 行）；+SP 因子 $34/8 = 4.25$ → 4.25 GiB ✓；full $2/194$ → 2 GiB ✓。换算链全部除以本例自身基线 194，GPT-3 的 114 不再混入；阶梯叙事（selective 不够 → TP+selective 已满足 → +SP 进一步压缩 → full 压到底但最贵）与数字自洽，决策表读法结论成立 ✓。

### S2-1（1.2 §9 指针）——ADDRESSED，关闭

§9 两处现为「与 models 2.3 §7 的 scores+probs 表同口径」「这正是 models 2.3 §7 那张 $O(T^{2})$ 内存表被消掉的方式」✓（models/02-transformer-anatomy/03-attention-mechanism.md §7 表确有 scores+probs ×32 层 = 64 GiB 行）。残留（S3 级）：本章参考节 flashattention 条目仍写「Theorem 2 的 IO 复杂度」，正文已改为 Theorem 2+5，参考条目可顺手同步。

### S2-2（1.3 单位）——ADDRESSED，关闭

§3 式 (4) 符号注现为「不超过 40 个二倍程（约 12 个十进制数量级）」✓（$2^{40.0} = 1.1 \times 10^{12}$ 复核一致）；§4 现为「离下溢地板有上百个二倍程」✓（$2^{-20}$ → $2^{-126}$ 为 106 个二倍程）。本章 §1 自定义的 binade 术语与两处用法现已一致。

### S2-3（1.2 §4 bf16 下溢数字）——ADDRESSED，关闭

现为「bf16 的最大可表示值约 $3.4 \times 10^{38}$，$p_y$ 低于约 $3 \times 10^{-39}$ 时 $-1/p_y$ 直接溢出为 $-\infty$；再往下的次正规区（最小正规数约 $1.2 \times 10^{-38}$、最小次正规数约 $9.2 \times 10^{-41}$）里 $p_y$ 本身的尾数逐位丢失」——逐数复核：$3.39 \times 10^{38}$ ✓、$1/3.39 \times 10^{38} = 2.9 \times 10^{-39}$ ✓、$2^{-126} \approx 1.18 \times 10^{-38}$ ✓、$2^{-133} \approx 9.2 \times 10^{-41}$ ✓，与 1.3 §1 表自洽。

### S2-4（1.1 篇幅）——ADDRESSED，关闭

CJK 实测 3254 → 3646（≥ 3500 达标）。新增内容分布在 §2（DAG 拓扑序与引擎调度的衔接）、§3（物化全链的 FLOPs 侧账）、§4（模式差折成 14 年 vs 0.1 秒）、§5（2N/4N 绝对数）、§6.2（256 MiB save 实例）、§6.3（retain_graph 量级）、§6.4（next_functions 观察）、§7（AOTAutograd / Compiled Autograd 一句）——逐处核验见下「新增问题」。

### S3-1 – S3-9——全部 ADDRESSED，关闭

- S3-1：1.3 §4 现为「有效位 8 位（含隐含位），乘积至多 16 位，不超 fp32 的 24 位」✓（与 §2 fp16「11 + 11 = 22」口径一致）。
- S3-2：1.2 §10 现为「早期版本实现注释的口径：…⇒ 块长 2048；当前 main 已改写为按显存预算常数求块长，思想相同」✓（与 repo 现状核实一致：v0.5.2 注释原文、main 的 `_CHUNK_MEM_CONST` 预算注释）。
- S3-3：1.4 §4 加版本注记「此类名核实于 PyTorch ≤ 2.10 的源码…更新的 release 已把该逻辑重构进 `_CheckpointFrame`，机制不变」✓；参考条目同步「（≤ 2.10 类名，新版本已重构）」✓。
- S3-4：references.md pytorch-repo 被引章节现为「1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.4」✓（controller 已补）。
- S3-5：1.2 §9 现为「论文 Theorem 2 的前向界与 Theorem 5 的反向界，同型」✓。
- S3-6：1.4 §3 符号注改为「分母为层前向的投影与 MLP matmul（不含 attention 本体 $4s^{2}h$）」，并新增「与论文的 2.7% / 1.6% 差 0.1 个百分点，差额来自分母口径：论文（Appendix A）的分母另含 attention 本体与 logits 项」✓（与我对 Appendix A 式 (7)(8) 的独立核算一致）。
- S3-7：1.4 §3 现为「530B 与 1T 规模上 selective + SP 一侧开销降到 2%（论文 Figure 8 的合并口径）」✓。
- S3-8：1.2 §4 现为「eager 反向是 log_softmax 与 nll_loss 两个 ATen 算子反向的复合，其代数效果恒等于 $p - e_y$」✓。
- S3-9：1.1 §2 图注现为「（adjoint 流，式 (4) 的逐节点 vjp——vjp 在 §3 定义）」✓。

### 新增问题（只查修复触及处）

- **N1（S2，新）**：1.1 §6.3 retain_graph 条新增句「同一图做两次反向，这段时间内的峰值激活近似翻倍」机制错误。retain_graph 的效果是保存量**驻留时间**延长到最后一次反向；同一图两次反向期间任何时刻只有一份保存量加反向工作区，峰值不翻倍（这正是 §6.3 同条目前半句「激活内存活到最后一次反向」的正确口径）。建议改为「…驻留在百 GiB 级，两次反向期间一直占着——retain_graph 是用显存驻留时长换『不重跑前向』的开关」。
- N2（S3，新）：1.4 §4 版本注记的指针「版本漂移的一般性风险见 training/README 的锚定纪律」悬空——training/README.md 无此命名内容（仅「前置知识」节有一句代码走读锚点列举）。建议落一句锚定纪律进 README 或删该分句。
- N3（S3，新）：1.4 §5 修复句「再叠 SP 压到 4.25 GiB，属进一步压缩换时间」——「换时间」方向与论文 Table 4 不符（22B 上 SP+selective 4% 低于 selective 单独的 7%，SP 摊薄 LN/dropout 反而省时）。建议「属进一步压缩（SP 的通信与时间账归 5.2）」。
- N4（S3，观察）：1.1 §5 新增「每 token 前向 $2N \approx 1.4 \times 10^{10}$」（不含输出头）与 §1/§4 的 $1.5 \times 10^{10}$（含输出头）相邻并存，两者各自正确、均可在 models 1.2 对号入座；建议 §5 处加半句互注（「含输出头约 $1.5 \times 10^{10}$」）。

### 复审结论

轮次 1 的全部 14 条 findings（S1×1、S2×4、S3×9）关闭。修复新引入 S2×1（N1）+ S3×3（N2–N4）。**P1 暂不放行**：N1 为一词之改（「峰值激活近似翻倍」→ 驻留时长口径），修复后 P1 放行、无需再复审（该节其余内容两轮核查均已闭环）；N2–N4 记入 S3 累计。1.5 全章无修复触及、本轮重读确认与轮次 1 结论一致。
