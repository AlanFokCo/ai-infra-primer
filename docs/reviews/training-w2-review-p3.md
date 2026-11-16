# training W2 批 1 对抗审查报告 — P3 两章 + 4.1

- 日期：2026-11-17
- 审查对象：`training/03-loss-functions/01-ce-family.md`（3.1）、`training/03-loss-functions/02-preference-and-contrastive-objectives.md`（3.2）、`training/04-data-parallelism/01-dp-and-ddp.md`（4.1）
- 审查人：独立 fresh-context evaluator（task-13-brief.md），与写作 agent 无共享上下文
- 输入：3 章 + STYLE_GUIDE.md + rubric.md + training/references.md + training spec §5（3.1/3.2/4.1）与 §6.3
- 结论：**不放行**（S1 ×1；S2 ×3；S3 ×5）

## 0. 机械检查（lint）

```
bash scripts/lint-chapter.sh training/03-loss-functions/01-ce-family.md                        → LINT OK
bash scripts/lint-chapter.sh training/03-loss-functions/02-preference-and-contrastive-objectives.md → LINT OK
bash scripts/lint-chapter.sh training/04-data-parallelism/01-dp-and-ddp.md                        → LINT OK
```

篇幅（CJK 汉字数，标准章 ≥3500）：3.1 = 3808 ✓；3.2 = 3620 ✓；4.1 = 3801 ✓。三章均有 ASCII 图（3.1 §1 数据流、3.2 §6 TRL 走读、4.1 §2.2 环图 + §3.2 时序图）✓。

## 1. 强制独立重推导（逐步中间过程）

### 1.1 4.1 ring all-reduce 通信量与 DP 平均梯度等价 —— PASS

**ring all-reduce 每 rank 发送量 $2(d-1)/d \cdot M$**（4.1 式 (2)）：

设 $d$ 个 rank 排成环，被归约向量 $M$ bytes 切成 $d$ 块、每块 $M/d$。

scatter-reduce 阶段（$d-1$ 步）：第 $s$ 步（$s = 1, \ldots, d-1$），rank $i$ 把手上块 $(i - s + 1) \bmod d$ 的部分和发给 $i+1$，同时从 $i-1$ 收一块并加进本地同名块。**不变式**：第 $s$ 步结束时，块 $c_j$ 在 rank $(j + s) \bmod d$ 上累加了恰好 $s + 1$ 份贡献（来自 rank $j, j+1, \ldots, j+s$）。归纳：$s = 0$ 时各 rank 持自有块（1 份贡献）✓；设 $s$ 步末成立，第 $s+1$ 步 rank $(j+s)$ 把 $c_j$（$s+1$ 份）发给 $(j+s+1)$，后者加进自有同名块（1 份）→ $s+2$ 份 ✓。$s = d-1$ 时 $c_j$ 在 rank $(j-1) \bmod d$ 集齐全部 $d$ 份贡献，且每 rank 恰持一个全和块（对照 4.1 的 $d = 4$ 例：step3 后 r1 持 c0、r2 持 c1、r3 持 c2、r0 持 c3 ✓）。每步每 rank 发一块 $M/d$ ⇒ scatter-reduce 累计 $(d-1) \cdot M/d$。all-gather 阶段同构（转发全和块、无加法）：再 $(d-1) \cdot M/d$。合计

$$C_{\mathrm{rank}} = 2 \cdot \frac{d-1}{d} \cdot M \qquad \checkmark$$

端点自检复核：$d = 2$ 时 $C = M$（每 rank 恰发全量一次，= 互换下界）✓；$d \to \infty$ 收敛 $2M$ ✓。数值复核（$M$ = 7B bf16 梯度 = $2 \times 7 \times 10^{9}$ B = 13.039 GiB）：$d = 2 \Rightarrow 13.04$ GiB；$d = 512 \Rightarrow 26.03$ GiB（≈ 2M，与 1.5 §4 的 26 GiB 同一数字 ✓）。星型基线：rank 0 收 $(d-1)M$ + 发 $(d-1)M$，链路承载 $2(d-1)M$；$d = 512$ 时单向 $511 \times 13.039$ GiB = 6.51 TiB ✓（4.1：「收发各约 6.5 TiB」✓）。时间模型 $T = 2(d-1)\alpha + 2(d-1)/d \cdot M\beta$（两阶段共 $2(d-1)$ 条消息 ✓）。

**DP 平均梯度等价**（4.1 式 (1)）：$g_r = (1/b)\sum_{j \in S_r} \nabla \ell_j$（分片内平均定义）；$\bar g = (1/d)\sum_r g_r = (1/(db))\sum_{r,j}\nabla \ell_j = \nabla\left[(1/(db))\sum_j \ell_j\right]$（梯度线性性）。等权前提（$|S_r|$ 全等）正文已声明 ✓。无近似 ✓。

### 1.2 3.1 logsumexp 稳定性与 z-loss 对 logit 的梯度 —— PASS（另发现一处关联断言错误，见 F-1）

**logsumexp 稳定**（3.1 式 (3)）：$\log \sum_v e^{z_v} = \log\left(e^{m}\sum_v e^{z_v - m}\right) = m + \log\sum_v e^{z_v - m}$，$m = \max_v z_v$。平移后 $z_v - m \le 0$ ⇒ 每项 $\in (0, 1]$、最大项恰为 1 ⇒ 和 $\in [1, V]$ ⇒ $\log$ 和 $\in [0, \log V]$ ⇒ 总值 $\in [m, m + \log V]$，永不溢出 ✓。对 $z_j$ 求导：$\partial/\partial z_j = e^{z_j}/Z = p_j$，与式 (1) 第一项合成 $p - e_y$ ✓。数值复核：$\ln(3.4 \times 10^{38}) = 88.72$（溢出点 ≈ 88 ✓）；$e^{88} = 1.65 \times 10^{38}$ ✓；fp16 $\ln 65504 = 11.09$ ✓；$\ln 128256 = 11.76$（≈ 11.8 ✓）。

**z-loss 梯度**（3.1 式 (7)）：$L_{z,t} = (\log Z_t)^2$，$\partial(\log Z)^2/\partial z_j = 2\log Z \cdot (1/Z) \cdot \partial Z/\partial z_j = 2\log Z \cdot e^{z_j}/Z = 2\log Z \cdot p_j$ ✓。平移方向内积：$\langle p - e_y, \mathbf 1\rangle = 0$ ✓；$\langle 2\log Z \cdot p, \mathbf 1\rangle = 2\log Z$ ✓。动力学（式 (8)）：$\dot{(\log Z)} = \sum_j p_j(-2\lambda_z\log Z \cdot p_j) = -2\lambda_z\lVert p\rVert^2\log Z$ ✓；不动点 $\log Z = 0$、变号反向、自稳定 ✓；$\lVert p\rVert^2 \in [1/V, 1]$ ✓（均匀 $V \cdot (1/V)^2 = 1/V$；one-hot $= 1$）。

**例外**：同段的「CE 上抬力」断言 $\eta(p_y - \lVert p\rVert^2) \ge 0$（非均匀严格为正）不成立——见 F-1（S1）。练习 3(b) 的平衡式 $\log Z^{\ast} = (p_y - \lVert p\rVert^2)/(2c_z\lVert p\rVert^2)$ 本身代数正确（代入 0.9 / 0.85 / 0.001 得 ≈ 29.4），但其前提数值恰好落在 $p_y > \lVert p\rVert^2$ 的区域——与 F-1 同根。

## 2. Findings

### S1（事实性致命，必须修复 + 复审）

**F-1 ｜ 3.1 §5（line 129）｜CE 梯度流抬升 $\log Z$ 的不等式为伪命题**

- 断言：「梯度下降下 $\mathrm{d}\log Z/\mathrm{d}t = p^{\top}(-\eta g_{\mathrm{CE}}) = \eta(p_y - \lVert p\rVert^2) \ge 0$（非均匀分布下严格为正）：CE 把正确类 logit 往上推，顺带抬高了 logit 水平」。
- 证据：等号部分正确（$\sum_j p_j \cdot (-\eta)(p_j - \delta_{jy}) = \eta(p_y - \lVert p\rVert^2)$ ✓），但 $\ge 0$ 不普遍成立。**反例**：$p = (0.4, 0.6)$、目标类 $y$ = 第一类：$p_y - \lVert p\rVert^2 = 0.4 - 0.52 = -0.12 < 0$——此时 CE 梯度流在**压低** $\log Z$（把更大的错误 logit 往下推的速度超过抬正确类）。正确的充分条件：$y$ 为 argmax（此时 $\sum_{j \ne y} p_j^2 \le p_y(1 - p_y)$ ⇒ $p_y - \lVert p\rVert^2 \ge 0$，非均匀严格为正）——即模型「答对且够尖」之后才成立；训练早期 / 难 token 上常不成立。
- 影响：该句是「z-loss 是 CE 上抬通道的配重」叙事（及其后练习 3(b) 的平衡式直觉）的力学依据；不修则读者拿到的因果链在训练前半段是反的。
- 建议修正：把无条件不等式改为带条件陈述，例如「$\mathrm{d}\log Z/\mathrm{d}t = \eta(p_y - \lVert p\rVert^2)$：目标位为 argmax 时非负（非均匀严格为正）——训练中后期 CE 持续把 $\log Z$ 往上抬；早期正确类非 argmax 时该力可反向，但这不改变 z-loss 锁定平移自由度的职责（z-loss 的不动点 $\log Z = 0$ 与 CE 方向无关）」。

### S2（实质问题，必须修复）

**F-2 ｜ 3.2 §6 走读图 + 代码事实 4（lines 133, 146）｜TRL DPO 的逐 token logp 路径过时**

- 断言：「逐 token logp 用 `trl/trainer/utils.py` 的 `selective_log_softmax`（docstring 自述：等价于 `torch.gather(logits.log_softmax(-1), ...)` 但不物化全表）」，标注 2026-11-17 核实。
- 证据：TRL main（2026-11-17 拉取）`trl/trainer/dpo_trainer.py` 中**已无** `selective_log_softmax` 调用；`_compute_loss` 的 per-token logp 来自 `model(**model_kwargs, labels=labels, fused_lm_head=True).log_probs`（fused LM head，`trl/trainer/utils.py` 的 `add_fused_lm_head` 补丁，返回 `FusedCausalLMOutput.log_probs`，同样不物化 logits）。该改动由 commit ba406309c7「Score DPO and KTO tokens with the fused LM head (#7390)」于 **2026-10-07** 落地——早于章节声称的核实日期。`selective_log_softmax` 仍在 `utils.py`（line 535），被 `selective_log_softmax_and_entropy` 等消费，但不在 DPO 逐 token logp 路径上。
- 建议修正：走读改写为 fused LM head 口径（机制结论反而加强：logp 在 kernel 内按 label  gather 产出，logits 全程不物化——与 3.1 的 FLCE 同型），或给 TRL 钉版本 tag 并注明 `selective_log_softmax` 为该版本口径。

**F-3 ｜ 4.1 §3.2 + 参考条目（lines 99, 148）｜`divide_by_initial_world_size` 不是构造参数**

- 断言：「当前实现把『除以 world size』折进归约过程，构造参数 `divide_by_initial_world_size=True` 为默认」。
- 证据（pytorch/pytorch main，2026-11-17）：`DistributedDataParallel.__init__` 签名无此参数（distributed.py line 823 起）；它是 **`join()` context manager 的参数**（line 2028，`divide_by_initial_world_size: bool = True`），由 `_DDPJoinHook` 写入 `ddp._divide_by_initial_world_size`（line 423），仅在 uneven-inputs join 路径读出（line 1803）。正常路径的平均语义由 C++ reducer 的 `div_factor_` 实现：`reducer.cpp` line 585 `div_factor_ = process_group_->getSize()`、line 436 `bucket_view.div_(div_factor_)`。
- 建议修正：改为「除 world size 由 reducer 的 `div_factor_`（= process group size）在归约时折入；`divide_by_initial_world_size`（默认 True）是 `join()` 的 uneven-inputs 参数，与常规平均无关」。

**F-4 ｜ 3.1 §1 接口条目 1（line 81）｜docstring 注记张冠李戴**

- 断言：「target 两种形态：类别 id（long）或类别概率分布（float，**文档注明此时**等价于 LogSoftmax + NLLLoss 的组合）」。
- 证据（`torch/nn/modules/loss.py` CrossEntropyLoss docstring，2026-11-17 拉取）：LogSoftmax + NLLLoss 等价注记（"Note that this case is equivalent to applying LogSoftmax on an input, followed by NLLLoss"）挂在**类别 id** 分支下；概率 target 分支的注记是 "useful when labels beyond a single class per minibatch item are required, such as for blended labels, label smoothing, etc."。注记存在但被挂错了分支（「此时」= 概率分支）。
- 建议修正：把该注记移回类别 id 分支（「类别 id（long；文档注明此形态等价于 LogSoftmax + NLLLoss 组合）或类别概率分布（float，label smoothing 走此路径）」）。

### S3（minor，记录累计）

- **F-5 ｜ 3.1 §5（line 119）**：Mesh-TF 谱系句的出处指针写「§3.1 原文口径」；该句实出 ST-MoE §3 引言段（"This is an adaptation of the z-loss used for final softmax logits in the Mesh Tensorflow codebase"），z-loss 本体在 §3.3 引入。建议改「§3 引言 / §3.3」。
- **F-6 ｜ 3.1 §3 显存表（lines 67-73）**：B=8 列三个分量截断而非四舍五入（2BTV 真值 7.8281 → 表写 7.82；4BTV 15.656 → 15.65；合计 31.29 = 截断分量之和，真值 31.3125）。总结行「31.3 GiB → 0.49 GiB」正确。建议统一四舍五入（7.83 / 15.66 / 31.31）。
- **F-7 ｜ 3.1 §6 教学代码（lines 163-172）**：注释标「式 (1)(2)」，但实现是无 mask 的全员 mean（式 (2) 含 $m_{b,t}$）。建议注释改「式 (1) + 全员 mean」或补 mask 参数。
- **F-8 ｜ 3.2 §6 代码事实 3（line 145）**：「`delta_score = chosen_logratios - rejected_logratios`」为代数化简转写；源码（main）是 `delta_score = chosen_scores - rejected_scores`，其中 `chosen_scores = chosen_logratios` 仅在默认 `f_divergence_type="reverse_kl"` 分支成立。默认路径下代数等价，建议加「默认 reverse_kl 分支下」六字。
- **F-9 ｜ 3.1 §2（line 59）**：autocast cast 策略的文件指针 `torch/amp/autocast_mode.py`——逐算子 fp32 清单实体在 `aten/src/ATen/autocast_mode.h`（`AT_FORALL_FP32` / `AT_FORALL_FP32_SET_OPT_DTYPE`，CUDA 注册见 `autocast_mode.cpp`）；py 侧只有 `autocast` 类与文档索引。论断本身为真（softmax / log_softmax 在 fp32 set-opt-dtype 表；CE 家族经 log_softmax + nll_loss 分解被覆盖；CPU 侧另有 `cross_entropy_loss` fp32 直接注册）。指针沿用 1.3 §2 惯例（W1 已放行），仅登记。

## 3. Verification log（抽查项 × 来源 × 结论）

### 3.1 论文一手来源（arXiv API / ar5iv 全文 / CrossRef，全部 2026-11-17 拉取）

| # | 抽查项 | 来源与引句 | 结论 |
| --- | --- | --- | --- |
| 1 | arXiv ID ↔ 标题（11 个） | export.arxiv.org API：2202.08906=ST-MoE、1512.00567=Rethinking Inception、2410.10989=Liger、2006.15704=PyTorch Distributed、1910.02054=ZeRO、1706.03741=Christiano、2305.18290=DPO、2405.09818=Chameleon、2405.14734=SimPO、1708.02002=Focal Loss、1909.08053=Megatron-LM | 全部一致 ✓ |
| 2 | z-loss 式 (5) 形式 $(1/B)\sum(\log\sum_j e^{x_j})^2$；Mesh-TF 谱系；$c_z = 0.001$ 扫描定；总 loss 式 (6) $L_{CE} + c_B L_B + c_z L_Z$；Table 4（4/6 vs 3/3 vs 3/3；-1.755 / -4.206 / -1.741）；App. B「1e-2 或更大 → z-loss 压到近零」；fp32 指数 aside；attention logits z-loss footnote | ar5iv 2202.08906 全文逐句："This is an adaptation of the z-loss used for final softmax logits in the Mesh Tensorflow codebase"（§3 引言）；Eq (5)(6)；Table 4 数字逐项一致；"With values of 1e-2, or larger, the z-loss shrinks near to zero"（App. B）；"we ensure all exponentiated tensors are cast to float32"（§3.4 aside）；footnote 4 "adding z-losses onto the attention logits which also improves model instability without hurting model quality" | 全部一致 ✓（谱系指针节号小误 → F-5；Table 4 的「验证集」限定词是合理转写：原文口径 "negative log perplexity on English"、eval capacity factor 2.0，split 未在表注重述——登记不改） |
| 3 | label smoothing：$q' = (1-\varepsilon)\delta + \varepsilon u$、$u = 1/K$、凸组合分解 $H(q',p) = (1-\varepsilon)H(q,p) + \varepsilon H(u,p)$、「防止最大 logit 远大于其余」机制表述 | ar5iv 1512.00567 §7：公式逐字一致；"LSR achieves the desired goal of preventing the largest logit from becoming much larger than all others"；分解式逐字一致；梯度 $\partial\ell/\partial z_k = p(k) - q(k)$ 原文同形 | 全部一致 ✓ |
| 4 | focal loss：$(1-p_t)^{\gamma}$ 调制、$\gamma = 2$ 主口径、dense 检测类别不均衡场景 | ar5iv 1708.02002："we found γ = 2 to work best in our experiments"；FL = CE 于 $\gamma = 0$ | 一致 ✓ |
| 5 | Liger：平均吞吐 +20% / 显存 −60%（对照 HF 实现） | arXiv 2410.10989 摘要："achieve on average a 20% increase in training throughput and a 60% reduction in GPU memory usage ... compared to HuggingFace implementations" | 一致 ✓ |
| 6 | DDP 论文：Fig. 2(a)(b) 60M fp32 参数分桶实测（Gloo ~500K 参数饱和、NCCL on NVLink 20M 未饱和）；Fig. 2(c)(d) ResNet-152 GPU 反向 ~250ms 与 NCCL on NVLink 同量级；Fig. 3(a) 梯度就绪乱序例（rank2: g1,g3,g4,g2）；Fig. 3(b) 未参与参数挂起；§3.2.2 分桶动机；§3.2.3 hook + 逆序 model.parameters() +「非完美解、工程开销最小的近似」；§3.2.4 no_sync 禁全部 hook；平均语义原话；§2.3 同引 ring-based 与 tree-based；bucket_cap_mb 默认 25MB | ar5iv 2006.15704 全文逐节核对：上述各条均逐字/逐项命中（"Gloo reaches pinnacle speed at around 500K parameters per input tensor, while there is no clear saturation signal for NCCL on NVLink with even 20M-parameter GPU tensors"；"The backward on GPU takes about 250ms to complete, which is in the same order of magnitude as NCCL on NVLink"；"as DDP always computes the average of all gradients and writes them back to parameter .grad field"；"In no_sync mode, all DDP hooks are disabled"；"using the reverse order of model.parameters() as the bucketing order ... not a perfect solution, but is an approximation that we can rely on with minimum engineering overhead"） | 全部一致 ✓ |
| 7 | ZeRO：$16\Psi = 4\Psi + K\Psi$（$K = 12$）；7.5B 标准 DP 模型状态 120 GB（Table 1） | ar5iv 1910.02054："$4\Psi + 12\Psi = 16\Psi$"；Table 1 首行 7.5B / $N_d$ = 1 → 120 GB | 一致 ✓ |
| 8 | Chameleon：输出侧 z-loss $10^{-5}\log^2 Z$ | ar5iv 2405.09818："we apply z-loss regularization ... by adding $10^{-5}\log^2 Z$ to our loss function"（final softmax 语境） | 一致 ✓ |
| 9 | **SimPO 动机句（写作方上报存疑项）**：「sum 换平均是平均 logp 隐式 reward 的出发点之一」 | ar5iv 2405.14734 §3 逐字："One solution is to use the summed token log probability as the reward, but this suffers from length bias – longer sequences tend to have lower log probabilities. ... To address this issue, we consider using the average log-likelihood as the implicit reward" | **存疑项解除**：章节表述与原文一致 ✓（length bias 确为论文明载动机之一，另一动机是与生成指标对齐） |
| 10 | Bradley & Terry 1952 书目：Biometrika 39(3/4):324-345 | CrossRef API：Biometrika, vol 39, issue 3/4, pp. 324-345, 1952 | 一致 ✓ |
| 11 | 公开 LLM 预训练配方无 label smoothing（缺席型断言） | ar5iv 2407.21783（Llama-3 herd）全文 grep "label smooth" = 0 次；GPT-3 / Llama-2 / DeepSeek-V3 / OLMo 2 未逐家全文复核（缺席型断言成本不对称），章节已正确标注证据类型为「披露中缺席」并给出逐家出处指针（2.2 §7 表已核存在且含此五家） | Llama-3 实证 ✓；其余按 hedge 后口径接受 |

### 3.2 代码一手来源（GitHub API raw，pytorch/main、linkedin/Liger-Kernel/main、NVIDIA/Megatron-LM/main、huggingface/trl/main，全部 2026-11-17 拉取）

| # | 抽查项 | 来源与引句 | 结论 |
| --- | --- | --- | --- |
| 12 | `torch/nn/modules/loss.py`：未归一化 logits（"do not need to be positive or sum to 1"）；mean 按未屏蔽权重和归一；构造签名；label_smoothing docstring（"mixture of the original ground truth and a uniform distribution" + 指向 1512.00567） | 文件拉取逐句核 | 全部一致 ✓（概率-target 等价注记挂错分支 → F-4） |
| 13 | `torch/nn/functional.py` log_softmax docstring 引句 | "While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable" | 引句一致 ✓ |
| 14 | autocast fp32 策略 | `aten/src/ATen/autocast_mode.h`：softmax / log_softmax 在 `AT_FORALL_FP32_SET_OPT_DTYPE`，nll_loss / logsumexp / binary_cross_entropy_with_logits 在 `AT_FORALL_FP32`（CUDA 注册见 autocast_mode.cpp line 193/199）；CPU 侧另有 `cross_entropy_loss` fp32 | 论断为真 ✓；py 文件指针不精确 → F-9 |
| 15 | `torch/nn/parallel/distributed.py`：bucket_cap_mb 默认 25 MiB（`_DEFAULT_BUCKET_CAP_MB = 25`、`_MB_TO_BYTES = 1024*1024`）；首桶缩小注释 "to reduce latency for early-computed gradients"；`no_sync` 翻转 `require_backward_grad_sync` + docstring 警告「forward 必须包在 context 内」；forward 读 `torch.is_grad_enabled() and self.require_backward_grad_sync`（line 1791/1865）；构造假设 docstring（参数预注册 + 跨进程同序） | 文件拉取逐行核 | 全部一致 ✓（`divide_by_initial_world_size` 张冠李戴 → F-3） |
| 16 | 首桶默认值 | `torch/csrc/distributed/c10d/reducer.hpp` line 29 `kDefaultFirstBucketBytes = 1024 * 1024`（1 MiB）；经 init.cpp 导出为 `dist._DEFAULT_FIRST_BUCKET_BYTES` | 「约 1 MB」✓ |
| 17 | Liger 路径与接口：`src/liger_kernel/transformers/fused_linear_cross_entropy.py` 的 `LigerFusedLinearCrossEntropyLoss`，`forward(self, lin_weight, _input, target, bias=None)`；`src/liger_kernel/ops/fused_linear_cross_entropy.py` 的 `LigerFusedLinearCrossEntropyFunction`；docstring「CE 是末端算子、梯度可在 forward 算完」 | 两文件拉取核："Since Cross Entropy Loss is the last layer, we can compute the gradient at the forward pass"；另有 line 419 "If cross entropy is the last layer, grad_output is 1.0" | 全部一致 ✓ |
| 18 | Megatron vocab 并行 CE：`megatron/core/ops/loss/cross_entropy.py` 存在，`VocabParallelCrossEntropy` + `calculate_logits_max` / `calculate_predicted_logits` / `calculate_cross_entropy_loss`；三次 [B,T] all-reduce（MAX → predicted SUM → sum_exp SUM）；`vocab_parallel_logits.float()` 升精度；旧路径 `megatron/core/tensor_parallel/cross_entropy.py` 为 deprecation shim | 两文件拉取逐行核：shim 原文 "Deprecated import path; use ``megatron.core.ops.loss.cross_entropy``" | 全部一致 ✓（两处 SUM 的书写顺序与源码互异——源码先 predicted 后 sum_exp，章节先 sum_exp 后 predicted；数学无关、逻辑序描述，登记不改） |
| 19 | TRL：数据三键 prompt/chosen/rejected；tokenize 三键 prompt_ids/chosen_ids/rejected_ids；collator 引句 "The first half of the batch corresponds to the chosen_ids..."；completion_mask prompt 0 / response 1；`_compute_loss` 内 `chosen_logratios = chosen_logps - ref_chosen_logps`；sigmoid 默认 `−F.logsigmoid(beta * delta_score)`；`logps = per_token_logps.sum(dim=1)`；`chunk(2, dim=0)  # batch is [chosen, rejected]`；precompute_ref_log_probs 与 ref_*_logps 键 | `trl/trainer/dpo_trainer.py` + `trl/trainer/utils.py` 拉取逐行核 | 除 F-2 外全部一致 ✓（F-8 为化简转写） |
| 20 | Megatron `megatron/core/distributed` 目录（DP 层、连续 grad buffer） | GitHub API 目录列举：`distributed_data_parallel.py`、`param_and_grad_buffer.py` 等在册 | 一致 ✓ |

### 3.3 数值重算（evaluator 独立计算）

| # | 项 | 重算 | 章节值 | 结论 |
| --- | --- | --- | --- | --- |
| 21 | bf16 logits（V=128256, T=4096） | 0.9785 GiB | ≈0.98 | ✓ |
| 22 | 8BTV 朴素合计（B=1 / B=8） | 3.914 / 31.3125 GiB | 3.91 / 31.29 | B=1 ✓；B=8 截断链误差 0.02 → F-6 |
| 23 | FLCE 增量峰值 2·M_c·V（M_c=2048） | 0.4893 GiB | 0.49 | ✓ |
| 24 | 「31 GiB ≈ H100 80GB 近四成」 | 31.3/80 = 39.1% | 近四成 | ✓ |
| 25 | fp32/bf16 溢出点 | ln(3.4e38)=88.72；e^88=1.65e38；fp16 ln(65504)=11.09 | ≈88 / 1.7e38 / ≈11 | ✓ |
| 26 | LS 最优 logit gap（V=128256, ε=0.1） | 13.959 nats | ≈14.0 | ✓ |
| 27 | LS 地板 H(q′) | 1.5012 nats | ≈1.50 | ✓ |
| 28 | log V（V=128256） | 11.762 | ≈11.8 | ✓ |
| 29 | z-loss 量级账 2×11.8 | 23.5；×0.001 ≈ CE 的 2.35% | 23.5、约 2% | ✓ |
| 30 | BT 校准：σ(0.693)、σ(2.2) | 0.6666、0.9003 | 2/3、≈0.9 | ✓ |
| 31 | ring $2(d-1)/d \cdot M$（d=8/64/512，M=13.039 GiB） | 22.82 / 25.72 / 26.03 GiB | 式 (2) 与练习 3 口径 | ✓ |
| 32 | 星型 d=512 单向 | 6.507 TiB | 约 6.5 TiB | ✓ |
| 33 | 1.5 数值例引用（m=8：209 / 26 GiB） | 2M×8 = 208.6 ≈ 209；2M ≈ 26 | 同 1.5 §4 | ✓（两章互洽） |
| 34 | 练习 3(b) 平衡值 | 0.05/0.0017 = 29.41 | 「估算量级」开放 | ✓ 代数正确（前提见 F-1） |
| 35 | **F-1 反例** | p=(0.4,0.6)、y=类1：$p_y - \lVert p\rVert^2 = -0.12 < 0$ | 章节断言 ≥0 | **证伪** → F-1 |

### 3.4 交叉引用与内部一致性（本地逐条核对）

| # | 指针 | 核对结果 |
| --- | --- | --- |
| 36 | 1.2 式 (5)（p−e_y）/ §10（FLCE、M_c=2048、2.9/24 GiB 口径）/ §11（反向/前向 ≈2→≈3）/ §4 归约注记 | 全部命中 ✓ |
| 37 | 1.5 式 (3)(4)(5)(6)（1/m、变长陷阱、B_tok=b·m·d·T、DP 合成式）/ §3（clip→step→zero_grad）/ §4（no_sync 账） | 全部命中 ✓ |
| 38 | 1.3 §2（autocast fp32 口径）、2.2 §7（五家配方表存在）、2.4 §6（placement 总图存在） | 命中 ✓ |
| 39 | models 1.1 式 (4)-(6)（NLL=CE=MLE、CE=H+KL）；models 1.2 式 (8)（(4+2c)N 优化器态 + 文中 16N 总量）；models 2.3 §4（max 减法）；models 2.7 §7/§8（QK-norm 上界 √d_k 变体口径、soft-capping 50/30 与 Gemma 3 弃用）；models 4.5 式 (5)(6)(12)（f_i/P_i 统计、aux loss、router z-loss）；models 6.2（InfoNCE 推导）；models 8.3 式 (1)（mask 语义）；models 8.5 式 (7)（Chameleon μ=1e-5） | 全部命中 ✓（1.2 式 (8) 指针指优化器态分量、16N 总量在同章文字/对账表——与全书既定指针惯例一致，接受） |
| 40 | 3.1/3.2/4.1 元数据锚点 ↔ references.md 条目（等级、key） | 一致 ✓（lint 项 4/5 同证） |

### 3.5 覆盖率声明

- 章内**全部** arXiv ID（11 个）经 API 核；**全部** repo 路径（torch ×4 文件 + reducer C++ ×2、Liger ×2、Megatron ×3、TRL ×2）经 GitHub API 拉原文核对；**全部**带引号的 docstring/论文原句逐字命中。
- 论文全文核对 7 篇（ST-MoE、DDP、ZeRO、label smoothing、focal、Chameleon、SimPO）；Bradley-Terry 1952 无 arXiv，经 CrossRef 核书目；BT 推导（Gumbel 差 = logistic）为 evaluator 独立重推导（§1 未列强制项但已推：$\int F(t+u)f(t)dt$ 经 $s = e^{-t}$ 代换 $= 1/(1+e^{-u})$ ✓，章节中间步无误）。
- 强制重推导 2 项完成（§1.1 / §1.2），中间过程与数值如上。
- 未覆盖：GPT-3 / Llama-2 / DeepSeek-V3 / OLMo 2 的 label smoothing 缺席逐家全文复核（缺席型断言，章节 hedge 口径正确，抽样 Llama-3 实证）；Megatron 两处 SUM 顺序描述差异按逻辑序接受。

## 4. 规范审查（STYLE_GUIDE + rubric A/B 类）

- lint 三章全绿（§0）；禁令词表机械扫描（值得注意的是/综上所述/让我们/强大的/……）零命中；含糊量词（很多/大量/较快/显著/略微/一些）零命中。
- 每节开场均为具体事实/公式/问题（rubric B5 ✓）；无历史叙事（§2 ✓——Bradley-Terry 1952 仅作 citation 与模型出处，无年代叙事）；总结均为表格、练习 2-4 题齐（§8 模板 ✓）。
- 术语无漂移（logits/loss mask/all-reduce/bucket 全章一致）；证据等级标注齐全且与 references.md 一致。
- 长句密度高（多条 250-600 字符行，math 剥离后）但均为分号/枚举结构的信息密集陈述，与 W1 已放行章节同型，不达 B2 的「一逗到底无停顿」阈值——登记观察，不报。

## 5. 结论

**不放行。** S1 ×1（F-1，3.1 §5 伪不等式——修复后 3.1 需复审）；S2 ×3（F-2 3.2 TRL 路径过时、F-3 4.1 构造参数张冠李戴、F-4 3.1 docstring 注记挂错分支）；S3 ×5（F-5~F-9 累计登记）。三项修复成本均低（均为局部改写，不动章节骨架）。强制重推导 2/2 PASS；写作方上报的 SimPO 存疑项经 ar5iv 原文逐字核对后**解除**（章节表述正确）。

## 6. 修复复审记录（2026-11-17 复审轮）

复审方式：重读三章当前全文（非 diff），逐条对照原 finding 的当前文本证据；lint 重跑三章全绿（`LINT OK` ×3）。

| Finding | 判定 | 当前文本证据 |
| --- | --- | --- |
| F-1（S1） | **ADDRESSED** | 3.1 §5 现文：「等号是逐字代数；力的符号有条件：目标位为 argmax 时 $\sum_j p_j^2 \le p_y \sum_j p_j = p_y$，力非负」——argmax 充分条件的一行证明数学正确（$p_j \le p_y$ 逐项放缩即得）；并补入反向算例「$p = (0.4, 0.6)$、目标为前者，$0.4 - 0.52 \lt 0$」（与本报告反例一致）；「训练中后期…多数位置已把正确类放上 argmax」的叙事限定得当，z-loss 职责与 CE 方向解耦的收束句成立。练习 3(b) 已同步：「符号条件见 §5：目标位为 argmax 时非负……该组参数下目标位是否 argmax、力是否为正，先判后代」（$p_y = 0.9 \gt 0.5$ 必为 argmax，自洽）。 |
| F-2（S2） | **ADDRESSED** | 3.2 §6 走读图与事实 4 已改为 fused LM head 口径。与 TRL main（2026-11-17 复核）逐点一致：`dpo_trainer.py` line 1041/1043 初始化处 `add_fused_lm_head(...)` 打补丁；前向 `model(..., labels=labels, fused_lm_head=True).log_probs`（line 1383 区）；docstring 口径「in tiles, only the positions whose next-token label is not −100」、返回 `FusedCausalLMOutput` 均逐字命中。历史注记「2026-10 前为 selective_log_softmax 的 gather 方案」经切换 commit（ba406309c7，2026-10-07）的父版本核实：旧版确用 `selective_log_softmax` / `selective_log_softmax_and_entropy`。 |
| F-3（S2） | **ADDRESSED** | 4.1 §3.2 现文：「『除以 world size』由 C++ reducer 在归约时折入（`reducer.cpp`：`div_factor_` 取 process group size、bucket 视图 `.div_(div_factor_)`）」与源码逐行一致（line 585 / 436）；新增「易混同名参数」句把 `divide_by_initial_world_size` 正确归为 `DDP.join()` 的 uneven-inputs 参数（join() line 2028 默认 True、`_DDPJoinHook` 写入、join 路径专用——与源码注释「divide grads by the static world size / reduced by the number of joined processes」一致）。参考条目同步改挂 `reducer.cpp`。 |
| F-4（S2） | **ADDRESSED** | 3.1 §1 接口条目 1 现文：「类别 id（long；文档注明此形态等价于 LogSoftmax + NLLLoss 的组合）或类别概率分布（float，文档列举的用途即 blended labels 与 label smoothing）」——等价注记归位类别 id 分支，概率分支用途与 docstring 原文（"such as for blended labels, label smoothing, etc."）一致。 |
| F-5（S3） | **ADDRESSED** | 3.1 §5 谱系指针改为「其 §3 引言的归因句与 §3.3 的本体定义」，与论文结构一致。 |
| F-6（S3） | **ADDRESSED** | 显存表 B=8 列改为 7.83 / 15.66 / 7.83 / 31.31，四舍五入正确（真值 7.8281 / 15.6563 / 31.3125）。 |
| F-7（S3） | **ADDRESSED** | 教学代码注释改为「式 (1) + 全员 mean（无 mask）」，式号与实现对齐。 |
| F-8（S3） | **ADDRESSED** | 3.2 §6 事实 3 改为「`delta_score = chosen_scores - rejected_scores`，默认 `f_divergence_type="reverse_kl"` 分支下 scores 即 logratios」，与源码逐字一致。 |
| F-9（S3） | 登记不改（按原建议） | 3.1 §2 仍指 `torch/amp/autocast_mode.py`——沿 1.3 §2 惯例保留，登记。 |

**修复新引入问题扫描（仅查修复触及处）**：

- S3（新，记为 F-10）：3.2 参考条目末行仍写「`trl/trainer/utils.py`（`selective_log_softmax`）」——走读主体已切换到 `add_fused_lm_head`，`selective_log_softmax` 只剩历史注记；参考条目的函数名应同步为 `add_fused_lm_head`（或并列两名）。minor，累计到下波清零。
- 观察（不定级）：3.1 §5 新文括号「支撑非退化时严格为正」的严格性边界有一格边角——等号还在「支撑上均匀且 y 并列最大」处成立（如两类各 0.5），不限于退化支撑；主线条件句（argmax ⇒ 非负）与算例均正确，此处为零测集措辞松紧，建议下波顺手改「非均匀支撑时严格为正」，不阻塞放行。

**复审结论：三章全部放行**（S1/S2 清零；S3 存量 6 条 = F-9 登记 + F-10 新增 + 观察 1 条，按下波统一清零口径处理）。
