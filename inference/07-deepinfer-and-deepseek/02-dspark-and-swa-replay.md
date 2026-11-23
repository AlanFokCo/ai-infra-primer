# 7.2 DSpark 与 SWA Bounded Replay（DSpark and SWA Bounded Replay）

> 固定 draft 5 个 token、平均只被接受 3.29 个——那 34.2% 打在低概率 token 上的 verify 算力，能不能按期望吞吐逐请求抢回来？SWA 窗口态一旦 miss，精确重建的 token 数随上下文与层数涨（20 层要 2,560 个）——能不能把它钉死在常数 128？本章走读 DSpark（引擎实现名 DFlash）的 Confidence Head → cumprod → 全局 top-k → 期望吞吐 argmax 变长 verify 链与它的 ZOS GPU 重算，证明 SWA bounded replay 的恒定成本与 hit policy 正确性，并对账 StateBlock 落盘字节 | 前置：4.4（接受规则、 $E[N_{\mathrm{tok}}]$ / $E[K]$ 口径、batch 浪费账）、4.5（EAGLE-2 confidence 剪枝前缀封闭）、6.2（overlap 与数据依赖分类）、2.2（前缀命中）、2.4 §8（StateBlock 字节与放置）、2.3 §4（MXFP8 格式账）、7.1（release 边界与 config recap） | 锚点：DeepInfer repo（教学参照 + 官方实现，main@a679bb86，`models/mtp/dflash.py` DFlash L82 / ConfidenceHead L46、`csrc/engine/mtp_varlen.hpp` MTPVarlen L133、`worker/cache_engine.py` StateBlock L221，2026-11-23 fresh tarball 核实）、DeepInfer doc（教学参照 + 官方披露，L217-245 / L303-324）、V4.1-Flash config [W]、YOCO [Sun et al., 2024] [A]（arXiv:2405.05254，模型侧源头指引）、TileLang [Wang et al., 2025] [A]（arXiv:2504.17577）

## 1. 两笔浪费：verify 算力与窗口态重建

fixed-5 投机解码每步给每个请求验 5 个 draft token，平均只接受 3.29 个——浪费比例 $1 - 3.29/5 = 0.342$（34.2%，DeepInfer 官方披露增益表，doc L309-316，核实 2026-11-23；accepted 口径 = 被接受 draft 数 $E[K]$，4.4 §3 口径注）。单请求下这 34.2% 浪费的是闲置算力，不花钱（4.4 §1：verify 计算只占步时 1.7%）；大 batch 短上下文下 verify 进入 compute-bound，每个被浪费的 token 都真实拉长墙钟（4.4 §6 已算 71.5% 浪费的账）。DSpark 的变长 verify 正是冲这笔账来的：按每个 draft token 的预测接受率排序，把 verify 预算分给边际收益最高的前缀子集，而非统一 $\gamma$。

第二笔浪费在 KV 侧。V4.1-Flash 的滑动窗口注意力（sliding window attention，SWA）只保留最近 128 个 token 的窗口态（config `sliding_window` = 128 [W]）；窗口态不进 paged block，住在 state cache 的固定 slot 里（doc L217）。前缀缓存命中后若窗口态 miss，朴素做法是把命中前缀重算一遍重建窗口——重算量随上下文长 $T$ 涨；CED（模型侧架构名，形式化归 models 维度，V4.1 待补）架构下 decoder 层的精确重建还要把依赖锥逐层展开，20 层要末 2,560 个 token（doc L241）。SWA bounded replay 把重算钉死在末 128 个 token：截断注意力窗到重算段内，成本与 $T$ 和层数无关（doc L237）。

**本章数值口径**（全章例题复用）：模型 = V4.1-Flash（40 层 + 3 nextn，encoder 20 层 / YOCO（You Only Cache Once）consumer 20 层， $d_{\mathrm{model}}$ = 5120，SWA 窗 $w$ = 128，落盘粒度 $b$ = 128，最大回溯 $\delta$ = 6，KEEP_TOKENS = 140）；DSpark 生产 draft $\gamma$ = 5（`dspark_block_size` = 5 [W]，`DEEPINFER_USE_NEXTN_DECODE` 默认 5）。§4 变长 verify 算例用教学批 $B$ = 2、nextn 5 与一条声明的教学成本曲线（非实测）；§6 生产增益表用官方 EP64/DP64 benchmark（SM100，官方披露）；§7-§9 用 $w$ = 128、 $L_{\mathrm{enc}} = L_{\mathrm{dec}}$ = 20、528 B/token。换口径处逐节声明。

符号（章内局部，spec §4.3 翻译）： $B$ = batch 内请求数； $T$ = 上下文长（doc 引文原记号 L 译为 $T$，L 在本章专归层数）； $L$ = 40 层， $L_{\mathrm{enc}} / L_{\mathrm{dec}}$ = 20； $H$ = 命中长（重算起点边界）， $H_1$ = block KV 命中长（doc 原记号 L1）； $Q$ = total_q（每步 verify 总 token 预算，含 golden）； $m = Q - B$ = 额外 draft 预算； $r_{i,j}$ = 第 $i$ 请求第 $j$ 位置的逐位置接受概率估计， $a_{i,j}$ = 前缀接受概率（cumprod）； $N_{\mathrm{out}}$ = 每步产出 token 数（沿 4.4 $N_{\mathrm{tok}}$ 先例）。选择计数用 $m$（ $k$ 专归 MoE top-k，spec §4.3；代码 / doc 的「top-k」是选择术语 prose，非 MoE $k$）；成本写 $\mathrm{cost}(Q)$（ $c$ 专归 CP 度，不占用）。

> **坑**：把 DSpark 当成另一种投机方法去找 `dspark.py`。触发条件：按 config 字段名 `dspark_*` 推引擎类名；现象：DeepInfer release 里没有 DSpark 类——引擎实现命名是 **DFlash**（`models/mtp/dflash.py` DFlash L82），config `dspark_*` 经 `hugging_face/config.py` `_build_dflash_cfg` L177 映射为 `cfg.dflash_cfg`。规避：DSpark 是模型 / 技术名（config `dspark_*`），DFlash 是引擎实现名，一句对照锁定；vLLM 侧 `spec_decode/` 同时有 `dspark/` 与 `dflash/` 目录（5.4 注），两引擎命名各自独立，勿混。

## 2. DSpark draft 侧：dspark_* config 与 DFlash 实现

draft 模型长什么样？模型侧结构形式化归 models 维度（V4.1 待补），本节只列 config 字段 [W] 与引擎读法，不推导 draft 的训练目标或网络结构。V4.1-Flash 的 `text_config` 里 DSpark 字段族（[DeepSeek-AI, V4.1-Flash config](../references.md)，[W]，hf-mirror 传输核实，2026-11-23 第四次重取）：

| config 字段 | 值 | 引擎读法（serving 侧） |
| --- | --- | --- |
| `dspark_block_size` | 5 | 一次 draft 的 token 数（= `DEEPINFER_USE_NEXTN_DECODE` 默认 5） |
| `dspark_noise_token_id` | 128799 | 块并行 draft 的占位 token（§2 下文） |
| `dspark_target_layer_ids` | [37, 38, 39] | draft 消费的 target 特征层（三层融合） |
| `dspark_markov_rank` | 256 | Markov logit 修正的低秩维 |
| `dspark_n_routed_experts` | 128 | draft 小 MoE 的 routed expert 数 |
| `dspark_num_experts_per_tok` | 3 | draft 小 MoE top-3（主模型是 384 routed top-6，7.1 recap） |

draft 是一个 128 expert top-3 的小 MoE（主模型 384 top-6 的约 1/3 宽、1/2 深路由），加一个 rank 256 的低秩 Markov 修正。引擎实现 DFlash（`models/mtp/dflash.py` L82，2026-11-23 核实）持 3 层 `DFlashDecoderLayer`（= config `num_nextn_predict_layers` = 3），关键部件逐件对 config：

- **特征融合投影**：`projection` = ReplicatedLinear( $3 \times d_{\mathrm{model}} \to d_{\mathrm{model}}$ )（L148），把 `dspark_target_layer_ids` = [37, 38, 39] 三层的 target hidden 拼成 $3 \times 5120 = 15360$ 维再投回 5120 维——draft 消费 target 已算出的多层特征，不重跑主干（4.5 §3 MTP「draft 复用 target 隐藏态」的同型接口）。
- **Markov 低秩修正**：`markov_w1` = VocabParallelEmbedding(vocab 129280 → 256)、`markov_w2` = ReplicatedLinear(256 → vocab)（L166-175）。采样第 $j$ 步时，上一步采出的 draft token 经 w1 查表得 256 维 embedding，再经 w2 映回 vocab 维加到 logits 上（`sample_drafts` L311-312：`logits = markov_w2(markov_w1(draft_token_ids), out_acc=logits)`）——一个 rank 256 的 token 级 Markov 修正，参数量 $129280 \times 256 \times 2 \approx 6.62 \times 10^{7}$（两个矩阵），相对主模型可忽略。
- **块并行 draft**：`_embed_draft_init_tokens` L230 把 block 5 个位置全填 `noise_token_id` 128799、位置 0 放 golden token（上一步已确认的 token），一次 forward 并行算出 5 个位置的 hidden；采样仍逐 step 进行（Markov 修正消费上一步采出的 token）。这是引擎实现口径的「块并行 draft」——draft 层前向一次打 5 位置，而非串行 5 次前向。
- **Confidence Head**：`ConfidenceHead` L46，docstring 逐字「Predict the acceptance probability of each draft token」。结构 = ReplicatedLinear( $d_{\mathrm{model}} + \mathrm{markov\_rank} \to 1$ )，前向是融合 TileLang op `confidence_head_fused`（`tilelang_ops/dflash.py` L10），语义逐字：`sigmoid(cat([hidden, markov_emb], -1) @ weight / temperature)`。temperature 逐 step 可配（env `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`，标量或 nextn 元组，environ.py L141-148）。输出 $r_{i,j} \in (0,1)$ 是**逐位置独立的接受概率估计**——尚未 cumprod（§3）。

`sample_drafts` L279 逐 step 采样 + 调 confidence head，产出 `MTPPredictInfo(token_ids, confidence_rates)` 经 `mtp_info_pool.store` L339 交给调度器。draft 侧 CUDA Graph 捕获走 `worker/dflash_cuda_graph_runner.py`（`capture_dflash_model` L532）。模型侧 draft 结构（小 MoE 路由、Markov 修正的训练目标、noise token 的语义）归 models 维度（V4.1 待补），本章只走读引擎接口。

> **提点**：Confidence Head 与 EAGLE-2 的 confidence 是同一思路的两个实现面。EAGLE-2 用 draft 概率之积近似路径接受率、剪枝动态 tree（4.5 命题 1）；DSpark 用一个专门的 sigmoid 头预测逐位置接受率、排序选线性前缀子集。两者都在「把 verify 算力花在边际收益最高的 token 上」，差别只在 draft 是树还是链、confidence 是概率积还是学习头。

## 3. cumprod：从逐位置概率到前缀接受概率

Confidence Head 输出的是逐位置概率 $r_{i,j}$，为什么调度器要用它的累乘？因为 verify 接受的是**前缀**：第 $j$ 个 draft token 被接受，当且仅当前 $j$ 个全部被接受（第一个拒绝处停止，4.4 §2 接受规则）。所以「前 $j$ 个全接受」的概率是条件概率链式乘积，引擎用 head 的逐位置输出近似各条件概率，累乘得前缀接受概率。`csrc/engine/scheduler.hpp` `process_MTPInfo` L1196 的累乘循环（L1247-1248）注释逐字：「modeling返回的confidence rate是每个token独立的，使用前要做一个累乘」：

$$
a_{i,j} = \prod_{u=1}^{j} r_{i,u} \qquad (1)
$$

**符号**： $r_{i,u}$ = 第 $i$ 请求第 $u$ 位置的逐位置接受概率估计（Confidence Head 输出）； $a_{i,j}$ = 前 $j$ 个 draft 全被接受的概率（前缀接受概率）。口径：「独立」是代码注释的估计口径——指 head 输出逐位置、尚未累乘，不是断言真实接受事件独立；前缀停止结构本身给出 $P(\text{前 } j \text{ 全接受}) = \prod_{u \le j} P(\text{第 } u \text{ 接受} \mid \text{前 } u-1 \text{ 接受})$，引擎以 head 输出近似各条件项。代入 §4 算例 seq0 的 $r$ = [0.90, 0.85, 0.80, 0.70, 0.60]： $a$ = [0.9, 0.765, 0.612, 0.4284, 0.25704]——逐位递减，因为每个 $r \le 1$。这个单调性是 §4 前缀可行性的全部依据。

`process_MTPInfo` 同时把上一轮的 `accept_rates` 移入 `prev_accept_rates`（L1239-1240：`prev_accept_rates = std::move(accept_rates)` 后 `accept_rates.clear()`），保留邻近两轮的概率供调度器配对使用（§4 的 N−2 / N−1 规则）；`sequence.hpp` L123 注释「累乘后」标明 `accept_rates` 存的是 $a$ 而非 $r$。累乘后的 $a$ 就是调度器排序与期望产出计算消费的唯一切片。

## 4. 变长 verify 的期望吞吐最优（强制重推导）

给定 $B$ 个请求各 5 个 draft、各自的 $a_{i,j}$，怎么选每请求验几个，使期望吞吐最大？路线：先把「期望产出」写成被选前缀概率之和（线性期望），再证固定预算下 top-m 选择最优且自动前缀可行（交换论证 + 单调性），最后对总预算逐前缀扫描 argmax（成本只依赖总量）——关键一步是 cumprod 单调性把「全局 top-m」与「逐请求前缀」两个约束合一。

**期望产出**。每请求每步恒产 1 个 golden token 位（最后确认 token 的前向产出下一 token，即 4.4 的 bonus / 重采样位），加上被接受的 draft。设被选去 verify 的 draft 位置集合为 $S$（前缀封闭），由线性期望：

$$
E[N_{\mathrm{out}}] = B + \sum_{(i,j) \in S} a_{i,j} \qquad (2)
$$

**符号**： $B$ = 请求数（golden 位恒验）； $a_{i,j}$ = 式 (1) 的前缀接受概率。第 $j$ 位「被验且被接受」的指示变量期望恰为 $a_{i,j}$：前缀封闭保证第 $j$ 位被验时前 $j-1$ 位也被验，而接受停在第一个拒绝处，故「第 $j$ 位被接受」等价于「前 $j$ 位全接受」，概率 $a_{i,j}$。代入 §4 算例 $B$ = 2、选 seq0 前 4 + seq1 前 2： $E[N_{\mathrm{out}}] = 2 + (0.9 + 0.765 + 0.612 + 0.4284) + (0.7 + 0.42) = 5.8254$ token/步。

**命题 1**（top-m 前缀可行性）。把全部 $B \times \gamma$ 个 $a_{i,j}$ 全局降序排序（同值按 flat index 升序，即请求序优先、位置靠前优先），取前 $m$ 个。则被选集合逐请求前缀封闭：若 $(i,j)$ 入选，则 $(i,1), \ldots, (i,j-1)$ 全入选。

**证明**（路线：cumprod 单调不增 + 阈值论证，两步）。其一，因 $r_{i,j} \le 1$，式 (1) 给 $a_{i,j+1} = a_{i,j} \cdot r_{i,j+1} \le a_{i,j}$——同一请求内 $a$ 沿位置单调不增。其二，取前 $m$ 等价于设阈值 $\tau$ = 第 $m$ 大的 $a$（此 $\tau$ 是证明内局部阈值记号，与 3.3 步时 $\tau(e)$、wide_ep 图 τ=3.25 无关），保留 $a \ge \tau$ 者（同值按 flat index 升序打破平局）。若 $(i,j)$ 入选（ $a_{i,j} \ge \tau$ ），则对 $u \lt j$ 有 $a_{i,u} \ge a_{i,j} \ge \tau$；同值时 $(i,u)$ 的 flat index 更小、排序更靠前，故 $(i,u)$ 必入选。逐位上推得前缀封闭。证毕。

边界反例（排序对象不可换成 raw confidence）：若按 $r$ 而非 $a$ 排序，前缀封闭破坏。取 seq0 $r$ = [0.30, 0.95]（ $a$ = [0.30, 0.285]）、seq1 $r$ = [0.50, 0.50]（ $a$ = [0.50, 0.25]）， $m$ = 2：按 $r$ 取 top-2 = 0.95（seq0 位 2）与 0.50（seq1 位 1）——seq0 位 2 入选而位 1 落选，不可行（位 2 无法脱离位 1 被验）；按 $a$ 取 top-2 = 0.50（seq1 位 1）与 0.30（seq0 位 1），可行，期望接受 0.80，高于 raw 选择的真实期望 0.785（seq0 位 2 须位 1 先接受，贡献 0.285 而非 0.95）。这正是 `mtp_varlen.hpp` L160 注释「需要保证 confidence_{i+1,...,nextn} 不影响 draft_token_i 是否被验证」的正确性要件：排序必须用 cumprod 后的 $a$，后续位置的概率不能把前面位置挤出预算。对照 4.5 命题 1（EAGLE-2 树版前缀封闭）：那里 confidence 沿树路径单调不增，这里沿链位置单调不增，同一阈值论证。

优化目标写成显式（期望吞吐 = 期望产出 / 步时，口径：每步期望产出 token 数除以该步墙钟）：

$$
\mathrm{throughput}(m) = \frac{B + \sum_{\text{top-}m} a_{i,j}}{\mathrm{cost}(B + m)}, \qquad m^{\ast} = \mathrm{argmax}_{m}\ \mathrm{throughput}(m) \qquad (3)
$$

**符号**： $\sum_{\text{top-}m} a_{i,j}$ = 全局前 $m$ 大的前缀接受概率之和； $\mathrm{cost}$ 见式 (4)； $m^{\ast}$ = 最优额外预算，选定总预算 $Q = B + m^{\ast}$ 即代码的 `best_bsz`。代入算例 $m$ = 6： $\mathrm{throughput}(6) = 5.8254 / 4.66 = 1.2501$ tok/ms，为全表最大（§4 表）。

**命题 2**（期望吞吐最优）。设成本只依赖总 token 数 $Q = B + m$（经 CUDA Graph size 与斜率、sampler，式 (4)），接受为前缀停止，概率用估计值 $a$。则「全局 top-m + 对 $m$ 扫描 argmax 式 (3)」在所有前缀可行的逐请求 draft 长度分配中最大化 $E[N_{\mathrm{out}}] / \mathrm{cost}(Q)$。

**证明**（路线：四步——线性期望、交换论证、可行性、穷举扫描）。第一步，由式 (2)，固定 $S$ 时期望产出 = $B + \sum_S a_{i,j}$，最大化产出等价于最大化 $\sum_S a_{i,j}$。第二步（交换论证），固定预算 $m$（ $|S| = m$ ）：若 $S$ 含某元素 $x$ 而不含更大的 $y$（ $a_y > a_x$ ），把 $x$ 换成 $y$ 使和严格增；故和最大的 $m$ 元集恰为 top-m。第三步（可行性），由命题 1，top-m 自动前缀封闭，是合法分配——无约束最优恰落在可行域内。第四步（比率扫描），成本只依赖 $Q = B + m$，故对每个 $m$ 算 $\mathrm{throughput}(m) = (B + \sum_{\text{top-}m} a) / \mathrm{cost}(B+m)$，取 argmax 即全局最优（穷举所有 $m = 0 \ldots B\gamma$ ）。证毕。口径：最优是**对估计概率 $a$**（N−2 轮，§4 下文）而言，非对真实接受率；成本曲线是 profiled + 在线拟合（式 (4)），非解析。

**成本模型与代码走读**。`csrc/engine/mtp_varlen.hpp` `MTPVarlen::schedule` L150 实现命题 2 的扫描。成本三项（L242-243）：

$$
\mathrm{cost}(Q) = \mathrm{baseline}(g(Q)) + \mathrm{slope} \times g(Q) + \mathrm{sampler} \qquad (4)
$$

**符号**： $g(Q)$ = 不小于 $Q$ 的最小已捕获 CUDA Graph batch size（`_decode_perf_curve` 上 `lower_bound`，L237-241）； $\mathrm{baseline}$ = 启动曲线在该 graph size 的耗时； $\mathrm{slope}$ = 上下文长度修正斜率（`ForwardRuntimeEstimator` L48，两层中位数：先按 graph size 聚合取中位数、再对各 graph 中位数取中位数，窗口 5000、每 graph 至少 10 样本，输入 $(forward\_ms - baseline)/graph\_size$，clamp $\ge 0$ 防负斜率外推）； $\mathrm{sampler}$ = `SamplerRuntimeEstimator` L20 滑窗 9 中位数（至少 5 样本，按 num_seqs 聚合、不足回退全局）。启动曲线来自 `model_runner.profile_decode_perf` L562（(batch_size, forward_ms) 对，`DecodePerfCache` 缓存）；运行时由 `llm_engine.post_process` L984 的 `update_mtp_runtime` L997 用 CUDA event elapsed（forward / sample 各一对 event，L1000-1001）喂入。代入算例 $Q$ = 8： $g(8)$ = 8、baseline 4.2 ms、slope 项 0.02 × 8 = 0.16 ms、sampler 0.3 ms，故 $\mathrm{cost}(8)$ = 4.2 + 0.16 + 0.3 = 4.66 ms（§4 表 $m$ = 6 行）。

扫描本体（L229-258）：`cur_expected` 从 `num_seqs`（golden）起步，`acc_queue_for_bsz` 按 $a$ 降序（同值 seq_idx 升序）逐元素累加，每加一个调 `update_best` 算 `cur_expected / time_ms` 并更新 `best_bsz`——这就是命题 2 第四步的逐前缀 argmax。`best_bsz` 即选定的 $Q$（total_q）。per-request 分配（non-ZOS 路径——ZOS = zero-overhead scheduling 零开销调度，见 §5 / 6.2；L262-266）：`seqlens_q` 全置 1，再对 `acc_queue_for_seqs_q` 的前 `best_bsz − num_seqs` 个逐条 +1——top-m 落到逐请求前缀。

**N−2 / N−1 配对规则**（L161-168 注释 + `schedule_cnt` / `process_cnt` 断言）：best_bsz 永远用 N−2 轮的 $a$ 算，per-request seqlen_q 用 N−1 轮（CPU 可见最新）的 $a$ 算。两个 counter 配对：`schedule_cnt == process_cnt` 时 `accept_rates` 是 N−1 轮、`prev_accept_rates` 是 N−2 轮；`schedule_cnt == process_cnt + 1`（ZOS 在途）时 `accept_rates` 是 N−2 轮、`prev` 是 N−3 轮。为什么预算用更旧的 N−2？因为总预算在 CPU 上定死、之后不可改（它决定 graph size 与 KV 分配），必须用保证已返回的概率；per-request 长度可在 GPU 上修正（§5），故用最新可见概率。`push_rates` 的边界：rates 为空时，有 draft 补 1.0（乐观占位进预算）、无 draft 补 0.0。

**全数字算例**（教学口径， $B$ = 2、nextn 5；成本曲线声明非实测：graph sizes [2, 4, 8, 16]、baseline [3.0, 3.4, 4.2, 5.8] ms、slope 0.02 ms/token、sampler 0.3 ms）。seq0 $r$ = [0.90, 0.85, 0.80, 0.70, 0.60] → $a$ = [0.9, 0.765, 0.612, 0.4284, 0.25704]；seq1 $r$ = [0.70, 0.60, 0.50, 0.40, 0.30] → $a$ = [0.7, 0.42, 0.21, 0.084, 0.0252]。全局降序：0.9(s0j1), 0.765(s0j2), 0.7(s1j1), 0.612(s0j3), 0.4284(s0j4), 0.42(s1j2), 0.25704(s0j5), 0.21(s1j3), 0.084(s1j4), 0.0252(s1j5)。逐 $m$ 扫描（python 复核 2026-11-23）：

| $m$ | $Q$ | $g(Q)$ | $E[N_{\mathrm{out}}]$ | cost (ms) | 期望吞吐 (tok/ms) |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 2 | 2.00000 | 3.34 | 0.5988 |
| 1 | 3 | 4 | 2.90000 | 3.78 | 0.7672 |
| 2 | 4 | 4 | 3.66500 | 3.78 | 0.9696 |
| 3 | 5 | 8 | 4.36500 | 4.66 | 0.9367 |
| 4 | 6 | 8 | 4.97700 | 4.66 | 1.0680 |
| 5 | 7 | 8 | 5.40540 | 4.66 | 1.1600 |
| 6 | 8 | 8 | 5.82540 | 4.66 | **1.2501** |
| 7 | 9 | 16 | 6.08244 | 6.42 | 0.9474 |
| 8 | 10 | 16 | 6.29244 | 6.42 | 0.9801 |
| 9 | 11 | 16 | 6.37644 | 6.42 | 0.9932 |
| 10 | 12 | 16 | 6.40164 | 6.42 | 0.9971 |

argmax 在 $m$ = 6，`best_bsz` = $Q$ = 8。top-6 = seq0 前 4（0.9, 0.765, 0.612, 0.4284）+ seq1 前 2（0.7, 0.42），故 `seqlens_q` = [5, 3]（各含 1 golden），verified = [4, 2]。两个结构事实读表即得：其一，吞吐非单调—— $m$ = 2 到 3 跨过 graph 边界 4→8，成本从 3.78 跳到 4.66，吞吐反降（0.9696→0.9367），所以必须穷举扫描而非贪心加到边际为负；其二， $m$ = 6 到 7 跨 8→16，成本 4.66→6.42，吞吐从峰值 1.2501 跌到 0.9474——最优预算恰停在 graph 边界 8 上。被砍的是 seq0 位 5（0.25704）与 seq1 位 3-5（0.21, 0.084, 0.0252），全是低概率尾部，accepted 几乎不损而成本省一档。

```
DSpark 一步数据流（B 个请求，draft 5）：
  draft 侧（DFlash 3 层 nextn）：
    golden + 4×noise(128799) → 块并行 forward → 5 位置 hidden
    逐 step 采样（Markov rank256 修正）→ draft tokens
    Confidence Head sigmoid → r[i,1..5]（逐位置独立概率）
  CPU schedule(N+1)（ZOS 下 confidence(N) 未回）：
    cumprod r → a（N−2 轮）→ 全局降序 → 逐前缀扫描 E[out]/cost → best_bsz = total_q
  GPU 修正（ZOS，§5）：
    remap gather confidence(N) → cumprod → rank counting top-m → seqlens_q = 1 + wins
    → D2H pinned buffer（event 定序）
  verify（target 主干，prefill 形态）：
    每请求打 seqlens_q[i] 个 token → 接受前缀 → 产出 N_out
```

## 5. ZOS：总预算在 CPU，per-request 长度在 GPU

async scheduling 下 `schedule(N+1)` 跑在 CPU 上时，step N 的 confidence 还没返回——per-request 长度怎么定？recap 6.2 的数据依赖分类：CPU 调度只消费数据无关量，数据有关量在 GPU 流上解析。DeepInfer 的 async pipeline 官方 doc 逐字（doc L334，教学参照 + 官方披露）：「while the GPU runs forward(N) → sample(N), the CPU concurrently runs schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1), leaving the GPU with no idle time throughout」（仿 SGLang / NanoFlow，6.2 §10 同源对照）。DSpark 与它的交互 doc L324 逐字：「When schedule(N+1) runs, the confidence from step N has not been returned yet, so CPU scheduling only fixes the total budget; the per-request lengths are recomputed on the GPU by a TileLang kernel (cumulative product, global sort for top-k, with a fused remap scatter), and the result is written back to a pinned buffer.」

分工落到代码：CPU 用 N−2 轮概率定 `best_bsz` = total_q（§4，不可改）；per-request 长度由 GPU TileLang kernel 用 step N 的实际 confidence（已在 GPU 上、尚未回 CPU）重算。入口 `worker/worker.py` `_correct_seqlens_qk` L191（docstring：「CPU 侧的 seqlens_q/k 在 ZOS 下滞后一步：与上一步 batch 有交集时，需按上一步采样结果在 GPU 上修正」），L219 调 `compute_seqlens_q_by_confidence_rates`（`tilelang_ops/schedule_mtp_varlen.py` L11），消费 `prev.sampler_output.confidence_rates_gpu`。

**TileLang kernel 两阶段**（`tilelang_ops/cuda/schedule_mtp_varlen.py` L14，`@tilelang.jit` + 两个 `T.Kernel`；TileLang DSL recap 3.4 §5 + 7.3 指引，[Wang et al., 2025](../references.md)，arXiv:2504.17577，[A]）：

- **Phase 1（cumprod via remap）**：每请求一个 block（threads 128），读 `remap[pid]`——当前批请求到上一批索引的映射，fresh seq 为 −1。remap = −1 时 acc 置 0（fresh seq 无上一轮 confidence，cumprod 全 0）；否则沿 nextn 累乘 `prev_confidence_rates[prev_idx, j]` 写 `cum_rates[pid, j]`。这就是 doc 说的「fused remap scatter」：remap gather 融进 cumprod kernel，不单独物化 gather 结果。
- **Phase 2（parallel rank counting）**：每请求 block 读自己的 nextn 个 cumprod 值，对全局 `num_seqs × nextn` 个元素并行数「击败自己」的个数——`other_rate > my_rate`，或同值且 `other_idx < my_flat_idx`（flat index 升序打破平局）——累加进 `rank_reducer`。rank 即全局降序排序中的位置（等价 global sort top-k，但不物化排序）。`extra_budget = total_q − num_seqs`，数自己有几个位置 `rank < extra_budget` 得 wins，`seqlens_q[pid] = 1 + wins`。

rank counting 与 stable sort top-k 的等价（证明要点，三步）：其一，定义 $\mathrm{rank}(i) = \lvert \lbrace j : a_j > a_i \text{ 或 } (a_j = a_i \text{ 且 } j \lt i) \rbrace \rvert$，则 rank 是 $0 \ldots n-1$ 的一个排列（每个元素被击败次数互异，平局由 index 全序打破）；其二，降序 stable 排序后第 $i$ 个元素的位置恰为 $\mathrm{rank}(i)$（排在它前面的正是击败它的元素）；其三，故 $\mathrm{rank}(i) \lt m$ 当且仅当 $i$ 在 top-m 内。参考实现（`schedule_mtp_varlen.py` L35-98）用 `torch.argsort(descending=True, stable=True)` 显式排序再数 rank，pytest L139 `assert torch.equal(seqlens_q, seqlens_q_ref)` 断言两实现逐元素相等（参数化覆盖 num_seqs 1-256、nextn 1-7、total_q 各档）。per-request 长度公式：

$$
q_i = 1 + \lvert \lbrace j \in \text{seq } i : \mathrm{rank}(j) \lt m \rbrace \rvert, \qquad m = Q - B \qquad (5)
$$

**符号**： $q_i$ = 第 $i$ 请求的 seqlens_q； $m$ = extra_budget。因 rank 是排列，恰有 $m$ 个位置满足 $\mathrm{rank} \lt m$，故 $\sum_i q_i = B + m = Q$ 恒成立——GPU 重算只在固定总预算内重分配，不改预算。代入 §4 算例 $Q$ = 8、 $B$ = 2、 $m$ = 6：top-6 落 seq0 四位 + seq1 两位， $q$ = [5, 3]，与 CPU non-ZOS 路径一致。fresh seq（remap = −1）cumprod 全 0、rank 垫底，通常只拿 golden 位（ $q_i = 1$ ）；仅当预算超过全部非零位置时才分到额外 token。

**seqlens_k 修正与 pinned 回写**。`compute_seqlens_k_zos`（`tilelang_ops/compute_seqlens_k_zos.py` L10）用实际接受数修 KV 长度：`seqlens_k[i] += seqlens_q[i]`，再对 remap 命中的请求减 `nextn − prev_num_accepted_tokens[j]`——上一轮 draft 打了 nextn 个但只接受 `num_accepted` 个，KV 长度须按真实接受数回退，不能按 draft 数。回写路径（worker.py L228-234）：H2D、GPU 修正、D2H 在当前流上定序，`seqlens_q_cpu[:n_seq].copy_(seqlens_q[:n_seq], non_blocking=True)` 拷入 pinned host buffer，record `d2h_evt`；消费端 L148-149 `correction.d2h_evt.synchronize()` 后才读 pinned 源（`_SeqlensCorrection` L39 注释：「发生 D2H 时，d2h_evt 完成后方可读取」）。CPU 调度线程不被 D2H 拖住——event 把同步推迟到真正读长度那一刻。

```
ZOS 时序（DSpark per-request 长度 GPU 重算）：
拍:        N−1                  N                     N+1
CPU:  sched(N)→launch fwd(N)→post(N−1)→sched(N+1)→launch fwd(N+1)→post(N)
GPU:        [forward(N−1)+sample(N−1)]  [forward(N)+sample(N)]  [forward(N+1)+sample(N+1)]
sched(N+1) 时：confidence(N) 还在 GPU → CPU 只定 total_q（N−2 轮 a）
              GPU kernel 用 confidence(N) 重算 seqlens_q → pinned D2H（d2h_evt）
              消费端读 pinned 前 synchronize(d2h_evt)
```

> **坑**：把 ZOS 下 CPU 算出的 seqlens_q 当最终值。触发条件：在 `_correct_seqlens_qk` 之前读 CPU metadata 的 per-request 长度做容量或正确性判断；现象：CPU 值是滞后一拍（N−1 / N−2 轮概率）的占位，GPU kernel 会用 step N 的实际 confidence 重写 pinned 源，两者可不同。规避：per-request 长度以 GPU 修正后的 pinned buffer 为准（event 同步后读）；只有 total_q 是 CPU 定死、GPU 不改的量。

## 6. 生产增益表：+4–7% 与 conc16 的 −1.5%（官方披露）

变长 verify 在生产上赚多少？DeepInfer 官方 doc L307-316 给全表（Open-PerfectBlend 数据集，EP64/DP64 benchmark，教学参照 + 官方披露，核实 2026-11-23）。doc L307 逐字：「dynamic-length verification cuts the number of draft tokens verified per step by about 30% (5.00 → 3.2–4.4) while leaving the average accepted length essentially unchanged (3.29 vs. 3.20). Computed from EP64/DP64 benchmark data, that works out to +4–7% throughput at the production operating point (~60 concurrent sequences per rank).」全表六行（doc L309-316，逐字）：

| conc (seqs/rank) | fixed verified | dynamic verified | fixed accepted | dynamic accepted | fixed tok/s/rank | dynamic tok/s/rank | gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 5 | 4.44 | 3.32 | 3.24 | 4891 | 4816 | −1.5% |
| 32 | 5 | 3.65 | 3.25 | 3.17 | 8964 | 9111 | +1.6% |
| 48 | 5 | 3.94 | 3.3 | 3.26 | 12032 | 12467 | +3.6% |
| 60 | 5 | 3.58 | 3.3 | 3.21 | 13730 | 14466 | +5.4% |
| 72 | 5 | 3.24 | 3.29 | 3.12 | 14962 | 15961 | +6.7% |
| 96 | 5 | 3.24 | 3.3 | 3.15 | 16817 | 18660 | +11.0% |

口径对齐（读表前必须钉死）：verified = 每步实际送验的 draft 数（= seqlens_q − 1，fixed 恒 5）；accepted = 被接受 draft 数 $E[K]$（4.4 §3 口径注，**不含** bonus 位，勿与 $E[N_{\mathrm{tok}}]$ 混）；「3.29 vs 3.20」是**表均值**口径——fixed accepted 均值 3.293、dynamic 均值 3.192（doc 取 3.20），非单行值（python 复核：六行均值 3.2933 / 3.1917）。gain 列由 tok/s 列复算全中（4816/4891−1 = −1.53%、14466/13730−1 = +5.36%、18660/16817−1 = +10.96%，doc 取一位小数）。dynamic verified 六行均值 3.68（较 5 降 26.4%），doc「about 30%」是其口径概述，表内逐行 3.24-4.44。

**conc16 负增益必须给**：dynamic 非普适赢。机制解读（本书口径，非 doc 断言）：低并发下 decode 步偏 memory-bound（4.4 §1），verify 计算藏在带宽下界里，砍 verified 几乎不省墙钟；而 accepted 从 3.32 跌到 3.24（−2.4%），产出实打实少了，净效果 −1.5%。随并发升高，verify 进入 compute-bound（4.4 §6），砍低概率尾部省的墙钟超过 accepted 损失，增益从 +1.6%（32）一路涨到 +11.0%（96）。另一个读表事实：dynamic verified 本身随并发递减（conc16 4.44 → conc96 3.24）——低并发成本曲线平，argmax 留更多 draft；高并发成本随 $Q$ 陡涨，argmax 砍得更狠。这正是命题 2 的预算扫描随成本曲线形状自适应的体现，而非固定砍 30%。生产操作点 ~60 并发/rank 落在 +5.4%（60 档）/ +6.7%（72 档），doc 概述 +4–7%。doc L320 另注：internal draft 略新、差异小（「The internal one is somewhat newer, but the performance difference is small」）。

> **提点**：这张表是「投机解码收益依赖负载操作点」的生产实证。4.4 §4 的加速比定理假设 verify 不增墙钟（memory-bound），单请求成立；batch compute-bound 区须按真实 verify 成本重算（4.4 §6）。DSpark 的变长预算把「验多少」从固定参数变成逐操作点最优决策——低并发少砍（甚至不砍，conc16 仍验 4.44）、高并发多砍，增益曲线随之从负转正。读任何投机加速数字前先问操作点并发与 bound 区。

## 7. SWA bounded replay：恒定成本重建（强制重推导）

窗口态 miss 后重算多少 token 才够？先钉 CED 架构的 serving 含义，再给 replay 机制与恒定成本命题。模型侧形式化归 models 维度（V4.1 待补），本节只引 config [W] 与 doc 官方披露口径。

**CED serving 含义**。V4.1-Flash 40 层分两侧：encoder（prefill）20 层 + YOCO consumer（decoder）20 层。引擎读法 `hugging_face/config.py` `_n_prefill_layer` L158 = `compress_ratios` 首个 ratio==1 的索引 = 20（config `compress_ratios` 43 项 = [0, 0, 2×18, 1×20, 0, 0, 0]，前 20 层 encoder、后 20 层 consumer、末 3 项 nextn [W]；`_build_attn_module_list` L146 断言 ratios 长度 = n_layer + nextn）。decoder 层 block KV 的来源，doc L241 逐字（官方披露）：「In the CED architecture the decoder layers' block KV cache is projected from the last encoder layer's hidden state」——decoder block KV 由末 encoder 层 hidden 投影而来，故 prefill 的 block KV 计算只发生在 encoder 层，decoder 层每层只需算自己的 SWA KV。config `kv_source_layer_ids` = [2, 8, 14, 20] 原样列出 [W]；引擎 `_source_map` L62 把每层映射到不大于它的最近源层（`layer_kv_map`）。

> **坑**：拿 doc「末 encoder 层 hidden 投影」与 config `kv_source_layer_ids` = [2, 8, 14, 20] 做模型侧调和。触发条件：试图从四个源层 id 反推「末层」到底是哪层、或推导投影的模型侧语义；现象：两源表述存在张力（doc 单数「last encoder layer」vs config 四个源层 id），强行调和会越界做模型侧推导。规避：维持 advisory（W0 B-9 裁定）——doc 口径标官方披露、config 字段原样 [W]，两源并列不调和；形式化归 models（V4.1 待补）。引擎实现口径可引 `v4_modules/yoco.py` `yoco_shared_hidden_required` L24 docstring：consumer SWA 的 `kv_swa` 由 shared hidden（首个 consumer 层的 attn-norm hidden）投影——这是引擎读法，非模型侧推导。

**replay 机制**。doc L237 逐字（官方披露）：「SWA bounded replay recomputes only the most recent 128 tokens and truncates the attention window to within the replayed segment: replaying from position s, the query at position i is computed only against keys in the range max(s, i−127) to i, which yields an approximate SWA KV cache. The cost of recomputation is thereby reduced to a constant, independent of context length and layer count.」即重算段从位置 $s$ 起，query $i$ 的可见 key 截断为：

$$
\mathrm{keys}(i) = [\max(s, i - 127),\ i] \qquad (6)
$$

**符号**： $s$ = 重算起点；窗口 $w$ = 128（ $i - 127$ 到 $i$ 共 128 个位置）。截断把可见域限制在重算段内（ $\max(s, \cdot)$ 保证不读段外 token），产出的是**近似** SWA KV（doc 口径「approximate」）——段首位置的窗被截断，近似是 bounded replay 声明过的代价，非精确重建。代入 $s$ = 9856：query $i$ = 9999 的可见 key = $[\max(9856, 9872), 9999] = [9872, 9999]$（完整 128 窗）；段首 query $i$ = 9856 的可见 key = $[9856, 9856]$（窗被截到 1 个位置，近似由此生）。

**命题 3**（bounded replay 恒定成本）。bounded replay 所需的输入 token 数恒为 $w$ = 128，与上下文长 $T$ 和模型层数 $L$ 无关；每层注意力成本至多 $w^2$ 对 query-key，与 $T$ 无关。

**证明**（路线：对层归纳证重算段闭合，关键一步是截断窗使每层只消费段内 hidden；三段式）。基例（第 1 层）：重算段输入 = 末 $w$ 个 token 的 embedding + 复用的 block KV / 投影 hidden（CED 下 decoder block KV 来自 encoder，已算好），共 $w$ 个 token 的段内数据。保持：设第 $\ell - 1$ 层在段内 $[s, s+w)$ 各位置的 hidden 已算出且只依赖段内输入。第 $\ell$ 层位置 $i \in [s, s+w)$ 的 SWA 注意力可见域由式 (6) 为 $[\max(s, i-127), i] \subseteq [s, s+w)$——全在段内；故第 $\ell$ 层在 $i$ 的输出只依赖第 $\ell - 1$ 层段内 hidden（与该层复用的 block KV），不消费段外 token。段在截断前向下闭合。终止：逐层成立， $L$ 层都只消费 $w$ 个 token 的段——所需输入 token 数 = $w$，与 $T$、 $L$ 无关；每层至多 $w$ 个 query 各 attend 至多 $w$ 个 key，即 $w^2$ 对，与 $T$ 无关。证毕。

口径注（「恒定」的精确含义）：「与层数无关」指**所需输入 token 数**不随层数涨（对比精确重建的依赖锥，下文），非指总 FLOPs 与层数无关——总 replay 计算仍随重算层数线性（ $L_{\mathrm{dec}} \times O(w^2 + w \cdot d_{\mathrm{model}})$ ），这与任何前向的层数口径相同。恒定的是段长 $w$ 与每层成本对 $T$ 的独立性。代入： $w^2 = 16384$ 对/层，20 层共 327,680 对——而全上下文 SWA prefill 每层是 $T \times w$ 对（ $T$ = 32768 时 4.19M 对/层），replay 把每层注意力从 $O(Tw)$ 压到 $O(w^2)$。

**对照精确重建的依赖锥**。不截断窗时，第 $\ell$ 层位置 $i$ 的 SWA 窗需要第 $\ell - 1$ 层在 $[i-127, i]$ 的 hidden，后者又需第 $\ell - 2$ 层更宽范围——依赖锥逐层向过去展宽，每层至多展宽一个窗口。20 层累计上界 $20 \times 128 = 2560$ 个 token，恰为 doc L241 逐字的精确重建数（官方披露）：「reconstructing it exactly would require running the last 2,560 tokens through 20 decoder layers」。2,560 = 20 × 128 是算术观察（层数 × 窗口 = 依赖锥上界）；锥的精确模型侧依赖结构归 models，本章只给 serving 侧上界。bounded replay 用式 (6) 的截断把锥砍断：每层闭合在 128 段内，输入 token 数从「随层数涨到 2,560」钉死为 128。

**两处使用**（doc L239 / L241）。encoder replay（L239 逐字）：「the prefix cache hit length now depends only on the block KV cache. When the SWA KV cache is missing, the last 128 tokens of the hit prefix are recomputed and run forward together with the non-cached suffix; the replayed segment only rebuilds the SWA KV cache, while the block KV cache is reused as-is, neither recomputed nor overwritten.」——命中长只由 block KV 定；SWA miss 时重算命中前缀末 128 + 非缓存 suffix，只重建 SWA，block KV 原样复用（不重算不覆盖）。decoder replay（L241）：prefill 计算只发生在 encoder 层（block KV 侧），decoder 层每 prefill 只重算 prompt 末 128 token，结果**只供 decode、永不入 prefix cache**——「nearly halving total prefill computation」。

**近乎减半账**。设上下文 $T$，encoder $L_{\mathrm{enc}}$ = 20 层全量 prefill、decoder $L_{\mathrm{dec}}$ = 20 层只 replay $w$ = 128。prefill 线性项（FFN / 投影，每 token 每层近似常数）计算比：

$$
\frac{L_{\mathrm{enc}} \cdot T + L_{\mathrm{dec}} \cdot w}{L \cdot T} \to \frac{L_{\mathrm{enc}}}{L} = \frac{1}{2} \quad (T \gg w) \qquad (7)
$$

代入 $T$ = 32768： $(20 \times 32768 + 20 \times 128) / (40 \times 32768) = 657920 / 1310720 = 0.50195$，省 49.8%（python 复核）； $T$ = 4096 时 0.5156（省 48.4%）， $T$ = 1048576 时 0.50006。SWA 注意力项同向：decoder 每层从 $T \times w$ 对压到 $w^2$ 对，比 $\to w/T \to 0$。两项叠加，总 prefill 近乎减半。口径：账取 doc replay 口径 $w$ = 128；实际保留后缀 KEEP_TOKENS = 140（§8），代入 140 得 0.50214，结论不变。「只供 decode 不入 prefix cache」的结构原因：128 段短于 20 层依赖锥（2,560），整段都在截断近似区内，近似态不能当精确前缀复用——§8 的 hit cap 保证每请求重算自己的尾部，近似不经缓存跨请求传播。

**YOCO 截断实现**（decoder replay 的引擎落点）。env `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`（environ.py L195-197）三值：off / mask / compact。语义逐字：mask「保留全部 query，仅收窄 SWA 可见域」；compact「真正移除截断区 query，以减少 consumer 层计算」——生产用 compact（README L25），mask 不省计算只收窄可见域（等价性验证 / 消融用）。`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` = 140（V4.1 默认）是 consumer 层保留的末尾 query 数。代码件：`v4_modules/yoco.py` `yoco_enabled` L14（n_prefill_layer < n_layer）、`is_yoco_consumer_layer` L19、`yoco_boundary_compact` L55 / `compact_yoco_boundary` L93（边界数据重排）；`input_metadata/yoco.py` `YocoMetadataMixin` L32（compaction / mask / truncation enabled + consumer view）；`examples/yoco/yoco_equiv.py` 数值等价验证。YOCO 模型侧源头（[Sun et al., 2024](../references.md)，arXiv:2405.05254，[A]）只指引不展开。

> **坑**：把 mask 模式当 compact 用去省计算。触发条件：设 TRUNCATION_MODE=mask 期望减半 prefill；现象：mask 保留全部 query，consumer 层计算量不降，只收窄 SWA 可见域（数值等价但无算力收益）；规避：省计算必须 compact（真移除截断区 query）；mask 只用于等价性对照。KEEP_TOKENS 须 ≥ window + MAX_BACKTRACKING = 134（§8），140 是配置余量。

## 8. hit policy 正确性：L−134 cap 与回退一块（强制重推导）

命中边界怎么定，才能保证窗口无洞、回溯不越界、block KV 复用精确？doc L245 逐字（官方披露）：「the hit length is determined by the block KV cache, and whether the last block needs recomputation is determined by the SWA KV cache — if the SWA KV cache misses, back off by one block and recompute. Concretely, under the YOCO truncation requirement, the maximum hit length for an input of length L is at most L − 134 (128-token window + 6 tokens of speculative backtracking). Under that cap, the block KV cache is queried first, giving a hit length L1 (a multiple of the 128-token block size), and then the SWA KV cache is queried. If an SWA KV cache exists in the range L1 − 128 to L1, the hit length is L1; otherwise it is L1 − 128.」翻译为本章记号（doc L → $T$、L1 → $H_1$）：

$$
H \le T - 134, \qquad H = \begin{cases} H_1 & \text{SWA 在 } [H_1 - 128, H_1] \text{ 存在} \\ H_1 - 128 & \text{否则（回退一块）} \end{cases} \qquad (8)
$$

**符号**： $H$ = 命中长 = 重算起点边界（SWA 态边界）； $H_1$ = block KV 命中长（128 的倍数， $\le T - 134$ ）；134 = $w + \delta$ = 128 窗 + 6 回溯。**block KV 复用边界恒为 $H_1$**——replay 只写 SWA、不碰 block KV（doc L239），SWA miss 时 $H = H_1 - 128$ 与 $H_1$ 差的 128 正是 replay 段（重算命中前缀末一块重建 SWA，block KV 仍复用到 $H_1$）。代入 $T$ = 10000：cap = 10000 − 134 = 9866，block 命中 $H_1$ = 9856（77 × 128，≤ cap）；SWA hit 则 $H$ = 9856、miss 则 $H$ = 9728（完整算例见下文）。

**命题 4**（hit policy 正确性）。设输入长 $T$，policy 按式 (8) 定 $H$。则：(1) 重算 suffix 长 $T - H \ge 134$；(2) decode 起点及最大回溯 $\delta$ = 6 内任意位置的 SWA 窗完全落在「存储窗或重算段」内，无洞；(3) block KV 复用与存储态逐字一致（与全量重算逐位相等另需 batch-invariant 前提，见证明）。

**证明**（路线：算术 + 窗覆盖 + 前缀不可变，三步）。(1) case A（SWA hit） $H = H_1 \le T - 134$，故 $T - H \ge 134$；case B（miss） $H = H_1 - 128$， $T - H \ge 134 + 128 = 262$。两 case 均 $\ge 134$。(2) 位置 $p$ 的 SWA 窗为 $[p - 127, p]$（128 个位置）。decode 起点 $p = T - 1$，窗起点 $T - 128$；最大回溯 $\delta$ 后最早考察位置 $p = T - 1 - \delta$，窗起点 $T - 128 - \delta$。窗无洞要求窗起点 $\ge H$（段内或存储窗覆盖），即 $T - H \ge 128 + \delta = 134$——由 (1) 成立。case A 另需首 suffix query（位置 $H_1$）的窗 $[H_1 - 127, H_1]$ 被存储 SWA 覆盖：存储窗 $[H_1 - 128, H_1] \supseteq [H_1 - 127, H_1]$，成立；case B 从 $H_1 - 128$ 重算，SWA 在 $[H_1 - 128, T)$ 重建（近似口径 §7 已声明）。(3) block KV 只依赖前缀（前缀不可变命题，models 2.4 / 2.2 recap），且 replay 不重算不覆盖 block KV（doc L239），故复用 block KV 与**存储态逐字一致**——这是无条件的（字节原样复用，不经任何重算）。若要与「全量重算」的结果逐位相等（bit-exact），则另需 batch-invariant kernel 前提（README L27 / 3.4 §6：跨 batch 形状固定 reduction 顺序才保逐位相等）；数学相等（同一前缀函数值）则由前缀不可变保证，不依赖该前提。证毕。

**134 = 128 + 6 的代码级**。 $\delta$ = `DEEPINFER_MAX_BACKTRACKING_TOKENS` = `max(v, USE_NEXTN_DECODE + 1)` = 5 + 1 = 6（environ.py L139；nextn 5 + 1 覆盖拒绝位重采样 / bonus）。inference SWA 窗 `get_inference_swa_block_size` L88 = window + MAX_BACKTRACKING = 128 + 6 = 134。KEEP_TOKENS 约束双重 enforce：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp` L355-357（`TORCH_CHECK_GE(keep_tokens, window_size + max_backtracking_tokens)`）+ `worker/model_runner.py` L108-112（同式 assert）。README L25 默认 140 是配置保留后缀（覆盖 134 有余量），与 doc 134 不矛盾——140 ≥ 134，多出的 6 是余量。

**全数字算例**（ $T$ = 10000）。cap = 10000 − 134 = 9866；block 命中 $H_1$ = 不超过 cap 的最大 128 倍数 = 9856（77 × 128）。case A（SWA hit）： $H$ = 9856，suffix = 144 ≥ 134；decode 起点 9999 窗 [9872, 9999]，9872 ≥ 9856 ✓；回溯 6 到位置 9993，窗 [9866, 9993]，9866 ≥ 9856 ✓（余量 10）。case B（miss）： $H$ = 9728，suffix = 272，覆盖更宽。cap 违反反例：若 $H$ = 9900（suffix 100 < 134），回溯位置 9993 的窗起点 9866 < 9900，缺 34 个 key（9866-9899）——窗洞，decode 注意力读到不完整窗口。这正是 cap 取 134 而非 128 的原因：多出的 6 给回溯留窗。

> **坑**：StateBlock 里两个「6」混为一谈。触发条件：把 doc L219 的 ring +6 与 doc L245 的回溯 6 当同一量；现象：两数同值不同义——ring +6 是 **slot 几何**（窗口外多 6 个位置，吸收 decode copy overshoot ≤5，§9），回溯 6 是**投机回溯深度**（= nextn 5 + 1，定 hit cap 与 inference 窗 134）。规避：分别声明——inference 窗 134 = 128 + 回溯 6；slot ring 另在窗口外 +6 吸收 copy 越界。两者都叫 6，语义无关。

## 9. StateBlock：slot 布局与落盘账

每序列的窗口态占多少字节、哪些落盘？recap 2.4 §8 的字节账（20 × 134 × 528 = 1,415,040 B ≈ 1.35 MiB），本节增量是代码级布局与 stored / dropped 分区。StateBlock（`worker/cache_engine.py` L221，release 内可走读）docstring 逐字同 doc L217：「分区原则：需要落盘的 state 全部放在前面的 stored region，不需要落盘但运行时仍需保留的 state 放在后面的 dropped region。PE→CE 传输两部分；to disk 只取 inference 布局的前 state_size_to_disk 字节，按长度截断即可排除整个 dropped region。」布局（低地址到高地址）：

```
StateBlock slot（每序列一个固定 slot）：
┌──────────────────────────────┬───────────────────────────────────────────┐
│ stored region                │ dropped region                            │
│ SWA（前 20 层，stored layers）│ YOCO consumer SWA（后 20 层）              │
│ 每层 134 token × 528 B       │ + ca_state / ica_state（C2 压缩边界累加器）│
│ = 70,752 B/层                │ fp32 [num_state_seqs, ring, 2, dim]       │
│ state_size_to_disk           │ 不落盘：恢复靠 prefill 重算 /              │
│ = 20 × 70,752 = 1,415,040 B  │ 压缩窗 128 对齐无跨边界依赖                │
└──────────────────────────────┴───────────────────────────────────────────┘
ring：inference 窗 134 = 128 + 6（回溯）；窗口外再 +6 吸收 copy overshoot ≤5
落盘只取前 state_size_to_disk 字节（长度截断排除 dropped region）
```

**stored / dropped 分区代码**。stored layers = 前 20（doc L217「For V4.1, the stored layers are the first 20」）；`_is_dropped_swa_layer` L113 判 `layer_id >= n_prefill_layer`（= 20）为 dropped——consumer SWA 恢复依赖 prefill 重算（§7 decoder replay），不落盘。`swa_store_num_layer` 数非 dropped 的 SWA 层 = 20，`swa_drop_num_layer` = 总 SWA 层 − 20。dropped region 另含 C2 压缩边界累加器 `ca_state` / `ica_state`（fp32 `[num_state_seqs, ring_size, 2, dim]`，KVCache docstring L206-219）：压缩窗边界天然对齐 128 落盘粒度，无跨边界依赖（doc L217），故不需落盘。压缩 ring `get_inference_compress_block_size` L131 = ratio + max(0, backtracking − 1) = 2 + 5 = 7（ratio 2 的 C2 层）；每 owning 层 ca_state 字节 = 7 × 2 × 512 × 4 = 28,672 B（fp32，compress_dim = 512），ica_state = 7 × 2 × 128 × 4 = 7,168 B（index_head_dim 128）。

**528 B/token 构成**。SWA KV 格式 = MXFP8（doc L223「The 528 B/token is the MXFP8 SWA KV format」）。KVCache docstring L216-218 逐字：`swa_cache [num_state_seqs, swa_block_size, 1, 528] uint8`，「Each V41 token stores [512B FP8 data | 16B UE8M0 scale]」。格式常量 `V41_FORMAT`（`tilelang_ops/cache_common.py` L109）：e4m3 元素 512 个 + per-32 ue8m0 scale，`bytes_per_token` = 512 × 1 + (512/32) × 1 = 528：

$$
528 = 512 \times 1 + \frac{512}{32} \times 1 \qquad (9)
$$

元素数口径（引擎读法，一手权威）：`kv_lora_rank = head_dim − qk_rope_head_dim` = 512 − 64 = 448（`hugging_face/config.py` L289，L288 注释逐字「HF's shared KV vector includes RoPE; the internal rank excludes it.」），加 rope 64 共 512 个元素（`V41_FORMAT.quantized_elements` = 512，含 64 RoPE 维全量化、无 bf16 部分）。此 512 元素读法由四方一手互证锁定：config.py L289 引擎导出 + `V41_FORMAT` L109（quantized_elements 512）+ FlashMLA main README（V4.1 `head_dim` 512，528 = 前 512 B e4m3「covering all 512 dimensions」、64 RoPE 维也量化，[DeepSeek-AI, FlashMLA](../references.md)，教学参照）+ doc L223 官方 528 B/token。**errata 前 1.2 / 2.3 曾写 V4.1「576 元素 / 主 KV 324 B」，系事实错误**（把 config `head_dim` 512 误当 kv_lora_rank 再加 rope 64，重复计 rope，V3 几何误套 V4.1），非合法替代口径——W4 审查裁定（X-1，README 审查记录区）走跨波数值 errata 修正为 512 元素 / 288 B，本章引擎读法为准。kernel 栈侧格式细节（528 / 288 B 的 kernel 消费与双平台）见 7.3。MXFP8 per-32 格式账 recap 2.3 §4 式 (4)，不重推。同格式族：主 KV FP4（`V41_FP4_FORMAT`）= 512/2 + 512/16 = 288 B（e2m1 + per-16 e4m3 scale），indexer FP4 = 128/2 + 128/32 = 68 B（`_get_qk_topk_bytes` L73）。

**落盘字节**。`state_size_to_disk = swa_store_total_bytes_inference_pad` L306 = stored 层数 × 每层 inference 窗字节（32B 对齐）：

$$
\mathrm{state\_size\_to\_disk} = 20 \times (134 \times 528) = 20 \times 70752 = 1415040 \ \mathrm{bytes} \qquad (10)
$$

每层 134 × 528 = 70,752 B，70,752 = 32 × 2,211 整除（32B 对齐无 pad，python 复核）；20 层 = 1,415,040 B ≈ 1.35 MiB（/ $2^{20}$ = 1.3495）。**仅 stored region 落盘口径**（W0 B-6）：1,415,040 是 `state_size_to_disk`，非 slot 总尺寸——slot 另含 dropped region（consumer SWA 20 层 + 压缩累加器）与 ring 余量，总尺寸 `total_bytes_inference` > `state_size_to_disk`（含 SWA 时整 block 按 lcm(32, 528) = 1056 B 对齐）。落盘只取前 `state_size_to_disk` 字节（`select_state_slice_to_store` L763，to disk 取 `state_size_to_disk`、PE→CE 取 `total_bytes_inference`，L790/800；`load_state_from_store_to_prefill_cache` L821 回载）。

> **坑**：把 1,415,040 B 标成 StateBlock slot 总尺寸。触发条件：引 doc L223 数字时省略「仅 stored region」限定；现象：slot 实际含 dropped region（consumer SWA + ca_state/ica_state）与 ring +6 余量，总尺寸更大，误标会低估显存 / 高估落盘占比。规避：1,415,040 = `state_size_to_disk` = 仅 stored region（20 层 × 134 × 528）落盘；slot 总尺寸另计（`total_bytes_inference`）。

**128 粒度落盘规则**（doc L219 官方披露）。SWA 只在 128 倍数位置落盘：prefill 末留 1 token 向下取整到 128 倍数，不足一个 128 单元本轮不落；chunked prefill 中途不落、只末 chunk 落。decode 跨 128 边界时把当前窗 copy 进 device 上的 replica region，finish 取最近完成 copy 向下取整到 128 边界；copy 位置至多越界 5 token，由 slot ring 窗口外 +6 吸收（§8 坑注的 ring 6）。kvstore ring 可大于 inference 窗：`get_kvstore_swa_block_size` L101 = inference 134 + 加项，加项 = (PREFILL_STORE_EXTRA_SIZE 128 + yoco extra) **当且仅当 `DEEPINFER_PREFILL_STORE_EXTRA_STATE` 开启**（environ.py L170 `EnvVar.bool()` 默认 False，L77 bool 默认），否则加项为 0；yoco extra = KEEP_TOKENS（截断开启时，`get_yoco_store_extra_tokens`）。docstring 逐字：「因为 kvstore 落盘只在 128（落盘粒度）的整倍数落，prefill 时可能不会触发落盘，多存一段才能保证多轮命中。」故 extra-state 开启且 compact 140 时 kvstore ring = 134 + 128 + 140 = 402 token/层；默认关闭时加项为 0、ring = inference 134。slot 数 `get_num_state_slots` L201 = int(max_num_seqs × 1.2) + 1（`NUM_STATE_SLOTS_RATIO` 1.2，max_num_seqs 64 → 77）。

**放置与传输**（recap 2.4 §1 / §8，doc 官方披露）。SWA KV 不进 persistent cache、留生产机内存（doc L223）；Entrypoint 内存目录（逻辑统一、物理按 key 分区）记录持有机，跨机 hit 走 400G VLAN pull-back（doc L231，L168 三层表 SWA 行：local host memory / single copy / nothing on 3FS / 400G VLAN）。SWA 与 block KV 共享 P-to-D 传输路径（doc L227）。release 边界（7.1 recap）：StateBlock / cache_engine 引擎侧在 release 可走读；Entrypoint 目录与跨机 fetch 编排不在 repo，只讲 doc 披露架构、不推测代码。

## 总结

| 论断 | 回指 |
| --- | --- |
| DSpark 引擎实现名 DFlash（config dspark_* ↔ dflash_cfg）；draft = 128 expert top-3 小 MoE + rank256 Markov + 三层特征融合 + 块并行 noise draft | §2、config 表、坑注 |
| Confidence Head = sigmoid(cat[hidden, markov_emb] @ w / T)，输出逐位置独立接受概率 r；cumprod 得前缀接受概率 a（前缀停止结构） | 式 (1)、§3、process_MTPInfo L1196 |
| 全局 top-m（按 a 降序）自动前缀封闭（cumprod 单调）；按 raw r 排序破坏前缀（反例） | 命题 1、§4、4.5 命题 1 对照 |
| 期望吞吐最优 = top-m + 对 m 扫描 argmax E[out]/cost；E[out] = B + Σ a；成本只依赖 Q（graph + slope + sampler） | 命题 2、式 (2)(3)(4)、MTPVarlen L150 |
| 算例 B=2 nextn5：best_bsz=8、seqlens_q=[5,3]、E[out]=5.8254、吞吐峰值 1.2501 tok/ms；吞吐非单调须穷举 | §4 表、练习 1 |
| N−2 轮定总预算（CPU 不可改）、N−1 轮定 per-request（GPU 可修正）；schedule_cnt/process_cnt 配对 | §4、mtp_varlen.hpp L161-168 |
| ZOS：CPU 只定 total_q，GPU TileLang kernel（cumprod via remap + rank counting top-m）重算 seqlens_q → pinned D2H（event 定序）；Σq_i = Q 恒成立 | 式 (5)、§5、doc L324、练习 4 |
| 生产表（官方披露）：verified 5→3.2-4.4、accepted 3.29 vs 3.20（均值口径）、+4-7% @~60 conc；conc16 −1.5% 负增益（低并发 memory-bound） | §6 全表、doc L309-316 |
| CED serving：encoder 20 / consumer 20（_n_prefill_layer=20）；decoder block KV 由末 encoder hidden 投影（doc 口径）；kv_source_layer_ids [2,8,14,20] [W] 不调和 | §7、B-9 advisory |
| bounded replay 截断窗 keys(i)=[max(s,i−127),i] → 输入 token 恒 128、每层 ≤w² 对，与 T、L 无关；精确重建依赖锥上界 20×128=2,560 | 命题 3、式 (6)、doc L237/L241 |
| decoder replay 近乎减半 prefill：(20T+20w)/(40T)→1/2，T=32768 省 49.8%；结果只供 decode 不入 prefix cache（128 段短于锥，近似） | 式 (7)、§7 |
| hit policy：H≤T−134、block 定 H_1、SWA 定 H_1 vs H_1−128；suffix≥134 保证回溯 6 窗无洞；block KV 复用与存储态 bit-exact | 命题 4、式 (8)、练习 3 |
| 134=128+6（MAX_BACKTRACKING=nextn5+1）；KEEP_TOKENS 140≥134 双重 enforce；ring+6 与回溯 6 两义 | §8、environ L139、cache_engine L88、坑注 |
| StateBlock：stored（前 20 层 SWA）/ dropped（consumer SWA + ca_state/ica_state）；state_size_to_disk=20×134×528=1,415,040 B 仅 stored 落盘非 slot 总尺寸 | 式 (9)(10)、§9、StateBlock L221、坑注 |
| 528=512+16（MXFP8 per-32，kv_lora_rank=448+rope64=512 元素）；128 粒度落盘、ring 吸收 overshoot≤5；SWA 留 host memory 不上 3FS | 式 (9)、§9、2.4 §8 recap |

## 练习

1.（计算 · [10]）§4 算例口径（ $B$ = 2、nextn 5，seq0 $r$ = [0.90, 0.85, 0.80, 0.70, 0.60]、seq1 $r$ = [0.70, 0.60, 0.50, 0.40, 0.30]）：(a) 用式 (1) 算两序列的 cumprod $a$；(b) 取 $m$ = 4（ $Q$ = 6），给出全局 top-4 落点与 `seqlens_q`；(c) 用式 (2) 算 $E[N_{\mathrm{out}}]$。校验：(b) $\sum q_i = Q = 6$ 且每序列被选位置成前缀（命题 1）；(c) 与 §4 表 $m$ = 4 行 $E[N_{\mathrm{out}}]$ = 4.977 一致。
2.（证明 · [20]）独立证明命题 1（top-m 前缀可行性）：(a) 由 $r_{i,j} \le 1$ 证 $a_{i,j}$ 沿位置单调不增；(b) 把 top-m 写成阈值 $\tau$ 论证同请求祖先必入选（含同值 flat index 升序打破平局）；(c) 构造按 raw confidence $r$ 排序破坏前缀封闭的反例（须现「位 2 入选、位 1 落选」）。校验：(c) 反例可用 §4 的 seq0 $r$ = [0.30, 0.95] / seq1 $r$ = [0.50, 0.50]、 $m$ = 2；论证结构与 4.5 命题 1（树版）同源。
3.（证明 · [30]）独立证明命题 4（hit policy 正确性）三结论：(a) 两 case 证 $T - H \ge 134$；(b) 证最大回溯 $\delta$ = 6 下窗无洞等价于 $T - H \ge 128 + \delta$，并说明 case A 首 suffix query 窗为何被存储 SWA 覆盖；(c) 用前缀不可变命题 + replay 不写 block KV 证 block KV 复用与存储态 bit-exact；(d) 构造 cap 违反反例（ $T$ = 10000、 $H$ = 9900），指出缺哪些 key。校验：134 = 128 + 6 对 environ.py L139 / cache_engine.py L88；(d) 反例须现窗洞（回溯位置 9993 窗起点 9866 < 9900，缺 34 key）；block KV 复用边界恒 $H_1$ 非 $H$。
4.（代码 · [30]）读 `tilelang_ops/cuda/schedule_mtp_varlen.py` 两个 `T.Kernel`（L14）+ 参考实现（`schedule_mtp_varlen.py` L35-98）+ pytest（L139），先写预测再对照：(a) phase 2 rank counting 的每 block 工作量是多少（用 num_seqs、nextn 表示），它与 `argsort(stable)` 全局排序为何等价（rank 定义）？(b) remap = −1 的 fresh seq 在 phase 1 / phase 2 各得到什么，最终 $q_i$ 通常是多少？(c) `compute_seqlens_k_zos` 修 seqlens_k 为何要减 `nextn − prev_num_accepted_tokens` 而非直接用 seqlens_q？校验：(a) 等价性对 pytest L139 `torch.equal` 与参考实现 rank 注释；(b) fresh seq cumprod 全 0、rank 垫底， $q_i = 1$（预算超全部非零位置时例外）；(c) 对 `compute_seqlens_k_zos.py` L10 docstring 的 in-place 公式。

## 参考

- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：DeepInfer 官方部署 QA，教学参照 + 官方披露（key: deepinfer-doc，2026-11-23 全文重核，doc v0.2 / 372 行）——L303 Confidence Head 变长 verify、L307-316 增益全表（conc16 −1.5% / 3.29 vs 3.20 均值口径）、L320 internal draft、L324 ZOS GPU 重算 pinned、L334 async pipeline；L217 StateBlock stored/dropped、L219 128 粒度 + ring +6、L223 state_size_to_disk 1,415,040 B MXFP8、L227 P-to-D 共享路径、L231 跨机 400G VLAN、L237 replay 恒定成本、L239 encoder replay、L241 decoder replay 2,560 / 近乎减半、L245 hit policy L−134
- [DeepInfer repo](../references.md)：deepseek-ai/DeepInfer，教学参照 + 官方实现（key: deepinfer-repo，main@a679bb86，2026-11-23 fresh tarball file:line 核实）——models/mtp/dflash.py DFlash L82 / ConfidenceHead L46 / _embed_draft_init_tokens L230 / sample_drafts L279 / projection L148 / markov L166-175 / mtp_info_pool.store L339；csrc/engine/scheduler.hpp process_MTPInfo L1196（cumprod L1247-1248）；csrc/engine/mtp_varlen.hpp MTPVarlen L133 / schedule L150 / ForwardRuntimeEstimator L48 / SamplerRuntimeEstimator L20 / N−2 注释 L161-168；worker/worker.py _correct_seqlens_qk L191 / L219 / pinned 回写 L228-234 / 消费 L148-149；worker/cache_engine.py StateBlock L221 / get_inference_swa_block_size L88 / get_kvstore_swa_block_size L101 / _is_dropped_swa_layer L113 / get_inference_compress_block_size L131 / get_num_state_slots L201 / KVCache docstring L206-219 / state_size_to_disk L306 / select_state_slice_to_store L763 / load_state_from_store_to_prefill_cache L821；tilelang_ops/schedule_mtp_varlen.py L11 / ref L35-98 / pytest L139 + cuda/schedule_mtp_varlen.py L14；tilelang_ops/dflash.py confidence_head_fused L10；tilelang_ops/cache_common.py V41_FORMAT L109；tilelang_ops/compute_seqlens_k_zos.py L10；hugging_face/config.py _source_map L62 / _n_prefill_layer L158 / _build_dflash_cfg L177 / kv_lora_rank L289；v4_modules/yoco.py L14 / L19 / L24 / L55 / L93；input_metadata/yoco.py YocoMetadataMixin L32；environ.py L139 / L141-148 / L195-201；csrc/engine/input_metadata/prepare_tensor_input_v4.cpp L355-357；worker/model_runner.py L108-112 / profile_decode_perf L562 / capture_dflash_model L532；engine/llm_engine.py post_process L984 / update_mtp_runtime L997
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json，[W]（key: v41-flash-config，hf-mirror 传输核实，2026-11-23 第四次重取）——dspark_block_size 5 / dspark_noise_token_id 128799 / dspark_target_layer_ids [37,38,39] / dspark_markov_rank 256 / dspark_n_routed_experts 128 / dspark_num_experts_per_tok 3、sliding_window 128、kv_source_layer_ids [2,8,14,20]、compress_ratios 43 项、num_hidden_layers 40 / num_nextn_predict_layers 3、head_dim 512 / qk_rope_head_dim 64（模型侧形式化归 models，V4.1 待补）
- [Sun et al., 2024](../references.md)：You Only Cache Once: Decoder-Decoder Architectures for Language Models，arXiv:2405.05254，[A]（key: yoco，microsoft/unilm YOCO 目录）——YOCO 模型侧源头（§7 指引不展开，CED / consumer 形式化归 models）
- [Wang et al., 2025](../references.md)：TileLang: A Composable Tiled Programming Model for AI Systems，arXiv:2504.17577，[A]（key: tilelang，tile-ai/tilelang）——schedule_mtp_varlen 的 @tilelang.jit + T.Kernel DSL（§5，recap 3.4 §5 + 7.3 指引）
- [DeepSeek-AI, FlashMLA](../references.md)：deepseek-ai/FlashMLA，教学参照（key: flashmla）——528 B/token = 512 B e4m3 + 16 B e8m0 per-32 格式，与 cache_engine KVCache docstring / doc L223 三方互证（§9）
- 跨维度指引（纯文字）：models 2.4（前缀不可变命题，block KV 复用精确的依据）、models 维度 V4.1 待补（CED / DSpark draft / compress_ratios 模型侧形式化）；inference 4.4（接受规则 / E[K] 口径 / batch 浪费账，回链本章 §6）、4.5（EAGLE-2 confidence 剪枝前缀封闭，命题 1 对照）、6.2（overlap 数据依赖分类与 ZOS 同源，§5 recap）、2.2（前缀命中）、2.3 §4（MXFP8 格式账式 (4)）、2.4 §1 / §8（StateBlock 字节与三层放置）、3.4 §5（TileLang DSL）、7.1（release 边界 / config recap / async scheduling）、7.3（kernel 栈与格式账对账）
