# 4.5 投机解码系统实现（Speculative Decoding Systems）

> 4.4 给了 draft-verify 的账：加速比 $F$ 由接受率 $\alpha$ 与成本比 $c$ 决定。那 draft 模型究竟从哪来、引擎怎么把 propose-verify-accept 接进调度与显存管理？本章走读 EAGLE 三代、Medusa、MTP self-speculative、Draft & Verify 与 REST 的机制，对照 vLLM / SGLang / DeepInfer 的实现路径，再算 verify 的 prefill 形态对 chunked prefill 与 PD 分离的调度含义 | 前置：4.4（接受规则、 $E[N_{\mathrm{tok}}]$ 、加速比定理、tree verification）、models 4.5（MTP 目标）、inference 1.2（decode 账）、2.1（块管理 / COW）、3.4（CUDA Graph）、4.3（chunked prefill） | 锚点：EAGLE 系列 [Li et al., 2024] [A]（arXiv:2401.15077 / 2406.16858 / 2503.01840，SafeAILab/EAGLE）、Medusa [Cai et al., 2024] [A]（arXiv:2401.10774，FasterDecoding/Medusa）、REST [He et al., 2023] [A]、Draft & Verify [Zhang et al., 2023] [A]、vLLM / SGLang [A]、V4.1-Flash config [W]、DeepInfer 官方 doc（教学参照 + 官方披露）

## 1. EAGLE 系列：feature 级自回归与动态 tree

token 级 draft 为什么不够准？因为 target 在某个位置采样出的 token 带有不确定性，draft 若只在 token 序列上自回归，误差会逐步累积。EAGLE-1（[Li et al., 2024](../references.md)，arXiv:2401.15077，[A]）换了自回归的对象：不在 token 级，而在 target 的 second-to-top-layer feature（倒数第二层隐藏态）级。所谓 feature 级自回归，是指 draft 网络预测 target 的下一位置 feature，再经 target 的 LM head 映回 token 分布。feature 比 token 更平滑、信息更完整，预测更准——这是 4.4 命题 2「接受率 = 1 − 总变差距离」的工程兑现：feature 级 draft 缩小了与 target 的分布距离。

但仅凭 feature 仍不够。target 位置 $i$ 的 feature $f_i$ 经 LM head 给出 $p(\cdot \mid f_i)$，采样出 token $x_{i+1}$；下一位置的 feature $f_{i+1}$ 依赖这个**实际采出的** $x_{i+1}$。若 draft 只拿 $f_i$ 预测 $f_{i+1}$，它不知道 $x_{i+1}$ 采成了什么，只能对 $x_{i+1}$ 的所有可能取平均——不确定性又回来了。EAGLE-1 的解法是引入前移一时间步的 token 序列（shifted token embedding）：把实际采出的 $x_{i+1}$ 的 embedding 拼进 draft 输入，消解这份不确定性。形式化（机制级）：

$$
\hat{f}_{i+1} = D(f_i, e_{i+1}), \qquad x_{i+2} \sim \mathrm{head}(\hat{f}_{i+1}) \qquad (1)
$$

**符号**： $f_i$ = target 倒数第二层在位置 $i$ 的 feature； $D$ = 轻量 draft 网络； $e_{i+1}$ = token $x_{i+1}$ 的 embedding（ $x_{i+1} \sim \mathrm{head}(f_i)$ 是上一步实际采出的 token）； $`\hat{f}_{i+1}`$ = draft 预测的下一位置 feature； $\mathrm{head}$ = target 的 LM head（冻结复用）。关键在「shifted」：feature $f_i$ 配的是 $x_{i+1}$（比 feature 前移一步的 token），而非 $x_i$。draft 网络复用 target 的 token embedding（冻结）与 LM head，只训中间的特征预测网络；repo 实现为单个 decoder layer（`eagle/model/cnets.py` 的 `LlamaDecoderLayeremb` / `Model.midlayer`，embedding 冻结 `requires_grad = False`，2026-11-23 archive 核实）。

成本比与加速。draft 只有一层 decoder layer，单步成本约 target 的 $1/L$（memory-bound 口径，按层权重占比估算；教学口径，精确值依 profiling）。代入 4.4 章脊（8B target $L$ = 32、 $\alpha$ = 0.8、 $\gamma$ = 4）： $c \approx 1/32 \approx 0.031$，4.4 式 (8) 给 $F = 3.3616/(4 \times 0.031 + 1) \approx 2.99$。对照独立小模型 draft（ $c$ = 0.1）的 2.40：本例固定 $\alpha$ = 0.8 只隔离 $c$ 的贡献， $c$ 从 0.1 降到 0.031 就把 $F$ 从 2.40 抬到 2.99；实际 feature 级 draft 的 $\alpha$ 通常也更高（4.4 命题 2，分布距离更小），两者叠加。EAGLE-1 论文实测口径：LLaMA2-Chat 70B 上 2.7–3.5× 延迟加速、吞吐翻倍、分布保持（[Li et al., 2024](../references.md)，摘要；分布保持即 4.4 命题 1 的拒绝采样，EAGLE 不改接受规则）。repo SafeAILab/EAGLE 覆盖三代（2026-11-23 核实）。

```
EAGLE-1 一步 draft 数据流（feature 级自回归）：
  target trunk（verify 时已算）──→ f_i（倒数第二层 feature）
                                      │
  上一步采出的 x_{i+1} ──→ embedding e_{i+1}（shifted）
                                      │
                          ┌───────────┴───────────┐
                          │  draft 网络 D（单层）   │  ← 唯一训练的部分
                          └───────────┬───────────┘
                                      ↓
                                 f̂_{i+1} ──→ target LM head（冻结）──→ x_{i+2}
                                      └──────── 回灌下一步 ────────┘
```

EAGLE-2（[Li et al., 2024](../references.md)，arXiv:2406.16858，[A]）改的是 draft tree 的形状。EAGLE-1 用静态 tree（固定分支因子），但接受率不仅依赖位置、还依赖上下文（4.4 §6 的方差）：有些前缀值得多展开、有些不值得。EAGLE-2 用 draft 模型的 confidence 分数近似接受率，动态调整 tree 结构（repo README 口径：confidence scores approximate acceptance rates）。所谓 confidence，是指一条候选路径上 draft 概率的乘积——它近似该路径被 target 接受的概率。剪枝规则：按 confidence 选 top-m 条路径送去 verify。这有一条结构性质保证剪枝结果仍是合法 tree。

**命题 1**（confidence 剪枝的前缀封闭性）。设路径 confidence 为沿途 draft 概率之积，各 draft 概率 $\in (0,1]$。按 confidence 取 top-m 条路径，则保留的节点集前缀封闭：任一被保留节点，其全部祖先也被保留。

**证明**（路线：先证 confidence 沿路径单调不增，再用 top-m 阈值论证祖先必然入选）。设节点 $v$ 的父节点为 $u$， $v$ 的 draft 概率为 $q_v \le 1$，则 $\mathrm{conf}(v) = \mathrm{conf}(u) \cdot q_v \le \mathrm{conf}(u)$——confidence 沿路径单调不增。取 top-m 等价于设一个阈值 $\tau$（第 $m$ 大的 confidence），保留 $\mathrm{conf} \ge \tau$ 的节点。若 $v$ 被保留（ $\mathrm{conf}(v) \ge \tau$ ），则 $\mathrm{conf}(u) \ge \mathrm{conf}(v) \ge \tau$，故 $u$ 也被保留；逐层上推得全部祖先保留。证毕。边界： $q_v = 1$ 时父子 confidence 相等，top-m 的并列须按「祖先优先」打破平局，否则可能保留子节点而漏掉父节点（实现上按树序遍历即可保证）。

数值实例（教学口径）：根的两个子候选 draft 概率 0.6（a1）、0.3（b1）；a1 的子候选 0.5（a2）、0.4（a3）；b1 的子候选 0.7（b2）、0.2（b3）。四条路径 confidence：a1-a2 = 0.30、a1-a3 = 0.24、b1-b2 = 0.21、b1-b3 = 0.06。预算 $m$ = 3 → 保留前三条，剪掉 b1-b3。保留节点 = a1、b1（第一层，作前缀）+ a2、a3、b2（第二层），共 5 个 draft token 送 verify。前缀封闭：b2 保留 → 其父 b1 必保留（0.3 $\ge$ 0.21），验证成立。对照静态 top-2 全展开（6 个 draft token），动态剪枝省 1 个 verify 节点而几乎不损接受率——被剪的 b1-b3 confidence 仅 0.06。

EAGLE-3（[Li et al., 2025](../references.md)，arXiv:2503.01840，[A]，NeurIPS'25）解决数据 scaling 下收益受限的问题。EAGLE-1/2 约束 draft 必须预测 target 的 feature（feature prediction 约束），且只用 top-layer feature（措辞注：README 的「top-layer」泛指 EAGLE-1/2 所用的单层 feature，即 §1 的倒数第二层 feature，论文摘要口径 second-to-top-layer；此处与 EAGLE-3 的多层融合对举）；EAGLE-3 两处改动（repo README 口径）：其一，training-time test——训练时模拟推理的 drafting 过程（让 draft 消费自己的预测而非 teacher-forcing 的 target feature），消除训练 / 推理失配；其二，多层 feature 融合——用低 / 中 / 高层语义特征的融合替代单一 top-layer feature（top-layer feature 只为 next-token 预测优化，信息受限）。repo 实现里 draft 的输入投影 `fc` 取 $3 \times$ hidden（三层融合，`eagle/model/cnets.py`，archive 核实）；vLLM 侧 EAGLE-3 的多层 feature 经 `aux_hidden_states` 传入（§5）。EAGLE-3 移除 feature prediction 约束后直接训 draft 预测 token，收益随训练数据 scaling 继续涨（EAGLE-1/2 会 plateau）。

## 2. Medusa：多 heads 与 typical acceptance

Medusa（[Cai et al., 2024](../references.md)，arXiv:2401.10774，[A]，FasterDecoding/Medusa）不引入新模型，而是在原模型顶上加多个 decoding heads：第 $k$ 个 head 直接预测位置 $t+k+1$ 的 token（跳过 $k$ 个中间位置），各 head 独立、非自回归。原模型不动、只训 heads 是 Medusa-1；Medusa-2 是全模型训练 recipe（加投机预测能力的同时保持原模型性能，repo README 口径）。各 head 产出多个候选 token，用 tree attention 组合成候选树（4.4 §5），再选最长合理前缀。

关键差别在接受规则。Medusa 用 typical acceptance（典型接受）：选最长的「合理」前缀——候选 token 在 target 分布下足够典型（概率高于阈值）就接受。这是启发式截断，**不是** 4.4 命题 1 的拒绝采样 + 残差修正，因此一般不保持 target 分布。

**命题 2**（typical acceptance 不保分布）。存在 target 分布 $p$、draft 分布 $q$ 与典型阈值，使 typical acceptance 的输出分布不等于 $p$。

**证明**（反例）。取词表 $\lbrace a, b \rbrace$， $p(a) = 0.6$、 $p(b) = 0.4$；draft 恒提议 $a$（ $q(a) = 1$ ）；典型阈值 $\tau = 0.5$。typical acceptance 见 $p(a) = 0.6 \ge \tau$，判 $a$ 典型、恒接受。输出分布为 $(1, 0)$，而 $p = (0.6, 0.4)$—— $b$ 永不产出，分布被改变。证毕。对照 4.4 命题 1：拒绝采样以 $\min(1, p(a)/q(a)) = 0.6$ 接受 $a$、拒绝后从残差采 $b$，输出恰为 $(0.6, 0.4)$。差别就在残差修正：typical acceptance 没有它，接受分支确定性地偏向高频 token。

所以 Medusa 与 EAGLE / Leviathan 是两类：EAGLE 用标准拒绝采样、严格分布保持（4.4 命题 1）；Medusa 用 typical acceptance、以分布偏移换更长的接受前缀与更简单的实现。Medusa 论文以经验质量（perplexity / 下游任务）论证偏移可接受，而非数学保证。加速口径：2.2–3.6×（repo README 更新版，多模型范围）。选型时这是一笔明确权衡：要无损选 EAGLE / 拒绝采样路线，要免残差计算、容忍轻微偏移选 Medusa。

> **提点**：「分布保持」是投机解码路线的分水岭。拒绝采样路线（Leviathan / Chen / EAGLE / REST / Draft & Verify / MTP）输出严格等于 target，可直接替换部署、无质量风险；typical acceptance 路线（Medusa）须按 workload 验证质量。读任何投机方法的加速数字前，先问它属于哪一类——加速比的分母里藏着质量口径。

## 3. MTP self-speculative：nextn 模块作 draft

DeepSeek 的 MTP（multi-token prediction）模块在推理时可直接作 draft，无需额外 draft 模型。模型侧 MTP 目标（训练时让主干额外预测未来多个 token）归 models 4.5，此处 recap 一句 + 指引，不展开目标函数。serving 侧的事实是：V3 有 1 个 nextn 预测层（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]），V4.1-Flash config 的 `num_nextn_predict_layers` = 3（[DeepSeek-AI, V4.1-Flash config](../references.md)，[W]，hf-mirror 传输核实，2026-11-23 重取）；模型侧形式化归 models 维度（V4.1 待补）。

机制：verify 时 target 主干已算出各位置的隐藏态，nextn 模块拿这个隐藏态 + 已采 token 的 embedding 预测下一个 token，链式回灌 $\gamma$ 步即得 draft 序列。draft 不重跑主干——只跑 nextn 模块本身，所以成本比 $c$ 极小。按层占比估算（memory-bound 教学口径，MoE expert 权重分布不均、精确值依 profiling）：V4.1 主干 40 层、单个 nextn 层约 $1/40 = 0.025$； $\gamma$ = 3 链式 draft、 $\alpha$ = 0.8 时，4.4 式 (8) 给 $F = (1-0.8^{4})/(0.2 \times (3 \times 0.025 + 1)) = 2.952/1.075 \approx 2.75$。nextn 层数从 V3 的 1 增到 V4.1 的 3，正是给链式 draft 更深的天然深度（ $\gamma$ 上限）。vLLM 实现：`v1/worker/gpu/spec_decode/mtp/speculator.py` 的 `MTPSpeculator`（继承 `TargetDependentARSpeculator`，2026-11-23 main 核实），draft 经 `load_eagle_model` 加载 nextn 模块、消费 target 隐藏态（§5 接口）。

MTP 路线的账面优势是「零额外 draft 训练 + 零额外主干计算」：nextn 模块随主模型一起训练（模型侧目标），推理时白拿一个 $c \approx 1/L$ 的 draft。代价是 draft 深度受 nextn 层数限（V3 天然 $\gamma$ = 1，V4.1 天然 $\gamma$ = 3；链式更深则接受率跌），且接受率受 MTP 训练质量约束。这与 EAGLE 的「专门训一个 feature draft」互补：MTP 复用模型自带部件，EAGLE 外挂轻量网络。

## 4. Draft & Verify、REST 与 n-gram：免训练路线

Draft & Verify（[Zhang et al., 2023](../references.md)，arXiv:2309.08168，[A]，ACL 2024，dilab-zju/self-speculative-decoding）是 self-speculative：draft 与 verify 用同一个模型。两阶段——drafting 选择性跳过中间层（layer skip）得到一个便宜的「子模型」前向出 draft token；verification 用原模型单次前向验证。免辅助模型、免额外训练、免额外内存、输出一致（用标准拒绝采样，4.4 命题 1）。跳哪些层由 Bayesian optimization 搜索（repo `searching.py` / `skip_layers.json`，archive 核实）。

成本比是这条路线的软肋。跳过一半层也只省一半计算： $L$ = 32、跳 16 层 → $c \approx 16/32 = 0.5$（memory-bound 口径，保留层权重占比）。代入 $\alpha$ = 0.7、 $\gamma$ = 3： $F = (1-0.7^{4})/(0.3 \times (3 \times 0.5 + 1)) = 2.533/2.5 \approx 1.01$——几乎不加速。对照 4.4 命题 3 的 $\alpha > c$ 条件： $\alpha$ = 0.7 > $c$ = 0.5 勉强满足，但 $c$ 太大把 $F$ 压到 1 附近。所以 layer skip 路线要加速，须 $\alpha$ 远高于 $c$（如 $\alpha$ = 0.9、 $c$ = 0.5 时 $F \approx 1.38$；跳得少则 $c$ 大、跳得多则 $\alpha$ 跌），权衡空间窄。它的价值在「零成本部署」：不训任何东西、不占额外显存，适合无法引入 draft 模型的场景。

REST（[He et al., 2023](../references.md)，arXiv:2311.08252，[A]，NAACL 2024，FasterDecoding/REST）用检索替代 draft 模型。机制（repo README 口径）：用当前上下文的最长后缀去 datastore 检索匹配文档，把检索到的续写构造成 Trie、剪掉低频分支，候选送 tree attention mask verify；接受从头起正确的 token、第一个错误之后拒绝。免训练、plug-and-play、可加速任意现成模型。成本比 $c \approx 0$（draft 是检索 + 查表，不跑 GPU 前向；检索延迟可在 CPU 侧与 GPU verify 重叠）。由 4.4 §4 提点， $c \approx 0$ 时即使 $\alpha$ 低也有正收益（ $F \approx (1-\alpha^{\gamma+1})/(1-\alpha)$ ）。datastore 可放 CPU 内存 / 盘，不占 HBM——这是它相对独立小模型的显存优势。

n-gram 是 REST 的退化形：不查外部 datastore，只查 prompt / 近期上下文的 n-gram 续写（Leviathan §3.6 的 bigram 即此，4.4 §4 提点已算 $\alpha \approx 0.2$、 $\gamma$ = 3 → 1.25×）。vLLM 的 `ngram/speculator.py` 与 SGLang 的 `ngram_worker.py` / `cpp_ngram/` 即此路线（§5）。三条免训练路线的共性： $c$ 小或零、部署门槛低，但 $\alpha$ 受候选来源覆盖面限——datastore 越广（REST）覆盖越好，纯查表（n-gram）覆盖最窄。

## 5. 引擎实现对照：propose-verify-accept 接口

引擎怎么把 4.4 的范式接进运行时？核心是一个 speculator 接口：propose 出 draft token，target verify，accept 决定保留前缀。本节机制例证片段级（模块全量走读归 5.4 / 6.x，spec §4.2 两层纪律）。

vLLM（[vLLM repo](../references.md)，[A]，2026-11-23 main 核实）：实现在 `vllm/v1/worker/gpu/spec_decode/`，三个基础 speculator——`eagle/speculator.py`（`EagleSpeculator`）、`ngram/speculator.py`、`mtp/speculator.py`（`MTPSpeculator`），均继承 `TargetDependentARSpeculator`。基类 `speculator.py` 的 `propose` 接口消费 target 的隐藏态：

```python
# vllm/v1/worker/gpu/spec_decode/speculator.py propose（机制摘录，教学用，非优化实现）
def propose(self, input_batch, attn_metadata, slot_mappings,
            last_hidden_states,   # [num_tokens, hidden_size]，target 隐藏态（EAGLE-1 取倒数第二层）
            aux_hidden_states,    # num_layers x [num_tokens, hidden_size]，EAGLE-3 多层 feature
            num_sampled, num_rejected, last_sampled, ...):
    # draft 复用 target 隐藏态 → 不重跑主干（MTP / EAGLE 的 c 小的接口体现）
    ...  # 返回 draft token ids [num_reqs, num_speculative_tokens]
```

`last_hidden_states` 与 `aux_hidden_states` 两个参数正是 §1 / §3 的接口兑现：EAGLE-1 用**倒数第二层** feature（second-to-top-layer，与 §1 一致；经 `last_hidden_states` 参数传入——参数名是「target 隐藏态」的泛称，对 EAGLE-1 实取倒数第二层，勿按字面读成末层）、EAGLE-3 用多层融合 feature（`aux_hidden_states`，`eagle/eagle3_utils.py`）、MTP 用 nextn 隐藏态——draft 都消费 target 已算出的隐藏态，不重跑主干。接受侧 `rejection_sampler.py`（`RejectionSampler`）实现 4.4 §2 的拒绝采样 + 残差修正。

一轮的运行时数据流把三个接口串起来：propose 拿上一步 target 隐藏态出 $\gamma$ 个 draft token（写入 speculative KV slot，`slot_mappings` 指定块位置）→ target 对 $\gamma+1$ 个位置一次前向（verify，prefill 形态）→ `RejectionSampler` 逐位比 $p_i$ 与 $q_i$、定接受前缀长度 $n$ → 接受路径的 KV 提交、拒绝分支的 slot 回滚，`num_sampled` / `num_rejected` 反馈给下一步 propose 与调度器。这条流里只有 verify 跑 target 主干，draft 与 accept 都是轻量算子——所以 $c$ 小的路线（EAGLE / MTP）整轮墙钟几乎等于一次 verify，4.4 式 (8) 的前提在接口层兑现。目录已大幅扩展（`dspark/` / `dflash/` / `suffix/` / `standalone_ar/` / `multi_module_mtp/` 等 + `acceptance_estimator.py` / `adaptive_verification.py`，2026-11-23 核实），本章只走读 eagle / ngram / mtp 三件（spec 大纲范围），其余一句「目录已扩展」，全量走读归 5.4。旧 `v1/spec_decode/` 现仅 metrics / dynamic 辅助（W0 口径）。

SGLang（[SGLang repo](../references.md)，[A]，2026-11-23 main 核实）：实现在 `srt/speculative/`。**版本漂移**：`eagle_worker.py` 已重构为 `eagle_worker_v2.py`（`class EAGLEWorkerV2(BaseSpecWorker)`，L1564）+ `eagle_worker_common.py` + `multi_layer_eagle_worker_v2.py`；另有 `eagle_draft_cuda_graph_runner.py` / `eagle_draft_extend_cuda_graph_runner.py`（draft 侧 CUDA Graph 捕获）、`ngram_worker.py` / `cpp_ngram/`、`dspark_components/`（`dspark_draft.py` / `dspark_block_accept_estimator.py` / `dspark_config.py`）、`spec_{info,utils}.py`。配置面同步迁移：speculative 参数已从 `server_args.py` 迁至 `runtime_context`（经 `get_schedule()` 读取，与 4.3 的 `chunked_prefill_size` 迁移同型）——引用配置按新路径，勿写 `server_args` 旧名。

DeepInfer DSpark：Confidence Head 变长 verify（4.4 §6 已立动机与账），实现在 `dspark_components/`（SGLang 侧）与 DeepInfer 内部；本章只前瞻指针 + 回链，Confidence Head 结构、接受率排序与吞吐最优子集选择归 7.2 正式走读（回链 4.5）。模型侧 draft 结构归 models 维度（V4.1 待补）。

| 引擎 | draft 路线 | 关键文件（2026-11-23 核实） | 接受实现 | 版本注 |
| --- | --- | --- | --- | --- |
| vLLM | EAGLE / n-gram / MTP（+ 扩展目录） | `v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py` | `rejection_sampler.py` | 旧 `v1/spec_decode/` 降辅助 |
| SGLang | EAGLE / n-gram / DSpark | `srt/speculative/eagle_worker_v2.py` / `ngram_worker.py` / `dspark_components/` | `target_verify.py` / `spec_sampling_mask.py` | eagle_worker → v2；配置迁 runtime_context |
| DeepInfer | DSpark（变长 verify） | `dspark_components/`（SGLang 侧）+ 内部 | Confidence Head 排序（7.2） | 官方披露口径 |

## 6. 接受率调优与调度交互

$\gamma$ 怎么选、接受率怎么测？ $\gamma$ 的最优值由 4.4 §4 的数值搜索定（使 $F$ 最大的整数，依赖 $\alpha$ 与 $c$）。接受率的测量口径： $\hat{\alpha}$ = 被接受 draft token 数 / 总 draft token 数（逐位置统计， $E[K]$ 口径，4.4 §3 口径注），勿与 $E[N_{\mathrm{tok}}]$ 混。batch 下 $\alpha$ 跨请求 / 位置有方差（4.4 §6），固定 $\gamma$ 浪费 verify 算力——自适应路线由此生：vLLM 的 `acceptance_estimator.py` / `adaptive_verification.py`（在线估计接受率、动态调 verify 长度，2026-11-23 核实存在，机制详展归 5.4）、SGLang 的 `online_acceptance_estimator.py` / `adaptive_spec_params.py`。Leviathan §3.5 给过 oracle 上界：若逐轮预知 $\beta$ 而变 $\gamma$，期望产出可达 $1/(1-\alpha)$，改善因子比固定 $\gamma$ 至多高约 60%（[Leviathan et al., 2022](../references.md)，§3.5）——这是自适应 $\gamma$ 的收益天花板。

verify 是 prefill 形态，这笔账改变调度。普通 decode 每请求每步打 1 个 token；投机 verify 每请求每步打 $\gamma+1$ 个 token（query 长度 $\gamma+1 > 1$，attention 逐序列、FFN 合批，与 4.3 的混合批同构）。三个调度含义：

其一，chunked prefill 的 token 预算须按 $B(\gamma+1)$ 计。vLLM 的 `max_num_batched_tokens`（4.3）限制每步总 token 数；开投机后 decode 步本身就吃 $B(\gamma+1)$ 个名额。代入 $B$ = 32、 $\gamma$ = 4：每步 verify 打 $32 \times 5 = 160$ token，若预算 512，则留给 prefill chunk 的只剩 $512 - 160 = 352$（不开投机时 decode 只占 32、留 480）——投机解码挤占 prefill 预算，长 prompt 的 TTFT 可能上升。调度器须在「投机加速 decode」与「prefill 排队」之间权衡（4.3 的 stall 账多了一个投机项）。

其二，PD 分离下 verify 归 D 实例、但算力形态是 prefill。verify 的 $\gamma+1$ token 前向是 compute-intensive（prefill 形态），落在 decode 实例上会抬高 D 侧算力压力——P/D 配比须把 verify 的额外计算计入 D 侧负载（8.2 的 goodput 与配比账多一个投机修正项）。draft 模型的放置（与 target 同卡 vs 独立）也影响 D 侧显存与算力分配。

其三，动态 shape 与 KV 回滚。接受长度逐轮变化（0 到 $\gamma$），verify 的输入 shape 随之变——CUDA Graph 捕获须按 piecewise 处理动态部分（3.4），SGLang 的 `eagle_draft_cuda_graph_runner.py` 即 draft 侧的图捕获。被拒绝分支的 speculative KV 须回滚（不提交），衔接 2.1 的块管理 / COW：接受路径继承共享前缀块，拒绝分支的独占块归池。

> **坑**：把投机 decode 步当普通 decode 步算 token 预算会低估每步负载 $\gamma+1$ 倍。触发条件：容量规划 / chunked prefill 预算按 $B$ 而非 $B(\gamma+1)$ 计；现象：prefill chunk 被挤占、TTFT 尖刺，或 D 实例算力过载；规避：verify 按 prefill 形态计 token 与 FLOPs（4.4 §1 / §6），预算与 P/D 配比显式含 $\gamma+1$ 因子。

## 总结

| 论断 | 回指 |
| --- | --- |
| EAGLE-1：feature 级自回归 + shifted token embedding 消不确定性；单层 draft $c \approx 1/L$，8B 口径 $F \approx 2.99$；分布保持 | §1、式 (1)、4.4 命题 1 |
| EAGLE-2：confidence（路径 draft 概率积）近似接受率、动态剪枝；剪枝结果前缀封闭 | 命题 1、§1 |
| EAGLE-3：training-time test 消训练/推理失配 + 低/中/高层 feature 融合替代 top-layer | §1 |
| Medusa：多 heads 非自回归 + tree attention + typical acceptance；一般不保分布（反例） | 命题 2、§2 |
| MTP self-spec：nextn 作 draft、复用 target 隐藏态；V3 nextn 1 → V4.1 nextn 3 [W]； $c \approx 1/40$ | §3、models 4.5 指引 |
| Draft & Verify：layer skip self-spec， $c \approx$ 保留层占比（偏大， $F \approx 1.01$）；免训练免额外内存 | §4 |
| REST / n-gram：检索 / 查表 draft， $c \approx 0$，低 $\alpha$ 也正收益；datastore 不占 HBM | §4、4.4 §4 提点 |
| vLLM `v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py`；SGLang eagle_worker_v2（漂移）；DeepInfer DSpark → 7.2 | §5 表 |
| verify 是 prefill 形态：token 预算按 $B(\gamma+1)$、PD 配比含 verify 算力、动态 shape + KV 回滚 | §6、4.3 / 8.2 / 3.4 / 2.1 |

## 练习

1.（计算 · [10]）Draft & Verify 口径： $L$ = 32、跳 16 层（ $c \approx 0.5$）、 $\alpha$ = 0.7、 $\gamma$ = 3，用 4.4 式 (8) 算 $F$；再算 EAGLE 单层 draft（ $c \approx 1/32$）、同 $\alpha$ = 0.8、 $\gamma$ = 4 的 $F$，比较两者。校验： $F$ 与 4.4 式 (8) 代入一致；Draft & Verify 的 $\alpha > c$（0.7 > 0.5）满足 4.4 命题 3 条件但 $F$ 仍接近 1，说明 $c$ 大压低加速。
2.（计算 · [20]）EAGLE-2 confidence 剪枝：根子候选 draft 概率 0.6 / 0.3，其下各两个子候选 0.5 / 0.4 与 0.7 / 0.2。(a) 算四条路径的 confidence；(b) 预算 $m$ = 3 时保留哪些路径、共几个 draft token 送 verify；(c) 验证保留集前缀封闭。校验：confidence = 沿途 draft 概率之积（命题 1）；子节点 confidence $\le$ 父节点；(b) 应剪掉 confidence 0.06 的路径。
3.（证明 · [30]）独立证明命题 1（confidence 剪枝前缀封闭）：(a) 证 confidence 沿路径单调不增（用 draft 概率 $\le 1$）；(b) 把 top-m 选择写成阈值 $\tau$ 论证祖先必然入选；(c) 给出 $q_v = 1$（父子 confidence 相等）时的边界反例与打破平局规则。校验：你的证明应蕴含「保留叶节点必保留其全部祖先」；边界反例说明为何须祖先优先打破平局。
4.（代码 · [30]）读 vLLM `v1/worker/gpu/spec_decode/speculator.py` 的 `propose` 接口与 `mtp/speculator.py`（main 2026-11-23）：先写出预测—— `propose` 为什么消费 `last_hidden_states` 与 `aux_hidden_states` 而非重新跑 target 主干？这两个参数各对应哪条 draft 路线（EAGLE-1 / EAGLE-3 / MTP）？再对照源码与 `MTPSpeculator(TargetDependentARSpeculator)` 验证。校验： `aux_hidden_states` 注释为 num_layers x [num_tokens, hidden_size]（多层 feature，EAGLE-3）；预测应与 §1 / §3 / §5 的「draft 复用 target 隐藏态」一致。

## 参考

- [Li et al., 2024](../references.md)：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty，arXiv:2401.15077，[A]（key: eagle，SafeAILab/EAGLE）——feature 级自回归 + shifted token embedding、LLaMA2-Chat 70B 2.7–3.5×（§1）
- [Li et al., 2024](../references.md)：EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees，arXiv:2406.16858，[A]（key: eagle2）——context-aware 动态 draft tree、confidence 剪枝（§1）
- [Li et al., 2025](../references.md)：EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test，arXiv:2503.01840，[A]（key: eagle3，NeurIPS'25）——training-time test + 多层 feature 融合（§1）
- [Cai et al., 2024](../references.md)：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads，arXiv:2401.10774，[A]（key: medusa，FasterDecoding/Medusa）——多 heads / Medusa-1/2 / tree attention / typical acceptance、2.2–3.6×（§2）
- [He et al., 2023](../references.md)：REST: Retrieval-Based Speculative Decoding，arXiv:2311.08252，[A]（key: rest，NAACL 2024，FasterDecoding/REST）——datastore 检索 + Trie 剪枝 + tree verify、免训练（§4）
- [Zhang et al., 2023](../references.md)：Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding，arXiv:2309.08168，[A]（key: draft-verify，ACL 2024，dilab-zju/self-speculative-decoding）——layer skip self-speculative、Bayesian optimization 选层（§4）
- [Leviathan et al., 2022](../references.md)：Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192，[B]（key: leviathan）——最优 $\gamma$ 数值搜索、oracle 上界 $1/(1-\alpha)$（§6）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——nextn 1 层（§3）
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json，[W]（key: v41-flash-config，hf-mirror 传输核实）——num_nextn_predict_layers = 3、40 层（§3）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——`v1/worker/gpu/spec_decode/{eagle,ngram,mtp}/speculator.py`、`speculator.py` propose、`rejection_sampler.py`、`eagle/eagle3_utils.py`、`acceptance_estimator.py` / `adaptive_verification.py`（§5 / §6，2026-11-23 main 核实）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——`srt/speculative/eagle_worker_v2.py`（EAGLEWorkerV2，L1564）/ `eagle_worker_common.py` / `ngram_worker.py` / `cpp_ngram/` / `dspark_components/` / `eagle_draft_cuda_graph_runner.py`、配置迁 runtime_context（§5 / §6，2026-11-23 核实）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：DeepInfer 官方部署 QA，教学参照 + 官方披露（key: deepinfer-doc）——DSpark 变长 verify（§5 指针，详展 7.2，回链 4.4 §6）
- 跨维度指引（纯文字）：models 4.5（MTP 目标，§3 recap）、models 维度 V4.1 待补（nextn / DSpark 模型侧形式化）；inference 4.4（接受规则 / 加速比 / tree verification）、1.2（decode 账）、2.1（块管理 / COW，KV 回滚）、3.4（CUDA Graph piecewise）、4.3（chunked prefill 预算）、5.4 / 6.x（引擎模块全量走读）、7.2（DSpark 正式走读，回链本章）、8.2（PD 配比）、9.2（引擎选型）
