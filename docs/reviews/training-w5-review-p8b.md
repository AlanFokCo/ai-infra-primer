# training W5 批 2 对抗审查报告 — 8.4/8.5 逐章 + 全波 5 章 sweep

日期：2026-11-18。evaluator：独立 fresh-context（与写作 agent、批 1 evaluator 无共享上下文）。立场：假设有错，一手来源证伪后才放行。
审查对象：`training/08-rl-math/04-grpo-and-variants.md`、`05-llm-rl-systems.md` 逐字；`01-`~`03-` 跨章扫读。标准：STYLE_GUIDE.md、rubric.md、training spec §4.3/§5（8.4/8.5 节）、spec 审查 C 的 (G−1)/G 先例。

## 0. 结论

**不放行（修复后放行）**。S1×1、S2×3、S3×4。四处必修项均为点修：S1 是 references.md `verl` 条目作者字段错误（首作者实为 Guangming Sheng）并传播到 6.3 与 8.5；S2 三处分别是 8.4 对照表 DAPO 行 KL 列与原文矛盾、8.5「式 3」悬空引用、8.5 符号 N_a 一章两义。数学主线（(G−1)/G 组基线偏差、k3 无偏性、GSPO 序列级 ratio、四模型显存账、权重同步账、HybridFlow Table 2）经独立重推导全部成立；两章四十余项数值与约 30 个 repo 路径/函数名经一手来源核对，除 findings 所列外全部命中。

## 1. Findings 表

| # | 级别 | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S1-1 | S1（张冠李戴的引用） | `training/references.md` 行 124（`verl` 条目作者字段 "Hu et al."）；传播面：8.5 行 3/85/94/144/203、6.3 行 62 | HybridFlow（arXiv:2409.19256）被引为 [Hu et al., 2024] | arXiv API 返回首作者 **Guangming Sheng**（作者列：Sheng, Zhang, Ye, Wu, Zhang, Zhang, Peng, Lin, Wu——无 Hu）；旁证二重：DAPO 论文引用键 `sheng2024hybridflow`、Dr.GRPO 论文 Table 2 作 "verl (Sheng et al. 2024)"。同章 8.5 与 6.3 中 OpenRLHF（2405.11143，首作者 Jian Hu，无误）与之同名同年并列，读者无法区分两条 [Hu et al., 2024] | references.md 作者字段改 "Sheng et al."；8.5 与 6.3 的 veRL 引用改 [Sheng et al., 2024]。根源在 W0 research pass，非本波写作引入；写作方已在 task-28 存疑 #3 上报，本审查确认属实 |
| S2-1 | S2（局部事实错误） | 8.4 §8 变体对照总表 DAPO 行「KL 位置与估计」列「同 GRPO」 | 表格称 DAPO 沿用 DeepSeekMath 的 loss 内 k3 KL | DAPO 原文 §2.3 "Removing KL Divergence"："we will exclude the KL term from our proposed algorithm"；其目标式 (8)/(10)/(12) 均无 KL 项；本章自身式 (10) 亦无 KL 项——表格与原文及本章公式双重矛盾。同表 GSPO / GSPO-token 行「同 GRPO 族」亦无原文支撑（GSPO §2 声明 "we omit the KL regularization term hereinafter for brevity"，目标式 (5)(13) 无 KL） | DAPO 行改「移除 KL（其 §2.3）」；GSPO 两行改「原文略去 KL（for brevity）」或等效措辞；表注相应收紧 |
| S2-2 | S2（内部断链） | 8.5 §7 对照表 veRL 行「3D-HybridEngine resharding（式 3 与 Table 2）」 | 引用本章式 (3) | 本章编号公式只有 (1)(2)(4)：§4 的 d_g 定义 display 未编号，其后截断 IS 式直接编号 (4)——式 (3) 不存在（task-28 report 自报「式 (3) 组微-DP 通信量」，实际未落编号） | 给 §4 的 d_g 定义式补编号 (3)（后续式号顺延），或表内改引「§4」 |
| S2-3 | S2（符号约定冲突） | 8.5 §2 符号表 vs §4 行 94 与行 108 符号表 | N_a 一章两义：§2 定义为 actor 参数量（式 (1) 的 16N_a 等），§4 用作 GPU 数（「全 N_a = t p d 卡 all-gather」「t_g p_g d_g d = t p d = N_a」） | 后者是 HybridFlow 论文原文记号（其 §5.4 "N_a GPUs"），移植时未与本章 §2 定义消歧；按 §2 定义读 §4 会得到「参数量 = tpd」的错义。spec §4.3 的精神是全书符号唯一 | §4 的 GPU 数换字母（如 N_g 或照论文原文加一句「此处 N_a 为论文原记号、指 GPU 数，与本章 §2 的参数量 N_a 无关」） |
| S3-1 | S3 | 8.4 §5「其附录 A 的推导把组均值 baseline 当作与样本无关处理并称修正后无偏」 | 对 Dr.GRPO 附录 A 机制的转述 | 附录 A 实际路线：显式写出 G/(G−1)·Ã = Â^RLOO 的等价链，并称该常数因子 "can be subsumed into the learning rate"——并非把均值「当作样本无关」。两说的最终口径一致（常数因子、lr 可吸收），章内结论无误，机制描述欠准 | 改为「其附录 A 经 RLOO 等价性论证无偏（G/(G−1) 因子吸收进 lr）」 |
| S3-2 | S3 | 8.4 §2 末段「RLOO 一族算法」 | RLOO 首现未展开全称 | STYLE_GUIDE §1 首次出现概念给定义；REINFORCE Leave-One-Out 未展开（LOO 概念本身同节已完整定义） | 首现处加「（REINFORCE Leave-One-Out）」 |
| S3-3 | S3 | 8.5 总结表「式 1」「式 2」 | 与正文「式 (1)」「式 (2)」括号体例不一 | 全章其余处均带括号 | 统一为「式 (N)」 |
| S3-4 | S3 | 8.5 §4「生成侧网格是 t_g × p_g × d_g × d——micro-DP 之外还套一层与训练 DP 同度的外层 DP」 | 该分解系本书读法，论文未命名外层 DP（task-28 存疑 #1 自报） | 与论文 Figure 8(b) 8 卡例及 Table 2 算术自洽（2·1·2·2 = 8 = tpd），内容无误；仅缺「本书读法」标记 | 加「（本书读法）」四字；内容可放行 |

## 2. 强制独立重推导（逐步中间过程）

### 2.1 8.4 组基线 (G−1)/G 偏差（对照章式 (4)-(6)）

设定（与章一致）：固定 q，i.i.d. 采 o_1..o_G ~ π_θ(·|q)，R_i = R(o_i)；ĝ = (1/G)Σ_i (R_i − R̄)∇log π_θ(o_i)，R̄ = (1/G)Σ_j R_j。记 ∇J_q = E[R(o)∇log π_θ(o|q)]。

- 第一项：E[(1/G)Σ_i R_i ∇log π(o_i)] = (1/G)·G·∇J_q = ∇J_q。
- 第二项：对固定 i，E[R̄∇log π(o_i)] = (1/G)Σ_j E[R_j ∇log π(o_i)]。j = i 项 = ∇J_q；j ≠ i 项由 o_j ⊥ o_i 拆成 E[R_j]·E[∇log π(o_i)]，score 零均值（E_{o~π}[∇log π] = ∇Σπ = 0）消去，= 0。故 E[R̄∇log π(o_i)] = (1/G)∇J_q，第二项合计 (1/G)Σ_i (1/G)∇J_q = (1/G)∇J_q。
- 合并：E[ĝ] = (1 − 1/G)∇J_q = **(G−1)/G · ∇J_q**。✓ 与章式 (5)(6) 逐步一致（章式 (5) 的 1/G² 前缀与逐项拆分正确）。

LOO 版：b_i = (1/(G−1))Σ_{j≠i} R_j 与 o_i 独立 ⇒ E[b_i ∇log π(o_i)] = E[b_i]·0 = 0 ⇒ E[ĝ_LOO] = ∇J_q 精确无偏。✓ 与章式 (7) 一致。与 spec 审查 C 的先例口径同。

### 2.2 k3 KL 估计器（无偏性 + 低方差机制）

记 x ~ q、r = p(x)/q(x)，估计 D_KL(q‖p) = E_q[−log r]。k3 = (r−1) − log r。

- 无偏：E_q[k3] = E_q[r] − 1 − E_q[log r] = ∫q·(p/q)dx − 1 + D_KL(q‖p) = 1 − 1 + D_KL = D_KL(q‖p)。✓ 与章式 (8) 一致。
- 恒正：log r ≤ r−1 ⇒ r − 1 − log r ≥ 0。✓
- 低方差机制：r−1 是零期望控制变量；近端 −log r ≈ −(r−1) + (r−1)²/2，与 r−1 负相关，Var(k3) ≈ Var(k1) − Var(r) < Var(k1)。✓ 与章内「负相关部分相消」一致；Bregman「曲线减切线」读法与 Schulman 博客原文一致。

方向核对（章式 (9)）：样本来自 π_old（内层首轮 = π_θ），取 q := π_θ、p := π_ref，r = π_ref/π_θ，k3 = π_ref/π_θ − log(π_ref/π_θ) − 1，估计 D_KL(π_θ‖π_ref)（前向，「当前背离参考」）。✓ 与 DeepSeekMath 原文式 (4) 逐字一致（ar5iv alttext 核对）。

### 2.3 8.5 四模型同驻显存账（式 (1)）

训练态：bf16 参数 2N + bf16 梯度 2N + fp32 master 4N + fp32 m,v 各 4N = 16N ✓（与 models 1.2 式 (8)、training 4.2 口径一致，已交叉核对两章原文）。纯推理 2N。

合计：16N_a + 16N_c + 2N_a(ref) + 2N_r(RM) + 2N_a(rollout) + KV；全同尺寸 N ⇒ (16+16+2+2+2)N = **38N + KV** ✓。
代数字：7B ⇒ 38×7e9 = 2.66e11 B = 266 GB ✓；单训练 actor 112 GB > 80 GB 单卡 ✓；70B ⇒ 2.66 TB ✓；128×80 GB = 10240 GB ≈ 10 TB ✓。
ZeRO-1：16N → 2N+2N+12N/d = 4N + 12N/d；d=64 ⇒ 4.1875N ≈ 4.2N ✓（与 4.2 表逐项一致）。
GRPO 去 critic + RLVR 去 RM：38−16−2 = 20N；省 18/38 = 47.4% ≈ 47% ✓。

### 2.4 8.5 权重同步成本账（式 (2)）

t_sync ≥ 2N_a/B（只含带宽项的下界口径，章内已声明）。7B：1.4e10 B ÷ 600 GB/s = 0.0233 s ✓；÷ 25 GB/s = 0.56 s ✓。70B：0.233 s ✓ / 5.6 s ✓。共享存储：140 GB ÷ (10-12 GB/s) ≈ 11.7–14 s ⇒「十几秒级」✓（10-12 GB/s 口径回链 4.2 §5，已核对 4.2 原文含此数）。

HybridFlow Table 2 复核：DS-Chat (tpd−1)/(tpd)·M、HybridFlow-V (tp−1)/(tp)·M、HybridFlow (d_g−1)/(tp)·M 与峰值/冗余三行——与论文 §5.4 逐项一致 ✓；等价写法 (tp − t_g·p_g)/(t_g·p_g·tp)·M 按 d_g = tp/(t_g p_g) 代入即等 ✓（自算）。8 卡例 (t,p,d)=(4,1,2)→(t_g,p_g)=(2,1)：d_g=2，系数 0.875/0.75/0.25M ✓；生成 TP 组 {G1,G3}{G2,G4}… 与 micro-DP 组 {G1,G2}… 与论文 Figure 8(b) 逐组一致 ✓；「每卡生成分片 = 本 micro-DP 组各卡训练分片之并」算术自洽（d_g·M/(tp) = M/(t_g p_g)）✓。

## 3. Verification log（抽查项 × 来源 × 结论）

来源获取方式：arXiv API（作者/标题）、ar5iv 全文（HTML，math alttext 逐式核对）、joschu.net 原文、arxiv.org/abs 现行摘要、GitHub API contents/raw（repo 路径与函数体，2026-11-18 核实）。

### 8.4（逐条）

| 章内断言 | 来源 | 结论 |
| --- | --- | --- |
| GRPO 目标式 (2)（min-clip + loss 内 KL + 1/G·1/\|o_i\| 双重平均） | DeepSeekMath ar5iv 式 (3) alttext | ✓ 逐符号一致 |
| advantage 式 (1) 组内 (r−mean)/std、全 token 同值 | 其 §4.1.2 | ✓ |
| critic "another model of comparable size" | 其 §4.1.1 原文 | ✓ 原句命中 |
| KL 进 loss 动机 "avoiding complicating the calculation of Â" | 其 §4.1.1 原文 | ✓ 原句命中 |
| KL 进 reward 式 (3)（InstructGPT 口径） | 其式 (2) alttext | ✓ 一致 |
| k3 估计器式 (9) = π_ref/π_θ − log − 1，自引 (Schulman 2020) | 其式 (4) alttext + 自引 | ✓ 一致 |
| 配置 G=64、β=0.04、lr 1e-6、max len 1024、batch 1024、μ=1 | 其 §4.2 | ✓ 全中 |
| Iterative GRPO：ref 刷新 + RM 10% replay | 其 §4.1.4 + Algorithm 1 | ✓ |
| RM 比较本性论证（组内相对与 RM 训练数据对齐） | 其 §4.1.1 | ✓ |
| TRL `grpo_trainer.py`：rewards − mean_grouped；÷(std+1e-4)；scale_rewards | huggingface/trl main 行 2935-2957 | ✓ 逐行命中 |
| k1/k2/k3 定义、f-散度二阶等价 f''(1)=1、Bregman 几何 | joschu.net/blog/kl-approx.html 原文 | ✓ |
| 博客数值：KL=0.005 时 k1 std 20×、k3 1.42×、k2 偏差 0.2%；KL=0.5 时 k2 偏差 50%（0.25/0.5）、k3 std 1.7 < k2 1.73、"strictly better" | 同上（含其数值表与结论句） | ✓ 全中 |
| (G−1)/G 组基线偏差、LOO 无偏 | 本书推导（§2.1 独立重推） | ✓ 成立 |
| Dr.GRPO 两偏置（response 级 1/\|o_i\|、question 级 std）+ Figure 5 reweighting 读法 | 2503.20783 ar5iv §3.1 + Figure 5  caption | ✓ |
| Dr.GRPO 修正（删两项归一；masked_mean 除数换 MAX_TOKENS 常数） | 其 §3.2 + 代码对照块 | ✓ |
| 极简配方 43.3% AIME 2024、27h @ 8×A100、Qwen2.5-Math-7B + MATH L3-5 + Qwen-Math 模板 | 其摘要/§1（alttext 取回 43.3%、27、8×）+ Table 4 | ✓ |
| 开源 PPO 实现长度偏置（OpenRLHF per-response；TRL/veRL per-batch）；「公式无偏、实现有偏」 | 其 §3.1 代码块注释与脚注 3 | ✓ |
| DAPO 主结果 50 vs R1-Zero-Qwen-32B 47 @ 50% 步数；vanilla GRPO 30 | 2503.14476 ar5iv 摘要/§1/§4.2 | ✓ |
| clip-higher：0.01→0.012 vs 0.9→1.08 算例；上截断 token 均概率 <0.2；ε_low=0.2/ε_high=0.28；下界不加宽理由 | 其 §3.1 + §4.1 | ✓ 全中 |
| dynamic sampling：约束 0<\|{正确}\|<G（其式 (11)）、Figure 3(b) 全对组上升、长尾同步吞吐注记 | 其 §3.2 | ✓ |
| token 级聚合式 (10)（其式 (12)，1/Σ\|o_i\|）；动机（长样本学不动/乱码罚不动/熵与长度反推） | 其 §3.3 | ✓ |
| overlong：过滤 + 软惩罚分段式 (11)（其式 (13)）；16384/4096/20480 | 其 §3.4 + §4.1 | ✓ |
| DAPO 配置：prompt 批 512、G=16、minibatch 512、16 次更新/rollout、AdamW 1e-6 + 20 步 warmup、avg@32、temp 1.0、top-p 0.7 | 其 §4.1 | ✓ 全中 |
| DAPO advantage 保留 std 归一 | 其式 (9) alttext | ✓（章内注记正确） |
| **DAPO 目标不含 KL**（其 §2.3 "exclude the KL term"） | 其 §2.3 + 式 (8)/(10)/(12) | **✗ 章内对照表 DAPO 行「同 GRPO」与原文矛盾 → S2-1** |
| GSPO 序列级 ratio 式 (12)（其式 (7)，长度归一几何平均） | 2507.18071 ar5iv 式 (7) alttext | ✓ |
| GSPO 目标 min-clip（其式 (5)） | 其式 (5) | ✓ |
| 梯度对照式 (13)(14)（其 §4.2 式 (10)(12)）；token 权重区间 (0,1+ε]/[1−ε,+∞) | 其 §4.2 | ✓ |
| clip 3e-4/4e-4 vs GRPO 0.2/0.27（调参后） | 其 §5.1 | ✓ |
| Qwen3-30B-A3B-Base 冷启动、训练稳定、精度优于 GRPO；已用于 Qwen3 系列 | 其 §5.1 | ✓ |
| MoE 免 Routing Replay（GRPO 需之；replay 机制描述） | 其 §5.1/§5.3 | ✓ |
| GSPO-token sg 公式（分母 sg[π_θ]）、数值恒等 s_i、同 advantage 时与 GSPO 严格一致 | 其 §4.3 式 (13)-(16) | ✓ |
| 「ill-posed」指控、不可逆 collapse（回退 checkpoint/调超参无效）、"unit of optimization objective should match the unit of reward" | 其 §3/摘要 | ✓ 原句命中 |
| DeepSeek-R1 §2.2.1 序列级简写 + 同 k3 估计器 | 2501.12948 ar5iv 式 (1)(2) alttext | ✓ |
| 算术例 0.9975^50 ≈ 0.882、^500 ≈ 0.286≈0.29；0.2/3e-4 ≈ 667≈「约 600 倍」；1/64 ≈ 1.6% | 自算 | ✓ |

### 8.5（逐条）

| 章内断言 | 来源 | 结论 |
| --- | --- | --- |
| 生成阶段占 PPO 时间 80%（LLaMA2-7B + A100） | OpenRLHF ar5iv §3.2（Figure 4 caption 配置） | ✓ 原句命中 |
| 共驻框架（TRL/DeepSpeed-Chat）无显存余量放大 batch | 其 §3.2 同段 | ✓ |
| OpenRLHF 默认分卡、70B+ 重新设计调度、colocate_actor_ref/colocate_critic_reward/ref_reward_offload 开关、vllm_num_engines/vllm_tensor_parallel_size | 其摘要/§3.1 + 文档 CLI 块 | ✓ |
| OpenRLHF 摘要加速比 1.22–1.68× | arxiv.org/abs/2405.11143 现行摘要（v6 系；ar5iv 旧版摘要无此句，章内标「摘要口径」准确） | ✓ |
| HybridFlow hybrid 控制模型（编排单控 + 模型内多控）、分层 API 解耦 | 2409.19256 ar5iv 摘要/§1-2/§4 | ✓ |
| 3DParallelWorker base class、collect/distribute 原语 | 其 §4.1 | ✓ 原名命中 |
| Table 2 三行（通信/峰值/冗余） | 其 §5.4 表 + 文字 | ✓ 逐项一致（§2.4 重推） |
| §5.4 等价写法 (tp−t_g p_g)/(t_g p_g·tp)·M | 其 §5.4 原式 | ✓ 且代数自洽 |
| Figure 8(b) 8 卡分组例 | 其 §5.3 文字（组名列举报刊） | ✓ 逐组一致 |
| 实现：~12k LoC、API 1.8k、3D-HybridEngine 2.4k；Ray+RPC+TensorDict；Megatron/FSDP/DeepSpeed 可插；vLLM KV 管理器分布式化；训练期 offload 生成权重、生成后 offload KV | 其 §7 | ✓ 全中 |
| 测试床 16 机 × 8 A100-80GB、NVLink 600 GB/s、节点间 200 Gbps；baselines 版本（DS-Chat v0.14.0 / OpenRLHF v0.2.5 / NeMo-Aligner v0.2.0） | 其 §8.1 | ✓ |
| PPO 加速比 7B 1.68–8.63 / 13B 2.70–18.96 / 34B 2.41–20.57 / 70B 5.17–17.98；全算法 1.53–20.57 | 其 Figure 9 caption + 摘要 | ✓ 全中 |
| 16N 分解指针 models 1.2 式 (8)、ZeRO-1 4N+12N/d 指针 4.2、PCIe 10-12 GB/s 指针 4.2 §5 | models/01-foundations/02-compute-accounting.md 行 203/259；training/04-data-parallelism/02-zero.md 行 14/140 | ✓ 指针目标均存在且口径一致 |
| veRL repo 15 路径（main_ppo.py、v1/ 四文件、core_algos、trainer_utils、rollout_corr_helper、engine/base、vllm_rollout、single_controller 等） | GitHub API contents（main，2026-11-18） | ✓ 全部 200 |
| `kl_penalty`：k1 logp 差、k2 半平方、k3 形式 + ±20/±10 clamp、k3+ straight-through 及注释理由 | core_algos.py 行 2188-2240 | ✓ 逐字命中 |
| `apply_kl_penalty`：token_level_rewards = token_level_scores − β·kld；β 来自 kl_ctrl | trainer_utils.py 行 29-64 | ✓ 逐行命中 |
| `ppo_loss` use_kl_loss → kl_loss × kl_loss_coef | workers/utils/losses.py 行 178-187 | ✓ |
| `rollout_corr_helper.py` docstring 三类 off-policy 来源（含 "vLLM BFloat16 vs FSDP FP32"）、token/序列级 IS、拒采样、KL/PPL/χ²/ESS | 该文件行 15-60 | ✓ 原句命中 |
| `_compute_old_log_prob` bypass（两策略）/decoupled（三策略）双模式 | v1/trainer_base.py 行 1774-1790 docstring | ✓ 原句命中 |
| `need_critic`：GRPO 类（非 GAE）→ 不建 critic | trainer/ppo/utils.py 行 96-112 | ✓ |
| `update_weights`/`release`/`resume`（wake_up）、`bucketed_weight_transfer.py` | vllm_rollout.py 行 195-210 + 文件 200 | ✓ |
| checkpoint_engine 三后端（nccl/nixl/mooncake） | GitHub API | ✓ 全部 200 |
| OpenRLHF repo：`train_ppo_ray.py`、`ppo_trainer.py`、`ppo_actor.py`（broadcast_to_vllm、NCCL 广播/ray.util.collective、ZeRO-3 GatheredParameters、_handle_cuda_ipc + reduce_tensor）、`vllm_engine.py`（update_weight/update_weight_cuda_ipc）、`ppo_critic.py`、`launcher.py` | GitHub API raw（main） | ✓ 逐函数命中 |
| **veRL 论文作者字段** | arXiv API：首作者 **Guangming Sheng** | **✗ → S1-1** |
| vLLM 只作链接不展开 | 章内仅接口事实 + 名点即归 inference | ✓ 合规 |
| 显存账 38N/266 GB/2.66 TB/省 47%、同步账 0.023/0.56/0.23/5.6 s、「十几秒级」 | §2.3-2.4 独立重算 | ✓ |
| **式 (3) 悬空引用**（§7 表） | 章内编号只有 (1)(2)(4) | **✗ → S2-2** |
| **N_a 一章两义**（§2 参数量 vs §4 GPU 数） | 章内行 34/94/108 对照；论文 §5.4 原记号 = GPU 数 | **✗ → S2-3** |

### 跨章 sweep（8.1-8.3 抽查）

| 项 | 结论 |
| --- | --- |
| 指针双向性：8.1↔8.4 baseline（8.1 §5.3「同 prompt 组均值→8.4」对 8.4 §1/§2 回引 8.1 无偏性定理）；8.2↔8.3 GAE（8.2 §6.2「clip 归 8.3」对 8.3 多处「GAE（8.2）」）；8.4↔8.5（8.4「四模型账归 8.5」对 8.5 §2/§5/§8 回引 8.4）；3.2 §5 在线/离线定义被 8.1 §8、8.3 §1、8.5 §5 一致使用 | ✓ 全部双向、口径一致 |
| 符号约定（spec §4.3）：G 专归 GRPO 组大小——8.4/8.5 合规（8.5 并行度数全用 d/t/p）；8.1 的 G_t 为回报（Sutton-Barto 标准记号，出现章无 GRPO 场）——登记观察，不判违规；ε（clip）/λ（GAE）/δ（TD 残差 vs TRPO 阈值，8.3 §3 显式消歧） | ✓ 合规 |
| 公式回指：8.4 §1「8.3 式 (10)」→ 8.3 式 (10) 即 CLIP+VF+S ✓；8.4 §3「8.3 §8 k1/k3 诊断」→ 8.3 §8 old_approx_kl/approx_kl ✓ 一致 | ✓ |
| rubric A/B 扫描（5 章）：禁令词、松散语气词、含糊量词（一些/很多/大量/较快/略微/显著）机器扫描 | ✓ 零命中（8.1「才显著」一处经人工复核为定量语境下的合法用法，不报） |
| GitHub 数学 flanking 机械扫描（开口前字符白名单、闭口后 ASCII 限制、闭口前空格），5 章 | ✓ 零命中（8.2 行 131 两处 `**$\gamma$-just` 为 `**` 元素边界白名单内的扫描器误报） |
| 篇幅：8.3 5695 / 8.4 5576（推导重章软目标 ≥5500）✓；8.2 3717 / 8.5 4200（≥3500）✓；8.1 5492，距软目标差 8 字——软目标，登记接受 | ✓ |
| 证据等级与 references.md 一致性（8.4/8.5 全部引用 key 逐条核对）：deepseekmath/deepseek-r1/dr-grpo/dapo/gspo [A]、schulman-kl/trl-repo 教学参照、verl/openrlhf/vllm/megatron-lm/fsdp/zero [A]、ppo [B]；arXiv ID 与标题经 API 复核；各论文首作者（Shao/DeepSeek-AI/Liu/Yu/Zheng/Kwon/Hu(OpenRLHF)）与引用姓氏一致——**除 S1-1** | ✓（除 S1-1） |

**覆盖率声明**：8.4/8.5 的全部编号公式、全部数值断言、全部 repo 路径与具名函数、全部引文作者均逐项核对（上表 60+ 行）；8.1-8.3 为 sweep 抽查（指针、符号、文风、渲染纪律、回指一致性），其逐条事实核属批 1 范围。

## 4. 机械检查

```
bash scripts/lint-chapter.sh training/08-rl-math/0{1..5}-*.md
01-policy-gradient.md        LINT OK
02-actor-critic-and-gae.md   LINT OK
03-ppo.md                    LINT OK
04-grpo-and-variants.md      LINT OK
05-llm-rl-systems.md         LINT OK
```
（2026-11-18 本机复跑，全 5 章通过；与两批写作 report 自报一致。）

## 5. 写作方存疑项独立裁定

task-27（8.3/8.4 写作）中属本批范围者：
1. **Dr.GRPO「无偏」口径对齐**（存疑 #4）：独立核对附录 A——实际路线为 RLOO 等价 + G/(G−1) 吸收进 lr，章内结论（常数因子、方向不偏、式 (6) 为准）正确；机制转述欠准 → S3-1，建议一句改准后接受。
2. **GSPO 组大小 G 未声明**（存疑 #5）：原文已核段落确无 G 值，章内未声明——纪律正确，接受现状。
3. **DAPO 失效模式对照为本书推演**（存疑 #6）：章内以「失效模式对照：」引导、与原文口径句分隔——标注充分，接受。

task-28（8.5 写作）：
1. **外层 DP 网格分解**（存疑 #1）：与论文 Figure 8(b)/Table 2 算术自洽，系本书读法未标记 → S3-4，补标记后接受。
2. **veRL repo 演化**（存疑 #2）：章内已带核实日期与「以路径与函数名为准」时效注记；本审查 2026-11-18 于 main 分支复核全部命中，接受。
3. **verl 作者字段**（存疑 #3）：裁定属实且超出「未改 references.md」的免责范围——错误已在 6.3/8.5 正文 → S1-1，必修。
4. **OpenRLHF 1.22–1.68× 摘要口径**（存疑 #4）：现行 abs 摘要原文命中（"speedups ranging from 1.22x to 1.68x"），接受。
5. **80% profile 配置归属**（存疑 #5）：章内已带「LLaMA2-7B + A100」配置限定，未外推，接受。

## 6. 放行条件

必修（S1/S2）：
1. S1-1：references.md `verl` 作者字段 → "Sheng et al."；8.5（5 处）与 6.3（1 处）引用键 → [Sheng et al., 2024]。
2. S2-1：8.4 §8 表 DAPO 行 KL 列改「移除 KL（其 §2.3）」；GSPO 两行同步改为原文口径。
3. S2-2：8.5 补式 (3) 编号或改表内引用。
4. S2-3：8.5 §4 GPU 数记号与 §2 参数量 N_a 消歧。

S3 清零判定：S3-1/S3-2/S3-3/S3-4 建议随本轮一并修复（均为一句级改动）；8.1 G_t 观察项判接受、8.1 篇幅差 8 字判接受（软目标）。

修复完成后本批可放行；S1 修复涉及 references.md 与 6.3（W3 已审结章），按修复闭环惯例由修复方一并提交、evaluator 复核引用键一致性即可，无需重审 6.3 全文。

## 修复复审记录（2026-11-18，复审轮）

复审方式：重读 8.4/8.5 当前全文相关节 + references.md `verl` 条目 + 6.3 §5 与参考列表（非 diff），逐条对照当前文本取证。

| finding | 判定 | 当前文本证据 |
| --- | --- | --- |
| S1-1（verl 作者张冠李戴） | **ADDRESSED** | references.md 行 124 作者字段 = "Sheng et al."；8.5 行 3/85/94/144/203 全部改引 [Sheng et al., 2024]（arXiv:2409.19256），同章 OpenRLHF 保持 [Hu et al., 2024]（2405.11143），同名碰撞消除；6.3 行 62 veRL → [Sheng et al., 2024]、行 63 OpenRLHF 保持 [Hu et al., 2024]、参考列表行 129/130 同步 |
| S2-1（DAPO 行 KL 列） | **ADDRESSED** | 8.4 §8 表 DAPO 行 = 「移除 KL（其 §2.3）」；GSPO/GSPO-token 两行 = 「原文略去 KL（for brevity）」；表注新增「DAPO 的移除是显式设计决定…GSPO 的略去是行文简化而非设计主张（其 §2 口径）」——与两原文 §2.3/§2 逐字口径一致，且与本章式 (10)（无 KL 项）自洽 |
| S2-2（式 3 悬空） | **ADDRESSED** | 8.5 §4 的 $d_g = tp/(t_g p_g)$ 已编号式 (3)；§7 表改引「式 (3) 与论文 Table 2」；§4 符号表内「按式 (3) 代入即等」——全章式号 (1)(2)(3)(4) 连续无断链 |
| S2-3（N_a 一章两义） | **ADDRESSED** | 8.5 §4 改用 $N_g = tpd$ 表 GPU 总数，符号表显式消歧：「$N_g$：actor 训练与生成共用的 GPU 总数（HybridFlow 论文原记号 $N_a$，与本章 §2 的参数量 $N_a$ 无关，此处换字母避歧）」——§2 参数量 $N_a$ 用法（式 (1)、总结表、§8）不受影响 |
| S3-1（Dr.GRPO 附录 A 转述） | **ADDRESSED** | 8.4 §5 现作「其附录 A 的无偏性论证走 RLOO 等价链——G/(G−1)·Ã = Â^RLOO（展开约简后即 leave-one-out 形式），常数因子 G/(G−1) 可被学习率吸收（原文口径）」，并补出与式 (6) 互为倒数的联系——与附录 A 实际推导逐式一致（复审时对照 2503.20783 ar5iv 附录 A 原式复核） |
| S3-2（RLOO 全称） | **ADDRESSED** | 8.4 §2 行 92：「RLOO（REINFORCE Leave-One-Out）一族算法」 |
| S3-3（式号体例） | **ADDRESSED** | 8.5 总结表三处均改「式 (1)」「式 (2)」「式 (4)」带括号体例 |
| S3-4（外层 DP 本书读法） | **ADDRESSED** | 8.5 §4 符号表末「…外层 DP（本书读法）」已标记 |

**新引入问题检查**（只查修复触及处）：8.4 行 144 新句的 G/(G−1) 与 (G−1)/G 倒数关系算术正确、Â^RLOO 记号就地定义、公式 backtick 包裹合规；8.5 的 $N_g$ 首现即定义、开口 flanking 合规；8.4 表注新增句无渲染风险。机器复扫三章 flanking（开口前字符/闭口后 ASCII/闭口前空格）零命中。未发现新引入问题。

**lint 复跑**（2026-11-18）：`04-grpo-and-variants.md` LINT OK；`05-llm-rl-systems.md` LINT OK；`06-megatron-and-stacks/03-modern-training-stacks.md` LINT OK。

**复审结论：全部 findings（S1×1、S2×3、S3×4）ADDRESSED，无新问题——W5 批 2 放行。**
