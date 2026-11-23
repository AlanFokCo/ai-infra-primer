# Inference W2 对抗审查 — P4 调度与 batching（4.1–4.5）

- 日期：2026-11-23
- 审查人：fresh-context adversarial evaluator（不继承写作上下文）
- 范围：`inference/04-scheduling-and-batching/{01-continuous-batching,02-scheduler-and-preemption,03-chunked-prefill,04-speculative-decoding,05-speculative-systems}.md`
- 依据：evaluator-brief（W1 通用协议）+ W2 dispatch；spec §5 P4 大纲 / §4.1 / §4.3 / §6.3；`inference/references.md`（W2 升级后等级）；`docs/research/inference-w2-sources.md`；STYLE_GUIDE / R2 / rubric。
- 方法：全部公式与数值独立重算；强制重推导独立重推；一手来源自行核对（arXiv HTML/txt 存档、USENIX PDF/HTML、GitHub codeload tarball 路径级、DeepInfer doc、hf-mirror config）；lint + 禁令词 + R2 逐条 + 习题分级核对。**只读审查，未改章节、未 commit、未派 subagent。**

## Status：PASS with findings（S1=0 / S2=4 / S3=7）。强制重推导核心公式全 PASS。

章节整体质量高：五章 lint 全 LINT OK；rubric A 禁令词 0 命中、无历史叙事、无松散语气词；全部被引 arXiv ID↔标题↔第一作者与 references 一致；全部被引论文数值/引文逐字命中（Leviathan Def 3.7/Thm 3.8/Cor 3.9/Thm 3.11、SpecInfer Table 1/Def 4.1/MSS Algorithm 2、FastServe 36ms/60ms/2.3GB/7×/31.4×/17.9×/300ms、Orca 36.9×/C2§3/selective batching、vLLM §4.5/§7.3 20%、Sarathi 三因素、SGLang 饥饿/DFS 最优、Llumnix 摘要 1.5×/36%/数量级 + README 4.6×/9.1×/15%、DeepInfer doc L137/139/143/303 + DSpark 增益表）；全部引擎 repo 文件/行号锚点路径级核实存在（含 research notes 未列的 SGLang `target_verify.py`/`spec_sampling_mask.py`/`online_acceptance_estimator.py`/`adaptive_spec_params.py` 四件均实在）；强制重推导 8 组核心公式独立重推全对。findings 集中在**命题陈述精度**（4.4 加速比 iff 量词、MSS 采样假设）、**一处机制误标**（4.5 EAGLE-1 feature 层）、**一处数值不自洽**（4.2 Llumnix 12×），及若干指针/口径 minor。无 S1，无需 S1 复审。

## Findings 表

| # | severity | 位置（file:line / §） | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | 04-speculative-decoding.md L131 / L141 / 总结 L240（命题 3） | 「$F>1$ 当且仅当 $\alpha>c$」——对**固定** $\gamma$ 不成立 | 独立重算反例：$\alpha$=0.8、$c$=0.7、$\gamma$=4 → $F=(1-0.8^5)/((0.2)(4\times0.7+1))=0.67232/0.76=0.885<1$，虽 $\alpha>c$。Leviathan Cor 3.9 原文（leviathan.txt L315）是「**If α>c, there exists γ** for which we'll get an improvement…」——存在量词。本章证明正文实际证的也是存在版（「取 γ=1 即满足，故**存在 γ** 使 F>1」），但命题陈述与总结表漏「存在 γ」。 | 命题 3 与总结表改「**存在 γ** 使 $F>1$ 当且仅当 $\alpha>c$」；可补「固定 γ 时 $F>1\iff\sum_{k=1}^{\gamma}(\alpha^k-c)>0$（式 (9)）」。公式 F 本身正确，仅量词缺失。 |
| F2 | **S2** | 04-speculative-decoding.md L193–195（命题 5，MSS 残差保持） | 题面未声明子候选**按 $x_s\sim q_s$ 采样**；按确定性 top-k 子候选读，所述接受规则不保分布 | 证明把接受贡献写成 $\min(p,q_1)$（向量），这**隐式假设** $x_1\sim q_1$ 采样（同命题 1「draft 采 x~q」）。若子候选为确定性 token（greedy expansion 的 top-k 即如此），接受质量是 $\min(1,p(x_s)/q_s(x_s))$ 的点质量而非 $\min(p,q_s)$。反例：$p$=(0.5,0.3,0.2)、$q$=(0.4,0.4,0.2)、确定性 top-2={a,b}：a 恒被接受（p(a)/q(a)=1.25>1），c 不在候选也不在残差 max(0,p−q)=(0.1,0,0) → P(c)=0≠0.2，分布破坏。另 SpecInfer Algorithm 2（specinfer.txt L470+）用 `s~rand(H)` **随机序**，非题面「任意固定顺序」（结论对任意序成立，但与原文算法不同）。 | 题面显式加「设子候选按 $x_s\sim q_s$ 独立采样」（与命题 1 对齐）；注明原文用随机序 rand(H)、本命题对任意固定序亦成立。MSS 随机解码本身保分布（Thm 4.2）✓，证明代数正确，仅题面采样假设缺失。 |
| F3 | **S2** | 05-speculative-systems.md L91（§5） | 「EAGLE-1 用**末层** feature」与本章 §1 及 EAGLE 论文矛盾 | §1 L7 明言 EAGLE-1 用「second-to-top-layer feature（**倒数第二层**隐藏态）」；EAGLE 论文/repo 同（research notes 4.5：second-to-top-layer）。L91 把参数名 `last_hidden_states` 误读为「末层 feature」。vLLM `get_drafter_hidden_states`（spec_decode/utils.py L22）对 EAGLE 返回 target 隐藏态，EAGLE-1 消费的是倒数第二层。 | L91 改「EAGLE-1 用**倒数第二层** feature（经 `last_hidden_states` 参数传入）、EAGLE-3 用多层融合 feature（`aux_hidden_states`）」。单短语误标，§1 正确。 |
| F4 | **S2** | 02-scheduler-and-preemption.md L114 / 总结 L189（§4 Llumnix） | 「迁移便宜约 **12×**」与式 (4) 分量不自洽 | 式 (4) $t_{\mathrm{mig}}=\kappa LT/\beta_{\mathrm{net}}+t_{\mathrm{last}}$；代入批量段 10.7 ms + 停拷 2.7 ms = **13.4 ms**，recompute 133 ms → 133/13.4 = **9.9×**（≈10×）。若按「暴露仅尾段」口径 133/2.7 = **49×**。12× ≈ 133/10.7（只除批量段、漏停拷），两读法均不匹配。 | 改「约 10×」（总传输+停拷 13.4 ms 口径）或显式声明比较基准；总结表同步。数值分量（10.7/2.7/133）本身正确，仅比值与分量不符。 |
| F5 | S3 | 03-chunked-prefill.md L65（§2） | 指针「1.2 **§4**『凑 batch 收益递减』」 | 该短语在 1.2 **§3**（02-prefill-decode-accounting.md L91，decode 算例段）；1.2 §4 是「临界 batch B*」。 | 改「1.2 §3」。 |
| F6 | S3 | 02-scheduler-and-preemption.md L66（§2） | 指针「与 **2.4 §3** 的账区分口径」 | 取回 vs 重算临界带宽账在 2.4 **§2**（04-kv-offload-and-tiering.md「## 2. 分层命中成本账」命题 1）；2.4 §3 是「传输带宽需求」。因子 2 论断本身正确（往返 vs 单向）。 | 改「2.4 §2」。 |
| F7 | S3 | 02-scheduler-and-preemption.md L144（命题 3 防饥饿推论） | 「新到达者恒从 1 起步、**无法插队到它前面**」作为永久论断不严谨 | 新到达者初始 ratio=1 确在等待者之后，但 $U_Y<U_X$ 的新请求 ratio 增速更快（$1/U_Y>1/U_X$），等待足够久后可反超。推论「有限时间内到队头」在**稳定负载**下成立（标准 HRRN 结论），但论证未排除持续小 $U$ 到达的反超。命题 3 (i)(ii)(iii) 本身证明正确。 | 推论补一句限定（「稳定负载下各请求等待有界，新到达者虽可后续反超但每条反超者随即被服务离队，故不永久饥饿」）或弱化为「新到达者不能**立即**插队」。结论标准、可修。 |
| F8 | S3 | 03-chunked-prefill.md L145（§3 下界） | 「$\tau_{\mathrm{step}}\ge10\tau_{\mathrm{oh}}$ 代入得 $c_{\mathrm{chunk}}\ge594$」 | 594 = 仅 GEMM 项解（$9/0.015159=593.7$）；含 chunk 自 attention 项的完整 $\tau_{\mathrm{step}}$ 解 ≈ **588**（$0.015159c+2.649\times10^{-7}c^2+1=10$）。差 ~1%，「数百 token 量级」结论不变。 | 注明 594 为略去自 attention 的 GEMM 项近似（完整解 ~588），或改 588。极 minor。 |
| F9 | S3 | 04-speculative-decoding.md L171–173（命题 4，tree mask 等价） | 证明缺 RoPE 位置前提 | 证明基例「可见集相同 + 输入 embedding 相同 → 第 1 层输出相同」对 RoPE 还需**节点位置编码 = 路径深度**（Q/K 含相对位置）。题面前提只列 mask + 前缀 KV 缓存。实现（SpecInfer topology-aware mask / FlashInfer）按深度赋位置，结论成立。 | 题面补一句「节点位置编码取路径深度（与独立序列一致）」，证明基例引用之。 |
| F10 | S3 | 01-continuous-batching.md L122（命题 3） | $1/\eta$ 增益隐式纯 decode 口径未声明 | 证明关键步「continuous 每步产出 B 个有用 token」要求 B 条全在 decode；新准入请求首步 prefill 不产 output token。static 侧同样略 prefill，故比值近似保持，但口径未声明。 | 命题 3 加口径注「纯 decode 口径（忽略新请求 prefill 步；prefill 混入时增益低于 $1/\eta$）」。结构结论正确。 |
| F11 | S3 | 03-chunked-prefill.md L59（§2 piggyback 边际账） | 节省 3.9 ms 用免 $\tau_{\mathrm{oh}}$ 基线（15.1），而迭代模型含 $\tau_{\mathrm{oh}}$=1 ms | 边际 11.2 = 混合 20.0 − 纯 prefill 8.8（oh 相消）；独立 decode 取 1.2 带宽下界 15.1（免 oh）→ 节省 3.9。两边同免 oh，内部自洽（B_D=512 反例同口径）；但「独立 decode iteration」按本章模型应为 16.1（含 oh），节省 4.9。口径选择未声明。 | 可选：加一句「边际账两边均略固定开销 $\tau_{\mathrm{oh}}$」口径注。数值自洽，仅清晰度。 |

> severity 裁定说明（沿 W1 P1 校准）：F1–F4 为可一手/独立重算证伪的局部数学/事实/数值错（brief S2=局部事实错/推导断裂），虽均可单点修复仍按 S2 计；F5–F11 为指针/口径/证明清晰度级（结果正确或已声明），按 S3。F2 处命题陈述精度边界（采样假设缺失致确定性读法可证伪），按 S2；writer 如主张「MSS 随机解码语境下采样为默认读法」可附证据走反驳通道复核。

## 强制重推导（独立重推，不信任 writer）— 核心公式全 PASS

- **4.2 recompute vs swap 临界带宽 β\***：$t_{\mathrm{recompute}}=(2NT+2LT^2d_{\mathrm{model}})/F_{\mathrm{eff}}$、$t_{\mathrm{swap}}=2\kappa LT/\beta_{\mathrm{pcie}}$；交叉相乘解 $\beta^{\ast}=\kappa LF_{\mathrm{eff}}/(N+LTd_{\mathrm{model}})$ ✓，分母随 T 严格增 → β\* 严格减 ✓。数值独立重算：T→0 极限 8.64 GB/s、T=4096 → 8.07、T=32768 → 5.50 ✓；t_swap 16.8/134 ms、t_recompute 133 ms/1.56 s、比值 7.9×/11.6×、attention 占比 6.7%/36% 全对 ✓。与 2.4 单向账因子 2 ✓。**PASS**（F6 仅指针）。
- **4.3 stall 账**：prefill T=32768 FLOPs 4.92e14+2.81e14=7.73e14 → 781 ms ✓；decode B_D=64/T̄=4096 (W+64κLT̄)/BW=15.1 ms ✓；781/15.1=51.7≈52× ✓；MFU0.5 → 1.56 s/104× ✓；T=4096 → 66.5 ms/4.4× ✓。**PASS**。
- **4.3 chunk-size 权衡**：式 (1) $C_{\mathrm{attn}}=4Ld(c_{\mathrm{chunk}}c+c_{\mathrm{chunk}}(c_{\mathrm{chunk}}+1)/2)$ 与 1.2 式 (1) 同口径（causal 精确和，无双重折扣）✓；命题 1 FLOPs 守恒裂项证明独立重推 ✓（T(T+1)/2，等长/不等长均成立）；式 (4) 重读 κLT²/(2c_chunk) ✓（135/1097 GB、40.4/328 ms）；式 (5) I_attn=4c_chunk·d/κ=4c_chunk ✓、硬下界 74 ✓；式 (6) c_hi=(τ_SLO−τ_oh−τ_attn)F/(2N)−B_D → 2489/精确 2485 ✓；区间空判据 τ_SLO=20ms → c_hi≈510<512 ✓；权衡表 1338/1037/845/797/785 ms 与 TPOT 16.1/16.1/20.0/43.4/144.1 ms 逐格独立重算全对 ✓。**PASS**（F8 仅 594 近似）。
- **4.3 DeepInfer compute-quota 二次求解**：式 (7) u=2Ld/F_eff≈0.53 ns、v0=2N/F_eff≈30300 ns、v1=4Ld/F_eff≈1.06 ns（u=v1/2 ✓）；式 (8) 正根独立重算 quota=50ms：c=8192→1260（回代 49.96ms）、c=65536→499（49.92ms）✓；DP MIN 40ms：c=8192→1012、c=65536→400 ✓；prefix 命中 c=4096→1412（+12%）✓；32768 帽双约束 ✓。**PASS**。系数教学回算 ×3 显式声明 ✓。
- **4.4 分布保持证明**（命题 1）：Z=1−Σmin(p,q)=1−A；P(out=x)=min(p,q)+max(0,p−q)=p ✓ 独立重推完整；3-token 实例 β=0.9/TV=0.1 ✓；命题 2 β=1−TV 恒等式 ✓。**PASS**。
- **4.4 期望产出**：capped geometric P(N=k)=α^{k−1}(1−α)、P(N=γ+1)=α^γ；尾和 E=(1−α^{γ+1})/(1−α) ✓；α=0.8/γ=4 → 3.3616、E[K]=2.3616 ✓；口径注（含 bonus vs DSpark accepted=E[K]）✓。**PASS**。
- **4.4 加速比定理**：F=(1−α^{γ+1})/((1−α)(γc+1))，c=c_d/c_v **无量纲** ✓；绝对成本式 (10) ×c_v 归一量纲正确（不乘则 tokens/s）✓；γ 表 1.64/2.03/2.27/2.40/2.46/2.47/2.45/2.41 逐格重算 ✓、最优 γ=6 ✓；c=0.3→γ3/1.55、α=0.5→γ2/1.46 ✓；n-gram α=0.2/γ3/c≈0 → 1.248≈1.25× ✓；Thm 3.11 运算量因子 (1−α)(γĉ+γ+1)/(1−α^{γ+1}) ✓（§6 3.79）。**公式 PASS**（F1 仅 iff 量词、F2 仅 MSS 题面）。
- **4.4 tree-verify 正确性**：命题 4 tree mask「可见集=祖先+自身」→ 逐路径等价独立 causal，对层归纳证明独立重推 ✓（F9 补 RoPE 位置前提）；命题 5 MSS 残差归纳代数正确（采样读法，F2）。**PASS（带 F2/F9 精度 finding）**。

## Writer 上报待核项裁定（独立核一手，不照单全收）

- **4.4 SpecInfer Table 1 覆盖率 68/77/81/84/85%**：**PASS**。specinfer.txt L606+ Table 1「success rate of verifying a token for LLaMA-7B using top-k from LLaMA-68M」，Greedy/Alpaca 行 k=1..5 = 68/77/81/84/85% 逐字 ✓；§3 ✓。「nodes each represent a candidate token sequence」L279 ✓；Def 4.1 TreeAttention(u)=Attention(S_u) L792 ✓；「computes the exact same attention output as incremental decoding」L835 ✓；DFS 更新共享 KV §4.2 ✓；「guarantees equivalence」L331 ✓。
- **4.4 Leviathan §3.6 n-gram α=0.2→1.25×**：**PASS**。leviathan.txt §3.6 逐字「T5-XXL 11B + trivial bigram, EnDe, α≈0.2 … 1.25X with γ=3」✓；§3.5 oracle 1/(1−α)、「up to ~60% higher」✓（4.5 §6 引）；Def 3.7/Thm 3.8/Cor 3.9/Def 3.2/Lemma 3.3/Thm 3.5/Def 3.10/Thm 3.11 编号与陈述全 ✓；「c always less than 0.05」§3.3 ✓；「two orders of magnitude smaller … usually performed best」§3.6 ✓；T5-XXL 2X-3X abstract ✓。c≈1/L memory-bound 教学估算在 4.5 §1/§3/§4 显式声明 ✓。
- **4.3 DeepInfer 拟合系数未披露 → 教学回算**：**PASS**。doc L143 逐字命中（compute_quota_limit ns / cost(q,c) 二次 / 32,768 帽 / DP all-reduce MIN）；章节 3 处显式声明教学回算值（「doc 未披露拟合系数…教学回算值，不是生产数值」「不可把教学值当 DeepInfer 参数引用」「四舍五入的教学值」）✓；MIN 机制解读标「doc 只给操作，机制解读如下（结构论证）」✓；L137 四档 / L139 pull 逐字 ✓。
- **4.3 Sarathi 无论文性能数引用**：**PASS**。章节引 Sarathi 仅机制（chunked-prefills/decode-maximal/三因素 GPU 内存·参数内存·最大序列长/PP bubble），全部数值（781/15.1/20.0/权衡表）为章脊推导，无张冠李戴论文性能数 ✓。sarathi.txt L313 三因素逐字 ✓。两相 roofline 教学抽象在式 (2) 显式声明（「教学两相集中模型，无 intra-device 重叠」）✓。
- **4.1/4.2 FastServe 36ms 论文原数**：**PASS**。fastserve.txt L706–708 逐字「2.3 GB … decoding ~60 ms … PCIe 4.0×16 full bandwidth ~36 ms」（OPT-175B 16×A100）✓，非反推带宽；Fig.8 OPT-2.7B/输出20/7× ✓；skip-join q_i≥t_init Algorithm 1 ✓；饥饿 α 默认 300ms ✓；31.4×/17.9× 摘要 ✓；2.9K Python+8.1K C++/CUDA/Ray/SwiftTransformer ✓；NSDI'26 标题逐字 ✓。
- **4.1 Orca C2 §3 / 36.9×**：**PASS**。orca PDF C2 在 §3「Challenges」✓；selective batching 引文 §1、token-wise §4/Fig.5 ✓；36.9× = **175B** vs FasterTransformer 同延迟（L805–809 + 摘要）✓，[B] + 系统级口径注 ✓；USENIX abstract 逐字命中 iteration-level 引文 ✓；FCFS/Algorithm 1 max_tokens 预留 §4.2 ✓。
- **4.2 SGLang 9 策略全集 + 旧 5 子集注 + 行号**：**PASS**。schedule_policy.py L188–204 CacheAware 5（LPM/DFS_WEIGHT/HRRN/SPF/UNIBOOST）+ CacheAgnostic 4（FCFS/LOF/RANDOM/ROUTING_KEY）✓；BOUNDED_LPM_WINDOW_SIZE=64 L181 ✓；HRRN ratio=1+waited/U、UNIBOOST Φ=arrival−b(uU)、max(work,1) L942、gamma 10.0 / alpha 64μs environ L727 全 ✓；旧五项子集注 ✓。
- **4.5 Medusa typical acceptance 不保分布（反例）**：**PASS**。命题 2 反例 p(a)=0.6/q(a)=1/τ=0.5 → 输出 (1,0)≠p 正确 ✓；Medusa README typical acceptance「longest plausible prefix」/Medusa-1/2/2.2–3.6× ✓。
- **4.5 EAGLE-1/2/3 机制**：**PASS（带 F3）**。EAGLE-1 feature+shifted embedding、EAGLE-2 confidence 剪枝（命题 1 前缀封闭证明正确）、EAGLE-3 training-time test+多层融合（cnets.py fc=3×hidden L532/534、midlayer L530、requires_grad=False L544）✓；唯 §5 L91「末层 feature」误标（F3）。
- **4.5 MTP nextn V3 1→V4.1 3 [W]**：**PASS**。v41-config-w2.json num_nextn_predict_layers=3 / num_hidden_layers=40 ✓；V3 nextn 1 [A deepseekv3] ✓；MTPSpeculator(TargetDependentARSpeculator)/load_eagle_model ✓。

## 跨章一致性

- **符号 §4.3**：α=接受率、γ=draft 长度、B=batch 全 batch 一致 ✓；4.2 FastServe 饥饿阈值 α 显式避冲突改文字 ✓、UNIBOOST gamma/alpha 译 g/u ✓；4.3 c_chunk + c=前缀长章首局部声明（「本章 c 是前缀长，不是 CP 度」）✓；4.4 c=c_d/c_v 无量纲 + ĉ 运算量比 ✓。无保留字母冲突。
- **attention 系数**：4.3 式 (1) 与 1.2 式 (1)（2LT²d，causal 已含）对齐，显式「1.2 系数 2 已含 causal…本章不再另打对折」+ 坑注 ✓（W1 裁定遵守，无双重折扣）。
- **命题编号**：逐章重启（4.1:1–3 / 4.2:1–3 / 4.3:1–2 / 4.4:1–5 / 4.5:1–2）✓。
- **等级升级**：specinfer [A]（非 [A/B]）、fastserve [A]（非 [B]）按 references W2 裁定使用 ✓；marlin [A] 属 P3 不在本批。
- **章脊口径**：4.1/4.2/4.3 复用 1.2/2.1/2.4 章脊（N=7.50e9 matmul、W=1.61e10、κL=128KiB、H100 989.5/3.35、F_eff=50% 教学假设）✓；4.4/4.5 换口径逐节声明 ✓。recap 指针 1.1 命题 1/式 (3)、1.2 式 (1)(2)(4)、1.3 式 (1)、2.1 命题 1/式 (2)/§4、2.4 §2 核实（F5/F6 两处节号误）。

## 规范 / 教学 / 习题

- lint 五章全 LINT OK；rubric A 禁令词 0、历史叙事 0、松散语气 0；术语首现判断句定义 + 英文保留 ✓；证据等级标注齐全 ✓。
- R2：节首 crux/设问/数值开场逐节合规（R2-1）；章脊锁定 + 换口径声明（R2-2）；公式后数字实例（R2-3）；命题-证明骨架 + 路线句（R2-4/5）；定义配正例 + 边界反例（R2-6）；口径注（R2-7）；坑注/提点块引用（R2-10/11）；总结表回指列（R2-12）✓。
- 习题分级：五章均 [10]/[20]/[30]/[30]（4.4 含 [40] 带 (a)(b)(c) 脚手架）非递减、≥1 题 [10]/[20]、≤1 题 [40]、每计算/代码题校验锚、≥1 题以正文命题为证明目标（4.1 练2→命题2、4.2 练2→命题1、4.3 练2→命题2、4.4 练3→命题5、4.5 练3→命题1）✓。校验锚独立核对可校验（4.1 练3 192/768 与 2.1§1 一致、4.2 练3 Φ_A≈−0.073/Φ_B≈−0.008 重算 ✓、4.3 练4 1260/499/1012 ✓、4.4 练1 1.64 ✓）。

## Verification log（抽查项 × 来源 × 结论）

| 抽查项 | 来源（一手） | 原文引句 / 数值 | 结论 |
| --- | --- | --- | --- |
| Leviathan Def 3.7 / Thm 3.8 / Cor 3.9 | leviathan.txt L298/302/315 | 「Let c, the cost coefficient…ratio…M_q…M_p」；F=(1−α^{γ+1})/((1−α)(γc+1))；「If α>c, there exists γ…≥(1+α)/(1+c)」 | 逐字 ✓（F1 量词） |
| Leviathan Def 3.2/Lemma 3.3/Thm 3.5/Thm 3.11 | leviathan.txt L241/245/262/338 | D_LK=Σ\|p−q\|/2；β=1−D_LK；运算量因子 (1−α)(γĉ+γ+1)/(1−α^{γ+1}) | ✓ |
| Leviathan n-gram / oracle / 2-3× | leviathan.txt §3.6/§3.5/abstract L117 | 「α≈0.2…1.25X with γ=3」；「1/(1−α)…~60% higher」；「2X-3X…identical outputs」 | ✓ |
| SpecInfer Table 1 | specinfer.txt L606+ | Greedy/Alpaca k=1..5 = 68/77/81/84/85% | ✓ |
| SpecInfer Def 4.1 / MSS Alg 2 / DFS | specinfer.txt L792/470+/835 | TreeAttention(u)=Attention(S_u)；VerifyStochastic `s~rand(H)`+残差 norm(max(0,p−q_s))；「exact same attention output」 | ✓（F2 随机序/采样） |
| FastServe 36ms/60ms/2.3GB/7×/300ms/31.4×/17.9× | fastserve.txt L706/707/642/635/111 | 「PCIe 4.0×16…about 36 ms」「decoding…about 60 ms」「2.3 GB」「7× larger than FCFS」「α…300 ms by default」「31.4×…17.9×」 | 逐字 ✓ |
| FastServe skip-join / 实现 / NSDI'26 | fastserve.txt L488/847；nsdi26.html L433 | 「q_i≥t_init」；「2.9K…Python…8.1K…C++/CUDA…Ray actor」；「Iteration-Level Preemptive Scheduling…」 | ✓ |
| Orca 36.9× / C2§3 / selective / FCFS | orca.txt L805/291/117/489；orca-usenix.html L313 | 「175B…6.81/0.185…36.9×」；C2 §3；「splits the batch…Attention…batching to other」；abstract 逐字 | ✓ |
| vLLM §4.5 recompute / §7.3 20% | vllm-paper.txt L245/427 | 「concatenated…new prompt…one prompt phase iteration」；「recomputation overhead never higher than 20% of swapping's latency」 | ✓ |
| Sarathi 三因素 / 机制 | sarathi.txt L313/133 | 「GPU memory M_G…parameter memory M_S…maximum sequence length L」；chunked-prefills+decode-maximal+piggyback | ✓（无性能数引用） |
| SGLang 饥饿 / DFS 最优 | sglang-paper.txt L201/198 | 「greedy cache-aware scheduling…can lead to starvation」；「optimal cache hit rate…DFS…cache size≥max request length…longest-shared-prefix-first≡DFS」（A.3） | ✓ |
| Llumnix 摘要 / README / migration | arxiv-p3p4-batch.xml L298；README L95/47；default.py L225/227 | 「order of magnitude…1.5x…36% cost」；「4.6x/9.1x…15% P99 TBT」「near-zero overhead」；MIGRATION_MAX_STAGES=3/LAST_STAGE_MAX_BLOCKS=16 | ✓（F4 比值） |
| DeepInfer compute-quota / DSpark | doc L143/137/139/303 + 增益表 | L143 逐字；四档；pull；「acceptance probability…varies…fixed-length wastes」；60conc verified 3.58/+5.4%、72conc/+6.7%、accepted 3.29/3.20 | 逐字 ✓ |
| vLLM scheduler 行号 | vllm-main tarball | core.py step L714/step_with_batch_queue L759；scheduler.py schedule L702+注释 L705/num_new_tokens L824/victim L918-923/preempt L1041/_preempt L1831；request_queue SchedulingPolicy L13-17/Priority L131；config/scheduler L47/104/112/232 | 全 ✓ |
| SGLang scheduler/policy/speculative 行号 | sglang-main tarball | scheduler event_loop_normal L2231/overlap L2266/get_next L4386/run_batch L5455/init_chunked L1468；schedule_batch Req L442/ScheduleBatch L2007→2009/check_decode_mem L3007/retract L3030；schedule_policy 9 策略 L188-204/window L181/calc_priority L279/HRRN L822/UNIBOOST L930-976/max(work,1) L942；environ L727=64.0；runtime_context get_schedule L1351；eagle_worker_v2/ngram_worker/cpp_ngram/dspark_components + target_verify/spec_sampling_mask/online_acceptance_estimator/adaptive_spec_params 均存在 | 全 ✓ |
| EAGLE cnets / vLLM spec_decode | EAGLE tarball cnets.py L387/530/532/544；vllm spec_decode/ | LlamaDecoderLayeremb/midlayer/fc=3×hidden/requires_grad=False；propose last_hidden_states+aux_hidden_states（num_layers×[num_tokens,hidden]）；MTPSpeculator(TargetDependentARSpeculator)/load_eagle_model；rejection_sampler/acceptance_estimator/adaptive_verification/eagle3_utils 存在 | ✓（F3 末层误标） |
| Medusa / REST / Draft&Verify / sarathi-serve / FlexFlow | 各 tarball README/源码 | Medusa 2.2-3.6×/Medusa-1/2/typical acceptance；REST datastore/Trie/prune/plug-and-play；searching.py/skip_layers.json；sarathi/orca/vllm/faster_transformer provider + benchmark README 2308.16369/enable_dynamic_chunking_schedule；FlexFlow inference 分支 spec_infer.cc/TREE_VERIFY_MODE | 全 ✓ |
| V4.1 config | v41-config-w2.json（hf-mirror） | num_nextn_predict_layers=3 / num_hidden_layers=40 | ✓ |
| 强制重推导 8 组 | 独立重算（见上节） | β\*/stall/c_hi/二次求解/分布保持/E[N_tok]/F/tree mask | 全 PASS |
| lint / 禁令词 / 习题分级 | scripts/lint-chapter.sh + grep | 五章 LINT OK；rubric A 0 命中；分级合规 | ✓ |

**覆盖率**：公式与数值实例 100% 独立重算；强制重推导 8 组独立重推；被引一手来源 ~98% 自行核对逐字（论文 HTML/txt 存档 + repo tarball 路径级 + DeepInfer doc + config；Orca PDF 经 orca.txt 提取 + USENIX HTML 双核）。全部被引 arXiv ID、论文数值/引文、引擎 repo 文件/行号（含 research notes 未列四件 SGLang 文件的存在性）核实。无「无 log 的确认无误」。

## 修复建议汇总

- S2 F1–F4 必修复：F1 加「存在 γ」量词（命题 3 + 总结）；F2 命题 5 题面加采样假设 + 随机序注；F3 L91 改「倒数第二层 feature」；F4 Llumnix 比值改 ~10× 或声明基准（正文 + 总结）。
- S3 F5–F11 累计下波清零；F5/F6（指针节号）/F9（RoPE 前提）/F10（纯 decode 口径）建议本波顺手修（清晰度收益高）；F7/F8/F11 可选。
- 含 S2 章节修复后 scoped 复核（F1/F2 → 4.4，F3 → 4.5，F4 → 4.2）。