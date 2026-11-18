# 7.5 长上下文扩展与中后期训练（Long-Context Extension and Mid-Training）

> 预训练尾段有两笔专项预算：一笔把窗口从几千 token 扩到十万级（staged extension，分几段逐级抬窗而不是一步到目标窗），一笔把数据配比推向数学 / 代码 / 长文档——GLM-4.5 TR 把后者命名为 mid-training。staged extension 的配方本体已在 models 4.6 §4 交付（Llama-3.1 六阶段 8K→128K 约 800B tokens、DeepSeek-V3 两阶段 + YaRN），本章不重复；本章补三件事：mid-training 与 post-training 的分界判据、扩窗的 training 侧成本账（tokens 预算 / 位置分布 / CP 启用时机）、「训练长度 ≠ 推理外推长度」的口径纪律 | 前置：5.2（CP 通信账与 causal 负载均衡）、7.1（数据配比、packing 与课程）、2.3（LR schedule 形态）；models 4.6（长上下文四堵墙与配方——纯文字指引） | 锚点：DeepSeek-V3 TR（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、Llama-3 herd paper（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）、GLM-4.5 TR（[GLM Team, 2025](../references.md)，arXiv:2508.06471，[A]）、Qwen2.5-1M TR（[Qwen Team, 2025](../references.md)，arXiv:2501.15383，[A]）、YaRN（[Peng et al., 2023](../references.md)，arXiv:2309.00071，[A]）

## 1. 尾段预算的量级：两个实测数

DeepSeek-V3 的 Table 1 把长上下文扩展单列成一行账。预训练 14.8T tokens 花 2664K H800 小时；扩窗（4K→32K→128K）花 119K；post-training 只花 5K（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）。Llama-3 的结构相同：405B 主阶段 15.6T tokens 全程 8K 窗，随后约 800B tokens 分六个阶段扩到 128K（herd §3.4.2，[A]），扩窗预算约为主阶段的 5%。

为什么都把长序列训练压到预训练末尾？herd 原文给出理由："we do not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length"。这笔账在 §3 展开成公式。

这两笔尾段预算，各家叫法不同：Llama-3 把它们并入预训练日程的收尾段（final stages），GLM-4.5 单独命名为 mid-training。本章先给命名与分界（§2），再给账（§3-§4）。

## 2. mid-training：预训练与后训练之间的第三带

GLM-4.5 TR §2.3 的标题就是定义："Mid-Training: Boost Reasoning & Agentic Capacity"。原文口径照录："After pre-training, we add several stages to further boost the model's performance on important application areas... these training stages utilize medium-size domain-specific datasets, including instruction data. Therefore, we denote these training stages as mid-training"（[GLM Team, 2025](../references.md)，arXiv:2508.06471，[A]）。它由三部分组成（§2.3 原文结构）：

- **repo 级代码训练**：同一仓库的文件拼接进单一上下文，学跨文件依赖；外加模型过滤的 issue / PR / commit（diff 式组织）。序列长度为此从 4K 抬到 32K，RoPE base 从 $10^{4}$ 调到 $10^{6}$；
- **合成推理数据**；
- **长上下文与 agent 训练**：窗长从 32K 抬到 128K，从预训练语料上采样长文档，并混入大规模合成 agent 轨迹。

mid-training 不是凭空加出来的阶段。GLM-4.5 的预训练本身已分两段：第一段以网页通用文档为主，第二段上采样 GitHub 源码与编码 / 数学 / 科学类网页（§2.2 口径）。「数学 / 代码配比上调」在预训练内部已经开始，mid-training 把它推到专项强度。另有两处配套口径。其一，packing 策略分段：预训练用随机截断（random truncation 兼作数据增广），mid-training 换 best-fit packing，避免截断推理链与 repo 级代码（§2.3；packing 机制本体归 7.1）。其二，LR 用一条 cosine 从峰值 $2.5 \times 10^{-4}$ 衰到 $2.5 \times 10^{-5}$，"until the end of mid-training"（§2.4）——预训练与 mid-training 共用同一条 schedule。

**与 post-training 的分界判据**。GLM-4.5 的 mid-training 已混入 instruction data，所以数据形态不能当分界。真正的判据是另外三条：

1. **loss**：mid-training 全程仍是 next-token 续训；post-training 换成 response-only 条件似然（masking 归 3.2）或偏好 / RL 目标（P8-P9）；
2. **schedule 连续性**：mid-training 走在预训练同一条 LR schedule 的收尾段上——GLM-4.5 的 cosine 末端、DeepSeek-V3 扩窗 LR $7.3 \times 10^{-6}$ 恰等于预训练末 LR（§4.3 原文 "matching the final learning rate from the pre-training stage"）、Llama-3 annealing 段 LR 线性归 0；post-training 另起 warmup 与小 LR（归 9.1）；
3. **量级**：mid-training 以十亿至数千亿 tokens 计（10^9–10^12 下沿；本书实例：DSv3 约 126B、Llama-3 约 800B）；SFT 以百万样本计，RL 以 prompts 与 rollout 计。

命名以 GLM-4.5 为锚，各家的结构大同小异。Llama-3 不用 mid-training 一词，对应物是 §3.4.2 的长上下文段加 §3.4.3 的 annealing：最后 40M tokens 上把 LR 线性退到 0，保持 128K 窗，上采样极高质量数据源，并对 checkpoint 做 Polyak 平均（§3.4.3，[A]）。herd §3.1.3 还把 annealing 用成数据价值评估工具：取 50% 进度处的 8B 模型，用 40B tokens、30% 新数据权重做退火，替代逐数据源的 scaling law 实验。annealing 的 schedule 形态（linear-to-0 与 cosine / WSD 末段的分工）归 2.3；它在尾段扮演两个角色——能力收尾与数据试纸。

拿这三条判据看一个实例：Qwen2.5-1M 的 §4。它的 post-training 是两阶段 SFT：第一阶段只用不超过 32,768 tokens 的短指令数据；第二阶段才混入最长 262,144 的长短混合，并刻意平衡长短比例，防止短任务遗忘；随后接 DPO 类离线 RL。三条判据全部越界——loss 是 response 条件似然，schedule 另起 SFT 配置，数据是指令样本。尽管窗长与 mid-training 段相同，归属仍是 post-training。长上下文能力在预训练扩展段建成，在 SFT 第二阶段对齐到指令场景，两段分工明确。

```
               数据                              窗长        loss / schedule
──────────────────────────────────────────────────────────────────────────────
pre-training   通用语料，随机截断 packing         4K-8K      NTP；cosine/WSD 主升段
               （14.8T，DSv3；15.6T，Llama-3 主阶段）
mid-training   数学/代码/repo/长文档上采样        32K-256K   NTP；同一条 schedule 收尾
               （best-fit packing 保完整推理链与 repo；量级 10^9–10^12 下沿：DSv3 ≈126B / Llama-3 ≈800B）
post-training  指令 + 偏好对 + RL rollout        视任务      条件似然 / RL；另起 schedule（P9）
               （DSv3 后训练仅 5K H800 小时，对照预训练 2664K）
```

## 3. 长上下文扩展的成本账（training 侧）【关键推导】

先回顾 models 4.6 式 (1)：causal attention 每层的 score 加权 FLOPs 是 $2W^{2}h$（ $h = H_q \cdot d_{\mathrm{head}}$；该章记作 $d$，本章按 training 约定改写为 $h$）。对配方师更有用的是 per-token 口径：同式除以窗内 token 数 $W$：

$$
\phi_{\mathrm{attn}}(W) = \frac{2W^{2}h}{W} = 2Wh \quad \text{（每 token 每层，fwd）} \qquad (1)
$$

**符号**： $W$ = 训练窗长； $h = H_q \cdot d_{\mathrm{head}}$ = 各注意力头维度之和； $\phi_{\mathrm{attn}}$ = attention score 加权部分的 fwd FLOPs / token（FlashAttention 不改这笔账，只改执行方式，models 4.6 §1.1）。预算 $B_k$ tokens、窗长 $W_k$ 的一个扩展阶段，fwd 总成本为：

$$
C_k = B_k \cdot \left( 2N_{\mathrm{act}} + 2hLW_k \right) \qquad (2)
$$

**符号**： $N_{\mathrm{act}}$ = 每 token 激活参数量（dense 模型即 $N$；MoE 用激活口径）； $L$ = 层数；fwd+bwd 统一乘 3（1.5 的 6N 口径），结构与下述推论不变。两条推论：

- **固定 token 预算，窗长乘 $s$ ⇒ attention 项乘 $s$**：扩窗的边际成本全部来自 attention 项；attention 与 dense 项打平的交叉点 $W^{\ast} = N_{\mathrm{act}}/(hL)$（与 models 4.6 式 (2) 同式，8B 形态约 $6 \times 10^{4}$）。
- **tokens/step 恒定时，序列数除以 $s$**：DeepSeek-V3 是公开数值最全的实例。扩窗阶段 1 的 batch 是 1920 条 × 32,768 tokens，阶段 2 是 480 条 × 131,072 tokens——每步 tokens 恒为 62,914,560（1920 × 32768 = 480 × 131072；§4.3，[A]），而每步 attention FLOPs 恰为 4 倍。两阶段各 1000 步，扩窗 tokens 合计约 126B（由 batch × steps 换算，论文未直接给 tokens 数，此处为换算口径）。它占全训练 tokens 约 0.8%，对照 Table 1 的卡时份额 4.3%：单价差约 5 倍，方向即式 (2) 的预测。精确系数含 MLA 的 score 结构与长序列 kernel 效率差异，本书不复算（练习 2 用 MHA 口径复算一个下界）。

**位置分布效应**。窗长只是上限。causal 下第 $i$ 个 token 的上下文长度是 $i$，且 packing 后 position_ids 逐文档重置（7.1）——token 实际受训的上下文长度由文档长度分布决定，不由窗长决定。自然语料的长度分布是重尾的，超长文档占比极小（models 4.6 §4 口径）；大 $i$ 区间的梯度样本随之稀薄，慢通道 RoPE 角度与远程检索模式得不到足够的训练信号。长文档上采样就是对位置分布做重加权。Qwen2.5-1M 的操作口径是：扩展各阶段 75% 序列取当前最大长度，25% 取较短序列（§3 Training Strategy 段，[A]）；另加三类合成长数据——FIM、按关键词 / 位置检索、段落重排——针对自然语料「局部连贯强于全局结构、next-token 预测用不上远程依赖」的结构性缺口（同节口径）。GLM-4.5 在 128K 段同样上采样长文档（§2.3）。推论：只抬窗不改配比，等于把按 $W$ 计的账单花在短上下文行为上。 $W$ 决定成本，文档长度分布决定买到多少长程信号。

**CP 启用时机**。式 (2) 的 $W_k$ 项同时是单卡压力源：attention FLOPs 与逐位置激活都随 $W$ 增长。把序列维切给 $c$ 张卡（CP）后，每卡 FLOPs 与激活约除以 $c$（causal 负载均衡后），代价是 K/V 分块环流的通信——推导与通信账总表归 5.2 §3-§6，此处只登记启用判据与实证锚点。两个触发条件，任一成立即该考虑 CP。其一，单卡激活账：scores 经 FlashAttention 不物化，但逐位置激活（attention 输入输出、MLP 中间态）仍随 $W$ 线性长，128K 单序列单层即 $W \cdot h$ 量级，乘上层数与 micro-batch 后逼近单卡 HBM 预算。其二，单步时延账：tokens/step 恒定时， $W$ 乘 $s$ 使每步 attention FLOPs 乘 $s$，不切开序列维则 step time 同步膨胀。当 $W$ 触发上述条件、而 DP/TP/PP 度数已被显存与通信约束固定时，序列维是最后可切的轴（5.2 §1）。herd 的口径是 CP "enable training on extremely long sequences up to 128K in length"，并把序列切成 $2c$ 块、每卡取两块做负载均衡（§3.3.2；即 5.2 §4 推过的 zigzag，5.2 已登记 herd 的 CP16 @ 128K 配置）。披露口径如实记录：DeepSeek-V3 / GLM-4.5 / Qwen2.5-1M 三份 TR 全文均无 context parallelism 配置披露（2026-11-17 本次全文核实），其 32K 以上各段的序列切分方案未公开。

## 4. 对照表与口径纪律

「训练长度」与「推理外推长度」在 Qwen2.5-1M 上相差 4 倍。训练侧只到 262,144 tokens——§3 Training Strategy 段的 Table 2 按 32,768 / 65,536 / 131,072 / 262,144 四档记录，TR 自述 progressive expansion 共五段，前两段 4K→32K 沿用 Qwen2.5 Base 的中间版本。1M 档能力由推理侧的 DCA + YaRN attention scaling 外推得到（§5.1，标题落在 "Inference and Deployment" 下，原文自称 "an inference technique"）。对照表因此只列 training 侧三列；RoPE 缩放参数细节、逐段校验门禁与 claimed vs effective 评测归 models 4.6 §4-§5：

| 模型 | 训练窗长路径（出处） | 扩窗 tokens 预算 | 长序列并行（披露口径） |
| --- | --- | --- | --- |
| Llama-3.1（405B 配方）[A] | 8K→128K，六阶段（§3.4.2） | 约 800B，约主阶段 15.6T 的 5% | CP， $2c$ 分块负载均衡（5.2 登记 CP16 @ 128K） |
| DeepSeek-V3 [A] | 4K→32K→128K，两阶段各 1000 步（§4.3） | 约 126B（batch × steps 换算口径） | 未披露；扩窗卡时 119K H800 小时（Table 1） |
| GLM-4.5 [A] | 预训练 4,096 → mid-training 32,768 → 131,072（§2.3-2.4）；RoPE base 随 32K 档 $10^{4} \to 10^{6}$ | 未单列（全程 23T 口径） | 未披露 |
| Qwen2.5-1M [A] | 32,768 / 65,536 / 131,072 / 262,144 四档（§3）；ABF base 依次 $10^{6}$ / $10^{6}$ / $5 \times 10^{6}$ / $10^{7}$ | 未披露 | 未披露 |

表里三格「未披露」本身也是信息。窗长路径与 RoPE base 阶梯可以公开——它们定义产品能力；扩窗花了多少 tokens、用什么并行切分则接近成本结构，恰是各家最不透明的一段。读这类报告时，「披露到什么粒度」与数字一样值得对照（Qwen3 TR 连超参数数值都不披露的先例见 7.2）。

外推侧只立纪律、不展开（机制归 models 4.6 §2.5）。DCA 把相对位置重编号进训练见过的值域，YaRN attention scaling 给 attention logits 补温度，两者合用，把 256K 训练的 Qwen2.5-1M 推到 1M 推理。TR 自己的消融即口径实例：只在 32K 上训练过的 Qwen2.5-7B/14B-Instruct 加 DCA 后，passkey retrieval 在 1M tokens 上仍超 80% 准确率（§5.1）。训练长度与推理外推长度是两个独立变量，写表混排即事实错误。反方向的依赖 TR 也登记了："training on longer sequences (up to 256k tokens) substantially improves the model's ability to extrapolate"——训练窗决定外推质量的上限。两列独立，但不无关。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| mid-training | GLM-4.5 §2.3 命名 [A] | 预训练尾段专项强化（数学 / 代码 / repo / 长文档 / agent）；分界 = loss（仍 NTP）+ schedule 连续 + 量级 |
| annealing | herd §3.4.3：末 40M tokens、LR→0、保持 128K、Polyak 平均 [A] | 尾段两角色：能力收尾 + 数据价值试纸（§3.1.3 的 40B @ 8B 口径） |
| per-token attention 成本 | 式 (1)： $2Wh$ / 层（fwd） | 固定 token 预算下随窗长线性增长；交叉点 $W^{\ast} = N_{\mathrm{act}}/(hL)$ |
| 阶段成本账 | 式 (2)： $C_k = B_k(2N_{\mathrm{act}} + 2hLW_k)$ | tokens/step 恒定 ⇒ 序列数随窗反比缩（DSv3 1920 × 32K = 480 × 128K） |
| DSv3 扩窗账 | 约 126B tokens（换算）≈ 0.8% tokens vs 4.3% 卡时 [A] | 长窗 token 单价约 5 倍 |
| 位置分布效应 | Qwen2.5-1M：75% 满长 + 25% 短 [A]；FIM / 检索 / 重排合成 | 窗长定成本，文档长度分布定长程信号 |
| CP 启用 | 单卡 attention 账超预算 ⇒ 切序列维；账归 5.2 | herd CP16 @ 128K；DSv3 / GLM-4.5 / Qwen 未披露 |
| 口径纪律 | 训练 256K ≠ 推理 1M（Qwen2.5-1M，DCA+YaRN 归 §5.1 推理侧） | 两列不混排；32K 训练 + DCA 已能 passkey @ 1M 超 80% |

## 练习

1.（推导）从式 (1) 出发：(a) 补全式 (2) 的推导，并说明 fwd+bwd 统一乘 3 为何不改变 §3 两条推论；(b) 固定总预算 $B$、目标窗长 $sW_0$（ $s > 1$ ）：比较「一步到位在 $sW_0$ 上投完全部预算」与「两段阶梯（ $\sqrt{s}W_0$ 与 $sW_0$ 各投 $B/2$ ）」的总成本，给出两者之比，说明 attention 主导极限下阶段化省多少、dense 主导极限下又如何；对照 models 4.6 §4「阶段化买的是逐段可校验的过程控制」的口径，讨论成本侧结论何时与之同向。

2.（计算）DeepSeek-V3 扩窗复算（MHA 近似口径，注：MLA 的 score 结构使精确系数偏离此口径，本题只取量级）：取 $N_{\mathrm{act}} = 37 \times 10^{9}$、 $L = 61$、 $h = 128 \times 128$。(a) 用式 (2) 算 4K / 32K / 128K 三档的每 token fwd 成本与 attention 占比；(b) 两阶段 tokens 各约 63B，算扩窗平均每 token 成本相对 4K 主阶段的倍数；(c) 对照卡时口径（预训练每万亿 tokens 180K H800 小时、扩窗 119K 小时 / 约 126B tokens，Table 1）算观测单价倍数，并给出 (b) 与 (c) 差距的两个候选来源。

3.（分析）判据应用：按 §2 的三条判据，把以下三段各归入 mid-training 或 post-training 并各给一句理由——(a) Qwen2.5-1M 的 SFT 第二阶段（长短混合、最长 262,144，§4）；(b) GLM-4.5 的 repo 级代码训练段（§2.3）；(c) DeepSeek-V3 的 4K→32K 扩窗阶段（§4.3）。

## 参考

- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——§4.3 两阶段扩窗超参与 YaRN 配置、Table 1 卡时账、14.8T 主阶段口径（2026-11-17 复核）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——§3.4.2 六阶段 8K→128K 约 800B、§3.4.3 annealing 40M tokens 与 Polyak 平均、§3.1.3 annealing 数据评估、§3.3.2 CP 负载均衡（2026-11-17 复核）
- [GLM Team, 2025](../references.md)：GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models，arXiv:2508.06471，[A]（key: glm-4.5）——§2.3 mid-training 定义与三段结构、best-fit packing 口径、§2.4 LR / 批量超参（2026-11-17 复核）
- [Qwen Team, 2025](../references.md)：Qwen2.5-1M Technical Report，arXiv:2501.15383，[A]（key: qwen2.5-1m）——§3 四档 32K→256K、ABF base 阶梯与 75/25 配比、三类合成长数据；§5.1 DCA + YaRN 推理外推与 passkey 消融（2026-11-17 复核）
- [Peng et al., 2023](../references.md)：YaRN: Efficient Context Window Extension of Large Language Models，arXiv:2309.00071，[A]（key: yarn）——attention 温度缩放出处；公式本体归 models 4.6 §2.3
