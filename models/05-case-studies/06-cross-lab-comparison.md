# 5.6 跨家对照与方法论（Cross-Lab Comparison and Methodology）

> Part 5 的收口章：把 5.1–5.5 五家的旗舰摆上同一张对照表（每格数字回指各章 config 实测与调研锁定值），用三笔账解释 MoE 全勤、subquadratic 结构（四家答题、GLM 留对照）与 MTP（三家明装）的收敛线，用 cache / 计算 / 实现复杂度三个维度拆 MLA 与 GQA 系的分歧，最后把前五章反复演示的读法沉淀为可独立执行的方法论——config.json 字段字典表、HF modeling 阅读路径、tech report 五步读法、30 分钟拆解 checklist | 前置：5.1–5.5（全部数值来源）、1.2–1.3（成本口径与全生命周期账）、2.4（KV cache 账）、2.5（RoPE 波长账）、2.8（config 读法）、4.1（MLA）、4.2（DSA）、4.4（hybrid 内存账）、4.5（MoE 经济与 MTP）、4.6（扩窗配方） | 锚点：Llama-4 Scout config + blog [W]、DeepSeek-V3.2 论文与官方 repo [A]、Qwen3-Next config + model card [W]、GLM-4.6 config [W]、Kimi K3 report [A]

## 1. 跨家对照总表

五家旗舰各取一行：Llama-4 Scout（5.1）、DeepSeek-V3.2-Exp（5.2）、Qwen3-Next-80B-A3B（5.3）、GLM-4.6（5.4）、Kimi K3（5.5）。数值口径沿用 P5 约定：config.json 实测 > tech report 架构表 > model card > blog；total/active 照抄官方口径不换算；per-token cache 为 bf16、全量驻留口径的线性斜率。

| 模型（证据） | attention 方案 | 位置编码 | MoE 粒度与路由 | norm | vocab | ctx（max_position_embeddings） | per-token cache | 激活比（官方口径） | MTP | hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-4 Scout [W] | GQA（ $g$=5，8 KV 头×128）+ chunk 8192 | iRoPE 3:1（每第 4 层 NoPE）；RoPE $\theta=5\times10^{5}$ + llama3 式 f16 | 16 routed + 1 shared，top-1，层层 MoE（Maverick 128 routed + 1 shared，隔层）；路由细节未公开 | RMSNorm（Pre-LN）+ QK-Norm（use_qk_norm=true；Maverick 无） | 202048 | 10,485,760 | 192 KiB | 17B/109B ≈ 15.6% | 公开材料无签名 | 否（chunked full） |
| DeepSeek-V3.2 [A] | MLA（576 元素/层）+ DSA top-2048 | MLA 解耦 RoPE 64 维； $\theta=10^{4}$ + YaRN f40（4096→163840） | 256+1 top-8，前 3 层 dense；sigmoid + aux-loss-free 偏置 + 组限定 8 选 4 | RMSNorm（Pre-LN） | 129280 | 163840 | 68.6 KiB（decode 读取压至 top-2048 行） | 37B/671B ≈ 5.5% | 有（ $D$=1，接受率 85–90%） | 否（sparse full） |
| Qwen3-Next [W] | 12 层 gated GQA（ $g$=8，2 KV 头×256）+ 36 层 GDN | full 层 partial RoPE 0.25（旋 64/256 维）， $\theta=10^{7}$；GDN 层无 RoPE | 512+1 top-10；norm_topk_prob | zero-centered RMSNorm + gated 变体 | 151936 | 262144（宣称可扩 1,010,000） | 24 KiB + 36 MiB 定长状态 | 3B/80B = 3.75% | 有（model card 列装） | 是（3:1） |
| GLM-4.6 [W] | GQA（ $g$=12，8 KV 头×128） | RoPE $\theta=10^{6}$ + partial 0.5（旋 64/128 维， $\lambda_{\max}$≈4.1M 原生覆盖） | 160+1 top-8，前 3 层 dense | RMSNorm + QK-Norm + QKV bias（叠加） | 151552 | 202752 | 368 KiB | 32B/355B ≈ 9.0%（含 MTP、不含词表） | 有（ $D$=1，整件 MoE 层） | 否 |
| Kimi K3 [A] | KDA : full 交错（层型比以 report 正文为准） | KDA 侧靠衰减门控、无 RoPE；full 侧本书未锁定 | LatentMoE 896 top-16（routed 口径 $\rho$≈1.8%） | 本书未锁定 | 本书未锁定（谱系内 163840） | 1M（report 口径） | 未锁定（full 层数未锁定；linear 侧定长状态） | 104B/2.8T ≈ 3.7% | 三件套之外未列 | 是 |

表注四条。其一，**cache 列口径**：GQA/MLA 行是「每 token 新增字节」的斜率；Qwen3-Next 与 K3 是 hybrid，斜率只由 full 层贡献，linear 层另付一份与 $T$ 无关的定长状态（Qwen3-Next 为 36 × 1 MiB = 36 MiB，4.4 §7）。其二，**激活比分母口径不一**：Qwen3-Next 官标「3B 激活、非 embedding 79B」，GLM 为 report 摘要口径（含 MTP 层、不含词表两侧，5.4 §4），Kimi 为 report 摘要口径，Scout 为 blog 口径——照抄不换算。其三，**K3 行只登记三件套坐标**：层数、层型比、full 层位置方案等以 report 正文为准（5.5 §2.5、4.4 §5），本书未锁定的格子标「未锁定」而不臆填——这是 5.4 §3 无 paper 读法纪律在对照表上的执行。其四，**MTP 列**：三家明装里 GLM 的取点最独特—— $D$=1 模块不是小头而是整件 MoE 层（161 expert + attention，`num_nextn_predict_layers=1`，report 原文 "we add an MoE layer as the MTP layer"，5.4 §2.4）；「无签名」= config 与公开 report 均未列该模块，不等于不存在，只等于公开证据边界到此为止。

沿列读四点观察。（1）**attention 列五落点互不重合**：纯 GQA（GLM）、GQA+chunk+iRoPE（Llama）、MLA+sparse（DeepSeek）、GQA:linear hybrid（Qwen）、KDA:full hybrid（Kimi）——4.1–4.4 的整条技术谱系在五家旗舰上各取一格，没有两家重合同一格。（2）**vocab 列五档**：129280 / 151552 / 151936 / 163840（K3 谱系）/ 202048，2.2 的 vocab 三方账（参数 $2Vd$、压缩率、长尾冷启动）在每家的取点不同；GLM 151552 与 Qwen 151936 同量级不同档，tokenizer 同源与否未锁定（5.4 §4）。（3）**ctx 列跨 64 倍**（163840 → 10,485,760，约 1.8 个数量级），且「声明窗」与「工作窗」要分开读：V3.2 的 163840 是 YaRN f40 外推结果（工作窗 128K），Scout 的 10M 是 blog 宣称口径，K3 的 1M 是 report 口径——4.6 的 RULER 方法论问题（声明窗 ≠ 有效窗）对五行全部适用。（4）**MoE 列五家全勤**，粒度从 16 到 896（56×，约 1.75 个数量级），routing 从 top-1 到 top-16——§2.1 解释为什么这一列必然全勤。

## 2. 收敛点：三笔共同的账

五家实验室独立做架构决策，落点却高度一致：MoE 五家全勤；subquadratic 结构四家答题——GLM-4.6 以原生 200K 窗加纯 GQA 留在对照位；MTP 三家明装。收敛不是互相抄，是同一批账各自算出来的解；没答题的那家本身就是对照。

### 2.1 为什么都上 MoE：推理经济公式

4.5 §1 把 1.3 的全生命周期成本改写成 MoE 口径（recap）：

$$
\mathrm{TC}(N_{\mathrm{tot}}, N_{\mathrm{act}}, D, Q) = \underbrace{6 N_{\mathrm{act}} D}_{\text{train}} + \underbrace{2 N_{\mathrm{act}} Q}_{\text{inference}}, \qquad L \approx L(N_{\mathrm{tot}}, D) \qquad (1)
$$

**符号**： $N_{\mathrm{tot}}$ / $N_{\mathrm{act}}$：总 / 激活参数； $D$：训练 token 数； $Q$：服务期推理 token 总量； $L$：loss 约束——由总参数与数据摊薄，不进 FLOPs 目标。式 (1) 的全部经济含义：服务成本只认 $N_{\mathrm{act}}$（延迟与吞吐定价），容量维度 $N_{\mathrm{tot}}$ 按显存与互联定价，两本账分开优化。 $Q$ 越大的服务方，最优解越把 $N_{\mathrm{tot}}/N_{\mathrm{act}}$ 拉大——这就是对照表 MoE 列全勤的机制：不是某家的偏好，是式 (1) 在 $Q$ 大 regime 下的唯一方向。

稀疏度谱（对照表激活比列）：Scout 15.6% → GLM-4.6 9.0% → V3.2 5.5% → Qwen3-Next 3.75% → K3 3.7%（routed 口径 1.8%），与 4.5 §1 登记的 $N_{\mathrm{tot}}/N_{\mathrm{act}}$ 谱（Mixtral ≈ 3.6 → Qwen3-30B ≈ 9.2 → V3 ≈ 18.1）同向单调。用 config 字段可自算的等价口径是 FFN 侧「等总参 dense 的 per-token FLOPs 倍数」 $N_e/(k+K_s)$：Scout 17/2 ≈ 8.5×（16 routed + 1 shared，blog/config 口径，5.1 §5.1/§6）、V3.2 257/9 ≈ 28.6×、GLM-4.6 161/9 ≈ 17.9×、Qwen3-Next 513/11 ≈ 46.6×、K3 896/16 = 56×（routed 口径）。代价侧同样收敛地出现：router 均衡机制（V3 的 aux-loss-free 偏置、Qwen3-Next 的 norm_topk_prob、Llama-4 未公开）、前导 dense 层端点加固（V3/GLM 取 3、K2/Kimi Linear 取 1）、expert 并行的 all-to-all 通信账（4.5 §7 预告，归系统维度）——式 (1) 只给方向，这三个旋钮是每家必须自己填的工程空格。

### 2.2 为什么四家上了 subquadratic：1M ctx 的 KV 账

2.4 的 KV cache 账按「每 token 斜率 × 窗长」写法 recap（GQA 系，bf16 字节系数 $b$=2）：

$$
M_{\mathrm{cache}} = \underbrace{2\thinspace b\thinspace L\thinspace H_{kv}\thinspace d_{\mathrm{head}}}_{\text{slope}} \cdot \hspace{5mu} T \qquad (2)
$$

**符号**： $L$：层数； $H_{kv}$：KV 头数； $d_{\mathrm{head}}$：头维； $T$：窗长。MLA 行把斜率因子换成 $(d_c + d_R)$ = 576 元素（4.1）；hybrid 行只有 full 层付斜率，linear 层付定长状态（4.4 §7）。把五家斜率代入 $T$ = 1,048,576（1M）：

```
per-token cache 斜率（bf16，全量驻留口径）与 1M 窗整窗账
GLM-4.6     368 KiB/token |########################|  @1M → 368 GiB/序列（其声明窗 202752 → 71.1 GiB）
Llama-4 S   192 KiB/token |#############         |  @1M → 192 GiB；@10M 全量口径 ≈ 1.9 TiB/序列
V3.2         68.6 KiB/token |#####               |  @1M → 68.6 GiB（声明窗 163840 → 10.7 GiB；工作窗 131072 → 8.6 GiB ≈ 9.2 GB）
Qwen3-Next   24 KiB/token |##                    |  @1M → 24 GiB + 36 MiB 定长
Kimi Linear  7.9 KiB/token |#                     |  @1M → 7.9 GiB + 20 MiB 定长（4.4 §7 原账）
```

纯 GQA 在 1M 窗下每序列 cache 192–368 GiB——单序列就顶满多张显卡的显存，batch 维彻底没有空间。这还只是内存；算力侧 full attention 每 token 打分约 $4 H_q d_{\mathrm{head}} T$（4.4 §7 口径），Scout 形态（40 头 × 128 维 × 48 层）在 $T$ = 1M 处仅 attention 打分一项就约 $4 \times 40 \times 128 \times 2^{20} \times 48 \approx 10^{12}$ FLOPs/token——每解一个 token 付一万亿次乘加，且随 $T$ 线性继续涨。内存斜率与算力斜率双双顶墙，subquadratic 就是必答题而不是探索题。

带 subquadratic 结构的四家分三条路线。**压读取（sparse）**：V3.2 的 DSA 保持 576 元素/token 全量驻留，每步只 gather indexer 选中的 top-2048 行——decode 读取 144 MB vs 全量 9.2 GB @131072（64×，4.2 §6），cache 总账不动、动的是每步带宽。**压层数（hybrid）**：Qwen3-Next 3:1 把斜率除以 4（48 层只 12 层付），Kimi Linear 名义 3:1 把斜率压到 7/27 ≈ 26%——cache −75% 的数字来源就是层数比（4.4 §7）；K3 把同一主线推到 frontier 规模。**压可见窗（chunk + NoPE）**：Llama-4 Scout 不换 mixer，RoPE 层按 chunk 8192 分块、每第 4 层换 NoPE 配对数温度（5.1 §5.2）——斜率不变，但把注意力范围与位置外推的代价重排；其 10M 宣称窗若全量驻留达 1.9 TiB/序列，工程兑现依赖 chunk 边界上的 cache 处置，该实现细节公开材料未锁定 [W]。**不在图上的一家**：GLM-4.6 以 $\theta=10^{6}$ + partial 0.5 原生覆盖 202752（ $\lambda_{\max}$ ≈ 4.1M，5.4 §2.4），窗长停在 200K 量级、368 KiB/token 的斜率照付——71.1 GiB 的整窗账在 5.4 §2.4 算过，它是「暂不答这道题」的对照样本。三条路线与 §3 的分歧是同一枚硬币的两面。

### 2.3 为什么 MTP 开始普及：speculative 收益

4.5 §6 的接受率账（recap）：草稿分布 $q$ 对目标分布 $p$ 的逐位置接受率 $\alpha$ 与 $D$ 步草稿的期望产出

$$
\alpha = 1 - D_{\mathrm{TV}}(p, q), \qquad \mathbb{E}[\text{tokens}/\text{forward}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} \qquad (3)
$$

**符号**： $D_{\mathrm{TV}}$：全变差距离； $\gamma$：每步验证的草稿长度（MTP $D$=1 即 $\gamma$=1， $\mathbb{E} = 1 + \alpha$）。V3 实测第二 token 接受率 85–90%（跨生成主题）， $\mathbb{E}$ ≈ 1.85–1.9 token/前向，端到端约 1.8× TPS（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）。

普及的机制原因有三层。**草稿白捡**：MTP 模块与主模型共享 embedding 与输出头、只加 $D$ 个完整 block——相比独立草稿模型，服务端零额外模型成本，且共享表示正是把 $D_{\mathrm{TV}}$ 压小的结构手段（4.5 §6.2）；GLM 的变体把这一个 block 加成整件 MoE 层，也只约占总参 1%（5.4 §4）。**无损**：接受规则 $\min(1, p(x)/q(x))$ 保证输出分布精确等于 $p$——加速不付质量（4.5 §6.2 的推导与练习）。**正交**：MTP 挂在主干之上不占层数预算，与 dense/MoE、full/hybrid 全部架构轴正交（4.4 §3 的登记），任何谱系都能后挂——V3.2 相对 V3 只动 attention 侧、MTP 原样继承（5.2 §1）即为实例。训练侧另有监督加密收益（每个位置对更远 horizon 供监督，4.5 §6.1）。

证据边界必须写明：公开列装的是三家——DeepSeek（V3 起，[A]）、Qwen3-Next（model card 列装，[W]）与 GLM（4.5 report 明写、4.6 config `num_nextn_predict_layers=1` 实测，5.4 §2.4）；Llama-4 与 K2/K3 的 config 与公开 report 均无 MTP 签名。GLM 的取点值得单独登记：它的 $D$=1 模块不是小头而是**整件 MoE 层**（161 expert + attention），训练口径的 355B/32B 都含这一层，推理期可整体弃置（弃置后激活 32.4B → 32.0B，5.4 §4）——MTP 的「模块粒度」由此多出一个公开坐标。「普及」在本书证据边界内的准确表述是：技术包已被三家旗舰验证，接管门槛（一个 block 到一整层 MoE + 接受率校准）低到任何一家都可后挂——是否已挂，以各家 config / report 签名而非传闻为准。

## 3. 分歧点：MLA vs linear+full hybrid vs 纯 GQA

收敛止于三笔账；落到 attention 的具体形态，五家分成三条路线。用 cache、计算、实现复杂度三个维度对照：

| 维度 | MLA（DeepSeek-V3.2、K2 系） | linear+full hybrid（Qwen3-Next、Kimi Linear / K3 系） | 纯 GQA（GLM-4.6、Llama-4 系） |
| --- | --- | --- | --- |
| cache 斜率 | 576 元素 = 1.125 KiB/层（61 层 → 68.6 KiB/token） | full 层斜率 × $1/(r{+}1)$ + linear 定长状态；full 层因家而异：Qwen3-Next 2 KiB/层（GQA，3:1 → 24 KiB/token）、Kimi Linear 1.125 KiB/层（MLA+NoPE，→ 7.9 KiB/token） | 4 KiB/层（48–92 层 → 192–368 KiB/token） |
| 计算 | decode 打分在 $128{+}64$ = 192 维上进行； $W^{UK}$/$W^{UV}$ 吸收使 up-projection 不进 decode 内循环（4.1）；prefill/训练多付 down+up 投影 | full 层打分（GQA 或 MLA，因家而异）× 少数层；linear 层每 token 定常 $\approx 4 H d_v d_k$ FLOPs，与 $T$ 无关（4.4 §7） | 与 MHA 同形、 $g$ 倍省 cache 读取与 KV 投影 FLOPs；无额外结构开销 |
| 实现复杂度 | 吸收重写 + decoupled RoPE（RoPE×低秩冲突是设计起点，4.1 §4）+ 专用 kernel | 双 mixer kernel（chunkwise linear + full）+ 层型调度 + 双 cache 管理（定长 + 线性） | MHA 的 reshape 级推广，任何 FlashAttention 族 kernel 直接支持 |

三个维度的读法各一句。**cache 维**：MLA 压斜率（每层的单价）、hybrid 压层数（付单价的层数）、纯 GQA 不压斜率靠 chunk 重排可见窗——1M 窗下三者的整窗账 68.6 GiB / 24 GiB / 192+ GiB 已在 §2.2 的图上排开。**计算维**：MLA 的吸收技巧把 decode 内循环压到 576 维 latent 上做（数学上与原式精确等价，4.1），代价是 prefill 与训练侧多一串投影、以及 kernel 需要认识这套重写；hybrid 把 decode 算力变成「大固定项 + 小线性项」，1M 处 full 层打分仍是全模型算力主项（4.4 §7 的两千倍对照）；纯 GQA 的算力侧无任何新机制，省多少完全由 $g$ 决定。**实现复杂度维**：GQA 最低（reshape 级）、MLA 居中（一套重写 + 一种 kernel）、hybrid 最高（两套 mixer、两种 cache、层型装配）——transformers 的 `modeling_qwen3_next.py` 对 GDN 层直接调 fla 库的 `chunk_gated_delta_rule` 而不自研（5.3 §4.3，[fla-org](../references.md)，[A]），就是这条复杂度账单的具体一笔。

质量证据侧三家也各有锚点：V2 报告 MLA 与同规模 MHA 相当或更强（[DeepSeek-AI, 2024](../references.md)，arXiv:2405.04434，[A]）——「cache 压缩必损检索分辨率」的直觉在 512 维共享 latent 上不成立；Kimi Linear 的 1.4T 同配方公平对照报告全面超过 MLA 基线（[Kimi Team, 2025](../references.md)，arXiv:2510.26692，[A]）；GQA-8（8 KV 组）作为 cache/质量 Pareto 中间点的定位出自原论文（[Ainslie et al., 2023](../references.md)，arXiv:2305.13245，[B]）。还有一个谱系级事实值得登记：K2 的 `architectures = DeepseekV3ForCausalLM`（5.5 §2.3）证明 MLA + DeepSeekMoE 这一组合已成为可跨实验室整体复用的架构类——分歧的各方都在成为公共零件箱的一部分。

## 4. 方法论：如何独立读懂一个新模型

前五章对五个谱系重复执行了同一套流程：拉 config → 复算参数 → 对账 cache 与 FLOPs → 读 modeling 代码 → 按证据等级消化 report。本节把它写成可迁移的四件工具。

### 4.1 config.json 字段字典表

config.json 是新模型最诚实的一份文件——权重形状由它唯一决定，营销语言进不来。按 2.8 的读法，每个字段都应对到公式或结构的一个具名位置：

| 字段 | 读出什么 | 公式 / 结构落点 | 本书实例 |
| --- | --- | --- | --- |
| `architectures` | 架构类归属；未列字段按类继承 | 同名类的全部默认结构与 norm/残差拓扑 | K2 = `DeepseekV3ForCausalLM`（5.5 §2.3） |
| `num_hidden_layers` / `hidden_size` | $L$ 与 $d$ | 参数账分母； $12Ld^{2}$ 速算（2.8） | 各章 config 精读节 |
| `num_attention_heads` / `num_key_value_heads` | $H_q$ / $H_{kv}$ → $g$ | cache 斜率 $2 H_{kv} d_{\mathrm{head}}$/层（式 (2)） | GLM-4.6 $g$=12（5.4 §4） |
| `head_dim` | 头维是否独立于 $d/H_q$ | $\neq$ hidden/heads 即独立投影维（2.8 的「常见误解」条目） | Qwen3-Next 256 ≠ 128；GLM-4.6 128 ≠ 53 |
| `kv_lora_rank` / `qk_rope_head_dim` | MLA 签名 | cache = $(d_c + d_R)$ 元素/层（4.1） | V3/K2 = 512+64 = 576 |
| `q_lora_rank` / `qk_nope_head_dim` / `v_head_dim` | MLA 投影维 | score 内维 = nope + rope | K2 五字段与 V3 逐值相同（5.5 §4） |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | $N_e$ / $k$ / $K_s$ | $\rho = (k+K_s)/N_e$；FLOPs 倍数 $N_e/(k+K_s)$ | Qwen3-Next 512/10/1（5.3 §4.1）；Scout 16+1（字段名 `num_local_experts`，5.1 §6） |
| `moe_intermediate_size` | expert 中间维 $d_{\mathrm{ff}}^{(e)}$ | 单 expert $3 d\thinspace d_{\mathrm{ff}}^{(e)}$（SwiGLU 三矩阵口径，2.6） | V3 2048 → 44.0M/expert（4.5 §1） |
| `first_k_dense_replace` | 前导 dense 层数 | 浅层表示未分化 / router 早期振荡两种读法（5.2 §6） | V3/GLM 取 3 vs K2/Kimi Linear 取 1 |
| `vocab_size` / `tie_word_embeddings` | 词表参数 | untied → $2Vd$ | Scout 202048（5.1 §5） |
| `max_position_embeddings` | 声明窗长 | 与 rope 块联读判原生/外推 | GLM-4.6 202752（5.4 §4） |
| `rope_theta`（+ `partial_rotary_factor`） | 频率 base 与旋转比例 | $\lambda_{\max} = 2\pi\theta^{1-2/d_{\mathrm{rope}}}$， $d_{\mathrm{rope}}$ 取旋转维数（2.5；partial 实例 5.4 §2.4） | GLM-4.6： $\theta=10^{6}$ × partial 0.5 → $d_{\mathrm{rope}}$=64 → ≈4.1M |
| `rope_scaling` | 扩窗配方 | 有效窗 ≈ factor × original（4.6） | V3 YaRN f40×4096；K2 f32×4096；Scout llama3 式 f16 |
| `attention_bias` / `use_qk_norm` | QKV bias 与 QK-Norm 存否 | 稳定性路线分野：纯 QK-Norm（Qwen3：bias=false）vs 叠加（GLM：bias 与 QK-Norm 两件都要）（5.3 §2.2 / 5.4 §2.4） | GLM-4.6 两字段皆 true；Qwen3 bias=false + QK-Norm；Scout use_qk_norm=true |
| 层型字段：`full_attention_interval` / `linear_attn_config` / `attn_type_list` | hybrid 层型序列 | 同一决策的三种序列化写法（5.3 §4.1） | Qwen3-Next interval=4；Kimi Linear 双列表；MiniMax-01 全列表 |
| `no_rope_layers` / `attention_chunk_size` / `attn_temperature_tuning` | iRoPE 签名 | NoPE 层比例、chunk 长、对数温度（5.1 §5.2） | Scout `[1,1,1,0,...]` |
| `interleave_moe_layer_step` | MoE 层密度 | 层层 vs 隔层 | Scout 1 vs Maverick 2（5.1 §5.1） |
| `scoring_func` / `topk_method` / `norm_topk_prob` | 打分形态 / 选择算法 / 归一化位置 | sigmoid 亲和 + noaux_tc（aux-loss-free）+ 选中集内归一（4.5 §3.2） | V3 三件套签名（5.2 §3.1/§6）；Qwen3-Next `norm_topk_prob=true` |
| `torch_dtype` | 权重存储精度 | 权重内存 = 参数量 × 字节（1.2） | V3 实测 bf16（5.2 §6）；其余按本书 cache 账口径取 bf16 |

两条反例教训与字典表同等重要。其一，**稳定性技术不一定有 config 签名**：K2 的 MuonClip/QK-clip 作用在训练循环的权重上，config 里一个字段都没有（5.5 §4）——config 无签名只能排除「结构性装置」，不能排除「训练期装置」。其二，**config 未列字段按 `architectures` 类继承并标注为推断**：K2 的 RMSNorm 结论就是经架构类从 V3 继承的推断而非实测（5.5 §4）——推断要写出来是推断。

### 4.2 HF modeling 文件阅读路径

config 给声明值，modeling 代码给真实计算图。推荐阅读路径（transformers 库惯例）：

```
config.json ──► configuration_*.py ──► modeling_*.py
（声明值）      （默认值 + 派生字段      ├─ *Attention.forward : 投影 shape / RoPE 施加位置 / cache 读写
                如 layer_types 生成）   ├─ *MLP / *MoE block   : router 打分 → top-k → 归一化 → 加权；shared 是否常驻
                                      └─ *Model / *ForCausalLM: 层栈装配 / norm 拓扑 / 输出头 tying / MTP 挂点
```

三处落点的读法各配一个本书实例。**Attention.forward**：先看四个投影的 shape（q_proj 是 $d \to H_q d_{\mathrm{head}}$,Qwen3-Next 的 2048→4096 就是在这里坐实 head_dim 256 的），再看 RoPE 在哪一步施加、旋多少维（partial_rotary_factor 的落点），最后看 KV cache 的写入与读取位置。锚点：transformers 的 `modeling_llama.py`（2.3/2.4 的走读对象，[HuggingFace](../references.md)，[A]）。**MoE block**：router 打分函数（softmax 还是 sigmoid）、top-k 选择、归一化位置（`norm_topk_prob` 的代码落点）、加权求和、shared expert 是否无条件并联——DeepSeek-V3 官方 `inference/model.py` 的 Gate（4.5 §3.3）与 Mixtral 的 `MixtralSparseMoeBlock`（4.5 §1）是这条路径的两个走读过的端点。**Model 层**：层型列表如何生成与装配（`full_attention_interval` → 48 项 `layer_types`，5.3 §4.1）、norm 在残差流的哪一侧、输出头是否 tying、MTP 模块挂在哪。`modeling_qwen3_next.py` 的三处实现级证据（zero-centered norm 的 zeros 初始化、输出门从 q_proj 切出、GDN 调 fla kernel，5.3 §4.3）示范了「config 一句话 = 代码三行」的互证密度。

### 4.3 tech report 五步读法

report 的价值分布极不均匀；五步按信息密度排序，每步配一个前五章的执行实例：

1. **架构表 ↔ config 互证 + 参数复算**。目标是闭合而非止步：GLM-4.6 全字段锁定后复算 ≈355.2B 对官方 355B（含 MTP、不含词表口径）闭合到 0.06% 以内，闭合同时反证两个未单列字段的取值——shared expert 与 routed 同宽、MTP 恰为一层（5.4 §4）；该章初稿曾以「残差 4.9B、归属未锁定」收尾，而 dense FFN 宽度与 MTP 字段本就在 config 里——「字段没读完不要宣布未锁定」，这条更正记录本身就是教材。同法样本：V3 的 671B 五项分解（5.2 §3.3）、K2 的 1T 复算「99% 是 expert 参数」（5.5 §3.1）。复算闭合是「读懂了」的最低标准。
2. **ablation 存在性检查**。逐项消融有无决定组件收益的证据等级：QK-Norm 无公开消融，收益只能登记为「宣称」并列开放问题（5.3 §5）；iRoPE 的超参（3:1、chunk 8192）同样无公开对照（5.1 §7）。没有 ablation 的组件不是错，是不能当结论用。
3. **数据与成本口径**。tokens 数横向对照（V3 14.8T / GLM-4.5 23T / K2 15.5T / Qwen3-Next 15T / Moonlight 5.7T）；成本数字先问口径再转述——V3 的 2.788M H800 GPU-hours 是 2664K+119K+5K 三段之和、且 GPU-hours ≠ 货币（5.2 §3.4）。**[W]/[C] 级材料不做数据断言**（5.4 §3 的纪律）。
4. **评测协议与对照公平性**。对照是否同配方：Kimi Linear 的 1.4T 同数据档是公平对照、5.7T 是发布档，两个数字服务的论证不同（4.4 §4）；长上下文宣称要过 4.6 的 RULER 口径检查（声明窗 ≠ 有效窗）。
5. **局限与未做项**。作者自列的 failure modes 与没做的实验是下一章问题的富矿；读完落一份开放问题清单——前五章每章末的清单（5.1 §7、5.2 §7、5.3 §5、5.4 §5、5.5 §5）示范了写法：每条可判定、指向具体的缺失证据。

### 4.4 30 分钟拆解 checklist

面对一个刚发布的新模型，按此时序执行（每格产出即停损点——时间到就登记「未锁定」，不臆填）：

| 时段 | 动作 | 产出 |
| --- | --- | --- |
| 0–5 min | 定证据等级（有无论文 / 开放权重 / config）；拉 config.json 与 model card | 证据标签（[A]/[B]/[C]/[W]）+ 口径声明 |
| 5–10 min | 骨架字段： $L$、 $d$、heads、vocab、ctx、dtype； $12Ld^{2} + 2Vd$ + MoE 项粗算参数 | 参数账与残差归属 |
| 10–15 min | attention 签名（ $g$ / MLA 五字段 / 层型字段）+ 位置块（ $\theta$、scaling、chunk） | cache 斜率 + 窗长口径（原生 vs 外推） |
| 15–20 min | MoE 字段： $N_e$、 $k$、 $K_s$、first_k_dense、 $d_{\mathrm{ff}}^{(e)}$；路由签名字段 | $\rho$ 与 per-token FLOPs 账 |
| 20–25 min | modeling 代码抽查一处：Attention.forward 或 MoE block | 一处实现级证据（或「无签名」登记） |
| 25–30 min | report / blog 五步速览；写三条开放问题 | 一页可归档笔记 |

前五章的五份 config 精读（5.1 §6、5.2 §6、5.3 §4、5.4 §4、5.5 §4）就是这张 checklist 各自的展开版——对照着读，能看到同一张表在五种证据等级下的执行差异。

## 总结

| 主题 | 结论 | 公式 / 数字 | 去向 |
| --- | --- | --- | --- |
| 对照总表 | 五家旗舰全 MoE；attention 五落点互不重合；ctx 跨 64 倍 | §1 表（11 列 × 5 行） | 回收 5.1–5.5 |
| MoE 收敛 | 式 (1)：服务认 $N_{\mathrm{act}}$、容量认 $N_{\mathrm{tot}}$ | $\rho$ 谱 15.6% → 3.7%；FLOPs 倍数 8.5×–56× | 4.5 |
| subquadratic 收敛 | 式 (2)：纯 GQA @1M = 192–368 GiB/序列；attention 打分 @1M ≈ $10^{12}$ FLOPs/token | 四家三条出路（GLM 留对照）：压读取（DSA 64×）/ 压层数（hybrid ÷$(r{+}1)$）/ 压可见窗（chunk） | 4.2 / 4.4 / 5.1 |
| MTP 收敛 | 式 (3)： $\alpha$ = 85–90% → $\mathbb{E}$ ≈ 1.85–1.9 → 约 1.8× TPS | 草稿白捡 + 无损 + 正交；三家明装（GLM 为整件 MoE 层变体）、Llama-4 与 K2/K3 无签名 | 4.5 §6、5.4 §2.4 |
| 分歧 | MLA 压斜率（1.125 KiB/层）/ hybrid 压层数（24 KiB/token）/ 纯 GQA 靠 chunk（4 KiB/层） | cache / 计算 / 实现复杂度三维表 | 4.1 / 4.4 |
| 方法论 | config 字段字典（19 行）+ modeling 三落点路径 + report 五步 + 30 min checklist | §4 四件工具 | 可迁移 |

## 练习

1.（计算）用式 (2) 重算五家的 per-token cache 斜率：Llama-4 Scout（48 层、 $H_{kv}$=8、 $d_{\mathrm{head}}$=128）、DeepSeek-V3.2（61 层 × 576 元素）、Qwen3-Next（12 full 层 × 1024 元素）、GLM-4.6（92 层、 $H_{kv}$=8、 $d_{\mathrm{head}}$=128）。把 $T$ = 1,048,576 代入验证 §2.2 图上的 192 / 68.6 / 24 / 368 GiB 四个数；再算 Scout 声明窗 10,485,760 下的全量驻留口径（≈1.9 TiB），并说明 chunk 8192 在「cache 是否可逐出」上留下的实现自由度。
2.（计算）MTP 收益账。 $D$=1、 $\alpha$ = 0.87 时用式 (3) 算 $\mathbb{E}$；若 $D$=2 且两位置接受独立同 $\alpha$， $\mathbb{E}$ 与 TPS 上限各为多少？指出「逐位置独立同分布」这一近似在两个方向上如何偏乐观（提示：相邻位置接受事件正相关时，几何级数高估还是低估长尾）。
3.（推导 + 分析）MLA 与 hybrid 的 decode 读取账。(a) 用 4.1 的机制说明 MLA 每步读取字节数与 head 数无关（ $W^{UK}$ 吸收后内循环只在 576 维 latent 上）；(b) 设 hybrid 的 linear:full = $r$:1、linear 层每层定长状态 $S$ 字节、full 层斜率 $c$ 字节/token，写出全模型 decode 内存 $M(T)$，证明「 $r$ 增大的边际收益饱和」——即 $M(T)$ 随 $r$ 的下界由哪一项钉死，并代入 Qwen3-Next 的数字（ $S$ = 1 MiB、 $c$ = 2 KiB、 $L$ = 48）找出「 $r$ 再翻倍只省多少」的具体量级。
4.（代码 + 调研）30 分钟 checklist 实操：任选一个 2025–2026 发布、带开放 config 的旗舰级模型（如 MiniMax-Text-01，其 config 本书 4.4 §6 已锁定，可用来交叉验证你的读法），严格按 §4.4 的六格时序执行并计时；产出一页笔记，必须含：证据等级与口径声明、参数复算与残差归属、cache 斜率与窗长口径、一处 modeling 代码证据、三条可判定的开放问题。对照本书 4.4 §6/§7 的既有数字复核你的第 2–4 格。

## 参考

- [Meta, Llama-4 blog + Scout/Maverick config](../references.md)：[W]（key: llama-4）——Scout 行全部 config 字段与 109B/17B、10M 窗的 blog 口径
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——llama3 式 rope_scaling 配方的出处谱系
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V2，arXiv:2405.04434，[A]（key: deepseekv2）——MLA 质量口径（与 MHA 相当或更强）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——MTP 接受率 85–90% 与约 1.8× TPS、671B/37B、路由三件套
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——V3.2 行沿用的骨架字段
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-V3.2，arXiv:2512.02556，[A]（key: deepseek-v3.2）——DSA top-2048、decode 读取 144 MB vs 9.2 GB
- [Dai et al., 2024](../references.md)：DeepSeekMoE，arXiv:2401.06066，[A]（key: deepseekmoe）——细粒度 + shared expert 的零件出处
- [Yang et al., 2025](../references.md)：Qwen3 Technical Report，arXiv:2505.09388，[A]（key: qwen3）——QK-Norm 替代 QKV bias 的分野出处
- [Qwen, Qwen3-Next-80B-A3B-Instruct model card / blog / config](../references.md)：[W]（key: qwen3-next）——Qwen3-Next 行全部字段、MTP 列装、1,010,000 外推宣称
- [GLM-4.5 Team, 2025](../references.md)：GLM-4.5，arXiv:2508.06471，[A]（key: glm-4.5）——355B/32B 与 23T tokens 的摘要口径
- [Z.ai, GLM-4.6 config.json 与 model card](../references.md)：[W]（key: glm-4.6）——GLM-4.6 行全部字段（2026-11-12 实测）
- [Z.ai, GLM-4.7 config.json 与 model card](../references.md)：[W]（key: glm-4.7）——同 checkpoint 迭代线的对照样本
- [Moonshot AI, Kimi-K2-Instruct config.json](../references.md)：[A]（key: kimi-k2-config）——`architectures = DeepseekV3ForCausalLM` 与 MLA 五字段
- [Kimi Team, 2025](../references.md)：Kimi K2，arXiv:2507.20534，[A]（key: kimi-k2）——1T/32B 口径
- [Kimi Team, 2025](../references.md)：Kimi Linear，arXiv:2510.26692，[A]（key: kimi-linear）——cache −75% = 层数比、1.4T 公平对照、decode 6.3×
- [Kimi Team, 2026](../references.md)：Kimi K3，arXiv:2607.24653，[A]（key: kimi-k3）——2.8T/104B、LatentMoE 896/16、Attention Residuals、1M ctx
- [Liu et al., 2025](../references.md)：Muon is Scalable for LLM Training（Moonlight），arXiv:2502.16982，[A]（key: moonlight）——5.7T tokens 口径
- [Ainslie et al., 2023](../references.md)：GQA，arXiv:2305.13245，[B]（key: gqa）——GQA-8（8 KV 组）Pareto 中间点定位
- [Leviathan et al., 2023](../references.md)：Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192，[B]（key: speculative-decoding）——接受规则与无损性的出处
- [Peng et al., 2023](../references.md)：YaRN，arXiv:2309.00071，[A]（key: yarn）——rope_scaling 字段的配方出处
- [fla-org](../references.md)：flash-linear-attention，[A]（key: fla-repo）——hybrid 实现复杂度账的 kernel 锚点
- [HuggingFace](../references.md)：transformers `modeling_llama.py`，[A]（key: llama-modeling）——§4.2 阅读路径锚点
- [MiniMax, 2025](../references.md)：MiniMax-01: Scaling Foundation Models with Lightning Attention，arXiv:2501.08313，[A]（key: minimax01）——§4.1 层型字段第三写法（`attn_type_list`）出处与练习 4 的对照模型
- [MiniMax, MiniMax-Text-01 config.json](../references.md)：[A]（key: minimax-text-01-config）——练习 4 的交叉验证锚点（4.4 §6 已锁定）
