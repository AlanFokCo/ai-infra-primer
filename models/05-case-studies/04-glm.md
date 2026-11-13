# 5.4 GLM 谱系：从自研 objective 到标准零件箱（The GLM Lineage）

> GLM 是本书五家案例中唯一一条从自研 objective 出发的谱系：GLM 2021 用 blank infilling + 2D 位置编码把理解与生成收进同一组权重（机制推导在 3.1，本章结算技术后果的账），GLM-130B 把该 objective 推到 130B 双语并留下 100B 级训练稳定性记录，ChatGLM → GLM-4 转入对话服务代，GLM-4.5/4.6/4.7 的 config 显示旗舰已整体收敛到标准零件箱（GQA + RoPE + 细粒度 MoE）；4.6 无独立 tech report、4.7 无 arXiv，本章顺带以其为「如何对待无 paper 模型」的示范样本。按 P5 固定模板展开：配置谱系表 → 标志性架构决策的公式级分析 → 无 paper 读法示范 → GLM-4.6 config 逐字段精读 → 技术开放问题 | 前置：2.4（KV cache 账）、2.5（RoPE 波长账）、2.7（Pre/Post-LN 与 warmup、DeepNorm 家族）、2.8（head_dim 误解）、3.1（blank infilling 式 (8)–(11)、可生成判定式 (16)）、4.1（GQA）、4.5（MoE 稀疏度谱）、5.2（first_k_dense_replace 读法）、5.3（QKV bias → QK-Norm 分野） | 锚点：THUDM/GLM（2021 代码，3.1 已锚定）[A]、zai-org/GLM-4 [A]、zai-org/GLM-4.5 [A]、zai-org/GLM-4.6 / GLM-4.7 HF config（2026-11-12 实测）[W]；论文：GLM [A]、GLM-130B [A]、ChatGLM 代际报告 [A]、GLM-4.5 [A]、GLM-4.5V [A]；[W]：GLM-4.6、GLM-4.7

## 1. 配置谱系表

GLM-4.6 的 config.json 与 GLM-4.5 逐字段 diff 只剩一项：max_position_embeddings 131072 → 202752（两 config 本轮同拉对照，2026-11-12）；GLM-4.7 的 config 再与 4.6 逐字段完全一致（同轮实测）。一条从自研 objective 出发的谱系，到旗舰代已经把全部架构字段写进了与 5.1–5.3 各家同型的标准零件箱。全谱系（列序按 W3 约定：model | total/active 参数 | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE | norm | vocab | ctx（max_position_embeddings）| rope_theta | 证据 key）：

| 模型 | total/active | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE | norm | vocab | ctx | rope_theta | key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM（2021） | — | — | — | — | — | Part A 双向 + Part B span 内因果（blank infilling；2D learned absolute 位置，无 RoPE） | 无（dense） | — | — | — | — | glm [A] |
| GLM-130B | 130B（dense） | — | — | — | — | 同 objective 族 + [gMASK] 文档级变体 | 无（dense） | — | — | — | — | glm-130b [A] |
| ChatGLM 系列 → GLM-4 | — | — | — | — | — | — | — | — | — | — | — | glm-4 [A] |
| GLM-4.5 | 355B / 32B（report 摘要口径：含 MTP、不含词表两侧） | 92 | 5120 | 96/8（ $g$=12） | 128 | GQA + RoPE（partial 0.5） | 160+1 top-8，first 3 dense；MTP ×1 | RMSNorm + QK-Norm（config 实测） | 151552 | 131072 | $10^{6}$ | glm-4.5 [A] + glm-4.6 config [W] |
| GLM-4.6 | 同 4.5（数字引 4.5 report，合法性见 §3） | 92 | 5120 | 96/8（ $g$=12） | 128 | GQA + RoPE（partial 0.5） | 160+1 top-8，first 3 dense；MTP ×1 | RMSNorm + QK-Norm（config 实测） | 151552 | 202752 | $10^{6}$ | glm-4.6 [W] |
| GLM-4.7 | 同 4.5 | 92 | 5120 | 96/8（ $g$=12） | 128 | GQA + RoPE（partial 0.5） | 160+1 top-8，first 3 dense；MTP ×1 | RMSNorm + QK-Norm（config 实测） | 151552 | 202752（API 侧标称 204800） | $10^{6}$ | glm-4.7 [W] |

表注五条。其一，「—」= 本轮 research pass 未实测该字段：GLM 2021 / GLM-130B / ChatGLM / GLM-4 各行的逐字段 config 未纳入本书实测序列，谱系结论不依赖这些格子；数值口径优先级为 config.json 实测 > tech report 架构表 > model card > blog（W3 约定页）。其二，total/active 不是 config 字段：355B/32B 引 GLM-4.5 report 摘要（[GLM-4.5 Team, 2025](../references.md)，arXiv:2508.06471，[A]），4.6/4.7 行照抄该数字的合法性来自 §3 的 config 同构验证，不是当然成立。其三，GLM-4.5/4.6/4.7 的架构字段以 GLM-4.6/4.7 config 实测为准（2026-11-12），4.5 行与 4.6 唯一的差异已在 diff 中排除（仅 ctx）。其四，head_dim 128 $\neq$ 5120/96 ≈ 53.3——2.8「head_dim 误解」条目的又一实例，q 投影实为 5120 → 12288、K/V 各 5120 → 1024。其五，GLM-4.7 的 API 侧 ctx 标称 204800（OpenRouter 旁证，2026-11-12）与 config 字段 202752 并存——服务口径与字段口径之差登记在此，不互相覆盖。其六，旗舰三行的 norm 列与 partial 标注为 config 实测（`rms_norm_eps`=$10^{-5}$ 的字段签名 + `use_qk_norm`=true + `partial_rotary_factor`=0.5，2026-11-12 fix round 复核 hf-mirror config）；MTP ×1 即 `num_nextn_predict_layers`=1，机制归 4.5 §6。

读表得出的结构观察是两段式。**objective 驱动期**（GLM 2021、GLM-130B）：谱系的独有机制（blank infilling、2D 位置编码、[gMASK]）由 objective 反向决定，填不满的列恰是「该期还没有标准零件箱」的侧面。**收敛期**（GLM-4.5 起）：每一列都能填，且取值与 DeepSeek/Qwen 同族——GQA、RoPE、细粒度 MoE + shared expert + dense 前导层、untied 词表。两段之间的过渡带是 ChatGLM → GLM-4：对话服务化（instruction / RLHF 归 training 维度），代际报告（[Team GLM, 2024](../references.md)，arXiv:2406.12793，[A]）一篇覆盖 GLM-130B 到 GLM-4 全家族，标题里的 All Tools 即工具调用定位；本 pass 未实测该过渡带各代 config，其架构字段本章不断言。GLM-4.5V（[GLM-V Team, 2025](../references.md)，arXiv:2507.01006，[A]）为视觉版（GLM-4.5V 与 GLM-4.1V-Thinking），多模态细节归 P8，本章只登记存在性。

## 2. 标志性架构决策的公式级分析

### 2.1 GLM 2021：一个 objective 买两种能力，账是多少

机制推导全部在 3.1 §4：span 长度采自 Poisson（ $\lambda$=3）、覆盖 15%、每条 span 塌缩成单个 [MASK] 得 Part A、乱序拼接成 Part B、目标函数对排列取期望（3.1 式 (8)）；2D 位置 id $(p^{(1)}, p^{(2)})$ = 空白地址 × 空内偏移（3.1 式 (9)–(11)）。本章不重复推导，只 recap 最小实例（0-indexed，与 3.1 §4.3 同例）并结算技术后果：

```
Part A（双向）       ‖  Part B（乱序 span，span 内因果）
x1 [M] x4 [M] x6    ‖  [s] x2 x3 [s] x5
p1: 0  1  2  3  4      1  1  1  3  3
p2: 0  0  0  0  0      0  1  2  0  1
ℓ : 0  0  0  0  0      0  1  1  0  1
```

**买到的一：可生成性判定通过。** 3.1 式 (16) 的条件 $c_{\pi(\tau)} \subseteq G_{\tau} \cup I$ 对 GLM 成立： $I$ = Part A（生成前完整已知），span 逐条生成、span 内因果，条件集 = $x^{c} \cup$ 已生成 span $\cup$ 本 span 前缀。对照组是 BERT：MLM 的条件集含未来真值，同一判定直接失败（3.1 §7）——GLM 与 BERT 的距离不是「双向程度」，是一条可判定不等式的成立与否。

**买到的二：理解与生成同体。** Part A 内部全双向（ $M$ 的 A-A 块全 0），填空条件分布以完整上下文为条件——同一组权重既是理解探针又是生成器；模式切换在数据侧（[MASK] 短空 / [gMASK] 文档级长空的位置与数量），不在架构侧。对照 T5 的同一件事需要 encoder-decoder 两个 stack（3.1 §1 已证两种写法同构，只是 $M$ 布局差异），GLM 用单 stack 单 loss 完成。这是「理解生成同体」的精确内容：同一权重集 + 同一目标函数，理解/生成只是 mask 布局的两个取值。

**付掉的：监督密度与序列开销。** 设原序列长 $T$、span 集 $\lbrace s_1, \ldots, s_m\rbrace$，输入序列总长

$$
|x_{\mathrm{in}}| = \big(T - \sum_{i} |s_i| + m\big) + \big(\sum_{i} (|s_i| + 1)\big) = T + 2m \qquad (1)
$$

**符号**：第一项为 Part A（每条 span 塌缩成 1 个 [MASK]），第二项为 Part B（每条 span 前置 1 个起始符 [s]）； $m$：span 条数。代入 $T$=512、15% 覆盖、均长 3： $m \approx 26$， $|x_{\mathrm{in}}| = 564 \approx 1.10\thinspace T$，被监督位置只有 span 内容 77 个。CLM 同一序列 512 个前向位置换 512 个监督位；GLM 564 个前向位置换 77 个——**每被监督 token 的前向成本约为 CLM 的 7.3 倍**。「理解力红利」按这个倍率付费；练习 3 推导一般式 $(1 + 2r/\mu)/r$。

**谱系坐标。** 本书五家案例中，GLM 是唯一 objective 自研的谱系（Llama / DeepSeek / Qwen / Kimi 全部 CLM 主线，5.1–5.3、5.5）；2D 位置编码是这个 objective 反向决定的架构件——缺 $p^{(1)}$ 不知在填哪个空、缺 $p^{(2)}$ 不知空内第几（3.1 §4.3 的必要性论证），需求清单由 blank infilling 写出，架构件只是应答。

### 2.2 2D 位置编码的退出：退化角读法

GLM-4.5/4.6/4.7 的 config 没有任何 2D 位置字段，有的是 rope_theta = $10^{6}$——谱系起点的标志性架构件没有进入当代旗舰。这不是败退叙事，机制层面的读法有两步。

第一步，服务形态落在 GLM objective 空间的退化角上。取 $m$=1、span = 后缀，GLM 逐符号退化为 prefix-LM（3.1 §4.4 已证）；对话服务的生成形态——给定 prompt、自回归续写——正是这个角。退化角上 $p^{(1)}$ 恒等于唯一空白的地址（无信息载荷）， $p^{(2)}$ 退化为后缀内偏移——位置需求与 CLM 完全相同，2D 寻址的两维各失去一维。

第二步，RoPE 是该需求的更优参数化。相对性（score 只依赖位置差）、无查表参数、外推账可算（2.5）——当「多空白寻址」不再是主用法，绝对位置表 ($T_{\max} + L_{\max}) \times d$ 的查表方案在每一条对照轴上都不占优。config 里 rope_theta 字段的出现与 2D 字段的缺席，是「退化角成为主用法」这一事实的序列化痕迹。

边界必须说清：config 只能证明**架构件**退出，证明不了 **objective** 退出。blank infilling 是数据侧构造（3.1 §4.2 的管线），在 config 里不留字段；GLM-4.5 预训练是否仍混合 blank infilling 变体，report 摘要层面未锁定，列入 §5。

### 2.3 GLM-130B：双语 130B 与稳定性记录

GLM-130B（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]）的论文标题即 *An Open Bilingual Pre-trained Model*：中英双语、130B、开放权重。objective 一脉延续——GLM 2021 的文档级长 blank 变体在此记为 [gMASK] token（3.1 §4.5 已登记出处），即「单条 span 取原文 50%–100%」的长文生成模式。

该论文的中心工程内容是把 100B 量级训练的不稳定现象写成可复核记录：loss spike 与发散的发生条件、处置手段逐项在案（[A]）。这类问题的一般数学形态本书已在 2.7 推导——Post-LN 的末层梯度集中（Xiong et al. 定理 1）、warmup 的数学必然性、DeepNorm 用 $(\alpha, \beta)$ 常数把更新幅度与深度解耦（2.7 式 (14)(14a)）。130B 论文里每一项处置与该机制家族的逐项对应，本 pass 未复核原文、不逐字段断言（其 config 未纳入实测序列，谱系表该行留白即此意）；该对应关系的建立留为练习 4 与 §5 开放问题。谱系定位一句话：它是 GLM objective 在 100B 量级的可训练性证据，且满足 [A] 三件套（paper + 官方代码谱系 + 开放权重），使 blank infilling 路线在 100B 档有可审计的公开实例。

### 2.4 GLM-4.5/4.6/4.7：旗舰字段的零件坐标

三代同 config（§3 验证同构），逐列对到账上。

**GQA $g$=12 与 KV cache 账。** cache 公式（2.4 式 (9)）：

$$
M_{\mathrm{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_{\mathrm{head}} \cdot T \cdot b \qquad (2)
$$

**符号**： $L$：层数； $H_{kv}$：KV head 数； $d_{\mathrm{head}}$：每头维度； $T$：序列长度； $b$：元素字节数（bf16 取 2）。代入 $L$=92、 $H_{kv}$=8、 $d_{\mathrm{head}}$=128：每 token $2 \times 92 \times 8 \times 128 = 188416$ 元素 = 368 KiB；整窗 $T$=202752 时 $\approx$ 71.1 GiB（ $B$=1）。MHA 反事实（ $H_{kv}$=96）恰乘 $g$=12：约 854 GiB——200K 级窗口同样把 GQA 从优化变成前提。 $g$=12 的取点介于 Llama-3.1-70B（ $g$=8）与 Qwen3-235B（ $g$=16）之间，5.6 对照表回收。

**MoE 稀疏度取点。** 160 routed + 1 shared、top-8，层内口径 $`\rho = 9/161 \approx 5.6\%`$；总量口径 32/355 ≈ 9.0%。在 4.5 §1 的稀疏度谱上居中：DeepSeek-V3 层内 3.5%（256+1 top-8）、Qwen3-Next 2.1%（512+1 top-10）、GLM 5.6%。expert 中间维 1536 与 Qwen3-235B 同值，是细粒度切分（4.5 式 (14)(15)：expert 数乘 $m$、中间维除 $m$）落在相近粒度档的信号。

**first_k_dense_replace = 3。** 前 3 层 dense——与 DeepSeek-V3 同取 3，5.2 §6 的两条机制读法照搬：浅层表示未分化、路由信号噪声最大，dense 前导层给统一基底变换；浅层 MoE 均衡回路早期最不稳，dense 起步把振荡隔离在表示基底之外。K2 取 1（5.5），三家取点无公开消融，列入 §5。另记两笔字段级证据。其一，GLM-4.6 config 的 MoE 字段名（first_k_dense_replace、moe_intermediate_size、norm_topk_prob、routed_scaling_factor、n_group/topk_group）与 DeepSeek-V3 config 同形——字段字典层面的谱系信号；但 architectures = Glm4MoeForCausalLM、model_type = glm4_moe（自有架构类，config 实测），强度因此低于 K2 的 architectures = DeepseekV3ForCausalLM（5.5 的复用证据）：GLM 借的是 schema 而非类。其二，路由的 config 落点：`norm_topk_prob = true` + `routed_scaling_factor = 2.5` 是 4.5 式 (16)(17) 族「选中集内归一 + 缩放常数」的签名（取点同 5.2 §6 的 V3）；`n_group = 1`、`topk_group = 1` 把 V3 的组限定分发（8 选 4，4.5 §7 的通信动机）退化为全域路由。打分函数形态（sigmoid/softmax）config 无字段，未锁定（§5）。

**MTP ×1：草稿模块是整件 MoE 层。** config 实测 `num_nextn_predict_layers = 1`；GLM-4.5 report 明写 "we add an MoE layer as the MTP layer"（[A]）——即 4.5 §6 的 $D$=1 链式模块在 GLM 的取点不是小头而是完整一层 MoE（161 expert + attention），训练期监督加密、推理期 self-speculative 草稿、可整体弃置（机制推导归 4.5 §6，不重复）。参数量账进 §4 式 (4)。

**attention_bias = true 且 use_qk_norm = true：bias 与 QK-Norm 叠加。** 两个字段同 config 实测（2026-11-12）：GLM-4.5/4.6/4.7 不是「保留 bias、拒绝 QK-Norm」，而是两件都要——GLM-4.5 report Table 1 明写 QK-Norm: Yes，正文自述 "incorporate QK-Norm to stabilize the range of attention logits"（[A]）。与 Qwen3 的分野因此不是二选一：Qwen3 是纯 QK-Norm（attention_bias=false，5.3 §2.2），GLM 是叠加——QK-Norm 把 logit 幅度结构性封顶 $|z| \le \sqrt{d_k}$，bias 保留逐通道常数位移的自由度，同一个稳定性问题（logit 失控 → softmax 饱和 → 梯度消失，2.3 §3.4）上 GLM 多付一份可学参数。bias 参数量每层 $12288 + 1024 + 1024 = 14336$ 个、92 层合计约 1.3M——参数的边角料；叠加是否冗余（封顶之后位移自由度还剩多少表达贡献）无公开消融，列入 §5。

**rope_theta = $10^{6}$ + partial 0.5，原生覆盖。** config 实测 `partial_rotary_factor = 0.5`：128 维 head_dim 只旋 64 维，位置信号集中在旋转维上、其余 64 维纯内容内积（2.5 §4.7 的 partial RoPE 账）。最慢旋转通道波长公式里 $d$ 取旋转维数：

$$
\lambda_{\max} = 2\pi\thinspace \theta^{\thinspace1 - 2/d_{\mathrm{rope}}} \qquad (3)
$$

**符号**： $\theta$：RoPE base； $d_{\mathrm{rope}}$：旋转维数 = partial_rotary_factor × head_dim。代入 $\theta$=$10^{6}$、 $d_{\mathrm{rope}}$=64： $\lambda_{\max} \approx 4.1 \times 10^{6}$，202752 的窗口占最慢波长的约 5.0%——`rope_scaling` 字段存在但值为 null（config 实测），原生覆盖成立，对照 DeepSeek-V3 的 $\theta$=$10^{4}$ 必须外挂 YaRN factor 40 才到 163840（5.2 §3.6）。131072 → 202752 的代际扩窗在架构字段上只有 max_position_embeddings 一格变动，扩窗配方（数据与训练日程）属训练侧，[W] 边界内不断言。注意 GLM 的 partial 与 DeepSeek-V3 的 64/192 同刻度不同处境：V3 的 nope 维是 MLA 权重吸收的前提（2.5 §4.7），GLM 是纯 GQA、无吸收需求——partial 在此的动机公开材料未说明（§5）。

**训练侧口径登记。** 355B/32B、23T tokens 多阶段训练、hybrid reasoning（thinking / direct 双模式）、ARC（Agentic, Reasoning, Coding）定位——全部引 GLM-4.5 report 摘要（[A]）。双模式的机制本体归 training 维度；模型侧痕迹与 Qwen3 thinking 同型（5.3 已处理），本章不展开。

## 3. 无 paper 模型怎么读：GLM-4.6/4.7 示范

GLM-4.6 无独立 tech report，GLM-4.7 连 arXiv 条目都没有（搜索 0 条，2026-11-12）——两代的正确处理流程本身值得写成一节，这是 5.6 方法论章的预演。事实样式（全部本轮实测，2026-11-12）：

1. **GLM-4.6 的 README 只有 58 行**：0 个参数数字、5 个链接指向 GLM-4.5 资源。参数级断言的唯一合法来源是 4.5 report 摘要（355B/32B，[A]）——但引用它的前提是先证架构同构，否则「4.6 = 355B」就是无出处转述。
2. **同构证明 = config 逐字段 diff**：GLM-4.5 与 4.6 两 config 同拉对照，唯一差异是 max_position_embeddings 131072 → 202752。架构同构成立，355B/32B 的援引才合法；「4.6 = 4.5 架构 + 长窗化」是 config 级事实。能力增量（编码等宣称）只是 model card 级证据，[W] 降级引用。
3. **GLM-4.7 的 config 与 4.6 逐字段完全一致**：同 checkpoint 迭代线，架构零变更、能力升级。能力宣称引旁证渠道（OpenRouter 模型清单，2026-11-12：「Z.ai's latest flagship…enhanced programming + multi-step reasoning」，ctx 标称 204800）——旁证一律标日期；标称 ctx 与 config 字段 202752 的口径差已在 §1 表注五登记。
4. **一般化为四步**：(i) 先拉 config，与最近前代逐字段 diff；(ii) 无 report 时，参数/配方数字找最近前代 report，且以 (i) 的同构验证为援引前提；(iii) model card / API 旁证全部标日期、按 [W] 降级；(iv) 训练配方类断言在无 paper 时不做。四步在 4.6/4.7 上的完整走样即本节——5.6 会把同一流程展开成通用 checklist。

一个反面提醒：4.6 README 的 58 行里没有任何可独立成立的架构数字，全部数值断言都要么落到 config（架构字段）要么落到 4.5 report（参数与配方）——「新模型发布即有新事实源」的直觉在这类迭代线上不成立，发布物的信息量要用行数与指针数衡量。

## 4. 旗舰 config 精读：GLM-4.6 逐字段

证据等级 [W]：架构事实以 HF config.json 实测为准（2026-11-12），GLM-4.5 report [A] 提供参数与配方口径。逐字段读法（按 2.8 §2 的方法：每个字段在公式里点名位置）：

| 字段 | 值 | 读法 |
| --- | --- | --- |
| `num_hidden_layers` | 92 | 3 dense + 89 MoE，由 `first_k_dense_replace` 切分 |
| `hidden_size` | 5120 | stream 宽度 $d$ |
| `num_attention_heads` / `num_key_value_heads` | 96 / 8 | GQA $g$=12；q 投影 5120 → 12288，K/V 各 → 1024 |
| `head_dim` | 128 | $\neq$ 5120/96 ≈ 53.3（2.8 §8 实例；§1 表注四） |
| `use_qk_norm` | true | QK-Norm：与 QKV bias 叠加（§2.4）；4.5 report Table 1「QK-Norm: Yes」互证 |
| `attention_bias` | true | QKV bias；叠加取点见 §2.4 |
| `n_routed_experts` / `n_shared_experts` | 160 / 1 | 细粒度 MoE，粒度档同 Qwen3-235B（1536 中间维）；shared 为常驻公共通道（4.5 §3.1） |
| `num_experts_per_tok` | 8 | 每 token 加 shared 共 9 个，层内 $`\rho \approx 5.6\%`$ |
| `moe_intermediate_size` | 1536 | 单 expert FFN 中间维（shared 同宽——式 (4) 闭合反证） |
| `intermediate_size` | 12288 | 前 3 个 dense 层的 FFN 中间维 |
| `first_k_dense_replace` | 3 | 前 3 层 dense；字段名与 DeepSeek-V3 同形（§2.4） |
| `norm_topk_prob` / `routed_scaling_factor` | true / 2.5 | 选中集内归一 + 缩放常数（4.5 式 (16)(17) 族落点，取点同 V3） |
| `n_group` / `topk_group` | 1 / 1 | 组限定分发退化为全域路由（对照 V3 的 8 选 4） |
| `num_nextn_predict_layers` | 1 | MTP $D$=1，整件 MoE 层（§2.4、式 (4)） |
| `max_position_embeddings` | 202752 | 4.5 → 4.6 唯一 diff 字段；model card「200K」互证 |
| `rope_theta` | $10^{6}$ | 式 (3)： $\lambda_{\max} \approx 4.1$M，原生覆盖全窗 |
| `partial_rotary_factor` | 0.5 | 只旋 128 维中的 64 维（2.5 §4.7） |
| `rope_scaling` | null | 字段存在、值为空——原生覆盖的 config 签名 |
| `hidden_act` | "silu" | gated FFN 门函数：SwiGLU 三矩阵口径坐实 |
| `rms_norm_eps` | $10^{-5}$ | RMSNorm 的字段签名（§1 谱系表 norm 列依据） |
| `vocab_size` | 151552 | 与 Qwen 151936 同量级不同档（2.2 五档账）；tokenizer 是否同源本 pass 未锁定 |
| `tie_word_embeddings` | false | 词表参数 $2Vd$ |
| `architectures` / `model_type` | Glm4MoeForCausalLM / glm4_moe | 自有架构类；借 DeepSeek schema 不借类（§2.4） |

层栈结构（shape 走读）：

```
GLM-4.6（92 层，d = 5120，untied 词表）
┌─ 层 1–3 : dense ── GQA(96Q/8KV × 128，RoPE θ=10^6 旋 64 维，QK-Norm + QKV bias) + SwiGLU FFN(d_ff = 12288)
├─ 层 4–92: MoE ──── GQA(同上) + router(160 选 8，top-k 内归一 × 2.5) + 160 routed + 1 shared（d_ff = 1536）
├─ 训练期外挂: MTP × 1 ── 整件 MoE 层（161 expert + attention；推理可弃，4.5 §6）
└─ 词表   : 151552 × 5120 × 2 ≈ 1.55B（embedding + 输出头；官方 355B/32B 口径不含此项）
```

**参数复算：全字段锁定后的闭合。** 355B 是 report 摘要口径——report Table 1 注明口径「含 MTP 层参数、不含 word embeddings 与输出层」（[A]）；config 侧所需字段已全部实测（fix round 1 复核 hf-mirror config，2026-11-12），复算因此可以闭合而非止步：

$$
N = \underbrace{90 \cdot 161 \cdot (3 \cdot 5120 \cdot 1536)}_{\text{experts (89 MoE + 1 MTP layers)}} + \underbrace{93 \cdot (5120 \cdot 12288 + 2 \cdot 5120 \cdot 1024 + 12288 \cdot 5120)}_{\text{attention (92 + 1 MTP layers)}} + \underbrace{3 \cdot (3 \cdot 5120 \cdot 12288)}_{\text{dense FFN}} + \underbrace{90 \cdot 5120 \cdot 160}_{\text{router}} \qquad (4)
$$

**符号**：expert 按 SwiGLU 三矩阵计（`hidden_act = "silu"` 实测，gated 口径坐实）；attention 项 = Q/K/V/O 四个投影；MTP = 一个完整 MoE 层（report 原文 "we add an MoE layer as the MTP layer"），其 expert / attention / router 已并入前三项的系数 90 / 93 / 90。数值：experts ≈ 341.9B、attention ≈ 12.7B、dense FFN ≈ 0.57B、router ≈ 0.07B，合计 ≈ 355.2B——与官方 355B 闭合到 0.2B（约 0.06%）以内，残差即 QKV bias（92 层 × 14336 ≈ 1.3M）、norm 增益与 MTP 拼接投影小项。闭合同时反证两个未单列字段的取值：shared expert 与 routed 同宽 1536（否则总和溢出 355B），MTP 恰为一层。词表两侧 $2Vd$ ≈ 1.55B 按官方口径不入账。本节初稿曾以「残差 4.9B、归属未锁定字段」收尾——dense FFN 宽度与 MTP 字段本就在 config 里，初稿的「未锁定」认定是漏读（fix round 1 更正）；残差框架本身保留。

**激活侧（每 token，同口径含 MTP、不含词表）**：attention $93 \times 136.3$M ≈ 12.7B + MoE 激活 $90 \times 9 \times 23.6$M ≈ 19.1B + dense FFN 0.57B ≈ 32.4B，与官方「32B activated」（report 摘要口径，[A]）吻合到 1%。推理期 MTP 可整体弃置（4.5 §6），弃置后激活 ≈ 32.0B——官方口径含 MTP 是训练配置口径，两个数字都登记。

## 5. 技术开放问题清单

1. **blank infilling 在旗舰代的存否不可见**：config 只能证明 2D 位置表退出（§2.2），GLM-4.5 预训练是否仍混合 blank infilling 变体，report 摘要层面无锁定证据——谱系「独门 objective」的当代存否是公开材料的缺口。
2. **GLM-130B 稳定性处置与机制家族的逐项对应未建立**：论文逐项处置（[A]）与 2.7 的 Post-LN / warmup / DeepNorm 分析之间的映射本书未复核，其 config 亦未实测。
3. **bias + QK-Norm 叠加的冗余度无消融**：GLM-4.5/4.6/4.7 两件同用（§2.4），Qwen3 纯 QK-Norm；封顶之后 bias 的位移自由度还剩多少表达贡献、去掉它对 GLM 的质量与训练稳定性曲线影响多少，公开材料无逐项数据。
4. **first_k_dense_replace 的最优性**：3（DeepSeek-V3、GLM-4.5/4.6）与 1（K2）之间无公开消融（同 5.2 §7 问题 1，跨谱系共用）。
5. **4.6/4.7 能力增量不可归因**：[W] 边界——config 同构下，ctx 扩窗配方与 post-training 改动均无 paper，能力宣称只能标日期登记（§3）。
6. **GLM 侧 partial RoPE 的动机未说明**：纯 GQA 无 MLA 权重吸收需求（对照 2.5 §4.7 的 V3 处境），64/128 旋转配比的收益项公开材料未给消融；router 打分函数形态（sigmoid/softmax）config 无字段、未锁定（§2.4）。

## 总结

| 内容 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| 谱系两段式 | §1 表 | objective 驱动期（2021–2022，机制自研）→ 收敛期（GLM-4.5 起，标准零件箱）；过渡带 ChatGLM→GLM-4 由代际报告一篇覆盖 |
| blank infilling 后果账 | 式 (1)；7.3× 监督成本（ $T$=512、15%、 $\mu$=3） | 买到：可生成判定通过 + 理解生成同体（单 stack 单 loss，模式切换在数据侧）；付掉：监督密度 15% + 序列开销 $2m$ |
| 2D 位置退出 | 退化角读法 | $m$=1 后缀特例 ⊂ prefix-LM（3.1 §4.4）；退化角上两维各失一维，RoPE 占优；objective 存否不可见（§5 问题 1） |
| GLM-130B | 130B 双语开放；[gMASK] | 100B 量级稳定性记录 [A]；与 2.7 机制家族的逐项对应留练习 4 |
| 旗舰零件坐标 | 式 (2)：71.1 GiB @ 202752（MHA 反事实 854 GiB）； $\rho$=9/161≈5.6%；式 (3)： $\lambda_{\max}$≈4.1M（partial 0.5） | $g$=12 居中；first 3 dense 同 V3；bias + QK-Norm 叠加 vs Qwen3 纯 QK-Norm；MTP ×1 为整件 MoE 层； $\theta$=$10^{6}$ 原生覆盖 |
| 无 paper 读法 | §3 四步 | 4.6 README 58 行 0 数字 5 链接；config diff 唯一字段；4.7 config 零差异；旁证标日期 |
| config 精读 | 式 (4)：全字段闭合复算 ≈ 355.2B vs 官方 355B（含 MTP、不含词表口径），残差 0.06% | 激活 ≈ 32.4B ≈ 官方 32B；MTP 弃置后 ≈ 32.0B；355B/32B 照抄官方口径 |
| 开放问题 | §5 六条 | objective 存否、130B 处置对应、叠加冗余度、dense 前导层数、[W] 归因、partial 动机与打分函数形态 |

## 练习

1.（计算）用式 (2) 复核：GLM-4.6（ $L$=92、 $H_{kv}$=8、 $d_{\mathrm{head}}$=128）在 $T$=202752、bf16、 $B$=1 下的 KV cache 总量（71.1 GiB），以及 MHA 反事实（ $H_{kv}$=96）的 854 GiB。再与 Qwen3-235B 的每 token 斜率 188 KiB（5.3 §2.1）对照：用 $g$=12 与 $g$=16 解释两家的取点差，并算 $B$=32 并发时两家各自的整窗 cache 总量。
2.（计算）参数账。(a) 按式 (4) 重算 355.2B 的四个分项，验证到 0.1B，并指出 90/93/90 三个系数各自由哪几层贡献；(b) 重算激活侧 32.4B 与 MTP 弃置后的 32.0B，说明官方「含 MTP、不含词表两侧」口径在两个数字上的体现；(c) 反证练习：若 shared expert 中间维实为 $2 \times 1536$，式 (4) 总和变为多少、与官方 355B 的差变为多少——以此说明「闭合反证字段取值」的论证逻辑。
3.（推导）推广式 (1)：覆盖率 $r$、span 均长 $\mu$ 时，证 $m = rT/\mu$、 $|x_{\mathrm{in}}| = T(1 + 2r/\mu)$，并推出「每被监督 token 的前向成本 / CLM」= $(1 + 2r/\mu)/r = 1/r + 2/\mu$。代入 $(r, \mu) \in \lbrace(0.15, 3), (0.15, 10), (0.50, 64)\rbrace$ 各算一倍率，并解释两个方向的渐近： $\mu$ 增大只把倍率压向 $1/r$（序列开销项消失）， $r$ 增大才把监督密度项压向 1——文档级 [gMASK] 变体（3.1 §4.2：span 取原文 50%–100%）同时取高 $r$ 高 $\mu$，倍率趋 1 的机制在此。
4.（代码 + 调研）(a) 写约 15 行教学代码（教学用，非优化实现）：输入两个 config.json 的 dict，输出逐字段 diff（键、旧值、新值）；以 GLM-4.5/4.6/4.7 三份 config 为测试用例，期望输出分别为单字段（max_position_embeddings）与零字段。(b) 读 GLM-130B 论文（arXiv:2210.02414）的稳定性章节，把每一项处置对应到 2.7 的机制（Post-LN 梯度集中、warmup、DeepNorm $(\alpha, \beta)$），列一张对照表——这是 §5 问题 2 的完整答卷。

## 参考

- [Du et al., 2021](../references.md)：GLM: General Language Model Pretraining with Autoregressive Blank Infilling，arXiv:2103.10360，[A]（key: glm）——blank infilling、2D 位置编码、Poisson span；2021 代码在 THUDM/GLM（3.1 已锚定，zai-org/GLM 非 2021 论文代码）
- [Zeng et al., 2022](../references.md)：GLM-130B: An Open Bilingual Pre-trained Model，arXiv:2210.02414，[A]（key: glm-130b）——双语 130B 开放权重、[gMASK]、100B 级稳定性记录
- [Team GLM, 2024](../references.md)：ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools，arXiv:2406.12793，[A]（key: glm-4）——ChatGLM → GLM-4 代际报告，All Tools 定位
- [GLM-4.5 Team, 2025](../references.md)：GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models，arXiv:2508.06471，[A]（key: glm-4.5）——355B/32B、23T tokens、hybrid reasoning 的摘要口径出处
- [GLM-V Team, 2025](../references.md)：GLM-4.5V and GLM-4.1V-Thinking，arXiv:2507.01006，[A]（key: glm-4.5v）——视觉版存在性登记，细节归 P8
- [Z.ai, GLM-4.6 config.json 与 model card](../references.md)：[W]（key: glm-4.6）——本章 §4 逐字段精读对象；无独立 tech report，README 58 行、5 个链接引 4.5（2026-11-12 实测）
- [Z.ai, GLM-4.7 config.json 与 model card](../references.md)：[W]（key: glm-4.7）——config 与 4.6 逐字段一致；无 arXiv（2026-11-12 实测）
- [Z.ai, zai-org/GLM-4 官方 repo](../references.md)：[A]（key: glm-4-repo）——GLM-4 系列代码锚点（THUDM/GLM-4 已 301 至此）
- [Z.ai, zai-org/GLM-4.5 官方 repo](../references.md)：[A]（key: glm-4.5-repo）——GLM-4.5 系列代码锚点
