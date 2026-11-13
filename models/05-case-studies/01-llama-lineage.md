# 5.1 Llama 谱系：开源参照系（The Llama Lineage）

> 本章把 Llama 四代（1 → 2 → 3.1 → 4）当作一条「每代只动一两个组件」的受控谱系来读：配置谱系总表与逐代 diff 表；五个标志性决策（标准配方、GQA、vocab 128K、128K ctx、MoE + iRoPE + early fusion）的公式级分析；Llama-4 Scout config 逐字段精读；技术开放问题清单 | 前置：2.2（vocab 三方账）、2.4（KV cache 式 (9)）、2.5（RoPE 式 (13)、NoPE、外推失效）、2.8（标准配方式 (1)–(4)、参数对账式 (6)）、4.1（GQA 账）、4.5（MoE 路由）、4.6（长上下文四笔账）| 锚点：Llama-1 论文 [A]、Llama-2 论文 [A]、Llama-3 herd 论文 [A]、Llama-3.1-8B/70B config [A]、Llama-4 blog + Scout/Maverick config [W]、meta-llama/llama-models repo [A]

## 1. 配置谱系总表

四代 12 个型号摆成一张表，先看不动的四件：RMSNorm + Pre-LN、SwiGLU、无 bias、non-tied embedding 从 Llama-1 到 Llama-4 逐代保留——2.8 式 (1) 的「归一化读、加法写」骨架逐代原样保留，动的全部在组件层。变的只有四个轴：attention 的 KV 头数、词表、上下文与 RoPE 频率、FFN 的 dense/MoE 形态。

| model | total/active 参数 | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE（experts/top-k/shared） | norm | vocab | ctx（max_position_embeddings） | rope_theta | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-1-7B | 6.7B | 32 | 4096 | 32/32 | 128 | MHA | — | RMSNorm | 32000 | 2048 | $10^{4}$ | llama-1 [A] |
| Llama-1-13B | 13.0B | 40 | 5120 | 40/40 | 128 | MHA | — | RMSNorm | 32000 | 2048 | $10^{4}$ | llama-1 [A] |
| Llama-1-33B | 32.5B | 60 | 6656 | 52/52 | 128 | MHA | — | RMSNorm | 32000 | 2048 | $10^{4}$ | llama-1 [A] |
| Llama-1-65B | 65.2B | 80 | 8192 | 64/64 | 128 | MHA | — | RMSNorm | 32000 | 2048 | $10^{4}$ | llama-1 [A] |
| Llama-2-7B | 6.7B | 32 | 4096 | 32/32 | 128 | MHA | — | RMSNorm | 32000 | 4096 | $10^{4}$ | llama-2 [A] |
| Llama-2-13B | 13.0B | 40 | 5120 | 40/40 | 128 | MHA | — | RMSNorm | 32000 | 4096 | $10^{4}$ | llama-2 [A] |
| Llama-2-70B | 69B | 80 | 8192 | 64/8 | 128 | GQA-8 | — | RMSNorm | 32000 | 4096 | $10^{4}$ | llama-2 [A] |
| Llama-3.1-8B | 8.03B | 32 | 4096 | 32/8 | 128 | GQA-8 | — | RMSNorm | 128256 | 131072 | $5\times10^{5}$ | llama-3.1-8b-config [A] |
| Llama-3.1-70B | ≈70.6B | 80 | 8192 | 64/8 | 128 | GQA-8 | — | RMSNorm | 128256 | 131072 | $5\times10^{5}$ | llama-3.1-70b-config [A] |
| Llama-3.1-405B | 405B | 126 | 16384 | 128/8 | 128 | GQA-8 | — | RMSNorm | 128256 | 131072 | $5\times10^{5}$ | llama3 [A]（架构表） |
| Llama-4 Scout | 109B/17B（blog 口径） | 48 | 5120 | 40/8 | 128 | GQA-8 + iRoPE 3:1 + chunk 8192 | 16+1 top-1（层层） | RMSNorm | 202048 | 10485760 | $5\times10^{5}$（+llama3 式 f16） | llama-4 [W] |
| Llama-4 Maverick | 400B/17B（blog 口径） | 48 | 5120 | 40/8 | 128 | GQA-8 + iRoPE 3:1 + chunk 8192 | 128+1 top-1（隔层） | RMSNorm | 202048 | 1048576 | $5\times10^{5}$ | llama-4 [W] |

表注三件事。其一，**口径**：Llama-1/2 各行的层数、宽度、头数与参数量出自各自论文的架构表（[Touvron et al., 2023a](../references.md)，arXiv:2302.13971，[A]；[Touvron et al., 2023b](../references.md)，arXiv:2307.09288，[A]）；Llama-3.1-8B/70B 为 config 实测（8.03B 即 2.8 式 (7) 对账到个位的值，70B 行参数量按同口径推算）；**405B 的 config 未公开**（gated 且无公开镜像），该行数字引 herd paper 架构表（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]），无法与 config 互证——谱系里唯一的证据缺口。其二，Llama-4 两行的 config 字段经 unsloth 镜像实测（2026-11-12，meta-llama gated），total/active 参数引官方 blog 原文口径（[Meta, Llama-4 blog + config](../references.md)，[W]）；norm/activation 等未单列字段以官方 repo 的模型定义为准（[Meta, llama-models repo](../references.md)，[A]）。其三，active 参数照抄 blog 口径（17B），不自行换算——其精确构成是 §7 的开放问题 3。

沿四个变量轴读这张表：KV 头数轴上，MHA 在 70B 处先破（Llama-2），随后 GQA-8 普及全系（Llama-3.1）并冻结为默认；词表轴上 32000 冻了两代，一步跳到 128256 再到 202048；上下文轴上 2048 → 4096 → 131072 → 10485760，每步都换一套 RoPE 侧工具（base、staged extension、chunk + 缩放）；FFN 轴上前三代 dense SwiGLU，Llama-4 换 MoE top-1。以下四节逐个轴拆账：每个改动换掉什么、解决什么、代价是什么。

## 2. Llama-1：标准配方的确立

Llama-1 论文的架构一节篇幅很短，因为它做的不是发明而是定齐：把当时散落在各家报告里的四个组件选择固定为同一套配方，并以 7B–65B 四档开放权重发布。相对 GPT-2 配方的逐项替换（机制推导全部在 P2 对应章）：

| 组件 | GPT-2 配方 | Llama-1 配方 | 机制与代价 |
| --- | --- | --- | --- |
| norm | LayerNorm（ $\gamma, \beta$ 双参数） | RMSNorm（仅 $\gamma$） | 去掉 re-centering：省一次均值归约与 $\beta$ 参数；代价是均值漂移不矫正（2.7 式 (5) 与 §3 的对照，[Zhang & Sennrich, 2019](../references.md)，arXiv:1910.07467，[B]） |
| norm 摆放 | Pre-LN + final LN | Pre-LN + final RMSNorm | 恒等梯度通路（2.7 式 (8)）保持不变 |
| 位置编码 | learned absolute 查表加进 stream | RoPE：attention 内旋转 | score 只依赖相对位移（2.5 式 (13)）；stream 全程不含位置向量（[Su et al., 2021](../references.md)，arXiv:2104.09864，[A]）；代价是外推出域（2.5 §6），§4.2 与 §5.2 的两代补救都从这条代价出发 |
| FFN | 两矩阵 GELU， $d_{\mathrm{ff}} = 4d$ | 三矩阵 SwiGLU， $d_{\mathrm{ff}} \approx 2.7d$ | 等参数规则：三矩阵 FFN 与两矩阵等参时中间维取 $\tfrac{2}{3} \times 4d \approx 2.67d$，11008 是向上取 256 倍数的落点（2.6 式 (6)(8) 与 §3.2，[Shazeer, 2020](../references.md)，arXiv:2002.05202，[B]） |
| bias | attention 与 FFN 全带 | 全无 | 低阶项（ $O(Ld)$），去掉是现代配方收敛点而非「被证有害」（2.8 §8 误解二） |
| tying | tied（ $W_U = E$） | untied | 多付 $Vd$ 参数（7B 档 $2Vd = 2.6\times10^{8}$，约 4%）换输入/输出表征解耦（2.2 §9，[Press & Wolf, 2016](../references.md)，arXiv:1608.05859，[B]） |

这套配方就是 2.8 式 (1)–(4) 在 MHA 退化形（ $H_{kv} = H_q$，分组映射 $g(h)$ 为恒等）下的完整形态：RMSNorm 两次「读」、attention 与 SwiGLU 两次「写」、final norm 收尾、独立输出头。四档尺寸（32 层 4096 宽 → 80 层 8192 宽）共用同一骨架，只有 $L$ 与 $d$ 在变。训练侧只登记一句：7B/13B 用 1.0T tokens、33B/65B 用 1.4T tokens（论文 §2 口径），数据配比与训练流程归 training 维度。

Llama-1 的谱系意义不在任何单点创新，而在于它把「哪个组件进标准配方」变成了可引用的默认答案：之后三年里 P2 各章推导的每个组件变体（2.5 的 ALiBi/NoPE、2.6 的 GEGLU、2.7 的 reordered norm、4.1 的 MLA），都是以这套配方为基准系的扰动实验。

## 3. Llama-2：GQA 的引入点

Llama-2 全系三个尺寸里，只有 70B 的架构表多出一格变化：KV 头数 64 → 8。逐代 diff：

| 字段 | Llama-1 | Llama-2 | 技术原因 |
| --- | --- | --- | --- |
| ctx | 2048 | 4096 | 窗口翻倍；RoPE base 不动（ $10^{4}$），最短账见 §4.2 的波长口径 |
| 70B attention | MHA（64 KV） | GQA-8（8 KV） | cache 账，下文的式 (1) |
| 70B $d_{\mathrm{ff}}$ | 22016（ $\approx 2.69d$） | 28672（ $= 3.5d$） | FFN 加宽，参数从 65.2B 档到 69B |
| 7B / 13B | MHA | MHA（不变） | cache 压力在旗舰先到期，见下文 |
| 训练 tokens | 1.0T/1.4T | 2T | 数据侧，归 training；chat 线 RLHF 同 |

换掉什么、解决什么，用 KV cache 公式算（2.4 式 (9)，逐 token 逐层 K+V 各存 $H_{kv} \times D$ 元素）：

$$
\mathrm{bytes} = 2 \thinspace L \thinspace H_{kv} \thinspace D \thinspace T \thinspace B \thinspace b \qquad (1)
$$

**符号**： $L$：层数； $H_{kv}$：KV 头数； $D$：head 维； $T$：序列长； $B$：并发序列数； $b$：每元素字节数（bf16 取 2）。

代入 70B（ $L = 80$、 $D = 128$、bf16）：

| 方案 | 每 token | $T = 4096$ 单序列 | $B = 32$ 并发 |
| --- | --- | --- | --- |
| MHA-64（反事实） | 2.5 MiB | 10 GiB | 320 GiB |
| GQA-8（实装） | 320 KiB | 1.25 GiB | 40 GiB |

70B 权重 bf16 约 138 GB：MHA 下 $B = 32$ 的 cache（320 GiB ≈ 344 GB）是权重的约 2.5 倍，GQA-8 把它压到 0.3 倍。式 (1) 里 $L, D$ 由架构定、 $T, B$ 由负载定， $H_{kv}$ 是架构侧唯一能动的因子——GQA 就是把这个因子从 64 拧到 8，cache 与其读取带宽同步除 8（decode 的 cache 读取在分母里的位置见 2.4 式 (12)，机制展开归 4.1 §3）。

代价写在表示里： $g = 8$ 个 Q 头共享一对 K/V，K 侧行空间从 64 份塌到 8 份（4.1 §2 的透镜账）。GQA 论文的 T5 族 uptraining 对照给出 Pareto 形态：GQA-8 质量与 MHA 基线几乎重合、decode 速度接近 MQA，故取 8 KV 头为中间点（[Ainslie et al., 2023](../references.md)，arXiv:2305.13245，[B]，质量/速度对照为其 Table 1 / Fig 6 口径；更左端的外推是本书按机制标注，4.1 §3 已声明边界）。同论文还给出 retro-fit 路径：从 MHA checkpoint 组内 mean-pool 初始化 + 约 5% 预训练算力续训（[B]，论文口径）——Llama-2 70B 是否走 uptraining 论文未声明，本书不断言；这条路径的存在性是 §8 的一个论据。

为什么只有 70B 上 GQA：7B 在 $T = 4096$ 的 MHA cache 是每序列 2 GiB（≈2.15 GB），对 13.4 GB 权重， $B = 7$ 起反超；70B 是 10 GiB 对 138 GB， $B$ 一到两位数 cache 就反客为主。cache 压力 $\propto L \cdot T \cdot B$，旗舰的 $L$ 翻倍、且它是服务成本的主要承担者——墙先撞到谁，谁就先换。Llama-3.1 把 GQA-8 普及到 8B，是同一笔账在更长窗口（131072）下对小模型也到期：8B 满窗 MHA 反事实 cache 64 GiB、GQA-8 后 16 GiB（2.4 §5 的实例行）。

## 4. Llama-3.1：vocab 与 ctx 的两笔账

Llama-3.1 相对 Llama-2 的 config diff 里，变化幅度最大的不是层数而是 vocab_size：32000 → 128256。逐代 diff：

| 字段 | Llama-2 | Llama-3.1 | 技术原因 |
| --- | --- | --- | --- |
| vocab | 32000（SentencePiece BPE） | 128256（BPE，128K） | 式 (2)(3) 的三方账 |
| ctx | 4096 | 131072 | base $10^{4} \to 5\times10^{5}$ + 六阶段扩窗 |
| attention | 70B 独享 GQA-8 | 全系 GQA-8 | §3 的账在 128K 窗下对小模型同样到期 |
| 8B 档 $d_{\mathrm{ff}}$ | 11008 | 14336（ $= 3.5d$） | FFN 加宽对齐 70B 档比例 |
| 405B 档 | 无 | 126 层、 $d = 16384$、128 Q/8 KV、 $d_{\mathrm{ff}} = 53248$ | herd paper 架构表 [A] |
| 训练 tokens | 2T | 15T | herd paper [A]；数据侧归 training |

### 4.1 vocab 32000 → 128256：一笔随规模变号的账

词表改动牵动三笔账（2.2 §7 的框架），其中两笔可以精确算。

**参数账**（untied 下输入 embedding 加输出头，2.2 §7 / 2.8 式 (6)）：

$$
N_{\mathrm{vocab}} = 2 V d \qquad (2)
$$

**符号**： $V$：词表； $d$：hidden 宽。代入：8B 档 $2 \times 128256 \times 4096 = 1.05\times10^{9}$，占 8.03B 总参数的 13.1%；若停留在 32K 词表则为 $2.62\times10^{8}$，改动净增约 $7.9\times10^{8}$ 参数。同一改动在 405B 档： $2 \times 128256 \times 16384 = 4.2\times10^{9}$，占 405B 的 1.0%。**词表升级的参数代价是相对小模型 10 倍于旗舰的**—— $2Vd$ 与层参数 $O(Ld^{2})$ 的比值随 $Ld$ 反比下降。

**token 数账**。herd paper 给出同族对照：Llama-3 的 128K tokenizer 压缩率 3.94 chars/token，Llama-2 的为 3.17（[Grattafiori et al., 2024](../references.md)，[A]；2.2 §7 已引）。同一文本的 token 数反比于压缩率：

$$
T' = T \cdot \frac{3.17}{3.94} \approx 0.805 \thinspace T \qquad (3)
$$

即同一语料 token 数减约 19.5%。这一笔的收益与规模无关：训练 FLOPs $6ND$ 的 $D$、decode 的串行步数、attention 的 $O(T^{2})$ 项都按 token 计，同一份字节预算下全线打八折。合起来看 8B 档的 per-byte 前向 FLOPs：旧组合（ $V = 32K$）每 token $2N + 2Vd \approx 1.42\times10^{10}$、每字符 $1.42\times10^{10} / 3.17 \approx 4.5\times10^{9}$；新组合 $1.50\times10^{10} / 3.94 \approx 3.8\times10^{9}$——降约 15%（按论文同一 chars/token 口径算相对账；字节口径随语种分布变，结论方向不变）。到 405B 档，层部分 $2N$ 主导更强，词表改动对 per-token FLOPs 的影响落到 1% 量级，式 (3) 的八折近乎白拿。

第三笔算不精确但要登记：长尾冷启动——词表越大，稀有 token 的 embedding 行获得的梯度更新越稀（2.2 §7），这是唯一不随规模摊薄的代价；多语言覆盖（fertility，2.2 §6）则是 128K 的主要收益面。Llama-4 再把词表抬到 202048，机制账同式 (2)(3)，[W] 下其 tokenizer 构成无公开拆解。

### 4.2 ctx 4096 → 131072：base、波长与六阶段扩窗

RoPE 侧的第一笔账是 base。频率 schedule $\theta_j = \mathrm{base}^{-2j/D}$（2.5 式 (16)）下，最慢通道（ $j = D/2 - 1$）的波长为

$$
\lambda_{\max} = \frac{2\pi}{\theta_{D/2-1}} = 2\pi \cdot \mathrm{base}^{(D-2)/D} \qquad (4)
$$

**符号**： $D = 128$（head 维），指数 $(D-2)/D \approx 0.984$。代入：base $10^{4}$ 时 $\lambda_{\max} \approx 5.4\times10^{4}$；base $5\times10^{5}$ 时 $\lambda_{\max} \approx 2.6\times10^{6}$。对照窗口：Llama-1 的 $2048 / 5.4\times10^{4} \approx 1/26$，Llama-3.1 的 $131072 / 2.6\times10^{6} \approx 1/20$——base 抬 50 倍，最慢通道在窗内依然转不完一个周期，「窗长 / 最慢波长」的比值保持在同一量级。这就是 4.6 §2 的 base 路线：慢通道的角度不出训练域（2.5 §6 的外推失效机制被结构性避开），代价是慢通道的位移分辨力在短窗内变粗——所以扩窗不能只拧 base，还要配训练侧。

训练侧是 staged extension：从 8K 出发分六个阶段扩到 128K，405B 的扩窗预算约 800B tokens（herd paper 口径，[A]；4.6 §4 已拆：阶段化买的是逐段可校验的过程控制，不是 FLOPs 节省）。405B 上的 GQA 账顺手一记：满窗 131072 单序列 cache $= 2 \times 126 \times 8 \times 128 \times 131072 \times 2$ 字节 $\approx 63$ GiB；反事实 MHA-128 为 1008 GiB——16 倍差。在 405B + 128K 这个工作点，GQA 不是优化而是前提：没有它，单序列 cache 就超过半张 80GB 卡的显存一个数量级。

## 5. Llama-4：MoE、iRoPE 与 early fusion（[W]）

Llama-4 没有论文。本节证据分级：架构事实以 Scout/Maverick 的 config.json 字段为准（meta-llama gated，经 unsloth 镜像实测 2026-11-12）与官方 repo 的模型定义（[A]）；规模数字（109B/17B、400B/17B）以官方 blog 原文为准（[W]）；训练数据、配方、评测一律不断言。逐代 diff：

| 字段 | Llama-3.1 | Llama-4 Scout / Maverick | 技术原因 |
| --- | --- | --- | --- |
| FFN | dense SwiGLU | 16 / 128 routed experts + 1 shared、top-1 | 参数-算力解耦，式 (5)(6) |
| MoE 频率 | — | 层层（step 1）/ 隔层（step 2） | 路由影响面折中，§5.1 |
| 位置编码 | 全层 RoPE | 3:1 iRoPE + chunk 8192 + 对数温度 | 式 (7)(8)，§5.2 |
| ctx | 131072 | 10485760 / 1048576 | chunk + NoPE 层 +（Scout）llama3 式 scaling f16 |
| 多模态 | 无 | early fusion：vision tower 同模型内 | §5.3 |
| vocab | 128256 | 202048 | 同式 (2)(3) 的账；tokenizer 构成未公开 [W] |
| $d$ / 层数 | 4096–16384 / 32–126 | 统一 5120 / 48 | MoE 下容量由 expert 数承担，不再靠加深加宽 |

### 5.1 MoE：16/128 routed + 1 shared、top-1、层层与隔层

MoE 层把 dense FFN 换成 $N_e$ 份 routed expert 加 router，Llama-4 在此基础上再加一份常驻 shared expert——官方 blog 明写「each token is sent to both the shared expert and one routed expert」（Scout 16 / Maverick 128 routed，各配 1 shared，[W]；官方 modeling 代码同证），即 4.5 §3.1 的 shared expert 结构。每 token 的 FFN 输出为

$$
y \hspace{5mu}=\hspace{5mu} \sum_{i \in \mathrm{Top1}(s(x))} g_i(x) \thinspace E_i(x) + E_{s}(x) \hspace{5mu}=\hspace{5mu} g_{i^{\ast }}(x) \thinspace E_{i^{\ast }}(x) + E_{s}(x) \qquad (5)
$$

**符号**： $x \in \mathbb{R}^{d}$：层输入； $E_i$：第 $i$ 份 routed SwiGLU expert（中间维 8192，config 字段 intermediate_size）； $E_{s}$：常驻 shared expert，不经路由、不乘门控； $s(x)$：router 打分；$`g_{i^{*}}`$：选中者的门控权重（官方实现为原始 sigmoid 分直接乘入，不归一化）。top-1 是 Switch 路线（4.5 §2.3，[Fedus et al., 2021](../references.md)，arXiv:2101.03961，[A]）：dispatch 决策、通信与 combine 对照 top-2 各减半；其「等算力下质量不降」是 Switch 论文在其规模与设置下的对照结论，向 128 experts 的外推无公开证据（§7 开放问题 2）。router 侧的打分形态则部分可见：config 有 `router_aux_loss_coef = 0.001`（4.5 §2 的 aux loss 路线，非 V3 式 aux-loss-free）与 jitter = 0.0（无探索噪声），官方实现的亲和度为 sigmoid（4.5 §3.2 同族）。

解耦账用激活参数占比表示（官方 blog 口径，[W]）：

$$
\rho_{\mathrm{act}} \hspace{5mu}=\hspace{5mu} \frac{N_{\mathrm{act}}}{N_{\mathrm{tot}}} \hspace{5mu}=\hspace{5mu} \frac{17}{109} \approx 15.6 \text{％} \hspace{5mu}\hspace{5mu} (\mathrm{Scout}), \qquad \frac{17}{400} \approx 4.3 \text{％} \hspace{5mu}\hspace{5mu} (\mathrm{Maverick}) \qquad (6)
$$

**符号**： $N_{\mathrm{act}}, N_{\mathrm{tot}}$：每 token 激活参数与总参数（blog 口径照抄）； $\rho_{\mathrm{act}}$ 是整机参数占比，与 4.5 的层内稀疏度 $\rho = (k + K_s)/N_e$ 是两个口径，勿混。机制定位即 1.2 式 (9) 的 $C \approx 6 N_{\mathrm{act}} D$：FLOPs 跟着 17B 走，容量摊薄跟着 109B/400B 走。Maverick 的 4.3% 与 DeepSeek-V3 的 37B/671B（≈5.5%）同量级——高稀疏是 2025 前后旗舰 MoE 的公共工作点（跨家对照归 5.6）。

Scout 与 Maverick 的第二个差异在 MoE 频率：Scout 的 `interleave_moe_layer_step = 1`，48 层层层 MoE；Maverick 的 step = 2，24 层 MoE 与 24 层 dense FFN 交替。隔层设计的机制读法：dense 层是每 token 必过的容量地板，不经过 router——4.5 §4 的路由病理（塌缩、均衡失控）影响面被限制在一半层内；代价是同 expert 数下 routed 容量减半，Maverick 用 128 个 routed expert（Scout 的 8 倍）补回，总量反而更大。两种点法孰优无公开对照，[W] 下只登记结构。

### 5.2 iRoPE：3:1 无 RoPE 层、chunk 8192 与对数温度

Llama-4 的 config 里有一个前三代没有的字段：`no_rope_layers = [1, 1, 1, 0, ...]`，按 4 层周期取值——每第 4 层不做 RoPE（iRoPE，interleaved RoPE 的 config 级实证），48 层 = 12 个周期。配套字段 `attention_chunk_size = 8192`、`attn_temperature_tuning = true`、`attn_scale = 0.1`。层型分工：

```
Llama-4 文本侧 48 层 = 12 × 4 层周期（no_rope_layers 3:1）
┌─ RoPE 层 ×3 ── 在 8192 chunk 内打分：相对位移恒 < 8192，RoPE 不出域
│       Q/K 旋转（rope_theta 5e5；Scout 另叠 llama3 式 scaling，factor 16、original 8192）
└─ NoPE 层 ×1 ── 全局可见集；score 乘对数温度 m(t)（式 (8)），补偿长程熵稀释
FFN 侧（与层型正交）：Scout 层层 16+1 top-1；Maverick 隔层 128+1 top-1
```

机制链条分两环。第一环，RoPE 层的打分只依赖相对位移（2.5 式 (13)）：

$$
\langle R_t q,\hspace{5mu} R_s k \rangle = \langle q,\hspace{5mu} R_{s-t} k \rangle \qquad (7)
$$

**符号**： $R_t$：位置 $t$ 的块对角旋转。把 RoPE 层的可见集裁进 8192 chunk，则 $s - t$ 永不超出训练域——2.5 §6 的外推失效（角度落在未见区间）被结构性删除，长程压力整体转移到 NoPE 层。NoPE 没有频率域可出（2.5 §5；NoPE 的长度泛化实证见 [Kazemnejad et al., 2023](../references.md)，arXiv:2305.19466，[B]），它面对的是另一堵墙。

第二环，NoPE 层在 $T$ 达千万级的可见集上遭遇熵稀释：4.6 §1.3 的账——目标 key 所需的 logit 优势随 $\ln T$ 增长， $\Delta \ge \ln T$。Llama-4 的补偿是对 NoPE 层乘一个随位置对数增长的温度（公式形态以官方实现为准，[W]）：

$$
m(t) = 1 + \alpha \ln\negthinspace\left(1 + \mathrm{floor}\negthinspace\left( \frac{t}{T_c} \right)\right), \qquad \alpha = 0.1,\hspace{5mu}\hspace{5mu} T_c = 8192 \qquad (8)
$$

**符号**： $t$：query 位置； $\mathrm{floor}(t / T_c)$：chunk 序号。在 Scout 满窗 $t \approx 1.05\times10^{7}$ 处， $m \approx 1 + 0.1 \ln(1281) \approx 1.72$。score 整体放大 $m$ 倍，等效于把可用的 $\Delta$ 预算放大同一倍数——对 $\ln T$ 稀释的一阶补偿。乘性温度不是加性 nat 的精确等价（它同时放大全部 score 差，峰值与噪声一起锐化），这是它的机制边界。诚实声明：3:1 比例、chunk 尺寸、 $\alpha = 0.1$ 三个超参均无公开 ablation，式 (8) 与上文的机制读法是与 2.5/4.6 机制一致的重构，不是论文结论（[W]）。

Scout 另有 `rope_scaling = {rope_type: llama3, factor: 16, original_max_position_embeddings: 8192}`：llama3 型压缩——慢通道压缩、快通道不动，属 4.6 §2 统一压缩率曲线 $\eta_j$ 所收四族（PI / NTK-aware / YaRN / LongRoPE）之外的另一条参数化——把 RoPE 层的基窗标定到 $8192 \times 16 = 131072$，恰与 3.1 的窗口对齐。Maverick 的 rope_scaling 为 null。两者差异的官方解释缺失，与 10M / 1M 窗的定位差异联动（§7 开放问题 6）。10M 这个数字本身按 4.6 §5 的方法论只能算 claimed context：无第三方 RULER 类评测复核，effective context 存疑（§7 开放问题 5）。

### 5.3 early fusion 多模态

config 顶层字段 `architectures = Llama4ForConditionalGeneration` 与 `vision_config`（34 层、hidden 1408、patch 14、`pixel_shuffle_ratio = 0.5`、16 heads）同模型共存——这是 early-fusion 多模态的 config 级证据：视觉 tower 不是外挂 adapter，而是与文本共享同一条流水线的前段。token 流与账：

```
image [H, W, 3]
  │ patch 14：N = HW/14² 个 patch（448×448 → 1024）
  ▼
vision tower（34 层，hidden 1408）
  │ pixel_shuffle_ratio 0.5：token 数 ×0.25（1024 → 256）
  ▼
projector ──► 与文本 token 同一 embedding 空间 ──► 同一条 residual stream（48 层）
```

448×448 的图以 256 个 token 进流，从第 1 层起与文本 token 在同一套 attention（含 iRoPE 层型）里互相可见。对照 late fusion（cross-attention 插层、LLM 主干冻结的路线）：early fusion 的代价是图像 token 全程付 LLM 深度的 FLOPs 与 cache（式 (1) 的 $T$ 直接加上 256 量级），收益是跨模态交互从浅层开始、主干无模态专用参数。架构层能讲的到此为止；图文数据、冻结与训练配方未公开（[W]），VLM 侧的完整处理归 P8。

## 6. 旗舰 config 精读：Llama-4 Scout

把 2.8 §2 的读法套到 Scout 的 config.json（meta-llama gated，经 unsloth 镜像实测 2026-11-12，[W]）上：每个结构字段对号到公式与账。文本侧（text_config）：

| config 字段 | 值 | 角色 / 进哪笔账 |
| --- | --- | --- |
| model_type | llama4 | modeling 代码派发 |
| num_hidden_layers | 48 | $L$；式 (1) 与参数账 |
| hidden_size | 5120 | $d$；stream 宽 |
| num_attention_heads | 40 | $H_q$；score 与 FLOPs 按它计 |
| num_key_value_heads | 8 | $H_{kv}$；式 (1) 的 cache 因子（GQA-8， $g = 5$） |
| head_dim | 128 | $D$；**显式字段**（ $5120/40$ 恰好整除，仍按 2.8 §8 误解一的口径先查显式值） |
| intermediate_size | 8192 | expert FFN 中间维；式 (5) 的 $E_i$ 尺寸 |
| num_local_experts | 16 | routed expert 数 $N_e$；Scout 16 / Maverick 128（另各配 1 份常驻 shared，式 (5) 的 $E_s$） |
| num_experts_per_tok | 1 | routed 侧 top-1；式 (5) 第一项塌成单项（注意：是 top-1，不是 top-2；shared 恒激活不在此字段） |
| interleave_moe_layer_step | 1 | 层层 MoE；Maverick 为 2（隔层） |
| no_rope_layers | [1,1,1,0,...] | iRoPE 3:1；式 (7)(8) 的层型分工 |
| attention_chunk_size | 8192 | RoPE 层 chunk 宽；式 (7) 的域界 |
| attn_temperature_tuning / attn_scale | true / 0.1 | 式 (8) 的开关与 $\alpha$ |
| floor_scale | 8192 | 式 (8) 的 $T_c$：对数温度的 chunk 粒度除数 |
| use_qk_norm | true（Scout；Maverick 无） | q/k 投影后各过一次 RMSNorm（2.7 §7 的 QK-Norm 变体，仅 RoPE 层）；Maverick 未启用 |
| rope_theta | 500000 | RoPE base；式 (4) 的 $\lambda_{\max} \approx 2.6\times10^{6}$ |
| rope_scaling | llama3 式，factor 16，original 8192 | 4.6 §2 的 $\eta_j$ 曲线；基窗 $8192 \times 16 = 131072$ |
| max_position_embeddings | 10485760 | claimed ctx；effective 存疑（§7 问题 5） |
| vocab_size | 202048 | $V$；式 (2) 的 $2Vd = 2.07\times10^{9}$ |

顶层与视觉侧：`architectures = Llama4ForConditionalGeneration`（early fusion 的 config 级证据，§5.3）；`vision_config` = 34 层 / hidden 1408 / patch 14 / `pixel_shuffle_ratio = 0.5` / 16 heads（token 账见 §5.3 图）。

按 2.8 式 (6) 的口径起算（每 expert 三矩阵 $3 \times 5120 \times 8192 \approx 125.8\text{M}$；shared expert 按同尺寸计）：routed experts 合计 $16 \times 48 \times 125.8\text{M} \approx 96.6\text{B}$，shared expert $48 \times 125.8\text{M} \approx 6.0\text{B}$；attention 每层 62.9M（Q 26.2M + K/V 10.5M + O 26.2M），48 层约 3.0B；embedding 加输出头 2.07B——合计约 107.7B，加 vision tower（34 层、 $d$ = 1408，按 $12d^{2}$ 速算 ≈ 0.8B）与 projector、norm 增益，对 blog 的 109B 闭合到约 1B 余量。active 侧同样闭合：attention 3.0B + 每 token 1 routed + 1 shared（ $2 \times 6.0\text{B}$）+ embedding/输出头 2.07B ≈ 17.2B ≈ blog 的 17B——「17B active」的口径由此可读为文本侧每 token 激活参数，不含 vision tower。两个 blog 数字同时对上，是这套字段读法的自洽性检验；剩余余量的逐项归属见练习 4。

## 7. 技术开放问题清单

1. **405B config 未公开**：谱系表 405B 行引 herd paper 架构表，无法与 config 互证；其 rms_norm_eps、rope_scaling 等字段级细节无公开实测渠道。
2. **Llama-4 router 的训练侧行为不可见**：打分形态是可见的——config 有 `router_aux_loss_coef = 0.001`（aux loss 路线）与 jitter = 0.0，官方实现用 sigmoid 亲和度；但 128 routed experts top-1 下的实际负载分布、塌缩控制效果与 aux loss 系数的调度过程无 paper / 训练日志可查。
3. **109B/17B 的余量归属**：shared expert 的存在性已由官方 blog（「each token is sent to both」）与官方 modeling 代码锁定，§6 的账据此把两个 blog 数字闭合到约 1B 余量；剩余未锁定项 = shared expert 的中间维是否恰等于 routed 的 8192（本账按等尺寸起算）、vision tower 与 projector 是否计入 blog 的 total 口径。字段级核对即可闭合，属练习 4。
4. **iRoPE 超参无 ablation**：3:1 比例、chunk 8192、 $\alpha = 0.1$ 各自的贡献没有公开对照；式 (8) 的温度机制是与 4.6 稀释账一致的重构，待实证。
5. **10M 的 effective context**：Scout 的 10485760 是 claimed 口径；按 4.6 §5 的方法论，needle 类测试之外需要 RULER 类多任务复核，目前无第三方结果。
6. **Scout 有 rope_scaling、Maverick 无**：同代同 base 的两个旗舰在长窗工具上取不同配置，官方未解释；与 10M / 1M 窗的定位差异联动。

## 8. 开放权重作为研究 baseline 的机制意义

谱系的参照系地位由三个机制性事实构成，与规模无关。其一，**对账可执行**：2.8 式 (6)(7) 那种精确到个位的参数闭环、式 (1) 那种 cache 实账，只在「权重 + config 双开」的模型上能跑到底——Llama-3.1-8B 是全书 P2/P4 的锚点模型正因此；405B 行与 Llama-4 行的证据降级（脚注与 [W] 标注）也从反面划出了这条边界。其二，**谱系内 diff 近似受控实验**：相邻两代骨架不动、每次只动一两个组件（GQA、vocab、ctx、MoE），跨代对照能把「哪个改动引起哪个账的变化」分离到单个变量——这是闭源谱系（只有 blog 数字）给不了的结构。其三，**retro-fit 类方法以开放 checkpoint 为前提**：GQA 的 mean-pool 初始化 + 5% 预算续训（4.1 §3，[B]）这类「在存量权重上改架构」的技术路线，其可复现性完全依赖权重公开；Llama 谱系各代 checkpoint 的开放使这类方法有了标准试验台。

## 总结

| 代际决策 | 公式 / 数值 | 换掉什么 | 代价 |
| --- | --- | --- | --- |
| Llama-1 标准配方 | 2.8 式 (1)–(4)：RMSNorm + Pre-LN + RoPE + SwiGLU + untied | GPT-2 的 LN / learned abs / GELU / tied | 外推出域（2.5 §6），后两代各补一次 |
| Llama-2 70B GQA-8 | 式 (1)：cache 2.5 MiB → 320 KiB/token； $B$ = 32 时 320 → 40 GiB | MHA 的 64 份 K/V 行空间 | 行空间塌到 8 份；GQA-8 为 Pareto 中间点（[B] 论文口径） |
| Llama-3.1 vocab 128K | 式 (2) $2Vd$（8B 占 13.1% / 405B 占 1.0%）；式 (3) token −19.5% | 32K SentencePiece | 长尾冷启动；小模型的参数占比 |
| Llama-3.1 ctx 128K | 式 (4)：base $5\times10^{5}$、 $\lambda_{\max}$ ≈ 2.6M；六阶段 800B tokens | 4K 窗 | 扩窗训练预算；405B cache 63 GiB/seq（GQA 为前提） |
| Llama-4 MoE top-1 | 式 (5)(6)：16/128 routed + 1 shared；17B 激活 / 109B、400B 总参 [W] | dense FFN | router 训练侧行为不可见；top-1 外推无公开对照 |
| Llama-4 iRoPE | 式 (7) 相对位移界 + 式 (8) $m(t)$ ≈ 1.72 @10M | 全层 RoPE | 超参无 ablation；10M 仅 claimed 口径 |
| Llama-4 early fusion | patch 14 + ps 0.5：448² 图 = 256 tokens | adapter 式 late fusion | 图像 token 付全层账；训练配方未公开 [W] |

## 练习

1.（计算）用式 (1) 重算 Llama-2-70B：若把 GQA-8 换成 GQA-2（ $H_{kv} = 2$）， $T = 4096$、 $B = 64$ 时 cache 总量对 138 GB 权重的比值是多少？再算 Llama-3.1-405B 在 $T = 131072$、 $B = 4$ 下 GQA-8 与反事实 MHA-128 的 cache 总量，验证「63 GiB × 4」与「约 3.9 TiB」两个数字。
2.（计算）词表账的规模扫描：对 $d \in \lbrace4096, 8192, 16384\rbrace$ 三档（对应 8B/70B/405B 宽），分别算 $V$ = 32000 与 128256 的 $2Vd$ 及其占该档总参数（8.03B / 70.6B / 405B）的比例；结合式 (3)，给出「词表升级在 per-byte FLOPs 上何时回本」的定性结论——假设层部分 per-token FLOPs 不变。
3.（推导）式 (8) 的对数温度与 4.6 §1.3 的 $\Delta \ge \ln T$：设 NoPE 层可见集为 $T$，目标 key 相对随机 key 的 score 优势为 $\Delta_0$。证明：乘性温度 $m$ 把「目标份额不被稀释」所需的 $\Delta_0$ 界降为 $\ln T / m$；并指出这一补偿对「多个候选 key 的 score 接近」的情形为何不成立（乘性缩放保持 score 差的相对比例）。
4.（代码 / 对账）按 §6 的口径写一段参数对账脚本（教学用，非优化实现）：输入 Scout 的 config 字段（48 层、 $d$ = 5120、40 Q / 8 KV、 $D$ = 128、expert 中间维 8192、16 routed + 1 shared、 $V$ = 202048），逐项加出 routed experts + shared experts + attention + embedding/输出头 + vision tower（34 层、 $d$ = 1408，按 $12d^{2}$ 速算），验证两个 blog 数字（109B total、17B active）闭合到约 1B 余量；再回答两个字段级问题：(a) shared expert 的中间维在 config / 权重清单里叫什么、是否恰等于 routed 的 8192；(b) 用 safetensors 索引核对 blog 的 109B 是否计入了 vision tower 与 projector。

## 参考

- [Touvron et al., 2023a](../references.md)：LLaMA: Open and Efficient Foundation Language Models，arXiv:2302.13971，[A]（key: llama-1）——标准配方与四档架构表、1.0T/1.4T tokens
- [Touvron et al., 2023b](../references.md)：Llama 2: Open Foundation and Fine-Tuned Chat Models，arXiv:2307.09288，[A]（key: llama-2）——70B GQA-8、4K ctx、2T tokens
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——405B 架构表、15T tokens、128K 词表与 3.17 → 3.94 chars/token 对照、六阶段扩窗
- [Meta, Llama-4 blog + Scout/Maverick config](../references.md)：[W]（key: llama-4）——Llama-4 全部架构字段与 109B/17B、400B/17B 规模数字（config 经 unsloth 镜像实测 2026-11-12）
- [Meta, llama-models repo](../references.md)：[A]（key: llama-models-repo）——Llama 1–4 官方模型定义与 tokenizer
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——GQA-8、vocab 128256、rope_theta $5\times10^{5}$、ctx 131072
- [Meta, Llama-3.1-70B config.json](../references.md)：[A]（key: llama-3.1-70b-config）——70B 档同口径
- [Ainslie et al., 2023](../references.md)：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints，arXiv:2305.13245，[B]（key: gqa）——GQA-8 Pareto 中间点与 mean-pool uptraining
- [Fedus et al., 2021](../references.md)：Switch Transformers，arXiv:2101.03961，[A]（key: switch）——top-1 路由的原始对照结论
- [Kazemnejad et al., 2023](../references.md)：The Impact of Positional Encoding on Length Generalization in Transformers，arXiv:2305.19466，[B]（key: nope）——NoPE 长度泛化实证，iRoPE 的机制依据
- [Su et al., 2021](../references.md)：RoFormer，arXiv:2104.09864，[A]（key: rope）——RoPE 相对性（式 (7) 的出处）
- [Zhang & Sennrich, 2019](../references.md)：Root Mean Square Layer Normalization，arXiv:1910.07467，[B]（key: rmsnorm）——RMSNorm 出处
- [Shazeer, 2020](../references.md)：GLU Variants Improve Transformer，arXiv:2002.05202，[B]（key: glu-variants）——SwiGLU 与等参数规则
- [Press & Wolf, 2016](../references.md)：Using the Output Embedding to Improve Language Models，arXiv:1608.05859，[B]（key: weight-tying）——untied 的 $Vd$ 账
