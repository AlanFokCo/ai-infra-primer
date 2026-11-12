# 8.1 VLM 架构分类学（VLM Architecture Taxonomy）

> 本章给「图像如何进 LLM」这一件事立分类框架：VLM 的形式化 $p(\text{text} \mid \text{image}, \text{prompt})$；组件链 vision encoder → connector → LLM；三个接入点（input-level 拼接 / cross-attention 注入 / 混合）的公式与代表系统；token 预算账（单图 576 → AnyRes 2880 → 视频 15K 级——context 压力的核心矛盾，8.2 压缩技术的动机）；冻结策略与表征分析（LLM embedding 空间缺什么视觉特征）；参数规模组合表 | 前置：2.2（embedding 查表与 vocab 账）、2.3（attention 公式）、2.4（causal mask 与 KV cache）、6.1（ViT：patchify、$N = HW/p^{2}$、2D 位置编码）、6.2（CLIP/SigLIP 对比塔） | 锚点：LLaVA [A]（haotian-liu/LLaVA）、Flamingo [B]（论文 + mlfoundations/open_flamingo 复现 [A]）、BLIP-2 [A]、Qwen2.5-VL [A]（QwenLM/Qwen3-VL + config 实测）、InternVL3 [A]

## 1. 形式化：VLM 建模的对象

VLM（vision-language model，视觉语言模型）在本书的理解方向上是一个条件语言模型：给定图像与文本 prompt，自回归地建模回答文本 $y = (y_1, \ldots, y_T)$：

$$
p(y \mid x_{\text{img}}, \text{prompt}) = \prod_{t=1}^{T} p\left( y_t \mid y_{<t},\; x_{\text{img}},\; \text{prompt} \right) \qquad (1)
$$

**符号**：$x_{\text{img}} \in \mathbb{R}^{H \times W \times 3}$：输入图像；$y_{<t}$：已生成的前缀；条件里的 prompt 指图像之外的文本指令部分。多图与视频是同一形式的直接推广：把 $x_{\text{img}}$ 换成图像集合或帧序列即可，条件侧的结构差异全部由 §3 的接入机制吸收。式 (1) 与 1.1 的纯文本 LM 只差一个条件对象——全部架构问题因此收敛成三个具体选择：图像以**什么形态**（连续特征还是离散 token）、**在哪个位置**（输入序列、attention 内部、还是 connector 里先压一遍）、**以多少 token** 进入这个条件。本章回答前两个（§3），第三个是 §4 与 8.2 的主线。生成方向（text → image）把条件与输出对调，涉及另一组建模选择，归 8.5；本章只管理解方向。训练侧一个贯穿全 Part 的事实先记一笔：式 (1) 的 loss 只在回答 token 上计，prompt 与图像部分的损失被 mask 掉——公式与代码走读归 8.3。

形态轴在理解方向几乎没有争议：主流全部走连续特征。离散化路线（图像经 VQ tokenizer 变成离散 id、查表进 embedding，6.3 的 VQ-VAE 族）在生成方向是自然选择（8.5），对理解方向却白付一笔量化损失——VQ 的码本瓶颈为重建保真而设，理解任务要的是语义可分性，不是像素可重建性；连续特征直通高层语义、不经过码本。本章的三个接入点因此都建立在连续 $z_{\text{img}}$ 之上。

## 2. 组件链：vision encoder → connector → LLM

当代 VLM 几乎全部是同一条三段链的实例：

```
x_img [H, W, 3]
  │  vision encoder（视觉塔，ViT 族——6.1；冻结或微调，§5）
  ▼
z_img [N, d_v]                    N = HW/p²（6.1 式 (3)）
  │  connector φ —— 换宽度 d_v → d_llm，可改 token 数 N → M
  ▼
v [M, d_llm]                      M：每图视觉 token 数
  │  接入 LLM（三选一：input / cross-attn / 混合，§3）
  ▼
LLM（语言塔，decoder-only——P2 全套结构原样复用）
```

形式化：视觉塔输出 $z_{\text{img}} \in \mathbb{R}^{N \times d_v}$（[CLS] 是否保留逐家不同；进 LLM 的一般只取 patch tokens），连接器（connector，亦称 projector）完成宽度与数量的转换：

$$
v = \phi\left( z_{\text{img}} \right) \in \mathbb{R}^{M \times d_{\text{llm}}} \qquad (2)
$$

**符号**：$d_v$：视觉塔宽度（CLIP ViT-L 为 1024）；$d_{\text{llm}}$：LLM 的 embedding 宽度；$M$：每图视觉 token（visual token）数——透传型 connector 下 $M = N$，压缩型下 $M < N$。三段各自的规模与冻结状态是 §5/§6 的主题；$\phi$ 的家族（线性 / MLP / Q-Former / resampler / pixel shuffle）是 8.2 的主战场，本章只用到它「把 $d_v$ 换成 $d_{\text{llm}}$」这一个接口性质。

为什么是三段的组合，而不是一座从零训的端到端单塔？答案是一笔摊销账：视觉塔的对比预训练（4 亿级图文对，6.2）与 LLM 的文本预训练（万亿级 token，1.3）各自已经付过天价算力，三段链让 VLM 直接继承两份资产，新增训练只发生在接缝处；端到端单塔则要把两份能力在同一份参数里重新学一遍，数据与算力账都不支持。（8.5 的统一生成模型会在更高的数据预算下重新挑战这个结论。）接缝处的设计自由度——接入位置与 token 数——正是本章与 8.2 的全部内容。

## 3. 三个接入点

### 3.1 (a) input-level：visual tokens 拼进输入序列（LLaVA 范式）

把 $v$ 当作 $M$ 个「外来 token」的 embedding 行，直接拼进 LLM 的输入序列——prompt 里用一个 `<image>` 占位符标位，collate 时把 $v$ 展开到该位置：

$$
e = \left[\, E(s_1), \ldots, E(s_a),\; v_1, \ldots, v_M,\; E(s_{a+1}), \ldots \,\right] \in \mathbb{R}^{(T + M) \times d_{\text{llm}}} \qquad (3)
$$

**符号**：$E$：LLM 文本 embedding 查表（2.2）；$s_i$：文本 token；$T$：文本长度。此后一切照旧：同一份 causal mask、同一份位置编码、同一份 next-token loss——对 LLM 而言，图像就是一段「不在词表里、但形状合法的 embedding 行序列」。

```
序列布局（input-level）：
[sys tokens] [user prompt 前半] [v_1 … v_M] [user prompt 后半] → 因果 LLM
                                  ↑ `<image>` 占位符展开处
```

这个范式出自 LLaVA（[Liu et al., 2023](../references.md)，arXiv:2304.08485，[A]）：视觉塔取 CLIP ViT-L/14，丢掉 [CLS]、取全部 patch token 的网格特征，connector 是单个线性层 $W \in \mathbb{R}^{d_{\text{llm}} \times d_v}$（后续 LLaVA-1.5 升级为两层 MLP 并把输入提到 336 档，[Liu et al., 2023](../references.md)，arXiv:2310.03744，[A]；linear vs MLP 的对照行归 8.2）。实现走读锚点：haotian-liu/LLaVA 的 llava/model 目录（[llava-repo](../references.md)，[A]）。

input-level 的代价与收益都写在式 (3) 里。收益是**零新结构**：多模态融合复用 LLM 全部 $L$ 层 self-attention 的容量，图文交互深度拉满，工程上对任何现成 LLM 都是「不改结构、只加数据」。代价有二：$M$ 个 token 全额占用 context 与 KV cache（§4 的账）；且 LLM 必须参与训练——它的 embedding 空间从没见过这种行（§5）。

多图是同一机制的重复：prompt 里放几个 `<image>` 占位符，collate 时按出现顺序逐段展开对应的 $v$；段间位置编码的排布（各图坐标独立编号还是统一递增）是 8.2 的 M-RoPE 话题。图像段在序列中的位置完全由数据格式决定——这与 (b) 路线「视觉在序列之外、靠层内注入」形成结构性对照。

### 3.2 (b) cross-attention-level：Flamingo 的 gated xattn

第二条路不把图像塞进输入序列，而是在冻结 LLM 的层间插入新层，让文本隐状态透过 cross-attention 去读视觉特征。Flamingo（[Alayrac et al., 2022](../references.md)，arXiv:2204.14198，[B]——DeepMind 未开源；代码对照走开放复现 mlfoundations/open_flamingo，open_flamingo/src/flamingo.py 的 gated_cross_attn_layers，[open-flamingo-repo](../references.md)，[A]）的插入块是门控 cross-attention（gated xattn）加门控 FFN 的组合：

$$
h \leftarrow h + \tanh(\alpha)\, \mathrm{XAttn}\left( h,\; z_{\text{img}} \right), \qquad \mathrm{XAttn}(h, z) = \mathrm{softmax}\left( \frac{Q(h)\, K(z)^{\top}}{\sqrt{d_k}} \right) V(z) \qquad (4)
$$

$$
h \leftarrow h + \tanh(\alpha')\, \mathrm{FFN}(h) \qquad (5)
$$

**符号**：$h \in \mathbb{R}^{B \times T \times d}$：冻结 LLM 某层的文本隐状态；query 来自文本、key/value 来自视觉特征，score 矩阵 shape $[B, H, T, N]$；$\alpha, \alpha'$：逐层可学标量门，**零初始化**。插入层与冻结层交替排布（interleaved），LM 原有参数一行不动；输入序列长度仍是 $T$，视觉不占输入位置。

零初始化的机制含义要写在数学上：$\tanh(0) = 0$，所以初始化时刻整个模型**逐点等于原冻结 LLM**——第一个 step 的 loss、行为与纯文本模型无差别，新结构是「从零被学开」的，而不是「一上手就扰动预训练权重分布」。门自身也不会卡死在零：

$$
\frac{\partial}{\partial \alpha} \tanh(\alpha) = 1 - \tanh^{2}(\alpha) \qquad (6)
$$

**符号**：$\alpha$：式 (4) 的逐层可学门控标量；右端为 $\tanh$ 的一阶导数。式 (6) 即「门在零点斜率为 1」：$\alpha = 0$ 时右端取值为 $1$，只要 XAttn 分支输出非零、上游梯度非零，$\alpha$ 第一步就收到满额梯度（完整推导留作练习 2）。这是 adaLN-Zero 家族（7.6）的同一招用在条件注入上：把「新模块起步 = identity」变成初始化时刻可证明的性质。Flamingo 论文的消融显示，去掉 tanh 门控的零初始化、让 xattn 输出按普通残差直接注入，会明显掉点（[B]，论文口径）——初始恒等性不是装饰，是这条路线保住冻结 LM 能力的关键设计。

(b) 的推理账与 (a) 结构性不同：视觉不占输入序列，LLM 自注意力的 KV cache 长度仍是 $T$ 而非 $T + M$；视觉特征的 K/V 在每个插入层各算一次、decode 全程复用。代价写在同一处：融合只发生在插入层——冻结层里的文本 token 看不到图像，图文交互深度由插入层的数量与位置决定。这就是 §3.4 对照表「交互深度」一栏的出处。

视觉侧还有一个配套件：Perceiver resampler——把变长的 $z_{\text{img}}$（多图/视频下长度不定）压成固定 64 个 latent（论文口径），使式 (4) 的 key/value 长度恒定。它与 Q-Former 同族，公式细节归 8.2。

### 3.3 (c) 混合：Q-Former 压缩后进 input-level

BLIP-2（[Li et al., 2023](../references.md)，arXiv:2301.12597，[A]）把前两件事组合起来：connector 不再逐 token 透传，而是用 $B$ 个可学 query 对冻结 ViT 的输出做 cross-attention，把 $N$ 个 patch 特征压成 $B$ 个向量，再投影到 $d_{\text{llm}}$、按式 (3) 拼进输入：

$$
v = W_{\text{proj}}\, \mathrm{QFormer}\left( q_{1:B},\; z_{\text{img}} \right) \in \mathbb{R}^{B \times d_{\text{llm}}}, \qquad M = B = 32 \qquad (7)
$$

**符号**：$q_{1:B}$：可学 query，BLIP-2 取 $B = 32$；$W_{\text{proj}}$：到 LLM 宽度的线性投影。「混合」的准确含义：Q-Former 内部的 cross-attention（query 读冻结 ViT 输出）就是 (b) 的读取机制，只不过它被收编进 connector、作用于 LLM 体外；最终接入仍按 (a) 拼进输入序列。(c) = (b) 的读取 × (a) 的落点。与纯 (a) 相比，$M$ 与图像分辨率解耦——256 个 patch 也好、2880 个 tile token 也好，进 LLM 的恒为 32 个。代价是「压什么、丢什么」的决定被集中压给 Q-Former 的训练目标（ITC/ITM/ITG 三目标与两阶段训练，8.2 详拆）；且它不是免费的——LLaVA-1.5 的讨论指出，Q-Former 类 resampler 方案的训练语料在数亿级，而其 1.2M 数据规模下两层 MLP 已够（论文讨论口径，非表内消融行；机制账与 linear/MLP 对照行归 8.2）。

### 3.4 三接入点对照

| 维度 | (a) input-level | (b) cross-attn-level | (c) 混合 |
| --- | --- | --- | --- |
| 改动位置 | 输入序列 | LLM 层间插新层 | connector 内部 + 输入序列 |
| LLM 预训练参数 | 参与训练 | 完全冻结 | 冻结（BLIP-2 口径） |
| 图文交互深度 | 全部 $L$ 层 self-attention | 仅插入层 | 全部 $L$ 层（压缩后） |
| 每图 token 数 $M$ | $N$（全额） | 0（不占输入序列） | $B$（常数） |
| 初始化行为 | 新 embedding 行扰动输入分布 | 恒等于原 LLM | 新 embedding 行扰动输入分布 |
| 代表系统 | LLaVA 系、Qwen-VL 系、InternVL 系 | Flamingo | BLIP-2 |

当代开放旗舰——Qwen2.5-VL（[Bai et al., 2025](../references.md)，arXiv:2502.13923，[A]）与 InternVL3（[Zhu et al., 2025](../references.md)，arXiv:2504.10479，[A]）——清一色选 (a)。生态因此围绕「input-level 下如何省 token、如何保分辨率」演化出整套压缩与分辨率技术，那就是 8.2 的内容。

## 4. token 预算账

复用 6.1 式 (3)：patch 边长 $p$ 时 $N = HW/p^{2}$，input-level 接入下 $M = N$。三笔标准账（全书约定页口径，实例可用公式现推复算）：

- **单图基线**：CLIP 系 ViT-L/14 的 224 档 $N = 16^{2} = 256$（初代 LLaVA），336 档 $N = 24^{2} = 576$（LLaVA-1.5 起的主流档）；
- **AnyRes 切图**：切成 $k$ 块各过一次塔、外加 1 张全局缩略图，$M = (k+1) \cdot N$——4 切实例 $5 \times 576 = 2880$（LLaVA-NeXT / LLaVA-OneVision 口径，[Li et al., 2024](../references.md)，arXiv:2408.03326，[A]；网格选择与位置编码处理归 8.2）；
- **视频**：$F$ 帧 × 每帧 token——1 分钟 @1fps、每帧 256 即 $60 \times 256 = 15360$（8.4 展开帧采样与时间压缩）。

把账挂到 LLM 侧读。context 压力 = 全部图/帧的 $M$ 之和 + 文本 token：一张 AnyRes 4 切图吃掉 2880，十张图就是 28800——对照 Qwen2.5-VL-7B 的 128000 窗口（[qwen25vl-config](../references.md)，[A]，config 实测），视觉部分轻易可以吃掉大半窗口。KV cache 同样按 token 计：该模型 LLM 侧 28 层、GQA 4 个 KV head、head_dim 128（config 实测），bf16 下每 token 每层 K+V 共 $2 \times 4 \times 128 \times 2 = 2048$ 字节，全层合计 $28 \times 2048 = 57344$ 字节 $\approx 56$ KB/token——一张 2880 token 的图仅 KV cache 增量就约 165 MB，prefill 的 attention 配对数还按 $T^{2}$ 计。token 预算（token budget）就是这个意义上的硬通货：**每个视觉 token 同时消耗窗口长度、显存与 prefill 算力**。

视频把这条账乘上时间轴——图像在这个视角下只是 $F = 1$ 的特例。帧率、分辨率、时长三个旋钮全部线性打进 token 数；8.4 的帧采样（uniform vs dynamic FPS）与时间压缩（相邻帧合并、temporal pooling）都是在这条乘法账上做除法。多图长文档同理：一本 30 页扫描 PDF 按 AnyRes 4 切就是 $30 \times 2880 = 86400$ 视觉 token，还没算文本，128K 窗口已经见底。

这条账定义了 8.2 的全部问题意识。connector 家族里凡是 $M < N$ 的设计——pixel shuffle（$M = N r^{2}$，Qwen 系的 spatial_merge=2 即 $r = 0.5$ 档，config 实测）、Q-Former（$M = B$）、Perceiver resampler（固定 64）——本质上都是 token 压缩器；分辨率方案（AnyRes / naive dynamic resolution / Pixtral 原生分辨率）则是在「保像素信息」与「付 token 税」之间选汇率。

## 5. 冻结策略与表征分析

三段链给出三个独立的冻结开关，各家的选择矩阵（论文 / tech report 口径，2026-11-12 锁定）：

| 系统 | vision encoder | connector | LLM |
| --- | --- | --- | --- |
| Flamingo [B] | 冻结（NFNet-F6，对比预训练） | 训练（resampler + xattn） | 冻结（Chinchilla） |
| BLIP-2 [A] | 冻结（ViT-g/14） | 训练（Q-Former，两阶段） | 冻结（FlanT5 / OPT） |
| LLaVA [A] | 冻结（CLIP ViT-L/14） | 训练（线性层 / MLP） | 微调 |
| Qwen2.5-VL [A] | 分阶段：预训练 phase 2 起解冻、post-training 再冻结 | 训练 | 训练 |

冻结的机制理由分两笔。**冻 LLM**（Flamingo / BLIP-2 路线）：保护预训练语言能力——few-shot、指令遵循这些资产是在纯文本语料上长出来的，多模态数据混进来微调就有冲淡风险；配合 zero-init 门（式 (4)-(6)），训练起点干脆就是原模型本身。代价是新容量全部集中在新插入层或 connector 里，融合深度受限（§3.4 表）。**冻 ViT**（除 Qwen 系外几乎全员）：CLIP/SigLIP 塔的表征是 4 亿级图文对对比预训练的产物（6.2，[Radford et al., 2021](../references.md)，arXiv:2103.00020，[A]），VLM 手里的图文对规模小几个量级，微调容易把现成的好表征拉坏；外加一笔工程账——塔冻结意味着 $z_{\text{img}}$ 可离线预计算、入缓存复用。Qwen2.5-VL 代表相反端点：vision tower 自研（window attention + spatial merge，结构归 8.2），预训练 phase 2 起全解冻参与联合训练，post-training（SFT/DPO）阶段再度冻结（tech report 口径）——数据量到位时联合训练上限更高，但冻结仍是训练末段的默认收束。

更深一层的问题是表征内容的错配：**LLM 的 embedding 空间缺什么视觉特征？** 文本 embedding 是 next-token 目标在纯文本上训出来的（2.2），本身没有任何视觉接地；接进来的 $v$ 必须独自充当「像素世界的摘要」。而 CLIP/SigLIP 特征处于**语义抽象层级**：对比目标（6.2）只监督「整图 ↔ 整句」的全局对齐，不监督逐 patch 的细节保持——颜色精确值、小字、物体计数、细粒度空间关系这些像素级信息在塔内逐层衰减，到 $z_{\text{img}}$ 已所剩无几。两个直接后果：(i) 进 LLM 用 dense patch token 而不用 [CLS] pooled 向量（LLaVA 的选择），至少保住空间网格结构；(ii) 小字 OCR / 计数类任务成为 VLM 的传统短板，直接推高分辨率路线（§4 的账）与视觉塔侧的 dense 监督补强（SigLIP 2 叠加逐 patch 目标正是补这一缺口，6.1 §8）。

一条正向证据来自多阶段配方：stage-1 通常冻结双塔、只训 connector（LLaVA 与 BLIP-2 的共同选择，配方细节归 8.3）——几十万级图文对就能把 $v$ 对齐进 LLM 的 embedding 空间并得到可用的图文问答。这件事本身说明 connector 学的是**空间对齐**（把语义已就位的特征变换进 LLM 可读的坐标系），而不是**内容补全**：视觉内容在塔里已经备好，缺的只是坐标变换。同一个失败模式从反面支持这一读法——connector 对齐不出视觉塔里不存在的信息，小字与计数的短板要补在分辨率与塔侧（§4、6.1 §8），堆 connector 容量没有效果。8.2 的 connector 设计空间里，「压缩 $M$」与「保像素细节」的对冲就建立在这一节的错配之上。

## 6. 参数规模组合表

真实组合（各家论文原表 / config 实测与现推口径，2026-11-12 锁定；vision 与 LLM 两列不跨列加总）：

| 系统 | vision encoder | LLM | 可训参数量级 | 接入点 |
| --- | --- | --- | --- | --- |
| LLaVA-13B | CLIP ViT-L/14（0.3B，6.1 §4 口径） | Vicuna-13B | 线性 $W$：$1024 \times 5120 \approx 5$ M | (a) |
| Flamingo-80B [B] | NFNet-F6（0.4B 级，冻结） | Chinchilla-70B（冻结） | resampler + xattn $\approx$ 10B | (b) |
| BLIP-2 | ViT-g/14（1B，冻结） | FlanT5-XXL 11B（冻结） | Q-Former 188M | (c) |
| Qwen2.5-VL-7B | 自研 ViT（0.6B 级：32 层、hidden 1280，config 现推） | 7B 级（28 层、宽 3584） | 全量（分阶段，见 §5） | (a) |

三个读法。其一，**vision 塔恒为 LLM 的零头**（0.3–1B 对 7–70B）：这是分工而非巧合——视觉侧所需容量远低于语言侧，把参数预算花在 LLM 上是各家共同结论；§5 的「冻 ViT」之所以便宜，部分原因正是它本来就小。其二，**接入点决定可训参数的位置与量级**：(a) 的 connector 是百万级（线性 $W$ 的参数量就是 $d_v \times d_{\text{llm}}$ 一笔），可训主体是 LLM 自身；(b) 把可训参数堆进插入层、堆到 10B 也不动原模型一行；(c) 折中在亿级。开放生态里 (a) 成主流、(b) 式微，工程解释直接：(a) 对任何现成 LLM 都是「不改结构、只加数据」，整个 P2/P4 的推理与训练栈原样复用；(b) 要求框架支持层间插层，且面对开源 LLM 权重时「冻结保能力」的收益需要用自家算力重新兑现。其三，这张表同时是边际收益表的底稿：vision 侧从 0.3B 升到 1B（BLIP-2 档）吃掉的参数预算，只相当于 LLM 从 7B 迈到 8B 的一步，而视觉表征的升级（分辨率、dense 监督）直接作用于 §5 点名的短板——LLM 已到 70B 级的组合里继续堆语言参数，边际收益通常不如把 vision 侧的分辨率与表征补齐。8.2 的 connector 与分辨率军备竞赛正是沿这个不等式展开。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 形式化 | 式 (1) | 条件 LM：与纯文本只差 $x_{\text{img}}$；架构 = 形态 × 位置 × token 数三选择 |
| 组件链 | 式 (2) | $z_{\text{img}}\, [N, d_v] \to v\, [M, d_{\text{llm}}]$；connector 只承诺换宽度 |
| input-level | 式 (3) | LLaVA 范式；融合最深、$M = N$ 全额占 context、LLM 须参训 |
| gated xattn | 式 (4)(5)(6) | zero-init ⇒ 起步恒等原 LLM；门在零点斜率 1 ⇒ 可学开；Flamingo [B] |
| 混合 | 式 (7) | Q-Former $M = B = 32$，token 数与分辨率解耦 |
| token 预算 | §4 | 224/p14 → 256；336/p14 → 576；AnyRes 4 切 → 2880；视频 1min@1fps → 15360；KV $\approx$ 56 KB/token（7B 档推导） |
| 冻结策略 | §5 表 | 冻 LLM 保语言能力；冻 ViT 省算 + 保表征；CLIP 特征是语义层级 ⇒ 像素细节短板 |
| 参数组合 | §6 表 | vision 0.3–1B × LLM 7–70B；connector 量级随接入点走（百万级 / 十亿级 / 亿级） |

## 练习

1.（计算）一张 $672 \times 1008$ 的图：(a) 等比缩放后裁进 $336 \times 336$、$p = 14$；(b) AnyRes 切成 $2 \times 3 = 6$ 块、每块走 336 档、另加 1 张全局缩略图；(c) naive dynamic、$p = 14$ 不切图。三种处理各给 LLM 多少视觉 token？(b) 相对 (a) 的 token 倍率与像素面积保真各是什么？
2.（推导）式 (4) 中把 $\tanh(\alpha)$ 换成裸标量门 $\alpha$（仍零初始化）：(i) 写出 $\partial L / \partial \alpha$ 在 $\alpha = 0$ 处的表达式（用上游梯度与 XAttn 输出表示），说明两种门在第一步梯度上的差别；(ii) 训练走远后 $\tanh$ 带来的额外性质是什么（提示：$|\tanh| < 1$）？结合 7.6 的 adaLN-Zero，说明「零初始化 + 有界门」为何是条件注入的反复出现的设计。
3.（计算）按 §4 口径（Qwen2.5-VL-7B LLM 侧：28 层、4 个 KV head、head_dim 128、bf16）：(i) 验证 56 KB/token；(ii) 一段含 4 张 AnyRes 4 切图 + 2000 文本 token 的对话，KV cache 总量多少？(iii) 若把视觉侧换成 $M = 32$ 的 Q-Former 式压缩，KV cache 省多少、代价是什么（用 §5 的错配论证回答）？
4.（代码）写 input-level 的 collate 函数（教学用，非优化实现）：输入 batch 的 token id 序列（含 `<image>` 占位符 id）与每图 $v \in \mathbb{R}^{M \times d_{\text{llm}}}$，输出按式 (3) 拼接后的 embedding 序列与对应的 loss mask（prompt 与占位符展开段不计 loss——8.3 伏笔）。

## 参考

- [Liu et al., 2023](../references.md)：Visual Instruction Tuning，arXiv:2304.08485，[A]（key: llava）——input-level 范式、线性 projector、CLIP ViT-L/14 取 patch 特征
- [Liu et al., 2023](../references.md)：Improved Baselines with Visual Instruction Tuning，arXiv:2310.03744，[A]（key: llava-1.5）——336 档 576 token 与 MLP connector（ablation 详拆归 8.2）
- [Alayrac et al., 2022](../references.md)：Flamingo: a Visual Language Model for Few-Shot Learning，arXiv:2204.14198，[B]（key: flamingo）——gated xattn 式 (4)(5)、zero-init 消融、Perceiver resampler 64 latents、冻结策略
- [Li et al., 2023](../references.md)：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models，arXiv:2301.12597，[A]（key: blip2）——Q-Former 混合接入式 (7)、$B = 32$、188M 参数、冻结矩阵
- [Bai et al., 2025](../references.md)：Qwen2.5-VL Technical Report，arXiv:2502.13923，[A]（key: qwen2.5-vl）——参数组合与全量训练口径
- [Zhu et al., 2025](../references.md)：InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models，arXiv:2504.10479，[A]（key: internvl3）——当代开放旗舰对照
- [Li et al., 2024](../references.md)：LLaVA-OneVision: Easy Visual Task Transfer，arXiv:2408.03326，[A]（key: llava-onevision）——AnyRes 4 切 2880 账
- [Radford et al., 2021](../references.md)：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020，[A]（key: clip）——视觉塔与 336/p14 网格、4 亿级图文对
- [Dosovitskiy et al., 2020](../references.md)：An Image is Worth 16x16 Words，arXiv:2010.11929，[A]（key: vit）——$N = HW/p^{2}$ 出处（6.1 式 (3)）
- [haotian-liu/LLaVA](../references.md)：[A]（key: llava-repo）——input-level 拼接走读锚点（llava/model 目录）
- [mlfoundations/open_flamingo](../references.md)：[A]（key: open-flamingo-repo）——gated xattn 开放复现走读锚点（open_flamingo/src/flamingo.py）
- [QwenLM/Qwen3-VL](../references.md)：[A]（key: qwen-vl-repo）——Qwen-VL 系列 repo（Qwen2.5-VL 已 301 归并至此）
- [Qwen/Qwen2.5-VL-7B-Instruct config.json](../references.md)：[A]（key: qwen25vl-config）——LLM 28L/3584/4KV/ctx 128000、vision 32L/hidden 1280/spatial_merge 2 实测
