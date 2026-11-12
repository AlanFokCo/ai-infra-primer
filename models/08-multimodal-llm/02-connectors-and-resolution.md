# 8.2 Connector 与分辨率处理（Connectors and Resolution）

> connector 的全部问题压成三个自由度——维度对齐 $d_v \to d_{llm}$、token 数变换 $N \to M$、表征翻译；本章按 token 数的组织方式展开四族 connector（等长 MLP / 定长瓶颈 Q-Former 与 Perceiver resampler / 规则压缩 pixel shuffle / 软查表），随后把分辨率问题算成账：固定 336 的信息损失（小字识别）、AnyRes tiling 的网格选择公式、Qwen2-VL naive dynamic resolution 的动态 token 数、**M-RoPE $(t,h,w)$ 三元组完整推导**（本章核心：文本三值同值、图像 $t$ 恒定 + $(h,w)$ 网格、视频 $t$ 递增，rotary 频段按 [16,24,24] 切三段）、Pixtral 免 tiling，收口于 connector 选型决策表 | 前置：2.5（RoPE 主命题式 (13)、频率 schedule 式 (16)、两实现约定）、6.1（patchify $N = HW/p^{2}$、RoPE-2D 式 (7)(8)、NaViT packing）、8.1（组件链记号 $z_{img}/\phi/v/M$、token 预算账） | 锚点：transformers Qwen2.5-VL modeling（`get_rope_index`、`mrope_section [16,24,24]`）[A]、OpenGVLab/InternVL（`pixel_shuffle` 实测）[A]、salesforce/LAVIS（`blip2_qformer.py`）[A]；BLIP-2 [A]、LLaVA-1.5 [A]、Qwen2-VL [A]、Qwen2.5-VL [A]、LLaVA-OneVision [A]、Pixtral [A]、InternVL [A]、Ovis [A]、MiniCPM-V [A]、Flamingo [B]

## 1. 问题与契约：connector 的三个自由度

8.1 的 token 账在结尾处埋着本章的全部问题：一张 336×336 的图经 CLIP ViT-L/14 出来是 576 个 token，AnyRes 切图后涨到 2880，一分钟视频按 256 token/帧算是 15K 级——LLM 的 context 是硬预算（Qwen2.5-VL-7B 给到 128000 已是旗舰口径，[Qwen2.5-VL-7B config](../references.md)，[A]），视觉 token 的「数量、形态、坐标」三件事必须在进 LLM 之前解决。解决场所就是 connector $\phi$。契约（8.1 记号，本章严格执行）：

$$
v = \phi(z_{img}), \qquad z_{img} \in \mathbb{R}^{N \times d_v}, \qquad v \in \mathbb{R}^{M \times d_{llm}} \qquad (1)
$$

**符号**：$z_{img}$：vision encoder 输出的 patch 特征序列（进 LLM 的一般只取 patch tokens，[CLS] 逐家标注）；$N = HW/p^{2}$：视觉 token 数（P6 约定沿用）；$d_v$：视觉塔宽度；$v$：拼进 LLM 输入序列的视觉段；$M$：每图视觉 token 数；$d_{llm}$：LLM stream 宽度。

式 (1) 拆开的三个自由度：(i) **维度对齐** $d_v \to d_{llm}$——任何 connector 都要做，哪怕只是一个线性层；(ii) **token 数变换** $N \to M$——$M$ 可以等于 $N$（等长映射）、被规则压缩（$M = N \cdot r^{2}$）、被压成定长（$M = B$ 与 $N$ 无关）；(iii) **表征翻译**——视觉特征分布与 LLM embedding 空间的对齐，这一步没有显式公式，由训练目标完成（8.3 展开）。本章谱系按自由度 (ii) 的组织方式分四族，这也是读全部 VLM connector 的统一坐标：

| 族 | $M$ 的形态 | 代表系统 | 详节 |
| --- | --- | --- | --- |
| 等长映射 | $M = N$ | LLaVA 线性、LLaVA-1.5 MLP、Ovis 软查表 | §2 |
| 定长瓶颈 | $M = B$（与 $N$ 无关） | BLIP-2 Q-Former（$B$ = 32）、Flamingo resampler（$B$ = 64） | §3、§4 |
| 规则压缩 | $M = N \cdot r^{2}$ | InternVL / Qwen 系 pixel shuffle（$r$ = 0.5 ⇒ $N/4$） | §5 |
| 相似度合并 | $M$ 内容自适应 | token merging 族（机制 outline） | §5 末 |

四族之外还有第零个问题：固定分辨率的视觉塔本身丢了多少信息。§6 先把这笔账算清，§7–§10 的 tiling、动态分辨率、M-RoPE、native resolution 都是对它的回应。

## 2. projector 家族：等长映射是最便宜的默认值

LLaVA 的 connector 是一个线性层，LLaVA-1.5 把它换成 2 层 MLP（[Liu et al., 2023](../references.md)，arXiv:2310.03744，[A]）。两者都是逐 token 独立作用、不混 token 的等长映射：

$$
v_i = W_2\, \mathrm{GELU}\!\left( W_1 z_i + b_1 \right) + b_2, \qquad W_1 \in \mathbb{R}^{d_{llm} \times d_v},\quad W_2 \in \mathbb{R}^{d_{llm} \times d_{llm}},\quad i = 1, \ldots, N \qquad (2)
$$

**符号**：$z_i$：$z_{img}$ 第 $i$ 行（单个 patch 的 $d_v$ 维特征）；$v_i \in \mathbb{R}^{d_{llm}}$：进 LLM 的第 $i$ 个视觉 token；GELU 为逐元素非线性（2.6）。线性 projector 是式 (2) 砍掉第一层：$v_i = W z_i + b$。

参数账直接读：式 (2) 共 $d_v d_{llm} + d_{llm}^{2}$ 个参数——注意现代组合里 $d_{llm} > d_v$，$d_{llm}^{2}$ 项占大头，简写 $2\, d_v d_{llm}$ 只在 $d_{llm} \approx d_v$ 时成立。代一档真实宽度——$d_v$ = 1024（ViT-L/14）、$d_{llm}$ = 4096——$1024 \times 4096 + 4096^{2} \approx 21$ M，相对 7B 级 LLM 本体仍是千分位级。等长映射 $M = N$：CLIP-336 网格的 576 个 token 原数进 LLM。

LLaVA-1.5 论文给了两点此后成为默认口径的陈述（[Liu et al., 2023](../references.md)，[A]）：(a) 线性 projector 换成 2 层 MLP 是 improved baseline 的有效组成（正文陈述，非消融表）；(b) Q-Former 类 resampler 方案的训练语料在数亿级，在其约 1.2M 的数据规模（caption 预训练 558K 条 + 指令微调 665K 条）下两层 MLP 已够（limitations 讨论口径，非表内消融行）。结论 (b) 的机制论证放到 §3.4：Q-Former 的 32-token 瓶颈要求先把「从 576 个 patch 里选哪些信息」这个选择函数学进约 1.9 亿参数，而 BLIP-2 把这个函数训出来用了 129M 图文对——与 1.2M 差两个数量级。容量与数据不匹配时，瓶颈不是压缩而是纯损失。**小数据下 MLP 够、Q-Former 不划算**，其反面同样成立：数据到 1 亿级、且要求冻结双塔不动预训练资产时，Q-Former 的账翻正（§3.4）。

等长族的一个变体是软查表。Ovis 把 connector 做成一张 visual embedding table：连续 patch 特征先经 visual tokenizer 映成「视觉词表」上的概率分布，再按分布对可学 embedding 表取期望——与文本侧的查表对齐在结构层完成，故称 structural embedding alignment（[Lu et al., 2024](../references.md)，arXiv:2405.20797，[A]；官方 repo 现为 ATH-MaaS/Ovis，[Ovis repo](../references.md)，[A]）。token 数仍是 $M = N$，变的是对齐机制：从「线性层学映射」换成「分布匹配一张可学表」。

## 3. Q-Former（BLIP-2）：$B$ 个可学 query 的瓶颈与三目标

Q-Former 是定长瓶颈族的标准件（[Li et al., 2023](../references.md)，arXiv:2301.12597，[A]）：无论输入网格多少 token，输出恒为 $B$ 个（论文主档 $B$ = 32，各 768 维）。32 ≪ 576 意味着它不是映射而是**取舍**——取舍函数从哪来、靠什么目标训出来，是本节的推导主线。代码走读锚点：salesforce/LAVIS 的 `lavis/models/blip2_models/blip2_qformer.py`（[LAVIS repo](../references.md)，[A]）。

### 3.1 结构：共享 self-attention 的双子模块

Q-Former 维护 $B$ 个可学 query 向量 $Q \in \mathbb{R}^{B \times d_q}$（$d_q$ = 768），与一个文本子模块**共享同一组 self-attention 层**（参数共享）。每个 block 内的数据流：

$$
Q' \;=\; Q + \mathrm{SelfAttn}\big( [Q;\, U] \big), \qquad Q'' \;=\; Q' + \mathrm{XAttn}\big( \mathrm{LN}(Q'),\; z_{img} \big), \qquad Q^{out} \;=\; Q'' + \mathrm{FFN}\big( \mathrm{LN}(Q'') \big) \qquad (3)
$$

**符号**：$U \in \mathbb{R}^{T \times d_q}$：文本 token 的嵌入（共享 self-attn 内与 queries 同序列计算，可见性由 §3.2 的 mask 控制）；$\mathrm{XAttn}(A, B)$：以 $A$ 为 query、$B$ 为 key/value 的 cross-attention；cross-attn 层隔层插入（每隔一个 self-attn 层插一个，论文口径）；$z_{img}$ 来自**冻结**的 ViT，全程不回传梯度。式 (3) 的 LN 位置按 2.7 的 Pre-LN 惯例写成教学化简；Q-Former 由 BERT-base 初始化，实际是 Post-LN——LN 施加在残差相加之后（2.7 的两种位置对照），不改变本节的维度与数据流结论。$B$ = 32 个 query 在 $L$ 层内反复做「self-attn 整合已读信息 + cross-attn 从 576 个 patch 里再取一轮」，最终输出 $B$ 行定长表征。形状走读：

```
z_img [N × d_v]（冻结 ViT，N = 576）        U [T × 768]（文本 token）
        │ 作为 K/V（隔层）                        │
        ▼                                        ▼
   Q [32 × 768] ──共享 self-attn([Q; U]，mask 按目标切换)──► 文本侧输出
        │ cross-attn ← z_img（Q 侧只进不出：文本永远看不到图像特征）
        ▼
   Q_out [32 × 768] ──（stage 2）线性投影──► [32 × d_llm] 拼进 LLM 前缀
```

「文本永远看不到图像特征」是结构事实：图像信息进入文本侧的唯一通道是 queries 这个 32×768 的瓶颈——这就是 Q-Former 作为接入点 (c)（8.1 三接入点）的形态：Q-Former 输出 tokens 走 input-level 拼接，但瓶颈在 connector 内部已经完成。

### 3.2 三目标的注意力 mask

同一套参数要服务三个目标，靠的是给共享 self-attn 换三张可见性 mask（论文 Figure 2 的口径，ASCII 重绘）：

```
        ITC（对比）              ITM（匹配）               ITG（生成）
      Q   T                    Q   T                     Q   T
   Q [ ■   □ ]              Q [ ■   ■ ]               Q [ ■   □ ]
   T [ □   ■ ]              T [ ■   ■ ]               T [ ■   ◤ ]
   queries 与文本互不可见      全双向、互相可见           queries 不见文本；
                                                       文本见全部 queries + causal 看文本
```
（■ = 可见，□ = 不可见，◤ = 下三角 causal mask，2.4 式 (1) 惯例。）

### 3.3 三目标公式

记号：batch 内 $E$ 对图文；图像 $i$ 的 query 输出 $\{ z_{q,i}^{(k)} \}_{k=1}^{B}$；文本 $j$ 取文本侧 [CLS] 位输出 $z_{t,j}$；$\bar{\cdot}$ 表示 $\ell_2$ 归一化；$\tau$ 为温度（P6 对比学习记号沿用）。

**ITC（image-text contrastive）**。先定义图文相似度为「$B$ 个 query 里与文本最对齐的那个」的内积：

$$
s_{ij} = \max_{k \in \{1, \ldots, B\}} \left\langle \bar z_{q,i}^{(k)},\ \bar z_{t,j} \right\rangle \qquad (4)
$$

再按 InfoNCE 的双向形式写 loss（单向式见 6.1 式 (13)，双向形式与 batch 负样本账归 6.2；mask 用 ITC 档，queries 与文本互不可见——相似度只经读出端的 $\max$ 耦合）：

$$
L_{ITC} = -\frac{1}{2E} \sum_{i=1}^{E} \left[ \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{E} \exp(s_{ij}/\tau)} + \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{E} \exp(s_{ji}/\tau)} \right] \qquad (5)
$$

式 (4) 的 $\max$ 是 Q-Former 与普通 CLIP 式对比的分界：整图-整句的对齐被下放到 query 粒度——32 个 query 各自可以专长于一种读法（主体、背景、文字区域……），相似度取其中最强响应。代价写在 6.1 的 InfoNCE 账里：跨样本归一化耦合原样存在，batch 内负样本就是全部负样本。

**ITM（image-text matching）**。mask 换全双向，每个 query 输出过一个共享线性分类头得 logit，$B$ 个 logit 取均值后过 sigmoid 当匹配概率：

$$
\ell_i^{(k)} = \mathrm{Linear}\!\left( z_{q,i}^{(k)} \right), \qquad p_i = \sigma\!\left( \frac{1}{B} \sum_{k=1}^{B} \ell_i^{(k)} \right), \qquad L_{ITM} = -\frac{1}{E} \sum_{i=1}^{E} \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big] \qquad (6)
$$

**符号**：$y_i \in \{0, 1\}$：配对标签；$\sigma(u) = 1/(1+e^{-u})$。ITM 是逐对二分类（与 6.1 式 (14) 的 sigmoid 读法同族），负样本按 ITC 相似度做 hard negative 挖掘（论文口径）：难负例迫使 queries 编码细节而非全局 gist——这是对瓶颈的「防偷懒」约束。

**ITG（image-grounded text generation）**。mask 换生成档（queries 互见、不见文本；文本见全部 queries、文本内部 causal），文本侧按 LM 目标生成 caption：

$$
L_{ITG} = -\sum_{m=1}^{T} \log p\!\left( u_m \mid u_{<m},\; Q \right) \qquad (7)
$$

**符号**：$u_m$：caption 第 $m$ 个 token；条件 $Q$ 经共享 self-attn 的可见性进入（文本每个位置都能 attend 全部 queries）。ITG 是三个目标里唯一强迫瓶颈「把图说全」的：caption 的每个词都要从 32 个 query 里读出来，query 表征必须承载可生成级别的细节。三目标联合优化（stage 1 同批计算）。

### 3.4 两阶段训练与「小数据不划算」的完整论证

| 阶段 | 冻结 | 可训 | 目标 | 数据（论文口径） |
| --- | --- | --- | --- | --- |
| stage 1：表征学习 | ViT | Q-Former（约 1.9 亿参数） | 式 (5)(6)(7) 联合 | 129M 图文对 |
| stage 2：生成学习 | ViT + LLM | Q-Former + 线性投影到 $d_{llm}$ | 图条件 LM loss | 同族 caption 数据 |

两阶段的分工对应 §1 的自由度：stage 1 学自由度 (ii)——把 $N$ 选成 $B$ 的取舍函数；stage 2 学自由度 (i)(iii)——把这 $B$ 个表征翻译进冻结 LLM 的 embedding 空间。LLM 全程冻结：BLIP-2 的设计目标是「不动预训练资产」，代价就是取舍函数必须由 connector 独力承担。

现在把 §2 的两点口径收回到机制上。Q-Former 的可训容量约 1.9 亿参数（queries + 12 层共享 self-attn + 隔层 cross-attn + FFN，论文口径），要学到的函数是「从 576 个 patch 表征里提取哪 32 个」——这个选择函数没有任何结构先验，queries 的初值对输入内容无感知，全部靠 ITC/ITM/ITG 的梯度从 129M 对里塑造。对照 MLP（式 (2)）：约 $10^{7}$ 参数、逐 token 结构固定、学的只是空间对齐——自由度 (ii) 干脆不做（$M = N$），把取舍问题推给下游 LLM 用 attention 自行解决。LLaVA-1.5 的 1.2M 条数据训不动前者、足够训后者，其 discussion 口径（Q-Former 类方案训练语料在数亿级、1.2M 规模下两层 MLP 已够，[Liu et al., 2023](../references.md)，[A]）就是这条容量-数据匹配账的论文侧印证。**两族没有绝对优劣，只有「数据规模 × 是否允许动 LLM」坐标系下的相对划算**——这条坐标系会原样出现在 §11 的决策表里。

## 4. Perceiver resampler（Flamingo）：变长进、定长出

Flamingo 的 resampler 与 Q-Former 同属定长瓶颈族，差别在结构更简：没有文本子模块、没有三目标，只有一组可学 latent $X \in \mathbb{R}^{B \times d}$ 对视觉特征做 cross-attention（[Alayrac et al., 2022](../references.md)，arXiv:2204.14198，[B]——DeepMind 未开源，公式引论文，代码对照开放复现 open_flamingo，[open_flamingo repo](../references.md)，[A]）：

$$
V = \mathrm{softmax}\!\left( \frac{(X W^{Q})\, (z_{img} W^{K})^{\top}}{\sqrt{d_k}} \right) (z_{img} W^{V}) \in \mathbb{R}^{B \times d} \qquad (8)
$$

**符号**：$X$：$B$ 个可学 latent（Flamingo 主档 $B$ = 64，论文口径）；$z_{img} \in \mathbb{R}^{N \times d_v}$：冻结视觉塔输出（视频则把 $F$ 帧特征与时间位置编码拼接后一起当 K/V）；$W^{Q}, W^{K}, W^{V}$：投影矩阵。式 (8) 之上再堆 latent 自注意力与 FFN 构成 resampler 塔。

定长化公式一句话：$M = B$，与 $N$ 无关。成本账也直接：每层 cross-attn 是 $B \times N$ 的 score 矩阵，复杂度 $O(BNd)$ 而非视觉侧全自注意力的 $O(N^{2}d)$——$N$ 大（视频帧 × 每帧 patch）时，瓶颈把二次项挡在了 LLM 之外。与 Q-Former 的对照：同样是可学 query 的瓶颈，Q-Former 靠三目标在 stage 1 专门塑造取舍函数，resampler 则把取舍直接交给下游 LLM 的图文交叉注意力目标端到端学（Flamingo 的 gated xattn 接入点 (b) 见 8.1）——瓶颈族内部的又一个「目标在哪学」的分工。

## 5. 规则压缩：pixel shuffle 推导与 token merging outline

### 5.1 pixel shuffle（space-to-depth）：2×2 相邻 token 合成 1 个

瓶颈族用「可学取舍」压 token；规则压缩族换了思路——不取舍，只做**无损重排 + 学习型投影**。pixel shuffle（空间到通道重排）把 $2 \times 2$ 相邻 patch 的特征沿通道维拼接，再线性投影回 $d_{llm}$。设压缩率 $r = 1/2$（$2 \times 2$ 合并），$z_{img}$ 排成 $h \times w$ 网格（$h = w = 24$、$d_v$ 通道的 CLIP-336 实例）：

$$
z'_{i,j} = \Big[ z_{2i,\, 2j};\; z_{2i,\, 2j+1};\; z_{2i+1,\, 2j};\; z_{2i+1,\, 2j+1} \Big] \in \mathbb{R}^{4 d_v}, \qquad v_{i,j} = W z'_{i,j} + b, \quad W \in \mathbb{R}^{d_{llm} \times 4 d_v} \qquad (9)
$$

**符号**：$[\,\cdot\,;\,\cdot\,;\,\cdot\,;\,\cdot\,]$：通道维拼接；$i, j \in \{0, \ldots, h/2 - 1\}$：合并后的网格坐标。token 数与通道数的变换律：

$$
M = N \cdot r^{2} = \frac{N}{4} \qquad (10)
$$

token 数乘 $r^{2}$（$r = 1/2$ 即 ÷4），通道数同步乘 $1/r^{2}$（×4）——两步互抵，总维度守恒。

形状走读（CLIP-336/14 ⇒ InternVL 风格 connector）：

```
z_img: 24×24 × d_v  （N = 576）
   │  式 (9) 拼接：2×2 邻域 → 通道 ×4（纯 reshape，无参数、无损）
   ▼
12×12 × 4 d_v
   │  W ∈ R^{d_llm × 4 d_v}（唯一可训处）
   ▼
v: 144 × d_llm      （M = 576/4 = 144）
```

两条性质把 pixel shuffle 与瓶颈族分开。**其一，拼接步无损**：式 (9) 只是指标重排，信息没有任何丢弃；取舍全部集中在投影 $W$ 上——当 $d_{llm} \ge 4 d_v$ 时存在满秩 $W$ 使整步无损，实际配置 $d_{llm} < 4 d_v$（如 3584 < 4096）时 $W$ 是一个学出来的低维瓶颈。**其二，合并谁由结构写死**：2×2 邻域是固定先验，不像 Q-Former 的 32 个 query 要从数据里学选择函数——这就是它在 academic 预算下好训的原因，参数与 MLP 同量级（$4 d_v d_{llm}$，约 $10^{7}$ 级）。成本收益记在 LLM 端：视觉段 token 数 ×1/4，attention 二次项 ×1/16，逐 token 线性项 ×1/4。

用法实测两家：InternVL 官方代码即 `pixel_shuffle(scale_factor = downsample_ratio = 0.5)`、`num_image_token = (H/p)^{2} · r^{2}`（`internvl_chat` 的 modeling 文件，[InternVL repo](../references.md)，[A]；论文 [Chen et al., 2023](../references.md)，arXiv:2312.14238，[A]）；Qwen 系同一族操作叫 spatial merge——`spatial_merge_size = 2`，等效 patch 从 14 变 28（[Qwen2.5-VL-7B config](../references.md)，[A]）。Qwen 把这个压缩与动态分辨率绑在一起用，见 §8。

### 5.2 token merging（相似度合并）：机制 outline

规则重排之外是内容自适应路线：按特征相似度把冗余 token 两两合并（典型做法：把 token 分两组、按 key 的余弦相似度做二部配对、合并最相似的若干对、特征取平均），压缩率可以逐层调节。两点机制注记：(a) 合并是内容决定的——信息量低的区域（天空、空白背景）被并得多，规则重排做不到这一点；(b) 代价是网格规则性被破坏——合并后 token 不再有整齐的 $(h, w)$ 坐标，与 2D RoPE / M-RoPE 这类按网格坐标旋转的位置编码相性最差。本条无收录条目，只给机制 outline；当代主流 VLM 的实际选择（InternVL、Qwen 系、LLaVA 系）全部落在规则重排一侧，与 (b) 的兼容性账一致。

## 6. 固定分辨率的信息损失账

connector 的四族都默认视觉塔输出 $z_{img}$ 是给定的。但视觉塔本身有一个更上游的瓶颈：固定输入分辨率。把这笔账算出来，§7–§10 的全部技术都是对它作答。

以 CLIP-336 档为例：任意输入图先 resize 到 336×336 再 patchify。取一张文档图 1568×1176（扫描件常见尺寸），缩放比 $336/1568 \approx 0.214$——图上一个 12 px 高的字符缩到 $12 \times 0.214 \approx 2.6$ px，而一个 patch 边长 14 px：**一个 patch 里混进约 5 行文字**。patch 嵌入是线性混合（6.1 式 (2)），5 行字的像素被加进同一个 token 的输入里，任何下游 connector 都无法把它们再分开。

损失发生在哪一步要定位准确：resize 是不可逆低通滤波，信息在 patchify 之前已经丢了，$N$ = 576 个 token 只是忠实地继承了这个损失。反向算保留账：要在 patch 粒度保留原图全部可分辨结构，需要 $(1568/14) \times (1176/14) = 112 \times 84 = 9408$ 个 token——固定 336 档的 16.3 倍。小字识别（OCR 类任务）是这笔账最敏感的探针：字符笔画是高频信号，恰好是低通第一个砍掉的成分。三条出路对应三节：切块保留局部全分辨率（§7 AnyRes）、不 resize 直接变长 patchify（§8 naive dynamic resolution）、视觉塔本身吃任意分辨率（§10 Pixtral native）。共同代价都是 $M$ 上涨——于是位置编码必须跟上：变长、变网格、还要兼容视频时间轴，这就是 §9 M-RoPE 的存在理由。

## 7. AnyRes tiling 数学（LLaVA-NeXT → LLaVA-OneVision）

AnyRes 的思路：不硬 resize 整图，而是把原图切成若干块，每块按视觉塔原生分辨率（$S \times S$ = 336×336）吃，外加一张整图缩略图保留全局（[Li et al., 2024](../references.md)，arXiv:2408.03326，[A]，LLaVA-OneVision 承继 LLaVA-NeXT 的 AnyRes 方案）。

**网格选择规则**（实现口径：LLaVA-NeXT 的 `select_best_resolution`，`llava/mm_utils.py`，[LLaVA-NeXT repo](../references.md)，[A]）。预定义候选网格集 $\mathcal{G} = \{ (r_h, r_w) \}$（行×列分块数）。对图像 $(H, W)$ 与候选 $(r_h, r_w)$，画布为 $(r_h S, \; r_w S)$，等比缩放图像使其恰好放进画布，并定义两个键：

$$
s = \min\!\left( \frac{r_h S}{H},\; \frac{r_w S}{W} \right), \quad H' = \operatorname{floor}(sH),\; W' = \operatorname{floor}(sW), \quad e = \min\!\left( H' W',\; HW \right), \quad \rho = r_h r_w S^{2} - e \qquad (11)
$$

**符号**：$e$：effective resolution——缩放后面积与原图面积取小（等比缩放不增像素，$e$ 即真正落进画布的源信息量）；$\rho$：wasted resolution——画布面积中未被有效像素填满的部分。选择规则是**字典序**：先取 $e$ 最大的候选，平局再取 $\rho$ 最小者——「最小浪费」只是平局后的第二键。读法：第一键让切块后的可用像素总量最大（分辨率信息优先），第二键才压 padding 虚账。两键会与「单看利用率」的规则分叉：800×1200 的图在 (1,2) 与 (2,2) 之间，字典序选出 (2,2)（$e$ = 301056 > 169344），而利用率 $e / (r_h r_w S^{2})$ 键会选 (1,2)（0.75 > 0.67）——写作口径按实现取前者。

**切块与拼接**。选定 $g^{*} = (r_h, r_w)$ 后 padding 到画布、切成 $k = r_h r_w$ 块，每块独立过同一视觉塔得 $N$ = 576 tokens；另把原图整体等比压进 $S \times S$ 得缩略图，再过同一视觉塔。token 总账（约定页口径）：

$$
M = (k + 1) \cdot N \qquad (12)
$$

4 切实例（$g^{*}$ = (2,2)）：$M = 5 \times 576 = 2880$（LLaVA-NeXT 口径，CLIP-336 每块 576 token；LLaVA-OneVision 换 SigLIP-384、每块 729 token，4 切为 $5 \times 729 = 3645$）。

```
原图 1568×1176                缩略图：整图 → 336×336 → 576 tokens（全局）
   │ 选网格 (r_h, r_w)：式 (11) 字典序（先 max e、平局 min ρ）
   ▼
resize 进 (r_h·336)×(r_w·336) 画布，padding，切 k 块
   │ 每块 336×336 → 同一 ViT → 576 tokens
   ▼
k 块局部 tokens（raster 序拼接，行间可学 newline 向量分隔）+ 缩略图 576
   │
   ▼
共 (k+1)·576 tokens 进 LLM
```

**位置编码处理**。每块内部用视觉塔原生的 24×24 位置嵌入——同一张表对全部块复用，坐标是块内局部坐标，ViT 侧不需要任何改造；**跨块布局不进 ViT 的位置系统**：块 token 在 LLM 序列里按 raster 序排布，行间以可学的 image-newline 向量分隔（LLaVA-NeXT 实现口径，[LLaVA-NeXT repo](../references.md)，[A]），块间相对位置由 LLM 自己的 1D 位置编码按序列序承载。全局上下文由缩略图那份 576 tokens 携带。一句话：2D 结构在 ViT 内局部承载、全局布局交给 LLM 的 1D 序——这套工程答案的软肋是 LLM 侧的 1D RoPE 对「第 600 个 token 属于右上块第 3 行」这类 2D 事实毫无显式先验，全靠学。§9 的 M-RoPE 正是把 2D（乃至 3D）坐标系统一收进 LLM 侧位置编码的正式解。

成本账两笔：视觉塔前向 ×$(k+1)$；LLM prefill 的视觉段长度 $(k+1) \cdot 576$，attention 二次项随之平方放大。AnyRes 的分辨率增益是用 token 预算买的——这是 §11 决策表里「灵活性」一列的价格。

## 8. Qwen2-VL naive dynamic resolution：任意 $H \times W$ 直接进

Qwen2-VL 把问题反过来问：为什么要有固定画布？（[Wang et al., 2024](../references.md)，arXiv:2409.12191，[A]，论文标题即 *...at Any Resolution*）。naive dynamic resolution（原生动态分辨率）的做法：不切块、不缩放——原图（仅等比调整到 28 的整数倍：patch 14 × merge 2，保证 merge 后网格整除）直接 patchify，token 数随分辨率自然涨落。管线：

$$
(H, W) \;\to\; \underbrace{\left( \tfrac{H}{14} \times \tfrac{W}{14} \right)}_{\text{patchify},\; p = 14} \;\to\; \underbrace{z_{img}}_{\text{ViT with 2D RoPE}} \;\to\; \underbrace{M = \frac{HW}{28^{2}} = \frac{HW}{784}}_{\text{spatial merge } 2 \times 2} \qquad (13)
$$

**符号**：$p$ = 14：patch 边长；三步依次为 patchify、变长 ViT（2D RoPE）、spatial merge 2×2——末步即 §5.1 的 $r$ = 0.5 档，等效 patch 28，故 $M = (H/28) \cdot (W/28)$。最终进 LLM 的视觉 token 数 $M$ 与像素数成正比——分辨率就是算力旋钮，预算由实现侧的 min/max pixel 上下界夹住（超出界先等比缩放到界内）。

这个方案成立依赖三个配套：(a) **视觉塔必须吃变长序列**——Qwen2-VL 的 ViT 用 2D RoPE（6.1 式 (7)(8) 的轴向旋转）替换 learned 位置表，任意网格坐标直接代入，不需要 6.1 §3.1 那种表插值；(b) **视觉塔的 $O(N^{2})$ 要有对策**——Qwen2.5-VL 的视觉塔 32 层里大部分是 window attention（窗口 112 px，即 8×8 patch），只在第 [7, 15, 23, 31] 层做全局 attention（[Qwen2.5-VL-7B config](../references.md)，[A]）；(c) **LLM 侧要有一个能同时装下「文本位置、图像 2D 网格、视频时间」的位置编码**——这就是 §9 的 M-RoPE，也是整条管线的落点。

三方案账面对照（同一张 1344×672 图，宽高比 2:1）：

| 方案 | 每图 token 数 $M$ | 算式 |
| --- | --- | --- |
| 固定 336（CLIP 档） | 576 | 常数（宽高比被压坏） |
| AnyRes（$S$ = 336，选网格 (1,2)，零浪费） | $3 \times 576$ = 1728 | 式 (12)，$k$ = 2 + 缩略图 |
| Qwen2-VL 系（$p$ = 14 + merge 2） | $1344 \times 672 / 784$ = 1152 | 式 (13) |
| Pixtral（$p$ = 16，无 merge） | $1344 \times 672 / 256$ = 3528 | §10 |

## 9. M-RoPE：$(t, h, w)$ 三元组分解（本章核心）

2.5 式 (13) 的 RoPE 把相对位置钉在 score 上，但 position id 是一个标量——一维。§7 末指出了 1D 序承载 2D 布局的先天缺陷；Qwen2-VL 的答法是把 position id 扩成三元组 $(t, h, w)$，把时间、行、列三个轴各自挂到 rotary 频段的一段上（[Wang et al., 2024](../references.md)，arXiv:2409.12191，[A]；Qwen2.5-VL 沿用并扩展，[Bai et al., 2025](../references.md)，arXiv:2502.13923，[A]）。走读锚点：transformers 的 `modeling_qwen2_5_vl.py`——`get_rope_index` 的三元组构造与 `mrope_section` 的频段切分（[transformers Qwen2.5-VL modeling](../references.md)，[A]）。

### 9.1 三元组构造规则

每个 token（无论文本还是视觉）领到一个三元组 $(t_s, h_s, w_s)$。记 $t_0$ 为当前图像/视频段的起始偏移（= 之前所有 token 三值的最大值 + 1），$H_g, W_g$ 为 merge 后的 patch 网格尺寸（$H_g = H/28$、$W_g = W/28$），视频共 $T$ 个时间步：

$$
(t_s, h_s, w_s) = \begin{cases} (c_s,\; c_s,\; c_s), & s\ \text{text token} \\ (t_0,\; t_0 + i,\; t_0 + j), & s\ \text{image patch } (i, j) \\ (t_0 + \tau_f,\; t_0 + i,\; t_0 + j), & s\ \text{video patch } (i, j) \text{ at step } f \end{cases} \qquad (14)
$$

**符号**：$c_s$：文本位置的连续计数，逐 token 加一；$i \in \{0, \ldots, H_g - 1\}$、$j \in \{0, \ldots, W_g - 1\}$：merge 后网格坐标；$\tau_f$：时间坐标，随时间步递增（两代规则不同，见下）。视觉段结束后，后续文本从 $t_0 + \max(T, H_g, W_g)$ 继续（三值恢复同值递增；实现口径的版本差异见 §9.4 注记）。

三条规则各有一句机制读法。**文本三值同值**：§9.3 将证明这使纯文本段与标准 1D RoPE 逐点相同——文本能力不被改造，这是整个设计能直接套在现成 LLM 上的原因。**图像 $t$ 恒定**：图像没有「时间」，$t$ 轴对图内 patch 不加区分（§9.3 将看到 $t$ 段对图内 attention 退化为纯内容项）；$(h, w)$ 承载 2D 网格。**视频 $t$ 递增**：Qwen2-VL 按帧序递增；Qwen2.5-VL 把 $t$ 轴与**绝对时间**对齐——dynamic FPS 采样下相邻帧的 $t$ 差正比于真实秒数间隔而非帧序号（[Bai et al., 2025](../references.md)，[A]；换算常数见 HF config `tokens_per_second = 2`，[qwen25vl-config](../references.md)，[A]，8.4 展开）——模型由此读到「事件的真实节奏」，而非被采样率扭曲的帧计数。

### 9.2 频段切分：三段共用一条几何级数

设 head 维 $d_h$ = 128（Qwen2.5-VL-7B 实测：3584 / 28 头），rotary 成对旋转共 $d_h / 2$ = 64 个频率通道。M-RoPE 把 64 个通道按 `mrope_section` 切成三段，实测配置（[Qwen2.5-VL-7B config](../references.md)，[A]）：

$$
(K_t, K_h, K_w) = (16, 24, 24), \qquad K_t + K_h + K_w = 64 = d_h / 2 \qquad (15)
$$

**符号**：$K_t, K_h, K_w$：分配给时间/行/列轴的旋转块数。频率 schedule 与 2.5 式 (16) 同一条几何级数：$\theta_j = base^{-2j/d_h}$，$j \in \{0, \ldots, 63\}$，$base$ = $10^{6}$（config 的 `rope_theta`）——三段共用同一条级数，只是角度来源的坐标不同。token $s$ 的旋转矩阵：

$$
R^{(s)} = \operatorname{diag}\Big( \underbrace{R(t_s \theta_0), \ldots, R(t_s \theta_{15})}_{\text{t-section: } 16},\ \underbrace{R(h_s \theta_{16}), \ldots, R(h_s \theta_{39})}_{\text{h-section: } 24},\ \underbrace{R(w_s \theta_{40}), \ldots, R(w_s \theta_{63})}_{\text{w-section: } 24} \Big) \in \mathbb{R}^{128 \times 128} \qquad (16)
$$

**符号**：$R(\varphi)$：2.5 式 (9) 的 2D 旋转块。$q/k$ 左乘 $R^{(s)}$ 后按常例算 score，$v$ 不旋转（2.5 §4.3 的理由原样成立）。实现注记：HF 代码按 half-split 约定把三段模式复制到前后两半维再成对旋转（2.5 §4.8 的配对约定），公式层与式 (16) 等价。

```
head 维 128 = 64 个 2D 旋转块（角度 = 轴坐标 × θ_j）
┌─────────────────────────┬────────────────────────────────────┬────────────────────────────────────┐
│  j = 0 … 15（t 段）      │  j = 16 … 39（h 段）                │  j = 40 … 63（w 段）                │
│  角度 t·θ_j（最高频）     │  角度 h·θ_j                         │  角度 w·θ_j（最低频）               │
│  λ：2π → ≈1.6×10²        │  λ：≈2.0×10² → ≈2.8×10⁴             │  λ：≈3.5×10⁴ → ≈5.1×10⁶             │
└─────────────────────────┴────────────────────────────────────┴────────────────────────────────────┘
λ_j = 2π·base^{2j/d_h} = 2π·10^{3j/32}（由 config base = 10⁶、d_h = 128 现推，2.5 §4.5 口径）
```

波长分布的设计读法：空间两轴拿到低频（长波长）通道——图像网格坐标范围大（$H_g, W_g$ 可达上百），长波长通道充当全局坐标（2.5 §4.5：不满圈的通道行为接近单调坐标）；$t$ 轴拿最高频的 16 个通道，编码局部时间次序。每轴谱密度相对 1D RoPE 降至 1/4 到 3/8——这是三轴瓜分同一预算的固有代价，2.5 §4.7 partial RoPE 的「谱覆盖窄一线」同账。

### 9.3 内积不变性的三元组版与三个推论

**主命题**。记块内项 $B_j(q, k; \varphi) = (q_{2j} k_{2j} + q_{2j+1} k_{2j+1}) \cos\varphi + (q_{2j+1} k_{2j} - q_{2j} k_{2j+1}) \sin\varphi$（2.5 式 (11) 的逐块写法），则对任意两个 token $m, n$（任意模态混合）：

$$
\left\langle R^{(m)} q,\; R^{(n)} k \right\rangle = \sum_{j=0}^{15} B_j\big(q, k;\; (t_n - t_m)\, \theta_j \big) + \sum_{j=16}^{39} B_j\big(q, k;\; (h_n - h_m)\, \theta_j \big) + \sum_{j=40}^{63} B_j\big(q, k;\; (w_n - w_m)\, \theta_j \big) \qquad (17)
$$

**证明**。每块仍是独立 2D 旋转，2.5 的三条引理逐块继承：$\big(R^{(m)}\big)^{\top} R^{(n)}$ 在第 $j$ 块上等于 $R\big( (\mathrm{axis}_j(n) - \mathrm{axis}_j(m))\, \theta_j \big)$，$\mathrm{axis}_j$ 按 $j$ 所属段取 $t / h / w$；内积按块分解求和（2.5 式 (13) 的逐块论证原样套用），角度差逐段代入即式 (17)。∎

score 的「位置部分」只含三轴相对位移 $(\Delta t, \Delta h, \Delta w)$ 的分段加权和——6.1 式 (8) 的 RoPE-2D 两轴版在这里扩成三轴，且轴的语义随 token 类型切换。三个推论把 §9.1 的规则兑现：

**推论 1（文本一致性）**。文本 token $m, n$：三值同值，$\Delta t = \Delta h = \Delta w = n - m$，式 (17) 三段合并为 $\sum_{j=0}^{63} B_j(q, k; (n-m)\theta_j)$——正是 2.5 式 (13) 的 64 通道标准 RoPE。**M-RoPE 在纯文本段与 1D RoPE 逐点相同**，文本建模行为零改造。

**推论 2（图内 attention）**。同一图像的两个 patch：$t$ 恒定 ⇒ $\Delta t = 0$ ⇒ $t$ 段的 $B_j$ 退化为 $q_{2j} k_{2j} + q_{2j+1} k_{2j+1}$（$\cos 0$ = 1、$\sin 0$ = 0）——16 个通道退成位置盲的纯内容项；空间相对位移 $(\Delta h, \Delta w)$ 由后 48 个通道承载。图像内部的 2D 几何由此进入 LLM 的 score。

**推论 3（图文/跨段对）**。文本 token $m$（三值 $m$）与图像 patch（$(t_0,\; t_0 + i,\; t_0 + j)$）：三段相位差为 $(t_0 - m,\; t_0 + i - m,\; t_0 + j - m)$——patch 相对文本的「距离」自动混入其网格坐标 $(i, j)$：同一文本右侧的两张图按 $t_0$ 排先后，图内右下角的 patch 在 $h/w$ 段读出更大的相对相位。多图、图文交错的序列因此不需要任何额外分隔符来标识 2D 归属。

### 9.4 `get_rope_index` 走读（教学伪码）

三元组构造的实现核心是一个运行上界 $pos$：文本段三值同值推进，视觉段按网格展开、占据 $[pos,\; pos + \max(T, H_g, W_g))$ 区间。教学伪码（非优化实现；对照 `modeling_qwen2_5_vl.py` 的 `get_rope_index`，[transformers Qwen2.5-VL modeling](../references.md)，[A]）：

```python
def get_rope_index(segments):
    # segments: 有序段列表；文本段给 length，视觉段给网格 (T, Hg, Wg)（T=1 即图像）
    pos, ids = 0, []                       # pos: 运行 id 上界
    for seg in segments:
        if seg.kind == "text":
            for s in range(seg.length):
                ids.append((pos + s, pos + s, pos + s))      # 式 (14) 第一支
            pos += seg.length
        else:                                                # 图像/视频段
            T, Hg, Wg = seg.grid
            for f in range(T):
                for i in range(Hg):
                    for j in range(Wg):
                        ids.append((pos + tau(f), pos + i, pos + j))   # 式 (14) 第二/三支
            pos += max(T, Hg, Wg)                            # 视觉段占掉的 id 区间
    return ids
```

**符号**：`tau(f)`：时间坐标函数——Qwen2-VL 为帧序递增，Qwen2.5-VL 与绝对秒数对齐（§9.1）。两条性质可直接从伪码读出：(a) 文本段的相对结构不被中间插入的图像改变——图像只是抬高了其后文本的绝对 id，而 RoPE 的 score 平移不变（2.5 式 (13) 推论），文本内部任意两 token 的差 $n - m$ 不变；(b) 三轴 id 在同一计数系统里单调推进，跨模态的「先后顺序」与模态内的「几何坐标」用同一组整数编码——这就是「多模态 RoPE」之名的实质。

实现注记（版本口径）：式 (14) 与伪码的推进规则 $\max(T, H_g, W_g)$ 对应 Qwen2-VL 论文与早期实现；当前 transformers main 的 `get_rope_index` 改为 `current_pos += max(grid_h, grid_w) // spatial_merge_size`——视频 $t$ 轴行程不进 max。图像（$T$ = 1）下两口径一致；长视频场景的 id 推进以 main 为准。三元组构造规则本身不受影响，本章推导照原样成立。

### 9.5 与 P9 的衔接

同一构造在视频生成侧再现：视频 DiT 的 3D/分解式 RoPE 同样把 $(t, h, w)$ 三轴挂到频段分段上（P9 视频生成章展开），与本节数学同构——差别只在理解侧的三元组由输入几何直接给定，生成侧的三轴坐标要服务 latent 网格的双向 attention。

## 10. Pixtral：native resolution——2D RoPE 进塔、免 tiling

Pixtral 把 Qwen2-VL 的思路再推一步：连 spatial merge 都不要，视觉塔原生吃任意分辨率（[Agrawal et al., 2024](../references.md)，arXiv:2410.07073，[A]；HF 权重 mistralai/Pixtral-12B-2409）。机制两件套：(a) 视觉塔内用 2D RoPE（6.1 式 (7)(8) 的轴向分解原样落在 ViT block 里），任意 $(h, w)$ 网格坐标直接代入，无表可查也就没有表要插值；(b) 免 tiling——不切块、不拼缩略图，$p$ = 16 直接 patchify：

$$
M = \frac{HW}{p^{2}} = \frac{HW}{256} \qquad (18)
$$

§8 的对照表里 Pixtral 是 3528 tokens——三方案中最贵，但每一笔都花在原生分辨率细节上，没有 AnyRes 缩略图与切块之间的信息重叠冗余。变长序列的批量工程沿 6.1 §6 的 NaViT packing 路线（分块对角 mask，式 (11) 的构造）。与 Qwen2-VL 的对照构成一个干净的轴：两者都要「任意 $H \times W$ 进塔」，差别只在 token 汇率——Pixtral 不合并（$p^{2}$ = 256/token），Qwen 系 merge 后 784/token，后者把 LLM 端预算压到前者的约 1/3（1152 / 3528），代价是 28 px 等效 patch 内的高频先得在视觉塔里被读出来。端侧坐标：MiniCPM-V 面向端侧预算，走「高分辨率切片 + resampler 定长压缩」的组合路线把 $M$ 压到最小档（[Yao et al., 2024](../references.md)，arXiv:2408.01800，[A]；[MiniCPM-V repo](../references.md)，[A]）——同一个决策问题在端侧预算下的取点。

## 11. connector 选型决策表

全章收口成一张表。三列判据对应三条账：$M$ 决定 LLM 端 token 预算；灵活性决定分辨率/视频输入的支持方式；训练成本决定需要的图文数据量级。

| connector | $M$（每图） | 分辨率灵活性 | 参数量级 | 数据需求 | 代表系统 |
| --- | --- | --- | --- | --- | --- |
| 线性 | $N$ | 跟随视觉塔网格（固定分辨率） | $d_v d_{llm}$ | 最小 | LLaVA [A] |
| MLP（2 层 GELU，式 (2)） | $N$ | 同上 | $d_v d_{llm} + d_{llm}^{2}$（1024/4096 档 ≈ 21M） | academic 级（约 1M 条）可训 | LLaVA-1.5 [A] |
| 软查表（visual embedding table） | $N$ | 同上 | embedding 表量级 | 对齐目标另设 | Ovis [A] |
| Q-Former（式 (3)-(7)） | $B$ = 32（定长） | 受冻结视觉塔网格限制 | 约 1.9 亿 | 1 亿级图文对（stage 1） | BLIP-2 [A] |
| Perceiver resampler（式 (8)） | $B$ = 64（定长） | 变长进、定长出（帧×patch 展平） | latent + resampler 塔 | 大配方内嵌 | Flamingo [B] |
| pixel shuffle + 投影（式 (9)） | $N/4$（$r$ = 0.5） | 任意 $H \times W$（配动态分辨率） | $4 d_v d_{llm}$ 档 | academic 级 | InternVL [A]、Qwen2-VL 系 [A] |

分辨率路线（与 connector 正交的另一轴）：固定 336（信息损失账 §6）→ AnyRes tiling（式 (11)(12)，LLaVA-NeXT/LLaVA-OneVision [A]）→ naive dynamic resolution（式 (13) + M-RoPE，Qwen2-VL/2.5-VL [A]）→ native resolution（式 (18)，Pixtral [A]）。取点逻辑：数据在千万级以下选等长族（MLP/shuffle）；要压 token 预算又保任意分辨率，shuffle + 动态分辨率是当代默认（InternVL、Qwen 系）；要求固定视觉预算且有 1 亿级预训练数据，瓶颈族（Q-Former/resampler）的取舍函数才训得出来；端侧把 $M$ 压到最小档，resampler 族回摆（MiniCPM-V）。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| connector 契约 | $v = \phi(z_{img})$，式 (1) | 三自由度：维度对齐 / token 数变换 / 表征翻译 |
| MLP projector | 式 (2) | 等长 $M = N$；$d_v d_{llm} + d_{llm}^{2}$ 参数（1024/4096 档 ≈ 21M）；LLaVA-1.5 正文+讨论口径：academic 数据规模下 MLP 已够 [A] |
| Q-Former | 式 (3)-(7)：共享 self-attn + 隔层 XAttn、三 mask、ITC/ITM/ITG | $M = B$ = 32 定长瓶颈；约 1.9 亿参数需 129M 对训练；小数据不划算的容量-数据账（§3.4） |
| Perceiver resampler | 式 (8) | $M = B$ = 64 与 $N$ 无关；cross-attn $O(BNd)$ 挡住二次项；Flamingo [B] |
| pixel shuffle | 式 (9)(10)：$M = N r^{2}$、通道 ×$1/r^{2}$ | 拼接无损、取舍在投影；InternVL $r$ = 0.5、Qwen spatial_merge 2；LLM 端二次项 ×1/16 |
| 固定分辨率损失账 | §6 | 1568×1176 → 336：12 px 字符缩到 2.6 px，一个 patch 混 5 行字；保细节需 9408 tokens ≈ 16×576 |
| AnyRes | 式 (11) 网格选择字典序（max $e$、平局 min $\rho$）、式 (12) $M = (k+1)N$ | 4 切 → 2880；块内 ViT 原生 PE、跨块 raster 序 + newline，全局靠缩略图 |
| naive dynamic resolution | 式 (13)：$M = HW/784$ | 任意 $H \times W$；配套：ViT 2D RoPE、window attention（112 px + 全局层 [7,15,23,31]）、M-RoPE |
| M-RoPE | 式 (14) 三元组、式 (15)(16) 频段切分 [16,24,24]、式 (17) 内积不变性 | 文本三值同值 ⇒ 纯文本 = 标准 RoPE（推论 1）；图内 $t$ 段退化、$h/w$ 承载 2D（推论 2）；跨段相位自动含网格坐标（推论 3）；视频 $t$ 对齐绝对时间（→ 8.4） |
| Pixtral native | 式 (18)：$M = HW/256$ | 2D RoPE 进塔、免 tiling；token 汇率最贵的原生细节路线 [A] |
| 决策表 | §11 | 三列判据：token 预算 / 分辨率灵活性 / 训练成本 |

## 练习

1.（推导）pixel shuffle 的无损条件与成本账。(a) 证明式 (9) 的拼接步是指标置换（给出显式双射），因此无损；(b) 证明「拼接 + 线性投影」整步存在无损实现的充要条件是 $d_{llm} \ge 4 d_v$（秩论证）；(c) 取 $d_v$ = 1024、$d_{llm}$ = 3584、$N$ = 576：算压缩前后 LLM 单层 attention 对该视觉段的二次项 FLOPs 比（2.3 式 (12) 口径）。
2.（推导）M-RoPE 的两个验证。(a) 从式 (14)(16) 出发证明推论 1：文本-文本对的式 (17) 逐通道等于 2.5 式 (13)；(b) 取文本 token $m$ = 5、图像段 $t_0$ = 10、patch $(i, j)$ = (2, 3)，写出该文本-图像对在三段的相位差，并解释为什么右下 patch 的 $w$ 段相位差大于左上 patch。
3.（计算）AnyRes 网格选择。图像 1200×800、$S$ = 336、候选网格 $\mathcal{G}$ = {(1,2), (2,1), (2,2), (3,2)}：按式 (11) 的字典序逐候选算 $e$ 与 $\rho$，给出 $g^{*}$ 与总 token 数 $M$（式 (12)，$N$ = 576）；再算各候选的利用率 $e / (r_h r_w S^{2})$，验证本例两种规则同解。
4.（代码）教学实现 `get_rope_index`（§9.4 伪码补全为可运行 Python）：输入段序列（文本长、图像网格 (1, $H_g$, $W_g$)、视频网格 ($T$, $H_g$, $W_g$)，`tau(f) = f`），断言三条性质：文本段三值相等且相邻差 1；图像段 $t$ 恒定、$h$ 取遍 $[t_0, t_0 + H_g)$；视觉段之后文本从 $t_0 + \max(T, H_g, W_g)$ 起。

## 参考

- [Li et al., 2023](../references.md)：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models，arXiv:2301.12597，[A]（key: blip2）——§3 Q-Former 结构、三目标式 (5)(6)(7)、两阶段与 $B$ = 32、129M 对、约 1.9 亿参数
- [Liu et al., 2023](../references.md)：Improved Baselines with Visual Instruction Tuning，arXiv:2310.03744，[A]（key: llava-1.5）——§2 MLP connector 与数据规模口径（558K + 665K；linear→MLP 为正文陈述、Q-Former 语料规模为 limitations 讨论，均非消融表）
- [Liu et al., 2023](../references.md)：Visual Instruction Tuning，arXiv:2304.08485，[A]（key: llava）——§2 线性 projector 出处
- [Alayrac et al., 2022](../references.md)：Flamingo: a Visual Language Model for Few-Shot Learning，arXiv:2204.14198，[B]（key: flamingo）——§4 Perceiver resampler（$B$ = 64）
- [Wang et al., 2024](../references.md)：Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution，arXiv:2409.12191，[A]（key: qwen2-vl）——§8 naive dynamic resolution、§9 M-RoPE 出处
- [Bai et al., 2025](../references.md)：Qwen2.5-VL Technical Report，arXiv:2502.13923，[A]（key: qwen2.5-vl）——§9.1 $t$ 轴绝对时间对齐、§8 window attention 视觉塔
- [Chen et al., 2023](../references.md)：InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks，arXiv:2312.14238，[A]（key: internvl）——§5.1 pixel shuffle 用法
- [Li et al., 2024](../references.md)：LLaVA-OneVision: Easy Visual Task Transfer，arXiv:2408.03326，[A]（key: llava-onevision）——§7 AnyRes 承继（SigLIP-384、每块 729 token，4 切 $5 \times 729 = 3645$）
- [Agrawal et al., 2024](../references.md)：Pixtral 12B，arXiv:2410.07073，[A]（key: pixtral）——§10 native resolution + 2D RoPE
- [Yao et al., 2024](../references.md)：MiniCPM-V: A GPT-4V Level MLLM on Your Phone，arXiv:2408.01800，[A]（key: minicpm-v）——§10 端侧高分辨率对照
- [Lu et al., 2024](../references.md)：Ovis: Structural Embedding Alignment for Multimodal Large Language Model，arXiv:2405.20797，[A]（key: ovis）——§2 软查表变体
- [Su et al., 2021](../references.md)：RoFormer: Enhanced Transformer with Rotary Position Embedding，arXiv:2104.09864，[A]（key: rope）——§9 RoPE 主命题与频率 schedule 的一维原型（recap → 2.5）
- [HuggingFace, transformers Qwen2.5-VL modeling](../references.md)：[A]（key: qwen25vl-modeling）——`get_rope_index` 与 `mrope_section` 走读锚点
- [Qwen, Qwen2.5-VL-7B-Instruct config.json](../references.md)：[A]（key: qwen25vl-config）——mrope_section [16,24,24]、spatial_merge 2、rope_theta 10⁶、ctx 128000、window 112 + 全局层 [7,15,23,31]
- [Salesforce, LAVIS repo](../references.md)：[A]（key: lavis-repo）——`lavis/models/blip2_models/blip2_qformer.py` 走读锚点
- [OpenGVLab, InternVL repo](../references.md)：[A]（key: internvl-repo）——`pixel_shuffle(scale_factor=0.5)`、`num_image_token=(H/p)²·r²` 实测
- [mlfoundations, open_flamingo repo](../references.md)：[A]（key: open-flamingo-repo）——Flamingo 开放复现对照（resampler 结构）
- [LLaVA team, LLaVA-NeXT repo](../references.md)：[A]（key: llava-next-repo）——AnyRes 实现口径（newline 分隔、网格选择）
- [OpenBMB, MiniCPM-V repo](../references.md)：[A]（key: minicpm-v-repo）——端侧方案对照
- [AIDC-AI, Ovis repo](../references.md)：[A]（key: ovis-repo，现 ATH-MaaS/Ovis）——visual embedding table 对照
