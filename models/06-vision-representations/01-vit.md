# 6.1 Vision Transformer：把图像变成序列（The Vision Transformer）

> 本章解决「图像如何变成 Transformer 能吃的输入」这一个问题：patchify 的 conv(k=p, s=p) ≡ 单线性层等价性推导、token 数与算力账（224/16 → 196、336/14 → 576）、[CLS] 与 pooling 读出口、2D 位置编码三族（learned 1D / sin-cos 2D / RoPE-2D 轴向旋转——8.2 M-RoPE 的伏笔）、与文本 decoder 的结构对照表；变体 Swin（window attention + shifted window mask）与 NaViT（变分辨率 packing 的 attention mask）；SigLIP 的 sigmoid pairwise loss 推导（vs InfoNCE：免跨样本归一化）与 SigLIP 2 的多目标融合 | 前置：2.3（attention 公式、式 (12) 的可见性系数 c、 $O(T^{2})$ 账）、2.4（causal mask 惯例、KV cache）、2.5（sinusoidal 式 (1)、RoPE 主命题式 (12)(13)）、2.6（FFN）、2.7（Pre-LN） | 锚点：ViT [A]、Swin [A]、NaViT [A]、SigLIP [A]、SigLIP 2 [A]、CLIP [A]（InfoNCE 对照）、google-research/vision_transformer [A]、google-research/big_vision [A]

## 1. Patchify：conv(k=p, s=p) ≡ 单线性层

2.3 的 attention 只吃一种输入：序列矩阵 $[B, T, d]$。图像是格点张量 $x \in \mathbb{R}^{H \times W \times C}$（惯例 $C = 3$），既没有天然顺序，也没有 embedding 表可查——图像不是离散符号，「查表」这条路从第一步就不存在。ViT 的全部适配层只有两个动作：把空间维切成不重叠的方格，每格做一次线性投影（[Dosovitskiy et al., 2020](../references.md)，arXiv:2010.11929，[A]）。本节证明这个适配层与一个特定卷积是同一个对象。

给定 patch 边长 $p$（设 $p$ 整除 $H, W$），把空间维切成 $(H/p) \times (W/p)$ 的网格；格子 $(i, j)$ 取出的子块为

$$
x^{(i,j)} = x\left[ ip : (i{+}1)p,\hspace{5mu} jp{:}(j{+}1)p,\hspace{5mu} : \right] \in \mathbb{R}^{p \times p \times C} \qquad (1)
$$

**符号**： $H, W$：图像高宽； $C$：通道数； $p$：patch 边长（论文主档 $p$ = 16）；下标方括号是切片记号。把 $x^{(i,j)}$ 按固定顺序（如 $(u, v, c)$ 行优先）摊平成 $\mathrm{vec}\left(x^{(i,j)}\right) \in \mathbb{R}^{p^{2}C}$，经单个线性层投影到 stream 宽度 $d$：

$$
z^{(i,j)} = W\thinspace \mathrm{vec}\left(x^{(i,j)}\right) + b, \qquad W \in \mathbb{R}^{d \times p^{2}C} \qquad (2)
$$

**符号**： $W, b$：patch 投影的权重与偏置； $z^{(i,j)} \in \mathbb{R}^{d}$：该 patch 的初始 token。token 数账直接由网格尺寸给出：

$$
N = \frac{HW}{p^{2}} \qquad (3)
$$

两个标准实例： $224$ px、 $p$ = 16 时 $N = 14^{2} = 196$； $336$ px、 $p$ = 14 时 $N = 24^{2} = 576$（CLIP 系 ViT-L/14 的高分辨率档即此网格，[Radford et al., 2021](../references.md)，arXiv:2103.00020，[A]）。prepend 一个 [CLS] 后，序列长 $N + 1$。

**等价性命题**。式 (1)(2) 的「切格 + reshape + 线性层」与卷积 $\mathrm{conv}(k = p,\ s = p)$（输出通道 $d$）逐点相同。

**证明**。写出卷积在输出位置 $(i, j)$、输出通道 $o$ 的值：

$$
(K \ast x)_{i,j,o} = \sum_{u=0}^{p-1} \sum_{v=0}^{p-1} \sum_{c=0}^{C-1} K_{u,v,c,o}\hspace{5mu} x_{ip+u,\hspace{5mu} jp+v,\hspace{5mu} c} + b_o \qquad (4)
$$

**符号**： $K \in \mathbb{R}^{p \times p \times C \times d}$：卷积核。kernel 尺寸等于 stride 时滑动窗互不重叠：每个输入像素恰好进入一个输出位置、恰好被采样一次。把三重指标 $(u, v, c)$ 按与 $\mathrm{vec}$ 相同的顺序排成 $p^{2}C$ 的长指标，定义 $W_{o,\thinspace (u,v,c)} = K_{u,v,c,o}$，式 (4) 右端即式 (2) 第 $o$ 行——逐分量相等。反方向：任给 $W$，按同一索引映射回填即得对应的 $K$。∎

所以 patch 嵌入不是「两种实现任选」，而是一个对象的两种写法：vit_jax 用 conv 写法（google-research/vision_transformer，`vit_jax/models_vit.py` 的 `nn.Conv`，[vision_transformer repo](../references.md)，[A]），多数 PyTorch 实现同法；reshape + `nn.Linear` 则是逐元素等价的展开形。这个等价有一个好用的推论——**patch 嵌入的 FLOPs 与 $p$ 无关**：输出 $HW/p^{2}$ 个、每个代价 $2p^{2}Cd$ FLOPs，合计 $2HWCd$。 $p$ 不影响这笔账，它影响的是下游： $p$ 减半， $N$ 乘 4，attention 本体的 $O(N^{2})$ 项乘 16、逐 token 的投影与 FFN 项乘 4。patch 大小就是视觉塔的「分辨率-算力」汇率，型号后缀的 /16、/14 标的正是这个汇率。参数账同样一笔算清： $W$ 有 $p^{2}Cd$ 个参数， $p$ = 16、 $d$ = 768 时为 $16^{2} \cdot 3 \cdot 768 \approx 0.59$ M——相对 ViT-B 的 86 M 总量（原文 Table 1）是零头。

教学实现与等价性断言（教学用，非优化实现）：

```python
import torch
import torch.nn as nn

# conv(k=p, s=p) 与 reshape+linear 的等价性（式 (2)(4)）
B, H, W, C, p, d = 2, 224, 224, 3, 16, 768
x = torch.randn(B, C, H, W)                          # PyTorch 通道序 [B, C, H, W]
conv = nn.Conv2d(C, d, kernel_size=p, stride=p)      # vit_jax 的写法
out_conv = conv(x)                                   # [B, d, H/p, W/p]

P = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
P = P.reshape(B, (H // p) * (W // p), p * p * C)     # [B, N, p²C]，N = 196，(u,v,c) 摊平序
W_lin = conv.weight.permute(0, 2, 3, 1).reshape(d, -1)  # [d, p²C]：同一 (u,v,c) 序
out_lin = P @ W_lin.T + conv.bias                    # [B, N, d]
assert torch.allclose(out_conv.flatten(2).transpose(1, 2), out_lin, atol=1e-5)
```

注意权重摊平序必须与 patch 的摊平序一致（上例同为 $(u, v, c)$）；两套写法混用时序一错、结果全错，断言就是为此而写。至此图像已经变成序列，全章的数据流是：

```
x [B, H, W, C]
 │  式 (1)：切成 (H/p)×(W/p) 网格，每格 p×p×C
 ▼
patches [B, N, p²C]                N = HW/p²（式 (3)）
 │  式 (2)：W ∈ R^{d×p²C}（≡ conv(k=p, s=p)，式 (4)）
 ▼
tokens [B, N, d]
 │  prepend x_cls；加位置嵌入 P ∈ R^{(N+1)×d}
 ▼
z_0 [B, N+1, d]
 │  ×L 个 block：z' = z + MHA(LN(z))；z ← z' + MLP(LN(z'))    ← 全双向、无 mask
 ▼
z_L [B, N+1, d]
 │  读出（§2：取 z_L 的第 0 行 / 均值 / attention pooling）
 ▼
y [B, d] → 任务头
```

## 2. [CLS] token 与读出（pooling）

序列进 encoder 之前还要做两件事：prepend 一个可学向量 $x_{\mathrm{cls}} \in \mathbb{R}^{d}$，加上位置嵌入 $P \in \mathbb{R}^{(N+1) \times d}$。ViT 的 block 是标准的 Pre-LN encoder 层（[Dosovitskiy et al., 2020](../references.md)，arXiv:2010.11929，[A]；Pre-LN 的稳定性论证在 2.7）：

$$
z_0 = \left[ x_{\mathrm{cls}};\hspace{5mu} z^{(1)};\hspace{5mu} \ldots;\hspace{5mu} z^{(N)} \right] + P, \qquad z'_l = z_{l-1} + \mathrm{MHA}\left( \mathrm{LN}(z_{l-1}) \right), \qquad z_l = z'_l + \mathrm{MLP}\left( \mathrm{LN}(z'_l) \right) \qquad (5)
$$

读出取第 $L$ 层第 0 行再过一次 LN： $y = \mathrm{LN}\left( z_L^{(0)} \right)$，接分类头。 $x_{\mathrm{cls}}$ 就是 BERT 的 [CLS] 做法（3.1）：一个不对应任何输入 patch 的专用槽位，靠双向 attention 逐层汇聚全图信息，最后只读它一个。

为什么一个槽位够用，机制在 mask 的有无上。文本 decoder 里 causal mask 把「见过全部前缀」的位置钉死在末位，读出口没有自由；视觉塔全双向、无 mask，**任何固定位置都能聚合全局**，读出口因此成为自由设计变量。三条常见路线：

- **[CLS] 槽位**：如上，汇聚参数摊进 $L$ 层的 attention；读出向量只有一份，序列里多付一个 token 的成本。
- **GAP（全局均值池化）**： $y = \frac{1}{N} \sum_{i=1}^{N} z_L^{(i)}$。零汇聚参数、等权重；CNN 时代的标准读出。官方实现把 GAP 列为一等读出选项（`vit_jax/models_vit.py` 的 `classifier='gap'`，[vision_transformer repo](../references.md)，[A]）——读出口是接口级设计变量，不是结构必需。
- **attention pooling**：CLS 的推广——用一个（或数个）可学 query 对 $N$ 个 patch token 做一次 cross-attention 读出。机制上 CLS ≈「固定可学 query 的 pooling，只不过分摊在 $L$ 层里做」；显式 attention pooling 把它集中到最后一层一次做完。

三者的共同点是输出一个 $d$ 维向量；需要 dense 输出（逐 patch 特征，分割/检测用）时则绕过 pooling、直接取 $z_L$ 的第 $1$ 到 $N$ 行——SigLIP 2 的 dense 目标（§8）用的正是这份逐 patch 表示。

## 3. 2D 位置编码：learned / sin-cos 2D / RoPE-2D

2.3 §6 的置换等变命题对 patch 序列原样成立：不注入位置信息，attention 把 $N$ 个 patch 当无序集合——左上角与右下角的 patch 换位，输出只是跟着换位。与文本的唯一差别是位置天生二维：patch 的坐标是 $(h, w)$， $h \in \lbrace0, \ldots, H/p - 1\rbrace$、 $w \in \lbrace0, \ldots, W/p - 1\rbrace$。三族方案按「2D 结构显式程度」递进。

### 3.1 learned 1D：ViT 原文的选择

把网格按 raster 序拉直成 $1, \ldots, N$，配一张可学查表 $P \in \mathbb{R}^{(N+1) \times d}$ 加进 $z_0$。2D 结构没有任何显式注入——「第 15 号与第 16 号相邻（同行）、与第 29 号相邻（同列下一行）」这件事要模型自己学。ViT 原文对学出的 $P$ 做了相似度分析：同一行、同一列的位置嵌入确实更相似，2D 网格的近邻结构从数据里长了出来（[Dosovitskiy et al., 2020](../references.md)，[A]）。这套方案的软肋与 2.5 §2.3 的 learned absolute 同族：换分辨率时 $N$ 变了、表没有对应的行。补救是把 $P$（去掉 CLS 行）reshape 回 $(H/p) \times (W/p)$ 网格，做 2D 插值后再拉直——ViT 原文的高分辨率 fine-tune 正是此法。它有效但暴露了本质：表里的位置没有参数化形式，新网格上的值是插出来的，不是定义出来的。

### 3.2 sin-cos 2D：轴向拼接

把 2.5 式 (1) 的一维 sinusoidal 沿两个轴各分一半通道： $d/2$ 维编码 $h$、 $d/2$ 维编码 $w$，拼接后加性注入：

$$
\mathrm{PE}_{2D}(h, w) = \left[ g(h)\hspace{5mu};\hspace{5mu} g(w) \right] \in \mathbb{R}^{d}, \qquad g(u)_{2i} = \sin\left( u \cdot base^{-4i/d} \right),\quad g(u)_{2i+1} = \cos\left( u \cdot base^{-4i/d} \right) \qquad (6)
$$

**符号**： $g(u) \in \mathbb{R}^{d/2}$：单轴的一维 sinusoidal， $u$ 为该轴坐标； $base$：频率基数（惯例 $10^{4}$）。频率 schedule 仍是 2.5 的几何级数，只是每个轴只分到一半通道、单轴谱密度减半。与 learned 表对照：零参数、任意分辨率有定义（坐标直接代入）、网格结构显式写进函数；代价是位置信号仍走 stream 加性注入，2.5 §2.2 指出的「相对性被 Q/K 投影搅浑」的问题原样存在。

### 3.3 RoPE-2D：轴向旋转

把 2.5 式 (12) 的块对角旋转按轴分组：head 维 $d_k$ 切两半，前半的 $d_k/4$ 个频率通道用 $h$ 坐标旋转、后半用 $w$ 坐标旋转：

$$
R_{h,w} = \operatorname{diag}\left( R(h\theta_0), \ldots, R\negthinspace\left(h\theta_{d_k/4-1}\right),\hspace{5mu} R(w\theta_0), \ldots, R\negthinspace\left(w\theta_{d_k/4-1}\right) \right) \qquad (7)
$$

**符号**： $R(\varphi)$：2.5 式 (9) 的 2D 旋转块； $\theta_j = base^{-4j/d_k}$：频率 schedule（每轴 $d_k/4$ 个通道，指数分母相应换成 $d_k/2$ 维子空间，即 $base^{-4j/d_k}$）。2.5 的三条引理逐块继承（每块仍是独立 2D 旋转），内积不变性因此升级为两轴版：

$$
\left\langle R_{h_m, w_m}\thinspace q,\hspace{5mu} R_{h_n, w_n}\thinspace k \right\rangle = \left\langle q,\hspace{5mu} R_{\Delta h,\thinspace \Delta w}\thinspace k \right\rangle, \qquad \Delta h = h_n - h_m,\quad \Delta w = w_n - w_m \qquad (8)
$$

**证明**：逐块套 2.5 式 (13) 的推导——前半块的相对位移是 $h_n - h_m$、后半块是 $w_n - w_m$，求和即得。∎

score 只含两轴相对位移，「向右 2 格、向下 1 格」这类关系被直接钉在内积上。代价与 2.5 §4.7 的 partial RoPE 同账：每轴只分到一半频率通道，单轴谱覆盖窄一线。伏笔登记：轴向分解对轴数不设限——8.2 的 M-RoPE 把同一构造扩到（时间、高度、宽度）等多轴、文本 token 只占其中一轴，式 (7)(8) 就是它的二维原型；实现侧的 cos/sin 预存表与 half-split / interleaved 配对约定完全复用 2.5 §4.8。

### 3.4 三族对照

| 方案 | 注入点 | 2D 结构 | 相对位移 | 换分辨率 | 出处 |
| --- | --- | --- | --- | --- | --- |
| learned 1D | stream 加性 | 隐式（训练中学出） | 无保证 | 网格插值 | ViT [A] |
| sin-cos 2D | stream 加性 | 轴向显式 | 内积层面（2.5 式 (4) 两轴版） | 坐标直接代入 | 2.5 式 (1) 推广 |
| RoPE-2D | q/k 乘性旋转 | 轴向显式 | score 只含 $(\Delta h, \Delta w)$（式 (8)） | 坐标直接代入 | 2.5 式 (12) 推广 |

## 4. 与文本 decoder 的结构对照

把 ViT 的 block（式 (5)）与 Part 2 的文本 decoder 并排，逐组件看差异：

| 组件 | ViT（视觉 encoder） | 文本 decoder（Part 2 基准） |
| --- | --- | --- |
| 输入 | patch 线性投影，式 (1)(2)；连续值 | token embedding 查表（2.2）；离散符号 |
| attention 可见性 | 全双向、无 mask | causal mask（2.4 式 (1)） |
| attention 本体系数 | $c = 4$（2.3 式 (12)） | $c = 2$（causal 打半） |
| 位置编码 | learned 1D / sin-cos 2D / RoPE-2D（§3） | RoPE 一维主流（2.5） |
| norm | Pre-LN，LayerNorm（式 (5)） | Pre-norm，RMSNorm 主流（2.7） |
| FFN | dense GELU，隐藏维 $4d$（ratio 恰为 4） | SwiGLU，等参 ratio ≈ 8/3（2.6） |
| 线性层 bias | 有 bias | 现代 dense 普遍 bias = false |
| 读出 | [CLS] / GAP / attention pooling（§2） | last-token → LM head（2.2） |
| 推理模式 | 单次前向一次结清，无 KV cache | 增量 decode + KV cache（2.4） |

三行值得展开。**可见性**：同长序列下视觉塔的 attention 本体 FLOPs 是 causal 的 2 倍，但视觉塔没有增量 decode——一次前向结清，KV cache 这套为自回归 decode 存在的结构（2.4）整体缺席；ViT 的 $T^{2}$ 账因此只以 §5、§6 要处理的「分辨率成本」形式出现，没有 cache 形态。**配方冻结**：ViT 的 LN + GELU + dense FFN + 有 bias 是 2020 年 encoder 的直接搬运，此后视觉塔配方改动远小于文本侧——SigLIP/SigLIP 2 的塔仍是这一族（LayerNorm + dense MLP），工程差异集中在数据与目标函数（§7、§8）而非 block 内部。**规模口径**：ViT-B/16 为 $L$ = 12、 $d$ = 768、12 头、MLP 3072、共 86 M；ViT-L/16 为 $L$ = 24、 $d$ = 1024、16 头、共 307 M（原文 Table 1，[A]）——B/L 两档 head 维均为 64，与文本侧 $d_k$ = 128 的主流（2.3）不同，是另一处历史分层。

## 5. 变体一：Swin——window attention 与 shifted window mask

全双向 attention 的 $O(N^{2})$ 在高分辨率下直接撞墙：密集预测任务要逐像素输出，输入动辄 $1024 \times 1024$， $p$ = 16 时 $N = 4096$、单头 score 矩阵 16 M 元素起步，且 stage 靠前分辨率更高。Swin 的答法（[Liu et al., 2021](../references.md)，arXiv:2103.14030，[A]）：把 attention 限制在不重叠的 $M \times M$ patch 窗口内（主档 $M$ = 7），窗口之间的信息靠「移位」交替打通；同时 stage 间做 patch merging（ $2 \times 2$ 相邻 patch 合并、通道翻倍），把金字塔结构从 CNN 搬回来。

**window partition**。把 $(H/p) \times (W/p)$ 的 token 网格重排成互不重叠的窗口，每窗 $M^{2}$ 个 token，attention 在每窗内独立计算。沿用 2.3 式 (12) 的 FLOPs 口径（投影项 $8Nd^{2}$ 不变，只动二次项）：

$$
\Omega(\mathrm{MSA}) = 8Nd^{2} + 4N^{2}d, \qquad \Omega(\mathrm{W\negthinspace-\negthinspace MSA}) = 8Nd^{2} + 4M^{2}Nd \qquad (9)
$$

**符号**： $N$：token 总数； $M$：窗口边长（patch 数）。论文原式以 MACs 计（ $\Omega(\mathrm{MSA}) = 4hwC^{2} + 2(hw)^{2}C$），式 (9) 是同一笔账乘 2 的 FLOPs 口径。二次项从 $N^{2}$ 压到 $M^{2}N$：Swin 用 $p$ = 4 起步， $224$ px 下首 stage $N = 56^{2} = 3136$、 $M$ = 7，二次项缩到 $M^{2}/N = 49/3136 = 1/64$；分辨率越高省得越多（练习 3）。

**shifted window mask**。纯窗内 attention 的跨窗交互为零，感受野被窗框钉死。Swin 交替两种划分：第 $l$ 层常规分窗，第 $l + 1$ 层把窗口原点平移 $(M/2, M/2)$（ $M$ 为奇数时取整数部分， $M$ = 7 即 $(3, 3)$）——新窗横跨旧窗边界，信息跨窗流动。高效实现不真的重排，而是做 cyclic shift：把 token 网格整体滚动 $(3, 3)$，窗口重新对齐成规则批次，代价是滚动后同处一窗的 token 可能来自不同的旧窗（或越界补位区）——这些「假装的邻接」用 mask 拆穿：

$$
A_{ij} = \begin{cases} 0, & \mathrm{win}(i) = \mathrm{win}(j) \\ -\infty, & \mathrm{win}(i) \neq \mathrm{win}(j) \end{cases} \qquad (10)
$$

**符号**： $\mathrm{win}(i)$：token $i$ 在移位前的窗口编号。 $A$ 按 2.4 的惯例在 softmax 前加进 score：同旧窗的配对不受影响，跨旧窗的配对打分被 $-\infty$ 清零。一次 cyclic shift 加一张查表式 mask，就把「移位分窗」变成了规则批量矩阵乘——这是工程上的省算构造，数学上就是式 (10) 这一个分段函数。window attention 作为一种固定稀疏 pattern，与一般稀疏 attention 的谱系关系见 4.2；本节只登记它的结构接口。

## 6. 变体二：NaViT——patch n' pack，变分辨率的 packing

固定输入尺寸是 ViT 框架里最贵的一条隐含假设：图像尺寸不齐时，要么裁剪/拉伸（宽高比被破坏），要么 padding 到统一尺寸（算力花在空气上）。NaViT（[Dehghani et al., 2023](../references.md)，arXiv:2307.06304，[A]）把 NLP 的 example packing 搬进视觉：多张图——任意分辨率、任意宽高比——各自的 patch 序列拼接进同一条长度 $T$ 的序列，用 mask 把图与图隔开。

**packing 的 attention mask 构造**。设打包后 token $i$ 属于第 $e(i) \in \lbrace1, \ldots, E\rbrace$ 张图：

$$
A_{ij} = \begin{cases} 0, & e(i) = e(j) \\ -\infty, & e(i) \neq e(j) \end{cases} \qquad (11)
$$

按打包顺序这是一张分块对角 mask（第 $e$ 块大小 $N_e \times N_e$）。与式 (10) 对照：Swin 的 mask 按「旧窗口」分组、NaViT 按「原图」分组，构造是同一个分段函数，分组的语义不同；两者都与因果无关——纯边界 mask。位置坐标逐图从 0 重新计数：若配 RoPE-2D（式 (7)），每张图的 $(h, w)$ 坐标系独立，打包不改变任何一张图内部的位置几何。算力账的改善是直接的：一张 196-token 的图与一张 576-token 的图，padding 方案要 $2 \times 576 = 1152$ 个槽位，packing 只占 $196 + 576 = 772$ 个——attention 成本随槽位数平方下降，省下的全是 padding 的虚账。

packing 带来的三个性质（摘要口径，[A]）：(a) 训练分布天然覆盖多分辨率、多宽高比，单份权重支持任意输入尺寸，不需要 §3.1 那种位置表插值；(b) 推理期获得一个平滑的分辨率-成本权衡旋钮——同一权重下降低输入分辨率，token 数线性降、attention 成本平方降，可以在部署侧按预算选点；(c) 训练期还可以逐图随机丢 patch（token dropping）进一步压成本。落地注记：分块对角变长 mask 与 flash 系 kernel 的 varlen 接口同构（4.1），packing 把 batch 维摊平进序列维后吞吐反而更稳；这条路线后来被 SigLIP 2 的 NaFlex 变体收编进塔本身（§8）。

## 7. SigLIP：sigmoid pairwise loss 的推导

视觉塔的应用大头不是分类头，而是与文本塔配对做对比预训练——把图文关系统一成单位球面上的相似度。约定记号（全书 P6 统一）：图像塔输出 $v_i$、文本塔输出 $t_j$，均做 $\ell_2$ 归一化；batch 内 $B$ 对样本，logit 取

$$
\ell_{ij} = \gamma \left\langle v_i, t_j \right\rangle + b \qquad (12)
$$

**符号**： $\gamma$：可学逆温度，对数参数化（可学参数是 $\log \gamma$）； $b$：可学偏置（SigLIP 引入，CLIP 无此项即 $b = 0$）。CLIP 侧 $\gamma$ 初始化 $1/0.07 \approx 14.3$、clamp 上限 100（论文口径），SigLIP 沿用同一参数化并加 $b$。

对照基准是 InfoNCE（image→text 方向；text→image 方向对称，完整推导归 6.2）：

$$
L_{\mathrm{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\ell_{ii})}{\sum_{j=1}^{B} \exp(\ell_{ij})} \qquad (13)
$$

SigLIP（[Zhai et al., 2023](../references.md)，arXiv:2303.15343，[A]）换了一种读法：把 $B^{2}$ 个配对各自当作独立的二分类—— $z_{ij} = +1$ 表示 $(i, j)$ 是匹配对、 $-1$ 表示不匹配——逐对算 sigmoid 交叉熵：

$$
L_{\mathrm{sig}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \log \sigma\left( z_{ij}\thinspace \ell_{ij} \right) \qquad (14)
$$

**符号**： $\sigma(u) = 1/(1 + e^{-u})$； $z_{ij} \in \lbrace+1, -1\rbrace$。形式差别只有一处：softmax 的行归一化分母没了。这一处的后果要写在梯度上。InfoNCE 对 logit 的梯度（softmax + CE 的标准结果）：

$$
\frac{\partial L_{\mathrm{InfoNCE}}}{\partial \ell_{ij}} = \frac{1}{B} \left( p_{ij} - \delta_{ij} \right), \qquad p_{ij} = \frac{\exp(\ell_{ij})}{\sum_{j'} \exp(\ell_{ij'})} \qquad (15)
$$

**符号**： $\delta_{ij}$：Kronecker delta。 $p_{ij}$ 依赖第 $i$ 行全部 logit——任何一个配对的更新都要先看到整行，这就是**跨样本归一化耦合**：分布式实现必须先 all-gather 全 batch 的 embedding 才能算分母；batch 缩小时负样本池同步缩水（batch 内其余 $B - 1$ 个就是全部负样本，这一视角的展开在 6.2）。sigmoid 侧的梯度（用 $\sigma'(u) = \sigma(u)\thinspace\sigma(-u)$，配合 $\log \sigma$ 的链式，留作练习 2）：

$$
\frac{\partial L_{\mathrm{sig}}}{\partial \ell_{ij}} = -\frac{1}{B}\hspace{5mu} z_{ij}\hspace{5mu} \sigma\left( -z_{ij}\thinspace \ell_{ij} \right) \qquad (16)
$$

右端只含 $(i, j)$ 自身——没有任何跨样本项。三个直接后果：(a) loss 是 $B^{2}$ 个独立项的和，可以任意分块（chunk）累加，显存占用与通信结构解耦，不再需要为归一化同步全 batch；(b) batch size 改变不改变逐对目标的语义——每一对始终是「是否匹配」的二分类——这就是摘要口径的「小 batch 友好」：sigmoid loss 在小 batch 上优于 softmax 对照，同时允许把 batch 继续放大（[Zhai et al., 2023](../references.md)，[A]）；(c) $B^{2}$ 个配对里只有 $B$ 个正样本（正例率 $1/B$），类别天然极度不平衡—— $b$ 的负初始化（论文取 $b = -10$）在训练早期把全部 logit 压向「不匹配」，避免正样本的梯度信号被海量负样本淹没。

一个边界要划清：免掉的是**归一化耦合**，不是**相似度矩阵**—— $\ell_{ij}$ 的 $B^{2}$ 个内积一个不少，算力账与 InfoNCE 同阶；改变的是优化目标的分解形式与分布式通信结构。这也是为什么 SigLIP 常被误读为「省算」——它省的是同步，不是 FLOPs。

## 8. SigLIP 2：多目标融合与当代默认塔

SigLIP 2（[Tschannen et al., 2025](../references.md)，arXiv:2502.14786，[A]）没有替换 sigmoid 对比目标，而是在其上叠加两类额外监督（全文取证口径）：

- **LocCa captioning**：给视觉塔接一个文本 decoder，加 captioning 目标（含带定位约束的描述任务）——把「整图对齐一句话」细化到「区域对齐描述」；
- **self-distillation**：教师-学生自蒸馏目标，同时覆盖全局读出与逐 patch 的 dense 特征。

外加两个工程口径变化：全家系 multilingual；提供 NaViT 式的 native-aspect-ratio 变体（NaFlex，即 §6 的 packing 路线收编进塔本身，不再是训练侧的临时技巧）。机制读法：对比目标（式 (14)）只约束「整图-整句」的全局几何，逐 patch 特征与区域级语义不受直接监督——captioning 给 token 级的语言接地、自蒸馏把 dense 特征对齐到教师，三个目标的分工恰好对应论文标题里的 semantic understanding、localization、dense features 三项（[Tschannen et al., 2025](../references.md)，[A]）。落地坐标：SigLIP 2 是当前 VLM 视觉塔的默认选项之一，P8（8.1–8.2）以这一族塔为基线展开；权重与训练代码走 big_vision 系（google-research/big_vision，`big_vision/models/vit.py`，[big_vision](../references.md)，[A]）。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| patchify | conv(k=p, s=p) ≡ linear（式 (1)(2)(4)）； $N = HW/p^{2}$（式 (3)） | 224/16 → 196、336/14 → 576；嵌入 FLOPs $2HWCd$ 与 $p$ 无关； $p$ 减半 → attention ×16 |
| [CLS] 与 pooling | 式 (5)；GAP $y = \frac{1}{N}\sum z^{(i)}$ | 双向无 mask ⇒ 读出口是设计变量；attention pooling 是 CLS 的集中式推广 |
| 2D 位置编码 | learned 1D 表 / sin-cos 2D 式 (6) / RoPE-2D 式 (7)(8) | 轴向分解：score 只含 $(\Delta h, \Delta w)$；为 8.2 M-RoPE 多轴扩展的原型 |
| 结构对照 | §4 表 | 无 mask（ $c = 4$）、Pre-LN + LayerNorm、GELU $4d$、无 KV cache；ViT-B/16 86 M、L/16 307 M |
| Swin | 式 (9)(10) | 窗内 attention：二次项 $N^{2} \to M^{2}N$（首 stage 缩到 1/64）；shifted window = cyclic shift + 查表 mask |
| NaViT | 式 (11) | 分块对角 packing mask；任意分辨率/宽高比；推理期分辨率-成本旋钮；token dropping |
| SigLIP | 式 (12)(14)(16) | 逐对二分类、免跨样本归一化 → 小 batch 友好、通信解耦； $b$ = −10 初值对冲 $1/B$ 正例率 |
| SigLIP 2 | §8 | sigmoid + LocCa captioning + self-distillation；multilingual + NaFlex；当代 VLM 默认视觉塔 |

## 练习

1.（计算）按式 (3) 与 2.3 式 (12) 的 $c = 4$ 口径、 $d$ = 1024，分别算三档配置的单层 attention 本体 FLOPs：(a) $336$ px、 $p$ = 14；(b) $448$ px、 $p$ = 14；(c) $448$ px、 $p$ = 28。由 (b)(c) 说明「同像素不同 $p$」与 (a)(b) 的「同 $p$ 不同像素」走的是同一条账； $p$ 与分辨率谁对 $N$ 的影响是平方的、谁是通过 $N$ 传导的？
2.（推导）从式 (14) 出发补全式 (16)：用 $\sigma'(u) = \sigma(u)\thinspace\sigma(-u)$ 与 $\log \sigma$ 的链式逐对求导。再证明式 (15) 不可能改写成逐对独立的形式：即对任意 $i$， $\partial L_{\mathrm{InfoNCE}} / \partial \ell_{ij}$ 必含 $\lbrace \ell_{ij'} \rbrace_{j' \neq j}$ 的项。这条「不可能」正是 SigLIP 解耦的结构根源。
3.（计算）Swin 口径（ $p$ = 4、 $M$ = 7）：(a) $224$ px（首 stage $N$ = 3136）下式 (9) 两式的二次项各是多少（用 $d$ 表示），比值验证 $1/64$；(b) 分辨率升到 $512$ px（ $N$ = 16384）重算比值，说明为什么高分辨率密集预测必须用 window 类结构；(c) 为什么投影项 $8Nd^{2}$ 不省？窗划分影响的是哪笔账、不影响哪笔账？
4.（代码）实现式 (7) 的 RoPE-2D（复用 2.5 §4.8 的 cos/sin 预存表，half-split 或 interleaved 任选）：head 维前半按 $h$ 坐标旋转、后半按 $w$ 坐标旋转。随机生成 $q, k$ 与两组 patch 坐标，在 $(h, w)$ 网格上扫描位移，验证式 (8)：同一 $(\Delta h, \Delta w)$ 的所有位置对的 score 一致到 $10^{-5}$。

## 参考

- [Dosovitskiy et al., 2020](../references.md)：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale，arXiv:2010.11929，[A]（key: vit）——式 (1)(2)(3)(5)、learned 1D 位置嵌入与 [CLS]、ViT-B/L 规模表、CLS vs GAP 对照
- [Liu et al., 2021](../references.md)：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows，arXiv:2103.14030，[A]（key: swin）——式 (9)(10)、window partition 与 shifted window mask、patch merging
- [Dehghani et al., 2023](../references.md)：Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution，arXiv:2307.06304，[A]（key: navit）——式 (11) 的 packing mask 与分辨率-成本权衡
- [Zhai et al., 2023](../references.md)：Sigmoid Loss for Language Image Pre-Training，arXiv:2303.15343，[A]（key: siglip）——式 (12)(14)(16)、小 batch 友好与 $b$ 初始化
- [Tschannen et al., 2025](../references.md)：SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features，arXiv:2502.14786，[A]（key: siglip2）——§8 多目标融合与 NaFlex
- [Radford et al., 2021](../references.md)：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020，[A]（key: clip）——InfoNCE 对照式 (13)、 $\gamma$ 对数参数化（init $1/0.07$、clamp 100）、336px/14 档位
- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——sinusoidal 位置编码（式 (6) 的一维原型）
- [Su et al., 2021](../references.md)：RoFormer: Enhanced Transformer with Rotary Position Embedding，arXiv:2104.09864，[A]（key: rope）——RoPE 主命题（式 (7)(8) 的一维原型与引理）
- [google-research/vision_transformer](../references.md)：[A]（key: vision-transformer-repo）——`vit_jax/models_vit.py`，conv 写法的 patch 嵌入
- [google-research/big_vision](../references.md)：[A]（key: big-vision）——`big_vision/models/vit.py`，SigLIP / SigLIP 2 / NaViT 训练代码
