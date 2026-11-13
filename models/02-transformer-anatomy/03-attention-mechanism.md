# 2.3 Attention 机制解剖（Anatomy of the Attention Mechanism）

> 本章把一个公式拆到底： $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^{\top}/\sqrt{d_k})\thinspace V$ ——检索直觉（argmax 为何必须软化）、QKV 三投影的角色、 $\sqrt{d_k}$ 的完整推导链（方差 → 饱和 → softmax Jacobian → 梯度消失）、multi-head 的参数账与子空间解释、permutation equivariance 证明（2.5 位置编码必要性的前提）、FLOPs 两项拆分与 $O(T^{2})$ 内存账、手写实现与融合算子的逐行对照 | 前置：1.1（softmax 数值注记、 $\Delta^{V-1}$）、1.2（ $2mnk$ FLOPs 口径、式 (6) 的 attention 二次项）、2.1（ $[B,T,H,D]$ 记法、scores 是唯一 $T^{2}$ 对象、 $12Ld^{2}$ 的 attention 项 $4d^{2}$）、2.2（embedding 输出即本章输入 $X$）| 锚点：Llama-3.1-8B config [A]（32 Q heads、 $d_k = 128$ 的 MHA 口径算例；GQA 留给 4.1）、OLMo-2-7B config 与 modeling 代码 [A]；Vaswani et al. [A]、Team OLMo et al. [A]

## 1. 检索直觉：argmax 不可导，softmax 可导

attention 的骨架是一套按内容查表的检索结构。给定查询向量 $q$ 与 $T$ 个候选键 $k_1, \ldots, k_T$（各配一个值 $v_s$），最直接的检索是硬选择：

$$
j^{\ast } = \operatorname{arg\thinspace max}_{s} \hspace{5mu} \langle q, k_s \rangle, \qquad o = v_{j^{\ast }} \qquad (1)
$$

**符号**： $q \in \mathbb{R}^{d_k}$：query，当前位置要找什么； $k_s \in \mathbb{R}^{d_k}$：key，位置 $s$ 广告的内容； $v_s \in \mathbb{R}^{d_k}$：value，位置 $s$ 被选中后交付的内容； $\langle \cdot, \cdot \rangle$：内积。

式 (1) 作为字典查询没有问题，作为可训练组件有一个致命伤： $o$ 对 $q$ 与 $k_s$ 分段常数——score 的微小扰动不改变 argmax 的结果，除非跨过并列边界。于是 $\partial o / \partial q$ 几乎处处为 0、在并列处不存在，梯度下降拿不到任何信号， $q$、 $k$、 $v$ 的表示无从学起。「哪个 key 最匹配」不可导，就必须把选择软化。

软化的标准做法是把 argmax 换成 softmax，输出从「选中一个值」变成「全部值的凸组合」：

$$
p_s = \frac{\exp\left(\langle q, k_s \rangle / \tau\right)}{\sum_{j=1}^{T} \exp\left(\langle q, k_j \rangle / \tau\right)}, \qquad o = \sum_{s=1}^{T} p_s \thinspace v_s \qquad (2)
$$

**符号**： $\tau > 0$：温度（temperature）； $p = (p_1, \ldots, p_T) \in \Delta^{T-1}$：注意力权重（attention weights）， $\Delta^{T-1}$ 为概率单纯形（1.1 式 (1)）。

两个极限把式 (1)(2) 焊在一起： $\tau \to 0^{+}$ 时 $p$ 收敛到 argmax 上的 one-hot（并列则均分），式 (2) 退回式 (1)； $\tau \to \infty$ 时 $p$ 趋于均匀分布，输出退化为全部 value 的平均，检索消失。中间温度的行为用三个 score 走一遍： $\langle q, k_s \rangle = (2.0, 0.5, -1.0)$、 $\tau = 1$ 时 $p = (0.786, 0.175, 0.039)$——第一名拿走约 79%，但另两名仍分到非零权重， $\partial o / \partial q$ 因此处处存在且非零。softmax 就是可导的 argmax。剩下唯一的问题是 $\tau$ 取多少：第 3 节的方差推导给出答案 $\tau = \sqrt{d_k}$，而且这个答案不是调参调出来的，是算出来的。

## 2. QKV 投影与 scaled dot-product attention

§1 的 $q, k, v$ 从哪来？从同一条 residual stream 经三个独立线性投影（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.2）。取一条序列的 stream 矩阵 $X \in \mathbb{R}^{T \times d}$（暂省 batch 维），单头口径：

$$
Q = X W^{Q}, \quad K = X W^{K}, \quad V = X W^{V}, \qquad W^{Q}, W^{K}, W^{V} \in \mathbb{R}^{d \times d_k} \qquad (3)
$$

**符号**： $d$：stream 宽度（config 的 hidden_size）； $d_k$：head 维，即 2.1 shape 记法 $[B,T,H,D]$ 中的 $D$，本章按论文口径记 $d_k$； $Q, K, V \in \mathbb{R}^{T \times d_k}$。三个投影无 bias——Llama-3.1-8B、OLMo-2-7B 的 attention_bias 均为 false（各 config 条目，[A]）。

同一份 $x_t$ 被读三次、读出三个角色： $q_t$ 是「位置 $t$ 要找什么」， $k_t$ 是「位置 $t$ 广告什么」， $v_t$ 是「位置 $t$ 被选中时交付什么」。分工全部由梯度塑造；公式本身只保证 $q$ 与 $k$ 同住 $\mathbb{R}^{d_k}$ 以便内积、 $v$ 供加权和。

代入 §1 的软检索（逐 query 一行），得到 scaled dot-product attention：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V \qquad (4)
$$

**符号**： $QK^{\top} \in \mathbb{R}^{T \times T}$：score 矩阵， $(t, s)$ 元为 $\langle q_t, k_s \rangle$；softmax 逐行作用于 $s$ 维；输出 $\in \mathbb{R}^{T \times d_k}$。 $1/\sqrt{d_k}$ 即式 (2) 的 $1/\tau$，推导在第 3 节。

数据流与 shape（ $H = 1$ 时）：

```
X [B, T, d]                          ← 2.1 式 (1) 中 RMSNorm(x) 的输出
 │  式 (3)：三个 d×d_k 投影
 ├──────────► Q [B, T, d_k] ──┐
 ├──────────► K [B, T, d_k] ──┼──► S = QKᵀ/√d_k   [B, T, T]   ← 唯一 T² 对象（2.1 §3）
 └──────────► V [B, T, d_k] ──┤        │ softmax（逐行，式 (2)）
                              │        ▼
                              │   P [B, T, T]    ← 行和为 1
                              │        │ PV
                              ▼        ▼
                          O = PV   [B, T, d_k]
```

score 的另一种读法值得登记： $\langle q_t, k_s \rangle = x_t^{\top} W^{Q} W^{K\top} x_s$。矩阵 $A = W^{Q} W^{K\top} \in \mathbb{R}^{d \times d}$ 的秩至多为 $d_k$——attention 通过一个 rank-$d_k$ 的双线性型（bilinear form）读 stream。代入锚点： $d$ = 4096、 $d_k$ = 128（Llama-3.1-8B 与 OLMo-2-7B 同值），每个 score 只经过 rank-128 的透镜，是 $d$ 维内容的 3% 截面。多头（第 5 节）就是并排摆放 $H$ 个这样的低秩透镜。

## 3. $\sqrt{d_k}$ 完整推导链

论文对缩放的全部说明只有一句话： $d_k$ 较大时点积的量级变大，把 softmax 推进梯度极小的区域，因此除以 $\sqrt{d_k}$（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.2.1）。本节把这句话展开成逐步可验的链条，分四段：方差 → 饱和 → Jacobian → 梯度消失。

### 3.1 第一步： $\mathrm{Var}(q \cdot k) = d_k$

**假设（A1，初始化时的近似）**： $q, k \in \mathbb{R}^{d_k}$ 的全部 $2 d_k$ 个分量相互独立，零均值、单位方差： $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$， $\mathbb{E}[q_i^{2}] = \mathbb{E}[k_i^{2}] = 1$。随机初始化的投影作用于归一化后的 stream 时，这是对现实的合理近似；它何时失效见 §3.4 末。

记 score $s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$。先算均值：

$$
\mathbb{E}[s] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \thinspace \mathbb{E}[k_i] = 0 \qquad (5)
$$

第一步是期望的线性性，第二步用 $q_i$ 与 $k_i$ 独立，第三步用零均值。再算二阶矩，把平方展开成双重求和：

$$
\mathbb{E}[s^{2}] = \sum_{i=1}^{d_k} \sum_{j=1}^{d_k} \mathbb{E}\left[q_i k_i q_j k_j\right] \qquad (6)
$$

$i \neq j$ 的项：四个因子相互独立且各自零均值， $\mathbb{E}[q_i k_i q_j k_j] = \mathbb{E}[q_i] \thinspace \mathbb{E}[k_i] \thinspace \mathbb{E}[q_j] \thinspace \mathbb{E}[k_j] = 0$。只剩 $d_k$ 个对角项 $i = j$：

$$
\mathrm{Var}(s) = \mathbb{E}[s^{2}] = \sum_{i=1}^{d_k} \mathbb{E}\left[q_i^{2}\right] \mathbb{E}\left[k_i^{2}\right] = \sum_{i=1}^{d_k} 1 = d_k \qquad (7)
$$

即 $\mathrm{std}(s) = \sqrt{d_k}$。 $s$ 是 $d_k$ 个独立零均值项之和， $d_k \gg 1$（实际工作点 128）时由中心极限定理近似 $\mathcal{N}(0, d_k)$。

量级直觉： $d_k$ = 128 时一个典型 score 的绝对值就是 $O(11)$；式 (2) 里 score 差 1 个 nat、权重差 $e$ 倍，差 11 个 nat 差约 $6 \times 10^{4}$ 倍。高维点积天然很大，不加处理，softmax 必然落在极端区。

### 3.2 第二步：大 score 把 softmax 推进饱和区

把一行 $T$ 个 score 看作近似 iid 的 $\mathcal{N}(0, d_k)$ 样本（同一 query 对全部 key）。 $T$ 个标准正态样本的最大值约为 $\sqrt{2 \ln T}$ 个标准差（正态次序统计量的标准估计），代入 $d_k$ = 128（ $\sigma_s = 11.3$）、 $T$ = 4096（ $\sqrt{2 \ln T} \approx 4.1$）走一遍量级：

- **不缩放**：行内最大 score 高出中心的量级为 $\sigma_s \sqrt{2 \ln T} \approx 46$ nat（首阶近似；计入最大值修正项后约 40 nat，与 Monte Carlo 中位数一致）。中位 key（score ≈ 0）相对最大值的权重 $e^{-40} \approx 4 \times 10^{-18}$：约七成 key 的权重**各自**低于 $10^{-15}$、机器精度下为零——是「各自」而非「合计」，顶部少数 key 仍握着可见权重。顶部沿名次的典型间距约 $\sigma_s / \sqrt{2 \ln T} \approx 2.8$ nat，权重按公比 $`r = e^{-2.8} \approx 6\%`$ 几何衰减：恒定间距近似下，非最大项合计是几何级数 $`\sum_{k \ge 1} r^{k} = r / (1 - r) \approx 6.5\%`$，即 $p_{\max} \approx 0.94$；更细一层，高斯次序统计量的间距随名次以 $\sim 1/k$ 收缩（近顶部更密），非最大合计升至约 15%–20%、 $p_{\max} \approx 0.8$（Monte Carlo 中位 0.84）。两个口径结论一致：单个 top key 握有 $O(1)$ 权重（0.8–0.94），其余 4095 个共分零头——对照均匀分布的 $1/T \approx 2.4 \times 10^{-4}$，这就是饱和（saturation）：softmax 输出逼近 one-hot。
- **缩放后**（ $z_s = s_s / \sqrt{d_k}$，方差为 1）：行内最大值约 4.1，其余近似标准正态， $\mathbb{E}[e^{z}] = e^{1/2} \approx 1.65$，于是 $p_{\max} \approx \dfrac{e^{4.1}}{e^{4.1} + 4095 \times 1.65} \approx 0.009$——接近均匀起步，每个 key 都握着非零权重。

饱和为什么要紧，答案写在梯度上。

### 3.3 第三步：softmax 的 Jacobian

$p = \mathrm{softmax}(z)$， $p_i = e^{z_i} / S$， $S = \sum_k e^{z_k}$。对 $z_j$ 求导（商的求导法则）：

$$
\frac{\partial p_i}{\partial z_j} = \frac{\delta_{ij} \thinspace e^{z_i} S - e^{z_i} e^{z_j}}{S^{2}} = p_i \left( \delta_{ij} - p_j \right), \qquad J = \mathrm{diag}(p) - p p^{\top} \qquad (8)
$$

**符号**： $\delta_{ij}$：Kronecker delta； $J \in \mathbb{R}^{T \times T}$：softmax 的雅可比（Jacobian）矩阵。 $i = j$ 时分子为 $e^{z_i} S - e^{2 z_i}$，得 $p_i (1 - p_i)$； $i \neq j$ 时第一项为 0，得 $-p_i p_j$；合并即式 (8)。 $J$ 对称，且每行和为 0（ $\sum_j p_i (\delta_{ij} - p_j) = p_i - p_i \cdot 1 = 0$）——行和为 0 正是「全体 logit 加同一常数不改变 softmax」（第 4 节）的微分形式。

### 3.4 第四步：饱和 ⇒ 梯度消失

设上游对权重行 $p$ 的梯度为 $g = \partial L / \partial p \in \mathbb{R}^{T}$。链式法则过 softmax，利用 $J$ 对称：

$$
\frac{\partial L}{\partial z_j} = \sum_{i=1}^{T} g_i \thinspace p_i \left( \delta_{ij} - p_j \right) = p_j \left( g_j - \bar{g} \right), \qquad \bar{g} = \sum_{i=1}^{T} p_i \thinspace g_i \qquad (9)
$$

即：logit 收到的梯度 = 自身权重 $p_j$ 乘「 $g_j$ 超出权重均值 $\bar{g}$ 的部分」。代入饱和形态 $p_m = 1 - \varepsilon$（ $m$ 为 argmax 位置），其余权重合计 $\varepsilon$：

- 非最大项 $j \neq m$： $\left| \partial L / \partial z_j \right| = p_j \left| g_j - \bar{g} \right| \le 2 p_j \lVert g \rVert_{\infty}$。 $p_j$ 本身是 $e^{-\Delta_j}$ 量级（ $\Delta_j$ 为与最大 score 的差距），§3.2 的中位 key（ $\Delta \approx 40$ nat）梯度约 $10^{-17} \lVert g \rVert_\infty$——机器精度下的零。
- 最大项 $j = m$： $\bar{g} = (1 - \varepsilon) g_m + O(\varepsilon)$，故 $g_m - \bar{g} = O(\varepsilon)$，梯度 $\le 2 \varepsilon (1 - \varepsilon) \lVert g \rVert_{\infty} = O(\varepsilon)$。

整行梯度被「饱和亏空」 $\varepsilon$ 统一压制： $\left\lVert \partial L / \partial z \right\rVert_{\infty} \le 2 \varepsilon \lVert g \rVert_{\infty}$。代入 §3.2 的初始化估算， $\varepsilon$ 在 0.06–0.2 量级，且 $d_k$、 $T$ 越大越饱和。极端情形 $\varepsilon \to 0$（精确 one-hot）时式 (9) 恒等于 0：one-hot 行是梯度的死胡同，无论上游信号多强，score 与上游的 Q/K 投影一概收不到更新。 $\varepsilon$ 不是固定常数——训练中权重幅度增长会继续拉大 logit 差距， $\varepsilon$ 向 0 滑，消失自我强化。

缩放令 $\tilde{z} = z / \sqrt{d_k}$，则 $\mathrm{Var}(\tilde{z}) = 1$：softmax 落在敏感区（权重分散， $p_j$ 与 $1 - p_j$ 都是 $O(1)$），式 (9) 的梯度尺度健康。 $\sqrt{d_k}$ 就是 §1 温度问题的答案： $\tau = \sqrt{d_k}$ 是「让 score 方差与 head 维无关」的取法。

**为什么幂次恰好是 $1/2$**。设缩放 $z / d_k^{\alpha}$： $\mathrm{Var} = d_k^{1 - 2\alpha}$。 $\alpha = 0$ 即不缩放，方差随 $d_k$ 涨、饱和； $\alpha = 1$ 时方差为 $1/d_k \to 0$，score 全部挤在 0 附近、softmax 恒均匀、检索信号消失——另一个死端。 $\alpha = 1/2$ 是唯一让方差与 $d_k$ 无关的幂次，前置常数取 1 是自然单位。两个死端之间只有一条线，这就是 $\sqrt{d_k}$ 的全部理由。

**假设（A1）何时失效**。三条：(a) 训练后 $q, k$ 是同一条 stream 的投影，分量彼此相关，且匹配对的 $\langle q_t, k_s \rangle$ 偏大正是 attention 要学出的结构——相关性是目的而非噪声（相关情形的方差推广留作练习 1）；(b) RoPE 按位置旋转 $q, k$（2.5），分量不再同分布；(c) 训练中 logit 幅度持续增长，超出初始化口径。所以本节推导管的是「训练能否启动」：不缩放则初始即饱和、梯度消失、结构无从学起；缩放把初始化钉在 $\mathrm{Var} = 1$ 的可训练区，之后饱不饱和由训练自由选择。对训练中 logit 失控增长的现代补救（OLMo 2 在 q/k 投影后各加一次 RMSNorm，见第 8 节代码走读）归 2.7 的 QK-Norm 一族。

## 4. softmax 的数值稳定：max 减法

1.1 §2 留了一个兑现点：实现里先减 $\max_j z_j$ 再取 $\exp$。不变性是一行恒等式——对任意常数 $c$：

$$
\mathrm{softmax}(z - c)_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} \thinspace e^{z_i}}{e^{-c} \sum_j e^{z_j}} = \mathrm{softmax}(z)_i \qquad (10)
$$

取 $c = \max_j z_j$：平移后最大 logit 为 0，全部 $e^{z_j - c} \le 1$ 不溢出，分母 $\ge 1$ 不除零。不减的代价是实打实的：fp32 与 bf16 的指数位同为 8 位、上限约 $3.4 \times 10^{38}$，而 $e^{88} \approx 1.7 \times 10^{38}$ 已经贴顶、 $e^{89}$ 溢出成 inf；fp16 上限 65504， $z \approx 11.1$ 即溢出——§3.2 中未缩放 score 的一个标准差（11.3）就够。inf 经归一化成 $\inf / \inf = \mathrm{NaN}$，一步毒化整个 batch。反方向（ $z_j - c$ 很负）只是下溢到 0、丢一个本就趋零的权重，无害。生产代码在数值上更进一步：OLMo 2 的 eager attention 把 score 升到 fp32 再过 softmax（第 8 节），融合 kernel 内部同法。

## 5. Multi-head attention：拆分、合并与参数账

单头把 score 限制在一个 rank-$d_k$ 双线性型里（§2）。multi-head attention（MHA）并排 $H$ 个独立投影组，再把输出拼回 $d$ 维（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.2.2）：

$$
\mathrm{head}_h = \mathrm{Attention}\left(X W_h^{Q}, X W_h^{K}, X W_h^{V}\right), \qquad \mathrm{MHA}(X) = \mathrm{concat}\left(\mathrm{head}_1, \ldots, \mathrm{head}_H\right) W^{O} \qquad (11)
$$

**符号**： $W_h^{Q}, W_h^{K}, W_h^{V} \in \mathbb{R}^{d \times d_k}$：第 $h$ 个头的三投影； $\mathrm{concat}(\cdot) \in \mathbb{R}^{T \times H d_k}$； $W^{O} \in \mathbb{R}^{H d_k \times d}$：输出投影。MHA 标准形取 $d_k = d / H$。原文取 $d$ = 512、 $H$ = 8、 $d_k$ = 64。

shape 走读（Llama-3.1-8B 数值： $d$ = 4096、 $H$ = 32、 $d_k$ = 128；其 K/V 头数 $H_{kv}$ = 8 的 GQA 口径归 4.1，本章按 $H_{kv} = H$ 的 MHA 标准形讲机制，无 GQA 的真实锚点是 OLMo-2-7B 的 32/32）：

```
X [B, T, 4096]
 │ W_q / W_k / W_v（各 d×d：全部 head 的 W_h 拼成一整块；实现见 §8）
 ▼
Q, K, V [B, T, 4096] ──view──► [B, T, 32, 128] ──transpose──► [B, 32, T, 128]
        │  每 head 独立：S_h = Q_h K_hᵀ / √128
        ▼
   S [B, 32, T, T] ──mask + softmax──► P [B, 32, T, T]     ← 唯一 T² 对象，H 倍于单头
        │  O_h = P_h V_h
        ▼
   O [B, 32, T, 128] ──transpose+reshape──► [B, T, 4096] ──W^O──► [B, T, 4096]
```

**参数账**。MHA 标准形下四个拼块各为 $d \times d$：三个 QKV 块加 $W^O$，合计 $4d^{2}$，与 $H$ 无关—— $H$ 加倍、 $d_k$ 减半，投影总宽度 $H d_k = d$ 不变。这正是 2.1 式 (3) 中 attention 项 $4d^{2}$ 的出处：OLMo-2-7B 恰为此值（真 MHA），Llama-3.1-8B 的 $2.5d^{2}$（GQA 砍 K/V 宽度）是在此基准上的偏移。FLOPs 同理不随 $H$ 变（第 7 节）；唯一随 $H$ 变的账面量是 scores 内存（ $\propto H$，第 7 节）。

**子空间解释**。每个头有自己的低秩透镜 $A_h = W_h^{Q} W_h^{K\top}$（rank ≤ 128）与自己的 value 读出 $W_h^{V}$： $H$ 个头并行地在 $H$ 组不同的 rank-128 截面上做检索，concat 把 $H$ 份 $d_k$ 维结果并排， $W^O$ 再把它们线性混回 stream。注意 multi-head 不是 ensemble：各头只在 $W^O$ 处汇合一次，头间无逐层交互；「不同头学不同模式」在机制层面的根据就是各头投影独立、透镜互不共享。 $d_k = d / H$ 是惯例而非铁律——DeepSeek-V3 的 qk head 维 = 128（nope）+ 64（rope）= 192 ≠ hidden/heads（[DeepSeek-V3 config](../references.md)，[A]），head 维与 stream 维解耦的完整讨论在 2.8。

## 6. Permutation equivariance：attention 本身不知道顺序

本节证明一个贯穿后文的结构性质：不含位置信息时，attention 对输入序列的置换是等变的——打乱输入顺序，输出原样跟着打乱，内容不变。

记 $\pi$ 为 $\lbrace1, \ldots, T\rbrace$ 的置换， $P_\pi \in \lbrace0, 1\rbrace^{T \times T}$ 为对应的置换矩阵（permutation matrix）， $(P_\pi)_{ij} = 1 \Leftrightarrow j = \pi(i)$，即 $(P_\pi X)_{i,:} = X_{\pi(i),:}$（第 $i$ 行取原第 $\pi(i)$ 行）。置换矩阵正交： $P_\pi^{\top} P_\pi = I$。

**命题**（置换等变，permutation equivariance）：无 mask 的 attention 满足 $\mathrm{Attention}(P_\pi Q, P_\pi K, P_\pi V) = P_\pi \thinspace \mathrm{Attention}(Q, K, V)$。

**证明**（三步）。第一步，score 矩阵被双置换：

$$
S' = (P_\pi Q)(P_\pi K)^{\top} = P_\pi Q K^{\top} P_\pi^{\top} = P_\pi S P_\pi^{\top}
$$

第二步，softmax 与双置换交换。 $(P_\pi S P_\pi^{\top})_{ij} = S_{\pi(i), \pi(j)}$；对第 $i$ 行做 softmax 时分母 $\sum_j e^{S_{\pi(i), \pi(j)}}$ 跑遍全部列、与列序无关，等于 $\sum_s e^{S_{\pi(i), s}}$。故

$$
\mathrm{softmax}\left(P_\pi S P_\pi^{\top}\right)_{ij} = \frac{e^{S_{\pi(i), \pi(j)}}}{\sum_s e^{S_{\pi(i), s}}} = \mathrm{softmax}(S)_{\pi(i), \pi(j)} = \left( P_\pi \thinspace \mathrm{softmax}(S) \thinspace P_\pi^{\top} \right)_{ij}
$$

即 $\mathrm{softmax}(P_\pi S P_\pi^{\top}) = P_\pi \thinspace \mathrm{softmax}(S) \thinspace P_\pi^{\top}$（常数因子 $1/\sqrt{d_k}$ 随同 $S$ 被置换，不影响论证）。第三步，乘 $V$，用正交性 $P_\pi^{\top} P_\pi = I$：

$$
\mathrm{Attention}(P_\pi Q, P_\pi K, P_\pi V) = P_\pi \thinspace \mathrm{softmax}(S) \thinspace P_\pi^{\top} \cdot P_\pi V = P_\pi \thinspace \mathrm{softmax}(S) \thinspace V = P_\pi \thinspace \mathrm{Attention}(Q, K, V)
$$

证毕。

**推论（整机层面）**。QKV 投影逐位置： $(P_\pi X) W^{Q} = P_\pi (X W^{Q})$； $W^O$、残差加、RMSNorm、FFN 同样逐位置（2.1 §1 的二分法）。于是无位置信息的完整 Transformer 是置换等变的： $\mathrm{TF}(P_\pi X) = P_\pi \thinspace \mathrm{TF}(X)$。

**后果**。等变映射把序列当集合处理：token 内容与顺序的关联在计算中不可达——主谓宾重排后每个位置的输出只是原输出的重排。「语序改变语义」这类函数无法由等变映射逐位置表达。要让模型感知顺序，位置信息必须从外部注入；注入点（stream 加性 vs attention 内旋转）、注入形式（绝对 / 相对 / 线性偏置）与各自的外推行为就是 2.5 的全部内容，本节命题是那里一切论证的起点。

两个边界注记。其一，causal mask 把绝对位置锚进计算图（第 $t$ 行只见前 $t$ 列），命题对任意 $\pi$ 不再成立——这恰好说明顺序感是架构选择而非 attention 自带；mask 只提供「前面有几个 token」的计数级弱位置信息，它能撑多远是 2.5 中 NoPE 证据回答的问题。其二，等变性是「无位置信息」条件下的性质陈述，不是缺陷清单——它精确指出了位置编码必须补的是什么。

## 7. 复杂度账：FLOPs 两项与 $O(T^{2})$ 内存

逐层、逐序列（ $B$ = 1）、MHA 标准形（ $H d_k = d$），按 1.2 的 $2mnk$ 口径：

- **投影项**（随 $T$ 线性）： $W_q, W_k, W_v, W_o$ 四次 $[T, d] \times [d, d]$ matmul，各 $2Td^{2}$，合计 $8Td^{2}$——这笔进 1.2 的 $2N$ 参数账。
- **attention 本体项**（随 $T$ 平方）： $QK^{\top}$ 每头 $2T^{2} d_k$、 $H$ 头合计 $2T^{2} d$； $PV$ 同额。full（双向）共 $4T^{2} d$；causal 只需下三角、减半为 $2T^{2} d$（即 2.1 §5 三指标表的那一行、1.2 式 (6) 的单层口径）。

合计：

$$
\mathrm{FLOPs}_{\mathrm{layer}} = \underbrace{8Td^{2}}_{\text{proj}} + \underbrace{c \thinspace T^{2} d}_{\text{attn}}, \qquad c = \begin{cases} 4 & \text{full} \\ 2 & \text{causal} \end{cases} \qquad (12)
$$

**符号**： $c$：可见性系数，causal mask 使打分与加权和的有效工作量减半（1.2 式 (6) 的「causal 平均 $T/2$」）。

两项的交叉点：causal 口径 $2T^{2}d = 8Td^{2}$ 得 $T = 4d$；full 口径得 $T = 2d$。 $d$ = 4096 时分别为 16384 与 8192——现代训练序列长度（数万）已越过交叉点：长序列 regime 下 attention 本体主导每层 FLOPs，且占比随 $T$ 增长。代入 Llama-3.1-8B 口径验证： $T$ = 4096 时本体 $1.4 \times 10^{11}$ vs 投影 $5.5 \times 10^{11}$（本体占 20%）； $T$ = 131072 时本体 $1.4 \times 10^{14}$ vs 投影 $1.8 \times 10^{13}$（本体 89%）。这是 Part 4 全部省算工作的瞄准点（4.1 精确加速、4.2 稀疏、4.3 线性化）。注意与 2.1 §5 的 $T = d$ 交叉点区分：那是 attention 与 recurrent 两种层型的对比，此处是 attention 内部两项的对比。

softmax 自身的逐元素开销（exp、归一化）为每头 $O(T^{2})$，相对 matmul 项是低阶；但它的内存不是低阶：

**内存账（ $O(T^{2})$ 的物化形态）**。朴素实现把 scores 与概率矩阵各存一份 $[B, H, T, T]$，反向传播要用。bf16（2 字节 / 元素）下单层单序列共 $4 H T^{2}$ 字节。代入 $H$ = 32：

| $T$ | scores 单张 | scores + probs | $\times 32$ 层 |
| --- | --- | --- | --- |
| 4096 | 1 GiB | 2 GiB | 64 GiB |
| 32768 | 64 GiB | 128 GiB | 4 TiB |
| 131072 | 1 TiB | 2 TiB | 64 TiB |

全是 2 的幂次： $131072 = 2^{17}$， $32 \times (2^{17})^{2} \times 2 = 2^{40}$ 字节 = 1 TiB。 $T$ = 131072 是 Llama-3.1-8B 的 max_position_embeddings：一条满窗序列单层就要物化 1 TiB，任何单卡装不下；32 层全部存下供反向更不可能。两条出路——反向时重算、分块且永不物化（online softmax）——是 4.1 FlashAttention 的全部动机；在此之前，教学实现（第 8 节）老老实实物化，小 $T$ 下完全够用。推理侧逐 token 生成时这层 $T^{2}$ 被 cache 结构改写，账在 2.4。

## 8. 手写实现与 F.scaled_dot_product_attention 对照

教学版（教学用，非优化实现：显式物化 $[B,H,T,T]$ 两张矩阵，无 kernel 融合）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """教学用，非优化实现。MHA 标准形（H_kv = H_q = H，d = H * d_k），
    bias=False 对齐 Llama-3.1 / OLMo-2 的 attention_bias=false。"""

    def __init__(self, d: int, H: int):
        super().__init__()
        assert d % H == 0
        self.H, self.D = H, d // H                  # D 即 d_k（2.1 记法）
        self.W_q = nn.Linear(d, d, bias=False)      # 式 (3)：H 个 W_h^Q 拼成一个 d×d 块
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)      # 式 (11) 的 W^O

    def forward(self, x):                           # x: [B, T, d]
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H, self.D).transpose(1, 2)   # [B, H, T, D]
        k = self.W_k(x).view(B, T, self.H, self.D).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.H, self.D).transpose(1, 2)
        s = q @ k.transpose(-2, -1) * self.D**-0.5  # [B, H, T, T]，式 (4)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        s = s.masked_fill(causal, float("-inf"))    # 未来位置置 -inf（causal mask，2.4）
        p = torch.softmax(s, dim=-1)                # 式 (2)；kernel 内部做式 (10) 的 max 减法
        o = p @ v                                   # [B, H, T, D]
        o = o.transpose(1, 2).reshape(B, T, self.H * self.D)   # concat heads，式 (11)
        return self.W_o(o)                          # [B, T, d]

# 对照：官方融合算子，同输入同输出，可直接断言等价
mha = MultiHeadAttention(d=4096, H=32)
x = torch.randn(2, 1024, 4096)
q = mha.W_q(x).view(2, 1024, 32, 128).transpose(1, 2)
k = mha.W_k(x).view(2, 1024, 32, 128).transpose(1, 2)
v = mha.W_v(x).view(2, 1024, 32, 128).transpose(1, 2)
o_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 默认 scale = D**-0.5
o_ref = mha.W_o(o_ref.transpose(1, 2).reshape(2, 1024, 4096))
assert torch.allclose(mha(x), o_ref, atol=1e-5, rtol=1e-5)  # 容差口径：CPU / fp32；GPU 融合 kernel 累加序不同，需放宽
```

逐维对照：

| 维度 | 教学版 | `F.scaled_dot_product_attention` |
| --- | --- | --- |
| scores 物化 | 显式两张 $[B,H,T,T]$， $O(T^{2})$ 内存（§7 表） | 不物化：flash / mem-efficient backend 分块 + online softmax，额外内存 $O(T)$（正确性证明与账在 4.1） |
| scale | 手写 `D**-0.5` | `scale` 参数，默认 $1/\sqrt{D}$，与式 (4) 一致 |
| mask | 手工上三角 `masked_fill(-inf)` | `is_causal=True`（语义同： $j > i$ 不可见），或 `attn_mask` 显式传入 |
| 数值稳定 | `torch.softmax` 内部 max 减法（式 (10)） | kernel 内同法，且以 fp32 累加 |
| 其他 | 无 dropout、无 GQA | `dropout_p`、`enable_gqa`（GQA 归 4.1） |

真实模型代码把同一张公式图写成生产形态。OLMo 2 官方实现走读（huggingface/transformers repo，`src/transformers/models/olmo2/modeling_olmo2.py`，类 `Olmo2Attention` 与函数 `eager_attention_forward`，[HuggingFace, modeling_olmo2.py](../references.md)，[A]；论文 [Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]；config 同见 [OLMo-2-7B config](../references.md)，[A]）：投影为四个 `nn.Linear`（`q_proj / k_proj / v_proj / o_proj`，bias 由 config.attention_bias 控制、7B 为 false）；`self.scaling = self.head_dim**-0.5`——式 (4) 的 $1/\sqrt{d_k}$ 原样。eager 路径与式 (4) 逐行对应：`torch.matmul(query, key_states.transpose(2, 3)) * scaling` → 加 mask → `nn.functional.softmax(..., dim=-1, dtype=torch.float32)`（升 fp32 再 softmax，§4 数值议题的生产写法）→ `torch.matmul(attn_weights, value_states)`。与教学版两处实质差异：其一，`Olmo2Attention` 在 q/k 投影后各过一次 RMSNorm（`q_norm` / `k_norm`）——§3.4 末「训练中 logit 增长」问题的工程回答，归 2.7 的 QK-Norm 一族；其二，默认不跑 eager 路径，而经 `ALL_ATTENTION_FUNCTIONS` 派发到 SDPA / FlashAttention——算子融合归 4.1，数学内容与本章式 (4) 恒等。Llama 的同构实现（`LlamaAttention`，同 repo `src/transformers/models/llama/modeling_llama.py`，[HuggingFace, modeling_llama.py](../references.md)，[A]）结构相同、另含 GQA 的 K/V 复用，逐行走读归 2.8。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 软检索 | argmax 不可导 → softmax 凸组合，式 (1)(2) | $\tau \to 0$ 退回硬检索；温度即缩放 |
| QKV 投影 | $W^{Q}, W^{K}, W^{V} \in \mathbb{R}^{d \times d_k}$，式 (3) | 同源三角色；score 是 rank-$d_k$ 双线性型 |
| attention 公式 | $\mathrm{softmax}(QK^{\top}/\sqrt{d_k}) V$，式 (4) | 唯一 $T^{2}$ 对象 $[B,H,T,T]$ |
| $\sqrt{d_k}$ 链 | $\mathrm{Var}(q \cdot k) = d_k$（式 (5)–(7)）→ 饱和（ $d_k$ = 128、 $T$ = 4096 时 $p_{\max} \approx 0.8$–$0.94$，中位 key 权重 $\sim 10^{-18}$）→ $J = \mathrm{diag}(p) - pp^{\top}$（式 (8)）→ $\partial L / \partial z_j = p_j (g_j - \bar{g})$（式 (9)） | 假设 (A1) 是初始化口径；幂次 $1/2$ 唯一；训练中 logit 增长归 2.7 |
| 数值稳定 | max 减法不变性，式 (10) | fp32/bf16 在 $z \approx 88$ 溢出、fp16 在 $z \approx 11$ 溢出；生产侧 fp32 softmax |
| multi-head | 式 (11)；参数 $4d^{2}$ 与 $H$ 无关 | $H$ 个 rank-$d_k$ 透镜并行；scores 内存 $\propto H$ |
| 置换等变 | $\mathrm{Attention}(P_\pi Q, P_\pi K, P_\pi V) = P_\pi \thinspace \mathrm{Attention}(Q, K, V)$ | 无位置信息的整机等变 ⇒ 位置编码必要 → 2.5 |
| 复杂度 | $8Td^{2} + 2T^{2}d$（causal），式 (12)；scores+probs $4HT^{2}$ 字节 | 交叉点 $T = 4d$； $T$ = 131072 时单层 2 TiB → FlashAttention（4.1） |

## 练习

1.（推导）(A1) 的相关推广：设 $(q_i, k_i)$ 跨 $i$ 独立、同分布为零均值联合高斯， $\mathbb{E}[q_i^{2}] = \mathbb{E}[k_i^{2}] = 1$， $\mathbb{E}[q_i k_i] = \rho$。用 Isserlis 公式（二元高斯四阶矩 $\mathbb{E}[q^{2} k^{2}] = 1 + 2\rho^{2}$）证明 $\mathbb{E}[q \cdot k] = d_k \rho$、 $\mathrm{Var}(q \cdot k) = d_k (1 + \rho^{2})$。据此说明：(a) 方差量级仍是 $\Theta(d_k)$， $\sqrt{d_k}$ 的幂次结论不变；(b) 均值 $d_k \rho$ 随 $(t, s)$ 对而异、不是行内公共平移——这正是训练要塑造的结构。
2.（推导）(a) 补全第 6 节第二步的引理：对任意 $S \in \mathbb{R}^{T \times T}$ 与置换矩阵 $P_\pi$， $\mathrm{softmax}(P_\pi S P_\pi^{\top}) = P_\pi \thinspace \mathrm{softmax}(S) \thinspace P_\pi^{\top}$（逐元素验证分母与列序无关）。(b) 证明 multi-head（式 (11) 的 concat + $W^{O}$）保持置换等变。(c) 举反例说明加 causal mask 后命题对逆序置换失效。
3.（计算）Llama-3.1-8B 口径（ $H$ = 32、 $d_k$ = 128、 $d$ = 4096、bf16、MHA 标准形）：(a) $T$ = 32768 时单层 scores+probs 的字节数；(b) 求 causal 口径下 attention 本体 FLOPs 超过投影 FLOPs 的最小 $T$；(c) $H$ 加倍、 $d_k$ 减半（ $d$ 不变），投影 FLOPs、本体 FLOPs、scores 内存各如何变化？由此说明「head 数」这个旋钮调的到底是哪笔账。
4.（代码）运行第 8 节教学版与 `F.scaled_dot_product_attention` 的等价性断言；固定 $d$，扫 $T \in \lbrace1024, 2048, 4096, 8192\rbrace$，用 `torch.cuda.max_memory_allocated` 记录教学版峰值显存，在 log-log 坐标下验证 $O(T^{2})$ 斜率（应接近 2），并解释 SDPA 版在同一扫描下为何近似线性。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——式 (3)(4)(11) 出自 §3.2，缩放动机出自 §3.2.1
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）—— $H$ = 32、 $d_k$ = 128、 $T_{\max}$ = 131072 算例
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——32/32 纯 MHA 对照锚点
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——head 维 192 ≠ hidden/heads 的反例（2.8 展开）
- [HuggingFace, transformers repo OLMo 2 modeling 代码](../references.md)：[A]（key: olmo2-modeling）——`Olmo2Attention` 与 `eager_attention_forward` 走读对象
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——`LlamaAttention`，逐行走读归 2.8
