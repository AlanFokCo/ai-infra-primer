# 2.5 位置编码全谱系（The Full Landscape of Positional Encodings）

> 本章把 2.3 的置换等变推论兑现成工程谱系：sinusoidal 的频率几何级数与相对位移性质、learned absolute 的外推未定义性、T5 分桶 bias 与 ALiBi 线性惩罚两个相对偏置方案；随后是本章的推导主线——RoPE 内积不变性（ $\langle R_m q, R_n k \rangle$ 只依赖 $m - n$）从零完整证明（2D 旋转 → 块对角 → 复数视角）、频率 schedule 与波长账、远程衰减性质、partial RoPE 的 DeepSeek-V3 64/192 账、half-split 与 interleaved 两种等价实现——最后以 NoPE 的隐式位置信号与 RoPE 外推失败预告（→ 4.6）收尾 | 前置：2.3（attention 公式、置换等变命题）、2.4（causal mask、KV cache 读写模式） | 锚点：Llama-3.1-8B config [A]（ $\theta = 500000$）、DeepSeek-V3 config [A]（qk_rope_head_dim=64 / qk_nope_head_dim=128、rope_scaling 块）、Qwen3-8B config [A]（ $\theta = 1000000$）、OLMo-2-7B config [A]（ $\theta = 500000$）、GPT-2 config [W]（learned absolute 活化石）；RoFormer [A]、ALiBi [A]、T5 [A]、NoPE [B]

## 1. 必要性兑现：2.3 的推论与设计自由度

2.3 §6 证明了：无位置信息的 attention（及整机）对输入置换等变——打乱 token 顺序，输出只是原输出的重排，「语序改变语义」这类函数无从表达。位置信息必须从外部注入。注入的全部自由度可以压成三个设计维度，本章按这三维展开谱系：

- **注入点**：stream 加性（输入处把位置向量加进 embedding，随后逐层可读）vs attention 内（只改 score 的算法，q/k/v 本身不携带位置）；
- **参照系**：绝对位置（位置 $t$ 的表示只含 $t$ 自己）vs 相对位置（score 只含位移 $i - j$）；
- **外推（extrapolation）行为**：位置函数在训练窗口之外是否有定义；有定义的部分是否仍是训练时见过的形状。

谱系总览（§2–§5 逐行展开）：

| 方案 | 注入点 | 参照系 | 外推 | 出处 |
| --- | --- | --- | --- | --- |
| sinusoidal | stream 加性 | 绝对公式，内积相对 | 函数处处有定义，出窗输入未训练 | Vaswani et al. [A] |
| learned absolute | stream 加性 | 绝对查表 | 未定义（索引越界） | GPT-2 [W] |
| T5 bias | score 加性 | 相对（log 分桶查表） | 溢出落末桶，分辨率归零 | Raffel et al. [A] |
| ALiBi | score 加性 | 相对（线性惩罚） | 函数处处有定义 | Press et al. [A] |
| RoPE | q/k 乘性旋转 | 相对（内积不变性） | 慢通道角度出训练域（§6） | Su et al. [A] |
| NoPE | 不注入 | causal mask 隐式计数 | 同 mask 本身 | Kazemnejad et al. [B] |

一个贯穿全章的注记：2.3 §6 的等变命题以「无 mask」为前提，而 causal mask 已把「前面有几个 token」这一计数级信号锚进计算图（2.4 §2 末）。谱系最后一行 NoPE 不注入任何显式编码、只吃这份信号，它能撑多远由 §5 的实验证据回答。

注入点的一维值得先展开，它决定位置信息的信息流形态。stream 加性方案只在第 0 层注入一次：位置向量与 token embedding 叠加在同一组基上，靠后续层学着把两者分离开读，且位置信号要在 $L$ 层残差流的反复改写中「存活」到最后一层。attention 内方案相反：RoPE 在每层 attention 重新旋转、bias 在每层 score 重新加——位置每层重新注入，不占 stream 容量；代价是位置信号进不了 FFN（2.1 §1 二分法：FFN 逐位置），FFN 对位置的全部感知只能来自 attention 的输出。两条路线没有对错，但它们失败的部位不同：前者死在信号被 stream 稀释与窗口越界，后者死在 score 先验的形状不合任务。

## 2. 绝对位置编码：sinusoidal 与 learned absolute

### 2.1 sinusoidal：频率几何级数

Transformer 原文的位置编码（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.5）：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right) \qquad (1)
$$

**符号**： $pos \in \lbrace0, \ldots, T-1\rbrace$：位置； $i \in \lbrace0, \ldots, d/2 - 1\rbrace$：频率通道； $d$：stream 宽度（原文 $d$ = 512，256 个通道）。 $PE(pos) \in \mathbb{R}^{d}$ 与 token embedding 相加后进入第一层——stream 加性注入点。

PE 与 token embedding 同维相加：两者量级是否匹配是初始化议题——PE 的范数恒为 $\sqrt{d/2}$（练习 1），embedding 范数由 init scale 决定（2.2 §8 的 0.02 惯例），原论文的取法是 embedding 乘 $\sqrt{d}$ 再相加（同篇 §3.4）。

把角频率记作 $\omega_i = 10000^{-2i/d}$，则 $PE(pos, 2i) = \sin(\omega_i \thinspace pos)$。 $i$ 从 0 走到 $d/2 - 1$， $\omega_i$ 从 1 几何衰减到约 $10^{-4}$：公比 $10000^{-2/d}$ 的几何级数。对应波长 $\lambda_i = 2\pi / \omega_i = 2\pi \cdot 10000^{2i/d}$，从 $2\pi \approx 6.3$ 涨到 $d$ = 512 时的 $2\pi \cdot 10000^{510/512} \approx 6.1 \times 10^{4}$——最短通道每约 6 个 token 振荡一个周期（编码局部次序），最长通道数万 token 才走完一圈（编码全局位置）。几何级数是对数刻度上的等距铺开：多尺度覆盖的最省事取法，同一个思想稍后会在 RoPE 的频率 schedule（§4.5）与 ALiBi 的 slope 级数（§3.2）里各出现一次。sin 与 cos 为什么成对：单用 sin，下一节式 (3) 的线性推移需要在等式右端读到 cos 分量——成对出现才让两角和公式闭环，推移变换保持在线性族内。

### 2.2 相对位移性质： $PE(pos+k)$ 是 $PE(pos)$ 的线性函数

论文 §3.5 选择这一形态的理由是：对任意固定偏移 $k$， $PE(pos+k)$ 可以写成 $PE(pos)$ 的线性函数，「相对位置」可被线性读出。推导就是两角和公式。固定通道 $i$，记 $\alpha = \omega_i \thinspace pos$、 $\beta = \omega_i k$：

$$
\begin{aligned}
\sin(\alpha + \beta) &= \sin\alpha \cos\beta + \cos\alpha \sin\beta \\
\cos(\alpha + \beta) &= \cos\alpha \cos\beta - \sin\alpha \sin\beta
\end{aligned} \qquad (2)
$$

排成矩阵：通道 $i$ 的两维经历一次只由 $k$ 决定的线性变换，

$$
\begin{bmatrix} PE(pos+k, 2i) \\ PE(pos+k, 2i+1) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE(pos, 2i) \\ PE(pos, 2i+1) \end{bmatrix} \qquad (3)
$$

全维写法： $PE(pos+k) = M_k \thinspace PE(pos)$， $M_k$ 是每通道一块的块对角矩阵，与 $pos$ 无关。注意这个 2×2 块正是旋转 $-\omega_i k$ 的矩阵——块对角旋转的雏形在这里已经出场：sinusoidal 的 $M_k$ 与 §4 RoPE 的 $R_m$ 是同一种对象、同一套频率 schedule，差别只在作用位置（stream 里的 PE vs attention 里的 q/k）。RoPE 就是把「PE 内部可被线性推移」升级为「q/k 直接被逐位置旋转」。

推论（PE 内积的相对性）。两个位置的 PE 内积：

$$
\langle PE(m), PE(n) \rangle = \sum_{i} \left[ \sin(\omega_i m)\sin(\omega_i n) + \cos(\omega_i m)\cos(\omega_i n) \right] = \sum_{i} \cos\left(\omega_i (m - n)\right) \qquad (4)
$$

只含 $m - n$。sinusoidal 的相对性藏在 PE 之间的内积里；但它注入在 stream 上，score 是 $\langle W^{Q}(x_m + PE(m)),\ W^{K}(x_n + PE(n)) \rangle$。把这个内积乘开成四项——内容·内容 $\langle W^Q x_m, W^K x_n \rangle$、内容·位置两项 $\langle W^Q x_m, W^K PE(n) \rangle$ 与 $\langle W^Q PE(m), W^K x_n \rangle$、位置·位置 $\langle W^Q PE(m), W^K PE(n) \rangle$——只有末项保有式 (4) 的相对形式，中间两项把绝对位置 $m$、 $n$ 分别与内容耦合，四项加权混合后「只含 $m - n$」不再成立。PE 与内容在投影中交叉相乘，式 (4) 的干净相对性被 $W^{Q}, W^{K}$ 搅浑。RoPE 的出发点就是把同一相对性直接钉在 score 上（§4.1 式 (8)）。

### 2.3 learned absolute：外推 = 未定义

可学绝对位置嵌入（learned absolute position embedding）把位置当 token 处理：一张可训练查找表 $P \in \mathbb{R}^{T_{\max} \times d}$，第 $t$ 行加进第 $t$ 个 token 的 embedding。GPT-2 的 `wpe` 即此形：n_positions = 1024、 $d$ = 768（[Radford et al., 2019](../references.md)，[W]，config 口径）。两个硬后果。其一，外推不是「退化」而是**未定义**：位置 $T_{\max}$ 之外表里没有行，位置 id 越界是非法访问，连函数值都不存在。其二，即便把表加长，新区间的行是随机初始化、从未被训练；且训练动态本身就不均——行 $t$ 只在长度超过 $t$ 的样本上收到梯度，短序列居多的语料里大行号的行训练信号天然稀薄——这和 §2.1 的 sinusoidal（函数在任意整数位置有定义，只是行为漂移）是两个层级的失败。参数账也记一笔：GPT-2 的表是 $1024 \times 768 \approx 0.79\text{M}$（约占总参数 0.6%），小头；若按 Llama-3.1 的窗口造表则是 $131072 \times 4096 \approx 0.54\text{B}$——加性绝对编码在长窗时代光查表就要半 B 参数，这条路自己把自己走死了。

## 3. 相对位置偏置族：T5 分桶与 ALiBi 线性惩罚

这一族不动 q/k/v——内容向量全程不接触位置，位置只决定「谁被注意」、不决定「交出什么」——只在 score 上加一个只依赖相对位置的偏置 $b$：

$$
s_{ij} = \frac{\langle q_i, k_j \rangle}{\sqrt{d_{\mathrm{head}}}} + b(i - j) \qquad (5)
$$

**符号**： $d_{\mathrm{head}}$：head 维（2.4 记法，即 2.3 的 $d_k$）； $b$：相对位置偏置（relative position bias），按头独立。偏置在 softmax 之前以加法进入，与 2.4 式 (1) 的 mask 同处一室、互不干扰：mask 管可见性， $b$ 管可见集合内的位置先验。

### 3.1 T5：log 分桶的查表 bias

T5 把相对位置离散成 32 个桶（bucket）、每桶一个可学标量，共 $32 \times H$ 个参数，原实现跨层共享（[Raffel et al., 2019](../references.md)，arXiv:1910.10683，[A]）。分桶规则（双向版，max_distance = 128）：正负方向各分 16 个桶号；每个方向上，偏移 0–7 各占一桶（精确区），8–128 按对数刻度分入其余 8 桶（log 区），超过 128 一律落末桶。单方向桶号：

$$
\mathrm{bucket}(\Delta) = \begin{cases} \Delta, & \Delta < 8 \\ 8 + \operatorname{floor}\left( \dfrac{\ln(\Delta / 8)}{\ln 16} \times 8 \right), & 8 \le \Delta \le 128 \\ 15, & \Delta > 128 \end{cases} \qquad (6)
$$

**符号**： $\Delta = |i - j|$； $\operatorname{floor}$ 为向下取整；方向位再加 16 区分正负。 $\Delta$ = 128 时对数项恰好取满 8、被 cap 到 15，与 $\Delta > 128$ 同桶。

log 区的桶边界有闭式： $\Delta = 8 \cdot 2^{k/2}$（ $k = 1, \ldots, 8$），即 11.3、16、22.6、32、45.3、64、90.5、128——每个桶比前一个宽约 $\sqrt{2}$ 倍：整数 $\Delta$ 下 $[8, 11]$ 落桶 8、 $[12, 15]$ 落桶 9、 $[16, 22]$ 落桶 10、 $[23, 31]$ 落桶 11、 $[32, 45]$ 落桶 12、 $[46, 63]$ 落桶 13、 $[64, 90]$ 落桶 14、 $[91, 128]$ 落桶 15。

设计读法：近处逐个分辨（0–7 各自一桶），远处对数粗分——位置分辨率随距离衰减是刻意的，「很远」与「更远一点」对多数任务无差别。外推行为也直接读出：距离超过 128 全部落末桶，函数有定义但分辨率归零——模型对超窗距离只剩「远」这一个比特。这是「查表型相对编码」的通用外推形态：不死，但变瞎。

### 3.2 ALiBi：不查表，直接线性惩罚

ALiBi（Attention with Linear Biases）把查表换成一条固定直线（[Press et al., 2021](../references.md)，arXiv:2108.12409，[A]）。causal 设置下 $j \le i$：

$$
s_{ij} = \frac{\langle q_i, k_j \rangle}{\sqrt{d_{\mathrm{head}}}} - m_h \cdot (i - j), \qquad m_h = 2^{-8h/H}, \quad h = 1, \ldots, H \qquad (7)
$$

**符号**： $m_h$：第 $h$ 个头的斜率（slope）； $H$：头数。双向推广把 $(i - j)$ 换成 $|i - j|$。

slope 是公比 $2^{-8/H}$ 的几何级数： $h = 1$ 时 $m_1 = 2^{-8/H}$（斜率最大、惩罚最狠、看得最近）， $h = H$ 时 $m_H = 2^{-8} \approx 0.0039$（惩罚最弱、看得最远）。 $H$ = 8 时 slopes 为 $2^{-1}, 2^{-2}, \ldots, 2^{-8}$； $H$ = 32 时从 $2^{-1/4} \approx 0.84$ 排到 $2^{-8}$，跨 $2^{7.75} \approx 215$ 倍。换个读法：距离每加 1，logit 减 $m_h$ nat——斜率 $m$ 的头「一个 nat 的预算覆盖 $1/m$ 个位置」， $m = 2^{-1}$ 的头 2 个位置就压掉 1 nat， $m = 2^{-8}$ 的头 256 个位置才压 1 nat。头间分工从「只看相邻」铺到「近乎全窗」。与 sinusoidal 的频率几何级数是同一多尺度思想，这里尺度轴是「随距离压分多快」。零新增参数、零查表、零桶边界，函数在任意距离有定义。

逐头看机制。先用 2.3 的平移不变性化简： $-m_h(i - j) = -m_h i + m_h j$，其中 $-m_h i$ 对第 $i$ 行是常数项、被 softmax 原样吃掉（2.3 式 (10)）——真正起作用的只剩 $+m_h j$：行内越近（ $j$ 越大）的 key 加分越多。 $j = i$ 处净惩罚为 0，随距离线性变负——远处 key 的 logit 被等比压低，attention 权重天然近密远疏。对照 2.3 §3 的饱和分析：ALiBi 是有意注入的「近端饱和」，用一条固定直线换掉了学出来的位置结构，可调维度只剩 $H$ 个 slope；RoPE（§4）则让位置以 $d/2$ 个频率乘性进入内积，与内容耦合。

外推机制分析（为什么 train short / test long 能成立）：(a) $b(\Delta) = -m_h \Delta$ 没有查表边界，测试时更大的 $\Delta$ 只是同一直线上的更大自变量——函数形态不变，不像 T5 落末桶、不像 learned absolute 越界；(b) 模型在窗内学到的是「logit ≈ 内容项 − $m_h \Delta$」的校准，外延时内容项的分布没变，惩罚项按同一线性率继续压低远处 key，softmax 归一化后有效感受野自动收缩回训练尺度；(c) 代数字走一遍：最小 slope $2^{-8}$ 的头在 $L$ = 1024 窗内的惩罚跨度是 $1024 \times 2^{-8} = 4$ nat；外推到 4096 时跨度 16 nat，最远 key 被压到相对 $e^{-12}$ 量级——行为平滑过渡到「该头只看近处」，而非数值崩溃。论文实测 1024 训练窗外推 2048+ 的 perplexity，与从头训练长窗的 sinusoidal 基线相当（[A]）。

## 4. RoPE：内积不变性的完整推导

### 4.1 目标形式化：造一个让 score 只含 $m - n$ 的变换

§2.2 末指出了 stream 加性注入的结构性缺陷：相对性被投影搅浑。RoPE（Rotary Position Embedding，旋转位置编码）把要求直接钉在 score 上（[Su et al., 2021](../references.md)，arXiv:2104.09864，[A]，RoFormer）：找一个作用于 q/k、带位置参数 $m$ 的变换 $f$，使

$$
\langle f(q, m),\ f(k, n) \rangle = g(q, k, m - n) \qquad (8)
$$

**符号**： $q, k \in \mathbb{R}^{d}$：某头某位置的 query / key 向量（已投影）； $g$：只依赖内容与相对位移的函数。记法复用提示：§2 的 $d$ 是 stream 宽度（原文 512），本节起 $d$ 指被旋转向量的维数——单头口径，锚点家族为 128（DeepSeek-V3 只旋其中 64，见 §4.7）。式 (8) 是「score 只含 $m-n$」的精确写法，也是「语言的规律与出现位置无关」这一先验的形式化：「主语谓语相邻」应当无论出现在文首还是文尾都同样被识别。本节证明 $f(q, m) = R_m q$（逐位置乘一个旋转矩阵）以显式形式解出式 (8)，且 $g$ 可以逐通道写出。整条构造链不含任何可学参数——位置先验由频率 schedule 一个超参数族（§4.5 的 base）完全决定。

### 4.2 2D 旋转引理

2 维旋转矩阵：

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \qquad (9)
$$

**引理 (i)（正交性）**： $R(\theta)^{\top} = R(-\theta)$。由 $\cos(-\theta) = \cos\theta$、 $\sin(-\theta) = -\sin\theta$：

$$
R(-\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} = R(\theta)^{\top}
$$

且 $R(\theta)^{\top} R(\theta) = I$（对角元 $\cos^{2}\theta + \sin^{2}\theta = 1$，非对角元相消）——旋转保范、保内积。

**引理 (ii)（群性质）**： $R(\alpha) R(\beta) = R(\alpha + \beta)$。乘开左上角元： $\cos\alpha\cos\beta - \sin\alpha\sin\beta = \cos(\alpha + \beta)$；右上角元： $-\cos\alpha\sin\beta - \sin\alpha\cos\beta = -\sin(\alpha + \beta)$；另两元同理——全部归为式 (2) 的两角和恒等式。引理 (i)(ii) 合起来说： $\lbrace R(\theta) : \theta \in \mathbb{R}\rbrace$ 在矩阵乘法下构成与 $(\mathbb{R}, +)$ 同态的群（角度相加 = 旋转复合）， $R(0) = I$、 $R(\theta)^{-1} = R(\theta)^{\top}$。RoPE 的全部相对性都是这条同态的推论。

**引理 (iii)（内积的相对性）**：对任意 $q, k \in \mathbb{R}^{2}$ 与实数 $a, b$，

$$
\langle R(a)\thinspace q,\ R(b)\thinspace k \rangle = (R(a)q)^{\top} R(b)\thinspace k = q^{\top} R(a)^{\top} R(b)\thinspace k = q^{\top} R(-a)\thinspace R(b)\thinspace k = q^{\top} R(b - a)\thinspace k = \langle q,\ R(b-a)\thinspace k \rangle \qquad (10)
$$

逐步点名：第一个等号是内积的矩阵写法；第二个等号是转置规则 $(AB)^{\top} = B^{\top} A^{\top}$；第三个等号是引理 (i)；第四个等号是引理 (ii)；末个等号回到内积记号。 $a, b$ 只以差 $b - a$ 出现——引理 (iii) 就是式 (8) 在 2 维的解：取 $f(q, m) = R(m\theta)\thinspace q$，则 $\langle f(q,m), f(k,n) \rangle = q^{\top} R\left((n-m)\theta\right) k$。

代数字验证： $q = (1, 2)$、 $k = (3, 1)$、 $a = 1.5$、 $b = 3.5$。左侧 $\langle R(1.5)q, R(3.5)k \rangle \approx \langle (-1.924, 1.139), (-2.459, -1.989) \rangle \approx 2.47$；右侧 $q^{\top} R(2)\thinspace k \approx (1, 2) \cdot (-2.158, 2.312) \approx 2.47$。两侧相合；而原始内积 $\langle q, k \rangle = 5$——旋转确实改变了内积，但改变量只由 $b - a = 2$ 决定。

把 $q^{\top} R(\varphi)\thinspace k$ 展开成坐标备用。 $R(\varphi) k = (\cos\varphi \cdot k_1 - \sin\varphi \cdot k_2,\ \sin\varphi \cdot k_1 + \cos\varphi \cdot k_2)$，与 $(q_1, q_2)$ 点乘、按 $\cos\varphi$ 与 $\sin\varphi$ 合并：

$$
q^{\top} R(\varphi)\thinspace k = (q_1 k_1 + q_2 k_2) \cos\varphi + (q_2 k_1 - q_1 k_2) \sin\varphi \qquad (11)
$$

**符号**：下标为 2 维块内坐标。两个系数各有一个读法： $q_1 k_1 + q_2 k_2$ 是块内同址内积（内容相似度）， $q_2 k_1 - q_1 k_2$ 是块内交叉内积（2D 叉积，方向差）。

### 4.3 块对角推广与主命题

$d$ 维（ $d$ 偶数）的做法：切成 $d/2$ 个 2 维块，第 $j$ 块以角速度 $\theta_j$ 旋转角度 $m\theta_j$：

$$
R_m = \operatorname{diag}\left( R(m\theta_0),\ R(m\theta_1),\ \ldots,\ R(m\theta_{d/2-1}) \right) \in \mathbb{R}^{d \times d} \qquad (12)
$$

**符号**： $\operatorname{diag}$ 为块对角； $\theta_0 > \theta_1 > \cdots > \theta_{d/2-1}$ 的取法在 §4.5。三条引理逐块继承： $R_m^{\top} R_m = I$（每块正交）； $R_m^{\top} = R_{-m}$； $R_a R_b = R_{a+b}$（每块群性质，块间不串）。

```
R_m 结构（d = 8 示意，θ_0 > θ_1 > θ_2 > θ_3）          频率轴（log 刻度）
┌                                              ┐      j = 0 ──────────────► j = d/2 − 1
│  cos mθ₀  −sin mθ₀                           │      λ₀ = 2π             λ_max = 2π·base^(1−2/d)
│  sin mθ₀   cos mθ₀                           │
│                cos mθ₁  −sin mθ₁             │      快通道（几 token 一圈）  慢通道（数万 token 一圈）
│                sin mθ₁   cos mθ₁             │      局部次序                全局位置
│                              cos mθ₂  −sin mθ₂│
│                              sin mθ₂   cos mθ₂│      每块 = 一个 2D 旋转，d/2 块 = d/2 个频率通道
│                                            ... │
└                                              ┘
```

**主命题（RoPE 内积不变性）**。对任意 $q, k \in \mathbb{R}^{d}$ 与位置 $m, n$：

$$
\langle R_m\thinspace q,\ R_n\thinspace k \rangle = \langle q,\ R_{n-m}\thinspace k \rangle = \sum_{j=0}^{d/2-1} \left[ \left( q_{2j} k_{2j} + q_{2j+1} k_{2j+1} \right) \cos\left( (n-m)\theta_j \right) + \left( q_{2j+1} k_{2j} - q_{2j} k_{2j+1} \right) \sin\left( (n-m)\theta_j \right) \right] \qquad (13)
$$

**证明**。第一等式： $\langle R_m q, R_n k \rangle = q^{\top} R_m^{\top} R_n\thinspace k = q^{\top} R_{-m} R_n\thinspace k = q^{\top} R_{n-m}\thinspace k$，逐步用块对角继承的正交性与群性质。第二等式： $R_{n-m}$ 块对角，内积按块分解为各 2 维块内积之和；第 $j$ 块上是 $\langle q_j, R\left((n-m)\theta_j\right) k_j \rangle$，以 $\varphi = (n-m)\theta_j$ 代入式 (11) 得该块贡献；对 $j$ 求和即式 (13)。∎

式 (13) 右端只含 $n - m$——式 (8) 达成，且 $g$ 是显式的：内容提供每块两个系数（同址内积与交叉内积），位置以 $\cos/\sin\left((n-m)\theta_j\right)$ 逐频率调制。一个内积变成 $d/2$ 个频率通道的加权和；score 对绝对位置平移不变——全体位置同加常数 $c$， $m - n$ 不变，score 逐点不变。

实现注记： $R_m$ 是 $d \times d$ 但只有 $2d$ 个非零元，任何实现都不物化它——预存 cos/sin 两张表、逐对套用式 (9) 的坐标公式即可（§4.8），时间与内存都从 $O(d^{2})$ 降到 $O(d)$。

应用位置与接缝： $q_m = R_m (W^{Q} u_m)$、 $k_n = R_n (W^{K} u_n)$， $v$ 不旋转；score $= \langle q_m, k_n \rangle / \sqrt{d_{\mathrm{head}}}$，mask 照旧加在 score 上（2.4 式 (1)）。 $v$ 不旋转的原因在结构里：位置只需要决定「谁被注意」，而 attention 权重已携带全部相对位置结构； $v$ 是被交付的内容，保持位置盲，内容读出才不随平移改变。两条兼容性注记：其一， $R_m$ 正交 ⟹ $\lVert R_m q \rVert = \lVert q \rVert$——旋转只改方向不改范数，2.3 §3 的 $\mathrm{Var}(q \cdot k) = d_k$ 方差账与 $\sqrt{d_k}$ 缩放结论原样成立，RoPE 不重排 score 的量级预算；其二，旋转在写 KV cache 之前完成——cache 里存的是已旋转的 $k_n$（2.4 §4.2 的不可变性不受影响），decode 第 $t$ 步的 $q$ 必须按全局绝对位置 $t$ 旋转，position_ids 不能按 decode 窗内局部计数。

### 4.4 复数视角：旋转 = 乘 $e^{im\theta}$

把第 $j$ 块的 2 维向量看成复数 $`\tilde{q}_j = q_{2j} + i\thinspace q_{2j+1}`$。2D 旋转恰是复乘 $e^{i\varphi}$：

$$
e^{i\varphi}\thinspace \tilde{q}_j = (\cos\varphi + i\sin\varphi)(q_{2j} + i\thinspace q_{2j+1}) = \left( q_{2j}\cos\varphi - q_{2j+1}\sin\varphi \right) + i \left( q_{2j}\sin\varphi + q_{2j+1}\cos\varphi \right) \qquad (14)
$$

实部、虚部逐一对上 $R(\varphi)\thinspace q_j$ 的两个坐标。于是 RoPE 的复数写法是逐通道乘相位： $`\tilde{q}_j \mapsto e^{i m \theta_j} \tilde{q}_j`$——工程实现里预存的 cos/sin 两张表，就是复数 $e^{i m \theta_j}$ 的实部与虚部分开存放。实内积式 (13) 则是 Hermitian 内积的实部：

$$
\langle R_m\thinspace q,\ R_n\thinspace k \rangle = \operatorname{Re} \sum_{j=0}^{d/2-1} \left( e^{i m \theta_j} \tilde{q}_j \right) \overline{\left( e^{i n \theta_j} \tilde{k}_j \right)} = \operatorname{Re} \sum_{j=0}^{d/2-1} \tilde{q}_j\thinspace \overline{\tilde{k}_j}\thinspace e^{i (m - n) \theta_j} \qquad (15)
$$

逐步：第一步逐块代入复数写法；第二步共轭把 $e^{i n \theta_j}$ 翻成 $e^{-i n \theta_j}$，与前一因子合并为 $e^{i(m-n)\theta_j}$。等价性验证： $`\tilde{q}_j \overline{\tilde{k}_j} = (q_{2j} k_{2j} + q_{2j+1} k_{2j+1}) + i (q_{2j+1} k_{2j} - q_{2j} k_{2j+1})`$，记为 $A_j + i B_j$； $\operatorname{Re}\left[(A_j + i B_j) e^{i(m-n)\theta_j}\right] = A_j \cos\left((m-n)\theta_j\right) - B_j \sin\left((m-n)\theta_j\right) = A_j \cos\left((n-m)\theta_j\right) + B_j \sin\left((n-m)\theta_j\right)$，逐项等于式 (13) 第 $j$ 块。∎

复数视角是波长账（§4.5）与外推缩放族（4.6）的公共语言：RoPE 就是每个频率通道把 q、k 之间的相对相位转过 $(m - n)\theta_j$。

### 4.5 频率 schedule 与波长账

$\theta_j$ 沿用 sinusoidal 的几何级数（与式 (1) 的 $\omega_i$ 同一 schedule）：

$$
\theta_j = base^{-2j/d}, \qquad \lambda_j = \frac{2\pi}{\theta_j} = 2\pi \cdot base^{2j/d}, \qquad j = 0, \ldots, d/2 - 1 \qquad (16)
$$

**符号**： $base$：config 字段 rope_theta； $\lambda_j$：第 $j$ 通道的波长（wavelength）——位置每走 $\lambda_j$ 个 token，该通道恰好转满一圈。 $\lambda_j$ 在对数刻度上等距铺满 $[2\pi,\ 2\pi \cdot base^{1 - 2/d}]$。快通道（ $j$ 小）编码局部次序，慢通道（ $j$ 大）编码全局位置； $d/2$ 个通道合起来相当于位置的多尺度展开，与 §2.1 同构。

真实 config 的波长账（前三家 $d_{\mathrm{head}}$ = 128 → 64 通道；DeepSeek-V3 只旋 64 维 → 32 通道，见 §4.7）：

| 模型 | base（rope_theta） | 通道数 | $\lambda_0$ | $\lambda_{\max} = 2\pi \cdot base^{1 - 2/d}$ | 原生窗口 |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 500000 | 64 | $2\pi$ | $\approx 2.6 \times 10^{6}$ | 131072 |
| OLMo-2-7B | 500000 | 64 | $2\pi$ | $\approx 2.6 \times 10^{6}$ | 4096 |
| Qwen3-8B | 1000000 | 64 | $2\pi$ | $\approx 5.1 \times 10^{6}$ | 40960 |
| DeepSeek-V3 | 10000 | 32 | $2\pi$ | $2\pi \cdot 10000^{62/64} \approx 4.7 \times 10^{4}$ | 4096 + YaRN ×40 |

（[Llama-3.1-8B config](../references.md)、[OLMo-2-7B config](../references.md)、[Qwen3-8B config](../references.md)、[DeepSeek-V3 config](../references.md)，均 [A]； $\lambda_{\max}$ 由 config 数值按式 (16) 自算。）

读表的一个角度是 $\lambda_{\max}$ 与原生窗口的比值：Llama-3.1 约 20 倍、Qwen3-8B 约 124 倍、OLMo-2 约 625 倍、DeepSeek-V3 约 11 倍（配 YaRN 扩展）——比值越大，慢通道在窗内覆盖的角度弧越短，越依赖「不出窗」的前提；这不是优劣排名，是各家对「窗口目标 vs 频率分辨率」权衡的不同取点。

沿通道轴走一遍具体数字（base = 10000、 $d$ = 128， $\lambda_j = 2\pi \cdot 10^{j/16}$）： $j$ = 0 时 $\lambda_0 = 2\pi \approx 6.3$， $j$ = 16 时约 63， $j$ = 32 时约 628， $j$ = 48 时约 6300， $j$ = 63 时约 $5.4 \times 10^{4}$——每 16 个通道波长乘 10。

**周期覆盖账**。通道 $j$ 在窗口 $W$ 内转满至少一圈 ⟺ $\lambda_j \le W$。注意「窗内不满圈」本身不是缺陷：慢通道在窗内只走一段弧，行为接近「全局位置的单调坐标」，窗内用它分辨大尺度位置反而干净；缺陷只在越窗那一刻触发。取上述对照组配 $W$ = 4096： $\lambda_j > 4096$ ⟺ $j \ge 46$——64 个通道里 18 个（28%）在窗内从未转满一圈，它们见过的角度只是 $[0,\ W\theta_j]$ 的一段弧。弧有多短可以直接算：最慢通道 $\theta_{63} = 10000^{-126/128} \approx 1.15 \times 10^{-4}$， $W\theta_{63} \approx 0.47$ rad——训练全程这个通道只在约 27° 的扇区里取值。位置一旦超出窗口，这些通道的角度进入训练中从未出现的区间——这就是 §6 外推失败的直接机制，而且失败集中在慢通道。抬 base 的账同样可算：Llama-3.1 的 base = 500000 把 $\lambda_{\max}$ 顶到约 20 倍窗口，窗口内慢通道的角度弧更短（最慢通道 $\theta_{63} \approx 2.5 \times 10^{-6}$、131072 窗内弧长约 0.32 rad）、相邻位置的相对角度差更小，长程分辨更细；对照侧 OLMo-2 用同一 base = 500000 配 4096 窗口： $\lambda_j > 4096$ ⟺ $j \ge 32$，即 64 个通道里一半在窗内转不满一圈——超窗使用这类配置，半数通道的角度直接出训练域。base 是「把频率轴对齐到目标窗口」的旋钮；DeepSeek-V3 走的不是抬 base 而是分频缩放（§6）。

### 4.6 远程衰减性质

直觉预期： $|m - n|$ 大时位置项应衰减——近处 token 默认更相关。RoPE 论文（§3.4.3）给出的论证：固定内容，式 (15) 的和 $\sum_j h_j\thinspace e^{i\Delta\theta_j}$（ $`h_j = \tilde{q}_j \overline{\tilde{k}_j}`$、 $\Delta = m - n$）随 $\Delta$ 增大时各通道相位以不同角速度散开、相互抵消； $\theta_j$ 几何衰减使通道角速度跨多个量级，去相（dephasing）随 $\Delta$ 迅速发生。一个端点值把上界钉死（以下取 $h_j$ 模长为 1 的同相位极端情形）： $\Delta = 0$ 时所有相位同相、 $\left| \sum_j e^{i\Delta\theta_j} \right|$ 取最大值 $d/2$； $\Delta$ 增大后各通道相位在对数频率轴上散开，和式进入振荡下行。论文用 Abel 分部求和给出该和的上界因子，并绘制了该界随 $\Delta$ 衰减的确定性曲线（非实测曲线）：即使在这一极端情形下， $\left| \sum_j e^{i\Delta\theta_j} \right|$ 也从 $d/2$ 随 $\Delta$ 趋势性衰减。

两个注记。其一，这是典型/平均行为，不是对任意 $q, k$ 的定理—— $h_j$ 由训练塑造，特定头的特定通道可以学会抵抗衰减；去相给的是先验形状，不是硬约束。且衰减不单调：振荡使和式在更大 $\Delta$ 处仍会取到局部峰（revival），只是峰值趋势性走低。其二，与 ALiBi 对照：ALiBi 的衰减是显式钉死的直线（式 (7)），RoPE 的衰减是去相涌现的振荡衰减——带涟漪的近密远疏。

### 4.7 partial RoPE：DeepSeek-V3 的 64/192 账

RoPE 不必旋满整个 head 维。DeepSeek-V3 的 qk head 维 = 192 = 128（qk_nope_head_dim，不旋转）+ 64（qk_rope_head_dim，旋转）（[DeepSeek-V3 config](../references.md)，[A]；2.3 §5 已用它做过 head_dim ≠ hidden/heads 的反例）。内积按维块可加分解：

$$
\langle q_m, k_n \rangle = \underbrace{\langle q_m^{\mathrm{nope}},\ k_n^{\mathrm{nope}} \rangle}_{\text{128 dims, position-blind}} + \underbrace{\langle R_m\thinspace q_m^{\mathrm{rope}},\ R_n\thinspace k_n^{\mathrm{rope}} \rangle}_{\text{64 dims, eq. (13), } m-n\ \text{only}} \qquad (17)
$$

**符号**： $q^{\mathrm{nope}}, k^{\mathrm{nope}} \in \mathbb{R}^{128}$：不旋转部分（no rotary）； $q^{\mathrm{rope}}, k^{\mathrm{rope}} \in \mathbb{R}^{64}$：旋转部分。位置信息由 32 个频率通道承担，内容匹配由 128 维纯内积承担——partial（decoupled）RoPE。

账分三笔。(a) 频率带宽基本不亏：通道数从 64 减到 32，但 $\lambda_{\max} = 2\pi \cdot base^{1 - 2/d_{\mathrm{rope}}}$ 主要由 base 决定，base = 10000 下从 $5.4 \times 10^{4}$ 到 $4.7 \times 10^{4}$，谱覆盖只窄了一线；少的是中间频率的采样密度。相对地，位置信号的承载被集中到 1/3 的维数上——score 的 192 维里只有 64 维对 $m - n$ 敏感，位置分辨的精度预算全押在这 32 个通道上。(b) nope 部分不经过任何位置变换， $k^{\mathrm{nope}}$ 可以被低秩压缩、压缩矩阵还能被吸收进 W_q 一侧——这是 4.1 MLA 的 W_UK absorption 成立的前提。障碍可以写清楚：若全维旋转，score 含 $\langle R_m q,\ R_n C c_n \rangle$ 形的项（ $C$ 为压缩矩阵、 $c_n$ 为压缩态），要把 $C$ 吸收进 q 侧必须让 $R_n$ 与 $C$ 换位，而 $R_n$ 逐块角度不同、 $C$ 是全维混合——一般有 $R_n C \neq C R_n$，换位即死。decoupled 是 absorption 的必要条件，不是顺手优化。DeepSeek-V3 把这条结构选择与 MLA 一起设计（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）。(c) 窗口账：DeepSeek-V3 原生窗 4096、rope_theta = 10000，配 §6 的 YaRN 块（factor 40）扩到 163840——base 与旋转维数是缩放方案的输入参数，不是独立旋钮。对照组：Llama-3.1 / Qwen3 / OLMo-2 全族旋满 128 维（各 config 无 partial 字段，默认全旋）——partial 是为 KV 压缩让路的结构选择，不是免费午餐。

### 4.8 两种等价实现：half-split 与 interleaved

式 (13) 的「2 维块」有两种维度配对约定：相邻对 $(2j, 2j+1)$（interleaved，RoFormer 原文一系）与首尾对 $(j, j + d/2)$（half-split，HF transformers 的 Llama 实现）。同一数学、两种维度排列；权重矩阵的行序跟着配对约定走，混用即错。教学实现（教学用，非优化实现）：

```python
import torch

def rope_tables(T: int, d: int, base: float):            # d: 旋转维度（偶数）
    inv_freq = base ** (-torch.arange(0, d, 2).float() / d)  # [d/2]：θ_j = base^(-2j/d)，式 (16)
    t = torch.arange(T).float()
    freqs = torch.outer(t, inv_freq)                           # [T, d/2]：m·θ_j
    return freqs.cos(), freqs.sin()                            # 各 [T, d/2]

def rope_interleaved(x, cos, sin):               # x: [B, H, T, d]，配对 (2j, 2j+1)
    x_ = x.reshape(*x.shape[:-1], -1, 2)                       # [B, H, T, d/2, 2]
    x1, x2 = x_[..., 0], x_[..., 1]                            # 各 [B, H, T, d/2]
    cos, sin = cos[None, None], sin[None, None]                # 广播 [1, 1, T, d/2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)                                     # 式 (9) 逐块作用

def rotate_half(x):                              # half-split：配对 (j, j+d/2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope_half_split(x, cos, sin):
    cos = torch.cat([cos, cos], dim=-1)[None, None]            # [1, 1, T, d]
    sin = torch.cat([sin, sin], dim=-1)[None, None]
    return x * cos + rotate_half(x) * sin
```

等价性：存在维度排列 $\sigma$（把相邻对映到首尾对）使两实现逐通道相同——half-split 的第 $j$ 对通道 $(x_j, x_{j+d/2})$ 上，`x * cos + rotate_half(x) * sin` 的第 $j$ 元为 $x_j \cos - x_{j+d/2} \sin$、第 $j + d/2$ 元为 $x_j \sin + x_{j+d/2} \cos$，正是式 (9)。数值断言留作练习 4。官方实现走读：transformers repo `src/transformers/models/llama/modeling_llama.py` 的 `rotate_half` 与 `apply_rotary_pos_emb` 即 half-split 形，cos/sin 由 `LlamaRotaryEmbedding` 内的 inv_freq 与 position_ids 逐前向外积现算（[HuggingFace, modeling_llama.py](../references.md)，[A]）。partial RoPE 的落地只差一步：对 x 的 rope 维切片过 `apply_rotary_pos_emb`、nope 维原样拼接。

## 5. NoPE：causal mask 自带的隐式位置信号

一个不注入任何位置编码的 decoder-only Transformer 并非位置盲——2.3 §6 的等变命题以「无 mask」为前提，而 causal mask（2.4 式 (3)）已经把位置锚进计算图：第 $t$ 行恰好看见 $t$ 个 key。换句话说，带 causal mask 的 attention 对一般置换本就不再等变——mask 自身就是位置敏感对象，等变命题的前提条件已被架构自己破坏。机制展开：设各位置 value 近似同分布（均值 $\mu$、方差 $\sigma^{2}$），均匀 attention 下位置 $t$ 的输出是 $t$ 个 value 的均值，其方差 $\sigma^{2} / t$ 随 $t$ 单调衰减——「输出的尺度与集中度」本身就是 $t$ 的可读签名，行归一化的分母里也明着含 $t$。第一层即可学出「均分 attention、从响应幅度读出 $1/t$」式的计数电路，位置信息沿层逐层加工。Kazemnejad et al. 在 decoder-only 设置下系统扫描了 NoPE 与各显式编码（[Kazemnejad et al., 2023](../references.md)，arXiv:2305.19466，[B]）：NoPE 训练窗内表现与显式方案相当，若干长度泛化设置中不落下风——causal 结构自带的位置信号足以支撑相当程度的位置行为。

边界同样清楚。其一，双向（encoder）场景没有 causal mask，NoPE 不适用——该结论只属于 decoder-only。其二，计数信号是「长度」而非「内容寻址的相对位移」：表达「向左 3 位」这类量需网络从计数特征里二次构造，RoPE 则把它免费钉在 score 上（式 (13)）。其三，该论文无官方代码（[B]），结论的外推范围以其实验设置为准。现实坐标：本章四个锚点 config（Llama-3.1 / OLMo-2 / Qwen3 / DeepSeek-V3）全部带 rope_theta 字段、无一家走 NoPE——NoPE 的定位是诊断性对照组：它标出「架构白送的位置信号」这条下限，显式编码的收益都应在它之上衡量。

## 6. RoPE 的外推失败：预告（→ 4.6）

§4.5 的周期覆盖账给出失败机制：窗口 $W$ 内，慢通道（ $\lambda_j > W$）只见过角度区间 $[0, W\theta_j]$；推理位置越过 $W$ 后，这些通道的 $\cos/\sin$ 取值是网络从未见过的输入，score 分布整体漂移，perplexity 在窗界处断崖式上升。先把一个常见误读排掉： $\cos/\sin$ 在任意角度都有定义、数值上毫无障碍——失败不是数值问题，是分布外问题：坏掉的是「网络训练时没见过这个区间的角度输入」这一事实。失败集中在慢通道——快通道已转过多圈、角度分布是满的——所以超窗时的退化不是均匀的：管全局位置的慢通道先坏、远程依赖先断，近程行为相对保持。修复方案因此都围绕频率轴做文章。最直接的修法是位置插值（Position Interpolation，PI）：目标窗 $W' = sW$ 时把所有角度除以倍率 $s$，即 $\theta_j \to \theta_j / s$——慢通道角度被压回训练域，但快通道的相邻位置相对角度差也同比例缩小 $s$ 倍，局部位置分辨被摊薄。NTK 与 YaRN 的出发点正是这个副作用：按频段分治，高频通道少动（保局部分辨）、低频通道多压（救外推）。DeepSeek-V3 的 config 就是 YaRN 字段的标准实例（[DeepSeek-V3 config](../references.md)，[A]）：

```
rope_scaling = { type: "yarn", factor: 40.0, beta_fast: 32, beta_slow: 1,
                 mscale: 1.0, mscale_all_dim: 1.0,
                 original_max_position_embeddings: 4096 }
```

账面对照： $4096 \times 40 = 163840$，恰等于 config 的 max_position_embeddings——factor 即窗口倍率；beta_fast / beta_slow 是分频边界（哪些通道算高频、哪些算低频），mscale 是缩放后的 logit 温度补偿。这组字段的推导（为什么按频率分治、factor 如何进入 $\theta_j$）连同 PI / NTK 的统一形式化在 4.6 展开；本节只登记接口——**RoPE 外推失败 = 慢通道角度出训练域**。另一个对照实例：Qwen2.5-7B-Instruct-1M 的 config 无 rope_scaling 字段，只把 rope_theta 抬到 $10^{7}$ 配 max_position_embeddings = 1010000（W1 research pass 实测，见 docs/research/w1-sources.md）——纯抬 base 与 YaRN 分频缩放是两条路线，对照也在 4.6。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 必要性 | 2.3 §6 置换等变推论 | 位置信息必须外部注入；三维自由度：注入点 / 参照系 / 外推 |
| sinusoidal | $PE(pos, 2i) = \sin(pos / 10000^{2i/d})$，式 (1) | 频率几何级数； $PE(pos+k) = M_k PE(pos)$（式 (3)）； $\langle PE(m), PE(n) \rangle = \sum_i \cos\omega_i(m-n)$（式 (4)） |
| learned absolute | 查表 $P \in \mathbb{R}^{T_{\max} \times d}$ | 外推 = 索引越界，未定义；GPT-2 $1024 \times 768$ [W] |
| T5 bias | $b(i-j)$ 查表，32 桶 log 分桶，式 (6) | 0–7 精确、8–128 对数、 $>128$ 落末桶；跨层共享 |
| ALiBi | $-m_h (i - j)$， $m_h = 2^{-8h/H}$，式 (7) | 零参数直线；外推 = 同一直线延自变量，感受野自动收缩 |
| RoPE 主命题 | $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$，式 (13) | 2D 旋转（引理 i–iii）→ 块对角（式 (12)）→ 复数相位（式 (15)）；score 平移不变 |
| 频率与波长 | $\theta_j = base^{-2j/d}$、 $\lambda_j = 2\pi \cdot base^{2j/d}$，式 (16) | Llama-3.1 base = 5×10⁵（ $\lambda_{\max} \approx 2.6 \times 10^{6}$）、Qwen3 10⁶、DeepSeek-V3 10⁴（32 通道） |
| 周期覆盖账 | $\lambda_j > W$ ⟺ 通道 $j$ 窗内转不满一圈 | base 10⁴ / $d$ 128 / $W$ 4096：18/64 通道不满圈 → 外推失败机制 |
| 远程衰减 | $\sum_j h_j e^{i\Delta\theta_j}$ 去相 | 平均行为非定理；对照 ALiBi 的显式直线 |
| partial RoPE | 式 (17)：128 nope + 64 rope | DeepSeek-V3 [A]；nope 部分使 MLA 的 W_UK 吸收可行（→ 4.1） |
| 两实现 | half-split（HF Llama `rotate_half`）/ interleaved（RoFormer 系） | 差一个维度排列；权重行序跟随约定 |
| NoPE | 无显式编码 | causal mask 的计数级信号；decoder-only 限定；[B] |
| 外推失败 | 慢通道角度出训练域 | PI / NTK / YaRN → 4.6；DeepSeek-V3 rope_scaling 块：factor 40 × 4096 = 163840 |

## 练习

1.（推导）补全式 (4) 的求和恒等式，并证明 sinusoidal 下 $\lVert PE(pos) \rVert^{2} = d/2$ 为常数（与 $pos$ 无关）——即 PE 落在半径 $\sqrt{d/2}$ 的球面上。
2.（推导）从式 (8) 反推 RoPE 的形式：设 $f(q, m)$ 对 $q$ 线性且保范。(a) 取 $m = n$ 推出 $f(\cdot, m)$ 必须正交；(b) 记 $R_m = f(\cdot, m) f(\cdot, 0)^{-1}$，证明 $R_0 = I$、 $R_m^{\top} = R_{-m}$、 $R_a R_b = R_{a+b}$，即 $\lbrace R_m\rbrace$ 是一元正交群；(c) 由此说明「按位置旋转」（块对角 2D 旋转）是式 (8) 的自然解族。这就是 RoPE 论文的推导路径。
3.（计算）base = 10000、 $d$ = 128、窗口 $W$ = 2048：多少通道在窗内转不满一圈？把 base 抬到 50000 重算（解 $2\pi \cdot base^{j/64} > W$ 的 $j$ 范围）。再算 Qwen3-8B（base = $10^{6}$、 $W$ = 40960）窗内不满圈的通道数，并解释「抬 base 对齐窗口」的含义。
4.（代码）(a) 验证 §4.8 两实现等价：随机 $x \in [B, H, T, d]$，构造维度排列 $\sigma$，断言 `rope_half_split(perm(x)) == perm(rope_interleaved(x))` 到浮点精度。(b) 验证式 (13) 的平移不变性：随机 $q, k$（ $d$ = 128、base = 10000），扫 $(m, n) \in [0, 64)^{2}$ 网格，检查 score 矩阵的 Toeplitz 性—— $s_{m,n}$ 只依赖 $m - n$（每条对角线上数值一致到 $10^{-5}$）。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——sinusoidal 式 (1) 与相对位移动机出自 §3.5
- [Su et al., 2021](../references.md)：RoFormer: Enhanced Transformer with Rotary Position Embedding，arXiv:2104.09864，[A]（key: rope）——式 (8)(13)(15)(16) 与 §4.6 远程衰减论证
- [Press et al., 2021](../references.md)：Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation，arXiv:2108.12409，[A]（key: alibi）——式 (7) 与外推机制
- [Raffel et al., 2019](../references.md)：Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer，arXiv:1910.10683，[A]（key: t5）——式 (6) 分桶 bias
- [Kazemnejad et al., 2023](../references.md)：The Impact of Positional Encoding on Length Generalization in Transformers，arXiv:2305.19466，[B]（key: nope）——§5 NoPE 证据
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——decoupled RoPE 与 MLA 的协同设计
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——rope_theta = 500000、窗口 131072
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——qk_rope_head_dim = 64 / qk_nope_head_dim = 128、rope_theta = 10000、rope_scaling 块
- [Qwen, Qwen3-8B config.json](../references.md)：[A]（key: qwen3-8b-config）——rope_theta = 1000000、窗口 40960
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——rope_theta = 500000、窗口 4096 对照组
- [Radford et al., 2019](../references.md)：[W]（key: gpt2）——learned absolute：n_positions = 1024
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——`rotate_half` / `apply_rotary_pos_emb`，half-split 实现
