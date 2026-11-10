# 2.1 架构总览与设计约束（Architecture Overview and Design Constraints）

> 本章把现代 decoder-only Transformer 定成三件全书复用的东西：residual stream 递推式「$x$ ← $x + f(\mathrm{LN}(x))$」、一次 forward 的完整 shape 走读（$[B, T, H, D]$ 记法）、参数量 $12Ld^{2}$ 标尺——然后用 Transformer 原文 Table 1 的三指标（每层计算量 / 串行操作数 / 最大路径长度）回答「为什么是 attention 而不是 recurrence」| 前置：1.1（$f_\theta$、$x_{<t}$、$V$、$d$），1.2（$2mnk$ 账、$C = 6ND$ 的 $N$ 口径） | 锚点：Llama-3.1-8B config [A]、OLMo-2-7B config [A]；Vaswani et al. [A]、Llama 3 herd [A]、OLMo 2 [A]

## 1. 一张 config.json 确定一台机器

Llama-3.1-8B 的 config.json 有效字段约十个：$L=32$、$d=4096$、$H_q=32$、$H_{kv}=8$、$d_{\mathrm{ff}}=14336$、$V=128256$ 等（[Llama-3.1-8B config](../references.md)，[A]）。这十个整数唯一决定了 $8.03 \times 10^{9}$ 个参数的组织方式与一次 forward 的全部张量形状。本章结束时，每个字段在结构图里的位置都是确定的（第 8 节逐字段映射）。

先把全景画出来。decoder-only 架构是一次 forward 中依次执行的 tensor 变换组合：

```
token ids  x_1..x_T             [B, T]         ← 整数下标（1.1 第 1 节）
    │
    ▼  查表（gather，无乘加，1.2 式 (2) 的口径）
输入 embedding  E ∈ R^{V×d}
    │
    ▼
x⁽⁰⁾  [B, T, d]  ─── residual stream 起点
    │
    │   ┌── decoder layer ℓ（同一结构重复 L 次）──────────┐
    │   │  x ← x + Attn_ℓ(RMSNorm₁(x))    跨位置混合       │
    │   │  x ← x + FFN_ℓ(RMSNorm₂(x))     逐位置变换       │
    │   └───────────────────────────────────────────────┘
    ▼
final RMSNorm  →  x⁽ᶠ⁾  [B, T, d]
    │
    ▼  z = W_U · x，W_U ∈ R^{V×d}（输出头，1.1 式 (2) 的 W）
logits  z  [B, T, V]
    │
    ▼  softmax（逐位置）
q_θ(· | x_{<t}) ∈ Δ^{V-1}     ← 1.1 式 (1) 的值域
```

读这张图只需要一个结构性事实：**全部运算分两类——attention 是唯一跨位置（沿 $T$ 维）混合信息的算子，其余一切（embedding 查表、RMSNorm、FFN、输出头、残差加）都逐位置独立**。这个二分法是第 5 节设计约束分析的枢纽，也是 2.4（causality 只需在 attention 的 mask 上落实）、2.6（FFN 可逐 token 理解）、4.1（KV cache 只缓存 attention 中间量）共同的前提。图中 tie_word_embeddings = false 的模型里 $E$ 与 $W_U$ 是独立的两份 $V \times d$ 参数（Llama-3.1-8B、OLMo-2-7B 均如此），weight tying 的取舍见 2.2。

参数在图里的住处也值得先点明：全部参数只住在三处——$E$ 、每层两个子层内部的投影矩阵、$W_U$ 。RMSNorm 每层只有两个 $d$ 维增益向量（$O(Ld)$ ，低阶项），残差加没有参数。换句话说，结构图里所有带箭头的变换中只有 matmul 携带参数——这与 1.2 第 3 节「每参数每 token 一次 FMA」的账直接接上：前向 FLOPs 由参数矩阵的形状完全决定，shape 走读（第 3 节）与参数账（第 4 节）是同一张图的两个读法。

## 2. Residual stream：形式化与总线视角

现代标准配方（Llama-3.1、OLMo-2 均为 RMSNorm + Pre-LN）的逐层递推：

$$
\begin{aligned}
x^{(0)}_{b,t,:} &= E_{x_{b,t}} \\
\tilde{x}^{(\ell)} &= x^{(\ell-1)} + \mathrm{Attn}_{\ell}\left(\mathrm{RMSNorm}\left(x^{(\ell-1)}\right)\right) \\
x^{(\ell)} &= \tilde{x}^{(\ell)} + \mathrm{FFN}_{\ell}\left(\mathrm{RMSNorm}\left(\tilde{x}^{(\ell)}\right)\right) \\
z_{b,t,:} &= W_U \, \mathrm{RMSNorm}\left(x^{(L)}_{b,t,:}\right)
\end{aligned} \qquad (1)
$$

**符号**：$x^{(\ell)} \in \mathbb{R}^{B \times T \times d}$ ：第 $\ell$ 层输出处的 residual stream，$\ell = 1, \ldots, L$ ；$\tilde{x}^{(\ell)}$ ：第 $\ell$ 层 attention 子层之后的中间 stream；$E \in \mathbb{R}^{V \times d}$ ：输入 embedding 矩阵，$E_{x_{b,t}}$ 为按 token id 查出的行向量；$\mathrm{Attn}_{\ell}$ 、 $\mathrm{FFN}_{\ell}$ ：第 $\ell$ 层的两个子层（各自带参数，内部结构见 2.3 / 2.6）；$\mathrm{RMSNorm}$ ：归一化算子，逐位置作用于 $d$ 维（公式与依据见 2.7）；$W_U \in \mathbb{R}^{V \times d}$ ：输出头；$z \in \mathbb{R}^{B \times T \times V}$ ：logits。

式 (1) 里每个子层的统一形式是「读出归一化后的 stream、把结果加回 stream」，即 Pre-LN 形式的 $x$ ← $x + f(\mathrm{LN}(x))$ ；LN 放在加法之后的 Post-LN 变体与两者的梯度行为差异见 2.7。递推可以直接展开：记 $a_\ell$ 、 $f_\ell$ 为第 $\ell$ 层 attention / FFN 子层的输出，则

$$
x^{(L)} = x^{(0)} + \sum_{\ell=1}^{L} a_{\ell} + \sum_{\ell=1}^{L} f_{\ell} \qquad (2)
$$

**符号**：$a_\ell = \mathrm{Attn}_{\ell}(\mathrm{RMSNorm}(x^{(\ell-1)}))$ ；$f_\ell = \mathrm{FFN}_{\ell}(\mathrm{RMSNorm}(\tilde{x}^{(\ell)}))$ 。输出是 $2L+1$ 个 $d$ 维向量之和。

从式 (2) 直接读出三条事实：

1. **stream 全程只加不覆盖**。$x^{(0)}$ 的贡献原样到达输出；反向传播时 $\partial x^{(L)} / \partial x^{(0)}$ 含单位阵项，深层网络的梯度通路不因层数而断——梯度流的完整机制在 2.7。
2. **各层读写同一向量总线**。每个子层从同一条 $d$ 维 stream 读（经 RMSNorm）、向同一条 stream 写（加法），层间通信的全部带宽就是每位置 $d$ 个浮点数。「residual stream」这个名字即来自这个总线视角。
3. **任意中间态可直接读出**。因为 $x^{(L)}$ 只是加和，把中间态 $x^{(\ell)}$ 乘上同一个 $W_U$ 也能得到一个合法的词表分布（logit lens 的操作），删掉和中某一项就对应「关掉」某个子层——4.7 机制可解释性整套方法（logit lens、activation patching）的数学前提就是式 (2) 的可加结构。

位置信息的注记：式 (1) 的 $x^{(0)}$ 只含内容向量。RoPE 家族（Llama-3.1、OLMo-2）把位置信息留在 attention 内部作用于 Q/K（2.5），stream 中不出现位置向量；GPT-2 一代的老配方才把可学习位置向量加进 $x^{(0)}$ 。两种选择的账见 2.5。

式 (2) 的反面同样信息量大：若把 LN 挪到加法之后（Post-LN，$x$ ← $\mathrm{LN}(x + f(x))$ ），加和每经过一次 LN 就被重新归一化，纯加法展开不再成立，$x^{(0)}$ 到输出的恒等通路被逐层重标定打断。这是 Post-LN 深堆叠时训练行为变差、需要 warmup 的结构根源；梯度尺度的严格结果（Xiong et al. 定理）在 2.7 展开，本章只登记可加结构的存亡与 LN 位置的对应关系。

## 3. 一次 forward 的完整 shape 走读

全书 shape 统一 $[B, T, H, D]$ 记法：$B$ = batch 中的序列数，$T$ = 序列长度，$H$ = head 数，$D$ = 每 head 维（即 1.2 的 $d_{\mathrm{head}}$ ）。MHA / GQA 标准形下 $d = H_q D$ 。逐行走读（Llama-3.1-8B 数值：$d=4096$、$H_q=32$、$H_{kv}=8$、$D=128$、$d_{\mathrm{ff}}=14336$、$V=128256$）：

| 步 | 运算 | 输入 shape | 输出 shape | 数值实例 |
| --- | --- | --- | --- | --- |
| 1 | embedding 查表 | $[B, T]$ | $[B, T, d]$ | $[B, T, 4096]$ |
| 2 | RMSNorm₁ | $[B, T, d]$ | $[B, T, d]$ | 逐位置 |
| 3 | QKV 投影 | $[B, T, d]$ | Q $[B, T, H_q, D]$；K、V $[B, T, H_{kv}, D]$ | $[B,T,32,128]$ / $[B,T,8,128]$ |
| 4 | RoPE（作用于 Q、K） | 同上 | shape 不变 | 2.5 |
| 5 | scores $= QK^{\top} / \sqrt{D}$ | Q、K | $[B, H_q, T, T]$ | 唯一的 $T^{2}$ 对象 |
| 6 | causal mask + softmax | $[B, H_q, T, T]$ | $[B, H_q, T, T]$ | 1.1 式 (8) 的 $M$ |
| 7 | 加权和 $PV$ | P、V | $[B, T, H_q, D]$ | 转置回 $T$ 在前 |
| 8 | concat heads + out 投影 | $[B, T, H_q D]$ | $[B, T, d]$ | $H_q D = 4096$ |
| 9 | 残差加（attention 半层完） | 两个 $[B, T, d]$ | $[B, T, d]$ | 式 (1) 第二行 |
| 10 | RMSNorm₂ + FFN：up / gate → SiLU ⊙ up → down | $[B, T, d]$ | $[B, T, d]$（内部经 $[B, T, d_{\mathrm{ff}}]$） | $d_{\mathrm{ff}} = 14336$ |
| 11 | 残差加；步 2–10 共重复 $L$ 次 | $[B, T, d]$ | $[B, T, d]$ | $L = 32$ |
| 12 | final RMSNorm + 输出头 $W_U$ | $[B, T, d]$ | $[B, T, V]$ | $V = 128256$ |

shape 注记：步 5–6 的 scores 习惯写作 $[B, H_q, T, T]$ ——head 维提到 $T$ 之前，是 batched matmul 的实现视角；逐 token 的量（Q/K/V、FFN 中间量）保持 $[B, T, H, D]$ 记法。

三个读表要点：

1. **$T$ 维只在步 5–7 混合**，其余各行对 $T$ 维逐位置独立。教师强制（1.1 第 4 节）一次前向给全部 $T$ 个位置打分，在硬件上就是这张表里非 attention 行的大批量 matmul。
2. **$d$ 全程不变**——残差加要求两侧同维，stream 宽度是整机的恒定带宽；扩张只发生在 FFN 内部的 $d_{\mathrm{ff}}$ 与 scores 的 $T$ 上。
3. **唯一随 $T^{2}$ 增长的对象是步 5/6 的 scores 矩阵**：1.2 第 6 节的 $2LT^{2}d$ FLOPs 账、第 8 节的激活账、4.1 的 FlashAttention 都围着它转。GQA 的痕迹在步 3：K/V 的 head 数 $H_{kv}=8 < H_q=32$ ，投影宽度砍半，但 scores 仍按 $H_q$ 计（机制见 4.1，FLOPs 口径见 1.2 第 6 节）。

代入 $B=1$ 、 $T=4096$ 走一遍量级：进入第一层时 stream 是 $4096 \times 4096 \approx 1.7 \times 10^{7}$ 个元素；QKV 投影后 Q 有 $4096 \times 4096$ 、K/V 各 $4096 \times 1024$ 个；scores 矩阵 $32 \times 4096 \times 4096 \approx 5.4 \times 10^{8}$ 个元素——全层最大对象；FFN 内部最宽处 $4096 \times 14336 \approx 5.9 \times 10^{7}$ 。一条 forward 里张量元素数的峰值出现在 scores 上且随 $T^{2}$ 涨，而参数矩阵的形状全程与 $T$ 无关——「参数量与序列长度解耦」在 shape 层面就是这么写的，这也是 1.2 第 6 节把 attention 二次项从 $2N$ 里单列出来的原因。

## 4. 参数量标尺：$12Ld^{2}$ 推导与对账

估算口径取标准形：MHA（$H_{kv} = H_q$ ）+ 两矩阵 FFN（$d_{\mathrm{ff}} = 4d$ ），bias 与 norm 参数为 $O(Ld)$ 低阶项忽略。逐项：

- attention 四个投影 $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ ：每层 $4d^{2}$ ；
- FFN 两矩阵 $W_1 \in \mathbb{R}^{d \times 4d}$ 、 $W_2 \in \mathbb{R}^{4d \times d}$ ：每层 $8d^{2}$ 。

合计每层 $12d^{2}$ ，乘 $L$ 层：

$$
N \approx 12Ld^{2}, \qquad \text{分解为} \; \underbrace{4d^{2}}_{\text{attn}} + \underbrace{8d^{2}}_{\text{FFN}} \; \text{每层} \qquad (3)
$$

embedding 单算（查表不进 FLOPs 账，参数账单列，口径同 1.2 第 2 节）：输入 embedding $Vd$ ；untied 时输出头是独立的第二份 $Vd$ 。总参数

$$
N_{\mathrm{total}} \approx 12Ld^{2} + 2Vd \quad (\text{untied}) \qquad (4)
$$

**符号**：$N$ ：层参数（不含 embedding 与输出头）；$N_{\mathrm{total}}$ ：全参数。tied 时式 (4) 的 $2Vd$ 退化为 $Vd$ 。

现代配方对标准形有两处系统性偏离，对账用两个锚点（config 数值：[Llama-3.1-8B config](../references.md)、[OLMo-2-7B config](../references.md)，均 [A]）：

- **GQA 使 attention 项小于 $4d^{2}$** ：K/V 投影宽度缩为 $H_{kv} D$ 。Llama-3.1-8B 的 $H_{kv} D = 1024 = d/4$ ，QKV + out 合计 $d^{2} + 2 \cdot d \cdot (d/4) + d^{2} = 2.5d^{2}$（41943040 个参数），比标准形省 $1.5d^{2}$ 。
- **SwiGLU 三矩阵使 FFN 项依赖 $d_{\mathrm{ff}}$ 取值** ：等参数规则取 $d_{\mathrm{ff}} = 8d/3$ 时 $3 d \, d_{\mathrm{ff}} = 8d^{2}$ ，与两矩阵标准形持平（[Shazeer, 2020](../references.md)，arXiv:2002.05202，[B]；推导在 2.6）。实际取值偏离此规则：OLMo-2-7B 取 $d_{\mathrm{ff}} = 11008 \approx 2.69d$（贴 $8d/3 \approx 10923$ 、向上取整到 256 的倍数），FFN $= 8.06d^{2}$ ；Llama-3.1-8B 取 $d_{\mathrm{ff}} = 14336 = 3.5d$ ，FFN $= 10.5d^{2}$（「为什么更宽」的账见 2.6）。

代入两个锚点（每层、以 $d^{2}$ 为单位，$d^{2} = 16777216$ ）：

| 项 | 标准形 | Llama-3.1-8B | OLMo-2-7B |
| --- | --- | --- | --- |
| attention | $4d^{2}$ | $2.5d^{2}$（GQA 4:1） | $4d^{2}$（MHA，$H_{kv}=32$） |
| FFN | $8d^{2}$ | $10.5d^{2}$（$d_{\mathrm{ff}}=3.5d$） | $8.06d^{2}$（$d_{\mathrm{ff}} \approx 2.69d$） |
| 每层合计 | $12d^{2}$ | $13d^{2}$ | $12.06d^{2}$ |
| 层参数（$\times 32$） | $6.44 \times 10^{9}$ | $6.98 \times 10^{9}$ | $6.48 \times 10^{9}$ |
| embedding + 输出头（untied，$2Vd$） | — | $1.05 \times 10^{9}$（$V=128256$） | $0.82 \times 10^{9}$（$V=100352$） |
| 总参数 | — | $8.03 \times 10^{9}$ | $7.30 \times 10^{9}$ |

Llama-3.1-8B 的两处偏离一负一正：GQA 省 $1.5d^{2}$ 、宽 FFN 加 $2.5d^{2}$ ，净 $+1d^{2}$ ，层参数恰好 $13Ld^{2}$（与 1.2 第 9 节逐项加出的 6979321856 一致）。OLMo-2-7B 几乎就是标准形，偏差 0.5% 来自 $d_{\mathrm{ff}}$ 的取整。$12Ld^{2}$ 的价值不在精度在结构：给定 $L$ 、 $d$ 、 $V$ 三个整数立刻得到参数量级（一成以内），且偏差本身携带架构信息——看到 $>12Ld^{2}$ 先查 FFN 宽度，看到 $<12Ld^{2}$ 先查 KV head 数。

用法分速算与对账两头。速算：拿到一个新模型的 config，先算 $12Ld^{2}$ 与 $2Vd$ ，参数量级与命名（7B / 8B）立即可核对。对账：实际层参数偏离 $12Ld^{2}$ 时，偏差就是架构 diff 的签名——上表中 Llama-3.1-8B 的 $+1d^{2}$ 同时记录了它的 GQA（$-1.5d^{2}$ ）与宽 FFN（$+2.5d^{2}$ ）；反过来，层参数恰为 $12Ld^{2}$ 的模型要么真是标准形，要么两处偏离恰好抵消，分辨办法是直接查 config 的 num_key_value_heads 与 intermediate_size 两个字段。这个标尺是 1.2 的 $C = 6ND$ 里 $N$ 的速算口径、1.3 scaling law 的 $N$ 定义、Part 5 逐家对账的基准。

## 5. 设计约束三指标：为什么是 attention

Transformer 原文用三个指标给序列建模层打分（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，Table 1；原文序列长度记 $n$ ，本书统一记 $T$ ）：

| 指标 | self-attention | recurrent |
| --- | --- | --- |
| 每层计算量 | $O(T^{2} d)$ | $O(T d^{2})$ |
| 串行操作数（sequential operations） | $O(1)$ | $O(T)$ |
| 任意两位置的最大路径长度 | $O(1)$ | $O(T)$ |

原文 Table 1 还有两行值得登记：self-attention (restricted)——每位置只 attend 半径 $r$ 的窗口——每层计算量降为 $O(rTd)$ 、路径长度升为 $O(T/r)$ ；卷积层（kernel 宽 $k$ ）为 $O(kTd^{2})$ 、串行 $O(1)$ 、路径 $O(\log_k T)$ 。这两行是 Part 4 的预告：稀疏 attention（4.2）就是把 restricted 行做实做细；所有替代架构都可以读作在这三个指标的张量上重新选点。

逐行读主表。

**每层计算量**。attention 的 $T^{2} d$ 是 $QK^{\top}$ 与 $PV$ 两笔：一层 full（双向）为 $4T^{2}d$ 、causal 为 $2T^{2}d$（1.2 式 (6)）；recurrent 每位置一次 $d \times d$ 矩阵乘，$T$ 步合计 $O(Td^{2})$ 。令 causal 口径两侧相等：$2T^{2}d = 2Td^{2}$ ，交叉点 $T = d$ ——$T < d$ 时 attention 的纯计算量反而更省。现代工作点 $d = 4096$ 而训练 $T$ 从数千到十余万，attention 在长序列端已经更贵；它赢的是下面两行。

**串行操作数**。指前向计算图中必须按顺序执行的最长运算链。attention 一层的 $T^{2}$ 个 score 互不依赖，全部可并行，链长 $O(1)$ ；recurrent 的 $h_t$ 依赖 $h_{t-1}$ ，链长 $O(T)$ 。这一行决定训练吞吐：$O(1)$ 串行意味着 $T$ 个位置的运算可以一次性铺开到硬件的全部并行单元上，1.1 第 4 节 teacher forcing 的并行性在架构层面的根据就是它；$O(T)$ 串行则意味着序列越长硬件闲置越多（第 6 节展开）。

**最大路径长度**。指任意两个位置之间信号要经过的最长变换链。位置 1 的信息传到位置 $T$ ：attention 里是一次 score，路径 $O(1)$ ；recurrence 里是 $T-1$ 次状态转移的复合，路径 $O(T)$ 。路径长度决定长程依赖的学习难度——每多一步，梯度多一次雅可比连乘、信号多一次被中间变换覆盖的机会；原文以此论证 self-attention 对长程依赖的可达性。代价被摊到了第一行的 $T^{2}$ 与 KV cache 上（2.4）。

一句话总括这笔交易：attention 用 $T^{2}$ 量级的计算与内存，买 $O(1)$ 的并行度与路径长度。$T$ 在数千到数万时交易划算；$T$ 继续涨，重新谈判它的就是 Part 4 的全部内容（4.1 KV 压缩、4.2 稀疏 attention、4.3 linear attention、4.4 hybrid）。

## 6. Recurrence 的两个数学代价

把三指标表右列展开成机制。recurrent 层

$$
h_t = f\left(h_{t-1}, x_t; \theta\right), \qquad h_t \in \mathbb{R}^{d_h} \qquad (5)
$$

**符号**：$h_t$ ：位置 $t$ 的隐状态；$d_h$ ：隐状态维；$f$ ：共享参数的转移函数。

**代价一：依赖链不可并行**。计算图里 $h_1 \to h_2 \to \cdots \to h_T$ 是一条深度 $T$ 的链，每步以前一步的输出为输入——关键路径长度 $T$ 是图结构性质，与实现无关，任何调度都无法压缩。训练时 $B$ 维可以并行，时间维不行：$h_t$ 的 matmul 必须等 $h_{t-1}$ 就绪；teacher forcing 能让 $T$ 个位置的 loss 并行计算，但前向隐状态仍要串行走完全链。硬件吞吐由「可用并行度 ÷ 关键路径」决定：序列每加长一倍，recurrence 的墙上时间跟着涨一倍，加卡救不了时间维。attention 的关键路径是常数（一层内所有 score 同时可算），序列加长只加并行工作量、不加串行深度，多出来的活由硬件并行度吸收。这正是三指标表第二行 $O(T)$ 与 $O(1)$ 之差的机制内容。

**代价二：固定维状态的信息瓶颈**。前缀 $x_{\le t}$ 有 $V^{t}$ 种取值，区分它们所需的信息量随 $t$ 线性涨（均匀分布上界为每 token $\log_{2} V$ bits）；而 $h_t$ 的维度 $d_h$ 与 $t$ 无关，容量被封顶。工程口径的上界账：fp32 存储下状态容量约 $32 \, d_h$ bits，当 $t \log_{2} V > 32 \, d_h$ 时映射必然多对一，旧信息被新信息挤出。代入一个中等词表量级（$V = 50257$ ，GPT-2 词表，数值经 1.1 第 1 节核实，$\log_{2} V \approx 15.6$ ）与 $d_h = 4096$ ：交叉点在 $t \approx 8400$ ——此后平均每写入一个新 token 都要以覆盖旧内容为代价。这是上界账而非不可能定理：实数向量的信息论严格表述要考虑量化精度与自然语言的冗余（实际熵率远低于 $\log_{2} V$ ），但「容量与 $t$ 无关、需求与 $t$ 成正比」这一结构性事实不变。瓶颈之上还叠着路径问题：即便信息存得下，让位置 $T$ 的输出受位置 1 影响也要经过 $T$ 次 $f$ 的复合——三指标表第三行。

对照 attention：位置 $t$ 的 query 直接读全部前缀的 K/V，信息不经过固定维瓶颈，路径 $O(1)$ ；代价是把「不压缩」物化成随 $t$ 线性增长的 KV cache（1.2 第 8 节账三，推导在 2.4）——用内存账替代信息瓶颈。Part 4.3 的 linear attention / SSM 在这两条代价之间重新选点：恢复固定维状态的 $O(1)$ 内存，同时用核技巧保住训练并行，而信息瓶颈以新的形式回来（4.3 末尾的 associative recall 下界论证）。

## 7. 三种架构形态（结构侧，细节 → Part 3）

同一套 residual stream + attention 组件，按可见性 mask $M$（1.1 第 7 节）与连接方式的不同，组装成三种形态：

```
encoder-only              encoder-decoder                     decoder-only
M ≡ 0（双向全可见）        enc：M ≡ 0                          M = 下三角（因果）
                          dec：下三角 + cross-attn 读 enc 输出
x1  x2  x3  x4            src ──► [enc 层 ×L] ──┐             x1→x2→x3→x4
 ↕   ↕   ↕   ↕                                ▼（每层 cross-attn）
[enc 层 ×L]               tgt ──► [dec 层 ×L] ──► 预测         [dec 层 ×L] ──► 预测下一 token
 ↕   ↕   ↕   ↕
每位置向量 → 分类 / 检索
BERT 家族                 T5 / Transformer 原文（翻译）        GPT、Llama、OLMo、Qwen、DeepSeek
```

- **encoder-only**：双向可见，每个位置输出一个向量，服务理解与检索类任务；不能做自回归生成，证明在 3.2。
- **encoder-decoder**：encoder 双向读源序列，decoder 因果生成目标序列，cross-attention 让 decoder 的每个位置 attend encoder 的全部输出。cross-attention 的依赖结构：Q 来自 decoder 的 stream，K/V 来自 encoder 的最终输出 $[B, T_{\mathrm{src}}, d]$ ——机制与 self-attention 相同，只是 K/V 的来源换成了另一条 stream（3.2 展开）。这是机器翻译的原生形态，也是 Transformer 原文的完整结构；本书主线取其 decoder 半边。
- **decoder-only**：纯因果，即 1.1 式 (8) 的 CLM 行；当前主流 LLM 的形态。

三种形态的训练信号密度不一样，这是机制层面登记 decoder-only 主流地位的方式（不是历史原因）：CLM 对一条长度 $T$ 的序列给出 $T$ 个监督位置，全部参数参与每个位置的预测；MLM 只在被 mask 的子集位置上计损失（mask 方案见 3.1 / 3.2），encoder 的输出头也只做这一件事；encoder-decoder 则要求成对数据，源序列位置不产生生成式监督。同样的原始文本预算下，decoder-only 从每条序列榨出的梯度信号最多——这是数据效率论证，与 1.1 第 4 节 teacher forcing 的「一条序列 = $T$ 个分类问题」直接同源。objective 侧（loss mask、腐蚀方案）已在 1.1 第 7 节统一为 $(x', M, \ell)$ 三元组；结构差异与互相不可替代性的证明在 Part 3 展开。本 Part 其余各章（2.2–2.8）全部在 decoder-only 形态内：逐个组件拆公式，2.8 装回整机。

## 8. 锚点走读：Llama-3.1-8B config.json ↔ 结构图

逐字段映射（[Llama-3.1-8B config](../references.md)，[A]）：

| config 字段 | 值 | 结构图中的位置 | 本书展开 |
| --- | --- | --- | --- |
| vocab_size | 128256 | $E$ 与 $W_U$ 的行数 $V$ | 1.1、2.2 |
| hidden_size | 4096 | residual stream 宽度 $d$ | 本章 |
| num_hidden_layers | 32 | 层数 $L$ | 本章 |
| num_attention_heads | 32 | Q head 数 $H_q$ | 2.3 |
| num_key_value_heads | 8 | K/V head 数 $H_{kv}$（GQA 4:1） | 4.1 |
| （导出）head_dim | 128 | $D = d / H_q$ | 2.3 |
| intermediate_size | 14336 | FFN 宽度 $d_{\mathrm{ff}} = 3.5d$ | 2.6 |
| rope_theta | 500000 | RoPE 基频（attention 内部作用于 Q/K） | 2.5 |
| max_position_embeddings | 131072 | 上下文窗口 $T_{\max}$ | 4.6 |
| tie_word_embeddings | false | $E$ 与 $W_U$ 独立（式 (4) 中 $2Vd$ 的来源） | 2.2 |

对照组 OLMo-2-7B（[OLMo-2-7B config](../references.md)，[A]；论文 [Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]）：同样 $L=32$ 、 $d=4096$ ，但 $H_{kv}=32$（纯 MHA，无 GQA）、 $d_{\mathrm{ff}}=11008 \approx 2.69d$ 、 $V=100352$ 、attention_bias = false、rms_norm_eps = $10^{-6}$ ——第 4 节对账表中「标准形另一侧」的实例。OLMo 2 的论文、权重、数据、训练日志全开放，本书核账类断言以它为默认对照锚点。Llama-3.1 谱系的官方技术报告（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]）给出 8B 在谱系中的位置（70B、405B 同配方放大）。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 架构二分 | attention 跨位置，其余逐位置 | 全景图的读法，式 (1) |
| residual stream | $x$ ← $x + f(\mathrm{LN}(x))$（Pre-LN） | 只加不覆盖；展开式 $x^{(L)} = x^{(0)} + \sum a_\ell + \sum f_\ell$ 是 4.7 的前提，式 (1)(2) |
| shape 记法 | $[B, T, H, D]$，$d = H_q D$ | 唯一 $T^{2}$ 对象：scores $[B, H_q, T, T]$ |
| 参数量标尺 | $N \approx 12Ld^{2}$（attn $4d^{2}$ + FFN $8d^{2}$），总 $+2Vd$（untied） | Llama-3.1-8B 恰好 $13Ld^{2}$，OLMo-2-7B $12.06d^{2}$/层，式 (3)(4) |
| 设计三指标 | attention $O(T^{2}d), O(1), O(1)$ vs recurrent $O(Td^{2}), O(T), O(T)$ | Transformer 原文 Table 1；交叉点 $T = d$ |
| recurrence 两代价 | 关键路径长 $T$；容量 $\approx 32 d_h$ bits 对需求 $t \log_{2} V$ | 不可并行 + 信息瓶颈，式 (5) |
| 三种形态 | encoder-only / encoder-decoder / decoder-only | mask 与连接方式的差异；细节 → Part 3 |

## 练习

1.（计算）用 OLMo-2-7B 的 config 数值（$L=32$、$d=4096$、$d_{\mathrm{ff}}=11008$、MHA、$V=100352$、untied）逐项重算每层参数与总参数，验证层参数对 $12Ld^{2}$ 的偏差在 1% 以内；再对 Llama-3.1-8B 验证层参数恰好为 $13Ld^{2}$ ，以 $d^{2}$ 为单位分解 GQA 与 $d_{\mathrm{ff}}=3.5d$ 各自的贡献（方向与大小）。
2.（计算）设计约束第一行：固定 $d=4096$ ，求每层计算量上 self-attention（causal，$2T^{2}d$ ）追平 recurrent（$2Td^{2}$ ）的 $T$ ；再把 recurrent 换成「隐状态维 $d_h = d/2$ 」的窄隐状态变体，交叉点怎么移动？由此说明「$T$ 小则 recurrent 省计算」的结论对宽度假设有多敏感。
3.（推导）从式 (1) 推出展开式 (2)，写明 $a_\ell$ 、 $f_\ell$ 的定义；据此说明：(a) 输出是 $2L+1$ 个 $d$ 维向量之和；(b) 「把第 $k$ 层 FFN 置零」等价于从和中删去一项 $f_k$ ——为什么这正对应 activation patching 的可行性（一句话，细节在 4.7）；(c) 若把 Pre-LN 换成 Post-LN（$x$ ← $\mathrm{LN}(x + f(x))$ ），式 (2) 的纯加法展开还成立吗？给出理由。
4.（计算 + 论证）recurrence 信息瓶颈的账：$d_h = 4096$ 、fp16 存储（16 bit/维）、$V = 50257$（GPT-2 词表）：求使 $t \log_{2} V > 16 \, d_h$ 成立的最小整数 $t$ ；指出该上界账的两个漏洞（fp16 是工程选择而非数学限制；自然语言冗余使实际熵率远低于 $\log_{2} V$ ），并说明为什么该论证仍能正确预测长上下文下的检索失败（提示：容量与 $t$ 无关这一点没有变）。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——设计约束三指标出自 Table 1
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）
- [Shazeer, 2020](../references.md)：GLU Variants Improve Transformer，arXiv:2002.05202，[B]（key: glu-variants）——$d_{\mathrm{ff}} = 8d/3$ 等参数规则出处，推导在 2.6
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）
