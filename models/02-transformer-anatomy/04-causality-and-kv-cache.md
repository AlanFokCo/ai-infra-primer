# 2.4 Causality 与 KV Cache（Causality and the KV Cache）

> 本章把 1.1 留下的 $(x', M, \ell)$ 接口做实：mask 的加性形式与 $-\infty$ 的极限语义、三种 mask 与三种 objective 的对照、自回归生成的 prefill / decode 两阶段伪码，然后是两个从零推导——KV 可缓存性（ $k_s, v_s$ 与未来步无关的归纳论证、 $q$ 不值得缓存的依赖结构分析）与 cache size 公式 $\mathrm{bytes} = 2 L H_{kv} d_{\mathrm{head}} T B b$ 的逐项账——最后落到 prefill 与 decode 的算术强度分野（roofline 入口）与 teacher forcing 的无泄漏证明 | 前置：1.1（ $(x', M, \ell)$ 三元组、teacher forcing、祖先采样）、1.2（ $2mnk$ 口径、前向 $2N$、式 (6) 的 attention 二次项、账三的登记处）、2.3（attention 公式、max 减法、 $[B, H, T, T]$ shape） | 锚点：Llama-3.1-8B / 70B config [A]、OLMo-2-7B config [A]、OLMo 2 与 Llama 的 modeling 代码 [A]；Vaswani et al. [A]、Shazeer（MQA）[B]

## 1. mask 的数学：加性形式与 $-\infty$ 的极限语义

1.1 式 (8) 把可见性压缩成一个矩阵 $M \in \lbrace0, -\infty\rbrace^{T \times T}$ ，在 softmax 之前以加法进入 score：

$$
\alpha_{ij} = \frac{\exp\left(s_{ij} + M_{ij}\right)}{\sum_{l=1}^{T} \exp\left(s_{il} + M_{il}\right)} \qquad (1)
$$

**符号**： $s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_{\mathrm{head}}}$ ：未加 mask 的 score（2.3 式 (4)；该章按论文口径把 head 维记作 $d_k$ ，本书 $d_{\mathrm{head}}$ 、 $D$ 、 $d_k$ 三者互通）； $\alpha_{ij}$ ：mask 后位置 $i$ 对位置 $j$ 的 attention 权重； $M_{ij} = 0$ 为可见， $M_{ij} = -\infty$ 为屏蔽。行号 $i$ 是被影响者，列号 $j$ 是被看见者——这个方向第 2 节要用。

为什么用加法而不是直接把权重置零：softmax 的输入是 logit，「权重为零」在 logit 侧的对应物就是 $-\infty$ ；在 logit 上加 $-\infty$ 与「把该列从归一化集合里删除」是同一操作的两种写法。展开成可见集形式看得最清楚。记位置 $i$ 的可见集 $V(i) = \lbrace j : M_{ij} = 0\rbrace$ ，则 $j \notin V(i)$ 时 $\exp(s_{ij} + M_{ij}) = \exp(-\infty) = 0$ ，分母只剩可见列：

$$
\alpha_{ij} = \begin{cases} \dfrac{\exp(s_{ij})}{\sum_{l \in V(i)} \exp(s_{il})}, & j \in V(i) \\ 0, & j \notin V(i) \end{cases} \qquad (2)
$$

mask 的语义因此是「重新划定 softmax 的归一化域」，不是「给 score 打折」：被屏蔽位置退出竞争，可见位置之间的相对结构与无 mask 时逐对相同——任意 $j, k \in V(i)$ 有 $\alpha_{ij} / \alpha_{ik} = \exp(s_{ij} - s_{ik})$ ，与 $M$ 无关。

本章主角是因果掩码（causal mask）：

$$
M^{\mathrm{causal}}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases} \qquad (3)
$$

即 $V(i) = \lbrace1, \ldots, i\rbrace$ ：位置 $i$ 恰好看见自己与自己的过去。这就是 1.1 式 (3) 的自回归分解在 attention 层的落实方式——2.1 §1 的二分法说过，attention 是唯一跨位置混合信息的算子，所以「不许看未来」只需要、也只能在这里落实；RMSNorm、FFN、残差加逐位置，天然不跨时间。这个 mask 形态出自 Transformer 原文的 decoder self-attention（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.2.3：把 softmax 输入中非法连接对应的位置置为 $-\infty$ ）。

**极限语义与实现**。 $-\infty$ 是数学写法，实现里是两种等价物：真正的 `float("-inf")`（IEEE 语义下 $-\infty$ 减有限数仍是 $-\infty$ ， $\exp(-\infty) = 0$ 精确成立），或一个足够大的负哨兵（ $-10^{9}$ 量级，或 dtype 最小值）。哨兵方案下 $s_{ij} + M_{ij}$ 仍是约 $-10^{9}$ 的有限数，经 2.3 式 (10) 的 max 减法后 $\exp$ 下溢到精确的 0——bf16 下 $\exp(-100)$ 已是机器 0，哨兵与真 $-\infty$ 在浮点输出上不可区分。哨兵取 dtype 最小值（bf16 / fp32 约 $-3.4 \times 10^{38}$ 、fp16 为 $-65504$ ）结论相同：有限 score 加上它仍压在 dtype 下界附近（bf16 / fp32 为 $-10^{38}$ 量级、fp16 为 $-6.6 \times 10^{4}$ 量级），max 减法后照旧下溢为 0；与真 $-\infty$ 唯一的行为差异在全掩行——哨兵方案不产 NaN 而是静默退化为均匀分布，因此更依赖对角线恒可见来防呆。max 减法本身与 mask 兼容：行最大值只可能出现在可见列（可见列有限、被掩列为 $-\infty$ ），减去它不改变任何被掩列的零值。真正的边界条件只有一个：若某行可见集为空，分母为 0、max 为 $-\infty$ ，整行产出 NaN。causal mask 的对角线恒可见（ $M_{ii} = 0$ ），每行至少看见自己，空集不会出现；2.3 §8 教学代码里 `torch.triu(..., diagonal=1)` 的偏移量 1 就是「保留下三角含对角线」的含义，偏移写错一格——对角线也被屏蔽或向右多放一格——分别对应 NaN 与泄漏（第 7 节）。

## 2. 三种 mask ↔ 三种 objective

1.1 第 7 节把训练目标统一为 $(x', M, \ell)$ 三元组： $x'$ 是实际输入、 $M$ 决定可见性、 $\ell$ 决定哪些位置计损失。三个 objective 的 mask 行并排对照（矩阵实例 $T$ = 4 已在 1.1 式 (8) 后给出，此处不重复）：

| objective | $x'$ | $M$ 的结构 | 可见集 $V(i)$ | $\ell$ | 可自回归生成 |
| --- | --- | --- | --- | --- | --- |
| CLM | $x$ 原样 | 下三角，式 (3) | $\lbrace1, \ldots, i\rbrace$ | 全 1 | 能 |
| MLM | 腐蚀序列（子集位置换成 mask token） | $M \equiv 0$ 全可见 | $\lbrace1, \ldots, T\rbrace$ | 仅被替换位 | 不能（证明在 3.2） |
| prefix-LM | $x$ 原样 | 前缀块全可见 + 续写区下三角 | 见下 | 仅续写区 | 续写区能 |

prefix-LM 的可见集按前缀长 $p$ 分段： $i \le p$ 时 $V(i) = \lbrace1, \ldots, p\rbrace$（前缀内双向）， $i > p$ 时 $V(i) = \lbrace1, \ldots, i\rbrace$（前缀加已生部分）。三族的差别全在 $V(i)$ 的形状上；objective 侧（腐蚀方案与 loss mask 的更多取值）在 3.1 展开，本章只固定 mask 这一维。

「能否自回归生成」由一条可判定的性质决定：

$$
V(i) \subseteq \lbrace1, \ldots, i\rbrace \quad \text{for all } i \qquad (4)
$$

**符号**：式 (4) 要求每个位置的可见集不含未来。第 7 节将证明它同时是三件事的条件：训练无泄漏、推理可生成、KV 可缓存。逐行对照上表：CLM 取等号；prefix-LM 在续写区满足（前缀属于输入、生成时刻已知）；MLM 的全 0 矩阵处处违反——位置 $t$ 的 logits 依赖 $x_{>t}$ ，而串行采样走到第 $t$ 步时未来尚不存在，该条件分布在生成时刻不可用。这是 encoder 形态不能直接做自回归生成的门槛，完整论证（包括「为什么不能用多轮迭代采样绕过」）在 3.2。

方向再强调一次：mask 限制的是「哪些列参与第 $i$ 行的归一化」。2.3 §6 的注记在这里接回——causal mask 把「前面有几个 token」这一计数级位置信号锚进了计算图，它是 NoPE 证据（2.5）所讨论的位置信息下限；prefix-LM 的前缀块则连这个计数信号都弱化（块内任意置换等变），这是块状可见性的隐性代价。

## 3. 自回归生成：prefill 一次 + decode 循环

训练时 teacher forcing 一次前向并行打全部位置（1.1 §4）；生成时未来不存在，只能串行。标准服务形态把一次生成拆成两阶段：

```
输入：prompt token ids x_1..x_{T_p}，模型 f_θ，采样器 sample(·)
──────────────────────────────────────────────────────────────
prefill（一次并行前向，长度 T_p）：
    logits [B, T_p, V] ← f_θ(x_{1:T_p})         # 与训练同形的 mask 前向
    每层保存 K^{(ℓ)}_{1:T_p}, V^{(ℓ)}_{1:T_p}      # cache 初始化，见 §4
    x_{T_p+1} ← sample(softmax(logits[:, T_p]))   # 只用最后位置的分布
decode（串行循环，t = T_p+1, T_p+2, ...）：
    logits [B, 1, V] ← f_θ(x_t；cache)            # 输入只有 1 个 token
    每层：k_t, v_t append 进 cache                  # §4.4 的读写模式
    x_{t+1} ← sample(softmax(logits[:, 1]))
    直到 x_{t+1} = eos 或 t = T_max
──────────────────────────────────────────────────────────────
```

两阶段的非对称是本章后半全部账的源头：prefill 与训练前向同构—— $T_p$ 个 token 的大 matmul 一次打完；decode 每步只进 1 个 token，全部 $L$ 层权重与不断变长的 cache 却都要为这 1 个 token 服务。1.1 §8 的祖先采样语义（逐步采出的序列是模型分布的精确样本）不依赖这个拆分——prefill 只是同一串行过程把第一段并行化的实现形式。

先算「没有 cache 会怎样」，作为 §4 的动机。朴素 decode 第 $t$ 步对长度 $t$ 的前缀重跑整条前向：参数项 $2Nt$ （1.2 式 (2)，每 token $2N$ ）、attention 项 $2Lt^{2}d$ （1.2 式 (6) 代入 $T = t$ ）。生成 $T$ 个 token 的总账：

$$
\mathrm{FLOPs}_{\mathrm{naive}} = \sum_{t=1}^{T} \left(2Nt + 2Lt^{2}d\right) = NT(T+1) + \frac{Ld}{3}\thinspace T(T+1)(2T+1) \approx NT^{2} + \frac{2}{3}LT^{3}d \qquad (5)
$$

**符号**： $N$ ：参与 matmul 的参数量（1.2 式 (2) 口径）； $L$ ：层数； $d$ ：stream 宽度。第二步用了 $\sum_t t = T(T+1)/2$ 与 $\sum_t t^{2} = T(T+1)(2T+1)/6$ 。

attention 项随 $T^{3}$ 涨。代入 Llama-3.1-8B（ $N \approx 6.98 \times 10^{9}$ 层参数口径、 $L$ = 32、 $d$ = 4096，[Llama-3.1-8B config](../references.md)，[A]）生成满窗 $T$ = 131072：attention 项约 $2.0 \times 10^{20}$ FLOPs——按 1.2 §9 推出的 $3.3 \times 10^{14}$ FLOPS 有效吞吐，相当于单卡跑约 7 天，而第 4 节的缓存把同一生成压到 $4.5 \times 10^{15}$ FLOPs（式 (8) 的 attention 项）。式 (5) 就是缓存要消灭的对象。

## 4. KV cache 从零推导

### 4.1 decode 步 $t$ 的 attention 展开

固定层 $\ell$ ，把 decode 第 $t$ 步的 attention 写成逐位置形式。记该层输入 stream 在位置 $s$ 处的归一化结果为 $u_s$（即 2.1 式 (1) 中 RMSNorm 的输出；batch 维与 head 维先省略，逐 head 同理）：

$$
q_t = u_t W^{Q}, \qquad k_s = u_s W^{K}, \qquad v_s = u_s W^{V} \qquad (6)
$$

$$
o_t = \sum_{s=1}^{t} p_{t,s} \thinspace v_s, \qquad p_{t,s} = \frac{\exp\left(\langle q_t, k_s \rangle / \sqrt{d_{\mathrm{head}}}\right)}{\sum_{j=1}^{t} \exp\left(\langle q_t, k_j \rangle / \sqrt{d_{\mathrm{head}}}\right)} \qquad (7)
$$

**符号**： $u_s \in \mathbb{R}^{d}$ ：层 $\ell$ 在位置 $s$ 的归一化后 stream； $q_t, k_s \in \mathbb{R}^{d_{\mathrm{head}}}$ ， $v_s \in \mathbb{R}^{d_{\mathrm{head}}}$（GQA 时 $k_s, v_s$ 属于 $H_{kv}$ 个 KV 头之一、被同组的 Q 头共享，机制在 4.1）； $o_t \in \mathbb{R}^{d_{\mathrm{head}}}$ ：该头在位置 $t$ 的输出。式 (7) 的求和上限是 $t$ 而不是 $T$ ——causal mask 式 (3) 把 $s > t$ 的列全部置零，逐行看就是截断（式 (2) 的可见集形式）。

式 (6)(7) 里出现的量分三类： $q_t$ 与权重 $p_{t,s}$ 是本步新算的； $k_t, v_t$ 是本步新量的； $k_s, v_s$（ $s < t$ ）是历史量。缓存论证要回答的核心问题：历史量 $k_s, v_s$ 与第 $s$ 步当时算出的值是否相同？若相同，存下来即可，不必在第 $t$ 步重算。

### 4.2 不可变性： $k_s, v_s$ 与未来无关

**命题（前缀不可变）**。causal mask 下，每个位置的 stream 只依赖该位置及其过去：对任意层 $\ell$ 与位置 $s$ ， $x_s^{(\ell)}$ 是 $(x_1, \ldots, x_s)$ 的函数，与 $x_{s+1}, \ldots$ 无关。推论： $k_s = \mathrm{RMSNorm}\left(x_s^{(\ell-1)}\right) W^{K}$ 与 $v_s$ 在第 $s$ 步算出后永不改变——追加任意数量的未来 token 都不会改动它们。

**证明**（对 $\ell$ 归纳）。基例 $\ell = 0$ ： $x_s^{(0)} = E_{x_s}$ 是 embedding 查表，只含 token $s$ 自身。归纳步：设第 $\ell - 1$ 层全部位置的 stream 只依赖各自过去。第 $\ell$ 层 attention 在位置 $s$ 的输出由式 (7) 给出、求和上限为 $s$ —— $o_s$ 只含 $j \le s$ 的 $k_j, v_j$ 与自己的 $q_s$ ；由归纳假设， $u_j$（ $j \le s$ ）是 $(x_1, \ldots, x_j)$ 的函数、进而是 $(x_1, \ldots, x_s)$ 的函数，故 $o_s$ 亦然。RMSNorm、FFN、残差加逐位置（2.1 §1 二分法），于是 $x_s^{(\ell)}$ 只依赖 $(x_1, \ldots, x_s)$ 。∎

因果性在证明里只被用了一次——式 (7) 的求和上限。把它换成全 $T$（MLM 的 $M \equiv 0$ ），归纳步立刻断裂： $o_s$ 会依赖 $x_{>s}$ ，追加 token 将改变所有历史位置的 stream，缓存失效。**可缓存性与因果性是同一个性质的两面**——式 (4) 第二次结账。

一个实现层细节要钉死：RoPE 在投影后按绝对位置旋转 $q_s, k_s$（2.5）。旋转角由位置 $s$ 决定、在写入 cache 之前完成，之后不再改动——cache 里存的是「已旋转」的 $k_s$ ，不可变性不受位置编码影响。

### 4.3 为什么 $q$ 不缓存：依赖结构论证

对称地问： $q_s$ 是否也值得存？看 $q_s$ 的消费方。 $q_s$ 只出现在 score 矩阵的第 $s$ 行—— $s_{s,j} = \langle q_s, k_j \rangle$ ， $j \le s$ 。第 $s$ 行只在第 $s$ 步被需要（计算 $p_{s,\cdot}$ 与 $o_s$ ）；之后的第 $\tau > s$ 步用的是第 $\tau$ 行，query 是 $q_\tau$ 。causal mask 下 score 矩阵只有下三角被使用，而 $q_s$ 恰好只服务第 $s$ 行的下三角段——一次性消费品。反过来 $k_s, v_s$ 服务第 $s$ 列，被所有未来行 $\tau \ge s$ 反复读取。**列被未来读、行只被自己读**：这就是「存 K/V、弃 Q」的依赖结构根源，不是工程习惯。代价也记一笔：若误存 Q，其体量是 $L H_q d_{\mathrm{head}} T B b$ ——式 (9) 的因子 2 已含 K、V 两份，故在 GQA 4:1（ $H_q = 4 H_{kv}$ ）下光 Q 一份就是式 (9) 的 2 倍字节、连 K/V 总量变 3 倍，且零次复用。

### 4.4 有 cache 的 decode：读写模式与 FLOPs

缓存形态：每层两份张量 $K, V \in [B, H_{kv}, t, D]$ ，第 $t$ 步沿 $t$ 维 append 一行、读出全量参与式 (7)。shape 走读（Llama-3.1-8B： $H_q$ = 32、 $H_{kv}$ = 8、 $D$ = 128）：

```
decode 步 t，单层
x_t [B, 1, 4096]
  │  QKV 投影（只算当前 token）
  ├─► q_t [B, 32, 1, 128]                     ← 一次性，用完即弃（§4.3）
  ├─► k_t [B, 8, 1, 128] ──┐ append（沿 t 维拼接）
  └─► v_t [B, 8, 1, 128] ──┘
              cache K [B, 8, t, 128] ──┐ 本步读出全量
              cache V [B, 8, t, 128] ──┘
                     │  s_t = q_t Kᵀ/√D        [B, 32, 1, t]   ← 单行 score
                     │  softmax → p_t          [B, 32, 1, t]
                     ▼  o_t = p_t V            [B, 32, 1, 128]
              concat heads → W^O → [B, 1, 4096]
```

一个由此而来的简化：decode 步不需要 mask——第 $t$ 行的可见集就是 cache 的全部列（ $s \le t$ 是 cache 的定义域），未来列尚不存在。mask 只在 prefill（ $T_p > 1$ 的并行前向）里有工作要做。

**FLOPs 账**。decode 第 $t$ 步：参数项 $2N$（1 个 token 过全部投影与 FFN）；attention 项每层每头 $4td_{\mathrm{head}}$（ $QK^{\top}$ 与 $PV$ 各 $2td_{\mathrm{head}}$ ）， $H_q$ 头合计每层 $4td$ 、 $L$ 层 $4Ltd$ 。生成 $T$ 步的总账：

$$
\mathrm{FLOPs}_{\mathrm{cache}} = \sum_{t=1}^{T} \left(2N + 4Ltd\right) = 2NT + 2Ld\thinspace T(T+1) \approx 2NT + 2LT^{2}d \qquad (8)
$$

**符号**：各项同式 (5)。

与式 (5) 对照：attention 项从 $\frac{2}{3}LT^{3}d$ 降到 $2LT^{2}d$ ，比值约 $T/3$ ——满窗 $T$ = 131072 时约 $4.4 \times 10^{4}$ 倍。更有信息量的对照在训练侧：式 (8) 与一次 teacher-forced 并行前向打 $T$ 个位置的 FLOPs（参数 $2NT$ + attention $2LT^{2}d$ ，1.2 式 (2)(6)）逐项相同——**缓存串行 decode 与并行前向的 FLOPs 账恒等**。cache 买来的不是算力节省，而是「不重算」换来的墙钟时间；代价是串行 $T$ 步、每步读全量 cache，这正是 §6 访存瓶颈的来源。这个恒等式同时是 §7 一致性论证的复杂度注脚。

代码落地（教学用，非优化实现：append 用 `torch.cat` 演示语义；生产实现预分配定长 buffer、按位置写入，避免每步整体拷贝）：

```python
def forward(self, x, cache=None):          # x: [B, T_new, d]；decode 时 T_new = 1
    B, T_new, _ = x.shape
    q = self.W_q(x).view(B, T_new, self.H, self.D).transpose(1, 2)   # [B, H, T_new, D]
    k = self.W_k(x).view(B, T_new, self.H, self.D).transpose(1, 2)
    v = self.W_v(x).view(B, T_new, self.H, self.D).transpose(1, 2)
    if cache is not None:                  # 拼接历史：式 (7) 求和上限 t 的物化
        k = torch.cat([cache[0], k], dim=2)     # [B, H, t, D]
        v = torch.cat([cache[1], v], dim=2)
    cache = (k, v)                           # 写回，供下一步读
    s = q @ k.transpose(-2, -1) * self.D**-0.5  # [B, H, T_new, t]
    if T_new > 1:                            # prefill 才需要 mask；decode 单行全可见
        t_all = k.size(2)
        causal = torch.triu(torch.ones(T_new, t_all, dtype=torch.bool, device=x.device),
                            diagonal=1 + t_all - T_new)   # 局部坐标下的上三角
        s = s.masked_fill(causal, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = p @ v                                # [B, H, T_new, D]
    o = o.transpose(1, 2).reshape(B, T_new, self.H * self.D)
    return self.W_o(o), cache
```

mask 的偏移量值得看一眼：带 cache 的 prefill 块（chunked prefill， $T_{new} > 1$ 且已有 $t - T_{new}$ 列历史）里，局部坐标 $(i, j)$ 对应绝对位置 $(t - T_{new} + i,\ j)$ ，可见条件是 $j \le i + (t - T_{new})$ ——正是 `diagonal = 1 + t_all - T_new` 那一行。纯 prefill（cache 为空、 $t = T_{new}$ ）退化为 2.3 §8 的 `diagonal=1` 。

官方实现把同一结构写成 cache 对象接口：attention 层每次 forward 收到 past key/value 容器，把当前步的 $k_t, v_t$ 更新进去再打分——OLMo 2 的 `Olmo2Attention` 与 `eager_attention_forward`（[HuggingFace, modeling_olmo2.py](../references.md)，[A]）、Llama 的 `LlamaAttention`（[HuggingFace, modeling_llama.py](../references.md)，[A]）都是这个形态，生成循环由 modeling 层之外的 generate 基建驱动。GQA 的组内广播在打分前发生，与 2.3 §8 的 MHA 教学版只差 K/V 的 head 数与复用方式（机制归 4.1）。

## 5. cache size 公式：逐项账与实例

把 §4.4 的缓存形态数成字节。每层的 K、V 各是 $[B, H_{kv}, T, D]$ 个元素，共 $L$ 层，每元素 $b$ 字节：

$$
\mathrm{Mem}_{\mathrm{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_{\mathrm{head}} \cdot T \cdot B \cdot b \text{ bytes} \qquad (9)
$$

逐项（1.2 §8 账三登记的就是这条公式，此处是它的推导落点）：

- **2**：K 与 V 两份，同形同量；
- **$L$**：每层独立一份——第 $\ell$ 层的 $k_s$ 由该层自己的 stream 算出，层间不可共享；
- **$H_{kv}$**：KV 头数。MHA 标准形 $H_{kv} = H_q$（OLMo-2-7B 即此形，[OLMo-2-7B config](../references.md)，[A]）；GQA 取 $H_{kv} < H_q$（Llama-3.1-8B 为 8 对 32，机制与参数账在 4.1）；极端情形 $H_{kv} = 1$ 是 MQA（multi-query attention）——全部 Q 头共享同一对 K/V，其论文标题就是这句账的宣言（[Shazeer, 2019](../references.md)，arXiv:1911.02150，[B]）；
- **$d_{\mathrm{head}}$**：head 维，本章锚点家族一律 128；
- **$T$**：当前序列长。**cache 随生成线性增长**，每步的增量是定长的 $2 L H_{kv} d_{\mathrm{head}} b$ 字节——这是与训练侧 $T^{2}$ scores 内存（2.3 §7）的本质区别：推理从不物化 $T \times T$ 的 score 矩阵，只存随 $T$ 线性的 K/V；
- **$B$**：batch 内序列数。逐序列内容不同，cache 不可共享；
- **$b$**：每元素字节数，bf16/fp16 = 2。cache 量化（int8 / fp8 等）直接把 $b$ 砍半再砍半，属 inference 维度。

**实例账**（bf16、 $B$ = 1、 $T$ 取各模型窗口上限；与 1.2 §8 同口径）：

| 模型 | $L$ | $H_{kv}$ | $d_{\mathrm{head}}$ | $T$ | 每 token 增量 | cache 总量 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 32 | 8 | 128 | 131072 | 128 KiB | 16 GiB |
| Llama-3.1-70B | 80 | 8 | 128 | 131072 | 320 KiB | 40 GiB |
| 8B 反事实 MHA（ $H_{kv}$ = 32） | 32 | 32 | 128 | 131072 | 512 KiB | 64 GiB |
| OLMo-2-7B（MHA 锚点） | 32 | 32 | 128 | 4096 | 512 KiB | 2 GiB |

（[Llama-3.1-8B config](../references.md)、[Llama-3.1-70B config](../references.md)、[OLMo-2-7B config](../references.md)，均 [A]。）全是 2 的幂次的账：8B 一行为 $2 \times 2^{5} \times 2^{3} \times 2^{7} \times 2^{17} \times 2 = 2^{34}$ bytes = 16 GiB。反事实行直接读出 GQA 的收益：同一台 8B 机器若用 MHA，cache 从 16 GiB 涨到 64 GiB——4:1 的 head 分组在 cache 上是精确的 4 倍削减（对照：QKV 投影参数只减 2 倍，不对称性的账见 1.2 练习 1；GQA 机制本身在 4.1）。MQA 再推到底： $H_{kv}$ = 1 时 8B 形态只剩 2 GiB。

读这张表的方式：**cache 与权重是并列的两大显存项**。8B 权重 bf16 约 16 GB（1.2 式 (7)），满窗单序列 cache 16 GiB——一条长上下文请求的 cache 与整个模型的权重同量级。 $B$ 条并发序列把 cache 乘 $B$ 而权重不变： $B$ = 32 时 8B 满窗 cache 达 512 GiB，权重仍是 16 GB——长上下文服务的显存大头是 cache 而不是参数。这就是 4.1（KV 压缩：GQA / MLA 把 $H_{kv} d_{\mathrm{head}}$ 压成低秩）、4.2（稀疏 attention：DSA 只读选中的 cache 块）与 inference 维度（paging、offload、量化）全部围绕式 (9) 做文章的账面根源；MLA 与 DSA 的特例账分别在 4.1 / 4.2 展开，DeepSeek 全系案例在 5.2。

## 6. prefill vs decode：算术强度分野与 roofline 入口

1.2 §10 已把「每 token 前向 $2N$ FLOPs、权重读一遍 $2N$ bytes」放在一起；本节把它接到 prefill / decode 两阶段上，并补上 cache 项。算术强度（arithmetic intensity） $I$ = FLOPs ÷ 访存字节数；屋顶线模型（roofline model）给出的吞吐上限 = $\min($ 峰值算力 $,\ I \times$ 带宽 $)$ ，分界线（ridge point）= 峰值算力 ÷ 带宽，现代加速器的取值在 $10^{2}$ FLOP/byte 量级（逐卡数字与测量方法归 gpu 维度）。

**prefill**： $T_p$ 个 token 一次前向，FLOPs = $2NT_p + 2LT_p^{2}d$ ；权重只从显存读一遍（ $2N$ bytes），激活读写随 $T_p$ 线性、在长 prompt 下被 matmul 摊销：

$$
I_{\mathrm{prefill}} \approx \frac{2NT_p}{2N} = T_p \qquad (10)
$$

prompt 上千 token 时 $I$ 在 $10^{3}$ 量级，远在 ridge point 右侧：compute-bound。prefill 就是「大 matmul」——2.3 §7 的投影项与 attention 项全部以 $T_p$ 为批量维铺开，硬件利用率高；serving 系统把 prefill 当训练类负载调度，原因在这里。

**decode**：每步 1 个 token，FLOPs = $2N + 4Ltd$（式 (8) 的被加项）；访存 = 权重 $2N$ bytes + 全量 cache 读取 $2 L H_{kv} d_{\mathrm{head}} t b$ bytes（式 (7) 的每个 $k_s, v_s$ 都要从显存搬进计算单元）：

$$
I_{\mathrm{decode}}(t) = \frac{2N + 4Ltd}{2N + 2 L H_{kv} d_{\mathrm{head}} t b} \qquad (11)
$$

**符号**：分子分母同为「每生成一个 token」口径； $t$ 为当前已缓存长度。

两个极限。短上下文端 $t \to 0$ ： $I \to 1$ FLOP/byte（分子分母同为 $2N$ ）——1.2 §10 的「decode 强度约等于 1」就是它。长上下文端代入 Llama-3.1-8B、 $t$ = 131072：分子 $1.50 \times 10^{10} + 6.87 \times 10^{10} = 8.4 \times 10^{10}$ FLOPs（参数项 $2N$ 取含输出头的全量口径，同 1.2 §9；attention 项在 $t = N / (2Ld) \approx 2.7 \times 10^{4}$ 处反超层参数项，此处 $N$ 为层参数口径 $6.98 \times 10^{9}$ ，与 1.2 §6 的交叉点 $`T^{*}`$ 同口径、恰为其一半）；分母 $1.61 \times 10^{10} + 1.72 \times 10^{10} = 3.3 \times 10^{10}$ bytes（权重 16 GB 与 cache 16 GiB 各半）； $I \approx 2.5$ 。cache 把强度从 1 抬到 2.5，离 $10^{2}$ 量级的 ridge 仍差近两个数量级：**memory-bound 的结论在整个窗口内成立**，只是「读什么」从纯权重变成权重与 cache 各半。

**batch 的作用与边界**。 $B$ 条序列合成一个 batch：FLOPs 变 $B(2N + 4Ltd)$ ；权重仍只读一遍，cache 逐序列独立故读 $B$ 份：

$$
I_{\mathrm{decode}}(t, B) = \frac{B \left(2N + 4Ltd\right)}{2N + B \cdot 2 L H_{kv} d_{\mathrm{head}} t b} \qquad (12)
$$

$B$ 小时权重项主导分母， $I \approx B$ ——这是 1.2 §10「拼命凑 batch」结论的适用区； $B$ 大到 cache 项主导后，强度饱和于 $(2N + 4Ltd) / (2 L H_{kv} d_{\mathrm{head}} t b)$ ，8B 满窗时这个上界约 $8.4 \times 10^{10} / 1.72 \times 10^{10} \approx 4.9$ 。长上下文 regime 里 batch 对强度的提升被 cache 读取封顶，「只要 cache 放得下就凑 batch」的有效区间在短中端——inference 维度把 continuous batching 与 cache 压缩放在一起优化，账面原因就在式 (12) 的分母结构。

两阶段瓶颈不同，调度与定价因此分离（prefill 按算力计价、decode 按带宽与显存驻留计价），系统展开属 inference 维度；roofline 的逐卡参数属 gpu 维度。本章交付的是式 (10)–(12) 加上「decode 每步必须读全量 cache」这一机制事实——那两个维度全部讨论的入口。

## 7. teacher forcing 与 causality：无泄漏论证

1.1 §4 的 teacher forcing 用一次并行前向打全部 $T$ 个位置的 loss；§3 的生成是串行的。训练-推理一致性需要一个保证：并行前向的第 $t$ 个 logits 不能偷看 $x_{>t}$ 。若 $q_\theta(x_t | \cdot)$ 的条件里混进 $x_t$ 自身，loss 可以抄答案降到 0 而模型什么也没学——训练目标直接退化。

**命题（无泄漏）**。causal mask 式 (3) 下，并行前向满足：对任意 $t$ ， $z_t = f(x_{\le t})$ ，即第 $t$ 个位置的 logits 是前缀的函数；且它与串行 decode 第 $t$ 步（cache 口径）的 logits 是同一数学对象。

**证明**。第一步与 §4.2 同一条归纳，方向从「过去不变」换成「未来不入」：逐层论证 $x_t^{(\ell)}$ 只依赖 $(x_1, \ldots, x_t)$ ——attention 第 $t$ 行的可见集被式 (3) 截断到 $V(t) = \lbrace1, \ldots, t\rbrace$ ，由式 (2)，被掩列的权重精确为 0、贡献为 $0 \cdot v_j = 0$（不是近似）；RMSNorm、FFN、残差加逐位置；归纳到第 $L$ 层，再过 final norm 与输出头（均逐位置），得 $z_t = f(x_{\le t})$ 。第二步对照 decode：§4.2 的不可变性保证 cache 中的 $k_s, v_s$ 与并行前向中位置 $s$ 的对应量相等，式 (7) 在两种执行方式下是同一个和式（浮点求和次序不同，数学对象相同）。∎

两条边界注记。其一，「精确为 0」依赖 mask 的位置与形式：加性 $-\infty$（或哨兵经 max 减法后下溢）在 softmax 之前给精确 0；若某实现把 mask 乘在 softmax 之后、或用有限小负数导致权重非零，泄漏结论要重查。2.3 §8 的教学代码与官方 eager 路径（`eager_attention_forward`）都是 softmax 前加性，与本节一致。其二，无泄漏不等于无训练-推理落差：1.1 §4 登记的暴露偏差（exposure bias）是条件来源不同（真实前缀 vs 模型自采前缀）的分布失配，与信息泄漏是两个问题——前者是训练动力学议题，归 training 维度；本节保证的是架构层面「该看不见的确实看不见」。

至此式 (4) 结完三笔账： $V(i) \subseteq \lbrace1, \ldots, i\rbrace$ 同时是无泄漏的条件（训练侧）、可生成的条件（推理侧，§2）、可缓存的条件（§4.2 归纳的支点）。causality 一个性质，三处结账；后续章节的分工也按此划开——4.1 压缩的是 §5 的字节账，4.3 的线性 attention 放弃显式 cache、把前缀压回固定维状态（即 2.1 §6 的 recurrence 路线换核重生），inference 维度接管 §6 的系统后果。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| mask 加性形式 | $\alpha_{ij} \propto \exp(s_{ij} + M_{ij})$ ，式 (1)(2) | 语义 = 重划 softmax 归一化域；被掩列权重精确 0 |
| causal mask | $M_{ij} = 0\ (j \le i)$ 、 $-\infty\ (j > i)$ ，式 (3) | 对角线恒可见 ⇒ 无空行 NaN；实现用 -inf 或下溢哨兵 |
| 三 mask ↔ 三 objective | §2 表；判定式 (4) | $V(i) \subseteq \lbrace1..i\rbrace$ = 可生成 / 无泄漏 / 可缓存的同一条性质 |
| 生成两阶段 | §3 伪码 | prefill 并行一次 + decode 串行循环；朴素 decode $\approx NT^{2} + \frac{2}{3}LT^{3}d$ ，式 (5) |
| KV 可缓存性 | 前缀不可变命题，式 (6)(7) | $k_s, v_s$ 与未来无关（对层归纳）； $q$ 只服务本行、零复用故不缓存 |
| decode 复杂度 | $2NT + 2LT^{2}d$ ，式 (8) | 与并行前向 FLOPs 恒等；省的是重算与墙钟，不是 FLOPs |
| cache size | $2 L H_{kv} d_{\mathrm{head}} T B b$ ，式 (9) | 8B @128K = 16 GiB；70B = 40 GiB；MHA 反事实 64 GiB；MQA（ $H_{kv}$ = 1）2 GiB |
| 算术强度 | prefill $\approx T_p$ 式 (10)；decode 式 (11)(12) | decode $I \approx 1$–$2.5 \ll$ ridge ⇒ memory-bound；batch 收益被 cache 读取封顶 |
| teacher forcing | 无泄漏命题 | 并行第 $t$ 位 logits ≡ 串行第 $t$ 步；exposure bias 另属 training |

## 练习

1.（推导）把 §4.2 的归纳证明补全为逐算子版本：对 (a) RMSNorm、(b) FFN、(c) 残差加、(d) 带 causal mask 的 attention，分别写出「位置 $s$ 的输出只依赖 $x_{\le s}$ 」的一步论证，并指明哪一步用到了 mask。再把 mask 换成 prefix-LM 的块状 $M$（前缀长 $p$ ），重述命题：哪些位置的 stream 在「续写区继续变长」时保持不变，哪些会变？
2.（计算）Qwen3-8B 口径（ $L$ = 36、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、原生窗口 $T$ = 40960、bf16；参数量按 config 字段约 $8.2 \times 10^{9}$ ）：(a) $B$ = 1 满窗 cache 多少 GiB？(b) $B$ = 32 时 cache 与权重各占多少，一张 80 GB 卡能否同时装下权重与这批 cache？(c) 若该模型改用 MHA（ $H_{kv}$ = 32），(a) 的数字变多少？
3.（计算）decode 强度的 batch 曲线：Llama-3.1-8B、 $t$ = 131072 下用式 (12) 算 $B \in \lbrace1, 4, 16, 64\rbrace$ 的 $I$ ；设 ridge point $R$ = 200 FLOP/byte，存不存在使 decode 转为 compute-bound 的 $B$ ？把 $t$ 换成 4096 重算同一组 $B$ ，结论怎么变？由此解释「凑 batch」的收益随上下文长度衰减。
4.（代码）基于 2.3 §8 的教学 MHA 与本章 §4.4 的 cache 版 forward：(a) 组装一个两层玩具模型（ $d$ = 128、 $H$ = 4），验证「带 mask 的并行前向第 $t$ 位输出」与「带 cache 的逐 token 串行 decode 第 $t$ 步输出」allclose（fp32、容差 $10^{-5}$ ）——§7 命题的可执行形式；(b) 扫 $T \in \lbrace128, 256, 512, 1024\rbrace$ ，对朴素 decode（每步重算前缀）与 cache decode 分别计时，在 log-log 坐标下验证每步成本从随 $t^{2}$ 涨到随 $t$ 涨的斜率变化。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——加性 $-\infty$ mask 出自 §3.2.3
- [Shazeer, 2019](../references.md)：Fast Transformer Decoding: One Write-Head is All You Need，arXiv:1911.02150，[B]（key: mqa）—— $H_{kv} = 1$ 的极端形与 KV cache 的服务侧动机
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [Meta, Llama-3.1-70B config.json](../references.md)：[A]（key: llama-3.1-70b-config）
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——MHA 对照锚点
- [HuggingFace, transformers repo OLMo 2 modeling 代码](../references.md)：[A]（key: olmo2-modeling）——`Olmo2Attention` / `eager_attention_forward` 的 cache 接口
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——`LlamaAttention`
