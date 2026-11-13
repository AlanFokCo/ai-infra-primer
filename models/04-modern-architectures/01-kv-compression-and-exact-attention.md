# 4.1 KV 压缩与精确 Attention 加速（KV Compression and Exact Attention Acceleration）

> 本章处理 decoder 的两笔硬账：2.4 式 (9) 的 KV cache 字节账与 2.3 §7 的 $O(T^{2})$ scores 物化账。前半把 cache 账中 $H_{kv} d_{\mathrm{head}}$ 一项一路压下去——MQA（ $H_{kv} \to 1$）、GQA（ $H_{kv} = H_q / g$）、MLA（低秩 latent + decoupled RoPE，含 RoPE 与低秩压缩冲突的证明与 W_UK / W_UV absorption 的对称推导）；后半把 scores 物化消灭——FlashAttention 的 online softmax 重缩放推导、tiling 流程、精确等价论证与 IO 复杂度对照 | 前置：2.3（attention 公式 (4)、 $O(T^{2})$ 内存账）、2.4（KV cache 式 (9)、prefill / decode、算术强度式 (11)(12)）、2.5（RoPE 主命题式 (13)、partial RoPE §4.7） | 锚点：DeepSeek-V3 config（MLA 维度字段）[A]、Llama-3.1-8B / 70B config（GQA 8 KV heads）[A]、Dao-AILab/flash-attention repo [A]；MQA [B]、GQA [B]、DeepSeek-V2 / V3 [A]、FlashAttention / FA2 [A]

## 1. 两笔账，两条战线

2.4 §5 的实例账：Llama-3.1-8B 满窗 131072 token 单序列 KV cache 16 GiB，与 bf16 权重同量级； $B$ = 32 并发即 512 GiB，权重仍是 16 GB。cache 公式 $\mathrm{bytes} = 2 L H_{kv} d_{\mathrm{head}} T B b$（2.4 式 (9)）里， $L$ 由架构给定， $T$、 $B$ 由负载给定， $b$ 的量化归 inference 维度——本章动的是 $H_{kv} d_{\mathrm{head}}$：每 token 每层必须存下的「键值信息总量」。MQA 与 GQA 直接砍头数；MLA 换一条思路——头数不动，把每个 token 的 KV 内容联合压进一个低秩 latent 向量，cache 只存 latent。

另一条战线在训练与 prefill 侧。2.3 §7 的账：朴素 attention 物化 $[B, H, T, T]$ 的 scores 与 probs 两张矩阵， $T$ = 131072 时单序列单层 2 TiB，任何单卡装不下。这笔账与 cache 无关（decode 从不物化 $T \times T$），卡的是训练与 prefill 的内存墙。FlashAttention 的答案不改数学——输出与标准 attention 逐点相同——而是改执行：分块 + online softmax，scores 永不落 HBM。

两条战线正交：KV 压缩改「存什么」，FlashAttention 改「怎么算」，可以叠加（DeepSeek-V3 即 MLA 与 FlashAttention 的组合）。§2–§4 走第一条线（MQA → GQA → MLA），§5 走第二条。

## 2. MQA： $H_{kv} = 1$ 的极端形

MHA 标准形每头一对 K/V 投影（2.3 式 (11)）。MQA（multi-query attention，[Shazeer, 2019](../references.md)，arXiv:1911.02150，[B]）把它推到底：全部 $H_q$ 个 query 头共享同一对 K/V：

$$
q_{t,h} = W_h^{Q} u_t, \qquad k_t = W^{K} u_t, \qquad v_t = W^{V} u_t \qquad (1)
$$

$$
o_{t,h} = \sum_{s=1}^{t} p_{t,s,h} \thinspace v_s, \qquad p_{t,s,h} = \frac{\exp\left(\langle q_{t,h}, k_s \rangle / \sqrt{d_{\mathrm{head}}}\right)}{\sum_{j=1}^{t} \exp\left(\langle q_{t,h}, k_j \rangle / \sqrt{d_{\mathrm{head}}}\right)} \qquad (2)
$$

**符号**： $u_t \in \mathbb{R}^{d}$：层输入的归一化后 stream（2.4 式 (6) 的量；2.4 用行向量约定 $u W$，本章起统一列向量约定 $W u$，两者转置等价）； $W_h^{Q} \in \mathbb{R}^{d_{\mathrm{head}} \times d}$：第 $h$ 个 query 头的投影，共 $H_q$ 块； $W^{K}, W^{V} \in \mathbb{R}^{d_{\mathrm{head}} \times d}$：**全局各一块**——MHA 里的逐头 $W_h^{K}, W_h^{V}$ 被合并成单份，K/V 投影参数从各 $d^{2}$ 缩到各 $d \cdot d_{\mathrm{head}}$（8B 形态缩 32 倍）； $k_t, v_t \in \mathbb{R}^{d_{\mathrm{head}}}$ 被全部头共享。

cache 账直接代公式： $H_{kv} = 1$ 进 2.4 式 (9)，每 token 每层 $2 d_{\mathrm{head}} b$ 字节，与 $H_q$ 无关。Llama-3.1-8B 形态（ $L$ = 32、 $d_{\mathrm{head}}$ = 128、bf16）每 token 增量从 GQA 的 128 KiB 降到 16 KiB（8 倍）、满窗 cache 从 16 GiB 降到 2 GiB；对照该章的反事实 MHA 行（512 KiB / 64 GiB）是 32 倍。

decode 侧还有一笔账。2.4 式 (12) 的分母 = 权重读一遍 + 每序列一份 cache 读取； $H_{kv}$ = 1 把 cache 项压到 GQA-8 的 1/8、MHA 的 1/32。代数字走一遍（8B 形态、 $t$ = 131072）：单序列 cache 2 GiB， $B$ = 32 时 batch 的 cache 读取 64 GiB 对权重 16 GB——cache 项仍在分母里，但 batch 饱和强度上界（2.4 §6 对 GQA-8 算出的约 4.9）外推 8 倍。饱和点仍低于 ridge（ $10^{2}$ 量级），memory-bound 不变，但「凑 batch 还有用」的区间被拉长了同一个倍数。MQA 的原始动机正是这个服务侧数字——论文标题就是「one write-head」（[Shazeer, 2019](../references.md)，[B]）；质量是后账。

代价写在机制里。2.3 §2：score 经 rank-$d_k$ 双线性型读 stream（列向量约定下 $A_h = (W_h^{Q})^{\top} W_h^{K} \in \mathbb{R}^{d \times d}$，rank ≤ $d_{\mathrm{head}}$），MHA 是 $H$ 个独立低秩透镜并排。MQA 不是减少透镜个数，而是强制全部透镜共用同一片 K 侧玻璃： $A_h^{MQA} = (W_h^{Q})^{\top} W^{K}$， $H_q$ 个头的 key 侧行空间被钉在同一组基上——检索分辨率塌缩到一份。value 侧同样：整条序列只有一条 $d_{\mathrm{head}}$ 维的交付通道， $H_q$ 个头各自加权求和的是同一些 $v_s$——交付带宽也塌缩到一份。MQA 论文报告的质量差距（[B]，无官方代码，按论文口径引）就是这两处塌缩的代价；§3 的 GQA 是「塌缩程度可调」的修复。

一个对照注记为 §4 铺垫：MQA 与 GQA 同 RoPE 无冲突——压缩的是头数，每个幸存的 K/V 头仍逐位置物化、照常旋转（2.5 §4.3 的应用位置不变，cache 里存的是已旋转的共享 $k_n$）。冲突只在「压缩每个 token 的 KV 内容本身」时出现——那是 MLA 要面对的，§4.1 的推导就从这里开始。

## 3. GQA：组内共享、uptraining 与 Pareto 前沿

GQA（grouped-query attention，[Ainslie et al., 2023](../references.md)，arXiv:2305.13245，[B]）把 MQA 的极端放松成一个旋钮： $H_q$ 个 query 头每 $g$ 个编成一组，组内共享一对 K/V，共 $H_{kv}$ 组：

$$
H_{kv} = H_q / g \qquad (3)
$$

$g$ = 1 退回 MHA（每组独占 K/V）， $g = H_q$ 即 MQA（全体一组）。GQA 论文的记号 GQA-$N$ 指 $N$ 个 KV 头，即 $N = H_{kv}$。三种形态的 K/V 分配（ $H_q$ = 32、 $g$ = 4 即 Llama-3.1-8B 形态）：

```
MHA (g = 1)              GQA (g = 4)                 MQA (g = 32)
Q  h1  h2  h3  h4        Q  h1 h2 h3 h4  h5..h8      Q  h1 h2 ... h32
   │   │   │   │            └────┬────┘  └──┬──          └──────┬──────┘
K  k1  k2  k3  k4        K      k1         k2   ...    K        k1（独一份）
V  v1  v2  v3  v4        V      v1         v2   ...    V        v1（独一份）
H_kv = 32                H_kv = 8                    H_kv = 1
```

cache 账：式 (3) 进 2.4 式 (9)， $\mathrm{bytes} = 2 L (H_q / g) d_{\mathrm{head}} T B b$——GQA-4 对 8B 形态就是 2.4 §5 的实例行（16 GiB 满窗）。参数账则不对称：K/V 两块投影缩 $g$ 倍，Q 与 O 不动（Llama-3.1-8B 的 attention 参数 $2.5 d^{2}$ vs MHA 标准形 $4 d^{2}$，2.1 §4 的口径）——GQA 省的几乎只有 cache 与其带宽，不是参数、也不是训练 FLOPs。这也是为什么它的质量损失远小于 MQA： $H_{kv}$ = 8 时 K 侧行空间从 1 份回到 8 份，检索分辨率只比 MHA 少 4 倍而不是 32 倍。

实现侧一句：cache 只按 $H_{kv}$ 份存储与回读，组内广播发生在打分前的视图层——K/V 从 $[B, H_{kv}, t, D]$ 扩视到 $[B, H_q, t, D]$，扩视不产生新字节；eager 路径显式 repeat，SDPA / FlashAttention 路径在 kernel 内广播（Llama modeling 代码的 K/V 复用即此形，[HuggingFace, modeling_llama.py](../references.md)，[A]）。广播同样出现在 2.4 §4.4 的 shape 走读里（ $s_t$ 是 $[B, 32, 1, t]$ 对 cache $[B, 8, t, 128]$ 的组内匹配）。注意广播不改账：省的是存储与带宽，打分 FLOPs 仍按 $H_q$ 个 query 头全付。

**uptraining**。GQA 论文给的迁移配方（[B]，按论文引）：从已训好的 MHA checkpoint 出发，把每组内各头的 $W_h^{K}$（同理 $W_h^{V}$）做 mean pooling 得到该组共享 K/V 投影的初始化，再用约原预训练算力 5% 的预算继续训练。初始化为什么取 mean：论文对照过三种取法——组内 mean pooling、取组内第一头、随机初始化——mean pooling 最优（[B]）；机制直观是各头的 $W_h^{K}$ 行空间方向不同，任取一头等于把组内其余头的检索结构硬对齐到该头，而 mean 保持组内方向的质心——零阶扰动最小。随后 5% 预算的继续训练把 pooled 权重与网络其余部分重新校准。这条路径的意义是 GQA 可以 retro-fit 到存量 MHA checkpoint 上，不必从头训。

**Pareto 数据形态**（论文在 T5 族各规模上的 uptraining 对照，[B]）：MQA 的 decode 最快、质量可见下降；GQA-8（8 KV 头）质量与 MHA 基线几乎重合，decode 速度接近 MQA——论文据此把 GQA-8 作为中间点选择（质量对照见 Table 1 / Fig 5，速度对照见 Fig 6）。注意证据边界：论文逐点给的是 MHA / GQA-8 / MQA 三个臂，「再向左（4、2、1）质量快速下滑」的悬崖形状是本书按 §2 塌缩机制做的外推标注，非论文数据。账侧读法：decode 每步的 cache 读取正比于 $H_{kv}$（2.4 式 (11) 的分母），从 8 压到 1 只再省 8 倍读取，而质量从「与 MHA 几乎重合」跳到「可见下降」（MQA 臂，论文口径）——读取收益线性、质量代价非线性，中间点因此合理。落地坐标与之一致：Llama-3.1-8B（ $H_q$ = 32、 $H_{kv}$ = 8， $g$ = 4）与 Llama-3.1-70B（ $H_q$ = 64、 $H_{kv}$ = 8， $g$ = 8）（[Meta, Llama-3.1-8B config](../references.md)、[Meta, Llama-3.1-70B config](../references.md)，均 [A]）——两家都把旋钮停在 $H_{kv}$ = 8。

## 4. MLA：低秩 latent + decoupled RoPE 的完整推导

### 4.1 动机：RoPE 与低秩压缩的数学冲突

GQA 压缩的是头数，每个被保留的 K/V 头仍是全尺寸 $d_{\mathrm{head}}$ 维。头数压缩有一块地板： $H_{kv}$ 是不小于 1 的整数，且 §3 的现实坐标是各家停在 $H_{kv}$ = 8——更左的 MQA 端点有可见质量代价（论文口径）——这条路线在「8 倍削减」附近已经走到尽头，要再压一个数量级只能压内容本身：不存 $k_s, v_s$（合计 $2 H_{kv} d_{\mathrm{head}}$ 元素），存一个低秩 latent $c_s \in \mathbb{R}^{d_c}$、 $d_c \ll 2 H_{kv} d_{\mathrm{head}}$，用的时候经上投影 $k_s = W^{UK} c_s$、 $v_s = W^{UV} c_s$ 展开。没有位置编码时这条路畅通——§4.4 会看到全部上投影都能被吸收，cache 只有 $d_c$。撞上 RoPE 就断了。

2.5 §4.3 的应用接缝： $k_n$ 在写 cache 之前按绝对位置旋转，score 为 $\langle R_m q_m, R_n k_n \rangle$。若 cache 只存压缩态 $c_n$，则 decode 第 $t$ 步需要的对象是

$$
s_{t,n} = \left\langle R_t q_t,\hspace{5mu} R_n \thinspace W^{UK} c_n \right\rangle = \left\langle q_t,\hspace{5mu} R_{n-t} \thinspace W^{UK} c_n \right\rangle \qquad (4)
$$

第二步用了 2.5 式 (13) 的内积不变性。 $R_{n-t}$ 卡在 $W^{UK}$ 与 $c_n$ 之间，且逐位置对不同。三条出路逐一封死：

- **缓存展开并旋转后的 $k_n$**： $k_n = R_n W^{UK} c_n$ 是 $H_{kv} d_{\mathrm{head}}$ 全尺寸——压缩白做；
- **把旋转预先折进压缩矩阵**：需要 $`\tilde{W}^{UK}_n = R_n W^{UK}`$ 逐位置一份，cache 里一份 $c_n$ 无法服务全部未来时刻；
- **让旋转与压缩换位**：若 $R_\Delta W^{UK} = W^{UK} R_\Delta$ 对全部 $\Delta$ 成立，则可先旋转 latent 再展开，cache 存 $R_n c_n$ 即可。但对全部 $\Delta$ 换位迫使 $W^{UK}$ 保持 RoPE 的每个 2D 频率块互不混合（块对角的 scaled-rotation 形式）——而压缩矩阵的全部意义是跨通道全混合（ $d_c$ 维之间任意重组），二者结构性不相容，一般地 $R_n W^{UK} \neq W^{UK} R_n$。

把第三条的交换子说透。块对角旋转族 $\lbrace R_\Delta : \Delta \in \mathbb{R}\rbrace$ 中，各块的角速度 $\theta_j$ 互不相同， $\Delta$ 变化时各块角度独立取值；与全部 $R_\Delta$ 交换的矩阵只能是逐块的 scaled rotation（第 $j$ 块上形如 $a_j I + b_j J$ 的 2×2 矩阵）——跨频率块的混合一概被排除。而 $W^{UK}$ 的工作恰恰是从 $d_c$ = 512 维 latent 跨通道重组出每头 128 维 key，是全混合映射：在交换子内则压缩不成立，做压缩则不在交换子内。

三条全死：低秩压缩要求 $k$ 是 $c$ 的固定线性像，RoPE 要求 $k$ 携带逐位置旋转——同一条 $k$ 上两个要求不可兼得。反过来，若位置根本不经旋转注入（NoPE，2.5 §5），朴素低秩畅通无阻——4.4 会见到 Kimi-Linear 的 full-attn 层正是「MLA + NoPE」的实例；本节要推导的 decoupled 结构，是为保住 RoPE 付出的设计税。这就是 MLA 的结构出发点（[DeepSeek-AI, 2024](../references.md)，arXiv:2405.04434，[A]，DeepSeek-V2；[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，V3 沿用同一结构）。

### 4.2 解法：拆开，不交换

MLA 的回答是把 key 拆成两份，各满足一个要求：**内容部分**走低秩压缩、全程不接触旋转（压缩态可缓存）；**位置部分**单独走一条小维通道、旋转后原样缓存（不压缩）。这就是 decoupled RoPE——2.5 §4.7 已从 partial RoPE 一侧预告过这条接缝：「decoupled 是 absorption 的必要条件」。完整构造（DeepSeek-V3 维度，config [A]；记号 $d_c$ = kv_lora_rank = 512、 $d_h$ = qk_nope_head_dim = 128、 $d_R$ = qk_rope_head_dim = 64、 $d_v$ = v_head_dim = 128、 $H$ = 128 heads）：

KV 联合低秩压缩——K 与 V 共用一份 latent：

$$
c_t^{KV} = W^{DKV} u_t \in \mathbb{R}^{d_c} \qquad (5)
$$

**符号**： $W^{DKV} \in \mathbb{R}^{d_c \times d}$：下投影； $u_t \in \mathbb{R}^{d}$：层输入 stream。注意与 2.4 式 (9) 的因子 2 对照：K、V 不再各存一份，一份 latent 同时是两者的源。

逐头上投影（用时展开，可以不展开——§4.3）：

$$
k_{t,h}^{C} = W_h^{UK} c_t^{KV} \in \mathbb{R}^{d_h}, \qquad v_{t,h}^{C} = W_h^{UV} c_t^{KV} \in \mathbb{R}^{d_v} \qquad (6)
$$

**符号**： $W_h^{UK} \in \mathbb{R}^{d_h \times d_c}$、 $W_h^{UV} \in \mathbb{R}^{d_v \times d_c}$：第 $h$ 头的 K/V 上投影。上标 $C$ 标记 content（位置盲）部分。

decoupled RoPE 通道——单独小维、旋转后缓存、全头共享：

$$
k_t^{R} = R_t \left( W^{KR} u_t \right) \in \mathbb{R}^{d_R} \qquad (7)
$$

**符号**： $W^{KR} \in \mathbb{R}^{d_R \times d}$： $k^R$ 的独立投影（不从 $c^{KV}$ 出）； $R_t$：2.5 式 (12) 的块对角旋转（ $d_R / 2$ = 32 个频率通道）。 $k^R$ 只有 64 维且全头共享——位置信号本身是头无关对象（位移 $t - s$ 对全部头同一含义，2.5 式 (13)），逐头各备一份会把 cache 项 2 放大 $H$ 倍而不加位置分辨力；共享是 cache 侧的自然取法。query 侧的 $q^{R}$ 则逐头各一份（式 (8)）——q 不占 cache，逐头只花参数与 FLOPs，买的是逐头各异的位置探针。这一不对称（k 共享、q 逐头）与 2.4 §4.3「列被未来读、行只被自己读」的依赖结构同形。

query 侧对称构造（同样低秩，V3 config q_lora_rank = 1536，记 $d_c'$）：

$$
c_t^{Q} = W^{DQ} u_t \in \mathbb{R}^{d_c'}, \qquad q_{t,h}^{C} = W_h^{UQ} c_t^{Q} \in \mathbb{R}^{d_h}, \qquad q_{t,h}^{R} = R_t \left( W_h^{QR} c_t^{Q} \right) \in \mathbb{R}^{d_R} \qquad (8)
$$

query 的低秩与 cache 无关（q 从不缓存，2.4 §4.3）——它省的是参数与 FLOPs，不是字节。拼合与打分（逐头 $q_{t,h} = [q_{t,h}^{C};\thinspace q_{t,h}^{R}]$、 $k_{s,h} = [k_{s,h}^{C};\thinspace k_s^{R}]$，维度 $d_h + d_R$ = 192）：

$$
s_{t,s,h} = \frac{\left\langle q_{t,h}^{C},\thinspace k_{s,h}^{C} \right\rangle + \left\langle q_{t,h}^{R},\thinspace k_s^{R} \right\rangle}{\sqrt{d_h + d_R}} \qquad (9)
$$

$$
o_{t,h} = \sum_{s=1}^{t} p_{t,s,h} \thinspace v_{s,h}^{C}, \qquad o_t = W^{O} \thinspace \operatorname{concat}_h \left( o_{t,h} \right) \qquad (10)
$$

**符号**： $p_{t,s,h}$：式 (9) 的 score 经 causal mask 与逐行 softmax 后的权重（2.4 式 (7) 同口径）； $`\operatorname{concat}_h`$：沿头维拼接， $o_t \in \mathbb{R}^{d}$； $W^{O} \in \mathbb{R}^{d \times H d_v}$。

式 (9) 即 2.5 式 (17) 的 partial RoPE 分解：128 维 nope 内积位置盲、64 维 rope 内积只含 $t - s$——§4.1 的冲突被「拆开」绕开：压缩通道上的 $k^{C}$ 从头到尾不旋转，旋转通道上的 $k^{R}$ 从头到尾不压缩。

数据流（V3 维度，逐层逐 token；cache 用 ═ 标出）：

```
u_t [d = 7168]（RMSNorm 后的 stream，2.4 式 (6) 口径）
 │        query 支路（每步现算，不缓存——2.4 §4.3）
 ├─ W^DQ ──► c_t^Q [1536] ──┬── W_h^UQ ──► q^C_{t,h} [128] ─┐
 │                          └── W_h^QR ──► R_t ──► q^R_{t,h} [64] ┐
 │        KV 支路                                                │  │
 ├─ W^DKV ──► c_t^KV [512] ══╗ cache 项 1                       │  │
 │              ║  W_h^UK / W_h^UV 挂在其上：                    │  │
 │              ║  训练/prefill 展开 k^C, v^C；decode 吸收（§4.3-4.4）│
 └─ W^KR ──► R_t ──► k_t^R [64] ══╝ cache 项 2（全头共享） ◄──────┘
        score = (⟨q^C, k^C⟩ + ⟨q^R, k^R⟩) / √192 → softmax → o
```

参数账也记一笔（由 V3 config 字段推算，[A]； $H d_h$ = $H d_v$ = 16384）： $W^{DQ}$ 7168×1536 ≈ 11.0M、 $W^{UQ}$ 1536×16384 ≈ 25.2M、 $W^{QR}$ 1536×8192 ≈ 12.6M、 $W^{DKV}$ 7168×512 ≈ 3.7M、 $W^{UK}$ 与 $W^{UV}$ 各 512×16384 ≈ 8.4M、 $W^{KR}$ 7168×64 ≈ 0.5M、 $W^{O}$ 16384×7168 ≈ 117.4M——单层合计 ≈ 187M。对照同头数 MHA 反事实（ $W^{Q}, W^{K}, W^{V}, W^{O}$ 各 7168×16384 ≈ 117.4M，合计 ≈ 470M）：MLA 的 attention 参数约为 MHA 的 40%，61 层累计 11.4B vs 28.7B。对 671B 总参数两者都是小头（MoE FFN 主导，4.5），但方向值得登记：MLA 同时省 cache 与参数，不是「拿参数换 cache」。

**cache 账**。每 token 每层只存式 (5) 与式 (7)： $d_c + d_R$ = 512 + 64 = **576 元素**（bf16 = 1152 字节）。对照（每 token 每层元素数，K+V 合计）：

| 方案 | 元素 / token / 层 | 相对 MHA-32 |
| --- | --- | --- |
| MHA（ $H_{kv}$ = 32，反事实 8B 形） | $2 \times 32 \times 128$ = 8192 | 1× |
| GQA-4（ $H_{kv}$ = 8） | $2 \times 8 \times 128$ = 2048 | 4× |
| MLA（512 + 64） | 576 | 14.2× |
| MQA（ $H_{kv}$ = 1） | $2 \times 128$ = 256 | 32× |

整模型实数（bf16）：DeepSeek-V3 共 61 层，每 token 70,272 字节 ≈ 68.6 KiB；满窗 $T$ = 163840 单序列 cache ≈ 10.7 GiB。反事实对照：V3 若用 MHA（ $H$ = 128、 $d_{\mathrm{head}}$ = 128），每 token 每层 32768 元素、满窗 ≈ 610 GiB——576 对 32768，56.9 倍。表也标出 MLA 的定位：cache 上介于 GQA-4 与 MQA 之间（576 vs 2048 vs 256）。

信息视角收尾：8192 → 576 的 14 倍压缩为什么不丢质量。MHA 的 $H$ 份 K/V 全部投影自同一条 stream，跨头相关性高，逐头 128 维合起来的有效秩远小于 $2 H d_{\mathrm{head}}$；MLA 的 512 维 latent 是对这份冗余的显式低秩利用。且 512 > 2 × 128（单头 K+V 合计 256 维）——latent 比任何单头的 KV 都宽：MLA 不是把 KV 压到比单头还小，是把 $H$ 份高相关副本换回一份共享的 512 维表示，再让每头经自己的 $W_h^{UK}$ 从中取 128 维私有截面。检索结构（头数、透镜数）一个没少——这与 §2 MQA 的「份额塌缩」是两种压法，质量差异（V2 论文报告 MLA 与同规模 MHA 相当或更强，[A]）的根源在此。

### 4.3 W_UK absorption：把上投影吸进 query

式 (5)–(10) 是「数学形态」。decode 的执行形态把两个上投影离线吸掉，attention 本体直接在 latent 空间跑。先看 K 侧。式 (9) 的内容项展开式 (6)(8)：

$$
\left\langle q_{t,h}^{C},\thinspace k_{s,h}^{C} \right\rangle = \left( W_h^{UQ} c_t^{Q} \right)^{\top} W_h^{UK} c_s^{KV} = \left( c_t^{Q} \right)^{\top} \underbrace{\left( W_h^{UQ} \right)^{\top} W_h^{UK}}_{\tilde{W}_h \in \mathbb{R}^{d_c' \times d_c}} c_s^{KV} \qquad (11)
$$

**符号**： $`\tilde{W}_h = (W_h^{UQ})^{\top} W_h^{UK}`$：第 $h$ 头的合并算子，离线预计算，无逐位置量。定义吸收后 query $`\tilde{q}_{t,h} = \tilde{W}_h \thinspace c_t^{Q} \in \mathbb{R}^{d_c}`$，score 变成

$$
s_{t,s,h} = \frac{\left\langle \tilde{q}_{t,h},\thinspace c_s^{KV} \right\rangle + \left\langle q_{t,h}^{R},\thinspace k_s^{R} \right\rangle}{\sqrt{d_h + d_R}} \qquad (12)
$$

$k^{C}$ 从计算图中消失：score 直接打在 cached latent $c_s^{KV}$ 上。decode 每步每 token 每层读的就是 576 元素，而不是展开后的全量 KV—— $2 H d_h + d_R$ = 2 × 128 × 128 + 64 = 32832 元素（ $H$ 份 $k^{C}$、 $H$ 份 $v^{C}$ 与共享 $k^{R}$）——读取削减 57 倍。2.4 §6 的口径下（decode $I \approx 1$–2.5、ridge $10^{2}$ 量级，memory-bound），这笔账直接换算成墙钟。

合并算子还有一个结构注记： $`\tilde{W}_h = (W_h^{UQ})^{\top} W_h^{UK} \in \mathbb{R}^{d_c' \times d_c}`$ 的秩 ≤ $d_h$ = 128——吸收形下每头对 latent 的 score 仍是一个 rank-128 双线性型，与 2.3 §2「score 经 rank-$d_k$ 透镜读 stream」逐项对应；变化只是透镜的观测对象从逐头独立的 $k_s$ 换成了共享 latent $c_s^{KV}$。检索分辨率不因吸收而变——吸收是执行层变换，数学对象（式 (11) 两端）逐点相等。

### 4.4 W_UV absorption：对称的一笔

V 侧同构。把加权和提到上投影之前（式 (10) 第一式代入式 (6)）：

$$
o_{t,h} = \sum_{s=1}^{t} p_{t,s,h} \thinspace W_h^{UV} c_s^{KV} = W_h^{UV} \underbrace{\sum_{s=1}^{t} p_{t,s,h} \thinspace c_s^{KV}}_{m_{t,h} \in \mathbb{R}^{d_c}} \qquad (13)
$$

线性性允许交换：先在 latent 空间做加权和得 $m_{t,h}$，再一次性上投影。代入输出投影 $o_t = \sum_h W_h^{O} o_{t,h}$（ $W^{O} = [W_1^{O} \cdots W_H^{O}]$ 按头分块， $W_h^{O} \in \mathbb{R}^{d \times d_v}$）：

$$
o_t = \sum_{h=1}^{H} W_h^{O} W_h^{UV} m_{t,h} = \sum_{h=1}^{H} \widehat{W}_h \thinspace m_{t,h}, \qquad \widehat{W}_h = W_h^{O} W_h^{UV} \in \mathbb{R}^{d \times d_c} \qquad (14)
$$

$`\widehat{W}_h`$ 同样离线预计算—— $W_h^{UV}$ 被吸进 $W^{O}$。两笔吸收完全对称：K 侧吸进 Q（式 (11)(12)），V 侧吸进 O（式 (13)(14)），合起来 attention 本体全部在 $d_c$ = 512 维 latent 空间执行——score 对 $c$ 打、加权和加 $c$、两头出入各过一次合并矩阵。两种执行形态并排：

| | 展开形（训练 / prefill） | 吸收形（decode） |
| --- | --- | --- |
| K / V | 物化 $k^{C}$、 $v^{C}$（每头 128 维） | 不物化，score 直接打在 $c_s^{KV}$ 上 |
| score 内维 | $d_h + d_R$ = 192 | $d_c + d_R$ = 576 |
| 每 token 每层 KV 读取 | $2 H d_h + d_R$ = 32832 元素 | $d_c + d_R$ = 576 元素 |
| regime | compute-bound，大 matmul 友好 | memory-bound，读取 57× 削减 |

**代价与边界**（诚实账三笔）。(a) 吸收后 query 有效维数从 $d_h + d_R$ = 192 涨到 $d_c + d_R$ = 576：score 的逐对 FLOPs 涨 3 倍。decode 是 memory-bound，多付 FLOPs 换 57 倍读取削减是净赚；这笔交换在 compute-bound 侧不成立。(b) 因此 prefill 与训练用**非吸收形**——先把 $K^{C}, V^{C}$ 展开成 192 / 128 维再正常算（大 batch matmul、 $d_c$ 不进 $QK^{\top}$ 的内维），decode 用吸收形；同一份权重、两张执行图。部署侧的落法是逐阶段切换： $`\tilde{W}_h`$ 与 $`\widehat{W}_h`$ 在加载时一次性预计算，prefill 走展开形、首个 decode 步起切吸收形，切换本身零成本。(c) 吸收只覆盖 nope 部分； $k^{R}$ 仍逐位置旋转、原样缓存——若当初选择全维旋转，式 (11) 里 $R_n$ 会卡在 $`\tilde{W}_h`$ 与 $c_s^{KV}$ 之间（§4.1 式 (4)），吸收当场死亡。顺带一句 MLA 与 §5 的关系：kernel 层完全不感知 MLA——展开形下它就是普通 attention（head 维 192），FlashAttention 原样可用；吸收形的 kernel 化是推理框架的事，归 inference 维度。DeepSeek 官方推理实现即按吸收形落地（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）；Kimi-Linear 的 full-attn 层给出同一结构的 NoPE 变体（无旋转则无需 decoupled），归 4.4。

## 5. FlashAttention：不物化 scores 的精确 attention

### 5.1 瓶颈是 IO，不是 FLOPs

2.3 §7 的内存账再数一遍（单层、 $B$ = 1、 $H$ 头合并、bf16）：朴素实现把 score 矩阵 $S$ 与概率矩阵 $P$ 各物化一份 $[H, T, T]$—— $T$ = 131072 时 2 TiB，且训练反向要用，不能算完即弃。物化的路径是： $Q K^{\top}$ 在片上算出 $S$ → 写 HBM → 读回做 softmax → $P$ 写回 HBM → 再读回算 $P V$。硬件侧的落差（FlashAttention 论文 §2 的口径，[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]）：A100 的 HBM 40–80 GB、带宽 1.5–2.0 TB/s；每 SM 片上 SRAM 192 KB、聚合带宽约 19 TB/s——容量差约五个数量级、带宽差约一个数量级。attention 本体的 matmul FLOPs 是 $\Theta(T^{2} d)$、HBM 流量是 $\Theta(T^{2})$ 元素，算术强度 $\Theta(d)$ ≈ 128 对 ridge $10^{2}$ 只勉强压线，再被 softmax、mask、dropout 这些逐元素非 matmul 操作的往返读写一拖，实测落在 memory-bound 区。

朴素实现的执行形态是三个 kernel 接力：matmul 产 $S$、softmax 逐行（含 mask、可能还有 dropout）、matmul 产 $O$——每次接力都经 HBM 中转， $T^{2}$ 对象被写出再读回共四趟。思路因此是：把 $Q, K, V$ 切成能驻留 SRAM 的块，全部中间结果留在片上，HBM 只读写 $Q, K, V, O$ 本身。障碍只有一个——softmax 的行归一化需要整行 score 的最大值与指数和（2.3 式 (10) 的 max 减法），分块后两者都不可得。online softmax 就是绕开它的数学工具。

### 5.2 online softmax：重缩放推导

设某行 logit 按块依次到达 $B_1, B_2, \ldots$（每块 $B_c$ 个 logit），记前 $j$ 块的并集为 $U_j$。维护两个 running 统计量，处理到第 $j$ 块时更新：

$$
m^{(j)} = \max\left( m^{(j-1)},\ \max_{s \in B_j} x_s \right), \qquad \ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \thinspace \ell^{(j-1)} + \sum_{s \in B_j} e^{x_s - m^{(j)}} \qquad (15)
$$

初始 $m^{(0)} = -\infty$、 $\ell^{(0)} = 0$（ $e^{-\infty}$ = 0，基例自动成立）。

**命题（不变量）**：处理完前 $j$ 块， $m^{(j)}$ 是已见 logit 的全局最大值，且 $\ell^{(j)} = \sum_{s \in U_j} e^{x_s - m^{(j)}}$——以当前全局 max 为基准的指数和。

**证明**（对 $j$ 归纳）。基例 $j$ = 1： $m^{(1)}$ 为块内 max， $\ell^{(1)} = \sum_{s \in B_1} e^{x_s - m^{(1)}}$，平凡。归纳步：设 $j - 1$ 成立，记 $U_{j-1}$ 为旧集合、 $B_j$ 为新块。max 的合并由定义即全局 max。对和式，第一项：

$$
e^{m^{(j-1)} - m^{(j)}} \thinspace \ell^{(j-1)} = e^{m^{(j-1)} - m^{(j)}} \sum_{s \in U_{j-1}} e^{x_s - m^{(j-1)}} = \sum_{s \in U_{j-1}} e^{x_s - m^{(j-1)} + m^{(j-1)} - m^{(j)}} = \sum_{s \in U_{j-1}} e^{x_s - m^{(j)}}
$$

即把旧和从旧基准 $m^{(j-1)}$ 平移到新基准 $m^{(j)}$——指数里新旧基准之差相互抵消，逐项不变。第二项是新块在新基准下的指数和。两项相加即 $U_j = U_{j-1} \cup B_j$ 上以 $m^{(j)}$ 为基准的指数和。∎

数值走一遍。设一行 logit 为 $x = (1, 2, 5, -1)$，分两块到达 $(1, 2)$ 与 $(5, -1)$。第一块： $m^{(1)}$ = 2， $\ell^{(1)} = e^{-1} + e^{0} \approx 1.368$。第二块：块内 max 为 5， $m^{(2)}$ = 5，重缩放因子 $e^{2-5} \approx 0.0498$， $\ell^{(2)} = 0.0498 \times 1.368 + e^{0} + e^{-6} \approx 0.0681 + 1.0025 = 1.0706$。直接以全局 max 5 为基准验证： $e^{-4} + e^{-3} + e^{0} + e^{-6} \approx 0.0183 + 0.0498 + 1 + 0.0025 = 1.0706$——逐位相同；分块只是多付了一次「换基准」，和式不变。 $O^{(j)}$ 的走法同构，只是把标量 $e^{x_s - m}$ 换成 $e^{x_s - m} v_s$。

重缩放因子 $e^{m_{old} - m_{new}} \le 1$ 只做「换基准」，每一步都是恒等变换，不引入任何近似。一个数值性质顺带成立：所有指数的自变量都不超过 0（当前基准恒为已见 max）， $e^{(\cdot)} \le 1$，overflow 在构造上不可能——2.3 §4 的 max 减法以防溢出形式被分块继承。输出累加器同步重缩放（ $v_s$ 为对应列的 value 向量）：

$$
O^{(j)} = e^{m^{(j-1)} - m^{(j)}} \thinspace O^{(j-1)} + \sum_{s \in B_j} e^{x_s - m^{(j)}} \thinspace v_s \qquad (16)
$$

同一条归纳给出不变量 $O^{(j)} = \sum_{s \in U_j} e^{x_s - m^{(j)}} v_s$。全部 $J$ 块处理完，

$$
o = \frac{O^{(J)}}{\ell^{(J)}} = \frac{\sum_s e^{x_s - m} v_s}{\sum_s e^{x_s - m}}
$$

右端即 2.3 式 (10) 同基准（全局 max）的精确 softmax 加权和。**FlashAttention 是精确等价，不是近似**：它与标准 attention 的数学对象逐点相同，差异只有浮点累加次序。这把它的定位与后两章划清：4.2 的稀疏 attention 改可见集、4.3 的线性 attention 换掉 softmax 核，都是改数学换复杂度；FlashAttention 一个数都不改。

### 5.3 tiling：两趟循环

分块执行（FA1 循环顺序：外层 KV 块、内层 Q 块；causal mask 的处理是跳过完全在对角线上方的块、只遮对角块）：

```
Q [T, d] 切成 T/B_r 行块          K, V [T, d] 切成 T/B_c 列块
┌────┐┌────┐┌────┐               外循环 j = 1 .. T/B_c：
│ Q1 ││ Q2 ││ Q3 │ ...             K_j, V_j 载入 SRAM（各 B_c×d，驻留本轮）
└────┘└────┘└────┘               内循环 i = 1 .. T/B_r:
 每个 Q_i 携三个片上量               Q_i 载入 SRAM（B_r×d）
   (m_i, l_i, O_i)                 S_ij = Q_i K_jᵀ（片上 [B_r, B_c]，不落 HBM）
   随内循环逐块更新                  按式 (15)(16) 更新 m_i, l_i, O_i
                                  内循环末：O_i ← O_i / l_i，写回 HBM（每块仅此一次）
```

块大小的账：SRAM 要同时驻留 $K_j, V_j$（各 $B_c d$）、当前 $Q_i$ 与累加器 $O_i$（各 $B_r d$）、中间块 $S_{ij}$（ $B_r B_c$）——FA 论文取 $B_c = \lceil M / (4d) \rceil$、 $B_r = \min(\lceil M / (4d) \rceil, d)$（ $M$ 为可用 SRAM、元素计，[A]）。代数字： $d$ = 64、 $M \approx 10^{5}$ 时 $B_c$ ≈ 390， $T$ = 16384 时 $Q$ 全程重读约 42 趟——重读是这笔设计里唯一的浪费，式 (17) 的 $M$ 进分母正源于此。causal 的省法也记一句：对角线右上方的块对完全不可见，整块跳过（不算也不读），只有对角块内部需要逐元素 mask——工作量与 IO 同步减半，与 2.3 式 (12) 的 causal 系数 2 对应。上图是 FA1 的循环序；§5.4 末 FA2 的「外循环移到 Q 块」改的正是这张图的外层。

内存账合口：前向除 $Q, K, V, O$ 本身外只多出每行两个标量（ $m_i, \ell_i$）与反向用的 $L_i$——额外内存 $O(T)$，对照 naive 的 $S, P$ 两张 $O(T^{2})$（2.3 §7 的表： $T$ = 131072 单层 2 TiB）。「满窗序列单层装不下」到「照常训练」的跨越就是这一行。

反向传播同一思想：不存 $P$（ $O(T^{2})$），只存输出 $O$ 与每行 logsumexp $L_i = m_i + \log \ell_i$（ $O(T)$）；反向需要 $S, P$ 时在片上按块重算。前向内存从 2.3 §7 的 $O(T^{2})$ 降到 $O(T)$——用重算的 FLOPs 换内存，正是 2.3 §7 末尾登记的「重算」出路的完整实现。

与 gradient checkpointing（1.2 的口径）划清界限：在 attention 模块外层做重算，前向仍要物化 $T \times T$ 中间态——瞬态峰值与 HBM 往返照旧，只是把「存到反向」换成「反向再算一遍」，IO 账一分不省。FlashAttention 的分块把「不物化」贯穿前向与反向： $S, P$ 从生到死都在 SRAM，反向重算的粒度也是块。区别不在「算不算两遍」，在「 $O(T^{2})$ 对象是否碰过 HBM」。

### 5.4 IO 复杂度对照

naive 实现的 HBM 流量（元素计）： $Q, K, V$ 读写 $O(T d)$， $S, P$ 各一写一读 $\Theta(T^{2})$——合计 $\Theta(T^{2} + T d)$， $T \gg d$ 时被 $T^{2}$ 项主导。

FlashAttention： $K, V$ 全程各读一遍 $O(T d)$； $Q$ 与 $O$ 在外循环每趟重读 / 重写，共 $T / B_c$ 趟、每趟 $O(T d)$，合计 $O(T^{2} d / B_c)$。代入 $B_c = \Theta(M / d)$：

$$
\mathrm{HBM\ bytes}_{FA} = O\left( \frac{T^{2} d^{2}}{M} \right) \qquad (17)
$$

与 naive 的比值 $\Theta(d^{2} / M)$： $d$ = 64（FA 论文 benchmark 的 head 维口径）、 $M \approx 10^{5}$ 元素时约 1/25。 $T$ 增大时两边同涨 $T^{2}$，FA 的常数小一个 $d^{2} / M$ 因子，且彻底免去 $P$ 的落盘与反向重读。

两句定位。其一，FLOPs 一分未省：causal 口径每层 $2 T^{2} d$（2.3 式 (12)）照付，收益全部来自访存与调度；实测吞吐数字归 gpu 维度的 benchmark 口径。其二，分块归一化作为数值技术早于 FlashAttention 存在；论文的贡献在组合——online softmax + tiling + 反向重算 + IO 复杂度分析——以及可执行的高性能 kernel（[Dao et al., 2022](../references.md)，[A]），并把「attention 的长序列瓶颈是 IO 而非 FLOPs」立为后续 kernel 设计的默认前提。

FA2 与 FA3 各一句话：FA2（[Dao, 2023](../references.md)，arXiv:2307.08691，[A]）保持同一数学，改并行划分——外循环移到 Q 块以并行化序列维、削减非 matmul FLOPs 与 warp 间冗余分工，同代硬件再快约一倍；FA3 面向 Hopper 代际的异步硬件特性（TMA、warp specialization、FP8）。kernel 实现归 gpu 维度，官方 repo Dao-AILab/flash-attention（[Dao-AILab, flash-attention](../references.md)，[A]）。

## 6. 接口与去向

本章交付两类账与两组推导。账：KV cache 的三种形态系数——MHA / GQA 的 $2 H_{kv} d_{\mathrm{head}}$（式 (3)）、MLA 的 $d_c + d_R$ = 576（§4.2 表）——将在 4.6 的长上下文内存对照与 5.2 的 DeepSeek 案例里直接代入；FlashAttention 的 IO 复杂度 $O(T^{2} d^{2} / M)$（式 (17)）是 gpu 维度 kernel 章的入口。推导：RoPE×低秩冲突（§4.1）与 absorption 的对称结构（§4.3–4.4）在 4.4 的 hybrid 模型（Kimi-Linear 的 MLA + NoPE 变体）中复用。方法论上，§2–§4 改「存什么」、§5 改「怎么算」，而 4.2（稀疏 attention：改可见集）与 4.3（线性 attention：改核函数）改「算什么」——Part 4 上半部分的全部省算工作可按这三个动词分类。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| MQA | 式 (1)(2)； $H_{kv}$ = 1 | 8B 形态 cache 16 KiB/token、满窗 2 GiB；K 侧行空间与 V 通道同时塌缩 → 质量损失 |
| GQA | $H_{kv} = H_q / g$，式 (3) | uptraining：组内 mean-pool 初始化（三臂对照最优）+ 约 5% 预训练算力；GQA-8 为论文 Fig 6 的中间点选择，Llama-3.1 8B / 70B 均取 $H_{kv}$ = 8 |
| RoPE×低秩冲突 | 式 (4) | $R_{n-t}$ 卡在 $W^{UK}$ 与 $c$ 之间； $R_\Delta C \neq C R_\Delta$（全换位 ⟺ 频率块不混合 ⟺ 非压缩） |
| MLA 构造 | 式 (5)–(10) | $c^{KV}$ = $W^{DKV} u$（512）+ decoupled $k^{R}$（64，旋转后缓存、全头共享）；query 侧同低秩（1536） |
| MLA cache 账 | 576 元素 / token / 层 | vs MHA-32 反事实 8192（14.2×）；V3 满窗 163840：10.7 GiB vs MHA 反事实 610 GiB（56.9×） |
| W_UK / W_UV absorption | 式 (11)(12) / (13)(14) | $`\tilde{W}_h = (W_h^{UQ})^{\top} W_h^{UK}`$、 $`\widehat{W}_h = W_h^{O} W_h^{UV}`$ 离线预计算；decode 吸收形（读 576 而非 32832 元素，57×）/ prefill 展开形 |
| online softmax | 式 (15)(16) | running max / sum，重缩放因子 $e^{m_{old} - m_{new}}$ 只换基准；归纳不变量 ⟹ 逐点精确 |
| FlashAttention | tiling 图；式 (17) | 精确等价非近似；HBM 流量 $O(T^{2} d^{2} / M)$ vs naive $\Theta(T^{2})$；反向重算，前向内存 $O(T)$ |
| FA2 / FA3 | 同一数学 | FA2 改并行划分（约一倍）；FA3 面向 Hopper 异步；细节归 gpu |

## 练习

1.（推导）MQA 的透镜账。2.3 §2：MHA 第 $h$ 头经 rank-$d_k$ 双线性型 $A_h = (W_h^{Q})^{\top} W_h^{K}$ 读 stream。(a) 写出 MQA 下第 $h$ 头的等价双线性型 $A_h^{MQA}$，说明 $H_q$ 个 $A_h^{MQA}$ 的「K 侧行空间」为什么被钉在同一组基上；(b) 设 $v_s$ 全头共享，证明对任意 attention 权重分配，每头输出 $o_{t,h} \in \mathrm{span}\lbrace v_s\rbrace_{s \le t}$，即全部头的读出共享同一条维数 ≤ $d_{\mathrm{head}}$ 的 value 通道——与 MHA 的 $H_q$ 份独立 $\lbrace v_{s,h}\rbrace$ 对照，说明「value 交付带宽与 $H_q$ 无关」的准确含义。
2.（推导）MLA absorption 的形状与账。设 $d_c$ = 512、 $d_c'$ = 1536、 $d_h$ = $d_v$ = 128、 $d_R$ = 64、 $H$ = 128。(a) 写出 $`W_h^{UQ}, W_h^{UK}, \tilde{W}_h, \widehat{W}_h`$ 的全部 shape 并验证式 (11)(14) 的矩阵乘法合法；(b) decode 第 $t$ 步，分别计算吸收形（score 对 $c$ 打）与非吸收形（先展开 $K^C, V^C$ 再算）的逐头 score FLOPs 与每 token 每层 KV 读取字节，验证「FLOPs 3 倍、读取 57 倍」的交换比；(c) prefill 长度 $T_p$ 时两形态的 $QK^{\top}$ FLOPs 各是多少？由此说明 prefill 用展开形的理由。
3.（计算）DeepSeek-V3 满窗账：61 层、每 token 每层 576 元素、bf16、 $T$ = 163840。(a) 单序列 cache 总量（GiB）；(b) 同窗口反事实 MHA（ $H$ = 128、 $d_{\mathrm{head}}$ = 128）与反事实 GQA-8（ $H_{kv}$ = 8）各多少；(c) $B$ = 16 并发下三者的 cache 总量；设权重 671B 参数以 fp8（1 字节）驻留约 671 GB，指出 cache 与权重在显存预算中的相对位置随 $B$ 的变化。
4.（代码）教学实现 online softmax forward（教学用，非优化实现）：随机 $Q, K, V$（ $T$ = 1024、 $d$ = 64、fp32），按式 (15)(16) 以块大小 $B_c$ = 128 实现 causal attention forward，与 `F.scaled_dot_product_attention(is_causal=True)` 断言 allclose（容差 $10^{-5}$）。扫 $B_c \in \lbrace32, 64, 128, 256\rbrace$：(a) 验证输出不随块大小改变（精确性的可执行形式）；(b) 记录峰值显存，对照 2.3 §7 教学版在同一 $T$ 下的物化账。

## 参考

- [Shazeer, 2019](../references.md)：Fast Transformer Decoding: One Write-Head is All You Need，arXiv:1911.02150，[B]（key: mqa）—— $H_{kv}$ = 1 的极端形与 cache 账
- [Ainslie et al., 2023](../references.md)：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints，arXiv:2305.13245，[B]（key: gqa）——组内共享、mean-pool uptraining（约 5% 算力）、cache-质量 Pareto
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，arXiv:2405.04434，[A]（key: deepseekv2）——MLA 构造与 W_UK / W_UV absorption（§2.1.2）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——MLA 沿用与官方推理实现
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——kv_lora_rank = 512、qk_rope_head_dim = 64、qk_nope_head_dim = 128、v_head_dim = 128、q_lora_rank = 1536、num_attention_heads = 128、61 层、窗口 163840
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——GQA： $H_q$ = 32、 $H_{kv}$ = 8
- [Meta, Llama-3.1-70B config.json](../references.md)：[A]（key: llama-3.1-70b-config）——GQA： $H_q$ = 64、 $H_{kv}$ = 8
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——GQA 的 K/V 组内广播实现
- [Dao et al., 2022](../references.md)：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，arXiv:2205.14135，[A]（key: flashattention）——online softmax、tiling、IO 复杂度 $O(T^{2} d^{2} / M)$、SRAM / HBM 硬件口径
- [Dao, 2023](../references.md)：FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning，arXiv:2307.08691，[A]（key: fa2）
- [Dao-AILab, flash-attention repo](../references.md)：[A]（key: flash-attention-repo）——官方 kernel 实现


