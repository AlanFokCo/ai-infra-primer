# 1.2 参数、计算、内存的账（Accounting for Parameters, Compute, and Memory）

> 本章把 config.json 里的几个整数变成训练与推理的 FLOPs 账和字节账：每 token 前向 ≈ 2N、训练总量 C ≈ 6ND、参数 bf16 = 2N bytes，外加激活与 KV cache 两笔随序列长度涨的账 | 前置：1.1（ $f_\theta$ 、 $d$ 、 $V$ 、 $T$ 记号） | 锚点：Llama-3.1-8B config [A]、DeepSeek-V3 config [A]、Kaplan et al. [B]、DeepSeek-V3 技术报告 [A]

## 1. 一笔真实账单：2.788M H800-hours 从哪来

DeepSeek-V3 技术报告给出四个数：671B 总参数、每 token 激活 37B、预训练语料 14.8T token（三者见摘要与模型架构节）、全训练耗 2.788M H800 GPU-hours——训练时长拆分 pre-training 2664K + context extension 119K + post-training 5K，见论文 Table 1 的成本行（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]；注意这是 GPU-hours 口径，不是货币总成本口径）。这四个数之间是什么关系？本章的答案是三个公式：每 token 前向 FLOPs $\approx 2N$ ，训练总计算量 $C \approx 6ND$ ，参数内存 $= 2N$ bytes（bf16），其中 $N$ 是非 embedding 参数量、 $D$ 是训练处理的 token 总数。推完之后，pre-training 的 2664K GPU-hours 与 14.8T token 就能互相核对——第 9 节会真正对上这笔账。

这套账值得先学的原因不是算术本身，而是它的用法：架构设计的每个决定——加宽 $d_{\mathrm{ff}}$ 、砍 KV head、换 MoE、拉长上下文窗口——在工程上都等价于把数字在参数、FLOPs、激活、KV cache 这几笔账之间搬来搬去。看懂账，config.json 里每个整数的代价就直接可读。

本章记号： $L$ = 层数（1.1 中 loss 写作 $L(\theta)$ ，本章不出现该用法）， $d$ = hidden_size， $d_{\mathrm{ff}}$ = intermediate_size， $H_q$ / $H_{kv}$ = Q / KV head 数， $d_{\mathrm{head}}$ = head 维， $V$ = 词表大小， $T$ = 序列长度， $B$ = batch 中的序列数。锚点数值：Llama-3.1-8B 的 $L=32$ 、 $d=4096$ 、 $d_{\mathrm{ff}}=14336$ 、 $H_q=32$ 、 $H_{kv}=8$ （GQA，grouped-query attention：4 个 Q head 共享一对 K/V，机制见 4.1）、 $d_{\mathrm{head}}=128$ 、 $V=128256$ （[Llama-3.1-8B config](../references.md)，[A]）；DeepSeek-V3 的 $L=61$ 、 $V=129280$ 、 671B 总参数 / 37B 激活（[DeepSeek-V3 config](../references.md)，[A]）。

## 2. 矩阵乘法的账：FLOPs = 2mnk

一切从一次内积开始。设 $X \in \mathbb{R}^{m \times k}$ 、 $W \in \mathbb{R}^{k \times n}$ ，乘积 $Y = XW$ 的每个元素是长度 $k$ 的内积：

$$
Y_{ij} = \sum_{s=1}^{k} X_{is} W_{sj}
$$

**符号**： $m$ ：左矩阵行数（token 维）； $k$ ：收缩维； $n$ ：输出维； $s$ ：求和哑指标。

每个 $Y_{ij}$ 要 $k$ 次乘法加 $k-1$ 次加法，共 $2k-1$ 次浮点运算； $Y$ 有 $mn$ 个元素，故

$$
\mathrm{FLOPs}(XW) = mn(2k-1) \approx 2mnk \qquad (1)
$$

**符号**： $\mathrm{FLOPs}(\cdot)$ ：该运算的浮点运算次数（floating-point operations，计数；速率单位 FLOPS 是每秒次数）。 $k$ 取千位量级时 $2k-1 \approx 2k$ 的相对误差小于 0.1%，全书忽略。

两点注记。第一，硬件一侧的对应物是融合乘加（fused multiply-add, FMA）：一条指令完成一次乘加，GPU 峰值算力按每周期 FMA 数乘 2 标称，所以式 (1) 既是算法账也是硬件账。部分文献按 MAC（multiply-accumulate）计数，1 MAC = 2 FLOPs，本书统一用 FLOPs，换算只差系数 2。第二，训练时左矩阵的 $m$ 是整个 batch 的 token 数 $B \cdot T$ ：teacher forcing（1.1 第 4 节）让一次前向并行打分全部 $T$ 个位置，所以「per token」账就是总账除以 token 数，本章默认按 per token 记。

## 3. 前向：每 token 2N

线性层 $y = xW$ 作用在单 token 上： $x \in \mathbb{R}^{1 \times k}$ 、 $W \in \mathbb{R}^{k \times n}$ ，由式 (1) 得 FLOPs $= 2kn$ ，恰好是 $W$ 元素数的 2 倍——每个参数恰好参与一次 FMA。decoder-only 主干除输入 embedding 查表（gather，无乘加）外全是线性层的复合：QKV 投影、out 投影、FFN 三个矩阵、输出头（1.1 式 (2) 的 $W$ ）都是 matmul。逐层求和：

$$
\mathrm{FLOPs}_{\mathrm{fwd}} \approx \sum_{W} 2 \thinspace |W| = 2N \qquad (2)
$$

**符号**： $N$ ：参与 matmul 的参数量； $|W|$ ：矩阵 $W$ 的元素个数。口径约定：输入 embedding 矩阵查表不耗 FLOPs，不计入 $N$ ；tie_word_embeddings = false 时输出头是独立矩阵且做 matmul，应计入 $N$ ——这是式 (2) 的全量口径；第 9 节实例为逐项对账改用层参数口径、输出头单列，两种口径的换算关系在该节开头给出。两者常被一起混淆：它们 shape 相同（都是 $V \times d$ ）、在内存账里各占一份 $Vd$ ，但 FLOPs 账里一个是查表（0）、一个是完整的 $2Vd$ ——Llama-3.1-8B 的输出头每 token $1.05 \times 10^{9}$ FLOPs，约占前向的 7%，不可忽略。RMSNorm、SiLU、residual add 等 elementwise 运算每层每 token 只有 $O(d)$ 次，相对 $2N$ 低约一个 $d$ 因子，忽略。

看个具体形状。Llama-3.1-8B 的 FFN up 投影是 $[1, 4096] \times [4096, 14336]$ ：由式 (1) 得 $2 \times 4096 \times 14336 = 1.17 \times 10^{8}$ FLOPs，而该矩阵有 $5.87 \times 10^{7}$ 个参数——恰好 2 倍。QKV 投影 $[1, 4096] \times [4096, 6144]$ 同理得 $5.03 \times 10^{7}$ 。每个线性层都满足这个 2 倍关系，求和即式 (2)。

式 (2) 有个不平凡的推论：固定 $N$ 时，per-token 前向 FLOPs 与 $N$ 如何分布在层数与宽度上无关——24 层宽模型与 48 层窄模型只要参数量相同，FLOPs 账就一样；深浅取舍改变的是并行度、优化行为与表达结构，不是这笔账。若只要估算，层参数有经验公式 $N \approx 12Ld^{2}$ （attention $4d^{2}$ + FFN $8d^{2}$ ，推导与 GQA / SwiGLU 对它的偏离在 2.1）；本章第 9 节直接用 config 精确值。

attention 的打分与加权和（ $QK^{\top}$ 与 $PV$ 两个 batched matmul， $P$ 为 softmax 归一化的 attention 权重矩阵，形式定义见 2.3）不含参数，不进式 (2)——它是随 $T^{2}$ 增长的另一笔账，第 6 节单算。

## 4. 反向：两倍前向

对 $y = xW$ ，记上游梯度 $G = \partial \mathcal{L} / \partial y \in \mathbb{R}^{1 \times n}$ （ $\mathcal{L}$ 为 loss，避免与层数 $L$ 混淆）。链式法则给出两个梯度：

$$
\frac{\partial \mathcal{L}}{\partial x} = G W^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial W} = \sum_{b=1}^{B_{\mathrm{tok}}} x_b^{\top} G_b \qquad (3)
$$

**符号**： $B_{\mathrm{tok}}$ ：该 step 参与梯度累加的 token 总数； $b$ ：token 下标； $x_b \in \mathbb{R}^{1 \times k}$ 、 $G_b \in \mathbb{R}^{1 \times n}$ 。

逐笔数：第一式是 $[1 \times n]$ 乘 $[n \times k]$ ，由式 (1) 每 token $2nk$ FLOPs，与前向相同。第二式是外积在 batch 上的累加：单个 token 的外积 $x_b^{\top} G_b$ 有 $kn$ 个元素、每元素一次乘法，看似只有 $nk$ ；但梯度要跨 $B_{\mathrm{tok}}$ 个 token 累加进同一份 $\partial \mathcal{L} / \partial W$ ，每个 token 对每个权重元素完成一次「乘并加进累加器」的 FMA，于是每 token $2nk$ FLOPs。两笔相加：

$$
\mathrm{FLOPs}_{\mathrm{bwd}} = 4N = 2 \times \mathrm{FLOPs}_{\mathrm{fwd}} \qquad (4)
$$

**符号**： $`\mathrm{FLOPs}_{\mathrm{bwd}}`$ ：反向传播每 token 的 FLOPs。

一句话版本：前向每个参数做 1 次 FMA，反向要为输入侧与权重侧各做 1 次，合计每参数每 token 6 次。这个「1 + 2」的结构是通用的：任何 $y = f(x, W)$ 的算子，反向都要给 $x$ 和 $W$ 各算一份梯度；线性层恰好两份都与前向同阶。式 (3) 第二式还埋了一个内存事实：算 $\partial \mathcal{L} / \partial W$ 需要前向的输入 $x_b$ ，所以反向开始前向的激活必须还在显存里——第 8 节账二的来源。

整条反向链上，梯度按与层序相反的顺序流动：输出头的 $\partial \mathcal{L} / \partial W$ 最先算（其输入是最后一层的隐状态），第 1 层的最后算。这决定了激活释放的顺序只能是「从后往前逐层释放」，也是梯度检查点按层边界切分的自然位置。

## 5. 合并：C ≈ 6ND 与适用条件

式 (2)(4) 合并，乘以训练全程处理的总 token 数 $D$ ：

$$
C = 6ND \qquad (5)
$$

**符号**： $C$ ：训练总计算量（FLOPs）； $D$ ：训练处理的 token 总数，重复数据按通过次数计。这个框架与逐运算的 per-token 分解出自 [Kaplan et al., 2020](../references.md)（arXiv:2001.08361，[B]）的附录。

式 (5) 是本书复用最多的账式：1.3 的 compute-optimal 推导以它为约束条件，4.5 用它换算 MoE 训练成本，Part 5 的案例章用它对账各家技术报告。它也反向使用：给定算力预算 $C$ ，式 (5) 圈定 $ND = C/6$ 的可行集合， $N$ 与 $D$ 怎么分是 1.3 的 compute-optimal 问题。量级直觉：Llama-3.1-8B 规模（ $N = 6.98 \times 10^{9}$ ，第 9 节）每训 1T token 要 $4.2 \times 10^{22}$ FLOPs。再反过来：一个 1024 卡集群、每卡持续 $3.4 \times 10^{14}$ FLOPS（第 9 节从 DeepSeek-V3 报告推出的量级），跑 30 天提供 $9.0 \times 10^{23}$ FLOPs——够 8B 规模训约 22T token，或 70B 规模训约 2.1T token；预算固定时「模型大一档」与「数据多一档」不可兼得，这就是 scaling law 要回答的分配问题（1.3）。

复用前先看清五条适用条件，每条对应一个被忽略的项：

1. **dense 前向**：每 token 激活全部参数。MoE 模型把 $N$ 换成激活参数 $N_{\mathrm{act}}$ 是近似（假设 expert 负载均衡），第 9 节的 DeepSeek-V3 算例即用此口径。
2. **忽略 attention 二次项**：成立条件是 $2LTd \ll 2N$ 即 $T \ll N/(Ld)$ ，换算实例见第 6 节。
3. **无激活重计算**：梯度检查点（gradient checkpointing）全开大约多付一次前向， $C$ 涨到 $8ND$ 量级（练习 4）；重计算策略属 training 维度。
4. **只计 matmul**：elementwise、通信、IO 不计；后两者归 cluster 维度。
5. **optimizer step 摊薄**：AdamW 的参数更新是 $O(N)$ 逐元素运算，摊到一个 step 的 $B_{\mathrm{tok}}$ 个 token 上每 token 只有 $O(N / B_{\mathrm{tok}})$ ，相对 $6N$ 可忽略。

## 6. attention 的二次项： $2LT^{2}d$ 单算

位置 $t$ 的 query 对 $t$ 个可见 key 各做一次长度 $d_{\mathrm{head}}$ 的内积（ $QK^{\top}$ ），再对 $t$ 个 value 做加权和（ $PV$ ）：每个 head 每个位置 $4 \thinspace t \thinspace d_{\mathrm{head}}$ FLOPs。乘 $H_q$ 个 head（ $H_q \thinspace d_{\mathrm{head}} = d$ ），对位置求和，乘 $L$ 层：

$$
C_{\mathrm{attn}} = L \sum_{t=1}^{T} 4td = 2LT(T+1)d \approx 2LT^{2}d \qquad (6)
$$

**符号**： $C_{\mathrm{attn}}$ ：一条长度 $T$ 序列一次前向的 attention matmul FLOPs。causal mask 使位置 $t$ 只见 $t$ 个 key（1.1 式 (8) 的下三角 $M$ ）， $\sum_{t} t = T(T+1)/2$ 把系数从 4 压到 2； $T$ 上千时 $T(T+1) \approx T^{2}$ ，近似误差小于 0.1%。不打 causal mask 的双向场景（encoder 类，见 3.2）系数回到 4。

平均每 token $2LTd$ ——对 $T$ 线性，对全序列就是 $T^{2}$ 。注意这笔账按 $H_q$ 计： $PV$ 的输出有 $H_q$ 个头，GQA 共享 K、V 削减的是 K/V 投影参数与 KV cache（第 7、8 节），不削减打分与加权和本身——所以即使 GQA 模型，式 (6) 里的 $d$ 仍是 $H_q d_{\mathrm{head}} = 4096$ 。它什么时候追平参数账 $2N$ ？令两者相等解出 $`T^{*} = N/(Ld)`$ 。代入 Llama-3.1-8B（ $N \approx 6.98 \times 10^{9}$ ，层参数口径，第 9 节开头声明并逐项加出；换全量口径 $`T^{*}`$ 差不到 8%，结论不变）： $`T^{*} = 6.98 \times 10^{9} / (32 \times 4096) \approx 5.3 \times 10^{4}`$ 。三个 $T$ 值下 attention 项占每 token 前向（ $2N$ + 输出头 $\approx 1.50 \times 10^{10}$ ）的比例：

| $T$ | $2LTd$ /token | 占前向比例 |
| --- | --- | --- |
| 4096 | $1.07 \times 10^{9}$ | 6.7% |
| 32768 | $8.59 \times 10^{9}$ | 36% |
| 131072 | $3.44 \times 10^{10}$ | 70% |

训练上下文远小于 53K 时式 (5) 是好近似；而 Llama-3.1 的服务窗口是 131072，此时 attention 项反超参数账两倍以上。一个常见误解在此纠正：FlashAttention 不减少这笔 FLOPs——它只把 $T \times T$ 的打分矩阵从显存里消掉（第 8 节账二），FLOPs 分文不少，省的是内存与访存带宽（4.1 展开）。长上下文场景的账必须带这一项——这是 4.6 的出发点。

## 7. 每层 FLOPs 分解与 shape 走读

把式 (2) 展开到单个 decoder 层。Llama-3.1-8B 数值，per token 前向：

| 运算 | shape（ $[m,k] \times [k,n]$ ） | FLOPs 公式 | 数值 |
| --- | --- | --- | --- |
| QKV 投影 | $[1, d] \times [d, (H_q + 2H_{kv}) \thinspace d_{\mathrm{head}}]$ | $2d (H_q + 2H_{kv}) \thinspace d_{\mathrm{head}}$ | 50.3 M |
| attention 打分与加权和 | $QK^{\top}$ 、 $PV$ （causal 平均 $T/2$ ） | $2Td$ | 33.6 M @ $T=4096$ |
| out 投影 | $[1, d] \times [d, d]$ | $2d^{2}$ | 33.6 M |
| FFN（SwiGLU 三矩阵） | $[1, d] \times [d, d_{\mathrm{ff}}]$ 两次 + $[1, d_{\mathrm{ff}}] \times [d_{\mathrm{ff}}, d]$ | $6 d \thinspace d_{\mathrm{ff}}$ | 352.3 M |

GQA 的痕迹在第一行： $H_q + 2H_{kv} = 32 + 16 = 48$ 个 head 维而非 MHA（multi-head attention， $H_{kv} = H_q$ 的标准形，2.3 展开）的 $3H_q = 96$ ，QKV 投影比 MHA 减半。FFN 占层内 81%——Llama-3.1 取 $d_{\mathrm{ff}} = 14336 = 3.5d$ ，比等参数经验规则 $(8/3)d$ 更宽，为什么是这个数的账见 2.6（SwiGLU 的结构与 SiLU 激活也在 2.6； $\odot$ 为逐元素乘）。四行里只有 attention 行随 $T$ 变： $T$ 从 4096 拉到 131072 时，其他三行不动，attention 行从 33.6 M 涨到 $1.07 \times 10^{9}$ ，从层内第三大户变成超过其余三行之和。读这张表的方式：矩阵形状列是推导（每个数来自 config 的哪个字段），FLOPs 公式列是机制，数值列是结论；换任何模型的 config 都能照此重算。

层合计（不计 attention 行） $4.36 \times 10^{8}$ FLOPs/token，乘 $L = 32$ 得 $1.396 \times 10^{10}$ ——正好是 $2 \times 6.98 \times 10^{9}$ ，式 (2) 闭环。一层前向的 shape 走读（元素数标在括号里）：

```
x (4096) ──RMSNorm──► (4096) ──QKV proj──► Q(4096) K(1024) V(1024)   50.3M FLOPs
 │                                          │ QK^T、PV（causal）       2Td
 │                                          ▼
 │                                     attn out (4096) ──O proj──► (4096)   33.6M
 ◄────────────── residual add ◄─────────────┘
 ▼
(4096) ──RMSNorm──► (4096) ──up/gate──► (14336)×2 ──SiLU⊙──► (14336) ──down──► (4096)   352.3M
 ◄────────────── residual add ◄─────────────┘
 ▼
下一层
```

## 8. 内存四账：参数、激活、KV cache、优化器态

显存预算分四笔，互相独立，各有独立的压缩手段。

**账一：参数**。bf16 存储每参数 2 bytes：

$$
\mathrm{Mem}_{\mathrm{param}} = 2N \text{ bytes} \qquad (7)
$$

**符号**：此处的 $N$ 是全部参数（含 embedding——embedding 不占 FLOPs 但占字节，两笔账的 $N$ 口径不同）。

系数 2 是精度选择：fp32 是 $4N$ ，bf16/fp16 是 $2N$ ，8-bit 量化是 $N$ ，4-bit 是 $N/2$ ——权重压缩线性地移动这笔账（量化方法与精度损失属 inference 维度）。Llama-3.1-8B 总参数 $8.03 \times 10^{9}$ （第 9 节逐项加），bf16 下 $1.61 \times 10^{10}$ bytes ≈ 16 GB，这是推理显存的地板——任何显存大于 16 GB 的卡都装得下权重，剩下的预算全给 KV cache 与激活。

**账二：激活**。式 (3) 第二式要求反向时前向的输入还在：每个 matmul 的输入与若干中间量都要存。一份无重计算、非 FlashAttention 的实现，逐层清点（每 token，bf16）：

| 存下的张量 | shape（每 token） | 元素数 |
| --- | --- | --- |
| RMSNorm1 输出 | $[d]$ | 4096 |
| Q、K、V | $[(H_q + 2H_{kv}) \thinspace d_{\mathrm{head}}]$ | 6144 |
| softmax 概率 $P$ | $[H_q, \thinspace t]$ ，causal 平均 $t = T/2$ | $16T$ |
| attention 输出 | $[d]$ | 4096 |
| RMSNorm2 输出 | $[d]$ | 4096 |
| FFN up / gate / SiLU⊙up | $[d_{\mathrm{ff}}]$ × 3 | 43008 |
| 层输入 $x$ | $[d]$ | 4096 |

每层每 token $65536 + 16T$ 个元素，即 $131072 + 32T$ bytes；全模型一个 step 的激活账为

$$
\mathrm{Mem}_{\mathrm{act}} \approx B \cdot T \cdot L \cdot (65536 + 16T) \cdot 2 \text{ bytes}
$$

**符号**：最后一项因子 2 是 bf16 字节数； $65536$ 来自上表定长项之和， $16T$ 来自 $P$ 。换个模型重算时，定长项的一般形式是每层每 token $\approx 4d + (H_q + 2H_{kv}) \thinspace d_{\mathrm{head}} + 3 d_{\mathrm{ff}}$ 个元素（norm 输出、attn 输出、层输入各一个 $d$ ，QKV 一份，SwiGLU 三份 $d_{\mathrm{ff}}$ ），随 $T$ 增长项是 $H_q \cdot T/2$ 个概率元素；具体实现存哪些张量有差异，但量级与增长阶由这两个结构项定。

代入数值看两个增长点。 $T = 4096$ 时每层每 token 256 KB，乘 32 层约 8 MB/token； $B=8$ 、 $T=4096$ 的一个 step 就是 256 GiB。把 $T$ 拉到 131072： $16T$ 项从 65536 涨到 $2.1 \times 10^{6}$ ，每 token 激活约 138 MB，其中 softmax 概率 $P$ 占 97%。随 $B \cdot T$ 线性的部分是激活检查点（activation checkpointing，重计算换显存，属 training 维度）的动机；随 $T$ 二次的部分是 FlashAttention 不存 $P$ 、反向时重算的动机（4.1 展开）。另外输出头 logits $[V]$ 每 token $128256 \times 4$ bytes（fp32）≈ 0.5 MB，大 batch 下需要分块 cross-entropy（training 维度）。

**账三：KV cache**。推理时每层的 K、V 可以缓存以避免逐 token 重算前缀——可缓存性的推导与完整公式在 2.4，此处只登记它在预算中的位置：

$$
\mathrm{Mem}_{\mathrm{KV}} = 2 \thinspace L \thinspace H_{kv} \thinspace d_{\mathrm{head}} \thinspace T \thinspace B \thinspace b \text{ bytes}
$$

**符号**： $b$ = 每元素字节数（bf16 = 2）；系数 2 来自 K、V 两份。代入 Llama-3.1-8B（ $H_{kv}=8$ 、 $d_{\mathrm{head}}=128$ 、满窗 $T=131072$ 、 $B=1$ 、bf16）得 16 GiB——与权重本身同量级（cache 16 GiB 对权重约 15 GiB），长上下文推理的第一大户。压缩它（MQA / GQA / MLA）是 4.1 的主题。

这笔账还连着第 10 节的算术强度：decode 每生成一个 token，都要把整个 cache 读一遍（当前 query 与全部历史 K、V 做 attention），即满窗时每 token 额外搬动 16 GiB，与读权重的 16 GB 同量级——长上下文服务的带宽预算里，cache 读取与权重读取是并列的两大项，这是 2.4 与 inference 维度展开的话题。

**账四：优化器态**。AdamW 每个参数带两个矩估计，混合精度训练再存一份 fp32 主权重：

$$
\mathrm{Mem}_{\mathrm{opt}} = \underbrace{4N}_{\text{fp32 master}} + \underbrace{cN}_{m} + \underbrace{cN}_{v} = (4 + 2c) \thinspace N \text{ bytes} \qquad (8)
$$

**符号**： $m$ 、 $v$ ：AdamW 的一阶、二阶矩估计； $c$ ：矩估计存储字节数。fp32 矩估计 $c=4$ 得 $12N$ ；8-bit 量化矩估计 $c=1$ 得 $6N$ ——优化器态 $6N \sim 12N$ bytes 的区间由此而来。

Llama-3.1-8B 的训练态权重相关账（bf16 工作权重 + bf16 梯度 + 优化器态）：

| 项 | fp32 矩估计（ $c=4$ ） | 8-bit 矩估计（ $c=1$ ） |
| --- | --- | --- |
| 工作权重 $2N$ | 16 GB | 16 GB |
| 梯度 $2N$ | 16 GB | 16 GB |
| fp32 master $4N$ | 32 GB | 32 GB |
| 矩估计 $2cN$ | 64 GB | 16 GB |
| 合计 | 128 GB | 80 GB |

单卡放不下，分片（ZeRO 族把这三类状态切到多卡）属 training 维度；本章只要记住数量级：训练一个模型的权重相关显存约 $16N$ bytes（fp32 矩估计口径， $c=4$ ），是推理权重账的 8 倍。

## 9. 实例演算：Llama-3.1-8B 与 DeepSeek-V3

**Llama-3.1-8B 参数逐项**（config 数值乘出来，config 字段见 [Llama-3.1-8B config](../references.md)，[A]）。先固定口径：本节实例中的 $N$ 取层参数口径，即下面逐项加出的 $6.98 \times 10^{9}$ ，untied 输出头作为单列加项；式 (2) 符号表的 $N$ 是全量口径（含输出头， $7.50 \times 10^{9}$ ）。两种口径的换算只有一行： $N_{\mathrm{full}} = N + Vd$ ，于是前向 $2N_{\mathrm{full}} = 2N + 2Vd$ ——下文「前向 $2N$ + 输出头 $2Vd$ 」的写法即按此展开， $`T^{*}`$ 与训练账同此口径。

- 每层 attention：Q 投影 $d^{2}$ = 16777216，K、V 投影 $2 \times d \times H_{kv} d_{\mathrm{head}} = 2 \times 4096 \times 1024$ = 8388608，out 投影 $d^{2}$ = 16777216，小计 41943040；
- 每层 FFN： $3 d \thinspace d_{\mathrm{ff}} = 3 \times 4096 \times 14336$ = 176160768；
- 层计 218103808，乘 $L=32$ 得 6979321856，即 $N \approx 6.98 \times 10^{9}$ ——式 (2)(6) 用的就是它；
- embedding $Vd$ = 525336576，独立输出头同值；总参数 $6979321856 + 2 \times 525336576 = 8029995008 \approx 8.03 \times 10^{9}$ ，「8B」名实相符。

per-token 账：前向 $2N$ + 输出头 $2Vd$ ≈ $1.40 \times 10^{10} + 1.05 \times 10^{9} = 1.50 \times 10^{10}$ FLOPs（15.0 G）；训练每 token $6N + 6Vd$ ≈ $4.50 \times 10^{10}$ FLOPs（45.0 G）； $T=4096$ 时 attention 项每 token 另加 $1.07 \times 10^{9}$ ，占前向约 7%。权重字节账同时闭环： $8.03 \times 10^{9} \times 2 = 1.61 \times 10^{10}$ bytes，即第 8 节账一的 16 GB。

**DeepSeek-V3（MoE，用激活口径）**： $N_{\mathrm{act}} = 37 \times 10^{9}$ 、 $D = 14.8 \times 10^{12}$ （[DeepSeek-AI, 2024](../references.md)，[A]）：

$$
C \approx 6 \times 37 \times 10^{9} \times 14.8 \times 10^{12} = 3.29 \times 10^{24} \text{ FLOPs} \qquad (9)
$$

**符号**： $N_{\mathrm{act}}$ ：MoE 每 token 实际激活的参数量，替代式 (5) 中的 $N$ 。

对账分三步，关键在第二步的口径对齐。第一步，训练每 token $6 N_{\mathrm{act}} = 2.22 \times 10^{11}$ FLOPs（222 G），前向 $7.4 \times 10^{10}$ （74 G）。第二步：式 (9) 的 $D$ = 14.8T 是 pre-training 语料，对应的工时也必须取 Table 1 拆分中的 pre-training 部分 $2.664 \times 10^{6}$ GPU-hours（全训练 2.788M = pre 2664K + context extension 119K + post-training 5K，extension 与 post 的 12.4 万卡时对应另一批 token 与 FLOPs），乘 3600 秒得 $9.59 \times 10^{9}$ GPU-秒，与式 (9) 相除得每卡平均持续 $3.43 \times 10^{14}$ FLOPS。第三步交叉验证：14.8T token 摊到 2.664M GPU-hour 是每卡每小时 $5.6 \times 10^{6}$ token、每秒约 1543 token；用每 token 222 G 反乘回去， $1543 \times 2.22 \times 10^{11} = 3.43 \times 10^{14}$ FLOPS，与第二步一致。若误用全训练 2.788M 小时数去除 pre-training 的 FLOPs，得到 $3.27 \times 10^{14}$ ——只差 5%，但口径已经混了。这个派生量本身模型无关；它对应多高的算力利用率（MFU，model FLOPs utilization）取决于 H800 的峰值口径，属 gpu 与 training 维度（练习 3 请读者自己查峰值并算出 MFU）。此处能下的结论是：14.8T token、37B 激活、2.664M pre-training GPU-hours 三个论文数字与 6ND 框架互相自洽。

两个锚点的四笔账并排：

| 账 | Llama-3.1-8B | DeepSeek-V3 |
| --- | --- | --- |
| 总参数 | $8.03 \times 10^{9}$ | $671 \times 10^{9}$ |
| 计入 FLOPs 的 $N$ | $7.50 \times 10^{9}$ （含输出头） | $37 \times 10^{9}$ （激活口径） |
| 前向 /token | $1.50 \times 10^{10}$ | $7.4 \times 10^{10}$ |
| 训练 /token | $4.50 \times 10^{10}$ | $2.22 \times 10^{11}$ |
| 权重 bf16 | 16 GB | 1.34 TB |
| KV cache（满窗、 $B=1$ ） | 16 GiB @ 131072 | 结构为 MLA，账见 4.1 |

最后看 MoE 的本质：DeepSeek-V3 总参数是 Llama-3.1-8B 的 84 倍，权重账 1.34 TB 对 16 GB；但 per-token 前向 FLOPs 只是 74 G 对 15 G，4.9 倍——MoE 把内存账与计算账解耦，参数可以随便加，每 token 只为激活的那部分付 FLOPs（路由机制与代价见 4.5）。两点口径注记：37B 激活假设 expert 均衡路由；MLA、MTP 等结构差异被 6ND 抹平，逐结构的精细分解见 5.2。

## 10. 两笔账的会合：算术强度

把式 (2) 与式 (7) 放在一起，除一下，得到推理 decode 阶段的关键比率：每生成一个 token，要算约 $2N$ FLOPs（长上下文时外加随 $t$ 线性增长的 attention 项，见本节末），同时至少把全部权重从显存读一遍，即 $2N$ bytes——算术强度（arithmetic intensity，FLOPs 与访存字节之比）约等于 1 FLOP/byte。任何实际 GPU 的「峰值 FLOPS ÷ 显存带宽」都比 1 高两个数量级以上（具体数值属 gpu 维度），所以 decode 吞吐的上限由带宽决定、算力大量闲置：这就是「decode 是访存瓶颈」的精确含义。

prefill 相反：一次前向处理 $T$ 个 prompt token，算 $2NT$ FLOPs 而权重只读一遍，算术强度 $\approx T$ 。 $T$ 上千时进入计算瓶颈区。decode 侧的补救也是同一公式：把 $B$ 条序列的请求合成一个 batch，每步算 $2NB$ FLOPs、权重仍只读一遍，算术强度升到 $\approx B$ ——只要 KV cache 放得下，batch 越大每卡吞吐越高，这是推理服务拼命凑 batch 的算术根源。同一个模型、同一份权重，不同阶段的瓶颈不同——推理系统把 prefill 与 decode 分开调度、分开定价，根源就是这几个数（系统展开属 inference 维度）。训练侧同理：一个 step 里 $B \cdot T$ 个 token 摊一份权重读取与更新，算术强度 $\propto B \cdot T$ ，这就是训练追求大 batch 的硬件原因，也是激活账（随 $B \cdot T$ 线性）必须被重计算压住的硬件原因。

KV cache 的存在性也写在这个框架里：decode 生成第 $t$ 个 token 时若无 cache，要对长度 $t$ 的前缀重跑整条前向，单 attention 一项就是 $2Lt^{2}d$ （式 (6) 代入 $T=t$ ），且随 $t$ 平方累积；有 cache 时该步只算当前 query 对 $t$ 个缓存 key 的打分与加权和，每层 $4td$ 、共 $4Ltd$ ——从二次降到线性。代价是第 8 节账三的字节数。可缓存性的推导（为什么 K、V 足够、为什么不需要缓存 Q）在 2.4。

## 总结

| 账 | 公式 | 要点 |
| --- | --- | --- |
| matmul | $2mnk$ | 每输出元素一次长度 $k$ 内积，式 (1) |
| 前向 | $2N$ /token | 每参数一次 FMA；embedding 查表不计，式 (2) |
| 反向 | $4N$ /token | 输入侧 $2N$ + 权重侧 $2N$ ，式 (3)(4) |
| 训练总量 | $C = 6ND$ | dense、无重算、 $T \ll N/(Ld)$ ；MoE 换 $N_{\mathrm{act}}$ ，式 (5) |
| attention 二次项 | $2LT^{2}d$ /序列 | causal 平均后每 token $2LTd$ ，式 (6) |
| 参数内存 | $2N$ bytes | bf16；embedding 计入内存口径，式 (7) |
| 激活 | $BTL(65536 + 16T) \times 2$ bytes（8B 实例） | 随 $BT$ 线性、随 $T^{2}$ 二次 |
| KV cache | $2 L H_{kv} d_{\mathrm{head}} T B b$ | 推导见 2.4；8B 满窗 16 GiB |
| 优化器态 | $(4 + 2c)N$ = $6N \sim 12N$ bytes | fp32 master + 两阶矩估计，式 (8) |

## 练习

1.（计算）GQA 的账：设 Llama-3.1-8B 改用 MHA（ $H_{kv} = H_q = 32$ ），重算 QKV 投影的每层参数与 per-token FLOPs，给出它使层参数总量与总参数各增加多少；再问：为什么 GQA 对 KV cache 的削减比例（4 倍）大于对 QKV 投影 FLOPs 的削减比例（2 倍）？
2.（计算）一个 1024 卡集群，每卡有效吞吐取第 9 节推出的 $3.4 \times 10^{14}$ FLOPS：用式 (5) 估算训练 Llama-3.1-8B 规模模型（ $N = 6.98 \times 10^{9}$ ）2T token 需要多少天；若把 $T$ 从 4096 提到 131072 且不做任何重算，先估算激活显存（第 8 节账二）会发生什么，再说训练时间按式 (6) 会如何变化。
3.（计算）查 H800 SXM 的 BF16 稠密峰值 FLOPS，由式 (9) 与 2.664M pre-training GPU-hours 反推 DeepSeek-V3 pre-training 的有效 MFU 区间；列出使 MFU 低于 100% 的机制性因素（通信、流水线气泡、重计算、激活/梯度内存墙），各自归哪个维度讨论。
4.（推导）证明梯度检查点全开时 $C \approx 8ND$ ：指出反向时多付的那次前向发生在哪一步、省下了哪笔内存；再说明选择性重算（只重算 attention 块内部）为什么通常比全开更划算——用第 8 节账二表中 $16T$ 项回答。

## 参考

- [Kaplan et al., 2020](../references.md)：Scaling Laws for Neural Language Models，arXiv:2001.08361，[B]（key: kaplan）——2N / 6ND 框架与逐运算分解出自其附录
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——671B / 37B / 14.8T token / 2.788M H800 GPU-hours（Table 1，全训练口径：pre 2664K + context extension 119K + post 5K）
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）
