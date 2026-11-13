# 5.3 Qwen 谱系：双轨规模化与 hybrid 转向（The Qwen Lineage）

> 本章把 Qwen 谱系读成一条架构演化链：Qwen1.5 承接标准配方（[W]，无 arXiv 论文）、Qwen2 全系普及 GQA 并给出 dense→MoE upcycling 的公开实例、Qwen2.5 补齐全尺寸档与 1M 长上下文版、Qwen3 确立 dense 6 档 + MoE 2 档双轨并以 QK-Norm 换掉 QKV bias、Qwen3-Next 把 P4 的零件（GDN、3:1 hybrid、512-expert MoE、MTP、zero-centered norm）整体装配、Qwen3-Max 转入 API-only（[C]）。按 P5 固定模板展开：配置谱系表 → 标志性架构决策的公式级分析 → vocab 151936 的多语言压缩率账 → Qwen3-Next-80B-A3B-Instruct 的 config 与 modeling 代码精读 → 技术开放问题 | 前置：2.2（vocab 三方账）、2.4（KV cache 式 (9)）、2.5（RoPE 波长账）、2.7（QK-Norm、zero-centered 增益）、2.8（head_dim 误解）、4.1（GQA）、4.3（GDN）、4.4（hybrid 内存账）、4.5（MoE 稀疏度与 MTP）、4.6（DCA 与 YaRN 温度） | 锚点：Qwen/Qwen3-Next-80B-A3B-Instruct（config + transformers `modeling_qwen3_next.py`）[W]、Qwen3-235B-A22B-Instruct-2507 config（HF 实测 2026-11-12）[A]、Qwen3-8B config [A]；论文：Qwen1 [A]、Qwen2 [A]、Qwen2.5 [A]、Qwen2.5-1M [A]、Qwen3 [A]；[W]：Qwen1.5、Qwen3-Next；[C]：Qwen3-Max

## 1. 配置谱系表

Qwen3-235B-A22B-Instruct-2507 的 config.json 里 `num_attention_heads=64` 而 `num_key_value_heads=4`——16 个 query head 共享一组 KV；同一份 config 里 `mlp_only_layers=[]`，即从第 0 层起层层 MoE，没有 DeepSeek-V3 `first_k_dense_replace=3` 那样的 dense 前导层（均 config 实测，[A]）。这两个字段已经把 Qwen 旗舰的技术取点说了一半：KV 侧压缩激进、MoE 化彻底。全谱系各代字段如下（列序按 W3 约定：model | total/active 参数 | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE（experts/top-k/shared）| norm | vocab | ctx（max_position_embeddings）| rope_theta | 证据 key）：

| 模型 | total/active | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE | norm | vocab | ctx | rope_theta | key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen1 | — | — | — | — | — | MHA（谱系起点，GQA 于 Qwen2 引入） | 无（dense） | — | — | — | — | qwen1 [A] |
| Qwen1.5 | — | — | — | — | — | 同上，标准配方承接代 | 无（dense 全尺寸） | — | — | — | — | qwen1.5 [W] |
| Qwen2 | dense 4 档（0.5B/1.5B/7B/72B）+ 57B-A14B | — | — | — | — | GQA 全系普及 | dense 主线 + 57B-A14B（upcycling 自 Qwen2-7B，report §2.2.3 明写，§2.3） | QKV bias（旧制，Qwen3 起移除） | 152064 | — | — | qwen2 [A] |
| Qwen2.5-7B-1M | — | — | — | — | — | GQA | 无（dense） | — | 152064 | 1010000 | $10^{7}$ | qwen2.5 [A]、qwen2.5-1m [A] |
| Qwen3-8B（dense 代表） | 8B 档 | 36 | 4096 | 32/8（ $g$=4） | 128 | GQA + RoPE | 无 | RMSNorm + QK-Norm，attention_bias=false | 151936 | 40960 | $10^{6}$ | qwen3 [A]、qwen3-8b-config [A] |
| Qwen3-30B-A3B | 30.5B / 3.3B | 48 | — | 32/4（ $g$=8） | — | GQA | 128 top-8，无 shared | 同上行 | 151936 | — | — | qwen3 [A] |
| Qwen3-235B-A22B（Instruct-2507） | 235B / 22B | 94 | 4096 | 64/4（ $g$=16） | 128 | GQA + RoPE | 128 top-8，无 shared，`mlp_only_layers=[]` | 同上行 | 151936 | 262144 | $5 \times 10^{6}$ | qwen3 [A] + config 实测 [A] |
| Qwen3-Next-80B-A3B | 80B / 3B（官方口径：非 embedding 79B） | 48 = 36 GDN + 12 full | 2048 | 16/2（full 层， $g$=8） | 256 | GDN : gated GQA 按 3:1 交错 | 512+1 top-10 | zero-centered RMSNorm | 151936 | 262144（厂商宣称可扩到 1010000） | $10^{7}$ | qwen3-next [W] |
| Qwen3-Max | 厂商宣称 >1T | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | 未披露 | qwen3-max [C] |

表注四条。其一，「—」= 未锁定字段（本轮 research pass 未实测），谱系定位与代际结论不依赖这些格子；Qwen2/2.5 的 vocab_size=152064 与 Qwen3-30B-A3B 的 48 层 / 32/4 heads 为本章 fix round 1 复核补入（分别为 HF config 实测与 Qwen3 report Table 2，2026-11-12）。数值口径优先级为 config.json 实测 > tech report 架构表 > model card > blog（W3 约定页）。其二，total/active 不是 config 字段：30.5B/3.3B 与 235B/22B 引 Qwen3 report 口径（[Yang et al., 2025](../references.md)，arXiv:2505.09388，[A]）；80B/3B 引 model card（[W]，2026-11-11 实测），照抄官方口径不自行换算。其三，两个长度字段勿混：Qwen3-8B config 的 max_position_embeddings=40960 是原生窗口，同 repo tokenizer_config 的 model_max_length=131072 是 YaRN 扩展口径（1.1 已核）；Qwen2.5-1M 的 config **无 rope_scaling 字段**——1M 靠 rope_theta $10^{7}$ + dual-chunk attention + YaRN 温度修正的组合（4.6 §2.5 已拆），不是频率侧的 YaRN ramp。其四，Qwen3-235B 行为 Instruct-2507 修订版的 config 实测（2026-11-12）；report 首版数字以 arXiv:2505.09388 为准。

rope_theta 一列也值得竖着读： $10^{6}$（Qwen3-8B）→ $5 \times 10^{6}$（235B）→ $10^{7}$（Qwen3-Next 与 Qwen2.5-1M）——目标窗口越长，base 越高，最长波长 $\lambda_{\max} = 2\pi \cdot \mathrm{base}^{1 - 2/d}$ 随之抬升（2.5 式 (16) 的波长账），慢通道的旋转周期才罩得住长窗里的位置差。

读表得出的结构性观察是双轨制。Qwen2 已有一档 MoE（57B-A14B）但主线是 dense；Qwen3 把双轨制度化——dense 线 6 个尺寸档（谱系表以实测的 Qwen3-8B 为代表）+ MoE 线 2 档（30B-A3B、235B-A22B，report 口径 [A]）；到 Qwen3-Next 一代只发 MoE hybrid，dense 线缺席。命名也把 MoE 口径写进型号：A3B / A22B 即 active 参数量。Qwen3 另有一条模型侧之外的主线：thinking / non-thinking 双模式——这是训练侧配方（post-training 与 RL），架构字段上无改动，归 training 维度，本章只登记其 token 层面的痕迹（§3 的 `<think>` 特殊 token）。

Qwen3-Max 单独一段，按 [C] 级约束写：截至 2026-11-12，该模型无 arXiv 论文、无开放权重（HF Qwen/Qwen3-Max 返回 404，本轮实测），仅以 API 形式服务；「>1T 参数」为厂商宣称（公告 URL 在案，页面为 JS 渲染、无法机械取证）。旁证渠道（OpenRouter 模型清单，2026-11-12）显示该 API 线已迭代至 qwen3.7-max / qwen3.8-max-prime（ctx 标称 1M）。除「2025-09 起作为 API-only 旗舰线存在」与上述厂商宣称外，本章不断言它的任何架构字段——它是谱系表里唯一填不出数字的一行，本身就是 [C] 级证据边界的实例。

## 2. 标志性架构决策的公式级分析

### 2.1 Qwen2 起全系 GQA：KV cache 账重算

Qwen2 report 把 GQA 做到全尺寸档（[Yang et al., 2024](../references.md)，arXiv:2407.10671，[A]；GQA 机制与 uptraining 出处 [Ainslie et al., 2023](../references.md)，arXiv:2305.13245，[B]）——在此之前 Llama-2 只在 70B 档用 GQA，Qwen2 的改动是「组共享 KV」从大模型补丁变成默认配置。机制与推导在 4.1，此处只把账代入 Qwen 实测 config。cache 公式（2.4 式 (9)）：

$$
M_{\mathrm{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_{\mathrm{head}} \cdot T \cdot b \qquad (1)
$$

**符号**： $L$：层数； $H_{kv}$：KV head 数； $d_{\mathrm{head}}$：每头维度； $T$：序列长度； $b$：元素字节数（bf16 取 2）；因子 2 = K、V 两份。

代入旗舰（ $L$=94、 $H_{kv}$=4、 $d_{\mathrm{head}}$=128、 $T$=262144、 $b$=2、 $B$=1）：每 token $2 \times 94 \times 4 \times 128 = 96256$ 元素 = 188 KiB，整窗 $188 \ \mathrm{KiB} \times 262144 \approx 47$ GiB。MHA 反事实（ $H_{kv}$ = 64）：恰乘 $g$ = 16 倍，约 752 GiB——单条序列就超出任何单卡 HBM，GQA 在这里不是优化而是 262K 窗口的前提。对照 Qwen3-8B（ $L$=36、 $H_{kv}$=8）：每 token $2 \times 36 \times 8 \times 128 \times 2\ \mathrm{B}$ = 144 KiB、整窗（40960）约 5.6 GiB。注意组数的代际收紧：8B 档 $g$=4（32/8）、旗舰 $g$=16（64/4）——模型越大、目标窗越长，KV 侧压缩越激进。服务端再补一笔：式 (1) 对 batch 是线性的，47 GiB 是单序列账， $B$ 路并发直接乘 $B$——旗舰的 $g$=16 把每路的斜率压到 188 KiB/token，才给并发留出让多路共存于同一显存预算的空间。decode 一侧的读取量同式（每 token 把整窗 cache 读一遍，2.4 §5 的算术强度分析）， $g$ 同时把 decode 的内存带宽账除以 16。代价机制同 4.1： $g$ 组共享等价于 KV 子空间的检索自由度除以 $g$，质量端靠 Q/KV 头数的总量冗余兜底。

### 2.2 QK-Norm 替代 QKV bias：logit 幅度的结构性上界

Qwen3 config 实测 `attention_bias=false`（Qwen3-8B 与 235B 一致，[A]），report 架构节写明引入 QK-Norm 替代此前谱系的 QKV bias（[Yang et al., 2025](../references.md)，arXiv:2505.09388，[A]；QK-Norm 原始出处 [Henry et al., 2020](../references.md)，arXiv:2010.04245，[B]，2.7 已定位）。要回答「换掉什么、买到什么」，先把 2.3 的失效模式摆出来：attention logit $z = \langle q, k \rangle / \sqrt{d_k}$，若 $q, k$ 的幅度在训练中漂移， $z$ 进入 softmax 饱和区，Jacobian $\mathrm{diag}(p) - p p^{\top}$ 的行梯度按 $p_i(\delta_{ij} - p_j)$ 消失——这是 logit 失控的链条。QK-Norm 在每个 head 内对 $q$、 $k$ 做 RMSNorm（单位增益口径）：

$$
\hat{q} = \sqrt{d_k}\thinspace \frac{q}{\lVert q \rVert}, \qquad \hat{k} = \sqrt{d_k}\thinspace \frac{k}{\lVert k \rVert}, \qquad z_{mn} = \frac{\langle R_m \hat{q},\ R_n \hat{k} \rangle}{\sqrt{d_k}} \qquad (2)
$$

**符号**： $q, k \in \mathbb{R}^{d_k}$：单头投影输出； $R_m$：位置 $m$ 的 RoPE 旋转（2.5）； $z_{mn}$：位置对 $(m, n)$ 的 attention logit。归一化使 $\lVert \hat{q} \rVert = \lVert \hat{k} \rVert = \sqrt{d_k}$。Cauchy–Schwarz 加旋转保范性（ $\lVert R_m x \rVert = \lVert x \rVert$）给出：

$$
|z_{mn}| \le \frac{\lVert R_m \hat{q} \rVert \cdot \lVert R_n \hat{k} \rVert}{\sqrt{d_k}} = \frac{\sqrt{d_k} \cdot \sqrt{d_k}}{\sqrt{d_k}} = \sqrt{d_k} \qquad (3)
$$

$d_k$ = 128 时全部 logit 落在 $[-11.3, 11.3]$，与位置差 $m - n$ 无关、与输入幅度无关——饱和区被结构性排除，不再依赖训练运气。注意归一化与 RoPE 的顺序在这里无关：旋转不改变范数，先归一还是先旋转，式 (3) 的界都成立。买到的是这个上界；换掉的是 bias 提供的逐通道常数位移自由度——RMSNorm 无 re-centering（2.7）， $q, k$ 的均值方向信息被移除。QKV bias 的参数量本身微不足道（235B 每层 $(64 + 2 \times 4) \times 128 = 9216$ 个），这次替换的实质是把「logit 稳定」从可学参数手里收归结构。

同一问题的另一解法作对照：Gemma 2 用 logit soft-capping（ $c \cdot \tanh(z / c)$，逐 logit 压回有界区间），Gemma 3 与 Qwen3 一样转用 QK-Norm（2.7 已登记，勿张冠李戴）。两条路线的差别在作用点：soft-capping 在 logit 算完之后压形状，QK-Norm 在 logit 算之前归一化输入——前者保留 $\lVert q \rVert \cdot \lVert k \rVert$ 的相对大小信息再截断，后者直接把幅度信息从 score 里拿掉。Qwen3 的选择意味着它的 attention score 完全由方向（夹角）决定，幅度通道让位给结构的其余部分。公开材料未见该项的逐项消融，独立贡献量化列入 §5 开放问题。

### 2.3 upcycling：dense→MoE 的权重移植

Qwen2-57B-A14B 的 MoE 化走的是 upcycling，源模型在 report 里有明文：§2.2.3 写道「Qwen2-57B-A14B is upscaled from Qwen2-7B」（ar5iv 全文复核，2026-11-12）。形状证据与之互证：HF config 实测 57B-A14B 为 hidden 3584 / 28 层 / FFN 中间维 18944，与 Qwen2-7B 逐项一致（fix round 1 复核）；而 Qwen1.5-32B 是 5120 / 64 层 / 27392——upcycling 要逐元素搬运 FFN 权重，shape 不同在几何上就不成立，「源是 Qwen1.5-32B」的流传说法由明文与形状双重排除。另校正一处归属：该技术常被记到 Moonlight 名下，Moonlight 论文全文 0 次提及 upcycling（W3 research pass 逐文核实）；公开谱系里的旗舰级实例是 Qwen2 这一例，5.5 章回引本节。

程序层面（report §2.2.2 Expert Initialization 节原文），这套 upcycling 不是朴素拷贝：设 expert 中间维 $h_E$、expert 数 $n$、源 dense FFN 中间维 $h_{\mathrm{FFN}}$，先把 FFN 整体复制 $\lceil n h_E / h_{\mathrm{FFN}} \rceil$ 份（保证可切出 $n$ 个 expert 的总宽度）；每份拷贝沿中间维 shuffle（逐拷贝不同，保证跨拷贝多样性）；再从这些拷贝中切出 $n$ 个 expert、丢弃剩余维度；最后对每个 fine-grained expert 的 50% 参数随机重初始化。公式化：

$$
W^{(e)}_i \leftarrow M_i \odot \big(P_i\thinspace \Pi\thinspace \widetilde{W}^{\mathrm{dense}}_{\mathrm{FFN}}\big) + \left(1 - M_i\right) \odot R_i, \qquad \Pr\left[ (M_i)_j = 1 \right] = 0.5 \qquad (4)
$$

**符号**： $`\widetilde{W}^{\mathrm{dense}}_{\mathrm{FFN}}`$：源 dense FFN 参数（SwiGLU 三矩阵同法处理）复制 $\lceil n h_E / h_{\mathrm{FFN}} \rceil$ 份后的总参数； $\Pi$：逐拷贝不同的中间维置换（shuffle）； $P_i$：第 $i$ 个 expert 的切片选择（从 shuffle 后拷贝中抽取 $h_E$ 宽的一段，剩余维度丢弃；fine-grained 切分思想同 4.5 式 (14)）； $M_i \in \lbrace0,1\rbrace$：逐参数保留掩码，50% 的位置保留继承值； $R_i$：随机初始化值； $1 - M_i$ 按广播理解；router 打分投影 $W_g \in \mathbb{R}^{d \times N_e}$（4.5 式 (1)）不继承任何旧权重，随机初始化。

账：57B total / 14B active。继承的覆盖是部分的——FFN 侧每个 expert 只有一半参数来自 dense checkpoint，另一半与 router 一起从零学起；attention 与 embedding 与 Qwen2-7B 逐项同 shape（3584 hidden / 28 层），直接继承。对照 4.5 式 (1) 的解耦框架：upcycling 用已有 dense checkpoint 替 MoE 的 $N_{\mathrm{tot}}$ 维度预付一部分预训练成本， $N_{\mathrm{act}}$ 口径决定此后的续训与推理价格。

对称性论证也随之改写。朴素拷贝（ $N_e$ 份完全相同拷贝）的隐患是均匀路由下各 expert 收相同梯度、分化只能后验发生；Qwen2 的实际程序在初始化时刻就人为打破对称，两道机制叠加：维度 shuffle 使各 expert 拿到 dense FFN 的不同中间维切片，50% 随机重初始化再去掉切片内部的一半相关性。开放问题因此不再是「分化需要多少 token」，而是这组超参数（50% 比例、shuffle 方案）的取舍依据——公开材料无消融（§5）。Qwen3 的 MoE 双档是否沿用这一初始化路径，官方未说明。

### 2.4 Qwen3-Next：四件套全部来自 P4 零件箱

Qwen3-Next 的 model card 用一行字给出布局：「12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))」（[W]）。这一行里没有新层型——48 层的每个组件都是 P4 已经拆过的零件，创新发生在装配层。逐组件 recap（公式推导一律回指，不重复）：

| 组件 | 公式出处 | 在 Qwen3-Next 的实例 |
| --- | --- | --- |
| GDN 状态更新 | 4.3 式 (14)（[Yang et al., 2024](../references.md)，arXiv:2412.06464，[A]） | `linear_num_key_heads=16`、`linear_num_value_heads=32`、头维 128、conv kernel 4；每 value 头状态 $S \in \mathbb{R}^{128 \times 128}$，每层 32 头合计 1 MiB 定长（bf16）；chunkwise 训练直接调 fla 的 `chunk_gated_delta_rule`（[fla-org](../references.md)，[A]） |
| gated attention 输出门 | 4.4 式 (2) | $`o_t = \sigma(W_g u_t) \odot \mathrm{attn}_t`$；实现上 gate 由 q_proj 输出切片得到（§4.2 代码证据） |
| 3:1 交错与内存账 | 4.4 式 (1) | `full_attention_interval=4`；斜率只剩 12 层 full 的 $12 \times 2\ \mathrm{KiB} = 24$ KiB/token，全 full 反事实的 1/4； $T$ = 262144 时线性项 6.0 GiB + 固定项 36 MiB（4.4 §7 已算） |
| 高稀疏 MoE | 4.5 式 (1)、稀疏度 $\rho = (k + K_s)/N_e$ | 512 routed + 1 shared、top-10，$`\rho = 11/513 \approx 2.1\%`$——4.5 稀疏度谱的最右端 |
| MTP | 4.5 式 (19)–(21) | model card 列 MTP 模块：训练期辅助 loss、推理期 self-speculative 草稿头，挂在主干之上不占 48 层预算 |
| zero-centered norm | 2.7（zero-centered 增益）+ 4.4 式 (3) | $y = \hat{x} \odot (1 + g)$、 $g_0 = 0$；weight decay 拉回恒等而非零映射 |
| partial RoPE | 2.5（部分旋转） | full 层 `head_dim=256` 中 `partial_rotary_factor=0.25`，只旋 64 维；`rope_theta=10^{7}` |

**符号**： $\rho$：激活比例； $K_s$：shared expert 数；其余符号同各出处章。

这一装配回答了 4.4 §8 设计空间网格上的一个具体落点：linear 侧取 GDN 标量门（非 KDA 对角门）、full 侧取 GQA $g$=8 加输出 sigmoid 门（非 MLA、非 NoPE）、位置编码取 partial RoPE 0.25 档、比例取 3:1 等距。网格远未填满，Qwen3-Next 的取点与 Kimi Linear 的取点互不重合——这本身就是「谱系读法」的示范：旗舰模型的架构新闻，读法是它在既有零件网格上的坐标。

## 3. vocab 151936：多语言压缩率的三笔账

Qwen3-8B repo 里的 vocab.json 恰有 151643 个条目，added_tokens_decoder 再挂 26 个特殊 token（id 151643–151668，含 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`），config 的 151936 = 151643 + 26 再向上 padding 对齐（2.2 实测，[A]）。tokenizer 实现是 byte-level BPE 的 Qwen2Tokenizer 家族，从 Qwen2 沿用到 Qwen3-Next，四代没换——**实现**是这个谱系里最稳定的组件；vocab_size 本身有过微调：Qwen2/2.5 代为 152064、Qwen3 代起为 151936（HF config 实测，2026-11-12）。词表规模（两代取点同属 150K 档）的取点逻辑要算三笔账。

**收益账：压缩率换上下文与 FLOPs。** 记压缩率 $\rho_{\mathrm{tok}}$ = bytes/token，一段 $B$ 字节的文本被切成

$$
T = \frac{B}{\rho_{\mathrm{tok}}} \qquad (5)
$$

个 token。byte-level BPE 的保底是 byte fallback：一个 UTF-8 中文字符占 3 字节，词表覆盖不到时按 3 个 byte token 编码；词表规模决定多少常见字符与高频片段能进单 token 档——词表越大，merge 表越长，长尾片段的 $\rho_{\mathrm{tok}}$ 越高，且这一收益在非拉丁文字上最陡（拉丁字母语言的 byte fallback 下界只是 1 token/字符，CJK 是 3）。 $\rho_{\mathrm{tok}}$ 上升同时改善三个量：同一文本的 $T$ 下降（固定 ctx 窗口装下的实际文本变多）、每文本训练 FLOPs $6ND$ 中的 $D$ 下降、perplexity 口径变化（1.1 的陷阱二：CE/token 跨 tokenizer 不可比）。2.2 的五档对照（50257 / 128256 / 151936 / 256128 / 262144）是各厂商按语料多语言配比取的不同工作点；Qwen 的 151936 档对应多语言重配比的取点。

**成本账一：词表参数 $2Vd$。** 全谱系 `tie_word_embeddings=false`，输入 embedding 与输出头是两份参数： $2Vd = 2 \times 151936 \times 4096 \approx 1.24$ B（ $d$=4096 档）。占比随模型规模稀释：8B 档约 15%（按型号名量级估），235B 档同一绝对值只占约 0.5%；Qwen3-Next 的 $d$=2048 把词表参数降到约 0.62 B、占 80B 的约 0.8%。结论形状是：多语言大词表是小模型档的奢侈、旗舰档的零头——谱系能把 151936 一路用到 235B，正因为旗舰账里它几乎免费。

**成本账二：输出头 FLOPs 与 softmax。** 每 token 的输出头是 $2Vd \approx 1.24$ GFLOPs 的一次大 matmul，外加对 $V$ 的 softmax 归一化（训练期每个位置都付）。对照 Qwen3-8B 层部分 $2N \approx 14$ GFLOPs 量级，输出头约占整机前向的 8%——这是 2.2 式 $T \times V \times d$ 三方账里 $V$ 侧的价格。另有两笔隐性成本登记在案：初始 loss 地板 $\ln 151936 \approx 11.9$（1.1）、长尾 token 的冷启动（2.2 §5）。

特殊 token 侧再记一笔：`<think>` / `</think>` 占 added token 段——thinking 模式在模型侧的全部痕迹就是这 26 个 id 里的两个，机制本体归 training 维度。

## 4. 旗舰精读：Qwen3-Next-80B-A3B-Instruct 的 config 与代码

证据等级 [W]：无 arXiv 论文，架构事实以 HF config.json 与 transformers `modeling_qwen3_next.py` 为准（2026-11-11 实测），训练配方数字按 model card 口径降级引用。

### 4.1 逐字段读法

| 字段 | 值 | 读法 |
| --- | --- | --- |
| `num_hidden_layers` | 48 | 36 linear + 12 full，由下个字段切分 |
| `full_attention_interval` | 4 | 每第 4 层为 full attention；transformers 的 configuration 代码据此生成 `layer_types` 列表（48 项，`"linear_attention"` ×36 + `"full_attention"` ×12），modeling 代码按列表逐层装配 |
| `hidden_size` | 2048 | stream 宽度 $d$ |
| `num_attention_heads` / `num_key_value_heads` | 16 / 2 | full 层 GQA $g$=8 |
| `head_dim` | 256 | **head_dim ≠ hidden/heads**：2048/16 = 128 ≠ 256——2.8「常见误解」条目的活实例；q 投影实为 2048→4096，K/V 各 2048→512 |
| `partial_rotary_factor` | 0.25 | RoPE 只旋 256 维中的 64 维（2.5 partial RoPE；与 MiniMax-01 的 64/128 同族不同刻度） |
| `rope_theta` | $10^{7}$ | 2.5 波长账的取点： $\lambda_{\max}$ 量级 $2\pi \cdot 10^{7}$，配合 262144 原生窗 |
| `max_position_embeddings` | 262144 | 原生窗口；可扩到 1010000 为 model card 宣称（[W] 降级） |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 | GDN 侧头数；状态 $S \in \mathbb{R}^{128 \times 128}$ 每 value 头，合计 32 × 128 × 128 ≈ 524 K 元素 = 1 MiB/层定长（4.4 §7） |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | GDN 状态两轴 |
| `linear_conv_kernel_dim` | 4 | GDN 输入侧的 short conv 局部混合（4.3 SSM 谱系惯例） |
| `num_experts` / `num_experts_per_tok` | 512 / 10 | routed 侧；$`\rho \approx 2.1\%`$（4.5 稀疏度谱右端） |
| `shared_expert_intermediate_size` | 512 | 1 个常驻 shared expert，中间维 512 → 约 $3 \times 2048 \times 512 \approx 3.1$ M 参数的小 FFN（4.5 §3.1 的公共知识通道） |
| `norm_topk_prob` | true | top-k 选中后再归一化门控权重——4.5 式 (17) 归一化位置的 config 开关 |
| `vocab_size` | 151936 | §3 的账 |
| `moe_intermediate_size` | 512 | routed expert 中间维（与 shared 同宽）；MoE 参数账由此闭合： $48 \times 512 \times 3 \times 2048 \times 512 \approx 77.3$ B，加 attention / GDN 投影、shared expert、router 等小项后与官方「非 embedding 79B」吻合（model card 口径） |

一个对照读法：hybrid 的层型配比在 config 里有三种写法——Qwen3-Next 用 `full_attention_interval` 周期生成、Kimi Linear 用 `linear_attn_config` 下两个显式列表（`kda_layers` / `full_attn_layers`，还带末层例外）、MiniMax-01 用 `attn_type_list` 全列表。同一结构决策的三种序列化方案，5.6 的 config 字段字典表会回收这个例子。

### 4.2 config → 层装配的 shape 走读

```
config.json                          modeling_qwen3_next.py
num_hidden_layers: 48
full_attention_interval: 4  ──►  layer_types[i] = "full_attention"  (i+1) % 4 == 0
                                              = "linear_attention"  否则
                                  ┌─ 36 × [GDN(k:16×128 → v:32×128, state 128×128) → MoE(512 top-10 + shared)]
                                  └─ 12 × [GatedAttn(16Q/2KV × 256, RoPE 旋 64 维) → MoE(同上)]
rope_theta: 1e7                    所有 full 层共享频率 base；GDN 层不吃 RoPE
```

一层 full 的逐 token KV 是 $2 \times 2 \times 256 = 1024$ 元素（2 KiB，bf16）——12 层合计 24 KiB/token，即 §2.4 表里的斜率来源；GDN 层每 token 读出/写入约 $4 \times 32 \times 128 \times 128 \approx 2.1$ M FLOPs，与 $T$ 无关（4.4 §1 的第二笔斜率）。

### 4.3 modeling 代码的三处实现级证据

读码锚点为 transformers 的 `modeling_qwen3_next.py`（[W] 配套证据，W2 读码确认）：

1. **zero-centered norm**：`Qwen3NextRMSNorm` 的 weight 以 `torch.zeros` 初始化，前向为 `output * (1.0 + self.weight)`——即 $y = \hat{x} \odot (1 + g)$、 $g_0 = 0$（4.4 式 (3)）。增益存成相对恒等的偏差，weight decay 的默认方向是回到恒等映射，对 48 层深主干是逐层「接近恒等、增量塑形」的先验。另有 `Qwen3NextRMSNormGated`（norm 后再乘一道 SiLU 门）挂在 GDN 输出侧。
2. **gated attention**：前向中 `attn_output * torch.sigmoid(gate)`，`gate` 由 q_proj 输出一并切出——即 $`o_t = \sigma(W_g u_t) \odot \mathrm{attn}_t`$（4.4 式 (2)）。2.6 的 GLU 门控思想被搬到 attention 输出：逐元素可关断的读出通道。
3. **GDN kernel 不自研**：chunkwise 路径直接调 fla 库的 `chunk_gated_delta_rule`（[fla-org](../references.md)，[A]）——transformers 只做封装，状态布局与 4.3 §8 的本书约定互为转置（4.4 练习 4 处理过同一差异）。

## 5. 技术开放问题清单

1. **QK-Norm 的独立贡献无量化**：report 与 config 只确认「QK-Norm 替代 QKV bias」的事实（§2.2），公开材料（截至 2026-11-12）无逐项消融——稳定性收益与均值信息移除的代价各是多少，缺数据。
2. **3:1 交错比的依据未公开**：Qwen3-Next 与 Kimi Linear 同取名义 3:1，MiniMax-01 取 7:1；公开文献中没有交错比 $r$ 的系统扫描（4.4 §8），各家的取点是工程判断而非公开结论。
3. **upcycling 的谱系下落与超参数依据**：Qwen2-57B-A14B 之后，Qwen3 的 30B-A3B / 235B-A22B 是否沿用 dense→MoE 初始化未见披露；该程序的 50% 随机重初始化比例与 shuffle 方案的取舍依据也无公开消融。
4. **Qwen3-Next 的 router 机制只能从代码读一半**：config 可见 `norm_topk_prob=true` 与 expert 尺寸，但打分函数形态、是否使用均衡辅助损失等训练侧事实无 report 可查（[W] 边界）。
5. **Qwen3-Max 全字段未披露**：[C] 级的边界实例——谱系表无法填表的一行，只能登记「API-only 旗舰线存在、>1T 为厂商宣称」（2026-11-12）。
6. **thinking 模式的架构代价未量化**：双模式在模型侧仅体现为词表中的 `<think>` 特殊 token 与聊天模板，训练配方归 training 维度；其对架构侧（如更长输出对 ctx 与 cache 的压力）的公开分析缺失。

## 总结

| 内容 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| 谱系表 | §1 表（13 列字段约定） | 双轨制：Qwen3 = dense 6 档 + MoE 2 档；Qwen3-Next 起 dense 缺席；Qwen3-Max [C] 无法填表 |
| GQA 全系化 | 式 (1)；235B 整窗 47 GiB vs MHA 752 GiB | $g$ 逐代收紧（8B 档 4 → 旗舰 16）；262K 窗的前提而非优化 |
| QK-Norm 替代 QKV bias | 式 (2)(3)： $\vert z \vert \le \sqrt{d_k} \approx 11.3$ | 结构性排除饱和区；界与位置、输入幅度无关；消融缺失 |
| upcycling | 式 (4)：复制 $\lceil n h_E / h_{\mathrm{FFN}} \rceil$ 份 + shuffle 切片 + 50% 重初始化 | 归 Qwen2（非 Moonlight）；report §2.2.3 明写源为 Qwen2-7B、shape 互证；对称性在初始化时人为打破 |
| Qwen3-Next 装配 | 48 = 36 GDN + 12 gated attn；512+1 top-10，$`\rho \approx 2.1\%`$ | 四件套全部出自 P4；创新在装配坐标（3:1、输出门、partial 0.25、zero-centered） |
| vocab 151936 | 式 (5)； $2Vd$ ≈ 1.24 B | 四代同一 tokenizer；多语言压缩率换 $T$ 与 $D$；参数占比 15%（8B）→ 0.5%（235B） |
| config 精读 | `full_attention_interval=4` 生成 `layer_types`；head_dim 256 ≠ 2048/16 | hybrid 层型三种 config 写法（对照 Kimi Linear 双列表、MiniMax-01 全列表） |
| 开放问题 | §5 六条 | QK-Norm 消融、 $r$ 扫描、upcycling 下落、router 细节、Max 披露、thinking 架构代价 |

## 练习

1.（计算）用式 (1) 复核：Qwen3-235B（ $L$=94、 $H_{kv}$=4、 $d_{\mathrm{head}}$=128）在 $T$=262144、bf16、 $B$=1 下的 KV cache 总量；Qwen3-8B（ $L$=36、 $H_{kv}$=8）在 $T$=40960 下的总量。再算两模型的「每 token cache」比值，并结合 $g$=16 与 $g$=4 解释为什么旗舰档反而每层更省。
2.（推导）补全式 (3)：写出 Cauchy–Schwarz 与旋转保范性的完整步骤。若 QK-Norm 带可学增益 $\gamma \in \mathbb{R}^{d_k}$（逐通道乘在归一化之后），给出 $|z_{mn}|$ 的新上界（用 $\lVert \gamma \rVert_{\infty}$ 表示），并说明该界为何仍与 $m - n$ 无关。
3.（计算）vocab 账。设某多语言语料在 50257 词表下 $\rho_{\mathrm{tok}}$ = 1.5 bytes/token、在 151936 词表下 $\rho_{\mathrm{tok}}$ = 3.5（假设值）：同一 100 GB 语料的 token 数各是多少？按 $C \approx 6ND$ 口径，训练 FLOPs 差几倍？再算两词表下 Qwen3-8B 尺寸（ $d$=4096、untied）的词表参数 $2Vd$ 各为多少。
4.（代码）读 transformers 的 `configuration_qwen3_next.py` 中 `layer_types` 的生成逻辑，写约 10 行教学代码：由 `num_hidden_layers=48` 与 `full_attention_interval=4` 生成 48 项层型列表并统计两类层数（教学用，非优化实现）。再打开 `modeling_qwen3_next.py`，指出 gate 切片与 zero-centered norm 各在哪一行表达式里（§4.3 的两处证据）。

## 参考

- [Bai et al., 2023](../references.md)：Qwen Technical Report，arXiv:2309.16609，[A]（key: qwen1）——谱系起点锚点
- [Qwen, Qwen1.5 发布博客](../references.md)：[W]（key: qwen1.5）——标准配方承接代，无 arXiv 论文
- [Yang et al., 2024](../references.md)：Qwen2 Technical Report，arXiv:2407.10671，[A]（key: qwen2）——GQA 全系普及、upcycling（57B-A14B ← Qwen2-7B：§2.2.3 明写源模型，§2.2.2 复制 + shuffle + 50% 重初始化程序）
- [Qwen Team, 2024](../references.md)：Qwen2.5 Technical Report，arXiv:2412.15115，[A]（key: qwen2.5）——全尺寸档
- [Yang et al., 2025](../references.md)：Qwen2.5-1M Technical Report，arXiv:2501.15383，[A]（key: qwen2.5-1m）——rope_theta $10^{7}$ + DCA + YaRN 温度的 1M 组合（config 无 rope_scaling 字段）
- [Yang et al., 2025](../references.md)：Qwen3 Technical Report，arXiv:2505.09388，[A]（key: qwen3）——dense 6 档 + MoE 2 档双轨、QK-Norm、30.5B/3.3B 与 235B/22B 口径、thinking 模式
- [Qwen, Qwen3-8B config.json](../references.md)：[A]（key: qwen3-8b-config）——36 层、32/8 heads、vocab 151936、rope_theta $10^{6}$、attention_bias=false
- [Qwen, Qwen3-Next-80B-A3B-Instruct model card / blog / config](../references.md)：[W]（key: qwen3-next）——48 层 3:1 hybrid、512+1 top-10、80B/3B、15T tokens 口径
- [Qwen, Qwen3-Max 官方公告](../references.md)：[C]（key: qwen3-max）——API-only、无 arXiv、HF 404 实测；>1T 为厂商宣称
- [Ainslie et al., 2023](../references.md)：GQA，arXiv:2305.13245，[B]（key: gqa）——§2.1 机制出处
- [Henry et al., 2020](../references.md)：Query-Key Normalization for Transformers，arXiv:2010.04245，[B]（key: qk-norm）——§2.2 机制出处
- [Yang et al., 2024](../references.md)：Gated Delta Networks，arXiv:2412.06464，[A]（key: gdn）——Qwen3-Next linear 层型出处
- [fla-org](../references.md)：flash-linear-attention，[A]（key: fla-repo）——`chunk_gated_delta_rule` kernel 锚点
