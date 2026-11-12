# 9.2 时空 DiT（Spatiotemporal DiT）

> 视频 DiT 的主干问题压成两笔账：时空 token 数 $N = t \cdot h \cdot w$ 进 full attention 的二次项后怎么活（full 3D vs factorized 的完整 FLOPs 推导、全量交叉点 $N = S + t + 2d$、三家 full-3D 系统的质量证据），以及条件与时间轴的进网方式（CogVideoX expert adaLN、HunyuanVideo dual→single stream + MLLM 文本编码器、Wan cross-attn + umT5、3D/分解式 RoPE 频段切分与 8.2 M-RoPE 的数学同构）；收尾是 Wan2.2 SNR-split MoE 全账——timestep 级确定性路由、27B 总参 / 14B 激活，与 4.5 token 级路由逐项对照 | 前置：9.1（latent 几何 $(\tau, s, s)$、首帧约定 $1 + (T-1)/\tau$）、7.6（DiT/adaLN-Zero 式 (8)、MMDiT 式 (10)-(12)、conditioning 三机制）、7.5（FM 目标式 (22)、log-SNR 式 (29)）、2.5（RoPE 内积不变性式 (13)、频率 schedule 式 (16)）、8.2（M-RoPE 三元组式 (14)-(17)）、4.5（token 级 MoE 路由）、4.1（FlashAttention 的 IO 账） | 锚点：zai-org/CogVideoX-5b config [A]、Tencent-Hunyuan/HunyuanVideo（`hyvideo/modules/models.py` 的 `rope_dim_list`）[A]、Wan-Video/Wan2.1（`wan/modules/model.py` 的 `rope_apply`）[A]、Wan-Video/Wan2.2 官方 README 与 Wan-AI/Wan2.2-T2V-A14B 双 expert 目录 [A]；CogVideoX [A]、HunyuanVideo [A]、Wan [A]、StepVideo-T2V [A]、LTX-Video [A]、Open-Sora 2.0 [A]、Movie Gen [B]、Sora [C]、Mochi [C]

## 1. 从 latent 网格到 token 序列：spacetime patchify

9.1 的压缩账把原始视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$ 压成 latent $z \in \mathbb{R}^{t \times h \times w \times c}$（压缩率 $(\tau, s, s)$，首帧约定 $t = 1 + (T-1)/\tau$；Wan-VAE 实例：81 帧 720p → $21 \times 90 \times 160 \times 16$）。本章从 $z$ 接手：DiT 主干不吃特征图，吃 token 序列。把 latent 网格切成 $p_t \times p \times p$ 的时空 patch（spacetime patch）、逐 patch 线性投影到宽度 $d$，得到长度为

$$
N = \frac{t}{p_t} \cdot \frac{h}{p} \cdot \frac{w}{p} \qquad (1)
$$

的 token 序列 $X \in \mathbb{R}^{N \times d}$。

**符号**：$t, h, w$：latent 网格的时间/行/列尺寸（本章 $t$ 在 §1–§5 的 FLOPs 账中指 latent 帧数；扩散时间步在 §4/§5 的 FM 语境（7.5 recap）与 §6 起登场——两个对象都沿用约定页记号，语境自明处不另立符号）；$p_t, p$：时间/空间 patch 边长；$c$：latent 通道（CogVideoX/HunyuanVideo/Wan 均为 16，9.1）；$d$：transformer 宽度。这个「先压进 latent 再切成时空 patch」的两段式表述出自 Sora 技术页（[OpenAI](../references.md)，[C]——闭源，只引其已披露的原话口径：视频先压缩到低维 latent 空间，再把表示分解成 spacetime patches，且「Sora is a diffusion transformer」）。

patchify 投影本身的账很小但值得记一次：每个 patch 是 $p_t p^{2} c$ 维向量，线性投影到 $d$ 维共 $2N \cdot p_t p^{2} c \cdot d$ FLOPs——代入 Wan 几何（patch=1 等效，$c$ = 16，$d$ = 5,120）得 $4.9 \times 10^{10}$ ≈ 0.05 TFLOPs，相对 §2.5 单层 attention 的 PFLOPs 级是四个数量级之下。patchify 的意义不在计算而在接口：它把「3D 网格」翻译成「带三维坐标的序列」，之后全部章节讨论的都是序列侧的设计。

本章的 FLOPs 账按约定页统一取 $p_t = p = 1$ 的等效口径（$N = t \cdot h \cdot w$，一个 latent 位点算一个 token）；有实测 patch 配置的系统逐家标注（CogVideoX 官方 config 实测 $p = 2$，$N$ 相应除 4，见 §2.3 表）。shape 走读：

```
z [t × h × w × c]   9.1 实例：Wan-VAE，81 帧 720p → 21 × 90 × 160 × 16
   │  3D patchify（p_t × p × p；约定页等效口径 p_t = p = 1）
   ▼
X [N × d]           N = t·h·w = 302,400；CogVideoX 实测 p = 2 → N ÷ 4
   │  + 时空位置编码（§8：不进 token 内容，只旋转 attention 的 q/k）
   ▼
L × DiT block（§3–§6 的四种形态：expert adaLN / dual→single / cross-attn / MoE）
   │
   ▼
逐 token 线性层回归速度场 v_θ（FM 目标，7.5 式 (22)）→ unpatchify 回 [t × h × w × c]
```

先把全章的靶子亮出来：$N$ = 302,400 个 token 的 full attention score 矩阵有 $N^{2} \approx 9.1 \times 10^{10}$ 项。图像 DiT 在 $1024^{2}$ 分辨率、DC-AE f32 下的 $N$ 才 1,024（7.6 §8）——视频把 $N$ 抬高了近 300 倍，二次项抬高近 $10^{5}$ 倍。这个二次项怎么付，是 §2 的全部内容。

## 2. Attention 拓扑账：full 3D vs factorized（本章核心推导）

### 2.1 单层 full attention 的 FLOPs

计数口径沿用 1.2（乘加 = 2 FLOPs），假设 MHA（$n_h$ 头、$d_h = d / n_h$；视频 DiT 普遍全头，无 GQA 压缩）。单层 self-attention 逐组件推：

- **QKV 投影**：$[N \times d] \times [d \times d]$ 的矩阵乘为 $2Nd^{2}$，Q/K/V 三个共 $6Nd^{2}$；输出投影再 $2Nd^{2}$。合计 $8Nd^{2}$。
- **score 矩阵** $QK^{\top}$：每头 $[N \times d_{h}] \times [d_{h} \times N]$ = $2N^{2} d_{h}$，$n_h$ 头合计 $2N^{2} d$。
- **softmax**：逐行 exp、归约、归一，约 $3N^{2} n_h$——相对 $2N^{2} d = 2N^{2} n_h d_h$ 低 $d_h$ 倍，说明在案、不进主账。
- **加权求和** $\mathrm{softmax}(\cdot) V$：与 score 同形，$2N^{2} d$。

合计：

$$
C_{\mathrm{full}} = \underbrace{8 N d^{2}}_{\text{proj}} + \underbrace{4 N^{2} d}_{\text{quad}} \qquad (2)
$$

**符号**：$N$：token 数；$d$：宽度。proj 项随 $N$ 线性、quad 项随 $N$ 平方——小 $N$ 时 proj 主导（图像 regime），大 $N$ 时 quad 接管。临界 $N = 2d$：Wan 14B 的 $d$ = 5,120，临界点 10,240 token，视频 token 数轻松越过一个数量级。

### 2.2 factorized：空间层 + 时间层

分解式 attention（factorized attention）把「每个 token 看全部 $N$ 个 token」拆成两个便宜的子层：帧内空间 attention（$t$ 帧各自独立，每帧一个 $S \times S$ 方阵，$S = h \cdot w$）+ 同位时间 attention（$S$ 个空间位各自独立，每位一个 $t \times t$ 方阵）。实现上是两次 reshape：$[t, S, d]$ 内对 $S$ 维做 attention，转置成 $[S, t, d]$ 后对 $t$ 维再做一次。两个子层各自带一套投影，逐组件：

- **空间子层**：全部 $N$ 个 token 过投影 $8Nd^{2}$；score 项 $t$ 帧各付 $4S^{2} d$ → $4tS^{2} d$。
- **时间子层**：投影再 $8Nd^{2}$；score 项 $S$ 个位各付 $4t^{2} d$ → $4St^{2} d$。

合计：

$$
C_{\mathrm{fact}} = 16 N d^{2} + 4 d \left( t S^{2} + S t^{2} \right) = 16 N d^{2} + 4 N d \left( S + t \right) \qquad (3)
$$

**符号**：$S = h \cdot w$：每帧 token 数；第二步用 $N = tS$ 提出公因子。二次项从 $4N^{2} d$ 降到 $4Nd(S + t)$——比值即省幅：

$$
\frac{C_{\mathrm{quad}}^{\mathrm{full}}}{C_{\mathrm{quad}}^{\mathrm{fact}}} = \frac{N^{2}}{N(S + t)} = \frac{t S}{t + S} \qquad (4)
$$

**符号**：与式 (1)–(3) 同。注意 $tS/(t+S)$ 是 $t$ 与 $S$ 的调和均值的一半：省幅由较小的那个维度钉死——$t \ll S$ 时（21 vs 14,400）省幅趋近 $t$，拉长时长对 factorized 的省幅几乎无贡献。

### 2.3 对比表与全量交叉点

式 (2)(3) 联立：factorized 在全量（含投影）上更省当且仅当 $4N^{2} d > 8Nd^{2} + 4Nd(S + t)$，两边消 $4Nd$ 得

$$
N > S + t + 2d \qquad (5)
$$

——factorized 多付的一份投影（$8Nd^{2}$）要求二次项的节省足够大才回本；视频几何下 $S + t + 2d$ 与 $N = tS$ 差一个数量级以上，FLOPs 账没有悬念。代入真实几何（patch=1 等效口径；CogVideoX 行按其官方 config 实测 $p$ = 2 折算）：

| 几何（系统） | $t \times h \times w$ | $N$ | $d$ | full 二次项/层，式 (2) | factorized 二次项/层，式 (3) | 比值，式 (4) |
| --- | --- | --- | --- | --- | --- | --- |
| 49 帧 480×720（CogVideoX-5B，$p$ = 2） | $13 \times 30 \times 45$ | 17,550 | 3,072 | 3.79 TFLOPs | 0.294 TFLOPs | 12.9× |
| 81 帧 720p（Wan-14B 几何） | $21 \times 90 \times 160$ | 302,400 | 5,120 | 1.87 PFLOPs | 89.3 TFLOPs | 21.0× |
| 120 帧 720p（5s@24fps 同几何） | $31 \times 90 \times 160$ | 446,400 | 5,120 | 4.08 PFLOPs | 131.9 TFLOPs | 30.9× |

逐格可复算：第一行 $t$ = 1 + (49−1)/4 = 13、$h \times w$ = (480/8/2)×(720/8/2) = 30×45；二次项 $4 \times 17550^{2} \times 3072 = 3.79 \times 10^{12}$。第三行 $t$ = 31（9.1 的 120 帧实例，首帧约定 $1 + 119/4$ 上取整）。42 层（CogVideoX）或 40 层（Wan）逐层乘上去即单步去噪的 attention 二次项总量：CogVideoX 档 159 TFLOPs vs factorized 12.3 TFLOPs；Wan 几何 74.9 PFLOPs vs 3.6 PFLOPs。

### 2.4 质量证据：省 13–31 倍的东西买不买

FLOPs 账一边倒，但 2024–2026 的前沿开放系统一边倒选 full 3D：CogVideoX 论文正文五处明示 3D full attention（[Yang et al., 2024](../references.md)，arXiv:2408.06072，[A]），StepVideo-T2V 摘要明示 DiT with 3D full attention（[Ma et al., 2025](../references.md)，arXiv:2502.10248，[A]），LTX-Video 摘要明示 full spatiotemporal self-attention（[HaCohen et al., 2025](../references.md)，arXiv:2501.00103，[A]）。中间形态也存在：Open-Sora-Plan v1.5 用稀疏化 3D attention（SUV sparse DiT）而非轴分解（v1.5.0 为官方 repo README 口径，[PKU-YuanGroup/Open-Sora-Plan](../references.md)，[A]；v1.3 论文为 [Lin et al., 2024](../references.md)，arXiv:2412.00131，[A]）——省 FLOPs 但不放弃学出来的时空耦合模式。

机制层面，factorized 省掉的是什么可以精确说清。它的 attention 图是 full $N^{2}$ 的一个稀疏子图：token $(t_{1}, p_{1})$ 与 $(t_{2}, p_{2})$（$p$ 为空间位，$t_{1} \neq t_{2}$、$p_{1} \neq p_{2}$）之间没有直接边，信息必须经中继 token $(t_{2}, p_{1})$ 或 $(t_{1}, p_{2})$ 两跳到达。可以数一下被砍的比例：有直接边的 token 对 = 同帧对 + 同位对 − 自身重复计数 = $tS^{2} + St^{2} - tS$；代入 Wan 几何（$t$ = 21、$S$ = 14,400）：直接边 $4.36 \times 10^{9}$，全部 token 对 $N^{2} = 9.14 \times 10^{10}$——**95.2% 的 token 对在 factorized 里没有直接边**。视频里最常见的对角耦合——物体几帧内移过大幅画面（$t$ 变、$p$ 也变）——恰好全部落在那 95.2% 里；它们只能靠中继位点转发：一层（空间、时间各一次）之内，每对跨帧跨位通信可用的两跳中继恰为 $(t_{2}, p_{1})$ 与 $(t_{1}, p_{2})$ 这 2 个位点。同一结构按每 token 口径读：一个 token 一层内的直接邻居只有 $S + t - 1$ 个（Wan 几何下占 $N$ 的 4.8%，恰为上文 95.2% 的补）。full 3D 则把「哪些位点对哪些位点重要」交回给 attention 权重本身去学，轴分解是把这个邻域结构硬编码进架构。这与 2.5 NoPE 一节的读法同型：硬编码省的是 FLOPs，花的是表达力预算。

factorized 内部还有两个自由度值得点名（均无本章收录条目，按机制分析给出）：(i) **顺序**——先空间后时间还是先时间后空间，在残差流（2.1）上复合顺序影响中继路径的构成；(ii) **堆叠方式**——两个子层合一 block（式 (3) 的口径）还是空间 block 与时间 block 逐层交替（投影开销减半、但每个跨帧跨位通信要等两层）。这两个自由度都不改变 $O(N(S + t))$ 的主项，只改常数因子与两跳路径的长度。

工程侧的使能条件也要记一笔：30 万 token 的 full attention 可行，是因为精确 attention kernel（FlashAttention，[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]；4.1 的 IO 账）用 online softmax（2.3）让 score 矩阵从不落盘——内存 $O(N)$ 级，FLOPs 仍 $O(N^{2})$。若把 bf16 的 score 矩阵真存下来，$N$ = 302,400、40 头就是 $N^{2} \times 2 \times 40 \approx 7.3$ TB——不是「慢」，是不可能。换言之，full 3D 是「FLOPs 贵但内存可行」，这正是 IO-aware kernel 把瓶颈从显存挪回算力之后，架构选择才敢纯按质量投票。

### 2.5 把二次项放回全模型：Wan 14B 单步全层账

孤立看二次项会夸大它的地位——把它放回完整 block 再结一次账。Wan 14B 官方 config 实测：$d$ = 5,120、40 层、40 头、$d_{ff}$ = 13,824、文本长 $M$ = 512（[Wan2.1 repo 与权重 config](../references.md)，[A]）。除 self-attention（式 (2)）外，block 还含 cross-attention（§5，Q 侧 $2Nd^{2}$ + K/V 侧 $4Md^{2}$ + score 与加权和 $4NMd$ + 输出 $2Nd^{2}$）与两层逐位置 FFN（$4Nd\,d_{ff}$；GLU 族三矩阵变体乘 1.5）。代入 $N$ = 302,400（81 帧 720p，patch=1 等效）：

| 组件 | 公式 | FLOPs/层 | 占比 |
| --- | --- | --- | --- |
| self-attn 投影 | $8Nd^{2}$ | 63.4 TFLOPs | 3.1% |
| self-attn 二次项 | $4N^{2} d$ | 1,873 TFLOPs | 91.1% |
| cross-attn 全部 | $4Nd^{2} + 4Md^{2} + 4NMd$ | 34.9 TFLOPs | 1.7% |
| FFN | $4Nd\,d_{ff}$ | 85.6 TFLOPs | 4.2% |
| 合计 | | 2.06 PFLOPs | |

×40 层 = 单步去噪 82.3 PFLOPs；×50 步采样 ≈ 4.1 EFLOPs 生成一段 81 帧视频（仅 DiT 前向，不含 VAE decode 与文本编码）。三个读数值得记住：(i) 二次项占 91%——720p 档的 Wan 就是一台 score 矩阵机器，§6 的 MoE 与 9.3 的步数蒸馏省的都是这 91% 所在的总账（蒸馏 50 步 → 4 步直接砍 92% 生成成本）；(ii) 文本条件 1.7% 近乎免费——文本侧的一切设计空间（§7）都不受成本约束；(iii) FFN 只剩 4.2%——LLM 里 FFN 主导参数账的格局（2.6）在视频 DiT 的 FLOPs 账里被 $N$ 彻底翻转。

这个占比对分辨率敏感：同模型在 480p×81 帧（$N$ = 131,040）下，二次项 351.7 TFLOPs、线性项按 $N$ 同比缩到约 79.8 TFLOPs，二次项占比降到 81.5%——$N$ 减半再多一点（302,400 → 131,040），二次项缩 5.3 倍、线性项缩 2.3 倍，低分辨率档的瓶颈开始向 FFN/投影侧回摆。「二次项占几成」因此不是一个模型属性，而是（模型档 × 分辨率档）的联合函数。

顺手做一次参数量的自洽校验（config 三字段反推 14B）：每层 self-attn QKVO $4d^{2}$ = 104.9M、cross-attn 四投影近似按 $4d^{2}$ 计 104.9M（文本侧 K/V 投影的输入维度以官方 config 为准，此处按 $d$ 近似，差异为低阶项）、FFN $2 d\, d_{ff}$ = 141.6M，每层合计 351.3M，×40 层 = 14.05B——与官方 14B 档吻合。这个校验同时说明 Wan 的参数几乎全在「逐 token 线性」部分（attention 二次项零参数），参数与 FLOPs 的解耦在视频 DiT 上比在 LLM 上更极端：参数账按 $d^{2}$ 长、FLOPs 账按 $N^{2} d$ 长，$N$ 是两者的杠杆比。

## 3. CogVideoX：expert adaLN + 拼接单流的 3D full attention

CogVideoX-5B 官方 config 实测（[zai-org/CogVideoX-5b config](../references.md)，[A]）：42 层、48 头、head_dim 64（→ $d$ = 3,072）、patch 2、文本经 T5-XXL（text_embed_dim 4,096）最长 226 token、sample_frames 49 → $t$ = 13、latent 60×90 → token 网格 13×30×45、$N$ = 17,550。模型档分 2B/5B（论文正文口径）。

结构上的署名设计是 expert transformer + expert adaptive LayerNorm（论文摘要原文口径，[Yang et al., 2024](../references.md)，[A]；官方代码锚点 [zai-org/CogVideo](../references.md)，[A]）：文本与视频 token 拼接成单流序列做 full attention（视频块内即 3D full attention），但 adaLN 的调制参数按模态分设——每个 block 的调制 MLP 对模态 $m$ 回归各自的逐通道系数：

$$
x_i \leftarrow x_i + \alpha_{m(i)} \odot \mathrm{Attn}\Big( \gamma_{m(i)} \odot \mathrm{LN}(x_i) + \beta_{m(i)} \Big), \qquad m(i) \in \{ \text{text}, \text{video} \} \qquad (6)
$$

**符号**：$x_i$：拼接序列第 $i$ 个 token；$m(i)$：其模态标签；$\gamma_m, \beta_m, \alpha_m \in \mathbb{R}^{d}$：模态 $m$ 各自的 scale/shift/gate，由该模态的调制 MLP 从条件向量回归；$\mathrm{Attn}$：拼接序列上的单组 QKV full attention。式 (6) 在「哪些参数按模态分」这条谱系轴上的位置恰好居中：7.6 式 (8) 的 adaLN-Zero 不分（单模态），式 (6) 只分调制参数，7.6 式 (10)-(12) 的 MMDiT 连 QKV 投影与 MLP 都按模态分设。CogVideoX 的选择读法：调制是「同一个去噪过程、两套状态标尺」——文本与视频的激活统计不同，共享 LN 系数会互相拉扯；而 QKV 共享则保留单流的参数效率与拼接 attention 的自由耦合（cross-attn 的「文本只读」限制在这里不存在，见 §7 对照表）。

成本侧回看 §2.3：拼接文本后 $N$ = 17,550 → 17,776，full 二次项 $4 \times 17776^{2} \times 3072 = 3.88$ TFLOPs/层——文本段把 attention 账单抬高了 2.6%（$4d(2NM + M^{2})$，$M$ = 226）。位置编码用 3D RoPE（论文口径），频段切分细节见 §8 的对照表。

expert adaLN 还继承了 adaLN-Zero 的初始化性质（7.6 式 (8) 的推导原样适用）：调制 MLP 末层零初始化 ⇒ 每个 block 以 identity 起步、整网初始预测为零，训练从良定义的标定点出发。差别只在「identity 起步」现在按模态各有一份 gate——文本流与视频流的通路开通速度可以不同，调制参数的逐 block 学习曲线因此自带模态维度的诊断价值。模型档分 2B/5B（论文正文口径，本节 config 数字均为 5B 档）；token 账（式 (1)）与模型档无关——$N$ 由输入几何与 patch 决定，架构决策与规模决策在这套系统里是分离的变量。

## 4. HunyuanVideo：dual→single stream 与 MLLM 文本编码器

HunyuanVideo 官方 README 口径为 13B+ 参数、开源模型中最大（[Kong et al., 2024](../references.md)，arXiv:2412.03603，[A]；该文 ar5iv 不可用，本章事实经官方 repo README 一手取证，[Tencent-Hunyuan/HunyuanVideo](../references.md)，[A]）。主干是 MMDiT 的一个变体：前段双流（视频 token 与文本 token 各自过独立参数的子层、attention 在拼接序列上联合计算——7.6 式 (10)-(12) 的同构形），后段单流（拼接后共享一套 QKV/MLP，退回 7.6 §3(c) 的 in-context 形态）——README §3.2 原文口径即「双流阶段各自独立 block、单流阶段拼接」。同一「double→single」思想在 FLUX 上只有 [C] 级披露（7.6 §7.3），HunyuanVideo 是它的开放 [A] 实例。

参数账：双流段每个 block 为两模态各付一份 QKV/MLP（约 2× 单流 block 参数），单流段一份。把双倍参数预算只花在前段的读法：双流的意义是让文本流在去噪的前若干层被视频状态逐层修正（7.6 §7.1 的「双向」论证），这个修正完成后，后段不再需要维持两套独立表征几何——参数共享的 in-context 形态足够。层数切分点官方 README 未给数值；代码默认 20 双流 + 40 单流（`mm_double_blocks_depth=20`、`mm_single_blocks_depth=40`，hyvideo/modules/models.py 实测）。

为什么修正集中在「前段」说得通（本书机制分析）：transformer 的层序与表征抽象层级相关是跨模态反复出现的现象（3.3 的 ICL 讨论、4.7 的回路测量都建立在浅层局部/深层全局的分层上）。文本条件要影响的是视频的全局构图与运动规划——这些信息在去噪轨迹上恰恰是高噪声段（$t$ 大）定下的（§6.1 的 SNR 分工读法、7.5 §8 的 log-SNR 论据同向）。前段双流让文本表征在全局决策发生之前完成对齐，后段单流只负责把已定型的条件便宜地读下去。这个读法与 FLUX 的 double→single 同型，但 FLUX 只有 [C] 级披露，HunyuanVideo 把它做成了可对照代码的 [A] 实例。

```
dual-stream 段（前段）                  single-stream 段（后段）
x_v ─┐ 各自 QKV / MLP / adaLN          [x_v ; x_t] ── 共享 QKV / MLP
     ├─ 联合 attention（拼接序列）  →      full attention（单流）
x_t ─┘ 文本流逐层被去噪状态修正          文本 token 继续在序列内被读
（7.6 式 (10)-(12) 同构）              （7.6 §3(c) 形态；出口只取视频流）
```

文本编码器是该系统的第二个署名选择：不用 CLIP/T5-XXL，改用 Decoder-Only MLLM，并在其后追加一个额外的双向 token 精修器（bidirectional token refiner）（均官方 README 原文口径）。README 对这一选择给了专门论证（三条理由，见其原文）；机制层面的读法（本书分析，非 README 原文转述）：(i) **监督密度**——CLIP 只对「整图 ↔ 整句」的全局对齐有对比信号，sequence 输出的逐 token 表示未被细粒度监督（7.6 §4 的论据）；decoder-only LM 每个位置都有 next-token 监督，逐 token 表征被充分塑形；(ii) **因果盲区**——causal LM 的第 $i$ 个 token 表征只见过 $1..i$，作为条件供给时右文信息缺失；bidirectional refiner 让每个条件 token 在进 DiT 之前见到完整 prompt，把这层盲区补掉；(iii) **指令语料继承**——MLLM 在指令跟随语料上训过，长而结构化的 prompt（镜头、主体、运动的复合描述）的解析能力直接继承。位置编码为 3D RoPE，代码实测 `rope_dim_list = [16, 56, 56]`（t/h/w 三轴），详见 §8。

## 5. Wan：DiT + FM + cross-attention 走读

Wan 的组合是四家主流里「最 LLM 化」的一版：标准 DiT block（self-attn + FFN、adaLN 族时间调制，7.6 式 (8) 谱系）+ 独立的 cross-attention 子层注入文本 + Flow Matching 目标（[Team Wan, 2025](../references.md)，arXiv:2503.20314，[A]，论文 §4.2；代码锚点 [Wan-Video/Wan2.1](../references.md)，[A]）。14B 档官方 config 实测：$d$ = 5,120、40 层、40 头（$d_h$ = 128）、$d_{ff}$ = 13,824、文本序列长 512；文本编码器为 umT5-XXL（论文 umT5 十五处 + 权重文件名 `models_t5_umt5-xxl-enc-bf16.pth` 互证）。

```
x [N × 5120]        N = t·h·w（patch=1 等效）；时间条件走 adaLN 族调制
 ├─ self-attn（3D full attention + 3D RoPE，40 头）── residual add
 ├─ cross-attn（Q ← x；K/V ← umT5-XXL 文本序列，长 512）── residual add
 └─ FFN（5120 → 13824 → 5120）── residual add
        ×40 层
出口：逐 token 线性层回归 v_θ —— FM 线性路径目标（7.5 式 (22)：v* = ε − x_0）
```

训练目标直接复用 7.5：$x_t = (1-t) x_0 + t\varepsilon$，网络回归条件速度 $\varepsilon - x_0$（式 (22)），采样沿 ODE 从 $t$ = 1 积分到 0（7.5 式 (23)）。§2.5 的全层账已把这套结构的成本钉死：self-attn 二次项 91%、cross-attn 1.7%、FFN 4.2%。cross-attn 的成本公式 $4NMd$ 对 $M$ 线性——文本长度从 77（CLIP 档）放到 512 的代价是二次项的千分位波动，这就是「文本条件设计不受成本约束」的定量含义。

## 6. Wan2.2 SNR-split MoE：timestep 级路由的完整账

Wan2.2 的 A14B 系列把 MoE 搬进了扩散时间轴。本节的全部断言以官方 README 为一手口径（[Wan-Video/Wan2.2](../references.md)，[A]——官方 repo + 开放权重、无论文，引用为 README 原文）；物证是 HF 仓的目录结构：Wan-AI/Wan2.2-T2V-A14B 下就是 `high_noise_model/` 与 `low_noise_model/` 两个子目录，各自 config 为 WanModel、$d$ = 5,120 / 40 层 / 40 头——与 Wan2.1-14B 同构（[Wan2.2-T2V-A14B](../references.md)，[A]）。

### 6.1 路由规则与公式

官方 README 原文口径：A14B 采用双 expert 设计——high-noise expert 负责去噪早期、专注整体布局（overall layout），low-noise expert 负责后期、精修视频细节；切换点由 SNR 决定，SNR 随去噪时间步 $t$ 单调递减，定义阈值步 $t_{moe}$ 对应 SNR 最小值的一半，$t < t_{moe}$ 时切到 low-noise expert（$t$ 向与 W4 约定一致：$t$ 大 = 噪声大 = SNR 小）。

把规则写成函数形式。FM 线性路径 $x_t = (1-t)x_0 + t\varepsilon$ 下，数据端单位方差时信号功率 $(1-t)^{2}$、噪声功率 $t^{2}$：

$$
\mathrm{SNR}(t) = \frac{(1-t)^{2}}{t^{2}}, \qquad \log \mathrm{SNR}(t) = -2 \, \mathrm{logit}(t) \qquad (7)
$$

**符号**：第二式即 7.5 式 (29)（log-SNR 与 logit 的负二倍关系）。$t \in (0, 1]$ 上 $\mathrm{SNR}(t)$ 严格单调递减（$(1-t)/t$ 递减显然），与 README「SNR decreases monotonically」口径一致。路由规则：

$$
e(t) = \begin{cases} e_{\text{high}} & t \ge t_{moe} \\ e_{\text{low}} & t < t_{moe} \end{cases}, \qquad \mathrm{SNR}(t_{moe}) = s^{*} \qquad (8)
$$

**符号**：$e_{\text{high}}, e_{\text{low}}$：high-noise / low-noise 两个 expert 网络；$s^{*}$：切换阈值，官方 README 口径为 $\mathrm{SNR}_{\min}$ 的一半（$\mathrm{SNR}_{\min}$ 为调度上的最小 SNR，其数值与取半的域 README 未展开，见下文的诚实边界）。路由规则的方向与 README 原文一致：$t$ 大（SNR 小，采样早期）走高噪声 expert，$t < t_{moe}$ 切低噪声 expert。

任一 SNR 阈值 $s^{*}$ 对应的时间步可由式 (7) 反解：$(1-t)/t = \sqrt{s^{*}}$，即

$$
t^{*} = \frac{1}{1 + \sqrt{s^{*}}} \qquad (9)
$$

——阈值与步数的换算是闭式的。诚实边界：$\mathrm{SNR}_{\min}$ 的数值与其取半的域（线性域还是对数域）官方 README 未展开，本章给出阈值规则的函数形式与逆映射，不杜撰具体数值；采样调度（shift 等，7.5 §8）改变的是式 (7) 里 $t$ 的有效标定，不改变「SNR 单调 ⇒ 阈值 ⇔ 时间步阈值」这一结构。

```
采样方向   t: 1 ───────────────────────────────► 0
SNR:      最小 ◄── 严格单调递减（式 (7)）──► 最大
expert:   [═ high-noise（布局/早期）═|══ low-noise（细节/后期）══]
                                    t_moe：SNR 阈值切换点，全程仅一次
```

### 6.2 27B / 14B 的账

官方 README 原文：每个 expert 约 14B 参数，总参数量 27B、每步仅 14B 激活。逐项入账：

- **内存**：27B × 2 bytes（bf16）≈ 54 GB——两个 expert 的权重都要在生成全程可用（切换发生在 $t_{moe}$ 一次），MoE 不省显存，与 4.5 的结论同型。
- **每步计算**：任一时刻只有一个 expert 在工作，单步 FLOPs 与一台 14B dense 逐步相同（§2.5 的账原样适用）。对照同存储的假想 dense-27B，省幅取决于放大轴：深度加倍（层数 ×2）时每步恰省一半；宽度放大（$d' = \sqrt{2}\,d$）时二次项只随 $d$ 走而不随参数平方走，按 §2.5 的 91%/9% 混合算总账为 ×1.47，实省约 32%——区间 32%–50%，全程 $K$ 步同比例。
- **容量分配**：dense-14B 要用 14B 覆盖全部 SNR 段；MoE 版每个 SNR 段独占 14B。赌注是「分段专精的收益 > 单模型全段共享的容量损失」——高噪声段决定布局与运动、低噪声段决定细节质感（README 分工口径），这与 7.5 §8 的 log-SNR 读法一致：语义结构在中高噪声段形成，低噪声段的回归残差由高频细节主导，两段的任务统计确实不同。
- **训练侧推论**（结构层面）：每个 expert 只需在自己的 SNR 段上达到指标，等价于把一条 $t$ 轴上的多任务问题切成两个单任务问题。
- **步数预算**：均匀 $K$ 步 Euler 采样（$t_k = 1 - k/K$，7.5 式 (23)）下，high-noise expert 服务 $t \ge t_{moe}$ 的步数份额由式 (9) 给出为 $1 - t^{*} = \sqrt{s^{*}} / (1 + \sqrt{s^{*}})$——阈值越低（越靠近噪声端切换），low-noise expert 承担的步数越多。 timestep shift（7.5 §8 的分辨率相关平移）改变 $t$ 的有效标定，也就等比例移动两 expert 的步数分配；切换点在「步数」与「SNR」两个坐标下是同一事件，SNR 坐标的好处是对 schedule 变换不变——这正是官方 README 用 SNR 而非步数定义切换点的原因（本书机制分析）。

### 6.3 与 4.5 token 级 MoE 的全对照

这不是 4.5 的 MoE 换了个场景，而是路由维度的更换：

| 轴 | 4.5 token 级 MoE（LLM） | Wan2.2 timestep 级 MoE |
| --- | --- | --- |
| 路由粒度 | 每 token × 每层一次决策 | 每 timestep 一次决策，全 token 共享同一 expert |
| router | 可学（打分 + top-$k$ + softmax 归一） | 确定性标量阈值（SNR，式 (8)），无可学参数 |
| 负载均衡 | 辅助损失（4.5 式 (6) 的 $\alpha N_e \sum f_i P_i$）或 aux-loss-free 控制器 | 概念不存在——均衡由调度在时间轴上的步数分配决定，非路由问题 |
| expert 粒度 | 单层 FFN 分片（$N_e$ 份） | 整个网络（40 层全体） |
| 通信 | 每层 all-to-all dispatch/combine | 步内零通信；一次生成只在 $t_{moe}$ 切换一次 |
| 激活比例 | $k / N_e$（如 8/64） | 14B / 27B ≈ 52% |

末行值得停留：token 级 MoE 的稀疏度换来的是容量/计算解耦（4.5），timestep 级 MoE 的 1/2 激活换来的更多是「专精化」而非解耦——它更像把两个按噪声段分工的 dense 模型缝成一个系统，切换逻辑由扩散理论（SNR 单调性）免费提供，router 训练、负载均衡、专家塌缩这些 4.5 的主战场在这里整章缺席。

## 7. Text conditioning 对照：cross-attn vs joint stream vs in-context

7.6 §3 给图像 DiT 立过条件注入的三机制谱系，本章四家系统恰好把谱系坐满。对照表（成本增量以 §2 的账为准，$M$ 为文本长度）：

| 机制 | 文本逐层更新 | attention FLOPs 增量 | 参数增量 | 本章实例 |
| --- | --- | --- | --- | --- |
| cross-attn（文本只读 K/V） | 否 | $+4NMd$ | 每层一组文本侧 K/V 投影（$+4d^{2}$ 量级） | Wan（$M$ = 512：+34.9 TFLOPs/层，为 self-attn 二次项的 1.9%） |
| joint stream（拼接共享 QKV） | 是 | $+4d(2NM + M^{2})$ | 0（共享） | CogVideoX（全程拼接单流，$M$ = 226：+2.6%） |
| joint stream + 模态分参（MMDiT 族） | 是 | 同上 | 文本流整套 QKV/MLP（≈ 翻倍/block） | HunyuanVideo 双流段；CogVideoX 只分 adaLN（式 (6)，折中） |
| in-context（7.6 §3(c) 纯拼接） | 是 | 同上 | 0 | HunyuanVideo 单流段 |

两个结构性读数。其一，成本全都不是约束：最大的一项（Wan cross-attn）也不到 2%——文本侧的设计选择可以完全按「条件质量」投票，这正是 §4 MLLM 编码器这类重资产选择可行的前提。其二，「文本是否逐层更新」是唯一实质差异：cross-attn 里文本表征在整个去噪过程中冻结（只读源），joint stream 里文本 token 逐层被视频去噪状态修正——HunyuanVideo 把修正集中在双流段、后段转共享，CogVideoX 用 expert adaLN 在共享流内保留模态标尺，Wan 干脆放弃文本侧更新、把预算全给视频自注意力。三种取舍目前都有 [A] 级开放系统背书，尚无公断。文本编码器侧的对照：T5-XXL 系（CogVideoX，4,096 维、226 token；Mochi 单 T5-XXL，[C]）、umT5-XXL（Wan，512 token）、Decoder-Only MLLM + bidirectional refiner（HunyuanVideo，§4）。

## 8. 时空位置编码：3D/分解式 RoPE 与 8.2 M-RoPE 的同构

视频 DiT 的 token 有三维坐标 $(t_i, h_i, w_i)$，位置编码要把三者都钉进 attention score。当代开放系统的答案是 2.5 RoPE 的轴向分解版：把每个 head 的 $d_h / 2$ 个旋转对（频率通道）切成三段 $J_t, J_h, J_w$，通道 $j$ 只被自己所属轴的坐标旋转。设 $\theta_j = base^{-2j/d_{h}}$（2.5 式 (16) 的频率 schedule），通道 $j$ 的旋转相位为

$$
\phi_j(t, h, w) = \theta_j \cdot \begin{cases} t & j \in J_t \\ h & j \in J_h \\ w & j \in J_w \end{cases} \qquad (10)
$$

**符号**：$(t, h, w)$：token 的三轴坐标；$\theta_j$：第 $j$ 个旋转对的角频率；$J_t \cup J_h \cup J_w = \{0, \ldots, d_h/2 - 1\}$：频段的连续切分（三段互不重叠）。$\tilde q_j \mapsto e^{i\phi_j}\tilde q_j$ 的复数写法与 2.5 §4.4 相同。

**与 8.2 M-RoPE 的同构是严格的**：8.2 式 (14)-(16) 给理解侧 VLM 的每个 token 构造 $(t, h, w)$ 三元组、按 `mrope_section` 切频段，式 (10) 是同一构造在生成侧的再现。内积不变性逐段成立——2.5 式 (13) 的证明按块对角结构重放一遍，三段各自只含本轴位移差：

$$
\left\langle R_{(t_1,h_1,w_1)}\, q,\; R_{(t_2,h_2,w_2)}\, k \right\rangle = g\big(q, k;\; \Delta t, \Delta h, \Delta w\big), \qquad \Delta t = t_2 - t_1,\ \Delta h = h_2 - h_1,\ \Delta w = w_2 - w_1 \qquad (11)
$$

**符号**：$g$：只依赖内容与三位移差的函数（8.2 式 (17) 同型）。差别只剩两处语境：理解侧的三元组由输入几何给定（8.2 §9.1），生成侧的坐标服务 latent 网格；且视频 DiT 的 attention 全双向（无 causal mask），式 (11) 对所有 token 对成立，无三角约束。

三家实测切分（旋转对计，$d_h$ = 128 → 64 对）：

| 系统 | $J_t$ | $J_h$ | $J_w$ | 出处（均代码实测） |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL（理解侧参照，8.2） | 16 | 24 | 24 | `mrope_section [16,24,24]` |
| HunyuanVideo | 8 | 28 | 28 | `rope_dim_list [16,56,56]`（维数口径，折半为对数） |
| Wan2.1 / Wan2.2 | 22 | 21 | 21 | `freqs.split([c−2⌊c/3⌋, ⌊c/3⌋, ⌊c/3⌋])`，$c$ = 64 → [22,21,21] |

```
64 个旋转对，θ_j = base^(−2j/128)（j 小 = 高频短波长，j 大 = 低频长波长）
Wan:          [0‥21]  t 轴 22 | [22‥42] h 轴 21 | [43‥63] w 轴 21   近均匀
HunyuanVideo: [0‥7]   t 轴  8 | [8‥35]  h 轴 28 | [36‥63] w 轴 28   空间加重
Qwen2.5-VL:   [0‥15]  t 轴 16 | [16‥39] h 轴 24 | [40‥63] w 轴 24   折中（理解侧）
              高频端 ──────────────────────────────────────► 低频端
```

机制读法：$t$ 轴坐标范围小（latent 帧数 13–33 量级），短波长通道在短程内即可无歧义分辨；$h/w$ 轴到 90–160，需要长波长通道覆盖大位移（2.5 §4.5 的波长账）。三家都把 $t$ 段放在高频端，分歧在预算分配：Wan 近均匀、HunyuanVideo 把 7/8 的通道给空间轴——$t$ 轴短到 8 个频率对也够用的判断被写进了代码。对照组是可学绝对位置表：固定表对没见过的 $(t, h, w)$ 组合无定义，在 §9 的 bucketing（多分辨率 × 多时长）下每桶一张表既贵又不可外推——RoPE 系天然变长，这是它在视频侧胜出的直接原因（与 8.2 §8 动态分辨率对视觉塔的同一要求呼应）。

还有一条一致性检查必须成立：图像 = 单帧视频。Wan-VAE 的首帧约定（9.1：首帧只做空间压缩）让一张图像与一段视频共享同一个 latent 空间；式 (10) 侧的对应性质是 $t$ 恒定时自动退化——全部 token 的 $\Delta t$ = 0，$J_t$ 段对 score 的贡献恒为 $g$ 的零位移项，位置编码退化为两轴的 RoPE-2D（6.1 式 (7)(8) 同族）。也就是说，图视频混训（9.3 的训练配方）在位置编码侧零改造：同一套频段切分同时服务「$t$ = 1 的图像桶」与「$t$ = 21 的视频桶」，坐标轴的语义由数据给定而非由架构硬编码。这是分解式 RoPE 相对 learned absolute 表的另一条结构性优势，与 8.2 推论 1（纯文本段 M-RoPE 退化为标准 1D RoPE）是同一条「子集一致性」性质在两个模态侧的实例。

## 9. 分辨率/时长 bucketing

训练要覆盖多分辨率 × 多时长，而 $N$ 随两者剧变——直接混batch会让 padding 浪费主导成本。标准做法是分桶（bucketing）：预设若干 $(T, H, W)$ 档，同档样本组 batch（必要时配合 NaViT 式 packing，6.1，[Dehghani et al., 2023](../references.md)，arXiv:2307.06304，[A]），跨桶变长由 RoPE（式 (10)）天然承载。Wan-VAE 几何下（$(\tau, s, s) = (4, 8, 8)$，patch=1 等效）的五桶账：

| 桶 | $t$ | $h \times w$ | $N$ | full 二次项相对成本（$N^{2}$ 归一） |
| --- | --- | --- | --- | --- |
| 480p × 33 帧 | 9 | 60 × 104 | 56,160 | 1.0× |
| 480p × 81 帧 | 21 | 60 × 104 | 131,040 | 5.4× |
| 720p × 33 帧 | 9 | 90 × 160 | 129,600 | 5.3× |
| 720p × 81 帧 | 21 | 90 × 160 | 302,400 | 29.0× |
| 720p × 120 帧（5s@24fps） | 31 | 90 × 160 | 446,400 | 63.2× |

逐格可复算：$t = 1 + (T-1)/4$（上取整），$h = H/8$、$w = W/8$。两个结构性读数：(i) 时长与分辨率对成本近似乘性（81/33 ≈ 2.5× 与 720/480 ≈ 2.3× 各自独立生效），因为 $N$ 里 $t$ 与 $S$ 本来就是乘积；(ii) 从最低桶到最高桶 63 倍单步成本差——「低分辨率 + 短时长先行、高分辨率 + 长时长收尾」的渐进 schedule（9.3 展开）不是锦上添花，而是让训练总成本落在可行域的必要条件。bucketing 同时是 9.1 深压缩 VAE 与 §2 全 3D attention 的黏合剂：VAE 把每桶的 $N$ 压小一个 $(\tau s^{2})$ 因子，attention 账按平方收下这个礼。

内存侧同样按桶算账：full attention 在 FlashAttention 下 score 不落盘，激活内存的主项是逐层保存的 $N \times d$ 隐状态（训练反向要用），按层数与 $N$ 线性增长；最高桶 $N$ = 446,400、$d$ = 5,120、40 层的激活在 bf16 下约 183 GB（$40 \times N \times d \times 2$ bytes，未计梯度检查点重算，1.2 式 (7) 口径）——这已经超过单卡显存，梯度检查点（1.2）与序列并行（training 维度的分工）因此不是优化项而是入场券。桶的选择于是同时是四本账的旋钮：单步 FLOPs（$N^{2}$）、激活内存（$N$）、单桶数据可得性（长视频数据稀少）、以及课程顺序（9.3）。

## 10. 其他开放系统一段式

**StepVideo-T2V**（[Ma et al., 2025](../references.md)，arXiv:2502.10248，[A]）：30B 参数、最长 204 帧，Video-VAE 为 16×16 空间 + 8× 时间压缩（深压缩一侧，9.1 对照表），DiT 用 3D full attention + FM，双语文本编码器两个。30B dense + full 3D 的组合把 §2.5 的账再抬一档——它是「参数堆容量、full attention 保耦合」路线的最大开放实例。

**Open-Sora 2.0**（[Zheng et al., 2025](../references.md)，arXiv:2503.09642，[A]；代码 [hpcaitech/Open-Sora](../references.md)）：11B、训练成本公开口径 \$200k，架构为 Video DC-AE + MMDiT——用深压缩 VAE（9.1 的 DC-AE 式 tradeoff）把 $N$ 压下来，再用 MMDiT 式双流（§4 谱系）做文本耦合。它回答的是「有限预算下怎么配」：压缩换 token 数、MMDiT 换文本质量。

**LTX-Video**（[HaCohen et al., 2025](../references.md)，arXiv:2501.00103，[A]；[Lightricks/LTX-Video](../references.md)）：把路线走到另一端——压缩率 1:192（每 token 对应 32×32×8 像素块），并把 patchify 从 transformer 输入挪进 VAE 输入（VAE 直接吐出 token 网格）。768×512、121 帧的视频只有 $24 \times 16 \times 16 \approx 6.1$K token：$N$ 小两个数量级，full spatiotemporal self-attention（摘要原文）的二次项不再是约束，2B 级（摘要口径）参数在 H100 上 5 秒视频 2 秒出（faster-than-real-time）。代价在 VAE 侧（9.1 的压缩-质量 tradeoff），不在 DiT 侧。

**Mochi 1**（[Genmo](../references.md)，[C]——无论文，只登记 repo/README 已披露信息）：10B AsymmDiT——non-square QKV、视觉流参数 ≈4× 文本流（更大 hidden 维度）、逐模态独立 MLP，单 T5-XXL 文本编码；配套 AsymmVAE 因果压缩 8×8 空间 × 6× 时间 → 12 通道 latent。「视觉流参数 ≈4× 文本流」是 §7 对照表上参数分配轴的一个极端取点：文本侧只保留够用的读取容量。

**Movie Gen**（[Polyak et al., 2024](../references.md)，arXiv:2410.13720，[B]——Meta 未开源代码/权重，只按论文拆口径）：最大视频 transformer 30B、最大上下文 73K video token（= 16s@16fps）、FM 目标，另有同步音频生成。73K token 这个数值得放进 §2 的坐标系：它小于 Wan 720p 的等效 302K（patch 口径不同），但已足以让 full attention 的训练账必须配序列并行——长上下文在视频 DiT 里的单位不是文本 token，而是「分辨率 × 时长」的乘积预算。

## 总结

| 系统（证据） | 参数 | attention 拓扑 | 文本条件 | 文本编码器 | 时空位置编码 |
| --- | --- | --- | --- | --- | --- |
| CogVideoX-5B [A] | 5B（另有 2B 档） | 3D full，[文;视] 拼接单流 | joint stream + expert adaLN（式 (6)） | T5-XXL（4,096 维，≤226） | 3D RoPE |
| HunyuanVideo [A] | 13B+ | dual→single stream，full | joint stream | Decoder-Only MLLM + bidirectional refiner | 3D RoPE，[8,28,28] 对 |
| Wan2.1-14B [A] | 14B | 3D full | cross-attn（$M$ = 512） | umT5-XXL | 3D RoPE，[22,21,21] |
| Wan2.2-A14B [A]（官方 README） | 27B 总 / 14B 激活 | 同 Wan2.1（两 expert 各一套） | 同 Wan2.1 | 同 Wan2.1 | 同 Wan2.1 |
| StepVideo-T2V [A] | 30B | 3D full | joint（双语文本编码器 ×2） | 双语 ×2 | — |
| Open-Sora 2.0 [A] | 11B | MMDiT（Video DC-AE latent） | joint stream | — | — |
| LTX-Video [A] | 2B 级 | full spatiotemporal（$N$ ≈ 6K，VAE 内 patchify） | — | — | — |
| Mochi [C] | 10B | AsymmDiT（视觉流参数 ≈4× 文本流，更大 hidden 维度、non-square QKV） | joint（非对称） | 单 T5-XXL | — |
| Movie Gen [B] | 30B | —（73K token 上下文，FM） | — | — | — |

| 公式 | 内容 |
| --- | --- |
| 式 (1) | $N = (t/p_t)(h/p)(w/p)$；patch=1 等效 $N = thw$ |
| 式 (2)(3) | full $= 8Nd^{2} + 4N^{2}d$；factorized $= 16Nd^{2} + 4Nd(S+t)$ |
| 式 (4)(5) | 二次项省幅 $tS/(t+S)$；全量交叉点 $N > S + t + 2d$ |
| 式 (6) | expert adaLN：调制参数按模态分设、attention 共享 |
| 式 (7)(8)(9) | $\mathrm{SNR}(t) = (1-t)^{2}/t^{2}$；$t_{moe}$ 阈值路由；$t^{*} = 1/(1+\sqrt{s^{*}})$ |
| 式 (10)(11) | 3D RoPE 频段切分；内积只依赖 $(\Delta t, \Delta h, \Delta w)$（8.2 M-RoPE 同构） |

## 练习

1.（推导）从式 (2)(3) 出发完整推导全量交叉点式 (5)：写出两边相等的临界方程、消去 $4Nd$ 的每一步，并验证 §2.3 表三行几何均满足 $N > S + t + 2d$。进一步回答：固定 $S$、把 $t$ 推向 1（单帧 = 图像），factorized 的二次项省幅式 (4) 趋于什么？这个极限与图像 DiT 不做时空分解的事实是否自洽？

2.（推导）Wan2.2 路由的两问。(a) 由式 (7) 证明 $\mathrm{SNR}(t)$ 在 $(0, 1]$ 上严格单调递减，并推出逆映射式 (9)。(b) 设采样用 $K$ 步均匀步长的 Euler（$t_k = 1 - k/K$，7.5 式 (23)），给定阈值 $s^{*}$，写出 high-noise expert 服务的步数 $K_{high}$ 的闭式，并讨论 $s^{*} \to 0$ 与 $s^{*} \to 1$ 两个极限下两 expert 的分工退化为什么。

3.（计算）按 §2.5 的组件口径，对 HunyuanVideo 几何（$t$ = 31、$h \times w$ = 90×160、$d$ = 3,072、$M$ = 512，patch=1 等效）重算一遍单层账：self-attn 投影、self-attn 二次项、cross-attn 等效项（按 joint stream 拼接计 $+4d(2NM + M^{2})$）、FFN（设 $d_{ff} = 4d$）。二次项占比与 Wan 14B 的 91% 相比如何？$d$ 减半而 $N$ 不变时占比怎么变——用式 (2) 解释。

4.（代码）教学实现 Wan 式 3D RoPE（教学用，非优化实现）：给定 $x \in [B, n_h, N, d_h]$（$d_h$ = 128）与坐标网格 $(t, h, w)$，按 [22,21,21] 切分频段、以 $\theta_j = base^{-2j/128}$ 构造 cos/sin 表并旋转 q/k。断言两条性质（随机张量、数值精度 $10^{-5}$）：(a) 式 (11)——任意两 token 的 score 只依赖 $(\Delta t, \Delta h, \Delta w)$，对全体坐标同加常数平移不变；(b) 把 $w$ 坐标恒置 0 后，结果退化为两轴版——频段 $J_w$ 对 score 的贡献恒为零。

## 参考

- [Yang et al., 2024](../references.md)：CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer，arXiv:2408.06072，[A]（key: cogvideox）——§2.4、§3（expert adaLN、3D full attention）
- [Z.ai, zai-org/CogVideo](../references.md)：[A]（key: cogvideo-repo）——§3 代码锚点
- [Z.ai, CogVideoX-5b config](../references.md)：[A]（key: cogvideox-config）——§2.3、§3（42L/48 头/patch 2/T5-XXL 4096）
- [Kong et al., 2024](../references.md)：HunyuanVideo: A Systematic Framework For Large Video Generative Models，arXiv:2412.03603，[A]（key: hunyuanvideo）——§4（13B+、dual→single、MLLM 编码器，README 口径）
- [Tencent, HunyuanVideo repo](../references.md)：[A]（key: hunyuanvideo-repo）——§4、§8（`rope_dim_list [16,56,56]`）
- [Team Wan, 2025](../references.md)：Wan: Open and Advanced Large-Scale Video Generative Models，arXiv:2503.20314，[A]（key: wan）——§5（DiT+FM+cross-attn、umT5-XXL）
- [Alibaba, Wan-Video/Wan2.1](../references.md)：[A]（key: wan2.1-repo）——§5、§8（`rope_apply` 频段切分）
- [Team Wan, Wan-Video/Wan2.2 官方 README](../references.md)：[A]（key: wan2.2）——§6（SNR-split MoE 全部口径，官方 README 原文）
- [Alibaba, Wan-AI/Wan2.2-T2V-A14B](../references.md)：[A]（key: wan2.2-a14b-hf）——§6（双 expert 目录物证）
- [Ma et al., 2025](../references.md)：Step-Video-T2V Technical Report，arXiv:2502.10248，[A]（key: stepvideo-t2v）——§2.4、§10
- [HaCohen et al., 2025](../references.md)：LTX-Video: Realtime Video Latent Diffusion，arXiv:2501.00103，[A]（key: ltx-video）——§2.4、§10（1:192、patchify 入 VAE）
- [Lightricks, LTX-Video repo](../references.md)：[A]（key: ltx-video-repo）——§10
- [Zheng et al., 2025](../references.md)：Open-Sora 2.0，arXiv:2503.09642，[A]（key: open-sora-2）——§10
- [HPC-AI Tech, Open-Sora repo](../references.md)：[A]（key: open-sora-repo）——§10
- [Lin et al., 2024](../references.md)：Open-Sora Plan，arXiv:2412.00131，[A]（key: open-sora-plan）——§2.4（v1.3 报告）
- [PKU-YuanGroup, Open-Sora-Plan repo](../references.md)：[A]（key: open-sora-plan-repo）——§2.4（v1.5 SUV sparse DiT，README 口径）
- [Genmo, Mochi repo](../references.md)：[C]（key: mochi）——§7、§10（AsymmDiT）
- [Polyak et al., 2024](../references.md)：Movie Gen，arXiv:2410.13720，[B]（key: moviegen）——§10（73K token、30B）
- [OpenAI, Sora 技术页](../references.md)：[C]（key: sora）——§1（spacetime patches 表述）
- [Dao et al., 2022](../references.md)：FlashAttention，arXiv:2205.14135，[A]（key: flashattention）——§2.4（online softmax 使 full 3D 内存可行）
- [Dehghani et al., 2023](../references.md)：Patch n' Pack: NaViT，arXiv:2307.06304，[A]（key: navit）——§9（packing）
- [Peebles & Xie, 2023](../references.md)：DiT，arXiv:2212.09748，[A]（key: dit）——§1、§3（adaLN-Zero 谱系，recap 7.6）
- [Esser et al., 2024](../references.md)：SD3/MMDiT，arXiv:2403.03206，[A]（key: sd3）——§3、§4（双流谱系，recap 7.6）
- [Lipman et al., 2022](../references.md)：Flow Matching，arXiv:2210.02747，[A]（key: flow-matching）——§5、§6（FM 目标，recap 7.5）
- [Liu et al., 2022](../references.md)：Rectified Flow，arXiv:2209.03003，[A]（key: rectified-flow）——§5、§6（线性路径，recap 7.5）
- [Wang et al., 2024](../references.md)：Qwen2-VL（M-RoPE），arXiv:2409.12191，[A]（key: qwen2-vl）——§8（同构对照，recap 8.2）
