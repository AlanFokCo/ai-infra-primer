# 8.5 统一理解与生成（Unified Understanding and Generation）

> 本章收口 Part 8：同一个模型同时做理解（image → text）与生成（text → image）。三条技术路线按「图像以什么形态进主干」分开——**路线 1 离散 token 全 AR**（Emu3 的「图像 = 语言」公式、Chameleon 的 early-fusion 与混合模态稳定性全套：QK-norm / norm 重排 / z-loss / dropout、Janus 的解耦双 encoder 与表征冲突的公式化、Show-o 的 AR + 离散扩散混合目标）；**路线 2 连续 latent**（Transfusion 完整公式： $L = L_{LM} + \lambda L_{DDPM}$、同样本内 text causal + image 块内双向的 mask 矩阵、BOI/EOI、不使用独立 U-Net 骨干（patch 接口为线性层或 U-Net 式 up/down 块可选）、Llama 式 1D RoPE——原文无 2D RoPE）；**路线 3 diffusion head**（LLM hidden state 驱动小型扩散头）；GPT-4o 生图 [C] 的已披露信息分析；收口于三路线对照总表（保真 / 编辑 / 训练稳定 / 推理成本 / 与 LLM scaling 兼容）| 前置：1.1（AR 分解与 NLL）、2.4（KV cache 账）、2.7（Pre-LN 梯度账、QK-Norm 式 (15a)(16)、reordered 摆放）、4.8（吸收态离散扩散式 (1)-(3)、无 KV cache 账）、6.3（VQ-VAE 式 (11)(14)、LFQ、离散 vs 连续选择表）、7.2（ $L_{simple}$ 与 $\varepsilon$-参数化）、7.5（FM 目标）、7.7（Emu3 三笔账、MaskGIT 轮数账）、8.1（组件链、形态轴、token 预算账）、8.3（loss masking）| 锚点：Emu3 [A]（baaivision/Emu3）、Chameleon [B]（权重撤回；结构走读 transformers modeling_chameleon.py）、Janus / Janus-Pro [A]（deepseek-ai/Janus）、Show-o / Show-o2 [A]（showlab/Show-o，show-o2/ 子目录）、Bagel [A]（ByteDance-Seed/Bagel）、Transfusion [B]（Meta 未开源，facebookresearch/Transfusion 404 实测）

## 1. 问题定义：一个参数集，两个方向的条件分布

8.1–8.4 的 VLM 只做理解方向： $p(\text{text} \mid \text{image}, \text{prompt})$。本章把生成方向收进同一个模型——同一个参数集 $\theta$ 同时承担

$$
\underbrace{p_\theta\left( y \mid x_{\text{img}},\thinspace c \right)}_{\text{understanding}} \qquad \text{and} \qquad \underbrace{p_\theta\left( x_{\text{img}} \mid c \right)}_{\text{generation}} \qquad (1)
$$

**符号**： $x_{\text{img}}$：图像； $c$：文本条件（prompt / 对话上下文）； $y$：回答文本； $\theta$：全部共享参数。训练样本两侧同形：图文对 $(x, y)$ 既可以是理解的监督（图在前、文在后），也可以是生成的监督（文在前、图在后）——统一模型在数据层面只是把两个方向的样本装进同一批序列。

真正的问题不在目标形式而在**形态冲突**，8.1 §1 已埋下伏笔：理解方向的主流是连续特征（VQ 码本为重建保真而训，语义可分性不是它的目标，离散化白付量化损失）；生成方向若走 AR 路线则几乎必然离散（softmax 输出头要有限词表，6.3 §7.4）。统一模型必须替图像选一个形态，三个答案引出三条路线：

| 路线 | 图像进主干的形态 | 生成目标 | 代表 |
| --- | --- | --- | --- |
| 1 离散 token | VQ code 网格（6.3） | next-token CE（可与文本同目标，或换离散扩散） | Emu3、Chameleon、Janus（生成侧）、Show-o |
| 2 连续 latent | VAE latent 切 patch | 主干内扩散（ $L_{DDPM}$，7.2） | Transfusion、Bagel（生成专家） |
| 3 diffusion head | 不进主干；主干只产条件向量 | 主干外的小型扩散头 | SEED-X / Emu2 式（无收录条目，机制描述） |

三条路线对同一组问题给出不同答案：图像表征的瓶颈在哪（码本 / VAE / head 容量）、两个方向的梯度在同一参数集上如何相处（Chameleon 的发散与 Janus 的冲突论证是同一场冲突的两个证词）、推理时生成循环在哪里跑（逐 token / 主干内逐步 / 主干外）。本章按路线展开，§9 收对照总表。

与 8.1 三段链的对照先钉住坐标：三段链里 vision encoder 与 connector 是理解方向的专用件，生成方向要么没有（纯理解模型），要么外接一台独立的扩散塔（7.6 的完整生态）。统一路线的激进之处是把「生成」从外挂模块收进主干内部——接口件消失，代价以另外的形态回来，这就是全章的账。

## 2. 路线 1（离散 token 全 AR）：Emu3 的「图像 = 语言」

### 2.1 统一序列与统一 softmax

离散路线的第一步是 tokenizer：图像 $x$ 经 VQ tokenizer（6.3 式 (11) 的最近邻量化，逐空间位点查 codebook）映成离散 id 网格

$$
x \hspace{5mu}\mapsto\hspace{5mu} \left( k_1, \ldots, k_N \right), \qquad k_i \in \lbrace1, \ldots, K\rbrace, \qquad N = \frac{HW}{f^{2}} \qquad (2)
$$

**符号**： $K$：视觉 codebook 大小； $f$：空间压缩率（6.3 记号）； $N$：每图 token 数。Emu3（[Wang et al., 2024](../references.md)，arXiv:2409.18869，[A]；走读锚点 [baaivision/Emu3](../references.md)）的实例账已在 7.7 §8 锁过：codebook $K = 32768$、空间 f8、 $512^{2}$ 图 = $64 \times 64 = 4096$ token，视频在同一 tokenizer 上加时域下采样（论文口径）。

第二步是把视觉 id 与文本 token 放进同一个词表——词表取不相交并 $V = V_{\text{text}} \cup V_{\text{img}}$，拼接后的序列 $u = (u_1, \ldots, u_S)$ 上只有一个 AR 分解：

$$
p_\theta(u) = \prod_{s=1}^{S} p_\theta\left( u_s \mid u_{<s} \right), \qquad u_s \in V, \qquad L = -\sum_{s} \log p_\theta\left( u_s \mid u_{<s} \right) \qquad (3)
$$

**符号**： $p_\theta$：单个 transformer（Emu3 为 8B，论文口径）+ 单个 softmax 头，输出维 $|V_{\text{text}}| + K$。loss 没有模态标记——每个位置的监督都是同一个 next-token CE，「图像 = 语言」的准确含义即此：**对主干而言，一个视觉 code 与一个 BPE token 在嵌入表、softmax、CE 三处地位完全相等**。模态的差别被全部压进 tokenizer（图像如何变成 id）与数据（序列里图文如何排），主干结构零特化。

理解方向在同一条式 (3) 里免费获得：图 → caption 的样本不过是「视觉 code 段在前、文本段在后」的序列，回答位置的 CE 与 8.3 式 (1) 的 loss masking 完全同构（只对回答段计 loss，视觉段与 prompt 屏蔽）。Emu3 论文的主张（摘要口径）：无 diffusion、无组合架构，单一 next-token 目标即在生成与感知任务上超过同期的任务专用模型——「够用」的证据由它提供，「代价」由下一小节算。

### 2.2 质量 vs VQ 信息瓶颈的账

式 (3) 的干净是拿信息论上的一道硬约束换的：图像一旦变成 $N$ 个 $K$ 值 id，每图的信息预算被钉死在

$$
I_{\text{img}} \le N \log_{2} K \qquad (4)
$$

**符号**： $I_{\text{img}}$：tokenizer 通道能携带的每图信息上限（bit）。Emu3 档代入： $4096 \times \log_{2} 32768 = 4096 \times 15 = 61440$ bit/图。对照连续 latent 的 f8 16ch 档（6.3 §7.2）： $64 \times 64$ 位点 × 16 通道，没有这种硬上界——连续通道的容量受重建训练精度与数值精度限制，而不是被码本与网格的组合数钉死。离散路线把上限写成整数，这就是「瓶颈」的准确含义。

瓶颈在两个方向分别兑现。**生成侧**：7.7 §8 (c) 已记——tokenizer 的重建质量（rFID）是全系统的保真上限，主干学得再好也只能在码本张成的子空间里选图；Emu3 把视频也压进同一 tokenizer，瓶颈税由三个模态分摊，tokenizer 训练被推到系统最高优先级。**理解侧**：这是 8.1 §1 形态冲突的回头账——理解本可以用连续特征（语义层级直通、无码本），Emu3 式统一让理解也走同一组离散 id：理解为统一支付了它本不必付的量化税。Emu3 选择接受这笔税，换回的是 7.7 §8 (a) 的接口账归零——无 vision tower、无 connector、无 cross-attention 接口，7.6 §3 的那整套跨模态搭桥结构整体消失。

这笔税交得值不值，取决于税率：codebook $K$ 与压缩率 $f$ 把式 (4) 的预算定在哪一档，以及训练数据能否把码本用满（6.3 §3.4 的 collapse 死锁在大词表下更凶，LFQ 的熵正则正是为此存在，6.3 式 (23)）。§4 的 Janus 给出的是另一种回答：税不交了——理解与生成各用各的 encoder，统一的范围收缩到主干。

## 3. Chameleon：早期融合（early-fusion）的代价——混合模态稳定性全套

### 3.1 早期融合与它的失效面

Chameleon（[Chameleon Team, 2024](../references.md)，arXiv:2405.09818，[B]——官方 repo 在、权重撤回；结构走读锚点 transformers 的 modeling_chameleon.py，[chameleon-modeling](../references.md)，看结构、不对权重）与 Emu3 同属式 (3) 的全 AR 离散路线，差别在训练起点与混合深度：Emu3 之外，Chameleon 是早期融合（early-fusion）路线的代表——图文同词表、同序列、**从头训**，没有任何预训练单模态塔作为初始化资产（对照 §5 的 Show-o 从预训练 LLM 起步）。这使它成为「混合模态从头训练会发生什么」的最干净实验场，而论文给出的答案是：**会发散，且发散有结构**。论文对发散机制的表述（[B]，论文口径）：共享参数下各模态通过增大自身范数互相竞争，激活一旦超出 bf16 的有效表示范围即发散；去掉图像生成支路的纯文本对照训练不发散——发散是混合模态特有的现象（单模态语境下同类问题称 logit drift）。另一个可操作的观测：发散可晚至训练进度的 20–30% 才出现，而最后一层输出范数的失控增长与其强相关——监控输出范数可提前预警。

失效面集中在四个尺度自由度上，每一处都对应一味药：

| 失效点 | 机制 | Chameleon 的药 |
| --- | --- | --- |
| attention logit 漂移 | 视觉 token 与文本 token 的表征分布不同， $q, k$ 范数随训练自由漂移 → 2.3 §3.4 的 softmax 饱和区 | QK-Norm（式 (5)） |
| FFN block 范数增长 | Pre-LN 恒等主干下层贡献沿深度加性堆积，两模态增量幅值分布不同加剧漂移 | norm 重排（式 (6)） |
| 输出 logit 整体平移 | softmax 对 logit 平移不变， $\log Z$ 无任何力约束——QK-Norm 管住了 attention 内部 softmax，管不住输出头 softmax 的 logit 漂移（论文口径） | z-loss（式 (7)） |
| 一般性训练不稳（7B 实证） | 7B 去掉 dropout 的 ablation 训练即发散（论文 Figure 口径）；施加位置在 attention 与 FFN 子层输出之后 | dropout（§3.5） |

四者全部为论文明载的混合模态稳定性技术（原文表述：query-key normalization、revised placement of layer norms、z-loss 与 dropout，"crucial for stable training in the mixed-modal setting"，[B]，论文口径）。下面逐个给机制账。

### 3.2 QK-Norm：把 logit 封顶从激活统计手里拿走

2.7 §7 已推完这条： $1/\sqrt{d_k}$ 只管初始化时刻的 logit 方差，训练中 $\langle q, k \rangle$ 的幅度没有任何机制封顶。Chameleon 把归一化搬进 attention 内部——2.7 式 (15a) 的现代变体形式：

$$
s_{ts} = \frac{\left\langle \mathrm{RMSNorm}(q_t),\hspace{5mu} \mathrm{RMSNorm}(k_s) \right\rangle}{\sqrt{d_k}}, \qquad |s| \le \sqrt{d_k} \qquad (5)
$$

**符号**： $s_{ts}$：位置 $t$ 对 $s$ 的 attention logit；上界为 2.7 式 (16) 的逐 head 口径（Cauchy-Schwarz 一行，RMSNorm 后 $\lVert q_h \rVert = \sqrt{d_k}$）。混合模态为什么更需要它：视觉 code 与文本 token 的 embedding 统计不同，共享的 $W_Q, W_K$ 投影出的 $q, k$ 范数分布因此双模态分化——某些 head 的 logit 被视觉段推出饱和区，梯度经 2.3 式 (9) 的饱和亏空衰减。式 (5) 把 logit 幅度钉死在 $\sqrt{d_k}$，漂移通道只剩归一化增益的学习（2.7 §7 的同一结论）。

### 3.3 norm 重排：给残差主干的加性增长上闸

标准 Pre-LN（2.7 主线）的 block 是 $x_{l+1} = x_l + f_l(\mathrm{Norm}(x_l))$——主干是恒等映射，梯度高速路由此而来（2.7 的残差梯度恒等式），代价是主干方差沿深度加性累加： $\mathrm{Var}(x_L) \approx \mathrm{Var}(x_0) + \sum_l \mathrm{Var}(f_l)$。Chameleon 报告把归一化移到对子层输出起效的重排布局（原文 "revised placement of layer norms"，[B] 论文口径；原文采用的策略引自 Swin 系 block 内归一化工作，收益表述为约束 FFN block 的范数增长；结构与 2.7 §9 的 OLMo-2 reordered 摆放 $x + \mathrm{Norm}(f(x))$ 同族）：

$$
x_{l+1} = x_l + \mathrm{Norm}\left( f_l\left( x_l \right) \right) \qquad (6)
$$

**符号**： $f_l$：attention 或 FFN 子层；Norm 落在子层**输出**上。尺度效应直接：每层对主干的增量先被压回单位尺度再累加， $\mathrm{Var}(x_L)$ 的加性增长被逐层截断——主干尺度漂移这条失效通道被结构性关闭。代价同样写在 2.7 的账里：恒等主干破缺，反向梯度每层多过一次 Norm 的 Jacobian（谱范数 $O(\sqrt{d} / \lVert x \rVert)$，2.7 引理），深层梯度不再直达——混合模态的稳定性收益与纯文本配方优化的方向相反，Chameleon 选择了前者。

### 3.4 z-loss：锁住 softmax 的自由平移度

softmax $p_v = e^{s_v} / Z$、 $Z = \sum_{v'} e^{s_{v'}}$ 对 logit 的整体平移不变（ $s_v \to s_v + c$ 不改变 $p$）——网络因此没有任何力学约束阻止 $\log Z$ 漂移；而 $\log Z$ 近似度量 logit 的整体水平，它的漂移等价于输出分布有效温度的漂移。z-loss 逐位置惩罚 log-partition：

$$
L_z = \mu \sum_{i \in \text{positions}} \left( \log Z_i \right)^{2}, \qquad Z_i = \sum_{v \in V} \exp\left( s_{i,v} \right) \qquad (7)
$$

**符号**： $\mu$：小权重，Chameleon 取 $10^{-5}$（论文口径：[B]）；求和覆盖全部计 loss 位置。机制： $\log Z_i$ 被钉向 0，logit 的整体尺度被锁死，输出头无法靠「全体 logit 一起涨」来锐化分布——锐化只能靠相对差距，即真正的判别学习。混合模态加剧漂移的通道： $V = V_{\text{text}} \cup V_{\text{img}}$ 两个子词表的先验频率相差悬殊（视觉 code 的使用分布由 tokenizer 决定，与文本 Zipf 分布形状完全不同），bias 项学习推着整体 logit 水平跑，式 (7) 是这个通道的配重。

### 3.5 dropout 与 ablation 证据：配方随规模重组

Chameleon 的 dropout 是标准的 Srivastava 式 dropout（[B]，论文口径）：**比率 0.1，施加在 attention 与 FFN 子层的输出之后，仅用于 Chameleon-7B 预训练**；SFT 阶段降为 0.05 并沿用预训练的 z-loss；Chameleon-34B 预训练不使用 dropout。它是针对子层激活的一般性正则，与模态无关——论文没有「随机置弃视觉 token」的 token 级设计，7B 的 ablation（去掉 dropout 训练即不稳，论文 Figure 口径）只证明它是 7B 稳定配方的必要成分。

ablation 证据的结构（论文口径，[B]）比单项数字更重要——**配方随规模加重且重组，而不是简单叠加**：

- **7B = QK-Norm + dropout + z-loss，三者缺一不可**。去掉 QK-Norm 或去掉 dropout 的对照实验训练即发散（论文 Figure 口径）；论文原文表述："For Chameleon-7B it was important to use both dropout and z-loss to achieve stability"。
- **34B = QK-Norm + z-loss + norm 重排**。QK-Norm + dropout 的组合不足以稳定 34B，需要追加 §3.3 的归一化重排；而该重排与 dropout 同用效果不好，34B 因此不用 dropout——**norm 重排与 dropout 部分互替**：论文回溯发现 7B 在使用 norm 重排后也可以不用 dropout 稳定训练。两档共同点是 QK-Norm 不可省（原文："QK-norm is essential in both cases"）。
- **混合模态特有**：去掉图像生成支路的纯文本对照不发散（§3.1 的发散机制即模态间范数竞争）。

这条架构教训的寿命比模型长：把单模态 LLM 的稳定配方（2.7 §9 表）直接搬进统一模型不成立；每个尺度自由度都要单独上锁；且**锁定它们的配方本身随规模改组**——7B 的三件套到 34B 换掉一味。Chameleon 论文公开的价值正在于这份失败模式清单。

## 4. Janus / Janus-Pro：解耦编码器——表征冲突的公式化

### 4.1 结构：双 encoder + 共享主干

Janus（[Wu et al., 2024](../references.md)，arXiv:2410.13848，[A]；走读锚点 [deepseek-ai/Janus](../references.md)，janus/models 目录，两代同 repo）保留式 (3) 的全 AR 主干与离散图像 token，但把视觉编码从「一个 tokenizer 服务两个方向」拆开：

```
理解方向:  x ──► SigLIP 塔（连续特征）──► MLP ──► v ──┐
                                                       ├─► 共享 AR 主干 ──► 文本 / 图像 code
生成方向:  x ◄── VQ decoder ◄── 图像 code ◄────────────┤
           x ──► VQ encoder ──► 图像 code ──►（训练时作序列段）
```

理解侧是 8.1 的标准三段链（SigLIP 系对比塔 + connector + input-level 拼接，6.1/6.2 的 SigLIP 口径）；生成侧是 6.3 的 VQ 离散 token（理解与生成各用各的视觉编码器，主干共享同一个）。Janus-Pro（[Chen et al., 2025](../references.md)，arXiv:2501.17811，[A]）继承结构与配方，做数据与模型规模的扩展（HF 权重实测：Janus-1.3B、Janus-Pro-7B 两档，2026-11-12）。

### 4.2 表征冲突：不变性 vs 充分性

为什么要拆？论文的论证是「两个任务需要不同抽象层级的视觉表征」；给这个直觉一个公式化（本书形式化；实验证据为论文口径）。设共享编码器 $E$ 同时服务两个目标。

理解侧的目标（8.1 式 (1) 的 loss）对 $E$ 的全部要求经过 $z = E(x)$ 传导：回答损失只监督语义判别信息。对任何保持任务标签的像素扰动 $\delta$（亮度抖动、传感器噪声、微移位——数据增强管线里全是这类变换），最优解满足

$$
E\left( x + \delta \right) \approx E(x) \qquad (8)
$$

**符号**： $\delta$：标签保持的像素扰动子空间。式 (8) 是不变性（invariance）约束：理解梯度把 $E$ 推向**压扁** $\delta$ 方向——这部分信息是理解的噪声。

生成侧的编码器由 VQ 重建目标训练（6.3 式 (14)）： $\lVert x - D(E(x)) \rVert^{2}$ 小要求 $E(x)$ 是 $x$ 的近似充分统计量——扰动 $\delta$ 必须能从 $E(x + \delta)$ 读出来，否则重建项直接受损：

$$
\left\Vert (x + \delta) - D\left( E(x + \delta) \right) \right\Vert^{2} \hspace{5mu}\ll\hspace{5mu} \left\Vert \delta \right\Vert^{2} \qquad (9)
$$

**符号**：式 (9) 是充分性（sufficiency）约束： $E$ 必须**保持** $\delta$ 方向的可分辨性。互信息口径的同一对冲突：理解要 $I(z; y)$ 高而 $I(z; \delta)$ 可弃，生成要 $I(z; x)$ 逼近重建下界——同一个 $z$ 上两个互信息目标方向相反。共享 $E$ 的总梯度 $\nabla_E \left( L_{\text{und}} + \mu L_{\text{gen}} \right)$ 在 $\delta$ 子空间上符号相冲，这是「抽象层级冲突」的公式级含义；6.3 §7.4 的选择表（连续族为扩散而设、离散族为 AR 而设）是同一冲突在 tokenizer 层面的投影。

实验证据（论文口径，[A]）：共享单一编码器的统一基线上理解与生成互相拖累，解耦后两端指标各自改善。注意论证边界：冲突证据只针对「单 encoder + 双任务」，不否定统一主干——Janus 的 LLM 仍由两个方向共享训练。代价也清楚：8.1 §2 的接口账部分回归（两个 encoder + 各自的 connector/tokenizer），系统从「一个 transformer」退回「一个主干 + 两副感官」；Janus-Pro 的 scaling 结果（论文口径）说明这个折中在数据与规模扩展下成立。

## 5. Show-o / Show-o2：AR + 离散扩散混合目标

### 5.1 混合目标：文本 CE + 图像去掩码

Show-o（[Xie et al., 2024](../references.md)，arXiv:2408.12528，[A]；走读锚点 [showlab/Show-o](../references.md)）保留离散 token 形态，但给图像段换生成目标——文本照旧 AR（论文称 NTP，next-token prediction），图像 token 走离散扩散（论文称 MTP，mask token prediction）。图像段的腐蚀过程是吸收-均匀混合（Absorbing-Uniform）离散扩散（论文 Eq 4-5）：每步转移矩阵 $Q_t = Q_t^{a} Q_t^{u}$，吸收分量以概率 $\alpha_t$ 把 token 换成 [MASK]，均匀分量以概率 $\beta_t$ 把非 [MASK] token 均匀扩散到词表的其余 code（逐 code 概率 $\nu_t = \beta_t / (K+1)$， $K$ 为视觉码本大小）——4.8 式 (1)-(3) 的纯吸收态是 $\beta_t = 0$ 的特例。记号防混：此处的 $\alpha_t$ 是**掩码概率**，与 4.8 式 (1) 的保留率 $\alpha_t$（ $\alpha_0 = 1$ 为干净端）语义相反，各按所在论文的原文口径。总目标（论文 Eq 9-11）：

$$
L = L_{\text{MTP}} + \alpha\thinspace L_{\text{NTP}}, \qquad L_{\text{NTP}} = -\sum_{i} \log p_\theta\left( v_i \mid v_{<i},\thinspace u \right) \qquad (10)
$$

$$
L_{\text{MTP}} = -\sum_{j:\thinspace m_j = 1} \log p_\theta\left( u_j \mid \tilde{u},\thinspace v \right) \qquad (11)
$$

**符号**： $u = (u_1, \ldots, u_M)$：图像 token； $v = (v_1, \ldots, v_N)$：文本 token； $\tilde{u}$：腐蚀后的图像段，掩码率由时间步控制、调度跟随 MaskGIT / Muse 的采样策略（论文口径）； $m_j = 1$：被掩位置； $\alpha$：NTP 项的配平超参。式 (10)(11) 按本书 loss 约定取负对数似然（最小化），论文原文以最大化似然形式书写（Eq 9-11），两者同构。两个与 4.8 族口径的差异要钉准：(i) 式 (11) 是**裸 CE 求和、无 $1/t$ 权重**——4.8 式 (3) 的 $1/t$ 是纯吸收态加线性调度的 ELBO 特例，Show-o 的 MTP 直接对被掩位置的对数似然求和；(ii) 腐蚀不是「只进 [MASK]」，均匀分量让非掩 token 也会跳到其他 code。式 (11) 与 MaskGIT（7.7 式 (8)，[Chang et al., 2022](../references.md)，arXiv:2202.04200，[A]）同族，差在承载它的网络同时还是文本 AR 模型——同一个主干在两个模态段上执行两种目标。

支撑这个混合的是 omni-attention（论文口径）：文本段 causal mask、图像段块内双向。这是「图像无天然顺序」的第二次出现——§6 的 Transfusion 在连续 latent 侧给出同一个 mask 设计，式 (15) 统一写出。与 Emu3 的对照因此精确：Emu3 让图像也服从严格 raster 序 AR（精确似然、 $N$ 步串行）；Show-o 放弃图像段的精确似然，换迭代去掩码的并行解码—— $T_r$ 轮全序列前向、每轮可提交任意多位置（4.8 的 decode 账与 7.7 的 MaskGIT 轮数账原样适用：甜点 8–12 轮，对照 raster AR 的 $N$ 步）。另一个结构性差别：Show-o 从预训练 LLM 权重初始化（论文口径），文本能力起步即高位，与 Emu3 / Chameleon 的从头训形成三对照——「借 LLM 资产」与「从头统一」是路线 1 内部的又一个分岔。

Show-o2（[Xie et al., 2025](../references.md)，arXiv:2506.15564，[A]）在同一骨架上做改进扩展（论文标题 *Improved Native Unified Multimodal Models*；数据与模态覆盖扩展至视频，论文与 repo README 口径，README 载 NeurIPS 2025 收录）；代码无独立 repo，在 showlab/Show-o 的 show-o2/ 子目录（[showo-repo](../references.md)，[A]）。

## 6. 路线 2（连续 latent）：Transfusion 完整公式

Transfusion（[Zhou et al., 2024](../references.md)，arXiv:2408.11039，[B]——Meta 未开源，facebookresearch/Transfusion 404 实测；以下公式与结构均为论文口径）回答的是：能不能不经过 VQ 量化、让扩散直接发生在 LLM 主干里？可以——把图像表示从离散 code 换成连续 patch latent，把图像段的目标从 CE 换成扩散回归，其余不动。本节是它的完整组装。

### 6.1 序列构造：BOI/EOI 与混合类型序列

图像先过连续 VAE（6.3 §2 族）得 latent 网格，切成 $n$ 个 patch 并线性投影到主干宽度 $d$；文本 token 照常查 embedding 表。两类向量混排进同一条序列，每张图由一对 special token 包住：

$$
u = \left[ \ldots,\hspace{5mu} E(s_a),\hspace{5mu} \mathrm{BOI},\hspace{5mu} \tilde{x}^{(1)}, \ldots, \tilde{x}^{(n)},\hspace{5mu} \mathrm{EOI},\hspace{5mu} E(s_{a+1}), \ldots \right] \qquad (12)
$$

**符号**： $E(s)$：文本 token 的 embedding 行； $\tilde{x}^{(j)} \in \mathbb{R}^{d}$：图像 patch latent（训练时为加噪版，见式 (13)）；BOI / EOI（begin/end-of-image）：词表里的普通 special token，位置按文本处理。patch 进出主干的局部接口，论文给出两种选项——简单线性层，或小型 U-Net up/down 块（论文口径）；且进入线性层之前，每个 patch 向量先加上时间步 $t$ 的 embedding（论文 footnote 口径）——「无 U-Net」的准确含义是**去噪主干**不是独立的 U-Net 塔（§6.4），不否认 patch 接口处可选的小型 U-Net 块。主干是 Llama 式组件（SwiGLU、RMSNorm、RoPE 的标准配方，2.8）；**位置编码沿用 1D RoPE——原文未对图像 patch 引入 2D RoPE**（arXiv 全文无 rotary 位置编码的结构改动，[B] 口径），patch 按 raster 序排布，2D 结构全靠学。对照 8.2 §9：理解向 VLM 已经用 M-RoPE 把 2D 坐标搬进位置编码，Transfusion 站在同一个 tradeoff 的另一端。

```
序列布局（式 (12)）：
[ text tokens ] [BOI] [ x̃⁽¹⁾ … x̃⁽ⁿ⁾ ] [EOI] [ text tokens ]
   离散位置      离散     连续 patch 位置     离散      离散
   └── softmax 头 + CE ──┘ └── 噪声头 + ε 回归 ──┘ └── softmax 头 ──┘
```

### 6.2 训练目标： $L = L_{LM} + \lambda L_{DDPM}$

训练时对每张图采一个共享噪声水平 $t$ 与逐 patch 独立噪声 $\varepsilon^{(j)} \sim \mathcal{N}(0, I)$，patch 输入用前向加噪版（7.2 式 (6)，schedule 口径沿用）：

$$
x_t^{(j)} = \sqrt{\bar\alpha_t}\hspace{5mu} x_0^{(j)} + \sqrt{1 - \bar\alpha_t}\hspace{5mu} \varepsilon^{(j)} \qquad (13)
$$

**符号**： $x_0^{(j)}$：干净 patch latent； $t$：噪声水平，**同一图像的全部 patch 共享一个 $t$**（per-token 独立噪声水平的变体归 Part 9 展开）； $x_t^{(j)}$ 经投影后占据式 (12) 的 patch 位置。

总目标是逐位置分配的异质组合：

$$
L = L_{LM} + \lambda\thinspace L_{DDPM}, \qquad L_{LM} = -\sum_{s \in \mathcal{T}} \log p_\theta\left( u_s \mid u_{<s} \right), \qquad L_{DDPM} = \mathbb{E}_{t,\thinspace \varepsilon}\hspace{5mu} \sum_{j=1}^{n} \left\Vert \varepsilon^{(j)} - \hat{\varepsilon}_\theta\left( u,\thinspace t \right)_j \right\Vert^{2} \qquad (14)
$$

**符号**： $\mathcal{T}$：文本 token 位置（含以 BOI 为目标的预测位）；EOI 由协议追加、非 CE 目标； $p_\theta$：softmax 头输出； $\hat{\varepsilon}_\theta(u, t)_j$：主干在 patch 位置 $j$ 的终层隐状态过线性噪声头，回归目标是式 (13) 注入的噪声（ $\varepsilon$-参数化，7.2 默认口径）； $\lambda$：两 loss 的配平标量。

式 (14) 能在**同一次前向**里成立，靠的是三条位置纪律：(i) 位置类型不相交——文本位置走 softmax 头计 CE，patch 位置走噪声头计回归，两个输出头不共享输出层参数；(ii) 条件结构兼容——patch 位置的条件（前缀文本 + BOI）经 causal attention 流入，与 7.6 的 cross-attention 条件化不同，这里是 in-context 条件：条件就是序列前缀，attention 自己完成注入；(iii) 梯度不冲突于输出头、只交汇于主干—— $\nabla_\theta L$ 两项相加，主干参数同时收 CE 与回归梯度， $\lambda$ 是两个 loss 量级的配平旋钮。与 Chameleon 的单 CE 对照：那是同质目标下的模态混合，这里是异质目标共训——Chameleon 的稳定性问题（§3）换成「两 loss 量级配平」的形态继续存在。

### 6.3 attention mask：text causal + image 块内双向

式 (14) 的 $L_{DDPM}$ 要求 patch 位置在预测噪声时看到全图（去噪本质上是全局结构推断），而 $L_{LM}$ 要求因果性。Transfusion 的 mask 把两者写进一个矩阵：设第 $k$ 个图像块的 patch 位置集为 $B_k$，

$$
M_{ij} = 1 \hspace{5mu}\Longleftrightarrow\hspace{5mu} j \le i \hspace{5mu}\hspace{5mu}\text{or}\hspace{5mu}\hspace{5mu} \left( i, j \in B_k \right) \qquad (15)
$$

**符号**： $M_{ij} = 1$ 允许位置 $i$ attend 位置 $j$。右端两项：因果项（一切位置可见历史）+ 块内项（同一图像块的 patch 互相全见）。BOI/EOI 按文本位置处理（只走因果项）。实例（序列 [w1 w2 BOI p1 p2 p3 EOI w3]，行 = query、列 = key）：

```
      w1  w2  BOI  p1  p2  p3  EOI  w3
w1    1   0   0    0   0   0   0    0
w2    1   1   0    0   0   0   0    0
BOI   1   1   1    0   0   0   0    0
p1    1   1   1    1   1   1   0    0     ← 块内双向行
p2    1   1   1    1   1   1   0    0
p3    1   1   1    1   1   1   0    0
EOI   1   1   1    1   1   1   1    0
w3    1   1   1    1   1   1   1    1
```

块内双向为什么不泄漏标签——这是与 AR 的本质区别所在，值得写成命题。位置 $i$ 的回归目标是 $\varepsilon^{(i)}$，它从 $x_t^{(j)}$（ $j \ne i$）能读到的信息是 $x_0^{(j)}$ 与 $\varepsilon^{(j)}$ 的混合；而各 patch 的噪声独立采样， $\varepsilon^{(j)} \perp \varepsilon^{(i)}$，故

$$
H\left( \varepsilon^{(i)} \mid \left\lbrace x_t^{(j)} \right\rbrace_{j \ne i},\thinspace \text{prefix} \right) = H\left( \varepsilon^{(i)} \right) \qquad (16)
$$

**符号**： $H$：条件熵。块内可见性不降低目标噪声的熵——模型要压低回归误差，唯一通道是借其他 patch 推断共享的干净结构 $x_0$，这正是去噪目标想学的方向（与 7.2 §3.3「网络只从 $(x_t, t)$ 推断」同机制，只是条件里多了同图其他加噪 patch）。三向对照钉死边界：Emu3 的严格 AR 下，位置 $i$ 的目标是**干净 id**，若让它看到未来的干净 id 则标签泄漏（泄漏的是答案本身）；Show-o 的图像段双向看到的未被掩位置是干净 id、但目标是**被掩位置**（答案是隐藏的，可见的是条件——BERT 式合法）；Transfusion 看到的是未来的**加噪**版、目标是噪声——可见内容里不含答案（式 (16)）。三种「未来可见性」的合法性由「目标与可见信息是否共享随机性」一句话裁决。

### 6.4 推理回路与成本账

生图时的回路（文本前缀的 KV cache 全程复用）：

```
1. 文本 AR decode …… 直到模型产出 BOI
2. 追加 n 个 patch 位置，初始化为纯噪声（连续记号的 t = 1 端，7.2 约定）
3. T 步去噪循环：每步全 patch 并行过主干（块内双向，式 (15)）
   → 噪声头出 ε_θ → 按 7.2 §4.4 的反向更新改写 patch 输入
4. 去噪过程结束后，由解码算法机械追加 EOI（论文口径："we append an EOI
   token to the predicted image, and switch back to LM mode"——EOI 不是模型
   采样产出），切回 LM 模式
5. patch 序列 → VAE decoder → 像素；EOI 之后文本继续 AR decode
```

「无 U-Net」的准确含义在此：7.6 的整套扩散塔（adaLN 调制、cross-attention 文本接口、独立参数）整体消失——去噪器就是主干自身，条件化就是前缀注意力。换来的是两个新账。(a) **patch 段无 cache**：每一步的 patch 输入 $x_t$ 都被改写，上一步的 K/V 全部失效，每步成本 ≈ patch 段的一次全量前向（4.8 的离散扩散同一笔账的连续版）。(b) **串行深度的换汇**：以主干前向的 token 位置数为单位（文本前缀 $S_0$ 一次 prefill 后走 cache），

$$
C_{\text{AR}} \approx S_0 + N, \qquad C_{\text{TF}} \approx S_0 + T \cdot n, \qquad C_{\text{head}} \approx S_0 + n_q + T \cdot n \cdot f \qquad (17)
$$

**符号**： $C_{\text{AR}}$：离散全 AR 路线（Emu3 式）的总量； $C_{\text{TF}}$：Transfusion； $C_{\text{head}}$：路线 3； $N$：离散路线的图像 token 数（Emu3 档 4096）； $n$：patch 数； $T$：去噪步数； $n_q$、 $f$：路线 3 的条件 token 数与 head/主干单位成本比（§7）。读法：AR 路线的总 token 前向数最少，但串行深度 = $N$（每步只产 1 token）；Transfusion 把串行深度换成 $T$（扩散步数，7.4 的采样器生态把它压到数十步档），每步宽度 $n$——延迟与总算力在两个路线间重新分配。论文的等 FLOPs 对照（与同族离散模型对比训练，论文口径 [B]）报告连续 latent 路线在图像质量与计算效率上占优；注意证据等级——[B] 无官方代码，该结论以论文原表为准。

## 7. 路线 3：diffusion head——把去噪循环移出主干

路线 2 把扩散搬进了主干；路线 3 反向操作：主干只做理解式的条件化，扩散循环整个外包给一个小型 head。机制骨架：LLM 终层 hidden state（生图指令位、或一组可学 query 位置的代表） $h \in \mathbb{R}^{d}$ 作为条件向量 $c$，一台小型扩散 / FM 解码器（参数比主干小一到两个量级）以 $c$ 为条件跑完整去噪循环，逐块输出图像 latent。SEED-X / Emu2 式系统属此族（本条无收录引用条目——W5 research pass 未收录，按机制描述处理，不引具体文献）。

与路线 2 的分界只有一句话：**去噪循环在哪个网络上跑**。Transfusion 是主干自己跑 $T$ 步；head 式是主干前向一次（或逐条件 token AR 若干步）交出 $c$， $T$ 步循环全部在小 head 内。账与弱点同源于此：

- **成本**：式 (17) 第三行——主干成本从 $T \cdot n$ 降为一次性，head 的 $T \cdot n \cdot f$ 项在 $f \ll 1$ 时近乎免费；路线 3 是三条路线里生图延迟结构最接近纯文本 decode 的。
- **条件化带宽**：每步去噪能读到的条件只有 $c$（或逐 patch 的对应行），不是主干全序列的逐层 attention——§6.2 的 in-context 条件注入（patch 全程 attend 前缀）被换成一次性的向量交接，细节条件化的通道收窄。
- **主干免疫**：主干可以完全不参与扩散训练（甚至冻结），文本能力零风险——这是 8.1 §5「冻结保资产」哲学在生成方向的延伸；代价是生成质量的全部容量压力落在 head 上。

Bagel（[Deng et al., 2025](../references.md)，arXiv:2505.14683，[A]——论文标题为 *Emerging Properties in Unified Multimodal Pretraining*，勿以系统名代标题；走读锚点 [ByteDance-Seed/Bagel](../references.md)）给出一个难以归入单一格子的组合：双视觉编码器（SigLIP 理解 + VAE 生成——Janus 式解耦）+ MoT（mixture-of-transformers）双专家结构（理解与生成 token 各走各的专家参数、joint attention 保持跨模态交互，论文口径）+ 生成侧在连续 VAE latent 上的扩散目标（Transfusion 式主干内扩散，但只在生成专家的参数里跑）；规模账 7B active / 14B total（repo README 口径，[A]）。读法：§4 的「解耦编码器」+ §6 的「主干内扩散」+ 4.5 的「参数按路由分工」——表征冲突问题（式 (8)(9)）的第三种答案：不解耦任务、不解耦编码器，而解耦参数。

## 8. GPT-4o 生图：[C] 口径的已披露信息分析

GPT-4o 的原生生图能力证据等级 [C]：无正式论文、无公开架构，公开信息只有官方发布材料与产品的可观测行为（本条无收录引用条目，按 [C] 纪律只讲已披露信息、不引 arXiv）。已披露信息两层：(a) 官方口径把其生图描述为模型以自回归方式逐段生成，与 DALL·E 系的外挂扩散路线明确区分；(b) 可观测行为——流式渲染中图像按水平带自上而下渐次清晰、文字渲染与多轮编辑一致性显著强于同期扩散产品。兼容性分析（明示：行为推断，非架构证据）：逐带渲染的暴露顺序与 raster 序 token 生成兼容（路线 1 的行为特征）；强指令跟随与长程构图一致性与「LLM 级深度的条件化」兼容（路线 1/2 共有）；但流式 preview 在扩散产品里同样可实现，行为证据不构成决定论。[C] 纪律下只登记「行为与哪些路线兼容」，不断言实现。

## 9. 对照总表：三条路线的账

| 维度 | 离散全 AR（Emu3 / Chameleon） | 离散混合（Show-o） | 连续 latent（Transfusion） | diffusion head（SEED-X 式） |
| --- | --- | --- | --- | --- |
| 表征瓶颈 | VQ 硬瓶颈 $N \log_2 K$（式 (4)） | 同左 | VAE 软瓶颈（6.3 §7.4） | VAE + head 容量 |
| 保真上限 | tokenizer rFID 封顶（7.7 §8 (c)） | 同左 | VAE 重建；无量化台阶 | head 容量 × 条件带宽 |
| 编辑形态 | AR 补全；交错图文原生 | mask 即编辑（inpainting 原生，7.7） | img2img/inpainting 生态可移植（7.4） | 取决于 head 的条件设计 |
| 训练稳定 | 单 CE 目标；混合模态需稳定性全套（§3 四件） | 双目标同主干；掩码与 CE 共存 | 异质 loss， $\lambda$ 配平（式 (14)） | 主干可冻结，最稳 |
| 推理成本（式 (17)） | $S_0 + N$，串行深度 $N$ | $T_r$ 轮全序列双向、无 cache（4.8 账） | $S_0 + T \cdot n$，串行深度 $T$ | $S_0 + n_q + T n f$，主干一次 |
| 与 LLM scaling / 基建兼容 | 完全兼容（纯 next-token；从头训则放弃 LLM 资产） | 高（AR 目标保留；LLM 初始化可借） | 高（主干即 Llama 式；loss 异质） | 最高（主干不动；P2 推理栈原样复用） |
| 证据与锚点 | Emu3 [A]、Chameleon [B] | Show-o / Show-o2 [A] | Transfusion [B]、Bagel（MoT 变体）[A] | 无收录条目（机制描述） |

选型读法与 7.7 §9 的三路线表互补：那张表按生成范式分（diffusion vs AR vs mask-predict），这张表按**统一模型内部的分工**分。格局判断（2026-11-12 口径）：离散全 AR 的系统最简、瓶颈最硬；连续 latent 把瓶颈换软、把异质共训的配平问题引进门；diffusion head 把工程风险降到零、把容量压力集中到 head。冲突问题的三个答案——Janus 解耦编码器、Bagel 解耦参数、Chameleon 硬扛并以稳定性技术全套支付成本——各有已验证的生存区，统一模型的架构尚未收敛。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 问题定义 | 式 (1) | 同一 $\theta$ 承担 $p(y \mid x, c)$ 与 $p(x \mid c)$；形态冲突（8.1 §1）是路线分岔的根 |
| 统一 AR | 式 (2)(3)；Emu3 $K$ = 32768、 $512^{2}$ = 4096 token（7.7 口径） | 单词表并集、单 softmax、单 CE；接口账归零 |
| VQ 瓶颈账 | 式 (4)： $I_{\text{img}} \le N \log_2 K$（Emu3 档 61440 bit/图） | 生成侧保真封顶 + 理解侧白付量化税 |
| Chameleon 稳定性 | 式 (5)(6)(7)（ $\mu = 10^{-5}$）+ dropout 0.1（子层输出后，7B；SFT 0.05；34B 不用） | 四失效点四味药：logit 封顶 / FFN 范数截断 / $\log Z$ 锁 / 一般正则；配方随规模重组：7B = QK-Norm+dropout+z-loss、34B = QK-Norm+z-loss+norm 重排，重排与 dropout 部分互替（[B] 论文口径） |
| Janus 冲突论证 | 式 (8)(9)：不变性 vs 充分性 | 同一 $z$ 上互信息目标相冲；解耦后双端改善（论文口径 [A]）；Janus-1.3B / Janus-Pro-7B（HF 实测） |
| Show-o 混合目标 | 式 (10)(11)： $L = L_{\text{MTP}} + \alpha L_{\text{NTP}}$ | 图像 MTP = 被掩位置裸 CE（无 $1/t$ 权重）；腐蚀为吸收-均匀混合（ $\alpha_t$ 进 [MASK]、 $\nu_t = \beta_t/(K+1)$ 均匀扩散）；omni-attention = 式 (15) 同族 |
| Transfusion | 式 (12)(13)(14)(15)(16)(17) | BOI/EOI 包夹连续 patch； $L = L_{LM} + \lambda L_{DDPM}$ 一次前向两 loss；mask = causal 或同块；无泄漏命题式 (16)；1D RoPE（原文无 2D） |
| diffusion head | 式 (17) 第三行 | 去噪循环移出主干；主干免疫 vs 条件带宽收窄；Bagel = 解耦编码器 + MoT 参数分工 + 主干内扩散（7B active / 14B total，README 口径） |
| GPT-4o [C] | 无公式 | 官方自回归口径 + 行为兼容性分析；不断言实现 |

## 练习

1.（推导）norm 重排的方差账。设 Pre-LN 块 $x_{l+1} = x_l + f_l(\mathrm{Norm}(x_l))$ 与重排块 $x_{l+1} = x_l + \mathrm{Norm}(f_l(x_l))$（式 (6)），假设各层子层输出与主干近似不相关。(a) 证明 Pre-LN 下 $\mathrm{Var}(x_L) = \mathrm{Var}(x_0) + \sum_l \mathrm{Var}(f_l)$（加性累加）；(b) 证明重排块下每层增量的方差被 Norm 压回常数级（与 $\mathrm{Var}(x_l)$ 无关），主干方差随深度线性增长的斜率与 Pre-LN 相差多少？(c) 用 2.7 的 Norm Jacobian 谱范数引理（ $O(\sqrt{d}/\lVert x \rVert)$）说明重排块对反向梯度恒等通路的影响——为什么 Chameleon 愿意为稳定性支付这笔梯度税？
2.（推导）未来可见性的合法性判据。对三种统一设计分别写出「位置 $i$ 的预测目标」「位置 $i$ 能看到的未来信息」与「目标与可见信息是否共享随机性」：(a) Emu3 严格 AR 图像段；(b) Show-o 图像段（部分位置被掩）；(c) Transfusion patch 块（式 (13)-(16)）。由 (c) 证明式 (16)；并说明若把 Transfusion 的回归目标从 $\varepsilon$ 换成 $x_0$、输入保持加噪版，无泄漏结论是否仍成立（提示： $x_t^{(j)}$ 含 $x_0^{(j)}$ 的缩放拷贝）。
3.（计算）推理成本账。取 $S_0 = 1000$ 文本前缀、Emu3 档 $N = 4096$、Transfusion 档 $n = 1024$ patch 与 $T = 50$ 步、head 档 $n_q = 32$、 $f = 1/50$。(a) 按式 (17) 分别计算三条路线的主干 token 前向数；(b) 按「串行深度」（不可并行的步数）重排三路线；(c) Show-o 档： $T_r = 10$ 轮、每轮全序列 $S_0 + N$ 无 cache，把它加进两表并说明它的成本结构更像哪条路线。
4.（代码）教学用，非优化实现：(a) 写一个函数，输入序列的模态分段标记（text / BOI / patch / EOI 位置列表），按式 (15) 输出 $S \times S$ 布尔 attention mask，并用 §6.3 的 8 位置实例验证；(b) 写 Transfusion 混合 loss：文本位置 CE + patch 位置 ε 回归（式 (14)），主干用两层 transformer，随机数据跑通一次前向 + 反向，断言两 loss 的梯度都非零且只交汇于主干参数（检查输出头参数的梯度归属）。

## 参考

- [Wang et al., 2024](../references.md)：Emu3: Next-Token Prediction is All You Need，arXiv:2409.18869，[A]（key: emu3）——式 (2)(3)(4) 的统一 AR 路线与 tokenizer 实例账（ $K$ = 32768、 $512^{2}$ = 4096 token）
- [BAAI, baaivision/Emu3](../references.md)：[A]（key: emu3-repo）——路线 1 走读锚点
- [Chameleon Team, 2024](../references.md)：Chameleon: Mixed-Modal Early-Fusion Foundation Models，arXiv:2405.09818，[B]（key: chameleon）——early-fusion 与稳定性全套（QK-Norm / norm 重排 / z-loss / dropout）出处；权重撤回
- [HuggingFace, transformers modeling_chameleon.py](../references.md)：[B]（key: chameleon-modeling）——Chameleon 结构走读锚点（看结构、不对权重）
- [Wu et al., 2024](../references.md)：Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation，arXiv:2410.13848，[A]（key: janus）——解耦双 encoder 与表征冲突论证
- [Chen et al., 2025](../references.md)：Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling，arXiv:2501.17811，[A]（key: janus-pro）——数据与规模扩展（Janus-1.3B / Janus-Pro-7B）
- [DeepSeek-AI, deepseek-ai/Janus](../references.md)：[A]（key: janus-repo）——Janus / Janus-Pro 走读锚点（janus/models 目录）
- [Xie et al., 2024](../references.md)：Show-o: One Single Transformer to Unify Multimodal Understanding and Generation，arXiv:2408.12528，[A]（key: show-o）——AR + 离散扩散混合目标式 (10)(11) 与 omni-attention
- [Xie et al., 2025](../references.md)：Show-o2: Improved Native Unified Multimodal Models，arXiv:2506.15564，[A]（key: show-o2）——同骨架改进扩展（含视频模态）
- [Show Lab, showlab/Show-o](../references.md)：[A]（key: showo-repo）——Show-o / Show-o2 走读锚点（show-o2/ 子目录）
- [Zhou et al., 2024](../references.md)：Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model，arXiv:2408.11039，[B]（key: transfusion）——式 (12)-(17)：BOI/EOI、 $L = L_{LM} + \lambda L_{DDPM}$、块内双向 mask、1D RoPE；Meta 未开源
- [Deng et al., 2025](../references.md)：Emerging Properties in Unified Multimodal Pretraining（BAGEL），arXiv:2505.14683，[A]（key: bagel）——双 encoder + MoT 双专家 + 连续 latent 扩散的组合
- [ByteDance Seed, ByteDance-Seed/Bagel](../references.md)：[A]（key: bagel-repo）——7B active / 14B total 规模账（README 口径）
- [van den Oord et al., 2017](../references.md)：Neural Discrete Representation Learning，arXiv:1711.00937，[A]（key: vq-vae）——式 (2) 的量化器出处（经 6.3）
- [Esser et al., 2020](../references.md)：Taming Transformers for High-Resolution Image Synthesis，arXiv:2012.09841，[A]（key: vqgan）——VQ tokenizer 的感知/对抗训练（经 6.3）
- [Yu et al., 2023](../references.md)：Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation，arXiv:2310.05737，[A]（key: magvit-v2）——「tokenizer 是质量上限」判断的出处（经 6.3 §6）
- [Ho et al., 2020](../references.md)：Denoising Diffusion Probabilistic Models，arXiv:2006.11239，[A]（key: ddpm）——式 (13)(14) 的加噪与 ε 回归口径（经 7.2）
- [Chang et al., 2022](../references.md)：MaskGIT: Masked Generative Image Transformer，arXiv:2202.04200，[A]（key: maskgit）——式 (11) 的去掩码生成族（经 4.8 / 7.7）
- [Zhai et al., 2023](../references.md)：Sigmoid Loss for Language Image Pre-Training，arXiv:2303.15343，[A]（key: siglip）——Janus / Bagel 理解侧对比塔（经 6.1/6.2）
- [Henry et al., 2020](../references.md)：Query-Key Normalization for Transformers，arXiv:2010.04245，[B]（key: qk-norm）——式 (5) 的组件出处（经 2.7 §7）
