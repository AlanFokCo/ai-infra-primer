# 4.6 长上下文（Long Context）

> 本章把「上下文长度」从一个 config 数字拆成四笔互相独立的账：attention FLOPs 的 $O(T^{2})$、KV cache 的 $O(T)$、softmax 的熵稀释（ $\ln T$）、RoPE 的角度出域；随后推导 RoPE 缩放族的统一形式——PI / NTK-aware / YaRN / LongRoPE 都是逐通道位置压缩率 $\eta_j$ 这一条曲线的不同参数化——以及 dual-chunk attention 的三段位置重编号；架构侧对照 SWA 堆叠、Ring Attention 与 GQA / MLA / DSA / KDA 在 1M token 下的 cache 实账；最后落到训练配方（staged extension、长文档上采样）与评测方法论（needle 的局限、RULER、claimed vs effective context） | 前置：2.3（softmax 方差账）、2.4（KV cache 式 (9)、decode 读取式 (11)）、2.5（RoPE 主命题式 (13)、波长账式 (16)、§6 外推失败机制）、4.1（MLA cache 576 元素 / token / 层、FlashAttention 定位）、4.2（SWA 账、attention sink、DSA top-2048）、4.3（KDA 固定状态） | 锚点：DeepSeek-V3 config（YaRN 字段实例）[A]、Qwen2.5-1M 报告 [A]、Llama-3.1 herd paper（128K 配方）[A]、Kimi-Linear config [A]；PI [B]、YaRN [A]、LongRoPE [B]、RULER [A]、StreamingLLM [A]

## 1. 四堵墙：长度方向的四笔独立账

模型卡片上「支持 1M token 上下文」这一个声明，背后是四笔互相独立的账：算力、显存、softmax 的统计结构、位置编码的值域。任何一笔不付清，claimed context 就只是 config 里的一个数字。本节逐笔公式化；§2–§4 的每个方法都可以标注它付的是哪一笔。

### 1.1 墙一：attention FLOPs 随 $T$ 平方增长

causal attention 每层的 score 加权 FLOPs 为 $2T^{2}d$（ $d = H_q \cdot d_{\mathrm{head}}$；2.3 §7 与 4.1 §1 的口径，FlashAttention 不改变这笔账，只改变它的执行方式）。代入 $T = 10^{6}$、Llama-3.1-8B 形态（ $d$ = 4096、 $L$ = 32，[Meta, Llama-3.1-8B config](../references.md)，[A]）：

$$
2 \times (10^{6})^{2} \times 4096 \approx 8.2 \times 10^{15} \qquad (1)
$$

单位 FLOPs，单层账；32 层合计 $\approx 2.6 \times 10^{17}$ FLOPs。对照线性部分：每 token $2N$（ $N \approx 8 \times 10^{9}$）乘 $10^{6}$ 个位置 $\approx 1.6 \times 10^{16}$ FLOPs——1M 处 attention 二次项已是线性部分的约 16 倍。交叉点有闭式：

$$
2T^{2}dL = 2NT \quad \Longrightarrow \quad T^{\ast } = \frac{N}{dL} \approx \frac{8 \times 10^{9}}{4096 \times 32} \approx 6.1 \times 10^{4} \qquad (2)
$$

**符号**： $N$：参数量（非 embedding 口径）；$`T^{*}`$：二次项开始主导的序列长度。8B 形态下 $`T^{*}`$ 约 6 万 token——超过这个长度，扩窗的边际成本全部来自 attention 项。时间维度同构： $2.6 \times 10^{17}$ FLOPs 在约 $10^{15}$ FLOPS（H100 bf16 dense 量级）单卡上的理想下界约 260 秒，100% MFU 不可达，实际再乘数倍——墙一首先是 prefill 延迟墙。

内存侧补一笔：若物化 scores 矩阵 $[B, H, T, T]$，单序列单层 $10^{12} \times 32 \times 2 \times 2$ B = 128 TB（ $10^{12}$ 个 score 位置 × 32 heads × scores 与 probs 两张 × bf16）——任何实现都不能走这条路。FlashAttention 的分块 + online softmax 把物化消掉（4.1 §5），但 $O(T^{2})$ 的 FLOPs 照付。墙一因此转化为一个调度问题：§3.2 的 Ring Attention 把它分摊到多设备，4.2 的稀疏化与 4.3 的线性化把它从指数上改写。

### 1.2 墙二：KV cache 随 $T$ 线性增长

2.4 式 (9)： $\mathrm{bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{\mathrm{head}} \cdot T \cdot B \cdot b$。 $T = 10^{6}$、 $B$ = 1、bf16 的实数对照（逐模型形态代入，除最后一行外均为公式外推，非该模型的官方窗口）：

| 形态（锚点） | 每 token 每层 | 层数 | $T = 10^{6}$ 单序列 cache |
| --- | --- | --- | --- |
| GQA-8（Llama-3.1-8B， $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128）[A] | $2 \times 8 \times 128 \times 2$ B = 4 KiB | 32 | 122 GiB |
| MLA（DeepSeek-V3，576 元素 / token / 层，4.1 §4）[A] | $576 \times 2$ B = 1152 B | 61 | $\approx 65.4$ GiB |
| DSA（DeepSeek-V3.2，主路同 MLA + indexer 小 cache，4.2 §6）[A] | 1152 B（驻留不变） | 61 | $\approx 65.4$ GiB |
| KDA hybrid（Kimi-Linear，7 层 MLA + 20 层固定状态）[A] | 7 层 $\times$ 1152 B + 20 层 $\times$ 0 | 27 | $\approx 7.5$ GiB + 20 MiB |

KDA 行的两笔：full-attn 层（MLA，576 元素 / token / 层）随 $T$ 线性： $7 \times 576 \times 10^{6} \times 2$ B $\approx 7.5$ GiB；KDA 层的状态是定数——每 head $d_v \times d_k$ = $128 \times 128$ 元素、32 heads、bf16 = 1 MiB / 层，与 $T$ 无关（4.3 §3 的 decode 账），20 层共 20 MiB。同尺度全 MLA 的 27 层反事实是 $27 \times 576 \times 10^{6} \times 2$ B $\approx 29$ GiB——hybrid 结构把线性增长项压到约 26%（论文 headline 数字的口径对照在 4.4，此处只记由 config 推出的账）。

decode 每步读取同构（2.4 式 (11) 分母的 cache 项）：GQA-8 在 1M 处每步要读 122 GiB，按约 3 TB/s 的 HBM 带宽量级下单序列理论下界约 40 ms / token——这是「1M decode 为什么慢」的账源，也是 4.3 / 4.4 的固定状态架构在长窗 decode 上加速的账源。DSA 的读法是另一维：cache 驻留不变，但每步只 gather top-2048 行（4.2 §6 的账），1M 处读取占比约 0.2%。

### 1.3 墙三：softmax 的熵稀释（attention dilution）

前两堵墙是工程账，这堵墙是统计结构：softmax 的行归一化意味着 attention mass 是总量恒为 1 的预算， $T$ 越大，摊得越薄。

**推导 1（熵上界 $\ln T$）**。 $T$ 个 key 上的任意 attention 分布 $p$ 满足

$$
H(p) = -\sum_{i=1}^{T} p_i \ln p_i \le \ln T \qquad (3)
$$

（Jensen 不等式： $H = \mathbb{E}[-\ln p_i] \le -\ln \mathbb{E}[p_i] = \ln T$），等号在均匀分布取得。读法：均匀摊开时每个 key 的保底份额是 $1/T$—— $T$ 涨 128 倍，单 key 份额缩 128 倍。模型要么接受稀释，要么把分布做尖。

**推导 2（分数差距稀释）**。设目标 key 的 logit 为 $\mu + \Delta$（ $\Delta$ 是内容优势），其余 $T - 1$ 个干扰 key 的 logit iid $\sim \mathcal{N}(\mu, \sigma^{2})$。mean-field 近似（分母用期望替代）：

$$
p^{\ast } = \frac{e^{\Delta}}{e^{\Delta} + \sum_{j=1}^{T-1} e^{z_j}} \approx \frac{e^{\Delta}}{e^{\Delta} + (T-1)\thinspace e^{\sigma^{2}/2}} \qquad (4)
$$

**符号**： $z_j$：干扰 logit 相对 $\mu$ 的偏差； $\mathbb{E}[e^{z}] = e^{\sigma^{2}/2}$（对数正态均值）。目标拿到半数 mass 的条件：

$$
p^{\ast } \ge \frac{1}{2} \quad \Longleftrightarrow \quad \Delta \ge \ln(T-1) + \frac{\sigma^{2}}{2} \approx \ln T + \frac{\sigma^{2}}{2} \qquad (5)
$$

**所需的 logit 优势随 $\ln T$ 线性增长**。模型在训练窗 $T_{\mathrm{train}}$ 上学到的 $\Delta$ 预算只够覆盖 $\ln T_{\mathrm{train}}$；窗口外推 $s$ 倍，凭空需要额外 $\ln s$ 个 nat——从 8K 到 1M（ $s$ = 128）是 $\ln 128 \approx 4.8$ nat。不付这笔账，目标 mass 按式 (4) 以约 $1/T$ 的速率稀释：「score 里 target 排第一」不等于「attention 给了 target 足够份额」。这就是 claimed 与 effective context 之间鸿沟的 softmax 侧机制（§5 的评测对照），也是 attention sink 的同一枚硬币：当没有 key 配得上 $\Delta \sim \ln T$ 级优势时，行归一化迫使 mass 倒向句首的 sink token（4.2 §4，[Xiao et al., 2023](../references.md)，arXiv:2309.17453，[A]）。

**与温度修正的接口**。softmax 温度恒等式（ $\beta = 1/t$、 $Z(\beta) = \sum_j e^{\beta z_j}$）：

$$
H = \ln Z(\beta) - \beta \thinspace \partial_{\beta} \ln Z, \qquad \frac{\mathrm{d}H}{\mathrm{d}\beta} = -\beta \thinspace \mathrm{Var}_{p}(z) \le 0 \qquad (6)
$$

提高 $\beta$（加锐）单调减熵——方向正确。§2.3 会看到 YaRN 的温度修正正是按 $\ln s$ 的函数形式补回这份 score 差距。

### 1.4 墙四：RoPE 角度出训练域

2.5 §6 已定格机制，此处只登记接口并补一个对偶观察。失败不是数值问题—— $\cos / \sin$ 在任意角度有定义——而是分布外问题：慢通道（ $\lambda_j > W$，窗内转不满一圈）训练全程只见过角度区间 $[0, W\theta_j]$，推理位置越过 $W$ 后这些通道的输入进入未训区间，score 分布漂移，perplexity 在窗界处断崖上升。失败集中在慢通道，所以管全局位置的远程依赖先断、近程行为相对保持。

对偶观察（为 §2 铺垫）：快通道也有失败模式，但方向相反。任何把位置压小的修复（ $m \to m/\eta$）都在压缩相邻位置的相位差——快通道 $\theta_0$ = 1 rad / token 的相位标定是网络对「相邻」「隔几个」这类短距关系的全部依据，压缩 $\eta$ 倍后短距相位差落入训练时对应「零距离」的邻域，局部分辨率先受损。缩放族的全部设计就是在「慢通道出域」与「快通道压缩」之间按频率分账。

## 2. RoPE 缩放族：统一成一条逐通道压缩率曲线

2.5 式 (16) 的频率 schedule $\theta_j = base^{-2j/d}$、波长 $\lambda_j = 2\pi \cdot base^{2j/d}$（ $d$ 为旋转维数， $j = 0, \ldots, d/2 - 1$）是本章的公共语言。设原训练窗 $W$、目标窗 $sW$（ $s$ 为扩展因子）。全章的统一形式化：**每种缩放方法都是给第 $j$ 个频率通道指定一个位置压缩率 $\eta_j \ge 1$**——角度 $m\theta_j$ 变成 $m\theta_j / \eta_j$，等价于位置 $m \to m / \eta_j$、波长 $\lambda_j \to \eta_j \lambda_j$。设计空间是一条曲线 $\eta : \lbrace0, \ldots, d/2-1\rbrace \to [1, s]$，端点约束由 §1.4 给定：慢通道要 $\eta_j \approx s$（把角度压回训练域），快通道要 $\eta_j \approx 1$（保局部分辨）。四种方法只是这条曲线的四种参数化。

### 2.1 PI：常数曲线 $\eta_j = s$

位置插值（position interpolation，PI；[Chen et al., 2023](../references.md)，arXiv:2306.15595，[B]：无官方 repo，按论文拆）是最直接的取法：所有位置除以 $s$，

$$
m \to \frac{m}{s} \qquad \Longleftrightarrow \qquad \theta_j \to \frac{\theta_j}{s} \qquad (7)
$$

（波长同乘 $s$。）即常数曲线 $\eta_j = s$。它精确解决了慢通道出域： $m \le sW$ 时 $(m/s)\theta_j \le W\theta_j$，全部通道的角度回到训练见过的区间——PPL 在 $sW$ 内不再断崖。

代价写在 §1.4 的对偶观察里：快通道的相邻位置相位差从 $\theta_0$ = 1 rad 压到 $1/s$ rad。网络训练时学到的「相位差—距离」标定被整体压缩 $s$ 倍，短距位置差落入训练分布中对应近乎零距离的邻域——局部次序分辨被摊薄。PI 论文的对策是小预算续训，把网络重新校准到压缩后的角度分布（论文口径：续训预算远小于预训练；[B]，无官方代码可对照）。要点：PI 不是免训练方案，「插值 + 短续训」才是一个整体。

### 2.2 NTK-aware：指数 ramp 曲线

NTK-aware 缩放（社区推导，与 PI 同期；无独立论文条目，按 YaRN 论文的综述口径处理，[Peng et al., 2023](../references.md)，arXiv:2309.00071，[A]）不动位置、动 base：

$$
base \to base \cdot s^{d/(d-2)} \qquad (8)
$$

代入 $\theta_j = base^{-2j/d}$ 展开（推导不跳步）：

$$
\theta'_j = \left( base \cdot s^{d/(d-2)} \right)^{-2j/d} = base^{-2j/d} \cdot s^{-\frac{2j}{d} \cdot \frac{d}{d-2}} = \theta_j \cdot s^{-2j/(d-2)} \qquad (9)
$$

即逐通道压缩率

$$
\eta_j = s^{2j/(d-2)} \qquad (10)
$$

端点检验： $j = 0$ 得 $\eta_0 = s^{0} = 1$——最快通道完全不动； $j = d/2 - 1$ 得 $\eta = s^{(d-2)/(d-2)} = s$——最慢通道吃满压缩。中间是按频率的几何 ramp。谱读法：通道 $j$ 的波长从 $\lambda_j$ 变为 $\lambda_j \cdot s^{2j/(d-2)}$，只有最慢通道的波长真正乘上 $s$。与 PI 对照：PI 是常数曲线，NTK-aware 是指数曲线，二者在最慢通道端点重合、在最快通道端点分叉——高频过度压缩正是这里省掉的。数值感： $d$ = 64、 $s$ = 40（DeepSeek-V3 的维度与倍率），中点通道 $j$ = 15 的 $\eta = 40^{30/62} \approx 6.0$。

为什么取指数形式而不是线性 ramp：频率轴本身是对数刻度的几何级数（2.5 式 (16)， $\theta_j$ 相邻通道差一个公比），在 $1$ 与 $s$ 之间分配压缩率时，「等比分配」是对数轴上的等距——每走一步通道，压缩率乘同一个因子 $s^{2/(d-2)}$，最慢通道恰好累计到 $s$。任何低频端多于高频端的线性分配都不是对数轴上的匀速，会在中段留下与频率 schedule 不对齐的折痕。

名字注记：「NTK」之名来自社区把它与神经正切核的尺度论证相联系；本书只登记机制层面的事实——它是一次保快通道的 base 重参数化，不引入任何新对象，等价于式 (10) 这条曲线。

### 2.3 YaRN：按圈数分三段 + 温度修正

YaRN（[Peng et al., 2023](../references.md)，arXiv:2309.00071，[A]，官方 repo jquesnelle/yarn）把 $\eta_j$ 曲线按「原窗内转几圈」显式分段。定义圈数比

$$
r_j = \frac{W}{\lambda_j} = \frac{W}{2\pi} \cdot base^{-2j/d} \qquad (11)
$$

**符号**： $r_j$：原窗 $W$ 内第 $j$ 通道完成的圈数； $r_j < 1$ 即 2.5 §4.5 的「窗内不满圈」通道。ramp 函数与缩放（ $\alpha < \beta$ 两个阈值）：

$$
\gamma_j = \begin{cases} 0, & r_j < \alpha \\ \dfrac{r_j - \alpha}{\beta - \alpha}, & \alpha \le r_j \le \beta \\ 1, & r_j > \beta \end{cases} \qquad (12)
$$

$$
\theta'_j = (1 - \gamma_j) \cdot \frac{\theta_j}{s} + \gamma_j \cdot \theta_j \qquad \Longleftrightarrow \qquad \eta_j = \frac{s}{1 + \gamma_j (s - 1)} \qquad (13)
$$

三段读法： $r_j > \beta$（快通道，窗内转 $\beta$ 圈以上，经验充分）—— $\gamma = 1$，不动； $r_j < \alpha$（慢通道，窗内不满 $\alpha$ 圈，正是出域元凶）—— $\gamma = 0$，吃满压缩 $s$；中间线性过渡。验证端点： $\gamma = 0$ 时 $\eta = s$， $\gamma = 1$ 时 $\eta = 1$，与 §2.2 的端点约束一致。

**DeepSeek-V3 config 走读**（W1 实测，[DeepSeek-V3 config](../references.md)，[A]）：`{type: "yarn", factor: 40.0, beta_fast: 32, beta_slow: 1, mscale: 1.0, mscale_all_dim: 1.0, original_max_position_embeddings: 4096}`。映射到式 (11)–(13)： $W$ = 4096、 $s$ = 40、 $\alpha$ = beta_slow = 1、 $\beta$ = beta_fast = 32。逐段数一遍通道（V3 旋转维 $d$ = 64，32 个通道， $\lambda_j = 2\pi \cdot 10^{4j/32}$）： $\lambda_j > W/\alpha$ = 4096 ⟺ $j \ge 23$——9 个最慢通道全压 40 倍； $\lambda_j < W/\beta$ = 128 ⟺ $j \le 10$——11 个最快通道不动； $j$ = 11 至 22 共 12 个通道走 ramp。窗口账： $4096 \times 40 = 163840$，恰为 config 的 max_position_embeddings。

**温度修正（熵不变性论证）**。YaRN 的第二件工具是 attention 温度。机制链条：位置压缩 $1/s$ 把相对距离压小 → 由位置差异贡献的 logit 变差被同比例压缩 → score 分布趋平 → 熵向 §1.3 的上界 $\ln(sW)$ 漂移 → 有效感受野被摊平。补偿手段是把 logits 乘 $\beta = 1/t$；式 (6) 已证明提高 $\beta$ 单调减熵，方向正确。取多大？§1.3 的账给出函数形式：扩窗引入的熵增量上界是 $\ln s$，故补偿应当是 $\ln s$ 的函数。YaRN 取（论文式，[A]）：

$$
\sqrt{\frac{1}{t}} = 0.1 \cdot \ln s + 1 \qquad \Longleftrightarrow \qquad \frac{1}{t} = \left( 0.1 \ln s + 1 \right)^{2} \qquad (14)
$$

函数形式（ $\ln s$ 的线性）由熵账钉死，系数 0.1 是论文的拟合值——推导到此为止，常数不进教科书。V3 实例： $s$ = 40 时 $\sqrt{1/t} \approx 1.369$，logits 放大 $(1.369)^{2} \approx 1.87$ 倍；config 的 mscale = 1.0 / mscale_all_dim = 1.0 正是式 (14) 的基础系数取值（官方推理代码按此式计算 attention 缩放，[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）——V3 的 YaRN 是带温度修正的完整形态。

### 2.4 LongRoPE：把曲线打开成逐维自由变量

LongRoPE（[Ding et al., 2024](../references.md)，arXiv:2402.13753，[B]：无官方代码，按论文拆）的观察：PI / NTK / YaRN 都是 $\eta_j$ 的低参数参数化（常数、单参数指数、三段 ramp），没有理由相信最优曲线落在这些族里。它把每通道的压缩率变成独立搜索变量：

$$
\eta^{\ast } = \operatorname{arg\thinspace min}_{\eta \in \Lambda}\ \mathrm{PPL}_{sW}\left( D_{\mathrm{val}};\ \mathrm{RoPE}(\theta_j / \eta_j) \right), \qquad \Lambda = \left\lbrace \eta : \eta_0 = 1,\ \eta_j \le \eta_{j+1},\ \eta_{d/2-1} \approx s \right\rbrace \qquad (15)
$$

**符号**： $\mathrm{PPL}_{sW}(D_{\mathrm{val}}; \cdot)$：目标窗长下验证集困惑度；约束集 $\Lambda$ 的三条—— $\eta_0$ = 1、单调不减、最慢通道吃满 $s$——直接来自 §1.4 的两端失效分析（快通道不动、压缩率随波长不降序没有意义）。搜索用进化算法在 $d/2$ 维空间进行；论文报告了 2048K 级扩展与短窗性能的恢复步骤（为短窗重新搜一组因子，[B] 论文口径）。

定位一句话：YaRN 的三段 ramp 是 $\eta_j$ 的手工 3 段参数化，LongRoPE 把它换成逐维搜索——自由度换适配性，代价是搜索管线本身，外加 [B] 级的证据边界（结论以论文实验设置为准）。

### 2.5 dual-chunk attention：不动频率，重编号位置

§2.1–2.4 全部在频率轴上做文章。dual-chunk attention（双块注意力，DCA；Qwen2.5-1M 报告采用并复述其三段结构，原始出处为 An et al. 的同名工作；[Yang et al., 2025](../references.md)，arXiv:2501.15383，[A]）走正交的一维：RoPE 频率原样，改的是喂给它的位置编号——目标是任意长序列的相对编号永不超出预训练窗 $W$。记 chunk 宽 $S$（ $S < W$）、位置 $i$ 的块号 $c_i = \operatorname{floor}(i / S)$、块内偏移 $r_i = i\ \mathrm{mod}\ S$。

key 侧编号逐块循环：

$$
P_k[j] = j\ \mathrm{mod}\ S \qquad (16)
$$

**符号**： $P_k[j]$：key $j$ 的位置编号。块身份被抹掉——编号只保块内顺序（块内越靠后编号越大），不保「第几块」；KV cache 里存的旋转后 key 因此与块号无关，可复用。

query 侧按块间距离备三张编号表：intra（同块）用 $P_q^{\mathrm{intra}}[i] = r_i$；inter（相隔两块及以上）用常数 $P_q^{\mathrm{inter}}[i] = W - 1$——钉在预训练窗末尾，保证 query 编号大于一切 key 编号（左到右的序不破）；successive（相邻块）在局部窗 $w$ 内抬升、窗外回落：

$$
P_q^{\mathrm{succ}}[i] = \begin{cases} S + r_i, & r_i < w \\ W - 1, & r_i \ge w \end{cases} \qquad (17)
$$

**符号**： $w$：局部窗宽（本节记号；与 §3.1 的 SWA 窗宽含义各自独立）。取 $w = W - S$ 时 successive 段编号在 $r_i = w$ 处与 inter 值相接： $S + (w - 1) = W - 1$，过渡连续。

三段合成相对编号（query $i$ 对 key $j$， $j \le i$）：

$$
\Delta(i, j) = \begin{cases} r_i - r_j, & c_j = c_i \quad \text{(intra-chunk)} \\ P_q^{\mathrm{succ}}[i] - r_j, & c_j = c_i - 1 \quad \text{(successive)} \\ (W - 1) - r_j, & c_j \le c_i - 2 \quad \text{(inter-chunk)} \end{cases} \qquad (18)
$$

三段读法。intra 段： $r_i - r_j = i - j$，真距离。successive 段： $r_i < w$ 时 $\Delta = S + r_i - r_j = i - j$ 也是真距离——跨块边界的近邻不被错标； $r_i \ge w$ 时回落 inter 映射。inter 段： $\Delta = (W - 1) - r_j \in [W - S,\ W - 1]$——**塌缩块身份、保留块内顺序**：「目标在第几块」不可见，「目标在块内多靠后」可见（ $r_j$ 越大距离越小，近者优先）；且取值落在预训练窗内的常数区间——编号上界是常数 $W - 1$，与序列总长 $T$ 无关。这就是 DCA 免训练外推的机制核心：位置编号永不超出训练分布， $T$ 可以任意大。

设计取向值得点破：inter 段的选择是「块内顺序优先于块间顺序」——远期上下文里，块内的局部语序（哪句跟哪句）比「这是第 37 块还是第 52 块」携带更多可用信号，而常数上界换来任意 $T$ 的合法性。代价同样写在结构里：两个不同的远块在位置侧完全不可区分，远程区分只能靠内容 score 自己拉开——回到 §1.3 的 $\Delta$ 预算。

Qwen2.5-1M 的落地组合（[A]）：rope_theta 抬到 $10^{7}$（base 路线，config 无 rope_scaling 字段，W1 实测——频率缩放侧确实不走 YaRN ramp）+ DCA 重编号 + **YaRN 温度修正**：报告原文明确推理侧始终叠加式 (14) 的 attention scaling 与 DCA 配合使用。三件工具各付各的账：base 付频率轴、DCA 付编号、YaRN 温度付熵漂移（§1.3）。

## 3. 架构手段：移动墙一与墙二

§2 的所有方案默认架构不动、只修位置编码——它们付墙四（顺带用温度补墙三），墙一墙二分文未动。本节的三类手段动架构本身。

### 3.1 SWA 堆叠与周期全局层（recap + 1M 账）

4.2 式 (3)： $L$ 层滑窗（窗宽 $w$）堆叠的感受野 $R \approx L \cdot w$。目标 1M、 $w$ = 4096 时全覆盖需 $L \approx 244$ 层——纯 SWA 在 1M 量级不现实。混合层是现实解：Gemma 3 的 5:1 local:global（4.2 §3，[Gemma Team, 2025](../references.md)，arXiv:2503.19786，[A]）在 131072 口径下 attention FLOPs 约为全 dense 的 18%。SWA 的直接贡献是删墙二：cache 封顶 $\min(T, w)$（4.2 式 (4)），「cache 随生成线性增长」被整条删除；它不删墙一的全局层部分，也不承诺精确长程检索（4.2 §8 的风险账）。1M 时代的分工定位：SWA 是局部层的默认件，长程通道由 §2 的缩放、周期全局层、动态选择（4.2 §5–§7）或固定状态层（4.3）承担。

SWA 与 §2 缩放族有一层交互值得点破：Gemma 3 给 local 层配 rope_theta $10^{4}$、global 层配 $10^{6}$（4.2 §3，[A]）。用本章的语言读：local 层见过的最大相对距离就是窗口 $w$（千级）， $2.5$ 式 (16) 的波长账只需覆盖到 $w$——base $10^{4}$ 的最长波长已数十倍于窗口，慢通道在窗内弧度足够；global 层才承担全窗 $T$ 的相对距离，长波长余量与缩放预算都该花在这里。分层配 base 的本质是把墙四拆成两层分别付：局部层永远不出域（所见距离有界），全局层用缩放族（或干脆用 §4 的扩训）付长程账。

### 3.2 Ring Attention：把序列维分片到多设备

墙一在单设备上是死墙（§1.1：FLOPs 照付、内存装不下）。Ring Attention 的回答是把序列维切成 $P$ 段、每设备持一段：设备 $p$ 常驻第 $p$ 段 Q 块（ $T/P$ 行）与第 $p$ 段 KV 块；第 $r$ 步，设备 $p$ 用 online softmax（4.1 式 (15)(16)）把本地 Q 块对当前持有的 KV 块累算进 running $(m, l, \mathrm{acc})$，同时把该 KV 块发给设备 $p+1$、接收来自 $p-1$ 的下一块； $P$ 步轮转后每个 Q 块见完全部 KV 块。

```
设备 p（常驻 Q_p 与 KV_p，P 步轮转）：

  Q_p (T/P 行) ──┬─► block-attn(Q_p, KV_p)        ─┐
                 ├─► block-attn(Q_p, KV_{p-1})      ─┤  online softmax 累算
                 ├─►  ...                            ─┤  running (m, l, acc)
                 └─► block-attn(Q_p, KV_{p-P+1})    ─┘
  KV 环流:  KV_p ──► 设备 p+1 ──► 设备 p+2 ──► … ──► 设备 p-1 ──► 回到 p
  （发送/接收与下一步 block-attn 计算重叠）
```

两个不变量。其一，数学逐点不变：online softmax 的分块归一化归纳不变量与 4.1 §5 同一个，跨设备只是分块边界的移动——Ring Attention 与 FlashAttention 是正交叠加的两层 tiling（一个在块内抗 HBM，一个在设备间抗单卡显存）。其二，FLOPs 总量不变： $O(T^{2})$ 拆成 $P$ 份并行，单设备内存 $O(T/P)$ 加一块在途 KV；墙一从「单卡内存 + 串行时长」改写为「聚合算力 + 通信带宽」。causal mask 下各设备的 Q 块可见 KV 量不同，朴素分块负载不均，工程上用均衡切分处理——通信调度、overlap 策略与拓扑映射的全部细节归 cluster 维度（边界划分见 [cluster/README.md](../../cluster/README.md)），此处只登记机制与账。

### 3.3 cache 经济学对照表

把 §1.2 的账按「每 token 每层的线性增长项」收口——这一列是长上下文架构的全部战场：

| 形态 | 每 token 每层（bf16） | $T = 10^{6}$ 增量 | 长程检索保证 | 锚点 |
| --- | --- | --- | --- | --- |
| GQA-8 | 4 KiB | 122 GiB（32 层） | 精确（全量可见） | Llama-3.1 [A] |
| MLA | 1152 B | 65.4 GiB（61 层） | 精确 | DeepSeek-V3 [A] |
| DSA | 1152 B 驻留 | 65.4 GiB（61 层），每步只读 top-2048 行 | 近似（选择器召回率） | DeepSeek-V3.2 [A] |
| KDA hybrid | 7/27 层 × 1152 B + 20/27 层 × 0 | ≈ 7.5 GiB + 20 MiB 固定状态 | KDA 层为有损压缩 | Kimi-Linear [A] |

读法：GQA → MLA 压的是「每 token 存多少」（4.1）；DSA 不动存量、动「每步读多少」（4.2）；KDA 类固定状态层动「是否随 $T$ 增长」（4.3）——三刀切在 2.4 式 (9) 的不同因子上。右侧一列是代价：存得越少或读得越挑，精确检索的保证越弱，最终都指向 §5 的评测问题：怎么量出「弱了多少」。

## 4. 训练配方：把窗口训出来

位置编码修好后（§2），权重还要在目标长度上见过数据。两份公开配方把结构讲清楚了。

**staged extension（分阶段扩窗）**。不直接上目标窗，分阶段抬：Llama-3.1 从 8K 出发分六个阶段扩到 128K，全程约 800B token 的扩窗预算（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，论文口径——该预算数字为旗舰 405B 模型的配方口径，较小模型沿用同一 schedule 结构）；DeepSeek-V3 报告两阶段扩窗、YaRN 落点 $4096 \times 40 = 163840$（[A]，论文口径；扩窗训练占 119K H800 GPU-hours，paper Table 1 的全训练口径内）。每阶段的公共结构：抬窗（配 §2 的缩放）→ 在长文档上采样后的混合数据上续训 → 双校验——长程能力（needle 类）与短上下文回归（防扩窗掉短任务点）同时达标才进下一阶段。为什么分阶段而不一步到位，可以用墙一算定量账：设阶段窗长按公比 $q$ 几何增长（ $W, qW, q^{2}W, \ldots, sW$），每阶段投入相近的 token 预算 $B_k$——注意到 token 数预算下 attention FLOPs $\propto B_k \cdot W_k$（每 token 的 attention 成本正比于窗长），几何级数窗长下总开销 $\propto \sum_k B_k q^{k}$，被末阶段主导；若把全部预算直接砸在末窗 $sW$ 上，等 token 量下开销相同，但模型从未在中等长度上被校验过——阶段化买的不是 FLOPs 节省（两者同阶），而是「逐段可校验」的过程控制：每个阶段末都能用 needle 类测试确认当前窗已吸收，失败时回退的成本只有一个阶段。同时，前几阶段用较短窗训练，墙一的 $T^{2}$ 项在浅阶段尚小，等价的 token 预算能买更多参数更新步数——浅阶段学内容与短程结构，深阶段只补长程适配，任务难度与序列长度同步爬坡。

**长文档上采样（long-document upsampling）**。机制必要性：预训练语料的长度分布重尾， $\ge$ 128K 的连续文档占比极小——直接训长窗，长程位置的梯度信号稀薄（与 2.5 §2.3 的 learned absolute「大行号行训练信号稀薄」同构）。做法：按长度桶重加权，提高长文档的采样份额（Llama-3.1 与 Qwen2.5-1M 报告均采用，[A]）。边界注记：上采样同时改变领域分布——长文档集中来自书籍 / 代码 / 长报告——配方要在「长程信号密度」与「领域分布漂移」之间权衡；各家的精确配比属未公开细节，本书只登记结构。Qwen2.5-1M 的组合是这条线的完整实例：分阶段续训 + 长文档上采样 + rope_theta $10^{7}$ + dual-chunk 与 YaRN attention scaling 的推理侧组合扩展到 1M（[Yang et al., 2025](../references.md)，arXiv:2501.15383，[A]；§2.5 末的口径）。

## 5. 评测方法论：claimed 与 effective 之间

**needle-in-a-haystack（大海捞针）**。方法：长文填充物中埋入一句事实（针），在（植入深度 × 上下文长度）网格上问答检索。它测的是最弱意义的长程能力——单针、单跳、答案与针近乎逐字对应。按 §1.3 的账看其局限：单根针只要造出 $\ln T$ 量级的 logit 优势就能被取出，而一个训练过的检索头对「反常插入句」产生的 $\Delta$ 通常远超阈值——needle 全绿只排除了最差失效模式，对多针区分、多跳链、聚合统计这些 $\Delta$ 预算更紧的任务没有区分度（RULER 论文 §1 的批评口径，[Hsieh et al., 2024](../references.md)，arXiv:2404.06654，[A]）。

**RULER**。13 个合成任务分四类（[A]，官方 repo NVIDIA/RULER），每一类对应一种不同的 $\Delta$ 预算压力（§1.3 的语言）：retrieval 类把 needle 变难——多针并存、多 key 干扰、单 key 对多 value——针与干扰项的 logit 差距被刻意压小，直接测式 (5) 的 $\Delta$ 预算余量；multi-hop tracing 类（变量追踪链： $x_1 \to x_2 \to \cdots \to x_k$ 的赋值链回溯）测的是「多次连续取用」——每一跳都要付一次 $\Delta$，任何一跳稀释则链断；aggregation 类（统计高频词 / 共现词）把单点检索换成对全窗的分布式计数，attention mass 必须同时覆盖大量位置，逼近均匀极限 $\ln T$ 的另一端；QA 类把真问题嵌入长干扰文本，测语义检索而非模板匹配。方法论贡献是把「上下文长度」从 config 字段变成行为测量：沿长度扫性能曲线，把跌破通过线（相对短窗基线的阈值，论文口径）的位置定义为该模型的 effective context（有效上下文）。论文的总体发现：多数受测模型的 effective context 显著短于其 claimed context（定性结论，[A]；逐模型数字以论文表格为准，本书不复录）。

**claimed vs effective 对照**（claimed 侧为 config / 报告实测口径）：

| 模型 | claimed（max_position_embeddings / 报告） | 手段 | 证据 |
| --- | --- | --- | --- |
| Llama-3.1-8B | 131072 | 六阶段扩训，rope_theta $5 \times 10^{5}$ | config [A] + herd paper [A] |
| DeepSeek-V3 | 163840 | YaRN ×40 on 4096，两阶段扩训 | config [A] + 报告 [A] |
| Qwen2.5-1M | 1010000 | rope_theta $10^{7}$ + DCA + YaRN 温度（频率侧无 rope_scaling） | config [A] + 报告 [A] |
| Kimi-Linear-48B | 1048576 | KDA hybrid（3:1 交错） | config [A] + 论文 [A] |
| Qwen3-Next-80B | 262144（原生；官方称可扩到 1010000） | GDN hybrid + rope_theta $10^{7}$ | config / model card [W] |
| MiniMax-Text-01 | 10240000 | lightning hybrid（论文摘要口径：训练 1M、推理外推 4M） | config [A] + 论文 [A] |

effective 侧只有行为测量能填。读法收束全章：claimed 回答的是「位置编号与 cache 是否允许这个长度」（墙二、墙四的工程面），effective 回答的是「 $\Delta$ 预算与训练信号是否跟到了这个长度」（墙三与 §4 的配方面）——§1 的四堵墙全部付清，两个数字才会重合。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 墙一：FLOPs | $2T^{2}d$ / 层；交叉点 $`T^{*} = N/(dL) \approx 6 \times 10^{4}`$（8B 形态），式 (1)(2) | 1M 单序列 attention ≈ $2.6 \times 10^{17}$ FLOPs；FlashAttention 不省 FLOPs |
| 墙二：KV cache | $2 L H_{kv} d_{\mathrm{head}} T B b$（2.4 式 (9)） | 1M：GQA-8 122 GiB / MLA-61L ≈ 65.4 GiB / KDA hybrid ≈ 7.5 GiB + 20 MiB 固定状态 |
| 墙三：熵稀释 | $H \le \ln T$ 式 (3)；$`p^{*} \ge 1/2 \Longleftrightarrow \Delta \ge \ln T + \sigma^{2}/2`$ 式 (5) | 所需 logit 优势随 $\ln T$ 增长；attention sink 同源 |
| 墙四：角度出域 | 慢通道 $\lambda_j > W$ 只见 $[0, W\theta_j]$（2.5 §6） | 分布外失效，非数值失效；修复损伤集中在快通道 |
| 统一形式 | $\theta_j \to \theta_j / \eta_j$， $\eta_j \in [1, s]$ 一条曲线 | PI / NTK / YaRN / LongRoPE 是同一曲线的四种参数化 |
| PI | $\eta_j = s$，式 (7) | 常数曲线；快通道过度压缩；需小预算续训校准 [B] |
| NTK-aware | $base \to base \cdot s^{d/(d-2)}$， $\eta_j = s^{2j/(d-2)}$，式 (9)(10) | 端点 $\eta_0$ = 1、 $\eta_{d/2-1}$ = $s$；纯频率重参数化 |
| YaRN | $r_j = W / \lambda_j$ 三段 ramp 式 (12)(13)；温度 $\sqrt{1/t} = 0.1 \ln s + 1$ 式 (14) | DeepSeek-V3： $W$ = 4096、 $s$ = 40、 $\alpha$ = 1、 $\beta$ = 32 → 9 通道全压、11 通道不动、12 通道 ramp； $1/t \approx 1.87$ |
| LongRoPE | $\operatorname{arg\thinspace min}_{\eta} \mathrm{PPL}_{sW}$，约束 $\eta_0$ = 1 且单调不减，式 (15) | 逐维搜索；[B] 证据边界 |
| dual-chunk | key 编号 $j\ \mathrm{mod}\ S$ + query 三张表，式 (16)–(18) | 塌缩块身份、保块内顺序；编号上界为常数 $W - 1$（与 $T$ 无关）；Qwen2.5-1M：base $10^{7}$ + DCA + YaRN 温度 |
| Ring Attention | 序列维 $P$ 段环流 + online softmax | 数学逐点不变；通信实现归 cluster 维度 |
| 训练配方 | staged extension + 长文档上采样 | Llama-3.1：六阶段 8K→128K、约 800B token（论文口径）；V3：两阶段，扩窗 119K H800 GPU-hours |
| 评测 | needle 只测单针单跳；RULER 13 任务 4 类定义 effective context | claimed（config）≠ effective（行为测量）；四堵墙付清才重合 |

## 练习

1.（推导）NTK-aware 的谱性质。(a) 从式 (8) 出发补全式 (9) 到式 (10) 的推导；(b) 证明 PI 下「窗内不满圈」的通道集合不随 $s$ 变化（即 $\lbrace j : s\lambda_j > sW\rbrace = \lbrace j : \lambda_j > W\rbrace$）；(c) 对 NTK-aware 给出不满圈通道集合随 $s$ 变化的条件式，并说明它在快通道端与慢通道端各自相对 PI 是收紧还是放松。
2.（计算）复算 §1 的三笔账。(a) 式 (1) 的 1M 单层 attention FLOPs 与式 (2) 的交叉点 $`T^{*}`$（Llama-3.1-8B 形态， $N \approx 8 \times 10^{9}$）；(b) §1.2 表中 GQA-8 与 MLA 两行在 $T = 10^{6}$ 的 GiB 数；(c) 由 Kimi-Linear config 的 27 层结构（7 MLA + 20 KDA、KDA 状态 32 heads × $128 \times 128$）推出 1M 总 cache，并算出相对 27 层全 MLA 反事实的削减比例。
3.（推导）熵稀释与温度。(a) 用 Jensen 不等式证明式 (3) 并给出等号条件；(b) 在 $\Delta$ 固定、 $T \to sT$ 下用式 (4) 写出 $`p^{*}`$ 的渐近衰减率；(c) 由式 (6) 说明「补偿熵增 $\ln s$ 量级 ⟹ $\beta$ 取 $\ln s$ 的函数」这一函数形式论证，并讨论为什么系数 0.1 不能由该论证推出。
4.（代码）教学实现三条 $\eta_j$ 曲线（教学用，非优化实现）：给定 $base$、 $d$、 $W$、 $s$、 $(\alpha, \beta)$，分别按式 (7)(10)(13) 计算逐通道压缩率，(a) 数值断言三者在 $j = d/2 - 1$ 端点都收敛到 $s$、NTK 与 YaRN 在 $j = 0$ 端为 1；(b) 以 DeepSeek-V3 字段（ $W$ = 4096、 $s$ = 40、 $\alpha$ = 1、 $\beta$ = 32、 $base$ = $10^{4}$、 $d$ = 64）验证「11 通道不动、12 通道 ramp、9 通道全压」的计数；(c) 把三条曲线套进 2.5 §4.8 的 `rope_tables`（把 `inv_freq` 逐项除以 $\eta_j$），断言改造后 $\theta'_j \in [\theta_j / s,\ \theta_j]$ 逐通道成立。

## 参考

- [Chen et al., 2023](../references.md)：Extending Context Window of Large Language Models via Positional Interpolation，arXiv:2306.15595，[B]（key: pi）——PI 的 $m \to m/s$ 与续训校准
- [Peng et al., 2023](../references.md)：YaRN: Efficient Context Window Extension of Large Language Models，arXiv:2309.00071，[A]（key: yarn）——三段 ramp、温度修正式 (14)、NTK-aware 的综述口径；官方 repo jquesnelle/yarn
- [Ding et al., 2024](../references.md)：LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens，arXiv:2402.13753，[B]（key: longrope）——逐维搜索目标式 (15)
- [Yang et al., 2025](../references.md)：Qwen2.5-1M Technical Report，arXiv:2501.15383，[A]（key: qwen2.5-1m）——dual-chunk attention 式 (16)–(18)（报告复述 An et al. 同名工作的三段结构）、1M 训练配方、YaRN attention scaling 与 DCA 的配合口径
- [Hsieh et al., 2024](../references.md)：RULER: What's the Real Context Size of Your Long-Context Language Models?，arXiv:2404.06654，[A]（key: ruler）——13 任务 4 类、effective context 定义、needle 局限；官方 repo NVIDIA/RULER
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——128K 六阶段扩窗配方
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——两阶段扩窗与 YaRN 温度实现
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——rope_scaling 块：factor 40、beta_fast 32、beta_slow 1、original_max_position_embeddings 4096
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——rope_theta $5 \times 10^{5}$、max_position_embeddings 131072
- [Xiao et al., 2023](../references.md)：Efficient Streaming Language Models with Attention Sinks，arXiv:2309.17453，[A]（key: streamingllm）——attention sink 与熵稀释的同源机制
- [Kimi Team, 2025](../references.md)：Kimi Linear: An Expressive, Efficient Attention Architecture，arXiv:2510.26692，[A]（key: kimi-linear）——KDA hybrid 的 1M 上下文锚点
- [Moonshot AI, Kimi-Linear-48B-A3B-Instruct config.json](../references.md)：[A]（key: kimi-linear-config）——20 KDA + 7 full-attn 层结构
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models，arXiv:2512.02556，[A]（key: deepseek-v3.2）——DSA 的 top-2048 读取（4.2 §6 交叉引用）
- [Gemma Team, 2025](../references.md)：Gemma 3 Technical Report，arXiv:2503.19786，[A]（key: gemma3）——SWA+full 5:1 混合层
- [Su et al., 2021](../references.md)：RoFormer: Enhanced Transformer with Rotary Position Embedding，arXiv:2104.09864，[A]（key: rope）——RoPE 频率 schedule 与波长账的源头
- [MiniMax, 2025](../references.md)：MiniMax-01: Scaling Foundation Models with Lightning Attention，arXiv:2501.08313，[A]（key: minimax01）——训练 1M / 外推 4M 口径
- [Qwen, Qwen3-Next-80B-A3B-Instruct model card 与 config](../references.md)：[W]（key: qwen3-next）——原生 262144、官方称可扩 1010000
