# 1.3 Scaling Laws 作为理论（Scaling Laws as a Theory）

> 本章回答一个分配问题：给定算力预算 $C$ ，1.2 的账式 $C = 6ND$ 只圈定了可行集合，把预算分给参数 $N$ 还是数据 $D$ 要靠 loss 的幂律形式；核心结果是 compute-optimal 分配律 $N_{\mathrm{opt}} \propto C^{a}$ 的完整推导，以及能力涌现（emergence）的测量学分析 | 前置：1.1（CE、PPL），1.2（$C = 6ND$ 与 $N$ 的口径、推理 $2N$ /token） | 锚点：Pythia（16 个模型 × 每模型 154 个 checkpoints）[A]、OLMo 2（权重 / 数据 / 日志全开放）[A]；Kaplan [B]、Chinchilla [B]、Wei [B]、Schaeffer [B]、DeepSeekMoE [A]

## 1. 幂律形式与 log-log 线性

Kaplan 等的测量结果是三条独立的幂律：在其余资源不构成瓶颈时，test loss 对参数量、数据量、算力分别是幂函数（[Kaplan et al., 2020](../references.md)，arXiv:2001.08361，[B]）：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C} \qquad (1)
$$

**符号**：$L$ ：held-out test 集上的 per-token CE（nats，1.1 式 (4) 的口径，全程固定 tokenizer 与测试集）；$N$ ：非 embedding 参数量（与 1.2 式 (2) 的 FLOPs 口径一致）；$D$ ：训练 token 总数；$C$ ：训练算力，原论文以 PF-days（ $10^{15}$ FLOPS $\times$ 86400 s $= 8.64 \times 10^{19}$ FLOPs）为单位；$N_c, D_c, C_c$ ：量纲常数；$\alpha_N, \alpha_D, \alpha_C$ ：幂律指数。口径注：$L(C)$ 的 $C$ 是 $C_{\min}$ （compute-optimal 口径，即每个模型按最优分配训练所耗的最小算力），$\alpha_C = 0.050$ 属于该口径；raw-C 律（不约束分配、按实际消耗算力计）的指数是 0.057，两个数不要混用。

论文拟合值： $\alpha_N \approx 0.076$ 、 $N_c \approx 8.8 \times 10^{13}$ ； $\alpha_D \approx 0.095$ 、 $D_c \approx 5.4 \times 10^{13}$ ； $\alpha_C \approx 0.050$ 、 $C_c \approx 3.1 \times 10^{8}$ PF-days（ $\approx 2.7 \times 10^{28}$ FLOPs）。测量条件是三条律的适用前提，必须写清： $L(N)$ 曲线在数据充足下训练到收敛测得， $L(D)$ 曲线在模型足够大下取 early stopping 最优点测得——各自假设另两个变量不构成瓶颈。

幂律的工程含义取对数后可见： $\log L = \alpha (\log x_c - \log x)$ ，log-log 坐标下是斜率 $-\alpha$ 的直线。每 10 倍投入换来固定比例的 loss 下降：代入 $\alpha_N = 0.076$ ，参数 10 倍、loss 乘 $10^{-0.076} \approx 0.84$ ；代入 $\alpha_C = 0.050$ ，算力 10 倍、loss 只乘 $10^{-0.050} \approx 0.89$ 。直线上没有拐点、没有饱和段——这是「规模即收益」的定量内容；反过来它也说明收益递减是幂律式的：10 倍算力换不到 11% 的 loss 下降。

三条单变量律回答不了分配问题：固定 $C$ 时 $N$ 与 $D$ 此消彼长， $L(N)$ 与 $L(D)$ 各自假设对方不构成瓶颈，拼不出约束优化。需要的是 $L(N, D)$ 的联合形式。

## 2. Chinchilla 联合形式：不可约项、容量项、数据项

Chinchilla（[Hoffmann et al., 2022](../references.md)，arXiv:2203.15556，[B]）把 loss 直接参数化为 $(N, D)$ 的二元函数：

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \qquad (2)
$$

**符号**：$E$ ：不可约损失，即 $N, D \to \infty$ 的渐近地板——自然语言本身的熵（1.1 式 (6) 中 $H(p)$ 的角色）；$A / N^{\alpha}$ ：容量项，数据无限时模型规模的残差；$B / D^{\beta}$ ：数据项，模型无限时有限数据的残差；$\alpha, \beta$ ：两个幂律指数，决定第 3 节的分配律。

拟合值（论文 Approach 3，对 $\log L$ 用 Huber loss（ $\delta = 10^{-3}$ ）做参数曲面拟合，论文 Eq. (10)）： $E = 1.69$ 、 $A = 406.4$ 、 $\alpha = 0.34$ 、 $B = 410.7$ 、 $\beta = 0.28$ 。训练点覆盖 70M 到 16B 参数、约 400 组 $(N, D)$ 组合。论文用三种互相独立的方法估计最优分配，第 3 节末尾会看到它们的差异本身就是结论的一部分：

- **Approach 1（训练曲线 frontier）**：固定一组模型规模，每个规模扫多个 $D$ ，取平滑后的 loss 下包络，再对每个 $C$ 找 frontier 上的最小点。
- **Approach 2（IsoFLOPs 剖面）**：固定 $C$ 扫 $N$ （ $D = C / 6N$ 随之确定），loss 对 $\log N$ 呈开口向上的曲线（上方 ASCII 图下半张即一个剖面），谷底即该 $C$ 的最优；对每个预算重复，得到 $N_{\mathrm{opt}}(C)$ 的点列再拟合幂律。
- **Approach 3（参数曲面拟合）**：即式 (2)，全部训练点一次拟合 $(E, A, B, \alpha, \beta)$ 。在 $\log L$ 上拟合（而非 $L$ ）并让 Huber 损失的小 $\delta$ 抑制离群点，是为了让三个数量级上的点按相对误差等权，不被大 loss 的早期训练点主导。

三法对应的统计假设不同：Approach 1/2 只信每条曲线的最优点，对单点噪声鲁棒但丢弃了曲线形状信息；Approach 3 用全部点、信息利用率高，但把式 (2) 的函数形式当作先验押上。第 3 节末会看到三法给出的分配指数弥散在 $\pm 0.04$ 内——这个宽度就是「compute-optimal」概念在当前测量精度下的分辨率。

```
log L
 │╲
 │ ╲___ E + A/N^α        ← D 充分大时只剩容量项
 │  ╲
 │   ╲___ E + B/D^β      ← N 充分大时只剩数据项
 │    ╲___
 │        ╲______ E      ← 不可约地板（语言本身的熵）
 └──────────────────────── log x        （x = N 或 D）

log L
 │ ╲             ╱
 │  ╲           ╱      ← 固定 C：N 太小则数据项 B/D^β 涨，
 │   ╲___U___╱           D 太小则容量项 A/N^α 涨
 │       ↑ 谷底 = compute-optimal（式 (5)）
 └──────────────────────── log N（此剖面上 log D = log(C/6) − log N）
```

## 3. compute-optimal 分配：Lagrangian 完整推导

问题：给定预算 $C$ ，在 1.2 式 (5) 的约束 $6ND = C$ 下最小化式 (2)：

$$
\min_{N, D > 0} \; L(N, D) = E + A N^{-\alpha} + B D^{-\beta} \qquad \text{s.t.} \quad 6ND = C \qquad (3)
$$

**符号**：$N, D$ ：决策变量；$C$ ：给定常数；其余同式 (2)。

构造 Lagrangian， $\lambda$ 为乘子：

$$
\Lambda(N, D, \lambda) = E + A N^{-\alpha} + B D^{-\beta} + \lambda (6ND - C)
$$

**符号**：$\lambda$ ：约束 $6ND = C$ 的 Lagrange 乘子，其绝对值是预算的边际价值（见本节末）。

一阶条件，对三个变量分别求偏导并置零：

$$
\frac{\partial \Lambda}{\partial N} = -\alpha A N^{-\alpha-1} + 6\lambda D = 0 \;\Rightarrow\; \alpha A N^{-\alpha-1} = 6\lambda D \qquad (4a)
$$

$$
\frac{\partial \Lambda}{\partial D} = -\beta B D^{-\beta-1} + 6\lambda N = 0 \;\Rightarrow\; \beta B D^{-\beta-1} = 6\lambda N \qquad (4b)
$$

$$
\frac{\partial \Lambda}{\partial \lambda} = 6ND - C = 0 \qquad (4c)
$$

式 (4a) 除以式 (4b)（两侧均为正，除法合法），乘子 $6\lambda$ 消去：

$$
\frac{\alpha A \, N^{-\alpha-1}}{\beta B \, D^{-\beta-1}} = \frac{D}{N} \;\Rightarrow\; \frac{\alpha A}{\beta B} \, N^{-\alpha} D^{\beta} = 1 \;\Rightarrow\; D^{\beta} = \frac{\beta B}{\alpha A} N^{\alpha} \;\Rightarrow\; D = \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{\alpha/\beta}
$$

**推导注**：第一步把分式整理为 $N$ 、 $D$ 的同次幂之比；第二步两边乘 $N^{\alpha} D^{\beta}$ ；第三步解出 $D$ 。

把上式代入约束 (4c)：

$$
6N \cdot \left(\frac{\beta B}{\alpha A}\right)^{1/\beta} N^{\alpha/\beta} = C \;\Rightarrow\; N^{(\alpha+\beta)/\beta} = \frac{C}{6} \left(\frac{\alpha A}{\beta B}\right)^{1/\beta}
$$

两边取 $\beta/(\alpha+\beta)$ 次幂，得 $N_{\mathrm{opt}}$ ；对称地（消去 $N$ 解 $D$ ）得 $D_{\mathrm{opt}}$ ：

$$
N_{\mathrm{opt}} = G \left(\frac{C}{6}\right)^{a}, \qquad D_{\mathrm{opt}} = G^{-1} \left(\frac{C}{6}\right)^{b}, \qquad a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta}, \quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} \qquad (5)
$$

**符号**：$a, b$ ：compute-optimal 分配指数；$G$ ：量纲常数。 $a + b = 1$ 不是巧合：它保证 $6 N_{\mathrm{opt}} D_{\mathrm{opt}} = 6 G G^{-1} (C/6)^{a+b} = C$ ，约束自动满足。

这是最小值而非极大值或鞍点，一步可验。用约束消去 $D$ ，记 $g(N) = L(N, C/6N) = E + A N^{-\alpha} + B (6N/C)^{\beta}$ ，则

$$
g'(N) = -\alpha A N^{-\alpha-1} + \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta-1} = N^{-\alpha-1} \left[ -\alpha A + \beta B \left(\frac{6}{C}\right)^{\beta} N^{\alpha+\beta} \right]
$$

**推导注**：第二步提取公因子 $N^{-\alpha-1}$ ，括号内两项的幂次合并为 $N^{(\beta-1)-(-\alpha-1)} = N^{\alpha+\beta}$ 。

方括号内是 $N$ 的严格增函数（ $\alpha + \beta > 0$ ），在 $N \to 0^{+}$ 时为 $-\alpha A < 0$ 、 $N \to \infty$ 时趋于 $+\infty$ ，恰有一个零点，即式 (5) 的驻点；$g'$ 由负变正，驻点是唯一全局最小。注意 $g''$ 并不恒正—— $\beta < 1$ 时数据项 $B(6N/C)^{\beta}$ 的二阶导为负， $g$ 不是全局凸函数，所以这里不能用「凸函数驻点即最小」的捷径，「提取公因子看 $g'$ 符号」的论证不受此影响，比 Hessian 判据干净（练习 2 展开）。

代入数值： $\alpha = 0.34$ 、 $\beta = 0.28$ 得

$$
a = \frac{0.28}{0.62} \approx 0.45, \qquad b = \frac{0.34}{0.62} \approx 0.55, \qquad G = \left(\frac{0.34 \times 406.4}{0.28 \times 410.7}\right)^{1/0.62} \approx 1.34 \qquad (6)
$$

**数值注**：$a$ 由舍入到两位小数的指数算得 0.4516；论文 Table 2 用未舍入的拟合参数报告 0.46，下同。

论文 Table 2 汇总三种方法： $a$ 在 $0.46 \sim 0.50$ 、 $b$ 在 $0.50 \sim 0.54$ 之间（Approach 1/2 接近对称的 $0.5$ ，Approach 3 参数拟合给出 $a$ 略小）——方法间的弥散本身就是测量精度的一部分。结论表述：预算 $C$ 增长时，compute-optimal 的模型与数据**近似各按** $\sqrt{C}$ **增长**，而不是 Kaplan 律的「大头给模型」（第 4 节）。

两个直接推论。第一，最优前沿上的 loss：把式 (5) 代回式 (2)，注意 $a\alpha = b\beta = \alpha\beta/(\alpha+\beta)$ ，两项同阶：

$$
L^{*}(C) - E = \left( A G^{-\alpha} + B G^{\beta} \right) \left(\frac{C}{6}\right)^{-\frac{\alpha\beta}{\alpha+\beta}} \propto C^{-0.154} \qquad (7)
$$

**符号**：$L^{*}(C)$ ：预算 $C$ 下最优分配的 loss；指数 $\alpha\beta/(\alpha+\beta) = 0.0952/0.62 \approx 0.154$ 。读法：10 倍算力换 $L - E$ 乘 $10^{-0.154} \approx 0.70$ 。乘子 $\lambda$ 的经济含义由包络定理给出： $\mathrm{d}L^{*}/\mathrm{d}C = -\lambda$ ，即预算的边际 loss 收益。

第二， $D_{\mathrm{opt}} / N_{\mathrm{opt}} = (C/6)^{b-a} / G^{2}$ 随 $C$ 缓变，**不是常数**——流传的「每参数 20 token」只是论文实际训练点（70B 参数 × 1.4T token）的比率，不是不变量。

## 4. Kaplan 与 Chinchilla 的分歧：0.73 对 0.5

同一类问题，Kaplan 论文的结论是 $N_{\mathrm{opt}} \propto C^{0.73}$ 、 $D_{\mathrm{opt}} \propto C^{0.27}$ ——预算增长时大头给模型（[Kaplan et al., 2020](../references.md)，[B]）。分歧量级不小：预算翻 100 倍，按 Kaplan 应把模型放大 $100^{0.73} \approx 29$ 倍，按 Chinchilla 只放大 10 倍。

分歧的根源在拟合方法，不在数据。两条机制性原因（[Hoffmann et al., 2022](../references.md)，[B]，论文对照讨论节）：

1. **学习率 schedule 未按训练长度退火**。Kaplan 的各规模模型用固定形状的 schedule，cosine 衰减不随实际训练的 $D$ 调整到零；训练在 schedule 中途停止的点上，loss 系统性高于充分退火后的值。Chinchilla 的 Approach 1/2 对每条曲线、每个 $(N, D)$ 点都把 schedule 调到匹配长度。小模型、多数据的点被旧 protocol 高估得最多，拟合出的前沿就被推向「大模型」一侧。
2. **单变量律拼不出联合优化**。Kaplan 的 $L(N)$ 在数据充足下测得、 $L(D)$ 在模型充足下测得，两者的测量条件互斥；从它们组装固定 $C$ 的最优点，与在 $(N, D)$ 网格上直接测联合曲面不是一回事。事实上把 Kaplan 自己的联合形式代入第 3 节的推导，得到的指数是 $\alpha_D / (\alpha_N + \alpha_D) \approx 0.56$ 而非 0.73（练习 1）——论文内部的单变量系数与其分配结论之间存在张力。

后果用一对真实配置看：Gopher 的 280B 参数 × 300B token（ $D/N \approx 1.1$ ）接近 Kaplan 律的选型；同一预算按 Chinchilla 三法预测的最优模型只有其约四分之一到七分之一大——Approach 1 在论文 Table 3（1 个 Gopher 预算行）给出 67B（280B 的约 1/4），Approach 3 参数拟合在 §3.3 正文给出 40B（约 1/7），区间即 40B–67B；实际训练取 70B × 1.4T token（ $D/N = 20$ ）并报告了更低的 loss（[Hoffmann et al., 2022](../references.md)，[B]）。「同等算力、更小模型、更多数据」从拟合方法修正变成了可执行的工程结论。

## 5. inference-aware 的修正：训练最优 ≠ 部署最优

式 (3) 的目标函数只有训练一侧。模型部署后每服务一个 token 还要付 $2N$ FLOPs（1.2 式 (2)，decode 口径），其中 $N$ 按 1.2 第 9 节的全量口径 $N_{\mathrm{full}}$ 取——含输出头、输入 embedding 查表不计；随上下文长度线性增长的 attention 项在长上下文下逐渐追平权重项（交叉点 $T^{*}$ 见 1.2 第 6 节），本章服务账按常规上下文取 $2N$ 主导口径。设模型终身推理总量为 $Q$ token，全生命周期的账是

$$
\mathrm{TC}(N, D) = \underbrace{6ND}_{\text{train}} + \underbrace{2NQ}_{\text{inference}} \qquad (8)
$$

**符号**：$\mathrm{TC}$ ：训练加推理的总 FLOPs；$Q$ ：服务期处理的总 token 数（prompt + 生成）。

目标函数一换，最优解就换： $Q$ 小（研究模型，跑完评测即弃）时式 (8) 退回式 (3)；$Q$ 大时 $N$ 在目标里出现两次，最优解向「小模型、多数据」偏移——哪怕训练更贵，推理侧每次省 $2\Delta N$ 总会赚回来。这是**过度训练（over-training）**：有意把模型训到远超 compute-optimal 的 $D$ 。边际视角看同一件事：在式 (8) 下减小 $N$ 的边际收益是每 token 省 $2Q$ FLOPs 推理费，边际代价是为保住 loss 必须追加数据（ $D$ 沿式 (2) 的等 loss 线上移），训练费增加；$Q$ 越大，省的一侧越重，最优点越偏向小模型——$Q \to \infty$ 的极限是「把数据加到 loss 地板 $E + A/N^{\alpha}$ 可接受为止」。

用 Llama-3.1-8B 做实账：它预训练 15T token（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]），按 1.2 第 9 节的口径（训练每 token $\approx 45$ G FLOPs，含输出头）总预算 $C \approx 6.8 \times 10^{23}$ FLOPs。三笔账并排：

| 方案 | $N$ | $D$ | $L$（式 (2) 估计） | 推理每 token |
| --- | --- | --- | --- | --- |
| Meta 实际选择 | $8.03 \times 10^{9}$ | 15T | 1.948 | $1.5 \times 10^{10}$ |
| 同 $C$ 的 compute-optimal（式 (5)(6) 解出） | $3.5 \times 10^{10}$ | 3.3T | 1.925 | $6.9 \times 10^{10}$ |
| 8B 达到同一 $L = 1.925$ 所需的 $D$（解式 (2)） | $8.03 \times 10^{9}$ | $\approx$ 48T | 1.925 | $1.5 \times 10^{10}$ |

**符号**：第二行由式 (5) 代入 $C = 6.8 \times 10^{23}$ 解得；第三行固定 $N$ 、对 $D$ 解式 (2)。推理列口径：真实模型（一、三行）按 1.2 第 9 节全量口径 $2N_{\mathrm{full}} = 1.5 \times 10^{10}$ （ $N_{\mathrm{full}} = 7.50 \times 10^{9}$ ，含输出头、输入 embedding 查表不计）；第二行的 $N$ 是式 (2) 口径的非 embedding 参数， $2N$ 即前向账，若再补输出头修正不足 2%、不改变结论。

读法：compute-optimal 方案用相同的训练预算买到低 $0.023$ nats 的 loss，代价是推理每 token 贵 $4.6$ 倍。若产品目标定在 $L = 1.925$ ，8B 方案要把训练成本抬高到约 $2.2 \times 10^{24}$ FLOPs（3.2 倍），打平条件为

$$
2.2 \times 10^{24} + 1.5 \times 10^{10} \, Q = 6.8 \times 10^{23} + 6.9 \times 10^{10} \, Q \;\Rightarrow\; Q^{*} \approx 3 \times 10^{13} \text{ tokens}
$$

**符号**：$Q^{*}$ ：两种方案总账相等的推理量。

$Q^{*}$ 只有训练数据量的 2 倍——对开放权重、下载量以亿次计的模型，实际 $Q$ 远超此数，过度训练在全生命周期口径下反而是省钱的选择。compute-optimal 律的正确读法由此明确：它是**训练预算的**最优分配，不是部署成本的最优；推理量 $Q$ 进入目标函数后，最优 $N$ 单调向小偏移（练习 3）。第 8 节的局限清单会说明这个框架还差哪些项。

## 6. MoE 的 scaling：$N$ 的角色由激活参数继承

MoE 模型每 token 只激活部分参数，1.2 式 (9) 已建立对应口径： $C \approx 6 N_{\mathrm{act}} D$ 。代入本章框架，scaling law 中 $N$ 的两个角色被拆开——进 FLOPs 账的是激活参数 $N_{\mathrm{act}}$ ，进容量（知识存储）的是总参数 $N_{\mathrm{tot}}$ 。经验结果：同等 FLOPs 预算下 MoE 的 loss 曲线整体低于 dense，且细粒度切分再压一截（[Dai et al., 2024](../references.md)，arXiv:2401.06066，[A]，DeepSeekMoE 的 IsoFLOPs 对照曲线）；论文摘要的锚点数字：DeepSeekMoE 16B（2.8B 激活）以约 40% 的计算量达到 LLaMA2-7B 的下游性能。

机制上与式 (2) 对应：固定 $C$ 时 MoE 等价于一个 $N_{\mathrm{act}}$ 量级的 dense 模型在付 FLOPs，但容量项由 $N_{\mathrm{tot}} \gg N_{\mathrm{act}}$ 摊薄——式 (2) 的 $A / N^{\alpha}$ 项在 MoE 上系统性更小。代价不进 FLOPs 账而进别的账：总参数的内存（1.2 式 (7)）、路由与负载均衡、expert 并行通信（4.5 展开路由机制，集群侧属 cluster 维度）。fine-grained expert（把 FFN 中间维切成 $m$ 倍细、expert 数与每 token 激活数同乘 $m$ ，保持 $N_{\mathrm{act}}$ 不变；再隔离若干共享 expert 常驻激活，DeepSeekMoE 的两项核心设计）提高的是专家组合的自由度：$m$ 倍细后可选的 expert 子集数从 $\binom{n}{k}$ 涨到 $\binom{mn}{mk}$ ，同一 token 可以被更细粒度的知识单元组合表达，scaling 曲线上体现为同一 $N_{\mathrm{act}}$ 下更低的 loss——4.5 给出路由公式后再回到这条曲线。

对分配问题的推论：MoE 的 compute-optimal 推导形式上仍是式 (3)–(5)，只是 $N$ 换成 $N_{\mathrm{act}}$ 、系数 $(A, \alpha)$ 换成 MoE 家族的拟合值；总参数 $N_{\mathrm{tot}}$ 成为近似免费的容量维度，其上限由内存与通信账决定，不由式 (5) 决定。

## 7. Emergence：定义、分类与测量学分析

loss 是平滑的幂律，下游任务分数却不一定。Wei 等给出**涌现能力（emergent ability）**的操作性定义：某能力在小模型上不存在（分数与随机基线不可区分）、在大模型上出现，scaling 曲线上呈相变状跳变（[Wei et al., 2022](../references.md)，arXiv:2206.07682，[B]）。论文把报告案例分两类：

- **few-shot 任务型**：多位算术、音译、词序还原等任务上，分数在某一规模前贴随机基线，之后急升；
- **prompting 策略型**：chain-of-thought、instruction following 等策略只在足够大的模型上带来正收益，小模型上无效甚至有害。

Schaeffer 等的反方分析（[Schaeffer et al., 2023](../references.md)，arXiv:2304.15004，[B]）：已报告的跳变绝大多数是 metric 选择的产物，不是模型行为的相变。统计事实：BIG-Bench 上报告的涌现案例中超过 92% 落在两类 metric 之下——Exact String Match 与 Multiple-Choice Grade。公式级分析分两步，说明「平滑能力 + 非线性 metric」如何制造跳变。

**机制一：复合非线性。** 设模型 per-token 正确率 $p(N) = 1 - c N^{-k}$ 随规模平滑上升（幂律误差，$c, k > 0$ 为常数）。一道需要 $L$ 个 token 全部正确的题目，Exact Match 分数是

$$
\mathrm{EM}(N) = p(N)^{L} = \left(1 - c N^{-k}\right)^{L} \approx \exp\left(-L \, c \, N^{-k}\right) \qquad (9)
$$

**符号**：$L$ ：题目答案的 token 数；$k$ ：per-token 能力的幂律改善率；近似号用了 $1 - x \approx e^{-x}$ 。

式 (9) 的过渡中心在 $L c N^{-k} \approx 1$ 处，即 $N^{*} \approx (cL)^{1/k}$ ——**「涌现点」位置是题目长度 $L$ 的函数**，同一模型在短任务上「已有能力」、在长任务上「尚未涌现」。过渡区（EM 从 0.1 到 0.9）在 $\log N$ 轴上宽约 $(1/k) \log_{10}(\ln 10 / 0.105) \approx 1.34 / k$ 个数量级，与 $L$ 无关；$L$ 增大只把整条曲线右移并压扁可见区间。数值感： $p$ 从 0.80 平滑升到 0.95（$+0.15$ ），$L = 8$ 时 EM 从 $0.168$ 跳到 $0.663$（ $+0.50$ ）——metric 把输入变化放大了 3 倍多。

**机制二：地板与阈值。** 评测集只有 $n$ 道题时，分数低于约 $1/n$ 的模型与「全错」统计上不可区分，绘图时整段贴在基线上—— $p$ 在地板区内的平滑增长全部不可见，跳变的「平地」部分是测量分辨率制造的。Multiple-Choice Grade 则是显式阈值化：当且仅当正确选项的 logit 为候选中最高时记 1、否则记 0——对连续的 logit 差取指示函数。设 logit 差的均值随 $N$ 平滑增长、噪声水平为 $\sigma$ ，任务分数是 $\Pr\{z_{\mathrm{correct}} > \max_{i \in \mathrm{wrong}} z_i\}$ 这类 probit 形曲线：均值穿过零点附近约 $2\sigma$ 宽的一段时，分数从接近 0 走完到接近 1 的全程；评测若只报 0/1 聚合，这段过渡在多宽的 $\log N$ 区间上完成，取决于 logit 差的增长率与 $\sigma$ ，与模型内部是否发生质变无关。两机制叠加：长地板 + 陡段，视觉上就是相变。

证据侧的关键对照有两个。其一，同一批模型输出（GPT-3 家族的多位算术）换用连续 metric（per-token edit distance）重画，跳变消失、呈平滑幂律；其二，反向构造：在训练动态已知完全平滑的 vision autoencoder 上，对重建质量套用阈值化 metric，同样能画出「涌现」曲线——artifact 的充分性被正面制造出来（[Schaeffer et al., 2023](../references.md)，[B]）。诚实的双方立场：Schaeffer 的分析证明了「metric artifact 足以产生已报告的大多数跳变」，并未证明真实相变不存在；Wei 的定义本身依赖「与随机基线不可区分」这一测量学概念，地板长度由评测预算 $n$ 决定而非模型性质。检验这类问题需要的是 checkpoint 密度：Pythia 的 16 个模型（70M–12B）每个开放 154 个中间 checkpoint、数据与训练顺序完全一致（[Biderman et al., 2023](../references.md)，arXiv:2304.01373，[A]）——对照多数公开工作每模型只放最终权重的做法，154 个沿训练轨迹的采样点让「平滑还是跳变」可以在训练动态维度上直接分辨；OLMo 2 进一步开放数据与训练日志（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]），使这类复核可以端到端重做。

## 8. 局限

幂律框架的适用边界，逐条对应一个被抹平的自由度：

1. **数据同质假设**。 $D$ 只数 token 个数，不问内容与重复度；数据混合比、去重、质量过滤的变动都会移动式 (2) 的系数，跨数据配方比较 loss 曲线前先核对这个口径。
2. **外推风险**。 $\alpha, \beta$ 在约两个数量级的 $(N, D)$ 网格上拟合，式 (5) 的使用场景却是向外推一两个数量级；系数的小误差经 $G$ 与指数放大成绝对预测的可见差异（第 4 节三法弥散即实例）。
3. **超参数被假设已调优**。幂律拟合的是「每个点都用好 LR、batch size、schedule」的包络面；训练现场的模型落在包络面之下，差距归 training 维度。
4. **架构被常数化**。系数在同族 dense decoder 内拟合；MoE（第 6 节）、混合架构、新位置编码都会改写 $(A, B, \alpha, \beta)$ ，跨架构比 loss 前先确认系数同源。
5. **loss 幂律 ≠ 能力幂律**。式 (7) 保证 $L^{*}$ 平滑下降，下游任务分数可以不平滑（第 7 节）——无论跳变是真实相变还是测量 artifact，「用 loss 曲线预测能力曲线」都要过 metric 这一关。

## 总结

| 内容 | 公式 | 要点 |
| --- | --- | --- |
| Kaplan 单变量律 | $L(N) = (N_c/N)^{\alpha_N}$ 等，式 (1) | $\alpha_N \approx 0.076$ 、 $\alpha_D \approx 0.095$ 、 $\alpha_C \approx 0.050$ ；log-log 直线，测量条件互斥 |
| Chinchilla 联合形式 | $L = E + A/N^{\alpha} + B/D^{\beta}$ ，式 (2) | $E = 1.69, A = 406.4, \alpha = 0.34, B = 410.7, \beta = 0.28$ ；$E$ 是语言熵地板 |
| compute-optimal 分配 | $N_{\mathrm{opt}} = G(C/6)^{a}$ ， $D_{\mathrm{opt}} = G^{-1}(C/6)^{b}$ ，式 (5) | $a = \beta/(\alpha+\beta) \approx 0.45$ ， $b = \alpha/(\alpha+\beta) \approx 0.55$ ；Table 2 三法 $a \in [0.46, 0.50]$ |
| 最优前沿 loss | $L^{*} - E \propto C^{-\alpha\beta/(\alpha+\beta)}$ ，式 (7) | 指数 $\approx 0.154$ ；10× 算力换 $L - E$ 乘 0.70 |
| Kaplan vs Chinchilla | $C^{0.73}$ 对 $C^{\approx 0.5}$ | 根源在拟合协议：LR schedule 未按 $D$ 退火；单变量律拼不出联合优化 |
| 全生命周期成本 | $\mathrm{TC} = 6ND + 2NQ$ ，式 (8) | $Q$ 大则最优 $N$ 向小偏移；Llama-3.1-8B × 15T 的打平 $Q^{*} \approx 3 \times 10^{13}$ token |
| MoE scaling | $C \approx 6 N_{\mathrm{act}} D$ | 激活参数继承 $N$ 的 FLOPs 角色；DeepSeekMoE 16B 以约 40% 计算量达 LLaMA2-7B 性能 |
| emergence 测量学 | $\mathrm{EM}(N) = (1 - cN^{-k})^{L}$ ，式 (9) | 涌现点 $N^{*} \propto L^{1/k}$ 是题目长度的人为函数；92%+ 案例集中于两类非线性 metric |

## 练习

1.（推导）Kaplan 论文使用的联合形式是 $L(N, D) = \left[ (N_c/N)^{\alpha_N/\alpha_D} + D_c/D \right]^{\alpha_D}$ 。对它重复第 3 节的推导（约束 $C \propto ND$ ），证明其隐含的分配指数是 $a = \alpha_D / (\alpha_N + \alpha_D)$ ，代入 $\alpha_N = 0.076$ 、 $\alpha_D = 0.095$ 得 $a \approx 0.56$ ；再对照论文报告的 $N_{\mathrm{opt}} \propto C^{0.73}$ ，说明这个张力支持第 4 节的哪一条解释。
2.（推导）补全式 (5) 的二阶论证：写出 $g(N) = L(N, C/6N)$ 的 $g''(N)$ ，指出 $\beta < 1$ 时数据项的二阶导为负、 $g''$ 不恒正；说明正文「提取公因子看 $g'$ 符号」的论证为什么不依赖凸性仍给出全局最小。再证明 $\beta \ge 1$ 时 $g$ 处处凸、此时「凸函数唯一驻点即全局最小」与符号论证等价。
3.（计算）给定 $C = 10^{24}$ FLOPs，用式 (5)(6) 求 $N_{\mathrm{opt}}$ 、 $D_{\mathrm{opt}}$ 与 $L^{*}$ ；一个 $8 \times 10^{9}$ 参数的模型要达到同一 $L^{*}$ 需要多少 token、训练成本变为几倍；若该模型终身推理量 $Q = 10^{14}$ token，用式 (8) 判断哪个方案总账更低。
4.（计算）取 $p(N) = 1 - cN^{-k}$ ，定 $c, k$ 使 $p(10^{9}) = 0.90$ 、 $p(10^{11}) = 0.99$ 。对 $L = 1, 8, 64$ 分别求式 (9) 的过渡中心 $N^{*}$ 与过渡宽度（dex）；若评测集只有 200 道题，画出三条曲线哪些区段被埋在地板下，并说明「同一模型、同一能力曲线，仅改题目长度就移动了涌现位置」。

## 参考

- [Kaplan et al., 2020](../references.md)：Scaling Laws for Neural Language Models，arXiv:2001.08361，[B]（key: kaplan）——三条单变量幂律与 $N_{\mathrm{opt}} \propto C^{0.73}$
- [Hoffmann et al., 2022](../references.md)：Training Compute-Optimal Large Language Models，arXiv:2203.15556，[B]（key: chinchilla）——联合形式式 (2) 与常数（Eq. 10）、三法分配指数（Table 2；Gopher 预算的 67B 预测见 Table 3，40B 见 §3.3 正文）、与 Kaplan 分歧的拟合方法分析
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——Llama-3.1-8B 预训练 15T token
- [Dai et al., 2024](../references.md)：DeepSeekMoE，arXiv:2401.06066，[A]（key: deepseekmoe）——MoE IsoFLOPs 曲线与 16B vs LLaMA2-7B 的 40% 计算量对照
- [Wei et al., 2022](../references.md)：Emergent Abilities of Large Language Models，arXiv:2206.07682，[B]（key: emergent-abilities）——emergence 定义与两类分类
- [Schaeffer et al., 2023](../references.md)：Are Emergent Abilities of Large Language Models a Mirage?，arXiv:2304.15004，[B]（key: emergence-mirage）——metric artifact 分析、92% 统计与换 metric 对照
- [Biderman et al., 2023](../references.md)：Pythia，arXiv:2304.01373，[A]（key: pythia）——16 模型 × 154 checkpoints 的 scaling/动态研究锚点
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——权重/数据/日志全开放锚点
