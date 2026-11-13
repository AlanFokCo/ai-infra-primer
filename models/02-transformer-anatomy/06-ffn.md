# 2.6 FFN 与非线性（FFN and Nonlinearities）

> 本章拆 2.1 式 (1) 里的第二个子层：两矩阵 FFN 的公式、逐位置性与 $d \to 4d \to d$ 参数账（ $8d^{2}$ / 层 ≈ 2/3 层参数）；激活函数的三个设计点——ReLU 硬门、GELU 的 $x\thinspace\Phi(x)$ 高斯 CDF 软门、GLU 族的门控乘积——以及 SwiGLU 把 $d_{\mathrm{ff}}$ 从 $4d$ 调到 $(8/3)d$ 的等参数推导；三家 $d = 4096$ 锚点的 intermediate_size 对账（OLMo-2-7B 11008 / Qwen3-8B 12288 / Llama-3.1-8B 14336，回答「为什么 Llama 不是 $4 \times 4096$」）；FFN-as-key-value-memory 分解（ $\mathrm{FFN}(x) = \sum_i v_i \thinspace \sigma(k_i \cdot x)$）与激活稀疏性证据；MoE 预告（→ 4.5）| 前置：1.2（ $2mnk$ FLOPs 口径、每层 FLOPs 分解表）、2.1（ $12Ld^{2}$ 账、逐位置二分法、residual stream 可加展开）、2.3（内积检索读法）| 锚点：Llama-3.1-8B config [A]、OLMo-2-7B config 与 modeling 代码 [A]、Qwen3-8B config [A]、Llama modeling 代码 [A]；Vaswani et al. [A]、Shazeer [B]、Hendrycks & Gimpel [B]、Geva et al. [B]、DeepSeek-V3 [A]

## 1. FFN 本体：逐位置两层 MLP

Transformer 原文的第二个子层是逐位置前馈网络（position-wise feed-forward network，FFN）（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762，[A]，§3.3）：

$$
\mathrm{FFN}(x) = \sigma(x W_1 + b_1)\thinspace W_2 + b_2 \qquad (1)
$$

**符号**： $x \in \mathbb{R}^{d}$：单个位置的 stream 向量（行向量约定，与 2.3 的 $Q = XW^{Q}$ 一致）； $W_1 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$：扩张投影； $W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$：收缩投影； $b_1 \in \mathbb{R}^{d_{\mathrm{ff}}}$、 $b_2 \in \mathbb{R}^{d}$：偏置； $\sigma$：逐元素激活函数，原文取 ReLU； $d_{\mathrm{ff}}$：FFN 中间维（config 字段 intermediate_size）。原文工作点 $d = 512$、 $d_{\mathrm{ff}} = 2048 = 4d$。

「逐位置」的精确含义：FFN 对 $[B, T, d]$ 的 $T$ 维逐位置施加同一组 $W_1, W_2$，实现上把 $B \cdot T$ 个位置摊平过一个共享权重的两层 MLP。这是 2.1 §1 二分法的另一半：attention 之外的一切逐位置。两个推论已经在前章用过——2.4 的因果归纳证明里 FFN 是不需要 mask 的那一步（不跨位置，无从泄漏未来）；2.5 §1 的注入点注记里，FFN 对位置的全部感知只能来自 attention 写回 stream 的内容。

层内分工也在这里定型：attention 子层对 value 做的是凸组合（2.3 式 (2)），跨位置混合，但逐位置的产出只有加权平均一种形态；FFN 给每个位置一次独立的非线性变换，是层内逐 token 表达能力的来源。体量账 2.1 §4 已算过： $W_1$、 $W_2$ 合计 $2\thinspace d\thinspace d_{\mathrm{ff}}$，取 $d_{\mathrm{ff}} = 4d$ 即每层 $8d^{2}$，占标准形层参数 $12d^{2}$ 的 $2/3$——整机参数的大头不在 attention 而在 FFN。这 $2/3$ 买了什么，§4 的记忆视角给出一个有证据的回答。

计算与内存侧的体量同样由 $d_{\mathrm{ff}}$ 决定：per-token FFN FLOPs $= 4\thinspace d\thinspace d_{\mathrm{ff}}$（标准形下 $16d^{2}$；SwiGLU 三矩阵为 $6\thinspace d\thinspace d_{\mathrm{ff}}$，见 1.2 §7）；激活账上 SwiGLU 每层每 token 要存门路、值路、乘积三份 $d_{\mathrm{ff}}$ 维向量（ $3\thinspace d_{\mathrm{ff}}$ 个元素，Llama-3.1-8B 即 43008 个，见 1.2 §8）——FFN 中间量是反向传播前显存里的大头之一，也是激活检查点（training 维度）要重的账。

shape 走读（标准形）：

```
x   [B, T, d]                        ← RMSNorm₂ 的输出（2.1 式 (1)）
 │  xW₁ + b₁: [B,T,d] × [d, 4d]
 ▼
h_pre  [B, T, 4d]                    ← 扩张：d → 4d
 │  σ（逐元素，shape 不变）
 ▼
h   [B, T, 4d]
 │  hW₂ + b₂: [B,T,4d] × [4d, d]
 ▼
FFN(x)  [B, T, d]                    ← 残差加回 stream
```

现代配方对式 (1) 的两处改动构成本章主线：激活 $\sigma$ 从 ReLU 换到 SwiGLU，矩阵从两个变三个（§2–§3）；bias 从标配变没有——Llama-3.1-8B 的 FFN 三个线性层全部 bias=False（transformers repo modeling_llama.py 的 `LlamaMLP`，[HuggingFace, modeling_llama.py](../references.md)，[A]），GPT-2 的 bias=True 是老配方（[Radford et al., 2019](../references.md)，[W]）。本章以下按无 bias 的现代形书写。

## 2. 激活 I：ReLU → GELU

### 2.1 ReLU 的硬门

$\mathrm{ReLU}(x) = \max(0, x)$：正半轴恒等、负半轴归零。作为门控器它有两个结构性事实。其一，负半轴导数精确为 0：预激活长期为负的单元收不到梯度、参数停更，即死区（dead zone）。其二，门控决策对输入量级不敏感：预激活是 $-0.01$ 还是 $-100$ 同样归零——「关」这个动作不带程度信息。第三个事实留给 §4：输出里精确为零的分量数逐输入变化，ReLU-FFN 的激活稀疏性是结构白送的——§4.2 证据链的第一条有一半来自这个逐元素性质。

### 2.2 GELU： $x \cdot \Phi(x)$ 的高斯 CDF 软门

GELU（Gaussian Error Linear Unit）把硬门换成标准正态 CDF 软门（[Hendrycks & Gimpel, 2016](../references.md)，arXiv:1606.08415，[B]）：

$$
\mathrm{GELU}(x) = x \thinspace \Phi(x), \qquad \Phi(x) = \frac{1}{2}\left(1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right) \qquad (2)
$$

**符号**： $\Phi$：标准正态分布的累积分布函数（CDF）， $\Phi(x) \in (0, 1)$； $\mathrm{erf}$：误差函数。

论文的构造视角：给输入乘一个随机 0/1 门 $m \sim \mathrm{Bernoulli}(\Phi(x))$——门以依赖输入的概率抽样，输入越大越可能被放行；取期望得 $\mathbb{E}[x \cdot m] = x\thinspace\Phi(x)$。GELU 就是这个随机门的期望化：dropout 的门与输入独立，GELU 的门概率由输入自身决定。期望化之后得到一个确定性光滑函数，两端渐近分明： $x \to +\infty$ 时 $\Phi \to 1$、退化为恒等； $x \to -\infty$ 时 $\Phi \to 0$，但任何有限 $x$ 处不精确为零。

导数（乘积法则）：

$$
\mathrm{GELU}'(x) = \Phi(x) + x \thinspace \phi(x) \qquad (3)
$$

**符号**： $\phi$：标准正态密度。对照 ReLU 读这条式子：负半轴梯度不再恒为零——硬死区消失；但 $\Phi(x) > 0$ 与 $x\thinspace\phi(x) < 0$ 相消， $\mathrm{GELU}'$ 在 $`x^{*} \approx -0.75`$ 处过零（函数唯一驻点，$`\mathrm{GELU}(x^{*}) \approx -0.17`$），向 $-\infty$ 端梯度渐近归零。软门消掉的是「梯度精确为零的区间」，不是「梯度小的区域」。

### 2.3 两个数值近似与 SiLU 接口

高斯 CDF 没有初等闭式，工程上用两个近似（同篇论文给出）：

- **logistic 近似**： $\Phi(x) \approx \sigma(1.702\thinspace x)$，故 $\mathrm{GELU}(x) \approx x\thinspace\sigma(1.702\thinspace x)$——正态 CDF 与 logistic 函数形状接近，1.702 是对齐系数（ $\sigma$ 为 sigmoid）；
- **tanh 近似**： $\mathrm{GELU}(x) \approx \dfrac{x}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\thinspace\left(x + 0.044715\thinspace x^{3}\right)\right)\right)$。

GPT-2 的 config 写着 activation = gelu_new（[Radford et al., 2019](../references.md)，[W]，config 口径）——HF 实现中 gelu_new 即 tanh 近似。

logistic 近似引出本章后半程的接口函数：

$$
\mathrm{SiLU}(x) = x \thinspace \sigma(x), \qquad \mathrm{Swish}_{\beta}(x) = x \thinspace \sigma(\beta x) \qquad (4)
$$

**符号**： $\sigma(x) = 1/(1 + e^{-x})$：sigmoid； $\beta$：门的陡度系数。对照 logistic 近似： $`\mathrm{GELU} \approx \mathrm{Swish}_{1.702}`$——两者几乎同一函数，差别只在 $\beta$。「 $x$ 乘一个 $(0,1)$ 的门」这一公共形式正是 GLU 族要推广的对象：下一步是把门从 $x$ 的逐元素函数换成另一路独立投影。

## 3. 激活 II：GLU 族与 SwiGLU

### 3.1 门控乘积的结构

门控线性单元（gated linear unit，GLU）的推广：门与值各自一路投影，逐元素相乘。Shazeer 把这一族接进 Transformer FFN（[Shazeer, 2020](../references.md)，arXiv:2002.05202，[B]）：式 (1) 中的 $\sigma(xW_1)$ 换成 $\mathrm{act}(xW) \odot xV$：

$$
\mathrm{FFN}_{\mathrm{GLU}}(x) = \big(\mathrm{act}(xW) \odot xV\big)\thinspace W_2 \qquad (5)
$$

**符号**： $W, V \in \mathbb{R}^{d \times d_{\mathrm{ff}}'}$：门路与值路投影； $\odot$：逐元素乘； $\mathrm{act}$：门函数，论文逐变体对照 sigmoid（GLU 原形）、ReLU（ReGLU）、GELU（GEGLU）、 $`\mathrm{Swish}_1`$（SwiGLU）； $W_2 \in \mathbb{R}^{d_{\mathrm{ff}}' \times d}$； $d_{\mathrm{ff}}'$：三矩阵结构的中间维，与两矩阵 $d_{\mathrm{ff}}$ 的关系由式 (8) 钉死。矩阵从两个变三个。

数据流：

```
x [B, T, d]
 ├─ xW [B, T, d_ff'] ── Swish₁ ──┐
 │                                ├─ ⊙ ── h [B, T, d_ff'] ── hW₂ ── FFN(x) [B, T, d]
 └─ xV [B, T, d_ff'] ────────────┘
```

论文在 T5 配方上的对照结论：GLU 变体在同参数同算力下优于 ReLU / GELU 两矩阵基线，SwiGLU 与 GEGLU 最好（[B]，无官方代码，结论粒度以论文实验设置为准）。现代开源锚点的默认选择是 SwiGLU：

$$
\mathrm{SwiGLU}(x) = \big(\mathrm{Swish}_{1}(xW) \odot xV\big)\thinspace W_2 \qquad (6)
$$

**符号**： $`\mathrm{Swish}_1(u) = u\thinspace\sigma(u)`$，即式 (4) 的 SiLU。Llama-3.1 与 OLMo 2 的官方实现均为此形（三个 bias-free 线性层 + SiLU 门，§3.4 走读）。

门控（gating）的数学作用分三条。

**数据依赖的乘性开关**。隐单元 $i$ 的输出为 $\sigma(k_i \cdot x) \cdot (u_i \cdot x)$：第一因子是输入决定的连续开关（值域 $(0,1)$），第二因子是未过非线性的线性读数。「检测什么模式」与「读出什么内容」由两路独立参数承担；对照 GELU，门 $\Phi(x)$ 与值 $x$ 同源同分量。

**双路梯度**。记 $g = xW$、 $v = xV$、 $h = \sigma(g) \odot v$，逐分量 Jacobian：

$$
\frac{\partial h_i}{\partial v_i} = \sigma(g_i), \qquad \frac{\partial h_i}{\partial g_i} = \sigma'(g_i)\thinspace v_i \qquad (7)
$$

**符号**： $\sigma'$：sigmoid 导数。值路的梯度被门缩放、门路的梯度被值缩放——只要另一因子非零，两路都收得到梯度。对照 ReLU-FFN： $u_i < 0$ 处导数精确为零。§2.1 的硬死区在三矩阵结构里被双路互救再软化一层。

代数字走一遍式 (7)： $g_i = 2$、 $v_i = 3$ 时 $\sigma(g_i) \approx 0.88$，值路梯度系数 0.88（近乎放行），门路梯度系数 $\sigma'(2) \times 3 \approx 0.105 \times 3 \approx 0.31$； $g_i = -2$、 $v_i = 3$ 时门近乎关闭（ $\sigma(-2) \approx 0.12$），值路梯度被压到 0.12，门路梯度仍是 $\sigma'(-2) \times 3 \approx 0.31$——门关着，门自己照样学。ReLU-FFN 在同样的负预激活处唯一的一路全黑。

**rank-1 二次型**。取 $\mathrm{act}$ 为恒等的极限：隐单元 $i$ 的输出是 $(k_i \cdot x)(u_i \cdot x) = x^{\top} (k_i u_i^{\top})\thinspace x$—— $x$ 的一个 rank-1 二次型。GLU 隐单元就是带软门的二次型；两矩阵 FFN 的隐单元 $\sigma(k \cdot x)$ 只是单方向的 ridge 函数。门控把逐层的逐位置函数类从「ridge 之和」扩到「带门二次型之和」，不靠加深度。

### 3.2 等参数推导： $d_{\mathrm{ff}}$ 从 $4d$ 到 $(8/3)d$

三矩阵比两矩阵多一路投影，要保住层参数与 per-token FLOPs，中间维必须收窄。两笔账（per-token，1.2 的 $2mnk$ 口径）：

| 结构 | 参数 | per-token FLOPs |
| --- | --- | --- |
| 两矩阵，中间维 $d_{\mathrm{ff}}$ | $2\thinspace d\thinspace d_{\mathrm{ff}}$ | $4\thinspace d\thinspace d_{\mathrm{ff}}$ |
| 三矩阵，中间维 $d_{\mathrm{ff}}'$ | $3\thinspace d\thinspace d_{\mathrm{ff}}'$ | $6\thinspace d\thinspace d_{\mathrm{ff}}'$ |

参数与 FLOPs 同为矩阵元素数的线性函数，一个等式同时钉死两笔账：

$$
3\thinspace d\thinspace d_{\mathrm{ff}}' = 2\thinspace d\thinspace d_{\mathrm{ff}} \hspace{5mu}\Longrightarrow\hspace{5mu} d_{\mathrm{ff}}' = \frac{2}{3}\thinspace d_{\mathrm{ff}}, \qquad d_{\mathrm{ff}} = 4d \hspace{5mu}\Longrightarrow\hspace{5mu} d_{\mathrm{ff}}' = \frac{8}{3}\thinspace d \qquad (8)
$$

这就是 Shazeer 论文的等参数设置（[B]）：GLU 变体取 $d_{\mathrm{ff}}' = (2/3)\thinspace d_{\mathrm{ff}}$，与基线同参数同算力对照，赢输才能归因于结构而非预算。对 $d = 4096$： $8d/3 = 10922.67$。实现侧还有一层取整：中间维通常取 256 的倍数（硬件 tiling 对齐）， $\ge 8d/3$ 的最小 256 倍数是 $43 \times 256 = 11008$——§3.3 的 OLMo-2-7B 就停在这里。

规则的适用范围也要写明：式 (8) 只管「矩阵数从 2 变 3 时中间维怎么换算」，不管 $d_{\mathrm{ff}}$ 绝对水平的选择—— $4d$ 本身出自原文工作点（式 (1) 注），没有推导支撑。§3.3 的对账会显示：实际取值以 $8/3$ 为基线、向上分布。

### 3.3 三家锚点对账：为什么 Llama-3.1 不是 $4 \times 4096$

$d = 4096$ 家族的真实 intermediate_size（config 条目均 [A]）：

| 模型 | $d_{\mathrm{ff}}$ | $d_{\mathrm{ff}}/d$ | 每层 FFN 参数（ $d^{2}$ 单位） | 每层 FFN 参数（绝对值） | FFN 占层参数 |
| --- | --- | --- | --- | --- | --- |
| 标准形（2.1 §4） | $4d$ | 4.0 | $8d^{2}$ | — | $`2/3 \approx 67\%`$ |
| OLMo-2-7B | 11008 | 2.6875 | $8.0625d^{2}$ | 135266304 | 66.8% |
| Qwen3-8B | 12288 | 3.0 | $9d^{2}$ | 150994944 | 78.3% |
| Llama-3.1-8B | 14336 | 3.5 | $10.5d^{2}$ | 176160768 | 80.8% |

绝对值列 $= 3\thinspace d\thinspace d_{\mathrm{ff}}$；占比分母「层参数」= FFN + attention（OLMo-2 为 MHA $4d^{2}$，另两家 GQA $2.5d^{2}$，分解见 2.1 §4）。FLOPs 侧同比例：per-token FFN FLOPs $= 6\thinspace d\thinspace d_{\mathrm{ff}}$，Llama-3.1-8B 为 352.3 M、占层 FLOPs 81%（1.2 §7 的表）——参数占比与 FLOPs 占比同数，因为每个线性层的 FLOPs 恒等于其参数量的 2 倍（1.2 式 (1)）。

逐行读。

**OLMo-2-7B 就是等参数规则的落地**。 $11008 = 43 \times 256$ 恰是 $\ge 10922.67$ 的最小 256 倍数，FFN 占比回到标准形的 $2/3$（论文 [Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]；config [A]）。

**Llama-3.1-8B 的 14336 = 3.5d，比等参数值宽 31.25%**（ $14336 = (8/3)d \times 1.3125$）。「为什么更宽」不写在任何 config 里，但预算账可以直接读：GQA 4:1 把 attention 从 $4d^{2}$ 省到 $2.5d^{2}$，FFN 从 $8d^{2}$ 加到 $10.5d^{2}$——省下的 $1.5d^{2}$ 全部补贴 FFN 还倒贴 $1d^{2}$，净 $+1d^{2}$ / 层，层参数 $13Ld^{2}$（2.1 §4 的账）。而 $4 \times 4096 = 16384$ 被三家共同否决：它是把两矩阵时代的宽度原样搬进三矩阵结构——若 Llama 取 $d_{\mathrm{ff}} = 16384$，FFN 将是 $3 \times d \times 4d = 12d^{2}$ / 层，在现状上再加 $1.5d^{2}$。等参数规则是起点不是铁律：它回答「换三矩阵后中间维收窄到多少」，不回答「总预算该给 FFN 多少」——后一个问题各家取值不同（2.69d / 3d / 3.5d），但表达形式一致：全部 256 倍数，全部在 $8/3$ 基线之上。这条 $8/3$ 规则是后续章节的复用接口：4.5 的 MoE expert 尺寸与 Part 5 的逐家对账都以它为基准。

整机口径再放大一层：Llama-3.1-8B 全模型 FFN 参数 $= 32 \times 176160768 \approx 5.64 \times 10^{9}$，占总参数 $8.03 \times 10^{9}$ 的 70.2%；OLMo-2-7B 为 $32 \times 135266304 \approx 4.33 \times 10^{9}$，占 $7.30 \times 10^{9}$ 的 59.3%（两家总参数见 2.1 §4 表）——标准形的「2/3」在 Llama 这类宽 FFN 配方上实际更接近 70%，在贴 $8/3$ 规则、MHA 不打折的 OLMo 上反而低于 $2/3$。

### 3.4 实现对照

transformers repo 的 `LlamaMLP`（[HuggingFace, modeling_llama.py](../references.md)，[A]）：前向为 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`，act_fn 为 SiLU——式 (6) 的逐行直译， $W$ = gate_proj、 $V$ = up_proj、 $W_2$ = down_proj；OLMo 2 的 modeling 文件同构（[HuggingFace, modeling_olmo2.py](../references.md)，[A]）。教学版（教学用，非优化实现）：

```python
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):                # 式 (6)：d → d_ff → d
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)     # W：门路
        self.up_proj   = nn.Linear(d, d_ff, bias=False)     # V：值路
        self.down_proj = nn.Linear(d_ff, d, bias=False)     # W_2

    def forward(self, x):                  # x: [B, T, d]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

与官方实现的差别只有类与字段命名。退回两矩阵基线（式 (1)，ReLU、 $d_{\mathrm{ff}} = 4d$）只需删掉 gate_proj 并把前向换成 `down_proj(F.relu(up_proj(x)))`——两代 FFN 的结构差异就这一行。

## 4. FFN 作为 key-value 记忆

### 4.1 Geva 分解：把两层 MLP 读成一张记忆表

两矩阵无 bias FFN 可以逐隐单元展开。记 $W_1$ 的第 $i$ 列为 $k_i \in \mathbb{R}^{d}$、 $W_2$ 的第 $i$ 行为 $v_i \in \mathbb{R}^{d}$，则 $`(xW_1)_i = k_i \cdot x`$，矩阵乘逐列逐行展开：

$$
\mathrm{FFN}(x) = \sum_{i=1}^{d_{\mathrm{ff}}} \sigma(k_i \cdot x)\hspace{5mu} v_i \qquad (9)
$$

**符号**： $k_i$：第 $i$ 号记忆的 key； $v_i$：第 $i$ 号记忆的 value（ $d$ 维，直接写回 residual stream）； $\sigma(k_i \cdot x)$：第 $i$ 号记忆对输入 $x$ 的取出系数。

Geva et al. 的读法（[Geva et al., 2020](../references.md)，arXiv:2012.14913，[B]）：FFN 是一张内容寻址的键值记忆（key-value memory）表—— $d_{\mathrm{ff}}$ 个条目，每条一根 key 一根 value；输入与全部 key 算内积、过 $\sigma$ 得系数，输出是 value 的加权和。与 attention 的同构逐项对照：

| | attention（2.3 式 (2)） | FFN（式 (9)） |
| --- | --- | --- |
| key 是什么 | 其他位置的 stream 投影（数据依赖） | $W_1$ 的列（参数，训练后固定） |
| 权重归一化 | softmax，行和为 1 | $\sigma$ 逐分量，无归一化 |
| 查表粒度 | 每位置查全体位置 | 每位置查同一张固定表 |
| value 写什么 | 其他位置的内容 | 参数化的 $d$ 维向量 |

差别本身就是结论：FFN 是**参数化记忆**——条目在训练后固定，上下文只能决定取出哪些（系数），不能改写条目。§1 留下的「 $2/3$ 参数买了什么」由此有了候选答案：一张 $d_{\mathrm{ff}} \times L$ 条目的记忆表（Llama-3.1-8B： $32 \times 14336 = 458752$ 条）。

表里「无归一化」一行值得展开。softmax 权重分一个总和为 1 的预算：多给位置 $s$ 一分就要从别处扣。FFN 系数没有总量约束——一个输入可以同时把多个条目开到很大（ $\sigma(k_i \cdot x) > 1$ 合法），输出是这些 value 的叠加。「一个 token 同时命中多个模式」由此表达：地址并行匹配，取出量各自独立。代价是条目间没有竞争机制，分工全靠训练塑造——§4.2 的稀疏性与模式分层是塑造出来的结果，不是结构保证。

SwiGLU 下同一分解成立，取出系数换成双因子乘积：

$$
\mathrm{SwiGLU}(x) = \sum_{i=1}^{d_{\mathrm{ff}}'} \underbrace{\mathrm{Swish}_{1}(g_i \cdot x)\thinspace (u_i \cdot x)}_{c_i(x)}\hspace{5mu} v_i \qquad (10)
$$

**符号**： $g_i$、 $u_i$： $W$、 $V$ 的第 $i$ 列； $v_i$： $W_2$ 的第 $i$ 行； $c_i(x)$：门与线性响应的乘积。记忆视角不被三矩阵结构破坏——「取出量」从单级匹配升级为「匹配 × 内容」两级。

### 4.2 证据链：稀疏激活、模式分层、value 的词表投影

论文的分析在 ReLU 激活的 T5 系上做（[B]，无官方代码，结论粒度以其实验设置为准），三条：

1. **激活稀疏性**。逐输入看，系数 $c_i(x)$ 呈重尾分布：绝大多数 key 的内积落在 $\sigma$ 的近零区（ReLU 下负预激活精确归零），激活质量集中在少数 key 上——单个位置实际取出的条目远少于 $d_{\mathrm{ff}}$。
2. **key 的模式分层**。人工检视各 key 的高激活输入：浅层 key 对应表面模式（词形、前后缀），深层 key 对应语义模式（话题、组合结构）——记忆条目按层组织成由浅到深的模式探测器。
3. **value 是可读出的输出分布**。把 $v_i$ 乘上输出头读回词表（logit lens 操作，2.1 §2），其诱导分布往往集中在「能接续触发该 key 的模式」的 token 上；各层 value 沿 residual stream 累加（2.1 式 (2)），最终分布由这些记忆的写回逐步精化。

机制含义：FFN 把「输入模式 → 输出修正」的关联物化成键值对存进参数；stream 总线视角下，每层 FFN 做的是查表取回若干 value 写回总线。4.7 的可解释性工具对 FFN 的操作（logit lens、activation patching）都以这个分解为前提。证据等级注记：这三条是 [B] 级结论，适用范围以论文的模型与任务为准；但式 (9) 本身是恒等式——任何两矩阵 FFN 都可以这样读，分解的有效性不依赖实验。

## 5. 从稀疏激活到 MoE：预告（→ 4.5）

§4.2 第一条证据有一个直接的工程读法：若每位置只有一小撮条目被取出，dense FFN「每 token 为全表 $d_{\mathrm{ff}}$ 付 FLOPs」就是把算力花在没点火的条目上。MoE（Mixture-of-Experts）把这个观察做成架构：单层 FFN 复制成 $n$ 份（expert），每位置由 router 选 top-$`k`$ 个激活——总记忆容量乘 $n$，per-token FLOPs 只乘 $k$。1.3 已在 scaling 曲线上登记过这笔 $N_{\mathrm{act}}$ / $N_{\mathrm{tot}}$ 分离的账；实现端的极端值是 DeepSeek-V3：每层 256 个 routed expert + 1 个共享 expert、每 token 激活 8 个、前 3 层仍为 dense FFN，671B 总参数 / 37B 激活（[DeepSeek-V3 config](../references.md)，[A]；[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）。对照式 (8) 的「等参数换结构」，MoE 是同一思想在 FFN 维度上的稀疏化升级：等 FLOPs 换容量。本章交给 4.5 的接口有两件：式 (8) 的等参数口径（单个 expert 的尺寸按 $8/3$ 规则起算），与式 (10) 的系数视角（router 的 top-$`k`$ 选择相当于把 $c_i(x)$ 的软稀疏改成条目粒度的硬稀疏）。路由公式、负载均衡与细粒度 expert 的推导见 4.5。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 两矩阵 FFN | $\mathrm{FFN}(x) = \sigma(xW_1)W_2$，式 (1) | 逐位置； $d_{\mathrm{ff}} = 4d$ → $8d^{2}$/层 ≈ 2/3 层参数；现代配方无 bias |
| GELU | $x\thinspace\Phi(x)$，式 (2)(3) | $\mathrm{Bernoulli}(\Phi(x))$ 随机门的期望；唯一驻点 $`x^{*} \approx -0.75`$； $`\approx \mathrm{Swish}_{1.702}`$；GPT-2 gelu_new = tanh 近似 [W] |
| SwiGLU | $`(\mathrm{Swish}_1(xW) \odot xV)W_2`$，式 (5)(6) | 门控三作用：数据依赖开关、双路梯度（式 (7)）、rank-1 二次型；同参同算力优于 ReLU / GELU 基线 [B] |
| 等参数规则 | $3d\thinspace d_{\mathrm{ff}}' = 2d\thinspace d_{\mathrm{ff}}$ ⟹ $d_{\mathrm{ff}}' = 8d/3$，式 (8) | $d = 4096$ → 10922.67；参数与 FLOPs 同一条件钉死；4.5 / Part 5 复用 |
| 三家对账 | OLMo-2 11008（≈2.69d，43×256）/ Qwen3 12288（3d）/ Llama-3.1 14336（3.5d） | FFN 占层 67% / 78% / 81%；规则是起点非铁律；16384 = 两矩阵宽度直搬，被共同否决 |
| FFN = KV 记忆 | $\mathrm{FFN}(x) = \sum_i \sigma(k_i \cdot x)\thinspace v_i$，式 (9)(10) | $d_{\mathrm{ff}} \times L$ 条参数化记忆（Llama-3.1-8B 458752 条）；稀疏激活、模式分层、value→词表分布 [B] |
| MoE 预告 | $n$ expert、top-$`k`$ 路由 | 等 FLOPs 换容量；DeepSeek-V3：256+1、top-8、671B/37B [A] → 4.5 |

## 练习

1.（推导）把式 (8) 一般化：两矩阵 FFN 中间维 $\alpha d$、GLU-FFN 中间维 $\beta d$，等参数等 FLOPs 给出 $\beta = 2\alpha/3$（验证 $\alpha = 4 \to \beta = 8/3$）。再计入 bias：两矩阵带 $b_1, b_2$、三矩阵带三个 bias 时等式怎么改？说明为什么 $d \gg 1$ 时结论不变。
2.（计算）Qwen3-8B（ $L = 36$、 $d = 4096$、 $d_{\mathrm{ff}} = 12288$、GQA $H_{kv} = 8$）：以 $d^{2}$ 为单位算每层 attention 与 FFN 参数、FFN 占层比、层参数对 $12Ld^{2}$ 的偏差；若 $d_{\mathrm{ff}}$ 改为 11008，层参数与 FFN 占层比各变多少？
3.（推导）(a) 从随机门定义 $\mathrm{GELU}(x) = \mathbb{E}[x \cdot m]$、 $m \sim \mathrm{Bernoulli}(\Phi(x))$ 推出式 (2)；(b) 推式 (3)，并数值验证 $`x^{*} \approx -0.75`$ 是唯一驻点（提示：分 $x \ge 0$ 与 $x < 0$ 讨论 $\Phi(x) + x\phi(x)$ 的符号）；(c) 由此解释「软门消掉硬死区」的精确含义——消掉的是梯度恒零的区间，不是梯度小的区域。
4.（推导 + 代码）(a) 从 $\mathrm{FFN}(x) = \sigma(xW_1)W_2$ 推出式 (9)，写明 $k_i$、 $v_i$ 与两矩阵的行/列对应；(b) 基于 §3.4 的教学 SwiGLUFFN，用随机 $x$ 在 fp64 下验证式 (10)：forward 输出与显式求和 $\sum_i c_i(x)\thinspace v_i$ allclose；(c) 可选：加载 OLMo-2-7B 开放权重（[A]），取一批真实输入统计某层 $|c_i|$，画激活质量的累积分布曲线，验证 §4.2 第一条。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——式 (1) 与 $d_{\mathrm{ff}} = 4d$ 工作点出自 §3.3
- [Hendrycks & Gimpel, 2016](../references.md)：Gaussian Error Linear Units (GELUs)，arXiv:1606.08415，[B]（key: gelu）——式 (2)(3)、随机门视角与两个数值近似
- [Shazeer, 2020](../references.md)：GLU Variants Improve Transformer，arXiv:2002.05202，[B]（key: glu-variants）——式 (5)(6)、GLU 变体对照与等参数设置式 (8)
- [Geva et al., 2020](../references.md)：Transformer Feed-Forward Layers Are Key-Value Memories，arXiv:2012.14913，[B]（key: ffn-memory）——式 (9) 与 §4.2 证据链
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——§5 的 671B/37B 口径
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——OLMo-2-7B 配方出处
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——intermediate_size = 14336
- [Qwen, Qwen3-8B config.json](../references.md)：[A]（key: qwen3-8b-config）——intermediate_size = 12288
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——intermediate_size = 11008
- [Radford et al., 2019](../references.md)：[W]（key: gpt2）——gelu_new 与 bias=True 老配方
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——`LlamaMLP`：gate/up/down_proj + SiLU
- [HuggingFace, transformers repo OLMo 2 modeling 代码](../references.md)：[A]（key: olmo2-modeling）——同构 MLP 对照
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——n_routed_experts = 256、num_experts_per_tok = 8、first_k_dense_replace = 3
