# 4.3 Linear Attention 与 State-Space Models（Linear Attention and State-Space Models）

> 本章把「换掉 softmax」这一件事推到底：kernel 视角的结合律重排（causal FLOPs $O(T^{2}d) \to O(Td^{2})$ 的完整账）与 FAVOR+ 随机特征、linear attention ≡ 矩阵状态 RNN 的等价性证明，然后沿「固定状态」主轴展开 RetNet（ $\gamma$ 衰减三形式）、DeltaNet（delta rule = 对当前联想的一步在线 GD）、GDN（ $\alpha$ 遗忘 $+$ 定点擦除 $+$ 写入）、GDN-2（erase/write 解耦）、KDA（对角门逐通道遗忘），补上 chunkwise 并行的 WY 表示与 Mamba/S6 的 ZOH 离散化推导，以固定 $d_v \times d_k$ 状态的 associative recall 上界收尾 | 前置：1.2（ $2mnk$ 口径）、2.3（attention 公式、FLOPs 两项账）、2.4（causal mask、KV cache 式 (9)、prefill/decode）、3.3 §5（linear attention 层模拟一步 GD 的构造） | 锚点：fla-org/flash-linear-attention [A]、NVlabs/GatedDeltaNet 与 GatedDeltaNet-2 [A]、MoonshotAI/Kimi-Linear [A]、state-spaces/mamba [A]、microsoft/unilm [A]；Katharopoulos et al. [A]、Choromanski et al. [A]、Sun et al. [A]、Yang et al. [A]、Hatamizadeh et al. [A]、Kimi Team [A]、Gu & Dao [A]、Dao & Gu [A]

## 1. Kernel 视角：换掉 softmax，交换结合顺序

2.3 式 (4) 加 causal mask（2.4 式 (3)）后逐行写出，缩放常数折进 score：

$$
o_t = \frac{\sum_{s=1}^{t} \mathrm{sim}(q_t, k_s)\thinspace v_s}{\sum_{s=1}^{t} \mathrm{sim}(q_t, k_s)}, \qquad \mathrm{sim}(q, k) = \exp\left(\frac{\langle q, k \rangle}{\sqrt{d_k}}\right) \qquad (1)
$$

**符号**： $q_t, k_s \in \mathbb{R}^{d_k}$； $v_s \in \mathbb{R}^{d_v}$；单头口径（多头逐头成立）。式 (1) 对 $\mathrm{sim}$ 的全部要求只有非负：保证分母不为零、权重可解释为凸组合系数。

Katharopoulos et al. 的替换（[Katharopoulos et al., 2020](../references.md)，arXiv:2006.16236，[A]；官方代码 idiap/fast-transformers）：把 $\exp$ 核换成特征映射（feature map）内积

$$
\mathrm{sim}(q, k) \approx \phi(q)^{\top}\phi(k), \qquad \phi: \mathbb{R}^{d_k} \to \mathbb{R}^{d_r}, \quad \phi(\cdot) \ge 0 \qquad (2)
$$

**符号**： $d_r$：特征维。原文取 $\phi(x) = \mathrm{elu}(x) + 1$（逐元素、非负、 $d_r = d_k$）。

代入式 (1)，分子分母各自换序：

$$
o_t = \frac{\sum_{s=1}^{t} v_s \left( \phi(k_s)^{\top} \phi(q_t) \right)}{\sum_{s=1}^{t} \phi(k_s)^{\top} \phi(q_t)} = \frac{\left( \sum_{s=1}^{t} v_s \phi(k_s)^{\top} \right) \phi(q_t)}{\left( \sum_{s=1}^{t} \phi(k_s) \right)^{\top} \phi(q_t)} \qquad (3)
$$

第二步只有标量 $\phi(k_s)^{\top}\phi(q_t)$ 的换位与求和合并。分子的核心对象 $\sum_s v_s\phi(k_s)^{\top}$ 是一个 $d_v \times d_r$ 矩阵——**全部历史被压进一份与 $T$ 无关的矩阵**；§3 证明它就是 RNN 状态。非因果情形的矩阵写法即结合律重排：

$$
O = \left( \Phi(Q)\Phi(K)^{\top} \right) V = \Phi(Q) \left( \Phi(K)^{\top} V \right) \qquad (4)
$$

**符号**： $\Phi(Q), \Phi(K) \in \mathbb{R}^{T \times d_r}$： $\phi$ 逐行作用； $V \in \mathbb{R}^{T \times d_v}$。左右两种结合顺序数学相等、FLOPs 不等（1.2 的 $2mnk$ 口径，单头）：

| 结合顺序 | 第一步 | 第二步 | 合计 |
| --- | --- | --- | --- |
| 左：先算核矩阵 | $\Phi(Q)\Phi(K)^{\top}$： $[T,d_r] \times [d_r,T]$， $2T^{2}d_r$ | $(\cdot)V$： $2T^{2}d_v$ | $2T^{2}(d_r + d_v)$ |
| 右：先算状态 | $\Phi(K)^{\top}V$： $[d_r,T] \times [T,d_v]$， $2Td_rd_v$ | $\Phi(Q)(\cdot)$： $2Td_rd_v$ | $4Td_rd_v$ |

causal 情形下左序只需下三角、约减半为 $T^{2}(d_r+d_v)$（2.3 §7 的同一系数）；右序不变——前缀结构天然因果，没有浪费的半张矩阵。取 $d_r = d_v = d$：左序 causal $2T^{2}d$ 对右序 $4Td^{2}$，比值 $T / (2d)$，交叉点 $T = 2d$。代入 head 维典型档 $d$ = 128： $T$ = 8192 时右序省 32 倍； $T$ = 131072 时省 512 倍。这就是「 $O(T^{2}d) \to O(Td^{2})$」的全部内容：不是新数学，是乘法结合律加一张 FLOPs 账。内存侧对照同构：左序的最大对象是 $T \times T$ 核矩阵（2.3 §7 的账），右序是 $d_r \times d_v$ 状态。

两个代价注记。其一，特征化本身要钱：Performer 型 $\phi$ 含一次 $d_k \times d_r$ 投影（§2），逐 token $2d_kd_r$，与右序主项同阶、不改结论。其二，右序的训练形态不是把递推跑 $T$ 步——那是串行灾难；可并行形态是 chunkwise，连同 delta-rule 族的版本在 §8 统一处理。

补一条视角注记：式 (4) 右序的 $\Phi(K)^{\top}V$ 不是普通中间矩阵——按前缀看，它正是 §3 式 (7) 的状态 $S_t$（布局注记：此处 $\Phi(K)^{\top}V \in \mathbb{R}^{d_r \times d_v}$ 是 key×value 布局，与 §3 起全书统一的状态约定 $S \in \mathbb{R}^{d_v \times d_r}$ 互为转置；§1-2 保留 attention 文献习惯的矩阵写法，§3 起一律用约定布局）。结合律重排因此同时产出两个对象：训练的矩阵算法与推理的递推状态；本章全部推导都从这一等式长出。左序没有这种递推对应物： $T \times T$ 核矩阵的第 $t$ 行依赖全部 $t$ 个 key，是天然的并行对象，也天然背着 $O(T^{2})$ 内存。

## 2. FAVOR+：用随机特征无偏逼近 softmax 核

§1 的替换是近似——能不能选 $\phi$ 使它逼近的对象恰好是 $\exp$ 核本身？Performer 的 FAVOR+（[Choromanski et al., 2021](../references.md)，arXiv:2009.14794，[A]）用高斯矩母函数搭桥。 $\omega \sim \mathcal{N}(0, I_{d_k})$ 时 $\mathbb{E}[e^{\omega^{\top}u}] = e^{\lVert u \rVert^{2}/2}$，取 $u = q + k$：

$$
\exp\left(q^{\top}k\right) = e^{-\lVert q \rVert^{2}/2}\thinspace e^{-\lVert k \rVert^{2}/2}\thinspace \mathbb{E}_{\omega}\left[ e^{\omega^{\top}(q+k)} \right] = \mathbb{E}_{\omega}\left[ \left( e^{\omega^{\top}q - \lVert q \rVert^{2}/2} \right) \left( e^{\omega^{\top}k - \lVert k \rVert^{2}/2} \right) \right] \qquad (5)
$$

右边恰是两个同构因子的期望内积。定义 $m$ 维随机特征：

$$
\phi(x) = \frac{1}{\sqrt{m}}\thinspace e^{-\lVert x \rVert^{2}/2} \begin{bmatrix} e^{\omega_1^{\top}x} \\ \vdots \\ e^{\omega_m^{\top}x} \end{bmatrix}, \qquad \omega_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, I_{d_k}) \qquad (6)
$$

则 $\mathbb{E}[\phi(q)^{\top}\phi(k)] = e^{q^{\top}k}$：softmax 核的**无偏** Monte Carlo 估计（ $1/\sqrt{d_k}$ 折进 $q$ 的预处理），误差随 $1/\sqrt{m}$ 收缩。「positive」指特征恒正——对照早期 trigonometric 特征（ $\sin / \cos$ 可负，分母 $\sum_s \phi(k_s)^{\top}\phi(q_t)$ 有近零的数值风险）；「orthogonal」指 $\omega_i$ 之间做正交化以进一步降方差（论文的方差分析）。

定位注记：FAVOR+ 逼近的是核，attention 输出仍带 $O(1/\sqrt{m})$ 级随机误差——与 FlashAttention 的数学精确（4.1）是两个物种；它买到的是 $O(T^{2}) \to O(Tm)$ 且 $\phi$ 固定、无需学习。

实务口径： $m$ 典型取 256 档， $d_k$ = 128 时特征维翻倍，右序主项成 $O(Tmd_v)$——只有 $T \gg m$ 才净赚；短序列下 FAVOR+ 两头不占优（误差在、FLOPs 没省）。它的生态位是「 $T$ 极大且可容忍随机近似」的场景：要精确回 4.1，要确定性线性化走 §5 起的 delta-rule 族。

## 3. Recurrent 形式：linear attention ≡ 矩阵状态 RNN

式 (3) 的两个前缀和是全章枢纽。定义（**全章与全书统一约定**：状态 $S_t \in \mathbb{R}^{d_v \times d_k}$、读出 $o_t = S_t\thinspace\phi(q_t)$、写入 $v_t\phi(k_t)^{\top}$、擦除右乘；delta-rule 族 $\phi$ = 恒等，下式中 $d_r$ 即 $d_k$）：

$$
S_t = \sum_{s=1}^{t} v_s \phi(k_s)^{\top} \in \mathbb{R}^{d_v \times d_r}, \qquad z_t = \sum_{s=1}^{t} \phi(k_s) \in \mathbb{R}^{d_r} \qquad (7)
$$

式 (3) 于是读作 $o_t = S_t\phi(q_t)\thinspace/\thinspace\left( z_t^{\top}\phi(q_t) \right)$。

**命题（linear attention ≡ 矩阵状态 RNN）**。取 $S_0 = 0$、 $z_0 = 0$，递推

$$
S_t = S_{t-1} + v_t \phi(k_t)^{\top}, \qquad z_t = z_{t-1} + \phi(k_t), \qquad o_t = \frac{S_t\thinspace \phi(q_t)}{z_t^{\top} \phi(q_t)} \qquad (8)
$$

与式 (3) 的因果 kernel attention 逐位置相等；反之对和式做差分即得递推。

**证明**。归纳： $t = 0$ 显然；设 $S_{t-1} = \sum_{s \le t-1} v_s\phi(k_s)^{\top}$，则 $S_t = \sum_{s \le t-1}(\cdot) + v_t\phi(k_t)^{\top} = \sum_{s \le t} v_s\phi(k_s)^{\top}$； $z_t$ 同理。代入读出： $S_t\phi(q_t) = \sum_{s\le t} v_s\left(\phi(k_s)^{\top}\phi(q_t)\right)$ 正是式 (3) 分子， $z_t^{\top}\phi(q_t)$ 是分母。反向： $S_t - S_{t-1} = v_t\phi(k_t)^{\top}$ 由和式定义立得。 $\square$

这就是「Transformers are RNNs」的标题结论（[Katharopoulos et al., 2020](../references.md)，arXiv:2006.16236，[A]）：**同一函数的两种算图**——训练走式 (3) 的前缀和并行形（§8 的 chunkwise），decode 走式 (8) 的递推形，输出逐位置同一（浮点累加序差异除外）。

等价性的用法是双向搬运：一边显然的性质另一边自动成立。递推侧显然的是 decode 定长内存、逐步 $O(d_vd_r)$；并行侧显然的是训练可 matmul 化、可借用 attention 的分析框架——§10 的 rank 界就在并行侧证明。两端差异只剩浮点：递推按时间序累加、并行按块累加，累加序不同带来末位差异，教学代码的 `allclose` 容差即为此设。另登记一个推论：任何「在线算法」若想与某个线性层序列等价，其状态更新必须能写成式 (8) 形的秩一累加——3.3 §5 的 ICL-GD 构造正是沿这条路走的。

**decode 账**（单头、逐 token）：秩一写入与读出各 $2d_vd_r$ FLOPs，与 $T$ 无关；状态常驻 $d_vd_r + d_r$ 个元素。对照 KV cache（2.4 式 (9)，单头每层）： $2td_{\mathrm{head}}b$ 字节且每步全量读出。代入 $d_r = d_v = d_{\mathrm{head}}$ = 128、bf16、 $t$ = 131072：cache 每 token 读 $2 \times 131072 \times 128 \times 2$ bytes = 64 MiB，状态 $128 \times 128 \times 2$ bytes = 32 KiB——2048 倍（ $= 2t/d$，随 $t$ 继续拉大；乘回头数层数不改比例）。

**状态容量含义**：递推把任意长前缀压进 $d_v \times d_r$ 个定数。这是省内存的根源，也是 §10 回忆上界的根源——同一枚硬币。

训练侧同样有账：式 (7) 的朴素 cumsum 实现要为每个位置物化一份 $d_v \times d_r$ 前缀状态，内存 $O(Td_vd_r)$——上面教学代码里的 `S` 正是这个形状， $T$ = 8192、 $d_v = d_r$ = 128、bf16 时已占 256 MiB（单头单序列）。所以工程形态从来不是全前缀 cumsum，而是 §8 的 chunkwise：块内并行、块间只传一份状态。

教学实现（教学用，非优化实现；生产 kernel 见 §8）：

```python
import torch
import torch.nn.functional as F

def phi(x):
    return F.elu(x) + 1.0               # Katharopoulos 特征，d_r = d_k

def linear_parallel(Q, K, V):
    """式 (3) 前缀和并行形。Q,K: [B,T,d_k]; V: [B,T,d_v]"""
    Qp, Kp = phi(Q), phi(K)
    S = torch.cumsum(V.unsqueeze(-1) * Kp.unsqueeze(-2), dim=1)  # [B,T,d_v,d_r]
    z = torch.cumsum(Kp, dim=1)                                  # [B,T,d_r]
    num = (S * Qp.unsqueeze(-2)).sum(-1)                         # S_t φ(q_t)
    den = (z * Qp).sum(-1, keepdim=True)                         # z_tᵀ φ(q_t)
    return num / den

def linear_recurrent(Q, K, V):
    """式 (8) 递推形，与 parallel 逐位置相等"""
    B, T, dk = Q.shape
    Kp, Qp = phi(K), phi(Q)
    S = torch.zeros(B, V.shape[-1], dk)   # S_0，d_v×d_r
    z = torch.zeros(B, dk)
    outs = []
    for t in range(T):
        S = S + V[:, t].unsqueeze(-1) * Kp[:, t].unsqueeze(-2)  # 写入 v_t φ(k_t)ᵀ
        z = z + Kp[:, t]
        o = (S * Qp[:, t].unsqueeze(-2)).sum(-1)
        outs.append(o / (z * Qp[:, t]).sum(-1, keepdim=True))
    return torch.stack(outs, dim=1)

Q, K, V = torch.randn(2, 64, 32), torch.randn(2, 64, 32), torch.randn(2, 64, 32)
assert torch.allclose(linear_parallel(Q, K, V), linear_recurrent(Q, K, V), atol=1e-5)
```

```
同一状态方程的三种算图（T = 6、chunk 长 C = 3 示意）：

 parallel（式 (4)，短序列）       recurrent（式 (8)，decode 形态）    chunkwise（§8，训练主力）
 ┌───────────────────┐            t = 1→2→3→4→5→6 串行             ┌─────┐     ┌─────┐
 │ T×T 掩码核矩阵     │            每步 rank-1 写入                  │ C×C │ ──► │ C×C │ ──► …
 │ 两次大 matmul     │            状态 d_v×d_k 常驻                 │matmul│     │matmul│
 └───────────────────┘                                             └─────┘     └─────┘
 最大对象 T×T                  每 token O(d_v·d_k) FLOPs          块间只传 d_v×d_k 状态（T/C 步串行）
```

## 4. RetNet：指数衰减 $\gamma$ 与三种计算形式

式 (8) 的加性状态对第 1 个与第 $t-1$ 个 token 一视同仁——远期写入永不衰减。RetNet（[Sun et al., 2023](../references.md)，arXiv:2307.08621，[A]；实现锚点 microsoft/unilm）给 score 乘上每头标量衰减 $\gamma \in (0,1)$（固定、非输入依赖），核变为 $\gamma^{\thinspace t-s}\langle q_t, k_s\rangle$（ $\phi$ = 恒等、无 $z_t$ 分母，尺度由逐头 GroupNorm 承担）。同一函数的三种形态：

**parallel**（对应式 (4) 左序加掩码）：

$$
O = \left( QK^{\top} \odot D \right) V, \qquad D_{ts} = \begin{cases} \gamma^{\thinspace t-s} & t \ge s \\ 0 & t < s \end{cases} \qquad (9)
$$

**recurrent**（对应式 (8)，去掉分母）：

$$
S_t = \gamma S_{t-1} + v_t k_t^{\top}, \qquad o_t = S_t q_t \qquad (10)
$$

展开 $S_t = \sum_{s \le t} \gamma^{\thinspace t-s} v_s k_s^{\top}$ 即回式 (9) 第 $t$ 行——等价性仍是一行归纳。

**chunkwise**（训练形态；chunk 长 $C$，第 $i$ 块的局部矩阵 $Q_i, K_i, V_i \in \mathbb{R}^{C \times d}$）：块内 parallel、块间传状态。对块内第 $\tau$ 行把求和劈成「本块 / 既往块」，既往块提出因子 $\gamma^{\tau}$：

$$
O_i = \left( Q_iK_i^{\top} \odot D \right) V_i \hspace{5mu}+\hspace{5mu} \Gamma\thinspace Q_i R_{i-1}^{\top}, \qquad R_i = \gamma^{C} R_{i-1} + V_i^{\top}\thinspace \bar{\Gamma} K_i \qquad (11)
$$

**符号**： $R_{i-1} = S_{(i-1)C} = \sum_{s \le (i-1)C} \gamma^{\thinspace(i-1)C-s} v_s k_s^{\top} \in \mathbb{R}^{d_v \times d_k}$：块末状态； $\Gamma = \mathrm{diag}(\gamma^{1}, \ldots, \gamma^{C})$：行 $\tau$ 的跨块衰减； $\bar{\Gamma} = \mathrm{diag}(\gamma^{C-1}, \ldots, \gamma^{0})$：块内第 $\sigma$ 行的写入到块末还剩 $C - \sigma$ 步衰减。推导留作练习 2。串行步数从 $T$ 降到 $T/C$，块内全是 matmul——chunkwise 因此成为训练的唯一可行形态。

三形式是同一函数的三种算图（§3 命题的带衰减版）。RetNet 同时立起本章的泛化模板：衰减 $\gamma$（固定标量） $\to$ $\alpha_t$（输入依赖标量，§6 GDN） $\to$ $\mathrm{diag}(\alpha_t)$（逐通道，§7 KDA）；写入 $vk^{\top}$ $\to$ 带定点擦除的 delta rule（§5）。

$\gamma$ 的物理读法是时间常数：第 $s$ 个 token 的写入到第 $t$ 步只剩 $\gamma^{\thinspace t-s}$，半衰期 $n_{1/2} = \ln 2 \thinspace/\thinspace \ln(1/\gamma)$—— $\gamma$ = 0.99 对应约 69 步， $\gamma$ = 0.999 对应约 693 步。RetNet 给每个头配不同的固定 $\gamma$，一层之内并存多个时间尺度的记忆：近程头管局部句法，远程头管段落级依赖。与 §3 的无衰减加性状态对照， $\gamma$ 把「无限期平均累积」换成「指数遗忘窗口」：状态范数有界（几何级数收敛），数值稳定性由此而来；尺度校准交给逐头 GroupNorm 而不是 $z_t$ 分母。

## 5. DeltaNet：delta rule 是一步在线梯度下降

加性状态的病灶在写入只做加法：沿 $k_t$ 方向的旧内容原样残留，读出 $S_tk_t = v_t\lVert k_t \rVert^{2} +$ 历史干扰。要「定点修改」，把状态当作一个正在在线学习的线性映射：每步让 $S$ 拟合当前联想 $k_t \mapsto v_t$，平方损失

$$
\mathcal{L}_t(S) = \frac{1}{2} \lVert Sk_t - v_t \rVert^{2}, \qquad \nabla_S \mathcal{L}_t = \left( Sk_t - v_t \right) k_t^{\top} \qquad (12)
$$

（梯度一步： $d\mathcal{L} = (Sk-v)^{\top}(dS\thinspace k) = \mathrm{tr}\left( k(Sk-v)^{\top} dS \right)$。）从 $S_{t-1}$ 出发做一步 GD、步长 $\beta_t$：

$$
S_t = S_{t-1} - \beta_t \left( S_{t-1}k_t - v_t \right) k_t^{\top} = S_{t-1} \left( I - \beta_t k_t k_t^{\top} \right) + \beta_t v_t k_t^{\top} \qquad (13)
$$

这就是 DeltaNet 的状态更新（[Yang et al., 2024](../references.md)，arXiv:2406.06484，[A]）——经典 delta rule 的矩阵状态形式。右式把一步 GD 重读成「擦除 $+$ 写入」： $(I - \beta_t k_t k_t^{\top})$ 右乘 = 沿 $k_t$ 方向的定点擦除（rank-1 downdate）， $\beta_t v_t k_t^{\top}$ = 写入。两条性质（设 $\lVert k_t \rVert = 1$；实现中 $k_t$ 经 L2 归一化）：

- **一步精确写入**： $\beta_t = 1$ 时 $S_tk_t = S_{t-1}(k_t - k_t) + v_t = v_t$——当前联想写入即精确可读；
- **正交方向不动**： $u \perp k_t$ 时 $(I - \beta k_tk_t^{\top})u = u$ 且 $k_t^{\top}u = 0$ 消去写入项，故 $S_tu = S_{t-1}u$——擦除是定点的，与全局衰减正交。

一般 $\beta_t \in (0,1)$（输入依赖、可学）插值于「不改」与「覆写」之间。同一「对联想集做 GD」的结构在 3.3 §5 作为 ICL 解释 (a) 出现过：那里是 linear attention 层模拟一步 GD，这里是 GD 直接定义状态更新。DeltaNet 论文的另一半贡献是把这条递推并行化到序列维——§8 的 WY 表示。

为什么只做一步、不做全量最小二乘：对全部历史 $\lbrace(k_s, v_s)\rbrace_{s \le t}$ 的 ridge 解（3.3 §5 式 (5) 的那条路线）要解 $\sum_s k_sk_s^{\top} + \lambda I$ 的正规方程——从头重解每步 $O(td_k^{2} + d_k^{3})$（构造 $d_k \times d_k$ 矩阵再求逆），增量维护逆矩阵（Sherman-Morrison 秩一更新）每步 $O(d_k^{2})$；一步在线 GD 只要 $O(d_vd_k)$，且常数小、无矩阵求逆的数值脆弱性。更本质的差别在目标而不在成本：全量解同等拟合所有历史对（无遗忘、干扰随 $t$ 累积），一步 GD 只对当前对负责、旧对被渐进挤出——§10 的容量界正是以这种形式显形。残差 $v_t - S_{t-1}k_t$ 另有一层读法：它是当前状态对 $k_t$ 的**预测误差**，delta rule 只把预测错的部分写进状态——已经记住的旧内容不重复写入，这是对加性状态「重复累加」病灶的直接修复。

## 6. GDN 与 GDN-2：遗忘门与 erase/write 解耦

DeltaNet 会定点修改，但不会主动遗忘：旧内容只能被同方向的后来写入挤占。GDN（[Yang et al., 2024](../references.md)，arXiv:2412.06464，[A]；NVlabs/GatedDeltaNet）补一个输入依赖的全局遗忘门 $\alpha_t \in (0,1)$：

$$
S_t = \alpha_t\thinspace S_{t-1} \left( I - \beta_t k_t k_t^{\top} \right) + \beta_t v_t k_t^{\top} \qquad (14)
$$

逐项（约定同前：状态 $d_v \times d_k$、右乘擦除）：

- $\alpha_t$：**全局遗忘**——整个旧状态等比收缩，RetNet $\gamma$ 的输入依赖版（也是 Mamba-2 的门控形态）；
- $(I - \beta_t k_t k_t^{\top})$：**定点擦除**——只动 $k_t$ 方向（§5 的 DeltaNet 项）；
- $\beta_t v_t k_t^{\top}$：**写入**；
- 次序：擦除作用于旧状态、写入在其后相加——当前步写入不被自己的擦除与遗忘衰减， $o_t = S_tq_t$ 含完整的 $\beta_t(k_t^{\top}q_t)v_t$ 项。

退化链把 §4-5 收口： $\alpha_t = 1$ 回 DeltaNet； $\beta_t = 1$、 $\alpha_t = \gamma$ 是「RetNet $+$ 定点覆写」。注意缺口： $\beta_t = 0$ 时写入项同步归零， $S_t = \alpha_t S_{t-1}$ 只是纯衰减到零——耦合的 $\beta$ 表达不了「关擦除、留写入」，精确的 RetNet（ $\gamma$ 衰减 $+$ 无擦除写入）在式 (14) 里取不到，需要擦除与写入各自独立（式 (15) 的动机）。遗忘与擦除正交：一个管「多久以前的内容整体淡出」，一个管「这个 key 方向的内容精确替换」。

代入一组数值走一遍式 (14)： $\lVert k_t \rVert = 1$、 $\beta_t$ = 0.5、 $\alpha_t$ = 0.9，取读出方向 $q_t = k_t$——旧内容沿 $k_t$ 方向被缩到 $0.9 \times (1 - 0.5) = 0.45$ 倍，新写入项贡献 $0.5\thinspace v_t$；三个旋钮的乘法语义直接可见。

数值读法： $\alpha_t$ = 0.99 时遗忘窗口约百步（§4 的半衰期账）， $\alpha_t \to 1$ 时退为「只定点修改、不整体遗忘」； $\beta_t$ 控制写入侵略性。两个门都由当前输入算出（输入依赖是标题「Gated」的所指），模型于是能按内容选择「记什么、忘多快」——相对 RetNet 的固定 $\gamma$，这是从「设计好的时间尺度」到「学出来的时间尺度」的一步。

GDN-2（[Hatamizadeh et al., 2026](../references.md)，arXiv:2605.22791，[A]；NVlabs/GatedDeltaNet-2）针对式 (14)(16) 中 $\beta_t$ 的**双重身份**：同一标量同时控制 key 侧擦除强度与 value 侧写入强度。论文的观察是这两个角色作用在状态的不同轴上——擦除是 key 侧操作（决定旧读出的哪些坐标被移除），写入是 value 侧操作（决定新 value 的哪些坐标被提交）——于是一并继承 KDA 的逐通道衰减、把两个角色拆成两个逐通道门（Gated Delta Rule-2，论文式 (10) 转置到本书约定布局）：

$$
S_t = S_{t-1}\thinspace \mathrm{diag}(\alpha_t) \left( I - (b_t \odot k_t)\thinspace k_t^{\top} \right) + (w_t \odot v_t)\thinspace k_t^{\top} \qquad (15)
$$

**符号**： $b_t \in [0,1]^{d_k}$：擦除门（key 侧逐 channel）； $w_t \in [0,1]^{d_v}$：写入门（value 侧逐 channel）； $\mathrm{diag}(\alpha_t)$：KDA 式逐通道衰减（key 侧，见 §7）。等价的残差写法：

$$
S_t = \underbrace{S_{t-1}\thinspace \mathrm{diag}(\alpha_t)}_{\text{decay}} + \big( \underbrace{w_t \odot v_t}_{\text{write}} - \underbrace{S_{t-1}\thinspace \mathrm{diag}(\alpha_t)\thinspace (b_t \odot k_t)}_{\text{gated read}} \big)\thinspace k_t^{\top}
$$

与式 (13) 对照：读出方向的 $k_t$ 换成门控后的 $b_t \odot k_t$（擦哪些 key 坐标可选），被写入的 $v_t$ 换成 $w_t \odot v_t$（提交哪些 value 坐标可选），写入方向的 $k_t$ 保持不动（保住 delta rule 的寻址结构）。

退化链（逐档可验）： $b_t = \beta_t \mathbf{1}$、 $w_t = \beta_t \mathbf{1}$ 精确回 KDA 式 (16)；再令 $\alpha_t = \alpha_t \mathbf{1}$ 回 GDN 式 (14)； $b_t = 0$、 $w_t = \mathbf{1}$、 $\alpha_t = \gamma \mathbf{1}$ 得精确 RetNet 式 (10)——最后一档在耦合形式里不可表达（GDN 的 $\beta = 0$ 会同时关掉写入），正是解耦的动机。论文另支持负特征值变体（擦除门放宽到 $[0,2]^{d_k}$）；chunkwise 实现把逐通道衰减吸收进非对称 erase 因子，保持与 KDA 同款的 WY 结构（§8）。实证口径（1.3B 参数、100B FineWeb-Edu tokens）：对 Mamba-2 / GDN / KDA / Mamba-3 变体总体占优，优势在 multi-key 检索任务上最大。

另注意式 (14)(15) 的读出都是 $o_t = S_tq_t$、无式 (8) 的 $z_t$ 分母：kernel 形式里分母防权重和爆炸，门控形式里这一职责改由谱性质承担—— $\lVert k_t \rVert = 1$、 $\beta \in (0,1)$ 时 $(I - \beta k_tk_t^{\top})$ 的特征值落在 $(0,1]$， $\alpha_t < 1$ 再加一重收缩，状态范数有界；尺度校准交给输出侧的归一化层。

## 7. KDA：逐通道遗忘（对角门）

GDN 的 $\alpha_t$ 是标量：整层状态同一遗忘速率。KDA（Kimi Delta Attention；[Kimi Team, 2025](../references.md)，arXiv:2510.26692，[A]；MoonshotAI/Kimi-Linear）把它换成 channel-wise 对角门 $\mathrm{diag}(\alpha_t)$， $\alpha_t \in (0,1)^{d_k}$：

$$
S_t = S_{t-1}\thinspace \mathrm{diag}(\alpha_t) \left( I - \beta_t k_t k_t^{\top} \right) + \beta_t v_t k_t^{\top} \qquad (16)
$$

（论文原式布局为 $S \in \mathbb{R}^{d_k \times d_v}$、更新写作 $(I - \beta_t k_tk_t^{\top})\thinspace \mathrm{Diag}(\alpha_t)\thinspace S_{t-1} + \beta_t k_tv_t^{\top}$，转置到本书约定布局即式 (16)；衰减门的 channel 轴是 key 侧——论文 chunkwise 记号的 $\Gamma \in \mathbb{R}^{C \times d_k}$ 可证。）逐列（逐 key channel）读： $`(S_t)_{:,j} = \alpha_{t,j}\thinspace (S_{t-1})_{:,j} + \beta_t\thinspace k_{t,j} \left( v_t - S_{t-1}\thinspace \mathrm{diag}(\alpha_t)\thinspace k_t \right)`$——状态第 $j$ 列（key 坐标 $j$ 的记忆迹）有自己的遗忘速率 $\alpha_{t,j}$，decay 项逐列干净分离；erase 与 write 则合成一个各列共享的 rank-1 修正向量 $\left( v_t - S_{t-1}\mathrm{diag}(\alpha_t)k_t \right)$（即在衰减后状态上求值的 delta-rule 残差），按 $k_{t,j}$ 逐列缩放到各列。各列遗忘速率由输入决定：一部分列配到近 1（长记忆槽位），一部分近 0（即写即弃）。标量 $\alpha$ 是 $\mathrm{diag}(\alpha \cdot \mathbf{1})$ 的特例，KDA 的遗忘结构严格包含 GDN。逐通道门的意义在时间尺度的配额： $d_k$ 个 key channel 各自独立衰减，一层之内可同时容纳「 $\alpha \approx 1$ 的长期档案槽」与「 $\alpha \approx 0$ 的即抛工作区」，且配额随输入动态划分；GDN 的标量门在每个时刻只能给全层选一个时间尺度。

结构观察：转移 $X \mapsto X\thinspace \mathrm{diag}(\alpha_t) \left( I - \beta_t k_t k_t^{\top} \right)$ 是「key 侧对角缩放 $+$ 秩一修正」的复合——DPLR（diagonal-plus-low-rank）形态；KDA 的 chunkwise kernel 用 DPLR 转移矩阵的特化变体（摘要口径）。同一「逐通道输入依赖衰减」在 §9 的 S6 里以对角 $A$ 出现：两谱系的状态方程是同一族对象，差别在 key 方向的定点擦除。与 Mamba-2 的关系一并登记：KDA 可看作把 Mamba-2 式的标量头衰减细化为逐 channel、再叠加 DeltaNet 的定点擦除——§9 式 (19) 的 $`\bar{a}_{t,i}`$ 与本节 $\alpha_{t,j}$ 是同一对象在两个谱系里的名字。Kimi Linear 的账目数字（KV cache 最多 $`-75\%`$、1M ctx decode 吞吐摘要口径 up to 6×、精确值 6.3× 见论文 Fig 1(b)）属于其 3:1 hybrid 形态，细账在 4.4。

## 8. Chunkwise 并行：WY 表示与 kernel 锚点

delta-rule 族的递推含连乘 $\prod_t (I - \beta_t k_t k_t^{\top})$，没有式 (9) 那种封闭 parallel 形；纯 recurrent 训练又在 $T$ 维串行、matmul 利用率坍塌。DeltaNet 论文的并行化借自数值线性代数的 WY 表示（blocked Householder 的标准工具）： $C$ 个秩一 downdate 的连乘恰是一次秩-$C$ 修正——

$$
\prod_{t=1}^{C} \left( I - \beta_t k_t k_t^{\top} \right) = I - K W K^{\top}, \qquad K = \begin{bmatrix} k_1 & \cdots & k_C \end{bmatrix} \in \mathbb{R}^{d_k \times C},\quad W \in \mathbb{R}^{C \times C} \ \text{triangular} \qquad (17)
$$

$C = 2$ 展开验证（系数对位）： $(I-\beta_1k_1k_1^{\top})(I-\beta_2k_2k_2^{\top}) = I - \beta_1k_1k_1^{\top} - \beta_2k_2k_2^{\top} + \beta_1\beta_2(k_1^{\top}k_2)\thinspace k_1k_2^{\top}$，与 $I - \sum_{ij}W_{ij}k_ik_j^{\top}$ 对系数得 $W_{11} = \beta_1$、 $W_{22} = \beta_2$、 $W_{12} = -\beta_1\beta_2\thinspace k_1^{\top}k_2$、 $W_{21} = 0$；一般 $C$ 逐列递推：设前 $j-1$ 列已定，第 $j$ 列由

$$
W_{jj} = \beta_j, \qquad W_{1:j-1,\thinspace j} = -\beta_j\thinspace W_{1:j-1,\thinspace 1:j-1} \left( K_{1:j-1}^{\top} k_j \right)
$$

回代得出（与 blocked Householder 的向前消元同构；推导留作练习 2(b)）。每列一次 $K^{\top}k_j$（ $O(Cd_k)$）加一次三角 matvec（ $O(C^{2})$），构造 $W$ 总代价 $O(C^{2}d_k)$，相对块内主项不涨价。结论先行记住：**擦除连乘可预先压缩成一次秩-$C$ 修正，串行由此变块并行**；反向传播版本与完整推导见 DeltaNet 论文。

有了 $W$，chunk 内全部工作 matmul 化：块末状态 $S_i = S_{i-1}(I - KWK^{\top}) +$（块内写入项）是 $d_v \times d_k$ 级 matmul；块内输出 = 块初状态对 $C$ 个 query 的读出 $+$ 块内修正（ $C \times C$ 级 matmul）。每块 $O\left( C^{2}(d_k + d_v) + Cd_kd_v \right)$，全序列对 $T$ 线性； $C$ 典型取 64 档。GDN/KDA 把同一骨架套在式 (14)(16) 上（ $\alpha$ 折进块内衰减因子；KDA 走 §7 的 DPLR 特化）。

反向一侧的账顺带记一笔：chunkwise 的反向与正向同构——对偶的 WY 形式把梯度同样写成块内 matmul 加状态传递；工程实现再配重算策略（块内中间量不物化、反向时按同式重算），激活内存压到块界状态级别。这与 4.1 FlashAttention「不物化大矩阵、反向重算」是同一套内存哲学，对象从 $T \times T$ scores 换成块内修正。

工程锚点：fla-org/flash-linear-attention（[fla-org](../references.md)，[A]）实现 DeltaNet/GDN/KDA 的 chunkwise kernel（`chunk_gated_delta_rule` 等），transformers 的 Qwen3-Next 建模代码直接调该函数（W2 调研读码确认）。**布局注记**：本书公式约定 $S \in \mathbb{R}^{d_v \times d_k}$、右乘擦除；HF 侧 kernel 把状态存成转置 $(d_k, d_v)$、左乘写入——两套约定互为转置，读码勿混。

## 9. Mamba/S6：连续系统的离散化与 selectivity

linear attention 谱系从「改 softmax」出发；SSM 谱系从连续动力学出发：

$$
h'(t) = Ah(t) + Bx(t), \qquad y(t) = Ch(t) \qquad (18)
$$

**符号**： $x(t) \in \mathbb{R}$：输入（逐 channel）； $h \in \mathbb{R}^{N}$：状态； $A \in \mathbb{R}^{N \times N}$ 取对角（S4D 谱系）； $B, C$：输入 / 读出向量；另有 $Dx$ 旁路（略）。

数字实现要离散化。取步长 $\Delta$、区间内 $x$ 恒定（零阶保持，zero-order hold / ZOH），解式 (18)：

$$
h(t+\Delta) = e^{\Delta A} h(t) + \left( \int_0^{\Delta} e^{\tau A} d\tau \right) B\thinspace x(t) = \bar{A}\thinspace h(t) + \bar{B}\thinspace x(t), \qquad \bar{A} = e^{\Delta A},\quad \bar{B} = A^{-1}\left( e^{\Delta A} - I \right) B \approx \Delta B \qquad (19)
$$

积分 $\int_0^{\Delta} e^{\tau A}d\tau = A^{-1}(e^{\Delta A} - I)$ 逐项成立；末步一阶展开 $e^{\Delta A} \approx I + \Delta A$ 得 $\bar{B} \approx \Delta B$——S6 论文口径即 $\bar{A} = \exp(\Delta A)$、 $\bar{B} \approx \Delta B$。对角 $A$ 下 $\exp$ 逐元素，离散化代价 $O(N)$； $\bar{A}$ 的对角元 $e^{\Delta a_i}$ 在 $a_i < 0$ 时落在 $(0,1)$，即逐通道衰减因子。

S4 一代的 $(\Delta, A, B, C)$ 全序列共享——LTI 系统，整序列等价于一个卷积核 $\bar{K} = (C\bar{B},\thinspace C\bar{A}\bar{B},\thinspace \ldots)$，FFT 并行训练。LTI 的天花板：对所有输入同一动力学，不能按内容决定记什么、忘什么。Mamba 的 S6（[Gu & Dao, 2023](../references.md)，arXiv:2312.00752，[A]；state-spaces/mamba）把参数输入化——selectivity（选择性）：

$$
\Delta_t = \mathrm{softplus}\left( w_{\Delta}^{\top} x_t + b_{\Delta} \right), \qquad B_t = W_B x_t, \qquad C_t = W_C x_t \qquad (20)
$$

**符号**： $\Delta_t$：逐 channel 标量步长（式 (20) 为示意标量形式，省略论文的 low-rank broadcast 细节——实际由一个标量投影广播到各 channel、加偏置后过 softplus）； $W_B, W_C$：输入到 $B$ / $C$ 的线性读出。后果链：时变系统 $\Rightarrow$ 卷积核不存在（滤波器系数依赖输入本身） $\Rightarrow$ 训练走并行 scan。scan 的账：递推 $`h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t`$ 总功 $O(TN)$、关键路径 $O(T)$，用 Blelloch 式前缀扫描把深度压到 $O(\log T)$——相邻二元合成 $(A_2, b_2) \circ (A_1, b_1) = (A_2A_1,\ A_2b_1 + b_2)$ 可结合，这是「线性递推皆可扫描」的代数根据；Mamba 的官方 kernel 再把状态常驻 SRAM、避免物化 $[T, N]$ 中间量（state-spaces/mamba）。与 §8 对照：同一类递推的两种并行化——scan 沿时间维做结合律树，chunkwise 沿块做矩阵化。逐 channel 递推 $`h_{t,i} = \bar{a}_{t,i} h_{t-1,i} + b_{t,i} x_{t,i}`$—— $`\bar{a}_{t,i} \in (0,1)`$ 正是 KDA $\alpha_{t,j}$ 的同款对象（§7）。 $\Delta_t$ 的角色值得单独读：由式 (19)， $`\bar{a}_{t,i} = e^{\Delta_{t,i} a_i}`$ 随 $\Delta$ 单调降—— $\Delta_t$ 大则旧状态快衰减、新输入权重大（关注当前）， $\Delta_t$ 小则状态近似冻结（忽略当前输入）。selectivity 因此等价于「逐 channel、逐时刻的读写门」，与 §6-7 的 $(\alpha_t, \beta_t)$ 同构；差别在 S6 没有 key 方向的擦除结构， $B_tx_t$ 的写入在各 channel 内独立。把 S6 的衰减接回 §4 的时间常数账： $`\ln \bar{a}_{t,i} = \Delta_{t,i}\thinspace a_i`$，半衰期 $n_{1/2} = \ln 2 \thinspace/\thinspace (\Delta_t |a_i|)$—— $\Delta_t$ 就是 SSM 侧的可学时间常数旋钮，与 RetNet 的 $\gamma$、GDN/KDA 的 $\alpha$ 算的是同一份账，三家只是把这个旋钮放在不同的轴上（标量 / 逐 channel）与不同的参数化里（固定 / 输入依赖）。Mamba-2 的 SSD 框架（[Dao & Gu, 2024](../references.md)，arXiv:2405.21060，[A]）把这座桥搭死：带衰减的 linear attention 核 $\gamma^{\thinspace t-s}\langle q_t, k_s\rangle = (\gamma^{t} q_t)^{\top}(\gamma^{-s} k_s)$ 恰是半可分（semiseparable）矩阵 $M_{ts} = C_t^{\top} A_{t:s} B_s$ 的标量态特例——SSM 与 linear attention 是同一线性递推的两种参数化，训练算法（scan 与 chunkwise matmul）也因此可以互译。

## 10. 对照表与理论边界：固定状态的 associative recall 上界

| 维度 | softmax attention（2.3/2.4） | kernel linear attention（§1-3） | delta-rule 族（§5-7） | S6 / Mamba（§9） |
| --- | --- | --- | --- | --- |
| 状态 | KV cache，随 $T$ 线性 | $S \in \mathbb{R}^{d_v \times d_r}$、 $z$，定长 | $S \in \mathbb{R}^{d_v \times d_k}$，定长 | $h \in \mathbb{R}^{N}$ / channel，定长 |
| 写入 | append $k_t, v_t$ | 秩一累加 $v_t\phi(k_t)^{\top}$ | 擦除 $+$ 写入 | 输入调制 $b_t x_t$ |
| 遗忘 | 无（全量驻留） | 无；RetNet 加 $\gamma$ | $\alpha_t$（标量 $\to$ 对角） | $e^{\Delta_t a_i}$（逐通道） |
| 读出 | $\mathrm{softmax}(qK^{\top}/\sqrt{d_k})V$ | $S\phi(q) \thinspace/\thinspace (z^{\top}\phi(q))$ | $Sq_t$ | $\langle c_t, h_t \rangle$ |
| 训练算图 | 并行 matmul | 结合律重排 / chunkwise | chunkwise（WY） | 并行 scan / SSD |
| decode 每 token | FLOPs $O(Td)$、读 cache $O(Td)$ | $O(d^{2})$ | $O(d^{2})$ | $O(dN)$ |
| 精确回忆 | 无结构上限 | 受 $d_r$ 限 | 受 $d_k$ 限 | 受 $N$ 限 |

上界的形式化——联想回忆（associative recall）：写入 $n$ 对 $(k_s, v_s)$，给 $k_i$ 须读出 $v_i$。

**命题（固定状态的精确回忆上界）**。给定 $k_1, \ldots, k_n \in \mathbb{R}^{d_k}$。以下两条等价：(i) 对任意 $v_1, \ldots, v_n \in \mathbb{R}^{d_v}$，存在 $S \in \mathbb{R}^{d_v \times d_k}$ 使 $Sk_i = v_i$ 对所有 $i$ 成立；(ii) $\lbrace k_i\rbrace$ 线性无关。推论：单头可精确回忆的任意联想对数 $\le d_k$；等价地， $n > d_k$ 时存在不可表示的值组合——精确回忆的误差下界严格非零。

**证明**。(ii) $\Rightarrow$ (i)： $K = [k_1 \cdots k_n]$ 列满秩时 $K^{\top}K$ 可逆，取 $S = V(K^{\top}K)^{-1}K^{\top}$，则 $SK = V$。(i) $\Rightarrow$ (ii)：若 $\sum_i c_i k_i = 0$ 有非零系数，则任何 $S$ 的读出满足 $\sum_i c_i (Sk_i) = 0$；取 $v_j = u \neq 0$（ $j$ 为 $c_j \neq 0$ 的任一下标）、其余 $v_i = 0$，则 $\sum_i c_i v_i = c_j u \neq 0$，矛盾。 $\square$

把这条界放进更大的图景：KV cache 是「数据库」极端——逐对原样存储、查询零失真、内存随 $n$ 付账；linear state 是「充分统计量」极端——把任意长前缀压成固定维矩阵、内存一口价。统计学里的经典结论（Pitman–Koopman 型）是：固定维的充分统计量只对受限分布族存在；任意键值联想的精确回忆恰是非参数极端，不存在固定维无损压缩。上面的命题把这句直觉落成矩阵状态下的秩条件—— $d_k$ 就是这份「充分统计量」能承载的独立联想数。

读法：上界 $d_k$ 与已写 token 数无关——第 $10^{6}$ 步写入的内容与第 1 步竞争同一份 $d_v \times d_k$。KV cache 无此界：每对 $(k_s, v_s)$ 原样驻留、内存随 $n$ 涨（2.4 式 (9)）——「固定状态 vs 增长 cache」的得失全从这条 rank 条件长出。

代入刻度： $d_k$ = 128 的单头，精确联想预算 128 对； $H$ 头并列为 $Hd_k$，但头间不互济——每个头仍各自受这条 rank 界约束。对照同层 KV cache 在 $t$ = 131072 时驻留 $t$ 对原样键值。注意界的前提是「任意 $v$」：自然数据的 $v$ 住在低维结构里，实际可用容量高于 worst case，但随 $n$ 的标度不变—— $n \gg d_k$ 后误差必然显形，差别只在什么任务上先显形。

近似版（加性状态的干扰账）：读出 $`\hat{o}_i = Sk_i = v_i\lVert k_i \rVert^{2} + \sum_{j \ne i} v_j\thinspace k_j^{\top}k_i`$；单位球面随机键的交叉内积 $\sim \mathcal{N}(0, 1/d_k)$，干扰能量 $\sim (n-1)/d_k$，信噪比 $\sim d_k / (n-1)$——可靠读出要求 $n = O(d_k)$，与 rank 界同刻度。代入数字： $d_k$ = 128、 $n$ = 1024 时 SNR $\approx 128/1023 \approx 0.125$（约 $-9$ dB），精确回忆早已失效； $n$ 回到 128 档才回到 0 dB 附近。擦除（§5）削「同方向残留」、遗忘（§6-7）削「远期累积」、衰减另加有效窗口 $\sim 1/(1-\gamma)$（ $\gamma^{n} = e^{-1}$ 解出 $n \approx 1/(1-\gamma)$）。用 rank 语言重述擦除的作用：不擦除时每次写入永久占用一份 residual 容量，定点擦除把 $k_t$ 方向上的旧占用清空再写入——容量可复用，但不扩容。三件工具都在与这条界周旋，都不能取消它。

这条界不说的东西同样要写清：它约束的是**任意值的精确回忆**；语言建模所需的长程统计多为低秩、可压缩结构，fixed-state 模型在 PPL 类指标上逼近 full attention 与这条界不冲突——冲突集中在逐 bit 复制与精确检索的任务族。界的另一面是 cache 的代价：精确回忆能力用随 $T$ 线性增长的内存换来，服务侧这笔账 2.4 已结。

实证同向：RetNet / GDN / KDA 论文的检索类 ablation 一致报告纯 linear 形态在精确回忆与复制任务上落后 full attention。工程回答是 hybrid——少数 full-attention 层保精确检索、linear 层扛吞吐与定长状态（4.4 的主题）；Kimi Linear 摘要「公平对照下全面超过 full attention」的结果正是在 3:1 KDA : full 交错形态下达成的——边界没有被打破，是被分层绕开。

## 总结

| 内容 | 公式 | 要点 |
| --- | --- | --- |
| kernel 替换 | 式 (2)： $\mathrm{sim} \approx \phi(q)^{\top}\phi(k)$ | 非负是唯一要求； $\phi = \mathrm{elu} + 1$ |
| 结合律重排 | 式 (4) 与 FLOPs 表 | causal $2T^{2}d \to 4Td^{2}$，交叉点 $T = 2d$ |
| FAVOR+ | 式 (5)(6) | 高斯 MGF ⇒ softmax 核无偏 MC； $O(1/\sqrt{m})$ 误差、非精确 |
| RNN 等价 | 式 (7)(8) 命题 | 和式归纳 $+$ 差分；训练并行 / decode 递推同一函数 |
| RetNet | 式 (9)(10)(11) | $\gamma$ 固定标量；chunkwise 是训练形态；泛化模板 |
| DeltaNet | 式 (12)(13) | $\min_S \lVert Sk_t - v_t \rVert^{2}$ 一步 GD； $\beta = 1$ 精确写入、正交不动 |
| GDN | 式 (14) | $\alpha$ 遗忘 $+$ $(I - \beta kk^{\top})$ 擦除 $+$ $vk^{\top}$ 写入 |
| GDN-2 | 式 (15) | channel-wise erase $b_t$（key 侧）/ write $w_t$（value 侧）解耦；回退链 KDA → GDN → RetNet |
| KDA | 式 (16) | $\mathrm{diag}(\alpha_t)$ key 侧逐通道遗忘（ $\alpha_t \in \mathbb{R}^{d_k}$）；DPLR 转移 |
| chunkwise | 式 (17) | WY： $C$ 个秩一 downdate $=$ 秩-$C$ 修正；fla kernel 锚点 |
| S6 | 式 (18)(19)(20) | ZOH： $\bar{A} = e^{\Delta A}$、 $\bar{B} \approx \Delta B$；selectivity 使卷积形失效 |
| SSD 对偶 | $\gamma^{\thinspace t-s}\langle q_t, k_s\rangle$ 半可分 | SSM $=$ linear attention 的另一参数化 |
| 回忆上界 | $SK = V$ 命题 | 精确联想 $\le d_k$ 对 / 头；SNR $\sim d_k/(n-1)$；hybrid 动机（4.4） |

## 练习

1.（推导）(a) 补全式 (5)：写出 $\omega \sim \mathcal{N}(0, I)$ 的矩母函数推导 $\mathbb{E}[e^{\omega^{\top}u}] = e^{\lVert u \rVert^{2}/2}$，并完成 $\mathbb{E}[\phi(q)^{\top}\phi(k)] = e^{q^{\top}k}$ 的最后一步。(b) 证明式 (6) 估计量的方差为 $O(1/m)$（提示： $\phi(q)^{\top}\phi(k) = \frac{1}{m}\sum_i (\cdot)$ 各项 iid，先算 $\mathbb{E}[e^{2\omega^{\top}x}]$）。(c) 说明 trigonometric 特征为何让式 (8) 的分母 $z_t^{\top}\phi(q_t)$ 有近零风险，而式 (6) 的 positive 特征没有。
2.（推导）(a) 补全式 (11)：从 recurrent 形 $S_t = \gamma S_{t-1} + v_tk_t^{\top}$ 出发，对块内第 $\tau$ 行把求和劈成「本块 $+$ 既往块」，验证跨块项系数恰为 $\gamma^{\tau}$、块末状态更新中第 $\sigma$ 行写入的系数为 $\gamma^{C-\sigma}$。(b) 验证 §8 的 WY 递推：设 $\prod_{t=1}^{j-1} (I - \beta_t k_tk_t^{\top}) = I - K_{1:j-1} W_{1:j-1,\thinspace1:j-1} K_{1:j-1}^{\top}$，证明加入第 $j$ 个因子后 $W_{jj} = \beta_j$、 $W_{1:j-1,\thinspace j} = -\beta_j\thinspace W_{1:j-1,\thinspace1:j-1} \left( K_{1:j-1}^{\top} k_j \right)$，并对 $C = 2$ 核对与 §8 展开式一致。
3.（推导）GDN 代数三问：(a) 证明 $\lVert k_t \rVert = 1$、 $\beta_t = 1$、 $\alpha_t = 1$ 时 $S_tk_t = v_t$；(b) 证明 $u \perp k_t$ 时 $S_tu = \alpha_t S_{t-1}u$（正交内容只被 $\alpha$ 缩放）；(c) 对一般 $\alpha, \beta$ 计算 $S_tk_t$，指出 $\beta_t$ 在结果中出现两次（擦除项与写入项），由此用代数语言说明 GDN-2 式 (15) 多出的自由度。
4.（计算 $+$ 代码）(a) 取 $H$ = 32、 $d_k = d_v$ = 128、bf16：算 linear 层每层状态字节数，与 $t$ = 131072 时同头数 KV cache 每层字节数（2.4 式 (9) 口径），给比值并说明它随 $t$ 的标度。(b) 跑通 §3 教学实现的等价性断言；把 `phi` 换成式 (6) 的 FAVOR+ 特征（ $m$ = 256），对照 softmax attention 输出，报告逐位置最大误差随 $m \in \lbrace64, 256, 1024\rbrace$ 的收缩速率。(c) 实现式 (13) 的 DeltaNet 递推（ $k$ 归一化），数值验证 §5 的「一步精确写入」与「正交不动」。

## 参考

- [Katharopoulos et al., 2020](../references.md)：Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention，arXiv:2006.16236，[A]（key: katharopoulos）——kernel 视角、结合律重排、式 (8) 等价性
- [Choromanski et al., 2021](../references.md)：Rethinking Attention with Performers，arXiv:2009.14794，[A]（key: performer）——FAVOR+ 随机特征与方差分析
- [Sun et al., 2023](../references.md)：Retentive Network: A Successor to Transformer for Large Language Models，arXiv:2307.08621，[A]（key: retnet）—— $\gamma$ 衰减与 parallel / recurrent / chunkwise 三形式；实现锚点 microsoft/unilm
- [Yang et al., 2024](../references.md)：Parallelizing Linear Transformers with the Delta Rule over Sequence Length，arXiv:2406.06484，[A]（key: deltanet）——delta rule 递推与其 WY 并行化
- [Yang et al., 2024](../references.md)：Gated Delta Networks: Improving Mamba2 with Delta Rule，arXiv:2412.06464，[A]（key: gdn）——式 (14)；NVlabs/GatedDeltaNet
- [Hatamizadeh et al., 2026](../references.md)：Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention，arXiv:2605.22791，[A]（key: gdn2）——erase/write 解耦式 (15)；NVlabs/GatedDeltaNet-2
- [Kimi Team, 2025](../references.md)：Kimi Linear: An Expressive, Efficient Attention Architecture，arXiv:2510.26692，[A]（key: kimi-linear）——KDA 式 (16)、DPLR chunkwise、cache / 吞吐数字；MoonshotAI/Kimi-Linear
- [Gu & Dao, 2023](../references.md)：Mamba: Linear-Time Sequence Modeling with Selective State Spaces，arXiv:2312.00752，[A]（key: mamba）——S6：ZOH 离散化与 selectivity；state-spaces/mamba
- [Dao & Gu, 2024](../references.md)：Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality，arXiv:2405.21060，[A]（key: mamba2）——SSD 对偶框架
- [fla-org](../references.md)：flash-linear-attention（fla）官方库，[A]（key: fla-repo）——`chunk_gated_delta_rule` 等 chunkwise kernel 走读锚点
