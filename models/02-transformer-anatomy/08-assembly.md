# 2.8 组装完整 Decoder Block（Assembling the Full Decoder Block）

> 本章把 2.1–2.7 拆开的组件装回整机：现代标准配方的完整 block 公式（全下标，GQA + RoPE + SwiGLU）、Llama-3.1-8B config.json 逐字段 ↔ 公式映射表、约 150 行 minimal decoder-only PyTorch、transformers `modeling_llama.py` 官方实现走读、单层单 token FLOPs 全算与参数量精确到个位的对账、并行块（parallel block，PaLM / GPT-J / Falcon）的公式与梯度/吞吐含义、四条常见误解 | 前置：2.1–2.7 全部（公式编号沿用各章），1.1（$q_\theta(\cdot | x_{<t})$），1.2（$2N$ 口径与 $N$ 的两种口径约定）| 锚点：Llama-3.1-8B config [A]（主）、OLMo-2-7B config [A]（对照）、nanoGPT（教学参照）；Vaswani et al. [A]、Grattafiori et al. [A]、Team OLMo et al. [A]

## 1. 整机公式：全下标的标准配方

2.1 式 (1) 用 $\mathrm{Attn}_{\ell}$ 与 $\mathrm{FFN}_{\ell}$ 两个黑箱搭起了 residual stream 递推；本节把黑箱打开，按 Llama-3.1-8B 的标准配方（RMSNorm + Pre-LN + GQA + RoPE + SwiGLU，全无 bias）写出单个 block 的完整形式。取层号 $\ell \in \{1, \ldots, L\}$，输入 $x^{(\ell-1)} \in \mathbb{R}^{B \times T \times d}$：

$$
\begin{aligned}
u^{(\ell)} &= \mathrm{RMSNorm}_1^{(\ell)}\left( x^{(\ell-1)} \right) \\
\tilde{x}^{(\ell)} &= x^{(\ell-1)} + \mathrm{Attn}^{(\ell)}\left( u^{(\ell)} \right) \\
w^{(\ell)} &= \mathrm{RMSNorm}_2^{(\ell)}\left( \tilde{x}^{(\ell)} \right) \\
x^{(\ell)} &= \tilde{x}^{(\ell)} + \mathrm{FFN}^{(\ell)}\left( w^{(\ell)} \right)
\end{aligned} \qquad (1)
$$

**符号**：$x^{(\ell)}$：第 $\ell$ 层输出的 residual stream；$\tilde{x}^{(\ell)}$：attention 子层之后的中间 stream；$u^{(\ell)}, w^{(\ell)}$：两次 RMSNorm（2.7 式 (5)，各有可学增益 $\gamma_1^{(\ell)}, \gamma_2^{(\ell)} \in \mathbb{R}^{d}$，无 $\beta$）的输出。两个子层的统一形态即「读出归一化后的 stream、把结果加回 stream」。

attention 子层内部（GQA + RoPE + causal mask，逐位置逐 head 全下标）：

$$
\begin{aligned}
q_{t,h} &= R_t \left( u_t^{(\ell)} W_h^{Q} \right), \qquad h = 1, \ldots, H_q \\
k_{s,g} &= R_s \left( u_s^{(\ell)} W_g^{K} \right), \qquad v_{s,g} = u_s^{(\ell)} W_g^{V}, \qquad g = 1, \ldots, H_{kv} \\
\alpha_{t,s,h} &= \frac{\exp\left( \langle q_{t,h},\, k_{s,\, g(h)} \rangle / \sqrt{D} \right)}{\sum_{j=1}^{t} \exp\left( \langle q_{t,h},\, k_{j,\, g(h)} \rangle / \sqrt{D} \right)}, \qquad o_{t,h} = \sum_{s=1}^{t} \alpha_{t,s,h} \, v_{s,\, g(h)} \\
\mathrm{Attn}^{(\ell)}(u)_t &= \mathrm{concat}\left( o_{t,1}, \ldots, o_{t,H_q} \right) W^{O}
\end{aligned} \qquad (2)
$$

**符号**：$u_t^{(\ell)} \in \mathbb{R}^{d}$：层 $\ell$ 位置 $t$ 的归一化后 stream（行向量约定，同 2.3）；$W_h^{Q} \in \mathbb{R}^{d \times D}$：第 $h$ 个 Q head 的投影（实现中全部 $W_h^{Q}$ 拼成一块 $d \times H_q D$ 的 `q_proj`，按列切片即得，见 §4）；$W_g^{K}, W_g^{V} \in \mathbb{R}^{d \times D}$：第 $g$ 个 KV head 的投影；$W^{O} \in \mathbb{R}^{H_q D \times d}$：输出投影；$D$：head 维；$g(h) = \operatorname{div}\left( h - 1,\ H_q / H_{kv} \right) + 1$：Q head 到 KV head 的分组映射（$\operatorname{div}$ 为整除；Llama-3.1-8B 下每 4 个连续 Q head 共享一对 K/V）；$R_t$：2.5 式 (12) 的块对角旋转矩阵，频率 schedule $\theta_j = base^{-2j/D}$（2.5 式 (16)），只作用于 q、k 不作用于 v（2.5 §4.3）；求和上限 $t$ 即 causal mask 的可见集形式（2.4 式 (2)(3)）；$1/\sqrt{D}$ 即 2.3 式 (4) 的缩放。

读式 (2) 按三个顺序要点。其一，**先投影后旋转**：RoPE 作用于投影后的 head 向量（$R_t (u_t W_h^{Q})$），不作用于 stream——位置信息全程不进入 $x^{(\ell)}$ 本身，这是 2.5 §1「attention 内注入」的公式形态，也是 2.1 §2「$x^{(0)}$ 只含内容向量」在整机公式里的落实。其二，**分组映射只出现在 k/v 侧**：$g(h)$ 让 8 个 KV 头服务 32 个 Q 头，但 score 的数量与输出头数仍按 $H_q$ 计——GQA 削减的是 K/V 投影宽度与 cache（2.4 §5 的 4 倍账），不削减打分与加权和的计算（1.2 §6 已按此口径记账）。其三，**求和上限 $t$ 就是 mask**：式 (2) 的逐位置写法与 2.4 式 (7) 是同一对象的两种记法；矩阵形态（2.3 式 (4) + 2.4 式 (1) 的加性 $M$）是它按行堆叠的样子，两种形态在 §3 / §4 的代码里各出现一次。

FFN 子层（SwiGLU 三矩阵，2.6 式 (6)）：

$$
\mathrm{FFN}^{(\ell)}(w)_t = \left( \mathrm{Swish}_1\left( w_t W^{gate} \right) \odot \left( w_t W^{up} \right) \right) W^{down} \qquad (3)
$$

**符号**：$W^{gate}, W^{up} \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$：门路与值路投影；$W^{down} \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$；$\mathrm{Swish}_1(u) = u \, \sigma(u)$（SiLU，2.6 式 (4)）；$\odot$：逐元素乘。

整机栈在 $L$ 个 block 首尾各加一件：

$$
x^{(0)}_t = E_{x_t}, \qquad z_t = W_U \, \mathrm{RMSNorm}_f\left( x^{(L)}_t \right), \qquad q_\theta\left( \cdot \mid x_{<t} \right) = \mathrm{softmax}\left( z_{t-1} \right) \qquad (4)
$$

**符号**：$E \in \mathbb{R}^{V \times d}$：输入 embedding（查表，2.2 式 (7)）；$\mathrm{RMSNorm}_f$：final norm（其必要性 = Pre-LN stream 范数随 $\sqrt{L}$ 增长，2.7 式 (12)）；$W_U \in \mathbb{R}^{V \times d}$：输出头，tie_word_embeddings = false 时与 $E$ 独立（2.2 §9）；$z_t \in \mathbb{R}^{V}$：位置 $t$ 的 logits；$q_\theta(\cdot | x_{<t})$：1.1 式 (2) 的逐位置条件分布。

式 (4) 的两端各管一件事：$E$ 把整数下标变成 $d$ 维向量（gather，不进 FLOPs 账，1.2 §3 口径）；$\mathrm{RMSNorm}_f$ 把随 $\sqrt{L}$ 变粗的 stream 压回 $O(\sqrt{d})$ 再交给输出头；$W_U$ 做一次 $d \to V$ 的 matmul（每 token $2Vd$ FLOPs，不可忽略，§5）；softmax 把 logits 变成分布，采样属 1.1 §8。整机公式在两个工作模式下是同一套：训练 / prefill 时 $T > 1$ 并行打分全部位置（teacher forcing，1.1 §4）；decode 时 $T = 1$、式 (2) 的求和上限由 KV cache 的当前长度扮演（2.4 §4.4）——式 (1)–(4) 一行不改，改的只有输入形状与 $k_{s,g}$、$v_{s,g}$ 的来源（现算 vs 缓存）。

接线图（一个 block；shape 为 Llama-3.1-8B 数值）：

```
x⁽ˡ⁻¹⁾ [B, T, 4096] ──┬─────────────────────────────────────► (+) ──► x̃⁽ˡ⁾
                       └─► RMSNorm₁ ──► QKV 投影 ──► RoPE(q,k) ──► softmax(QKᵀ/√D + mask)·V
                                              (GQA: K/V 8 头广播到 32)
                                          ──► concat 32 头 ──► W^O ──► [B, T, 4096] ──►┘
x̃⁽ˡ⁾ ──┬────────────────────────────────────────────────────► (+) ──► x⁽ˡ⁾ [B, T, 4096]
         └─► RMSNorm₂ ──► gate/up 投影 [B, T, 14336]×2 ──► SiLU ⊙ ──► down 投影 ──►┘
```

式 (1)–(4) 就是「现代标准配方」的全部结构内容：2.1 的可加展开、2.4 的可缓存性、2.7 的恒等梯度通路全部由式 (1) 的「归一化读、加法写」形态决定，组件内部（式 (2)(3)）换掉任何一件——RoPE 换 ALiBi、SwiGLU 换 GELU、GQA 换 MHA——stream 层面的结论逐项保留。这正是把 P2 拆成七章的根据：接口是式 (1)，组件是可替换的实现。

整机栈（式 (4)）的形状走读，参数住处在右侧标出：

```
ids [B, T]
  │  gather（0 FLOPs）                       E ∈ R^{V×d}
  ▼
x⁽⁰⁾ [B, T, d]
  │  ┌─ ×L 个 block（式 (1)–(3)）────────── 每层：W^Q/W^K/W^V/W^O + 三矩阵 FFN + γ₁, γ₂
  ▼
x⁽ᴸ⁾ [B, T, d]
  │  RMSNorm_f                              γ_f ∈ R^{d}
  │  matmul                                 W_U ∈ R^{V×d}（untied 时独立）
  ▼
logits z [B, T, V]  ──softmax──►  q_θ(· | x_{<t})
```

参数只住在五处：$E$、每层的 attention 四矩阵、每层的 FFN 三矩阵、norm 增益、$W_U$——§6 的对账就是把这五处逐个加出来。

**变体注记（对照锚点，归属严格按 2.7 §9）**。OLMo-2-7B 不是 Pre-LN：norm 在子层输出侧（reordered），$x \leftarrow x + \mathrm{RMSNorm}(f(x))$（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]）；其 attention 另在 q/k 投影后各过一次 RMSNorm（QK-norm 全维变体，`modeling_olmo2.py` 的 `q_norm` / `k_norm`，[HuggingFace, modeling_olmo2.py](../references.md)，[A]）。logit 稳定器各家不同：Llama-3.1 无（仅 $1/\sqrt{D}$ 缩放）；Gemma 2 用 soft-capping（attn 50.0 / final 30.0，无 QK-norm）；Gemma 3 以 QK-norm 取代 soft-capping，且每个子层前后各一次 RMSNorm。这些替换都只动式 (1)(2) 的局部，动不了整机骨架。

## 2. config.json ↔ 公式映射表（逐字段）

Llama-3.1-8B 的 config.json 逐字段映射到式 (1)–(4) 的符号（数值：[Llama-3.1-8B config](../references.md)，[A]）：

| config 字段 | 值 | 符号 / 角色 | 进入公式 |
| --- | --- | --- | --- |
| vocab_size | 128256 | $V$：$E$ 与 $W_U$ 的行数、logits 维 | 式 (4)；2.2 §7 三方账 |
| hidden_size | 4096 | $d$：stream 宽度 | 式 (1)–(4) 全部 |
| num_hidden_layers | 32 | $L$：block 重复次数 | 式 (1) |
| num_attention_heads | 32 | $H_q$：Q head 数 | 式 (2)；scores 与 FLOPs 按它计（1.2 式 (6)） |
| num_key_value_heads | 8 | $H_{kv}$：KV head 数，进分组映射 $g(h)$ | 式 (2)；KV cache（2.4 式 (9)） |
| （导出）head_dim | 128（$= d / H_q$） | $D$：$1/\sqrt{D}$ 的缩放、RoPE 的 $D/2$ 个频率通道 | 式 (2) |
| intermediate_size | 14336 | $d_{\mathrm{ff}} = 3.5d$ | 式 (3)；2.6 §3.3 |
| hidden_act | silu | $\mathrm{Swish}_1$ | 式 (3) 门函数 |
| rope_theta | 500000 | $base$：$\theta_j = base^{-2j/D}$ | 式 (2) 的 $R_t$；2.5 式 (16) |
| max_position_embeddings | 131072 | $T_{\max}$：位置 id 定义域、cache 上限 | 2.4 §5；2.5 §4.5 |
| rms_norm_eps | 见注 | $\epsilon$：RMSNorm 分母的稳定项 | 2.7 式 (5) |
| attention_bias | false | 式 (2) 四投影无 bias 项 | 式 (2)；2.3 §2 |
| mlp_bias | false | 式 (3) 三矩阵无 bias | 式 (3)；2.6 §1 |
| tie_word_embeddings | false | $W_U$ 独立于 $E$，参数账 $+2Vd$ | 式 (4)；2.2 §9 |
| attention_dropout | 0.0 | 训练期正则，推理恒等 | 不进公式 |
| （bos/eos_token_id、torch_dtype 等） | — | 采样终止 / 存储精度口径 | 不进 block 公式 |

注：rms_norm_eps 的 Llama-3.1-8B 字段值本书未独立核实（HF gated repo，W1 research pass 口径）；已核实的对照值是 Qwen3-8B 与 OLMo-2-7B 的 $10^{-6}$（各 config 条目，[A]）。

读表三条规矩。其一，**导出字段与显式字段分开**：head_dim 在该 config 中不出现、按 $d / H_q$ 导出；新版 modeling 代码允许它被显式覆盖（§8 误解一）。其二，**缺省字段按默认理解**：attention_dropout 等训练侧字段不进推理公式。其三，**每个结构字段都能在公式里点名位置**——反过来，读一个陌生模型的 config 时，把每个字段在这张表里对号入座，五个整数（$L, d, H_q, H_{kv}, d_{\mathrm{ff}}$）加 $V$ 就钉死了参数量与 FLOPs 账（§5、§6），剩下的是 norm 摆放与 logit 稳定器两个不进 config 标准字段的配方差异（§1 末表）。这张字典表是 5.6「30 分钟拆解新模型」checklist 的字段级底稿。

把同一张表套到对照锚点上走一遍，验证它的可迁移性。OLMo-2-7B（[OLMo-2-7B config](../references.md)，[A]）：$L = 32$、$d = 4096$、$H_q = H_{kv} = 32$、$d_{\mathrm{ff}} = 11008$、$V = 100352$、rope_theta = 500000、rms_norm_eps = $10^{-6}$、attention_bias = false、tie = false。式 (2) 里唯一的结构变化是 $g(h)$ 退化为恒等映射（$H_q / H_{kv} = 1$，每头各配一对 K/V，MHA 标准形）；式 (3) 不变，只是 $d_{\mathrm{ff}}$ 收窄到 $2.69d$。这一步 mapping 能直接读出的是参数与 FLOPs 账的变化（§6 练习口径）；读不出的是两件事——norm 摆在子层输出侧（reordered）与 q/k 投影后的两次 RMSNorm——它们不写进 config 的任何标准字段，只能去 `modeling_olmo2.py`（[A]）里看 `Olmo2DecoderLayer.forward` 的接线顺序与 `Olmo2Attention` 的 `q_norm` / `k_norm`。这就是本表的能力边界，也是它的用法：config 管「形状与账」，modeling 代码管「接线与归一化位置」，两边各读一遍才算读完一个模型。

## 3. minimal 实现：约 150 行整合 P2 全组件

教学用，非优化实现：显式物化 $[B, H, T, T]$ scores（2.3 §7 的 $O(T^{2})$ 内存照付）、无 KV cache（2.4 的 cache 版 forward 可直接换装）、无 kernel 融合；组件逐一对应 2.2–2.7 的教学版。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):                       # 2.7 式 (5)：只有 gamma，无 beta
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):                       # x: [B, T, d]
        var = x.pow(2).mean(-1, keepdim=True)   # RMS(x)^2，沿特征维归约
        return self.weight * (x * torch.rsqrt(var + self.eps))


class RotaryEmbedding(nn.Module):               # 2.5 式 (16)：只预存 inv_freq
    def __init__(self, D, base):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, D, 2).float() / D)   # theta_j
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, T, device):               # 每次 forward 由 inv_freq x 位置现算
        freqs = torch.outer(torch.arange(T, device=device).float(), self.inv_freq)
        return freqs.cos(), freqs.sin()         # 各 [T, D/2]


def rotate_half(x):                             # half-split 配对 (j, j+D/2)，2.5 §4.8
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):                    # x: [B, H, T, D]；式 (2) 的 R_t
    cos = torch.cat([cos, cos], dim=-1)[None, None]   # [1, 1, T, D]
    sin = torch.cat([sin, sin], dim=-1)[None, None]
    return x * cos + rotate_half(x) * sin       # 式 (9)（2.5）逐对通道旋转


class GQAAttention(nn.Module):                  # 式 (2)：GQA + RoPE + causal mask
    def __init__(self, d, H_q, H_kv):
        super().__init__()
        self.H_q, self.H_kv = H_q, H_kv
        self.D = d // H_q                       # 导出 head_dim；显式 head_dim 情形见 §8
        self.W_q = nn.Linear(d, H_q * self.D, bias=False)
        self.W_k = nn.Linear(d, H_kv * self.D, bias=False)
        self.W_v = nn.Linear(d, H_kv * self.D, bias=False)
        self.W_o = nn.Linear(H_q * self.D, d, bias=False)

    def forward(self, x, cos, sin):             # x: [B, T, d]（u^(l)，已归一化）
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.H_q, self.D).transpose(1, 2)   # [B, H_q, T, D]
        k = self.W_k(x).view(B, T, self.H_kv, self.D).transpose(1, 2)  # [B, H_kv, T, D]
        v = self.W_v(x).view(B, T, self.H_kv, self.D).transpose(1, 2)
        q = apply_rope(q, cos, sin)             # RoPE 只作用 q、k，不作用 v
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.H_q // self.H_kv, dim=1)   # g(h) 的物化：组内广播
        v = v.repeat_interleave(self.H_q // self.H_kv, dim=1)   # [B, H_q, T, D]
        s = q @ k.transpose(-2, -1) * self.D**-0.5              # [B, H_q, T, T]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        s = s.masked_fill(causal, float("-inf"))                # 2.4 式 (3)
        p = torch.softmax(s, dim=-1)                            # alpha，式 (2)
        o = p @ v                                               # [B, H_q, T, D]
        o = o.transpose(1, 2).reshape(B, T, self.H_q * self.D)  # concat heads
        return self.W_o(o)                                      # [B, T, d]


class SwiGLUFFN(nn.Module):                     # 式 (3) = 2.6 式 (6)
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)   # W^gate（门路）
        self.up_proj = nn.Linear(d, d_ff, bias=False)     # W^up（值路）
        self.down_proj = nn.Linear(d_ff, d, bias=False)   # W^down

    def forward(self, x):                       # x: [B, T, d]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):                  # 式 (1)：Pre-LN 两次「读-写」
    def __init__(self, d, H_q, H_kv, d_ff, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d, eps)
        self.attn = GQAAttention(d, H_q, H_kv)
        self.norm2 = RMSNorm(d, eps)
        self.ffn = SwiGLUFFN(d, d_ff)

    def forward(self, x, cos, sin):             # x: [B, T, d]
        x = x + self.attn(self.norm1(x), cos, sin)   # h <- h + Attn(RMSNorm(h))
        x = x + self.ffn(self.norm2(x))              # h <- h + FFN(RMSNorm(h))
        return x


class MiniDecoder(nn.Module):                   # 式 (4)：embed -> L blocks -> final norm -> head
    def __init__(self, V=128256, d=4096, L=32, H_q=32, H_kv=8, d_ff=14336,
                 base=500000.0, eps=1e-6):
        super().__init__()                      # 默认参数 = Llama-3.1-8B config
        self.embed = nn.Embedding(V, d)         # E（2.2 式 (7)）
        self.rot = RotaryEmbedding(d // H_q, base)
        self.layers = nn.ModuleList([DecoderBlock(d, H_q, H_kv, d_ff, eps) for _ in range(L)])
        self.norm_f = RMSNorm(d, eps)           # final RMSNorm（2.7 式 (12) 的收尾）
        self.lm_head = nn.Linear(d, V, bias=False)   # W_U，untied（tie_word_embeddings=false）

    def forward(self, ids):                     # ids: [B, T]
        cos, sin = self.rot(ids.shape[1], ids.device)
        x = self.embed(ids)                     # x^(0) [B, T, d]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm_f(x))     # logits [B, T, V]


# 冒烟测试（缩小 config，跑通 shape 与参数账）
model = MiniDecoder(V=1024, d=256, L=4, H_q=8, H_kv=2, d_ff=704)
logits = model(torch.randint(0, 1024, (2, 16)))
assert logits.shape == (2, 16, 1024)
n = sum(p.numel() for p in model.parameters())
expect = 1024 * 256 + 4 * (256 * 8 * 32 + 2 * 256 * 2 * 32 + 32 * 8 * 256 + 3 * 256 * 704
                           + 2 * 256) + 256 + 1024 * 256   # §6 式 (6) 的逐项加
assert n == expect
```

组件 ↔ 前文对照：

| 教学类 | 公式出处 | 组件章节 |
| --- | --- | --- |
| `RMSNorm` | 2.7 式 (5) | 2.7 |
| `RotaryEmbedding` / `apply_rope` | 2.5 式 (12)(16)，half-split 实现 | 2.5 §4.8 |
| `GQAAttention` | 2.3 式 (4)(11) + 组内广播（机制归 4.1）+ causal mask | 2.3 / 2.4 |
| `SwiGLUFFN` | 2.6 式 (6) | 2.6 |
| `DecoderBlock` | 本章式 (1) = 2.1 式 (1) | 2.1 |
| `MiniDecoder` | 式 (4)：embedding（2.2 式 (7)）、final norm（2.7 §5.2）、输出头（1.1 式 (2)） | 2.2 |

冒烟测试的最后一行断言就是 §6 参数量公式的可执行形式：把 config 五个整数代入式 (6)，与 `sum(p.numel())` 逐项相等——教学实现正确性的第一条检查不是输出对不对，是参数账平不平。

这份实现刻意不含的东西同样值得点名：初始化方案（`nn.Linear` 默认初始化，未按 2.2 §8 的 0.02 惯例重设）、混合精度（全程 fp32 默认 dtype，无 fp32 归约）、KV cache（2.4 的 cache 版 forward 留作练习 3）、dropout 与训练循环、分布式钩子。它们每一条在官方实现里都对应一层工程代码（§4 末表），但都不是式 (1)–(4) 的内容——教学版的边界就是公式的边界。冒烟测试把 config 缩到 $d = 256$ 是为了在 CPU 上秒级跑完；把默认参数（Llama-3.1-8B 原尺寸）实例化出来需要约 32 GB 内存（fp32 权重），数学上同一台机器，工程上另是一笔内存账（1.2 式 (7) 的 4 字节版本）。

教学参照对照：nanoGPT（[Karpathy, nanoGPT](../references.md)，教学参照）是 GPT-2 配方（[Radford et al., 2019](../references.md)，[W]）的最小实现，其 `Block.forward` 与 §3 的 `DecoderBlock` 骨架相同（`x = x + attn(ln_1(x))`、`x = x + mlp(ln_2(x))`，外加 final `ln_f`）——式 (1) 的「归一化读、加法写」自 GPT-2 起就是这个形态（GPT-2 报告 §2.2：LN 移到每个子块输入侧、末块后再加一次 LN，即 Pre-LN + final norm）。两代配方的差异全部在组件层：

| 组件 | nanoGPT / GPT-2 | §3 / Llama-3.1 配方 |
| --- | --- | --- |
| norm 类型 | LayerNorm（$\gamma$、$\beta$ 双参数，eps $10^{-5}$） | RMSNorm（仅 $\gamma$，eps $10^{-6}$ 量级） |
| norm 摆放 | Pre-LN + final LN | Pre-LN + final RMSNorm（OLMo-2 为 reordered 变体） |
| 位置编码 | learned absolute 查表（1024 行，加进 stream） | RoPE（attention 内旋转，stream 无位置向量） |
| 投影 bias | True（attention 与 FFN 全带） | False |
| tying | tied（$W_U = E$，124M 对账见 2.2 §9） | untied（独立 $W_U$，$+Vd$） |
| FFN | 两矩阵 GELU，$d_{\mathrm{ff}} = 4d$（gelu_new = tanh 近似） | 三矩阵 SwiGLU，$d_{\mathrm{ff}} = 3.5d$ |
| attention | MHA（$H_{kv} = H_q$） | GQA（$H_{kv} = H_q / 4$） |

读这张表的方式：左列到右列的每一行都对应 P2 某一章的一次推导（LayerNorm→RMSNorm 是 2.7 §3，learned abs→RoPE 是 2.5，bias 与 tying 是 §8 误解二三，GELU→SwiGLU 是 2.6，MHA→GQA 是 4.1 的预告）。骨架七年未动，动的是组件——这就是「现代标准配方」的精确含义：式 (1) 的接线 + 右列的组件选择。

## 4. 官方实现走读：modeling_llama.py

走读对象：huggingface/transformers repo 的 `src/transformers/models/llama/modeling_llama.py`（[HuggingFace, modeling_llama.py](../references.md)，[A]）。下面按类给出关键行摘录与式 (1)–(4) 的对照。

**`LlamaRMSNorm.forward`**（式 (1) 的两个 norm 与式 (4) 的 final norm 共用此类）：

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

与 2.7 式 (5) 逐行对应，参数只有 `weight`（即 $\gamma$，无 $\beta$）；与 §3 教学版的唯一实质差异是精度处理：归约与除法在 fp32 里做、再降回输入 dtype——bf16 下 `pow(2).mean` 的归约误差被这一步挡在门外。OLMo 2 的 `q_norm` / `k_norm`（`modeling_olmo2.py`，[A]）是同一形态搬进 attention 内部（2.7 §7）。

**`LlamaAttention.__init__`** 的四行决定式 (2) 的形状：

```python
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads   # = 4
self.scaling = self.head_dim**-0.5
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
# k_proj / v_proj 的输出宽 num_key_value_heads * head_dim；o_proj 反方向
```

`head_dim` 先查显式字段、缺省才用 $d / H_q$ 导出——§8 误解一的代码根据。`scaling = head_dim**-0.5` 是 2.3 式 (4) 的 $1/\sqrt{d_k}$ 原样。

**`LlamaAttention.forward`** 骨架（式 (2) 的生产形态）：

```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [B, H_q, T, D]
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)     # [B, H_kv, T, D]
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, ...)
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, ...)
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

三步与式 (2) 的对应：投影 + view/transpose 得到 $[B, H, T, D]$；`apply_rotary_pos_emb` 是 2.5 §4.8 的 half-split 旋转（`x * cos + rotate_half(x) * sin`），只作用 q、k；`past_key_values.update` 是 2.4 §4.4 读写模式的生产形态（预分配 Cache 对象，不是教学版的 `torch.cat`）。打分发派给 `attention_interface`：eager 路径的 `eager_attention_forward` 与式 (2) 逐行同构，逐行对照：

| eager 代码行 | 式 (2) 的对应 |
| --- | --- |
| `repeat_kv(key, module.num_key_value_groups)` | $g(h)$ 的物化：8 对 K/V 复制成 32 份 |
| `torch.matmul(query, key_states.transpose(2, 3)) * scaling` | $\langle q_{t,h}, k_{s,g(h)} \rangle / \sqrt{D}$ |
| `attn_weights = attn_weights + causal_mask` | 加性 $M$（2.4 式 (1)），求和截到 $s \le t$ |
| `softmax(..., dim=-1, dtype=torch.float32).to(query.dtype)` | $\alpha_{t,s,h}$；升 fp32 再 softmax（2.3 §4 的生产写法） |
| `torch.matmul(attn_weights, value_states)` | $o_{t,h} = \sum_{s \le t} \alpha_{t,s,h} v_{s,g(h)}$ |

默认不跑 eager，而经 `ALL_ATTENTION_FUNCTIONS` 派发到 sdpa / flash——数学内容与式 (2) 恒等，算子融合归 4.1。

**`LlamaMLP.forward`** 一行即式 (3)：

```python
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

`act_fn` 由 `config.hidden_act`（silu）查表得到；三个投影的 bias 由 `config.mlp_bias` 控制（Llama-3.1-8B 为 false，2.6 §1）。

**`LlamaDecoderLayer.forward`** 是式 (1) 的逐行直译：

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

**`LlamaModel` / `LlamaForCausalLM`**（式 (4)）：`embed_tokens` 查表后依次过 `self.layers`，`self.norm` 收尾，`lm_head` 出 logits；`tie_word_embeddings` 为 true 时 `lm_head.weight` 与 `embed_tokens.weight` 共享同一参数（Llama-3.1-8B 为 false）。`LlamaForCausalLM.forward` 的训练分支把 logits 与 labels 交给 loss 工具做移位对齐——$z_{t-1}$ 对 $x_t$ 的逐位置错位监督（1.1 §4 的 teacher forcing 口径），这一步在 block 结构之外，式 (4) 末行的条件分布写法就是它的数学形态。位置编码的组装位置值得单独点名：模型级持有一个 `LlamaRotaryEmbedding`，每次 forward 以 `position_embeddings = self.rotary_emb(hidden_states, position_ids)` 现算一次 cos/sin、传给全部层共享（RoPE 只依赖位置与频率，与层无关）；其内部只有 `inv_freq` 是预存 buffer，cos/sin 由 `inv_freq` 与 `position_ids` 逐前向外积现算（fp32 下 `freqs = inv_freq @ position_ids`，再 `cat([freqs, freqs])` 后取 cos/sin）——与 2.5 §4.8 口径一致，不是「初始化时预建 $[T_{\max}, D]$ 大表」的旧式实现。decode 时 `position_ids` 携带全局绝对位置，与 2.5 §4.3 的接缝注记（q 必须按全局位置旋转）咬合。

mask 的组装同样在模型级完成一次：`LlamaModel.forward` 依据 attention 实现类型生成一份加性 causal mask（下三角外为 $-\infty$，2.4 式 (3) 的加性形态），与 cos/sin 一起作为参数传给每层——每个 block 收到的「位置与可见性」输入因此完全相同，层间差异全部来自各自的权重矩阵。对照教学版：§3 把 mask 的构造放在了每个 attention 内部（逐层重复造同一张上三角），这是教学上的就近原则，不是语义差异。

教学版与官方版的差异汇总：

| 维度 | §3 教学版 | 官方 modeling_llama.py |
| --- | --- | --- |
| cos/sin | 每 forward 由 inv_freq × arange($T$) 现算，逐层传入 | 模型级每 forward 由 inv_freq × position_ids 现算一次、各层共享；支持非连续 position_ids（cache / packing） |
| causal mask | 手工 `triu` + `masked_fill(-inf)` | mask 基建生成加性 $-\infty$ mask；sdpa 路径用 `is_causal` |
| GQA 广播 | `repeat_interleave` 显式复制 K/V | eager 用 `repeat_kv` 同形复制；sdpa/flash 不物化（kernel 内广播） |
| softmax / RMSNorm dtype | 输入 dtype | 归约与 softmax 升 fp32 再降回 |
| KV cache | 无（2.4 教学版为 `torch.cat` 语义演示） | `Cache` 对象预分配、按层 `update` |
| attention 后端 | 手写物化 scores | `ALL_ATTENTION_FUNCTIONS` 派发到 sdpa / flash / flex |
| 生产钩子 | 无 | 初始化方案、张量并行切分、梯度检查点、量化 |

走读的读法：官方代码里凡是不出现在式 (1)–(4) 的部分，都可以归入上表后四行——数值精度、内存管理、算子派发、分布式。公式与实现之间隔着的就是这四层工程，不多不少。术语注：表中「张量并行」（tensor parallelism）指把单层权重矩阵按行/列切分到多张卡、行并行 GEMM 的尾部需要一次 all-reduce 合并部分和的分布式方案；其系统展开属 cluster 维度，本书只用到「每层 all-reduce 次数」这一账面量（§7）。核对方式也是现成的：把 §3 教学版与该文件并排打开，逐类对照上表七行；任何一行对不上，要么是你读漏了生产钩子，要么是 modeling 代码升级改了接线——后者值得停下来查 commit，因为接线变化就是配方变化（§1 变体注记的各家差异在代码里就体现为这些行）。

## 5. 单层单 token FLOPs 全算

把式 (2)(3) 的每个运算按 1.2 的 $2mnk$ 口径数一遍（Llama-3.1-8B 数值，per token、单层、causal 平均口径 $T/2$；attention 行取 $T = 4096$）。这张表是 1.2 §7 按本章公式的重推，并把那里声明忽略的低阶项一并量化：

| 行 | 运算 | FLOPs 公式 | 代入 | 数值 |
| --- | --- | --- | --- | --- |
| 1 | QKV 投影 | $2d\left( H_q + 2H_{kv} \right) D$ | $2 \times 4096 \times 48 \times 128$ | 50.3 M |
| 2 | RoPE（q、k） | $3D \left( H_q + H_{kv} \right)$（每 2 维块 4 乘 2 加） | $3 \times 128 \times 40$ | 0.015 M |
| 3 | score + 加权和 | $2Td$（$QK^{\top}$ 与 $PV$ 各 $Td$） | $2 \times 4096 \times 4096$ | 33.6 M |
| 4 | softmax（exp + 归一化） | $\sim 2 H_q T$（逐元素） | $2 \times 32 \times 4096$ | 0.26 M |
| 5 | out 投影 | $2d^{2}$ | $2 \times 4096^{2}$ | 33.6 M |
| 6 | RMSNorm × 2 | $\sim 4d$ 每个（平方、归约、rsqrt、乘 $\gamma$） | $8 \times 4096$ | 0.03 M |
| 7 | 残差加 × 2 | $2d$ | $2 \times 4096$ | 0.008 M |
| 8 | FFN 三矩阵 | $6 d \, d_{\mathrm{ff}}$ | $6 \times 4096 \times 14336$ | 352.3 M |
| 9 | SiLU + ⊙ | $\sim 5 d_{\mathrm{ff}}$ | $5 \times 14336$ | 0.07 M |

matmul 行（1、5、8）合计：

$$
\mathrm{FLOPs}_{\mathrm{layer}} = \underbrace{2d\left( H_q + 2H_{kv} \right) D + 2d^{2} + 6 d\, d_{\mathrm{ff}}}_{\text{matmul} \;\to\; 2N} + \underbrace{2Td}_{\text{attention}} + O\left( d + d_{\mathrm{ff}} + H_q T \right) \qquad (5)
$$

**符号**：第一大括号是 1.2 式 (2) 的层内成分（每参数一次 FMA 的逐行展开）；$2Td$ 是 1.2 式 (6) 的 per-token 形态；$O(\cdot)$ 余项 = 第 2、4、6、7、9 行。

数值闭环：matmul 行合计 $4.362 \times 10^{8}$，乘 $L = 32$ 得 $1.396 \times 10^{10} = 2N$（层参数口径 $N = 6.98 \times 10^{9}$，1.2 §9）——式 (5) 与「每参数一次 FMA」互相验证。低阶行合计约 $3.9 \times 10^{5}$，占 matmul 行的 0.09%，比值约千倍、与 $d = 4096$ 同量级——1.2 §3「elementwise 相对 $2N$ 低约一个 $d$ 因子」的忽略口径在这里被量化而非断言。整机 per-token 前向：层部分 $1.396 \times 10^{10}$ + 输出头 $2Vd = 1.05 \times 10^{9}$ ≈ $1.50 \times 10^{10}$ FLOPs；$T = 4096$ 时 attention 本体再加 $1.07 \times 10^{9}$（占 6.7%）。

decode 口径注记：上表是训练 / prefill 的并行前向（$T$ 个位置一次打完，第 3 行按 causal 平均 $T/2$ 计）。逐 token 生成时（2.4 §3 的 decode 阶段），第 1、5、8 行的 per-token 数值逐项不变（投影与 FFN 本就是逐位置的），第 3 行变成 $4td$（当前 query 对 $t$ 个缓存 K/V 的打分与加权和，2.4 §4.4），mask 行整行消失（cache 的定义域即可见集）——同一张表的两个读法，对应 2.4 式 (8) 的两项。

九行里只有第 3、4 两行随 $T$ 变，这正是式 (5) 把 $2Td$ 单列的原因。$T$ 方向的走向可以直接推：令 attention 行等于同层 matmul 行，$2Td = 2d\left( H_q + 2H_{kv} \right) D + 2d^{2} + 6 d\, d_{\mathrm{ff}}$，解出 $T = \left( H_q + 2H_{kv} \right) D + d + 3 d_{\mathrm{ff}} = 6144 + 4096 + 43008 \approx 5.3 \times 10^{4}$——与 1.2 §6 的交叉点 $T^{*} = N / (Ld) \approx 5.3 \times 10^{4}$ 是同一方程的两种写法（彼处按全机层参数口径，此处按单层逐行口径，数值相合）。$T = 131072$（满窗）时第 3 行涨到 $1.07 \times 10^{9}$，超过 matmul 三行之和的两倍——1.2 §6 三档表（6.7% / 36% / 70%）的行级来源就是这张表。读这张表的方式与 1.2 §7 相同：公式列是推导（每个系数来自式 (2)(3) 的哪个运算），代入列的每个整数都能在 §2 的映射表里指回一个 config 字段——换任何模型的 config 照此重算，这就是「config 决定一切账」的操作含义。

## 6. 参数量精确对账：从 $12Ld^{2}$ 到个位数

把 §2 的字段逐个搬进参数账（每一项的来源运算标在括号里）：

$$
N_{\mathrm{total}} = \underbrace{Vd}_{E} + L \left( \underbrace{d^{2}}_{W^Q} + \underbrace{2 d H_{kv} D}_{W^K, W^V} + \underbrace{d^{2}}_{W^O} + \underbrace{3 d\, d_{\mathrm{ff}}}_{\text{FFN}} + \underbrace{2d}_{\gamma_1, \gamma_2} \right) + \underbrace{d}_{\gamma_f} + \underbrace{Vd}_{W_U} \qquad (6)
$$

**符号**：$\gamma_1, \gamma_2, \gamma_f$：每层两个 RMSNorm 增益与 final norm 增益（各 $d$ 个参数——norm 不是无参算子，只是低阶）；其余各项同式 (2)(3)(4)。

代入 Llama-3.1-8B（$V = 128256$、$d = 4096$、$L = 32$、$H_{kv} = 8$、$D = 128$、$d_{\mathrm{ff}} = 14336$）逐项加：embedding $525336576$；每层 attention $41943040$（$= 2.5d^{2}$，GQA 痕迹）、FFN $176160768$、norm $8192$，层计 $218112000$，乘 32 得 $6979584000$；final norm $4096$；输出头 $525336576$。合计：

$$
N_{\mathrm{total}} = 525336576 + 6979584000 + 4096 + 525336576 = 8030261248 \qquad (7)
$$

「8B」名实相符（$8.03 \times 10^{9}$）。三层读法。其一，**norm 项确实低阶但不是零**：$(2L+1)d = 266240$ 个参数，占 0.0033%——2.1 §4「$O(Ld)$ 低阶项忽略」的速算口径与精确账的差就住在这里；对账到个位时不能省。其二，**与 2.1 的 $12Ld^{2}$ 速算对账**：速算给层参数 $6.44 \times 10^{9}$，精确值 $6.98 \times 10^{9}$，偏差 $+8.4\%$ 的分解签名（GQA $-1.5d^{2}$、宽 FFN $+2.5d^{2}$）已在 2.1 §4 登记；本章把 embedding、输出头、norm 全部入账后，速算与精确值的分工是「量级与签名」对「个位」。其三，**对账练习的操作版**：拿到任何 [A] 级模型，式 (6) 左端用 config 字段算、右端用 `sum(p.numel())` 量（或官方公布的参数量），两边对不上时——漏 tied/untied 的 $Vd$、漏 GQA 的 $H_{kv}$、漏 norm 增益、把 head_dim 当 $d/H_q$ 硬算——四种错法正好对应 §8 的四条误解。

两档精度各有用武之地。速算档（2.1 式 (3) 的 $12Ld^{2}$ + $2Vd$）用于扫读：看到一个新模型先算它，量级与偏差签名（GQA、宽 FFN）三十秒内可读。精算档（式 (6)）用于落账：Part 5 逐家对账、复现实现的参数断言（§3 冒烟测试）、MoE 激活参数换算（4.5 的前置）都要个位级闭环——对不上的每一百万参数都对应一个具体的结构字段，不许用「约」字糊弄过去。

对照锚点再算一遍，把「接线差异进不进参数账」这个问题做实。OLMo-2-7B（$L = 32$、$d = 4096$、$H_{kv} = 32$ MHA、$d_{\mathrm{ff}} = 11008$、$V = 100352$、untied）：每层 attention $4d^{2} = 67108864$、FFN $3 d\, d_{\mathrm{ff}} = 135266304$，矩阵层计 $202375168$，乘 32 得 $6476005376$；embedding 加输出头 $2Vd = 822083584$。norm 侧多一笔：reordered 接线下每层仍是两个子层 norm（作用于子层输出，参数量不变），但 attention 内部多了 `q_norm` / `k_norm` 两个全投影维 RMSNorm（2.7 §7 的全维变体，各 $d = 4096$ 个增益），即每层 4 个 norm、$4d = 16384$ 个参数，全机 norm 参数 $32 \times 16384 + 4096 = 528384$——是 Llama-3.1-8B 同项（266240）的两倍。合计 $6476005376 + 822083584 + 528384 = 7298617344 \approx 7.30 \times 10^{9}$，与 2.1 §4 表的总量级一致。结构差异（reordered、QK-norm）不改变式 (2)(3) 的矩阵形状，只改变 norm 增益的个数与接线位置——这就是为什么 config 逐字段表（§2）能定参数账、却定不了全部结构，两边各读一遍的分工在参数账上同样成立。

## 7. 结构变体：并行块（parallel attention/FFN block）

式 (1) 的两个子层串行读写 stream：FFN 读到的是 attention 写过的 $\tilde{x}^{(\ell)}$。另一条接线方式让两个子层并行——同一份归一化输入、两路输出同时加回（PaLM / GPT-J / Falcon 的块形态；PaLM 论文归后续波次，本节只登记公式与机制含义，不引其实验数字）：

$$
x^{(\ell)} = x^{(\ell-1)} + \mathrm{Attn}^{(\ell)}\left( \mathrm{Norm}^{(\ell)}\left( x^{(\ell-1)} \right) \right) + \mathrm{FFN}^{(\ell)}\left( \mathrm{Norm}^{(\ell)}\left( x^{(\ell-1)} \right) \right) \qquad (8)
$$

**符号**：$\mathrm{Norm}^{(\ell)}$：单个归一化（该族用 LayerNorm；换成 RMSNorm 不改变本节论证）；两个分支读同一份 $\mathrm{Norm}^{(\ell)}(x^{(\ell-1)})$。

与串行形（式 (1)）逐项对照：

| 维度 | 串行块（式 (1)） | 并行块（式 (8)） |
| --- | --- | --- |
| norm 次数 / block | 2（$\gamma_1, \gamma_2$） | 1 |
| 分支依赖 | FFN 输入含 attention 当层输出 | 两分支独立，同读 $x^{(\ell-1)}$ |
| block 内复合深度 | 每 block 两级复合（全机 $2L$ 级子层复合） | 每 block 一级并列（$L$ 级复合） |
| 参数量 / FLOPs | $2.5d^{2} + 3 d\, d_{\mathrm{ff}}$ / 层（GQA 口径） | 逐项相同（唯一差：每层 norm 增益 $2d \to d$，见上行） |
| GEMM 形态 | 两批输入不同，天然先后 | QKV 与 gate/up 读同一输入，可拼成一个大 GEMM；out 与 down 同理（输出相加） |
| 张量并行通信 | 每层两次行并行 all-reduce | 两分支输出求和后合并为一次 |

梯度含义可以精确点名，不靠传闻。把式 (8) 按 2.1 §2 展开：$x^{(L)} = x^{(0)} + \sum_{\ell} a_{\ell} + \sum_{\ell} f_{\ell}$——与串行 Pre-LN 的展开式逐项同形（差别只在 $f_{\ell}$ 的输入是 $x^{(\ell-1)}$ 还是 $\tilde{x}^{(\ell)}$），于是 2.7 式 (8) 的恒等项逐字保留：上游梯度经恒等通道原样直达每一层，并行块没有引入任何打断可加结构的东西（norm 仍在分支输入侧，不在加法之后）。变的是分支梯度的通路数，可以写出来对照。串行形里第 $\ell$ 层 FFN 分支对 $x^{(\ell-1)}$ 的 Jacobian 是复合的：

$$
\text{(serial)} \quad \frac{\partial f_{\ell}}{\partial x^{(\ell-1)}} = \frac{\partial f_{\ell}}{\partial \tilde{x}^{(\ell)}} \left( I + \frac{\partial a_{\ell}}{\partial x^{(\ell-1)}} \right) \;; \qquad \text{(parallel)} \quad \frac{\partial f_{\ell}}{\partial x^{(\ell-1)}} = \frac{\partial f_{\ell}}{\partial \mathrm{Norm}(x^{(\ell-1)})} \, J_{\mathrm{Norm}} \qquad (9)
$$

**符号**：$a_{\ell} = \mathrm{Attn}^{(\ell)}(\mathrm{Norm}(x^{(\ell-1)}))$；$J_{\mathrm{Norm}}$：归一化的 Jacobian（2.7 式 (10) 一族）。串行形的第一式里 $\partial a_{\ell} / \partial x^{(\ell-1)}$ 就是那条「经 attention 输出的间接路径」：FFN 的梯度要过 attention 的 Jacobian 回流到层输入。并行形的第二式没有这一项——反向图上每层少一条跨分支边，两条分支的梯度各自只经自己的子层 Jacobian 与同一个 $J_{\mathrm{Norm}}$ 到达 $x^{(\ell-1)}$。表示侧的对称事实：串行的 FFN 加工的是「刚混完上下文」的 stream，并行的 FFN 只能读「本层 attention 动手之前」的快照；跨 block 的接力仍在（下一层的 norm 读到加和后的 stream），所以并行形损失的是层内接力，不是层间接力。

吞吐含义全部来自「两分支读同一输入」。代入 Llama-3.1-8B 的形状走一遍：串行形里 attention 侧读 $\mathrm{Norm}_1$ 输出的是一个 $d \times 6144$ 的 QKV GEMM，FFN 侧读 $\mathrm{Norm}_2$ 输出的是一个 $d \times 28672$ 的 gate/up GEMM——两次发射、两个输入张量、中间隔着 attention 的全部计算；并行形里两者读同一份归一化输出，可以列拼成一个 $4096 \times 34816$ 的 GEMM 一次发射，输出侧的 out（$4096 \times 4096$）与 down（$14336 \times 4096$）同样可以行拼、结果相加。GEMM 变大变少，tensor core 利用率上升、kernel 发射与调度气泡减少；张量并行下每层两次行并行 all-reduce 合并为一次（两分支输出先求和再通信）；norm 归约每层两次变一次。式 (5) 的账一项不动——省的是墙钟时间里的非计算成分，不是 FLOPs。这与 2.4 §4.4 的 cache 账同构：两处都是「省墙钟 ≠ 省账」。

推理侧同构：decode 每步是 $B \times 1$ 的小 GEMM，权重读取主导（2.4 §6 的算术强度分析），融合不减少权重字节数，但减少每步的 kernel 发射次数与调度间隙——在小 GEMM  regime 里这些非计算成分的占比更大，并行块的墙钟收益在 decode 端只会更明显。

采用记录：PaLM、GPT-J、Falcon 走并行块换训练吞吐；Llama 谱系主配方保持串行（式 (1)）。实证取舍（并行形在其规模下的质量差与速度收益）属各家训练报告，本书不引数字；机制层面的全部差别已在上表与式 (9)。

## 8. 常见误解清单

1. **head_dim = hidden_size / num_attention_heads 是巧合不是定义**。Llama-3.1-8B 的 $128 = 4096 / 32$ 恰好整除成立；Qwen3-8B 的 config 则把 head_dim = 128 写成了显式字段（[Qwen3-8B config](../references.md)，[A]）——同一数值，一种是导出、一种是声明；DeepSeek-V3 的 qk head 维 $= 128$（nope）$+ 64$（rope）$= 192 \neq$ hidden/heads（[DeepSeek-V3 config](../references.md)，[A]；2.3 §5、2.5 §4.7 已用）是显式字段脱离整除关系的实例。代码侧的根据：`LlamaAttention.__init__` 先查显式 `head_dim` 字段、缺省才整除导出（§4）。读陌生 config 时先查有没有显式 head_dim，再决定能不能整除；参数账（$W_h^{Q}$ 的形状）与 RoPE 通道数（$D/2$）都跟在 head_dim 后面，硬整除会连环错。
2. **bias 的有无是配方分野，不是默认值问题**。attention_bias 字段三家全为 false（Llama-3.1-8B、Qwen3-8B、OLMo-2-7B，各 config 条目，[A]）；FFN 侧 Llama-3.1-8B 另有显式 mlp_bias = false，Qwen3 / OLMo-2 的 config 无此字段、其 FFN 无 bias 以 modeling 代码为准（OLMo 2 侧见 `modeling_olmo2.py` 的 `Olmo2MLP`，三个投影硬编码 bias=False，[HuggingFace, modeling_olmo2.py](../references.md)，[A]）——这正呼应 §2 的分工：config 管形状与账，接线与缺省在代码里。GPT-2 与 nanoGPT 老配方全为 true（[Radford et al., 2019](../references.md)，[W]；[Karpathy, nanoGPT](../references.md)，教学参照）。bias 参数只占 $O(Ld)$ 低阶（2.1 §4 口径），去掉它不是「bias 被证明有害」，是现代配方的收敛点；照抄 nanoGPT 的 bias=True 去复现 Llama 类模型是错配。另外两件事不要混：投影的 bias 与 norm 的 $\beta$——LayerNorm 有 $\beta$（2.7 式 (1)），RMSNorm 没有（2.7 式 (5)），「这个模型无 bias」断言的是前者。
3. **tied vs untied 差的是一整个 $Vd$**。tie_word_embeddings = false 时输出头是独立的第二份词表矩阵：Llama-3.1-8B 为 $0.53 \times 10^{9}$、占总参数 6.5%；GPT-2 为 tied（[W]，2.2 §9 的 124M 对账）。对账时漏掉这个字段就凭空「丢」5 亿参数（§6 的四种错法之一）；梯度耦合的取舍（同一份 $E$ 两路梯度）在 2.2 §9，此处只登记口径：判定只看 config 一个布尔字段。
4. **block 之外的参数与 FLOPs 不要漏**。「整机 = $L$ 个 block」的口径会漏掉三样：final RMSNorm（$d$ 个参数 + 每 token 一次归约，2.7 §5.2 给出它必须存在的理由）、输入 embedding（$Vd$ 参数、0 FLOPs——查表不是 matmul，1.2 §3）、输出头（$Vd$ 参数 + $2Vd$ FLOPs，Llama-3.1-8B 上占前向约 7%，1.2 §3 式 (2) 符号表）。§5、§6 的闭环之所以精确到个位，靠的就是这三样都入账。

四条误解共享同一个检出流程：先读 config（§2 的表），再读 modeling 代码的 `__init__`（投影宽度、bias、norm 个数都在那里），最后 `sum(p.numel())` 与式 (6) 对账收尾。任何一步与预期不符，回头查这四条——它们的覆盖面就是实践里参数账对不上的主要原因。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 整机公式 | 式 (1)–(4)：Pre-LN 两子层 + GQA/RoPE/SwiGLU 全下标 | 接口是「归一化读、加法写」；组件可替换、骨架不动 |
| config ↔ 公式 | §2 逐字段表 | 五个整数（$L, d, H_q, H_{kv}, d_{\mathrm{ff}}$）+ $V$ 钉死全部账；5.6 checklist 底稿 |
| minimal 实现 | §3 约 150 行 | 教学用，非优化实现；参数断言 = 式 (6) 可执行形式 |
| 官方走读 | modeling_llama.py：`LlamaRMSNorm` / `LlamaAttention` / `LlamaMLP` / `LlamaDecoderLayer` | cos/sin 逐前向由 inv_freq × position_ids 现算、各层共享；fp32 归约；算子派发 |
| FLOPs 全算 | 式 (5)；matmul 行 $4.36 \times 10^{8}$/层/ token | 低阶项占 0.09%（$\sim d$ 因子）；整机 $2N + 2Vd \approx 1.50 \times 10^{10}$ |
| 参数对账 | 式 (6)(7)：$8030261248$ | norm 增益 $(2L+1)d = 266240$ 入账才到个位；速算与精算分工 |
| 并行块 | 式 (8)：$x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x)) + \mathrm{FFN}(\mathrm{Norm}(x))$ | 恒等通路保留（2.7 式 (8) 逐字成立）；省 GEMM 发射与 all-reduce，不省 FLOPs |
| 误解四条 | head_dim 非整除（DeepSeek-V3 192）/ bias 分野 / tied 的 $Vd$ / block 之外三项 | §8 |

## 练习

1.（计算）MHA 反事实对账：把 Llama-3.1-8B 的 num_key_value_heads 改为 32（其余字段不动），用式 (6) 重算 $N_{\mathrm{total}}$；「8B」名称是否还成立？再把 intermediate_size 改回等参数规则的最小 256 倍数（2.6 §3.2），两个改动叠加后总参数是多少？（提示：attention 项回到 $4d^{2}$，FFN 回到 $3d \times 11008$。）
2.（计算）用 Qwen3-8B config（$L = 36$、$d = 4096$、$H_q = 32$、$H_{kv} = 8$、$D = 128$、$d_{\mathrm{ff}} = 12288$、$V = 151936$、untied，[Qwen3-8B config](../references.md)，[A]）重做 §5 的全表与 §6 的对账：单层 matmul FLOPs、整机 per-token 前向（含输出头）、$N_{\mathrm{total}}$ 精确值；验证层参数 $= 11.5 d^{2} \times 36$ 与 $2N$ 闭环。
3.（代码）在 §3 教学版上：(a) 把 `DecoderBlock` 改成式 (8) 的并行形，断言参数量不变、同输入下输出改变；(b) 参考 2.4 §4.4 给 `GQAAttention` 加 cache，验证「带 mask 的并行前向第 $t$ 位 logits」与「逐 token decode 第 $t$ 步 logits」allclose（fp32、容差 $10^{-5}$）——2.4 §7 命题的整机版；(c) 用 §3 的断言方法对 Qwen3-8B 的 config（练习 2）核对 `sum(p.numel())`。
4.（推导）并行块的梯度：写出式 (8) 的残差展开式，并证明 2.7 式 (8) 的恒等项对它逐字成立（指明 norm 位置是前提）；再写出串行形（式 (1)）中 $f_{\ell}$ 对 $x^{(\ell-1)}$ 的完整 Jacobian 路径（含经 attention 输出的间接项），说明并行形恰好删掉这一项——「block 内复合深度 $2L \to L$」的精确含义。

## 参考

- [Vaswani et al., 2017](../references.md)：Attention Is All You Need，arXiv:1706.03762，[A]（key: transformer）——decoder block 的原始形态（Post-LN + sinusoidal），本章式 (1) 是其现代配方的落点
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——标准配方（RMSNorm + Pre-LN + GQA + RoPE + SwiGLU）出处
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——reordered norm 对照配方
- [Meta, Llama-3.1-8B config.json](../references.md)：[A]（key: llama-3.1-8b-config）——§2 映射表与 §5/§6 全部数值
- [AllenAI, OLMo-2-7B config.json](../references.md)：[A]（key: olmo-2-7b-config）——对照锚点
- [Qwen, Qwen3-8B config.json](../references.md)：[A]（key: qwen3-8b-config）——练习 2 数值
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——head_dim = 192 ≠ hidden/heads 反例（§8 误解一）
- [Radford et al., 2019](../references.md)：[W]（key: gpt2）——bias=True / tied / learned absolute / GELU 的组件层老配方对照（norm 摆放为 Pre-LN，报告 §2.2）
- [Karpathy, nanoGPT](../references.md)：教学参照（key: nanogpt）——GPT-2 配方的教学实现对照
- [HuggingFace, transformers repo Llama modeling 代码](../references.md)：[A]（key: llama-modeling）——§4 走读对象（`LlamaRMSNorm` / `LlamaAttention` / `LlamaMLP` / `LlamaDecoderLayer` / `LlamaRotaryEmbedding`）
- [HuggingFace, transformers repo OLMo 2 modeling 代码](../references.md)：[A]（key: olmo2-modeling）——`q_norm` / `k_norm` 对照


