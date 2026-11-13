# 4.4 Hybrid 架构（linear + full attention 交错）（Hybrid Architectures: Interleaving Linear and Full Attention）

> 本章把 4.3 §10 的 rank 界落成工程形态：固定 $d_v \times d_k$ 状态的精确联想容量受限、KV cache 无界但内存随 $T$ 线性——hybrid 用 $r:1$ 的 linear : full 层交错把 decode 内存的斜率除以 $(r+1)$，检索职责集中给少数 full 层。逐一走读 Qwen3-Next（3:1 GDN : gated attention、zero-centered norm、MTP）、Kimi Linear（3:1 KDA : MLA、full 层 NoPE 的双重论证、cache −75% 与 6.3× 吞吐的论文数字）、Kimi K3（KDA + LatentMoE + Attention Residuals）、Jamba 与 MiniMax-01（7:1），最后给全模型内存统一账与设计空间三旋钮 | 前置：2.4（KV cache 式 (9)）、2.7（zero-centered 增益）、4.1（MLA 576 元素账、RoPE×低秩冲突）、4.3（GDN 式 (14)、KDA 式 (16)、rank 界） | 锚点：Qwen/Qwen3-Next-80B-A3B-Instruct（model card + config + transformers `modeling_qwen3_next.py`）[W]、moonshotai/Kimi-Linear-48B-A3B-Instruct config [A]、Kimi Linear 论文 [A]、MiniMax-Text-01 [A]、Jamba [A]、Kimi K3 [A]、fla-org/flash-linear-attention [A]

## 1. 互补性论证：两条边界的拼接

4.3 §10 的结论：固定状态 $S \in \mathbb{R}^{d_v \times d_k}$ 的单头精确联想容量不超过 $d_k$ 对， $n \gg d_k$ 后误差必然显形；softmax attention 的 KV cache（2.4 式 (9)）逐对原样驻留、检索无结构上限，代价是内存随 $T$ 线性。两个极端的致命项互为对方的强项——hybrid 架构（linear state 层与 full attention 层交错堆叠）把同一模型的 $L$ 层分成两组：

- $L_{\mathrm{lin}}$ 个 linear 层（GDN / KDA / Mamba / lightning）：状态定长、decode 每 token $O(H d_v d_k)$；
- $L_{\mathrm{full}}$ 个 full attention 层：KV cache 随 $T$ 增长、精确检索无结构上限。

decode 单序列内存（ $B$ = 1、元素字节数 $b$）：

$$
M(T) = \underbrace{L_{\mathrm{lin}} \cdot H\thinspace d_v d_k \cdot b}_{\mathrm{fixed}} + \underbrace{L_{\mathrm{full}} \cdot c_{\mathrm{kv}} \cdot T \cdot b}_{\mathrm{growth}} \qquad (1)
$$

**符号**： $H$：linear 层的（value）头数； $d_v, d_k$：状态两轴（4.3 约定布局）； $c_{\mathrm{kv}}$：full 层每 token 每层 KV 元素数（GQA： $2 H_{kv} d_{\mathrm{head}}$，4.1 式 (3)；MLA： $d_c + d_R$，4.1 §4.2）。

交错比 $r:1$（ $r = L_{\mathrm{lin}} / L_{\mathrm{full}}$）把线性项斜率压到全 full 反事实的 $1/(r+1)$：3:1 是 4 倍、7:1 是 8 倍（§7 实数）。注意省的是斜率不是截距——固定项在几千 token 后就被线性项超过，hybrid 的账在长上下文才兑现。

把 4.3 §10 的刻度代入一下，看清「受限」到底限在哪。单头 $d_k$ = 128 时精确联想预算 128 对；随机键下加性状态的读出信噪比约为 $d_k / (n-1)$， $n$ = 1024 时已跌到约 0.125（约 −9 dB）——逐位复制类任务在千 token 量级就出问题，远早于任何长上下文窗口。而同一层换成 KV cache：每对 $(k_s, v_s)$ 原样驻留，读出误差为零，代价是 $T$ = 131072 时单头单层 64 MiB 的读取量（4.3 §3 的 decode 账）。这就是 4.3 §10 结尾「数据库 vs 充分统计量」两个极端的实数形状：一个把内存当检索精度付，一个把检索精度当内存省。hybrid 的赌注是：自然语言负载里需要逐位精确检索的 key 是稀疏的，把检索能力集中到少数层、让多数层只扛压缩后的统计结构，两个极端可以各取所长。

decode 算力一侧同构：linear 层每 token 读出 $+$ 写入约 $4 H d_v d_k$ FLOPs，与 $T$ 无关；full 层每 token 打分约 $4 H_q d_{\mathrm{head}} T$ FLOPs，随 $T$ 线性。hybrid 因此同时压两笔斜率——内存（式 (1)）与 decode FLOPs——而保留的检索能力按层数只剩 $1/(r+1)$ 份。 $r$ 的全部权衡就压在这两个比值上。

检索一侧的机制论证：精确回忆只需 full 层兜底。一项内容只要驻留在某个 full 层的 cache 里，该层就能在任意距离外逐位读回；linear 层在其间充当压缩传输。等距铺开（每 $r+1$ 层一个 full）还有一层回路角度的好处：4.7 的 induction 式「前驱 + 检索」两段回路需要两个 attention 层先后配合，等距布局保证序列任意位置附近都有可用的层对。这是本书的机制读法，不是任一论文的 ablation 结论；各论文的实证边界见 §8。

## 2. 交错模式对照表

五家已发布模型的层配置（config / 论文口径，证据等级见锚点）：

| 模型 | 总层数 | linear : full | linear 层型 | full 层型 | full 层位置 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Next-80B-A3B [W] | 48 | 36 : 12（3:1） | GDN（4.3 式 (14)） | gated attention（输出 sigmoid 门，§3） | `full_attention_interval=4`，每第 4 层 |
| Kimi-Linear-48B-A3B [A] | 27 | 20 : 7（名义 3:1） | KDA（4.3 式 (16)） | MLA + NoPE（§4） | `full_attn_layers=[4,8,…,24,27]`，末层例外 |
| MiniMax-Text-01 [A] | 80 | 70 : 10（7:1） | lightning attention | softmax attention（partial RoPE 64/128） | 每第 8 层（8,16,…,80） |
| Jamba-v0.1 [A] | 32 | 28 : 4（7:1） | Mamba（S6，4.3 §9） | 标准 attention | 每 8 层块各 1 层；MoE 隔层替换 FFN |

表下两注。其一，Kimi Linear 的「3:1」是名义周期：27 层凑不满整数周期，前 24 层严格 3:1（full 在 4,8,…,24），第 27 层额外加一个 full attention（config 的 `linear_attn_config` 两列表实测，[A]）。其二，Kimi K3 的层数与交错比以 report 正文为准（本章 §5 只登记其架构三件套坐标）；同谱系另有 NVIDIA Nemotron-H（Mamba-2 与 attention 交错）——本书 references 未收录其条目，仅登记谱系坐标、不作正文论据。

## 3. Qwen3-Next：3:1 GDN : gated attention

证据等级 [W]：无 arXiv 论文，架构事实以 HF config 与 transformers `modeling_qwen3_next.py` 为准（2026-11-11 实测），训练配方类数字按 model card 口径降级引用。

**布局**。model card 原文：12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))——48 层 = 36 GDN + 12 full attention，每个 mixer 后都跟 MoE。config 字段 `full_attention_interval=4`，transformers 配置代码据此生成 layer_types。MoE 侧：512 experts、top-10、1 个 shared expert（`shared_expert_intermediate_size=512`）；总参数 80B、激活 3B、预训练 15T tokens（均 model card 口径）。

```
一个周期（4 层，model card 布局重绘）：
┌─ GDN ─► MoE ─┐ ┌─ GDN ─► MoE ─┐ ┌─ GDN ─► MoE ─┐ ┌─ Gated Attn ─► MoE ─┐   × 12
  状态 32×128×128 固定                                KV 2×2×256 / token
```

**GDN 侧**。`linear_num_key_heads=16`、`linear_num_value_heads=32`、`linear_key_head_dim = linear_value_head_dim = 128`、`linear_conv_kernel_dim=4`——每 value 头状态 $S \in \mathbb{R}^{128 \times 128}$（4.3 约定布局），32 个 value head 合计每层 1 MiB 定长状态（bf16，§7）。chunkwise 训练直接调 fla 的 `chunk_gated_delta_rule`（transformers 读码确认，4.3 §8 锚点；kernel 状态布局与本书约定互为转置）。

**gated attention 侧**。16 Q heads / 2 KV heads（4.1 口径下 $g$ = 8 的 GQA）、`head_dim=256`、`partial_rotary_factor=0.25`（RoPE 只旋 256 维中的 64 维，2.5 §4.7 的 partial RoPE）。代入 2.4 式 (9)：每 full 层每 token 的 KV 是 $2 \times 2 \times 256$ = 1024 元素——12 层合计 24 KiB / token，是 §7 表的斜率来源。输出带 sigmoid 门：

$$
o_t = \sigma\left(W_g u_t\right) \odot \mathrm{attn}_t \qquad (2)
$$

**符号**： $u_t$：层输入 stream； $\mathrm{attn}_t$：标准 causal attention 输出； $W_g u_t \in \mathbb{R}^{H d_v}$：逐元素门控向量。实现级证据：门控向量由 Q 投影一并切出——`modeling_qwen3_next.py` 的前向是 `attn_output * torch.sigmoid(gate)`，gate 为 q_proj 输出的一段切片。即 2.6 的 GLU 门控思想从 FFN 搬到 attention 输出：逐元素可关断的读出通道。

**zero-centered norm**。`Qwen3NextRMSNorm` 的增益以 `torch.zeros` 初始化、前向为

$$
y = \hat{x} \odot \left( 1 + g \right), \qquad g_0 = 0 \qquad (3)
$$

**符号**： $\hat{x}$：RMS 归一化后的 stream； $g$：逐通道增益。这正是 2.7 的 zero-centered 增益定义——增益存成相对恒等的偏差，weight decay 把 $g$ 拉回恒等映射而非零映射：对 48 层深的 hybrid 主干，这个先验让「逐层接近恒等、增量式塑形」成为衰减动力学的默认方向，与 2.7 的 DeepNorm 讨论同向但实现更轻。另有 `Qwen3NextRMSNormGated`（norm 后再乘一道 SiLU 门），挂在 GDN 输出侧。

**MTP 与窗口**。model card 另列 Multi-Token Prediction（MTP）模块：训练时作为多步预测的辅助 loss，推理时可作 self-speculative 解码的草稿头——目标式与接受率结构统一在 4.5 处理，本章只登记它与层型配比正交这一事实（草稿头挂在主干之上，不占 36 : 12 的层数预算）。上下文原生 262,144（config `max_position_embeddings`、`rope_theta=10^{7}`）、可扩到 1,010,000（model card）；长度泛化手段归 4.6。其余 config 字段一并登记：hidden 2048、vocab 151936（与 Qwen3 同 tokenizer）。

## 4. Kimi Linear：3:1 KDA : MLA 与 full 层 NoPE

证据等级 [A]：论文 arXiv:2510.26692 + 官方 repo MoonshotAI/Kimi-Linear + 开放权重（config 2026-11-11 实测）。

**布局**。27 层 = 20 KDA + 7 full attention：config 的 `linear_attn_config.kda_layers=[1,2,3,5,…,26]`、`full_attn_layers=[4,8,12,16,20,24,27]`（同字段下 `head_dim=128`、`num_heads=32`、`short_conv_kernel_size=4`）。KDA 更新式即 4.3 式 (16)：key 侧逐通道遗忘 $\mathrm{diag}(\alpha_t)$、 $\alpha_t \in \mathbb{R}^{d_k}$，加 delta-rule 定点擦除；20 层合计 20 MiB 定长状态（§7）。MoE 侧：256 experts、top-8、1 shared、sigmoid router、`routed_scaling_factor=2.446`、`first_k_dense_replace=1`（首层 dense）。总参数 48B、激活 3B（论文）。其余 config 字段：hidden 2304、intermediate 9216、`moe_intermediate_size=1024`、`model_max_length=1048576`（1M）、vocab 163840、`tie_word_embeddings=false`。顶层另有 `head_dim=72`、`num_attention_heads=32`、`rope_theta=10^{4}` 字段——full 层的 KV 维度以 `mla_*` 字段为准（下段），顶层字段按 config 实测登记、不作维度论据。

**full-attn 层：MLA + NoPE**。full 层用 MLA（`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`，与 DeepSeek 谱系同字段），但 `mla_use_nope=true`——RoPE 关闭。这个选择回答两个问题：

1. **为什么可行**：位置对称性已经不在了。NoPE 的顾虑本是 attention 对位置盲（2.5 §5；[Kazemnejad et al., 2023](../references.md)，arXiv:2305.19466，[B] 的对照实验），但那条顾虑的前提是「位置编码只能由 attention 层自己注入」。hybrid 里这个前提不成立：stream 在进入每个 full 层之前已被 KDA 层加工过，而 KDA 递推（4.3 式 (16)）对输入顺序非交换—— $S_t$ 由时序连乘 $\prod_s \mathrm{diag}(\alpha_s)(I - \beta_s k_s k_s^{\top})$ 与秩一写入构成，置换输入序列必然改变状态与读出（练习 2 给出严格表述）。位置信息以「学出来的时间尺度」的形式写进 stream 特征，full 层只需按内容检索，顺序由 linear 层兜底。等距 3:1 布局还保证顺序信号的传输距离有界：周期段内任意 full 层与其上游最近的 full 层之间恰隔 3 层 KDA，与最近 KDA 层至多隔 3 层。
2. **为什么想要**：NoPE 拆掉 4.1 §4.1 的 RoPE×低秩冲突。无旋转则 $R_{n-t}$ 不再卡在 $W^{UK}$ 与 $c$ 之间，朴素低秩吸收畅通——cache 保持 $d_c + d_R$ = 576 元素 / token / 层（4.1 §4.2 口径；NoPE 只关旋转、不动 config 字段），decode 直接对 latent 打分，4.1 §4.3-4.4 的两笔吸收（ $\tilde{W}_h$ 与 $\widehat{W}_h$）无需 decoupled 通道的配合即全量成立。在这里 RoPE 是花钱买一份 KDA 已经免费提供的位置信号，还顺手堵死吸收——去掉它是双赢，不是妥协。

**论文数字**（摘要口径）：公平对照实验用 1.4T tokens（Kimi Linear / GDN-H / MLA 三模型共享语料与 recipe），最终发布 checkpoint 用 5.7T tokens（对齐 Moonlight 预训练量，对照见论文 Appendix D）——两档勿混。KV cache 最多 −75%（§7 对账：7/27 层保持 cache，斜率剩 25.9%）；1M 上下文 decode 吞吐 up to 6×（摘要）、精确值 6.3×（Fig 1(b) 的 TPOT 对照）。摘要同时声明这是公平对照下首个全面超过 full attention 的线性注意力架构——注意该结论在 3:1 hybrid 形态下达成，不是纯 KDA（§8 再议）。

## 5. Kimi K3：KDA 谱系的下一档

Kimi K3（arXiv:2607.24653，[A]；权重全量发布）把 4.3-4.4 的 KDA 主线搬进 frontier 规模：2.8T 总参数 / 104B 激活（激活率约 3.7%）、1M 上下文、原生视觉。架构三件套（report 口径；训练与 RL 细节归 training 维度）：

- **KDA 沿用**：mixer 主线不换——§4 的 KDA : full 交错与 NoPE 论证是其直接前身，K3 报告将其作为既定基座继承。
- **LatentMoE**：896 routed experts、每 token 激活 16（激活率约 1.8%，routed 口径）；对照 Kimi Linear 的 256 / top-8 与 Qwen3-Next 的 512 / top-10（两家均含 1 个 shared expert，含 shared 口径激活率约 3.5% 与 2.1%，同 4.5 的 $\rho = (k + K_s)/N_e$ 定义），稀疏度再上一档。「Latent」前缀的机制细节以 report 正文为准，本书不展开。
- **Attention Residuals**：report 在 KDA / full 主干上引入的残差新件；机制细节同样以 report 正文为准——它属于「交错主干之上再叠结构件」的一层自由度（§8 设计空间）。

## 6. Jamba 与 MiniMax-01：SSM 谱系与 7:1

**Jamba**（arXiv:2403.19887，[A]；ai21labs/Jamba-v0.1）：SSM : attention : MoE 三层交错——32 层分 4 块，每块 8 层中 1 层 attention、7 层 Mamba（S6，4.3 §9），MoE 每隔一层替换 FFN（16 experts、top-2）。52B 总参数 / 12B 激活、256K 上下文（论文摘要口径）。三层交错各管一件事：Mamba 层给定长状态扛记忆、attention 层保精确检索、MoE 层在不涨激活算力的前提下扩容量——本章的「linear : full」两旋钮之外，Jamba 把稀疏度（4.5）作为第三个正交旋钮一并交错进同一主干。与 §3-4 的对照点在层型选择：linear 侧是 SSM 谱系（无 key 方向定点擦除，4.3 §9 的谱系划分），full 层不做输出门控、不用 MLA——交错思想相同，层型取自另一代零件箱。

**MiniMax-Text-01**（arXiv:2501.08313，[A]；HF repo 名是 MiniMaxAI/MiniMax-Text-01）：80 层 = 70 lightning attention + 10 full（config `attn_type_list` 实测；full 在每第 8 层，1-indexed 8,16,…,80）——7:1，把交错比推到本章最激进的一档。456B 总 / 45.9B 激活（论文摘要）、32 experts top-2；hidden 6144，full 层 64 Q heads / 8 KV heads、`rotary_dim=64`（128 维里 partial RoPE 旋一半，与 §3 的 0.25 档同族不同刻度）。深度方向的稳定件是 `postnorm=true` 加 DeepNorm 式残差缩放（config `layernorm_*_alpha≈3.557`、`beta=1.0`）——80 层的 hybrid 主干用 2.7 的 DeepNorm 方案压住残差流幅度，与 Qwen3-Next 的 zero-centered 先验是同一深度稳定性问题的两种解法。训练上下文 1M、推理外推 4M（摘要原文）——4M 档的内存账见 §7。lightning attention 是带衰减的 linear attention 变体，与 4.3 §4 的 RetNet 形态同族（并行 / 递推两形式、chunkwise 训练形态同 4.3 §8），状态约定同型。

## 7. 内存账统一表：固定 state vs 线性 KV

把式 (1) 逐项代入各模型 config（ $B$ = 1、bf16、 $b$ = 2）。每层数字：

| 模型 | linear 层每层状态（固定） | full 层每层 KV（每 token） |
| --- | --- | --- |
| Qwen3-Next | 32 v-heads × 128 × 128 = 524,288 元素 = 1 MiB | 2 × 2 × 256 = 1024 元素 = 2 KiB |
| Kimi Linear | 32 heads × 128 × 128 = 1 MiB | MLA：512 + 64 = 576 元素 = 1152 B |
| MiniMax-Text-01 | lightning 状态同型 $H d_v d_k$（config 未实测字段不列数） | 2 × 8 × 128 = 2048 元素 = 4 KiB |

全模型 decode 内存（式 (1)，各取典型窗长）：

| 模型 | 固定项 | 线性项斜率 | 代入窗长 $T$ | 线性项 | 全 full 反事实 | 斜率比 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Next | 36 × 1 MiB = 36 MiB | 12 × 2 KiB = 24 KiB / token | 262,144 | 6.0 GiB | 48 层全 full：96 KiB / token → 24 GiB | 1/4 |
| Kimi Linear | 20 × 1 MiB = 20 MiB | 7 × 1152 B ≈ 7.9 KiB / token | 1,048,576 | 7.9 GiB | 27 层全 MLA：30.4 KiB / token → 30.4 GiB | 7/27 ≈ 1/3.9 |
| MiniMax-Text-01 | — | 10 × 4 KiB = 40 KiB / token | 4,194,304（4M 外推档） | 160 GiB | 80 层全 full：320 KiB / token → 1.25 TiB | 1/8 |

三笔对账。其一，Kimi Linear 的斜率比 7/27 = 25.9% 即 cache 减少 74.1%——论文摘要「up to −75%」的来源就是 3:1 交错的层数比，不是额外的压缩技术。其二，固定项在线性项面前很快退场：Kimi Linear 的每层交叉点约 1 MiB / 1152 B ≈ 910 token，全模型口径约 2600 token 后线性项主导——hybrid 的真实收益是「斜率 ÷ (r+1)」，窗越长兑现越足；反过来，短窗（千 token 内）部署下 hybrid 的内存优势基本不兑现，这时它买到的是 decode 算力与吞吐。其三，decode 吞吐同向放大：1M 上下文下 Kimi Linear 每 token 的 full 层 cache 读取约 7.9 GiB，全 MLA 反事实约 30 GiB（KDA 层每 token 另读 20 × 1 MiB 定长状态，与 $T$ 无关）——读取比值约 3.9× 且随 $T$ 继续拉大；论文实测 6.3×（Fig 1(b)）还包含 kernel 与算力侧因素，按论文口径引。

decode 算力账顺带登记（§1 的第二笔斜率）。linear 层每 token 读出加写入约 $4 H d_v d_k$：Qwen3-Next 每 GDN 层 $4 \times 32 \times 128 \times 128 \approx 2.1$M FLOPs，固定；full 层每 token 打分约 $4 H_q d_{\mathrm{head}} T$ = $4 \times 16 \times 256 \times T$ ≈ $16 T$ K FLOPs， $T$ = 262,144 时约 4.3G FLOPs——单层 full 的打分已是单层 GDN 的两千倍，且差距随 $T$ 线性拉大。hybrid 把 decode 的逐 token 算力同样变成「大固定项 + 小线性项」，这是 6.3× 吞吐数字在内存账之外的另一半账面来源。

## 8. 质量边界与设计空间

**弱项任务族**：精确复制与多 key 精确检索。4.3 §10 的 rank 界是纯 linear 形态的硬约束，hybrid 把它转化为「检索容量集中在 1/(r+1) 的层」。各论文的 ablation 同向：RetNet / GDN / KDA 的检索类对照一致报告纯 linear 落后 full attention（4.3 §10 已登记）；hybrid 形态把差距补回——Kimi Linear 摘要声明公平对照（1.4T 档）下全面超过 MLA 基线，GDN-2 报告 multi-key 检索优势最大（1.3B / 100B tokens 口径，4.3 §6）。强弱边界因此不是「hybrid 能不能检索」，而是「r 能推到多大」：7:1 的 MiniMax-01 把检索职责压到 10 层，其 4M 外推声明（摘要）与 3:1 系的 1M 数字是当前公开坐标的两端；跨模型检索质量的统一横评尚缺（4.6 的 RULER 方法论问题在此同样适用）。

**强项**：长上下文 PPL 与吞吐 / 内存（§7）；另有一笔间接收益——full 层变少后，4.1-4.2 的全部 cache 技术（GQA / MLA / sparse）只需施加在少数层上，工程面收窄。

**设计空间三旋钮**：

1. **比例 $r$**：3:1（Qwen3-Next、Kimi Linear 名义）vs 7:1（MiniMax-01、Jamba）——检索密度与内存斜率的直接交换。公开坐标里尚无系统的 $r$ 扫描：各家停在哪一档是工程判断而非理论结果，这是设计空间里最大的空白格。
2. **位置**：等距周期（interval=4、每第 8 层）vs Kimi Linear 的「周期 + 末层例外」——第 27 层额外 full，输出端最后一道精确检索保险。位置布局的消融数据公开很少：首层 dense / 末层 full 这类「端点加固」是多家共识性手感，但「full 层该集中还是该铺开」没有论文级对照。
3. **门控方式**：linear 侧 GDN 标量门 vs KDA 对角门（4.3 §6-7）；full 侧 Qwen3-Next 的输出 sigmoid 门（式 (2)）vs Kimi 的无门 MLA；位置编码侧 partial RoPE（0.25 / 0.5 档）vs NoPE（§4 论证）——门控与位置编码是 full 层自己的两个子旋钮，与 $r$ 正交。

旋钮之外还有一个层型选择维度：linear 侧可在 4.3 §10 对照表的四档里挑（S6 / RetNet 式固定衰减 / GDN 标量门 / KDA 对角门），full 侧可在 4.1 的 cache 谱系里挑（MHA / GQA / MLA）。本章五家在这张二维网格上的落点互不重合——网格远未填满，填格子的依据应当来自 §7 的账与 §8 前半的边界，而不是谱系惯性。

## 总结

| 内容 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| 互补性 | 式 (1) | rank 界（4.3 §10）+ 线性 cache（2.4 式 (9)）⇒ 斜率 ÷ (r+1)，截距不省 |
| Qwen3-Next | 式 (2)(3)；48 = 36 GDN + 12 gated attn | interval=4；输出 sigmoid 门（gate 切自 q_proj）；zero-centered 增益 $g_0 = 0$；MTP 归 4.5；[W] |
| Kimi Linear | 27 = 20 KDA + 7 MLA-NoPE | full_attn_layers 末层例外；NoPE 双重论证（KDA 已破位置对称 + 拆 RoPE×低秩冲突）；cache −75% = 层数比；decode 6.3×（Fig 1(b)）；训练 1.4T / 5.7T 两档；[A] |
| Kimi K3 | 2.8T / 104B | KDA 沿用 + LatentMoE（896 / 16）+ Attention Residuals（细节以 report 为准）；[A] |
| Jamba | 32 = 28 Mamba + 4 attn；MoE 隔层 16 / top-2 | SSM 谱系交错形态；52B / 12B、256K（摘要）；[A] |
| MiniMax-01 | 80 = 70 lightning + 10 full（7:1） | 456B / 45.9B；1M 训练 / 4M 外推（摘要）；DeepNorm 式 postnorm；[A] |
| 内存统一账 | §7 两表 | 每层 1 MiB 定长状态 vs 每层 0.6-2 KiB / token KV；交叉点约 $10^{3}$ token |
| 设计空间 | $r$ / 位置 / 门控 | 检索密度 ↔ 内存斜率；末层例外；门控与位置编码为 full 层子旋钮 |

## 练习

1.（计算）用式 (1) 复核 Qwen3-Next 在扩展窗 $T$ = 1,010,000 下的全模型 decode 内存；同窗长下「48 层全 GDN」与「48 层全 full」两个反事实各是多少？写出三者的截距与斜率结构，并解释为什么「全 GDN」反事实在实践中不成立（提示：§1 的 rank 界论证）。
2.（推导）Kimi Linear 的 NoPE 论证。设某层 KDA 的读出为 $o_t = S_t q_t$（4.3 式 (16)）。证明：对任意非平凡位置置换 $\pi$，KDA 层输出序列不满足置换等变（与 2.3 的 attention 置换等变性对照），即顺序信息已进入 stream；由此说明「full 层 NoPE 仍可获得位置信息」的完整机制链条，并指出该论证对「full 层放在第 1 层」为何失效。
3.（计算）MiniMax-Text-01 在 4M 外推档、 $B$ = 8 并发下，full 层 cache 总量与单卡 80 GB HBM 的关系；若要塞进单卡，full 层每 token 每层 KV 元素数 $c_{\mathrm{kv}}$ 需压到什么量级？列出可走的 4.1 路线（GQA 组数 / MLA latent 维）各一条。
4.（代码）对照 transformers `modeling_qwen3_next.py` 的 `Qwen3NextAttention` 门控切片与 `Qwen3NextRMSNorm` 前向，写约 30 行教学版复现式 (2)(3)（教学用，非优化实现）；再对 `torch_recurrent_gated_delta_rule` 的状态布局 $(d_k, d_v)$ 与本书约定 $S \in \mathbb{R}^{d_v \times d_k}$ 的差异写一段注记（4.3 §8 布局注记）。

## 参考

- [Qwen, Qwen3-Next-80B-A3B-Instruct model card / blog / config](../references.md)：[W]（key: qwen3-next）——48 层布局、full_attention_interval=4、gated attention 与 zero-centered norm 实现、15T 训练口径
- [Kimi Team, 2025](../references.md)：Kimi Linear: An Expressive, Efficient Attention Architecture，arXiv:2510.26692，[A]（key: kimi-linear）——KDA、cache −75%、decode 6.3×（Fig 1(b)）、1.4T / 5.7T 两档训练
- [Moonshot AI, Kimi-Linear-48B-A3B-Instruct config.json](../references.md)：[A]（key: kimi-linear-config）——20 KDA + 7 full、mla_use_nope=true、MLA 字段
- [MiniMax, 2025](../references.md)：MiniMax-01: Scaling Foundation Models with Lightning Attention，arXiv:2501.08313，[A]（key: minimax01）——7:1 交错、456B / 45.9B、1M / 4M 口径
- [MiniMax, MiniMax-Text-01 config.json](../references.md)：[A]（key: minimax-text-01-config）——attn_type_list 70+10、rotary_dim=64、postnorm
- [Lieber et al., 2024](../references.md)：Jamba: A Hybrid Transformer-Mamba Language Model，arXiv:2403.19887，[A]（key: jamba）——SSM : attention : MoE 交错、52B / 12B
- [Kimi Team, 2026](../references.md)：Kimi K3: Open Frontier Intelligence，arXiv:2607.24653，[A]（key: kimi-k3）——LatentMoE、Attention Residuals、2.8T / 104B
- [Yang et al., 2024](../references.md)：Gated Delta Networks，arXiv:2412.06464，[A]（key: gdn）——Qwen3-Next linear 层的层型出处
- [Kazemnejad et al., 2023](../references.md)：The Impact of Positional Encoding on Length Generalization in Transformers，arXiv:2305.19466，[B]（key: nope）——NoPE 可行性对照
- [fla-org](../references.md)：flash-linear-attention，[A]（key: fla-repo）——`chunk_gated_delta_rule` kernel 锚点
