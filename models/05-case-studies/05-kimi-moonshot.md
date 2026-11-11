# 5.5 Kimi / Moonshot 谱系（The Kimi / Moonshot Lineage）

> Kimi-K2 的 config.json 第一行写着 `architectures: ["DeepseekV3ForCausalLM"]`——不是笔误，是这个谱系最诚实的一个字段：Moonshot 的旗舰站在 DeepSeek-V3 的架构类上，自己的增量集中在三处——Muon 系的训练稳定性技术（Moonlight 的规模化证据、K2 的 MuonClip/QK-clip）、KDA/linear attention 支线（Kimi Linear → K3 的 hybrid 主干）、以及 LatentMoE 一档更激进的稀疏度。本章按 P5 固定模板展开：配置谱系表 → MuonClip 的 qk-clip 公式详拆与 K2/V3 逐字段 diff → 1T 参数复算与 576 元素 cache 账 → K2 config 精读 → 开放问题 | 前置：2.3（softmax 饱和 §3.4）、2.5（RoPE 波长账）、2.6（SwiGLU）、2.7（QK-Norm、soft-capping、weight decay）、4.1（MLA 的 576 元素账）、4.3 §7（KDA 式 (16)）、4.4（hybrid 内存账、Kimi Linear 与 K3 走读）、4.5（sigmoid 路由、shared expert、aux-loss-free）、4.6（YaRN）、5.2（DeepSeek-V3 config 精读）、5.3 §2.3（upcycling 归属） | 锚点：MoonshotAI/Kimi-K2 [A]、moonshotai/Kimi-K2-Instruct config [A]、MoonshotAI/Moonlight [A]、MoonshotAI/Kimi-Linear [A]；论文：Moonlight（Muon is Scalable）2502.16982 [A]、K1.5 2501.12599 [A]、K2 2507.20534 [A]、Kimi Linear 2510.26692 [A]、K3 2607.24653 [A]；Muon 原始出处 Keller Jordan 博客 [W]

## 1. 配置谱系表

谱系的公开技术面从 Moonlight 开始：更早的 Kimi 服务期（长上下文产品形态）无论文、无公开 config，证据等级低于 [C]，本章不登记其架构字段——谱系表只填有出处的格子。字段口径按 P5 约定（config 实测优先；total/active 照抄官方摘要口径，各家是否含 embedding 等分母差异未对齐，不自行换算）：

| 模型 | total/active | layers | hidden | heads Q/KV | head_dim | attention 方案 | MoE | norm | vocab | ctx | rope_theta | 证据 key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Moonlight-16B-A3B | 16B / 3B（摘要口径） | — | — | — | — | —（config 本轮未实测） | 含 shared expert + sigmoid 路由（ar5iv 取证） | — | — | — | — | moonlight [A]、moonlight-repo [A] |
| Kimi k1.5 | —（RL 论文，模型侧字段未在锁定材料） | — | — | — | — | — | — | — | — | — | — | kimi-k1.5 [A]（RL 出处） |
| Kimi-K2（Instruct） | 1T / 32B（摘要口径） | 61 | 7168 | 64（MLA 名义） | 128 + 64 | MLA | 384+1 top-8，首 1 层 dense | RMSNorm（架构类继承，§4） | 163840 | 131072（YaRN f32 on 4096） | $5 \times 10^{4}$ | kimi-k2 [A] + kimi-k2-config [A] |
| Kimi-Linear-48B-A3B | 48B / 3B | 27 = 20 KDA + 7 full | 2304 | 32（KDA 侧）/ MLA（full 侧） | 128（KDA）/ 576 latent（full） | KDA : MLA-NoPE，名义 3:1（末层例外） | 256+1 top-8，首 1 层 dense，sigmoid router | — | 163840 | 1048576 | $10^{4}$（顶层字段） | kimi-linear [A] + kimi-linear-config [A] |
| Kimi K3 | 2.8T / 104B（report 口径） | — | — | — | — | KDA 沿用 + Attention Residuals | LatentMoE 896 routed / 16 active | — | — | 1M | — | kimi-k3 [A] |

表注四条。其一，「—」= 本轮 research pass（2026-11-12）未锁定的字段；Moonlight 行的架构事实只有三条经 ar5iv 全文取证（sigmoid routing、shared expert、SwiGLU 各出现 ≥2 次），逐字段 config 未纳入实测序列。其二，K2 行的 heads 写「名义」：MLA 不逐头存 KV，cache 是一份共享 latent（4.1 §4），`num_key_value_heads` 的同值是占位——5.2 §1 的脚注一在此原样适用。其三，head_dim 列的 128 + 64 = qk_nope + qk_rope（score 内维 192，v 侧 128 另计）——与 DeepSeek-V3 完全同构，2.8 §8「head_dim $\neq$ hidden/heads」的反例在 K2 上再演一次：7168 / 64 = 112，与 192 无关。其四，Kimi Linear 的「3:1」是名义周期：27 层里前 24 层严格 3:1，第 27 层额外加一个 full attention（config `linear_attn_config` 两列表实测，4.4 §1 表注）。

竖着读两列。vocab 列：K2 与 Kimi Linear 共享 163840——谱系内 tokenizer 口径统一，与 Qwen 四代共用 151936（5.3 §3）是同一做法：词表是谱系里最黏的字段。ctx 列：131072（K2，YaRN 扩展）→ 1048576（Kimi Linear，hybrid 结构撑 1M）→ 1M（K3）——长窗主战场由 full-attention 谱系移交 hybrid 谱系，机制原因正是 4.4 §7 的斜率账（full 层 cache 随 $T$ 线性，KDA 层定长）。

证据等级横着读也值得一句：全表五个模型全部 [A]——论文、官方 repo、开放权重三齐，这是 P5 六章里证据等级最整齐的一张谱系表（对照 5.1 的 Llama-4 [W]、5.3 的 Qwen3-Max [C]、5.4 的 GLM-4.6/4.7 [W]）。含义是本章每个架构断言都能回指论文原文或 config 实测；唯一的 [W] 条目是 Muon 的原始出处（Keller Jordan 博客，无 arXiv），而它影响的是训练侧细节，不影响任何架构字段的定级。

## 2. 标志性架构决策的公式级分析

### 2.1 Moonlight：一篇 optimizer 论文里的 MoE 配方

先正名：Moonlight 的出处论文标题是 *Muon is Scalable for LLM Training*（[Liu et al., 2025](../references.md)，arXiv:2502.16982，[A]；官方 repo MoonshotAI/Moonlight [A]）——论文的主角是优化器，Moonlight-16B-A3B 是它的规模化证据件：3B 激活 / 16B 总参的 MoE，5.7T tokens 训练（摘要口径）。Muon 本身的原始出处是 Keller Jordan 的博客（[W]，无 arXiv）：对矩阵形参数的动量更新做正交化（Newton–Schulz 迭代把更新矩阵的奇异值推向 1），公式细节与超参归 training 维度，本章只引出处。模型侧要登记的是：这篇 optimizer 论文顺手公开了 Moonshot 当时的 MoE 配方取点——sigmoid routing、shared expert、SwiGLU expert（ar5iv 全文关键词取证，各 ≥2 次）。MoE 层输出照 4.5 的框架写：

$$
y_t = \sum_{i \in \mathrm{TopK}(s_t)} g_{i,t}\, E_i(u_t) + E_{\mathrm{shared}}(u_t), \qquad s_{i,t} = \sigma(u_t^{\top} e_i) \qquad (1)
$$

**符号**：$u_t \in \mathbb{R}^{d}$：MoE 层输入；$e_i \in \mathbb{R}^{d}$：router 的第 $i$ 个 expert 嵌入；$s_{i,t}$：sigmoid 亲和度（4.5 式 (16)，全 expert 归一化竞争推迟到 top-k 之后）；$g_{i,t}$：选中集内归一化后的门控权重（4.5 式 (17)）；$E_i$：第 $i$ 个 routed expert 的 SwiGLU FFN（2.6 式 (7) 三矩阵形）；$E_{\mathrm{shared}}$：常驻激活的 shared expert（4.5 §3.1，捕捉跨 token 公共知识、让 routed expert 专注差异化）。三件事都是 DeepSeekMoE→V3 谱系先立的项（4.5 式 (14)–(16)），Moonlight 的取点说明 2025 年初这套配方已是跨实验室的公共知识——5.2 的「架构基座」观察在此先垫一笔，§2.3 的 K2 把它推到旗舰规模。

归属纠正（与 5.3 §2.3 互查）：dense→MoE 的 upcycling（dense FFN 复制成 $N_e$ 份 expert 初始化）常被记到 Moonlight 名下；Moonlight 论文全文 0 次提及 upcycling（W3 research pass 逐文核实），公开谱系里的旗舰级实例是 Qwen2-57B-A14B（upcycle 自 Qwen1.5-32B，公式与对称性代价分析见 5.3 式 (4)）。Moonlight 的 16B-A3B 是否 from-scratch 训练，论文口径以 Muon 可扩展性为主线，初始化路径未在锁定材料中——列入 §5。

另记一笔谱系内回流：Kimi Linear 论文的发布档 checkpoint 用 5.7T tokens，正是「对齐 Moonlight 预训练量」（4.4 §4，论文 Appendix D 口径）——Moonlight 的训练量成了谱系内后续公平对照的基准刻度。

### 2.2 MuonClip：qk-logit 越阈时缩放 $\gamma$ 的事后控制器

K2 摘要给出的稳定性答案是 MuonClip = Muon + QK-clip（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]，摘要原文「a novel QK-clip technique to address training instability」；成效口径：15.5T tokens 全程 zero loss spike，摘要声明）。先看它要管的问题（2.3 §3.4 与 2.7 §7 已立）：attention logit

$$
z_{ij}^{h} = \frac{\langle q_i^{h},\ k_j^{h} \rangle}{\sqrt{d_h}} = \frac{1}{\sqrt{d_h}}\, u_i^{\top} \bigl(W_Q^{h}\bigr)^{\top} W_K^{h}\, u_j \qquad (2)
$$

**符号**：$h$：head 序号；$u_i \in \mathbb{R}^{d}$：位置 $i$ 的层输入；$W_Q^{h}, W_K^{h} \in \mathbb{R}^{d_h \times d}$：该头的 Q/K 投影；$d_h$：头内维。$z$ 对 $(W_Q^{h}, W_K^{h})$ 是**联合二次**的（对各自分别是线性）——这条代数性质决定下面的缩放方案。训练中 $q, k$ 的幅度无机制封顶时，$|z|$ 持续长大，softmax 滑进饱和区，Jacobian $\mathrm{diag}(p) - p p^{\top}$ 的行梯度按 $p_i(\delta_{ij} - p_j)$ 消失——logit 失控链条（2.3 §3.4）。

QK-clip 的机制骨架（逐头、步后、非梯度）：监控各头当前 logit 的峰值

$$
S_{\max}^{h} = \max_{i,\, j}\ z_{ij}^{h} \qquad (3)
$$

越阈 $\tau$ 即触发，触发头两侧的投影矩阵按同一因子收缩：

$$
\gamma^{h} = \min\left(1,\ \frac{\tau}{S_{\max}^{h}}\right), \qquad W_Q^{h} \leftarrow \sqrt{\gamma^{h}}\, W_Q^{h}, \qquad W_K^{h} \leftarrow \sqrt{\gamma^{h}}\, W_K^{h} \qquad (4)
$$

对账一行：$z$ 对两侧各是线性，$\sqrt{\gamma^{h}} \times \sqrt{\gamma^{h}} = \gamma^{h}$，触发后新峰值 $\gamma^{h} S_{\max}^{h} = \tau$，未触发头 $\gamma^{h} = 1$ 为恒等。$\sqrt{\gamma}$ 均摊而非单侧缩放（单侧 $W_Q \leftarrow \gamma W_Q$ 对 logit 等价）的理由是两侧范数同比收缩、避免一侧矩阵范数被单独压塌（练习 2 讨论两方案在 weight decay 下的长期差异）。注意缩放是逐通道各向同性的：head 内所有 $q, k$ 方向同比例缩小，logit 的相对结构不变，只是该头的 attention 分布被整体「降温」$\gamma^{h}$ 倍、熵抬升。

把式 (3)(4) 放进时间维度读：$S_{\max}^{h}$ 是一条随训练演化的轨迹，式 (4) 是它的反射壁——越界即被精确压回 $\tau$，未越界时优化器照常工作、可以把 logit 再度养大。稳态因此不是不动点而是贴壁振荡，触发频率由高走低的曲线对应「logit 漂移速度对监控周期」的赛跑结果。这与 4.5 §3.2 (i) 对 aux-loss-free 偏置的极限环分析同型：控制回路类药物的稳态都不是点，而是围绕目标的带。MuonClip 的名字由此也读得通——Muon 管更新几何（动量更新正交化、谱形受控），QK-clip 管 logit 幅值的事后投影，「优化器 + 控制器」两段拼成一个词。触发频率的实测曲线本轮未锁定（§5 开放问题）；摘要的 zero loss spike 声明（15.5T tokens 全程，[A]）按声明口径登记，本书不作独立验证。

这个设计的谱系位置要说准——它是**第三种 logit 稳定路线**，与 2.7 的两条既有路线同一目标、不同作用点：

| 路线 | 出处 | 作用点 | 介入时机 | 上界形状 | 对 forward 的侵入 |
| --- | --- | --- | --- | --- | --- |
| QK-Norm | [Henry et al., 2020](../references.md)，arXiv:2010.04245，[B]；Qwen3 实例 5.3 §2.2 | $q, k$ 表示（归一化） | 恒时、每个前向 | $\vert z \vert \le \sqrt{d_k}$（结构性，与输入无关） | 每步改写表示，幅度信息从 score 移除 |
| soft-capping | Gemma 2 独占（[Gemma Team, 2024](../references.md)，arXiv:2408.00118，[A]）；Gemma 3 已弃 | logit 函数形状 $c\tanh(z/c)$ | 恒时、每个前向 | $\vert z' \vert < c$（tanh 值域） | 每步压缩 score，尾部梯度衰减 |
| QK-clip | K2（[A]） | 参数 $W_Q, W_K$ | 事件驱动（越阈才触发）、步后 | $\tau$（触发时精确压回） | 不触发时 forward 逐位不变 |

哲学对照一并登记：QK-clip 不进损失函数、无梯度路径、按监控信号行事——与 4.5 式 (17)(18) 的 aux-loss-free 偏置控制器是同一设计哲学（DeepSeek-V3 把负载均衡从优化目标降级为控制回路，K2 把 logit 稳定从 norm 结构手里交给事后投影；本书注）。MuonClip 的命名也读得通：Muon 的更新经正交化后谱形受控，但谱形受控不直接给出 $\lVert W_Q \rVert$ 累计漂移的上界，QK-clip 补的是这一缺口的**效果侧**——不改更新规则，改参数的事后投影，监控信号取 logit 而非范数。阈值 $\tau$ 的取值、监控 batch 的构造、触发频率随训练的衰减曲线，本轮未在锁定材料中——以 report 正文为准，列入 §5。

为什么这条路线出现在这个谱系而不是别处，账在 §2.3：K2 的 config 未引入 QK-Norm 类字段（谱系表 norm 列无 QK-Norm 记录），稳定性答案整个压在 MuonClip 上；而 QK-Norm 路线在 Qwen3（5.3 §2.2）与 Gemma 3（2.7 §7）那里已经走通。两条路线的逐项质量消融公开材料未见——§5。

### 2.3 K2 = DeepSeek-V3 架构类的第二次实例化

`architectures: ["DeepseekV3ForCausalLM"]` 意味着 transformers 加载 K2 权重时跑的是 DeepSeek 的 modeling 代码——MLA + DeepSeekMoE 的组合从「一家旗舰」变成「跨实验室架构基座」（5.2 §6 拓扑注记的回收）。把 5.2 §6 的逐字段账与 K2 config 实测并排（K2 侧为 2026-11-12 实测，[A]）：

| 字段 | DeepSeek-V3（5.2 §6） | Kimi-K2 | 机制读法 |
| --- | --- | --- | --- |
| `num_hidden_layers` | 61 | 61 | 层数原样 |
| `hidden_size` | 7168 | 7168 | stream 宽度原样 |
| `num_attention_heads` | 128 | 64 | head 数减半；MLA cache 与头数无关（576 元素，§3.2），省的是 q/o 投影参数与 score 侧 FLOPs |
| MLA 五字段 | kv 512 / q 1536 / nope 128 / rope 64 / v 128 | 逐字段相同 | cache 几何与 4.1 §4 的账整体继承 |
| MoE | 256+1 top-8 | 384+1 top-8 | routed expert +50%，中间维 2048 不变——总参增量的主来源（§3.1） |
| `first_k_dense_replace` | 3 | 1 | dense 前导层 3 → 1：5.2 §7 开放问题的另一端点取点 |
| `vocab_size` | 129280 | 163840 | 词表加大（与 Kimi Linear 共享 tokenizer 口径，§1） |
| `rope_theta` | $10^{4}$ | $5 \times 10^{4}$ | 最慢通道波长 $\lambda_{\max} \sim 2\pi \cdot 5 \times 10^{4}$（2.5 §6 波长账），配合 YaRN |
| `rope_scaling` | YaRN f40 on 4096 → 163840 | YaRN f32 on 4096 → 131072 | 同一配方不同刻度（4.6 §2.3）；$4096 \times 32 = 131072$ 恰为 `max_position_embeddings` |

未列入本表的字段（`scoring_func`、`n_group`、`routed_scaling_factor`、`intermediate_size` 等）本轮未对 K2 config 锁定——不臆填，这就是 5.6 方法论章「config 未实测字段不臆填」纪律的实例。

两行读法。其一，K2 相对 V3 的增量预算几乎全投在 MoE 宽度上：384 vs 256 个 expert 是 1T 对 671B 总参差的主项（§3.1 复算），其余骨架字段按兵不动——在验证过的架构类上加宽稀疏度，是风险最低的规模推档方式。其二，agentic 定位（论文标题 *Open Agentic Intelligence*，摘要口径）在架构字段上的痕迹不在新层型，而在容量/激活比与长窗：agentic 工作负载 = 多轮工具调用的长上下文与状态管理（4.8 的架构含义分析），32B 激活把每 token 的 decode 成本钉在 V3 同档（37B 口径邻域），1T 容量与 131072 窗口服务的是负载侧。

K1.5 一句定位（[Kimi Team, 2025](../references.md)，arXiv:2501.12599，[A]）：大规模 RL 配方论文（long-CoT 与 RL 规模化），架构字段未在锁定材料中——差异在训练侧，归 training 维度，本章不展开。

### 2.4 Kimi Linear / KDA：谱系内的 hybrid 支线（recap）

推导一律回指，此处只登记谱系坐标。KDA（Kimi Delta Attention）把 GDN 的标量遗忘门换成 key 侧逐通道对角门（4.3 §7 式 (16)：$S_t = S_{t-1}\, \mathrm{diag}(\alpha_t) \left( I - \beta_t k_t k_t^{\top} \right) + \beta_t v_t k_t^{\top}$，$\alpha_t \in (0,1)^{d_k}$），一层之内可同时容纳长记忆槽位与即抛工作区；chunkwise kernel 走 DPLR 特化（fla 库锚点）。Kimi Linear 把它装进 27 层 hybrid：20 KDA + 7 full（full 层为 MLA + NoPE——NoPE 的双重论证在 4.4 §4：KDA 递推已破位置对称性，且拆掉 RoPE×低秩冲突后 MLA 吸收全量成立）；cache 账 −75% 就是层数比 7/27 ≈ 25.9% 的直接读数，1M ctx decode 吞吐 6.3×（论文 Fig 1(b)）——细账与对账在 4.4 §7，不重算。账摘要照 4.4 §7 的口径抄一份：20 个 KDA 层合计 20 MiB 定长状态（与 $T$ 无关），7 个 full 层每 token 共 7.9 KiB——1M 上下文整窗 7.9 GiB，全 MLA 反事实 30.4 GiB，读取比值约 3.9× 且随 $T$ 继续拉大。这条支线回答的正是谱系表 ctx 列的问题：1M 窗口的成本结构不允许全量 full attention，hybrid 是把窗口推进一位数量级的结构条件。训练两档口径勿混：公平对照 1.4T tokens / 发布档 5.7T tokens（对齐 Moonlight，§2.1）。

### 2.5 K3：坐标登记

K3（[Kimi Team, 2026](../references.md)，arXiv:2607.24653，[A]；权重全量发布）把 KDA 主线推到 frontier 规模：2.8T 总参 / 104B 激活（激活率约 3.7%）、1M 上下文、原生多模态（native vision）。架构三件套（report 口径）：KDA 沿用（§2.4 的 KDA : full 交错是其直接前身）；LatentMoE——896 routed experts、每 token 激活 16（routed 口径激活率约 1.8%，对照 4.5 的 $\rho = (k + K_s)/N_e$ 与 K2 的 9/385 ≈ 2.3%）；Attention Residuals——KDA / full 主干之上的残差新件。后两件与 4.4 §5 同口径：机制细节以 report 正文为准，本书不展开。训练与 RL 细节归 training 维度。

## 3. K2 的账：1T 复算与 576 元素 cache

### 3.1 从 config 字段复算 1T

「1T total / 32B activated」是摘要口径（[A]），config 字段足以复算到百分位。单 expert 参数与 V3 完全相同（SwiGLU 三矩阵，$3 \times 7168 \times 2048 \approx 44.0$ M）；MLA attention 单层逐项（$q$ 低秩 $7168 \times 1536 + 1536 \times 12288$；$kv$ 下投影 $7168 \times 576$；$k$ 上投影 $512 \times 12288$；$v$ 上投影 $512 \times 8192$；$o$ 投影 $8192 \times 7168$）≈ 103.2 M（4.1 §4.2 的逐项账同法，head 数 64 代入）：

```
Kimi-K2 层栈（61 层，d = 7168；architectures = DeepseekV3ForCausalLM）
┌─ 层 1：dense —— MLA（cache 576 元素/token/层）+ SwiGLU FFN（intermediate_size 未锁定）
├─ 层 2–61：MoE × 60 —— MLA + router（384 选 8）+ 385 个 expert（d_ff = 2048）
└─ 训练期控制回路：MuonClip —— 逐头监控 S_max，越阈 τ 则 W_Q、W_K 同乘 √γ（式 (4)，非梯度）
```

| 块 | 算术 | 参数 |
| --- | --- | --- |
| 60 个 MoE 层 | $60 \times 385 \times 44.0$ M | ≈ 1017.3 B |
| attention × 61 | $61 \times 103.2$ M | ≈ 6.3 B |
| router × 60 | $60 \times 7168 \times 384$ | ≈ 0.17 B |
| embedding | $163840 \times 7168$ | ≈ 1.17 B（tie 字段未锁定，若 untied 再 ×2） |
| 首层 dense FFN | intermediate_size 未锁定 | 记余项（若同 V3 的 18432：≈ 0.40 B） |
| 合计 | | ≈ 1025–1026 B ≈ 1T ✓ |

读数：1T 里 99% 以上是 expert 参数——attention、router、词表加起来是零头的零头。「1T 模型」的准确图像是「23100 个 44M 的 FFN 加一套 61 层 MLA」，这是 4.5 §1 参数/计算解耦框架的极端实例。激活侧（每 token）：MoE 层 9 个 expert（8 routed + 1 shared）$60 \times 9 \times 44.0$ M ≈ 23.8 B，加 attention 6.3 B、router 0.17 B、首层 dense 余项 ≈ 30.3 B；官方口径「32B activated」（摘要，[A]）照抄不换算，差额即官方对词表侧等项的计法（W3 约定页口径条款）。MoE 层内稀疏度 $\rho = 9/385 \approx 2.3\%$——介于 DeepSeek-V3（9/257 ≈ 3.5%）与 Qwen3-Next（11/513 ≈ 2.1%）之间，K3 的 LatentMoE 再压到约 1.8%（routed 口径 16/896）。

### 3.2 cache 账：与 V3 同一个 9.2 GB

MLA 的 cache 与头数无关（4.1 §4.2）：每 token 每层存 $d_c + d_R = 512 + 64 = 576$ 元素。K2 的总量：

$$
M_{\mathrm{KV}} = L \cdot (d_c + d_R) \cdot T \cdot b \qquad (5)
$$

**符号**：$L$ = 61；$T$：序列长度；$b$ = 2（bf16）。每 token $61 \times 576 \times 2$ B ≈ 68.6 KiB；$T$ = 131072 整窗 ≈ 9.2 GB——与 5.2 §4 的 V3 数字逐项相同（同 $L$、同 576、同窗口量级）。这就是 §2.3 heads 减半读法的另一面：128 → 64 省的是参数与 FLOPs，cache 一分钱没省——MLA 的 cache 由 latent 秩（512）与 rope 维（64）决定，与 head 数解耦。反事实对照留给练习 3（同尺寸 MHA 的 20480 元素 / token / 层、35.6 倍差）。

decode 侧再补一笔（2.4 §5 的算术强度框架）：每生成一个 token，61 层的 cache 要被完整读一遍——$T$ = 131072 时即每步约 9.2 GB 的 HBM 读取，这是长窗 decode 单序列每 token 延迟的带宽下限；heads 减半不影响这笔读取，它省在投影 FLOPs 上。要压每步读取，只有两条结构路线：压每层 cache 的几何（MLA 的 576 已是这一路的成果），或压每步要读的层数与行数——后者即 hybrid（Kimi Linear / K3，4.4 §7）与稀疏检索（DeepSeek-V3.2 的 DSA，4.2 §6 与 5.2 §4）两条路。两条路的代表恰好都在本章的邻域里，这是 Kimi 谱系与 DeepSeek 谱系在长窗成本上的分工图。

## 4. 旗舰 config 精读：moonshotai/Kimi-K2-Instruct config.json

按 2.8 §2 的方法读：每个字段在公式里点名位置。本轮实测字段摘录（教学用摘录，非完整文件；2026-11-12 实测）：

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 64,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 384,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,
  "first_k_dense_replace": 1,
  "vocab_size": 163840,
  "max_position_embeddings": 131072,
  "rope_theta": 50000,
  "rope_scaling": {"type": "yarn", "factor": 32.0,
                   "original_max_position_embeddings": 4096}
}
```

**骨架字段**。hidden 7168 / 61 层 / vocab 163840——内部结构 = 1 个 dense 前导层 + 60 个 MoE 层。`architectures` 一行是全文最重字段：架构类复用 DeepSeek-V3（§2.3），norm、FFN 形、残差拓扑等未实测字段由该类继承（谱系表 norm 列的 RMSNorm 即此推断，标注为推断而非实测）。

**MLA 块**。五字段与 V3 逐值相同（q 1536 / kv 512 / nope 128 / rope 64 / v 128）——4.1 式 (5)–(9) 的账原样成立：$c^{KV} \in \mathbb{R}^{512}$，score 内维 $128 + 64 = 192$，cache 576 元素 / token / 层。差异只有 `num_attention_heads = 64`（V3 为 128）：score 侧计算与 q/o 投影减半，cache 不动（§3.2）。质量侧代价（检索头数减半）无公开消融，列入 §5。

**MoE 块**。384 routed + 1 shared、top-8、中间维 2048：单 expert 44.0 M，每 token 激活 9 个 expert。`first_k_dense_replace = 1`：只有第 1 层 dense——与 V3 / GLM-4.5 的 3 形成取点对照（5.2 §6 的两条机制读法：浅层表示未分化、router 早期均衡振荡；Moonshot 的判断是把这层保险压到最低限度 1 层，Kimi Linear 同取 1）。`scoring_func` 等路由细节字段本轮未锁定（§2.3 表注）；Moonlight 的 sigmoid routing 取证（§2.1）与架构类继承给出方向性线索，但 K2 的 router 字段值本身不臆填。

**位置块**。`rope_theta = 5e4` + YaRN factor 32 on 4096 → 131072：与 V3 的「$10^{4}$ + f40 → 163840」同配方不同刻度（4.6 §2.3 的逐通道 ramp 与 mscale 温度修正同族适用）。注意 K2 的原生窗只有 4096，131072 整窗靠 YaRN 外推——工作窗口径与 Kimi Linear 的 1M（hybrid 结构账，4.4 §7）是两条不同的长窗路线，同谱系并存。

**未出现而值得点名的字段**：无 QK-Norm 对应配置项、无 logit soft-capping 参数——logit 稳定三路线（§2.2 表）里 K2 的选择不在 config 里，在训练循环的事后控制器里。「稳定性技术不一定有 config 签名」本身就是 config 读法的一课（5.6 方法论章回收）。

## 5. 技术开放问题清单

1. **QK-clip 的工作点未锁定**：阈值 $\tau$ 取值、监控 batch 构造、触发频率随训练的衰减曲线，本轮未在锁定材料中（以 report 正文为准）；它与 QK-Norm 路线的逐项质量/稳定性消融公开材料未见——三路线对照表（§2.2）目前是机制层对照，不是实证层对照。
2. **Moonlight 的初始化路径未披露**：from-scratch 还是 upcycle 类初始化未在锁定材料；upcycling 的归属与公式归 Qwen2（5.3 §2.3），勿张冠李戴。
3. **heads 128 → 64 的质量账**：MLA cache 不变（§3.2），动机是参数与 score 侧 FLOPs；检索头数减半的质量代价无公开消融——4.1 §4.2「latent 秩兜底检索分辨率」的论证是否覆盖头数减半，开放。
4. **`first_k_dense_replace` 的端点取点**：1（K2、Kimi Linear）对 3（V3、GLM-4.5）——5.2 §7 的开放问题在 K2 侧多一个数据点，但跨家对照仍非受控消融。
5. **LatentMoE 与 Attention Residuals 的机制细节**：K3 两件套以 report 正文为准（4.4 §5 同口径）；896 / 16 的稀疏度是否逼近路由与通信开销的反折点（4.5 §6 的进行时问题），无公开扫描。
6. **谱系的 optimizer 路径未完整公开**：Moonlight 立 Muon 可扩展性、K2 立 MuonClip；K1.5 / Kimi Linear / K3 各代的优化器与稳定性配方细节归 training 维度或未披露。

## 总结

| 内容 | 公式 / 数字 | 要点 |
| --- | --- | --- |
| 谱系表 | §1 表（13 列约定） | 公开技术面自 Moonlight 起；vocab 163840 谱系内共享；长窗主战场移交 hybrid 支线 |
| Moonlight | 式 (1)；16B / 3B、5.7T tokens | 论文正名 *Muon is Scalable for LLM Training*；配方取点 = sigmoid 路由 + shared expert + SwiGLU；upcycling 归 Qwen2（→ 5.3 §2.3） |
| MuonClip / QK-clip | 式 (2)(3)(4)：$\gamma^{h} = \min(1, \tau / S_{\max}^{h})$，双侧乘 $\sqrt{\gamma^{h}}$ | 逐头、步后、非梯度的事后控制器；与 QK-Norm / soft-capping 并列第三路线；15.5T tokens zero loss spike（摘要口径） |
| K2 = V3 架构类 | §2.3 diff 表；`architectures = DeepseekV3ForCausalLM` | MLA + DeepSeekMoE 成跨实验室基座；增量预算主投 MoE 宽度（256 → 384）与词表 |
| 1T 复算 | §3.1 表：MoE ≈ 1017.3 B + attn ≈ 6.3 B | 99% 是 expert 参数；激活 ≈ 30.3 B 对官方 32B 口径；$\rho = 9/385 \approx 2.3\%$ |
| cache 账 | 式 (5)：576 元素 / token / 层，整窗 ≈ 9.2 GB | MLA cache 与头数无关——heads 减半省参数与 FLOPs，不省 cache |
| Kimi Linear / KDA | recap → 4.3 式 (16)、4.4 §4/§7 | 20 KDA + 7 MLA-NoPE；cache −75% = 层数比；decode 6.3×；5.7T 对齐 Moonlight |
| K3 | 2.8T / 104B；LatentMoE 896/16；Attention Residuals；1M ctx；native vision | 坐标登记，机制以 report 正文为准（→ 4.4 §5） |
| 开放问题 | §5 六条 | QK-clip 工作点、初始化路径、heads 质量账、dense 前导端点、K3 两件套机制、optimizer 路径 |

## 练习

1.（计算）参数复算。(a) 由 §4 字段重算 §3.1：验证 60 个 MoE 层 ≈ 1017.3 B、attention ≈ 6.3 B（103.2 M 逐项列出）；(b) 重算激活侧 ≈ 30.3 B，列出与官方 32B 口径的差额可能落在哪些项；(c) 若 `num_experts_per_tok` 从 8 提到 16（其余不变），激活参数与 $\rho$ 各变为多少？
2.（推导）QK-clip。(a) 由式 (2) 证明 $z_{ij}^{h}$ 对 $(W_Q^{h}, W_K^{h})$ 联合二次、对各自线性；(b) 验证式 (4) 触发后新峰值恰为 $\tau$、未触发头为恒等；(c) 设改为单侧缩放 $W_Q^{h} \leftarrow \gamma^{h} W_Q^{h}$：证明 logit 效果相同，并结合 2.7 的 weight decay 讨论两方案长期范数轨迹的定性差异（提示：衰减把两侧范数往零拉，单侧缩放在下一步衰减后两侧范数比如何变化）。
3.（计算）cache 对照。K2 若换成同尺寸 MHA（64 头、qk 192 维、v 128 维、其余不变）：(a) 每 token 每层 cache 是多少元素，对 576 元素是几倍？(b) $T$ = 131072、bf16 下整窗各为多少 GiB？(c) 用 4.1 §4 的机制说明为什么 MLA 的这一比值与 head 数无关。
4.（代码）config diff 教学代码（教学用，非优化实现）：把 5.2 §6 与本章 §4 的两段 config 摘录存成两个字典，写约 20 行代码输出逐字段 diff 表，并按「架构字段（层型/MLA/路由机制）」「规模字段（expert 数/vocab/heads）」「配方字段（rope_scaling）」三类给每个差异字段打标签——用输出复核 §2.3 表的一行结论。

## 参考

- [Liu et al., 2025](../references.md)：Muon is Scalable for LLM Training（Moonlight 模型出处论文），arXiv:2502.16982，[A]（key: moonlight）——16B/3B、5.7T tokens、sigmoid routing、shared expert、SwiGLU 取证
- [Keller Jordan](../references.md)：Muon: An optimizer for hidden layers in neural networks（博客，无 arXiv），[W]（key: muon-blog）——Muon 原始出处，公式归 training 维度
- [Moonshot AI, MoonshotAI/Moonlight](../references.md)：[A]（key: moonlight-repo）——Moonlight-16B-A3B 权重与 Muon 分布式实现
- [Kimi Team, 2025](../references.md)：Kimi k1.5: Scaling Reinforcement Learning with LLMs，arXiv:2501.12599，[A]（key: kimi-k1.5）——RL 定位，归 training 维度
- [Kimi Team, 2025](../references.md)：Kimi K2: Open Agentic Intelligence，arXiv:2507.20534，[A]（key: kimi-k2）——1T/32B、MuonClip/QK-clip、15.5T tokens zero loss spike、agentic 定位（摘要口径）
- [Moonshot AI, Kimi-K2-Instruct config.json](../references.md)：[A]（key: kimi-k2-config）——§4 精读对象：DeepseekV3ForCausalLM、384+1 top-8、MLA 五字段、YaRN f32
- [Moonshot AI, MoonshotAI/Kimi-K2](../references.md)：[A]（key: kimi-k2-repo）——K2 model card 与部署说明
- [Kimi Team, 2025](../references.md)：Kimi Linear: An Expressive, Efficient Attention Architecture，arXiv:2510.26692，[A]（key: kimi-linear）——KDA 出处；cache −75%、decode 6.3×、1.4T / 5.7T 两档
- [Moonshot AI, Kimi-Linear-48B-A3B-Instruct config.json](../references.md)：[A]（key: kimi-linear-config）——20 KDA + 7 full、mla_use_nope、vocab 163840
- [Kimi Team, 2026](../references.md)：Kimi K3: Open Frontier Intelligence，arXiv:2607.24653，[A]（key: kimi-k3）——2.8T/104B、LatentMoE 896/16、Attention Residuals、1M ctx、native vision
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——架构类来源；sigmoid 路由 4.5 式 (16)
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——§2.3 diff 的 V3 侧
- [Dai et al., 2024](../references.md)：DeepSeekMoE，arXiv:2401.06066，[A]（key: deepseekmoe）——shared expert 与细粒度切分公式出处
- [Yang et al., 2024](../references.md)：Qwen2 Technical Report，arXiv:2407.10671，[A]（key: qwen2）——upcycling 正确归属（5.3 §2.3）
- [Henry et al., 2020](../references.md)：Query-Key Normalization for Transformers，arXiv:2010.04245，[B]（key: qk-norm）——§2.2 对照路线一
- [Gemma Team, 2024](../references.md)：Gemma 2，arXiv:2408.00118，[A]（key: gemma2）——§2.2 对照路线二（soft-capping）
- [Peng et al., 2023](../references.md)：YaRN，arXiv:2309.00071，[A]（key: yarn）——§4 位置块配方出处
