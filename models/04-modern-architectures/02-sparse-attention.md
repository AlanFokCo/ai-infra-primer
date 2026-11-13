# 4.2 稀疏注意力（Sparse Attention）

> 本章把「改可见集」这条省算路线推到底：分类学两轴（静态 pattern vs 动态选择、token 级 vs 块级）；SWA 的窗口账与 $L \cdot w$ 感受野归纳推导（Mistral-7B 实例）；SWA+full 混合层的 Gemma 3 5:1 走读（sliding_window_pattern=6）；attention sink 的 softmax 归一化解释与 StreamingLLM 处方；NSA 的 compression / selection / sliding 三分支公式（[B]：无官方 repo，按论文拆）；DSA 的 lightning indexer → top-2048 → MLA 只在选中位置算的链路，与 sparse-from-dense 训练配方；MoBA 的 query 级块路由与 SWA+MoBA hybrid；以召回类任务的质量风险收尾 | 前置：2.3（softmax 饱和与 Jacobian 式 (8)）、2.4（可见集式 (2)、cache 式 (9)、decode 强度式 (11)(12)）、2.5（RoPE 两种维度配对约定）、4.1（MLA cache 账 576 元素 / token / 层、FlashAttention 不改数学的定位） | 锚点：Mistral-7B（arXiv:2310.06825）[A]、Gemma 3（arXiv:2503.19786，HF gated 挂论文条目）[A]、StreamingLLM（arXiv:2309.17453）[A]、NSA（arXiv:2502.11089）[B]、DeepSeek-V3.2-Exp（arXiv:2512.02556）[A]、MoonshotAI/MoBA（arXiv:2502.13189）[B]、Longformer（arXiv:2004.05150）[A]

## 1. 账的位置：改可见集，不改 softmax

2.4 式 (2) 把 masked attention 写成可见集形式： $\alpha_{ij} = \exp(s_{ij}) / \sum_{l \in V(i)} \exp(s_{il})$， $j \in V(i)$。稀疏注意力（sparse attention）只做一件事——把 causal 全量可见集 $V(i) = \lbrace1, \ldots, i\rbrace$ 收缩成子集，score、softmax、QKV 投影、KV cache 的写法全部不动。这正是本章与前后两章的分界线：4.1 的 FlashAttention 不改数学（输出逐点相同），4.3 换掉 softmax 核（线性化），本章改的是「谁参与归一化」。

收缩可见集买到三笔账：

- **训练 / prefill 算力**：score 对数从 $T(T+1)/2$ 降到 $\sum_i |V(i)|$，逐对成本每头 $4 d_{\mathrm{head}}$ FLOPs（2.3 §7 口径）；
- **decode 每步读取**：2.4 式 (11) 分母的 cache 项只保留可见集对应的行；
- **cache 容量**（仅当可见集有界）：滑窗类把 $T$ 换成常数 $w$，cache 封顶，见 §2。

代价只有一类，但致命：可见集外的 token 对输出的贡献恒为精确的 0——精确检索负载（needle 类）一旦目标落在可见集外，该层没有任何机制能把它找回来。§8 处理这笔风险账。

**分类学两轴**。轴一：静态 pattern（ $V(i)$ 只依赖位置坐标、编译期确定——滑窗、扩张窗、固定全局 token；谱系起点是 Longformer 的「滑窗 + 任务相关 global token」[A]）vs 动态选择（ $V(i)$ 依赖输入内容、运行时打分取 top-K——NSA 的 selection、DSA 的 indexer、MoBA 的块路由）。轴二：粒度——token 级（DSA）vs 块级（NSA、MoBA；块是连续显存读取与 tensor core tile 的天然单位）。第三个性质单列：训练需求——StreamingLLM 免训练插入，NSA 与主模型从头联合训练，DSA 从 dense checkpoint 继续训，MoBA 声明可免训练插入。

两轴的差别不只是算法口味。静态 pattern 编译期可知：显存布局、kernel 形状、cache 淘汰策略都可以在加载时一次定死，系统的全部优化空间随之打开。动态选择的可见集到运行时才知道，打分器本身成为新的成本项与新的故障点——§6 会看到 DSA 为打分器付出的续训管线（约一万亿 token）正是这笔成本的真实量级。反过来，静态 pattern 无法随内容调整可见域：局部性假设失效的负载（长程检索、跨文档引用）上，静态方案只能靠 §3 那种周期性全局层硬撑。

| 方法 | 可见集结构 | 动态？ | 粒度 | 每 query 的 key 位量级 | 训练 |
| --- | --- | --- | --- | --- | --- |
| SWA（Mistral-7B） | 最近 $w$ 个 | 否 | token | $O(w)$ | 从头训 |
| SWA+full 混合（Gemma 3） | 5/6 层滑窗 + 1/6 层全量 | 否 | token | 混合 | 从头训 |
| sink+滑窗（StreamingLLM） | 首 4 个 + 最近 $w$ 个 | 否 | token | $O(w)$ | 免训插入 |
| NSA | 压缩全局 + top-$`n`$ 块 + 滑窗 | 是（块分数） | 块 | $O(t/d + n l' + w)$ | 原生联合训练 |
| DSA（DeepSeek-V3.2） | indexer top-2048 | 是（打分器） | token | $O(K)$ + indexer 全量打分 | 从 dense 续训 |
| MoBA | top-$`k`$ 块（+SWA hybrid） | 是（块路由） | 块 | $O(k \cdot n_b)$ | 免训插入声明 |

## 2. SWA：滑窗、三笔账与 $L \cdot w$ 感受野推导

滑窗 attention（sliding window attention，SWA）把可见集钉在最近 $w$ 个位置上（ $w$ 含自身）：

$$
V(i) = \lbrace \max(1,\thinspace i - w + 1), \ldots, i \rbrace \qquad (1)
$$

**符号**： $w$：窗口宽度（每层每头相同）； $i$：query 位置。可见集是编译期确定的位置函数——轴一上的纯静态端点。

pattern 与逐层信息扩散（ $T = 8$、 $w$ = 3；行 = query、列 = key，■ = 可见）：

```
  1 2 3 4 5 6 7 8      第 1 层：位置 8 直接看到 {6,7,8}
1 ■                    第 2 层：位置 8 读位置 6 的输出，
2 ■ ■                        而位置 6 在第 1 层看过 {4,5,6}
3 ■ ■ ■                      ⇒ 间接覆盖到 4
4 · ■ ■ ■                每多一层，最远可见位置左移 w-1 = 2
5 · · ■ ■ ■
6 · · · ■ ■ ■
7 · · · · ■ ■ ■
8 · · · · · ■ ■ ■
```

**感受野推导**。设 $\rho_\ell(t)$ 为位置 $t$ 在第 $\ell$ 层输出中可依赖的最早位置（信息可达下界）。对 $\ell$ 归纳：

- 基例 $\ell = 0$： $\rho_0(t) = t$（embedding 只含自身）；
- 归纳步：第 $\ell$ 层 attention 读第 $\ell - 1$ 层 $[t - w + 1,\ t]$ 各位置的输出（FFN、norm、残差逐位置，不扩展依赖），故

$$
\rho_\ell(t) = \rho_{\ell-1}(t - w + 1) \qquad (2)
$$

展开 $L$ 层得 $\rho_L(t) = t - L(w-1)$，即感受野大小

$$
R = L(w-1) + 1 \approx L \cdot w \qquad (3)
$$

**符号**： $L$：层数； $R$：位置 $t$ 的输出可依赖的位置总数。距离超过 $L(w-1)$ 的依赖在计算图里不存在——不是近似误差，是没有边。

**Mistral-7B 实例**（[Jiang et al., 2023](../references.md)，arXiv:2310.06825，[A]）： $L$ = 32、sliding_window = 4096。式 (3) 精确值 $R$ = $32 \times 4095 + 1$ = 131041；论文按 $L \cdot w$ = 131072 的口径写作「约 131K」——两者差 $L - 1$ 个位置，工程口径忽略。对照：dense 层 $R$ = $T$，SWA 用 32 层才把单个 query 的可见域撑到 131K，且越靠远端链路越长、被逐层稀释。

三笔账（Mistral-7B 口径： $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16，[A]）：

1. **训练 attention FLOPs**： $T \gg w$ 时 score 对数 $\approx T w$， $L$ 层合计 $4 L T w d$（ $d$ = $H_q d_{\mathrm{head}}$）；full causal 为 $2 L T^2 d$（1.2 式 (6) 口径）。比值 $2w / T$： $T$ = 32768 处为 25%， $T$ = 131072 处为 6.25%——窗口是常数，节省比例随 $T$ 线性放大。
2. **cache 封顶**：可见集有界 ⇒ ring buffer 只留最近 $w$ 行，式 (4) 把 2.4 式 (9) 的 $T$ 换成 $\min(T, w)$：

$$
\mathrm{Mem}_{\mathrm{SWA}} = 2 \cdot L \cdot H_{kv} \cdot d_{\mathrm{head}} \cdot \min(T, w) \cdot B \cdot b \qquad (4)
$$

代入： $2 \times 32 \times 8 \times 128 \times 4096 \times 2$ B = 512 MiB 每序列，与 $T$ 无关；full cache 在 $T$ = 32768 处是 4 GiB、131072 处 16 GiB。SWA 把「cache 随生成线性增长」这一条直接删除。
3. **decode 每步读取**：同式 (4) 的 $\min(T, w)$ 项，强度账（2.4 式 (11)）分母里的 cache 项封顶。

实现注记两条。其一，ring buffer 下 RoPE 仍按绝对位置旋转后写入（2.4 §4.2 的注记不变），buffer 槽位取位置对 $w$ 取模——槽位复用安全，因为被覆盖的列已滑出可见集；Mistral-7B 的 rolling buffer cache 即此形态（[A]，论文设计描述）。其二，扩张窗（dilated window，Longformer 已含）是静态轴上的常见变体：窗内按固定步长抽样可见，感受野乘以扩张率，代价是相邻分辨率变差——本章不展开，谱系归入静态轴。

SWA 不承诺长程检索——它承诺的是「局部结构精确、长程结构靠层堆叠稀释扩散」。需要点对点的长程精确召回时，静态窗口必须让位给全局机制：§3 的周期全量层、§4 的 sink 保底、§5–§7 的动态选择，都是这个缺口的不同补法。谱系起点 Longformer（[Beltagy et al., 2020](../references.md)，arXiv:2004.05150，[A]）即「滑窗 + 任务相关 global token」：默认局部，点名位置（如分类头的 [CLS]）双向全通——把「哪些位置值得全局可见」写死进 pattern，是静态轴上的最早答案。

## 3. SWA+full 混合层：Gemma 3 的 5:1 走读

混合层模式把 §2 的缺口补在架构里：大多数层滑窗（局部、便宜），按固定周期插入全量层（全局、 $O(T^2)$ 但层数少）。Gemma 3 取 5:1 local:global（[Gemma Team, 2025](../references.md)，arXiv:2503.19786，[A]；HF 仓库 gated，数值挂论文条目）：

- **层交错**：每 6 层一组，前 5 层 local（SWA），第 6 层 global（full causal）；HF config 的实现字段是 `sliding_window_pattern = 6`——「每第 6 层不套窗口」。local 层窗口 1024（论文口径）。
- **RoPE 分工**（论文口径）：global 层 base 提到 $10^{6}$、local 层保持 $10^{4}$。机制读法：local 层见过的相对距离最远就是窗口 1024，而 $10^{4}$ base 的最长波长约 $2\pi \times 10^{4}$、数十倍覆盖窗口——local 层不需要长波长余量；长距离外推预算全部留给 global 层。与 4.6 的 base-波长账同源，此处只登记分工结构。

信息流的形状：每个 local 组在 $w$ 邻域内做局部聚合 → 周期末的 global 层把各邻域的汇总广播到全前缀 → 下一个 local 组在更新后的表示上再局部细化。式 (3) 的 $L(w-1)$ 上限被 global 层周期性重置——任意位置对任意 query 的依赖链最长只需穿过 $\le 5$ 个 local 层加 1 个 global 层。全局层放在组末而非组首的机制读法：local 组先各自聚合、global 一次性广播的是新鲜汇总，下一组读到的是已更新的全前缀表示。

混合比与 attention FLOPs 账：每 $p$ 层一组（Gemma 3： $p$ = 6），组内成本 $(p-1) \cdot 4Twd + 2T^{2}d$，全 dense 基线 $p \cdot 2T^{2}d$，比值

$$
r = \frac{1}{p} + \frac{2(p-1)\thinspace w}{p\thinspace T} \qquad (5)
$$

**符号**： $r$：混合层 attention FLOPs 相对全 dense 的比例。代入 $p$ = 6、 $w$ = 1024、 $T$ = 131072： $r \approx 0.167 + 0.013 \approx 0.18$——六分之一的全局层把长程依赖保住，attention 总算力仍只有全 dense 的约 18%。cache 账同构：local 层按式 (4) 封顶 $w$ = 1024，global 层全量 $T$，cache 由占比 $1/p$ 的 global 层主导。

与 4.4 的对照（本章交付的谱系坐标之一）：Gemma 3 是「softmax 局部 + softmax 全局」，4.4 的 Qwen3-Next / Kimi-Linear 是「linear state 局部 + softmax 全局」——同一条「廉价局部层 + 稀疏全局层」的混合骨架，局部层的核不同；encoder 侧的 ModernBERT 2:1（每 3 层 1 个 global，3.2 章）是同一骨架在双向 mask 下的形态。混合比（5:1 vs 2:1 vs 4.4 的 3:1 linear:full）是各家用质量-算力账定出来的工作点，不是原理约束。

## 4. Attention sinks：softmax 归一化压力的出口

现象（StreamingLLM，[Xiao et al., 2023](../references.md)，arXiv:2309.17453，[A]）：给训练好的 full-attention 模型套上纯滑窗 cache，一旦文本长度超过窗口、首 token 被驱逐，PPL 立刻爆炸；保留开头少数几个 token 的 KV（论文工作点 4 个，称 attention sink）加滑窗，流式到 4M token PPL 稳定（论文数字）。驱逐几个看似无语义的句首 token，为什么会崩？答案在 softmax 的归一化约束里，分三段。

**第一段：「不 attend」不可表达**。2.4 式 (2) 的可见集形式下，每一行权重满足 $\sum_{s \in V(t)} p_{t,s} = 1$——行和恒为 1，softmax 没有「全体置零」这个选项。当某 query 在当前上下文里没有任何相关 key 时，质量必须倒给某个位置；最优选择是倒给一个「无害」位置：其 $v$ 对输出的污染最小。

**第二段：首 token 是天然的无害倾倒点**。把带 sink 的输出分解（sink 占质量 $\pi$）：

$$
o_t = \pi \cdot v_1 + (1 - \pi) \cdot \bar{v}_{\mathrm{rest}}, \qquad \bar{v}_{\mathrm{rest}} = \frac{1}{1 - \pi} \sum_{s \ge 2} p_{t,s} v_s \qquad (6)
$$

**符号**： $\pi = p_{t,1}$：倒在首 token 上的质量； $`\bar{v}_{\mathrm{rest}}`$：其余位置的加权均值。两个旋钮合读： $\pi \to 1$ 把「无关但非零」的 $`\bar{v}_{\mathrm{rest}}`$ 压掉， $\Vert v_1\Vert \to 0$ 让 sink 自身的贡献消失——两头都推到时 $o_t \to 0$，这个 head 对该 query 近似静默，而均匀摊开（softmax 的另一默认解）只能给出全体 value 的均值，既不接近零也不受控。训练因此有明确梯度方向：给「无相关 key」的 query 抬升 $\langle q, k_1 \rangle$、压低 $\Vert v_1\Vert$。为什么恰好是位置 1：causal mask 下 $1 \in V(t)$ 对一切 $t$ 成立——它是唯一在所有行、所有层、整条训练数据里都待在归一化域内的位置，是唯一能被稳定塑造成「公共默认 key」的候选。论文的观测与此一致：sink 质量集中在开头几个 token，与其语义无关。

**第三段：驱逐 = 抽掉归一化的出口，剩余列等比放大**。设一行 logit 为 $z_1, \ldots, z_t$， $Z = \sum_j e^{z_j}$， $Z' = \sum_{j \ge 2} e^{z_j}$。驱逐位置 1 后 $`p'_s = e^{z_s} / Z'`$；由 $1 - p_1 = Z' / Z$ 得

$$
p'_s = \frac{p_s}{1 - p_1}, \qquad s \ge 2 \qquad (7)
$$

**符号**： $p_s$：驱逐前权重。 $p_1$ 接近 1（sink 吸收了该行大部分质量）时放大因子 $1/(1-p_1)$ 巨大——原本被 sink 压住的一整行低分 key 被等比抬到主导，输出分布剧变。这不是「窗口不够大」的检索失败，是归一化结构崩塌。

**StreamingLLM 处方**（[A]，论文数字）：cache = 首 4 个 token + 最近 $w$ 个的滑窗；RoPE 按 cache 内坐标重编号（保持相对距离在训练分布内）。4 而不是 1 的机制读法：不同 head 学到的倾倒方向不同——论文观察到 sink 质量分散在开头若干 token 上（首 token、句首标点等），只保 1 个等于删掉一部分头的出口，那些行的权重仍被部分重分配，PPL 部分恶化；4 个覆盖多数头的倾倒点。显式替代：训练时前置一个专职可学 sink token，把隐含约定变成结构（论文验证可行）。

sink 还有一层副作用值得登记：2.3 §3 的饱和分析里，行内最大 score 越高、其余列的相对权重被压得越低——sink 行天然处于半饱和态，该行对其余列的有效检索分辨率下降。这不是 bug：sink 质量高的行本来就是模型标出的「近似 no-op」行，它打算少做事；真正要检索的行不会把大量质量倒进 sink。

设计含义是普适的：一切驱逐早期 token 的稀疏 / 流式方案都必须回答「归一化压力往哪倒」。§5–§7 的动态方案里，sink 位置通常在打分器下天然高分（它们的 $k$ 被训练成对任意 $q$ 都有高 logit），入选概率高——但这是涌现性质，不是结构保证；显式保留 sink 位是更稳的工程默认。

## 5. NSA：原生可训的三分支稀疏 [B]

NSA（Native Sparse Attention，[Yuan et al., 2025](../references.md)，arXiv:2502.11089，**[B]：无官方 repo——deepseek-ai/NSA 不存在，本节公式按论文拆**）把可见集拆成三条互补支路，学一组门控合并：

```
                 query q_t
                   │
 ┌─────────────────┼───────────────────┐
 │ compression     │ selection         │ sliding
 │ 块 MLP 压缩      │ 块分数=压缩支路     │ 最近 w 个
 │ (块长 l 步长 d)  │ 权重聚合→top-n 块   │ 原始 K/V
 │ → ~t/d 个压缩 key│ → 选中块原始 K/V    │
 ▼                 ▼                   ▼
 o^cmp(全局粗粒度)  o^slc(选择细粒度)    o^win(局部细粒度)
 └─────── g^cmp ───┴── g^slc ──┬── g^win ┘   门控（MLP+sigmoid，逐头）
                               ▼
                              o_t
```

**Compression（全局粗粒度）**。键值序列按块长 $l$、步长 $d$ 切块（ $d < l$ 时块间重叠），每块经 MLP 压成单向量：

$$
\tilde{k}^{cmp}_i = f^{cmp}_K\left(k_{(i-1)d+1}, \ldots, k_{(i-1)d+l}\right), \qquad o^{cmp}_t = \mathrm{Attn}\left(q_t, \tilde{K}^{cmp}_{:t}, \tilde{V}^{cmp}_{:t}\right) \qquad (8)
$$

**符号**： $`f^{cmp}_K`$：MLP，输入为块内 $l$ 个 key 的拼接加块内位置编码，输出单个 $d_{\mathrm{head}}$ 维压缩 key（ $\tilde{v}^{cmp}$ 同构）；压缩序列长约 $t / d$——全局信息以 $d$ 倍稀释的分辨率全员在场，needle 的「有没有、大概在哪个块」由这条支路回答。

**Selection（选择细粒度）**。细粒度块（块长 $l'$、不重叠）的重要性分数从 compression 支路的真实 attention 权重聚合而来（对齐情形 $l = d = l'$ 时直接复用 $`p^{cmp}_t`$；一般情形对重叠的压缩块求和），取 top-$`n`$ 块、用块内**原始** K/V 算精确 attention：

$$
p^{slc}_t[j] = \sum_{i:\ \mathrm{block}_i\thinspace \cap\thinspace \mathrm{sel}_j \neq \varnothing} p^{cmp}_t[i], \qquad \mathcal{I}_t = \operatorname{TopN}\left( p^{slc}_t,\ n \right) \qquad (9)
$$

$$
o^{slc}_t = \mathrm{Attn}\left( q_t,\ \lbrace K_j, V_j\rbrace_{j \in \mathcal{I}_t} \right) \qquad (10)
$$

**符号**： $`p^{cmp}_t = \mathrm{softmax}(q_t^{\top} \tilde{K}^{cmp}_{:t})`$：compression 支路的权重行（逐头）； $`\mathcal{I}_t`$：得分最高的 $n$ 个细块下标。用「粗粒度支路真实打过的分」驱动细粒度选择，选择器与主路共享梯度信号——这是 NSA「native」的第一层含义。

**Sliding（局部细粒度）**。 $`o^{win}_t = \mathrm{Attn}(q_t, K_{t-w+1:t}, V_{t-w+1:t})`$，近处精确召回不依赖任何打分器。

**门控合并**：

$$
o_t = g^{cmp}_t \cdot o^{cmp}_t + g^{slc}_t \cdot o^{slc}_t + g^{win}_t \cdot o^{win}_t \qquad (11)
$$

**符号**： $g_t \in \mathbb{R}^{3}$（逐头各一组）由层输入经 MLP + sigmoid 产生。三条支路全部可微，与主模型从头联合训练——「native」的第二层含义：稀疏模式参与训练动力学，不是推理时补丁。

**硬件对齐**（论文设计口径）：selection 块长取 64——对齐 GPU tensor core tile 粒度与连续显存读取；且 GQA 组内全部 query 头共享同一份块选择（group-consistent），选中块整块载入、一次 gather-matmul 服务全组，把「动态选择」做成 kernel 友好的规则访存。

论文工作点（[B]，按论文引）： $l$ = 32、 $d$ = 16、 $l'$ = 64、 $n$ = 16、 $w$ = 512。代入每 query 的 key 位账：compression $t/16$ + selection $16 \times 64$ + sliding $512$； $t$ = 131072 时约 $8192 + 1024 + 512 \approx 9.7\text{K}$，为全量 131K 的 7.4%。注意 compression 项仍随 $t$ 线性涨——NSA 对全局的处理是稀释，不是删除。

与 4.3 的对照能看清 compression 支路的定位：线性 attention 把全部历史压成固定大小的状态矩阵（不随 $t$ 涨），NSA 的压缩序列长 $t/d$ 仍随 $t$ 线性涨——前者付固定容量的信息论上界（associative recall 受限，4.3 §末），后者付线性增长的读取账但保留「按内容把局部放大」的通道：selection 支路就是在压缩图上按内容买回分辨率的机制。

## 6. DSA：lightning indexer + 稀疏化 MLA 的工业落地

DeepSeek-V3.2-Exp（[DeepSeek-AI, 2025](../references.md)，arXiv:2512.02556，[A]；官方 repo deepseek-ai/DeepSeek-V3.2-Exp）把选择器做成一个独立轻量网络（lightning indexer），主 attention 沿用 4.1 的 MLA——cache 格式不变（每 token 每层 576 元素），变的是每步读哪几行：

```
q_t ──► lightning indexer（H^I 个小维度头，ReLU 打分，可 FP8）
            │  对全部历史逐 token 打分 I_{t,s}
            ▼
        TopK（K = 2048）──► 选中下标集 S_t
                                │
MLA cache（全量驻留：c^KV [512] + k^R [64] / token / 层）
                                │  只 gather 选中 2048 行
                                ▼
                    MLA(q_t; {c^KV_s, k^R_s}_{s∈S_t}) ──► o_t
```

**打分**。indexer 对每个历史位置逐 token 打分（论文公式，[A]）：

$$
I_{t,s} = \sum_{j=1}^{H^{I}} w^{I}_{t,j} \cdot \mathrm{ReLU}\left( \langle q^{I}_{t,j},\ k^{I}_s \rangle \right) \qquad (12)
$$

**符号**： $H^{I}$：indexer 头数； $`q^{I}_{t,j} \in \mathbb{R}^{d_I}`$：query 位置 $t$ 在第 $j$ 个 indexer 头上的投影； $`k^{I}_s \in \mathbb{R}^{d_I}`$：token $s$ 的 indexer key（头间共享一份）； $`w^{I}_{t,j}`$：query 侧产生的逐头标量权重。ReLU 取代 softmax——逐点打分免归一化、对吞吐友好，且整条 indexer 链路可按 FP8 执行（论文口径）；小维度 + 低精度使「对全历史打分」这件听起来 $O(T^2)$ 的事在常数上远低于 MLA 主路（打分维度 $d_I$ 对主路的 $d_c + d_R$ = 576）。indexer 自带一份小维 key cache（ $d_I$ 维 FP8 / token），字节远小于主 cache。

**选择与主路**。取分数最高的 $K$ 个位置，MLA 只在选中集上算：

$$
\mathcal{S}_t = \operatorname{TopK}\left( \lbrace I_{t,s}\rbrace_{s < t},\ K \right), \qquad K = 2048 \qquad (13)
$$

**符号**： $K$ = 2048 为 sparse 训练阶段每 query 的选中数（论文原文口径）。主路即 4.1 式 (9)(10) 的 MLA，求和域从 $\lbrace1..t\rbrace$ 换成 $`\mathcal{S}_t`$。

**账**。训练 / prefill：主 attention 从 $O(T^{2})$ 降到 $O(T \cdot K)$，indexer 打分 $O(T^{2} d_I)$ 但常数小、可 FP8。decode（61 层 MLA、 $t$ = 131072）：每步主路 cache 读取从全量 131072 行降到 2048 行——1.5625%；字节数 $61 \times 2048 \times 576 \times 2$ B ≈ 144 MB / 步，全量对照 $61 \times 131072 \times 576 \times 2$ B ≈ 9.2 GB / 步。**cache 总量不变**（全量 576 元素 / token 仍驻留，选中是每步的运行时行为），减的是带宽与 FLOPs——正是 2.4 式 (11) 分母里 cache 项的削减。论文报告长上下文场景端到端显著加速（[A]；论文正文为定性表述加逐长度成本曲线，按 Fig 3 曲线读数约削减一半）。

**sparse-from-dense 训练配方**（从 DeepSeek-V3.1-Terminus 的 128K dense checkpoint 出发，[A]，论文口径）。起点选型本身就是设计：V3.1-Terminus 已是 128K 上下文就绪的 dense MLA 模型，sparse 化只动「读哪些」、不动「怎么存」——MLA cache 格式（576 元素 / token / 层）全程不变，续训路线因此不必触碰 4.1 的压缩结构。两阶段：

1. **Dense warm-up**：冻结主模型全部参数，只训 indexer。目标是让 indexer 的分布逼近主 attention 的分布（论文口径）：对每个 query $t$，把主模型全部 attention 头在历史位置上的注意力分数跨头求和聚合，再 L1 归一化成全模型共享的单一目标分布 $p_{t,:}$；indexer 侧取 $\mathrm{softmax}(I_{t,:})$，loss 为逐位置 KL

$$
p_{t,s} = \frac{\sum_{h} A^{(h)}_{t,s}}{\sum_{s'}\thinspace \sum_{h} A^{(h)}_{t,s'}}, \qquad \mathcal{L}_{\mathrm{warm}} = \sum_{t} D_{KL}\left( p_{t,:} \thinspace\Vert\thinspace \mathrm{softmax}(I_{t,:}) \right) \qquad (14)
$$

**符号**： $`A^{(h)}_{t,s}`$：主模型第 $h$ 头在位置对 $(t, s)$ 上的注意力分数；先跨头求和、再对历史位置 L1 归一化。注意聚合次序不可换：KL 对第一参数非线性，「先头聚合再取 KL」与「逐头 KL 再求和」不等价——论文取前者，target 是全模型一份的 $p_{t,:}$ 而不是逐头一份。

主模型不动 ⇒ dense 行为零扰动，indexer 学会「主模型会看哪里」——稀疏化的信息瓶颈被打分器蒸馏预先填平。
2. **Sparse 训练**：解冻全模型、top-K 生效，端到端语言建模继续训：lr 7.3e-6、15000 步、每步 480 条 128K 序列。顺手一记规模账： $480 \times 131072 \approx 6.3 \times 10^{7}$ token / 步，15000 步合计约 $9.4 \times 10^{11}$ token——sparse 化改造的数据预算接近一万亿 token，「免训练」在这条路线上不成立。indexer 在这一阶段的监督也未断流：top-K 选择本身不可微，语言建模 loss 的梯度到不了打分器——indexer 继续吃式 (14) 同形的 KL 对齐（target 同为头聚合分布，截断到选中集），始终被锚在「主模型真实在看哪里」上，而不是随主模型漂移。

实现勘误一条（官方 repo README 公告，2025-11-17）：indexer 的 RoPE 输入用 non-interleaved 布局，MLA 主路用 interleaved——2.5 的两种维度配对约定在同一模型里并存，写 kernel 时混用即错。

与 NSA 的对照：粒度上 DSA 是 token 级选择（NSA selection 是块级）；打分器上 DSA 用独立可学小网络 + KL 蒸馏（NSA 复用 compression 支路的真实权重）；共同点是选择器都可学、都从主模型的行为里取监督信号。

## 7. MoBA：query 级块路由

MoBA（[Lu et al., 2025](../references.md)，arXiv:2502.13189，**[B]：官方 repo MoonshotAI/MoBA 仅代码、无开放权重，本节公式按论文拆**）把 MoE 的路由结构搬进可见集选择：历史分块、块即「专家」、query 对每块打分、top-$`k`$ 块入选。历史序列按块长 $n_b$ 切块，打分用块内 key 的均值池化向量：

$$
\bar{k}_i = \frac{1}{n_b} \sum_{k \in \mathrm{block}_i} k, \qquad s_i = \langle q_t,\ \bar{k}_i \rangle, \qquad \mathcal{I}_t = \operatorname{TopK}\left( \lbrace s_i\rbrace,\ k \right) \qquad (15)
$$

$$
o_t = \mathrm{softmax}\left( \frac{q_t\thinspace K_{\mathcal{I}_t}^{\top}}{\sqrt{d_{\mathrm{head}}}} \right) V_{\mathcal{I}_t} \qquad (16)
$$

**符号**： $`\bar{k}_i \in \mathbb{R}^{d_{\mathrm{head}}}`$：块 $i$ 的广告向量（均值池化）； $s_i$：query 对块 $i$ 的路由分； $`K_{\mathcal{I}_t}, V_{\mathcal{I}_t}`$：选中块内全部原始 K/V 的拼接。因果性两条硬约束：含未来位置的块不进候选集；query 当前所在块恒入选（局部连续性免费保底，不靠打分器）。

三个设计要点。**免训练插入**（论文声明）：打分器零参数（一次均值池化加一次内积），可直接套在已训好的 full-attention 模型上；继续训练进一步缩小与 full 的差距。**kernel 兼容**：块是连续内存，选中块拼接后就是一段变长 dense attention，FlashAttention 的 varlen 接口直接承接——动态选择与 4.1 的精确 kernel 正交叠加。**SWA+MoBA hybrid**：近处滑窗恒可见、远处走块路由，把「近处召回不依赖打分器质量」做成结构保证（论文给出的部署形态）。

与 DSA 的对照把谱系补完：MoBA 的打分器是零参数的均值内积，表达力限于「块均值方向与 query 对齐」；DSA 的 indexer 是可学多头网络，能蒸馏主模型的真实注意力行为，代价是训练管线（§6 的一万亿 token）。粒度上块级选择命中即召回整块——块内分辨率免费，块间分辨率就是打分器的上限。

## 8. 质量风险：召回是稀疏的软肋

把 §1 的那笔代价账落成失败模式分类与保底结构盘点。

**失败模式三类**：

1. **可见集外驱逐（静态方案的硬边界）**：SWA 下目标距 query 超过 $L(w-1)$ 时计算图里根本没有边（式 (3)），任何打分、任何训练都救不回单层外的信息。sink 缺失是此类的特例但机制不同（§4）：不是检索失败，是归一化压力失去出口后的整行权重重分配，表现为 PPL 爆炸——StreamingLLM 的对照数据（纯窗口超窗即崩 vs 4 sink 稳定到 4M，[A]）是本章最干净的一组因果证据。
2. **打分器漏选（动态方案的上限）**：needle 所在位置的 indexer 分或块分排不进 top-K，效果等同于被驱逐。动态方案的召回率上限 = 打分器的召回率——DSA 用 KL 蒸馏把主模型行为灌进 indexer（式 (14)）、NSA 直接复用 compression 支路的真实 attention 权重（式 (9)），都是在抬高这个上限；MoBA 的零参数均值打分器最便宜，上限也最依赖「needle 所在块的均值方向与 query 对齐」这一假设。
3. **保底结构对照**：NSA 的 compression（全局稀释在场）+ sliding（近处硬保证）；MoBA 的当前块恒选 + SWA hybrid；Gemma 3 每 6 层一次感受野重置；StreamingLLM 的显式 sink 位。每家都给「窗口不够 / 打分失败」留了兜底——评估一个稀疏方案，先看它的兜底覆盖哪类失败。

**证据面**（各方案自报，证据等级随行）：NSA 报告与 full attention 基线持平或更好，含长上下文检索任务（[B]，无官方代码，结论按论文口径引）；DeepSeek-V3.2-Exp 报告与 V3.1-Terminus 各基准持平（[A]）；MoBA 报告与 full 基线可比（[B]）。评测口径警告：单针 needle 任务易饱和（模型学会「找异常 token」即满分），区分不了「可见集够大」与「真会检索」；多针、键值追踪等复合检索的系统性讨论与 RULER 数据见 4.6——稀疏方案宣称的「等效上下文长度」应以该类复合任务为准。

最后登记一条部署侧的风险形状：动态方案的打分器在训练分布内召回率很高，分布外（新文体、新语言、工具产出的格式化文本）可能静默退化——稀疏召回缺陷不像显存溢出那样有声，它表现为答案悄悄变差。上线侧的对应做法是保留 dense 对照探针集、周期性复测召回；这已属 deployment 议题，此处只登记风险形状。

## 总结

本章向 5.2 / 5.5 交付的稀疏方案谱系：

| 方法 | 可见集结构 | 动态 / 粒度 | 每 query key 位（工作点， $t$ = 131072） | cache 影响 | 训练需求 | 锚点（等级） |
| --- | --- | --- | --- | --- | --- | --- |
| SWA | 最近 $w$ | 静态 / token | $w$ = 4096 | 封顶 512 MiB（Mistral-7B） | 从头训 | Mistral-7B [A] |
| SWA+full | 5:1 交错 | 静态 / token | 混合（ $r$ ≈ 18%，式 (5)） | global 层主导 | 从头训 | Gemma 3 [A] |
| sink+SWA | 首 4 + 最近 $w$ | 静态 / token | $w$ + 4 | 封顶 | 免训插入 | StreamingLLM [A] |
| NSA | 压缩+选择+滑窗 | 动态 / 块 | ≈ 9.7K（7.4%） | 不减 | 原生联合训 | arXiv:2502.11089 [B] |
| DSA | indexer top-2048 | 动态 / token | $K$ = 2048（1.6%） | 不减（读取减 64×） | 从 dense 续训 ≈ 0.94T token | DeepSeek-V3.2-Exp [A] |
| MoBA | top-$`k`$ 块（+SWA） | 动态 / 块 | $k \cdot n_b$ | 不减 | 免训插入声明 | MoonshotAI/MoBA [B] |

公式清单：SWA 可见集式 (1)；感受野 $\rho_L(t) = t - L(w-1)$ 式 (2)(3)；SWA cache 封顶式 (4)；混合层 FLOPs 比式 (5)；sink 输出分解式 (6) 与驱逐放大式 (7)；NSA 三分支式 (8)–(10) 与门控式 (11)；DSA indexer 打分式 (12)、top-K 式 (13)、KL 蒸馏式 (14)；MoBA 块路由式 (15)(16)。

## 练习

1.（推导）补全式 (3) 的归纳证明：写明「第 $\ell$ 层读窗一步」如何给出递推式 (2)（提示：FFN / norm / 残差逐位置，依赖只经 attention 扩展）。然后：(a) 代入 Mistral-7B（ $L$ = 32、 $w$ = 4096）算精确感受野，说明论文「约 131K」口径与精确值差在哪；(b) 若窗口逐层交替 $w_1, w_2, w_1, w_2, \ldots$，写出 $R$ 的通式；(c) 证明：在纯 SWA 栈中，若把其中一层换成 global（ $V(i) = \lbrace1..i\rbrace$），则该层之后所有层的 $\rho_\ell(t) = 1$。
2.（推导）sink 账：设一行 logit $z_1, \ldots, z_t$，sink 位置 1 占质量 $\pi$。(a) 由式 (7) 出发，证明驱逐位置 1 后任意两个幸存位置的权重比不变（ $`p'_s / p'_j = p_s / p_j`$），但输出从 $o = \pi v_1 + (1-\pi)\bar v_{\mathrm{rest}}$ 变为 $\bar v_{\mathrm{rest}}$——何时这个变化剧烈？(b) 证明 $\pi \to 1$ 且 $\Vert v_1\Vert \to 0$ 时 $o_t \to 0$，并把它与「softmax 均匀摊开」这一另一默认解比较：为什么均匀解不能近似 head 静默（提示：全体 value 均值的范数下界由什么决定）？
3.（计算）DSA decode 账：61 层 MLA、576 元素 / token / 层、bf16、 $t$ = 131072、 $K$ = 2048。(a) 验证正文的 144 MB / 步 vs 9.2 GB / 步；(b) 若 indexer key 为 $d_I$ 维 FP8（每 token 每层 $d_I$ 字节），全量打分每步需读 $61 \times t \times d_I$ 字节—— $d_I$ = 128 时这笔账是多少，与主路省下的相比如何？由此说明为什么 indexer「 $O(T^2)$ 打分」在 decode 侧仍成立（提示：2.4 §6 的 memory-bound 结论）。
4.（代码）教学实现（非优化）：(a) 给 2.3 §8 的 MHA 加 SWA mask（式 (1)），验证 $T \le w$ 时与 full causal 输出逐点相同、 $T > w$ 时窗口内行为一致；(b) 实现 MoBA 式块路由：均值池化打分、top-$`k`$ 选块、当前块恒选、未来块屏蔽，构造一个 needle 序列（目标块与 query 相关、其余随机），对比 full / SWA / MoBA 在 needle 距离从 $w$ 内移到 $L \cdot w$ 外时的召回行为。

## 参考

- [Jiang et al., 2023](../references.md)：Mistral 7B，arXiv:2310.06825，[A]（key: mistral-7b）——SWA $w$ = 4096、 $L$ = 32、 $H_{kv}$ = 8 锚点
- [Beltagy et al., 2020](../references.md)：Longformer，arXiv:2004.05150，[A]（key: longformer）——静态 pattern 谱系起点
- [Gemma Team, 2025](../references.md)：Gemma 3 Technical Report，arXiv:2503.19786，[A]（key: gemma3）——5:1 local:global、 $w$ = 1024、RoPE base 分工
- [Xiao et al., 2023](../references.md)：Efficient Streaming Language Models with Attention Sinks，arXiv:2309.17453，[A]（key: streamingllm）——sink 现象、4 sink token、4M 流式稳定
- [Yuan et al., 2025](../references.md)：Native Sparse Attention，arXiv:2502.11089，[B]（key: nsa）——三分支公式按论文拆，无官方 repo
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-V3.2（DSA 出处），arXiv:2512.02556，[A]（key: deepseek-v3.2）——indexer 公式、 $K$ = 2048、续训配方、RoPE 布局勘误
- [Lu et al., 2025](../references.md)：MoBA，arXiv:2502.13189，[B]（key: moba）——块路由公式、免训练插入、SWA+MoBA hybrid
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——MLA 576 元素 cache 账与 61 层口径（V3.2 沿用）
