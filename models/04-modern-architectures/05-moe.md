# 4.5 MoE 与 MTP（Mixture-of-Experts and Multi-Token Prediction）

> 本章拆两件围绕「每个 token 该算多少」的设计：MoE 把 FFN 复制成 $N_e$ 份、每 token 只激活 $k$ 份，实现参数与计算的解耦（FLOPs ∝ 激活参数，容量 ∝ 总参数）；MTP 在训练时给主模型挂上预测更远 token 的辅助模块，推理时翻过来当 self-speculative 的草稿。推导主线：Shazeer 2017 noisy top-k 门控 → 负载均衡辅助损失 $\alpha N_e \sum_i f_i P_i$ 的完整推导（协方差分解、对角最小性、梯度动力学）→ Switch 的 top-1 与容量因子 → z-loss → expert choice → DeepSeekMoE 细粒度 + 共享 → DeepSeek-V3 的 sigmoid 路由、aux-loss-free 偏置控制器与组限定分发 → router 病理学 → MTP 目标与接受率公式 | 前置：2.6（FFN 账、等参数规则式 (8)、MoE 预告 §5）、1.2（$6ND$ 口径）、1.3（compute-optimal 分配、全生命周期成本式 (8)、MoE scaling §6）| 锚点：DeepSeek-V3（官方推理代码 inference/model.py 的 Gate / MoE 模块）[A]、Mixtral-8x7B（HF modeling_mixtral 的 MixtralSparseMoeBlock）[A]、Qwen3-30B-A3B [A]；Shazeer et al. [B]、GShard [B]、Switch [A]、ST-MoE [B]、expert choice [B]、DeepSeekMoE [A]、DeepSeek-V3 [A]、Medusa [A]、speculative decoding [B]

## 1. 解耦：容量按总参数付内存，计算按激活付 FLOPs

2.6 §5 留下的接口在这里兑现。MoE 层把一个 FFN 换成 $N_e$ 个并行 FFN（expert）加一台 router：

$$
\mathrm{MoE}(x) = \sum_{i=1}^{N_e} g_i(x)\, E_i(x), \qquad g_i(x) = \begin{cases} \tilde{g}_i(x), & i \in \mathrm{TopK}\big(s(x), k\big) \\ 0, & \text{otherwise} \end{cases} \qquad (1)
$$

**符号**：$x \in \mathbb{R}^{d}$：单位置的 stream 向量；$E_i$：第 $i$ 个 expert，即一份独立参数的 SwiGLU FFN（2.6 式 (6)）；$s(x) \in \mathbb{R}^{N_e}$：router 打分（亲和度，affinity）；$k$：每 token 激活数（top-k）；$\tilde{g}$：选中后归一化的门控权重，各代方案不同（§2–§3 展开）；$g_i$ 至多 $k$ 个非零——稀疏性就写在这行 cases 里。

账（单个 MoE 层，expert 中间维 $d_{\mathrm{ff}}^{(e)}$，SwiGLU 三矩阵口径，对照 2.6 §3.1 的表）：

| 口径 | dense FFN | MoE 层 |
| --- | --- | --- |
| 参数 | $3\, d\, d_{\mathrm{ff}}$ | $N_e \cdot 3\, d\, d_{\mathrm{ff}}^{(e)}$ |
| per-token FLOPs | $6\, d\, d_{\mathrm{ff}}$ | $(k + K_s) \cdot 6\, d\, d_{\mathrm{ff}}^{(e)}$ |

**符号**：$K_s$：常驻激活的 shared expert 数（§3.1；无 shared 的设计 $K_s = 0$）。参数乘 $N_e$、FLOPs 只乘 $k + K_s$——激活比例 $\rho = (k + K_s)/N_e$ 就是稀疏度。1.2 式 (9) 已按此口径记账：$C \approx 6 N_{\mathrm{act}} D$——FLOPs 账里出现的只有激活参数。

三家锚点的稀疏度谱（数值出处见表内证据等级）：

| 模型 | routed expert / 层 | shared | $k$ | 总参 / 激活 | $N_{\mathrm{tot}}/N_{\mathrm{act}}$ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Mixtral-8x7B | 8 | 0 | 2 | 46.7B / 12.9B | ≈ 3.6 | [A]（arXiv:2401.04088） |
| Qwen3-30B-A3B | 128 | 0 | 8 | 30.5B / 3.3B | ≈ 9.2 | [A]（arXiv:2505.09388） |
| DeepSeek-V3 | 256 | 1 | 8 | 671B / 37B | ≈ 18.1 | [A]（config + arXiv:2412.19437） |

稀疏度一路走高：Mixtral 的 8 选 2 还是「半个 dense」，V3 的 257 选 9 把 $\rho$ 压到 3.5%。谱系右端的参照是 Qwen3-Next：512 routed + 1 shared、top-10（[W]，config 与 model card 口径，无 arXiv 论文）——$\rho \approx 2.1\%$。

**Chinchilla-inference 联合优化视角**。1.3 §5 的全生命周期目标 $\mathrm{TC} = 6ND + 2NQ$（1.3 式 (8)）里 $N$ 出现两次：训练一次、推理 $Q$ 次。MoE 把这笔账改写为

$$
\mathrm{TC}(N_{\mathrm{tot}}, N_{\mathrm{act}}, D, Q) = \underbrace{6 N_{\mathrm{act}} D}_{\text{train}} + \underbrace{2 N_{\mathrm{act}} Q}_{\text{inference}}, \qquad L \approx L(N_{\mathrm{tot}}, D) \qquad (2)
$$

**符号**：$Q$：服务期推理 token 总量；右侧第二式是质量侧约束——loss 由总参数与数据决定，1.3 §6 的论证：1.3 式 (2) 的 $A/N^{\alpha}$ 项在 MoE 上由 $N_{\mathrm{tot}} \gg N_{\mathrm{act}}$ 摊薄。1.3 §5 的结论是「$Q$ 大则最优 $N$ 向小偏移」；MoE 把这个结论推到极端——服务成本只认 $N_{\mathrm{act}}$，容量维度 $N_{\mathrm{tot}}$ 不再出现在 FLOPs 目标里，其上限改由内存（bf16 下 2 bytes/参数，1.2 式 (7)）与 §7 的通信账钉死。参数与计算解耦的全部经济含义就是：$N_{\mathrm{act}}$ 按延迟与吞吐的经济学定价，$N_{\mathrm{tot}}$ 按显存与互联定价，两本账分开优化。

实数走一遍 DeepSeek-V3（config [A]：61 层、前 3 层 dense 即 first_k_dense_replace = 3、hidden $d$ = 7168、expert 中间维 2048）：单个 expert 参数 $= 3 \times 7168 \times 2048 \approx 44.0$ M；单 MoE 层参数 $= 257 \times 44.0 \approx 11.3$ B，58 层合计 ≈ 656 B（其余为 dense 层、attention 与 embedding，凑齐 671B）。激活侧：每 token 每层走 9 个 expert ≈ 396 M 参数、per-token MoE FLOPs $= 6 \times 7168 \times 2048 \times 9 \approx 7.9 \times 10^{8}$。对照：同参数的 dense FFN 每 token 要付 $257/9 \approx 28.6$ 倍的 FLOPs。

记忆视角再收一次尾。2.6 §4 把 dense FFN 读成一张 $d_{\mathrm{ff}} \times L$ 条目的键值记忆表；MoE 是这张表的 expert 粒度分片——每层 $N_e$ 张子表、每 token 只查 $k$ 张，router 的 top-k 选择相当于把 2.6 式 (10) 的软稀疏系数 $c_i(x)$ 换成条目集合粒度的硬稀疏。2.6 §5 预告的两个接口在此全部落地：单 expert 尺寸按等参数规则起算（§3.1 的式 (14) 是它在稀疏维度的推广），系数视角则解释了塌缩为何是容量的实质损失——$N_e$ 张子表若只用到几张，总参数摊薄容量的承诺就失效，§4 的病理学与 §2.2 的均衡力学由此获得同一读法。

Mixtral 的官方实现是这节账的最小代码锚点：transformers 的 `modeling_mixtral.py` 中 `MixtralSparseMoeBlock`（[HuggingFace](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)）：router logits 过 softmax、`top_k = 2` 取两家、归一化后按 expert 加权求和——式 (1) 的逐行直译，token-choice 族的最简实例。

## 2. Router 机制谱系：从噪声探索到辅助损失

### 2.1 Shazeer 2017：noisy top-k gating

稀疏门控的原典（[Shazeer et al., 2017](../references.md)，arXiv:1701.06538，[B]，无官方代码，按论文拆公式）在打分里注入逐 expert 的可学噪声：

$$
H(x)_i = (x W_g)_i + \varepsilon \cdot \mathrm{softplus}\Big(\big(x W_{\mathrm{noise}}\big)_i\Big), \qquad \varepsilon \sim \mathcal{N}(0, 1) \qquad (3)
$$

$$
G(x) = \mathrm{softmax}\Big(\mathrm{KeepTopK}\big(H(x), k\big)\Big), \qquad \mathrm{KeepTopK}(v, k)_i = \begin{cases} v_i, & v_i \text{ in top-}k \\ -\infty, & \text{otherwise} \end{cases} \qquad (4)
$$

**符号**：$W_g \in \mathbb{R}^{d \times N_e}$：打分投影；$W_{\mathrm{noise}}$：噪声幅度投影；$\mathrm{softplus}(u) = \log(1 + e^{u})$ 保证噪声幅度非负；$\varepsilon$：逐次前向重采的标准正态噪声。

两个机制要点。其一，KeepTopK 把 top-k 之外压成精确 $-\infty$，softmax 后精确为 0——硬稀疏由此而来。其二，噪声是 router 的探索项：暂居下风的 expert 偶被抽中，从而获得 token 与梯度；噪声幅度逐 expert 可学，等于把「该不该探索」本身参数化。同论文的 importance 损失（按 expert 的重要度变异系数惩罚不均衡）是下一节主角的雏形。

### 2.2 负载均衡辅助损失：$\alpha N_e \sum_i f_i P_i$ 的完整推导

定义（Switch 口径，[Fedus et al., 2021](../references.md)，arXiv:2101.03961，[A]；官方代码在 google-research/t5x）：对一个 batch 的 $T$ 个 token，

$$
f_i = \frac{1}{kT} \sum_{t=1}^{T} \mathbb{1}\big\{i \in \mathrm{TopK}(s(x_t))\big\}, \qquad P_i = \frac{1}{T} \sum_{t=1}^{T} p_i(x_t) \qquad (5)
$$

**符号**：$f_i$：实际路由到 expert $i$ 的 token 分数（token fraction，$k$ 归一化使 $\sum_i f_i = 1$；top-1 时退化为普通份额）；$P_i$：router 对 expert $i$ 的平均概率（softmax 路由下 $\sum_i p_i(x_t) = 1$ 逐 token 成立，故 $\sum_i P_i = 1$）。

辅助损失：

$$
L_{\mathrm{aux}} = \alpha\, N_e \sum_{i=1}^{N_e} f_i\, P_i \qquad (6)
$$

**符号**：$\alpha$：辅助损失系数。推导三条，第三条给出它真正起作用的机制。

**(i) 协方差分解。** 两个归一化条件给出 $\sum_i (f_i - 1/N_e) = \sum_i (P_i - 1/N_e) = 0$。把 $f_i P_i = (f_i - 1/N_e)(P_i - 1/N_e) + (f_i + P_i)/N_e - 1/N_e^{2}$ 对 $i$ 求和，交叉项为零：

$$
\sum_{i} f_i P_i = \sum_{i} \left(f_i - \frac{1}{N_e}\right)\left(P_i - \frac{1}{N_e}\right) + \frac{1}{N_e} \qquad (7)
$$

记第一项为 $N_e\, \widehat{\mathrm{Cov}}(f, P)$（在专家的均匀测度下的经验协方差），代回式 (6)：

$$
L_{\mathrm{aux}} = \alpha\left(1 + N_e^{2}\, \widehat{\mathrm{Cov}}(f, P)\right) \qquad (8)
$$

完美均衡（$f_i = P_i = 1/N_e$）时 $\widehat{\mathrm{Cov}} = 0$、$L_{\mathrm{aux}} = \alpha$。损失实际惩罚的是「实际负载 $f$ 与平均概率 $P$ 的协同偏离」——单看一边平坦不算均衡。

**(ii) 对角最小性（乘积在均匀分布最小的证明）。** 考察路由完全硬化的极限：每 token 的概率质量集中在被选 expert 上，此时 $f = P$，式 (6) 退化为 $\alpha N_e \sum_i f_i^{2}$。由 Cauchy–Schwarz，

$$
\left(\sum_{i=1}^{N_e} f_i^{2}\right)\left(\sum_{i=1}^{N_e} 1^{2}\right) \ge \left(\sum_{i=1}^{N_e} f_i\right)^{2} = 1 \;\Longrightarrow\; \sum_{i} f_i^{2} \ge \frac{1}{N_e} \qquad (9)
$$

等号当且仅当 $f_i$ 全等，即均匀分布是对角族 $f = P$ 上的唯一最小点。对角族不是人为限制：训练推进时 top-k router 的概率趋于尖锐，工作点落在对角附近，式 (9) 的最小性随锐化渐近生效。

**(iii) 梯度动力学。** $f_i$ 是 router logits 的分段常数函数（top-k 指示，梯度几乎处处为 0），$\nabla L_{\mathrm{aux}}$ 只经 $P_i$ 流动。式 (5)(6) 直接求导：

$$
\frac{\partial L_{\mathrm{aux}}}{\partial p_i(x_t)} = \frac{\alpha N_e}{T}\, f_i \qquad (10)
$$

每条 token 上 expert $i$ 的概率被以正比于 $f_i$ 的力度下压；softmax 的逐 token 归一把「下压高负载者」转化为「相对抬升低负载者」。这是均值回复动力学：任何 $f$ 偏离均匀的配置都会产生把它推回去的力，内点不动点只有均匀分布。

**诚实边界**：在整个 $(f, P)$ 乘积单纯形上，双线性形 $\sum_i f_i P_i$ 的无约束最小在反相关顶点（趋于 0），不在均匀点——式 (6) 不是「最小即均匀」的凸景观，可用性来自「$f$ 与 $P$ 由同一 router 生成」的可达集与梯度方向。这个边界有推论：既然起作用的是均值回复方向而非损失值本身，同一个不动点就可以用不进损失函数的控制器实现——§3.2 的 aux-loss-free 路由正是这条路。

**为什么不用熵**：$H(P)$ 只看平均概率分布。极端情形所有 token 都路由到同一 expert 时，若该 expert 对每条 token 的概率只是略高，$P$ 仍可接近平坦、熵接近最大——熵对「塌缩但分数接近」失效，式 (6) 的乘积结构则直接读出 $f$ 与 $P$ 的错位。

### 2.3 Switch：top-1、容量因子与 token 溢出

GShard（[Lepikhin et al., 2020](../references.md)，arXiv:2006.16668，[B]）先把 MoE 搬进 encoder-decoder Transformer：top-2 路由、每 expert 设容量、第二 expert 溢出时随机改派到有余量的 expert。Switch（[Fedus et al., 2021](../references.md)，[A]）把 $k$ 压到 1，三个设计点：

- **top-1 路由**：每 token 只发一个 expert，dispatch 决策、通信与 combine 对照 top-2 各减半；论文的等算力对照显示质量不降（[A]，结论粒度以其实验设置为准）。
- **容量因子（capacity factor）**：每个 expert 每 batch 只处理固定额度

$$
C_e = \left\lceil \mathrm{CF} \cdot \frac{T_{\mathrm{batch}}}{N_e} \right\rceil \qquad (11)
$$

**符号**：$\mathrm{CF} \ge 1$：容量因子（论文对照 1.0 / 1.25 / 1.5 / 2.0，常用工作点 1.25）；$T_{\mathrm{batch}}$：一个 batch 的 token 总数。容量是内存对齐的需要：expert 的输入 batch 必须定长才能向量化，$\mathrm{CF}$ 是「为负载波动预留多少缓冲」的工程旋钮。
- **token 溢出机制**：expert $i$ 满员后，轮到它的 token 被丢弃——该层 FFN 输出取零，token 仅靠 residual stream 穿层，等效于跳过该层。质量影响随 CF 的扫描呈两段：溢出率压在小概率区（CF = 1.25）时影响可忽略；CF 压向 1.0、或 router 不均衡抬高溢出率时，质量显著退化。丢弃不可微，被丢 token 的 router 梯度断供——所以 Switch 把式 (6) 的 top-1 特例与容量机制绑定使用：aux loss 负责把溢出率压下去，容量机制负责兜底。

### 2.4 z-loss：只压数值、不管均衡

ST-MoE（[Zoph et al., 2022](../references.md)，arXiv:2202.08906，[B]）给 router logits 加幅度惩罚：

$$
L_z = \frac{1}{T} \sum_{t=1}^{T} \left( \log \sum_{j=1}^{N_e} e^{\ell_j(x_t)} \right)^{2} \qquad (12)
$$

**符号**：$\ell_j(x_t)$：token $t$ 的 router logits。$\log \sum_j e^{\ell_j}$ 是 $\max_j \ell_j$ 的光滑上界，平方惩罚把 logits 整体压向小幅度：exp 不溢出、softmax 远离饱和区。动机是数值稳定（router 计算常用低精度）与训练发散抑制（ST-MoE 的稳定性消融，[B]）。它对所有 logits 一视同仁地下压，不含任何负载信号——与式 (6) 正交，两损失在该配方中叠加使用。

### 2.5 expert choice：把选择方向反过来

token-choice 路由的结构性困境：每 token 选 $k$ 个 expert，均衡只能靠辅助压力逼近。expert choice（[Zhou et al., 2022](../references.md)，arXiv:2202.09368，[B]）反转选择方向——每个 expert 选固定数目的 token：

$$
S = X W_g \in \mathbb{R}^{T \times N_e}, \qquad p_{t, e} = \frac{e^{S_{t,e}}}{\sum_{t'=1}^{T} e^{S_{t',e}}} \qquad (13)
$$

**符号**：$X \in \mathbb{R}^{T \times d}$：一个 batch（或序列）的 token 表示；softmax 沿 token 轴归一（逐 expert 一列）。expert $e$ 取 $p_{\cdot, e}$ 的 top-$C$ 个 token，$C = kT/N_e$。**天然均衡**：每个 expert 恰好处理 $C$ 个 token，负载均衡由构造保证、辅助损失整项删除。代价转移到 token 侧：一个 token 可被 0 到 $N_e$ 个 expert 选中，per-token 计算量不再恒定。

**自回归不适配**的机制原因：expert 的 top-$C$ 竞争横跨全序列，token $t$ 是否被 expert $e$ 选中依赖其他（包括未来）token 的分数。训练时整序列可见，没有问题；decode 时未来 token 尚不存在，要么路由决策随序列增长而翻悔（破坏 prefix 一致性、KV cache 失效），要么把竞争窗口限制在历史内（论文给 AR 场景的补救，牺牲均衡的精确性）。expert choice 的主场因此在 encoder 与非自回归场景；decoder-only LLM 的主流仍是 token-choice。

## 3. DeepSeek 线：细粒度、共享与无辅助损失路由

### 3.1 DeepSeekMoE：$m$ 倍细粒度 + 常驻共享的参数分配

DeepSeekMoE（[Dai et al., 2024](../references.md)，arXiv:2401.06066，[A]）拧两个旋钮。其一，fine-grained 切分：expert 数乘 $m$、中间维除 $m$、top-k 乘 $m$：

$$
N_e \to m N_e, \qquad d_{\mathrm{ff}}^{(e)} \to \frac{d_{\mathrm{ff}}^{(e)}}{m}, \qquad k \to m\, k \qquad (14)
$$

两笔账同时不变：参数 $m N_e \times 3 d\, d_{\mathrm{ff}}^{(e)}/m = N_e \times 3 d\, d_{\mathrm{ff}}^{(e)}$；per-token FLOPs $m k \times 6 d\, d_{\mathrm{ff}}^{(e)}/m = k \times 6 d\, d_{\mathrm{ff}}^{(e)}$。变的是组合自由度：

$$
\binom{m N_e}{m k} \gg \binom{N_e}{k} \qquad (15)
$$

1.3 §6 已登记组合数论证，此处补全参数不变性一侧：式 (14) 是等参数等算力的纯结构操作（与 2.6 式 (8) 同一思想），收益只能归因于更细的知识单元组合。其二，shared expert：划出 $K_s$ 个 expert 常驻激活、不经路由，捕捉跨 token 的公共知识，让 routed expert 专注差异化内容。效果锚点：DeepSeekMoE 16B（2.8B 激活）以约 40% 计算量达到 LLaMA2-7B 的下游性能（[A]，论文摘要口径）。

### 3.2 DeepSeek-V3：sigmoid 亲和度、aux-loss-free 偏置、组限定分发

V3 路由全套公式（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]；config [A]）。亲和度用 sigmoid 而非 softmax：

$$
s_{i,t} = \sigma\big(x_t^{\top} e_i\big) \qquad (16)
$$

**符号**：$e_i \in \mathbb{R}^{d}$：router 的第 $i$ 个 expert 嵌入。softmax 把全部 expert 放进逐 token 的归一化竞争；sigmoid 让各 expert 分数相互独立，归一化推迟到 top-k 之后、只在选中集合内进行（V2 起的路由函数选择，[DeepSeek-AI, 2024](../references.md)，arXiv:2405.04434，[A]）。

选择用加偏置的分数、权重用原始分数：

$$
i \in \mathrm{TopK}\big(\{ s_{j,t} + b_j \}_{j=1}^{256},\; k = 8\big), \qquad g_{i,t} = \frac{s_{i,t}}{\sum_{j \in \mathrm{TopK}} s_{j,t}} \cdot r \qquad (17)
$$

**符号**：$r$：routed_scaling_factor = 2.5（config 字段）；$b_i$：持久偏置——**只进选择、不进权重**，$g$ 由无偏置的 $s$ 算出。

**aux-loss-free 偏置更新**（替代式 (6) 的控制器）。每个训练步结束，按本步观测负载更新：

$$
b_i \leftarrow b_i + \Delta \cdot (\mathrm{target} - \mathrm{observed})_i \qquad (18)
$$

**符号**：$\mathrm{observed}_i$：本步 expert $i$ 的负载份额（式 (5) 的 $f_i$ 同族量）；$\mathrm{target}$：均匀目标；$\Delta$：更新步长。V3 的实现取符号步——过载减 $\gamma$、欠载加 $\gamma$，$\gamma = 10^{-3}$ 为训练大部分时间的工作点（[A]，论文 §2.1.2）。走读三条性质。

**(i) 收敛行为**。式 (18) 是负载误差上的离散积分器：$b_i$ 累积历史误差，稳态要求 $\mathrm{observed} = \mathrm{target}$，否则 $b_i$ 继续单向漂移——任何不动点必为均衡负载。符号步版本的稳态不是点而是幅度 $\sim \gamma$ 的极限环：偏置在均衡值附近振荡，负载偏差被 $\gamma$ 对应的分数位移界定。$\gamma$ 越小稳态波纹越小、响应越慢，$10^{-3}$ 是响应速度与波纹的折中点。

**(ii) 不扭曲路由分布**。三个结构事实。(a) 偏置只出现在 TopK 的掩码位置：式 (17) 的权重由无偏置分数计算，被选中的 expert 拿多少权重不受 $b$ 影响；(b) $b$ 是按规则更新的状态量、不在计算图里、无梯度路径——LM 损失对 router 参数（式 (16) 的 $e_i$）的梯度与无控制器时逐位相同；(c) 对照式 (6)：aux loss 的梯度直接进 logits（式 (10)），等于在 LM 目标上恒定叠加一股均值回复力，router 学到的分布被这股力持续塑形。aux-loss-free 把「均衡」从优化目标降级为控制回路，目标函数里只剩 LM 一项。诚实注记：权重经 top-k 集合的归一化分母仍间接受选择集影响——偏置改变「选谁」，不改变「选中者的相对权重」。

**(iii) 与 aux loss 的不动点等价**。§2.2 (iii) 的均值回复方向与式 (18) 的控制方向一致：同一均衡点，前者用梯度实现、后者用前馈偏置实现。

**seq-mask balancing（序列级兜底）**。控制器保证的是 batch 级均衡，单条序列内仍可能极端集中。V3 补一个序列级辅助损失——式 (6) 的 $f_i$、$P_i$ 改在单序列内统计（sigmoid 路由下 $P_i$ 按归一化的 sigmoid 分数计），系数取极小（论文口径：extremely small），只兜底不主导。

**node-limited routing（组限定分发）**。256 个 expert 分 8 组，先按组内亲和度之和选 top-4 组、再在选中组内取 top-8（V2 的 group-limited 路由沿用，[A]；config 字段 n_group = 8、topk_group = 4）。每 token 的分发目标节点数 $\le 4$——这条规则的收益不在模型侧，在 §7 的通信账。

### 3.3 官方代码走读：DeepSeek-V3 inference/model.py 的 Gate

对照官方推理代码（[deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)，`Gate` / `MoE` 类，[A]）。教学缩略版（教学用，非优化实现；省略组掩码与分布式分发）：

```python
class Gate(nn.Module):                    # 式 (16)(17)(18) 的对应物
    def __init__(self, d, n_experts=256, topk=8, route_scale=2.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, d))      # e_i 的堆叠
        self.bias = nn.Parameter(torch.zeros(n_experts),
                                 requires_grad=False)              # b_i：控制回路状态
        self.topk, self.route_scale = topk, route_scale

    def forward(self, x):                   # x: [T, d]
        scores = torch.sigmoid(x @ self.weight.T)        # s_{i,t}，式 (16)
        choice = scores + self.bias                      # 偏置只进选择
        idx = choice.topk(self.topk, dim=-1).indices     # TopK(s + b, 8)
        w = scores.gather(-1, idx)                       # 权重取无偏置分数
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.route_scale   # 式 (17)
        return idx, w
```

逐行对账：`sigmoid` ↔ 式 (16)；`choice` 与 `scores` 分离 ↔ 「偏置只进选择不进权重」（性质 ii-a）；`gather` 回原始分数 ↔ 式 (17) 的分子；`requires_grad=False` ↔ 控制器不走梯度（性质 ii-b）。与官方实现的差异：官方版多出 n_group / topk_group 的组掩码（组限定分发的代码落点）与推理路径的 expert 分发；另一处工程痕迹是 bias 字段挂在 hidden 维 7168 的条件分支上——该字段为 V3 口径硬编码。偏置的实际更新发生在训练循环（按式 (18) 统计全步负载后改写），推理代码里它只是常数。

## 4. Router 病理学：三种失效与诊断指标

MoE 特有的失效全部围绕 router 的反馈回路（素材分散于 Switch [A]、ST-MoE [B]、DeepSeekMoE [A] 的诊断讨论，各按其口径）：

1. **塌缩（collapse）**：正反馈动力学——偶然领先的 expert 收到更多 token、独占梯度、分数更高、再收更多 token。终态是少数 expert 承载绝大多数 token，MoE 退化为小 dense。§2.1 的噪声探索与 §2.2 的均值回复力存在的原因就是这个回路。诊断：$\max_i f_i$ 漂移、负载分布的变异系数上升、router 输出分布的熵单调下降。
2. **死 expert（dead experts）**：塌缩的另一极——长期零 token 的 expert 收不到梯度、停在随机初始化，此后更难被选中，形成死锁。2.6 §2.1 的死区在层粒度重演。诊断：$\min_i f_i \approx 0$ 的 expert 占比、各 expert 参数更新范数的分层。
3. **振荡（oscillation）**：均衡回路的增益过高——$\alpha$ 过大（式 (6)）或 $\Delta$ 过大（式 (18)）——负载在两组配置间来回跳，训练 loss 出现同频毛刺。诊断：$f_i$ 跨步时间序列的负自相关、aux loss 或偏置更新量的振荡幅度。

诊断口径注记：上述指标全部逐层统计——塌缩与死 expert 常只在个别层发生，整网平均会洗平信号。GShard 的随机改派（§2.3）亦可读作一条抗塌缩机制：溢出 token 被随机送往有余量的 expert，给低负载 expert 续上梯度流，代价是路由决策不再纯由亲和度决定。

设计读法：$\alpha$ 与 $\Delta$ 是同一旋钮的两种实现——均衡回路的增益。增益过低塌缩、过高振荡，工作点在两者之间；这与 §3.2 (i) 的 $\gamma$ 折中是同一条控制论常识在两个配方里的现身。

## 5. MoE × scaling laws：细粒度缩放的曲线证据

1.3 §6 已建立框架：$C \approx 6 N_{\mathrm{act}} D$ 下，FLOPs 角色由 $N_{\mathrm{act}}$ 继承、容量项由 $N_{\mathrm{tot}}$ 摊薄。本章补两条实证。其一，DeepSeekMoE 的 IsoFLOPs 对照（[A]）：同算力下 MoE 的 loss 曲线整体低于 dense，细粒度切分（式 (14)）再压一截——组合自由度（式 (15)）兑现为 loss 差；该论文的粒度消融进一步显示：固定激活参数、只增大切分倍数 $m$，loss 单调改善，直到路由与通信开销反超。其二，工作点终点：DeepSeek-V3 以 671B 总参 / 37B 激活、14.8T token 训练，pre-training 用 2.664M H800 GPU-hours（全训练 2.788M，口径见 1.2 §9；[A]）。按 $C = 6 N_{\mathrm{act}} D \approx 3.3 \times 10^{24}$ FLOPs 的预算，dense 等价物只养得起 37B 量级的容量；MoE 把容量推到 671B，边际成本不进 FLOPs 账而进内存与 §7 的通信账。细粒度仍在推进：Qwen3-Next 的 512+1、top-10（[W]）把 $\rho$ 压到 2.1%；router 成本与 expert 通信何时反超容量收益是进行时问题，本章不下结论。

## 6. MTP：训练辅助目标，推理自投机

### 6.1 模块结构与目标

DeepSeek-V3 的多 token 预测（multi-token prediction，MTP）模块（[A]，论文 §2.2）：主模型之外挂 $D$ 个顺序模块（V3 取 $D = 1$，预测「下一个的下一个」token）。第 $k$ 个模块：

$$
h_i^{\prime k} = M_k \left[ \mathrm{RMSNorm}\big(h_i^{k-1}\big);\ \mathrm{RMSNorm}\big(\mathrm{Emb}(t_{i+k})\big) \right] \qquad (19)
$$

$$
h^{k} = \mathrm{TRM}_k\big(h^{\prime k}\big), \qquad P_{i+k+1}^{k} = \mathrm{OutHead}\big(h_i^{k}\big) \qquad (20)
$$

**符号**：$h_i^{k-1}$：上一层（$k = 1$ 时是主模型 backbone）位置 $i$ 的表示；$\mathrm{Emb}$ 与 $\mathrm{OutHead}$：与主模型共享的 embedding 与输出头；$[;\;]$：特征轴拼接；$M_k \in \mathbb{R}^{d \times 2d}$：拼接投影；$\mathrm{TRM}_k$：一个完整 transformer block；$P^{k}$：模块 $k$ 对位置 $i+k+1$ 的预测分布。链式结构的关键：模块 $k$ 吃模块 $k-1$ 的表示与真实 token $t_{i+k}$ 的嵌入——预测条件始终是真实前缀，因果链完整。

shape 走读（$D = 1$，位置 $i$）：

```
h_i [d]（backbone 输出）  Emb(t_{i+1}) [d]（真实下一 token）
 └─ RMSNorm ──┐            ┌── RMSNorm ─┘
              ▼            ▼
            拼接 [2d] → M₁ → [d] → TRM₁ → h_i¹ [d] → OutHead → P_{i+2}¹ [V]
```

目标函数：

$$
L_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_{\mathrm{MTP}}^{k}, \qquad L_{\mathrm{MTP}}^{k} = \frac{1}{T} \sum_{i} \mathrm{CE}\big(P_{i+k+1}^{k},\ t_{i+k+1}\big) \qquad (21)
$$

**符号**：$\lambda$：MTP 权重；CE 口径同 1.1 式 (4)。梯度路径值得写明：$L_{\mathrm{MTP}}^{1}$ 经式 (19) 回传到 backbone 表示 $h_i$，主模型参数同时被下一 token 目标与次一 token 目标塑形；$\mathrm{Emb}$ 与 $\mathrm{OutHead}$ 共享意味着 MTP 误差也直接训练词表两侧。推理时这些模块可整体丢弃、恢复标准自回归模型——辅助目标是纯训练期装置，§6.2 再把它翻过来用。训练收益按论文口径是监督信号加密：每个位置除下一 token 外还对更远 horizon 提供监督，表示被迫编码更长程的可预测结构，数据效率由此提高。

### 6.2 推理：self-speculative 与接受率

推理时 MTP 模块翻过来当草稿模型（self-speculative）：模块 1 以 backbone 表示为条件先猜 $t+2$，主模型一次前向并行验证两个位置。接受规则与接受率公式出自 speculative decoding（[Leviathan et al., 2023](../references.md)，arXiv:2211.17192，[B]）：草稿分布 $q$、目标分布 $p$，采样 $x \sim q$ 后以概率 $\min(1, p(x)/q(x))$ 接受；拒绝则从归一化的 $(p - q)_{+}$ 重采。该规则保证输出分布精确等于 $p$——无损加速（练习 3 证）。

整体接受率：

$$
\alpha = \sum_{x} q(x)\, \min\left(1, \frac{p(x)}{q(x)}\right) = \sum_{x} \min\big(p(x), q(x)\big) = 1 - D_{\mathrm{TV}}(p, q) \qquad (22)
$$

**符号**：$D_{\mathrm{TV}}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$：全变差距离（第二步：$\min(p,q) = (p + q - |p - q|)/2$，求和即得）。草稿越贴近目标，$\alpha$ 越高——MTP 模块与主模型共享 embedding、输出头且条件于真实前缀，正是把 $D_{\mathrm{TV}}$ 压小的结构手段。

每步草稿 $\gamma$ 个 token 时，单次主模型前向的期望产出：

$$
\mathbb{E}[\text{tokens}] = \sum_{j=0}^{\gamma} \Pr\{j \text{ accepted}\} = \sum_{j=0}^{\gamma} \alpha^{j} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} \qquad (23)
$$

（几何级数；逐位置接受独立同分布是近似，实际相邻位置相关。）$D = 1$ 即 $\gamma = 1$：$\mathbb{E} = 1 + \alpha$。V3 报告第二 token 接受率 85%–90%（跨生成主题，[A]），期望产出 1.85–1.9 token/前向，论文实测端到端约 1.8× TPS。系统侧问题——验证步与 batch 的交互、拒绝时 KV cache 的处置、草稿与主模型的算子融合——归 inference 维度。

### 6.3 Medusa：并行头的对照

Medusa（[Cai et al., 2024](../references.md)，arXiv:2401.10774，[A]；FasterDecoding/Medusa）把链式模块换成并行头：backbone 最后隐状态 $h_t$ 上并排 $K$ 个小 MLP，第 $k$ 头独立预测 $t + k + 1$：

$$
P^{(k)}(\cdot \mid t) = \mathrm{softmax}\Big( W^{U} \big( h_t + W_2^{(k)}\, \mathrm{SiLU}(W_1^{(k)} h_t) \big) \Big) \qquad (24)
$$

**符号**：$W_1^{(k)}, W_2^{(k)}$：第 $k$ 头的残差式小 MLP；$W^{U}$：共享输出头。与 MTP 三点对照。**结构**：Medusa 头之间无依赖、无 embedding 链——头 $k$ 拿不到 $t+k$ 的真实嵌入，条件信息比式 (19) 少一环；MTP 的链式输入正是补上这一环。**参数**：Medusa 只加 $K$ 个小头（Medusa-1 冻结 backbone）；MTP 加 $D$ 个完整 block 但共享 embedding 与输出头。**验证**：并行头的联合预测靠树状 attention 在候选组合的笛卡尔积上批量验证；链式 MTP 天然产出单条草稿链。两条路线的读法相同——把训练时多学的未来预测器，在推理时折现成每前向多产出的 token。

## 7. 系统接口：expert 并行与 all-to-all 预告

MoE 的稀疏激活把 dense 层换成跨设备分发问题：expert 分片到不同卡（expert parallelism，EP——$N_{\mathrm{tot}}$ 的内存账使分片成为容量前提而非优化选项），每 token 的隐状态发往其 top-$k$ expert 所在设备、算完收回——dispatch 与 combine 两次 all-to-all。通信账（每 token 每 MoE 层，bf16，按逐 expert 发送的上界口径）：$2 \cdot k \cdot d \cdot b = 2 \times 8 \times 7168 \times 2 \approx 229$ KB（V3 口径），乘 58 个 MoE 层。node-limited routing（§3.2，每 token $\le 4$ 个目标节点）把 all-to-all 的扇出从「全设备」压成有界子集——路由规则直接为通信服务。负载不均的系统化身是热点卡（straggler）：$f_i$ 不均不只是统计问题，还是尾延迟问题，这是均衡回路的第二层动机。进一步主题——all-to-all 的带宽与拓扑、计算-通信重叠（DualPipe）、expert 并行的负载再均衡（EPLB）、EP 与张量并行的切分权衡——全部归 cluster 维度；本章交付的接口量是路由数学（式 (1)–(18)）加两个数：每 token 每 MoE 层 $2kd$ 字节的通信上界、$\le M$ 节点的扇出上界。

## 总结

| 概念 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| MoE 层 | 式 (1)；参数 $N_e \cdot 3d\,d_{\mathrm{ff}}^{(e)}$，FLOPs $(k+K_s)\cdot 6d\,d_{\mathrm{ff}}^{(e)}$ | 解耦：FLOPs ∝ $N_{\mathrm{act}}$，容量 ∝ $N_{\mathrm{tot}}$；联合视角式 (2) |
| 锚点稀疏度 | Mixtral 8 选 2（46.7B/12.9B）[A]；Qwen3-30B-A3B 128 选 8（30.5B/3.3B）[A]；DeepSeek-V3 256+1 选 8（671B/37B）[A] | $\rho$：25% → 6.25% → 3.5% |
| noisy top-k | 式 (3)(4) [B] | 可学噪声 = 参数化探索；KeepTopK 给硬稀疏 |
| aux loss | $\alpha N_e \sum f_i P_i$，式 (6)–(10) [A] | 协方差分解式 (8)；对角最小性式 (9)（CS）；梯度只走 $P$、均值回复式 (10)；全局最小不在均匀（诚实边界） |
| Switch | top-1；容量式 (11)；溢出 = 跳层 [A] | CF 常用 1.25；溢出率小则质量影响可忽略 |
| z-loss | 式 (12) [B] | 压 logits 幅度，数值稳定；与 aux 正交叠加 |
| expert choice | 式 (13) [B] | 均衡由构造保证；token 侧计算量可变；AR 不适配（未来依赖） |
| DeepSeekMoE | 式 (14)(15) [A] | $m$ 倍切分等参等算力；组合数 $\binom{mN_e}{mk}$；16B/2.8B ≈ 40% 算力达 LLaMA2-7B |
| DeepSeek-V3 路由 | 式 (16)(17)(18) [A] | sigmoid 亲和；偏置只进选择不进权重；$\gamma = 10^{-3}$ 符号步 = 积分控制器；seq 级兜底 + 组限定（8 组选 4） |
| router 病理 | 塌缩 / 死 expert / 振荡 | 诊断：$\max f_i$、熵、死 expert 占比、$f_i$ 负自相关 |
| MTP | 式 (19)–(21) [A] | 链式模块、共享 Emb/OutHead；训练加密监督；推理 self-speculative：$\alpha = 1 - D_{\mathrm{TV}}$ 式 (22)、$\mathbb{E} = (1-\alpha^{\gamma+1})/(1-\alpha)$ 式 (23)；V3 接受率 85–90%、约 1.8× TPS |
| Medusa 对照 | 式 (24) [A] | 并行头、无链；树状验证 |
| EP 接口 | 每 token 每层 $2kd\,b$ bytes；扇出 $\le M$ | 通信/重叠/EPLB 归 cluster 维度 |

## 练习

1.（推导）(a) 由 $\sum_i f_i = \sum_i P_i = 1$ 推出式 (7)，写明交叉项为何为零；(b) 完成式 (9) 的 Cauchy–Schwarz 步骤并给出等号条件；(c) 构造一对单纯形向量 $(f, P)$ 使 $\sum_i f_i P_i < 1/N_e$，并结合「$f$、$P$ 由同一 router 生成」说明该点为何不构成对式 (6) 的反驳；(d) 由式 (10) 出发，写出 $L_{\mathrm{aux}}$ 对 logit $\ell_j(x_t)$ 的完整梯度（经 softmax Jacobian），说明高 $f$ expert 的概率质量被推向何处。
2.（计算）DeepSeek-V3 单 MoE 层：$d$ = 7168、$d_{\mathrm{ff}}^{(e)}$ = 2048、256 routed + 1 shared、top-8。(a) 层参数与每 token 激活参数；(b) per-token FLOPs，以及等激活 dense FFN 的中间维应取多少；(c) 若改为 512 个中间维 1024 的 expert、top-16 + 1 个同样变细的 shared，用式 (14) 检验参数与 FLOPs 是否不变，组合数变几倍（$\binom{512}{16}$ 对 $\binom{256}{8}$，用对数估）。
3.（推导）(a) 由接受-重采规则证明 speculative decoding 的输出分布精确等于 $p$（提示：对「接受 $x \sim q$」与「拒绝后重采」两路求边际）；(b) 由该规则推式 (22) 的 $\alpha = 1 - D_{\mathrm{TV}}$；(c) 在逐位置 iid 接受假设下推式 (23)；(d) 取 $\alpha = 0.87$、$\gamma = 1$ 算期望产出，并说明 V3 实测约 1.8× 与理论值之间的差来自哪里（草稿模块自身前向、验证开销）。
4.（计算 + 推导）Switch 容量账：batch 65536 token、$N_e$ = 128、CF = 1.25。(a) 每 expert 容量 $C_e$；(b) 写出溢出 token 在该层的前向公式（提示：输出 = residual 直传）；(c) 若负载呈 Zipf 形 $f_i \propto i^{-1.2}$，数值估计溢出 token 占比（允许用代码模拟）。

## 参考

- [Shazeer et al., 2017](../references.md)：Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer，arXiv:1701.06538，[B]（key: shazeer-moe）——式 (3)(4) 与 importance 损失雏形
- [Lepikhin et al., 2020](../references.md)：GShard，arXiv:2006.16668，[B]（key: gshard）——top-2、容量与随机改派
- [Fedus et al., 2021](../references.md)：Switch Transformers，arXiv:2101.03961，[A]（key: switch；官方代码 google-research/t5x）——式 (5)(6)(11)、top-1 与溢出机制
- [Zoph et al., 2022](../references.md)：ST-MoE，arXiv:2202.08906，[B]（key: stmoe）——式 (12) z-loss 与稳定性消融
- [Zhou et al., 2022](../references.md)：Mixture-of-Experts with Expert Choice Routing，arXiv:2202.09368，[B]（key: expert-choice）——式 (13) 与 AR 不适配分析
- [Dai et al., 2024](../references.md)：DeepSeekMoE，arXiv:2401.06066，[A]（key: deepseekmoe）——式 (14)(15) 与 16B/2.8B 锚点
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V2，arXiv:2405.04434，[A]（key: deepseekv2）——sigmoid 亲和与 group-limited 路由
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——式 (16)–(21)、$\gamma$ = 0.001、seq 级兜底、MTP 接受率 85–90% 与 1.8× TPS（§2.1.2/2.2）
- [DeepSeek-AI, DeepSeek-V3 config.json](../references.md)：[A]（key: deepseek-v3-config）——n_routed_experts = 256、num_experts_per_tok = 8、first_k_dense_replace = 3、routed_scaling_factor = 2.5、61 层
- [deepseek-ai/DeepSeek-V3 官方推理代码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)：[A]——`Gate` / `MoE` 类，§3.3 走读对象
- [Jiang et al., 2024](../references.md)：Mixtral of Experts，arXiv:2401.04088，[A]（key: mixtral）——8 选 2、46.7B/12.9B；代码锚点 `modeling_mixtral.py` 的 `MixtralSparseMoeBlock`
- [Yang et al., 2025](../references.md)：Qwen3 Technical Report，arXiv:2505.09388，[A]（key: qwen3）——Qwen3-30B-A3B：128 experts、top-8、30.5B/3.3B
- [Qwen, Qwen3-Next-80B-A3B-Instruct model card 与 config](../references.md)：[W]（key: qwen3-next）——512 experts、top-10、1 shared 的高稀疏度对照
- [Cai et al., 2024](../references.md)：Medusa，arXiv:2401.10774，[A]（key: medusa；FasterDecoding/Medusa）——式 (24) 并行头与树状验证
- [Leviathan et al., 2023](../references.md)：Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192，[B]（key: speculative-decoding）——接受规则与式 (22)(23)
- [Hoffmann et al., 2022](../references.md)：Training Compute-Optimal Large Language Models，arXiv:2203.15556，[B]（key: chinchilla）——§1 联合视角的源头（1.3 式 (8)）
