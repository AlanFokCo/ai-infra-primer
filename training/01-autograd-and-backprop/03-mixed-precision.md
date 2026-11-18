# 1.3 混合精度与 FP8（Mixed Precision and FP8）

> 本章回答「训练中每个张量该用多少位」。先给五种浮点格式（fp32 / fp16 / bf16 / E4M3 / E5M2）的位布局与范围对照；再推 fp16 混合精度三件套与 loss scaling（下溢界 $2^{-24}$ 、scale 恒等式、dynamic scaling 的进退规则）；然后论证 bf16 为何免 scaling；最后逐条拆 DeepSeek-V3 的全 E4M3 FP8 方案（ $128 \times 128$ 权重块、 $1 \times 128$ 激活 tile、CUDA Core 高精度累加、E5M6 例外），全部对回论文原文 | 前置：1.1（反向 = 逐节点 vjp、save 的内存语义）、1.2（matmul 反向公式）、models 1.2（ $2N$ / $6ND$ 与显存四账） | 锚点：DeepSeek-V3 技术报告 §3.3 [A]、PyTorch AMP（`torch.amp`，教学参照 + 官方实现）；[Micikevicius et al., 2017] [B]、[Micikevicius et al., 2022] [B]

## 1. 浮点格式账：指数位买范围，尾数位买精度

一个 fp16 数为什么最大只能表到 65504？答案写在位布局里。本章全部议题都落在同一行公式上：IEEE 754 风格的二进制浮点数分三段——符号位、指数字段、尾数字段，正规数（normal）的值为

$$
x = (-1)^{s} \cdot 2^{e - b} \cdot (1 + m), \qquad m = \sum_{i=1}^{M} m_i \cdot 2^{-i} \in [0, 1) \qquad (1)
$$

**符号**： $s \in \lbrace 0, 1\rbrace$：符号位； $E$：指数字段位宽； $e$：指数字段的整数值（正规数取 $1 \le e \le 2^{E} - 2$ ）； $b$：指数偏置（bias）； $M$：尾数位宽； $m_i \in \lbrace 0, 1\rbrace$：第 $i$ 个尾数位；隐含前导 1 不占位宽。指数字段全零时是次正规数（subnormal）：值 $(-1)^{s} \cdot 2^{1 - b} \cdot m$ ，没有隐含前导 1，最小可表正值 $2^{1 - b - M}$ 。

这一行公式给出两个独立的读数维度。指数位买**动态范围**：正规数覆盖 $2^{1-b}$ 到约 $2^{2^{E} - 1 - b}$ ，跨度以二倍程（binade，相邻 2 的幂之间的区间）计。尾数位买**相对精度**：1 附近相邻可表数的间隔（machine epsilon）为 $2^{-M}$ ，舍入到最近时的最大相对误差是它的一半 $2^{-M-1}$ 。训练涉及的五种格式按此口径对照（E4M3 / E5M2 各行数值出自 [Micikevicius et al., 2022](../references.md)，arXiv:2209.05433，[B]，Table 1；其余为 IEEE 754 标准值）：

| 格式 | 位布局（符号/指数/尾数） | bias | 最大正规数 | 最小正规数 | 最小次正规数 | 1 附近间隔 | 特殊值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 1 / 8 / 23 | 127 | $3.40 \times 10^{38}$ | $2^{-126}$ | $2^{-149}$ | $2^{-23}$ | inf、NaN 齐全 |
| fp16 | 1 / 5 / 10 | 15 | 65504 | $2^{-14}$ | $2^{-24}$ | $2^{-10}$ | inf、NaN 齐全 |
| bf16 | 1 / 8 / 7 | 127 | $3.39 \times 10^{38}$ | $2^{-126}$ | $2^{-133}$ | $2^{-7}$ | inf、NaN 齐全 |
| E4M3 | 1 / 4 / 3 | 7 | 448 | $2^{-6}$ | $2^{-9}$ | $2^{-3}$ | 无 inf，单一 NaN |
| E5M2 | 1 / 5 / 2 | 15 | 57344 | $2^{-14}$ | $2^{-16}$ | $2^{-2}$ | inf、NaN 齐全 |

表里有两个 fp16 边界值值得记住：最大值 65504 等于 $(2 - 2^{-10}) \cdot 2^{15}$ ；最小正规数 $2^{-14} \approx 6.1 \times 10^{-5}$ ，最小次正规数 $2^{-24} \approx 6.0 \times 10^{-8}$ 。第 3 节 loss scaling 的全部推导都建立在这两个下界上。E4M3 与 E5M2 的分歧在设计原则。E5M2 守 IEEE 惯例（保留 inf 与多个 NaN 模式），相当于「少尾数位版的 fp16」（论文原话），可与 fp16 直接互转。E4M3 相反：不表 inf、只留一个 NaN 码位，把腾出的特殊值码位换成正规数——最大值从 240 扩到 448，动态范围从 17 个 binade 扩到 18 个（[Micikevicius et al., 2022](../references.md)，[B]，§3.1）。

位宽减半买两样东西：算力与带宽。[Micikevicius et al., 2017](../references.md)（arXiv:1710.03740，[B]，§1）记录当时 GPU 的半精度吞吐为单精度的 2 至 8 倍；存储与访存随位宽线性减半。第 2 节起的全部技术，都是为这两样收益付出的代价。

## 2. 混合精度三件套

直接用 fp16 存全部状态并训练，会掉精度（机制在第 3 节展开）。Micikevicius et al. 的对策是三件套（[Micikevicius et al., 2017](../references.md)，arXiv:1710.03740，[B]，§3）：

- 第一，**fp32 master weights**（主权重）：optimizer 持有一份 fp32 副本；
- 第二，**loss scaling**：放大 loss 让梯度离开 fp16 的下溢区；
- 第三，**半精度前向反向 + fp32 累加**：算乘用半精度，累加用 fp32。

更新规则写成

$$
\theta_{t+1} = \theta_{t} - \eta \cdot u(g_t), \qquad w_t = \mathrm{round}_{16}(\theta_t) \qquad (2)
$$

**符号**： $\theta$：fp32 主权重，由 optimizer 持有并更新； $w$：前向反向实际使用的半精度副本，每步从 $\theta$ 舍入得到； $g_t = \nabla_w \mathcal{L}$：半精度算出的梯度； $u(\cdot)$：optimizer 更新方向（SGD 下 $u = g$ ）； $\eta$：学习率。前向与反向用 $w$ 跑，梯度回流到 fp32 的 $\theta$ 上累加。

master weights 为什么必须 fp32？论文 §3.1 给出两个相互独立的机制，都可精确算账。

**机制一：更新量下溢**。fp16 最小次正规数是 $2^{-24}$ ，比它小的值在 fp16 中就是零。权重梯度本身已经偏小：论文在 Mandarin 语音识别模型的梯度直方图（其 Figure 2(b)）上量出约 5% 的权重梯度值指数小于 −24，再乘 $\eta$（典型 $10^{-3}$ 量级，约合 $2^{-10}$ ）之后下溢面更宽。如果更新直接作用在 fp16 权重上，这部分坐标每步原地不动，长期累积成可观的精度损失。

**机制二：大数吃小数**。浮点加法先对齐阶码：小数的有效位右移 $\Delta e$ 位（ $\Delta e$ 为两数指数差）。fp16 有效位共 11 位（1 位隐含 + 10 位尾数）：当 $|\theta| / |\Delta\theta| > 2048 = 2^{11}$ 时， $\Delta\theta$ 被右移至少 12 位，全部移出、舍入不可恢复地为零（右移恰好 11 位时，部分情形可由舍入救回）。fp32 有效位 24 位，同法算出安全比值上限 $2^{24} \approx 1.7 \times 10^{7}$ ——同样的更新量在 fp32 主副本上不会丢。论文用 Mandarin 模型做了对照：更新 fp32 主副本则与纯 fp32 训练曲线重合，直接更新 fp16 权重则相对精度损失 80%。

内存账也要算公平：master 副本使权重存储较纯 fp32 训练多 50%，但训练显存的大头是激活（models 1.2 账二）。激活同样转半精度后，论文口径总显存「roughly halved」。

第三件套是 **fp32 累加**：半精度相乘、fp32 累加器。两个 fp16 相乘是精确的——有效位至多 11 + 11 = 22 位，不超 fp32 的 24 位，指数范围也够——误差全部来自 $K$ 次累加。fp32 累加器把每次加法的相对误差压在 $2^{-24}$ 量级。Tensor Core 的 GEMM 数据通路正是这个结构；第 6 节会看到，FP8 时代连这条都要重新审视。

## 3. Loss scaling：下溢界、恒等式与进退规则

fp16 的正规数指数居中在 $[2^{-14}, 2^{15}]$ ，而实际梯度直方图由小幅度值主导（负指数占绝大多数）。[Micikevicius et al., 2017](../references.md)（[B]，§3.2）用 Multibox SSD 检测器的激活梯度直方图展示了这一错位：fp16 可表范围的大半空着，而梯度直方图的长尾落在最小可表值之下、被压成零。loss scaling 的修正在数学上是一行恒等式。把 loss 乘上常数 $S$ ，由梯度的线性性：

$$
\widetilde{\mathcal{L}} = S \cdot \mathcal{L} \quad\Longrightarrow\quad \nabla_{w} \widetilde{\mathcal{L}} = S \cdot \nabla_{w} \mathcal{L}, \qquad \theta - \eta \cdot \frac{\nabla_{w} \widetilde{\mathcal{L}}}{S} = \theta - \eta \cdot \nabla_{w} \mathcal{L} \qquad (3)
$$

**符号**： $S > 0$：loss scale； $\widetilde{\mathcal{L}}$：放大后的 loss。右式的含义：在实数算术中，先放大再除回是恒等操作，什么都没变。全部工程内容在**哪一步以什么精度执行**：反向传播的整条链在放大 $S$ 倍的量级上计算，远离 $2^{-24}$ 的下溢下界；除回 $1/S$ 发生在更新前的最后一刻，且除法在 fp32 侧做。常数 $S$ 穿过反向链逐节点传播（每个 vjp 都是线性的），不需要逐算子改动。

$S$ 的可行窗口由两界夹出。记该步全部梯度中最大绝对值为 $g_{\max}$ 、需要保留的最小梯度为 $g_{\min}$ 。上端不溢出，要求 $S \cdot g_{\max} \lt 65504$ ；下端不压死，要求 $S \cdot g_{\min}$ 不低于次正规下界 $2^{-24}$ （更保守的口径是正规下界 $2^{-14}$ ——次正规区有效尾数随幅度递减，精度已经在丢）：

$$
\frac{2^{-24}}{g_{\min}} \le S \lt \frac{65504}{g_{\max}} \qquad (4)
$$

**符号**：左界取次正规下界是宽松口径（次正规区精度已有损失，保守口径换 $2^{-14}$ ）；窗口存在当且仅当 $g_{\max} / g_{\min} \lt 65504 \cdot 2^{24} \approx 1.1 \times 10^{12}$ ——梯度分布的动态范围只要不超过 40 个二倍程（约 12 个十进制数量级），窗口就非空。论文报告常数 scale 在 8 到 32768 之间对多个网络有效（许多网络甚至不需要 scaling），选取规则即式 (4) 的上端。由于 unscale 在 optimizer step 之前完成，梯度裁剪阈值、weight decay 等超参一律不用改（[Micikevicius et al., 2017](../references.md)，[B]，§3.2）。

固定 $S$ 不知道 $g_{\max}$ 的先验。论文在展望部分给出动态方案：监测梯度是否出现溢出（inf / NaN），溢出则跳过本次更新并调小 $S$ 。PyTorch 的 `GradScaler` 把这条规则参数化实现（pytorch/pytorch repo，`torch/amp/grad_scaler.py`，类 `GradScaler` 的 `scale` / `unscale_` / `step` / `update`，[PyTorch, pytorch/pytorch repo](../references.md)，教学参照 + 官方实现）：

$$
S_{t+1} = \begin{cases} S_{t} / 2 & \text{inf/NaN found (skip step)} \\ 2 \thinspace S_{t} & \text{2000 clean steps in a row} \\ S_{t} & \text{otherwise} \end{cases} \qquad (5)
$$

**符号**：默认值出自源码签名：初始 $S_0 = 2^{16}$ ，回退因子 0.5，增长因子 2.0，增长间隔 2000 步。语义很直白： $S$ 越大，梯度下溢越少，但离溢出上限越近。算法把 $S$ 维持在「偶尔溢出」的边缘：`step` 先 unscale 并检查，发现 inf/NaN 就跳过本次更新（参数不受污染），`update` 再按式 (5) 调价。文档口径：校准期的头若干步跳过属正常，稳态后每几百至几千步至多跳一次。

一个完整的 fp16 AMP 训练步（教学用，非优化实现）：

```python
# 教学用，非优化实现：fp16 混合精度一个 step 的骨架
scaler = torch.amp.GradScaler("cuda")          # init_scale=2**16, growth_interval=2000

for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        # autocast 按算子维护 cast 策略：GEMM / conv 类降为 fp16，
        # softmax / 交叉熵 / norm 类保持 fp32（torch/amp/autocast_mode.py）
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()   # loss *= S，式 (3) 的放大
    scaler.unscale_(optimizer)      # 梯度 /= S 并检查 inf/NaN（裁剪须在 unscale 之后）
    scaler.step(optimizer)          # 有 inf/NaN 则跳过本次更新
    scaler.update()                 # 式 (5)：溢出回退或按计数增长
```

数据流对照（式 (2)(3) 的合流）：

```
 fp32 主权重 θ（optimizer 持有）
     │ round_16                                   ┌─── S·g（fp16 梯度）
     ▼                                            │    │ unscale: ÷S + inf/NaN 检查
 fp16 副本 w ──► 前向（GEMM fp16×fp32 累加）──► L │ ×S  ▼
     │                                   backward┘   fp32 梯度 g
     └── fp16 激活存给反向（1.4 的账）              └──► optimizer.step(θ)
```

## 4. bf16：用尾数换指数，scaling 消失

bf16 是 fp32 的截断：同样 8 位指数，只留 7 位尾数。对照第 1 节的表，它的动态范围与 fp32 完全相同（最小正规数 $2^{-126}$ ）。第 3 节那种「梯度由 $2^{-20}$ 量级的小值主导」的分布，离下溢下界有上百个二倍程——下溢与溢出在训练中都不构成约束，loss scaling 整个机制随之消失。这就是现代 LLM 预训练默认 bf16 而非 fp16 的原因：少一套需要在线校准、溢出时跳步的活动部件。

代价在横轴的另一端：machine epsilon $2^{-7} \approx 7.8 \times 10^{-3}$ ，是 fp16 的 8 倍。两个配套事实因此不能省。其一，master weights 更不能省。第 2 节机制二的有效位论证原样适用：bf16 有效位 8 位，安全比值上限从 fp16 的 $2^{11} = 2048$ 收紧到 $2^{8} = 256$ —— $|\theta| / |\Delta\theta|$ 超过 256 的坐标在 bf16 直更下原地不动，比 fp16 苛刻一档。其二，累加照旧 fp32。bf16 乘积同样精确——有效位 8 位（含隐含位），乘积至多 16 位，不超 fp32 的 24 位——误差全在累加，与 fp16 情形同构。

optimizer 内部各状态的精度需求并不均匀。DeepSeek-V3 的口径（arXiv:2412.19437，[A]，§3.3.3）是一份现成的分级实例：AdamW 的一阶、二阶矩估计降到 BF16 存储（「without incurring observable performance degradation」），master weights 与梯度（梯度要跨 micro-batch 累加，归 1.5）保持 FP32（「to ensure numerical stability」）。读法与本章机制一致：矩估计只参与求更新方向、短期重生，容忍低精度；主权重承担上万步的长期累加，是机制二的直接暴露面，必须 fp32。

## 5. FP8：E4M3/E5M2 分工与 per-tensor scaling

FP8 只有 8 位：E4M3 与 E5M2 两种格式把同一份预算花在两头（第 1 节表）。FP8 格式论文的推荐分工（[Micikevicius et al., 2022](../references.md)，arXiv:2209.05433，[B]，§3 原话）是：**E4M3 用于权重与激活，E5M2 用于梯度**。前向张量幅度集中、吃相对精度（3 位尾数）；梯度分布宽、吃动态范围（5 位指数）。部分网络可只用其中一种格式训练。该论文同时给出三条配套口径：

1. per-tensor scaling：选缩放因子使张量最大幅度贴近格式最大值，超出者饱和到最大值；
2. fp16 式动态跳步在 FP8 下不可用——动态范围太窄、溢出太频繁，「resulting in too many skipped updates」；
3. 反量化在 GEMM 中每个点积只做一次，被多次乘加摊薄。

实证口径：把 GEMM 输入（激活、权重、激活梯度）FP8 化后，图像分类与机器翻译的一批网络，训练精度落在高精度基线的 run-to-run 波动之内。

FP8 的误差来源可以拆成三项，它们恰好对应下一节 DeepSeek-V3 的三件对策：

1. **量化误差**：尾数只有 2 至 3 位，组内最大元素的相对舍入误差上限 $2^{-4}$（E4M3）或 $2^{-3}$（E5M2）；
2. **范围误差**：per-tensor 共用一个 scale，一个 outlier 通道就把 scale 抬高，全张量的小元素被压向次正规下界（E4M3 为 scale 的 $2^{-9}$ 倍）之下归零；
3. **累加误差**：8 位输入的 GEMM 若累加器位宽不足， $K$ 维点积的相对误差随 $K$ 增长（第 6.4 节的实测数字）。

## 6. DeepSeek-V3 的 FP8 方案全拆（论文 §3.3 口径）

DeepSeek-V3 的预训练全程（14.8T token）在其自研 FP8 混合精度框架下完成。方案先在两个较小规模（与 DeepSeek-V2-Lite、DeepSeek-V2 相当）上各训约 1T token 验证：相对 BF16 基线的 loss 偏差稳定低于 0.25%（「well within the acceptable range of training randomness」）。以下全部口径出自 [DeepSeek-AI, 2024](../references.md)（arXiv:2412.19437，[A]，§3.3），按误差来源分组。

### 6.1 框架：算子级精度分配

Linear 算子的三个 GEMM——Fprop（前向）、Dgrad（激活反向）、Wgrad（权重反向）——全部以 FP8 执行：输入 FP8，输出 BF16 或 FP32。论文口径，这条设计「theoretically doubles the computational speed compared with the original BF16 method」。Wgrad 走 FP8 还带来一个副产品：前向激活可以 FP8 缓存供反向使用（1.4 的激活账随之减半：2 字节变 1 字节）。保持高精度的算子清单：embedding、输出头、MoE gating、normalization、attention——这些要么对低精度敏感，要么计算占比低、升精度代价可忽略。master weights、weight gradients、optimizer states 同样保持高精度（存储细节见第 4 节与 §6.5）。

### 6.2 全部张量 E4M3（Mantissa over Exponents）

前作的 hybrid 方案按第 5 节的分工执行：E4M3 用于 Fprop，E5M2 用于 Dgrad 与 Wgrad（DeepSeek-V3 论文对 NVIDIA 2024b / Peng et al. / Sun et al. 方案的描述）。V3 的选择相反，论文原话「we adopt the E4M3 format on all tensors for higher precision」。可行性归因于细粒度量化。E4M3 的弱点是动态范围只有 18 个 binade；小组量化让同组元素共享指数位——每组自带 scale，组内只需覆盖该组的局部范围，格式全局范围的短板就这样被粒度补足。

### 6.3 Fine-grained quantization：粒度账

量化粒度：激活按 $1 \times 128$ tile 分组（每 token 每 128 通道一个 scale），权重按 $128 \times 128$ 块分组（每 128 输入通道 × 128 输出通道一个 scale）。scale 在线计算（online quantization）：每组当场取最大绝对值求 scale 再量化；对照前作的 delayed quantization——维护历史迭代的 amax 序列外推当前值（论文对 tensor-wise 框架的描述）。量化与误差的形式化：

$$
s_g = \frac{\max_{i \in g} |x_i|}{448}, \qquad \hat{x}_i = Q_{\mathrm{E4M3}}\left( \frac{x_i}{s_g} \right), \qquad x_i \approx s_g \cdot \hat{x}_i \qquad (6)
$$

**符号**： $g$：一个量化组（tile 或块）； $s_g$：该组的 fp32 scale； $Q_{\mathrm{E4M3}}$：舍入到 E4M3 可表值；448：E4M3 最大正规数。误差结构：浮点量化是尺度不变的， $x_i$ 的相对误差由其所在 binade 的 ulp 决定（正规区上限 $2^{-4}$ ），与 $s_g$ 无关；真正的损失发生在幅度跌破该组次正规下界 $2^{-9} \cdot s_g$ 时——归零。per-tensor 与 blockwise 的差别全在这个下界上：per-tensor 时 $s_g$ 由全张量最大值（含 outlier）定，下界对全张量生效；blockwise 时 outlier 只抬高自己所在 $128 \times 128$ 块的 scale，其余块的小元素保住。

粒度开销是可以算的。权重块 $128 \times 128$ = 16384 个 FP8 元素（16384 字节）配一个 fp32 scale（4 字节）：开销 $4 / 16384 \approx 0.024$% 。激活 tile $1 \times 128$ ：128 字节数据配 4 字节 scale，开销 3.1%。粒度每细一倍，scale 存储按维数平方级上涨；128 这个档位是误差收益与存储开销的折中点。论文附录 B.2 另记录：激活若像权重一样按块分组会引发训练不稳定，故激活沿 token 维取 $1 \times 128$ 。

### 6.4 高精度累加：每 128 个元素提升一次

第三项误差来源在硬件里。V3 团队实测，H800 的 FP8 GEMM 在 Tensor Core 内的累加精度「limited to retaining around 14 bits」——对照 fp32 累加的 24 位尾数。在 $K = 4096$ 的随机矩阵 GEMM 预实验中，最大相对误差接近 2%，且随 $K$ 增大而恶化（大 batch、宽模型的常态）。对策是把部分和周期性地提升到 CUDA Core：每累计 $N_C = 128$ 个元素（恰合 4 条 WGMMA 指令），把 Tensor Core 内的低位宽部分和搬进 CUDA Core 的 fp32 寄存器做全精度累加；per-group scale 的反量化乘子融合在这一步完成，代价可控。重叠设计：H800 上两条 WGMMA 并发执行，一个 warpgroup 做提升时另一个继续 MMA，Tensor Core 利用率不受提升操作拖累。论文口径： $N_C = 128$ 是在不引入实质开销的前提下切实提高精度的最小间隔。

### 6.5 存储与通信的低精度化

激活缓存：Linear 的输入以 FP8 缓存供 Wgrad 使用（§6.1 的副产品落地），有两处例外。第一，attention 之后的 Linear 输入：这份激活同时被 attention 反向消费、对精度敏感，专用自定义 E5M6 格式存储；反向时量化 tile 从 $1 \times 128$ 转为 $128 \times 1$ ，全部 scale 取 2 的整数幂（round scaled）——幂次 scale 的乘除是精确的指数加减，不再引入额外量化误差。第二，MoE 中 SwiGLU 算子的输入：存 FP8、反向时重算其输出（重计算换显存的一次局部应用，机制归 1.4）。通信：MoE up-projection 之前的激活量化到 FP8 再 dispatch，down-projection 之前的激活梯度同法；前向与反向的 combine 环节保持 BF16 不动。optimizer 态：矩估计 BF16、master weights 与梯度 FP32（第 4 节）。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 浮点格式 | 式 (1)；五种格式对照表 | 指数位买范围、尾数位买精度；E4M3 无 inf 换 18 个 binade |
| 三件套 | 式 (2) | fp32 master（更新下溢 $2^{-24}$ 、比值上限 $2^{11}$ / $2^{8}$ / $2^{24}$ ）+ 半精度前向反向 + fp32 累加 |
| loss scaling | 式 (3)(4)(5) | 放大-除回恒等；窗口 $2^{-24}/g_{\min} \le S \lt 65504/g_{\max}$ ；GradScaler： $2^{16}$ 起、溢出减半跳步、2000 步翻倍 |
| bf16 | 8 指数位 = fp32 同范围 | 免 scaling；master 仍须 fp32（界收紧到 256）；LLM 默认组合 |
| FP8 分工 | E4M3 权重/激活、E5M2 梯度（2209.05433） | per-tensor scaling + 饱和；跳步法不可用 |
| DeepSeek-V3 | 式 (6)； $N_C = 128$ | 全 E4M3； $1 \times 128$ 激活 / $128 \times 128$ 权重块；在线量化；14 位累加 → CUDA Core 提升；E5M6 例外；验证 0.25% |

## 练习

1.（推导）把第 2 节机制二推广成一般形式：设格式有效位 $p$（含隐含位），证明 $|\theta| / |\Delta\theta| > 2^{p}$ 时更新量在加法对齐后不可恢复地为零；代入 $p$ = 11、8、24 复核 2048、256、 $2^{24}$ 三档。再说明为什么「更新 fp32 主副本」同时修掉机制一与机制二。
2.（计算）某层梯度幅度落在 $[2^{-20}, 2^{-3}]$ ：(a) 用式 (4) 求 fp16 下 $S$ 的可行窗口（两种下界口径各算一次）；(b) 若实际梯度峰值在 $2^{0}$ 附近而 $S$ 从 GradScaler 默认的 $2^{16}$ 起步，推演前三步的溢出检测、跳步与回退过程，给出 $S$ 的轨迹。
3.（计算）Llama-3.1-8B 的 FFN up 投影权重为 $14336 \times 4096$（models 1.2 实例）：分别计算 per-tensor 与 $128 \times 128$ blockwise 量化的 scale 总字节数与相对 FP8 数据的开销占比；若该矩阵存在一个幅度为典型值 100 倍的 outlier 通道，定性说明两种粒度下次正规下界 $2^{-9} s_g$ 对小元素的影响差异。
4.（代码）教学用：按第 3 节骨架跑一个 fp16 + GradScaler 的最小训练循环，在某一步人为注入会产生 inf 的 loss，打印 scale 轨迹与 step 跳过事件；再切到 bf16 去掉 scaler 重跑，对照第 4 节的论证。

## 参考

- [Micikevicius et al., 2017](../references.md)：Mixed Precision Training，arXiv:1710.03740，[B]（key: mixed-precision）——三件套、下溢界 $2^{-24}$ 、比值 2048、常数 scale 8 至 32768、动态跳步建议（展望节）
- [Micikevicius et al., 2022](../references.md)：FP8 Formats for Deep Learning，arXiv:2209.05433，[B]（key: fp8-formats）——E4M3/E5M2 位布局与特殊值设计（Table 1、§3.1）、推荐分工、per-tensor scaling 与跳步不适用的论证
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——§3.3 全部口径：全 E4M3、细粒度量化粒度、 $N_C$ = 128 提升累加、14 位累加实测、E5M6 例外、2 幂 scale、optimizer 态分级、0.25% 验证
- [PyTorch, pytorch/pytorch repo](../references.md)：教学参照 + 官方实现（key: pytorch-repo）——`torch/amp/grad_scaler.py`（式 (5) 默认参数与跳步逻辑）、`torch/amp/autocast_mode.py`（autocast 上下文与算子 cast 策略）
