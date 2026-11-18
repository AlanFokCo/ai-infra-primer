# 7.2 预训练配方：lr / bsz / warmup / decay / µP（Pretraining Recipes: LR, Batch Size, Warmup, Decay, and µP）

> 本章把 P2 的理论（2.2 的 AdamW、2.3 的 schedule 与噪声尺度、2.5 的 Muon）落成一张七家真实配方对照表：GPT-3 / Llama-2 / Llama-3 / DeepSeek-V3 / OLMo 2 / GLM-4.5 / Qwen3 的 lr、batch（tokens 口径）、warmup、decay、wd、 $\beta$ 、 $\epsilon$ 、clip，逐字段带节号级出处；披露口径的差异也入表（Qwen3 只披露方法、不披露数值）。表后是三块推导与对照：batch ramp-up 的步数账与噪声尺度读法（DeepSeek-V3 与 GLM-4.5 两家 ramp 实证并列）、µP 的宽度缩放结论表与推导 outline、µTransfer 的成本账，外加 Moonlight「RMS 匹配替代调参」的对照哲学 | 前置：2.2（AdamW 超参结构）、2.3（schedule 族、critical batch size、batch ramp 的规律陈述）、2.5 §4-5（Moonlight 更新式与 RMS 匹配）、1.5（tokens 换算口径）；models 1.3（Chinchilla 分配律，纯文字指引） | 锚点：OLMo 2 [A]（全开放配方）、DeepSeek-V3 [A]、Llama-3 herd [A]、GLM-4.5 [A]、Tensor Programs V（[Yang et al., 2022](../references.md)，arXiv:2203.03466，[B]）、Moonlight [A]、GPT-3 [B]、Llama-2 [A]、MiniCPM [A]

## 1. 七家配方对照表

这张表是本章的核心对象。先立口径：batch 一律按 1.5 式 (5) 折成每步 tokens 数（序列数 × 序列长）；lr 指峰值；「步」指 optimizer step。逐字段出处见表下注；原文没披露的一律标「未披露」，不用二手转述补齐。

| 字段 | GPT-3 175B [B] | Llama-2 70B [A] | Llama-3 405B [A] | DeepSeek-V3 671B/37B [A] | OLMo 2 7B [A] | GLM-4.5 355B/32B [A] | Qwen3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| optimizer | Adam（原文口径） | AdamW | AdamW | AdamW | AdamW | Muon（embedding / bias / RMSNorm 增益除外） | 未披露 |
| 峰值 lr | $0.6 \times 10^{-4}$ | $1.5 \times 10^{-4}$ | $8 \times 10^{-5}$ | $2.2 \times 10^{-4}$ | $3 \times 10^{-4}$ | $2.5 \times 10^{-4}$ | 未披露 |
| batch（tokens/step） | 3.2M | 4M | 4M → 8M → 16M（两次翻倍） | 12.6M → 62.9M（线性 ramp） | 4.19M（1024 序列 × 4096） | 16M → 64M（ramp） | 未披露 |
| warmup | 线性，前 375M tokens | 线性 2000 步 | 线性 8000 步 | 线性 2000 步 | 线性 2000 步 | 有 warmup 段，步数未披露 | 未披露 |
| decay 与下限 | cosine，260B tokens 内降至峰值 10%，其后恒定 | cosine 降至峰值 10% | cosine，1.2M 步内降至 $8 \times 10^{-7}$ | 恒定至 10T tokens → cosine 4.3T 降至 $2.2 \times 10^{-5}$ → 末 500B 两段恒定（ $2.2 \times 10^{-5}$ / 333B， $7.3 \times 10^{-6}$ / 167B） | cosine 按 5T tokens 标定、4T 处截断；mid-training 线性降至 0 | cosine 降至 $2.5 \times 10^{-5}$（至 mid-training 末） | 未披露 |
| weight decay | 0.1 | 0.1 | 当步 lr 的 0.1 倍（scaling-law 实验口径） | 0.1 | 0.1（embedding 除外） | 0.1 | 未披露 |
| $\beta_1$ / $\beta_2$ | 0.9 / 0.95 | 0.9 / 0.95 | 未披露 | 0.9 / 0.95 | 0.9 / 0.95 | —（Muon 动量 $\mu = 0.95$ ） | 未披露 |
| $\epsilon$ | $10^{-8}$ | $10^{-5}$ | 未披露 | 未披露 | $10^{-8}$ | —（Muon 无此参数） | 未披露 |
| 梯度裁剪 | 1.0 | 1.0 | 未披露 | 1.0 | 1.0 | 未披露 | 未披露 |
| tokens 总量 | 300B | 2.0T | 15.6T | 14.8T | 4T + mid-training 3×50B | 23T（含 mid-training） | 36T |
| 初始化 | 未披露 | 未披露 | 未披露 | 全部参数 std 0.006 | 截断正态 std 0.02 | 未披露 | 未披露 |

逐家出处（均为报告原文节号，数字经 ar5iv 全文逐条核对）：

- **GPT-3**：[Brown et al., 2020](../references.md)，arXiv:2005.14165，[B]。lr/batch 缩放见 Table 2.1（175B 行：3.2M tokens、 $0.6 \times 10^{-4}$ ；125M 行：0.5M tokens、 $6.0 \times 10^{-4}$ ）；optimizer、warmup、decay、ramp、wd、clip 见 App. B。原文写「Adam」，未说明 wd 的解耦语义——2.2 §7 的纪律注记同样适用。300B tokens 见 Table 2.1 表注。
- **Llama-2**：[Touvron et al., 2023](../references.md)，arXiv:2307.09288，[A]。optimizer 与 $\beta$ 、 $\epsilon = 10^{-5}$ 、warmup 2000 步、cosine 降至峰值 10%、wd 0.1、clip 1.0 均见 §2.2「Hyperparameters」段；4M tokens 的全局 batch 见 Table 1 表注；lr $1.5 \times 10^{-4}$ 为 34B/70B 档（7B/13B 档为 $3 \times 10^{-4}$ ，Table 1）；2.0T tokens 见 §2.1。
- **Llama-3**：[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]。405B 的峰值 lr、warmup 8000 步、cosine 降至 $8 \times 10^{-7}$ / 1.2M 步、batch 两次翻倍（4M → 8M → 16M tokens，节点在 252M 与 2.87T tokens——「252M」按原文照录，量级存疑，同 2.3 §5 的标注）见 §3.4.1；wd = 0.1 × 当步 lr 是 §3.2.1 scaling-law 实验的口径；optimizer 超参未披露（不可引为 $\beta_2 = 0.95$ 的出处，2.2 §7 同纪律）；15.6T tokens 见 §1 引言。
- **DeepSeek-V3**：[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]。§4.2 全段：AdamW（ $\beta_1 = 0.9$ 、 $\beta_2 = 0.95$ 、wd 0.1）、warmup 2000 步、峰值 $2.2 \times 10^{-4}$ 、恒定至 10T、cosine 4.3T 降至 $2.2 \times 10^{-5}$ 、末 500B 两段恒定、clip 1.0、batch 3072 → 15360 序列（seqlen 4K，折 12.6M → 62.9M tokens）于前 469B tokens 线性 ramp。初始化 std 0.006 见同节（Model Hyper-Parameters）； $\epsilon$ 未披露；14.8T tokens 见 §4.1/§4.2。
- **OLMo 2 7B**：[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]。Table 4（batch 1024 × 4096、clip 1.0、峰值 $3 \times 10^{-4}$ 、warmup 2000 步、cosine 按 5T 标定 4T 截断；13B 档 batch 2048、峰值 $9 \times 10^{-4}$ ）； $\beta$ 值与 wd 0.1 为 OLMo-core 发布配置口径（2.2 §7 已核）； $\epsilon = 10^{-8}$ 的改动与动机见 §3.4.1；embedding 免 wd 见 §3.4.2；初始化 0.02 截断正态见 §2.3 与 §3.2；mid-training 线性降至 0、3 × 50B tokens 退火加模型平均见 §2.3/§4。
- **GLM-4.5**：[GLM Team, 2025](../references.md)，arXiv:2508.06471，[A]。§2.4 全段：Muon（Newton-Schulz 5 步、动量 0.95、update RMS 匹配 0.2，embedding/bias/RMSNorm 增益除外——这些参数的优化器原文未点名）、cosine（明确不用 WSD，原文给出 early experiments 的欠拟合理由）、warmup 0 → $2.5 \times 10^{-4}$ 、衰减至 $2.5 \times 10^{-5}$ 、batch ramp 16M → 64M tokens（前 500B tokens）、wd 0.1、无 dropout。23T tokens 为摘要口径（含 mid-training）；warmup 步数与梯度裁剪未披露。
- **Qwen3**：[Qwen Team, 2025](../references.md)，arXiv:2505.09388，[A]。36T tokens、119 种语言。超参的披露口径是方法而非数值：「we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions……we set the predicted optimal learning rate and batch size strategy for each dense or MoE model」——lr 与 batch 由 scaling-law 拟合预测，具体数值未披露。

读这张表的第一条纪律：把「未披露」当作表的一部分来读。七家里只有 OLMo 2 一家全字段开放，连 config 与数据都可下载。披露口径本身就是配方研究的可观测对象：披露越少的字段（ $\epsilon$ 、初始化、Qwen3 整行），越是各家最想守住的部分。

表内还有三组结构性读法。

第一组，**decay 形态的分化**。cosine 一家占五席（GPT-3、Llama-2、Llama-3、OLMo 2 主段、GLM-4.5）；DeepSeek-V3 是「恒定平台 + cosine + 双段常数尾」的梯形变体（2.3 §3 的 WSD 读法）；OLMo 2 的收尾独一档：cosine 不按计划跑完，4T 处截断后转入 mid-training，lr 线性降到零再加模型平均。下限口径分两派：收到峰值的 10%（GPT-3、Llama-2；GPT-3 其后恒定在 10%），或收到接近零（Llama-3 的 $8 \times 10^{-7}$ 为峰值的 1%，OLMo 2 mid-training 直接到 0）。2.3 §3 已注明无公开消融支持统一结论，此处按原样并列。

第二组，**warmup 的占比**。按步数计的两家可比：DeepSeek-V3 的 2000 步对总日程（14.8T / 62.9M ≈ 23.5 万步）约 0.9%，Llama-3 的 8000 步对 1.2M 步约 0.7%，都在 2.3 §2 给的「1% 以内」区间内。

第三组，**wd 与 clip 的收敛**。wd = 0.1 与 clip = 1.0 在全部披露字段里无一偏移（Llama-3 的 0.1×lr 写法在 schedule 意义上与之相容——2.2 §6 的面积读法）。这两个字段是事实上已被冻结的惯例；分歧集中在 lr、batch、decay 形状这三个与规模和预算耦合的字段上。

把四种 decay 结构画在 tokens 轴上（纵轴 lr 归一到峰值，横轴不按比例）：

```
lr/peak
1.0 ┤ DeepSeek-V3:  ╭──────────────╮            OLMo 2 7B:
    │              (恒定 10T)      ╲ cosine         (cosine 按 5T 标定)
    │                              ╲ 4.3T      ╭─╮
    │                               ╲         ╱   ╲ 4T 处截断
0.1 ┤                                ╲      ╱     ┃━━━━ mid-training
    │   Llama-3 405B:                 ╲━━━━      ┃     线性 → 0
    │     ╭╮                          常数尾     ┃     (3×50B + 平均)
    │    ╱  ╲  cosine                   (500B)   ┃
    │   ╱    ╲  1.2M 步                      GLM-4.5:
    │  ╱      ╲ 至 1%                        (cosine 全程,
0   └────────────────────────► tokens         至 mid-training 末)
    warmup 段全部 <1% 日程（图中略）
```

四种形态共享的只有一件事：峰值之前都有一段短 warmup，峰值之后都单调不增。在此之上，cosine（Llama-3、GLM-4.5）、梯形加常数尾（DeepSeek-V3）、截断转线性归零（OLMo 2）三种分歧，没有公开的跨家消融可以裁决。2.3 §3 的 MiniCPM 风洞证据支持两条：「decay 段占约 10% tokens 足够」、「梯形 vs cosine 同预算等价」。各家形状差异在当前证据等级下应读作工程偏好，不是性能结论。

## 2. 读表：tokens 口径与两条规模规律

跨表比较前先做口径换算（1.5 §2 立的规矩）：batch 只认 tokens/step， $B_{\mathrm{tok}} = b \cdot m \cdot d \cdot T$ 。表内的 batch 有四种报告形态：GPT-3 直接按 tokens 报告；Llama-2/3 按 tokens 报告；DeepSeek-V3 按序列数报告（3072/15360 × 4096 折算）；OLMo 2 按序列数报告（1024/2048 × 4096 折算）。全部折成同一列之后才可比较。schedule 的横轴也要换算：GPT-3 的 warmup 按 tokens 计（375M），其余按 step 计，两者相差一个随时间变化的 batch 因子——ramp 期尤其不可混（§3 的账）。读表的标准动作顺序：先折 tokens 口径；再按模型规模排序读 lr 与 batch 两列；然后把 decay 形状与 tokens 预算对着读（decay 的总长在 token 轴上才有意义）；最后核 optimizer 字段（ $\beta_2$ 、 $\epsilon$ 、clip）——这三个字段的变化由稳定性动机驱动（2.2 §4 的 OLMo 2 $\epsilon$ 实验即一例），与规模规律无关。

两条规模规律在表上直接可读（2.3 §6 已立，此处补全字段）：

1. **lr 随规模降**。GPT-3 一族从 125M 的 $6.0 \times 10^{-4}$ 单调降到 175B 的 $0.6 \times 10^{-4}$（Table 2.1 八档全部单调）。跨家族横向看：70B 级 $1.5 \times 10^{-4}$ ，405B 级 $8 \times 10^{-5}$ ，671B 级 MoE $2.2 \times 10^{-4}$ 。MoE 按激活参数读数（37B）偏高半个量级，按总参读数偏低——这个错位本身就是 MoE lr 口径尚未收敛的信号。表内唯一反向数据点是 OLMo 2：13B 的峰值（ $9 \times 10^{-4}$ ）高于 7B（ $3 \times 10^{-4}$ ），且 13B 跑满 5T cosine 未截断。这是原文披露的事实，不强行并入规律。
2. **batch 随规模涨**。GPT-3 一族 0.5M → 3.2M tokens；2024–2025 落点 4M（7B 级）→ 16M（405B）→ 62.9M（671B MoE）。机制读法归 2.3 §5：噪声尺度在同 loss 水平下与模型尺寸近似无关、随训练推进而涨；更大模型推进到更低的 loss，工作点随之移向大 batch 区； $\epsilon_{\max}$ 则受稳定性约束随规模收紧。

一个工作例把换算流程走一遍。DeepSeek-V3 的 batch 报告值是「15360 序列」：折 tokens 为 $15360 \times 4096 \approx 62.9$ M tokens/step。这是 1.5 式 (5) 的口径—— $b \cdot m \cdot d$ 的分配原文未披露、也不必披露，等效 batch 才是配方语义的全部。全程 14.8T tokens 对应约 23.5 万个 optimizer step（按末值 batch 估下界）。同理，Llama-3 405B 的 1.2M 步日程按末值 16M tokens/step 折算约 19.2T tokens 的吞吐上限，与披露的 15.6T 同量级；差额由 ramp 期的小 batch 吸收（ramp 段约 33 万步运行在 16M 以下；长上下文阶段 batch 恒为 16M，herd Table 4 口径，不参与缺口）。这类「步数 × batch = tokens」的交叉验算是读任何配方表的第一动作：两个口径对不上，多半是有未写明的 ramp 或中途换过序列长。

## 3. Batch ramp-up：两家 ramp 实证与步数账

表中有四家显式做了 batch ramp-up（训练前期用小 batch，随后调大）：GPT-3（32k tokens 线性升至全值，前 4–12B tokens，随规模而异，App. B）、DeepSeek-V3（3072 → 15360 序列，前 469B tokens，§4.2）、Llama-3 405B（两次翻倍，§3.4.1）、GLM-4.5（16M → 64M tokens，前 500B tokens，§2.4）。把 ramp 段单独算账。本书计算的口径：设 batch 随已消费 tokens 线性增长， $B(\tau) = B_0 + (B_1 - B_0) \thinspace \tau / D_r$ 。则 ramp 段的 optimizer step 数为

$$
S_{\mathrm{ramp}} = \int_{0}^{D_r} \frac{d\tau}{B(\tau)} = \frac{D_r}{B_1 - B_0} \thinspace \ln \frac{B_1}{B_0} \qquad (1)
$$

**符号**： $D_r$：ramp 段消耗的 tokens； $B_0, B_1$：ramp 起止的 tokens/step； $S_{\mathrm{ramp}}$：ramp 段的 optimizer step 数。代入 DeepSeek-V3（ $D_r = 469\text{B}$ 、 $B_0 = 12.6\text{M}$ 、 $B_1 = 62.9\text{M}$ ）： $S_{\mathrm{ramp}} \approx 469 \times 10^{9} / (50.3 \times 10^{6}) \times \ln 5 \approx 9320 \times 1.609 \approx 15000$ 步。若同段 tokens 按恒定 62.9M 跑，只需约 7450 步——ramp 把该段的更新次数放大到约 2.0 倍。GLM-4.5（ $D_r = 500\text{B}$ 、16M → 64M）： $\approx 500 \times 10^{9} / (48 \times 10^{6}) \times \ln 4 \approx 10400 \times 1.386 \approx 14400$ 步；恒定 64M 口径约 7800 步，放大 1.84 倍。

ramp 的账面效果永远是「同样的 tokens、更多的更新」。它为什么无害乃至有益？答案在 2.3 §5 的噪声尺度框架：最优步长下的单步期望进展 $`\Delta L_{\mathrm{opt}}(B) = \Delta L_{\max} / (1 + \mathcal{B}_{\mathrm{noise}} / B)`$（2.3 式 (7)）。训练早期 $`\mathcal{B}_{\mathrm{noise}}`$ 小， $`B \gg \mathcal{B}_{\mathrm{noise}}`$ 时每多花一倍的 tokens 只换来可忽略的单步进展增量，把这段 tokens 拆成更多小步反而接近无损。反过来，ramp 终点的 $B_1$ 由墙钟效率决定（大 batch 摊薄通信与 kernel 固定开销，4.x 的账）。ramp 就是把 batch 从「早期不亏」平滑推到「晚期必须」的那条曲线。DeepSeek-V3 的 ramp 段占总预算 469B / 14.8T ≈ 3.2%，GLM-4.5 为 500B / 23T ≈ 2.2%——两家都把 ramp 收在训练的前几个百分点内，与 2.3 §5「噪声尺度早期增长最快」的漂移规律同向。

把上面的效率论证再压一步：单步期望进展除以单步成本（tokens），得到按 token 计的进展率

$$
\frac{\Delta L_{\mathrm{opt}}(B)}{B} = \frac{\Delta L_{\max}}{B + \mathcal{B}_{\mathrm{noise}}} \qquad (2)
$$

**符号**：各量同 2.3 式 (7)—— $`\Delta L_{\max}`$ 为 $B \to \infty$ 的单步进展上限， $`\mathcal{B}_{\mathrm{noise}}`$ 为噪声尺度。式 (2) 关于 $B$ 严格单调减：纯按 token 效率，batch 越小越划算。那为什么不全程小 batch？两个下界把它顶住。(i) 稳定性：步长有与 batch 无关的上限 $\epsilon_{\max}$ （2.3 式 (7)）；batch 越小，达到同一 token 预算所需的步数越多，总进展被 $\epsilon_{\max}$ 封顶后，小 batch 的 token 效率优势不再兑现为总 loss 优势。(ii) 墙钟：单步的通信与 kernel 固定开销不随 batch 线性缩（4.x 的通信账与 5.x 的并行账），过小的 batch 让 GPU 喂不饱。ramp 的精确语义由此成立： $B(t)$ 从「硬件能喂饱的最小值」起步，沿噪声尺度漂移的方向追向 $`\mathcal{B}_{\mathrm{crit}}(t)`$ ，终值由并行效率锁定。三家的 ramp 形态各有写法：GPT-3 与 DeepSeek-V3 是 tokens 轴上的线性爬升；Llama-3 是两次倍增的阶梯（含序列长 4096 → 8192 的同步翻倍——batch 与序列长联动，长上下文的成本账归 7.5）；GLM-4.5 是 tokens 轴线性。没有公开证据表明形态差异（线性 vs 阶梯）对终点性能有可分辨的影响；共同点是幅度（3–5 倍区间）与位置（前 2%–5% tokens）高度一致。

## 4. µP：宽度缩放的结论表与推导 outline

§1 的表按家族逐行核对后，会看到一个共性问题：这些 lr 都是**在该规模上单独调出来或外推出来的**。GPT-3 八档各配一个 lr，OLMo 2 为 7B 扫过一轮 lr（2.3 §3 的交叉点实验）。µP（maximal update parametrization，[Yang et al., 2022](../references.md)，arXiv:2203.03466，[B]）要消掉的正是这项成本：让最优超参在宽度方向上不变，小模型上扫出的 lr 直接搬到大模型。

**问题现象**（论文 §3）：标准参数化（standard parametrization, SP——框架默认初始化配同一套 lr）下，固定 lr 训不同宽度的网络，最优 lr 随宽度系统性漂移，「更宽更差」的曲线交叉真实出现；同一组超参在 µP 下则随宽度单调改善。漂移的机制根源：SP 的初始化只保证前向激活尺度合理，不保证**更新量**的尺度随宽度不变。把这句话说细：SP 下 Adam 的逐坐标更新幅度与宽度无关（lr 即坐标量级），而隐藏层的输出是 $n$ 个坐标对相关输入的累积。宽度 $n$ 翻倍，同样的 lr 造成的层功能扰动近似翻倍；要维持同样的功能扰动就得把 lr 压小——这就是「最优 lr 随宽度漂移」在 SP 下的形态。µP 的 desideratum (iii) 把这个漂移显式消掉：它要求每层的功能扰动在宽度极限下既不消失也不发散，lr 的宽度指数由此被唯一确定。还有一个等价读法来自 kernel regime 的对照：若初始化与 lr 的缩放使某些层在宽度极限下几乎不动（更新量随宽度趋于零），极限网络就退化为随机特征核、不做特征学习。µP 是「让所有层都保持特征学习」的**唯一**参数化（论文引其系列前作的形式化结论），HP 稳定性是这个唯一性的副产品。

**推导 outline**（论文附录 J 的 Desiderata J.1 路线，逐步对应）：µP 由三条 desiderata 唯一确定——(i) 任意时刻每个（预）激活向量的坐标为 $\Theta(1)$ 量级；(ii) 网络输出为 $O(1)$（初始化处允许趋于零）；(iii) 所有参数在宽度极限下都被「尽可能多地」更新而不发散（maximal feature learning——排除 lazy/kernel regime）。对三类参数分别推出坐标尺度（ $n$ = 层宽）：

- **输出层权重** $W \in \mathbb{R}^{1 \times n}$ ：输入 $x$ 坐标 $\Theta(1)$ ，要 $|Wx| = O(1)$ 须 $W$ 坐标 $\Theta(1/n)$——初始化 $W_{\alpha\beta} \sim \mathcal{N}(0, \Theta(1/n^{2}))$（方差 $1/\mathrm{fan\\_in}^{2}$ ），且 lr 取 $\Theta(1/n)$ ，使 $\Delta W$ 的坐标同样保持 $\Theta(1/n)$ 。
- **隐藏层权重** $W \in \mathbb{R}^{n \times n}$ ：前向尺度由 CLT 给出 $`\mathrm{Var}((Wx)_i) = n \cdot \sigma^{2} \cdot \Theta(1)`$ ，坐标 $\Theta(1)$ 要求初始化方差 $\sigma^{2} = \Theta(1/n)$（即 $1/\mathrm{fan\\_in}$ ，与 SP 相同）；更新侧要求 $\Delta W$ 坐标 $\Theta(1/n)$ 才能使 $(W_0 + \Delta W) \thinspace x$ 保持 $\Theta(1)$ ——关键机制是 $\Delta W$ 与 $x$ **相关**（梯度是 $x$ 的函数，逐坐标符号对齐而非随机抵消），故 $`(\Delta W \thinspace x)_i`$ 按 $n$ 倍（ $\Delta W$ 坐标乘以坐标数）而非 $\sqrt{n}$ 倍累积。Adam 的逐坐标更新幅度就是 lr 本身（2.2 §2 的 sign 型结构），于是 Adam lr $= \Theta(1/n)$ ；SGD 下可以算出 $W$ 的梯度坐标为 $\Theta(1/n)$（经输出层的尺度回推）， $\Theta(1)$ 的 lr 已够。
- **输入层权重与 bias**：fan_in 是词表/输入维，相对宽度为常数，初始化方差 $1/\mathrm{fan\\_in}$ 不变；Adam lr 取 $\Theta(1)$ 即可，SGD lr 取 $\Theta(\mathrm{fan\\_out})$ 。

三条 desiderata 把每个自由度的宽度指数固定下来，结果就是论文 Table 3 的结论表（本书按原表核对转录；括号内为 SP 值，仅列不同之处）：

| 字段 | 输入权重与全部 bias | 输出权重 | 隐藏权重 |
| --- | --- | --- | --- |
| 初始化方差 | $1/\mathrm{fan\\_in}$ | $1/\mathrm{fan\\_in}^{2}$（SP： $1/\mathrm{fan\\_in}$ ） | $1/\mathrm{fan\\_in}$（与 SP 同） |
| SGD lr | $\mathrm{fan\\_out}$（SP：1） | $1/\mathrm{fan\\_in}$（SP：1） | 1 |
| Adam lr | 1 | $1/\mathrm{fan\\_in}$（SP：1） | $1/\mathrm{fan\\_in}$（SP：1） |

对 Transformer 实例化（ $\mathrm{fan\\_in} \approx d_{\mathrm{model}}$ ）：隐藏矩阵与输出头的 Adam lr 都按 $1/d_{\mathrm{model}}$ 缩放——「宽度翻倍、lr 减半」是 µP 对 LLM 最直接的一条推论；embedding 行（输入权重）lr 不随宽度变。论文另给一组等价写法（其 Lemma J.1 与 Table 8/9）：同一 µP 语义可以在「参数乘子 + 初始化 + lr」三者之间不同分配——SGD 下 $(A, B, C) \mapsto (A\theta, B/\theta, C/\theta^{2})$ 、Adam 下 $(A\theta, B/\theta, C/\theta)$ 保持函数逐点不变。工程实现里用「输出头乘子 $1/\mathrm{fan\\_in}$ 、lr 不变」的写法，与上表完全等价。

Transformer 还有一处 µP 特有的改动，在 attention 内部（论文 Definition 4.1）：logit 的归一化从 $1/\sqrt{d}$ 改成 $1/d$（ $q, k$ 各为 $d$ 维时）。理由是同一个相关性机制的两面：初始化处 $q$ 与 $k$ 不相关，内积按 CLT 以 $\sqrt{d}$ 涨， $1/\sqrt{d}$ 是对的；训练中两者变得相关， $q^{\top} k$ 由大数律按 $d$ 涨，要保 logit 为 $\Theta(1)$ 就该除 $d$ 。这条改动与 models 2.3 的 $\sqrt{d_k}$ 推导不矛盾——那里推的是前向在初始化附近的尺度，这里要的是训练全程的尺度不变性，两者回答不同时刻的问题。结论表的全部条目可按层特化（论文 Table 2 注记：lr、初始化方差、参数乘子都允许逐层取值），µP 因此是一张「逐层缩放律」，而非一个全局规则。

µP 的实证形态也值得记住：同一组超参下，µP 网络的训练 loss 随宽度**单调**改善（论文 Figure 1/3/7 的曲线互不交叉，「wider is always better」），SP 则出现更宽更差的交叉。这个性质本身就是工程资产：代理模型的结论按「更宽只会更好」的方向外推，不存在 SP 下「宽度过了某点反而退化」的陷阱。

**纪律注记**：上表是「宽度缩放」的精确结论；推导 outline 中「 $\Delta W$ 与 $x$ 相关 ⇒ 按 $n$ 倍累积」一步的严格处理，依赖论文附录 J 的矩阵-向量乘积分类（其 Table 14：随机/相关矩阵情形分列）。本书只给结论与机制读法，不替代该附录的完整论证。

## 5. µTransfer：小代理调参的成本账

µP 的回报形态是 µTransfer（论文 §7）：在缩窄的代理模型上扫超参，零样本迁移到目标模型。论文的 GPT-3 案例（§7.4）：代理是宽度 256 的约 40M 参数模型，目标是宽度 4096 的 6.7B（参数量差 168 倍）。代理上随机搜索的总调参成本，约为目标模型一次预训练成本的 7%。迁移后的 6.7B 超过按原配方重训的 6.7B 基线，在评测集上达到两倍大的 13B 的水平（论文 Table 7/11 口径）。成本账的结构是普适的：调参预算从此与目标模型规模**解耦**——无论目标多大，代理固定在一个可负担的宽度上扫。

迁移的适用边界，论文自己列得很清楚（§6 与 Table 1）：(i) 可迁移的是优化器相关、初始化、参数乘子三类（lr、momentum、schedule 形状等）；(ii) 以正则化为功能的超参（dropout、weight decay）不随宽度迁移——正则强度本来就是模型规模与数据规模的函数；(iii) 跨维度迁移有实证条件，宽度、depth、batch、序列长、训练时长各自存在最小门槛（论文给的参考量级：宽 256、深 4、batch 32、序列 128、步数 5000）；(iv) 两个已知失效点：初始化标准差不随深度迁移（深度方向要固定 init、只调其余），且深度迁移只对 Pre-LN 成立。LLM 预训练的主流形态（Pre-LN、宽度远比深度可调）恰好落在 µP 表现最好的区域。

把 µTransfer 与「每档规模各扫一遍」的朴素路线放在同一本账上。无迁移时， $s$ 个规模档、每档 $g$ 个超参点，扫描成本为 $s \cdot g \cdot C_{\mathrm{proxy}}(n)$ ，且代理成本 $C_{\mathrm{proxy}}$ 本身随档位宽度上涨。µTransfer 把它压成一档： $g \cdot C_{\mathrm{proxy}}(n_0)$ 外加目标模型的一次正式训练， $n_0$ 固定在最小可靠宽度。GPT-3 案例的 7% 即来自这个结构（40M 代理上完整随机搜索一次）。再对照一条不用 µP 的代理路线：MiniCPM 的风洞实验（0.009B–0.17B 族上拟合最优 batch 随 loss 的幂律，2.3 §5 已引）同样用小模型外推大模型的超参，但它外推的是**规律**（幂律的系数），µTransfer 外推的是**数值**（lr 本身）——前者需要假设规律的形式可拟合，后者由参数化保证逐点不变。两条路线不互斥：规律外推做先验，µTransfer 做精调，是 §1 表中「Qwen3 式 scaling-law 预测 + µP 式不变量」叠加的理论形态。

## 6. 对照路线：Moonlight 的 RMS 匹配

µP 不是唯一一条「大模型免调参」路线。Moonlight 的处方（2.5 §4 已全推导，此处只收对照）：给 Muon 补上 weight decay（ $\lambda = 0.1$ ），再把每个矩阵参数的更新 RMS 逐形状对齐到 0.2（乘以 $0.2 \cdot \sqrt{\max(A, B)}$ ，抵消引理 1 的形状依赖）。之后 Muon 直接复用 AdamW 搜好的 lr 与 wd，开箱即用（[Liu et al., 2025](../references.md)，arXiv:2502.16982，[A]，§2.2 与 §3.2）。GLM-4.5 的 §1 表行证明这条路线已进入旗舰生产配方（355B 全程 Muon + RMS 0.2）。

两条路线的哲学并排：µP 改**参数化**本身，让最优超参成为宽度的不变量，调参在小代理上完成后零样本迁移；RMS 匹配不动参数化，把**更新幅度**固定到一个与形状无关的绝对工作点（RMS = 0.2，对齐 AdamW 的经验区间 $0.2\text{–}0.4$ ），让「为 AdamW 调好的超参」直接可搬。前者的不变量写在初始化与 lr 缩放里，后者的不变量写在更新量纲上。值得补记 RMS 匹配路线的两个技术细节（均出自 Moonlight 原文，2.5 §4 有全推导）。其一，wd 不是可选装饰：放大规模后裸 Muon 的权重与层输出 RMS 持续增长、溢出 bf16 的高精度区间，800M / 100B tokens（约 5 倍 compute-optimal 数据量）的对照实验显示，加 wd 后过训练区段的 validation loss 低于裸 Muon 与 AdamW 两侧。其二，RMS 的目标值 0.2 来自附录 A 的五点扫描（Table 8 口径：2k 步、约 2B tokens 处的 validation loss，0.05 / 0.1 / 0.2 / 0.4 / 0.8 五档依次为 3.503 / 3.374 / 3.325 / 3.314 / 3.543——0.2 与 0.4 两档优于其余三档、彼此之间只差 0.011）。纪律注记（沿 2.5 §8 的裁定）：Moonlight 全文不含 µP / transfer 内容；两条路线的实证对照（不同规模上的迁移精度）目前是开放问题，本章只并陈、不裁决。

## 7. Tokens 预算、loss 目标与初始化细节

**tokens 预算**的第一性框架是 models 1.3 的 compute-optimal 分配律（Chinchilla 口径 $D^{\ast}/N \approx 20$ tokens/param，纯文字指引，不重复推导）。§1 表按这个标尺重读：DeepSeek-V3 14.8T / 671B ≈ 22 tok/param（按总参），贴着 Chinchilla 点；Llama-3 405B 15.6T ≈ 38.5 tok/param，约 2 倍过训；Llama-3 8B 同吃 15.6T ≈ 1950 tok/param，是深度过训练。herd 论文对这个偏离给的是推理侧的理由（§1 引言原文）：「we also train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget」——模型要服役无数次前向，训练期多花的 FLOPs 用推理期的参数效率买回。inference-aware 的过训练决策由此成为配方设计的一部分：tokens 预算不再只由训练算力决定，而由「训练成本 + 服役期推理成本」的联合账决定（本书分析；Chinchilla 框架本身不含推理项）。loss 目标一侧，各家普遍用 scaling-law 拟合外推目标 loss/能力，再反推 tokens（Qwen3 的整行「方法披露」、Llama-3 §3.2.1 的 IsoFLOPs 流程都是这个形态）。MiniCPM 给的证据方向更激进：用 WSD 的 checkpoint 复用结构把数据-模型二维扫描摊薄后，其拟合出的最优数据/模型比明显高于 Chinchilla 口径（[Hu et al., 2024](../references.md)，arXiv:2404.06395，[A]，§4-5 的方法与结论）——「小模型深过训」在 2B 级同样成立，与 herd 的推理侧理由互相印证。读表时的最后一个换算由此固定：先按 models 1.3 的分配律读出 Chinchilla 点，再按推理服役量决定过训倍数，tokens 预算 = 倍数 × $N$ × 20。

**初始化细节**（§1 表末行的展开）。 $\sigma = 0.02$ 是 GPT 系沿袭下来的惯例值，OLMo 2 用它做全部参数的截断正态初始化（[Team OLMo et al., 2025](../references.md)，[A]，§2.3）。值得登记的是 OLMo 2 同时记录了**从哪改过来**：旧版 OLMo-0424 用深度缩放初始化（scaled initialization，Zhang et al. 2019，EMNLP，经 OLMo 2 转引；该路线把残差分支的输出投影按深度压小，典型形式为方差乘 $1/(2L)$ ），OLMo 2 换到平坦的 0.02 后配合 §3 的其余稳定手段运行良好（§3.2）。对照点：DeepSeek-V3 全部可学参数 std 0.006（[DeepSeek-AI, 2024](../references.md)，[A]，§4.2），比 0.02 小约 3 倍，与其 7168 的宽度同向。固定 fan_in 缩放下 $\sigma \propto 1/\sqrt{d}$ 的直觉账给出 $\sqrt{4096/7168} \times 0.02 \approx 0.015$ ，量级方向一致，但 0.006 更保守——此处为本书的对照读法，两家原文均未给推导。µP 视角下，这些「惯例」都应该读成参数化选择而非普适常数：换参数化（Table 3 的输出行 $1/\mathrm{fan\\_in}^{2}$ ）就换 init，惯例值只在 SP 语境内成立。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| 配方对照表 | 七家全字段并列（§1）；未披露入表：Llama-3 的 $\beta$ / $\epsilon$ /clip、Qwen3 整行数值、GLM-4.5 的 warmup 步数与 clip | 逐字段节号见 §1 注 |
| 规模规律 | lr 随规模降（GPT-3 Table 2.1 八档单调）、batch 随规模涨；OLMo 2 13B > 7B 为披露的反向数据点 | §2；2.3 §6 |
| batch ramp | $S_{\mathrm{ramp}} = D_r \ln(B_1/B_0)/(B_1 - B_0)$ 式 (1)；DSv3 ≈ 15000 步（2.0×）、GLM-4.5 ≈ 14400 步（1.84×）；ramp 段占总 tokens 2%–3% | §3；2.3 式 (7) |
| µP 结论表 | 隐藏/输出 Adam lr $\propto 1/\mathrm{fan\\_in}$ ；输出 init 方差 $1/\mathrm{fan\\_in}^{2}$ ；输入/bias 不随宽度变 | §4 表 = TP-V Table 3 [B] |
| µP 推导 | 三 desiderata ⇒ 坐标尺度唯一钉死； $\Delta W$ 与 $x$ 相关按 $n$ 倍累积是关键步（严格处理归 TP-V 附录 J） | §4 |
| µTransfer | 40M 代理 → 6.7B 目标（168×）、调参成本 7%、达 13B 水平；正则类不迁移、深度迁移仅 Pre-LN | TP-V §7.4 / §6 [B] |
| Moonlight 对照 | wd 0.1 + update RMS 0.2 ⇒ 复用 AdamW 超参；参数化不变量（µP）vs 更新量纲不变量（RMS 匹配）并陈 | 2.5 §4；GLM-4.5 §2.4 为生产实例 |
| tokens 预算 | Chinchilla $D/N \approx 20$ 标尺；405B ≈ 38.5、8B ≈ 1950 tok/param；inference-aware 过训（herd §1 引言原文） | models 1.3（纯文字指引） |
| 初始化 | 0.02 惯例（OLMo 2，从 Zhang et al. 2019 深度缩放改来）vs 0.006（DSv3）；init 是参数化选择 | §7 |

## 练习

1.（推导）补全 §4 输出层一行：设 $W \in \mathbb{R}^{1 \times n}$ 、 $x$ 坐标 $\Theta(1)$ 且各坐标近似独立，(a) 由 $|Wx| = O(1)$ 推出初始化坐标尺度 $\Theta(1/n)$（注意与「随机向量内积按 $\sqrt{n}$ 累积」的直觉对照——输出层在 µP 下为什么取 $1/n$ 而非 $1/\sqrt{n}$ ？提示：desideratum (iii) 要求训练后 $W$ 相对初始化有 $\Theta(1/n)$ 量级的新增相关成分）；(b) 说明 Adam lr 取 $\Theta(1/n)$ 时 $\Delta W$ 坐标为何保持 $\Theta(1/n)$ 。

2.（计算）ramp 账复核：用 §3 的线性假设与式 (1)，(a) 复核 DeepSeek-V3（469B tokens、12.6M → 62.9M）与 GLM-4.5（500B、16M → 64M）的 $S_{\mathrm{ramp}}$ 与相对恒定末值的步数放大倍数；(b) 若 DSv3 的 ramp 改为「前 469B 内 batch 翻倍三次」的阶梯式（Llama-3 式），步数与线性 ramp 差多少（按每段等 tokens 均分近似）？(c) 讨论：ramp 段的 lr 是否应随 batch 同步调整？用 2.3 §4-5 的框架给出两种相反意见各自的依据。

3.（计算）tokens 口径换算：某报告给出「global batch 15360 序列、seqlen 4096」，训练栈为 $b = 4$ 、 $d = 256$ 。 (a) 求 $m$ 与 $B_{\mathrm{tok}}$ ；(b) 该配方要在 $d = 512$ 、 $b = 2$ 的集群上复现， $m$ 应取多少？激活侧哪一笔账会变化（回链 1.4/1.5）？

4.（分析）µP 的 Table 3 把输出层的 Adam lr 与隐藏层同取 $1/\mathrm{fan\\_in}$ ，但 embedding（输入权重）取常数 lr。GLM-4.5 与 Moonlight 的工程边界（embedding / lm_head 走 AdamW 系、隐藏矩阵走 Muon 系）与这张表的分区方式有何对应？若要把 µP 与 RMS 匹配叠加使用（隐藏层 µP 化 + Muon 更新），指出 §4 结论表中哪一行的语义会先被破坏，并说明理由。

## 参考

- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）——Table 2.1 的 lr/batch 缩放；App. B 的 optimizer/warmup/decay/ramp/wd/clip
- [Touvron et al., 2023](../references.md)：Llama 2，arXiv:2307.09288，[A]（key: llama2）——§2.2 超参段与 Table 1（lr 分档、4M tokens batch 表注）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——§3.4.1 配方（lr/warmup/decay/batch 翻倍）、§3.2.1 的 wd = 0.1×lr、§1 引言的过训练口径与 15.6T
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——§4.2 全字段（含 init std 0.006 与 batch ramp）、§4.1 的 14.8T
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——Table 4 全字段、§3.4.1（ $\epsilon$ ）、§3.4.2（embedding 免 wd）、§2.3/§3.2（init 0.02 与其改动来源）、§2.3/§4（mid-training 线性退火与模型平均）
- [GLM Team, 2025](../references.md)：GLM-4.5，arXiv:2508.06471，[A]（key: glm-4.5）——§2.4 全字段（Muon + RMS 0.2、cosine 2.5e-4 → 2.5e-5、ramp 16M → 64M、wd 0.1）
- [Yang et al., 2022](../references.md)：Tensor Programs V，arXiv:2203.03466，[B]（key: tensor-programs-v）——Table 3（µP 结论表）、Desiderata J.1 与附录 J 推导、Lemma J.1 等价形式、§7.4 GPT-3 µTransfer（40M→6.7B、7% 成本）、§6 迁移边界
- [Liu et al., 2025](../references.md)：Muon is Scalable for LLM Training（Moonlight），arXiv:2502.16982，[A]（key: moonlight）——wd + RMS 0.2 对齐与超参复用主张（§2.2/§3.2）
- [Hu et al., 2024](../references.md)：MiniCPM，arXiv:2404.06395，[A]（key: minicpm）——WSD 配方形态（7.1 §5 的数据课程出处；本章 §1 GLM-4.5 行「不用 WSD」的对照出处）
- [Qwen Team, 2025](../references.md)：Qwen3 Technical Report，arXiv:2505.09388，[A]（key: qwen3）——36T tokens、119 语言、scaling laws 预测超参而不披露数值（§1 表行口径）
