# 2.3 LR Schedule、Warmup 与 Batch 联动（LR Schedules, Warmup, and Batch Coupling）

> 本章回答 lr 的三个结构问题：开头为什么必须从小往上爬（warmup 的两个独立论证：models 2.7 的 Post-LN 梯度剖面 + RAdam 对 Adam 早期 $\hat{v}$ 方差的解析）、主体到收尾走什么曲线（constant / cosine / linear / WSD 梯形，含 MiniCPM 的 checkpoint 复用机制）、 $\eta$ 与 batch $B$ 怎么联动（linear scaling rule 的假设链、sqrt 规则、McCandlish 的 critical batch size 推导与测量）——全部落到 GPT-3 / Llama-3 / DeepSeek-V3 / OLMo 2 的真实日程上 | 前置：2.1（梯度噪声 $\Sigma/B$）、2.2（AdamW 更新式 (2)(9)、bias correction）；models 2.7 §5（Pre/Post-LN 梯度分析，纯文字指引） | 锚点：OLMo 2 [A]、Llama-3 herd [A]、MiniCPM [A]、DeepSeek-V3 [A]；McCandlish et al. [B]、Goyal et al. [B]、RAdam [A]

## 1. 问题设定： $\eta(t)$ 的三个自由度

先看四个真实日程，问题的形状就清楚了。DeepSeek-V3：linear warmup 2000 步到 $2.2 \times 10^{-4}$，恒定到 10T tokens，cosine 衰减 4.3T tokens 到 $2.2 \times 10^{-5}$，最后 500B tokens 走两段常数（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§4.2）。Llama-3 405B：linear warmup 8000 步，cosine 从 $8 \times 10^{-5}$ 衰减到 $8 \times 10^{-7}$，全程 1,200,000 步（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.4.1）。OLMo 2 7B：linear warmup 2000 步到 $3 \times 10^{-4}$，cosine 计划 5T tokens、在 4T 处截断转入 mid-training（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§3.5）。GPT-3：linear warmup 覆盖前 375M tokens，cosine 在 260B tokens 内降到峰值的 10%，其后恒定（[Brown et al., 2020](../references.md)，arXiv:2005.14165，[B]，App. B）。

四家没有一家用恒定学习率（learning rate, lr；本章正文一律记 lr），而且结构高度一致：warmup 段（千步级，占总步数 0.1%–1% 量级）+ 主体段（恒定或高平台）+ 收尾段（衰减到峰值的 1%–10% 量级）。把日程写成 $\eta_t = \eta_{\max} \cdot s(t)$，问题就拆成三个自由度： $s(t)$ 开头怎么爬（§2）、主体到收尾走什么曲线（§3）、 $\eta_{\max}$ 与 batch $B$ 怎么联动（§4、§5）。

## 2. Warmup 的两个独立论证

warmup 的形式统一为：前 $W$ 步把 $\eta_t$ 从 0（或一个小常数）线性升到 $\eta_{\max}$。它有两个互相独立的必要性论证，一个来自架构，一个来自优化器统计；两者缺一时，实践中都见过发散。

**论证 A：初始化处的梯度剖面（架构侧）**。models 2.7 §5 的 Pre/Post-LN 梯度分析给出两个事实：Post-LN 结构在初始化时，最后一层 FFN 的梯度上界为 $O(d\sqrt{\ln d})$、与深度 $L$ 无关；参数梯度靠近输出端大、向输入端逐层衰减。也就是说，逐层安全的 lr 本该逐层不同、输出端最小。warmup 用一个时变的全局 $\eta(t)$ 近似这个逐层条件：把最初几百步的有效更新压到各层统计能适应的尺度内；几百步后梯度范数的层间比例趋于稳定，单一 $\eta$ 重新可行。Pre-LN 的 $1/\sqrt{L}$ 因子自带保守的初始化梯度尺度，把这个危机大幅缓解——但现代 LLM（全是 Pre-LN 或其变体）仍保留千步级 warmup，因为危机的另一半在优化器侧。

**论证 B： $\hat{v}$ 的早期方差（优化器侧）**。2.2 §3 的 bias correction 修正的是矩估计的**均值**；早期 $\hat{v}$ 的**方差**是另一个问题。 $t$ 很小时， $`\hat{v}_t`$ 只由个位数梯度样本构成，逐坐标的 $1/\sqrt{\hat{v}}$ 是一个高方差随机变量，步长被随机放大。RAdam（[Liu et al., 2020](../references.md)，arXiv:1908.03265，[A]）把这个方差解析出来：把 $v_t$ 的分布近似为自由度 $\rho$ 的 scaled-inv- $\chi^{2}$，自适应学习率 $`\psi_t = 1/\sqrt{\hat{v}_t}`$ 的方差近似为

$$
\mathrm{Var}[\psi_t] \approx \frac{\rho_t}{2(\rho_t - 2)(\rho_t - 4)\thinspace \sigma^{2}}, \qquad \rho_t = \rho_{\infty} - \frac{2t \thinspace \beta_2^{t}}{1 - \beta_2^{t}}, \qquad \rho_{\infty} = \frac{2}{1 - \beta_2} - 1 \qquad (1)
$$

**符号**： $\rho_t$：EMA 等价的简单移动平均窗长（自由度），由「EMA 与 SMA 质心对齐」解出； $\sigma^{2}$：梯度平方的尺度参数。注意 $\rho_1 = 1$ 对任意 $\beta_2$ 精确成立（一行代数可验），且早期 $\rho_t \approx t$。方差随步数收敛的速度由式 (1) 完全决定： $\rho_t \le 4$ 时方差发散（ $\beta_2 = 0.999$ 与 $0.95$ 都在第 4、5 步之间跨过 4）， $\rho_t = 5$ 处的方差比 $\rho_t = 500$ 处大两个数量级以上（原文数值口径）。RAdam 的处置： $\rho_t \le 4$ 时关闭自适应项、退化为 SGD-momentum； $\rho_t \gt 4$ 时给更新乘修正因子

$$
r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\thinspace \rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} - 2)\thinspace \rho_t}} \qquad (2)
$$

 $r_t$ 从 0 单调升到 1（ $\beta_2 = 0.999$ 时 $r_5 \approx 0.017$、 $r_{10} \approx 0.049$ ）。把 $r_t$ 换成 $\min(t, W)/W$，就是 linear warmup 的形状——原文的结论值得直引其意：warmup 是一种方差缩减技术，RAdam 把它从启发式变成有解析形式的方差截断。式 (1) 还附带一个退化判据： $\beta_2 \le 0.6$ 时 $`\rho_{\infty} \le 4`$，方差永不收敛，RAdam 退化为 SGD-momentum。实践中 LLM 预训练照旧用显式 warmup 而不是 RAdam：RAdam 没有改变 $\hat{v}$ 的测量事实，只是把响应自动化；warmup 长度仍按千步级设置。

两个论证给出同一条处方：初始阶段的逐坐标有效步长必须压小。warmup 结束步数 $W$ 的量级经验：OLMo 2 取 2000 步、Llama-3 405B 取 8000 步、DeepSeek-V3 取 2000 步、GPT-3 按 token 计取 375M tokens（各家出处见 §1）——都在总日程的 1% 以内。

## 3. Schedule 族：constant、cosine、linear、WSD

warmup 之后是主体与收尾。四个常用族如下（统一写成 $\eta_t = \eta_{\max} \cdot s(t)$， $W$ 为 warmup 结束步、 $T$ 为衰减开始步、 $S$ 为总步数）：

$$
\text{constant: } s(t) = 1; \qquad \text{linear decay: } \eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{S - t}{S - W}; \qquad \text{cosine: } \eta_t = \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos\frac{\pi\thinspace (t - W)}{S - W}\right) \qquad (3)
$$

**符号**： $\eta_{\min}$：收尾下限。cosine 在 $t = W$ 处取 $\eta_{\max}$、 $t = S$ 处取 $\eta_{\min}$，两端导数为 0，起步和收尾都平缓；linear 全程匀速下降；constant 没有收尾。

**WSD（warmup-stable-decay，梯形）**把「主体」和「收尾」显式拆开（[Hu et al., 2024](../references.md)，arXiv:2404.06395，[A]，§4）：

$$
\eta_s = \begin{cases} \eta \cdot s / W, & s \lt W \quad \text{（warmup）} \\ \eta, & W \le s \le T \quad \text{（stable）} \\ \eta \cdot f(s - T), & T \lt s \le S \quad \text{（decay）} \end{cases} \qquad (4)
$$

**符号**： $f$：从 1 递减的退火函数（MiniCPM 生产训练用指数退火，decay 段 5000 步、约 20B tokens）。WSD 的机制优势不在曲线形状，而在**工程结构**：stable 段可以任意延长，训练到任意 checkpoint 都能分叉出一个 decay 段收尾。MiniCPM 的风洞实验（0.036B 模型）显示三件事：decay 段一开始 loss 骤降、追平同 token 数的 cosine；stable 段 checkpoint 复用续训更久、再 anneal，仍追平对应更长的 cosine；decay 段占总 token 约 10% 足够、2.5% 不足。对照 cosine：cosine 的 $T = S$ 在定死 token 预算时最优（MiniCPM 复现了 Hoffmann et al. 的观察： $T \gt S$ 或 $T \lt S$ 都更差），但它把预算写死在曲线里，改预算就要从头重训。WSD 把「预算决策」推迟到 decay 段开始前。2024 年之后的公开配方里，梯形结构反复出现：MiniCPM 的 WSD（式 (4)）之外，DeepSeek-V3 的「恒定 10T + cosine 4.3T + 常数尾段」（§1）可以读作 WSD 梯形加一条长尾。

四种曲线的形状对照（归一化 $\eta_{\max} = 1$， $W$ 与 decay 段画得夸张以便辨认）：

```
η/η_max
1.0 ┤    WSD：                  cosine(T=S)：
    │      ╭━━━━━━━━━━━╮            ╭╮
    │     ╱             ╲          ╱  ╲
    │    ╱ warmup        ╲        ╱    ╲
    │   ╱   稳定段         ╲     ╱      ╲
    │   ╱  （可任意延长）    ╲   ╱        ╲
0.1 ┤ ╱                     ╲_╱          ╲___ η_min
    │╱                       decay       ╲
0   └────────────────────────────► t      └──(约10%)
    0   W          T         S
```

收尾下限 $\eta_{\min}$ 的口径各家不同：GPT-3 与 Llama-2 的 cosine 都收到峰值的 10%（GPT-3 其后恒定在 10%；Llama-2 原文为「decay final learning rate down to 10% of the peak」，[Touvron et al., 2023](../references.md)，[A]，§2）；OLMo 2 的 mid-training 直接把 lr 线性收到 0。收到 0 还是收到一成，目前无公开消融支持统一结论——7.2 的配方表按原样并列两派口径。

**lr 与训练时长的耦合**有一个记录在案的实证。OLMo 2 把 7B 的峰值 lr 从 $3 \times 10^{-4}$ 扫到 $30 \times 10^{-4}$：更高的 lr 早期 loss 一律更低，但最终被更低的 lr 反超—— $3 \times 10^{-4}$ 对 $6 \times 10^{-4}$ 的交叉点在 200B tokens 之后； $30 \times 10^{-4}$ 在 warmup 期间就出现未能完全恢复的 loss spike，被放弃（[A]，§3.5）。两个教训：评 lr 必须跑到收尾，短 horizon 的 lr 扫描会得出系统性偏大的结论；warmup 期是 spike 最先暴露的窗口，与 §2 的两个论证一致。OLMo 2 的另一发现：mid-training 阶段把 lr 线性降到 0 时，50B 与 100B tokens 两种退火长度的训练 loss 等效——收尾段的总量比形状更要紧。

## 4. lr × Batch：linear scaling 与 sqrt 规则

batch 放大 $k$ 倍，lr 该放大多少？Goyal et al. 的 linear scaling rule 给出答案和它的假设链（[Goyal et al., 2017](../references.md)，arXiv:1706.02677，[B]，§2.1）。对照两条轨迹：一条用小 batch $n$、lr $\eta$ 走 $k$ 步，另一条用大 batch $kn$、lr $\hat{\eta}$ 走一步：

$$
w_{t+k} = w_t - \eta \thinspace \frac{1}{n} \sum_{j \lt k} \sum_{x \in \mathcal{B}_j} \nabla l(x, w_{t+j}), \qquad \hat{w}_{t+1} = w_t - \hat{\eta} \cdot \frac{1}{kn} \sum_{j \lt k} \sum_{x \in \mathcal{B}_j} \nabla l(x, w_t) \qquad (5)
$$

**符号**： $`\mathcal{B}_j`$：第 $j$ 个小 batch； $l(x, w)$：样本 $x$ 的损失。关键在一个近似：若 $k$ 步内参数漂移足够小，则 $\nabla l(x, w_{t+j}) \approx \nabla l(x, w_t)$，左式的双重求和就等于大 batch 梯度，两式逐点相等的充要条件是 $\hat{\eta} = k\eta$——**batch 乘 $k$、lr 乘 $k$**。规则的全部效力来自这条假设链。失效边界论文自己也给了两条。第一，训练初期参数变化快，近似最先破——这正是 warmup 要补的缺口（Goyal 的 gradual warmup：从较小的常数 lr 分几个 epoch 爬到 $\hat{\eta}$）。第二， $k$ 不能无限大：ImageNet 上准确率稳定保持到 batch 约 8k，再往上迅速退化；拐点的精确位置由 §5 的噪声尺度给出。理论侧还有一个旁证：Goyal 引用 Bottou et al. 的综述指出两点——linear scaling 下求解器按「已见样本数」走同一条训练曲线；且 lr 不应超过一个与 batch 无关的上限。这为 warmup 和线性规则的饱和端各给了一个理论注脚。

Adam 类优化器上的常见做法是 sqrt 规则（ $\eta \propto \sqrt{B}$ ）。机制论证如下（本书分析，无单一出处，标注为经验规则）。Adam 的逐坐标更新是 sign 型有界量，步长被 $\eta$ 近似封顶——Adam 原文给的稀疏情形上界为 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$（0.9/0.999 时 3.16 $\eta$ ）；一般构造下的上确界数值估计约 7.3 $\eta$，仍与 $\eta$ 同量级（2.2 §3）。梯度估计的相对噪声按标准差计随 $1/\sqrt{B}$ 缩放（ §5 式 (10) 后的相对误差式开方）。要保持「噪声对更新方向的扰动」相对恒定， $\eta$ 就随 $\sqrt{B}$ 增长。sqrt 规则介于 linear（SGD、小 batch 区）与 constant（饱和区）之间；它的适用边界同样由噪声尺度控制， §5 的公式把三者的关系统一起来。

## 5. Critical batch size：噪声尺度的推导与测量

McCandlish et al.（[McCandlish et al., 2018](../references.md)，arXiv:1812.06162，[B]）把「batch 多大算够」变成一个可测量、可推导的量。设真梯度 $g$、真 Hessian $H$、逐样本梯度协方差 $\Sigma$。batch $B$ 的梯度估计 $G_{\mathrm{est}}$ 满足 $\mathbb{E}[G_{\mathrm{est}}] = g$、 $\mathrm{cov}(G_{\mathrm{est}}) = \Sigma / B$。沿估计梯度走一步 $\theta - \epsilon G_{\mathrm{est}}$，对真损失做二阶展开并取期望：

$$
\mathbb{E}\left[L(\theta - \epsilon \thinspace G_{\mathrm{est}})\right] = L(\theta) - \epsilon \thinspace \lVert g \rVert^{2} + \frac{\epsilon^{2}}{2}\left(g^{\top} H g + \frac{\mathrm{tr}(H\Sigma)}{B}\right) \qquad (6)
$$

**符号**：一次项只吃期望；二次项把噪声以 $\mathrm{tr}(H\Sigma)/B$ 的形式放进来。对 $\epsilon$ 求极小（一阶导置零）：

$$
\epsilon_{\mathrm{opt}}(B) = \frac{\epsilon_{\max}}{1 + \mathcal{B}_{\mathrm{noise}}/B}, \qquad \Delta L_{\mathrm{opt}}(B) = \frac{\Delta L_{\max}}{1 + \mathcal{B}_{\mathrm{noise}}/B}, \qquad \epsilon_{\max} = \frac{\lVert g \rVert^{2}}{g^{\top}Hg} \qquad (7)
$$

$$
\mathcal{B}_{\mathrm{noise}} = \frac{\mathrm{tr}(H\Sigma)}{g^{\top} H g}, \qquad \mathcal{B}_{\mathrm{simple}} = \frac{\mathrm{tr}(\Sigma)}{\lVert g \rVert^{2}} \qquad (8)
$$

**符号**： $`\mathcal{B}_{\mathrm{noise}}`$：噪声尺度（noise scale），论文式 (2.8) 的定义，带 Hessian 预条件； $`\mathcal{B}_{\mathrm{simple}}`$：论文式 (2.9) 的 simple noise scale——良条件化简（ $H \propto I$ ）后的形式，论文实测两者只差一个小的常数倍，工程上测的是 $`\mathcal{B}_{\mathrm{simple}}`$； $`\epsilon_{\mathrm{opt}}(B)`$：期望意义下的最优步长（即此局部二次模型下的 lr）； $`\Delta L_{\mathrm{opt}}(B)`$：对应的单步期望损失下降， $`\Delta L_{\max}`$ 为其 $B \to \infty$ 极限。

式 (7) 同时回答三个问题。小 batch regime（ $`B \ll \mathcal{B}_{\mathrm{noise}}`$ ）： $`\epsilon_{\mathrm{opt}} \approx \epsilon_{\max} \cdot B / \mathcal{B}_{\mathrm{noise}} \propto B`$、 $`\Delta L_{\mathrm{opt}} \propto B`$，batch 翻倍单步进展翻倍——linear scaling rule 就是这个 regime 的近似。大 batch regime（ $`B \gg \mathcal{B}_{\mathrm{noise}}`$ ）： $`\epsilon_{\mathrm{opt}} \to \epsilon_{\max}`$、 $`\Delta L_{\mathrm{opt}} \to \Delta L_{\max}`$，饱和了，再加 batch 只浪费算力——这是 §4 失效边界 (ii) 的精确位置。拐点本身：把全程的「步数 $S$ × 样本数 $E$」权衡对式 (7) 逐段积分，论文得到

$$
\frac{S}{S_{\min}} - 1 = \left(\frac{E}{E_{\min}} - 1\right)^{-1}, \qquad \mathcal{B}_{\mathrm{crit}} = \frac{E_{\min}}{S_{\min}} \qquad (9)
$$

并预言 $`\mathcal{B}_{\mathrm{crit}} \approx \mathcal{B}_{\mathrm{noise}}`$（取训练全程的适当平均； $S_{\min}$、 $E_{\min}$ 为达到给定性能的最少步数与最少样本数）。多任务实测（图像、LM、RL）中，simple noise scale 确实预测了效率折半的 batch 量级（原文 Figure 4 与 Table 1）。

这套推导的适用边界，由原文 §2.4 自列的六条 caveat 圈定，照录要点。(1) 短视界偏差：式 (7) 是单步局部最优，贪心的局部最优不保证全局最优，噪声本身可能有益（此时真正的临界 batch 更小）。(2) 病态条件：大曲率方向上的振荡使式 (7) 不再精确反映单步进展，但式 (9) 的权衡仍预测相对速度。(3) $`\mathcal{B}_{\mathrm{simple}}`$ 的精度以 $H$ 的病态程度为界，不同梯度分量的噪声尺度可以差很多。(4) 假设 lr 取到式 (7) 的最优：实践中 lr 调不到恰最优，且噪声尺度经「温度」依赖 lr（App. C）。(5) 二次近似：三阶项不可忽略时模型失真。(6) 只谈训练损失：泛化 gap（大 batch 训练 loss 好、测试 loss 差）不在框架内，原文引用后续工作称调好超参时未见该 gap。对 LLM 预训练，最要紧的是 (1)(4)：batch 决策被 token 预算与算力预算联合约束，单步局部账必须读成全程账。

**测量**只需要两笔梯度范数账。由 $\mathbb{E}\lVert G_B \rVert^{2} = \lVert g \rVert^{2} + \mathrm{tr}(\Sigma)/B$，在两个 batch 大小 $B_s \lt B_\ell$ 下各测一次梯度范数平方的期望，解二元线性方程组：

$$
\mathrm{tr}(\Sigma) = \frac{B_s \thinspace B_\ell}{B_\ell - B_s}\left(\mathbb{E}\lVert G_{B_s} \rVert^{2} - \mathbb{E}\lVert G_{B_\ell} \rVert^{2}\right), \qquad \lVert g \rVert^{2} = \frac{B_\ell \thinspace \mathbb{E}\lVert G_{B_\ell} \rVert^{2} - B_s \thinspace \mathbb{E}\lVert G_{B_s} \rVert^{2}}{B_\ell - B_s} \qquad (10)
$$

两式相除即 $`\mathcal{B}_{\mathrm{simple}}`$。论文 App. A.1 的实现方案在数据并行训练里几乎零开销：all-reduce 前的各 rank 小 batch 局部梯度与 all-reduce 后的大 batch 全量梯度，各提供一次范数测量，零额外通信。式 (10) 的副产品是梯度估计的相对误差： $`\mathbb{E}\lVert G_{\mathrm{est}} - g \rVert^{2} / \lVert g \rVert^{2} = \mathcal{B}_{\mathrm{simple}}/B`$——噪声尺度就是「估计与真值在 $L^{2}$ 意义下可比」的那个 batch。

**随训练漂移**。噪声尺度随 loss 下降而增长：论文记录 critical batch size 在一次训练全程可涨一个数量级以上，且同 loss 水平下与模型尺寸基本无关；「温度」 $T = \epsilon/B$ 一定时，平衡态噪声尺度正比于 $B/\epsilon$（论文 App. C）。温度口径还给出 ramp 与 decay 的统一读法：decay 段降 $\epsilon$、ramp 段升 $B$，两者都在降温度，所以 batch ramp 与 lr decay 常出现在同一个训练阶段（DSv3 与 GPT-3 的日程都是这个结构）——它们是同一旋钮的两个刻度。独立证据：MiniCPM 在 0.009B–0.17B 模型族上拟合出 C4 上的最优 batch 随 loss 的幂律 $bs = 1.21 \times 10^{9} / L^{6.24}$（[Hu et al., 2024](../references.md)，[A]，§3.2）——loss 越低、最优 batch 越大，方向与噪声尺度漂移一致。这给生产中常见的 batch ramp-up 一个理论背书：batch 随训练进程追赶漂移中的 $`\mathcal{B}_{\mathrm{crit}}`$。三个真实实例。GPT-3 把 batch 从 32k tokens 线性升到全值（前 4–12B tokens，随模型规模而异）[B]。DeepSeek-V3 把 batch 从 3072 升到 15360 条序列（前 469B tokens；seqlen 4K，折算约 12.6M → 62.9M tokens/步）[A]。Llama-3 405B 从 4M tokens（seq 4096）翻倍到 8M（同时 seq 升 8192）、再翻倍到 16M，原文动机句为「早期小 batch 保稳定、随后增大提效率」[A]——第一次翻倍的节点论文文本作「252M tokens」，按 4M tokens/步仅约 63 步，量级可疑（疑为 252B 之误），此处存疑标注；第二个节点 2.87T tokens 按原文。ramp 的位置也有共同规律：DSv3 的 ramp 占前 469B / 14.8T ≈ 3% tokens、GPT-3 占前 4–12B / 260B+ ≈ 2%–5%——追赶噪声尺度的窗口集中在训练前段，因为 $`\mathcal{B}_{\mathrm{crit}}`$ 的早期增长最快。

## 6. 配方规律与 7.2 预告

把 §1 的日程与 GPT-3 的缩放表放在一起，两条规律直接可读。GPT-3 Table 2.1（[B]）：从 125M 到 175B，峰值 lr 从 $6.0 \times 10^{-4}$ 单调降到 $0.6 \times 10^{-4}$，batch 从 0.5M tokens 升到 3.2M tokens——「lr 随规模降、batch 随规模涨」的最早系统披露。2024–2025 的数值落点同向：OLMo 2 7B 取 $3 \times 10^{-4}$ / 4M tokens，DeepSeek-V3（671B 总参、37B 激活）取 $2.2 \times 10^{-4}$ / 62.9M tokens，Llama-3 405B 取 $8 \times 10^{-5}$ / 16M tokens。机制上， §5 给了这两条规律一个统一读法：噪声尺度在同 loss 水平下与模型尺寸近似无关、随 loss 下降而涨；更大模型的训练推进到更低的 loss，working point 因此移向大 batch 区，而 $\epsilon_{\max}$ 受稳定性约束随规模收紧。逐字段的配方对照表（warmup 步数、decay 形状、batch ramp、 $\beta_2$ 、wd、clip、µP 缩放）连同 Qwen3 等更新配方归 7.2 展开。本章的三问——warmup 的必要性、schedule 的形状、 $\eta$ 与 $B$ 的联动——是读那张表的理论框架。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| warmup 论证 A | models 2.7 §5（纯文字指引） | Post-LN 初始化梯度近输出端大、与 $L$ 无关 ⇒ 初始大步长必炸；warmup = 时变全局 $\eta$ 近似逐层安全 lr |
| warmup 论证 B | 式 (1)(2) | $\hat{v}$ 早期方差解析： $\rho_t \le 4$ 方差发散、 $r_t$ 截断 ≈ warmup 的形状； $W$ 占日程 1% 以内 |
| schedule 族 | 式 (3)(4) | cosine 两端导数为零、预算写死；WSD 梯形 stable 段可任意延长、任意 checkpoint 分叉 decay（约 10% tokens 足够）；DSv3 = 梯形 + 长尾 |
| lr × horizon | OLMo 2 §3.5 [A] | 高 lr 早期好、被低 lr 反超（交叉点 >200B tokens）；lr 评估必须跑到收尾 |
| linear scaling | 式 (5) | $\eta \propto B$ ⟺ 小步近似 $\nabla l(x, w_{t+j}) \approx \nabla l(x, w_t)$；失效：warmup 期与 $`B \gtrsim \mathcal{B}_{\mathrm{noise}}`$ |
| sqrt 规则 | §4（经验规则） | sign 型更新 + 噪声按 $1/\sqrt{B}$ 缩放的启发式，边界由噪声尺度控制 |
| critical batch | 式 (6)–(10) | $`\epsilon_{\mathrm{opt}}(B) = \epsilon_{\max}/(1 + \mathcal{B}_{\mathrm{noise}}/B)`$； $`\mathcal{B}_{\mathrm{simple}} = \mathrm{tr}(\Sigma)/\lVert g \rVert^{2}`$；两 batch 梯度范数测量法；随训练漂移（MiniCPM： $bs \propto L^{-6.24}$） |
| 配方规律 | GPT-3 Table 2.1 [B]；DSv3 / Llama-3 / OLMo 2 [A] | lr 随规模降、batch 随规模涨；batch ramp 追赶漂移的 $`\mathcal{B}_{\mathrm{crit}}`$；全字段表归 7.2 |

## 练习

1.（推导）补全式 (7)：对式 (6) 关于 $\epsilon$ 求导置零，解出 $`\epsilon_{\mathrm{opt}}(B)`$ 并回代得 $`\Delta L_{\mathrm{opt}}(B)`$；指出推导用到了哪些近似（二次展开、固定 $H$），以及论文 §2.4 自列的哪一条 caveat 对 LLM 训练最要紧。
2.（推导）linear scaling 的误差项：把 $\nabla l(x, w_{t+j})$ 在 $w_t$ 处 Taylor 展开到一阶，写出式 (5) 两轨迹之差的首项，说明 (a) 该误差与 $\eta$、 $k$ 的幂次关系；(b) 为什么 warmup 期（参数快速移动）该近似最先失效。
3.（计算）RAdam 的窗长账：证明 $\rho_1 = 1$ 对任意 $\beta_2$ 精确成立；分别对 $\beta_2 = 0.999$ 与 $0.95$ 计算 $\rho_t$ 与 $r_t$（ $t = 1, \ldots, 10$ ），验证两者都在第 4、5 步之间跨过 $\rho_t = 4$，并对比两组 $r_{10}$（ $0.049$ vs $0.348$ ）说明小 $\beta_2$ 下修正因子上来的速度差多少——这就是 2.2 §7「 $\beta_2$ = 0.95 缩短二阶矩记忆窗」在 warmup 期的对偶面。
4.（计算）GPT-3 的 lr-batch 缩放：125M（0.5M tokens、 $6.0 \times 10^{-4}$ ）对 175B（3.2M tokens、 $0.6 \times 10^{-4}$ ）：batch 比 6.4 倍，linear 规则预测 lr 比 1/6.4、sqrt 规则预测 $1/\sqrt{6.4}$，实际为 1/10。计算两个预测与实际的差距，并用 §5 的框架解释：为什么规模放大三个数量级后，「规模本身移动了噪声尺度与 $\epsilon_{\max}$ 」比任何固定幂次规则更重要。

## 参考

- [Goyal et al., 2017](../references.md)：Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour，arXiv:1706.02677，[B]（key: linear-scaling）——linear scaling rule 与假设链、gradual warmup（式 (5)）
- [McCandlish et al., 2018](../references.md)：An Empirical Model of Large-Batch Training，arXiv:1812.06162，[B]（key: critical-batch）——式 (6)–(10)、 $`\mathcal{B}_{\mathrm{noise}}`$ 与 $`\mathcal{B}_{\mathrm{simple}}`$ 的区分、温度口径
- [Liu et al., 2020](../references.md)：On the Variance of the Adaptive Learning Rate and Beyond，arXiv:1908.03265，[A]（key: radam）——式 (1)(2) 与「warmup 是方差缩减」的论证
- [Hu et al., 2024](../references.md)：MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies，arXiv:2404.06395，[A]（key: minicpm）——WSD 式 (4)、checkpoint 复用与 10% decay 段实验、最优 batch 幂律
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——warmup + 恒定 + cosine + 常数尾段日程与 batch ramp（§4.2）
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——405B 日程与 batch 两次翻倍（§3.4.1；「252M tokens」节点存疑，见 §5）
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——7B 日程、lr 扫描交叉点、线性收尾等效（§3.5）
- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）——Table 2.1 的 lr/batch 缩放律与 warmup/ramp 口径（App. B）
