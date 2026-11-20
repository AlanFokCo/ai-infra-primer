# 7.3 训练稳定性：Loss Spike、监控与干预（Training Stability: Loss Spikes, Monitoring, and Intervention）

> 训练为什么会炸、怎么提前看见、炸了怎么办——本章依次回答这三个问题：loss spike 的四类触发机制（数据 / 优化器 / 数值 / 硬件，即 2.4 §5 预告的分类学）、以 grad norm 与 update ratio 为核心的监控指标（含 spike 检测判据与正常区间的推导）、带适用判据的四档干预手册。各家记录全部回原文核实：GLM-130B 的 spike 日志与 EGS、OLMo 2 的八项干预与 spike score、DeepSeek-V3 的零不可恢复 spike 口径、Llama-3 的 466 次中断统计、Kimi K2 的 MuonClip | 前置：2.4（梯度裁剪、稳定原语 placement 总图、GLM-130B 案例）、2.5 §7（MuonClip）、2.2（AdamW 的 $m$ / $v$ 动力学与 $\epsilon$ regime）、2.3（warmup 论证）、3.1 §5（z-loss 梯度与 injection point 对照表）| 锚点：GLM-130B [A]（arXiv:2210.02414 §3）、OLMo 2 [A]（arXiv:2501.00656 §3）、DeepSeek-V3 [A]（arXiv:2412.19437）、Llama-3 herd [A]（arXiv:2407.21783 §3.3.4）、Kimi K2 [A]（arXiv:2507.20534 §2.1）

## 1. spike 的可观测形态：grad norm 先行，loss 后到

本章口径：监控与干预的算例统一取 Llama-3 405B、16384 卡一档；各家稳定性记录按其自身口径给出，不跨模型混用数字。

训练崩溃之前，机器先发出了警告。GLM-130B 的训练日志里有一条少见的带逐步记录的实证：「a training collapse usually lags behind a 'spike' in gradient norm by a few training steps」（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]，§3）。翻译过来：梯度范数先突刺，数步之后 loss 才崩溃。grad norm 突刺是先行指标，不是崩溃的同义词。

OLMo 2 在自己的基线 OLMo-0424 上独立看到了同一形态（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§3 开头的问题清单）：loss 突刺之前，往往先有更剧烈的 grad norm 突刺；模型越大，突刺越频繁；grad norm 的慢速爬升与突刺频率正相关。两条曲线合流之后，训练走向发散。

先行性直接决定检测器架在哪里：架在梯度端，比架在 loss 端早数步。判据本身只需要一行滚动统计。设被监控序列为 $x_t$（grad norm、loss 或其他标量指标），取最近 $W$ 步的滚动均值 $`\bar{x}_t^{(W)}`$ 与滚动标准差 $`\sigma_t^{(W)}`$，标准化偏离定义为

$$
s_t = \frac{x_t - \bar{x}_t^{(W)}}{\sigma_t^{(W)}}, \qquad \text{spike 判据：} \quad s_t \gt k \qquad (1)
$$

**符号**： $W$：基线窗口； $k$：阈值倍数。OLMo 2 的 spike score 就是式 (1) 的触发率版本：「时间序列中与最近 1000 个值的滚动均值相距至少 7 个标准差的点所占百分比」（其 §3.2 的 Spike score 段落，[A]）。换句话说，取 $W = 1000$、 $k = 7$，统计式 (1) 在序列上的触发频率——这就是 spike score，主要用于 loss 与 grad L2 norm 两条曲线。

$k = 7$ 的误报账可以先算。标准正态的右尾 $`\Pr(s \gt 7) \approx 1.3 \times 10^{-12}`$，一万步里期望误报约 $10^{-8}$ 次——高斯假设下几乎永不误报。真实梯度序列是重尾的，但 $k = 7$ 仍把日常抖动与 spike 分得足够开。OLMo 2 全章的消融收益（ §6 的 0.40 → 0.03 等数字）都用这个口径报告。

窗口 $W$ 怎么选？两边都有代价。 $W$ 大，基线稳，但跟不上 phase 切换——warmup 结束、schedule 拐点这类系统性漂移会被误报成 spike。 $W$ 小， $\sigma_t^{(W)}$ 的估计噪声大。OLMo 2 的 $W = 1000$ 是在 7B/13B 全量训练上跑通的工作点，不是定理。换 schedule 形态（如 WSD 的 decay 拐点，2.3）时，基线窗要跨过拐点重估（本书工程口径）。

> **坑**：基线窗跨过 warmup 段——系统性斜升被当成 spike，现象是 phase 交界处的告警风暴。规避：基线从 warmup 结束后起算（练习 4 复现此现象）。

本节的两条记录还说明一件事：spike 不是单点事件，而是一个**过程**——先行（grad norm）、爆发（loss spike）、收尾（自行恢复或崩溃）三段。监控要抓第一段，干预要在第二段之前。 §4 的流水线与 §5 的手册就按这个顺序组织。

## 2. 机制分类学：四类触发通道

spike 从哪里来？2.4 §5 预告的四类框架给出骨架，本节逐类兑现：每类给机制、实证与对策。

### 2.1 数据触发：坏 batch

OLMo 2 的数据侧证据是最干净的一组（§3.1，[A]）。回查发生突刺的 batch，含长重复 n-gram 的序列高频出现——论文给出具体实例，比如同一段字符串重复数十次的 token 流。

但因果是统计性的，不是决定论的。论文自己列了三条反例形态：同一段重复序列在大模型上触发突刺，在小模型上未必；同一序列换一种 shuffle 后可能不再触发；同样的序列也出现在没有突刺的 batch 里。所以结论只能落在频率上：滤掉这类序列，平均而言突刺变少。

OLMo 2 的对策是两层过滤。第一层在数据清洗时做：删除一切含「32 个及以上重复」的 n-gram（n 取 1 到 13 个 token）的文档。第二层在 trainer 里做：加载时再检测一遍，命中就在计算 loss 时 mask 掉对应序列（[A]，§3.1 与 §2.1 数据管线口径）。

机制上怎么理解（本书分析）？坏 batch 把单步梯度 $g_t$ 推出它的历史分布，等价于把参数朝一个曲率未知的方向迈了一步 $\eta \cdot u$。LR 处于高位时，这一步足以把参数带出当前盆地边缘，后续几步在坏地形上继续放大偏差。shuffle 依赖（同一序列换个顺序就不触发）说明触发取决于「坏样本出现的时机 × 当时的优化器状态」，而不是样本本身的固有属性。这条性质在 §5.2 的 rewind 手段里会被正面利用。

### 2.2 优化器触发：二阶矩的宿醉与失配

Adam 的二阶矩是 spike 的受害者兼放大器。2.4 §4 的账这里直接用：典型量级下 $\hat{v}$ 稳定在 $v_0$；一个 $k$ 倍幅度的 spike 把该坐标 $v$ 瞬时抬高约 $(1-\beta_2) \thinspace k^{2} \thinspace v_0$（ $\beta_2 = 0.95$、 $k = 100$ 时约 500 倍），随后以时间常数 $1/(1-\beta_2) \approx 20$ 步指数回落。也就是说，spike 过后约 20 步里更新幅度被持续压缩——形象一点叫「宿醉」；spike 方向在 $m$ 里还要驻留约 $1/(1-\beta_1) \approx 10$ 步。

宿醉有两层后果。第一，若 spike 已把参数推坏，宿醉期的小步长会拖慢恢复。第二，若 spike 是数值性的（ §2.3），优化器状态把一次性污染固化成数十步的统计偏差。这条账也是干预纪律的来源：rewind（回滚到早先 checkpoint 重训）必须连优化器状态一起回滚（ §5.2），而干净的故障恢复可以不动它（7.4 §5）。

$\epsilon$ 是优化器侧另一个被实证过的旋钮。OLMo 2 把 AdamW 的 $\epsilon$ 从 $10^{-5}$ 降到 $10^{-8}$，早期更新幅度放开，grad norm 更快收敛，且此后**永久更低**（其 §3.4.1，[A]）。机制解释在 2.2 §4 的两个 regime：早期 $\hat{v}$ 普遍偏小， $\epsilon = 10^{-5}$ 会把大批坐标压在线性 regime，更新被钳小。

Muon 侧有另一种优化器触发的表型。Moonlight 的训练日志记录：max attention logit 在特定层持续爬升、突破 100，而 loss 与 grad norm 全程平稳（AdamW 对照更健康）。K2 的中规模实验里，裸 Muon 下该指标迅速破 1000，并伴随 loss spike 乃至偶发发散（完整口径与 MuonClip 对策归 2.5 §7）。注意纪律：「为什么 Muon 下 logit 更易漂移」在公开文献里只有现象学记录，没有机制定论（2.5 §9 第 5 条），本书不补推测。

### 2.3 数值触发：溢出与 logit 爆炸

GLM-130B 的数值侧记录（§3，[A]）是 fp16 混合精度训练中公开记录最完整的一份。它的混合精度方案是 FP16 前向反向 + FP32 优化器态与主权重（Apex O2 口径）。训练早期就频繁 loss spike，且「随训练推进越来越频繁」。spike 分两种形态：一部分没有明确原因、自行恢复；另一部分以 grad norm 骤升为前兆，数步后 loss spike 甚至 NaN。

论文点名了两个随规模放大的数值问题。第一，Pre-LN 下主支数值在深层可以极大——对策是换成 DeepNorm 型 Post-LN（其 §2.1；LN 变体的梯度分析归 models 2.7）。第二，attention score 大到超出 FP16 动态范围（fp16 的 softmax 溢出账见 models 2.3 §4：logit 约 11 即溢出）。

被排除的选项同样有信息量：CogView 的 PB-Relax（去 bias 加极值扣除）在 GLM-130B 上不能避免发散；BLOOM-176B 的 BF16 方案在 GLM 的平台上要多耗约 $15\text{％}$ 显存（梯度累加转 FP32），且不支持 V100；BLOOM 的 embedding norm 被其作者报告损害 zero-shot 能力（以上均为 GLM-130B §3 的转述口径）。

attention logit 爆炸的稳定原语全部在模型侧，本章只登记指针：QK-norm（结构性封顶）与 soft-capping（tanh 软钳）的公式走读归 models 2.7 §7-8；z-loss 有两个作用点——router 侧（models 4.5 式 (12)，ST-MoE 谱系）与输出侧（models 8.5 式 (7)，Chameleon）；训练侧的梯度推导与三者 injection point 对照表已在 3.1 §5 交付。这类手段在分类学里的位置：把数值触发通道在**前向内部**封死，让 §1 的检测器根本看不到 spike。OLMo 2 给了一条同层的实证：把 RMSNorm 移到子层输出侧（reordered norm）并叠加 QK-norm，梯度 spike score 从 0.108 降到 0.069（其 §3.3.2，[A]）——架构摆放本身就是稳定性手段。

### 2.4 硬件触发：SDC 与故障统计口径

大规模训练里硬件故障有多频繁？Llama-3 405B 的 54 天预训练窗口给出了唯一公开的逐类统计（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.3.4 及 Table 5）：466 次作业中断，其中 47 次计划内（固件升级等自动维护与运维操作），419 次意外。意外中断里约 $78\text{％}$ 归因于确认或疑似的硬件问题（含静默数据损坏与计划外主机维护），GPU 类是最大类别（正文口径：占全部意外中断的 $58.7\text{％}$ ）。整个窗口只需 3 次显著人工干预，其余由自动化处理，有效训练时间保持在 $90\text{％}$ 以上。Table 5 行项摘录（次数与 % 均为原表口径）：

| 组件 | 类别 | 次数 | 原表占比 |
| --- | --- | --- | --- |
| Faulty GPU | GPU | 148 | $30.1\text{％}$ |
| GPU HBM3 Memory | GPU | 72 | $17.2\text{％}$ |
| Software Bug | Dependency | 54 | $12.9\text{％}$ |
| Network Switch/Cable | Network | 35 | $8.4\text{％}$ |
| Host Maintenance | 计划外维护 | 32 | $7.6\text{％}$ |
| GPU SRAM Memory | GPU | 19 | $4.5\text{％}$ |
| GPU System Processor | GPU | 17 | $4.1\text{％}$ |
| Silent Data Corruption | GPU | 6 | $1.4\text{％}$ |
| GPU Thermal Interface + Sensor | GPU | 6 | $1.4\text{％}$ |

口径注记（本书核对）：正文「GPU 类占 $58.7\text{％}$ 」与 Table 5 GPU 类逐行合计（268 次 / 419 次 $\approx 64\text{％}$ ）存在小差异。根源可定位：原表 % 列的 GPU 六行相加恰为 $58.7\text{％}$ ——正文数字是 % 列之和；而 % 列首行自身口径异常（148 次对 419 分母应为 $35.3\text{％}$ 而非 $30.1\text{％}$ ，其余行与 419 分母相容）——原文的 % 列与次数列不自洽，表文差异源于此。本书按原文各自照录、不做调和。

SDC（silent data corruption，静默数据损坏）单独立类的理由是它不报错：计算结果错，硬件状态却正常，loss 曲线上表现为无数据规律、无先行预警的偏移。6 次 / 54 天听着少，但它是四类触发里唯一无法从日志提前检出的通道（本书分析）。检出只能走间接路径： §1 的统计判据先报警，再对疑似 rank 做定点硬件测试定位。另有一组数字给出规模感：同步训练下，单个 GPU 故障就可能需要重启整个作业（ §3.3.4 原文）。checkpoint 与自动恢复如何把这类中断的代价压小，归 7.4。

四类通道的汇总（检测信号 → 一手对策）：

| 触发类 | 典型指纹 | 一手对策 |
| --- | --- | --- |
| 数据 | spike 与特定 batch 相关；换 shuffle 后不复现 | n-gram 过滤 + mask；skip batch / rewind 换数据（ §5） |
| 优化器 | spike 后 $`\hat{v}`$ 宿醉约 $1/(1-\beta_2)$ 步；或 max logit 慢爬（Muon） | clip 兜底（2.4）； $\epsilon$ 下调（OLMo 2）；MuonClip（2.5 §7） |
| 数值 | attention score / logit 幅度逼近精度上限；fp16 溢出 | QK-norm / soft-capping / z-loss（models 2.7、3.1 §5）；混合精度配置（1.3） |
| 硬件 | 无数据规律、无预警；伴随机群事件 | SDC 间接检出 + 硬件隔离；中断恢复归 7.4 |

## 3. 监控指标体系：六个量与两条推导

监控什么、阈值定在哪？指标设计的第一原则已在 2.4 §3 立住：不存在跨模型通用的正常区间，每个指标按本模型自身历史（warmup 结束后的分位数）定基线。本节做两件事：先把全书用到的六个监控量登记成表，再用两条推导把「正常区间」从经验变成可算的数。

| 指标 | 定义 | 正常形态 | 异常形态（指向） |
| --- | --- | --- | --- |
| grad norm | 全体梯度拼接的 L2 范数（2.4 式 (1)，clip 接口白送） | warmup 后收窄进稳态带 | 突刺（spike 先行指标， §1）；缓慢单调爬升（OLMo-0424 形态， §2.1 类） |
| param norm | $\lVert \theta \rVert$（全局或逐层） | 有 wd 时收敛到准稳态（式 (4)） | 单调爬升（wd 不足，Moonlight 的 wd 动机）；塌陷（wd 过度，OLMo 2 embedding 案例） |
| update ratio | $r = \lVert \Delta\theta \rVert / \lVert \theta \rVert$（式 (2)） | 每步约 $10^{-3}$ 量级（式 (3) 推算），随 LR schedule 同步起伏 | 无 LR 变化而跳变（spike 指纹）；趋零（更新被压死） |
| logit max | batch 内最大 attention logit（2.5 式 (13) 的 $S_{\max}$ 口径） | 有界（K2 以 $\tau = 100$ 封顶） | 持续爬升破百（Muon 栈的爆炸前兆， §2.2） |
| attention entropy | 各行注意力分布的熵（本书工程口径） | 稳定带 | 塌缩向 one-hot（饱和）或摊平向均匀（注意力失焦） |
| LR | schedule 当值 | 与 recipe 逐点一致 | 任何偏离 = 实现 bug 的最快检出 |

**命题**（update ratio 的正常区间）：在 AdamW 更新与统计平稳假设下，逐层更新比的量级由学习率与参数尺度之比决定，落在每步千分之一档。**推导**如下，先给定义：逐层（或逐张量）的更新比

$$
r_\ell = \frac{\lVert \Delta\theta_\ell \rVert}{\lVert \theta_\ell \rVert} \qquad (2)
$$

**符号**： $\theta_\ell$：第 $\ell$ 个参数张量； $\Delta\theta_\ell$：当步更新量。代入 AdamW 更新（2.2 式 (9)） $\Delta\theta = -\eta \thinspace u - \eta \lambda \theta$（ $u = \hat{m} / (\sqrt{\hat{v}} + \epsilon)$ ）。设 $u$ 与 $\theta$ 不相关（ $\langle u, \theta \rangle \approx 0$ ），范数平方拆开： $\lVert \Delta\theta \rVert^{2} \approx \eta^{2} \lVert u \rVert^{2} + \eta^{2} \lambda^{2} \lVert \theta \rVert^{2}$，逐坐标取 RMS 口径得

$$
r_\ell \approx \eta \thinspace \sqrt{\left( \frac{\mathrm{RMS}(u_\ell)}{\mathrm{RMS}(\theta_\ell)} \right)^{2} + \lambda^{2}} \qquad (3)
$$

三角不等式另给出逐项对齐的最坏上界 $\eta \thinspace \mathrm{RMS}(u_\ell) / \mathrm{RMS}(\theta_\ell) + \eta \lambda$。 $\mathrm{RMS}(u)$ 的实证区间为 $0.2\text{–}0.4$（Moonlight §2.2 正文口径，2.5 §4 已核）。代入具体数字：Llama-3 405B 的峰值 LR 是 $\eta = 8 \times 10^{-5}$（herd Table 3），初始化口径下 $\mathrm{RMS}(\theta) \approx 0.02$，主项比值 $10$ 到 $20$，于是 $r \approx 8 \times 10^{-5} \times (10 \text{–} 20)$，即约 $8 \times 10^{-4}$ 到 $1.6 \times 10^{-3}$ 每步；wd 项进根号的份额为 $\lambda^{2} / 15^{2} \approx 4 \times 10^{-5}$，对 $r$ 的贡献可忽略。

**正常区间的结构由此而来**（本书推算，输入假设如上）：量级 $10^{-3}$ 每步，逐层同档，形状跟随 LR schedule。三种异常各有指向： $r$ 无 LR 变化而跳升，分子出 spike（梯度端病灶）； $r$ 趋零，分母爆炸或 $u$ 被压死（宿醉或 $\epsilon$ regime 错配）；某层 $r$ 系统性偏离全场，该层即病灶（GLM-130B 的 embedding 异常就是这种指纹）。

**命题**（wd 平衡下的 param norm 准稳态）：带解耦衰减时，参数 RMS 收敛到由学习率、衰减系数与更新幅度决定的准稳态，不无限漂移。**证明路线**：逐坐标平方期望得二阶矩递推，稳态时令两端相等，小量近似后解出；关键假设是更新与参数不相关。带解耦 wd 的更新是 $\theta_t = (1 - \eta\lambda) \thinspace \theta_{t-1} - \eta \thinspace u_t$。设 $u$ 统计平稳（ $\mathrm{RMS}(u)$ 恒定）且与 $\theta$ 不相关，逐坐标取平方期望：

$$
\mathbb{E}[\theta_t^{2}] = (1 - \eta\lambda)^{2} \thinspace \mathbb{E}[\theta_{t-1}^{2}] + \eta^{2} \thinspace \mathbb{E}[u^{2}]
$$

稳态时令两端相等，并用 $(1 - \eta\lambda)^{2} \approx 1 - 2\eta\lambda$（ $\eta\lambda \ll 1$ ），解出

$$
\mathrm{RMS}(\theta^{\ast}) \approx \sqrt{\frac{\eta}{2\lambda}} \cdot \mathrm{RMS}(u) \qquad (4)
$$

代入 $\eta = 8 \times 10^{-5}$、 $\lambda = 0.1$、 $\mathrm{RMS}(u) = 0.3$： $\mathrm{RMS}(\theta^{\ast}) \approx \sqrt{4 \times 10^{-4}} \times 0.3 \approx 6 \times 10^{-3}$，与 $0.02$ 的初始化同量级。结论：wd 把参数幅度固定在 $\sqrt{\eta / \lambda}$ 决定的尺度上，而不是放任漂移。监控含义：param norm 持续爬升，意味着 $\mathrm{RMS}(u)$ 在涨（梯度尺度漂移）或 wd 等效变弱——它是对「LR / wd 是否被悄悄改动」的独立校验。诚实边界：式 (4) 依赖不相关假设与平稳性，属本书推导的工程近似（文献中有同类平衡关系的独立研究，本书 references 未收录，不引为出处）；练习 1 讨论假设放松后的修正方向。

## 4. 检测流水线：从判据到动作

判据怎么接到动作上？式 (1) 落地为逐指标的分层流水线：硬层零延迟，软层带基线。

```
逐步信号            检测层                            动作层（§5）
─────────────   ─────────────────────────────   ─────────────────────
loss / grad     非有限检查（NaN/Inf，2.4 §1        触发即 skip batch（§5.1）
                的 error_if_nonfinite）
grad norm       式 (1)：s_t > k（OLMo 2 工作点    单次→告警；连续→rewind 评估
                W=1000, k=7）                     （§5.2）
update ratio    偏离自身基线带（式 (3) 结构）      逐层定位→查该层梯度 / 参数
logit max       S_max > τ（Muon 栈，τ=100）       MuonClip 自动收缩（2.5 §7）
─────────────   ─────────────────────────────   ─────────────────────
全部序列进 spike score 周报（7σ / 1000 窗触发率，OLMo 2 §3.2 口径），
跨 phase 比较稳定性回归
```

两个设计点值得单说。其一，**grad norm 是主检测面**： §1 的先行性（GLM-130B：崩溃滞后 grad norm 突刺数步）给出数步干预窗口，loss 端判据只是确认。其二，**非有限检查是唯一硬判据**：它不依赖基线、零误报，实现上就在 clip 的范数计算里（2.4 §1 的 `error_if_nonfinite=True` 把 NaN/Inf 从静默缩放变成显式异常）——检测层的最低成本项应该最先架。

## 5. 干预手册：四档手段与适用判据

检测到 spike 之后做什么？干预按代价升序排列。每一档给三样东西：触发判据、机制语义、先例。

### 5.1 skip batch（步级，代价约零）

**判据**：当步 loss 或 grad 非有限；或式 (1) 触发且幅度超硬阈（如 $s_t \gt 2k$，本书口径）。**动作**：该 batch 不反传、不更新（等价于从数据流剔除），schedule 照常前进。**先例**：OPT-175B 的处理就是手动跳过数据加调超参（GLM-130B §3 转述，[A]）；OLMo 2 的 trainer 侧 n-gram mask 是同一思想的自动化，粒度从「整个 batch」细化到「坏序列」（ §2.1）。**代价账**：一个 batch 约 $10^{7}\text{–}10^{8}$ token（各家末值 batch 口径：Llama-3 的 16M 到 GLM-4.5 的 64M，详表归 7.2），相对 $10^{13}$ 量级的总预算可忽略。但连续触发说明触发源未被隔离——升级到 §5.2。

### 5.2 rewind + 换数据（段级，代价 = 回滚段重算）

**判据**：spike 反复发生且模式定位到数据段，或 loss 在宿醉窗口内不自行恢复。**动作**：回到 spike 前最近的健康 checkpoint（完整内容语义归 7.4 §1），跳过或重排触发段数据后重训。

**合法性**来自 §2.1 的顺序依赖：同一序列是否触发 spike 依赖数据顺序（OLMo 2 §3.1），回滚后换 shuffle 即换触发面，不必精确复现原顺序。注意一条纪律：**rewind 必须连优化器状态一起回滚**（ §2.2 的宿醉账），否则旧 $\hat{v}$ 会把污染带进新轨迹。

**代价账**：回滚 $K$ 步即损失 $K$ 步算力。以 16K 卡集群每步约 6 s 的量级（7.4 §2 的账）， $K = 1000$ 约 1.7 小时墙钟，占 54 天窗口总步数的约千分之一。GLM-130B 全程仅 3 次晚期发散，且进一步收缩 embedding 梯度即可稳定（ §6）——定点处理之后，rewind 的跨度可以很短。

### 5.3 重新 warmup（段级，配合 §5.2）

**判据**：回滚跨度大，或更换了任何稳定性相关超参（ $\epsilon$、clip 阈值、wd 名单）后重启。**动作**：从回滚点把 LR 重新线性爬升到 schedule 当值，窗口远短于初始 warmup（数百步量级，本书工程口径）。**语义**：初始 warmup 针对的是冷优化器状态与未稳的梯度统计（2.3 的双重论证）；rewind 后的优化器状态虽来自 checkpoint，但若回滚点本身处于不稳区段，重爬 LR 等价于把 §2.2 的宿醉窗口让过去。**对照**：干净故障的例行恢复续训不需要重新 warmup——论证归 7.4 §5。

### 5.4 局部降 LR / 延长 decay（段级到 phase 级）

**判据**：不稳定与 LR 相位相关（峰值平台期反复 spike），且前三档无效或代价过高。**动作**：降峰值、延长 decay，或把当前 phase 的 LR 曲线整体下移。**先例**：OLMo 2 的 §3.5 即 LR 专项研究（7B 峰值 $3 \times 10^{-4}$ 的选取过程，[A]）。**语义**： §2.2 的账里，LR 是所有触发通道后果的统一乘子——降 LR 不治病根、只降当量，适合作为冲到 schedule 末段前的保底；若 spike 源在数据或硬件，降 LR 只是推迟复发。

四档的决策序一句话版：步级异常先 skip；段级反复则 rewind + 换数据 + 重 warmup；phase 级顽固才动 LR；任何一档落地后， §3 各指标的基线带都要重估。

## 6. 各家稳定性记录：四份一手口径

各家在真实规模上遇到了什么、又怎么收场？四份一手口径如下。

- **GLM-130B**（[Zeng et al., 2022](../references.md)，arXiv:2210.02414，[A]，§2-3）：FP16 Apex O2，早期频繁 spike 且随训练加剧。spike 分两类形态：无因自恢复、grad norm 先行后崩溃至 NaN。根因定位到三处：embedding 异常梯度（早期比其他层高几个数量级）、Pre-LN 深层无界、attention score 超 FP16 范围。对策是 DeepNorm 型 Post-LN 加 EGS（收缩因子 $\alpha = 0.1$ ，「wipes out most spikes」；公式与逐层对照归 2.4 §5）。最终完整训练只经历 3 次晚期发散，硬件故障造成的失败反而更多——这是「稳定性工程收敛后，硬件成为主要中断源」的最早公开实例。
- **OLMo 2**（[Team OLMo et al., 2025](../references.md)，arXiv:2501.00656，[A]，§3）：八项干预逐项带消融：n-gram 过滤（§3.1）；统一 $\mathcal{N}(0, 0.02^{2})$ 初始化替换逐层缩放初始化（§3.2，测试设置下旧初始化快速失稳，新初始化无 loss spike、grad spike score 0.40 → 0.03）；回归 RMSNorm（§3.3.1）；reordered norm + QK-norm（§3.3.2，梯度 spike score 0.108 → 0.069）；z-loss 系数 $10^{-4}$ 的 $(\log Z)^{2}$（§3.3.3）； $\epsilon$ 从 $10^{-5}$ 到 $10^{-8}$（§3.4.1）；embedding 免 wd（§3.4.2，spike score 0.16 vs 0.092）；LR 专项（§3.5）。这是公开文献里唯一逐项带消融数字的稳定性工程清单。
- **DeepSeek-V3**（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，摘要与 §1）：「整个训练过程未遭遇任何不可恢复的 loss spike，也未执行任何回滚」——14.8T token 全程、2.788M H800 GPU 小时。口径注记：这是**结果声明**（零不可恢复 spike），不等于零 spike 事件。它的数值与架构侧设计（FP8 方案、aux-loss-free 路由）分归 1.3 与 models 4.5。
- **Kimi K2**（[Kimi Team, 2025](../references.md)，arXiv:2507.20534，[A]，§2.1）：MuonClip（Muon + wd + RMS 匹配 + QK-Clip，其 Algorithm 1 的组合体）加 WSD schedule，15.5T token 零 loss spike。QK-Clip 把 max logit 封顶 $\tau = 100$；约 $30\text{％}$ 步数后 logit 自行回落， $\tau$ 全程不调。机制、公式与 MLA 分量规则归 2.5 §7。它是 §2 分类学中「优化器触发」类目前唯一有完整工程闭环的公开案例：监控量 $S_{\max}$ 逐 head，执行器是 post-update 权重缩放，不进计算图、不污染梯度统计。

横向读法：四份记录横跨 130B dense、7B-13B、671B MoE、1T MoE 四个量级，对策收敛到同构的三层：**架构层**（norm 摆放、QK-norm、DeepNorm）、**优化器层**（ $\epsilon$、wd 名单、EGS、MuonClip）、**数据层**（n-gram 过滤）。每一层对应 §2 的一个触发通道；硬件层的中断由 7.4 的容错链路承接。「没有单一银弹」与「不可恢复 spike 可声明为零」两份记录之间，差异不在某一个手段，而在三层手段的叠加与 §3-4 的监控闭环。

## 总结

| 主题 | 结论 | 公式 / 出处 |
| --- | --- | --- |
| spike 形态 | grad norm 突刺先行于 loss 崩溃数步；检测架在梯度端 | GLM-130B §3 [A]；OLMo 2 §3 [A] |
| 检测判据 | 式 (1)： $`s_t = (x_t - \bar{x}_t^{(W)}) / \sigma_t^{(W)} \gt k`$；OLMo 2 工作点 $W$=1000、 $k$=7，spike score = 触发率 | OLMo 2 §3.2 [A] |
| 四类触发 | 数据（n-gram，顺序依赖）/ 优化器（ $`\hat{v}`$ 宿醉 $1/(1-\beta_2)$ 步、Muon logit 漂移）/ 数值（fp16 溢出、logit 爆炸）/ 硬件（SDC 不可提前检出） | §2 表；GLM-130B / OLMo 2 / herd Table 5 |
| update ratio | 式 (2)(3)： $r \approx \eta \sqrt{(\mathrm{RMS}(u)/\mathrm{RMS}(\theta))^{2} + \lambda^{2}}$；正常量级约 $10^{-3}$ 每步、随 schedule 同步 | 本书推算（RMS 区间引 Moonlight 口径） |
| param norm 准稳态 | 式 (4)： $\mathrm{RMS}(\theta^{\ast}) \approx \sqrt{\eta/(2\lambda)} \cdot \mathrm{RMS}(u)$；爬升 = wd 不足或 $u$ 漂移 | 本书推导（不相关假设，练习 1） |
| 干预四档 | skip batch（步级）→ rewind + 换数据（段级，连优化器状态回滚）→ 重新 warmup（配合回滚）→ 局部降 LR（保底） | §5；OPT / OLMo 2 / GLM-130B 先例 |
| 各家记录 | GLM-130B：3 次晚期发散 + EGS；OLMo 2：八项干预带消融；DeepSeek-V3：14.8T 零不可恢复 spike；K2：15.5T 零 spike（MuonClip） | §6，全部 [A] |
| 中断统计 | 16K GPU × 54 天：466 次中断、419 次意外、约 $78\text{％}$ 硬件、3 次人工干预、 $90\text{％}$ + 有效训练时间 | herd §3.3.4 Table 5 [A] |

## 练习

1.（计算 · [20]）update ratio 扫描：取 Llama-3 8B 峰值 $\eta = 3 \times 10^{-4}$（herd Table 3）、 $\lambda = 0.1$、 $\mathrm{RMS}(u) \in [0.2, 0.4]$、 $\mathrm{RMS}(\theta) = 0.02$：先写出你对（b）的数量级预测，再计算。(a) 算 $r$ 的区间与 wd 项占比；(b) cosine decay 到峰值 $10\text{％}$ 时 $r$ 应为多少？(c) 若实测 $r$ 在 decay 段不随 LR 下行，按 §3 的异常指向列出两个候选解释并给出区分它们的监控量。校验：（b）的结果应恰为（a）区间的十分之一——更新比对学习率线性。

2.（推导 · [30]）式 (4) 的完整版：从 $\theta_t = (1-\eta\lambda) \thinspace \theta_{t-1} - \eta \thinspace u_t$ 出发，写出逐坐标二阶矩递推时保留互相关项 $\mathbb{E}[\theta_{t-1} u_t]$，指出正文的不相关假设在哪一步进入；讨论 $u$ 与 $\theta$ 负相关（梯度把参数往回拉，wd 与梯度同向）与正相关两种情形下，真实稳态 $\mathrm{RMS}(\theta^{\ast})$ 相对式 (4) 的偏移方向。

3.（分析 · [30]）判据稳健性：设稳态段 grad norm 近似对数正态而非正态，论证式 (1) 在固定 $k$ 下的误报率相对高斯情形上升还是下降（提示：比较两种分布的标准化尾部）；据此解释 OLMo 2 为什么用「滚动窗 + 触发率百分比」而不是固定绝对阈值作 spike score，并说明 warmup 段为什么不应计入基线窗。

4.（代码 · [30]）教学实现 spike 检测器（教学用，非优化实现）：合成一条 grad norm 序列——对数正态底噪 + warmup 段斜升 + 每 2000 步注入一个 8 倍突刺；先写出你预期的检出延迟量级，再动手。实现式 (1) 的滚动判据（ $W = 500$， $k \in \lbrace 5, 7 \rbrace$ 两组），统计误报率与检出延迟（从突刺到报警的步数）；验证「基线窗跨过 warmup 段时误报集中在 phase 交界」的现象，并给出修复（基线从 warmup 结束后起算）。校验：阈值取 7 时误报率应比取 5 低至少一个量级；修复后 warmup 段的误报应消失。

## 参考

- [Zeng et al., 2022](../references.md)：GLM-130B: An Open Bilingual Pre-trained Model，arXiv:2210.02414，[A]（key: glm-130b）——§3 的 spike 记录（两类形态、grad norm 先行、embedding 异常梯度、3 次晚期发散）、混合精度与 DeepNorm / EGS 对策、OPT-175B 与 BLOOM-176B 先例转述
- [Team OLMo et al., 2025](../references.md)：2 OLMo 2 Furious，arXiv:2501.00656，[A]（key: olmo2）——§3 稳定性深潜：OLMo-0424 症状清单、n-gram 过滤（§3.1）、初始化（§3.2，spike score 0.40 → 0.03）、reordered norm + QK-norm（§3.3.2，0.108 → 0.069）、z-loss（§3.3.3）、 $\epsilon$（§3.4.1）、embedding 免 wd（§3.4.2）、spike score 定义（§3.2）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——摘要与 §1 的零不可恢复 spike / 零回滚口径、14.8T token 与 2.788M H800 GPU 小时
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——§3.3.4 与 Table 5 的中断统计（466 / 419 / 47、约 $78\text{％}$ 硬件、GPU $58.7\text{％}$ 、SDC 6 次、3 次人工干预、 $90\text{％}$ + 有效训练时间）；Table 3 的峰值 LR（8B: $3 \times 10^{-4}$、405B: $8 \times 10^{-5}$）
- [Kimi Team, 2025](../references.md)：Kimi K2: Open Agentic Intelligence，arXiv:2507.20534，[A]（key: kimi-k2）——§2.1 的 MuonClip 与 15.5T token 零 spike 口径（公式归 2.5 §7）
- [Liu et al., 2025](../references.md)：Muon is Scalable for LLM Training（Moonlight），arXiv:2502.16982，[A]（key: moonlight）——附录 D 的 max attention logit 记录、AdamW 更新 RMS 0.2–0.4 的 §2.2 正文口径（经 2.5 §4 转引）
