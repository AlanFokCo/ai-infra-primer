# Training W1 审查报告 · 批 2（章节 2.1–2.3 对抗审查）

- 日期：2026-11-17
- 审查人：独立 fresh-context evaluator（批 2，与写作方无共享上下文）
- 对象：`training/02-optimizers/01-sgd-and-momentum.md`、`02-adam-and-adamw.md`（推导重章）、`03-lr-schedules.md`
- 输入清单全部读取：3 章全文、STYLE_GUIDE.md、`docs/reviews/2026-11-15-style-audit/rubric.md`、`training/references.md`、training spec §5 的 2.1–2.3 节

## 结论

**S1 = 1；S2 = 3；S3 = 6。不放行**：S1（Adam 步长「严格 < η」的错误定理化）与 3 项 S2 修复后复审。三章的事实底盘整体扎实：12 个 arXiv ID 全命中、七篇论文的关键断言逐句回原文命中、PyTorch 行级走读与 torch 2.8 源码逐项一致、2.2 §9 对拍脚本本机复跑差异 1.19e-7 与文内「1.2e-7」一致。

**写作方存疑项的裁定（Llama-3 batch 翻倍节点「252M tokens」）**：回 herd 论文 2407.21783 原文（ar5iv 全文，§3.4.1）核实——论文文本逐字为 "double these values to a batch size of 8M sequences of 8,192 tokens **after pre-training 252M tokens**"，第二节点 "double the batch size again to 16M after pre-training on **2.87T tokens**"。**2.3 §5 的引用与存疑标注均无误**：正文如实照录原文「252M tokens」、按 4M tokens/步折算约 63 步指出量级可疑、疑为 252B 之误的处置（照录原文 + 显式存疑 + 不擅自改写）完全正确，不记 finding。（2.2 引用同一论文的节号定位错误另记 S2-2。）

## Findings 表

### S1（1 项，必须修复并复审）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S1-1 | 2.2 §3 观察 (ii)；同断言外溢至 2.3 §4（「Adam 的逐坐标更新 $\|\hat m/(\sqrt{\hat v}+\epsilon)\|<1$ 是 sign 型有界量，步长被 $\eta$ 直接封顶（2.2 §3）」） | 「由 EMA 权重（和为 1）的 Cauchy–Schwarz 不等式，逐坐标有 $\hat m^2\le\hat v$，故校正后每坐标每步位移严格小于 $\eta$（ $\|\hat m/(\sqrt{\hat v}+\epsilon)\|<1$ ）」 | C–S 成立要求 $\hat m$ 与 $\hat v$ 用**同一组**归一化权重；$\beta_1\ne\beta_2$ 时两通道权重不同，结论不成立。独立数值反例（本 evaluator 复算）：$\beta_1=0.9,\beta_2=0.999$，$t=100$，$g_i=0$（$i\le90$）、$g_i=1$（$91\le i\le100$）⇒ $m_t=1-0.9^{10}=0.6513$、$\hat m=0.6513/(1-0.9^{100})\approx0.6513$；$v_t=1-0.999^{10}=0.009955$、$\hat v=0.009955/(1-0.999^{100})\approx0.1046$； $\|\hat m/\sqrt{\hat v}\|\approx2.01>1$ 。 $\beta_2=0.95$ 同构造也得 $1.025>1$ 。Adam 原文（1412.6980 §2.1，PDF 逐字核对）自己的口径只是**近似**有界：「 $\|\Delta_t\|\le\alpha(1-\beta_1)/\sqrt{1-\beta_2}$ （最稀疏情形， $(1-\beta_1)>\sqrt{1-\beta_2}$ 时）、否则 $\|\Delta_t\|\le\alpha$ …approximately bounded by $\alpha$（ $\|\Delta_t\|\lessapprox\alpha$ ）」——按 0.9/0.999，稀疏最坏上界即 3.16α，与反例量级一致 | 删去「严格小于」定理化表述，改为双口径：(a) 平稳态下 $\hat m\to\mu$、 $\hat v\to\mu^2+\sigma^2$，Jensen 给出 $\hat m^2\le\hat v$（本章 §2 的 SNR 段已有此正确版本）；(b) 非平稳瞬时步可超过 $\eta$，原文口径的最坏上界为 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$（0.9/0.999 时为 3.16η）。2.3 §4 的「直接封顶」改为「近似封顶（最坏情形上界 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$）」——sqrt 规则的启发式论证不受影响 |

### S2（3 项，必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 2.1 §1（噪声尺度定义）、§2 式 (7) 上下文、总结表「二阶矩分解」行 | 把 $\mathrm{tr}\,\Sigma/\lVert g\rVert^2$ 命名为「梯度噪声尺度（gradient noise scale） $\mathcal B_{\mathrm{noise}}$ 」 | McCandlish 原文（ar5iv 1812.06162 逐式核对）：式 (2.8) 定义 $\mathcal B_{\mathrm{noise}}=\mathrm{tr}(H\Sigma)/(g^\top Hg)$（带 Hessian），式 (2.9) 才定义 $\mathcal B_{\mathrm{simple}}=\mathrm{tr}(\Sigma)/\lVert g\rVert^2$（ $H\propto I$ 化简）。2.1 的推导（L-光滑上界 $g^\top Hg\le L\lVert g\rVert^2$ 路线）产出的恰是 simple 版。同书 2.3 §5 式 (8) 已按原文术语正确区分两者——2.1 与 2.3 符号互相冲突，且违反 spec 纪律「论文区分 $\mathcal B_{\mathrm{simple}}$ 与 $\mathcal B_{\mathrm{noise}}$，引用时按原论文术语」（spec 审查已裁定「critical batch 正名 $\mathcal B_{\mathrm{simple}}$ 」） | 2.1 全文（含式 (7)、总结表）把该量改名为 $\mathcal B_{\mathrm{simple}}$ ，并在首次出现处加一句「McCandlish 原文另定义带 Hessian 的 $\mathcal B_{\mathrm{noise}}=\mathrm{tr}(H\Sigma)/(g^\topHg)$，两者差一个小常数倍（其式 (2.8)/(2.9)）；本章 L-光滑口径对应 simple 版，完整区分见 2.3」 |
| S2-2 | 2.2 §6（「Llama-3 的 scaling-law 实验…§4.2.1」）+ 2.2 参考区 llama3 条目（同节号，两处） | 「每步 weight decay 为当步学习率的 0.1 倍」的出处定位为 herd 论文 §4.2.1 | ar5iv 2407.21783 全文核对：该句（"The weight decay at each step is set to 0.1 times the learning rate at that step"）唯一出现在 **§3.2.1 Scaling Laws**；论文的 §4.2.1 是 Preference Data（post-training），与 optimizer 无关。断言内容本身正确（已核对原文），仅节号定位错误；2.3 §5 对同一论文的 §3.4.1 定位是对的 | 两处「§4.2.1」改「§3.2.1」 |
| S2-3 | 2.2 §7 第三条规律 | 「代入 models 1.2 式 (8) 的优化器状态账，矩估计字节系数 $c$ 从 4 降到 2， $(4+2c)N$ 从 $12N$ 降到 $10N$ bytes；671B 模型这一项省 $2N$ bytes 量级」 | models 1.2 式 (8)（models/01-foundations/02-compute-accounting.md:188 核对）为 $\mathrm{Mem}_{\mathrm{opt}}=(4+2c)N$（fp32 master 4N + 两阶矩各 cN）。 $c:4\to2$ 代入得 $12N\to 8N$ 、省 $4N$——与 DSv3 §3.3.3「两阶矩转 BF16、master 与梯度仍 fp32」（ar5iv 逐句核对）自洽。章内 $10N$ 与「省 $2N$」两处数值均错（少算一半） | 改「从 $12N$ 降到 $8N$ bytes；671B 模型这一项省 $4N$ bytes 量级（约 2.7 GB 级，671B×4 bytes）」 |

### S3（6 项，记录；下波清零时逐条判定）

| # | 位置 | 断言 | 证据 / 处置建议 |
| --- | --- | --- | --- |
| S3-1 | 2.2 §5 中段 | 「对任意满足 $\beta_1<\sqrt{\beta_2}$ 的常数 $\beta_1,\beta_2$，都存在使 Adam 平均 regret 非零的**随机**问题（Theorem 2）」 | 原文（ar5iv 1904.09237）Theorem 2 是 **online**（确定性序列，mod- $C$ 构造）版本，Theorem 3 才是 stochastic 版本（正文随后一句「随机版（Theorem 3）」自身是对的）。建议改「在线问题（Theorem 2）」 |
| S3-2 | 2.1 §4 | 「NAG，[Sutskever et al., 2013] 式 (2) 同处」 | PMLR v28 sutskever13 PDF 逐式核对：原文 CM 为式 (1)(2)，NAG 为式 (3)(4)。「式 (2)」指向错误/歧义。建议改「其式 (3)–(4)」 |
| S3-3 | 2.3 §5 测量段 | 「大 batch 的梯度范数由各 rank 的局部范数**拼出**」 | 原文 App A.1 的机制是「averaging **前后**各取一次范数」（ $B_{\mathrm{small}}$ = all-reduce 前的 local batch、 $B_{\mathrm{big}}$ = 后的 global batch）；全局范数不能由局部范数拼出（缺交叉项）。建议改「all-reduce 前后的两份梯度各提供一次范数测量，零额外通信」 |
| S3-4 | 2.2 §6 式 (9) 后的 folding 句 | 「把 schedule 乘子折进学习率…就是 PyTorch 的形式： $\theta\leftarrow\theta(1-\mathrm{lr}\cdot\lambda)-\cdots$」 | 原文 Algorithm 2 的衰减因子是 $\eta_t\lambda$、torch 是 $\mathrm{lr}\cdot\mathrm{wd}$； $\mathrm{lr}=\eta_t\alpha$ 折合下 $\mathrm{wd}\equiv\lambda/\alpha$——单位换算被静默略过（本书 2.1 式 (10) 正是以这类因子为纪律对象）。建议加半句「即 torch 的 `weight_decay` 对应原文 $\lambda/\alpha$」 |
| S3-5 | training/references.md 条目回写 | olmo2「被引章节」缺 2.2（2.2 §4/§7/§8/参考均引）；deepseekv3 缺 2.2、2.3（两章均引） | 按 references.md 写入规则回写 |
| S3-6 | 2.2 §7 末 | 无 7.2 前瞻指针 | 2.1 §7、2.3 §6 均有「全字段对照表归 7.2」；2.2 §7 的超参表正是该表 optimizer 列，建议末段补一句指针 |

## 强制独立重推导（中间过程全录）

### 2.1(a) momentum 速度形式 ↔ EMA 形式等价（式 (8)(9)(10)）

速度形式 $v_{t+1}=\beta v_t+\hat g_t$、 $\theta_{t+1}=\theta_t-\eta v_{t+1}$、 $v_0=0$；EMA 形式 $m_{t+1}=\beta m_t+(1-\beta)\hat g_t$、 $\theta_{t+1}=\theta_t-\eta' m_{t+1}$、 $m_0=0$。

1. 令 $m_t=(1-\beta)v_t$： $(1-\beta)v_{t+1}=\beta(1-\beta)v_t+(1-\beta)\hat g_t$ 即 $m_{t+1}=\beta m_t+(1-\beta)\hat g_t$ ✓；初值 $m_0=(1-\beta)\cdot0=0$ 相同，归纳成立 ✓。
2. 更新侧： $\theta_t-\eta v_{t+1}=\theta_t-\frac{\eta}{1-\beta}m_{t+1}$ ，与 EMA 形式恒等 ⇔ $\eta'=\eta/(1-\beta)$ ✓（式 (10)）。
3. 展开式 (11)： $v_{t+1}=\sum_{s=1}^{t+1}\beta^{t+1-s}\hat g_s$、权重和 $(1-\beta^{t+1})/(1-\beta)\to1/(1-\beta)$ ✓；有效步长放大 $1/(1-\beta)$ ✓；90% 权重覆盖 $k\ge\ln0.1/\ln\beta$：数值复算 0.9/0.95/0.99 → 21.9/44.9/229.1，与章内 22/45/229 一致 ✓。
4. LR-in-velocity 约定（式 (12)）代换 $u_t=-v_t/\varepsilon$ 化回式 (8)、定常 LR 等价 ✓；LR 变化时历史项需乘 $\eta_{t+1}/\eta_t$（Goyal §3 式 (9)–(10) 原文核对命中 ✓）。

**判定：等价证明无断步，换算因子正确。**

### 2.1(b) 二次碗速率：GD $(\kappa-1)/(\kappa+1)$、heavy ball $(\sqrt\kappa-1)/(\sqrt\kappa+1)$（式 (15)–(20)）

GD 侧：特征基下 $e'=(1-\eta\lambda)e$； $\lvert1-\eta\lambda\rvert$ 关于 $\lambda$ 凸 ⇒ 端点取极；端点平衡 $1-\eta\mu=-(1-\eta L)$ ⇒ $\eta^\ast=2/(L+\mu)$ ✓； $\rho_{\mathrm{GD}}=1-2\mu/(L+\mu)=(L-\mu)/(L+\mu)=(\kappa-1)/(\kappa+1)$ ✓。

HB 侧，逐步重推：

1. 逐坐标递推 $e_{t+1}=(1+\beta-\eta\lambda)e_t-\beta e_{t-1}$（由式 (17) 逐分量展开 ✓），状态矩阵 $M$ 与特征方程 $r^2-(1+\beta-\eta\lambda)r+\beta=0$ ✓。
2. 判别式 $<0$ ⇒ 共轭复根，Vieta 根之积 $=\beta$ ⇒ $\lvert r_1\rvert=\lvert r_2\rvert=\sqrt\beta$，与 $\lambda$ 无关 ✓（「复根区内速率统一 $\sqrt\beta$」成立）。
3. 最优调参把 $[\mu,L]$ 两端压在复根区边界： $1+\beta-\eta\mu=2\sqrt\beta$、 $1+\beta-\eta L=-2\sqrt\beta$（式 (19) ✓）。相加： $2(1+\beta)=\eta(L+\mu)$ ⇒ $\eta=2(1+\beta)/(L+\mu)$ ✓；相减： $\eta(L-\mu)=4\sqrt\beta$ ✓。
4. 消 $\eta$： $\frac{L-\mu}{L+\mu}=\frac{4\sqrt\beta/\eta}{2(1+\beta)/\eta}=\frac{2\sqrt\beta}{1+\beta}$ ✓。记 $s=\sqrt\beta$： $(\kappa-1)(1+s^2)=2s(\kappa+1)$ ⇒ $s=\frac{(\kappa+1)-2\sqrt\kappa}{\kappa-1}=\frac{(\sqrt\kappa-1)^2}{(\sqrt\kappa-1)(\sqrt\kappa+1)}=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ ✓（取 $<1$ 根）。
5. 回代： $1+\beta=2(\kappa+1)/(\sqrt\kappa+1)^2$ ⇒ $\eta^\ast=\frac{2\cdot2(\kappa+1)/(\sqrt\kappa+1)^2}{\mu(\kappa+1)}=\frac{4}{(\sqrt L+\sqrt\mu)^2}$ ✓（式 (20) 三式全中）。
6. 数值表复算： $\kappa=10^2$： $\rho$ 0.98020/0.81818，步数 $\ln10^6/(-\ln\rho)$ = 691/69 ✓； $\kappa=10^4$： 0.99980/0.98020，69078/691 ✓——与章内表格逐位一致。

**判定：推导完整成立，与 Polyak 1964 经典结论一致；数值表全部复算命中。**

### 2.2(a) Adam bias correction（式 (3)(4)）

1. 递推展开： $m_t=(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i$ ✓（式 (3)）。
2. 权重和： $(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}=(1-\beta_1)\frac{1-\beta_1^t}{1-\beta_1}=1-\beta_1^t$ ✓。
3. 平稳假设 $\mathbb{E}[g_i]=\mathbb{E}[g]$ 下 $\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]$ ✓（式 (4)）； $v_t$ 同理 ✓。非平稳残差 $\zeta$ 的口径与 Adam 原文 §3（PDF 核对：「 $\mathbb{E}[v_t]=\mathbb{E}[g_t^2](1-\beta_2^t)+\zeta$，stationary 时 $\zeta=0$ 」）一致 ✓。
4. 量化数复算： $1-0.9^{t}$ 于 $t=1/10/50$ = 0.10/0.651/0.995 ✓； $1-0.999^t$ = 0.001/0.010/0.049， $1-0.999^{1000}=0.632$ ✓； $1-0.95^{50}=0.923$ ✓。未校正首步系数 $(1-\beta_1)/\sqrt{1-\beta_2}$： 0.9/0.999 → 3.162 ✓、0.9/0.95 → 0.4472 ✓。
5. **例外**：观察 (ii) 的「 $\hat m^2\le\hat v$ 逐坐标恒成立」为假（S1-1，反例见上）。同章 §2 的平稳态 SNR 版本（ $\hat m\to\mu,\hat v\to\mu^2+\sigma^2$ ）本身正确 ✓。

### 2.2(b) AdamW 解耦证明（式 (7)(8)(9) 与 Prop 1–3 对照）

1. SGD 等价（Prop 1）： $\theta_t=\theta_{t-1}-\alpha(\nabla f_t+\lambda'\theta_{t-1})=(1-\alpha\lambda')\theta_{t-1}-\alpha\nabla f_t$ ✓；与 decay 率 $\lambda$ 的解耦式恒等 ⇔ $\lambda=\alpha\lambda'$（ $\lambda'=\lambda/\alpha$）✓ 与原文 Prop 1 逐字一致。
2. Adam 下不等价的必要性（Prop 2 形式化）：L2 轨迹衰减项 $-\alpha M_t\lambda'\theta$、解耦轨迹 $-\lambda\theta$；逐步相同（ $\theta\ne0$ 步） ⇒ $\alpha M_t\lambda'=\lambda I$ ⇒ $M_t=(\lambda/(\alpha\lambda'))I$，与 $M_t\ne kI$ 矛盾 ✓——与原文 Prop 2 陈述（ar5iv 1711.05101 核对）一致；练习 2 排除 $\theta=0$ 退化步的提示正确 ✓。
3. 扭曲三层（逐坐标缩放/EMA 滞后/二阶矩交叉项 $2\lambda'\theta\odot g+\lambda'^2\theta^2$ ）与式 (8) 展开一致 ✓；数值例（ $\eta=3\times10^{-4}$、 $\lambda'=0.1$、 $\hat v_1=1,\hat v_2=10^{-6}$ ⇒ 相对衰减 $3\times10^{-5}$ vs $3\times10^{-2}$，1000 倍）复算 ✓。
4. 固定预条件子（Prop 3）：正则项 $\frac{\lambda}{2\alpha}\lVert\theta\odot\sqrt s\rVert^2$ 的梯度 $(\lambda/\alpha)s\odot\theta$ 穿过 $M=\mathrm{diag}(s)^{-1}$ 得 $-\lambda\theta$ ✓；与原文附录证明（系数 $\lambda'/2$、 $\lambda=\alpha\lambda'$ ）一致 ✓（正文排印的 $\lambda'/2\alpha$ 系数与附录证明不一致系原文排版问题，章内取的是与证明自洽的正确形式）。
5. 式 (9) 与原文 Algorithm 2 第 12 行逐字一致 ✓；normalized weight decay $\lambda=\lambda_{\mathrm{norm}}\sqrt{b/(BT)}$ 与原文 App. B.1 逐字一致 ✓。
6. PyTorch 走读（torch 2.8.0 源码核对）：`AdamW(Adam)` 置 `decoupled_weight_decay=True` ✓、 `__setstate__` 强制旗标 ✓、 `param.mul_(1-lr*wd)` / `grad.add(param, alpha=wd)` 分岔 ✓、 `exp_avg.lerp_` / `addcmul_` ✓、 `step_size=lr/bc1` 与 `denom=sqrt(v)/sqrt(bc2)+eps` 的代数恒等 ✓、amsgrad 的 `torch.maximum(...)` + max 进 denom ✓、capturable 折 step_size 进 denom 的源码注释 ✓、step 置 CPU 的注释（"kernel launches are costly on CUDA and XLA"）✓、AdamW docstring 衰减行先于矩更新、amsgrad「先 max 再校正」✓；Reddi 原文 Algorithm 2 不施加 bias correction ✓（章内差异注记准确）。
7. 对拍脚本本机复跑（torch 2.8.0）：fp32 25 步最大差 **1.1920929e-7**（章内「实测 1.2e-7」✓）、float64 6.9e-18 ✓。

**判定：解耦证明完整正确；源码走读与对拍数值全部命中。唯一断裂是 §3 观察 (ii) 的伪不变量（S1-1）。**

### 2.3(a) linear scaling rule 假设链（式 (5)）

Goyal 原文 §2.1 式 (3)(4) 与章内式 (5) 逐字对应 ✓；「若 $\nabla l(x,w_{t+j})\approx\nabla l(x,w_t)$ 则两轨迹相似**当且仅当** $\hat\eta=k\eta$ 」——原文 "the two updates are similar only if we set $\hat\eta=k\eta$" ✓（充要表述有原文支持）。失效边界两条（warmup 期参数快变； $k$ 不能无限大、ImageNet ~8k 后迅速退化）与原文逐句命中 ✓；Bottou 旁证两条（linear scaling 下按「已见样本数」同曲线；lr 不超过与 batch 无关的上限）与原文 §2.1 末段命中 ✓；gradual warmup「5 个 epoch 爬到 $\hat\eta$」✓。

### 2.3(b) critical batch size 推导与测量（式 (6)–(10)）

1. 二阶展开取期望： $\mathbb{E}[L(\theta-\epsilon G_{\mathrm{est}})]=L-\epsilon\lVert g\rVert^2+\frac{\epsilon^2}{2}(g^\top Hg+\mathrm{tr}(H\Sigma)/B)$ ✓（展开用到 $\mathbb{E}[G]=g$ 与 $\mathbb{E}[G^\top HG]=g^\top Hg+\mathrm{tr}(H\,\mathrm{Cov}\,G)$，正确）——与原文式 (2.5) 一致 ✓。
2. 对 $\epsilon$ 求导置零： $\epsilon_{\mathrm{opt}}=\frac{\lVert g\rVert^2}{g^\topHg+\mathrm{tr}(H\Sigma)/B}=\frac{\epsilon_{\max}}{1+\mathcal B_{\mathrm{noise}}/B}$、 $\epsilon_{\max}=\lVert g\rVert^2/(g^\topHg)$ ✓；回代得 $\Delta L_{\mathrm{opt}}=\frac{\lVert g\rVert^4/2}{g^\topHg+\mathrm{tr}(H\Sigma)/B}=\frac{\Delta L_{\max}}{1+\mathcal B_{\mathrm{noise}}/B}$ ✓——与原文式 (2.6)(2.7) 一致 ✓。
3. 式 (8) 两定义与原文式 (2.8)(2.9) 逐字一致 ✓；式 (9) 与原文式 (2.11)(2.12) 一致 ✓（「 $\mathcal B_{\mathrm{crit}}\approx\mathcal B_{\mathrm{noise}}$ 」原文同句 ✓）。
4. 式 (10) 独立重解： $E_B=\lVert g\rVert^2+S/B$ 两点联立 ⇒ $S=\frac{B_sB_\ell}{B_\ell-B_s}(E_{B_s}-E_{B_\ell})$、 $\lVert g\rVert^2=\frac{B_\ell E_{B_\ell}-B_sE_{B_s}}{B_\ell-B_s}$ ✓——与原文式 (A.2) 一致 ✓；相对误差 $=\mathcal B_{\mathrm{simple}}/B$ 与原文式 (2.10) 一致 ✓。
5. 数值/出处复核：温度 $T\approx\epsilon/B$（小 batch regime）与「noise scale ∝ 1/T」（App. C）✓；「critical batch 全程涨一个数量级以上」（§5.1 原文 ✓）；「同 loss 下与模型尺寸基本无关」（Figure 8 段原文 ✓）；§2.4 六条 caveat 照录核对全中 ✓；MiniCPM $bs=1.21\times10^9/L^{6.24}$（C4、0.009B/0.03B/0.17B 拟合，§3.2）✓；RAdam 式 (1)(2) 与原文式 (5)、Algorithm 2 逐字一致， $\rho_1=1$ 代数成立， $\beta_2=0.999/0.95$ 均在第 4、5 步间跨过 $\rho_t=4$（复算 $\rho_4=3.997/3.872$、 $\rho_5=4.996/4.795$）， $r_5=0.0173$、 $r_{10}=0.0490$（0.999）与 $r_{10}=0.3484$（0.95）复算全中 ✓。

**判定：2.3 的两项强制重推导全部成立；唯一问题是 2.1 的术语错位（S2-1）与「直接封顶」的外溢断言（S1-1 的另一半）。**

## Verification log（强制）

抽查覆盖率说明：三章内全部 arXiv ID（12/12）、全部论文级数值断言中抽取 38 项逐句回原文/源码（命中率：37 中 1 错——S2-2 节号）、PyTorch 行级走读 11 项全中、数值复算 31 项（30 中 1 错——S2-3 的 10N/2N）、强制重推导 5 组（4 组全过 + 2.2(a) 中 1 条伪不变量——S1-1）。

| # | 抽查项 | 来源（核法） | 结论 |
| --- | --- | --- | --- |
| 1 | 12 个 arXiv ID↔标题（1812.06162/1706.02677/1412.6980/1711.05101/1904.09237/2005.14165/2307.09288/2412.19437/2501.00656/2404.06395/1908.03265/2407.21783） | arXiv export API 逐 ID | 全中 ✓ |
| 2 | Llama-3 405B 日程（8e-5、warmup 8000、cosine→8e-7、1.2M 步、batch 4M→8M@252M tokens→16M@2.87T、动机句） | ar5iv 2407.21783 §3.4.1 逐字 | 全中 ✓（**「252M tokens」为原文逐字**；writer 存疑标注处置正确） |
| 3 | herd 论文未披露 optimizer $\beta/\epsilon$ | ar5iv 全文检索 AdamW/β | ✓ 未披露（纪律项通过，2.2 表标注正确） |
| 4 | herd「wd = 0.1×lr」节号定位 | ar5iv 全文检索 + 节标题定位 | **§3.2.1，非章内所写 §4.2.1**（S2-2） |
| 5 | GPT-3 App B（ $\beta_2$=0.95、 $\epsilon$=1e-8、wd 0.1、clip 1.0、cosine 260B 到 10% 后恒定、warmup 375M tokens、batch 32k→全值/前 4–12B tokens） | ar5iv 2005.14165 逐字 | 全中 ✓ |
| 6 | GPT-3 Table 2.1（125M:0.5M/6e-4 → 175B:3.2M/0.6e-4） | 同上，表格抽取 | ✓（2.3 §6 与练习 4 数值中） |
| 7 | Llama-2 §2（AdamW 0.9/0.95/1e-5、wd 0.1、clip 1.0、cosine 到 10% peak、warmup 2000） | ar5iv 2307.09288 逐字 | 全中 ✓ |
| 8 | DSv3 §4.2（0.9/0.95/wd 0.1/clip 1.0、 $\epsilon$ 未披露、2K warmup→2.2e-4→10T 恒定→4.3T cosine→2.2e-5→500B 两段常数 333B+167B、batch 3072→15360 序列/前 469B） | ar5iv 2412.19437 逐字 | 全中 ✓；12.6M/62.9M tokens 折算 ✓；469B/14.8T≈3% ✓ |
| 9 | DSv3 §3.3.3（一、二阶矩 BF16、master+梯度 fp32、「无可观测性能下降」） | 同上逐字 | 内容 ✓；**章内 $(4+2c)N$ 代入数值错**（S2-3） |
| 10 | OLMo 2 §3.4.1（ $\epsilon$ 1e-5→1e-8、梯度范数更快更低）、§3.4.2（embedding no-wd、LN Jacobian ∝1/‖x‖）、§3.5（warmup 2000→3e-4、cosine 5T 截 4T、30e-4 spike 放弃、3e-4 vs 6e-4 交叉 >200B、50B/100B 线性收尾等效） | ar5iv 2501.00656 逐字 | 全中 ✓ |
| 11 | OLMo-core `src/scripts/train/OLMo2/OLMo2-7B.py`（betas (0.9,0.95)、wd 0.1、`OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))`、warmup 2000、clip 1.0、batch 1024×4096≈4M tokens） | GitHub contents API + raw 逐行 | 全中 ✓ |
| 12 | McCandlish 式 (2.2)(2.5)(2.6)(2.7)(2.8)(2.9)(2.10)(2.11)(2.12)、App A.1/A.2 两 batch 测量、App C 温度 $T\approx\epsilon/B$ 与噪声尺度 ∝1/T、§2.4 六条 caveat、「漂移一个数量级以上」、「同 loss 与模型尺寸基本无关」、「 $\mathcal B_{\mathrm{simple}}$ 与 $\mathcal B_{\mathrm{noise}}$ 差小常数倍」 | ar5iv 1812.06162 逐式 | 全中 ✓；**2.1 把 $\mathcal B_{\mathrm{simple}}$ 误名为 $\mathcal B_{\mathrm{noise}}$**（S2-1） |
| 13 | Goyal §2.1 假设链（式 (3)(4)）、§3 momentum correction（式 (9)(10)、 $\eta_{t+1}/\eta_t$ 因子、warmup 期失稳）、Nesterov m=0.9、wd 1e-4、 $\eta=0.1\cdot B/256$、8k 退化边界、Bottou 两点 | ar5iv 1706.02677 逐句 | 全中 ✓ |
| 14 | Reddi 反例（ $f_t=Cx$ @ $t\bmod3=1$、 $C>2$、 $\beta_1=0,\beta_2=1/(1+C^2)$、 $x^\ast=-1$、Adam→+1 最坏点、 $\Gamma_t$ 不定号）、Theorem 1–4、AMSGrad max 修正、 $O(1/\sqrt T)$、「大 $\beta_2$ 有缓解」注记、Algorithm 2 无 bias correction | ar5iv 1904.09237 逐句 + $C=3$ 三步循环独立复算（8.1/1.71/1.07、净 +0.68α、AMSGrad 净 −0.35α ✓） | 全中 ✓；**Theorem 2 误标「随机」**（S3-1） |
| 15 | AdamW 原文 Prop 1/2/3、Algorithm 2 第 12 行、normalized wd（App B.1）、CIFAR-10/ImageNet32x32 实验（§4–5） | ar5iv 1711.05101 逐式 | 全中 ✓ |
| 16 | Adam 原文默认（1e-3/0.9/0.999/1e-8「tested problems」口径）、bias correction 的 $\zeta$ 残差、**原文步长口径仅为近似有界 $\|\Delta\|\lessapprox\alpha$（稀疏最坏 $(1-\beta_1)/\sqrt{1-\beta_2}$ 倍）** | arXiv PDF（v9）逐句 | 默认/残差 ✓；**章内「严格 <η」与原文口径冲突且可构造反例**（S1-1） |
| 17 | Sutskever 2013（CM/NAG 式号、 $\mu$ 递增 schedule 公式、 $\mu_{\max}\in\{0.999,0.995,0.99,0.9,0\}$、RNN 前 1000 步 0.9 后切 $\{0,0.9,0.98,0.995\}$） | PMLR v28 sutskever13 PDF 逐字 | 内容全中 ✓；**章内「式 (2) 同处」指针错（应为式 (3)–(4)）**（S3-2） |
| 18 | MiniCPM（WSD 式 (1) 分段、指数退火 5000 步/20B tokens、10% 够/2.5% 不足、checkpoint 复用追平更长 cosine、Hoffmann $T\lessgtr S$ 复现于 0.036B、 $bs=1.21\times10^9/L^{6.24}$ @0.009B–0.17B C4 §3.2） | ar5iv 2404.06395 逐式 | 全中 ✓ |
| 19 | RAdam（方差近似式 (5)、 $\rho_\infty=2/(1-\beta_2)-1$、 $\rho_t$ 式、 $\rho_t>4$ 判据、 $r_t$ 式、「 $\rho=5$ 比 500 大两个数量级」、warmup=方差缩减） | ar5iv 1908.03265 逐式 + 数值复算（ $\rho_t$/$r_t$ 两 $\beta_2$ × $t=1..10$） | 全中 ✓ |
| 20 | torch/optim/sgd.py（`buf.mul_.add_(alpha=1-dampening)`、首步 clone、wd 进梯度、nesterov 重排、docstring Sutskever 差异注、示例 lr=0.1 momentum=0.9）；adam.py/adamw.py 逐行（见重推导 2.2(b)-6）；optimizer.py 存在 | GitHub contents API + raw 源码（torch 2.8.0） | 全中 ✓ |
| 21 | 2.2 §9 对拍脚本可复现性 | 本机执行（torch 2.8.0）：fp32 max diff 1.1920929e-7（章内 1.2e-7 ✓）、float64 6.9e-18 ✓、assert 通过 | ✓ |
| 22 | 2.1 全部数值读数（1/(1-β²)=5.26/10.26/50、(1-β)/(1+β)=1/19/1/39/1/199、滞后 9/19/99、90% 覆盖 22/45/229、二次碗速率表 4+4 格） | python 复算 | 全中 ✓ |
| 23 | models 2.7 §5 跨章引用（Post-LN 末层 $O(d\sqrt{\ln d})$ 与 $L$ 无关、逐层衰减、warmup 近似逐层 lr、Pre-LN $1/\sqrt L$） | models/02-transformer-anatomy/07-norm-and-residual.md §5 核对 | ✓（2.3 §2 转述准确） |
| 24 | models 1.2 式 (8) $(4+2c)N$ 存在性 | models/01-foundations/02-compute-accounting.md:188 | 公式存在 ✓；2.2 代入数值错（S2-3） |
| 25 | Polyak 1964 书目（USSR Comp. Math. & Math. Phys. 4(5):1-17，无 arXiv） | 标准书目常识 + 结论独立重推一致 | ✓ |
| 26 | 篇幅（CJK 汉字，去公式/代码）：2.1=3523、2.2=5531、2.3=3541 | 本机统计 | 达标 ✓（2.2 过推导重章 5500 软目标） |

## 机械检查

```
bash scripts/lint-chapter.sh training/02-optimizers/01-sgd-and-momentum.md → LINT OK（exit 0）
bash scripts/lint-chapter.sh training/02-optimizers/02-adam-and-adamw.md   → LINT OK（exit 0）
bash scripts/lint-chapter.sh training/02-optimizers/03-lr-schedules.md     → LINT OK（exit 0）
```

## 教学与规范审查摘要

- **教学**：除 S1-1 外无推导断步；2.1 §5 式 (19)→(20) 的二次方程求解留给练习 2(b) 属合理分工；符号表逐式在位；前置引用（1.5、models 1.1/1.2/2.7）与实际依赖吻合； $\mu$ 在 2.1 §3–4（动量系数）与 §5（最小特征值）双用但均局部定义、分属两文献传统，不记。
- **规范**：rubric A 类零命中（全词表扫描）；无历史叙事（Polyak/Sutskever 仅作出处）；每节以事实/公式/问题开场；总结均为表格；练习 2–4 题齐；2.3 §5「4–12B/260B+≈2%–5%」的取整有 ≈ 掩护，不记。

## 修复复审记录（2026-11-17 复审轮）

复审方式：三章当前全文重读（非 diff），逐条对照原 finding；修复触及处做新问题检查；lint 重跑。

### 逐条判定

| # | 判定 | 当前文本证据 |
| --- | --- | --- |
| S1-1 | **ADDRESSED** | 2.2 §3 (ii) 现为双口径：平稳态「Jensen 不等式给出 $\hat m^2\le\hat v$（即 §2 的 SNR 口径）」数学成立；「非平稳的瞬时步可以超过 $\eta$，因为 $\hat m$ 与 $\hat v$ 用两组不同权重，Cauchy–Schwarz 不适用」——算例数值 $\hat m\approx0.65$、 $\hat v\approx0.10$、位移 $\approx2.0\eta$ 与本 evaluator 反例复算（0.6513/0.1046/2.014）一致； $\eta(1-\beta_1)/\sqrt{1-\beta_2}=3.16\eta$ 以「原文（§2.1）自己的口径」归因（与 1412.6980 §2.1 原文一致），结论「 $\eta$ 的语义是典型步长量级而非严格上界」正确。2.2 §2 同步为「典型量级（严格上界见 §3 末）」；2.3 §4 同步为「近似封顶——平稳态渐近不超过 $\eta$、非平稳情形的最坏上界为 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$（2.2 §3）」。**残留一处过量化见下方新增 S3-7** |
| S2-1 | **ADDRESSED** | 2.1 §1 现写「本章取简化形式 $\mathcal B_{\mathrm{simple}}=\mathrm{tr}\Sigma/\lVert g\rVert^2$。McCandlish 原文另定义带 Hessian 的 $\mathcal B_{\mathrm{noise}}=\mathrm{tr}(H\Sigma)/(g^\topHg)$，两者差一个小常数倍（其式 (2.8)/(2.9)）；本章的 L-光滑口径对应的正是 simple 版，完整区分归 2.3」；式 (7)、§2 两条 regime bullet、总结表全部改用 $\mathcal B_{\mathrm{simple}}$；全章 grep 无残留 $\mathcal B_{\mathrm{noise}}$ 符号 |
| S2-2 | **ADDRESSED** | 2.2 §6 与参考区 llama3 条目两处均已改「§3.2.1」 |
| S2-3 | **ADDRESSED** | 现文「 $(4+2c)N$ 从 $12N$ 降到 $8N$ bytes；671B 模型这一项省 $4N$ bytes 量级（约 2.7 TB，SI 口径）」——数值与单位均正确。**writer 对原建议文案的反驳成立**：本报告 S2-3 建议修正中「约 2.7 GB 级」单位有误（ $671\times10^9\times4\ \mathrm{B}=2.68\ \mathrm{TB}$，非 GB）；落文 2.7 TB 正确，记 evaluator 建议文案失误一条 |
| S3-1 | **ADDRESSED** | 2.2 §5 现为「…非零的在线问题（Theorem 2）。随机版（Theorem 3）…」 |
| S3-2 | **ADDRESSED** | 2.1 §4 现为「其式 (3)–(4)」 |
| S3-3 | **ADDRESSED** | 2.3 §5 现为「all-reduce 前的各 rank 小 batch 局部梯度与 all-reduce 后的大 batch 全量梯度各提供一次范数测量，零额外通信」——与 App. A.1 机制一致 |
| S3-4 | **ADDRESSED** | 2.2 §6 新增「一处单位换算：两边的衰减因子分别为 $\eta_t\lambda$ 与 $\mathrm{lr}_t\cdot\mathrm{wd}$，由 $\mathrm{lr}_t=\eta_t\alpha$ 折合即得 torch 的 `weight_decay` 对应原文的 $\lambda/\alpha$」——换算正确 |
| S3-5 | **ADDRESSED**（前序修复覆盖，复审核验） | references.md：deepseekv3 被引章节现含 2.2, 2.3；olmo2 现含 2.2 |
| S3-6 | **ADDRESSED** | 2.2 §7 末新增「本节这张表只覆盖 optimizer 本体；lr / bsz / warmup / decay / wd / $\beta_2$ / clip 的全字段真实配方对照表归 7.2」 |

### 修复引入的新问题（只查修复触及处）

- **S3-7（新增，记录级）**：2.3 §4 同步句「非平稳情形的最坏上界为 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$」把 Adam 原文的稀疏情形数字写成了普适最坏上界。复审复核：取 Cauchy–Schwarz 最优序列 $g_i\propto(\beta_1/\beta_2)^{t-i}$，真实上确界 $\sqrt{(1-\beta_1)^2/\left((1-\beta_2)(1-\beta_1^2/\beta_2)\right)}$ 随 $t$ 增长—— $\beta_1=0.9,\beta_2=0.999$ 时 $t=100/200/2000/10000$ 处比值达 2.24/3.10/6.76/7.27（数值验证，与 C-S sup 逐位一致）， $t\gtrsim$ 数百即超过 3.16η。2.2 §3 的同数字以「原文口径」归因、不犯此错；仅 2.3 §4 的表述越界。建议改为「非平稳瞬时步可超 $\eta$ 一个 O(1) 因子（原文稀疏口径 $\eta(1-\beta_1)/\sqrt{1-\beta_2}$=3.16η，2.2 §3 算例 2.0η）」。该句处在显式标注「本书分析/经验规则」的启发式段落内，主结论（近似封顶、 $\eta$ 随 $\sqrt B$ 增长）不受影响，定 S3。
- 其余修复触及处（2.1 §1 噪声尺度段、2.2 §3 (ii) 全段、2.2 §6 单位换算句、2.2 §7 末句、2.3 §5 测量句）复查无新问题；2.1 改写句与原文式 (2.8)/(2.9) 的对应复核命中。

### lint 重跑

```
bash scripts/lint-chapter.sh training/02-optimizers/01-sgd-and-momentum.md → LINT OK（exit 0）
bash scripts/lint-chapter.sh training/02-optimizers/02-adam-and-adamw.md   → LINT OK（exit 0）
bash scripts/lint-chapter.sh training/02-optimizers/03-lr-schedules.md     → LINT OK（exit 0）
```

### 复审结论

**S1/S2 全部关闭（4/4）；S3-1~6 全部关闭；新增记录级 S3-7 一项。放行 2.1–2.3**（S3-7 随下波清零统一处置）。
