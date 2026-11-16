# Training W1 审查报告 · 批 3（2.4–2.5 逐章 + 全波 10 章跨章一致性 sweep）

- 日期：2026-11-17
- 审查人：独立 fresh-context evaluator（批 3，与写作方及批 1/批 2 evaluator 无共享上下文）
- 对象：`training/02-optimizers/04-gradient-clipping-and-stability.md`、`05-muon-and-spectral.md` 逐字审查；`training/01-autograd-and-backprop/` + `training/02-optimizers/` 全 10 章 sweep
- 输入清单全部读取：两章全文、其余 8 章全文、STYLE_GUIDE.md、rubric.md、training/references.md、training spec（§4.3/§5/§6.3-6.4）

## 结论

**S1 = 0；S2 = 5；S3 = 10。不放行**：5 项 S2 全部修复后复审（均为局部改写，预计一轮关闭）。S1 为零——无张冠李戴、无虚构来源、无错误公式骨架；两章的主体事实核查通过率极高，2.5 的三组强制重推导独立复核成立（例外见 S2-3/S2-4 两个数值口径）。

**写作方存疑项的裁定（Moonlight 式 (4) vs 官方代码的 wd 口径）**：代码与论文式 (4) **逐项等价，不存在差异**——式 (4) 中 $0.2\sqrt{\max(A,B)}$ 只作用于 $O_t$ 项，$\eta_t$ 同时乘两项；代码 `p.mul_(1 - lr*wd)`（wd 项乘未调整 lr = $\eta_t\lambda W$）+ `p.add_(u, alpha=-adjusted_lr)`（更新项乘 $0.2\sqrt{\max}\cdot\eta_t$）恰好逐项对应。正文「其二」断言的差异是幻影（误把 $0.2\sqrt{\max}$ 当成应作用于 wd 项），且「本书以代码为准登记此差异」会诱导复算的读者把 wd 项错误地乘上 $0.2\sqrt{\max}$——记 S2-1，须改写。

## Findings 表

### S2（5 项，必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 2.5 §4 第三段注记「其二」+ 参考区 moonlight 条目（「wd 的 lr 口径差异」） | 「weight decay 用未调整的 lr（`p.mul_(1 - lr * wd)`）而更新用 `adjusted_lr`——wd 项不被 $0.2\sqrt{\max}$ 放大，与式 (12) 把 $\eta_t$ 乘在两项上的字面写法不同」 | 论文式 (4) 原文 $\mathbf{W}_t=\mathbf{W}_{t-1}-\eta_t(0.2\cdot\mathbf{O}_t\cdot\sqrt{\max(A,B)}+\lambda\mathbf{W}_{t-1})$（ar5iv 2502.16982）；代码（MoonshotAI/Moonlight `examples/toy_train.py`，master sha 8dbde98，L142-148/L200/L203）`adjusted_lr = lr*0.2*sqrt(max(A,B))`、`p.data.mul_(1-lr*wd)`、`p.data.add_(u, alpha=-adjusted_lr)`——逐项展开 = $W-\eta_t\lambda W-\eta_t\cdot0.2\sqrt{\max}\cdot O$，与式 (4) 恒等。KellerJordan/muon.py 同结构（wd 用 `lr`、缩放因子乘在 update 上） | 改写为：代码把式 (12) 拆成「wd 项乘未调整 lr、更新项乘 adjusted_lr」两行执行，与式 (12) 逐项恒等——$0.2\sqrt{\max}$ 只属于 $O$ 项，wd 项的 $\eta_t$ 就是未调整的 lr；参考区条目同步删「wd 的 lr 口径差异」 |
| S2-2 | 2.5 §9 第 6 条 + §9 末段 | 「本书锚点（Moonlight / K2）覆盖了预训练与 SFT 的可行性陈述，RL 阶段用 Muon 的公开证据阙如」；「第三个（微调与 RL）在本书截稿的锚点文献内仍开放」 | K2 论文（ar5iv 2507.20534）§3.1 原文：「We employ the Muon optimizer in our post-training and recommend its use for fine-tuning with K2」；§3.2.3（RL Algorithm，K1.5 式目标）原文：「As in SFT, we employ the Muon optimizer to minimize this objective」——K2 在 1T 规模的 SFT 与 RL 都用 Muon，公开证据存在 | 改为：K2 明言 SFT 与 RL 均用 Muon（§3.1/§3.2.3），预训练之外已有 1T 级公开实证；仍开放的口径收窄为「缺 AdamW 对照消融与独立复现」。§9 末段第三个 open question 的「仍开放」同步修正 |
| S2-3 | 2.5 §3 数值复核 bullet 2 + 总结表 NS 行 | 「5 步把 $(0, 1]$ 的输入压进约 $[0.47, 1.14]$」 | 本 evaluator 独立复算（numpy，2×10⁶ 点网格 + 解析复核）：5 步像在 $[10^{-3},1]$ 上为 $[0.4705, 1.2024]$，上界 1.2024 在 $\sigma_0\approx0.5545$（一步即达）与 $\sigma_0\approx0.0041$（五步路径）两处取得；$\sigma_0<10^{-3}$ 时像低于 0.47（本章下一条 bullet 自己承认小端抬不进 $[0.5,1.5]$，两句相邻表述自相矛盾） | 改为「5 步把 $\gtrsim10^{-3}$ 的输入压进约 $[0.47, 1.20]$」；总结表同步 |
| S2-4 | 2.5 §3 系数的故事段 | 经典系数 $(2,-1.5,0.5)$「$\sigma_0 = 0.001$ 时要约 20 步」 | 独立复算轨迹：0.001→0.032（5 步，与表中 0.0320 一致）→0.807（10 步）→0.997（11 步）→0.9999（12 步）；任何合理收敛口径下都是约 11–12 步，不是 20 | 改为「要约 12 步」或「十余步（约为 Muon 系数的 2 倍步数）」 |
| S2-5 | 2.4 §1 首段 + §3 出处链表 GPT-3 行（两处） | GPT-3 引文「we clip the global norm of the gradient at 1.0」标注出处为「§2.1」 | ar5iv 2005.14165 全文检索：该句唯一位于 **Appendix B**（Details of Model Training：「we use Adam with $\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-8}$, we clip the global norm of the gradient at 1.0, and we use cosine decay…」），§2.1 无此句；且本书 2.2 §7 表与 2.3 §1 对同一论文均正确标注 App. B——2.4 自相矛盾 | 两处「§2.1」改「App. B」 |

### S3（10 项，记录；下波清零时逐条判定）

| # | 位置 | 断言 | 证据 / 处置建议 |
| --- | --- | --- | --- |
| S3-1 | 2.5 §3 bullet 3 + §9 第 4 条 | 「$\sigma_0 \ge 1.28$ 时轨道发散（$\sigma_0 = 1.25$ 仍有界）」 | 复算发散阈值 ≈ 1.2637（1.2637 有界、1.264 发散，5000 步轨道判定）。陈述方向不错但不紧致；建议改「$\sigma_0 \gtrsim 1.26$」 |
| S3-2 | 2.5 §3 系数的故事段 | 经典系数「从任意 $\sigma_0 \in (0, 1]$ 单调收敛到 1」 | $f(\sigma)-\sigma = 0.5\sigma(1-\sigma^2)(2-\sigma^2)$ 在 $\sigma\approx0.949$ 处 $f$ 峰值 ≈1.0009——可微幅过冲 1 再回落（本章表格经典列 1.0005 已自证非单调）。建议删「单调」或加注 |
| S3-3 | 2.5 §6 分布式段 | Distributed Muon「在 ZeRO-1 之上增加两次集合操作」 | 论文 §2.3 原文为 "two additional operations"：1 次 DP gather（集合通信）+ 1 次全矩阵 NS（计算，非通信）。建议改「两项新增操作（一次集合通信 + 一次全矩阵计算）」 |
| S3-4 | 2.5 §4 第四段 | 「于是式 (12) 把 AdamW 式解耦 wd 并入」 | 论文中 wd 在式 (3) 已进入（$W_t=W_{t-1}-\eta_t(O_t+\lambda W_{t-1})$），式 (4) 是再叠尺度匹配。建议「于是论文式 (3) 先把解耦 wd 并入、式 (4) 再叠尺度匹配」 |
| S3-5 | 2.5 §4 第一段 | 「论文式 (1)」标签 | 论文印刷版式 (1) 为 $O_t=\mathrm{NewtonSchulz}(M_t)$，Nesterov 输入 $\mu M_t+G_t$ 在脚注 1（"In practice, we follow (Jordan et al. 2024)…"）。正文已披露脚注来源，但式号标签建议写「论文式 (1) + 脚注 1 的 Nesterov 变体」以免对不上原文 |
| S3-6 | 2.4 §5 首句 | 「GLM-130B 是公开文献里稳定性记录最完整的一次百亿级训练」 | GLM-130B 为 130B = $1.3\times10^{11}$，中文量级为「千亿级」；「百亿级」差一个数量级。建议改「百亿-千亿量级」或「130B 级」 |
| S3-7 | 2.2 §7 | 缺 7.2 前瞻指针 | 2.1 §7 与 2.3 §6 均有「全字段对照表归 7.2」；2.2 §7 表正是该表的 optimizer 列却未指。建议 §7 末加一句指针（spec 7.2 大纲含 $\beta_2$/wd/clip 全字段） |
| S3-8 | training/references.md | 被引章节回写遗漏 | deepseekv3 缺 2.2/2.3/2.4；olmo2 缺 2.2/2.4；gpt3 缺 2.4；llama3 缺 2.2/2.5；adamw 缺 2.4；pytorch-repo 缺 1.3/1.5（逐章正文核对） |
| S3-9 | 1.2 §9 | $N$ 局部复用为序列长（FA 论文记号） | 已显式声明「仅本节……与本章其余各节的 $M$ 不相混」，与 spec §4.3 的 $N$=参数量约定不冲突——建议判定「接受」并登记 |
| S3-10 | 2.5 §3 vs §4 | $B$ 双用：§3 的 $B$=流经该层 token 数；§4 引理 1 的 $[A,B]$=矩阵形状（Moonlight 原文记号） | 各自局部定义、无公式混用；§4 为逐式回原文所需——建议判定「接受」并登记 |

## 强制独立重推导（2.5，中间过程全录）

### (a) 谱范数约束最速下降 = msign（式 (2)–(5)）

目标：$\min_\Delta \langle G,\Delta\rangle_F$ s.t. $\lVert\Delta\rVert_2\le\eta$，$G$ 满秩。

1. 取完整 SVD $G=U\Sigma V^\top$（$U$ $n\times n$、$V$ $m\times m$ 正交），令 $D=U^\top\Delta V$。谱范数酉不变：$\Vert D\Vert_2=\Vert\Delta\rVert_2\le\eta$ ✓。
2. 目标改写：$\langle G,\Delta\rangle_F=\mathrm{tr}(G^\top\Delta)=\mathrm{tr}(V\Sigma^\top U^\top\Delta)=\mathrm{tr}(\Sigma^\top U^\top\Delta V)=\mathrm{tr}(\Sigma^\top D)=\sum_i\sigma_iD_{ii}$（迹循环）✓，非对角元不进目标 ✓。
3. 对角元界：$\vert D_{ii}\vert=\vert e_i^\top De_i\vert\le\lVert D\rVert_2\le\eta$（谱范数 $=\sup_{\lVert u\rVert=\lVert v\rVert=1}u^\top Dv$ 的直接推论）✓。
4. 下界：$\sum_i\sigma_iD_{ii}\ge-\sum_i\sigma_i\vert D_{ii}\vert\ge-\eta\sum_i\sigma_i=-\eta\lVert G\rVert_\ast$ ✓——对偶范数配对（谱 ↔ 核）✓。
5. 取等构造 $D=-\eta I$（方阵满秩）⇒ $\Delta^\ast=-\eta UV^\top$ ✓；矩形满秩（$n>m$、$r=m$）时 $UV^\top=U_{[:,1:m]}V^\top$ 自动截断，$\lVert\Delta^\ast\rVert_2=\eta$、目标 $=-\eta\sum\sigma_i$ 同达成 ✓。
6. 唯一性：取等要求全部 $D_{ii}=-\eta$；$\vert D_{ii}\vert=\lVert D\rVert_2$ 蕴涵第 $i$ 列全部质量压在第 $i$ 分量（$\vert D_{ii}\vert\le\lVert De_i\rVert\le\lVert D\rVert_2$ 两端挤紧）⇒ 非对角元全零 ⇒ $D=-\eta I$ 唯一 ✓。
7. msign 三形式：$GG^\top=U\Sigma^2U^\top$ ⇒ $(GG^\top)^{-1/2}G=U\Sigma^{-1}U^\top U\Sigma V^\top=UV^\top$ ✓；另一式对称 ✓。极分解读法（离 $G$ 最近的半正交矩阵）为标准结论 ✓。
8. Frobenius 对照：Cauchy-Schwarz 给 $\Delta^\ast=-\eta G/\lVert G\rVert_F$、下降量 $\eta\lVert G\rVert_F$ vs 谱版 $\eta\lVert G\rVert_\ast$ ✓；§2 对照表两行数值正确 ✓。

**判定：证明完整、无断步、无循环；章内「满秩先证、秩亏留练习」的边界声明与实际论证范围一致。练习 1 的解集陈述（$D_{ii}=-\eta$ 对 $i\le r$、其余自由且 $\lVert D\rVert_2\le\eta$；$-\eta U_rV_r^\top$ 为最小 Frobenius 元）经独立验证正确。**

### (b) Newton-Schulz 五次迭代：系数与收缩动力学（式 (7)–(9) 与数值复核表）

1. 结构引理：$X_k=U\mathrm{diag}(\sigma^{(k)})V^\top$ ⇒ $X_kX_k^\top=U\mathrm{diag}(\sigma^2)U^\top$ ⇒ 式 (7) 右端 $=U\mathrm{diag}(a\sigma+b\sigma^3+c\sigma^5)V^\top$，即式 (8) 的逐奇异值标量 map $f$ ✓；奇异向量严格不动 ✓。
2. 系数 $(3.4445,-4.7750,2.0315)$ 与 KellerJordan/muon.py `zeropower_via_newtonschulz5`、Moonlight toy_train.py、博客三者逐字一致 ✓；经典值 $(2,-1.5,0.5)$ 为博客 baseline ✓。
3. 独立数值复核（numpy 标量迭代 + 2×10⁶ 点网格 + 5000 步轨道判定；矩阵级 PyTorch 复跑）：
   - 章内表格 20 个数值（5 个 $\sigma_0$ × 1/3/5 步 × 两组系数）**全部逐位命中** ✓；
   - $f(1)=0.701\ne1$ ✓；长程 4-环复算 $\lbrace0.6819,0.7529,1.0469,1.1343\rbrace$，与章内 $\lbrace0.68,1.13,0.75,1.05\rbrace$ 一致 ✓；
   - $a^5=484.876\approx485$ ✓；小端带宽 $\sigma_0\gtrsim10^{-3}$（$485\sigma_0\approx0.5$）自洽 ✓；
   - **5 步像的上界实为 ≈1.2024 而非 1.14**（S2-3）；$(0,1]$ 全域下界并非 0.47（S2-3 同案）；
   - 发散阈值复算 ≈1.2637，章内「$\ge1.28$」方向对但不紧致（S3-1）；
   - 经典系数 $f(1)=1$、$f'(1)=0$ ✓；$\sigma_0=0.001$ 收敛实为约 11–12 步（S2-4）；「单调」不严（S3-2）。
4. 归一化前提：$\lVert G\rVert_F\ge\lVert G\rVert_2$ ⇒ 除以 Frobenius 范数保证奇异值 $\le1$ ✓；「归一化是正确性前提而非精度技巧」与发散区复核一致 ✓（发散阈值 >1 进一步坐实）。
5. 跨步不累积论证（每步独立调用 + 首步归一化洗白 + 奇异向量不动）成立 ✓。
6. 开销账：每步 $4m^2n+2m^3\le6nm^2$（$m\le n$）✓ 与博客 $2(2nm^2+m^3)$ 一致；总比 $\le Tm/B$ ✓；NanoGPT $5\cdot768/524288=0.0073$ ✓、405B 规模 $5\cdot16384/1.6\times10^7=0.0051$ ✓（两个数字均为博客原文口径，已回原文核对）。

### (c) 尺度匹配因子（引理 1 + 式 (12)）

1. 引理 1 补全证明独立复核：$O=U_rV_r^\top$、列正交 ⇒ $\mathrm{RMS}(O)^2=\frac{1}{mn}\sum_{k,k'}\delta_{kk'}\delta_{kk'}=\frac{r}{mn}$（交叉项消失的两步 $\sum_iU_{ik}U_{ik'}=\delta_{kk'}$、$\sum_jV_{jk}V_{jk'}=\delta_{kk'}$ 正确）；满秩 $r=m$ ⇒ $\mathrm{RMS}=\sqrt{1/n}=\sqrt{1/\max(A,B)}$ ✓。numpy 实例（$[768,3072]$ 高斯矩阵）：数值 0.01804219591217581 vs $\sqrt{1/3072}$=0.018042195912175804，15 位相合 ✓。
2. 后果两 bullet（宽 MLP $[H,4H]$ 多压 $\sqrt4=2$ 倍；GQA/MLA 逐 KV-head 拆小矩阵 $\max(A,B)$ 过小 ⇒ 更新过大）与论文 §2.2 两 bullet 逐条对上 ✓。
3. AdamW 理论 RMS≈1 + 脚注 2（$\beta_1<\beta_2$、$\epsilon>0$ 压到 1 以下）✓；经验区间 0.2–0.4 ✓；附录 A 扫描 $\lbrace0.05,0.1,0.2,0.4,0.8\rbrace$ 取 0.2 ✓（Table 8 原文 "0.2 RMS and 0.4 RMS performed similarly and much better…opted to control…to 0.2"）。
4. 式 (12) 与论文式 (4) 逐字一致 ✓；**代码逐项等价于式 (4)，正文「其二」的差异断言不成立（S2-1）**；其三（Muon 原始 repo $\sqrt{\max(1,A/B)}$ vs 论文 $\sqrt{\max(A,B)}$ + 脚注 3）✓ 与 muon.py L40 `update *= max(1, update.size(-2)/update.size(-1))**0.5` 及论文脚注 3 双双命中。

## Verification log（强制）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| 2.5 式 (10) momentum/Nesterov（论文式 (1)+脚注 1 口径） | ar5iv 2502.16982 §2.1 式 (1) 与 footnote 1（"In practice, we follow (Jordan et al. 2024) to use a Nesterov-style momentum by putting $\mu M_t+\nabla\mathcal{L}_t$ to the Newton-Schulz iteration"） | 成立（式号标签不精确，S3-5） |
| 2.5 式 (12) = 论文式 (4)；0.2；$\sqrt{\max(A,B)}$；$\lambda$ | 同上 §2.2 式 (4) 原文 | 成立 |
| **代码 wd 口径（writer 存疑项）** | MoonshotAI/Moonlight `examples/toy_train.py` master sha 8dbde98（L142-148/L189-203）+ 论文式 (4) | **代码 ≡ 式 (4)，正文差异断言为幻影 → S2-1** |
| Moonlight toy_train.py：EMA 注记、momentum 形式、NS 系数、bf16、转置、docstring 两警告、AdamW 分支 betas/eps、参数二分 | 同文件全文 + KellerJordan/muon.py L34-41（`momentum.lerp_(grad,1-beta)`、`grad.lerp_(momentum,beta)`） | 全部成立；EMA 等价换算 $G_t+\mu M_t=(1-\mu)^{-1}(\mu\tilde M_t+(1-\mu)G_t)$ 代数独立验证成立 |
| 引理 1 与附录 A；RMS 扫描；$\mu=0.95$、$N=5$（§2.2 "N=10 更准但不更好"）；bf16 RMS 增长动机；800M/100B（∼5×）wd 实验 | ar5iv 2502.16982 §2.1-2.2、附录 A/B | 成立（wd 首次出现于式 (3)，S3-4） |
| 约 2× claim：摘要 "~2× computational efficiency…compute optimal"；§3.2 "approximately 52% of the training FLOPs"；Llama 架构 dense 系列；附录 B 网格搜索（N-D 对先定、lr/bsz 后搜、迭代 2-3 轮）；图 3/表 3 | 同上 摘要/§3.2/附录 B 原文逐句 | 成立（$1/0.52\approx1.92$ 复算 ✓） |
| Moonlight 规模：2.24B 激活/15.29B 总（3B/16B 含 embedding）、deepseek-v3-small、5.7T token、表 4/5、图 1b Pareto | 同上 §3.3 与摘要 | 成立 |
| 附录 D：max attention logit 破 100、比例 ≈10⁻⁴、自行回落、AdamW 更健康、RMSNorm gamma wd「crucial」；附录 F/§3.4：1.2T checkpoint、>90% 矩阵 SVD 熵更高 | 同上附录 D/F 原文 | 成立 |
| Distributed Muon：ZeRO-1 冲突、两项新增操作、内存最优且数学不变、通信比 $(1,1.25]$ | 同上 §2.3 与 Algorithm 1 | 成立（「两次集合操作」措辞 → S3-3） |
| Cesista 归属（§2.1）；Bernstein & Newhouse 框架归属；Cesista 链接 | 同上 §2.1；muon 博客 | 成立 |
| **µP/transfer 缺席核查（纪律项）** | 全文正则 `transfer|µP|muP|maximal update`（大小写不敏感）零命中；2.5 §8 纪律注记在文 | 纪律被遵守 ✓ |
| Muon 博客：系数调参准则（$\varepsilon$ 高达 0.3 不伤 loss、目标长程 $[0.7,1.3]$、ad-hoc 梯度法）、经典系数 $(2,-1.5,0.5)$、bf16 稳定 vs coupled Newton 需 fp32、Nesterov 默认、开销 $Tm/B$ 与两场景数字、内存 = SGD-momentum、QKV 拆分、embedding 的 modular-norm（Large et al.）与输出头「纯实证」、"speculate…rare directions" 原词、三个 open questions、谱系（Kovarik 1970/Björck & Bowie 1971/Higham） | kellerjordan.github.io/posts/muon/ 全文 | 成立（「约 20 步」「单调」「[0.47,1.14]」三处为章内新增数值，见 S2-3/S2-4/S3-2——均非博客口径） |
| muon.py：系数、docstring「$US'V^\top$，$S'_{ii}\sim\mathrm{Uniform}(0.5,1.5)$」、转置行、$\sqrt{\max(1,A/B)}$、隐藏层限定 docstring | KellerJordan/Muon contents API 全文 | 成立 |
| K2：式 (13)(14)（$S_{\max}^h$ 含 $1/\sqrt d$、$\gamma_h$ per-head、$\alpha=0.5$）、MLA 分量规则（$q^C,k^C$ 乘 $\sqrt{\gamma_h}$、$q^R$ 乘 $\gamma_h$、共享 $k^R$ 不动）、post-update 不进计算图、少数 head、9B/53B 破 1000、$\tau=100$、30% 步后回落、15.5T 零 spike、Algorithm 1 组合、WSD、1.04T/32B | ar5iv 2507.20534 §2.1/§2.5/图 2-3 | 成立 |
| **K2 后训练优化器（推翻项）** | 同上 §3.1（"We employ the Muon optimizer in our post-training"）、§3.2.3（"As in SFT, we employ the Muon optimizer to minimize this objective"） | **2.5 §9 第 6 条与末段错误 → S2-2** |
| GLM-130B：§2.2 裁剪值 1.0（"clip gradients using a clipping value of 1.0"）；§3 "lags behind a 'spike' in gradient norm by a few training steps"；embedding 梯度 "several magnitude larger"；EGS 式（word_embedding*α+detach*(1-α)）；α=0.1 "wipes out most spikes…negligible latency"；3 次晚期发散且进一步收缩仍有效；Apex O2（FP16 前向反向+FP32 态与主权重）；spike 两形态（无因自恢复 vs 梯度骤升前兆至 NaN）；越来越频繁；DeepNorm | ar5iv 2210.02414 §2.2/§3/表 11 | 全部成立（「百亿级」量词 → S3-6） |
| GPT-3 引句定位 | ar5iv 2005.14165 全文检索「clip」唯一命中于 Appendix B | **章内标 §2.1，错误 → S2-5**（2.2/2.3 均正确标 App. B） |
| DeepSeek-V3：「The gradient clipping norm is set to 1.0.」 | ar5iv 2412.19437 §4.2 | 成立 |
| OLMo 2：超参表 Gradient Clipping = 1.0（7B/13B 两列） | ar5iv 2501.00656 §2.3 表 | 成立 |
| PyTorch 实现：`_get_total_norm`（foreach_norm + norm-of-norms）、`error_if_nonfinite` 抛错、`clip_coef = max_norm/(total_norm+1e-6)`、clamp max=1.0、注释（避免 `if clip_coef<1` 的 CPU↔GPU 同步）、返回裁剪前 total_norm | pytorch/pytorch `torch/nn/utils/clip_grad.py`（contents API 全文，L49-243） | 成立 |
| 2.4 §1 教学代码与官方实现等价断言 | 本机 torch 2.8.0 实跑（未触发与触发两 regime） | 通过 |
| 2.4 数值：44.4°（45−arctan(1/100)=44.43°）；$(1-\beta_2)k^2=500$、$\sqrt{501}\approx22$；$1/(1-\beta_2)=20$、$1/(1-\beta_1)=10$ | 独立复算 | 成立 |
| 2.4 式 (2) 投影证明与练习 1 的 ℓ∞ 结论 | 独立重推（Cauchy-Schwarz 等号条件；盒子投影逐坐标独立） | 成立 |
| 跨章指针：1.5↔4.1（spec 登记的前瞻对）、2.4→3.1/7.3、2.5→7.2、2.4↔2.5 双向、2.4↔1.5 双向、models 侧目标（2.3 §4 fp16≈11 溢出、2.7 §5/§5.1/§6/§7/§8、4.5 式 (12) z-loss、8.5 式 (7) z-loss、5.5 MLA、1.2 式 (8) 优化器态账） | 逐章 grep + models 目录与目标章核对 | 成立；唯 2.2 缺 7.2 指针（S3-7） |
| 符号约定（spec §4.3）：$N$=参数量（1.1 ✓）；$d$=DP（1.5 ✓）、$t$=TP（1.4 ✓）；无任何章用 $G$ 表 DP 度数；GRPO 未出场；梯度记号 $g$（2.4）/$G_t$（2.5）/$G_{\mathrm{est}}$（2.3）各自章内一致 | 全 10 章通读 | 成立（局部豁免两例登记：S3-9/S3-10） |
| 术语一致性与 AI 味 rubric A 词表全扫（「值得注意的是/综上所述/让我们/本节将」等 13 词） | 全 10 章 grep | 零命中 |
| lint（机械门禁） | `bash scripts/lint-chapter.sh` 逐章 | **10/10 LINT OK** |
| 篇幅：2.4 = 3674 CJK 字（≥3500 标准章）；2.5 = 5648（≥5500 推导重章软目标） | 脚本计数（剥离代码与公式） | 达标 |

**覆盖率**：2.4 的可核查事实断言约 15 项、2.5 约 35 项（含 20 格数值表逐项），全部经一手来源对照或独立复算，无「确认无误但无来源」项；sweep 覆盖全 10 章的符号、术语、指针、模板、篇幅、禁令词、references 回写七个面。

## 备注

- 批 1（P1 五章）与批 2（2.1–2.3）的逐章事实核查不在本报告范围；本报告对 1.x/2.1-2.3 的发现仅限 sweep 层（S3-7/S3-8/S3-9），未对其论文口径做独立深挖。
- S2-2 修复时会改动 2.5 §9 的两处句子；注意同步核对 §9 第 2 条（「小 batch 与微调未验证」引 toy_train.py docstring）与修复后第 6 条的措辞相容（docstring 警告是官方实现的保守声明，K2 的 SFT/RL 实证是另一来源的正面证据，两者并陈不矛盾）。
- S3 项按协议进入下波清零清单，逐条判定修复/接受/驳回；S3-9、S3-10 建议直接判「接受」。

---

## 修复复审记录（2026-11-17 复审轮）

复审范围：重读 2.4/2.5 当前全文（非 diff），逐条判定；重点复算 S2-1/S2-3/S2-4；lint 重跑；2.2 §7 指针与 references.md 回写确认。

| finding | 判定 | 当前文本证据 / 复算 |
| --- | --- | --- |
| S2-1（wd 幻影差异） | **ADDRESSED** | 2.5 §4 其二现文：「代码把式 (12) 拆成两行执行，逐项恒等：`p.mul_(1 - lr * wd)` 落实 $-\eta_t\lambda W_{t-1}$ 项，`p.add_(u, alpha=-adjusted_lr)` 配合上行落实 $-\eta_t\cdot0.2\sqrt{\max(A,B)}\cdot O_t$ 项——尺度因子只属于正交化项，wd 项的 $\eta_t$ 就是未调整的 `lr`」——与 toy_train.py L200/L203 及论文式 (4) 逐项对账恒等，幻影消除；参考区条目已删「wd 的 lr 口径差异」 |
| S2-2（RL 证据阙如） | **ADDRESSED** | §9 第 6 条现引 K2 §3.1/§3.2.3 两句原文并给出收窄后的开放口径（缺 AdamW 对照消融与独立复现）；末段改为「第三个由 K2 的 post-training 实践部分回答……强度低于预训练侧的 scaling law 证据」。新增句「Moonlight 侧『Muon 预训练 checkpoint 配 Muon 微调性能最好』（K2 §3.1 转引）」经 K2 原文复核成立（"the conclusion of our previous work [46] that a Muon-pre-trained checkpoint produces the best performance with Muon fine-tuning"，[46]=Moonlight） |
| S2-3（5 步像区间） | **ADDRESSED** | §3 bullet 2 与总结表现均为「$\gtrsim10^{-3}$ 的输入压进约 $[0.47, 1.20]$」，并附细网格复算口径。复审独立复算（2×10⁶ 点网格）：$[10^{-3},1]$ 的像 = $[0.4705, 1.2024]$、最大值在 $\sigma_0\approx0.0041$——headline 与最大点命中；括号内下限 0.473 与复算 0.4705 差在第三位小数（网格密度差异），headline 口径正确，不另立项 |
| S2-4（经典系数 20 步） | **ADDRESSED** | 现文「要约 12 步（11 步约 0.997、12 步约 1.0000）」；复审复算轨迹：11 步 0.9968、12 步 0.99996——命中 |
| S2-5（GPT-3 定位符） | **ADDRESSED** | 2.4 §1 与 §3 表两处均改「附录 B」，与 2.2 §7 / 2.3 §1 口径一致 |
| S3-1（发散阈值） | ADDRESSED | 现文「阈值约为 1.264（1.25 有界、1.27 起发散）」、§9 第 4 条改「$\gtrsim1.26$」；复审复算阈值 1.2637——主数字命中（括号内「1.27 起」为测试锚点表述，可接受） |
| S3-2（「单调」） | ADDRESSED | 现文「逐步递增、但有轻微过冲——$f$ 在 $\sigma\approx0.89$ 处峰值约 1.002，随后回落」。复审解析复核：$f'(\sigma)=0$ 于 $\sigma=\sqrt{0.8}\approx0.894$、峰值 $f\approx1.0018$——其数值比本员原报（0.949/1.0009）更准，正确 |
| S3-3（两次集合操作） | ADDRESSED | §6 现文「新增两项操作：一次 DP gather 集合通信……加一次全矩阵 NS 计算（……只保留本地分片对应块）」——与论文 §2.3 "two additional operations" 及 discard 语义一致 |
| S3-4（wd 式号） | ADDRESSED | §4 第四段现文「论文式 (3) 据此先把……wd 并入，式 (4)（即本章式 (12)）再叠尺度匹配因子」，并正确写出式 (3) 形态 |
| S3-5（式 (1) 标签） | ADDRESSED | §4 第一段改标「论文式 (1) + 脚注 1 的 Nesterov 变体」，并明说印刷版式 (1) 输入为 $M_t$、式 (10) 按脚注口径写出 |
| S3-6（百亿级） | ADDRESSED | 2.4 §5 改「130B 级训练」 |
| S3-7（2.2 缺 7.2 指针） | ADDRESSED | 2.2 §7 末现增「本节这张表只覆盖 optimizer 本体；lr / bsz / warmup / decay / wd / $\beta_2$ / clip 的全字段真实配方对照表归 7.2」 |
| S3-8（references 回写） | ADDRESSED（一项残留） | deepseekv3/olmo2/gpt3/llama3/pytorch-repo 五行已补齐所缺章节；**adamw 行被引章节仍为「2.2」，缺 2.4**（2.4 §4 与参考区均引用该条目）——一句话残留，转下波清零 |
| S3-9 / S3-10 | 接受（controller 判定） | 登记动作随 W1 收尾的 README 审查记录行写入（当前 README 尚无 W1 行，属波末流程项，不阻塞） |

**修复引入新问题检查（仅限修复触及处）**：新增文本的引用（K2 §3.1/§3.2.3、[46]=Moonlight）回原文复核无误；数值修正全部经独立复算命中；无新公式、无新指针、无格式破坏。

**lint 重跑**：2.4、2.5 均 `LINT OK`（2026-11-17 复审轮）。

**复审结论**：S2×5 全部 ADDRESSED，S3 十项九项关闭、一项残留一句话（adamw 行缺 2.4，转下波清零）。**2.4 / 2.5 放行；全波 sweep 层无未关闭项——W1 批 3 通道关闭。**
