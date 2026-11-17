# training W5 批 1（8.1-8.3）对抗审查报告

日期：2026-11-18。审查人：独立 fresh-context evaluator（与写作 agent 无共享上下文）。
对象：`training/08-rl-math/01-policy-gradient.md`、`02-actor-critic-and-gae.md`、`03-ppo.md`。
标准：STYLE_GUIDE.md + docs/reviews/2026-11-15-style-audit/rubric.md + training spec §4.3 / §5（8.1-8.3）。

立场声明：默认内容有错，逐条以一手来源复核。writer 自述 PPO 原文（1707.06347）拿不到：复核确认 **ar5iv 对该文是 stub 页**（"Untitled Document / See pages 1-last of ppo-min.pdf"，无正文）；改用 **arXiv 官方 PDF 直取成功**（12 页全文，逐节核完）——8.3 的关键声称因此全部按原文（而非转引）裁定。TRPO/GAE 同法取 arXiv PDF；Sutton-Barto 第 2 版取 incompleteideas.net 官方 PDF（73MB，断点续传三轮）；CleanRL/OpenRLHF/Baselines/TRL 经 GitHub contents API 取源码；5 个 arXiv ID 经 export.arxiv.org API 复核标题。

## 一、强制独立重推导（三项全部 PASS，中间过程如下）

### 8.1 策略梯度定理两形式 + baseline 无偏 + 最优 baseline

**(a) 轨迹形式**：J(θ)=Σ_τ p_θ(τ)R(τ) ⇒ ∇J=Σ_τ ∇p_θ·R（有限轨迹集逐字成立）=Σ_τ p_θ·R·∇log p_θ（∇p=p∇log p，p>0）=E_τ[R·Σ_t ∇logπ_θ(a_t|s_t)]（log p_θ(τ) 展开后 p_0、P 被 ∇_θ 清零）。与式 (4)(5)(7)(9) 逐步一致；score 零均值式 (8) 四步链（期望展开→π·∇logπ=∇π→求和交换→归一）逐步成立。

**(b) 状态分布形式**：∇V^π=Σ_a[∇π·Q+π·∇Q]，∇Q^{π,γ}(s,a)=γΣ_{s'}P∇V(s')（r 无 θ 依赖）⇒ 不动点迭代 ∇V(s)=φ(s)+γΣ_{s'}P^π∇V(s') 展开为 Σ_k γ^k Σ_x Pr(s→x,k,π)φ(x)，对 s_0~p_0 取期望收拢为 Σ_x d^γ(x)φ(x)=Σ_t γ^t E[∇logπ·Q]。与式 (15)-(19) 一致；恒等式无比例常数、归一化常数 Z 的两口径（持续 1/(1−γ)、episodic 期望回合长度）均正确。

**(c) baseline 无偏**：E_a[∇logπ·b(s)]=b·Σ_a∇π=b·∇1=0（式 20 ✓）。

**(d) 最优 baseline**：f(b)=E_a[‖g‖²(Q−b)²]，f'(b)=−2E[‖g‖²(Q−b)]=0 ⇒ b*=E[‖g‖²Q]/E[‖g‖²]，f''=2E‖g‖²>0 极小（式 23 ✓）。§5.2 算例脚本复算：E[X]=1.5（与 b 无关，自检通过）、b*=1、三档 Var=2.25/0.75/1.5938，与正文全对。

### 8.2 GAE

伸缩：Σ_{l<k}γ^lδ_{t+l} 相邻项逐对相消 ⇒ Σγ^l r+γ^k V(s_{t+k})−V(s_t)（式 9 ✓）。系数收集：δ_{t+l} 总系数=(1−λ)γ^l Σ_{k>l}λ^{k−1}=(γλ)^l（式 12/13 ✓）。偏差分解式 (11)（γ^k 折扣的 k 步后价值误差 + 只依赖 s_t 的 baseline 型项）与 PG 相关偏差因子 γ(1−λ)/(1−γλ)（式 14）独立重推一致；数值复算：λ=0/0.9/0.95 → 因子 0.99/0.908/0.832、窗长 1/9.2/16.8 步。练习 3 手算复现（δ=(0.1,−0.2,0.4,0.4)；λ=1 组 Â_t=R−V(s_t) 逐项成立）。

### 8.3 PPO clip 逐情形

Â>0：r≤1+ε 时 min 取未截断项、梯度照常；r>1+ε 时 (1+ε)Â 钉平、梯度零 ⇒ L=Â·min(r,1+ε)（式 8 ✓）。Â<0：min 对负系数方向反转，r<1−ε 时钉平；r>1+ε 时保留未截断项 ⇒ L=Â·max(r,1−ε)（式 9 ✓）。悲观性 min≤L^CPI 逐点成立 ✓；反向永不截断 ✓。ASCII 图数值（Â=2：1.0/1.6/2.0/2.4/2.4/2.4；Â=−2：−3.0/−2.0/−1.6/−1.6/−1.6）脚本复算全对，与论文 Figure 1 文字描述（正左负右、r=1 起点、截断点 1±ε）一致。

## 二、Findings 表

### S1（事实性致命）：0 起

### S2（实质问题）：9 起

| # | 位置 | 断言 | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 8.3 §5 式 (10) 引言行、§6 第 2 条、参考节 | PPO 联合目标为「论文式 (10)」 | 1707.06347v2 PDF：L^{CLIP+VF+S} 编号为**式 (9)**；式 (10) 是 A3C 式有限视界 advantage 估计器 | 三处「论文式 (10)」改「论文式 (9)」 |
| S2-2 | 8.3 §5 式 (10) 符号注 | 「论文 MuJoCo 口径 c_1=1、c_2=0」 | 论文 Table 5 的 VF coeff c_1=1 属 **Atari**；MuJoCo §6.1 明示「don't share parameters (so c1 is irrelevant), don't use an entropy bonus」 | 改：MuJoCo 口径 c_1 无涉（分离网络）、c_2=0；c_1=1 归 Atari |
| S2-3 | 8.3 §7 第二条 | 「论文未做系统网格扫描」「『0.1 到 0.3 都可用』非论文消融」 | 论文 §6.1 Table 1 即 ε∈{0.1,0.2,0.3} 消融（7 任务×3 seed，归一得分 0.76/0.82/0.70，均远胜无截断 −0.39）；§6.1 正文「which we search over」 | 承认 Table 1 三点消融；若意在「无逐任务细网格/无 LLM 域消融」须换措辞 |
| S2-4 | 8.3 §8（early-stop 行）、总结表 | 「break——跳出的是 epoch 循环」 | CleanRL ppo.py 源码：该行在 **minibatch 循环**内（`for epoch` → `for start` → … → break）；break 只结束本 epoch 剩余 minibatch，外层 epoch 循环继续 | 改「跳出本 epoch 的 minibatch 循环；后续 epoch 仍会进入并通常再次触发」；总结表同步 |
| S2-5 | 8.3 §3 | TRPO 两步替换「（其 §6）」 | 1502.05477 §4「Optimization of Parameterized Policies」其式 (11)(12)；§6 是「Practical Algorithm」（CG+线搜索） | 「其 §6」改「其 §4」 |
| S2-6 | 8.1 §4.3 两处、总结表、参考节 | PG 定理状态分布推导路线与比例常数口径引「Sutton-Barto §13.1」 | S&B 2 版目录与正文：§13.1=Policy Approximation and its Advantages（目标定义）；不动点证明框与「episodic 常数=平均回合长度」句均在 **§13.2 The Policy Gradient Theorem**（教材 PDF p.324-325 核对） | 四处「§13.1」改「§13.2」（§3 的「§13.1-13.2」区间引用可留） |
| S2-7 | 8.1 §5.2 末、参考节 | 最优 baseline（式 23）「同一推导路线见 Sutton-Barto §13.4」 | S&B §13.4 全文核对：只有无偏性式 (13.10) 与「value estimate 是自然选择」的定性讨论，**无方差最小化推导**；该结果标准出处为 Greensmith et al. 2004 | 删除该归属或改引 Greensmith et al. 2004 |
| S2-8 | 8.2 §4 数值对照段 | 「八成把握的好结局仍有**一半样本**给出负权重」 | 例设 R~Bernoulli(0.8)：负权重概率=P(R=0)=0.2（两成），非一半 | 「一半样本」改「两成样本」 |
| S2-9 | 8.1 式 (25)（N=批内轨迹数）与 §8（2N FLOPs，N=参数量）；8.3 §1（N=参数量）与 §5/§8（N=并行环境数） | N 同章两义，且与全书约定冲突 | spec §4.3 全书约定 N=模型参数量；并行度数禁用 N/G | 计数改名 N_env/N_traj 或用文字，避免与参数量 N 并存 |

### S3（minor）：9 起

| # | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| S3-1 | 8.3 §4 开头 | 引语 "destructively large" 原文在 §2.1 评述 L^PG 多步复用；L^CPI 处原文为 "excessively large" | 换引 "excessively large" 或改挂 L^PG 语境 |
| S3-2 | 8.2 §5.2 末 | 「LLM 场景的取值惯例见 §7」——§7 未给 λ 的 LLM 惯例（悬空指针） | 删指针或改为随 8.3/8.4 实现口径展开 |
| S3-3 | 8.2 §5.2 | 「约 0.83 γ」——精确 0.832（≈0.84γ）；字面读 0.83γ=0.822 | 写「约 0.83（≈0.84γ）」 |
| S3-4 | 8.3 §8 末 | 「曲线逐点对齐（其文档页表格口径）」——CleanRL docs 是平均回报对照表（490.04 vs 497.54 等），非曲线逐点 | 改「平均回报与 Baselines 对照同量级」 |
| S3-5 | 8.3 §5 循环不变量二 | 「CleanRL 文档把『用随机取数代替打乱切分』列为常见错误实现之一」——docs 页只列 13 条标题+代码行号，无此定性；代码事实（shuffle+切片）成立 | 改指代码事实，去掉「文档列为常见错误」归属 |
| S3-6 | training/references.md | 被引章节漏回写：`christiano` 缺 8.1、`openrlhf` 缺 8.3 | 回写两字段 |
| S3-7 | 8.3 §4.1 | 「§7 的 clipfracs 就是它的实现」——实现在 §8 | 「§7」改「§8」 |
| S3-8 | 8.3 §8 | 「clipfracs…即『梯度被关掉的样本比例』的实现」——clipfracs 计 \|r−1\|>ε 双向，口径比梯度关断集宽（如 A>0 且 r<1−ε 计入但未被关断） | 补口径差异一句 |
| S3-9 | 8.3 §3 | 「vine 只在理论对照中出现」——TRPO 论文 §8 实验同时跑了 vine TRPO（Table 2 并列成绩） | 改「后续实现全部走 single path；论文实验中 vine 仅作对照变体」 |

## 三、Verification log

来源获取：PPO/TRPO/GAE = arXiv 官方 PDF（arxiv.org/pdf，pypdf 全文抽取）；S&B = incompleteideas.net RLbook2020.pdf（73MB，三段续传，548 页，目录+正文核对）；repo 源码 = GitHub contents API；arXiv ID = export.arxiv.org API；CleanRL docs = docs.cleanrl.dev 静态页。

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| PPO 式 (7) clip 形式、ε=0.2、Figure 1 双图结构、悲观下界表述、一阶相等注记 | 1707.06347 PDF §5 | 一致 |
| PPO 联合目标编号、value target 形式 | 同上 §6 式 (8)(9)(10) | **章内三处误引式 (10)→S2-1** |
| PPO adaptive β 规则 | 同上 §4 | 一致 |
| PPO surrogate 对比（7 任务、1M 步、Table 1 名次） | 同上 §6.1 | 一致；ε 消融存在→S2-3 |
| PPO MuJoCo/Atari 超参两组（含 c_1/c_2 归属） | 同上 Table 3/5 + Baselines ppo1 run_mujoco.py + ppo2/defaults.py | MuJoCo 组一致；c_1=1 错置→S2-2；Atari K=4 归 Baselines 正确（论文 K=3） |
| TRPO 式 (2)(3)(4)、Theorem 1（C=4εγ/(1−γ)²）、D_TV²≤D_KL、δ=0.01、single path/vine、解析 FIM、CG k=10、10% 子采样、线搜索、步长式 √(2δ/g^TF⁻¹g) | 1502.05477 PDF §2-§6、§8、附录 C | 逐项一致；节号→S2-5；vine 表述→S3-9 |
| GAE δ 定义、k 步（论文式 11-14）、Â^(∞)（式 15）、GAE（式 16）、端点（式 17/18）、「bias smaller as k→∞」 | 1506.02438 PDF §3 | 一致 |
| GAE γ-just（Def 1/Prop 1）、g^γ 口径、footnote 1（A^{π,γ} 代 A^π 的偏差）、「γ<1 恒有偏/λ<1 只在 V 不准时有偏」「最优 λ 明显低于最优 γ」、TD(λ) 同型 | 同上 §2-§3 | 逐句一致；章内域限定正确 |
| CleanRL ppo.py 全部走读事实（结构/GAE 块/loss 合成/默认超参/双 KL/clipfracs/anneal/eps=1e-5/target_kl=None/dones 口径/logical_or） | GitHub API（master，sha edde76b） | 逐项一致；break 层级→S2-4；clipfracs 口径→S3-8 |
| CleanRL docs（13 条、三任务对照表） | docs.cleanrl.dev | →S3-4、S3-5 |
| OpenRLHF PolicyLoss 解耦 clip_eps_low/high | GitHub API | 一致 |
| TRL 无 ppo_trainer.py | GitHub API（trl/trainer 22 项） | 一致 |
| arXiv ID×5 标题 | arXiv API | 全对 |
| Williams 1992 出处与更新式、Sutton 1999 会议与定理口径 | PPO 文献条目交叉核+权威书目 | 一致（无 arXiv 原文，未转述具体数字，风险低） |
| S&B：§13.1/13.2/13.3/13.4/§9.3/§11.5/§3.5-3.6/§6.1/第 12 章 | RLbook2020.pdf 目录+正文 | §13.3-13.4 REINFORCE✓、§9.3 半梯度✓、§11.5 Bellman error 梯度下降✓、§3.5-3.6✓、§6.1✓、ch12✓；**§13.1→S2-6、§13.4 最优 baseline→S2-7** |
| 全部数值算例（8.1 §4.1/§5.2/§7.1、8.2 §4/式14/练习3、8.3 clip 数值/488/320/χ² 恒等式） | 本地 Python 复算 | 全对，除→S2-8、S3-3 |
| rubric A 禁令词、公式源码禁令（\operatorname/align*/\\,/\\\{/裸\*/\(<\)ASCII 等）、开口 flanking | 本地 grep | 零命中 |
| 模板六件、lint、CJK 字数 | lint-chapter.sh + 本地统计 | 三章 LINT OK；5492/3717/5695（8.1 软目标 99.9%） |

**覆盖率**：arXiv ID 5/5；repo 路径 4/4；论文级公式与超参声称逐式/逐项；数值算例 8/8 复算；S&B 节号引用 8/8；rubric A/B 机械项与模板项全查；三章正文推导链 100% 走读。

## 四、writer 存疑四项独立裁定

1. **episodic 归一化常数**：内容口径成立（S&B 原文 "the constant of proportionality is the average length of an episode"），但出处节是 §13.2 非 §13.1——并入 S2-6。
2. **§9.3/§11.5 节号**：教材目录直核——§9.3=Stochastic-gradient and Semi-gradient Methods、§11.5=Gradient Descent in the Bellman Error（residual gradient 所在节），8.2 引用**正确**。
3. **GAE λ 经验域限定**：合格。正文已限定「论文连续控制任务域口径」未外推 LLM；论文 §3 原文逐字对上。
4. **8.1 CJK 5492 vs 5500**：达标口径成立（软目标，差 0.15%）。

## 五、规范审查摘要

rubric A 零命中；每节以事实/公式/问题开场；无空洞过渡；术语一致；公式源码禁令零命中。符号约定：λ/ε 全书一致，G 未误用，TRPO δ 与 TD 残差 δ_t 的冲突已显式注明 ✓；N 两义 → S2-9。教学审查：推导无断步、符号均有符号表、无循环解释、前置与 metadata 一致；练习 4 题/章齐。唯一 B1 擦边（8.1 §5.3「显著」）同句有机制支撑，不判违规。

## 结论

**不放行**（S1=0、S2=9、S3=9）。三项强制重推导全部 PASS，核心数学与论文级事实主干正确；9 起 S2 均为引用节号/归属/局部事实类，修复成本低；按协议 S1/S2 修复后复审本批。

---

## 修复复审记录（2026-11-18 复审轮）

复审方式：三章当前全文重读（非 diff），逐条对照原 findings 判定；修复触及处另查是否引入新问题；lint 重跑。

### 逐条判定

**S2 组：**

| # | 判定 | 当前文本证据 |
| --- | --- | --- |
| S2-1 | ADDRESSED | 8.3 §5「（论文式 (9)，本章编号 (10)）」、§6 开头「论文式 (7)(9)」、§6 第 2 条「不在论文式 (9) 里」、参考节「总目标式 (10)（论文式 (9)）」——四处全部改对 |
| S2-2 | ADDRESSED | 8.3 §5 符号注改为「MuJoCo 口径用分离网络、不用熵奖励——其 §6.1 明言 "coefficient c_1 is irrelevant"；c_1=1、c_2=0.01 属 Atari 口径（其附录 Table 5）」——与原文一致 |
| S2-3 | **NOT ADDRESSED** | 8.3 §7 第二条原文未动：「『0.1 到 0.3 都可用』的说法来自后续复现文献与实现文档，非论文消融」——仍与论文 §6.1 Table 1（ε∈{0.1,0.2,0.3} 三点消融，得分 0.76/0.82/0.70）直接矛盾（§4.2 与 §7 第一条新增的 Table 1 引用是别处的正确修补，未触及此句） |
| S2-4 | **驳回（evaluator 误报，向写作方致歉）** | 复审用 `cat -e` 钉死缩进：`for epoch` 8 空格、`for start` 12 空格、`if args.target_kl...` 12 空格（epoch 体层、minibatch 循环之外）、`break` 16 空格——break 确跳出 **epoch** 循环，章文「跳出的是 epoch 循环，已算完的本 minibatch 更新照常生效」与源码一致，原判有误，撤销；总结表「early-stop 跳 epoch」同样成立 |
| S2-5 | ADDRESSED | 8.3 §3 改为「两步替换（其 §4，式 (11)(12) 处）」——与 TRPO §4「Optimization of Parameterized Policies」式 (11)(12) 对上 |
| S2-6 | ADDRESSED | 8.1 §4.3 两处、总结表、参考节共四处「§13.1」全部改「§13.2」；§3 的区间引用「§13.1-13.2」按原判定保留，合规 |
| S2-7 | ADDRESSED | 8.1 §5.2 改引「Greensmith et al. 2004（JMLR 5:1471-1530，无 arXiv，[B]）为标准出处；Sutton-Barto §13.4 给出无偏性与定性讨论」——书目正确，§13.4 角色改述与教材一致；references.md 已补 `greensmith2004` 条目（controller 录入，字段合规） |
| S2-8 | ADDRESSED | 8.2 §4 改为「八成把握的好结局仍有两成样本（R=0，概率 0.2）给出负权重」——数值正确 |
| S2-9 | **部分 ADDRESSED** | 8.1 式 (25)(26) 与 §6-8 全部改 `N_traj` 并加约定注「本书约定 N 专归参数量」（式 (25) 符号注）✓；8.2 数据流图同步 ✓；**8.3 未改**：§1「6N/2N FLOPs（N=参数量）」与 §5「N 个并行环境」「N=1/N=8」、§8「N=4 环境」「[T, N]」同章两义并存依旧 |

**S3 组：**

| # | 判定 | 当前文本证据 |
| --- | --- | --- |
| S3-1 | ADDRESSED | 8.3 §4 改为「论文原文是 "an excessively large policy update"（"destructively large" 一词在 §2 评述朴素策略梯度处）」——两处原文位置均准确 |
| S3-2 | ADDRESSED | 8.2 §5.2 末改「LLM 场景的实现口径随 8.3/8.4 展开」，悬空指针消除 |
| S3-3 | ADDRESSED | 8.2 §5.2 改「界仍约 0.83（即 0.84 γ）」——0.832≈0.84×0.99，精确 |
| S3-4 | ADDRESSED | 8.3 §8 改「平均回报与 OpenAI Baselines 同量级（其文档页表格口径：CartPole-v1 490.04±6.12 对 497.54±4.02 等）」——数值与 docs 表逐项一致 |
| S3-5 | **NOT ADDRESSED** | 8.3 §5 循环不变量二原文未动：「（CleanRL 文档把『用随机取数代替打乱切分』列为常见错误实现之一）」——docs 页仍只见 13 条标题+代码行号，无此定性 |
| S3-6 | ADDRESSED | references.md：`christiano` 被引含 8.1、`openrlhf` 被引含 8.3（controller 回写核实） |
| S3-7 | ADDRESSED | 8.3 §4.1 改「§8 的 `clipfracs`」 |
| S3-8 | ADDRESSED | 8.3 §8 新增口径注记「clipfracs 是双向计数……梯度真正被关掉的样本还要按 Â_t 符号取单向子集，clipfracs 因此是关断量的上界口径」——复核成立（关断集 ⊂ {\|r−1\|>ε}） |
| S3-9 | ADDRESSED | 8.3 §3 改「vine 在论文实验中作为对照变体与 single path 并列报告（其 Table 1/2），之后未获沿用」——Table 1（Atari 结果，含 vine/single path 两行）与 Table 2（连续控制参数表，标题含 vine and single path）均核实 |

### 修复引入新问题检查（限修复触及处）

- 8.1 §8 同句并用 `N_traj` 与 `N`（参数量），同句内已有「N 为参数量」标注，不构成新冲突；8.1 新增 Greensmith 引文的 key/书目/[B] 分级均合规，且 lint 检查项 4 不受非 arXiv 条目影响。
- 8.3 §5 式 (10) 双编号标注「论文式 (9)，本章编号 (10)」清晰，无歧义。
- 8.3 §8 新增的 clipfracs 口径注记与 Table 1 引用（§4.2 末段「其 §6.1（Table 1）」）复核无误。
- 其余修复均为局部措辞/节号替换，未波及周边公式与推导。

### lint 重跑记录

```
bash scripts/lint-chapter.sh training/08-rl-math/01-policy-gradient.md → LINT OK
bash scripts/lint-chapter.sh training/08-rl-math/02-actor-critic-and-gae.md → LINT OK
bash scripts/lint-chapter.sh training/08-rl-math/03-ppo.md → LINT OK
```

### 复审结论

S2-1/2/5/6/7/8 关闭；S2-4 撤回（误报）；**S2-3 与 S2-9（8.3 部分）未关闭**；S3 组 8 关闭、S3-5 未关闭（可留待下波清零）。按协议 S2 未清零——**暂不放行**：8.3 待修两处均为一句话级改动（§7 第二条改写为承认 Table 1 消融存在；§5/§8 的环境数 N 改 N_env 或文字）。修完此两处即满足放行条件，无需全量重审。
