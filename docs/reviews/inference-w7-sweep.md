# W7 全书一致性 Sweep 报告 · inference 维度（37 章 / 9 Part）

日期：2026-11-24。Reviewer：fresh-context 独立全量 sweep（不继承写作上下文）。**只读审查，未改章节、未 commit。**
范围：HOLISTIC 一致性 + 完整性 sweep（spec §7 W7 / §6.6 步骤 4）——**非公式重推导**（各波 evaluator 已独立重推，本轮不重复）。
基准：inference spec §2 / §4.3 / §6.6 / §7 + STYLE_GUIDE + R2 style sheet（12 条）+ rubric + inference/README.md（进度表 / 依赖图 / 盘点表 / errata 表）+ inference/references.md（83 条）。
方法：机械扫描（Python 解析 37+51+39 章库存、指针归因、符号/escape/练习/总结/refs 交叉核；node 用生产 classifyBlock + FIGURES 注册表做图 census）+ 逐处上下文裁定。脚本 /tmp/w7-sweep/（sweep.py / sweep2.py / census.mjs），中间产物 findings.txt / pointers2.txt / symbols.txt。

## 0. 总判定

- **Status：PASS with findings（S1 = 0，S2 = 1，S3 = 5）。** 无事实性错误；唯一实质问题是 9.1 §7 的 $G$ 符号占用冲突（S2-1）。
- **图 census：COMPLETE。** 114 图（part01=7 / 02=9 / 03=12 / 04=12 / 05=14 / 06=9 / 07=16 / 08=19 / 09=16）与波次记录逐一吻合；全 37 章 73 个无语言围栏 ASCII 块 **73/73 被注册场景图替换，diagram 通用兜底 = 0**；33 个带语言围栏（29 python / 2 cpp / 2 text）均为代码或**注册模块注释明文登记的 intentional keep**（8.5 §1 曲线图 / 9.1 章脊全链，后者另有 §1 签名场景图覆盖同内容）。无未覆盖结构块、无悬空注册、无越界块号、无缺失 afterHeading 锚。
- **符号一致性：1 issue（S2-1），其余 CLEAN。** V4.1 MLA = 512 元素（448+64）全维度无残留 576（43 处 576/1152 命中逐一裁定：全部 V3 行显式代际标注、X-1 坑注/errata 历史叙述、或无关数值/坐标）；b/B/bytes/κ/γ/α/λ/d_model/d_latent/d_rope/E/e/k/c_chunk/n_P/n_D/𝒢 全合 §4.3；翻译硬规则执行到位（2.1/2.3/2.4 recap 翻译注、4.2 源码 gamma/alpha→g/u、9.3 training B→BW、3.2 α/k 局部声明）。
- **指针闭合：CLEAN。** 全部章内 §/式 指针、跨章 N.N §M / 式(K) 指针、跨维度 models/training 指针逐一解析到存在目标（含摘要「跨维度指引」密集行与论文 § 引用分类）；盘点表 W1-W6 收窄行全部执行，保留泛指向行（models 4.8 §2 MoD/early-exit、models 1.1 §8 叙事句 + README 入口活链接、models 1.2 §9 / 5.2 §3.4 MFU 归属）与记录一致；models/training 全量 inference 提及扫描无漏收窄。
- lint 37/37 LINT OK；练习四档/校验锚/命题闭环 37/37 过；总结回指 37/37 过；refs 83 条 7 字段 schema / 被引章节 forward+reverse 交叉核 clean；篇幅 37/37 达标（12 重章 ≥5500、9.2 短篇 2738≥2500）。
- **W5/W6 残留 advisory（lint display 反转义漏检）本轮 raw 扫描补偿：全 37 章 display+inline+codespan 禁用形 0 命中**（内容干净；lint 脚本本身的 display 覆盖缺口为工具项，见 §6）。

## 1. Findings 表

| # | Severity | 位置（file:line） | 问题 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | S2 | inference/09-production/01-capacity-and-cost.md L238 / L240 式(13) / L244 / L246 / L313 / L325（对照 L16 章脊、inference/09-production/03-rl-rollout-serving.md L20 / L156 / L185） | **$G$ 符号占用冲突**：9.1 §7 用 $G$ = 单集群 GPU 数（收入式 (13) 及其符号表/坑注/总结/练习 4），违反 spec §4.3 两条硬规则——goodput 行「$G$ 在 training 专归 GRPO 组大小，跨维度保持一致不占用」、PD 实例行「禁裸 P / D / G 当计数」；且与 9.1 自身章脊 L16「禁裸 P / D / G 当计数」声明自相矛盾，与同 Part 同波的 9.3「G = GRPO 组大小（training 约定）」构成章间冲突。W6 逐章审查聚焦 sleep-level S1 与数值重推，漏检此跨章符号项 | 9.1 §7 的 $G$ 全部改 $n_{\mathrm{GPU}}$（1.1 式 (3) 已有约定）：式 (13)、L244 符号表、L246 坑注、L313 总结行、L325 练习 4(b)；或至少补「与 9.3 GRPO 组大小 $G$ 不同义」消歧注（不推荐——仍违反「不占用」硬规则）。数值与公式结构不动 |
| S3-1 | S3 | inference/05-vllm/02-vllm-scheduler-kv.md L130；inference/06-sglang/02-sglang-overlap-scheduler.md L17 / L218；inference/07-deepinfer-and-deepseek/02-dspark-and-swa-replay.md L13（×2）；inference/09-production/01-capacity-and-cost.md L189；inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md L101 | **裸「§4.3」记法跨波不一致**：5 章 6 处用裸「§4.3」指 **spec §4.3**（统一记号 / 翻译 / 批内 B / 换算注明），而其余 13 章均写「spec §4.3」；裸写易被读作章内 §4.3（这些章无 subsection）。另 7.4 L101「§4.3 的 all-reduce MIN」实指**章 4.3**（紧随「（4.3 §4 已述）」消歧），同一记法第三种含义 | 6 处 spec 指向补「spec」前缀；7.4 L101 改「4.3 §4 的 all-reduce MIN」。纯指针句最小修订，不动公式数值 |
| S3-2 | S3 | inference/07-deepinfer-and-deepseek/02-dspark-and-swa-replay.md L295（总结行）/ L304（练习 3(c)） | **W4 非阻塞观察残留（仍 open）**：总结「block KV 复用 bit-exact」与练习 3(c)「证 block KV 复用 bit-exact」用裸 bit-exact 简写；命题 4(3) 与证明（L228/L230）已严谨区分「与存储态逐字一致（无条件）」vs「与全量重算逐位相等（需 batch-invariant 前提）」，简写处缺「与存储态」限定，读者可能误读为无条件 bit-exact | 两处补「与存储态」限定（W4 fix-review 原建议）；证明骨架不动 |
| S3-3 | S3 | inference/07-deepinfer-and-deepseek/02-dspark-and-swa-replay.md L264 | **W4 非阻塞观察残留（仍 open）**：「1.2 / 2.3 的 V4.1『576 元素 / 主 KV 324 B』是事实错误」——X-1 errata 已纠正 1.2/2.3，此句现在读作对现行文本的断言（读者查 1.2/2.3 只见 512），应读作历史记录 | 改「errata 前 1.2 / 2.3 曾写『576 元素 / 主 KV 324 B』，系事实错误（X-1 已纠）」措辞；数值与裁定叙述不动 |
| S3-4 | S3 | site/scripts/lib/figures/inference-part01.mjs（3 处 non-mono seal-deep cap）/ inference-part02.mjs（16 处）；对照 site/src/styles/figure.css L167 `.dia-cap { fill: var(--ink-soft) }` | **W2 carryover「kit/CSS 一次性修全存量」未落地（站点侧）**：.dia-cap CSS fill 覆盖 SVG fill 属性，part03-09 均有 inline-style 提权 wrapper，**W1 两模块 part01/02 无**——其 non-mono `color: 'var(--seal-deep)'` 图注实际渲染为 --ink-soft，与 W2+ 模块图注配色不一致（mono cap 不受影响，.dia-mono 无 fill 规则） | 给 part01/02 加与 part03-09 相同的 cap wrapper，或 kit/CSS 一次性全局修（.dia-cap 去 fill / kit cap 默认 inline style）。属站点图品质一致性项，不影响 census 完整性 |
| S3-5 | S3 | site/scripts/lib/figures/inference-part06.mjs L503 / L506（6.3 chunk 流水图，title L573） | **V3 图标签卫生（可选，W4 fix-review §3.6 同族）**：图注「24576 token × 70272 B」「512 页 × 61 层」用 V3 几何（κL=70272 = 61×1152）但无显式「V3」代际 tag；章正文 L48 符号表与练习 2 校验锚已声明「1.2 §3 几何表 V3 行」，61 层本身亦可区分 V4.1（40 层），故仅标签卫生 | 图注或 title 补「V3」tag（沿 part02 L334 / part06 L174 先例）；可选 |

## 2. 图 census 完整性（spec §6.6 步骤 4）

方法：node 脚本 import 生产 `ascii-figure.mjs` 的 `classifyBlock` 与 9 个 `inference-pNN.mjs` 的 FIGURES 注册表，逐章复现 build-content 的块序号语义（无语言围栏 0 计）。

| 项 | 结果 |
| --- | --- |
| 注册图总数 / 分 Part | 114 = 7/9/12/12/14/9/16/19/16，与 W1-W6 波次记录逐一吻合 |
| 无语言围栏 ASCII 块 | 73 块，**73/73 被 replacesBlock 注册替换** |
| diagram 通用兜底 | **0**（结构图兜底归零达标） |
| grid / algo 豁免块 | 0 / 0（无语言围栏块全部被场景图覆盖，无豁免需求） |
| 带语言围栏 | 33（29 python / 2 cpp / 2 text）；2 个 text 块为 diagram 类内容但**注册模块注释明文登记 intentional keep**：part08 头注「带语言围栏（8.3 cpp / 8.5 text）管线逐字保留，不画」、part09 头注「9.1 唯一围栏带语言（```text 章脊全链）→ 管线逐字保留，四张全 afterHeading」；9.1 该链另有 §1 签名场景图覆盖 |
| 注册健康 | 无重复注册、无悬空 slug、无越界块号、41 个 afterHeading 锚全部命中 |

**判定：COMPLETE，无未覆盖结构块。** 唯一站点侧品质项为 S3-4（part01/02 cap 配色）。

## 3. 符号一致性（spec §4.3）

- **V4.1 MLA = 512（448+64）全维度 CLEAN**：576/1152/46080/12960/17476 + 旧 X-1 派生值（45 KiB / 12.7 / 1.41 / 22.5 / 47.2 / 6.04 / 386.5 / 2.81 / 324 B / 1.78 等）全量扫描，命中逐一裁定——V3 行全部显式代际标注（1.2 L62/L235、2.3 L124、3.1 §5、6.3 L48、7.3 L56/L160、8.2 L165、8.3 L9）；X-1 坑注/errata 叙述（1.2 L53、7.1 L246、7.2 L264、8.2 L167）为裁定留痕；其余为无关数值（4.3 混合批 576 行、4.2 视觉 576 embedding、6.2 24576、9.1 平均上下文 1152、FLOPs 2.81e14）或图坐标。站点 9 个图模块同扫，V4.1 标签均 512/1024/40960/11520/288，无残留。
- **b/B/bytes**：b 专归 block size、B 专归 batch、bytes 不占字母——9 处 b-字节命中全部为 recap 翻译注（models b→bytes）或 b=block size 用法；混场章（8.1/7.4/9.1）无 B 表 bytes。
- **goodput = 𝒢**：1.1/1.3/5.4/8.2/9.1 均 \mathcal{G}，无章用裸 G 表 goodput；1.1 定义要件（per-GPU 归一 + attainment 目标 + 最大请求率）完整，8.2/9.1 显式沿用式 (3)。
- **γ/α**：draft 长度/接受率专归；3.2 AWQ/SmoothQuant α 与 OCP k 有局部声明 + 与 4.4 消歧；1.4 α_freq/α_pres/α_len 下标区分；4.2 源码 gamma/alpha 按硬规则译 g/u；8.1 L194 论文明示不沿用其 α/γ。
- **λ**：1.3 / 9.1 均有「非 GAE 参数」声明（8.5 亦有）。**d/e/k/c**：d_model/d_latent/d_rope 几何行合表；6.3 d=attn_dp_size、c=attn_cp_size 沿 spec；4.3 c_chunk、7.4 cost(q,c) 章首局部声明。**n_P/n_D**：8.2/9.1/8.5/9.3 实例计数用 n_P/n_D 或声明局部 n/m，**唯 9.1 §7 裸 G 当计数 → S2-1**。货币：9.1 一律 USD / US\$ 转义，无裸 \$ 配对陷阱。

**判定：1 issue（S2-1），其余 CLEAN。**

## 4. 指针闭合

- **章内指针**：37 章全部 §N / 式(N) 解析到本章存在的节/公式（公式库存含单行 $$ 与多行 $$ 两种编号形）；命题编号逐章从 1 连续重启，证明降级「留作练习 N」全部指向存在练习。
- **跨章指针**：N.N §M / N.N 式(K)（含摘要「跨维度指引」密集行、括号内多式簇、式 (10)-(12) 范围、/§ 续接）全部解析；5.4 总结采样公式表表头显式「1.4 的公式 / 命题」 scoped；6.2 L190/L214、7.3 L95、5.3 L172 等句内回指均由先行章节号锚定，目标存在。
- **跨维度指针**：models/training 章号 + § + 式逐一对照 51+39 章库存（含 8.1 L265 training 5.1-5.5 密集行、3.1 L350 models 4.1 §4.3-4.4 吸收形式 (11)-(14)、9.3 L218 training 8.5 §2/§4/§5/§7/§8）——全部存在。论文 § 引用（vLLM §4.x/§7.x、SGLang §3.1、Leviathan §3.x、DistServe §4.x/§6.x、Beyond-the-Buzz §3.x/§5.1、Llumnix §4.x/§6.x、Mooncake §4.4、Orca §4.2、FastServe §4.1、OpenRLHF §3.2）分类为外部引用，不入书内解析。
- **盘点表闭合**：W1-W6 全部收窄行经 errata 表执行并与现行文本核对；保留行（models 4.8 §2、models 1.1 §8 叙事句、models 1.2 §9 / 5.2 §3.4）与「扫描后保留」节一致；models/training 全量 inference 提及（含 references/README）扫描，未编号提及 28 处全部为叙事句 / 代码路径（inference/model.py、v1-inference.yaml）/ 公式标签（\text{inference}）/ PyTorch API（inference_mode）/ 论文引文，无漏收窄指针。

**判定：CLEAN（无悬空指针）。** 记法一致性项见 S3-1。

## 5. R2 文风均匀性 / 练习 / refs / 总结（逐波拉齐核对）

| 检查项 | 覆盖 | 结果 |
| --- | --- | --- |
| R2-1 节首 crux/设问/数值开场 | 37 章全部 ## 节 | 2 处事实式开场（1.4 §4 / 8.1 §6）裁定合规（R2-1 允许具体事实）；无「本节介绍」式开场 |
| R2-2 章脊口径锁定 | 37/37 | 全有（7.2 用「本章数值口径」变体）；换口径处随文声明 |
| R2-3 公式后 300 字内全数字实例 | 全 display 公式近邻代理扫描 + 跨波抽读（W1 2.2 / W2 4.3 / W3 6.1 / W4 7.3 / W5 8.3 / W6 9.4） | 模式均匀：公式→符号表→全数字代入；代理 flags 逐一裁定为窗口伪影（连续 display / 行内数学数字 / 推导链末端实例），无真实缺口 |
| R2-4 命题-证明骨架 + 逐章重启 | 37/37 | 命题 1..N 连续；证明（路线/要点）骨架；降级练习全部存在 |
| R2-5 ≥4 步推导路线句 | 路线句 census + 抽读 | 推导重章均有「路线：N 步——关键一步」；短证明用（要点）合规 |
| R2-6 定义配例 / 边界反例 | 抽读 | 命题反例密集（7.2 命题 1 raw-confidence 反例、4.3 命题 2 漏序列反例、7.4 命题 1 非 argmax 反例等） |
| R2-7 口径注 | 37/37 | 每章 7-59 处口径声明（（口径：…）/ 口径：/ **口径** 变体并存，实质均匀） |
| R2-8 术语判断句 | 37 章 | 「所谓 X，是指」为主；走读章（6.2/7.2/8.4）用代码锚 + 机制句定义，首现即定义实质满足（ZOS 首现 L91 即展开），记法随章型变异，不构成违规 |
| R2-9 逻辑关节词 | 抽读 | 若/则/因为/所以/否则/此时 均匀 |
| R2-10/11 提点/坑块引用 | 37/37 | 坑注 37/37（1-6 处/章，触发条件-现象-规避三段）；提点 26/37（无类比内容章 0 处合规） |
| R2-12 总结回指 | 37/37 | 每行含 式/§/练习/命题 回指（8.5 行 8「命题 5」为有效回指） |
| 练习四档 | 37/37 | 难度非递减、≥1 [10]/[20]、≤1 [40] 且 (a)(b)(c) 脚手架、计算/代码题校验锚、≥1 题以正文命题为证明目标（补全/闭环/推广/独立证明形均核到） |
| refs | 83 条 / 9 组 | 7 字段 schema、无竖线（含转义）、无混合档、日期合法、key 唯一、被引章节 forward（参考节 key）+ reverse（作者/arXiv 提及）交叉核 **0 缺口**；正文 arXiv ID 全部在 inference refs（lint 检查项 4，无 training-only 混入） |
| 篇幅 | 37/37 | 12 重章 5770-7847 ≥5500；9.2 2738 ≥2500；标准章 ≥3500 |
| lint | 37/37 | LINT OK |
| 反转义 raw 扫描（W5/W6 advisory 补偿） | 37 章 display+inline+codespan | 禁用形（\, \; \: \! \{ \} \% \# \| operatorname 等）**0 命中**；9.1 两处 \% 为 $`...`$ 保护形（合规）、US\$38 转义（合规） |
| W5/W6 修复保持 | 抽查 | 8.5 命题 4 方向（f_max=1−u ≥ 1−u_max）、9.3 sleep level 1 权重 offload 语义、8.2/8.5 反转义清除——均在位无回归 |

## 6. 残留项处置与工具观察

- **W4 非阻塞观察 2 项**：仍 open，本轮登记为 S3-2 / S3-3（收官顺手清）。
- **W5/W6 lint display 反转义漏检 advisory**：内容侧本轮 raw 扫描补偿完毕（0 命中）；`scripts/lint-math-boundary.py` 剥 display 后扫描的覆盖缺口仍在——建议后续工具波补 display 覆盖（本轮只读，未改脚本）。
- **W2 carryover kit cap() fill / .dia-cap CSS**：part03-09 局部提权在位，part01/02 未覆盖 → S3-4；kit/CSS 一次性全局修仍未落地。
- **W2 carryover fig-review harness slug@after 键撞 / richText metadata 链接渲染**：站点工具项，不影响本轮内容 census（build-content 按 title 双插正常），留站点波。

## 7. Verification log（防 rubber-stamp）

| 抽查项 | 方法 / 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- |
| 章库存 | Python 解析 H1/##/###/$$ 编号（37 inference + 51 models + 39 training） | 库存完整 | 127/127 章 |
| 指针解析 | 哨兵掩码 + 上下文裁定（pointers2.txt 174 flags 逐类核） | 0 悬空；flags 全为论文 § / spec § / 句内回指 / 解析器归因伪影（models 4.1 吸收形、training-scoped 行经目标库存独立验证存在） | 37 章全量 + 密集行逐处 |
| 跨维度闭合 | models/training 全文件 inference 提及扫描 + 盘点表/errata 表逐行对照 | 无漏收窄；保留行与记录一致 | models 51 + training 39 + 2 references + 2 README |
| MLA 几何 | 576/1152/46080/12960/17476 + 旧派生值全扫，43+ 命中逐处读上下文 | V4.1=512 无残留；V3 行全标注 | 37 章 + 9 图模块 |
| 符号 | b/B/bytes/𝒢/γ/α/λ/d/e/k/c/n_P/n_D/货币 定向扫描 + 声明核对 | 唯 S2-1（9.1 G） | 37 章 |
| 反转义 | raw display+inline+codespan 扫描（排除 \$ 转义与代码围栏） | 0 禁用形 | 37 章全量 |
| 图 census | node + 生产 classifyBlock + FIGURES 注册表 | 114 图 / 73 块全替换 / 0 兜底 / 2 documented keep | 37 章 + 9 模块 |
| 练习/总结/命题 | 逐章解析练习节/总结表/命题编号 | 全过 | 37/37 |
| refs | 表解析 + forward/reverse 交叉核 | 83 条 schema clean、被引 0 缺口 | 83/83 条 |
| lint | bash scripts/lint-chapter.sh 逐章 | LINT OK | 37/37 |
| 篇幅 | CJK 计数（剥围栏/公式） | 全达标 | 37/37 |
| W5/W6 修复保持 | 8.5 命题 4 / 9.3 sleep / 8.2-8.5 escape 定向 grep | 在位 | 3 项 |

**收官建议**：S2-1 修复（9.1 §7 G→n_GPU，纯符号替换不动数值）+ S3-1…S3-5 顺手清后，inference 维度可锁死 37/9 断言收官；站点侧 S3-4 与 kit/CSS 全局修可并入收官站点 QA 波。
