# Inference 维度 Spec 对抗审查报告 · Evaluator D（设计合理性 / 边界纪律 / 站点与 lint 方案）

- 日期：2026-11-22
- 审查对象：`docs/superpowers/specs/2026-11-22-inference-dimension-design.md`（全文 550 行，含两份附录与用户裁定区）
- 参照：`STYLE_GUIDE.md`（全文）、`docs/superpowers/specs/2026-11-16-training-dimension-design.md`（先例；该文件被 `.gitignore` 排除、仅存于主 checkout 工作目录，本审查以只读绝对路径调取 §4.1/§4.5 核对）、`docs/reviews/training-spec-review-d.md`（先例 D 路报告）、`docs/reviews/2026-11-20-textbook-style-sheet.md`、`docs/reviews/2026-11-15-style-audit/rubric.md`（存在性）、`docs/reviews/training-figure-census.md`、models/training 已发布章节（2.4 / 1.2 / 4.1 / 4.2 / 4.5 / 4.8、training 1.3 / 1.5 / 5.1 / 5.4 / 7.2 / 8.5 / 6.3）、站点代码（build-content.mjs / dimensions.mjs / index.astro / [dim] 三页 / DependencyGraph.astro / links.mjs / references.mjs / ascii-figure.mjs / fig-check.mjs / qa-shots.mjs / check-links.mjs / package.json）、`scripts/lint-chapter.sh` / `lint-math-boundary.py` / `rewrite-gate.py`、`inference/README.md`（现有占位）
- 职责边界：不做 arXiv ID↔标题 / repo 路径 / 生产数字的事实核查（归 A/B/C）；只查依赖图与波次、算术、三层边界与 recap 纪律、符号一致性、R2/图子循环协议自洽、§8/§4.5 方案与代码现状逐条对照。代码现状以本工作树为准。
- 结论：**0 S1 / 8 S2 / 8 S3 —— 暂不放行，S2 全部修复（或附证据驳回）后放行**。管线主体（波次制、评审闭环、图子循环、站点维度层复用）设计成立且与 training 先例对齐；问题集中在 recap 指针精度、与已发布 models/training 文本的边界/符号接口、§8 两处与代码不符的机制描述。

---

## Findings 表

### S2-1 · KV cache size 公式引用错号：models 2.4 式 (5) 实为式 (9)

- **位置**：spec §4.2（L99「KV cache size 公式 models 2.4 式 (5)」）、2.1 大纲（L200「KV cache 容量账 recap models 2.4 式 (5) + 指引」）
- **断言**：spec 两处把 KV 容量公式指向 models 2.4 式 (5)，但式 (5) 是「无 cache 朴素 decode 总 FLOPs」（$NT^2 + \frac{2}{3}LT^3d$），KV size 公式 $\mathrm{Mem}_{\mathrm{KV}} = 2 L H_{kv} d_{\mathrm{head}} T B b$ 是式 (9)（§5）。
- **证据**：`models/02-transformer-anatomy/04-causality-and-kv-cache.md` L83（式 (5) = FLOPs_naive）、L188（式 (9) = Mem_KV）、L212（「围绕式 (9) 做文章」）；models 4.1 L7 亦引「2.4 式 (9)」。
- **建议修正**：§4.2 与 2.1 两处「式 (5)」改「式 (9)」。此错号会随 recap 写进 2.1 正文与 metadata，并被 W0 指针收窄继承，必须在 spec 层纠正。

### S2-2 · 1.2「强度 ∝ B」立项为关键推导 + 强制重推导，与 models 已交付内容实质重复，且违反 §4.2 自身纪律

- **位置**：spec 1.2 大纲（L180-181 关键推导「强度 ∝ B；临界 B*…」）、§6.3（L434「1.2 强度 ∝ B + 临界 B*…」）、§4.2（L99「prefill / decode 算术强度 models 2.4 §6……不重复推导」）
- **断言**：decode batch 强度 $I \approx B$（权重主导区）及其饱和上界已由 models 1.2 §10 与 models 2.4 §6 式 (12) 完整交付；spec 一面把「models 2.4 §6 算术强度」列为 recap-only，一面把「强度 ∝ B」列为 1.2 关键推导与 evaluator 强制重推导项——内部矛盾 + 跨维度重复（training-spec-review-d S2-2「16N 分解」同型）。另：models 2.4 §6 已给出长上下文饱和上界（8B 满窗 ≈4.9 ≪ ridge ~10²），即长上下文下 decode 永不 compute-bound；1.2 的大纲未带此饱和存在条件，「到 compute-bound 临界 B*」的表述若不加限定会与 models 已发布结论冲突。
- **证据**：`models/01-foundations/02-compute-accounting.md` L247（「算术强度升到 ≈ B」）；`models/02-transformer-anatomy/04-causality-and-kv-cache.md` L239（式 (12) $I_{\mathrm{decode}}(t,B)$）、L242（「$B$ 小时……$I \approx B$」「强度饱和于……≈4.9」「batch 对强度的提升被 cache 读取封顶」）。
- **建议修正**：§4.2 recap 清单补「decode batch 强度 $I\approx B$ 与饱和上界（models 1.2 §10、2.4 §6 式 (12)）」；1.2 关键推导收窄为「临界 $B^{\ast}$（ridge 对齐，**含饱和上界 > ridge 的存在条件**）+ MoE 每 expert batch 与 EP 无关 + Wide-EP 每卡流量账」；§6.3 的 1.2 条相应改为「$B^{\ast}$ 存在条件与临界值 + MoE/Wide-EP 账」（强度 ∝ B 作 recap 忠实性核对，不作新推导）。1.2 真正的增量（$B^{\ast}$、MoE、Wide-EP、KV 几何对照表）不受影响。

### S2-3 · 1.3 roofline 定义与 prefill/decode 定位重复 models 2.4 §6 已交付的 roofline 入口，§4.2 recap 清单漏列

- **位置**：spec 1.3 大纲（L186「roofline 模型（峰值 FLOPS、HBM 带宽、ridge point = FLOPS/BW）；decode memory-bound / prefill compute-bound 的定位」、L187 关键推导「roofline 定位」）、§4.2 清单
- **断言**：roofline 模型定义（吞吐上限 = min(峰值, I×带宽)、ridge = 峰值÷带宽）与「prefill compute-bound / decode 全窗口 memory-bound」定位结论是 models 2.4 §6 的交付物（式 (10)-(12) + 明文本章交付声明）；1.3 把定义与定位重新列为内容与关键推导，未声明 recap。
- **证据**：`models/02-transformer-anatomy/04-causality-and-kv-cache.md` L216（roofline / ridge 定义）、L221（式 (10) $I_{\mathrm{prefill}}\approx T_p$）、L234（「memory-bound 的结论在整个窗口内成立」）、L244（「本章交付的是式 (10)–(12)……那两个维度全部讨论的入口」）。
- **建议修正**：§4.2 清单补「roofline 入口与两阶段定位（models 2.4 §6 式 (10)-(12)）」；1.3 内容改「roofline 入口与定位 recap 一句 + 指引 → 本章增量：逐卡 datasheet 参数表（见 S2-4 边界裁定）、batch 三方权衡曲线、排队论（M/M/1 / Kingman）、SLO 操作点」；关键推导「roofline 定位」改「 datasheet 参数代入的定量操作点」。

### S2-4 · 已发布 models/training 指针把 inference 要讲的主题判给 gpu/cluster：三处归属冲突未入 errata 清单，且 §2 errata 示例章号有误

- **位置**：spec §2 errata 通道（L45）、1.3 硬件参数表（L186）、7.4 EPLB（L348）
- **断言**：(a) models 2.4 §6 明文「逐卡数字与测量方法归 gpu 维度」「roofline 的逐卡参数属 gpu 维度」，而 inference 1.3 交付 H100/H200/B200/Ascend 950 逐卡峰值与带宽表——spec §2 只把「roofline 硬件参数的**微架构来源**」判给 gpu，比已发布指针窄，两者冲突。(b) models 4.5 §7 与 training 5.4 §4 均明文「EPLB（的系统实现）归 cluster 维度」，而 inference 7.4 交付 EPLB 算法 + serving 部署（冗余副本 / 统计窗 / 重分配 / moe_lb 官方数字）——第二处冲突。(c) §2 errata 示例「models 4.5 §7」有误：models 4.5 泛指向 inference 的指针在 **§6**（MTP 自投机，L309）；§7 是 EP 接口节，已被 training errata 收窄为「训练侧归 training 5.3/5.4、硬件侧归 cluster」，不含 inference 泛指针。(d) §2 errata 行文「把泛指向收窄到具体章号」未显式覆盖「把 gpu/cluster 归属句重判为三层划分」这类修订（training 先例 S2-5 的修复恰恰是这种重判）。
- **证据**：`models/02-transformer-anatomy/04-causality-and-kv-cache.md` L216 / L244；`models/04-modern-architectures/05-moe.md` L321（§7 标题）、L327 / L345（EPLB 归 cluster）、L258（§6 MTP 标题）、L309（系统侧归 inference）；`training/05-model-parallelism/04-expert-parallelism.md` L79（「EPLB 一类的系统实现归 cluster 维度」）；`docs/reviews/training-spec-review-d.md` S2-5（errata 通道先例）。
- **建议修正**：§2 三层划分显式补两句裁定——「serving 账所用的 datasheet 参数表（官方规格事实）归 inference 1.3，微架构来源与测量方法归 gpu」「EPLB 的 serving 侧部署与算法归 inference 7.4，硬件/网络拓扑归 cluster」；errata 行文扩为「仅限指针句的最小修订（泛指向收窄到具体章号，**或按三层划分重判归属句**）」；W0 盘点清单显式列入 models 2.4 §6（gpu 句）、models 4.5 §7 与 training 5.4 §4（EPLB 句）三处目标修订句；§2 示例「models 4.5 §7」改「models 4.5 §6」。

### S2-5 · §4.3 符号表与 models/training 已发布符号三方冲突（b / t / d / γ / α），修订注的「先例一致」声称不实，且无 recap 翻译规则

- **位置**：spec §4.3 符号表与消歧硬规则（L104-118）、修订注（L120）、§4.2 recap 清单
- **断言**：inference 定 $b$ = block size，但 **models 的 $b$ = 每元素字节数**（2.4 式 (9) 与逐项账）、**training 的 $b$ = micro-batch 序列数**（5.1 通信账 $2bsh$/$8bshL$、7.2/1.5 练习「$b=4$、$d=256$」）——三个维度三种 $b$。修订注称「bytes 一律文字 / $\mathrm{bytes}$……与 training 全书先例……一致」不实：training 并非「bytes 不占字母」，而是 $b$ 被 micro-batch 占用、bf16 字节数写成系数 2。冲突落在 spec 强制 recap 的接口上：2.1 recap models 式 (9)（$b$=bytes）同章引入 $b$=block size；8.1 recap training 5.1 通信账（$b$=micro-batch）同章用 $t$=TP 度。另有 models 2.4 式 (11)/(12) 的 $t$=已缓存长度、$d$=hidden 维，与 inference $t$=TP 度、$d$=DP 度冲突；training 5.4 §4 偏置控制器用 $\gamma$=更新步长、$\alpha$=seq 级 aux 权重，与 inference $\gamma$=draft 长度、$\alpha$=接受率冲突（7.4 recap training 5.4 时触发）。§4.3 只消歧 inference 内部，未给跨维度 recap 翻译规则。
- **证据**：`models/02-transformer-anatomy/04-causality-and-kv-cache.md` L188 / L199（$b$ 每元素字节数）、L229-232（$t$ 当前已缓存长度）、L226（$4Ltd$ 的 $d$=hidden）；`training/05-model-parallelism/01-tensor-parallelism.md` L3 / L188（「$2$：bf16 字节数」，$bsh$ 为激活体量）、`training/07-pretraining/02-recipes.md` L162 与 `training/01-autograd-and-backprop/05-gradient-accumulation.md` L182（$b$=micro-batch）；`training/05-model-parallelism/04-expert-parallelism.md` L79（$\gamma=10^{-3}$、$\alpha=10^{-4}$、$b_i$ 偏置）。正面先例：training 5.4 metadata 把 models 的「$2kd$ 元素（bf16 合 $4kd$ bytes）」翻译成本维度口径再 recap——翻译规则有书内成例。
- **建议修正**：§4.3 修订注更正先例声称（models $b$=bytes/元素、training $b$=micro-batch，inference $b$=block size 是**有意的维度局部第三赋值**）；加一条硬规则：「recap models/training 公式时一律把兄弟维度符号翻译为本维度约定（models $b$→$\mathrm{bytes}$、models $t$(缓存长)→$T$ 或声明局部记号、training $b$(micro-batch)→$B$ 并声明口径、training 5.4 $\gamma/\alpha$→文字），附一句口径注；禁止在 inference 章内沿用兄弟维度字母的兄弟语义」。混场章示例可补 2.1 / 8.1（现列表 8.1/7.4/9.1 按「并行度数与 batch 同章」定义，无误但不覆盖 recap 接口）。

### S2-6 · §8 站点方案与代码现状不符：不存在 `live: true` 开关；W1 上线必需完整 DIMENSIONS.inference UI 对象（缺失即构建崩溃）；FORTHCOMING.inference 旧窄范围文案未列入更新

- **位置**：spec §8（L484「落地页 inference 行 `live: true` + 状态翻转」、L485 只列 DependencyGraph 几何入 dimensions.mjs）
- **断言**：(a) 代码无 `live` 字段——落地页 live 状态由 manifest 章数自动驱动（`index.astro` L26-28 `const live = chapterCount(key) > 0`），`[dim]/index.astro` getStaticPaths 同（L7-11）；spec 描述的「设 live:true」机制不存在。(b) W1 一旦有 inference 章进 manifest，`[dim]/index.astro` L15 `const cfg = DIMENSIONS[dim]` 与 L22 `cfg.scope`（无兜底）、`index.astro` L134 `DIMENSIONS[d.key].depNoteShort`（对象本身无兜底）、`DependencyGraph.astro` L9 `DIMENSIONS[dim].graph`（无兜底）都要求 dimensions.mjs 存在**完整** DIMENSIONS.inference 对象（zh / orderLabel / scope / lede / ledgerNote / depNoteShort / depNote / prereqs / prereqNote / firstChapter / graph）——§8 只提图几何与 live 翻转，未列 UI 对象为必交付项，照字面执行 W1 站点构建即 TypeError。(c) `dimensions.mjs` L96 FORTHCOMING.inference 仍是旧窄范围「continuous batching、PagedAttention、speculative decoding」；若 W1 时 DIMENSIONS.inference 缺位，落地页行按 L29 兜底显示该窄文案（且随后 L134 崩溃）；spec 未提替换/移除该占位。DIMENSION_ORDER 已含 'inference'（L92），无需改动——spec 未声称需改，此点相符。
- **证据**：`site/src/pages/index.astro` L26-29 / L134；`site/src/pages/[dim]/index.astro` L7-11 / L15 / L22 / L39；`site/src/components/DependencyGraph.astro` L9-10；`site/src/lib/dimensions.mjs` L8-90（training 完整对象先例）/ L92 / L95-99。builder-check（spec L545）已把 UI 文案委派 builder 从 spec 撰写——部分缓解，但 §8 文本仍与机制不符且漏列必交付对象。
- **建议修正**：§8 路由/落地页条改写为：「live 状态由 manifest 章数自动翻转（无 live 开关）；W0 在 dimensions.mjs 建完整 DIMENSIONS.inference UI 对象（字段照 training 先例，文案由 builder 从本 spec §1/§2/§5 撰写，graph 几何随 W0 README 同步产出）；同时移除 FORTHCOMING.inference 占位（其窄范围文案已过时）；DIMENSION_ORDER 已含 inference 无需改」。

### S2-7 · §4.5 STYLE_GUIDE 泛化清单漏「序言维度级细化豁免扩至 inference」：§4.1 的 [A] 口径与 STYLE_GUIDE §6 在 spec 自身冲突规则下成死结

- **位置**：spec §4.1（L87「[A] = 正式 paper + official code」）、§4 序言（L81「与 STYLE_GUIDE 冲突处以 STYLE_GUIDE 为准」）、§4.5（L130 泛化清单：标题与适用范围行 / §10 指针 / §11 usage）
- **断言**：STYLE_GUIDE §6 [A] = paper + repo + **open weights 三者齐**（L99）；其序言（L3）的维度级细化豁免**只点名 training**（「training 的维度级细化——证据分级对方法/框架锚点的口径……以该 spec §4 为准」）。inference §4.1 把 [A] 放宽为 paper + code（无 weights），按 spec 自己的「冲突以 STYLE_GUIDE 为准」规则，全部 [A] 引擎锚点（vLLM / SGLang / FlashInfer / GPTQ / AWQ……）在逐字读 STYLE_GUIDE 的 fresh-context evaluator 面前都是 S2 约定违规。training 先例靠 W0 在序言加豁免句闭合（training-spec-review-d S3-6 明记「靠序言……兜底——可接受」），但 inference §4.5 的泛化清单照抄 training §4.5 的三项枚举，未含序言豁免扩展——W0 照单执行则死结留到 W1。
- **证据**：`STYLE_GUIDE.md` L3（豁免只列 training）/ L99（[A] 三者齐）；training spec（主 checkout）§4.1 L83（「[A] = 正式 paper + official code（若锚点是模型则另需 open weights）」）；`docs/reviews/training-spec-review-d.md` S3-6（序言兜底成例）；spec L130（三项枚举）与 L139（「fresh-context evaluator 逐字读该文件」）。
- **建议修正**：§4.5 清单加第四项：「序言维度级细化豁免句扩至 inference（证据分级对方法/框架锚点的口径、跨维度引用规则、符号约定表以本 spec §4 为准），与 training 豁免句并列」。§4.1 可顺带补 training 式括号注「（若锚点是模型则另需 open weights）」——本维度唯一模型锚点 V3 满足三者齐，V4.1 走 [W]，无实际影响。

### S2-8 · 机制章代码走读与引擎走读章重访同一批文件，两层深度划分 / recap 纪律未声明（2.1↔5.2 文件清单逐字相同）

- **位置**：spec 2.1（L200）vs 5.2（L290）；4.1（L252）/ 4.3（L264）vs 5.2（L290）/ 6.2（L316）；8.4（L374）vs 6.3（L322）/ 7.4（L348）
- **断言**：2.1「vLLM V1 代码走读：block_pool.py、kv_cache_manager.py、kv_cache_coordinator.py、single_type_kv_cache_manager.py（hybrid KV 协调）」与 5.2「KV manager 走读（同四文件 + interface/layout，hybrid KV 协调 ↔ 2.1）」文件清单与主题逐字重合；4.1/4.3 的 scheduler/chunked-prefill 走读与 5.2/6.2 同模块重访；8.4 的 SGLang PD / DeepInfer VE/P/D 与 6.3 / 7.4 已走读内容重合。「机制章取例证 → 引擎章全量走读 + 公式逐行对照」的两层设计本身成立（§5「引擎走读立于机制之上」），但 spec 没有任何条款声明两层的取材边界与重访 recap 纪律——§4.2 的 recap 纪律只覆盖 models/training。W3 写 5.2 时 2.1 已发布，写作者要么重复走读、要么交付不足，evaluator 无判定基准。对照：6.2↔3.4 有显式回链、7.4↔1.2 有「伏笔兑现」、8.1↔7.4 有「recap + 指引」——说明 spec 有点对点的标记习惯，缺的是一条总则。
- **证据**：spec L200 vs L290（四文件逐字相同）；L252 / L264 vs L290 / L316；L374（8.4「SGLang disaggregation……↔ 6.3」「DeepInfer VE/P/D……7.4 回链」）vs L322 / L348。先例：training-spec-review-d S2-2/S2-3/S2-4 对重复面均要求显式 recap 声明。
- **建议修正**：§4.2 加一条维度内纪律：「机制章（P1-P4）代码取材限于机制例证片段（大纲声明片段范围）；引擎走读章（P5-P7）做模块全量走读 + 公式逐行对照，重访机制章已展示片段一律 recap + 指引；对照章（8.4 / 9.2）只 recap 各系统实现要点 + 指针，增量限定为对照表与选型论断」。5.2 / 8.4 大纲各补一句「2.1 / 4.x / 6.3 / 7.4 已走读片段 recap + 指引」。

### S3-1 · 前瞻引用计数「三处」少计一处（4.4→7.2）；个别目标章大纲未显式携带回链义务

- **位置**：spec §5 图后说明（L169「三处章级前瞻引用」）、4.4（L270「前瞻 7.2 + 回链」）、3.2 大纲、7.1 大纲、8.1 大纲
- **断言与证据**：(a) 4.4 亦用 DSpark 概念于定义章之前（L270），7.2 回链已覆盖「4.4 / 4.5」（L336）——闭合完整，但「三处」应为四处（2.3→3.2、4.4→7.2、4.5→7.2、1.4→9.4）。(b) 2.3→3.2 的回链义务写在 §5 说明（L169「回链 2.3」）与 2.3 大纲（L212），3.2 大纲本身未显式写「回链 2.3」（7.2 / 9.4 大纲均显式写了回链，体例不一）。(c) 6.2 承诺「7.1 回链」（L316，async scheduling 同源对照）、3.4 承诺「为 6.2 / 7.1 埋伏笔」（L244），但 7.1 大纲无 async scheduling 条目与 6.2 回链。(d) 5.3 / 6.3 前向指针「↔ 8.1 公式」，8.3 / 8.4 有显式回链（↔ 5.4 / 6.3），8.1 大纲未显式回链 5.3 / 6.3（锚点行含 vLLM/SGLang 分布式实现，隐含覆盖）。
- **建议修正**：计数改「四处」并列入 4.4→7.2；3.2 大纲补「格式账回链 2.3」；7.1 大纲补「async scheduling（仿 SGLang/NanoFlow）↔ 6.2 回链」；8.1 大纲补「5.3 / 6.3 分布式实现回链」。均为大纲一行字，不改交付物。

### S3-2 · §8 两处工作项描述与代码不符（references 非条目字段；links.mjs 无需扩展）；W0 起步断言值与滚动序列未显式给出

- **位置**：spec §8（L483「references 文件」、L486「links.mjs 扩展」、L483 断言滚动只给 W1=8/2 与 W6=37/9）
- **断言与证据**：(a) DIMENSIONS 条目实际字段为 key / parts / asserts（build-content.mjs L41-74），references 路径由 key 派生硬读（L194 `readFileSync(path.join(SRC,'references.md'))`，无 try/catch）——「references 文件」不是条目字段（spec 沿用了代码 L38 注释的宽松说法）；功能无差，但 W0 顺序约束（references.md 必须先于条目存在）值得显式写出，spec W0 行（L467）的枚举顺序已隐含正确。(b) links.mjs 已维度泛化（L45-46 `../references.md` → `/<dim>/references/`；L48-51 跨维度 README；L53-58 章链），无需「扩展」，builder-check（L545）亦确认「已维度泛化」——§8 措辞改「覆盖核对」。(c) 断言逻辑支持 W0 起步 {0,0,0}（Part 目录缺失 try/catch 跳过 L94-101；===/===/>= L199-205；空骨架 parseReferences → groups=[] → refCount 0），但 §8 未显式给 W0={0,0,refsMin:0} 与中间滚动值 W2=17/4、W3=24/6、W4=28/7、W5=33/8（builder-check 把 refsMin 委派 builder，chapters/parts 可由「当波应发布章数」推出）。
- **建议修正**：§8 条目描述改「key / 9 Part parts 表 / 断言组；references.md 置于 inference/references.md（路径由 key 派生、构建时硬读，须先于条目存在）」；links 条改「已维度泛化，W0 仅覆盖核对」；断言条补 W0 起步值与完整滚动序列。

### S3-3 · §6.6「ascii-figure 兜底应为零」未区分结构图与 algo/grid 深色图版，与管线分类及 training census 先例有字面冲突

- **位置**：spec §6.6 步骤 4（L459）、§8 图条目（L487「ascii-figure 兜底归零」）
- **断言与证据**：classifyBlock 三分类 diagram / table / code（ascii-figure.mjs L249-253），伪代码/目录树/纯表格按设计保留 algo/grid 深色图版（build-content.mjs L123 注释）；training census 先例明文「低 = 伪代码 loop / 模板文本 / 目录树 / 纯表格，维持现有 algo / grid 深色图版即可，不建议注册场景图」（training-figure-census.md L10）。§6.6 步骤 1 限定「ASCII 结构图块」，步骤 4 却写「全 37 章 ASCII 块 census……兜底应为零」——字面读会要求把伪代码面板也替换成场景图。
- **建议修正**：步骤 4 改「全 37 章 **diagram 类** ASCII 结构图块 census，无残留未注册块（结构图通用兜底归零）；algo/grid 深色图版按 census 优先级规则豁免（同 training 先例）」。

### S3-4 · §4.3 统一表未收 E（expert 总数）/ k（top-k）/ c（CP 度），E/e 大小写对正是该表要防的混淆类

- **位置**：spec §4.3 表（L106-117）、1.2（L180 $Bk/E$）、7.4（L348 $E/e$、384 routed）、8.1（L356 CP serving、决策元组 t/p/e/d）
- **断言与证据**：表已占 $e$=EP 度，但核心公式里的 $E$（expert 总数）与 $k$（top-k）未声明；$E/e$ 与 $e$ 的大小写对是 serving 文献典型混淆点（表的存在理由）。CP 度在 training 用 $c$（5.5 world = t×c×p×e×d），8.1 讲 CP serving 但表与决策元组均未收 $c$。各章自带符号表（STYLE_GUIDE §3）可局部定义，但统一表漏收跨章复用的度数/几何符号削弱其消歧功能。
- **建议修正**：表补三行：$E$（expert 总数，与 $e$=EP 度区分）、$k$（MoE top-k）、$c$（CP 度，沿用 training；若 8.1 决策元组不含 CP 则注明「CP 仅 LoongServe 对照，不入默认决策元组」）。

### S3-5 · 4.4 大纲公式草记号两处待规范（E[α] 重载、加速比量纲），移交 A 路数学核查确认

- **位置**：spec 4.4（L270）
- **断言与证据**：(a)「期望接受长度 $E[\alpha] = (1-\alpha^{\gamma+1})/(1-\alpha)$」——§4.3 定 $\alpha$=接受率（常数），$E[\alpha]$ 字面是「接受率的期望」，与几何级数和语义不符（Leviathan 原文记期望接受 token 数，非 $E[\alpha]$）。(b)「walltime improvement factor：$(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c_d + c_v))$ 形式」——若 $c_d/c_v$ 为绝对单步成本（§4.3 定义），该式量纲是 token/成本而非无量纲加速比；Leviathan 定理的 improvement factor 用成本比 $c=c_d/c_v$，写作 $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$，或绝对成本式需乘 $c_v$。spec 有「形式」hedge 且 §6.3 强制重推导会在写作期兜住，故 S3；但大纲作为 writer 输入宜先规范。
- **建议修正**：期望接受长度改记 $E[\tau]$ 或 $\bar{n}_{\mathrm{acc}}$（符号表补一行）；加速比注明「$c_d/c_v$ 按 verify 成本归一（$c=c_d/c_v$）」或补 $c_v$ 因子。请 A 路在数学核查中确认 Leviathan 定理口径后落定。

### S3-6 · 附录「25 repo」声称与枚举 24 个不符

- **位置**：spec §5 引言（L143「25 个关键 repo 经 GitHub API 预核实」）、附录（L528「GitHub 路径核实（25 repo……）」）
- **断言与证据**：逐个数附录枚举：vllm / sglang / DeepInfer / TensorRT-LLM / lmdeploy / nixl / dynamo / LMCache / Mooncake（9）+ deepseek-ai 括号 8 个（FlashMLA/DeepGEMM/DeepEP/EPLB/3FS/DeepSelect/TileKernels/deepseek-recipe）+ tilelang / flashinfer / flash-attention / xgrammar / outlines / guidance / marlin（7）= **24**。§4.1 提到的 IST-DASLab/gptq、mit-han-lab/llm-awq、SafeAILab/EAGLE 不在枚举内（EAGLE 仍 ⟨verify⟩）。
- **建议修正**：补录漏计 repo 或改「24 repo」；若 gptq/llm-awq 已核实则补入枚举。

### S3-7 · §9 风险 4 的 S2 触发词表（warp/SASS/tensor core）与 spec 自身大纲的策略级/披露级提及存在字面张力

- **位置**：spec §9 风险 4（L495「出现 warp / SASS / tensor core 指令级内容即 S2」）、3.1（L226「warp partitioning、async warp specialization——算法与策略级」）、7.3（L342「缺一个手调 SASS 优化 → decode 慢 5-10%」官方披露）
- **断言与证据**：§2 边界与 3.1 大纲明确策略级（FA3 warp specialization 属调度策略）与披露归因（SASS 差距是官方披露事实、非指令级教学）合法；但风险 4 的触发句以关键词列举，keyword 式 evaluator 可能误伤 spec 自己大纲里的合法提及。
- **建议修正**：风险 4 补一句「策略级调度描述（3.1 FA3 warp specialization）与官方披露归因（7.3 SASS 差距口径）不属指令级内容，不触发」。

### S3-8 · §4.1 证据分级对「无 paper 无 repo 的开放技术博客」未给等级口径（FlashDecoding / batch-invariance）

- **位置**：spec §4.1（L92 老经典条只要求「标注原始出处并注明无 arXiv」，未给等级）、3.1 锚点（L228「FlashDecoding 博客（无 arXiv）」无等级）、1.4 / 3.4（batch-invariance 博客 ⟨verify⟩ 无等级）
- **断言与证据**：[B] 需 paper、[C] 定义为闭源披露、教学参照定义为无 paper 的教学 repo——FlashDecoding（CRFM 博客，无 paper 无独立 repo）与 Thinking Machines batch-invariance 博客三者都不贴合；STYLE_GUIDE §7 要求 metadata 锚点含证据等级，写作期这些锚点无规可依。DeepInfer 生产数字已有「教学参照 + 官方实现 + 官方披露口径注」成例可仿。
- **建议修正**：§4.1 老经典条补等级口径，例如「无 paper 无 repo 的官方/机构技术博客：[C] 级并注明『开放技术博客、非闭源』，或挂教学参照 + 出处注（与 DeepInfer 官方披露口径同构）；二选一全维度统一」。

---

## Verification log

| # | 检查项 | 依据（文件:行 / 引句） | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| 1 | 依赖图无环 + 边枚举 | spec L147-167 mermaid：18 边（P1→P2/P3/P4；P2/P3→P4；P2/P3/P4→P5；P2/P4→P6；P2/P3/P4→P7；P5/P6/P7→P8；P7/P8→P9），全部 i<j | 无环（P1..P9 即拓扑序）；18/18 边与章级依赖语义相符（P2→P5 KV 管理、P3→P5 backend、P4→P5 调度等） | 18/18 边逐条 |
| 2 | 波次序尊重依赖 | spec §7 L465-474：W1=P1+P2、W2=P3+P4、W3=P5+P6、W4=P7、W5=P8、W6=P9 | 跨波边全部严格前序（P5 需 P2/P3/P4 ⊂ W1+W2 ✓；P8 需 P5/P6/P7 ⊂ W3+W4 ✓；P9 需 P7/P8 ⊂ W4+W5 ✓）；波内边 P1→P2、P3→P4 由 builder-check L545「波内并行写作依赖（training W1 先例、固定大纲）」覆盖 | 9 Part × 6 波全映射 |
| 3 | 前瞻引用 + 回链闭合 | 2.3→3.2：spec L169/L212（「格式账正式交付在 3.2……回链」）；4.5→7.2：L276/L336（「4.4 / 4.5 回链」）；1.4→9.4：L192/L406（「1.4 / 5.4 / 6.1 回链」）；4.4→7.2：L270/L336 | 三处指定前瞻均有回链且方向指向后波（W1→W2、W2→W4、W1→W6）✓；另发现第四处 4.4→7.2 已闭合但计数漏 → S3-1；3.2/7.1/8.1 大纲回链显式度不一 → S3-1 | 4/4 前瞻对 + 5.3/6.3→8.x 指针类核对 |
| 4 | Part/章数算术 | spec §3 L63-71 目录注释 vs §5 各 Part 标题 vs 大纲逐章枚举 | 4+4+4+5+4+3+4+5+4=37 ✓，三处一致；P6=3 章（6.1-6.3）✓ | 37/37 章逐一点数 |
| 5 | 波次算术 + 滚动断言 | spec §7 表：8+9+7+4+5+4=37 ✓；W1=8/2 Part、W6=37/9 与累积序列 W2=17/4、W3=24/6、W4=28/7、W5=33/8 自洽 | 算术 ✓；中间值未显式列出 → S3-2(c) | 6 内容波全量 |
| 6 | 12 重章 ↔【重】标注 ↔ §6.3 清单 | §4.4 L124 清单（1.2/2.1/2.4/3.1/3.2/4.3/4.4/6.2/7.2/8.1/8.2/9.1）vs 大纲【重】标注（L179/199/217/225/231/263/269/315/335/355/361/387）vs §6.3 L434-441 | 12/12/12 三方逐一对上 ✓；短篇 9.2 标注一致（L124/L393）✓；§6.3 各章子集 ⊆ 该章关键推导（2.4/3.2/4.3/4.4/8.1/8.2/9.1 为真子集，合法）✓ | 12/12 重章 + 37 章【重】扫描 |
| 7 | 边界：算子 vs gpu | spec §2 L38（算法/tiling/IO/接口 in，指令级 out）；3.1 L226「算法与策略级，指令级归 gpu」；3.3 L238「lop3/bfe 归 gpu」；7.3 L342 边界声明；§9 风险 4 L495 | 三层划分声明完整、各算子章均有边界句 ✓；风险 4 关键词触发与策略级提及的字面张力 → S3-7 | P3 全 4 章 + 7.3 |
| 8 | 边界：分布式 vs cluster | spec §2 L39（传输语义/带宽账/系统设计 in，RDMA/NVLink/NCCL 硬件 out）；3.3 DeepEP「语义与账，硬件归 cluster」；8.3 L368 边界声明；GDR 在 §2 显式 in-scope | serving 侧分布式与硬件层划分一致 ✓；8.1「无 backward」增量定位与 training P5 recap 衔接 ✓ | 3.3 / 8.1 / 8.3 / 8.4 |
| 9 | 边界：V4.1 模型侧 | spec §2 L40 / §4.2 L100（config 引用 + 一句 recap + 指针，禁推导）；逐章核对 1.2 L180 / 2.1 L200 / 3.1 L226 / 3.2 L232 / 4.5 L276 / 7.1 L330 / 7.2 L336 / 7.4 L348 / 9.4 L409 | 全部 V4.1 触点均为 config 字段 + serving 含义，7.1/7.2/7.4/4.5 带显式「不推导/模型侧归 models」标记 ✓；EAGLE/Medusa draft 结构属投机系统（§2 in-scope）非目标模型架构，不构成泄漏 ✓；无 scope leak | 9/9 V4.1 触点章 |
| 10 | recap 目标存在性与章号 | models 2.4=04-causality-and-kv-cache.md ✓、4.1 §4=MLA（L64）/§4.3-4.4 absorption（L164/182）✓、4.2 §6=DSA（L193）/§8=质量风险（L262）✓、4.5=05-moe.md ✓；training 1.3 §6=V3 FP8 128×128（metadata L3）✓、5.1 TP f/g + GQA/MLA（metadata L3）✓、5.4 负载不均 §4（L75）✓、5.5=combining ✓、7.3=stability ✓、8.4=GRPO ✓、8.5=RL systems ✓ | §4.2 清单章号/节号全部命中 ✓，唯式 (5) 错号 → S2-1；§2 errata 示例 models 4.5 §7 应为 §6 → S2-4(c) | §4.2 清单 10 项逐条 + §2 示例 5 处 |
| 11 | recap 纪律 vs models 已交付 | models 1.2 L247（I≈B）、2.4 L216-244（roofline 入口 + 式 (10)-(12) + I≈B + 饱和上界 4.9 + 交付声明） | 1.2「强度 ∝ B」与 1.3「roofline 定位」重复已交付内容且 §4.2 漏列 → S2-2 / S2-3；1.2 的 B* / MoE / Wide-EP 与 1.3 的排队论 / 三方权衡为真增量 ✓ | models 1.2 §10 + 2.4 §5-6 全文核对 |
| 12 | 已发布指针归属冲突 | models 2.4 L216/L244（roofline 逐卡参数归 gpu）；models 4.5 L327/L345 + training 5.4 L79（EPLB 归 cluster）；spec 1.3 参数表 / 7.4 EPLB | 三处冲突 → S2-4；errata 通道机制存在（training S2-5 先例）但行文与示例清单未覆盖 → S2-4(d) | models/training 全库「归 inference」指针 grep + gpu/cluster 归属句抽查 |
| 13 | errata 通道事实基础 | 泛指针实测：models 1.1 L270（含 ../../inference/README.md 活链接，links.mjs L48-51 可重写 ✓）、1.2 L155/187/247、2.4 L199/212/242/244/256、4.1 L7、4.5 L309（§6）、4.8 L66/96/112、training 8.5 L46/52、6.3 L87/91 | W0 盘点有实据 ✓；现有 inference/README.md 占位（窄范围）W0 重建后链接仍解析 ✓；「仅限指针句、不动公式数值」+ rewrite-gate 豁免（builder-check L545，xref-multiset 会误报）自洽 ✓ | 11 个文件 20+ 处指针 |
| 14 | §4.3 内部一致性 | 修订后表（L106-118）：B/b/bytes/γ/λ/𝒢/P-D 无双重赋值 ✓；𝒢 避 G（training GRPO 专占）✓；\mathcal 在白名单（STYLE_GUIDE L62）✓；λ 到达率声明规则 ✓；1.2 大纲已用 \mathrm{bytes}（L180）✓ | 内部自洽 ✓；E/k/c 漏收 → S3-4；4.4 E[α]/加速比草记 → S3-5 | 表 9 行 × 消歧 5 条 × 大纲符号触点 |
| 15 | 跨维度符号冲突 | models 2.4 L188/L199（b=bytes/元素）、L229-232（t=缓存长）、L226（d=hidden）；training 5.1 L3/L188（2bsh，2=bf16 字节数，b=micro-batch）、7.2 L162 / 1.5 L182（b=micro-batch）、5.4 L79（γ/α=偏置参数、b_i）；spec L120 修订注「与 training 全书先例一致」 | 三方 b 冲突 + t/d/γ/α recap 接口冲突，修订注先例声称不实 → S2-5；翻译规则有书内成例（training 5.4 metadata 把 models 2kd·b 译为 4kd bytes） | models 2.4 / training 5.1 / 5.4 / 7.2 / 1.5 符号段全核 |
| 16 | DIMENSIONS 条目格式 | build-content.mjs L41-74：{key, parts:[{num,dir,titleEn,titleZh,tagline,wave}], asserts:{chapters,parts,refsMin}}；L38 注释「key / 源目录 / PARTS 表 / references 文件 / 断言组」；references 路径 key 派生 L194 | spec §8「key / 9 Part parts 表 / references 文件 / 断言组」与注释口径一致、与字面字段差一项（references 非字段）→ S3-2(a)；功能等价（inference/references.md 正是硬读路径） | 条目全字段 + models/training 双先例 |
| 17 | W0 起步断言可行性 | build-content.mjs L94-101（Part 目录缺失 try/catch 跳过、files=0 continue）→ chapterCount=0 / parts=0；L199-205（===/===/>=）；references.mjs L9-44（空骨架 → groups=[] → refCount 0）；L194 references.md 硬读无 try/catch | {0,0,refsMin:0} 可通过 ✓；references.md 必须先于条目存在 ✓（spec W0 行 L467 枚举顺序：references 骨架在 DIMENSIONS 接入前 ✓）；W0 时 manifest.dimensions.inference.stats.chapters=0 → 落地页 forthcoming、[dim] 不生成 inference 页（getStaticPaths 过滤）✓ 无崩溃 | 断言三分支 + 空骨架解析路径 |
| 18 | DIMENSION_ORDER / FORTHCOMING | dimensions.mjs L92（ORDER 已含 inference ✓）；L96 FORTHCOMING.inference scope=「continuous batching、PagedAttention、speculative decoding」（旧窄范围，与 inference/README.md 占位同源） | ORDER 无需改 ✓；FORTHCOMING 窄文案过时、spec 未列更新 → S2-6(c) | 两常量逐字 |
| 19 | live 机制与 UI 对象必需性 | index.astro L26-28（manifest 驱动 live，无 live 字段）/ L29（DIMENSIONS[key] ?? FORTHCOMING 兜底）/ L134（DIMENSIONS[d.key].depNoteShort 无对象兜底）；[dim]/index.astro L15/L22（cfg 无兜底）；DependencyGraph.astro L9（graph 无兜底）/L10（parts 有兜底，建设期节点不可点 ✓） | spec「live:true + 状态翻转」机制不存在；W1 必需完整 DIMENSIONS.inference 对象否则三处 TypeError → S2-6(a)(b)；「DependencyGraph 维度化已落地」声称属实 ✓ | 落地页 + [dim] 三页 + 组件全链路 |
| 20 | links.mjs 泛化 | links.mjs L45-46（../references.md → /<dim>/references/，dim 参数化）/ L48-51（../../<dim>/README.md → GitHub，任意维度）/ L53-58（章链）；build-content L118-121（重写残留断言） | 已维度泛化、inference 零改动 ✓；spec「扩展」措辞 → S3-2(b)；builder-check L545 同结论 ✓ | 三条重写规则 + 残留断言 |
| 21 | lint check-4 与 §4.5 | lint-chapter.sh L2 usage `<models\|training>/...`、L12-16 case（models/* / training/* / 默认 refs="" 跳过）、L17-22 文件存在 + arXiv 条目检查；lint-math-boundary.py grep models/training/inference 零命中 | spec §4.5「case 分支 + inference/* + usage 同步 + 51+39 回归」与代码逐字对上 ✓；无分支时 inference 章静默跳过检查 4，W0 加分支为强制前提 ✓；lint-math-boundary「不变」正确 ✓ | check-4 全分支 + 8 项检查扫读 |
| 22 | STYLE_GUIDE 泛化清单 vs 现状 | STYLE_GUIDE L1 标题（两维度）/ L3 序言豁免（只 training）/ L5 适用范围行 / L99 [A] 三者齐 / L152 §10 指针 / L168 §11 usage；training spec §4.5（主 checkout L115-119，三项枚举）；training-spec-review-d S3-6（序言兜底成例） | §4.5 三项枚举与文件结构对上 ✓，但漏序言豁免扩展 → §4.1 [A] 死结 → S2-7 | STYLE_GUIDE 全文 180 行逐节 |
| 23 | R2 自始协议输入与规则 | rubric.md 存在 ✓；style sheet L234-249（R2-1…R2-12 齐全）/ L268-273（§4.3 配额：≥1 题 [10]/[20]、≤1 题 [40] + (a)(b)(c)、非递减、≥1 证明题）/ L275-283（§4.4 校验锚三选一）；spec §4.6 L137 逐条比对 | spec 练习规则与 style sheet §4 逐字一致 ✓；R2-1…12 列举与 style sheet 编号对应 ✓；rewrite-gate.py docstring（HEAD 对照）+ L99-101（新文件降级 metadata-only + flag）→「新写作不适用」操作指引正确 ✓；对照/短篇章 proof-exercise 由 builder-check L545（training 6.3 模式）覆盖 ✓ | §4.6 全 6 条 × style sheet §3-4 |
| 24 | 图子循环机制 | fig-check.mjs L2/L14-17（任意模块名 import，FIGURES 数组校验）；figures/STYLE.md / QUALITY.md / CENSUS.md / figure-kit.mjs 存在 ✓；build-content L14-21（readdir 全量加载）/ L26-31（dim 字段缺省 models）；training-part02.mjs dim:'training' + 6 处 replacesBlock/afterHeading ✓；spec §6.6 与 §8 签名场景清单逐字相同（10 个）✓；场景↔波次映射 W1（paged/radix/分层）W2（spec tree）W3（overlap）W4（DSpark/EPLB/Wide-EP）W5（PD）W6（容量）每波有交付 ✓ | 机制全部落地、inference-pNN.mjs 命名被 readdir/fig-check 兼容 ✓；「兜底归零」与 algo/grid 分类的字面冲突 → S3-3（training-figure-census L10 先例）；18 边几何密于 training（11 边），builder 手绘时注意可读性（提示，非 finding） | 图管线 5 件 + 双清单比对 + 6 波映射 |
| 25 | QA 命令与样章 | package.json L12 qa 脚本与 spec §8 L488 命令逐字一致 ✓；qa-shots.mjs L11-25 PAGES 硬编码 models+training 清单（需追加 inference）；check-links.mjs grep 维度词零命中（dist 通用爬取）✓；样章 1.2∈W1 / 4.4∈W2 / 7.4∈W4 与 builder-check L545「分阶段加」一致 ✓ | QA 方案与现状相符 ✓；样章追加为 W1/W2/W4 渐进工作，spec 措辞已由 builder-check 闭合 | QA 全命令 + 3 样章波次归属 |
| 26 | 附录算术 | arXiv：14+15+16+2+1=48 查、14+14+16+2+1=47 中、1 错纠正 ✓（L520-524）；repo 枚举逐个数=24 vs 声称 25（L143/L528） | arXiv 计数自洽 ✓（ID 真伪归 A/B/C）；repo 计数 → S3-6 | 48 查分组 + 24 repo 逐一点数 |
| 27 | 维度内两层走读重叠 | spec 2.1 L200 vs 5.2 L290（block_pool/kv_cache_manager/kv_cache_coordinator/single_type_kv_cache_manager 四文件逐字相同 + 同 hybrid KV 主题）；4.1 L252 / 4.3 L264 vs 5.2 L290 / 6.2 L316；8.4 L374 vs 6.3 L322 / 7.4 L348 | 两层设计成立但深度划分/recap 总则缺失 → S2-8；点对点标记（6.2↔3.4、7.4↔1.2 伏笔兑现、8.1↔7.4 recap）部分存在 ✓ | P2/P4 走读章 × P5/P6 对应章 + 8.4 |
| 28 | 先例与环境 | .gitignore「docs/superpowers/」→ 先例 spec 仅存主 checkout（本工作树无），training spec 以只读绝对路径调取 §4.1/§4.5 ✓；models 51 / training 39 章 find 计数 ✓；inference/README.md 占位存在（W0 重建，§3 L61）✓；training-spec-review-d 格式/Severity 口径沿用 ✓ | 先例对齐核查可行（training §6.3 强制清单宽于重章、inference 取等集，均为合法设计选择）；环境限制已记录，不影响结论 | 先例 D 路报告全文 + 目录计数 |

**无 log 即未审查自证**：本报告每条结论均给出文件与行号/引句证据；arXiv ID↔标题、repo 路径存在性、DeepInfer 生产数字、V4.1 config 字段值等事实性断言一律未做，归 evaluator A/B/C（S3-5 已显式移交 A 路）。

## 放行条件

S2-1 ~ S2-8 全部修复（或逐条附证据驳回）后放行；S3 可并入 W0 一并清零。优先级提示：S2-6 / S2-7 阻塞 W0 站点与 STYLE_GUIDE 改造（照现状文本执行会在 W1 触发构建崩溃与系统性 [A] 约定 findings）；S2-1 / S2-2 / S2-4 / S2-5 阻塞 W1 写作输入（recap 指针、重复立项、符号翻译规则必须先于 writer brief 锁定）；S2-3 / S2-8 可与 W1/W2 修复同步。
