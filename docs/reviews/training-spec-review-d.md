# Training 维度 Spec 对抗审查报告 · Evaluator D（设计合理性 / 规范一致性 / 边界检查）

- 日期：2026-11-16
- 审查对象：`docs/superpowers/specs/2026-11-16-training-dimension-design.md`（全文）
- 参照：`docs/superpowers/specs/2026-11-09-models-dimension-design.md`、`STYLE_GUIDE.md`、`models/README.md`、`models/` 目录抽查（1.2 / 2.7 / 4.5 / 4.6 / 5.2 / 5.5 / 8.3 / 8.5 / 9.4）、`site/scripts/build-content.mjs`、`site/scripts/lib/`（links / katex-guard / rehype-heading-ids / figures）、`site/scripts/qa-shots.mjs`、`site/package.json`、`scripts/lint-chapter.sh`
- 职责边界：不做事实核查（arXiv ID / repo 路径归 evaluator A/B/C）；只查结构、覆盖、一致性、重复。代码现状以工作树为准（含未提交改动）。
- 结论：**0 S1 / 8 S2 / 6 S3 —— 暂不放行，S2 全部修复后放行**（与 models spec 审查先例同级处置）。

---

## Findings 表

### S2-1 · 缺一张正式的 Part 依赖图

- **位置**：spec §5（全章大纲）/ §7（W0）
- **断言**：spec 内无任何 Part 依赖图；依赖关系只隐含在波次排序与各章大纲的内联互指里。
- **证据**：models 先例的 mermaid 依赖图在 `models/README.md` 98-111 行（W0 建），models spec 正文同样无图——本 spec 与先例持平（§3 已计划 training/README.md 含依赖图）。但 training 的跨 Part 边比 models 密：P1→P2/P4/P5（1.5→4.1 前瞻、1.2→5.1）、P2→P7（2.3/2.5→7.2）、P3→P9（3.2→9.2/9.3）、P4→P5（4.4→5.5）、P5↔P6、P5→P7（5.2→7.5）、P8→P9（8.x→9.2/9.4/9.5），且存在一处章级双向互指（1.5 用 DDP `no_sync` 先于 4.1 讲 DDP，4.1 回链 1.5——大纲两侧均已内联声明，属合法前瞻引用，但无图无法在全貌层核验无环）。
- **建议修正**：spec §5 前补一张 Part 依赖图（mermaid 或 ASCII），显式标注 1.5↔4.1 为「前瞻引用 + 回链」；W0 原样落到 `training/README.md`。依赖方向核验：P1→P2→{P3,P4}→P5→P6→P7→P8→P9 无环（本审查已逐边核对）。

### S2-2 · 4.2 的「$16N$ 分解」与 models 1.2 式 (8) 实质重复

- **位置**：spec 4.2 大纲（内容首项与关键推导「$16N$ 分解」）、§6.3 强制重推导清单「4.2 ZeRO $16N$ 分解与各阶通信」
- **断言**：混合精度 Adam 三态分解（2N + 2N + 4N + 8N = 16N bytes）models 已完整给出，training 4.2 按新推导立项，违反本 spec §4.2 自己的「models 已推导的公式一律 recap 一句 + 纯文字指引」纪律。
- **证据**：`models/01-foundations/02-compute-accounting.md` 式 (8) $\mathrm{Mem}_{\mathrm{opt}}=(4+2c)N$ 及紧随的对账表（工作权重 2N / 梯度 2N / fp32 master 4N / 矩估计 2cN / 合计 16N，Llama-3.1-8B 实例 128 GB），且明文「分片（ZeRO 族）属 training 维度」——即 models 已把 16N 交付为既有事实、只把分片留给 training。§4.2 的 recap 示例清单（online softmax / RoPE / MoE aux loss / MTP / VLM loss masking / LoRA）漏列此项。
- **建议修正**：§4.2 清单补「optimizer 三态分解（models 1.2 式 (8)）」；4.2 大纲改为「三态分解 recap 链 models 1.2 式 (8)」、关键推导只留「各阶通信量」；§6.3 对应改为「4.2 ZeRO 各阶通信量与取回时序」。

### S2-3 · 3.1 z-loss 与 models 4.5 式 (12) / 8.5 式 (7) 双重重复风险

- **位置**：spec 3.1 大纲（「z-loss（log Z 惩罚公式、抑制 logit 爆炸的机制……）」、关键推导「z-loss 梯度」）、2.4（z-loss → 3.1 指针）
- **断言**：z-loss 公式在 models 已出现两次（router 侧与输出侧），training 3.1 计划第三次完整呈现且未列入 §4.2 recap 清单。
- **证据**：`models/04-modern-architectures/05-moe.md` §2.4 式 (12)（ST-MoE router logits 侧，「只压数值、不管均衡」）；`models/08-multimodal-llm/05-unified-understanding-generation.md` §3.4 式 (7)（Chameleon 输出 logit 侧，「锁住 softmax 的自由平移度」）——后者正是 training 3.1 的框定（输出侧、抑制 logit 爆炸），而 3.1 的 Sources 只列 ST-MoE（2202.08906，router 侧出处）。7.3 的「数值触发（……attention logit 爆炸——链 models 2.7 与 3.1 z-loss）」进一步混用 router/输出两个作用点。
- **建议修正**：§4.2 清单补 z-loss；3.1 改为「z-loss 公式 recap（router 侧链 models 4.5 式 (12)、输出侧链 models 8.5 式 (7)）」、保留两处增量——z-loss 梯度推导与 z-loss / QK-norm / soft-capping injection point 对照表；3.1 Sources 按作用点分列 ST-MoE 与 Chameleon。

### S2-4 · 7.5 staged extension 与 models 4.6 §4 实质重复

- **位置**：spec 7.5 大纲
- **断言**：7.5 计划的「staged 扩展逐段配方（Llama-3 / DeepSeek-V3 / Qwen2.5-1M 对照）+ 长文档上采样」与 models 4.6 §4 已交付内容同锚点同题材，重复面约占 7.5 计划内容一半。
- **证据**：`models/04-modern-architectures/06-long-context.md` §4「训练配方：把窗口训出来」——Llama-3.1 六阶段 8K→128K 约 800B token（逐段几何级数成本账已推）、DeepSeek-V3 两阶段 + YaRN 落点 163840、Qwen2.5-1M 完整组合（分阶段续训 + 上采样 + rope_theta 1e7 + dual-chunk）、长文档上采样机制与领域漂移边界注记。7.5 大纲只把「RoPE base 调整的数学」链 4.6，未把配方表本身列为 recap。
- **建议修正**：7.5 大纲改为「staged extension 配方 recap 一句 + 链 models 4.6 §4」，本章增量限定为：mid-training 概念与分界、annealing 配比上调、长序列成本账与 CP 启用时机（回链 5.2）、位置分布。若确需对照表，注明「表链 models 4.6 §4，本章只补 training 侧维度（tokens 预算 / 并行启用点）」。

### S2-5 · EP / DualPipe 归属指针三向冲突，且无 errata 通道

- **位置**：spec 5.3（DualPipe）、5.4（EP 通信账与 EP×TP×DP 组合）、§2（三层划分示例）、§3（「models 只链接不改动」）
- **断言**：models 4.5 §7 把「计算-通信重叠（DualPipe）、EP 与 TP 的切分权衡」指路到 cluster 维度；training spec 把 DualPipe 收进 5.3、EP 组合账收进 5.4——已发布的 models 文本与本 spec 的归属声明冲突，而「training 只链接不改动 models」没有给这类指针错误留修订通道。
- **证据**：`models/04-modern-architectures/05-moe.md` 313 行：「进一步主题——all-to-all 的带宽与拓扑、计算-通信重叠（DualPipe）、expert 并行的负载再均衡（EPLB）、EP 与张量并行的切分权衡——全部归 cluster 维度」（该文写于 training 维度存在之前）；同章 331 行 EP 接口表同样写「归 cluster 维度」。本 spec §2 的三层划分（模型侧 models 4.5 → 训练算法与通信账 P5.4 → 硬件 cluster）是正确划分，但与已发布文本的指针方向不一致。
- **建议修正**：spec §2 或 §4.2 增设「跨维度指针 errata」条款：允许 training 波次对 models 已发布文本做仅限指针句的最小修订（逐条记入 training/README 审查记录区）；W3 前把 models 4.5 §7 的「归 cluster」指针收窄为「训练侧归 training P5、硬件/拓扑归 cluster」。另注意 5.4 的 all-to-all 通信账应以 models 4.5 §7 已交付的「每 token 每层 2kd bytes 上界、≤M 扇出」为 recap 起点（大纲未声明）。

### S2-6 · §4.3 符号表与大纲正文冲突（G vs d），GRPO/DP 混场消歧不足

- **位置**：§4.3 符号表（$G$ = DP 副本数 / world size）vs 5.5 大纲（world = $t \times c \times p \times e \times d$）；4.1 大纲（$G$ 副本）
- **断言**：DP 度数在 spec 内有两个符号（§4.3 与 4.1 用 $G$，5.5 用 $d$）；GRPO 组大小 $G$ 与 DP $G$ 的消歧约定（「同字母时上下文区分，GRPO 章节内优先」）对 8.5 / 9.2 这类系统章（GRPO 与并行度数同章出现）不够用。
- **证据**：§4.3 表行「并行度数 $G$（DP 副本数 / world size）」；5.5 大纲「GPU 数乘积公式：world = $t \times c \times p \times e \times d$」；8.5 同时出现「GRPO 去 critic 的节省」与 world-size 级显存/同步账；9.2 PPO 全链路同涉。
- **建议修正**：全书统一 DP 度数为 $d$（4.1/4.2/§4.3 同步改，与 5.5 一致），$G$ 专归 GRPO 组大小；§4.3 加一条「混场章（5.5 / 8.5 / 9.2）禁止用 $G$ 表 DP」。

### S2-7 · 9.3 TRL 锚点标 [A]，与 §4.1 自身口径冲突

- **位置**：9.3 大纲锚点行「TRL `dpo_trainer`（路径 2026-11-16 核实）[A]」
- **断言**：§4.1 明文把 TRL 列为「无独立论文的官方维护库」= 教学参照；builder-check finding 5 已修 1.2 / 3.1 / 9.1 三处同类问题，9.3 是漏网实例。
- **证据**：§4.1「教学参照 = ……TRL（无独立论文的官方维护库）」；9.1「TRL sft_trainer（官方实现）[教学参照 + 官方实现]」、8.5「huggingface/trl [教学参照 + 官方实现]」均为正确口径；8.3 / 8.4 的 TRL 锚点未标等级（可一并补齐）。
- **建议修正**：9.3 改为「TRL dpo_trainer [教学参照 + 官方实现]」；8.3 / 8.4 的 TRL 锚点补同标注。

### S2-8 · §8 站点方案缺口：manifest 消费方与 DependencyGraph 维度化未列入

- **位置**：§8（站点集成计划）
- **断言**：§8 列了 PARTS 抽象、产物分桶、路由、落地页、搜索、链接、图、QA 八项，但漏了两个必炸点：(a) manifest.json 结构从 `parts/flat` 变为 `dimensions.<key>` 后，7 处消费方全部要改；(b) `DependencyGraph.astro` 是 models 专版手绘 SVG（10 个 Part 硬编码坐标 + 读 `manifest.parts`），training 上线需要维度化或另作。
- **证据**：消费方清单——`site/src/pages/index.astro`（`manifest.parts`、`manifest.stats`）、`pages/models/index.astro`（同）、`pages/models/[...slug].astro`（`manifest.parts`、`manifest.flat`）、`pages/models/references.astro`（`references.json` 顶层结构，现变为按维度分键）、`components/DependencyGraph.astro`（`manifest.parts[n-1]`）、`layouts/Base.astro`、`pages/og-card.astro`。工作树未提交的 build-content.mjs WIP 已把 manifest 改为 `dimensions` 结构、references.json 改为按维度分键——消费方不改则 models 站立即回归失败。
- **建议修正**：§8 增列一项「manifest / references.json 结构变更的下游消费方改造清单（7 处 import）+ DependencyGraph 维度化（training Part 依赖图随 W0 README 同步产出）」。

### S3-1 · 9.1「loss masking 精确语义与实现」措辞强于 recap 纪律

- 位置：9.1 大纲（内容与关键推导首项）。models 8.3 式 (1) 已有 loss masking 公式与 LLaVA 代码走读；§4.2 清单已列「VLM loss masking」。建议 9.1 改为「loss masking 公式 recap 一句 + 链 models 8.3」，本章增量 = 多轮对话 masking 策略（全部 assistant 轮 vs 末轮）、chat template 机制、TRL 实现走读。

### S3-2 · 1.5 的 DDP `no_sync` 走读先于 4.1

- 位置：1.5（实现走读含 DDP `no_sync` 上下文）vs 4.1（DDP 正式走读）。双向互指已在大纲内联声明，属合法前瞻引用；建议在依赖图（S2-1）与 1.5 章首 metadata 的「前置」行显式标注，并确认 1.5 只取 `no_sync` 的 API 语义、不展开 DDP 机制。

### S3-3 · 附录预核实的 focal loss（1708.02002）在大纲无落点

- 位置：附录「数据/稳定性/杂项 16 查」含 focal loss ✓，但 §5 大纲无任何章节引用。建议或删或归入 3.1 一句对照（CE 家族变体）。

### S3-4 · §6「四门」与五个小节计数措辞

- 位置：§6 引言「完全沿用 models spec §6 的四门」——models §6 与本 spec §6 均为 6.1-6.5 五个小节。建议改「四道质量门 + commit gate」或直接「沿用 models spec §6 流程」。

### S3-5 · qa-shots.mjs 截图清单为 models-only

- 位置：§8 QA 项。QA 命令清单与 package.json `qa` 脚本逐字一致（✓），但 `site/scripts/qa-shots.mjs` 13-19 行的截图页面清单硬编码 models 路由；建议 §8 注明 W1 起向清单追加 training 样章（公式密 / 表格密 / 推导重章各一）。

### S3-6 · §4.5 / §8 描述的 W0 工作在工作树已有未提交 WIP，批准时注意状态同步

- 位置：§4.5（lint 参数化、STYLE_GUIDE 行文泛化）、§8（DIMENSIONS 抽象）、§3（training/references.md 骨架）。
- 证据：工作树未提交改动已含——`scripts/lint-chapter.sh` 按路径首段选维度（检查项 4 已参数化）；`STYLE_GUIDE.md` 适用范围行 / §6 链接口径 / §10 指针 / §11 用法已泛化（§6 [A] 定义正文未动，靠序言「training 的维度级细化以 spec §4 为准」兜底——可接受）；`training/references.md` 骨架已建；`build-content.mjs` DIMENSIONS 双维度结构与按维度断言组已成型。
- 另注（非 spec 缺陷，供 W0 实现者）：WIP 代码中 `replaceDiagrams` / `insertSignatureFigures` 的调用点已传 `(body, dim.key, slug)` 三参，但函数签名仍为 `(body, slug)`，且内部 `sceneKey(slug, myBlock)` 未升到三参新签名——场景图注册查找会全部失配，续作时需修。

---

## Verification log

| # | 检查项 | 依据（读过/核对过的材料） | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| 1 | 覆盖完备性（任务书 (a)-(g) + 用户补充 3 项） | spec §5 全文逐章 vs 任务书逐条 | 全部有映射，无缺漏：(a)→1.1/1.2/1.3/1.4/1.5；(b)→2.1-2.5、3.1/3.2；(c)→6.1/6.2/6.3；(d)→4.1-4.4、5.1-5.5（DDP→ZeRO→FSDP、TP、SP/CP、PP、EP、组合账）；(e)→7.1-7.4（+7.5 扩展）；(f)→9.1-9.5、3.2（在线/离线定义）、9.4（判据表）；(g)→8.1-8.5（PG 定理/REINFORCE/AC/GAE/PPO/GRPO 推导、KL k3/advantage/引擎拆分）。用户补充：ZeRO optimizer→4.2+6.2 distributed optimizer；最新进展→2.5 Muon/MuonClip、5.3 DualPipe/ZB、7.2 各家 2025-2026 配方、8.4 DAPO/GSPO/Dr.GRPO、9.5 RLVR/R1/K1.5/Qwen3；论文与解析图→逐章 Sources + STYLE_GUIDE §5 每章 ASCII 图 + §8 figures 管线 | 任务书 7 域逐条核对 = 100%；39/39 章被至少一条任务书项覆盖 |
| 2 | Part 依赖图与循环 | models/README.md 98-111（mermaid 先例）、spec §3/§5/§7、逐条跨章互指核对 | 无正式图 → S2-1；依赖方向逐边核对无环；1.5↔4.1 为合法双向互指（前瞻引用 + 回链均内联声明）；models spec 正文同样无图（先例持平，但见 S2-1 理由） | 9 Part 间全部有向边枚举核对 |
| 3 | 与 models 的边界（6 个指定高危对 + 自找 2 对） | models 2.7（QK-norm/soft-capping 全文相关节）、4.5 式 (12) 与 §7、8.5 §3.4 式 (7)、4.6 §4、8.3 式 (1)、5.5 §2.2、1.2 式 (8)、5.2 §5 | 8.1 RL 动机 vs models 1.1：models 1.1 无任何 RL 内容，干净；2.5 MuonClip vs models 5.5：spec 明示「模型侧公式链 models 5.5、本章只讲训练动力学侧」= 合法 recap；9.1 vs 8.3：纪律已声明、措辞偏强 = S3-1；3.1 z-loss vs 2.7/4.5：2.7 无 z-loss，4.5 与 8.5 各有一处公式 → S2-3；5.4 EP vs 4.5：指针冲突 → S2-5；7.5 vs 4.6：实质重复 → S2-4；自找：4.2 的 16N vs 1.2 式 (8) → S2-2；1.3 FP8 vs 5.2（models 已指 training，干净）；9.5 R1 vs 5.2 §5（models 已指 training，干净） | 指定 6 对全查 + 自找 4 对 |
| 4 | spec 内部一致性 | §4.1 vs STYLE_GUIDE §6；§4.3 vs 4.1/5.5/8.4 大纲；§4.4 vs 大纲【重】逐章 vs §6.3；§7 算术；§3 目录章数 | §4.4 的 12 章（1.2/2.2/2.5/4.2/5.1/5.3/7.2/8.1/8.3/8.4/9.2/9.3）与大纲【重】标注逐一相符；§6.3 强制清单覆盖 18 章 ⊃ 12 重章（多出的 1.3/1.4/2.1/2.3/4.1/8.2 与 models 先例同构——强制清单本就宽于重章，无冲突）；§7 算术 10+6+8+5+5+5=39 ✓，§3 各 Part 章数 5+5+2+4+5+3+5+5+5=39 ✓；§4.1 与 STYLE_GUIDE §6 的张力经 §4 序言「维度级细化」+ 工作树 STYLE_GUIDE 序言兜底闭合（可接受）；符号冲突 → S2-6；TRL 标级 → S2-7；附录 48 查 = 16+16+16 算术 ✓ | §4 全部子节 × §5 39 章 × §6.3 清单 × §7 表，全量比对 |
| 5 | 流程完整性（对 models §6/§7） | models spec §6.1-6.5、§7；本 spec §6/§7；`docs/reviews/2026-11-15-style-audit/rubric.md` 存在性 | research / write / review / fix / commit 五节齐全；references schema 沿用声明（§6.1.4）✓；README 进度表格式继承（§7「唯一进度事实源」、§6.5 随同提交、§6.4 审查记录区）✓；文风 rubric 并入（§6.2/§6.3）为本 spec 新增项，rubric.md 文件存在 ✓；「四门」措辞 → S3-4 | models §6/§7 逐节比对 |
| 6 | 站点方案完整性（§8 vs 代码现状） | 工作树 build-content.mjs（DIMENSIONS WIP 全文）、links.mjs、katex-guard.mjs、rehype-heading-ids.mjs、qa-shots.mjs、check-links.mjs、package.json、index.astro、DependencyGraph.astro 及全部 manifest/references 消费方 | §8 描述与现状相符的点：PARTS/断言硬编码（51/10/≥100）✓、落地页 training 行 live:false ✓、links.mjs 两处扩展点（`../references.md` 硬写 /models/ 前缀；`../../<dim>/README.md` 跨维度规则）与 §8 描述逐字对上 ✓、KaTeX guard（throwOnError:false + 红字签名 + build:done throw）✓、KaTeX MathML 层 data-pagefind-ignore 在 rehype-heading-ids 全局插件中、对新维度自动生效 ✓、figures 目录 readdir 全量加载 + dim 字段缺省 models 的注册机制与「按维度分文件」兼容 ✓、QA 命令与 package.json qa 脚本一致 ✓。缺口 → S2-8（manifest 消费方 7 处 + DependencyGraph）、S3-5（qa-shots 清单）；WIP 签名 bug 注记 → S3-6 | §8 八项逐一 vs 代码核对；lib/ 10 个模块读了 5 个关键件（links/katex-guard/rehype-heading-ids/figures 注册/references 消费） |
| 7 | 篇幅与节奏 | §4.4 + 大纲【重】分布 + models 先例（51 章 / 7 内容波） | 39 章 / 6 内容波（+W0 脚手架 +W7 收尾），models 为 51 章 / 7 内容波；内容波均 6.5 章（models 7.3）；W1=10 章为最大波（models W1=11，同量级）；12 重章分布 W1:3 / W2:1 / W3:2 / W4:1 / W5:3 / W6:2——W5 密度最高（5 章中 3 重章）与 §7 备注「推导密度最高的一波」自洽；分布前重后匀，与「P1-P2 全书基石」定位一致 | 12/12 重章定位核对；7 波章数全量 |

**无 log 即未审查自证**：本报告每条结论均给出文件与行/节号证据；事实性断言（arXiv/repo/论文章节归属）一律未做，归 evaluator A/B/C。

## 放行条件

S2-1 ~ S2-8 全部修复（或逐条附证据驳回）后放行；S3 可并入 W0 一并清零。
