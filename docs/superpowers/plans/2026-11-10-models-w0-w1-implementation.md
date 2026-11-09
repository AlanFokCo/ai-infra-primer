# models 维度 W0+W1 实施计划（脚手架 + Foundations + Transformer Anatomy）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 models 维度的 W0 脚手架（README/STYLE_GUIDE/references schema/进度表/占位维度）与 W1 写作（P1 Foundations 3 章 + P2 Transformer Anatomy 8 章），含 research pass 与独立 evaluator 对抗审查质量门。

**Architecture:** 波次制内容生产流水线：research pass（锁定全部事实源）→ 逐章写作（每章 = 一次可提交交付物 + lint 自检）→ 独立 evaluator 对抗审查（强制重推导清单子集 + verification log）→ 修复复审 → commit gate。所有章节的唯一事实源是 spec §5 对应条目与 references.md。

**Tech Stack:** Markdown + GitHub MathJax 公式、PyTorch 教学片段（片段级）、WebSearch/WebFetch 调研、子 agent 对抗审查、git 分章提交。

**Spec:** `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（已过 4 路对抗审查，commit `4ebc02e`）

## Global Constraints

（从 spec §4 逐条继承，所有任务隐含遵守）

- 中文正文；术语保英文（attention、token、embedding、KV cache、latent 等不翻译）；首次出现概念用「中文（English Term）」
- **禁止历史叙事**：无时间线、无人物故事、无「某年某司发布」背景段；论文只作 citation
- 「为什么」一律用机制/数学/账回答；章末总结只准表格或公式清单
- 公式：行内 `$...$`、独立 `$$...$$`；每公式后跟符号表；推导不跳关键步；张量 shape 用 `[B, T, H, D]` 记法
- 证据分级 [A]（paper+repo+weights 齐全）/ [B]（有 paper 无官方代码）/ [C]（闭源）/ [W]（开放权重无论文）/ 教学参照（不占等级）
- 所有数值事实带出处；arXiv ID 未经 references.md 收录（含核实日期）不得出现在正文
- 章节模板：metadata 块（定位/前置/锚点）→ 正文 → 总结（表格/公式清单）→ 练习（2-4 题）→ 参考
- 篇幅：推导重章 6000-9000 字（本波 2.3/2.4/2.5/2.7/2.8）、标准章 4000-7000 字
- 代码片段标注「教学用，非优化实现」；引用官方实现给 repo+文件路径+函数名
- 图：ASCII 为主；禁外部图片链接；论文图自绘并注明出处
- 去 AI 味禁令：「在当今…」「值得注意的是」「综上所述」「总的来说」「让我们…」、排比堆砌、空洞形容词

## Review Focus

（spec 暗示但任务级 lint 覆盖不到的失败模式，逐条挂到拥有者任务）

1. **公式推导本身错误**（lint 只能查格式查不了数学）→ T19 evaluator 强制重推导清单：1.3 Chinchilla Lagrangian、2.3 √d_k、2.4 KV cache、2.5 RoPE、2.7 Pre/Post-LN 梯度
2. **GitHub MathJax 渲染与本地预览不一致** → T6 渲染探测 + 白名单锁定（无远端时以 GitHub 官方文档为准固化保守子集）
3. **正文引用与 references.md 漂移**（写作中新增/改名引用）→ 每章 lint 步骤含双向一致性命令；T19 复查
4. **数值事实过时或凭记忆** → T7 research pass 全部落实到手来源并记录核实日期；evaluator A 已核实的数值直接采用其结论
5. **evaluator rubber-stamp（无证据放行）** → T19 验收标准要求 verification log（抽查项×来源×结论×覆盖率），无 log 视为未审查

---

## W0 · 脚手架

### Task 1: 根 README 重写

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: spec §3 仓库结构
- Produces: 仓库门面；指向 `models/README.md`、`STYLE_GUIDE.md`

- [ ] **Step 1: 写作**——内容：一句话定位（hands-on primer：from tokens to teraflops）；五维度导航表（models=进行中，training/inference/gpu/cluster=占位，各一句话范围）；models 阅读路径摘要（10 Part 依赖顺序一句话）；规范链接（STYLE_GUIDE.md、models/references.md）；项目状态徽章式文本（当前波次 W1）
- [ ] **Step 2: 自检**——`grep -c "models/" README.md` ≥ 3；无占位符（`grep -nE "TODO|TBD" README.md` 无输出）
- [ ] **Step 3: Commit**——`git add README.md && git commit -m "docs: rewrite root README with five-dimension navigation"`

### Task 2: STYLE_GUIDE.md

**Files:**
- Create: `STYLE_GUIDE.md`

**Interfaces:**
- Consumes: spec §4.1-4.9 全部条款
- Produces: 写作与审查的唯一标准；T6 可能回改其「公式语法白名单」一节

- [ ] **Step 1: 写作**——逐节落地 spec §4：语言 / 纯技术规则（含禁止历史叙事）/ 公式（含白名单一节，初版写入候选集：display math、`aligned`、`cases`、`bmatrix/pmatrix`、上下标、`\mathbb{}`、`\mathrm{}`、`\bar{}`、范数、`\langle\rangle`；标注「T6 探测后锁定」）/ 代码 / 图 / 引用与证据分级（[A]/[B]/[C]/[W]/教学参照完整定义，含 Mistral 类回升 [A] 的判定：有正式 paper 即不受 [W] 限制）/ 锚点规则 / 章节模板（完整模板代码块 + 篇幅约定）/ 去 AI 味 checklist（禁令词表原样列入）/ evaluator 协议摘要（指向 spec §6.3-6.4）
- [ ] **Step 2: 自检**——`grep -cE "^###" STYLE_GUIDE.md` ≥ 8；禁令词表逐项在文中出现（`grep -c "综上所述" STYLE_GUIDE.md` ≥ 1 等）
- [ ] **Step 3: 创建章节 lint 脚本** `scripts/lint-chapter.sh`（强制执行 STYLE_GUIDE 的可机械检查部分，供 T8-T18 每章调用）：

```bash
#!/usr/bin/env bash
# usage: bash scripts/lint-chapter.sh models/02-transformer-anatomy/03-attention-mechanism.md
set -u
f="$1"; fail=0
# 1. AI 味禁令词
if grep -nE "在当今|值得注意的是|综上所述|总的来说|众所周知|让我们一|随着.*快速发展" "$f"; then echo "FAIL: 禁令措辞"; fail=1; fi
# 2. 模板小节齐全
for s in "## 总结" "## 练习" "## 参考"; do grep -qF "$s" "$f" || { echo "FAIL: 缺 $s"; fail=1; }; done
# 3. metadata 块（前 8 行内存在 > 引用行）
head -8 "$f" | grep -q "^> " || { echo "FAIL: 缺 metadata 块"; fail=1; }
# 4. 正文出现的 arXiv ID 必须在 references.md 有条目
for id in $(grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" "$f" | sed 's/arXiv://' | sort -u); do
  grep -q "$id" models/references.md || { echo "FAIL: references.md 缺 $id"; fail=1; }
done
# 5. 证据等级标注存在
grep -qE "\[(A|B|C|W)\]|教学参照" "$f" || { echo "FAIL: 无证据等级标注"; fail=1; }
# 6. \$\$ 定界符配平
n=$(grep -o '\$\$' "$f" | wc -l | tr -d ' '); [ $((n % 2)) -eq 0 ] || { echo "FAIL: \$\$ 不配平"; fail=1; }
[ "$fail" -eq 0 ] && echo "LINT OK: $f"
exit $fail
```

- [ ] **Step 4: 脚本自检**——`bash scripts/lint-chapter.sh README.md` 预期报 FAIL（根 README 非章节，证明脚本会报错而非恒通过）；对一个手工构造的合规样例文件跑一遍预期 LINT OK
- [ ] **Step 5: Commit**——`git add STYLE_GUIDE.md scripts/lint-chapter.sh && git commit -m "docs: add STYLE_GUIDE and chapter lint script"`

### Task 3: references.md 骨架与 schema

**Files:**
- Create: `models/references.md`

**Interfaces:**
- Consumes: spec §4.6、S3-7 修复决议
- Produces: 条目 schema（T7 及后续所有波次按此写入）；术语表挂载点

- [ ] **Step 1: 写作**——文件头：用途与写入规则（先收录后引用）；schema 定义与一条示例：
```
| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
```
示例行：`| transformer | Attention Is All You Need | Vaswani et al. | arXiv:1706.03762 | [A] | 2026-11-10 | 2.1, 2.3, 2.4 |`
附术语表 section（空表头，后续波次累积：英文术语 | 中文处理 | 首次出现章节）
- [ ] **Step 2: 自检**——schema 七字段齐全（表头 grep）
- [ ] **Step 3: Commit**——`git add models/references.md && git commit -m "docs: add references.md schema and glossary mount point"`

### Task 4: models/README.md（目录 + 依赖图 + 进度表）

**Files:**
- Create: `models/README.md`

**Interfaces:**
- Consumes: spec §5 全大纲、§7 波次表
- Produces: 全维度唯一进度事实源；T8-T18 每章完成后更新对应行

- [ ] **Step 1: 写作**——内容：models 维度定位一段话；前置知识（线代/概率/Python）；10 Part × 51 章完整目录（每章一行：编号 + 文件链接 + 一句话）；Mermaid 依赖图（P1→P2→P3→P4→P5 主干，P2→P6→P7→P8、P6→P10(RVQ)、P7→P9 支链）；进度表（51 行 × 状态列：未开始/已写/审查中/已关闭，初始全部「未开始」）；审查记录区（各波 evaluator 结论追加处）
- [ ] **Step 2: 自检**——进度表行数 = 51（`grep -c "| 未开始 |" models/README.md` = 51）；目录链接全部指向将创建的文件路径（路径与 T8-T18 Files 一致）
- [ ] **Step 3: Commit**——`git add models/README.md && git commit -m "docs: add models README (TOC, dependency graph, progress table)"`

### Task 5: 兄弟维度占位 README

**Files:**
- Create: `training/README.md`、`inference/README.md`、`gpu/README.md`、`cluster/README.md`

**Interfaces:**
- Consumes: spec §2 范围边界
- Produces: 维度占位与接口声明（models 章节的交叉链接目标）

- [ ] **Step 1: 写作**——每个文件：维度范围一段（从 spec §2 out-of-scope 条目改写为该维度的 in-scope 声明）+ 与 models 的接口说明（models 哪些章节交叉链接过来：training←P1.1/P5 训练侧交叉链接、inference←2.4 prefill/decode 与 4.5 MTP 系统侧、gpu←4.1 FlashAttention kernel、cluster←4.5 EP/4.6 Ring Attention）+ 「未开工」状态标注
- [ ] **Step 2: 自检**——四文件存在且各 ≥ 5 行
- [ ] **Step 3: Commit**——`git add training inference gpu cluster && git commit -m "docs: add sibling dimension placeholder READMEs"`

### Task 6: GitHub MathJax 渲染探测与白名单锁定

**Files:**
- Create: `docs/math-render-probe.md`
- Modify: `STYLE_GUIDE.md`（「公式语法白名单」一节）

**Interfaces:**
- Consumes: T2 的候选白名单
- Produces: 锁定的公式语法白名单（全书公式只能使用其子集）

- [ ] **Step 1: 取证**——WebFetch GitHub 官方文档（docs.github.com「Writing mathematical expressions」）确认支持/不支持的构造；记录来源与日期
- [ ] **Step 2: 写探测文件**——`docs/math-render-probe.md`：每个候选构造一个最小用例（display `$$...$$`、`aligned`、`cases`、`bmatrix`、`\text{}`、`\mathbb{R}`、`\bar\alpha`、`\langle q,k\rangle`、范数 `\|x\|`、`\sqrt{d_k}`、行内 `$...$` 紧贴中文字符的边界情况），每项注明「预期渲染结果」；文件头注明：推送 GitHub 后人工核对一次，失败项从白名单移除
- [ ] **Step 3: 锁定白名单**——按 Step 1 证据更新 STYLE_GUIDE（无远端时的保守锁定；探测文件留待首次推送后复核）
- [ ] **Step 4: Commit**——`git add docs/math-render-probe.md STYLE_GUIDE.md && git commit -m "docs: lock math syntax whitelist with render probe"`

---

## W1 · P1 Foundations + P2 Transformer Anatomy（11 章）

### Task 7: W1 research pass（事实源锁定）

**Files:**
- Modify: `models/references.md`
- Create: `docs/research/w1-sources.md`

**Interfaces:**
- Consumes: spec §5 P1/P2 条目、spec 附录审查记录（Evaluator A 已核实清单）
- Produces: T8-T18 写作的唯一事实源；references.md 的 P1/P2 条目

- [ ] **Step 1: 直接收录 spec 审查已核实项**（标核实日期 2026-11-10，来源注明「spec 审查 Evaluator A」）：1706.03762 / 2001.08361 / 2203.15556 / 2005.14165 / 2206.07682 / 2304.15004 / 2401.06066 / 2412.19437 / 2407.21783 / 2501.00656 / 1508.07909 / 1804.10959 / 1609.08144 / 1608.05859 / 1911.02150 / 2104.09864 / 2108.12409 / 1910.10683 / 2305.19466 / 2002.05202 / 1606.08415 / 2012.14913 / 1607.06450 / 1910.07467 / 2002.04745 / 2203.00555 / 2408.00118 / 2310.06825 / 2505.09388 / 2412.15115；已核实模型数值直接入库：Llama-3.1-8B（4096/32L/32Q/8KV/14336/θ=500000/vocab 128256/ctx 131072）、Llama-3.1-70B（8192/80/64/8KV）、DeepSeek-V3（61 层/256+1/top-8/first_k_dense_replace=3/kv_lora_rank=512/qk_rope_head_dim=64/qk_nope_head_dim=128/671B/37B/14.8T/2.788M H800-h）、Pythia 154 checkpoints/model、Gemma 2 soft-capping（attn 50.0/final 30.0）、Gemma 3（5:1、QK-norm、无 soft-capping）、Mistral-7B sliding_window=4096、Qwen3 MoE 128 experts top-8、MiniMax-01 7:1
- [ ] **Step 2: 新鲜核实项**（WebFetch arXiv abs 页 + HF config）：Gemma 3 ⟨verify⟩ 2503.19786；QK-norm ⟨verify⟩ 2010.04245；softmax bottleneck ⟨verify⟩ 1711.03953；BLT ⟨verify⟩ 2412.09871；Pythia 论文 ⟨verify⟩ 2304.01373；Qwen3-8B config（vocab_size、head 数、θ）与 Qwen3 tokenizer 事实；OLMo-2-7B config；**Xiong et al. 2002.04745 定理精确形式**（fetch 原文 §3，落实 Pre/Post-LN 梯度结论的准确表述——spec 2.7 的 ⟨verify⟩ 项）；GPT-2 条目用 OpenAI 官方页 + HF `openai-community/gpt2`（无 arXiv，[W]）
- [ ] **Step 3: 写入**——references.md 按 schema 逐条录入（含证据等级、核实日期、被引章节映射）；`docs/research/w1-sources.md` 写调研笔记（每章：可用数值/config 摘录 + 来源 URL + 注意事项，如「Gemma 3 无 soft-capping，勿沿用 Gemma 2 数字」）
- [ ] **Step 4: 自检**——T8-T18 所需 ID 全部在 references.md（对照各章 Sources 清单逐一 grep）；`wc -l docs/research/w1-sources.md` ≥ 60
- [ ] **Step 5: Commit**——`git add models/references.md docs/research/w1-sources.md && git commit -m "research: lock W1 sources (P1+P2)"`

### Task 8: 章节 1.1 `01-formulation.md` — 语言建模的数学形式化

**Files:**
- Create: `models/01-foundations/01-formulation.md`

**Interfaces:**
- Consumes: T7 的 references.md / w1-sources.md；STYLE_GUIDE.md；spec §5-1.1
- Produces: 全书第一章；三种 objective 的 mask 形式化定义被 2.4/3.1 引用

**内容要求（spec §5-1.1 全量）**：模型定义 $f_\theta: X_{<t} \to \Delta^V$；链式法则 → autoregressive factorization 完整推导；NLL = cross-entropy；CE = H(p) + KL(p‖q_θ)、bits/token；perplexity = exp(CE) 及陷阱（vocab 相关、跨 tokenizer 不可比）；teacher forcing；CLM/MLM/prefix-LM 统一为 attention mask 矩阵 + loss mask 形式化（细节留 P3）；token/vocab/context window 形式化；采样生成简介（temperature/top-p 公式，交叉链接 inference）；**softmax bottleneck**（输出头 rank 限制，一小节）。锚点算例：Llama-3.1-8B 与 Qwen3-8B 真实 config 数值。Sources：2001.08361、2005.14165、1711.03953、GPT-2（[W]）

- [ ] **Step 1: 从 w1-sources.md 取本章数值与 ID**（Qwen3-8B config 值等）
- [ ] **Step 2: 按模板写作**（4000-7000 字；每公式带符号表；链式法则推导不跳步）
- [ ] **Step 3: lint**——`bash scripts/lint-chapter.sh models/01-foundations/01-formulation.md` 预期 `LINT OK`
- [ ] **Step 4: 更新进度表**——models/README.md 对应行 未开始→已写
- [ ] **Step 5: Commit**——`git add models/01-foundations/01-formulation.md models/README.md && git commit -m "content(1.1): language modeling formulation"`

### Task 9: 章节 1.2 `02-compute-accounting.md` — 参数、计算、内存的账

**Files:**
- Create: `models/01-foundations/02-compute-accounting.md`

**Interfaces:**
- Consumes: 同上 + 1.1（f_θ 定义）
- Produces: C≈6ND 推导（被 1.3、4.5、5.x 复用）；KV cache 公式指针 → 2.4

**内容要求（spec §5-1.2 全量）**：matmul FLOPs=2mnk；前向 2N/token；反向=2×前向 → **C≈6ND 完整推导**与适用条件（不含 attention 二次项）；attention O(L·T²·d) 单算；每层 FLOPs 分解表（QKV/attn/out proj/FFN）；内存四账（参数 bf16=2N bytes、激活逐层 shape 账、KV cache 完整公式指针 → 2.4、优化器态 6N~12N bytes 交叉链接 training）；实例演算：Llama-3.1-8B 与 DeepSeek-V3（671B/37B）per-token 账（数值取 T7 已核实清单）。Sources：2001.08361 附录、2412.19437

- [ ] **Step 1-5：同 Task 8 结构**（取数 → 按模板写作（标准章 4000-7000 字）→ `bash scripts/lint-chapter.sh models/01-foundations/02-compute-accounting.md` → 进度表 → `git commit -m "content(1.2): compute and memory accounting"`）

### Task 10: 章节 1.3 `03-scaling-laws.md` — Scaling Laws

**Files:**
- Create: `models/01-foundations/03-scaling-laws.md`

**Interfaces:**
- Consumes: 1.2（6ND 是 Chinchilla 推导的约束）
- Produces: compute-optimal 公式（被 4.5 MoE scaling、P5 各章引用）

**内容要求（spec §5-1.3 全量）**：幂律形式与 log-log 线性；Kaplan $L(N)=(N_c/N)^{\alpha_N}$ 及 L(D)/L(C)；Chinchilla $L(N,D)=E+A/N^\alpha+B/D^\beta$；**compute-optimal Lagrangian 推导**（C=6ND 约束下 min L → $N_{opt}\propto C^a, D_{opt}\propto C^b$）；Kaplan vs Chinchilla 差异与拟合方法原因；inference-aware tradeoff（小模型 over-training 的账：Llama-3.1-8B×15T tokens）；MoE fine-grained scaling law（activated params 主导，DeepSeekMoE 曲线）；emergence 定义与 Wei 分类；**Schaeffer 测量学解释**（连续 vs 离散 metric 制造「跳变」的公式级分析）；局限。锚点：Pythia（154 checkpoints）、OLMo 2。Sources：2001.08361、2203.15556、2206.07682、2304.15004、2401.06066、2304.01373、2501.00656

- [ ] **Step 1-5：同 Task 8 结构**（本章含强制重推导项 Chinchilla Lagrangian——写作时完整展开）→ `bash scripts/lint-chapter.sh models/01-foundations/03-scaling-laws.md` → `git commit -m "content(1.3): scaling laws"`

### Task 11: 章节 2.1 `01-overview.md` — 架构总览与设计约束

**Files:**
- Create: `models/02-transformer-anatomy/01-overview.md`

**Interfaces:**
- Consumes: P1 全部
- Produces: residual stream 形式化与 shape 记法（P2 全 Part 复用）；参数量公式（P5 复用）

**内容要求（spec §5-2.1 全量）**：架构 = tensor 变换组合（ASCII 全景图，自绘）；residual stream 形式化 $x \leftarrow x + f(\mathrm{LN}(x))$（Pre-LN）；residual stream 视角（各层读写同一向量总线，为 4.7 埋伏笔）；一次 forward 完整 shape 表（`[B, T, H, D]` 记法）；**参数量 ≈12Ld² 推导**（attention 4d² + FFN 8d²，embedding 单算）；设计约束三指标（Transformer 原文 Table 1 框架：每层计算量 / sequential ops / 最大路径长度 O(1) vs O(n)）；recurrence 的两个数学代价（不可并行：隐状态依赖链；信息瓶颈：固定 d_h 携带 T token 信息量论证）；三种架构形态结构图（细节 → P3）。锚点：Llama-3.1-8B config.json 逐字段 ↔ 结构图映射；OLMo 2。Sources：1706.03762、2407.21783、2501.00656

- [ ] **Step 1-5：同 Task 8 结构** → `bash scripts/lint-chapter.sh models/02-transformer-anatomy/01-overview.md` → `git commit -m "content(2.1): architecture overview and design constraints"`

### Task 12: 章节 2.2 `02-tokenization-and-embeddings.md` — Tokenization 与 Embedding

**Files:**
- Create: `models/02-transformer-anatomy/02-tokenization-and-embeddings.md`

**Interfaces:**
- Consumes: 1.1（token/vocab 定义）
- Produces: vocab 账（P5 各章引用）；tying 分析（2.8 复用）

**内容要求（spec §5-2.2 全量）**：BPE 训练算法 + 手算小例子 + 压缩视角；byte-level BPE 无 OOV 机制；WordPiece 似然驱动；**Unigram LM 目标 $\max_\theta \sum_x \log \sum_{z\in S(x)} p(z)$ 与 EM outline**、Viterbi 切分；fertility / compression ratio 公式；**vocab 三方权衡账**（T×V×d；32K/128K/150K 真实对照：Llama-2/3、Qwen3、Gemma——数值取 T7）；embedding = one-hot × W_E 等价性；初始化 scale（0.02 惯例及依据）；**weight tying 梯度耦合分析**；softmax bottleneck（输出头 rank 限制，承接 1.1 一小节此处给公式）；**tokenizer-free 对照段**：BLT byte-level entropy patching（一小节，ID 以 T7 核实为准）；真实 tokenizer 走读：Llama-3.1 与 Qwen3（HF tokenizers/sentencepiece 代码）。Sources：1508.07909、1804.10959、1609.08144、1608.05859、1711.03953、BLT ⟨T7 锁定⟩

- [ ] **Step 1-5：同 Task 8 结构** → `git commit -m "content(2.2): tokenization and embeddings"`

### Task 13: 章节 2.3 `03-attention-mechanism.md` — Attention 机制解剖（推导重章）

**Files:**
- Create: `models/02-transformer-anatomy/03-attention-mechanism.md`

**Interfaces:**
- Consumes: 2.1（shape 记法）、2.2（embedding）
- Produces: attention 公式与复杂度账（2.4/4.1/4.2/4.3/6.1/7.6 全部依赖）；permutation equivariance（2.5 依赖）

**内容要求（spec §5-2.3 全量）**：检索直觉（hard argmax 不可导 → soft softmax 可导）；QKV 投影 $W^Q,W^K,W^V \in \mathbb{R}^{d\times d_k}$；$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$；**√d_k 完整推导**（q,k 分量 iid 零均值单位方差 → Var(q·k)=d_k → 不缩放 |q·k|~√d_k 进 softmax 饱和区 → softmax Jacobian $\mathrm{diag}(p)-pp^\top$ → 梯度 ∝ p_i(δ−p_j) → 饱和时梯度消失——逐步展开）；multi-head（head_i、concat·W^O、参数账、子空间解释）；**permutation equivariance 证明**（Attention(πX)=πAttention(X)）→ 位置编码必要性（→ 2.5）；复杂度 FLOPs O(T²d+Td²) 两项拆分、内存 O(T²)；数值稳定（softmax max 减法）；手写 PyTorch（~40 行，标注「教学用，非优化实现」）↔ F.scaled_dot_product_attention 对照。锚点：Llama-3.1-8B（32 Q heads MHA 口径）、OLMo 2 modeling 代码。Sources：1706.03762

- [ ] **Step 1-5：同 Task 8 结构**（6000-9000 字；含 ≥2 张 ASCII 图）→ `git commit -m "content(2.3): attention mechanism"`

### Task 14: 章节 2.4 `04-causality-and-kv-cache.md` — Causality 与 KV Cache（推导重章）

**Files:**
- Create: `models/02-transformer-anatomy/04-causality-and-kv-cache.md`

**Interfaces:**
- Consumes: 2.3（attention 公式）、1.1（三 objective）
- Produces: KV cache 公式（1.2 指针落点、4.1/4.6/5.x 复用）；prefill/decode 算术强度（inference/gpu 交叉链接源）

**内容要求（spec §5-2.4 全量）**：mask 数学（$M_{ij}=0\ (j\le i),\ -\infty\ (j>i)$，加性形式与极限语义）；三种 mask ↔ P1 三 objective 对照表；autoregressive 生成算法伪码（prefill+decode 循环）；**KV cache 从零推导**（decode 步展开 → k_i,v_i (i<t) 与 t 无关 → 可缓存；q 不缓存的依赖结构论证）；**cache size 公式 bytes = 2·L·H_kv·d_head·T·B·b 逐项解释（MHA 时 H_kv=H_q；GQA 详 → P4.1）**；实例账：Llama-3.1-8B/70B @128K（MLA/DSA 特例 → P4.1/P4.2，案例 → P5.2）；prefill vs decode 算术强度分析（compute-bound vs memory-bound，roofline 入口，交叉链接 gpu/inference）；teacher forcing 与 causality 一致性（训练无信息泄漏论证）。Sources：1706.03762、1911.02150

- [ ] **Step 1-5：同 Task 8 结构**（6000-9000 字）→ `git commit -m "content(2.4): causality and KV cache"`

### Task 15: 章节 2.5 `05-positional-encoding.md` — 位置编码全谱系（推导重章）

**Files:**
- Create: `models/02-transformer-anatomy/05-positional-encoding.md`

**Interfaces:**
- Consumes: 2.3（permutation equivariance 推论）
- Produces: RoPE 数学（4.1 MLA decoupled RoPE、4.6 缩放族、8.2 M-RoPE、9.2 时空 RoPE 全部依赖）

**内容要求（spec §5-2.5 全量）**：必要性 = 2.3 推论；**sinusoidal**（$PE(pos,2i)=\sin(pos/10000^{2i/d})$、频率几何级数、相对位移性质 PE(pos+k)=R_k·PE(pos) 推导）；learned absolute（外推=未定义）；**相对位置族**：T5 bias（log 分桶）、ALiBi（score − m_h·|i−j|、slope $2^{-8h/H}$、外推机制分析）；**RoPE 完整推导**（2D 旋转 → 块对角 → 复数视角 → **⟨R_m q, R_n k⟩ 只依赖 (m−n) 完整证明** → 频率 schedule θ_j=base^{−2j/d} → 波长 λ_j=2π·base^{2j/d} → 远程衰减性质 → partial RoPE：DeepSeek-V3 64/192、Qwen 用法的账）；两种等价实现代码（half-split/interleave）；**NoPE**（causal mask 隐式位置信号，机制论证+文献）；RoPE 外推失败预告（→ P4.6）。锚点：Llama-3.1（θ=500000）、DeepSeek-V3（qk_rope_head_dim=64/qk_nope=128）、Qwen2.5（YaRN 字段）。Sources：2104.09864、2108.12409、1910.10683、2305.19466

- [ ] **Step 1-5：同 Task 8 结构**（6000-9000 字）→ `git commit -m "content(2.5): positional encoding"`

### Task 16: 章节 2.6 `06-ffn.md` — FFN 与非线性

**Files:**
- Create: `models/02-transformer-anatomy/06-ffn.md`

**Interfaces:**
- Consumes: 2.1（12Ld² 账）
- Produces: SwiGLU 参数等价（4.5 MoE、P5 复用）；FFN 记忆视角（4.7 复用）

**内容要求（spec §5-2.6 全量）**：$\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2$；d→4d→d 账（8d²/层 ≈ 2/3 层参数）；激活演化：ReLU → GELU（x·Φ(x) 高斯 CDF 视角）→ **GLU 族 / SwiGLU**（$(\mathrm{Swish}(xW)\odot xV)W_2$、gating 数学作用、**d_ff 从 4d 调至 (8/3)d 保参数恒定的推导**）；**FFN-as-key-value-memory**（$\mathrm{FFN}(x)=\sum_i v_i\,\sigma(k_i\cdot x)$，Geva 分解、激活稀疏性证据）；MoE 预告（→ P4.5）。锚点：Llama-3.1（intermediate_size=14336 的数字账：为什么不是 4×4096）、OLMo 2。Sources：2002.05202、1606.08415、2012.14913

- [ ] **Step 1-5：同 Task 8 结构** → `git commit -m "content(2.6): ffn and nonlinearities"`

### Task 17: 章节 2.7 `07-norm-and-residual.md` — Normalization 与 Residual（推导重章）

**Files:**
- Create: `models/02-transformer-anatomy/07-norm-and-residual.md`

**Interfaces:**
- Consumes: 2.1（residual stream）、2.3（softmax 饱和）
- Produces: Pre/Post-LN 结论（P5 各章引用）；QK-norm（4.4/8.5 复用）

**内容要求（spec §5-2.7 全量）**：LayerNorm 公式与 scale/shift invariance；为什么 NLP 不用 BatchNorm（三条数学原因）；RMSNorm（去掉 re-centering 丢了什么的论证）；**Pre-LN vs Post-LN 梯度分析**（Xiong et al. 推导骨架 → warmup 的数学必然性——定理精确形式以 T7 核实为准）；QK-Norm（解决 attention logits 爆炸，Gemma-2/ViT-22B）；residual 梯度流 $\partial L/\partial x_l=\partial L/\partial x_L(1+\partial/\partial x_l\sum f_i)$ identity 项；zero-centered norm（Gemma/Qwen3-Next）；DeepNorm 缩放系数；**logit soft-capping（仅 Gemma 2：attn 50.0/final 30.0；Gemma 3 已弃用——勿张冠李戴）**。锚点：Gemma 3（QK-norm、pre+post RMSNorm）、Gemma 2（soft-capping 出处）、Llama-3.1、Qwen3-Next。Sources：1607.06450、1910.07467、2002.04745、2203.00555、2408.00118、2503.19786（T7 核实）、2010.04245（T7 核实）

- [ ] **Step 1-5：同 Task 8 结构**（6000-9000 字）→ `git commit -m "content(2.7): normalization and residual"`

### Task 18: 章节 2.8 `08-assembly.md` — 组装完整 Decoder Block

**Files:**
- Create: `models/02-transformer-anatomy/08-assembly.md`

**Interfaces:**
- Consumes: 2.1-2.7 全部
- Produces: P2 收口；config.json 字段字典表（5.6 方法论复用）；minimal 实现（教学参照代码）

**内容要求（spec §5-2.8 全量）**：现代标准配方完整 block 公式（h ← h + Attn(RMSNorm(h)); h ← h + FFN(RMSNorm(h))，全下标，GQA+RoPE+SwiGLU）；**config.json ↔ 公式映射表**（逐字段）；~150 行 minimal decoder-only PyTorch（整合 P2 全组件，标注「教学用，非优化实现」）；官方实现走读（transformers `modeling_llama.py`：LlamaAttention.forward / LlamaMLP / LlamaRMSNorm 摘录 ↔ 公式对照，给文件路径+函数名）；单层单 token FLOPs 全算（Llama-3.1-8B 数值）；参数量对账练习；**结构变体：parallel attention/FFN block**（PaLM/GPT-J/Falcon：h ← h + Attn(LN(x)) + FFN(LN(x)) 单 LN 并行分支公式与梯度/吞吐含义）；常见误解清单（head_dim ≠ hidden/heads 反例、bias 有无、tied embeddings）。锚点：Llama-3.1-8B（主）、OLMo 2（对照）、nanoGPT（教学参照）。Sources：1706.03762、2407.21783、2501.00656

- [ ] **Step 1-5：同 Task 8 结构**（6000-9000 字）→ `git commit -m "content(2.8): assembling the modern decoder block"`

### Task 19: W1 对抗审查（独立 evaluator）

**Files:**
- Read-only 审查对象：`models/01-foundations/*.md`、`models/02-transformer-anatomy/*.md`
- Create: `docs/reviews/w1-review-report.md`（evaluator 报告落盘）

**Interfaces:**
- Consumes: T8-T18 全部章节成品 + STYLE_GUIDE.md + models/references.md + spec §5 P1/P2 条目
- Produces: findings 表（S1/S2/S3）+ verification log；T20 的修复清单

- [ ] **Step 1: 起独立 fresh-context evaluator agent**（不继承写作上下文；agent 类型 general-purpose）。给它的指令要点：立场=假设有错直到一手来源证明；输入=11 章文件 + STYLE_GUIDE.md + references.md + spec §5 对应条目；三任务=事实核查（自行 WebFetch 对照，抽查所有公式/数值/ID/repo 路径/attribution）+ 教学审查（推导断步、未定义符号、循环解释、前置缺口）+ 规范审查（STYLE_GUIDE 逐条，重点抓历史叙事违规、AI 味、术语翻译、证据等级缺失、练习/总结缺失）
- [ ] **Step 2: 强制重推导子集（W1）**——evaluator 必须独立重推导并在 log 中给出关键中间步：1.3 Chinchilla compute-optimal Lagrangian、2.3 Var(q·k)=d_k 全链、2.4 KV cache 公式与可缓存性论证、2.5 RoPE 内积不变性证明、2.7 Pre/Post-LN 梯度结论
- [ ] **Step 3: 验收 evaluator 报告**——必须含 verification log（抽查项 × 来源 URL/引句 × 结论 × 抽查覆盖率）；无 log 视为未审查，打回重审；报告落盘 `docs/reviews/w1-review-report.md`
- [ ] **Step 4: Commit**——`git add docs/reviews/w1-review-report.md && git commit -m "review: W1 adversarial review report"`

### Task 20: W1 修复、复审与 commit gate

**Files:**
- Modify: 被 findings 命中的章节文件
- Modify: `models/README.md`（进度表 11 行 → 已关闭；审查记录区追加结论）
- Modify: `docs/superpowers/specs/2026-11-09-models-dimension-design.md`（波次完成记录追加一行）

- [ ] **Step 1: 修复**——S1/S2 全部修复（数值/公式/引用类必须改到有来源支撑）；writer 异议走反驳通道（附来源引文，提交 evaluator 复核，复核维持才强制改）；S3 记录留待下一波清零
- [ ] **Step 2: 复审**——含 S1 findings 的章节由原 evaluator 复审该章（SendMessage 唤醒同 agent）；复审通过才标已关闭
- [ ] **Step 3: lint 回归**——对被修改章节重跑 `bash scripts/lint-chapter.sh <file>` 全 OK
- [ ] **Step 4: 进度与记录**——models/README.md 进度表更新 + 审查记录（findings 数/关闭状态）；spec 文末「波次完成记录」追加「2026-11-XX W0+W1 完成：11 章 + 审查 N 轮、S1 x 起、S2 y 起，全部关闭」
- [ ] **Step 5: Commit**——`git add -A && git commit -m "content: W1 fixes and review closure (P1+P2 complete)"`

---

## 完成定义（W0+W1）

- W0 六个脚手架文件就位，lint 脚本可用
- 11 章全部通过：lint OK + evaluator 审查无未关闭 S1/S2 + verification log 落盘
- models/README.md 进度表 11 行已关闭；spec 波次记录已追加
