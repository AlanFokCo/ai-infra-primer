# STYLE_GUIDE — models 维度写作与审查唯一标准

本文件是写作 agent 与 evaluator agent 的共同唯一标准。条款定义源自 `docs/superpowers/specs/2026-11-09-models-dimension-design.md` §4（写作规范，4.1-4.9）；evaluator 协议摘要对应 spec §6.3-6.4。写作与审查出现分歧时以本文件为准；本文件未覆盖的情形回查 spec。

适用范围：models 维度全部章节（10 Part / 51 章，W1-W7 各波）。每章提交前必须在仓库根目录通过 `bash scripts/lint-chapter.sh <chapter-file>`（检查项见本文 §11）；lint 通过只是下限，不替代 §9 自查与 evaluator 审查。

## 条款

### 1. 语言

- 中文正文；专业术语保持英文：attention、token、embedding、MoE、router、KV cache、score、latent、codec 等不翻译。
- 首次出现的概念用「中文（English Term）」格式并给出定义；全书术语表由 `models/references.md` 附带。
- 语气：资深工程师对同事讲话，直接、具体。

### 2. 纯技术规则（用户硬要求）

- **禁止历史叙事**：不写时间线、不写人物故事、不写「某年某司发布某模型」式背景段。
- 论文只以 citation 形式出现，作为公式与论断的证据来源。
- 「为什么」一律用机制、数学、账（FLOPs / 内存 / cache / 数据效率）回答，不用年代顺序回答。
- 章末禁止空洞评论段；总结只允许用表格或公式清单。

### 3. 公式

- GitHub 原生 LaTeX：行内 `$...$`，独立 `$$...$$`。
- 每个公式后跟符号表；推导不跳关键步；重要公式编号（章内 `(1)(2)...`）便于交叉引用。
- 张量 shape 统一 `[B, T, H, D]` 记法，关键运算处标注 shape。
- GitHub 渲染探测：§3.1 白名单已按官方文档保守锁定（T6，2026-11-10）；因锁定时仓库无远端，首次推送 GitHub 后用 `docs/math-render-probe.md` 逐项人工复核，失败项从白名单移除并回改受影响章节。

#### 3.1 公式语法白名单（已锁定 · T6 · 2026-11-10）

**状态：已锁定。** 依据为 GitHub 官方文档与 MathJax 官方文档（证据见本节末 [G1]-[G3] / [M1]-[M2]）。锁定时仓库尚无远端、无法直接观测渲染，故只取文档证据支持的最保守子集；`docs/math-render-probe.md` 为下表每项构造备有最小用例（P1-P15），首次推送 GitHub 后人工核对，失败项从本表删除并回改受影响章节。

全书公式只允许使用下表语法：

| 类别 | 语法 | 证据 |
| --- | --- | --- |
| 行内公式 | `$...$`；`$` 内侧不留空格（社区惯例，[G1] 未明文，待 P12 探测复核）；与中文相邻时两侧加空格，仍不渲染改用 `` $`...`$ `` | [G1] |
| display 公式 | `$$...$$` 独立成块，前后空行 | [G1][G2] |
| 对齐环境 | `\begin{aligned} ... \end{aligned}`（仅在 `$$` 块内使用） | [M1] |
| 分段环境 | `\begin{cases} ... \end{cases}` | [M1] |
| 矩阵环境 | `\begin{bmatrix} ... \end{bmatrix}`、`\begin{pmatrix} ... \end{pmatrix}` | [M1] |
| 上下标 / 求和连乘 | `x_{i}^{2}`、`\sum_{i=1}^{n}`、`\prod_{s=1}^{t}` | [G1][M2] |
| 分式 / 根式 | `\frac{a}{b}`、`\sqrt{d_k}` | [G1][M2] |
| 基础函数 / 关系与运算 | `\exp`、`\log`、`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\top`、`\in`、`\le`、`\ge`、`\neq`、`\infty`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\quad`、`\qquad` | [M2] |
| 希腊字母 | 小写常用集 `\alpha`、`\beta`、`\gamma`、`\delta`、`\epsilon`、`\varepsilon`、`\zeta`、`\eta`、`\theta`、`\vartheta`、`\iota`、`\kappa`、`\lambda`、`\mu`、`\nu`、`\xi`、`\pi`、`\rho`、`\sigma`、`\tau`、`\upsilon`、`\phi`、`\varphi`、`\chi`、`\psi`、`\omega`（变体 `\varsigma`、`\varpi`、`\varrho`、`\varkappa` 未收）；大写 `\Gamma`、`\Delta`、`\Theta`、`\Lambda`、`\Pi`、`\Sigma`、`\Phi`、`\Psi`、`\Omega` | [M2] |
| 关系 / 箭头 / 近似 | `\to`、`\rightarrow`、`\mapsto`、`\approx`、`\sim`、`\simeq`、`\propto`、`\equiv`、`\ll`、`\gg`、`\subset`、`\subseteq`、`\supset`、`\supseteq` | [M2] |
| 分式变体 / 积分 / 大型运算 | `\dfrac`、`\tfrac`、`\int`、`\oint`、`\bigcup`、`\bigcap` | [M2] |
| 微分 / 修饰符号 | `\partial`、`\nabla`、`\hat{}`、`\tilde{}`、`\vec{}`、`\dot{}`、`\ddot{}`、`\overline{}`、`\underline{}`、`\widehat{}`、`\widetilde{}` | [M2] |
| 定界符 | 伸缩配对 `\left( \right)`、`\left[ \right]`、`\left\{ \right\}`、`\left\| \right\|`、`\left\langle \right\rangle`、`\left. ... \right\|`（`.` 为空占位）；裸 `\{`、`\}`、`\|` 为 base symbol，允许直接使用（无需 `\left\right`） | [G1][M2] |
| 其他 | `\mathcal{}`、`\operatorname{}`、`\operatorname*{}`（星号变体 [M1]/[M2] 未单列，P15 重点复核，复核前优先用不带星形式）、`\,`、`\stackrel{}{}`、`\underbrace{}_{}`、`\overbrace{}^{}`、`\times`、`\setminus`、`\forall`、`\exists`、`\oplus`、`\oslash`、`\ldots`、`\cdots`、`\vdots`、`\ddots` | [M2]（`\operatorname` 亦见 [M1]） |
| 数集 / 算子字体 | `\mathbb{R}`、`\mathrm{softmax}` | [M2] |
| 均值记号 | `\bar\alpha` | [M2] |
| 范数 / 竖线 | `\lVert x \rVert`、`\|x\|`、`\vert`、`\Vert`、`\mid`（不走 `\left\right` 的单独竖线：条件概率 `p(y \mid x)`、集合构造、表格单元格场景） | [M1][M2] |
| 内积 | `\langle q, k \rangle` | [M2] |
| 公式内文字 | `\text{...}`（内放中文为探测项 P7b，未复核前只放 ASCII） | [M2] |

**表格渲染注**：GFM 表格按词法切列、不识别 code span；上表 code span 内的 `\|` 是 GFM 转义写法，渲染后显示为单个竖线字符（反斜杠不显示）。这些单元格所述语法的真实 LaTeX 源码为反斜线紧接竖线，与 `docs/math-render-probe.md` P11/P15 源码区一致。

**禁止**（不在白名单，一律不用）：`align` / `align*` / `equation` / `gather` / `split` / `multline` 等带编号或顶层多行环境（对齐需求由 `aligned` 覆盖）；`\newcommand` / `\DeclareMathOperator` 等宏定义；`\boxed`、`\color` 等装饰命令；`math` 围栏代码块（全书统一用 `$$`，避免两套写法并存）。

**证据**（访问日期均为 2026-11-10）：

- [G1] GitHub Docs, "Writing mathematical expressions"：<https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions>。要点：渲染引擎为 MathJax；行内 `$...$` 与 `` $`...`$ `` 两种定界；块级 `$$...$$`；数学内 `\$`、数学外 `<span>$</span>` 转义美元符号；Issues / Discussions / PR / wikis / Markdown 文件均渲染；官方示例含 `\sqrt{}`、`\sum_{k=1}^n`、上下标、`\leq`。
- [G2] GitHub Blog, "Math support in Markdown"（2022-05-19 发布，2024-05-21 更新）：<https://github.blog/news-insights/product-news/math-support-in-markdown/>。`$` / `$$` 定界符原生支持，MathJax 渲染。
- [G3] GitHub Changelog（2022-06-28）`math` 围栏代码块：<https://github.blog/changelog/2022-06-28-fenced-block-syntax-for-mathematical-expressions/>。
- [M1] MathJax Docs, "ams" 扩展：<https://docs.mathjax.org/en/latest/input/tex/extensions/ams.html>。原文："This extension is already loaded in all the components that include the TeX input jax, other than `input/tex-base`"；环境清单明确含 `aligned`、`cases`、`bmatrix`、`pmatrix`；宏清单含 `\lVert`、`\rVert`。
- [M2] MathJax Docs, "Supported TeX/LaTeX commands"：<https://docs.mathjax.org/en/latest/input/tex/macros/index.html>。base 宏含 `\text`、`\mathbb`、`\mathrm`、`\sqrt`、`\langle`、`\rangle`、`\bar`、`\vert`、`\Vert`、`\mid`、`\frac`、`\sum`、`\prod`，及基础函数 / 关系与运算符 `\exp`、`\log`、`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\top`、`\in`、`\le`、`\ge`、`\neq`、`\infty`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\quad`、`\qquad`（2026-11-10 逐项核对存在）；希腊字母小写常用集（`\varsigma`/`\varpi`/`\varrho`/`\varkappa` 变体亦存在但未收）与常用大写、关系/箭头/近似符、`\dfrac`/`\tfrac`/`\int`/`\oint`/`\bigcup`/`\bigcap`、`\partial`/`\nabla`/修饰符号（`\hat`/`\tilde`/`\vec`/`\dot`/`\ddot`/`\overline`/`\underline`/`\widehat`/`\widetilde`）、`\left`/`\right` 伸缩定界符（[G1] 官方示例亦用 `\left( \right)`）、`\mathcal`/`\operatorname`/`\stackrel`/`\underbrace`/`\overbrace`、`\times`/`\setminus`/`\forall`/`\exists`/`\oplus`/`\oslash`、`\ldots`/`\cdots`/`\vdots`/`\ddots`、薄空格 `\,`、`\{`、`\}` 与裸竖线符号同页逐项核对存在（2026-11-10）。

**已知边界**（官方文档未覆盖，列为探测项，首次推送后复核）：行内 `$...$` 紧贴中文字符或中文标点的解析行为（探测项 P12）——写作规则定为「行内公式与中文之间一律加空格」；`\text{}` 内放中文（探测项 P7b）；`\operatorname*` 星号变体未在 [M1]/[M2] 单列（探测项 P15，复核前优先用 `\operatorname`）。

### 4. 代码

- 讲解性 PyTorch 片段，清晰优先于效率，标注「教学用，非优化实现」。
- 引用官方实现时链接到 repo + 文件路径（+ 关键函数名），写作时核实路径存在。
- 教学重写版与官方实现并列展示时，明确差异点。

### 5. 图

- ASCII 文本图为主（架构图、数据流、shape 走读）；流程图/依赖图可用 Mermaid。
- 禁止引用外部图片链接（防失效）；论文原图不复制，用 ASCII 重绘并注明出处。

### 6. 引用与证据分级

- 正文内 `[Vaswani et al., 2017]` 格式，链接到 `models/references.md` 对应条目。
- 证据等级显式标注（五个标签）：
  - **[A]** = 正式 paper（arXiv）+ official repo + open weights，三者齐——公式拆解可对照真实代码。
  - **[B]** = 有 paper、无官方代码（或代码不全）——按论文拆公式，社区实现辅助并注明。
  - **[C]** = 闭源（仅 blog / model card / 媒体披露）——只讲已披露信息，禁止推测当事实。
  - **[W]** = 开放权重 + config 但无正式 paper（如 Mistral-7B、GLM-4.6、Llama-4）——架构事实以 config / modeling 代码为准，训练配方类断言降级谨慎处理。
  - **教学参照** = 无 paper 的教学 repo（nanoGPT、lucidrains 复现系列）——不占证据等级，仅作代码对照。
- **回升判定（Mistral 类情形）**：[W] 级模型后续找到正式 paper 的，有正式 paper 即不受 [W] 限制——按 [A]/[B] 条件重新定级：正式 paper + official repo + open weights 三者齐 → [A]；有 paper 但官方代码缺失或不全 → [B]。
- **所有数值事实必须有出处**；arXiv ID 一律经该波 research pass 复核后才写入正文。

### 7. 锚点规则

- 每章开头 metadata 块列 anchors：锚点模型（repo 链接到具体文件）+ 关键论文（arXiv）+ 证据等级。
- **代码/config 走读锚点优先 2024-2026 模型**且满足 [A] 级；老经典（Transformer / BERT / DDPM / CLIP / VQ-VAE / Whisper 原文）只作公式源头，不作代码走读对象。
- 无论文的流行模型（FLUX、Mochi）只作 [B]/[C] 级补充对照。

### 8. 章节模板

每章严格使用以下模板：

```
# <Part编号>.<章编号> <中文标题>（English Title）

> 一句话定位 | 前置章节 | 锚点模型与论文（含证据等级）

## 正文各节（以事实或问题开场）

## 总结（表格或公式清单）

## 练习（2-4 题：推导题 / 计算题 / 代码题）

## 参考（本章引用列表，链接 references.md）
```

篇幅约定：

- 推导重章 6000-9000 字（如 7.2 / 4.3 / 7.5）；
- 标准章 4000-7000 字；
- 短篇/展望章 2500-4000 字（如 4.8）；
- 每章至少 1 张 ASCII 结构图或 shape 走读图。

`models/references.md` 条目 schema 在 W0 定义，字段：key / 标题 / 作者 / arXiv ID 或 URL / 证据等级 / 核实日期 / 被引章节。

### 9. 去 AI 味 checklist（evaluator 审查项）

写作完成即逐条自查；evaluator 逐条对照审查。

- 禁用模板化开头与过渡：「在当今…」「值得注意的是」「综上所述」「总的来说」「让我们…」。
- lint 脚本机械执行的禁令词表（§11 检查项①，逐字匹配）：「在当今」「值得注意的是」「综上所述」「总的来说」「众所周知」「让我们一…」类句式（如「让我们一起」）「随着…快速发展」。
- 禁用排比堆砌、空洞形容词（「强大的」「革命性的」——除非引用原文并标注）。
- 每节以具体事实、公式或问题开场。
- 数字优先于形容词；机制优先于评价。

### 10. Evaluator 协议摘要（spec §6.3-6.4）

完整条款见 spec §6.3-6.4，此处为执行摘要：

- 每波写作完成后，起**独立 fresh-context evaluator agent**（不继承写作上下文）。输入：该波章节文件 + 本文件 + `models/references.md` 对应 source list + spec 对应章节要求。
- 三项任务：
  1. **事实核查**：抽查所有公式、数值、日期、arXiv ID、repo 路径、attribution，evaluator 自己 WebFetch sources 对照；**独立重新推导数学重章的关键公式**（强制清单见 spec §6.3：1.3 Chinchilla Lagrangian / 2.3 attention √d_k / 2.4 KV cache / 2.5 RoPE / 2.7 Pre-Post-LN 梯度 / 4.1 MLA / 4.3 DeltaNet-GDN-KDA / 4.5 MoE routing / 6.2 InfoNCE / 6.3 ELBO / 7.2 DDPM / 7.4 CFG / 7.5 FM / 10.2 CTC——每波按覆盖章节指定子集）。
  2. **教学审查**：推导断步、未定义符号、循环解释、前置知识缺口（对照 Part 依赖图）。
  3. **规范审查**：本文件逐条对照——重点抓历史叙事违规、AI 味、术语翻译、证据等级缺失、练习/总结缺失。
- severity 判定：S1 = 事实性致命（张冠李戴的引用、错误公式、不存在的 repo/模型、虚构数值）；S2 = 实质问题（约定冲突、推导断裂、协议缺口、局部事实错误）；S3 = minor（措辞、格式、可选增补）。
- 输出：findings 表（severity + 位置 + 断言 + 证据 + 建议修正）**+ verification log（抽查项 × 来源 URL 或原文引句 × 结论 × 抽查覆盖率）——无 log 的「确认无误」一律视为未审查（防 rubber-stamp）**。
- 修复闭环：S1/S2 必须修复；S3 累计，下一波复审时统一清零（逐条判定：修复 / 接受 / 驳回）。writer 对某条 finding 持异议时可附证据（来源引文）提交 evaluator 复核，复核仍维持才强制修复。含 S1 findings 的章节修复后由 evaluator 复审该章。审查结论（轮次、findings 数、关闭状态）记录到 `models/README.md` 进度表。

### 11. 机械检查：lint 脚本

`scripts/lint-chapter.sh` 强制执行本规范的可机械检查部分，T8-T18 每章提交前在仓库根目录运行：

```bash
bash scripts/lint-chapter.sh models/<part-dir>/<chapter-file>.md
```

检查项：

1. AI 味禁令词（§9 词表）；
2. 模板小节齐全（`## 总结` / `## 练习` / `## 参考`）；
3. metadata 块（前 8 行内存在 `> ` 引用行）；
4. 正文出现的 arXiv ID 在 `models/references.md` 有条目；
5. 证据等级标注存在（`[A]` / `[B]` / `[C]` / `[W]` / 教学参照）；
6. `$$` 定界符配平。

全部通过输出 `LINT OK`、退出码 0；任一失败输出对应 `FAIL: ...`、退出码 1。
