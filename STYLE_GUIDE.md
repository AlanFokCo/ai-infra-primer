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
- GitHub 渲染探测：白名单与配套写作规则已于 2026-11-13 完成 GitHub 实测复核（P1-P21，结果见 `docs/math-render-probe.md` 核对记录）；以下三条为实测确认的硬性规则。
- **开口侧 flanking（实测，P22）**：行内开口 `$` 的前一个字符只允许是空白、行首、`(`、或 Markdown 元素边界（`**`、code span、链接、另一个公式之后）——其余任何字符都会让 GitHub 不解析该公式、渲染为裸源码，包括 CJK 字符、全角标点、ASCII 字母/数字、`-` `/` `=` `×` `÷` `→` `,` `.` `[` `!` `?` `:` `;` 等。处置按场景：算符/标点后补一个空格（`= $x$`、`÷ $c_t$`）；连字复合词用 `` $`...`$ `` 包裹保持紧凑（`` rank-$`d_k`$ ``、`` top-$`k`$ ``、`` score/$`\varepsilon`$ ``）——backtick 形式开口豁免 flanking，但仍不能紧贴 ASCII 字母/数字。
- **闭口侧 flanking（实测，P23）**：闭口 `$` 前不能是空白（`$x = $` 不解析）；闭口 `$` 后不能紧跟 ASCII 字母或数字（`$0.79$M`、`$>$128` 不解析）——单位收进公式（`$0.79\text{M}$`）、比较式写 `$>128$`。闭口后紧跟中文、全角/半角标点不受影响。`` $`...`$ `` 的闭口侧不受字母/数字限制。
- **结构限制（实测，P25）**：行内公式不能含 `\begin{bmatrix}` 等矩阵环境（不解析，矩阵一律用 `$$` display）；GFM 表格单元格内公式含裸 `|` 会被表格切分单元格，写 `\|`（表格转义还原为 `|` 后进 MathJax；范数等双竖线语义在表格内直接写 `\Vert`，不要写 `\|`）；区间 `$A$–$B$` 的第二段开口紧贴 `–` 不解析，合并为单公式 `$A\text{–}B$`；「函数名（行内公式）」且公式内再含圆括号时不解析（如 `Bernoulli($\Phi(x)$)`），函数名收进公式（`$\mathrm{Bernoulli}(\Phi(x))$`）。
- **反斜杠反转义陷阱（实测）**：GitHub 的 Markdown 管线先把数学内容里「`\` + ASCII 标点」反转义、再交给 MathJax——`\{`→`{`（集合花括号消失）、`\|`→`|`（双竖线变单竖线）、`\,`→`,`、`\;`→`;`、`\!`→`!`（间距命令变成可见标点）、`\%`→`%`（注释符吞掉行内后文）、`\#`→`#`（MathJax 直接报错）。源码禁用这些写法，映射：`\{`/`\}`→`\lbrace`/`\rbrace`；`\|`→`\Vert`；`\,`→`\thinspace`；`\;`→`\hspace{5mu}`；`\:`→`\hspace{4mu}`；`\!`→`\negthinspace`；`\ `（反斜杠空格）实测不受影响。例外：GFM 表格单元格内 `\|` 是表格转义（渲染为单竖线 `|`），保持原样。
- **强调符冲突与 `\%`、`\#` 的保护形式（实测）**：行内公式含 `*`，或 `_` 前面是标点（`}`、`'`、`)` 等，如 `\mathrm{Attn}_{\ell}`、`y'_s`）时，强调配对会吞掉两个公式之间的全部内容；含 `\%`/`\#` 同受反转义陷阱。行内一律改用 `` $`...`$ `` 包裹（code span 保护内容，且开口不受 flanking 限制、可紧贴中文）。display 块无 backtick 形式：`*` 改写为 `\ast`（同形），`\%`/`\#` 改写为 `\text{％}`/`\text{＃}`（全角）；`_` 前接字母的常规下标（`x_t`、`q_\theta`）不受影响，display 块内 `_` 实测不受影响。

#### 3.1 公式语法白名单（已锁定 · T6 · 2026-11-10）

**状态：已锁定；2026-11-13 经 GitHub 实测复核（P1-P19 全部核对、新增 P20-P21；同日二轮边界探测新增 P22-P25，记录见 `docs/math-render-probe.md` 核对记录区）。** 依据为 GitHub 官方文档与 MathJax 官方文档（证据见本节末 [G1]-[G3] / [M1]-[M2]）外加 2026-11-13 渲染实测（证据 [R1]，见本节末）。初版锁定时仓库尚无远端、只取文档证据支持的最保守子集；实测复核确认表中构造全部可渲染，并据此确认 §3 三条硬规则（开口侧 flanking、反斜杠反转义、`*` 强调冲突）。注意：白名单约束的是**源码写法**——`\,`、`\;`、`\:`、`\!`、`\{`、`\}`、`\|`、`\%`、`\#`、裸 `*` 虽是合法 MathJax，但会被 GitHub 的 Markdown 预处理破坏，已从源码白名单移除（替代写法见下表与 §3）。

全书公式只允许使用下表语法：

| 类别 | 语法 | 证据 |
| --- | --- | --- |
| 行内公式 | `$...$`；`$` 内侧不留空格（P12e 实测不渲染）；开口 `$` 不得紧贴 CJK 字符/全角标点（P12a/P12d 实测不渲染），与前文留一个空格，闭口侧不受限（实测）；含 `*`、`\%`、`\#` 时改用 `` $`...`$ ``（P12c/P21 实测渲染且内容受保护） | [G1][R1] |
| display 公式 | `$$...$$` 独立成块，前后空行 | [G1][G2] |
| 对齐环境 | `\begin{aligned} ... \end{aligned}`（仅在 `$$` 块内使用） | [M1] |
| 分段环境 | `\begin{cases} ... \end{cases}` | [M1] |
| 矩阵环境 | `\begin{bmatrix} ... \end{bmatrix}`、`\begin{pmatrix} ... \end{pmatrix}` | [M1] |
| 上下标 / 求和连乘 | `x_{i}^{2}`、`\sum_{i=1}^{n}`、`\prod_{s=1}^{t}` | [G1][M2] |
| 分式 / 根式 | `\frac{a}{b}`、`\sqrt{d_k}` | [G1][M2] |
| 基础函数 / 关系与运算 | `\exp`、`\log`、`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\top`、`\in`、`\le`、`\ge`、`\neq`、`\infty`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\quad`、`\qquad` | [M2] |
| 希腊字母 | 小写常用集 `\alpha`、`\beta`、`\gamma`、`\delta`、`\epsilon`、`\varepsilon`、`\zeta`、`\eta`、`\theta`、`\vartheta`、`\iota`、`\kappa`、`\lambda`、`\mu`、`\nu`、`\xi`、`\pi`、`\rho`、`\sigma`、`\tau`、`\upsilon`、`\phi`、`\varphi`、`\chi`、`\psi`、`\omega`（变体 `\varsigma`、`\varpi`、`\varrho`、`\varkappa` 未收）；大写 `\Gamma`、`\Delta`、`\Theta`、`\Lambda`、`\Pi`、`\Sigma`、`\Phi`、`\Psi`、`\Omega` | [M2] |
| 关系 / 箭头 / 近似 | `\to`、`\rightarrow`、`\mapsto`、`\Rightarrow`、`\Longrightarrow`、`\Leftarrow`、`\Leftrightarrow`、`\Longleftrightarrow`、`\approx`、`\sim`、`\simeq`、`\propto`、`\equiv`、`\ll`、`\gg`、`\subset`、`\subseteq`、`\supset`、`\supseteq` | [M2] |
| 分式变体 / 积分 / 大型运算 | `\dfrac`、`\tfrac`、`\int`、`\oint`、`\bigcup`、`\bigcap` | [M2] |
| 微分 / 修饰符号 | `\partial`、`\nabla`、`\hat{}`、`\tilde{}`、`\vec{}`、`\dot{}`、`\ddot{}`、`\overline{}`、`\underline{}`、`\widehat{}`、`\widetilde{}` | [M2] |
| 定界符 | 伸缩配对 `\left( \right)`、`\left[ \right]`、`\left\langle \right\rangle`、`\left. ... \right|`（`.` 为空占位）；花括号与双竖线用 `\lbrace`、`\rbrace`、`\Vert`（含 `\left\lbrace \right\rbrace`、`\left\Vert \right\Vert` 伸缩形式）；裸 `|` 单竖线不受影响 | [G1][M2][R1] |
| 固定尺寸定界符 | `\big(` `\big)`、`\Big(` `\Big)`、`\bigg(` `\bigg)`、`\Bigg(` `\Bigg)` 配对（`\bigl` / `\bigr` 等左右变体同收）——W1 波末收编（2026-11-11），2.2 式 (4)(13) 与 2.6 式 (5)(6) 已有使用先例；probe 文件 P17 待首次推送后复核 | [M2] |
| 补充符号 | `\lceil` `\rceil`、`\ne`（`\neq` 同义）、`\leftarrow`、`\cup`、`\circ`、`\perp`、`\square`——W2 波末收编（2026-11-12），W2 章节已有使用先例（`\lceil`/`\rceil`：4.1 §5、4.5 式 (11)；`\cup`：3.1 式 (9)(10)(16)；`\circ`：4.3 §9；`\perp`/`\square`/`\ne`：4.3 §5/§10；`\leftarrow`：4.7 式 (9)(10)）；probe 文件 P18 待首次推送后复核 | [M2] |
| 补充符号（W4） | `\det`、`\varnothing`——W4 波末收编（2026-11-12），章节已有使用先例（`\det`：7.1 式 (4) 及同节正文、练习 2；`\varnothing`：4.2 式 (9)、7.4 §4 CFG 段式 (17)(18)(20)(23) 及正文）；probe 文件 P19 待首次推送后复核 | [M2] |
| 间距 | `\ `（控制空格）、`\quad`、`\qquad`；细间距/负间距用 `\thinspace`、`\negthinspace`，其余用 `\hspace{<N>mu}`（如 5mu 代 `\;`）；`\,`、`\:`、`\;`、`\!` 因反转义陷阱禁用（§3） | [M2][R1] |
| 补充命令（2026-11-13 收编，W1-W7 章节在用、渲染实测正常，本条补账） | `\ell`、`\notin`、`\cap`、`\mathbf`、`\leftrightarrow`、`\mp`、`\angle`、`\lfloor`、`\rfloor`、`\longrightarrow` | [M2][R1] |
| 反转义替代命令（2026-11-13 收编） | `\lbrace`、`\rbrace`、`\Vert`、`\thinspace`、`\negthinspace`、`\hspace`、`\ast`（display 内代 `*`）；display 内 `\%`/`\#` 用 `\text{％}`/`\text{＃}` | [M2][R1] |
| 其他 | `\mathcal{}`、`\operatorname{}`、`\operatorname*{}`（星号变体 [M1]/[M2] 未单列，P15 重点复核，复核前优先用不带星形式）、`\binom{n}{k}`、`\dbinom{n}{k}`、`\tbinom{n}{k}`、`\Pr`、`\stackrel{}{}`、`\underbrace{}_{}`、`\overbrace{}^{}`、`\times`、`\setminus`、`\forall`、`\exists`、`\oplus`、`\oslash`、`\ldots`、`\cdots`、`\vdots`、`\ddots` | [M2]（`\operatorname`、`\binom` 族亦见 [M1]） |
| 数集 / 算子字体 | `\mathbb{R}`、`\mathrm{softmax}` | [M2] |
| 均值记号 | `\bar\alpha` | [M2] |
| 范数 / 竖线 | `\lVert x \rVert`、`\|x\|`、`\vert`、`\Vert`、`\mid`（不走 `\left\right` 的单独竖线：条件概率 `p(y \mid x)`、集合构造、表格单元格场景） | [M1][M2] |
| 内积 | `\langle q, k \rangle` | [M2] |
| 公式内文字 | `\text{...}`（内放中文为探测项 P7b，未复核前只放 ASCII） | [M2] |

**表格渲染注**：GFM 表格按词法切列、不识别 code span；上表 code span 内的 `\|` 是 GFM 转义写法，渲染后显示为单个竖线字符（反斜杠不显示）。同样的原因，表格单元格内的行内公式里 `\|` 也按表格转义处理、渲染为单竖线——这是表格内唯一允许保留 `\|` 的位置（2026-11-13 实测确认）。

**禁止**（不在白名单，一律不用）：`align` / `align*` / `equation` / `gather` / `split` / `multline` 等带编号或顶层多行环境（对齐需求由 `aligned` 覆盖）；`\newcommand` / `\DeclareMathOperator` 等宏定义；`\boxed`、`\color` 等装饰命令；`math` 围栏代码块（全书统一用 `$$`，避免两套写法并存）。

**证据**（访问日期均为 2026-11-10）：

- [G1] GitHub Docs, "Writing mathematical expressions"：<https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions>。要点：渲染引擎为 MathJax；行内 `$...$` 与 `` $`...`$ `` 两种定界；块级 `$$...$$`；数学内 `\$`、数学外 `<span>$</span>` 转义美元符号；Issues / Discussions / PR / wikis / Markdown 文件均渲染；官方示例含 `\sqrt{}`、`\sum_{k=1}^n`、上下标、`\leq`。
- [G2] GitHub Blog, "Math support in Markdown"（2022-05-19 发布，2024-05-21 更新）：<https://github.blog/news-insights/product-news/math-support-in-markdown/>。`$` / `$$` 定界符原生支持，MathJax 渲染。
- [G3] GitHub Changelog（2022-06-28）`math` 围栏代码块：<https://github.blog/changelog/2022-06-28-fenced-block-syntax-for-mathematical-expressions/>。
- [M1] MathJax Docs, "ams" 扩展：<https://docs.mathjax.org/en/latest/input/tex/extensions/ams.html>。原文："This extension is already loaded in all the components that include the TeX input jax, other than `input/tex-base`"；环境清单明确含 `aligned`、`cases`、`bmatrix`、`pmatrix`；宏清单含 `\lVert`、`\rVert`。
- [M2] MathJax Docs, "Supported TeX/LaTeX commands"：<https://docs.mathjax.org/en/latest/input/tex/macros/index.html>。base 宏含 `\text`、`\mathbb`、`\mathrm`、`\sqrt`、`\langle`、`\rangle`、`\bar`、`\vert`、`\Vert`、`\mid`、`\frac`、`\sum`、`\prod`，及基础函数 / 关系与运算符 `\exp`、`\log`、`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\top`、`\in`、`\le`、`\ge`、`\neq`、`\infty`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\quad`、`\qquad`（2026-11-10 逐项核对存在）；希腊字母小写常用集（`\varsigma`/`\varpi`/`\varrho`/`\varkappa` 变体亦存在但未收）与常用大写、关系/箭头/近似符、`\dfrac`/`\tfrac`/`\int`/`\oint`/`\bigcup`/`\bigcap`、`\partial`/`\nabla`/修饰符号（`\hat`/`\tilde`/`\vec`/`\dot`/`\ddot`/`\overline`/`\underline`/`\widehat`/`\widetilde`）、`\left`/`\right` 伸缩定界符（[G1] 官方示例亦用 `\left( \right)`）与同页 base 定界符族 `\big` / `\Big` / `\bigg` / `\Bigg`（2026-11-11 收编时核对）、`\mathcal`/`\operatorname`/`\stackrel`/`\underbrace`/`\overbrace`、`\times`/`\setminus`/`\forall`/`\exists`/`\oplus`/`\oslash`、`\ldots`/`\cdots`/`\vdots`/`\ddots`、薄空格 `\,`、`\{`、`\}` 与裸竖线符号同页逐项核对存在（2026-11-10）；大写箭头 `\Rightarrow`、`\Longrightarrow`、`\Leftarrow`、`\Leftrightarrow`、`\Longleftrightarrow`，间距 `\;`、`\:`、`\!`，`\Pr`，及 `\binom`、`\dbinom`、`\tbinom`（亦见 [M1]）同页逐项核对存在（2026-11-10）；补充符号 `\lceil`、`\rceil`、`\ne`、`\leftarrow`、`\cup`、`\circ`、`\perp`、`\square` 同页逐项核对存在（2026-11-12，W2 波末收编）；函数 `\det` 与空集符 `\varnothing` 同页逐项核对存在（2026-11-12，W4 波末收编）；`\ell`、`\notin`、`\cap`、`\mathbf`、`\leftrightarrow`、`\mp`、`\angle`、`\lfloor`、`\rfloor`、`\longrightarrow`、`\lbrace`、`\rbrace`、`\thinspace`、`\negthinspace`、`\hspace`、`\ast` 同页逐项核对存在（2026-11-13，渲染复核时收编）。
- [R1] GitHub 渲染实测（2026-11-13）：经 `gh api repos/AlanFokCo/ai-infra-primer/contents/<path> -H "Accept: application/vnd.github.html+json"` 拉取 51 章渲染后 HTML 逐式核对，并用 agent-browser 加载 github.com 页面检查 MathJax 实际输出（MathML `data-latex` 与 `merror` 报错）；临时 gist 探测确认开口侧 flanking 边界、`` $`...`$ `` 保护形式与替代命令可用性。探测用例与结果记录见 `docs/math-render-probe.md`。

**已知边界复核结论（2026-11-13 两轮实测，[R1]）**：行内 `$...$` 开口 `$` 的前字符白名单为「空白 / 行首 / `(` / Markdown 元素边界」，紧贴其余任何字符（CJK、全角标点、ASCII 字母数字、`-` `/` `=` `×` 等）均不渲染（P12a/P12d + P22 确认，规则升级为 §3 硬规则）；闭口 `$` 前不能是空白、后不能紧跟 ASCII 字母或数字（P23 确认），紧贴中文与全角/半角标点不受影响；`` $`...`$ `` 形式开口豁免 flanking（但仍不可紧贴 ASCII 字母）、闭口豁免字母/数字限制（P24）；行内公式不能含矩阵环境、表格内裸 `|` 会切分单元格、区间写法应合并为 `$A\text{–}B$`（P25）；`$` 内侧留空格不渲染（P12e 确认）；`\text{}` 内中文渲染正常（P7b）；`\operatorname*` 星号变体在 GitHub 的 MathJax 配置下报「macros are not allowed」错误（P15 实测）——禁用星号变体，一律用 `\operatorname`；`\medspace`、`\thickspace` MathJax 不支持（按字面文本显示），禁用。

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

篇幅约定（字数口径：正文 CJK 汉字数，不含标点 / 公式 / 代码——2026-11-12 controller 裁定，取代旧「总字符数」口径）：

- 标准章 ≥ 3500 字；
- 推导重章 ≥ 5500 字（软目标；如 7.2 / 4.3 / 7.5）；
- 短篇/展望章 ≥ 2500 字（如 4.8）；
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
