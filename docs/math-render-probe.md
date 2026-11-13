# GitHub MathJax 渲染探测（math-render-probe）

> **用途**：本文件为 `STYLE_GUIDE.md` §3.1 公式语法白名单中的每一项构造提供一个最小用例（P1-P19，2026-11-13 复核后新增 P20-P21）。
> ~~首次推送 GitHub 后逐项人工核对~~ ——**已于 2026-11-13 完成复核**，结论与三处新发现的处置见文末「核对记录」；P11/P15 两个用例已按复核结论改写为现行白名单语法。

## 证据来源（访问日期均为 2026-11-10）

1. **[G1] GitHub Docs《Writing mathematical expressions》**
   <https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions>
   - 渲染引擎为 MathJax（"GitHub's math rendering capability uses MathJax"）；渲染范围覆盖 Issues、Discussions、pull requests、wikis 与 Markdown 文件。
   - 行内：`$...$`；或 `` $`...`$ ``（官方说明：当表达式含有与 Markdown 语法冲突的字符时用后者）。
   - 块级：新起一行用 `$$` 定界；或用 `math` 围栏代码块（三反引号后接 math），此时无需 `$$`。
   - 美元符号转义：数学表达式内部用 `\$`；数学表达式之外但处于同一行时用 `<span>$</span>`。
   - 官方示例覆盖：`\sqrt{3x-1}`、`(1+x)^2`、`\left( \sum_{k=1}^n a_k b_k \right)^2`、`\leq`。
   - `.md` 文件中的换行提示：块级公式前的文本换行需以行尾反斜杠等方式构造（本仓库写作统一在 `$$` 块前后留空行，规避此问题）。
2. **[G2] GitHub Blog《Math support in Markdown》**（2022-05-19 发布，2024-05-21 更新）
   <https://github.blog/news-insights/product-news/math-support-in-markdown/>
   - `$` / `$$` 定界符原生支持，TeX/LaTeX 语法，MathJax 渲染。
3. **[G3] GitHub Changelog《Fenced block syntax for mathematical expressions》**（2022-06-28）
   <https://github.blog/changelog/2022-06-28-fenced-block-syntax-for-mathematical-expressions/>
   - `math` 围栏代码块语法可用（本仓库白名单仍统一用 `$$`，见 STYLE_GUIDE §3.1 禁止项说明）。
4. **[M1] MathJax Docs《ams》扩展**
   <https://docs.mathjax.org/en/latest/input/tex/extensions/ams.html>
   - "The ams extension implements AMS math environments and macros … This extension is already loaded in all the components that include the TeX input jax, other than `input/tex-base`"。
   - 环境清单明确含：`aligned`、`cases`、`bmatrix`、`pmatrix`（另含 align、equation、gather、split 等，未收入白名单）。
   - 宏清单含：`\lVert`、`\rVert`、`\operatorname`、`\boxed` 等。
5. **[M2] MathJax Docs《Supported TeX/LaTeX commands》**
   <https://docs.mathjax.org/en/latest/input/tex/macros/index.html>
   - base 宏含：`\text`、`\mathbb`、`\mathrm`、`\sqrt`、`\langle`、`\rangle`、`\bar`、`\vert`、`\Vert`、`\frac`、`\sum`、`\prod` 等。
   - 2026-11-10 逐项核对同页存在（提取该页全部 1536 个命令 token 逐一比对）：希腊字母小写常用集（`\alpha`-`\omega` 含 `\varepsilon`/`\vartheta`/`\varphi`/`\iota`/`\upsilon`；变体 `\varsigma`/`\varpi`/`\varrho`/`\varkappa` 存在但未收）与大写 `\Gamma`/`\Delta`/`\Theta`/`\Lambda`/`\Pi`/`\Sigma`/`\Phi`/`\Psi`/`\Omega`；`\to`/`\rightarrow`/`\mapsto`/`\approx`/`\sim`/`\simeq`/`\propto`/`\equiv`/`\ll`/`\gg`/`\subset`/`\subseteq`/`\supset`/`\supseteq`；`\dfrac`/`\tfrac`/`\int`/`\oint`/`\bigcup`/`\bigcap`；`\partial`/`\nabla`/`\hat`/`\tilde`/`\vec`/`\dot`/`\ddot`/`\overline`/`\underline`/`\widehat`/`\widetilde`；`\left`/`\right`；`\mathcal`/`\operatorname`/`\stackrel`/`\underbrace`/`\overbrace`；`\times`/`\setminus`/`\forall`/`\exists`/`\oplus`/`\oslash`；`\ldots`/`\cdots`/`\vdots`/`\ddots`；薄空格 `\,`；`\{`、`\}` 与裸竖线符号（该页符号表）；大写箭头 `\Rightarrow`/`\Longrightarrow`/`\Leftarrow`/`\Leftrightarrow`/`\Longleftrightarrow`、间距 `\;`/`\:`/`\!`、`\Pr`、`\binom`/`\dbinom`/`\tbinom`（后三者亦见 [M1]）。
   - 注：`\operatorname*`（星号变体，limits 式）未在该页与 ams 页单列，列为 P15 重点复核项。

**官方文档未覆盖、需本文件实测确认的边界**：

- 行内 `$...$` 紧贴中文字符时的解析行为：GitHub 官方文档未说明；社区报告（如中文标点紧邻强调标记渲染失败的同类问题）表明定界符 flanking 判定可能失败。写作规则已定为「行内公式与中文之间一律加空格」，本文件 P12 复核该边界。
- `\text{}` 内放中文（P7b）。

## 探测用例

每个用例给出「源码」与「实时渲染」（推送 GitHub 后的实际渲染区），并注明「预期结果」。

### P1 · display 块 `$$...$$`（含 `\frac`、`\sum_{j=1}^{n}`、上下标、`\mathrm{}`）

源码：

```tex
$$
\mathrm{softmax}(s_i) = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}
$$
```

实时渲染：

$$
\mathrm{softmax}(s_i) = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}
$$

预期结果：独立成行的居中大公式；分数线、求和号上下限、上标、`\mathrm` 正体均正确显示。

### P2 · 行内 `$...$`（含 `\sqrt{}`、下标），与中文之间加空格

源码：`缩放因子 $\sqrt{d_k}$ 防止内积随维度增大。`

实时渲染：缩放因子 $\sqrt{d_k}$ 防止内积随维度增大。

预期结果：行内小公式正常渲染，与前后中文混排。

### P3 · `aligned` 环境（在 `$$` 块内）

源码：

```tex
$$
\begin{aligned}
s_{ij} &= q_i^{\top} k_j \\
\alpha_{ij} &= \frac{\exp(s_{ij})}{\sum_{l=1}^{T} \exp(s_{il})}
\end{aligned}
$$
```

实时渲染：

$$
\begin{aligned}
s_{ij} &= q_i^{\top} k_j \\
\alpha_{ij} &= \frac{\exp(s_{ij})}{\sum_{l=1}^{T} \exp(s_{il})}
\end{aligned}
$$

预期结果：两行公式在 `&`（等号）处对齐；`\\` 换行生效。

### P4 · `cases` 环境

源码：

```tex
$$
m_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$
```

实时渲染：

$$
m_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

预期结果：左侧大括号、两行分段、条件列对齐均正确。

### P5 · `bmatrix` 环境

源码：

```tex
$$
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}
$$
```

实时渲染：

$$
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}
$$

预期结果：方括号矩阵，2×2 元素就位。

### P6 · `pmatrix` 环境

源码：

```tex
$$
v = \begin{pmatrix} v_1 \\ v_2 \\ v_3 \end{pmatrix}
$$
```

实时渲染：

$$
v = \begin{pmatrix} v_1 \\ v_2 \\ v_3 \end{pmatrix}
$$

预期结果：圆括号列向量，三行就位。

### P7a · `\text{}`（ASCII 内容）

源码：

```tex
$$
\mathrm{loss} = -\log p_{\text{target}}
$$
```

实时渲染：

$$
\mathrm{loss} = -\log p_{\text{target}}
$$

预期结果：下标内 `target` 以正体文字显示，不倾斜、不断字母间距。

### P7b · `\text{}` 内放中文（边界探测，官方文档未说明）

源码：

```tex
$$
q_i \ge 0 \quad \text{对所有 } i
$$
```

实时渲染：

$$
q_i \ge 0 \quad \text{对所有 } i
$$

预期结果：中文「对所有」在公式内正体显示。**若失败**：白名单 `\text{}` 条目加注「仅放 ASCII」，中文注释放到公式块外的正文。

### P8 · `\mathbb{R}`

源码：`嵌入向量 $x \in \mathbb{R}^{d}$ 的每个分量是标量。`

实时渲染：嵌入向量 $x \in \mathbb{R}^{d}$ 的每个分量是标量。

预期结果：黑板粗体 R，上标 d 就位。

### P9 · `\bar\alpha`

源码：`记 $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ 为累积乘积。`

实时渲染：记 $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ 为累积乘积。

预期结果：α 上方横线、连乘号上下限正确。

### P10 · `\langle q, k \rangle`

源码：`$s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_k}$ 是未缩放的注意力分数。`

实时渲染： $s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_k}$ 是未缩放的注意力分数。

预期结果：尖括号内积符号正确（不是小于/大于号）。

### P11 · 范数与竖线（`\lVert \rVert`、`\vert`、`\Vert`、`\mid`）

源码：`残差分支要求 $\lVert x \rVert_2$ 有界；条件概率 $p(y \mid x)$ 用关系竖线；单竖线 $\vert x \vert$、双竖线 $\Vert x \Vert$ 亦直接书写；集合 $\lbrace x \mid x > 0 \rbrace$ 与裸竖线 $|x|$ 为 base symbol 直写。`

实时渲染：残差分支要求 $\lVert x \rVert_2$ 有界；条件概率 $p(y \mid x)$ 用关系竖线；单竖线 $\vert x \vert$、双竖线 $\Vert x \Vert$ 亦直接书写；集合 $\lbrace x \mid x > 0 \rbrace$ 与裸竖线 $|x|$ 为 base symbol 直写。

预期结果：范数显示为双竖线、下标 2 就位；`\mid` 两侧带关系间距；`\vert`/`\Vert` 显示为单 / 双竖线；`\lbrace`/`\rbrace` 显示为花括号，裸 `|` 显示为单竖线。

**2026-11-13 复核**：初版用例中的 `$\|x\|_2$` 与 `$\{ ... \}$` 实测被反转义陷阱破坏（`\|`→`|` 双竖线变单竖线；`\{`/`\}`→`{}` 花括号不显示）——两种写法已从白名单移除并回改全书；本用例已更新为替代写法，`\lbrace`/`\rbrace`/`\Vert` 渲染正确（见 P20/P21）。

### P12 · 行内 `$...$` 与中文相邻的边界（官方文档未说明，重点核对）

- P12a 紧贴无空格。源码：`缩放因子$\sqrt{d_k}$防止内积过大。`
  实时渲染：缩放因子$\sqrt{d_k}$防止内积过大。
  预期结果：**未知**。若不渲染（显示原始 `$` 字符），确认写作规则「行内公式与中文之间必须加空格」不可违反。
- P12b 两侧加空格。源码：`缩放因子 $\sqrt{d_k}$ 防止内积过大。`
  实时渲染：缩放因子 $\sqrt{d_k}$ 防止内积过大。
  预期结果：正常渲染。
- P12c `` $`...`$ `` 语法。源码：`` 缩放因子 $`\sqrt{d_k}`$ 防止内积过大。 ``
  实时渲染：缩放因子 $`\sqrt{d_k}`$ 防止内积过大。
  预期结果：正常渲染（官方记载用于含 Markdown 冲突字符的表达式）。
- P12d 中文标点相邻。源码：`归一化权重为$\alpha_{ij}$，对$v_j$加权求和。`
  实时渲染：归一化权重为$\alpha_{ij}$，对$v_j$加权求和。
  预期结果：**未知**。若 `，` 与 `$` 相邻导致不渲染，写作规则追加「行内公式不得紧贴中文标点」。
- P12e 定界符内侧空格（`$` 内侧不留空格规则的对照）。源码：`$ \sqrt{d_k} $`
  实时渲染：$ \sqrt{d_k} $
  预期结果：**未知**（社区惯例为不渲染；官方文档未明文）。若不渲染，确认写作规则「`$` 内侧不留空格」不可违反。

### P13 · 其余基础函数 / 关系与运算符

说明：`\exp`、`\log`、`\top`、`\le`、`\infty`、`\ge`、`\quad`、`\in` 已由 P1/P3/P4/P7/P8 覆盖；本用例覆盖白名单其余基础宏（`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\neq`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\qquad`）。全部为 [M2] 直接列出的 base 宏（2026-11-10 逐项核对存在）。

源码：

```tex
$$
\ln x, \quad \sin x, \quad \cos x, \quad \tanh x, \quad \min_{i} a_i, \quad \max_{i} a_i, \quad \sup_{i} a_i, \quad \inf_{i} a_i
$$

$$
a \neq b, \quad \pm 1, \quad a \cdot b, \quad A \odot B, \quad A \otimes B, \quad x \qquad y
$$
```

实时渲染：

$$
\ln x, \quad \sin x, \quad \cos x, \quad \tanh x, \quad \min_{i} a_i, \quad \max_{i} a_i, \quad \sup_{i} a_i, \quad \inf_{i} a_i
$$

$$
a \neq b, \quad \pm 1, \quad a \cdot b, \quad A \odot B, \quad A \otimes B, \quad x \qquad y
$$

预期结果：两行均正常渲染；`\min`/`\max`/`\sup`/`\inf` 在 display 模式下下标移到正下方；`\odot`/`\otimes` 显示为带圈点乘 / 叉乘。MathJax 遇未知命令会红色报错，逐项目检即可定位失败宏。

### P14 · 希腊字母（基础命令，[M2] 直接证据）

源码：

```tex
$$
\alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta, \iota, \kappa, \lambda, \mu, \nu, \xi, \pi, \rho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega
$$

$$
\Gamma, \Delta, \Theta, \Lambda, \Pi, \Sigma, \Phi, \Psi, \Omega
$$
```

实时渲染：

$$
\alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta, \iota, \kappa, \lambda, \mu, \nu, \xi, \pi, \rho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega
$$

$$
\Gamma, \Delta, \Theta, \Lambda, \Pi, \Sigma, \Phi, \Psi, \Omega
$$

预期结果：小写 26 个、大写 9 个希腊字形全部显示；未知命令红色报错，逐项目检定位。

### P15 · 关系/箭头/积分/微分/修饰/伸缩定界符/其他（基础命令，[M2] 直接证据；`\operatorname*` 为重点复核项）

源码：

```tex
$$
x \approx y, \quad x \sim y, \quad x \simeq y, \quad x \propto y, \quad x \equiv y, \quad a \ll b, \quad a \gg b, \quad A \subset B, \quad A \subseteq B, \quad A \supset B, \quad A \supseteq B
$$

$$
x \to y, \quad x \rightarrow y, \quad x \mapsto y, \quad \int_0^1 f, \quad \oint_C f, \quad \bigcup_{i} A_i, \quad \bigcap_{i} A_i, \quad \dfrac{a}{b}, \quad \tfrac{a}{b}
$$

$$
\partial_i f, \quad \nabla f, \quad \hat{x}, \quad \tilde{x}, \quad \vec{x}, \quad \dot{x}, \quad \ddot{x}, \quad \overline{x}, \quad \underline{x}, \quad \widehat{x}, \quad \widetilde{x}
$$

$$
\left( \frac{a}{b} \right), \quad \left[ x \right], \quad \left\lbrace x \right\rbrace, \quad \left| x \right|, \quad \left\Vert x \right\Vert, \quad \left\langle q, k \right\rangle, \quad \left. \frac{\partial f}{\partial x} \right|_{x=0}
$$

$$
\mathcal{L}(\theta), \quad \operatorname{arg\thinspace min}_{\theta} \mathcal{L}(\theta), \quad x \stackrel{\mathrm{def}}{=} y, \quad \underbrace{\frac{1}{n} \sum_{i=1}^{n} L_i}_{\text{mean}}, \quad \overbrace{x_1 + \cdots + x_n}^{n}, \quad A \times B, \quad A \setminus B, \quad A \oplus B, \quad A \oslash B, \quad \forall x, \quad \exists y, \quad x_1, \ldots, x_n, \quad \vdots, \quad \ddots
$$
```

实时渲染：

$$
x \approx y, \quad x \sim y, \quad x \simeq y, \quad x \propto y, \quad x \equiv y, \quad a \ll b, \quad a \gg b, \quad A \subset B, \quad A \subseteq B, \quad A \supset B, \quad A \supseteq B
$$

$$
x \to y, \quad x \rightarrow y, \quad x \mapsto y, \quad \int_0^1 f, \quad \oint_C f, \quad \bigcup_{i} A_i, \quad \bigcap_{i} A_i, \quad \dfrac{a}{b}, \quad \tfrac{a}{b}
$$

$$
\partial_i f, \quad \nabla f, \quad \hat{x}, \quad \tilde{x}, \quad \vec{x}, \quad \dot{x}, \quad \ddot{x}, \quad \overline{x}, \quad \underline{x}, \quad \widehat{x}, \quad \widetilde{x}
$$

$$
\left( \frac{a}{b} \right), \quad \left[ x \right], \quad \left\lbrace x \right\rbrace, \quad \left| x \right|, \quad \left\Vert x \right\Vert, \quad \left\langle q, k \right\rangle, \quad \left. \frac{\partial f}{\partial x} \right|_{x=0}
$$

$$
\mathcal{L}(\theta), \quad \operatorname{arg\thinspace min}_{\theta} \mathcal{L}(\theta), \quad x \stackrel{\mathrm{def}}{=} y, \quad \underbrace{\frac{1}{n} \sum_{i=1}^{n} L_i}_{\text{mean}}, \quad \overbrace{x_1 + \cdots + x_n}^{n}, \quad A \times B, \quad A \setminus B, \quad A \oplus B, \quad A \oslash B, \quad \forall x, \quad \exists y, \quad x_1, \ldots, x_n, \quad \vdots, \quad \ddots
$$

预期结果：五行全部渲染；`\left...\right` 定界符随内容伸缩，`\left. \right|` 只显示右竖线；`\underbrace`/`\overbrace` 带上下标注；`\dfrac` 大分式、`\tfrac` 小分式；`\operatorname{arg\thinspace min}` 显示直立算子名。

**2026-11-13 复核**：初版用例实测两处失败——`\operatorname*`（星号变体）在 GitHub 的 MathJax 配置下报「The following macros are not allowed: operatorname」，**星号变体禁用、一律用 `\operatorname`**；`\left\{ x \right\}` 中 `\{` 被反转义为 `{` 导致 `\left{` 报「Missing or unrecognized delimiter for \left」——花括号一律用 `\lbrace`/`\rbrace` 形式（本用例已更新）。

### P16 · 大写箭头 / 间距 / 二项式 / `\Pr`（基础命令，[M2] 直接证据；`\binom` 族亦见 [M1]）

源码：

```tex
$$
a = b \Rightarrow c = d, \quad a = b \Longrightarrow c = d, \quad P \Leftarrow Q, \quad P \Leftrightarrow Q, \quad P \Longleftrightarrow Q
$$

$$
x\,y, \quad x\:y, \quad x\;y, \quad x\!y
$$

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}, \quad \dbinom{n}{k}, \quad \tbinom{n}{k}, \quad \Pr(A \mid B) = \frac{\Pr(A \cap B)}{\Pr(B)}
$$
```

实时渲染：

$$
a = b \Rightarrow c = d, \quad a = b \Longrightarrow c = d, \quad P \Leftarrow Q, \quad P \Leftrightarrow Q, \quad P \Longleftrightarrow Q
$$

$$
x\,y, \quad x\:y, \quad x\;y, \quad x\!y
$$

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}, \quad \dbinom{n}{k}, \quad \tbinom{n}{k}, \quad \Pr(A \mid B) = \frac{\Pr(A \cap B)}{\Pr(B)}
$$

预期结果：三行全部渲染；间距行四处间距宽窄可辨（`\!` 最窄、`\;` 最宽）；`\binom` 带括号上下堆叠，`\dbinom` 大字号、`\tbinom` 小字号；`\Pr` 正体算子、`\mid` 条件竖线间距正确。

### P17 · `\big` / `\Big` / `\bigg` / `\Bigg` 固定尺寸定界符（基础命令，[M2] 直接证据；W1 波末随白名单收编补入，2026-11-11）

源码：

```tex
$$
\big( x + y \big), \quad \Big( \frac{a}{b} \Big), \quad \bigg( \sum_{i=1}^{n} x_i \bigg), \quad \Bigg( \dfrac{a}{\dfrac{b}{c}} \Bigg)
$$
```

实时渲染：

$$
\big( x + y \big), \quad \Big( \frac{a}{b} \Big), \quad \bigg( \sum_{i=1}^{n} x_i \bigg), \quad \Bigg( \dfrac{a}{\dfrac{b}{c}} \Bigg)
$$

预期结果：四组括号全部渲染，尺寸逐档递增可辨；不渲染或尺寸不递增则回改 2.2 式 (4)(13) 与 2.6 式 (5)(6) 改用 `\left(...\right)`。

### P18 · `\lceil` / `\rceil` / `\ne` / `\leftarrow` / `\cup` / `\circ` / `\perp` / `\square` 补充符号复合用例（基础命令，[M2] 直接证据；W2 波末随白名单收编补入，2026-11-12）

源码：

```tex
$$
B_c = \lceil M / (4d) \rceil, \quad i \ne j, \quad h \leftarrow 0, \quad A \cup B, \quad (f \circ g)(x), \quad u \perp k, \quad \square
$$
```

实时渲染：

$$
B_c = \lceil M / (4d) \rceil, \quad i \ne j, \quad h \leftarrow 0, \quad A \cup B, \quad (f \circ g)(x), \quad u \perp k, \quad \square
$$

预期结果：单复合行全部渲染——`\lceil\rceil` 为取整括号、`\ne` 为不等号（与 `\neq` 同形）、`\leftarrow` 为左箭头、`\cup` 为并集、`\circ` 为复合圆圈、`\perp` 为正交符、`\square` 为空心方块（证毕符）。任一符号红色报错则从白名单移除该符号并回改对应 W2 章节（3.1 式 (9)(10)(16)、4.1 §5、4.3 §5/§9/§10、4.5 式 (11)、4.7 式 (9)(10)）。

### P19 · `\det` / `\varnothing` 复合用例（基础命令，[M2] 直接证据；W4 波末随白名单收编补入，2026-11-12）

源码：

```tex
$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|, \quad \varepsilon_\theta(x_t, t, \varnothing)
$$
```

实时渲染：

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|, \quad \varepsilon_\theta(x_t, t, \varnothing)
$$

预期结果：`\det` 渲染为直立函数名 det（行列式）、`\varnothing` 渲染为空集符 ∅；任一红色报错则从白名单移除该符号并回改对应章节（`\det`：7.1 式 (4) 及同节正文、练习 2；`\varnothing`：4.2 式 (9)、7.4 §4 CFG 段式 (17)(18)(20)(23) 及正文）。

### P20 · 2026-11-13 收编命令复合用例（[M2] 直接证据 + 渲染实测 [R1]）

源码：

```tex
$$
\ell \notin \mathbb{R}, \quad A \cap B, \quad \mathbf{x}, \quad a \leftrightarrow b, \quad b \mp c, \quad \angle \theta, \quad \lfloor x \rfloor, \quad u \longrightarrow v, \quad i^{\ast}
$$
```

实时渲染：

$$
\ell \notin \mathbb{R}, \quad A \cap B, \quad \mathbf{x}, \quad a \leftrightarrow b, \quad b \mp c, \quad \angle \theta, \quad \lfloor x \rfloor, \quad u \longrightarrow v, \quad i^{\ast}
$$

预期结果：全部渲染——`\ell` 为花体 ℓ、`\notin` 为斜杠属于号、`\cap` 交集、`\mathbf` 粗体、双箭头/正负号/角符号/下取整/长箭头/星号各就各位。

### P21 · 反转义陷阱替代写法与保护形式（2026-11-13 新增，全部实测）

实时渲染（每行一种确认可用的写法）：

- 花括号： $\lbrace 0, 1 \rbrace$；双竖线： $\Vert x \Vert$；细间距： $a \thinspace b$；5mu 间距： $a \hspace{5mu} b$；负间距： $a \negthinspace b$；控制空格： $a \ b$。
- backtick 保护形式（内容含 `*` / `\%` / `\#` / 标点后 `_` 时使用，可紧贴中文）：$`x^{*}`$ 、 $`15\%`$ 、紧贴中文$`x`$测试、$`\mathrm{Attn}_{\ell}`$。
- display 内星号：$$i^{\ast}$$（display 无 backtick 形式；display 内 `\%`/`\#` 用 `\text{％}`/`\text{＃}`）。

**禁用写法及实测后果**：`\{`/`\}`→花括号不显示；`\|`→双竖线变单竖线；`\,`/`\;`/`\!`→显示为逗号/分号/叹号；`\%`→`%` 注释符吞掉后文；`\#`→MathJax 报错；行内 `$...$` 含裸 `*` 或标点后 `_`（`}_`、`'_` 等）→强调配对吞内容；display 内裸 `*`→变成 `_`（display 内 `_` 不受影响）。`\medspace`、`\thickspace` MathJax 不支持（显示为字面文本），同样禁用。


### P22 · 行内开口 `$` 的前字符全集（2026-11-13 二轮实测，临时 gist 探测）

**结论**：开口 `$` 的前一个字符只允许是**空白、行首、`(`、Markdown 元素边界**（`**` 粗体、code span、另一个公式之后，因为元素边界使公式位于文本节点开头）；紧贴其余任何字符一律不解析、渲染为裸源码。实测失败的前字符：ASCII 字母（`aa$x$`）、数字（`a1$x$`）、`-` `/` `=` `+` `<` `>` `*` `'` `"` `!` `?` `:` `;` `,` `.` `[` `{`、`–` `—` `×` `÷` `→`、CJK 字符、全角标点（`，` `（`）。

实时渲染（安全写法对照）：

- 空白前置：开口前留空格 $x$ 渲染正常；`(` 前置：（ $x$ ）渲染正常；元素边界：**$x$ = 1** 渲染正常。
- 连字复合词用 backtick 保护保持紧凑：rank-$`d_k`$、top-$`k`$、score/$`\varepsilon`$；字母相邻只能加空格（见 P24）。

### P23 · 行内闭口 `$` 的边界（2026-11-13 二轮实测）

**结论**：闭口 `$` 前不能是空白；后不能紧跟 ASCII 字母或数字；后紧跟中文、全角/半角标点（`.` `,` `)` `-` `/` `%` `+` `=`）、行尾均渲染正常。

实时渲染（安全写法对照）：

- 单位收进公式内部： $\approx 0.79\text{M}$（替代 `...$M`）；比较式 $>128$（替代 `$>$128`）。

实测失败（均显示为裸源码）：`$0.79$M`（闭口后字母）、`$x$1`（闭口后数字）、`$x = $`（闭口头空白）、`$ x$`（开口后空白）。

### P24 · backtick 保护形式的边界豁免（2026-11-13 二轮实测）

**结论**：backtick 形式开口豁免 flanking——可紧贴中文、全角括号、`(`、`-`、`/`、`=`；但仍**不能紧贴 ASCII 字母或数字**（`aa$`x`$` 不解析）。闭口豁免字母/数字限制：$`x`$M、$`x`$1 渲染正常。内容以数字开头正常：$`1 + 1`$。

### P25 · 结构限制（2026-11-13 二轮实测）

- **行内矩阵环境不解析**：行内 `$...$` 含 `\begin{bmatrix}` 整体不渲染——矩阵一律用 `$$` display（display 块全书零失败）。
- **表格单元格内裸 `|` 切分单元格**：GFM 表格先于数学解析执行，公式内的裸 `|` 会把单元格一切为二——表格内公式写 `\|`（表格转义还原为 `|` 后进 MathJax），如表格行内 $z_{\|c\|+t}$ 渲染正常；范数双竖线语义在表格内直接写 `\Vert`（`\|` 经表格转义后只剩单竖线）。
- **区间写法**：`$A$–$B$` 的第二段开口紧贴 `–` 不解析——合并为单公式： $0.8\text{–}0.94$、 $10^{4}\text{–}10^{5}$ 渲染正常。
- **嵌套圆括号**：`Bernoulli($\Phi(x)$)`（公式内嵌套圆括号且外层紧跟括号）不解析——函数名收进公式： $\mathrm{Bernoulli}(\Phi(x))$ 渲染正常。

## 核对清单

| 用例 | 覆盖的白名单构造 | 预期 | 核对结果 |
| --- | --- | --- | --- |
| P1 | display `$$...$$`；`\frac`；`\sum_{j=1}^{n}`；上下标；`\mathrm{}` | 渲染 | 渲染正常 ✓ |
| P2 | 行内 `$...$`；`\sqrt{}`；下标 | 渲染 | 渲染正常 ✓ |
| P3 | `aligned` | 渲染 | 渲染正常 ✓ |
| P4 | `cases` | 渲染 | 渲染正常 ✓ |
| P5 | `bmatrix` | 渲染 | 渲染正常 ✓ |
| P6 | `pmatrix` | 渲染 | 渲染正常 ✓ |
| P7a | `\text{}`（ASCII） | 渲染 | 渲染正常 ✓ |
| P7b | `\text{}` 内中文（边界） | 未知，待实测 | 渲染正常 ✓（中文 \text 可用） |
| P8 | `\mathbb{R}` | 渲染 | 渲染正常 ✓ |
| P9 | `\bar\alpha`；`\prod_{s=1}^{t}` | 渲染 | 渲染正常 ✓ |
| P10 | `\langle q, k \rangle` | 渲染 | 渲染正常 ✓ |
| P11 | `\lVert x \rVert`；`\|x\|`；`\vert`；`\Vert`；`\mid`；裸 `\{`、`\}`、`\|` | 渲染 | 部分失败已处置：`\|`→`|`（双竖线变单竖线）、`\{`/`\}` 花括号不显示——从白名单移除并回改全书；用例已改写为 `\lbrace`/`\rbrace`/`\Vert` ✓ |
| P12a-e | 行内 `$...$` 与中文/中文标点相邻、`$` 内侧空格（边界） | a/d/e 不渲染，b/c 渲染 | 符合预期 ✓（a/d/e 为故意失败对照例；规则已由 P22-P24 扩展为完整边界规则） |
| P13 | 基础函数/关系与运算符：`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\neq`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\qquad` | 渲染 | 渲染正常 ✓ |
| P14 | 希腊字母小写常用集 26 个 + 大写 9 个（基础命令，[M2] 直接证据） | 渲染 | 渲染正常 ✓ |
| P15 | 关系/箭头/近似；`\dfrac`/`\tfrac`/`\int`/`\oint`/`\bigcup`/`\bigcap`；`\partial`/`\nabla`/修饰符号；`\left...\right` 伸缩定界符；`\mathcal`/`\operatorname*`/`\stackrel`/`\underbrace`/`\overbrace`/`\times`/`\setminus`/`\forall`/`\exists`/`\oplus`/`\oslash`/省略号/`\,`（基础命令，[M2] 直接证据） | 渲染；`\operatorname*` 重点复核 | 两处失败已处置：`\operatorname*` 报 macros are not allowed（禁用星号变体）；`\left\{` 因反转义报缺定界符（改用 `\left\lbrace`）；其余渲染正常 ✓ |
| P16 | 大写箭头（`\Rightarrow` 等 5 个）；间距 `\:`、`\;`、`\!`（`\,` 见 P15）；`\binom`/`\dbinom`/`\tbinom`；`\Pr`（基础命令，[M2] 直接证据） | 渲染 | 源码层面渲染但内容被反转义破坏：`\,`→`,`、`\;`→`;`、`\!`→`!`（间距变标点）——三种写法禁用，替代为 `\thinspace`/`\hspace{5mu}`/`\negthinspace`；其余渲染正常 ✓ |
| P17 | `\big` / `\Big` / `\bigg` / `\Bigg` 固定尺寸定界符（基础命令，[M2] 直接证据；2026-11-11 随 W1 波末白名单收编补入） | 渲染且尺寸逐档递增 | 渲染正常 ✓ |
| P18 | `\lceil` / `\rceil` / `\ne` / `\leftarrow` / `\cup` / `\circ` / `\perp` / `\square` 补充符号复合用例（基础命令，[M2] 直接证据；2026-11-12 随 W2 波末白名单收编补入） | 渲染 | 渲染正常 ✓ |
| P19 | `\det` / `\varnothing` 复合用例（基础命令，[M2] 直接证据；2026-11-12 随 W4 波末白名单收编补入） | 渲染 | 渲染正常 ✓ |
| P20 | 2026-11-13 收编命令复合用例（`\ell`/`\notin`/`\cap`/`\mathbf`/`\leftrightarrow`/`\mp`/`\angle`/`\lfloor`/`\rfloor`/`\longrightarrow`/`\ast`） | 渲染 | 渲染正常 ✓ |
| P21 | 反转义替代写法与保护形式（`\lbrace`/`\rbrace`/`\Vert`/`\thinspace`/`\hspace{5mu}`/`\negthinspace`/`` $`...`$ ``） | 渲染 | 渲染正常 ✓（临时 gist 探测 + 本页复核双重确认） |

| P22 | 开口 `$` 前字符全集（白名单：空白/行首/`(`/元素边界） | 白名单外均不渲染 | 符合预期 ✓（临时 gist 探测确认） |
| P23 | 闭口 `$` 边界（前非空白、后非字母数字） | 违规均不渲染 | 符合预期 ✓（同上） |
| P24 | backtick 形式边界豁免（开口豁免 flanking 但不紧贴字母数字；闭口豁免字母数字） | 豁免成立 | 渲染正常 ✓（同上） |
| P25 | 结构限制（行内矩阵环境、表格裸 `|`、区间合并、嵌套圆括号） | 违规不渲染/安全写法渲染 | 符合预期 ✓（全书失败实例定位 + 安全写法复核） |

注：本表 code span 内的 `\|` 是 GFM 表格转义写法，渲染后显示为单个竖线字符（反斜杠不显示）；真实 LaTeX 源码以 P11 源码区为准。

## 核对记录

| 日期 | 核对人 | 结论 | 失败项及处置 |
| --- | --- | --- | --- |
| 2026-11-13 | Codex（gh api 渲染管线 + agent-browser 检视 MathJax 输出） | P1-P19 全部核对；P12a/d/e 按预期不渲染（规则成立）、P12b/c 与其余用例渲染正常；新发现并处置三处：反转义陷阱（`\{`/`\|`/`\,`/`\;`/`\!`/`\%`/`\#`）、`*` 强调冲突、`\operatorname*` 被禁——STYLE_GUIDE §3/§3.1 已更新，全书 51 章已回改 | 失败项及处置见 P11/P15/P16 行与 §3；新增 P20/P21 复核通过 |
| 2026-11-13（二轮） | Codex（全书 51 章渲染 HTML 残差扫描 + 临时 gist 边界探测 ×2） | 一轮修复后残差 144 处全部归因并修复：开口前字符白名单收窄为「空白/行首/`(`/元素边界」（P22）、闭口后禁跟字母数字（P23）、backtick 形式边界豁免（P24）、结构限制（P25）；新增 `scripts/lint-math-boundary.py` 机械检查，51 章 + README + references 全过 | 失败写法已回改全书：连字复合词 36 处改 backtick、算符/标点前补空格 19 处、单位收进 `\text{}` 12 处、区间合并 7 处、结构修复 7 处（行内矩阵改 display、`$>128$`、内容尾部空格、嵌套圆括号、表格范数 `\Vert`、表格裸 `|` ×2） |
