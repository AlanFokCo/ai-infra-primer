# GitHub MathJax 渲染探测（math-render-probe）

> **用途**：本仓库暂无远端，GitHub 的 MathJax 渲染效果无法直接观测。本文件为 `STYLE_GUIDE.md` §3.1 公式语法白名单中的每一项构造提供一个最小用例（P1-P19）。
> **首次推送 GitHub 后，必须以渲染视图打开本文件、逐项人工核对一次**；任一用例渲染失败，将对应构造从 `STYLE_GUIDE.md` §3.1 白名单移除，并回改已使用它的章节。核对结果填入文末「核对记录」表。

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

实时渲染：$s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_k}$ 是未缩放的注意力分数。

预期结果：尖括号内积符号正确（不是小于/大于号）。

### P11 · 范数与竖线（`\lVert \rVert`、`\|`、`\vert`、`\Vert`、`\mid`）

源码：`残差分支要求 $\lVert x \rVert_2$ 有界；简写 $\|x\|_2$ 等价；条件概率 $p(y \mid x)$ 用关系竖线；单竖线 $\vert x \vert$、双竖线 $\Vert x \Vert$ 亦直接书写；集合 $\{ x \mid x > 0 \}$ 与裸竖线 $|x|$ 为 base symbol 直写。`

实时渲染：残差分支要求 $\lVert x \rVert_2$ 有界；简写 $\|x\|_2$ 等价；条件概率 $p(y \mid x)$ 用关系竖线；单竖线 $\vert x \vert$、双竖线 $\Vert x \Vert$ 亦直接书写；集合 $\{ x \mid x > 0 \}$ 与裸竖线 $|x|$ 为 base symbol 直写。

预期结果：两种范数写法显示为双竖线、下标 2 就位；`\mid` 两侧带关系间距；`\vert`/`\Vert` 显示为单 / 双竖线；`\{`、`\}` 显示为花括号，裸 `|` 显示为单竖线。

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
\left( \frac{a}{b} \right), \quad \left[ x \right], \quad \left\{ x \right\}, \quad \left| x \right|, \quad \left\| x \right\|, \quad \left\langle q, k \right\rangle, \quad \left. \frac{\partial f}{\partial x} \right|_{x=0}
$$

$$
\mathcal{L}(\theta), \quad \operatorname*{arg\,min}_{\theta} \mathcal{L}(\theta), \quad x \stackrel{\mathrm{def}}{=} y, \quad \underbrace{\frac{1}{n} \sum_{i=1}^{n} L_i}_{\text{mean}}, \quad \overbrace{x_1 + \cdots + x_n}^{n}, \quad A \times B, \quad A \setminus B, \quad A \oplus B, \quad A \oslash B, \quad \forall x, \quad \exists y, \quad x_1, \ldots, x_n, \quad \vdots, \quad \ddots
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
\left( \frac{a}{b} \right), \quad \left[ x \right], \quad \left\{ x \right\}, \quad \left| x \right|, \quad \left\| x \right\|, \quad \left\langle q, k \right\rangle, \quad \left. \frac{\partial f}{\partial x} \right|_{x=0}
$$

$$
\mathcal{L}(\theta), \quad \operatorname*{arg\,min}_{\theta} \mathcal{L}(\theta), \quad x \stackrel{\mathrm{def}}{=} y, \quad \underbrace{\frac{1}{n} \sum_{i=1}^{n} L_i}_{\text{mean}}, \quad \overbrace{x_1 + \cdots + x_n}^{n}, \quad A \times B, \quad A \setminus B, \quad A \oplus B, \quad A \oslash B, \quad \forall x, \quad \exists y, \quad x_1, \ldots, x_n, \quad \vdots, \quad \ddots
$$

预期结果：五行全部渲染；`\left...\right` 定界符随内容伸缩，`\left. \right|` 只显示右竖线；`\underbrace`/`\overbrace` 带上下标注；`\dfrac` 大分式、`\tfrac` 小分式；`\operatorname*{arg\,min}` 的 `\theta` 在 display 模式位于正下方——**若 `\operatorname*` 红色报错，白名单该行退化为 `\operatorname`（不带星）**。

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

## 核对清单

| 用例 | 覆盖的白名单构造 | 预期 | 核对结果 |
| --- | --- | --- | --- |
| P1 | display `$$...$$`；`\frac`；`\sum_{j=1}^{n}`；上下标；`\mathrm{}` | 渲染 | 待首次推送后填写 |
| P2 | 行内 `$...$`；`\sqrt{}`；下标 | 渲染 | 待首次推送后填写 |
| P3 | `aligned` | 渲染 | 待首次推送后填写 |
| P4 | `cases` | 渲染 | 待首次推送后填写 |
| P5 | `bmatrix` | 渲染 | 待首次推送后填写 |
| P6 | `pmatrix` | 渲染 | 待首次推送后填写 |
| P7a | `\text{}`（ASCII） | 渲染 | 待首次推送后填写 |
| P7b | `\text{}` 内中文（边界） | 未知，待实测 | 待首次推送后填写 |
| P8 | `\mathbb{R}` | 渲染 | 待首次推送后填写 |
| P9 | `\bar\alpha`；`\prod_{s=1}^{t}` | 渲染 | 待首次推送后填写 |
| P10 | `\langle q, k \rangle` | 渲染 | 待首次推送后填写 |
| P11 | `\lVert x \rVert`；`\|x\|`；`\vert`；`\Vert`；`\mid`；裸 `\{`、`\}`、`\|` | 渲染 | 待首次推送后填写 |
| P12a-e | 行内 `$...$` 与中文/中文标点相邻、`$` 内侧空格（边界） | a/d/e 未知，b/c 渲染 | 待首次推送后填写 |
| P13 | 基础函数/关系与运算符：`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\neq`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\qquad` | 渲染 | 待首次推送后填写 |
| P14 | 希腊字母小写常用集 26 个 + 大写 9 个（基础命令，[M2] 直接证据） | 渲染 | 待首次推送后填写 |
| P15 | 关系/箭头/近似；`\dfrac`/`\tfrac`/`\int`/`\oint`/`\bigcup`/`\bigcap`；`\partial`/`\nabla`/修饰符号；`\left...\right` 伸缩定界符；`\mathcal`/`\operatorname*`/`\stackrel`/`\underbrace`/`\overbrace`/`\times`/`\setminus`/`\forall`/`\exists`/`\oplus`/`\oslash`/省略号/`\,`（基础命令，[M2] 直接证据） | 渲染；`\operatorname*` 重点复核 | 待首次推送后填写 |
| P16 | 大写箭头（`\Rightarrow` 等 5 个）；间距 `\:`、`\;`、`\!`（`\,` 见 P15）；`\binom`/`\dbinom`/`\tbinom`；`\Pr`（基础命令，[M2] 直接证据） | 渲染 | 待首次推送后填写 |
| P17 | `\big` / `\Big` / `\bigg` / `\Bigg` 固定尺寸定界符（基础命令，[M2] 直接证据；2026-11-11 随 W1 波末白名单收编补入） | 渲染且尺寸逐档递增 | 待首次推送后填写 |
| P18 | `\lceil` / `\rceil` / `\ne` / `\leftarrow` / `\cup` / `\circ` / `\perp` / `\square` 补充符号复合用例（基础命令，[M2] 直接证据；2026-11-12 随 W2 波末白名单收编补入） | 渲染 | 待首次推送后填写 |
| P19 | `\det` / `\varnothing` 复合用例（基础命令，[M2] 直接证据；2026-11-12 随 W4 波末白名单收编补入） | 渲染 | 待首次推送后填写 |

注：本表 code span 内的 `\|` 是 GFM 表格转义写法，渲染后显示为单个竖线字符（反斜杠不显示）；真实 LaTeX 源码以 P11 源码区为准。

## 核对记录

| 日期 | 核对人 | 结论 | 失败项及处置 |
| --- | --- | --- | --- |
| （待首次推送 GitHub 后填写） | | | |
