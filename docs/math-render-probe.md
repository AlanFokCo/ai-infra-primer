# GitHub MathJax 渲染探测（math-render-probe）

> **用途**：本仓库暂无远端，GitHub 的 MathJax 渲染效果无法直接观测。本文件为 `STYLE_GUIDE.md` §3.1 公式语法白名单中的每一项构造提供一个最小用例（P1-P13）。
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

### P11 · 范数 `\lVert x \rVert` 与 `\|x\|`

源码：`残差分支要求 $\lVert x \rVert_2$ 有界；简写 $\|x\|_2$ 等价。`

实时渲染：残差分支要求 $\lVert x \rVert_2$ 有界；简写 $\|x\|_2$ 等价。

预期结果：两种写法都显示为双竖线范数，下标 2 就位。

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
| P11 | `\lVert x \rVert`；`\|x\|` | 渲染 | 待首次推送后填写 |
| P12a-e | 行内 `$...$` 与中文/中文标点相邻、`$` 内侧空格（边界） | a/d/e 未知，b/c 渲染 | 待首次推送后填写 |
| P13 | 基础函数/关系与运算符：`\ln`、`\sin`、`\cos`、`\tanh`、`\min`、`\max`、`\sup`、`\inf`、`\neq`、`\pm`、`\cdot`、`\odot`、`\otimes`、`\qquad` | 渲染 | 待首次推送后填写 |

## 核对记录

| 日期 | 核对人 | 结论 | 失败项及处置 |
| --- | --- | --- | --- |
| （待首次推送 GitHub 后填写） | | | |
