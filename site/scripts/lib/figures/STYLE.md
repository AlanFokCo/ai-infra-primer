# 签名图创作规范（figures/*.mjs 必须遵守）

目标是论文级插图：精确、克制、信息密度高。动画只表现**机制**（数据流动、状态填充、旋转、脉冲），禁止入场动画与装饰性动效。

## 模块格式

`site/scripts/lib/figures/partNN-<topic>.mjs`:

```js
export const FIGURES = [
  {
    slug: '02-transformer-anatomy/04-causality-and-kv-cache',  // 章节 slug（目录/文件名）
    afterHeading: /^### 4\.4 有 cache 的 decode/,              // 插入锚点：该标题行之后
    title: 'KV cache 的写与读',                                 // 图左上角标签
    svg: () => `<svg class="fig-svg sig" viewBox="0 0 720 300" ...>...</svg>`,
  },
];
```

## 设计令牌（CSS 变量，预览页已注入）

- 底色 `--paper #f7f6f2`、面板 `--paper-raised #fdfcf9`、沉面 `--paper-sunk #efede6`
- 墨色 `--ink #181b20`、次 `--ink-soft #454b57`、弱 `--ink-faint #636a78`
- 细线 `--rule #d9d6cc`、更细 `--rule-soft #e7e5db`
- 唯一强调色：朱砂 `--seal #a83228`、淡染 `--seal-wash #f3e5e2`、深 `--seal-deep #8a2620`

## 字与尺寸

- 英文/数字：IBM Plex Mono（`font-family='IBM Plex Mono, ui-monospace, monospace'`），正文级 12px、注记 10.5px
- 中文：`'Noto Sans SC, PingFang SC, sans-serif'` 13px / 11px
- viewBox 宽 720（内容栏 624 + 余量），高度按内容 240–360；所有文字必须在 viewBox 内
- 模块 = 圆角 rect（rx 3–6）+ 1–1.2px 描边 + 水洗填充；连线 stroke `#a9a498` 1.2px 圆角端；箭头用实心小三角 path

## 角色配色（与引擎一致）

- attention/score/softmax 系：`fill #f6e9e5 stroke #cf9d92`
- norm 系：`fill #e9efe9 stroke #a9bfa9`
- 投影/线性系：`fill #e9edf3 stroke #aebcd2`
- 数据/输入输出系：`fill #f1efe8 stroke #c6c0af`
- cache/状态系：`fill #f3ecdc stroke #cbb98d`
- 通用 op：`fill #eeeef2 stroke #b6b6c4`

## 动画原语（必须用这套 class，门控与降级已全局处理）

- `class="sig-flow"` + 子 `<animateMotion path="M...">`：朱砂光点沿路流动（数据流）
- `class="sig-cell" style="--i:0.35"`：单元格按 --i 秒延迟填入（网格填充）
- `class="sig-bar" style="--i:0"`：条形 scaleX 生长（负载/账）
- `class="sig-rotor" style="--dur:6s"` + 行内 `transform-origin: Xpx Ypx`：绕点连续旋转（RoPE）
- `class="sig-diag" style="--i:0"`：脉冲呼吸（对角线/关键点强调）
- `class="sig-step" style="--i:0"`：步序点依次亮起
- `class="sig-dashflow"`：线体虚线流动（梯度/残差通路）
- 静态辅助元素不加 class 即静止。所有动画自动在入视口时启动、reduced-motion 时呈终态

## 自检（必须执行）

```bash
cd site && node scripts/fig-check.mjs <你的模块名>
```

然后逐张看 `qa/figs-<模块名>/fig-N.png`：无文字溢出 viewBox、无元素重叠、动画在截图时刻（3.5s）应已呈现完整语义（不依赖读者等待）、锚点正则与章节真实标题匹配。不达标就改到达标。

## 选题原则

图必须教东西：选该章最关键的机制（公式结构、数据流、状态演化、账的分解），不重复该章已有 ASCII 图已覆盖的内容（引擎已把 ASCII 转成图；签名图要画 ASCII 表达不了或表达得吃力的）。
