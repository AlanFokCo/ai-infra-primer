# 场景插图创作规范（figures/*.mjs 必须遵守）

目标：data-on-ack 式数据流转图——**完整构图、圆角节点卡、贝塞尔连线、脉冲光点流动、虚线分区**。禁止网格直译 ASCII 感；禁止入场动画（动画只表现机制：数据流动/状态填充/旋转/脉冲）。

## 组件库：`site/scripts/lib/figure-kit.mjs`

```js
import { scene, node, edge, zone, cap, matrix, bars, timeline } from '../figure-kit.mjs';

const body = [];
const nodes = [];
const a = node({ id: 'a', x: 40, y: 60, w: 150, h: 64, role: 'data', title: 'token ids', sub: '[B, T]', mono: true });
nodes.push(a); body.push(a.svg);
body.push(edge(a, b, { nodes, pulse: true, dur: 2.4, delay: 0, uid: 'myfig', eid: 0 }));  // a.right → b.left 自动贝塞尔 + 脉冲光点
return scene(720, 300, body, { label: '无障碍描述' }).svg();
```

- `node({id,x,y,w,h,role,title,sub,mono,titleSize})`：rx12 圆角卡，title 主行 + sub 副行；role ∈ `data / proj / attn / norm / cache / op / plain`（各自浅染色）。节点对象有 `.right/.left/.top/.bottom` 端口点。
- `edge(from, to, opts)`：from/to 可为节点对象、节点 id、或 `[x,y]`；`portFrom/portTo` 换端口（如 `a.top`）；`pulse:true` 加朱砂渐变脉冲（SMIL animateMotion，流动方向 = from→to）；`strong` 加粗、`dash` 虚线、`arrow:false` 去箭头。`uid` 传本图唯一前缀（防 id 冲突）。
- `zone({x,y,w,h,label,tone:'ink'|'seal'})`：虚线分区。
- `cap(x,y,text,{mono,size,color,anchor,weight})`：标注文本。
- `matrix({x,y,n,m,cell,gap,on,hot,warm,label})`：格点矩阵（on 实格 / hot 朱砂 / warm 赭）。
- `bars({x,y,h,unit,segs:[{w,tone,label,sub}]})`：构成/账条（tone ∈ ink/seal/soft/warm）。
- `timeline({x,y,w,ticks:[{x,label}],spans:[{x1,x2,tone,label}]})`：时间轴。

## 调色与字体

- 纸面 `--paper-raised`，墨 `--ink`，弱 `--ink-faint`，细线 `--rule`，朱砂 `--seal`/`--seal-wash`/`--seal-deep`
- CJK 正文用 `sans`（cap 默认），数字/代码/shape 用 `mono:true`
- viewBox 宽 720 左右，高按内容 240–400；所有文字必须在 viewBox 内

## 机制动画原语（门控与降级全局处理）

kit 的 `pulse` 之外可用：`class="sig-cell" style="--i:0.35"`（按 --i 秒延迟亮起）、`sig-cell-text`（同上文本）、`sig-bar`、`sig-rotor`（+ `style="--dur:6s; transform-origin: Xpx Ypx"`）、`sig-diag`、`sig-step`、`sig-dashflow`（虚线流动）。静态元素不加类即静止。

## 交付格式

```js
export const FIGURES = [
  // 替换某个 ASCII 块（块序号 = 该章第 N 个无语言围栏，从 0 计）：
  { slug: '02-transformer-anatomy/03-attention-mechanism', replacesBlock: 0, title: 'attention 数据流', svg: () => { ... } },
  // 或新增插入（标题锚点后）：
  { slug: '...', afterHeading: /^### 2\.1 /, title: '...', svg: () => { ... } },
];
```

## 铁律

- **内容必须来自章节原文**（数字、shape、机制措辞），先读章节相关小节再构图。
- 伪代码块（输入/步骤/赋值序列）**不要**画——管线自动用纸面算法面板。
- 自检：`cd site && node scripts/fig-check.mjs <模块名>` → 逐张 Read `qa/figs-<模块名>/fig-N.png`：无文字出 viewBox、无元素重叠、脉冲/动画在 3.5s 截图时刻语义完整、整体一眼就是论文插图。
- 自查锚点：`grep -n '^##\|^###' models/<part>/<file>.md` 确认 afterHeading 精确命中一行。
