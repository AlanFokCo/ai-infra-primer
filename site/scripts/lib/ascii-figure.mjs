// ASCII 图 → 论文级矢量图引擎 v2
// 语义层：文本运行 → 角色配色芯片（chip）；结构线 → 切齐到芯片边缘的连线；
// (+) → 残差加和圆圈；[shape] 标签 → 芯片旁维度小注；朱砂光点沿边流动。
// 对齐与布局沿用作者 ASCII 网格语义（CJK 双宽、组合符零宽）。

const CELL_W = 8;
const LINE_H = 21;
const PAD = 22;
const FONT_SIZE = 13;

const H_CHARS = new Set('─━═');
const V_CHARS = new Set('│┃║');
const JUNCTION = new Set('┌┍┏┐┑┓└┕┗┘┙┛├┝┠┣┤┥┫┬┰┳┴┸┻┼╂╋╠╣╔╗╚╝╦╩╬╭╮╯╰');
const ARROW = new Set('►◄▲▼');
const STRUCTURAL = new Set([...H_CHARS, ...V_CHARS, ...JUNCTION, ...ARROW]);

function charWidth(ch) {
  const cp = ch.codePointAt(0);
  if (cp >= 0x0300 && cp <= 0x036f) return 0;
  if (cp >= 0x20d0 && cp <= 0x20ff) return 0;
  if (
    (cp >= 0x1100 && cp <= 0x115f) ||
    (cp >= 0x2e80 && cp <= 0x9fff) ||
    (cp >= 0xac00 && cp <= 0xd7a3) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0xff00 && cp <= 0xff60) ||
    (cp >= 0x20000 && cp <= 0x2fa1f)
  ) return 2;
  return 1;
}

const displayCols = (s) => {
  let n = 0;
  for (const ch of s) n += charWidth(ch) || 0;
  return n;
};

const isH = (ch) => ch && (H_CHARS.has(ch) || JUNCTION.has(ch) || ch === '►' || ch === '◄');
const isV = (ch) => ch && (V_CHARS.has(ch) || JUNCTION.has(ch) || ch === '▼' || ch === '▲');
const isStruct = (ch) => ch && STRUCTURAL.has(ch);

/** 判断一块 ASCII 是否是结构图 */
export function isDiagram(lines) {
  let arrows = 0;
  let junctions = 0;
  for (const line of lines) {
    for (const ch of line) {
      if (ARROW.has(ch)) arrows += 1;
      else if (JUNCTION.has(ch)) junctions += 1;
    }
  }
  return arrows + junctions >= 2 || (arrows >= 1 && lines.some((l) => l.includes('──')));
}

/** 对齐网格表（token / p1 / p2 / ℓ 这类逐列对齐块）：
 *  ≥2 行「短标签 + 冒号 + ≥2 token」，且冒号后 token 起始列跨行对齐（±3 列）。
 *  伪代码的「输入：…」式散文行不满足对齐条件，回落为算法面板。 */
export function isTableBlock(lines) {
  if (isDiagram(lines)) return false;
  const labeled = [];
  for (const l of lines) {
    const m = l.match(/^(.{1,12}?)\s*[:：]\s+(\S.*)$/);
    if (!m) continue;
    const label = m[1].trim();
    if (displayCols(label) > 10) continue;
    const tokens = m[2].trim().split(/\s{2,}|\s+/).filter(Boolean);
    if (tokens.length < 2) continue;
    labeled.push({ labelEnd: displayCols(m[1]), tokens });
  }
  if (labeled.length < 2) return false;
  // 至少一行 ≥4 token（掩码表特征），伪代码的孤立「输入：…」行不满足
  return labeled.some((r) => r.tokens.length >= 4);
}

/**
 * 对齐表格块 → 单元格芯片网格图。
 * 列位沿用作者 ASCII 列坐标；`|` → 虚线分隔；[M]/[s]/特殊 token 朱砂染色。
 */
export function renderGridFigure(lines) {
  const rows = lines.map((line) => {
    const cells = [];
    for (const ch of line) {
      const w = charWidth(ch);
      if (w === 0 && cells.length > 0) cells[cells.length - 1].ch += ch;
      else cells.push({ ch, w });
    }
    return cells;
  });
  let cols = 0;
  const maps = rows.map((cells) => {
    const m = new Map();
    let c = 0;
    for (const cell of cells) {
      m.set(c, cell.ch);
      c += cell.w;
    }
    cols = Math.max(cols, c);
    return m;
  });
  const nRows = rows.length;
  const cx = (c) => PAD + c * CELL_W;
  const cy = (r) => PAD + r * LINE_H;

  const parts = [];
  const cellsOut = [];
  const rowLabels = [];
  const dividers = [];

  for (let r = 0; r < nRows; r++) {
    const m = maps[r];
    let c = 0;
    while (c < cols) {
      const ch = m.get(c);
      if (ch === '|') {
        dividers.push({ x: cx(c) + 4 });
        c += 1;
        continue;
      }
      if (ch && ch !== ' ') {
        const start = c;
        let text = '';
        let lastEnd = c;
        while (c < cols) {
          const cc = m.get(c);
          if (!cc || cc === '|') break;
          if (cc === ' ') {
            // 空格：看下一个非空 token 的距离；1 格空格视为 token 内部分隔
            let gap = 0, ahead = c;
            while (ahead < cols && maps[r].get(ahead) === ' ') { gap++; ahead++; }
            const nextCh = maps[r].get(ahead);
            if (!nextCh || nextCh === '|' || gap >= 2) break;
            text += ' ';
            c += 1;
            continue;
          }
          text += cc;
          c += charWidth(cc) || 1;
          lastEnd = c;
        }
        const t = text.trim();
        if (t === ':' || t === '') { c = Math.max(c, lastEnd); continue; }
        if (t.endsWith(':') && start < 12) {
          rowLabels.push({ x: cx(start), y: cy(r) + 4, text: t.replace(/:$/, '') });
        } else if (/^[（(]/.test(t)) {
          // 括号注记保持纯文本
          cellsOut.push({ col: start, row: r, text: t, cols: Math.max(1, lastEnd - start), plain: true });
        } else {
          cellsOut.push({ col: start, row: r, text: t, cols: Math.max(1, lastEnd - start) });
        }
        c = Math.max(c, lastEnd);
      } else c += 1;
    }
  }

  const W = PAD * 2 + cols * CELL_W + 6;
  const H = PAD * 2 + (nRows - 1) * LINE_H + 12;
  parts.push(
    `<svg class="fig-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="对齐网格" xmlns="http://www.w3.org/2000/svg" font-size="12">`
  );
  for (const d of dividers) {
    parts.push(
      `<line x1="${d.x}" y1="${PAD - 6}" x2="${d.x}" y2="${H - PAD + 2}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`
    );
  }
  for (const rl of rowLabels) {
    parts.push(`<text class="t" x="${rl.x}" y="${rl.y + 2}" fill="var(--ink-faint)">${esc(rl.text)}</text>`);
  }
  for (const cell of cellsOut) {
    if (cell.plain) {
      parts.push(`<text class="t" x="${cx(cell.col)}" y="${cy(cell.row) + 4}">${esc(cell.text)}</text>`);
      continue;
    }
    const w = Math.max(24, cell.cols * CELL_W + 10);
    const special = /^\[(M|s|M\*|mask|eos|bos)\]$/i.test(cell.text);
    parts.push(
      `<g class="cell${special ? ' cell-hot' : ''}">` +
        `<rect x="${cx(cell.col) - 4}" y="${cy(cell.row) - 8}" width="${w}" height="20" rx="4"/>` +
        `<text x="${cx(cell.col) - 4 + w / 2}" y="${cy(cell.row) + 5}" text-anchor="middle" textLength="${w - 10}" lengthAdjust="spacingAndGlyphs">${esc(cell.text)}</text>` +
        `</g>`
    );
  }
  parts.push('</svg>');
  return parts.join('');
}

/** 伪代码块 → 纸面算法面板（行号 gutter；# 注释弱化分色） */
export function renderAlgo(lines) {
  const items = lines
    .map((l, i) => {
      const cmIdx = l.indexOf(' # ');
      let body;
      if (cmIdx > 0) {
        body = `${esc(l.slice(0, cmIdx))}<span class="algo-cm">${esc(l.slice(cmIdx))}</span>`;
      } else {
        body = esc(l) || ' ';
      }
      return `<li><span class="algo-ln">${String(i + 1).padStart(2, '0')}</span><span class="algo-text">${body}</span></li>`;
    })
    .join('');
  return `<div class="algo"><ol>${items}</ol></div>`;
}

/** 块分类：diagram 结构图 / table 对齐网格 / code 伪代码 */
export function classifyBlock(lines) {
  if (isDiagram(lines)) return 'diagram';
  if (isTableBlock(lines)) return 'table';
  return 'code';
}

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/* 角色判定：芯片配色语义 */
const ROLES = [
  ['attn', /attention|attn|softmax|score|qkv|mha|gqa|mla|nsa|dsa|moba/i],
  ['norm', /rmsnorm|layernorm|norm|ln[₁₂f]?\b/i],
  ['cache', /cache|缓存|state|状态/i],
  ['proj', /投影|matmul|proj|linear|w\^|gate\/up|down|up\s|fc\b|mlp|ffn/i],
  ['data', /emb|token|输入|输出|ids|stream|x̃|x\^|logits|数据|波形|文本|图像|mel/i],
  ['op', /concat|split|sample|reshape|rope|silu|gelu|tanh|gather|scatter/i],
];
function roleOf(text) {
  for (const [role, re] of ROLES) if (re.test(text)) return role;
  return 'plain';
}

const ADD_RE = /^\(\s*[+＋]\s*\)$/; // (+) 残差加和

/**
 * 渲染一个 ASCII 图为 SVG 字符串（v2 组件化）。
 * @param {string[]} lines 围栏内文本行
 */
export function renderFigure(lines) {
  // ---------- 网格化 ----------
  const rows = lines.map((line) => {
    const cells = [];
    for (const ch of line) {
      const w = charWidth(ch);
      if (w === 0 && cells.length > 0) cells[cells.length - 1].ch += ch;
      else cells.push({ ch, w });
    }
    return cells;
  });
  let cols = 0;
  const maps = rows.map((cells) => {
    const m = new Map();
    let c = 0;
    for (const cell of cells) {
      m.set(c, cell.ch);
      c += cell.w;
    }
    cols = Math.max(cols, c);
    return m;
  });
  const nRows = rows.length;
  const cx = (c) => PAD + c * CELL_W;
  const cy = (r) => PAD + r * LINE_H;

  // ---------- 文本运行（≥2 空格断列） ----------
  const runs = [];
  for (let r = 0; r < nRows; r++) {
    const m = maps[r];
    let c = 0;
    while (c < cols) {
      const ch = m.get(c);
      if (ch && !isStruct(ch) && ch !== ' ') {
        const start = c;
        let text = '';
        let lastEnd = c;
        while (c < cols) {
          const cc = m.get(c);
          if (!cc || isStruct(cc)) break;
          if (cc === ' ') {
            let gap = 0, ahead = c;
            while (ahead < cols && maps[r].get(ahead) === ' ') { gap++; ahead++; }
            const nextCh = maps[r].get(ahead);
            if (gap >= 2 || !nextCh || isStruct(nextCh)) break;
            text += ' ';
            c += 1;
            continue;
          }
          text += cc;
          c += charWidth(cc) || 1;
          lastEnd = c;
        }
        runs.push({ col: start, row: r, text, cols: Math.max(1, lastEnd - start) });
        c = Math.max(c, lastEnd);
      } else c += 1;
    }
  }

  // ---------- 拆分 shape 标签 + 判定芯片 ----------
  // 与线/箭相邻的短运行 → 芯片；[...] 标签若是芯片的尾巴 → 维度小注；其余保持文本
  const chips = []; // {x,y,w,text,role,row,col,cols,dims:[]}
  const plainTexts = []; // {x,y,text,cols,row}
  const dimNotes = []; // {x,y,text}（芯片旁或独立的小字）

  const touched = (r, c0, c1) => {
    for (let rr = Math.max(0, r - 1); rr <= Math.min(nRows - 1, r + 1); rr++) {
      for (let cc = Math.max(0, c0 - 1); cc <= c1 + 1; cc++) {
        const ch = maps[rr].get(cc);
        if (ch && isStruct(ch)) return true;
      }
    }
    return false;
  };

  for (const run of runs) {
    // 拆 [...] shape 段
    const pieces = [];
    let cursor = run.col;
    let last = 0;
    const re = /(\[[\w ,;×/:.-]+\])/g;
    let m;
    while ((m = re.exec(run.text))) {
      if (m.index > last) {
        const t = run.text.slice(last, m.index);
        pieces.push({ text: t, col: cursor, kind: 'text' });
        cursor += displayCols(t);
      }
      pieces.push({ text: m[1], col: cursor, kind: 'shape' });
      cursor += displayCols(m[1]);
      last = m.index + m[1].length;
    }
    if (last < run.text.length) pieces.push({ text: run.text.slice(last), col: cursor, kind: 'text' });

    const isChipCandidate =
      run.cols <= 30 && touched(run.row, run.col, run.col + run.cols);
    const addCircle = ADD_RE.test(run.text.trim());

    if (addCircle) {
      chips.push({ x: cx(run.col), y: cy(run.row) + 4, w: 22, text: '+', role: 'add', row: run.row, col: run.col, cols: run.cols, dims: [], circle: true });
      continue;
    }
    if (!isChipCandidate) {
      // 全部 plain 文本（shape 段也变 dim 小注）
      for (const p of pieces) {
        if (p.kind === 'shape') dimNotes.push({ x: cx(p.col), y: cy(run.row) + 4, text: p.text });
        else plainTexts.push({ x: cx(p.col), y: cy(run.row) + 4, text: p.text, cols: displayCols(p.text), row: run.row });
      }
      continue;
    }
    // 括号注记（以 （ 或 ( 开头）永远保持纯文本
    if (/^[（(]/.test(run.text.trim())) {
      for (const p of pieces) {
        if (p.kind === 'shape') dimNotes.push({ x: cx(p.col), y: cy(run.row) + 4, text: p.text });
        else plainTexts.push({ x: cx(p.col), y: cy(run.row) + 4, text: p.text, cols: displayCols(p.text), row: run.row });
      }
      continue;
    }

    // 芯片：text 段进芯片；末尾的 shape 段收进芯片作第二行维度注，中间的 shape 段作右侧小注
    const texts = pieces.filter((p) => p.kind === 'text');
    const shapes = pieces.filter((p) => p.kind === 'shape');
    const label = texts.map((p) => p.text).join('').trim();
    const labelCols = texts.reduce((n, p) => n + displayCols(p.text), 0);
    if (!label) {
      for (const p of pieces) dimNotes.push({ x: cx(p.col), y: cy(run.row) + 4, text: p.text });
      continue;
    }
    const lastPiece = pieces[pieces.length - 1];
    const tailShape = lastPiece && lastPiece.kind === 'shape' ? lastPiece : null;
    const midShapes = shapes.filter((s) => s !== tailShape);
    const shapeCols = tailShape ? displayCols(tailShape.text) : 0;
    const chipW = Math.max(30, Math.max(labelCols, shapeCols) * CELL_W + 16);
    const chip = {
      x: cx(run.col) - 6,
      y: cy(run.row) + 4 - (tailShape ? 17 : 12),
      w: chipW,
      h: tailShape ? 36 : 24,
      text: label,
      dim: tailShape ? tailShape.text : null,
      role: roleOf(label),
      row: run.row,
      col: run.col,
      cols: run.cols,
    };
    for (const s of midShapes) {
      dimNotes.push({ x: chip.x + chip.w + 6, y: cy(run.row) + 4, text: s.text });
    }
    chips.push(chip);
  }

  // ---------- 线段（H/V 极大运行），端点切齐芯片边缘 ----------
  const segs = [];
  const arrows = [];

  const chipAt = (r, c) =>
    chips.find(
      (ch) => r >= ch.row - 0.5 && r <= ch.row + 0.5 && cx(c) >= ch.x - 2 && cx(c) <= ch.x + ch.w + 2
    );

  for (let r = 0; r < nRows; r++) {
    const m = maps[r];
    let c = 0;
    while (c < cols) {
      const ch = m.get(c);
      if (ch && (H_CHARS.has(ch) || JUNCTION.has(ch) || ARROW.has(ch))) {
        const start = c;
        let cls = 'edge';
        while (c < cols) {
          const cc = m.get(c);
          if (!cc || !(H_CHARS.has(cc) || JUNCTION.has(cc) || ARROW.has(cc))) break;
          if (cc === '═') cls = 'edge-strong';
          c += 1;
        }
        let x1 = cx(start), x2 = cx(c);
        const y = cy(r) + 4;
        const cLeft = chipAt(r, start);
        const cRight = chipAt(r, c - 1);
        if (cLeft && x1 > cLeft.x && x1 < cLeft.x + cLeft.w) x1 = cLeft.x + cLeft.w;
        if (cRight && x2 > cRight.x && x2 < cRight.x + cRight.w) x2 = cRight.x;
        if (x2 - x1 > 2) segs.push({ x1, y1: y, x2, y2: y, cls });
      } else c += 1;
    }
  }
  for (let c = 0; c < cols; c++) {
    let r = 0;
    while (r < nRows) {
      const ch = maps[r].get(c);
      if (ch && (V_CHARS.has(ch) || JUNCTION.has(ch) || ARROW.has(ch))) {
        const start = r;
        let cls = 'edge';
        while (r < nRows) {
          const cc = maps[r].get(c);
          if (!cc || !(V_CHARS.has(cc) || JUNCTION.has(cc) || ARROW.has(cc))) break;
          if (cc === '║') cls = 'edge-strong';
          r += 1;
        }
        let y1 = cy(start), y2 = cy(r);
        const x = cx(c) + 4;
        // 孤立单格竖线（└┐┌┘├┤ 角点的残影）无上下结构邻居则丢弃
        if (r - start === 1) {
          const above = start > 0 ? maps[start - 1].get(c) : null;
          const below = r < nRows ? maps[r].get(c) : null;
          const nb = (z) => z && (V_CHARS.has(z) || JUNCTION.has(z));
          if (!nb(above) && !nb(below)) continue;
        }
        // 竖线端点切芯片上下沿
        for (const chp of chips) {
          const top = chp.y, bot = chp.y + chp.h;
          if (x >= chp.x - 2 && x <= chp.x + chp.w + 2) {
            if (y1 > top && y1 < bot && start !== chp.row) y1 = bot;
            if (y2 > top && y2 < bot) y2 = top;
          }
        }
        if (y2 - y1 > 2) segs.push({ x1: x, y1, x2: x, y2, cls });
      } else r += 1;
    }
  }

  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = maps[r].get(c);
      if (ch === '►') arrows.push({ x: cx(c + 1) + 1, y: cy(r) + 4, dir: 'r' });
      else if (ch === '◄') arrows.push({ x: cx(c) - 1, y: cy(r) + 4, dir: 'l' });
      else if (ch === '▼') arrows.push({ x: cx(c) + 4, y: cy(r + 1) - 2, dir: 'd' });
      else if (ch === '▲') arrows.push({ x: cx(c) + 4, y: cy(r) + 2, dir: 'u' });
    }
  }

  // ---------- 芯片碰撞消解：相交的后者降级为纯文本 + 维度注 ----------
  const collide = (a, b) =>
    a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  for (let i = 1; i < chips.length; i++) {
    for (let j = 0; j < i; j++) {
      if (!chips[j].demoted && collide(chips[i], chips[j])) {
        const c = chips[i];
        c.demoted = true;
        plainTexts.push({ x: cx(c.col), y: cy(c.row) + 4, text: c.text, cols: displayCols(c.text), row: c.row });
        if (c.dim) dimNotes.push({ x: cx(c.col) + displayCols(c.text) * CELL_W + 8, y: cy(c.row) + 4, text: c.dim });
        break;
      }
    }
  }
  const liveChips = chips.filter((c) => !c.demoted);

  // ---------- 输出 ----------
  const W = PAD * 2 + cols * CELL_W + 10;
  const H = PAD * 2 + (nRows - 1) * LINE_H + 10;
  const parts = [];
  parts.push(
    `<svg class="fig-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="结构图" xmlns="http://www.w3.org/2000/svg" font-size="${FONT_SIZE}">`
  );

  // 边
  for (const s of segs) {
    parts.push(`<line class="${s.cls}" x1="${s.x1}" y1="${s.y1}" x2="${s.x2}" y2="${s.y2}" />`);
  }
  // 数据流光点
  for (const s of segs) {
    const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    if (len < 40) continue;
    let { x1, y1, x2, y2 } = s;
    const rev = arrows.some(
      (a) =>
        (a.dir === 'l' && Math.abs(a.y - s.y1) < 2 && Math.abs(a.x - s.x1) < CELL_W * 2) ||
        (a.dir === 'u' && Math.abs(a.x - s.x1) < 2 && Math.abs(a.y - s.y1) < LINE_H)
    );
    if (rev) [x1, y1, x2, y2] = [x2, y2, x1, y1];
    const dur = Math.max(1.2, len / 55);
    const dots = len > 140 ? 2 : 1;
    for (let i = 0; i < dots; i++) {
      const begin = i === 0 ? '0s' : `-${(dur / 2).toFixed(2)}s`;
      parts.push(
        `<circle class="flow" r="2.6"><animateMotion dur="${dur.toFixed(2)}s" begin="${begin}" repeatCount="indefinite" path="M ${x1} ${y1} L ${x2} ${y2}" /></circle>`
      );
    }
  }
  // 箭头
  for (const a of arrows) {
    const d =
      a.dir === 'r' ? `M${a.x},${a.y} l-8,-4.5 v9 z` :
      a.dir === 'l' ? `M${a.x},${a.y} l8,-4.5 v9 z` :
      a.dir === 'd' ? `M${a.x},${a.y} l-4.5,-8 h9 z` :
      `M${a.x},${a.y} l-4.5,8 h9 z`;
    parts.push(`<path class="arrow" d="${d}" />`);
  }
  // 芯片
  for (const ch of liveChips) {
    if (ch.circle) {
      parts.push(
        `<g class="chip role-add"><circle cx="${ch.x + 11}" cy="${ch.y}" r="10" /><text x="${ch.x + 11}" y="${ch.y + 4.5}" text-anchor="middle">+</text></g>`
      );
      continue;
    }
    const fs = displayCols(ch.text) > 20 ? 11.5 : FONT_SIZE;
    // 长标签用 textLength 锁进芯片内宽；短标签不设（避免拉伸）
    const fit = displayCols(ch.text) > 12 ? ` textLength="${ch.w - 14}" lengthAdjust="spacingAndGlyphs"` : '';
    if (ch.dim) {
      parts.push(
        `<g class="chip role-${ch.role}">` +
          `<rect x="${ch.x}" y="${ch.y}" width="${ch.w}" height="${ch.h}" rx="6" />` +
          `<text x="${ch.x + ch.w / 2}" y="${ch.y + 15}" text-anchor="middle" font-size="${fs}"${fit}>${esc(ch.text)}</text>` +
          `<text class="dim" x="${ch.x + ch.w / 2}" y="${ch.y + 29}" text-anchor="middle" font-size="10"${fit}>${esc(ch.dim)}</text>` +
          `</g>`
      );
      continue;
    }
    parts.push(
      `<g class="chip role-${ch.role}">` +
        `<rect x="${ch.x}" y="${ch.y}" width="${ch.w}" height="${ch.h}" rx="6" />` +
        `<text x="${ch.x + ch.w / 2}" y="${ch.y + ch.h / 2 + (fs * 0.36)}" text-anchor="middle" font-size="${fs}"${fit}>${esc(ch.text)}</text>` +
        `</g>`
    );
  }
  // 维度小注
  for (const d of dimNotes) {
    parts.push(`<text class="dim" x="${d.x}" y="${d.y + 2}" font-size="10.5">${esc(d.text)}</text>`);
  }
  // 纯文本注记
  for (const t of plainTexts) {
    parts.push(
      `<text class="t" x="${t.x}" y="${t.y + 2}" textLength="${t.cols * CELL_W}" lengthAdjust="spacing">${esc(t.text)}</text>`
    );
  }
  parts.push('</svg>');
  return parts.join('');
}
