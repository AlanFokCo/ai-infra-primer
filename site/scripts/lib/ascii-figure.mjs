// ASCII 图 → 语义 SVG 图引擎
// 网格解析（CJK 双宽、组合符零宽）→ 结构线/箭头/文本运行 → 矢量 SVG。
// 对齐保证：每个文本运行带 textLength + lengthAdjust="spacing"，与字体度量无关。

const CELL_W = 8;
const LINE_H = 21;
const PAD = 18;
const FONT_SIZE = 13;

// 结构字符集
const H_CHARS = new Set('─━═─');
const V_CHARS = new Set('│┃║');
const JUNCTION = new Set('┌┍┏┐┑┓└┕┗┘┙┛├┝┠┣┤┥┫┬┰┳┴┸┻┼╂╋╠╣╔╗╚╝╦╩╬╭╮╯╰');
const ARROW = new Set('►◄▲▼');
const STRUCTURAL = new Set([...H_CHARS, ...V_CHARS, ...JUNCTION, ...ARROW]);

// 字符显示宽度：CJK/全角 = 2，组合符（如 x̃ 的 ◌̃）= 0，其余 = 1
function charWidth(ch) {
  const cp = ch.codePointAt(0);
  if (cp >= 0x0300 && cp <= 0x036f) return 0; // combining diacriticals
  if (cp >= 0x20d0 && cp <= 0x20ff) return 0; // combining marks for symbols
  if (
    (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo
    (cp >= 0x2e80 && cp <= 0x9fff) || // CJK 各部
    (cp >= 0xac00 && cp <= 0xd7a3) || // Hangul syllables
    (cp >= 0xf900 && cp <= 0xfaff) || // CJK compat
    (cp >= 0xff00 && cp <= 0xff60) || // fullwidth forms
    (cp >= 0x20000 && cp <= 0x2fa1f)
  ) return 2;
  return 1;
}

/** 把源码行展开为 cell 数组：每 cell { ch, w }，返回 {rows, cols} */
function toGrid(lines) {
  const rows = lines.map((line) => {
    const cells = [];
    for (const ch of line) {
      const w = charWidth(ch);
      if (w === 0 && cells.length > 0) {
        cells[cells.length - 1].ch += ch; // 组合符并入前一字符
      } else {
        cells.push({ ch, w });
      }
    }
    return cells;
  });
  return rows;
}

/** grid → 每行等宽 cell 索引图：cells[r][c] = 字符或 null */
function toCellMap(rows) {
  let maxCols = 0;
  const maps = rows.map((cells) => {
    const m = new Map();
    let c = 0;
    for (const cell of cells) {
      m.set(c, cell.ch);
      c += cell.w;
    }
    maxCols = Math.max(maxCols, c);
    return m;
  });
  return { maps, cols: maxCols };
}

const isH = (ch) => ch && (H_CHARS.has(ch) || JUNCTION.has(ch) || ch === '►' || ch === '◄');
const isV = (ch) => ch && (V_CHARS.has(ch) || JUNCTION.has(ch) || ch === '▼' || ch === '▲');
const isStruct = (ch) => ch && STRUCTURAL.has(ch);

/** 判断一块 ASCII 是否是结构图（否则保持深色代码板） */
export function isDiagram(lines) {
  let arrows = 0;
  let junctions = 0;
  for (const line of lines) {
    for (const ch of line) {
      if (ARROW.has(ch)) arrows += 1;
      else if (JUNCTION.has(ch)) junctions += 1;
    }
  }
  return arrows + junctions >= 2 || (arrows >= 1 && junctions >= 0 && lines.some((l) => l.includes('──')));
}

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const SHAPE_RE = /^\[[\w ,;×/:.-]+\]$/; // [B, T, 4096] 等 shape 标签

/**
 * 渲染一个 ASCII 图为 SVG 字符串。
 * @param {string[]} lines 围栏内文本行（已去尾空行）
 * @param {{ figId?: string }} opts
 * @returns {string} svg html
 */
export function renderFigure(lines, opts = {}) {
  const rows = toGrid(lines);
  const { maps, cols } = toCellMap(rows);
  const nRows = rows.length;

  const segs = []; // {x1,y1,x2,y2,cls}
  const texts = []; // {x,y,text,cells,cls}
  const arrows = []; // {x,y,dir}

  const cx = (c) => PAD + c * CELL_W;
  const cy = (r) => PAD + r * LINE_H;

  // ---- 水平线段：逐行扫连续 isH 运行 ----
  for (let r = 0; r < nRows; r++) {
    const m = maps[r];
    let c = 0;
    while (c < cols) {
      const ch = m.get(c);
      if (ch && (H_CHARS.has(ch) || JUNCTION.has(ch) || ARROW.has(ch))) {
        let start = c;
        let cls = 'edge';
        while (c < cols) {
          const cc = m.get(c);
          if (!cc || !(H_CHARS.has(cc) || JUNCTION.has(cc) || ARROW.has(cc))) break;
          if (cc === '═' || cc === '║') cls = 'edge-strong';
          c += 1;
        }
        segs.push({ x1: cx(start), y1: cy(r) + 4, x2: cx(c), y2: cy(r) + 4, cls, row: r, c1: start, c2: c });
      } else {
        c += 1;
      }
    }
  }

  // ---- 垂直线段：逐列扫连续 isV 运行 ----
  for (let c = 0; c < cols; c++) {
    let r = 0;
    while (r < nRows) {
      const ch = maps[r].get(c);
      if (ch && (V_CHARS.has(ch) || JUNCTION.has(ch) || ARROW.has(ch))) {
        let start = r;
        let cls = 'edge';
        while (r < nRows) {
          const cc = maps[r].get(c);
          if (!cc || !(V_CHARS.has(cc) || JUNCTION.has(cc) || ARROW.has(cc))) break;
          if (cc === '║') cls = 'edge-strong';
          r += 1;
        }
        segs.push({ x1: cx(c) + 4, y1: cy(start), x2: cx(c) + 4, y2: cy(r), cls, col: c, r1: start, r2: r });
      } else {
        r += 1;
      }
    }
  }

  // ---- 箭头头 ----
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = maps[r].get(c);
      if (ch === '►') arrows.push({ x: cx(c + 1) + 1, y: cy(r) + 4, dir: 'r' });
      else if (ch === '◄') arrows.push({ x: cx(c) - 1, y: cy(r) + 4, dir: 'l' });
      else if (ch === '▼') arrows.push({ x: cx(c) + 4, y: cy(r + 1), dir: 'd' });
      else if (ch === '▲') arrows.push({ x: cx(c) + 4, y: cy(r), dir: 'u' });
    }
  }

  // ---- 文本运行：非结构 cell，按 ≥2 空格分列 ----
  for (let r = 0; r < nRows; r++) {
    const m = maps[r];
    let c = 0;
    while (c < cols) {
      const ch = m.get(c);
      if (ch && !isStruct(ch) && ch !== ' ') {
        let start = c;
        let text = '';
        let lastEnd = c;
        while (c < cols) {
          const cc = m.get(c);
          if (!cc || isStruct(cc)) break;
          if (cc === ' ') {
            // 预览下一个非空 cell 距几个空位；≥2 空格断列
            let gap = 0;
            let ahead = c;
            while (ahead < cols && maps[r].get(ahead) === ' ') { gap += 1; ahead += 1; }
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
        const cellSpan = Math.max(1, lastEnd - start);
        texts.push({ x: cx(start), y: cy(r) + 4, text, cells: cellSpan, row: r, col: start });
        c = Math.max(c, lastEnd);
      } else {
        c += 1;
      }
    }
  }

  // shape 标签 pill 化 + CJK 原子化：
  // - 运行内部的 [...] 拆为 pill 子段（按列宽原位对齐）
  // - CJK/全角字逐字独立渲染（不占 textLength，避免 spacing 拉伸破坏字形间距）；
  //   ASCII 运行保留 textLength=spacing 以对齐网格
  const pillified = [];
  const SHAPE_PIECE = /(\[[\w ,;×/:.-]+\])/g;

  const displayCols = (s) => {
    let n = 0;
    for (const ch of s) n += charWidth(ch) || 0;
    return n;
  };

  // 把一段文本拆成渲染单元：CJK 独字、其余连续成串
  function atomize(text, startCol, baseY, cls, out) {
    let buf = '';
    let bufCol = startCol;
    let col = startCol;
    const flush = () => {
      if (buf) out.push({ text: buf, col: bufCol, cols: displayCols(buf), cls, y: baseY });
      buf = '';
    };
    for (const ch of text) {
      const w = charWidth(ch);
      if (w === 2) {
        flush();
        out.push({ text: ch, col, cols: 2, cls, cjk: true, y: baseY });
        col += 2;
      } else if (w === 0) {
        if (buf) buf += ch;
        else if (out.length > 0) out[out.length - 1].text += ch; // 行首组合符并入前一单元
      } else {
        if (!buf) bufCol = col;
        buf += ch;
        col += 1;
      }
    }
    flush();
  }

  for (const t of texts) {
    let cursorCol = t.col;
    const rest = t.text;
    SHAPE_PIECE.lastIndex = 0;
    let m;
    let lastIdx = 0;
    let any = false;
    while ((m = SHAPE_PIECE.exec(rest))) {
      any = true;
      if (m.index > lastIdx) {
        const before = rest.slice(lastIdx, m.index);
        atomize(before, cursorCol, t.y, null, pillified);
        cursorCol += displayCols(before);
      }
      atomize(m[1], cursorCol, t.y, 'shape', pillified);
      cursorCol += displayCols(m[1]);
      lastIdx = m.index + m[1].length;
    }
    if (!any) {
      atomize(rest, cursorCol, t.y, null, pillified);
    } else if (lastIdx < rest.length) {
      atomize(rest.slice(lastIdx), cursorCol, t.y, null, pillified);
    }
  }
  for (const p of pillified) {
    p.x = cx(p.col);
  }

  const W = PAD * 2 + cols * CELL_W;
  const H = PAD * 2 + (nRows - 1) * LINE_H + 8;

  const parts = [];
  parts.push(
    `<svg class="fig-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="结构图"` +
      ` xmlns="http://www.w3.org/2000/svg" font-size="${FONT_SIZE}">`
  );

  for (const s of segs) {
    parts.push(
      `<line class="${s.cls}" x1="${s.x1}" y1="${s.y1}" x2="${s.x2}" y2="${s.y2}" />`
    );
  }

  // ---- 数据流：token 光点沿边行进（SMIL animateMotion；JS 负责 in-view 启停与降级） ----
  for (const s of segs) {
    const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    if (len < 40) continue; // 短连接不出光点
    // 方向：看运行两端有无箭头；默认 L→R / 上→下
    let x1 = s.x1, y1 = s.y1, x2 = s.x2, y2 = s.y2;
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
        `<circle class="flow" r="2.4">` +
          `<animateMotion dur="${dur.toFixed(2)}s" begin="${begin}" repeatCount="indefinite" ` +
          `path="M ${x1} ${y1} L ${x2} ${y2}" />` +
          `</circle>`
      );
    }
  }

  for (const a of arrows) {
    const d =
      a.dir === 'r' ? `M${a.x},${a.y} l-7,-4 v8 z` :
      a.dir === 'l' ? `M${a.x},${a.y} l7,-4 v8 z` :
      a.dir === 'd' ? `M${a.x},${a.y} l-4,-7 h8 z` :
      `M${a.x},${a.y} l-4,7 h8 z`;
    parts.push(`<path class="arrow" d="${d}" />`);
  }
  for (const t of pillified) {
    const len = t.cols * CELL_W;
    const lenAttr = t.cjk ? '' : ` textLength="${len}" lengthAdjust="spacing"`;
    if (t.cls === 'shape') {
      parts.push(
        `<g class="pill"><rect x="${t.x - 3}" y="${t.y - 11}" width="${len + 6}" height="17" rx="3" />` +
          `<text x="${t.x}" y="${t.y + 2}"${lenAttr}>${esc(t.text)}</text></g>`
      );
    } else {
      const cls = t.cjk ? 't cjk' : 't';
      parts.push(
        `<text class="${cls}" x="${t.x}" y="${t.y + 2}"${lenAttr}>${esc(t.text)}</text>`
      );
    }
  }
  parts.push('</svg>');
  return parts.join('');
}
