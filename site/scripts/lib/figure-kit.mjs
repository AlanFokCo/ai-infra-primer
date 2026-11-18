// figure-kit：场景式插图的组件库（data-on-ack 式语言，本站纸面+朱砂调色）
// 场景 = {w, h, body: [...]}；节点卡 / 贝塞尔连线+脉冲 / 虚线分区 / 标注 / 网格 / 条 / 时间轴。
// 约定：所有坐标作者手算；edge 的 from/to 是节点 id 或 [x,y] 点；port 自动取节点边中点。

let FIG_UID = 0;

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/* ---------- 场景 ---------- */
// 脉冲渐变：页面级唯一定义（Chapter.astro 注入隐藏 SVG defs），edge 引用固定 id。
// （同页多图共享；url(#id) 在文档内全局解析。）
export const PULSE_GRAD_ID = 'fig-pulse-grad';
export const PULSE_GRAD_DEFS =
  `<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>` +
  `<radialGradient id="${PULSE_GRAD_ID}" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs></svg>`;

export function scene(w, h, body, opts = {}) {
  const uid = `fg${(FIG_UID++).toString(36)}`;
  return {
    w, h, uid,
    svg: () =>
      `<svg class="fig-svg dia" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(opts.label ?? '机制图')}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">` +
      body.join('') +
      `</svg>`,
  };
}

/* ---------- 节点卡：rx14 圆角、标题+副标、角色浅染 ---------- */
export function node({ id, x, y, w, h = 64, role = 'plain', title, sub, mono = false, titleSize = 14.5, subSize }) {
  const cx = x + w / 2;
  const lines = [];
  if (title) lines.push(`<text class="dia-title${mono ? ' dia-mono' : ''}" x="${cx}" y="${sub ? y + h / 2 - 4 : y + h / 2 + 5}" text-anchor="middle" font-size="${titleSize}">${esc(title)}</text>`);
  if (sub) lines.push(`<text class="dia-sub${mono ? ' dia-mono' : ''}" x="${cx}" y="${y + h / 2 + 16}" text-anchor="middle"${subSize ? ` font-size="${subSize}"` : ''}>${esc(sub)}</text>`);
  return {
    id, x, y, w, h,
    right: [x + w, y + h / 2], left: [x, y + h / 2], top: [x + w / 2, y], bottom: [x + w / 2, y + h],
    svg: `<g class="dia-node dia-node-${role}" data-node="${esc(id ?? title)}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12"/>${lines.join('')}</g>`,
  };
}

/* ---------- 连线：贝塞尔 + 可选脉冲 ---------- */
// 端点解析：[x,y] 点 | 节点对象（默认 right/left 端口）| 节点 id 字符串
const pt = (ref, nodes, side) => {
  if (Array.isArray(ref)) return ref;
  if (ref && typeof ref === 'object' && ref.right) {
    if (side === 'from') return ref._portFrom ?? ref.right;
    return ref._portTo ?? ref.left;
  }
  const n = nodes.find((n) => n.id === ref || n.title === ref);
  if (!n) throw new Error(`edge 端点不存在: ${ref}`);
  if (side === 'from') return n._portFrom ?? n.right;
  return n._portTo ?? n.left;
};

export function edge(from, to, { nodes = [], portFrom, portTo, pulse = false, dur = 2.4, delay = 0, strong = false, dash = false, arrow = true, uid = 'fg', eid = 0 } = {}) {
  // 端口覆盖：调用方可传节点对象并挂 _portFrom/_portTo，或直接传 [x,y]
  if (from && typeof from === 'object' && !Array.isArray(from) && portFrom) from._portFrom = portFrom;
  if (to && typeof to === 'object' && !Array.isArray(to) && portTo) to._portTo = portTo;
  const a = pt(from, nodes, 'from');
  const b = pt(to, nodes, 'to');
  const dx = Math.min(56, Math.max(18, Math.abs(b[0] - a[0]) * 0.45));
  const dy = Math.min(56, Math.max(18, Math.abs(b[1] - a[1]) * 0.45));
  // 水平主导用横向控制点，纵向主导用纵向
  const horizontalish = Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1]);
  const d = horizontalish
    ? `M${a[0]} ${a[1]} C${a[0] + dx} ${a[1]}, ${b[0] - dx} ${b[1]}, ${b[0]} ${b[1]}`
    : `M${a[0]} ${a[1]} C${a[0]} ${a[1] + dy}, ${b[0]} ${b[1] - dy}, ${b[0]} ${b[1]}`;
  const id = `${uid}-e${eid}`;
  const cls = `dia-edge${strong ? ' strong' : ''}${dash ? ' dash' : ''}${arrow ? ' arrowed' : ''}`;
  const out = [`<path id="${id}" class="${cls}" d="${d}"/>`];
  if (pulse) {
    // calcMode spline + keyPoints/keyTimes：单程 ease-in-out 配速（端点慢、中段快），
    // 循环有呼吸感；保持单向（0→1 后回绕），不破坏有向数据流语义
    out.push(
      `<circle class="dia-pulse" r="4.2" fill="url(#${PULSE_GRAD_ID})">` +
      `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" ` +
      `calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1">` +
      `<mpath xlink:href="#${id}" href="#${id}"/></animateMotion></circle>`
    );
  }
  return out.join('');
}

/* ---------- 虚线分区 ---------- */
export function zone({ x, y, w, h, label, tone = 'ink' }) {
  return (
    `<g class="dia-zone-g"><rect class="dia-zone dia-zone-${tone}" x="${x}" y="${y}" width="${w}" height="${h}" rx="14"/>` +
    (label ? `<text class="dia-zone-label" x="${x + 16}" y="${y + 22}">${esc(label)}</text>` : '') +
    `</g>`
  );
}

/* ---------- 文本标注 ---------- */
export function cap(x, y, text, { mono = false, size = 11.5, color = 'var(--ink-faint)', anchor = 'start', weight } = {}) {
  const w = weight ? ` font-weight="${weight}"` : '';
  return `<text class="${mono ? 'dia-mono' : 'dia-cap'}" x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}"${w}>${esc(text)}</text>`;
}

/* ---------- 小图标级原件 ---------- */
// 网格（mask/attention 矩阵）：on(i,j) 决定实格；hot(i,j) 朱砂格
export function matrix({ x, y, n, m = n, cell = 22, gap = 3, on, hot, warm, label }) {
  const out = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const isOn = on(i, j);
      const cls = hot && hot(i, j) ? 'dia-mx-hot' : warm && warm(i, j) ? 'dia-mx-warm' : isOn ? 'dia-mx-on' : 'dia-mx-off';
      out.push(`<rect class="${cls}" x="${x + j * (cell + gap)}" y="${y + i * (cell + gap)}" width="${cell}" height="${cell}" rx="3"/>`);
    }
  }
  if (label) out.push(cap(x, y - 10, label, { mono: true }));
  return out.join('');
}

// 条（账/负载/构成）：segments = [{w, tone, label?, sub?}]
export function bars({ x, y, h = 22, unit = 4, segs, label }) {
  const out = [];
  let cx0 = x;
  for (const s of segs) {
    const w = Math.max(2, s.w * unit);
    out.push(`<rect class="dia-bar dia-bar-${s.tone ?? 'ink'}" x="${cx0}" y="${y}" width="${w}" height="${h}" rx="3"/>`);
    if (s.label) out.push(`<text class="dia-sub" x="${cx0 + w / 2}" y="${y - 6}" text-anchor="middle">${esc(s.label)}</text>`);
    if (s.sub) out.push(`<text class="dia-sub dia-mono" x="${cx0 + w / 2}" y="${y + h + 14}" text-anchor="middle">${esc(s.sub)}</text>`);
    cx0 += w + 3;
  }
  if (label) out.push(cap(x, y - 6, label, { mono: true }));
  return out.join('');
}

// 时间轴：ticks = [{x, label}], 主轴 + 刻度 + 可选段条 spans=[{x1,x2,tone,label}]
export function timeline({ x, y, w, ticks = [], spans = [], label }) {
  const out = [`<line class="dia-edge strong" x1="${x}" y1="${y}" x2="${x + w}" y2="${y}"/>`];
  for (const t of ticks) {
    out.push(`<line class="dia-edge" x1="${t.x}" y1="${y - 5}" x2="${t.x}" y2="${y + 5}"/>`);
    if (t.label) out.push(`<text class="dia-sub dia-mono" x="${t.x}" y="${y + 18}" text-anchor="middle">${esc(t.label)}</text>`);
  }
  for (const s of spans) {
    out.push(`<rect class="dia-bar dia-bar-${s.tone ?? 'seal'}" x="${s.x1}" y="${y - 9}" width="${s.x2 - s.x1}" height="10" rx="5"/>`);
    if (s.label) out.push(`<text class="dia-sub" x="${(s.x1 + s.x2) / 2}" y="${y - 10}" text-anchor="middle">${esc(s.label)}</text>`);
  }
  if (label) out.push(cap(x, y - 12, label, { mono: true }));
  return out.join('');
}
