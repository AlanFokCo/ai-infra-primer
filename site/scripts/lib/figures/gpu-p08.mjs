// gpu P8（08-nccl-collectives）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs / gpu-p06.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）。
// 消息量硬规则：ring all-reduce 总量 2(n-1)/n·M vs 单阶段 (n-1)/n·M（勿混 2×）；总流量下界 2(n-1)M。
// census：8.1 block0 ring vs a2a 流量图；8.2 block0 两阶段环 / block1 归约树+广播树 / block2 逻辑→物理 / block3 流水时序；
//         8.3 block0 12-rank 双二叉树；8.4 block0 init 流程 / block1 GDR vs 非 GDR / block2 三协议 line / block3 dispatch 树
//         （8.4 两个 [c] 围栏不计；8.1 §7 API 表为 markdown 表不画）。
//         签名：8.1 §3 八原语消息量、8.3 §4 ring/tree 交叉 M*、8.4 §4 transport 四件套。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

// 矩形边界锚点：从 (cx,cy) 朝 (tx,ty) 方向 exits 矩形 w×h 的边界点
const anchor = (cx, cy, w, h, tx, ty) => {
  const dx = tx - cx, dy = ty - cy;
  const t = Math.min((w / 2) / Math.abs(dx || 1e-9), (h / 2) / Math.abs(dy || 1e-9));
  return [cx + dx * t, cy + dy * t];
};

/* ============================================================
 * 8.1 §3（afterHeading）— 八原语消息量账（均衡四 vs root 四）
 * ============================================================ */
function primitiveVolume() {
  const UID = 'g8a';
  const body = [];

  body.push(cap(36, 24, '八原语消息量账：M = 全量数据字节、分片 M/n —— 均衡四与 n 解耦，root 四有热点（n=8, M=1 GB）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 648, h: 150, label: '均衡四原语：每 rank 发送与 n 解耦 · 无 root 热点 · 收发对称' }));
  const bal = [
    ['all-reduce', '2(n-1)/n·M', 1.75, '1.75 GB', 'seal'],
    ['all-gather', '(n-1)/n·M', 0.875, '0.875 GB', 'soft'],
    ['reduce-scatter', '(n-1)/n·M', 0.875, '0.875 GB', 'soft'],
    ['all-to-all', '(n-1)/n·M', 0.875, '0.875 GB · 每 peer M/n = 0.125', 'soft'],
  ];
  bal.forEach(([name, formula, v, lab, tone], i) => {
    const y = 96 + i * 26;
    body.push(cap(52, y, name, { mono: true, size: 10.5 }));
    body.push(cap(170, y, formula, { mono: true, size: 9 }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="350" y="${y - 12}" width="${v * 160}" height="16" rx="3"/>`);
    body.push(cap(350 + v * 160 + 8, y, lab, { mono: true, size: 9.5 }));
  });

  body.push(zone({ x: 36, y: 214, w: 648, h: 150, label: 'root 四原语：root 链路是潜在热点 —— 负载有界 vs 依赖实现' }));
  const root = [
    ['broadcast', 'root ≥ M · chain 每转发者 M', 1.0, 'root 1 GB · tree 分摊 → 8.3', 'warm'],
    ['reduce', '非 root M · star root 收 (n-1)M', 1.0, '非 root 1 GB · root 收 7 GB', 'warm'],
    ['gather', '非 root M/n · root 收 (n-1)/n·M', 0.125, '0.125 GB · root 收 0.875（有界）', 'soft'],
    ['scatter', 'root (n-1)/n·M · 各收 M/n', 0.875, 'root 发 0.875 GB（有界）', 'soft'],
  ];
  root.forEach(([name, formula, v, lab, tone], i) => {
    const y = 262 + i * 26;
    body.push(cap(52, y, name, { mono: true, size: 10.5 }));
    body.push(cap(170, y, formula, { size: 9 }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="350" y="${y - 12}" width="${v * 160}" height="16" rx="3"/>`);
    body.push(cap(350 + v * 160 + 8, y, lab, { mono: true, size: 9.5 }));
  });

  body.push(cap(36, 386, '命题 1（a2a）：M 切 n 份每份 M/n、第 j 份发 rank j —— 每 rank (n-1)·M/n、每 peer M/n，自环份留本地不发', { size: 10 }));
  body.push(cap(36, 404, '坑：每 rank / 每 peer 口径差 (n-1) 倍；含自环多计 M/n = 125 MB；ncclAlltoAll 的 count = 每 peer 元素数', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 422, 'AR = RS + AG：(n-1)/n + (n-1)/n = 2(n-1)/n —— ZeRO 梯度 RS + 参数 AG 与 DP 同构，通信预算一文未涨', { size: 10 }));

  return scene(720, 438, body, { label: '八原语每 rank 发送账：均衡四 all-reduce 1.75 GB 其余 0.875 GB 与 n 解耦，root 四 broadcast reduce 依赖实现 gather scatter 有界，a2a 每 peer 0.125 GB 自环不发' }).svg();
}

/* ============================================================
 * 8.1 §4（block 0）— ring vs all-to-all 流量模式（n=4）
 * ============================================================ */
function ringVsA2a() {
  const UID = 'g8b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '流量模式：ring 只与 2 邻居通信（持续均匀负载）vs all-to-all 完全图（单跳直达、全 bisection 需求）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：ring 方形环
  body.push(zone({ x: 36, y: 48, w: 300, h: 220, label: 'ring all-reduce（n=4）' }));
  const r = [
    node({ id: 'r0', x: 76, y: 88, w: 64, h: 36, role: 'data', title: 'r0', mono: true, titleSize: 11 }),
    node({ id: 'r1', x: 232, y: 88, w: 64, h: 36, role: 'data', title: 'r1', mono: true, titleSize: 11 }),
    node({ id: 'r2', x: 232, y: 196, w: 64, h: 36, role: 'data', title: 'r2', mono: true, titleSize: 11 }),
    node({ id: 'r3', x: 76, y: 196, w: 64, h: 36, role: 'data', title: 'r3', mono: true, titleSize: 11 }),
  ];
  nodes.push(...r); body.push(...r.map((n) => n.svg));
  body.push(edge(r[0], r[1], { nodes, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'e0', arrow: false }));
  body.push(arrowHead(r[1].left[0], r[1].left[1], 1, 0));
  body.push(edge(r[1], r[2], { nodes, portFrom: r[1].bottom, portTo: r[2].top, pulse: true, dur: 1.2, delay: 0.4, uid: UID, eid: 'e1', arrow: false }));
  body.push(arrowHead(r[2].top[0], r[2].top[1], 0, 1));
  body.push(`<path class="dia-edge" d="M 232 214 H 140"/>`);
  body.push(pathPulse('M 232 214 H 140', { dur: 1.2, delay: 0.8 }));
  body.push(arrowHead(140, 214, -1, 0));
  body.push(`<path class="dia-edge" d="M 108 196 V 124"/>`);
  body.push(pathPulse('M 108 196 V 124', { dur: 1.2, delay: 1.2 }));
  body.push(arrowHead(108, 124, 0, -1));
  body.push(cap(186, 284, '2(n-1) = 6 步转发', { size: 9.5, anchor: 'middle' }));
  body.push(cap(186, 298, '邻居链路持续均匀负载', { size: 9.5, anchor: 'middle' }));

  // 右：a2a 完全图 K4
  body.push(zone({ x: 384, y: 48, w: 300, h: 220, label: 'all-to-all（完全图）' }));
  const q = [
    node({ id: 'q0', x: 424, y: 88, w: 64, h: 36, role: 'attn', title: 'r0', mono: true, titleSize: 11 }),
    node({ id: 'q1', x: 580, y: 88, w: 64, h: 36, role: 'attn', title: 'r1', mono: true, titleSize: 11 }),
    node({ id: 'q2', x: 580, y: 196, w: 64, h: 36, role: 'attn', title: 'r2', mono: true, titleSize: 11 }),
    node({ id: 'q3', x: 424, y: 196, w: 64, h: 36, role: 'attn', title: 'r3', mono: true, titleSize: 11 }),
  ];
  nodes.push(...q); body.push(...q.map((n) => n.svg));
  const sides = [
    ['M 488 106 H 580', 580, 106, 1, 0, 0.2],
    ['M 612 124 V 196', 612, 196, 0, 1, 0.6],
    ['M 580 214 H 488', 488, 214, -1, 0, 1.0],
    ['M 456 196 V 124', 456, 124, 0, -1, null],
  ];
  sides.forEach(([d, ax, ay, dx, dy, dl], i) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    if (dl !== null) body.push(pathPulse(d, { dur: 1.2, delay: dl }));
    body.push(arrowHead(ax, ay, dx, dy));
  });
  body.push(`<path class="dia-edge dash" d="M 482 124 L 586 196"/>`);
  body.push(pathPulse('M 482 124 L 586 196', { dur: 1.4, delay: 1.4 }));
  body.push(arrowHead(586, 196, 104, 72));
  body.push(`<path class="dia-edge dash" d="M 586 124 L 482 196"/>`);
  body.push(arrowHead(482, 196, -104, 72));
  body.push(cap(534, 284, '单跳直达 · 每 peer M/n = 0.125 GB', { size: 9.5, anchor: 'middle' }));
  body.push(cap(534, 298, '全 bisection 需求 + 突发 · 偏斜 incast（→10.3）', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 322, '语义差一条：all-reduce 有归约（数据被求和改变）；a2a 纯数据搬运 —— EP combine 加权求和在接收后本地做', { size: 10 }));
  body.push(cap(36, 340, '均衡 a2a 是一个排列（每 rank 收发 (n-1)/n·M）：NVSwitch 非阻塞对任意排列线速，域内总注入不变', { size: 10 }));

  return scene(720, 356, body, { label: 'n=4 流量模式对比：ring 四节点方形环六步转发邻居链路均匀负载，all-to-all 完全图单跳直达每 peer M/n 全 bisection 需求与偏斜 incast' }).svg();
}

/* ============================================================
 * 8.2 §2（block 0）— ring 两阶段：scatter-reduce + all-gather
 * ============================================================ */
function ringTwoPhase() {
  const UID = 'g8c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'ring all-reduce 两阶段：scatter-reduce + all-gather 各 n-1 步、每步每 rank 发 M/n —— 合计 2(n-1)/n·M', { size: 12, color: 'var(--ink-soft)' }));

  // 左：逻辑环 n=4
  body.push(zone({ x: 36, y: 48, w: 240, h: 200, label: '逻辑环（n=4）' }));
  const r = [
    node({ id: 'r0', x: 60, y: 80, w: 56, h: 32, role: 'data', title: 'r0', mono: true, titleSize: 10.5 }),
    node({ id: 'r1', x: 196, y: 80, w: 56, h: 32, role: 'data', title: 'r1', mono: true, titleSize: 10.5 }),
    node({ id: 'r2', x: 196, y: 180, w: 56, h: 32, role: 'data', title: 'r2', mono: true, titleSize: 10.5 }),
    node({ id: 'r3', x: 60, y: 180, w: 56, h: 32, role: 'data', title: 'r3', mono: true, titleSize: 10.5 }),
  ];
  nodes.push(...r); body.push(...r.map((n) => n.svg));
  body.push(edge(r[0], r[1], { nodes, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'e0', arrow: false }));
  body.push(arrowHead(r[1].left[0], r[1].left[1], 1, 0));
  body.push(edge(r[1], r[2], { nodes, portFrom: r[1].bottom, portTo: r[2].top, pulse: true, dur: 1.2, delay: 0.35, uid: UID, eid: 'e1', arrow: false }));
  body.push(arrowHead(r[2].top[0], r[2].top[1], 0, 1));
  body.push(`<path class="dia-edge" d="M 196 196 H 116"/>`);
  body.push(pathPulse('M 196 196 H 116', { dur: 1.2, delay: 0.7 }));
  body.push(arrowHead(116, 196, -1, 0));
  body.push(`<path class="dia-edge" d="M 88 180 V 112"/>`);
  body.push(pathPulse('M 88 180 V 112', { dur: 1.2, delay: 1.05 }));
  body.push(arrowHead(88, 112, 0, -1));
  body.push(cap(156, 236, '每步每 rank 发 M/n', { size: 9, anchor: 'middle' }));

  // 右上：6 步时间轴（手工刻度，字级可控）
  body.push(cap(310, 60, '2(n-1) = 6 步（n=4 示意）', { mono: true, size: 10 }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="310" y="81" width="186" height="10" rx="5"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="496" y="81" width="186" height="10" rx="5"/>`);
  body.push(cap(403, 74, 'scatter-reduce（3 步）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(589, 74, 'all-gather（3 步）', { size: 9.5, anchor: 'middle' }));
  body.push(`<line class="dia-edge strong" x1="310" y1="90" x2="682" y2="90"/>`);
  for (let s = 1; s <= 6; s++) {
    const x = 310 + (s - 0.5) * 62;
    body.push(`<line class="dia-edge" x1="${x}" y1="85" x2="${x}" y2="95"/>`);
    body.push(cap(x, 108, String(s), { mono: true, size: 9, anchor: 'middle' }));
  }
  body.push(cap(310, 130, 'scatter-reduce：部分和沿环累加，终态每 rank 恰持一个全和块', { size: 9.5 }));
  body.push(cap(310, 146, 'all-gather：各 rank 把自己的全和块沿环转发，终态集齐 n 个全和块', { size: 9.5 }));

  // 右下：两阶段终态矩阵（行 = rank，列 = chunk）
  body.push(matrix({ x: 330, y: 190, n: 4, cell: 18, gap: 3, on: () => true, hot: (i, j) => i === j, warm: (i, j) => i !== j, label: 'SR 终态' }));
  body.push(matrix({ x: 540, y: 190, n: 4, cell: 18, gap: 3, on: () => true, hot: () => true, label: 'AG 终态' }));
  for (let i = 0; i < 4; i++) {
    body.push(cap(322, 190 + i * 21 + 13, `r${i}`, { mono: true, size: 8, anchor: 'end' }));
    body.push(cap(532, 190 + i * 21 + 13, `r${i}`, { mono: true, size: 8, anchor: 'end' }));
    body.push(cap(330 + i * 21 + 9, 283, `c${i}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(cap(540 + i * 21 + 9, 283, `c${i}`, { mono: true, size: 8, anchor: 'middle' }));
  }
  body.push(cap(330, 302, 'hot = 全和块 · warm = 部分和', { size: 8.5 }));
  body.push(cap(540, 302, '集齐 n 个全和块 = 完整归约向量', { size: 8.5 }));

  body.push(cap(36, 330, '每 rank 发送 = (n-1)·M/n + (n-1)·M/n = 2(n-1)/n·M（式 1）；n=8, M=1 GB → 1.75 GB（接收同量）', { size: 10 }));
  body.push(cap(36, 348, '单阶段（RS 或 AG）每 rank (n-1)/n·M = 0.875 GB —— all-reduce 总量是它的 2 倍，勿混（§3 硬规则）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 366, '归纳不变式 recap training 4.1 §2.2：第 s 步每块部分和累加 s+1 份贡献 —— 本章不重推', { size: 10 }));

  return scene(720, 382, body, { label: 'ring 两阶段：逻辑环每步发 M/n，六步时间轴前三步 scatter-reduce 终态每 rank 持一个全和块对角，后三步 all-gather 终态集齐全部全和块，每 rank 合计 2(n-1)/n·M' }).svg();
}

/* ============================================================
 * 8.2 §3（block 1）— 下界证明：归约树 + 广播树两类不相交
 * ============================================================ */
function lowerBoundTrees() {
  const UID = 'g8d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '下界证明骨架：切 chunk → 按信息状态分类传输 → 证不相交 → 计数相加 —— 总流量 ≥ 2(n-1)M', { size: 12, color: 'var(--ink-soft)' }));

  // 归约树（部分和流入 r_c）
  body.push(zone({ x: 36, y: 48, w: 310, h: 170, label: '归约树 · 携带部分和（全和尚不存在）' }));
  const ra = node({ id: 'ra', x: 60, y: 84, w: 64, h: 32, role: 'data', title: 'r_a', mono: true, titleSize: 10.5 });
  const rb = node({ id: 'rb', x: 60, y: 148, w: 64, h: 32, role: 'data', title: 'r_b', mono: true, titleSize: 10.5 });
  const md = node({ id: 'md', x: 170, y: 116, w: 64, h: 32, role: 'op', title: '…', mono: true, titleSize: 10.5 });
  const rc = node({ id: 'rc', x: 262, y: 116, w: 64, h: 32, role: 'attn', title: 'r_c', mono: true, titleSize: 10.5 });
  nodes.push(ra, rb, md, rc); body.push(ra.svg, rb.svg, md.svg, rc.svg);
  [[ra, md, 0], [rb, md, 0.3], [md, rc, 0.6]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `u${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(294, 164, '首个持齐全和', { size: 8, anchor: 'middle' }));
  body.push(cap(190, 200, 'n-1 个贡献各首次离开源 rank 一次', { size: 9, anchor: 'middle' }));
  body.push(cap(190, 214, '源互异 · 每次 ≥ M/n → 流量 ≥ (n-1)·M/n', { size: 9, anchor: 'middle' }));

  // 广播树（全和流出 r_c）
  body.push(zone({ x: 374, y: 48, w: 310, h: 170, label: '广播树 · 携带全和（全和已存在）' }));
  const rc2 = node({ id: 'rc2', x: 398, y: 116, w: 64, h: 32, role: 'attn', title: 'r_c', mono: true, titleSize: 10.5 });
  const md2 = node({ id: 'md2', x: 496, y: 116, w: 64, h: 32, role: 'op', title: '…', mono: true, titleSize: 10.5 });
  const ra2 = node({ id: 'ra2', x: 596, y: 84, w: 64, h: 32, role: 'data', title: 'r_a', mono: true, titleSize: 10.5 });
  const rb2 = node({ id: 'rb2', x: 596, y: 148, w: 64, h: 32, role: 'data', title: 'r_b', mono: true, titleSize: 10.5 });
  nodes.push(rc2, md2, ra2, rb2); body.push(rc2.svg, md2.svg, ra2.svg, rb2.svg);
  [[rc2, md2, 0.9], [md2, ra2, 1.2], [md2, rb2, 1.5]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `d${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(530, 200, 'n-1 个 rank 各须流入 ≥ M/n 全和', { size: 9, anchor: 'middle' }));
  body.push(cap(530, 214, '广播树流量 ≥ (n-1)·M/n', { size: 9, anchor: 'middle' }));

  // 不相交相加链
  body.push(cap(360, 236, '两类传输不相交（一次传输要么携带部分和、要么携带全和）→ 字节直接相加', { size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  const t1 = node({ id: 't1', x: 36, y: 252, w: 130, h: 48, role: 'proj', title: '(n-1)·M/n', sub: '归约树界', mono: true, titleSize: 12, subSize: 8.5 });
  const t2 = node({ id: 't2', x: 196, y: 252, w: 130, h: 48, role: 'proj', title: '(n-1)·M/n', sub: '广播树界', mono: true, titleSize: 12, subSize: 8.5 });
  const t3 = node({ id: 't3', x: 356, y: 252, w: 150, h: 48, role: 'attn', title: '2(n-1)·M/n', sub: '每 chunk 下界', mono: true, titleSize: 12, subSize: 8.5 });
  const t4 = node({ id: 't4', x: 536, y: 252, w: 148, h: 48, role: 'attn', title: '2(n-1)M', sub: '总流量下界（式 3）', mono: true, titleSize: 13, subSize: 8.5 });
  nodes.push(t1, t2, t3, t4); body.push(t1.svg, t2.svg, t3.svg, t4.svg);
  body.push(cap(181, 280, '+', { size: 14, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(cap(341, 280, '=', { size: 14, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(cap(521, 280, '×n', { size: 12, anchor: 'middle', color: 'var(--ink-faint)' }));
  [[t1, t2, 1.6], [t2, t3, 1.9], [t3, t4, 2.2]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `s${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(521, 312, 'n chunk 求和', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 330, '推论（负载均衡）：max_r S_r ≥ (1/n)·Σ S_r ≥ 2(n-1)/n·M —— ring 每步每 rank 同量、天然负载均衡，达界 → 带宽最优', { size: 10 }));
  body.push(cap(36, 350, 'n=8, M=1 GB：总流量下界 14 GB · 每 rank 下界 1.75 GB = ring 实发 1.75 GB（达界）', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 372, '单阶段界单独归属：归约树界单独 = RS 总下界 (n-1)M、广播树界 = AG —— 每 rank (n-1)/n·M 是总量之半', { size: 10 }));
  body.push(cap(36, 390, '坑：把单阶段 0.875 GB 当 all-reduce 总下界会少算一半 —— 「ring 达界」变「ring 超界 2×」假象', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 408, '弱界 S_r ≥ M 只数「自有份离开」、漏中继转发（约再一个 M）—— 差 1.75× 不足闭合；必须切 chunk 分类传输', { size: 10 }));

  return scene(720, 424, body, { label: '下界证明：归约树 n-1 个贡献各离开源一次加广播树 n-1 个 rank 各流入全和，两类不相交相加每 chunk 2(n-1)M/n，乘 n 得总流量 2(n-1)M，负载均衡每 rank 2(n-1)/n·M ring 达界' }).svg();
}

/* ============================================================
 * 8.2 §4（block 2）— 逻辑环落到物理拓扑（八边环 + switch 路径）
 * ============================================================ */
function ringPhysical() {
  const UID = 'g8e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '逻辑环落到物理拓扑：环邻居关系是逻辑的 —— HGX 每跳实际走 GPU→NVSwitch→GPU', { size: 12, color: 'var(--ink-soft)' }));

  // 左：8 rank 逻辑环（八边形）
  body.push(zone({ x: 36, y: 48, w: 330, h: 240, label: '逻辑环（NCCL 搜索得到）' }));
  const cx = 201, cy = 175, R = 88;
  const centers = [];
  for (let k = 0; k < 8; k++) {
    const a = (-90 + k * 45) * Math.PI / 180;
    centers.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
  }
  const rn = centers.map(([px, py], k) => {
    const n = node({ id: `n${k}`, x: px - 20, y: py - 13, w: 40, h: 26, role: k === 0 ? 'attn' : 'data', title: String(k), mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let k = 0; k < 8; k++) {
    const [ax, ay] = centers[k], [bx, by] = centers[(k + 1) % 8];
    const p = anchor(ax, ay, 40, 26, bx, by);
    const q = anchor(bx, by, 40, 26, ax, ay);
    const d = `M ${p[0].toFixed(1)} ${p[1].toFixed(1)} L ${q[0].toFixed(1)} ${q[1].toFixed(1)}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.6, delay: k * 0.2 }));
    body.push(arrowHead(q[0], q[1], bx - ax, by - ay));
  }
  body.push(cap(201, 302, '每 rank 只与 prev/next 通信 · 环排列任意（非阻塞）', { size: 9.5, anchor: 'middle' }));

  // 右：物理 switch 路径
  body.push(zone({ x: 390, y: 48, w: 294, h: 130, label: '物理路径（HGX H100）' }));
  const gi = node({ id: 'gi', x: 406, y: 88, w: 80, h: 52, role: 'data', title: 'GPU_i', sub: 'rank i', mono: true, titleSize: 11, subSize: 8 });
  const sw = node({ id: 'sw', x: 506, y: 88, w: 86, h: 52, role: 'op', title: 'NVSwitch', sub: 'crossbar', mono: true, titleSize: 10.5, subSize: 8 });
  const gj = node({ id: 'gj', x: 608, y: 88, w: 60, h: 52, role: 'data', title: 'GPU_i+1', mono: true, titleSize: 9.5 });
  nodes.push(gi, sw, gj); body.push(gi.svg, sw.svg, gj.svg);
  body.push(edge(gi, sw, { nodes, pulse: true, dur: 1.1, delay: 0.3, uid: UID, eid: 'p0', arrow: false }));
  body.push(arrowHead(sw.left[0], sw.left[1], 1, 0));
  body.push(edge(sw, gj, { nodes, pulse: true, dur: 1.1, delay: 0.8, uid: UID, eid: 'p1', arrow: false }));
  body.push(arrowHead(gj.left[0], gj.left[1], 1, 0));
  body.push(cap(537, 158, '每跳单向 raw 450 / 模型 370.8 GB/s', { size: 9, anchor: 'middle' }));
  body.push(cap(537, 196, '环排列任意：非阻塞保证任意排列线速接通', { size: 9.5, anchor: 'middle' }));
  body.push(`<path class="dia-edge dash" d="M 366 168 C 378 168, 378 114, 390 114"/>`);

  body.push(cap(36, 322, 'ncclBuildRings（rings.cc L29）：沿 next 走 n 步生成环 + 两道校验 —— loopback 回起点（L51）+ 全 rank 恰覆盖一次', { size: 10 }));
  body.push(cap(36, 340, '直连拓扑边界：环边须落真实直连链路、每对带宽均分 —— 下界是拓扑无关的流量账，达界所需带宽由拓扑供给', { size: 10 }));

  return scene(720, 356, body, { label: '八 rank 逻辑环每条边映射为 GPU 经 NVSwitch crossbar 到下一 GPU 的物理路径，每跳有效 370.8 GB/s，ncclBuildRings 两道校验保证单循环全覆盖' }).svg();
}

/* ============================================================
 * 8.2 §5（block 3）— chunking 流水时序 + 多 ring 带宽聚合
 * ============================================================ */
function ringPipeline() {
  const UID = 'g8f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'chunking/pipelining：不减少字节数，但让 n 条邻居链路同时忙 —— 保证下界带宽可达', { size: 12, color: 'var(--ink-soft)' }));

  // 步 × rank 网格（scatter-reduce，n=4）
  for (let s = 1; s <= 3; s++) body.push(cap(135 + (s - 1) * 180 + 75, 60, `步 ${s}`, { mono: true, size: 10, anchor: 'middle' }));
  const sends = [[0, 3, 2], [1, 0, 3], [2, 1, 0], [3, 2, 1]];
  sends.forEach((row, i) => {
    const y = 70 + i * 44;
    body.push(cap(120, y + 20, `r${i}`, { mono: true, size: 10, anchor: 'end' }));
    row.forEach((c, s) => {
      const n = node({ id: `c${i}${s}`, x: 135 + s * 180, y, w: 150, h: 30, role: 'data', title: `发 c${c}`, mono: true, titleSize: 10.5 });
      nodes.push(n); body.push(n.svg);
    });
    const d = `M 135 ${y + 36} H 645`;
    body.push(`<line class="dia-edge" x1="135" y1="${y + 36}" x2="645" y2="${y + 36}"/>`);
    body.push(pathPulse(d, { dur: 1.8, delay: i * 0.3 }));
  });
  body.push(`<line class="dia-edge" x1="135" y1="252" x2="285" y2="252"/>`);
  body.push(`<line class="dia-edge" x1="315" y1="252" x2="645" y2="252"/>`);
  body.push(cap(210, 266, '填充期', { size: 9, anchor: 'middle' }));
  body.push(cap(480, 266, '稳态：n 条邻居链路同时满载', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 288, '延迟项 = 流水线填充/排空 2(n-1)α（与 M 无关、不随 chunk 数叠加）；带宽项 = 稳态传输 2(n-1)/n·Mβ', { size: 10 }));
  body.push(cap(36, 306, '每字节延迟摊薄 2(n-1)α/M → 0：n=8, M=1 GB → 2.8e-14 s/B ≈ β 的 1/96 —— 大消息牢固带宽主导', { size: 10 }));

  // 多 ring 带宽聚合
  body.push(cap(36, 330, '多 ring 并行（k channel 各搬 M/k）：带宽项 ÷ k、延迟项不变（式 6）', { size: 10.5 }));
  const ks = [['k=1', 4.72, 'soft', '4.72 ms'], ['k=4', 1.18, 'warm', '1.18 ms'], ['k=8', 0.59, 'seal', '0.59 ms（延迟项恒 0.028 ms）']];
  ks.forEach(([lab, t, tone, val], i) => {
    const y = 342 + i * 22;
    body.push(cap(110, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="120" y="${y}" width="${t * 80}" height="14" rx="3"/>`);
    body.push(cap(120 + t * 80 + 8, y + 11, val, { mono: true, size: 9.5 }));
  });
  body.push(cap(36, 410, '口径：线性聚合假设各 channel 走不同 link 子集；k 增到注入带宽饱和后不再增益 —— MAXCHANNELS 64 是结构上限', { size: 9.5 }));

  return scene(720, 424, body, { label: 'scatter-reduce 流水时序：四 rank 三步步步发一块链路不空等，填充期后稳态四链路满载；多 ring k=1/4/8 带宽项 4.72/1.18/0.59 ms 延迟项不变' }).svg();
}

/* ============================================================
 * 8.3 §3（block 0）— 12-rank 双二叉树（mirror 互补）
 * ============================================================ */
function doubleBinaryTree() {
  const UID = 'g8g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '双二叉树（12-rank mirror 例，trees.cc 注释）：没有 rank 在两棵树都是内部节点 —— 各处理 M/2', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, cx, cy, role, label) => {
    const n = node({ id, x: cx - 16, y: cy - 12, w: 32, h: 24, role, title: label ?? id.replace('t', ''), mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
    return [cx, cy];
  };
  const ln = (a, b, dl) => {
    const d = `M ${a[0]} ${a[1] + 12} L ${b[0]} ${b[1] - 12}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    if (dl !== null) body.push(pathPulse(d, { dur: 1.4, delay: dl }));
    body.push(arrowHead(b[0], b[1] - 12, b[0] - a[0], b[1] - a[1]));
  };
  const hz = (a, b, dl) => {
    const d = `M ${a[0] + 16} ${a[1]} H ${b[0] - 16}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    if (dl !== null) body.push(pathPulse(d, { dur: 1.4, delay: dl }));
    body.push(arrowHead(b[0] - 16, b[1], b[0] - a[0], 0));
  };

  // 第一棵：内部节点 = 偶数 rank
  body.push(zone({ x: 36, y: 48, w: 320, h: 240, label: '第一棵（btree）· 内部节点 = 偶数' }));
  const T1 = {
    0: mk('t0', 70, 84, 'attn'), 8: mk('t8', 250, 84, 'attn'), 4: mk('t4', 110, 136, 'attn'),
    2: mk('t2', 70, 188, 'attn'), 6: mk('t6', 150, 188, 'attn'), 10: mk('t10', 250, 188, 'attn'),
    1: mk('t1', 48, 240, 'plain'), 3: mk('t3', 92, 240, 'plain'), 5: mk('t5', 128, 240, 'plain'),
    7: mk('t7', 172, 240, 'plain'), 9: mk('t9', 228, 240, 'plain'), 11: mk('t11', 272, 240, 'plain'),
  };
  hz(T1[0], T1[8], 0.2);
  ln(T1[0], T1[4], 0); ln(T1[4], T1[2], null); ln(T1[4], T1[6], 0.5);
  ln(T1[8], T1[10], 0.5); ln(T1[2], T1[1], null); ln(T1[2], T1[3], null);
  ln(T1[6], T1[5], null); ln(T1[6], T1[7], null); ln(T1[10], T1[9], null); ln(T1[10], T1[11], 1.1);

  // 第二棵：mirror，内部节点 = 奇数 rank（root 11）
  body.push(zone({ x: 380, y: 48, w: 304, h: 240, label: '第二棵（mirror）· 内部节点 = 奇数' }));
  const T2 = {
    3: mk('t3b', 418, 84, 'proj', '3'), 11: mk('t11b', 598, 84, 'proj', '11'), 7: mk('t7b', 558, 136, 'proj', '7'),
    1: mk('t1b', 418, 188, 'proj', '1'), 5: mk('t5b', 518, 188, 'proj', '5'), 9: mk('t9b', 598, 188, 'proj', '9'),
    0: mk('t0b', 400, 240, 'plain', '0'), 2: mk('t2b', 440, 240, 'plain', '2'), 4: mk('t4b', 496, 240, 'plain', '4'),
    6: mk('t6b', 540, 240, 'plain', '6'), 8: mk('t8b', 576, 240, 'plain', '8'), 10: mk('t10b', 620, 240, 'plain', '10'),
  };
  hz(T2[11], T2[3], 0.3);
  ln(T2[11], T2[7], 0.6); ln(T2[3], T2[1], null); ln(T2[7], T2[5], null); ln(T2[7], T2[9], 0.9);
  ln(T2[1], T2[0], null); ln(T2[1], T2[2], null); ln(T2[5], T2[4], null); ln(T2[5], T2[6], null);
  ln(T2[9], T2[8], null); ln(T2[9], T2[10], 1.2);

  body.push(cap(36, 306, '朱染 = 第一棵内部（偶）· 蓝染 = 第二棵内部（奇）· 纸面 = 叶子 —— 每 rank 至多在一棵树当内部节点', { size: 9.5 }));
  body.push(cap(36, 328, '单树缺陷：内部节点发 3D（上行 D + 下行 2D）、叶子发 D，root 收 2D 热点 —— 每 rank 可达 3M，带宽不最优', { size: 10 }));
  body.push(cap(36, 346, '双树：各处理 D = M/2，每 rank ≤ 3D + D = 2M —— 与 ring 同 leading term（差 2M/n），带宽最优且延迟 O(log n)', { size: 10, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(36, 364, 'n=8, M=1 GB：双树 ≈2 GB 对 ring 1.75 GB（差 0.25 GB = 2M/n）；ncclGetDtree：偶 mirror / 奇 shift（trees.cc L90）', { size: 10 }));

  return scene(720, 380, body, { label: '12-rank 双二叉树：第一棵内部节点为偶数 rank，mirror 第二棵内部节点为奇数 rank，互补无重叠，各处理一半数据每 rank 至多 3D 加 D 等于 2M 带宽最优' }).svg();
}

/* ============================================================
 * 8.3 §4（afterHeading）— ring/tree 交叉 M*（实测带宽差才有交叉）
 * ============================================================ */
function ringTreeCrossover() {
  const UID = 'g8h';
  const body = [];

  body.push(cap(36, 24, 'ring/tree 交叉 M*：理想模型带宽项相同、无交叉 —— 交叉必须来自实测有效带宽差', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴：x = M (MB) 0..4 → 100..660；y = T (μs) 0..80 → 310..90
  const X = (m) => 100 + m * 140;
  const Y = (t) => 310 - t * 2.75;
  body.push(`<line class="dia-edge strong" x1="100" y1="310" x2="660" y2="310"/>`);
  body.push(arrowHead(660, 310, 1, 0));
  body.push(`<line class="dia-edge strong" x1="100" y1="310" x2="100" y2="90"/>`);
  body.push(arrowHead(100, 90, 0, -1));
  for (let m = 0; m <= 4; m++) {
    body.push(`<line class="dia-edge" x1="${X(m)}" y1="305" x2="${X(m)}" y2="315"/>`);
    body.push(cap(X(m), 328, String(m), { mono: true, size: 9, anchor: 'middle' }));
  }
  for (let t = 0; t <= 80; t += 20) {
    body.push(`<line class="dia-edge" x1="95" y1="${Y(t)}" x2="100" y2="${Y(t)}"/>`);
    body.push(cap(92, Y(t) + 3, String(t), { mono: true, size: 9, anchor: 'end' }));
  }
  body.push(cap(660, 344, 'M (MB)', { size: 9.5, anchor: 'end' }));
  body.push(cap(100, 78, 'T (μs)', { size: 9.5 }));

  // T_ring = 28 + 4.7195·M；T_tree = 12 + 13.4615·M（μs, M in MB）
  body.push(`<path d="M 100 ${Y(28)} L 660 ${Y(28 + 4 * 4.7195)}" fill="none" stroke="var(--seal)" stroke-width="2"/>`);
  body.push(`<path d="M 100 ${Y(12)} L 660 ${Y(12 + 4 * 13.4615)}" fill="none" stroke="var(--ink-soft)" stroke-width="2"/>`);
  body.push(cap(656, Y(28 + 4 * 4.7195) - 8, 'ring', { mono: true, size: 10, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(656, Y(12 + 4 * 13.4615) - 8, 'tree', { mono: true, size: 10, anchor: 'end', color: 'var(--ink-soft)' }));
  body.push(cap(108, Y(28) - 6, '28 μs', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(108, Y(12) + 14, '12 μs', { mono: true, size: 8.5, color: 'var(--ink-soft)' }));

  // 交叉点 M* = 1.83 MB, T ≈ 36.6 μs
  const mx = X(1.83), my = Y(36.64);
  body.push(`<path class="dia-edge dash" d="M ${mx} ${my} V 310"/>`);
  body.push(`<path class="dia-edge dash" d="M 100 ${my} H ${mx}"/>`);
  body.push(`<circle cx="${mx}" cy="${my}" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(108, my - 6, 'T ≈ 36.6 μs', { mono: true, size: 8.5 }));
  body.push(cap(mx, 344, 'M* ≈ 1.83 MB', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(200, 150, 'M < M*：tree 赢', { size: 10, anchor: 'middle', color: 'var(--ink-soft)' }));
  body.push(cap(200, 166, '（延迟项主导）', { size: 9, anchor: 'middle' }));
  body.push(cap(540, 250, 'M > M*：ring 赢', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(540, 266, '（实测带宽项主导）', { size: 9, anchor: 'middle' }));
  body.push(cap(100, 364, '斜率 = 2(n-1)/n·β_eff：ring 1/370.8 GB/s 对 tree 1/130 GB/s（教学假设，示意 tree 实测折扣；真实值以 nccl-tests 为准）', { size: 9.5 }));

  body.push(cap(36, 386, 'M* = n(n-1-log₂n)·α / ((n-1)·Δβ)（式 2）；n=8, α=2 μs, Δβ = 4.995e-12 s/B → M* ≈ 1.83 MB', { mono: true, size: 10 }));
  body.push(cap(36, 404, '前提 Δβ > 0：理想模型 Δβ = 0 除零、M* 无定义 —— 「从纯 α-β 凭空解 M*」是坑（带宽项相消、方程与 M 无关）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 422, '三临界勿混：M* 1.83 MB（算法间）≠ M_c 5.93 MB（ring 内部两项，8.1）≠ C* 667.4 KB（协议间，8.4）', { size: 10 }));

  return scene(720, 438, body, { label: 'ring 与 tree 时间随消息尺寸直线：ring 截距 28 μs 斜率小，tree 截距 12 μs 斜率大，交点 M* 约 1.83 MB 处 36.6 μs，小消息 tree 赢大消息 ring 赢，交叉来自实测带宽差' }).svg();
}

/* ============================================================
 * 8.4 §2（block 0）— ncclCommInitRank 初始化全流程
 * ============================================================ */
function ncclInitFlow() {
  const UID = 'g8i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'ncclCommInitRank 全流程：建组拆成异步 job 链 —— 拓扑 → 路径 → 图搜索 → transport → devComm（2.32.7-1）', { size: 12, color: 'var(--ink-soft)' }));

  const a = node({ id: 'a', x: 36, y: 48, w: 190, h: 44, role: 'proj', title: 'ncclCommInitRank', sub: 'init.cc L3071 · 公共 API', mono: true, titleSize: 11, subSize: 8 });
  const b = node({ id: 'b', x: 270, y: 48, w: 190, h: 44, role: 'op', title: 'InitRankDev → Func', sub: 'L2975 / L2229 · 异步 job', mono: true, titleSize: 10.5, subSize: 8 });
  const bs = node({ id: 'bs', x: 514, y: 48, w: 170, h: 44, role: 'norm', title: 'bootstrapInit', sub: 'L2314 · 带外 allgather TCP', mono: true, titleSize: 10.5, subSize: 7.5 });
  nodes.push(a, b, bs); body.push(a.svg, b.svg, bs.svg);
  body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'j0', arrow: false }));
  body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  body.push(edge(b, bs, { nodes, pulse: true, dur: 1.1, delay: 0.3, uid: UID, eid: 'j1', arrow: false }));
  body.push(arrowHead(bs.left[0], bs.left[1], 1, 0));
  body.push(`<path class="dia-edge" d="M 365 92 V 120"/>`);
  body.push(pathPulse('M 365 92 V 120', { dur: 1.0, delay: 0.6 }));
  body.push(arrowHead(365, 120, 0, 1));

  body.push(zone({ x: 36, y: 120, w: 648, h: 150, label: 'initTransportsRank（init.cc L1219）—— 初始化主体' }));
  const stages = [
    ['拓扑检测', 'NVML/PCI → XML → topo', 'cache', 'topo.cc L1994'],
    ['路径计算', 'PATH 梯子 + 带宽', 'proj', 'paths.cc'],
    ['图搜索', 'ring/tree/collNet/nvls', 'attn', 'search.cc L1245'],
    ['transport 建立', 'canConnect → setup', 'op', 'init.cc L1957'],
    ['devComm', 'host→device 拷贝', 'data', 'L2076 / L847'],
  ];
  const sn = stages.map(([title, sub, role, fl], i) => {
    const n = node({ id: `s${i}`, x: 52 + i * 124, y: 156, w: 116, h: 64, role, title, sub, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(52 + i * 124 + 58, 236, fl, { mono: true, size: 7.5, anchor: 'middle' }));
    return n;
  });
  for (let i = 0; i < 4; i++) {
    body.push(edge(sn[i], sn[i + 1], { nodes, pulse: true, dur: 1.1, delay: 0.5 + i * 0.3, uid: UID, eid: `t${i}`, arrow: false }));
    body.push(arrowHead(sn[i + 1].left[0], sn[i + 1].left[1], 1, 0));
  }
  body.push(cap(52, 258, 'SM90_NVLINK_BW 20.6 / SM100 40.1（模型值 vs raw 25/50 两层账）· ringGraph maxChannels 32 · treeGraph 取 ring nChannels', { size: 8.5 }));

  body.push(cap(36, 292, '图搜索 ncclTopoCompute 逐算法缓存；ncclTopoGraph 持 bwIntra/bwInter/nChannels —— 8.2/8.3 带宽账的代码载体', { size: 10 }));
  body.push(cap(36, 310, '路径梯子：PATH_NVL 1 / NVB 2 / C2C 3 / PIX 4 / PXB 5 / PXN 7 / PHB 8 / SYS 9（越小越近）—— transport 选择与 LL128 门控用', { size: 10 }));

  return scene(720, 326, body, { label: 'ncclCommInitRank 经异步 job 链分 bootstrap 带外 allgather 与 initTransportsRank 五阶段：拓扑检测、路径计算、逐算法图搜索、transport 建立、devComm 拷贝' }).svg();
}

/* ============================================================
 * 8.4 §4（afterHeading）— transport 四件套 + NVLS 独立 + 延迟账
 * ============================================================ */
function transportArchitecture() {
  const UID = 'g8j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'transport 四件套：原语是语义、链路是放置 —— canConnect 按拓扑逐一试，NVLS 独立路径不在表', { size: 12, color: 'var(--ink-soft)' }));

  const src = node({ id: 'src', x: 36, y: 64, w: 150, h: 64, role: 'data', title: 'rank 对 (src, dst)', sub: 'communicator 放置决定路径', titleSize: 11, subSize: 8 });
  const cc = node({ id: 'cc', x: 230, y: 64, w: 130, h: 64, role: 'op', title: 'canConnect', sub: '按拓扑逐一试', mono: true, titleSize: 11, subSize: 8.5 });
  nodes.push(src, cc); body.push(src.svg, cc.svg);
  body.push(edge(src, cc, { nodes, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 's', arrow: false }));
  body.push(arrowHead(cc.left[0], cc.left[1], 1, 0));

  const cards = [
    ['p2p · NVLink/PCIe 直访', 'p2pMap L358 · cudaIpcOpenMemHandle · useRead', 'attn'],
    ['shm · 同机共享内存', 'P2P 不可行（ACS/IOMMU）· proxy 线程拷贝', 'norm'],
    ['net · 跨机 RDMA/socket', 'GDR 协商 · sendProxyProgress L1327', 'proj'],
    ['collNet · IB SHARP 链', 'sharedBuffersInit L416 · 部署归 10.2', 'cache'],
  ];
  cards.forEach(([title, sub, role], i) => {
    const y = 48 + i * 60;
    const n = node({ id: `tr${i}`, x: 410, y, w: 274, h: 52, role, title, sub, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    const d = `M 360 96 C 385 96, 385 ${y + 26}, 410 ${y + 26}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.3, delay: 0.3 + i * 0.3 }));
    body.push(arrowHead(410, y + 26, 1, 0));
  });

  body.push(zone({ x: 36, y: 170, w: 324, h: 118, tone: 'seal', label: 'NVLS 独立路径（不在 ncclTransports[]）' }));
  const nv = node({ id: 'nv', x: 52, y: 204, w: 292, h: 52, role: 'attn', title: 'NVLS · NVSwitch 网内归约', sub: 'nvls.cc / multicast.cc · cuMulticastCreate L270', titleSize: 10.5, subSize: 7.5 });
  nodes.push(nv); body.push(nv.svg);
  body.push(`<path class="dia-edge dash" d="M 295 128 C 295 158, 352 162, 352 188 C 352 200, 344 196, 340 204"/>`);
  body.push(arrowHead(340, 204, 0, 1));
  body.push(cap(240, 160, '网内归约 ≠ 点对点搬运', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 312, 'transport 每跳硬件延迟（ring + Simple，cost_model.cc hwLatencies 校准值）：', { size: 10.5 }));
  const lats = [['NVLink', 3.4, 'seal', '3.4 μs'], ['PCI', 5.7, 'warm', '5.7 μs（1.7×）'], ['NET', 14.0, 'soft', '14.0 μs（≈4.1× NVLink）']];
  lats.forEach(([lab, t, tone, val], i) => {
    const y = 324 + i * 22;
    body.push(cap(130, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="140" y="${y}" width="${t * 26}" height="14" rx="3"/>`);
    body.push(cap(140 + t * 26 + 8, y + 11, val, { mono: true, size: 9.5 }));
  });
  body.push(cap(400, 335, '「高频集合留 NVLink 域内」的延迟侧定量依据', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(400, 357, '（带宽侧 9× 差 recap 7.2 §7）', { size: 9.5 }));
  body.push(cap(36, 392, '口径：3.4/5.7/14.0 μs 是 NCCL 内置校准常数，与教学假设 α=2 μs 不混；真实值随平台/驱动变，以实测为准', { size: 9.5 }));

  return scene(720, 408, body, { label: 'transport 选择：rank 对经 canConnect 按拓扑分到 p2p shm net collNet 四件套，NVLS 网内归约走独立路径；每跳延迟 NVLink 3.4 PCI 5.7 NET 14.0 μs' }).svg();
}

/* ============================================================
 * 8.4 §5（block 1）— proxy 线程：GDR 控制面 vs 非 GDR 数据面
 * ============================================================ */
function proxyGdr() {
  const UID = 'g8k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'proxy 线程：GPU kernel 调不了 NIC 的 host 侧 API —— GDR 下只做控制面，非 GDR 还做数据面', { size: 12, color: 'var(--ink-soft)' }));

  // GDR 路径
  body.push(cap(36, 60, 'GDR 路径：数据面零拷贝', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const k = node({ id: 'k', x: 36, y: 72, w: 120, h: 48, role: 'attn', title: 'GPU kernel', sub: '写数据', titleSize: 11, subSize: 8.5 });
  const h = node({ id: 'h', x: 200, y: 72, w: 140, h: 48, role: 'cache', title: 'GPU HBM 缓冲', sub: 'NCCL_STEPS 8 槽', titleSize: 11, subSize: 8.5 });
  const n = node({ id: 'n', x: 384, y: 72, w: 110, h: 48, role: 'proj', title: 'NIC', sub: 'RDMA READ', titleSize: 11, subSize: 8.5 });
  const f = node({ id: 'f', x: 538, y: 72, w: 146, h: 48, role: 'data', title: 'fabric', sub: '远端 GPU HBM', titleSize: 11, subSize: 8.5 });
  nodes.push(k, h, n, f); body.push(k.svg, h.svg, n.svg, f.svg);
  [[k, h, 0], [h, n, 0.5], [n, f, 1.0]].forEach(([a2, b2, dl], i) => {
    body.push(edge(a2, b2, { nodes, pulse: true, dur: 1.3, delay: dl, uid: UID, eid: `g${i}`, arrow: false }));
    body.push(arrowHead(b2.left[0], b2.left[1], 1, 0));
  });
  body.push(cap(362, 66, 'RDMA READ', { mono: true, size: 7.5, anchor: 'middle' }));
  const px = node({ id: 'px', x: 200, y: 150, w: 140, h: 40, role: 'op', title: 'proxy 线程', sub: '只提交 + 轮询', titleSize: 10.5, subSize: 8 });
  nodes.push(px); body.push(px.svg);
  body.push(`<path class="dia-edge dash" d="M 270 150 V 120"/>`);
  body.push(arrowHead(270, 120, 0, -1));
  body.push(cap(280, 138, 'posted', { mono: true, size: 8 }));
  body.push(`<path class="dia-edge dash" d="M 340 170 C 380 170, 439 150, 439 120"/>`);
  body.push(arrowHead(439, 120, 0, -1));
  body.push(cap(400, 186, 'isend / test（done++）', { size: 8, anchor: 'middle' }));
  body.push(cap(36, 206, 'proxy 只提交描述符不拷数据 —— NIC 经 RDMA 直读 GPU HBM（GDR 硬件机制归 9.2）', { size: 9.5 }));

  // 非 GDR 路径
  body.push(cap(36, 236, '非 GDR 路径（host bounce）：多一次 D2H + CPU 拷贝', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const k2 = node({ id: 'k2', x: 36, y: 248, w: 110, h: 48, role: 'attn', title: 'GPU kernel', titleSize: 10.5 });
  const h2 = node({ id: 'h2', x: 180, y: 248, w: 120, h: 48, role: 'cache', title: 'GPU HBM', titleSize: 10.5 });
  const hb = node({ id: 'hb', x: 334, y: 248, w: 130, h: 48, role: 'data', title: 'host 缓冲', sub: 'D2H 落地', titleSize: 10.5, subSize: 8 });
  const n2 = node({ id: 'n2', x: 498, y: 248, w: 90, h: 48, role: 'proj', title: 'NIC', titleSize: 10.5 });
  const f2 = node({ id: 'f2', x: 622, y: 248, w: 62, h: 48, role: 'data', title: 'fabric', titleSize: 9.5 });
  nodes.push(k2, h2, hb, n2, f2); body.push(k2.svg, h2.svg, hb.svg, n2.svg, f2.svg);
  [[k2, h2, 1.3], [h2, hb, 1.6], [hb, n2, 1.9], [n2, f2, 2.2]].forEach(([a2, b2, dl], i) => {
    body.push(edge(a2, b2, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `b${i}`, arrow: false }));
    body.push(arrowHead(b2.left[0], b2.left[1], 1, 0));
  });
  body.push(cap(317, 266, 'proxy 拷', { size: 8, anchor: 'middle' }));
  body.push(cap(36, 312, '数据面也经 proxy 线程拷贝（shm 同）—— 带宽延迟都差', { size: 9.5 }));

  // 三计数器状态机
  body.push(cap(36, 340, 'net 进度状态机（sendProxyProgress L1327）：三计数器在 8 个缓冲槽上轮转', { size: 10.5 }));
  const c1 = node({ id: 'c1', x: 36, y: 352, w: 140, h: 36, role: 'op', title: 'posted', sub: '向 GPU 发布可写槽', mono: true, titleSize: 10.5, subSize: 7.5 });
  const c2 = node({ id: 'c2', x: 220, y: 352, w: 140, h: 36, role: 'proj', title: 'transmitted', sub: '向 NIC 提交 isend/irecv', mono: true, titleSize: 10.5, subSize: 7.5 });
  const c3 = node({ id: 'c3', x: 404, y: 352, w: 140, h: 36, role: 'norm', title: 'done', sub: 'test 轮询到完成', mono: true, titleSize: 10.5, subSize: 7.5 });
  nodes.push(c1, c2, c3); body.push(c1.svg, c2.svg, c3.svg);
  [[c1, c2, 2.4], [c2, c3, 2.7]].forEach(([a2, b2, dl], i) => {
    body.push(edge(a2, b2, { nodes, pulse: true, dur: 1.0, delay: dl, uid: UID, eid: `c${i}`, arrow: false }));
    body.push(arrowHead(b2.left[0], b2.left[1], 1, 0));
  });
  body.push(cap(684, 366, 'NCCL_STEPS = 8', { mono: true, size: 9, anchor: 'end' }));
  body.push(cap(684, 382, 'maxDepth 防槽位覆写', { size: 8.5, anchor: 'end' }));

  return scene(720, 404, body, { label: 'GDR 路径 GPU kernel 写 HBM 缓冲后 NIC 经 RDMA 直读零拷贝，proxy 只提交描述符与轮询；非 GDR 多一次 proxy D2H 拷贝到 host 缓冲；posted transmitted done 三计数器八槽轮转' }).svg();
}

/* ============================================================
 * 8.4 §6（block 2）— LL / LL128 / Simple 三协议 line 布局
 * ============================================================ */
function protocolLines() {
  const UID = 'g8l';
  const body = [];

  body.push(cap(36, 24, '三协议 line 布局：同步开销与带宽利用率不可兼得 —— flag 跟在 data 后，flag 即同步', { size: 12, color: 'var(--ink-soft)' }));

  // LL 16B line
  body.push(cap(36, 90, 'LL · 16 B line', { mono: true, size: 10.5, weight: '650' }));
  const ll = [['data1 4B', 'warm'], ['flag1 4B', 'seal'], ['data2 4B', 'warm'], ['flag2 4B', 'seal']];
  ll.forEach(([lab, tone], i) => {
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="${170 + i * 80}" y="72" width="80" height="28" rx="3"/>`);
    body.push(cap(170 + i * 80 + 40, 90, lab, { mono: true, size: 8, anchor: 'middle', color: tone === 'seal' ? '#fdfcf9' : 'var(--ink-soft)' }));
  });
  body.push(cap(510, 90, 'η = 8/16 = 50%', { mono: true, size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(170, 116, 'readLL 自旋等 flag1==flag && flag2==flag —— 无独立 fence，省 proxy 往返（延迟最低）', { size: 9 }));

  // LL128 128B line
  body.push(cap(36, 166, 'LL128 · 128 B line', { mono: true, size: 10.5, weight: '650' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="170" y="148" width="450" height="28" rx="3"/>`);
  body.push(cap(395, 166, 'data 15×8B = 120B', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink-soft)' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="620" y="148" width="30" height="28" rx="3"/>`);
  body.push(cap(635, 166, 'flag', { mono: true, size: 7, anchor: 'middle', color: '#fdfcf9' }));
  body.push(cap(170, 192, 'η = 93.75% · 需 128B 写原子性，仅 NVLink 保证（门控 intra ≤ PATH_NVB，勿绕过）', { size: 9 }));

  // Simple 512B grain
  body.push(cap(36, 242, 'Simple · 512 B grain', { mono: true, size: 10.5, weight: '650' }));
  body.push(`<rect x="170" y="224" width="480" height="28" rx="3" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1.1"/>`);
  body.push(cap(410, 242, '无 per-line flag · 全数据（同步 = proxy + fence + connFifo step）', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink-soft)' }));
  body.push(cap(660, 242, 'η → 1', { mono: true, size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(170, 268, '大消息 flag 开销摊薄趋零、满带宽；代价每跳延迟最高（Simple 3.4 μs vs LL 0.6 μs）', { size: 9 }));

  // 带宽项账
  body.push(cap(36, 296, 'M=1 GB, n=8 ring（C=1.75 GB）带宽项：', { size: 10.5 }));
  const rows = [['LL', 9.44, 'soft', '9.44 ms（一半带宽换最低延迟）'], ['LL128', 5.03, 'warm', '5.03 ms'], ['Simple', 4.72, 'seal', '4.72 ms']];
  rows.forEach(([lab, t, tone, val], i) => {
    const y = 308 + i * 22;
    body.push(cap(130, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="140" y="${y}" width="${t * 40}" height="14" rx="3"/>`);
    body.push(cap(140 + t * 40 + 8, y + 11, val, { mono: true, size: 9.5 }));
  });

  body.push(cap(36, 380, '协议交叉 C* = Δlat·BW·η/(1-η) = 1.8 μs × 370.8 GB/s = 667.4 KB（每 rank）→ M* ≈ 381.4 KB；C<C* LL 赢、C>C* Simple 赢', { size: 10 }));
  body.push(cap(36, 398, '坑：PCIe 平台强开 LL128 → flag 与 data 非原子到达、数据损坏（docs 逐字）；LL128 只在 NVLink 启用，勿绕过门控', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 414, body, { label: '三协议 line 布局：LL 16B 四槽 data flag 各半效率 50%，LL128 128B 十五 data 槽加一 flag 槽 93.75%，Simple 512B 无 flag 趋 100%；带宽项 9.44 对 5.03 对 4.72 ms' }).svg();
}

/* ============================================================
 * 8.4 §9（block 3）— device dispatch：编译期特化 + 函数指针表
 * ============================================================ */
function deviceDispatch() {
  const UID = 'g8m';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'device dispatch：编译期模板特化 + 函数指针表 —— 运行期零分支，选择成本编译期一次付清', { size: 12, color: 'var(--ink-soft)' }));

  const d1 = node({ id: 'd1', x: 36, y: 48, w: 200, h: 44, role: 'proj', title: 'ncclDevKernel_Generic', sub: 'common.cu L23 · __global__', mono: true, titleSize: 10, subSize: 7.5 });
  const d2 = node({ id: 'd2', x: 280, y: 48, w: 180, h: 44, role: 'op', title: 'ncclKernelMain', sub: 'common.h L420 · SpecializedFnId', mono: true, titleSize: 10.5, subSize: 7.5 });
  const d3 = node({ id: 'd3', x: 504, y: 48, w: 180, h: 44, role: 'data', title: 'RunWorkBatch → RunWorkColl', sub: 'L303 · <Fn,T,RedOp,Algo,Proto>', mono: true, titleSize: 9, subSize: 7.5 });
  nodes.push(d1, d2, d3); body.push(d1.svg, d2.svg, d3.svg);
  [[d1, d2, 0], [d2, d3, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `k${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 120, w: 648, h: 140 }));
  body.push(cap(52, 252, 'per (algo × proto) 特化骨架（all_reduce.h · generate.py 实例化）', { size: 9 }));
  const leaves = [
    ['RING + Simple', 'runRing L14 · 两阶段逐 chunk', 'attn', 'prev/next 各 1'],
    ['TREE + Simple', 'runTreeSplit L146 · 上下行重叠', 'proj', '归约 70% / 广播 30%'],
    ['TREE + LL/LL128', 'runTreeSplit（无 treeUpDown）', 'norm', 'LL 树走 split 重叠'],
    ['NVLS + Simple', 'scatter/gather/reduce-bcast L390', 'cache', '归约搬进 NVSwitch'],
  ];
  leaves.forEach(([title, sub, role, note], i) => {
    const x = 52 + i * 160;
    const n = node({ id: `l${i}`, x, y: 156, w: 148, h: 64, role, title, sub, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    const d = `M 594 92 C 594 124, ${x + 74} 124, ${x + 74} 156`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.4, delay: 0.8 + i * 0.3 }));
    body.push(arrowHead(x + 74, 156, 0, 1));
    body.push(cap(x + 74, 236, note, { mono: true, size: 7.5, anchor: 'middle' }));
  });

  body.push(cap(36, 282, '函数指针表 ncclDevFuncTable 按 (coll × redop × type × algo × proto) 索引 —— kernel 数爆炸式特化，按架构/精度裁剪', { size: 10 }));
  body.push(cap(36, 300, 'runTreeSplit：Simple 取 nthreads/2；LL/LL128 取 70% 归约 / 30% 广播 —— 收 3 路归约比发 3 路更耗计算（注释逐字）', { size: 10 }));
  body.push(cap(36, 318, '与 CUTLASS manifest 同构（5.2 §9）：组合成本前置编译期，热路径只留一次查表；channel ↔ CTA 一一对应（blockIdx.x）', { size: 10 }));

  return scene(720, 334, body, { label: 'device dispatch 链：ncclDevKernel_Generic 经 ncclKernelMain 与 RunWorkColl 按算法协议分到 runRing runTreeSplit NVLS 四骨架，编译期特化运行期查表零分支' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/01-collective-primitives',
    afterHeading: /^## 3\. /,
    title: '八原语消息量账：均衡四与 n 解耦（AR 1.75 GB）vs root 四热点',
    svg: primitiveVolume,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/01-collective-primitives',
    replacesBlock: 0,
    title: 'ring vs a2a 流量模式：邻居均匀负载 vs 完全图单跳直达',
    svg: ringVsA2a,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/02-ring-algorithms',
    replacesBlock: 0,
    title: 'ring 两阶段时序：scatter-reduce + all-gather 各 n-1 步 = 2(n-1)/n·M',
    svg: ringTwoPhase,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/02-ring-algorithms',
    replacesBlock: 1,
    title: '下界证明：归约树 + 广播树不相交 → 总流量 ≥ 2(n-1)M，ring 达界',
    svg: lowerBoundTrees,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/02-ring-algorithms',
    replacesBlock: 2,
    title: '逻辑环 → 物理拓扑：每跳 = GPU→NVSwitch→GPU（非阻塞任意排列）',
    svg: ringPhysical,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/02-ring-algorithms',
    replacesBlock: 3,
    title: 'chunking 流水：填充期后 n 链路满载；多 ring 带宽项 ÷ k',
    svg: ringPipeline,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/03-tree-algorithms',
    replacesBlock: 0,
    title: '双二叉树（12-rank mirror）：内部节点互补，每 rank ≤ 3D+D = 2M',
    svg: doubleBinaryTree,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/03-tree-algorithms',
    afterHeading: /^## 4\. /,
    title: 'ring/tree 交叉 M* ≈ 1.83 MB：实测 β_eff 差才有交叉（理想模型无）',
    svg: ringTreeCrossover,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/04-nccl-implementation',
    replacesBlock: 0,
    title: 'ncclCommInitRank 全流程：bootstrap + 拓扑/路径/图搜索/transport/devComm',
    svg: ncclInitFlow,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/04-nccl-implementation',
    afterHeading: /^## 4\. /,
    title: 'transport 四件套 + NVLS 独立：p2p/shm/net/collNet 按拓扑选',
    svg: transportArchitecture,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/04-nccl-implementation',
    replacesBlock: 1,
    title: 'proxy 线程：GDR 控制面零拷贝 vs 非 GDR 数据面 D2H（三计数器 8 槽）',
    svg: proxyGdr,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/04-nccl-implementation',
    replacesBlock: 2,
    title: 'LL/LL128/Simple line 布局：50% / 93.75% / →100% 效率换延迟',
    svg: protocolLines,
  },
  {
    dim: 'gpu',
    slug: '08-nccl-collectives/04-nccl-implementation',
    replacesBlock: 3,
    title: 'device dispatch：编译期特化 + 函数指针表，运行期零分支',
    svg: deviceDispatch,
  },
];