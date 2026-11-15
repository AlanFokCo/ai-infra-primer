// 核心签名图（figure-kit 场景语言）：KV cache / RoPE / mask 三格 / MoE 路由 / CLIP 矩阵
// 动画 = 机制：脉冲光点沿贝塞尔边流动、单元格填充、旋臂连续旋转——门控 .fig.in，reduced-motion 静态。
import { scene, node, edge, zone, cap, matrix, bars } from './figure-kit.mjs';

/* ============ 2.4 §4.4 — KV cache 的写与读 ============ */
export function kvCacheFill() {
  const body = [];
  const nodes = [];

  body.push(cap(34, 40, 'decode 步（串行）', { mono: true }));
  const steps = [];
  for (let i = 0; i < 4; i++) {
    const n = node({ id: `t${i}`, x: 32, y: 58 + i * 54, w: 96, h: 38, role: 'data', title: `t = Tₚ+${i + 1}`, mono: true, titleSize: 12 });
    nodes.push(n); steps.push(n);
    body.push(n.svg);
  }

  body.push(zone({ x: 196, y: 46, w: 326, h: 212, label: 'cache（每层 K,V 各一份）' }));
  const gx = 238, gy = 88, cw = 33, chh = 25, gap = 5;
  for (let t = 0; t < 7; t++) {
    body.push(cap(gx + t * (cw + gap) + cw / 2, gy - 9, String(t + 1), { mono: true, size: 10, anchor: 'middle' }));
  }
  body.push(cap(gx - 10, gy + 17, 'K', { mono: true, anchor: 'end', color: 'var(--ink-soft)' }));
  body.push(cap(gx - 10, gy + 17 + chh + gap, 'V', { mono: true, anchor: 'end', color: 'var(--ink-soft)' }));
  for (let t = 0; t < 7; t++) {
    for (let band = 0; band < 2; band++) {
      const y = gy + band * (chh + gap);
      if (t < 3) {
        body.push(`<rect class="dia-mx-off" x="${gx + t * (cw + gap)}" y="${y}" width="${cw}" height="${chh}" rx="4" style="fill:var(--paper-sunk)"/>`);
        body.push(cap(gx + t * (cw + gap) + cw / 2, y + 16, `${band ? 'v' : 'k'}${t + 1}`, { mono: true, size: 10, anchor: 'middle', color: 'var(--ink-faint)' }));
      } else {
        body.push(`<rect class="dia-mx-hot sig-cell" style="--i:${((t - 3) * 0.5 + band * 0.25).toFixed(2)}" x="${gx + t * (cw + gap)}" y="${y}" width="${cw}" height="${chh}" rx="4"/>`);
        body.push(`<text class="sig-cell-text" style="--i:${((t - 3) * 0.5 + band * 0.25 + 0.1).toFixed(2)}" font-family="IBM Plex Mono, monospace" font-size="10.5" x="${gx + t * (cw + gap) + cw / 2}" y="${y + 16}" fill="var(--seal)" text-anchor="middle">${band ? 'v' : 'k'}${t + 1}</text>`);
      }
    }
  }

  const read = node({ id: 'read', x: 560, y: 88, w: 132, h: 62, role: 'attn', title: 'score × 全 cache', sub: 's = 1..t 读出', titleSize: 12 });
  body.push(read.svg);
  body.push(cap(560, 192, 'append-only 写：每步一列', { size: 11 }));
  body.push(cap(560, 211, '读：每步全量乘（§5 的账）', { size: 11 }));

  steps.forEach((n, i) => {
    const colX = gx + (3 + i) * (cw + gap) + cw / 2;
    body.push(edge(n, [colX, gy + chh / 2], { nodes, pulse: true, dur: 2.4, delay: i * 0.5, uid: 'kv', eid: i, arrow: false }));
  });
  body.push(edge([gx + 7 * (cw + gap) + 2, gy + chh / 2], read, { nodes, portFrom: null, pulse: true, dur: 2.8, delay: 0.6, uid: 'kv', eid: 9, strong: true }));

  return scene(720, 276, body, { label: 'decode 循环中 KV cache 逐位置填充' }).svg();
}

/* ============ 2.5 §4.2 — RoPE 对偶旋转 ============ */
export function ropeRotate() {
  const body = [];
  body.push(cap(34, 36, '同一位置 m，作用在 d/2 个正交二维平面上：各转 m·θ_i，频率几何递减', { size: 12, color: 'var(--ink-soft)' }));

  const circles = [
    { cx: 130, label: 'θ₁', sub: 'i 小 · 短波长 · 局部', dur: '3s' },
    { cx: 350, label: 'θ₂', sub: '中间频率', dur: '6.5s' },
    { cx: 570, label: 'θ_d/2', sub: 'i 大 · 长波长 · 远程', dur: '13s' },
  ];
  circles.forEach((c) => {
    body.push(`<circle cx="${c.cx}" cy="150" r="72" fill="none" stroke="var(--rule)" stroke-width="1.2"/>`);
    body.push(`<line x1="${c.cx}" y1="150" x2="${c.cx + 52}" y2="150" stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 4"/>`);
    body.push(
      `<g class="sig-rotor" style="--dur:${c.dur}; transform-origin: ${c.cx}px 150px">` +
      `<line x1="${c.cx}" y1="150" x2="${c.cx + 52}" y2="${150 - 52 * 0.71}" stroke="var(--ink)" stroke-width="1.6"/>` +
      `<circle cx="${c.cx + 52}" cy="${150 - 52 * 0.71}" r="5" fill="var(--seal)"/>` +
      `</g>`
    );
    body.push(cap(c.cx, 248, c.label, { mono: true, size: 13, anchor: 'middle', color: 'var(--ink)' }));
    body.push(cap(c.cx, 266, c.sub, { size: 10.5, anchor: 'middle' }));
  });
  return scene(720, 286, body, { label: 'RoPE 二维对偶旋转' }).svg();
}

/* ============ 3.1 §1 — mask 形状即 objective 形状 ============ */
export function maskGrids() {
  const body = [];
  const N = 6, cell = 22, gap = 3;
  const grids = [
    { x: 60, title: 'CLM', sub: '下三角 · 只见左侧', fill: (i, j) => j <= i },
    { x: 296, title: 'MLM', sub: '全量 · 双向', fill: () => true },
    { x: 532, title: 'prefix-LM', sub: '前缀块内双向 + 因果', fill: (i, j) => j <= i || j < 3 },
  ];
  grids.forEach((g, gi) => {
    body.push(cap(g.x, 46, g.title, { mono: true, size: 13, color: 'var(--ink)', weight: 600 }));
    body.push(cap(g.x, 64, g.sub, { size: 10.5 }));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const on = g.fill(i, j);
        body.push(
          `<rect class="${on ? 'dia-mx-on' : 'dia-mx-off'} sig-cell" style="--i:${((i + j) * 0.06 + gi * 0.5).toFixed(2)}" x="${g.x + j * (cell + gap)}" y="${80 + i * (cell + gap)}" width="${cell}" height="${cell}" rx="4"/>`
        );
      }
    }
  });
  body.push(cap(60, 262, '行 = query t · 列 = key s；实格可见、虚格 −∞ 屏蔽', { mono: true, size: 10.5 }));
  return scene(720, 280, body, { label: '三种 objective 的 attention mask' }).svg();
}

/* ============ 4.5 §2 — MoE top-2 路由 ============ */
export function moeRouting() {
  const body = [];
  const nodes = [];
  body.push(cap(34, 40, 'tokens', { mono: true }));
  const ty = [64, 150, 236];
  const toks = ['x₁', 'x₂', 'x₃'].map((t, i) => {
    const n = node({ id: `t${i}`, x: 32, y: ty[i] - 20, w: 64, h: 40, role: 'data', title: t, mono: true, titleSize: 13 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const router = node({ id: 'router', x: 190, y: 126, w: 118, h: 64, role: 'proj', title: 'router', sub: 'softmax(W·x)', mono: true });
  nodes.push(router); body.push(router.svg);

  const ey = [46, 108, 170, 222];
  const load = [2, 1, 2, 1];
  const experts = ['E1', 'E2', 'E3', 'E4'].map((t, j) => {
    const n = node({ id: t, x: 470, y: ey[j] - 20, w: 118, h: 42, role: 'op', title: t, sub: `load ${load[j]}/3`, mono: true });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  toks.forEach((t) => body.push(edge(t, router.left, { nodes, uid: 'moe', eid: `in${t.id}`, arrow: false })));
  const picks = [[0, 2], [1, 2], [0, 3]];
  toks.forEach((t, i) => {
    picks[i].forEach((j, k) => {
      body.push(edge(router, experts[j], { nodes, portFrom: router.right, pulse: true, dur: 2.0 + i * 0.3, delay: i * 0.4 + k * 0.2, uid: 'moe', eid: `p${i}${k}` }));
    });
  });
  body.push(cap(34, 306, 'top-2：每 token 只付 2/4 个 expert 的 FLOPs，容量却按全量参数计 — 解耦即此意', { size: 11.5 }));
  return scene(720, 320, body, { label: 'MoE：token 经 router 选 top-2 expert' }).svg();
}

/* ============ 6.2 §2.1 — CLIP 双塔与相似度矩阵 ============ */
export function clipMatrix() {
  const body = [];
  const nodes = [];
  body.push(cap(40, 38, '图像塔（ViT）', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(196, 38, '文本塔（transformer）', { size: 12, color: 'var(--ink-soft)' }));
  const us = [], vs = [];
  for (let i = 0; i < 3; i++) {
    const u = node({ id: `u${i}`, x: 36, y: 56 + i * 62, w: 72, h: 44, role: 'data', title: `u${i + 1}`, sub: 'img', mono: true });
    const v = node({ id: `v${i}`, x: 196, y: 56 + i * 62, w: 86, h: 44, role: 'data', title: `v${i + 1}`, sub: 'text', mono: true });
    nodes.push(u, v); us.push(u); vs.push(v);
    body.push(u.svg, v.svg);
  }
  const mx = 420, my = 56, cell = 46, gap = 7;
  body.push(cap(mx, 42, 'S_ij = ⟨u_i, v_j⟩/τ（单位球面）', { mono: true, size: 10.5 }));
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const diag = i === j;
      body.push(
        `<rect class="${diag ? 'dia-mx-hot sig-diag' : 'dia-mx-off'}" ${diag ? `style="--i:${(i * 0.5).toFixed(1)}"` : ''} x="${mx + j * (cell + gap)}" y="${my + i * (cell + gap)}" width="${cell}" height="${cell}" rx="6"/>` +
        cap(mx + j * (cell + gap) + cell / 2, my + i * (cell + gap) + cell / 2 + 4, `S${i + 1}${j + 1}`, { mono: true, size: 11, anchor: 'middle', color: diag ? 'var(--seal)' : 'var(--ink-faint)' })
      );
    }
  }
  body.push(cap(mx, my + 3 * (cell + gap) + 22, '行方向：图 i 分类到 B 个文本；列方向对称', { size: 10.5 }));
  for (let i = 0; i < 3; i++) {
    body.push(edge(us[i], [mx + i * (cell + gap), my + i * (cell + gap) + cell / 2], { nodes, portFrom: us[i].right, pulse: true, dur: 2.6, delay: i * 0.4, uid: 'clip', eid: `u${i}`, arrow: false }));
    body.push(edge(vs[i], [mx + i * (cell + gap), my + i * (cell + gap) + cell / 2 + 10], { nodes, portFrom: vs[i].right, pulse: true, dur: 2.6, delay: i * 0.4 + 0.2, uid: 'clip', eid: `v${i}`, arrow: false }));
  }
  return scene(720, 268, body, { label: 'CLIP：双塔编码，相似度矩阵对角即正样本' }).svg();
}

/* ============ 注册表 ============ */
export const SIGNATURE_FIGURES = [
  { slug: '02-transformer-anatomy/04-causality-and-kv-cache', afterHeading: /^### 4\.4 有 cache 的 decode/, title: 'KV cache 的写与读', svg: kvCacheFill },
  { slug: '02-transformer-anatomy/05-positional-encoding', afterHeading: /^### 4\.2 2D 旋转引理/, title: 'RoPE 的二维对偶旋转', svg: ropeRotate },
  { slug: '03-objectives-and-variants/01-objectives-and-masks', afterHeading: /^## 1\. 统一框架/, title: 'mask 形状即 objective 形状', svg: maskGrids },
  { slug: '04-modern-architectures/05-moe', afterHeading: /^## 2\. Router 机制谱系/, title: 'top-k 路由与负载', svg: moeRouting },
  { slug: '06-vision-representations/02-clip-and-contrastive', afterHeading: /^### 2\.1 相似度矩阵/, title: 'batch 相似度矩阵：对角即正样本', svg: clipMatrix },
];
