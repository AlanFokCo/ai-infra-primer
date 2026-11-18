// Part 1（01-foundations）场景图：替换各章 block 0 的 ASCII 网格
// 规范见 STYLE.md；组件库见 figure-kit.mjs。
// 注意：kit 的 edge 脉冲引用 url(#<uid>-pg) 渐变，这里每个图自建同名 defs 保证解析。

import { scene, node, edge, zone, cap, bars } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/直线通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 1.1 §4 — teacher forcing：T 个位置一次前向并行打分（block 0）
 * 原文：输入 <bos> w1..w4 → 一次前向 → q1..q5（每个 ∈ Δ^(V−1)），
 * 监督目标 = 输入右移一位（w1..w4 <eos>），逐位置 −log q_t(w_t) 取平均 = 式 (4)
 * ============================================================ */
function teacherForcing() {
  const UID = 'tf';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const CX = [138, 262, 386, 510, 634]; // 5 列中心
  const CW = 84; // 卡宽
  const colX = (i) => CX[i] - CW / 2;

  body.push(cap(96, 24, 'teacher forcing：条件一律用数据真实 token，T 个位置一次前向并行打分', { size: 12, color: 'var(--ink-soft)' }));
  CX.forEach((cx, i) => body.push(cap(cx, 48, `t = ${i + 1}`, { mono: true, size: 10.5, anchor: 'middle' })));

  // 行标签（左缘）
  body.push(cap(88, 82, '输入', { size: 11, anchor: 'end' }));
  body.push(cap(88, 221, '输出', { size: 11, anchor: 'end' }));
  body.push(cap(88, 235, 'q_t ∈ Δ^(V−1)', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(88, 278, '监督目标', { size: 11, anchor: 'end' }));
  body.push(cap(88, 292, '= 输入右移一位', { size: 9.5, anchor: 'end' }));
  body.push(cap(88, 330, '逐位置 loss', { size: 10, anchor: 'end' }));

  // 输入行：<bos> 为边界 token（cache 琥珀），w1..w4 为数据 token
  const inputs = ['<bos>', 'w₁', 'w₂', 'w₃', 'w₄'].map((t, i) => {
    const n = node({ id: `in${i}`, x: colX(i), y: 56, w: CW, h: 42, role: i === 0 ? 'cache' : 'data', title: t, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // f_θ：一次前向（宽节点横贯 5 列）
  const fth = node({ id: 'fth', x: 96, y: 118, w: 580, h: 52, role: 'op', title: 'f_θ：一次前向', sub: '5 个位置并行打分 · batch/序列维铺开', mono: true });
  nodes.push(fth); body.push(fth.svg);

  // 输出行 q_t（proj 蓝 = 输出头分布）
  const qs = ['q₁', 'q₂', 'q₃', 'q₄', 'q₅'].map((t, i) => {
    const n = node({ id: `q${i}`, x: colX(i), y: 204, w: CW, h: 42, role: 'proj', title: t, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // 输入 → f_θ → q：5 条并行竖流，脉冲同发（并行的可视化）
  inputs.forEach((n, i) => {
    body.push(edge(n, [CX[i], 118], { nodes, portFrom: n.bottom, pulse: true, dur: 1.6, delay: i * 0.18, uid: UID, eid: `a${i}`, arrow: false }));
    body.push(edge([CX[i], 170], qs[i], { nodes, portTo: qs[i].top, pulse: true, dur: 1.6, delay: 0.5 + i * 0.18, uid: UID, eid: `b${i}`, arrow: false }));
  });

  // 监督目标行：w1..w4（与输入行错位一列 = 右移），<eos> 边界
  const targets = ['w₁', 'w₂', 'w₃', 'w₄', '<eos>'].map((t, i) => {
    const n = node({ id: `tg${i}`, x: colX(i), y: 260, w: CW, h: 32, role: i === 4 ? 'cache' : 'data', title: t, mono: true, titleSize: 12 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  // q_t 与目标 w_t 配对（列内短边，不跨元素）
  qs.forEach((q, i) => body.push(edge(q, targets[i], { nodes, portFrom: q.bottom, portTo: targets[i].top, uid: UID, eid: `p${i}` })));

  // 逐位置 loss（朱砂小字）→ 汇流括号 → 式 (4)
  const lossY = 326;
  CX.forEach((cx, i) => {
    body.push(cap(cx, lossY, `−log q${['₁', '₂', '₃', '₄', '₅'][i]}(${['w₁', 'w₂', 'w₃', 'w₄', '<eos>'][i]})`, { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
    body.push(`<line class="dia-edge" x1="${cx}" y1="${lossY + 6}" x2="${cx}" y2="${lossY + 16}"/>`);
  });
  body.push(`<line class="dia-edge strong" x1="${CX[0]}" y1="${lossY + 16}" x2="${CX[4]}" y2="${lossY + 16}"/>`);
  body.push(`<line class="dia-edge strong" x1="386" y1="${lossY + 16}" x2="386" y2="${lossY + 26}"/>`);
  const nll = node({ id: 'nll', x: 256, y: lossY + 26, w: 260, h: 40, role: 'attn', title: 'L = (1/T) Σ_t −log q_t(w_t)', sub: '式 (4)：T 个分类问题取平均', mono: true, titleSize: 12 });
  nodes.push(nll); body.push(nll.svg);
  body.push(cap(96, lossY + 88, '训练并行（本图，T 个分布一次出）⇄ 生成串行（每 token 一次前向）—— 2.4 KV cache 的根源', { size: 11 }));

  return scene(720, 448, body, { label: 'teacher forcing：输入右移一位对齐监督目标，T 个位置一次前向并行' }).svg();
}

/* ============================================================
 * 1.2 §7 — 单层 FLOPs 走读：attn 子层 + FFN 子层（block 0）
 * 原文（Llama-3.1-8B，per token）：QKV 50.3M · QK^T/PV 2Td · O proj 33.6M ·
 * FFN 352.3M（层内 81%）；层计 4.36×10^8 × L=32 = 1.396×10^10 = 2N（式 2 闭环）
 * ============================================================ */
function layerFlopsFlow() {
  const UID = 'ac';
  const body = [pulseDefs(UID)];
  const nodes = [];

  // 残差 stream 主干（左缘竖线）
  const TX = 72;
  const entry = node({ id: 'x0', x: 26, y: 26, w: 92, h: 40, role: 'data', title: 'x (4096)', mono: true, titleSize: 12.5 });
  nodes.push(entry); body.push(entry.svg);
  body.push(`<line class="dia-edge strong" x1="${TX}" y1="66" x2="${TX}" y2="316"/>`);
  body.push(pathPulse(`M ${TX} 66 V 316`, { dur: 4.5, delay: 0, uid: UID, r: 3.6 }));
  [78, 238].forEach((y) => body.push(`<circle cx="${TX}" cy="${y}" r="2.6" fill="var(--ink-soft)"/>`));

  // ---- 子层 1：attention ----
  body.push(zone({ x: 116, y: 32, w: 560, h: 92 }));
  body.push(cap(128, 50, 'attention 子层', { size: 10.5 }));
  const rn1 = node({ id: 'rn1', x: 128, y: 56, w: 92, h: 44, role: 'norm', title: 'RMSNorm', titleSize: 12.5 });
  const qkv = node({ id: 'qkv', x: 244, y: 56, w: 156, h: 44, role: 'proj', title: 'QKV proj', sub: 'Q4096 K1024 V1024', mono: true, titleSize: 12.5 });
  const att = node({ id: 'att', x: 424, y: 56, w: 120, h: 44, role: 'attn', title: 'QK^T · PV', sub: 'causal', mono: true, titleSize: 12.5 });
  const opr = node({ id: 'opr', x: 568, y: 56, w: 96, h: 44, role: 'proj', title: 'O proj', sub: '2d²', mono: true, titleSize: 12.5 });
  [rn1, qkv, att, opr].forEach((n) => { nodes.push(n); body.push(n.svg); });
  // FLOPs 小账（组件正下方）
  body.push(cap(322, 116, '50.3M', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(484, 116, '2Td', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(616, 116, '33.6M', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // ---- 子层 2：FFN（SwiGLU 三矩阵）----
  body.push(zone({ x: 116, y: 192, w: 536, h: 92 }));
  body.push(cap(128, 210, 'FFN 子层（SwiGLU 三矩阵）', { size: 10.5 }));
  const rn2 = node({ id: 'rn2', x: 128, y: 216, w: 92, h: 44, role: 'norm', title: 'RMSNorm', titleSize: 12.5 });
  const upg = node({ id: 'upg', x: 244, y: 216, w: 150, h: 44, role: 'proj', title: 'up / gate', sub: '→ 14336 ×2', mono: true, titleSize: 12.5 });
  const sil = node({ id: 'sil', x: 418, y: 216, w: 84, h: 44, role: 'op', title: 'SiLU ⊙', mono: true, titleSize: 12.5 });
  const dwn = node({ id: 'dwn', x: 526, y: 216, w: 110, h: 44, role: 'proj', title: 'down', sub: '→ 4096', mono: true, titleSize: 12.5 });
  [rn2, upg, sil, dwn].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(440, 276, '352.3M = 6·d·d_ff', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 读边（tap → 子层）与子层内流（脉冲）
  body.push(edge([TX, 78], rn1, { nodes, portTo: rn1.left, uid: UID, eid: 't1', arrow: false }));
  body.push(edge(rn1, qkv, { nodes, pulse: true, dur: 1.4, delay: 0.15, uid: UID, eid: 'a1' }));
  body.push(edge(qkv, att, { nodes, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'a2' }));
  body.push(edge(att, opr, { nodes, pulse: true, dur: 1.4, delay: 0.65, uid: UID, eid: 'a3' }));
  body.push(edge([TX, 238], rn2, { nodes, portTo: rn2.left, uid: UID, eid: 't2', arrow: false }));
  body.push(edge(rn2, upg, { nodes, pulse: true, dur: 1.4, delay: 1.1, uid: UID, eid: 'f1' }));
  body.push(edge(upg, sil, { nodes, pulse: true, dur: 1.4, delay: 1.35, uid: UID, eid: 'f2' }));
  body.push(edge(sil, dwn, { nodes, pulse: true, dur: 1.4, delay: 1.6, uid: UID, eid: 'f3' }));

  // 写回：子层输出 → ⊕（L 形回流通道，避让组件正下方的小账）
  const add = (cx, cy) =>
    `<circle cx="${cx}" cy="${cy}" r="9" fill="var(--paper)" stroke="var(--ink-soft)" stroke-width="1.3"/>` +
    `<text class="dia-mono" x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" fill="var(--ink)">+</text>`;
  const r1 = `M 616 100 V 134 H ${TX + 9}`;
  const r2 = `M 581 260 V 292 H ${TX + 9}`;
  body.push(`<path class="dia-edge" d="${r1}"/>`);
  body.push(`<path class="dia-edge" d="${r2}"/>`);
  body.push(pathPulse(r1, { dur: 2.2, delay: 0.9, uid: UID }));
  body.push(pathPulse(r2, { dur: 2.2, delay: 1.85, uid: UID }));
  body.push(add(TX, 134), add(TX, 292));
  body.push(cap(TX - 14, 136, '残差写回', { size: 9.5, anchor: 'end' }));

  // 主干收尾：下一层
  body.push(`<polygon points="${TX},322 ${TX - 5},312 ${TX + 5},312" fill="var(--ink-soft)"/>`);
  body.push(cap(TX, 340, '下一层 ×32', { size: 10.5, anchor: 'middle' }));

  // 层内构成条（T=4096，attention 的 2Td 另计）：小段不写内联标签，统一图例行说明
  body.push(cap(196, 312, '层内 FLOPs 构成 @ T = 4096（attention 的 2Td 项另计）', { mono: true, size: 10.5 }));
  body.push(bars({
    x: 196, y: 322, h: 16, unit: 1.05,
    segs: [
      { w: 50.3, tone: 'soft' },
      { w: 33.6, tone: 'soft' },
      { w: 352.3, tone: 'seal' },
    ],
  }));
  body.push(cap(196, 356, 'QKV 50.3M · O proj 33.6M · FFN 352.3M（层内 81%）', { mono: true, size: 10 }));
  body.push(cap(36, 384, '层计 436M（不含 2Td）× L=32 = 1.396×10^10 = 2N —— 式 (2) 闭环', { mono: true, size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(36, 402, 'T 4096 → 131072 时 2Td：33.6M → 1.07×10^9，反超其余三行之和（§6）', { size: 10.5 }));

  return scene(720, 414, body, { label: '单层 shape 走读：attention 与 FFN 两个子层各经 RMSNorm 读、残差写回，FLOPs 逐组件标注' }).svg();
}

/* ============================================================
 * 1.3 §2–§3 — log-log 下的幂律切片与 IsoFLOP 剖面（block 0）
 * 左：式 (2) 两个切片 E + A/N^α、E + B/D^β 弯向地板 E（E=1.69, A=406.4, α=0.34, B=410.7, β=0.28）
 * 右：固定 C 的剖面 g(N) = L(N, C/6N)，谷底 = compute-optimal（式 5）
 * 注：ASCII 原图臂注记与式 (3)–(5) 相反，本图按数学标注（左臂 A/N^α、右臂 B/D^β）
 * ============================================================ */
function logLogScaling() {
  const UID = 'sl';
  const body = [pulseDefs(UID)];
  const E = 1.69, A = 406.4, AL = 0.34, B = 410.7, BE = 0.28;

  // 共享 y 映射：log10 L ∈ [0.18, 0.62] → y ∈ [228, 44]（地板 E 抬到 208，标签有位）
  const yOf = (L) => 228 - ((Math.log10(L) - 0.18) / 0.44) * 184;
  // 面板 A 的 x 映射：log10 x ∈ [7.5, 11.5] → [76, 340]
  const axA = (lx) => 76 + ((lx - 7.5) / 4) * 264;
  // 面板 B 的 x 映射：log10 N ∈ [7, 12] → [424, 700]
  const axB = (ln) => 424 + ((ln - 7) / 5) * 276;

  const sample = (f, lo, hi, step, ax) => {
    const pts = [];
    for (let v = lo; v <= hi + 1e-9; v += step) {
      const y = Math.max(44, yOf(f(v)));
      pts.push(`${ax(v).toFixed(1)} ${y.toFixed(1)}`);
    }
    return 'M' + pts.join(' L');
  };

  /* ---- 面板 A（左）：两条切片 ---- */
  body.push(cap(76, 26, '① 式 (2) 的切片：另一项充分大时，log-log 下弯向地板', { size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(76, 38, 'log L', { mono: true, size: 10.5 }));
  // 坐标轴
  body.push(`<line class="dia-edge" x1="72" y1="44" x2="72" y2="228"/>`);
  body.push(`<line class="dia-edge" x1="72" y1="228" x2="344" y2="228"/>`);
  body.push(`<polygon points="72,40 67.5,48 76.5,48" fill="var(--ink-soft)"/>`);
  body.push(`<polygon points="348,228 340,223.5 340,232.5" fill="var(--ink-soft)"/>`);
  // 地板 E（标签带纸面光晕压在虚线上）
  body.push(`<line x1="72" y1="${yOf(E).toFixed(1)}" x2="344" y2="${yOf(E).toFixed(1)}" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 4" opacity="0.75"/>`);
  body.push(`<text x="80" y="${(yOf(E) + 13).toFixed(1)}" font-size="10.5" fill="var(--seal)" style="paint-order:stroke;stroke:var(--paper-raised);stroke-width:3.5">E = 1.69：不可约地板（语言本身的熵）</text>`);
  // 切片曲线（真实数值采样）
  const c1 = sample((v) => E + A * Math.pow(10, -AL * v), 7.5, 11.5, 0.1, axA);
  const c2 = sample((v) => E + B * Math.pow(10, -BE * v), 8.0, 11.5, 0.1, axA);
  body.push(`<path class="dia-edge strong" d="${c1}"/>`);
  body.push(`<path class="dia-edge strong" d="${c2}" style="stroke: rgb(168 50 40 / 0.55)"/>`);
  body.push(pathPulse(c1, { dur: 4.2, delay: 0.2, uid: UID, r: 3.4 }));
  body.push(pathPulse(c2, { dur: 4.2, delay: 1.4, uid: UID, r: 3.4 }));
  // 曲线标注（光晕压线，经典论文贴线标注）
  const halo = 'paint-order:stroke;stroke:var(--paper-raised);stroke-width:3.5';
  body.push(`<text x="250" y="146" font-size="10.5" font-family="IBM Plex Mono, monospace" text-anchor="middle" fill="var(--seal-deep)" style="${halo}">E + B/D^β（N 充分大）</text>`);
  body.push(`<text x="200" y="186" font-size="10.5" font-family="IBM Plex Mono, monospace" text-anchor="middle" fill="var(--ink-soft)" style="${halo}">E + A/N^α（D 充分大）</text>`);
  body.push(cap(208, 246, 'log x（x = N 或 D）', { mono: true, size: 10.5, anchor: 'middle' }));

  // 面板分隔
  body.push(`<line x1="384" y1="36" x2="384" y2="240" stroke="var(--rule-soft)" stroke-width="1"/>`);

  /* ---- 面板 B（右）：固定 C 的 IsoFLOP 剖面 ---- */
  body.push(cap(424, 26, '② 固定 C：g(N) = L(N, C/6N) 的剖面', { size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(424, 38, 'log L', { mono: true, size: 10.5 }));
  body.push(`<line class="dia-edge" x1="424" y1="44" x2="424" y2="228"/>`);
  body.push(`<line class="dia-edge" x1="424" y1="228" x2="700" y2="228"/>`);
  body.push(`<polygon points="424,40 419.5,48 428.5,48" fill="var(--ink-soft)"/>`);
  body.push(`<polygon points="704,228 696,223.5 696,232.5" fill="var(--ink-soft)"/>`);
  // 地板 E（右图同一条）
  body.push(`<line x1="424" y1="${yOf(E).toFixed(1)}" x2="700" y2="${yOf(E).toFixed(1)}" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 4" opacity="0.45"/>`);
  body.push(cap(696, 222, 'E', { mono: true, size: 10, anchor: 'end', color: 'var(--seal)' }));

  // g(N)：取 C 使谷底落在 log10 N ≈ 9.5（数值按式 (2) 实算）
  const C6 = Math.pow(10, 21.06); // C/6
  const g = (ln) => E + A * Math.pow(10, -AL * ln) + B * Math.pow(10, BE * (ln - 21.06));
  const uPath = sample(g, 7, 12, 0.08, axB);
  body.push(`<path class="dia-edge strong" d="${uPath}"/>`);
  body.push(pathPulse(uPath, { dur: 4.6, delay: 0.4, uid: UID, r: 3.4 }));

  // 谷底（数值解）
  let nv = 7, gv = Infinity;
  for (let ln = 7; ln <= 12; ln += 0.005) { const v = g(ln); if (v < gv) { gv = v; nv = ln; } }
  const vx = axB(nv), vy = yOf(gv);
  body.push(`<line x1="${vx.toFixed(1)}" y1="${(vy + 6).toFixed(1)}" x2="${vx.toFixed(1)}" y2="228" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>`);
  body.push(`<circle class="sig-diag" style="--i:1" cx="${vx.toFixed(1)}" cy="${vy.toFixed(1)}" r="9" fill="none" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(`<circle cx="${vx.toFixed(1)}" cy="${vy.toFixed(1)}" r="4.5" fill="var(--seal)"/>`);
  body.push(`<text x="${(vx + 10).toFixed(1)}" y="223" font-size="10" font-family="IBM Plex Mono, monospace" fill="var(--seal-deep)" style="paint-order:stroke;stroke:var(--paper-raised);stroke-width:3.5">N_opt</text>`);

  // 两臂注记（含渐近斜率），避开曲线本体
  body.push(cap(432, 76, 'N 太小：A/N^α 涨（斜率 −α = −0.34）', { size: 10.5 }));
  body.push(cap(698, 100, 'N 太大（D 太小）：B/D^β 涨（+β = +0.28）', { size: 10.5, anchor: 'end' }));

  body.push(cap(562, 246, 'log N（此剖面 log D = log(C/6) − log N）', { mono: true, size: 10.5, anchor: 'middle' }));
  body.push(cap(424, 264, '谷底 = compute-optimal（式 5）', { size: 10.5, color: 'var(--seal-deep)' }));
  body.push(cap(424, 279, 'N_opt = G·(C/6)^0.45，D_opt = G⁻¹·(C/6)^0.55', { mono: true, size: 10, color: 'var(--ink-faint)' }));

  return scene(720, 292, body, { label: 'log-log 坐标：左为式 (2) 两条切片弯向地板 E，右为固定算力下 loss 对 log N 的 U 形剖面' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '01-foundations/01-formulation',
    replacesBlock: 0,
    title: 'teacher forcing：一次前向，T 个位置并行',
    svg: teacherForcing,
  },
  {
    slug: '01-foundations/02-compute-accounting',
    replacesBlock: 0,
    title: '单层 FLOPs 走读：两个子层与残差写回',
    svg: layerFlopsFlow,
  },
  {
    slug: '01-foundations/03-scaling-laws',
    replacesBlock: 0,
    title: 'log-log 下的幂律切片与 IsoFLOP 剖面',
    svg: logLogScaling,
  },
];
