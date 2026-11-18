// training Part 5（05-model-parallelism）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；脉冲渐变用页面级 PULSE_GRAD_ID（勿自建 per-scene defs）。
// 数字与机制措辞全部来自章节原文（5.1 式 (1)-(11)、5.2 式 (1)(2)、5.3 式 (3)-(10)、5.4 式 (1)）。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 5.1 §2 — 列切 / 行切两条通信规则（block 0）
 * 原文 t=2 走读：列切前向无通信、反向 dX = Σ G_p A_pᵀ = 1 次 AR；
 * 行切前向 Y = Σ X_p A_p = 1 次 AR、反向全本地
 * ============================================================ */
function colVsRow() {
  const UID = 'tr51a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'Y = XA 只有两种切法；通信落点由 matmul 反向（1.2 式 2）直接决定', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // ---- 左：列切 ----
  body.push(zone({ x: 32, y: 44, w: 316, h: 236, label: '列切（column parallel）' }));
  // 切分字形：X 全量 ×2 + A 竖劈
  body.push(`<rect class="dia-mx-on" x="60" y="76" width="26" height="40" rx="3"/>`);
  body.push(cap(73, 128, 'X [m,k]', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(73, 142, '两卡全量复制', { size: 9, anchor: 'middle' }));
  body.push(cap(100, 96, '×', { size: 13, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(`<rect class="dia-mx-hot" x="116" y="76" width="20" height="40" rx="3"/>`);
  body.push(`<rect class="dia-mx-warm" x="138" y="76" width="20" height="40" rx="3"/>`);
  body.push(cap(137, 128, 'A₀│A₁ [k, n/2]', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(137, 142, '按列分块', { size: 9, anchor: 'middle' }));

  const cr0 = node({ id: 'cr0', x: 200, y: 68, w: 134, h: 34, role: 'proj', title: 'rank0：Y₀ = X·A₀', sub: '[m, n/2]', mono: true, titleSize: 10.5, subSize: 9.5 });
  const cr1 = node({ id: 'cr1', x: 200, y: 110, w: 134, h: 34, role: 'proj', title: 'rank1：Y₁ = X·A₁', sub: '[m, n/2]', mono: true, titleSize: 10.5, subSize: 9.5 });
  nodes.push(cr0, cr1); body.push(cr0.svg, cr1.svg);
  body.push(edge([158, 96], cr0, { nodes, portTo: cr0.left, pulse: true, dur: 1.6, delay: 0.2, uid: UID, eid: 0, arrow: false }));
  body.push(edge([158, 96], cr1, { nodes, portTo: cr1.left, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 1, arrow: false }));

  const cF = node({ id: 'cF', x: 52, y: 168, w: 282, h: 40, role: 'norm', title: '前向：无通信', sub: 'Y = [Y₀|Y₁] 逻辑拼接——下游逐位置 ⇒ 永不物化', titleSize: 11.5, subSize: 9.5 });
  const cB = node({ id: 'cB', x: 52, y: 220, w: 282, h: 44, role: 'attn', title: '反向：1 次 all-reduce', sub: 'dX = G₀A₀ᵀ + G₁A₁ᵀ · 消息 [m,k]', mono: true, titleSize: 11.5, subSize: 9.5 });
  nodes.push(cF, cB); body.push(cF.svg, cB.svg);
  body.push(pathPulse('M 334 242 L 346 242', { dur: 1.4, delay: 0.6 }));

  // ---- 右：行切 ----
  body.push(zone({ x: 372, y: 44, w: 316, h: 236, label: '行切（row parallel）' }));
  body.push(`<rect class="dia-mx-hot" x="392" y="72" width="44" height="18" rx="3"/>`);
  body.push(`<rect class="dia-mx-warm" x="392" y="92" width="44" height="18" rx="3"/>`);
  body.push(cap(414, 126, 'A₀/A₁ [k/2,n]', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(414, 140, '按行分块', { size: 9, anchor: 'middle' }));
  body.push(cap(452, 96, '×', { size: 13, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(`<rect class="dia-mx-hot" x="466" y="72" width="13" height="40" rx="3"/>`);
  body.push(`<rect class="dia-mx-warm" x="481" y="72" width="13" height="40" rx="3"/>`);
  body.push(cap(508, 126, 'X₀│X₁ [m,k/2]', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(508, 140, '沿收缩维同步切', { size: 9, anchor: 'middle' }));

  const rr0 = node({ id: 'rr0', x: 548, y: 68, w: 130, h: 34, role: 'proj', title: 'rank0：X₀·A₀', sub: '部分和 [m, n]', mono: true, titleSize: 10.5, subSize: 9.5 });
  const rr1 = node({ id: 'rr1', x: 548, y: 110, w: 130, h: 34, role: 'proj', title: 'rank1：X₁·A₁', sub: '部分和 [m, n]', mono: true, titleSize: 10.5, subSize: 9.5 });
  nodes.push(rr0, rr1); body.push(rr0.svg, rr1.svg);
  body.push(edge([522, 96], rr0, { nodes, portTo: rr0.left, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 2, arrow: false }));
  body.push(edge([522, 96], rr1, { nodes, portTo: rr1.left, pulse: true, dur: 1.6, delay: 0.5, uid: UID, eid: 3, arrow: false }));

  const rF = node({ id: 'rF', x: 392, y: 168, w: 282, h: 40, role: 'attn', title: '前向：1 次 all-reduce', sub: 'Y = X₀A₀ + X₁A₁ · 消息 [m,n]', mono: true, titleSize: 11.5, subSize: 9.5 });
  const rB = node({ id: 'rB', x: 392, y: 220, w: 282, h: 44, role: 'norm', title: '反向：无通信', sub: 'dX_p = G·A_pᵀ、dA_p = X_pᵀ·G 全本地', mono: true, titleSize: 11.5, subSize: 9.5 });
  nodes.push(rF, rB); body.push(rF.svg, rB.svg);

  body.push(cap(36, 304, '两式传的都不是参数，而是激活或其梯度（[m,n] / [m,k] 的逐 token 张量）——「TP 通信搬激活」的出处', { size: 11 }));
  body.push(cap(36, 324, '边界张量始终在每卡完整：X（列切输入）或 G（行切下游梯度）必有一方复制——结构性冗余，5.2 的 SP 对象', { size: 10.5 }));

  return scene(720, 340, body, { label: '列切与行切对照：列切反向一次 all-reduce，行切前向一次 all-reduce' }).svg();
}

/* ============================================================
 * 5.1 §6 — 一层 TP 全貌：f/g 位置图（block 1）
 * 原文：两支路各 f 入口 g 出口；前向 g×2、反向 f×2 = 每层 4 次 all-reduce，
 * 消息同为 [b,s,h]（式 9：2bsh bytes）
 * ============================================================ */
function tpLayerMap() {
  const UID = 'tr51b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 26, '一层 transformer 的 TP 通信位置（每卡视角）：f = 前向恒等 / 反向 AR；g = 前向 AR / 反向恒等', { size: 12, color: 'var(--ink)', weight: 600 }));
  body.push(cap(36, 44, '复制区段（残差 / RMSNorm / dropout 各卡全量 [b,s,h]）——5.2 的 SP 对象', { size: 10 }));

  const x = node({ id: 'x', x: 36, y: 142, w: 76, h: 56, role: 'data', title: 'x', sub: '[b,s,h]', mono: true, titleSize: 14 });
  nodes.push(x); body.push(x.svg);

  const branch = (y, tag, mids) => {
    const f = node({ id: `f${tag}`, x: 152, y, w: 46, h: 44, role: 'cache', title: 'f', sub: '反向 AR', mono: true, titleSize: 14, subSize: 9 });
    nodes.push(f); body.push(f.svg);
    let prev = f;
    mids.forEach((m, i) => {
      const n = node({ id: `${tag}m${i}`, x: m.x, y, w: m.w, h: 44, role: m.role, title: m.title, sub: m.sub, mono: m.mono ?? false, titleSize: m.ts ?? 11.5, subSize: 9.5 });
      nodes.push(n); body.push(n.svg);
      body.push(edge(prev, n, { nodes, pulse: true, dur: 1.7, delay: 0.15 * i + (tag === 'b' ? 0.5 : 0), uid: UID, eid: `${tag}${i}` }));
      prev = n;
    });
    const g = node({ id: `g${tag}`, x: 622, y, w: 46, h: 44, role: 'attn', title: 'g', sub: '前向 AR', mono: true, titleSize: 14, subSize: 9 });
    nodes.push(g); body.push(g.svg);
    body.push(edge(prev, g, { nodes, pulse: true, dur: 1.7, delay: 0.6 + (tag === 'b' ? 0.5 : 0), uid: UID, eid: `${tag}g` }));
    return { f, g };
  };

  // 支路 1：attention
  const b1 = branch(64, 'a', [
    { x: 218, w: 128, role: 'proj', title: 'QKV 列切', sub: '[h, 3h/t]', mono: true },
    { x: 362, w: 128, role: 'attn', title: '逐 head attention', sub: '按 head 切 · 免通信', ts: 11 },
    { x: 506, w: 100, role: 'proj', title: 'out proj 行切', sub: '[h/t, h]', mono: true, ts: 10.5 },
  ]);
  // 支路 2：MLP
  const b2 = branch(176, 'b', [
    { x: 218, w: 128, role: 'proj', title: 'W₁ 列切', sub: '[h, d_ff/t]', mono: true },
    { x: 362, w: 128, role: 'op', title: 'φ 逐位置', sub: '免通信（逐字精确）' },
    { x: 506, w: 100, role: 'proj', title: 'W₂ 行切', sub: '[d_ff/t, h]', mono: true, ts: 10.5 },
  ]);

  // x → 两个 f
  body.push(edge(x, b1.f, { nodes, portFrom: x.right, portTo: b1.f.left, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'x0', arrow: false }));
  body.push(edge(x, b2.f, { nodes, portFrom: x.right, portTo: b2.f.left, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'x1', arrow: false }));
  // 残差弧（虚线，从 x 顶绕过支路汇到 g 后）
  body.push(`<path class="dia-edge dash" d="M 74 142 C 74 30, 640 30, 676 64"/>`);
  body.push(`<path class="dia-edge dash" d="M 74 198 C 74 262, 640 262, 676 220"/>`);
  body.push(cap(690, 40, '+残差', { size: 10, anchor: 'end' }));
  body.push(cap(690, 244, '+残差', { size: 10, anchor: 'end' }));

  body.push(cap(360, 262, '前向通信 g × 2；反向通信 f × 2 —— 每层 4 次 all-reduce，消息同为 [b,s,h]', { size: 11.5, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));
  body.push(cap(360, 284, '消息量 M_msg = 2bsh bytes（式 9）；单层单 micro-batch 合计 8bsh，全模型 8bshL；每卡再乘 2(t−1)/t（式 10）', { mono: true, size: 10.5, anchor: 'middle' }));
  body.push(cap(360, 304, '405B 实例：NVLink 约 17 ms 对 360 ms 计算可藏；同一字节量搬上 IB ≈ 300 ms 再也藏不住 ⇒ TP ≤ 8、留节点内', { size: 10, anchor: 'middle' }));

  return scene(720, 320, body, { label: '一层 TP 全貌：attention 与 MLP 两支路各以 f 进、g 出，每层 4 次 all-reduce' }).svg();
}

/* ============================================================
 * 5.2 §2 — TP-only vs TP+SP 通信恒等（block 0，中优先级）
 * 原文：all-reduce 拆成 all-gather + reduce-scatter，字节数逐字节相等；
 * 换来复制区段激活 ÷t（sbh(34+5as/h)/t）
 * ============================================================ */
function tpVsSp() {
  const UID = 'tr52a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'SP 不引入新通信：把每次 all-reduce 沿环拆成两半，分别摆到 TP 区段的两个边界上', { size: 12, color: 'var(--ink)', weight: 600 }));

  const col = (cx, tag, rows) => {
    body.push(zone({ x: cx - 14, y: 44, w: 316, h: 236, label: tag }));
    let prev = null;
    rows.forEach((r, i) => {
      const n = node({ id: `${tag}${i}`, x: cx, y: 72 + i * 52, w: 272, h: 42, role: r.role, title: r.title, sub: r.sub, mono: r.mono ?? false, titleSize: r.ts ?? 11.5, subSize: 9.5 });
      nodes.push(n); body.push(n.svg);
      if (prev) body.push(edge(prev, n, { nodes, portFrom: prev.bottom, portTo: n.top, pulse: true, dur: 1.5, delay: 0.2 + i * 0.25, uid: UID, eid: `${tag}${i}`, arrow: false }));
      prev = n;
    });
  };

  col(56, 'TP-only（5.1）', [
    { role: 'data', title: 'x 全量复制 [b,s,h]', sub: '复制区段：norm / dropout / 残差', mono: true, ts: 11 },
    { role: 'plain', title: '入口 f：恒等', sub: '无通信', mono: true },
    { role: 'attn', title: 'TP 区段', sub: 'QKV 列切 → attn/MLP → 行切', ts: 11.5 },
    { role: 'cache', title: '出口 g：all-reduce', sub: '2(t−1)/t · M', mono: true },
  ]);
  col(392, 'TP + SP', [
    { role: 'data', title: 'x_shard [b,s/t,h]', sub: '复制区段沿序列维切 t 份', mono: true, ts: 11 },
    { role: 'proj', title: '入口 g：all-gather', sub: '(t−1)/t · M', mono: true },
    { role: 'attn', title: 'TP 区段（不变）', sub: 'QKV 列切 → attn/MLP → 行切', ts: 11.5 },
    { role: 'cache', title: '出口 ḡ：reduce-scatter', sub: '(t−1)/t · M', mono: true, ts: 11 },
  ]);

  body.push(cap(360, 312, '每方向每块：2(t−1)/t·M = (t−1)/t·M + (t−1)/t·M —— 逐字节恒等', { mono: true, size: 11, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));
  body.push(cap(360, 332, '换来：复制区段全部激活 ÷t，系数 10 那份被切走：sbh(10+24/t+5as/(ht)) → sbh(34+5as/h)/t（论文 Table 2）', { size: 10, anchor: 'middle' }));

  return scene(720, 348, body, { label: 'TP-only 与 TP+SP 对照：通信量恒等，复制区段激活除以 t' }).svg();
}

/* ============================================================
 * 5.2 §3 — Ring Attention 轮转网格（block 1）
 * 原文 4×4：Q_i × K_jV_j 命中矩阵（✓/✗）；每步 K/V 块发给下一 rank；
 * (m,ℓ,O) 逐块合并；c 步后 = 全量 causal softmax·V
 * ============================================================ */
function ringAttention() {
  const UID = 'tr52b';
  const body = [];

  body.push(cap(36, 28, 'Ring Attention（c = 4）：K/V 块沿环轮转，每段 query 在 c−1 步内见过全部 K/V 块', { size: 12.5, color: 'var(--ink)', weight: 600 }));
  body.push(cap(36, 48, '每步：发当前 K/V 块给下一 rank、收上一 rank 的块（点对点流水，与当步计算重叠）', { size: 10.5 }));

  const GX = 150, GY = 76, CWd = 118, CH = 42, CG2 = 8;
  // 列头 round
  for (let r = 0; r < 4; r++) body.push(cap(GX + r * (CWd + CG2) + CWd / 2, 84, `round ${r}`, { mono: true, size: 10, anchor: 'middle' }));
  for (let i = 0; i < 4; i++) {
    body.push(cap(GX - 14, GY + 14 + i * (CH + CG2) + CH / 2, `rank${i}`, { mono: true, size: 10.5, anchor: 'end' }));
    for (let r = 0; r < 4; r++) {
      const j = ((i - r) % 4 + 4) % 4; // 本轮持有的 KV 块
      const hit = j <= i; // causal：rank i 只需块 0..i
      const x = GX + r * (CWd + CG2), y = GY + 14 + i * (CH + CG2);
      const cls = hit ? 'dia-mx-hot sig-cell' : 'dia-mx-off';
      const sty = hit ? ` style="--i:${(0.3 + r * 0.55 + i * 0.12).toFixed(2)}"` : '';
      body.push(`<rect class="${cls}"${sty} x="${x}" y="${y}" width="${CWd}" height="${CH}" rx="7"/>`);
      const tcol = hit ? 'var(--seal-deep)' : 'var(--ink-faint)';
      body.push(cap(x + CWd / 2, y + CH / 2 + 4, `Q${i} × K${j}V${j} ${hit ? '✓' : '✗'}`, { mono: true, size: 10.5, anchor: 'middle', color: tcol, weight: hit ? 650 : undefined }));
    }
  }
  // ✗ 注记（右上留白区）
  body.push(cap(700, 44, '✗ = causal 下整块被 mask', { size: 9.5, anchor: 'end' }));
  body.push(cap(700, 60, '倾斜比 2c/(c+1) → striped / zigzag 修法（§4）', { size: 9.5, anchor: 'end' }));

  body.push(cap(36, 306, '每 rank 内：(m, ℓ, O) 逐块合并（online softmax 引理，models 4.1）——c 步后与一次性全量 causal softmax·V 逐点相同', { size: 11 }));
  body.push(cap(36, 330, '每卡每层前向发送 ≈ 4·b·s·a_kv·d_head bytes（式 2），与 c 几乎无关；显存从 O(s) 降到 O(s/c) + 一块工作缓冲', { size: 10.5 }));

  return scene(720, 348, body, { label: 'Ring Attention 轮转网格：4 rank × 4 round 的 Q×KV 命中矩阵' }).svg();
}

/* ============================================================
 * 5.3 §2 — GPipe 甘特图（block 0）：p = 4，m = 4，T = 14
 * 原文格点逐格照抄；气泡 (p−1)/(m+p−1)；峰值激活 m 份/stage
 * ============================================================ */
const GP_PIPE = [
  ['F1', 'F2', 'F3', 'F4', '', '', '', '', '', '', 'B1', 'B2', 'B3', 'B4'],
  ['', 'F1', 'F2', 'F3', 'F4', '', '', '', '', '', 'B1', 'B2', 'B3', 'B4', ''],
  ['', '', 'F1', 'F2', 'F3', 'F4', '', '', '', 'B1', 'B2', 'B3', 'B4', '', ''],
  ['', '', '', 'F1', 'F2', 'F3', 'F4', 'B1', 'B2', 'B3', 'B4', '', '', ''],
];

function gpipeGantt() {
  const UID = 'tr53a';
  const body = [];

  body.push(cap(36, 28, 'GPipe：每个 stage 先把 m 个前向全跑完，再跑 m 个反向（p = 4，m = 4，T = 14）', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  const X0 = 104, Y0 = 56, CWd = 40, CH = 30, G = 4;
  for (let t = 0; t < 14; t++) body.push(cap(X0 + t * (CWd + G) + CWd / 2, Y0 - 8, `${t + 1}`, { mono: true, size: 8.5, anchor: 'middle' }));
  GP_PIPE.forEach((row, i) => {
    body.push(cap(X0 - 14, Y0 + i * (CH + G) + CH / 2 + 4, `stage${i + 1}`, { mono: true, size: 10.5, anchor: 'end' }));
    row.forEach((cell, t) => {
      const x = X0 + t * (CWd + G), y = Y0 + i * (CH + G);
      if (!cell) {
        body.push(`<rect class="dia-mx-off" x="${x}" y="${y}" width="${CWd}" height="${CH}" rx="5"/>`);
        return;
      }
      const isF = cell[0] === 'F';
      body.push(`<rect class="${isF ? 'dia-bar dia-bar-ink' : 'dia-bar dia-bar-seal'} sig-cell" style="--i:${(0.2 + t * 0.17).toFixed(2)}" x="${x}" y="${y}" width="${CWd}" height="${CH}" rx="5"/>`);
      body.push(cap(x + CWd / 2, y + CH / 2 + 4, cell, { mono: true, size: 10.5, anchor: 'middle', color: 'var(--paper)' }));
    });
  });
  // 气泡标注（stage1 中段 6 格）
  const midX1 = X0 + 4 * (CWd + G), midX2 = X0 + 10 * (CWd + G) - G;
  body.push(`<path class="dia-edge" d="M ${midX1} ${Y0 + 4 * (CH + G) + 8} H ${midX2}" />`);
  body.push(cap((midX1 + midX2) / 2, Y0 + 4 * (CH + G) + 24, 'stage1 中段空闲 = 气泡的一部分', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 268, 'makespan T = (m+p−1)(t_f+t_b)；气泡 = (p−1)(t_f+t_b)，占比 (p−1)/(m+p−1)（式 3/5）——消气泡靠 m 对 p 过采样', { mono: true, size: 10.5 }));
  body.push(cap(36, 288, '代价：每 stage 须为全部 m 个 in-flight micro-batch 存激活（峰值 m 份）；flush 后统一参数更新，同步语义与单机累加逐步相同', { size: 10.5 }));
  body.push(cap(36, 308, 'GPipe 经验规则 m ≥ 4p 时气泡「几乎可忽略」——大 p 极限下账面仍是约 20% 空闲', { size: 10.5 }));

  return scene(720, 324, body, { label: 'GPipe 甘特图：全前向再全反向，前后各 p−1 格气泡' }).svg();
}

/* ============================================================
 * 5.3 §3 — 1F1B 甘特图（block 1）：p = 4，m = 8，T = 22
 * 原文格点逐格照抄；B 级联逐段错开一格；峰值 in-flight stage i ≤ p−i+1
 * ============================================================ */
const ONE_F1B = [
  ['F1', 'F2', 'F3', 'F4', '', '', '', 'B1', 'F5', 'B2', 'F6', 'B3', 'F7', 'B4', 'F8', 'B5', '', 'B6', '', 'B7', '', 'B8'],
  ['', 'F1', 'F2', 'F3', '', '', 'B1', 'F4', 'B2', 'F5', 'B3', 'F6', 'B4', 'F7', 'B5', 'F8', 'B6', '', 'B7', '', 'B8', ''],
  ['', '', 'F1', 'F2', '', 'B1', 'F3', 'B2', 'F4', 'B3', 'F5', 'B4', 'F6', 'B5', 'F7', 'B6', 'F8', 'B7', '', 'B8', '', '', ''],
  ['', '', '', 'F1', 'B1', 'F2', 'B2', 'F3', 'B3', 'F4', 'B4', 'F5', 'B5', 'F6', 'B6', 'F7', 'B7', 'F8', 'B8', '', '', '', ''],
];

function oneF1bGantt() {
  const UID = 'tr53b';
  const body = [];

  body.push(cap(36, 28, '1F1B：warmup（stage i 先跑 p−i 个 F）→ 稳态 1F1B 交替 → cooldown 排空（p = 4，m = 8，T = 22）', { size: 12, color: 'var(--ink)', weight: 600 }));

  const X0 = 104, Y0 = 54, CWd = 23, CH = 28, G = 3;
  for (let t = 0; t < 22; t += 2) body.push(cap(X0 + t * (CWd + G) + CWd, Y0 - 8, `${t + 1}`, { mono: true, size: 8, anchor: 'middle' }));
  const b1cells = [];
  ONE_F1B.forEach((row, i) => {
    body.push(cap(X0 - 14, Y0 + i * (CH + G) + CH / 2 + 4, `stage${i + 1}`, { mono: true, size: 10, anchor: 'end' }));
    row.forEach((cell, t) => {
      const x = X0 + t * (CWd + G), y = Y0 + i * (CH + G);
      if (!cell) {
        body.push(`<rect class="dia-mx-off" x="${x}" y="${y}" width="${CWd}" height="${CH}" rx="4"/>`);
        return;
      }
      const isF = cell[0] === 'F';
      const isB1 = cell === 'B1';
      body.push(`<rect class="${isF ? 'dia-bar dia-bar-ink' : 'dia-bar dia-bar-seal'} sig-cell" style="--i:${(0.15 + t * 0.115).toFixed(2)}" x="${x}" y="${y}" width="${CWd}" height="${CH}" rx="4"/>`);
      body.push(cap(x + CWd / 2, y + CH / 2 + 3.5, cell, { mono: true, size: 8.5, anchor: 'middle', color: 'var(--paper)' }));
      if (isB1) b1cells.push([x + CWd / 2, y + CH / 2]);
    });
  });
  // B1 级联斜线
  const d = `M ${b1cells[3][0]} ${b1cells[3][1]} L ${b1cells[2][0]} ${b1cells[2][1]} L ${b1cells[1][0]} ${b1cells[1][1]} L ${b1cells[0][0]} ${b1cells[0][1]}`;
  body.push(`<path class="dia-edge sig-dashflow" d="${d}" stroke="var(--seal-deep)" stroke-width="1.6"/>`);
  body.push(cap(b1cells[0][0] + 30, b1cells[0][1] - 24, 'B1 级联逐段错开一格', { size: 9.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 236, '气泡不变：T 与 GPipe 逐位相等（式 3）；变的是峰值激活——in-flight 从 m 份压到 stage i ≤ p−i+1 ≤ p 份（式 7）', { size: 11 }));
  body.push(cap(36, 256, '早反向 = 早释放：B(i,j) 只依赖 B(i+1,j) 与本段 F(i,j)，梯度一到就立刻反向、释放该 micro-batch 的激活', { size: 10.5 }));
  body.push(cap(36, 276, 'm ≫ p 时「气泡小」与「显存小」第一次同时可行；warmup 数 p−i 的作用是给显存封顶', { size: 10.5 }));

  return scene(720, 292, body, { label: '1F1B 甘特图：前向反向交错，B1 级联错开一格，峰值激活不超过 p 份' }).svg();
}

/* ============================================================
 * 5.3 §5 — Zero Bubble：B/W 拆分依赖 + 梯形 vs 平行四边形（block 3）
 * 原文：g_y ─┬→ B（g_x，链式不可推迟）└→ W（g_W，本地自由排序）；
 * ZB-H2 把 W 填进尾部气泡，梯形 → 平行四边形
 * ============================================================ */
function zeroBubble() {
  const UID = 'tr53c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, '把反向拆成 B 与 W：链式依赖只穿过 B，W 是调度器手里的自由填充物', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // 左：依赖图
  const gy = node({ id: 'gy', x: 40, y: 60, w: 74, h: 44, role: 'data', title: 'g_y', sub: '上游梯度', mono: true, titleSize: 14 });
  const bn = node({ id: 'bn', x: 170, y: 44, w: 168, h: 44, role: 'attn', title: 'B：g_x = (∂f/∂x)ᵀ g_y', sub: 'input-grad · 链式依赖', mono: true, titleSize: 10.5, subSize: 9.5 });
  const wn = node({ id: 'wn', x: 170, y: 112, w: 168, h: 44, role: 'op', title: 'W：g_W = (∂f/∂W)ᵀ g_y', sub: 'weight-grad · 自由排序', mono: true, titleSize: 10.5, subSize: 9.5 });
  nodes.push(gy, bn, wn); body.push(gy.svg, bn.svg, wn.svg);
  body.push(edge(gy, bn, { nodes, pulse: true, dur: 1.5, delay: 0.1, uid: UID, eid: 0 }));
  body.push(edge(gy, wn, { nodes, pulse: true, dur: 1.5, delay: 0.4, uid: UID, eid: 1, dash: true }));
  const prev = node({ id: 'prev', x: 382, y: 44, w: 120, h: 44, role: 'proj', title: '传回前一 stage', sub: '不可推迟', titleSize: 11, subSize: 9.5 });
  const buf = node({ id: 'buf', x: 382, y: 112, w: 120, h: 44, role: 'cache', title: '本地梯度缓冲', sub: '同 stage 终点', titleSize: 11, subSize: 9.5 });
  nodes.push(prev, buf); body.push(prev.svg, buf.svg);
  body.push(edge(bn, prev, { nodes, pulse: true, dur: 1.5, delay: 0.6, uid: UID, eid: 2, strong: true }));
  body.push(edge(wn, buf, { nodes, pulse: true, dur: 1.5, delay: 0.8, uid: UID, eid: 3, dash: true }));
  body.push(cap(40, 182, 'T_B + T_W = 2T_F，T_W < T_F < T_B（论文 Table 1）', { mono: true, size: 10 }));
  body.push(cap(40, 200, '打包成一个 backward() 人为加长了链上的串行段', { size: 10 }));

  // 右列：两种轮廓（梯形 vs 平行四边形）
  const mini = (ox, oy, pitch, label, rows, cols, wOf) => {
    body.push(cap(ox, oy - 10, label, { size: 10.5, color: 'var(--ink)', weight: 600 }));
    rows.forEach((band, i) => {
      for (let c = 0; c < cols; c++) {
        const inB = c >= band[0] && c <= band[1];
        const inW = wOf && wOf(i, c);
        const x = ox + c * pitch, y = oy + i * 21;
        if (inW) body.push(`<rect class="dia-bar dia-bar-seal sig-cell" style="--i:${(1.2 + i * 0.2).toFixed(2)}" x="${x}" y="${y}" width="${pitch - 3}" height="16" rx="3"/>`);
        else if (inB) body.push(`<rect class="dia-bar dia-bar-ink sig-cell" style="--i:${(0.3 + c * 0.08).toFixed(2)}" x="${x}" y="${y}" width="${pitch - 3}" height="16" rx="3"/>`);
        else body.push(`<rect class="dia-mx-off" x="${x}" y="${y}" width="${pitch - 3}" height="16" rx="3"/>`);
      }
      body.push(cap(ox - 8, oy + i * 21 + 12, `s${i + 1}`, { mono: true, size: 8.5, anchor: 'end' }));
    });
  };
  // 1F1B 梯形：14 列，s_i ▓ 占 i..i+9
  mini(436, 236, 17, '1F1B（梯形）', [[0, 9], [1, 10], [2, 11], [3, 12]], 14, null);
  // ZB-H2 平行四边形：16 列，s_i ▓ 占 i..i+9，W 固定 12..15（尾部气泡被 W 填平）
  mini(436, 352, 16, 'ZB-H2（平行四边形）· W 填入尾部气泡', [[0, 9], [1, 10], [2, 11], [3, 12]], 16, (i, c) => c >= 12);

  body.push(cap(36, 448, '等时长口径：气泡 1F1B 3(p−1)T_F → ZB-H1 = 1/3 → ZB-H2 = 0；显存价格 p·M_B → (2p−1)·M_B（≈1F1B 两倍）', { mono: true, size: 10.5 }));
  body.push(cap(36, 468, 'ZB-H2 的 optimizer 同步用 post-validation 旁路：step 前只做逐段部分归约，全局校验推迟到下一轮 warmup（§4）', { size: 10.5 }));

  return scene(720, 484, body, { label: 'Zero Bubble：B/W 拆分依赖图与 1F1B 梯形、ZB-H2 平行四边形轮廓' }).svg();
}

/* ============================================================
 * 5.4 §1 — EP 一次 MoE 前向（block 0）
 * 原文：permute 复制 → all-to-all dispatch → 本地 expert FFN（各卡不等批量）
 * → all-to-all combine 加权归约；每 token 每层 2kd 元素（bf16 4kd bytes）
 * ============================================================ */
function epForward() {
  const UID = 'tr54a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 26, 'EP 组内一次 MoE 层前向（e = 4 卡、N_e = 8 expert、每卡 2 个、top-2 示意）', { size: 12.5, color: 'var(--ink)', weight: 600 }));
  body.push(cap(36, 44, 'permute：按 (token, expert) 对把隐状态复制成 k 份（dispatch 矩阵 P；式 1）', { size: 10.5 }));

  const X4 = [46, 220, 394, 566];
  // Band 1：各卡本地 token
  const t1 = X4.map((x, j) => {
    const n = node({ id: `t${j}`, x, y: 58, w: 130, h: 40, role: 'data', title: `card${j} 本地 token`, sub: 'X [n, d]', mono: true, titleSize: 11, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  // dispatch 带
  body.push(zone({ x: 32, y: 126, w: 664, h: 42, label: 'all-to-all dispatch · X′ = P·X（按路由精确投递，无归约语义）', tone: 'seal' }));
  // Band 2：本地 expert FFN + 不等批量条
  const loads = [0.55, 0.95, 0.35, 0.75];
  const t2 = X4.map((x, j) => {
    const n = node({ id: `e${j}`, x, y: 192, w: 130, h: 44, role: 'proj', title: `card${j}：E${2 * j}·E${2 * j + 1}`, sub: '本地 expert FFN', mono: true, titleSize: 11, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    body.push(`<rect class="dia-bar dia-bar-warm sig-bar" style="--i:${(0.8 + j * 0.3).toFixed(1)}" x="${x + 10}" y="246" width="${(110 * loads[j]).toFixed(0)}" height="10" rx="4"/>`);
    body.push(cap(x + 65, 268, `批量 R${j}（不等）`, { size: 9, anchor: 'middle' }));
    return n;
  });
  // combine 带
  body.push(zone({ x: 32, y: 286, w: 664, h: 42, label: 'all-to-all combine · Y = Pᵀ(W ⊙ F(X′)) · 按 w_ti 加权归约（归约在本地）', tone: 'seal' }));
  // Band 3：输出
  const t3 = X4.map((x, j) => {
    const n = node({ id: `y${j}`, x, y: 352, w: 130, h: 38, role: 'data', title: `card${j} 输出 Y`, sub: '回 token 原位', mono: true, titleSize: 11, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // 边：token → dispatch 带 → expert → combine 带 → 输出（每张卡进出各一条代表流线，跨卡由带内 all-to-all 表达）
  for (let j = 0; j < 4; j++) {
    body.push(edge(t1[j], [86 + j * 174, 126], { nodes, portFrom: t1[j].bottom, pulse: true, dur: 1.4, delay: j * 0.2, uid: UID, eid: `d${j}`, arrow: false }));
    body.push(edge([86 + j * 174, 168], t2[j], { nodes, portTo: t2[j].top, pulse: true, dur: 1.4, delay: 0.4 + j * 0.2, uid: UID, eid: `e${j}`, arrow: false }));
    body.push(edge(t2[j], [86 + j * 174, 286], { nodes, portFrom: t2[j].bottom, pulse: true, dur: 1.4, delay: 0.8 + j * 0.2, uid: UID, eid: `c${j}`, arrow: false }));
    body.push(edge([86 + j * 174, 328], t3[j], { nodes, portTo: t3[j].top, pulse: true, dur: 1.4, delay: 1.2 + j * 0.2, uid: UID, eid: `y${j}`, arrow: false }));
  }

  body.push(cap(36, 420, '通信上界：每 token 每层 2kd 元素（bf16 合 4kd bytes）；层时间 ∝ max_j R_j —— 负载不均不是统计问题、是尾延迟问题（式 4）', { size: 11 }));
  body.push(cap(36, 440, 'k ≪ e 是 MoE 常态（V3：k = 8 对 e = 64），all-to-all 按路由投递的账严格优于 all-gather；V3 实例 229 KB × 58 层', { size: 10.5 }));

  return scene(720, 456, body, { label: 'EP 前向：permute、all-to-all dispatch、本地 expert FFN、all-to-all combine 加权归约' }).svg();
}

/* ============================================================
 * 5.3 §4 — interleaved chunk 映射（block 2，中优先级）
 * 原文：p=4/v=2 共 8 chunk；一 micro 前向蛇形路径 d1.c1→…→d4.c8
 * ============================================================ */
function interleavedMap() {
  const UID = 'tr53d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'interleaved 1F1B：每台设备持 v = 2 个不相邻层段（p = 4，共 8 chunk；原 16 层每 chunk 2 层）', { size: 12, color: 'var(--ink)', weight: 600 }));

  const rows = [
    { dev: 'device1', chunks: [{ id: 'c1', layers: '层 1–2' }, { id: 'c5', layers: '层 9–10' }] },
    { dev: 'device2', chunks: [{ id: 'c2', layers: '层 3–4' }, { id: 'c6', layers: '层 11–12' }] },
    { dev: 'device3', chunks: [{ id: 'c3', layers: '层 5–6' }, { id: 'c7', layers: '层 13–14' }] },
    { dev: 'device4', chunks: [{ id: 'c4', layers: '层 7–8' }, { id: 'c8', layers: '层 15–16' }] },
  ];
  const cardAt = {};
  rows.forEach((r, i) => {
    const y = 60 + i * 62;
    body.push(cap(96, y + 30, r.dev, { mono: true, size: 11, anchor: 'end' }));
    r.chunks.forEach((c, k) => {
      const n = node({ id: c.id, x: 130 + k * 250, y, w: 190, h: 46, role: k === 0 ? 'proj' : 'attn', title: `chunk ${c.id.slice(1)}`, sub: c.layers, mono: true, titleSize: 12.5, subSize: 10 });
      nodes.push(n); body.push(n.svg);
      cardAt[c.id] = n;
    });
  });
  // 蛇形路径：c1→c2→c3→c4→c5→c6→c7→c8
  const path = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'];
  for (let i = 0; i < path.length - 1; i++) {
    const a = cardAt[path[i]], b = cardAt[path[i + 1]];
    body.push(edge(a, b, { nodes, portFrom: i < 3 ? a.bottom : a.top, portTo: i < 3 ? b.top : b.bottom, pulse: true, dur: 1.5, delay: i * 0.3, uid: UID, eid: i }));
  }
  // c4 → c5 的折返（从 device4 回到 device1）
  body.push(edge(cardAt.c4, cardAt.c5, { nodes, portFrom: cardAt.c4.right, portTo: cardAt.c5.right, pulse: true, dur: 2.2, delay: 1.2, uid: UID, eid: 'rt' }));

  body.push(cap(36, 328, '一个 micro-batch 的前向路径两次横贯设备行：d1.c1 → … → d4.c4 → d1.c5 → … → d4.c8', { size: 11 }));
  body.push(cap(36, 348, '每 chunk 计算 ÷v、流水线级数 ×v ⇒ 气泡 ÷v（式 8）；代价：P2P 消息数与字节数 ×v；约束：m 须为 p 的整数倍', { size: 10.5 }));

  return scene(720, 364, body, { label: 'interleaved 1F1B 的设备-chunk 映射与蛇形前向路径' }).svg();
}

/* ============================================================
 * 5.5 §2 — rank 编号几何（block 0，中优先级）
 * 原文：rank = i_tp + 2·i_cp + 4·i_pp + 8·i_dp；四族组 stride 1/2/4/8；
 * herd Figure 5 实例：GPU0 与 GPU1 同 TP、GPU2 同 CP、GPU4 同 PP、GPU8 同 DP
 * ============================================================ */
function rankGeometry() {
  const UID = 'tr55a';
  const body = [];

  body.push(cap(36, 28, '16 rank 的分组几何：rank = i_tp + 2·i_cp + 4·i_pp + 8·i_dp —— 越靠左的并行维 stride 越小、组内 rank 越相邻', { size: 12, color: 'var(--ink)', weight: 600 }));

  const X0 = 60, CWd = 32, G = 4, Y0 = 60;
  for (let r = 0; r < 16; r++) {
    const x = X0 + r * (CWd + G);
    body.push(`<g class="dia-node dia-node-data"><rect x="${x}" y="${Y0}" width="${CWd}" height="30" rx="7"/><text class="dia-title dia-mono" x="${x + CWd / 2}" y="${Y0 + 20}" text-anchor="middle" font-size="11.5">${r}</text></g>`);
  }
  // 分组括弧：stride 1 TP / 2 CP / 4 PP / 8 DP（∪ 形连接组内成员底部）
  const families = [
    { name: 'TP stride 1', stride: 1, y: 108, tone: 'var(--seal-deep)' },
    { name: 'CP stride 2', stride: 2, y: 132, tone: 'rgb(24 27 32 / 0.5)' },
    { name: 'PP stride 4', stride: 4, y: 156, tone: 'rgb(24 27 32 / 0.5)' },
    { name: 'DP stride 8', stride: 8, y: 180, tone: 'var(--seal-deep)' },
  ];
  // 配对清单按 stride 定义直接枚举（rank 低位差恰为该族 stride）
  const pairs = {
    1: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    2: [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]],
    4: [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
    8: [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]],
  };
  families.forEach((f, fi) => {
    const list = pairs[f.stride];
    list.forEach(([a, b], gi) => {
      const x1 = X0 + a * (CWd + G) + CWd / 2, x2 = X0 + b * (CWd + G) + CWd / 2;
      body.push(`<path class="sig-cell" style="--i:${(0.3 + fi * 0.5 + gi * 0.06).toFixed(2)}" d="M ${x1} ${Y0 + 30} V ${f.y} H ${x2} V ${Y0 + 30}" fill="none" stroke="${f.tone}" stroke-width="1.4"/>`);
    });
    body.push(cap(650, f.y + 4, f.name, { mono: true, size: 9, color: f.tone }));
  });

  body.push(cap(36, 216, 'herd Figure 5 实例：GPU0 与 GPU1 同 TP 组、与 GPU2 同 CP 组、与 GPU4 同 PP 组、与 GPU8 同 DP 组', { size: 11 }));
  body.push(cap(36, 236, '通信越频繁的维度放越小的 stride：TP 每层 4 次 all-reduce 收进节点内 NVLink；PP 只传边界激活，放最大 stride 跨节点', { size: 10.5 }));

  return scene(720, 252, body, { label: '16 个 rank 上 TP/CP/PP/DP 四族组的 stride 分层括号' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { dim: 'training', slug: '05-model-parallelism/01-tensor-parallelism', replacesBlock: 0, title: '列切 / 行切：两条通信规则', svg: colVsRow },
  { dim: 'training', slug: '05-model-parallelism/01-tensor-parallelism', replacesBlock: 1, title: '一层 TP 全貌：f/g 位置与 4 次 all-reduce', svg: tpLayerMap },
  { dim: 'training', slug: '05-model-parallelism/02-sequence-and-context-parallelism', replacesBlock: 0, title: 'TP-only vs TP+SP：通信恒等', svg: tpVsSp },
  { dim: 'training', slug: '05-model-parallelism/02-sequence-and-context-parallelism', replacesBlock: 1, title: 'Ring Attention：K/V 轮转网格', svg: ringAttention },
  { dim: 'training', slug: '05-model-parallelism/03-pipeline-parallelism', replacesBlock: 0, title: 'GPipe：全前向再全反向', svg: gpipeGantt },
  { dim: 'training', slug: '05-model-parallelism/03-pipeline-parallelism', replacesBlock: 1, title: '1F1B：气泡不动，峰值激活 ≤ p', svg: oneF1bGantt },
  { dim: 'training', slug: '05-model-parallelism/03-pipeline-parallelism', replacesBlock: 2, title: 'interleaved：设备 × chunk 蛇形映射', svg: interleavedMap },
  { dim: 'training', slug: '05-model-parallelism/03-pipeline-parallelism', replacesBlock: 3, title: 'Zero Bubble：B/W 拆分与轮廓', svg: zeroBubble },
  { dim: 'training', slug: '05-model-parallelism/04-expert-parallelism', replacesBlock: 0, title: 'EP：一次 MoE 前向的两次 all-to-all', svg: epForward },
  { dim: 'training', slug: '05-model-parallelism/05-combining-strategies', replacesBlock: 0, title: 'rank 编号几何：四族组的 stride 分层', svg: rankGeometry },
];
