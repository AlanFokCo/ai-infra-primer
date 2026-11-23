// inference Part 1（01-principles-and-accounting）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 training-part01.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-cell --i ≤ 2.6（3.5s 前点亮完毕）。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性（类选择器 > 表现属性）——
// 显式传 color 时补 inline style 提权，保证朱砂强调真正生效（不改 kit API；W7 S3-4 补齐，与 part03-09 一致）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

// 手工路径上的脉冲（L 形/回环/折线通道用 path 直写；渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

// 显式箭头三角（kit 的 marker-end 被 CSS 关闭；方向 = 脉冲 + 布局 + 箭头三重表达）
const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

/* ============================================================
 * 1.1 §1（block 0）— 请求全链路 + SLO 口径
 * ============================================================ */
function requestLifecycle() {
  const UID = 'i1a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '请求全链路：墙钟时间花在哪 —— prompt 1024 / 输出 512 · TTFT 0.5 s · TPOT 30 ms（教学口径）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 104, label: 'TTFT = t₁ − t_arrival：tokenize + 排队 + prefill（高负载下排队是主项）' }));
  const mk = (id, x, w, role, title, sub, mono = false) => {
    const n = node({ id, x, y: 72, w, h: 52, role, title, sub, mono, titleSize: 12.5, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const c1 = mk('c1', 52, 100, 'data', '客户端 HTTP', 'prompt + 采样参数');
  const a1 = mk('a1', 180, 110, 'op', 'API 协议层', '解析 · 鉴权 · 校验');
  const t1 = mk('t1', 318, 100, 'op', 'tokenize', '文本 → token ids', true);
  const q1 = mk('q1', 446, 100, 'cache', '排队', 'waiting queue', true);
  const p1 = mk('p1', 574, 100, 'attn', 'prefill', '并行前向全部 prompt');
  [[c1, a1], [a1, t1], [t1, q1], [q1, p1]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: i * 0.25, uid: UID, eid: `t${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  const ft = node({ id: 'ft', x: 574, y: 156, w: 100, h: 26, role: 'plain', title: '首 token t₁', mono: true, titleSize: 11 });
  nodes.push(ft); body.push(ft.svg);
  body.push(edge(p1, ft, { nodes, portFrom: p1.bottom, portTo: ft.top, pulse: true, dur: 1, delay: 1.0, uid: UID, eid: 'ft' }));
  body.push(arrowHead(ft.top[0], ft.top[1], 0, 1));

  body.push(zone({ x: 36, y: 196, w: 648, h: 104, label: 'decode 循环：iteration 级调度（Orca）—— 步间隔 = TBT，batch 每步重组' }));
  const mk2 = (id, x, w, role, title, sub) => {
    const n = node({ id, x, y: 228, w, h: 52, role, title, sub, titleSize: 12.5, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const d1 = mk2('d1', 574, 100, 'attn', 'decode 循环', '每序列每步 1 token');
  const d2 = mk2('d2', 418, 120, 'op', 'detokenize', '增量 · UTF-8 跨 chunk');
  const d3 = mk2('d3', 252, 130, 'data', '流式返回', 'SSE chunk 逐 token');
  const d4 = mk2('d4', 52, 130, 'data', '客户端', '逐 token 可见');
  [[d1, d2], [d2, d3], [d3, d4]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.left, portTo: b.right, pulse: true, dur: 1.2, delay: 1.3 + i * 0.25, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(b.right[0], b.right[1], -1, 0));
  });
  const loop = 'M 600 280 V 292 H 648 V 280';
  body.push(`<path class="dia-edge strong" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 1.6, delay: 2.0 }));
  body.push(arrowHead(648, 280, 0, -1));
  body.push(cap(596, 296, 'iteration 循环', { size: 8.5, anchor: 'end' }));

  body.push(edge(ft, d1, { nodes, portFrom: ft.bottom, portTo: d1.top, pulse: true, dur: 1, delay: 1.15, uid: UID, eid: 'pd' }));
  body.push(arrowHead(d1.top[0], d1.top[1], 0, 1));
  body.push(cap(612, 208, '首 token + 全量 KV cache', { size: 9.5, anchor: 'end' }));

  body.push(`<rect class="dia-bar dia-bar-seal" x="52" y="346" width="19" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-ink" x="74" y="346" width="582" height="18" rx="3"/>`);
  body.push(cap(52, 342, 'TTFT 0.5 s（3%）', { size: 9.5 }));
  body.push(cap(365, 342, 'decode 循环 511 × 30 ms = 15.33 s（97%）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(52, 384, 'E2E 分解（式 2）：E2E = 0.5 + 511 × 0.03 = 15.83 s —— 长输出请求的延迟主战场在 decode 步时', { size: 10.5 }));
  body.push(cap(52, 402, '短输出 T_out = 8：E2E = 0.71 s、TTFT 占 70% —— agent 负载（长 prompt · 短输出）以 TTFT 为 SLO 主项', { size: 10.5 }));
  body.push(cap(52, 420, 'SLO 写在百分位上（如 P99 TTFT ≤ 2 s）；均值对长尾不敏感：0.68 s 均值 vs P99 = 5 s', { size: 10 }));

  return scene(720, 436, body, { label: '请求全链路：TTFT 段与 decode 段的数据流，E2E 分解为 TTFT 加 511 个 TBT 间隔' }).svg();
}

/* ============================================================
 * 1.2 §3（block 0）— decode 一个 iteration 的 HBM 流量
 * ============================================================ */
function decodeHbmTraffic() {
  const UID = 'i1b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'decode 一个 iteration 的 HBM 流量（bf16，8B 口径）：权重读一遍共享，KV 逐序列读全量', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 300, h: 236, label: 'HBM（每步全读）' }));
  const W = node({ id: 'W', x: 56, y: 76, w: 260, h: 48, role: 'cache', title: '权重 W = 16.1 GB', sub: 'bf16 · 全 batch 共享一遍', mono: true, titleSize: 12.5, subSize: 9.5 });
  nodes.push(W); body.push(W.svg);
  const kvs = [140, 176, 224].map((y, i) => {
    const n = node({ id: `kv${i}`, x: 56, y, w: 260, h: 30, role: 'data', title: `序列 ${['1', '2', 'B'][i]} · κLT = 0.537 GB`, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(186, 218, '⋯（B 份逐序列独立）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(56, 272, 'κLT = 4096 B × 32 × T（T = 4096）', { mono: true, size: 9.5 }));

  const C = node({ id: 'C', x: 430, y: 120, w: 254, h: 84, role: 'proj', title: 'decode iteration（B 序列）', sub: 'FLOPs = B·f(T)', mono: true, titleSize: 13, subSize: 10 });
  nodes.push(C); body.push(C.svg);
  body.push(edge(W, [430, 140], { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'w' }));
  body.push(arrowHead(430, 140, 1, 0));
  body.push(cap(373, 104, '读一遍 · 共享', { size: 9.5, anchor: 'middle' }));
  kvs.forEach((k, i) => {
    body.push(edge(k, [430, 140 + i * 22], { nodes, pulse: true, dur: 1.6, delay: 0.3 + i * 0.2, uid: UID, eid: `k${i}` }));
    body.push(arrowHead(430, 140 + i * 22, 1, 0));
  });
  body.push(cap(390, 262, '读全量 · B 份独立', { size: 9.5 }));
  body.push(cap(557, 224, 'GEMM [B,4096]×[4096,·]', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(557, 240, 'attention 逐序列算', { size: 9, anchor: 'middle' }));

  body.push(cap(140, 308, 'W 16.1 GB', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(420, 308, 'B·κLT = 64 × 0.537 = 34.3 GB', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="316" width="177" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="232" y="316" width="377" height="18" rx="3"/>`);
  body.push(cap(52, 352, '合计 50.4 GB → 步时下界 ≈ 15.1 ms @ H100 3.35 TB/s（memory-bound）', { mono: true, size: 10 }));
  body.push(cap(52, 370, 'I = B·f(T)/(W + B·κLT) ≈ 21.8 ≪ ridge ≈ 295；Θ ≈ 64/0.0151 ≈ 4240 tokens/s', { mono: true, size: 10 }));
  body.push(cap(52, 388, 'B：64 → 128：步时 15.1 → 25.3 ms、吞吐 4240 → 5060（+19%）—— 凑 batch 收益递减', { size: 10.5 }));

  return scene(720, 400, body, { label: 'decode 每步 HBM 流量：权重读一遍全 batch 共享，KV cache 逐序列读全量，合计 W 加 B 份 κLT' }).svg();
}

/* ============================================================
 * 1.2 §6（block 1）— Wide-EP 布局走读（e=4、E=8、top-2 示意）
 * ============================================================ */
function wideEpDispatch() {
  const UID = 'i1c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Wide-EP 布局走读（e = 4、E = 8、top-2 示意）：每卡流量 ∝ 1/e，总流量不变', { size: 12, color: 'var(--ink-soft)' }));

  const B = node({ id: 'B', x: 250, y: 44, w: 220, h: 48, role: 'data', title: '全局 batch B', sub: 'DP 组 1..4 各 B/4 · attention 逐组本地算', titleSize: 13, subSize: 9 });
  nodes.push(B); body.push(B.svg);
  body.push(cap(52, 116, '每 token 选 k=2 → Bk 个分配对', { size: 9.5 }));
  body.push(cap(668, 116, 'all-to-all dispatch', { mono: true, size: 9.5, anchor: 'end' }));

  const ranks = [52, 214, 376, 538].map((x, i) => {
    const n = node({ id: `r${i}`, x, y: 140, w: 140, h: 64, role: 'proj', title: `rank ${i + 1} · expert ${2 * i + 1},${2 * i + 2}`, sub: '各收 Bk/E · 读 2 份权重', mono: true, titleSize: 11, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  ranks.forEach((r, i) => {
    body.push(edge(B, r, { nodes, portFrom: B.bottom, portTo: r.top, pulse: true, dur: 1.4, delay: i * 0.2, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(r.top[0], r.top[1], 0, 1));
  });

  const CB = node({ id: 'cb', x: 180, y: 252, w: 360, h: 40, role: 'op', title: 'all-to-all combine：输出发回原 DP 组加权归约', titleSize: 11.5 });
  nodes.push(CB); body.push(CB.svg);
  ranks.forEach((r, i) => {
    body.push(edge(r, [240 + i * 80, 252], { nodes, portFrom: r.bottom, pulse: true, dur: 1.4, delay: 0.9 + i * 0.15, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(240 + i * 80, 252, 0, 1));
  });

  body.push(cap(52, 320, '每卡流量 = (E/e)·W_exp·bytes·L_moe（式 13）；e 卡合计 = E·W_exp·bytes·L_moe —— 与 e 无关', { mono: true, size: 10 }));
  const rows = [
    ['e = 8', 359, '81.7 GB · 24.4 ms', 332],
    ['e = 64', 45, '10.2 GB · 3.05 ms', 350],
    ['e = 128', 22, '5.11 GB · 1.52 ms', 368],
  ];
  rows.forEach(([lab, w, val, y]) => {
    body.push(cap(52, y + 10, lab, { mono: true, size: 9.5 }));
    body.push(`<rect class="dia-bar dia-bar-${w > 100 ? 'soft' : 'seal'}" x="110" y="${y}" width="${w}" height="12" rx="3"/>`);
    body.push(cap(110 + w + 8, y + 10, val, { mono: true, size: 9.5 }));
  });
  body.push(cap(476, 360, '每卡强度 128 FLOP/byte（B=2048 fp8）', { size: 9.5 }));
  body.push(cap(476, 378, '—— 与 e 无关（命题 3）', { size: 9.5 }));
  body.push(cap(52, 396, 'V3 fp8 routed 合计 654 GB；带宽换算沿用 H100 3.35 TB/s（教学口径）—— 强度与流量是两笔账', { size: 10 }));

  return scene(720, 408, body, { label: 'Wide-EP：全局 batch 经 all-to-all dispatch 到每卡两个 expert，combine 归约回原 DP 组；每卡流量反比于 e' }).svg();
}

/* ============================================================
 * 1.3 §2（block 0）— roofline 与两阶段定位（示意，非按数值比例）
 * ============================================================ */
function roofline() {
  const UID = 'i1d';
  const body = [];

  body.push(cap(36, 24, 'roofline：可达上限 = min(F_peak, I·β) —— decode 全窗口落左斜线、prefill 落右平台（示意，非按数值比例）', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴
  body.push(`<line class="dia-edge strong" x1="90" y1="288" x2="90" y2="56"/>`);
  body.push(`<line class="dia-edge strong" x1="90" y1="288" x2="668" y2="288"/>`);
  body.push(arrowHead(90, 56, 0, -1));
  body.push(arrowHead(668, 288, 1, 0));
  body.push(cap(90, 48, '可达吞吐上限', { size: 10, anchor: 'middle' }));
  body.push(cap(668, 304, 'I（FLOP/字节）', { size: 10, anchor: 'end' }));

  // roofline：斜线 + 平台
  body.push(`<path class="dia-edge strong" d="M 90 288 L 330 96 L 668 96"/>`);
  body.push(`<line class="dia-edge dash" x1="330" y1="96" x2="330" y2="288"/>`);
  body.push(cap(330, 304, 'R = F_peak/β ≈ 295', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(500, 84, 'F_peak = 989.5 TFLOPS（BF16 dense · compute-bound 平台）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(100, 150, '斜率 = β = 3.35 TB/s', { mono: true, size: 9.5 }));
  body.push(cap(100, 166, 'memory-bound 斜线', { size: 9.5 }));

  // decode：强度被饱和上界封顶（脉冲沿斜线推进到 I_sat 即止）
  body.push(`<line class="dia-edge dash" x1="150" y1="288" x2="150" y2="240"/>`);
  body.push(cap(146, 232, 'I_sat ≈ 34', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<circle cx="102" cy="278" r="4.5" fill="var(--seal)"/>`);
  const push = 'M 90 288 L 150 240';
  body.push(pathPulse(push, { dur: 2.2, delay: 0 }));
  body.push(cap(96, 320, 'decode：I ≈ 1–34（T=4096 饱和 ≈34）—— 带宽决定', { size: 9.5 }));

  // prefill：右平台
  body.push(`<circle cx="520" cy="96" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(520, 116, 'prefill：I ≈ T_p = 4096 ≫ R —— 算力决定', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 344, '凑 batch 只把强度从 ≈1 抬向饱和上界（脉冲所示），永远跨不过 ridge：T_c ≈ 393（1.2 命题 1）', { size: 10.5 }));
  body.push(cap(36, 362, '换卡对 decode 的收益几乎全来自带宽项：H200 β = 4.8 TB/s → R ≈ 206；B200 算力带宽同比放大 → R ≈ 292', { size: 10.5 }));

  return scene(720, 372, body, { label: 'roofline：memory-bound 斜线与 compute-bound 平台，decode 强度被饱和上界封顶在 ridge 左侧，prefill 在右平台' }).svg();
}

/* ============================================================
 * 1.3 §4（afterHeading）— batch 三方权衡：X(B) 饱和 × TPOT(B) 仿射 × SLO 操作点
 * 原文：X(32)≈3225、X(64)≈4252、饱和 β/K≈6240；TPOT(32)=9.92、TPOT(64)=15.05 ms；
 * τ=10 ms → B_max≈32.5 取 32；KV 容量上限 ≈119；三卡对照 59/5949、113/11351
 * ============================================================ */
function batchTradeoff() {
  const UID = 'i1e';
  const body = [];

  body.push(cap(36, 24, 'batch 三方权衡（8B / H100，T = 4096）：X(B) 饱和、TPOT(B) 仿射增 —— 操作点被 SLO 夹出', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴（B：0..128 → x 90..660）
  body.push(`<line class="dia-edge strong" x1="90" y1="268" x2="660" y2="268"/>`);
  body.push(`<line class="dia-edge strong" x1="90" y1="268" x2="90" y2="60"/>`);
  body.push(arrowHead(660, 268, 1, 0));
  body.push(arrowHead(90, 60, 0, -1));
  // 双轴：左 token/s（X 曲线）、右 ms（TPOT 线）
  body.push(`<line class="dia-edge" x1="660" y1="268" x2="660" y2="60"/>`);
  body.push(cap(84, 64, 'token/s', { mono: true, size: 8, anchor: 'end' }));
  body.push(cap(666, 64, 'ms', { mono: true, size: 8 }));
  [[268, '0'], [196, '10'], [124, '20']].forEach(([y, t]) => {
    body.push(`<line class="dia-edge" x1="655" y1="${y}" x2="665" y2="${y}"/>`);
    body.push(cap(666, y + 3, t, { mono: true, size: 8 }));
  });
  [[90, '0'], [232, '32'], [375, '64'], [660, '128']].forEach(([x, t]) => {
    body.push(`<line class="dia-edge" x1="${x}" y1="263" x2="${x}" y2="273"/>`);
    body.push(cap(x, 284, t, { mono: true, size: 9, anchor: 'middle' }));
  });
  body.push(cap(90, 298, 'B（decode batch）→', { size: 9.5 }));

  // 饱和渐近线 + X(B) 曲线（ink）
  body.push(`<line class="dia-edge dash" x1="90" y1="93" x2="660" y2="93"/>`);
  body.push(cap(94, 87, '饱和 β/K ≈ 6240 token/s', { mono: true, size: 9 }));
  const xc = 'M 90 268 C 140 226, 180 196, 232 178 C 284 160, 320 154, 375 149 C 450 141, 560 131, 660 126';
  body.push(`<path class="dia-edge strong" d="${xc}"/>`);
  body.push(pathPulse(xc, { dur: 3, delay: 0 }));
  body.push(cap(150, 246, 'X(B) = βB/(W+BK)', { mono: true, size: 9 }));

  // TPOT(B) 仿射线（seal）+ SLO τ=10ms
  body.push(`<path d="M 90 233.5 L 660 85.8" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(cap(300, 222, 'TPOT(B) = (W+BK)/β', { mono: true, size: 9, color: 'var(--seal-deep)' }));
  body.push(`<line x1="90" y1="196" x2="660" y2="196" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 5"/>`);
  body.push(cap(94, 190, 'TPOT SLO τ = 10 ms', { size: 9, color: 'var(--seal-deep)' }));

  // 操作点 B=32（SLO 与容量两上限的较小者）
  body.push(`<line class="dia-edge dash" x1="235" y1="196" x2="235" y2="268"/>`);
  body.push(`<circle cx="232" cy="178" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(243, 214, '操作点 B=32', { size: 9.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(243, 228, 'X≈3225 · TPOT 9.92 ms', { mono: true, size: 9 }));
  // B=64 对照点
  body.push(`<circle cx="375" cy="149" r="3.5" fill="var(--ink)"/>`);
  body.push(cap(381, 170, 'B=64：X≈4252（+32%）', { mono: true, size: 9 }));
  body.push(cap(381, 184, 'TPOT 15.05 ms 越 SLO', { size: 9, color: 'var(--seal-deep)' }));
  // KV 容量上限
  body.push(`<line class="dia-edge dash" x1="620" y1="100" x2="620" y2="268"/>`);
  body.push(cap(620, 298, '容量上限 ≈119', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 330, 'B_max = (βτ−W)/K ≈ 32.5 → 取 32；操作点 = min(TPOT SLO 上限, KV 容量上限)，再叠 ρ 余量（§5）', { size: 10.5 }));
  body.push(cap(36, 348, '同 τ = 10 ms 换卡：H200 B_max≈59 · X≈5949；B200 B_max≈113 · X≈11351 —— 带宽升级定量兑现', { size: 10.5 }));
  body.push(cap(36, 366, 'static batching 第三曲线：TTFT 排队 ≈ C(B)/2 随 B 单调增；continuous batching 消去 head-of-line（4.1）', { size: 10 }));

  return scene(720, 380, body, { label: 'batch 三方权衡：吞吐曲线饱和于 β/K，TPOT 仿射增，TPOT SLO 与 KV 容量夹出操作点 B=32' }).svg();
}

/* ============================================================
 * 1.4 §3（block 0）— logits processor 链（顺序是语义）
 * ============================================================ */
function processorChain() {
  const UID = 'i1f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'logits processor 链（B 条序列并行，每行独立）：顺序是语义的一部分', { size: 12, color: 'var(--ink-soft)' }));

  const chain = [
    ['n1', 56, 'data', 'logits z [B, V]'],
    ['n2', 122, 'data', "z' [B, V]"],
    ['n3', 188, 'data', "z'/τ [B, V]"],
    ['n4', 254, 'data', 'masked [B, V]'],
    ['n5', 320, 'proj', 'p [B, V]'],
    ['n6', 386, 'attn', 'token id [B]'],
  ].map(([id, y, role, title]) => {
    const n = node({ id, x: 110, y, w: 160, h: 40, role, title, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 5; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, portFrom: chain[i].bottom, portTo: chain[i + 1].top, pulse: true, dur: 1.1, delay: i * 0.3, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(chain[i + 1].top[0], chain[i + 1].top[1], 0, 1));
  }
  // 变换名贴左缘
  [['penalty 式(5)(6)', 112, 9.5], ['÷ τ 式(1)', 178, 9.5], ['mask→−∞ 式(2)(4)', 244, 9], ['log_softmax 式(7)', 306, 9.5], ['+ 再归一化 式(3)', 320, 9.5], ['Gumbel-max 式(8)', 376, 9.5]].forEach(([t, y, sz]) => {
    body.push(cap(100, y, t, { mono: true, size: sz, anchor: 'end' }));
  });

  // 右栏注记（与各级对齐）
  body.push(cap(300, 100, 'repetition 乘性 (5) / frequency·presence 加性 (6)', { size: 9.5 }));
  body.push(cap(300, 114, '两者不可交换（命题 2）—— 链顺序改变输出', { size: 9.5 }));
  body.push(cap(300, 166, 'τ < 1 锐化 · τ > 1 平坦 · τ → 0 = greedy（式 1）', { size: 9.5 }));
  body.push(cap(300, 232, 'top-k / top-p / min-p / grammar mask：非法位 → −∞', { size: 9.5 }));
  body.push(cap(300, 246, 'grammar：自动机状态 s → 合法集 A(s)（§5 命题 5）', { size: 9.5 }));
  body.push(cap(300, 298, '减 max：exp 参数 ≤ 0 不溢出（式 7，z=[1000,1001,999]）', { size: 9.5 }));
  body.push(cap(300, 312, '截断再归一化：核内和 = 1、被截位 = 0（式 3）', { size: 9.5 }));
  body.push(cap(300, 364, 'argmax(z/τ + g)，g ~ Gumbel(0,1) ≡ categorical', { size: 9.5 }));
  body.push(cap(300, 378, '无需全词表 softmax —— fused sampling kernel 的依据', { size: 9.5 }));

  // Gumbel 噪声芯片喂入采样边
  const g = node({ id: 'g', x: 300, y: 398, w: 150, h: 28, role: 'cache', title: 'g ~ Gumbel(0,1)', mono: true, titleSize: 10.5 });
  nodes.push(g); body.push(g.svg);
  body.push(edge(g, chain[5], { nodes, portFrom: g.left, portTo: chain[5].right, pulse: true, dur: 1.2, delay: 1.5, dash: true, uid: UID, eid: 'g' }));
  body.push(arrowHead(chain[5].right[0], chain[5].right[1], -1, 0));
  body.push(cap(460, 416, '噪声逐 token 独立', { size: 9 }));

  return scene(720, 452, body, { label: 'logits processor 链：penalty、温度、截断 mask、log_softmax 再归一化、Gumbel-max 采样逐级变换' }).svg();
}

/* ============================================================
 * 1.4 §2（afterHeading）— top-p 截断与再归一化（玩具分布）+ min-p 对照
 * 原文：p=[0.4,0.25,0.15,0.12,0.08]、C=[0.4,0.65,0.80,0.92,1.0]、p_th=0.8 → j*=3；
 * p'=[0.5,0.3125,0.1875,0,0]；min-p η=0.3 → 阈 0.12 保留 4 个、核质量 0.92
 * ============================================================ */
function topPNucleus() {
  const UID = 'i1g';
  const body = [];
  const P = [0.4, 0.25, 0.15, 0.12, 0.08];
  const C = [0.4, 0.65, 0.8, 0.92, 1.0];
  const XS = [70, 140, 210, 280, 350];
  const base = 168, sc = 120;

  body.push(cap(36, 24, '截断采样玩具例 p = [0.4, 0.25, 0.15, 0.12, 0.08]：核怎么选、怎么再归一化', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- top-p 面板 ---- */
  body.push(zone({ x: 36, y: 40, w: 430, h: 160, label: 'top-p（p_th = 0.8）' }));
  P.forEach((p, i) => {
    const h = p * sc;
    const tone = i < 3 ? 'seal' : 'soft';
    body.push(`<g class="sig-bar" style="--i:${(i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${tone}" x="${XS[i]}" y="${base - h}" width="50" height="${h}" rx="3"/></g>`);
    body.push(cap(XS[i] + 25, 182, String(p), { mono: true, size: 9, anchor: 'middle' }));
  });
  // 累积阶梯线（与概率同轴）
  const pts = XS.map((x, i) => [x + 25, base - C[i] * sc]);
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < 5; i++) d += ` L ${pts[i][0]} ${pts[i - 1][1]} L ${pts[i][0]} ${pts[i][1]}`;
  body.push(`<path class="dia-edge strong" d="${d}"/>`);
  body.push(pathPulse(d, { dur: 2.4, delay: 0.2 }));
  // 累积值标注避开阶梯竖线：C1/C2 标上方，C3 标交叉点右下自由象限，C4/C5 入 j* 注记
  body.push(cap(pts[0][0], pts[0][1] - 8, '0.4', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(pts[1][0], pts[1][1] - 8, '0.65', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(pts[2][0] + 6, pts[2][1] + 12, '0.80', { mono: true, size: 8.5 }));
  body.push(`<line x1="60" y1="72" x2="440" y2="72" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 5"/>`);
  body.push(cap(440, 66, 'p_th = 0.8', { mono: true, size: 9, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(235, 196, 'j* = 3：C = [0.4, 0.65, 0.80, 0.92, 1.0]，C₃ 首达阈 · 含跨阈 token', { size: 9, anchor: 'middle' }));

  /* ---- 再归一化行 ---- */
  body.push(cap(52, 224, '再归一化（式 3）：p′ = p / 0.8（核内）· 被截位 = 0 —— 和为 1、核内相对比例不变', { size: 10 }));
  const PR = [0.5, 0.3125, 0.1875, 0, 0];
  PR.forEach((p, i) => {
    if (p > 0) {
      const h = p * sc;
      body.push(`<rect class="dia-bar dia-bar-seal" x="${XS[i]}" y="${286 - h}" width="50" height="${h}" rx="3"/>`);
    } else {
      body.push(`<rect x="${XS[i]}" y="280" width="50" height="6" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="2 3"/>`);
    }
    body.push(cap(XS[i] + 25, 300, String(p), { mono: true, size: 9, anchor: 'middle' }));
  });

  /* ---- min-p 对照面板 ---- */
  body.push(zone({ x: 486, y: 40, w: 198, h: 160, label: 'min-p（η = 0.3）' }));
  const MX = [502, 538, 574, 610, 646];
  P.forEach((p, i) => {
    const h = p * sc;
    body.push(`<rect class="dia-bar dia-bar-${i < 4 ? 'seal' : 'soft'}" x="${MX[i]}" y="${base - h}" width="28" height="${h}" rx="3"/>`);
  });
  body.push(`<line x1="500" y1="153.6" x2="676" y2="153.6" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 5"/>`);
  body.push(cap(676, 148, 'η·p_max = 0.12', { mono: true, size: 8, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(585, 196, '砍 0.08 · 核质量 0.92', { size: 8.5, anchor: 'middle' }));
  body.push(cap(585, 224, 'p′ ≈ [0.435, 0.272,', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(585, 238, '0.163, 0.130, 0]', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(585, 262, '按相对峰值切：分布越尖核越小', { size: 8.5, anchor: 'middle' }));
  body.push(cap(585, 276, '同分布上比 top-p 更宽松', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 330, 'top-k：核大小固定 —— 尖分布仍强留 k 个（含坏 token）、平分布砍太多（结构性缺陷）；top-p / min-p 核随分布自适应', { size: 10 }));

  return scene(720, 344, body, { label: 'top-p 截断：累积质量阶梯首达 0.8 取前三 token 为核，再归一化；min-p 按相对峰值阈值保留四个' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/01-request-lifecycle-and-slos',
    replacesBlock: 0,
    title: '请求全链路：TTFT 段 × decode 段 × E2E 分解',
    svg: requestLifecycle,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/02-prefill-decode-accounting',
    replacesBlock: 0,
    title: 'decode 每步 HBM 流量：权重共享一遍 × KV 逐序列全读',
    svg: decodeHbmTraffic,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/02-prefill-decode-accounting',
    replacesBlock: 1,
    title: 'Wide-EP：dispatch/combine 与每卡流量 ∝ 1/e',
    svg: wideEpDispatch,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/03-roofline-and-tradeoffs',
    replacesBlock: 0,
    title: 'roofline：decode 左斜线（饱和封顶）× prefill 右平台',
    svg: roofline,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/03-roofline-and-tradeoffs',
    afterHeading: /^## 4\. /,
    title: 'batch 三方权衡：X(B) 饱和 × TPOT(B) 仿射 × SLO 操作点',
    svg: batchTradeoff,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/04-sampling-and-decoding',
    replacesBlock: 0,
    title: 'logits processor 链：顺序是语义的一部分',
    svg: processorChain,
  },
  {
    dim: 'inference',
    slug: '01-principles-and-accounting/04-sampling-and-decoding',
    afterHeading: /^## 2\. /,
    title: 'top-p 核 = 累积首达阈的最小前缀 · 再归一化 · min-p 对照',
    svg: topPNucleus,
  },
];
