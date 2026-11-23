// inference Part 4（04-scheduling-and-batching）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs / inference-part02.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言伪代码围栏不画（管线自动算法面板）。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性（类选择器 > 表现属性）——
// 显式传 color 时补 inline style 提权，保证朱砂强调 / 深底白字真正生效（不改 kit API）。
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

// iteration 格（时间线用）：useful = 有用计算（ink），idle = 空转（虚线），join = 补位点亮（seal）
const iterCell = (x, y, kind, { w = 48, h = 18, lit = null } = {}) => {
  const fill = kind === 'useful' ? 'var(--ink)' : kind === 'join' ? 'var(--seal)' : 'none';
  const stroke = kind === 'idle' ? ' stroke="var(--rule)" stroke-dasharray="3 3"' : '';
  const inner = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}"${stroke}/>`;
  return lit !== null ? `<g class="sig-cell" style="--i:${lit}">${inner}</g>` : inner;
};

/* ============================================================
 * 4.1 §1（block 0）— static vs continuous 时间线（Orca Fig.1 结构）
 * 原文：R1(2) R2(4) R3(3) R4(8)，R5 iter3 到达；static 空转 + 队头阻塞；
 *       continuous 完成即离开、到达即补位；η=23.4%、增益 1/η=4.27×
 * ============================================================ */
function staticVsContinuous() {
  const UID = 'i4a';
  const body = [];

  body.push(cap(36, 24, 'static vs continuous batching（Orca Fig.1 示意重绘）：■ 有用计算 · □ 空转 · R5 于 iter3 到达', { size: 12, color: 'var(--ink-soft)' }));

  const colX = (i) => 150 + i * 62; // iter 1..8
  const iters = [1, 2, 3, 4, 5, 6, 7, 8];

  /* ---- static ---- */
  body.push(zone({ x: 36, y: 44, w: 648, h: 154, label: 'static batching（run-to-completion）：同进同出 —— 早 finish 空转、新请求等整批' }));
  iters.forEach((n, i) => body.push(cap(colX(i) + 24, 74, `iter ${n}`, { mono: true, size: 8.5, anchor: 'middle' })));
  const rows = [['R1 (2 tok)', 2], ['R2 (4 tok)', 4], ['R3 (3 tok)', 3], ['R4 (8 tok)', 8]];
  rows.forEach(([lab, len], r) => {
    const y = 84 + r * 24;
    body.push(cap(52, y + 13, lab, { size: 9.5 }));
    for (let i = 0; i < 8; i++) body.push(iterCell(colX(i), y, i < len ? 'useful' : 'idle'));
  });
  body.push(cap(52, 185, 'R5 ↓iter3 到达', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(`<line class="dia-edge dash" x1="${colX(2)}" y1="180" x2="${colX(7) + 48}" y2="180"/>`);
  body.push(arrowHead(colX(7) + 52, 180, 1, 0));
  body.push(cap(colX(4), 192, '等待整批结束（队头阻塞）→ iter9 才进批', { size: 8.5, anchor: 'middle' }));

  /* ---- continuous ---- */
  body.push(zone({ x: 36, y: 204, w: 648, h: 168, tone: 'seal', label: 'continuous batching（iteration-level）：完成即离开、到达即补位 —— batch 每步重选' }));
  iters.forEach((n, i) => body.push(cap(colX(i) + 24, 234, `iter ${n}`, { mono: true, size: 8.5, anchor: 'middle' })));
  rows.forEach(([lab, len], r) => {
    const y = 244 + r * 24;
    body.push(cap(52, y + 13, lab, { size: 9.5 }));
    for (let i = 0; i < 8; i++) {
      if (i < len) body.push(iterCell(colX(i), y, 'useful'));
    }
    if (len < 8) body.push(cap(colX(len) + 4, y + 13, '离开 ↩ 块回池', { size: 8, color: 'var(--seal-deep)' }));
  });
  body.push(cap(52, 351, 'R5 ↓iter3 到达', { size: 9.5, color: 'var(--seal-deep)' }));
  for (let i = 2; i < 8; i++) body.push(iterCell(colX(i), 338, 'join', { lit: 0.4 + (i - 2) * 0.2 }));
  body.push(cap(colX(2) + 150, 368, '立即补位（R1 腾出的 slot）', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 384, '利用率（式 1）：η = ΣT_o / (B·maxT_o) —— 教学 workload 8 条（7×64 + 512）：960/4096 ≈ 23.4%，近八成 slot-步容量被空转吃掉', { size: 10 }));
  body.push(cap(36, 402, '吞吐账（式 4）：饱和到达 + 等步时 + 纯 decode → 增益 = 1/η = 4.27×（static 512 步 vs continuous 120 步）；Orca 实测 36.9× 是系统级口径', { size: 9.5 }));
  body.push(cap(36, 420, 'padding 另一笔账（式 2）：7×512+1×2048 补齐到 8×2048 → 浪费 10752 token（65.6%）；FCFS 约束服务顺序而非完成顺序（晚到短请求可先返回）', { size: 9.5 }));

  return scene(720, 432, body, { label: 'static batching 短请求完成后空转新请求等整批，continuous batching 完成即离开到达即补位，每个迭代都有满批有用计算' }).svg();
}

/* ============================================================
 * 4.1 §3（block 1）— selective batching：Split → 逐序列 Attn → Merge
 * 原文：A(2) B(3) C(2) → Flatten [7,4096]；权重只读一遍 I=7 vs 逐请求 2–3
 * ============================================================ */
function selectiveBatching() {
  const UID = 'i4b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'selective batching：position-wise 算子摊平合批、attention 逐序列（Orca Fig.5：7 token 批）', { size: 12, color: 'var(--ink-soft)' }));

  const reqs = [['A', 76, '请求 A（2 tok）'], ['B', 128, '请求 B（3 tok）'], ['C', 180, '请求 C（2 tok）']].map(([id, y, title]) => {
    const n = node({ id, x: 36, y, w: 128, h: 36, role: 'data', title, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const F = node({ id: 'F', x: 208, y: 110, w: 138, h: 72, role: 'proj', title: 'Flatten [7, 4096]', sub: 'QKV/FFN/LN 合批', mono: true, titleSize: 12, subSize: 9 });
  nodes.push(F); body.push(F.svg);
  body.push(cap(277, 200, '权重只读一遍', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  reqs.forEach((r, i) => {
    body.push(edge(r, F, { nodes, pulse: true, dur: 1.3, delay: i * 0.2, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(F.left[0], F.left[1], 1, 0));
  });

  body.push(cap(372, 100, 'Split', { mono: true, size: 9.5, anchor: 'middle' }));
  const attns = [['aA', 76, 'Attn(A)：q[2] × KV_A'], ['aB', 128, 'Attn(B)：q[3] × KV_B'], ['aC', 180, 'Attn(C)：q[2] × KV_C']].map(([id, y, title]) => {
    const n = node({ id, x: 400, y, w: 168, h: 36, role: 'attn', title, mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  attns.forEach((a, i) => {
    body.push(edge(F, a, { nodes, pulse: true, dur: 1.3, delay: 0.6 + i * 0.2, uid: UID, eid: `a${i}` }));
    body.push(arrowHead(a.left[0], a.left[1], 1, 0));
  });
  body.push(cap(484, 232, '逐序列 · KV 私有 · causal mask 按序列定义', { size: 9, anchor: 'middle' }));

  const M = node({ id: 'M', x: 614, y: 110, w: 70, h: 72, role: 'proj', title: 'Merge', sub: '[7, 4096]', mono: true, titleSize: 12, subSize: 9 });
  nodes.push(M); body.push(M.svg);
  attns.forEach((a, i) => {
    body.push(edge(a, M, { nodes, pulse: true, dur: 1.3, delay: 1.2 + i * 0.2, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(M.left[0], M.left[1], 1, 0));
  });
  body.push(cap(649, 200, '→ FFN/LN…', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 262, '命题 2：position-wise 输出行只依赖输入行 → 任意划分「分块算再拼接 = 整批算」；attention 不是 position-wise，跨序列合并改变输出', { size: 10 }));
  body.push(cap(36, 280, '反例：两序列 token 合进同一 attention 不加隔离 mask → 可见集扩到另一序列 KV、softmax 归一化域改变（且泄漏另一请求内容）', { size: 10 }));
  body.push(cap(36, 298, '合批收益落在权重读：[7,4096]×[4096,6144] 权重一遍 ≈ 5.03×10⁷ 字节、I = 7 FLOP/byte；逐请求 3 次 GEMV 读 3 遍、I 只有 2–3', { size: 10 }));
  body.push(cap(36, 316, 'selective batching 不改 FLOPs，改的是权重流量与 kernel launch 次数 —— 离 ridge ≈ 295 同样远，但访存流量差 3 倍', { size: 10 }));

  return scene(720, 330, body, { label: 'selective batching：三条请求摊平成 7 行大张量合批跑投影与 FFN，attention 前 Split 逐序列私有 KV 计算，再 Merge 回整批' }).svg();
}

/* ============================================================
 * 4.1 §4（block 2）— 每步循环：waiting / running / 抢占 / schedule→execute→update
 * ============================================================ */
function stepLoop() {
  const UID = 'i4c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'continuous batching 每步循环：schedule → execute（1 iteration）→ update —— batch 组成在步边界重选', { size: 12, color: 'var(--ink-soft)' }));

  const NR = node({ id: 'NR', x: 36, y: 84, w: 84, h: 40, role: 'plain', title: 'new req', mono: true, titleSize: 11 });
  const W = node({ id: 'W', x: 158, y: 76, w: 158, h: 56, role: 'cache', title: 'waiting', sub: '到达未准入 · 不占 KV 块', mono: true, titleSize: 13, subSize: 9 });
  const R = node({ id: 'R', x: 408, y: 76, w: 158, h: 56, role: 'attn', title: 'running', sub: '在批 · 持 KV 块 · 吃 token 预算', mono: true, titleSize: 13, subSize: 9 });
  const OUT = node({ id: 'OUT', x: 614, y: 84, w: 70, h: 40, role: 'plain', title: '返回', titleSize: 11 });
  [NR, W, R, OUT].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(NR, W, { nodes, pulse: true, dur: 1, delay: 0, uid: UID, eid: 'nw' }));
  body.push(arrowHead(W.left[0], W.left[1], 1, 0));
  body.push(edge(W, R, { nodes, pulse: true, dur: 1.4, delay: 0.3, uid: UID, eid: 'wr' }));
  body.push(arrowHead(R.left[0], R.left[1], 1, 0));
  body.push(cap(362, 68, '准入：token 预算 + 块可用 + watermark', { size: 8.5, anchor: 'middle' }));
  body.push(edge(R, OUT, { nodes, pulse: true, dur: 1, delay: 1.6, uid: UID, eid: 'ro' }));
  body.push(arrowHead(OUT.left[0], OUT.left[1], 1, 0));
  body.push(cap(590, 78, '完成', { size: 8.5, anchor: 'middle' }));
  body.push(cap(649, 140, '块释放回池', { size: 8.5, anchor: 'middle' }));

  // 抢占回路（running → waiting，下方折线）
  const pre = 'M 487 132 V 164 H 237 V 132';
  body.push(`<path class="dia-edge dash" d="${pre}"/>`);
  body.push(pathPulse(pre, { dur: 2, delay: 1.2 }));
  body.push(arrowHead(237, 132, 0, -1));
  body.push(cap(362, 158, '抢占（4.2）：分配失败 → 选 victim · 释放块 · 重排队', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 每步循环三段
  body.push(zone({ x: 36, y: 188, w: 648, h: 92, label: '每步循环（vLLM EngineCore.step() / SGLang event_loop_normal 同构）' }));
  const S = node({ id: 'S', x: 52, y: 218, w: 180, h: 44, role: 'op', title: 'schedule', sub: '定组成 · 分块（running 先）', mono: true, titleSize: 12, subSize: 8.5 });
  const E = node({ id: 'E', x: 270, y: 218, w: 190, h: 44, role: 'proj', title: 'execute', sub: '一次前向 + 采样', mono: true, titleSize: 12, subSize: 8.5 });
  const U = node({ id: 'U', x: 498, y: 218, w: 170, h: 44, role: 'op', title: 'update', sub: '追加 token · 判完成 · 释放', mono: true, titleSize: 12, subSize: 8.5 });
  [S, E, U].forEach((n) => { nodes.push(n); body.push(n.svg); });
  [[S, E], [E, U]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: 0.5 + i * 0.4, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  const loop = 'M 583 262 V 278 H 142 V 262';
  body.push(`<path class="dia-edge" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 1.8, delay: 1.4 }));
  body.push(arrowHead(142, 262, 0, -1));
  body.push(cap(362, 274, '下一步（完成即补位闭合）', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 306, '两个预算：max_num_batched_tokens（每步 token 上限，prefill+decode 共用）× max_num_seqs（并发上限）；watermark 只约束 waiting', { size: 9.5 }));
  body.push(cap(36, 324, 'vLLM V1：running 先 waiting 后 · 本步有抢占就不准入新请求（防振荡）；统一 token 视图：num_tokens_with_spec − num_computed_tokens', { size: 9.5 }));
  body.push(cap(36, 342, 'prefill 一步吃 prompt 长个 token、decode 每步吃 1 个 —— 两类请求在同一 token 预算下竞争（4.3 stall 的根源）', { size: 10 }));

  return scene(720, 356, body, { label: '每步循环：新请求进 waiting，经预算与块准入进 running，完成即返回释放块，分配失败触发抢占回 waiting；schedule execute update 三段逐 iteration 重选 batch' }).svg();
}

/* ============================================================
 * 4.2 §1（block 0）— 三队列架构：准入 / 抢占两路恢复
 * 原文：waiting→running 准入；recompute 恢复重排队；swap→swapped→swap-in 续跑；
 *       vLLM V1 无 swapped（只 recompute）；victim 选择随策略
 * ============================================================ */
function threeQueues() {
  const UID = 'i4d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '三队列架构与抢占恢复两条路（vLLM 论文 §4.5 设计空间）—— V1 现役无 swapped，抢占只走 recompute', { size: 12, color: 'var(--ink-soft)' }));

  const W = node({ id: 'W', x: 60, y: 110, w: 160, h: 64, role: 'cache', title: 'waiting', sub: '到达 / 抢占重排队 · 不占 KV 块', mono: true, titleSize: 14, subSize: 8.5 });
  const R = node({ id: 'R', x: 480, y: 110, w: 160, h: 64, role: 'attn', title: 'running', sub: '在批 · 持 KV 块 · 吃 token 预算', mono: true, titleSize: 14, subSize: 8.5 });
  const S = node({ id: 'S', x: 480, y: 252, w: 160, h: 64, role: 'data', title: 'swapped', sub: 'KV 在 host · 块已释放 · 状态保留', mono: true, titleSize: 14, subSize: 8.5 });
  [W, R, S].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 准入
  body.push(edge(W, R, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'admit' }));
  body.push(arrowHead(R.left[0], R.left[1], 1, 0));
  body.push(cap(350, 128, '准入（预算 + 块 + watermark）', { size: 9.5, anchor: 'middle' }));

  // recompute 恢复：running 顶部 → waiting 顶部（上方折线，不与准入交叉）
  const rec = 'M 560 110 V 72 H 140 V 110';
  body.push(`<path class="dia-edge dash" d="${rec}"/>`);
  body.push(pathPulse(rec, { dur: 2.2, delay: 0.6 }));
  body.push(arrowHead(140, 110, 0, 1));
  body.push(cap(350, 64, 'recompute 恢复：释放 KV · num_computed_tokens 清零 · 重排队（恢复时一次 prefill）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 抢占 swap：running 底 → swapped 顶
  body.push(edge(R, S, { nodes, portFrom: R.bottom, portTo: S.top, pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'pre' }));
  body.push(arrowHead(S.top[0], S.top[1], 0, 1));
  body.push(cap(576, 212, '块分配失败 → 抢占', { size: 9, anchor: 'middle' }));
  body.push(cap(576, 226, '换出 KV 到 host（D2H）', { size: 9, anchor: 'middle' }));

  // swap-in：swapped 左 → running 底（左侧弧线，避开抢占竖线）
  const si = 'M 480 284 C 430 284, 430 190, 520 176';
  body.push(`<path class="dia-edge" d="${si}"/>`);
  body.push(pathPulse(si, { dur: 1.8, delay: 1.6 }));
  body.push(arrowHead(520, 176, 1, -0.3));
  body.push(cap(424, 232, 'swap-in 恢复：', { size: 9, anchor: 'end' }));
  body.push(cap(424, 246, '换回即从断点续跑', { size: 9, anchor: 'end' }));

  body.push(cap(36, 344, 'victim 选择：vLLM FCFS 抢 running[-1]（最晚到）· PRIORITY 抢最低优先级且最晚到；SGLang retract 最短输出先牺牲（至少保留一条）', { size: 9.5 }));
  body.push(cap(36, 362, '版本事实：vLLM V1 当前 main 无 swapped 队列 —— _preempt_request 直接释放块、请求插回 waiting 队头；三队列是设计空间不是每个引擎的现役实现', { size: 9.5 }));
  body.push(cap(36, 380, 'SGLang retract：释放设备 KV · 可选留 host 备份（offload_kv），有备份从备份恢复、无备份且不可重建则 abort；机制与策略分离 —— 换策略不动块池', { size: 9.5 }));

  return scene(720, 394, body, { label: '三队列状态机：waiting 经准入进 running，抢占两条恢复路，recompute 释放 KV 重排队，swap 换出到 host 成 swapped 再换回续跑' }).svg();
}

/* ============================================================
 * 4.2 §2（afterHeading）— recompute vs swap 成本账 + β* 临界带宽
 * 原文：T=4096 swap 16.8ms vs recompute 133ms（7.9×）；T=32768 134ms vs 1.56s；
 *       β* = κL·F_eff/(N+LTd)：8.64→8.07→5.50 GB/s 递减；有效带宽塌陷坑
 * ============================================================ */
function recomputeVsSwap() {
  const UID = 'i4e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '抢占恢复成本账：recompute = 一次 prefill vs swap = 往返 KV —— 临界带宽 β* 随 T 递减（8B @H100）', { size: 12, color: 'var(--ink-soft)' }));

  const V = node({ id: 'V', x: 36, y: 76, w: 190, h: 64, role: 'data', title: '被抢占请求', sub: 'T = 4096 · KV = 512 MiB', titleSize: 12.5, subSize: 9.5 });
  nodes.push(V); body.push(V.svg);
  const RC = node({ id: 'RC', x: 300, y: 52, w: 372, h: 48, role: 'proj', title: 'recompute：已生成 token 拼回 prompt，一次 prefill 重建全部 KV', sub: '6.58×10¹³ FLOPs ÷ 494.75 TFLOPS（50% 峰值）≈ 133 ms', titleSize: 11, subSize: 9 });
  const SW = node({ id: 'SW', x: 300, y: 124, w: 372, h: 48, role: 'cache', title: 'swap：D2H 换出 + H2D 换回（往返 2 × 512 MiB）', sub: '1.07 GB ÷ 64 GB/s（PCIe Gen5 理论）≈ 16.8 ms —— 理想拷贝下快 7.9×', titleSize: 11, subSize: 9 });
  [RC, SW].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(V, RC, { nodes, portFrom: V.right, portTo: RC.left, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'rc' }));
  body.push(arrowHead(RC.left[0], RC.left[1], 1, 0));
  body.push(edge(V, SW, { nodes, portFrom: V.right, portTo: SW.left, pulse: true, dur: 1.5, delay: 0.4, uid: UID, eid: 'sw' }));
  body.push(arrowHead(SW.left[0], SW.left[1], 1, 0));
  body.push(cap(262, 70, '不是 T 次 decode', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // β*(T) 曲线：x 90..650（T 0..36000），y 208..300（β 0..9 GB/s）
  const X = (T) => 90 + (T / 36000) * 560;
  const Y = (b) => 300 - b * 10.2;
  body.push(`<line class="dia-edge strong" x1="90" y1="300" x2="660" y2="300"/>`);
  body.push(`<line class="dia-edge strong" x1="90" y1="300" x2="90" y2="200"/>`);
  body.push(arrowHead(660, 300, 1, 0));
  body.push(arrowHead(90, 200, 0, -1));
  body.push(cap(84, 206, 'β* (GB/s)', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(660, 314, 'T', { mono: true, size: 9, anchor: 'end' }));
  [[4096, '4096'], [32768, '32768']].forEach(([t, lab]) => {
    body.push(`<line class="dia-edge" x1="${X(t)}" y1="295" x2="${X(t)}" y2="305"/>`);
    body.push(cap(X(t), 318, lab, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  // β*(T) = 8.64·N/(N+L·T·d)：N=7.5e9, L·d=131072 → β*(T)=8.64/(1+T·1.7476e-5)
  const beta = (T) => 8.64 / (1 + T * 1.7476e-5);
  let d = '';
  for (let T = 0; T <= 36000; T += 1500) d += `${T === 0 ? 'M' : 'L'} ${X(T).toFixed(1)} ${Y(beta(T)).toFixed(1)} `;
  body.push(`<path class="dia-edge strong" d="${d}"/>`);
  body.push(pathPulse(d, { dur: 2.6, delay: 0.3 }));
  [[0, 8.64, 'T→0 上确界 8.64'], [4096, 8.07, '8.07'], [32768, 5.5, '5.50']].forEach(([t, b, lab], i) => {
    body.push(`<circle cx="${X(t)}" cy="${Y(b)}" r="4" fill="var(--seal)"/>`);
    body.push(cap(X(t) + (i === 0 ? 8 : 6), Y(b) - (i === 0 ? 16 : 8), lab, { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  });
  body.push(cap(400, 268, 'β* = κL·F_eff / (N + L·T·d_model)（式 3）严格单调递减 —— 长上下文更偏向 swap', { size: 9.5, anchor: 'middle' }));
  body.push(cap(400, 284, 'PCIe Gen5 理论 64 GB/s ≫ β* → 理想账 swap 赢；attention 二次项占比 6.7%（T=4096）→ 36%（T=32768）', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 344, '两档对照：T=4096 swap 16.8 ms vs recompute 133 ms（7.9×）· T=32768 swap 134 ms vs recompute 1.56 s（11.6×）', { mono: true, size: 9.5 }));
  body.push(cap(36, 362, '坑：β_pcie 是有效带宽 —— 小 b 按块 swap 效率塌陷（b=16 拷贝次数是 b=64 的 4×）；论文 §7.3 实测 recompute ≤ 20% swap 延迟 → V1 默认 recompute', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(36, 380, '两笔修正：prefix cache 命中压低 recompute（式 1 是上界）；133 ms ≈ 26 个整批 decode 步（5.13 ms/步）—— 抢占吞吐代价按频率摊', { size: 9.5 }));

  return scene(720, 394, body, { label: '抢占恢复两条路的成本：recompute 是一次 prefill 约 133 毫秒，swap 往返 KV 约 16.8 毫秒；临界带宽曲线随序列长递减，但有效带宽塌陷使 V1 默认 recompute' }).svg();
}

/* ============================================================
 * 4.2 §3（block 1）— FastServe skip-join MLFQ + proactive swap
 * 原文：q1=8 逐级翻倍；skip-join 按 t_init 入队；周期提升防饥饿（300ms）；
 *       512 token 经 504 步沉 Q7；KV 峰值 7× FCFS；proactive swap 重叠
 * ============================================================ */
function fastserveMlfq() {
  const UID = 'i4f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'FastServe skip-join MLFQ：按预测 prefill 时间选入队层级 —— prefill 不被腰斩、抢占粒度 = 单 token', { size: 12, color: 'var(--ink-soft)' }));

  // 队列阶梯
  const qs = [
    ['Q1（q=8）', 70, '短作业', 1, '完成即返回'],
    ['Q2（q=16）', 102, '中作业', 2, ''],
    ['Q3（q=32）', 134, '长作业', 4, 'quantum 耗尽 → 降级'],
    ['Q7', 182, '超长作业', 6, ''],
  ];
  qs.forEach(([lab, y, job, n, note], qi) => {
    body.push(cap(52, y + 14, lab, { mono: true, size: 9.5 }));
    body.push(`<rect x="140" y="${y}" width="220" height="22" rx="6" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="1.1"/>`);
    for (let j = 0; j < n; j++) {
      body.push(`<g class="sig-cell" style="--i:${(qi * 0.3 + j * 0.12).toFixed(2)}"><rect x="${148 + j * 26}" y="${y + 5}" width="18" height="12" rx="2" fill="var(--seal)"/></g>`);
    }
    body.push(cap(372, y + (note ? 0 : 14), note || job, { size: 8.5, color: note ? 'var(--ink-faint)' : 'var(--ink-soft)' }));
  });
  body.push(cap(250, 168, '⋯（quantum 逐级翻倍）', { size: 8.5, anchor: 'middle' }));
  // 降级箭头（右侧）
  body.push(`<path class="dia-edge" d="M 366 92 V 100"/>`);
  body.push(arrowHead(366, 100, 0, 1));
  body.push(`<path class="dia-edge" d="M 366 124 V 132"/>`);
  body.push(arrowHead(366, 132, 0, 1));
  // 提升回路（左侧，Q7 → Q1）
  const prom = 'M 140 193 H 120 V 81 H 140';
  body.push(`<path class="dia-edge dash" d="${prom}"/>`);
  body.push(pathPulse(prom, { dur: 2.2, delay: 1.0 }));
  body.push(arrowHead(140, 81, 1, 0));
  body.push(cap(120, 214, '周期性提升防饥饿（阈值默认 300 ms）', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));

  // skip-join 右面板
  const NA = node({ id: 'NA', x: 470, y: 64, w: 214, h: 44, role: 'data', title: '新到达：profile t_init', sub: '半信息无关：输出未知、迭代时间可预测', titleSize: 11, subSize: 8 });
  nodes.push(NA); body.push(NA.svg);
  const sj = 'M 680 108 V 145 H 360';
  body.push(`<path class="dia-edge dash" d="${sj}"/>`);
  body.push(pathPulse(sj, { dur: 1.8, delay: 0.5 }));
  body.push(arrowHead(360, 145, -1, 0));
  body.push(cap(577, 128, 'skip：跳过所有 q < t_init 的队列', { size: 8.5, anchor: 'middle' }));
  body.push(cap(577, 166, '直接入满足 q_i ≥ t_init 的最高 Q_i', { size: 8.5, anchor: 'middle' }));
  body.push(cap(577, 182, '→ 第一次 iteration 在 quantum 内完成（命题 2）', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(577, 198, '例：t_init 落在 Q3 → 不抢占则队头阻塞回潮、', { size: 8, anchor: 'middle' }));
  body.push(cap(577, 210, '抢占则中间激活作废 —— skip-join 两难都避', { size: 8, anchor: 'middle' }));

  // proactive swap
  body.push(zone({ x: 36, y: 228, w: 648, h: 74, label: 'proactive swapping：低优先级队列 KV 主动换出，即将调度者提前换入 —— swap 移出关键路径' }));
  const PK = node({ id: 'PK', x: 52, y: 256, w: 150, h: 34, role: 'cache', title: 'pending KV（各队列）', titleSize: 10 });
  const HOST = node({ id: 'HOST', x: 280, y: 256, w: 120, h: 34, role: 'data', title: 'host 内存', titleSize: 10.5 });
  const NEXT = node({ id: 'NEXT', x: 478, y: 256, w: 190, h: 34, role: 'attn', title: '即将调度者：提前 swap-in', titleSize: 10 });
  [PK, HOST, NEXT].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(PK, HOST, { nodes, pulse: true, dur: 1.3, delay: 0.8, uid: UID, eid: 'out' }));
  body.push(arrowHead(HOST.left[0], HOST.left[1], 1, 0));
  body.push(cap(241, 268, 'swap-out', { mono: true, size: 8, anchor: 'middle' }));
  body.push(edge(HOST, NEXT, { nodes, pulse: true, dur: 1.3, delay: 1.4, uid: UID, eid: 'in' }));
  body.push(arrowHead(NEXT.left[0], NEXT.left[1], 1, 0));
  body.push(cap(439, 268, '与 running 计算重叠', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 326, '教学数值（q₁=8 步逐级翻倍）：输出 8 token 在 Q1 跑完；512 token 经 504 步沉到 Q7 —— 新到短作业最多等 8 步（FCFS 要等 512 步）', { size: 9.5 }));
  body.push(cap(36, 344, 'MLFQ 代价：在途 = running + pending，峰值 KV 可达 FCFS 7×（论文 Fig.8）；OPT-175B KV 2.3 GB · 步 60 ms · swap 36 ms → 重叠藏进计算', { size: 9.5 }));
  body.push(cap(36, 362, 'kill + recompute 低优先级不行：浪费算力，且与饥饿提升组合可能循环互杀死锁；端到端对照 vLLM 同延迟约束吞吐至多 31.4× / 17.9×（摘要口径）', { size: 9.5 }));

  return scene(720, 376, body, { label: 'FastServe 多级反馈队列：新请求按预测 prefill 时间跳级入队保证首迭代完整，长作业逐级下沉并周期提升防饥饿，pending KV 主动换出换入与计算重叠' }).svg();
}

/* ============================================================
 * 4.3 §1（block 0）— stall 时间线：不开 vs 开 chunked prefill
 * 原文：T=32768 prefill 781ms 独占 → TBT 尖刺 52×；c_chunk=512 → 64 chunk
 *       每 iteration piggyback 64 decode：混合 20/21ms，TBT 无尖刺
 * ============================================================ */
function stallTimeline() {
  const UID = 'i4g';
  const body = [];

  body.push(cap(36, 24, 'stall 账：长 prefill 独占一个 iteration → TBT 尖刺 52×；chunked + piggyback → TBT 有界（8B @H100 · B_D=64）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 不开 ---- */
  body.push(zone({ x: 36, y: 44, w: 648, h: 98, label: '不开 chunked prefill：T=32768 prompt 到达，prefill 独占 781 ms（峰值下界）' }));
  body.push(`<g class="sig-bar" style="--i:0"><rect x="52" y="76" width="430" height="26" rx="4" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.1"/></g>`);
  body.push(cap(267, 93, 'prefill 独占 781 ms（MFU=0.5 → 1.56 s）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(`<rect x="490" y="76" width="80" height="26" rx="4" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="1.1"/>`);
  body.push(cap(530, 93, 'decode 15 ms', { size: 8.5, anchor: 'middle' }));
  body.push(`<rect x="578" y="76" width="80" height="26" rx="4" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="1.1"/>`);
  body.push(cap(618, 93, 'decode 15 ms', { size: 8.5, anchor: 'middle' }));
  body.push(cap(52, 126, '64 条 decode 的 TBT：', { size: 9.5 }));
  body.push(cap(267, 126, '781 ms（尖刺 ≈52×）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(530, 126, '15 ms', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(618, 126, '15 ms', { mono: true, size: 9, anchor: 'middle' }));

  /* ---- 开 ---- */
  body.push(zone({ x: 36, y: 154, w: 648, h: 118, tone: 'seal', label: '开 chunked prefill：c_chunk=512 → 64 个 chunk，每 iteration piggyback 全部 64 条 decode' }));
  const mixed = [[52, '混合₁', '20 ms'], [172, '混合₂', '20 ms'], [352, '混合₆₄', '21 ms']];
  mixed.forEach(([x, lab, dur], i) => {
    // 双层：上 = prefill chunk（seal），下 = decode 64×1（ink）
    body.push(`<g class="sig-cell" style="--i:${(i * 0.4).toFixed(1)}"><rect x="${x}" y="186" width="112" height="13" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="0.9"/></g>`);
    body.push(`<g class="sig-cell" style="--i:${(0.2 + i * 0.4).toFixed(1)}"><rect x="${x}" y="201" width="112" height="13" rx="3" fill="var(--ink)"/></g>`);
    body.push(cap(x + 56, 196, 'chunk 512 tok', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
    body.push(cap(x + 56, 211, 'decode 64×1 tok', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
    body.push(cap(x + 56, 230, `${lab} ${dur}`, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  body.push(cap(312, 205, '⋯', { size: 12, anchor: 'middle' }));
  body.push(`<rect x="472" y="186" width="80" height="28" rx="4" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="1.1"/>`);
  body.push(cap(512, 204, 'decode 15 ms', { size: 8.5, anchor: 'middle' }));
  body.push(cap(512, 230, 'prefill 完成', { size: 8, anchor: 'middle' }));
  body.push(cap(568, 196, '每个 iteration 都产出', { size: 8.5 }));
  body.push(cap(568, 209, '1 token/序列 →', { size: 8.5 }));
  body.push(cap(568, 222, 'TBT = iteration 时长', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 254, 'TBT：20 / 20 / ⋯ / 21 / 15 ms —— 尖刺从 52× 压到 ≈1.4×（命题 2：stall-free 的 TBT 上界 = 最大 iteration 步时）', { size: 9 }));

  body.push(cap(36, 296, '机制：batch 是执行的原子单位，批内所有序列共享 iteration 时长（1.1 命题 1）—— T=4096 的 prefill 66.5 ms 已是 4.4× 尖刺，不是极端长度专利', { size: 9.5 }));
  body.push(cap(36, 314, '两手合起来才 stall-free：chunking 给单 iteration 非 decode 工作量封顶 + 混合批保证 decode 每步产出；非抢占 —— 只改分批粒度', { size: 9.5 }));
  body.push(cap(36, 332, '尖刺按逐间隔 TBT 样本度量，不是请求级 TPOT 均值 —— 均值被几百个正常间隔稀释，尖刺看不见（1.1 坑注）', { size: 10 }));

  return scene(720, 346, body, { label: 'stall 时间线对照：不开 chunk 时长 prefill 独占 iteration 使 decode TBT 尖刺 52 倍，开 chunk 后每个混合 iteration 都带全部 decode，TBT 等于约 20 毫秒的步时' }).svg();
}

/* ============================================================
 * 4.3 §2（block 1）— 混合批 shape 走读 + piggyback 账
 * 原文：X[576,4096]；GEMM 相 max(8.7,4.8)=8.7ms；attention 相 10.3ms；
 *       iteration 20.0ms；piggyback 省 3.9ms（权重读 4.8−1.0）；B_D<317
 * ============================================================ */
function mixedBatchShape() {
  const UID = 'i4h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '混合批 shape 走读（c_chunk=512 + B_D=64）：GEMM 按 token 合批、attention 逐序列 —— piggyback 摊掉权重读', { size: 12, color: 'var(--ink-soft)' }));

  const X = node({ id: 'X', x: 36, y: 88, w: 150, h: 64, role: 'data', title: 'X [576, 4096]', sub: '512 prefill 行 + 64 decode 行', mono: true, titleSize: 12.5, subSize: 8.5 });
  nodes.push(X); body.push(X.svg);

  const G = node({ id: 'G', x: 250, y: 52, w: 260, h: 56, role: 'proj', title: 'QKV/O 投影 + FFN GEMM', sub: '576 行全部合批 · 权重读一遍 W = 16.1 GB', titleSize: 12, subSize: 8.5 });
  nodes.push(G); body.push(G.svg);
  body.push(edge(X, G, { nodes, portFrom: X.right, portTo: G.left, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'xg' }));
  body.push(arrowHead(G.left[0], G.left[1], 1, 0));
  body.push(cap(524, 70, 'GEMM 相：max(2N×576/F, W/BW)', { mono: true, size: 8.5 }));
  body.push(cap(524, 84, '= max(8.7, 4.8) = 8.7 ms', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(524, 98, 'compute-bound · decode 64 行搭', { size: 8.5 }));
  body.push(cap(524, 111, '同一次权重读 —— piggyback', { size: 8.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 250, y: 128, w: 434, h: 132, label: 'attention 相：逐序列拆开 —— 形状异构，padding 合批白算' }));
  const CA = node({ id: 'CA', x: 266, y: 158, w: 200, h: 48, role: 'attn', title: 'prefill chunk', sub: 'Q[512,32,128] × KV[c+512,8,128]', mono: true, titleSize: 11, subSize: 8 });
  const DA = node({ id: 'DA', x: 486, y: 158, w: 182, h: 48, role: 'attn', title: 'decode 序列 ×64', sub: 'Q[1,32,128] × 自己 KV[T̄ᵢ,8,128]', mono: true, titleSize: 11, subSize: 8 });
  [CA, DA].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(X, CA, { nodes, portFrom: X.bottom, portTo: CA.left, pulse: true, dur: 1.5, delay: 0.5, uid: UID, eid: 'xa' }));
  body.push(arrowHead(CA.left[0], CA.left[1], 1, 0));
  body.push(cap(266, 224, 'causal mask · varlen / paged kernel（3.1）', { size: 8.5 }));
  body.push(cap(486, 224, '各读各的 KV，不可共享', { size: 8.5 }));
  body.push(cap(266, 238, 'decode KV 读 64×0.537 GB = 34.4 GB → 10.3 ms（memory-bound）', { size: 8.5 }));
  body.push(cap(266, 250, 'chunk 自 attn 0.07 ms + decode attn 计算 0.14 ms，可忽略', { size: 8.5 }));

  // 三相账条：24px/ms
  body.push(cap(52, 272, 'iteration ≈ GEMM 相 + attention 相 + τ_oh（式 2 两相集中模型）：', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-ink" x="52" y="280" width="209" height="16" rx="3"/></g>`);
  body.push(cap(156, 292, 'GEMM 8.7 ms', { size: 8.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-warm" x="264" y="280" width="247" height="16" rx="3"/></g>`);
  body.push(cap(387, 292, 'attention 10.3 ms', { size: 8.5, anchor: 'middle' }));
  body.push(`<g class="sig-bar" style="--i:0.8"><rect class="dia-bar dia-bar-seal" x="514" y="280" width="24" height="16" rx="3"/></g>`);
  body.push(cap(546, 292, 'τ_oh 1 ms', { mono: true, size: 9 }));
  body.push(cap(36, 312, 'iteration ≈ 8.7 + 10.3 + 1(τ_oh) = 20.0 ms = TBT（对比不切 781 ms 尖刺）', { mono: true, size: 9 }));

  body.push(cap(36, 330, 'piggyback 边际账：混合 20.0 − 纯 prefill 8.8 = decode 边际 11.2 ms < 独立 decode 15.1 → 省 ≈3.9 ms（主项 = 摊掉的权重读 4.8 − 1.0 = 3.8 ms）', { size: 9.5 }));
  body.push(cap(36, 348, '节省为正条件 B_D < WF/(2N_act·BW) ≈ 317；B_D=512 边际反贵 2.9 ms —— 但 stall-free 延迟收益仍在：吞吐节省有条件，延迟有界无条件', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(36, 366, 'attention 相地板随 B_D 抬：64 → 119（容量上限，§2）KV 读 34.4 → 63.9 GB、10.3 → 19.1 ms —— decode-maximal 也有吞吐-延迟账', { size: 9.5 }));

  return scene(720, 376, body, { label: '混合批数据流：576 行 token 矩阵 GEMM 全合批权重读一遍，attention 拆成 chunk 与 64 条 decode 各算各的；三相时间账合计 20 毫秒，decode 搭权重读便车' }).svg();
}

/* ============================================================
 * 4.3 §4（block 2）— DP all-reduce MIN：组预算取最紧 rank
 * 原文：rank1 50ms / rank2 40ms → MIN 40ms；c=8192→q=1012、c=65536→q=400；
 *       同步步时 = max(各 rank) ≤ 预算；MIN 缩预算不是缩 q
 * ============================================================ */
function dpQuotaMin() {
  const UID = 'i4i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DeepInfer compute-quota × DP all-reduce MIN：组预算取最紧 rank —— 同预算、不同前缀、不同 chunk', { size: 12, color: 'var(--ink-soft)' }));

  const R1 = node({ id: 'R1', x: 36, y: 64, w: 184, h: 56, role: 'data', title: 'rank 1', sub: '本地 quota 50 ms · c = 8192', mono: true, titleSize: 12.5, subSize: 9 });
  const R2 = node({ id: 'R2', x: 36, y: 150, w: 184, h: 56, role: 'data', title: 'rank 2', sub: '本地 quota 40 ms · c = 65536', mono: true, titleSize: 12.5, subSize: 9 });
  const MIN = node({ id: 'MIN', x: 282, y: 104, w: 160, h: 62, role: 'op', title: 'all-reduce MIN', sub: '组预算 = 40 ms', mono: true, titleSize: 12.5, subSize: 9.5 });
  const SOL = node({ id: 'SOL', x: 502, y: 104, w: 182, h: 62, role: 'proj', title: '各按 40 ms 解式 (8)', sub: 'q* = 正根 · 向下取整 · cap 32768', titleSize: 11.5, subSize: 8.5 });
  [R1, R2, MIN, SOL].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(R1, MIN, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'r1' }));
  body.push(arrowHead(MIN.left[0], MIN.left[1], 1, 0));
  body.push(edge(R2, MIN, { nodes, pulse: true, dur: 1.3, delay: 0.3, uid: UID, eid: 'r2' }));
  body.push(arrowHead(MIN.left[0], MIN.left[1], 1, 0));
  body.push(edge(MIN, SOL, { nodes, pulse: true, dur: 1.3, delay: 0.8, uid: UID, eid: 'ms' }));
  body.push(arrowHead(SOL.left[0], SOL.left[1], 1, 0));

  const Q1 = node({ id: 'Q1', x: 502, y: 200, w: 182, h: 36, role: 'attn', title: 'c=8192 → q = 1012', mono: true, titleSize: 11.5 });
  const Q2 = node({ id: 'Q2', x: 502, y: 252, w: 182, h: 36, role: 'attn', title: 'c=65536 → q = 400', mono: true, titleSize: 11.5 });
  [Q1, Q2].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(SOL, Q1, { nodes, portFrom: SOL.bottom, portTo: Q1.top, pulse: true, dur: 1.1, delay: 1.3, uid: UID, eid: 'q1' }));
  body.push(arrowHead(Q1.top[0], Q1.top[1], 0, 1));
  // Q2 走右侧外绕，避免竖线穿过 Q1 卡
  const q2p = 'M 684 135 H 696 V 270 H 684';
  body.push(`<path class="dia-edge" d="${q2p}"/>`);
  body.push(pathPulse(q2p, { dur: 1.6, delay: 1.6 }));
  body.push(arrowHead(684, 270, -1, 0));

  body.push(cap(36, 216, 'attention 各 rank 本地算；', { size: 9.5 }));
  body.push(cap(36, 232, 'FFN/MoE 集合通信每步同步 →', { size: 9.5 }));
  body.push(cap(36, 248, '墙钟由最慢 rank 决定：', { size: 9.5 }));
  body.push(cap(36, 264, '预算宽的 rank 取大 chunk 会超装，', { size: 9.5 }));
  body.push(cap(36, 280, '全组等它 —— MIN 缩到最紧预算', { size: 9.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 316, '同步步时 = max(各 rank 实际步时) ≤ 预算（保守界）—— MIN 缩的是预算，不是直接缩 q；q 的缩减由式 (8) 逐 rank 完成', { size: 10 }));
  body.push(cap(36, 334, '代价：有余裕的 rank 利用不足；缓解 = prefill 四长度档（<16K…≥256K）各设 quota cap，按上轮份额重分配（doc L137）', { size: 9.5 }));

  return scene(720, 348, body, { label: 'DP 组时间预算：两个 rank 本地 quota 经 all-reduce MIN 取最紧 40 毫秒，各 rank 按组预算解二次成本模型，前缀长的解出更小 chunk，同步步时有界' }).svg();
}

/* ============================================================
 * 4.4 §1（block 0）— 一轮 draft-verify（γ=4）+ 接受账
 * 原文：draft 串行 x1..x4；verify 一次前向 p1..p5；接受前 n + 重采样/bonus；
 *       α=0.8 c=0.1：E[N_tok]=3.36、E[K]=2.36、F=2.40
 * ============================================================ */
function draftVerifyRound() {
  const UID = 'i4j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '一轮 draft-verify（γ=4 · α=0.8 · c=0.1）：γ 次串行 M_q + 1 次并行 M_p —— 闲置算力换被接受的 token', { size: 12, color: 'var(--ink-soft)' }));

  // draft
  body.push(zone({ x: 36, y: 44, w: 648, h: 80, label: 'draft：M_q（0.8B）串行猜 γ=4 个 token，每步 ≈ c·T（便宜但只能串行）' }));
  const ds = [['x₁', 'x₁ ~ q(·|prefix)'], ['x₂', 'x₂ ~ q(·|prefix,x₁)'], ['x₃', 'x₃ ~ q(·|prefix,x₁,x₂)'], ['x₄', 'x₄ ~ q(·|prefix,x₁..x₃)']].map(([id, title], i) => {
    const n = node({ id, x: 52 + i * 158, y: 74, w: 142, h: 36, role: 'data', title, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 3; i++) {
    body.push(edge(ds[i], ds[i + 1], { nodes, pulse: true, dur: 1, delay: i * 0.3, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(ds[i + 1].left[0], ds[i + 1].left[1], 1, 0));
  }

  // verify
  body.push(zone({ x: 36, y: 136, w: 648, h: 92, label: 'verify：M_p（8B）一次前向打 γ+1=5 个位置（prefill 形态 · 权重仍只读一遍 ≈ T）' }));
  body.push(cap(52, 172, '输入 [prefix, x₁, x₂, x₃, x₄] → 输出 5 个位置分布 p₁..p₅（p₅ = bonus 位）', { mono: true, size: 9.5 }));
  const ps = [1, 2, 3, 4, 5].map((k, i) => {
    const title = k === 1 ? 'p1(·|prefix)' : `p${k}(·|prefix,x1..x${k - 1})`;
    const n = node({ id: `p${k}`, x: 52 + i * 126, y: 176, w: 110, h: 36, role: 'attn', title, mono: true, titleSize: k === 1 ? 9.5 : 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(edge([360, 124], [360, 136], { pulse: true, dur: 0.8, delay: 1.0, uid: UID, eid: 'dv' }));
  body.push(arrowHead(360, 136, 0, 1));

  // accept
  body.push(zone({ x: 36, y: 240, w: 648, h: 92, label: 'accept：逐位比 pᵢ 与 qᵢ —— 接受前 n 个，第 n+1 位从 p（或残差）采样（示意 n=3）' }));
  const acc = [
    [52, '接受 ✓', 'var(--seal-wash)', 'var(--seal)', 0.3],
    [178, '接受 ✓', 'var(--seal-wash)', 'var(--seal)', 0.6],
    [304, '接受 ✓', 'var(--seal-wash)', 'var(--seal)', 0.9],
    [430, '拒绝 → 残差采样', '#f3ecdc', '#cbb98d', 1.2],
    [556, 'bonus 位不用', 'none', 'var(--rule)', null],
  ];
  acc.forEach(([x, lab, fill, stroke, lit], i) => {
    const inner = `<rect x="${x}" y="270" width="112" height="30" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="1.1"${i === 4 ? ' stroke-dasharray="3 3"' : ''}/>` +
      `<text class="dia-sub dia-mono" x="${x + 56}" y="289" text-anchor="middle" font-size="9" fill="var(--ink)">${lab}</text>`;
    body.push(lit !== null ? `<g class="sig-cell" style="--i:${lit}">${inner}</g>` : inner);
  });
  body.push(cap(52, 320, '产出 = n 个 draft token + 1 个 target token（n≤γ）· 接受规则 min(1, p/q) + 残差 norm(max(0, p−q)) → 输出分布严格 = p（命题 1，任意 q）', { size: 9 }));

  body.push(cap(36, 356, 'E[N_tok] = (1−α⁵)/(1−α) = 3.36 token/轮（含 bonus，式 5）· E[K] = 2.36（式 6 = DSpark accepted 口径）· 两口径恒差 1', { mono: true, size: 9.5 }));
  body.push(cap(36, 374, 'F = E[N_tok]/(γc+1) = 3.36/1.4 = 2.40 —— 每 token 墙钟降到 42%；verify 5 token 计算仅 0.087 ms（4.97 ms 步时下界的 1.7%）→ 并行 verify 几乎不增墙钟', { size: 9.5 }));
  body.push(cap(36, 392, '最优 γ=6（F≈2.47）：分子随 γ 饱和、分母线性涨；存在 γ 使 F>1 当且仅当 α>c（命题 3）—— c≈0 路线（n-gram/检索）低 α 也正收益', { size: 9.5 }));

  return scene(720, 406, body, { label: '一轮投机解码：draft 模型串行猜四个 token，target 一次前向并行验证五个位置，逐位拒绝采样接受前缀并在首个错误处残差重采样，期望产出 3.36 个 token' }).svg();
}

/* ============================================================
 * 4.4 §5（block 1）— token tree + tree mask 可见集 + 覆盖率
 * 原文：top-2 展开两层 7 节点 ≡ 4 路径；a2→{root,a1,a2}；
 *       覆盖率 68/77/81/84/85%；接受路径 KV 提交、拒绝回滚
 * ============================================================ */
function tokenTreeVerify() {
  const UID = 'i4k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'token tree 验证（expansion top-2 展开两层）：一次前向打 7 节点 ≡ 4 条路径各自 causal 验证（SpecInfer）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, ts = 11) => {
    const n = node({ id, x, y, w, h, role, title, mono: true, titleSize: ts });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const root = mk('root', 48, 150, 110, 44, 'cache', 'root', 12);
  body.push(cap(103, 210, '已验证前缀', { size: 8.5, anchor: 'middle' }));
  const a1 = mk('a1', 226, 84, 76, 36, 'data', 'a₁');
  const b1 = mk('b1', 226, 224, 76, 36, 'data', 'b₁');
  const a2 = mk('a2', 366, 48, 70, 32, 'attn', 'a₂');
  const a3 = mk('a3', 366, 120, 70, 32, 'attn', 'a₃');
  const b2 = mk('b2', 366, 188, 70, 32, 'attn', 'b₂');
  const b3 = mk('b3', 366, 260, 70, 32, 'plain', 'b₃');
  const tree = [[root, a1], [root, b1], [a1, a2], [a1, a3], [b1, b2], [b1, b3]];
  tree.forEach(([a, b], i) => {
    const vert = Math.abs(b.left[1] - a.right[1]) > Math.abs(b.left[0] - a.right[0]);
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.4, delay: i * 0.18, uid: UID, eid: `t${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], vert ? 0 : 1, vert ? 1 : 0));
  });
  // 接受路径高亮（root→a1→a2）：朱砂虚线流动
  const hit = ['M 158 166 C 190 160, 196 106, 226 102', 'M 302 96 C 330 90, 338 70, 366 64'];
  hit.forEach((d, i) => {
    body.push(`<path class="sig-dashflow" d="${d}" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
    body.push(pathPulse(d, { dur: 1.5, delay: 1.2 + i * 0.4 }));
  });
  body.push(cap(250, 44, '例：接受路径 root-a₁-a₂', { size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(401, 302, '拒绝分支 KV 回滚', { size: 8, anchor: 'middle' }));
  body.push(cap(103, 236, '4 条路径：root-a₁-a₂ /', { size: 8, anchor: 'middle' }));
  body.push(cap(103, 248, 'root-a₁-a₃ / root-b₁-b₂ /', { size: 8, anchor: 'middle' }));
  body.push(cap(103, 260, 'root-b₁-b₃', { size: 8, anchor: 'middle' }));

  // 右：可见集卡
  body.push(zone({ x: 470, y: 44, w: 214, h: 118, label: 'tree mask 可见集 = 祖先 + 自身' }));
  body.push(cap(486, 84, 'a₂ → {root, a₁, a₂}', { mono: true, size: 10 }));
  body.push(cap(486, 102, 'b₃ → {root, b₁, b₃}', { mono: true, size: 10 }));
  body.push(cap(486, 124, '不含兄弟分支（b₁/b₂/b₃ ∉ a₂ 可见集）', { size: 8.5 }));
  body.push(cap(486, 140, '位置编码 = 路径深度（RoPE 相对位置一致）', { size: 8.5 }));
  body.push(cap(486, 154, '共享前缀 KV 已缓存 → 命题 4 逐层归纳等价', { size: 8.5 }));

  // 覆盖率条（Table 1）：值标在条内，避免与注记打架
  body.push(cap(470, 186, 'top-k 覆盖率（LLaMA-7B · Alpaca）', { size: 9 }));
  body.push(cap(470, 198, 'k 1→2 +9pp、4→5 +1pp：top-2/3 已够', { size: 8 }));
  const cov = [[1, 68], [2, 77], [3, 81], [4, 84], [5, 85]];
  cov.forEach(([k, v], i) => {
    const x = 478 + i * 40, h = (v - 60) * 2.4;
    body.push(`<g class="sig-bar" style="--i:${(0.3 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${i === 1 ? 'seal' : 'soft'}" x="${x}" y="${266 - h}" width="28" height="${h}" rx="3"/></g>`);
    body.push(cap(x + 14, 280, `k=${k}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(cap(x + 14, 276 - h, `${v}%`, { mono: true, size: 8, anchor: 'middle', color: i === 1 ? 'var(--paper-raised)' : 'var(--ink)' }));
  });

  body.push(cap(36, 322, '命题 4：可见集恰为路径前缀 → 逐层 Q/K/V 与单独跑路径相同 → tree attention 输出 = 独立 causal 验证；mask 把树还原成逐路径验证', { size: 9.5 }));
  body.push(cap(36, 340, '接受两条路：greedy VerifyGreedy 沿「子 token = target argmax」边下行取最长匹配前缀；stochastic MSS 多候选拒绝采样（残差更新，分布保持）', { size: 9.5 }));
  body.push(cap(36, 358, 'KV 语义：树前向为所有节点写 KV —— 接受路径继承共享前缀块、拒绝分支独占块归池（2.1 COW / 块表）；DFS 遍历更新共享 KV 免逐序列各存一份', { size: 9.5 }));

  return scene(720, 372, body, { label: 'token tree：根为已验证前缀，top-2 展开两层共七节点，tree mask 让每节点只见祖先加自身，一次前向等价四条路径独立验证；分支覆盖率边际递减' }).svg();
}

/* ============================================================
 * 4.5 §1（block 0）— EAGLE-1 feature 级自回归数据流
 * 原文：f̂_{i+1} = D(f_i, e_{i+1})；shifted embedding 消不确定性；
 *       单层 draft c≈1/32 → F≈2.99；head/embedding 冻结复用
 * ============================================================ */
function eagleFeatureFlow() {
  const UID = 'i4l';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'EAGLE-1 feature 级自回归：draft 预测 target 下一位置 feature，shifted token 消解采样不确定性', { size: 12, color: 'var(--ink-soft)' }));

  const TR = node({ id: 'TR', x: 36, y: 56, w: 180, h: 52, role: 'proj', title: 'target trunk（32 层）', sub: 'verify 时已算 · draft 不重跑', titleSize: 11.5, subSize: 8.5 });
  const FI = node({ id: 'FI', x: 36, y: 156, w: 180, h: 52, role: 'data', title: 'f_i', sub: '倒数第二层 feature（非末层）', mono: true, titleSize: 13, subSize: 8.5 });
  const XI = node({ id: 'XI', x: 36, y: 256, w: 180, h: 52, role: 'data', title: 'x_{i+1}', sub: '上一步实际采出的 token', mono: true, titleSize: 13, subSize: 8.5 });
  const EM = node({ id: 'EM', x: 268, y: 256, w: 160, h: 52, role: 'norm', title: 'embedding e_{i+1}', sub: 'shifted · 冻结复用 target 表', mono: true, titleSize: 11, subSize: 8 });
  const D = node({ id: 'D', x: 292, y: 150, w: 190, h: 64, role: 'attn', title: 'draft 网络 D（单层）', sub: 'f_hat(i+1) = D(f_i, e_i+1)', mono: true, titleSize: 12, subSize: 8.5 });
  const HD = node({ id: 'HD', x: 540, y: 150, w: 144, h: 64, role: 'proj', title: 'target LM head', sub: '冻结复用', titleSize: 12, subSize: 9 });
  const XO = node({ id: 'XO', x: 540, y: 256, w: 144, h: 52, role: 'data', title: 'x_{i+2}', sub: '~ head(f_hat(i+1))', mono: true, titleSize: 13, subSize: 9 });
  [TR, FI, XI, EM, D, HD, XO].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(TR, FI, { nodes, portFrom: TR.bottom, portTo: FI.top, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'tf' }));
  body.push(arrowHead(FI.top[0], FI.top[1], 0, 1));
  body.push(edge(XI, EM, { nodes, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'xe' }));
  body.push(arrowHead(EM.left[0], EM.left[1], 1, 0));
  body.push(edge(FI, D, { nodes, pulse: true, dur: 1.3, delay: 0.6, uid: UID, eid: 'fd' }));
  body.push(arrowHead(D.left[0], D.left[1], 1, 0));
  body.push(edge(EM, D, { nodes, portFrom: EM.top, portTo: D.bottom, pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: 'ed' }));
  body.push(arrowHead(D.bottom[0], D.bottom[1], 0, -1));
  body.push(edge(D, HD, { nodes, pulse: true, dur: 1.3, delay: 1.2, uid: UID, eid: 'dh' }));
  body.push(arrowHead(HD.left[0], HD.left[1], 1, 0));
  body.push(edge(HD, XO, { nodes, portFrom: HD.bottom, portTo: XO.top, pulse: true, dur: 1.1, delay: 1.6, uid: UID, eid: 'hx' }));
  body.push(arrowHead(XO.top[0], XO.top[1], 0, 1));
  // 回灌回路（x=440 避开 embedding 卡）
  const fb = 'M 612 308 V 330 H 440 V 214';
  body.push(`<path class="dia-edge dash" d="${fb}"/>`);
  body.push(pathPulse(fb, { dur: 2.2, delay: 2.0 }));
  body.push(arrowHead(440, 214, 0, -1));
  body.push(cap(452, 326, '回灌下一步（feature 级自回归链）', { size: 9 }));
  body.push(cap(280, 228, '唯一训练部分（单层 decoder layer）', { size: 8.5, anchor: 'end' }));
  body.push(cap(230, 140, '仅凭 f_i 不知 x_{i+1} 采成什么 → 对所有可能取平均、不确定性回来；shifted embedding 消解它', { size: 8.5 }));

  body.push(cap(36, 356, '成本账：单层 draft c ≈ 1/L = 1/32 ≈ 0.031 → F = 3.36/(4×0.031+1) ≈ 2.99（对照独立小模型 c=0.1 的 2.40；feature 级 α 通常更高，两者叠加）', { size: 9.5 }));
  body.push(cap(36, 374, 'EAGLE-2：confidence（路径 draft 概率积）近似接受率、动态剪枝 top-m（前缀封闭）；EAGLE-3：training-time test + 多层 feature 融合（fc=3×hidden）', { size: 9.5 }));
  body.push(cap(36, 392, '论文实测：LLaMA2-Chat 70B 2.7–3.5× 延迟加速、吞吐翻倍、分布保持 —— EAGLE 不改接受规则，仍是 4.4 命题 1 的标准拒绝采样', { size: 9.5 }));

  return scene(720, 406, body, { label: 'EAGLE-1 数据流：target 倒数第二层 feature 配上一步实际采出 token 的 shifted embedding，喂单层 draft 网络预测下一 feature，经冻结 LM head 出 token 并回灌' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/01-continuous-batching',
    replacesBlock: 0,
    title: 'static vs continuous 时间线：空转/队头阻塞 vs 完成即离开、到达即补位',
    svg: staticVsContinuous,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/01-continuous-batching',
    replacesBlock: 1,
    title: 'selective batching：Flatten 合批 × Split 逐序列 attention × Merge',
    svg: selectiveBatching,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/01-continuous-batching',
    replacesBlock: 2,
    title: '每步循环：waiting/running 双预算准入 × 抢占回路 × schedule→execute→update',
    svg: stepLoop,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/02-scheduler-and-preemption',
    replacesBlock: 0,
    title: '三队列架构：准入 × recompute 重排队 × swap 换出换回',
    svg: threeQueues,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/02-scheduler-and-preemption',
    afterHeading: /^## 2\. /,
    title: 'recompute vs swap：一次 prefill vs 往返 KV × β* 临界带宽递减',
    svg: recomputeVsSwap,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/02-scheduler-and-preemption',
    replacesBlock: 1,
    title: 'FastServe skip-join MLFQ：按 t_init 入队 × 周期提升 × proactive swap',
    svg: fastserveMlfq,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/03-chunked-prefill',
    replacesBlock: 0,
    title: 'stall 时间线：781 ms 尖刺 52× vs 混合批 TBT ≈ 20 ms',
    svg: stallTimeline,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/03-chunked-prefill',
    replacesBlock: 1,
    title: '混合批 shape：GEMM 合批 × attention 逐序列 × piggyback 三相账',
    svg: mixedBatchShape,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/03-chunked-prefill',
    replacesBlock: 2,
    title: 'DP all-reduce MIN：组预算取最紧 rank × 同预算不同前缀解不同 q',
    svg: dpQuotaMin,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/04-speculative-decoding',
    replacesBlock: 0,
    title: '一轮 draft-verify：γ 串行 draft × 1 并行 verify × 接受前缀 + 残差',
    svg: draftVerifyRound,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/04-speculative-decoding',
    replacesBlock: 1,
    title: 'token tree：7 节点一次前向 ≡ 4 路径 causal × 可见集 = 祖先 + 自身',
    svg: tokenTreeVerify,
  },
  {
    dim: 'inference',
    slug: '04-scheduling-and-batching/05-speculative-systems',
    replacesBlock: 0,
    title: 'EAGLE-1 feature 自回归：f_i × shifted embedding → 单层 draft → 冻结 head',
    svg: eagleFeatureFlow,
  },
];
