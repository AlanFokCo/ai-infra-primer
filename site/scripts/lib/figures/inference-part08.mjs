// inference Part 8（08-distributed-and-pd）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part07.mjs / inference-part05.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言围栏（8.3 cpp / 8.5 text）管线逐字保留，不画。
// 数值口径：V4.1 MLA = 512 元素（448 latent + 64 rope，X-1 跨波裁定），NOT 576（V3 几何）；
//           BW* = 0.376 GB/s（V3 FP8）；moe_lb 1.074/1.045/1.090；NIXL 18 插件；TP 每层 2 次（非 4 非 1）。

import { scene, node, edge, zone, cap as kitCap, bars, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 覆盖 cap() 的 fill 属性 —— 显式 color 补 inline style 提权（W2 finding，不改 kit API）。
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

const sealDash = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="none" stroke="var(--seal)" stroke-width="1.3" stroke-dasharray="4 4"/>`;

/* ============================================================
 * 8.1 §2（block 0）— TP serving 前向归约位点：每层 2 次 × L（签名）
 * 原文：f/g 共轭前向-only 位置图；命题 1 每层恰 2 处、每步 2L=80；
 *       M_msg=2Bd=614,400 B；τ_TP=80×16.19=1295.6 μs≈1.30 ms，带宽项 7.4% 延迟主导
 * ============================================================ */
function tpForwardSites() {
  const UID = 'i8a';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'serving 前向-only：每层归约位点恰 2 处（g×2），每步 2L = 80 次 all-reduce —— 不是训练的 4 次也不是 1 次', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.8) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const chain = (arr, delays, eid) => arr.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(arr[i - 1], n, { nodes, pulse: true, dur: 1, delay: delays[i - 1], uid: UID, eid: `${eid}${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 40, w: 648, h: 168, label: '每层前向位置图（每卡视角 · decode 每序列 1 token · 复制区段：残差 / RMSNorm 全量）' }));
  const x = mk('x', 48, 104, 84, 56, 'data', 'x', '[B, d_model]', 12, 7);
  const f1 = mk('f1', 148, 64, 40, 44, 'plain', 'f', '恒等', 11, 6.5);
  const qkv = mk('qkv', 200, 64, 96, 44, 'proj', 'QKV 列切', '逐 head 分', 10, 6.8);
  const attn = mk('attn', 308, 64, 104, 44, 'attn', '逐 head attention', 'H_kv=1 latent 共享', 9, 6.3);
  const out = mk('out', 424, 64, 104, 44, 'proj', 'out proj 行切', '部分和', 10, 6.8);
  const g1 = mk('g1', 540, 64, 44, 44, 'op', 'g', 'AR①', 11, 6.5);
  const p1 = mk('p1', 596, 64, 72, 44, 'norm', '(+残差)', null, 9);
  body.push(sealDash(540, 64, 44, 44));
  const f2 = mk('f2', 148, 140, 40, 44, 'plain', 'f', '恒等', 11, 6.5);
  const w1 = mk('w1', 200, 140, 96, 44, 'proj', 'W1 列切', 'gate/up', 10, 6.8);
  const phi = mk('phi', 308, 140, 104, 44, 'op', 'φ 逐位置', 'SwiGLU / MoE 路由', 9.5, 6.3);
  const w2 = mk('w2', 424, 140, 104, 44, 'proj', 'W2/MoE down 行切', '部分和', 8.5, 6.8);
  const g2 = mk('g2', 540, 140, 44, 44, 'op', 'g', 'AR②', 11, 6.5);
  const p2 = mk('p2', 596, 140, 72, 44, 'norm', '(+残差)', null, 9);
  body.push(sealDash(540, 140, 44, 44));

  body.push(edge(x, f1, { nodes, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'a0' }));
  body.push(arrowHead(f1.left[0], f1.left[1], 1, 0));
  chain([f1, qkv, attn, out, g1, p1], [0.2, 0.4, 0.6, 0.8, 1.0], 'a');
  body.push(edge(x, f2, { nodes, pulse: true, dur: 1.1, delay: 0.1, uid: UID, eid: 'b0' }));
  body.push(arrowHead(f2.left[0], f2.left[1], 1, 0));
  chain([f2, w1, phi, w2, g2, p2], [0.3, 0.5, 0.7, 0.9, 1.1], 'b');
  body.push(cap(52, 196, '反向通信：无（推理不执行反向图）—— f 的归约永不触发，前向只有 g 触发', { size: 8 }));

  body.push(zone({ x: 36, y: 224, w: 648, h: 110, label: 'decode 延迟账（t=8 · B=60 · L=40 · NVLink 900 GB/s · τ_lat=15 μs 教学值）' }));
  mk('s1', 52, 250, 180, 48, 'cache', '每步 2L = 80 次归约', 'M_msg = 2Bd = 614,400 B ≈ 0.61 MB', 9.5, 6.3);
  mk('s2', 248, 250, 200, 48, 'attn', '每步 τ_TP ≈ 1.30 ms', '80 × (15 + 1.19) μs = 1295.6 μs', 10, 6.8);
  mk('s3', 464, 250, 204, 48, 'plain', '延迟项主导', 'B=1 消息缩 60× 时间只降 7.25%', 10, 6.5);
  body.push(`<rect class="dia-bar dia-bar-ink" x="52" y="310" width="570" height="10" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="625" y="310" width="43" height="10" rx="3"/>`);
  body.push(cap(337, 318, '延迟项 1.2 ms（92.6%）', { size: 6.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(646, 318, 'BW 7.4%', { size: 6, anchor: 'middle', color: 'var(--paper-raised)' }));

  body.push(cap(36, 352, '训练 4 次 = 前向 g×2 + 反向 f×2；去掉 backward 剩前向的 2（attention out-proj + MLP/MoE down-proj 各 1）——「减半再误减」是坑', { size: 9.5 }));
  body.push(cap(36, 370, 'SP：原语换 reduce-scatter + all-gather 对，次数翻倍但位点仍每层 2 处；vocab parallel embedding 另加 1 处前向归约（本命题不计）', { size: 9.5 }));
  body.push(cap(36, 388, '跨节点 IB 50 GB/s：带宽项 21.5 μs 超延迟项 → TP 留 scale-up domain 内；DeepInfer 装得下 + SLA 可达不付 TP（doc L69）', { size: 9.5 }));

  return scene(720, 410, body, { label: 'TP serving 前向位置图：复制区段经 f 恒等分出 attention 与 MLP 两条支路，各在行切投影后贴一个 g 前向 all-reduce，每层两处每步八十次；延迟账带宽项仅百分之七点四，延迟项主导' }).svg();
}

/* ============================================================
 * 8.1 §3（block 1）— decode PP token 级流水：层间 p−1 次 vs TP 层内 2L 次
 * 原文：p=2 mb1/mb2 每格 t_s；TPOT_PP = p·t_s+(p−1)τ_p2p，开销 5 μs vs TP 1.30 ms；
 *       PP 买容量；命题 2 KV 读 PP 沿层切 / DP 沿请求切 / TP(MLA) 复制
 * ============================================================ */
function ppTokenPipeline() {
  const UID = 'i8b';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'decode PP token 级流水：层间只送 p−1 个边界 vs TP 层内归约 2L 次 —— 延迟开销差两个量级，PP 买的是容量', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 150, label: 'p=2 · 在飞 mb1/mb2 · 每格 t_s（满流水两 stage 同忙）' }));
  const tx = [120, 230, 340, 450, 560];
  tx.forEach((x, i) => body.push(cap(x + 50, 72, `t${i}`, { size: 8, anchor: 'middle', mono: true })));
  body.push(cap(40, 95, 'stage1', { size: 8.5, weight: '650' }));
  body.push(cap(40, 108, 'L/2 层', { size: 6.5 }));
  body.push(cap(40, 155, 'stage2', { size: 8.5, weight: '650' }));
  body.push(cap(40, 168, 'L/2 层', { size: 6.5 }));
  const cell = (x, y, mb, tone) => {
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="${x}" y="${y}" width="100" height="30" rx="4"/>`);
    body.push(cap(x + 50, y + 19, mb, { size: 8, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  };
  ['mb1', 'mb2', 'mb1', 'mb2', 'mb1'].forEach((mb, i) => cell(tx[i], 76, mb, mb === 'mb1' ? 'seal' : 'ink'));
  ['mb1', 'mb2', 'mb1', 'mb2'].forEach((mb, i) => cell(tx[i + 1], 136, mb, mb === 'mb1' ? 'seal' : 'ink'));
  for (let i = 0; i < 4; i++) {
    const a = [tx[i] + 50, 106], b = [tx[i + 1] + 50, 136];
    const d = `M${a[0]} ${a[1]} C${a[0]} ${a[1] + 14}, ${b[0]} ${b[1] - 14}, ${b[0]} ${b[1]}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1, delay: 0.2 + i * 0.3, r: 3 }));
    body.push(arrowHead(b[0], b[1], 0, 1));
  }
  body.push(cap(40, 126, 'P2P', { size: 7, mono: true }));
  body.push(cap(390, 182, 'mb1 每 token 关键路径 = 2·t_s + 1·τ_p2p（遍历 p 段 + p−1 次边界送）', { size: 8.5, anchor: 'middle' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.8) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  mk('c1', 36, 204, 200, 60, 'cache', 'TPOT_PP = p·t_s + (p−1)τ_p2p', 'p=2：开销 = 1×5 = 5 μs', 8.5, 7);
  mk('c2', 252, 204, 200, 60, 'attn', 'vs TP 1.30 ms/step', '层间 1 跳 vs 层内 80 次归约 · ≈260×', 10, 6.8);
  mk('c3', 468, 204, 216, 60, 'proj', 'PP 买容量非单请求延迟', '每卡持 1/p 权重+KV · 聚合 HBM BW p·BW 用满', 9, 6.3);
  body.push(cap(36, 282, 'τ_p2p = τ_hop + 2Bd/BW：B=64 消息 655,360 B、带宽项 0.73 μs，教学值 ≈5 μs（1 跳、无 ring 的 2(t−1) 跳）', { size: 8.5 }));

  body.push(zone({ x: 36, y: 296, w: 648, h: 76, label: '命题 2：KV 读的切分 vs 复制（V4.1 bf16 κ=1024 B/token/层 · L=40 · T=4096 · B=64）' }));
  mk('k1', 52, 326, 190, 40, 'proj', 'PP 沿层切', '(B/m)κ(L/p)T = 2.5 GiB/stage', 9, 6.3);
  mk('k2', 258, 326, 190, 40, 'attn', 'DP 沿请求切', '(B/d)κLT = 1.34 GB/rank', 9, 6.3);
  mk('k3', 464, 326, 204, 40, 'cache', 'TP（MLA H_kv=1）复制', 'BκLT = 10.74 GB/rank · 与 t 无关', 8, 6.3);

  body.push(cap(36, 388, 'prefill PP 气泡 (p−1)/(m+p−1) 与训练 GPipe 同形；无 backward 激活驻留、m 可开大 —— 气泡 vs tail TTFT 是同一枚硬币', { size: 9.5 }));
  return scene(720, 402, body, { label: 'decode PP token 级流水：两个 stage 交替处理 mb1 mb2，stage 边界点对点送中间张量，每 token 关键路径两段步时加一次边界传输仅五微秒，远小于 TP 每步一点三毫秒；PP 沿层切 KV 读而 TP 在 MLA 上复制' }).svg();
}

/* ============================================================
 * 8.1 §4（afterHeading）— EP dispatch/combine 延迟账 × 可重叠上界（签名）
 * 原文：每 MoE 层 2 次 a2a、每步 2L_moe=80；逐 peer 30 KiB / 共享出口 4.30 μs（FP8/IB）；
 *       bf16/NVLink 0.48 μs → 未重叠上界 1.24 ms；Mega MoE/TBO 重叠；木桶 max_j R_j → EPLB moe_lb
 * ============================================================ */
function epDispatchCombine() {
  const UID = 'i8c';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'EP serving：每 MoE 层 dispatch + combine 两次 all-to-all —— 逐 peer KB 级延迟主导，且可融合重叠（vs TP 全暴露）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 400, h: 236, label: 'MoE 层数据流（B=64 · d=e=8 · k=6 · E=384+1）' }));
  const tok = mk('tok', 52, 68, 150, 40, 'data', 'hidden [B/d, d_model]', '8 请求/rank', 8.5, 6.5);
  const disp = mk('disp', 52, 124, 360, 30, 'op', 'dispatch（all-to-all）：每 token 发往 k=6 个 expert 所在 rank', null, 8.5);
  const ex = [mk('e0', 52, 170, 110, 40, 'proj', 'r0 experts', '48 + EPLB 副本', 8.5, 6),
              mk('e3', 182, 170, 110, 40, 'proj', 'r3 experts', '48 + EPLB 副本', 8.5, 6),
              mk('e7', 312, 170, 110, 40, 'proj', 'r7 experts', '48 + EPLB 副本', 8.5, 6)];
  const comb = mk('comb', 52, 226, 360, 30, 'op', 'combine（all-to-all）：输出发回 home rank（加权求和在接收后本地）', null, 8.5);
  body.push(edge(tok, disp, { nodes, portFrom: tok.bottom, portTo: disp.top, pulse: true, dur: 1, delay: 0, uid: UID, eid: 'd0' }));
  body.push(arrowHead(disp.top[0], disp.top[1], 0, 1));
  ex.forEach((n, i) => {
    body.push(edge(disp, n, { nodes, portFrom: [n.x + n.w / 2, disp.y + disp.h], portTo: n.top, pulse: true, dur: 1, delay: 0.3 + i * 0.2, uid: UID, eid: `d${i + 1}` }));
    body.push(arrowHead(n.top[0], n.top[1], 0, 1));
    body.push(edge(n, comb, { nodes, portFrom: n.bottom, portTo: [n.x + n.w / 2, comb.y], pulse: true, dur: 1, delay: 0.9 + i * 0.2, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(n.x + n.w / 2, comb.y, 0, 1));
  });
  body.push(cap(52, 268, 'EPLB 冗余副本承载热 expert · 木桶 max_j R_j 落成 tail TPOT', { size: 7.5 }));

  body.push(zone({ x: 452, y: 40, w: 232, h: 236, label: '每步暴露延迟账' }));
  mk('r1', 468, 68, 200, 44, 'cache', '逐 peer 30 KiB', '245,760/8 B · 单份 chunk 0.61 μs', 10, 6.5);
  mk('r2', 468, 120, 200, 44, 'attn', '共享出口 BW 项 4.30 μs', 'FP8/IB 50 GB/s · τ_a2a 15 μs 主导（BW 22%）', 8, 5.8);
  mk('r3', 468, 172, 200, 44, 'proj', '每步 2L_moe = 80 次 a2a', 'bf16/NVLink 0.48 μs → 未重叠上界 1.24 ms', 8.5, 5.8);
  mk('r4', 468, 224, 200, 40, 'norm', '重叠 → 暴露压向 0', 'Mega MoE 融合 · TBO/SBO 双批', 9, 6.3);

  body.push(zone({ x: 36, y: 292, w: 648, h: 92, label: 'EPLB 残差偏斜 moe_lb（slowest rank / mean · EPLB 已生效 · 7.4 §3）' }));
  body.push(`<line x1="180" y1="316" x2="180" y2="374" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`);
  body.push(cap(180, 380, '1.0 理想', { size: 6.5, anchor: 'middle' }));
  [['prefill EP4DP4', 1.074, 322], ['prefill EP2DP2', 1.045, 342], ['decode EP64', 1.090, 362]].forEach(([lab, v, y]) => {
    body.push(cap(52, y + 10, lab, { size: 8 }));
    const w = (v - 1) * 2000;
    body.push(`<g class="sig-bar" style="--i:${(0.2 + (y - 322) / 100).toFixed(1)}"><rect class="dia-bar dia-bar-seal" x="180" y="${y}" width="${w.toFixed(0)}" height="12" rx="3"/></g>`);
    body.push(cap(180 + w + 6, y + 10, v.toFixed(3), { size: 7.5, mono: true, color: 'var(--seal-deep)' }));
  });

  body.push(cap(36, 402, '两种链路读数勿混：0.61 μs = 逐 peer 单份 chunk 链路时间；4.30 μs = 每 rank 全部字节过共享出口聚合（fabric 各 peer 并行，不乘 e−1）', { size: 9.5 }));
  body.push(cap(36, 420, 'TP 1.30 ms 基本全暴露（层内屏障）vs EP 1.24 ms 未重叠上界可融合重叠 —— §5 DP attention 选 EP 而非 MoE TP 的账面理由之一', { size: 9.5 }));
  body.push(cap(36, 438, '拓扑：EP 域收进 scale-up NVLink；Wide-EP decode e=64–128 跨超节点（整超节点单 EP 组）；跨域 RDMA low-latency kernel 绕两阶段转发', { size: 9.5 }));
  return scene(720, 458, body, { label: 'EP serving 数据流：hidden 经 dispatch all-to-all 发往六个 expert 所在 rank，专家计算后 combine 发回 home rank；每步八十次未重叠上界一点二四毫秒可融合重叠压向零；EPLB 残差偏斜一点零七四到一点零九零' }).svg();
}

/* ============================================================
 * 8.1 §5（block 2）— DP attention × #DP=#EP 边界对齐（签名）
 * 原文：world=8 d=e=8 a=1；attention 0 次集合通信；dispatch/combine 2 次 a2a；
 *       home rank = dispatch 源 = combine 目的 → 边界 0；三形式字节 91.8:45.9:39.3；
 *       KV 读 10.74→1.34 GB（8×）；T=131072 320 GiB TP 不可行 / DP 40 GiB
 * ============================================================ */
function dpAttentionBoundary() {
  const UID = 'i8d';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'DP attention × #DP=#EP：attention 按请求切（KV 本地）、MoE 按 expert 切 —— home rank 对齐，边界 0 次额外通信', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 72, label: 'attention（DP, a=1）：各 rank 独立算 B/8=8 条请求 · KV 本地不复制 —— 0 次集合通信' }));
  const ar = [];
  for (let i = 0; i < 8; i++) ar.push(mk(`a${i}`, 48 + i * 80, 66, 72, 36, 'attn', `r${i}`, '8 reqs', 10, 6.5));

  body.push(`<rect x="48" y="140" width="632" height="32" rx="10" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(cap(364, 160, 'EP all-to-all ×2：dispatch（每 token 发往 k=6 expert rank）↓ · combine（输出发回 home rank）↑', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  for (let i = 0; i < 8; i++) {
    const cx = 48 + i * 80 + 36;
    body.push(`<path class="dia-edge" d="M${cx} 102 V140"/>`);
    body.push(`<path class="dia-edge" d="M${cx} 172 V200"/>`);
    if (i % 2 === 0) body.push(pathPulse(`M${cx} 102 V140`, { dur: 1, delay: 0.2 + i * 0.12, r: 3 }));
    if (i % 2 === 1) body.push(pathPulse(`M${cx} 200 V172`, { dur: 1, delay: 1.2 + i * 0.12, r: 3 }));
  }

  body.push(zone({ x: 36, y: 200, w: 648, h: 72, label: 'MoE（EP）：expert 切 8 份 · 每 rank E/e=48 + EPLB 冗余副本 —— dispatch + combine 2 次 a2a' }));
  for (let i = 0; i < 8; i++) mk(`e${i}`, 48 + i * 80, 226, 72, 36, 'proj', `e${i}`, '48 experts', 10, 6);
  body.push(cap(360, 288, 'combine 目的 = home rank = 下一层 attention 所在 rank → 边界 0 次额外 all-gather / reduce-scatter（命题 3）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 300, w: 220, h: 92, label: '每 rank 每步字节（MB）' }));
  [['纯 TP', 91.8, 'ink', 334, 0.2], ['DP+MoE TP', 45.9, 'soft', 356, 0.4], ['DP+EP', 39.3, 'seal', 378, 0.6]].forEach(([lab, v, tone, y, di]) => {
    body.push(cap(52, y - 4, `${lab} ${v}`, { size: 7 }));
    body.push(`<g class="sig-bar" style="--i:${di}"><rect class="dia-bar dia-bar-${tone}" x="52" y="${y}" width="${(v * 1.6).toFixed(0)}" height="12" rx="3"/></g>`);
  });
  body.push(zone({ x: 272, y: 300, w: 200, h: 92, label: 'KV 读/步（T=4096）' }));
  body.push(cap(288, 334, 'TP: 10.74 GB → 3.21 ms', { size: 8.5, mono: true }));
  body.push(cap(288, 352, 'DP: 1.34 GB → 0.40 ms', { size: 8.5, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(288, 372, '每卡 KV 读降 8×', { size: 9, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(288, 388, 'T=131072: 320 GiB TP 超 HBM / DP 40 GiB', { size: 6.3 }));
  body.push(zone({ x: 488, y: 300, w: 196, h: 92, label: '暴露延迟（ms）' }));
  body.push(cap(504, 334, '纯 TP 1.30 全暴露', { size: 8.5 }));
  body.push(cap(504, 352, 'DP+MoE TP 0.65 减半', { size: 8.5 }));
  body.push(cap(504, 370, 'DP+EP 1.24 未重叠上界', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(504, 388, '次数不减，赢在重叠 + KV', { size: 7 }));

  body.push(cap(36, 410, 'MoE TP 形式边界 all-gather + reduce-scatter = 1 次 all-reduce 等价（SGLang v0.4 原始形态：MoE 前 AG / MoE 后 redistribute）', { size: 9.5 }));
  body.push(cap(36, 428, '#DP=#EP（doc L33 逐字）：prefill EP2–8/DP2–8（HBM 最小）· decode EP64–128/DP64–128（Wide-EP 最大）—— 耦合扩张 B=d·B_local', { size: 9.5 }));
  body.push(cap(36, 446, 'SGLang v0.4 实测：8×H100 DeepSeek-Coder-V2-Instruct-FP8 decode 吞吐 1.9× —— 三笔账（KV 容量 / KV 读 BW / 可重叠通信）的经验兑现', { size: 9.5 }));
  return scene(720, 470, body, { label: 'DP attention 边界对齐：八个 rank 各自独立算八条请求的 attention 零集合通信，经 dispatch 与 combine 两次 all-to-all 穿过 MoE 专家层，home rank 全程一致边界零额外通信；每卡 KV 读降八倍，三形式字节比九十一点八比四十五点九比三十九点三' }).svg();
}

/* ============================================================
 * 8.2 §1（block 0）— 干扰时序：colocated TBT 尖刺 52× vs PD 两池无尖刺（签名）
 * 原文：32768 tok prefill 781 ms vs decode 15.1 ms（52×）；4096 tok 66.5 ms（4.4×）；
 *       P 池 compute-bound 小 batch / D 池 memory-bound 大 batch；决策耦合；命题 1 min
 * ============================================================ */
function pdInterference() {
  const UID = 'i8e';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, '一张卡伺候不了两个 SLO：32768-token prefill 劫持同批 decode —— TBT 尖刺 52×，PD 分离物理隔离两相', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 118, label: 'colocated：prefill 与 decode 共享同一组 GPU（迭代级调度 · batch 是执行原子单位）' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="72" width="12" height="26" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="67" y="72" width="560" height="26" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-soft" x="630" y="72" width="12" height="26" rx="3"/>`);
  body.push(cap(347, 89, '新请求整条 prefill 781 ms（32768 tok · 计算下界）', { size: 8, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(58, 108, 'decode 15.1', { size: 6, anchor: 'middle' }));
  body.push(cap(636, 108, 'decode 15.1', { size: 6, anchor: 'middle' }));
  body.push(cap(52, 128, 'r1 TBT:', { size: 8.5 }));
  mk('t1', 110, 108, 80, 32, 'plain', '15.1 ms', null, 9);
  const spike = mk('t2', 260, 104, 130, 40, 'attn', '781 ms', '尖刺 52×', 11, 7);
  body.push(sealDash(260, 104, 130, 40));
  mk('t3', 460, 108, 80, 32, 'plain', '15.1 ms', null, 9);
  body.push(`<path class="dia-edge" d="M190 124 H260"/>`);
  body.push(`<path class="dia-edge" d="M390 124 H460"/>`);
  body.push(pathPulse('M110 124 H540', { dur: 2.6, delay: 0.3 }));
  body.push(cap(560, 128, '4096 tok 也 4.4×', { size: 7.5 }));

  body.push(zone({ x: 36, y: 174, w: 648, h: 130, label: 'PD 分离：两池各自迭代 —— P compute-bound 小 batch · D memory-bound 大 batch' }));
  body.push(cap(52, 222, 'P 池', { size: 8.5, weight: '650' }));
  [['prefill r2', 100], ['prefill r3', 246], ['prefill r4', 392]].forEach(([t, x]) => {
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="206" width="140" height="24" rx="4"/>`);
    body.push(cap(x + 70, 222, t, { size: 8, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  });
  const kv = 'M170 230 C170 244, 170 248, 170 262';
  body.push(`<path class="dia-edge strong" d="${kv}"/>`);
  body.push(pathPulse(kv, { dur: 1.2, delay: 0.6 }));
  body.push(arrowHead(170, 262, 0, 1));
  body.push(cap(180, 250, 'KV(r2) + 首 token 元数据（传输账 §5）', { size: 7.5 }));
  body.push(cap(52, 278, 'D 池', { size: 8.5, weight: '650' }));
  [['decode r1,r2 · 15.1 ms', 100, 180], ['15.1 ms', 286, 100], ['15.1 ms', 392, 100], ['15.1 ms', 498, 100]].forEach(([t, x, w]) => {
    body.push(`<rect class="dia-bar dia-bar-seal" x="${x}" y="262" width="${w}" height="24" rx="4"/>`);
    body.push(cap(x + w / 2, 278, t, { size: 7.5, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  });
  body.push(cap(52, 300, 'r1 TBT: 无尖刺', { size: 8.5, color: 'var(--seal-deep)' }));
  [190, 336, 442, 548].forEach((x) => body.push(cap(x, 300, '15.1', { size: 7.5, anchor: 'middle', mono: true })));

  body.push(zone({ x: 36, y: 320, w: 648, h: 68, label: '决策耦合：两相偏好相反 —— 配置只能按更严的 SLO 选，另一相被迫 over-provision' }));
  mk('cp', 52, 346, 290, 32, 'proj', 'prefill：compute-bound → TP intra-op 压 TTFT', null, 8);
  mk('cd', 360, 346, 308, 32, 'attn', 'decode：memory-bound → 大 batch + inter-op 扩 TPOT', null, 8);

  body.push(cap(36, 402, '命题 1：G_co = min(G_TTFT, G_TPOT)；优先级调度也救不了（prefill 优先 → decode 排队 / decode 优先 → prefill 等待 + 纯 decode 批算力闲置）', { size: 9.5 }));
  body.push(cap(36, 420, '分离同时攻击两层：物理隔离消干扰（代价 KV 传输 §5）+ 两相各自独立选并行度与 batch（解除决策耦合）—— 实例 = 一份完整权重的资源单元', { size: 9.5 }));
  return scene(720, 438, body, { label: '干扰时序对照：colocated 一条七百八十一毫秒 prefill 劫持同批 decode 造成五十二倍 TBT 尖刺；PD 分离后 P 池连续 prefill 经 KV 传输喂 D 池混合批，r1 的 TBT 恒为十五点一毫秒无尖刺；两相并行度偏好相反决策耦合' }).svg();
}

/* ============================================================
 * 8.2 §2（afterHeading）— goodput 分解：min 约束 → 调和形 → 整数 replication（签名）
 * 原文：G_co=1.6（13B/A100/θ=90%）；g_P=5.6 g_D=10；G*=56/15.6=3.59；
 *       2P+1D R=10/3GPU=3.33（取整损失 7%，2.1×）；1P+2D 1.87；两级分解；90 Gbps 账
 * ============================================================ */
function goodputDecomp() {
  const UID = 'i8f';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'goodput 分解：解除 min 约束 → 两相各达单 SLO 极限 → 调和容量 3.59 → 整数 2P+1D 3.33 rps/GPU（2.1×）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 400, h: 230, label: 'attainment 示意（13B/A100 · θ=90% · 教学曲线非实测）' }));
  const X = (l) => 70 + l * 28.3;
  body.push(`<line x1="70" y1="86" x2="410" y2="86" stroke="var(--seal)" stroke-width="1" stroke-dasharray="4 4"/>`);
  body.push(cap(74, 82, 'θ = 90%', { size: 7, color: 'var(--seal-deep)' }));
  body.push(`<line class="dia-edge" x1="70" y1="230" x2="410" y2="230"/>`);
  body.push(`<line class="dia-edge" x1="70" y1="70" x2="70" y2="230"/>`);
  body.push(cap(66, 74, 'Attain', { size: 7.5, anchor: 'end' }));
  body.push(cap(410, 258, 'λ (req/s)', { size: 7.5, anchor: 'end' }));
  const curves = [
    { pts: [[0, 99], [1, 97], [1.6, 90], [2.6, 70], [4.1, 44], [6.1, 20], [8.1, 8]], tone: 'var(--ink-soft)', name: 'colocate', nx: 298, ny: 222 },
    { pts: [[0, 99], [3, 97], [5.6, 90], [7.1, 78], [9.1, 56], [11.1, 31]], tone: 'var(--seal)', name: 'P-only', nx: 383, ny: 196 },
    { pts: [[0, 99], [6, 98], [10, 90], [11.1, 81], [12, 72]], tone: '#aebcd2', name: 'D-only', nx: 410, ny: 122 },
  ];
  curves.forEach((c) => {
    const pts = c.pts.map(([l, a]) => `${X(l).toFixed(1)},${(70 + (100 - a) * 1.6).toFixed(1)}`).join(' ');
    body.push(`<polyline class="dia-edge strong" points="${pts}" style="stroke:${c.tone}"/>`);
    body.push(cap(c.nx, c.ny, c.name, { size: 7.5, anchor: 'end', color: c.tone }));
  });
  [[1.6, 'G_co=1.6', 0.3], [5.6, 'g_P=5.6', 0.6], [10, 'g_D=10', 0.9]].forEach(([l, lab, di]) => {
    body.push(`<g class="sig-cell" style="--i:${di}"><circle cx="${X(l).toFixed(1)}" cy="86" r="4" fill="var(--seal)" stroke="var(--seal-wash)" stroke-width="1.5"/></g>`);
    body.push(`<line x1="${X(l).toFixed(1)}" y1="230" x2="${X(l).toFixed(1)}" y2="235" stroke="var(--ink-faint)" stroke-width="1"/>`);
    body.push(cap(X(l), 246, lab, { size: 7, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  });

  body.push(zone({ x: 452, y: 40, w: 232, h: 230, label: 'per-GPU goodput（rps/GPU）' }));
  [['colocate 1.6', 1.6, 'soft', 76], ['2P+1D 3.33（2.1×）', 3.33, 'seal', 116], ['连续最优 G* 3.59', 3.59, 'ink', 156], ['1P+2D 1.87（瓶颈翻 P）', 1.87, 'warm', 196]].forEach(([lab, v, tone, y], i) => {
    body.push(cap(468, y - 6, lab, { size: 7.5 }));
    body.push(`<g class="sig-bar" style="--i:${(0.2 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${tone}" x="468" y="${y}" width="${(v * 38).toFixed(0)}" height="16" rx="3"/></g>`);
  });
  body.push(cap(468, 236, 'G* = g_P·g_D/(g_P+g_D) = 56/15.6', { size: 8, mono: true }));
  body.push(cap(468, 252, '调和形：< min(g_P,g_D)、不低于其一半', { size: 7.5 }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 286, w: 648, h: 74, label: '两级分解（Algorithm 1）+ 整数取整' }));
  const s1 = mk('s1', 52, 310, 180, 36, 'op', '每相枚举可行配置', 'simulator 二分搜 max goodput', 8.5, 6);
  const s2 = mk('s2', 252, 310, 180, 36, 'proj', 'replication 向上取整', '⌈R/(a_P·g_P)⌉ + ⌈R/(a_D·g_D)⌉', 8.5, 6);
  const s3 = mk('s3', 452, 310, 216, 36, 'cache', '2P+1D：R=10 rps / 3 GPU', '取整损失 ~7% · 仍 colocate 2.1×', 8.5, 6);
  [[s1, s2, 0.4], [s2, s3, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(52, 352, '传输可略前提：OPT-66B 512tok KV 1.13 GB × 10 rps = 90 Gbps ≪ IB 800 Gbps / NVLink 600 GB/s（带宽账结论非假设）', { size: 7.5 }));

  body.push(cap(36, 378, '配比是优化变量不是常数：1P+2D → R=min(5.6,20)=5.6、per-GPU 掉到 1.87 —— 瓶颈切到 prefill 侧（§4 xPyD 推导入口）', { size: 9.5 }));
  body.push(cap(36, 396, '低节点亲和（Algorithm 2）：placement 约束耦合两相可行集、分解前提失效须联合枚举；175B 一对 P+D 700 GB > 单节点 640 GB 显存约束', { size: 9.5 }));
  return scene(720, 420, body, { label: 'goodput 分解：attainment 曲线上 colocate 被更严 SLO 约束在一点六 rps，分离后 prefill 与 decode 各达五点六与十；调和容量三点五九，整数 2P+1D 三点三三为 colocate 二点一倍，反向 1P+2D 瓶颈翻转' }).svg();
}

/* ============================================================
 * 8.2 §4（afterHeading）— xPyD 配比：λ 与 TPOT 相消（签名）
 * 原文：n_P≥1.55（算力）；C=153.6 → n_D≥1.09（KV 带宽）；1.4:1 取整 1:1；
 *       命题 4 双重无关；扫描 agent 32.8 / MLA 0.29 / H200 0.76；doc L91 ~1:1 + 64P+64D
 * ============================================================ */
function xpydRatio() {
  const UID = 'i8g';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'xPyD 配比：λ 与 TPOT 精确相消 —— 只剩 workload 形状（T_in/T_out/T̄）× 硬件（F_P/β_D），不是 1:1 常数', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const chain = (arr, delays, eid) => arr.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(arr[i - 1], n, { nodes, pulse: true, dur: 1, delay: delays[i - 1], uid: UID, eid: `${eid}${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 40, w: 648, h: 76, label: 'P 侧：算力需求 → n_P（章脊 8B bf16 @ H100 · λ=10 · T_in=4096 · MFU 0.5 · ρ=0.8）' }));
  chain([
    mk('p1', 52, 66, 150, 36, 'data', 'λ·2N_act·T_in', '6.144e14 FLOPs/s', 9, 6.3),
    mk('p2', 222, 66, 150, 36, 'op', '÷ F_P·ρ_P', '3.958e14 FLOPs/s', 9, 6.3),
    mk('p3', 392, 66, 130, 36, 'cache', 'n_P ≥ 1.55', 'μ_P = 8.05 req/s/GPU', 10, 6.3),
  ], [0, 0.3], 'p');
  body.push(cap(540, 88, 'prefix 命中折减 T_in（本节全 miss）', { size: 6.8 }));

  body.push(zone({ x: 36, y: 132, w: 648, h: 76, label: 'D 侧：Little 并发 → KV 带宽需求 → n_D（T_out=512 · TPOT SLO 30 ms · κL=131072 B/token）' }));
  chain([
    mk('d1', 52, 158, 150, 36, 'data', 'C = λ·T_out·TPOT', '153.6 并发序列', 9, 6.3),
    mk('d2', 222, 158, 170, 36, 'op', 'C·κL·T̄ / TPOT', '2.92 TB/s（T̄=4352）', 9, 6.3),
    mk('d3', 412, 158, 130, 36, 'cache', 'n_D ≥ 1.09', '÷ β_D·ρ_D = 2.68 TB/s', 10, 6.3),
  ], [0.6, 0.9], 'd');
  body.push(cap(556, 180, '核对: 24.0+4.8=28.8 ≤ 30 ms', { size: 6.5 }));

  body.push(zone({ x: 36, y: 224, w: 648, h: 72, label: '配比 + 两个相消（命题 4：KV 流量口径下与 λ、TPOT 无关）' }));
  body.push(cap(114, 258, 'n_P 1.55', { size: 8, anchor: 'middle' }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-ink" x="52" y="262" width="124" height="16" rx="3"/></g>`);
  body.push(cap(222, 258, 'n_D 1.09', { size: 8, anchor: 'middle' }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="179" y="262" width="87" height="16" rx="3"/></g>`);
  body.push(cap(300, 274, '≈ 1.4:1 → 向上取整 2P+2D（1:1）', { size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(52, 286, 'λ 相消：两池需求同 ∝λ', { size: 8 }));
  body.push(cap(250, 286, 'TPOT 相消：C ∝TPOT × 读取频率 ∝1/TPOT', { size: 8 }));

  body.push(zone({ x: 36, y: 312, w: 648, h: 96, label: 'workload × 硬件扫描（n_P=1.55 不变 · n_D 条按比例）' }));
  [['chatbot 4096/512', 1.09, '1.09（1:1）', 338], ['agent 4096/8192', 32.8, '32.8（1:21）', 356], ['MLA 几何 chatbot', 0.29, '0.29', 374], ['H200 chatbot', 0.76, '0.76', 392]].forEach(([lab, v, val, y], i) => {
    body.push(cap(52, y + 9, lab, { size: 7.5 }));
    const w = Math.max(2, v * 11);
    body.push(`<g class="sig-bar" style="--i:${(0.3 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${i === 1 ? 'seal' : 'soft'}" x="180" y="${y}" width="${w.toFixed(0)}" height="10" rx="3"/></g>`);
    body.push(cap(180 + w + 6, y + 9, val, { size: 7.5, mono: true, color: i === 1 ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });

  body.push(cap(36, 426, 'T_out ×16 → n_D ×30（T_out 与 T̄ 双因子）；MLA κL ×0.268 → n_D 同因子；H200 β ×1.43 → n_D ×0.70 —— 两池可配不同卡（Splitwise 异构）', { size: 9.5 }));
  body.push(cap(36, 444, 'DeepInfer ~1:1 = 生产流量经验（doc L91）、64P+64D = EP64 最小高效单元量子化；需求配比→部署配比阶梯取整；Beyond-buzz 动态 rate matching（3.5/0.5）', { size: 9.5 }));
  return scene(720, 462, body, { label: 'xPyD 配比推导：P 侧算力需求得一点五五，D 侧 Little 并发折 KV 带宽需求得一点零九，约一点四比一取整一比一；lambda 与 TPOT 相消只剩 workload 形状与硬件；agent 长输出使 D 池涨到三十二点八，MLA 压缩与 H200 带宽使 D 池便宜' }).svg();
}

/* ============================================================
 * 8.2 §5（afterHeading）— 传输 vs 重算：带宽阈值 BW*（签名）
 * 原文：两边同 ∝T → T 相消；BW*=κL·F_eff/(2N_act)=0.376 GB/s≈3 Gbps（V3 FP8）；
 *       跨 DC 6.25=16.6×；V4.1 512 元素 FP4 288B κL=11520（0.328×）；二次项 T=32768→0.271（72%）；
 *       命题 6 layer-wise 全重叠同阈值；长度依赖三渠道
 * ============================================================ */
function bwThreshold() {
  const UID = 'i8h';
  const body = [];
  body.push(cap(36, 24, '传输 vs 重算：两边同 ∝T → T 相消，临界是带宽阈值 BW* = 0.376 GB/s ≈ 3 Gbps —— 不是临界上下文长度', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 300, h: 150, label: 'per-token 账（V3 MLA FP8 @ H100 FP8）' }));
  body.push(cap(52, 72, '传输每 token：κL = 35136 B', { size: 8.5, mono: true }));
  body.push(cap(52, 90, '重算每 token：2N_act = 7.4e10 FLOPs', { size: 8.5, mono: true }));
  body.push(cap(52, 108, '→ 93.5 μs/token（F_eff = 791.6 TFLOPS）', { size: 8.5, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(52, 130, '两边 ×T → T 相消（T>0 同除不等号不变）', { size: 8.5 }));
  body.push(cap(52, 152, 'BW* = κL·F_eff/(2N_act) = 0.376 GB/s', { size: 9, mono: true, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 172, 'T=4096：传输 23.0 ms vs 重算 382.9 ms', { size: 7.5 }));

  body.push(zone({ x: 352, y: 40, w: 332, h: 150, label: '带宽轴（log · GB/s）—— 阈值左重算赢 / 右传输赢' }));
  const LX = (v) => 368 + (Math.log10(v) + 1) * 71.4;
  body.push(`<rect x="368" y="90" width="${(LX(0.376) - 368).toFixed(0)}" height="40" fill="#f3ecdc" opacity="0.6"/>`);
  body.push(`<rect x="${LX(0.376).toFixed(0)}" y="90" width="${(668 - LX(0.376)).toFixed(0)}" height="40" fill="var(--seal-wash)" opacity="0.5"/>`);
  body.push(`<line class="dia-edge strong" x1="368" y1="130" x2="668" y2="130"/>`);
  [[0.1, '0.1'], [1, '1'], [10, '10'], [100, '100'], [1000, '1000']].forEach(([v, lab]) => {
    body.push(`<line x1="${LX(v).toFixed(0)}" y1="130" x2="${LX(v).toFixed(0)}" y2="135" stroke="var(--ink-faint)" stroke-width="1"/>`);
    body.push(cap(LX(v), 146, lab, { size: 7, anchor: 'middle', mono: true }));
  });
  body.push(`<line x1="${LX(0.376).toFixed(0)}" y1="90" x2="${LX(0.376).toFixed(0)}" y2="130" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(`<g class="sig-cell" style="--i:0.3"><circle cx="${LX(0.376).toFixed(0)}" cy="90" r="4" fill="var(--seal)"/></g>`);
  body.push(cap(LX(0.376), 82, 'BW* 0.376', { size: 8, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(`<line x1="${LX(6.25).toFixed(0)}" y1="100" x2="${LX(6.25).toFixed(0)}" y2="130" stroke="var(--ink)" stroke-width="1.3"/>`);
  body.push(cap(LX(6.25), 96, '跨 DC 6.25', { size: 7.5, anchor: 'middle' }));
  body.push(cap((LX(0.376) + LX(6.25)) / 2, 112, '16.6×', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(`<line x1="${LX(50).toFixed(0)}" y1="104" x2="${LX(50).toFixed(0)}" y2="130" stroke="var(--ink-faint)" stroke-width="1"/>`);
  body.push(cap(LX(50), 108, 'IB 50', { size: 6.5, anchor: 'middle' }));
  body.push(`<line x1="${LX(900).toFixed(0)}" y1="100" x2="${LX(900).toFixed(0)}" y2="130" stroke="var(--ink-faint)" stroke-width="1"/>`);
  body.push(cap(LX(900), 96, 'NVLink 900', { size: 6.5, anchor: 'middle' }));
  body.push(cap(388, 162, '重算便宜', { size: 6.5, anchor: 'middle' }));
  body.push(cap(538, 162, '传输便宜', { size: 6.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(pathPulse(`M ${LX(0.376).toFixed(0)} 130 H 668`, { dur: 2.6, delay: 0.5, r: 3.4 }));

  body.push(zone({ x: 36, y: 206, w: 300, h: 120, label: 'V4.1 方向更强（X-1 跨波裁定）' }));
  body.push(cap(52, 238, 'MLA = 512 元素（448 latent + 64 rope）', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 254, 'NOT V3 的 576 —— 几何不适用 V4.1', { size: 7.5 }));
  body.push(cap(52, 274, '主 KV per-16 FP4：288 B/token/层', { size: 8.5, mono: true }));
  body.push(cap(52, 292, 'κL = 288×40 = 11520 = V3 的 0.328×', { size: 8.5, mono: true }));
  body.push(cap(52, 310, 'BW* 同比例更低 · 传输优势更大', { size: 8 }));

  body.push(zone({ x: 352, y: 206, w: 332, h: 120, label: '长度依赖三渠道 + 双重身份' }));
  body.push(cap(368, 238, '① 二次项：BW*(T) = κL·F_eff/(2N_act+2LTd)', { size: 8 }));
  body.push(cap(368, 254, 'T=32768 → 0.271 GB/s（线性项的 72%）', { size: 8, mono: true }));
  body.push(cap(368, 274, '② TTFT 预算：绝对传输时间 ∝T 须装进份额', { size: 8 }));
  body.push(cap(368, 292, '③ 重算干扰：塞回 D 池 = TPOT 尖刺回归', { size: 8 }));
  body.push(cap(368, 310, '命题 6：layer-wise 全重叠条件也是 BW*', { size: 8, color: 'var(--seal-deep)' }));

  body.push(cap(36, 344, '每层 KV 2.36 MB @6.25 GB/s = 0.38 ms ≪ 每层计算 6.28 ms → 完全藏住；暴露 ≈ max(0, t_xfer − t_hideable)（式 13）', { size: 9.5 }));
  body.push(cap(36, 362, '坑：临界写成「临界上下文长度」在所有长度上同向错误 —— 长度只从二次项 / TTFT 预算 / 重算干扰三条渠道进入', { size: 9.5 }));
  body.push(cap(36, 380, '重算 = 传输失败的 fallback 不是常态路径（隔离重算资源口径）；小 T 固定延迟主导传输时间，per-token 账须加回', { size: 9.5 }));
  return scene(720, 400, body, { label: '传输与重算都线性于上下文长，相消后临界只剩带宽阈值零点三七六 GB/s；对数带宽轴上跨 DC 每 GPU 六点二五 GB/s 是阈值十六点六倍传输远优；V4.1 五百一十二元素 FP4 使阈值更低；layer-wise 全重叠条件同为该阈值' }).svg();
}

/* ============================================================
 * 8.3 §1（block 0）— serialized vs layer-wise：暴露 85.9 → 2.68 ms（÷L=32）（签名）
 * 原文：t_P=1.060 s、t_xfer=85.9 ms（4.295 GB @50 GB/s）、τ_c=33.1 τ_x=2.68；
 *       命题 1 r_L=τ_c+τ_x+(L−1)max；BW=1 GB/s 反例 3.27 s；Splitwise <512 切换；首 token aux
 * ============================================================ */
function layerwisePipeline() {
  const UID = 'i8i';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'P2D 时序：layer-wise 把传输藏进 prefill 计算 —— 暴露 85.9 → 2.68 ms，恰 ÷L=32（计算主导区）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 96, label: 'serialized（full 一次性）：KV 全部算完才存在、才能传 —— 暴露全量 85.9 ms' }));
  body.push(cap(52, 85, 'prefill SM', { size: 7.5 }));
  body.push(`<rect class="dia-bar dia-bar-ink" x="110" y="70" width="430" height="22" rx="4"/>`);
  body.push(cap(325, 85, '全部 L=32 层 · t_P = 1.060 s', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(52, 109, 'RDMA NIC', { size: 7.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="543" y="98" width="35" height="14" rx="3"/>`);
  body.push(`<line x1="580" y1="70" x2="580" y2="112" stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(cap(584, 85, 'DECODE →', { size: 7, mono: true }));
  body.push(cap(543, 128, '全部 KV 4.295 GB · 暴露 85.9 ms', { size: 7 }));
  body.push(pathPulse('M 543 105 H 578', { dur: 1.2, delay: 0.4, r: 3 }));

  body.push(zone({ x: 36, y: 152, w: 648, h: 130, label: 'layer-wise（逐层流式）：每层算完即传，NIC 与 SM 两种资源重叠 —— 暴露只剩末层尾巴' }));
  body.push(cap(52, 199, 'prefill SM', { size: 7.5 }));
  const cw = 12.43, pitch = 13.43; // 32 格 × 13.43 = 430 px = t_P 1.060 s（与 serialized 同比例）
  for (let i = 0; i < 32; i++) {
    body.push(`<rect class="dia-bar dia-bar-ink" x="${(110 + i * pitch).toFixed(1)}" y="184" width="${cw}" height="22" rx="2"/>`);
  }
  body.push(cap(116, 180, 'L0', { size: 6.5, mono: true }));
  body.push(cap(110 + 31 * pitch + cw / 2, 180, 'L31', { size: 6.5, anchor: 'middle', mono: true }));
  body.push(cap(52, 223, 'RDMA NIC', { size: 7.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="124" y="214" width="418" height="10" rx="3"/>`);
  body.push(pathPulse('M 124 219 H 542', { dur: 2.8, delay: 0.3 }));
  body.push(`<line x1="543" y1="184" x2="543" y2="224" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="3 3"/>`);
  body.push(cap(548, 199, '末层到齐 → DECODE', { size: 7 }));
  body.push(cap(325, 246, '暴露仅末层尾巴 2.68 ms（= τ_x = 85.9/32）—— serialized 的 1/32', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(325, 264, '每层传输 2.68 ms ≪ 每层计算 33.1 ms → 藏进计算（NIC 条细、SM 格宽）', { size: 7.5, anchor: 'middle' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 8.5, ss = 6.3) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 298, w: 648, h: 76, label: '命题 1：r_L = τ_c + τ_x + (L−1)·max(τ_c, τ_x) · 暴露 = max(τ_x, t_xfer − (L−1)τ_c)' }));
  mk('c1', 52, 322, 190, 40, 'cache', '计算主导 τ_c ≥ τ_x', 'τ_c=33.1 ≫ τ_x=2.68 ms → 暴露 τ_x', 8.5, 6);
  mk('c2', 258, 322, 190, 40, 'attn', '传输主导反例', 'BW=1 GB/s：τ_x=134.2 → 暴露 3.27 s', 8.5, 6);
  mk('c3', 464, 322, 204, 40, 'proj', '分界 τ_c = τ_x', '即 t_P = t_xfer ⟺ BW = BW*（§6）', 9, 6.3);

  body.push(cap(36, 392, 'Splitwise 裁定：H100 <512 tok serialized / 大 prompt layer-wise；第二 token 开销 64% → 16.5%（3.9×），E2E 3% → 0.8%', { size: 9.5 }));
  body.push(cap(36, 410, '实测尾巴近似常数（A100 ~8 / H100 ~5 ms）= 同步 + 启动 + 对齐固定开销 —— 勿反推拟合 τ_x（坑）；结构账用命题 1', { size: 9.5 }));
  body.push(cap(36, 428, '首 token 在 prefill 侧采样，id/logprobs 经 aux 通道随 KV 传；decode PREBUILT 只填 metadata，第一次真实 forward = 第二 token 的 DECODE 步', { size: 9.5 }));
  return scene(720, 446, body, { label: 'P2D 两种时序：serialized 全部算完再传暴露八十五点九毫秒；layer-wise 逐层流式 NIC 细条藏在 SM 宽格计算下，暴露只剩末层二点六八毫秒恰为层数三十二分之一；计算主导与传输主导两情形分界即带宽阈值' }).svg();
}

/* ============================================================
 * 8.3 §3（block 1）— NIXL 双平面：控制面 metadata × 数据面 RDMA（签名）
 * 原文：getLocalMD→blob→loadRemoteMD；registerMem(reg_dlist)；postXferReq(xfer_dlist)；
 *       NIXL_WRITE/READ initiator 视角；18 插件含 mooncake；GDR 非独立插件 = UCX GPU 直写
 * ============================================================ */
function nixlPlanes() {
  const UID = 'i8j';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'NIXL 双平面：metadata 控制面（小字节）协商地址，KV 数据面（大字节）RDMA 直传 —— KV 不走控制通道', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 320, h: 150, label: '控制面（metadata · 小字节）' }));
  const pa = mk('pa', 52, 70, 120, 40, 'op', 'prefill agent', 'nixlAgent L35', 9, 6.5);
  const blob = mk('blob', 210, 70, 120, 40, 'data', 'MD blob', 'getLocalMD L512', 9, 6.5);
  const da = mk('da', 210, 130, 120, 40, 'op', 'decode agent', 'loadRemoteMD L543', 8.5, 6);
  body.push(edge(pa, blob, { nodes, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'm0' }));
  body.push(arrowHead(blob.left[0], blob.left[1], 1, 0));
  body.push(edge(blob, da, { nodes, portFrom: blob.bottom, portTo: da.top, pulse: true, dur: 1.1, delay: 0.4, uid: UID, eid: 'm1' }));
  body.push(arrowHead(da.top[0], da.top[1], 0, 1));
  body.push(cap(52, 126, 'registerMem(reg_dlist) L118', { size: 7, mono: true }));
  body.push(cap(52, 142, 'side-channel / etcd / Redis', { size: 7 }));
  body.push(cap(52, 158, '加 agent = 提供 MD · 删 = invalidate', { size: 6.8 }));
  body.push(cap(52, 174, 'Metadata Handler 缓存 · 控制路径非数据路径', { size: 6.8 }));

  body.push(zone({ x: 372, y: 40, w: 312, h: 150, label: '数据面（KV · 大字节）' }));
  const pv = mk('pv', 388, 76, 120, 48, 'cache', 'prefill VRAM', 'KV buffer（HBM）', 9, 6.5);
  const dv = mk('dv', 548, 76, 120, 48, 'cache', 'decode VRAM', '预分配槽位', 9, 6.5);
  body.push(edge(pv, dv, { nodes, pulse: true, dur: 1.4, delay: 0.4, strong: true, uid: UID, eid: 'k0' }));
  body.push(arrowHead(dv.left[0], dv.left[1], 1, 0));
  body.push(cap(528, 70, 'RDMA', { size: 8, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(528, 140, 'NIXL_WRITE / READ', { size: 7.5, anchor: 'middle', mono: true }));
  body.push(cap(388, 158, 'postXferReq(xfer_dlist) L345 异步', { size: 7, mono: true }));
  body.push(cap(388, 172, 'getXferStatus L355 poll · IN_PROG → SUCCESS', { size: 6.8 }));

  body.push(zone({ x: 36, y: 206, w: 648, h: 96, label: 'src/plugins/ —— 18 插件目录（2026-11-23 tarball 逐目录核实）· 核心 backend UCX + GDS' }));
  const chip = (x, y, w, name, hot = false) => {
    body.push(`<rect x="${x}" y="${y}" width="${w}" height="20" rx="5" fill="${hot ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${hot ? 'var(--seal)' : '#c6c0af'}" stroke-width="${hot ? 1.3 : 0.9}"/>`);
    body.push(cap(x + w / 2, y + 14, name, { size: 7, anchor: 'middle', mono: true, color: hot ? 'var(--seal-deep)' : 'var(--ink-soft)' }));
  };
  let cx = 52;
  [['ucx', 36], ['cuda_gds', 58], ['mooncake', 62, true], ['hf3fs', 40], ['obj', 30], ['posix', 38], ['libfabric', 56], ['gpunetio', 56], ['gpudirect_tcpxo', 92]].forEach(([n, w, hot]) => {
    chip(cx, 236, w, n, hot); cx += w + 8;
  });
  cx = 52;
  [['doca_memos', 68], ['azure_blob', 66], ['redis', 38], ['uccl', 34], ['rocm_ais', 54], ['infinia', 46], ['gusli', 38], ['telemetry', 58], ['tracing', 50]].forEach(([n, w]) => {
    chip(cx, 264, w, n); cx += w + 8;
  });
  body.push(cap(52, 296, 'NIXL 经 mooncake 插件可用 Mooncake TE 作 backend —— 可组合上下层，非互斥竞争（§4）', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 320, 'descriptor 三型：Basic（addr+len+devId）/ Stride（stride==len 即连续）/ Blob（带 MD）；xfer_dlist=Basic vs reg_dlist=Blob 不可互代（坑）', { size: 9.5 }));
  body.push(cap(36, 338, 'GDR 非独立 backend：核心 UCX+GDS（NVMe）；GDR 是 UCX 下 GPU 直写路径 —— 按旧文档「UCX/GDR/RDMA」并列找插件会落空', { size: 9.5 }));
  body.push(cap(36, 356, 'initiator 视角：谁 post 谁发起、与数据流向无关 —— WRITE=push / READ=pull；引擎侧 vLLM NixlPush L415 / NixlPull L374、SGLang NixlKVManager L531', { size: 9.5 }));
  body.push(cap(36, 374, 'conductor 进程管理 agent（每 agent 可管多 GPU、全局唯一 name）；memory（CPU/GPU）+ storage（file/block/object）统一抽象', { size: 9.5 }));
  return scene(720, 394, body, { label: 'NIXL 双平面：控制面 prefill agent 导出元数据 blob 经 side-channel 交 decode agent 加载，数据面 prefill VRAM 经 RDMA 直写 decode VRAM；十八个插件目录含 mooncake 可组合，GDR 是 UCX 下 GPU 直写路径非独立插件' }).svg();
}

/* ============================================================
 * 8.3 §4（block 2）— Mooncake TE：topology matrix × 多 NIC 池化（签名）
 * 原文：cpu:0→mlx5_0/1、cpu:1→mlx5_2/3、gpu:0→mlx5_4（本地 PCIe switch）；
 *       >64 KB 切 slice；BW_pool=Σ NIC BW=4×50=200 GB/s；每层 134.2 MB→2048 slice→0.67 ms；
 *       Segment 进程级 / Buffer 注册级；SIEVE 驱逐
 * ============================================================ */
function mooncakeTopology() {
  const UID = 'i8k';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'Mooncake TE：topology matrix 选路 × >64 KB 切 slice 多 NIC 池化 —— 聚合 BW = Σ NIC BW，4×400G = 200 GB/s', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const nic = (x, y, name, hot = false) => {
    body.push(`<rect x="${x}" y="${y}" width="64" height="22" rx="6" fill="${hot ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${hot ? 'var(--seal)' : '#b6b6c4'}" stroke-width="${hot ? 1.3 : 1}"/>`);
    body.push(cap(x + 32, y + 15, name, { size: 7.5, anchor: 'middle', mono: true, color: hot ? 'var(--seal-deep)' : 'var(--ink-soft)' }));
  };
  body.push(zone({ x: 36, y: 40, w: 300, h: 180, label: '本地节点（topology matrix 广播集群）' }));
  body.push(cap(52, 79, 'cpu:0', { size: 8, mono: true }));
  nic(110, 64, 'mlx5_0'); nic(182, 64, 'mlx5_1');
  body.push(cap(52, 113, 'cpu:1', { size: 8, mono: true }));
  nic(110, 98, 'mlx5_2'); nic(182, 98, 'mlx5_3');
  body.push(cap(52, 147, 'gpu:0', { size: 8, mono: true }));
  nic(110, 132, 'mlx5_4', true);
  body.push(cap(182, 147, '本地 PCIe switch（GDR）', { size: 6.8 }));
  body.push(cap(52, 176, 'preferred = 本地 NUMA / PCIe switch · secondary 故障备选', { size: 7 }));
  body.push(cap(52, 192, 'NIC 暂不可用 → 自动选替代路径重提交', { size: 7 }));

  body.push(zone({ x: 352, y: 40, w: 332, h: 180, label: '请求 > 64 KB → 切 slice 多路径（全 NIC 协同）' }));
  const req = mk('req', 368, 70, 300, 40, 'data', 'KV 请求 134.2 MB/层（κT）', '> 64 KB → 2048 个 slice', 9, 6.5);
  const slices = [['mlx5_0', 'slice0', 368], ['mlx5_1', 'slice1', 446], ['mlx5_4', 'slice2', 524], ['mlx5_2', 'slice3', 602]];
  slices.forEach(([name, sl, x], i) => {
    const n = mk(`s${i}`, x, 130, 70, 36, 'op', name, sl, 8, 6);
    body.push(edge(req, n, { nodes, portFrom: [x + 35, req.y + req.h], portTo: n.top, pulse: true, dur: 1, delay: 0.2 + i * 0.2, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(n.top[0], n.top[1], 0, 1));
  });
  body.push(cap(368, 182, 'BW_pool = Σ BW_j = 4×50 = 200 GB/s', { size: 9, mono: true, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(368, 200, '每层传输 2.68 → 0.67 ms（单 NIC → 4 NIC）', { size: 8 }));

  body.push(zone({ x: 36, y: 236, w: 648, h: 84, label: '两个核心抽象 + endpoint 管理' }));
  mk('seg', 52, 262, 190, 44, 'cache', 'Segment（进程级）', '每进程一个 · name=hostname', 9, 6.3);
  mk('buf', 258, 262, 190, 44, 'proj', 'Buffer（注册级）', 'registerLocalMemory · rkey+NIC 亲和', 8.5, 5.8);
  mk('bat', 464, 262, 204, 44, 'op', 'BatchTransfer', 'READ/WRITE 请求数组 · 异步 poll', 9, 6.3);
  body.push(cap(52, 314, 'endpoint pooling + SIEVE 驱逐 · waiting_list ~1 Hz heartbeat 回收 —— Segment≠Buffer（坑：注册粒度是 Buffer）', { size: 7.5 }));

  body.push(cap(36, 338, '拓扑感知避开共同瓶颈（PCIe switch / UPI 不超限）—— 聚合账前提：slice 足够分散 + 各路径无共同瓶颈', { size: 9.5 }));
  body.push(cap(36, 356, 'NIXL mooncake 插件把 Mooncake TE 接成一个 backend：选型 =「引擎接口层用谁 / 传输底座用谁」两层决策，非互斥竞争', { size: 9.5 }));
  body.push(cap(36, 374, '版本漂移（HEAD）：新增 getTaskCongestionState / Ascend Direct Transport / disconnect_all_peers；核心 Segment/BatchTransfer/topology 语义不变', { size: 9.5 }));
  return scene(720, 394, body, { label: 'Mooncake 拓扑感知多网卡池化：topology matrix 按内存类型分 preferred 与 secondary NIC 列表，大于六十四 KB 的请求切两千零四十八个 slice 分散到四张网卡，聚合带宽二百 GB/s 每层传输降到零点六七毫秒；Segment 进程级 Buffer 注册级' }).svg();
}

/* ============================================================
 * 8.3 §6.3（block 3）— 跨 DC P/D：压缩使可行、可靠性付代价（签名）
 * 原文：doc L97 ~400 Gbps/8 GPU、每 GPU 6.25 GB/s；V4.1 FP4 377.5 MB→60.4 ms（余量 3.31×）；
 *       bf16 214.7 / V3 FP8 184.2 / 8B GQA 687.2 ms 超预算；压缩比 3.56×；BW* 0.376 的 16.6×；相邻 DC
 * ============================================================ */
function crossDC() {
  const UID = 'i8l';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, '跨 DC P/D：压缩使可行（V4.1 FP4 60.4 ms 预算余量 3.31×），可靠性付代价 —— 光纤易断、部署相邻 DC', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 44, w: 200, h: 100, label: 'DC-A（prefill）' }));
  mk('pa', 52, 76, 168, 52, 'proj', 'P 实例 ×n_P', 'V4.1 L=40 · FP4 288 B/token/层', 10, 6);
  body.push(zone({ x: 484, y: 44, w: 200, h: 100, label: 'DC-B（decode）' }));
  mk('db', 500, 76, 168, 52, 'attn', 'D 实例 ×n_D', 'Wide-EP · EP64–128', 10, 6.5);
  body.push(`<path class="dia-edge strong" d="M236 102 H484" style="stroke:var(--seal);stroke-width:2"/>`);
  body.push(pathPulse('M236 102 H484', { dur: 1.6, delay: 0.3 }));
  body.push(arrowHead(484, 102, 1, 0));
  body.push(cap(360, 92, '~400 Gbps / 8 GPU', { size: 9, anchor: 'middle', mono: true, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(360, 118, '压缩 KV（FP4）· 每 GPU 6.25 GB/s', { size: 8, anchor: 'middle' }));
  body.push(cap(360, 134, '可靠性代价：光纤易被施工挖断', { size: 7.5, anchor: 'middle', color: 'var(--seal)' }));

  body.push(zone({ x: 36, y: 160, w: 648, h: 130, label: 'T=32768 · 每 GPU 6.25 GB/s · TTFT 传输预算 0.2 s（条按比例）' }));
  body.push(`<line x1="324" y1="184" x2="324" y2="278" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="4 4"/>`);
  body.push(cap(324, 286, '预算 200 ms', { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));
  [['V4.1 FP4（生产）', 60.4, 'seal', '60.4 ms ✓ 余量 3.31×', 190], ['V4.1 bf16（反事实）', 214.7, 'soft', '214.7 ms 超', 214], ['V3 系 FP8（对照）', 184.2, 'warm', '184.2 ms 超', 238], ['8B GQA bf16（章脊）', 687.2, 'soft', '687.2 ms 超', 262]].forEach(([lab, v, tone, val, y], i) => {
    body.push(cap(52, y + 11, lab, { size: 7.5 }));
    const w = v * 0.72;
    body.push(`<g class="sig-bar" style="--i:${(0.2 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${tone}" x="180" y="${y}" width="${w.toFixed(0)}" height="14" rx="3"/></g>`);
    if (i >= 2) body.push(cap(180 + w - 6, y + 11, val, { size: 7, anchor: 'end', mono: true, color: 'var(--ink)' }));
    else body.push(cap(180 + w + 6, y + 11, val, { size: 7.5, mono: true, color: i === 0 ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });

  mk('c1', 36, 306, 310, 60, 'cache', '压缩比 3.56×', 'bf16 40960 / FP4 11520 B/token · 压缩是带宽需求的直接除数', 10, 6.3);
  mk('c2', 362, 306, 322, 60, 'attn', 'BW* 对照：6.25 / 0.376 = 16.6×', 'V3 FP8 BW*=0.376 GB/s ≈ 3 Gbps · 传输远优于重算（doc L97 自洽）', 9, 6);

  body.push(cap(36, 384, 'doc L97 逐字："compresses the KV cache substantially → splitting P and D across data centers is technically feasible (at some cost in reliability)"', { size: 9.5 }));
  body.push(cap(36, 402, 'DeepSeek 推理节点部署在不同但相邻的 DC（缩短光纤路径）；layer-wise 重叠 + 压缩共同压低暴露 —— 带宽账与可靠性账并列（8.5 容错）', { size: 9.5 }));
  return scene(720, 420, body, { label: '跨数据中心 PD：DC-A prefill 经约四百 Gbps 每八 GPU 的相邻 DC 光纤把 FP4 压缩 KV 送到 DC-B decode，每 GPU 六点二五 GB/s；四口径传输时间条对照二百毫秒预算，只有 FP4 六十点四毫秒达标余量三点三倍；压缩比三点五六倍，可靠性代价光纤易断' }).svg();
}

/* ============================================================
 * 8.3 §5（afterHeading）— GDR：RDMA NIC 直写 HBM 不占 SM（签名）
 * 原文：三条拷贝路径 SM kernel / copy engine / GDR；doc L191 逐字；
 *       VRAM→VRAM 不经 host（Dynamo）；bounce buffer ≥1024 desc ∧ ≤16 KiB，2048×8KiB→1 chunk
 * ============================================================ */
function gdrPaths() {
  const UID = 'i8m';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, '谁付搬运成本：GDR 让 RDMA NIC 直写 HBM 不占 SM —— copy engine 细粒度差、SM kernel 与计算争资源', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 190, label: '三条拷贝路径（2.4 §4 recap × P2D 语境）' }));
  const lane = (idx, y, lab, mid, midRole, midSub, warn, hot = false) => {
    body.push(cap(52, y + 22, lab, { size: 8, weight: '650', color: hot ? 'var(--seal-deep)' : 'var(--ink-soft)' }));
    const a = mk(`l${idx}a`, 150, y, 90, 36, 'cache', 'GPU HBM', null, 9);
    const m = mk(`l${idx}m`, 280, y, 120, 36, midRole, mid, midSub, 8.5, 5.8);
    const b = mk(`l${idx}b`, 440, y, 90, 36, 'cache', hot ? '远端 HBM' : 'GPU HBM', null, 9);
    if (hot) body.push(sealDash(280, y, 120, 36));
    body.push(edge(a, m, { nodes, pulse: true, dur: 1.1, delay: hot ? 0.2 : 0.6, strong: hot, uid: UID, eid: `l${idx}0` }));
    body.push(arrowHead(m.left[0], m.left[1], 1, 0));
    body.push(edge(m, b, { nodes, pulse: true, dur: 1.1, delay: hot ? 0.5 : 0.9, strong: hot, uid: UID, eid: `l${idx}1` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
    body.push(cap(560, y + 22, warn, { size: 6.8 }));
  };
  lane(0, 70, 'SM kernel', 'SM copy kernel', 'attn', '占 SM · 与计算争资源', '灵活但挤占计算');
  lane(1, 122, 'copy engine', 'CUDA copy engine', 'op', '不占 SM · 逐描述符开销', '细粒度差（开销主导）');
  lane(2, 174, 'GDR', 'RDMA NIC 直写', 'proj', '不占 SM · 饱和 PCIe', 'VRAM→VRAM 不经 host', true);

  body.push(cap(36, 248, 'doc L191 逐字："an RDMA NIC saturates PCIe bandwidth without consuming SMs" —— swap-in 用 GDR 而非 copy engine', { size: 8.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 264, w: 648, h: 96, label: 'bounce buffer：小 fragment 聚合成大拷贝（TRT-LLM · ctx TP ≠ gen TP / attention DP）' }));
  const b1 = mk('b1', 52, 292, 150, 44, 'data', '2048 descriptor × 8 KiB', '= 16 MiB 小 fragment write', 8, 5.8);
  const b2 = mk('b2', 222, 292, 130, 44, 'op', 'sender gather', '连续缓冲', 9, 6.5);
  const b3 = mk('b3', 372, 292, 130, 44, 'proj', 'chunk 32 MiB 发送', 'descriptor 2048 → 1', 8.5, 6.3);
  const b4 = mk('b4', 522, 292, 146, 44, 'cache', 'receiver scatter', '回 KV cache', 9, 6.5);
  [[b1, b2, 0.2], [b2, b3, 0.5], [b3, b4, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `bb${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(52, 352, '默认启用：≥1024 descriptor ∧ 平均 ≤16 KiB（head-matched 布局如 MLA / 对称 TP 走标准 NIXL）', { size: 7.5 }));

  body.push(cap(36, 378, 'DeepInfer swap-in = host shm → HBM 的 H2D，同样 GDR 逐层重叠 forward（2.4 三段流水）；Dynamo P2D = VRAM→VRAM 直传不经 host', { size: 9.5 }));
  body.push(cap(36, 396, '小片段问题：ctx 与 gen attention 切分不同 → 一次 KV write = 大量小 fragment 各 1 descriptor、描述符开销主导 —— bounce 是「小块聚合」的 P2D 版', { size: 9.5 }));
  body.push(cap(36, 414, '硬件协议层（RDMA RC/UC/QP、NIC、PCIe 拓扑、NVLink 域、NCCL）归 cluster 维度 —— 本章只讲传输语义与带宽账', { size: 9.5 }));
  return scene(720, 432, body, { label: '三条拷贝路径对比：SM kernel 占 SM 与计算争资源，CUDA copy engine 不占 SM 但细粒度逐描述符开销差，GDR 让 RDMA NIC 直写 HBM 不占 SM 饱和 PCIe 为最优；bounce buffer 把两千零四十八个八 KiB 小 fragment 聚合成一个三十二 MiB chunk 描述符降为一' }).svg();
}

/* ============================================================
 * 8.4 §4.1（block 0）— Dynamo disagg 编排：KV-aware 路由 → NIXL 直传（签名）
 * 原文：Client→Frontend→Router→Prefill→Decode；disaggregated_params；NIXL VRAM→VRAM non-blocking；
 *       成本模型 A 820 vs B 300（命题 1）；runtime xPyD；backend 握手时序
 * ============================================================ */
function dynamoFlow() {
  const UID = 'i8n';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'Dynamo = 引擎之上的编排层（不替换 vLLM/SGLang/TRT-LLM）：KV-aware 路由 → prefill → NIXL 直传 → decode', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 150, label: 'disagg 编排流程（disagg-serving.md）' }));
  const cl = mk('cl', 52, 72, 80, 40, 'data', 'Client', null, 9.5);
  const fr = mk('fr', 152, 72, 90, 40, 'norm', 'Frontend', null, 9.5);
  const ro = mk('ro', 262, 72, 130, 40, 'op', 'Router（KV-aware）', '成本模型选最低 cost', 8.5, 6);
  const pf = mk('pf', 412, 72, 120, 40, 'proj', 'Prefill worker', '算 KV → disagg_params', 8.5, 5.8);
  const de = mk('de', 552, 72, 116, 40, 'attn', 'Decode worker', 'generate → stream', 8.5, 6);
  [[cl, fr, 0], [fr, ro, 0.2], [ro, pf, 0.4], [pf, de, 0.6]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(402, 66, 'select P worker', { size: 6.5, anchor: 'middle' }));
  body.push(cap(542, 66, 'select D + 注入 MD', { size: 6.5, anchor: 'middle' }));
  const kv = 'M472 112 C472 140, 610 140, 610 112';
  body.push(`<path class="dia-edge strong" d="${kv}" style="stroke:var(--seal);stroke-width:1.8"/>`);
  body.push(pathPulse(kv, { dur: 1.4, delay: 0.8 }));
  body.push(arrowHead(610, 112, 0, -1));
  body.push(cap(541, 152, 'KV transfer：NIXL VRAM→VRAM 直传 · non-blocking', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 176, 'Decode → Frontend → Client（stream 回程）；握手元数据走控制面、KV 数据面不经握手通道', { size: 7.5 }));

  body.push(zone({ x: 36, y: 206, w: 320, h: 130, label: 'KV-aware 成本模型（式 1 · block 当量 · 教学构造值）' }));
  body.push(cap(52, 240, 'cost = s_P·max(0, A_P+I_P−C_ov) + (A_D+I_D) + w_R·R', { size: 7.5, mono: true }));
  body.push(cap(52, 258, 'A：cache-hot 但重载', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-soft" x="52" y="264" width="230" height="12" rx="3"/></g>`);
  body.push(cap(288, 274, '820', { size: 7.5, mono: true }));
  body.push(cap(52, 290, 'B：cache-cold 但轻载', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-seal" x="52" y="296" width="84" height="12" rx="3"/></g>`);
  body.push(cap(142, 306, '300 ✓ B 胜出', { size: 7.5, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(52, 326, '命中信用 80 只折 prefill 项，被活跃负载 620 压倒', { size: 7 }));

  body.push(zone({ x: 372, y: 206, w: 312, h: 130, label: 'runtime-reconfigurable xPyD + backend 握手' }));
  body.push(cap(388, 240, '加入：register discovery + 发布 RuntimeConfig（KV capacity）', { size: 7.3 }));
  body.push(cap(388, 256, '移除：drain 活跃请求 + deregister · router 自动发现', { size: 7.3 }));
  body.push(cap(388, 276, 'SGLang bootstrap_info：decode 立即开始 + KV 并行', { size: 7.3 }));
  body.push(cap(388, 292, 'vLLM kv_transfer_params：decode 等 prefill 完', { size: 7.3 }));
  body.push(cap(388, 308, 'TRT-LLM opaque_state：prefill 同步（同 vLLM）', { size: 7.3 }));
  body.push(cap(388, 326, '配比不是部署时定死 —— 8.2 xPyD 原理的运行态实现', { size: 7.3, color: 'var(--seal-deep)' }));

  body.push(cap(36, 354, '控制面握手元数据（小字节）/ 数据面 NIXL 直传不经握手通道 —— 与 NIXL MD「控制非数据」、SGLang bootstrap「纯控制面」同一设计原则', { size: 9.5 }));
  body.push(cap(36, 372, '「decode 立即开始」= 控制面准入与预分配立即，生成仍须 KV 到齐才进 DECODE（PREBUILT 不跑 forward）—— 时序差异非数据面正确性', { size: 9.5 }));
  body.push(cap(36, 390, '命题 1：cache-hot 不保证选中（A 820 vs B 300）；命题 2：零信用 + 等负载退化为纯 LB —— index 的价值全在信用项', { size: 9.5 }));
  body.push(cap(36, 408, 'planner 双模：throughput 设副本下界 + load FPM 在线回归反应式修正（SLA-driven，8.5 对照）；KV hit rate + spec decode accept 预测器', { size: 9.5 }));
  return scene(720, 426, body, { label: 'Dynamo disagg 编排：Client 经 Frontend 到 KV-aware Router 选 prefill worker，prefill 算 KV 返回传输元数据，Router 选 decode 注入元数据，KV 经 NIXL VRAM 到 VRAM 直传不经路由；成本模型 cache-hot 重载八百二十输给 cache-cold 轻载三百；runtime xPyD 运行时加入移除' }).svg();
}

/* ============================================================
 * 8.4 §8（afterHeading）— 六系统对照：KV 路径三型 × 编排三型 × 开源完整度（签名）
 * 原文：直传 vLLM/SGLang/Dynamo/TRT-LLM（egress/ingress 账）/ 存储中转 DeepInfer（offload 账）/
 *       pool MemServe；编排内置/插件/外部；DeepInfer 引擎在编排外 vs MemServe 无 repo 两种「不全」
 * ============================================================ */
function pdSystemsCompare() {
  const UID = 'i8o';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, '六系统对照：KV 路径三型决定性能账形态，开源完整度决定落地成本 —— 不是功能清单打勾选型', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 11, ss = 6.2) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 104, label: '点对点直传 —— 适用 8.3 egress/ingress 账 + 命题 1 暴露时间' }));
  const direct = [
    ['vLLM', 'NIXL push/pull + MultiConnector', 'connector 插件 + 外部编排', '引擎+connector 全 [A]'],
    ['SGLang', 'mooncake/nixl 引擎内置', '内置 bootstrap 纯控制面', '引擎+disagg 全 [A]'],
    ['Dynamo', 'NIXL VRAM→VRAM 直传', '外部 PrefillRouter 编排', '编排层全 · 引擎用上游'],
    ['TRT-LLM', 'Python V2 + NIXL + bounce', '外部 orchestrator / Dynamo', '引擎+disagg 全'],
  ];
  direct.forEach(([name, path, orch, rel], i) => {
    const x = 52 + i * 160;
    mk(`d${i}`, x, 68, 148, 64, 'proj', name, path, 11, 6.2);
    body.push(cap(x + 74, 124, rel, { size: 6, anchor: 'middle' }));
    body.push(cap(x + 74, 140, orch, { size: 6.5, anchor: 'middle', color: 'var(--ink-faint)' }));
  });

  body.push(zone({ x: 36, y: 160, w: 648, h: 84, label: '存储中转 —— 适用 2.4 offload 取回账（无计算可重叠）+ 三段流水' }));
  mk('di', 52, 188, 300, 44, 'cache', 'DeepInfer — 3FS kvcache-store 中转', '3FS 读 → host shm → GDR H2D · 引擎侧在 release / 编排外', 9, 5.8);
  mk('ve', 372, 188, 296, 44, 'plain', 'VE / PE / CE 三角色 + 专用存储', '任何机命中任何前缀 · 存储计算 autoscaling 解耦 · 故障域隔离', 9, 5.8);

  body.push(zone({ x: 36, y: 260, w: 648, h: 76, label: 'pool 统一传输 —— layer-by-layer 与直传同构但经 pool 分配' }));
  mk('ms', 52, 288, 300, 40, 'data', 'MemServe — MemPool（NCCL/socket）', 'allocation → transmission → insertion · 无官方 repo [B]', 8.5, 5.8);
  mk('gt', 372, 288, 296, 40, 'plain', 'global prompt tree locality 路由', '与 Dynamo KV-aware 同目标不同实现（论文树 vs 生产 index）', 8, 5.8);

  body.push(zone({ x: 36, y: 352, w: 648, h: 72, label: '选型账：两种「不全」不可混为一谈', tone: 'seal' }));
  body.push(cap(52, 386, 'DeepInfer：引擎侧在 release、编排侧仅 doc → 「自建编排」决策（成本已知）', { size: 8 }));
  body.push(cap(52, 404, 'MemServe：无官方 repo → 「是否值得复现论文」决策（机制参照，非生产部署选项）', { size: 8 }));

  body.push(cap(36, 440, '选型主轴 = KV 路径（延迟跳数）+ 开源完整度（自建成本）；弹性：Dynamo runtime xPyD / SGLang ScaleCohort / DeepInfer grace + EP failover（8.5）', { size: 9.5 }));
  return scene(720, 458, body, { label: '六系统按 KV 路径分三型：vLLM SGLang Dynamo TRT-LLM 点对点直传适用 egress ingress 账，DeepInfer 三FS 存储中转适用 offload 取回账，MemServe MemPool 统一传输；编排分内置插件外部三型；DeepInfer 引擎在编排外与 MemServe 无 repo 是两种不同的不全' }).svg();
}

/* ============================================================
 * 8.5 §2（afterHeading）— 伸缩触发账：non-cached tokens × running requests（签名）
 * 原文：doc L251；R_nc=λ(1−h)T_in=2.048M → n_P=⌈7.761⌉=8；C=λT_outτ=10,240 → n_D=⌈11.33⌉=12；
 *       命题 1 n_P∝(1−h)、h=1 decode 仍>0；rate-limit clip Δ_up=2 → 4 周期；配比 1:1.5 教学
 * ============================================================ */
function autoscaleTriggers() {
  const UID = 'i8p';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'autoscaling 触发账（doc L251）：prefill 看 non-cached token 率、decode 看 running request 数 —— 两信号分离', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.3) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const chain = (arr, delays, eid) => arr.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(arr[i - 1], n, { nodes, pulse: true, dur: 1, delay: delays[i - 1], uid: UID, eid: `${eid}${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 40, w: 648, h: 76, label: 'prefill：non-cached token 率定机数（算力需求 · 命中不耗 prefill 算力）' }));
  chain([
    mk('p1', 52, 66, 160, 36, 'data', 'R_nc = λ(1−h)T_in', '1000×0.5×4096 = 2.048M tok/s', 8.5, 5.8),
    mk('p2', 232, 66, 150, 36, 'op', '÷ C_P（单机）', '263,867 tok/s（8 GPU）', 9, 6.3),
    mk('p3', 402, 66, 120, 36, 'cache', '7.761 → ⌈⌉ = 8 台', null, 9.5),
  ], [0, 0.3], 'p');
  body.push(cap(540, 88, 'h=1 → R_nc=0、n_P=0', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 132, w: 648, h: 76, label: 'decode：running request 数定机数（Little 并发 · KV HBM 容量闸）' }));
  chain([
    mk('d1', 52, 158, 160, 36, 'data', 'C = λ·T_out·τ', '1000×512×0.02 = 10,240 reqs', 8.5, 5.8),
    mk('d2', 232, 158, 150, 36, 'op', '÷ C_D（单机）', '903.5 reqs（60 GiB/GPU）', 9, 6.3),
    mk('d3', 402, 158, 120, 36, 'cache', '11.33 → ⌈⌉ = 12 台', null, 9.5),
  ], [0.6, 0.9], 'd');
  body.push(cap(540, 180, 'h=1 decode 仍 n_D>0', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 224, w: 320, h: 120, label: '命题 1：n_P ∝ (1−h)（取整前）' }));
  [['h=0.5', 8, '8 台', 252], ['h=0.8', 4, '4 台', 276], ['h=1.0', 0, '0（decode 仍 12）', 300]].forEach(([lab, v, val, y], i) => {
    body.push(cap(52, y + 10, lab, { size: 7.5, mono: true }));
    const w = Math.max(2, v * 22);
    body.push(`<g class="sig-bar" style="--i:${(0.2 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${i === 2 ? 'soft' : 'seal'}" x="110" y="${y}" width="${w}" height="12" rx="3"/></g>`);
    body.push(cap(110 + w + 6, y + 10, val, { size: 7.5, mono: true, color: i === 2 ? 'var(--ink-faint)' : 'var(--seal-deep)' }));
  });
  body.push(cap(52, 330, '命中省 prefill 算力，不省 decode 并发', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 372, y: 224, w: 312, h: 120, label: 'rate-limit + 上下界（式 5 · 防抖防越界）' }));
  const steps = [8, 10, 12, 14, 16];
  const sn = steps.map((v, i) => mk(`st${i}`, 388 + i * 58, 256, 50, 28, i === 4 ? 'attn' : 'norm', String(v), null, 9));
  sn.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(sn[i - 1], n, { nodes, pulse: true, dur: 0.8, delay: 0.3 + i * 0.25, uid: UID, eid: `r${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });
  body.push(cap(388, 306, '目标 16 · Δ_up=2 → 4 个周期才到位', { size: 7.5 }));
  body.push(cap(388, 322, 'clip(target, n−Δ_down, n+Δ_up) · n_min≤n≤n_max', { size: 7, mono: true }));
  body.push(cap(388, 338, '代价：响应滞后，尖峰前几周期靠冗余（§5）', { size: 7 }));

  body.push(cap(36, 362, '章脊配比 n_P:n_D = 8:12 = 1:1.5 = 容量闸主导的教学负载，非 DeepInfer 生产配比（生产 ~1:1 经验、64P+64D 最小单元，doc L91）', { size: 9.5 }));
  body.push(cap(36, 380, 'decode 第二道闸 = HBM 带宽：并发高到每步 KV 读逼近带宽上限，TPOT 先于容量撞 SLO —— 生产 sizing 取两道闸较小者', { size: 9.5 }));
  body.push(cap(36, 398, 'C_P = F_eff/(2N_act)·g_P = 32,983×8；每请求 KV = T̄·κL = 4352×128 KiB ≈ 0.531 GiB，每 GPU 并发 60/0.531 ≈ 112.9', { size: 9.5 }));
  body.push(cap(36, 416, '负载信号不直接保证延迟 —— 负载到延迟隔着排队与干扰，靠冗余余量吸收（§3/§5）；Dynamo SLA planner 直接闭环对照', { size: 9.5 }));
  return scene(720, 434, body, { label: '伸缩触发账：prefill 由 non-cached token 率二百零四万八千除以单机容量得八台，decode 由 Little 并发一万零二百四十除以单机并发容量得十二台；命中率线性消减 prefill 需求但不省 decode 并发；rate-limit 每周期最多增两台目标十六需四周期' }).svg();
}

/* ============================================================
 * 8.5 §4（afterHeading）— grace 1 min：长尾请求连同 KV 一次性转发（签名）
 * 原文：doc L253；t_fwd=κL·T_rem/β_net=0.5369/50≈10.7 ms vs 重算 133 ms（12×）；
 *       Llumnix 多阶段尾段 2.7 ms；命题 3 一次性∝T_rem vs 多阶段常数；>3000 token 超 60 s
 * ============================================================ */
function graceMigration() {
  const UID = 'i8q';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'grace 1 min（doc L253）：放短请求自然收尾，窗口末把长尾请求连同 KV 一次性转发 —— 暴露 10.7 ms、释放整机', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 8.5, ss = 6) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 140, label: 'CE 收到 shutdown 通知 → grace 窗口 → 一次性转发 → 释放整台 64 GPU 实例' }));
  body.push(`<line class="dia-edge strong" x1="60" y1="110" x2="660" y2="110"/>`);
  body.push(`<rect class="dia-bar dia-bar-soft" x="80" y="96" width="400" height="14" rx="7"/>`);
  body.push(cap(280, 88, 'grace 1 min：短请求自然结束（s 级）', { size: 8, anchor: 'middle' }));
  [150, 220, 300].forEach((x, i) => {
    body.push(`<circle cx="${x}" cy="103" r="3.5" fill="var(--ink)"/>`);
    body.push(`<path class="dia-edge" d="M${x} 110 V122"/>`);
    body.push(arrowHead(x, 122, 0, 1));
  });
  body.push(cap(200, 136, '短请求自然走 ↓', { size: 7, anchor: 'middle' }));
  [420, 440, 460].forEach((x) => body.push(`<circle cx="${x}" cy="103" r="3.5" fill="var(--seal)"/>`));
  body.push(cap(440, 136, '长尾几条仍在', { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(`<path class="dia-edge strong" d="M480 103 H560" style="stroke:var(--seal);stroke-width:1.8"/>`);
  body.push(pathPulse('M480 103 H560', { dur: 1, delay: 1.2 }));
  body.push(arrowHead(560, 103, 1, 0));
  mk('ce', 560, 84, 90, 38, 'cache', '其他 CE', 'KV 一次性', 8.5, 6);
  body.push(cap(520, 78, 't_fwd 10.7 ms', { size: 7.5, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(620, 136, '释放整机', { size: 7.5, anchor: 'middle' }));
  body.push(cap(80, 158, '通知', { size: 7, anchor: 'middle' }));
  body.push(cap(480, 158, '窗口末', { size: 7, anchor: 'middle' }));
  body.push(cap(605, 158, '续 decode', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 196, w: 648, h: 96, label: '暴露 downtime 对照（T_rem=4096 · KV 0.5 GiB · β_net=50 GB/s · 条按比例）' }));
  [['Llumnix 多阶段', 2.7, 'seal', '2.7 ms（尾段常数）', 224], ['grace 一次性', 10.7, 'ink', '10.7 ms（∝T_rem）', 248], ['重算（fallback）', 133, 'warm', '133 ms（12× 一次性）', 272]].forEach(([lab, v, tone, val, y], i) => {
    body.push(cap(52, y + 11, lab, { size: 7.5 }));
    const w = v * 3;
    body.push(`<g class="sig-bar" style="--i:${(0.2 + i * 0.2).toFixed(1)}"><rect class="dia-bar dia-bar-${tone}" x="180" y="${y}" width="${w.toFixed(0)}" height="14" rx="3"/></g>`);
    body.push(cap(180 + w + 6, y + 11, val, { size: 7.5, mono: true, color: i === 1 ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });

  mk('c1', 36, 308, 310, 60, 'cache', '1 min 窗口为何宽裕', '自然结束 >60 s 须 T_out_rem > 3000 token（τ=20 ms）', 9.5, 6.3);
  mk('c2', 362, 308, 322, 60, 'attn', '窗口主作用：放短请求走、少迁几条', '转发 ms ≪ 自然结束 s · 迁走即释放整机（EP64 长尾浪费）', 8.5, 6);

  body.push(cap(36, 386, '命题 3：一次性暴露 ∝ T_rem（全量 KV/带宽）；多阶段批量段被持续 decode 掩盖、只暴露尾段 ≤16 块 ≈ 2.7 ms（常数）—— 量纲不同', { size: 9.5 }));
  body.push(cap(36, 404, '坑：grace 一次性 ≠ Llumnix 多阶段 live migration —— grace = scale-in 计划性动作（非延迟热路径）；多阶段服务在线 LB/碎片整理（延迟敏感）', { size: 9.5 }));
  return scene(720, 422, body, { label: 'grace 迁移时序：CE 收到 shutdown 通知进入一分钟窗口，短请求自然结束长尾几条留下，窗口末把剩余 KV 一次性转发到其他 CE 暴露十点七毫秒后释放整机；对照多阶段尾段二点七毫秒与重算一百三十三毫秒，一次性比重算便宜十二倍且不丢已生成 token' }).svg();
}

/* ============================================================
 * 8.5 §5（afterHeading）— 冗余 quota：u>0.8 禁 scale-in × f_max≥0.2（签名）
 * 原文：doc L257 ready 机池 + 在线>需求；u=n_demand/n_online；n_online≥1.25×；
 *       命题 4 f_max=1−u≥0.2；100/80 u=0.8：f=0.2 剩 80=需求 ✓ / f=0.21 剩 79<80 ✗；u=0.5→f_max 0.5
 * ============================================================ */
function redundancyQuota() {
  const UID = 'i8r';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, '冗余 quota（doc L257）：u > 0.8 禁 scale-in —— 利用率上界 = 容错余量下界，同一笔 20% 既是伸缩缓冲也是故障落点', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 8.5, ss = 6) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 648, h: 96, label: 'n_online = 100 · n_demand = 80 · u = 0.8（最坏情形）' }));
  body.push(`<rect class="dia-bar dia-bar-ink" x="52" y="76" width="480" height="24" rx="4"/>`);
  body.push(`<rect x="532" y="76" width="120" height="24" rx="4" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(cap(292, 92, '需求 80 台（u=0.8）', { size: 8.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(592, 92, 'spare 20', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(`<line x1="532" y1="70" x2="532" y2="106" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="4 4"/>`);
  body.push(cap(532, 66, 'u_max=0.8 禁 scale-in 线', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 122, 'n_online ≥ n_demand/0.8 = 1.25× —— 至少多备 25%（相对需求）余量', { size: 8 }));

  body.push(zone({ x: 36, y: 152, w: 320, h: 130, label: '命题 4：f_max = 1−u ≥ 0.2（下界）' }));
  body.push(cap(52, 186, 'f=0.2：剩 80 = 需求 ✓（边界取等）', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-ink" x="52" y="192" width="192" height="12" rx="3"/></g>`);
  body.push(cap(52, 222, 'f=0.21：剩 79 < 80 缺口 ✗', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-warm" x="52" y="228" width="190" height="12" rx="3"/></g>`);
  body.push(`<line x1="244" y1="188" x2="244" y2="244" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(cap(248, 206, '需求线 80', { size: 6.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 260, 'u=0.5（需求 50）→ f_max=0.5：可吸收比例随冗余增大', { size: 7.3 }));
  body.push(cap(52, 276, '0.8 保证的是下界 0.2 而非上界', { size: 7.3, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 372, y: 152, w: 312, h: 130, label: '冗余两形态（doc L257）' }));
  mk('r1', 388, 180, 280, 40, 'cache', 'ready 机池（unoccupied）', 'scale-out 库存 · 就绪机即时上线（免冷启动分钟级）', 8.5, 5.8);
  mk('r2', 388, 228, 280, 40, 'attn', '在线实例 > 计算需求', '故障缺口由 rerouting 填（ms）· 重启在后台慢慢做', 8.5, 5.8);

  body.push(cap(36, 300, '冗余成本实打实：多备 20 台闲置/低载；收益 = 免冷启动延迟 + 免 SLO 违约 —— 把分钟级不可用换成 ms 级迁移 + 一笔闲置成本', { size: 9.5 }));
  body.push(cap(36, 318, '策略（何时缩）与机制（故障怎么接）分离：缩容策略留余量，rerouting 机制用余量 —— 0.8 把容错余量写进伸缩策略护栏', { size: 9.5 }));
  body.push(cap(36, 336, '故障 f > f_max 落点不足：缺口须含等待重启项（练习 4b）；训推共调度（doc L119）ready 池被训练借走 = 风险（§7）', { size: 9.5 }));
  body.push(cap(36, 354, '成本账（USD/M tokens 与冗余定价）归 9.1；quota 账按机数计，大 EP 实例内部 GPU 组成与 failover 见 §6 —— 两层粒度不混', { size: 9.5 }));
  return scene(720, 372, body, { label: '冗余 quota：在线一百台中需求八十台利用率零点八， spare 二十台既是伸缩缓冲也是故障落点；命题四可吸收故障比例下界零点二，故障二十台剩八十恰等于需求二十一台则缺口；ready 机池免冷启动在线余量接 rerouting' }).svg();
}

/* ============================================================
 * 8.5 §6（afterHeading）— EP failover：重定 EP size × rerouting 保 SLO（签名）
 * 原文：doc L265 单通信域单 GPU 故障整实例不可用、重启重定 EP size；EP64→EP56 384/56≈6.857、×1.143；
 *       doc L261 重启分钟级、容错靠冗余+rerouting；命题 5 缺口 10.7 ms<20 ms vs 60 s=3000× SLO
 * ============================================================ */
function epFailover() {
  const UID = 'i8s';
  const body = [];
  const nodes = [];
  body.push(cap(36, 24, 'EP failover（doc L265）：单通信域单 GPU 故障 → 整实例不可用 —— 重启重定 EP size，SLO 靠 rerouting 而非缩短重启', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 8.5, ss = 6) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(zone({ x: 36, y: 40, w: 320, h: 200, label: '大 EP 实例：全部 GPU 单通信域（all-to-all 须全员）' }));
  body.push(cap(60, 72, 'EP64（64 GPU）', { size: 8, mono: true }));
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
    const cx = 60 + j * 18, cy = 78 + i * 18;
    const failed = i === 2 && j === 3;
    body.push(`<circle cx="${cx}" cy="${cy}" r="4.5" fill="${failed ? 'var(--seal)' : 'var(--ink)'}"/>`);
    if (failed) {
      body.push(`<path d="M${cx - 6} ${cy - 6} L${cx + 6} ${cy + 6} M${cx + 6} ${cy - 6} L${cx - 6} ${cy + 6}" stroke="var(--seal-deep)" stroke-width="1.6"/>`);
    }
  }
  body.push(`<path class="dia-edge strong" d="M190 140 H236"/>`);
  body.push(pathPulse('M190 140 H236', { dur: 1, delay: 0.5, r: 3 }));
  body.push(arrowHead(236, 140, 1, 0));
  body.push(cap(213, 132, '重启重定', { size: 6.5, anchor: 'middle' }));
  body.push(cap(240, 82, 'EP56（实收）', { size: 8, mono: true, color: 'var(--seal-deep)' }));
  for (let i = 0; i < 7; i++) for (let j = 0; j < 8; j++) {
    body.push(`<circle cx="${240 + j * 14}" cy="${90 + i * 14}" r="3.5" fill="var(--ink)"/>`);
  }
  body.push(cap(60, 214, '红 ✗ = 故障卡 → 整实例不可用', { size: 7, color: 'var(--seal-deep)' }));
  body.push(cap(52, 226, '384/64 = 6 → 384/56 ≈ 6.857 experts/GPU', { size: 7.5, mono: true }));
  body.push(cap(52, 238, '每卡权重流量 ×64/56 ≈ 1.143（+14.3%）· EPLB 重排 expert', { size: 7 }));

  body.push(zone({ x: 372, y: 40, w: 312, h: 200, label: '命题 5：SLO 缺口与重启时间解耦' }));
  const rp1 = mk('rp1', 388, 72, 130, 36, 'cache', '请求 rerouting', 'ms 级 · 健康落点', 8.5, 6);
  const rp2 = mk('rp2', 540, 72, 128, 36, 'attn', '续 decode', '缺口 10.7 ms < 20 ms SLO ✓', 8, 5.6);
  body.push(edge(rp1, rp2, { nodes, pulse: true, dur: 1, delay: 0.3, uid: UID, eid: 'rr' }));
  body.push(arrowHead(rp2.left[0], rp2.left[1], 1, 0));
  const rs1 = mk('rs1', 388, 132, 130, 36, 'plain', '等待重启', 'T_restart ≈ 60 s', 8.5, 6.5);
  const rs2 = mk('rs2', 540, 132, 128, 36, 'data', 'stall 3000× SLO', '60 s / 20 ms ✗ 量纲走不通', 8, 5.6);
  body.push(edge(rs1, rs2, { nodes, dash: true, uid: UID, eid: 'rs' }));
  body.push(arrowHead(rs2.left[0], rs2.left[1], 1, 0));
  body.push(cap(388, 192, '缩短 60→30 s 仍 1500× —— 这条路量纲上走不通', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(388, 208, '容错 = PE/CE quota 冗余 + 请求级 rerouting（doc L261）', { size: 7.5 }));
  body.push(cap(388, 224, '重启 = 容量恢复（慢后台）· rerouting = 请求保活（快前台）', { size: 7 }));

  body.push(zone({ x: 36, y: 256, w: 648, h: 84, label: '两层分工：引擎侧 elastic EP 状态机（6.3）vs 平台层 failover（doc）' }));
  mk('e1', 52, 284, 300, 44, 'proj', '引擎侧：运行时扩缩不靠重启', 'SGLang ScaleCohort L30 / StateManager L132 + expert 备份', 8.5, 5.8);
  mk('e2', 372, 284, 296, 44, 'op', '平台侧：故障后实收 GPU 数为 EP', '重启重定 + EPLB 重排（分钟级一部分）· vLLM elastic_ep 对照', 8, 5.8);

  body.push(cap(36, 358, '重定 size 代价：每卡负载升、整实例聚合吞吐降（卡少了）；收益 = 实例仍可用 —— 健康 GPU 继续服务，而非整机停摆等齐 64 张', { size: 9.5 }));
  body.push(cap(36, 376, '式 (8) 是每卡权重流量结构性比例（∝E/e），非实测吞吐比 —— 吞吐还受通信/同步与 EP 拐点影响（EP>32 递减，7.4）', { size: 9.5 }));
  body.push(cap(36, 394, '机制-策略分离：目标不是让慢机制变快，而是让快机制不必等慢机制 —— 冗余 quota 是两者之间的缓冲（§5）', { size: 9.5 }));
  return scene(720, 412, body, { label: 'EP failover：六十四卡单通信域一张卡故障整实例不可用，重启从健康 GPU 重定 EP 为五十六每卡专家六点八五七流量升一点一四三倍；SLO 靠请求级 rerouting 缺口十点七毫秒小于二十毫秒而非等待六十秒重启三千倍 SLO；引擎侧运行时状态机与平台层重启重定两层分工' }).svg();
}

/* ============================================================
 * 注册表：dim:'inference'，slug=08-distributed-and-pd/<file-stem>
 * replacesBlock 序号经 census 核实（无语言围栏 0 计）；afterHeading 各命中唯一一行。
 * ============================================================ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/01-serving-parallelism',
    replacesBlock: 0,
    title: 'TP serving：每层 2 次前向归约 × L，每步 2L = 80 次（非 4 非 1）',
    svg: tpForwardSites,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/01-serving-parallelism',
    replacesBlock: 1,
    title: 'decode PP token 级流水：层间 p−1 次 P2P vs TP 层内 2L 次',
    svg: ppTokenPipeline,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/01-serving-parallelism',
    afterHeading: /^## 4\. /,
    title: 'EP serving：dispatch/combine 延迟账 × 可重叠上界 1.24 ms',
    svg: epDispatchCombine,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/01-serving-parallelism',
    replacesBlock: 2,
    title: 'DP attention × #DP=#EP：边界 0 次额外通信，KV 读降 8×',
    svg: dpAttentionBoundary,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/02-pd-disaggregation',
    replacesBlock: 0,
    title: '干扰时序：32768-token prefill 劫持同批 decode，TBT 尖刺 52×',
    svg: pdInterference,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/02-pd-disaggregation',
    afterHeading: /^## 2\. /,
    title: 'goodput 分解：min 约束 → 调和形 3.59 → 整数 2P+1D 3.33 rps/GPU',
    svg: goodputDecomp,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/02-pd-disaggregation',
    afterHeading: /^## 4\. /,
    title: 'xPyD 配比：λ 与 TPOT 相消，只剩 workload 形状 × 硬件参数',
    svg: xpydRatio,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/02-pd-disaggregation',
    afterHeading: /^## 5\. /,
    title: '传输 vs 重算：临界是带宽阈值 BW* = 0.376 GB/s，不是长度',
    svg: bwThreshold,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/03-kv-transfer',
    replacesBlock: 0,
    title: 'P2D 时序：layer-wise 暴露 85.9 → 2.68 ms（÷L=32，计算主导）',
    svg: layerwisePipeline,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/03-kv-transfer',
    replacesBlock: 1,
    title: 'NIXL 双平面：metadata 控制面 × KV 数据面 RDMA 直传（18 插件）',
    svg: nixlPlanes,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/03-kv-transfer',
    replacesBlock: 2,
    title: 'Mooncake TE：topology matrix 选路 × >64 KB 切 slice 多 NIC 池化 200 GB/s',
    svg: mooncakeTopology,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/03-kv-transfer',
    replacesBlock: 3,
    title: '跨 DC P/D：FP4 压缩使可行（60.4 ms 余量 3.31×），可靠性付代价',
    svg: crossDC,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/03-kv-transfer',
    afterHeading: /^## 5\. /,
    title: 'GDR：RDMA NIC 直写 HBM 不占 SM × bounce buffer 聚合小 fragment',
    svg: gdrPaths,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/04-pd-systems',
    replacesBlock: 0,
    title: 'Dynamo disagg 编排：KV-aware 路由 → prefill → NIXL 直传 → decode',
    svg: dynamoFlow,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/04-pd-systems',
    afterHeading: /^## 8\. /,
    title: '六系统对照：KV 路径三型 × 编排三型 × 开源完整度',
    svg: pdSystemsCompare,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/05-elasticity-and-fault-tolerance',
    afterHeading: /^## 2\. /,
    title: '伸缩触发账：prefill non-cached tokens（8 台）× decode running reqs（12 台）',
    svg: autoscaleTriggers,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/05-elasticity-and-fault-tolerance',
    afterHeading: /^## 4\. /,
    title: 'grace 1 min：长尾请求连同 KV 一次性转发，暴露 10.7 ms ≪ 重算 133 ms',
    svg: graceMigration,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/05-elasticity-and-fault-tolerance',
    afterHeading: /^## 5\. /,
    title: '冗余 quota：u > 0.8 禁 scale-in，f_max = 1−u ≥ 0.2 故障落点',
    svg: redundancyQuota,
  },
  {
    dim: 'inference',
    slug: '08-distributed-and-pd/05-elasticity-and-fault-tolerance',
    afterHeading: /^## 6\. /,
    title: 'EP failover：单卡故障整实例不可用 → 重定 EP size × rerouting 保 SLO',
    svg: epFailover,
  },
];