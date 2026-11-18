// training Part 1（01-autograd-and-backprop）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 part01.mjs / part04.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。

import { scene, node, edge, zone, cap, bars, PULSE_GRAD_ID } from '../figure-kit.mjs';

// 手工路径上的脉冲（L 形/回环/折线通道用 path 直写；渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 1.1 §2（block 0）— 双层 MLP 计算图：前向数据流 × 反向 adjoint 流
 * 原文：y = (x·W1)·W2 + b，L = mean(y²)；前向节点 = 算子、边 = tensor（标 shape）；
 * 反向 = 式 (4) 的逐节点 vjp，种子 dL/dL = 1；dL/dh1 = dL/dh2·W2ᵀ，
 * dL/dW1 = xᵀ·dL/dh1，dL/dx = dL/dh1·W1ᵀ
 * ============================================================ */
function computationGraph() {
  const UID = 't1a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, '最小非平凡例子 y = (x·W₁)·W₂ + b，L = mean(y²)：前向数据流与反向 adjoint 流共用同一张 DAG', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(36, 38, '反向 = 式 (4) 的逐节点 vjp：每步只调该算子的闭式，Jacobian 永不物化', { size: 10.5 }));

  /* ---- 左：前向 lane（自上而下） ---- */
  body.push(zone({ x: 36, y: 50, w: 300, h: 352, label: '前向（节点 = 算子，边 = tensor）' }));
  const fx = (id, y, title, sub, role = 'data') => {
    const n = node({ id, x: 66, y, w: 130, h: 40, role, title, sub, mono: true, titleSize: 13 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const nx = fx('x', 76, 'x', '[B,T,d1]');
  const nh1 = fx('h1', 148, 'h1', '[B,T,d2]');
  const nh2 = fx('h2', 220, 'h2', '[B,T,d3]');
  const ny = fx('y', 292, 'y', '[B,T,d3]');
  const nL = fx('L', 352, 'L', '标量', 'attn');

  // 前向边（算子名贴在左缘）
  const fwd = [
    [nx, nh1, 'mm(W₁)'],
    [nh1, nh2, 'mm(W₂)'],
    [nh2, ny, 'add(+b)'],
    [ny, nL, 'mean(·²)'],
  ];
  fwd.forEach(([a, b, label], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.3, delay: i * 0.35, uid: UID, eid: `f${i}` }));
    body.push(cap(58, (a.bottom[1] + b.top[1]) / 2 + 4, label, { mono: true, size: 9.5, anchor: 'end' }));
  });
  // 参数边（cache 小芯片喂进前向边中点）
  const params = [
    ['W1', 'W₁ [d1,d2]', 119],
    ['W2', 'W₂ [d2,d3]', 191],
    ['b', 'b [d3]', 263],
  ];
  params.forEach(([id, label, y], i) => {
    const p = node({ id, x: 216, y, w: 104, h: 26, role: 'cache', title: label, mono: true, titleSize: 10.5 });
    nodes.push(p); body.push(p.svg);
    body.push(edge(p, [131, y + 13], { nodes, portFrom: p.left, uid: UID, eid: `p${i}`, arrow: true }));
  });
  body.push(cap(268, 306, '参数边', { size: 9.5, anchor: 'middle' }));

  /* ---- 右：反向 lane（自下而上，adjoint） ---- */
  body.push(zone({ x: 376, y: 50, w: 320, h: 352, label: '反向（式 (4) 逐节点 vjp）' }));
  const bx = (id, y, title, sub) => {
    const n = node({ id, x: 436, y, w: 140, h: 40, role: 'proj', title, sub, mono: true, titleSize: 12 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const bseed = node({ id: 'seed', x: 436, y: 352, w: 140, h: 40, role: 'attn', title: 'dL/dL = 1', sub: '种子 adjoint', mono: true, titleSize: 12 });
  nodes.push(bseed); body.push(bseed.svg);
  const bh2 = bx('bh2', 292, 'dL/dh2', '[B,T,d3]');
  const bh1 = bx('bh1', 220, 'dL/dh1', '[B,T,d2]');
  const bxx = bx('bx', 148, 'dL/dx', '[B,T,d1]');

  // 损失 → 种子
  body.push(edge(nL, bseed, { nodes, pulse: true, dur: 1.6, delay: 1.6, uid: UID, eid: 's', strong: true }));
  // 反向链（向上），公式贴右缘
  const bwd = [
    [bseed, bh2, 'mean/add 的 vjp（逐元素）'],
    [bh2, bh1, 'dL/dh1 = dL/dh2·W2ᵀ'],
    [bh1, bxx, 'dL/dx = dL/dh1·W1ᵀ'],
  ];
  bwd.forEach(([a, b, label], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.top, portTo: b.bottom, pulse: true, dur: 1.3, delay: 2.1 + i * 0.35, uid: UID, eid: `b${i}` }));
    body.push(cap(586, (a.top[1] + b.bottom[1]) / 2 + 4, label, { mono: true, size: 9.5 }));
  });
  body.push(cap(36, 428, '权重梯度：dL/dW1 = xᵀ·dL/dh1，dL/dW2 = h1ᵀ·dL/dh2（跨 token 求和，§2 推论 1）', { mono: true, size: 10 }));
  body.push(cap(36, 446, 'fan-out 处沿全部路径求和（式 (2)）—— .grad 的 += 累加语义由此而来', { size: 10.5 }));

  return scene(720, 458, body, { label: '双层 MLP 的前向数据流与反向 adjoint 流：反向从种子 dL/dL=1 出发逐节点执行 vjp' }).svg();
}

/* ============================================================
 * 1.2 §2（block 0）— matmul 前向/反向三式与 FLOPs 账
 * 原文：前向 A[m,k]@B[k,n]=Y[m,n] 2mnk；反向 dA = G@Bᵀ 2mnk、dB = Aᵀ@G 2mnk；
 * 合计 4mnk = 2×前向（6ND 反向 4N 的精确出处）
 * ============================================================ */
function matmulBackward() {
  const UID = 't1b0';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'matmul 的反向 = 两次同形 matmul：每个可微输入各一份 vjp，FLOPs 比恰为 2', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 前向行 ---- */
  body.push(cap(56, 52, '前向', { size: 11, weight: '650' }));
  const A = node({ id: 'A', x: 56, y: 64, w: 120, h: 40, role: 'data', title: 'A [m, k]', mono: true, titleSize: 12.5 });
  const B = node({ id: 'B', x: 220, y: 64, w: 120, h: 40, role: 'data', title: 'B [k, n]', mono: true, titleSize: 12.5 });
  const mm = node({ id: 'mm', x: 412, y: 64, w: 150, h: 40, role: 'proj', title: 'Y = A·B', sub: '[m, n]', mono: true, titleSize: 12.5 });
  [A, B, mm].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(A, mm, { nodes, pulse: true, dur: 1.4, uid: UID, eid: 'fa' }));
  body.push(edge(B, mm, { nodes, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'fb' }));
  body.push(cap(294, 52, '2mnk', { mono: true, size: 11, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));

  /* ---- 反向行 ---- */
  body.push(cap(56, 140, '反向', { size: 11, weight: '650' }));
  const G = node({ id: 'G', x: 560, y: 152, w: 140, h: 44, role: 'attn', title: 'G = ∂L/∂Y', sub: '[m, n]', mono: true, titleSize: 12 });
  const Bt = node({ id: 'Bt', x: 380, y: 152, w: 110, h: 44, role: 'cache', title: 'Bᵀ [n, k]', sub: 'save 自前向', mono: true, titleSize: 11.5 });
  const dA = node({ id: 'dA', x: 190, y: 152, w: 130, h: 44, role: 'proj', title: 'dA = G·Bᵀ', sub: '[m, k]', mono: true, titleSize: 12 });
  const At = node({ id: 'At', x: 380, y: 226, w: 110, h: 44, role: 'cache', title: 'Aᵀ [k, m]', sub: 'save 自前向', mono: true, titleSize: 11.5 });
  const dB = node({ id: 'dB', x: 190, y: 226, w: 130, h: 44, role: 'proj', title: 'dB = Aᵀ·G', sub: '[k, n]', mono: true, titleSize: 12 });
  [G, Bt, dA, At, dB].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(G, Bt, { nodes, pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: 'g1' }));
  body.push(edge(Bt, dA, { nodes, pulse: true, dur: 1.2, delay: 1.15, uid: UID, eid: 'g2' }));
  body.push(edge(G, At, { nodes, portFrom: G.bottom, portTo: At.right, pulse: true, dur: 1.4, delay: 1.05, uid: UID, eid: 'g3' }));
  body.push(edge(At, dB, { nodes, pulse: true, dur: 1.2, delay: 1.3, uid: UID, eid: 'g4' }));
  body.push(cap(350, 146, '2mnk', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(350, 220, '2mnk', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 形状对账
  body.push(cap(56, 300, '形状对账：G·Bᵀ = [m,n]×[n,k] = [m,k] ✓ · Aᵀ·G = [k,m]×[m,n] = [k,n] ✓（梯度必与参数同形）', { mono: true, size: 10 }));
  body.push(cap(56, 322, '合计 4mnk = 2 × 前向 —— 「反向 ≈ 2× 前向」的精确出处，6ND 的 4N 由此而来', { size: 11, color: 'var(--seal-deep)', weight: '650' }));

  return scene(720, 340, body, { label: 'matmul 反向：dA = G·Bᵀ 与 dB = Aᵀ·G 各一次同形 matmul，合计恰为前向两倍' }).svg();
}

/* ============================================================
 * 1.2 §10（block 1）— fused / chunked CE：logits 永不全量物化
 * 原文：H [M,d] 按行分块 M_c → 片上投影 Z_c = H_c·Wᵀ [M_c,V]（临时 buffer，不落 HBM）
 * → 行内 logsumexp（V 维流式）→ partial loss（跨块累加）；反向 p−e_y 就地覆写 Z_c
 * → dH_c = (p−e_y)·W，dW += H_cᵀ·(p−e_y)（跨块累加）
 * ============================================================ */
function fusedChunkedCE() {
  const UID = 't1b1';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'fused / chunked CE：M 行分块过输出头，[M, V] logits 永不全量物化', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 前向 lane ---- */
  const H = node({ id: 'H', x: 36, y: 66, w: 120, h: 52, role: 'data', title: 'H [M, d]', sub: '隐状态（本就要存）', mono: true, titleSize: 12.5 });
  nodes.push(H); body.push(H.svg);
  body.push(cap(96, 140, '按行分块 · 块长 M_c', { size: 10, anchor: 'middle' }));

  // 分块 chunk ×3（逐块滑入，sig-cell 点亮）
  const chunks = [];
  for (let i = 0; i < 3; i++) {
    const c = node({ id: `c${i}`, x: 176 + i * 14, y: 46 + i * 34, w: 96, h: 30, role: 'data', title: `H_c${['¹', '²', '³'][i]}`, mono: true, titleSize: 11 });
    nodes.push(c); chunks.push(c);
    body.push(`<g class="sig-cell" style="--i:${(i * 0.5).toFixed(1)}">${c.svg}</g>`);
  }
  body.push(cap(224, 172, '… 逐块滑入', { size: 9.5, anchor: 'middle' }));

  // 片上 buffer zone（不落 HBM）
  body.push(zone({ x: 306, y: 40, w: 200, h: 148, label: '片上 buffer（不落 HBM）', tone: 'seal' }));
  const Z = node({ id: 'Z', x: 330, y: 88, w: 152, h: 52, role: 'proj', title: 'Z_c = H_c·Wᵀ', sub: '[M_c, V] 临时物化', mono: true, titleSize: 12 });
  nodes.push(Z); body.push(Z.svg);

  const LS = node({ id: 'ls', x: 546, y: 56, w: 150, h: 40, role: 'op', title: '行内 logsumexp', sub: 'V 维流式 max / sum', mono: true, titleSize: 12 });
  const PL = node({ id: 'pl', x: 546, y: 130, w: 150, h: 44, role: 'attn', title: 'partial loss', sub: '标量 · 跨块累加', mono: true, titleSize: 12 });
  nodes.push(LS, PL); body.push(LS.svg, PL.svg);

  body.push(edge(H, chunks[1], { nodes, pulse: true, dur: 1.5, uid: UID, eid: 'h0' }));
  chunks.forEach((c, i) => {
    body.push(edge(c, [330, 114], { nodes, portFrom: c.right, pulse: true, dur: 1.5, delay: 0.3 + i * 0.5, uid: UID, eid: `c${i}`, arrow: i === 2 }));
  });
  body.push(edge(Z, LS, { nodes, portFrom: Z.right, portTo: LS.left, pulse: true, dur: 1.4, delay: 1.1, uid: UID, eid: 'z1' }));
  body.push(edge(LS, PL, { nodes, portFrom: LS.bottom, portTo: PL.top, pulse: true, dur: 1.2, delay: 1.5, uid: UID, eid: 'z2' }));

  /* ---- 反向 lane：p − e_y 就地覆写 ---- */
  const PE = node({ id: 'pe', x: 330, y: 232, w: 152, h: 44, role: 'attn', title: 'p − e_y（式 5）', sub: '就地覆写 Z_c', mono: true, titleSize: 12 });
  nodes.push(PE); body.push(PE.svg);
  body.push(`<circle class="sig-diag" style="--i:1" cx="406" cy="182" r="8" fill="none" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(edge(Z, PE, { nodes, portFrom: Z.bottom, portTo: PE.top, pulse: true, dur: 1.3, delay: 2.0, uid: UID, eid: 'b0', dash: true }));

  const dH = node({ id: 'dH', x: 60, y: 296, w: 210, h: 48, role: 'proj', title: 'dH_c = (p − e_y)·W', sub: '[M_c, d]', mono: true, titleSize: 12 });
  const dW = node({ id: 'dW', x: 436, y: 296, w: 240, h: 48, role: 'proj', title: 'dW += H_cᵀ·(p − e_y)', sub: '[d, V] · 跨块累加（§2 推论 1）', mono: true, titleSize: 12 });
  nodes.push(dH, dW); body.push(dH.svg, dW.svg);
  body.push(edge(PE, dH, { nodes, portFrom: PE.left, portTo: dH.top, pulse: true, dur: 1.5, delay: 2.4, uid: UID, eid: 'b1' }));
  body.push(edge(PE, dW, { nodes, portFrom: PE.right, portTo: dW.top, pulse: true, dur: 1.5, delay: 2.55, uid: UID, eid: 'b2' }));
  body.push(cap(406, 224, '反向（块内重算或前向顺手完成）', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 376, '峰值显存 O(M_c·V)：M_c = 2048、V = 128256 → 块 logits ≈ 0.49 GiB（完整 [M,V] bf16 logits 7.8 GiB 的 1/16）', { mono: true, size: 10 }));
  body.push(cap(36, 394, 'logits 是损失的直接输入，checkpointing 省不掉它 —— fused CE 必须单独立账；Liger 口径：吞吐 +20%、显存 −60%', { size: 10.5 }));

  return scene(720, 408, body, { label: 'fused/chunked CE：隐状态按行分块在片上投影出 logits 块，logsumexp 流式聚合，反向以 p−e_y 就地覆写' }).svg();
}

/* ============================================================
 * 1.3 §3（block 0）— fp16 混合精度环路
 * 原文：fp32 主权重 θ → round_16 → fp16 副本 w → 前向（GEMM fp16×fp32 累加）→ L
 * → ×S loss scaling → backward（S·g fp16 梯度）→ unscale ÷S + inf/NaN 检查
 * → fp32 梯度 g → optimizer.step(θ)
 * ============================================================ */
function mixedPrecisionLoop() {
  const UID = 't1c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'fp16 混合精度一个 step：式 (3) 先放大 S 倍再除回，实数恒等 —— 全部工程在「哪一步以什么精度执行」', { size: 12, color: 'var(--ink-soft)' }));

  /* fp32 高精度侧（左列） */
  body.push(zone({ x: 30, y: 40, w: 180, h: 330, label: 'fp32（高精度侧）' }));
  const TH = node({ id: 'th', x: 52, y: 66, w: 136, h: 52, role: 'cache', title: 'θ fp32 主权重', sub: 'optimizer 持有', mono: true, titleSize: 12 });
  const UN = node({ id: 'un', x: 52, y: 170, w: 136, h: 48, role: 'op', title: 'unscale', sub: '÷S + inf/NaN 检查', mono: true, titleSize: 12 });
  const G3 = node({ id: 'g3', x: 52, y: 240, w: 136, h: 40, role: 'data', title: 'g fp32', mono: true, titleSize: 12.5 });
  const ST = node({ id: 'st', x: 52, y: 302, w: 136, h: 46, role: 'op', title: 'optimizer.step', sub: '更新 θ', mono: true, titleSize: 11.5 });
  [TH, UN, G3, ST].forEach((n) => { nodes.push(n); body.push(n.svg); });
  // inf/NaN 哨卡光环（卡在 unscale 闸门角上）
  body.push(`<circle class="sig-diag" style="--i:1.2" cx="192" cy="168" r="7" fill="none" stroke="var(--seal)" stroke-width="1.2"/>`);

  /* fp16 前向反向区（右侧大区） */
  body.push(zone({ x: 240, y: 40, w: 450, h: 236, label: 'fp16 前向反向（乘 fp16 · 累加 fp32）', tone: 'seal' }));
  const W = node({ id: 'w', x: 262, y: 66, w: 130, h: 48, role: 'data', title: 'w fp16 副本', sub: 'round_16(θ)', mono: true, titleSize: 12 });
  const FW = node({ id: 'fw', x: 432, y: 66, w: 150, h: 48, role: 'op', title: '前向', sub: 'GEMM fp16×fp32 累加', mono: true, titleSize: 12.5 });
  const LL = node({ id: 'll', x: 616, y: 66, w: 62, h: 48, role: 'attn', title: 'L', mono: true, titleSize: 14 });
  const SC = node({ id: 'sc', x: 610, y: 150, w: 74, h: 40, role: 'op', title: '×S', sub: '放大', mono: true, titleSize: 12.5 });
  const BK = node({ id: 'bk', x: 392, y: 150, w: 150, h: 48, role: 'op', title: '反向', sub: 'vjp 链（线性 ⇒ S 穿透）', mono: true, titleSize: 12.5 });
  const SG = node({ id: 'sg', x: 240, y: 150, w: 118, h: 48, role: 'data', title: 'S·g', sub: 'fp16 梯度', mono: true, titleSize: 12.5 });
  [W, FW, LL, SC, BK, SG].forEach((n) => { nodes.push(n); body.push(n.svg); });

  /* 环路连线（顺时针脉冲） */
  body.push(edge(TH, W, { nodes, portFrom: TH.right, portTo: W.left, pulse: true, dur: 1.4, uid: UID, eid: 'a0' }));
  body.push(cap(225, 106, 'round_16', { mono: true, size: 9, anchor: 'middle' }));
  body.push(edge(W, FW, { nodes, pulse: true, dur: 1.4, delay: 0.3, uid: UID, eid: 'a1' }));
  body.push(edge(FW, LL, { nodes, pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: 'a2' }));
  body.push(edge(LL, SC, { nodes, portFrom: LL.bottom, portTo: SC.top, pulse: true, dur: 1.1, delay: 0.9, uid: UID, eid: 'a3' }));
  body.push(edge(SC, BK, { nodes, portFrom: SC.left, portTo: BK.right, pulse: true, dur: 1.3, delay: 1.1, uid: UID, eid: 'a4' }));
  body.push(edge(BK, SG, { nodes, pulse: true, dur: 1.2, delay: 1.35, uid: UID, eid: 'a5' }));
  body.push(edge(SG, UN, { nodes, portFrom: SG.left, portTo: UN.right, pulse: true, dur: 1.3, delay: 1.6, uid: UID, eid: 'a6' }));
  body.push(edge(UN, G3, { nodes, portFrom: UN.bottom, portTo: G3.top, pulse: true, dur: 1.1, delay: 1.85, uid: UID, eid: 'a7' }));
  body.push(edge(G3, ST, { nodes, portFrom: G3.bottom, portTo: ST.top, pulse: true, dur: 1.1, delay: 2.05, uid: UID, eid: 'a8' }));
  // step → θ 回环（左缘 L 形）
  const ret = 'M 52 325 H 18 V 92 H 52';
  body.push(`<path class="dia-edge strong" d="${ret}"/>`);
  body.push(pathPulse(ret, { dur: 2.2, delay: 2.3 }));
  body.push(cap(14, 292, 'θ ← θ − η·u(g)', { mono: true, size: 9, anchor: 'start' }));

  // 旁路：fp16 激活存给反向
  body.push(edge(W, BK, { nodes, portFrom: W.bottom, portTo: BK.top, uid: UID, eid: 'x', dash: true, arrow: true }));
  body.push(cap(382, 138, 'fp16 激活存给反向（1.4 的账）', { size: 9, anchor: 'middle' }));

  body.push(cap(240, 306, 'S 的可行窗口（式 4）：2⁻²⁴/g_min ≤ S < 65504/g_max —— 下不压死、上不溢出', { mono: true, size: 10 }));
  body.push(cap(240, 322, 'bf16 与 fp32 同动态范围 ⇒ 这套 scaling 整个消失（§4），但 master weights 更不能省', { size: 10.5 }));
  body.push(cap(240, 340, 'S₀ = 2¹⁶ · 2000 步无溢出则 ×2 · 发现 inf/NaN 则跳步并 ÷2（GradScaler 口径）', { mono: true, size: 9.5 }));

  return scene(720, 390, body, { label: 'fp16 混合精度环路：fp32 主权重舍入出 fp16 副本跑前向反向，梯度经 unscale 与溢出检查后回 fp32 更新' }).svg();
}

/* ============================================================
 * 1.4 §2（block 0）— 分段 checkpoint 时序
 * 原文：前向 k 段只存边界 x0..xk；反向逐段取边界重算（激活即用即弃）后 vjp；
 * 峰值 = k 份 checkpoint + 1 段重算激活 = 式 (2) M(k) = a(k + L/k)
 * ============================================================ */
function checkpointTimeline() {
  const UID = 't1d';
  const body = [];
  const nodes = [];
  const K = 4; // 示意 4 段（原文 k 段 + 省略号）
  const BOUND = [96, 210, 324, 438, 552]; // x0..x4 中心
  const BY = 76; // 前向 lane y

  body.push(cap(36, 22, 'L 层切 k 段：前向只存 k 个段边界，反向逐段「取边界 → 重算 → vjp → 即弃」', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 前向 lane ---- */
  body.push(cap(56, BY + 22, '前向', { size: 11, weight: '650' }));
  const bounds = BOUND.map((cx, i) => {
    const n = node({ id: `x${i}`, x: cx - 22, y: BY, w: 44, h: 36, role: 'cache', title: `x${['₀', '₁', '₂', '₃', '₄'][i]}`, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(`<g class="sig-cell" style="--i:${(i * 0.35).toFixed(2)}">${n.svg}</g>`);
    return n;
  });
  const loss = node({ id: 'loss', x: 630, y: BY, w: 56, h: 36, role: 'attn', title: 'L', mono: true, titleSize: 13 });
  nodes.push(loss); body.push(loss.svg);
  for (let i = 0; i < K; i++) {
    body.push(edge(bounds[i], bounds[i + 1], { nodes, pulse: true, dur: 1.1, delay: i * 0.35, uid: UID, eid: `f${i}` }));
    body.push(cap((BOUND[i] + BOUND[i + 1]) / 2, BY - 8, `段${i + 1}`, { size: 9.5, anchor: 'middle' }));
  }
  body.push(edge(bounds[K], loss, { nodes, pulse: true, dur: 1.1, delay: K * 0.35, uid: UID, eid: 'fl' }));
  // 驻留括号
  body.push(`<line class="dia-edge strong" x1="${BOUND[0]}" y1="${BY + 46}" x2="${BOUND[K]}" y2="${BY + 46}"/>`);
  body.push(cap((BOUND[0] + BOUND[K]) / 2, BY + 60, 'k 份边界 checkpoint 全程驻留（cache 色）', { size: 10, anchor: 'middle' }));

  /* ---- 反向 lane ---- */
  const KY = 168;
  body.push(cap(56, KY + 24, '反向', { size: 11, weight: '650' }));
  // 当前活动段（段4）：重算窗口 + vjp；其余段以省略号暗示
  const rc = node({ id: 'rc', x: 438, y: KY, w: 136, h: 44, role: 'attn', title: '重算段k', sub: '激活即用即弃', mono: true, titleSize: 12 });
  const vj = node({ id: 'vj', x: 288, y: KY, w: 110, h: 44, role: 'op', title: 'vjp', sub: '段内逐算子', mono: true, titleSize: 12.5 });
  nodes.push(rc, vj); body.push(rc.svg, vj.svg);
  // 取边界（虚线从 x3 落下）
  body.push(edge([BOUND[3], BY + 36], rc, { nodes, portTo: rc.top, uid: UID, eid: 'take', dash: true }));
  body.push(cap(300, 156, '取 x_{k−1}（+ 恢复 RNG 态）', { size: 9.5, anchor: 'middle' }));
  body.push(edge(loss, rc, { nodes, portFrom: loss.bottom, portTo: rc.right, pulse: true, dur: 1.4, delay: 2.2, uid: UID, eid: 'b0' }));
  body.push(edge(rc, vj, { nodes, pulse: true, dur: 1.3, delay: 2.5, uid: UID, eid: 'b1' }));
  body.push(cap(210, KY + 44, '… 段k−1 … 段2 … 段1 同法', { size: 10.5, anchor: 'middle' }));
  body.push(edge(vj, [120, KY + 22], { nodes, pulse: true, dur: 1.6, delay: 2.9, uid: UID, eid: 'b2', dash: true }));

  /* ---- 峰值账 ---- */
  body.push(bars({
    x: 96, y: 268, h: 18, unit: 34,
    segs: [
      { w: 4, tone: 'warm' },
      { w: 1, tone: 'seal' },
    ],
  }));
  body.push(cap(164, 258, 'k 份 checkpoint · k·a 驻留', { size: 10, anchor: 'middle' }));
  body.push(cap(246, 262, '1 段重算 · (L/k)·a 峰值', { size: 10 }));
  body.push(cap(96, 316, '峰值内存 M(k) = k·a + (L/k)·a（式 2）—— k* = √L 时 M = 2a√L（式 3），激活从 O(L) 压到 O(√L)', { mono: true, size: 10.5 }));
  body.push(cap(96, 334, '代价：一次额外前向 ⇒ step 3 份 → 4 份（+33%）；selective 只重算 attention 核心可压到个位数 %（§3）', { size: 10.5 }));

  return scene(720, 352, body, { label: 'gradient checkpointing：前向只存段边界，反向逐段重算激活后 vjp，峰值内存 k·a + (L/k)·a' }).svg();
}

/* ============================================================
 * 1.5 §4（block 0）— naive vs no_sync 双泳道
 * 原文：4 个 micro-batch；naive 每 micro 反向都触发一轮 all-reduce（每 step 4 轮）；
 * no_sync 前 m−1 个 hook 关闭，末 micro 一次同步（每 step 1 轮）→ clip → step → zero_grad
 * ============================================================ */
function gradAccumNoSync() {
  const UID = 't1e';
  const body = [];
  const nodes = [];
  const COLS = [150, 290, 430, 570]; // micro 1..4 中心
  const CW = 112;

  body.push(cap(36, 22, 'DDP + 梯度累加：no_sync 把 all-reduce 从每 step m 轮压到 1 轮 —— 通信总量 ÷ m', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(52, 44, '时间 →', { size: 10 }));
  COLS.forEach((cx, i) => body.push(cap(cx, 44, `micro ${i + 1}`, { mono: true, size: 10, anchor: 'middle' })));

  const lane = (y, label, lastOnly) => {
    body.push(cap(88, y + 24, label, { mono: true, size: 11, anchor: 'end', weight: '650' }));
    const cells = COLS.map((cx, i) => {
      const n = node({ id: `${label}${i}`, x: cx - CW / 2, y, w: CW, h: 36, role: 'op', title: 'fwd + bwd', mono: true, titleSize: 11 });
      nodes.push(n); body.push(n.svg);
      return n;
    });
    cells.forEach((c, i) => {
      const cx = COLS[i];
      const isAR = !lastOnly || i === COLS.length - 1;
      if (isAR) {
        const ar = node({ id: `${label}ar${i}`, x: cx - 30, y: y + 52, w: 60, h: 26, role: 'attn', title: 'AR', mono: true, titleSize: 11 });
        nodes.push(ar); body.push(ar.svg);
        body.push(edge(c, ar, { nodes, portFrom: c.bottom, portTo: ar.top, pulse: true, dur: 1.2, delay: 0.4 + i * 0.3, uid: UID, eid: `${label}${i}` }));
      } else {
        body.push(`<g class="sig-cell" style="--i:${(0.3 + i * 0.3).toFixed(1)}">` +
          `<rect x="${cx - 34}" y="${y + 52}" width="68" height="26" rx="8" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>` +
          `<text class="dia-cap" x="${cx}" y="${y + 69}" font-size="9.5" text-anchor="middle">hook 关</text></g>`);
      }
    });
    return cells;
  };

  lane(64, 'naive', false);
  lane(174, 'no_sync', true);
  body.push(cap(660, 122, '每 step 4 轮', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(660, 232, '每 step 1 轮', { size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));

  // 收尾序列
  const fin = node({ id: 'fin', x: 500, y: 282, w: 196, h: 34, role: 'plain', title: 'clip → step → zero_grad', mono: true, titleSize: 11 });
  nodes.push(fin); body.push(fin.svg);
  body.push(edge([570, 252], fin, { nodes, portTo: fin.top, uid: UID, eid: 'fin', strong: true }));
  body.push(cap(96, 300, '同步的是「累加完成的全 batch 梯度」—— clip 必须在 AR 之后（2.4 的顺序语义）', { size: 10 }));
  body.push(cap(96, 318, '账：bf16 梯度 2N bytes（N = 7×10⁹ ≈ 13 GiB）；m = 8 时每 rank 每 step 209 → 26 GiB', { mono: true, size: 10 }));

  return scene(720, 336, body, { label: 'naive 每个 micro-batch 都 all-reduce；no_sync 关掉同步 hook，只在最后一个 micro 反向后通信一轮' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/01-computation-graph-and-autograd',
    replacesBlock: 0,
    title: '计算图双联：前向数据流 × 反向 adjoint 流',
    svg: computationGraph,
  },
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/02-operator-backward',
    replacesBlock: 0,
    title: 'matmul 反向：两次同形 matmul = 2× 前向',
    svg: matmulBackward,
  },
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/02-operator-backward',
    replacesBlock: 1,
    title: 'fused / chunked CE：logits 永不全量物化',
    svg: fusedChunkedCE,
  },
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/03-mixed-precision',
    replacesBlock: 0,
    title: 'fp16 混合精度环路：scale · unscale · 溢出跳步',
    svg: mixedPrecisionLoop,
  },
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/04-gradient-checkpointing',
    replacesBlock: 0,
    title: '分段 checkpoint：存边界、反向重算即用即弃',
    svg: checkpointTimeline,
  },
  {
    dim: 'training',
    slug: '01-autograd-and-backprop/05-gradient-accumulation',
    replacesBlock: 0,
    title: 'no_sync：all-reduce 从 m 轮/step 压到 1 轮',
    svg: gradAccumNoSync,
  },
];
