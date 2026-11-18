// Part 7（07-diffusion-image-generation）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 part01.mjs。
// part07.mjs 的 5 张插入式签名图（噪声日程/SDE-ODE 轨迹/CFG 几何/路径几何/MMDiT 矩阵）不动，
// 本模块画各章 ASCII 原块的内容。数字、shape、机制措辞全部取自对应章节原文。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/弧线/循环通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

// 实心小三角箭头：尖端 (x,y)，运动方向单位向量 (ux,uy)
const tri = (x, y, ux, uy, s = 7, fill = 'var(--ink-soft)') => {
  const px = -uy, py = ux;
  const bx = x - s * ux, by = y - s * uy, w = s * 0.42;
  return (
    `<polygon points="${x.toFixed(1)},${y.toFixed(1)} ${(bx + w * px).toFixed(1)},${(by + w * py).toFixed(1)} ` +
    `${(bx - w * px).toFixed(1)},${(by - w * py).toFixed(1)}" fill="${fill}"/>`
  );
};

/* ============================================================
 * 7.1 §8 — 本 Part 章节地图（block 0）
 * 原文：7.1 → {7.2, 7.5}；6.3 已备 latent/tokenizer → 7.6；7.2 → 7.3 ← 7.5；
 *       7.3 → 7.4；{7.4, 7.6} → 7.7 收口。依赖序即写作与阅读序。
 * ============================================================ */
function chapterMap() {
  const UID = 'p7map';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  const m71 = add(node({ id: 'm71', x: 275, y: 16, w: 170, h: 42, role: 'attn', title: '7.1 家族地图（本章）', titleSize: 12.5 }));

  body.push(zone({ x: 24, y: 76, w: 240, h: 330, label: '推导链' }));
  const m72 = add(node({ id: 'm72', x: 44, y: 104, w: 200, h: 58, role: 'proj', title: '7.2 DDPM 完整推导', sub: '前向闭式 · ELBO → L_simple', titleSize: 12.5, subSize: 10 }));
  const m73 = add(node({ id: 'm73', x: 44, y: 212, w: 200, h: 58, role: 'proj', title: '7.3 Score 与 SDE/ODE', sub: 'DSM 等价 · PF-ODE · 精确似然', titleSize: 12.5, subSize: 10 }));
  const m74 = add(node({ id: 'm74', x: 44, y: 320, w: 200, h: 58, role: 'proj', title: '7.4 采样器与 Guidance', sub: 'DDIM · CFG · solver · 蒸馏族', titleSize: 12.5, subSize: 10 }));
  body.push(cap(40, 398, '离散（DDPM）→ 连续（SDE/ODE）→ 采样账', { size: 9.5 }));

  const m75 = add(node({ id: 'm75', x: 310, y: 104, w: 200, h: 58, role: 'proj', title: '7.5 Flow Matching', sub: 'conditional FM · 线性路径', titleSize: 12.5, subSize: 10 }));

  body.push(zone({ x: 500, y: 16, w: 212, h: 194, label: '工作空间与架构' }));
  const m63 = add(node({ id: 'm63', x: 516, y: 48, w: 180, h: 32, role: 'cache', title: '6.3 latent / tokenizer（已备）', mono: true, titleSize: 10.5 }));
  const m76 = add(node({ id: 'm76', x: 516, y: 104, w: 180, h: 58, role: 'proj', title: '7.6 LDM / DiT / MMDiT', sub: 'latent 空间 · 条件化 · 架构', titleSize: 12, subSize: 10 }));

  const m77 = add(node({ id: 'm77', x: 230, y: 424, w: 320, h: 52, role: 'data', title: '7.7 T2I 系统与 AR 生图（收口）', sub: 'SD3 / SDXL / Imagen · LlamaGen / VAR / MaskGIT', titleSize: 12.5, subSize: 10 }));

  // 依赖流（脉冲方向 = 阅读序）
  body.push(edge(m71, m72, { nodes, portFrom: m71.bottom, portTo: m72.top, pulse: true, dur: 2.2, uid: UID, eid: 0 }));
  body.push(edge(m71, m75, { nodes, portFrom: m71.bottom, portTo: m75.top, pulse: true, dur: 2.2, delay: 0.3, uid: UID, eid: 1 }));
  body.push(edge(m63, m76, { nodes, portFrom: m63.bottom, portTo: m76.top, dash: true, uid: UID, eid: 2 }));
  body.push(edge(m72, m73, { nodes, portFrom: m72.bottom, portTo: m73.top, pulse: true, dur: 2.2, delay: 0.5, uid: UID, eid: 3 }));
  body.push(edge(m75, [214, 212], { nodes, portFrom: m75.bottom, pulse: true, dur: 2.6, delay: 0.7, uid: UID, eid: 4 }));
  body.push(edge(m73, m74, { nodes, portFrom: m73.bottom, portTo: m74.top, pulse: true, dur: 2.2, delay: 0.9, uid: UID, eid: 5 }));
  body.push(edge(m74, [320, 424], { nodes, portFrom: m74.bottom, pulse: true, dur: 2.6, delay: 1.1, uid: UID, eid: 6 }));
  body.push(edge(m76, [500, 424], { nodes, portFrom: m76.bottom, pulse: true, dur: 3.0, delay: 1.1, uid: UID, eid: 7 }));

  body.push(cap(40, 496, '依赖序即阅读序：7.2 → 7.3 → 7.4 推导链；7.5 第二套参数化汇入 7.3；6.3 → 7.6 给空间与架构；7.7 系统收口', { size: 11 }));

  return scene(720, 512, body, { label: 'Part 7 章节依赖地图：7.2 到 7.4 推导链、7.5 汇入、6.3 到 7.6、7.7 收口' }).svg();
}

/* ============================================================
 * 7.2 §1 — 前向 q 链 / 反向 p_θ 链 + 跳步直达（block 0）
 * 原文：q(x_t|x_{t-1}) = N(√α_t·x_{t-1}, β_t·I)；p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), σ_t²·I)；
 *       q(x_t|x_0) = N(√ᾱ_t·x_0, (1−ᾱ_t)·I)（式 5）；训练并行、采样串行，T = 1000。
 * ============================================================ */
function ddpmChain() {
  const UID = 'p7chain';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(36, 30, '前向 q（固定，无参数）：q(x_t | x_{t−1}) = N(√α_t · x_{t−1},  β_t · I)', { mono: true, size: 11 }));

  const Y = 104, H = 46;
  const x0 = add(node({ id: 'x0', x: 36, y: Y, w: 84, h: H, role: 'data', title: 'x_0', mono: true }));
  const x1 = add(node({ id: 'x1', x: 168, y: Y, w: 72, h: H, role: 'plain', title: 'x_1', mono: true }));
  const x2 = add(node({ id: 'x2', x: 288, y: Y, w: 72, h: H, role: 'plain', title: 'x_2', mono: true }));
  const xd = add(node({ id: 'xd', x: 396, y: Y, w: 56, h: H, role: 'plain', title: '···', mono: true }));
  const xT1 = add(node({ id: 'xT1', x: 488, y: Y, w: 84, h: H, role: 'plain', title: 'x_{T−1}', mono: true, titleSize: 12 }));
  const xT = add(node({ id: 'xT', x: 620, y: Y, w: 84, h: H, role: 'cache', title: 'x_T', sub: '≈ N(0, I)', mono: true, subSize: 10 }));

  // 跳步直达（式 5）：x_0 → x_t 闭式，弧形虚线
  const jump = 'M 78 104 C 200 60, 420 60, 530 104';
  body.push(`<path class="dia-edge dash" d="${jump}"/>`);
  body.push(tri(530, 102, 0.55, 0.83, 7));
  body.push(pathPulse(jump, { dur: 3.6, uid: UID, r: 3.6 }));
  body.push(cap(304, 50, '跳步直达（§2.2）：q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t) · I)（式 5）', { mono: true, size: 10, anchor: 'middle' }));

  // 前向 q 边（脉冲 = 加噪方向）
  const chain = [x0, x1, x2, xd, xT1, xT];
  for (let i = 0; i < chain.length - 1; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, pulse: true, dur: 2.0, delay: i * 0.35, uid: UID, eid: `q${i}` }));
    const mx = (chain[i].x + chain[i].w + chain[i + 1].x) / 2;
    body.push(cap(mx, Y - 10, 'q', { mono: true, size: 9.5, anchor: 'middle' }));
  }

  // 反向 p_θ 弧（下凸，方向右 → 左 = 去噪生成）
  for (let i = 1; i < chain.length; i++) {
    const a = chain[i], b = chain[i - 1];
    const ax = a.x + a.w / 2, bx = b.x + b.w / 2, yb = Y + H;
    const d = `M ${ax} ${yb} C ${ax} ${yb + 30}, ${bx} ${yb + 30}, ${bx} ${yb + 2}`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(tri(bx, yb + 1, 0, -1, 6));
    body.push(pathPulse(d, { dur: 2.8, delay: i * 0.3, uid: UID, r: 3.6 }));
    body.push(cap((ax + bx) / 2, yb + 26, 'p_θ', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  }
  body.push(cap(36, 214, '反向 p_θ（学习）：p_θ(x_{t−1} | x_t) = N(μ_θ(x_t, t), σ_t² · I)（式 10）— 采样串行 T = 1000 步', { mono: true, size: 11, color: 'var(--seal-deep)' }));

  body.push(cap(78, 248, '数据端 t = 0', { size: 10.5, anchor: 'middle' }));
  body.push(cap(704, 248, '噪声端 t = T（连续记号 t = 1）', { size: 10.5, anchor: 'end' }));

  // 不对称账：训练并行 vs 采样串行
  body.push(zone({ x: 36, y: 264, w: 316, h: 62 }));
  body.push(cap(52, 288, '训练：x_0 与 t 一步合成 x_t（式 5）', { size: 11 }));
  body.push(cap(52, 307, '任意 t 并行采出 —— 类似 1.1 teacher forcing', { size: 10.5 }));
  body.push(zone({ x: 368, y: 264, w: 316, h: 62 }));
  body.push(cap(384, 288, '采样：x_T 起 ancestral T 步串行（式 10）', { size: 11 }));
  body.push(cap(384, 307, '每步一次网络评估 = 1000 次 NFE（→ 7.4 整章）', { size: 10.5 }));

  return scene(720, 344, body, { label: 'DDPM 前向加噪链与反向去噪链：跳步直达使训练并行、采样串行' }).svg();
}

/* ============================================================
 * 7.2 §4.4 — 训练循环 vs 采样循环（block 1）
 * 原文：训练 = repeat{x_0 ~ p_data；t ~ Uniform{1..T}；ε ~ N(0,I)；x_t = √ᾱx_0+√(1−ᾱ)ε（式 6）；
 *       对 ‖ε−ε_θ(x_t,t)‖² 一步梯度下降} until 收敛；
 *       采样 = x_T ~ N(0,I)；for t=T..1：z ~ N(0,I)（t=1 时 z=0）；x_{t-1} = (1/√α_t)(x_t
 *       − β_t/√(1−ᾱ_t)·ε_θ(x_t,t)) + σ_t·z（式 10)(18)）；return x_0。
 * ============================================================ */
function ddpmTrainSample() {
  const UID = 'p7ts';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  /* ---- 左：训练（并行）---- */
  body.push(zone({ x: 20, y: 44, w: 336, h: 244, label: '训练（ε-参数化 + L_simple）— 并行' }));
  const sx0 = add(node({ id: 'sx0', x: 36, y: 84, w: 88, h: 36, role: 'data', title: 'x_0 ~ p_data', mono: true, titleSize: 10 }));
  const st = add(node({ id: 'st', x: 132, y: 84, w: 88, h: 36, role: 'plain', title: 't ~ U{1..T}', mono: true, titleSize: 10 }));
  const seps = add(node({ id: 'seps', x: 228, y: 84, w: 88, h: 36, role: 'plain', title: 'ε ~ N(0, I)', mono: true, titleSize: 10 }));
  const comp = add(node({ id: 'comp', x: 126, y: 148, w: 116, h: 44, role: 'op', title: '合成 x_t（式 6）', titleSize: 11 }));
  const enet = add(node({ id: 'enet', x: 36, y: 216, w: 120, h: 46, role: 'attn', title: 'ε_θ(x_t, t)', sub: '一次前向', mono: true, titleSize: 11.5, subSize: 10 }));
  const loss = add(node({ id: 'loss', x: 216, y: 216, w: 124, h: 46, role: 'attn', title: '‖ε − ε_θ‖²', sub: '一步梯度下降', mono: true, titleSize: 11.5, subSize: 10 }));

  body.push(edge(sx0, [155, 148], { nodes, portFrom: sx0.bottom, pulse: true, dur: 1.6, uid: UID, eid: 't0' }));
  body.push(edge(st, [184, 148], { nodes, portFrom: st.bottom, pulse: true, dur: 1.6, delay: 0.15, uid: UID, eid: 't1' }));
  body.push(edge(seps, [213, 148], { nodes, portFrom: seps.bottom, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 't2' }));
  body.push(edge(comp, enet, { nodes, portFrom: comp.bottom, portTo: enet.top, pulse: true, dur: 1.6, delay: 0.5, uid: UID, eid: 't3' }));
  body.push(edge(enet, loss, { nodes, pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 't4' }));
  body.push(edge(seps, [278, 216], { nodes, portFrom: seps.bottom, dash: true, uid: UID, eid: 't5' }));
  // repeat 回路（虚线）：loss → 回到采样源
  const loopT = 'M 278 262 L 278 281 L 80 281 L 80 124';
  body.push(`<path class="dia-edge dash" d="${loopT}"/>`);
  body.push(tri(80, 124, 0, -1, 6));
  body.push(cap(182, 273, 'repeat … until 收敛（式 26 直译）', { mono: true, size: 9.5, anchor: 'middle' }));

  /* ---- 右：采样（串行）---- */
  body.push(zone({ x: 364, y: 44, w: 336, h: 244, label: '采样（ancestral，T 步串行）' }));
  const xT = add(node({ id: 'xT', x: 382, y: 84, w: 100, h: 40, role: 'cache', title: 'x_T ~ N(0, I)', mono: true, titleSize: 10 }));
  const sen = add(node({ id: 'sen', x: 382, y: 158, w: 110, h: 48, role: 'attn', title: 'ε_θ(x_t, t)', sub: '1 次 NFE / 步', mono: true, titleSize: 11, subSize: 10 }));
  const upd = add(node({ id: 'upd', x: 548, y: 158, w: 136, h: 48, role: 'op', title: 'x_{t−1} 更新', sub: '式 (10)(18)', mono: true, titleSize: 11.5, subSize: 10 }));
  const zz = add(node({ id: 'zz', x: 548, y: 232, w: 136, h: 34, role: 'plain', title: 'z ~ N(0,I)（t=1 时 z=0）', mono: true, titleSize: 9.5 }));
  const x0o = add(node({ id: 'x0o', x: 596, y: 84, w: 96, h: 40, role: 'data', title: 'return x_0', mono: true, titleSize: 10.5 }));

  body.push(edge(xT, sen, { nodes, portFrom: xT.bottom, portTo: sen.top, pulse: true, dur: 1.8, uid: UID, eid: 's0' }));
  body.push(edge(sen, upd, { nodes, pulse: true, dur: 1.8, delay: 0.25, uid: UID, eid: 's1' }));
  body.push(edge(zz, upd, { nodes, portFrom: zz.top, portTo: upd.bottom, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 's2' }));
  // 串行回路：x_{t−1} → 下一步的 x_t
  const loopS = 'M 612 158 C 612 122, 462 118, 452 154';
  body.push(`<path class="dia-edge strong" d="${loopS}"/>`);
  body.push(tri(452, 155, -0.18, 0.98, 6));
  body.push(pathPulse(loopS, { dur: 2.2, delay: 0.7, uid: UID, r: 3.6 }));
  body.push(cap(534, 116, 't ← t−1 · 重复 T 次', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(upd, x0o, { nodes, portFrom: [646, 158], portTo: x0o.bottom, pulse: true, dur: 1.8, delay: 1.0, uid: UID, eid: 's3' }));
  body.push(cap(668, 146, 't=1 后', { size: 9, anchor: 'middle' }));

  body.push(cap(24, 316, '训练三行核心 = 抽 t、采样式合成 x_t、MSE 回归 ε —— 简单到不像 1000 层模型的目标函数', { size: 11 }));
  body.push(cap(24, 334, '采样每步一次网络评估：T = 1000 次 NFE —— 7.4 的全部采样器与蒸馏工作都在砍这笔账', { size: 11 }));

  return scene(720, 350, body, { label: 'DDPM 训练循环（并行合成监督）与采样循环（T 步串行）对照' }).svg();
}

/* ============================================================
 * 7.2 §5.3 — ε_θ 的 I/O 契约：U-Net 下-上采样 + skip（block 2）
 * 原文：x_t [B,3,32,32] 与 t [B]（sinusoidal embedding）进 U-Net → ε_θ(x_t,t) 同 shape；
 *       down 32²→16²→8²→4²（ResBlock×N，每块注入 t）；mid ResBlock+self-attn；up 4²→…→32²（skip 拼接）。
 * ============================================================ */
function unetContract() {
  const UID = 'p7unet';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  const xt = add(node({ id: 'xt', x: 24, y: 56, w: 108, h: 46, role: 'data', title: 'x_t', sub: '[B,3,32,32]', mono: true, subSize: 10 }));
  const tt = add(node({ id: 'tt', x: 24, y: 220, w: 108, h: 36, role: 'plain', title: 't  [B]', mono: true, titleSize: 12 }));
  const emb = add(node({ id: 'emb', x: 24, y: 272, w: 108, h: 42, role: 'norm', title: 'sinusoidal', sub: 'embedding', mono: true, titleSize: 11, subSize: 10 }));

  body.push(zone({ x: 156, y: 16, w: 436, h: 300, label: 'U-Net（输入输出同 shape 契约）' }));
  // down 列（左）→ mid（底）→ up（右）
  const d32 = add(node({ id: 'd32', x: 190, y: 56, w: 84, h: 34, role: 'proj', title: '32²', mono: true, titleSize: 12 }));
  const d16 = add(node({ id: 'd16', x: 190, y: 120, w: 84, h: 34, role: 'proj', title: '16²', mono: true, titleSize: 12 }));
  const d8 = add(node({ id: 'd8', x: 190, y: 184, w: 84, h: 34, role: 'proj', title: '8²', mono: true, titleSize: 12 }));
  const mid = add(node({ id: 'mid', x: 330, y: 236, w: 140, h: 46, role: 'attn', title: 'mid：4²', sub: 'ResBlock + self-attn', mono: true, titleSize: 12, subSize: 9.5 }));
  const u8 = add(node({ id: 'u8', x: 500, y: 184, w: 84, h: 34, role: 'proj', title: '8²', mono: true, titleSize: 12 }));
  const u16 = add(node({ id: 'u16', x: 500, y: 120, w: 84, h: 34, role: 'proj', title: '16²', mono: true, titleSize: 12 }));
  const u32 = add(node({ id: 'u32', x: 500, y: 56, w: 84, h: 34, role: 'proj', title: '32²', mono: true, titleSize: 12 }));

  body.push(edge(xt, d32, { nodes, pulse: true, dur: 1.6, uid: UID, eid: 'i0' }));
  body.push(edge(d32, d16, { nodes, portFrom: d32.bottom, portTo: d16.top, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'd1' }));
  body.push(edge(d16, d8, { nodes, portFrom: d16.bottom, portTo: d8.top, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'd2' }));
  body.push(edge(d8, mid, { nodes, portFrom: d8.bottom, portTo: [360, 236], pulse: true, dur: 1.6, delay: 0.6, uid: UID, eid: 'd3' }));
  body.push(edge(mid, u8, { nodes, portFrom: [440, 236], portTo: u8.bottom, pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 'u0' }));
  body.push(edge(u8, u16, { nodes, portFrom: u8.top, portTo: u16.bottom, pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'u1' }));
  body.push(edge(u16, u32, { nodes, portFrom: u16.top, portTo: u32.bottom, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'u2' }));

  // skip 拼接（水平虚线，细节直达）
  const skips = [
    ['d32', 'u32', 73], ['d16', 'u16', 137], ['d8', 'u8', 201],
  ];
  skips.forEach(([a, b, yy], i) => {
    const d = `M 274 ${yy} L 500 ${yy}`;
    body.push(`<path class="dia-edge dash" d="${d}"/>`);
    body.push(tri(498, yy, 1, 0, 6));
    body.push(pathPulse(d, { dur: 2.2, delay: 0.3 + i * 0.3, uid: UID, r: 3.2 }));
  });
  body.push(cap(387, 166, 'skip 拼接：细节直达', { size: 9.5, anchor: 'middle' }));
  body.push(cap(232, 230, 'down：ResBlock × N', { size: 9.5, anchor: 'middle' }));

  const eo = add(node({ id: 'eo', x: 600, y: 56, w: 108, h: 46, role: 'attn', title: 'ε_θ(x_t, t)', sub: '[B,3,32,32]', mono: true, titleSize: 11, subSize: 10 }));
  body.push(edge(u32, eo, { nodes, pulse: true, dur: 1.6, delay: 1.4, uid: UID, eid: 'o0' }));

  // t 注入：emb → 每个 ResBlock（汇一条虚线进 mid 区，文字说明）
  const tinj = 'M 132 293 C 220 293, 280 268, 330 262';
  body.push(edge(tt, emb, { nodes, portFrom: tt.bottom, portTo: emb.top, uid: UID, eid: 'e0' }));
  body.push(`<path class="dia-edge dash" d="${tinj}"/>`);
  body.push(pathPulse(tinj, { dur: 2.6, uid: UID, r: 3.2 }));
  body.push(cap(170, 312, 't 经 scale-shift 注入每个 ResBlock（7.6 adaLN 是它的推广）', { size: 9.5 }));

  body.push(cap(24, 340, '同 shape 契约 ⇒ 下-上采样 + skip：细节走 skip 直达，语义在 bottleneck 处理（结构详拆连同 DiT 路线 → 7.6）', { size: 11 }));
  body.push(cap(24, 358, '本章只需要 I/O 契约：x_t 与 ε_θ 同为 [B, C, H, W]；条件只有标量 t（文本等条件的注入机制是 7.6 的主题）', { size: 10.5 }));

  return scene(720, 374, body, { label: 'DDPM U-Net 契约：下采样-上采样加 skip 拼接，t 经 sinusoidal embedding 注入每个 ResBlock' }).svg();
}

/* ============================================================
 * 7.3 §4–§6 — 前向 SDE 与两条反向动力学（block 0）
 * 原文：前向 SDE dx = f dt + g dW（固定无参数，t:0→1），密度演化 Fokker-Planck（式 16）；
 *       反向（采样 t:1→0）唯一未知量 s_θ ≈ ∇log p_t：逆时 SDE（式 17，随机）与 PF-ODE（式 20，确定）。
 * ============================================================ */
function sdeOdeMap() {
  const UID = 'p7sde';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  const x0 = add(node({ id: 'x0', x: 30, y: 148, w: 104, h: 52, role: 'data', title: 'x_0', sub: 'p_data', mono: true }));
  const x1 = add(node({ id: 'x1', x: 586, y: 148, w: 104, h: 52, role: 'cache', title: 'x_1', sub: '≈ N(0, I)', mono: true }));

  // 前向 SDE：上方拱（t: 0 → 1 加噪）
  const fwd = 'M 82 148 C 82 108, 634 108, 634 148';
  body.push(`<path class="dia-edge strong" d="${fwd}"/>`);
  body.push(pathPulse(fwd, { dur: 3.6, uid: UID }));
  body.push(cap(360, 66, '前向 SDE（固定，无参数）· 加噪 t: 0 → 1：dx = f(x,t) dt + g(t) dW', { mono: true, size: 11, anchor: 'middle' }));
  body.push(cap(360, 86, '密度演化：Fokker-Planck（式 16）—— VP / VE 只是 (f, g) 的选取（式 13/15）', { size: 10, anchor: 'middle' }));

  // 反向动力学分区：两条方程竖排，x_1 右侧扇入、左出汇回 x_0（无交叉）
  body.push(zone({ x: 150, y: 196, w: 420, h: 148, tone: 'seal' }));
  const sde = add(node({ id: 'sde', x: 330, y: 214, w: 224, h: 46, role: 'attn', title: '逆时 SDE（式 17）· 随机', sub: 'dx=[f−g²s_θ]dt+g dW̄', mono: true, titleSize: 10.5, subSize: 9.5 }));
  const ode = add(node({ id: 'ode', x: 330, y: 282, w: 224, h: 46, role: 'attn', title: 'PF-ODE（式 20）· 确定', sub: 'dx/dt = f − ½g²s_θ', mono: true, titleSize: 10.5, subSize: 9.5 }));

  body.push(edge(x1, sde, { nodes, portFrom: x1.bottom, portTo: sde.right, pulse: true, dur: 2.4, uid: UID, eid: 'r0' }));
  body.push(edge(x1, ode, { nodes, portFrom: [638, 200], portTo: ode.right, pulse: true, dur: 2.4, delay: 0.35, uid: UID, eid: 'r1' }));
  body.push(edge(sde, [100, 200], { nodes, portFrom: sde.left, pulse: true, dur: 2.4, delay: 0.5, uid: UID, eid: 'r2' }));
  body.push(edge(ode, [74, 200], { nodes, portFrom: ode.left, pulse: true, dur: 2.4, delay: 0.85, uid: UID, eid: 'r3' }));

  body.push(cap(166, 234, 'Euler–Maruyama 离散', { size: 9.5 }));
  body.push(cap(166, 248, '≈ DDPM ancestral（7.2）', { size: 9.5 }));
  body.push(cap(166, 300, 'Euler = DDIM σ=0（7.4）', { size: 9.5 }));
  body.push(cap(166, 314, 'Heun / DPM-Solver（7.4）', { size: 9.5 }));
  body.push(cap(166, 338, '反向动力学（采样 t: 1→0）· 唯一未知量 s_θ ≈ ∇log p_t', { mono: true, size: 10 }));

  body.push(cap(24, 366, '整个框架只需要一个网络 s_θ(x, t)，其余全是闭式动力学（§5 合龙）—— §7：同一网络、同一边缘族，', { size: 11 }));
  body.push(cap(24, 384, '全部采样器是这两条方程的不同离散化；PF-ODE 另给精确似然：散度积分（式 23）', { size: 11 }));

  return scene(720, 398, body, { label: 'score-SDE 全景：前向 SDE 加噪，反向走逆时 SDE 或 PF-ODE，唯一未知量是 score' }).svg();
}

/* ============================================================
 * 7.4 §1 — ancestral 采样环：T 步 = T 次 NFE（block 0）
 * 原文：x_T ~ N(0,I)；for t=T..1：ε_θ = 网络前向(x_t,t)（1 次 NFE）；
 *       x̂_0 = (x_t − √(1−ᾱ_t)ε_θ)/√ᾱ_t（Tweedie，7.3 式 10）；z ~ N(0,I)（t=1 时 z=0）；
 *       x_{t-1} = (1/√α_t)(x_t − β_t/√(1−ᾱ_t)ε_θ) + σ_t·z；return x_0。
 * ============================================================ */
function ancestralLoop() {
  const UID = 'p7loop';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  const xT = add(node({ id: 'xT', x: 24, y: 120, w: 100, h: 46, role: 'cache', title: 'x_T ~ N(0, I)', mono: true, titleSize: 10 }));

  body.push(zone({ x: 150, y: 36, w: 470, h: 226, label: 'ancestral 采样环（7.2 §4.4）· 每步 1 次 NFE' }));
  const xt = add(node({ id: 'xt', x: 176, y: 120, w: 100, h: 48, role: 'data', title: 'x_t', sub: '当前噪声图', mono: true, subSize: 10 }));
  const en = add(node({ id: 'en', x: 316, y: 66, w: 140, h: 48, role: 'attn', title: 'ε_θ(x_t, t)', sub: '网络前向 · 1 次 NFE', mono: true, titleSize: 11.5, subSize: 10 }));
  const upd = add(node({ id: 'upd', x: 496, y: 118, w: 104, h: 48, role: 'op', title: 'x_{t−1} 更新', sub: '式 (10)(18)', mono: true, titleSize: 11, subSize: 10 }));
  const zz = add(node({ id: 'zz', x: 490, y: 202, w: 104, h: 30, role: 'plain', title: 'z ~ N(0, I)', mono: true, titleSize: 10.5 }));
  const tw = add(node({ id: 'tw', x: 160, y: 204, w: 196, h: 32, role: 'plain', title: 'x̂_0 = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t', mono: true, titleSize: 9 }));

  // 环：x_t → ε_θ → 更新 → x_{t−1} 回到 x_t
  body.push(edge(xt, en, { nodes, portFrom: xt.top, portTo: en.left, pulse: true, dur: 1.8, uid: UID, eid: 'l0' }));
  body.push(edge(en, upd, { nodes, portFrom: en.right, portTo: upd.top, pulse: true, dur: 1.8, delay: 0.25, uid: UID, eid: 'l1' }));
  const back = 'M 548 166 C 548 200, 226 200, 226 172';
  body.push(`<path class="dia-edge strong" d="${back}"/>`);
  body.push(tri(226, 172, 0, -1, 6));
  body.push(pathPulse(back, { dur: 2.4, delay: 0.55, uid: UID, r: 3.6 }));
  body.push(cap(387, 186, 't ← t−1 · 重复 T 次（串行）', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(zz, upd, { nodes, portFrom: zz.top, portTo: upd.bottom, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'l2' }));

  // Tweedie 读出（虚线，不进环）
  body.push(edge(en, tw, { nodes, portFrom: en.bottom, portTo: [258, 204], dash: true, uid: UID, eid: 'l3' }));
  body.push(cap(160, 252, 'Tweedie 读出（7.3 式 10）· 不进环', { size: 9 }));

  const x0 = add(node({ id: 'x0', x: 624, y: 120, w: 84, h: 46, role: 'data', title: 'return x_0', mono: true, titleSize: 10 }));
  body.push(edge(xT, xt, { nodes, portFrom: xT.right, portTo: xt.left, pulse: true, dur: 1.8, uid: UID, eid: 'l4' }));
  body.push(edge(upd, x0, { nodes, portFrom: upd.right, portTo: x0.left, pulse: true, dur: 1.8, delay: 1.0, uid: UID, eid: 'l5' }));
  body.push(cap(612, 108, 't=1 后出环', { size: 9, anchor: 'middle' }));

  body.push(cap(24, 288, '原生 DDPM = 1000 次串行 NFE：§2 换离散化（DDIM）· §3 换阶数（DPM-Solver）', { size: 11 }));
  body.push(cap(24, 306, '§4 加引导（CFG，每步 2 次 NFE）· §5 换模型（蒸馏族）—— 接口不变：一切只经 ε_θ(x_t, t) 接触模型', { size: 11 }));

  return scene(720, 322, body, { label: 'ancestral 采样环：每步一次网络评估加噪回注，T 步串行，Tweedie 只读出' }).svg();
}

/* ============================================================
 * 7.4 §3.2 — 步数-质量曲线族（block 1）
 * 原文：ancestral 随机在 <100 步误差急升；DDIM η 越小低步数段越好；DPM-Solver K=1/2/3
 *       阶数平移整条曲线（1→2 收益 ≫ 2→3，10–20 NFE 甜区）；蒸馏学生在 1–4 步极左端单独一条。
 * ============================================================ */
function nfeQuality() {
  const UID = 'p7nfe';
  const body = [pulseDefs(UID)];

  // 坐标轴：x = NFE（log10 0..3 → 92..690），y = 质量（上 = 好，36..232）
  body.push(`<line class="dia-edge" x1="92" y1="36" x2="92" y2="232"/>`);
  body.push(tri(92, 32, 0, -1, 7));
  body.push(`<line class="dia-edge" x1="92" y1="232" x2="700" y2="232"/>`);
  body.push(tri(704, 232, 1, 0, 7));
  body.push(cap(56, 46, '质量 ↑', { size: 11, anchor: 'middle' }));
  body.push(cap(700, 268, 'NFE（log 轴）', { mono: true, size: 10, anchor: 'end' }));
  const X = (v) => 92 + (Math.log10(v) / 3) * 598;
  [[1, '1'], [4, '4'], [10, '10'], [50, '50'], [100, '100'], [1000, '1000']].forEach(([v, s]) => {
    body.push(`<line class="dia-edge" x1="${X(v)}" y1="228" x2="${X(v)}" y2="236"/>`);
    body.push(cap(X(v), 250, s, { mono: true, size: 9.5, anchor: 'middle' }));
  });

  // 曲线族（示意形状，非实测描点；右端同网络质量收敛）
  const curves = [
    { d: 'M 291 190 C 360 148, 400 116, 430 102 C 520 82, 600 74, 690 70', cls: 'dia-edge strong', stroke: '#a9a498' },
    { d: 'M 211 186 C 260 160, 280 150, 291 144 C 340 122, 400 104, 690 66', cls: 'dia-edge dash', stroke: '#a9a498' },
    { d: 'M 211 156 C 260 134, 280 128, 291 122 C 360 100, 480 84, 690 62', cls: 'dia-edge', stroke: '#8b867a' },
    { d: 'M 211 138 C 270 118, 291 112, 330 102 C 430 86, 560 72, 690 58', cls: 'dia-edge strong', stroke: null },
    { d: 'M 211 114 C 270 100, 291 96, 330 90 C 430 78, 560 66, 690 54', cls: 'dia-edge strong', stroke: null },
    { d: 'M 211 106 C 270 94, 291 90, 330 86 C 430 74, 560 62, 690 50', cls: 'dia-edge strong', stroke: null },
    { d: 'M 92 78 C 140 82, 180 88, 211 94', cls: 'dia-edge strong', stroke: 'var(--seal)' },
  ];
  curves.forEach((c) => {
    const style = c.stroke ? ` style="stroke:${c.stroke}"` : '';
    body.push(`<path class="${c.cls}" d="${c.d}"${style}/>`);
  });
  // ancestral 在 <10 步继续急坠（虚线延出）
  body.push(`<path class="dia-edge dash" d="M 291 190 C 268 206, 250 220, 236 228" style="stroke:#a9a498"/>`);
  body.push(pathPulse(curves[0].d, { dur: 5.2, uid: UID, r: 3.4 }));
  body.push(pathPulse(curves[4].d, { dur: 4.2, delay: 0.8, uid: UID, r: 3.4 }));
  body.push(pathPulse(curves[6].d, { dur: 2.2, delay: 0.4, uid: UID, r: 3.4 }));

  // ≤4 步分界（曲率是墙）
  body.push(`<line x1="211" y1="40" x2="211" y2="228" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 4" opacity="0.55"/>`);
  body.push(cap(219, 46, '≤4 步：轨迹曲率是墙 → §5 蒸馏 / reflow', { size: 9.5, color: 'var(--seal-deep)' }));

  // 图例（左下空区，线样 + 文字）
  const legend = [
    ['DDPM ancestral（随机）', '#a9a498', false],
    ['DDIM η=1（虚）/ η=0', '#8b867a', true],
    ['DPM-Solver K=1 / 2 / 3', null, false],
    ['蒸馏学生（§5）：1–4 步', 'var(--seal)', false],
  ];
  legend.forEach(([s, c, dash], i) => {
    const y = 158 + i * 18;
    body.push(`<line x1="100" y1="${y - 3}" x2="124" y2="${y - 3}" stroke="${c ?? 'rgb(24 27 32 / 0.45)'}" stroke-width="1.6"${dash ? ' stroke-dasharray="4 3"' : ''}/>`);
    body.push(cap(130, y, s, { mono: true, size: 9.5 }));
  });
  body.push(cap(100, 140, '同一网络：阶数平移整条曲线', { size: 9.5 }));
  body.push(cap(100, 126, '（1→2 收益 ≫ 2→3）', { size: 9 }));

  // ancestral 读法注记
  body.push(cap(450, 116, 'ancestral：步数 <100 误差急升', { size: 9.5 }));
  body.push(cap(450, 130, '随机项在低步数不再被平均掉', { size: 9.5 }));
  body.push(cap(450, 158, '10–20 NFE：solver 甜区', { size: 9.5 }));

  body.push(cap(92, 272, '示意形状，非实测描点；各曲线交点的定量位置见各论文原表（§3.2）', { size: 10 }));

  return scene(720, 286, body, { label: 'NFE-质量曲线族：随机采样低步数崩、DDIM 与 solver 阶数平移、蒸馏在极左端' }).svg();
}

/* ============================================================
 * 7.5 §1 — flow map 与 pushforward（block 0）
 * 原文：p_0 = p_data、p_1 = N(0,I)；每个点沿 dx/dt = u_t(x)（式 1）；轨迹互不相交（ODE 解唯一）；
 *       p_t = [ψ_t]* p_0（式 2 pushforward）；边界条件只钉两端；采样方向 t: 1 → 0。
 * ============================================================ */
function flowMap() {
  const UID = 'p7flow';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  // t 轴
  body.push(`<line class="dia-edge strong" x1="80" y1="44" x2="640" y2="44"/>`);
  body.push(tri(646, 44, 1, 0, 7));
  body.push(cap(80, 32, 't: 0（数据端）', { mono: true, size: 10 }));
  body.push(cap(640, 32, 't: 1（噪声端）', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(360, 60, '前向 = 加噪 t: 0 → 1 · 采样方向相反：t: 1 → 0', { size: 10, anchor: 'middle' }));

  const p0 = add(node({ id: 'p0', x: 28, y: 120, w: 110, h: 54, role: 'data', title: 'p_0 = p_data', sub: 'x_0', mono: true, titleSize: 11, subSize: 10 }));
  const p1 = add(node({ id: 'p1', x: 582, y: 120, w: 110, h: 54, role: 'cache', title: 'p_1 = N(0, I)', sub: 'x_1 = ε', mono: true, titleSize: 11, subSize: 10 }));

  // 三条互不相交的轨迹（层流）
  const flows = [
    'M 138 133 C 300 108, 420 104, 582 128',
    'M 138 147 C 300 143, 420 141, 582 145',
    'M 138 161 C 300 186, 420 190, 582 162',
  ];
  flows.forEach((d, i) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 4.2, delay: i * 0.7, uid: UID, r: 3.4 }));
  });
  // 沿轨迹的样本点 ●（t = 1/3、2/3 处，三次贝塞尔求值）
  const bezPts = [
    [[138, 133], [300, 108], [420, 104], [582, 128]],
    [[138, 147], [300, 143], [420, 141], [582, 145]],
    [[138, 161], [300, 186], [420, 190], [582, 162]],
  ];
  const evalBez = (pts, t) => {
    const u = 1 - t;
    const x = u ** 3 * pts[0][0] + 3 * u * u * t * pts[1][0] + 3 * u * t * t * pts[2][0] + t ** 3 * pts[3][0];
    const y = u ** 3 * pts[0][1] + 3 * u * u * t * pts[1][1] + 3 * u * t * t * pts[2][1] + t ** 3 * pts[3][1];
    return [x, y];
  };
  bezPts.forEach((pts) => {
    [1 / 3, 2 / 3].forEach((t) => {
      const [x, y] = evalBez(pts, t);
      body.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" fill="var(--ink-soft)"/>`);
    });
  });

  // 中间密度截面
  body.push(`<line x1="360" y1="90" x2="360" y2="204" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`);
  body.push(cap(360, 84, 'p_t = [ψ_t]* p_0（式 2 pushforward）', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(360, 218, '密度连续变形：p_0 → p_t → p_1', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 252, '每个点沿 dx/dt = u_t(x) 运动（式 1）· ODE 解唯一 ⇒ 轨迹互不相交 ⇒ 生成是确定性映射（固定 x_1 即固定输出）', { size: 11 }));
  body.push(cap(36, 272, '边界条件只钉两端（p_0、p_1）；u_t 可为任意网络（U-Net / DiT）—— 不需要可逆层（对照 7.1 §5 离散 flow）', { size: 11 }));

  return scene(720, 288, body, { label: 'CNF：速度场把 p_data 连续推前到高斯噪声，轨迹互不相交' }).svg();
}

/* ============================================================
 * 7.5 §6 — 条件轨迹（直线交叉）vs 边际 ODE 轨迹（弯曲绕行）（block 1）
 * 原文：条件轨迹固定端点对：直线、恒速 ε−x₀（式 21）；不同端点对的直线互相交叉，
 *       边际场 u_t = E[ε−x₀ | x_t] 在交叉处取后验平均 —— 学到的 ODE 轨迹仍弯（比 VP 少）。
 * ============================================================ */
function condVsMarginal() {
  const UID = 'p7cv';
  const body = [pulseDefs(UID)];

  /* ---- 左：条件轨迹 ---- */
  body.push(zone({ x: 16, y: 44, w: 336, h: 196, label: '条件轨迹（固定端点对）：直线 · 恒速 ε−x₀（式 21）' }));
  const SL = [[48, 96, 'x₀'], [48, 140, 'x₀′'], [48, 184, 'x₀″']];
  const EL = [[328, 176, 'ε'], [328, 140, 'ε′'], [328, 104, 'ε″']];
  SL.forEach(([x, y, s]) => {
    body.push(`<circle cx="${x}" cy="${y}" r="3" fill="var(--ink)"/>`);
    body.push(cap(x - 8, y + 4, s, { mono: true, size: 10.5, anchor: 'end' }));
  });
  EL.forEach(([x, y, s]) => {
    body.push(`<circle cx="${x}" cy="${y}" r="3" fill="var(--ink)"/>`);
    body.push(cap(x + 8, y + 4, s, { mono: true, size: 10.5 }));
  });
  const condLines = [
    'M 48 96 L 328 176',
    'M 48 140 L 328 140',
    'M 48 184 L 328 104',
  ];
  condLines.forEach((d, i) => {
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 3.2, delay: i * 0.5, uid: UID, r: 3.4 }));
  });
  body.push(`<circle cx="188" cy="140" r="5" fill="none" stroke="var(--seal)" stroke-width="1.3"/>`);
  body.push(cap(188, 124, '交叉处：后验混合', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(48, 226, 't: 0 → 1（训练回归的目标 = 各线的恒定速度）', { mono: true, size: 9.5 }));

  /* ---- 右：边际 ODE 轨迹 ---- */
  body.push(zone({ x: 368, y: 44, w: 336, h: 196, label: '边际 ODE 轨迹：u_t = E[ε−x₀ | x_t]（弯曲）' }));
  const SR = [[400, 96, 'x₀'], [400, 140, 'x₀′'], [400, 184, 'x₀″']];
  const ER = [[680, 176, 'ε'], [680, 140, 'ε′'], [680, 104, 'ε″']];
  SR.forEach(([x, y, s]) => {
    body.push(`<circle cx="${x}" cy="${y}" r="3" fill="var(--ink)"/>`);
    body.push(cap(x - 8, y + 4, s, { mono: true, size: 10.5, anchor: 'end' }));
  });
  ER.forEach(([x, y, s]) => {
    body.push(`<circle cx="${x}" cy="${y}" r="3" fill="var(--ink)"/>`);
    body.push(cap(x + 8, y + 4, s, { mono: true, size: 10.5 }));
  });
  const margPaths = [
    'M 400 96 C 500 88, 580 96, 680 104',
    'M 400 140 C 460 140, 520 178, 680 140',
    'M 400 184 C 500 192, 580 184, 680 176',
  ];
  margPaths.forEach((d, i) => {
    body.push(`<path class="dia-edge strong" d="${d}" style="stroke: rgb(168 50 40 / 0.5)"/>`);
    body.push(pathPulse(d, { dur: 3.2, delay: i * 0.5 + 0.25, uid: UID, r: 3.4 }));
  });
  // 条件线交叉处（幽灵标记）：边际轨迹弯曲绕行、互不相交
  body.push(`<circle cx="540" cy="140" r="9" fill="none" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="2 3" opacity="0.7"/>`);
  body.push(cap(540, 112, '绕开交叉区', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(400, 226, 't: 0 → 1（学到的 v_θ 生成的轨迹，采样反向 1 → 0）', { mono: true, size: 9.5 }));

  body.push(cap(24, 268, '直线交叉 ⇒ 边际场在交叉处取后验平均 ⇒ 边际 ODE 轨迹仍弯，只是比 VP 路径弯得少（§10 的账）', { size: 11 }));
  body.push(cap(24, 288, '交叉越多（数据多峰、维度高）边际越弯；reflow：用当前模型自产 (x₀, ε̂) 端点对再训一轮，', { size: 11 }));
  body.push(cap(24, 306, '新端点对的连线天然贴近已有轨迹、交叉减少 —— 逐轮拉直（7.4 §5.4）', { size: 11 }));

  return scene(720, 322, body, { label: '条件轨迹是交叉直线，边际 ODE 轨迹因后验平均而弯曲绕行' }).svg();
}

/* ============================================================
 * 7.6 §1 — 像素 ↔ latent：扩散的两阶段分工（block 0）
 * 原文：z₀ = s·E(x) [4,64,64]，前向加噪闭式直达任意 t（z_t = √ᾱ_t z₀ + √(1−ᾱ_t)ε），
 *       采样在 latent 上走 K 步，ẑ₀ 经 D(ẑ₀/s) 一次 decoder 前向回像素；s = 0.18215（§2）。
 * ============================================================ */
function latentPipeline() {
  const UID = 'p7lat';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(zone({ x: 16, y: 60, w: 144, h: 168, label: '像素空间' }));
  const x = add(node({ id: 'x', x: 28, y: 100, w: 120, h: 44, role: 'data', title: 'x', sub: '[3, 512, 512]', mono: true, subSize: 10 }));
  const xh = add(node({ id: 'xh', x: 28, y: 168, w: 120, h: 44, role: 'data', title: 'x̂', sub: '[3, 512, 512]', mono: true, subSize: 10 }));

  body.push(zone({ x: 244, y: 60, w: 392, h: 168, tone: 'seal', label: 'latent 空间（扩散的工作空间）' }));
  const z0 = add(node({ id: 'z0', x: 258, y: 100, w: 110, h: 48, role: 'proj', title: 'z₀ = s·E(x)', sub: '[4, 64, 64]', mono: true, titleSize: 11, subSize: 10 }));
  const zt = add(node({ id: 'zt', x: 402, y: 100, w: 104, h: 48, role: 'cache', title: 'z_t', sub: '加噪 · 任意 t', mono: true, subSize: 10 }));
  const dn = add(node({ id: 'dn', x: 532, y: 100, w: 104, h: 48, role: 'attn', title: '去噪网络', sub: 'ε_θ / v_θ', mono: true, titleSize: 12, subSize: 10 }));
  const zh0 = add(node({ id: 'zh0', x: 532, y: 168, w: 104, h: 44, role: 'proj', title: 'ẑ₀', sub: '去噪完成', mono: true, subSize: 10 }));

  body.push(edge(x, z0, { nodes, pulse: true, dur: 2.0, uid: UID, eid: 'e' }));
  body.push(cap(203, 108, 'E（6.3）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(203, 140, '感知压缩 · 一次前向', { size: 9, anchor: 'middle' }));
  body.push(edge(z0, zt, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 'f' }));
  body.push(edge(zt, dn, { nodes, pulse: true, dur: 1.8, delay: 0.55, uid: UID, eid: 'g' }));
  // 采样自环：K 步串行
  const selfLoop = 'M 548 100 C 556 74, 612 74, 620 100';
  body.push(`<path class="dia-edge strong" d="${selfLoop}"/>`);
  body.push(tri(620, 100, 0.35, 0.94, 6));
  body.push(pathPulse(selfLoop, { dur: 2.0, delay: 0.8, uid: UID, r: 3.4 }));
  body.push(cap(584, 70, '采样 K 步（串行，7.4/7.5）', { size: 9.5, anchor: 'middle' }));
  body.push(edge(dn, zh0, { nodes, portFrom: dn.bottom, portTo: zh0.top, pulse: true, dur: 1.8, delay: 1.1, uid: UID, eid: 'h' }));
  // decoder 回程
  const dec = 'M 532 190 C 420 190, 260 190, 152 190';
  body.push(`<path class="dia-edge" d="${dec}"/>`);
  body.push(tri(152, 190, -1, 0, 6));
  body.push(pathPulse(dec, { dur: 2.8, delay: 1.4, uid: UID, r: 3.6 }));
  body.push(cap(340, 208, 'D(ẑ₀ / s)：一次 decoder 前向（仅采样结束跑一次）', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 252, 'z_t = √ᾱ_t · z₀ + √(1−ᾱ_t) · ε —— 前向加噪闭式直达任意 t（7.2 式 5/6 的 latent 版）；s = 0.18215（§2 单位方差化）', { size: 10.5 }));
  body.push(cap(36, 272, '位点 512² = 262144 → 64² = 4096（f8：64× 削减）· L_simple / L_RF 把 x₀ 换成 z₀ 原样成立（latent 化对扩散数学透明）', { size: 10.5 }));
  body.push(cap(36, 290, 'autoencoder 训练一次、被任意多轮扩散实验复用（摊销）—— 感知压缩归 autoencoder，语义压缩归 latent 上的扩散', { size: 10.5 }));

  return scene(720, 304, body, { label: 'LDM 两阶段：autoencoder 感知压缩，扩散在 latent 空间加噪去噪' }).svg();
}

/* ============================================================
 * 7.6 §7.1 — MMDiT 一个 block 的双流数据账（block 1）
 * 原文：Z ∈ R[N,d]（图像 latent token）、Y ∈ R[M,d]（文本 token）各过 adaLN(c) 与各自 QKV 权重；
 *       序列维拼接 → S = QKᵀ/√d_h 联合 softmax → 拆回 O_z/O_y → 各自 W_out 残差 + MLP；
 *       c = pooled 文向 + emb(t)；出口只有 Z，Y 只被读、不出图。
 * ============================================================ */
function mmditFlow() {
  const UID = 'p7mm';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  const c = add(node({ id: 'c', x: 270, y: 14, w: 180, h: 34, role: 'cache', title: 'c = pooled 文向 + emb(t)', mono: true, titleSize: 10 }));
  const Z = add(node({ id: 'Z', x: 80, y: 64, w: 170, h: 44, role: 'data', title: 'Z ∈ R[N,d]', sub: '图像 latent token', mono: true, subSize: 10 }));
  const Y = add(node({ id: 'Y', x: 470, y: 64, w: 170, h: 44, role: 'proj', title: 'Y ∈ R[M,d]', sub: '文本 token', mono: true, subSize: 10 }));
  const az = add(node({ id: 'az', x: 80, y: 128, w: 170, h: 38, role: 'norm', title: 'adaLN_z(c)', mono: true, titleSize: 11.5 }));
  const ay = add(node({ id: 'ay', x: 470, y: 128, w: 170, h: 38, role: 'norm', title: 'adaLN_y(c)', mono: true, titleSize: 11.5 }));
  const qz = add(node({ id: 'qz', x: 80, y: 186, w: 170, h: 40, role: 'proj', title: 'QKV^z = Z·W^z', sub: '图像流自持权重', mono: true, titleSize: 11, subSize: 10 }));
  const qy = add(node({ id: 'qy', x: 470, y: 186, w: 170, h: 40, role: 'proj', title: 'QKV^y = Y·W^y', sub: '文本流自持权重', mono: true, titleSize: 11, subSize: 10 }));
  const cat = add(node({ id: 'cat', x: 300, y: 206, w: 120, h: 44, role: 'op', title: '拼接 [N+M]', sub: '序列维', mono: true, titleSize: 11, subSize: 10 }));
  const joint = add(node({ id: 'joint', x: 280, y: 276, w: 160, h: 52, role: 'attn', title: 'S = QKᵀ/√d_h', sub: 'softmax 跨 N+M 个 key', mono: true, titleSize: 11.5, subSize: 9.5 }));
  const zUpd = add(node({ id: 'zUpd', x: 70, y: 352, w: 190, h: 44, role: 'data', title: 'Z += α_z⊙O_z·W_O^z ；+ MLP_z', mono: true, titleSize: 9.5 }));
  const yUpd = add(node({ id: 'yUpd', x: 460, y: 352, w: 190, h: 44, role: 'proj', title: 'Y += α_y⊙O_y·W_O^y ；+ MLP_y', mono: true, titleSize: 9.5 }));

  // 条件广播（虚线）
  body.push(`<path class="dia-edge dash" d="M 330 48 C 330 96, 240 88, 240 126"/>`);
  body.push(`<path class="dia-edge dash" d="M 390 48 C 390 96, 480 88, 480 126"/>`);
  // 主流
  body.push(edge(Z, az, { nodes, portFrom: Z.bottom, portTo: az.top, pulse: true, dur: 1.5, uid: UID, eid: 'z0' }));
  body.push(edge(az, qz, { nodes, portFrom: az.bottom, portTo: qz.top, pulse: true, dur: 1.5, delay: 0.15, uid: UID, eid: 'z1' }));
  body.push(edge(Y, ay, { nodes, portFrom: Y.bottom, portTo: ay.top, pulse: true, dur: 1.5, uid: UID, eid: 'y0' }));
  body.push(edge(ay, qy, { nodes, portFrom: ay.bottom, portTo: qy.top, pulse: true, dur: 1.5, delay: 0.15, uid: UID, eid: 'y1' }));
  body.push(edge(qz, cat, { nodes, portFrom: qz.right, portTo: cat.left, pulse: true, dur: 1.8, delay: 0.35, uid: UID, eid: 'c0' }));
  body.push(edge(qy, cat, { nodes, portFrom: qy.left, portTo: cat.right, pulse: true, dur: 1.5, delay: 0.35, uid: UID, eid: 'c1' }));
  body.push(edge(cat, joint, { nodes, portFrom: cat.bottom, portTo: joint.top, pulse: true, dur: 1.6, delay: 0.55, uid: UID, eid: 'j0' }));
  body.push(edge(joint, zUpd, { nodes, portFrom: [310, 328], portTo: zUpd.top, pulse: true, dur: 1.8, delay: 0.75, uid: UID, eid: 'o0' }));
  body.push(edge(joint, yUpd, { nodes, portFrom: [400, 328], portTo: yUpd.top, pulse: true, dur: 1.8, delay: 0.75, uid: UID, eid: 'o1' }));
  body.push(cap(218, 340, 'O_z = O[:N]', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(494, 340, 'O_y = O[N:]', { mono: true, size: 9, anchor: 'middle' }));

  // 联合矩阵 2×2 分块小图（S_yz = 文读图，cross-attn 没有的对称块）
  const MX = 478, MY = 262, CS = 26, CG = 3;
  const cells = [
    [0, 0, 'zz', false], [1, 0, 'zy', true], [0, 1, 'yz', true], [1, 1, 'yy', false],
  ];
  cells.forEach(([j, i, s, hot]) => {
    const cx = MX + j * (CS + CG), cy = MY + i * (CS + CG);
    body.push(
      `<rect class="${hot ? 'dia-mx-hot' : 'dia-mx-on'}"${hot && s === 'yz' ? ' style="--i:0" class="dia-mx-hot sig-diag"' : ''} x="${cx}" y="${cy}" width="${CS}" height="${CS}" rx="5"/>` +
      cap(cx + CS / 2, cy + CS / 2 + 3.5, s, { mono: true, size: 9, anchor: 'middle', color: hot ? 'var(--seal-deep)' : 'var(--paper)' })
    );
  });
  body.push(cap(MX, MY - 8, '式 11 分块', { mono: true, size: 9 }));
  body.push(cap(MX, MY + 2 * (CS + CG) + 12, 'S_yz：文读图（双向）', { size: 9, color: 'var(--seal-deep)' }));

  body.push(cap(24, 416, '双向：S_yz 是 cross-attn 没有的对称块（与 §3 (a) 的本质差异）· 出口只有 Z（末端 adaLN + 线性层 → v_θ），', { size: 11 }));
  body.push(cap(24, 434, 'Y 只被读、不出图 —— 文本流逐层被图像去噪状态修正，是手段不是产出；每个 block × L 层', { size: 11 }));

  return scene(720, 448, body, { label: 'MMDiT 双流数据账：各自 QKV、拼接序列联合 attention、拆回各自更新' }).svg();
}

/* ============================================================
 * 7.7 §1 — SD3 整机数据流（block 0）
 * 原文：prompt → CLIP-L + OpenCLIP-G（sequence 特征维拼接；pooled ⊕ emb(t) = adaLN 条件 c）
 *       + T5-XXL（sequence 序列维拼进 Y）→ 条件序列 Y；z_1 = ε → patchify → Z → MMDiT × L
 *       → v_θ(Z, t, c)；每步 ṽ = v(∅) + w·(v(c) − v(∅))（CFG），z ← z − h·ṽ；
 *       z_0 → ÷ scaling factor → VAE decoder → 像素图。
 * ============================================================ */
function sd3Pipeline() {
  const UID = 'p7sd3';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  /* ---- 左：文本条件（三 encoder）---- */
  body.push(zone({ x: 16, y: 16, w: 246, h: 352, label: '文本条件（三 encoder，7.6 §4）' }));
  const prompt = add(node({ id: 'prompt', x: 32, y: 52, w: 90, h: 34, role: 'plain', title: 'prompt', mono: true, titleSize: 11 }));
  const cl = add(node({ id: 'cl', x: 60, y: 116, w: 160, h: 34, role: 'proj', title: 'CLIP-L', sub: '123M', mono: true, titleSize: 11.5, subSize: 10 }));
  const og = add(node({ id: 'og', x: 60, y: 158, w: 160, h: 34, role: 'proj', title: 'OpenCLIP-G', sub: '695M', mono: true, titleSize: 11.5, subSize: 10 }));
  const t5 = add(node({ id: 't5', x: 60, y: 200, w: 160, h: 34, role: 'proj', title: 'T5-XXL', sub: '4.7B', mono: true, titleSize: 11.5, subSize: 10 }));
  const Y = add(node({ id: 'Y', x: 36, y: 256, w: 196, h: 44, role: 'proj', title: '条件序列 Y（文本流）', sub: 'CLIP 特征维拼 + T5 序列维拼', titleSize: 11, subSize: 9.5 }));
  const c = add(node({ id: 'c', x: 36, y: 312, w: 196, h: 34, role: 'cache', title: 'c = pooled ⊕ emb(t)', mono: true, titleSize: 9.5 }));

  // prompt 分发总线（右侧绕行，避免穿过 encoder 卡）
  body.push(`<path class="dia-edge" d="M 122 69 L 238 69"/>`);
  body.push(pathPulse('M 122 69 L 238 69', { dur: 1.4, uid: UID, r: 3.4 }));
  const bus = [
    ['M 238 69 C 238 110, 230 133, 222 133', 0.1],
    ['M 238 69 C 242 140, 232 175, 222 175', 0.25],
    ['M 238 69 C 246 170, 236 217, 222 217', 0.4],
  ];
  bus.forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.6, delay: dl, uid: UID, r: 3.4 }));
  });
  // encoder → Y：左口出、贴左缘下汇入 Y.top（不穿卡）
  const fan = [
    ['M 60 133 C 36 190, 56 246, 100 256', 0.5],
    ['M 60 175 C 44 216, 84 250, 134 256', 0.65],
    ['M 60 217 C 66 240, 110 252, 168 256', 0.8],
  ];
  fan.forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.8, delay: dl, uid: UID, r: 3.4 }));
  });
  // pooled 支路（虚线）：CLIP-L + OpenCLIP-G → c
  body.push(`<path class="dia-edge dash" d="M 70 150 C 30 220, 50 280, 90 310"/>`);
  body.push(`<path class="dia-edge dash" d="M 70 192 C 40 240, 66 290, 116 310"/>`);
  body.push(cap(40, 358, 'pooled：CLIP-L + OpenCLIP-G', { size: 9 }));

  /* ---- 中：采样循环 ---- */
  body.push(zone({ x: 292, y: 16, w: 300, h: 352, tone: 'seal', label: '采样循环 ×K（t: 1 → 0）' }));
  const z1 = add(node({ id: 'z1', x: 316, y: 56, w: 100, h: 40, role: 'cache', title: 'z_1 = ε', sub: '~ N(0, I)', mono: true, titleSize: 11, subSize: 10 }));
  const pat = add(node({ id: 'pat', x: 316, y: 120, w: 100, h: 36, role: 'op', title: 'patchify', mono: true, titleSize: 11.5 }));
  const Z = add(node({ id: 'Z', x: 316, y: 180, w: 100, h: 40, role: 'data', title: 'Z', sub: '图像流 token', mono: true, subSize: 9.5 }));
  const mm = add(node({ id: 'mm', x: 316, y: 252, w: 140, h: 84, role: 'attn', title: 'MMDiT × L', sub: 'Y 被读也被更新', titleSize: 12.5, subSize: 10 }));
  const v = add(node({ id: 'v', x: 480, y: 272, w: 100, h: 40, role: 'proj', title: 'v_θ(Z, t, c)', mono: true, titleSize: 10 }));
  const upd = add(node({ id: 'upd', x: 480, y: 180, w: 100, h: 40, role: 'op', title: 'z ← z − h·ṽ', mono: true, titleSize: 10.5 }));

  body.push(edge(z1, pat, { nodes, portFrom: z1.bottom, portTo: pat.top, pulse: true, dur: 1.6, uid: UID, eid: 's0' }));
  body.push(edge(pat, Z, { nodes, portFrom: pat.bottom, portTo: Z.top, pulse: true, dur: 1.6, delay: 0.2, uid: UID, eid: 's1' }));
  body.push(edge(Z, mm, { nodes, portFrom: Z.bottom, portTo: mm.top, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 's2' }));
  body.push(edge(mm, v, { nodes, pulse: true, dur: 1.6, delay: 0.6, uid: UID, eid: 's3' }));
  body.push(edge(v, upd, { nodes, portFrom: v.top, portTo: upd.bottom, pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 's4' }));
  body.push(edge(upd, Z, { nodes, portFrom: upd.left, portTo: Z.right, pulse: true, dur: 1.6, delay: 1.0, uid: UID, eid: 's5' }));
  body.push(cap(530, 168, 'K 步串行', { size: 9.5, anchor: 'middle' }));
  body.push(edge(Y, mm, { nodes, portFrom: Y.right, portTo: mm.left, pulse: true, dur: 2.2, delay: 0.5, uid: UID, eid: 'c0' }));
  body.push(edge(c, mm, { nodes, portFrom: c.right, portTo: [316, 318], pulse: true, dur: 2.2, delay: 0.65, uid: UID, eid: 'c1' }));

  /* ---- 右：解码列（循环收敛后走一次）---- */
  const z0 = add(node({ id: 'z0', x: 616, y: 56, w: 96, h: 38, role: 'data', title: 'z_0', mono: true }));
  const ds = add(node({ id: 'ds', x: 616, y: 120, w: 96, h: 36, role: 'op', title: '÷ s 还原', titleSize: 11 }));
  const vae = add(node({ id: 'vae', x: 616, y: 180, w: 96, h: 40, role: 'norm', title: 'VAE decoder', mono: true, titleSize: 9.5 }));
  const img = add(node({ id: 'img', x: 616, y: 244, w: 96, h: 40, role: 'data', title: '像素图 x̂', titleSize: 11.5 }));

  const exit = 'M 530 180 C 530 75, 560 75, 612 75';
  body.push(`<path class="dia-edge strong" d="${exit}"/>`);
  body.push(tri(612, 75, 1, 0.02, 6));
  body.push(pathPulse(exit, { dur: 2.6, delay: 1.3, uid: UID, r: 3.6 }));
  body.push(cap(538, 52, '循环收敛（t = 0）', { size: 9.5 }));
  body.push(edge(z0, ds, { nodes, portFrom: z0.bottom, portTo: ds.top, pulse: true, dur: 1.6, delay: 1.5, uid: UID, eid: 'd0' }));
  body.push(edge(ds, vae, { nodes, portFrom: ds.bottom, portTo: vae.top, pulse: true, dur: 1.6, delay: 1.65, uid: UID, eid: 'd1' }));
  body.push(edge(vae, img, { nodes, portFrom: vae.bottom, portTo: img.top, pulse: true, dur: 1.6, delay: 1.8, uid: UID, eid: 'd2' }));
  body.push(cap(656, 302, 'decoder 仅采样结束跑一次', { size: 9, anchor: 'middle' }));

  body.push(cap(24, 386, '每步：ṽ = v(∅) + w·(v(c) − v(∅))（CFG，7.4 §4.2）· 生成目标 = rectified flow（7.5 §6），回归 ε − z₀', { size: 10.5 }));

  return scene(720, 402, body, { label: 'SD3 整机：三文本编码器出条件序列与 c，latent 经 patchify 进 MMDiT 循环采样，VAE 解码出图' }).svg();
}

/* ============================================================
 * 7.7 §5 — VAR：next-scale 尺度链（block 1）
 * 原文：r_1..r_10：1×1 → 2×2 → 3×3 → … → 16×16，共享 codebook V = 4096；
 *       条件前缀 = 全部更粗尺度的完整 map；p(r_k | r_{<k})：r_k 内 h_k·w_k 位点一步并行；
 *       重建 ẑ = Σ_k upsample（查表向量）。256² 档：Σ h_k w_k = 680 token、K = 10。
 * ============================================================ */
function varScales() {
  const UID = 'p7var';
  const body = [pulseDefs(UID)];

  body.push(cap(340, 38, 'p(r_k | r_{<k})：r_k 内 h_k·w_k 个位点一步并行（式 6）· 共享 codebook V = 4096', { mono: true, size: 9.5, anchor: 'middle' }));

  // 尺度卡：r_1 / r_2 / r_3 / ··· / r_10（格点数 = 真实分辨率）
  const CY = 64, CH = 130;
  const cards = [
    { name: 'r_1', x: 36, w: 96, n: 1, cell: 26 },
    { name: 'r_2', x: 166, w: 96, n: 2, cell: 22 },
    { name: 'r_3', x: 292, w: 96, n: 3, cell: 15 },
    { name: '···', x: 414, w: 56, dots: true },
    { name: 'r_10', x: 504, w: 156, n: 16, cell: 6.5 },
  ];
  cards.forEach((cd, ci) => {
    body.push(`<rect x="${cd.x}" y="${CY}" width="${cd.w}" height="${CH}" rx="12" fill="var(--paper-raised)" stroke="rgb(24 27 32 / 0.22)" stroke-width="1.2"/>`);
    const cx = cd.x + cd.w / 2;
    body.push(cap(cx, CY - 10, cd.name, { mono: true, size: 12, anchor: 'middle', color: 'var(--ink)' }));
    if (cd.dots) {
      body.push(cap(cx, CY + CH / 2 + 5, '···', { mono: true, size: 15, anchor: 'middle' }));
      return;
    }
    const gap = cd.n === 16 ? 1.5 : 4;
    const total = cd.n * cd.cell + (cd.n - 1) * gap;
    const ox = cx - total / 2, oy = CY + CH / 2 - total / 2;
    for (let i = 0; i < cd.n; i++) {
      for (let j = 0; j < cd.n; j++) {
        body.push(
          `<rect class="dia-mx-on sig-cell" style="--i:${(ci * 0.5 + (i + j) * 0.02).toFixed(2)}" x="${(ox + j * (cd.cell + gap)).toFixed(1)}" y="${(oy + i * (cd.cell + gap)).toFixed(1)}" width="${cd.cell}" height="${cd.cell}" rx="2"/>`
        );
      }
    }
    body.push(cap(cx, CY + CH + 20, `${cd.n}×${cd.n}`, { mono: true, size: 10, anchor: 'middle' }));
  });

  // 链：r_1 → r_2 → r_3 → ··· → r_10（尺度自回归）
  const chainEdges = [[132, 166], [262, 292], [388, 414], [470, 504]];
  chainEdges.forEach(([a, b], i) => {
    const d = `M ${a} ${CY + CH / 2} L ${b} ${CY + CH / 2}`;
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(tri(b - 1, CY + CH / 2, 1, 0, 6));
    body.push(pathPulse(d, { dur: 1.6, delay: i * 0.4, uid: UID, r: 3.4 }));
  });

  // 条件前缀：全部更粗尺度的完整 map → r_k（虚线拱）
  const pre1 = 'M 84 194 C 200 250, 420 250, 548 198';
  const pre2 = 'M 214 194 C 320 240, 460 240, 564 198';
  [pre1, pre2].forEach((d, i) => {
    body.push(`<path class="dia-edge dash" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 3.2, delay: 0.5 + i * 0.4, uid: UID, r: 3.2 }));
  });
  body.push(tri(548, 198, 0.42, -0.91, 6));
  body.push(tri(564, 198, 0.42, -0.91, 6));
  body.push(cap(330, 272, '条件前缀 = 全部更粗尺度的完整 map（block 因果：尺度间因果、尺度内双向）', { size: 9.5, anchor: 'middle' }));

  body.push(cap(24, 296, '重建：ẑ = Σ_k upsample（查表向量）—— tokenizer 是多尺度 VQ（残差量化同 6.3 §5 RVQ 家族）', { size: 11 }));
  body.push(cap(24, 314, '256² 档：尺度序列 1,2,3,4,5,6,8,10,13,16 · 总 token Σ h_k w_k = 680，串行仅 K = 10 步（raster AR：256 步）', { size: 11 }));

  return scene(720, 330, body, { label: 'VAR：从 1×1 到 16×16 的尺度自回归，每步全位点并行，前缀是全部更粗尺度' }).svg();
}

/* ============================================================
 * 7.7 §7 — MaskGIT：置信度 remask 的迭代并行解码（block 2）
 * 原文：t=0 全 mask；每轮全位点并行预测 → 按置信度保留 n_t 个继续 mask；
 *       n_t = ⌈γ(t/T)·N⌉，γ 为 cos schedule（前期大批兑现、后期小步精修）；t=T 时 n=0。
 * ============================================================ */
function maskgitRounds() {
  const UID = 'p7mgit';
  const body = [pulseDefs(UID)];

  const CELL = 17, GAP = 4, COLS = 7, ROWS = 2;
  const GW = COLS * CELL + (COLS - 1) * GAP; // 143
  const grids = [
    { x: 36, t: 't = 0', n: 'n = N', committed: [] },
    { x: 208, t: 't = 1', n: 'n = ⌈γN⌉', committed: [1, 4, 8, 11] },
    { x: 380, t: 't = 2', n: 'n = ⌈γN⌉', committed: [0, 1, 3, 4, 6, 8, 9, 11, 13] },
    { x: 552, t: 't = T', n: 'n = 0', committed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] },
  ];
  grids.forEach((g, gi) => {
    const cx = g.x + GW / 2;
    body.push(cap(cx, 46, g.t, { mono: true, size: 11, anchor: 'middle', color: 'var(--ink)' }));
    for (let k = 0; k < COLS * ROWS; k++) {
      const i = Math.floor(k / COLS), j = k % COLS;
      const on = g.committed.includes(k);
      const xx = g.x + j * (CELL + GAP), yy = 64 + i * (CELL + GAP);
      if (on) {
        body.push(
          `<rect class="dia-mx-hot sig-cell" style="--i:${(gi * 0.55 + k * 0.04).toFixed(2)}" x="${xx}" y="${yy}" width="${CELL}" height="${CELL}" rx="3"/>`
        );
      } else {
        body.push(`<rect class="dia-mx-on" x="${xx}" y="${yy}" width="${CELL}" height="${CELL}" rx="3"/>`);
      }
    }
    body.push(cap(cx, 140, g.n, { mono: true, size: 9.5, anchor: 'middle' }));
  });
  // 轮间箭头（脉冲 = 一轮解码）
  [[179, 208], [351, 380]].forEach(([a, b], i) => {
    const d = `M ${a} 92 L ${b} 92`;
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(tri(b - 1, 92, 1, 0, 6));
    body.push(pathPulse(d, { dur: 1.6, delay: i * 0.5, uid: UID, r: 3.4 }));
  });
  body.push(cap(537, 96, '…', { mono: true, size: 15, anchor: 'middle' }));
  const dLast = 'M 545 92 L 552 92';
  body.push(`<path class="dia-edge strong" d="${dLast}"/>`);

  // 图例
  body.push(`<rect class="dia-mx-on" x="36" y="160" width="14" height="14" rx="3"/>`);
  body.push(cap(56, 171, '= mask（未兑现）', { size: 10 }));
  body.push(`<rect class="dia-mx-hot" x="196" y="160" width="14" height="14" rx="3"/>`);
  body.push(cap(216, 171, '已兑现（本轮按置信度保留）', { size: 10 }));

  body.push(cap(36, 200, '每轮三步：(a) 被 mask 位点并行预测分布并采样 → (b) 算置信度（可加温度噪声调多样性）→ (c) 置信度最低的一批重新 mask', { size: 10.5 }));
  body.push(cap(36, 220, '留下的 mask 数由 schedule 定：n_t = ⌈γ(t/T)·N⌉，γ = cos —— 前期 γ 降得快、大批兑现；后期放缓、小步精修（式 8）', { size: 10.5 }));
  body.push(cap(36, 238, 'MaskGIT：8 轮一张 256² 图（甜点 8–12 轮，ImageNet FID 6.18）vs raster AR 256 步严格串行 —— 每轮保留「收回重出」的权利', { size: 10.5 }));

  return scene(720, 254, body, { label: 'MaskGIT 迭代并行解码：全 mask 出发，每轮并行预测并按置信度重掩码' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '07-diffusion-image-generation/01-generative-model-map', replacesBlock: 0, title: 'Part 7 章节地图：推导链与收口', svg: chapterMap },
  { slug: '07-diffusion-image-generation/02-ddpm', replacesBlock: 0, title: '前向 q 链 / 反向 p_θ 链：跳步直达', svg: ddpmChain },
  { slug: '07-diffusion-image-generation/02-ddpm', replacesBlock: 1, title: '训练循环 vs 采样循环', svg: ddpmTrainSample },
  { slug: '07-diffusion-image-generation/02-ddpm', replacesBlock: 2, title: 'ε_θ 的 I/O 契约：U-Net 下-上采样 + skip', svg: unetContract },
  { slug: '07-diffusion-image-generation/03-score-sde', replacesBlock: 0, title: '前向 SDE 与两条反向动力学', svg: sdeOdeMap },
  { slug: '07-diffusion-image-generation/04-sampling-and-guidance', replacesBlock: 0, title: 'ancestral 采样环：T 步 = T 次 NFE', svg: ancestralLoop },
  { slug: '07-diffusion-image-generation/04-sampling-and-guidance', replacesBlock: 1, title: '步数-质量曲线：solver 阶数与蒸馏', svg: nfeQuality },
  { slug: '07-diffusion-image-generation/05-flow-matching', replacesBlock: 0, title: 'flow map：密度的 pushforward', svg: flowMap },
  { slug: '07-diffusion-image-generation/05-flow-matching', replacesBlock: 1, title: '条件直线 vs 边际弯曲', svg: condVsMarginal },
  { slug: '07-diffusion-image-generation/06-latent-diffusion-and-dit', replacesBlock: 0, title: '像素 → latent：扩散的两阶段分工', svg: latentPipeline },
  { slug: '07-diffusion-image-generation/06-latent-diffusion-and-dit', replacesBlock: 1, title: 'MMDiT 一个 block 的双流数据账', svg: mmditFlow },
  { slug: '07-diffusion-image-generation/07-t2i-systems-and-ar-image', replacesBlock: 0, title: 'SD3 整机数据流', svg: sd3Pipeline },
  { slug: '07-diffusion-image-generation/07-t2i-systems-and-ar-image', replacesBlock: 1, title: 'VAR：next-scale 的尺度链', svg: varScales },
  { slug: '07-diffusion-image-generation/07-t2i-systems-and-ar-image', replacesBlock: 2, title: 'MaskGIT：置信度 remask 的迭代并行解码', svg: maskgitRounds },
];
