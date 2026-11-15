// Part 9（09-video-generation）场景图：替换各章全部 ASCII 块（每块一张 replacesBlock）
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 part01.mjs。
// 注意：part08-10.mjs 的两张签名图（9.1 压缩账 / 9.2 拓扑账）保留不动；本文件只画 ASCII 原块内容。
// 内容全部来自章节原文（数字、shape、机制措辞）。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid)/pathPulse 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/直线/弧线通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 9.1 §2 — Video VAE 全链（block 0）
 * V（121×720×1280×3，bf16 ≈ 669 MB）→ Encoder 3D 因果卷积级联（空间 (1,2,2)、时间 (2,1,1)
 * 首帧独立 τ=4 合 1、通道 3→16、RMSNorm）→ μ,σ → z = μ+σ⊙ε ×s → z（31×90×160×16，14.3 MB）
 * → DiT 去噪（9.2）/ Decoder 镜像 → V̂
 * ============================================================ */
function videoVaeChain() {
  const UID = 'va';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 26, 'Video VAE 全链：像素 ⇄ latent；时间维全部因果（只向过去看）', { size: 12, color: 'var(--ink-soft)' }));

  const V = add(node({ id: 'V', x: 24, y: 64, w: 126, h: 68, role: 'data', title: 'V ∈ R^{T×H×W×3}', mono: true, titleSize: 12 }));
  body.push(cap(87, 148, 'bf16 ≈ 669 MB', { mono: true, size: 10, anchor: 'middle' }));

  body.push(zone({ x: 182, y: 44, w: 246, h: 120, label: 'Encoder：3D 因果卷积级联' }));
  const enc = add(node({ id: 'enc', x: 200, y: 76, w: 210, h: 44, role: 'op', title: 'causal conv3d 级联', sub: '通道 3 → c = 16 · RMSNorm', titleSize: 13 }));
  body.push(cap(198, 140, '空间 stride (1,2,2)：H→H/8，W→W/8', { mono: true, size: 9.5 }));
  body.push(cap(198, 155, '时间 stride (2,1,1)：首帧独立，τ=4 帧合 1', { mono: true, size: 9.5 }));

  const musig = add(node({ id: 'musig', x: 452, y: 64, w: 120, h: 68, role: 'proj', title: 'μ, σ', sub: 'q(z|V) 对角高斯' }));

  const z = add(node({ id: 'z', x: 452, y: 196, w: 150, h: 64, role: 'cache', title: 'z：31×90×160×16', mono: true, titleSize: 11.5, sub: 'bf16 ≈ 14.3 MB' }));
  body.push(cap(505, 168, 'z = μ + σ⊙ε，×s', { mono: true, size: 10, anchor: 'end' }));

  const dit = add(node({ id: 'dit', x: 632, y: 200, w: 80, h: 56, role: 'attn', title: 'DiT 去噪', titleSize: 12, sub: '（9.2）' }));

  const dec = add(node({ id: 'dec', x: 182, y: 196, w: 246, h: 64, role: 'op', title: 'Decoder：镜像上采样栈', sub: '镜像因果 —— 同样只向过去看', titleSize: 13 }));
  const Vh = add(node({ id: 'Vh', x: 24, y: 196, w: 126, h: 64, role: 'data', title: 'V̂ ∈ R^{T×H×W×3}', mono: true, titleSize: 12 }));

  body.push(edge(V, enc, { nodes, pulse: true, dur: 2.0, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(enc, musig, { nodes, pulse: true, dur: 2.0, delay: 0.3, uid: UID, eid: 1 }));
  body.push(edge(musig, z, { nodes, portFrom: musig.bottom, portTo: z.top, pulse: true, dur: 2.0, delay: 0.6, uid: UID, eid: 2 }));
  body.push(edge(z, dec, { nodes, portFrom: z.left, portTo: dec.right, pulse: true, dur: 2.0, delay: 0.9, uid: UID, eid: 3 }));
  body.push(edge(dec, Vh, { nodes, portFrom: dec.left, portTo: Vh.right, pulse: true, dur: 2.0, delay: 1.15, uid: UID, eid: 4 }));
  body.push(edge(z, dit, { nodes, portFrom: z.right, portTo: dit.left, pulse: true, dur: 1.6, delay: 1.35, uid: UID, eid: 5 }));

  body.push(cap(24, 292, '数值体积 3.32×10⁸ → 7.14×10⁶（≈46.5×，长视频极限 48×，差额即首帧 +1）；bf16 669 MB → 14.3 MB', { mono: true, size: 10.5, color: 'var(--ink-soft)' }));
  body.push(cap(24, 310, '两本账：attention 关心 N = t·h·w（按 τs² 缩），存储带宽关心数值体积（按 3τs²/c 缩）—— c 是兑换率（式 3/4）', { size: 10.5 }));

  return scene(720, 324, body, { label: 'Video VAE：原始视频经 3D 因果卷积 encoder 压成 latent，DiT 在 latent 空间去噪，decoder 镜像重建' }).svg();
}

/* ============================================================
 * 9.1 §2.1 — k_t = 3 时间卷积：对称 vs 因果 padding（block 1）
 * 对称：y[2] ← x1,x2,x3（读到未来）、y[0] ← pad,x0,x1（首帧读未来）
 * 因果（左补 k_t−1=2 帧）：y[2] ← x0,x1,x2（只见过去与现在）、y[0] ← ·,·,x0（首帧只含自身）
 * ============================================================ */
function causalPadding() {
  const UID = 'cp';
  const body = [pulseDefs(UID)];
  body.push(cap(24, 26, 'k_t = 3 的时间卷积（空间维略去）：因果 = 时间维只在左端补 k_t − 1 = 2 帧（式 5）', { size: 12, color: 'var(--ink-soft)' }));

  // 数据帧芯片 / 补零芯片
  const cell = (x, label, w = 38) =>
    `<rect fill="var(--paper)" stroke="var(--rule)" stroke-width="1.1" x="${x}" y="70" width="${w}" height="30" rx="6"/>` +
    `<text class="dia-mono" x="${x + w / 2}" y="89" font-size="11.5" fill="var(--ink)" text-anchor="middle">${label}</text>`;
  const padCell = (x, w = 38) =>
    `<rect class="dia-mx-off" x="${x}" y="70" width="${w}" height="30" rx="6"/>` +
    `<text class="dia-mono" x="${x + w / 2}" y="89" font-size="11.5" fill="var(--ink-faint)" text-anchor="middle">·</text>`;
  const thin = (d) => `<path class="dia-edge" d="${d}"/>`;
  const leak = (d) =>
    `<path d="${d}" fill="none" stroke="var(--seal)" stroke-width="1.4" stroke-dasharray="4 3"/>`;

  /* ---- 左：对称 padding（非因果）---- */
  body.push(zone({ x: 16, y: 44, w: 336, h: 196, label: '对称 padding（非因果）' }));
  body.push(cap(44, 89, '输入', { size: 10, anchor: 'end' }));
  const sx = [48, 91, 134, 177, 220, 263]; // pad + x0..x4（w=38）
  body.push(padCell(sx[0]));
  for (let i = 0; i < 5; i++) body.push(cell(sx[i + 1], `x${i}`));
  const y0s = node({ id: 'y0s', x: 70, y: 170, w: 64, h: 30, role: 'proj', title: 'y[0]', mono: true, titleSize: 12 });
  const y2s = node({ id: 'y2s', x: 190, y: 170, w: 64, h: 30, role: 'proj', title: 'y[2]', mono: true, titleSize: 12 });
  body.push(y0s.svg, y2s.svg);
  // y[0] ← pad, x0, x1（x1 是未来：泄漏）
  body.push(thin('M 67 100 C 67 130, 102 142, 102 168'));
  body.push(thin('M 110 100 C 110 128, 102 144, 102 168'));
  body.push(leak('M 153 100 C 153 126, 102 142, 102 168'));
  body.push(pathPulse('M 153 100 C 153 126, 102 142, 102 168', { dur: 2.0, delay: 0.2, uid: UID, r: 3.4 }));
  // y[2] ← x1, x2, x3（x3 是未来：泄漏）
  body.push(thin('M 153 100 C 153 126, 222 142, 222 168'));
  body.push(thin('M 196 100 C 196 124, 222 140, 222 168'));
  body.push(leak('M 239 100 C 239 126, 222 142, 222 168'));
  body.push(pathPulse('M 239 100 C 239 126, 222 142, 222 168', { dur: 2.0, delay: 0.6, uid: UID, r: 3.4 }));
  body.push(cap(102, 222, '首帧读 x1：未来泄漏 ✗', { size: 10, anchor: 'middle', color: 'var(--seal)' }));
  body.push(cap(222, 222, '读 x3：未来泄漏 ✗', { size: 10, anchor: 'middle', color: 'var(--seal)' }));

  /* ---- 右：因果 padding（左补 2 帧）---- */
  body.push(zone({ x: 376, y: 44, w: 328, h: 196, label: '因果 padding（左补 k_t−1 = 2 帧）' }));
  body.push(cap(392, 89, '输入', { size: 10, anchor: 'end' }));
  const cx = [396, 436, 476, 516, 556, 596, 636]; // pad pad x0..x4（w=36）
  body.push(padCell(cx[0], 36), padCell(cx[1], 36));
  for (let i = 0; i < 5; i++) body.push(cell(cx[i + 2], `x${i}`, 36));
  const y0c = node({ id: 'y0c', x: 440, y: 170, w: 64, h: 30, role: 'proj', title: 'y[0]', mono: true, titleSize: 12 });
  const y2c = node({ id: 'y2c', x: 560, y: 170, w: 64, h: 30, role: 'proj', title: 'y[2]', mono: true, titleSize: 12 });
  body.push(y0c.svg, y2c.svg);
  // y[0] ← ·,·,x0（首帧只含自身）
  body.push(thin('M 414 100 C 414 130, 472 144, 472 168'));
  body.push(thin('M 454 100 C 454 126, 472 140, 472 168'));
  body.push(thin('M 494 100 C 494 124, 472 140, 472 168'));
  body.push(pathPulse('M 494 100 C 494 124, 472 140, 472 168', { dur: 2.0, delay: 0, uid: UID, r: 3.4 }));
  // y[2] ← x0,x1,x2（只见过去与现在）
  body.push(thin('M 494 100 C 494 128, 592 146, 592 168'));
  body.push(thin('M 534 100 C 534 126, 592 142, 592 168'));
  body.push(thin('M 574 100 C 574 122, 592 138, 592 168'));
  body.push(pathPulse('M 534 100 C 534 126, 592 142, 592 168', { dur: 2.0, delay: 0.4, uid: UID, r: 3.4 }));
  body.push(cap(472, 222, '首帧只含自身 ✓', { size: 10, anchor: 'middle', color: 'var(--ink)' }));
  body.push(cap(592, 222, '只见过去与现在 ✓', { size: 10, anchor: 'middle', color: 'var(--ink)' }));

  body.push(cap(24, 268, '因果性只加在时间轴；空间维保持对称 padding —— DiT 对同帧空间双向可见（9.2 full 3D 无空间 mask），无泄漏问题', { size: 10.5 }));
  body.push(cap(24, 286, 'y[0] 只含 x0 ⇒ 孤立图像与视频首帧逐位同一 latent —— 图视频混训与 T2I 权重继承的结构前提（§2.2）', { size: 10.5 }));

  return scene(720, 300, body, { label: 'k_t=3 时间卷积：对称 padding 读到未来帧，因果 padding 左补两帧、只见过去与现在' }).svg();
}

/* ============================================================
 * 9.2 §1 — spacetime patchify：z → X [N×d] → L×DiT → v_θ（block 0）
 * z [t×h×w×c]（Wan-VAE：81 帧 720p → 21×90×160×16）→ 3D patchify（等效 p=1）→ X [N×d]
 * N = t·h·w = 302,400；+ 3D RoPE（不进 token 内容，只旋转 q/k，§8）→ L×DiT block（§3–§6 四种形态）
 * → 逐 token 线性层回归 v_θ（FM 目标，7.5 式 22）→ unpatchify 回 [t×h×w×c]
 * ============================================================ */
function spacetimePatchify() {
  const UID = 'sp';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 26, '从 latent 网格到 token 序列：spacetime patchify（Sora 两段式口径）—— 之后全是序列侧设计', { size: 12, color: 'var(--ink-soft)' }));

  const z = add(node({ id: 'z', x: 20, y: 64, w: 132, h: 64, role: 'data', title: 'z [t×h×w×c]', mono: true, titleSize: 11, sub: '21×90×160×16' }));
  body.push(cap(86, 148, 'Wan-VAE · 81 帧 720p', { size: 10, anchor: 'middle' }));

  const pt = add(node({ id: 'pt', x: 162, y: 64, w: 112, h: 64, role: 'op', title: '3D patchify', titleSize: 13, sub: 'p_t × p × p', mono: false }));
  body.push(cap(218, 166, '等效 p=1；CogVideoX p=2 → N÷4', { mono: true, size: 9.5, anchor: 'middle' }));

  const X = add(node({ id: 'X', x: 306, y: 64, w: 114, h: 64, role: 'proj', title: 'X [N×d]', mono: true, titleSize: 12.5, sub: 'N = t·h·w' }));
  body.push(cap(306, 160, 'N = 302,400 ⇒ N² ≈ 9.1×10¹⁰（§2 的靶子）', { mono: true, size: 10.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 470, y: 44, w: 234, h: 104, label: 'L × DiT block（§3–§6）' }));
  const dit = add(node({ id: 'dit', x: 486, y: 76, w: 202, h: 48, role: 'attn', title: 'DiT block ×L', sub: 'adaLN·双流·cross·MoE' }));

  const v = add(node({ id: 'v', x: 500, y: 184, w: 160, h: 48, role: 'proj', title: '逐 token 线性 → v_θ', titleSize: 12, sub: 'FM 目标（7.5 式 22）' }));
  body.push(cap(580, 252, '→ unpatchify 回 [t×h×w×c]', { mono: true, size: 10, anchor: 'middle' }));

  body.push(cap(566, 178, '⊕ 3D RoPE：只旋转 q/k（§8）', { size: 9.5, anchor: 'end' }));
  body.push(cap(24, 252, 'patchify 投影 ≈ 0.05 TFLOPs —— 意义在接口不在计算', { size: 10 }));

  body.push(edge(z, pt, { nodes, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(pt, X, { nodes, pulse: true, dur: 1.8, delay: 0.35, uid: UID, eid: 1 }));
  body.push(edge(X, dit, { nodes, pulse: true, dur: 1.8, delay: 0.7, uid: UID, eid: 2 }));
  body.push(edge(dit, v, { nodes, portFrom: dit.bottom, portTo: v.top, pulse: true, dur: 1.8, delay: 1.05, uid: UID, eid: 3 }));

  return scene(720, 268, body, { label: 'latent 网格经 3D patchify 成 token 序列，过 L 层 DiT block，逐 token 回归速度场' }).svg();
}

/* ============================================================
 * 9.2 §4 — HunyuanVideo：dual-stream 段 → single-stream 段（block 1）
 * 前段：x_v / x_t 各自 QKV·MLP·adaLN，联合 attention（拼接序列，7.6 式 10–12 同构），文本流逐层被修正
 * 后段：[x_v ; x_t] 拼接共享 QKV/MLP，full attention 单流，出口只取视频流（7.6 §3(c)）
 * 代码默认 mm_double_blocks_depth=20 + mm_single_blocks_depth=40
 * ============================================================ */
function hunyuanStreams() {
  const UID = 'hy';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'HunyuanVideo：前段双流各持参数、联合 attention；后段拼接单流共享参数（README §3.2 + 代码实测 20 + 40）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- dual-stream 段 ---- */
  body.push(zone({ x: 16, y: 40, w: 336, h: 230, label: 'dual-stream 段（前段 ×20）' }));
  const xv = add(node({ id: 'xv', x: 26, y: 66, w: 96, h: 44, role: 'data', title: 'x_v', mono: true, sub: '视频 token' }));
  const xt = add(node({ id: 'xt', x: 26, y: 178, w: 96, h: 44, role: 'data', title: 'x_t', mono: true, sub: '文本 token' }));
  const cardV = add(node({ id: 'cardV', x: 134, y: 62, w: 96, h: 52, role: 'proj', title: 'QKV·MLP_v', mono: true, titleSize: 11, sub: 'adaLN_v' }));
  const cardT = add(node({ id: 'cardT', x: 134, y: 174, w: 96, h: 52, role: 'proj', title: 'QKV·MLP_t', mono: true, titleSize: 11, sub: 'adaLN_t' }));
  const attn = add(node({ id: 'attn', x: 262, y: 104, w: 76, h: 80, role: 'attn', title: '联合 attn', titleSize: 12, sub: '[N+M]' }));
  body.push(cap(32, 242, '联合 attention 于拼接序列（7.6 式 10–12 同构）', { size: 9.5 }));
  body.push(cap(32, 256, '文本流逐层被去噪状态修正（双向修正）', { size: 9.5, color: 'var(--seal-deep)' }));

  /* ---- single-stream 段 ---- */
  body.push(zone({ x: 384, y: 40, w: 328, h: 230, label: 'single-stream 段（后段 ×40）' }));
  const cc = add(node({ id: 'cc', x: 400, y: 108, w: 92, h: 72, role: 'data', title: '[x_v ; x_t]', mono: true, titleSize: 11.5, sub: '拼接单流' }));
  const sh = add(node({ id: 'sh', x: 524, y: 108, w: 120, h: 72, role: 'proj', title: '共享 QKV·MLP', titleSize: 12, sub: '单流 full attn' }));
  body.push(edge(sh, [584, 202], { nodes, portFrom: sh.bottom, uid: UID, eid: 9 }));
  body.push(`<polygon points="584,208 579,200 589,200" fill="var(--ink-soft)"/>`);
  body.push(cap(584, 220, '出口只取视频流（文本不出图）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(400, 242, '文本 token 继续在序列内被读（7.6 §3(c) 形态）', { size: 9.5 }));
  body.push(cap(400, 256, '参数一份：退回 in-context 形态', { size: 9.5 }));

  /* ---- 连线 ---- */
  body.push(edge(xv, cardV, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 0, arrow: false }));
  body.push(edge(xt, cardT, { nodes, pulse: true, dur: 1.6, delay: 0.15, uid: UID, eid: 1, arrow: false }));
  body.push(edge(cardV, attn, { nodes, portTo: [262, 124], pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 2 }));
  body.push(edge(cardT, attn, { nodes, portTo: [262, 164], pulse: true, dur: 1.6, delay: 0.45, uid: UID, eid: 3 }));
  body.push(edge(attn, cc, { nodes, strong: true, pulse: true, dur: 2.2, delay: 0.9, uid: UID, eid: 4 }));
  body.push(edge(cc, sh, { nodes, pulse: true, dur: 1.6, delay: 1.2, uid: UID, eid: 5 }));

  body.push(cap(24, 292, '双流的预算花在前段：文本条件在全局决策（高噪声段）定型前完成对齐；后段共享 = 便宜地读已定型条件', { size: 10.5 }));
  body.push(cap(24, 308, '代码默认 mm_double_blocks_depth = 20，mm_single_blocks_depth = 40（hyvideo/modules/models.py 实测）', { mono: true, size: 10.5 }));

  return scene(720, 322, body, { label: 'HunyuanVideo：前段双流各自参数联合 attention，后段拼接单流共享参数' }).svg();
}

/* ============================================================
 * 9.2 §5 — Wan DiT block ×40：self / cross / FFN（block 2）
 * x [N×5120]（N = t·h·w；时间条件走 adaLN 族调制）
 *  ├─ self-attn（3D full + 3D RoPE，40 头）── residual add
 *  ├─ cross-attn（Q←x；K/V←umT5-XXL，长 512）── residual add
 *  └─ FFN（5120→13824→5120）── residual add     ×40 层
 * 出口：逐 token 线性层回归 v_θ（FM：v* = ε − x₀）
 * 占比（§2.5 单层账）：self 二次项 91.1% / cross 1.7% / FFN 4.2%
 * ============================================================ */
function wanBlock() {
  const UID = 'wb';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };
  const TX = 80;
  const addCirc = (cy) =>
    `<circle cx="${TX}" cy="${cy}" r="9" fill="var(--paper)" stroke="var(--ink-soft)" stroke-width="1.3"/>` +
    `<text class="dia-mono" x="${TX}" y="${cy + 5}" text-anchor="middle" font-size="14" fill="var(--ink)">+</text>`;

  const x = add(node({ id: 'x', x: 24, y: 16, w: 116, h: 44, role: 'data', title: 'x [N × 5120]', mono: true, titleSize: 12, sub: 'N = t·h·w' }));

  // 残差主干
  body.push(`<line class="dia-edge strong" x1="${TX}" y1="60" x2="${TX}" y2="404"/>`);
  body.push(pathPulse(`M ${TX} 60 V 404`, { dur: 5, delay: 0, uid: UID, r: 3.6 }));

  body.push(zone({ x: 150, y: 56, w: 300, h: 276, label: 'DiT block ×40（Wan2.1-14B）' }));

  const self = add(node({ id: 'self', x: 180, y: 84, w: 240, h: 52, role: 'attn', title: 'self-attn', sub: '3D full + 3D RoPE · 40 头' }));
  const cross = add(node({ id: 'cross', x: 180, y: 168, w: 240, h: 52, role: 'attn', title: 'cross-attn', sub: 'Q←x · K/V←umT5（长 512）' }));
  const ffn = add(node({ id: 'ffn', x: 180, y: 252, w: 240, h: 52, role: 'proj', title: 'FFN', sub: '5120 → 13824 → 5120' }));

  // 读边（主干 → 子层）与写回（子层 → ⊕ → 主干）
  const taps = [110, 194, 278];
  const nodes3 = [self, cross, ffn];
  nodes3.forEach((n, i) => {
    body.push(`<circle cx="${TX}" cy="${taps[i]}" r="2.6" fill="var(--ink-soft)"/>`);
    body.push(edge([TX, taps[i]], n, { nodes, portTo: n.left, uid: UID, eid: `t${i}`, arrow: false }));
  });
  const returns = [
    { d: `M 300 136 V 152 H ${TX + 9}`, cy: 152, delay: 0.7 },
    { d: `M 300 220 V 236 H ${TX + 9}`, cy: 236, delay: 1.5 },
    { d: `M 300 304 V 320 H ${TX + 9}`, cy: 320, delay: 2.3 },
  ];
  returns.forEach((r) => {
    body.push(`<path class="dia-edge" d="${r.d}"/>`);
    body.push(pathPulse(r.d, { dur: 2.2, delay: r.delay, uid: UID }));
    body.push(addCirc(r.cy));
  });
  body.push(cap(TX - 14, 154, '残差写回', { size: 9.5, anchor: 'end' }));

  // umT5 文本（冻结只读 K/V）
  const umt5 = add(node({ id: 'umt5', x: 500, y: 164, w: 176, h: 52, role: 'cache', title: 'umT5-XXL 文本序列', titleSize: 12, sub: 'M = 512 · 冻结只读' }));
  body.push(edge(umt5, cross, { nodes, portFrom: umt5.left, portTo: cross.right, dash: true, pulse: true, dur: 2.4, delay: 0.9, uid: UID, eid: 'kv' }));
  body.push(cap(460, 180, 'K/V', { mono: true, size: 9.5, anchor: 'middle' }));

  // 占比（§2.5 单层账）
  body.push(cap(452, 84, '占比（§2.5 单层账）', { size: 9.5 }));
  body.push(cap(452, 112, '≈ 91%', { mono: true, size: 11, color: 'var(--seal-deep)', weight: 600 }));
  body.push(cap(452, 210, '≈ 1.7%', { mono: true, size: 11 }));
  body.push(cap(452, 282, '≈ 4.2%', { mono: true, size: 11 }));
  body.push(cap(696, 240, '时间条件走 adaLN 族调制（7.6 式 8）', { size: 9.5, anchor: 'end' }));

  // 出口：逐 token 线性层回归 v_θ
  body.push(`<path class="dia-edge" d="M ${TX} 404 H 172"/>`);
  body.push(`<polygon points="180,404 171,400 171,408" fill="var(--ink-soft)"/>`);
  body.push(pathPulse(`M ${TX} 332 V 404 H 174`, { dur: 1.6, delay: 2.8, uid: UID }));
  add(node({ id: 'v', x: 180, y: 380, w: 240, h: 48, role: 'proj', title: '逐 token 线性层 → v_θ', titleSize: 12.5, sub: 'FM：v* = ε − x₀（式 22）' }));

  body.push(cap(24, 456, '单层合计 2.06 PFLOPs ×40 层 = 82.3 PFLOPs/步；×50 步 ≈ 4.1 EFLOPs 一段 81 帧视频（§2.5）', { mono: true, size: 10.5, color: 'var(--ink-soft)' }));
  body.push(cap(24, 474, '二次项 91% ・ cross-attn 1.7% ・ FFN 4.2% —— 文本条件近乎免费 ⇒ 文本侧设计不受成本约束（§7）', { size: 10.5 }));

  return scene(720, 486, body, { label: 'Wan DiT block：self-attn、cross-attn（umT5 K/V）、FFN 各经残差写回，×40 层后回归速度场' }).svg();
}

/* ============================================================
 * 9.2 §6.1 — Wan2.2 SNR-split MoE：timestep 级确定性路由（block 3）
 * 采样方向 t: 1 → 0；SNR = (1−t)²/t² 严格单调递减（式 7）
 * expert：high-noise（布局/早期）｜low-noise（细节/后期），t_moe 处切换、全程仅一次（式 8）
 * ============================================================ */
function snrSplitMoe() {
  const UID = 'moe';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'Wan2.2 SNR-split MoE：路由 = 确定性 SNR 阈值（式 8），router 无可学参数、切换全程仅一次', { size: 12, color: 'var(--ink-soft)' }));

  // SNR(t) 曲线（式 7 实采样；x：t=1→0，y：log SNR）
  const px = (t) => 80 + (1 - t) * 560;
  const py = (t) => Math.max(58, Math.min(168, 113 - 2 * Math.log((1 - t) / t) * (110 / 14)));
  const pts = [];
  for (let t = 0.03; t <= 0.9701; t += 0.02) pts.push(`${px(t).toFixed(1)} ${py(t).toFixed(1)}`);
  const curve = 'M' + pts.join(' L');
  body.push(`<path class="dia-edge strong" d="${curve}"/>`);
  body.push(pathPulse(curve, { dur: 4.5, delay: 0.2, uid: UID, r: 3.6 }));
  body.push(cap(90, 70, 'log SNR = −2·logit(t)（式 7）', { mono: true, size: 10 }));
  body.push(cap(64, 160, 'SNR 最小（t=1）', { size: 9.5, color: 'var(--seal)' }));
  body.push(cap(660, 76, 'SNR 最大（t→0）', { size: 9.5, anchor: 'end', color: 'var(--seal)' }));

  // t_moe 阈值线（数值 README 未给，示意位置）
  body.push(`<line x1="260" y1="56" x2="260" y2="196" stroke="var(--seal)" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>`);
  body.push(`<circle cx="260" cy="125" r="4" fill="var(--seal)"/>`);
  body.push(cap(260, 48, 't_moe：SNR(t_moe) = s*（示意）', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 采样方向轴
  body.push(`<line class="dia-edge strong" x1="60" y1="196" x2="660" y2="196"/>`);
  body.push(`<polygon points="666,196 656,192 656,200" fill="var(--ink-soft)"/>`);
  body.push(cap(80, 186, 't = 1 · 纯噪声', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(640, 186, 't = 0 · 数据', { mono: true, size: 10, anchor: 'middle' }));

  // expert 服务段条
  body.push(`<rect class="dia-bar dia-bar-seal" x="70" y="210" width="190" height="16" rx="8"/>`);
  body.push(`<rect class="dia-bar dia-bar-ink" x="260" y="210" width="390" height="16" rx="8"/>`);
  body.push(cap(165, 246, 'high-noise expert · 布局/早期', { size: 10, anchor: 'middle' }));
  body.push(cap(455, 246, 'low-noise expert · 细节/后期', { size: 10, anchor: 'middle' }));

  const hi = add(node({ id: 'hi', x: 90, y: 262, w: 150, h: 44, role: 'op', title: 'high-noise expert', mono: true, titleSize: 11.5, sub: '≈14B · 整体布局' }));
  const lo = add(node({ id: 'lo', x: 380, y: 262, w: 150, h: 44, role: 'op', title: 'low-noise expert', mono: true, titleSize: 11.5, sub: '≈14B · 细节精修' }));
  body.push(edge([165, 226], hi, { nodes, portTo: hi.top, uid: UID, eid: 0, arrow: false }));
  body.push(edge([455, 226], lo, { nodes, portTo: lo.top, uid: UID, eid: 1, arrow: false }));

  body.push(cap(24, 324, '总参 27B（两 expert 全程在显存）· 每步只激活 14B ≈ 52% —— 单步 FLOPs 与一台 14B dense 相同（§6.2）', { size: 10.5 }));
  body.push(cap(24, 340, 't_moe 由 SNR 阈值定义：SNR(t_moe) = SNR_min/2；步数换算 t* = 1/(1+√s*)（式 9）—— 对 schedule 变换不变', { mono: true, size: 10.5 }));

  return scene(720, 354, body, { label: 'Wan2.2：SNR 单调递减，t_moe 阈值处从 high-noise expert 切到 low-noise expert，全程一次' }).svg();
}

/* ============================================================
 * 9.2 §8 — 3D RoPE 频段切分：64 个旋转对的三家分配（block 4）
 * θ_j = base^(−2j/128)；Wan [22,21,21] 近均匀 / HunyuanVideo [8,28,28] 空间加重 /
 * Qwen2.5-VL [16,24,24] 折中（理解侧）；高频端 → 低频端
 * ============================================================ */
function ropeBands() {
  const UID = 'rb';
  const body = [pulseDefs(UID)];
  const X0 = 120, U = 8, H = 26;

  body.push(cap(24, 26, '3D RoPE 频段切分：64 个旋转对（d_h = 128 折半），θ_j = base^(−2j/128)（式 10；三家均代码实测）', { size: 12, color: 'var(--ink-soft)' }));

  const rows = [
    { name: 'Wan2.1 / 2.2', y: 56, segs: [['t', 22, 'seal'], ['h', 21, 'ink'], ['w', 21, 'soft']], note: '近均匀' },
    { name: 'HunyuanVideo', y: 116, segs: [['t', 8, 'seal'], ['h', 28, 'ink'], ['w', 28, 'soft']], note: '空间加重' },
    { name: 'Qwen2.5-VL', y: 176, segs: [['t', 16, 'seal'], ['h', 24, 'ink'], ['w', 24, 'soft']], note: '折中（理解侧）' },
  ];
  for (const r of rows) {
    body.push(cap(112, r.y + 17, r.name, { mono: true, size: 10.5, anchor: 'end', color: 'var(--ink-soft)' }));
    let cx = X0;
    for (const [axis, n, tone] of r.segs) {
      const w = n * U - 3;
      body.push(`<rect class="dia-bar dia-bar-${tone}" x="${cx}" y="${r.y}" width="${w}" height="${H}" rx="4"/>`);
      body.push(`<text class="dia-mono" x="${cx + w / 2}" y="${r.y + 17}" font-size="10.5" fill="${tone === 'soft' ? 'var(--ink)' : 'var(--paper)'}" text-anchor="middle">${axis} · ${n}</text>`);
      cx += n * U;
    }
    body.push(cap(640, r.y + 17, r.note, { size: 10 }));
  }

  // 频率轴：高频端 → 低频端
  body.push(`<line class="dia-edge strong" x1="${X0}" y1="224" x2="632" y2="224"/>`);
  body.push(`<polygon points="640,224 630,220 630,228" fill="var(--ink-soft)"/>`);
  body.push(cap(120, 246, '高频端（j 小 · 短波长）', { size: 10 }));
  body.push(cap(632, 246, '低频端（j 大 · 长波长）', { size: 10, anchor: 'end' }));

  body.push(cap(24, 272, 't 段都压在高频端：t 轴范围 13–33，短波长即可分辨；h/w 到 90–160 需长波长盖大位移 · t 恒定 ⇒ J_t 段退化 = 图视频混训零改造（§8）', { size: 10.5 }));

  return scene(720, 284, body, { label: '64 个旋转对按 t/h/w 三轴切段：Wan 近均匀、HunyuanVideo 空间加重、Qwen2.5-VL 折中' }).svg();
}

/* ============================================================
 * 9.3 §3.2 — CausVid 非对称蒸馏（block 0）
 * 双向教师（50 步 FM，基座 Wan2.1-T2V-1.3B）：① ODE 配对预训（式 4）② 冻结提供 s_data
 * 因果学生 G_ψ（块因果 mask）：全体 chunk 各采 t^j（含 0），带噪历史条件下逐 chunk 一步预测、拼接成 x
 * x 加噪到随机 t → s_ψ（在线 fake score，DSM）；梯度 = (s_ψ − s_data)·∂x/∂ψ（式 6，无 GAN）
 * ============================================================ */
function causvidAsym() {
  const UID = 'cv';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'CausVid 非对称蒸馏：双向教师出 real score，因果学生被拉回数据分布（式 4–6，无 GAN）', { size: 12, color: 'var(--ink-soft)' }));

  const teacher = add(node({ id: 'teacher', x: 60, y: 44, w: 230, h: 64, role: 'attn', title: '双向教师（冻结）', sub: '50 步 FM · Wan2.1-T2V-1.3B' }));
  const sdata = add(node({ id: 'sdata', x: 480, y: 44, w: 190, h: 64, role: 'norm', title: 's_data（real score）', mono: true, titleSize: 12, sub: '冻结教师在 x_t 求值' }));
  const student = add(node({ id: 'student', x: 60, y: 216, w: 210, h: 70, role: 'op', title: '因果学生 G_ψ', sub: '块因果 mask（§4）' }));
  const sfake = add(node({ id: 'sfake', x: 480, y: 216, w: 190, h: 64, role: 'proj', title: 's_ψ（fake score）', mono: true, titleSize: 12, sub: '在线网络 · DSM 并行训练' }));

  // ① ODE 配对回归预训（式 4）：教师 → 学生
  body.push(edge(teacher, student, { nodes, portFrom: teacher.bottom, portTo: student.top, pulse: true, dur: 2.2, delay: 0, uid: UID, eid: 0 }));
  body.push(cap(178, 166, '① ODE 配对回归预训（式 4）', { size: 10 }));
  // ② 冻结，提供 s_data：教师 → s_data
  body.push(edge(teacher, sdata, { nodes, pulse: true, dur: 2.2, delay: 0.3, uid: UID, eid: 1 }));
  body.push(cap(375, 64, '② 冻结，提供 real score', { size: 10, anchor: 'middle' }));
  // 学生输出 x → 加噪到随机 t → s_ψ（前向道）
  body.push(edge(student, sfake, { nodes, portFrom: [270, 232], portTo: [480, 232], pulse: true, dur: 2.2, delay: 0.5, uid: UID, eid: 2 }));
  body.push(cap(375, 220, '学生输出 x（逐 chunk 一步拼接）→ 加噪到随机 t', { size: 10, anchor: 'middle' }));
  // s_data → s_ψ（score 差）
  body.push(edge(sdata, sfake, { nodes, portFrom: sdata.bottom, portTo: sfake.top, pulse: true, dur: 2.2, delay: 0.7, uid: UID, eid: 3 }));
  body.push(cap(586, 166, 'score 差', { size: 10 }));
  // 梯度回传（强边，s_fake → student）
  body.push(edge(sfake, student, { nodes, portFrom: [480, 262], portTo: [270, 262], strong: true, pulse: true, dur: 2.6, delay: 0.9, uid: UID, eid: 4 }));
  body.push(cap(375, 284, '∇ = (s_ψ − s_data)·∂x/∂ψ（式 6 · 无 GAN）', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(24, 326, '反传只穿过每 chunk 的一次前向；4 步复合只在推理（Algorithm 2）', { size: 10.5 }));
  body.push(cap(24, 344, '非对称：教师 score 含全视频统计（含未来帧），学生只见过去 —— 全局一致性蒸进因果权重；不动点 p_ψ = p_data 不变', { size: 10.5 }));

  return scene(720, 360, body, { label: 'CausVid：双向教师冻结出 real score，因果学生逐 chunk 一步预测，DMD 梯度回传（无 GAN）' }).svg();
}

/* ============================================================
 * 9.3 §4.2 — KV cache 流式推理双层循环（block 1）
 * prefill：文本条件编码（一次）＋（可选）初始帧 K/V 写入 cache
 * 外层 chunk k = 1..M：x ← ε；内层 j = 1..K（K=4）：v ← G_ψ(x, s_j; cache)，x ← x − h·v；
 * emit chunk k（t=0），干净 K/V append 进 cache（历史只读，当前 chunk K/V 每内层步重算）
 * ============================================================ */
function streamLoop() {
  const UID = 'kv';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'KV cache 流式推理：双层循环 = decode 步内嵌 K 步退火（2.4 §3 在视频 DiT 上的逐项重放）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- prefill（一次）---- */
  body.push(zone({ x: 16, y: 44, w: 172, h: 152, label: 'prefill（一次）' }));
  const txt = add(node({ id: 'txt', x: 28, y: 72, w: 148, h: 40, role: 'proj', title: '文本条件编码', titleSize: 12.5 }));
  const initf = add(node({ id: 'initf', x: 28, y: 124, w: 148, h: 44, role: 'data', title: '初始帧 latent', titleSize: 12, sub: '（可选）' }));

  /* ---- KV cache ---- */
  body.push(zone({ x: 218, y: 44, w: 228, h: 152, label: 'KV cache（append-only）' }));
  ['C₁', 'C₂', 'C₃'].forEach((t, i) => {
    add(node({ id: `c${i}`, x: 234 + i * 50, y: 84, w: 42, h: 26, role: 'cache', title: t, mono: true, titleSize: 11 }));
  });
  body.push(cap(392, 101, '…', { mono: true, size: 13 }));
  body.push(cap(234, 134, '干净态 K/V：emit 时写一次', { mono: true, size: 9.5 }));
  body.push(cap(234, 150, '之后被全部后续 chunk × K 步复读', { size: 9.5 }));

  /* ---- 外层：chunk 自回归 ---- */
  body.push(zone({ x: 486, y: 28, w: 218, h: 248, label: '外层 chunk k = 1..M（自回归）' }));
  const noise = add(node({ id: 'noise', x: 502, y: 64, w: 170, h: 44, role: 'data', title: 'x ← ε ~ N(0, I)', mono: true, titleSize: 11.5, sub: '当前 chunk · t = 1' }));
  const G = add(node({ id: 'G', x: 502, y: 124, w: 150, h: 52, role: 'op', title: 'G_ψ(x, s_j ; cache)', mono: true, titleSize: 10.5, sub: 'K/V 每步重算' }));
  const emit = add(node({ id: 'emit', x: 502, y: 216, w: 150, h: 44, role: 'cache', title: 'emit chunk k', mono: true, titleSize: 11.5, sub: 't = 0 · 干净 K/V' }));
  // 内层退火环（K = 4）
  body.push(`<path class="dia-edge dash" d="M 654 136 C 690 136, 690 164, 654 164"/>`);
  body.push(pathPulse('M 654 136 C 690 136, 690 164, 654 164', { dur: 1.2, delay: 0.5, uid: UID, r: 3.2 }));
  body.push(cap(676, 190, '内层 ×4：x ← x − h·v', { mono: true, size: 9.5, anchor: 'end' }));

  /* ---- 连线 ---- */
  body.push(edge(txt, [218, 97], { nodes, portFrom: txt.right, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 0, arrow: false }));
  body.push(edge(initf, [218, 130], { nodes, portFrom: initf.right, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 1, arrow: false }));
  body.push(edge(noise, G, { nodes, portFrom: noise.bottom, portTo: G.top, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 2 }));
  body.push(edge(G, emit, { nodes, portFrom: G.bottom, portTo: emit.top, pulse: true, dur: 1.6, delay: 0.9, uid: UID, eid: 3 }));
  // 历史 K/V 只读：cache → G
  body.push(edge([446, 97], G, { nodes, portTo: G.left, dash: true, pulse: true, dur: 2.0, delay: 0.6, uid: UID, eid: 4 }));
  body.push(cap(466, 92, '历史只读', { size: 9.5, anchor: 'middle' }));
  // 干净 K/V append：emit → cache
  body.push(edge(emit, [446, 130], { nodes, portFrom: emit.left, pulse: true, dur: 2.2, delay: 1.2, uid: UID, eid: 5 }));
  body.push(cap(440, 210, '干净 K/V append', { mono: true, size: 9.5, anchor: 'end' }));

  body.push(cap(24, 322, '2.4 §3 逐项映射：prefill ↔ 文本条件＋初始帧；decode 一步 ↔ 外层一个 chunk；唯一增量 = 内嵌 K 步扩散退火', { size: 10.5 }));
  body.push(cap(24, 340, '二次项总账：K·2dN²(1+1/M) vs K_T·4dN² → ≈ 1/25（式 7：步数 12.5× × 因果三角 2×）', { mono: true, size: 10.5 }));

  return scene(720, 356, body, { label: '流式生成：prefill 一次写文本与初始帧 K/V，外层逐 chunk 自回归、内层 K 步退火，干净 K/V append 进 cache' }).svg();
}

/* ============================================================
 * 9.3 §5.2 — Diffusion Forcing：t_1..t_n 配置空间（block 2）
 * 角点一：全序列扩散 [s s s s s s]（共享 t，联合退火；9.2 标准训练只覆盖此截面）
 * 中间带（DF 训练覆盖）：[0 0 0 s s s] 流式配置历史钉 0；[0 0 s' s s s] 滑窗近端留小噪声
 * 角点二：next-token [0 0 0 0 0 s]（逐帧 AR，teacher forcing 扩散版；内层退火塌缩 = LLM 式点估计）
 * ============================================================ */
function dfConfigSpace() {
  const UID = 'df';
  const body = [pulseDefs(UID)];

  body.push(cap(24, 24, 'Diffusion Forcing：每 token 一个噪声水平 t_i ∈ [0,1]（式 8）—— 配置空间 π^⊗n 上的三个代表点', { size: 12, color: 'var(--ink-soft)' }));

  // 配置条带：0 = 干净（虚格）/ s = 加噪（朱砂）/ s' = 小噪声（淡朱砂）
  const CW = 26, GAP = 4;
  const cellAt = (x, y, v) => {
    if (v === 0)
      return `<rect class="dia-mx-off" x="${x}" y="${y}" width="${CW}" height="26" rx="4"/>` +
        cap(x + CW / 2, y + 17, '0', { mono: true, size: 10, anchor: 'middle' });
    if (v === 2)
      return `<rect x="${x}" y="${y}" width="${CW}" height="26" rx="4" fill="var(--seal-wash)" stroke="var(--seal)" stroke-opacity="0.45"/>` +
        cap(x + CW / 2, y + 17, "s'", { mono: true, size: 11, anchor: 'middle', color: 'var(--seal)' });
    return `<rect class="dia-mx-hot" x="${x}" y="${y}" width="${CW}" height="26" rx="4"/>` +
      cap(x + CW / 2, y + 17, 's', { mono: true, size: 11, anchor: 'middle', color: 'var(--seal-deep)' });
  };
  const strip = (x0, y, vals) => vals.map((v, i) => cellAt(x0 + i * (CW + GAP), y, v)).join('');

  // 角点一：全序列扩散
  body.push(cap(40, 64, '角点一：全序列扩散', { size: 10.5, weight: 600, color: 'var(--ink)' }));
  body.push(strip(40, 88, [1, 1, 1, 1, 1, 1]));
  body.push(cap(40, 140, '共享 t、联合退火', { size: 9.5 }));
  body.push(cap(40, 154, '9.2 标准训练只覆盖此截面', { size: 9.5 }));

  // 中间带：DF 训练覆盖（两行）
  body.push(cap(272, 64, '中间带：DF 训练覆盖', { size: 10.5, weight: 600, color: 'var(--ink)' }));
  body.push(strip(272, 88, [0, 0, 0, 1, 1, 1]));
  body.push(strip(272, 124, [0, 0, 2, 1, 1, 1]));
  body.push(cap(272, 168, '上：流式配置，历史钉 0（§4 的输入）', { size: 9.5 }));
  body.push(cap(272, 182, '下：滑窗，近端历史留小噪声 s′>0', { size: 9.5 }));

  // 角点二：next-token
  body.push(cap(504, 64, '角点二：next-token', { size: 10.5, weight: 600, color: 'var(--ink)' }));
  body.push(strip(504, 88, [0, 0, 0, 0, 0, 1]));
  body.push(cap(504, 140, '逐帧 AR（teacher forcing 扩散版）', { size: 9.5 }));
  body.push(cap(504, 154, '内层退火塌缩 → LLM 式点估计', { size: 9.5 }));

  // 组间连接（谱系方向）
  body.push(`<path class="dia-edge dash" d="M 220 101 H 268"/>`);
  body.push(`<path class="dia-edge dash" d="M 452 101 H 500"/>`);
  body.push(pathPulse('M 220 101 H 268', { dur: 1.8, delay: 0.3, uid: UID, r: 3.2 }));
  body.push(pathPulse('M 452 101 H 500', { dur: 1.8, delay: 0.8, uid: UID, r: 3.2 }));

  // 谱系轴
  body.push(`<line class="dia-edge strong" x1="60" y1="210" x2="660" y2="210"/>`);
  body.push(`<polygon points="666,210 656,206 656,214" fill="var(--ink-soft)"/>`);
  body.push(pathPulse('M 60 210 H 660', { dur: 6, delay: 0.5, uid: UID, r: 3.4 }));
  body.push(cap(60, 230, '全序列联合退火（共享 t 截面）', { size: 10 }));
  body.push(cap(360, 230, 'DF 训练分布 π^⊗n 覆盖整个空间（式 9：所有子序列的变分下界）', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(660, 230, '逐帧自回归（历史钉 0）', { size: 10, anchor: 'end' }));

  return scene(720, 246, body, { label: 'Diffusion Forcing 配置空间：全序列扩散与逐帧 AR 是同一训练分布的两个角点，中间带由 DF 覆盖' }).svg();
}

/* ============================================================
 * 9.4 §1.1 — I2V 输入拼接：36 = 16 + 16 + 4（block 0）
 * x_t [t×h×w×16] 噪声 latent；z_c [t×h×w×16] 条件 latent（锚点帧填 s·E(I_c)，其余 0）；
 * m [t×h×w×4] 锚点 mask（帧 0 全 1，首尾帧形态再加帧 t−1）→ 通道维拼接 [x_t; z_c; m] 36
 * → patchify + 线性投影（仅第一层换权重 16→36）→ X [N×d] 进时空 DiT（9.2）
 * ============================================================ */
function i2vConcat() {
  const UID = 'i2v';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'I2V 输入侧：每个去噪步的通道维拼接（Wan2.1-I2V config 实测 in_dim = 36 = 16+16+4）', { size: 12, color: 'var(--ink-soft)' }));

  const xt = add(node({ id: 'xt', x: 30, y: 56, w: 196, h: 56, role: 'data', title: 'x_t [t×h×w×16]', mono: true, titleSize: 11, sub: '噪声 latent' }));
  const zc = add(node({ id: 'zc', x: 30, y: 124, w: 196, h: 56, role: 'cache', title: 'z_c [t×h×w×16]', mono: true, titleSize: 11, sub: '锚点填 s·E(I_c)，余 0' }));
  const m = add(node({ id: 'm', x: 30, y: 192, w: 196, h: 56, role: 'norm', title: 'm [t×h×w×4]', mono: true, titleSize: 11, sub: '帧 0 全 1（+末帧）' }));

  const cc = add(node({ id: 'cc', x: 290, y: 110, w: 170, h: 68, role: 'op', title: '[x_t ; z_c ; m]', mono: true, titleSize: 11.5, sub: '通道维拼接 → 36 ch' }));
  const pt = add(node({ id: 'pt', x: 506, y: 110, w: 180, h: 68, role: 'proj', title: 'patchify + 线性投影', titleSize: 12, sub: '仅第一层换权重 16→36' }));
  const X = add(node({ id: 'X', x: 506, y: 216, w: 180, h: 52, role: 'attn', title: 'X [N×d]', mono: true, titleSize: 12, sub: '进时空 DiT（9.2）' }));

  body.push(edge(xt, cc, { nodes, portFrom: xt.right, portTo: [290, 126], pulse: true, dur: 2.0, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(zc, cc, { nodes, portFrom: zc.right, portTo: [290, 144], pulse: true, dur: 2.0, delay: 0.2, uid: UID, eid: 1 }));
  body.push(edge(m, cc, { nodes, portFrom: m.right, portTo: [290, 162], pulse: true, dur: 2.0, delay: 0.4, uid: UID, eid: 2 }));
  body.push(edge(cc, pt, { nodes, pulse: true, dur: 2.0, delay: 0.6, uid: UID, eid: 3 }));
  body.push(edge(pt, X, { nodes, portFrom: pt.bottom, portTo: X.top, pulse: true, dur: 2.0, delay: 0.8, uid: UID, eid: 4 }));

  body.push(cap(30, 268, 'z_c：锚点帧位填 s·E(I_c)，其余全 0；m：preserved 指示按 VAE 时间 stride 重排为 4 通道（§5.1.1）', { size: 9.5 }));
  body.push(cap(30, 288, '第一层参数增量 20×5120 ≈ 1.0×10⁵ = 14B 的百万分之七；每步投影 +0.06 TFLOPs（单层 2.06 PFLOPs 总账的零头）', { mono: true, size: 10.5 }));
  body.push(cap(30, 306, '新增投影层零初始化：训练起点不扰动基座 —— 「起点即原模型」（§2 的同一纪律）', { size: 10.5 }));

  return scene(720, 320, body, { label: 'I2V：噪声 latent、条件 latent、锚点 mask 通道维拼接成 36 通道，第一层投影换权重后进入时空 DiT' }).svg();
}

/* ============================================================
 * 9.4 §7 — keyframe 链（block 1）
 * I_1 → [段 1: p(V_1|I_1,c_1)] → V_1 → last(Dec(V_1)) = I_2 → [段 2: p(V_2|I_2,c_2)] → V_2 → …
 * 首尾帧接口（§1.4）；或 I_2 独立 T2I：硬切镜头
 * ============================================================ */
function keyframeChain() {
  const UID = 'kf';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'keyframe 链：p(V_1:K | I_1:K, c_1:K) = Π_k p(V_k | I_k, c_k)（式 12）—— 段间只经关键帧接口传一致性', { size: 12, color: 'var(--ink-soft)' }));

  const I1 = add(node({ id: 'I1', x: 24, y: 88, w: 48, h: 56, role: 'data', title: 'I₁', mono: true, titleSize: 13 }));
  const seg1 = add(node({ id: 'seg1', x: 104, y: 88, w: 150, h: 56, role: 'op', title: '段 1 生成', sub: 'p(V₁ | I₁, c₁)' }));
  const V1 = add(node({ id: 'V1', x: 286, y: 88, w: 56, h: 56, role: 'cache', title: 'V₁', mono: true, titleSize: 13 }));
  const I2 = add(node({ id: 'I2', x: 374, y: 88, w: 48, h: 56, role: 'data', title: 'I₂', mono: true, titleSize: 13 }));
  const seg2 = add(node({ id: 'seg2', x: 454, y: 88, w: 150, h: 56, role: 'op', title: '段 2 生成', sub: 'p(V₂ | I₂, c₂)' }));
  const V2 = add(node({ id: 'V2', x: 636, y: 88, w: 56, h: 56, role: 'cache', title: 'V₂', mono: true, titleSize: 13 }));

  body.push(edge(I1, seg1, { nodes, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(seg1, V1, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 1 }));
  body.push(edge(V1, I2, { nodes, pulse: true, dur: 1.8, delay: 0.6, uid: UID, eid: 2 }));
  body.push(cap(352, 82, 'last(Dec(V₁))', { mono: true, size: 9, anchor: 'middle' }));
  body.push(edge(I2, seg2, { nodes, pulse: true, dur: 1.8, delay: 0.9, uid: UID, eid: 3 }));
  body.push(edge(seg2, V2, { nodes, pulse: true, dur: 1.8, delay: 1.2, uid: UID, eid: 4 }));
  body.push(edge(V2, [712, 116], { nodes, dash: true, arrow: false, uid: UID, eid: 5 }));
  body.push(cap(712, 122, '…', { mono: true, size: 14, anchor: 'end' }));

  // 接口注记
  body.push(`<line class="dia-edge" x1="179" y1="144" x2="179" y2="158"/>`);
  body.push(`<line class="dia-edge" x1="398" y1="144" x2="398" y2="158"/>`);
  body.push(cap(179, 174, '首尾帧接口（§1.4）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(398, 174, '或独立 T2I：硬切镜头', { size: 9.5, anchor: 'middle' }));

  body.push(cap(24, 206, '段间无 attention：一致性只经 I_k 接口传递，不在关键帧里的信息有逐段漂移的自由度（与 9.3 exposure bias 同族）', { size: 10 }));
  body.push(cap(24, 224, '成本可加：总成本 = Σ 各段单段账（9.2 §2.5），无跨段二次项；独立关键帧的段可全部并行', { size: 10 }));

  return scene(720, 240, body, { label: 'keyframe 链：每段以关键帧与镜头 prompt 独立生成，上一段末帧经 VAE decode 作下一段首帧' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '09-video-generation/01-video-vae', replacesBlock: 0, title: '像素 ⇄ latent 全链：V → μ,σ → z → V̂', svg: videoVaeChain },
  { slug: '09-video-generation/01-video-vae', replacesBlock: 1, title: 'k_t = 3 时间卷积：对称 padding vs 因果 padding', svg: causalPadding },
  { slug: '09-video-generation/02-spatiotemporal-dit', replacesBlock: 0, title: 'spacetime patchify：z → X [N×d] → L×DiT → v_θ', svg: spacetimePatchify },
  { slug: '09-video-generation/02-spatiotemporal-dit', replacesBlock: 1, title: 'dual-stream 段 → single-stream 段（HunyuanVideo）', svg: hunyuanStreams },
  { slug: '09-video-generation/02-spatiotemporal-dit', replacesBlock: 2, title: 'Wan DiT block ×40：self / cross / FFN', svg: wanBlock },
  { slug: '09-video-generation/02-spatiotemporal-dit', replacesBlock: 3, title: 'SNR-split MoE：t_moe 阈值切换，全程一次', svg: snrSplitMoe },
  { slug: '09-video-generation/02-spatiotemporal-dit', replacesBlock: 4, title: '3D RoPE 频段切分：64 旋转对的三家分配', svg: ropeBands },
  { slug: '09-video-generation/03-training-and-acceleration', replacesBlock: 0, title: 'CausVid 非对称蒸馏：ODE 预训 + DMD 主阶段', svg: causvidAsym },
  { slug: '09-video-generation/03-training-and-acceleration', replacesBlock: 1, title: '流式双层循环：chunk 自回归 × K 步退火', svg: streamLoop },
  { slug: '09-video-generation/03-training-and-acceleration', replacesBlock: 2, title: 'Diffusion Forcing：逐 token 噪声水平配置空间', svg: dfConfigSpace },
  { slug: '09-video-generation/04-control-and-editing', replacesBlock: 0, title: 'I2V 输入拼接：x_t ; z_c ; m = 36 通道', svg: i2vConcat },
  { slug: '09-video-generation/04-control-and-editing', replacesBlock: 1, title: 'keyframe 链：段间一致性只经接口帧', svg: keyframeChain },
];
