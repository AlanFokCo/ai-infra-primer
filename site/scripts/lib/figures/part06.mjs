// Part 6（06-vision-representations）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；数字与措辞全部来自章节原文。
// 注意：kit 的 edge 脉冲引用 url(#<uid>-pg) 渐变，这里每个图自建同名 defs 保证解析。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/弧形通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 6.1 §1 末 — ViT 全章数据流（block 0）
 * 原文：x [B,H,W,C] →(式1 切格) patches [B,N,p²C] →(式2 W·vec+b ≡ conv(k=p,s=p))
 * tokens [B,N,d] →(prepend x_cls + 位置嵌入 P) z_0 →(×L block，全双向无 mask)
 * z_L →(§2 读出) y [B,d] → 任务头；N = HW/p²：224/16→196、336/14→576
 * ============================================================ */
function vitShapeFlow() {
  const UID = 'vf';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, '数据流全景：图像 → patch 序列 → 双向 encoder → 读出（式 1–5）', { size: 12, color: 'var(--ink-soft)' }));

  // patchify 适配层分区（ViT 仅有的两个动作）
  body.push(zone({ x: 30, y: 34, w: 640, h: 218, label: 'patchify 适配层（式 1–4）—— ViT 仅有的两个动作' }));

  // 左：图像格点图标（4×4 代表 14×14 网格，一格热 = 一个 patch）
  body.push(`<rect x="56" y="76" width="88" height="88" rx="4" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1.1"/>`);
  for (let k = 1; k < 4; k++) {
    body.push(`<line x1="${56 + k * 22}" y1="76" x2="${56 + k * 22}" y2="164" stroke="var(--rule)" stroke-width="1"/>`);
    body.push(`<line x1="56" y1="${76 + k * 22}" x2="144" y2="${76 + k * 22}" stroke="var(--rule)" stroke-width="1"/>`);
  }
  body.push(`<rect class="dia-mx-hot" x="78" y="98" width="22" height="22" rx="3"/>`);
  body.push(cap(100, 180, '224 × 224', { mono: true, size: 10, anchor: 'middle' }));

  // 主链：六级 shape
  const defs = [
    { id: 'x', y: 64, title: 'x [B, H, W, C]', role: 'data' },
    { id: 'patches', y: 132, title: 'patches [B, N, p²C]', role: 'data' },
    { id: 'tokens', y: 200, title: 'tokens [B, N, d]', role: 'data' },
    { id: 'z0', y: 276, title: 'z_0 [B, N+1, d]', role: 'data' },
    { id: 'zL', y: 352, title: 'z_L [B, N+1, d]', role: 'data' },
    { id: 'y', y: 428, title: 'y [B, d] → 任务头', role: 'proj' },
  ];
  const ns = defs.map((d) => {
    const n = node({ id: d.id, x: 200, y: d.y, w: 240, h: 40, role: d.role, title: d.title, mono: true, titleSize: 12 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // 图标 → x；热格 ⇢ patches（虚线：每格 p×p×C → 一个 patch）
  body.push(edge([144, 120], ns[0], { nodes, portTo: ns[0].left, pulse: true, dur: 2.2, delay: 0, uid: UID, eid: 'icon', arrow: false }));
  body.push(edge([100, 120], ns[1], { nodes, portTo: ns[1].left, dash: true, uid: UID, eid: 'cell', arrow: false }));

  // 链式竖流（脉冲级联下行）
  for (let i = 0; i < 5; i++) {
    body.push(edge(ns[i], ns[i + 1], { nodes, portFrom: ns[i].bottom, portTo: ns[i + 1].top, pulse: true, dur: 2.0, delay: i * 0.3, uid: UID, eid: `c${i}`, arrow: false }));
  }

  // 边注（右侧，引导短tick）；第 3 条下移避开 patchify 分区底边虚线（y=252）
  const mids = [118, 186, 266, 334, 410];
  mids.forEach((m) => body.push(`<line class="dia-edge" x1="446" y1="${m}" x2="464" y2="${m}"/>`));
  const notes = [
    ['式 (1)：切成 (H/p)×(W/p) 网格', '每格 p×p×C · N = HW/p²（式 3）'],
    ['式 (2)：zⁱʲ = W·vec(xⁱʲ) + b', 'W ∈ R^{d×p²C} ≡ conv(k=p, s=p)（式 4）'],
    ['prepend x_cls（可学槽位）', '+ 位置嵌入 P ∈ R^{(N+1)×d}'],
    ['×L block：z′ = z + MHA(LN(z))', 'z ← z′ + MLP(LN(z′))；全双向、无 mask'],
    ['读出（§2）', '第 0 行 / GAP / attention pooling'],
  ];
  notes.forEach(([a, b], i) => {
    body.push(cap(472, mids[i] - 4, a, { mono: true, size: 10.5 }));
    body.push(cap(472, mids[i] + 9, b, { mono: true, size: 10.5 }));
  });

  body.push(cap(36, 484, 'N = HW/p²：224/16 → 196（+CLS → 197）、336/14 → 576；p 减半 → N×4、attention ×16', { mono: true, size: 10.5 }));

  return scene(720, 498, body, { label: 'ViT 数据流：patchify 把图像切成 token 序列，加 CLS 与位置嵌入后过 L 层双向 encoder，读出为 y' }).svg();
}

/* ============================================================
 * 6.2 §2.1 — CLIP 一次训练步的数据流（block 0）
 * 原文：图像 x_1..x_B → 视觉塔 f → W_v → 单位化 → V [B,d] ┐
 *                                                        ├→ S = VTᵀ/τ [B,B]
 *       文本 y_1..y_B → 文本塔 g → W_t → 单位化 → T [B,d] ┘
 *       对角 S_ii = 正样本（标签）；同行/列其余 B−1 = 负样本
 *       行 softmax+CE（image→text，标签 i）/ 列对称 → L = ½(L_i2t + L_t2i)
 * ============================================================ */
function clipDualTower() {
  const UID = 'cf';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(20, 30, '一次训练步：双塔 → 相似度矩阵 S → 两个方向的 B 类分类（式 1–5）', { size: 12, color: 'var(--ink-soft)' }));

  // 双塔分区
  body.push(zone({ x: 10, y: 40, w: 470, h: 190, label: '双塔前向（式 1：投影 → ℓ₂ 单位化）' }));

  const lane = (y, tag, tower, towerSub, proj, emb, eid0, dly0) => {
    const stages = [
      { id: `${eid0}in`, x: 20, w: 96, role: 'data', title: tag, mono: true, titleSize: 12 },
      { id: `${eid0}t`, x: 128, w: 100, role: 'op', title: tower, sub: towerSub, titleSize: 13 },
      { id: `${eid0}w`, x: 240, w: 56, role: 'proj', title: proj, mono: true, titleSize: 12.5 },
      { id: `${eid0}n`, x: 308, w: 72, role: 'norm', title: '单位化', sub: 'ℓ₂', titleSize: 12.5 },
      { id: `${eid0}e`, x: 392, w: 84, role: 'data', title: emb, mono: true, titleSize: 12 },
    ];
    const ns2 = stages.map((s) => {
      const n = node({ ...s, y, h: 48 });
      nodes.push(n); body.push(n.svg);
      return n;
    });
    for (let i = 0; i < 4; i++) {
      body.push(edge(ns2[i], ns2[i + 1], { nodes, pulse: true, dur: 1.8, delay: dly0 + i * 0.15, uid: UID, eid: `${eid0}${i}` }));
    }
    return ns2[4];
  };
  const V = lane(70, '图像 x₁..x_B', '视觉塔 f', 'ViT（6.1）', 'W_v', 'V [B, d]', 'a', 0);
  const T = lane(164, '文本 y₁..y_B', '文本塔 g', '[EOS] 读出', 'W_t', 'T [B, d]', 'b', 0.2);

  // 汇合：相似度矩阵
  const S = node({ id: 'S', x: 520, y: 100, w: 170, h: 64, role: 'attn', title: 'S = VTᵀ/τ', sub: '[B, B] 一次矩阵乘', mono: true, titleSize: 13 });
  nodes.push(S); body.push(S.svg);
  body.push(edge(V, S, { nodes, portFrom: V.right, portTo: S.left, pulse: true, dur: 2.0, delay: 0.8, uid: UID, eid: 'vs' }));
  body.push(edge(T, S, { nodes, portFrom: T.right, portTo: S.left, pulse: true, dur: 2.0, delay: 1.0, uid: UID, eid: 'ts' }));

  // batch 对比分区
  body.push(zone({ x: 20, y: 252, w: 688, h: 164, label: 'batch 结构即负样本供给（§2.1）' }));

  // 小相似度矩阵：对角热格 = 正样本
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      body.push(`<rect class="${i === j ? 'dia-mx-hot' : 'dia-mx-off'}" x="${44 + j * 18}" y="${286 + i * 18}" width="15" height="15" rx="3"/>`);
    }
  }
  body.push(cap(44, 352, '对角 = 正样本（标签）', { size: 10 }));
  body.push(cap(44, 367, '同行/列其余 B−1 = 负样本', { size: 10 }));

  // 两个方向的 CE → 平均
  const ceR = node({ id: 'ceR', x: 250, y: 278, w: 210, h: 50, role: 'op', title: '行 softmax + CE', sub: 'image→text · 标签 = i', titleSize: 13 });
  const ceC = node({ id: 'ceC', x: 480, y: 278, w: 210, h: 50, role: 'op', title: '列 softmax + CE', sub: 'text→image · 标签 = j', titleSize: 13 });
  const L = node({ id: 'L', x: 365, y: 348, w: 210, h: 46, role: 'attn', title: 'L = ½ (L_i2t + L_t2i)', mono: true, titleSize: 12 });
  [ceR, ceC, L].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(S, ceR, { nodes, portFrom: S.bottom, portTo: ceR.top, pulse: true, dur: 2.2, delay: 1.2, uid: UID, eid: 'sr' }));
  body.push(edge(S, ceC, { nodes, portFrom: S.bottom, portTo: ceC.top, pulse: true, dur: 2.2, delay: 1.35, uid: UID, eid: 'sc' }));
  body.push(edge(ceR, [430, 348], { nodes, portFrom: ceR.bottom, pulse: true, dur: 1.8, delay: 1.7, uid: UID, eid: 'rl' }));
  body.push(edge(ceC, [510, 348], { nodes, portFrom: ceC.bottom, pulse: true, dur: 1.8, delay: 1.85, uid: UID, eid: 'cl' }));

  body.push(cap(700, 408, '两方向共享同一 S：一次 B×B 矩阵乘，两份 softmax', { size: 10, anchor: 'end' }));

  return scene(720, 430, body, { label: 'CLIP 训练步：双塔各自前向并单位化，相似度矩阵的对角为正样本标签，行/列两个 B 类交叉熵取平均' }).svg();
}

/* ============================================================
 * 6.3 §2 末 — VAE 数据流（block 0）
 * 原文：x → E_φ（conv 下采样→两个线性头）→ μ_φ, σ_φ [B,h,w,c]
 *       →(式 9：z = μ + σ⊙ε，ε~N(0,I)，采样在梯度图内) z → D_θ → x̂
 *       loss = −log p_θ(x|z)（≈MSE）+ β·KL（式 7 闭式无采样）
 * ============================================================ */
function vaeFlow() {
  const UID = 'ae';
  const body = [pulseDefs(UID)];
  const nodes = [];

  // encode 分区
  body.push(zone({ x: 8, y: 24, w: 380, h: 284, label: 'encode：摊销后验 q_φ(z|x)（§2.1）' }));

  const n1 = node({ id: 'x', x: 120, y: 56, w: 240, h: 40, role: 'data', title: 'x [B, H, W, C]', mono: true, titleSize: 12.5 });
  const n2 = node({ id: 'E', x: 120, y: 116, w: 240, h: 44, role: 'op', title: '编码器 E_φ', sub: 'conv 下采样', titleSize: 13 });
  const n3 = node({ id: 'ms', x: 120, y: 180, w: 240, h: 48, role: 'proj', title: 'μ_φ(x), σ_φ(x)', sub: '高斯参数 [B, h, w, c]', mono: true, titleSize: 13 });
  const n4 = node({ id: 'z', x: 120, y: 248, w: 240, h: 48, role: 'data', title: 'z = μ_φ + σ_φ ⊙ ε', sub: '式 (9) · z [B, h, w, c]', mono: true, titleSize: 12.5 });
  const eps = node({ id: 'eps', x: 16, y: 254, w: 88, h: 36, role: 'cache', title: 'ε ~ N(0, I)', mono: true, titleSize: 11.5 });
  [n1, n2, n3, n4, eps].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // decode 分区
  body.push(zone({ x: 8, y: 320, w: 380, h: 164, label: 'decode：重建 x̂ = D_θ(z)' }));
  const n5 = node({ id: 'D', x: 120, y: 364, w: 240, h: 44, role: 'op', title: '解码器 D_θ', sub: 'conv 上采样', titleSize: 13 });
  const n6 = node({ id: 'xh', x: 120, y: 432, w: 240, h: 44, role: 'data', title: 'x̂ [B, H, W, C]', mono: true, titleSize: 12.5 });
  [n5, n6].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 主链竖流
  body.push(edge(n1, n2, { nodes, portFrom: n1.bottom, portTo: n2.top, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 'e1', arrow: false }));
  body.push(edge(n2, n3, { nodes, portFrom: n2.bottom, portTo: n3.top, pulse: true, dur: 1.8, delay: 0.35, uid: UID, eid: 'e2', arrow: false }));
  body.push(edge(n3, n4, { nodes, portFrom: n3.bottom, portTo: n4.top, pulse: true, dur: 1.8, delay: 0.7, uid: UID, eid: 'e3', arrow: false }));
  body.push(edge(eps, n4, { nodes, portFrom: eps.right, portTo: n4.left, pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 'ep', arrow: false }));
  body.push(edge(n4, n5, { nodes, portFrom: n4.bottom, portTo: n5.top, pulse: true, dur: 2.2, delay: 1.05, uid: UID, eid: 'e4', arrow: false }));
  body.push(edge(n5, n6, { nodes, portFrom: n5.bottom, portTo: n6.top, pulse: true, dur: 1.8, delay: 1.4, uid: UID, eid: 'e5', arrow: false }));

  // 间隙注记（右侧）
  body.push(cap(372, 108, 'q_φ(z|x) = N(μ_φ, diag σ_φ²)', { mono: true, size: 10 }));
  body.push(cap(372, 173, '摊销：一张网络服务所有 x（§2.1）', { size: 10 }));
  body.push(cap(372, 241, '式 (9)：梯度经 μ,σ 直通（式 10）', { size: 10 }));
  body.push(cap(372, 420, '[B, h, w, c] → [B, H, W, C]', { mono: true, size: 10 }));

  // loss：recon 从 x̂ 进；KL 闭式支路从 μ,σ 经右侧虚线走廊进（无采样）
  const L = node({ id: 'L', x: 430, y: 424, w: 280, h: 52, role: 'attn', title: 'L = −log p_θ(x|z) + β·KL', sub: '式 (8)：重建 + 先验匹配', mono: true, titleSize: 12.5 });
  nodes.push(L); body.push(L.svg);
  body.push(edge(n6, L, { nodes, portFrom: n6.right, portTo: L.left, pulse: true, dur: 1.8, delay: 1.7, uid: UID, eid: 'rl' }));
  const kl = 'M 360 204 H 616 V 424';
  body.push(`<path class="dia-edge dash" d="${kl}"/>`);
  body.push(pathPulse(kl, { dur: 2.4, delay: 1.9, uid: UID }));
  body.push(cap(488, 196, 'β·KL(q_φ ‖ N(0, I))（式 7 闭式）', { mono: true, size: 10, anchor: 'middle' }));

  body.push(cap(24, 504, '式 (5)：ELBO = E_q[log p_θ(x|z)] − KL(q_φ ‖ p)；式 (8) 加 β —— 现代连续 AE（§7）的目标骨架', { mono: true, size: 10.5 }));

  return scene(720, 518, body, { label: 'VAE 数据流：编码器出高斯参数，重参数化采样 z，解码器重建；loss = 重建项 + β·KL 闭式项' }).svg();
}

/* ============================================================
 * 6.3 §3.2 — VQ 前向硬量化 / 反向 STE 梯度拷贝（block 1）
 * 原文：前向 z_e ──argmin 查表──▶ z_q ─▶ D ─▶ x̂ ─▶ L
 *       反向 ∂L/∂z_e ◀── 原样拷贝（式 13）── ∂L/∂z_q ◀── ∂L/∂x̂
 *       k* = argmin_k ‖z_e(i,j) − e_k‖（式 11）；拷贝误差 ∝ ‖z_q − z_e‖，
 *       由 commitment 项 β‖z_e − sg[e]‖² 压住（式 14，β = 0.25）
 * ============================================================ */
function vqStraightThrough() {
  const UID = 'vq';
  const body = [pulseDefs(UID)];
  const nodes = [];

  // codebook 分区 + entry 行（k* = 3 热格）
  body.push(zone({ x: 150, y: 16, w: 300, h: 76, label: 'codebook E_cb：K 个 entry' }));
  for (let k = 0; k < 6; k++) {
    const hot = k === 2;
    body.push(
      `<rect class="${hot ? 'dia-mx-hot' : ''}" x="${170 + k * 40}" y="52" width="30" height="24" rx="4" ` +
      `${hot ? '' : 'fill="var(--paper)" stroke="var(--rule)"'}/>` +
      `<text class="dia-mono" x="${185 + k * 40}" y="68" font-size="9.5" text-anchor="middle" fill="${hot ? 'var(--seal-deep)' : 'var(--ink-faint)'}">e${['₁', '₂', '₃', '₄', '₅', '₆'][k]}</text>`
    );
  }
  body.push(cap(400, 108, 'k* = argmin_k ‖z_e(i,j) − e_k‖（式 11）', { mono: true, size: 10, anchor: 'middle' }));

  // 前向链
  body.push(cap(24, 108, '前向', { size: 11 }));
  const ze = node({ id: 'ze', x: 24, y: 120, w: 100, h: 48, role: 'data', title: 'z_e', mono: true, titleSize: 13 });
  const Q = node({ id: 'Q', x: 152, y: 120, w: 120, h: 48, role: 'op', title: 'argmin 查表', sub: '最近邻 · 式 11', titleSize: 13 });
  const zq = node({ id: 'zq', x: 296, y: 120, w: 100, h: 48, role: 'cache', title: 'z_q', sub: '= e_{k*}', mono: true, titleSize: 13 });
  const D = node({ id: 'D', x: 420, y: 120, w: 64, h: 48, role: 'op', title: 'D', mono: true, titleSize: 13 });
  const xh = node({ id: 'xh', x: 508, y: 120, w: 90, h: 48, role: 'data', title: 'x̂', mono: true, titleSize: 13 });
  const L = node({ id: 'L', x: 622, y: 120, w: 88, h: 48, role: 'attn', title: 'L', sub: '式 (14)', mono: true, titleSize: 13 });
  [ze, Q, zq, D, xh, L].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge([305, 78], Q, { nodes, portTo: Q.top, dash: true, uid: UID, eid: 'lk' }));
  [
    [ze, Q, 0], [Q, zq, 0.3], [zq, D, 0.55], [D, xh, 0.8], [xh, L, 1.05],
  ].forEach(([a, b, d], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.8, delay: d, uid: UID, eid: `f${i}` }));
  });

  // 反向梯度链（左流）
  body.push(cap(24, 232, '反向', { size: 11 }));
  const gX = node({ id: 'gx', x: 536, y: 236, w: 88, h: 36, role: 'plain', title: '∂L/∂x̂', mono: true, titleSize: 10.5 });
  const gQ = node({ id: 'gq', x: 302, y: 236, w: 88, h: 36, role: 'plain', title: '∂L/∂z_q', mono: true, titleSize: 10.5 });
  const gE = node({ id: 'ge', x: 30, y: 236, w: 88, h: 36, role: 'plain', title: '∂L/∂z_e', mono: true, titleSize: 10.5 });
  [gX, gQ, gE].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(L, [580, 236], { nodes, portFrom: L.bottom, dash: true, pulse: true, dur: 1.6, delay: 1.3, uid: UID, eid: 'b0' }));
  body.push(edge(zq, gQ, { nodes, portFrom: zq.bottom, portTo: gQ.top, dash: true, uid: UID, eid: 'b1', arrow: false }));
  body.push(edge(gX, gQ, { nodes, portFrom: gX.left, portTo: gQ.right, dash: true, pulse: true, dur: 1.8, delay: 1.5, uid: UID, eid: 'b2' }));
  // STE：梯度原样拷贝，把 argmin 当恒等映射（朱砂弧形虚线，绕过 Q）
  const ste = 'M 302 254 C 276 302, 144 302, 118 254';
  body.push(`<path class="dia-edge dash" style="stroke: var(--seal-deep); stroke-opacity: 0.55" d="${ste}"/>`);
  body.push(pathPulse(ste, { dur: 1.6, delay: 1.9, uid: UID }));
  body.push(cap(210, 306, 'STE：原样拷贝（式 13）· 雅可比 = I', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge(gE, [74, 168], { nodes, portFrom: gE.top, dash: true, pulse: true, dur: 1.6, delay: 2.2, uid: UID, eid: 'b3' }));

  body.push(cap(24, 330, '前向值 = 硬量化 z_q（式 12）；反向把 argmin 当恒等映射 —— decoder 端梯度原样拷到 encoder 端（式 13）', { size: 10.5 }));
  body.push(cap(24, 350, '拷贝误差 ∝ ‖z_q − z_e‖；commitment 项 β‖z_e − sg[e]‖² 压住（式 14，β = 0.25）', { mono: true, size: 10.5 }));

  return scene(720, 364, body, { label: 'VQ 的前向硬量化与反向 STE 梯度拷贝：argmin 查表不可微，反向按恒等映射把梯度从 z_q 拷回 z_e' }).svg();
}

/* ============================================================
 * 6.3 §5 — RVQ 残差量化级联（block 2）
 * 原文：z_e ─▶ Q1 ─▶ e1 ─┐；└─ r2 = z_e − e1 ─▶ Q2 ─▶ e2 ─┤；
 *       └─ r3 = r2 − e2 ─▶ Q3 ─▶ e3 ─┤（共 n_q 级）… ẑ = e1+e2+e3
 *       误差 = ‖r_{n_q+1}‖ = ‖z_e − ẑ‖；STE 逐级适用（式 12–13）
 * ============================================================ */
function rvqCascade() {
  const UID = 'rq';
  const body = [pulseDefs(UID)];
  const nodes = [];

  // 残差脊柱分区
  body.push(zone({ x: 12, y: 52, w: 576, h: 110, label: '残差递推 r_{i+1} = r_i − e_i（式 20）· STE 逐级适用' }));

  const ze = node({ id: 'ze', x: 24, y: 84, w: 90, h: 52, role: 'data', title: 'z_e', sub: 'h×w×d_e', mono: true, titleSize: 13 });
  nodes.push(ze); body.push(ze.svg);
  const qs = ['Q₁', 'Q₂', 'Q₃'].map((t, i) => {
    const n = node({ id: `q${i}`, x: 160 + i * 150, y: 84, w: 84, h: 52, role: 'op', title: t, sub: '最近邻', mono: true, titleSize: 13 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  // 脊柱流（残差传递）
  body.push(edge(ze, qs[0], { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 's0' }));
  body.push(edge(qs[0], qs[1], { nodes, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 's1' }));
  body.push(edge(qs[1], qs[2], { nodes, pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 's2' }));
  ['r₁', 'r₂', 'r₃'].forEach((t, i) => body.push(cap(137 + i * 150, 152, t, { mono: true, size: 10.5, anchor: 'middle' })));

  // 级联继续（共 n_q 级）
  body.push(edge(qs[2], [572, 110], { nodes, dash: true, uid: UID, eid: 's3', arrow: false }));
  body.push(`<circle cx="576" cy="110" r="2.5" fill="var(--ink-soft)"/>`);
  body.push(cap(592, 152, '··· 共 n_q 级', { mono: true, size: 10 }));

  // 每级量化出 e_i，汇入求和轨
  const es = ['e₁ ∈ C₁', 'e₂ ∈ C₂', 'e₃ ∈ C₃'].map((t, i) => {
    const n = node({ id: `e${i}`, x: 168 + i * 150, y: 190, w: 76, h: 40, role: 'cache', title: t, mono: true, titleSize: 12 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  qs.forEach((q, i) => {
    body.push(edge(q, es[i], { nodes, portFrom: q.bottom, portTo: es[i].top, pulse: true, dur: 1.5, delay: 0.35 + i * 0.4, uid: UID, eid: `t${i}`, arrow: false }));
  });

  // 求和轨 → ẑ
  const rail = 'M 206 252 H 612';
  [206, 356, 506].forEach((x) => body.push(`<line class="dia-edge" x1="${x}" y1="230" x2="${x}" y2="252"/>`));
  body.push(`<path class="dia-edge strong" d="${rail}"/>`);
  body.push(pathPulse(rail, { dur: 1.8, delay: 1.5, uid: UID, r: 3.6 }));
  const sum = node({ id: 'sum', x: 620, y: 230, w: 88, h: 44, role: 'op', title: 'ẑ = Σ eᵢ', mono: true, titleSize: 12 });
  nodes.push(sum); body.push(sum.svg);

  // 末级残差 = 误差
  const err = node({ id: 'err', x: 596, y: 170, w: 116, h: 34, role: 'attn', title: '误差 ‖r_{n_q+1}‖', mono: true, titleSize: 10.5 });
  nodes.push(err); body.push(err.svg);
  body.push(edge([576, 110], err, { nodes, portTo: err.top, dash: true, pulse: true, dur: 2.0, delay: 2.2, uid: UID, eid: 'er', arrow: false }));
  body.push(cap(654, 218, '= ‖z_e − ẑ‖', { mono: true, size: 10, anchor: 'middle' }));

  body.push(cap(24, 312, '每位点 = n_q 个下标 (k₁..k_{n_q}) ∈ [K]^{n_q}；误差 = ‖r_{n_q+1}‖ = ‖z_e − ẑ‖', { mono: true, size: 10.5 }));
  body.push(cap(24, 330, '级数越深、残差越集中、同 K 量化半径越小（经验性质）；只传前 m 级即低码率版', { size: 10.5 }));

  return scene(720, 344, body, { label: 'RVQ 残差级联：残差沿脊柱逐级过量化器，每级选出码 e_i 汇入求和得 ẑ，末级残差即总误差' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '06-vision-representations/01-vit',
    replacesBlock: 0,
    title: 'ViT 数据流：patchify → encoder → 读出',
    svg: vitShapeFlow,
  },
  {
    slug: '06-vision-representations/02-clip-and-contrastive',
    replacesBlock: 0,
    title: 'CLIP 一次训练步：双塔汇合与双向 CE',
    svg: clipDualTower,
  },
  {
    slug: '06-vision-representations/03-autoencoders-and-tokenizers',
    replacesBlock: 0,
    title: 'VAE 数据流：重参数化与两项 loss',
    svg: vaeFlow,
  },
  {
    slug: '06-vision-representations/03-autoencoders-and-tokenizers',
    replacesBlock: 1,
    title: 'VQ 前向硬量化 / 反向 STE 梯度拷贝',
    svg: vqStraightThrough,
  },
  {
    slug: '06-vision-representations/03-autoencoders-and-tokenizers',
    replacesBlock: 2,
    title: 'RVQ 残差级联：逐级细化与求和',
    svg: rvqCascade,
  },
];
