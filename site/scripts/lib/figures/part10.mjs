// Part 10（10-audio-speech）场景图：替换四章共 10 个 ASCII 块（replacesBlock，序号见 CENSUS.md）
// 规范见 STYLE.md；组件库见 figure-kit.mjs。签名图（STFT→mel / CTC 数值验证 / Moshi 延迟账）
// 在 part08-10.mjs，本文件不重复。
// 注意：kit 的 edge 脉冲引用 url(#<uid>-pg) 渐变，每个图自建同名 defs 保证解析。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/折线通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

// 残差/求和圆圈 ⊕
const addDot = (cx, cy) =>
  `<circle cx="${cx}" cy="${cy}" r="9" fill="var(--paper)" stroke="var(--ink-soft)" stroke-width="1.3"/>` +
  `<text class="dia-mono" x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" fill="var(--ink)">+</text>`;

/* ============================================================
 * 10.1 §4（block 0）— neural codec 共用骨架：Enc → z → RVQ → codes → Σ → Dec
 * 原文：24 kHz → 总 stride 320 → 75 fps latent；RVQ r₁=z, eᵢ=Qᵢ(rᵢ), rᵢ₊₁=rᵢ−eᵢ；
 * ẑ = Σᵢ eᵢ；训练期另挂 multi-scale STFT 判别器（式 9）；bitrate 式 (7)
 * ============================================================ */
function codecSkeleton() {
  const UID = 'p10a';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 26, 'SoundStream / EnCodec / DAC 共用骨架：波形 ⇄ latent ⇄ 每帧 n_q 个码；判别器只在训练期挂', { size: 12, color: 'var(--ink-soft)' }));

  // 左列（编码方向，自上而下）
  const x = node({ id: 'x', x: 36, y: 44, w: 124, h: 44, role: 'data', title: 'x[n]', sub: '24 kHz 波形', mono: true });
  const enc = node({ id: 'enc', x: 36, y: 116, w: 124, h: 48, role: 'op', title: 'Encoder', sub: '总 stride 320' });
  const z = node({ id: 'z', x: 36, y: 200, w: 124, h: 48, role: 'cache', title: 'z ∈ R^{T′×d}', sub: 'fps = 75', mono: true, titleSize: 12.5 });
  [x, enc, z].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 中：RVQ 残差链 + 码芯片
  body.push(zone({ x: 222, y: 84, w: 288, h: 196, label: 'RVQ（共 n_q 级）' }));
  const qs = [], ks = [];
  for (let i = 0; i < 4; i++) {
    const q = node({ id: `q${i}`, x: 238 + i * 68, y: 120, w: 52, h: 40, role: 'op', title: `Q${['₁', '₂', '₃', '₄'][i]}`, mono: true, titleSize: 13 });
    const k = node({ id: `k${i}`, x: 238 + i * 68, y: 196, w: 52, h: 32, role: 'data', title: `k${['₁', '₂', '₃', '₄'][i]}`, mono: true, titleSize: 12 });
    nodes.push(q, k); qs.push(q); ks.push(k);
    body.push(q.svg, k.svg);
  }
  body.push(cap(366, 176, '残差 r₁ = z，rᵢ₊₁ = rᵢ − eᵢ', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(366, 258, '每帧 n_q 个码 (k₁,…,k_{n_q}) —— 比特流本体', { size: 10, anchor: 'middle' }));
  body.push(cap(498, 142, '…', { mono: true, size: 13, anchor: 'middle', color: 'var(--ink-faint)' }));

  // 右列（解码方向，自下而上）
  const sum = node({ id: 'sum', x: 566, y: 200, w: 124, h: 44, role: 'attn', title: 'ẑ = Σᵢ eᵢ', sub: '查表求和', mono: true });
  const dec = node({ id: 'dec', x: 566, y: 116, w: 124, h: 48, role: 'op', title: 'Decoder', sub: '转置卷积镜像' });
  const xh = node({ id: 'xh', x: 566, y: 44, w: 124, h: 44, role: 'data', title: 'x̂[n]', sub: '重建波形', mono: true });
  [sum, dec, xh].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 编码流
  body.push(edge(x, enc, { nodes, portFrom: x.bottom, portTo: enc.top, pulse: true, dur: 1.6, uid: UID, eid: 'a' }));
  body.push(edge(enc, z, { nodes, portFrom: enc.bottom, portTo: z.top, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'b' }));
  body.push(edge(z, qs[0], { nodes, portFrom: z.right, portTo: qs[0].left, pulse: true, dur: 1.8, delay: 0.6, uid: UID, eid: 'c' }));
  // 残差链（Q 间横边 = 残差传递）
  for (let i = 0; i < 3; i++) {
    body.push(edge(qs[i], qs[i + 1], { nodes, pulse: true, dur: 1.2, delay: 0.6 + i * 0.25, uid: UID, eid: `r${i}` }));
  }
  // Q → 码
  qs.forEach((q, i) => {
    body.push(edge(q, ks[i], { nodes, portFrom: q.bottom, portTo: ks[i].top, uid: UID, eid: `q${i}` }));
  });
  // 码 → 求和（重建侧）
  ks.forEach((k, i) => {
    body.push(edge(k, sum, { nodes, portFrom: k.right, portTo: sum.left, pulse: i === 3, dur: 2.0, delay: 1.2 + i * 0.15, uid: UID, eid: `s${i}`, arrow: i === 3 }));
  });
  body.push(edge(sum, dec, { nodes, portFrom: sum.top, portTo: dec.bottom, pulse: true, dur: 1.6, delay: 1.6, uid: UID, eid: 'd' }));
  body.push(edge(dec, xh, { nodes, portFrom: dec.top, portTo: xh.bottom, pulse: true, dur: 1.6, delay: 1.9, uid: UID, eid: 'e' }));

  // 判别器（训练期，虚线挂在 x̂ 上）
  body.push(zone({ x: 330, y: 292, w: 360, h: 76, label: '训练期：multi-scale STFT 判别器 {D_s}（式 9）', tone: 'seal' }));
  body.push(cap(346, 334, 'N_s ∈ {2048,…,128}：五对 (Δt, Δf) 互补', { mono: true, size: 9.5 }));
  body.push(cap(346, 350, 'L_adv = Σ_s ℓ(D_s(|STFT(x)|), D_s(|STFT(x̂)|))', { mono: true, size: 9.5, color: 'var(--ink-faint)' }));
  body.push(`<path class="dia-edge dash" d="M 662 88 C 700 130, 700 220, 686 292"/>`);
  body.push(cap(712, 196, '作用于 x̂', { size: 9.5, anchor: 'end' }));

  // bitrate 账（左下）
  body.push(cap(36, 306, 'b = n_q · log₂K · fps（式 7）', { mono: true, size: 11.5, color: 'var(--ink)' }));
  body.push(cap(36, 326, 'SoundStream 6 kbps = 8 × 10 × 75', { mono: true, size: 10 }));
  body.push(cap(36, 344, '只传前 m 级即低码率档（dropout 白送）', { size: 10 }));
  body.push(cap(36, 362, 'EnCodec 五挡：1.5 – 24 kbps', { size: 10, color: 'var(--ink-faint)' }));

  return scene(720, 380, body, { label: 'neural codec 骨架：Encoder 到 latent，RVQ 逐级量化出码，查表求和后 Decoder 镜像重建' }).svg();
}

/* ============================================================
 * 10.1 §5.1（block 1）— Mimi split-RVQ：语义 VQ 与声学 RVQ 并联求和
 * 原文：ẑ = e_sem + Σᵢ eᵢ（式 11）；蒸馏 L = Σ(1−cos(g(e_sem), u_t))（式 12，对齐 WavLM）；
 * n_q = 8（1+7）、K = 2048（11 bit）、12.5 fps → 1.1 kbps / 100 tok/s（式 13）
 * ============================================================ */
function mimiSplit() {
  const UID = 'p10b';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, 'split-RVQ：语义一路、声学一路并联，输出求和 —— 蒸馏只作用在语义支路，声学链专心补重建', { size: 12, color: 'var(--ink-soft)' }));

  const z = node({ id: 'z', x: 36, y: 116, w: 110, h: 56, role: 'cache', title: 'latent z', sub: '12.5 fps', mono: true, titleSize: 13 });
  nodes.push(z); body.push(z.svg);
  body.push(cap(91, 192, '总 stride 1920', { size: 9, anchor: 'middle' }));

  // 语义支路（上）
  const qsem = node({ id: 'qsem', x: 224, y: 52, w: 150, h: 48, role: 'proj', title: 'Q_sem（语义 VQ）', sub: '单级 plain VQ', mono: true, titleSize: 12 });
  const wavlm = node({ id: 'wavlm', x: 440, y: 30, w: 150, h: 42, role: 'data', title: 'WavLM u_t', sub: '50 Hz → pool ×4', mono: true, titleSize: 12 });
  [qsem, wavlm].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(515, 92, '蒸馏：L = Σ_t (1 − cos(g(e_sem), u_t))（式 12）', { mono: true, size: 9.5, anchor: 'middle' }));

  // 声学支路（下）：RVQ ×7 残差链（画 Q₁ Q₂ … Q₇）
  body.push(zone({ x: 208, y: 150, w: 316, h: 92, label: '声学 RVQ ×7：逐级残差 e₁..e₇' }));
  const q1 = node({ id: 'q1', x: 224, y: 182, w: 54, h: 38, role: 'op', title: 'Q₁', mono: true, titleSize: 13 });
  const q2 = node({ id: 'q2', x: 296, y: 182, w: 54, h: 38, role: 'op', title: 'Q₂', mono: true, titleSize: 13 });
  const q7 = node({ id: 'q7', x: 440, y: 182, w: 54, h: 38, role: 'op', title: 'Q₇', mono: true, titleSize: 13 });
  [q1, q2, q7].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(376, 206, '⋯', { mono: true, size: 15, anchor: 'middle', color: 'var(--ink-faint)' }));

  // 求和与出口
  const sum = node({ id: 'sum', x: 566, y: 118, w: 140, h: 52, role: 'attn', title: 'ẑ = e_sem + Σᵢeᵢ', sub: '式 (11)', mono: true, titleSize: 11 });
  nodes.push(sum); body.push(sum.svg);
  body.push(addDot(552, 142));
  body.push(cap(636, 192, '→ decoder', { mono: true, size: 10, anchor: 'middle' }));

  // 边
  body.push(edge(z, qsem, { nodes, portFrom: z.top, portTo: qsem.left, pulse: true, dur: 1.8, uid: UID, eid: 'a' }));
  body.push(edge(z, q1, { nodes, portFrom: z.right, portTo: q1.left, pulse: true, dur: 1.8, delay: 0.2, uid: UID, eid: 'b' }));
  body.push(edge(q1, q2, { nodes, pulse: true, dur: 1.2, delay: 0.5, uid: UID, eid: 'c1' }));
  body.push(edge(qsem, [552, 142], { nodes, portFrom: qsem.right, pulse: true, dur: 2.0, delay: 0.8, uid: UID, eid: 'd', arrow: false }));
  body.push(edge(q7, [552, 142], { nodes, portFrom: q7.right, pulse: true, dur: 2.0, delay: 1.0, uid: UID, eid: 'e', arrow: false }));
  body.push(edge([552, 142], sum, { nodes, portTo: sum.left, uid: UID, eid: 'sum' }));
  body.push(cap(448, 124, 'e_sem', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(522, 198, 'e₇', { mono: true, size: 9.5, anchor: 'middle' }));
  // 蒸馏虚线（语义支路 ↔ WavLM 教师；走低弧避开标题行）
  body.push(`<path class="dia-edge dash" d="M 374 68 C 402 58, 414 54, 440 51"/>`);

  // 账
  body.push(cap(36, 278, 'n_q = 8（1 语义 + 7 声学）· K = 2048（11 bit）· 12.5 fps → 8×11×12.5 = 1.1 kbps（式 13）', { mono: true, size: 10.5, color: 'var(--ink)' }));
  body.push(cap(36, 298, 'token rate = 8 × 12.5 = 100 tok/s —— 10.4 全双工数学（80 ms 栅格）的地基', { size: 10.5 }));

  return scene(720, 316, body, { label: 'Mimi split-RVQ：语义 VQ 与七级声学 RVQ 并联，求和进 decoder' }).svg();
}

/* ============================================================
 * 10.2 §2.3（block 0）— CTC 状态网格与三类步进
 * 原文：y = ab → y′ = (∅,a,∅,b,∅)，S = 5；s_{t+1}−s_t ∈ {0,1,2}；
 * 起点 s∈{1,2}、终点 s∈{S−1,S}（式 9/11）；紧约束 T ≥ U + 重复处数
 * ============================================================ */
function ctcGrid() {
  const UID = 'p10c';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 24, 'y = ab → y′ = (∅, a, ∅, b, ∅)，S = 5；节点 (t,s) 权重 = P_t(y′_s)，路径概率 = 途经节点连乘', { size: 12, color: 'var(--ink-soft)' }));

  const XS = [120, 230, 340, 450, 550]; // t = 1..4, T
  const YS = { 5: 76, 4: 122, 3: 168, 2: 214, 1: 260 }; // 行：s 越大越靠上
  const SYM = { 1: '∅', 2: 'a', 3: '∅', 4: 'b', 5: '∅' };

  // 轴标签
  XS.forEach((x, i) => body.push(cap(x, 60, i === 4 ? 't = T' : `t = ${i + 1}`, { mono: true, size: 10, anchor: 'middle' })));
  for (let s = 1; s <= 5; s++) {
    body.push(cap(64, YS[s] + 4, `s=${s}`, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(cap(88, YS[s] + 4, SYM[s], { mono: true, size: 12, anchor: 'middle', color: SYM[s] === '∅' ? 'var(--ink-faint)' : 'var(--seal)' }));
  }
  // 全部格点
  for (let t = 0; t < 5; t++) {
    for (let s = 1; s <= 5; s++) {
      body.push(`<circle cx="${XS[t]}" cy="${YS[s]}" r="3" fill="var(--ink-faint)" opacity="0.45"/>`);
    }
  }
  // 起点/终点候选（式 9/11 的自由度）
  [[0, 1], [0, 2]].forEach(([t, s]) => body.push(`<circle cx="${XS[t]}" cy="${YS[s]}" r="8" fill="none" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="3 3"/>`));
  [[4, 4], [4, 5]].forEach(([t, s]) => body.push(`<circle cx="${XS[t]}" cy="${YS[s]}" r="8" fill="none" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="3 3"/>`));
  body.push(cap(120, 288, '起点 s ∈ {1, 2}（式 9）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(550, 44, '终点 s ∈ {S−1, S}（式 11）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 示例路径 π = (∅, a, a, b, ∅)：覆盖三类步进
  const P = [[0, 1], [1, 2], [2, 2], [3, 4], [4, 5]];
  const seg = (a, b) => `M ${XS[a[0]]} ${YS[a[1]]} L ${XS[b[0]]} ${YS[b[1]]}`;
  const durs = [1.6, 1.6, 1.8, 1.6];
  for (let i = 0; i < 4; i++) {
    body.push(`<path d="${seg(P[i], P[i + 1])}" fill="none" stroke="var(--seal)" stroke-width="2" stroke-linecap="round"/>`);
    body.push(pathPulse(seg(P[i], P[i + 1]), { dur: durs[i], delay: i * 0.55, uid: UID, r: 3.8 }));
  }
  P.forEach(([t, s]) => body.push(`<circle cx="${XS[t]}" cy="${YS[s]}" r="4.6" fill="var(--seal)"/>`));

  // 步进注记（沿路径就地标注）
  body.push(cap(158, 230, '↗ s→s+1：恒合法', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(285, 204, '═ 停留 s→s：重复帧，被 B 合并吸收', { size: 9.5, anchor: 'middle' }));
  body.push(cap(395, 150, '║ s→s+2：y′₄ = b ≠ a = y′₂，跳格开放', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(508, 84, '↗ s→s+1', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(600, 180, 'π = (∅, a, a, b, ∅)', { mono: true, size: 9.5, anchor: 'start' }));
  body.push(cap(600, 198, 'B(π) = ab ✓', { mono: true, size: 9.5, anchor: 'start', color: 'var(--ink-faint)' }));

  // 底部约束
  body.push(cap(36, 312, '可达带：t 帧至多走到 s = 2t；紧约束 T ≥ U + |{i : y_i = y_{i−1}}|（每处重复标签强制消耗一帧 blank）', { size: 10.5 }));
  body.push(cap(36, 330, '枚举 ∼ C(T,U) 不可行 → 前向 DP 在 O(T·U) 精确边缘化（§2.4 式 8–11）', { size: 10.5, color: 'var(--ink-faint)' }));

  return scene(720, 344, body, { label: 'CTC 状态网格：扩展标签序列上的单调行走，三类步进与起终点自由度' }).svg();
}

/* ============================================================
 * 10.2 §4.2（block 1）— RNN-T 二维格点：右移 = blank，上移 = 发标签
 * 原文：α(t,u) 递推（式 18）；终止 P(y|x) = α(T,U)·P(∅|T,U)（式 19）；
 * 同帧可连续上移多发 → 无 T≥U 约束；joint h = tanh(Wf·f_t + Wg·g_u + b)（式 17）
 * ============================================================ */
function rnntLattice() {
  const UID = 'p10d';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 24, '格点 (t,u)：发射分布 P(k | t,u)（式 17）—— u 携带标签历史 g_u，复杂性从转移规则搬进发射概率', { size: 12, color: 'var(--ink-soft)' }));

  const XS = [130, 240, 350, 460]; // t = 1,2,3,T
  const YS = [252, 196, 140, 84]; // u = 0,1,2,U
  const TL = ['t = 1', 't = 2', 't = 3', 't = T'];
  const UL = ['u = 0', 'u = 1', 'u = 2', 'u = U'];
  XS.forEach((x, i) => body.push(cap(x, 276, TL[i], { mono: true, size: 10, anchor: 'middle' })));
  YS.forEach((y, i) => body.push(cap(104, YS[i] + 4, UL[i], { mono: true, size: 9.5, anchor: 'end' })));

  // 全部合法迁移（浅色底）
  for (let t = 0; t < 4; t++) {
    for (let u = 0; u < 4; u++) {
      if (t < 3) body.push(`<line class="dia-edge" x1="${XS[t]}" y1="${YS[u]}" x2="${XS[t + 1]}" y2="${YS[u]}" opacity="0.5"/>`);
      if (u < 3) body.push(`<line class="dia-edge" x1="${XS[t]}" y1="${YS[u]}" x2="${XS[t]}" y2="${YS[u + 1]}" opacity="0.5"/>`);
    }
  }
  // 格点
  for (let t = 0; t < 4; t++) {
    for (let u = 0; u < 4; u++) {
      body.push(`<circle cx="${XS[t]}" cy="${YS[u]}" r="3.4" fill="var(--paper-raised)" stroke="var(--ink-soft)" stroke-width="1"/>`);
    }
  }

  // 示例路径：(1,0) →∅→ (2,0) →y₁→ (2,1) →∅→ (3,1) →y₂→ (3,2) →∅→ (T,2) →y_U→ (T,U)
  const path = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3]];
  const seg = (a, b) => `M ${XS[a[0]]} ${YS[a[1]]} L ${XS[b[0]]} ${YS[b[1]]}`;
  for (let i = 0; i < path.length - 1; i++) {
    body.push(`<path d="${seg(path[i], path[i + 1])}" fill="none" stroke="var(--seal)" stroke-width="2" stroke-linecap="round"/>`);
    body.push(pathPulse(seg(path[i], path[i + 1]), { dur: 1.5, delay: i * 0.45, uid: UID, r: 3.8 }));
  }
  path.forEach(([t, u]) => body.push(`<circle cx="${XS[t]}" cy="${YS[u]}" r="4.4" fill="var(--seal)"/>`));
  // 迁移标签
  body.push(cap(185, 244, '∅', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(252, 228, 'y₁', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(295, 188, '∅', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(362, 172, 'y₂', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(405, 132, '∅', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(472, 116, 'y_U', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  // 终点：须以 ∅ 迁移收尾（式 19）
  body.push(`<circle cx="${XS[3]}" cy="${YS[3]}" r="9" fill="none" stroke="var(--seal)" stroke-width="1.3" stroke-dasharray="3 3"/>`);
  body.push(cap(482, 70, '终点再以 ∅ 收尾', { size: 9.5, anchor: 'start', color: 'var(--seal-deep)' }));

  // 右侧规则注记
  body.push(cap(530, 150, '→ blank 迁移：消费一帧', { size: 10, anchor: 'start' }));
  body.push(cap(530, 170, '↑ 标签迁移：发 y_{u+1}，留原帧', { size: 10, anchor: 'start' }));
  body.push(cap(530, 190, '同帧可连续上移多发', { size: 10, anchor: 'start', color: 'var(--seal-deep)' }));
  body.push(cap(530, 206, '→ 无 CTC 的 T ≥ U 约束', { size: 10, anchor: 'start', color: 'var(--seal-deep)' }));

  // 底部递推与成本
  body.push(cap(36, 302, 'α(t,u) = α(t−1,u)·P(∅|t−1,u) + α(t,u−1)·P(y_u|t,u−1)（式 18）—— 两类来源互斥且穷尽', { mono: true, size: 10 }));
  body.push(cap(36, 320, 'P(y|x) = α(T,U)·P(∅|T,U)（式 19）；成本税：O(TU) 次 joint softmax，logit 张量 [T, U, |V|+1]', { mono: true, size: 10, color: 'var(--ink-faint)' }));

  return scene(720, 336, body, { label: 'RNN-T 二维格点：右移消费帧、上移发标签，路径为两者的任意交错' }).svg();
}

/* ============================================================
 * 10.3 §2（block 0）— 级联 TTS 管线：G2P → duration → LR 展开 → acoustic → vocoder
 * 原文：LR((h_u),(d_u)) 纯复制展开（式 3，T = Σd_u）；variance adaptor + pitch/energy（式 4）
 * ============================================================ */
function ttsCascade() {
  const UID = 'p10e';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, '级联 = 把 z 拆成显式模块：duration 钉节奏骨架（式 3），pitch/energy 补韵律色彩（式 4）', { size: 12, color: 'var(--ink-soft)' }));

  // Row A：文本 → G2P → 音素 → duration
  const txt = node({ id: 'txt', x: 36, y: 44, w: 84, h: 42, role: 'data', title: '文本', titleSize: 13 });
  const g2p = node({ id: 'g2p', x: 156, y: 44, w: 84, h: 42, role: 'op', title: 'G2P', mono: true });
  const pho = node({ id: 'pho', x: 276, y: 44, w: 120, h: 42, role: 'data', title: 'y₁..y_U', sub: '音素序列', mono: true });
  const dur = node({ id: 'dur', x: 436, y: 44, w: 150, h: 42, role: 'proj', title: 'duration model', sub: 'forced alignment', mono: true, titleSize: 12 });
  [txt, g2p, pho, dur].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(txt, g2p, { nodes, pulse: true, dur: 1.4, uid: UID, eid: 'a' }));
  body.push(edge(g2p, pho, { nodes, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'b' }));
  body.push(edge(pho, dur, { nodes, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'c' }));
  body.push(cap(511, 106, 'd = (3, 1, 2)：每音素帧数', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));

  // LR 区：h_u 复制展开为 T 帧
  body.push(zone({ x: 36, y: 126, w: 470, h: 176, label: 'length regulator：h_u 重复 d_u 次（式 3，纯复制、无参数）' }));
  const hs = [
    { t: 'h₁', d: 3, y: 156 },
    { t: 'h₂', d: 1, y: 204 },
    { t: 'h₃', d: 2, y: 252 },
  ].map((h, i) => {
    const n = node({ id: `h${i}`, x: 64, y: h.y, w: 64, h: 40, role: 'data', title: h.t, sub: `d = ${h.d}`, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const frames = [];
  const owner = [0, 0, 0, 1, 2, 2];
  for (let i = 0; i < 6; i++) {
    const f = node({ id: `f${i}`, x: 212 + i * 48, y: 196, w: 38, h: 34, role: 'cache', title: ['h₁', 'h₁', 'h₁', 'h₂', 'h₃', 'h₃'][i], mono: true, titleSize: 10.5 });
    nodes.push(f); frames.push(f); body.push(f.svg);
  }
  hs.forEach((h, i) => {
    const hn = nodes.find((n) => n.id === `h${i}`);
    frames.forEach((f, j) => {
      if (owner[j] === i) {
        body.push(edge(hn, f, { nodes, portFrom: hn.right, portTo: f.top, pulse: true, dur: 1.6, delay: 0.6 + j * 0.15, uid: UID, eid: `lr${j}` }));
      }
    });
  });
  body.push(cap(494, 290, 'T = Σ d_u = 6 帧', { mono: true, size: 9.5, anchor: 'end' }));

  // variance adaptor（侧输入）
  const va = node({ id: 'va', x: 546, y: 146, w: 140, h: 48, role: 'op', title: 'variance adaptor', sub: '+ pitch · energy', mono: true, titleSize: 11.5 });
  nodes.push(va); body.push(va.svg);
  body.push(edge(va, [500, 226], { nodes, portFrom: va.left, uid: UID, eid: 'va' }));

  // Row B：frames → acoustic → mel → vocoder → 波形
  const ac = node({ id: 'ac', x: 120, y: 316, w: 140, h: 44, role: 'proj', title: 'acoustic model', mono: true, titleSize: 12 });
  const mel = node({ id: 'mel', x: 300, y: 316, w: 110, h: 44, role: 'data', title: 'mel', sub: 'x ∈ R^{T×K}', mono: true });
  const voc = node({ id: 'voc', x: 450, y: 316, w: 100, h: 44, role: 'op', title: 'vocoder', mono: true, titleSize: 12 });
  const wav = node({ id: 'wav', x: 590, y: 316, w: 100, h: 44, role: 'data', title: 'x̂[n]', sub: '波形', mono: true });
  [ac, mel, voc, wav].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([308, 230], ac, { nodes, portTo: ac.top, pulse: true, dur: 1.8, delay: 1.4, uid: UID, eid: 'f2a', arrow: false }));
  body.push(edge(ac, mel, { nodes, pulse: true, dur: 1.4, delay: 1.7, uid: UID, eid: 'am' }));
  body.push(edge(mel, voc, { nodes, pulse: true, dur: 1.4, delay: 1.9, uid: UID, eid: 'mv' }));
  body.push(edge(voc, wav, { nodes, pulse: true, dur: 1.4, delay: 2.1, uid: UID, eid: 'vw' }));

  return scene(720, 384, body, { label: '级联 TTS：G2P 出音素，duration 定帧数，length regulator 复制展开，acoustic model 出 mel，vocoder 出波形' }).svg();
}

/* ============================================================
 * 10.3 §6（block 1）— VALL-E 两阶段：AR 出第一层码，NAR 逐层精修
 * 原文：式 (15) 按层链式分解；式 (16) 第一层 AR 逐帧；式 (17) j=2..8 层内 T 帧并行；
 * 步数账：摊平 8T 严格串行 → T 步 AR + 7 趟并行（75 fps 档 ≈ 600 → 75+7）
 * ============================================================ */
function valleTwoStage() {
  const UID = 'p10f';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, '按量化器层链式分解（式 15）：第 1 层 AR 逐帧（式 16），第 2..8 层 NAR 层内并行（式 17）', { size: 12, color: 'var(--ink-soft)' }));

  // 条件输入
  body.push(zone({ x: 24, y: 44, w: 150, h: 180, label: '条件输入' }));
  const y = node({ id: 'y', x: 44, y: 70, w: 110, h: 44, role: 'data', title: '音素序列 y', mono: true, titleSize: 12 });
  const pr = node({ id: 'pr', x: 44, y: 140, w: 110, h: 52, role: 'cache', title: 'C̃_{:,1}', sub: '3 秒 prompt', mono: true, titleSize: 12 });
  [y, pr].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(99, 212, '第一层码', { size: 9, anchor: 'middle' }));

  // Stage 1：AR LM 逐帧串行
  body.push(zone({ x: 200, y: 44, w: 210, h: 180, label: 'AR LM（decoder-only，式 16）' }));
  const chips = [];
  for (let i = 0; i < 4; i++) {
    const c = node({ id: `c${i}`, x: 222 + i * 44, y: 84, w: 36, h: 34, role: 'cache', title: `c${['₁', '₂', '₃', '₄'][i]}`, mono: true, titleSize: 11 });
    nodes.push(c); chips.push(c); body.push(c.svg);
  }
  for (let i = 0; i < 3; i++) {
    body.push(edge(chips[i], chips[i + 1], { nodes, pulse: true, dur: 1.2, delay: i * 0.35, uid: UID, eid: `ar${i}` }));
  }
  body.push(cap(306, 142, '逐帧串行 t = 1..T', { size: 9.5, anchor: 'middle' }));
  const c1 = node({ id: 'c1', x: 222, y: 160, w: 168, h: 34, role: 'cache', title: 'c_{:,1}（第一层全 T 帧就绪）', mono: true, titleSize: 10.5 });
  nodes.push(c1); body.push(c1.svg);
  body.push(edge(y, [222, 101], { nodes, portFrom: y.right, uid: UID, eid: 'in1' }));
  body.push(edge(pr, [222, 110], { nodes, portFrom: pr.right, uid: UID, eid: 'in2' }));
  body.push(cap(36, 250, '3 秒 prompt = 前缀条件化：音色从序列条件流入，无 speaker embedding', { size: 10 }));

  // Stage 2：NAR LM 逐层推进、层内并行
  body.push(zone({ x: 450, y: 44, w: 246, h: 180, label: 'NAR LM：j = 2..8 逐层（式 17）' }));
  const rows = [{ j: 2, y: 76 }, { j: 3, y: 104 }, { j: 4, y: 132 }];
  rows.forEach((r, ri) => {
    body.push(cap(466, r.y + 14, `j=${r.j}`, { mono: true, size: 9.5 }));
    for (let c = 0; c < 4; c++) {
      body.push(`<rect class="dia-mx-warm sig-cell" style="--i:${(1.0 + ri * 0.35).toFixed(2)}" x="${500 + c * 34}" y="${r.y}" width="28" height="20" rx="3"/>`);
    }
  });
  body.push(cap(573, 166, '⋮', { mono: true, size: 14, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(cap(466, 194, 'j=8', { mono: true, size: 9.5 }));
  for (let c = 0; c < 4; c++) {
    body.push(`<rect class="dia-mx-warm sig-cell" style="--i:2.1" x="${500 + c * 34}" y="180" width="28" height="20" rx="3"/>`);
  }
  body.push(cap(573, 216, '层内 T 帧并行，层间串行', { size: 9.5, anchor: 'middle' }));
  body.push(edge(c1, [450, 150], { nodes, portFrom: c1.right, pulse: true, dur: 1.8, delay: 0.9, uid: UID, eid: 's12' }));
  body.push(cap(420, 138, '条件：c_{:,<j} 全部', { size: 9, anchor: 'middle' }));

  // 出口
  const dec = node({ id: 'dec', x: 280, y: 300, w: 150, h: 44, role: 'op', title: 'codec decoder', mono: true, titleSize: 12 });
  const wav = node({ id: 'wav', x: 480, y: 300, w: 110, h: 44, role: 'data', title: 'x̂[n]', sub: '波形', mono: true });
  [dec, wav].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(`<path class="dia-edge" d="M 573 224 V 322 H 430"/>`);
  body.push(pathPulse('M 573 224 V 322 H 430', { dur: 2.2, delay: 1.6, uid: UID }));
  body.push(edge(dec, wav, { nodes, pulse: true, dur: 1.4, delay: 2.0, uid: UID, eid: 'dw' }));

  body.push(cap(36, 368, '串行步数账：摊平 8T ≈ 600 步/秒 → T 步 AR + 7 趟层内并行 ≈ 75 + 7（75 fps 档）', { mono: true, size: 10.5, color: 'var(--ink)' }));

  return scene(720, 384, body, { label: 'VALL-E 两阶段：AR 逐帧生成第一层码，NAR 逐层并行精修第 2 至 8 层' }).svg();
}

/* ============================================================
 * 10.3 §7（block 2）— CosyVoice 三段式：LLM → semantic tokens → FM → HiFi-GAN
 * 原文：token 为 supervised semantic tokens（ASR encoder 内插 VQ）；FM 以 tokens +
 * speaker embedding 为条件回归 mel；HiFi-GAN 复用 §3；token rate 25 Hz
 * ============================================================ */
function cosyPipeline() {
  const UID = 'p10g';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, 'LM 拿语言先验与 in-context，FM + vocoder 拿音质 —— codec 的 RVQ 层级被整个绕开（单流逐帧）', { size: 12, color: 'var(--ink-soft)' }));

  // Row 1：文本 → LLM → semantic tokens
  const txt = node({ id: 'txt', x: 36, y: 64, w: 84, h: 48, role: 'data', title: '文本', titleSize: 13 });
  const llm = node({ id: 'llm', x: 160, y: 64, w: 150, h: 48, role: 'attn', title: 'LLM', sub: 'text-to-token · AR', mono: true });
  const tok = node({ id: 'tok', x: 340, y: 64, w: 160, h: 48, role: 'cache', title: 'semantic tokens', sub: 'ASR encoder 内插 VQ', mono: true, titleSize: 12 });
  [txt, llm, tok].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(txt, llm, { nodes, pulse: true, dur: 1.4, uid: UID, eid: 'a' }));
  body.push(edge(llm, tok, { nodes, pulse: true, dur: 1.4, delay: 0.25, uid: UID, eid: 'b' }));

  // speaker 条件（侧输入）
  const spk = node({ id: 'spk', x: 530, y: 96, w: 150, h: 34, role: 'cache', title: 'speaker embedding', mono: true, titleSize: 10.5 });
  nodes.push(spk); body.push(spk.svg);

  // Row 2：FM → mel → HiFi-GAN → 波形
  const fm = node({ id: 'fm', x: 520, y: 165, w: 160, h: 52, role: 'proj', title: 'conditional FM decoder', sub: '→ mel（OT 路径）', mono: true, titleSize: 11.5 });
  const mel = node({ id: 'mel', x: 340, y: 170, w: 110, h: 42, role: 'data', title: 'mel', mono: true });
  const hfg = node({ id: 'hfg', x: 160, y: 170, w: 130, h: 42, role: 'op', title: 'HiFi-GAN', sub: '§3 部件复用', mono: true });
  const wav = node({ id: 'wav', x: 36, y: 170, w: 90, h: 42, role: 'data', title: 'x̂[n]', sub: '波形', mono: true });
  [fm, mel, hfg, wav].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(tok, fm, { nodes, portFrom: tok.right, portTo: fm.top, pulse: true, dur: 1.8, delay: 0.5, uid: UID, eid: 'c' }));
  body.push(edge(spk, fm, { nodes, portFrom: spk.bottom, portTo: fm.top, uid: UID, eid: 'spk' }));
  body.push(edge(fm, mel, { nodes, portFrom: fm.left, portTo: mel.right, pulse: true, dur: 1.6, delay: 0.9, uid: UID, eid: 'fm' }));
  body.push(edge(mel, hfg, { nodes, portFrom: mel.left, portTo: hfg.right, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'mh' }));
  body.push(edge(hfg, wav, { nodes, portFrom: hfg.left, portTo: wav.right, pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'hw' }));

  body.push(cap(36, 246, 'token 25 Hz ⇒ AR 段每秒 25 步；token 来源：多语 ASR encoder 内的 VQ 瓶颈（supervised semantic tokens）', { size: 10.5 }));
  body.push(cap(36, 264, 'CosyVoice 2：chunk-aware causal FM 一套权重四档 mask，流式/非流式同模型（式 18）', { size: 10, color: 'var(--ink-faint)' }));

  return scene(720, 280, body, { label: 'CosyVoice 三段式：LLM 出语义 token，FM 以 token 与说话人条件回归 mel，HiFi-GAN 出波形' }).svg();
}

/* ============================================================
 * 10.3 §8（block 3）— F5-TTS infilling：上下文干净段 + 挖空段，filler 文本轴
 * 原文：挖空段 x_t（噪声→数据），损失只算 m=1（式 21）；文本 filler 补齐至 T（式 19）；
 * x_t = (1−t)x₀ + tε，u_t = ε − x₀（式 20）
 * ============================================================ */
function f5Infilling() {
  const UID = 'p10h';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 24, 'infilling：mask 位置即任务指定符 —— TTS / 编辑 / 去噪同一模型；对齐被写进输入（filler，式 19）', { size: 12, color: 'var(--ink-soft)' }));

  // mel 帧轴
  body.push(cap(36, 64, 'mel 帧轴 t = 1..T', { mono: true, size: 10 }));
  body.push(cap(80, 78, '上下文段：干净 x₀（条件输入）', { size: 10, anchor: 'start' }));
  body.push(cap(330, 78, '挖空段：x_t（噪声 → 数据，m = 1 处计损失）', { size: 10, anchor: 'start', color: 'var(--seal-deep)' }));
  const ctxChips = [], mskChips = [];
  for (let i = 0; i < 4; i++) {
    const c = node({ id: `x0_${i}`, x: 80 + i * 58, y: 88, w: 52, h: 40, role: 'data', title: 'x₀', mono: true, titleSize: 12 });
    const m = node({ id: `xt_${i}`, x: 330 + i * 58, y: 88, w: 52, h: 40, role: 'attn', title: 'x_t', mono: true, titleSize: 12 });
    nodes.push(c, m); ctxChips.push(c); mskChips.push(m);
    body.push(c.svg, m.svg);
  }
  body.push(`<line x1="310" y1="60" x2="310" y2="136" stroke="var(--rule)" stroke-width="1" stroke-dasharray="4 4"/>`);
  body.push(edge(ctxChips[3], mskChips[0], { nodes, dash: true, pulse: true, dur: 1.8, uid: UID, eid: 'ctx' }));
  body.push(cap(310, 152, '双向上下文作条件流入', { size: 9.5, anchor: 'middle' }));

  // 文本轴：filler 补齐
  body.push(cap(36, 190, '文本轴：参考 chars + 目标 chars，filler 补齐至 T（式 19）', { mono: true, size: 10 }));
  const CHARS = ['r', 'e', 'f', '─', '目', '标', '文', '本', '─', '─'];
  CHARS.forEach((c, i) => {
    const filler = c === '─';
    const n = node({ id: `ch${i}`, x: 80 + i * 52, y: 204, w: 44, h: 30, role: filler ? 'plain' : 'data', title: c, mono: filler, titleSize: 11.5 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(edge([420, 204], [420, 128], { nodes, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 'cond' }));
  body.push(cap(434, 172, '条件 c（ConvNeXt V2 精炼）', { size: 9.5, anchor: 'start' }));

  // 底部
  body.push(cap(36, 258, '训练 = 随机挖一段学回填；zero-shot 推理 = 目标整句挖掉（条件：参考音频 + 参考/目标文本）', { size: 10.5 }));
  body.push(cap(36, 276, 'x_t = (1−t)·x₀ + t·ε，u_t = ε − x₀（式 20）；L = ‖(v_θ(x_t, t, c) − u_t) ⊙ m‖²（式 21）', { mono: true, size: 9.5, color: 'var(--ink-faint)' }));

  return scene(720, 292, body, { label: 'F5-TTS infilling：上下文段给干净 x0 作条件，挖空段从噪声回归数据，损失只计挖空段' }).svg();
}

/* ============================================================
 * 10.4 §4.1（block 0）— 12.5 Hz 栅格上的三流并行
 * 原文：用户流 aᵘ 为观测（Mimi 实时供给）；模型文本 w（depth 0）与模型音频 aᵐ（depth 1..8）
 * 为生成目标；c_s = Temporal(x_{<s}) 只含 ≤ s−1 帧全流 token，同帧用户码本不进本帧生成
 * ============================================================ */
function moshiStreams() {
  const UID = 'p10i';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 22, '80 ms 栅格（12.5 Hz）上三条流并行；同帧三流互为「同时」—— 模型帧 s 最早反映用户 ≤ s−1 帧', { size: 12, color: 'var(--ink-soft)' }));

  const CX = [180, 320, 460, 600]; // 帧 s-2, s-1, s, s+1
  const FL = ['s−2', 's−1', 's', 's+1'];
  CX.forEach((x, i) => {
    body.push(cap(x, 44, FL[i], { mono: true, size: 10, anchor: 'middle' }));
    body.push(`<line x1="${x}" y1="54" y2="262" stroke="var(--rule-soft)" stroke-width="1" stroke-dasharray="2 5"/>`);
  });

  // 泳道标签
  body.push(cap(142, 90, '用户流 aᵘ', { size: 10, anchor: 'end' }));
  body.push(cap(142, 104, '（观测 · Mimi 实时供给）', { size: 9, anchor: 'end', color: 'var(--ink-faint)' }));
  body.push(cap(142, 165, '模型文本 w', { size: 10, anchor: 'end' }));
  body.push(cap(142, 179, '（Inner Monologue · depth 0）', { size: 9, anchor: 'end', color: 'var(--ink-faint)' }));
  body.push(cap(142, 240, '模型音频 aᵐ', { size: 10, anchor: 'end' }));
  body.push(cap(142, 254, '（8 码本 · depth 1..8）', { size: 9, anchor: 'end', color: 'var(--ink-faint)' }));

  // 用户流：观测点 + 实时供给线
  body.push('<line class="dia-edge strong" x1="180" y1="92" x2="648" y2="92"/>');
  body.push('<polygon points="656,92 646,88 646,96" fill="var(--ink-soft)"/>');
  CX.forEach((x, i) => {
    body.push(`<circle class="sig-step" style="--i:${i}" cx="${x}" cy="92" r="5" fill="var(--ink)"/>`);
  });
  body.push(pathPulse('M 180 92 H 648', { dur: 4.2, uid: UID, r: 3.4 }));

  // 模型流：w / aᵐ 在 s−1、s 两帧已生成，s+1 尚未
  const wChips = [], amChips = [];
  [['s−1', 320], ['s', 460]].forEach(([t, x], i) => {
    const w = node({ id: `w${i}`, x: x - 32, y: 150, w: 64, h: 34, role: 'proj', title: `w_{${t}}`, mono: true, titleSize: 11 });
    const a = node({ id: `a${i}`, x: x - 32, y: 224, w: 64, h: 36, role: 'attn', title: '8 码本', mono: true, titleSize: 10.5 });
    nodes.push(w, a); wChips.push(w); amChips.push(a);
    body.push(w.svg, a.svg);
  });
  [600].forEach((x) => {
    body.push(`<rect x="${x - 32}" y="150" width="64" height="34" rx="10" fill="none" stroke="var(--rule)" stroke-dasharray="3 4"/>`);
    body.push(`<rect x="${x - 32}" y="224" width="64" height="36" rx="10" fill="none" stroke="var(--rule)" stroke-dasharray="3 4"/>`);
  });

  // 上下文瓶颈 c：c_{s-1} → c_s 沿轨推进；c_s 供给本帧生成
  [320, 460].forEach((x, i) => {
    body.push(`<circle cx="${x}" cy="124" r="8" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.2"/>`);
    body.push(cap(x, 112, i === 0 ? 'c_{s−1}' : 'c_s', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  });
  body.push('<line class="dia-edge strong" x1="328" y1="124" x2="452" y2="124"/>');
  body.push(pathPulse('M 328 124 H 452', { dur: 2.0, delay: 0.4, uid: UID, r: 3.4 }));
  // c → 本帧 w（直下）与 aᵐ（绕过 w 卡）
  [320, 460].forEach((x, i) => {
    body.push(`<path class="dia-edge" d="M ${x} 132 V 150"/>`);
    const around = `M ${x + 9} 130 C ${x + 38} 148, ${x + 42} 196, ${x + 18} 222`;
    body.push(`<path class="dia-edge" d="${around}"/>`);
    body.push(pathPulse(`M ${x} 132 V 150`, { dur: 1.2, delay: 0.8 + i * 0.5, uid: UID, r: 3.2 }));
    body.push(pathPulse(around, { dur: 1.6, delay: 1.1 + i * 0.5, uid: UID, r: 3.2 }));
  });
  // 历史全流 → c_s（含 ≤ s−1 帧用户码）
  body.push('<path class="dia-edge dash" d="M 326 96 C 370 108, 412 112, 451 121"/>');

  // 底部注记
  body.push(cap(36, 290, '● = 输入观测；↓ = 该帧生成；c_s = Temporal(x_{<s})：只含 ≤ s−1 帧的全流 token', { size: 10.5 }));
  body.push(cap(36, 308, '同帧用户码本不进本帧生成 —— 一帧反应滞后正是 §6.2「2 帧 = 160 ms」的结构来源', { size: 10.5, color: 'var(--ink-faint)' }));

  return scene(720, 322, body, { label: 'Moshi 三流并行：用户流观测、模型文本与音频在 80ms 栅格上逐帧生成' }).svg();
}

/* ============================================================
 * 10.4 §4.3（block 1）— RQ-Transformer 一帧的生成流
 * 原文：c_s → Depth step 0（文本头）→ … → step 8（码本 8 头），e(·) 逐级回注（式 6）；
 * 帧末 v_s = Σ 全流 embedding（含本帧用户码），h_s = Temporal(v_{≤s})，c_{s+1} = W_o h_s（式 5）
 * ============================================================ */
function rqFrame() {
  const UID = 'p10j';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 22, 'c_s 由上一帧末备好（式 5）；帧内 9 步 = 链式法则对固定深度序的逐字展开，无近似（式 6）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：c_s
  const cs = node({ id: 'cs', x: 36, y: 64, w: 96, h: 44, role: 'op', title: 'c_s', sub: '帧 s 上下文', mono: true });
  nodes.push(cs); body.push(cs.svg);

  // 中：Depth 链
  body.push(zone({ x: 180, y: 40, w: 290, h: 232, label: 'Depth Transformer（小：6 层 · d = 1024）' }));
  const s0 = node({ id: 's0', x: 206, y: 66, w: 238, h: 36, role: 'proj', title: 'step 0：P(w_s | c_s) — 文本头', mono: true, titleSize: 10.5 });
  const s1 = node({ id: 's1', x: 206, y: 118, w: 238, h: 36, role: 'proj', title: 'step 1：P(aᵐ_{s,1} | c_s, w_s) — 码本 1', mono: true, titleSize: 10 });
  const s8 = node({ id: 's8', x: 206, y: 206, w: 238, h: 36, role: 'proj', title: 'step 8：P(aᵐ_{s,8} | c_s, w_s, aᵐ_{s,<8})', mono: true, titleSize: 10 });
  [s0, s1, s8].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(368, 186, '⋮ 逐级自回归', { size: 10, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(edge(cs, s0, { nodes, portFrom: cs.right, portTo: s0.left, pulse: true, dur: 1.6, uid: UID, eid: 'c0' }));
  body.push(edge(s0, s1, { nodes, portFrom: s0.bottom, portTo: s1.top, pulse: true, dur: 1.2, delay: 0.4, uid: UID, eid: 'd1' }));
  body.push(cap(336, 112, 'e(w_s)', { mono: true, size: 9 }));
  body.push(edge(s1, s8, { nodes, portFrom: s1.bottom, portTo: s8.top, dash: true, uid: UID, eid: 'd2' }));
  body.push(cap(336, 168, 'e(aᵐ_{s,1}) ↘', { mono: true, size: 9 }));

  // 右：帧末闭环
  body.push(zone({ x: 510, y: 40, w: 190, h: 250, label: '帧末（式 5）' }));
  const all = node({ id: 'all', x: 524, y: 66, w: 166, h: 36, role: 'data', title: 'aᵘ_s · w_s · aᵐ_s', sub: '本帧全流 token', mono: true, titleSize: 10.5 });
  const sum = node({ id: 'sum', x: 524, y: 124, w: 166, h: 40, role: 'attn', title: 'v_s = Σ 全流 embedding', mono: true, titleSize: 10.5 });
  const tmp = node({ id: 'tmp', x: 524, y: 184, w: 166, h: 44, role: 'attn', title: 'h_s = Temporal(v_{≤s})', sub: '帧级因果 Transformer', mono: true, titleSize: 10.5 });
  const cn = node({ id: 'cn', x: 524, y: 246, w: 166, h: 36, role: 'cache', title: 'c_{s+1} = W_o h_s', mono: true, titleSize: 10.5 });
  [all, sum, tmp, cn].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(all, sum, { nodes, portFrom: all.bottom, portTo: sum.top, uid: UID, eid: 'as' }));
  body.push(edge(sum, tmp, { nodes, portFrom: sum.bottom, portTo: tmp.top, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'st' }));
  body.push(edge(tmp, cn, { nodes, portFrom: tmp.bottom, portTo: cn.top, pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'tc' }));
  body.push('<line class="dia-edge strong" x1="607" y1="282" x2="607" y2="304"/>');
  body.push('<polygon points="607,310 602,300 612,300" fill="var(--ink-soft)"/>');
  body.push(cap(626, 306, '→ 帧 s+1', { size: 9.5, anchor: 'start' }));
  // 本帧生成物回写帧末求和
  body.push(edge(s8, sum, { nodes, portFrom: s8.right, portTo: sum.left, pulse: true, dur: 2.0, delay: 0.9, uid: UID, eid: 'fb' }));
  body.push(cap(478, 178, 'w_s, aᵐ 回写 →', { size: 9, anchor: 'end' }));

  // 底部账
  body.push(cap(36, 330, '每帧成本：Temporal 1 步 + Depth 9 步 ≪ 展平基线 17 token 步；随 S 增长项的 Q² = 289 倍乘子降回 1 倍', { size: 10.5 }));
  body.push(cap(36, 348, 'KV cache：Temporal 每帧 +1 条；Depth cache 帧内 ≤ 9 条、帧末作废（10 分钟对话的 cache 读取量少 17×）', { size: 10, color: 'var(--ink-faint)' }));

  return scene(720, 364, body, { label: 'RQ-Transformer 一帧：c_s 经 9 步深度自回归出文本与 8 码本，帧末求和回 Temporal 产 c_{s+1}' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '10-audio-speech/01-audio-representations', replacesBlock: 0, title: 'neural codec 骨架：Enc → RVQ → Dec', svg: codecSkeleton },
  { slug: '10-audio-speech/01-audio-representations', replacesBlock: 1, title: 'Mimi split-RVQ：语义/声学并联求和', svg: mimiSplit },
  { slug: '10-audio-speech/02-asr', replacesBlock: 0, title: 'CTC 状态网格与三类步进', svg: ctcGrid },
  { slug: '10-audio-speech/02-asr', replacesBlock: 1, title: 'RNN-T 二维格点：右移 / 上移', svg: rnntLattice },
  { slug: '10-audio-speech/03-tts', replacesBlock: 0, title: '级联管线：duration + LR 复制展开', svg: ttsCascade },
  { slug: '10-audio-speech/03-tts', replacesBlock: 1, title: 'VALL-E：AR 第一层 + NAR 逐层精修', svg: valleTwoStage },
  { slug: '10-audio-speech/03-tts', replacesBlock: 2, title: 'CosyVoice：LLM + FM + HiFi-GAN', svg: cosyPipeline },
  { slug: '10-audio-speech/03-tts', replacesBlock: 3, title: 'F5-TTS：infilling 与 filler 对齐', svg: f5Infilling },
  { slug: '10-audio-speech/04-speech-llm-and-duplex', replacesBlock: 0, title: '12.5 Hz 栅格上的三流并行', svg: moshiStreams },
  { slug: '10-audio-speech/04-speech-llm-and-duplex', replacesBlock: 1, title: 'RQ-Transformer：一帧的 9 步深度生成', svg: rqFrame },
];
