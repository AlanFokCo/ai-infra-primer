// Part 4（04-modern-architectures）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 part01.mjs。
// 每图自建 pulseDefs（与 scene() 内 defs 同构、id 用本图前缀），供 edge(uid) 的脉冲解析。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（L 形/环形/左缘绕行通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

const SUBS = ['₁', '₂', '₃', '₄'];

/* ============================================================
 * 4.1 §3 — MHA / GQA / MQA 三态对照（block 0）
 * 原文：H_q = 32；GQA g = 4 ⇒ H_kv = 8（Llama-3.1-8B）；MQA 全局独一份。
 * 账（8B 形态，bf16，全层每 token）：MHA 512 KiB（反事实）→ GQA-8 128 KiB → MQA 16 KiB。
 * ============================================================ */
function mhaGqaMqa() {
  const UID = 'p4mha';
  const body = [pulseDefs(UID)];
  const zones = [
    { x: 16, label: 'MHA（g = 1）', sub: '每头一对 K/V', stat1: 'H_kv = 32（反事实）', stat2: '512 KiB / token' },
    { x: 252, label: 'GQA（g = 4）', sub: '组内 4 头共享', stat1: 'H_kv = 8', stat2: '128 KiB / token' },
    { x: 488, label: 'MQA（g = 32）', sub: '全 32 头共享', stat1: 'H_kv = 1', stat2: '16 KiB / token' },
  ];
  body.push(cap(16, 22, '谁和谁共享一对 K/V（H_q = 32 口径）——2.4 式 (9) 里被动手的因子只有 H_kv', { size: 12, color: 'var(--ink-soft)' }));
  zones.forEach((z, zi) => {
    body.push(zone({ x: z.x, y: 36, w: 216, h: 204, label: z.label }));
    body.push(cap(z.x + 16, 76, z.sub, { size: 10.5 }));
    const qx = (k) => z.x + 21 + k * 46;
    // Q 头行
    for (let k = 0; k < 4; k++) {
      const title = zi === 2 && k === 3 ? '…' : `h${SUBS[k]}`;
      const q = node({ id: `z${zi}q${k}`, x: qx(k), y: 88, w: 36, h: 30, role: 'data', title, mono: true, titleSize: 12 });
      body.push(q.svg);
      if (zi === 0) {
        // MHA：1:1 各配一对 K/V
        const kv = node({ id: `z0kv${k}`, x: qx(k), y: 150, w: 36, h: 30, role: 'cache', title: `k${SUBS[k]}v${SUBS[k]}`, mono: true, titleSize: 10.5 });
        body.push(kv.svg);
        body.push(edge(q.bottom, kv.top, { pulse: true, dur: 1.8, delay: k * 0.25, uid: UID, eid: `m${k}` }));
      } else {
        // GQA / MQA：扇入共享的一份 K/V
        body.push(edge(q.bottom, [z.x + 108, 150], { pulse: true, dur: 1.8, delay: 0.3 + k * 0.25, uid: UID, eid: `z${zi}f${k}` }));
      }
    }
    if (zi === 1) {
      const kv = node({ id: 'z1kv', x: 252 + 80, y: 150, w: 56, h: 34, role: 'cache', title: 'k₁v₁', mono: true, titleSize: 11 });
      body.push(kv.svg);
      body.push(cap(252 + 148, 171, '……共 8 组', { size: 10 }));
    }
    if (zi === 2) {
      const kv = node({ id: 'z2kv', x: 488 + 72, y: 150, w: 72, h: 34, role: 'cache', title: 'k, v', mono: true, titleSize: 11 });
      body.push(kv.svg);
      body.push(cap(488 + 72, 200, '独一份', { size: 10, anchor: 'start' }));
    }
    body.push(cap(z.x + 16, 214, z.stat1, { mono: true, size: 11, color: 'var(--ink)' }));
    body.push(cap(z.x + 16, 230, z.stat2, { mono: true, size: 11, color: 'var(--seal-deep)' }));
  });
  body.push(cap(16, 262, '每 token 全层 cache（8B 形态，bf16）：MHA 512 KiB → GQA-8 128 KiB → MQA 16 KiB；满窗 131072：64 → 16 → 2 GiB', { size: 11 }));
  body.push(cap(16, 282, 'GQA 省的是 cache 与带宽（Q/O 参数不动、打分 FLOPs 照付）；Llama-3.1 8B / 70B 都把旋钮停在 H_kv = 8', { size: 11 }));
  return scene(720, 296, body, { label: 'MHA / GQA / MQA：Q 头与共享 K/V 的对应关系与 cache 账' }).svg();
}

/* ============================================================
 * 4.1 §4.2 — MLA 数据流（block 1）
 * 原文：u_t [7168] → query 支路 W^DQ → c^Q[1536] → q^C[128] / q^R[64]（经 R_t）；
 * KV 支路 W^DKV → c^KV[512] ═ cache 项 1；W^KR → R_t → k^R[64] ═ cache 项 2（全头共享）；
 * score = (⟨q^C,k^C⟩ + ⟨q^R,k^R⟩) / √192 → softmax → o。
 * ============================================================ */
function mlaFlow() {
  const UID = 'p4mla';
  const body = [pulseDefs(UID)];
  body.push(cap(24, 26, 'MLA 数据流（V3 维度）：cache 只存 c^KV 与 k^R 两份', { size: 12, color: 'var(--ink-soft)' }));

  const ut = node({ id: 'ut', x: 24, y: 170, w: 110, h: 60, role: 'data', title: 'u_t', sub: 'd = 7168', mono: true });
  const cq = node({ id: 'cq', x: 210, y: 40, w: 120, h: 48, role: 'proj', title: "c_t^Q", sub: "d_c' = 1536", mono: true });
  const qc = node({ id: 'qc', x: 420, y: 16, w: 140, h: 44, role: 'data', title: 'q^C_{t,h}', sub: '128 · W_h^UQ', mono: true, titleSize: 12.5 });
  const qr = node({ id: 'qr', x: 420, y: 76, w: 140, h: 44, role: 'data', title: 'q^R_{t,h}', sub: '64 · 经 R_t', mono: true, titleSize: 12.5 });
  const ckv = node({ id: 'ckv', x: 210, y: 190, w: 120, h: 52, role: 'cache', title: 'c_t^KV', sub: 'd_c = 512', mono: true });
  const kr = node({ id: 'kr', x: 210, y: 282, w: 140, h: 48, role: 'cache', title: 'k_t^R', sub: '64 · 全头共享', mono: true });
  const sc = node({ id: 'sc', x: 550, y: 150, w: 150, h: 76, role: 'attn', title: 'score → softmax', sub: '/ √192 · 式 (9)', mono: true, titleSize: 12 });
  const oo = node({ id: 'oo', x: 570, y: 286, w: 110, h: 48, role: 'data', title: 'o_t', sub: '式 (10)', mono: true });
  body.push(zone({ x: 180, y: 158, w: 220, h: 192, label: 'cache：576 元素 / token / 层', tone: 'seal' }));
  [ut, cq, qc, qr, ckv, kr, sc, oo].forEach((n) => body.push(n.svg));
  body.push(cap(24, 152, 'RMSNorm 后的 stream', { size: 10 }));

  // 支路边
  body.push(edge(ut.top, cq.left, { pulse: true, dur: 2.0, delay: 0, uid: UID, eid: 'a' }));
  body.push(edge(ut.right, ckv.left, { pulse: true, dur: 2.0, delay: 0.15, uid: UID, eid: 'b' }));
  body.push(edge(ut.bottom, kr.left, { pulse: true, dur: 2.0, delay: 0.3, uid: UID, eid: 'c' }));
  body.push(edge(cq.right, qc.left, { pulse: true, dur: 1.8, delay: 0.4, uid: UID, eid: 'd' }));
  body.push(edge(cq.right, qr.left, { pulse: true, dur: 1.8, delay: 0.5, uid: UID, eid: 'e' }));
  body.push(edge(ckv.right, sc.left, { pulse: true, dur: 2.2, delay: 0.7, uid: UID, eid: 'f' }));
  body.push(edge(kr.right, sc.bottom, { pulse: true, dur: 2.2, delay: 0.8, uid: UID, eid: 'g' }));
  body.push(edge(qc.right, sc.top, { pulse: true, dur: 2.2, delay: 0.6, uid: UID, eid: 'h' }));
  body.push(edge(qr.right, sc.top, { pulse: true, dur: 2.2, delay: 0.65, uid: UID, eid: 'i' }));
  body.push(edge(sc.bottom, oo.top, { pulse: true, dur: 2.0, delay: 1.0, uid: UID, eid: 'j' }));

  body.push(cap(128, 96, 'W^DQ', { mono: true, size: 10 }));
  body.push(cap(140, 186, 'W^DKV', { mono: true, size: 10 }));
  body.push(cap(118, 268, 'W^KR → R_t', { mono: true, size: 10 }));
  body.push(cap(420, 140, 'query 支路：每步现算、不缓存（2.4 §4.3）', { size: 10 }));
  body.push(cap(196, 262, 'W_h^UK / W_h^UV 挂在其上', { mono: true, size: 9.5 }));
  body.push(cap(196, 276, '训练展开 · decode 吸收（§4.3-4.4）', { size: 9.5 }));
  body.push(cap(210, 366, 'decoupled：k^C 全程不旋转 · k^R 全程不压缩——§4.1 冲突的解法', { size: 10.5 }));
  body.push(cap(24, 390, 'score =（⟨q^C, k^C⟩ + ⟨q^R, k^R⟩）/ √192：128 维 nope 位置盲 + 64 维 rope 只含 t − s（式 9）', { mono: true, size: 10.5 }));
  return scene(720, 404, body, { label: 'MLA：低秩 latent c^KV 与 decoupled k^R 双支路进 cache，score 分 nope/rope 两项' }).svg();
}

/* ============================================================
 * 4.1 §5.3 — FlashAttention tiling（block 2）
 * 原文：Q 切 T/B_r 行块、K,V 切 T/B_c 列块；外循环 j 载 K_j,V_j 驻留 SRAM；
 * 内循环 i 载 Q_i，S_ij = Q_iK_j^T 片上不落 HBM，按式 (15)(16) 更新 (m_i,l_i,O_i)，
 * 内循环末 O_i ← O_i/l_i 写回（每块仅此一次）。
 * ============================================================ */
function flashTiling() {
  const UID = 'p4fa';
  const body = [pulseDefs(UID)];
  body.push(zone({ x: 16, y: 30, w: 200, h: 300, label: 'HBM' }));
  body.push(zone({ x: 256, y: 30, w: 448, h: 300, tone: 'seal' }));
  body.push(cap(690, 318, 'SRAM（片上 · 每 SM 192 KB）', { mono: true, size: 10, anchor: 'end' }));

  body.push(cap(32, 72, 'Q 行块', { mono: true, size: 10 }));
  const qBlocks = [];
  for (let i = 0; i < 4; i++) {
    const q = node({ id: `Q${i}`, x: 32, y: 76 + i * 34, w: 54, h: 28, role: 'data', title: `Q${SUBS[i]}`, mono: true, titleSize: 11 });
    qBlocks.push(q); body.push(q.svg);
  }
  body.push(cap(104, 72, 'K,V 列块', { mono: true, size: 10 }));
  for (let j = 0; j < 3; j++) {
    const k = node({ id: `K${j}`, x: 104, y: 104 + j * 42, w: 32, h: 28, role: 'cache', title: `K${SUBS[j]}`, mono: true, titleSize: 11 });
    const v = node({ id: `V${j}`, x: 142, y: 104 + j * 42, w: 32, h: 28, role: 'cache', title: `V${SUBS[j]}`, mono: true, titleSize: 11 });
    body.push(k.svg, v.svg);
  }
  const oi = node({ id: 'Oi', x: 32, y: 240, w: 88, h: 32, role: 'data', title: 'O_i', mono: true, titleSize: 11.5 });
  body.push(oi.svg);
  body.push(cap(32, 292, '内循环末写回：O_i ← O_i/l_i', { size: 10 }));
  body.push(cap(32, 308, '每块仅此一次', { size: 10 }));

  // SRAM 内：左列 K_j / V_j（外循环驻留），中上 Q_i（内循环流入），右下 S_ij 与累加器
  const kj = node({ id: 'KjS', x: 290, y: 54, w: 76, h: 32, role: 'cache', title: 'K_j', mono: true, titleSize: 12 });
  const vj = node({ id: 'VjS', x: 290, y: 94, w: 76, h: 32, role: 'cache', title: 'V_j', mono: true, titleSize: 12 });
  const qi = node({ id: 'QiS', x: 420, y: 54, w: 100, h: 40, role: 'data', title: 'Q_i', sub: 'B_r×d', mono: true, titleSize: 12.5 });
  const sij = node({ id: 'Sij', x: 420, y: 140, w: 130, h: 48, role: 'attn', title: 'S_ij = Q_iK_jᵀ', sub: '片上·不落 HBM', mono: true, titleSize: 11 });
  const acc = node({ id: 'acc', x: 380, y: 230, w: 190, h: 56, role: 'op', title: '(m_i, l_i, O_i)', sub: '式 (15)(16) 重缩放', mono: true, titleSize: 12 });
  [kj, vj, qi, sij, acc].forEach((n) => body.push(n.svg));

  // 装载通道（分道避让）：② Q 从顶 lane 流入，① K/V 低 lane 直入左列
  const qLoad = 'M 86 90 C 170 82, 290 38, 420 60';
  const kLoad = 'M 120 104 C 180 104, 240 76, 290 70';
  const vLoad = 'M 158 104 C 200 106, 250 110, 290 110';
  body.push(`<path class="dia-edge" d="${qLoad}"/>`);
  body.push(`<path class="dia-edge" d="${kLoad}"/>`);
  body.push(`<path class="dia-edge" d="${vLoad}"/>`);
  body.push(pathPulse(qLoad, { dur: 2.2, delay: 0, uid: UID }));
  body.push(pathPulse(kLoad, { dur: 2.4, delay: 0.2, uid: UID }));
  body.push(pathPulse(vLoad, { dur: 2.4, delay: 0.35, uid: UID }));
  body.push(cap(196, 50, '②', { mono: true, size: 12, color: 'var(--seal)' }));
  body.push(cap(204, 124, '①', { mono: true, size: 12, color: 'var(--seal)' }));

  body.push(edge(kj.right, [460, 140], { pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 'ks' }));
  body.push(edge(qi.bottom, [490, 140], { pulse: true, dur: 1.6, delay: 0.6, uid: UID, eid: 'qs' }));
  const vPath = 'M 328 126 C 340 175, 390 215, 430 230';
  body.push(`<path class="dia-edge" d="${vPath}"/>`);
  body.push(pathPulse(vPath, { dur: 2.0, delay: 0.85, uid: UID }));
  body.push(edge(sij.bottom, [475, 230], { pulse: true, dur: 1.6, delay: 1.05, uid: UID, eid: 'sa' }));
  body.push(edge(acc.left, oi.right, { pulse: true, dur: 2.4, delay: 1.3, uid: UID, eid: 'wb', strong: true }));
  body.push(cap(475, 308, 'S, P 从生到死都在 SRAM', { size: 10.5, anchor: 'middle' }));

  body.push(cap(24, 350, '① 外循环 j = 1..T/B_c：K_j, V_j 载入 SRAM，驻留本轮', { size: 11 }));
  body.push(cap(24, 370, '② 内循环 i = 1..T/B_r：Q_i 携 (m_i, l_i, O_i) 逐块更新；Q 全程重读约 42 趟（T = 16384、B_c ≈ 390）', { size: 11 }));
  body.push(cap(24, 394, 'causal：对角线右上整块跳过（不算也不读），工作量与 IO 同步减半；前向额外内存 O(T)（每行两个标量 m_i, l_i）', { size: 11 }));
  body.push(cap(24, 414, 'HBM 流量 O(T²d²/M)（式 17）vs naive Θ(T²)：d = 64、M ≈ 10⁵ 时约 1/25 —— 精确等价，一个数都不改', { size: 11, color: 'var(--ink-soft)' }));
  return scene(720, 430, body, { label: 'FlashAttention：两趟循环把 scores 留在片上，HBM 只读写 Q,K,V,O' }).svg();
}

/* ============================================================
 * 4.2 §2 — SWA 窗口 pattern 与逐层感受野（block 0）
 * 原文：T = 8、w = 3 的 band matrix；第 1 层位置 8 看到 {6,7,8}，
 * 第 2 层经位置 6 间接覆盖到 4；ρ_L(t) = t − L(w−1) ⇒ R = L(w−1)+1 ≈ L·w（式 2/3）。
 * Mistral-7B：L = 32、w = 4096 ⇒ R = 131041 ≈ 131K。
 * ============================================================ */
function swaField() {
  const UID = 'p4swa';
  const body = [pulseDefs(UID)];
  body.push(cap(40, 32, '滑窗可见集 V(i) = {i−w+1 .. i}（式 1，w = 3）× 逐层扩散：每多一层，最远可见左移 w−1 = 2', { size: 12, color: 'var(--ink-soft)' }));

  // 左：8×8 band pattern（行 = query，列 = key）
  const mx = 68, my = 98, cell = 20, gap = 3, N = 8;
  for (let j = 0; j < N; j++) body.push(cap(mx + j * (cell + gap) + cell / 2, 90, String(j + 1), { mono: true, size: 9.5, anchor: 'middle' }));
  for (let i = 0; i < N; i++) body.push(cap(mx - 10, my + i * (cell + gap) + 15, String(i + 1), { mono: true, size: 9.5, anchor: 'end' }));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const on = j <= i && i - j < 3;
      body.push(
        `<rect class="${on ? 'dia-mx-on sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${((i - j) * 0.08 + i * 0.05).toFixed(2)}"` : ''} x="${mx + j * (cell + gap)}" y="${my + i * (cell + gap)}" width="${cell}" height="${cell}" rx="3"/>`
      );
    }
  }
  body.push(cap(mx, my + N * (cell + gap) + 16, '行 = query · 列 = key（实格可见）', { size: 10 }));

  // 右：逐层覆盖（layer 1/2/3 的热格逐层左扩）
  const rx = 310, rw = 38, rgap = 8, step = rw + rgap;
  body.push(cap(651, 64, 'query t = 8', { size: 10.5, anchor: 'middle' }));
  body.push(`<polygon points="651,70 645,80 657,80" fill="var(--seal)"/>`);
  for (let k = 0; k < N; k++) body.push(cap(rx + k * step + rw / 2, 92, String(k + 1), { mono: true, size: 9.5, anchor: 'middle' }));
  const cover = [5, 3, 1]; // layer 1/2/3 的最早可见 idx（0基）
  for (let l = 0; l < 3; l++) {
    const y = 100 + l * 40;
    body.push(cap(298, y + 17, `第 ${l + 1} 层`, { size: 10.5, anchor: 'end' }));
    for (let k = 0; k < N; k++) {
      const on = k >= cover[l];
      body.push(
        `<rect class="${on ? 'dia-mx-hot sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${(0.4 + l * 0.5).toFixed(1)}"` : ''} x="${rx + k * step}" y="${y}" width="${rw}" height="26" rx="5"/>`
      );
    }
  }
  body.push(cap(rx, 240, '第 ℓ 层读 [t−w+1, t] 的输出 ⇒ ρ_ℓ(t) = ρ_{ℓ−1}(t−w+1)（式 2）', { size: 10.5 }));

  body.push(cap(40, 318, 'ρ_L(t) = t − L(w−1) ⇒ R = L(w−1) + 1 ≈ L·w（式 3）；Mistral-7B：L = 32、w = 4096 ⇒ R = 131041 ≈ 131K', { mono: true, size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(40, 338, '距离超过 L(w−1) 的依赖在计算图里不存在——不是近似误差，是没有边；SWA 承诺「局部精确、长程靠层堆叠稀释」', { size: 11 }));
  return scene(720, 354, body, { label: 'SWA：w=3 的窗口 pattern 与三层堆叠下位置 8 的感受野左扩' }).svg();
}

/* ============================================================
 * 4.2 §5 — NSA 三分支 + 门控合并（block 1）
 * 原文：compression（块 MLP 压缩，l=32 步长 d=16 → t/d 压缩 key）、
 * selection（块分数 = 压缩支路权重聚合 → top-n=16 块，l'=64，原始 K/V）、
 * sliding（最近 w=512 原始 K/V）；o = g^cmp·o^cmp + g^slc·o^slc + g^win·o^win（式 11）。
 * ============================================================ */
function nsaBranches() {
  const UID = 'p4nsa';
  const body = [pulseDefs(UID)];
  const q = node({ id: 'qt', x: 310, y: 24, w: 100, h: 40, role: 'data', title: 'query q_t', mono: true, titleSize: 12.5 });
  body.push(q.svg);

  const zs = [
    { x: 16, label: 'compression（全局粗粒度）', mech: { id: 'c1', title: 'MLP 块压缩', sub: 'l=32 步长 d=16' }, out: { id: 'o1', title: 'o^cmp', sub: '全局粗粒度' } },
    { x: 252, label: 'selection（选择细粒度）', mech: { id: 's1', title: 'top-n 块选择', sub: "l' = 64 · n = 16" }, out: { id: 'o2', title: 'o^slc', sub: '选择细粒度' } },
    { x: 488, label: 'sliding（局部细粒度）', mech: { id: 'w1', title: '滑窗', sub: 'w = 512 原始 K/V' }, out: { id: 'o3', title: 'o^win', sub: '局部细粒度' } },
  ];
  const mechs = [], outs = [];
  zs.forEach((z) => {
    body.push(zone({ x: z.x, y: 84, w: 216, h: 176, label: z.label }));
    const m = node({ id: z.mech.id, x: z.x + 20, y: 112, w: 176, h: 48, role: 'proj', title: z.mech.title, sub: z.mech.sub, mono: true, titleSize: 12 });
    const o = node({ id: z.out.id, x: z.x + 48, y: 192, w: 120, h: 44, role: 'attn', title: z.out.title, sub: z.out.sub, mono: true, titleSize: 12.5 });
    mechs.push(m); outs.push(o);
    body.push(m.svg, o.svg);
  });
  mechs.forEach((m, i) => {
    body.push(edge(q.bottom, m.top, { pulse: true, dur: 2.0, delay: i * 0.2, uid: UID, eid: `q${i}` }));
    body.push(edge(m.bottom, outs[i].top, { pulse: true, dur: 1.6, delay: 0.6 + i * 0.15, uid: UID, eid: `mo${i}` }));
  });
  // selection 的块分数来自 compression 支路（式 9）
  body.push(edge(mechs[0].right, mechs[1].left, { dash: true, uid: UID, eid: 'cs' }));
  body.push(cap(244, 126, '式 9', { mono: true, size: 9.5, anchor: 'middle' }));

  const gate = node({ id: 'gate', x: 280, y: 296, w: 196, h: 48, role: 'op', title: '门控合并', sub: 'MLP + sigmoid · 逐头', titleSize: 12.5 });
  const ot = node({ id: 'ot', x: 516, y: 296, w: 90, h: 48, role: 'data', title: 'o_t', sub: '式 (11)', mono: true });
  body.push(gate.svg, ot.svg);
  const glabels = ['g^cmp', 'g^slc', 'g^win'];
  const gmids = [[176, 262], [345, 258], [500, 268]];
  outs.forEach((o, i) => {
    body.push(edge(o.bottom, gate.top, { pulse: true, dur: 1.8, delay: 1.0 + i * 0.1, uid: UID, eid: `g${i}` }));
    body.push(cap(gmids[i][0], gmids[i][1], glabels[i], { mono: true, size: 10, color: 'var(--seal-deep)' }));
  });
  body.push(edge(gate.right, ot.left, { pulse: true, dur: 1.6, delay: 1.4, uid: UID, eid: 'go' }));

  body.push(cap(16, 372, 't = 131072 时每 query key 位：8192 + 1024 + 512 ≈ 9.7K ≈ 全量 7.4%；compression 项仍随 t 线性涨（稀释，非删除）', { mono: true, size: 10.5 }));
  body.push(cap(16, 392, '三分支全可微、与主模型从头联合训练（native 之义）；selection 块长 64 对齐 tensor core，GQA 组内共享同一份块选择', { size: 10.5 }));
  return scene(720, 404, body, { label: 'NSA：compression / selection / sliding 三分支经门控合并' }).svg();
}

/* ============================================================
 * 4.2 §6 — DSA：lightning indexer → top-2048 → MLA（block 2）
 * 原文：q_t → indexer（H^I 小维度头，ReLU 打分，可 FP8）对全历史逐 token 打分 I_{t,s}（式 12）
 * → TopK（K = 2048，式 13）→ S_t；MLA cache 全量驻留（576 元素），只 gather 选中行；
 * MLA(q_t; {c^KV_s, k^R_s}_{s∈S_t}) → o_t。decode：144 MB/步 vs 全量 9.2 GB/步。
 * ============================================================ */
function dsaIndexer() {
  const UID = 'p4dsa';
  const body = [pulseDefs(UID)];
  const hot = new Set([2, 5, 9, 13]);

  const q = node({ id: 'qt', x: 30, y: 40, w: 90, h: 40, role: 'data', title: 'q_t', mono: true });
  const idx = node({ id: 'idx', x: 170, y: 28, w: 210, h: 56, role: 'proj', title: 'lightning indexer', sub: 'H^I 头 · ReLU · 可 FP8', mono: true, titleSize: 12.5 });
  body.push(q.svg, idx.svg);
  body.push(edge(q.right, idx.left, { pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'q' }));

  // 历史打分条
  body.push(cap(430, 32, '对全部历史逐 token 打分 I_{t,s}（式 12）', { mono: true, size: 10 }));
  for (let s = 0; s < 16; s++) {
    const on = hot.has(s);
    body.push(
      `<rect class="${on ? 'dia-mx-hot sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${(0.5 + s * 0.04).toFixed(2)}"` : ''} x="${430 + s * 16.5}" y="44" width="14" height="24" rx="3"/>`
    );
  }
  body.push(edge(idx.right, [430, 56], { pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 's' }));

  const topk = node({ id: 'topk', x: 560, y: 116, w: 150, h: 48, role: 'op', title: 'TopK', sub: 'K = 2048 → S_t', mono: true, titleSize: 12.5 });
  body.push(topk.svg);
  body.push(edge([640, 68], topk.top, { pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 't' }));

  body.push(zone({ x: 30, y: 216, w: 470, h: 112, label: 'MLA cache 全量驻留：c^KV [512] + k^R [64] / token / 层', tone: 'seal' }));
  for (let s = 0; s < 16; s++) {
    const on = hot.has(s);
    body.push(
      `<rect class="${on ? 'dia-mx-hot sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${(1.2 + s * 0.04).toFixed(2)}"` : ''} x="${44 + s * 25}" y="252" width="22" height="26" rx="4"/>`
    );
  }
  body.push(cap(44, 300, '全量 576 元素 / token 驻留（cache 总量不变）', { size: 10 }));

  const gatherD = 'M 635 164 C 600 196, 540 220, 486 240';
  body.push(`<path class="dia-edge dash" d="${gatherD}"/>`);
  body.push(pathPulse(gatherD, { dur: 2.2, delay: 1.1, uid: UID }));
  body.push(cap(448, 192, '只 gather 选中 2048 行', { size: 10 }));

  const mla = node({ id: 'mla', x: 530, y: 244, w: 170, h: 52, role: 'attn', title: 'MLA 主路', sub: '只在 S_t 上算', titleSize: 12.5 });
  const ot = node({ id: 'ot', x: 620, y: 316, w: 80, h: 36, role: 'data', title: 'o_t', mono: true, titleSize: 12.5 });
  body.push(mla.svg, ot.svg);
  body.push(edge([460, 265], mla.left, { pulse: true, dur: 2.0, delay: 1.5, uid: UID, eid: 'm', strong: true }));
  body.push(edge(mla.bottom, ot.top, { pulse: true, dur: 1.6, delay: 1.8, uid: UID, eid: 'o' }));

  body.push(cap(30, 368, 'decode（61 层，t = 131072）：每步主路读取 2048 行 ≈ 144 MB / 步 vs 全量 131072 行 ≈ 9.2 GB / 步（1.5625%）', { mono: true, size: 10.5 }));
  body.push(cap(30, 388, '训练 / prefill：主 attention O(T²) → O(T·K)；indexer 打分 O(T²d_I) 但 d_I 小、可 FP8 —— 减的是带宽与 FLOPs，不是存量', { size: 10.5 }));
  return scene(720, 404, body, { label: 'DSA：indexer 全历史打分取 top-2048，MLA 只在选中行上计算' }).svg();
}

/* ============================================================
 * 4.3 §3 末 — 同一状态方程的三种算图（block 0）
 * 原文：parallel（式 4，T×T 掩码核矩阵，两次大 matmul，最大对象 T×T）；
 * recurrent（式 8，t = 1→…→6 串行，每步 rank-1 写入，状态 d_v×d_k 常驻，每 token O(d_v·d_k)）；
 * chunkwise（§8，C×C matmul 块链，块间只传 d_v×d_k 状态，T/C 步串行）。T = 6、C = 3 示意。
 * ============================================================ */
function threeGraphs() {
  const UID = 'p4lin';
  const body = [pulseDefs(UID)];
  const zs = [
    { x: 16, label: 'parallel（式 4 · 短序列）' },
    { x: 252, label: 'recurrent（式 8 · decode）' },
    { x: 488, label: 'chunkwise（§8 · 训练主力）' },
  ];
  zs.forEach((z) => body.push(zone({ x: z.x, y: 48, w: 216, h: 204, label: z.label })));

  // parallel：6×6 掩码核矩阵（下三角）
  const mx = 68, my = 92, cell = 16, gap = 3;
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j < 6; j++) {
      const on = j <= i;
      body.push(
        `<rect class="${on ? 'dia-mx-on sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${((i + j) * 0.07).toFixed(2)}"` : ''} x="${mx + j * (cell + gap)}" y="${my + i * (cell + gap)}" width="${cell}" height="${cell}" rx="3"/>`
      );
    }
  }
  body.push(cap(30, 222, 'T×T 掩码核矩阵 · 两次大 matmul', { size: 10 }));
  body.push(cap(30, 238, '最大对象 T×T（2.3 §7 的账）', { size: 10 }));

  // recurrent：t1..t6 串行链 + 常驻状态卡
  for (let k = 0; k < 6; k++) {
    const tk = node({ id: `t${k}`, x: 270 + k * 31, y: 96, w: 26, h: 26, role: 'data', title: `t${k + 1}`, mono: true, titleSize: 9.5 });
    body.push(tk.svg);
    if (k > 0) body.push(edge([270 + (k - 1) * 31 + 26, 109], tk.left, { pulse: true, dur: 1.4, delay: k * 0.3, uid: UID, eid: `r${k}` }));
    if (k === 5) body.push(edge(tk.bottom, [376, 158], { pulse: true, dur: 1.4, delay: 2.0, uid: UID, eid: 'rw' }));
  }
  const st = node({ id: 'st', x: 270, y: 158, w: 180, h: 44, role: 'cache', title: 'S_t', sub: 'd_v×d_k 常驻', mono: true, titleSize: 12.5 });
  body.push(st.svg);
  body.push(cap(266, 222, '每步 rank-1 写入 v_tφ(k_t)ᵀ', { mono: true, size: 9.5 }));
  body.push(cap(266, 238, '每 token O(d_v·d_k) · 与 T 无关', { size: 10 }));

  // chunkwise：C×C 块链，块间只传状态
  const c1 = node({ id: 'c1', x: 508, y: 96, w: 74, h: 40, role: 'op', title: 'C×C', sub: 'matmul', mono: true, titleSize: 12 });
  const c2 = node({ id: 'c2', x: 606, y: 96, w: 74, h: 40, role: 'op', title: 'C×C', sub: 'matmul', mono: true, titleSize: 12 });
  body.push(c1.svg, c2.svg);
  body.push(edge(c1.right, c2.left, { pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 'cw' }));
  body.push(cap(688, 120, '…', { mono: true, size: 14 }));
  body.push(cap(508, 158, '块间只传 S（d_v×d_k）', { size: 10 }));
  body.push(cap(502, 222, '块内并行 · T/C 步串行', { size: 10 }));
  body.push(cap(502, 238, 'T = 6、chunk 长 C = 3 示意', { size: 10 }));

  body.push(cap(16, 272, '同一状态方程的三种算图：训练走 chunkwise、decode 走 recurrent——输出逐位置相等（浮点累加序除外）', { size: 11, color: 'var(--ink-soft)' }));
  return scene(720, 288, body, { label: 'linear attention 的 parallel / recurrent / chunkwise 三种算图' }).svg();
}

/* ============================================================
 * 4.4 §3 — Qwen3-Next 一个周期（block 0）
 * 原文：12 × (3 × (GDN → MoE) → 1 × (Gated Attention → MoE))；48 层 = 36 GDN + 12 full；
 * GDN 状态 32×128×128 固定（1 MiB/层）；gated attention KV 2×2×256 = 1024 元素 / token（2 KiB/层）。
 * ============================================================ */
function hybridPeriod() {
  const UID = 'p4hyb';
  const body = [pulseDefs(UID)];
  body.push(zone({ x: 16, y: 40, w: 688, h: 150, label: '一个周期（4 层，model card 布局重绘）' }));
  body.push(cap(660, 66, '× 12', { mono: true, size: 13, weight: 600, color: 'var(--ink)' }));

  const chips = [];
  for (let p = 0; p < 4; p++) {
    const x0 = 32 + p * 154;
    const isFull = p === 3;
    const mixer = node({
      id: `mx${p}`, x: x0, y: 88, w: 76, h: 44,
      role: isFull ? 'attn' : 'op',
      title: isFull ? 'Gated Attn' : 'GDN',
      sub: isFull ? 'full' : 'linear',
      mono: true, titleSize: isFull ? 11 : 12.5,
    });
    const moe = node({ id: `moe${p}`, x: x0 + 82, y: 88, w: 64, h: 44, role: 'proj', title: 'MoE', sub: 'top-10', mono: true, titleSize: 12 });
    chips.push(mixer, moe);
    body.push(mixer.svg, moe.svg);
    body.push(edge(mixer.right, moe.left, { pulse: true, dur: 1.4, delay: p * 0.5 + 0.15, uid: UID, eid: `in${p}` }));
    if (p > 0) body.push(edge(chips[(p - 1) * 2 + 1].right, mixer.left, { pulse: true, dur: 1.4, delay: p * 0.5, uid: UID, eid: `bt${p}` }));
  }
  body.push(cap(32, 158, 'GDN 层：状态 32×128×128 固定（每层 1 MiB）', { size: 10.5 }));
  body.push(cap(32, 176, 'Gated Attn 层：KV 2×2×256 = 1024 元素 / token（每层 2 KiB）', { size: 10.5 }));
  body.push(cap(360, 158, '每个 mixer 后跟 MoE（512 experts · top-10 · 1 shared）', { size: 10.5 }));
  body.push(cap(360, 176, 'gated attention 输出带 sigmoid 门（式 2）', { size: 10.5 }));

  body.push(cap(24, 214, '48 层 = 12 ×（3 GDN + 1 gated attention）；config 字段 full_attention_interval = 4', { size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(24, 234, '斜率 ÷（r+1）：r = 3 ⇒ decode 内存线性项为全 full 反事实的 1/4（式 1）——省的是斜率，不是截距', { size: 11 }));
  return scene(720, 248, body, { label: 'Qwen3-Next 层周期：三个 GDN+MoE 与一个 gated attention+MoE' }).svg();
}

/* ============================================================
 * 4.5 §6.1 — MTP 模块流（block 0）
 * 原文 shape 走读（D = 1，位置 i）：h_i [d] 与 Emb(t_{i+1}) [d] 各经 RMSNorm，
 * 拼接 [2d] → M₁ → [d] → TRM₁ → h_i¹ [d] → OutHead → P_{i+2}¹ [V]（式 19/20）。
 * ============================================================ */
function mtpFlow() {
  const UID = 'p4mtp';
  const body = [pulseDefs(UID)];
  body.push(cap(40, 22, 'DeepSeek-V3 MTP 模块（D = 1，位置 i）：预测条件始终是真实前缀——模块 k 吃 h^{k−1} 与 Emb(t_{i+k})（式 19）', { size: 11.5, color: 'var(--ink-soft)' }));

  const hi = node({ id: 'hi', x: 40, y: 34, w: 150, h: 48, role: 'data', title: 'h_i [d]', sub: 'backbone 输出', mono: true, titleSize: 12.5 });
  const em = node({ id: 'em', x: 280, y: 34, w: 190, h: 48, role: 'data', title: 'Emb(t_{i+1}) [d]', sub: '真实下一 token · 共享', mono: true, titleSize: 12 });
  const rn1 = node({ id: 'rn1', x: 40, y: 114, w: 150, h: 34, role: 'norm', title: 'RMSNorm', titleSize: 12 });
  const rn2 = node({ id: 'rn2', x: 280, y: 114, w: 190, h: 34, role: 'norm', title: 'RMSNorm', titleSize: 12 });
  const cc = node({ id: 'cc', x: 170, y: 182, w: 120, h: 44, role: 'op', title: '拼接', sub: '[2d]', mono: true, titleSize: 12.5 });
  const m1 = node({ id: 'm1', x: 330, y: 182, w: 90, h: 44, role: 'proj', title: 'M₁', sub: '2d → d', mono: true, titleSize: 12.5 });
  const trm = node({ id: 'trm', x: 460, y: 182, w: 100, h: 44, role: 'op', title: 'TRM₁', sub: '完整 block', mono: true, titleSize: 12.5 });
  const h1 = node({ id: 'h1', x: 600, y: 182, w: 90, h: 44, role: 'data', title: 'h_i¹', sub: '[d]', mono: true, titleSize: 12.5 });
  const oh = node({ id: 'oh', x: 510, y: 252, w: 100, h: 44, role: 'proj', title: 'OutHead', sub: '与主模型共享', titleSize: 12 });
  const pp = node({ id: 'pp', x: 644, y: 252, w: 70, h: 44, role: 'attn', title: 'P¹_{i+2}', sub: '[V]', mono: true, titleSize: 12 });
  [hi, em, rn1, rn2, cc, m1, trm, h1, oh, pp].forEach((n) => body.push(n.svg));

  body.push(edge(hi.bottom, rn1.top, { pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'a' }));
  body.push(edge(em.bottom, rn2.top, { pulse: true, dur: 1.4, delay: 0.1, uid: UID, eid: 'b' }));
  body.push(edge(rn1.bottom, [200, 182], { pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'c' }));
  body.push(edge(rn2.bottom, [260, 182], { pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 'd' }));
  body.push(edge(cc.right, m1.left, { pulse: true, dur: 1.4, delay: 0.55, uid: UID, eid: 'e' }));
  body.push(edge(m1.right, trm.left, { pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 'f' }));
  body.push(edge(trm.right, h1.left, { pulse: true, dur: 1.4, delay: 0.85, uid: UID, eid: 'g' }));
  body.push(edge(h1.bottom, oh.top, { pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'h' }));
  body.push(edge(oh.right, pp.left, { pulse: true, dur: 1.4, delay: 1.1, uid: UID, eid: 'i' }));

  body.push(cap(40, 330, 'L_MTP = (λ/D) Σ_k CE(P^k_{i+k+1}, t_{i+k+1})（式 21）：监督信号加密，表示被迫编码更长程的可预测结构', { mono: true, size: 10.5 }));
  body.push(cap(40, 350, 'Emb 与 OutHead 与主模型共享；推理时模块可整体丢弃恢复标准 AR，或翻作 self-speculative 草稿头（§6.2：V3 第二 token 接受率 85–90%）', { size: 10.5 }));
  return scene(720, 368, body, { label: 'MTP：h_i 与真实下一 token 的嵌入经 RMSNorm 拼接，过 M₁ 与 TRM₁ 预测次一 token' }).svg();
}

/* ============================================================
 * 4.6 §3.2 — Ring Attention（block 0）
 * 原文：设备 p 常驻 Q_p（T/P 行）与 KV_p；第 r 步用 online softmax 把本地 Q 块对当前
 * KV 块累算进 running (m, l, acc)，同时把该 KV 块发给 p+1、接收来自 p−1 的下一块；
 * P 步轮转后每个 Q 块见完全部 KV；发送/接收与下一步计算重叠。
 * ============================================================ */
function ringAttention() {
  const UID = 'p4rng';
  const body = [pulseDefs(UID)];
  const devs = [
    { x: 40, y: 40, p: 0 }, { x: 460, y: 40, p: 1 },
    { x: 460, y: 220, p: 2 }, { x: 40, y: 220, p: 3 },
  ];
  const DEV_SUBS = ['₀', '₁', '₂', '₃'];
  devs.forEach((d) => {
    body.push(zone({ x: d.x, y: d.y, w: 180, h: 100, label: `设备 ${d.p}` }));
    const q = node({ id: `q${d.p}`, x: d.x + 14, y: d.y + 24, w: 68, h: 32, role: 'data', title: `Q${DEV_SUBS[d.p]}`, mono: true, titleSize: 12 });
    const kv = node({ id: `kv${d.p}`, x: d.x + 96, y: d.y + 24, w: 68, h: 32, role: 'cache', title: `KV${DEV_SUBS[d.p]}`, mono: true, titleSize: 12 });
    body.push(q.svg, kv.svg);
    body.push(cap(d.x + 14, d.y + 84, '常驻 · T/P 行', { size: 9.5 }));
  });
  // KV 环流（顺时针）
  body.push(edge([204, 80], [460, 80], { pulse: true, dur: 2.6, delay: 0, uid: UID, eid: 'r0' }));
  body.push(edge([600, 96], [600, 244], { pulse: true, dur: 2.6, delay: 0.65, uid: UID, eid: 'r1' }));
  const ringB = 'M 566 260 C 566 276, 556 284, 536 284 L 224 284 C 210 284, 204 274, 204 264';
  body.push(`<path class="dia-edge" d="${ringB}"/>`);
  body.push(pathPulse(ringB, { dur: 2.6, delay: 1.3, uid: UID }));
  body.push(edge([160, 244], [160, 96], { pulse: true, dur: 2.6, delay: 1.95, uid: UID, eid: 'r3' }));
  body.push(cap(332, 68, 'KV 环流', { size: 10.5, anchor: 'middle' }));

  const ctr = node({ id: 'ctr', x: 245, y: 140, w: 230, h: 72, role: 'attn', title: 'block-attn(Q_p, KV_*)', sub: '+ online softmax 累算', mono: true, titleSize: 12 });
  body.push(ctr.svg);
  body.push(cap(360, 230, 'running (m, l, acc)', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(360, 246, 'P 步轮转后每个 Q 块见完全部 KV', { size: 10, anchor: 'middle' }));

  body.push(cap(24, 348, '数学逐点不变：与 4.1 FlashAttention 是同一个分块归一化归纳不变量——块内抗 HBM × 设备间抗单卡显存，正交叠加的两层 tiling', { size: 10.5 }));
  body.push(cap(24, 368, 'FLOPs 总量不变：O(T²) 拆 P 份并行；单设备内存 O(T/P) + 一块在途 KV；发送/接收与下一步计算重叠；causal 下用均衡切分', { size: 10.5 }));
  return scene(720, 382, body, { label: 'Ring Attention：Q 驻留、KV 沿设备环流转，online softmax 逐步累算' }).svg();
}

/* ============================================================
 * 4.7 §1 — residual stream 是 read-modify-write 总线（block 0）
 * 原文：x⁽⁰⁾ ═► … ═► x⁽ᴸ⁾ ═► logits；组件写入 a_ℓ, f_ℓ（只加不覆盖）；
 * logit lens：W_U x⁽¹⁾ / W_U x⁽²⁾ / … / W_U·RMSNorm(x⁽ᴸ⁾)；
 * ablation = 从和中删去一项；patching = 换成它在别的输入下的取值（§6）。
 * ============================================================ */
function streamBus() {
  const UID = 'p4bus';
  const body = [pulseDefs(UID)];
  body.push(cap(40, 26, 'logit lens（§2）：任意中间态乘同一 W_U 都是合法的词表读出（式 1）', { size: 11.5, color: 'var(--ink-soft)' }));

  // 上方：logit lens 读出（虚线 = 观测，不改 stream）
  const lens = [
    { id: 'l1', x: 150, t: 'W_U x⁽¹⁾', w: 120 },
    { id: 'l2', x: 310, t: 'W_U x⁽²⁾', w: 120 },
    { id: 'lL', x: 470, t: 'W_U·RMSNorm(x⁽ᴸ⁾)', w: 180 },
  ];
  lens.forEach((l) => {
    const n = node({ id: l.id, x: l.x, y: 44, w: l.w, h: 34, role: 'proj', title: l.t, mono: true, titleSize: 11.5 });
    body.push(n.svg);
  });

  // 主干：x⁽⁰⁾ → x⁽¹⁾ → x⁽²⁾ → … → x⁽ᴸ⁾ → logits
  const xs = [
    { id: 'x0', x: 40, t: 'x⁽⁰⁾' },
    { id: 'x1', x: 180, t: 'x⁽¹⁾' },
    { id: 'x2', x: 320, t: 'x⁽²⁾' },
    { id: 'xL', x: 500, t: 'x⁽ᴸ⁾' },
  ];
  xs.forEach((s) => body.push(node({ id: s.id, x: s.x, y: 126, w: 70, h: 44, role: 'data', title: s.t, mono: true, titleSize: 13 }).svg));
  const lg = node({ id: 'lg', x: 610, y: 126, w: 80, h: 44, role: 'proj', title: 'logits', mono: true, titleSize: 12.5 });
  body.push(lg.svg);
  body.push(cap(445, 153, '…', { mono: true, size: 16, anchor: 'middle' }));
  for (let i = 0; i < xs.length - 1; i++) {
    body.push(edge([xs[i].x + 70, 148], [xs[i + 1].x, 148], { pulse: true, dur: 1.8, delay: i * 0.3, uid: UID, eid: `b${i}`, strong: true }));
  }
  body.push(edge([570, 148], lg.left, { pulse: true, dur: 1.8, delay: 0.9, uid: UID, eid: 'bl', strong: true }));

  // lens 虚线读出
  body.push(edge([215, 126], [210, 78], { dash: true, uid: UID, eid: 'r1' }));
  body.push(edge([355, 126], [370, 78], { dash: true, uid: UID, eid: 'r2' }));
  body.push(edge([535, 126], [560, 78], { dash: true, uid: UID, eid: 'r3' }));

  // 下方：组件写入（只加不覆盖）
  const wr = [
    { id: 'a1', x: 150, t: 'a₁', to: 200 },
    { id: 'f1', x: 250, t: 'f₁', to: 280 },
    { id: 'aL', x: 450, t: 'a_L, f_L', to: 510, w: 90 },
  ];
  wr.forEach((w) => {
    body.push(node({ id: w.id, x: w.x, y: 206, w: w.w ?? 60, h: 30, role: 'op', title: w.t, mono: true, titleSize: 11.5 }).svg);
    body.push(edge([w.x + (w.w ?? 60) / 2, 206], [w.to, 170], { uid: UID, eid: `w${w.id}` }));
  });
  body.push(cap(150, 258, '写入（只加不覆盖；对冲是唯一「删除」）', { size: 10.5 }));
  body.push(cap(40, 282, 'ablation = 从和中删去一项；patching = 把一项换成它在别的输入下的取值（§6）——纯加法结构（2.1 式 2）是全章地基', { size: 11 }));
  return scene(720, 296, body, { label: 'residual stream 总线：组件只加写入，logit lens 从任意中间层读出' }).svg();
}

/* ============================================================
 * 4.7 §4 — SAE：把 stream 换成稀疏特征坐标（block 1）
 * 原文：h ∈ R^d → encoder（ReLU / TopK / 门控）→ z ∈ R^n（稀疏：L0 ≪ n）
 * → decoder D（列 = 特征方向，单位范数）→ ĥ；loss = ‖h − ĥ‖² + λ‖z‖₁（式 7）。
 * ============================================================ */
function saeFlow() {
  const UID = 'p4sae';
  const body = [pulseDefs(UID)];
  body.push(cap(30, 26, '字典学习：把 stream 向量换成稀疏特征坐标（式 6/7；n ≫ d，常见数十至数百倍）', { size: 11.5, color: 'var(--ink-soft)' }));

  const h = node({ id: 'h', x: 30, y: 90, w: 110, h: 52, role: 'data', title: 'h ∈ R^d', sub: 'stream', mono: true, titleSize: 12.5 });
  const enc = node({ id: 'enc', x: 190, y: 84, w: 160, h: 64, role: 'proj', title: 'encoder', sub: 'ReLU/TopK/门控', mono: true, titleSize: 12.5 });
  const dec = node({ id: 'dec', x: 470, y: 196, w: 170, h: 48, role: 'proj', title: 'decoder D', sub: '列 = 特征方向', titleSize: 12.5 });
  const hh = node({ id: 'hh', x: 190, y: 196, w: 110, h: 44, role: 'data', title: 'ĥ', sub: '重构', mono: true, titleSize: 12.5 });
  [h, enc, dec, hh].forEach((n) => body.push(n.svg));

  // z：稀疏特征条
  body.push(cap(390, 80, 'z ∈ R^n · 稀疏（L0 ≪ n）', { mono: true, size: 10 }));
  const hotZ = new Set([2, 7, 11]);
  for (let s = 0; s < 15; s++) {
    const on = hotZ.has(s);
    body.push(
      `<rect class="${on ? 'dia-mx-hot sig-cell' : 'dia-mx-off'}" ${on ? `style="--i:${(0.6 + s * 0.05).toFixed(2)}"` : ''} x="${390 + s * 20}" y="92" width="16" height="34" rx="4"/>`
    );
  }

  body.push(edge(h.right, enc.left, { pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'a' }));
  body.push(edge(enc.right, [390, 109], { pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'b' }));
  body.push(edge([558, 126], [555, 196], { pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 'c' }));
  body.push(edge(dec.left, hh.right, { pulse: true, dur: 1.6, delay: 1.0, uid: UID, eid: 'd' }));
  body.push(edge(hh.top, h.bottom, { dash: true, uid: UID, eid: 'e' }));
  body.push(cap(116, 172, '‖h−ĥ‖²', { mono: true, size: 10 }));

  body.push(cap(30, 268, 'loss = ‖h − ĥ‖² + λ‖z‖₁（式 7）：重构保真 × L1 压稀疏；沿 λ 扫 Pareto 前沿，按 L0 = ‖z‖₀ 归档', { mono: true, size: 10.5 }));
  body.push(cap(30, 288, 'L1 shrinkage 的修正两族：TopK（L0 ≡ k）与 JumpReLU 门控（式 8）——Gemma Scope 即 JumpReLU，宽档 16K–1M 特征', { size: 10.5 }));
  return scene(720, 302, body, { label: 'SAE：h 经 encoder 得稀疏 z，再经 decoder D 重构 ĥ' }).svg();
}

/* ============================================================
 * 4.7 §5 — IOI circuit 三类 head 接力（block 2）
 * 原文：duplicate token heads（早层）在 John 第二次出现处标记「出现过」→ 写入 stream →
 * S-inhibition heads（中层）在 END 写入抑制，使下游对 S 的 attend 下降 →
 * name mover heads（晚层 L9H6/L9H9/L10H0）从 IO（Mary）拷贝名字表征到 END，
 * 经 W_U 直接抬升 Mary 的 logit（logit lens 逐层可见）。
 * ============================================================ */
function ioiCircuit() {
  const UID = 'p4ioi';
  const body = [pulseDefs(UID)];
  body.push(cap(40, 26, '「When Mary and John went to the store, John gave a drink to ___」→ 应填 Mary（IO）', { size: 11.5, color: 'var(--ink-soft)' }));

  // 模板 token 行
  const toks = [
    { x: 40, w: 96, t: 'Mary · IO', role: 'data' },
    { x: 150, w: 90, t: 'John · S', role: 'data' },
    { x: 254, w: 44, t: '…', role: 'plain' },
    { x: 318, w: 100, t: 'John · S 再现', role: 'cache' },
    { x: 442, w: 44, t: '…', role: 'plain' },
    { x: 510, w: 110, t: 'END → ___', role: 'plain' },
  ];
  toks.forEach((t, i) => body.push(node({ id: `tk${i}`, x: t.x, y: 40, w: t.w, h: 36, role: t.role, title: t.t, titleSize: 11.5 }).svg));

  // 三类 head 接力链
  const dup = node({ id: 'dup', x: 300, y: 110, w: 260, h: 52, role: 'op', title: 'duplicate token heads · 早层', sub: '标记「此 token 出现过」', titleSize: 12 });
  const sinh = node({ id: 'sinh', x: 300, y: 184, w: 260, h: 52, role: 'op', title: 'S-inhibition heads · 中层', sub: '在 END 写入抑制信号', titleSize: 12 });
  const nm = node({ id: 'nm', x: 300, y: 258, w: 260, h: 52, role: 'attn', title: 'name mover heads · 晚层', sub: 'L9H6 / L9H9 / L10H0', mono: true, titleSize: 12 });
  const lg = node({ id: 'lg', x: 300, y: 330, w: 260, h: 44, role: 'proj', title: '经 W_U：logit(Mary) ↑', sub: 'logit(S) 被抑制 ↓', titleSize: 12 });
  [dup, sinh, nm, lg].forEach((n) => body.push(n.svg));

  body.push(edge([368, 76], [390, 110], { dash: true, uid: UID, eid: 'mk' }));
  body.push(cap(398, 100, '标记', { size: 10 }));
  body.push(edge(dup.bottom, sinh.top, { pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'ds' }));
  body.push(cap(442, 180, '写入 stream', { size: 10 }));
  body.push(edge(sinh.bottom, nm.top, { pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 'sn' }));
  body.push(edge(nm.bottom, lg.top, { pulse: true, dur: 1.6, delay: 1.1, uid: UID, eid: 'nl' }));

  // Mary（IO 源）绕行左缘拷贝到 name mover
  const cp = 'M 88 76 C 30 120, 26 250, 296 284';
  body.push(`<path class="dia-edge dash" d="${cp}"/>`);
  body.push(pathPulse(cp, { dur: 2.4, delay: 0.9, uid: UID }));
  body.push(cap(44, 176, '拷贝名字表征', { size: 10 }));

  body.push(cap(575, 126, 'LD = logit(IO) − logit(S)', { mono: true, size: 9.5 }));
  body.push(cap(575, 150, '26 heads / 7 类角色', { size: 10 }));
  body.push(cap(575, 170, 'GPT-2 small · 144 heads', { size: 10 }));

  body.push(cap(40, 398, '另有 previous token / induction heads 供流位置与重复信息（3.3 §6 的回路在此复用为原料）', { size: 10.5 }));
  body.push(cap(40, 418, 'backup name movers：主组件被 knockout 后补上——「ablation 后行为仍在」≠ 该组件无功能（§5）', { size: 10.5 }));
  return scene(720, 432, body, { label: 'IOI circuit：duplicate token → S-inhibition → name mover 三类 head 接力' }).svg();
}

/* ============================================================
 * 4.8 §1 — masked diffusion：前向逐 token 腐蚀 × 反向联合去掩码（block 0）
 * 原文：t: 0→1 逐位置独立以概率 1−α_t 替换为 [M]（式 1，吸收态）；
 * 反向 f_θ 对全部 [M] 联合预测 → 按调度提交一个子集 → t 减小（式 2/3）；
 * 同步提交的位置按条件独立处理 ⟺ 对精确联合后验的近似：并行度 = 质量-成本旋钮。
 * ============================================================ */
function diffusionMask() {
  const UID = 'p4dif';
  const body = [pulseDefs(UID)];
  body.push(cap(96, 36, '前向（式 1）：t: 0 → 1，逐位置独立腐蚀为吸收态 [M]——一旦掩码永不返回', { size: 11.5, color: 'var(--ink-soft)' }));

  const tokens = ['天', '气', '很', '好', '我', '们', '出', '去', '玩'];
  const rows = [
    { t: 't = 0.0', y: 56, masked: new Set(), fresh: new Set() },
    { t: 't = 0.3', y: 116, masked: new Set([1, 4, 7]), fresh: new Set([1, 4, 7]), i: 0.4 },
    { t: 't = 0.7', y: 176, masked: new Set([0, 1, 3, 4, 6, 7]), fresh: new Set([0, 3, 6]), i: 0.9 },
    { t: 't = 1.0', y: 236, masked: new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]), fresh: new Set([2, 5, 8]), i: 1.5 },
  ];
  rows.forEach((r) => {
    body.push(cap(16, r.y + 23, r.t, { mono: true, size: 10.5 }));
    tokens.forEach((ch, k) => {
      const x = 104 + k * 60;
      if (!r.masked.has(k)) {
        body.push(node({ id: `t${r.y}k${k}`, x, y: r.y, w: 52, h: 34, role: 'data', title: ch, titleSize: 13 }).svg);
      } else if (r.fresh.has(k)) {
        body.push(
          `<g class="dia-node dia-node-plain"><rect class="sig-cell" style="--i:${r.i}" x="${x}" y="${r.y}" width="52" height="34" rx="12"/>` +
          `<text class="sig-cell-text" style="--i:${(r.i + 0.1).toFixed(1)}" font-family="IBM Plex Mono, monospace" font-size="12" x="${x + 26}" y="${r.y + 22}" fill="var(--ink-faint)" text-anchor="middle">[M]</text></g>`
        );
      } else {
        body.push(
          `<g class="dia-node dia-node-plain"><rect x="${x}" y="${r.y}" width="52" height="34" rx="12" style="fill:var(--paper-sunk)"/>` +
          `<text font-family="IBM Plex Mono, monospace" font-size="12" x="${x + 26}" y="${r.y + 22}" fill="var(--ink-faint)" text-anchor="middle">[M]</text></g>`
        );
      }
    });
  });

  // 前向腐蚀方向（左缘下行虚线流）
  body.push(`<line class="dia-edge sig-dashflow" x1="88" y1="66" x2="88" y2="250"/>`);
  body.push(`<polygon points="88,258 82.5,248 93.5,248" fill="var(--ink-soft)"/>`);
  // 反向去掩码（右缘上行虚线流）
  body.push(`<line class="dia-edge sig-dashflow" x1="672" y1="252" x2="672" y2="68"/>`);
  body.push(`<polygon points="672,60 666.5,70 677.5,70" fill="var(--seal)"/>`);
  body.push(`<text class="dia-cap" font-size="11" transform="rotate(-90 694 160)" x="694" y="160" text-anchor="middle" fill="var(--ink-faint)">反向：联合去掩码（式 2）</text>`);

  const ft = node({ id: 'ft', x: 270, y: 292, w: 220, h: 48, role: 'attn', title: 'f_θ：双向 Transformer', sub: '对全部 [M] 位置联合预测', titleSize: 12 });
  body.push(ft.svg);
  body.push(edge([380, 270], ft.top, { pulse: true, dur: 1.8, delay: 1.8, uid: UID, eid: 'f' }));
  body.push(edge(ft.right, [668, 262], { pulse: true, dur: 1.8, delay: 2.0, uid: UID, eid: 'r', dash: true }));

  body.push(cap(96, 366, '按调度提交一个子集 → t 减小；同步提交的位置按条件独立处理 ⟺ 对精确联合后验的近似——并行度本身就是质量-成本旋钮', { size: 10.5 }));
  body.push(cap(96, 386, '训练：L = E_{t∼U(0,1)} (1/t) Σ_{i : 被掩} CE(x_0^i, p̃_θ)（式 3）——掩码率本身是被积变量', { mono: true, size: 10 }));
  return scene(720, 400, body, { label: 'masked diffusion：四档掩码率下 [M] 逐步吞掉序列，f_θ 反向联合去掩码' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '04-modern-architectures/01-kv-compression-and-exact-attention', replacesBlock: 0, title: 'MHA / GQA / MQA：谁和谁共享 K/V', svg: mhaGqaMqa },
  { slug: '04-modern-architectures/01-kv-compression-and-exact-attention', replacesBlock: 1, title: 'MLA：低秩 latent + decoupled RoPE 数据流', svg: mlaFlow },
  { slug: '04-modern-architectures/01-kv-compression-and-exact-attention', replacesBlock: 2, title: 'FlashAttention tiling：scores 永不落 HBM', svg: flashTiling },
  { slug: '04-modern-architectures/02-sparse-attention', replacesBlock: 0, title: 'SWA 窗口 pattern 与逐层感受野', svg: swaField },
  { slug: '04-modern-architectures/02-sparse-attention', replacesBlock: 1, title: 'NSA：三分支可见集与门控合并', svg: nsaBranches },
  { slug: '04-modern-architectures/02-sparse-attention', replacesBlock: 2, title: 'DSA：lightning indexer → top-2048 → MLA', svg: dsaIndexer },
  { slug: '04-modern-architectures/03-linear-attention-and-ssm', replacesBlock: 0, title: '同一状态方程的三种算图', svg: threeGraphs },
  { slug: '04-modern-architectures/04-hybrid-architectures', replacesBlock: 0, title: 'Qwen3-Next：3 GDN + 1 gated attention 周期', svg: hybridPeriod },
  { slug: '04-modern-architectures/05-moe', replacesBlock: 0, title: 'MTP 模块流（D = 1）', svg: mtpFlow },
  { slug: '04-modern-architectures/06-long-context', replacesBlock: 0, title: 'Ring Attention：KV 环流、Q 驻留', svg: ringAttention },
  { slug: '04-modern-architectures/07-interpretability-primer', replacesBlock: 0, title: 'residual stream 总线：可读、可删、可换', svg: streamBus },
  { slug: '04-modern-architectures/07-interpretability-primer', replacesBlock: 1, title: 'SAE：h → 稀疏特征 z → 重构 ĥ', svg: saeFlow },
  { slug: '04-modern-architectures/07-interpretability-primer', replacesBlock: 2, title: 'IOI circuit：三类 head 的接力', svg: ioiCircuit },
  { slug: '04-modern-architectures/08-frontier-outlook', replacesBlock: 0, title: 'masked diffusion：前向腐蚀 × 反向去掩码', svg: diffusionMask },
];
