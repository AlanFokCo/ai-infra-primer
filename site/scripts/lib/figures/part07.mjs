// Part 7（扩散与图像生成）签名动图：噪声日程 / SDE↔ODE 轨迹 / CFG 几何 / 路径几何 / MMDiT 双流
// 设计语言与 figures-signature.mjs 一致：纸面板、发丝墨线、mono 标注、朱砂数据点。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='13'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

const svgOpen = (w, h, label) =>
  `<svg class="fig-svg sig" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;

/** 实心小三角箭头：尖端 (x,y)，运动方向单位向量 (ux,uy)，尺寸 s */
const ah = (x, y, ux, uy, s, fill) => {
  const px = -uy, py = ux;
  const bx = x - s * ux, by = y - s * uy;
  const w = s * 0.42;
  return (
    `<path d='M ${x.toFixed(1)} ${y.toFixed(1)} L ${(bx + w * px).toFixed(1)} ${(by + w * py).toFixed(1)} ` +
    `L ${(bx - w * px).toFixed(1)} ${(by - w * py).toFixed(1)} Z' fill='${fill}'/>`
  );
};

/** 确定性伪随机（mulberry32）：同一 seed 每次生成同一批散点 */
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 椭圆散点云 */
function cloud(cx, cy, n, rx, ry, rng, r = 1.7, op = 0.55) {
  let s = '';
  for (let i = 0; i < n; i++) {
    const a = rng() * 2 * Math.PI;
    const rad = Math.sqrt(rng());
    s += `<circle cx='${(cx + Math.cos(a) * rad * rx).toFixed(1)}' cy='${(cy + Math.sin(a) * rad * ry).toFixed(1)}' r='${r}' fill='var(--ink-faint)' opacity='${op}'/>`;
  }
  return s;
}

/* ============ 7.2 §2.3 — DDPM 噪声日程：x_0 随 t 被噪声淹没 ============ */
function ddpmSchedule() {
  const w = 720, h = 340;
  const parts = [svgOpen(w, h, 'DDPM 噪声日程：同一 x_0 随 t 增大被噪声逐层淹没，SNR 单调下降')];
  parts.push(
    `<text ${MONO_S} x='24' y='22' fill='var(--ink-soft)'>x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε（式 6）：同一 x₀ 随 t 被噪声淹没 — 每格 = 信号层 ×√ᾱ_t ＋ 噪声层 ×√(1−ᾱ_t)</text>`
  );

  // 一排小方块：linear schedule 的 7 个采样时刻（数值取自 §2.3 表）
  const T = [1, 100, 200, 500, 700, 900, 1000];
  const ABAR = [0.9999, 0.897, 0.659, 0.0786, 0.00697, 2.75e-4, 4.04e-5];
  const ABAR_TXT = ['0.9999', '0.897', '0.659', '0.0786', '0.00697', '2.8e−4', '4.0e−5'];
  const SNR_TXT = ['+40.0', '+9.4', '+2.9', '−10.7', '−21.5', '−35.6', '−43.9'];
  const NOISE_PAL = ['#f7f6f2', '#e7e5db', '#d9d6cc', '#c9c4b8', '#a9a498', '#8b867a', '#636a78', '#454b57'];
  const G = 8, CELL = 6.75, TILE = 54, TY = 40;

  // 行标签
  parts.push(`<text ${MONO_S} x='24' y='110' fill='var(--ink-faint)'>t</text>`);
  parts.push(`<text ${MONO_S} x='24' y='126' fill='var(--ink-faint)'>ᾱ_t</text>`);
  parts.push(`<text ${MONO_S} x='24' y='142' fill='var(--ink-faint)'>SNR_dB</text>`);

  T.forEach((t, i) => {
    const x0 = 78 + i * 90;
    const rng = mulberry(1000 + t);
    // 底层
    parts.push(`<rect x='${x0}' y='${TY}' width='${TILE}' height='${TILE}' fill='var(--paper)'/>`);
    // 信号层（8×8 盘形图案，opacity = √ᾱ_t）
    const sigOp = Math.sqrt(ABAR[i]).toFixed(4);
    let sig = `<g opacity='${sigOp}'>`;
    for (let r = 0; r < G; r++) {
      for (let c = 0; c < G; c++) {
        const d2 = (r - 3.5) ** 2 + (c - 3.5) ** 2;
        if (d2 > 9.5) continue;
        const fill = d2 <= 2.3 ? '#181b20' : d2 <= 5.5 ? '#454b57' : '#a9a498';
        sig += `<rect x='${(x0 + c * CELL).toFixed(2)}' y='${(TY + r * CELL).toFixed(2)}' width='${CELL}' height='${CELL}' fill='${fill}'/>`;
      }
    }
    parts.push(sig + '</g>');
    // 噪声层（白噪点，opacity = √(1−ᾱ_t)）
    const noiOp = Math.sqrt(1 - ABAR[i]).toFixed(4);
    let noi = `<g opacity='${noiOp}'>`;
    for (let r = 0; r < G; r++) {
      for (let c = 0; c < G; c++) {
        const fill = NOISE_PAL[Math.floor(rng() * NOISE_PAL.length)];
        noi += `<rect x='${(x0 + c * CELL).toFixed(2)}' y='${(TY + r * CELL).toFixed(2)}' width='${CELL}' height='${CELL}' fill='${fill}'/>`;
      }
    }
    parts.push(noi + '</g>');
    parts.push(`<rect x='${x0}' y='${TY}' width='${TILE}' height='${TILE}' fill='none' stroke='var(--rule)'/>`);
    // 三行注记
    const cx = x0 + TILE / 2;
    parts.push(`<text ${MONO_S} x='${cx}' y='110' fill='var(--ink)' text-anchor='middle'>t=${t}</text>`);
    parts.push(`<text ${MONO_S} x='${cx}' y='126' fill='var(--ink-soft)' text-anchor='middle'>${ABAR_TXT[i]}</text>`);
    const snrFill = SNR_TXT[i].startsWith('−') ? 'var(--seal-deep)' : 'var(--ink-soft)';
    parts.push(`<text ${MONO_S} x='${cx}' y='142' fill='${snrFill}' text-anchor='middle'>${SNR_TXT[i]}</text>`);
  });

  // t 轴 + 流光点（前向 = 加噪方向）
  parts.push(`<line x1='78' y1='160' x2='664' y2='160' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(ah(672, 160, 1, 0, 7, 'var(--ink-soft)'));
  parts.push(`<text ${MONO_S} x='672' y='176' fill='var(--ink-faint)' text-anchor='end'>t ↑（前向 = 加噪）</text>`);
  parts.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'>` +
      `<animateMotion dur='7s' repeatCount='indefinite' path='M 78 160 L 664 160'/></circle>`
  );

  // SNR_dB 曲线（linear vs cosine；坐标：t∈[0,1000] → x 78..672；dB∈[+45,−45] → y 192..300）
  const px = (t) => 78 + 0.594 * t;
  const py = (db) => 192 + (45 - db) * 1.2;
  parts.push(`<text ${MONO_S} x='78' y='186' fill='var(--ink-faint)'>SNR_dB = 10·log₁₀(ᾱ_t/(1−ᾱ_t))（式 9）</text>`);
  parts.push(`<line x1='78' y1='192' x2='78' y2='300' stroke='var(--rule)'/>`);
  parts.push(`<line x1='78' y1='300' x2='672' y2='300' stroke='var(--rule)'/>`);
  parts.push(`<line x1='78' y1='${py(0)}' x2='672' y2='${py(0)}' stroke='var(--rule)' stroke-dasharray='3 3'/>`);
  parts.push(`<text ${MONO_S} x='72' y='201' fill='var(--ink-faint)' text-anchor='end'>+40</text>`);
  parts.push(`<text ${MONO_S} x='72' y='249' fill='var(--ink-faint)' text-anchor='end'>0</text>`);
  parts.push(`<text ${MONO_S} x='72' y='297' fill='var(--ink-faint)' text-anchor='end'>−40</text>`);

  const LIN = [[1, 40], [100, 9.4], [200, 2.9], [500, -10.7], [700, -21.5], [800, -28.1], [900, -35.6], [1000, -43.9]];
  const COS = [[1, 43.8], [100, 15.4], [200, 9.5], [500, -0.1], [700, -5.9], [800, -9.8], [900, -16.1]];
  const linPts = LIN.map(([t, db]) => `${px(t).toFixed(1)} ${py(db).toFixed(1)}`);
  const cosPts = COS.map(([t, db]) => `${px(t).toFixed(1)} ${py(db).toFixed(1)}`);
  // cosine 末端穿出图框底边（SNR → −∞）
  parts.push(`<polyline points='${cosPts.join(' ')} 674 301.5' fill='none' stroke='var(--ink-soft)' stroke-width='1.6'/>`);
  parts.push(ah(679.5, 306, 0.86, 0.51, 7, 'var(--ink-soft)'));
  parts.push(`<polyline points='${linPts.join(' ')}' fill='none' stroke='var(--seal)' stroke-width='1.6'/>`);
  // linear 曲线上与方块对应的 7 个采样点
  LIN.forEach(([t, db], i) => {
    if (t === 800) return;
    parts.push(
      `<circle class='sig-step' style='--i:${(i * 0.4).toFixed(1)}' cx='${px(t).toFixed(1)}' cy='${py(db).toFixed(1)}' r='2.4' fill='var(--seal)'/>`
    );
  });
  // 流光点沿 linear SNR 曲线下行（与 t 轴光点同周期）
  parts.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'>` +
      `<animateMotion dur='7s' repeatCount='indefinite' path='M ${linPts.join(' L ')}'/></circle>`
  );

  // 图例与 x 轴
  parts.push(`<line x1='400' y1='306' x2='422' y2='306' stroke='var(--seal)' stroke-width='1.6'/>`);
  parts.push(`<text ${MONO_S} x='426' y='310' fill='var(--ink-soft)'>linear（β: 1e−4→0.02，式 7）</text>`);
  parts.push(`<line x1='592' y1='306' x2='614' y2='306' stroke='var(--ink-soft)' stroke-width='1.6'/>`);
  parts.push(`<text ${MONO_S} x='618' y='310' fill='var(--ink-soft)'>cosine（式 8）</text>`);
  parts.push(`<text ${MONO_S} x='78' y='310' fill='var(--ink-faint)'>t（T = 1000）→</text>`);

  parts.push(
    `<text ${CJK_S} x='24' y='328' fill='var(--ink-faint)'>√ᾱ_T = 0.0064 ＜ 量化台阶 1/127.5 ≈ 0.0078 ⇒ linear 末 22 步浪费；cosine 恰在 t = T 归零（仅末 6 步低于台阶）、半程 SNR ≈ 0 dB（§2.3 的账）</text>`
  );
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 7.3 §6.2 — 逆时 SDE 轨迹族 vs PF-ODE 确定性轨迹 ============ */
function sdeOdeTrajectories() {
  const w = 720, h = 356;
  const parts = [svgOpen(w, h, '逆时 SDE 的随机轨迹族与 PF-ODE 的确定性双射共享同一边缘密度族')];

  // SDE 随机路径生成器：同一初值可发散到不同 mode
  function sdePath(sx, sy, ex, ey, rng) {
    const xs = [150, 240, 330, 420, 510, 575];
    let d = `M ${sx} ${sy}`;
    for (const x of xs) {
      const p = (x - sx) / (ex - sx);
      const amp = 26 * Math.sin(Math.PI * p) + 6;
      const y = sy + (ey - sy) * p + (rng() * 2 - 1) * amp;
      d += ` L ${x} ${y.toFixed(1)}`;
    }
    return d + ` L ${ex} ${ey}`;
  }

  // ---- 上幅：逆时 SDE ----
  parts.push(
    `<text ${MONO_S} x='24' y='26' fill='var(--ink-soft)'>逆时 SDE（式 17）：dx = [f − g²·s_θ] dt + g·dW̄ — 每步注噪、轨迹随机（同一初值可落不同 mode）</text>`
  );
  for (const sx of [240, 380, 510]) {
    parts.push(`<line x1='${sx}' y1='40' x2='${sx}' y2='158' stroke='var(--rule)' stroke-dasharray='2 3'/>`);
  }
  // 端点云
  parts.push(cloud(80, 98, 12, 24, 24, mulberry(7)));
  parts.push(cloud(630, 74, 7, 11, 8, mulberry(8)));
  parts.push(cloud(630, 124, 7, 11, 8, mulberry(9)));
  // 中间时刻边缘密度点云（与下幅逐点相同）
  parts.push(cloud(240, 98, 9, 30, 16, mulberry(21), 1.6, 0.5));
  parts.push(cloud(380, 88, 4, 14, 6, mulberry(22), 1.6, 0.5));
  parts.push(cloud(380, 110, 4, 14, 6, mulberry(23), 1.6, 0.5));
  parts.push(cloud(510, 80, 3, 10, 5, mulberry(24), 1.6, 0.5));
  parts.push(cloud(510, 116, 3, 10, 5, mulberry(25), 1.6, 0.5));
  // 普通随机路径 ×4
  const plains = [
    sdePath(80, 88, 630, 74, mulberry(31)),
    sdePath(80, 88, 630, 136, mulberry(32)),
    sdePath(80, 108, 630, 66, mulberry(33)),
    sdePath(80, 108, 630, 120, mulberry(34)),
  ];
  for (const d of plains) {
    parts.push(`<path d='${d}' fill='none' stroke='#a9a498' stroke-width='1.1' stroke-linecap='round'/>`);
  }
  // 高亮：同一初值 S2 的两条发散路径
  const hp1 = sdePath(80, 98, 630, 82, mulberry(41));
  const hp2 = sdePath(80, 98, 630, 128, mulberry(42));
  for (const d of [hp1, hp2]) {
    parts.push(`<path d='${d}' fill='none' stroke='var(--seal)' stroke-opacity='0.55' stroke-width='1.2' stroke-linecap='round'/>`);
  }
  parts.push(`<circle class='sig-diag' style='--i:0' cx='80' cy='98' r='5' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  parts.push(`<text ${MONO_S} x='110' y='88' fill='var(--seal-deep)'>同一 x(1)</text>`);
  parts.push(`<text ${MONO_S} x='80' y='152' fill='var(--ink-faint)' text-anchor='middle'>x(1) ~ N(0, I)</text>`);
  parts.push(`<text ${MONO_S} x='630' y='152' fill='var(--ink-faint)' text-anchor='middle'>x(0) ~ p_data</text>`);
  // 流光：沿两条高亮随机路径
  parts.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='4.2s' repeatCount='indefinite' path='${hp1}'/></circle>`,
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='4.2s' begin='0.9s' repeatCount='indefinite' path='${hp2}'/></circle>`
  );

  // ---- 中间注记 ----
  parts.push(
    `<text ${CJK_S} x='360' y='180' fill='var(--ink-faint)' text-anchor='middle'>同一网络 s_θ、同一边缘密度族 { p_t } — 两幅点云逐 t 相同；差别只在路径层面</text>`
  );

  // ---- 下幅：PF-ODE ----
  parts.push(
    `<text ${MONO_S} x='24' y='196' fill='var(--ink-soft)'>PF-ODE（式 20）：dx/dt = f − ½g²·s_θ — 无扩散项、x(1) ↦ x(0) 是确定性双射（轨迹互不相交）</text>`
  );
  for (const sx of [240, 380, 510]) {
    parts.push(`<line x1='${sx}' y1='210' x2='${sx}' y2='328' stroke='var(--rule)' stroke-dasharray='2 3'/>`);
  }
  parts.push(cloud(80, 268, 12, 24, 24, mulberry(7)));
  parts.push(cloud(630, 244, 7, 11, 8, mulberry(8)));
  parts.push(cloud(630, 294, 7, 11, 8, mulberry(9)));
  parts.push(cloud(240, 268, 9, 30, 16, mulberry(21), 1.6, 0.5));
  parts.push(cloud(380, 258, 4, 14, 6, mulberry(22), 1.6, 0.5));
  parts.push(cloud(380, 280, 4, 14, 6, mulberry(23), 1.6, 0.5));
  parts.push(cloud(510, 250, 3, 10, 5, mulberry(24), 1.6, 0.5));
  parts.push(cloud(510, 286, 3, 10, 5, mulberry(25), 1.6, 0.5));
  const odeStarts = [250, 258, 266, 274, 282, 290];
  const odeEnds = [236, 244, 252, 288, 294, 300];
  const odePaths = odeStarts.map(
    (s, i) => `M 80 ${s} C 250 ${s}, 390 ${odeEnds[i]}, 630 ${odeEnds[i]}`
  );
  for (const d of odePaths) {
    parts.push(`<path d='${d}' fill='none' stroke='#a9a498' stroke-width='1.1' stroke-linecap='round'/>`);
  }
  parts.push(`<text ${MONO_S} x='80' y='322' fill='var(--ink-faint)' text-anchor='middle'>x(1) ~ N(0, I)</text>`);
  parts.push(`<text ${MONO_S} x='630' y='322' fill='var(--ink-faint)' text-anchor='middle'>x(0) ~ p_data</text>`);
  parts.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='3.6s' repeatCount='indefinite' path='${odePaths[1]}'/></circle>`,
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='3.6s' begin='1.1s' repeatCount='indefinite' path='${odePaths[4]}'/></circle>`
  );

  // 底部：方向与时刻注记
  parts.push(`<text ${CJK_S} x='24' y='344' fill='var(--ink-faint)'>采样方向</text>`);
  parts.push(`<line x1='92' y1='340' x2='138' y2='340' stroke='#a9a498' stroke-width='1.2'/>`);
  parts.push(ah(146, 340, 1, 0, 6, 'var(--ink-soft)'));
  parts.push(`<text ${MONO_S} x='152' y='344' fill='var(--ink-faint)'>t: 1 → 0</text>`);
  const stripT = [[240, 't = 0.7'], [380, 't = 0.45'], [510, 't = 0.2']];
  for (const [x, s] of stripT) {
    parts.push(`<text ${MONO_S} x='${x}' y='344' fill='var(--ink-faint)' text-anchor='middle'>${s}</text>`);
  }
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 7.4 §4.2 — CFG 的向量合成几何 ============ */
function cfgGeometry() {
  const w = 720, h = 344;
  const parts = [svgOpen(w, h, 'CFG：epsilon_uncond 与 epsilon_cond 的向量合成，w 沿条件方向外推')];
  parts.push(
    `<text ${MONO_S} x='24' y='22' fill='var(--ink-soft)'>ε̃ = ε_u + w·Δ（式 18），Δ = ε_c − ε_u — 沿条件方向的外推：w = 1 是纯条件采样，w ＞ 1 越过它</text>`
  );

  const O = [110, 232], U = [296, 178], C = [364, 134];
  const D = [C[0] - U[0], C[1] - U[1]]; // (68, -44)
  const du = Math.hypot(D[0], D[1]);
  const Dn = [D[0] / du, D[1] / du];

  // ε_u、ε_c 向量（墨）
  const uLen = Math.hypot(U[0] - O[0], U[1] - O[1]);
  parts.push(
    `<line x1='${O[0]}' y1='${O[1]}' x2='${(U[0] - (6 * (U[0] - O[0])) / uLen).toFixed(1)}' y2='${(U[1] - (6 * (U[1] - O[1])) / uLen).toFixed(1)}' stroke='var(--ink)' stroke-width='1.6'/>`
  );
  parts.push(ah(U[0], U[1], (U[0] - O[0]) / uLen, (U[1] - O[1]) / uLen, 8, 'var(--ink)'));
  const cLen = Math.hypot(C[0] - O[0], C[1] - O[1]);
  parts.push(
    `<line x1='${O[0]}' y1='${O[1]}' x2='${(C[0] - (6 * (C[0] - O[0])) / cLen).toFixed(1)}' y2='${(C[1] - (6 * (C[1] - O[1])) / cLen).toFixed(1)}' stroke='var(--ink)' stroke-width='1.6'/>`
  );
  parts.push(ah(C[0], C[1], (C[0] - O[0]) / cLen, (C[1] - O[1]) / cLen, 8, 'var(--ink)'));
  parts.push(`<circle cx='${O[0]}' cy='${O[1]}' r='3' fill='var(--ink)'/>`);
  parts.push(`<text ${MONO_S} x='${O[0] - 8}' y='${O[1] + 16}' fill='var(--ink-faint)'>x_t</text>`);

  // w 射线（朱砂）：w ∈ [−0.35, 3.25]
  const P0 = [U[0] - 0.35 * D[0], U[1] - 0.35 * D[1]];
  const PE = [U[0] + 3.25 * D[0], U[1] + 3.25 * D[1]];
  parts.push(
    `<line x1='${P0[0].toFixed(1)}' y1='${P0[1].toFixed(1)}' x2='${(PE[0] - 6 * Dn[0]).toFixed(1)}' y2='${(PE[1] - 6 * Dn[1]).toFixed(1)}' stroke='var(--seal)' stroke-width='1.6'/>`
  );
  parts.push(ah(PE[0], PE[1], Dn[0], Dn[1], 8, 'var(--seal)'));
  // Δ 段（虚线强调）
  parts.push(
    `<line x1='${U[0]}' y1='${U[1]}' x2='${C[0]}' y2='${C[1]}' stroke='var(--seal-deep)' stroke-width='1.3' stroke-dasharray='4 3'/>`
  );
  // w 刻度（0/1/2/3）+ 标签
  const perp = [-Dn[1], Dn[0]];
  const ticks = [[0, 'w = 0', 270, 170, 'end'], [1, 'w = 1 · 纯条件', 378, 152, 'start'], [2, 'w = 2', 440, 104, 'start'], [3, 'w = 3', 508, 60, 'start']];
  for (const [wi, label, lx, ly, anch] of ticks) {
    const P = [U[0] + wi * D[0], U[1] + wi * D[1]];
    parts.push(
      `<line x1='${(P[0] - 5 * perp[0]).toFixed(1)}' y1='${(P[1] - 5 * perp[1]).toFixed(1)}' x2='${(P[0] + 5 * perp[0]).toFixed(1)}' y2='${(P[1] + 5 * perp[1]).toFixed(1)}' stroke='var(--seal-deep)' stroke-width='1.2'/>`
    );
    parts.push(`<text ${MONO_S} x='${lx}' y='${ly}' fill='var(--seal-deep)' text-anchor='${anch}'>${label}</text>`);
  }
  parts.push(`<text ${MONO_S} x='508' y='76' fill='var(--ink-faint)'>…w ∈ [2, 15]</text>`);

  // 合成向量 ε̃（w = 3）
  const P3 = [U[0] + 3 * D[0], U[1] + 3 * D[1]];
  const pLen = Math.hypot(P3[0] - O[0], P3[1] - O[1]);
  parts.push(
    `<line x1='${O[0]}' y1='${O[1]}' x2='${(P3[0] - (7 * (P3[0] - O[0])) / pLen).toFixed(1)}' y2='${(P3[1] - (7 * (P3[1] - O[1])) / pLen).toFixed(1)}' stroke='var(--seal)' stroke-width='2'/>`
  );
  parts.push(ah(P3[0], P3[1], (P3[0] - O[0]) / pLen, (P3[1] - O[1]) / pLen, 9, 'var(--seal)'));

  // 向量名标签
  parts.push(`<text ${MONO_S} x='292' y='208' fill='var(--ink)' text-anchor='end'>ε_u = ε_θ(∅)</text>`);
  parts.push(`<text ${MONO_S} x='372' y='114' fill='var(--ink)' text-anchor='end'>ε_c = ε_θ(c)</text>`);
  parts.push(`<text ${MONO_S} x='342' y='172' fill='var(--seal-deep)'>Δ = ε_c − ε_u</text>`);
  parts.push(`<text ${MONO_S} x='380' y='96' fill='var(--seal-deep)' text-anchor='end'>ε̃ = ε_u + w·Δ</text>`);

  // 流光点：沿 w 射线滑动（w 增大方向）
  parts.push(
    `<circle class='sig-flow' r='2.8' fill='var(--seal)'>` +
      `<animateMotion dur='3s' repeatCount='indefinite' path='M ${U[0]} ${U[1]} L ${PE[0].toFixed(1)} ${PE[1].toFixed(1)}'/></circle>`
  );

  // 分隔线
  parts.push(`<line x1='24' y1='246' x2='696' y2='246' stroke='var(--rule-soft)'/>`);

  // 左下：w 的分布语义（式 19 tilt）
  parts.push(
    `<text ${MONO_S} x='24' y='264' fill='var(--ink-faint)'>w 的分布语义（式 19）：q_w(x) ∝ p(x)·p(y|x)^w</text>`
  );
  parts.push(`<line x1='40' y1='314' x2='330' y2='314' stroke='var(--rule)'/>`);
  parts.push(
    `<path d='M 60 314 C 95 314, 105 284, 150 284 C 195 284, 205 314, 240 314' fill='var(--paper-sunk)' stroke='var(--ink-soft)'/>`
  );
  parts.push(
    `<path d='M 130 314 C 158 314, 168 278, 196 278 C 226 278, 236 314, 264 314' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`
  );
  parts.push(`<text ${MONO_S} x='108' y='306' fill='var(--ink-faint)' text-anchor='middle'>p(x)</text>`);
  parts.push(`<text ${MONO_S} x='238' y='296' fill='var(--seal-deep)'>q_w · w=7.5</text>`);
  parts.push(
    `<text ${CJK_S} x='24' y='332' fill='var(--ink-faint)'>w ＞ 1 = 比真后验更尖：对齐 ↑、多样性 ↓（§4.3）</text>`
  );

  // 右下：范数账（式 20）
  parts.push(
    `<text ${MONO_S} x='400' y='264' fill='var(--ink-faint)'>范数账（式 20）：‖ε̃‖² = ‖ε_u‖² + 2w⟨ε_u,Δ⟩ + w²‖Δ‖²</text>`
  );
  const bars = [['w = 0', 55, '×1.0'], ['w = 1', 79, '×1.4'], ['w = 3', 128, '×2.2']];
  bars.forEach(([label, len, ratio], i) => {
    const y = 276 + i * 18;
    parts.push(`<text ${MONO_S} x='446' y='${y + 10}' fill='var(--ink-soft)' text-anchor='end'>${label}</text>`);
    parts.push(
      `<rect class='sig-bar' style='--i:${i}' x='452' y='${y}' width='${len}' height='12' rx='2' fill='var(--seal-wash)' stroke='var(--seal)'/>`
    );
    parts.push(`<text ${MONO_S} x='${452 + len + 6}' y='${y + 10}' fill='var(--ink-faint)'>${ratio}</text>`);
  });
  parts.push(
    `<text ${CJK_S} x='400' y='332' fill='var(--ink-faint)'>‖ε̃‖ 随 w 膨胀 ⇒ x̂_0 出界、过饱和（§4.3）</text>`
  );
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 7.5 §7 — 路径几何：VP 圆弧 vs rectified flow 弦 ============ */
function rfVsVpPaths() {
  const w = 720, h = 300;
  const parts = [svgOpen(w, h, '同一端点对张成的平面里，VP 路径是圆弧、rectified flow 路径是弦')];
  parts.push(
    `<text ${MONO_S} x='24' y='22' fill='var(--ink-soft)'>固定端点对 (x₀, ε)：x_t = α_t·x₀ + σ_t·ε 只活在它们张成的平面 — 逐 t 坐标 (α_t, σ_t) 扫出的曲线 = 条件路径</text>`
  );

  const Ox = 140, Oy = 232, R = 176;
  const D = [Ox + R, Oy], N = [Ox, Oy - R];
  // 坐标轴
  parts.push(`<line x1='${Ox}' y1='${Oy}' x2='${Ox + R + 28}' y2='${Oy}' stroke='#a9a498' stroke-width='1.2'/>`);
  parts.push(ah(Ox + R + 34, Oy, 1, 0, 7, 'var(--ink-soft)'));
  parts.push(`<line x1='${Ox}' y1='${Oy}' x2='${Ox}' y2='${Oy - R - 16}' stroke='#a9a498' stroke-width='1.2'/>`);
  parts.push(ah(Ox, Oy - R - 22, 0, -1, 7, 'var(--ink-soft)'));
  parts.push(`<text ${MONO_S} x='210' y='252' fill='var(--ink-faint)' text-anchor='middle'>α_t · 沿 x₀ 方向</text>`);
  parts.push(`<text ${MONO_S} x='148' y='40' fill='var(--ink-faint)'>σ_t · 沿 ε 方向</text>`);
  // 端点
  parts.push(`<circle cx='${D[0]}' cy='${D[1]}' r='3.5' fill='var(--ink)'/>`);
  parts.push(`<circle cx='${N[0]}' cy='${N[1]}' r='3.5' fill='var(--ink)'/>`);
  parts.push(`<text ${MONO_S} x='${D[0]}' y='268' fill='var(--ink)' text-anchor='middle'>t = 0 · x₀ 数据端</text>`);
  parts.push(`<text ${MONO_S} x='152' y='60' fill='var(--ink)'>t = 1 · ε 噪声端</text>`);

  // VP 圆弧（α²+σ²=1 ⇒ 单位圆 quarter arc）
  const arcPath = `M ${D[0]} ${D[1]} A ${R} ${R} 0 0 0 ${N[0]} ${N[1]}`;
  parts.push(`<path d='${arcPath}' fill='none' stroke='var(--ink-soft)' stroke-width='1.7'/>`);
  // 弧上箭头（θ = 67.5°）
  const th = (67.5 * Math.PI) / 180;
  const ap = [Ox + R * Math.cos(th), Oy - R * Math.sin(th)];
  parts.push(ah(ap[0], ap[1], -Math.sin(th), -Math.cos(th), 8, 'var(--ink-soft)'));

  // RF 弦
  const chordPath = `M ${D[0]} ${D[1]} L ${N[0]} ${N[1]}`;
  parts.push(`<path d='${chordPath}' fill='none' stroke='var(--seal)' stroke-width='1.8'/>`);
  const cp = [D[0] - 0.72 * R, D[1] - 0.72 * R];
  parts.push(ah(cp[0], cp[1], -0.707, -0.707, 8, 'var(--seal)'));

  // t 刻度：弦上均匀；弧上按 cosine schedule 均匀角度
  for (const t of [0.25, 0.5, 0.75]) {
    const x = D[0] - t * R, y = D[1] - t * R;
    parts.push(`<circle cx='${x}' cy='${y}' r='2.2' fill='var(--paper-raised)' stroke='var(--seal)' stroke-width='1.1'/>`);
    const a = (t * Math.PI) / 2;
    const ax = Ox + R * Math.cos(a), ay = Oy - R * Math.sin(a);
    parts.push(`<circle cx='${ax.toFixed(1)}' cy='${ay.toFixed(1)}' r='2.2' fill='var(--paper-raised)' stroke='var(--ink-soft)' stroke-width='1.1'/>`);
  }
  parts.push(`<text ${MONO_S} x='240' y='150' fill='var(--ink-faint)'>t = 0.5</text>`);
  parts.push(`<text ${MONO_S} x='276' y='102' fill='var(--ink-faint)'>t = 0.5</text>`);

  // 曲率与长度注记
  parts.push(`<text ${MONO_S} x='214' y='158' fill='var(--seal-deep)' text-anchor='end'>κ = 0（直）</text>`);
  parts.push(`<text ${MONO_S} x='286' y='124' fill='var(--ink-soft)'>κ ≠ 0（恒弯）</text>`);
  parts.push(`<text ${MONO_S} x='270' y='214' fill='var(--seal-deep)' text-anchor='end'>弦长 √2 ≈ 1.414</text>`);
  parts.push(`<text ${MONO_S} x='296' y='92' fill='var(--ink-soft)'>弧长 π/2 ≈ 1.571</text>`);

  // 双流光点：同端点、同周期、不同路径（前向 = 加噪方向）
  parts.push(
    `<circle class='sig-flow' r='2.8' fill='var(--seal)'><animateMotion dur='5.5s' repeatCount='indefinite' path='${arcPath}'/></circle>`,
    `<circle class='sig-flow' r='2.8' fill='var(--seal)'><animateMotion dur='5.5s' repeatCount='indefinite' path='${chordPath}'/></circle>`
  );

  // 右栏：图例与机制注记
  parts.push(`<line x1='396' y1='58' x2='418' y2='58' stroke='var(--ink-soft)' stroke-width='1.7'/>`);
  parts.push(`<text ${MONO_S} x='424' y='62' fill='var(--ink-soft)'>VP / DDPM：(√ᾱ_t, √(1−ᾱ_t))（式 5）</text>`);
  parts.push(`<line x1='396' y1='80' x2='418' y2='80' stroke='var(--seal)' stroke-width='1.8'/>`);
  parts.push(`<text ${MONO_S} x='424' y='84' fill='var(--seal-deep)'>rectified flow：(1−t, t)（式 20）</text>`);
  const notes = [
    '· VP：α²+σ² = 1 ⇒ 钉在圆弧上；schedule 只改配速、不改弧',
    '· 弧长 π/2 ≈ 1.571 vs 弦长 √2 ≈ 1.414 — 多 11%、全程弯曲',
    '· Euler 误差 = h²/2·‖ẍ‖（式 32）⇒ 直轨迹可大步长',
    '· 直线是条件轨迹；边际轨迹因端点对交叉仍弯（§6）',
    '· reflow：自产端点对重训，逐轮拉直（7.4 §5.4）',
    '· 光点：前向 t: 0 → 1（加噪）；采样反向 1 → 0',
  ];
  notes.forEach((s, i) => {
    parts.push(`<text ${CJK_S} x='396' y='${112 + i * 20}' fill='var(--ink-soft)'>${s}</text>`);
  });
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 7.6 §7.1 — MMDiT 双流各自 QKV + 联合 attention 矩阵 ============ */
function mmditDualStream() {
  const w = 720, h = 360;
  const parts = [svgOpen(w, h, 'MMDiT：图像流与文本流各自 QKV，在拼接序列上联合 attention，再分流回各自 MLP')];
  parts.push(
    `<text ${MONO_S} x='24' y='22' fill='var(--ink-soft)'>两流各持一套 block 参数（QKV / MLP / adaLN 各自的 W^z 与 W^y）；attention 在拼接序列 [N+M] 上联合计算（式 10–12）</text>`
  );

  // 左列：两流各自的 adaLN 与 QKV
  const chip = (x, y, cw, ch, fill, stroke, lines) => {
    parts.push(`<rect x='${x}' y='${y}' width='${cw}' height='${ch}' rx='3' fill='${fill}' stroke='${stroke}' stroke-width='1.1'/>`);
    lines.forEach((ln, i) => {
      const [txt, font, color] = ln;
      parts.push(`<text ${font} x='${x + cw / 2}' y='${y + (lines.length === 1 ? ch / 2 + 4 : 17 + i * 15)}' fill='${color}' text-anchor='middle'>${txt}</text>`);
    });
  };
  // 图像流 Z
  chip(24, 52, 132, 40, '#f1efe8', '#c6c0af', [
    ['Z ∈ R[N,d]', MONO, 'var(--ink)'],
    ['图像 latent token', CJK_S, 'var(--ink-faint)'],
  ]);
  chip(24, 102, 132, 22, '#e9efe9', '#a9bfa9', [['adaLN_z(c)', MONO_S, 'var(--ink-soft)']]);
  chip(24, 134, 132, 26, '#e9edf3', '#aebcd2', [['QKV^z = Z·W^z', MONO_S, 'var(--ink)']]);
  // 文本流 Y
  chip(24, 256, 132, 40, '#e9edf3', '#aebcd2', [
    ['Y ∈ R[M,d]', MONO, 'var(--ink)'],
    ['文本 token', CJK_S, 'var(--ink-faint)'],
  ]);
  chip(24, 224, 132, 22, '#e9efe9', '#a9bfa9', [['adaLN_y(c)', MONO_S, 'var(--ink-soft)']]);
  chip(24, 188, 132, 26, '#e9edf3', '#aebcd2', [['QKV^y = Y·W^y', MONO_S, 'var(--ink)']]);
  // 流内小箭头
  for (const [x1, y1, x2, y2] of [
    [90, 92, 90, 100], [90, 124, 90, 132],
    [90, 256, 90, 248], [90, 224, 90, 216],
  ]) {
    parts.push(`<line x1='${x1}' y1='${y1}' x2='${x2}' y2='${y2}' stroke='#a9a498' stroke-width='1.1'/>`);
  }

  // 拼接节点
  parts.push(`<line x1='156' y1='147' x2='204' y2='170' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(`<line x1='156' y1='201' x2='204' y2='178' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(`<circle cx='212' cy='174' r='4.5' fill='var(--paper-raised)' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  parts.push(`<text ${MONO_S} x='212' y='158' fill='var(--ink-faint)' text-anchor='middle'>拼接 [N+M]</text>`);
  parts.push(`<line x1='218' y1='174' x2='296' y2='174' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(ah(302, 174, 1, 0, 7, 'var(--ink-soft)'));

  // 联合 attention 矩阵（N:M = 3:1）
  const MX = 306, MY = 62, MS = 188, NB = 141;
  parts.push(`<text ${MONO_S} x='306' y='50' fill='var(--ink-soft)'>S = QKᵀ/√d_h（联合，N+M）</text>`);
  // S_zz / S_yy（自注意力块，attn 色系）
  parts.push(`<rect x='${MX}' y='${MY}' width='${NB}' height='${NB}' fill='#f6e9e5' stroke='#cf9d92' stroke-width='1.1'/>`);
  parts.push(`<rect x='${MX + NB}' y='${MY + NB}' width='${MS - NB}' height='${MS - NB}' fill='#f6e9e5' stroke='#cf9d92' stroke-width='1.1'/>`);
  // 交叉块（朱砂水洗）；S_yz 为 cross-attn 没有的对称块 — 脉冲强调
  parts.push(`<rect x='${MX + NB}' y='${MY}' width='${MS - NB}' height='${NB}' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`);
  parts.push(
    `<rect class='sig-diag' style='--i:0' x='${MX}' y='${MY + NB}' width='${NB}' height='${MS - NB}' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.5'/>`
  );
  // 块标签
  parts.push(`<text ${MONO_S} x='${MX + NB / 2}' y='${MY + 66}' fill='var(--ink)' text-anchor='middle'>S_zz</text>`);
  parts.push(`<text ${CJK_S} x='${MX + NB / 2}' y='${MY + 82}' fill='var(--ink-faint)' text-anchor='middle'>图·图</text>`);
  parts.push(`<text ${MONO_S} x='${MX + NB + (MS - NB) / 2}' y='${MY + 64}' fill='var(--seal-deep)' text-anchor='middle'>S_zy</text>`);
  parts.push(`<text ${CJK_S} x='${MX + NB + (MS - NB) / 2}' y='${MY + 80}' fill='var(--seal-deep)' text-anchor='middle'>图读文</text>`);
  parts.push(`<text ${MONO_S} x='${MX + NB / 2}' y='${MY + NB + 20}' fill='var(--seal-deep)' text-anchor='middle'>S_yz</text>`);
  parts.push(`<text ${CJK_S} x='${MX + NB / 2}' y='${MY + NB + 36}' fill='var(--seal-deep)' text-anchor='middle'>文读图</text>`);
  parts.push(`<text ${MONO_S} x='${MX + NB + (MS - NB) / 2}' y='${MY + NB + 20}' fill='var(--ink)' text-anchor='middle'>S_yy</text>`);
  parts.push(`<text ${CJK_S} x='${MX + NB + (MS - NB) / 2}' y='${MY + NB + 36}' fill='var(--ink-faint)' text-anchor='middle'>文·文</text>`);
  // 行 softmax：一行横跨 N+M 个 key
  parts.push(`<line x1='${MX + 6}' y1='${MY + 100}' x2='${MX + MS - 12}' y2='${MY + 100}' stroke='var(--ink-faint)' stroke-width='1'/>`);
  parts.push(ah(MX + MS - 6, MY + 100, 1, 0, 6, 'var(--ink-soft)'));
  parts.push(
    `<text ${MONO_S} x='306' y='266' fill='var(--ink-faint)'>softmax 按行、跨 N+M 个 key 联合归一（式 11）</text>`
  );

  // 拆流：O_z / O_y 回各自输出投影
  parts.push(`<line x1='494' y1='120' x2='554' y2='98' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(ah(560, 96.6, 0.94, -0.34, 7, 'var(--ink-soft)'));
  parts.push(`<text ${MONO_S} x='524' y='104' fill='var(--ink-faint)' text-anchor='middle'>O_z</text>`);
  parts.push(`<line x1='494' y1='226' x2='554' y2='231.4' stroke='#a9a498' stroke-width='1.2' stroke-linecap='round'/>`);
  parts.push(ah(560, 232, 0.996, 0.09, 7, 'var(--ink-soft)'));
  parts.push(`<text ${MONO_S} x='524' y='248' fill='var(--ink-faint)' text-anchor='middle'>O_y</text>`);
  chip(560, 72, 148, 48, '#f1efe8', '#c6c0af', [
    ['Z += α_z⊙O_z·W_O^z', MONO_S, 'var(--ink)'],
    ['+ MLP_z（式 8 同形）', MONO_S, 'var(--ink-faint)'],
  ]);
  chip(560, 208, 148, 48, '#e9edf3', '#aebcd2', [
    ['Y += α_y⊙O_y·W_O^y', MONO_S, 'var(--ink)'],
    ['+ MLP_y · 不出图', MONO_S, 'var(--ink-faint)'],
  ]);

  // 流光：Z/Y 两路 → 拼接 → 矩阵 → 分流
  parts.push(
    `<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='1.4s' repeatCount='indefinite' path='M 156 147 L 204 170'/></circle>`,
    `<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='1.4s' begin='0.15s' repeatCount='indefinite' path='M 156 201 L 204 178'/></circle>`,
    `<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='1.4s' begin='0.6s' repeatCount='indefinite' path='M 218 174 L 298 174'/></circle>`,
    `<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='1.6s' begin='1.1s' repeatCount='indefinite' path='M 496 120 C 520 120, 536 100, 556 98'/></circle>`,
    `<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='1.6s' begin='1.25s' repeatCount='indefinite' path='M 494 226 C 520 226, 536 231, 556 232'/></circle>`
  );

  // 底部注记
  parts.push(
    `<text ${MONO_S} x='24' y='316' fill='var(--ink-faint)'>c = pooled 文向 + emb(t) — 两流共享条件向量（adaLN 入口，式 8）</text>`
  );
  parts.push(
    `<text ${CJK_S} x='24' y='332' fill='var(--ink-faint)'>S_yz（朱砂脉冲）= 文读图 — cross-attn 没有的对称块：文本逐层被图像去噪状态修正（双向 = 与 §3 (a) 的本质差异）</text>`
  );
  parts.push(
    `<text ${CJK_S} x='24' y='348' fill='var(--ink-faint)'>出口只有 Z（末端 adaLN + 线性层回归 v_θ，7.5 式 22）；Y 只被读、不出图 — 多付的账 = 文本流 QKV/MLP + M² 自注意力</text>`
  );
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '07-diffusion-image-generation/02-ddpm',
    afterHeading: /^### 2\.3 noise schedule/,
    title: '噪声日程：信号逐层淹没与 SNR',
    svg: ddpmSchedule,
  },
  {
    slug: '07-diffusion-image-generation/03-score-sde',
    afterHeading: /^### 6\.2 确定性采样/,
    title: '同一边缘族：逆时 SDE vs PF-ODE 轨迹',
    svg: sdeOdeTrajectories,
  },
  {
    slug: '07-diffusion-image-generation/04-sampling-and-guidance',
    afterHeading: /^### 4\.2 CFG/,
    title: 'CFG 的向量合成：ε̃ = ε_u + w·Δ',
    svg: cfgGeometry,
  },
  {
    slug: '07-diffusion-image-generation/05-flow-matching',
    afterHeading: /^## 7\. 与 DDPM/,
    title: '路径几何：VP 圆弧 vs rectified flow 弦',
    svg: rfVsVpPaths,
  },
  {
    slug: '07-diffusion-image-generation/06-latent-diffusion-and-dit',
    afterHeading: /^### 7\.1 从 DiT 到 MMDiT/,
    title: 'MMDiT 双流与联合 attention 矩阵',
    svg: mmditDualStream,
  },
];
