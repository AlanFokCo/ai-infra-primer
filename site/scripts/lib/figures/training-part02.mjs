// training Part 2（02-optimizers）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；脉冲渐变用页面级 PULSE_GRAD_DEFS 固定 id。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// 手工路径上的脉冲（轨迹线/回环用 path 直写；渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 2.1 §3（block 0）— heavy ball 速度形式 × EMA 形式：同一算法两种参数化
 * 原文：v ← β·v + ĝ，θ ← θ − η·v（式 8）；m ← β·m + (1−β)·ĝ，θ ← θ − η′·m（式 9）；
 * 变量代换 m = (1−β)·v，η′ = η/(1−β)（式 10）
 * ============================================================ */
function momentumTwoForms() {
  const UID = 't2a0';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'momentum 的两种写法是同一算法的两种参数化：换算只隔一个因子 1−β（式 10）', { size: 12, color: 'var(--ink-soft)' }));

  // 上通道：速度形式（式 8）
  body.push(cap(56, 54, '式 (8) 速度形式（PyTorch 约定）', { size: 10.5, weight: '650' }));
  const g1 = node({ id: 'g1', x: 56, y: 66, w: 76, h: 44, role: 'data', title: 'ĝ_t', mono: true, titleSize: 13 });
  const v = node({ id: 'v', x: 216, y: 62, w: 224, h: 48, role: 'op', title: 'v ← β·v + ĝ', sub: '速度累加器（权重和 → 1/(1−β)）', mono: true, titleSize: 12.5 });
  const t1 = node({ id: 't1', x: 524, y: 62, w: 164, h: 48, role: 'op', title: 'θ ← θ − η·v', mono: true, titleSize: 12.5 });
  [g1, v, t1].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 下通道：EMA 形式（式 9）
  body.push(cap(56, 190, '式 (9) EMA 形式', { size: 10.5, weight: '650' }));
  const g2 = node({ id: 'g2', x: 56, y: 202, w: 76, h: 44, role: 'data', title: 'ĝ_t', mono: true, titleSize: 13 });
  const m = node({ id: 'm', x: 216, y: 198, w: 224, h: 48, role: 'op', title: 'm ← β·m + (1−β)·ĝ', sub: 'EMA 平滑器（权重和 = 1 量级）', mono: true, titleSize: 12.5 });
  const t2 = node({ id: 't2', x: 524, y: 198, w: 164, h: 48, role: 'op', title: 'θ ← θ − η′·m', mono: true, titleSize: 12.5 });
  [g2, m, t2].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 两通道流
  body.push(edge(g1, v, { nodes, pulse: true, dur: 1.4, uid: UID, eid: 'a' }));
  body.push(edge(v, t1, { nodes, pulse: true, dur: 1.4, delay: 0.25, uid: UID, eid: 'b' }));
  body.push(edge(g2, m, { nodes, pulse: true, dur: 1.4, delay: 0.5, uid: UID, eid: 'c' }));
  body.push(edge(m, t2, { nodes, pulse: true, dur: 1.4, delay: 0.75, uid: UID, eid: 'd' }));

  // 变量代换桥（虚线）
  const br = node({ id: 'br', x: 226, y: 130, w: 204, h: 34, role: 'cache', title: 'm = (1−β)·v', mono: true, titleSize: 11 });
  nodes.push(br); body.push(br.svg);
  body.push(edge(v, br, { nodes, portFrom: v.bottom, portTo: br.top, uid: UID, eid: 'e', dash: true, arrow: false }));
  body.push(edge(br, m, { nodes, portFrom: br.bottom, portTo: m.top, uid: UID, eid: 'f', dash: true, arrow: false }));
  body.push(cap(452, 151, 'η′ = η/(1−β)（式 10）', { mono: true, size: 10.5 }));

  body.push(cap(36, 272, '混用两形式的超参，复现就悄悄偏了 —— 换写法必须联动 η（β = 0.9 时有效步长差 10 倍，式 11）', { size: 10.5 }));

  return scene(720, 290, body, { label: 'momentum 速度形式与 EMA 形式的双通道对照，中间以变量代换 m=(1−β)v 桥接' }).svg();
}

/* ============================================================
 * 2.1 §5（block 1）— 二次碗截面：GD zigzag vs heavy ball
 * 原文：GD 的 η ≤ 2/L 被快方向压住、慢方向每步只缩 ≈1−2/κ；heavy ball 历史速度
 * 在快方向符号交替自相抵消、慢方向同号累加；ρ_GD = (κ−1)/(κ+1)，ρ_HB = (√κ−1)/(√κ+1)
 * 轨迹为真实模拟：示意碗 κ = 6.25（μ = 1，L = 6.25），GD 取 η* = 2/(L+μ)，
 * HB 取式 (20) 的最优 β*、η*（零初速 e_{-1} = e_0）
 * ============================================================ */
function quadraticBowl() {
  const UID = 't2a1';
  const body = [];
  const MU = 1, L = 16, KAPPA = L / MU;
  const CX = 270, CY = 180, AX = 230, AY = 57.5; // 屏幕映射：半轴比 √κ = 4
  const sx = (u) => CX + u * AX;
  const sy = (v) => CY - v * AY;

  body.push(cap(36, 22, '二次碗截面（H 的两个特征方向）：GD 在碗壁间 zigzag，heavy ball 快方向抵消、慢方向加速', { size: 12, color: 'var(--ink-soft)' }));

  // 等高线椭圆（半轴 ∝ 1/√λ）；最外一道 c = 1.5 罩住 heavy ball 的冲顶过冲
  [1.5, 1, 0.55, 0.25, 0.08].forEach((c) => {
    body.push(`<ellipse cx="${CX}" cy="${CY}" rx="${(AX * Math.sqrt(c)).toFixed(1)}" ry="${(AY * Math.sqrt(c)).toFixed(1)}" fill="none" stroke="rgb(24 27 32 / 0.18)" stroke-width="1.1"/>`);
  });
  // 特征轴
  body.push(`<line class="dia-edge" x1="30" y1="${CY}" x2="${510}" y2="${CY}"/>`);
  body.push(`<polygon points="514,${CY} 505,${CY - 4.5} 505,${CY + 4.5}" fill="var(--ink-soft)"/>`);
  body.push(`<line class="dia-edge" x1="${CX}" y1="304" x2="${CX}" y2="56"/>`);
  body.push(`<polygon points="${CX},52 ${CX - 4.5},61 ${CX + 4.5},61" fill="var(--ink-soft)"/>`);
  body.push(cap(508, CY + 18, '慢方向（曲率 μ）', { size: 10, anchor: 'end' }));
  body.push(cap(CX + 10, 52, '快方向（曲率 L = κμ）', { size: 10 }));
  // 最优点
  body.push(`<circle cx="${CX}" cy="${CY}" r="5" fill="var(--seal)"/>`);
  body.push(cap(CX + 12, CY - 10, 'θ*', { mono: true, size: 11, color: 'var(--seal-deep)', weight: '650' }));

  // GD 轨迹（η* = 2/(L+μ)）：快方向符号交替，24 步仍未走完
  const etaG = 2 / (L + MU);
  const u0 = 0.62, v0 = 0.245; // c = u² + κv² ≈ 1.34
  const gd = [[u0, v0]];
  for (let t = 1; t <= 20; t++) {
    const [uu, vv] = gd[t - 1];
    gd.push([(1 - etaG * MU) * uu, (1 - etaG * L) * vv]);
  }
  const gdPath = 'M' + gd.map(([u, v]) => `${sx(u).toFixed(1)} ${sy(v).toFixed(1)}`).join(' L');
  body.push(`<path class="dia-edge strong" d="${gdPath}"/>`);
  body.push(pathPulse(gdPath, { dur: 4.6, delay: 0.3, r: 3.8 }));

  // heavy ball 轨迹（式 20 最优调参，零初速）
  const sq = Math.sqrt(KAPPA);
  const beta = Math.pow((sq - 1) / (sq + 1), 2);
  const etaH = 4 / Math.pow(Math.sqrt(L) + Math.sqrt(MU), 2);
  const step = (em, em1, lam) => (1 + beta - etaH * lam) * em - beta * em1;
  const hb = [[u0, v0]];
  let [um, um1] = [u0, u0], [vm, vm1] = [v0, v0];
  for (let t = 1; t <= 10; t++) {
    const un = step(um, um1, MU), vn = step(vm, vm1, L);
    hb.push([un, vn]);
    [um, um1] = [un, um]; [vm, vm1] = [vn, vm];
  }
  const hbPath = 'M' + hb.map(([u, v]) => `${sx(u).toFixed(1)} ${sy(v).toFixed(1)}`).join(' L');
  body.push(`<path d="${hbPath}" fill="none" stroke="var(--seal)" stroke-width="1.8" stroke-linecap="round"/>`);
  body.push(pathPulse(hbPath, { dur: 2.6, delay: 0.9, r: 3.8 }));

  // 图例与结论（右栏）
  body.push(`<line x1="540" y1="66" x2="572" y2="66" class="dia-edge strong"/>`);
  body.push(cap(580, 70, 'GD（η* = 2/(L+μ)）', { mono: true, size: 10 }));
  body.push(`<line x1="540" y1="88" x2="572" y2="88" stroke="var(--seal)" stroke-width="1.8"/>`);
  body.push(cap(580, 92, 'heavy ball（式 20 最优）', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(540, 124, 'ρ_GD = (κ−1)/(κ+1)', { mono: true, size: 10.5 }));
  body.push(cap(540, 140, '≈ 1 − 2/κ', { mono: true, size: 10.5 }));
  body.push(cap(540, 164, 'ρ_HB = (√κ−1)/(√κ+1)', { mono: true, size: 10.5, color: 'var(--seal-deep)' }));
  body.push(cap(540, 180, '≈ 1 − 2/√κ —— 条件数开方', { size: 10.5, color: 'var(--seal-deep)' }));
  body.push(cap(540, 214, '步数（ε = 10⁻⁶，§5 表）', { size: 10 }));
  body.push(cap(540, 230, 'κ = 10²：691 → 69 步', { mono: true, size: 10 }));
  body.push(cap(540, 246, 'κ = 10⁴：69078 → 691 步', { mono: true, size: 10 }));

  body.push(cap(36, 330, 'GD：步长被快方向压住（η ≤ 2/L 才不发散），慢方向每步只缩 ≈ 1 − 2/κ', { size: 10.5 }));
  body.push(cap(36, 347, 'heavy ball：历史速度在快方向符号交替自相抵消、在慢方向同号累加 —— 全方向统一以 √β 收缩', { size: 10.5 }));
  body.push(cap(36, 364, '示意碗 κ = 16（为可读性取的压扁比）；轨迹为式 (1)(17) 最优调参下的真实模拟，速率公式与步数账为原文口径', { size: 9.5 }));

  return scene(720, 380, body, { label: '二次碗截面：GD 轨迹在快方向 zigzag 缓慢推进，heavy ball 平滑加速直达最优点' }).svg();
}

/* ============================================================
 * 2.2 §2–§6（block 0）— Adam 双通道汇合；L2 与 AdamW 两条 λθ 支路
 * 原文：m ← β1·m + (1−β1)·g → m̂ = m/(1−β1^t)；v ← β2·v + (1−β2)·g⊙g → v̂；
 * θ ← θ·(1−ηλ) − η·m̂/(√v̂ + ε)；L2：g ← g + λθ（进两通道被 √v̂ 缩放）；
 * AdamW：θ ← θ·(1−ηλ)（λθ 不经过 m / v / √v̂）
 * ============================================================ */
function adamChannels() {
  const UID = 't2b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'Adam = 双通道矩估计 + 逐坐标归一；L2 与 AdamW 的差别只在 λθ 走哪条路（Prop 2）', { size: 12, color: 'var(--ink-soft)' }));

  const g = node({ id: 'g', x: 36, y: 132, w: 84, h: 48, role: 'data', title: 'g_t', mono: true, titleSize: 13 });
  const m = node({ id: 'm', x: 200, y: 56, w: 220, h: 48, role: 'op', title: 'm ← β₁·m + (1−β₁)·g', sub: '一阶矩（EMA）', mono: true, titleSize: 12 });
  const v = node({ id: 'v', x: 200, y: 168, w: 220, h: 48, role: 'op', title: 'v ← β₂·v + (1−β₂)·g⊙g', sub: '二阶原始矩（非方差）', mono: true, titleSize: 12 });
  const mh = node({ id: 'mh', x: 480, y: 56, w: 200, h: 48, role: 'proj', title: 'm̂ = m/(1−β₁ᵗ)', sub: '偏差校正', mono: true, titleSize: 12 });
  const vh = node({ id: 'vh', x: 480, y: 168, w: 200, h: 48, role: 'proj', title: 'v̂ = v/(1−β₂ᵗ)', sub: '偏差校正', mono: true, titleSize: 12 });
  const upd = node({ id: 'upd', x: 430, y: 262, w: 260, h: 58, role: 'attn', title: 'θ ← θ·(1−ηλ) − η·m̂/(√v̂ + ε)', sub: 'wd 项直通参数 · 梯度项逐坐标归一', mono: true, titleSize: 11.5 });
  [g, m, v, mh, vh, upd].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // 主流
  body.push(edge(g, m, { nodes, portFrom: g.top, portTo: m.left, pulse: true, dur: 1.5, uid: UID, eid: 'a' }));
  body.push(edge(g, v, { nodes, portFrom: g.right, portTo: v.left, pulse: true, dur: 1.5, delay: 0.2, uid: UID, eid: 'b' }));
  body.push(edge(m, mh, { nodes, pulse: true, dur: 1.3, delay: 0.4, uid: UID, eid: 'c' }));
  body.push(edge(v, vh, { nodes, pulse: true, dur: 1.3, delay: 0.6, uid: UID, eid: 'd' }));
  body.push(edge(mh, upd, { nodes, portFrom: mh.bottom, portTo: upd.top, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 'e' }));
  body.push(edge(vh, upd, { nodes, portFrom: vh.bottom, portTo: upd.top, pulse: true, dur: 1.4, delay: 1.05, uid: UID, eid: 'f' }));

  // L2 支路（进梯度，污染两通道）
  const l2 = node({ id: 'l2', x: 36, y: 56, w: 130, h: 40, role: 'cache', title: 'L2：g += λθ', sub: '式 (8)', mono: true, titleSize: 11.5 });
  nodes.push(l2); body.push(l2.svg);
  body.push(edge(l2, g, { nodes, portFrom: l2.bottom, portTo: g.top, uid: UID, eid: 'g', dash: true }));
  body.push(cap(36, 116, 'λθ 进 m、v 两通道，被 √v̂ 逐坐标缩放', { size: 9.5 }));

  // AdamW 支路（解耦，直通参数）
  const aw = node({ id: 'aw', x: 150, y: 270, w: 220, h: 44, role: 'cache', title: 'AdamW：θ ← θ·(1−ηλ)', sub: '式 (9)：λθ 不经过 m / v / √v̂', mono: true, titleSize: 11.5 });
  nodes.push(aw); body.push(aw.svg);
  body.push(edge(aw, upd, { nodes, pulse: true, dur: 1.6, delay: 1.3, uid: UID, eid: 'h', dash: true, strong: true }));

  body.push(cap(36, 346, 'torch 实现的分岔只有一行：param.mul_(1 − lr·wd)（AdamW）vs grad.add(param)（L2）—— §9 源码走读', { size: 10.5 }));

  return scene(720, 362, body, { label: 'Adam 的 m/v 双通道经偏差校正汇合更新；L2 把 λθ 混进梯度通道，AdamW 让 λθ 直通参数' }).svg();
}

/* ============================================================
 * 2.3 §3（block 0）— WSD vs cosine 两条 LR 曲线
 * 原文：统一 η_t = η_max·s(t)；WSD = warmup / stable / decay 三段（式 4），
 * stable 可任意延长；cosine 在 t = S 收到 η_min（≈ 10% 峰值口径）
 * ============================================================ */
function lrSchedules() {
  const UID = 't2c';
  const body = [];
  const OX = 80, OY = 250, X1 = 660;
  const YMAX = 70, YMIN = 232; // η/η_max = 1.0 与 0.1 的纵坐标
  const W = 170, T = 420, S = 600;
  const ey = (eta) => OY - eta * 180; // η ∈ [0, 1.05] → y

  body.push(cap(36, 22, 'η_t = η_max·s(t) 的两个家族：WSD 把预算决策推迟到 decay 段开始前；cosine 把预算写死在曲线里', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴
  body.push(`<line class="dia-edge" x1="${OX}" y1="36" x2="${OX}" y2="${OY}"/>`);
  body.push(`<line class="dia-edge" x1="${OX}" y1="${OY}" x2="${X1}" y2="${OY}"/>`);
  body.push(`<polygon points="${OX},32 ${OX - 4.5},41 ${OX + 4.5},41" fill="var(--ink-soft)"/>`);
  body.push(`<polygon points="${X1 + 4},${OY} ${X1 - 5},${OY - 4.5} ${X1 - 4},${OY + 4.5}" fill="var(--ink-soft)"/>`);
  body.push(cap(OX - 8, 46, 'η/η_max', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(OX - 8, ey(1) + 4, '1.0', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(OX - 8, ey(0.1) + 4, '0.1', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(OX - 8, OY + 4, '0', { mono: true, size: 10, anchor: 'end' }));
  body.push(`<line x1="${OX}" y1="${ey(0.1)}" x2="${X1}" y2="${ey(0.1)}" stroke="var(--rule)" stroke-width="0.8" stroke-dasharray="3 4"/>`);
  body.push(cap(X1 + 2, ey(0.1) - 6, 'η_min ≈ 10%', { mono: true, size: 9.5, anchor: 'end', color: 'var(--ink-faint)' }));

  // x 轴刻度与区间标
  [[OX, '0'], [W, 'W'], [T, 'T'], [S, 'S']].forEach(([x, l]) => {
    body.push(`<line class="dia-edge" x1="${x}" y1="${OY - 4}" x2="${x}" y2="${OY + 4}"/>`);
    body.push(cap(x, OY + 18, l, { mono: true, size: 10, anchor: 'middle' }));
  });
  body.push(cap((OX + W) / 2, OY + 34, 'warmup', { size: 10, anchor: 'middle' }));
  body.push(cap((W + T) / 2, OY + 34, 'stable（WSD 独有）', { size: 10, anchor: 'middle' }));
  body.push(cap((T + S) / 2, OY + 34, 'decay', { size: 10, anchor: 'middle' }));

  // WSD（墨线）：warmup → stable → decay
  const wsd = `M${OX} ${OY} L${W} ${ey(1)} L${T} ${ey(1)} C${T + 60} ${ey(1) - 4}, ${S - 80} ${ey(0.35)}, ${S} ${ey(0.1)}`;
  body.push(`<path class="dia-edge strong" d="${wsd}"/>`);
  body.push(pathPulse(wsd, { dur: 4.2, delay: 0.3, r: 3.6 }));
  // stable 段可延长的虚线暗示
  body.push(`<path d="M${T} ${ey(1)} L${T + 120} ${ey(1)}" class="dia-edge dash"/>`);
  body.push(cap(T + 66, ey(1) - 10, 'stable 可任意延长', { size: 9.5, anchor: 'middle' }));
  body.push(cap((W + T) / 2 - 40, ey(1) - 12, 'WSD（梯形）', { size: 10.5, anchor: 'middle', weight: '650' }));

  // cosine（朱砂）：共用 warmup（虚线，微偏 2px 让两线都可见），其后 cos 收到 η_min
  body.push(`<path d="M${OX + 2} ${OY} L${W + 2} ${ey(1) + 2}" stroke="var(--seal)" stroke-width="1.6" fill="none" stroke-dasharray="4 4" opacity="0.7"/>`);
  const cosPts = [];
  for (let i = 0; i <= 28; i++) {
    const t = W + ((S - W) * i) / 28;
    const eta = 0.1 + 0.45 * (1 + Math.cos((Math.PI * (t - W)) / (S - W)));
    cosPts.push(`${t.toFixed(1)} ${ey(eta).toFixed(1)}`);
  }
  const cosD = 'M' + cosPts.join(' L');
  body.push(`<path d="${cosD}" fill="none" stroke="var(--seal)" stroke-width="1.8" stroke-linecap="round"/>`);
  body.push(pathPulse(cosD, { dur: 4.2, delay: 1.2, r: 3.6 }));
  body.push(cap(470, 150, 'cosine（T = S）', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(OX, 318, '两条曲线共用 warmup 段（cosine 侧以朱砂虚线画出）；WSD 的 decay 一开始 loss 骤降、追平同 token 数 cosine（MiniCPM 风洞）', { size: 10.5 }));
  body.push(cap(OX, 335, 'decay 段占总 token ~10% 足够、2.5% 不足；收尾口径：GPT-3 / Llama-2 收到峰值 10%，OLMo 2 mid-training 线性收到 0', { size: 10.5 }));

  return scene(720, 352, body, { label: 'WSD 与 cosine 两条学习率曲线：WSD 的稳定段可任意延长，cosine 在 S 处收到 η_min' }).svg();
}

/* ============================================================
 * 2.4 §6（block 0）— 稳定原语 placement 总图：七站截面 + 监控总线
 * 原文：数据 batch / 前向 / logits / loss / 反向梯度 / optimizer 更新 / 参数 七站，
 * 各挂防护原语；底部监控总线 grad norm · update ratio · logit max · attention entropy
 * ============================================================ */
function stabilityPlacement() {
  const UID = 't2d';
  const body = [];
  const nodes = [];
  const ST = [
    ['数据 batch', '过滤 / 配比', '→ 7.1'],
    ['前向', 'QK-norm', 'soft-capping'],
    ['logits', 'z-loss', '→ 3.1'],
    ['loss', 'warmup', '→ 2.3'],
    ['反向梯度', '混合精度 · EGS', '→ 1.3 · 式(3)'],
    ['optimizer 更新', 'grad clip', '+ wd（→ 2.2）'],
    ['参数', 'MuonClip', 'QK-clip（→ 2.5）'],
  ];
  const CXS = [55, 152, 249, 346, 443, 540, 637];

  body.push(cap(36, 22, '稳定原语 placement：每个问题在它的原生截面处理，串扰最少（§6）', { size: 12, color: 'var(--ink-soft)' }));

  // 主链七站
  const stations = ST.map(([t], i) => {
    const n = node({ id: `s${i}`, x: CXS[i] - 43, y: 52, w: 86, h: 40, role: 'op', title: t, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  stations.forEach((s, i) => {
    if (i) body.push(edge(stations[i - 1], s, { nodes, pulse: true, dur: 1.6, delay: i * 0.25, uid: UID, eid: `s${i}` }));
  });
  body.push(cap(36, 46, '一个训练 step 的数据流 →', { size: 10 }));

  // 每站防护挂牌
  ST.forEach(([, g1, g2], i) => {
    const gd = node({ id: `g${i}`, x: CXS[i] - 47, y: 126, w: 94, h: 46, role: 'cache', title: g1, sub: g2, titleSize: 10, subSize: 9 });
    nodes.push(gd); body.push(`<g class="sig-cell" style="--i:${(0.2 + i * 0.25).toFixed(2)}">${gd.svg}</g>`);
    body.push(edge([CXS[i], 92], [CXS[i], 126], { nodes, uid: UID, eid: `h${i}`, dash: true, arrow: false }));
  });

  // 监控总线
  body.push(cap(40, 224, '监控总线', { size: 10.5, weight: '650' }));
  body.push(`<line x1="40" y1="244" x2="690" y2="244" stroke="var(--seal)" stroke-width="1.3" class="sig-dashflow"/>`);
  const MET = [
    ['grad norm', 170, 100],
    ['update ratio', 320, 112],
    ['logit max', 466, 96],
    ['attention entropy', 610, 140],
  ];
  MET.forEach(([t, cx, w], i) => {
    const n = node({ id: `m${i}`, x: cx - w / 2, y: 230, w, h: 28, role: 'plain', title: t, mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
  });
  // 反向梯度站与总线的虚线回授
  body.push(edge([443, 172], [443, 230], { nodes, uid: UID, eid: 'fb', dash: true, arrow: false }));
  body.push(cap(452, 206, 'clip 返回值白送 total_norm', { size: 9 }));

  body.push(cap(40, 290, 'grad norm spike 先行于 collapse 数步、多由 embedding 层异常梯度引起（GLM-130B §3）—— 分位数定阈值 + 触发率监控', { size: 10.5 }));
  body.push(cap(40, 307, 'LR 端注记：warmup / schedule 改的是 LR 这个标量序列、不改数据流（→ 2.3），贯穿全程', { size: 10.5 }));

  return scene(720, 324, body, { label: '一个训练 step 的七个截面上各挂稳定原语，底部监控总线收集 grad norm 等四类信号' }).svg();
}

/* ============================================================
 * 2.5 §3（block 0）— Newton-Schulz 迭代与奇异值谱压缩
 * 原文：X₀ = G/‖G‖_F（谱压进 [0,1]）；T = 5 步 A = X_k X_kᵀ（2m²n）、
 * B = b·A + c·(A·A)（2m³）、X_{k+1} = a·X_k + B·X_k（2m²n），全程 bf16；
 * O = X₅ ≈ U S′ Vᵀ，S′ ∈ [0.5,1.5] 邻域（5 步把 [10⁻³,1] 压进约 [0.47,1.20]）
 * ============================================================ */
function newtonSchulz() {
  const UID = 't2e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'Newton-Schulz 正交化：只改奇异值、不动奇异向量 —— X_{k+1} = U·diag(f(σ_k))·Vᵀ（式 8）', { size: 12, color: 'var(--ink-soft)' }));

  const G = node({ id: 'G', x: 36, y: 62, w: 104, h: 48, role: 'data', title: 'G [n, m]', mono: true, titleSize: 12 });
  const NM = node({ id: 'NM', x: 186, y: 62, w: 168, h: 48, role: 'op', title: 'X₀ = G/‖G‖_F', sub: '谱范数压进 [0, 1]', mono: true, titleSize: 12 });
  nodes.push(G, NM); body.push(G.svg, NM.svg);

  // 迭代环（zone）
  body.push(zone({ x: 396, y: 40, w: 244, h: 176, label: '× T = 5 · bf16' }));
  const Xk = node({ id: 'Xk', x: 412, y: 62, w: 100, h: 40, role: 'data', title: 'X_k', sub: '[m, n]', mono: true, titleSize: 12 });
  const A = node({ id: 'A', x: 540, y: 56, w: 92, h: 44, role: 'op', title: 'A = X·Xᵀ', sub: '2m²n', mono: true, titleSize: 10.5 });
  const B = node({ id: 'B', x: 540, y: 122, w: 92, h: 44, role: 'op', title: 'B = bA + cA²', sub: '2m³', mono: true, titleSize: 10 });
  const Xn = node({ id: 'Xn', x: 406, y: 132, w: 118, h: 44, role: 'proj', title: 'X_{k+1}', sub: 'aX + BX · 2m²n', mono: true, titleSize: 11, subSize: 9 });
  [Xk, A, B, Xn].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(Xk, A, { nodes, pulse: true, dur: 1.1, uid: UID, eid: 'l0' }));
  body.push(edge(A, B, { nodes, portFrom: A.bottom, portTo: B.top, pulse: true, dur: 1.1, delay: 0.25, uid: UID, eid: 'l1' }));
  body.push(edge(B, Xn, { nodes, portFrom: B.left, portTo: Xn.right, pulse: true, dur: 1.1, delay: 0.5, uid: UID, eid: 'l2' }));
  body.push(edge(Xn, Xk, { nodes, portFrom: Xn.top, portTo: Xk.bottom, pulse: true, dur: 1.1, delay: 0.75, uid: UID, eid: 'l3' }));

  // 输入与输出
  body.push(edge(G, NM, { nodes, pulse: true, dur: 1.3, uid: UID, eid: 'in0' }));
  body.push(edge(NM, Xk, { nodes, portFrom: NM.right, portTo: Xk.left, pulse: true, dur: 1.3, delay: 0.3, uid: UID, eid: 'in1' }));
  const O = node({ id: 'O', x: 556, y: 218, w: 140, h: 44, role: 'attn', title: 'O = X₅ [n, m]', sub: '≈ U S′ Vᵀ（转置回原向）', mono: true, titleSize: 11.5, subSize: 9 });
  nodes.push(O); body.push(O.svg);
  body.push(edge(Xn, O, { nodes, portFrom: Xn.bottom, portTo: O.left, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'out' }));

  /* ---- 奇异值谱：归一化后散落 → 5 步压进 [0.5, 1.5] 邻域 ---- */
  const SY = 322;
  body.push(cap(56, 290, 'σ(X₀)：归一化后散落 (0, 1]', { size: 10 }));
  body.push(cap(446, 290, 'σ(X₅)：压进 [0.5, 1.5] 邻域（实测 [0.47, 1.20]，§3 表）', { size: 10 }));
  body.push(`<line class="dia-edge" x1="66" y1="${SY}" x2="326" y2="${SY}"/>`);
  body.push(cap(66, SY + 16, '0', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(326, SY + 16, '1', { mono: true, size: 9.5, anchor: 'middle' }));
  [0.001, 0.01, 0.1, 0.5, 1.0].forEach((s, i) => {
    body.push(`<circle class="sig-cell" style="--i:${(0.2 + i * 0.2).toFixed(1)}" cx="${(66 + s * 260).toFixed(1)}" cy="${SY}" r="3.6" fill="var(--ink)"/>`);
  });
  body.push(cap(56, 356, 'f(σ) = aσ + bσ³ + cσ⁵，(a,b,c) = (3.4445, −4.7750, 2.0315)（式 9）', { mono: true, size: 9.5 }));

  body.push(cap(379, SY - 10, '5 步 f(σ)', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(`<path class="dia-edge" d="M330 ${SY} H 424"/>`);
  body.push(`<polygon points="428,${SY} 419,${SY - 4.5} 419,${SY + 4.5}" fill="var(--ink-soft)"/>`);

  // 刻度带 [0, 2] → x = 446 + σ·122
  body.push(`<rect x="${446 + 0.5 * 122}" y="${SY - 10}" width="${122}" height="20" rx="4" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="0.9" stroke-dasharray="3 3"/>`);
  body.push(`<line class="dia-edge" x1="446" y1="${SY}" x2="690" y2="${SY}"/>`);
  [0, 0.5, 1.0, 1.5, 2.0].forEach((s) => {
    body.push(cap(446 + s * 122, SY + 16, String(s), { mono: true, size: 9.5, anchor: 'middle' }));
  });
  [0.4705, 0.6989, 0.7121, 0.7654, 0.6964].forEach((s, i) => {
    body.push(`<circle class="sig-cell" style="--i:${(1.2 + i * 0.2).toFixed(1)}" cx="${(446 + s * 122).toFixed(1)}" cy="${SY}" r="3.6" fill="var(--seal)"/>`);
  });

  body.push(cap(36, 380, '开销 ≤ 6T·nm² FLOPs，对照该层前向+反向 6nmB：相对开销 ≤ T·m/B ≈ 0.7%（m = 768）/ 0.5%（405B 档 m = 16384）', { mono: true, size: 10 }));

  return scene(720, 396, body, { label: 'Newton-Schulz 五步迭代回路，左下为归一化后散落的奇异值，右下为迭代后压进 [0.5,1.5] 邻域的谱' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'training',
    slug: '02-optimizers/01-sgd-and-momentum',
    replacesBlock: 0,
    title: 'momentum 两种写法：速度形式 × EMA 形式',
    svg: momentumTwoForms,
  },
  {
    dim: 'training',
    slug: '02-optimizers/01-sgd-and-momentum',
    replacesBlock: 1,
    title: '二次碗：GD zigzag vs heavy ball —— κ 开方',
    svg: quadraticBowl,
  },
  {
    dim: 'training',
    slug: '02-optimizers/02-adam-and-adamw',
    replacesBlock: 0,
    title: 'Adam 双通道汇合；L2 与 AdamW 的 λθ 支路',
    svg: adamChannels,
  },
  {
    dim: 'training',
    slug: '02-optimizers/03-lr-schedules',
    replacesBlock: 0,
    title: 'WSD vs cosine：stable 段把预算决策推迟',
    svg: lrSchedules,
  },
  {
    dim: 'training',
    slug: '02-optimizers/04-gradient-clipping-and-stability',
    replacesBlock: 0,
    title: '稳定原语 placement：七站截面 + 监控总线',
    svg: stabilityPlacement,
  },
  {
    dim: 'training',
    slug: '02-optimizers/05-muon-and-spectral',
    replacesBlock: 0,
    title: 'Newton-Schulz：5 步把奇异值压进 [0.5, 1.5]',
    svg: newtonSchulz,
  },
];
