// gpu P4（04-kernel-memory-optimization）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs / gpu-p02.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-bar --i ≤ 0.8。
// census：4.1 block0 sector 判定图；4.2 block0 bank 映射；
//         4.3 block0 tiling 数据流 / block1 双缓冲时序；4.4 block0 3-stage 流水时序。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 gpu-p01 先例）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

// 手工路径上的脉冲（渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

// 显式箭头三角（kit 的 marker-end 被 CSS 关闭）
const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

// sector 条带：nSec 个 sector 框，每框 8 个 32-bit 字；wordCls(j, sec) 决定字的填充
const sectorStrip = (x, y, nSec, secW, gap, wordCls, { secLabel, wordH = 14 } = {}) => {
  const out = [];
  for (let s = 0; s < nSec; s++) {
    const sx = x + s * (secW + gap);
    out.push(`<g class="dia-node dia-node-cache"><rect x="${sx}" y="${y}" width="${secW}" height="${wordH + 14}" rx="6"/></g>`);
    const ww = (secW - 6) / 8;
    for (let j = 0; j < 8; j++) {
      out.push(`<rect class="${wordCls(j, s)}" x="${(sx + 3 + j * ww).toFixed(1)}" y="${y + 7}" width="${(ww - 1.2).toFixed(1)}" height="${wordH}" rx="1.5"/>`);
    }
    if (secLabel) out.push(cap(sx + secW / 2, y - 4, secLabel(s), { mono: true, size: 7.5, anchor: 'middle' }));
  }
  return out.join('');
};

/* ============================================================
 * 4.1 §2（block 0）— coalescing：distinct sector 判定
 * ============================================================ */
function coalescingSectors() {
  const UID = 'g4a';
  const body = [];

  body.push(cap(36, 24, 'transaction 数 = distinct 32 B sector 数（式 1）—— 跨度公式两头错：stride-16 高估 63 对 32、misaligned 低估 4 对 5', { size: 12, color: 'var(--ink-soft)' }));

  // s=1：4 sector 全有用
  body.push(cap(52, 86, 's=1 连续', { mono: true, size: 10 }));
  body.push(sectorStrip(150, 64, 4, 112, 10, () => 'dia-mx-on', { secLabel: (s) => `sector ${s}` }));
  body.push(pathPulse('M 152 78 H 626', { dur: 1.6, delay: 0 }));
  body.push(cap(684, 82, '4 sec', { mono: true, size: 9, anchor: 'end' }));
  body.push(cap(684, 96, 'η = 1', { mono: true, size: 9, anchor: 'end', color: 'var(--ink)' }));
  body.push(cap(150, 112, '→ 4 sector · 流量 128 B · η = 1：32 lane × 4 B 连续，有用 128 B（lane 0…7 / 8…15 / 16…23 / 24…31）', { size: 9 }));

  // s=2：8 sector 半有用（示前两个 + ⋯）
  body.push(cap(52, 158, 's=2 隔字', { mono: true, size: 10 }));
  body.push(sectorStrip(150, 136, 2, 112, 10, (j) => (j % 2 === 0 ? 'dia-mx-on' : 'dia-mx-warm'), { secLabel: (s) => `sector ${s}` }));
  body.push(pathPulse('M 152 150 H 378', { dur: 1.4, delay: 0.5 }));
  body.push(cap(398, 158, '⋯', { size: 12 }));
  body.push(cap(430, 158, '8 sector · 流量 256 B · η = 1/2（赭字 = 搬来但无用）', { size: 9.5 }));
  body.push(cap(150, 184, 'lane i 读字 2i → sector 号 ⌊2i/8⌋ = ⌊i/4⌋ 取遍 0–7；每 sector 8 个字只有 4 个有用 —— 与 BP「stride of 2 → 50%」逐字对上', { size: 9 }));

  // s=16：32 distinct sector，每 lane 独占（示前 8 个）
  body.push(cap(52, 230, 's=16', { mono: true, size: 10 }));
  body.push(cap(52, 244, '64 B stride', { mono: true, size: 8 }));
  body.push(sectorStrip(150, 208, 8, 52, 8, (j) => (j === 0 ? 'dia-mx-on' : 'dia-mx-off'), { secLabel: (s) => `${2 * s}`, wordH: 12 }));
  body.push(pathPulse('M 152 222 H 618', { dur: 1.8, delay: 1.0 }));
  body.push(cap(632, 226, '⋯', { size: 12 }));
  body.push(cap(684, 222, '32 sec', { mono: true, size: 9, anchor: 'end' }));
  body.push(cap(684, 236, 'η = 1/8', { mono: true, size: 9, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(150, 252, 'lane i → 字节 64i → sector 2i：编号 0, 2, 4, …, 62 共 32 个 distinct sector，每 lane 独占一个 · 流量 1024 B', { size: 9 }));
  body.push(cap(52, 274, '跨度 = 64×31 + 4 = 1988 B，⌈1988/32⌉ = 63 ≠ 32 —— 跨度把地址间的空洞也计成流量，高估近 2 倍（未触及的 sector 根本不搬）', { size: 10, color: 'var(--seal-deep)', weight: '650' }));

  // misaligned：连续 128 B 但基址偏移 4 B → 5 sector
  body.push(cap(52, 318, 'misaligned', { mono: true, size: 9.5 }));
  body.push(cap(52, 332, '基址偏移 4 B', { size: 8 }));
  body.push(sectorStrip(150, 296, 5, 96, 8, (j, s) => {
    if (s === 0) return j === 0 ? 'dia-mx-warm' : 'dia-mx-on';
    if (s === 4) return j === 0 ? 'dia-mx-on' : 'dia-mx-warm';
    return 'dia-mx-on';
  }, { secLabel: (s) => `sector ${s}` }));
  body.push(pathPulse('M 152 310 H 658', { dur: 1.6, delay: 1.5 }));
  body.push(cap(150, 344, '→ 区间 [4,132) 跨 sector 0–4 共 5 个；⌈128/32⌉ = 4 漏掉首尾残 sector —— 低估；η = 128/160 = 0.8（首尾残段各整体搬）', { size: 9 }));

  body.push(cap(52, 370, '两条不变量（BP 逐字）：只请求 sector 内部分字仍搬整 sector；lane 顺序任意置换 transaction 数不变 —— 数的是触及 sector 的集合', { size: 10 }));
  body.push(cap(52, 388, '跨度公式只在对齐基址 + stride ≤ 8 字的稠密访问下恰好成立（n_sec = 4s）；line 128 B 只是 tag 粒度、不参与流量账（2.2 §3）', { size: 10 }));

  return scene(720, 406, body, { label: 'coalescing 判定：stride-1 触及 4 个 sector 全有用，stride-2 触及 8 个半有用，stride-16 触及 32 个 distinct sector 而跨度公式给 63；misaligned 连续 128 B 跨 5 个 sector 而跨度公式给 4' }).svg();
}

/* ============================================================
 * 4.1 §3（afterHeading）— stride 饱和律 η = 1/min(s, 8)
 * ============================================================ */
function strideSaturation() {
  const UID = 'g4b';
  const body = [];

  body.push(cap(36, 24, 'stride 饱和律：η = 1/min(s, 8)（式 2）—— 饱和地板来自 32 条 lane，不来自 stride；无界 1/s 在 s ≥ 8 处错', { size: 12, color: 'var(--ink-soft)' }));

  // 左：有效有用带宽条（η·BW）
  body.push(cap(52, 60, '有效有用带宽 η·BW（对齐 fp32 warp load，有用恒 128 B）', { size: 9.5 }));
  const rows = [['s=1', 3352, 'soft'], ['s=2', 1676, 'warm'], ['s=4', 838, 'warm'], ['s=8', 419, 'seal'], ['s=16', 419, 'seal']];
  rows.forEach(([lab, v, tone], i) => {
    const y = 72 + i * 32;
    body.push(cap(170, y + 13, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone} sig-bar" style="--i:${(i * 0.15).toFixed(2)}" x="180" y="${y}" width="${(v * 0.125).toFixed(1)}" height="18" rx="3"/>`);
    body.push(cap(180 + v * 0.125 + 8, y + 13, `${v} GB/s`, { mono: true, size: 9 }));
  });
  body.push(`<line class="dia-edge dash" x1="232.4" y1="66" x2="232.4" y2="230"/>`);
  body.push(cap(232, 246, '32-sector 地板 419 GB/s', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 右：饱和机制
  body.push(cap(470, 116, 's ≥ 8：字节 stride 4s ≥ 32 B', { size: 9.5, weight: '650', color: 'var(--ink)' }));
  body.push(cap(470, 132, '→ 每 lane 独占一个 sector', { size: 9.5 }));
  body.push(sectorStrip(470, 144, 4, 44, 8, (j) => (j === 0 ? 'dia-mx-on' : 'dia-mx-off'), { wordH: 12 }));
  body.push(pathPulse('M 472 156 H 668', { dur: 1.6, delay: 0.8 }));
  body.push(cap(470, 188, 'sector 数被 warp 的 32 条 lane 封顶', { size: 9 }));
  body.push(cap(470, 204, 's=8 与 s=16 同价：饱和区内调 stride', { size: 9 }));
  body.push(cap(470, 218, '无带宽收益（改了白改）', { size: 9 }));
  body.push(cap(470, 240, '饱和区想提带宽只能改访问模式：', { size: 9.5, weight: '650', color: 'var(--ink)' }));
  body.push(cap(470, 256, 'SoA 布局（§5）/ shared 重排（4.2）', { size: 9 }));

  body.push(cap(52, 284, '一般元素宽度形：η = 1/min(s, 32/bytes) —— 饱和点 s·bytes = 32；fp64 在 s=4 饱和：s=2 → 1/2，s=4 与 8 都给 1/4', { mono: true, size: 10 }));
  body.push(cap(52, 302, 's=16 按无界 1/s 估 209.5 GB/s，实际地板 419 —— 低估 2 倍，反过来会高估「stride 16 改 8」的优化收益（两者同价）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 320, '全数字表（对齐 fp32）：s=1 → 4 sec/128 B，2 → 8/256，4 → 16/512，8 → 32/1024，16 → 32/1024 —— 流量封在 1024 B', { mono: true, size: 9.5 }));
  body.push(cap(52, 338, 'copy 带宽账（式 3）：t = 2·n_vec·bytes/(η·BW) —— 40.04 μs（s=1）对 320.33 μs（s≥8），8 倍差全由 sector 浪费支付', { size: 10 }));

  return scene(720, 356, body, { label: 'stride 饱和律：有效有用带宽从 s=1 的 3352 GB/s 按 stride 倒数下降，s=8 起饱和在 419 GB/s 地板；字节 stride 达到 32 B 后每 lane 独占一个 sector，sector 数被 32 条 lane 封顶' }).svg();
}

/* ============================================================
 * 4.2 §2（block 0）— bank 映射与冲突判定
 * ============================================================ */
function bankConflict() {
  const UID = 'g4c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '32 banks × 4 B：bank = ⌊a/4⌋ mod 32 —— k 路冲突串行 k 周期（k = max_b u_b），同址 broadcast 例外', { size: 12, color: 'var(--ink-soft)' }));

  // 字 → bank 映射条（w0..w31 → bank 0..31）
  for (let j = 0; j < 32; j++) {
    const x = 52 + j * 19;
    if (j % 4 === 0) body.push(cap(x + 8, 58, `w${j}`, { mono: true, size: 7, anchor: 'middle' }));
    body.push(`<g class="dia-node dia-node-data"><rect x="${x}" y="64" width="16" height="24" rx="3"/></g>`);
    body.push(cap(x + 8, 80, String(j), { mono: true, size: 7.5, anchor: 'middle', color: 'var(--ink)' }));
  }
  body.push(pathPulse('M 54 76 H 658', { dur: 2.0, delay: 0 }));
  body.push(cap(52, 108, '字 0–31 落 bank 0–31，字 32 回 bank 0 —— 每 128 B 一个 bank 循环；无冲突时 32 bank 并行，一拍服务整 warp = 128 B/cycle', { size: 9.5 }));

  // 四个判定案例
  const cases = [
    ['c0', 36, 'stride-1', '32 bank 各 1 字 · k=1', 'norm', '128 B/cycle'],
    ['c1', 200, 'stride-32 列访问', '全落同一 bank · k=32', 'attn', '4 B/cycle'],
    ['c2', 364, '同址 broadcast', '同一字一拍送全部 lane', 'norm', '一拍（multicast）'],
    ['c3', 528, '字 0 + 字 32', '同 bank 不同址 · k=2', 'op', '64 B/cycle'],
  ];
  cases.forEach(([id, x, title, sub, role, bw]) => {
    const n = node({ id, x, y: 130, w: 150, h: 60, role, title, sub, titleSize: 11, subSize: 8 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(x + 75, 208, bw, { mono: true, size: 9, anchor: 'middle', color: id === 'c1' ? 'var(--seal-deep)' : 'var(--ink-soft)' }));
  });
  body.push(cap(52, 236, 'stride-32：4 B/cycle @1.83 GHz = 7.3 GB/s < HBM 每 SM 份额 25.4 GB/s（3352/132）—— 把数据搬进 shared 不等于变快', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(52, 262, '冲突判定（命题 1）：u_b = bank b 内被请求的 distinct 字数（同址计 1），k = max u_b；硬件拆成 k 个 conflict-free 子请求、每请求一拍', { size: 10 }));
  body.push(cap(52, 280, '64-bit 账：double 连续 k=2 但 256 B/2 拍 = 128 B/cycle 峰值，非冲突损失 —— 冲突损失看 k 相对「有用字节 ÷ 128 B」下界多出几倍', { size: 10 }));
  body.push(cap(52, 298, '两套粒度：同一个 stride-16，global 饱和 η=1/8（32 sector 地板），shared k=16（16 路冲突、8 B/cycle = 14.6 GB/s）—— 勿互搬直觉', { size: 10 }));
  body.push(cap(52, 316, 'k 是服务周期（吞吐），与 29-cycle 延迟两笔账；NCU wavefronts/requests = 平均 k，excessive wavefronts 指示冲突损失', { size: 10 }));

  return scene(720, 334, body, { label: 'bank 映射与冲突判定：字地址除 4 模 32 得 bank 号，每 128 B 一个循环；stride-1 无冲突 128 B 每周期，stride-32 全落一 bank 串行 32 拍只有 4 B 每周期，同址 broadcast 一拍，同 bank 不同址仍冲突' }).svg();
}

/* ============================================================
 * 4.2 §4（afterHeading）— padding vs swizzle：两条消冲突路
 * ============================================================ */
function paddingSwizzle() {
  const UID = 'g4d';
  const body = [];

  body.push(cap(36, 24, '消冲突两条路：padding 改行距（gcd 定律）、swizzle 置换行内（异或双射）—— 向量化路径必须保 16 B 对齐', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(52, 52, 'k = gcd(s, 32)（式 3）：s=1→1、2→2、16→16、32→32；列访问 stride = tile 宽 w —— shared 没有饱和地板，只有数论结构', { mono: true, size: 10 }));

  // 左：padding
  body.push(zone({ x: 36, y: 68, w: 316, h: 228, label: 'padding：+1 列（标量路径）' }));
  body.push(cap(52, 104, 'tile[32][33]：stride 32→33，gcd(33,32)=1 → k=1', { mono: true, size: 9 }));
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 8; c++) {
      const x = 52 + c * 28 + r * 4, y = 116 + r * 22;
      const hot = c === 2;
      body.push(`<rect class="${hot ? 'dia-mx-hot' : 'dia-mx-off'}" x="${x}" y="${y}" width="26" height="16" rx="3"/>`);
      body.push(cap(x + 13, y + 12, String((r * 33 + c) % 32), { mono: true, size: 7.5, anchor: 'middle', color: hot ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
    }
  }
  body.push(cap(292, 158, '← 列访问', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 214, '列访问 bank 号 = (tx+ty) mod 32 取遍 32 bank ≡ unit stride', { size: 8.5 }));
  body.push(cap(52, 234, '代价：容量 +3.125%（4096→4224 B）；行距 132 B 破坏 16 B 对齐', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 252, 'p=2 反例：gcd(34,32)=2 仍 2 路 —— 有效条件 gcd(w+p,32)=1', { size: 8.5 }));
  body.push(cap(52, 270, 'float4/ldmatrix 路径失效（自然对齐违反 = UB，4.1 §4）', { size: 8.5, color: 'var(--seal-deep)' }));

  // 右：swizzle
  body.push(zone({ x: 368, y: 68, w: 316, h: 228, label: 'swizzle：Swizzle<3,4,3>（向量化/MMA 路径）' }));
  body.push(cap(384, 104, 'off_sw = off ⊕ ((off≫7) mod 8)·16（式 4）', { mono: true, size: 9 }));
  body.push(cap(384, 120, '行号 XOR 进 16 B chunk 号：第 r 行数据落 chunk r', { size: 8.5 }));
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const x = 384 + c * 28, y = 130 + r * 15;
      body.push(`<rect class="${c === r ? 'dia-mx-hot' : 'dia-mx-off'}" x="${x}" y="${y}" width="26" height="13" rx="2"/>`);
    }
  }
  body.push(cap(620, 188, '← 8 行散到', { size: 8.5 }));
  body.push(cap(620, 202, '8 个 chunk', { size: 8.5 }));
  body.push(cap(384, 262, '8 行 × 4 字覆盖 32 bank 各 1 字：ldmatrix k=8→1（8 倍服务周期差）', { size: 8.5 }));
  body.push(cap(384, 280, '双射且对合：施加两次还原；保 128 B line 归属与 16 B 对齐', { size: 8.5 }));

  // transpose 账
  body.push(cap(52, 316, '教学 transpose（32×32 tile、32 warps）每 tile shared 服务周期（行写 k=1 + 列读 k）：', { size: 9.5 }));
  body.push(cap(240, 338, 'naive [32][32]', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal sig-bar" style="--i:0" x="250" y="326" width="316.8" height="16" rx="3"/>`);
  body.push(cap(575, 338, '1056 周期 · 577.0 ns', { mono: true, size: 9 }));
  body.push(cap(240, 364, 'padded [32][33]', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-ink sig-bar" style="--i:0.2" x="250" y="352" width="19.2" height="16" rx="3"/>`);
  body.push(cap(278, 364, '64 周期 · 34.97 ns —— 16.5×（列读 4 对 128 B/cycle）', { mono: true, size: 9 }));

  body.push(cap(52, 390, 'BP 官方三级锚（V100）：12.8 → 140.2 → 199.4 GB/s = global 合并 10.95× + padding 1.42×；官方例冲突在列写侧、transpose 在列读侧，同一笔账', { size: 9 }));
  body.push(cap(52, 408, 'TMA 硬件 swizzle：tensor-map 带 32/64/96/128 B 模式，硬件直接按 swizzle 布局写 shared（4.4）；CUTLASS swizzle layout 走读归 5.2', { size: 9.5 }));

  return scene(720, 426, body, { label: 'padding 与 swizzle 两条消冲突路：padding 加一列使 stride 33 与 32 互素，列访问跨 bank 无冲突但破坏 16 B 对齐；swizzle 行号异或进 chunk 号，双射对合保对齐，ldmatrix 冲突从 8 路降到 1；transpose 每 tile 1056 对 64 周期' }).svg();
}

/* ============================================================
 * 4.3 §3（block 0）— GEMM tiling 数据流：global→shared→register
 * ============================================================ */
function gemmTiling() {
  const UID = 'g4e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'GEMM tiling：一次装载服务整个 tile 的消费 —— 复用因子 1 → O(T)，强度 0.5 → 64 FLOP/byte（bf16，T_tile=128）', { size: 12, color: 'var(--ink-soft)' }));

  const hbm = node({ id: 'hbm', x: 36, y: 64, w: 110, h: 64, role: 'data', title: 'HBM', sub: 'A/B 矩阵', titleSize: 13, subSize: 9 });
  nodes.push(hbm); body.push(hbm.svg);
  body.push(zone({ x: 184, y: 48, w: 182, h: 124, label: 'shared（每 K 步 16 KB）' }));
  const at = node({ id: 'at', x: 196, y: 74, w: 158, h: 40, role: 'cache', title: 'A tile 128×32', sub: 'HBM 读一次', titleSize: 11, subSize: 8 });
  const bt = node({ id: 'bt', x: 196, y: 122, w: 158, h: 40, role: 'cache', title: 'B tile 32×128', sub: 'HBM 读一次', titleSize: 11, subSize: 8 });
  const reg = node({ id: 'reg', x: 420, y: 64, w: 150, h: 64, role: 'proj', title: 'register fragment', sub: '每线程 8 A 值 + 8 B 值', titleSize: 11.5, subSize: 8.5 });
  const mma = node({ id: 'mma', x: 610, y: 64, w: 74, h: 64, role: 'attn', title: 'MMA', sub: '外积', titleSize: 12, subSize: 8 });
  const ct = node({ id: 'ct', x: 420, y: 180, w: 150, h: 52, role: 'attn', title: 'C tile 128×128', sub: '64 fp32 累加器/线程', titleSize: 11.5, subSize: 8 });
  const wb = node({ id: 'wb', x: 610, y: 180, w: 74, h: 52, role: 'plain', title: '写 HBM', sub: '末步一次', titleSize: 9.5, subSize: 7.5 });
  nodes.push(at, bt, reg, mma, ct, wb);
  body.push(at.svg, bt.svg, reg.svg, mma.svg, ct.svg, wb.svg);

  body.push(edge(hbm, at, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'a' }));
  body.push(arrowHead(at.left[0], at.left[1], 1, 0));
  body.push(edge(hbm, bt, { nodes, portFrom: hbm.right, portTo: bt.left, pulse: true, dur: 1.3, delay: 0.3, uid: UID, eid: 'b' }));
  body.push(arrowHead(bt.left[0], bt.left[1], 1, 0));
  body.push(edge(at, reg, { nodes, pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: 'c' }));
  body.push(edge(bt, reg, { nodes, portFrom: bt.right, portTo: reg.left, pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: 'd' }));
  body.push(arrowHead(reg.left[0], reg.left[1], 1, 0));
  body.push(edge(reg, mma, { nodes, pulse: true, dur: 1.1, delay: 1.2, uid: UID, eid: 'e' }));
  body.push(arrowHead(mma.left[0], mma.left[1], 1, 0));
  body.push(edge(mma, ct, { nodes, portFrom: mma.bottom, portTo: ct.top, pulse: true, dur: 1.2, delay: 1.6, uid: UID, eid: 'f' }));
  body.push(arrowHead(ct.top[0], ct.top[1], 0, 1));
  body.push(edge(ct, wb, { nodes, pulse: true, dur: 1.1, delay: 2.0, uid: UID, eid: 'g' }));
  body.push(arrowHead(wb.left[0], wb.left[1], 1, 0));

  body.push(cap(184, 190, 'A 每元素被 T_N = 128 个输出消费', { size: 8.5 }));
  body.push(cap(184, 204, 'B 每元素被 T_M = 128 个输出消费', { size: 8.5 }));
  body.push(cap(684, 152, 'K 方向累加', { size: 8.5, anchor: 'end' }));
  body.push(cap(52, 244, 'K/T_K = 128 步：每步装载 (T_M+T_N)·T_K 元素、计算 2·T_M·T_N·T_K FLOPs；C 累加器全程驻留 register，末步写回一次', { size: 9 }));

  body.push(cap(52, 262, 'I_blk = 2·T_M·T_N/((T_M+T_N)·bytes) = T_tile/bytes（式 2，方阵 tile）；T_K 约去 —— 强度与 K、T_K 无关', { mono: true, size: 10, weight: '650' }));
  body.push(cap(170, 286, 'naive', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="274" width="2.5" height="16" rx="1"/>`);
  body.push(cap(190, 286, '0.5 FLOP/byte · 274.88 GB · 82.00 ms（O(1) 复用，结构性 memory-bound）', { mono: true, size: 9 }));
  body.push(cap(170, 312, 'tiled 128', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal sig-bar" style="--i:0.3" x="180" y="300" width="320" height="16" rx="3"/>`);
  body.push(cap(508, 312, 'I=64 · 2.147 GB · 640.7 μs · 128×', { mono: true, size: 9 }));

  body.push(cap(52, 336, 'ridge R ≈ 295：I=64 仍 memory-bound（per-block、无 L2 复用）；达 ridge 需 T_tile ≥ 590 超预算 —— 真实 GEMM 靠 L2 跨 block 复用进 compute-bound（5.2）', { size: 9 }));
  body.push(cap(52, 356, '总流量（式 3）= MNK(1/T_M + 1/T_N)·bytes：4096³ × (2/128) × 2 = 2.147 GB；C 写回 33.55 MB 占 1.56%，低阶项', { mono: true, size: 9.5 }));
  body.push(cap(52, 374, 'BP 逐字：收益来自避免冗余搬运（复用），不是来自合并 —— coalescing 治「一次搬运浪费几个 sector」，tiling 治「同一数据搬几次」，两笔账正交', { size: 9.5 }));
  body.push(cap(52, 392, 'tile 扫描：T=32 → I=16（8.590 GB）、64 → 32（4.295 GB）、128 → 64（2.147 GB）、256 → 128（1.074 GB）—— 强度随边长线性升', { size: 9.5 }));

  return scene(720, 410, body, { label: 'GEMM tiling 数据流：HBM 的 A/B tile 每 K 步读一次进 shared，A 每元素被 128 个输出消费，经寄存器 fragment 喂 MMA 外积，C 累加器驻留 128 步末步写回；强度从 naive 0.5 抬到 64 FLOP/byte，流量 274.88 GB 降到 2.147 GB' }).svg();
}

/* ============================================================
 * 4.3 §6（block 1）— double buffering 时序
 * ============================================================ */
function doubleBuffer() {
  const UID = 'g4f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '双缓冲：计算 tile k 的同时预取 k+1 —— τ_db = max(t_comp, t_lat+t_load)，藏的是等待、不是流量', { size: 12, color: 'var(--ink-soft)' }));

  const tc = node({ id: 'tc', x: 36, y: 52, w: 180, h: 44, role: 'attn', title: 't_comp = 139.9 ns', sub: '2·T_M·T_N·T_K / F_SM', mono: true, titleSize: 11.5, subSize: 8 });
  const tl = node({ id: 'tl', x: 250, y: 52, w: 180, h: 44, role: 'cache', title: 't_load = 645.2 ns', sub: '16 KB ÷ 25.4 GB/s 份额', mono: true, titleSize: 11.5, subSize: 8 });
  const tt = node({ id: 'tt', x: 464, y: 52, w: 180, h: 44, role: 'proj', title: 't_lat = 261.6 ns', sub: '478.8 cycles @1.83 GHz [B]', mono: true, titleSize: 11.5, subSize: 8 });
  nodes.push(tc, tl, tt); body.push(tc.svg, tl.svg, tt.svg);
  body.push(cap(52, 118, '装载完成时间 ≈ t_lat + t_load = 906.8 ns（首字节延迟 + 传输服务，上界口径；单份 16 KB 在飞已超 6.6 KB 带宽延迟积）', { mono: true, size: 9.5 }));

  // 同步版时序（0.42 px/ns）
  body.push(cap(140, 152, '同步版', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="150" y="136" width="380.9" height="20" rx="3"/>`);
  body.push(cap(340, 150, '装载 tile k（t_lat + t_load）', { size: 8.5, anchor: 'middle', color: 'var(--ink)' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="530.9" y="136" width="58.8" height="20" rx="3"/>`);
  body.push(cap(560, 150, '计算 k', { size: 8, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(`<rect x="589.7" y="136" width="60" height="20" rx="3" fill="none" stroke="rgb(24 27 32 / 0.22)" stroke-dasharray="4 4"/>`);
  body.push(cap(620, 150, '装载 k+1…', { size: 7.5, anchor: 'middle' }));
  body.push(cap(150, 174, 'τ_sync = 1046.7 ns/tile —— 计算只占 13.4%，等待全暴露', { size: 9 }));

  // 双缓冲时序
  body.push(cap(140, 212, '计算', { size: 9, anchor: 'end' }));
  body.push(cap(140, 244, '装载', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="150" y="196" width="58.8" height="20" rx="3"/>`);
  body.push(cap(179, 210, 'tile k', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(`<rect x="208.8" y="196" width="322.1" height="20" rx="3" fill="none" stroke="rgb(24 27 32 / 0.22)" stroke-dasharray="4 4"/>`);
  body.push(cap(220, 210, '等装载完成（766.9 ns）', { size: 8 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="530.9" y="196" width="58.8" height="20" rx="3"/>`);
  body.push(cap(560, 224, 'tile k+1', { size: 7, anchor: 'middle' }));
  body.push(cap(600, 210, '⋯', { size: 11 }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="150" y="228" width="380.9" height="20" rx="3"/>`);
  body.push(cap(160, 242, '装载 tile k+1（算 k 之初发射）', { size: 8.5, color: 'var(--ink)' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="530.9" y="228" width="119" height="20" rx="3"/>`);
  body.push(cap(590, 242, 'tile k+2…', { size: 8, anchor: 'middle', color: 'var(--ink)' }));
  body.push(pathPulse('M 152 238 H 648', { dur: 2.2, delay: 0.4 }));
  body.push(pathPulse('M 152 206 H 588', { dur: 2.2, delay: 0 }));
  body.push(cap(150, 268, '稳态 τ_db = max(139.9, 906.8) = 906.8 ns（装载主导）—— 加速 1046.7/906.8 = 1.154，上界 2（t_lat→0 且两时间平衡）', { size: 9.5, weight: '650' }));

  body.push(cap(52, 292, '完全隐藏 iff t_comp ≥ t_lat+t_load，必要条件 I_blk ≥ R —— 本例 I=64 ≪ 295，双缓冲只藏掉计算那一段；T_K=64：1831.8/1552.0 = 1.180', { size: 10 }));
  body.push(cap(52, 310, '容量代价：S_sh 16→32 KB，shared 项 13→6 blocks；register 项 2 更紧、b_act 不变 —— register 绑定下 occupancy 无损（容量仍实付）', { size: 10 }));
  body.push(cap(52, 328, '坑：双缓冲不是带宽优化 —— max 里 t_load 地板纹丝不动；load-bound 提强度（大 tile/L2）或加 stage（4.4：3-stage 645.2 ns，1.62×）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 346, body, { label: '双缓冲时序：同步版每 tile 装载 906.8 ns 加计算 139.9 ns 串行共 1046.7 ns；双缓冲计算 tile k 与装载 tile k+1 重叠，稳态每 tile 取两者最大值 906.8 ns，加速 1.154' }).svg();
}

/* ============================================================
 * 4.4 §4（block 0）— 3-stage 流水时序
 * ============================================================ */
function multiStagePipeline() {
  const UID = 'g4g';
  const body = [];

  body.push(cap(36, 24, '3-stage 流水（S=3：2 份预取 + 1 份计算）：算 tile k 之初发射 tile k+2 —— 2 份装载在飞，完成被 (S−1)τ_S 盖住（带宽主导）', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(52, 52, '序章：装 tile 0、tile 1（填满预取份）；首 tile 装载无计算可藏，填充段占 K/T_K = 128 步的约 1/128，可略', { size: 9.5 }));

  // 时序（130 px/tile 稳态槽；t_comp 28.2 px、t_lat+t_load 182.7 px）
  body.push(cap(140, 86, '计算', { size: 9, anchor: 'end' }));
  body.push(cap(140, 138, '装载', { size: 9, anchor: 'end' }));
  for (let i = 0; i < 4; i++) {
    const x = 150 + i * 130;
    body.push(cap(x + 14, 68, `tile ${i}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(`<rect class="dia-bar dia-bar-seal" x="${x}" y="72" width="28.2" height="20" rx="3"/>`);
    if (i < 3) body.push(`<rect x="${x + 28.2}" y="72" width="101.8" height="20" rx="3" fill="none" stroke="rgb(24 27 32 / 0.16)" stroke-dasharray="4 4"/>`);
  }
  body.push(cap(229, 86, '等下一份 full 缓冲', { size: 7.5, anchor: 'middle' }));
  body.push(cap(150, 108, '计算 139.9 ns/tile；稳态槽余量等缓冲 —— τ 由 t_load 地板封顶', { size: 8.5 }));
  body.push(cap(150, 122, '算 tile k 之初发射 tile k+2（缓冲 (k−1) mod S 刚释放）', { size: 8.5 }));
  const loads = [[150, 182.7, 'tile 2'], [280, 182.7, 'tile 3'], [410, 182.7, 'tile 4'], [540, 109.9, 'tile 5…']];
  loads.forEach(([x, w, lab], i) => {
    body.push(`<rect class="dia-bar dia-bar-warm" x="${x}" y="124" width="${w}" height="20" rx="3"/>`);
    // tile 4 标签靠左：3.5s 快照脉冲落点（≈x490）不压字
    if (i === 2) body.push(cap(x + 8, 138, lab, { mono: true, size: 8, color: 'var(--ink)' }));
    else body.push(cap(x + w / 2, 138, lab, { mono: true, size: 8, anchor: 'middle', color: 'var(--ink)' }));
  });
  body.push(pathPulse('M 152 134 H 648', { dur: 2.4, delay: 0.3 }));
  body.push(cap(52, 168, '恒有 2 份装载重叠：2 × 645.2 = 1290.4 ≥ 906.8 —— 不等待条件 (S−1)·τ_S ≥ t_lat + t_load 满足', { size: 9.5, weight: '650' }));

  body.push(cap(52, 194, 'τ_S = max(t_comp, t_load, (t_lat+t_load)/(S−1))（式 1）：S=2 → 906.8；S=3 → max(139.9, 645.2, 453.4) = 645.2 ns', { mono: true, size: 9.5 }));
  body.push(cap(52, 212, 'S_min = 1 + ⌈906.8/645.2⌉ = 3（式 2）—— 带宽地板由强度决定（4.3 式 2），流水只能逼近、不能突破；再加 stage 无收益', { size: 10, weight: '650' }));

  // 加速对比条（0.28 px/ns）
  [['同步版', 1046.7, 'soft', 'ns/tile（基准）'], ['双缓冲 S=2', 906.8, 'warm', '1.154×'], ['3-stage S=3', 645.2, 'seal', '1.622× 对同步']].forEach(([lab, v, tone, note], i) => {
    const y = 228 + i * 26;
    body.push(cap(240, y + 12, lab, { mono: true, size: 9, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone} sig-bar" style="--i:${(i * 0.2).toFixed(1)}" x="250" y="${y}" width="${(v * 0.28).toFixed(1)}" height="16" rx="3"/>`);
    body.push(cap(250 + v * 0.28 + 8, y + 12, `${v} ns · ${note}`, { mono: true, size: 9 }));
  });

  body.push(cap(52, 326, '延迟主导推论：预取 stage 数 × t_comp ≥ t_lat —— T_K=32 需 S≥3（261.6/139.9 = 1.87）；T_K=64 需 S≥2（279.8 ≥ 261.6 盖住首字节延迟）', { size: 9.5 }));
  body.push(cap(52, 344, 'stage 容量：S_sh = S·m_tile = 48 KB（T_K=32，shared 项 4 blocks）/ 96 KB（T_K=64）；register 项 2 更紧 → O=0.25 不变，免费加到 shared 触顶', { size: 9 }));
  body.push(cap(52, 362, '两级 overlap 正交：3.3 stream 级把 H2D 藏进 kernel 执行（copy engine）；本节 tile 级把 HBM 装载藏进 tile 计算（cp.async/TMA）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 380, body, { label: '3-stage 流水时序：序章装满两份预取，计算 tile 0 到 3 每槽 645.2 ns，装载 tile 2 起错开发射恒有两份在飞；稳态取计算、带宽、摊薄完成时间三项最大值 645.2 ns，对同步加速 1.622' }).svg();
}

/* ============================================================
 * 4.4 §3（afterHeading）— TMA descriptor + mbarrier tx-count
 * ============================================================ */
function tmaMbarrier() {
  const UID = 'g4h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'TMA：descriptor 驱动的硬件搬运 —— elected 单线程发 2 条指令（A/B 各一），mbarrier tx-count 按字节记账（cp.async 要 1024 条）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 300, h: 150, label: 'cp.async（sm_80+）：软件逐线程' }));
  body.push(cap(52, 84, '256 线程 × 4 条 = 1024 条 cp.async', { mono: true, size: 9.5, weight: '650', color: 'var(--ink)' }));
  body.push(cap(52, 100, '每条 16 B（4/8/16 粒度）· 每条附地址算术', { size: 9 }));
  body.push(cap(52, 118, 'global→shared 免中间 register（降 register 压力）', { size: 9 }));
  body.push(cap(52, 136, '完成：commit_group 打包 · wait_group N 留最新 N 组', { size: 9 }));
  body.push(cap(52, 154, '组间无序；16 B 粒度可 .cg bypass L1', { size: 9 }));
  body.push(cap(52, 172, '地址生成开销随 tile 字节线性涨 → TMA 的动机', { size: 9, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 368, y: 48, w: 316, h: 150, label: 'TMA（sm_90+）：descriptor 硬件' }));
  const tm = node({ id: 'tm', x: 384, y: 74, w: 130, h: 44, role: 'cache', title: 'tensor-map 128 B', sub: 'host 侧一次性创建', titleSize: 10.5, subSize: 8 });
  const el = node({ id: 'el', x: 534, y: 74, w: 134, h: 44, role: 'attn', title: 'elected 单线程', sub: '2 条 bulk.tensor · A/B 各一', titleSize: 10.5, subSize: 7.5 });
  nodes.push(tm, el); body.push(tm.svg, el.svg);
  body.push(edge(tm, el, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 't' }));
  body.push(arrowHead(el.left[0], el.left[1], 1, 0));
  body.push(cap(384, 140, '1D–5D 张量 · stride/offset/边界全硬件接管', { size: 9 }));
  body.push(cap(384, 158, '硬件 swizzle 32/64/96/128 B · cluster 内 multicast', { size: 9 }));
  body.push(cap(384, 176, 'cuTensorMapEncodeTiled → __grid_constant__ 参数', { mono: true, size: 8 }));

  // mbarrier 记账链
  const ex = node({ id: 'ex', x: 36, y: 224, w: 150, h: 48, role: 'norm', title: 'expect_tx 16384 B', sub: 'A 8192 + B 8192（每 stage）', mono: true, titleSize: 10.5, subSize: 8 });
  const mv = node({ id: 'mv', x: 222, y: 224, w: 150, h: 48, role: 'op', title: 'TMA 硬件搬运', sub: '完成自动 complete_tx 扣减', titleSize: 11, subSize: 8 });
  const mb = node({ id: 'mb', x: 408, y: 224, w: 150, h: 48, role: 'attn', title: 'mbarrier tx-count', sub: '字节计数归零 + 到达数齐', titleSize: 10.5, subSize: 8 });
  const cs = node({ id: 'cs', x: 594, y: 224, w: 90, h: 48, role: 'proj', title: '消费者', sub: 'try_wait 放行', titleSize: 10.5, subSize: 7.5 });
  nodes.push(ex, mv, mb, cs); body.push(ex.svg, mv.svg, mb.svg, cs.svg);
  [[ex, mv, 0.4], [mv, mb, 0.8], [mb, cs, 1.2]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `b${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(52, 296, 'tx-count 本质是字节计数（wp 逐字）—— 与 4.3 强度账的流量同一本账；Hopper 等待 sleep 对前代 spin（不烧发射槽）', { size: 9.5 }));

  body.push(cap(52, 320, '发射账：cp.async 1024 条（地址软件）对 TMA 2 条（A/B 各一，地址硬件）—— tile 越大差距越大；descriptor 2×128 B 占流量 1.2×10⁻⁷', { mono: true, size: 9.5 }));
  body.push(cap(52, 344, '完成只认 async-group 与 mbarrier（PTX 逐字）：__syncthreads() 不保证 cp.async 完成 —— 先 wait 再 barrier，两步缺一不可（wait 是 per-thread）', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(52, 362, '适用：大 tile/多维/边界多用 TMA（GEMM/FA/im2col）；细粒度/散乱/sm_80 兼容用 cp.async；不规则小张量 descriptor 建了不值', { size: 10 }));
  body.push(cap(52, 380, 'CUTLASS PipelineTmaAsync：full/empty barrier Stages 份各，producer_acquire/expect/commit 对 consumer_wait/release；commit 可 No-Op（5.2）', { size: 9.5 }));

  return scene(720, 398, body, { label: 'TMA 与 mbarrier：cp.async 需 1024 条软件逐线程指令，TMA 由 128 B tensor-map 描述符加 elected 单线程两条指令（A/B 各一）硬件搬运；expect_tx 记入 16384 B（A 8192 + B 8192），硬件完成自动扣减，tx-count 归零且到达数齐才放行消费者' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/01-coalescing-vectorization',
    replacesBlock: 0,
    title: 'coalescing 判定：distinct sector 数 —— 跨度公式 63 对 32、4 对 5',
    svg: coalescingSectors,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/01-coalescing-vectorization',
    afterHeading: /^## 3\. /,
    title: 'stride 饱和律：η = 1/min(s, 8) —— 32-sector 地板 419 GB/s',
    svg: strideSaturation,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/02-shared-memory-bank-conflicts',
    replacesBlock: 0,
    title: 'bank 映射与冲突判定：k = max u_b —— stride-32 只剩 7.3 GB/s',
    svg: bankConflict,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/02-shared-memory-bank-conflicts',
    afterHeading: /^## 4\. /,
    title: 'padding vs swizzle：gcd 定律与异或双射 —— transpose 1056 对 64 周期',
    svg: paddingSwizzle,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/03-tiling-data-reuse',
    replacesBlock: 0,
    title: 'GEMM tiling 数据流：复用 1 → O(T)，强度 0.5 → 64 FLOP/byte',
    svg: gemmTiling,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/03-tiling-data-reuse',
    replacesBlock: 1,
    title: 'double buffering：τ = max(t_comp, t_lat+t_load) —— 加速 1.154',
    svg: doubleBuffer,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/04-async-copy-pipelining',
    replacesBlock: 0,
    title: '3-stage 流水：2 份在飞盖住 906.8 ns —— 带宽地板 645.2 ns（1.622×）',
    svg: multiStagePipeline,
  },
  {
    dim: 'gpu',
    slug: '04-kernel-memory-optimization/04-async-copy-pipelining',
    afterHeading: /^## 3\. /,
    title: 'TMA descriptor + mbarrier：1024 条对 2 条，tx-count 按字节记账',
    svg: tmaMbarrier,
  },
];