// gpu P2（02-memory-hierarchy）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs / gpu-p01.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）。
// census：2.1 block0 HBM 栈侧视+顶视；2.2 block0 层级路径 / block1 line/sector；
//         2.3 block0 register operand 供给教学模型；2.4 block0 三区判定流程 / block1 实测 roofline。

import { scene, node, edge, zone, cap as kitCap, bars, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 inference 先例）。
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

/* ============================================================
 * 2.1 §2（block 0）— HBM 栈几何 → 带宽四因子乘积
 * ============================================================ */
function hbmStack() {
  const UID = 'g2a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'HBM3 栈几何 → 3352 GB/s：5 栈 × 16 通道 × 5.238 Gbps × 64 bit ÷ 8 —— 宽总线换带宽', { size: 12, color: 'var(--ink-soft)' }));

  // 侧视：8-high DRAM + base logic die + TSV
  body.push(zone({ x: 36, y: 44, w: 250, h: 216, label: '侧视：8-high DRAM die + base logic die' }));
  for (let i = 0; i < 8; i++) {
    const y = 68 + i * 18;
    body.push(`<g class="dia-node dia-node-data"><rect x="52" y="${y}" width="160" height="14" rx="4"/></g>`);
    if (i === 0) body.push(cap(58, y + 11, 'DRAM die 7', { mono: true, size: 8 }));
    if (i === 7) body.push(cap(58, y + 11, 'DRAM die 0', { mono: true, size: 8 }));
  }
  body.push(cap(132, 140, '⋮', { size: 11, anchor: 'middle' }));
  body.push(`<g class="dia-node dia-node-op"><rect x="52" y="216" width="160" height="24" rx="6"/></g>`);
  body.push(cap(58, 232, 'base logic die（PHY/测试）', { size: 8.5 }));
  body.push(`<line class="dia-edge strong" x1="192" y1="68" x2="192" y2="240"/>`);
  body.push(pathPulse('M 192 68 V 240', { dur: 2.0, delay: 0.2 }));
  body.push(cap(238, 140, 'TSV', { mono: true, size: 9 }));
  body.push(cap(238, 154, '垂直互连', { size: 8.5 }));
  body.push(cap(161, 254, 'microbump → 硅中介层（CoWoS）', { size: 8.5, anchor: 'middle' }));
  body.push(`<path class="dia-edge dash" d="M 161 260 V 272"/>`);
  body.push(cap(161, 284, 'GPU die（MC×10，每栈 2 个）', { size: 9, anchor: 'middle' }));

  // 顶视：16 channel × 64 bit → 2 MC
  body.push(zone({ x: 320, y: 44, w: 364, h: 160, label: '顶视（每栈）：1024-bit = 16 channel × 64 bit' }));
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 8; c++) {
      const k = r * 8 + c;
      const x = 336 + c * 44, y = 76 + r * 36;
      body.push(`<g class="dia-node dia-node-cache"><rect x="${x}" y="${y}" width="40" height="30" rx="6"/></g>`);
      body.push(cap(x + 20, y + 19, `ch${k}`, { mono: true, size: 8.5, anchor: 'middle' }));
    }
  }
  body.push(cap(510, 160, 'ch0–7 → MC 0（512-bit）', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(510, 176, 'ch8–15 → MC 1（512-bit）', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(336, 194, '每 channel 再分 2 个 pseudo-channel（各 32 bit），全栈 32 个', { size: 8.5 }));

  // 带宽四因子乘积链
  const fac = [
    ['f0', 36, 110, '5 栈', 'n_stack（披露）'],
    ['f1', 166, 120, '16 ch/栈', 'n_ch（Micron 互证）'],
    ['f2', 306, 140, '5.238 Gbps/pin', '2619 MHz DDR ×2'],
    ['f3', 466, 110, '64 bit/ch', 'w_ch（每通道口径）'],
  ].map(([id, x, w, title, sub]) => {
    const n = node({ id, x, y: 300, w, h: 56, role: 'proj', title, sub, mono: true, titleSize: 12, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const res = node({ id: 'res', x: 596, y: 300, w: 88, h: 56, role: 'attn', title: '3352', sub: 'GB/s（式 2）', mono: true, titleSize: 15, subSize: 8.5 });
  nodes.push(res); body.push(res.svg);
  for (let i = 0; i < 3; i++) {
    body.push(edge(fac[i], fac[i + 1], { nodes, pulse: true, dur: 1.1, delay: i * 0.3, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(fac[i + 1].left[0], fac[i + 1].left[1], 1, 0));
    body.push(cap((fac[i].right[0] + fac[i + 1].left[0]) / 2, 318, '×', { size: 11, anchor: 'middle', color: 'var(--ink-faint)' }));
  }
  body.push(edge(fac[3], res, { nodes, pulse: true, dur: 1.1, delay: 0.9, uid: UID, eid: 'm3' }));
  body.push(arrowHead(res.left[0], res.left[1], 1, 0));
  body.push(cap(586, 318, '÷8', { size: 10, anchor: 'middle', color: 'var(--ink-faint)' }));

  body.push(cap(52, 380, '总线宽 5×16×64 = 5120-bit —— 与 Table 3「Memory Interface 5120-bit HBM3」逐字一致', { mono: true, size: 10 }));
  body.push(cap(52, 398, '等价每栈口径式 (3)：BW = n_stack·f_pin·1024/8 同值（1024 = n_ch·w_ch，两式不混代）', { mono: true, size: 10 }));
  body.push(cap(52, 416, '坑：1024 代入含 n_ch 的式 (1) 把通道数算两遍 ⇒ 高估 16×（53.6 TB/s 荒谬值是自检信号）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 434, 'vendor 互证：6.4 Gbps → 819.2 GB/s/栈（Samsung 逐字）；H200 6 栈 / 6.25 Gbps 反推 → 4.8 TB/s', { size: 10 }));

  return scene(720, 452, body, { label: 'HBM3 栈几何：侧视 8 层 DRAM die 经 TSV 到 base logic die，顶视 16 通道分属两个 MC；带宽等于 5 栈乘 16 通道乘 5.238 Gbps 乘 64 bit 除 8' }).svg();
}

/* ============================================================
 * 2.2 §2（block 0）— 内存层级金字塔：容量-延迟-带宽
 * ============================================================ */
function cacheHierarchy() {
  const UID = 'g2b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '内存层级金字塔：每往下一层容量大一个量级以上、延迟逐级抬升 —— 没有「又快又大」的层', { size: 12, color: 'var(--ink-soft)' }));

  const tiers = [
    ['t0', 210, 180, 56, 'proj', 'register file · 256 KB/SM', 'operand 管线内 · 4055 GB/s/SM 上界'],
    ['t1', 170, 260, 116, 'cache', 'shared · 228 KB/SM 最大档', '29.0 cycles [B] · 128 B/cycle = 253 GB/s'],
    ['t2', 130, 340, 176, 'cache', 'L1 · 256 KB 统一 − carveout', '40.7 cycles [B] · 每周期带宽官方未披露'],
    ['t3', 85, 430, 236, 'cache', 'L2 · 50 MB（两分区各 25 MB）', '263.0 cycles [B] · 全芯片共享'],
    ['t4', 40, 520, 296, 'data', 'HBM3 · 80 GB（5 栈 × 16 ch）', '478.8 cycles [B] · 3352 GB/s（每 SM 份额 25.4）'],
  ].map(([id, x, w, y, role, title, sub]) => {
    const n = node({ id, x, y, w, h: 52, role, title, sub, titleSize: 12, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // miss 下行路径（sector 32 B 粒度）
  const miss1 = 'M 130 202 C 104 214, 85 224, 85 236';
  const miss2 = 'M 85 262 C 62 274, 40 284, 40 296';
  body.push(`<path class="dia-edge strong" d="${miss1}"/>`);
  body.push(`<path class="dia-edge strong" d="${miss2}"/>`);
  body.push(pathPulse(miss1, { dur: 1.4, delay: 0.4 }));
  body.push(pathPulse(miss2, { dur: 1.4, delay: 1.2 }));
  body.push(arrowHead(85, 236, 0, 1));
  body.push(arrowHead(40, 296, 0, 1));
  body.push(cap(40, 218, 'L1 miss', { size: 8.5 }));
  body.push(cap(44, 258, 'L2 miss', { size: 8.5 }));

  // 作用域括注
  body.push(`<line class="dia-edge" x1="572" y1="56" x2="572" y2="228"/>`);
  body.push(cap(580, 140, '每 SM 私有', { size: 9.5 }));
  body.push(cap(580, 156, '×132 份', { mono: true, size: 9.5 }));
  body.push(`<line class="dia-edge" x1="572" y1="236" x2="572" y2="348"/>`);
  body.push(cap(580, 288, '全芯片共享', { size: 9.5 }));
  body.push(cap(580, 304, '（一份）', { size: 9.5 }));

  body.push(cap(40, 362, '左侧下行 = miss 路径（L1 miss 按 sector 32 B 粒度向下取数）', { size: 9 }));
  body.push(cap(52, 380, '片上比值（per-SM 同口径）：register : shared : HBM 份额 = 4055 : 253 : 25.4 ≈ 160 : 10 : 1', { mono: true, size: 10 }));
  body.push(cap(52, 398, '归一化：私有层 ×132 得芯片聚合（register ≈535 TB/s）；共享层 ÷132 得每 SM 份额 —— 混比差 132×', { size: 10 }));
  body.push(cap(52, 416, 'kernel 主线：把工作集沿金字塔往上搬 —— tiling 搬进 shared/register（4.3）、persisting 留在 L2（§5）', { size: 10 }));

  return scene(720, 432, body, { label: '内存层级金字塔：register、shared、L1、L2、HBM 五层容量递增延迟递增，miss 按 sector 粒度下行，私有层与共享层作用域括注' }).svg();
}

/* ============================================================
 * 2.2 §2（afterHeading）— AMAT：三类互斥命中概率 × 全程延迟
 * ============================================================ */
function amat() {
  const UID = 'g2c';
  const body = [];

  body.push(cap(36, 24, 'AMAT = 三类互斥命中概率 × 各层全程延迟的凸组合（h₁ = 0.5、h₂ = 0.5 示例；load-to-use 口径）', { size: 12, color: 'var(--ink-soft)' }));

  const rows = [
    ['weight 0.5', 'L1 hit：h₁ = 0.5 × t_L1 40.7', 20.35, '= 20.35', 'warm'],
    ['weight 0.25', 'L2 hit：(1−h₁)h₂ = 0.25 × t_L2 263.0', 65.75, '= 65.75', 'soft'],
    ['weight 0.25', '两级都 miss：(1−h₁)(1−h₂) = 0.25 × t_HBM 478.8', 119.7, '= 119.7', 'seal'],
  ];
  rows.forEach(([wlab, lab, contrib, val, tone], i) => {
    const y = 84 + i * 48;
    body.push(cap(52, y + 14, wlab, { mono: true, size: 9.5 }));
    body.push(cap(180, y - 6, lab, { size: 9.5 }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="180" y="${y}" width="${contrib * 2.2}" height="20" rx="3"/>`);
    body.push(cap(180 + contrib * 2.2 + 8, y + 14, `${val} cycles`, { mono: true, size: 9.5 }));
  });
  body.push(cap(52, 232, 't_avg = 20.35 + 65.75 + 119.7 = 205.8 cycles ≈ 112.5 ns（@1.83 GHz）—— 三权重非负、和为 1', { mono: true, size: 10.5, weight: '650' }));

  body.push(cap(52, 258, 'h₁ 灵敏度（h₂ = 0.5 固定）：命中替代 miss，平均延迟单调下降', { size: 10 }));
  [['h₁ 0.5', 205.8], ['h₁ 0.8', 106.7], ['h₁ 0.9', 73.7]].forEach(([lab, v], i) => {
    const y = 272 + i * 24;
    body.push(cap(170, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${i === 2 ? 'seal' : 'soft'}" x="180" y="${y}" width="${v * 1.6}" height="14" rx="3"/>`);
    body.push(cap(180 + v * 1.6 + 8, y + 11, `${v} cycles`, { mono: true, size: 9.5 }));
  });

  body.push(cap(52, 352, '界：t_L1 ≤ t_avg ≤ t_HBM，对 h₁/h₂ 单调不增（命题 1）；h₁ 0.5→0.9 降 64% —— tiling 的延迟侧收益', { size: 10 }));
  body.push(cap(52, 370, '带宽侧：到达 HBM 的请求比例 = (1−h₁)(1−h₂)；h₁=0.8、h₂=0.5 → 仅 10%，HBM 带宽占用降到 1/10', { size: 10 }));

  return scene(720, 386, body, { label: 'AMAT 凸组合：L1 命中权重 0.5 乘 40.7、L2 命中 0.25 乘 263、全 miss 0.25 乘 478.8，合计 205.8 cycles；h1 提升平均延迟单调下降' }).svg();
}

/* ============================================================
 * 2.2 §3（block 1）— line/sector 粒度与 coalesced/散乱流量账
 * ============================================================ */
function lineSector() {
  const UID = 'g2d';
  const body = [];

  body.push(cap(36, 24, 'line 128 B = 4 × sector 32 B：line 管 tag「在哪」、sector 管搬运「搬多少」—— 流量按 distinct sector 计', { size: 12, color: 'var(--ink-soft)' }));

  // 一条 cache line = 4 sector
  const ranges = ['[0,32) B', '[32,64) B', '[64,96) B', '[96,128) B'];
  for (let c = 0; c < 4; c++) {
    const x = 52 + c * 150;
    body.push(`<g class="dia-node dia-node-cache"><rect x="${x}" y="64" width="146" height="40" rx="8"/></g>`);
    body.push(cap(x + 73, 82, `sector ${c}`, { mono: true, size: 10.5, anchor: 'middle' }));
    body.push(cap(x + 73, 96, ranges[c], { mono: true, size: 8.5, anchor: 'middle' }));
  }
  body.push(cap(52, 122, 'miss 时按需要的 sector 取数：warp load 只触及 1 个 sector 就只搬 32 B，不是 128 B', { size: 9.5 }));

  // coalesced：32 线程连续 → 4 sector
  body.push(cap(52, 152, 'coalesced', { size: 10.5, weight: '650' }));
  body.push(cap(52, 168, '流量 128 B · 有用 128 B', { mono: true, size: 9 }));
  body.push(cap(518, 140, '32 线程 × 4 B 连续 → 覆盖 4 sector', { size: 9, anchor: 'middle' }));
  body.push(`<rect class="dia-bar dia-bar-ink" x="430" y="146" width="176" height="16" rx="3"/>`);
  for (let i = 0; i < 32; i++) body.push(`<line class="dia-edge" x1="${433 + i * 5.4}" y1="148" x2="${433 + i * 5.4}" y2="160" stroke="var(--paper-raised)"/>`);
  body.push(cap(518, 176, 'η = 128/128 = 1（recap 1.4 s_warp = 4）', { mono: true, size: 9, anchor: 'middle' }));

  // 散乱：每线程隔 128 B → 32 sector
  body.push(cap(52, 212, '全散乱', { size: 10.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 228, '流量 1024 B · 有用 128 B', { mono: true, size: 9 }));
  body.push(cap(518, 200, '每线程隔 128 B → 触及 32 sector', { size: 9, anchor: 'middle' }));
  for (let i = 0; i < 8; i++) {
    const x = 430 + i * 22;
    body.push(`<rect class="dia-bar dia-bar-soft" x="${x}" y="206" width="18" height="16" rx="3"/>`);
    body.push(`<rect class="dia-bar dia-bar-seal" x="${x + 7}" y="210" width="4" height="8" rx="1"/>`);
  }
  body.push(cap(612, 218, '⋯', { size: 11 }));
  body.push(cap(518, 236, 'η = 128/1024 = 12.5%（每 sector 只 4 B 有用）', { mono: true, size: 9, anchor: 'middle' }));

  // 有效带宽账
  body.push(cap(52, 262, '有效有用带宽 BW_eff = η · BW（式 3）：', { size: 10 }));
  body.push(cap(240, 287, '标称 3352', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="250" y="276" width="402" height="14" rx="3"/>`);
  body.push(cap(240, 311, '散乱有用 419', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="250" y="300" width="50.3" height="14" rx="3"/>`);
  body.push(cap(310, 311, 'GB/s = 3352/8 —— 标称带宽没变，有用带宽掉到 1/8', { size: 9.5 }));

  body.push(cap(52, 340, '坑：流量按 line 数或地址跨度算会高估散乱流量 4×（只触及 1 个 sector 的 line 不搬整条）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 358, 'sector 口径也解释 1.4 式 (2)：HBM 队列按 sector 服务（服务率除 32 B），不按 line、不按字节', { size: 10 }));

  return scene(720, 374, body, { label: 'cache line 128 B 含 4 个 32 B sector；coalesced load 覆盖 4 sector 全有用，散乱 load 触及 32 sector 只有八分之一有用，有效带宽 419 GB/s' }).svg();
}

/* ============================================================
 * 2.3 §2（block 0）— register bank/operand collector 教学模型
 * ============================================================ */
function regOperand() {
  const UID = 'g2e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'register operand 供给（教学模型：假设 4 bank，官方 bank 数未披露）—— operand collector 把收集与发射解耦', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 648, h: 104, label: 'register file 分区（每 processing block 64 KB）' }));
  for (let c = 0; c < 4; c++) {
    const x = 52 + c * 160;
    body.push(`<g class="dia-node dia-node-data"><rect x="${x}" y="76" width="150" height="56" rx="8"/></g>`);
    body.push(cap(x + 75, 96, `bank ${c}`, { mono: true, size: 10.5, anchor: 'middle' }));
    body.push(cap(x + 75, 116, `r${c},r${c + 4},r${c + 8},…`, { mono: true, size: 9, anchor: 'middle' }));
  }

  // 两个教学实例
  body.push(cap(52, 186, 'FMA 读 r1, r2, r3：', { mono: true, size: 10 }));
  [1, 2, 3].forEach((b, i) => {
    const n = node({ id: `na${i}`, x: 250 + i * 80, y: 170, w: 70, h: 26, role: 'norm', title: `bank ${b}`, mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(500, 186, '各一个 → 无冲突，operand 一拍取齐', { size: 9.5, color: 'var(--ink-soft)' }));
  body.push(cap(52, 226, '改读 r4, r8, r12：', { mono: true, size: 10 }));
  const hot = [0, 1, 2].map((i) => {
    const n = node({ id: `nb${i}`, x: 250 + i * 80, y: 210, w: 70, h: 26, role: 'attn', title: 'bank 0', mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(500, 226, '全在 bank 0 → 3 路冲突，读出要 3 拍', { size: 9.5, color: 'var(--seal-deep)' }));

  const col = node({ id: 'col', x: 250, y: 252, w: 300, h: 48, role: 'op', title: 'operand collector（缓冲）', sub: '跨周期收集齐 3 源 operand', titleSize: 12, subSize: 9 });
  nodes.push(col); body.push(col.svg);
  hot.forEach((h, i) => {
    body.push(edge(h, col, { nodes, portFrom: h.bottom, portTo: col.top, pulse: true, dur: 1.2, delay: 0.3 + i * 0.3, uid: UID, eid: `c${i}` }));
  });
  body.push(arrowHead(col.top[0], col.top[1], 0, 1));
  body.push(cap(52, 272, '收集未完成 → 该指令等待；', { size: 9.5 }));
  body.push(cap(52, 288, '调度器先发别的就绪指令', { size: 9.5 }));

  const dp = node({ id: 'dp', x: 250, y: 320, w: 140, h: 36, role: 'op', title: 'dispatch', sub: 'operand 齐', mono: true, titleSize: 11, subSize: 8 });
  const ex = node({ id: 'ex', x: 430, y: 320, w: 150, h: 36, role: 'attn', title: '执行管线', titleSize: 11.5 });
  nodes.push(dp, ex); body.push(dp.svg, ex.svg);
  body.push(edge(col, dp, { nodes, portFrom: col.bottom, portTo: dp.top, pulse: true, dur: 1.1, delay: 1.6, uid: UID, eid: 'd' }));
  body.push(arrowHead(dp.top[0], dp.top[1], 0, 1));
  body.push(edge(dp, ex, { nodes, pulse: true, dur: 1.1, delay: 2.0, uid: UID, eid: 'x' }));
  body.push(arrowHead(ex.left[0], ex.left[1], 1, 0));

  body.push(cap(52, 376, '2048 B/cycle 是 4 条 FMA 满发射的 operand 需求上界（无冲突假设，1.2 式 2）；bank 冲突把有效供给压到其下', { size: 10 }));
  body.push(cap(52, 394, 'disclosed 减压路径：wgmma 的 B 操作数从 shared 经描述符直喂 Tensor Core、跳过 register 中转（1.3 §4）', { size: 10 }));

  return scene(720, 410, body, { label: 'register operand 供给教学模型：四个 bank 分区，读 r1 r2 r3 无冲突一拍取齐，读 r4 r8 r12 同 bank 三路冲突三拍；operand collector 缓冲收集后 dispatch' }).svg();
}

/* ============================================================
 * 2.3 §6（afterHeading）— register spill 带宽-延迟双代价账
 * ============================================================ */
function spillCost() {
  const UID = 'g2f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'register spill 双代价：带宽账 × 延迟账（32 warps × 1000 迭代 × 每线程每迭代 spill 4 reg）', { size: 12, color: 'var(--ink-soft)' }));

  const reg = node({ id: 'reg', x: 36, y: 56, w: 150, h: 48, role: 'data', title: 'register', sub: '片上 · 4055 GB/s/SM', titleSize: 12, subSize: 8.5 });
  const loc = node({ id: 'loc', x: 250, y: 56, w: 180, h: 48, role: 'cache', title: 'local memory', sub: '逻辑线程私有 · 物理在 HBM', titleSize: 12, subSize: 8.5 });
  const hbm = node({ id: 'hbm', x: 494, y: 56, w: 190, h: 48, role: 'data', title: 'HBM backing', sub: '与 global 同延迟带宽', titleSize: 12, subSize: 8.5 });
  nodes.push(reg, loc, hbm); body.push(reg.svg, loc.svg, hbm.svg);
  body.push(edge(reg, loc, { nodes, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 's' }));
  body.push(arrowHead(loc.left[0], loc.left[1], 1, 0));
  body.push(cap(218, 48, 'st.local', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(edge(loc, hbm, { nodes, pulse: true, dur: 1.2, delay: 0.4, uid: UID, eid: 'h' }));
  body.push(arrowHead(hbm.left[0], hbm.left[1], 1, 0));
  const back = 'M 589 104 V 122 H 340 V 104';
  body.push(`<path class="dia-edge dash" d="${back}"/>`);
  body.push(pathPulse(back, { dur: 2.0, delay: 0.9 }));
  body.push(arrowHead(340, 104, 0, -1));
  body.push(cap(465, 136, 'ld.local（读回）—— 本来零 HBM 流量的 register 数据，每次用都走一遍 L1/L2/HBM', { size: 9, anchor: 'middle' }));

  body.push(cap(52, 162, '流量：32 线程 × 4 reg × 4 B × 读写 2 = 1024 B/warp/迭代 → ×32 warp ×1000 迭代 = 32.768 MB', { mono: true, size: 10 }));
  body.push(cap(52, 180, 'spill 访存是 coalesced 的：连续 32-bit 字由连续 thread ID 访问，同相对地址完全合并（代价在流量本身）', { size: 9.5 }));

  body.push(cap(52, 208, '带宽账 t_BW = M_spill ÷ 份额（式 2）：', { size: 10 }));
  body.push(cap(240, 227, '32.768 MB ÷ 25.4 GB/s', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="250" y="216" width="${1.29 * 220}" height="16" rx="3"/>`);
  body.push(cap(250 + 1.29 * 220 + 8, 228, '≈ 1.29 ms', { mono: true, size: 10, weight: '650' }));
  body.push(cap(52, 254, '延迟账 t_lat = ⌈n_spill/MLP⌉ · t_HBM（式 1，全 miss 最坏）：', { size: 10 }));
  body.push(cap(240, 273, 'MLP 4：⌈4/4⌉ = 1 批', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="250" y="262" width="${0.262 * 220}" height="16" rx="3"/>`);
  body.push(cap(250 + 0.262 * 220 + 8, 274, '≈ 0.262 ms', { mono: true, size: 10 }));
  body.push(cap(240, 297, 'MLP 1：⌈4/1⌉ = 4 批', { mono: true, size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="250" y="286" width="${1.05 * 220}" height="16" rx="3"/>`);
  body.push(cap(250 + 1.05 * 220 + 8, 298, '≈ 1.05 ms', { mono: true, size: 10 }));

  body.push(cap(52, 326, '两口径独立、实际耗时取其大者 —— MLP 1→4 延迟暴露降 4 倍（1.4 ILP 账在 spill 侧的直接应用）', { size: 10 }));
  body.push(cap(52, 344, '带宽侧从 register 4055 GB/s/SM 掉到 HBM 份额 25.4 GB/s/SM，约 160 倍落差（recap 1.2 §3）', { size: 10 }));
  body.push(cap(52, 362, '坑：压 R_th 提 occupancy 压到触发 spill 得不偿失 —— 检测看 PTX .local 与 Nsight local 流量，显著上升即回退', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 380, '每线程 local 上限 512 KB；L2 缓存 register spill、工作集小常命中 L1（40.7 cycles），实际暴露低于式 (1)', { size: 10 }));

  return scene(720, 396, body, { label: 'spill 双代价：register 值经 st.local 落到 HBM backing 再 ld.local 读回；32.768 MB 流量带宽账 1.29 ms，延迟账随 MLP 为 0.262 或 1.05 ms' }).svg();
}

/* ============================================================
 * 2.4 §5（block 0）— 三区判定流程：compute / memory / latency
 * ============================================================ */
function boundFlow() {
  const UID = 'g2g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '三区判定：compute / memory / latency-bound —— latency-bound 不是第三条 roof，是 roof 之下的区域', { size: 12, color: 'var(--ink-soft)' }));

  const inp = node({ id: 'inp', x: 36, y: 56, w: 220, h: 52, role: 'data', title: '实测 (I_ach, F_ach)', sub: '+ occupancy / 在飞并发 C', mono: true, titleSize: 12, subSize: 9 });
  const d1 = node({ id: 'd1', x: 36, y: 140, w: 220, h: 48, role: 'op', title: 'I_ach ≥ R（≈295）？', sub: 'ridge 左右', mono: true, titleSize: 12, subSize: 8.5 });
  const d2 = node({ id: 'd2', x: 36, y: 220, w: 220, h: 48, role: 'op', title: 'C ≥ L_req（≈208 sectors）？', sub: '并发够不够', mono: true, titleSize: 11, subSize: 8.5 });
  nodes.push(inp, d1, d2); body.push(inp.svg, d1.svg, d2.svg);
  body.push(edge(inp, d1, { nodes, portFrom: inp.bottom, portTo: d1.top, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'i' }));
  body.push(arrowHead(d1.top[0], d1.top[1], 0, 1));
  body.push(edge(d1, d2, { nodes, portFrom: d1.bottom, portTo: d2.top, pulse: true, dur: 1.1, delay: 0.5, uid: UID, eid: 'n' }));
  body.push(arrowHead(d2.top[0], d2.top[1], 0, 1));
  body.push(cap(156, 208, '否', { size: 9.5 }));

  const cb = node({ id: 'cb', x: 360, y: 56, w: 324, h: 52, role: 'norm', title: 'compute-bound 区', sub: '看 u_F：指令 mix / MMA 利用率（5.2 / 6.1）', titleSize: 12.5, subSize: 9 });
  const mb = node({ id: 'mb', x: 360, y: 140, w: 324, h: 52, role: 'cache', title: 'memory-bound 区', sub: '看 u_B：coalescing / 流量 / 融合（4.1 / 4.3 / 5.4）', titleSize: 12.5, subSize: 9 });
  const lb = node({ id: 'lb', x: 360, y: 220, w: 324, h: 52, role: 'attn', title: 'latency-bound 区', sub: '提 occupancy / ILP（3.4 / 1.4）或 tiling 降延迟（4.3）', titleSize: 12.5, subSize: 9 });
  nodes.push(cb, mb, lb); body.push(cb.svg, mb.svg, lb.svg);
  body.push(edge(d1, cb, { nodes, portFrom: d1.right, portTo: cb.left, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'y1' }));
  body.push(arrowHead(cb.left[0], cb.left[1], 1, 0));
  body.push(cap(300, 112, '是', { size: 9.5, anchor: 'middle' }));
  body.push(edge(d2, mb, { nodes, portFrom: d2.right, portTo: mb.left, pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: 'y2' }));
  body.push(arrowHead(mb.left[0], mb.left[1], 1, 0));
  body.push(cap(300, 192, '是', { size: 9.5, anchor: 'middle' }));
  const no2 = 'M 146 268 V 292 H 522 V 272';
  body.push(`<path class="dia-edge strong" d="${no2}"/>`);
  body.push(pathPulse(no2, { dur: 1.8, delay: 1.3 }));
  body.push(arrowHead(522, 272, 0, -1));
  body.push(cap(156, 286, '否', { size: 9.5 }));

  body.push(cap(52, 322, '式 (7)：F_ach = I·BW·min(1, C/L_req) —— C < L_req 点落斜线下方（latency-bound）；C ≥ L_req 被斜线封顶（memory-bound）', { mono: true, size: 10 }));
  body.push(cap(52, 340, 'Nsight 签名：long_scoreboard 主导且 not_selected ≈ 0，与 C < L_req 是同一件事的两个投影（recap 1.4 §7）', { size: 10 }));

  return scene(720, 356, body, { label: '三区判定流程：实测点先判强度对 ridge，再判在飞并发对所需并发，分别落 compute-bound、memory-bound、latency-bound 三区' }).svg();
}

/* ============================================================
 * 2.4 §7（block 1）— 实测 roofline：A/B/C 三点与利用率损失
 * ============================================================ */
function rooflineMeasured() {
  const UID = 'g2h';
  const body = [];

  body.push(cap(36, 24, '实测 roofline（log-log · 示意非按数值比例）：点到 roof 的垂直距离 = 利用率损失（三点教学假设口径）', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴
  body.push(`<line class="dia-edge strong" x1="100" y1="330" x2="100" y2="60"/>`);
  body.push(`<line class="dia-edge strong" x1="100" y1="330" x2="680" y2="330"/>`);
  body.push(arrowHead(100, 60, 0, -1));
  body.push(arrowHead(680, 330, 1, 0));
  body.push(cap(100, 52, 'F_ach (TFLOPS, log)', { size: 9.5, anchor: 'middle' }));
  body.push(cap(680, 348, 'I_ach (FLOP/byte, log)', { size: 9.5, anchor: 'end' }));
  body.push(cap(92, 94, '989.5', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(92, 154, '700', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(92, 283, '8.375', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(100, 348, '1', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(170, 348, '2.5', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(540, 348, '400', { mono: true, size: 8.5, anchor: 'middle' }));

  // roof：斜线 + 平台 + ridge
  body.push(`<path class="dia-edge strong" d="M 100 330 L 430 90 L 680 90"/>`);
  body.push(`<line class="dia-edge dash" x1="430" y1="90" x2="430" y2="330"/>`);
  body.push(cap(430, 362, 'R ≈ 295', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(560, 84, 'F_peak = 989.5（compute 平台）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(560, 70, 'compute-bound 区', { size: 9.5, anchor: 'middle' }));
  body.push(cap(150, 150, '斜率 = BW = 3.35 TB/s', { mono: true, size: 9.5 }));
  body.push(cap(150, 166, 'memory-bound 斜线区', { size: 9.5 }));
  body.push(cap(440, 300, 'latency-bound = roof 之下的区域（非第三条 roof）', { size: 9.5 }));

  // A：compute-bound，loss 29.3%
  body.push(`<line class="dia-edge dash" x1="540" y1="150" x2="540" y2="90"/>`);
  body.push(`<circle cx="540" cy="150" r="5" fill="var(--seal)"/>`);
  body.push(cap(548, 124, 'loss 29.3%', { mono: true, size: 9 }));
  body.push(cap(548, 166, 'A (I=400, F=700)', { mono: true, size: 9 }));
  body.push(cap(548, 180, 'u_F = 70.7%', { mono: true, size: 9 }));

  // B / C：同强度 I=2.5，memory-bound 贴斜线 vs latency-bound 斜线下方
  body.push(`<line class="dia-edge dash" x1="170" y1="279" x2="170" y2="310"/>`);
  body.push(`<circle cx="170" cy="288" r="5" fill="var(--seal)"/>`);
  body.push(`<circle cx="170" cy="310" r="5" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(pathPulse('M 170 310 V 288', { dur: 1.6, delay: 1.0 }));
  body.push(cap(182, 284, 'B (I=2.5, F=7.125)', { mono: true, size: 9 }));
  body.push(cap(182, 298, 'u_B = 85.1%（memory-bound）', { mono: true, size: 9 }));
  body.push(cap(182, 316, 'C (I=2.5, F=4.19)', { mono: true, size: 9 }));
  body.push(cap(182, 328, 'u = 50%（latency-bound）', { mono: true, size: 9 }));

  body.push(cap(52, 380, '三点读法（式 11）：A loss 29.3% 在算力侧（指令 mix/MMA 间隙）；B loss 14.9% 在带宽侧（coalescing/调度间隙）；C loss 50% 在并发侧', { size: 10 }));
  body.push(cap(52, 398, '层级 roofline：同一 kernel 对 L1/L2/DRAM 各一条斜线；tiling 提复用把点移向更高 ceiling、算法强度不变（Nsight §2.9）', { size: 10 }));

  return scene(720, 414, body, { label: '实测 roofline 示意：A 点 compute-bound 距平台 loss 29.3%，B 点贴 memory 斜线 u_B 85.1%，C 点同强度并发减半落斜线下方 u 50%，提并发把 C 拉回 B' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/01-hbm-and-memory-subsystem',
    replacesBlock: 0,
    title: 'HBM 栈几何 → 带宽：5 栈 × 16 ch × 5.238 Gbps × 64 bit ÷ 8',
    svg: hbmStack,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/02-cache-hierarchy',
    replacesBlock: 0,
    title: '内存层级金字塔：容量递增 × 延迟递增（160 : 10 : 1）',
    svg: cacheHierarchy,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/02-cache-hierarchy',
    afterHeading: /^## 2\. /,
    title: 'AMAT：三类互斥命中概率 × 全程延迟（205.8 cycles 示例）',
    svg: amat,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/02-cache-hierarchy',
    replacesBlock: 1,
    title: 'line 128 B = 4 sector：coalesced η=1 vs 散乱 η=12.5%',
    svg: lineSector,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/03-onchip-storage-registers',
    replacesBlock: 0,
    title: 'register operand 供给：bank 冲突 × operand collector（教学模型）',
    svg: regOperand,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/03-onchip-storage-registers',
    afterHeading: /^## 6\. /,
    title: 'spill 双代价账：带宽 1.29 ms × 延迟 0.262–1.05 ms（MLP 4/1）',
    svg: spillCost,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/04-bandwidth-accounting-roofline',
    replacesBlock: 0,
    title: '三区判定：compute / memory / latency-bound 决策流',
    svg: boundFlow,
  },
  {
    dim: 'gpu',
    slug: '02-memory-hierarchy/04-bandwidth-accounting-roofline',
    replacesBlock: 1,
    title: '实测 roofline：A/B/C 三点 —— 点到 roof 距离 = 利用率损失',
    svg: rooflineMeasured,
  },
];
