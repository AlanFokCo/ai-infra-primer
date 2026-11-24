// gpu P1（01-gpu-microarchitecture）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-cell --i ≤ 2.6。
// census：1.1 block0 die 框图 / block1 CoWoS；1.2 block0 SM 框图；
//         1.3 block0 = PTX 语法（伪代码，不画）/ block1 mma.sync 路径 / block2 tcgen05 路径；
//         1.4 block0 流水线时序 / block1 warp 调度。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 inference 先例）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

// 手工路径上的脉冲（L 形/回环/折线通道用 path 直写；渐变引用页面级固定 id）
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
 * 1.1 §2（block 0）— GPC/TPC/SM 三级分组 + harvest 账
 * ============================================================ */
function chipHierarchy() {
  const UID = 'g1a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'GH100 三级分组：8 GPC × 9 TPC × 2 SM = 144 SM（full die）—— 出货按 TPC 粒度 harvest', { size: 12, color: 'var(--ink-soft)' }));

  // 层级链：die → GPC → TPC → SM
  const chain = [
    ['die', 36, 150, 'GH100 full die', '144 SM · 814 mm² · 4N'],
    ['gpc', 216, 150, 'GPC ×8', '每 GPC 9 TPC = 18 SM'],
    ['tpc', 396, 150, 'TPC（出货 ×66）', '2 SM + 共享纹理资源'],
    ['sm', 576, 108, 'SM ×132', 'block 驻留单位'],
  ].map(([id, x, w, title, sub]) => {
    const n = node({ id, x, y: 44, w, h: 52, role: id === 'sm' ? 'attn' : 'proj', title, sub, titleSize: 12.5, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 3; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `h${i}` }));
    body.push(arrowHead(chain[i + 1].left[0], chain[i + 1].left[1], 1, 0));
  }
  body.push(cap(36, 112, '物理分组与 harvest 粒度，不是编程模型层级 —— threadIdx/blockIdx 里不出现 GPC/TPC', { size: 9.5 }));

  // die 框图：8 GPC + 分区 crossbar + L2 两分区
  body.push(zone({ x: 36, y: 124, w: 648, h: 200, label: 'GH100 full die（whitepaper Fig 6 重绘）' }));
  const gpcs = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 4; c++) {
      const k = r * 4 + c;
      const n = node({ id: `g${k}`, x: 52 + c * 154, y: 150 + r * 60, w: 146, h: 50, role: 'plain', title: `GPC ${k}`, sub: 'TPC×9 · SM SM…', mono: true, titleSize: 11.5, subSize: 8.5 });
      nodes.push(n); body.push(n.svg); gpcs.push(n);
    }
  }
  body.push(`<line class="dia-edge strong" x1="52" y1="272" x2="668" y2="272"/>`);
  body.push(cap(360, 268, '分区 crossbar（partitioned）', { size: 9, anchor: 'middle' }));
  const l2a = node({ id: 'l2a', x: 120, y: 280, w: 180, h: 34, role: 'cache', title: 'L2 分区 0 · 25 MB', mono: true, titleSize: 11 });
  const l2b = node({ id: 'l2b', x: 420, y: 280, w: 180, h: 34, role: 'cache', title: 'L2 分区 1 · 25 MB', mono: true, titleSize: 11 });
  nodes.push(l2a, l2b); body.push(l2a.svg, l2b.svg);
  body.push(edge(l2a, l2b, { nodes, pulse: true, dur: 2.0, delay: 0.9, uid: UID, eid: 'l2' }));
  body.push(edge(l2b, l2a, { nodes, portFrom: l2b.left, portTo: l2a.right, dur: 2.0, uid: UID, eid: 'l2r', arrow: false }));
  body.push(cap(360, 318, '合计 50 MB', { size: 9, anchor: 'middle' }));

  // harvest 账
  body.push(cap(52, 342, 'harvest 账（TPC 粒度屏蔽，良率机制）：144 − 132 = 12 SM = 6 TPC', { size: 10.5 }));
  body.push(bars({
    x: 52, y: 374, h: 18, unit: 3.6,
    segs: [{ w: 132, tone: 'soft' }, { w: 12, tone: 'seal' }],
  }));
  body.push(cap(289.6, 368, 'H100 SXM5 出货 132 SM / 66 TPC（91.7%）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(551.8, 368, '屏蔽 6 TPC', { size: 9.5, anchor: 'middle' }));
  body.push(cap(52, 406, '正例：66 TPC × 2 SM/TPC = 132 SM（Table 3 逐字）；H100 PCIe = 57 TPC / 114 SM', { mono: true, size: 10 }));
  body.push(cap(52, 424, '反例：把 full die 144 当出货 SM 数代入峰值式，BF16 高估 144/132 = 9.1%（1079 对 989.5 TFLOPS）', { size: 10 }));
  body.push(cap(52, 442, 'MC×6（每控制器 512-bit）×2 侧 · full 12 控制器 / 6 HBM3 栈 · NVLink×18 links · PCIe Gen5 x16', { size: 9.5 }));

  return scene(720, 456, body, { label: 'GH100 三级分组：die 到 GPC 到 TPC 到 SM 的层级链，8 GPC 框图与 L2 两分区，harvest 屏蔽 6 TPC 得 132 SM' }).svg();
}

/* ============================================================
 * 1.1 §5（block 1）— CoWoS 2.5D 封装剖面 + B200 双 die
 * ============================================================ */
function cowosPackage() {
  const UID = 'g1b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'CoWoS 2.5D 封装：硅中介层撑 5120-bit 宽总线毫米级走线 —— B200 双 die 经 NV-HBI 成单一致芯片', { size: 12, color: 'var(--ink-soft)' }));

  // H100 剖面：GPU die + 5 HBM 栈 → 中介层 → 基板
  const die = node({ id: 'die', x: 52, y: 56, w: 150, h: 64, role: 'proj', title: 'GPU die', sub: '814 mm² · TSMC 4N', titleSize: 13, subSize: 9.5 });
  nodes.push(die); body.push(die.svg);
  const stacks = [0, 1, 2, 3, 4].map((i) => {
    const n = node({ id: `st${i}`, x: 232 + i * 90, y: 56, w: 80, h: 64, role: 'cache', title: `HBM3 栈${i}`, sub: '1024-bit · 16 GB', titleSize: 10.5, subSize: 8 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(684, 52, '← 5 栈', { size: 9.5, anchor: 'end' }));
  body.push(`<line class="dia-edge strong" x1="52" y1="140" x2="668" y2="140"/>`);
  body.push(cap(360, 155, '硅中介层（silicon interposer）—— 5120-bit 内存接口走线（毫米级）+ NVLink / PCIe IO', { size: 9.5, anchor: 'middle' }));
  body.push(`<line class="dia-edge" x1="52" y1="162" x2="668" y2="162"/>`);
  body.push(cap(360, 176, '封装基板（substrate）', { size: 9.5, anchor: 'middle' }));
  [die, ...stacks].forEach((n, i) => {
    body.push(edge(n, [n.x + n.w / 2, 140], { nodes, portFrom: n.bottom, pulse: i === 0, dur: 1.4, delay: i * 0.15, uid: UID, eid: `b${i}`, arrow: false }));
  });
  const bus = 'M 60 140 H 660';
  body.push(pathPulse(bus, { dur: 2.6, delay: 0.6 }));
  body.push(cap(52, 196, 'H100 SXM5：5 栈 / 10 个 512-bit MC；full GH100 支持 6 栈 / 12 控制器 —— H200 补第 6 栈到 141 GB / 4.8 TB/s', { size: 9.5 }));

  // B200 双 die
  body.push(zone({ x: 36, y: 212, w: 648, h: 104, label: 'B200 双 die（Blackwell brief 重绘）' }));
  const d0 = node({ id: 'd0', x: 100, y: 244, w: 160, h: 52, role: 'proj', title: 'die 0（4NP）', sub: 'reticle-limit', titleSize: 12.5, subSize: 9 });
  const d1 = node({ id: 'd1', x: 460, y: 244, w: 160, h: 52, role: 'proj', title: 'die 1（4NP）', sub: 'reticle-limit', titleSize: 12.5, subSize: 9 });
  nodes.push(d0, d1); body.push(d0.svg, d1.svg);
  body.push(edge(d0, d1, { nodes, strong: true, pulse: true, dur: 1.8, delay: 1.2, uid: UID, eid: 'hbi' }));
  body.push(edge(d1, d0, { nodes, portFrom: d1.left, portTo: d0.right, strong: true, arrow: false, uid: UID, eid: 'hbi2' }));
  body.push(arrowHead(d1.left[0], d1.left[1], 1, 0));
  body.push(arrowHead(d0.right[0], d0.right[1], -1, 0));
  body.push(cap(360, 240, 'NV-HBI 10 TB/s', { mono: true, size: 11, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(360, 262, '单一致芯片（one fully coherent chip）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(52, 336, '两 reticle-limit die · 合计 208B 晶体管 + 8 HBM3e 栈（raw 8×24 = 192 GB；HGX usable 180 GB）', { size: 10 }));
  body.push(cap(52, 354, 'NV-HBI 10 TB/s 高于 HGX B200 HBM 7.7 TB/s —— 跨 die 访问不先于本地 HBM 成为带宽瓶颈', { size: 10 }));
  body.push(cap(52, 372, '坑：NV-HBI（die↔die）≠ NVLink-C2C（GB200 Grace↔Blackwell）≠ NVLink5（GPU↔GPU）—— 三条互连三个层级', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 388, body, { label: 'CoWoS 剖面：GPU die 与 5 个 HBM3 栈经硅中介层到基板；B200 两个 reticle-limit die 经 NV-HBI 10 TB/s 互连成单一致芯片' }).svg();
}

/* ============================================================
 * 1.2 §2（block 0）— SM 内部框图：4 processing block · 单发射
 * ============================================================ */
function smInternals() {
  const UID = 'g1c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Hopper SM（CC 9.0，Fig 7 重绘）：4 个 processing block，每 block 恰 1 scheduler + 1 dispatch —— 单发射', { size: 12, color: 'var(--ink-soft)' }));
  body.push(zone({ x: 36, y: 40, w: 648, h: 330, label: 'SM 边界：一个 thread block 整体驻留一个 SM，不可跨 SM 拆分' }));

  // 4 个 processing block（2×2）
  const blocks = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const k = r * 2 + c;
      const x = 52 + c * 316, y = 64 + r * 128;
      body.push(`<g class="dia-node dia-node-op"><rect x="${x}" y="${y}" width="300" height="118" rx="12"/></g>`);
      body.push(cap(x + 14, y + 18, `processing block ${k}`, { size: 11, weight: '650', color: 'var(--ink)' }));
      const sch = node({ id: `s${k}`, x: x + 14, y: y + 28, w: 138, h: 28, role: 'attn', title: 'Warp Scheduler ×1', mono: true, titleSize: 10 });
      const dis = node({ id: `d${k}`, x: x + 14, y: y + 64, w: 138, h: 28, role: 'op', title: 'Dispatch Unit ×1', mono: true, titleSize: 10 });
      nodes.push(sch, dis); body.push(sch.svg, dis.svg);
      body.push(edge(sch, dis, { nodes, portFrom: sch.bottom, portTo: dis.top, pulse: true, dur: 1.2, delay: k * 0.3, uid: UID, eid: `i${k}` }));
      body.push(arrowHead(dis.top[0], dis.top[1], 0, 1));
      body.push(cap(x + 14, y + 108, '32 thread/clk = 1 条 warp 指令/周期', { size: 8.5 }));
      body.push(cap(x + 166, y + 40, 'L0 I-cache', { mono: true, size: 9 }));
      body.push(cap(x + 166, y + 56, 'RF 16384×32-bit = 64 KB', { mono: true, size: 9 }));
      body.push(cap(x + 166, y + 76, '32 FP32 · 16 INT32', { mono: true, size: 9 }));
      body.push(cap(x + 166, y + 92, '16 FP64 · 1 TC(4th)', { mono: true, size: 9 }));
      body.push(cap(x + 166, y + 108, '8 LD/ST · SFU', { mono: true, size: 9 }));
      blocks.push([sch, dis]);
    }
  }
  // SM 级共享
  const tma = node({ id: 'tma', x: 52, y: 322, w: 150, h: 38, role: 'norm', title: 'TMA（Hopper 新增）', sub: 'global↔shared 异步批量拷贝', titleSize: 10.5, subSize: 8 });
  const l1 = node({ id: 'l1', x: 222, y: 322, w: 300, h: 38, role: 'cache', title: '256 KB L1 Data Cache / Shared Memory', sub: '统一物理 · carveout 可配（shared 最大档 228 KB）', titleSize: 10.5, subSize: 8 });
  const tex = node({ id: 'tex', x: 542, y: 322, w: 126, h: 38, role: 'plain', title: '4 Tex', sub: '纹理采样', titleSize: 10.5, subSize: 8 });
  nodes.push(tma, l1, tex); body.push(tma.svg, l1.svg, tex.svg);
  body.push(cap(52, 316, 'SM 级共享', { size: 9, weight: '650' }));

  body.push(cap(52, 392, 'register file 合计：4×16384 = 65536 个 32-bit = 256 KB/SM（全 GPU 132×256 KB ≈ 33 MB）', { mono: true, size: 10 }));
  body.push(cap(52, 410, '命题 1（单发射互证）：4 scheduler × 1 dispatch × 32 lane = 128 FP32 results/cycle/SM（Fig 7 + BP Table 5 两路同值）', { mono: true, size: 10 }));
  body.push(cap(52, 428, '全数字链：128 × 2 FLOPs/FMA × 1.98 GHz × 132 SM = 66.9 TFLOPS —— 发射宽度错一倍，峰值就错一倍', { mono: true, size: 10 }));
  body.push(cap(52, 446, '坑：双发射是 Kepler 术语（每 scheduler 2 dispatch）；若成立吞吐应 256，与 Table 5 的 128 矛盾（133.8 对 66.9 TFLOPS）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 462, body, { label: 'Hopper SM 框图：四个 processing block 各含一个 warp scheduler 与一个 dispatch unit 及寄存器分区，SM 级共享 TMA、256 KB L1/shared 与 4 Tex' }).svg();
}

/* ============================================================
 * 1.2 §5（afterHeading）— occupancy 四约束取 min（C1 走读）
 * ============================================================ */
function occupancyMin() {
  const UID = 'g1d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'occupancy 物理上界：四约束取 min（C1：256 线程/block · R_th = 64 · S_sh = 0）—— 四项皆 blocks/SM', { size: 12, color: 'var(--ink-soft)' }));

  const cons = [
    ['c0', 'block slot', 'n_b ≤ 32', '32', 'plain'],
    ['c1', 'warp slot', '⌊64/8⌋', '8', 'data'],
    ['c2', 'register', '⌊65536/(64×256)⌋', '4', 'attn'],
    ['c3', 'shared', 'S_sh = 0 → 不绑定', '∞', 'plain'],
  ].map(([id, name, formula, val, role], i) => {
    const n = node({ id, x: 36, y: 56 + i * 62, w: 252, h: 52, role, title: `${name}：${val} blocks`, sub: formula, mono: true, titleSize: 12, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(36, 316, '每 block 同时消耗四种资源，任一耗尽即装不下 —— 故取 min 不取 sum', { size: 9.5 }));

  const mn = node({ id: 'mn', x: 344, y: 148, w: 112, h: 64, role: 'attn', title: 'min = 4', sub: 'B_max（blocks/SM）', mono: true, titleSize: 14, subSize: 8.5 });
  nodes.push(mn); body.push(mn.svg);
  cons.forEach((c, i) => {
    body.push(edge(c, mn, { nodes, pulse: true, strong: i === 2, dur: 1.4, delay: i * 0.25, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(mn.left[0], mn.left[1], 1, 0));
  });
  body.push(cap(296, 136, 'register 项最紧 → 绑定', { size: 9.5, color: 'var(--seal-deep)' }));

  const wa = node({ id: 'wa', x: 512, y: 96, w: 172, h: 52, role: 'proj', title: 'W_active = 4×8', sub: '= 32 warps（式 4）', mono: true, titleSize: 12.5, subSize: 9.5 });
  const oc = node({ id: 'oc', x: 512, y: 200, w: 172, h: 52, role: 'proj', title: 'O = 32/64 = 0.50', sub: 'occupancy 上界（式 5）', mono: true, titleSize: 12.5, subSize: 9.5 });
  nodes.push(wa, oc); body.push(wa.svg, oc.svg);
  body.push(edge(mn, wa, { nodes, portFrom: mn.right, portTo: wa.left, pulse: true, dur: 1.2, delay: 1.1, uid: UID, eid: 'w' }));
  body.push(arrowHead(wa.left[0], wa.left[1], 1, 0));
  body.push(edge(wa, oc, { nodes, portFrom: wa.bottom, portTo: oc.top, pulse: true, dur: 1.2, delay: 1.5, uid: UID, eid: 'o' }));
  body.push(arrowHead(oc.top[0], oc.top[1], 0, 1));

  body.push(cap(36, 344, '灵敏度：R_th 64→32 ⇒ reg 项 ⌊65536/8192⌋ = 8，B_max = 8、O = 1.0 —— register 压力减半换 occupancy 翻倍', { mono: true, size: 10 }));
  body.push(cap(36, 362, 'C5 反例（不可 launch）：255×1024 = 261120 > 65536（3.98 倍）⇒ reg 项 0 ⇒ B_max = 0，kernel 根本跑不起来', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 380, '命题 3：实际分配 register warp 粒度取整 + shared 128 B 粒度 + 1 KB 预留 ⇒ 实际 resident blocks ≤ B_max（只减不增）', { size: 10 }));

  return scene(720, 396, body, { label: 'occupancy 四约束取 min：block slot 32、warp 项 8、register 项 4、shared 无穷，min 得 4 blocks，乘 8 warps 得 32 active warps，occupancy 0.5' }).svg();
}

/* ============================================================
 * 1.3 §2（block 1）— mma.sync 操作数路径 + k 方向累加回环
 * ============================================================ */
function mmaSyncPath() {
  const UID = 'g1e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'mma.sync m16n8k16 操作数路径：A/B 须经寄存器中转，D = A×B + C（BF16 乘 · FP32 累加）', { size: 12, color: 'var(--ink-soft)' }));

  const hbm = node({ id: 'hbm', x: 36, y: 64, w: 104, h: 56, role: 'cache', title: 'HBM', sub: 'A/B tile', titleSize: 13, subSize: 9 });
  const shm = node({ id: 'shm', x: 184, y: 64, w: 136, h: 56, role: 'cache', title: 'shared memory', sub: '异步搬入（4.4）', titleSize: 12, subSize: 9 });
  const frg = node({ id: 'frg', x: 364, y: 64, w: 150, h: 56, role: 'data', title: 'register fragment', sub: 'A 4 regs · B 2 · C 4（每线程）', mono: true, titleSize: 11.5, subSize: 8 });
  const tc = node({ id: 'tc', x: 558, y: 64, w: 126, h: 56, role: 'attn', title: 'Tensor Core', sub: '每 processing block 1 个', titleSize: 12, subSize: 8 });
  nodes.push(hbm, shm, frg, tc); body.push(hbm.svg, shm.svg, frg.svg, tc.svg);
  [[hbm, shm, 'cp.async/TMA'], [shm, frg, 'ldmatrix'], [frg, tc, '']].forEach(([a, b, lab], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.3, delay: i * 0.35, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
    if (lab) body.push(cap((a.right[0] + b.left[0]) / 2, 56, lab, { mono: true, size: 8.5, anchor: 'middle' }));
  });

  const acc = node({ id: 'acc', x: 558, y: 180, w: 126, h: 56, role: 'proj', title: 'accumulator D', sub: '4 regs FP32（每线程）', mono: true, titleSize: 11.5, subSize: 8 });
  nodes.push(acc); body.push(acc.svg);
  body.push(edge(tc, acc, { nodes, portFrom: tc.bottom, portTo: acc.top, pulse: true, dur: 1.2, delay: 1.1, uid: UID, eid: 'd' }));
  body.push(arrowHead(acc.top[0], acc.top[1], 0, 1));
  body.push(cap(612, 156, 'D = A×B + C', { mono: true, size: 9.5, anchor: 'end' }));

  // k 方向累加回环：D 当 C 读回
  const loop = 'M 558 208 H 439 V 126';
  body.push(`<path class="dia-edge strong" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 2.2, delay: 1.5 }));
  body.push(arrowHead(439, 122, 0, -1));
  body.push(cap(445, 250, '下一条 mma.sync 把 D 当 C 读回（k 方向累加）', { size: 9.5, anchor: 'middle' }));

  const epi = node({ id: 'epi', x: 558, y: 292, w: 126, h: 40, role: 'plain', title: 'epilogue 写回', titleSize: 11.5 });
  nodes.push(epi); body.push(epi.svg);
  body.push(edge(acc, epi, { nodes, portFrom: acc.bottom, portTo: epi.top, pulse: true, dur: 1.2, delay: 2.1, uid: UID, eid: 'e' }));
  body.push(arrowHead(epi.top[0], epi.top[1], 0, 1));
  body.push(cap(630, 272, '循环 k 步后', { size: 9 }));

  // 指令几何（fragment 形状示意）
  body.push(matrix({ x: 52, y: 214, n: 16, m: 16, cell: 4, gap: 1, on: () => true, label: 'A [16,16] BF16' }));
  body.push(matrix({ x: 176, y: 214, n: 16, m: 8, cell: 4, gap: 1, on: () => true, label: 'B [16,8]' }));
  body.push(matrix({ x: 256, y: 214, n: 16, m: 8, cell: 4, gap: 1, on: () => true, warm: () => true, label: 'C/D [16,8] FP32' }));
  body.push(cap(52, 316, 'F_instr = 2×16×8×16 = 4096 FLOPs（式 1）；A 256 元素 = 512 B/warp = 每线程 8 个', { mono: true, size: 10 }));
  body.push(cap(52, 334, 't_issue = 4096/1024 = 4 cycles（每 TC 稳态发射间隔，c_TC = 1024 反推自 1.1）', { mono: true, size: 10 }));
  body.push(cap(52, 352, '4 block × 4096/4 = 4096 FLOPs/cycle/SM —— 与 1.1 反推 c_BF16 = 4096 闭环', { mono: true, size: 10 }));
  body.push(cap(52, 370, 'fragment = 矩阵摊到 warp 32 线程，不是每线程复制全矩阵（误认会高估 32 倍寄存器占用）', { size: 10 }));

  return scene(720, 388, body, { label: 'mma.sync 数据流：HBM 经 cp.async 到 shared，ldmatrix 到寄存器 fragment，Tensor Core 算 D 等于 A×B+C，累加器 D 当 C 读回形成 k 方向循环' }).svg();
}

/* ============================================================
 * 1.3 §6（block 2）— tcgen05 与 TMEM：累加器搬出寄存器
 * ============================================================ */
function tcgen05Path() {
  const UID = 'g1f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'tcgen05（Blackwell sm_100a）：累加器搬出寄存器 —— TMEM 256 KB/CTA 恰等于一个 SM 的 RF 容量', { size: 12, color: 'var(--ink-soft)' }));

  const shm = node({ id: 'shm', x: 36, y: 64, w: 140, h: 56, role: 'cache', title: 'shared memory', sub: 'A/B 经描述符喂入', titleSize: 12, subSize: 9 });
  const tc = node({ id: 'tc', x: 250, y: 64, w: 160, h: 56, role: 'attn', title: 'Tensor Core（第 5 代）', sub: 'D = A×B + C · FP32 累加', titleSize: 11.5, subSize: 8.5 });
  const tmem = node({ id: 'tmem', x: 484, y: 60, w: 200, h: 64, role: 'cache', title: 'Tensor Memory（TMEM）', sub: '512 列 × 128 行 × 32-bit', titleSize: 12, subSize: 9.5 });
  nodes.push(shm, tc, tmem); body.push(shm.svg, tc.svg, tmem.svg);
  body.push(edge(shm, tc, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'a' }));
  body.push(arrowHead(tc.left[0], tc.left[1], 1, 0));
  body.push(cap(213, 56, '描述符', { size: 9, anchor: 'middle' }));
  body.push(edge(tc, tmem, { nodes, pulse: true, dur: 1.3, delay: 0.5, uid: UID, eid: 'b' }));
  body.push(arrowHead(tmem.left[0], tmem.left[1], 1, 0));
  body.push(cap(447, 56, 'D 驻 TMEM', { size: 9, anchor: 'middle' }));

  const reg = node({ id: 'reg', x: 484, y: 180, w: 200, h: 44, role: 'data', title: 'register', sub: 'tcgen05.ld（epilogue 搬回）', mono: true, titleSize: 12, subSize: 8.5 });
  const wb = node({ id: 'wb', x: 250, y: 180, w: 160, h: 44, role: 'plain', title: '写回 HBM', titleSize: 12 });
  nodes.push(reg, wb); body.push(reg.svg, wb.svg);
  body.push(edge(tmem, reg, { nodes, portFrom: tmem.bottom, portTo: reg.top, pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'c' }));
  body.push(arrowHead(reg.top[0], reg.top[1], 0, 1));
  body.push(edge(reg, wb, { nodes, portFrom: reg.left, portTo: wb.right, pulse: true, dur: 1.2, delay: 1.5, uid: UID, eid: 'd' }));
  body.push(arrowHead(wb.right[0], wb.right[1], -1, 0));

  body.push(cap(52, 252, 'TMEM = 512×128×4 B = 262144 B = 256 KB/CTA（式 5）—— Blackwell 给累加器另配与寄存器堆等大的专属存储', { mono: true, size: 10 }));
  body.push(cap(52, 270, '寄存器压力（1.3 命题 3）由此解除：大 N 不再逼低 occupancy；wgmma N=256 时累加器曾吃 128 regs/thread（上限 255 的一半）', { size: 10 }));
  body.push(cap(52, 288, 'cta_group::2：两个相邻 SM 的 CTA 协同执行一条 MMA，各持一半累加器在自己的 TMEM —— 有效 M 维翻倍', { size: 10 }));
  body.push(cap(52, 306, '对比 wgmma：累加器从 register 搬到 TMEM，寄存器腾出给 occupancy/ILP；A/B 仍走 shared 描述符（B shared-only）', { size: 10 }));

  return scene(720, 324, body, { label: 'tcgen05 数据流：shared 描述符喂 A/B 到第 5 代 Tensor Core，累加器 D 驻 TMEM，epilogue 经 tcgen05.ld 搬回寄存器写回 HBM' }).svg();
}

/* ============================================================
 * 1.3 §4（afterHeading）— 操作数复用账：16× FLOPs 不需 16× 端口
 * ============================================================ */
function operandReuse() {
  const UID = 'g1g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '操作数复用：16× FLOPs 为什么不需要 16× 寄存器端口 —— A/B 读一次，阵列内复用多次', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(52, 56, '一条 mma.sync m16n8k16（BF16）的寄存器读写流量（§2 fragment 布局）', { size: 10.5 }));
  body.push(bars({
    x: 52, y: 78, h: 20, unit: 0.3,
    segs: [{ w: 512, tone: 'soft' }, { w: 256, tone: 'warm' }, { w: 512, tone: 'soft' }, { w: 512, tone: 'seal' }],
  }));
  [['读 A 512 B', 128.8], ['读 B 256 B', 247], ['读 C 512 B', 365.2], ['写 D 512 B', 521.8]].forEach(([t, cx]) =>
    body.push(cap(cx, 72, t, { size: 9.5, anchor: 'middle' })));
  [['16×16', 128.8], ['16×8', 247], ['FP32', 365.2], ['FP32', 521.8]].forEach(([t, cx]) =>
    body.push(cap(cx, 112, t, { mono: true, size: 9, anchor: 'middle' })));
  body.push(cap(52, 130, '合计 1792 B（读 1280 + 写 512）', { mono: true, size: 10 }));

  const mma = node({ id: 'mma', x: 52, y: 140, w: 300, h: 56, role: 'attn', title: 'MMA：1792 B ÷ 4096 FLOPs', sub: '= 0.4375 B/FLOP（式 3）', mono: true, titleSize: 12.5, subSize: 10 });
  const fma = node({ id: 'fma', x: 420, y: 140, w: 264, h: 56, role: 'data', title: '标量 warp FMA：512 B ÷ 64 FLOPs', sub: '= 8 B/FLOP（1.2 式 1，3 读 1 写不复用）', mono: true, titleSize: 11, subSize: 8.5 });
  nodes.push(mma, fma); body.push(mma.svg, fma.svg);
  body.push(cap(386, 172, 'vs', { size: 11, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(cap(52, 216, '比值 8 / 0.4375 ≈ 18.3 —— MMA 每 FLOP 寄存器带宽需求只有标量 FMA 的约 1/18', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(52, 244, '每 block 每周期 operand 需求（命题 2）：', { size: 10.5 }));
  body.push(cap(180, 262, 'MMA 流（每 4 周期一条）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="188" y="252" width="${448 * 0.62}" height="14" rx="3"/>`);
  body.push(cap(188 + 448 * 0.62 + 8, 263, '1792/4 = 448 B/cycle', { mono: true, size: 9.5 }));
  body.push(cap(180, 286, '满 FMA 流（每周期一条）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="188" y="276" width="${512 * 0.62}" height="14" rx="3"/>`);
  body.push(cap(188 + 512 * 0.62 + 8, 287, '512 B/cycle（1.2 式 1）', { mono: true, size: 9.5 }));
  body.push(cap(188, 308, '448 < 512 —— MMA 流反而比满 FMA 流更省 operand 带宽', { size: 10, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(52, 330, '全 SM：4×448 = 1792 B/cycle < 满 FMA 上界 2048 B/cycle（1.2 式 2）—— 寄存器堆端口预算撑得住', { mono: true, size: 10 }));
  body.push(cap(52, 348, '机制：A 的一行广播给 n 列、B 的一列广播给 m 行 —— 1792 B 寄存器读喂了 4096 个 FLOP', { size: 10 }));
  body.push(cap(52, 366, 'wgmma 把复用推到极致：B 走 shared 描述符跳过寄存器 —— 96 B/cycle = shared 峰值 128 的 75%，留 25% 余量', { size: 10 }));

  return scene(720, 382, body, { label: '操作数复用账：MMA 一条指令寄存器流量 1792 B 除 4096 FLOPs 得 0.4375 B/FLOP，对比标量 FMA 8 B/FLOP；每周期需求 448 低于 512' }).svg();
}

/* ============================================================
 * 1.4 §2（block 0）— 执行管线时序：延迟 ≠ 吞吐
 * ============================================================ */
function pipelineTiming() {
  const UID = 'g1h';
  const body = [];
  const nodes = [];
  const X = (c) => 150 + (c - 1) * 66; // 周期 c 的列左沿

  body.push(cap(36, 24, '延迟 ≠ 吞吐：FMA 延迟 ~4 cycles，吞吐仍每周期 1 条 —— 流水线把两者解耦（单 scheduler · 单发射）', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(140, 56, '周期', { size: 10, anchor: 'end' }));
  for (let c = 1; c <= 7; c++) body.push(cap(X(c) + 30, 56, String(c), { mono: true, size: 10, anchor: 'middle' }));

  // scheduler 行：每周期发射 1 条
  body.push(cap(140, 88, 'scheduler 0', { mono: true, size: 10, anchor: 'end' }));
  for (let c = 1; c <= 4; c++) {
    const n = node({ id: `i${c}`, x: X(c), y: 68, w: 60, h: 30, role: 'op', title: `FMA${c - 1}`, mono: true, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
  }
  body.push(cap(X(5) + 30, 88, '…', { size: 12, anchor: 'middle' }));
  body.push(cap(616, 88, '← 每周期发射 1 条（吞吐）', { size: 8.5 }));

  // 三条 FMA 流水线（E1..E4，逐周期右移）
  const lanes = [
    ['FMA0', 1, 106, '写回@4'],
    ['FMA1', 2, 146, '写回@5'],
    ['FMA2', 3, 186, '@6'],
  ];
  lanes.forEach(([name, start, y, wb], li) => {
    body.push(cap(140, y + 18, `${name} 流水线`, { mono: true, size: 10, anchor: 'end' }));
    for (let s = 0; s < 4; s++) {
      const n = node({ id: `${name}e${s}`, x: X(start + s), y, w: 60, h: 26, role: s === 3 ? 'attn' : 'proj', title: `E${s + 1}`, mono: true, titleSize: 10 });
      nodes.push(n); body.push(n.svg);
    }
    body.push(cap(X(start + 4) + 30, y + 18, wb, { mono: true, size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
    const d = `M ${X(start)} ${y + 33} H ${X(start + 3) + 60}`;
    body.push(pathPulse(d, { dur: 1.6, delay: 0.2 + li * 0.4 }));
  });
  body.push(cap(150, 236, '← 4 周期延迟（发射到写回）；流水线每周期仍接受新指令', { size: 9.5 }));

  // 依赖链 vs 独立链
  body.push(cap(52, 248, '依赖链：FMA1 读 FMA0 结果 → 须等 @4 → 每 4 周期才 1 条（吞吐塌到 1/4）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(timeline({
    x: 150, y: 270, w: 456,
    ticks: [1, 2, 3, 4, 5, 6, 7].map((c) => ({ x: X(c) + 30, label: String(c) })),
    spans: [{ x1: X(1) + 24, x2: X(1) + 36, tone: 'seal' }, { x1: X(5) + 24, x2: X(5) + 36, tone: 'seal' }],
  }));
  body.push(cap(52, 304, '独立链：FMA0..FMA3 互不依赖 → 每周期 1 条（吞吐满）', { size: 10 }));
  body.push(timeline({
    x: 150, y: 326, w: 456,
    ticks: [1, 2, 3, 4, 5, 6, 7].map((c) => ({ x: X(c) + 30, label: String(c) })),
    spans: [1, 2, 3, 4].map((c) => ({ x1: X(c) + 24, x2: X(c) + 36, tone: 'ink' })),
  }));

  body.push(cap(52, 356, 'Little\'s law 热身：λ = 1 warp 指令/cycle × t_lat = 4 ⇒ L = 4 条在飞 —— 1 warp ILP 4 或 4 warp TLP 4（§4）', { mono: true, size: 10 }));
  body.push(cap(52, 374, '延迟逐层涨 shared 29 → L2 263 → HBM 479 cycles [B]、吞吐逐层跌 —— 藏哪层延迟就按哪层备并发', { size: 10 }));

  return scene(720, 390, body, { label: 'FMA 流水线时序：scheduler 每周期发射一条，FMA0 到 FMA2 各走四级流水线错开一拍；依赖链每四周期一条，独立链每周期一条' }).svg();
}

/* ============================================================
 * 1.4 §3（block 1）— warp 调度与 scoreboard 依赖追踪
 * ============================================================ */
function warpScheduler() {
  const UID = 'g1i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'warp 调度与 scoreboard：eligible = 源就绪 + 管线可用 + 同步过 —— 单发射每周期挑 1 个', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 180, h: 150, label: 'resident warps' }));
  const warps = [0, 1, 2].map((i) => {
    const n = node({ id: `w${i}`, x: 52, y: 74 + i * 38, w: 148, h: 30, role: 'data', title: `warp ${i}`, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(126, 192, '⋯ 16 slots/scheduler（64/4 均分反推）', { size: 8.5, anchor: 'middle' }));

  body.push(zone({ x: 250, y: 48, w: 220, h: 150, label: '逐 warp 判 eligible' }));
  body.push(cap(266, 86, '① 源寄存器就绪？', { size: 10.5 }));
  body.push(cap(280, 102, 'scoreboard：pending 写回已清除', { size: 8.5 }));
  body.push(cap(266, 126, '② 执行管线可用？', { size: 10.5 }));
  body.push(cap(280, 142, 'FP32 / TC / LSU / SFU 不忙', { size: 8.5 }));
  body.push(cap(266, 166, '③ 同步点已过？', { size: 10.5 }));
  body.push(cap(280, 182, 'barrier / fence / wait_group', { size: 8.5 }));

  const el = node({ id: 'el', x: 510, y: 56, w: 174, h: 44, role: 'norm', title: 'eligible warps', sub: '挑 1 个（单发射）', titleSize: 12, subSize: 8.5 });
  const dp = node({ id: 'dp', x: 510, y: 118, w: 174, h: 38, role: 'op', title: 'dispatch', mono: true, titleSize: 12 });
  const ex = node({ id: 'ex', x: 510, y: 174, w: 174, h: 40, role: 'attn', title: '执行管线', sub: 'FP32 / TC / LSU / SFU', titleSize: 12, subSize: 8.5 });
  nodes.push(el, dp, ex); body.push(el.svg, dp.svg, ex.svg);
  body.push(edge(warps[1], [250, 123], { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'j' }));
  body.push(arrowHead(250, 123, 1, 0));
  body.push(edge([470, 100], el, { nodes, pulse: true, dur: 1.3, delay: 0.5, uid: UID, eid: 'e' }));
  body.push(arrowHead(el.left[0], el.left[1], 1, 0));
  body.push(edge(el, dp, { nodes, portFrom: el.bottom, portTo: dp.top, pulse: true, dur: 1.1, delay: 1.0, uid: UID, eid: 'd' }));
  body.push(arrowHead(dp.top[0], dp.top[1], 0, 1));
  body.push(edge(dp, ex, { nodes, portFrom: dp.bottom, portTo: ex.top, pulse: true, dur: 1.1, delay: 1.4, uid: UID, eid: 'x' }));
  body.push(arrowHead(ex.top[0], ex.top[1], 0, 1));

  // stall 分支
  body.push(`<path class="dia-edge dash" d="M 360 198 V 226"/>`);
  body.push(cap(372, 222, '不 eligible → stall，按原因分类（Nsight ProfilingGuide §2.4.7）', { size: 9.5 }));
  const stalls = [
    ['long_scoreboard', '等 global/local 结果 · 访存依赖', 'memory dependency', 'attn'],
    ['wait', '等固定延迟执行依赖 · FMA 4 周期链', 'execution dependency', 'op'],
    ['barrier', '等 __syncthreads / fence', 'sync', 'op'],
    ['not_selected', 'eligible 未挑中 · 健康信号', 'not selected', 'norm'],
  ];
  stalls.forEach(([name, sub, cls, role], i) => {
    const n = node({ id: `st${i}`, x: 36 + i * 172, y: 240, w: 156, h: 64, role, title: name, sub, mono: true, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(36 + i * 172 + 78, 318, cls, { mono: true, size: 8.5, anchor: 'middle' }));
  });

  body.push(cap(52, 330, 'not_selected 为什么是健康信号：warp 已 eligible、只是这周期挑了别的 —— 可调度 warp 供过于求，发射槽不缺候选', { size: 10 }));
  body.push(cap(52, 348, 'long_scoreboard 主导且 not_selected ≈ 0 = 延迟没藏住的典型签名 —— 在飞并发不足（§4 Little\'s law）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 366, 'wgmma 异步：同步点从逐寄存器 pending 变成逐 group 计数（wait_group N）；之前仍可发射地址计算/TMA/softmax', { size: 10 }));

  return scene(720, 384, body, { label: 'warp 调度：resident warps 逐 warp 判三个 eligible 条件，eligible 者挑一个经 dispatch 进执行管线；不 eligible 按四类 stall 原因分类' }).svg();
}

/* ============================================================
 * 1.4 §4（afterHeading）— Little's law：藏住 HBM 延迟的并发账
 * ============================================================ */
function littlesLaw() {
  const UID = 'g1j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Little\'s law：藏住 479 周期 HBM 延迟，每 SM 需 ≈208 个 sector 在飞 —— L = λ · t_lat', { size: 12, color: 'var(--ink-soft)' }));

  const lam = node({ id: 'lam', x: 36, y: 56, w: 250, h: 60, role: 'proj', title: 'λ_SM = 0.434 sectors/cycle', sub: '25.4 GB/s ÷ 32 B ÷ 1.83 GHz（式 2）', mono: true, titleSize: 12, subSize: 9 });
  const tlat = node({ id: 'tlat', x: 330, y: 56, w: 180, h: 60, role: 'data', title: 't_HBM = 478.8 cycles', sub: '实测 [B] · 合 261.6 ns', mono: true, titleSize: 11.5, subSize: 9 });
  const L = node({ id: 'L', x: 554, y: 56, w: 130, h: 60, role: 'attn', title: 'L_SM ≈ 208', sub: 'sectors ≈ 6.6 KB（式 3）', mono: true, titleSize: 13, subSize: 8 });
  nodes.push(lam, tlat, L); body.push(lam.svg, tlat.svg, L.svg);
  body.push(cap(308, 76, '×', { size: 14, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(edge(lam, tlat, { nodes, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'l' }));
  body.push(arrowHead(tlat.left[0], tlat.left[1], 1, 0));
  body.push(edge(tlat, L, { nodes, pulse: true, dur: 1.5, delay: 0.4, uid: UID, eid: 't' }));
  body.push(arrowHead(L.left[0], L.left[1], 1, 0));
  body.push(cap(36, 136, '服务率取瓶颈资源（HBM）的、且 per-SM 归一化 —— 3352/132 = 25.4 GB/s 份额', { size: 9.5 }));

  const s1 = node({ id: 's1', x: 554, y: 152, w: 130, h: 48, role: 'op', title: '÷ s_warp = 4', sub: 'coalesced fp32', mono: true, titleSize: 12, subSize: 8.5 });
  const s2 = node({ id: 's2', x: 330, y: 152, w: 180, h: 48, role: 'op', title: '= 52 条 warp load', sub: '在飞（ILP 1 口径）', mono: true, titleSize: 12, subSize: 8.5 });
  const s3 = node({ id: 's3', x: 36, y: 152, w: 250, h: 48, role: 'attn', title: 'O = 52/64 = 0.81', sub: 'ILP 1 需 52 个 active warp', mono: true, titleSize: 13, subSize: 9 });
  nodes.push(s1, s2, s3); body.push(s1.svg, s2.svg, s3.svg);
  body.push(edge(L, s1, { nodes, portFrom: L.bottom, portTo: s1.top, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 'c1' }));
  body.push(arrowHead(s1.top[0], s1.top[1], 0, 1));
  body.push(edge(s1, s2, { nodes, portFrom: s1.left, portTo: s2.right, pulse: true, dur: 1.2, delay: 1.4, uid: UID, eid: 'c2' }));
  body.push(arrowHead(s2.right[0], s2.right[1], -1, 0));
  body.push(edge(s2, s3, { nodes, portFrom: s2.left, portTo: s3.right, pulse: true, dur: 1.2, delay: 1.8, uid: UID, eid: 'c3' }));
  body.push(arrowHead(s3.right[0], s3.right[1], -1, 0));

  body.push(cap(52, 232, 'O_min = L_SM/(s_warp · ILP · 64)（式 5）—— ILP 翻倍，所需 occupancy 减半（命题 2：W_active·ILP ≥ 52）', { size: 10.5 }));
  const rows = [['ILP 1', 0.81, '52 warp'], ['ILP 2', 0.41, '26 warp'], ['ILP 4', 0.20, '13 warp']];
  rows.forEach(([lab, v, note], i) => {
    const y = 248 + i * 24;
    body.push(cap(140, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${i === 0 ? 'seal' : 'warm'}" x="150" y="${y}" width="${v * 400}" height="14" rx="3"/>`);
    body.push(cap(150 + v * 400 + 8, y + 11, `O_min ${v}（${note}）`, { mono: true, size: 9.5 }));
  });

  body.push(cap(52, 336, '1.2 C1（32 warp）配 ILP 2：32×2 = 64 ≥ 52 藏住；C2（16 warp）须 ILP 4 —— occupancy 低未必藏不住，ILP 补上乘积', { size: 10 }));
  body.push(cap(52, 354, '坑：服务率误取 LSU 发射率（1 load/cycle）配 479 ⇒ 479 warp，远超 64 slots，高估 9.2× —— λ 与 t_lat 须指同一排队系统', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 372, '全芯片带宽延迟积 ≈ 877 KB 在飞（3352 GB/s × 261.6 ns）；shared 口径只需 116 sectors（服务率 4/cycle × 29 cycles）', { size: 10 }));

  return scene(720, 388, body, { label: 'Little\'s law 并发账：HBM 服务率 0.434 sectors/cycle 乘延迟 478.8 得 208 sectors，折 52 条 warp load，ILP 1 需 occupancy 0.81；ILP 翻倍所需 occupancy 减半' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/01-chip-organization-and-generations',
    replacesBlock: 0,
    title: 'GPC/TPC/SM 三级分组：full die 144 SM → harvest 132 SM',
    svg: chipHierarchy,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/01-chip-organization-and-generations',
    replacesBlock: 1,
    title: 'CoWoS 2.5D 封装与 B200 双 die：NV-HBI 10 TB/s 单一致芯片',
    svg: cowosPackage,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/02-sm-internals',
    replacesBlock: 0,
    title: 'SM 内部框图：4 processing block × 单发射 = 128 FP32/cycle',
    svg: smInternals,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/02-sm-internals',
    afterHeading: /^## 5\. /,
    title: 'occupancy 物理上界：四约束取 min（C1 走读 → O = 0.5）',
    svg: occupancyMin,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/03-tensor-core-mma',
    replacesBlock: 1,
    title: 'mma.sync 数据流：fragment → Tensor Core → FP32 累加器（k 回环）',
    svg: mmaSyncPath,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/03-tensor-core-mma',
    replacesBlock: 2,
    title: 'tcgen05 与 TMEM：累加器搬出寄存器（256 KB/CTA = RF 容量）',
    svg: tcgen05Path,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/03-tensor-core-mma',
    afterHeading: /^## 4\. /,
    title: '操作数复用账：0.4375 对 8 B/FLOP —— 16× 吞吐不需 16× 端口',
    svg: operandReuse,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/04-instruction-pipeline-latency-hiding',
    replacesBlock: 0,
    title: '流水线时序：延迟 4 cycles ≠ 吞吐每周期 1 条（依赖链 vs 独立链）',
    svg: pipelineTiming,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/04-instruction-pipeline-latency-hiding',
    replacesBlock: 1,
    title: 'warp 调度与 scoreboard：eligible 三条件 + stall 四分类',
    svg: warpScheduler,
  },
  {
    dim: 'gpu',
    slug: '01-gpu-microarchitecture/04-instruction-pipeline-latency-hiding',
    afterHeading: /^## 4\. /,
    title: 'Little\'s law：208 sectors = 52 warp load —— ILP 翻倍 O_min 减半',
    svg: littlesLaw,
  },
];
