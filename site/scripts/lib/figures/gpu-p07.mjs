// gpu P7（07-nvlink-nvswitch）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs / gpu-p06.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）。
// 方向口径硬规则（7.1 §1）：头条 900/1800 = 双向总和；per-link 25/50 = 单向；per-GPU 单向 = 头条 ÷ 2。
// census：7.1 block0 NVSwitch 8-GPU crossbar / block1 三层互连；7.2 block0 HGX 拓扑（bash 围栏不计）；
//         7.3 block0 peer 地址映射 / block1 bounce vs P2P / block2 cuFile 调用序列（伪代码，不画）。
//         签名：7.1 §2 五代乘法链、7.1 §4 域聚合 + NVL72。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 gpu-p01 先例）。
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
 * 7.1 §2（afterHeading）— NVLink 五代乘法链 + 方向口径
 * ============================================================ */
function nvlinkGenerations() {
  const UID = 'g7a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'NVLink 五代：带宽是乘出来的 —— link 数是 per-GPU 涨幅主因子，头条 = 双向总和（方向硬规则）', { size: 12, color: 'var(--ink-soft)' }));

  // H100 乘法链：per-link 信令几何 → per-GPU 单向 → 双向头条
  const n1 = node({ id: 'n1', x: 36, y: 44, w: 200, h: 56, role: 'proj', title: '2 lanes × 100G PAM4 ÷ 8', sub: '= 25 GB/s per-link 单向（式 1）', mono: true, titleSize: 10.5, subSize: 8.5 });
  const n2 = node({ id: 'n2', x: 280, y: 44, w: 190, h: 56, role: 'data', title: '× 18 links', sub: '= 450 GB/s per-GPU 单向', mono: true, titleSize: 12, subSize: 8.5 });
  const n3 = node({ id: 'n3', x: 514, y: 44, w: 170, h: 56, role: 'attn', title: '× 2 双向', sub: '= 900 GB/s 头条（式 2）', mono: true, titleSize: 12, subSize: 8.5 });
  nodes.push(n1, n2, n3); body.push(n1.svg, n2.svg, n3.svg);
  [[n1, n2, 0], [n2, n3, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.3, delay: dl, uid: UID, eid: `c${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 120, '每 link 两方向各一组独立差分对（全双工）可同时满载 —— 因子 2 不是营销取整；V100 raw 25.78 → 有效 25（编码约 3%）', { size: 9.5 }));

  // 五代表：per-GPU 单向条 + 双向头条列
  body.push(cap(120, 150, 'links × lanes × 信令/lane', { size: 9 }));
  body.push(cap(315, 150, 'per-link', { size: 9, anchor: 'end' }));
  body.push(cap(340, 150, 'per-GPU 单向（GB/s）', { size: 9 }));
  body.push(cap(700, 150, '双向头条', { size: 9, anchor: 'end' }));
  const gens = [
    ['1.0 · P100', '4 × 8 × 20 Gbps', '20', 80, '160', 'soft'],
    ['2.0 · V100', '6 × 8 × 25.78', '25', 150, '300', 'soft'],
    ['3.0 · A100', '12 × 4 × 50', '25', 300, '600', 'soft'],
    ['4.0 · H100', '18 × 2 × 100 PAM4', '25', 450, '900', 'seal'],
    ['5.0 · B200', '18 × 2 × 200 PAM4', '50', 900, '1.8 TB/s', 'warm'],
  ];
  gens.forEach(([name, geo, pl, uni, bidir, tone], i) => {
    const y = 176 + i * 34;
    body.push(cap(36, y, name, { mono: true, size: 10 }));
    body.push(cap(120, y, geo, { mono: true, size: 8.5 }));
    body.push(cap(315, y, pl, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="340" y="${y - 12}" width="${uni * 0.28}" height="16" rx="3"/>`);
    body.push(cap(340 + uni * 0.28 + 6, y, String(uni), { mono: true, size: 9 }));
    body.push(cap(700, y, bidir, { mono: true, size: 10, anchor: 'end', color: i >= 3 ? 'var(--seal-deep)' : undefined, weight: i >= 3 ? '650' : undefined }));
  });

  body.push(cap(36, 352, '信令近似逐代翻倍、lane 数 A100 起减半，两者相消 → per-link 停 25 三代；link 数 4→6→12→18→18 才是涨幅主因子', { size: 10 }));
  body.push(cap(36, 370, '对账：NVLink4/3 = 900/600 = 1.5×（Hopper wp 逐字）；NVLink5/4 = 1800/900 = 2×（brief 逐字）；P100 vs PCIe Gen3 = 5×', { size: 10 }));
  body.push(cap(36, 388, '坑：头条 900 当单向代入 → 带宽高估 2×、通信时间低估 2×；per-GPU 单向 = 头条 ÷ 2 = 450，per-link 25 已是单向', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 404, body, { label: 'NVLink 五代乘法链：H100 两 lane 乘 100G PAM4 除 8 得 25 GB/s per-link 单向，乘 18 links 得 450 单向，乘 2 得 900 双向头条；五代表 link 数 4 到 18 是涨幅主因子' }).svg();
}

/* ============================================================
 * 7.1 §3（block 0）— NVSwitch 非阻塞 crossbar：8-GPU 全互连
 * ============================================================ */
function nvswitchCrossbar() {
  const UID = 'g7b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'NVSwitch 非阻塞 crossbar：链路不再按 peer 切死 —— 任意单对可达满注入带宽（HGX H100 8-GPU）', { size: 12, color: 'var(--ink-soft)' }));

  // 8 GPU 行
  const gpus = [];
  for (let i = 0; i < 8; i++) {
    const n = node({ id: `g${i}`, x: 36 + i * 82, y: 48, w: 76, h: 44, role: 'data', title: `GPU${i}`, sub: '18 links', mono: true, titleSize: 11, subSize: 8 });
    nodes.push(n); body.push(n.svg); gpus.push(n);
  }
  // 中间 6 GPU 的汇接短线（GPU0/GPU7 用高亮单对路径代替）
  for (let i = 1; i <= 6; i++) {
    const cx = 36 + i * 82 + 38;
    body.push(`<line class="dia-edge" x1="${cx}" y1="92" x2="${cx}" y2="166"/>`);
    body.push(arrowHead(cx, 168, 0, 1));
  }

  // 4 颗 NVSwitch + crossbar
  body.push(zone({ x: 36, y: 168, w: 648, h: 118, label: '4 × 第三代 NVSwitch · 非阻塞 crossbar' }));
  const sws = [];
  for (let i = 0; i < 4; i++) {
    const n = node({ id: `s${i}`, x: 52 + i * 152, y: 196, w: 140, h: 44, role: 'op', title: `NVSwitch${i}`, sub: '64 ports · 3.2 TB/s 双向', mono: true, titleSize: 10.5, subSize: 8 });
    nodes.push(n); body.push(n.svg); sws.push(n);
    body.push(`<line class="dia-edge" x1="${122 + i * 152}" y1="240" x2="${122 + i * 152}" y2="262"/>`);
  }
  body.push(`<line class="dia-edge strong" x1="52" y1="262" x2="648" y2="262"/>`);
  body.push(cap(360, 278, '任意输入端口 ↔ 任意输出端口线速接通（每端口至多出现一次）', { size: 8.5, anchor: 'middle' }));

  // 高亮单对路径：GPU0 → NVSwitch0 → crossbar → NVSwitch3 → GPU7
  const pIn = 'M 74 92 C 74 140, 122 150, 122 196';
  body.push(`<path class="dia-edge strong" d="${pIn}"/>`);
  body.push(pathPulse(pIn, { dur: 1.4, delay: 0 }));
  body.push(arrowHead(122, 196, 0, 1));
  body.push(pathPulse('M 122 262 H 578', { dur: 1.6, delay: 0.5 }));
  const pOut = 'M 578 196 C 578 150, 648 140, 648 92';
  body.push(`<path class="dia-edge strong" d="${pOut}"/>`);
  body.push(pathPulse(pOut, { dur: 1.4, delay: 1.0 }));
  body.push(arrowHead(648, 92, 0, -1));
  body.push(cap(360, 302, '单对路径：GPU0 → NVSwitch crossbar → GPU7 · 单对峰值 900 GB/s 双向', { size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));

  // 端口账 + 128 vs 900 对照
  body.push(cap(36, 324, '端口账：8×18 = 144 links 占 144 交换端口（4×64 = 256 总，平均 36/chip）；4 chips × 3.2 = 12.8 TB/s 双向容量', { size: 10 }));
  body.push(cap(36, 348, '2-GPU 通信单对双向：', { size: 10 }));
  body.push(cap(140, 367, '无 switch', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="150" y="356" width="${128 * 0.55}" height="14" rx="3"/>`);
  body.push(cap(228, 367, '直连均分 900/7 ≈ 128 GB/s 每对', { mono: true, size: 9.5 }));
  body.push(cap(140, 391, 'NVSwitch', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="150" y="380" width="${900 * 0.55}" height="14" rx="3"/>`);
  body.push(cap(645, 376, 'NVSwitch 单对 900 GB/s —— 7×', { size: 9.5, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(36, 414, '全对总注入两者相同（7×128 ≈ 900）—— 增益在单对峰值与调度灵活性；「任意对同时各 900」需 6300 注入，超上限不成立', { size: 9.5 }));

  return scene(720, 430, body, { label: 'HGX H100 八 GPU 各 18 links 汇入四颗第三代 NVSwitch 非阻塞 crossbar，单对路径 GPU0 经交换矩阵到 GPU7 峰值 900 GB/s 双向，直连均分每对仅 128' }).svg();
}

/* ============================================================
 * 7.1 §4（afterHeading）— 域聚合恒等式 + GB200 NVL72 域
 * ============================================================ */
function nvlinkDomain() {
  const UID = 'g7c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'NVLink 域聚合带宽：域内全部 GPU 注入带宽之和 —— 不是「任意两卡同时跑满」（命题 1 恒等式）', { size: 12, color: 'var(--ink-soft)' }));

  // 恒等式链
  const d1 = node({ id: 'd1', x: 36, y: 44, w: 170, h: 56, role: 'data', title: 'n_GPU 个 GPU', sub: '各带 n_link 条 link', titleSize: 12, subSize: 8.5 });
  const d2 = node({ id: 'd2', x: 250, y: 44, w: 220, h: 56, role: 'proj', title: 'per-GPU 双向', sub: '2·n_link·BW_link（式 2）', mono: true, titleSize: 12, subSize: 8.5 });
  const d3 = node({ id: 'd3', x: 514, y: 44, w: 170, h: 56, role: 'attn', title: 'BW_domain', sub: '域聚合（双向口径·式 4）', mono: true, titleSize: 12, subSize: 8.5 });
  nodes.push(d1, d2, d3); body.push(d1.svg, d2.svg, d3.svg);
  body.push(cap(228, 64, '×', { size: 15, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(cap(492, 64, '=', { size: 15, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(edge(d1, d2, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'i0', arrow: false }));
  body.push(arrowHead(d2.left[0], d2.left[1], 1, 0));
  body.push(edge(d2, d3, { nodes, pulse: true, dur: 1.3, delay: 0.4, uid: UID, eid: 'i1', arrow: false }));
  body.push(arrowHead(d3.left[0], d3.left[1], 1, 0));

  // 域表
  body.push(cap(36, 130, '域', { size: 9 }));
  body.push(cap(262, 130, 'GPU 数', { size: 9, anchor: 'middle' }));
  body.push(cap(352, 130, 'per-GPU 双向', { size: 9, anchor: 'middle' }));
  body.push(cap(470, 130, '域聚合（双向）', { size: 9 }));
  const domains = [
    ['HGX H100/H200 8-GPU', '8', '900 GB/s', '7.2 TB/s', '产品页 + blog'],
    ['HGX B200 8-GPU', '8', '1.8 TB/s', '14.4 TB/s', 'datasheet + brief'],
    ['GB200 NVL72', '72', '1.8 TB/s', '129.6 ≈ 130 TB/s', 'brief Table 2'],
    ['NVLink5 最大域', '576', '1.8 TB/s', '1036.8 TB/s（>1 PB/s）', 'brief + blog'],
  ];
  domains.forEach(([name, ng, per, agg, src], i) => {
    const y = 152 + i * 22;
    body.push(cap(36, y, name, { size: 10 }));
    body.push(cap(262, y, ng, { mono: true, size: 10, anchor: 'middle' }));
    body.push(cap(352, y, per, { mono: true, size: 9.5, anchor: 'middle' }));
    body.push(cap(470, y, agg, { mono: true, size: 10, color: 'var(--seal-deep)', weight: '650' }));
    body.push(cap(640, y, src, { size: 8.5 }));
  });
  body.push(cap(36, 246, '2×8×18×25 = 7200 GB/s = 7.2 TB/s —— 每条物理 link 恰一个 GPU 端点，按 GPU 求和每 link 恰计一次（无重无漏）', { size: 9.5 }));

  // NVL72 机架构图：18 compute trays ↔ 9 switch trays
  body.push(zone({ x: 36, y: 264, w: 280, h: 150, label: '18 compute trays（4 GPU+2 Grace）' }));
  for (let i = 0; i < 18; i++) {
    const y = 288 + i * 7;
    for (let j = 0; j < 4; j++) body.push(`<rect x="${52 + j * 46}" y="${y}" width="44" height="5" rx="1.5" fill="#ecf0f6" stroke="#aebcd2" stroke-width="0.8"/>`);
    for (let k = 0; k < 2; k++) body.push(`<rect x="${236 + k * 30}" y="${y}" width="28" height="5" rx="1.5" fill="#ecf2ec" stroke="#a9bfa9" stroke-width="0.8"/>`);
  }
  body.push(zone({ x: 404, y: 264, w: 280, h: 150, label: '9 switch trays（144 ports/tray）' }));
  for (let i = 0; i < 9; i++) {
    body.push(`<rect x="420" y="${288 + i * 14}" width="248" height="9" rx="2" fill="#f5efe2" stroke="#cbb98d" stroke-width="0.8"/>`);
  }
  [['M 316 292 C 350 292, 370 294, 404 294', 0.3], ['M 316 390 C 350 390, 370 392, 404 392', 0.7], ['M 316 406 C 350 406, 370 408, 404 408', 1.1]].forEach(([d, dl], i) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.8, delay: dl }));
  });
  body.push(cap(360, 322, '72×18', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(360, 336, '= 1296', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(360, 350, '= 9×144', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(360, 368, '铜缆 cartridge · 恰好全连接', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 438, 'NVL72 = 72 Blackwell + 36 Grace · 9 trays × 144 ports × 100 GB/s 双向 = 129.6 ≈ 130 TB/s；液冷机架，跨 tray 仍在域内', { size: 10 }));
  body.push(cap(36, 456, '提点：聚合把每字节发送端计一次、接收端计一次；任意单对仍受每 GPU 注入上限约束；跨机经 NIC 不是同一域', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 472, body, { label: '域聚合恒等式 GPU 数乘 per-GPU 双向得域聚合，四域 7.2 到 1036.8 TB/s；GB200 NVL72 十八 compute trays 经铜缆连九 switch trays，1296 端口恰好全连接' }).svg();
}

/* ============================================================
 * 7.1 §5（block 1）— 三层互连：NV-HBI / NVLink-C2C / NVLink5
 * ============================================================ */
function threeLayerInterconnect() {
  const UID = 'g7d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'B200 一颗芯片身上的三条 chip-to-chip 互连：三层三对象，带宽差一个量级 —— 勿混（recap 1.1 §5）', { size: 12, color: 'var(--ink-soft)' }));

  // B200 封装内 die-to-die
  body.push(zone({ x: 36, y: 48, w: 430, h: 120, label: 'B200 封装（die-to-die）' }));
  const die0 = node({ id: 'die0', x: 60, y: 88, w: 150, h: 56, role: 'proj', title: 'die 0', sub: 'reticle-limit · 4NP', titleSize: 12.5, subSize: 9 });
  const die1 = node({ id: 'die1', x: 292, y: 88, w: 150, h: 56, role: 'proj', title: 'die 1', sub: 'reticle-limit · 4NP', titleSize: 12.5, subSize: 9 });
  nodes.push(die0, die1); body.push(die0.svg, die1.svg);
  body.push(edge(die0, die1, { nodes, strong: true, pulse: true, dur: 1.6, delay: 0.2, uid: UID, eid: 'hbi', arrow: false }));
  body.push(edge(die1, die0, { nodes, portFrom: die1.left, portTo: die0.right, strong: true, uid: UID, eid: 'hbi2', arrow: false }));
  body.push(arrowHead(die1.left[0], die1.left[1], 1, 0));
  body.push(arrowHead(die0.right[0], die0.right[1], -1, 0));
  body.push(cap(251, 80, 'NV-HBI 10 TB/s', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(251, 132, '（方向未披露）', { size: 8, anchor: 'middle' }));
  body.push(cap(251, 152, '单一致芯片', { size: 9, anchor: 'middle' }));

  // Grace ↔ die0：NVLink-C2C
  const grace = node({ id: 'grace', x: 60, y: 240, w: 150, h: 56, role: 'norm', title: 'Grace CPU', sub: 'GB200：1 Grace + 2 B200', titleSize: 12, subSize: 8.5 });
  nodes.push(grace); body.push(grace.svg);
  const c2c = 'M 135 144 C 135 180, 135 204, 135 240';
  body.push(`<path class="dia-edge strong" d="${c2c}"/>`);
  body.push(pathPulse(c2c, { dur: 1.5, delay: 0.8 }));
  body.push(arrowHead(135, 240, 0, 1));
  body.push(arrowHead(135, 144, 0, -1));
  body.push(cap(145, 190, 'NVLink-C2C', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(145, 206, '900 GB/s 双向', { mono: true, size: 9.5 }));
  body.push(cap(145, 222, 'CPU↔GPU · 一致统一内存', { size: 9 }));

  // die1 ↔ 另一颗 B200：NVLink5
  const other = node({ id: 'other', x: 520, y: 88, w: 164, h: 56, role: 'data', title: '另一颗 B200', sub: 'GPU↔GPU · 入 NVLink 域', titleSize: 12, subSize: 8.5 });
  nodes.push(other); body.push(other.svg);
  body.push(edge(die1, other, { nodes, strong: true, pulse: true, dur: 1.6, delay: 1.2, uid: UID, eid: 'nvl', arrow: false }));
  body.push(edge(other, die1, { nodes, portFrom: other.left, portTo: die1.right, strong: true, uid: UID, eid: 'nvl2', arrow: false }));
  body.push(arrowHead(other.left[0], other.left[1], 1, 0));
  body.push(arrowHead(die1.right[0], die1.right[1], -1, 0));
  body.push(cap(493, 80, 'NVLink5', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(493, 94, '1.8 TB/s', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(493, 106, '双向', { size: 8, anchor: 'middle' }));

  // PCIe 对照
  const pcie = node({ id: 'pcie', x: 520, y: 240, w: 164, h: 56, role: 'plain', title: 'PCIe Gen5 x16', sub: '128 GB/s 双向 · 非一致 DMA', titleSize: 11.5, subSize: 8.5 });
  nodes.push(pcie); body.push(pcie.svg);
  body.push(`<path class="dia-edge dash" d="M 602 144 C 602 180, 602 204, 602 240"/>`);
  body.push(arrowHead(602, 240, 0, 1));
  body.push(cap(612, 190, 'GPU↔host CPU', { size: 9 }));
  body.push(cap(612, 206, '对照：NVLink5 的 1/14', { size: 9 }));

  body.push(cap(36, 322, '三层排序：NV-HBI 10 TB/s（方向未披露按原值引）> NVLink5 1.8 TB/s 双向 > C2C 900 GB/s 双向 > PCIe 128 GB/s 双向', { size: 10 }));
  body.push(cap(36, 340, 'NV-HBI 高于 HBM 7.7 TB/s：跨 die 访问不先于本地 HBM 成瓶颈；C2C 支持 CHI/CXL，对 PCIe Gen6 PHY 6× 能效', { size: 10 }));
  body.push(cap(36, 358, '坑：「双 die 经 NVLink-C2C 连接」「C2C 10 TB/s」式转述把三层混为一谈 —— die↔die / CPU↔GPU / GPU↔GPU 各连各的对象', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 374, body, { label: '三层互连：B200 封装内双 die 经 NV-HBI 10 TB/s 成单一致芯片，Grace 经 NVLink-C2C 900 GB/s 双向一致统一内存，GPU 间经 NVLink5 1.8 TB/s 双向入域，PCIe 128 对照' }).svg();
}

/* ============================================================
 * 7.2 §2（block 0）— HGX 拓扑：ring 一跳映射 + PCIe 侧路径
 * ============================================================ */
function hgxTopologyRing() {
  const UID = 'g7e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'HGX H100 8-GPU：144 links 全汇 NVSwitch —— ring 每跳 = GPU→crossbar→GPU；PCIe 另走 root complex', { size: 12, color: 'var(--ink-soft)' }));

  const gpus = [];
  for (let i = 0; i < 8; i++) {
    const n = node({ id: `g${i}`, x: 36 + i * 82, y: 48, w: 76, h: 40, role: 'data', title: `GPU${i}`, sub: '18 links', mono: true, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg); gpus.push(n);
  }
  for (let i = 0; i < 8; i++) {
    if (i === 5 || i === 6) continue; // 高亮 ring 一跳代替
    const cx = 36 + i * 82 + 38;
    body.push(`<line class="dia-edge" x1="${cx}" y1="88" x2="${cx}" y2="138"/>`);
    body.push(arrowHead(cx, 140, 0, 1));
  }

  body.push(zone({ x: 36, y: 140, w: 648, h: 76 }));
  const fab = node({ id: 'fab', x: 52, y: 170, w: 616, h: 32, role: 'op', title: '非阻塞 crossbar · 任意环排列线速接通', mono: true, titleSize: 11 });
  nodes.push(fab); body.push(fab.svg);

  // ring 一跳高亮：GPU5 → crossbar → GPU6（竖线落在 crossbar 标题两侧空白）
  body.push(`<path class="dia-edge strong" d="M 484 88 V 196"/>`);
  body.push(pathPulse('M 484 88 V 196', { dur: 1.2, delay: 0 }));
  body.push(arrowHead(484, 196, 0, 1));
  body.push(pathPulse('M 484 196 H 566', { dur: 1.0, delay: 0.4 }));
  body.push(`<path class="dia-edge strong" d="M 566 196 V 88"/>`);
  body.push(pathPulse('M 566 196 V 88', { dur: 1.2, delay: 0.8 }));
  body.push(arrowHead(566, 88, 0, -1));
  body.push(cap(525, 128, 'ring 一跳', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(360, 232, '每跳单向 raw 450 / NCCL 模型有效 370.8 GB/s（18×20.6，η = 82.4%；SM100 40.1 → 721.8）', { size: 10, anchor: 'middle' }));
  body.push(cap(36, 252, '聚合账：4 × 第三代 NVSwitch = 256 ports · 12.8 TB/s 双向；每 GPU 18 links 逐颗分布闭源未披露，只做聚合账', { size: 9.5 }));

  // PCIe 侧路径（H2D）
  body.push(cap(36, 280, 'PCIe 侧路径（H2D）：带宽 = 路径各段最小值（式 4）', { size: 10.5, weight: '650' }));
  const h1 = node({ id: 'h1', x: 36, y: 294, w: 120, h: 44, role: 'cache', title: 'host DRAM', sub: 'NUMA node 0', titleSize: 10.5, subSize: 8 });
  const rc = node({ id: 'rc', x: 200, y: 294, w: 180, h: 44, role: 'op', title: 'CPU 内存控制器 / root complex', titleSize: 9.5 });
  const px = node({ id: 'px', x: 424, y: 294, w: 130, h: 44, role: 'proj', title: 'PCIe Gen5 x16', sub: '64 GB/s 单向', mono: true, titleSize: 10.5, subSize: 8 });
  const gd = node({ id: 'gd', x: 598, y: 294, w: 86, h: 44, role: 'data', title: 'GPU', titleSize: 11 });
  nodes.push(h1, rc, px, gd); body.push(h1.svg, rc.svg, px.svg, gd.svg);
  [[h1, rc, 0.2], [rc, px, 0.5], [px, gd, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `p${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 360, '跨 socket：host buffer 在另一 NUMA node → 多一跳 inter-socket link：min(64, 40) = 40 GB/s = 本地 62.5%（40 为教学假设）', { size: 10, color: 'var(--seal-deep)' }));

  body.push(cap(36, 382, 'SKU 边界：4-GPU HGX 无 NVSwitch（直连 P2P 均分账）；H100 PCIe 版 NVLink = 600 GB/s 双向（bridge）≠ SXM 900', { size: 10 }));
  body.push(cap(36, 400, '节点内 vs 机间：NVLink 单向 450 / 400G NIC 线速 50 = 9×（B200 18×）—— 高频集合留域内，跨域按 rail 分层', { size: 10 }));

  return scene(720, 416, body, { label: 'HGX 八 GPU 全汇 NVSwitch，ring 一跳 GPU5 经 crossbar 到 GPU6 每跳有效 370.8 GB/s；PCIe H2D 路径 host DRAM 经 root complex 与 x16 到 GPU，跨 socket 取 min 40' }).svg();
}

/* ============================================================
 * 7.3 §3（block 0）— peer access 地址映射：同一指针两卡通用
 * ============================================================ */
function peerAddressMapping() {
  const UID = 'g7f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Peer access 地址映射：启用后 peer 分配映射进本设备地址空间 —— 同一指针值，两卡通用（UVA）', { size: 12, color: 'var(--ink-soft)' }));

  // 启用前
  body.push(cap(36, 66, '启用前：', { size: 10.5, weight: '650' }));
  const b0 = node({ id: 'b0', x: 110, y: 48, w: 200, h: 36, role: 'plain', title: 'GPU0 kernel 解引用 p1', titleSize: 10.5 });
  const b1 = node({ id: 'b1', x: 380, y: 48, w: 250, h: 36, role: 'plain', title: '无效：p1 不在 GPU0 地址空间', titleSize: 10.5 });
  nodes.push(b0, b1); body.push(b0.svg, b1.svg);
  body.push(`<path class="dia-edge dash" d="M 310 66 H 380"/>`);
  body.push(cap(345, 60, '✗', { size: 13, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 启用后：GPU0 虚拟地址空间 ↔ GPU1 HBM
  body.push(zone({ x: 36, y: 110, w: 300, h: 170, label: 'GPU0（当前设备）' }));
  const va = node({ id: 'va', x: 52, y: 140, w: 268, h: 44, role: 'proj', title: '虚拟地址空间（UVA）', sub: 'p1 → 映射到 GPU1 物理页', titleSize: 11, subSize: 8.5 });
  const kn = node({ id: 'kn', x: 52, y: 210, w: 268, h: 44, role: 'attn', title: 'GPU0 kernel', sub: 'load/store/atomic 解引用 p1', titleSize: 11, subSize: 8.5 });
  nodes.push(va, kn); body.push(va.svg, kn.svg);
  body.push(`<path class="dia-edge" d="M 186 210 V 184"/>`);
  body.push(pathPulse('M 186 210 V 184', { dur: 1.2, delay: 0.2 }));
  body.push(arrowHead(186, 184, 0, -1));
  body.push(cap(196, 202, '解引用 p1', { size: 8.5 }));

  body.push(zone({ x: 420, y: 110, w: 264, h: 170, label: 'GPU1（peer 设备）' }));
  const hbm = node({ id: 'hbm', x: 436, y: 140, w: 232, h: 44, role: 'cache', title: 'HBM 物理页', sub: 'p1 所指的物理数据', titleSize: 11, subSize: 8.5 });
  nodes.push(hbm); body.push(hbm.svg);

  body.push(edge(va, hbm, { nodes, strong: true, pulse: true, dur: 1.6, delay: 0.6, uid: UID, eid: 'map', arrow: false }));
  body.push(arrowHead(hbm.left[0], hbm.left[1], 1, 0));
  body.push(cap(378, 152, '映射', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(378, 178, 'peer access', { mono: true, size: 8, anchor: 'middle' }));

  const tx = 'M 320 232 H 552 V 184';
  body.push(`<path class="dia-edge strong" d="${tx}"/>`);
  body.push(pathPulse(tx, { dur: 1.8, delay: 1.2 }));
  body.push(arrowHead(552, 184, 0, -1));
  body.push(cap(436, 250, 'load/store → NVLink/PCIe 事务 → GPU1 HBM', { size: 8.5 }));

  body.push(cap(36, 302, '启用三步：cudaDeviceCanAccessPeer（查可行）→ cudaDeviceEnablePeerAccess（单向，双向各一次）→ 拷贝/解引用', { size: 9.5 }));
  body.push(cap(36, 320, '全局生效：对 peer 全部先前与后续分配生效，开销随 peer 数乘性涨 → 改用 VMM API 按需显式分配 peer-accessible 区域', { size: 10 }));
  body.push(cap(36, 338, '8-peer 上限：非 NVSwitch 系统每设备 ≤8 peer 连接 → 全互连 n−1 ≤ 8 即 n ≤ 9；NVSwitch 系统不受此限', { size: 10 }));
  body.push(cap(36, 356, '一致性 scope：跨卡同步用 thread_scope_system；peer atomic 仅在单个 GPU 访问该对象时保证原子性', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 372, body, { label: 'peer access 地址映射：启用前 GPU0 解引用 p1 无效，启用后 GPU0 虚拟地址空间映射到 GPU1 HBM 物理页，kernel load store 经 NVLink 事务到达，同一指针两卡通用' }).svg();
}

/* ============================================================
 * 7.3 §4（block 1）— P2P vs bounce buffer：省两跳的带宽账
 * ============================================================ */
function p2pVsBounce() {
  const UID = 'g7g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'P2P vs host bounce：同一条 cudaMemcpyPeer，路径决定 14 倍差 —— API 名不保证路径', { size: 12, color: 'var(--ink-soft)' }));

  // bounce 路径
  body.push(cap(36, 60, 'bounce 路径（peer access 未启用 / 不可行）：串行两跳', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const a1 = node({ id: 'a1', x: 36, y: 72, w: 130, h: 48, role: 'cache', title: 'GPU A HBM', sub: '读 M', titleSize: 11, subSize: 8.5 });
  const hb = node({ id: 'hb', x: 268, y: 72, w: 170, h: 48, role: 'data', title: 'host DRAM buffer', sub: '写 M + 读 M = 2M', titleSize: 11, subSize: 8.5 });
  const b1 = node({ id: 'b1', x: 540, y: 72, w: 144, h: 48, role: 'cache', title: 'GPU B HBM', sub: '写 M', titleSize: 11, subSize: 8.5 });
  nodes.push(a1, hb, b1); body.push(a1.svg, hb.svg, b1.svg);
  body.push(edge(a1, hb, { nodes, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'd2h', arrow: false }));
  body.push(arrowHead(hb.left[0], hb.left[1], 1, 0));
  body.push(cap(202, 90, 'D2H · PCIe 跳 1', { mono: true, size: 8, anchor: 'middle' }));
  body.push(edge(hb, b1, { nodes, pulse: true, dur: 1.4, delay: 0.5, uid: UID, eid: 'h2d', arrow: false }));
  body.push(arrowHead(b1.left[0], b1.left[1], 1, 0));
  body.push(cap(489, 90, 'H2D · PCIe 跳 2', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(36, 138, 'CPU/copy engine 参与中转 · host DRAM 流量 2M · 串行端到端 = 谐波均值 (1/BW₁ + 1/BW₂)⁻¹', { size: 9.5 }));

  // P2P 路径
  body.push(cap(36, 164, 'P2P 路径（启用，NVLink 域内）：单跳', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const a2 = node({ id: 'a2', x: 36, y: 176, w: 130, h: 48, role: 'cache', title: 'GPU A HBM', sub: '读 M', titleSize: 11, subSize: 8.5 });
  const b2 = node({ id: 'b2', x: 540, y: 176, w: 144, h: 48, role: 'cache', title: 'GPU B HBM', sub: '写 M', titleSize: 11, subSize: 8.5 });
  nodes.push(a2, b2); body.push(a2.svg, b2.svg);
  body.push(edge(a2, b2, { nodes, strong: true, pulse: true, dur: 1.6, delay: 1.0, uid: UID, eid: 'p2p', arrow: false }));
  body.push(arrowHead(b2.left[0], b2.left[1], 1, 0));
  body.push(cap(353, 194, 'NVLink 单跳 M · 450 GB/s 单向', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(353, 212, 'host DRAM 流量 = 0 · 省 D2H + H2D 两跳', { size: 9, anchor: 'middle' }));

  // 时间账
  body.push(cap(36, 250, 'M = 1 GB 时间账（raw 峰值口径）：', { size: 10.5 }));
  const rows = [
    ['bounce 串行 32 GB/s', 31.25, 'soft', '31.25 ms'],
    ['bounce 流水 64 GB/s', 15.625, 'warm', '15.625 ms'],
    ['NVLink P2P 450 GB/s', 2.22, 'seal', '2.22 ms · 对串行 14.06× / 对流水 7.03×'],
  ];
  rows.forEach(([lab, t, tone, val], i) => {
    const y = 262 + i * 24;
    body.push(cap(170, y + 11, lab, { size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="180" y="${y}" width="${t * 13}" height="14" rx="3"/>`);
    body.push(cap(180 + t * 13 + 8, y + 11, val, { mono: true, size: 9.5 }));
  });

  body.push(cap(36, 348, '增益两层：路径增益（NVLink 换 PCIe 450/64 ≈ 7×）× 跳数增益（单跳换两跳，串行口径 2×）= 14.06×', { size: 10, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(36, 366, 'PCIe P2P（同 root complex、ACS 关）64 GB/s → 15.625 ms：只拿跳数增益 2×，与流水 bounce 稳态持平', { size: 10 }));
  body.push(cap(36, 384, '坑：API 报值 ≠ 必然 P2P —— 未启用或被 IOMMU/ACS 挡掉仍经 host（~32 GB/s）；先启用再用 nvbandwidth 核对路径', { size: 10 }));

  return scene(720, 400, body, { label: 'bounce 路径 GPU A 经 D2H 到 host DRAM 再 H2D 到 GPU B，host 流量 2M 串行 31.25 ms；P2P 单跳 NVLink 450 GB/s 2.22 ms，host 零流量，快 14.06 倍' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/01-nvlink-nvswitch-architecture',
    afterHeading: /^## 2\. /,
    title: 'NVLink 五代乘法链：18×25 = 450 单向 → ×2 = 900 双向头条',
    svg: nvlinkGenerations,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/01-nvlink-nvswitch-architecture',
    replacesBlock: 0,
    title: 'NVSwitch 非阻塞 crossbar：8-GPU 全互连，单对 900 对直连均分 128',
    svg: nvswitchCrossbar,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/01-nvlink-nvswitch-architecture',
    afterHeading: /^## 4\. /,
    title: 'NVLink 域聚合恒等式 + GB200 NVL72：7.2 / 14.4 / 130 TB/s',
    svg: nvlinkDomain,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/01-nvlink-nvswitch-architecture',
    replacesBlock: 1,
    title: '三层互连勿混：NV-HBI 10 TB/s / C2C 900 双向 / NVLink5 1.8 双向',
    svg: threeLayerInterconnect,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/02-node-topology-bandwidth',
    replacesBlock: 0,
    title: 'HGX 拓扑：ring 每跳 = GPU→NVSwitch→GPU（有效 370.8 GB/s）',
    svg: hgxTopologyRing,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/03-gpudirect-p2p',
    replacesBlock: 0,
    title: 'Peer access 地址映射：UVA 同一指针两卡通用（8-peer 上限 n≤9）',
    svg: peerAddressMapping,
  },
  {
    dim: 'gpu',
    slug: '07-nvlink-nvswitch/03-gpudirect-p2p',
    replacesBlock: 1,
    title: 'P2P vs bounce：单跳 2.22 ms 对串行两跳 31.25 ms（14.06×）',
    svg: p2pVsBounce,
  },
];
