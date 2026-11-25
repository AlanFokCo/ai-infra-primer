// gpu P10（10-fabric-and-scale-communication）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p07.mjs / gpu-p08.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；只用 dia-pulse（reduced-motion CSS 隐藏 → 静态终态）。
// 口径硬规则：1:1 = 可重排无阻塞（非严格）；分层 ring 每 NIC 2(k-1)/k·M/8 vs 平环 2(n-1)/n·M；
//   SHARP 每 rank 发送 M（「up to 2×」上界披露非保证）；DeepEP 双锚 v1.2.1 机制 / main V2.5 接口（V1 hook ≠ 当前接口）；
//   ring 步拍 = 最慢边；overlap 阈值 BW 取环最慢边带宽。
// census：10.1 block0 两级 leaf-spine / block1 rail-optimized 4 rail；10.2 block0 分层 ring 三相；
//         10.3 block0 normal 两阶段（§4 ```c WarpRole 为伪代码不画）；
//         10.4 block0 双缓冲时序 / block1 逻辑环→物理 fabric（§3 两个 ```c 围栏不计）。
//         签名：10.2 §4 SHARP 网内聚合、10.3 §5 LL hook、10.3 §7 a2a incast。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 gpu-p07 先例）。
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
 * 10.1 §2/§3（block 0）— 两级 leaf-spine 端口几何 + 收敛比 + 规模账
 * ============================================================ */
function fatTreeClos() {
  const UID = 'g10a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '两级 leaf-spine（1:1）：每 leaf 半口下行半口上行、每台 spine 接全部 leaf —— NIC 总数 = k²/2', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 648, h: 182, label: '端口几何（k 口交换机，1:1 每 leaf k/2 下行 + k/2 上行）' }));
  const s1 = node({ id: 's1', x: 100, y: 84, w: 120, h: 36, role: 'op', title: 'spine_1', mono: true, titleSize: 10.5 });
  const s2 = node({ id: 's2', x: 300, y: 84, w: 120, h: 36, role: 'op', title: 'spine_2', mono: true, titleSize: 10.5 });
  const s3 = node({ id: 's3', x: 500, y: 84, w: 120, h: 36, role: 'op', title: 'spine_{k/2}', mono: true, titleSize: 10.5 });
  nodes.push(s1, s2, s3); body.push(s1.svg, s2.svg, s3.svg);
  body.push(cap(460, 106, '…', { size: 12, anchor: 'middle' }));
  const l1 = node({ id: 'l1', x: 60, y: 160, w: 100, h: 36, role: 'proj', title: 'leaf_1', mono: true, titleSize: 10.5 });
  const l2 = node({ id: 'l2', x: 220, y: 160, w: 100, h: 36, role: 'proj', title: 'leaf_2', mono: true, titleSize: 10.5 });
  const l3 = node({ id: 'l3', x: 440, y: 160, w: 100, h: 36, role: 'proj', title: 'leaf_k', mono: true, titleSize: 10.5 });
  nodes.push(l1, l2, l3); body.push(l1.svg, l2.svg, l3.svg);
  body.push(cap(380, 182, '…', { size: 12, anchor: 'middle' }));
  // NIC 组（每 leaf k/2 下行）
  [64, 224, 444].forEach((x0) => {
    for (let j = 0; j < 3; j++) body.push(`<rect x="${x0 + j * 28}" y="210" width="24" height="8" rx="2" fill="#ecf0f6" stroke="#aebcd2" stroke-width="0.8"/>`);
  });
  body.push(cap(560, 218, 'NIC（每 leaf k/2 张）', { size: 8 }));
  // leaf→spine 上行（拓扑链接无向；高亮 leaf_1→spine_2 一条脉冲）
  [['M 110 160 C 110 140, 160 140, 160 120', null], ['M 110 160 C 110 136, 360 140, 360 120', 0], ['M 110 160 C 110 132, 560 140, 560 120', null], ['M 270 160 C 270 140, 360 140, 360 120', null], ['M 490 160 C 490 140, 560 140, 560 120', null]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge${dl !== null ? ' strong' : ''}" d="${d}"/>`);
    if (dl !== null) { body.push(pathPulse(d, { dur: 1.4, delay: dl })); body.push(arrowHead(360, 120, 0, -1)); }
  });
  body.push(cap(36, 246, '每台 spine 的 k 口接全部 leaf（leaf 台数 ≤ k）；每 leaf k/2 上行接互异 spine → spine 台数 = k/2', { size: 9.5 }));
  body.push(cap(36, 264, 'NIC 总数 = k 台 leaf × k/2 口 = k²/2（命题 2，等号可达）—— radix k 是规模杠杆：k 翻倍两级规模翻 4 倍', { size: 9.5 }));

  body.push(cap(36, 290, 'ρ = k_down/k_up（式 1）；全二分带宽 iff k_up ≥ k_down（命题 1：割容量账 + König 边着色，排列流量全部线速）', { size: 10 }));
  body.push(cap(36, 308, '1:1 = 可重排无阻塞 —— 严格无阻塞 Clos 须 2k_down−1 中平面；谈「无阻塞」先注口径', { size: 10, color: 'var(--seal-deep)' }));

  // 规模账表
  body.push(cap(36, 334, '层级', { size: 9 }));
  body.push(cap(180, 334, '公式', { size: 9 }));
  body.push(cap(330, 334, 'k=64', { size: 9 }));
  body.push(cap(450, 334, 'k=144', { size: 9 }));
  const tiers = [
    ['两级', 'k²/2', '2048', '10368（XDR 手册逐字闭合）'],
    ['三级', 'k³/4', '65536', '两级 32×（多一层跳数）'],
    ['pattern', 'k^n_tier/2^(n_tier−1)', '每加一层 ×k/2', 'n=2/3 退化为上两式'],
  ];
  tiers.forEach(([a, b, c, d], i) => {
    const y = 356 + i * 22;
    body.push(cap(36, y, a, { size: 9.5 }));
    body.push(cap(180, y, b, { mono: true, size: 9.5 }));
    body.push(cap(330, y, c, { mono: true, size: 9.5 }));
    body.push(cap(450, y, d, { size: 9 }));
  });

  body.push(cap(36, 428, '超售 3:1（k=64，48/16）：NIC 3072（+50%）但二分带宽 ÷3 —— 拿每 NIC 跨割带宽换 NIC 总数/端口成本', { size: 10 }));
  body.push(cap(36, 446, '集合是全局流量、跨割流量大：超售 fabric 上集合带宽被割容量封顶 —— AI factory 推荐 1:1 fat-tree（官方逐字）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 462, body, { label: '两级 leaf-spine 1:1 端口几何：每 leaf 半口下行接 NIC 半口上行接全部 spine，NIC 总数 k 平方除 2，k=144 得 10368 与官方逐字闭合；全二分 iff 上行不少于下行，超售 3:1 换端口成本丢二分带宽' }).svg();
}

/* ============================================================
 * 10.1 §4（block 1）— rail-optimized：每 GPU 一条 rail + 聚合 × NIC 数
 * ============================================================ */
function railOptimized() {
  const UID = 'g10b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'rail-optimized：每张 GPU 的 rail 接不同 leaf —— 同号跨机 1 交换跳，聚合带宽 × NIC 数', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 180, h: 208, label: 'node A' }));
  body.push(zone({ x: 504, y: 48, w: 180, h: 208, label: 'node B' }));
  for (let i = 0; i < 4; i++) {
    const y = 80 + i * 44;
    const ga = node({ id: `ga${i}`, x: 52, y, w: 64, h: 32, role: 'data', title: `GPU${i}`, mono: true, titleSize: 10 });
    const na = node({ id: `na${i}`, x: 132, y, w: 64, h: 32, role: 'proj', title: `NIC${i}`, mono: true, titleSize: 10 });
    const lf = node({ id: `lf${i}`, x: 300, y, w: 120, h: 32, role: 'op', title: `leaf（rail ${i}）`, mono: true, titleSize: 10 });
    const nb = node({ id: `nb${i}`, x: 520, y, w: 64, h: 32, role: 'proj', title: `NIC${i}`, mono: true, titleSize: 10 });
    const gb = node({ id: `gb${i}`, x: 600, y, w: 64, h: 32, role: 'data', title: `GPU${i}`, mono: true, titleSize: 10 });
    nodes.push(ga, na, lf, nb, gb); body.push(ga.svg, na.svg, lf.svg, nb.svg, gb.svg);
    body.push(`<line class="dia-edge" x1="116" y1="${y + 16}" x2="132" y2="${y + 16}"/>`);
    body.push(`<line class="dia-edge" x1="584" y1="${y + 16}" x2="600" y2="${y + 16}"/>`);
    const d1 = `M 196 ${y + 16} H 300`;
    const d2 = `M 420 ${y + 16} H 520`;
    body.push(`<path class="dia-edge" d="${d1}"/>`);
    body.push(pathPulse(d1, { dur: 1.3, delay: i * 0.3 }));
    body.push(arrowHead(300, y + 16, 1, 0));
    body.push(`<path class="dia-edge" d="${d2}"/>`);
    body.push(pathPulse(d2, { dur: 1.3, delay: i * 0.3 + 0.5 }));
    body.push(arrowHead(520, y + 16, 1, 0));
  }
  body.push(cap(360, 270, '示意 4 rail（8-GPU 节点 8 rail）· 各 rail 独立 leaf-spine 平面 · 节点内 GPU 经 NVSwitch 全互连', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 294, '跳数账：同号跨机（GPU_i↔GPU_i）走本 rail leaf = 1 交换跳；跨 rail 直走 = leaf_i→spine→leaf_j 3 跳', { size: 10 }));
  body.push(cap(36, 312, 'PXN 第三条路：先经 NVSwitch 域内搬到「与目的同 rail」的本地 GPU 再出口 → 仍 1 跳（NVLink 9× 高，多搬一跳域内不亏）', { size: 10 }));

  body.push(cap(36, 340, '聚合账（式 6/7）：BW_agg = n_NIC × min(NIC, PCIe) = 8×50 = 400 GB/s · vs NVLink 注入 3600 = 9×', { size: 10.5 }));
  body.push(cap(170, 363, '单 NIC 口径', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="352" width="${37.5 * 4}" height="14" rx="3"/>`);
  body.push(cap(338, 363, '37.5 ms', { mono: true, size: 9.5 }));
  body.push(cap(170, 387, '8 rail 聚合', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="376" width="${4.6875 * 4}" height="14" rx="3"/>`);
  body.push(cap(207, 387, '4.6875 ms —— 恰好 ÷8（n=16 节点、M=1 GB 节点粒度环）', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 416, '成本侧：rail 间连接可以更轻（超售）—— PXN 把跨 rail 流量压到域内，端口预算省下来（blog 逐字）', { size: 10 }));
  body.push(cap(36, 434, '三层一致：物理接线（每 rail 独立 leaf）+ PCIe 亲和（独立 x16，9.2）+ NCCL channel 映射 —— 任何一层错位式 6 打折', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 450, body, { label: 'rail-optimized 四 rail 示意：node A 与 node B 的同号 GPU 经各自 NIC 接同一台 rail leaf，跨机一跳；八 rail 聚合 400 GB/s 使节点粒度环带宽项 37.5 降到 4.6875 毫秒恰好除 8' }).svg();
}

/* ============================================================
 * 10.2 §3（block 0）— 分层 ring 三相：节点内 RS + 机间 AR + 节点内 AG
 * ============================================================ */
function hierRing() {
  const UID = 'g10c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '分层 ring 三相（k=8 节点 × 8 GPU，rail-optimized）：每 NIC 流量与 n 解耦，只与 k 和 M/8 有关', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 56, w: 200, h: 150, label: '相 1 · 节点内 reduce-scatter' }));
  body.push(cap(52, 92, 'NVLink 环（8 rank，每节点独立）', { size: 9 }));
  body.push(cap(52, 110, 'GPU0→GPU1→…→GPU7→GPU0', { mono: true, size: 8.5 }));
  body.push(cap(52, 132, '每 GPU 持 M/8 分片的节点内全和', { size: 9 }));
  body.push(cap(52, 154, '跨机字节 0', { size: 9, color: 'var(--seal-deep)' }));
  body.push(zone({ x: 260, y: 56, w: 200, h: 150, tone: 'seal', label: '相 2 · 机间 all-reduce' }));
  body.push(cap(276, 92, 'IB 环 k=8 节点 · 8 rail 并行', { size: 9 }));
  body.push(cap(276, 110, 'rail j: node0.GPUj→node1.GPUj', { mono: true, size: 8 }));
  body.push(cap(276, 126, '→…→node7.GPUj→node0.GPUj', { mono: true, size: 8 }));
  body.push(cap(276, 148, '归约 M/8 分片 · 每 rail 一跳 leaf', { size: 9 }));
  body.push(cap(276, 170, '8 NIC 全用', { size: 9, color: 'var(--seal-deep)' }));
  body.push(zone({ x: 484, y: 56, w: 200, h: 150, label: '相 3 · 节点内 all-gather' }));
  body.push(cap(500, 92, 'NVLink 环（8 rank）', { size: 9 }));
  body.push(cap(500, 114, '把 M/8 全和分片扩回全量 M', { size: 9 }));
  body.push(cap(500, 136, '跨机字节 0', { size: 9, color: 'var(--seal-deep)' }));
  [[236, 260, 0.3], [460, 484, 0.9]].forEach(([x1, x2, dl], i) => {
    const d = `M ${x1} 130 H ${x2}`;
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.0, delay: dl }));
    body.push(arrowHead(x2, 130, 1, 0));
  });

  // 每 NIC 流量账
  body.push(cap(36, 232, '每 NIC 流量（命题 1a）：分层 2(k−1)/k·M/8 vs 平环单通道 2(n−1)/n·M（节点连续排只喂一张 NIC）', { size: 10 }));
  body.push(cap(170, 255, '分层 8 rail', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="244" width="${0.21875 * 160}" height="14" rx="3"/>`);
  body.push(cap(223, 255, '218.75 MB', { mono: true, size: 9.5 }));
  body.push(cap(170, 279, '平环单通道', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="268" width="${1.96875 * 160}" height="14" rx="3"/>`);
  body.push(cap(503, 279, '1.96875 GB —— 比值 9.0×', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  // 时间账
  body.push(cap(36, 308, '时间账（命题 1b）：T_lat = 14α_NVL + 2(k−1)α_NET = 224 μs；T_bw = NVLink 4.72 + NIC 4.375 ms', { size: 10 }));
  body.push(cap(170, 331, '分层 T_hier', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="320" width="${9.32 * 8}" height="14" rx="3"/>`);
  body.push(cap(263, 331, '9.32 ms', { mono: true, size: 9.5 }));
  body.push(cap(170, 355, '平环单通道', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="344" width="${39.375 * 8}" height="14" rx="3"/>`);
  body.push(cap(503, 355, '39.375 ms —— 加速 4.2×', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 384, '口径注：分层三相是教学推导；NCCL 实际用多 channel 平环 + rail 映射逼近（每 NIC ≈0.246 GB 同量级），非显式三相 kernel', { size: 9.5 }));
  body.push(cap(36, 402, '边界反例：k=1（单节点）机间项 0、退化纯节点内 RS+AG —— 分层收益完全来自「把跨机流量摊到多 rail」', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 418, body, { label: '分层 ring 三相：节点内 NVLink reduce-scatter 得 M/8 节点内全和，机间 IB 环八 rail 并行归约分片，节点内 all-gather 扩回全量；每 NIC 218.75 MB 对平环 1.969 GB，总时间 9.32 对 39.375 毫秒' }).svg();
}

/* ============================================================
 * 10.2 §4（afterHeading）— SHARP 网内聚合：聚合树 + 出口/延迟账
 * ============================================================ */
function sharpAggregation() {
  const UID = 'g10d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'SHARP：把归约搬进 IB 交换机 —— 每 rank 出口 M（ring 2(n−1)/n·M，比 →2），延迟项 O(log n)', { size: 12, color: 'var(--ink-soft)' }));

  const root = node({ id: 'root', x: 300, y: 56, w: 120, h: 40, role: 'attn', title: '根交换机 · 全和', titleSize: 10.5 });
  const m1 = node({ id: 'm1', x: 140, y: 132, w: 120, h: 36, role: 'op', title: '聚合节点', titleSize: 10.5 });
  const m2 = node({ id: 'm2', x: 460, y: 132, w: 120, h: 36, role: 'op', title: '聚合节点', titleSize: 10.5 });
  const r0 = node({ id: 'r0', x: 60, y: 208, w: 80, h: 32, role: 'data', title: 'rank · M', mono: true, titleSize: 9.5 });
  const r1 = node({ id: 'r1', x: 220, y: 208, w: 80, h: 32, role: 'data', title: 'rank · M', mono: true, titleSize: 9.5 });
  const r2 = node({ id: 'r2', x: 420, y: 208, w: 80, h: 32, role: 'data', title: 'rank · M', mono: true, titleSize: 9.5 });
  const r3 = node({ id: 'r3', x: 580, y: 208, w: 80, h: 32, role: 'data', title: 'rank · M', mono: true, titleSize: 9.5 });
  nodes.push(root, m1, m2, r0, r1, r2, r3); body.push(root.svg, m1.svg, m2.svg, r0.svg, r1.svg, r2.svg, r3.svg);
  // 上传（实线脉冲）：rank → 聚合节点 → 根
  [['M 100 208 C 100 186, 170 186, 170 168', 0], ['M 260 208 C 260 186, 230 186, 230 168', 0.2], ['M 200 132 C 200 112, 330 112, 330 96', 0.5], ['M 460 208 C 460 186, 490 186, 490 168', 0.3], ['M 620 208 C 620 186, 550 186, 550 168', 0.5], ['M 520 132 C 520 112, 390 112, 390 96', 0.8]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.3, delay: dl }));
  });
  body.push(arrowHead(170, 168, 0, -1)); body.push(arrowHead(230, 168, 0, -1)); body.push(arrowHead(330, 96, 0, -1));
  body.push(arrowHead(490, 168, 0, -1)); body.push(arrowHead(550, 168, 0, -1)); body.push(arrowHead(390, 96, 0, -1));
  // 广播（虚线下行）：根 → 聚合节点 → rank
  [['M 300 76 C 260 76, 200 100, 200 132'], ['M 420 76 C 460 76, 520 100, 520 132'], ['M 140 150 C 100 150, 100 180, 100 208'], ['M 260 150 V 208'], ['M 460 150 V 208'], ['M 580 150 C 620 150, 620 180, 620 208']].forEach(([d]) => {
    body.push(`<path class="dia-edge dash" d="${d}"/>`);
  });
  body.push(arrowHead(200, 132, 0, 1)); body.push(arrowHead(520, 132, 0, 1));
  body.push(arrowHead(100, 208, 0, 1)); body.push(arrowHead(260, 208, 0, 1)); body.push(arrowHead(460, 208, 0, 1)); body.push(arrowHead(620, 208, 0, 1));
  body.push(cap(36, 260, '上传：每 rank 自有 M 只发一次，聚合节点在交换机内归约子树部分和再上传；虚线 = 根全和 M 向下广播', { size: 9.5 }));

  // 出口账
  body.push(cap(36, 288, 'n=64、M=1 GB、NIC 50 GB/s 出口账（命题 2）：', { size: 10.5 }));
  body.push(cap(170, 311, 'ring 每 rank', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="300" width="${39.375 * 6}" height="14" rx="3"/>`);
  body.push(cap(424, 311, '1.96875 GB → 39.375 ms', { mono: true, size: 9.5 }));
  body.push(cap(170, 335, 'SHARP 每 rank', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="324" width="${20 * 6}" height="14" rx="3"/>`);
  body.push(cap(308, 335, '1 GB → 20 ms · 比 1.97（→2 上界）', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  // 万卡延迟账
  body.push(cap(36, 364, '万卡延迟项（n=10240、α_NET=14 μs）：ring 2(n−1)α vs tree 族 2⌈log₂n⌉α', { size: 10.5 }));
  body.push(cap(170, 387, 'ring', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="376" width="${286.7 * 1.6}" height="14" rx="3"/>`);
  body.push(cap(639, 370, '286.7 ms', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(170, 411, 'tree 族', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="400" width="3" height="14" rx="1.5"/>`);
  body.push(cap(191, 411, '392 μs —— 731×：万卡小消息必须 tree/SHARP', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 440, 'NVLS 对照：NVLS = NVLink 域内 NVSwitch 归约、IB SHARP = 跨机；每 rank 发送均 M；NVLS_TREE + CollNet 可组合', { size: 9.5 }));
  body.push(cap(36, 458, 'SAT 流式：大消息 >16 KB 阈值、小消息 OST 低延迟；「up to 2×」是上界披露非保证值（端点注入瓶颈口径）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 474, body, { label: 'SHARP 聚合树：rank 上传自有 M 一次，交换机内聚合节点归约子树部分和，根得全和向下广播；n=64 出口 20 对 ring 39.375 毫秒，万卡延迟项 392 微秒对 ring 286.7 毫秒 731 倍' }).svg();
}

/* ============================================================
 * 10.3 §4（block 0）— DeepEP normal 两阶段 WarpRole + 节点去重
 * ============================================================ */
function deepepNormal() {
  const UID = 'g10e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DeepEP normal 两阶段（v1.2.1 锚）：RDMA 发到远端同 nvl_rank GPU → NVLink 转发到 expert 卡', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 56, w: 200, h: 120, label: 'node A（源）' }));
  const gi = node({ id: 'gi', x: 52, y: 92, w: 168, h: 52, role: 'attn', title: 'GPU_i · kRDMASender', sub: 'nvshmemi_ibgda_put_nbi_warp', mono: true, titleSize: 10, subSize: 7.5 });
  nodes.push(gi); body.push(gi.svg);
  body.push(zone({ x: 420, y: 56, w: 264, h: 120, label: 'node B（目的）' }));
  const bi = node({ id: 'bi', x: 436, y: 92, w: 110, h: 52, role: 'proj', title: 'GPU_i', sub: 'Forwarder（L690）', mono: true, titleSize: 10.5, subSize: 7.5 });
  const bj = node({ id: 'bj', x: 584, y: 92, w: 84, h: 52, role: 'data', title: 'GPU_j', sub: 'kNVLReceivers', mono: true, titleSize: 10.5, subSize: 7 });
  nodes.push(bi, bj); body.push(bi.svg, bj.svg);
  const d1 = 'M 220 118 H 436';
  body.push(`<path class="dia-edge strong" d="${d1}"/>`);
  body.push(pathPulse(d1, { dur: 1.6, delay: 0 }));
  body.push(arrowHead(436, 118, 1, 0));
  body.push(cap(328, 104, '阶段 1 · RDMA 节点去重', { size: 9, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  const d2 = 'M 546 118 H 584';
  body.push(`<path class="dia-edge strong" d="${d2}"/>`);
  body.push(pathPulse(d2, { dur: 0.9, delay: 0.8 }));
  body.push(arrowHead(584, 118, 1, 0));
  body.push(cap(565, 84, '阶段 2', { size: 8, anchor: 'middle' }));
  body.push(cap(565, 136, 'NVLink', { mono: true, size: 8, anchor: 'middle' }));
  // 两阶段语义注记（全宽行，避开 zone 虚线竖边）
  body.push(cap(36, 188, '阶段 1 目的 = 同 nvl_rank=i（translate_dst_rdma_rank normal 分支译节点号，LL 分支译具体 GPU）', { size: 8.5 }));
  body.push(cap(36, 202, '阶段 2「RDMA consumers and NVL producers」（L690-691）：从 rdma_recv_buffer 取 token 经 NVLink 转发', { size: 8.5 }));

  // WarpRole 五角色
  body.push(cap(36, 222, 'WarpRole 五角色（internode.cu L370-376）：', { size: 10 }));
  const roles = [
    ['kRDMASender', 96, 'proj'], ['kRDMASenderCoordinator', 150, 'proj'], ['kRDMAAndNVLForwarder', 140, 'attn'],
    ['kForwarderCoordinator', 132, 'attn'], ['kNVLReceivers', 96, 'attn'],
  ];
  let rx = 36;
  roles.forEach(([t, w, role], i) => {
    const n = node({ id: `wr${i}`, x: rx, y: 234, w, h: 28, role, title: t, mono: true, titleSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    rx += w + 8;
  });
  body.push(cap(36, 280, '阶段 1 sender warps 7（L1008）· combine 对称（NVLink 归拢 + IB 回传，internode_combine L935）', { size: 9 }));

  // 节点去重账
  body.push(cap(36, 304, '节点去重（命题 1）：RDMA 副本 = D_node（每命中节点一份）vs 逐 expert 直发 = D_card', { size: 10 }));
  body.push(cap(170, 327, '逐副本口径', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="316" width="${242.2 * 1.2}" height="14" rx="3"/>`);
  body.push(cap(479, 327, '242.2 MB（4096×8×7168×1.03125）', { mono: true, size: 9 }));
  body.push(cap(170, 351, '节点去重 RDMA', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="340" width="${109.0 * 1.2}" height="14" rx="3"/>`);
  body.push(cap(319, 351, '109.0 MB（×3.6 节点）· 2.22×', { mono: true, size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(36, 376, 'V3 组限定：D_card 7.6 / D_node 3.6 = 2.11× —— 多份副本搬运留 NVLink 域（~160 GB/s），跨机只发去重一份', { size: 9.5 }));

  body.push(cap(36, 400, 'README 性能表（H800+CX7）：intranode EP8 153/158 GB/s（NVLink）；internode EP64 51/50 —— 趋 CX7 线速，RDMA 段喂满', { size: 9.5 }));
  body.push(cap(36, 418, 'V2.5（main 锚）：hybrid 模式为两阶段后继（gin.put<ncclTeamTagRail>、multi-rail 友好）；接口走读归 §6', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 434, body, { label: 'DeepEP normal 两阶段：源 GPU_i 经 RDMA 把去重后每节点一份发到目的节点同 nvl_rank GPU_i，Forwarder 从 RDMA 缓冲消费经 NVLink 转发到 expert 卡 GPU_j；逐副本 242.2 MB 对节点去重 109 MB 省 2.22 倍' }).svg();
}

/* ============================================================
 * 10.3 §5（afterHeading）— LL hook：SEND/RECV 相分离 + IBGDA + V2.5 硬注
 * ============================================================ */
function deepepLLHook() {
  const UID = 'g10f';
  const body = [];

  body.push(cap(36, 24, 'DeepEP low-latency（v1.2.1 锚）：纯 RDMA 单跳 + hook 重叠 —— SEND 相只 issue，hook() 触发 RECV 相', { size: 12, color: 'var(--ink-soft)' }));

  // hook 时序
  body.push(`<line class="dia-edge strong" x1="36" y1="110" x2="660" y2="110"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="36" y="101" width="114" height="10" rx="5"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="150" y="101" width="330" height="10" rx="5"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="480" y="101" width="120" height="10" rx="5"/>`);
  body.push(pathPulse('M 150 110 H 480', { dur: 2.0, delay: 0.4 }));
  [[150], [480]].forEach(([x]) => body.push(`<line class="dia-edge" x1="${x}" y1="103" x2="${x}" y2="117"/>`));
  body.push(cap(93, 92, 'SEND 相', { size: 9.5, anchor: 'middle', weight: '650' }));
  body.push(cap(315, 92, 'RDMA 后台搬（接收侧零 SM）∥ 计算侧跑别的活', { size: 9.5, anchor: 'middle' }));
  body.push(cap(540, 92, 'RECV 相', { size: 9.5, anchor: 'middle', weight: '650' }));
  body.push(cap(93, 132, '只 issue 请求不接收', { size: 8.5, anchor: 'middle' }));
  body.push(cap(150, 148, '返回 hook', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(480, 148, '调 hook()', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(540, 132, '收数据（pull 式重叠）', { size: 8.5, anchor: 'middle' }));

  // IBGDA + 预注册
  body.push(zone({ x: 36, y: 166, w: 324, h: 92, label: 'IBGDA：GPU 直构 WQE，免 CPU proxy' }));
  body.push(cap(52, 200, 'mlx5_wqe_ctrl_seg GPU 直构（ibgda_device.cuh L63）', { mono: true, size: 8.5 }));
  body.push(cap(52, 218, 'put_nbi_warp L339 / quiet L495 · 单边 WRITE 语义', { mono: true, size: 8.5 }));
  body.push(cap(52, 240, 'GDR 数据面零拷贝 + IBGDA 控制面零 CPU', { size: 9, color: 'var(--seal-deep)' }));
  body.push(zone({ x: 396, y: 166, w: 288, h: 92, label: '预注册 + 节点内漂移' }));
  body.push(cap(412, 200, 'num_rdma_bytes 预注册 + num_qps_per_rank=24 长驻', { size: 8.5 }));
  body.push(cap(412, 218, '2025.06 起 LL 节点内可用 NVLink（#173）', { size: 8.5 }));
  body.push(cap(412, 240, '「纯 RDMA」指跨机路径，非全部路径', { size: 9, color: 'var(--seal-deep)' }));

  // ΔT 账
  body.push(cap(36, 284, '命题 2：ΔT = T_normal − T_LL = α_NVL + mβ_NVL（式 2）—— normal 两阶段多付一跳 NVLink', { size: 10 }));
  body.push(cap(170, 307, 'decode 30.72 KB', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="296" width="${2.19 * 40}" height="14" rx="3"/>`);
  body.push(cap(276, 307, '2.19 μs ≈ α_NVL（带宽项可略）', { mono: true, size: 9 }));
  body.push(cap(170, 331, 'prefill 1 MB', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="180" y="320" width="${8.25 * 40}" height="14" rx="3"/>`);
  body.push(cap(518, 331, '8.25 μs', { mono: true, size: 9 }));
  body.push(cap(36, 356, '读数：小消息 LL 省一跳固定延迟赢延迟；大消息 normal 靠节点去重（命题 1）赢带宽 —— 两族各对应一个 regime', { size: 9.5 }));

  // V2.5 硬注
  body.push(zone({ x: 36, y: 372, w: 648, h: 92, tone: 'seal', label: '版本漂移硬注（main V2.5 锚）—— V1 hook 语义 ≠ 当前接口' }));
  body.push(cap(52, 406, 'V1 全移除：统一 EPBuffer（hybrid/direct）+ NCCL Gin GDAKI 后端，NVSHMEM 不再依赖（README L18）', { size: 9 }));
  body.push(cap(52, 424, 'EP dispatch/combine 占 SM（zero-SM RDMA EP 不支持）；V1「零 SM」仅指 LL 接收侧 —— 版本与范围都不同', { size: 9 }));
  body.push(cap(52, 442, '「IBGDA」V2.5 仅存注释、后端实为 Gin GDAKI；QP 解析 direct min(sms,9) / hybrid sms×16+1', { size: 9 }));

  return scene(720, 480, body, { label: 'DeepEP low-latency hook 时序：SEND 相只 issue RDMA 请求返回 hook，后台 RDMA 接收侧零 SM 与计算重叠，调 hook 触发 RECV 相；IBGDA GPU 直构 WQE 免 CPU proxy；V2.5 已移除 V1 改 EPBuffer 加 Gin 且 EP 占 SM' }).svg();
}

/* ============================================================
 * 10.3 §7（afterHeading）— a2a incast：均衡（均匀）流量 vs 偏斜 σC
 * ============================================================ */
function a2aIncast() {
  const UID = 'g10g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'a2a incast：均衡（均匀）流量入口恰线速 C —— 偏斜 σ 向热卡集中，入口 σC，拥塞 iff σ>1 且持续', { size: 12, color: 'var(--ink-soft)' }));

  // 均衡（均匀）流量
  body.push(zone({ x: 36, y: 56, w: 324, h: 150, label: '均匀流量（σ=1）：入口恰线速' }));
  const rc = node({ id: 'rc', x: 240, y: 112, w: 100, h: 44, role: 'data', title: 'receiver', sub: '入口 = C', titleSize: 10.5, subSize: 8.5 });
  nodes.push(rc); body.push(rc.svg);
  [92, 132, 172].forEach((y, i) => {
    const s = node({ id: `s${i}`, x: 52, y, w: 80, h: 30, role: 'proj', title: `sender`, mono: true, titleSize: 9.5 });
    nodes.push(s); body.push(s.svg);
    const d = `M 132 ${y + 15} C 180 ${y + 15}, 190 134, 240 134`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.3, delay: i * 0.3 }));
  });
  body.push(arrowHead(240, 134, 1, 0));
  body.push(cap(186, 100, '各 C/(n−1)', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(52, 222, '每 sender 出口 C 均分 n−1 份 → 入口 (n−1)·C/(n−1) = C，不累积', { size: 9 }));

  // 偏斜 incast
  body.push(zone({ x: 396, y: 56, w: 288, h: 150, tone: 'seal', label: '偏斜 σ：入口 σC > C' }));
  const hot = node({ id: 'hot', x: 560, y: 112, w: 108, h: 44, role: 'attn', title: '热卡入口 σC', sub: '超额 (σ−1)C 累积', titleSize: 10, subSize: 7.5 });
  nodes.push(hot); body.push(hot.svg);
  [92, 132, 172].forEach((y, i) => {
    const s = node({ id: `t${i}`, x: 412, y, w: 70, h: 30, role: 'proj', title: `sender`, mono: true, titleSize: 9.5 });
    nodes.push(s); body.push(s.svg);
    const d = `M 482 ${y + 15} C 520 ${y + 15}, 520 134, 560 134`;
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.1, delay: i * 0.25 }));
  });
  body.push(arrowHead(560, 134, 1, 0));
  body.push(cap(521, 100, '各 σC/(n−1)', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(412, 222, '队列以 (σ−1)C 累积 → ECN 标记 / PFC pause（9.3）', { size: 9 }));

  // 填队列账
  body.push(cap(36, 252, 'σ', { size: 9 }));
  body.push(cap(150, 252, '入口 GB/s', { size: 9 }));
  body.push(cap(260, 252, '超额 GB/s', { size: 9 }));
  body.push(cap(380, 252, '填满 64 MB（教学缓冲）', { size: 9 }));
  const skews = [
    ['1.09（EPLB 残差）', '54.5', '4.5', '14.2 ms'],
    ['2', '100', '50', '1.28 ms'],
    ['4', '200', '150', '0.427 ms'],
  ];
  skews.forEach(([a, b, c, d], i) => {
    const y = 274 + i * 22;
    body.push(cap(36, y, a, { size: 9.5 }));
    body.push(cap(150, y, b, { mono: true, size: 9.5 }));
    body.push(cap(260, y, c, { mono: true, size: 9.5 }));
    body.push(cap(380, y, d, { mono: true, size: 10, color: i > 0 ? 'var(--seal-deep)' : undefined, weight: i > 0 ? '650' : undefined }));
    body.push(`<rect class="dia-bar dia-bar-${i === 0 ? 'soft' : 'seal'}" x="480" y="${y - 11}" width="${parseFloat(d) * 8}" height="14" rx="3"/>`);
  });
  body.push(cap(36, 344, '读数：偏斜越大队列填得越快、ECN 标记越早触发 —— 拥塞全来自偏斜，不是 a2a 语义固有（σ=1 反例）', { size: 9.5 }));

  body.push(cap(36, 372, '拥塞差：ring all-reduce 邻居链路持续均匀负载、无 incast（每 rank 只收 2 邻居）；a2a 完全图 + 偏斜 → 突发 incast', { size: 10 }));
  body.push(cap(36, 390, '治理两手：模型侧 EPLB 压 σ（moe_lb 1.090 → 吞吐 91.7%）；网络侧 ECN/PFC 调参 + AR 散列（10.2 §6/§7）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 408, 'DCQCN 论文评估 16:1/32:1 incast —— 正是 a2a 热卡多对一的形态（9.3 recap）', { size: 9.5 }));

  return scene(720, 424, body, { label: 'a2a incast 对照：均匀流量每 sender 出口均分 n-1 份（双随机）入口恰线速 C；偏斜使热卡入口 σC 超额 σ-1 倍 C 累积，σ=2 时 1.28 毫秒填满 64MB 队列，拥塞当且仅当 σ 大于 1 且持续' }).svg();
}

/* ============================================================
 * 10.4 §4（block 0）— Ring Attention 双缓冲 overlap 时序 + 阈值/总时间
 * ============================================================ */
function ringAttentionOverlap() {
  const UID = 'g10h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Ring Attention 双缓冲轮转：t_comm ≤ t_compute ⟺ 完全藏住 —— overlap 条件是 roofline 把带宽换成互连', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 648, h: 150, label: '双缓冲轮转时序（rank r，缓冲 A/B 交替，非 causal 口径）' }));
  body.push(cap(52, 95, 'compute 流', { size: 9 }));
  body.push(cap(52, 143, 'comm 流', { size: 9 }));
  const ck = node({ id: 'ck', x: 150, y: 76, w: 240, h: 30, role: 'attn', title: 'attn(Q_r, A)', mono: true, titleSize: 10 });
  const ck1 = node({ id: 'ck1', x: 400, y: 76, w: 240, h: 30, role: 'attn', title: 'attn(Q_r, B)', mono: true, titleSize: 10 });
  const gk = node({ id: 'gk', x: 150, y: 124, w: 234, h: 30, role: 'proj', title: 'send A→r+1 | recv B←r−1', mono: true, titleSize: 8.5 });
  const gk1 = node({ id: 'gk1', x: 400, y: 124, w: 234, h: 30, role: 'proj', title: 'send B→r+1 | recv A←r−1', mono: true, titleSize: 8.5 });
  nodes.push(ck, ck1, gk, gk1); body.push(ck.svg, ck1.svg, gk.svg, gk1.svg);
  // 脉冲走 comm 卡下方通道（y162），不压卡内标题字带
  body.push(pathPulse('M 150 162 H 384', { dur: 1.6, delay: 0 }));
  body.push(pathPulse('M 400 162 H 634', { dur: 1.6, delay: 0.8 }));
  body.push(cap(267, 178, '一个 ncclGroup', { size: 8, anchor: 'middle' }));
  body.push(cap(517, 178, '一个 ncclGroup', { size: 8, anchor: 'middle' }));
  // 依赖：compute k+1 等 recv k；group k+1 等 compute k（WAR）
  body.push(`<path class="dia-edge dash" d="M 384 139 C 394 139, 394 91, 400 91"/>`);
  body.push(arrowHead(400, 91, 1, 0));
  body.push(cap(300, 192, '依赖：compute 等 recv 完成（虚线）；group 等上一步 compute 完成（WAR，见坑注）', { size: 8.5 }));
  body.push(cap(36, 216, 't_comm = 0.977·t_compute → comm 段被 compute 段完全覆盖，步时间 = t_compute（通信零成本，非流量为零）', { size: 9.5 }));

  // overlap 条件 + 阈值表
  body.push(cap(36, 242, '命题 1：T_blk ≥ bytes·F_eff/(2g·BW)（式 4）⟺ I_blk ≥ R_comm（式 5，roofline 同构：HBM 带宽换互连）', { size: 10 }));
  body.push(cap(36, 264, '环放置', { size: 9 }));
  body.push(cap(250, 264, '最慢边 BW', { size: 9 }));
  body.push(cap(360, 264, '阈值', { size: 9 }));
  body.push(cap(470, 264, 'T_blk=8192 读法', { size: 9 }));
  const thr = [
    ['跨机 CX7（n=16）', '50', '8000 token', '比值 0.977 勉强藏住（余量 2.3%）'],
    ['NVLink 域（n=8）', '370.8', '1079 token', '比值 0.066 轻松藏住'],
    ['跨机 T_blk=4096（n=32）', '50', '8000 token', '比值 1.95 通信暴露约一半'],
  ];
  thr.forEach(([a, b, c, d], i) => {
    const y = 286 + i * 22;
    body.push(cap(36, y, a, { size: 9.5 }));
    body.push(cap(250, y, b, { mono: true, size: 9.5 }));
    body.push(cap(360, y, c, { mono: true, size: 9.5 }));
    body.push(cap(470, y, d, { size: 9, color: i === 2 ? 'var(--seal-deep)' : undefined }));
  });

  // 总时间账
  body.push(cap(36, 360, '命题 2：t_ring = t_compute + (n−1)·max(t_comm, t_compute)（式 6）—— 章脊几何 n=16', { size: 10 }));
  body.push(cap(170, 383, 'ring overlap', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="372" width="${87.96 * 2.6}" height="14" rx="3"/>`);
  body.push(cap(417, 383, '87.96 ms = 16×5.498', { mono: true, size: 9.5 }));
  body.push(cap(626, 383, '省 47.8%', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(170, 407, '串行无重叠', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="396" width="${168.49 * 2.6}" height="14" rx="3"/>`);
  body.push(cap(626, 407, '168.49 ms', { mono: true, size: 9.5 }));

  body.push(cap(36, 436, 'WAR 竞态：recv 覆写缓冲 A 须等 compute 流上一步 kernel 完成 event —— 否则读到半覆写 KV、静默错误（无 NaN 哨兵）', { size: 9.5 }));
  body.push(cap(36, 454, 'send/recv 不融进 ncclGroupStart/End → 死锁（两边互等）；全 mask 步零计算但 send/recv 不能跳（垮环同步节奏）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 470, body, { label: 'Ring Attention 双缓冲时序：compute 流 attn A 与 attn B 各 240 宽，comm 流 group 234 宽被完全覆盖，依赖为 compute 等 recv、group 等上一步 compute；跨机阈值 8000 token 比值 0.977，ring 总时间 87.96 对串行 168.49 毫秒省 47.8%' }).svg();
}

/* ============================================================
 * 10.4 §6（block 1）— 逻辑环 → 物理 fabric：rail 对齐反转 + 最慢边步拍
 * ============================================================ */
function ringPhysicalMapping() {
  const UID = 'g10i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '逻辑环摆到物理 fabric：rail 对齐反转 node 1 遍历 —— 两条机间边都落同 rail GPU 对（1 跳 leaf）', { size: 12, color: 'var(--ink-soft)' }));

  // 朴素 rank 序
  body.push(zone({ x: 36, y: 56, w: 324, h: 150, label: '朴素 rank 序：机间边跨 rail' }));
  const n0 = node({ id: 'n0', x: 52, y: 92, w: 130, h: 44, role: 'data', title: 'node 0', sub: 'r0→r1→…→r7', mono: true, titleSize: 10.5, subSize: 8 });
  const n1 = node({ id: 'n1', x: 212, y: 92, w: 130, h: 44, role: 'data', title: 'node 1', sub: 'r8→r9→…→r15', mono: true, titleSize: 10.5, subSize: 8 });
  nodes.push(n0, n1); body.push(n0.svg, n1.svg);
  body.push(`<path class="dia-edge dash" d="M 182 114 H 212"/>`);
  body.push(arrowHead(212, 114, 1, 0));
  body.push(cap(197, 104, '跨 rail', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  const back1 = 'M 277 136 V 176 H 117 V 136';
  body.push(`<path class="dia-edge dash" d="${back1}"/>`);
  body.push(arrowHead(117, 136, 0, -1));
  body.push(cap(197, 170, 'r15→r0 跨 rail', { size: 8, anchor: 'middle' }));
  body.push(cap(52, 196, '机间边 r7→r8、r15→r0 跨 rail（经 spine，多跳）', { size: 8.5 }));

  // rail 对齐
  body.push(zone({ x: 396, y: 56, w: 288, h: 150, tone: 'seal', label: 'rail 对齐：同 rail 1 跳' }));
  const m0 = node({ id: 'm0', x: 412, y: 92, w: 110, h: 44, role: 'data', title: 'node 0', sub: 'r0→…→r7', mono: true, titleSize: 10.5, subSize: 8 });
  const m1 = node({ id: 'm1', x: 562, y: 92, w: 110, h: 44, role: 'attn', title: 'node 1（反转）', sub: 'r15→r14→…→r8', mono: true, titleSize: 9.5, subSize: 7.5 });
  nodes.push(m0, m1); body.push(m0.svg, m1.svg);
  const e1 = 'M 522 114 H 562';
  body.push(`<path class="dia-edge strong" d="${e1}"/>`);
  body.push(pathPulse(e1, { dur: 0.9, delay: 0 }));
  body.push(arrowHead(562, 114, 1, 0));
  body.push(cap(542, 104, 'rail 7', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  const back2 = 'M 617 136 V 176 H 467 V 136';
  body.push(`<path class="dia-edge strong" d="${back2}"/>`);
  body.push(pathPulse(back2, { dur: 1.6, delay: 0.5 }));
  body.push(arrowHead(467, 136, 0, -1));
  body.push(cap(542, 170, 'r8→r0（rail 0）', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(412, 196, '机间边 r7→r15（rail 7）、r8→r0（rail 0）—— 同 rail 一跳 leaf', { size: 8.5 }));

  body.push(cap(36, 232, '步拍 = 最慢边 = 机间 M_blk/50 GB/s = 5.369 ms；节点内 14 跳 370.8 GB/s 只决定非瓶颈 rank 的 slack', { size: 10 }));

  // 阈值表
  body.push(cap(36, 260, '环放置', { size: 9 }));
  body.push(cap(250, 260, '最慢边 BW', { size: 9 }));
  body.push(cap(360, 260, '阈值（bf16 MHA）', { size: 9 }));
  body.push(cap(480, 260, '章脊 T_blk=8192 读法', { size: 9 }));
  const rows = [
    ['单节点 NVLink 域（n=8）', '370.8', '1079 token', '比值 0.066 轻松藏住'],
    ['跨机 CX7 NDR400（n=16）', '50', '8000 token', '比值 0.977 勉强藏住'],
    ['跨机 CX8 + Gen5（n=16）', '64（PCIe 限）', '6250 token', '比值 0.763 余量放宽'],
  ];
  rows.forEach(([a, b, c, d], i) => {
    const y = 282 + i * 22;
    body.push(cap(36, y, a, { size: 9.5 }));
    body.push(cap(250, y, b, { mono: true, size: 9.5 }));
    body.push(cap(360, y, c, { mono: true, size: 9.5 }));
    body.push(cap(480, y, d, { size: 9 }));
  });

  body.push(cap(36, 354, 'raw 9× 带宽差（450/50）在模型有效口径是 7.4×（370.8/50）—— 阈值比同比例：8000/1079 ≈ 7.4', { size: 10 }));
  body.push(cap(36, 372, '三条工程路径：度数放置（CP 留 NVLink 域 / NVL72）/ 带宽升级（CX8）/ 多 NIC 聚合（rail）—— 阈值 BW 取环最慢边', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 390, 'causal 保守口径（zigzag 充分条件）跨机阈值翻倍 16000 token —— 章脊几何低于线：causal 超长序列对跨机带宽更敏感', { size: 9.5 }));

  return scene(720, 406, body, { label: '逻辑环到物理 fabric：朴素 rank 序机间边跨 rail 经 spine 多跳，rail 对齐反转 node 1 使 r7 到 r15 与 r8 到 r0 同 rail 一跳；步拍取最慢边机间 5.369 毫秒，域内阈值 1079 对跨机 8000 token' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/01-fabric-architecture-topology',
    replacesBlock: 0,
    title: 'fat-tree/Clos：1:1 全二分 iff k_up≥k_down，两级 k²/2 = 10368 NIC',
    svg: fatTreeClos,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/01-fabric-architecture-topology',
    replacesBlock: 1,
    title: 'rail-optimized：每 GPU rail 接不同 leaf，聚合 ×NIC 数（400 GB/s、÷8）',
    svg: railOptimized,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/02-topology-aware-collectives-sharp',
    replacesBlock: 0,
    title: '分层 ring 三相：每 NIC 218.75 MB vs 平环 1.969 GB（9.32 vs 39.4 ms）',
    svg: hierRing,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/02-topology-aware-collectives-sharp',
    afterHeading: /^## 4\. /,
    title: 'SHARP 网内聚合：每 rank 出口 M（→2×），万卡延迟 286.7 ms → 392 μs',
    svg: sharpAggregation,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/03-ep-alltoall-implementation',
    replacesBlock: 0,
    title: 'DeepEP normal 两阶段（v1.2.1）：RDMA 同 nvl_rank + NVLink 转发，去重 2.11×',
    svg: deepepNormal,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/03-ep-alltoall-implementation',
    afterHeading: /^## 5\. /,
    title: 'DeepEP LL hook（v1.2.1）：SEND/RECV 相分离 + IBGDA（V2.5 → Gin GDAKI）',
    svg: deepepLLHook,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/03-ep-alltoall-implementation',
    afterHeading: /^## 7\. /,
    title: 'a2a incast：热卡入口 σC，拥塞 iff σ>1 持续（σ=2 填 64 MB 1.28 ms）',
    svg: a2aIncast,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/04-point-to-point-sequence-parallel',
    replacesBlock: 0,
    title: 'Ring Attention 双缓冲 overlap：t_comm≤t_compute 全藏（87.96 vs 168.49 ms）',
    svg: ringAttentionOverlap,
  },
  {
    dim: 'gpu',
    slug: '10-fabric-and-scale-communication/04-point-to-point-sequence-parallel',
    replacesBlock: 1,
    title: '逻辑环放置：rail 对齐反转使机间边同 rail 1 跳，步拍 = 最慢边 5.369 ms',
    svg: ringPhysicalMapping,
  },
];
