// gpu P11（11-system-integration）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p09.mjs / gpu-p10.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；只用 dia-pulse（reduced-motion CSS 隐藏 → 静态终态）。
// 口径硬规则（图须晚于章 fix，W5 fig-2 教训）：
//   11.2 hook gap/wall = 33.3%（busy 3960 / wall 5934 / duty 0.667 / graph 1.50×）；roofline latency 点标 D（非 C，避并发符号 C）；
//   B200 SM = 148 [B] 微基准口径（官方 datasheet 不印）；MIG L2 随 memory slice 方向，H100 1g.20gb L2=1/8 例外 vs B200 1g.45gb 2/8；
//   XID 74 Blackwell 不适用 → 149 NETIR；ECC row remap 512 = 8× legacy 64；thermal HW slowdown ÷2 → compute-bound 上界 ÷2（memory-bound 不变）；
//   NVLink 18 link 断 1 → 425/450 = 94.4%；PCIe H2D = min(64,40) = 40（62.5%）；域聚合 7.2/14.4/130 TB/s。
// census：11.1 block0 形态层级 / block1 PCIe NUMA 拓扑 / block2 MIG 切片几何；
//         11.2 block0 nsys timeline / block1 三区判定流程 / block2 调优决策树 / block3 achieved roofline；
//         11.3 block0 检出-定位-处置链 / block1 XID 分类树 / block2 健康检查四层。
//         签名：11.3 §3 ECC CE/UE+remap、11.3 §5.2 thermal throttle、11.3 §6 链路计数器+降级。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 gpu-p07 先例）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

// 手工路径上的脉冲（L 形/回环/折线通道用 path 直写；渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}" ` +
  `calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1"/>` +
  `</circle>`;

// 显式箭头三角（kit 的 marker-end 被 CSS 关闭）
const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

/* ============================================================
 * 11.1 §2（block 0）— 形态层级：HGX 基板 → DGX 整机 → NVL72 机架
 * ============================================================ */
function serverForms() {
  const body = [];

  body.push(cap(36, 24, '形态层级：基板 → 整机 → 机架超节点 —— NVLink 域 8 → 8 → 72 GPU，铭牌功率 10.2 → 14.3 → ≈120 kW', { size: 12, color: 'var(--ink-soft)' }));

  // 三形态卡（横向递进）
  body.push(zone({ x: 36, y: 52, w: 204, h: 176, label: 'HGX 基板 · 交付 OEM' }));
  for (let j = 0; j < 8; j++) body.push(`<rect x="${52 + j * 19}" y="84" width="16" height="12" rx="2" fill="#f3f1ea" stroke="#c6c0af" stroke-width="0.8"/>`);
  for (let j = 0; j < 4; j++) body.push(`<rect x="${52 + j * 26}" y="102" width="20" height="9" rx="2" fill="#ecf0f6" stroke="#aebcd2" stroke-width="0.8"/>`);
  body.push(cap(52, 126, '8×SXM GPU + NVSwitch 板上互连', { size: 8.5 }));
  body.push(cap(52, 142, 'H100: 4 颗 / B200: 2 颗', { mono: true, size: 8.5 }));
  body.push(cap(52, 162, '无 CPU / 主机内存', { size: 9.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(52, 180, 'host 侧（PCIe/NIC/机箱）= OEM 自由度', { size: 8 }));
  body.push(cap(52, 196, '同基板不同 OEM 拓扑可不同', { size: 8 }));
  body.push(cap(52, 212, '只买基板得不到可开机系统', { size: 8 }));

  body.push(zone({ x: 258, y: 52, w: 204, h: 176, label: 'DGX 整机 · NVIDIA 自有品牌' }));
  body.push(cap(274, 88, '+ 2×Xeon + 2 TB DIMM + 8×CX7', { size: 8 }));
  body.push(cap(274, 104, '+ 6×3.3 kW PSU + 机箱', { size: 8 }));
  body.push(cap(274, 128, 'DGX H100: 8U / 10.2 kW max', { mono: true, size: 9 }));
  body.push(cap(274, 144, '风冷 1105 CFM @80% fan PWM', { size: 8 }));
  body.push(cap(274, 166, 'DGX B200: 10U / 14.3 kW max', { mono: true, size: 9 }));
  body.push(cap(274, 182, '风冷 1550 CFM', { size: 8 }));
  body.push(cap(274, 206, 'NVLink 域 = 8 GPU（机箱内）', { size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(zone({ x: 480, y: 52, w: 204, h: 176, tone: 'seal', label: 'GB200 NVL72 · ≈120 kW 液冷' }));
  body.push(cap(496, 88, '18 compute trays（2 Grace + 4 GPU/tray）', { size: 7.5 }));
  body.push(cap(496, 106, '= 72 GPU + 36 Grace', { mono: true, size: 9 }));
  body.push(cap(496, 126, '9 switch trays（2×72-port chip/tray）', { size: 7.5 }));
  body.push(cap(496, 142, '+ 8 power shelves · 无源铜缆背板', { size: 8 }));
  body.push(cap(496, 162, 'CPU↔GPU: NVLink-C2C 900 GB/s 双向', { size: 7.5 }));
  body.push(cap(496, 184, 'NVLink 域 = 72 GPU（跨 tray 在域内）', { size: 8.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(496, 202, 'cold plate + manifold 液冷', { size: 8 }));
  body.push(cap(496, 218, '网络/存储仍风冷 + leak detection', { size: 7.5 }));

  // 递进箭头
  [['M 240 140 H 258', 0], ['M 462 140 H 480', 0.4]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.2, delay: dl }));
  });
  body.push(arrowHead(258, 140, 1, 0)); body.push(arrowHead(480, 140, 1, 0));
  body.push(cap(249, 244, '+ CPU/内存/NIC/存储/PSU/机箱', { size: 7.5, anchor: 'middle' }));
  body.push(cap(471, 244, 'switch tray 把域扩到机架级', { size: 7.5, anchor: 'middle' }));

  // 域聚合账
  body.push(cap(36, 268, 'NVLink 域聚合（双向，recap 7.1 式 (4)）—— GPU 数 9×，域聚合 18×（NVL72 vs H100）', { size: 10 }));
  const dom = [['DGX H100（8 GPU）', 7.2, '7.2 TB/s'], ['DGX B200（8 GPU）', 14.4, '14.4 TB/s'], ['NVL72（72 GPU）', 129.6, '130 TB/s（129.6）']];
  dom.forEach(([lab, v, val], i) => {
    const y = 280 + i * 22;
    body.push(cap(170, y + 11, lab, { size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${i === 2 ? 'seal' : 'soft'}" x="180" y="${y}" width="${v * 3}" height="14" rx="3"/>`);
    body.push(cap(180 + v * 3 + 8, y + 11, val, { mono: true, size: 9.5, color: i === 2 ? 'var(--seal-deep)' : undefined, weight: i === 2 ? '650' : undefined }));
  });

  body.push(cap(36, 360, 'GPU-only TDP ≠ 铭牌：5.6/10.2 = 54.9% · 8/14.3 = 55.9% · 86.4/120 = 72% —— 设计一律取铭牌（GPU-only 只算占比）', { size: 10 }));
  body.push(cap(36, 378, '每 GPU 系统级功率：H100 1.275 / B200 1.7875 / NVL72 1.667 kW —— 机架摊薄基础设施，每 GPU 反低于 B200', { size: 9.5 }));
  body.push(cap(36, 400, '液冷是物理必然：ζ_water/ζ_air ≈ 3483；120 kW @15 K 需风 6.67 m³/s（≈14100 CFM）vs 水 1.9 L/s（流量 ÷~3500）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 422, 'N+N 冗余（式 1）：0.5×8 shelf×6 PSU×5.5 kW = 132 ≥ 120；DGX 4×3.3 = 13.2 ≥ 10.2 / 5×3.3 = 16.5 ≥ 14.3（≥3 PSU boot 是降级阈值）', { size: 9.5 }));
  body.push(cap(36, 444, '跨机箱同机架 ≠ 同域：流量下 NIC 走 RDMA（节点出口 8×CX7 = 400 GB/s 单向，recap 7.2）；TP/EP > 8 → NVL72 或跨机', { size: 9.5 }));

  return scene(720, 460, body, { label: '服务器三形态递进：HGX 基板八 SXM GPU 加 NVSwitch 无 CPU 交付 OEM，DGX 整机加 CPU 内存 NIC 电源机箱成八卡域十點二至十四點三千瓦风冷，GB200 NVL72 机架十八 compute tray 加九 switch tray 成七十二卡域约一百二十千瓦液冷；域聚合七點二至十四點四至一百三十 TB/s，GPU-only 占铭牌五十四點九至七十二百分比，液冷因水空气体积热容差三千四百八十三倍' }).svg();
}

/* ============================================================
 * 11.1 §5（block 1）— DGX H100 PCIe/NUMA 拓扑 + affinity 三规则
 * ============================================================ */
function pcieNuma() {
  const UID = 'g11b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DGX H100 PCIe/NUMA 拓扑（教学示意，非官方接线）：每 socket 4 GPU + 4 NIC，H2D 取路径各段 min', { size: 12, color: 'var(--ink-soft)' }));

  // 两个 NUMA socket
  body.push(zone({ x: 36, y: 52, w: 300, h: 210, label: 'NUMA node 0 · CPU0（Xeon 8480C）' }));
  body.push(zone({ x: 384, y: 52, w: 300, h: 210, label: 'NUMA node 1 · CPU1（Xeon 8480C）' }));

  const dram0 = node({ id: 'dram0', x: 52, y: 86, w: 128, h: 44, role: 'cache', title: '本地 DRAM', sub: 'NUMA node 0', titleSize: 10, subSize: 8 });
  const cpu0 = node({ id: 'cpu0', x: 196, y: 86, w: 124, h: 44, role: 'op', title: 'CPU0 · 8480C', sub: 'root complex', mono: true, titleSize: 9, subSize: 8 });
  const dram1 = node({ id: 'dram1', x: 544, y: 86, w: 124, h: 44, role: 'cache', title: '本地 DRAM', sub: 'NUMA node 1', titleSize: 10, subSize: 8 });
  const cpu1 = node({ id: 'cpu1', x: 400, y: 86, w: 124, h: 44, role: 'op', title: 'CPU1 · 8480C', sub: 'root complex', mono: true, titleSize: 9, subSize: 8 });
  nodes.push(dram0, cpu0, dram1, cpu1); body.push(dram0.svg, cpu0.svg, dram1.svg, cpu1.svg);
  body.push(`<line class="dia-edge" x1="180" y1="108" x2="196" y2="108"/>`);
  body.push(`<line class="dia-edge" x1="524" y1="108" x2="544" y2="108"/>`);

  // inter-socket link
  const isd = 'M 320 108 H 400';
  body.push(`<path class="dia-edge strong" d="${isd}"/>`);
  body.push(pathPulse(isd, { dur: 1.4, delay: 0.2 }));
  body.push(arrowHead(400, 108, 1, 0)); body.push(arrowHead(320, 108, -1, 0));
  body.push(cap(360, 98, 'inter-socket link', { size: 8, anchor: 'middle' }));
  body.push(cap(360, 126, '40 GB/s（教学假设）', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));

  // PCIe switch + GPU/NIC 扇出
  const sw0 = node({ id: 'sw0', x: 52, y: 158, w: 268, h: 30, role: 'proj', title: 'PCIe switch（Gen5 lane 扇出 · 教学示意）', titleSize: 8.5 });
  const sw1 = node({ id: 'sw1', x: 400, y: 158, w: 268, h: 30, role: 'proj', title: 'PCIe switch（Gen5 lane 扇出 · 教学示意）', titleSize: 8.5 });
  nodes.push(sw0, sw1); body.push(sw0.svg, sw1.svg);
  body.push(`<line class="dia-edge" x1="258" y1="130" x2="258" y2="158"/>`);
  body.push(`<line class="dia-edge" x1="462" y1="130" x2="462" y2="158"/>`);
  body.push(cap(266, 148, 'PCIe Gen5', { mono: true, size: 7.5 }));
  body.push(cap(470, 148, 'PCIe Gen5', { mono: true, size: 7.5 }));
  for (let j = 0; j < 4; j++) {
    const x0 = 52 + j * 68, x1 = 400 + j * 68;
    body.push(`<line class="dia-edge" x1="${x0 + 30}" y1="188" x2="${x0 + 30}" y2="200"/>`);
    body.push(`<line class="dia-edge" x1="${x1 + 30}" y1="188" x2="${x1 + 30}" y2="200"/>`);
    body.push(`<rect x="${x0}" y="200" width="60" height="20" rx="4" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1"/>`);
    body.push(cap(x0 + 30, 214, `GPU${j}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(`<rect x="${x0}" y="226" width="60" height="18" rx="4" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1"/>`);
    body.push(cap(x0 + 30, 239, `NIC${j}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(`<rect x="${x1}" y="200" width="60" height="20" rx="4" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1"/>`);
    body.push(cap(x1 + 30, 214, `GPU${j + 4}`, { mono: true, size: 8, anchor: 'middle' }));
    body.push(`<rect x="${x1}" y="226" width="60" height="18" rx="4" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1"/>`);
    body.push(cap(x1 + 30, 239, `NIC${j + 4}`, { mono: true, size: 8, anchor: 'middle' }));
  }
  body.push(cap(52, 254, 'CX7 400G · GPU-NIC 同侧 rail 对齐（10.1）', { size: 7.5 }));
  body.push(cap(400, 254, 'CX7 400G · GPU-NIC 同侧 rail 对齐', { size: 7.5 }));

  // H2D 本地路径脉冲（DRAM0 → CPU0 → switch → GPU0）
  [['M 180 108 H 196', 0], ['M 258 130 V 158', 0.3], ['M 82 188 V 200', 0.6]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 0.9, delay: dl }));
  });
  body.push(arrowHead(196, 108, 1, 0)); body.push(arrowHead(258, 158, 0, 1)); body.push(arrowHead(82, 200, 0, 1));

  // affinity 三规则
  body.push(cap(36, 286, '规则 1 · host buffer 绑 GPU 本地 NUMA node：H2D 带宽 = min(PCIe, inter-socket)（recap 7.2 式 (4)，不重推）', { size: 10 }));
  body.push(cap(170, 309, '本地 Gen5 x16', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="298" width="${64 * 3.5}" height="14" rx="3"/>`);
  body.push(cap(412, 309, '64 GB/s', { mono: true, size: 9.5 }));
  body.push(cap(170, 331, '跨 socket（教学例）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="320" width="${40 * 3.5}" height="14" rx="3"/>`);
  body.push(cap(328, 331, 'min(64, 40) = 40 GB/s —— 只剩本地 62.5%', { mono: true, size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(36, 356, '绑定用 numactl --membind，对比测量用 nvbandwidth（方法归 7.2 §6）；同 switch 下 GPU/NIC 可不经 root complex 直通', { size: 9 }));
  body.push(cap(36, 374, '规则 2 · GPU 与 NIC 同 PCIe switch/root complex → GDR 最优：min(NIC 50, PCIe 64) = 50 GB/s NIC 线速限（9.2）；rail 对齐（10.1）', { size: 9.5 }));
  body.push(cap(36, 392, '规则 3 · nvidia-smi topo -m 核对逐对路径类型（NVLink / 同 switch / 经 root complex / 跨 socket）；GPU↔GPU 走 NVLink/NVSwitch 不走 PCIe 树', { size: 9.5 }));

  body.push(zone({ x: 36, y: 408, w: 648, h: 68, tone: 'seal', label: 'GB200 增量：CPU-GPU 数据路径不再经 PCIe' }));
  body.push(cap(52, 440, 'Grace↔Blackwell NVLink-C2C 900 GB/s 双向一致 = 7× PCIe Gen5 128 GB/s 双向（recap 7.1 式 (5)）', { size: 9 }));
  body.push(cap(52, 458, 'H2D 不再受 PCIe/NUMA 约束；PCIe 退到 NIC/NVMe 通道（每 compute tray 4×CX7 + 2×BF3）', { size: 9 }));

  return scene(720, 492, body, { label: 'DGX H100 PCIe NUMA 教学拓扑：两 Xeon 各成 NUMA node 挂本地 DRAM 与 root complex，经 PCIe Gen5 switch 扇出四 GPU 四 NIC 同侧 rail 对齐，inter-socket link 教学假设四十 GB/s；H2D 取路径 min 跨 socket min 六十四四十得四十只剩本地六十二點五百分比，GPU NIC 同 switch GDR 最优五十 GB/s NIC 线速限，GB200 C2C 九百 GB/s 七倍 PCIe 免此账' }).svg();
}

/* ============================================================
 * 11.1 §6（block 2）— MIG 切片几何：8 memory slice + 7 SM slice
 * ============================================================ */
function migSlices() {
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'MIG 切片几何（H100 7×1g.10gb placement）：memory slice 1/8 × 8 + SM slice 1/7 × 7 —— GI_k = mem_k + SM sl_k', { size: 12, color: 'var(--ink-soft)' }));

  // 显存侧 8 片
  body.push(cap(36, 56, '显存侧：8 个 memory slice，每片 1/8（容量 10 GB + 带宽 1/8 + MC + L2 1/8）', { size: 10 }));
  const memW = (640 - 7 * 4) / 8;
  for (let k = 0; k < 8; k++) {
    const x = 36 + k * (memW + 4);
    const idle = k === 7;
    const n = node({ id: `mem${k}`, x, y: 64, w: memW, h: 44, role: idle ? 'cache' : 'data', title: `mem${k}`, sub: idle ? '空闲 1/8' : '10 GB · 1/8', mono: true, titleSize: 9, subSize: 7 });
    nodes.push(n); body.push(n.svg);
  }
  // SM 侧 7 片
  body.push(cap(36, 220, 'SM 侧：7 个 SM slice，每片 1/7（roughly ≈18.9 SM，132/7 = 18.86 分数记账）', { size: 10 }));
  const smW = (640 - 6 * 4) / 7;
  for (let k = 0; k < 7; k++) {
    const x = 36 + k * (smW + 4);
    const n = node({ id: `sm${k}`, x, y: 158, w: smW, h: 44, role: 'proj', title: `SM sl${k}`, sub: '1/7 · ≈18.9 SM', mono: true, titleSize: 9, subSize: 7 });
    nodes.push(n); body.push(n.svg);
  }
  // GI 配对连线 mem_k → SM sl_k
  for (let k = 0; k < 7; k++) {
    const mx = 36 + k * (memW + 4) + memW / 2;
    const sx = 36 + k * (smW + 4) + smW / 2;
    const d = `M ${mx.toFixed(1)} 108 C ${mx.toFixed(1)} 133, ${sx.toFixed(1)} 133, ${sx.toFixed(1)} 158`;
    body.push(`<path class="dia-edge" d="${d}"/>`);
    if (k === 3) body.push(pathPulse(d, { dur: 1.4, delay: 0.2 }));
  }
  // mem7 无配对
  body.push(`<path class="dia-edge dash" d="M ${(36 + 7 * (memW + 4) + memW / 2).toFixed(1)} 108 V 128"/>`);
  body.push(cap(638, 140, '无 SM 切片可配', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 240, 'GI_k = mem_k + SM sl_k → 7 实例：SM 7/7 用满、显存 7/8 —— mem7 = 1/8 显存（10 GB）必然空闲（命题 2）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 258, '实例上界 = min(SM 切片, memory 切片) = min(7, 8) = 7（A30 两类各 4 → 上界 4）；分区守恒：任意合法组合分数和 ≤ 1', { size: 9.5 }));

  // GI/CI 两级 + 资源分数账
  body.push(zone({ x: 36, y: 274, w: 324, h: 110, label: '两级分区：GI = 内存 QoS 边界 / CI = SM 子集' }));
  body.push(cap(52, 306, 'GI = GPU 切片 + 引擎（DMA/NVDEC）', { size: 8.5 }));
  body.push(cap(52, 322, '—— 内存保护 + 带宽 QoS（邻居抢不走 1/8）', { size: 8.5 }));
  body.push(cap(52, 342, 'CI = GI 内 SM 切片子集，共享 GI 显存/引擎', { size: 8.5 }));
  body.push(cap(52, 358, '各有专属 SM；1g.10gb = 1+1 切片', { size: 8.5 }));
  body.push(cap(52, 376, 'MPS ≠ 物理分区：百分比逻辑共享、无错误隔离', { size: 8, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 384, y: 274, w: 300, h: 110, label: '1g.10gb 资源分数账（式 3，H100 主口径）' }));
  body.push(cap(400, 306, 'f_SM = s_SM/7 · f_mem = s_mem/8', { mono: true, size: 9 }));
  body.push(cap(400, 326, 'SM: 132/7 ≈ 18.9（roughly）', { mono: true, size: 8.5 }));
  body.push(cap(400, 344, '容量: 80/8 = 10 GB', { mono: true, size: 8.5 }));
  body.push(cap(400, 362, '带宽: 3350/8 = 418.75 GB/s', { mono: true, size: 8.5 }));
  body.push(cap(400, 380, 'L2: 50/8 = 6.25 MB', { mono: true, size: 8.5 }));

  // L2 方向 + 跨代例外硬注
  body.push(zone({ x: 36, y: 400, w: 648, h: 76, tone: 'seal', label: 'L2 随 memory slice 方向切分（每片 1/8），不是随 SM slice' }));
  body.push(cap(52, 432, '具体数值一律以官方 profile 表 L2 列为准，不做跨代外推：H100 1g.20gb（2 memory slice）L2 = 1/8（6.25 MB）例外行', { size: 9 }));
  body.push(cap(52, 450, 'B200 对应 1g.45gb 印 L2 = 2/8 —— 两代表不一致；按 memory slice 数反推 L2 会错成 12.5 MB（§6 坑注）', { size: 9 }));
  body.push(cap(52, 468, 'B200 1g.23gb：SM 148/7 ≈ 21.1 [B]、180/8 = 22.5 GB（名 23gb 舍入）、7700/8 = 962.5 GB/s', { size: 9 }));

  return scene(720, 492, body, { label: 'MIG 切片几何 H100 七乘 1g.10gb：显存侧八 memory slice 每片八分之一含十 GB 容量带宽 MC 与 L2，SM 侧七 slice 每片七分之一约十八點九 SM，GI_k 配 mem_k 加 SM sl_k 得七实例 SM 用满显存七分之八，mem7 十 GB 必然空闲；实例上界 min 七八得七，GI 内存 QoS 边界 CI 共享显存引擎，L2 随 memory 方向但 H100 1g.20gb 八分之一例外对 B200 1g.45gb 八分之二，B200 SM 一百四十八为 B 口径' }).svg();
}

/* ============================================================
 * 11.2 §2（block 0）— Nsight Systems timeline：间隙归因 + 三段账
 * ============================================================ */
function nsysTimeline() {
  const body = [];

  body.push(cap(36, 24, 'Nsight Systems timeline：CPU 提交间隔 18 μs > kernel 12 μs —— GPU 等 CPU，间隙占墙钟 33.3%（教学假设）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 648, h: 180, label: '单 pass trace（不 replay / 不 flush / 不锁频，接近生产态）' }));
  // 行标签
  body.push(cap(52, 96, 'CPU 线程行', { size: 8.5 }));
  body.push(cap(52, 130, 'NVTX 行', { size: 8.5 }));
  body.push(cap(52, 166, 'GPU kernel 行', { size: 8.5 }));
  body.push(cap(52, 206, 'GPU copy 行', { size: 8.5 }));

  const x0 = 160, pitch = 72, kw = 48, off = 12;
  // CPU 行：连续忙碌底条 + launch 刻度
  body.push(`<rect class="dia-bar dia-bar-soft" x="${x0}" y="84" width="${pitch * 6}" height="16" rx="3"/>`);
  for (let i = 0; i < 6; i++) body.push(`<rect x="${x0 + i * pitch}" y="84" width="6" height="16" rx="2" fill="#a83228"/>`);
  body.push(cap(x0 + 10, 78, 'launch', { mono: true, size: 7.5 }));
  body.push(cap(x0 + pitch * 6 + 8, 96, 'CPU 连轴转', { size: 7.5 }));
  body.push(pathPulse(`M ${x0} 92 H ${x0 + pitch * 6}`, { dur: 2.0, delay: 0 }));
  // NVTX 行
  body.push(`<rect x="${x0}" y="118" width="${pitch * 6}" height="16" rx="4" fill="#f5efe2" stroke="#cbb98d" stroke-width="1"/>`);
  body.push(cap(x0 + pitch * 3, 130, 'decode_step N（NVTX range → 投影到 GPU timeline）', { size: 8, anchor: 'middle' }));
  // GPU kernel 行：间隙底衬 + kernel 卡
  for (let i = 0; i < 6; i++) {
    if (i < 5) body.push(`<rect x="${x0 + i * pitch + off + kw}" y="152" width="${pitch - kw}" height="20" fill="#f3ecdc"/>`);
    const n = node({ id: `k${i}`, x: x0 + i * pitch + off, y: 152, w: kw, h: 20, role: 'attn', title: `K${i + 1}`, mono: true, titleSize: 8 });
    body.push(n.svg);
  }
  body.push(cap(x0 + off + kw + (pitch - kw) / 2, 186, '间隙 6 μs', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(x0 + pitch * 6 + 8, 166, 'GPU 有隙', { size: 7.5, color: 'var(--seal-deep)' }));
  // GPU copy 行
  body.push(`<rect x="${x0}" y="198" width="60" height="14" rx="3" fill="#ecf2ec" stroke="#a9bfa9" stroke-width="1"/>`);
  body.push(cap(x0 + 30, 209, 'H2D', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(`<rect x="${x0 + pitch * 5 + 12}" y="198" width="60" height="14" rx="3" fill="#ecf2ec" stroke="#a9bfa9" stroke-width="1"/>`);
  body.push(cap(x0 + pitch * 5 + 42, 209, 'D2H', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(x0 + 80, 209, 'CE 引擎 · 区分 H2D/D2H/D2D', { size: 7.5 }));
  // correlation 虚线（API0 → K1）
  body.push(`<path class="dia-edge dash" d="M ${x0 + 3} 100 C ${x0 + 3} 126, ${x0 + off + kw / 2} 126, ${x0 + off + kw / 2} 152"/>`);
  body.push(cap(x0 + 34, 140, 'correlation', { mono: true, size: 7 }));
  body.push(cap(52, 222, 'memcpy 行区分 copy engine · CUDA graph trace 三档 graph/node/host-only（node 档开销高）', { size: 7.5 }));

  // 三段账
  body.push(cap(36, 252, '三段账（330 kernel × 12 μs、dispatch 18 μs；式 (1)(2)(3)）：t_busy = 3960 μs · t_wall = 329×18+12 = 5934 μs · duty = 0.667', { size: 10 }));
  const sc = 0.06;
  body.push(cap(170, 277, '修复前（CPU-bound）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="266" width="${3960 * sc}" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="${180 + 3960 * sc}" y="266" width="${1974 * sc}" height="14" rx="3"/>`);
  body.push(cap(544, 277, '5934 μs · duty 0.667', { mono: true, size: 9 }));
  body.push(cap(180 + 3960 * sc + 1974 * sc / 2, 292, 'gap 1974 μs = 33.3%', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(170, 313, 'graph 修复后', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="302" width="${3960 * sc}" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="${180 + 3960 * sc}" y="302" width="3" height="14" rx="1.5"/>`);
  body.push(cap(426, 313, '≈3965 μs · 加速 1.50×', { mono: true, size: 9, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 340, 'signature：GPU 行有隙、CPU 行连轴转、无 sync API → CPU-bound 确诊；修法 = CUDA Graph 整段录图一次重放（recap 3.3）', { size: 9.5 }));
  body.push(cap(36, 358, '间隙归因三类：CPU 忙无 sync → CPU-bound / cudaStreamSynchronize·event wait → 同步等待 / CPU 闲 GPU 等前序 → GPU 侧依赖', { size: 9.5 }));
  body.push(cap(36, 376, 'drill-down 顺序：先修占比最大的损失段（间隙 33.3%），再进 kernel 内部 —— 1.50× 来自提交路径，kernel 一个没动', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 392, body, { label: 'Nsight Systems 四行 timeline：CPU 线程行连轴转每十八微秒 launch 一次，NVTX range 投影 GPU 时间轴，GPU kernel 行 K1 到 K6 各十二微秒间有六微秒间隙，copy 行 H2D D2H 走 CE；三百三十 kernel 三段账 busy 三千九百六十 wall 五千九百三十四 duty 零點六六七间隙百分之三十三點三，CUDA Graph 修复后墙钟约三千九百六十五加速一點五倍 kernel 一个没动' }).svg();
}

/* ============================================================
 * 11.2 §4（block 1）— Bottleneck 三区判定流程 + 四因子分解
 * ============================================================ */
function bottleneckFlow() {
  const UID = 'g11e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Bottleneck 三区判定：nsys 先拆三段，NCU 读数接进判定每个输入 —— I_ach 对 R、C 对 L_req（recap 2.4）', { size: 12, color: 'var(--ink-soft)' }));

  // 左侧判定流
  const n1 = node({ id: 'n1', x: 60, y: 52, w: 180, h: 32, role: 'op', title: 'nsys 三段分解', titleSize: 10.5 });
  const d1 = node({ id: 'd1', x: 60, y: 104, w: 180, h: 32, role: 'proj', title: 'duty 低？', titleSize: 10.5 });
  const n2 = node({ id: 'n2', x: 50, y: 156, w: 200, h: 48, role: 'op', title: '按时间贡献取 top kernel → NCU', sub: 'I_ach=FLOPs/bytes · u_B/u_F←SOL · C←O×ILP×s_warp', titleSize: 9, subSize: 6.5 });
  const d2 = node({ id: 'd2', x: 60, y: 224, w: 180, h: 32, role: 'proj', title: 'I_ach ≥ R？', mono: true, titleSize: 10.5 });
  const d3 = node({ id: 'd3', x: 60, y: 276, w: 180, h: 32, role: 'proj', title: 'C ≥ L_req（208）？', mono: true, titleSize: 10 });
  nodes.push(n1, d1, n2, d2, d3); body.push(n1.svg, d1.svg, n2.svg, d2.svg, d3.svg);
  [['M 150 84 V 104', 0, 104], ['M 150 136 V 156', 0.2, 156], ['M 150 204 V 224', 0.4, 224], ['M 150 256 V 276', 0.6, 276]].forEach(([d, dl, ey]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.0, delay: dl }));
    body.push(arrowHead(150, ey, 0, 1));
  });
  body.push(cap(158, 148, '否：时间都在 kernel 里', { size: 7.5 }));
  body.push(cap(158, 268, '否', { size: 7.5 }));

  // 右侧三区结果卡 + 系统级分支
  const sys = node({ id: 'sys', x: 440, y: 52, w: 244, h: 56, role: 'norm', title: '系统级分支（duty 低）', sub: 'CUDA Graph / fusion / stream 重组（§5）', titleSize: 10.5, subSize: 8 });
  const comp = node({ id: 'comp', x: 440, y: 146, w: 244, h: 60, role: 'attn', title: 'compute-bound（I ≥ R）', sub: '上限 F_peak · 看 u_F 与指令 mix（5.2/6.1）', titleSize: 10.5, subSize: 8 });
  const mem = node({ id: 'mem', x: 440, y: 236, w: 244, h: 60, role: 'proj', title: 'memory-bound（I < R，C ≥ L_req）', sub: '上限 I·BW · 看 Sectors/Req 与流量（4.1/4.3/5.4）', titleSize: 9.5, subSize: 7.5 });
  const lat = node({ id: 'lat', x: 440, y: 326, w: 244, h: 60, role: 'cache', title: 'latency-bound（C < L_req）', sub: 'F_ach = I·BW·min(1, C/L_req) · stall 签名交叉验证', titleSize: 10, subSize: 7.5 });
  nodes.push(sys, comp, mem, lat); body.push(sys.svg, comp.svg, mem.svg, lat.svg);
  // 分支边（是）
  [['M 240 120 C 340 120, 340 80, 440 80', 0.1], ['M 240 240 C 340 240, 340 176, 440 176', 0.3], ['M 240 292 C 340 292, 340 266, 440 266', 0.5]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.3, delay: dl }));
  });
  body.push(arrowHead(440, 80, 1, 0)); body.push(arrowHead(440, 176, 1, 0)); body.push(arrowHead(440, 266, 1, 0));
  body.push(cap(300, 108, '是', { size: 8 })); body.push(cap(300, 228, '是', { size: 8 })); body.push(cap(300, 282, '是', { size: 8 }));
  // latency 肘形（否）
  const elb = 'M 150 308 V 356 H 440';
  body.push(`<path class="dia-edge" d="${elb}"/>`);
  body.push(pathPulse(elb, { dur: 1.5, delay: 0.7 }));
  body.push(arrowHead(440, 356, 1, 0));
  body.push(cap(158, 330, '否', { size: 7.5 }));

  body.push(cap(36, 406, 'stall 签名 = 交叉验证非替代：long_scoreboard 主导 + not_selected ≈ 0 = 延迟没藏住（recap 1.4）；occupancy 已高仍 long_scoreboard → 查 coalescing/带宽打满', { size: 9 }));

  // 四因子分解
  body.push(cap(36, 428, '命题 1 · 端到端四因子：BW_e2e = BW·u_B·η·duty（式 4）—— 哪个因子离 1 最远先修哪个', { size: 10 }));
  body.push(cap(170, 451, '修复前（duty 0.667）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="440" width="${1901 * 0.12}" height="14" rx="3"/>`);
  body.push(cap(416, 451, '1901 GB/s（56.7%）', { mono: true, size: 9 }));
  body.push(cap(170, 473, 'graph 后（duty 0.999）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="462" width="${2846 * 0.12}" height="14" rx="3"/>`);
  body.push(cap(530, 473, '2846 GB/s（84.9%）', { mono: true, size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(36, 496, 'u_B 0.85→1 只有 1.18×，修间隙有 1.50× —— 四因子哪个离 1 最远先修哪个（drill-down 定量依据）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 512, body, { label: 'Bottleneck 三区判定流程：nsys 三段分解判 duty 低走系统级分支 graph 融合 stream，否则取 top kernel NCU 采集 I_ach u_B u_F 与并发 C，I_ach 不小于 R 为 compute-bound 看 u_F 指令 mix，否则 C 不小于 L_req 二百零八为 memory-bound 看 Sectors/Req，否则 latency-bound 用 stall 签名交叉验证；命题一四因子 BW_e2e 等于 BW 乘 u_B 乘 eta 乘 duty，修复前一千九百零一 GB/s 五十六點七百分比 graph 后二千八百四十六八十四點九百分比' }).svg();
}

/* ============================================================
 * 11.2 §5（block 2）— 调优决策树：从区到手段 + 命题 2 对账
 * ============================================================ */
function tuningTree() {
  const body = [];

  body.push(cap(36, 24, '调优决策树：从区到手段 —— 先系统级后 kernel 级；每手段有机制章（recap，本节交付选择与对账）', { size: 12, color: 'var(--ink-soft)' }));

  const lanes = [
    { y: 48, h: 72, tone: 'ink', label: 'duty 低（系统级）—— 承接 §2', rows: [
      ['CPU-bound', 'CUDA Graph（3.3）/ fusion 减 kernel 数（5.4）'],
      ['sync/依赖', 'stream/event 重组（3.3）/ copy-compute 重叠（3.3/4.4）'],
    ] },
    { y: 132, h: 124, tone: 'ink', label: 'memory-bound（I < R，u_B 高）', rows: [
      ['Sectors/Req > 最优', 'coalescing / 向量化 / SoA 布局（4.1）'],
      ['mio_throttle + shared k 路冲突', 'padding / swizzle（4.2）'],
      ['复用低、HBM 流量大', 'tiling（4.3）/ fusion 免中间 tensor（5.4）'],
      ['字节本身大', '低精度格式 / 量化（6.1/6.2）'],
    ] },
    { y: 268, h: 72, tone: 'ink', label: 'latency-bound（C < L_req，long_scoreboard + not_selected≈0）', rows: [
      ['occupancy 绑定', 'register/shared 预算、launch_bounds（3.4）'],
      ['ILP 不足', '展开独立链（1.4）/ 异步搬运（4.4）'],
    ] },
    { y: 352, h: 72, tone: 'ink', label: 'compute-bound（I ≥ R，u_F 高）', rows: [
      ['MMA 占比低', '指令 mix / tile 几何（5.2/6.1）'],
      ['精度未用满', 'FP8/FP4 路径（6.1）'],
    ] },
  ];
  lanes.forEach((ln) => {
    body.push(zone({ x: 36, y: ln.y, w: 648, h: ln.h, tone: ln.tone, label: ln.label }));
    ln.rows.forEach(([sig, tech], i) => {
      const ry = ln.y + 44 + i * 22;
      body.push(cap(52, ry, sig, { size: 8.5 }));
      body.push(`<path class="dia-edge" d="M 215 ${ry - 4} H 245"/>`);
      body.push(arrowHead(245, ry - 4, 1, 0));
      body.push(cap(252, ry, tech, { size: 8.5 }));
    });
  });
  // running example 脉冲（coalescing 路）
  body.push(pathPulse('M 215 172 H 245', { dur: 1.2, delay: 0.3 }));

  body.push(cap(36, 446, '命题 2 · 带宽饱和调优加速 = 流量比 V_before/V_after（式 5）：strideCopy 1024→128 MiB、320.33→40.04 μs、8×', { size: 9.5 }));
  body.push(cap(36, 464, 'u_B 前后均 100% —— 带宽利用率没变，变的是有用字节占比；「u_B 到顶」≠「优化到顶」（靠 Sectors/Req 与 excessive 定位）', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 480, body, { label: '调优决策树四泳道：duty 低系统级分 CPU-bound 走 CUDA Graph 融合与 sync 依赖走 stream 重组重叠；memory-bound 按 Sectors/Req 超最优走 coalescing 向量化 SoA、mio_throttle shared 冲突走 padding swizzle、复用低走 tiling fusion、字节大走低精度量化；latency-bound occupancy 绑定走 register 预算 launch_bounds、ILP 不足走展开独立链异步搬运；compute-bound MMA 占比低走指令 mix tile 几何、精度未用满走 FP8 FP4；命题二带宽饱和加速等于流量比 strideCopy 八倍 u_B 前后均百分之百' }).svg();
}

/* ============================================================
 * 11.2 §6（block 3）— Achieved roofline：A/B/D/E 四点 + D→B 轨迹
 * ============================================================ */
function achievedRoofline() {
  const body = [];
  const X = (I) => 90 + (Math.log10(I) + 1.3) / 4.3 * 570;
  const Y = (F) => 360 - (Math.log10(F) + 1) / 4.3 * 300;

  body.push(cap(36, 24, 'Achieved roofline（H100 BF16 主口径，log-log，示意非按比例）：点到 roof 距离 = 改进机会（recap 2.4 §7）', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标轴
  body.push(`<line class="dia-edge strong" x1="90" y1="360" x2="90" y2="60"/>`);
  body.push(`<line class="dia-edge strong" x1="90" y1="360" x2="660" y2="360"/>`);
  body.push(arrowHead(90, 60, 0, -1)); body.push(arrowHead(660, 360, 1, 0));
  body.push(cap(90, 52, 'F_ach（TFLOPS，log）', { size: 9 }));
  body.push(cap(660, 388, 'I_ach（FLOP/byte，log）', { size: 9, anchor: 'end' }));
  // 刻度
  [0.1, 1, 10, 100, 1000].forEach((v) => {
    body.push(`<line class="dia-edge" x1="${X(v).toFixed(1)}" y1="360" x2="${X(v).toFixed(1)}" y2="365"/>`);
    body.push(cap(X(v), 376, String(v), { mono: true, size: 8, anchor: 'middle' }));
    body.push(`<line class="dia-edge" x1="85" y1="${Y(v).toFixed(1)}" x2="90" y2="${Y(v).toFixed(1)}"/>`);
    body.push(cap(82, Y(v) + 3, String(v), { mono: true, size: 8, anchor: 'end' }));
  });

  // roof：斜线 I×BW + 水平 F_peak
  const ridge = 295;
  body.push(`<path class="dia-edge strong" d="M ${X(0.05).toFixed(1)} ${Y(0.05 * 3.35).toFixed(1)} L ${X(ridge).toFixed(1)} ${Y(989.5).toFixed(1)}"/>`);
  body.push(`<path class="dia-edge strong" d="M ${X(ridge).toFixed(1)} ${Y(989.5).toFixed(1)} H 660"/>`);
  body.push(`<path class="dia-edge dash" d="M ${X(ridge).toFixed(1)} 360 V ${Y(989.5).toFixed(1)}"/>`);
  body.push(cap(X(ridge), 376, 'R ≈ 295', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(656, 74, 'F_peak = 989.5', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(150, 246, '斜线 = I × BW', { size: 8.5 }));
  body.push(cap(150, 260, 'BW = 3.35 TB/s', { mono: true, size: 8 }));

  // 四点
  const pts = [
    { I: 400, F: 700, role: 'attn', r: 5 },
    { I: 2.5, F: 7.125, role: 'proj', r: 5 },
    { I: 2.5, F: 2.58, role: 'cache', r: 5 },
    { I: 0.083, F: 0.238, role: 'data', r: 4 },
  ];
  const fills = { attn: '#a83228', proj: '#5a7099', cache: '#cbb98d', data: '#8b867a' };
  pts.forEach((p) => body.push(`<circle cx="${X(p.I).toFixed(1)}" cy="${Y(p.F).toFixed(1)}" r="${p.r}" fill="${fills[p.role]}" stroke="#fff" stroke-width="1"/>`));
  // D→B 垂直上移（ILP 修复）
  const db = `M ${X(2.5).toFixed(1)} ${(Y(2.58) - 5).toFixed(1)} V ${(Y(7.125) + 5).toFixed(1)}`;
  body.push(`<path class="dia-edge strong" d="${db}"/>`);
  body.push(pathPulse(db, { dur: 1.2, delay: 0.3 }));
  body.push(arrowHead(X(2.5), Y(7.125) + 5, 0, -1));
  body.push(cap(305, 248, 'ILP 1→4', { size: 8, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(305, 262, 'C 64→256 sectors', { mono: true, size: 7.5, anchor: 'end' }));
  // 点标注
  body.push(cap(600, 112, 'A (400, 700)', { mono: true, size: 8 }));
  body.push(cap(600, 126, 'compute-bound', { size: 8 }));
  body.push(cap(600, 140, 'u_F 70.7%（算力侧）', { size: 7.5 }));
  body.push(cap(325, 226, 'B (2.5, 7.125) memory-bound', { mono: true, size: 8.5 }));
  body.push(cap(325, 240, 'u_B 85.1% · 沿斜线向 roof = u_B 收窄', { size: 8 }));
  body.push(cap(325, 262, 'D (2.5, 2.58) latency-bound', { mono: true, size: 8.5 }));
  body.push(cap(325, 276, 'loss 69.2% · ILP 修复拉回斜线', { size: 8 }));
  body.push(cap(128, 338, 'E (0.083, 0.238) elementwise', { mono: true, size: 8 }));

  body.push(cap(36, 412, '三条轨迹：D→B 垂直上移（强度不变、并发补齐，latency 修复）· B 沿斜线向 roof（u_B 收窄，η 因子）· tiling 提复用 → 换更高 L2/L1 ceiling', { size: 9 }));
  body.push(cap(36, 430, '点标口径：latency-bound 点在 2.4 §7 标 C，本章改标 D（避与并发符号 C 冲突）；坐标与损失读数不变', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(36, 448, '端到端对账（式 6）：MFU_step = duty × 忙碌加权 u_F = 0.667×0.551% = 0.368% —— decode MFU 天然低，优化目标是 u_B 与 duty 不是 MFU', { size: 9.5 }));
  body.push(cap(36, 466, '两类 kernel 都贴斜线（u_B 85.1%）：GEMV 类 30×100 μs（75.8%）+ elementwise 300×3.2 μs（24.2%）；剩余 14.9% 带宽损失由 memory chart 分解', { size: 9 }));

  return scene(720, 482, body, { label: 'Achieved roofline H100 BF16 log-log：水平 roof F_peak 九百八十九點五 TFLOPS 斜线 I 乘 BW 三点三五 TB/s ridge 约二百九十五；A 四百七百 compute-bound u_F 七十點七百分比，B 二點五七點一二五 memory-bound u_B 八十五點一百分比，D 二點五二點五八 latency-bound loss 六十九點二百分比，E 零點零八三零點二三八 elementwise；D 到 B 垂直上移 ILP 一到四 C 六十四到二百五十六 sectors，点标 D 非 C 避并发符号冲突，MFU_step 零點三六八百分比目标是 u_B 与 duty' }).svg();
}

/* ============================================================
 * 11.3 §2（block 0）— 故障检出-定位-处置链 → 系统层恢复 → 集群指针
 * ============================================================ */
function faultChain() {
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'GPU 故障：检出 → 定位 → 处置（机制，本章）→ 系统层恢复（策略）→ 集群编排（pointer）', { size: 12, color: 'var(--ink-soft)' }));

  const ev = node({ id: 'ev', x: 250, y: 52, w: 220, h: 32, role: 'attn', title: '故障事件（硬件侧）', titleSize: 11 });
  nodes.push(ev); body.push(ev.svg);

  body.push(zone({ x: 36, y: 100, w: 648, h: 64, label: '检出面（§3-7）' }));
  ['XID（kernel log）', 'ECC 计数', '链路计数器', '温度功率'].forEach((t, i) => {
    const n = node({ id: `det${i}`, x: 52 + i * 158, y: 130, w: 140, h: 28, role: 'data', title: t, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  });

  const loc = node({ id: 'loc', x: 250, y: 180, w: 220, h: 36, role: 'proj', title: '定位：哪个子系统？', sub: 'app bug 还是 HW？（XID 分类 §4 + 归因表）', titleSize: 10.5, subSize: 7.5 });
  nodes.push(loc); body.push(loc.svg);

  body.push(zone({ x: 36, y: 232, w: 648, h: 64, label: '处置（§3-7）' }));
  ['RESTART_APP', 'RESET_GPU（deferred）', 'Drain and Reset', 'RMA'].forEach((t, i) => {
    const n = node({ id: `act${i}`, x: 52 + i * 158, y: 262, w: 140, h: 28, role: 'op', title: t, mono: true, titleSize: 8 });
    nodes.push(n); body.push(n.svg);
  });

  body.push(zone({ x: 36, y: 312, w: 648, h: 72, label: '系统层恢复（recap，不重推）' }));
  const serv = node({ id: 'serv', x: 52, y: 342, w: 300, h: 34, role: 'norm', title: 'serving：rerouting 到健康实例', sub: 'ms 级缺口（inference 8.5 §6）', titleSize: 9.5, subSize: 7.5 });
  const train = node({ id: 'train', x: 368, y: 342, w: 300, h: 34, role: 'norm', title: 'training：隔离坏卡 → 换卡 → checkpoint 续训', sub: 'training 7.4 §5-6', titleSize: 8.5, subSize: 7.5 });
  nodes.push(serv, train); body.push(serv.svg, train.svg);

  const clu = node({ id: 'clu', x: 230, y: 400, w: 260, h: 32, role: 'plain', title: '集群级编排 → cluster 维度', sub: '节点 drain / 备机池 / 作业重调度', titleSize: 10, subSize: 7.5 });
  nodes.push(clu); body.push(clu.svg);

  // 纵向流
  [['M 360 84 V 100', 0], ['M 360 164 V 180', 0.3], ['M 360 216 V 232', 0.6], ['M 360 296 V 312', 0.9], ['M 360 384 V 400', 1.2]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.0, delay: dl }));
    body.push(arrowHead(360, parseFloat(d.split('V ')[1]), 0, 1));
  });

  body.push(cap(36, 452, '三层各消费下一层结论：硬件交付「哪张卡、哪个子系统、什么动作」，系统层决定「何时迁请求/重启/续训」，集群编排节点级动作', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(36, 470, '前提：故障已被检出并定位到具体卡与子系统 —— rerouting 要知道哪张卡坏，checkpoint 恢复要知道换哪张卡，SDC 判据要知道对哪个 rank 定点测试', { size: 9 }));

  return scene(720, 486, body, { label: 'GPU 故障检出定位处置链：故障事件硬件侧经检出面 XID kernel log ECC 计数链路计数器温度功率，定位哪个子系统 app bug 还是 HW，处置 RESTART_APP RESET_GPU deferred Drain and Reset RMA，接系统层恢复 serving rerouting 到健康实例 ms 级缺口与 training 隔离坏卡换卡 checkpoint 续训，最后集群级编排节点 drain 备机池作业重调度归 cluster 维度；三层各消费下一层结论' }).svg();
}

/* ============================================================
 * 11.3 §4（block 1）— XID 分类处置树 + 154 五档 + 代际漂移
 * ============================================================ */
function xidTree() {
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'XID 分类处置：先分类再动作 —— 31/43 多 app bug，48 HW only，74 Blackwell → 149（Release 620）', { size: 12, color: 'var(--ink-soft)' }));

  const root = node({ id: 'root', x: 36, y: 52, w: 160, h: 36, role: 'op', title: 'dmesg: NVRM: Xid', mono: true, titleSize: 9.5 });
  nodes.push(root); body.push(root.svg);
  body.push(`<line class="dia-edge strong" x1="116" y1="88" x2="116" y2="363"/>`);

  const rows = [
    { y: 96, code: '13/31/43', role: 'norm', main: 'app 级：重启 app + memcheck 查代码', sub: '31 重复同一 GPU → dcgmi diag（EUD）排 HW' },
    { y: 138, code: '48/94', role: 'cache', main: 'contained UE：重启受影响 app，方便时 reset', sub: 'SRAM DBE → 查 sram_threshold_exceeded flag' },
    { y: 180, code: '63', role: 'cache', main: 'remap 记录：安排 reset（deferred）使 pending 生效', sub: '63/64/177 给定 ECC 事件只见三者之一' },
    { y: 222, code: '64/95', role: 'attn', main: 'remap 失败 / uncontained：立即 reset', sub: '64 才需立即 RESET_GPU / CONTACT_SUPPORT' },
    { y: 264, code: '74 / 149', role: 'proj', main: '查 nvlink 计数器与对端（74 可远端同情错误）', sub: '74 Hopper 及更早 / 149 Blackwell NVLink5' },
    { y: 306, code: '79', role: 'attn', main: 'PCIe 不可达：查 kernel PCI log，CONTACT_SUPPORT', sub: '掉总线非重启 app 能解决，通常节点级干预' },
    { y: 348, code: '154', role: 'op', main: '读 recovery action 五档，定处置级别', sub: '先读 154 定级别，再回查触发它的具体 XID' },
  ];
  rows.forEach((r, i) => {
    body.push(`<line class="dia-edge" x1="116" y1="${r.y + 15}" x2="150" y2="${r.y + 15}"/>`);
    const n = node({ id: `x${i}`, x: 150, y: r.y, w: 120, h: 30, role: r.role, title: r.code, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    body.push(`<path class="dia-edge" d="M 270 ${r.y + 15} H 286"/>`);
    body.push(arrowHead(286, r.y + 15, 1, 0));
    body.push(cap(292, r.y + 13, r.main, { size: 8.5 }));
    body.push(cap(292, r.y + 27, r.sub, { size: 7.5 }));
    if (i === 0) body.push(pathPulse(`M 116 ${r.y + 15} H 150`, { dur: 1.2, delay: 0.2 }));
  });

  body.push(cap(36, 396, 'XID 154 汇总 recovery action 五档：None / Drain P2P / Drain and Reset / GPU Reset Required / Node Reboot Required', { size: 8.5 }));
  body.push(cap(36, 412, '74 可能是远端故障「同情错误」：远端 GPU 挂掉本端也报 74 —— 先用 nvidia-smi nvlink 查链路状态与对端再定界', { size: 8.5 }));

  body.push(zone({ x: 36, y: 428, w: 648, h: 60, tone: 'seal', label: '代际漂移硬注' }));
  body.push(cap(52, 462, '74 适用列 B-x00/GB-x00/Rubin = NO → Blackwell NVLink5 错误报 149 NETIR（按 144-150 decode 表解码再定处置）', { size: 9 }));
  body.push(cap(52, 478, 'SXid 仅 Hopper 及更早；NVL72 交换侧错误走 Fabric Manager 机制，勿把 SXid 表套到 NVL72', { size: 9 }));

  return scene(720, 500, body, { label: 'XID 分类处置树：dmesg NVRM Xid 分七支，十三三十一四十三 app 级重启 app memcheck 查代码三十一重复同一 GPU 走 dcgmi diag EUD，四十八九十四 contained UE 重启受影响 app 方便时 reset，六十三 remap 记录安排 deferred reset，六十四九十五 remap 失败 uncontained 立即 reset，七十四 Hopper 一百四十九 Blackwell 查 nvlink 计数器与对端，七十九 PCIe 不可达 CONTACT_SUPPORT，一百五十四读 recovery action 五档 None Drain P2P Drain and Reset GPU Reset Required Node Reboot Required；七十四 Blackwell 不适用报一百四十九 NETIR，SXid 仅 Hopper 及更早 NVL72 走 Fabric Manager' }).svg();
}

/* ============================================================
 * 11.3 §7（block 2）— 健康检查四层 + diag r1-r4 + RMA 边界
 * ============================================================ */
function healthLayers() {
  const body = [];

  body.push(cap(36, 24, '健康检查四层（被动 → 主动 → 带外）：DCGM 常驻筛查 → 异常触发 nsys/NCU 下钻 → 故障确诊', { size: 12, color: 'var(--ink-soft)' }));

  const bands = [
    { y: 52, h: 72, label: '持续监控（in-band）：DCGM 字段 / nvidia-smi', lines: [
      'ECC 计数（volatile/aggregate）· row remap · NVLink CRC/replay/recovery',
      '温度功率 · throttle 计数器（μs）· PCIe replay（202）',
    ] },
    { y: 136, h: 64, label: '事件驱动：XID（kernel log）→ 分类处置（§4）', lines: [
      'DCGM Field Value Watches / Active Health Checks / Policy Notification',
      '监控到动作的闭环不需要人盯日志',
    ] },
    { y: 212, h: 64, label: '主动诊断：dcgmi diag r1-r4（作业前门 / 定位 / 体检）', lines: [
      'r1/r2 作业前快速门 · r3 故障定位（内嵌 nvbandwidth/nccl_tests）· r4 深度体检',
      'dcgmi diag -r 3 / -r "pcie,diagnostic" / -p diagnostic.test_duration=300.0',
    ] },
    { y: 288, h: 64, label: '带外（out-of-band）：BMC / SMBPBI flag', lines: [
      'SRAM threshold 等 flag 经 in-band（NVML）+ out-of-band（SMBPBI）两路可读',
      'RMA 判定：Field Diagnostic（§3.4）—— DCGM diag 不服务 RMA',
    ] },
  ];
  bands.forEach((b, i) => {
    body.push(zone({ x: 36, y: b.y, w: 648, h: b.h, label: b.label }));
    b.lines.forEach((t, j) => body.push(cap(52, b.y + 40 + j * 18, t, { size: 8.5 })));
    if (i < 3) {
      const d = `M 360 ${b.y + b.h} V ${bands[i + 1].y}`;
      body.push(`<path class="dia-edge strong" d="${d}"/>`);
      body.push(pathPulse(d, { dur: 1.0, delay: i * 0.3 }));
      body.push(arrowHead(360, bands[i + 1].y, 0, 1));
    }
  });

  body.push(cap(36, 372, 'diag run level（高级含低级）：r1 <2.5 s（software）· r2 8GPU <10.5 min（+pcie/memory）· r3 <35 min（+nvbandwidth/nccl_tests）· r4 <2.25 h（+memtest/pulse）', { size: 8.5 }));
  body.push(cap(36, 390, '用法分层：r1/r2 作业前快速门 · r3 故障定位（XID 31 重复同一 GPU → RUN_DCGMEUD）· r4 周期性维护；r2 vs r4 约 13× 成本差', { size: 9 }));
  body.push(cap(36, 410, '边界（官方逐字）：DCGM diag 不提供全面硬件诊断、不主动修复、不替代 Field Diagnostic、不服务 RMA 流程 —— RMA 判定归 Field Diagnostic', { size: 9, color: 'var(--seal-deep)' }));

  return scene(720, 426, body, { label: '健康检查四层被动到主动到带外：持续监控 in-band DCGM 字段 nvidia-smi 盯 ECC 计数 row remap NVLink CRC replay recovery 温度功率 throttle 计数器，事件驱动 XID kernel log 分类处置加 DCGM watch policy 通知，主动诊断 dcgmi diag r1 到 r4 作业前门定位体检 r3 内嵌 nvbandwidth nccl_tests，带外 BMC SMBPBI flag RMA 判定 Field Diagnostic；diag r1 小于二點五秒 r2 八 GPU 小于十點五分钟 r3 小于三十五分钟 r4 小于二點二五小时约十三倍成本差，DCGM diag 不替代 Field Diagnostic 不服务 RMA' }).svg();
}

/* ============================================================
 * 11.3 §3（afterHeading）— ECC CE/UE + containment + remap 512/64 + RMA
 * ============================================================ */
function eccRemap() {
  const UID = 'g11j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'ECC 生产账：CE = 趋势信号，UE = 事件信号 —— containment 限影响半径，row remap 512 = 8× legacy 64', { size: 12, color: 'var(--ink-soft)' }));

  const src = node({ id: 'src', x: 250, y: 50, w: 220, h: 32, role: 'data', title: 'HBM bit flip（SECDED）', titleSize: 10.5 });
  const ce = node({ id: 'ce', x: 52, y: 100, w: 290, h: 56, role: 'norm', title: 'CE · 可纠正（SBE）', sub: '硬件修正 · 无 corruption · 计数累加作业无感', titleSize: 10.5, subSize: 8 });
  const ue = node({ id: 'ue', x: 378, y: 100, w: 290, h: 56, role: 'attn', title: 'UE · 不可纠正（DBE）', sub: '检得出、纠不了 · 数据不可信 → XID 48', titleSize: 10.5, subSize: 8 });
  nodes.push(src, ce, ue); body.push(src.svg, ce.svg, ue.svg);
  body.push(cap(52, 172, '趋势信号（预防性维护输入 §8）', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(378, 172, '事件信号（处置流程触发 §4）', { size: 8.5, color: 'var(--seal-deep)' }));
  [['M 300 82 C 250 82, 230 100, 197 100', 0], ['M 420 82 C 470 82, 490 100, 523 100', 0.3]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.2, delay: dl }));
  });
  body.push(arrowHead(197, 100, 0, 1)); body.push(arrowHead(523, 100, 0, 1));

  body.push(cap(36, 192, '计数口径：volatile 自 driver load（无 persistence 每次 app 重置）/ aggregate 终身持久（Inforom）；breakdown Device Memory/Register File/L1/L2/Texture', { size: 8.5 }));

  body.push(zone({ x: 36, y: 206, w: 648, h: 68, label: 'Error containment（Ampere+）—— UE 影响限遭遇错误的 app，多数 UE 无需 reset' }));
  const cont = node({ id: 'cont', x: 52, y: 236, w: 300, h: 28, role: 'norm', title: 'Contained（XID 94）· 限单 app → RESTART_APP', titleSize: 8.5 });
  const unc = node({ id: 'unc', x: 368, y: 236, w: 300, h: 28, role: 'attn', title: 'Uncontained（XID 95）· 影响多 app → RESET_GPU', titleSize: 8.5 });
  nodes.push(cont, unc); body.push(cont.svg, unc.svg);
  body.push(cap(36, 290, 'dynamic page offlining：driver 把 UE 所在 page 标不可用，当前与新 workload 不再分配 —— 可用容量可见地少一页（§7.3）', { size: 8.5 }));

  // row remap 容量账（左）+ RAS Repair（右）
  body.push(zone({ x: 36, y: 306, w: 324, h: 110, label: 'Row remapping（硬件换行）' }));
  body.push(cap(52, 342, '容量：row remap 512 vs legacy retirement 64 = 8×', { size: 8.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="52" y="350" width="${512 * 0.5}" height="12" rx="3"/>`);
  body.push(cap(314, 360, '512', { mono: true, size: 8.5, color: 'var(--seal-deep)', weight: '650' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="370" width="${64 * 0.5}" height="12" rx="3"/>`);
  body.push(cap(90, 380, '64（legacy）', { mono: true, size: 8.5 }));
  body.push(cap(52, 396, 'GPU reset 生效 · 终身持久 · 无软件空洞、容量不减', { size: 8 }));
  body.push(cap(52, 410, 'CE remap 可被 UE 替换（spare row 优先保 UE）', { size: 8 }));

  body.push(zone({ x: 384, y: 306, w: 300, h: 110, tone: 'seal', label: 'Blackwell RAS Repair（select 产品）' }));
  body.push(cap(400, 342, '同 bank 2 次 row remap 后，第 3 次 UE 触发', { size: 8.5 }));
  body.push(cap(400, 360, 'spare 粒度升级：DRAM channel 或 L2 slice', { size: 8.5 }));
  body.push(cap(400, 378, '成功标记 XID 160 · 需 reboot 生效', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(400, 394, '（比 row remap 的 GPU reset 更重一级）', { size: 8 }));
  body.push(cap(400, 410, 'repair 失败才继续 remap 直至 flag', { size: 8 }));

  body.push(zone({ x: 36, y: 432, w: 648, h: 64, tone: 'seal', label: 'RMA 触发 = 三条件析取（命题 1）—— 8 UE rows/bank 是充分路径非必要' }));
  body.push(cap(52, 468, '① bank 已 remap 8 UE rows 后再 attempt ② 对已 remap row 再 attempt（可总数 <8）③ UE remap 总数达 512 后再 attempt', { size: 8.5 }));
  body.push(cap(52, 484, 'SRAM threshold exceeded：parity >4 UCE/bank 或 SECDED >2/bank；Field Diagnostic 判 RMA（DCGM diag 不服务 RMA）', { size: 8.5 }));

  return scene(720, 512, body, { label: 'ECC 生产账：HBM bit flip SECDED 分 CE 可纠正 SBE 硬件修正无 corruption 计数累加作业无感为趋势信号，UE 不可纠正 DBE 检得出纠不了数据不可信报 XID 四十八为事件信号；volatile 自 driver load aggregate 终身 Inforom；Ampere containment UE 限遭遇 app contained 九十四 RESTART_APP uncontained 九十五 RESET_GPU 多数无需 reset，page offlining 容量少一页；row remap 五百一十二对 legacy 六十四八倍 GPU reset 生效终身持久无软件空洞 CE 可被 UE 替换，Blackwell RAS Repair 同 bank 两次 remap 后第三次 UE 触发 channel 或 L2 slice 替换 XID 一百六十 reboot 生效；RMA 三条件析取八 UE rows per bank 充分非必要 SRAM parity 超四 SECDED 超二' }).svg();
}

/* ============================================================
 * 11.3 §5.2（afterHeading）— Thermal throttle：两级降频 + 吞吐上界
 * ============================================================ */
function thermalThrottle() {
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Thermal throttle：HW slowdown 核心频率 ÷2 或更多 → compute-bound 吞吐上界 ÷2，memory-bound 不变（命题 2）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：温度三档
  const t1 = node({ id: 't1', x: 60, y: 64, w: 210, h: 36, role: 'norm', title: 'GPU Max Operating Temp', sub: 'SW Thermal Slowdown 触发', mono: true, titleSize: 9, subSize: 7.5 });
  const t2 = node({ id: 't2', x: 60, y: 128, w: 210, h: 36, role: 'cache', title: 'GPU Slowdown Temp', sub: 'HW Thermal Slowdown 触发', mono: true, titleSize: 9, subSize: 7.5 });
  const t3 = node({ id: 't3', x: 60, y: 192, w: 210, h: 36, role: 'attn', title: 'GPU Shutdown Temp', sub: '关机保护', mono: true, titleSize: 9, subSize: 7.5 });
  nodes.push(t1, t2, t3); body.push(t1.svg, t2.svg, t3.svg);
  body.push(`<path class="dia-edge" d="M 165 100 V 128"/>`); body.push(arrowHead(165, 128, 0, 1));
  body.push(`<path class="dia-edge" d="M 165 164 V 192"/>`); body.push(arrowHead(165, 192, 0, 1));
  body.push(cap(172, 118, 'SW：软件算法降频低于 requested', { size: 7.5 }));
  body.push(cap(172, 182, 'HW：核心频率砍半或更多', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(60, 244, '温度 ↑（nvidia-smi 温度字段三档）', { size: 8.5 }));

  // 右：频率/峰值账（label 左、bar 中、value 右）
  body.push(cap(340, 64, 'H100 FP32（1980 MHz 域，recap 1.1 §3）：', { size: 9.5 }));
  body.push(cap(470, 87, '正常 boost', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="480" y="76" width="${1980 * 0.07}" height="14" rx="3"/>`);
  body.push(cap(626, 87, '1980 MHz', { mono: true, size: 8.5 }));
  body.push(cap(470, 109, 'HW slowdown（边界值）', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="480" y="98" width="${990 * 0.07}" height="14" rx="3"/>`);
  body.push(cap(556, 109, '990 MHz（÷2）', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));

  body.push(cap(470, 139, 'FP32 峰值上界', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="480" y="128" width="${66.9 * 2}" height="14" rx="3"/>`);
  body.push(cap(620, 139, '66.9 TFLOPS', { mono: true, size: 8.5 }));
  body.push(cap(470, 161, '降频后', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="480" y="150" width="${33.45 * 2}" height="14" rx="3"/>`);
  body.push(cap(554, 161, '33.45 TFLOPS（÷2）', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));

  body.push(cap(340, 190, 'compute-bound kernel 原跑 80% 峰值（53.5 TFLOPS）→ 上界掉到 26.8 TFLOPS', { size: 9 }));
  body.push(cap(340, 210, 'memory-bound kernel 上界 = I·BW（HBM 3352 GB/s）—— 不随核心频率变', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(340, 228, '「降频伤算力不伤带宽」的定量形式（roofline 水平支 ∝ f，斜支由 HBM 几何定）', { size: 8.5 }));

  // 底：φ 占比账
  body.push(cap(36, 268, 'throttle 时间占比 φ（教学模型：throttle 期间恰减半、kernel compute-bound）：X_eff ≤ X(1 − φ/2)（式 1）', { size: 10 }));
  body.push(cap(170, 291, 'φ = 0.10', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="280" width="${95 * 3}" height="14" rx="3"/>`);
  body.push(cap(473, 291, '95% of 无 throttle', { mono: true, size: 9 }));
  body.push(cap(170, 313, 'φ = 0.50', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="180" y="302" width="${75 * 3}" height="14" rx="3"/>`);
  body.push(cap(413, 313, '75%', { mono: true, size: 9 }));
  body.push(cap(36, 340, '读数：偶发 throttle 代价小，持续占比才是冷却系统问题的信号（§8 预防性维护）—— 查冷却而非查 GPU', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(36, 358, '监控面：clocks_event_reasons_counters（μs）：sw_thermal / hw_thermal / sw_power_cap / hw_power_brake / sync_boost 各一路', { size: 9 }));
  body.push(cap(36, 376, '口径：假设降频后 kernel 仍处原 roofline 区；若降频把工作点推过 ridge 须用 min 式重判；「减半」取官方「factor of 2 or more」边界值', { size: 8.5 }));

  return scene(720, 392, body, { label: 'Thermal throttle 两级降频：温度三档 Max Operating Temp 触发 SW Thermal Slowdown 软件降频、Slowdown Temp 触发 HW Thermal Slowdown 核心频率砍半或更多、Shutdown Temp 关机保护；H100 FP32 一千九百八十 MHz 峰值六十六點九 TFLOPS，HW slowdown 边界值九百九十 MHz 峰值上界三十三點四五减半，原跑八十百分比峰值五十三點五的 compute-bound kernel 上界掉到二十六點八，memory-bound 上界 I 乘 BW 不随频率变；占比 φ 式 X_eff 不大于 X 乘一减 φ 除二，φ 零點一得九十五百分比 φ 零點五得七十五百分比，持续占比是冷却问题信号' }).svg();
}

/* ============================================================
 * 11.3 §6（afterHeading）— 链路三族计数器 + replay goodput + link 降级
 * ============================================================ */
function linkDegradation() {
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'NVLink/IB 链路错误：三族计数器严重度阶梯 —— replay goodput = BW(1−p)，link 降级按 link 计（18 断 1 → 94.4%）', { size: 12, color: 'var(--ink-soft)' }));

  // 三族阶梯
  const crc = node({ id: 'crc', x: 36, y: 56, w: 200, h: 64, role: 'norm', title: 'CRC flit error', sub: '400 系列 / total 409 · 物理层误码', mono: true, titleSize: 10, subSize: 7.5 });
  const rep = node({ id: 'rep', x: 260, y: 56, w: 200, h: 64, role: 'cache', title: 'replay error', sub: '420 系列 / total 429 · 链路层重传', mono: true, titleSize: 10, subSize: 7.5 });
  const rec = node({ id: 'rec', x: 484, y: 56, w: 200, h: 64, role: 'attn', title: 'recovery error', sub: '430 系列 / total 439 · 重训练/重建链', mono: true, titleSize: 10, subSize: 7.5 });
  nodes.push(crc, rep, rec); body.push(crc.svg, rep.svg, rec.svg);
  body.push(cap(46, 134, '轻：replay 消化', { size: 8.5 }));
  body.push(cap(270, 134, '中：误码率趋势信号', { size: 8.5 }));
  body.push(cap(494, 134, '重：链路短暂不可用 → XID 74/149', { size: 8.5, color: 'var(--seal-deep)' }));
  [['M 236 88 H 260', 0], ['M 460 88 H 484', 0.4]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge strong" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.2, delay: dl }));
    body.push(arrowHead(parseFloat(d.split('H ')[1]), 88, 1, 0));
  });
  body.push(cap(248, 80, '升级', { size: 7.5, anchor: 'middle' }));
  body.push(cap(472, 80, '升级', { size: 7.5, anchor: 'middle' }));
  body.push(cap(36, 154, 'flit = 链路层传输/检错最小单位；replay 与 IB link-level retry 同构（传输层不感知）；recovery = 链路级事件（重训练期间不可用）', { size: 8.5 }));

  // 左：goodput 账
  body.push(cap(36, 182, '命题 3 · 几何重传：BW_eff = BW(1−p)（式 2）', { size: 9.5 }));
  const gp = [['p = 10⁻⁶', '损失 0.0001%（可忽略）'], ['p = 0.01', '损失 1%'], ['p = 0.1', '损失 10%']];
  gp.forEach(([a, b], i) => {
    const y = 200 + i * 18;
    body.push(cap(52, y, a, { mono: true, size: 9 }));
    body.push(cap(150, y, b, { size: 9, color: i === 0 ? undefined : 'var(--seal-deep)' }));
  });
  body.push(cap(36, 262, '偶发 CRC 不必惊慌 —— 威胁是误码率趋势（物理层劣化前兆）', { size: 8.5 }));
  body.push(cap(36, 278, '与 recovery 事件（集合 max 语义下全局停顿）', { size: 8.5, color: 'var(--seal-deep)' }));

  // 右：link 降级
  body.push(cap(384, 182, 'H100 NVLink 18 link × 25 GB/s 单向（recap 7.1）', { size: 9.5 }));
  for (let j = 0; j < 18; j++) {
    const failed = j === 8;
    body.push(`<rect x="${384 + j * 15}" y="196" width="12" height="34" rx="2" fill="${failed ? '#a83228' : '#c9c4b8'}" stroke="${failed ? '#a83228' : '#a9a498'}" stroke-width="0.8"/>`);
    if (failed) body.push(cap(384 + j * 15 + 6, 244, '断 1', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  }
  body.push(cap(384, 262, 'k=1 → 17×25 = 425 GB/s = 满配 450 的 94.4%', { mono: true, size: 9, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(384, 278, 'k=2 → 400（88.9%）；双向同比例（850 对 900 GB/s）', { size: 8.5 }));

  // 底：max 语义 + 诊断
  body.push(cap(36, 306, '单 link 断只损失 ~5.6%，为何紧急？集合取 max（recap 9.3 §7）：经过降级 link 的 rank 变慢，整个集合被它拖住', { size: 9.5 }));
  body.push(cap(36, 324, '9.3 IB 算例同构：n=16、M=1 GB，一条 rail 50→25 GB/s，集合完成 37.5→75 ms 翻倍 —— NVLink 域同构（粒度 rail→link）', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(36, 342, '诊断：nvidia-smi nvlink -e 定位错误上涨 link → 查对端（74 同情错误 §4.2）→ 隔离/换缆（cable/connector/transceiver/seating/thermal）', { size: 9 }));
  body.push(cap(36, 360, '坑：偶发 CRC ≠ 链路故障（replay 消化，代价 ~p）；「无 XID」≠ 链路健康（计数器可在 XID 前持续上涨）—— 看趋势与 recovery，不看单点计数', { size: 9, color: 'var(--seal-deep)' }));

  return scene(720, 376, body, { label: 'NVLink IB 链路错误三族计数器严重度阶梯：CRC flit error 四百系列 total 四百零九物理层误码轻 replay 消化，replay error 四百二十系列 total 四百二十九链路层重传中误码率趋势信号，recovery error 四百三十系列 total 四百三十九重训练重建链重链路短暂不可用升级 XID 七十四一百四十九；命题三几何重传 goodput BW 乘一减 p，p 百万分之一损失零點零零零一百分比可忽略 p 零點零一损失一百分比 p 零點一损失十百分比；H100 十八 link 乘二十五 GB/s 单向断一 k 等于一十七乘二十五四百二十五满配四百五十的九十四點四百分比 k 二四百八百八點九百分比，集合取 max 降级 rank 拖住整体九點三算例三十七點五变七十五毫秒翻倍' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '11-system-integration/01-server-forms-mig',
    replacesBlock: 0,
    title: '形态层级：HGX 基板 → DGX 整机 → NVL72 机架（域 8→8→72，10.2→14.3→≈120 kW）',
    svg: serverForms,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/01-server-forms-mig',
    replacesBlock: 1,
    title: 'PCIe/NUMA 拓扑：每 socket 4 GPU+4 NIC，H2D = min(64,40) = 40（62.5%）',
    svg: pcieNuma,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/01-server-forms-mig',
    replacesBlock: 2,
    title: 'MIG 切片几何：memory 1/8×8 + SM 1/7×7，7 实例 SM 用满显存 7/8（L2 跨代例外）',
    svg: migSlices,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/02-profiling-tuning',
    replacesBlock: 0,
    title: 'Nsight timeline：busy 3960 / wall 5934 / gap 33.3%，graph 修复 1.50×（kernel 没动）',
    svg: nsysTimeline,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/02-profiling-tuning',
    replacesBlock: 1,
    title: 'Bottleneck 三区判定：I_ach 对 R、C 对 L_req(208) + 四因子 BW_e2e（1901→2846）',
    svg: bottleneckFlow,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/02-profiling-tuning',
    replacesBlock: 2,
    title: '调优决策树：从区到手段（六类）+ 命题 2 加速 = 流量比（strideCopy 8×）',
    svg: tuningTree,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/02-profiling-tuning',
    replacesBlock: 3,
    title: 'Achieved roofline：A/B/D/E 四点，D→B 垂直上移（ILP 1→4），点标 D 非 C',
    svg: achievedRoofline,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    replacesBlock: 0,
    title: '故障链：检出 → 定位 → 处置（机制）→ 系统层恢复（策略）→ 集群编排（pointer）',
    svg: faultChain,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    replacesBlock: 1,
    title: 'XID 分类处置：31/43 app 级、48 HW、63 deferred、74 Blackwell→149、154 五档',
    svg: xidTree,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    replacesBlock: 2,
    title: '健康检查四层：监控 → 事件 → diag r1-r4 → 带外（RMA 归 Field Diagnostic）',
    svg: healthLayers,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    afterHeading: /^## 3\. /,
    title: 'ECC 生产账：CE 趋势 / UE 事件（XID 48），containment 94/95，remap 512=8×64',
    svg: eccRemap,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    afterHeading: /^### 5\.2 /,
    title: 'Thermal throttle：HW slowdown ÷2 → compute-bound 上界 ÷2（66.9→33.45），φ 账',
    svg: thermalThrottle,
  },
  {
    dim: 'gpu',
    slug: '11-system-integration/03-faults-reliability',
    afterHeading: /^## 6\. /,
    title: '链路错误：CRC/replay/recovery 阶梯，goodput BW(1−p)，18 link 断 1 → 94.4%',
    svg: linkDegradation,
  },
];