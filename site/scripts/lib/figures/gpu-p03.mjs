// gpu P3（03-cuda-programming-model）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs / gpu-p02.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-cell --i ≤ 1.2 / sig-bar --i ≤ 0.6。
// census：3.1 block0 四级层级 / block1 divergence 时序 / block2 调度波次；
//         3.2 block0 backing 映射 / block1 = PTX atom 语法（伪代码，不画）；
//         3.3 block0 stream→work queue / block1 overlap 时序 / block2 graph 三阶段；
//         3.4 block0 occupancy 计算流 / block1 NCU Utilization 堆叠条。

import { scene, node, edge, zone, cap as kitCap, bars, PULSE_GRAD_ID } from '../figure-kit.mjs';

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

/* ============================================================
 * 3.1 §1（block 0）— 四级层级：逻辑总量 vs 物理驻留
 * ============================================================ */
function threadHierarchy() {
  const UID = 'g3a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '四级层级：65536 blocks × 256 threads = 16777216 threads —— 逻辑总量按波次流过 132 个 SM', { size: 12, color: 'var(--ink-soft)' }));

  // 层级链：grid → block → warp → thread
  const chain = [
    ['grid', 36, 160, 'grid', '65536 blocks · block 间无顺序'],
    ['blk', 226, 160, 'thread block', '≤1024 线程 · 驻留单 SM'],
    ['warp', 416, 120, 'warp', '32 线程 · 发射单位'],
    ['th', 566, 118, 'thread', '1 lane · 独立 PC（Volta+）'],
  ].map(([id, x, w, title, sub]) => {
    const n = node({ id, x, y: 48, w, h: 56, role: id === 'warp' ? 'attn' : 'proj', title, sub, titleSize: 13, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 3; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `h${i}` }));
    body.push(arrowHead(chain[i + 1].left[0], chain[i + 1].left[1], 1, 0));
  }
  body.push(cap(211, 44, '含 ×65536', { size: 8.5, anchor: 'middle' }));
  body.push(cap(401, 44, '含 ×8 warp', { size: 8.5, anchor: 'middle' }));
  body.push(cap(551, 44, '含 ×32', { size: 8.5, anchor: 'middle' }));
  body.push(cap(36, 124, 'block 是驻留与调度单位、warp 是执行单位 —— 逻辑线程总数与物理驻留线程数由此拆开', { size: 9.5 }));

  // 物理驻留分区
  body.push(zone({ x: 36, y: 136, w: 648, h: 156, label: '物理驻留（本章口径 R_th = 32、S_sh = 0 → 8 blocks/SM、O = 1.0）' }));
  body.push(`<g class="dia-node dia-node-plain"><rect x="52" y="166" width="280" height="104" rx="12"/></g>`);
  body.push(cap(66, 184, 'SM 0（132 个之一）', { size: 10.5, weight: '650', color: 'var(--ink)' }));
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 4; c++) {
      const k = r * 4 + c;
      const n = node({ id: `b${k}`, x: 64 + c * 66, y: 192 + r * 36, w: 60, h: 28, role: 'data', title: `block ${k}`, mono: true, titleSize: 9 });
      nodes.push(n); body.push(n.svg);
    }
  }
  const n1 = node({ id: 'n1', x: 368, y: 166, w: 150, h: 44, role: 'proj', title: '1056 blocks', sub: '132 SM × 8 blocks', mono: true, titleSize: 12.5, subSize: 8.5 });
  const n2 = node({ id: 'n2', x: 368, y: 226, w: 150, h: 44, role: 'proj', title: '8448 warps', sub: '1056 × 8 warps/block', mono: true, titleSize: 12.5, subSize: 8.5 });
  const n3 = node({ id: 'n3', x: 548, y: 196, w: 136, h: 44, role: 'attn', title: '270336', sub: 'threads 同时驻留', mono: true, titleSize: 14, subSize: 8.5 });
  nodes.push(n1, n2, n3); body.push(n1.svg, n2.svg, n3.svg);
  body.push(edge([332, 218], n1, { nodes, pulse: true, dur: 1.3, delay: 0.9, uid: UID, eid: 'r1' }));
  body.push(arrowHead(n1.left[0], n1.left[1], 1, 0));
  body.push(edge(n1, n2, { nodes, portFrom: n1.bottom, portTo: n2.top, pulse: true, dur: 1.1, delay: 1.3, uid: UID, eid: 'r2' }));
  body.push(arrowHead(n2.top[0], n2.top[1], 0, 1));
  body.push(edge(n2, n3, { nodes, pulse: true, dur: 1.1, delay: 1.7, uid: UID, eid: 'r3' }));
  body.push(arrowHead(n3.left[0], n3.left[1], 1, 0));
  body.push(cap(52, 284, '逻辑总量 524288 warps ÷ 驻留 8448 warps ≈ 62 个波次（§5）—— grid 尺寸是策略，波次调度是机制', { size: 9.5 }));

  body.push(cap(52, 316, 'w_b = ⌈threads_block/32⌉：256 线程 → 8 warps；257 线程 → 9 warps、末 warp 仅 1 活跃 lane，其余 31 lane 生来空转', { mono: true, size: 10 }));
  body.push(cap(52, 334, 'block 不可跨 SM（PG 逐字）：shared 是每 SM 片上 SRAM、block barrier 是 SM 内硬件屏障 —— 跨 SM 两者都失去物理载体', { size: 10 }));
  body.push(cap(52, 352, 'warp 不是编程对象：CUDA C++ 没有 warp 类型、threadIdx 不直接给 warp 编号 —— 硬件调度粒度向编程模型的「泄漏」', { size: 10 }));
  body.push(cap(52, 370, '索引双射（命题 1）：t_global = blockIdx.x·blockDim.x + threadIdx.x；100×256 + 37 = 25637 唯一分解（带余除法）', { mono: true, size: 10 }));

  return scene(720, 388, body, { label: '四级层级：grid 含 65536 个 block，block 含 8 个 warp，warp 含 32 线程；物理驻留 132 SM 乘 8 blocks 得 1056 blocks、8448 warps、270336 线程，逻辑总量约 62 个波次' }).svg();
}

/* ============================================================
 * 3.1 §3（block 1）— SIMT divergence：路径串行、mask 占槽
 * ============================================================ */
function simtDivergence() {
  const UID = 'g3b';
  const body = [];

  body.push(cap(36, 24, 'divergence：每条被走到的路径各按整 warp 发射一遍，被 mask 的 lane 仍占槽 —— 等代价 k 路径 η = 1/k', { size: 12, color: 'var(--ink-soft)' }));

  // 左：两条发射槽的 lane 活跃图（m = 16 示意）
  body.push(zone({ x: 36, y: 48, w: 330, h: 182, label: 'if (cond) 路径 A else 路径 B（m 真 / 32−m 假）' }));
  const laneRow = (y, active, i0) => {
    for (let j = 0; j < 32; j++) {
      const on = active(j);
      const cls = on ? 'dia-mx-hot' : 'dia-mx-off';
      const sig = on ? ` sig-cell" style="--i:${(i0 + j * 0.012).toFixed(2)}` : '';
      body.push(`<rect class="${cls}${sig}" x="${52 + j * 9.6}" y="${y}" width="8" height="16" rx="2"/>`);
    }
  };
  body.push(cap(52, 84, '发射槽 1：路径 A —— m lane 活跃，32−m 被 mask', { size: 9.5 }));
  laneRow(92, (j) => j < 16, 0.1);
  body.push(pathPulse('M 52 100 H 358', { dur: 1.6, delay: 0 }));
  body.push(cap(52, 136, '发射槽 2：路径 B —— 32−m lane 活跃，m 被 mask', { size: 9.5 }));
  laneRow(144, (j) => j >= 16, 0.7);
  body.push(pathPulse('M 52 152 H 358', { dur: 1.6, delay: 0.8 }));
  body.push(cap(52, 186, '两侧串行，共 2 个发射槽', { size: 9.5, weight: '650', color: 'var(--ink)' }));
  body.push(cap(52, 204, 'uniform（m = 32 或 0）：只执行被走到的一侧，1 个发射槽', { size: 9 }));
  body.push(cap(52, 220, '分歧只发生在 warp 内；warp 间无论同路异路都独立执行', { size: 9 }));

  // 右：D1–D5 案例账（槽数条 + η）
  body.push(cap(396, 64, '案例账（c_j 以 warp 指令计，式 3）', { size: 10.5, weight: '650', color: 'var(--ink)' }));
  const cases = [
    ['D1 tid 奇偶 16/16 · 1/1', 2, '1/2'],
    ['D2 tid%32==0 1/31 · 1/1', 2, '1/2'],
    ['D3 warp-uniform 32/0', 1, '1'],
    ['D4 1/31 · 代价 10/2', 12, '0.1875'],
    ['D4′ 31/1 · 代价 10/2', 12, '0.8125'],
    ['D5 嵌套二分 8×4 · 1×4', 4, '1/4'],
  ];
  cases.forEach(([lab, slots, eta], i) => {
    const y = 80 + i * 25;
    body.push(cap(396, y + 11, lab, { mono: true, size: 8.5 }));
    body.push(`<rect class="dia-bar dia-bar-${i === 2 ? 'soft' : 'seal'} sig-bar" style="--i:${(i * 0.1).toFixed(1)}" x="540" y="${y}" width="${slots * 8}" height="14" rx="3"/>`);
    body.push(cap(684, y + 11, `η=${eta}`, { mono: true, size: 8.5, anchor: 'end' }));
  });
  body.push(cap(396, 236, '条长 = 发射槽数；D1/D2 同价 —— 槽数只与路径是否非空有关', { size: 8.5 }));

  body.push(cap(52, 262, 'η = Σ m_j·c_j / (32·Σ c_j)（式 3）；等代价 c_j = c ⇒ η = 1/k，与 m_j 分布无关（命题 2）', { mono: true, size: 10.5, weight: '650' }));
  body.push(cap(52, 282, 'D2 只分歧 1 个线程照样 1/2：「分歧人数少所以代价小」是错误直觉；D3 把分支粒度提到 warp 级（tid/32），分歧代价清零', { size: 10 }));
  body.push(cap(52, 300, '不等代价时分布敏感度回来：D4 0.1875 对 D4′ 0.8125 差 4.33 倍 —— 重路径上的人越多越划算；嵌套深度 d 至多 2^d 路径', { size: 10 }));
  body.push(cap(52, 318, '数据分布定分歧频率：ReLU 形分支 p=0.5 → η≈0.5；p=0.99 → 期望槽 2−0.99³² ≈ 1.275、η≈0.784（聚合口径，非 E[η]）', { size: 10 }));
  body.push(cap(52, 336, '两笔独立罚单：divergence 折计算槽（本节）、coalescing 折带宽（4.1）—— 按 bound 类型各管一类损失，不叠加相乘', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 354, body, { label: 'SIMT divergence：路径 A 与路径 B 各占一个发射槽串行执行，被 mask 的 lane 仍占槽；D1 与 D2 等代价同给 η 二分之一，D3 uniform 一个槽，不等代价 D4 对 D4′ 差 4.33 倍' }).svg();
}

/* ============================================================
 * 3.1 §5（block 2）— block 调度与波次量化
 * ============================================================ */
function blockScheduling() {
  const UID = 'g3c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'block 调度器按空闲容量分发（GigaThread 为 Fermi 时代术语）—— 无顺序保证；65536 blocks = 62 满波 + 尾 64', { size: 12, color: 'var(--ink-soft)' }));

  const grid = node({ id: 'grid', x: 36, y: 52, w: 150, h: 52, role: 'data', title: 'grid', sub: '65536 blocks', mono: true, titleSize: 13, subSize: 9 });
  const sch = node({ id: 'sch', x: 240, y: 52, w: 220, h: 52, role: 'op', title: 'block 调度器（chip 级）', sub: '枚举后分发到有空闲容量的 SM', titleSize: 12, subSize: 8.5 });
  nodes.push(grid, sch); body.push(grid.svg, sch.svg);
  body.push(edge(grid, sch, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'g' }));
  body.push(arrowHead(sch.left[0], sch.left[1], 1, 0));
  body.push(cap(480, 70, '两级调度：chip 级分发 block，', { size: 9.5 }));
  body.push(cap(480, 86, 'SM 级 warp scheduler 发射 warp（1.2）', { size: 9.5 }));
  body.push(cap(480, 102, 'PG 代表版本均无 GigaThread 一词 —— 机制未变、名字退役', { size: 9 }));

  body.push(zone({ x: 36, y: 128, w: 648, h: 100, label: '驻留 1056 blocks = 8448 warps（132 SM × 8 blocks/SM）' }));
  const sms = [0, 1, 2, 3].map((k) => {
    const label = k === 2 ? '⋯' : `SM ${k === 3 ? 131 : k} [8 blocks]`;
    const n = node({ id: `sm${k}`, x: 52 + k * 160, y: 158, w: 144, h: 40, role: k === 2 ? 'plain' : 'cache', title: label, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(edge(sch, [350, 128], { nodes, portFrom: sch.bottom, pulse: true, dur: 1.0, delay: 0.5, uid: UID, eid: 's0', arrow: false }));
  body.push(arrowHead(350, 128, 0, 1));
  body.push(cap(52, 220, 'block 终止 → 腾出 slot/register/shared → 下一 block 补位；装不下至少一个 block 时 kernel 直接 launch 失败（1.2 C5）', { size: 9 }));

  // 波次量化条
  body.push(cap(52, 254, '波次量化账：65536 / 1056 = 62.06 → 62 个满波次 + 尾部 64 blocks', { size: 10.5, weight: '650', color: 'var(--ink)' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="36" y="272" width="600" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="639" y="272" width="6" height="18" rx="2"/>`);
  body.push(pathPulse('M 40 281 H 636', { dur: 2.6, delay: 0.6 }));
  body.push(cap(336, 268, '波次 1（1056）→ 波次 2 → ⋯ → 波次 62', { size: 9, anchor: 'middle' }));
  body.push(cap(642, 268, '尾 64', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 304, '65536 − 62×1056 = 64：尾波只有 64/1056 ≈ 6.1% 驻留容量在干活，其余 SM 空转等尾 —— 尾波量化损失', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(52, 322, '两条缓解：grid-stride loop 把 block 数压到驻留容量 1056（每线程 ⌈16777216/270336⌉ = 63 次迭代，波次 62→1）；', { size: 10 }));
  body.push(cap(52, 340, '或把 grid 尺寸凑成驻留容量的倍数（配置取舍归 3.4 / 11.2）。warp 视角同账：524288 ÷ 8448 = 62.06，两口径一致', { size: 10 }));
  body.push(cap(52, 358, '无顺序保证（PG 逐字）：不能假设 block 0 先跑、不能按 blockIdx 推断 SM 编号、不能同 kernel 内做 block 间生产者-消费者', { size: 10 }));

  return scene(720, 376, body, { label: 'block 调度：grid 经 chip 级调度器按空闲容量分发到 132 个 SM，驻留 1056 blocks；65536 blocks 走 62 个满波次加尾部 64 blocks，尾波仅 6.1% 容量在干活' }).svg();
}

/* ============================================================
 * 3.2 §1（block 0）— 六地址空间 → 四种物理 backing
 * ============================================================ */
function addressBacking() {
  const UID = 'g3d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '六个逻辑空间、四种物理 backing：local 逻辑私有、物理却在 HBM —— 逻辑作用域 ≠ 物理位置', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 250, h: 252, label: '逻辑空间（编程模型）' }));
  const spaces = [
    ['reg', 'register · 每线程'],
    ['loc', 'local · 每线程（逻辑私有）'],
    ['sh', 'shared · 每 block'],
    ['gl', 'global · 全 grid + host'],
    ['ct', 'constant · 每 grid 只读 64 KB'],
    ['tx', 'texture/surface · 只读路径'],
  ].map(([id, title], i) => {
    const n = node({ id, x: 52, y: 74 + i * 37, w: 218, h: 30, role: id === 'loc' ? 'attn' : 'data', title, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  body.push(zone({ x: 396, y: 48, w: 288, h: 252, label: '物理 backing（H100 SXM）' }));
  const rf = node({ id: 'rf', x: 412, y: 74, w: 256, h: 44, role: 'data', title: '片上 RF 256 KB/SM', sub: '管线内读出 · 无 load 延迟口径', titleSize: 12, subSize: 8.5 });
  const sram = node({ id: 'sram', x: 412, y: 130, w: 256, h: 44, role: 'cache', title: '片上 SRAM（与 L1 统一 256 KB）', sub: '29.0 cycles [B] · carveout 可配', titleSize: 11.5, subSize: 8.5 });
  const hbm = node({ id: 'hbm', x: 412, y: 186, w: 256, h: 56, role: 'proj', title: 'HBM 80 GB（经 L1/L2）', sub: '478.8 cycles 全 miss [B] · local 恒经 L2（CC 5.x+）', titleSize: 12, subSize: 8 });
  const cc = node({ id: 'cc', x: 412, y: 254, w: 256, h: 34, role: 'plain', title: '+ constant cache · texture cache（只读）', titleSize: 10 });
  nodes.push(rf, sram, hbm, cc); body.push(rf.svg, sram.svg, hbm.svg, cc.svg);

  const links = [[spaces[0], rf, 0], [spaces[2], sram, 0.4], [spaces[1], hbm, 0.8], [spaces[3], hbm, 1.2], [spaces[4], hbm, 1.6], [spaces[5], hbm, 2.0]];
  links.forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: i < 3, dur: 1.4, delay: dl, uid: UID, eid: `l${i}`, dash: i >= 3 }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(292, 116, 'spill', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(292, 128, '（2.3）', { size: 8, anchor: 'middle' }));

  body.push(cap(52, 322, 'HBM 对 shared 延迟比 478.8/29.0 = 16.5 倍 —— 把 block 协作数据从 global 搬进 shared，延迟账缩 16.5 倍（带宽账归 4.3）', { size: 10, weight: '650' }));
  body.push(cap(52, 340, 'constant 64 KB = 65536/4 = 16384 个 fp32：warp 全线程读同一地址一次广播服务；读不同地址按地址数拆分请求（PG 逐字）', { size: 10 }));
  body.push(cap(52, 358, 'local 每线程上限 512 KB；spill 三类触发：动态索引数组、过大结构、register 超预算 ——「局部变量数组开个动态下标」不是免费的', { size: 10 }));
  body.push(cap(52, 376, '只读旁路边界：ld.global.nc（__ldg / const __restrict__）走非一致缓存，内存模型不适用（PTX 逐字）—— 数据不得与写并发', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 394, body, { label: '六地址空间映射到四种物理 backing：register 到片上 RF，shared 到片上 SRAM，local、global、constant、texture 落 HBM 各带专属缓存；HBM 对 shared 延迟比 16.5 倍' }).svg();
}

/* ============================================================
 * 3.2 §2（afterHeading）— scope 五级 + fence 三级 + 排序/可见性
 * ============================================================ */
function fenceScope() {
  const UID = 'g3e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'scope 是一等公民：同步成本随线程距离增长 —— fence 只管 ordering，可见性须配对（命题 1）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：scope 五级漏斗
  body.push(zone({ x: 36, y: 48, w: 316, h: 168, label: 'thread_scope 五级（libcu++ enum · 嵌套）' }));
  const scopes = [
    ['system · 本 GPU + host + peer GPU', 284],
    ['device · 本 GPU 全部线程（270336）', 252],
    ['cluster · Hopper thread block cluster', 220],
    ['block · 同一 thread block（256 线程）', 188],
    ['thread · 仅自身', 156],
  ];
  scopes.forEach(([lab, w], i) => {
    const n = node({ id: `sc${i}`, x: 52, y: 72 + i * 26, w, h: 24, role: i === 1 ? 'attn' : 'proj', title: lab, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(52, 210, 'thread ⊂ block ⊂ cluster ⊂ device ⊂ system；选最小够用 scope', { size: 8.5 }));

  // 右：fence 三级映射
  body.push(zone({ x: 368, y: 48, w: 316, h: 168, label: 'fence 三级 ≡ PTX（sm_70+ membar ≡ fence.sc）' }));
  const fences = [
    ['__threadfence_block()', 'membar.cta ≡ fence.sc.cta'],
    ['__threadfence()', 'membar.gl ≡ fence.sc.gpu'],
    ['__threadfence_system()', 'membar.sys ≡ fence.sc.sys'],
  ];
  fences.forEach(([api, ptx], i) => {
    const y = 76 + i * 40;
    const a = node({ id: `fa${i}`, x: 384, y, w: 140, h: 30, role: 'op', title: api, mono: true, titleSize: 9 });
    const b = node({ id: `fb${i}`, x: 548, y, w: 120, h: 30, role: 'norm', title: ptx, mono: true, titleSize: 7.5 });
    nodes.push(a, b); body.push(a.svg, b.svg);
    body.push(cap(536, y + 19, '≡', { size: 10, anchor: 'middle' }));
  });
  body.push(cap(384, 202, '均 seq_cst · 与访存空间无关（PG 逐字）', { size: 8.5 }));

  // 下：消息传递因果链
  body.push(zone({ x: 36, y: 232, w: 648, h: 104, label: '正确消息传递：排序边 + 同步边合成因果链（命题 1c）' }));
  const steps = [
    ['p0', '写 data = 42', '普通 store'],
    ['p1', '__threadfence()', '排序边'],
    ['p2', 'flag.store(1, release)', '同步边（release 自带排序）'],
    ['p3', 'flag.load(acquire)', '同步边（消费者自旋）'],
    ['p4', '读 data → r == 42', '因果链完整 · 有保证'],
  ].map(([id, title, sub], i) => {
    const n = node({ id, x: 52 + i * 128, y: 262, w: 108, h: 44, role: i === 1 ? 'op' : i === 4 ? 'norm' : 'data', title, sub, mono: i < 4, titleSize: [8.5, 8.5, 7.5, 8, 8.5][i], subSize: 7 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 4; i++) {
    body.push(edge(steps[i], steps[i + 1], { nodes, pulse: true, dur: 1.1, delay: i * 0.4, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(steps[i + 1].left[0], steps[i + 1].left[1], 1, 0));
  }
  body.push(cap(52, 326, 'fence 单独不构成 happens-before（PG 逐字）：只有排序边没有同步边，读侧仍可能读旧值；acq_rel fetch_add 一条可替代 fence + relaxed 组合', { size: 9 }));

  body.push(cap(52, 356, 'pre-SM90 cluster scope 自动放宽为 device；device 覆盖 132 SM × 2048 = 270336 线程、block 只 256 —— 差 1056 倍', { size: 10 }));
  body.push(cap(52, 374, 'morally strong 三条件：program order 或 strong 操作 scope 互含 + 同一 proxy + 地址完全重叠；冲突而无因果序又非 morally strong = data race', { size: 10 }));
  body.push(cap(52, 392, '正确性的价格：atomic 轮询落 L2 一致点 263.0 cycles ≈ 143.7 ns；命中自己 L1 旧值只要 40.7 cycles —— 快 6.46 倍但错', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 410, body, { label: 'scope 五级嵌套与 fence 三级映射：threadfence_block、device、system 对应 membar cta、gl、sys；消息传递链写数据经 fence 与 release store，消费者 acquire load 后读数据有保证' }).svg();
}

/* ============================================================
 * 3.3 §2（block 0）— stream → work queue 假依赖
 * ============================================================ */
function streamQueue() {
  const UID = 'g3f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'stream 是编程对象、work queue 是物理载体（默认 8 条）—— 共享队列强加假依赖，把独立工作串行化', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 48, w: 200, h: 210, label: '编程对象（stream）' }));
  const streams = [0, 1, 2].map((i) => {
    const n = node({ id: `st${i}`, x: 52, y: 76 + i * 38, w: 168, h: 28, role: 'data', title: `stream ${i + 1}`, mono: true, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(136, 196, '⋮', { size: 11, anchor: 'middle' }));
  const st9 = node({ id: 'st9', x: 52, y: 208, w: 168, h: 28, role: 'attn', title: 'stream 9（超出 8 条）', mono: true, titleSize: 9.5 });
  nodes.push(st9); body.push(st9.svg);

  body.push(zone({ x: 280, y: 48, w: 180, h: 210, label: '物理载体（默认 8 条）' }));
  const q0 = node({ id: 'q0', x: 296, y: 96, w: 148, h: 40, role: 'op', title: 'work queue 0', mono: true, titleSize: 11 });
  const q7 = node({ id: 'q7', x: 296, y: 196, w: 148, h: 40, role: 'op', title: 'work queue 7', mono: true, titleSize: 11 });
  nodes.push(q0, q7); body.push(q0.svg, q7.svg);
  body.push(cap(370, 168, '⋮', { size: 11, anchor: 'middle' }));

  const sm = node({ id: 'sm', x: 508, y: 76, w: 176, h: 44, role: 'attn', title: 'SM（kernel）', sub: 'block 调度器按容量分发', titleSize: 12, subSize: 8.5 });
  const ce = node({ id: 'ce', x: 508, y: 176, w: 176, h: 44, role: 'cache', title: 'copy engine（memcpy）', sub: '独立 DMA · 不占 SM', titleSize: 11, subSize: 8.5 });
  nodes.push(sm, ce); body.push(sm.svg, ce.svg);

  streams.forEach((s, i) => {
    body.push(edge(s, q0, { nodes, pulse: i === 1, dur: 1.3, delay: i * 0.3, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(q0.left[0], q0.left[1], 1, 0));
  });
  body.push(edge(st9, q0, { nodes, portFrom: st9.right, portTo: [296, 126], dash: true, uid: UID, eid: 's9' }));
  body.push(cap(258, 236, '共 queue', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge(q0, sm, { nodes, pulse: true, dur: 1.3, delay: 1.0, uid: UID, eid: 'k' }));
  body.push(arrowHead(sm.left[0], sm.left[1], 1, 0));
  body.push(edge(q0, ce, { nodes, portFrom: q0.right, portTo: ce.left, pulse: true, dur: 1.3, delay: 1.4, uid: UID, eid: 'c' }));
  body.push(arrowHead(ce.left[0], ce.left[1], 1, 0));
  body.push(cap(508, 148, '共享队列 → 假依赖 → 串行', { size: 9, color: 'var(--seal-deep)' }));

  body.push(cap(52, 282, 'CUDA_DEVICE_MAX_CONNECTIONS：1–32、默认 8；CC 8.0+ 另有 CUDA_DEVICE_MAX_COPY_CONNECTIONS', { mono: true, size: 9 }));
  body.push(cap(52, 298, '按 async copy engine 设拷贝队列；stream 数超过 work queue 数时多条 stream 共享同一队列', { size: 9.5 }));
  body.push(cap(52, 316, 'PG 逐字：独立任务映射到同一 workqueue 可能引入 false dependence → 串行 ——「stream 数」≠「并发度」', { size: 10 }));
  body.push(cap(52, 334, 'NULL stream 隐式同步：两非默认 stream 间夹默认 stream 操作即串行 —— 出路 NonBlocking / per-thread default', { size: 10 }));
  body.push(cap(52, 352, 'stream 内有序、stream 间无序：同 stream「H2D → kernel」无需显式同步；跨 stream 顺序要 event（§3）', { size: 10 }));

  return scene(720, 370, body, { label: 'stream 映射到有限 work queue：三条 stream 共享 queue 0，超出的 stream 9 也与之共队列引入假依赖；queue 通往 SM 与 copy engine，默认八条队列' }).svg();
}

/* ============================================================
 * 3.3 §4（block 1）— copy-compute 三资源流水时序
 * ============================================================ */
function overlapTimeline() {
  const UID = 'g3g';
  const body = [];

  body.push(cap(36, 24, '三资源流水：H2D 引擎 / SM / D2H 引擎各自独立 —— τ_pipe = Σ + (n_chunk−1)·max，瓶颈 H2D 封顶稳态', { size: 12, color: 'var(--ink-soft)' }));

  // 串行面板（按时间比例：500 px / 3205.79 μs）
  body.push(zone({ x: 36, y: 48, w: 648, h: 112, label: '串行：τ_seq = 8 × (262.144 + 7.508 + 131.072) = 3205.79 μs' }));
  const S1 = 500 / 3205.79;
  body.push(cap(140, 92, 'copy 引擎', { size: 9, anchor: 'end' }));
  body.push(cap(140, 118, 'SM', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="150" y="80" width="${(2097.152 * S1).toFixed(1)}" height="16" rx="3"/>`);
  body.push(cap(150 + 2097.152 * S1 / 2, 92, 'H2D（全部）2097.152 μs', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink)' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="${(150 + 2157.214 * S1).toFixed(1)}" y="80" width="${(1048.576 * S1).toFixed(1)}" height="16" rx="3"/>`);
  body.push(cap(150 + 2157.214 * S1 + 1048.576 * S1 / 2, 92, 'D2H（全部）1048.576 μs', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink)' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="${(150 + 2097.152 * S1).toFixed(1)}" y="106" width="${(60.062 * S1).toFixed(1)}" height="16" rx="2"/>`);
  body.push(`<line class="dia-edge" x1="${(150 + 2157.214 * S1).toFixed(1)}" y1="114" x2="${(150 + 2157.214 * S1 + 8).toFixed(1)}" y2="114"/>`);
  body.push(cap(150 + 2157.214 * S1 + 12, 118, 'kernel 60.062 μs · 占端到端 1.9%', { size: 8.5 }));
  body.push(`<line class="dia-edge strong" x1="150" y1="138" x2="650" y2="138"/>`);
  [[0, '0'], [1000, '1000'], [2000, '2000'], [3000, '3000 μs']].forEach(([t, lab]) => {
    const x = 150 + t * S1;
    body.push(`<line class="dia-edge" x1="${x.toFixed(1)}" y1="133" x2="${x.toFixed(1)}" y2="143"/>`);
    body.push(cap(x, 154, lab, { mono: true, size: 8, anchor: 'middle' }));
  });

  // 流水面板（按时间比例：500 px / 2235.73 μs）
  body.push(zone({ x: 36, y: 176, w: 648, h: 136, label: '三资源流水（双 copy engine，8 chunks 错开）：τ_pipe = 400.724 + 7×262.144 = 2235.73 μs' }));
  const S2 = 500 / 2235.73;
  const cw = 262.144 * S2;
  body.push(cap(140, 218, 'H2D 引擎', { size: 9, anchor: 'end' }));
  body.push(cap(140, 250, 'SM', { size: 9, anchor: 'end' }));
  body.push(cap(140, 278, 'D2H 引擎', { size: 9, anchor: 'end' }));
  for (let i = 0; i < 8; i++) {
    body.push(`<rect class="dia-bar dia-bar-warm" x="${(150 + i * cw).toFixed(1)}" y="206" width="${(cw - 0.8).toFixed(1)}" height="16" rx="3"/>`);
    body.push(cap(150 + i * cw + cw / 2, 218, `c${i + 1}`, { mono: true, size: 8, anchor: 'middle', color: 'var(--ink)' }));
    body.push(`<rect class="dia-bar dia-bar-seal" x="${(150 + (i + 1) * cw).toFixed(1)}" y="238" width="2.6" height="16" rx="1"/>`);
    const dx = 150 + cw + 7.508 * S2 + i * cw;
    body.push(`<rect class="dia-bar dia-bar-soft" x="${dx.toFixed(1)}" y="266" width="${(131.072 * S2 - 0.8).toFixed(1)}" height="16" rx="3"/>`);
    body.push(cap(dx + 131.072 * S2 / 2, 278, `d${i + 1}`, { mono: true, size: 7.5, anchor: 'middle', color: 'var(--ink)' }));
  }
  body.push(cap(212, 234, 'k1…k8 各 7.508 μs —— 计算藏进 H2D 背后（稳态由最慢阶段 H2D 封顶）', { size: 8.5 }));
  body.push(pathPulse('M 152 214 H 618', { dur: 2.4, delay: 0 }));
  body.push(pathPulse('M 210 246 H 618', { dur: 2.4, delay: 0.6 }));
  body.push(pathPulse('M 212 274 H 648', { dur: 2.4, delay: 1.2 }));
  body.push(`<line class="dia-edge strong" x1="150" y1="296" x2="650" y2="296"/>`);
  [[0, '0'], [500, '500'], [1000, '1000'], [1500, '1500'], [2000, '2000 μs']].forEach(([t, lab]) => {
    const x = 150 + t * S2;
    body.push(`<line class="dia-edge" x1="${x.toFixed(1)}" y1="291" x2="${x.toFixed(1)}" y2="301"/>`);
    body.push(cap(x, 310, lab, { mono: true, size: 8, anchor: 'middle' }));
  });

  body.push(cap(52, 336, '每 chunk（8 份、各 2097152 元素）：m_in 16 MiB → 262.144 μs；m_out 8 MiB → 131.072 μs；m_comp 24 MiB → 7.508 μs（式 1）', { mono: true, size: 9 }));
  body.push(cap(52, 354, '节省 Δτ = 7×(400.724−262.144) = 970.06 μs（30.3%）；加速 1.434，渐近上界 Σ/max = 1.529 —— 不是 3（H2D 占 Σ 65%）', { size: 9.5, weight: '650' }));
  body.push(cap(52, 372, '单 copy engine：拷贝忙碌 3145.73 μs 是下界，节省至多 60.06 μs（1.9%）= 全部 kernel 时间；双引擎 970 μs，差 16 倍', { size: 9.5 }));
  body.push(cap(52, 390, 'overlap 三条件：asyncEngineCount > 0 + page-locked + 不同非默认 stream；pageable 静默退化为同步', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 408, body, { label: 'copy-compute 重叠时序：串行 3.206 ms，H2D 全部、kernel、D2H 全部顺序执行；三资源流水八个 chunk 错开，H2D 引擎连续、SM 计算细条藏在其后、D2H 引擎排空，总时间 2.236 ms，加速 1.434' }).svg();
}

/* ============================================================
 * 3.3 §5（block 2）— CUDA Graph 三阶段 + launch 摊销
 * ============================================================ */
function graphAmortize() {
  const UID = 'g3h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'CUDA Graph 把定义与执行分离：实例化一次预建工作描述符，重放一次提交整张 DAG —— 摊掉固定提交开销', { size: 12, color: 'var(--ink-soft)' }));

  const stages = [
    ['def', 36, 200, 'definition（定义）', 'stream capture / graph API · 只录不跑'],
    ['inst', 268, 200, 'instantiation（实例化）', 'cudaGraphInstantiate · 预建全部工作描述符'],
    ['exec', 500, 184, 'execution（执行）', 'cudaGraphLaunch × n_rep · 整张 DAG 一次'],
  ].map(([id, x, w, title, sub]) => {
    const n = node({ id, x, y: 52, w, h: 60, role: id === 'inst' ? 'attn' : 'proj', title, sub, titleSize: 12, subSize: id === 'exec' ? 7.5 : 8 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(edge(stages[0], stages[1], { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'a' }));
  body.push(arrowHead(stages[1].left[0], stages[1].left[1], 1, 0));
  body.push(edge(stages[1], stages[2], { nodes, pulse: true, dur: 1.3, delay: 0.5, uid: UID, eid: 'b' }));
  body.push(arrowHead(stages[2].left[0], stages[2].left[1], 1, 0));
  body.push(cap(136, 132, 'capture 期间 launch 只录不跑（PG 逐字）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(368, 132, '一次性 t_inst ≈ 400 μs', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(592, 132, '每次 t_graph ≈ 3.4 μs', { mono: true, size: 9.5, anchor: 'middle' }));

  body.push(cap(52, 170, '每 kernel 暴露开销三档（blog V100 / CUDA 10.1 实测 · 时代注：绝对值随硬件世代漂移，~μs 量级口径）', { size: 10 }));
  [['sync 9.6 − 2.9', 6.7, 'seal'], ['async 3.8 − 2.9', 0.9, 'warm'], ['graph 3.4 − 2.9', 0.5, 'ink']].forEach(([lab, v, tone], i) => {
    const y = 184 + i * 28;
    body.push(cap(240, y + 12, lab, { mono: true, size: 9, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone} sig-bar" style="--i:${(i * 0.2).toFixed(1)}" x="250" y="${y}" width="${v * 40}" height="16" rx="3"/>`);
    body.push(cap(250 + v * 40 + 8, y + 12, `${v} μs`, { mono: true, size: 9.5 }));
  });
  body.push(cap(250, 278, 'kernel 执行本身 2.9 μs（数组 500000 元素、512 threads/block）；暴露开销 = 每 kernel 总时间 − 执行时间', { size: 9 }));

  body.push(cap(52, 304, '盈亏平衡（命题 2）：n_rep* = t_inst/(n_kernel·t_launch − t_graph) = 400/0.4 = 1000 次；20000 次 → 0.02 μs/kernel', { mono: true, size: 9.5, weight: '650' }));
  body.push(cap(52, 326, '引擎侧（inference 3.4）：330 kernel/步 × 5 μs = 1.65 ms，graph 省 ≈1.645 ms/步 —— 平衡 ≪ 1 次重放，decode 第一步就赚', { size: 9.5 }));
  body.push(cap(52, 344, '首次 launch 慢约 33%：映射在 launch 时才做（PG），cudaGraphUpload 挪到预热；拓扑变更须重实例化，参数变更走 update', { size: 9.5 }));
  body.push(cap(52, 362, '静态三要求：指针/shape/grid/kernel 序列固定；内容动态合法（replay 读当前值）；cross-stream 须 rejoin origin', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 380, body, { label: 'CUDA Graph 三阶段：定义只录不跑，实例化一次性四百微秒预建工作描述符，执行每次三点四微秒；暴露开销 sync 6.7、async 0.9、graph 0.5 微秒，盈亏平衡一千次重放' }).svg();
}

/* ============================================================
 * 3.4 §2（block 0）— occupancy 精确计算流（粒度修正 → 四约束 min）
 * ============================================================ */
function occupancyFlow() {
  const UID = 'g3i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '从物理上界到精确计算：register 按 warp 取整、shared 按 128 B 取整 + 1 KB 预留 —— b_act ≤ B_max，精确值以 API 为准', { size: 12, color: 'var(--ink-soft)' }));

  const use = node({ id: 'use', x: 36, y: 52, w: 280, h: 56, role: 'data', title: 'kernel 资源用量', sub: '--resource-usage / -Xptxas -v → R_th, S_sh, threads_block', titleSize: 12.5, subSize: 7.5 });
  const inv = node({ id: 'inv', x: 404, y: 52, w: 280, h: 56, role: 'cache', title: '硬件库存（1.2 §1）', sub: 'device properties / arch_traits → R_file, S_total, W_slots', titleSize: 12.5, subSize: 7.5 });
  const gran = node({ id: 'gran', x: 180, y: 140, w: 360, h: 48, role: 'op', title: '粒度修正（只增不减）', sub: 'R_blk warp 粒度向上取整 · S_alloc = ⌈S_sh/128⌉×128 + 1024 B', mono: true, titleSize: 12, subSize: 8.5 });
  const mn = node({ id: 'mn', x: 180, y: 216, w: 360, h: 56, role: 'attn', title: '式 (1) 四约束取 min → b_act', sub: 'min(32, ⌊64/w_b⌋, ⌊65536/R_blk⌋, ⌊233472/S_alloc⌋)', mono: true, titleSize: 12.5, subSize: 9 });
  const wa = node({ id: 'wa', x: 180, y: 300, w: 170, h: 44, role: 'proj', title: '式 (2) W_active', sub: '= b_act · w_b', mono: true, titleSize: 11.5, subSize: 9 });
  const oc = node({ id: 'oc', x: 370, y: 300, w: 170, h: 44, role: 'attn', title: '式 (3) O', sub: '= W_active / 64', mono: true, titleSize: 12.5, subSize: 9 });
  const api = node({ id: 'api', x: 560, y: 216, w: 124, h: 56, role: 'norm', title: 'occupancy API', sub: '权威精确值', titleSize: 11, subSize: 8.5 });
  nodes.push(use, inv, gran, mn, wa, oc, api);
  body.push(use.svg, inv.svg, gran.svg, mn.svg, wa.svg, oc.svg, api.svg);

  body.push(edge(use, gran, { nodes, portFrom: use.bottom, portTo: gran.top, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'u' }));
  body.push(edge(inv, gran, { nodes, portFrom: inv.bottom, portTo: gran.top, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'i' }));
  body.push(arrowHead(gran.top[0], gran.top[1], 0, 1));
  body.push(edge(gran, mn, { nodes, portFrom: gran.bottom, portTo: mn.top, pulse: true, dur: 1.1, delay: 0.7, uid: UID, eid: 'g' }));
  body.push(arrowHead(mn.top[0], mn.top[1], 0, 1));
  body.push(edge(mn, wa, { nodes, portFrom: mn.bottom, portTo: wa.top, pulse: true, dur: 1.1, delay: 1.1, uid: UID, eid: 'w' }));
  body.push(arrowHead(wa.top[0], wa.top[1], 0, 1));
  body.push(edge(wa, oc, { nodes, pulse: true, dur: 1.1, delay: 1.5, uid: UID, eid: 'o' }));
  body.push(arrowHead(oc.left[0], oc.left[1], 1, 0));
  body.push(edge(api, mn, { nodes, portFrom: api.left, portTo: mn.right, dash: true, uid: UID, eid: 'a' }));
  body.push(arrowHead(mn.right[0], mn.right[1], -1, 0));
  body.push(cap(622, 290, 'cudaOccupancyMaxActive', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(622, 302, 'BlocksPerMultiprocessor', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(622, 316, '内含各架构分配粒度规则', { size: 8, anchor: 'middle' }));
  body.push(cap(52, 306, '配套：MaxPotential', { mono: true, size: 7.5 }));
  body.push(cap(52, 318, 'BlockSize 反解配置；', { size: 7.5 }));
  body.push(cap(52, 330, 'MaxActiveClusters（4.4）', { mono: true, size: 7.5 }));

  body.push(cap(52, 356, '最小实例（3.1 基线）：256 线程、R_th = 32、S_sh = 0 → min(32, 8, ⌊65536/8192⌋ = 8, ∞) = 8 blocks，W_active = 64，O = 1.0', { mono: true, size: 9 }));
  body.push(cap(52, 374, '官方算例（CC 10.0 库存 = CC 9.0）：768 线程 → 2 blocks 75%（warp 绑定）；32 线程 → 32 blocks 50%（slot 绑定）；100 KB shared → 2 blocks', { size: 8.5 }));
  body.push(cap(52, 392, 'CC 9.0 register 分配单位未逐字披露：手算给理想上界、精确值以 API 为准；CC 7.0 例 37 regs × 320 线程手算 5 对官方 4 —— 粒度架构特定', { size: 9 }));
  body.push(cap(52, 410, 'b_act ≤ B_max（1.2 命题 3 方向）：取整只增占用 —— 式 (1) 从理想上界往下修，不会往上修', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 428, body, { label: 'occupancy 精确计算流：kernel 资源用量与硬件库存经粒度修正，四约束取 min 得 b_act，乘 warps 每 block 得 W_active，除 64 得 occupancy；occupancy API 内含各架构粒度给权威精确值' }).svg();
}

/* ============================================================
 * 3.4 §6（block 1）— NCU Utilization 三态堆叠条（K2 register 绑定）
 * ============================================================ */
function ncuUtilization() {
  const UID = 'g3j';
  const body = [];

  body.push(cap(36, 24, 'NCU Utilization 三态 = 四约束取 min 的可视化：哪种资源的 unallocated 段最大，哪种就是绑定约束', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(52, 56, 'K2 kernel（256 threads/block、R_th = 96、S_sh = 32 KB）→ b_act = 2、W_active = 16、O = 0.25，register 绑定', { size: 10 }));

  // 四条资源堆叠条（总宽 440 px = 资源库存）
  const rows = [
    ['register', 49152 / 65536, 'unused', '49152/65536'],
    ['shared', 67584 / 233472, 'limiter', '67584/233472'],
    ['warp slot', 16 / 64, 'limiter', '16/64'],
    ['block slot', 2 / 32, 'limiter', '2/32'],
  ];
  rows.forEach(([lab, frac, rest, val], i) => {
    const y = 76 + i * 40;
    const aw = frac * 440;
    body.push(cap(170, y + 15, lab, { mono: true, size: 10, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-ink sig-bar" style="--i:${(i * 0.2).toFixed(1)}" x="180" y="${y}" width="${aw.toFixed(1)}" height="22" rx="3"/>`);
    body.push(`<rect class="dia-bar dia-bar-${rest === 'unused' ? 'warm' : 'soft'}" x="${(180 + aw + 2).toFixed(1)}" y="${y}" width="${(440 - aw - 2).toFixed(1)}" height="22" rx="3"/>`);
    body.push(cap(684, y + 15, val, { mono: true, size: 8.5, anchor: 'end' }));
  });
  body.push(`<rect class="dia-bar dia-bar-ink" x="180" y="252" width="14" height="10" rx="2"/>`);
  body.push(cap(200, 261, 'allocated（已分给驻留 block）', { size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="360" y="252" width="14" height="10" rx="2"/>`);
  body.push(cap(380, 261, 'unallocated due to other limiter（被绑定约束卡住）', { size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-warm" x="180" y="272" width="14" height="10" rx="2"/>`);
  body.push(cap(200, 281, 'unused（不足一个 block 最小需求的零头，粒度取整的可视化）', { size: 9 }));

  body.push(cap(52, 306, '读法：register 条满 + 零头、shared 条大段 unallocated → register 绑定；四约束 min(32, 8, ⌊65536/24576⌋ = 2, ⌊233472/33792⌋ = 6) = 2 blocks', { size: 9.5 }));
  body.push(cap(52, 324, 'Graphs 视图锯齿 = 粒度悬崖：每个台阶跳变对应 register/shared 悬崖；block 尺寸选平台区、避开悬崖右侧；理论 ≠ achieved（尾波/不均/瞬态）', { size: 9.5 }));
  body.push(cap(52, 342, '旧 Excel spreadsheet 已废（NCU 逐字 replaces）—— 用 NCU Calculator 或运行时 API，两者同源；可存 .ncu-occ、Python 脚本化扫描', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 360, body, { label: 'NCU Utilization 三态堆叠条：K2 kernel 的 register 条已满加零头，shared、warp slot、block slot 大段 unallocated 因 register 绑定分不出去，四约束取 min 得 2 blocks、occupancy 0.25' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/01-thread-hierarchy-simt',
    replacesBlock: 0,
    title: '四级层级：16777216 threads 的逻辑总量 → 270336 的物理驻留（62 波）',
    svg: threadHierarchy,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/01-thread-hierarchy-simt',
    replacesBlock: 1,
    title: 'SIMT divergence：路径串行、mask 占槽 —— 等代价 k 路径 η = 1/k',
    svg: simtDivergence,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/01-thread-hierarchy-simt',
    replacesBlock: 2,
    title: 'block 调度与波次量化：62 满波 + 尾 64 blocks（6.1% 容量）',
    svg: blockScheduling,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/02-memory-model-address-spaces',
    replacesBlock: 0,
    title: '六地址空间 → 四种物理 backing：local 逻辑私有、物理在 HBM',
    svg: addressBacking,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/02-memory-model-address-spaces',
    afterHeading: /^## 2\. /,
    title: 'scope 五级 + fence 三级：ordering 边 ≠ 可见性，须配对（命题 1）',
    svg: fenceScope,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/03-stream-event-graph',
    replacesBlock: 0,
    title: 'stream → work queue（默认 8 条）：共享队列的假依赖',
    svg: streamQueue,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/03-stream-event-graph',
    replacesBlock: 1,
    title: 'copy-compute 三资源流水：3.206 ms → 2.236 ms（加速 1.434，上界 1.529）',
    svg: overlapTimeline,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/03-stream-event-graph',
    replacesBlock: 2,
    title: 'CUDA Graph 三阶段：t_inst 400 μs 摊到 n_rep —— 盈亏平衡 1000 次',
    svg: graphAmortize,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/04-occupancy-resource-constraints',
    replacesBlock: 0,
    title: 'occupancy 精确计算：粒度修正 → 四约束 min → API 权威值',
    svg: occupancyFlow,
  },
  {
    dim: 'gpu',
    slug: '03-cuda-programming-model/04-occupancy-resource-constraints',
    replacesBlock: 1,
    title: 'NCU Utilization 三态：unallocated 段最大者 = 绑定约束（K2）',
    svg: ncuUtilization,
  },
];