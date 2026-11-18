// training Part 4（04-data-parallelism）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；脉冲渐变用页面级 PULSE_GRAD_ID（勿自建 per-scene defs）。
// 数字与机制措辞全部来自章节原文（4.1 式 (2)(3)、4.2 式 (1)-(3)、4.3 式 (2)、4.4 §6 决策树）。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// 手工路径上的脉冲（kit 只支持自动贝塞尔；L 形/直线通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 4.1 §2.2 — ring all-reduce 两阶段（block 0）
 * 原文：d = 4，c0..c3 四块；scatter-reduce 3 步后 r1 持 c0 全和、r2 持 c1、r3 持 c2、r0 持 c3；
 * all-gather 3 步后人人集齐。每 rank 发送 2(d−1)/d·M ≈ 2M bytes（式 2）
 * ============================================================ */
function ringAllReduce() {
  const UID = 'tr41a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'ring all-reduce（d = 4）：scatter-reduce 收齐部分和 → all-gather 扩散全和', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // 左：四 rank 环（r0→r1→r2→r3→r0），各 rank 标注 scatter-reduce 后持有的全和块
  const R = [
    { id: 'r0', x: 48, y: 66, title: 'rank 0', sub: '收齐 c₃ 全和' },
    { id: 'r1', x: 236, y: 66, title: 'rank 1', sub: '收齐 c₀ 全和' },
    { id: 'r2', x: 236, y: 178, title: 'rank 2', sub: '收齐 c₁ 全和' },
    { id: 'r3', x: 48, y: 178, title: 'rank 3', sub: '收齐 c₂ 全和' },
  ];
  const rn = R.map((r) => {
    const n = node({ id: r.id, x: r.x, y: r.y, w: 118, h: 52, role: 'proj', title: r.title, sub: r.sub, mono: true, titleSize: 13, subSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  // 环边（顺时针），脉冲沿发送方向
  body.push(edge(rn[0], rn[1], { nodes, pulse: true, dur: 2.0, delay: 0.0, uid: UID, eid: 0, arrow: false }));
  body.push(edge(rn[1], rn[2], { nodes, portFrom: rn[1].bottom, portTo: rn[2].top, pulse: true, dur: 2.0, delay: 0.5, uid: UID, eid: 1, arrow: false }));
  body.push(edge(rn[2], rn[3], { nodes, portFrom: rn[2].left, portTo: rn[3].right, pulse: true, dur: 2.0, delay: 1.0, uid: UID, eid: 2, arrow: false }));
  body.push(edge(rn[3], rn[0], { nodes, portFrom: rn[3].top, portTo: rn[0].bottom, pulse: true, dur: 2.0, delay: 1.5, uid: UID, eid: 3, arrow: false }));
  body.push(cap(197, 150, '环：每步每 rank 发 M/d', { size: 10, anchor: 'middle' }));
  body.push(cap(197, 166, '发给 r+1、收自 r−1', { size: 10, anchor: 'middle' }));

  // 右：两阶段的块持有矩阵（行 = rank，列 = 块 c0..c3）
  const MX = 448, CW = 15, CG = 5;
  const cellX = (j) => MX + 44 + j * (CW + CG);
  // 阶段一：scatter-reduce
  body.push(zone({ x: 420, y: 44, w: 276, h: 128, label: '① scatter-reduce · d−1 = 3 步' }));
  for (let j = 0; j < 4; j++) body.push(cap(cellX(j) + CW / 2, 86, `c${j}`, { mono: true, size: 9.5, anchor: 'middle' }));
  for (let i = 0; i < 4; i++) {
    body.push(cap(MX + 36, 94 + i * 19 + 11, `r${i}`, { mono: true, size: 9.5, anchor: 'end' }));
    for (let j = 0; j < 4; j++) {
      const full = j === (i + 3) % 4; // r1c0 r2c1 r3c2 r0c3
      const cls = full ? 'dia-mx-hot sig-cell' : 'dia-mx-off';
      const sty = full ? ` style="--i:${(0.3 + i * 0.45).toFixed(2)}"` : '';
      body.push(`<rect class="${cls}"${sty} x="${cellX(j)}" y="${94 + i * 19}" width="${CW}" height="${CW}" rx="3"/>`);
    }
  }
  body.push(cap(558, 190, '每块在某 rank 集齐全部 d 份贡献', { size: 10, anchor: 'middle' }));

  // 阶段二：all-gather
  body.push(zone({ x: 420, y: 204, w: 276, h: 128, label: '② all-gather · 3 步', tone: 'seal' }));
  for (let i = 0; i < 4; i++) {
    body.push(cap(MX + 36, 254 + i * 19 + 11, `r${i}`, { mono: true, size: 9.5, anchor: 'end' }));
    for (let j = 0; j < 4; j++) {
      body.push(`<rect class="dia-mx-warm sig-cell" style="--i:${(1.6 + i * 0.35 + j * 0.08).toFixed(2)}" x="${cellX(j)}" y="${254 + i * 19}" width="${CW}" height="${CW}" rx="3"/>`);
    }
  }
  body.push(cap(558, 348, '全和块沿环转发，人人集齐 c₀…c₃', { size: 10, anchor: 'middle' }));

  body.push(cap(36, 366, '每 rank 发送合计 2·(d−1)/d·M ≈ 2M bytes（式 2）——带宽账与 d 基本无关，这是 ring 撑到万卡的核心性质', { size: 11 }));
  body.push(cap(36, 386, '对照星型基线：rank 0 单点 2(d−1)·M，随 d 线性膨胀（7B bf16 梯度 M ≈ 13 GiB，d = 512 时每步约 6.5 TiB）', { size: 10.5 }));

  return scene(720, 400, body, { label: 'ring all-reduce：scatter-reduce 与 all-gather 两阶段，每 rank 约 2M 字节' }).svg();
}

/* ============================================================
 * 4.1 §3.2 — bucket 就绪即通信、与反向重叠（block 1）
 * 原文时间线：L4→L1 反向计算条，bucket 就绪即挂异步 AR，finalize 收尾写回 .grad
 * 构造期：参数按注册逆序入桶，默认 25 MiB、首桶约 1 MB（反向最先就绪）
 * ============================================================ */
function bucketOverlap() {
  const UID = 'tr41b';
  const body = [];

  body.push(cap(36, 28, 'DDP 反向期：梯度就绪即通信——桶按参数注册逆序依次就绪，AR 与后续反向并行', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // 时间轴
  body.push(`<line class="dia-edge strong" x1="60" y1="52" x2="684" y2="52"/>`);
  body.push(cap(684, 44, '时间 →', { size: 10, anchor: 'end' }));

  // Lane 1：GPU 反向计算（L4 → L1）
  body.push(cap(36, 92, 'GPU 计算', { size: 10.5 }));
  const layers = ['L4 bwd', 'L3 bwd', 'L2 bwd', 'L1 bwd'];
  const BX = 92, BW = 108, BG = 14;
  layers.forEach((t, i) => {
    const x = BX + i * (BW + BG);
    body.push(`<rect class="dia-bar dia-bar-ink sig-bar" style="--i:${(i * 0.8).toFixed(1)}" x="${x}" y="76" width="${BW}" height="22" rx="5"/>`);
    body.push(cap(x + BW / 2, 91, layers[i], { mono: true, size: 10.5, anchor: 'middle', color: 'var(--paper)' }));
  });

  // bucket 就绪点（每个计算条尾端）
  body.push(cap(36, 132, 'bucket 就绪', { size: 10.5 }));
  layers.forEach((_, i) => {
    const x = BX + i * (BW + BG) + BW;
    body.push(`<circle class="sig-cell" style="--i:${(0.5 + i * 0.7).toFixed(2)}" cx="${x}" cy="126" r="5" fill="var(--seal)"/>`);
    body.push(cap(x, 148, `B${i}`, { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  });
  body.push(cap(684, 166, 'B0 = 反向最先就绪的一桶（首桶缩小至约 1 MB，通信尽早启动）', { size: 9.5, anchor: 'end' }));

  // Lane 2：通信流（AR(B0)..AR(B3) 与后续计算重叠）
  body.push(cap(36, 192, '通信流', { size: 10.5 }));
  layers.forEach((_, i) => {
    const x = BX + i * (BW + BG) + BW + 4;
    const w = BW + BG - 8;
    body.push(`<rect class="dia-bar dia-bar-seal sig-bar" style="--i:${(1.2 + i * 0.8).toFixed(1)}" x="${x}" y="176" width="${w}" height="18" rx="5"/>`);
    body.push(cap(x + w / 2, 189, `AR(B${i})`, { mono: true, size: 9.5, anchor: 'middle', color: 'var(--paper)' }));
    // 就绪点 → AR 起点
    body.push(`<line class="dia-edge dash" x1="${x - 4}" y1="132" x2="${x}" y2="174"/>`);
  });

  // finalize
  const fin = node({ id: 'fin', x: 560, y: 210, w: 136, h: 44, role: 'op', title: 'finalize', sub: '等全部完成、写回 .grad', titleSize: 12.5, subSize: 9.5 });
  body.push(fin.svg);
  body.push(`<line class="dia-edge" x1="${BX + 3 * (BW + BG) + BW + 4 + BW + BG - 8}" y1="194" x2="628" y2="210"/>`);
  body.push(pathPulse(`M ${BX + 3 * (BW + BG) + BW + 4 + BW + BG - 8} 194 L 628 210`, { dur: 1.6, delay: 2.6 }));

  body.push(cap(36, 282, 'hook 机制：每个参数的 AccumulateGrad 装同一 post-hook；桶内 hook 全触发 → 最后一个 hook 发异步 all-reduce，随即返回继续反向', { size: 10.5 }));
  body.push(cap(36, 302, 'overlap 判据：桶通信耗时 ≤ 该桶就绪后的剩余反向时间；桶默认 25 MiB（论文 Fig. 2 的粒度-耗时实测）', { size: 10.5 }));
  body.push(cap(36, 322, 'finalize 之后才轮到 clip → step → zero_grad；平均语义由 reducer 的 div_factor_ 在归约时折入（§3.2）', { size: 10.5 }));

  return scene(720, 338, body, { label: 'DDP bucket overlap：反向按层逆序推进，桶就绪即发异步 all-reduce 与后续计算重叠' }).svg();
}

/* ============================================================
 * 4.2 §2 — DP vs ZeRO-1 通信阶段对照（block 0，中优先级）
 * 原文：两者每 rank 各 2N 元素、严格相等；ZeRO-1 把第二段的内容从梯度换成更新后参数
 * ============================================================ */
function dpVsZero1() {
  const UID = 'tr42a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, '通信量不变性：DP 的一次 all-reduce = reduce-scatter + all-gather，ZeRO-1 让两段各落在本来就需要的事件上', { size: 12, color: 'var(--ink)', weight: 600 }));

  const lane = (y, tag, stages) => {
    body.push(zone({ x: 32, y: y - 26, w: 656, h: 78, label: tag }));
    const ns = stages.map((s, i) => {
      const n = node({ id: `${tag}${i}`, x: 52 + i * 162, y, w: 142, h: 44, role: s.role, title: s.title, sub: s.sub, mono: s.mono ?? false, titleSize: s.ts ?? 12, subSize: 10 });
      nodes.push(n); body.push(n.svg);
      return n;
    });
    for (let i = 1; i < ns.length; i++) {
      body.push(edge(ns[i - 1], ns[i], { nodes, pulse: true, dur: 1.7, delay: (tag === 'DP' ? 0 : 0.4) + i * 0.3, uid: UID, eid: `${tag}${i}` }));
    }
    return ns;
  };

  lane(72, 'DP', [
    { role: 'data', title: '本地前向反向', sub: '全量梯度' },
    { role: 'attn', title: 'all-reduce', sub: 'RS+AG · 2N 元素', mono: true, ts: 12.5 },
    { role: 'op', title: '全量参数更新', sub: '各 rank 同一份' },
    { role: 'plain', title: '（无参数同步）', sub: '参数本就全量', ts: 11 },
  ]);
  lane(190, 'ZeRO-1', [
    { role: 'data', title: '本地前向反向', sub: '全量梯度' },
    { role: 'attn', title: 'reduce-scatter', sub: 'N 元素 → 自己分区', mono: true, ts: 11.5 },
    { role: 'op', title: '1/d 分区更新', sub: '只更新自己的 fp32 分片', mono: true, ts: 11.5 },
    { role: 'proj', title: 'all-gather 参数', sub: 'N 元素 · 拼回全量', mono: true, ts: 11.5 },
  ]);

  // 等量大括注
  body.push(cap(360, 262, '每 rank 通信量：2N = N + N，严格相等（论文 §7.2.1 "same communication volume as DP"）', { size: 11, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));
  body.push(cap(360, 282, '区别只有一个：第二段搬运的内容从「归约后的梯度」换成「更新后的参数」——同为 bf16、同 N 个元素', { size: 10.5, anchor: 'middle' }));

  return scene(720, 296, body, { label: 'DP 与 ZeRO-1 的通信阶段对照：通信量严格相等，各 2N 元素' }).svg();
}

/* ============================================================
 * 4.2 §4.1 — ZeRO-3 / FSDP 逐层取回节奏（block 1）
 * 原文：前向正序 AG→用→释放；反向逆序再 AG；梯度 RS 归 owner；
 * 常驻 = 2N/d 参数 + 12N/d 优化器 + 2N/d 梯度；通信 3N 元素 = ×1.5（式 3）
 * ============================================================ */
function zero3Rhythm() {
  const UID = 'tr42b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'ZeRO-3 一个 step 的取回时序：参数「聚 → 算 → 散」，反向再见一次参数（matmul 反向的硬输入）', { size: 12, color: 'var(--ink)', weight: 600 }));

  // 前向链 L1→L4
  body.push(cap(36, 62, '前向（正序，可预取下一层）', { size: 10.5, color: 'var(--ink)' }));
  const fw = [];
  for (let i = 0; i < 4; i++) {
    const x = 60 + i * 164;
    const ag = node({ id: `ag${i}`, x, y: 76, w: 64, h: 34, role: 'cache', title: `AG`, sub: `L${i + 1}`, mono: true, titleSize: 11.5, subSize: 9.5 });
    const f = node({ id: `fw${i}`, x: x + 76, y: 76, w: 76, h: 34, role: 'attn', title: `fwd L${i + 1}`, mono: true, titleSize: 11 });
    nodes.push(ag, f); body.push(ag.svg, f.svg);
    body.push(edge(ag, f, { nodes, pulse: true, dur: 1.2, delay: i * 0.4, uid: UID, eid: `a${i}`, arrow: false }));
    // 释放虚线回常驻带
    body.push(`<line class="dia-edge dash" x1="${x + 114}" y1="110" x2="${x + 114}" y2="152"/>`);
    body.push(cap(x + 118, 146, '释放', { size: 9, color: 'var(--ink-faint)' }));
    fw.push(f);
  }
  // 前向链连接（跨组）
  for (let i = 0; i < 3; i++) {
    body.push(`<path class="dia-edge" d="M ${60 + i * 164 + 152} 93 C ${60 + i * 164 + 168} 93, ${60 + (i + 1) * 164 - 8} 93, ${60 + (i + 1) * 164} 93"/>`);
    body.push(pathPulse(`M ${60 + i * 164 + 152} 93 C ${60 + i * 164 + 168} 93, ${60 + (i + 1) * 164 - 8} 93, ${60 + (i + 1) * 164} 93`, { dur: 1.6, delay: 0.4 + i * 0.4 }));
  }

  // 常驻带
  const res = node({ id: 'res', x: 60, y: 156, w: 596, h: 40, role: 'data', title: '常驻显存（全 step）：2N/d 参数分片 + 12N/d 优化器分片 + 2N/d 梯度分片', sub: '暂态（不入常驻）：当前层全量参数 + 预取中的下一层——不随 d 缩', mono: true, titleSize: 11.5, subSize: 10 });
  nodes.push(res); body.push(res.svg);

  // 反向链 L4→L1（右到左）
  body.push(cap(36, 230, '反向（逆序）：再 AG 一次；梯度 RS 归 owner', { size: 10.5, color: 'var(--ink)' }));
  const bw = [];
  for (let i = 0; i < 4; i++) {
    const L = 4 - i;
    const x = 60 + i * 164;
    const ag = node({ id: `bag${i}`, x, y: 244, w: 64, h: 34, role: 'cache', title: `AG`, sub: `L${L}`, mono: true, titleSize: 11.5, subSize: 9.5 });
    const b = node({ id: `bw${i}`, x: x + 76, y: 244, w: 76, h: 34, role: 'attn', title: `bwd L${L}`, mono: true, titleSize: 11 });
    nodes.push(ag, b); body.push(ag.svg, b.svg);
    body.push(edge(ag, b, { nodes, pulse: true, dur: 1.2, delay: 0.3 + i * 0.4, uid: UID, eid: `b${i}`, arrow: false }));
    // RS 标记
    body.push(`<rect class="dia-bar dia-bar-seal sig-cell" style="--i:${(1.0 + i * 0.4).toFixed(2)}" x="${x + 76}" y="284" width="76" height="10" rx="4"/>`);
    body.push(cap(x + 114, 306, `RS(L${L}) 梯度归 owner`, { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
    bw.push(b);
  }
  for (let i = 0; i < 3; i++) {
    body.push(`<path class="dia-edge" d="M ${60 + i * 164 + 152} 261 C ${60 + i * 164 + 168} 261, ${60 + (i + 1) * 164 - 8} 261, ${60 + (i + 1) * 164} 261"/>`);
    body.push(pathPulse(`M ${60 + i * 164 + 152} 261 C ${60 + i * 164 + 168} 261, ${60 + (i + 1) * 164 - 8} 261, ${60 + (i + 1) * 164} 261`, { dur: 1.6, delay: 0.7 + i * 0.4 }));
  }

  body.push(cap(360, 336, '通信账：前向 AG ≈ N + 反向 AG ≈ N + 梯度 RS ≈ N = 3N 元素 = DP 的 ×1.5 上限（式 3）；更新后参数同步 = 0', { size: 11, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));

  return scene(720, 352, body, { label: 'ZeRO-3 逐层节奏：前向 all-gather 用后释放，反向再聚一次，梯度 reduce-scatter 归 owner' }).svg();
}

/* ============================================================
 * 4.3 §3 — FSDP 单 unit 生命周期与预取重叠（block 0，中优先级）
 * 原文：unit ℓ−1 计算中预取 AG(ℓ)；AG 完成→全量就绪→前向→释放（reshard）；
 * 反向再 AG(ℓ) → vjp → 全量梯度短暂存在 → RS(ℓ) → 释放
 * 峰值：M = 16N/d + 2P_u(2+n_pf) + M_act + M_buf（式 2）
 * ============================================================ */
function fsdpUnitLifecycle() {
  const UID = 'tr43a';
  const body = [];

  body.push(cap(36, 28, 'FSDP 单 unit 生命周期：预取与上 unit 计算重叠，用完即释放（逐 block 包装时 P_u ≈ N/L）', { size: 12, color: 'var(--ink)', weight: 600 }));

  // 时间轴
  body.push(`<line class="dia-edge strong" x1="60" y1="52" x2="684" y2="52"/>`);
  body.push(cap(684, 44, '时间 →', { size: 10, anchor: 'end' }));

  // Lane 1：计算
  body.push(cap(36, 92, '计算', { size: 10.5 }));
  const comps = [
    { x: 92, w: 150, t: 'unit ℓ−1 计算', tone: 'ink' },
    { x: 262, w: 150, t: 'unit ℓ 前向', tone: 'seal' },
    { x: 432, w: 100, t: 'ℓ+1 …', tone: 'soft' },
    { x: 556, w: 120, t: 'unit ℓ 反向 vjp', tone: 'seal' },
  ];
  comps.forEach((c, i) => {
    body.push(`<rect class="dia-bar dia-bar-${c.tone} sig-bar" style="--i:${(i * 0.7).toFixed(1)}" x="${c.x}" y="76" width="${c.w}" height="22" rx="5"/>`);
    body.push(cap(c.x + c.w / 2, 91, c.t, { mono: true, size: 10.5, anchor: 'middle', color: 'var(--paper)' }));
  });

  // Lane 2：通信（预取 AG(ℓ) 与 ℓ−1 计算并行；反向前再 AG；RS 收尾）
  body.push(cap(36, 142, '通信', { size: 10.5 }));
  const comms = [
    { x: 100, w: 134, t: 'AG(ℓ) 预取', tone: 'warm', note: '与 ℓ−1 计算重叠' },
    { x: 440, w: 96, t: 'AG(ℓ) 再取', tone: 'warm', note: '' },
    { x: 566, w: 110, t: 'RS(ℓ)', tone: 'seal', note: '梯度归约并分片' },
  ];
  comms.forEach((c, i) => {
    body.push(`<rect class="dia-bar dia-bar-${c.tone} sig-bar" style="--i:${(0.5 + i * 0.9).toFixed(1)}" x="${c.x}" y="126" width="${c.w}" height="18" rx="5"/>`);
    body.push(cap(c.x + c.w / 2, 138, c.t, { mono: true, size: 9.5, anchor: 'middle', color: c.tone === 'warm' ? 'var(--ink)' : 'var(--paper)' }));
    if (c.note) body.push(cap(c.x + c.w / 2, 166, c.note, { size: 9 }));
  });

  // 生命周期里程碑（unit ℓ）
  body.push(cap(36, 196, 'unit ℓ 状态', { size: 10.5 }));
  const ms = [
    { x: 262, t: 'AG 完成 → 全量就绪' },
    { x: 412, t: '前向完 → 释放（reshard）' },
    { x: 556, t: '反向完 → RS → 释放参数与梯度' },
  ];
  ms.forEach((m, i) => {
    body.push(`<circle class="sig-cell" style="--i:${(1.2 + i * 0.6).toFixed(2)}" cx="${m.x}" cy="192" r="4.5" fill="var(--seal)"/>`);
    body.push(cap(m.x + 10, 196, m.t, { size: 10 }));
  });

  // 常驻带
  body.push(zone({ x: 32, y: 216, w: 656, h: 40, label: '' }));
  body.push(cap(48, 241, '常驻（全 step）：分片参数 2N/d + 分片梯度 2N/d + optimizer 态 12N/d', { mono: true, size: 11 }));

  body.push(cap(36, 284, '峰值显存（式 2）：M_peak = 16N/d + 2·P_u·(2 + n_pf) + M_act + M_buf', { mono: true, size: 11.5, color: 'var(--seal-deep)', weight: 600 }));
  body.push(cap(36, 304, '7B · d=8 · L=32 · n_pf=1：常驻 ≈ 13.0 GiB，暂态 ≈ 1.2 GiB（仅常驻 9%）——「全分片 ≈ 显存 ÷d」在逐 block 粒度下成立', { size: 10.5 }));

  return scene(720, 320, body, { label: 'FSDP 单 unit 生命周期：预取 all-gather 与上 unit 计算重叠，用后释放' }).svg();
}

/* ============================================================
 * 4.4 §6 — DP / ZeRO / FSDP 家族决策树（block 0）
 * 原文 17 行决策树逐层分叉；两条总则：同能装下取浅分片；
 * 可行 d = 显存下界 × 通信上界
 * ============================================================ */
function memoryDecisionTree() {
  const UID = 'tr44a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, '每个节点的判定只有一行账：显存五项 vs 标称容量；字节 ÷ 带宽 vs 单步计算', { size: 12, color: 'var(--ink)', weight: 600 }));

  // 根判定
  const q0 = node({ id: 'q0', x: 36, y: 48, w: 236, h: 52, role: 'op', title: '三态 + 激活 + buffer', sub: '装进单卡？（式 1 逐项代入）', titleSize: 12.5, subSize: 10.5 });
  nodes.push(q0); body.push(q0.svg);

  // 分支 1：16N 装得下 → DDP → ZeRO-1
  const ddp = node({ id: 'ddp', x: 356, y: 48, w: 150, h: 48, role: 'data', title: 'DDP', sub: '实现最简 · bucket overlap', titleSize: 13, subSize: 10 });
  const z1 = node({ id: 'z1', x: 540, y: 48, w: 156, h: 48, role: 'proj', title: 'ZeRO-1', sub: '通信同 DDP，白拿 12N/d', mono: true, titleSize: 13, subSize: 9.5 });
  nodes.push(ddp, z1); body.push(ddp.svg, z1.svg);
  body.push(edge(q0, ddp, { nodes, pulse: true, dur: 1.5, delay: 0.1, uid: UID, eid: 0 }));
  body.push(cap(314, 64, '16N 装得下', { size: 9.5, anchor: 'middle' }));
  body.push(edge(ddp, z1, { nodes, pulse: true, dur: 1.5, delay: 0.4, uid: UID, eid: 1 }));
  body.push(cap(523, 64, '想削 optimizer 态', { size: 9, anchor: 'middle' }));

  // 分支 2：装不下 → ZeRO-2
  const z2 = node({ id: 'z2', x: 356, y: 124, w: 236, h: 48, role: 'proj', title: 'ZeRO-2 / FSDP reshard=False', sub: '2N + 14N/d · 通信仍 ≈ 4N', mono: true, titleSize: 11.5, subSize: 10 });
  nodes.push(z2); body.push(z2.svg);
  body.push(edge(q0, z2, { nodes, portFrom: q0.bottom, portTo: z2.left, pulse: true, dur: 1.6, delay: 0.6, uid: UID, eid: 2 }));
  body.push(cap(318, 132, '2N+14N/d 装得下', { size: 9.5, anchor: 'end' }));
  body.push(cap(474, 192, '生产实例：Llama-3 405B 的 FSDP 即此档（不前向后重分片参数）', { size: 9.5, anchor: 'middle' }));

  // 分支 3：还不行 → ZeRO-3
  const z3 = node({ id: 'z3', x: 356, y: 208, w: 236, h: 48, role: 'attn', title: 'ZeRO-3 / FSDP reshard=True', sub: '16N/d · 无地板 · 通信 ×1.5', mono: true, titleSize: 11.5, subSize: 10 });
  nodes.push(z3); body.push(z3.svg);
  body.push(edge(q0, z3, { nodes, portFrom: q0.bottom, portTo: z3.left, pulse: true, dur: 1.7, delay: 0.8, uid: UID, eid: 3 }));
  body.push(cap(318, 200, '还不行', { size: 9.5, anchor: 'end' }));

  // ZeRO-3 的三条补救：左侧竖脊 + 短横支
  const SPINE_X = 330;
  body.push(`<path class="dia-edge dash" d="M ${z3.x + 30} 256 C ${z3.x + 10} 268, ${SPINE_X} 272, ${SPINE_X} 288 L ${SPINE_X} 388"/>`);
  const rem = [
    { y: 272, title: '通信顶不住 → hybrid / HSDP', sub: '分片组收进节点，跨机 ÷ d_s（4.3 式 1）' },
    { y: 330, title: '峰值仍超 → 先动激活', sub: 'selective → full ckpt（+2–7% / +30–40%）' },
    { y: 388, title: '加卡也不可行 → CPU offload', sub: 'ZeRO-Offload（4.2 §5）· PCIe 下界 4N/step' },
  ];
  rem.forEach((r, i) => {
    const n = node({ id: `rem${i}`, x: 372, y: r.y, w: 324, h: 46, role: 'plain', title: r.title, sub: r.sub, titleSize: 11, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    body.push(`<line class="dia-edge" x1="${SPINE_X}" y1="${r.y + 23}" x2="372" y2="${r.y + 23}"/>`);
    body.push(pathPulse(`M ${SPINE_X} ${r.y + 23} L 372 ${r.y + 23}`, { dur: 1.2, delay: 1.2 + i * 0.25, r: 3.4 }));
  });

  // 判定线之外
  const tp = node({ id: 'tp', x: 36, y: 344, w: 236, h: 60, role: 'cache', title: '判定线之外 → TP / PP（P5）', sub: '激活或单步延迟成瓶颈；组合归 5.5', titleSize: 11.5, subSize: 10 });
  nodes.push(tp); body.push(tp.svg);
  body.push(edge(q0, tp, { nodes, portFrom: q0.bottom, portTo: tp.top, pulse: true, dur: 1.8, delay: 1.2, uid: UID, eid: 9, dash: true }));

  body.push(cap(36, 448, '总则：同样装得下时取分片更浅的（ZeRO-1/2 通信与 DDP 相同）；可行 d 区间 = 显存下界 × 通信上界', { size: 11, color: 'var(--seal-deep)', weight: 600 }));

  return scene(720, 464, body, { label: '显存决策树：DDP、ZeRO-1/2/3、hybrid、checkpoint、offload 到 TP/PP 的逐层分叉' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'training',
    slug: '04-data-parallelism/01-dp-and-ddp',
    replacesBlock: 0,
    title: 'ring all-reduce：scatter-reduce + all-gather',
    svg: ringAllReduce,
  },
  {
    dim: 'training',
    slug: '04-data-parallelism/01-dp-and-ddp',
    replacesBlock: 1,
    title: 'DDP bucket overlap：梯度就绪即通信',
    svg: bucketOverlap,
  },
  {
    dim: 'training',
    slug: '04-data-parallelism/02-zero',
    replacesBlock: 0,
    title: 'DP vs ZeRO-1：通信量严格相等',
    svg: dpVsZero1,
  },
  {
    dim: 'training',
    slug: '04-data-parallelism/02-zero',
    replacesBlock: 1,
    title: 'ZeRO-3 逐层节奏：聚 → 算 → 散',
    svg: zero3Rhythm,
  },
  {
    dim: 'training',
    slug: '04-data-parallelism/03-fsdp',
    replacesBlock: 0,
    title: 'FSDP 单 unit 生命周期与预取重叠',
    svg: fsdpUnitLifecycle,
  },
  {
    dim: 'training',
    slug: '04-data-parallelism/04-memory-and-comms-accounting',
    replacesBlock: 0,
    title: 'DP / ZeRO / FSDP 家族决策树',
    svg: memoryDecisionTree,
  },
];
