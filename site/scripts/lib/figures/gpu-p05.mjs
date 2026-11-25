// gpu P5（05-kernel-compute-primitives）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-cell --i ≤ 2.6。
// census：5.1 block0 reduction tree / block1 shfl_down / block2 Blelloch；
//         5.2 block0 嵌套循环（伪代码，不画）/ block1 五层实体 / block2 producer-consumer 体（伪代码，不画）/ block3 mainloop 时序；
//         5.3 block0 三变体 IO / block1 ln_fwd 五段（伪代码，不画）；5.4 block0 三类策略。

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
 * 5.1 §2（block 0）— reduction tree + 层级映射（达 Ω(n_el) 下界）
 * ============================================================ */
function reductionTree() {
  const UID = 'g5a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'reduction tree：每元素恰读一次达 Ω(n_el) 下界 —— 8 元素例 W=7 / D=3，层级拼接 depth≈81', { size: 12, color: 'var(--ink-soft)' }));

  // 左：8 元素二叉树
  body.push(zone({ x: 36, y: 44, w: 340, h: 182, label: '8 元素 reduction tree（W = 7 · D = 3）' }));
  const leafX = [56, 96, 136, 176, 216, 256, 296, 336];
  leafX.forEach((x, i) => body.push(cap(x, 80, `x${i}`, { mono: true, size: 10, anchor: 'middle' })));
  const l0 = [['a', 76], ['b', 156], ['c', 236], ['d', 316]].map(([t, cx]) => {
    const n = node({ id: t, x: cx - 18, y: 94, w: 36, h: 22, role: 'proj', title: t, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const l1 = [['ab', 116], ['cd', 276]].map(([t, cx]) => {
    const n = node({ id: t, x: cx - 22, y: 136, w: 44, h: 22, role: 'proj', title: t, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const root = node({ id: 'abcd', x: 170, y: 178, w: 52, h: 24, role: 'attn', title: 'abcd', mono: true, titleSize: 11 });
  nodes.push(root); body.push(root.svg);
  // 叶 → 层0
  leafX.forEach((x, i) => {
    const tgt = l0[Math.floor(i / 2)];
    body.push(edge([x, 84], [tgt.x + tgt.w / 2, 94], { nodes, pulse: i % 2 === 0, dur: 1.2, delay: i * 0.12, uid: UID, eid: `lf${i}`, arrow: false }));
  });
  // 层0 → 层1 → 根
  [[l0[0], l1[0]], [l0[1], l1[0]], [l0[2], l1[1]], [l0[3], l1[1]]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: 0.5 + i * 0.15, uid: UID, eid: `m${i}`, arrow: false }));
  });
  [[l1[0], root], [l1[1], root]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: 1.1 + i * 0.2, uid: UID, eid: `r${i}`, arrow: false }));
  });
  body.push(cap(196, 216, '层0→1→2：4 / 2 / 1 次并行加法 ← 结果', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 右：层级映射
  body.push(zone({ x: 396, y: 44, w: 288, h: 246, label: '层级映射（本章配置 256 线程/block）' }));
  const hier = [
    ['register：64 元素/线程串行链', 'grid-stride 装载 · HBM 每元素读一次', 'data'],
    ['warp：32 lane → shuffle 5 步', '免 shared（§3 命题 3）', 'cache'],
    ['block：8 partial → shared 8 字', 'warp0 shuffle 3 步（§4）', 'norm'],
    ['grid：1024 partial → 末趟归约', 'atomic / two-pass / cooperative', 'proj'],
  ].map(([t, s, role], i) => {
    // h=44：sub 基线 y+h/2+16=y+38，距卡底 y+44 留 6px，不骑边框
    const n = node({ id: `h${i}`, x: 412, y: 70 + i * 56, w: 256, h: 44, role, title: t, sub: s, titleSize: 10.5, subSize: 8 });
    nodes.push(n); body.push(n.svg); return n;
  });
  for (let i = 0; i < 3; i++) {
    // 短竖边不加 pulse（端点 dwell 会压 sub 字区）；流向由箭头 + 右侧侧bus光点表现
    body.push(edge(hier[i], hier[i + 1], { nodes, portFrom: hier[i].bottom, portTo: hier[i + 1].top, uid: UID, eid: `h${i}`, arrow: false }));
    body.push(arrowHead(hier[i + 1].top[0], hier[i + 1].top[1], 0, 1));
  }
  body.push(pathPulse('M 676 70 V 282', { dur: 2.4, delay: 0.6 }));

  body.push(cap(36, 310, '主口径 n_el = 2^24 = 16777216 fp32：读 67.11 MB ÷ 3352 GB/s = 20.02 μs（命题 1 下界；partial 4 KB 占 0.0061%）', { size: 10 }));
  body.push(cap(36, 330, 'work/depth（命题 2）：W = n_el−1 = 16777215、D = log2 n_el = 24；串行 depth 16777215 —— work 同、depth 差 70 万倍', { size: 10 }));
  body.push(cap(36, 350, '实际 depth = 层级拼接 63+5+3+10 ≈ 81 步；Brent 调度 W/p + D ≈ 64+24 = 88（p = 262144 线程，reduce6 注释点名）', { size: 10 }));
  body.push(cap(36, 370, '优化目标是省搬运不是省加法：强度 I ≈ 1/bytes = 0.25 FLOP/byte，离 fp32 ridge ≈20 差约 80 倍（295 是 BF16 TC ridge）→ memory-bound', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 386, body, { label: 'reduction tree：8 元素二叉树 W 等于 7 D 等于 3，右侧 register 到 warp 到 block 到 grid 层级映射，主口径 2 的 24 次方元素读 67.11 MB 得 20.02 微秒下界' }).svg();
}

/* ============================================================
 * 5.1 §3（block 1）— warp shuffle：down 形 lane 态 + butterfly
 * ============================================================ */
function warpShuffle() {
  const UID = 'g5b';
  const body = [];
  const nodes = [];
  const cx = [140, 212, 284, 356, 428, 500, 572, 644]; // 8 lane 列中心

  body.push(cap(36, 24, 'warp shuffle：免 shared 的寄存器交换 —— down 形 lane0 持和、xor butterfly 全 lane 持和', { size: 12, color: 'var(--ink-soft)' }));

  // down 形 lane 态表
  body.push(zone({ x: 36, y: 44, w: 648, h: 140, label: '__shfl_down_sync：8 lane 最小例（offset 4,2,1）；32 lane 全形 16,8,4,2,1 共 5 步' }));
  body.push(cap(48, 78, 'lane', { mono: true, size: 9 }));
  cx.forEach((x, i) => body.push(cap(x, 78, String(i), { mono: true, size: 9.5, anchor: 'middle', color: 'var(--ink)' })));
  const rows = [
    ['初始', ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], [0, 0, 0, 0, 0, 0, 0, 0]],
    ['off=4', ['x0+x4', 'x1+x5', 'x2+x6', 'x3+x7', 'x4', 'x5', 'x6', 'x7'], [0, 0, 0, 0, 1, 1, 1, 1]],
    ['off=2', ['x0+x4+x2+x6', 'x1+x5+x3+x7', 'x2+x6', 'x3+x7', '·', '·', '·', '·'], [0, 0, 0, 0, 1, 1, 1, 1]],
    ['off=1', ['Σx0..x7', '·', '·', '·', '·', '·', '·', '·'], [2, 1, 1, 1, 1, 1, 1, 1]],
  ];
  rows.forEach(([lab, cells, faint], ri) => {
    const y = 100 + ri * 24;
    body.push(cap(48, y, lab, { mono: true, size: 9 }));
    cells.forEach((t, ci) => {
      const hot = faint[ci] === 2;
      const fill = hot ? 'var(--seal-wash)' : 'var(--paper-raised)';
      const stroke = hot ? 'var(--seal)' : 'var(--rule)';
      body.push(`<rect x="${cx[ci] - 34}" y="${y - 14}" width="68" height="19" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`);
      body.push(cap(cx[ci], y, t, { mono: true, size: 8.5, anchor: 'middle', color: faint[ci] === 1 ? 'var(--ink-faint)' : hot ? 'var(--seal-deep)' : 'var(--ink)' }));
    });
  });
  body.push(cap(36, 196, '表列 shuffle 源值语义：越界不环绕（PG「will not wrap around」）；v += shfl 下高位 lane 实持翻倍旧值，仅 lane0 累加轨迹干净、结果只取 lane0', { size: 9.5 }));

  // butterfly 对照
  body.push(zone({ x: 36, y: 208, w: 648, h: 74, label: '__shfl_xor_sync：butterfly（broadcast 式全归约）' }));
  cx.forEach((x, i) => {
    body.push(`<rect x="${x - 30}" y="240" width="60" height="20" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1"/>`);
    body.push(cap(x, 254, 'Σ', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
    if (i < 7) body.push(`<path class="dia-edge" d="M${x + 30} 250 H${cx[i + 1] - 30}"/>`);
  });
  body.push(cap(360, 274, '第 d 步 lane ℓ 与 ℓ⊕2^d 交换相加，5 步后每 lane 依赖扇入覆盖全 32 lane（FA2 quad 行归约即 4 lane 子组 butterfly）', { size: 9, anchor: 'middle' }));

  // shared 流量账
  body.push(cap(36, 300, 'block 级 shared 流量（命题 3，w_b=8）：shuffle 方案 64 B 对朴素 shared tree 4084 B —— 省 63.8×', { size: 10.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="36" y="310" width="${64 * 0.11}" height="14" rx="3"/>`);
  body.push(cap(36 + 64 * 0.11 + 8, 321, '64 B（shuffle）', { mono: true, size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="200" y="310" width="${4084 * 0.11}" height="14" rx="3"/>`);
  body.push(cap(200 + 4084 * 0.11 + 8, 321, '4084 B', { mono: true, size: 9 }));
  body.push(cap(36, 344, 'conflict 0（shared 流量 0）· barrier 1 对 8 · shuffle 吞吐 32 results/cycle/SM；redux.sync 加法仅 u32/s32 → fp32 求和仍走 shuffle tree', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 358, body, { label: 'warp shuffle：down 形八 lane 逐 offset 状态表 lane0 持全和，xor butterfly 每 lane 持全和，block 级 shared 流量 64 B 对朴素 4084 B 省 63.8 倍' }).svg();
}

/* ============================================================
 * 5.1 §5（block 2）— Blelloch scan：up-sweep 树 + down-sweep 前缀
 * ============================================================ */
function blellochScan() {
  const UID = 'g5c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Blelloch scan：up-sweep 归约子块和 + down-sweep 分发前缀 —— 2 倍 work 换 O(log n_el) depth', { size: 12, color: 'var(--ink-soft)' }));

  // up-sweep 树（x = [1..8]）
  body.push(zone({ x: 36, y: 44, w: 648, h: 176, label: 'up-sweep（右节点 += 左节点，层 d 步长 2^{d+1}）：x = [1,2,3,4,5,6,7,8]' }));
  const leafX = [72, 154, 236, 318, 400, 482, 564, 646];
  const leafV = [1, 2, 3, 4, 5, 6, 7, 8];
  leafX.forEach((x, i) => body.push(cap(x, 196, String(leafV[i]), { mono: true, size: 10, anchor: 'middle' })));
  const up0 = [[3, 113], [7, 277], [11, 441], [15, 605]].map(([v, cx], i) => {
    const n = node({ id: `u${i}`, x: cx - 20, y: 150, w: 40, h: 22, role: 'proj', title: String(v), mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const up1 = [[10, 195], [26, 523]].map(([v, cx], i) => {
    const n = node({ id: `v${i}`, x: cx - 22, y: 108, w: 44, h: 22, role: 'proj', title: String(v), mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const rt = node({ id: 'rt', x: 337, y: 66, w: 44, h: 24, role: 'attn', title: '36', mono: true, titleSize: 12 });
  nodes.push(rt); body.push(rt.svg);
  leafX.forEach((x, i) => {
    const tgt = up0[Math.floor(i / 2)];
    // dur2.4/delay2.3 → 3.5s 时相位恰在中段，光点不落在节点数字上
    body.push(edge([x, 188], [tgt.x + tgt.w / 2, 172], { nodes, pulse: i % 2 === 0, dur: 2.4, delay: 2.3, uid: UID, eid: `ul${i}`, arrow: false }));
  });
  // 短内边（up0→up1→root）改 top→bottom 正确走向且不加 pulse（端点 dwell 会遮 10/26/36）
  [[up0[0], up1[0]], [up0[1], up1[0]], [up0[2], up1[1]], [up0[3], up1[1]]].forEach(([a, b], i) =>
    body.push(edge(a, b, { nodes, portFrom: a.top, portTo: b.bottom, uid: UID, eid: `um${i}`, arrow: false })));
  [[up1[0], rt], [up1[1], rt]].forEach(([a, b], i) =>
    body.push(edge(a, b, { nodes, portFrom: a.top, portTo: b.bottom, uid: UID, eid: `ur${i}`, arrow: false })));
  body.push(cap(392, 82, '根 = 总和 36 → 置根为 0', { size: 9, color: 'var(--seal-deep)' }));

  // down-sweep 结果 strip
  body.push(cap(36, 244, 'down-sweep（左↔右 swap：左=右旧值，右=左旧值+右旧值）→ exclusive scan y = [0,1,3,6,10,15,21,28]', { size: 10.5 }));
  const yV = [0, 1, 3, 6, 10, 15, 21, 28];
  leafX.forEach((x, i) => {
    body.push(`<rect x="${x - 26}" y="254" width="52" height="22" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1"/>`);
    body.push(cap(x, 269, String(yV[i]), { mono: true, size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  });
  body.push(cap(360, 292, '核对：末项 28 + x[7]=8 = 总和 36 ✓；一次 swap 同时完成前缀下传（左子）与右子前缀构造（命题 5 不变式）', { size: 9.5, anchor: 'middle' }));

  body.push(cap(36, 318, 'work/depth（命题 4）：up/down 各 n_el−1 加法 + (n_el−1) 交换，W = 2(n_el−1)、D = 2log2 n_el；2^24 给 33554430 加法 / depth 48', { size: 10 }));
  body.push(cap(36, 338, '对 offset-doubling（work 约 4.03 亿）省 12× work；scan HBM 读+写 2n_el·bytes = 134.22 MB / 40.04 μs，恰 reduction 2 倍', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 354, body, { label: 'Blelloch scan：up-sweep 二叉树归约出根总和 36，置根零后 down-sweep 交换分发前缀得 exclusive scan 0 1 3 6 10 15 21 28，work 两倍 depth 对数' }).svg();
}

/* ============================================================
 * 5.2 §3（block 1）— CUTLASS 五层实体 + tile 层级数据流
 * ============================================================ */
function cutlassHierarchy() {
  const UID = 'g5d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'CUTLASS 4.9.0 五层实体（SM90）：device→kernel→collective→CuTe→instruction，tile 128×128×64', { size: 12, color: 'var(--ink-soft)' }));

  // 左：五层实体链（自顶向下）
  const layers = [
    ['device', 'GemmUniversal', 'gemm_universal.h L142/L289 · initialize L398', 'proj'],
    ['kernel', 'GemmUniversal<Mainloop,Epilogue,Sched>', 'sm90_gemm_tma_warpspecialized.hpp L65 · operator() L269', 'proj'],
    ['collective', 'CollectiveMma + CollectiveEpilogue', 'sm90_mma_tma_gmma_ss_warpspecialized.hpp L72', 'norm'],
    ['CuTe', 'TiledMMA / ThrMMA + Layout', 'mma_atom.hpp L211/L460 · layout.hpp L99', 'data'],
    ['instruction', 'wgmma.mma_async m64n128k16', 'mma_sm90_gmma.hpp · bf16 SS 形 L2842', 'attn'],
  ].map(([name, ent, file, role], i) => {
    const n = node({ id: `L${i}`, x: 36, y: 56 + i * 62, w: 320, h: 52, role, title: `${name}：${ent}`, sub: file, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg); return n;
  });
  for (let i = 0; i < 4; i++) {
    body.push(edge(layers[i], layers[i + 1], { nodes, portFrom: layers[i].bottom, portTo: layers[i + 1].top, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `c${i}`, arrow: false }));
    body.push(arrowHead(layers[i + 1].top[0], layers[i + 1].top[1], 0, 1));
  }
  body.push(cap(36, 372, '2.x 显式目录 threadblock/warp/thread → 3.x/4.x collective + CuTe 代数（层级藏模板组合里）', { size: 9 }));

  // 右：tile 层级数据流（HBM→L2→shared→TC）
  body.push(zone({ x: 380, y: 44, w: 304, h: 320, label: 'tile 层级数据流（每 block 每 K tile）' }));
  const hbm = node({ id: 'hbm', x: 396, y: 70, w: 272, h: 40, role: 'cache', title: 'HBM → L2（TMA）', sub: '32768 B 无 L2；有效 ÷11.49 ≈ 2852 B', mono: true, titleSize: 11, subSize: 8 });
  const shm = node({ id: 'shm', x: 396, y: 138, w: 272, h: 40, role: 'cache', title: 'shared（4 stage × 32768 B）', sub: 'A/B 描述符直喂 · swizzle 落位', mono: true, titleSize: 11, subSize: 8 });
  const tc = node({ id: 'tc', x: 396, y: 206, w: 272, h: 40, role: 'attn', title: 'Tensor Core（wgmma ×8/tile）', sub: 'shared→TC 49152 B（B 读两遍）· 96 B/cycle', mono: true, titleSize: 11, subSize: 8 });
  const reg = node({ id: 'reg', x: 396, y: 274, w: 272, h: 40, role: 'proj', title: 'register 累加器（fp32）', sub: '128 regs/thread · k 方向归约 D 驻留', mono: true, titleSize: 11, subSize: 8 });
  nodes.push(hbm, shm, tc, reg); body.push(hbm.svg, shm.svg, tc.svg, reg.svg);
  [[hbm, shm, 'TMA'], [shm, tc, '描述符'], [tc, reg, 'D=A×B+C']].forEach(([a, b, lab], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.3, delay: 0.4 + i * 0.4, uid: UID, eid: `f${i}`, arrow: false }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
    body.push(cap(672, a.bottom[1] + 16, lab, { mono: true, size: 8, anchor: 'end' }));
  });
  body.push(cap(396, 336, '强度 64 → 735（L2 波前修正，命题 1）', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(396, 352, 'occupancy：shared 128 KB → 1 block/SM 绑定', { size: 9 }));

  body.push(cap(36, 396, '每 tile wgmma = (T_M/64)(T_N/128)(T_K/16) = 2×1×4 = 8 条；8×262144 = 2097152 FLOPs = 2×128×128×64 闭合（命题 2）', { size: 10 }));
  body.push(cap(36, 416, '512 cycles = 279.8 ns @1.83 GHz 与 t_comp 逐位吻合；SS fragment 是 64-bit 描述符不是数据，A/B 本体留 shared', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 432, body, { label: 'CUTLASS 五层实体 device kernel collective CuTe instruction 自顶向下，右侧 tile 数据流 HBM 经 L2 到 shared 到 Tensor Core 到 register 累加器，每 tile 八条 wgmma' }).svg();
}

/* ============================================================
 * 5.2 §5（block 3）— SM90 warp-specialized mainloop 流水时序
 * ============================================================ */
function gemmMainloop() {
  const UID = 'g5e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'SM90 warp-specialized mainloop（Stages=4 · T_K=64）：producer 发 TMA、consumer 发 wgmma，release 滞后 1 tile', { size: 12, color: 'var(--ink-soft)' }));

  // 三泳道时序
  const X = (c) => 150 + c * 116; // stage 列左沿
  body.push(cap(140, 66, 'stage', { size: 9.5, anchor: 'end' }));
  for (let c = 0; c < 4; c++) body.push(cap(X(c) + 56, 66, c === 0 ? 'k' : `k+${c}`, { mono: true, size: 9.5, anchor: 'middle' }));

  // producer 泳道
  body.push(cap(140, 96, 'producer WG0', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(140, 110, '(warp0 elected)', { mono: true, size: 8, anchor: 'end' }));
  for (let c = 0; c < 4; c++) {
    const n = node({ id: `p${c}`, x: X(c), y: 80, w: 110, h: 30, role: 'norm', title: `TMA A+B tile${c}`, mono: true, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  }
  body.push(cap(X(0) + 56, 126, 'acq→2 条 op · expect_tx 32768', { mono: true, size: 8, anchor: 'middle' }));

  // consumer 泳道
  body.push(cap(140, 162, 'consumer WG1', { mono: true, size: 9.5, anchor: 'end' }));
  for (let c = 0; c < 4; c++) {
    const n = node({ id: `c${c}`, x: X(c) + 30, y: 146, w: 110, h: 30, role: 'attn', title: `wgmma×8 tile${c}`, mono: true, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  }
  body.push(cap(X(0) + 86, 192, 'wait→gemm→commit→wait<1>→release', { mono: true, size: 8, anchor: 'middle' }));

  // shared 缓冲环
  body.push(cap(140, 226, 'shared 缓冲', { mono: true, size: 9.5, anchor: 'end' }));
  for (let c = 0; c < 4; c++) {
    body.push(`<rect x="${X(c)}" y="212" width="110" height="26" rx="6" fill="#f5efe2" stroke="#cbb98d" stroke-width="1.2"/>`);
    body.push(cap(X(c) + 56, 229, `stage ${c}`, { mono: true, size: 9, anchor: 'middle', color: 'var(--ink-soft)' }));
  }
  const ring = `M ${X(3) + 110} 225 H ${X(3) + 130} V 250 H ${X(0) - 10} V 225 H ${X(0)}`;
  body.push(`<path class="dia-edge dash" d="${ring}"/>`);
  body.push(pathPulse(ring, { dur: 2.6, delay: 0.8 }));
  body.push(cap(360, 264, '4 份循环复用（union：mainloop 与 epilogue 不同时用 shared）', { size: 9, anchor: 'middle' }));

  // producer→consumer 交接脉冲
  for (let c = 0; c < 3; c++) {
    body.push(edge([X(c) + 56, 110], [X(c) + 86, 146], { nodes, pulse: true, dur: 1.2, delay: 0.3 + c * 0.4, uid: UID, eid: `h${c}`, arrow: false }));
  }

  body.push(cap(36, 292, 'release 时序：warpgroup_wait<1>（L547）先于 consumer_release（L551）—— wgmma 异步读 shared，提前 release 会覆写在读 tile', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 312, 'pipeline 效率（4.4 命题 1）：无 L2 τ=1290.4 ns 装载主导；L2 波前修正 τ=279.8 ns 计算主导 —— 同 kernel 流量假设换档 bound 翻转', { size: 10 }));
  body.push(cap(36, 332, 'L2 修正后 S_min = 1+⌈(261.6+112.3)/279.8⌉ = 3，Stages=4 留一份余量；cluster 2×1 multicast 再省 25%（1.610 GB）', { size: 10 }));

  return scene(720, 348, body, { label: 'SM90 mainloop 三泳道时序：producer 每 stage 发两条 TMA，consumer 错开发八条 wgmma，shared 四份循环，release 滞后一个 tile 防覆写' }).svg();
}

/* ============================================================
 * 5.2 §2（afterHeading）— L2 波前复用：4.61 倍缺口翻转 bound
 * ============================================================ */
function l2Wavefront() {
  const UID = 'g5f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'L2 波前复用：per-block 强度 64 低于 ridge 295，√132≈11.5 倍复用把 GEMM 推进平台区（4.61 倍缺口）', { size: 12, color: 'var(--ink-soft)' }));

  // 波前网格示意（r×c block 共享 A 行 panel / B 列 panel）
  body.push(zone({ x: 36, y: 44, w: 300, h: 230, label: '波前 b=132 block（1 block/SM）' }));
  body.push(cap(52, 84, 'A 行 panel ×r', { mono: true, size: 9 }));
  body.push(cap(200, 84, 'B 列 panel ×c', { mono: true, size: 9 }));
  // 方阵波前网格示意（对角热格 = 同 block 复用 A 行/B 列）
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
    const hot = i === j;
    body.push(`<rect class="${hot ? 'dia-mx-hot' : 'dia-mx-on'}" x="${60 + j * 36}" y="${94 + i * 24}" width="32" height="20" rx="3" opacity="${hot ? 1 : 0.55}"/>`);
  }
  body.push(cap(186, 252, '方阵波前 r=c=√132≈11.49', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(52, 268, 'A tile 行内共享 · B tile 列内共享', { size: 8.5 }));

  // 右：因子对比 + bound 翻转
  const f1 = node({ id: 'f1', x: 360, y: 56, w: 324, h: 52, role: 'data', title: '行优先 raster（c=32 封顶）', sub: '因子 2b/(r+c) = 7.31 · 有效强度 467.7', mono: true, titleSize: 11, subSize: 9 });
  const f2 = node({ id: 'f2', x: 360, y: 122, w: 324, h: 52, role: 'attn', title: '方阵波前（r=c=√132）', sub: '因子 11.49 · 有效强度 735.3 · 流量 186.9 MB', mono: true, titleSize: 11, subSize: 9 });
  nodes.push(f1, f2); body.push(f1.svg, f2.svg);
  body.push(cap(360, 196, '命题 1：缩减因子 2b/(r+c)，方阵波前给 √b', { size: 9.5, color: 'var(--seal-deep)' }));

  // 强度条对 ridge
  body.push(cap(360, 224, '有效强度对 ridge 295（bf16）：', { size: 9.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="360" y="234" width="${64 * 0.32}" height="13" rx="3"/>`);
  body.push(cap(360 + 64 * 0.32 + 6, 244, '64（斜线区）', { mono: true, size: 8.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="360" y="252" width="${735 * 0.32}" height="13" rx="3"/>`);
  body.push(cap(360 + 735 * 0.32 + 6, 262, '735（平台区）', { mono: true, size: 8.5 }));
  body.push(`<line class="dia-edge strong" x1="${360 + 295 * 0.32}" y1="230" x2="${360 + 295 * 0.32}" y2="268"/>`);
  body.push(cap(360 + 295 * 0.32, 280, 'ridge 295', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 300, '带宽口径（方阵波前 + C 写 33.55 MB）：(186.9+33.55) MB ÷ 3352 GB/s = 65.8 μs < 算力地板 138.9 μs → compute-bound', { size: 10 }));
  body.push(cap(36, 320, '工作集核验：波前 panel ≈ 24.1 MB < L2 50 MB（容量侧成立）；raster/swizzle（sm90_tile_scheduler.hpp）是波前形状旋钮', { size: 10 }));
  body.push(cap(36, 340, '坑：拿 per-block 强度 64 断言 memory-bound —— 式 (2)(3) 是「无 L2 复用」上界口径，整 kernel bound 须用波前修正或实测', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 356, body, { label: 'L2 波前复用：132 block 方阵波前共享 A 行 B 列 panel，缩减因子根号 132 约 11.49，有效强度 64 抬到 735 越过 ridge 295，bound 从 memory 翻到 compute' }).svg();
}

/* ============================================================
 * 5.3 §2（block 0）— softmax 三变体 IO + online reduction
 * ============================================================ */
function softmaxVariants() {
  const UID = 'g5g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'softmax 读几遍决定快慢：行驻留达 2n_el·bytes 下界 —— 差的全是搬运，不是 FLOPs', { size: 12, color: 'var(--ink-soft)' }));

  // 三变体读写 strip（全宽行，值右对齐）
  const variants = [
    ['two-pass', '↓x(max) ↓x(sum) ↓x(norm) ↑y', '3 读 1 写', 268.44, 80.08, 'soft'],
    ['online stream', '↓x(max+sum 融合) ↓x(norm) ↑y', '2 读 1 写', 201.33, 60.06, 'warm'],
    ['行驻留', '↓x(load 片上) [max→sum→norm] ↑y', '1 读 1 写', 134.22, 40.04, 'seal'],
  ];
  variants.forEach(([name, io, rw, mb, us, tone], i) => {
    const y = 56 + i * 62;
    const n = node({ id: `v${i}`, x: 36, y, w: 150, h: 48, role: i === 2 ? 'attn' : 'plain', title: name, sub: rw, titleSize: 11.5, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(200, y + 18, io, { mono: true, size: 9.5 }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="200" y="${y + 26}" width="${mb * 1.3}" height="13" rx="3"/>`);
    body.push(cap(684, y + 36, `${mb} MB · ${us} μs`, { mono: true, size: 9, anchor: 'end' }));
  });
  body.push(cap(36, 246, '口径 4096×4096 fp32（n_el·bytes = 67.11 MB）；归约阶段 two-pass 2 读对 online 1 读正是差额来源（命题 1）', { size: 9.5 }));

  // online reduction 机制（底部横条）
  body.push(zone({ x: 36, y: 258, w: 648, h: 64, label: 'online 增量更新（式 2）：换基准因子 ≤1 是恒等变换（精确等价）' }));
  const m = node({ id: 'm', x: 52, y: 282, w: 180, h: 34, role: 'proj', title: 'm = max(m_old, 块 max)', mono: true, titleSize: 9.5 });
  const l = node({ id: 'l', x: 252, y: 282, w: 200, h: 34, role: 'attn', title: 'ℓ = e^{m_old−m}·ℓ_old + Σe^{x−m}', mono: true, titleSize: 8.5 });
  const o = node({ id: 'o', x: 472, y: 282, w: 196, h: 34, role: 'norm', title: 'O = e^{m_old−m}·O_old + Σe^{x−m}·v', mono: true, titleSize: 8.5 });
  nodes.push(m, l, o); body.push(m.svg, l.svg, o.svg);
  body.push(edge(m, l, { nodes, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'ml', arrow: false }));
  body.push(edge(l, o, { nodes, pulse: true, dur: 1.2, delay: 0.8, uid: UID, eid: 'lo', arrow: false }));

  body.push(cap(36, 340, '行驻留条件 n_row·bytes ≤ S_sh：4096 fp32 = 16 KB ≪ 227 KB opt-in；每 block 一行、256 线程各持 16 元素于 register', { size: 10 }));
  body.push(cap(36, 360, 'warp/block 归约映射（5.1）：线程内 16 串行 → shuffle 5 步 → 8 partial 进 shared 32 B → warp0 shuffle 3 步（shared 流量 64 B）', { size: 10 }));
  body.push(cap(36, 380, 'compute 侧：exp 走 SFU 16 results/cycle/SM，地板 4.34 μs ≪ 带宽地板 40.04 μs（9.2× 裕量）→ memory-bound 钉死', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 400, 'exp2 trick（FA2 softmax.h L79/L86）：log2e 折进 scale，exp2f 对齐 SFU 原生 ex2，乘减融成单条 FFMA', { size: 10 }));

  return scene(720, 416, body, { label: 'softmax 三变体 IO：two-pass 三读一写 268 MB，online 两读一写 201 MB，行驻留一读一写 134 MB 达下界；底部 online 增量更新 m ℓ O 换基准恒等' }).svg();
}

/* ============================================================
 * 5.3 §3（afterHeading）— Welford 在线方差 + Chan 并行合并
 * ============================================================ */
function welfordChan() {
  const UID = 'g5h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '方差不能拿 E[x²]−E[x]² 硬算：灾难消去 fp32 算出 −8（真值 1.25）—— Welford 维护中心化 M₂', { size: 12, color: 'var(--ink-soft)' }));

  // 左：朴素 vs Welford
  body.push(zone({ x: 36, y: 44, w: 320, h: 150, label: 'x = [10001,10002,10003,10004]（均值 10002.5）' }));
  const naive = node({ id: 'nv', x: 52, y: 70, w: 288, h: 44, role: 'attn', title: '朴素 E[x²]−E[x]² = −8.0', sub: '两相近大数相减 · ulp 8 ≫ 真差 1.25', titleSize: 11, subSize: 8.5 });
  const welf = node({ id: 'wf', x: 52, y: 126, w: 288, h: 44, role: 'norm', title: 'Welford M₂/c = 1.25 ✓', sub: '累加中心化平方 (x−μ)² · 无大数相减', titleSize: 11, subSize: 8.5 });
  nodes.push(naive, welf); body.push(naive.svg, welf.svg);
  body.push(cap(196, 186, 'E[x²]≈1.0005×10⁸ 落 ulp 8 网格，有效数字整体丢失', { size: 8.5, anchor: 'middle' }));

  // 右：Welford 状态 + Chan 合并
  body.push(zone({ x: 380, y: 44, w: 304, h: 150, label: 'Welford 增量（式 4）+ Chan 合并（式 5）' }));
  body.push(cap(396, 84, 'c←c+1 · δ=x−μ · μ←μ+δ/c', { mono: true, size: 9.5 }));
  body.push(cap(396, 100, 'M₂←M₂+δ·(x−μ_new)（命题 2 不变式）', { mono: true, size: 9.5 }));
  const a = node({ id: 'a', x: 396, y: 112, w: 130, h: 36, role: 'data', title: 'A=[1..4]', sub: 'c=4 μ=2.5 M₂=5', mono: true, titleSize: 9.5, subSize: 7.5 });
  const b = node({ id: 'b', x: 540, y: 112, w: 130, h: 36, role: 'data', title: 'B=[5..8]', sub: 'c=4 μ=6.5 M₂=5', mono: true, titleSize: 9.5, subSize: 7.5 });
  const mg = node({ id: 'mg', x: 448, y: 156, w: 176, h: 34, role: 'attn', title: '合并 μ=4.5 · M₂=42', mono: true, titleSize: 10 });
  nodes.push(a, b, mg); body.push(a.svg, b.svg, mg.svg);
  body.push(edge(a, mg, { nodes, portFrom: a.bottom, portTo: mg.top, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'am', arrow: false }));
  body.push(edge(b, mg, { nodes, portFrom: b.bottom, portTo: mg.top, pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: 'bm', arrow: false }));

  body.push(cap(36, 216, 'Chan 合并：M₂ = M₂,a + M₂,b + (μ_a−μ_b)²·c_a·c_b/c —— 5+5+(2.5−6.5)²×4×4/8 = 42，与 [1..8] 直算逐位相同（命题 3）', { size: 10 }));
  body.push(cap(36, 236, 'TE 归约层级（utils.cuh）：per-warp 基例 two-pass → warp 间 Chan 合并（warp_chan_upd_dynamic L672-701）→ 跨 CTA workspace+barrier', { size: 10 }));
  body.push(cap(36, 256, '数值稳定机制：M₂ 量级是方差本身（小），不是 E[x²]（大）—— 全程无「两个大数相减」，cancellation 无从发生', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 276, '坑：为省一趟归约用朴素公式，大偏移激活下 fp32 算出负方差/0 → rsqrt 得 NaN/Inf、loss 跳变；省归约收益远小于数值崩溃代价', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 292, body, { label: 'Welford 在线方差：朴素公式灾难消去算出负 8 而 Welford 给 1.25，右侧增量状态与 Chan 并行合并 A B 两组得均值 4.5 中心化平方和 42' }).svg();
}

/* ============================================================
 * 5.4 §3（block 0）— 三类融合策略（驻留层 + 代表实体）
 * ============================================================ */
function fusionStrategies() {
  const UID = 'g5i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '三类融合策略：各融不同的东西、各驻留不同的片上层 —— 中间值免 HBM 往返是同一笔账', { size: 12, color: 'var(--ink-soft)' }));

  const cards = [
    ['① 逐元素链', 'pointwise→pointwise→…', '中间值在 register', ['residual+ReLU · SiLU+mul', '量化 cast 链'], ['Inductor/Triton 自动生成'], 'data'],
    ['② reduction+逐元素', '归约→normalize→cast', '整行驻留 register/shared', ['LayerNorm/RMSNorm', '+γ/β+quant'], ['TE kernel（5.3）', '行驻留是前提'], 'norm'],
    ['③ GEMM epilogue', 'mainloop 累加器→bias/act/quant→store', '累加器在 register', ['CUTLASS ThreadEpilogueOp', '（5.2 §7）'], ['D=act(α·acc+β·C+bias)', '一趟写回'], 'attn'],
  ];
  cards.forEach(([name, flow, layer, rep, note, role], i) => {
    const x = 36 + i * 220;
    const n = node({ id: `s${i}`, x, y: 56, w: 204, h: 56, role, title: name, sub: flow, titleSize: 11.5, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    body.push(zone({ x, y: 124, w: 204, h: 130 }));
    body.push(cap(x + 14, 146, layer, { size: 9.5, weight: '650', color: 'var(--ink)' }));
    body.push(cap(x + 14, 170, '代表：', { size: 9 }));
    rep.forEach((t, li) => body.push(cap(x + 14, 186 + li * 14, t, { size: 8.5 })));
    note.forEach((t, li) => body.push(cap(x + 14, 224 + li * 14, t, { size: 8.5, color: 'var(--ink-faint)' })));
    body.push(edge(n, [x + 102, 124], { nodes, portFrom: n.bottom, pulse: true, dur: 1.3, delay: i * 0.35, uid: UID, eid: `s${i}`, arrow: false }));
  });

  body.push(cap(36, 282, 'FlashAttention 是最大融合先例：S,P 从生到死在 SRAM，HBM 只读写 Q/K/V/O —— IO 从 Θ(T²) 降到 O(T²d²_head/M_SRAM)（6.3）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 302, 'dequant 融合进 GEMM mainloop（Marlin W4A16）是更激进的一类（6.2）；三类都受 register/shared 预算约束（命题 2）', { size: 10 }));

  return scene(720, 318, body, { label: '三类融合策略：逐元素链中间值在 register，reduction 加逐元素整行驻留，GEMM epilogue 累加器在 register，各配代表实体' }).svg();
}

/* ============================================================
 * 5.4 §2（afterHeading）— 融合 IO 节省账（k=4 链）+ 预算约束
 * ============================================================ */
function fusionSaving() {
  const UID = 'g5j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '融合省的是中间 tensor 的写+读：k=4 链未融合 8×33.55=268.44 MB，融合后首读尾写 67.11 MB', { size: 12, color: 'var(--ink-soft)' }));

  // 未融合链（4 op 各一读一写）
  body.push(cap(36, 56, '未融合（k=4，每算子一读一写，中间 3 tensor 各写一次读一次）：', { size: 10 }));
  const ops = ['op1', 'op2', 'op3', 'op4'];
  ops.forEach((t, i) => {
    const x = 60 + i * 156;
    const n = node({ id: `o${i}`, x, y: 68, w: 96, h: 34, role: 'plain', title: t, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    if (i < 3) {
      body.push(edge(n, [x + 156, 85], { nodes, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `u${i}`, arrow: false }));
      body.push(cap(x + 126, 78, '↓写↑读', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
      body.push(cap(x + 126, 116, 'HBM 往返', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
    }
  });
  body.push(cap(36, 134, '流量 2k·n_el·bytes = 8×33.55 = 268.44 MB（prefill T=4096 bf16）', { mono: true, size: 9.5 }));

  // 融合链（中间值 register）
  body.push(cap(36, 162, '融合（中间值驻 register，HBM 只剩链首读 + 链尾写）：', { size: 10 }));
  const fz = node({ id: 'fz', x: 60, y: 174, w: 560, h: 40, role: 'attn', title: 'op1 → op2 → op3 → op4（一个 kernel，中间值不落 HBM）', titleSize: 11.5 });
  nodes.push(fz); body.push(fz.svg);
  body.push(edge([40, 194], fz, { nodes, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'in', arrow: false }));
  body.push(cap(40, 186, '↓读', { mono: true, size: 8.5 }));
  body.push(edge(fz, [660, 194], { nodes, pulse: true, dur: 1.4, delay: 0.8, uid: UID, eid: 'out', arrow: false }));
  body.push(cap(648, 186, '↑写', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(36, 232, '流量 2n_el·bytes = 67.11 MB；节省 2(k−1)n_el·bytes = 201.33 MB / 60.06 μs（命题 1）', { mono: true, size: 9.5, color: 'var(--seal-deep)' }));

  // decode vs prefill + 预算
  body.push(cap(36, 262, 'decode B=64（0.5 MiB）：k=4 省 3.15 MB / 0.94 μs，但省 3 次 launch ≈15 μs 是 IO 项 16 倍 → launch 主导', { size: 10 }));
  body.push(cap(36, 282, 'prefill T=4096（33.55 MB）：IO 项 60.06 μs 主导 —— 判断融合值不值先问 tensor 多大、链多长', { size: 10 }));
  body.push(cap(36, 302, '预算约束（命题 2）：R_th≈R₀+w；R₀=32→8 blocks，融合 w=64→R_th=96→2 blocks（occupancy 悬崖）；强压到 64 regs 则 spill 反噬', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 322, '净收益：memory-bound 融合净赚（norm/softmax）；compute-bound 权衡（GEMM epilogue 不占 mainloop TC 时间）；fan-out 使融合非无条件免费', { size: 10 }));

  return scene(720, 338, body, { label: '融合 IO 节省：k 等于 4 链未融合八遍 tensor 流量 268 MB，融合后首读尾写 67 MB 省 201 MB，decode launch 主导 prefill IO 主导，register 预算悬崖' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/01-reduction-scan',
    replacesBlock: 0,
    title: 'reduction tree：每元素恰读一次达下界 —— W=7/D=3，层级拼接 depth≈81',
    svg: reductionTree,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/01-reduction-scan',
    replacesBlock: 1,
    title: 'warp shuffle：down 形 lane0 持和、xor butterfly 全 lane 持和（64 B 对 4084 B）',
    svg: warpShuffle,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/01-reduction-scan',
    replacesBlock: 2,
    title: 'Blelloch scan：up-sweep 子块和 + down-sweep 前缀分发（2 倍 work 换 O(log n) depth）',
    svg: blellochScan,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/02-gemm-kernel-design',
    replacesBlock: 1,
    title: 'CUTLASS 五层实体 + tile 层级数据流：device→kernel→collective→CuTe→wgmma',
    svg: cutlassHierarchy,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/02-gemm-kernel-design',
    replacesBlock: 3,
    title: 'SM90 mainloop 流水：producer TMA / consumer wgmma，release 滞后 1 tile',
    svg: gemmMainloop,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/02-gemm-kernel-design',
    afterHeading: /^## 2\. /,
    title: 'L2 波前复用：√132≈11.49 倍把强度 64 抬到 735，bound 翻转 compute',
    svg: l2Wavefront,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/03-softmax-layernorm-rmsnorm',
    replacesBlock: 0,
    title: 'softmax 三变体 IO：行驻留 1 读 1 写达下界，online 换基准恒等',
    svg: softmaxVariants,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/03-softmax-layernorm-rmsnorm',
    afterHeading: /^## 3\. /,
    title: 'Welford/Chan：朴素方差 −8 对 Welford 1.25，中心化 M₂ 免灾难消去',
    svg: welfordChan,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/04-operator-fusion',
    replacesBlock: 0,
    title: '三类融合策略：逐元素链 / reduction+逐元素 / GEMM epilogue（各驻留层）',
    svg: fusionStrategies,
  },
  {
    dim: 'gpu',
    slug: '05-kernel-compute-primitives/04-operator-fusion',
    afterHeading: /^## 2\. /,
    title: '融合 IO 节省：k=4 省 201.33 MB / 60.06 μs —— (k−1) 个中间 tensor 免落地',
    svg: fusionSaving,
  },
];
