// gpu P6（06-low-precision-attention-kernels）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p01.mjs / gpu-p05.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；sig-cell --i ≤ 2.6。
// census：6.1 block0 位布局 / block1 MX 块；6.2 block0 物化对融合 / block1 Marlin mainloop / block2 lop3 dequant；
//         6.3 block0 FA2 mainloop；6.4 block0 seesaw（依 fed4ca4 修正图：K0/K1 各 9 TMA、V 同源视图、跨 WG 更新）。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

/* ============================================================
 * 6.1 §2（block 0）— 低精度格式位布局（s/e/m 比例 + 范围/ULP）
 * ============================================================ */
function formatBitLayout() {
  const UID = 'g6a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '位布局账：指数位买动态范围、尾数位买相对精度 —— BF16 与 FP16 同 16 bit 分配相反', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(120, 56, '位布局（s/e/m 按位宽比例）', { size: 9.5 }));
  body.push(cap(450, 56, 'bias', { size: 9.5, anchor: 'middle' }));
  body.push(cap(540, 56, '最大有限', { size: 9.5, anchor: 'middle' }));
  body.push(cap(640, 56, 'ULP@1', { size: 9.5, anchor: 'middle' }));

  const fmt = [
    ['FP16', 1, 5, 10, 15, '65504', '2^-10'],
    ['BF16', 1, 8, 7, 127, '3.39e38', '2^-7'],
    ['TF32', 1, 8, 10, 127, '3.40e38', '2^-10'],
    ['E4M3', 1, 4, 3, 7, '448', '2^-3'],
    ['E5M2', 1, 5, 2, 15, '57344', '2^-2'],
    ['E2M1', 1, 2, 1, 1, '6', '2^-1'],
  ];
  fmt.forEach(([name, s, e, m, bias, mx, ulp], i) => {
    const y = 70 + i * 40;
    const total = s + e + m;
    const W = 300, x0 = 120;
    const ws = W * s / total, we = W * e / total, wm = W * m / total;
    body.push(cap(36, y + 4, name, { mono: true, size: 10.5 }));
    body.push(`<rect x="${x0}" y="${y - 12}" width="${ws}" height="20" rx="2" fill="#181b20"/>`);
    body.push(`<rect x="${x0 + ws}" y="${y - 12}" width="${we}" height="20" rx="2" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1"/>`);
    body.push(`<rect x="${x0 + ws + we}" y="${y - 12}" width="${wm}" height="20" rx="2" fill="#f5efe2" stroke="#cbb98d" stroke-width="1"/>`);
    if (we > 26) body.push(cap(x0 + ws + we / 2, y + 2, `e${e}`, { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink-soft)' }));
    if (wm > 26) body.push(cap(x0 + ws + we + wm / 2, y + 2, `m${m}`, { mono: true, size: 8.5, anchor: 'middle', color: 'var(--ink-soft)' }));
    body.push(cap(450, y + 4, String(bias), { mono: true, size: 9.5, anchor: 'middle' }));
    body.push(cap(540, y + 4, mx, { mono: true, size: 9.5, anchor: 'middle' }));
    body.push(cap(640, y + 4, ulp, { mono: true, size: 9.5, anchor: 'middle' }));
  });

  body.push(cap(36, 322, 'BF16 拿 3 个尾数位换指数位：范围 ≈40 → ≈261 binade（与 FP32 同量级），ULP 2^-10 → 2^-7 —— bf16 训练免 loss scaling 的格式根据', { size: 10 }));
  body.push(cap(36, 342, 'E4M3 canonical-NaN 最大 448（误用 IEEE 闭式得 480，把 NaN 码位当有限数）；E2M1 NONE 无特殊值、16 码全有限', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 362, 'E2M1 正数集 {0,0.5,1,1.5,2,3,4,6}，最大相对舍入误差 25%（E4M3 6.25% 的 4 倍）—— FP4 靠细粒度 scale 压误差（§4）', { size: 10 }));

  return scene(720, 378, body, { label: '低精度格式位布局：FP16 BF16 TF32 E4M3 E5M2 E2M1 的符号指数尾数按位宽比例条，配 bias 最大有限数与 ULP，BF16 与 FP16 分配相反' }).svg();
}

/* ============================================================
 * 6.1 §4（block 1）— MX/NVFP4 块 scale 布局 + 粒度权衡
 * ============================================================ */
function mxBlockScale() {
  const UID = 'g6b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '块缩放：每若干元素共享一个 scale 把局部动态范围收进格式窗口 —— 粒度换精度', { size: 12, color: 'var(--ink-soft)' }));

  // MXFP4 块
  body.push(cap(36, 76, 'MXFP4', { mono: true, size: 11 }));
  body.push(cap(130, 56, '1B', { mono: true, size: 8, anchor: 'middle' }));
  body.push(`<rect x="120" y="62" width="20" height="24" rx="3" fill="#f5efe2" stroke="#cbb98d" stroke-width="1.2"/>`);
  body.push(`<rect x="143" y="62" width="320" height="24" rx="3" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1.2"/>`);
  body.push(cap(303, 78, 'E2M1 ×32 = 16B', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--ink-soft)' }));
  body.push(cap(480, 78, '17B/32 = 0.53125 B/元素 · 压缩 3.76×', { size: 9.5 }));
  body.push(cap(112, 78, 'E8M0', { mono: true, size: 8, anchor: 'end' }));

  // NVFP4 块
  body.push(cap(36, 126, 'NVFP4', { mono: true, size: 11 }));
  body.push(cap(130, 106, '1B', { mono: true, size: 8, anchor: 'middle' }));
  body.push(`<rect x="120" y="112" width="20" height="24" rx="3" fill="#f5efe2" stroke="#cbb98d" stroke-width="1.2"/>`);
  body.push(`<rect x="143" y="112" width="160" height="24" rx="3" fill="#ecf0f6" stroke="#aebcd2" stroke-width="1.2"/>`);
  body.push(cap(223, 128, 'E2M1 ×16 = 8B', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--ink-soft)' }));
  body.push(cap(320, 128, '9B/16 = 0.5625 B/元素 · 3.56×', { size: 9.5 }));
  body.push(cap(320, 144, '+ tensor 级 FP32 global scale 4B（摊薄）', { size: 8.5 }));
  body.push(cap(112, 128, 'UE4M3', { mono: true, size: 8, anchor: 'end' }));

  // 粒度-精度权衡
  body.push(cap(36, 176, 'scale 台阶与动态范围利用率（组 absmax a=6.1，E2M1 元素最大 V=6）：', { size: 10 }));
  body.push(cap(120, 197, 'E8M0 台阶 100%', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="130" y="186" width="${50.8 * 3}" height="14" rx="3"/>`);
  body.push(cap(130 + 50.8 * 3 + 6, 197, '50.8%', { mono: true, size: 9 }));
  body.push(cap(120, 221, 'UE4M3 台阶 12.5%', { size: 9, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="130" y="210" width="${90.4 * 3}" height="14" rx="3"/>`);
  body.push(cap(130 + 90.4 * 3 + 6, 221, '90.4%', { mono: true, size: 9 }));

  body.push(cap(36, 248, 'E8M0 纯指数（bias 127）乘 scale = 指数加减、无尾数舍入；UE4M3 = 7-bit 无符号（8-bit 容器 MSB 补零），非 E4M3', { size: 10 }));
  body.push(cap(36, 268, 'NVFP4 用 +5.9% 字节（0.5625/0.53125）买 8 倍细 scale 台阶 + 2 倍细块 —— 压缩比略低（3.56 对 3.76）却精度更稳', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 288, '每元素字节：MXFP8 1.03125（1.94×）· MXFP4 0.53125（3.76×）· NVFP4 0.5625（3.56×）· INT4-g128 0.515625（3.88×）', { size: 10 }));

  return scene(720, 304, body, { label: 'MX 与 NVFP4 块 scale 布局：MXFP4 一字节 E8M0 加 32 个 E2M1，NVFP4 一字节 UE4M3 加 16 个 E2M1，粒度权衡利用率 50.8 对 90.4' }).svg();
}

/* ============================================================
 * 6.1 §5（afterHeading）— 吞吐比 = 指令 k 几何比（datasheet 对账）
 * ============================================================ */
function throughputKRatio() {
  const UID = 'g6c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '两个 2 倍从哪来：吞吐比 = 指令 k 几何比（同 m/n、稳态发射间隔不变）—— 位宽减半 datapath 每周期 2 倍元素', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(36, 56, 'H100：同 m64n128 几何', { size: 10.5 }));
  const b16 = node({ id: 'b16', x: 36, y: 64, w: 170, h: 40, role: 'proj', title: 'BF16 m64n128k16', sub: 'F=262144 · c_SM=4096', mono: true, titleSize: 10, subSize: 8 });
  const f8 = node({ id: 'f8', x: 36, y: 112, w: 170, h: 40, role: 'attn', title: 'FP8 m64n128k32', sub: 'F=524288 · c_SM=8192', mono: true, titleSize: 10, subSize: 8 });
  nodes.push(b16, f8); body.push(b16.svg, f8.svg);
  body.push(`<rect class="dia-bar dia-bar-soft" x="220" y="72" width="${16 * 8}" height="16" rx="3"/>`);
  body.push(cap(220 + 64, 66, 'k16', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="220" y="120" width="${32 * 8}" height="16" rx="3"/>`);
  body.push(cap(220 + 128, 114, 'k32', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(560, 84, 't_issue 64 cyc', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(560, 132, 't_issue 64 cyc', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(648, 108, '2×', { size: 16, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 172, 'B200：BF16 k16 基准', { size: 10.5 }));
  const rows = [
    ['BF16 k16', 16, 'soft', '1×'],
    ['unpacked k32', 32, 'warm', '2×'],
    ['packed MXF4 k64', 64, 'seal', '4×'],
  ];
  rows.forEach(([name, k, tone, ratio], i) => {
    const y = 182 + i * 24;
    body.push(cap(36, y + 12, name, { mono: true, size: 9.5 }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="220" y="${y}" width="${k * 6}" height="15" rx="3"/>`);
    body.push(cap(220 + k * 6 + 8, y + 12, ratio, { mono: true, size: 9.5, color: 'var(--seal-deep)' }));
  });

  body.push(cap(36, 268, 'datasheet 对账（命题 3 一致性）：H100 1978.9/989.4 = 2.0001 ↔ k32/k16；B200 9000/2250 = 4 ↔ UMMA_K 64/16', { size: 10 }));
  body.push(cap(36, 288, '机制：操作数位宽减半，固定宽度乘加 datapath 每周期处理 2 倍元素 → 指令 k 几何翻倍、发射间隔不变即吞吐翻倍', { size: 10 }));
  body.push(cap(36, 308, '坑：FP4 喂进 MXF8F6F4 unpacked 路径只 2 倍（UMMA_K 32）；4 倍须 packed MXF4/NVFP4（UMMA_K 64、K-major）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 324, body, { label: '吞吐比等于指令 k 几何比：H100 BF16 k16 对 FP8 k32 同发射间隔得 2 倍，B200 packed MXF4 k64 得 4 倍，unpacked k32 只 2 倍' }).svg();
}

/* ============================================================
 * 6.2 §2（block 0）— 物化三遍 对 融合一遍（dequant 必须融合）
 * ============================================================ */
function materializeVsFused() {
  const UID = 'g6d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'dequant 必须融合：物化 FP16 中间矩阵多出一整遍 HBM 写+读，比 bf16 基线还慢 2.26 倍', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(36, 52, '物化（materialize）：HBM 三遍', { size: 10.5, color: 'var(--seal-deep)' }));
  const m1 = node({ id: 'm1', x: 36, y: 64, w: 100, h: 40, role: 'data', title: 'INT4 权重', sub: '0.515625 B/权重', titleSize: 10, subSize: 7.5 });
  const m2 = node({ id: 'm2', x: 206, y: 64, w: 110, h: 40, role: 'op', title: 'dequant kernel', sub: '读 INT4 写 FP16', titleSize: 10, subSize: 7.5 });
  const m3 = node({ id: 'm3', x: 386, y: 64, w: 130, h: 40, role: 'cache', title: 'FP16 中间矩阵', sub: '落 HBM（写2B+读2B）', titleSize: 10, subSize: 7.5 });
  const m4 = node({ id: 'm4', x: 586, y: 64, w: 98, h: 40, role: 'plain', title: 'cuBLAS GEMM', sub: '读 FP16 2B', titleSize: 10, subSize: 7.5 });
  nodes.push(m1, m2, m3, m4); body.push(m1.svg, m2.svg, m3.svg, m4.svg);
  [[m1, m2, '读0.515625'], [m2, m3, '写2B'], [m3, m4, '读2B']].forEach(([a, b, lab], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.3, delay: i * 0.3, uid: UID, eid: `m${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
    body.push(cap((a.right[0] + b.left[0]) / 2, 78, lab, { mono: true, size: 7.5, anchor: 'middle' }));
  });

  body.push(cap(36, 132, '融合（fused）：HBM 一遍，FP16 中间值寄存器内生灭', { size: 10.5, color: 'var(--seal-deep)' }));
  const f1 = node({ id: 'f1', x: 36, y: 144, w: 100, h: 40, role: 'data', title: 'INT4 权重', sub: '0.515625 B/权重', titleSize: 10, subSize: 7.5 });
  const f2 = node({ id: 'f2', x: 206, y: 144, w: 280, h: 40, role: 'attn', title: 'GEMM mainloop 寄存器内 dequant', sub: 'lop3 + magic-number（§4）', titleSize: 10.5, subSize: 8 });
  const f3 = node({ id: 'f3', x: 556, y: 144, w: 128, h: 40, role: 'proj', title: 'Tensor Core', sub: 'mma.m16n8k16', titleSize: 10.5, subSize: 8 });
  nodes.push(f1, f2, f3); body.push(f1.svg, f2.svg, f3.svg);
  body.push(edge(f1, f2, { nodes, pulse: true, dur: 1.3, delay: 0.2, uid: UID, eid: 'f0', arrow: false }));
  body.push(arrowHead(f2.left[0], f2.left[1], 1, 0));
  body.push(cap(171, 158, '读0.515625', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(edge(f2, f3, { nodes, pulse: true, dur: 1.3, delay: 0.7, uid: UID, eid: 'f1', arrow: false }));
  body.push(arrowHead(f3.left[0], f3.left[1], 1, 0));

  body.push(cap(36, 210, '脊柱层 [4096,4096]（16.78M 权重）HBM 流量 / 带宽下界时间：', { size: 10 }));
  const bars3 = [
    ['物化 75.7 MB / 22.6 μs', 75.7, 'soft'],
    ['bf16 33.55 MB / 10.0 μs', 33.55, 'warm'],
    ['融合 8.65 MB / 2.58 μs', 8.65, 'seal'],
  ];
  bars3.forEach(([lab, v, tone], i) => {
    const y = 220 + i * 24;
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="36" y="${y}" width="${v * 6}" height="14" rx="3"/>`);
    body.push(cap(36 + v * 6 + 8, y + 11, lab, { mono: true, size: 9 }));
  });

  body.push(cap(36, 302, '物化/融合 = 4.515625/0.515625 = 8.76；物化比 bf16 还慢 2.26× —— 量化本为省带宽，物化反而涨带宽（命题 1）', { size: 10 }));
  body.push(cap(36, 322, '融合对 bf16 加速 10.0/2.58 = 3.88（理想 2/0.515625）；临界 batch≈76，实测近理想只到 16–32（dequant 指令竞争发射槽）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 338, body, { label: '物化对融合：物化读 INT4 写 FP16 再读 FP16 三遍 HBM，融合只读 INT4 一遍寄存器内 dequant，流量 75.7 对 8.65 MB，物化比 bf16 慢 2.26 倍' }).svg();
}

/* ============================================================
 * 6.2 §3（block 1）— Marlin mainloop 数据流（cache policy 分化）
 * ============================================================ */
function marlinMainloop() {
  const UID = 'g6e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Marlin mainloop（STAGES=4 · T_M=16/T_N=128/T_K=128）：B 立即驱逐、A/C 常驻 L2，dequant 在寄存器', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(96, 56, 'HBM', { size: 9.5, anchor: 'middle' }));
  body.push(cap(266, 56, 'shared（4 stage）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(456, 56, 'register', { size: 9.5, anchor: 'middle' }));
  body.push(cap(630, 56, 'Tensor Core', { size: 9.5, anchor: 'middle' }));

  const a1 = node({ id: 'a1', x: 36, y: 70, w: 120, h: 40, role: 'data', title: 'A FP16', sub: '激活', titleSize: 10.5, subSize: 8 });
  const b1 = node({ id: 'b1', x: 36, y: 130, w: 120, h: 40, role: 'data', title: 'B INT4', sub: '权重', titleSize: 10.5, subSize: 8 });
  const s1 = node({ id: 's1', x: 36, y: 190, w: 120, h: 40, role: 'data', title: 'scale', sub: 'per-group', titleSize: 10.5, subSize: 8 });
  const a2 = node({ id: 'a2', x: 196, y: 70, w: 140, h: 40, role: 'cache', title: 'sh_a', sub: 'cp.async.cg', mono: true, titleSize: 10.5, subSize: 8 });
  const b2 = node({ id: 'b2', x: 196, y: 130, w: 140, h: 40, role: 'cache', title: 'sh_b', sub: 'cg + evict_first', mono: true, titleSize: 10.5, subSize: 8 });
  const s2 = node({ id: 's2', x: 196, y: 190, w: 140, h: 40, role: 'cache', title: 'sh_s', sub: 'cp.async 新组时', mono: true, titleSize: 10.5, subSize: 8 });
  const reg = node({ id: 'reg', x: 376, y: 100, w: 160, h: 120, role: 'proj', title: '', titleSize: 10.5 });
  const tc = node({ id: 'tc', x: 576, y: 130, w: 108, h: 60, role: 'attn', title: 'Tensor Core', sub: 'm16n8k16', titleSize: 10.5, subSize: 8 });
  nodes.push(a1, b1, s1, a2, b2, s2, reg, tc);
  body.push(a1.svg, b1.svg, s1.svg, a2.svg, b2.svg, s2.svg, reg.svg, tc.svg);
  body.push(cap(456, 120, 'register fragments', { size: 10.5, anchor: 'middle', weight: '650', color: 'var(--ink)' }));
  body.push(cap(456, 140, 'frag_a（ldmatrix m8n8.x4）', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(456, 158, 'frag_b_quant → lop3/dequant', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(456, 176, '→ FragB · frag_s → hmul2', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(456, 198, '（FP16 中间值不落 HBM）', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));

  [[a1, a2], [b1, b2], [s1, s2]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.3, delay: i * 0.3, uid: UID, eid: `h${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(edge(a2, reg, { nodes, portFrom: a2.right, portTo: reg.top, pulse: true, dur: 1.3, delay: 0.4, uid: UID, eid: 'ra', arrow: false }));
  body.push(edge(b2, reg, { nodes, portFrom: b2.right, portTo: reg.left, pulse: true, dur: 1.3, delay: 0.7, uid: UID, eid: 'rb', arrow: false }));
  body.push(edge(s2, reg, { nodes, portFrom: s2.right, portTo: reg.bottom, pulse: true, dur: 1.3, delay: 1.0, uid: UID, eid: 'rs', arrow: false }));
  body.push(edge(reg, tc, { nodes, pulse: true, dur: 1.3, delay: 1.4, uid: UID, eid: 'rt', arrow: false }));
  body.push(arrowHead(tc.left[0], tc.left[1], 1, 0));
  body.push(cap(556, 152, 'mma', { mono: true, size: 8, anchor: 'middle' }));

  body.push(cap(36, 252, 'B 用 cp.async.cg + evict_first：权重只读一次、不污染 L2（L2 留给 A/C）；A/C 常驻 L2 是命题 2「A-L2 口径」成立的工程手段', { size: 10 }));
  body.push(cap(36, 272, 'A 常驻 L2 强度 62.06 对 A 走 HBM 41.8（比 bf16 3.88 对 2.94）；split-K partial 经 shared 归约（thread_block_reduce L466）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 288, body, { label: 'Marlin mainloop 数据流：A FP16 B INT4 scale 经 cp.async 进 shared 四 stage，寄存器内 ldmatrix 与 lop3 dequant 与 hmul2，喂 Tensor Core，B evict_first A C 常驻 L2' }).svg();
}

/* ============================================================
 * 6.2 §4（block 2）— lop3 + magic-number 指令级 dequant
 * ============================================================ */
function lop3Dequant() {
  const UID = 'g6f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'magic-number dequant：FP16 [1024,2048) ULP=1，nibble OR 进 0x6400 尾数精确 —— 无需 I2F 转换指令', { size: 12, color: 'var(--ink-soft)' }));

  // q nibble strip
  body.push(cap(36, 68, 'q (int32)', { mono: true, size: 10 }));
  for (let i = 0; i < 8; i++) {
    const x = 120 + i * 60;
    body.push(`<rect x="${x}" y="54" width="56" height="22" rx="3" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1"/>`);
    body.push(cap(x + 28, 69, `n${7 - i}`, { mono: true, size: 9.5, anchor: 'middle' }));
  }
  body.push(cap(120, 92, 'LO 取 n0,n4 · HI 取 n1,n5 · q>>8 取 n2,n3,n6,n7（抽取序对齐 reshuffle）', { mono: true, size: 8.5 }));

  // LO 路径
  const l1 = node({ id: 'l1', x: 36, y: 104, w: 180, h: 36, role: 'op', title: 'lop3<0xea>(q,LO,EX)', sub: '(q&LO)|EX', mono: true, titleSize: 9.5, subSize: 7.5 });
  const l2 = node({ id: 'l2', x: 256, y: 104, w: 140, h: 36, role: 'data', title: 'fp16(1024+n)', mono: true, titleSize: 10 });
  const l3 = node({ id: 'l3', x: 436, y: 104, w: 120, h: 36, role: 'op', title: '−1032', sub: 'hsub2', mono: true, titleSize: 10, subSize: 7.5 });
  const l4 = node({ id: 'l4', x: 596, y: 104, w: 88, h: 36, role: 'attn', title: 'n−8', mono: true, titleSize: 11 });
  nodes.push(l1, l2, l3, l4); body.push(l1.svg, l2.svg, l3.svg, l4.svg);
  [[l1, l2], [l2, l3], [l3, l4]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `l${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 156, 'LO 路径', { size: 9, color: 'var(--ink-faint)' }));

  // HI 路径
  const h1 = node({ id: 'h1', x: 36, y: 164, w: 180, h: 36, role: 'op', title: 'lop3<0xea>(q,HI,EX)', sub: '(q&HI)|EX', mono: true, titleSize: 9.5, subSize: 7.5 });
  const h2 = node({ id: 'h2', x: 256, y: 164, w: 140, h: 36, role: 'data', title: 'fp16(1024+16n)', mono: true, titleSize: 10 });
  const h3 = node({ id: 'h3', x: 436, y: 164, w: 120, h: 36, role: 'op', title: '×1/16−72', sub: 'hfma2', mono: true, titleSize: 10, subSize: 7.5 });
  const h4 = node({ id: 'h4', x: 596, y: 164, w: 88, h: 36, role: 'attn', title: 'n−8', mono: true, titleSize: 11 });
  nodes.push(h1, h2, h3, h4); body.push(h1.svg, h2.svg, h3.svg, h4.svg);
  [[h1, h2], [h2, h3], [h3, h4]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: 0.4 + i * 0.3, uid: UID, eid: `h${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 216, 'HI 路径', { size: 9, color: 'var(--ink-faint)' }));

  // 常数账
  const consts = ['EX=0x6400=1024', 'SUB=0x6408=1032', 'MUL=0x2c00=1/16', 'ADD=0xd480=−72'];
  consts.forEach((t, i) => {
    const n = node({ id: `k${i}`, x: 36 + i * 164, y: 230, w: 150, h: 34, role: 'plain', title: t, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
  });

  body.push(cap(36, 288, 'lop3 LUT 0xea=(a&b)|c 一条完成「掩码取 nibble + OR magic」；第二 nibble 用 >>8（非 bfe）；zero-point −8 融进 SUB/ADD', { size: 10 }));
  body.push(cap(36, 308, '每 8 权重 13 条 CUDA-core（2 lop3+1 hsub2+1 hfma2+2 hmul2 ×2 组+>>8）+ 2 mma = 1.625+0.25/权重；全落 INT/FP16 管线无 I2F', { size: 10 }));
  body.push(cap(36, 328, 'B=16 每权重 16 MAC 落 25–50 预算内 → dequant 藏进 mma 执行阴影（m 内循环 L443）；B>25–50 预算耗尽加速塌', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 344, body, { label: 'lop3 magic-number dequant：q 八 nibble 经 LO 与 HI 两路掩码 OR 0x6400 得 fp16 1024+n 与 1024+16n，减 1032 或乘 1/16 减 72 均得 n−8，无需 I2F' }).svg();
}

/* ============================================================
 * 6.3 §4（block 0）— FA2 mainloop tile 流水（Q outer / KV inner + rescale）
 * ============================================================ */
function fa2Mainloop() {
  const UID = 'g6g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'FA2 mainloop：Q outer（seq 维并行）+ KV inner 逆序 —— 两条 GEMM 夹 online softmax，O 驻寄存器到 epilogue', { size: 12, color: 'var(--ink-soft)' }));

  const q = node({ id: 'q', x: 36, y: 70, w: 140, h: 50, role: 'cache', title: 'Q tile（shared）', sub: 'Q outer · 驻留单份', titleSize: 10.5, subSize: 8 });
  const kv = node({ id: 'kv', x: 36, y: 150, w: 140, h: 50, role: 'cache', title: 'K/V tile（shared）', sub: 'KV inner 逆序 · cp.async', titleSize: 10.5, subSize: 8 });
  const sg = node({ id: 'sg', x: 210, y: 70, w: 140, h: 50, role: 'attn', title: '③ S = Q·Kᵀ', sub: 'TC mma.sync m16n8k16', titleSize: 10.5, subSize: 8 });
  const as = node({ id: 'as', x: 210, y: 150, w: 140, h: 50, role: 'proj', title: 'acc_s（fp32 reg）', sub: '+ mask（仅 masking 迭代）', titleSize: 10.5, subSize: 8 });
  const sm = node({ id: 'sm', x: 390, y: 150, w: 150, h: 60, role: 'norm', title: '④ softmax_rescale_o', sub: 'exp2 换基准 · m/ℓ 更新', titleSize: 10.5, subSize: 8 });
  const p = node({ id: 'p', x: 390, y: 230, w: 150, h: 40, role: 'data', title: 'P = bf16(acc_s)', sub: 'C→A layout 重排（reg）', titleSize: 10.5, subSize: 8 });
  const pv = node({ id: 'pv', x: 580, y: 230, w: 104, h: 40, role: 'attn', title: '⑤ O += P·V', sub: 'gemm_rs', titleSize: 10.5, subSize: 8 });
  const ao = node({ id: 'ao', x: 580, y: 150, w: 104, h: 50, role: 'proj', title: 'acc_o（reg）', sub: 'fp32 驻留', titleSize: 10.5, subSize: 8 });
  const ep = node({ id: 'ep', x: 580, y: 70, w: 104, h: 40, role: 'plain', title: 'epilogue', sub: 'O·inv_sum + LSE', titleSize: 10.5, subSize: 8 });
  nodes.push(q, kv, sg, as, sm, p, pv, ao, ep);
  body.push(q.svg, kv.svg, sg.svg, as.svg, sm.svg, p.svg, pv.svg, ao.svg, ep.svg);

  body.push(edge(q, sg, { nodes, pulse: true, dur: 1.3, delay: 0, uid: UID, eid: 'qs', arrow: false }));
  body.push(arrowHead(sg.left[0], sg.left[1], 1, 0));
  body.push(edge(kv, sg, { nodes, portFrom: kv.right, portTo: sg.bottom, pulse: true, dur: 1.3, delay: 0.2, uid: UID, eid: 'ks', arrow: false }));
  body.push(edge(sg, as, { nodes, portFrom: sg.bottom, portTo: as.top, pulse: true, dur: 1.2, delay: 0.5, uid: UID, eid: 'sa', arrow: false }));
  body.push(arrowHead(as.top[0], as.top[1], 0, 1));
  body.push(edge(as, sm, { nodes, pulse: true, dur: 1.2, delay: 0.8, uid: UID, eid: 'asm', arrow: false }));
  body.push(arrowHead(sm.left[0], sm.left[1], 1, 0));
  body.push(edge(sm, p, { nodes, portFrom: sm.bottom, portTo: p.top, pulse: true, dur: 1.2, delay: 1.1, uid: UID, eid: 'smp', arrow: false }));
  body.push(arrowHead(p.top[0], p.top[1], 0, 1));
  body.push(edge(p, pv, { nodes, pulse: true, dur: 1.2, delay: 1.4, uid: UID, eid: 'ppv', arrow: false }));
  body.push(arrowHead(pv.left[0], pv.left[1], 1, 0));
  body.push(edge(pv, ao, { nodes, portFrom: pv.top, portTo: ao.bottom, pulse: true, dur: 1.2, delay: 1.7, uid: UID, eid: 'pao', arrow: false }));
  body.push(arrowHead(ao.bottom[0], ao.bottom[1], 0, 1));
  body.push(edge(ao, ep, { nodes, portFrom: ao.top, portTo: ep.bottom, pulse: true, dur: 1.2, delay: 2.0, uid: UID, eid: 'aep', arrow: false }));
  body.push(arrowHead(ep.bottom[0], ep.bottom[1], 0, 1));
  // rescale 反馈：acc_o ← softmax（换基准乘）
  body.push(edge(ao, sm, { nodes, portFrom: ao.left, portTo: sm.right, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'resc', arrow: false }));
  body.push(arrowHead(sm.right[0], sm.right[1], -1, 0));
  body.push(cap(560, 168, 'rescale', { mono: true, size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  // V 转置供 PV：kv → pv 走底部 L 形
  const vpath = 'M 106 200 V 292 H 632 V 270';
  body.push(`<path class="dia-edge dash" d="${vpath}"/>`);
  body.push(pathPulse(vpath, { dur: 2.4, delay: 0.6 }));
  body.push(arrowHead(632, 270, 0, -1));
  body.push(cap(350, 286, 'V ldmatrix.trans（硬件转置，sVt 同址视图）', { mono: true, size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 322, 'Q outer：grid = n_m×B×H_q = 32×1×32 = 1024 CTA（seq 维变并行维）；KV inner 逆序（末块唯一需 mask，masking 2 迭代）', { size: 10 }));
  body.push(cap(36, 342, 'gemm_rs：P 寄存器源（C→A layout 零移动）；warp 4 沿 M 各 16 行，行统计量不出 warp —— quad allreduce 2 条 shfl 免 shared', { size: 10 }));
  body.push(cap(36, 362, 'non-matmul 削减：sum 跨线程归约延后到 epilogue；终除变倒数乘；exp2 折 log2e 单条 FFMA + MUFU.EX2', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 378, body, { label: 'FA2 mainloop 环形数据流：Q 与 KV shared 喂 S GEMM 得 acc_s，softmax rescale 换基准，P 寄存器源喂 gemm_rs 累加 acc_o，rescale 反馈回 acc_o，epilogue 倒数乘加 LSE' }).svg();
}

/* ============================================================
 * 6.3 §3（afterHeading）— tile 预算：shared 松 / register 紧
 * ============================================================ */
function faTileBudget() {
  const UID = 'g6h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'tile 上界由 shared 与 register 两预算取 min —— FA2 是 register 绑定（252/255），不是 shared', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 320, h: 120, label: 'shared 预算（式 4/5）' }));
  body.push(`<rect x="52" y="80" width="288" height="18" rx="4" fill="none" stroke="var(--rule)" stroke-width="1.2"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="52" y="80" width="${64 / 227 * 288}" height="18" rx="4"/>`);
  body.push(cap(52, 116, '64 KB / 227 KB（28%）—— 松', { size: 9.5 }));
  body.push(cap(52, 134, '(T_M+2T_N)·d·bytes = (128+128)·128·2', { mono: true, size: 8.5 }));
  body.push(cap(52, 152, 'FA3 multi-stage：160 KB（n_stage=2）', { mono: true, size: 8.5 }));

  body.push(zone({ x: 380, y: 44, w: 304, h: 120, label: 'register 预算（式 6）' }));
  body.push(`<rect x="396" y="80" width="272" height="18" rx="4" fill="none" stroke="var(--rule)" stroke-width="1.2"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="396" y="80" width="${252 / 255 * 272}" height="18" rx="4"/>`);
  body.push(cap(396, 116, '252 / 255（余 3）—— 紧', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(396, 134, '192(O+S) + 32(P) + 28(aux)', { mono: true, size: 8.5 }));
  body.push(cap(396, 152, 'T_M(d+1.5T_N)/n_th + R_aux ≤ 255', { mono: true, size: 8.5 }));

  const tm = node({ id: 'tm', x: 250, y: 190, w: 220, h: 50, role: 'attn', title: 'T_M ≤ min(780, 129.7) = 128', sub: 'register 绑定 · 64 倍数（atom 铺砌）', mono: true, titleSize: 11, subSize: 8.5 });
  nodes.push(tm); body.push(tm.svg);
  body.push(edge([196, 164], tm, { nodes, portTo: tm.left, pulse: true, dur: 1.3, delay: 0.3, uid: UID, eid: 'sh', arrow: false }));
  body.push(edge([532, 164], tm, { nodes, portTo: tm.right, strong: true, pulse: true, dur: 1.3, delay: 0.6, uid: UID, eid: 'rg', arrow: false }));
  body.push(arrowHead(tm.right[0], tm.right[1], -1, 0));

  body.push(cap(36, 262, 'K/V 分计（系数 2，kSmemKVSize *2）；sVt 是 layout 视图不复制第二份 V；epilogue O 复用 Q 缓冲（预算不加项）', { size: 10 }));
  body.push(cap(36, 282, '下一档 T_M=192 需 364 regs 超限 43%；occupancy 层 register 项 2 blocks/SM（O=0.125），低 occupancy 靠 cp.async+ILP 藏延迟', { size: 10 }));
  body.push(cap(36, 302, 'FA3 换解法：2 MMA WG 沿 M 切分各 64 行 → 160 regs/WG；setmaxnreg 24/240 再分配，1 block/SM 寄存器+shared 双绑定', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 318, body, { label: 'FA tile 预算：shared 64KB 占 227KB 松，register 252 占 255 紧，T_M 上界取 min 780 与 129.7 得 128 register 绑定' }).svg();
}

/* ============================================================
 * 6.3 §5（afterHeading）— FA3 warp specialization + pingpong + SFU/TC
 * ============================================================ */
function fa3Pingpong() {
  const UID = 'g6i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'FA3：WG0 producer（TMA）+ WG1/WG2 consumer —— named barrier token 环强制 TC 相位交替（pingpong）', { size: 12, color: 'var(--ink-soft)' }));

  // producer 泳道
  body.push(cap(140, 82, 'WG0 load', { mono: true, size: 9, anchor: 'end' }));
  for (let c = 0; c < 4; c++) {
    const n = node({ id: `t${c}`, x: 150 + c * 128, y: 66, w: 120, h: 26, role: 'norm', title: `TMA K/V k${c}`, mono: true, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  }
  // WG1 / WG2 交错泳道
  body.push(cap(140, 122, 'WG1 MMA', { mono: true, size: 9, anchor: 'end' }));
  body.push(cap(140, 162, 'WG2 MMA', { mono: true, size: 9, anchor: 'end' }));
  const w1 = [['TC', 'attn'], ['SFU', 'norm'], ['TC', 'attn'], ['SFU', 'norm']];
  const w2 = [['SFU', 'norm'], ['TC', 'attn'], ['SFU', 'norm'], ['TC', 'attn']];
  w1.forEach(([t, role], c) => {
    const n = node({ id: `a${c}`, x: 150 + c * 128, y: 106, w: 120, h: 26, role, title: `${t}(n${c >> 1})`, mono: true, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  });
  w2.forEach(([t, role], c) => {
    const n = node({ id: `b${c}`, x: 150 + c * 128, y: 146, w: 120, h: 26, role, title: `${t}(n${c >> 1})`, mono: true, titleSize: 9 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(360, 190, '同列 = 同时刻：WG1 做 SFU(softmax) 时 WG2 占 TC，反之亦然 —— token 环 warp_scheduler_barrier_sync/arrive', { size: 9, anchor: 'middle' }));
  body.push(cap(360, 206, 'IntraWGOverlap：fwd_step 发 S-GEMM(n)+PV-GEMM(n+1)（wg_wait=-1）后做 softmax(n) —— WG 内再藏一层', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 232, '每 KV tile（T_M=T_N=128）：SFU 559.6 ns（16384 exp2 ÷16/cycle）对 TC 1119 ns（S+PV）= 50%', { size: 10 }));
  body.push(`<rect class="dia-bar dia-bar-norm" x="36" y="242" width="${559.6 * 0.4}" height="14" rx="3" fill="#a9bfa9"/>`);
  body.push(cap(36 + 559.6 * 0.4 + 8, 253, 'SFU 559.6 ns', { mono: true, size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="36" y="260" width="${1119 * 0.4}" height="14" rx="3"/>`);
  body.push(cap(36 + 1119 * 0.4 + 8, 271, 'TC 1119 ns', { mono: true, size: 9 }));
  body.push(cap(36, 292, '串行天花板 1119/(1119+559.6) = 66.7% < 报告 74.8%（740/989.5）→ 重叠是必需不是打磨；不重叠 util 天花板就是 2/3', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 312, 'TMA EVICT_LAST（KV 跨 CTA 重读）对照 FlashMLA EVICT_FIRST（无复用流式）—— cache hint 跟复用结构走，不是固定配方', { size: 10 }));

  return scene(720, 328, body, { label: 'FA3 pingpong：WG0 TMA producer，WG1 WG2 两 consumer 泳道 TC 与 SFU 相位交错，named barrier token 环；SFU 559.6 ns 对 TC 1119 ns 串行天花板 66.7 低于报告 74.8' }).svg();
}

/* ============================================================
 * 6.4 §2（block 0）— MLA seesaw 双 WG 数据流（依修正图）
 * ============================================================ */
function mlaSeesaw() {
  const UID = 'g6j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'MLA seesaw：单输出竖切 O_L/O_R + 双 KV block 跷跷板 —— FA3 双输出 pingpong 在 d_v=512 放不下', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(36, 48, '输出 64×512 fp32 = 32768 regs = R_file/2 → 每 SM 仅一份输出；竖切 O_L/O_R 各 64×256 = 16384 regs，两 WG 各持一半', { size: 10, color: 'var(--seal-deep)' }));

  const k0 = node({ id: 'k0', x: 36, y: 64, w: 150, h: 40, role: 'cache', title: 'K0（64×576）', sub: '9 份 TMA 逐份启动 GEMM', titleSize: 10.5, subSize: 8 });
  const k1 = node({ id: 'k1', x: 216, y: 64, w: 150, h: 40, role: 'cache', title: 'K1（64×576）', sub: '9 份 TMA（576/64=9）', titleSize: 10.5, subSize: 8 });
  const m = node({ id: 'm', x: 420, y: 64, w: 120, h: 40, role: 'attn', title: 'running max m', sub: '两 WG 共享', titleSize: 10.5, subSize: 8 });
  nodes.push(k0, k1, m); body.push(k0.svg, k1.svg, m.svg);
  body.push(cap(560, 80, 'V0/V1 = K 同源', { size: 8.5 }));
  body.push(cap(560, 94, 'latent 512 列视图（8×64）', { size: 8.5 }));
  body.push(cap(560, 108, '不另装载 · 竖切 V0L/V0R…', { size: 8.5 }));

  body.push(zone({ x: 36, y: 120, w: 300, h: 100 }));
  body.push(cap(52, 214, 'WG0（持 O_L 64×256）', { mono: true, size: 10.5, color: 'var(--ink-faint)' }));
  const c0 = node({ id: 'c0', x: 52, y: 146, w: 130, h: 40, role: 'proj', title: 'p0 = q·K0ᵀ', sub: 'exp(p0−m_new0)', mono: true, titleSize: 10, subSize: 8 });
  const c1 = node({ id: 'c1', x: 196, y: 146, w: 124, h: 40, role: 'attn', title: 'O_L 更新', sub: '+p0·V0L · rescale(s0)', titleSize: 10, subSize: 7.5 });
  nodes.push(c0, c1); body.push(c0.svg, c1.svg);
  body.push(edge(c0, c1, { nodes, pulse: true, dur: 1.2, delay: 0.4, uid: UID, eid: 'c', arrow: false }));
  body.push(arrowHead(c1.left[0], c1.left[1], 1, 0));

  body.push(zone({ x: 384, y: 120, w: 300, h: 100 }));
  body.push(cap(400, 214, 'WG1（持 O_R 64×256）', { mono: true, size: 10.5, color: 'var(--ink-faint)' }));
  const d0 = node({ id: 'd0', x: 400, y: 146, w: 130, h: 40, role: 'proj', title: 'p1 = q·K1ᵀ', sub: 'exp(p1−m_new1)', mono: true, titleSize: 10, subSize: 8 });
  const d1 = node({ id: 'd1', x: 544, y: 146, w: 124, h: 40, role: 'attn', title: 'O_R 更新', sub: '+p1·V1R · rescale(s0·s1)', titleSize: 10, subSize: 7.5 });
  nodes.push(d0, d1); body.push(d0.svg, d1.svg);
  body.push(edge(d0, d1, { nodes, pulse: true, dur: 1.2, delay: 0.7, uid: UID, eid: 'd', arrow: false }));
  body.push(arrowHead(d1.left[0], d1.left[1], 1, 0));

  body.push(edge(k0, c0, { nodes, portFrom: k0.bottom, portTo: c0.top, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'kc', arrow: false }));
  body.push(edge(k1, d0, { nodes, portFrom: k1.bottom, portTo: d0.top, pulse: true, dur: 1.2, delay: 0.2, uid: UID, eid: 'kd', arrow: false }));
  body.push(edge(m, c0, { nodes, portFrom: m.bottom, portTo: c0.top, dash: true, dur: 1.4, delay: 0.5, uid: UID, eid: 'mc', arrow: false }));
  body.push(edge(m, d0, { nodes, portFrom: m.bottom, portTo: d0.top, dash: true, pulse: true, dur: 1.4, delay: 0.8, uid: UID, eid: 'md', arrow: false }));

  // 跨 WG 更新
  body.push(zone({ x: 36, y: 240, w: 648, h: 80, label: '跨 WG 更新（seesaw 核心：用对方的 p 补自己半区）' }));
  const p0 = node({ id: 'p0', x: 60, y: 268, w: 70, h: 30, role: 'proj', title: 'p0', mono: true, titleSize: 10 });
  const or = node({ id: 'or', x: 200, y: 268, w: 80, h: 30, role: 'attn', title: 'O_R', mono: true, titleSize: 10 });
  const p1 = node({ id: 'p1', x: 400, y: 268, w: 70, h: 30, role: 'proj', title: 'p1', mono: true, titleSize: 10 });
  const ol = node({ id: 'ol', x: 540, y: 268, w: 80, h: 30, role: 'attn', title: 'O_L', mono: true, titleSize: 10 });
  nodes.push(p0, or, p1, ol); body.push(p0.svg, or.svg, p1.svg, ol.svg);
  body.push(edge(p0, or, { nodes, pulse: true, dur: 1.2, delay: 1.2, uid: UID, eid: 'x1', arrow: false }));
  body.push(arrowHead(or.left[0], or.left[1], 1, 0));
  body.push(cap(165, 312, '·V0R（WG1 用 WG0 p0）', { size: 8, anchor: 'middle' }));
  body.push(edge(p1, ol, { nodes, pulse: true, dur: 1.2, delay: 1.5, uid: UID, eid: 'x2', arrow: false }));
  body.push(arrowHead(ol.left[0], ol.left[1], 1, 0));
  body.push(cap(505, 312, '·V1L（WG0 用 WG1 p1）', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 342, '重叠：WG0 做 softmax/rescale（CUDA/SFU）时 WG1 占 TC，反之亦然；半区 rescale 因子 = 自上次更新以来全部 max 位移累积（O_R 为两 scale 之积）', { size: 10 }));
  body.push(cap(36, 362, '正确性 = online softmax 不变量按半区维护（deep-dive 逐字：mathematically equivalent to FA online softmax）', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 382, 'EVICT_FIRST（decode 每 KV block 单 CTA 消费、无跨 CTA 复用）对照 FA3 EVICT_LAST —— cache hint 跟复用结构走', { size: 10 }));

  return scene(720, 398, body, { label: 'MLA seesaw 双 warpgroup：K0 K1 各九份 TMA，WG0 算 p0 更新 O_L，WG1 算 p1 更新 O_R，共享 running max，跨 WG 用对方 p 补半区 O_R 加 p0 V0R 与 O_L 加 p1 V1L' }).svg();
}

/* ============================================================
 * 6.4 §4（afterHeading）— paged attention：block table gather
 * ============================================================ */
function pagedGather() {
  const UID = 'g6k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'paged gather：逻辑连续不变、物理散落经块表翻译 —— 代价在页表读取与跳转粒度，不在页内', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(36, 52, 'slot(t) = bt[⌊t/b⌋]·b + r(t)（r = t mod b）', { mono: true, size: 10.5 }));

  const log = [0, 1, 2].map((i) => {
    const n = node({ id: `lg${i}`, x: 36 + i * 220, y: 64, w: 200, h: 30, role: 'data', title: `逻辑块${i}（t ${i * 64}..${i * 64 + 63}）`, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const bt = [0, 2, 1].map((p, i) => {
    const n = node({ id: `bt${i}`, x: 36 + i * 220, y: 120, w: 200, h: 30, role: 'op', title: `bt[${i}] = ${p}`, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const phys = [0, 2, 1].map((p, i) => {
    const n = node({ id: `ph${p}`, x: 36 + i * 220, y: 180, w: 200, h: 40, role: 'cache', title: `物理页${p}`, sub: 'b=64 token×d_head 连续 · 16B 满 sector', titleSize: 9.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg); return n;
  });
  for (let i = 0; i < 3; i++) {
    body.push(edge(log[i], bt[i], { nodes, portFrom: log[i].bottom, portTo: bt[i].top, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `lb${i}`, arrow: false }));
    body.push(arrowHead(bt[i].top[0], bt[i].top[1], 0, 1));
    body.push(edge(bt[i], phys[i], { nodes, portFrom: bt[i].bottom, portTo: phys[i].top, pulse: true, dur: 1.2, delay: 0.4 + i * 0.3, uid: UID, eid: `bp${i}`, arrow: false }));
    body.push(arrowHead(phys[i].top[0], phys[i].top[1], 0, 1));
  }
  body.push(cap(360, 236, '逻辑序 0,1,2 → 物理序 0,2,1（散落）；页内连续满 sector、页间跳转是新的一段连续流', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 262, 'b=64/H_kv=8/d_head=128/bf16：一页 128 KiB（NHD 全头），单 head slice 16 KiB（HND）；块表 T/b int32 = 256 B 驻 L2 无压力', { size: 10 }));
  body.push(cap(36, 282, 'TMA per-page（页尺寸对齐）对 cp.async gather（PagedKVNonTMA → PipelineAsync 线程 arrive 计数）；页尺寸任意退 gather', { size: 10 }));
  body.push(cap(36, 302, 'PagedKV 地址状态吃寄存器 → FP8 headdim>192 禁 IntraWGOverlap（tile_size.h L51）—— 6.3 命题 1 预算在 paged 路径更紧', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 318, body, { label: 'paged gather：逻辑块经块表 bt 翻译到散落物理页，页内连续满 sector 页间跳转，TMA per-page 对 cp.async gather 两路' }).svg();
}

/* ============================================================
 * 6.4 §6（afterHeading）— split-KV merge（base-2 LSE）+ scheduler
 * ============================================================ */
function splitKvMerge() {
  const UID = 'g6l';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'split-KV combine：各段 partial (O_i, LSE_i) 经 LSE 加权归约拼回完整输出 —— ⊕ 结合交换任意树形等价', { size: 12, color: 'var(--ink-soft)' }));

  const parts = [0, 1, 2, 3].map((i) => {
    const t = i === 3 ? '… (O_{S−1}, LSE_{S−1})' : `(O_${i}, LSE_${i})`;
    const n = node({ id: `p${i}`, x: 36 + i * 164, y: 64, w: 150, h: 40, role: 'data', title: t, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg); return n;
  });
  const m01 = node({ id: 'm01', x: 118, y: 130, w: 140, h: 36, role: 'op', title: 'merge ⊕', mono: true, titleSize: 10 });
  const m23 = node({ id: 'm23', x: 446, y: 130, w: 140, h: 36, role: 'op', title: 'merge ⊕', mono: true, titleSize: 10 });
  const mf = node({ id: 'mf', x: 282, y: 196, w: 160, h: 44, role: 'attn', title: 'O = Σw_i·O_i / Σw_i', sub: 'LSE = log2(Σw) + m', mono: true, titleSize: 10, subSize: 8 });
  nodes.push(m01, m23, mf); body.push(m01.svg, m23.svg, mf.svg);
  [[parts[0], m01], [parts[1], m01], [parts[2], m23], [parts[3], m23]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: i * 0.2, uid: UID, eid: `pm${i}`, arrow: false }));
  });
  [[m01, mf], [m23, mf]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: 0.8 + i * 0.2, uid: UID, eid: `mf${i}`, arrow: false }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });

  body.push(cap(36, 262, 'base-2：w_i = 2^{LSE_i−m}（m = max LSE）；ptx_exp2/ptx_log2 原生 SFU（与 FA2 exp2 同构）；vec 化 d_head 次 FMA', { size: 10 }));
  body.push(cap(36, 282, '空态守卫：差 clamp 到 −inf（exp2 得 0）+ d_sum 守卫把 0/0 兜成 0 —— 两空 split 合并免 NaN，无特判分支', { size: 10 }));
  body.push(cap(36, 302, 'scheduler 二分：n_new = Σ⌈T_i/(b·c)⌉，n_new·H_kv ≤ G_max（occupancy 实测）；例 c=5 页→13 段×8=104≤112，c=4→128 超限 → S=13', { size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(36, 322, 'cascade 两级：共享前缀状态 + 独立后缀状态各算一遍再 MergeState —— 前缀 KV 读从 B 次降到每 query tile 一次；η=2.76%（T=4096）', { size: 10 }));

  return scene(720, 338, body, { label: 'split-KV combine：S 段 partial O 与 LSE 两两 merge 成归约树得最终 O，base-2 LSE 加权，空态守卫，scheduler 二分约束 n_new 乘 H_kv 不超 G_max' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/01-low-precision-formats',
    replacesBlock: 0,
    title: '低精度格式位布局：s/e/m 比例 + 范围/ULP —— BF16 与 FP16 分配相反',
    svg: formatBitLayout,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/01-low-precision-formats',
    replacesBlock: 1,
    title: 'MX/NVFP4 块 scale：E8M0 对 UE4M3 粒度权衡（利用率 50.8% 对 90.4%）',
    svg: mxBlockScale,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/01-low-precision-formats',
    afterHeading: /^## 5\. /,
    title: '吞吐比 = 指令 k 几何比：H100 2×（k32/k16）、B200 4×（UMMA_K 64/16）',
    svg: throughputKRatio,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/02-quantization-kernels',
    replacesBlock: 0,
    title: '物化三遍 对 融合一遍：dequant 必须融合（物化比 bf16 慢 2.26×）',
    svg: materializeVsFused,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/02-quantization-kernels',
    replacesBlock: 1,
    title: 'Marlin mainloop：B evict_first / A/C 常驻 L2，dequant 在寄存器',
    svg: marlinMainloop,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/02-quantization-kernels',
    replacesBlock: 2,
    title: 'lop3 + magic-number：nibble OR 0x6400 尾数精确还原 n−8，无 I2F',
    svg: lop3Dequant,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/03-flashattention-kernel',
    replacesBlock: 0,
    title: 'FA2 mainloop：Q outer / KV inner 逆序 + 两 GEMM 夹 online rescale',
    svg: fa2Mainloop,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/03-flashattention-kernel',
    afterHeading: /^## 3\. /,
    title: 'tile 预算：shared 松（64/227 KB）/ register 紧（252/255）→ T_M=128',
    svg: faTileBudget,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/03-flashattention-kernel',
    afterHeading: /^## 5\. /,
    title: 'FA3 pingpong：WG 间 TC/SFU 相位交错 —— 串行天花板 66.7% < 74.8%',
    svg: fa3Pingpong,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/04-attention-variant-kernels',
    replacesBlock: 0,
    title: 'MLA seesaw：单输出竖切 O_L/O_R + 双 KV block 跨 WG 更新（K 9 份 TMA）',
    svg: mlaSeesaw,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/04-attention-variant-kernels',
    afterHeading: /^## 4\. /,
    title: 'paged gather：块表翻译逻辑→物理散落页，页内连续 / 页间跳转',
    svg: pagedGather,
  },
  {
    dim: 'gpu',
    slug: '06-low-precision-attention-kernels/04-attention-variant-kernels',
    afterHeading: /^## 6\. /,
    title: 'split-KV merge：base-2 LSE 加权归约树 + scheduler 二分 n_new·H_kv≤G_max',
    svg: splitKvMerge,
  },
];
