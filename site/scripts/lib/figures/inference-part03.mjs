// inference Part 3（03-inference-operators）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs / inference-part02.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言伪代码围栏不画（管线自动算法面板）。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性（类选择器 > 表现属性）——
// 显式传 color 时补 inline style 提权，保证朱砂强调 / 深底白字真正生效（不改 kit API）。
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
 * 3.1 §2（block 0）— split-KV 数据流（B=1、H_kv=8、S=14；8B 口径）
 * 原文：T=131072 切 14 段（view）→ 112 CTA ≥ 108 SM → 段内 FA 产 (O_i,LSE_i)
 *       → ⊕ 归约（式 3）；M_red ≈ 451.5 KiB/层、η ≈ 0.086%；数值例逐位相同
 * ============================================================ */
function splitKvFlow() {
  const UID = 'i3a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'split-KV：把 KV 维变成并行维（B=1、H_kv=8、S=14；8B 口径）—— 112 CTA 铺满 A100 108 SM', { size: 12, color: 'var(--ink-soft)' }));

  // KV 长条：14 段 view
  body.push(zone({ x: 36, y: 40, w: 648, h: 78, label: 'KV 序列 T = 131072（逻辑连续 · 物理按块散落 2.1）—— 切段 = view，无数据移动' }));
  const segW = 42, segX = 56;
  for (let i = 0; i < 14; i++) {
    body.push(`<rect x="${segX + i * (segW + 2)}" y="74" width="${segW}" height="20" rx="4" fill="${i === 0 || i === 1 || i === 13 ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${i === 0 || i === 1 || i === 13 ? 'var(--seal)' : '#c6c0af'}" stroke-width="1"/>`);
  }
  body.push(cap(segX + 21, 88, '1', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(segX + 65, 88, '2', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(segX + 13 * 44 + 21, 88, '14', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(segX + 7 * 44, 88, '⋯', { size: 10, anchor: 'middle' }));
  body.push(cap(56, 110, '每段 ⌈131072/14⌉ = 9363 token · 每段 × 8 KV head → n_CTA = B·H_kv·S = 112（式 1）', { mono: true, size: 9.5 }));

  // 三个 split 卡（1 / 2 / 14）+ 省略
  const mk = (id, x, title, sub) => {
    const n = node({ id, x, y: 140, w: 158, h: 56, role: 'attn', title, sub, mono: true, titleSize: 12, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const s1 = mk('s1', 52, 'split 1 · 段内 FA', 'online softmax');
  const s2 = mk('s2', 236, 'split 2 · 段内 FA', 'online softmax');
  const s14 = mk('s14', 510, 'split 14 · 段内 FA', 'online softmax');
  body.push(cap(452, 172, '⋯', { size: 14, anchor: 'middle' }));
  [[s1, 0], [s2, 0.25], [s14, 0.5]].forEach(([s, dl], i) => {
    const ax = [77, 121, 641][i];
    body.push(edge([ax, 94], [s.x + 79, 140], { pulse: true, dur: 1.3, delay: dl, uid: UID, eid: `k${i}` }));
    body.push(arrowHead(s.x + 79, 140, 0, 1));
  });

  // partial workspace
  const ws = [52, 236, 510].map((x, i) => {
    const n = node({ id: `w${i}`, x, y: 226, w: 158, h: 34, role: 'cache', title: `(O_${[1, 2, 14][i]}, LSE_${[1, 2, 14][i]})`, mono: true, titleSize: 11.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  [s1, s2, s14].forEach((s, i) => {
    body.push(edge(s, ws[i], { nodes, portFrom: s.bottom, portTo: ws[i].top, pulse: true, dur: 1, delay: 0.7 + i * 0.2, uid: UID, eid: `w${i}` }));
    body.push(arrowHead(ws[i].top[0], ws[i].top[1], 0, 1));
  });
  body.push(cap(452, 247, '⋯', { size: 12, anchor: 'middle' }));

  // reduction → 输出
  const R = node({ id: 'R', x: 180, y: 296, w: 250, h: 52, role: 'op', title: 'reduction kernel：⊕ 归约（式 3）', sub: 'm = max(LSEᵢ, LSEⱼ) · w = exp(LSE − m)', titleSize: 12.5, subSize: 9 });
  nodes.push(R); body.push(R.svg);
  const O = node({ id: 'O', x: 510, y: 296, w: 158, h: 52, role: 'attn', title: '(O, LSE)', sub: '与不切逐位相同（命题 1）', mono: true, titleSize: 13, subSize: 9 });
  nodes.push(O); body.push(O.svg);
  ws.forEach((w, i) => {
    body.push(edge(w, [230 + i * 75, 296], { nodes, portFrom: w.bottom, pulse: true, dur: 1.1, delay: 1.3 + i * 0.2, uid: UID, eid: `r${i}` }));
    body.push(arrowHead(230 + i * 75, 296, 0, 1));
  });
  body.push(edge(R, O, { nodes, pulse: true, dur: 1.2, delay: 2.0, uid: UID, eid: 'ro' }));
  body.push(arrowHead(O.left[0], O.left[1], 1, 0));

  body.push(cap(360, 362, 'partial 写 workspace —— LSE 一律 fp32（bf16 使权重失真 · 破坏 batch invariance）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(36, 374, '数值例 x=(1,2,5,−1)：LSE₁=2.3133 / LSE₂=5.0025 → w₁=0.0679、w₂=1 → O=(1.8853, 0.9783) —— 与全行 softmax 逐位相同', { mono: true, size: 9.5 }));
  body.push(cap(36, 392, '归约账（式 4）：M_red ≈ 2×14×32×516 = 451.5 KiB/层，占 KV 读 512 MiB 的 η ≈ 0.086% —— 归约是小头，并行度是主项', { size: 10 }));
  body.push(cap(36, 410, 'e2e 按 Amdahl 折扣：8B B=1 attention 5.13 ms + 权重 4.81 ms → 上界 ≈ 2.1×；博客 34B 长窗口口径 attention 50× / e2e 8×', { size: 10 }));

  return scene(720, 424, body, { label: 'split-KV：KV 序列切 14 段视图，各段并行 FlashAttention 产出 O 与 LSE 状态，归约核按 LSE 加权合并成与不切逐位相同的输出' }).svg();
}

/* ============================================================
 * 3.1 §4（block 1）— cascade 两级（B=4 共享前缀，行块=4）
 * 原文：级 1 前缀读一遍进 shared memory；级 2 后缀 1×1 块；merge_state 式 (3)
 *       IO 账 B=32：9 GiB → 1.25 GiB（省 86.1%）；与 prefix caching 正交
 * ============================================================ */
function cascadeTwoLevel() {
  const UID = 'i3b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'cascade 两级 attention（B=4 共享前缀 P · 行块=4 示意）：前缀读一遍、后缀各读各的、⊕ 合并', { size: 12, color: 'var(--ink-soft)' }));

  // 级 1
  body.push(zone({ x: 36, y: 44, w: 320, h: 224, label: '级 1：前缀 P（T_p=2048）· 大行块 BSR' }));
  const P = node({ id: 'P', x: 52, y: 128, w: 128, h: 60, role: 'cache', title: '前缀 P 的 KV', sub: '读一遍进 shared memory', titleSize: 12, subSize: 8.5 });
  nodes.push(P); body.push(P.svg);
  const qs = [76, 120, 164, 208].map((y, i) => {
    const n = node({ id: `q${i}`, x: 224, y, w: 116, h: 32, role: 'data', title: `q${i + 1} → (O_P, LSE_P)`, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  qs.forEach((q, i) => {
    body.push(edge(P, q, { nodes, pulse: true, dur: 1.4, delay: i * 0.2, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(q.left[0], q.left[1], 1, 0));
  });
  body.push(cap(196, 254, '4 query 共享一次前缀读取', { size: 9, anchor: 'middle' }));

  // 级 2
  body.push(zone({ x: 376, y: 44, w: 308, h: 224, label: '级 2：独立后缀 Uᵢ · 1×1 块 BSR' }));
  const us = [76, 120, 164, 208].map((y, i) => {
    const n = node({ id: `u${i}`, x: 392, y, w: 276, h: 32, role: 'attn', title: `U${i + 1}（T_s=256）→ (O_U, LSE_U)`, mono: true, titleSize: 10 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(cap(530, 254, '每 query 走 global / L2 读自己的后缀', { size: 9, anchor: 'middle' }));

  // merge
  const M = node({ id: 'M', x: 180, y: 292, w: 360, h: 46, role: 'op', title: 'merge_state（式 3）逐 query：前缀状态 ⊕ 后缀状态', sub: 'Vᵢ = P ⊔ Uᵢ 不相交并 → 输出 = 全量 attention（命题 2）', titleSize: 12, subSize: 9 });
  nodes.push(M); body.push(M.svg);
  body.push(edge([196, 268], [280, 292], { pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'm1' }));
  body.push(arrowHead(280, 292, 0, 1));
  body.push(edge([530, 268], [440, 292], { pulse: true, dur: 1.2, delay: 1.2, uid: UID, eid: 'm2' }));
  body.push(arrowHead(440, 292, 0, 1));
  body.push(cap(360, 356, '前缀 HBM 读取：B 次 → ⌈B/行块⌉ 次（前缀不可变保证共享状态合法）', { size: 9.5, anchor: 'middle' }));

  // IO 账条（B=32 口径）：9 GiB → 512px 比例尺
  body.push(cap(52, 386, 'B=32 · T_p=2048 · T_s=256 · κL=128 KiB（式 5）', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-soft" x="52" y="394" width="460" height="14" rx="3"/></g>`);
  body.push(cap(520, 405, 'M_naive = 9 GiB', { mono: true, size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="52" y="416" width="13" height="14" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-warm" x="68" y="416" width="51" height="14" rx="3"/></g>`);
  body.push(cap(128, 427, 'M_cas ≈ 256 MiB 前缀 + 1 GiB 后缀 = 1.25 GiB —— 省 86.1%', { mono: true, size: 9 }));
  body.push(cap(36, 446, 'e2e 实测 ITL −13.73%（8B · n=4）：Amdahl + 短序列 attention 份额低 —— IO 省 86% ≠ e2e 省 86%', { size: 9.5 }));
  body.push(cap(36, 462, '与 2.2 正交：prefix caching 省容量与 prefill（跨时间复用），cascade 省同步内前缀重复读（跨请求带宽共享），可叠加', { size: 9.5 }));

  return scene(720, 474, body, { label: 'cascade 两级：四个 query 共享一次前缀 KV 读取算前缀状态，各请求独立算后缀状态，merge state 合并；前缀流量从 B 份压到一份' }).svg();
}

/* ============================================================
 * 3.1 §7（block 2）— DSA decode 选择链（V4.1：T=131072、K=512）
 * 原文：indexer K cache → 打分 → candidate_block_max(b_c=16) → 块 TopK
 *       → DeepSelect token TopK → restore → sparse MLA；主路 20 MiB vs indexer 1.25 GiB
 * ============================================================ */
function dsaSelectChain() {
  const UID = 'i3c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DSA decode 选择链（V4.1 口径 T=131072 · K=512）：稀疏化把带宽瓶颈搬到 indexer K', { size: 12, color: 'var(--ink-soft)' }));

  const chain = [
    ['n0', 48, 'cache', 'indexer K cache [T, 128] / 层', '头间共享一份 · bf16 教学口径（生产 MXFP4）'],
    ['n1', 110, 'op', 'indexer 打分 → scores [T]', 'DeepGEMM mqa_logits（3.3）· ReLU 加权内积'],
    ['n2', 172, 'op', 'candidate_block_max：b_c=16 池化 → 块分数 [T/16]', '块级 TopK 选 K_b 块 → 压实 K_b×16 列'],
    ['n3', 234, 'op', 'DeepSelect token TopK（K = 512）', 'Scan / Filter / Compact · 每元素恰读一遍'],
    ['n4', 296, 'data', 'restore：block_id×16 + offset → indices [B, s_q, 512]', '页地址已 bake · kernel 不再消费块表'],
    ['n5', 358, 'attn', 'sparse MLA decode（FlashMLA）：只 gather 512 行', '输出 (O, LSE) · 全量的 0.39%'],
  ].map(([id, y, role, title, sub]) => {
    const n = node({ id, x: 216, y, w: 330, h: 48, role, title, sub, titleSize: 11.5, subSize: 8.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 5; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, portFrom: chain[i].bottom, portTo: chain[i + 1].top, pulse: true, dur: 1, delay: i * 0.3, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(chain[i + 1].top[0], chain[i + 1].top[1], 0, 1));
  }
  // 左栏注记
  body.push(cap(204, 76, '全量 T 行', { size: 9, anchor: 'end' }));
  body.push(cap(204, 138, '全历史逐 token 打分', { size: 9, anchor: 'end' }));
  body.push(cap(204, 200, '预选：K_b ≥ K+1 才精确', { size: 9, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(204, 262, '期望处理 ≈ 16% T（式 9）', { size: 9, anchor: 'end' }));
  body.push(cap(204, 324, '与 2.1 式 (3) 同形', { size: 9, anchor: 'end' }));
  body.push(cap(204, 386, '主路只读选中行', { size: 9, anchor: 'end' }));
  // 右栏注记
  body.push(cap(558, 76, '128 维 ≪ 主路 512', { size: 9 }));
  body.push(cap(558, 138, '但读取是全量 T', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(558, 200, '最后有效块强制 +inf', { size: 9 }));
  body.push(cap(558, 262, '2–20× torch.topk', { size: 9 }));
  body.push(cap(558, 324, '无效位置 −1', { size: 9 }));
  body.push(cap(558, 386, '256× 稀疏', { size: 9 }));

  body.push(cap(36, 430, '读取账（式 10）：主路 512×1024×40 = 20 MiB/步（全量对照 5 GiB）· indexer 131072×128×2×40 = 1.25 GiB = 64× 主路', { mono: true, size: 9.5 }));
  body.push(cap(36, 448, '换 MXFP4 indexer（0.53125 字节/元素）→ ≈340 MiB，仍 17× 主路、较 bf16 省 3.76× —— indexer 精度/格式成为新杠杆（2.3 / 3.2）', { size: 10 }));

  return scene(720, 460, body, { label: 'DSA 选择链：indexer 对全历史打分，块级预选压实候选，DeepSelect 选出 512 行，还原页地址后稀疏 MLA 只 gather 选中行；瓶颈搬到 indexer 读取' }).svg();
}

/* ============================================================
 * 3.2 §1（block 0）— 三族算法分类：共同目标 × 三种改映射
 * ============================================================ */
function quantThreeFamilies() {
  const UID = 'i3d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '三族量化算法：同一个目标 min ‖WX − ŴX‖²，三种改量化映射的方式', { size: 12, color: 'var(--ink-soft)' }));

  const G = node({ id: 'G', x: 230, y: 44, w: 260, h: 52, role: 'proj', title: 'min ‖WX − ŴX‖²', sub: '层重建误差（Ŵ = 量化后权重）', mono: true, titleSize: 14, subSize: 9.5 });
  nodes.push(G); body.push(G.svg);

  const fam = [
    ['f1', 36, 'attn', 'GPTQ', '改舍入决策 · 二阶误差补偿', '量化一列 → 误差按逆 Hessian', '补偿到未量化列，再量化下一列'],
    ['f2', 260, 'norm', 'AWQ', '改量化格点 · 逐通道缩放', '放大 salient 权重 ×s、激活 ÷s', '等价变换下相对舍入误差下降'],
    ['f3', 484, 'cache', 'SmoothQuant / LLM.int8', '改分布形状 · outlier 处理', '迁移：激活 outlier 摊到权重', '隔离：outlier 维走 fp16'],
  ].map(([id, x, role, title, sub, l1, l2]) => {
    const n = node({ id, x, y: 146, w: 200, h: 92, role, title, sub, titleSize: 13, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(x + 100, 220, l1, { size: 9, anchor: 'middle' }));
    body.push(cap(x + 100, 234, l2, { size: 9, anchor: 'middle' }));
    return n;
  });
  fam.forEach((f, i) => {
    body.push(edge(G, f, { nodes, portFrom: G.bottom, portTo: f.top, pulse: true, dur: 1.4, delay: i * 0.3, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(f.top[0], f.top[1], 0, 1));
  });

  // 路线分 lane
  body.push(zone({ x: 36, y: 268, w: 424, h: 52, label: 'weight-only（W4A16）：省权重读带宽，利好 memory-bound decode' }));
  body.push(cap(52, 306, 'bf16 → 4 bit：权重读字节 ÷4、同 batch 强度 ×4（1.2 式 4）', { size: 9.5 }));
  body.push(zone({ x: 484, y: 268, w: 200, h: 52, label: 'W8A8：权重激活同压 INT8' }));
  body.push(cap(500, 306, '标准 INT8 GEMM 即可', { size: 9.5 }));
  body.push(edge(fam[0], [140, 268], { nodes, portFrom: fam[0].bottom, dash: true, arrow: false, uid: UID, eid: 'l1' }));
  body.push(edge(fam[1], [350, 268], { nodes, portFrom: fam[1].bottom, dash: true, arrow: false, uid: UID, eid: 'l2' }));
  body.push(edge(fam[2], [584, 268], { nodes, portFrom: fam[2].bottom, dash: true, arrow: false, uid: UID, eid: 'l3' }));

  body.push(cap(36, 348, '三族并非互斥：GPTQ / AWQ 作用在 weight-only 路线，SmoothQuant / LLM.int8 作用在 W8A8 路线', { size: 10 }));
  body.push(cap(36, 366, '格式字节账（§2）是所有路线共用的底座 —— 先算清一个元素实际占几字节，再谈算法怎么压', { size: 10 }));

  return scene(720, 380, body, { label: '量化三族算法从共同层重建目标分出：GPTQ 改舍入决策、AWQ 改量化格点、SmoothQuant 与 LLM.int8 改分布形状，分别落在 weight-only 与 W8A8 两条路线' }).svg();
}

/* ============================================================
 * 3.2 §2（afterHeading）— 量化格点与格式字节账
 * 原文：q=4 absmax=1.0 → Δ=0.125，w=0.6 落 0.625 误差 0.025；粒度四档；
 *       INT4-g128 对称 8.25 MiB（3.88×）开销 3.125%；MXFP4 0.53125 / NVFP4 0.5625
 * ============================================================ */
function quantLatticeBytes() {
  const UID = 'i3e';
  const body = [];

  body.push(cap(36, 24, '量化格式账：一个 INT4 权重实际占几字节 —— 格点映射 + scale 摊薄（脊柱 W 4096×4096 · bf16 = 32 MiB）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 左：对称格点 ---- */
  body.push(zone({ x: 36, y: 44, w: 340, h: 180, label: '对称格点（式 1）：Δ = absmax / 2^(q−1)' }));
  const ax = 64, aw = 284, ay = 130; // 0..1 → 284px
  body.push(`<line class="dia-edge strong" x1="${ax}" y1="${ay}" x2="${ax + aw}" y2="${ay}"/>`);
  for (let k = 0; k <= 8; k++) {
    const x = ax + k * (aw / 8);
    body.push(`<line class="dia-edge" x1="${x}" y1="${ay - 5}" x2="${x}" y2="${ay + 5}"/>`);
    if (k % 2 === 0) body.push(cap(x, ay + 20, (k * 0.125).toFixed(3).replace(/0+$/, '').replace(/\.$/, '.0'), { mono: true, size: 8.5, anchor: 'middle' }));
  }
  body.push(cap(ax + aw / 16, ay - 12, 'Δ = 0.125', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  // w=0.6 → 0.625
  const wx = ax + 0.6 * aw, qx = ax + 0.625 * aw;
  body.push(`<circle cx="${wx}" cy="${ay - 34}" r="4" fill="var(--ink)"/>`);
  body.push(cap(wx + 8, ay - 38, 'w = 0.6', { mono: true, size: 9.5 }));
  body.push(edge([wx, ay - 30], [qx, ay - 6], { pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'q' }));
  body.push(`<circle cx="${qx}" cy="${ay}" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(64, 176, 'q=4 · absmax=1.0：落格点 0.625（5×0.125）· 舍入误差 0.025', { size: 9.5 }));
  body.push(cap(64, 194, '对称格点 [−8, 7]（负侧多一格）；非对称（式 2）加 zero-point', { size: 9 }));
  body.push(cap(64, 210, '→ scale 开销翻倍（GPTQ 用 min-max，AWQ 用 absmax）', { size: 9 }));

  /* ---- 右：粒度四档 ---- */
  body.push(zone({ x: 396, y: 44, w: 288, h: 180, label: '粒度 = 哪些元素共享一个 Δ' }));
  const gx = 412, gw = 256;
  const gran = [
    [90, 'per-tensor：整矩阵 1 个 scale', 'one'],
    [126, 'per-channel：每输出通道 1 个', 'rows'],
    [162, 'group：每 g=128 输入通道（AWQ/GPTQ）', 'group'],
    [198, 'block：32×32 二维块（V3/V4.1）', 'block'],
  ];
  gran.forEach(([y, lab, kind]) => {
    body.push(cap(gx, y - 4, lab, { size: 8.5 }));
    if (kind === 'one') body.push(`<rect x="${gx}" y="${y}" width="${gw}" height="14" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1"/>`);
    if (kind === 'rows') for (let r = 0; r < 2; r++) body.push(`<rect x="${gx}" y="${y + r * 8}" width="${gw}" height="6" rx="2" fill="${r ? '#f3ecdc' : 'var(--seal-wash)'}" stroke="${r ? '#cbb98d' : 'var(--seal)'}" stroke-width="0.8"/>`);
    if (kind === 'group') for (let s = 0; s < 4; s++) body.push(`<rect x="${gx + s * 65}" y="${y}" width="61" height="14" rx="3" fill="${s === 1 ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${s === 1 ? 'var(--seal)' : '#c6c0af'}" stroke-width="1"/>`);
    if (kind === 'block') for (let r = 0; r < 2; r++) for (let c = 0; c < 4; c++) body.push(`<rect x="${gx + c * 65}" y="${y + r * 8}" width="61" height="6" rx="2" fill="${r === 0 && c === 2 ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${r === 0 && c === 2 ? 'var(--seal)' : '#c6c0af'}" stroke-width="0.8"/>`);
  });

  /* ---- 下：字节账 ---- */
  body.push(cap(36, 252, 'bytes_elem（式 3）= (k·bytes_data + bytes_scale + bytes_zp) / k · group 对称开销（式 4）= 16/(gq)', { mono: true, size: 10 }));
  // 32 MiB → 512px（16px/MiB）
  body.push(cap(52, 282, 'bf16', { size: 9.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="110" y="272" width="512" height="14" rx="3"/>`);
  body.push(cap(630, 283, '32 MiB', { mono: true, size: 9 }));
  body.push(cap(52, 306, 'INT4-g128', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-ink" x="110" y="296" width="128" height="14" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="241" y="296" width="4" height="14" rx="2"/></g>`);
  body.push(cap(254, 307, '8 MiB 数据 + 256 KiB scale = 8.25 MiB → 3.88×（开销 3.125%，非对称 6.25%）', { mono: true, size: 9 }));
  body.push(cap(36, 336, '块格式把 scale 收进格式：MXFP8（32×E4M3 + E8M0）1.03125 字节/元素 · 1.94× | MXFP4（E2M1）0.53125 · 3.76×', { size: 9.5 }));
  body.push(cap(36, 354, 'NVFP4（16×E2M1 + UE4M3 + FP32 global）0.5625 · 3.56×（SM100）| V4.1 权重块 32×32 FP8 + ue8m0：(1024+1)/1024 ≈ 1.00098', { size: 9.5 }));
  body.push(cap(36, 372, '粒度越细 outlier 只抬高自己所在组的 scale，其余组小元素保住 —— 精度收益与 scale 开销方向相反，g 是折中旋钮', { size: 10 }));

  return scene(720, 386, body, { label: '量化格式账：对称格点把权重就近映到 Δ 间隔格点；粒度四档决定 scale 数量；INT4 加 g128  scale 合计 8.25 MiB，对 bf16 压缩 3.88 倍' }).svg();
}

/* ============================================================
 * 3.2 §3（block 1）— GPTQ 误差补偿 toy + B=128 块更新
 * 原文：w=(0.6,0.24) 格点 {0,0.5,1.0}；e₁=0.1 → δ₂=+0.05 → 0.29 落 0.5；
 *       MSE 0.1832 → 0.1032（−43.7%）；lazy batch + Cholesky
 * ============================================================ */
function gptqCompensation() {
  const UID = 'i3f';
  const body = [];

  body.push(cap(36, 24, 'GPTQ：已量化列的误差按逆 Hessian 补偿到未量化列（toy w=(0.6, 0.24) · 格点 {0, 0.5, 1.0}）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 左：格点 + 补偿 ---- */
  body.push(zone({ x: 36, y: 44, w: 400, h: 236, label: '误差补偿（式 7）：δ_F = −(e_q / [H⁻¹]_qq) · [H⁻¹]_F,q' }));
  const ax = 76, aw = 320, ay = 150; // 0..1 → 320px：0→76, 0.5→236, 1.0→396
  body.push(`<line class="dia-edge strong" x1="${ax}" y1="${ay}" x2="${ax + aw}" y2="${ay}"/>`);
  [[0, '0'], [0.5, '0.5'], [1.0, '1.0']].forEach(([v, lab]) => {
    const x = ax + v * aw;
    body.push(`<line class="dia-edge" x1="${x}" y1="${ay - 7}" x2="${x}" y2="${ay + 7}"/>`);
    body.push(cap(x, ay + 24, lab, { mono: true, size: 10, anchor: 'middle' }));
  });
  // 列 1：w1=0.6 → 0.5（上方）
  const w1x = ax + 0.6 * aw, q1x = ax + 0.5 * aw;
  body.push(`<circle cx="${w1x}" cy="${ay - 40}" r="4" fill="var(--ink)"/>`);
  body.push(cap(w1x + 8, ay - 44, 'w₁ = 0.6', { mono: true, size: 9.5 }));
  body.push(edge([w1x, ay - 36], [q1x + 3, ay - 8], { pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'q1' }));
  body.push(arrowHead(q1x + 3, ay - 8, 0, 1));
  body.push(cap(150, ay - 66, '先量化列 1：落 0.5 · e₁ = 0.1', { size: 9.5, color: 'var(--seal-deep)' }));
  // 列 2：w2=0.24 +0.05 → 0.29 → 0.5（下方）
  const w2x = ax + 0.24 * aw, c2x = ax + 0.29 * aw;
  body.push(`<circle cx="${w2x}" cy="${ay + 44}" r="4" fill="var(--ink)"/>`);
  body.push(cap(w2x - 8, ay + 48, 'w₂ = 0.24', { mono: true, size: 9.5, anchor: 'end' }));
  const comp = `M ${w2x} ${ay + 44} L ${c2x} ${ay + 44}`;
  body.push(`<path class="sig-dashflow" d="${comp}" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(pathPulse(comp, { dur: 1.2, delay: 0.9 }));
  body.push(arrowHead(c2x, ay + 44, 1, 0));
  body.push(cap(c2x + 8, ay + 48, '+δ₂ = 0.05 → 0.29', { mono: true, size: 9.5, color: 'var(--seal-deep)' }));
  body.push(edge([c2x, ay + 40], [q1x + 6, ay + 8], { pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'q2' }));
  body.push(arrowHead(q1x + 6, ay + 8, 0, -1));
  body.push(cap(252, ay + 74, '再量化落 0.5', { size: 9.5, anchor: 'middle' }));
  body.push(cap(52, 248, 'H = 2XXᵀ = [[4,2],[2,4]] · [H⁻¹]₁₁ = 1/3 · [H⁻¹]₂₁ = −1/6', { mono: true, size: 9.5 }));
  body.push(cap(52, 266, 'δ₂ = −(0.1/(1/3))×(−1/6) = +0.05 —— 列 2 逐权重误差 0.24→0.26 反涨，输出误差反降', { size: 9 }));

  /* ---- 右：MSE 对比 ---- */
  body.push(zone({ x: 456, y: 44, w: 228, h: 236, label: '输出 MSE ½eᵀHe（式 5 目标）' }));
  const base = 236, sc = 560; // 0.1832→102.6px, 0.1032→57.8px
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-soft" x="492" y="${base - 0.1832 * sc}" width="56" height="${0.1832 * sc}" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.7"><rect class="dia-bar dia-bar-seal" x="592" y="${base - 0.1032 * sc}" width="56" height="${0.1032 * sc}" rx="3"/></g>`);
  body.push(cap(520, base - 0.1832 * sc - 8, '0.1832', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(620, base - 0.1032 * sc - 8, '0.1032', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(520, base + 18, 'RTN', { size: 10, anchor: 'middle' }));
  body.push(cap(620, base + 18, 'GPTQ', { size: 10, anchor: 'middle' }));
  body.push(cap(520, base + 34, 'ŵ=(0.5, 0.0)', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(620, base + 34, 'ŵ=(0.5, 0.5)', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(570, 120, '−43.7%', { size: 13, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));

  /* ---- 下：B=128 块更新 ---- */
  body.push(zone({ x: 36, y: 296, w: 648, h: 84, label: '工程三步松弛：任意列序（H_F 全行共享）· B=128 Lazy Batch-Updates · Cholesky 预计算' }));
  const blk = [52, 214, 376, 538];
  blk.forEach((x, i) => {
    body.push(`<rect x="${x}" y="330" width="140" height="24" rx="5" fill="${i === 0 ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${i === 0 ? 'var(--seal)' : '#c6c0af'}" stroke-width="1.1"/>`);
    body.push(cap(x + 70, 346, ['块 0（128 列）', '块 1（128 列）', '⋯', '块 31'][i], { mono: true, size: 9.5, anchor: 'middle' }));
  });
  const glob = 'M 192 360 H 660';
  body.push(`<path class="dia-edge dash" d="${glob}"/>`);
  body.push(pathPulse(glob, { dur: 2, delay: 1.2 }));
  body.push(arrowHead(660, 360, 1, 0));
  body.push(cap(122, 372, '块内逐列量化 · 只更新本块后续列', { size: 8.5, anchor: 'middle' }));
  body.push(cap(430, 372, '块末：累积误差 E 全局更新所有剩余列', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 404, '复杂度 O(d_out·d_in³) → O(max{d_out·d_in², d_in³})：逆更新每列一次而非每权重一次（脊柱 4096× 降）；dampening λ = 1% 平均对角', { size: 9.5 }));
  body.push(cap(36, 422, '补偿利用的是误差经 X 的相关结构，不是逐权重误差最小 —— 这就是「二阶」相对 RTN 的全部增量（H 对角占优时退化为 RTN）', { size: 9.5 }));

  return scene(720, 436, body, { label: 'GPTQ 玩具例：列 1 量化误差 0.1 按逆 Hessian 补偿列 2 加 0.05 再量化，输出 MSE 从 0.1832 降到 0.1032；工程上 B128 块内更新加 Cholesky 预计算' }).svg();
}

/* ============================================================
 * 3.2 §5（block 2）— SmoothQuant 迁移（α=0.5，outlier 通道）
 * 原文：m_X=100、m_W=1 → s=10，两边 max 都 10（几何平均）；
 *       有效级数 2.56 → 25.6；离线融合零运行时 scaling
 * ============================================================ */
function smoothquantMigration() {
  const UID = 'i3g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'SmoothQuant：把激活 outlier 的量化难度迁移到权重（α=0.5 · outlier 通道 m_X=100 · m_W=1）', { size: 12, color: 'var(--ink-soft)' }));

  const X = node({ id: 'X', x: 52, y: 64, w: 180, h: 56, role: 'data', title: '激活 X · 通道 j', sub: 'max 100（outlier ~100×）', titleSize: 12.5, subSize: 9 });
  const XH = node({ id: 'XH', x: 330, y: 64, w: 180, h: 56, role: 'norm', title: '平滑 X̂ · 通道 j', sub: 'max 100/10 = 10', titleSize: 12.5, subSize: 9 });
  const W = node({ id: 'W', x: 52, y: 156, w: 180, h: 56, role: 'cache', title: '权重 W · 通道 j', sub: 'max 1（分布平坦）', titleSize: 12.5, subSize: 9 });
  const WH = node({ id: 'WH', x: 330, y: 156, w: 180, h: 56, role: 'norm', title: '调整 Ŵ · 通道 j', sub: 'max 1×10 = 10', titleSize: 12.5, subSize: 9 });
  [X, XH, W, WH].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(X, XH, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'x' }));
  body.push(arrowHead(XH.left[0], XH.left[1], 1, 0));
  body.push(cap(281, 80, '÷s = 10', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge(W, WH, { nodes, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 'w' }));
  body.push(arrowHead(WH.left[0], WH.left[1], 1, 0));
  body.push(cap(281, 172, '×s = 10', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 右侧：难度均分
  body.push(zone({ x: 548, y: 64, w: 136, h: 148, tone: 'seal', label: '难度均分' }));
  body.push(cap(616, 108, '都 = 10', { size: 14, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(616, 130, '√(m_X·m_W)', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(616, 150, '= √100', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(616, 176, '命题 4：同通道内', { size: 8.5, anchor: 'middle' }));
  body.push(cap(616, 190, '权重/激活均分', { size: 8.5, anchor: 'middle' }));
  body.push(edge(XH, [548, 92], { nodes, pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'j1' }));
  body.push(edge(WH, [548, 184], { nodes, pulse: true, dur: 1.2, delay: 1.2, uid: UID, eid: 'j2' }));

  body.push(cap(52, 240, 's_j = absmax(X_j)^α / absmax(W_j)^(1−α)（式 15）= 100^0.5 = 10 · α=0.5 是 OPT/BLOOM sweet spot（GLM-130B 取 0.75）', { mono: true, size: 9.5 }));

  // 有效量化级数账
  body.push(cap(52, 272, 'per-tensor INT8 有效级数 = 256 × mᵢ/m（正常通道 mᵢ=1）：', { size: 9.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="282" width="13" height="14" rx="3"/>`);
  body.push(cap(74, 293, '迁移前 m=100 → ≈2.56 级（信息基本尽毁）', { mono: true, size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-seal" x="52" y="304" width="128" height="14" rx="3"/></g>`);
  body.push(cap(188, 315, '迁移后 m=10 → 25.6 级（10× 改善；α=1 才满级 256）', { mono: true, size: 9 }));

  body.push(cap(36, 348, '全精度等价（式 14）：(W diag(s))(diag(s)⁻¹ X) = WX —— diag(s)⁻¹ 离线融进前一层 LayerNorm，运行时零额外 scaling', { size: 10 }));
  body.push(cap(36, 366, '正常通道 m_X=m_W=1 → s=1、max 仍 1：均分的是同一通道内的难度，不是跨通道激活幅度拉平（命题 4 口径）', { size: 10 }));
  body.push(cap(36, 384, '对照 LLM.int8()：outlier 维隔离走 fp16（mixed-precision 硬件难高效）vs SmoothQuant 摊平后全矩阵统一 INT8（标准 GEMM）', { size: 10 }));

  return scene(720, 398, body, { label: 'SmoothQuant 迁移：激活除以 s 权重乘以 s，outlier 通道两边最大幅度都变成几何平均 10，正常通道有效量化级数从 2.56 升到 25.6，变换离线融合' }).svg();
}

/* ============================================================
 * 3.3 §4（block 0）— grouped GEMM 两种 layout（E/e=4，M_blk=128）
 * 原文：M_j=[32,17,128,5] pad=[96,111,0,123]；contiguous 动态 shape vs
 *       masked 静态容量 + GPU mask；w=64.5%；V3 μ=32 → 75%
 * ============================================================ */
function groupedGemmLayouts() {
  const UID = 'i3h';
  const body = [];

  body.push(cap(36, 24, 'MoE grouped GEMM 两种 layout（每卡 E/e=4 个本地 expert · M_blk=128）：padding 发生在 M 轴（token 维）', { size: 12, color: 'var(--ink-soft)' }));

  // 比例尺：512 行 → 592px（1.15625px/行），每段 128 行 = 148px
  const M = [32, 17, 128, 5], pad = [96, 111, 0, 123];
  const segX = (i) => 52 + i * 152;
  const rowPx = 148 / 128;

  body.push(zone({ x: 36, y: 44, w: 648, h: 122, label: 'contiguous（训练 fwd / prefill：CPU 知道每个 M_j）—— token 按 expert 排序拼接，每段对齐 M_blk' }));
  M.forEach((m, i) => {
    const x = segX(i), vw = m * rowPx;
    body.push(`<g class="sig-bar" style="--i:${(i * 0.25).toFixed(2)}"><rect class="dia-bar dia-bar-ink" x="${x}" y="82" width="${vw}" height="26" rx="3"/></g>`);
    if (pad[i] > 0) body.push(`<rect x="${x + vw + 1}" y="82" width="${pad[i] * rowPx - 1}" height="26" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
    body.push(cap(x + 74, 76, `expert ${i}`, { size: 9, anchor: 'middle' }));
    body.push(cap(x + 74, 124, `M=${m} + pad ${pad[i]}`, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  [0, 128, 256, 384, 512].forEach((off, i) => {
    body.push(cap(52 + i * 152 - (i === 4 ? 4 : 0), 148, String(off), { mono: true, size: 8.5, anchor: i === 0 ? 'start' : i === 4 ? 'end' : 'middle' }));
  });
  body.push(cap(360, 162, 'offsets = [0,128,256,384,512] · 形状 [512, K] 随路由变 → 动态 shape', { size: 8.5, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 182, w: 648, h: 124, label: 'masked（decode + CUDA Graph：CPU 不知道 M_j）—— 静态容量 buffer + GPU 侧 mask' }));
  M.forEach((m, i) => {
    const x = segX(i), vw = m * rowPx;
    body.push(`<rect x="${x}" y="222" width="148" height="26" rx="3" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="1.1"/>`);
    body.push(`<g class="sig-bar" style="--i:${(0.8 + i * 0.25).toFixed(2)}"><rect class="dia-bar dia-bar-seal" x="${x}" y="222" width="${vw}" height="26" rx="3"/></g>`);
    body.push(cap(x + 74, 216, '容量 C', { size: 8.5, anchor: 'middle' }));
    body.push(cap(x + 74, 264, `有效 ${m}（mask）`, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  body.push(cap(52, 286, '形状 [4, C, K] 静态 · kernel 只算有效 block · buffer 形状不变 → 可被 CUDA Graph 捕获（典型输入 = DeepEP low-latency 输出）', { size: 9 }));

  body.push(cap(36, 330, 'padding 账（式 3）：w = Σpad / Σ(M+pad) = 330/512 = 64.5% = 1 − 45.5/128 —— 命题 2(b)：每段非空且 ≤ M_blk 时分布无关', { mono: true, size: 9.5 }));
  body.push(cap(36, 348, 'V3 decode B=1024 → μ = Bk/E = 32 → w = 1 − 32/128 = 75% 精确成立；prefill μ=512 → ≈11.0% —— decode 小 M 是重灾区', { size: 10 }));
  body.push(cap(36, 366, 'masked 不消除块粒度浪费（不足一 tile 仍按整块算），消除的是 CPU 同步与动态 shape；选型：prefill contiguous · decode+CG masked', { size: 9.5 }));
  body.push(cap(36, 384, '容量代价（式 4）：4×1024×7168×1 = 29.4 MB/层/rank，是有效数据 0.92 MB 的 32× —— 绝对值小，块粒度计算浪费才是主成本', { size: 10 }));

  return scene(720, 398, body, { label: 'grouped GEMM 两种布局：contiguous 按 expert 拼接每段补齐到 128 行、形状动态；masked 静态容量加 GPU 侧 mask 可图捕获；小 batch 下 padding 浪费高达 75%' }).svg();
}

/* ============================================================
 * 3.3 §2（afterHeading）— W4A16：dequant 融 mainloop × batch regime 翻转
 * 原文：0.515625 字节/权重（4.125 bit）→ S_ideal≈3.88；B_max≈76；
 *       B=16 两侧 memory-bound 3.88×；B=128 INT4 翻 compute-bound → 2.31×
 * ============================================================ */
function w4a16Mainloop() {
  const UID = 'i3i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'W4A16 GEMM：dequant 融进 mainloop —— 字节节省只在 memory-bound 区兑现（H100 · [4096,4096] 层）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 648, h: 132, label: '朴素做法物化中间矩阵（节省被吃掉）vs Marlin：tile 载入后寄存器内即时还原' }));
  // 朴素 lane
  const n1 = node({ id: 'n1', x: 52, y: 80, w: 110, h: 34, role: 'cache', title: 'INT4 权重', mono: true, titleSize: 11 });
  const n2 = node({ id: 'n2', x: 222, y: 80, w: 120, h: 34, role: 'op', title: 'dequant kernel', mono: true, titleSize: 11 });
  const n3 = node({ id: 'n3', x: 402, y: 80, w: 170, h: 34, role: 'data', title: 'FP16 中间矩阵 · 落 HBM', titleSize: 10.5 });
  const n4 = node({ id: 'n4', x: 622, y: 80, w: 50, h: 34, role: 'proj', title: 'GEMM', mono: true, titleSize: 10 });
  [n1, n2, n3, n4].forEach((n) => { nodes.push(n); body.push(n.svg); });
  [[n1, n2], [n2, n3], [n3, n4]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: i * 0.2, uid: UID, eid: `n${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(281, 76, '读 4-bit 写 16-bit', { size: 8, anchor: 'middle' }));
  body.push(cap(587, 76, '再读一遍', { size: 8, anchor: 'middle' }));
  // Marlin lane
  const m1 = node({ id: 'm1', x: 52, y: 130, w: 110, h: 34, role: 'cache', title: 'INT4 tile 载入', titleSize: 10.5 });
  const m2 = node({ id: 'm2', x: 222, y: 130, w: 180, h: 34, role: 'op', title: '寄存器 dequant → FP16', titleSize: 10.5 });
  const m3 = node({ id: 'm3', x: 462, y: 130, w: 130, h: 34, role: 'proj', title: 'tensor core 计算', titleSize: 10.5 });
  [m1, m2, m3].forEach((n) => { nodes.push(n); body.push(n.svg); });
  [[m1, m2], [m2, m3]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: 0.6 + i * 0.25, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(608, 151, '中间结果', { size: 8.5 }));
  body.push(cap(608, 164, '不落 HBM', { size: 8.5, color: 'var(--seal-deep)' }));

  // regime 账条：30px/μs
  body.push(cap(52, 202, 'B = 16（两侧 memory-bound）：bf16 33.6 MB → 10.0 μs；INT4-g128 8.65 MB → 2.58 μs；计算 0.54 μs ≪ 2.58', { size: 9.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="210" width="300" height="13" rx="3"/>`);
  body.push(cap(360, 221, 'bf16 10.0 μs', { mono: true, size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-seal" x="52" y="227" width="77" height="13" rx="3"/></g>`);
  body.push(cap(137, 238, 'INT4 2.58 μs → 加速 ≈ 3.88× 成立', { mono: true, size: 9, color: 'var(--seal-deep)' }));

  body.push(cap(52, 268, 'B = 128 > B_max ≈ 76（式 2）：INT4 计算 4.34 μs > 带宽 2.58 → 翻转 compute-bound；bf16 仍 memory-bound 10.0 μs', { size: 9.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="276" width="300" height="13" rx="3"/>`);
  body.push(cap(360, 287, 'bf16 10.0 μs', { mono: true, size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-warm" x="52" y="293" width="130" height="13" rx="3"/></g>`);
  body.push(cap(190, 304, 'INT4 卡在计算 4.34 μs → 加速缩到 2.31×（字节还在省，瓶颈换了边）', { mono: true, size: 9 }));

  body.push(cap(36, 338, 'bytes_weight = q/8 + bytes_scale/g = 0.5 + 2/128 = 0.515625（4.125 bit）· S_ideal = 16/4.125 ≈ 3.88（README 3.87× 同账）', { mono: true, size: 9.5 }));
  body.push(cap(36, 356, 'B_max = R·bytes_weight/2 ≈ 76 —— Marlin 近理想加速保持到 16–32（此前 kernel 仅 1–2），64–128 仍显著，vLLM e2e 至多 2.8×', { size: 9.5 }));
  body.push(cap(36, 374, '差距在工程：dequant 与 tensor core 竞争发射槽、双流水线同时饱和才兑现；tiling 随 token 数自适应 → 非 batch-invariant（3.4）', { size: 9.5 }));

  return scene(720, 388, body, { label: 'W4A16 两条执行路径对比：朴素 dequant 物化 FP16 中间矩阵多一遍 HBM 流量，Marlin 在 mainloop 寄存器内还原直喂 tensor core；batch 超过 76 后瓶颈翻到算力侧加速缩水' }).svg();
}

/* ============================================================
 * 3.3 §5（afterHeading）— Mega MoE 融合重叠 × EP 拐点 τ(e)
 * 原文：未融合 τ_comm+τ_gemm vs 融合 max+ε；τ(e)=max(A/e,F)+g(e)，
 *       A=195.2ms F=6.1ms g=0.02(e−1) → e*=32；表 24.5/12.5/6.72/7.36/8.64
 * ============================================================ */
function megaMoeEpKnee() {
  const UID = 'i3j';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Fused MoE：Mega MoE 把通信重叠进计算 —— EP 扩张存在拐点 e* = A/F（V3 口径教学参数）', { size: 12, color: 'var(--ink-soft)' }));

  // 未融合链
  body.push(zone({ x: 36, y: 44, w: 648, h: 78, label: '未融合：五段独立 kernel 串行，通信时间全暴露 τ ≈ τ_comm + τ_gemm' }));
  const chain = [
    ['d', 52, 'cache', 'EP dispatch'],
    ['g1', 180, 'proj', 'grouped GEMM1'],
    ['s', 308, 'op', 'SwiGLU'],
    ['g2', 436, 'proj', 'grouped GEMM2'],
    ['c', 564, 'cache', 'EP combine'],
  ].map(([id, x, role, title]) => {
    const n = node({ id, x, y: 74, w: 108, h: 34, role, title, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < 4; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, pulse: true, dur: 1, delay: i * 0.25, uid: UID, eid: `u${i}` }));
    body.push(arrowHead(chain[i + 1].left[0], chain[i + 1].left[1], 1, 0));
  }

  // Mega MoE
  body.push(zone({ x: 36, y: 134, w: 648, h: 96, tone: 'seal', label: 'Mega MoE：单 mega-kernel（FP8×FP4 / FP8×FP8）' }));
  const mega = node({ id: 'mega', x: 52, y: 164, w: 380, h: 40, role: 'attn', title: 'dispatch + linear1/2 + SwiGLU + combine 融进一个 kernel', sub: 'τ ≈ max(τ_comm, τ_gemm) + ε —— 取 max，不再相加', titleSize: 11.5, subSize: 9 });
  nodes.push(mega); body.push(mega.svg);
  body.push(`<rect class="dia-bar dia-bar-warm" x="466" y="170" width="186" height="13" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-ink" x="466" y="188" width="186" height="13" rx="3"/>`);
  body.push(cap(559, 164, '通信与计算同时跑（重叠）', { size: 8.5, anchor: 'middle' }));
  const ov = 'M 466 176 H 652';
  body.push(pathPulse(ov, { dur: 1.8, delay: 0.8 }));
  body.push(cap(52, 216, 'scale-up 域内（NVLink）τ_comm 小到藏进计算；跨域 RDMA 延迟藏不住 —— 域内通用解，非跨域通用解（官方披露 doc L285）', { size: 8.5 }));

  // τ(e) 曲线：x 90..650，y 250..372（0..26ms → 4.69px/ms）
  const X = (e) => 90 + Math.log2(e / 8) * 140; // 8→90, 16→230, 32→370, 64→510, 128→650
  const Y = (t) => 372 - t * 4.69;
  body.push(`<line class="dia-edge strong" x1="90" y1="372" x2="660" y2="372"/>`);
  body.push(`<line class="dia-edge strong" x1="90" y1="372" x2="90" y2="244"/>`);
  body.push(arrowHead(660, 372, 1, 0));
  body.push(arrowHead(90, 244, 0, -1));
  body.push(cap(84, 250, 'τ (ms)', { mono: true, size: 8.5, anchor: 'end' }));
  [8, 16, 32, 64, 128].forEach((e) => {
    body.push(`<line class="dia-edge" x1="${X(e)}" y1="367" x2="${X(e)}" y2="377"/>`);
    body.push(cap(X(e), 390, String(e), { mono: true, size: 9, anchor: 'middle' }));
  });
  body.push(cap(370, 406, 'EP 度 e（#DP = #EP 耦合扩张）', { size: 9, anchor: 'middle' }));
  // A/e 虚线
  const ae = [[8, 24.4], [16, 12.2], [32, 6.1], [64, 3.05], [128, 1.52]];
  body.push(`<path class="dia-edge dash" d="M ${ae.map(([e, t]) => `${X(e)} ${Y(t)}`).join(' L ')}"/>`);
  body.push(cap(150, Y(16.5), 'A/e（带宽项）', { size: 8.5 }));
  // F 水平线
  body.push(`<line x1="90" y1="${Y(6.1)}" x2="650" y2="${Y(6.1)}" stroke="var(--seal)" stroke-width="1.1" stroke-dasharray="5 5"/>`);
  body.push(cap(656, Y(6.1) + 3, 'F = 6.1', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  // τ(e) 实线 + 点
  const tau = [[8, 24.5], [16, 12.5], [32, 6.72], [64, 7.36], [128, 8.64]];
  const td = `M ${tau.map(([e, t]) => `${X(e)} ${Y(t)}`).join(' L ')}`;
  body.push(`<path class="dia-edge strong" d="${td}"/>`);
  body.push(pathPulse(td, { dur: 2.6, delay: 0.4 }));
  tau.forEach(([e, t], i) => {
    const knee = e === 32;
    body.push(`<circle cx="${X(e)}" cy="${Y(t)}" r="${knee ? 5 : 3.5}" fill="${knee ? 'var(--seal)' : 'var(--ink)'}"/>`);
    body.push(cap(X(e) + (i === 0 ? 8 : 0), Y(t) - 9, t.toFixed(2).replace(/0$/, ''), { mono: true, size: 8.5, anchor: i === 0 ? 'start' : 'middle', color: knee ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });
  body.push(cap(X(32), Y(6.72) + 20, 'e* = A/F = 32（最小值）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(480, 258, 'τ(e) = max(A/e, F) + g(e)', { mono: true, size: 9, anchor: 'middle' }));

  body.push(cap(36, 428, 'A = 195.2 ms（654 GB fp8 ÷ 3.35 TB/s）· F = 6.1 ms · g(e) = 0.02(e−1) ms（教学参数）；e ≥ e* 后带宽项不再贡献下降，边际收益 ≤ 0', { size: 9.5 }));
  body.push(cap(36, 446, '官方披露：EP > 32 带宽不再是瓶颈、通信/同步开销主导（doc L295）；Mega MoE 把 g(e) 重叠进计算外推拐点 —— 但 g(e) 自身增长终会主导', { size: 9.5 }));

  return scene(720, 458, body, { label: 'Mega MoE 把 dispatch 到 combine 五段融进单 kernel 使通信与计算重叠；EP 步时曲线在 e 等于 32 处取最小值，再扩张通信开销主导步时回升' }).svg();
}

/* ============================================================
 * 3.4 §3（block 0）— eager vs CUDA Graph vs piecewise
 * 原文：330 kernel × 5μs = 1.65ms（B=1 占 34.3%）；replay 省 1.645ms；
 *       piecewise 在 attention 切分；静态三要求、内容动态合法
 * ============================================================ */
function cudaGraphPiecewise() {
  const UID = 'i3k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'eager vs CUDA Graph（decode 一步 n_ker ≈ 330 × t_launch 5 μs = 1.65 ms —— B=1 步时的 34.3%）', { size: 12, color: 'var(--ink-soft)' }));

  // eager
  body.push(zone({ x: 36, y: 44, w: 648, h: 92, label: 'eager：CPU 逐个提交，GPU 间隙 = CPU 提交开销' }));
  body.push(cap(52, 82, 'CPU', { mono: true, size: 9 }));
  for (let i = 0; i < 6; i++) {
    body.push(`<rect x="${92 + i * 62}" y="70" width="52" height="16" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="0.9"/>`);
    body.push(cap(118 + i * 62, 82, 'launch', { mono: true, size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  }
  body.push(cap(478, 82, '⋯ ×330', { mono: true, size: 9 }));
  body.push(cap(52, 112, 'GPU', { mono: true, size: 9 }));
  for (let i = 0; i < 6; i++) {
    body.push(`<rect x="${100 + i * 62}" y="100" width="40" height="16" rx="3" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="0.9"/>`);
    body.push(cap(120 + i * 62, 112, `k${i + 1}`, { mono: true, size: 8, anchor: 'middle' }));
  }
  body.push(cap(478, 112, '⋯（间隙 = 提交）', { size: 8.5 }));
  body.push(cap(560, 126, '步时 = τ_GPU + 330×t_launch', { mono: true, size: 8.5 }));

  // graph
  body.push(zone({ x: 36, y: 148, w: 648, h: 84, label: 'CUDA Graph：DAG 一次录制（capture），整体重放（replay）' }));
  body.push(cap(52, 186, 'CPU', { mono: true, size: 9 }));
  body.push(`<rect x="92" y="174" width="150" height="16" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="0.9"/>`);
  body.push(cap(167, 186, 'replay（一次提交）', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 216, 'GPU', { mono: true, size: 9 }));
  for (let i = 0; i < 8; i++) {
    body.push(`<rect x="${92 + i * 46}" y="204" width="44" height="16" rx="3" fill="var(--paper-raised)" stroke="#c6c0af" stroke-width="0.9"/>`);
    body.push(cap(114 + i * 46, 216, i < 3 ? `k${i + 1}` : i === 7 ? 'k330' : '⋯', { mono: true, size: 8, anchor: 'middle' }));
  }
  const flow = 'M 92 212 H 460';
  body.push(pathPulse(flow, { dur: 1.8, delay: 0.3 }));
  body.push(cap(478, 216, '录好的 DAG 连续执行', { size: 8.5 }));
  body.push(cap(560, 226, '省 ≈ 329×5 μs = 1.645 ms/step', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));

  // piecewise
  body.push(zone({ x: 36, y: 244, w: 648, h: 104, tone: 'seal', label: 'piecewise：在 attention 算子处切分（动态 shape 排除在图外）—— 逐层重复' }));
  const p1 = node({ id: 'p1', x: 52, y: 276, w: 180, h: 44, role: 'proj', title: 'norm+qkv+cast 子图', sub: 'replay（静态 shape）', titleSize: 11, subSize: 8.5 });
  const at = node({ id: 'at', x: 268, y: 276, w: 180, h: 44, role: 'attn', title: 'attention（eager）', sub: '读当前 T / block table', titleSize: 11, subSize: 8.5 });
  const p2 = node({ id: 'p2', x: 484, y: 276, w: 180, h: 44, role: 'proj', title: 'o+FFN+norm 子图', sub: 'replay（静态 shape）', titleSize: 11, subSize: 8.5 });
  [p1, at, p2].forEach((n) => { nodes.push(n); body.push(n.svg); });
  [[p1, at], [at, p2]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: 0.8 + i * 0.4, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(358, 336, '切分点 = splitting_ops（vLLM 默认 attention 算子列表）；grid 依赖序列长的 kernel 捕进图会按旧参数执行', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 372, '静态三要求：指针固定 · shape/grid 固定 · kernel 序列固定 —— 内容动态合法：replay 读 buffer 当前内容，新数据写进同一 buffer 即可', { size: 10 }));
  body.push(cap(36, 390, 'vLLM CUDAGraphMode 默认 FULL_AND_PIECEWISE：decode 批走 full graph · prefill/混合批走 piecewise；batch 向上 pad 到最小可用捕获 size', { size: 10 }));
  body.push(cap(36, 408, '坑：replay 读的是捕获时的指针 —— 新输入必须 copy 进捕获 buffer（cudagraph_copy_inputs），否则输出不随输入变', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 422, body, { label: 'eager 每 kernel 一次 CPU 提交留间隙，CUDA Graph 一次重放整张 DAG；piecewise 在 attention 处切分，静态段 replay 动态 attention eager 执行' }).svg();
}

/* ============================================================
 * 3.4 §7（block 1）— NanoFlow 设备内 nano-batch 流水
 * 原文：τ_c=6 τ_m=4 τ_n=2 ms → 串行 12 vs 流水稳态 6；F_max=Σ/max≤3；
 *       3 nano-batch；干扰折扣；1.91× / 50–72% optimal
 * ============================================================ */
function nanoflowPipeline() {
  const UID = 'i3l';
  const body = [];

  body.push(cap(36, 24, 'NanoFlow 设备内 nano-batch 流水（3 nano-batch × 3 类执行单元；τ_c=6 / τ_m=4 / τ_n=2 ms 教学参数）', { size: 12, color: 'var(--ink-soft)' }));

  const lanes = ['compute 单元', 'memory 单元', 'network 单元'];
  /* ---- 串行：44px/ms，x140 起 ---- */
  body.push(zone({ x: 36, y: 44, w: 648, h: 106, label: '串行（eager 整批）：三类资源轮流忙、互相闲 —— 一批 12 ms' }));
  lanes.forEach((l, i) => body.push(cap(52, 82 + i * 26, l, { size: 9 })));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-ink" x="140" y="70" width="264" height="17" rx="3"/></g>`);
  body.push(cap(272, 83, 'GEMM（整批）6 ms', { size: 8.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(`<rect x="404" y="70" width="264" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-warm" x="404" y="96" width="176" height="17" rx="3"/></g>`);
  body.push(cap(492, 109, 'attn（整批）4 ms', { size: 8.5, anchor: 'middle' }));
  body.push(`<rect x="140" y="96" width="264" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(`<rect x="580" y="96" width="88" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="580" y="122" width="88" height="17" rx="3"/></g>`);
  body.push(cap(624, 135, 'comm 2 ms', { size: 8.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(`<rect x="140" y="122" width="440" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);

  /* ---- 流水：22px/ms，0..24ms ---- */
  body.push(zone({ x: 36, y: 162, w: 648, h: 118, label: 'nano-batch 流水：G/A/C = 各 nano-batch 的 GEMM / attention / comm 段（按时间比例）' }));
  lanes.forEach((l, i) => body.push(cap(52, 200 + i * 26, l, { size: 9 })));
  const T = (t) => 140 + t * 22;
  // compute: G1[0,6] G2[6,12] G3[12,18]
  [[0, 'G1'], [6, 'G2'], [12, 'G3']].forEach(([t, lab], i) => {
    body.push(`<g class="sig-bar" style="--i:${(i * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-ink" x="${T(t)}" y="188" width="132" height="17" rx="3"/></g>`);
    body.push(cap(T(t) + 66, 201, lab, { mono: true, size: 9, anchor: 'middle', color: 'var(--paper-raised)' }));
  });
  body.push(`<rect x="${T(18)}" y="188" width="88" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  // memory: A1[6,10] A2[12,16] A3[18,22]
  [[6, 'A1'], [12, 'A2'], [18, 'A3']].forEach(([t, lab], i) => {
    body.push(`<g class="sig-bar" style="--i:${(0.3 + i * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-warm" x="${T(t)}" y="214" width="88" height="17" rx="3"/></g>`);
    body.push(cap(T(t) + 44, 227, lab, { mono: true, size: 9, anchor: 'middle' }));
  });
  body.push(`<rect x="140" y="214" width="132" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(`<rect x="${T(22)}" y="214" width="44" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  // network: C1[10,12] C2[16,18] C3[22,24]
  [[10, 'C1'], [16, 'C2'], [22, 'C3']].forEach(([t, lab], i) => {
    body.push(`<g class="sig-bar" style="--i:${(0.6 + i * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-seal" x="${T(t)}" y="240" width="44" height="17" rx="3"/></g>`);
    body.push(cap(T(t) + 22, 253, lab, { mono: true, size: 8.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  });
  body.push(`<rect x="140" y="240" width="220" height="17" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  const pf = `M ${T(0)} 196 H ${T(18)}`;
  body.push(pathPulse(pf, { dur: 2.4, delay: 0.5 }));
  body.push(cap(140, 272, '填充 6 ms', { size: 8, anchor: 'middle' }));
  body.push(cap(T(21), 272, '排空', { size: 8, anchor: 'middle' }));
  body.push(cap(400, 272, '稳态：每 6 ms 产出一批（串行 12 ms）—— 3 批合计 24 ms vs 串行 36 ms', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 306, 'F_max = (τ_c+τ_m+τ_n)/max(τ_c,τ_m,τ_n) ≤ 3（式 4）：本例 12/6 = 2×；资源越不均衡上界越低，全均衡取 3；填充/排空瞬态另计', { size: 10 }));
  body.push(cap(36, 324, '干扰折扣：并发算子共享 SM 与 L2、单算子比独占慢 —— 搜索必须把干扰计进目标；实测 1.91×、达 optimal 的 50–72%（摘要口径）', { size: 10 }));
  body.push(cap(36, 342, '伏笔：nano-batch 流水是异步调度的算子层基础 —— SGLang overlap scheduler（6.2）与 DeepInfer 异步调度直接承此路线（doc L334）', { size: 10 }));

  return scene(720, 356, body, { label: 'NanoFlow 流水：串行时三类执行单元轮流忙共 12 毫秒，切成三个 nano-batch 后阶梯重叠，稳态每 6 毫秒产出一批，理想加速上界为三段和除以最长段' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '03-inference-operators/01-decode-attention-kernels',
    replacesBlock: 0,
    title: 'split-KV：KV 切段并行 × (O, LSE) 状态归约',
    svg: splitKvFlow,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/01-decode-attention-kernels',
    replacesBlock: 1,
    title: 'cascade 两级：前缀读一遍 × 后缀各读各的 × ⊕ 合并',
    svg: cascadeTwoLevel,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/01-decode-attention-kernels',
    replacesBlock: 2,
    title: 'DSA 选择链：indexer 全量打分 → 两级 TopK → 稀疏 gather',
    svg: dsaSelectChain,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/02-weight-quantization',
    replacesBlock: 0,
    title: '三族量化算法：同一重建目标 × 三种改映射',
    svg: quantThreeFamilies,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/02-weight-quantization',
    afterHeading: /^## 2\. /,
    title: '量化格点与格式字节账：Δ 映射 × 粒度四档 × scale 摊薄',
    svg: quantLatticeBytes,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/02-weight-quantization',
    replacesBlock: 1,
    title: 'GPTQ 误差补偿：逆 Hessian 分摊 × MSE −43.7% × B=128 块更新',
    svg: gptqCompensation,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/02-weight-quantization',
    replacesBlock: 2,
    title: 'SmoothQuant 迁移：α=0.5 难度均分 × 有效级数 10×',
    svg: smoothquantMigration,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/03-quantized-gemm-and-moe-kernels',
    afterHeading: /^## 2\. /,
    title: 'W4A16：dequant 融 mainloop × batch 越界瓶颈换边',
    svg: w4a16Mainloop,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/03-quantized-gemm-and-moe-kernels',
    replacesBlock: 0,
    title: 'grouped GEMM 两 layout：contiguous 动态拼接 vs masked 静态容量',
    svg: groupedGemmLayouts,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/03-quantized-gemm-and-moe-kernels',
    afterHeading: /^## 5\. /,
    title: 'Mega MoE 通信重叠 × EP 拐点 τ(e) = max(A/e, F) + g(e)',
    svg: megaMoeEpKnee,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/04-fusion-and-graph-capture',
    replacesBlock: 0,
    title: 'eager vs CUDA Graph vs piecewise：launch 账与动态 shape 切分',
    svg: cudaGraphPiecewise,
  },
  {
    dim: 'inference',
    slug: '03-inference-operators/04-fusion-and-graph-capture',
    replacesBlock: 1,
    title: 'NanoFlow nano-batch 流水：Σ/max ≤ 3 的设备内重叠',
    svg: nanoflowPipeline,
  },
];
