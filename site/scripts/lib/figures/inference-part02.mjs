// inference Part 2（02-kv-cache-systems）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 training-part01.mjs / inference-part01.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

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

// KV slot 小格（块内 token 位）：lit = decode 写入点亮（sig-cell），empty = 未写虚线格
const slot = (x, y, label, { lit = null, empty = false, w = 40, h = 26 } = {}) => {
  const rect = empty
    ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="none" stroke="var(--rule)" stroke-dasharray="2 3"/>`
    : `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${lit !== null ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${lit !== null ? 'var(--seal)' : '#c6c0af'}" stroke-width="1.1"/>`;
  const txt = label ? `<text class="dia-sub dia-mono" x="${x + w / 2}" y="${y + h / 2 + 3.5}" text-anchor="middle" font-size="9.5" fill="${lit !== null ? 'var(--seal-deep)' : 'var(--ink)'}">${label}</text>` : '';
  const inner = rect + txt;
  return lit !== null ? `<g class="sig-cell" style="--i:${lit}">${inner}</g>` : inner;
};

// 块芯片（COW / 分页池用）：tone = shared(seal) / own(warm) / free(dash)
const chip = (x, y, label, tone, { w = 38, h = 24, lit = null } = {}) => {
  const fill = tone === 'shared' ? 'var(--seal-wash)' : tone === 'own' ? '#f3ecdc' : 'var(--paper-raised)';
  const stroke = tone === 'shared' ? 'var(--seal)' : tone === 'own' ? '#cbb98d' : 'var(--rule)';
  const dash = tone === 'free' ? ' stroke-dasharray="2 3"' : '';
  const inner =
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="1.1"${dash}/>` +
    `<text class="dia-sub dia-mono" x="${x + w / 2}" y="${y + h / 2 + 3.5}" text-anchor="middle" font-size="9">${label}</text>`;
  return lit !== null ? `<g class="sig-cell" style="--i:${lit}">${inner}</g>` : inner;
};

/* ============================================================
 * 2.1 §1（block 0）— 连续分配三浪费 vs 分页（b=4 示意）
 * ============================================================ */
function contigVsPaged() {
  const UID = 'i2a';
  const body = [];

  body.push(cap(36, 24, '连续分配的三种浪费 vs 分页（b = 4 示意）：实测有效显存仅 20.4%–38.2%', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 120, label: '连续分配：按 M = 2048 预留整段（请求间留空隙）' }));
  body.push(cap(52, 86, '请求 A', { size: 10, weight: '650' }));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-ink" x="110" y="72" width="32" height="16" rx="3"/></g>`);
  body.push(`<rect x="142" y="72" width="224" height="16" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(cap(372, 84, 'M=2048 · T=256 → 浪费 1792 slot = 224 MiB（87.5%）', { mono: true, size: 9 }));
  body.push(cap(52, 112, '请求 B', { size: 10, weight: '650' }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-ink" x="110" y="98" width="24" height="16" rx="3"/></g>`);
  body.push(`<rect x="134" y="98" width="232" height="16" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(cap(372, 110, 'M=2048 · T=192 → 预留 / 内部碎片', { mono: true, size: 9 }));
  body.push(cap(52, 138, '空闲区', { size: 10, weight: '650' }));
  body.push(`<rect x="110" y="124" width="187" height="16" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-dasharray="3 3"/>`);
  body.push(cap(304, 136, '1500 slot 空隙 ← 外部碎片：装不下任何 M=2048', { size: 9, color: 'var(--seal-deep)' }));
  body.push(`<rect x="110" y="146" width="10" height="8" rx="2" fill="var(--ink)"/>`);
  body.push(cap(124, 154, '已用', { size: 8.5 }));
  body.push(`<rect x="160" y="146" width="10" height="8" rx="2" fill="none" stroke="var(--rule)" stroke-dasharray="2 2"/>`);
  body.push(cap(174, 154, '预留 / 内部碎片', { size: 8.5 }));
  body.push(`<rect x="262" y="146" width="10" height="8" rx="2" fill="var(--seal-wash)" stroke="var(--seal)"/>`);
  body.push(cap(276, 154, '外部碎片', { size: 8.5 }));

  body.push(zone({ x: 36, y: 176, w: 648, h: 130, label: '分页分配：物理块定长、无需连续（块间无空隙，按需取用）' }));
  body.push(cap(52, 214, 'A 逻辑', { size: 9.5 }));
  body.push(chip(110, 202, '0', 'shared', { lit: 0.3 }));
  body.push(chip(150, 202, '1', 'shared', { lit: 0.6 }));
  body.push(cap(129, 238, '0→7', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(169, 238, '1→1', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(52, 262, 'B 逻辑', { size: 9.5 }));
  body.push(chip(110, 250, '0', 'own', { lit: 0.9 }));
  body.push(chip(150, 250, '1', 'own', { lit: 1.2 }));
  body.push(cap(129, 286, '0→2', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(169, 286, '1→5', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(250, 214, '物理池', { size: 9.5 }));
  const pool = [7, 2, 1, 5, 4, 3, 6, 0];
  pool.forEach((n, i) => {
    const tone = n === 7 || n === 1 ? 'shared' : n === 2 || n === 5 ? 'own' : 'free';
    const lit = tone === 'free' ? null : [0.3, 0.9, 0.6, 1.2][[7, 2, 1, 5].indexOf(n)];
    body.push(chip(250 + i * 46, 202, String(n), tone, { w: 40, h: 26, lit }));
  });
  body.push(cap(250, 246, 'A 占 7,1 · B 占 2,5 · 其余空闲 —— 无预留、无外部碎片', { size: 9 }));
  body.push(cap(250, 262, '内部碎片 ≤ 每请求最后一块（命题 1：waste ≤ b−1）', { size: 9 }));
  body.push(cap(250, 286, '同一 48 GiB 池：连续 192 条 vs 分页 768 条 —— 4× 并发 → 权重 GEMM 强度同翻 4×', { size: 9.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 330, '三种浪费各个击破：按需分配消灭预留 · 定长块消灭外部碎片 · 内部碎片压到 ≤ 1 块/请求', { size: 10.5 }));
  body.push(cap(36, 348, '连续动态扩容不可行：逐 token 搬运累计 ≈ T²κL/2（T=4096 ≈ 1.1 TB ≈ 0.33 s）；分页每步 O(1) 登记', { size: 10 }));
  body.push(cap(36, 366, '预留 vs 内部碎片是时间性的：存活期叫预留，终止后坐实为内部碎片', { size: 10 }));

  return scene(720, 380, body, { label: '连续分配的预留、内部、外部三种浪费，对照分页定长块按需取用：物理池无空隙，每请求浪费不超过一块' }).svg();
}

/* ============================================================
 * 2.1 §2（block 1）— 块表走读（论文 Fig.6：7 token、b=4）+ decode 两步追加
 * ============================================================ */
function blockTableWalk() {
  const UID = 'i2b';
  const body = [];

  body.push(cap(36, 24, '块表走读（论文 Fig.6：prompt 7 token、b = 4）：逻辑 → 物理的地址翻译 + decode 追加', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 200, h: 250, label: '逻辑视图（连续序列）' }));
  body.push(cap(52, 84, '逻辑 0', { size: 9 }));
  ['t0', 't1', 't2', 't3'].forEach((t, i) => body.push(slot(52 + i * 44, 88, t)));
  body.push(cap(52, 148, '逻辑 1', { size: 9 }));
  ['t4', 't5', 't6'].forEach((t, i) => body.push(slot(52 + i * 44, 152, t)));
  body.push(slot(52 + 3 * 44, 152, 't7', { lit: 1.2 }));
  body.push(cap(52, 212, '逻辑 2（第 2 步追加）', { size: 9 }));
  body.push(slot(52, 216, 't8', { lit: 2.0 }));
  [1, 2, 3].forEach((i) => body.push(slot(52 + i * 44, 216, '', { empty: true })));

  // 块表卡
  body.push(`<rect x="266" y="100" width="180" height="140" rx="12" fill="var(--paper-raised)" stroke="rgb(24 27 32 / 0.22)" stroke-width="1.2"/>`);
  body.push(cap(356, 120, '块表（append-only）', { size: 10.5, anchor: 'middle', weight: '650' }));
  body.push(cap(280, 146, '0 → 物理 7', { mono: true, size: 10 }));
  body.push(cap(432, 146, '#filled=4', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(280, 172, '1 → 物理 1', { mono: true, size: 10 }));
  body.push(cap(432, 172, '#filled 3→4', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(`<g class="sig-cell-text" style="--i:2.0">` + cap(280, 198, '2 → 物理 3', { mono: true, size: 10 }) + cap(432, 198, '#filled=1', { mono: true, size: 8.5, anchor: 'end' }) + `</g>`);
  body.push(cap(356, 224, '表项 = 物理块号 + 已填充数', { size: 8.5, anchor: 'middle' }));

  body.push(zone({ x: 476, y: 44, w: 208, h: 250, label: '物理池（散落定长块）' }));
  body.push(cap(492, 84, '物理 7', { size: 9 }));
  ['t0', 't1', 't2', 't3'].forEach((t, i) => body.push(slot(492 + i * 44, 88, t)));
  body.push(cap(492, 148, '物理 1', { size: 9 }));
  ['t4', 't5', 't6'].forEach((t, i) => body.push(slot(492 + i * 44, 152, t)));
  body.push(slot(492 + 3 * 44, 152, 't7', { lit: 1.2 }));
  body.push(cap(492, 212, '物理 3（新分配）', { size: 9 }));
  body.push(slot(492, 216, 't8', { lit: 2.0 }));
  [1, 2, 3].forEach((i) => body.push(slot(492 + i * 44, 216, '', { empty: true })));

  // 翻译流：逻辑 → 块表 → 物理
  [[224, 101, 266, 142, 0], [224, 165, 266, 168, 0.3], [224, 229, 266, 194, 2.0]].forEach(([ax, ay, bx, by, dl], i) => {
    body.push(edge([ax, ay], [bx, by], { pulse: true, dur: 1.3, delay: dl, uid: UID, eid: `l${i}` }));
    body.push(arrowHead(bx, by, bx - ax, by - ay));
  });
  [[446, 142, 492, 101, 0.5], [446, 168, 492, 165, 0.8], [446, 194, 492, 229, 2.2]].forEach(([ax, ay, bx, by, dl], i) => {
    body.push(edge([ax, ay], [bx, by], { pulse: true, dur: 1.3, delay: dl, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(bx, by, bx - ax, by - ay));
  });

  body.push(cap(36, 318, '式 (3)：j(t) = ⌊t/b⌋ · r(t) = t − j·b · slot(t) = bt[j]·b + r —— t=5：j=1, r=1 → slot = 1×4+1 = 5', { mono: true, size: 10 }));
  body.push(cap(36, 340, 'decode 第 1 步：t7 → 逻辑 1 偏移 3 → 写入物理 1 空 slot（不分配新块，#filled 3→4）', { size: 10 }));
  body.push(cap(36, 358, 'decode 第 2 步：t8 → 逻辑 1 已满 → 新逻辑 2 → 分配物理 3、块表追加一项（映射不改、只追加）', { size: 10 }));
  body.push(cap(36, 376, '#filled 管「读多少」：末块未写 slot 是垃圾值，kernel 据此 mask —— 分块不改 attention 可见集', { size: 10 }));

  return scene(720, 392, body, { label: '块表把逻辑块映射到物理块并记录已填充数；decode 第一步写半满块空 slot，第二步追加新块表项' }).svg();
}

/* ============================================================
 * 2.1 §4（block 2）— 块池状态机：ref_cnt / free 队列头尾 / touch 复活
 * ============================================================ */
function blockLifecycle() {
  const UID = 'i2c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '块池状态机：ref_cnt = 0 ≠ 内容失效 —— 驱逐发生在重新分配那一刻', { size: 12, color: 'var(--ink-soft)' }));

  const A = node({ id: 'A', x: 56, y: 128, w: 150, h: 56, role: 'attn', title: '分配中', sub: 'ref_cnt ≥ 1', mono: true, titleSize: 13, subSize: 10 });
  const Z = node({ id: 'Z', x: 280, y: 128, w: 120, h: 56, role: 'op', title: 'ref_cnt = 0', sub: '无人引用 = 可驱逐', mono: true, titleSize: 12, subSize: 9 });
  const H = node({ id: 'H', x: 480, y: 56, w: 204, h: 60, role: 'cache', title: '有 hash → free 队列尾', sub: '驱逐候选 · 内容仍有效（FIFO/LRU）', titleSize: 12, subSize: 8.5 });
  const N = node({ id: 'N', x: 480, y: 216, w: 204, h: 60, role: 'plain', title: '无 hash → free 队列头', sub: 'LIFO 立即复用（GPU locality）', titleSize: 12, subSize: 8.5 });
  [A, Z, H, N].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(A, Z, { nodes, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'az' }));
  body.push(arrowHead(Z.left[0], Z.left[1], 1, 0));
  body.push(cap(243, 204, '请求结束 / 共享解除 → ref_cnt 归零', { size: 9, anchor: 'middle' }));
  body.push(edge(Z, H, { nodes, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'zh' }));
  body.push(arrowHead(H.left[0], H.left[1], 1, -0.6));
  body.push(cap(470, 76, '有 hash', { size: 9, anchor: 'end' }));
  body.push(edge(Z, N, { nodes, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'zn' }));
  body.push(arrowHead(N.left[0], N.left[1], 1, 0.6));
  body.push(cap(452, 196, '无 hash', { size: 9, anchor: 'end' }));

  // touch 复活（上弧，虚线朱砂）
  const touch = 'M 480 66 C 380 26, 220 26, 131 128';
  body.push(`<path class="dia-edge dash" d="${touch}" stroke="var(--seal)"/>`);
  body.push(pathPulse(touch, { dur: 2, delay: 0.8 }));
  body.push(arrowHead(131, 128, -0.5, 1));
  body.push(cap(300, 60, 'touch：ref_cnt+=1 · 摘除队列', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  // get_new_blocks 覆盖（中弧）
  const gn = 'M 480 104 C 400 92, 280 92, 180 128';
  body.push(`<path class="dia-edge" d="${gn}"/>`);
  body.push(pathPulse(gn, { dur: 2, delay: 1.2 }));
  body.push(arrowHead(180, 128, -0.5, 1));
  body.push(cap(320, 86, 'get_new_blocks：清 hash 登记 · ref_cnt=1 · 内容将被覆盖', { size: 8.5, anchor: 'middle' }));
  // 无 hash 队列头立即复用（下弧）
  const reuse = 'M 480 246 C 380 282, 240 282, 150 184';
  body.push(`<path class="dia-edge" d="${reuse}"/>`);
  body.push(pathPulse(reuse, { dur: 2, delay: 1.6 }));
  body.push(arrowHead(150, 184, -0.5, -1));
  body.push(cap(320, 290, 'get_new_blocks 取队列头：立即覆盖复用', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 318, 'free_block_queue 双向链表：头 = LIFO 无缓存价值块，尾 = FIFO 缓存块按 LRU 驱逐；touch 命中即复活', { size: 10 }));

  return scene(720, 330, body, { label: '块池状态机：释放后按有无 hash 分流到 free 队列尾或头；touch 复活驱逐候选，get_new_blocks 才覆盖内容' }).svg();
}

/* ============================================================
 * 2.1 §5（block 3）— beam 共享演化与单块 COW（论文 Fig.9 语义）
 * ============================================================ */
function beamCow() {
  const UID = 'i2d';
  const body = [];

  body.push(cap(36, 24, 'beam 宽度 2 的共享演化（论文 Fig.9 语义）：继承前缀 = 复制块表映射 + ref_cnt+1，不搬 KV 字节', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 44, w: 210, h: 210, label: '第 t 步' }));
  body.push(cap(52, 94, 'A', { size: 10, weight: '650' }));
  body.push(chip(70, 78, 'P0', 'shared')); body.push(chip(112, 78, 'P1', 'shared')); body.push(chip(154, 78, 'a1', 'own'));
  body.push(cap(52, 154, 'B', { size: 10, weight: '650' }));
  body.push(chip(70, 138, 'P0', 'shared')); body.push(chip(112, 138, 'P1', 'shared')); body.push(chip(154, 138, 'b1', 'own'));
  body.push(cap(140, 190, 'P0 P1 共享 ref_cnt=2', { size: 8.5, anchor: 'middle' }));
  body.push(cap(140, 206, 'a1 / b1 各独占 ref_cnt=1', { size: 8.5, anchor: 'middle' }));
  body.push(cap(140, 232, 'prompt 块只存一份', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 258, y: 44, w: 210, h: 210, label: '第 t+1 步：B 分裂出 C' }));
  body.push(cap(274, 94, 'B', { size: 10, weight: '650' }));
  body.push(chip(292, 78, 'P0', 'shared')); body.push(chip(334, 78, 'P1', 'shared')); body.push(chip(376, 78, 'b1', 'shared'));
  body.push(cap(274, 154, 'C', { size: 10, weight: '650' }));
  body.push(chip(292, 138, 'P0', 'shared')); body.push(chip(334, 138, 'P1', 'shared')); body.push(chip(376, 138, "b1'", 'own', { lit: 1.4 }));
  body.push(edge([395, 108], [395, 130], { pulse: true, dur: 1, delay: 1.2, uid: UID, eid: 'cow' }));
  body.push(arrowHead(395, 134, 0, 1));
  body.push(cap(402, 124, '单块 COW', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(363, 190, 'C 块表 = B 逐项复制映射', { size: 8.5, anchor: 'middle' }));
  body.push(cap(363, 206, 'P0 P1 [b1] 各 ref_cnt+1', { size: 8.5, anchor: 'middle' }));
  body.push(cap(363, 232, '仅写共享半满块才拷贝一次', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 480, y: 44, w: 204, h: 210, label: '第 t+2 步：A 被淘汰' }));
  body.push(cap(496, 94, 'B', { size: 10, weight: '650' }));
  body.push(chip(514, 78, 'P0', 'shared')); body.push(chip(556, 78, 'P1', 'shared')); body.push(chip(598, 78, 'b1', 'own'));
  body.push(cap(496, 154, 'C', { size: 10, weight: '650' }));
  body.push(chip(514, 138, 'P0', 'shared')); body.push(chip(556, 138, 'P1', 'shared')); body.push(chip(598, 138, "b1'", 'own'));
  body.push(`<g opacity="0.35">` + chip(514, 176, 'a1', 'free') + `</g>`);
  body.push(cap(582, 208, 'A 淘汰：a1 ref_cnt→0 回池', { size: 8.5, anchor: 'middle' }));
  body.push(cap(582, 224, 'P0 P1 ref_cnt 只减 1 · 内容不动', { size: 8.5, anchor: 'middle' }));
  body.push(cap(582, 240, '共享块从未被写（命题 2）', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 282, '拷贝账（式 8）：copy = b·κL = 16 × 128 KiB = 2 MiB ≈ 0.63 μs；整段 prompt 拷贝 256 MiB ≈ 80 μs —— 差 128× = T_p/b', { mono: true, size: 10 }));
  body.push(cap(36, 300, 'n = 4、T_p = 2048：save = (n−1)·T_p·κL = 768 MiB = prompt KV 的 75%；beam 共享至多省 55%（论文 §6.3，workload 依赖）', { size: 10 }));
  body.push(cap(36, 318, '块边界结束的前缀全程零拷贝；COW 只发生在尾部半满共享块被写入时（OS fork 语义直译）', { size: 10 }));

  return scene(720, 336, body, { label: 'beam 共享演化：prompt 块共享一份，分裂只复制块表映射加引用计数，写共享半满块时触发单块 COW' }).svg();
}

/* ============================================================
 * 2.2 §3（block 0）— RadixAttention 基数树 + LRU 叶优先 + 命中路径
 * ============================================================ */
function radixTree() {
  const UID = 'i2e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'RadixAttention 基数树（论文 Fig.3 简化）：边标 = token 子序列，节点 value = KV 池索引（不是拷贝）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, ts = 11) => {
    const n = node({ id, x, y, w, h, role, title, titleSize: ts });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const root = mk('root', 40, 168, 64, 36, 'plain', 'root', 11);
  const A = mk('A', 160, 76, 130, 40, 'cache', 'system prompt A');
  const F = mk('F', 160, 180, 130, 40, 'cache', 'few-shot 示例');
  const B = mk('B', 160, 284, 130, 40, 'cache', 'system prompt B');
  const q1 = mk('q1', 350, 48, 110, 36, 'data', '第 1 轮问');
  const q2 = mk('q2', 350, 112, 110, 36, 'data', '第 2 轮问');
  const f1 = mk('f1', 350, 164, 110, 36, 'data', '问题 1');
  const f2 = mk('f2', 350, 220, 110, 36, 'data', '问题 2');
  const bq = mk('bq', 350, 284, 110, 36, 'data', '第 1 轮问');
  const a1 = mk('a1', 520, 48, 110, 36, 'attn', '第 1 轮答');
  const fa = mk('fa', 520, 164, 110, 36, 'attn', '答案 1');

  // 非命中边：kit 贝塞尔 + 脉冲（纵向主导边终点切线向下，横向向右）
  const tree = [[root, F], [root, B], [A, q2], [F, f1], [F, f2], [B, bq], [f1, fa]];
  tree.forEach(([a, b], i) => {
    const vert = Math.abs(b.left[1] - a.right[1]) > Math.abs(b.left[0] - a.right[0]);
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.6, delay: i * 0.18, uid: UID, eid: `e${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], vert ? 0 : 1, vert ? 1 : 0));
  });
  // 会话 1 第 2 轮命中路径：朱砂虚线流动 + 脉冲（单线，不与灰边重叠）
  const hit = ['M 104 186 C 130 186, 134 96, 160 96', 'M 290 96 C 316 96, 324 66, 350 66', 'M 460 66 C 486 66, 494 66, 520 66'];
  const hitTo = [A, q1, a1];
  hit.forEach((d, i) => {
    body.push(`<path class="sig-dashflow" d="${d}" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
    body.push(pathPulse(d, { dur: 1.6, delay: 0.3 + i * 0.4 }));
    body.push(arrowHead(hitTo[i].left[0], hitTo[i].left[1], 1, 0));
  });
  body.push(cap(400, 36, '会话 1 第 2 轮：全路径命中', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(480, 130, '…', { size: 12, anchor: 'middle' }));
  body.push(cap(480, 238, '…', { size: 12, anchor: 'middle' }));
  body.push(cap(480, 302, '…', { size: 12, anchor: 'middle' }));
  body.push(cap(560, 100, '叶 = LRU 驱逐候选', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(560, 214, 'lock_ref 锁 running 路径', { size: 9 }));

  body.push(cap(36, 330, 'LRU 叶优先：逐叶损害最小；父在孩子全逐后才成叶入堆 —— 共享祖先的命中被保留到最后一刻', { size: 10 }));
  body.push(cap(36, 348, 'match_prefix O(T)：整段匹配下行、部分匹配 _split_node（只拷索引 tensor，KV 池不动）；缓存与 running 共池', { size: 10 }));
  body.push(cap(36, 366, '缓存感知调度 LPM ≈ DFS 序：cache ≥ 最长请求时最优命中（贪心有饥饿风险，4.2）', { size: 10 }));

  return scene(720, 380, body, { label: '基数树：边标为 token 子序列、节点存 KV 池索引；会话命中走全路径，LRU 叶优先驱逐、lock_ref 保护 running 路径' }).svg();
}

/* ============================================================
 * 2.3 §1（block 0）— KV 字节三因子 × 三条路线节省账
 * ============================================================ */
function kvThreeFactors() {
  const UID = 'i2f';
  const body = [];

  body.push(cap(36, 24, 'KV 总字节 = κ × L × T × B：三个可砍因子对应三条路线（章脊 8B：κL = 128 KiB/token）', { size: 12, color: 'var(--ink-soft)' }));

  const fac = (id, x, w, role, title, sub) => {
    const n = node({ id, x, y: 64, w, h: 56, role, title, sub, mono: true, titleSize: 16, subSize: 9 });
    body.push(n.svg);
    return n;
  };
  const K = fac('K', 52, 120, 'cache', 'κ', '4096 B/token/层');
  body.push(cap(186, 98, '×', { size: 14, anchor: 'middle' }));
  fac('L', 204, 80, 'data', 'L', '32 层');
  body.push(cap(298, 98, '×', { size: 14, anchor: 'middle' }));
  const T = fac('T', 316, 120, 'data', 'T', '131072（满窗）');
  body.push(cap(450, 98, '×', { size: 14, anchor: 'middle' }));
  fac('B', 468, 80, 'data', 'B', '1');
  body.push(cap(562, 98, '=', { size: 14, anchor: 'middle' }));
  fac('S', 580, 104, 'attn', '16 GiB', '单序列满窗');

  body.push(`<line class="dia-edge dash" x1="112" y1="120" x2="112" y2="146"/>`);
  body.push(cap(112, 166, '量化：16 bit → q bit', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(112, 182, '（有界扰动 · token 全在）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(112, 202, '几何：MLA 576 元素（V3）', { size: 9.5, anchor: 'middle' }));
  body.push(cap(112, 218, '（训练期低秩 · 不可 retrofit）', { size: 8.5, anchor: 'middle' }));
  body.push(`<line class="dia-edge dash" x1="376" y1="120" x2="376" y2="146"/>`);
  body.push(cap(376, 166, '驱逐：T → ηT', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(376, 182, '（无界丢失 · 召回风险）', { size: 8.5, anchor: 'middle' }));

  const rows = [
    ['bf16 基线', 512, 'ink', '', 242],
    ['2 bit 量化', 64, 'seal', '2 GiB（理想 8×）', 264],
    ['η=20% 驱逐', 102, 'warm', '3.2 GiB（5×）', 286],
    ['先驱逐后量化', 13, 'seal', '0.4 GiB（η×q/16 = 1/40）', 308],
  ];
  rows.forEach(([lab, w, tone, val, y]) => {
    body.push(cap(52, y + 11, lab, { size: 9.5 }));
    body.push(`<g class="sig-bar" style="--i:${((y - 242) / 22 * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-${tone}" x="150" y="${y}" width="${w}" height="14" rx="3"/></g>`);
    if (val) body.push(cap(150 + w + 8, y + 11, val, { mono: true, size: 9 }));
  });
  body.push(cap(430, 275, 'scale 开销 16/(gq)：g=32→25%', { mono: true, size: 9 }));
  body.push(cap(430, 297, 'g=128→6.25%（式 3）', { mono: true, size: 9 }));
  body.push(cap(430, 319, '组合召回风险仍由驱逐侧定', { size: 9 }));

  body.push(cap(36, 344, '量化改「每元素存几字节」· 驱逐改「存几个元素」· MLA 改「元素是什么」—— 前两者运行时可切换，后者上线不可改', { size: 10 }));

  return scene(720, 356, body, { label: 'KV 字节账 κLTB 的三个可砍因子：量化砍每元素字节、驱逐砍 token 数、MLA 砍几何；节省账条对照' }).svg();
}

/* ============================================================
 * 2.3 §2（afterHeading）— 量化粒度格点：g=2 反例 × 持久 outlier 隔离
 * 原文：反例 token1=(0,1)、token2=(2,3)：per-token 界 1/6 < per-channel 界 1/3；
 * outlier M=8、m=1、q=2：per-token 界 ≥ 7/6 ≈ 1.17 vs per-channel 0.333
 * ============================================================ */
function quantLattice() {
  const UID = 'i2g';
  const body = [];

  body.push(cap(36, 24, '量化粒度 = 哪些元素共享一个极差 R：误差界 Δ/2 = R/(2(2^q−1)) 由 R 单独决定（q = 2）', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 左：g=2 反例 ---- */
  body.push(zone({ x: 36, y: 44, w: 320, h: 200, label: '反例 g=2：per-channel 反而输（横截划分）' }));
  body.push(cap(140, 80, '通道 1', { size: 9, anchor: 'middle' }));
  body.push(cap(220, 80, '通道 2', { size: 9, anchor: 'middle' }));
  body.push(cap(96, 106, 'token 1', { size: 9, anchor: 'end' }));
  body.push(cap(96, 146, 'token 2', { size: 9, anchor: 'end' }));
  const cell = (x, y, v, hot = false) =>
    `<rect x="${x}" y="${y}" width="64" height="32" rx="4" fill="${hot ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${hot ? 'var(--seal)' : '#c6c0af'}" stroke-width="1.1"/>` +
    `<text class="dia-sub dia-mono" x="${x + 32}" y="${y + 21}" text-anchor="middle" font-size="12" fill="var(--ink)">${v}</text>`;
  body.push(cell(108, 86, '0')); body.push(cell(188, 86, '1'));
  body.push(cell(108, 126, '2')); body.push(cell(188, 126, '3'));
  // per-token 行括弧（右）
  body.push(`<path d="M 260 86 h 8 v 32 h -8" fill="none" stroke="var(--seal)" stroke-width="1.3"/>`);
  body.push(cap(272, 102, 'R=1', { mono: true, size: 8.5 }));
  body.push(cap(272, 114, '界 1/6', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  body.push(`<path d="M 260 126 h 8 v 32 h -8" fill="none" stroke="var(--seal)" stroke-width="1.3"/>`);
  body.push(cap(272, 142, 'R=1', { mono: true, size: 8.5 }));
  body.push(cap(272, 154, '界 1/6', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  // per-channel 列括弧（下）
  body.push(`<path d="M 108 166 v 8 h 64 v -8" fill="none" stroke="var(--ink-soft)" stroke-width="1.3"/>`);
  body.push(cap(140, 188, 'R=2 · 界 1/3', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(`<path d="M 188 166 v 8 h 64 v -8" fill="none" stroke="var(--ink-soft)" stroke-width="1.3"/>`);
  body.push(cap(220, 188, 'R=2 · 界 1/3', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(196, 214, 'per-token 界 0.167 < per-channel 0.333', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(196, 230, '「per-channel 普适更好」被否证', { size: 9, anchor: 'middle' }));

  /* ---- 右：持久 outlier ---- */
  body.push(zone({ x: 376, y: 44, w: 308, h: 200, label: '持久 outlier（M ≥ 3m）：per-channel 隔离' }));
  body.push(cap(470, 80, '正常 1', { size: 8.5, anchor: 'middle' }));
  body.push(cap(540, 80, '正常 2', { size: 8.5, anchor: 'middle' }));
  body.push(cap(610, 80, 'outlier', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(436, 103, 't1', { size: 8.5, anchor: 'end' }));
  body.push(cap(436, 135, 't2', { size: 8.5, anchor: 'end' }));
  body.push(cap(436, 167, 't3', { size: 8.5, anchor: 'end' }));
  const vals = [['-1', '1', '8'], ['0', '-1', '9'], ['1', '0', '10']];
  vals.forEach((row, r) => row.forEach((v, c) => {
    const x = 442 + c * 70, y = 86 + r * 32;
    const inner =
      `<rect x="${x}" y="${y}" width="56" height="26" rx="4" fill="${c === 2 ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${c === 2 ? 'var(--seal)' : '#c6c0af'}" stroke-width="1.1"/>` +
      `<text class="dia-sub dia-mono" x="${x + 28}" y="${y + 17}" text-anchor="middle" font-size="10.5" fill="var(--ink)">${v}</text>`;
    // outlier 列逐 token 点亮 = 「持久」分布事实的状态填充
    body.push(c === 2 ? `<g class="sig-cell" style="--i:${0.3 + r * 0.3}">${inner}</g>` : inner);
  }));
  body.push(cap(470, 196, 'R=2→界 0.333', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(540, 196, 'R=2→界 0.333', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(610, 196, 'R=2→细 scale', { mono: true, size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(530, 214, 'per-token：R ≥ M−m = 7 → 界 ≥ 1.17', { size: 8.5, anchor: 'middle' }));
  body.push(cap(530, 230, '正常通道被 outlier 绑架（塌到两格点）', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 272, 'KIVI 分布事实：K 有持久 outlier 通道 → K per-channel；V 无通道结构 → V per-token（非对称 2 bit、tuning-free）', { size: 10 }));
  body.push(cap(36, 290, '章脊代入：127 正常通道 [−1,1]（m=1）+ 1 outlier [8,10]（M=8）：per-token 界 ≥ 7/6 ≈ 1.17 vs per-channel 0.333', { mono: true, size: 9.5 }));
  body.push(cap(36, 308, '最坏 token R=11：格点 −1, 2.67, 6.33, 10 —— 正常通道幅度信息尽失（相对界 92%）；per-channel 正常通道 Δ ≈ 0.667', { size: 9.5 }));
  body.push(cap(36, 326, '节省账：2 bit 理想 8×（128→16 KiB/token、16→2 GiB）；scale 开销 16/(gq)；尾部不足一组留全精度（residual）', { size: 9.5 }));
  body.push(cap(36, 344, '命题 1 只对 per-tensor 占优；对 per-token 无普适界 —— 粒度选择依据分布事实，不是定理', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 360, body, { label: '量化粒度格点：g=2 反例中 per-token 误差界更小；持久 outlier 分布下 per-channel 把正常通道误差限回自身极差' }).svg();
}

/* ============================================================
 * 2.4 §1（block 0）— 三层存储账 + 按 KV 类型分列放置 + swap 双向
 * ============================================================ */
function tieredStorage() {
  const UID = 'i2h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '三层存储账（GPU 侧口径）：按 KV 类型分列放置，不是统一热度 L1/L2/L3', { size: 12, color: 'var(--ink-soft)' }));

  const H = node({ id: 'H', x: 60, y: 56, w: 210, h: 64, role: 'attn', title: 'HBM', sub: '3.35 TB/s · 百 ns · 80 GB', mono: true, titleSize: 14, subSize: 9.5 });
  const D = node({ id: 'D', x: 60, y: 176, w: 210, h: 64, role: 'cache', title: 'host DRAM', sub: 'GPU 侧受 PCIe 封顶 · μs · TB', titleSize: 13, subSize: 9 });
  const S = node({ id: 'S', x: 60, y: 296, w: 210, h: 64, role: 'proj', title: 'SSD（3FS 集群）', sub: '≥40 GB/s 读 / 5 GB/s 写 · ms', titleSize: 13, subSize: 9 });
  [H, D, S].forEach((n) => { nodes.push(n); body.push(n.svg); });

  // HBM ↔ DRAM（D2H / H2D 双通道）
  body.push(`<path class="dia-edge" d="M 150 120 V 176"/>`);
  body.push(pathPulse('M 150 120 V 176', { dur: 1.6, delay: 0 }));
  body.push(arrowHead(150, 176, 0, 1));
  body.push(`<path class="dia-edge" d="M 180 176 V 120"/>`);
  body.push(pathPulse('M 180 176 V 120', { dur: 1.6, delay: 0.8 }));
  body.push(arrowHead(180, 120, 0, -1));
  body.push(cap(280, 140, 'PCIe Gen5 单向 ~64 GB/s', { mono: true, size: 9.5 }));
  body.push(cap(280, 156, 'D2H swap-out ↓ / H2D swap-in ↑', { size: 9 }));
  // DRAM ↔ SSD
  body.push(`<path class="dia-edge" d="M 150 240 V 296"/>`);
  body.push(pathPulse('M 150 240 V 296', { dur: 1.6, delay: 0.4 }));
  body.push(arrowHead(150, 296, 0, 1));
  body.push(`<path class="dia-edge" d="M 180 296 V 240"/>`);
  body.push(pathPulse('M 180 296 V 240', { dur: 1.6, delay: 1.2 }));
  body.push(arrowHead(180, 240, 0, -1));
  body.push(cap(280, 260, 'RDMA 网络 + 3FS 栈', { mono: true, size: 9.5 }));
  body.push(cap(280, 276, '单服务器 ≥40 GB/s 读（写 5 GB/s · 不对称 8×）', { size: 9 }));

  // 放置策略（右栏一）与容量/写路径（右栏二）
  body.push(cap(300, 80, 'in-flight block KV + state KV', { mono: true, size: 9.5 }));
  body.push(cap(300, 96, '（推理 GPU HBM）', { size: 9 }));
  body.push(cap(300, 200, 'SWA KV：单副本 · 不上 3FS', { size: 9.5 }));
  body.push(cap(300, 216, '跨机命中走 400G VLAN 拉取', { size: 9 }));
  body.push(cap(300, 320, 'paged block KV：专用 3FS 集群', { size: 9.5 }));
  body.push(cap(300, 336, '无应用层副本 · 靠 3FS 自身冗余', { size: 9 }));
  body.push(cap(560, 80, '逻辑容量 ~14.4 TB/GPU', { mono: true, size: 9 }));
  body.push(cap(560, 96, '集群 ~100 PiB SSD', { mono: true, size: 9 }));
  body.push(cap(560, 200, 'write-back best-effort：', { size: 9 }));
  body.push(cap(560, 216, '写可丢 · 读是正确性边界', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(560, 320, '20 × 30.73 TB/服务器', { mono: true, size: 9 }));
  body.push(cap(560, 336, '保留 24h 数据', { mono: true, size: 9 }));

  body.push(cap(36, 384, '容量账：8B 满窗 16 GiB/seq · B=32 → 512 GiB = 6.9× 单卡 HBM —— offload 把 batch 上限从「装得下」改绑「取回得及」', { size: 10 }));
  body.push(cap(36, 402, '延迟列 = GPU 侧访问口径（含 PCIe / 网络 / 文件系统栈）；介质本征另账：DRAM ~50–100 ns / NVMe ~10–100 μs', { size: 9.5 }));

  return scene(720, 412, body, { label: '三层存储：HBM、host DRAM、SSD 以 PCIe 与 RDMA 相连，swap-out 与 swap-in 双向流动，按 KV 类型分列放置' }).svg();
}

/* ============================================================
 * 2.4 §4（block 1）— 三段流水：3FS 读 / H2D GDR / forward 逐层重叠
 * ============================================================ */
function swapInPipeline() {
  const UID = 'i2i';
  const body = [];

  body.push(cap(36, 24, 'DeepInfer 三段流水：存储读 / H2D / forward 各占一种资源轴，互不抢占', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(52, 60, '时间 →', { size: 9.5 }));
  body.push(`<line class="dia-edge dash" x1="110" y1="56" x2="668" y2="56"/>`);

  const lanes = [
    ['3FS 读', '独立进程 · 网络+SSD', 76, 'proj', 190, 0],
    ['H2D (GDR)', '引擎线程 · PCIe', 146, 'cache', 272, 0.5],
    ['forward', 'SM + HBM', 216, 'attn', 354, 1.0],
  ];
  lanes.forEach(([lab, sub, y, role, x0, i0]) => {
    body.push(cap(52, y + 19, lab, { size: 10, weight: '650' }));
    body.push(cap(52, y + 33, sub, { size: 8 }));
    for (let l = 0; l < 4; l++) {
      const n = node({ id: `${lab}${l}`, x: x0 + l * 82, y, w: 64, h: 30, role, title: `L${l}`, mono: true, titleSize: 10 });
      body.push(`<g class="sig-cell" style="--i:${(i0 + l * 0.3).toFixed(1)}">${n.svg}</g>`);
    }
  });
  body.push(cap(510, 95, '…', { size: 12 }));
  body.push(cap(592, 165, '…', { size: 12 }));

  // 逐层依赖（数据沿三段前进）
  const deps = [
    [222, 106, 304, 146, 0.6], [304, 176, 386, 216, 1.1],
    [304, 106, 386, 146, 0.9], [386, 176, 468, 216, 1.4],
  ];
  deps.forEach(([ax, ay, bx, by, dl], i) => {
    body.push(edge([ax, ay], [bx, by], { pulse: true, dur: 1.4, delay: dl, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(bx, by, bx - ax, by - ay));
  });
  body.push(cap(560, 196, '第 i 层计算 ∥ 第 i+1 层传输', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 282, '逐层 KV = κT = 128 MiB：PCIe 传输 2.1 ms ≪ 每层 prefill 计算 ≈ 51 ms —— 传输完全藏进计算（2.2 式 4 同构）', { mono: true, size: 9.5 }));
  body.push(cap(36, 300, 'GDR：RDMA NIC 绕过 host memory 与 CPU 拷贝栈直写 HBM、不占 SM；copy engine 细粒度块拷贝差（b=64 → 16384 次小拷贝）', { size: 9.5 }));
  body.push(cap(36, 318, '匹配（CHECK，metadata only）在路由前、不可隐藏；取 KV 异步独立进程；swap-in 引擎线程逐层与 forward 重叠', { size: 9.5 }));

  return scene(720, 332, body, { label: '三段流水：3FS 读、H2D GDR、forward 三条资源轴逐层错开并行，第 i 层计算与第 i+1 层传输重叠' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/01-paged-attention',
    replacesBlock: 0,
    title: '连续三浪费 vs 分页：每请求浪费 ≤ 1 块',
    svg: contigVsPaged,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/01-paged-attention',
    replacesBlock: 1,
    title: '块表走读：逻辑→物理翻译 + decode 两步追加',
    svg: blockTableWalk,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/01-paged-attention',
    replacesBlock: 2,
    title: '块池状态机：ref_cnt=0 是驱逐候选，不是失效',
    svg: blockLifecycle,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/01-paged-attention',
    replacesBlock: 3,
    title: 'beam 共享演化：继承 = 复制映射，写时才单块 COW',
    svg: beamCow,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/02-prefix-caching',
    replacesBlock: 0,
    title: 'RadixAttention 基数树：命中路径 × LRU 叶优先',
    svg: radixTree,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/03-kv-quantization-and-eviction',
    replacesBlock: 0,
    title: 'KV 字节三因子：量化 × 驱逐 × MLA 的节省账',
    svg: kvThreeFactors,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/03-kv-quantization-and-eviction',
    afterHeading: /^## 2\. /,
    title: '量化粒度格点：g=2 反例 × 持久 outlier 隔离（q=2）',
    svg: quantLattice,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/04-kv-offload-and-tiering',
    replacesBlock: 0,
    title: '三层存储账：HBM / DRAM / SSD × 按 KV 类型放置',
    svg: tieredStorage,
  },
  {
    dim: 'inference',
    slug: '02-kv-cache-systems/04-kv-offload-and-tiering',
    replacesBlock: 1,
    title: '三段流水：3FS 读 × H2D GDR × forward 逐层重叠',
    svg: swapInPipeline,
  },
];
