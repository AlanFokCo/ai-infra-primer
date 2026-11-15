// Part 3（03-objectives-and-variants）场景图：替换三章共 7 个 ASCII 块（replacesBlock 口径见 CENSUS.md）
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 figures/part01.mjs。
// 每图自建 <uid>-pg 朱砂渐变供 edge 脉冲解析（同 part01 做法）。

import { scene, node, edge, zone, cap, matrix } from '../figure-kit.mjs';

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（L 形 / 大弧线通道，kit 只支持自动贝塞尔）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

// 小 token 芯片：角色浅染与 dia-node 同系；tone ∈ data / ghost / hot(朱砂·腐蚀位) / warm(琥珀·起始符)
const TOK = {
  data: ['#f3f1ea', '#c6c0af', 'var(--ink)'],
  ghost: ['var(--paper-sunk)', 'var(--rule)', 'var(--ink-faint)'],
  hot: ['var(--seal-wash)', 'var(--seal)', 'var(--seal-deep)'],
  warm: ['#f3ecdc', '#cbb98d', 'var(--ink)'],
};
const tok = (x, y, w, h, text, tone = 'data', fs = 10.5, cls = '', style = '') => {
  const [fill, stroke, color] = TOK[tone];
  return (
    `<g${cls ? ` class="${cls}"` : ''}${style ? ` style="${style}"` : ''}>` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="1.1"/>` +
    `<text class="dia-mono" x="${x + w / 2}" y="${y + h / 2 + fs * 0.36}" font-size="${fs}" fill="${color}" text-anchor="middle" font-weight="600">${esc(text)}</text></g>`
  );
};

// ℓ 小方格：1 = 朱砂（计入损失，sig-cell 亮起），0 = 虚线空
const lsq = (cx, y, on, i) =>
  on
    ? `<rect class="dia-mx-hot sig-cell" style="--i:${i.toFixed(2)}" x="${cx - 4.5}" y="${y}" width="9" height="9" rx="2"/>`
    : `<rect class="dia-mx-off" x="${cx - 4.5}" y="${y}" width="9" height="9" rx="2"/>`;

/* ============================================================
 * 3.1 §1（block 0）：同一条 8 token 序列在五种 objective 下的 (x′, ℓ, M)
 * 原文：CLM 原样/下三角/全 1；MLM 15% 选中 80/10/10、M≡0、ℓ 只落选中位；
 * prefix 前缀块双向+后缀因果、ℓ 落后缀；GLM Part A 双向 + Part B 乱序 span；
 * T5 sentinel（enc M≡0 / dec 因果+cross）、ℓ 只落 target
 * ============================================================ */
function fiveObjectives() {
  const UID = 'o5';
  const body = [pulseDefs(UID)];
  const X0 = 130;
  body.push(cap(36, 26, '同一条 8 token 序列在五种 objective 下的取值：腐蚀输入 x′ × loss mask ℓ × attention mask M', { size: 11, color: 'var(--ink-soft)' }));

  // 表头：原序列
  body.push(cap(36, 61, '原序列 x', { mono: true, size: 11, color: 'var(--ink)' }));
  const ORIG = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'slept'];
  ORIG.forEach((t, i) => body.push(tok(X0 + i * 56, 44, 52, 26, t, 'ghost')));

  const lanes = [
    {
      name: 'CLM', sub: '式 (2) · 零腐蚀', toks: ORIG, tone: () => 'data', ell: () => 1, step: 56, w: 52,
      grid: { n: 6, on: (i, j) => j <= i },
    },
    {
      name: 'MLM', sub: '式 (3) · 80/10/10', toks: ['the', '[M]', 'sat', 'on', 'the', 'mat', 'and', '[M]'],
      tone: (i) => (i === 1 || i === 7 ? 'hot' : 'data'), ell: (i) => (i === 1 || i === 7 ? 1 : 0), step: 56, w: 52,
      grid: { n: 6, on: () => true },
    },
    {
      name: 'prefix-LM', sub: '式 (7) · 损失落后缀', toks: ORIG, tone: () => 'data',
      ell: (i) => (i >= 4 ? 1 : 0), step: 56, w: 52,
      grid: { n: 6, on: (i, j) => j <= i || j < 3 },
      divider: { after: 3, left: '前缀 p=4 · 双向', right: '续写区 · 因果' },
    },
    {
      name: 'GLM', sub: '式 (8) · 乱序填空',
      toks: ['the', '[M]', 'on', 'the', 'mat', '[M]', 'slept', '[s]', 'cat', 'sat', '[s]', 'and'],
      tone: (i) => (i === 1 || i === 5 ? 'hot' : i === 7 || i === 10 ? 'warm' : 'data'),
      ell: (i) => (i === 8 || i === 9 || i === 11 ? 1 : 0), step: 36, w: 33, fs: 8.5,
      grid: { n: 6, on: (i, j) => (i < 3 && j < 3) || (i >= 3 && (j < 3 || j <= i)) },
      divider: { after: 6, left: 'Part A · 双向', right: 'Part B · 乱序拼接' },
    },
    {
      name: 'T5', sub: '式 (12) · sentinel',
      toks: ['the', '<X>', 'on', 'the', 'mat', '<Y>', 'slept', '<X>', 'cat', 'sat', '<Y>', 'and', '<Z>'],
      tone: (i) => ([1, 5, 7, 10, 12].includes(i) ? 'hot' : 'data'),
      ell: (i) => (i >= 7 ? 1 : 0), step: 33.5, w: 31, fs: 8.5, grids2: true,
      divider: { after: 6, left: 'enc · M≡0', right: 'dec · 因果 + cross' },
    },
  ];

  lanes.forEach((L, li) => {
    const y = 94 + li * 60;
    body.push(cap(34, y + 16, L.name, { mono: true, size: 12, color: 'var(--ink)', weight: 600 }));
    body.push(cap(34, y + 30, L.sub, { size: 9 }));
    L.toks.forEach((t, i) => body.push(tok(X0 + i * L.step, y, L.w, 26, t, L.tone(i), L.fs ?? 10.5)));
    body.push(cap(X0 - 8, y + 40, 'ℓ', { mono: true, size: 9.5, anchor: 'end' }));
    L.toks.forEach((_, i) => body.push(lsq(X0 + i * L.step + L.w / 2, y + 32, L.ell(i), li * 0.4 + i * 0.08)));
    if (L.divider) {
      const dx = X0 + (L.divider.after + 1) * L.step - 4;
      body.push(`<line class="dia-edge dash" x1="${dx}" y1="${y - 4}" x2="${dx}" y2="${y + 44}"/>`);
      body.push(cap(X0, y - 8, L.divider.left, { size: 9 }));
      body.push(cap(dx + 6, y - 8, L.divider.right, { size: 9 }));
    }
    if (L.grids2) {
      body.push(matrix({ x: 584, y: y + 16, n: 2, cell: 7, gap: 2, on: () => true }));
      body.push(matrix({ x: 610, y: y - 6, n: 4, cell: 7, gap: 2, on: (i, j) => j <= i }));
      body.push(cap(614, y + 44, 'enc / dec', { mono: true, size: 8.5, anchor: 'middle' }));
    } else {
      body.push(matrix({ x: 596, y: y - 6, n: L.grid.n, cell: 7, gap: 2, on: L.grid.on }));
    }
  });

  // 图例与注记
  body.push(cap(36, 392, 'M 网格 = 6×6 形状示意（实格 = 可见；T5 为 enc/dec 双网格）——与上方三格签名图同一语义', { size: 10 }));
  body.push(tok(130, 404, 30, 16, '[M]', 'hot', 8.5));
  body.push(cap(166, 416, '腐蚀位 [M] / sentinel', { size: 9.5 }));
  body.push(tok(330, 404, 26, 16, '[s]', 'warm', 8.5));
  body.push(cap(362, 416, 'span 起始符', { size: 9.5 }));
  body.push(lsq(452, 407, 1, 0));
  body.push(cap(462, 416, 'ℓ = 1 计损', { size: 9.5 }));
  body.push(`<rect class="dia-mx-off" x="546" y="407" width="9" height="9" rx="2"/>`);
  body.push(cap(562, 416, 'ℓ = 0', { size: 9.5 }));
  body.push(cap(36, 440, 'ℓ 支撑：CLM 全位置 · MLM 仅选中位（15%，此例 2/8）· prefix 后缀 · GLM 只落 span 内容 · T5 只落 target —— §8 分母差异', { size: 10 }));

  return scene(720, 452, body, { label: '五种 objective 在同一序列上的腐蚀输入、loss mask 与 attention mask 对照' }).svg();
}

/* ============================================================
 * 3.1 §4.3（block 1）：GLM 实例——spans [x2 x3]（起点 1）、[x5]（起点 4），π = 恒等
 * token: x1 [M] x4 [M] x6 | [s] x2 x3 [s] x5
 * p1: 0 1 2 3 4 | 1 1 1 3 3 ；p2: 0 0 0 0 0 | 0 1 2 0 1 ；ℓ 只落 span 内容
 * ============================================================ */
function glm2dLayout() {
  const UID = 'glm';
  const body = [pulseDefs(UID)];
  const X0 = 70, STEP = 58, W = 52;
  const cx = (i) => X0 + i * STEP + W / 2;
  let e = 0;
  const E = () => e++;

  body.push(cap(36, 24, 'GLM blank infilling：每条 span 在输入中塌缩成单个 [M]，乱序拼回 Part B；(p⁽¹⁾, p⁽²⁾) = 空白地址 × 空内偏移', { size: 11, color: 'var(--ink-soft)' }));

  // 顶行：原序列（span 暖色高亮 + 括号）
  const ORIG = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'];
  ORIG.forEach((t, i) => body.push(tok(X0 + i * STEP, 48, W, 26, t, (i === 1 || i === 2 || i === 4) ? 'warm' : 'ghost')));
  body.push(cap(56, 65, '原序列', { size: 10, anchor: 'end' }));
  const bracket = (x1, x2, label) =>
    `<path class="dia-edge" d="M${x1} 44 V40 H${x2} V44"/>` + cap((x1 + x2) / 2, 34, label, { size: 9, anchor: 'middle' });
  body.push(bracket(X0 + STEP, X0 + 2 * STEP + W, 'span · 起点 1'));
  body.push(bracket(X0 + 4 * STEP, X0 + 4 * STEP + W, 'span · 起点 4'));

  // 分区（先画，芯片压其上）
  body.push(zone({ x: 62, y: 100, w: 296, h: 140, label: 'Part A · 双向 · p⁽²⁾≡0' }));
  body.push(zone({ x: 366, y: 100, w: 286, h: 140, label: 'Part B · 乱序 span（π = 恒等）', tone: 'seal' }));

  // 塌缩边：原序列 span → Part A 的 [M]（虚线 + 脉冲）
  body.push(edge([cx(1), 74], [cx(1), 130], { dash: true, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: E(), arrow: false }));
  body.push(edge([cx(2), 74], [cx(1) + 6, 130], { dash: true, pulse: true, dur: 1.8, delay: 0.2, uid: UID, eid: E(), arrow: false }));
  body.push(edge([cx(4), 74], [cx(3) + 6, 130], { dash: true, pulse: true, dur: 1.8, delay: 0.4, uid: UID, eid: E(), arrow: false }));

  // 主行：拼接序列
  const MAIN = ['x1', '[M]', 'x4', '[M]', 'x6', '[s]', 'x2', 'x3', '[s]', 'x5'];
  MAIN.forEach((t, i) => {
    const tone = (i === 1 || i === 3) ? 'hot' : (i === 5 || i === 8) ? 'warm' : 'data';
    body.push(tok(X0 + i * STEP, 136, W, 32, t, tone));
  });
  body.push(cap(56, 156, 'token', { mono: true, size: 10, anchor: 'end' }));

  // p1 / p2 / ℓ 三行
  const P1 = [0, 1, 2, 3, 4, 1, 1, 1, 3, 3];
  const P2 = [0, 0, 0, 0, 0, 0, 1, 2, 0, 1];
  const ELL = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1];
  body.push(cap(56, 192, 'p⁽¹⁾', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(56, 210, 'p⁽²⁾', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(56, 228, 'ℓ', { mono: true, size: 10, anchor: 'end' }));
  P1.forEach((v, i) => body.push(cap(cx(i), 192, String(v), { mono: true, size: 10.5, anchor: 'middle', color: i >= 5 ? 'var(--seal-deep)' : 'var(--ink-faint)', weight: i >= 5 ? 600 : undefined })));
  P2.forEach((v, i) => body.push(cap(cx(i), 210, String(v), { mono: true, size: 10.5, anchor: 'middle', color: i >= 5 ? 'var(--seal-deep)' : 'var(--ink-faint)', weight: i >= 5 ? 600 : undefined })));
  let k = 0;
  ELL.forEach((v, i) => body.push(lsq(cx(i), 218, v === 1, 0.3 + (v ? k++ : i) * 0.25)));

  body.push(cap(36, 264, 'Part B 的 p⁽¹⁾ = 所填空白 [M] 的地址 k（此例 1 与 3），p⁽²⁾ = 空内偏移（起始符 0）——「填哪个空、空内第几」两维寻址（式 9/10）', { size: 10.5 }));
  body.push(cap(36, 282, '注入：e_t = E[x_t] + P⁽¹⁾[p⁽¹⁾] + P⁽²⁾[p⁽²⁾]（式 11）；Part A 的 p⁽²⁾≡0 兼任类型位', { size: 10.5 }));
  body.push(cap(36, 300, 'ℓ 只落 Part B 内容 token（x₂ x₃ x₅）；生成时在 [M] 位（k）起逐 token 产出 (k, j)，直到该 span 的结束符（§4.3）', { size: 10.5 }));

  return scene(720, 320, body, { label: 'GLM blank infilling 的拼接布局与二维位置编码实例' }).svg();
}

/* ============================================================
 * 3.1 §5.1（block 2）：T5 sentinel——被掩 span 物理移出输入，按序移入 target
 * 原序列 the cat sat on the mat and slept；[cat sat]→<X>、[and]→<Y>；<Z> 收尾
 * ============================================================ */
function t5Sentinel() {
  const UID = 't5s';
  const body = [pulseDefs(UID)];
  let e = 0;
  const E = () => e++;

  body.push(cap(36, 24, 'span corruption：选中 span 在 enc 输入中替换为 sentinel，被移除内容按序移入 dec target（§5.1）', { size: 11, color: 'var(--ink-soft)' }));

  // 行 1：原序列（span 暖色）
  const ORIG = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'slept'];
  ORIG.forEach((t, i) => body.push(tok(90 + i * 58, 44, 52, 26, t, (i === 1 || i === 2 || i === 6) ? 'warm' : 'ghost')));
  body.push(cap(84, 61, '原序列 x', { mono: true, size: 10.5, anchor: 'end' }));

  // 行 2：enc 输入
  body.push(zone({ x: 62, y: 92, w: 500, h: 68, label: 'enc 输入 · M_enc ≡ 0（双向）' }));
  const ENC = ['the', '<X>', 'on', 'the', 'mat', '<Y>', 'slept'];
  ENC.forEach((t, i) => body.push(tok(90 + i * 66, 122, 60, 28, t, (i === 1 || i === 5) ? 'hot' : 'data', 10)));
  const encCx = (i) => 90 + i * 66 + 30;

  // 行 3：dec target
  body.push(zone({ x: 62, y: 186, w: 500, h: 68, label: 'dec target · 因果 + cross-attn（式 12）', tone: 'seal' }));
  const DEC = ['<X>', 'cat', 'sat', '<Y>', 'and', '<Z>'];
  DEC.forEach((t, i) => body.push(tok(90 + i * 70, 216, 64, 28, t, (i === 0 || i === 3 || i === 5) ? 'hot' : 'data', 10)));
  const decCx = (i) => 90 + i * 70 + 32;

  // 塌缩边：原序列 span → enc sentinel
  body.push(edge([90 + 58 + 26, 70], [encCx(1) - 4, 118], { dash: true, pulse: true, dur: 1.7, delay: 0, uid: UID, eid: E(), arrow: false }));
  body.push(edge([90 + 2 * 58 + 26, 70], [encCx(1) + 4, 118], { dash: true, pulse: true, dur: 1.7, delay: 0.15, uid: UID, eid: E(), arrow: false }));
  body.push(edge([90 + 6 * 58 + 26, 70], [encCx(5) + 6, 118], { dash: true, pulse: true, dur: 1.7, delay: 0.3, uid: UID, eid: E(), arrow: false }));

  // 记录边：enc sentinel → dec（sentinel 定界虚线 + 内容移入朱砂粗线）
  body.push(cap(294, 178, '被移除内容按序移入 target', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge([encCx(1) - 3, 150], [decCx(0) + 4, 212], { dash: true, uid: UID, eid: E(), arrow: false }));
  body.push(edge([encCx(1) + 3, 150], [decCx(1), 212], { strong: true, pulse: true, dur: 2.0, delay: 0.5, uid: UID, eid: E(), arrow: false }));
  body.push(edge([encCx(1) + 7, 150], [decCx(2), 212], { strong: true, pulse: true, dur: 2.0, delay: 0.65, uid: UID, eid: E(), arrow: false }));
  body.push(edge([encCx(5) - 3, 150], [decCx(3) + 4, 212], { dash: true, uid: UID, eid: E(), arrow: false }));
  body.push(edge([encCx(5) + 3, 150], [decCx(4), 212], { strong: true, pulse: true, dur: 2.0, delay: 0.8, uid: UID, eid: E(), arrow: false }));

  body.push(cap(decCx(5), 268, '<Z> = 收尾 sentinel', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(36, 288, 'sentinel = 词表 extra_id 保留位（论文口径词表 32000；checkpoint 口径 +100 = 32100）', { size: 10 }));
  body.push(cap(36, 306, 'T=512 账：m=77、n_s=26 → enc 长 461、target 长 104 ≈ 20%——decoder 侧计算同比例压缩（§5.2）；ℓ 只落 dec 序列', { size: 10 }));

  return scene(720, 320, body, { label: 'T5 span corruption 的 sentinel 替换与 target 构造' }).svg();
}

/* ============================================================
 * 3.2 §7（block 0）：enc-dec 数据流（decode 第 t 步，逐层，省略 B/H 维）
 * x→encoder(M≡0,一次前向)→H^enc →(每层 W^K,W^V)→ K^enc,V^enc：一次计算全程复用
 * y_{<t}→causal self-attn(KV cache)→u_t→q_t；o_t = softmax(q_t K^encᵀ/√d)V^enc
 * ============================================================ */
function encDecFlow() {
  const UID = 'ed';
  const body = [pulseDefs(UID)];
  const nodes = [];
  let e = 0;
  const E = () => e++;

  body.push(cap(36, 24, 'decode 第 t 步（逐层，省略 B/H 维）：cross K/V 一次计算、全程复用；self KV cache 随 t 线性涨', { size: 11, color: 'var(--ink-soft)' }));

  // ---- encoder 区 ----
  body.push(zone({ x: 30, y: 46, w: 216, h: 314, label: 'encoder · M≡0 · 一次前向' }));
  ['x1', 'x2', '…', 'xS'].forEach((t, i) => body.push(tok(46 + i * 48, 84, 40, 28, t, t === '…' ? 'ghost' : 'data', 11)));
  const enc = node({ id: 'enc', x: 52, y: 140, w: 170, h: 44, role: 'attn', title: 'encoder stack', mono: true, titleSize: 12.5 });
  nodes.push(enc); body.push(enc.svg);
  body.push(cap(137, 178, 'M ≡ 0 · 全可见', { size: 9, anchor: 'middle' }));
  const hEnc = node({ id: 'henc', x: 52, y: 216, w: 170, h: 44, role: 'data', title: 'H^enc [S, d]', mono: true, titleSize: 12.5 });
  nodes.push(hEnc); body.push(hEnc.svg);
  body.push(cap(52, 290, '一次前向 · x 完整已知', { size: 9.5 }));
  body.push(cap(52, 306, 'M≡0 ⇒ 不可 AR 生成（§2）', { size: 9 }));
  [66, 114, 162, 210].forEach((x, i) => {
    body.push(edge([x, 112], [137, 134], { pulse: true, dur: 1.8, delay: i * 0.15, uid: UID, eid: E(), arrow: false }));
  });
  body.push(edge([137, 184], [137, 212], { pulse: true, strong: true, dur: 1.4, delay: 0.7, uid: UID, eid: E(), arrow: false }));

  // ---- 中央：cross K/V（定长 cache）----
  const kv = node({ id: 'kv', x: 300, y: 140, w: 200, h: 64, role: 'cache', title: 'K^enc, V^enc', sub: '[S, d_head] · 每层一份', mono: true, titleSize: 12.5 });
  nodes.push(kv); body.push(kv.svg);
  body.push(edge([222, 238], [296, 176], { pulse: true, strong: true, dur: 2.0, delay: 1.0, uid: UID, eid: E(), arrow: false }));
  body.push(cap(226, 200, 'W^K, W^V 逐层', { mono: true, size: 9 }));
  body.push(cap(384, 224, '一次计算 · 全程复用', { size: 9.5, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(384, 240, '定长 S · 不随 t 涨', { size: 9.5, anchor: 'end' }));

  // ---- decoder 区 ----
  body.push(zone({ x: 510, y: 46, w: 190, h: 314, label: 'decoder · 第 t 步' }));
  ['y1', 'y2', '…'].forEach((t, i) => body.push(tok(526 + i * 48, 84, 40, 28, t, t === '…' ? 'ghost' : 'data', 11)));
  const self = node({ id: 'self', x: 526, y: 140, w: 158, h: 40, role: 'attn', title: 'causal self-attn', mono: true, titleSize: 12 });
  const uT = node({ id: 'ut', x: 526, y: 196, w: 158, h: 36, role: 'data', title: 'u_t', mono: true, titleSize: 12 });
  const qt = node({ id: 'qt', x: 526, y: 248, w: 158, h: 36, role: 'proj', title: 'q_t = u_t · W^Q', mono: true, titleSize: 11.5 });
  [self, uT, qt].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(cap(605, 134, 'self KV cache 随 t 线性涨', { size: 9, anchor: 'middle' }));
  [546, 594, 642].forEach((x, i) => {
    body.push(edge([x, 112], [605, 134], { pulse: true, dur: 1.6, delay: 0.3 + i * 0.15, uid: UID, eid: E(), arrow: false }));
  });
  body.push(edge([605, 180], [605, 196], { pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: E(), arrow: false }));
  body.push(edge([605, 232], [605, 248], { pulse: true, dur: 1.2, delay: 1.1, uid: UID, eid: E(), arrow: false }));

  // ---- cross-attn 合成 ----
  const op = node({ id: 'op', x: 300, y: 248, w: 200, h: 48, role: 'attn', title: 'cross-attn → o_t', mono: true, titleSize: 12 });
  nodes.push(op); body.push(op.svg);
  body.push(cap(400, 290, 'softmax(q_t K^encᵀ /√d) V^enc', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--ink-faint)' }));
  body.push(edge([400, 204], [400, 244], { pulse: true, strong: true, dur: 1.6, delay: 1.3, uid: UID, eid: E(), arrow: false }));
  body.push(cap(414, 230, '每步读同一份', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(edge([526, 266], [504, 272], { pulse: true, dur: 1.4, delay: 1.6, uid: UID, eid: E(), arrow: false }));
  const out = node({ id: 'out', x: 300, y: 324, w: 200, h: 36, role: 'data', title: 'o_t → FFN → 下一层', mono: true, titleSize: 11.5 });
  nodes.push(out); body.push(out.svg);
  body.push(edge([400, 296], [400, 320], { pulse: true, dur: 1.4, delay: 2.0, uid: UID, eid: E(), arrow: false }));

  body.push(cap(36, 388, 'cross-attn 无 mask（可见集恒为 {1..S}）；decoder 唯一的 mask 是 self-attn 下三角 —— prefix-LM 块结构的两种参数化（§7 要点 2）', { size: 10 }));
  body.push(cap(36, 406, 'cache 账（T5-base，bf16，S=512）：cross ≈ 1.8×10⁷ B（≈18 MiB）恒定；self 在 t=512 同量级、此后线性超过（2.4 式 9 口径）', { size: 10 }));

  return scene(720, 420, body, { label: 'encoder-decoder 的 decode 数据流：cross K/V 定长复用，self cache 随步增长' }).svg();
}

/* ============================================================
 * 3.2 §10（block 1）：ModernBERT-base 层栈（22 层，global 周期 3）
 * global：M≡0 · RoPE θ=1.6×10⁵；local：总窗 128（±64）· θ=10⁴；
 * 输入 unpad 拼接 [ΣᵢTᵢ, 768]（cu_seqlens 边界）；输出 MLM head（tied，vocab 50368）
 * ============================================================ */
function modernBertStack() {
  const UID = 'mb';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 22, 'ModernBERT-base：22 层 × local/global 交替（global 周期 3）· unpad 入栈 · tied MLM head —— 形态不变（M≡0 + MLM），零件全换（§10）', { size: 11, color: 'var(--ink-soft)' }));

  // 顶部输出头与底部输入
  const outH = node({ id: 'mlm', x: 200, y: 32, w: 320, h: 36, role: 'proj', title: 'MLM head · tied Emb', sub: 'vocab 50368 · 同形表征', mono: true, titleSize: 12 });
  body.push(outH.svg);
  const inp = node({ id: 'inp', x: 200, y: 348, w: 320, h: 40, role: 'data', title: 'unpad 拼接 [ΣᵢTᵢ, 768]', sub: '边界 cu_seqlens · varlen', mono: true, titleSize: 12 });
  body.push(inp.svg);

  // 层栈（顶 → 底）：21 G / 20 L / 19 L / 18 G / ⋯ / 2 L / 1 L / 0 G
  const ROWS = [
    { label: 'layer 21', g: true, y: 80 },
    { label: 'layer 20', g: false, y: 114 },
    { label: 'layer 19', g: false, y: 114 + 34 },
    { label: 'layer 18', g: true, y: 148 + 34 },
    { label: 'layer 2', g: false, y: 252 },
    { label: 'layer 1', g: false, y: 286 },
    { label: 'layer 0', g: true, y: 320 },
  ];
  body.push(cap(360, 228, '⋮ 周期重复（G L L）', { size: 9.5, anchor: 'middle' }));
  // 自底向上的 sig-cell 亮起次序：layer 0 最先
  const order = [6, 5, 4, 3, 2, 1, 0];
  ROWS.forEach((r, idx) => {
    const fill = r.g ? '#f9eeea' : '#f0f0f4';
    const stroke = r.g ? '#cf9d92' : '#b6b6c4';
    const i = order[idx];
    body.push(
      `<g class="sig-cell" style="--i:${(0.15 + i * 0.18).toFixed(2)}">` +
      `<rect x="220" y="${r.y}" width="280" height="26" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.1"/>` +
      `<text class="dia-mono" x="234" y="${r.y + 17}" font-size="10.5" fill="var(--ink)">${r.label} · ${r.g ? 'global attn' : 'local attn'}</text>` +
      `<text class="dia-mono" x="488" y="${r.y + 17}" font-size="8.5" fill="var(--ink-faint)" text-anchor="end">${r.g ? 'θ = 1.6×10⁵' : 'θ = 10⁴'}</text>` +
      `</g>`
    );
  });

  // 左缘前向通道：输入 → 输出，逐层递送
  body.push(`<line class="dia-edge strong" x1="120" y1="368" x2="120" y2="50"/>`);
  body.push(pathPulse('M 120 368 V 50', { dur: 3.2, delay: 0, uid: UID }));
  [93, 127, 161, 195, 265, 299, 333].forEach((y) => {
    body.push(`<line class="dia-edge" x1="120" y1="${y}" x2="218" y2="${y}"/>`);
  });
  body.push(cap(120, 396, '前向 ↑', { size: 9.5, anchor: 'middle' }));

  // 右侧图例与字段账
  body.push(`<rect x="530" y="92" width="20" height="14" rx="4" fill="#f9eeea" stroke="#cf9d92" stroke-width="1.1"/>`);
  body.push(cap(556, 103, 'global attn（M≡0）· 全序列混合', { size: 9.5 }));
  body.push(cap(556, 119, 'RoPE θ = 1.6×10⁵（长波远程）', { size: 9 }));
  body.push(`<rect x="530" y="140" width="20" height="14" rx="4" fill="#f0f0f4" stroke="#b6b6c4" stroke-width="1.1"/>`);
  body.push(cap(556, 151, 'local attn · 总窗 128（每侧 64）', { size: 9.5 }));
  body.push(cap(556, 167, 'RoPE θ = 10⁴（高频分辨）', { size: 9 }));
  body.push(cap(530, 205, 'global 周期 3：层 0,3,…,21', { size: 9.5, color: 'var(--ink)' }));
  body.push(cap(530, 221, '（8 层 global / 14 层 local）', { size: 9 }));
  body.push(cap(530, 255, 'unpadding：attention 账', { size: 9.5, color: 'var(--ink)' }));
  body.push(cap(530, 271, 'B·T_max² → ΣᵢTᵢ²', { mono: true, size: 9 }));
  body.push(cap(530, 305, '22 层 · hidden 768 · 12 heads', { mono: true, size: 9 }));
  body.push(cap(530, 321, 'ctx 8192 · vocab 50368', { mono: true, size: 9 }));

  return scene(720, 400, body, { label: 'ModernBERT-base 层栈：local/global 交替与双 RoPE base' }).svg();
}

/* ============================================================
 * 3.3 §4（block 0）：s_K 布局（K = 2，情感分类例）
 * s_K = τ(x1,y1) ⊕ τ(x2,y2) ⊕ τ(x_q)；prefill 一次付清（cache 常驻），模型续写位产 ŷ
 * ============================================================ */
function iclLayout() {
  const UID = 'icl';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 24, 's_K = τ(x₁,y₁) ⊕ τ(x₂,y₂) ⊕ τ(x_q)（式 2 · K = 2，情感分类例）', { size: 11.5, color: 'var(--ink-soft)' }));

  const segCard = (x, header, l1, l2, l2tone) =>
    `<rect x="${x}" y="48" width="200" height="96" rx="12" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1.2"/>` +
    cap(x + 100, 68, header, { mono: true, size: 10, anchor: 'middle' }) +
    cap(x + 100, 94, l1, { size: 11.5, anchor: 'middle', color: 'var(--ink)' }) +
    cap(x + 100, 118, l2, { size: 11.5, anchor: 'middle', color: l2tone });
  body.push(segCard(40, 'τ(x₁, y₁)', '输入: 节奏拖沓', '标签: 负（y₁）', 'var(--seal-deep)'));
  body.push(segCard(256, 'τ(x₂, y₂)', '输入: 摄影极佳', '标签: 正（y₂）', 'var(--seal-deep)'));
  body.push(segCard(472, 'τ(x_q)', '输入: 值回票价', '标签: ____', 'var(--ink-faint)'));
  body.push(cap(248, 100, '⊕', { mono: true, size: 14, anchor: 'middle' }));
  body.push(cap(464, 100, '⊕', { mono: true, size: 14, anchor: 'middle' }));

  // prefill 区：整段前缀一次编码，KV cache 常驻会话
  body.push(zone({ x: 40, y: 156, w: 632, h: 56, label: 'prefill 一次付清 · s_K 的 KV cache 常驻会话' }));
  body.push(cap(56, 196, '2·L·H_kv·d_head·|s_K|·b 字节（2.4 式 9，B=1）· 同一会话后续 query 复用同一份 cache（§2）', { mono: true, size: 9 }));

  // 续写位
  const yhat = node({ id: 'yhat', x: 536, y: 240, w: 120, h: 44, role: 'attn', title: 'ŷ = 正', sub: 'argmax / 采样', mono: true, titleSize: 13 });
  body.push(yhat.svg);
  body.push(edge([572, 144], [596, 236], { pulse: true, strong: true, dur: 1.8, delay: 0.4, uid: UID, eid: 0, arrow: false }));
  body.push(cap(596, 300, '模型续写位：ŷ ~ p_θ(· | s_K)', { mono: true, size: 9.5, anchor: 'middle' }));

  body.push(cap(40, 246, '无梯度：θ 逐 bit 不变（§4）', { size: 10, color: 'var(--ink)' }));
  body.push(cap(40, 264, '信息通道只有 token —— 无任务 embedding 旁路', { size: 10 }));
  body.push(cap(40, 282, '「学习」只发生在激活：KV cache + residual stream（§2）', { size: 10 }));
  body.push(cap(40, 308, 'K=0 zero-shot · K=1 one-shot · K≥2 few-shot；多数任务 few>one>zero，差距随规模扩大（GPT-3 报告）', { size: 10 }));

  return scene(720, 322, body, { label: 'ICL 的 prompt 布局：两段演示加查询，prefill 一次付清，续写位产出标签' }).svg();
}

/* ============================================================
 * 3.3 §6（block 1）：induction head 两层回路
 * 第 1 层 previous-token head：A⁽¹⁾_{j+1,j} ≈ 1 拷贝前驱（式 6）；
 * 第 2 层 induction head：query(t)「我是 A」prefix-match key(i+1)「前驱是 A」，
 * value(i+1)=B copy 写入 t 的 stream → W_U 抬升 logit(B)（式 7）
 * ============================================================ */
function inductionLayout() {
  const UID = 'ind';
  const body = [pulseDefs(UID)];
  let e = 0;
  const E = () => e++;

  body.push(cap(36, 22, '任务原型：[A][B] … [A] 之后该接 [B]——「上次 A 后面跟的是 B」，两层两头分工（式 6/7）', { size: 11, color: 'var(--ink-soft)' }));

  // 第 2 层区（上）
  body.push(zone({ x: 40, y: 36, w: 660, h: 128, label: '第 2 层 · induction head（在位置 t 工作）' }));
  const lg = node({ id: 'lg', x: 572, y: 68, w: 120, h: 36, role: 'attn', title: 'logit(B) ↑', sub: 'W_U 抬升', mono: true, titleSize: 12.5 });
  const qc = node({ id: 'qc', x: 408, y: 72, w: 120, h: 36, role: 'proj', title: 'query(t)', sub: '我是 A', mono: true, titleSize: 12.5 });
  const kc = node({ id: 'kc', x: 250, y: 72, w: 140, h: 36, role: 'cache', title: 'key(i+1)', sub: '前驱是 A', mono: true, titleSize: 12.5 });
  [lg, qc, kc].forEach((n) => body.push(n.svg));
  body.push(cap(399, 66, 'prefix match', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge([408, 90], [392, 90], { dash: true, strong: true, pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: E(), arrow: false }));

  // 序列行（y=180）：… [A]_i [B]_{i+1} … [A]_t [B]̂
  const SEQ = [null, '[A]', '[B]', null, '[A]', '[B]'];
  SEQ.forEach((t, i) => {
    const cxI = 140 + i * 74 + 32;
    if (t === null) {
      body.push(cap(cxI, 199, '…', { mono: true, size: 14, anchor: 'middle' }));
    } else {
      const tone = i === 4 ? 'warm' : i === 5 ? 'hot' : 'data';
      const cls = i === 5 ? 'sig-cell' : '';
      body.push(tok(140 + i * 74, 180, 64, 30, t, tone, 12, cls, i === 5 ? '--i:1.6' : ''));
    }
  });
  body.push(cap(246, 224, 'i', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(320, 224, 'i+1', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(468, 224, 't', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(542, 224, 't+1（预测）', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(320, 240, 'stream：我是 B + 前驱是 A', { size: 8.5, anchor: 'middle' }));

  // q / k 竖边（芯片 → 角色卡）
  body.push(edge([468, 180], [468, 112], { pulse: true, dur: 1.2, delay: 0.4, uid: UID, eid: E(), arrow: false }));
  body.push(edge([320, 180], [320, 112], { pulse: true, dur: 1.2, delay: 0.2, uid: UID, eid: E(), arrow: false }));

  // copy：value(i+1)=B 写入 t 的 stream（朱砂弧，行于角色卡之下）
  const copyPath = 'M 336 178 C 380 120, 420 120, 460 178';
  body.push(`<path class="dia-edge strong" style="stroke: rgb(168 50 40 / 0.55)" d="${copyPath}"/>`);
  body.push(pathPulse(copyPath, { dur: 2.0, delay: 0.9, uid: UID }));
  body.push(cap(56, 150, 'copy：value(i+1) = B → 写入 t 的 stream', { size: 9.5, color: 'var(--seal-deep)' }));

  // t 的 stream → logit(B)（W_U 读出）
  const lgPath = 'M 484 178 C 560 168, 600 128, 632 104';
  body.push(`<path class="dia-edge" d="${lgPath}"/>`);
  body.push(pathPulse(lgPath, { dur: 1.8, delay: 1.4, uid: UID }));

  // 第 1 层区（下）：previous-token head
  body.push(zone({ x: 40, y: 258, w: 660, h: 56, label: '第 1 层 · previous-token head' }));
  const prevPath = 'M 246 210 C 246 302, 320 302, 320 210';
  body.push(`<path class="dia-edge dash" d="${prevPath}"/>`);
  body.push(pathPulse(prevPath, { dur: 1.6, delay: 0, uid: UID, r: 3.6 }));
  body.push(cap(352, 306, 'A⁽¹⁾_{j+1, j} ≈ 1：把前驱表征拷进当前 stream（式 6）', { size: 9 }));

  body.push(cap(36, 336, 'prefix matching：QK 组合双线性型 = 「token 相等」检测器（式 7）；copying：OV 搬运 B 的表征', { size: 9.5 }));
  body.push(cap(36, 352, '测量：随机序列重复两遍，induction score = pattern 落在偏移对角线的 mass（TransformerLens）；Pythia：与 few-shot 分数同步起跳（§8）', { size: 9.5 }));

  return scene(720, 364, body, { label: 'induction head 两层回路：前驱拷贝、前缀匹配、复制写入、logit 抬升' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '03-objectives-and-variants/01-objectives-and-masks',
    replacesBlock: 0,
    title: '五种 objective：同序列 (x′, ℓ, M) 对照',
    svg: fiveObjectives,
  },
  {
    slug: '03-objectives-and-variants/01-objectives-and-masks',
    replacesBlock: 1,
    title: 'GLM：span 塌缩与二维位置 (p⁽¹⁾, p⁽²⁾)',
    svg: glm2dLayout,
  },
  {
    slug: '03-objectives-and-variants/01-objectives-and-masks',
    replacesBlock: 2,
    title: 'T5 span corruption：sentinel 替换',
    svg: t5Sentinel,
  },
  {
    slug: '03-objectives-and-variants/02-encoder-architectures',
    replacesBlock: 0,
    title: 'enc-dec 数据流：cross K/V 一次计算、全程复用',
    svg: encDecFlow,
  },
  {
    slug: '03-objectives-and-variants/02-encoder-architectures',
    replacesBlock: 1,
    title: 'ModernBERT-base 层栈：local/global 交替',
    svg: modernBertStack,
  },
  {
    slug: '03-objectives-and-variants/03-decoder-only-and-icl',
    replacesBlock: 0,
    title: 's_K 布局：prefill 一次付清',
    svg: iclLayout,
  },
  {
    slug: '03-objectives-and-variants/03-decoder-only-and-icl',
    replacesBlock: 1,
    title: 'induction head：prefix match + copy 布局',
    svg: inductionLayout,
  },
];
