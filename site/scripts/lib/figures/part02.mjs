// Part 2（02-transformer-anatomy）场景图：替换各章需画块的 ASCII 网格
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准见 part01.mjs。
// 跳过：04 block 0（prefill/decode 伪代码 → 算法面板）、05 block 1（yarn config JSON）。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// 脉冲渐变由页面级 PULSE_GRAD_DEFS 注入（Chapter.astro / fig-check），edge 与手工路径共用同一 id
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

// 残差加和圆圈
const addDot = (cx, cy) =>
  `<circle cx="${cx}" cy="${cy}" r="9" fill="var(--paper)" stroke="var(--ink-soft)" stroke-width="1.3"/>` +
  `<text class="dia-mono" x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" fill="var(--ink)">+</text>`;

// 双向箭头（encoder 的 ↕）
const darrow = (x, y1, y2) =>
  `<line class="dia-edge" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}"/>` +
  `<polygon points="${x},${y1} ${x - 3.5},${y1 + 6} ${x + 3.5},${y1 + 6}" fill="var(--ink-soft)"/>` +
  `<polygon points="${x},${y2} ${x - 3.5},${y2 - 6} ${x + 3.5},${y2 - 6}" fill="var(--ink-soft)"/>`;

/* ============================================================
 * 2.1 §1 — 一次 forward 全景（block 0）
 * 原文：ids [B,T] → E gather（无乘加）→ x⁰ [B,T,d] → ×L（Attn 跨位置 / FFN 逐位置）
 * → final RMSNorm → z = W_U·x [B,T,V] → softmax → q_θ ∈ Δ^{V−1}
 * ============================================================ */
function overviewForward() {
  const UID = 'p2ov';
  const body = [];
  const nodes = [];

  body.push(cap(40, 20, '一次 forward：token ids → logits → 逐位置分布（2.1 式 (1) 的整机形态）', { size: 12, color: 'var(--ink-soft)' }));

  const ids = node({ id: 'ids', x: 270, y: 32, w: 180, h: 36, role: 'data', title: 'token ids x₁..x_T', sub: '[B, T]', mono: true, titleSize: 12.5, subSize: 10 });
  const emb = node({ id: 'emb', x: 270, y: 92, w: 180, h: 42, role: 'cache', title: 'E ∈ R^{V×d}', sub: '查表 gather · 无乘加', mono: true, titleSize: 12.5, subSize: 10 });
  const x0 = node({ id: 'x0', x: 270, y: 158, w: 180, h: 40, role: 'data', title: 'x⁽⁰⁾ [B, T, d]', sub: 'residual stream 起点', mono: true, titleSize: 12.5, subSize: 10 });
  [ids, emb, x0].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(ids, emb, { nodes, portFrom: ids.bottom, portTo: emb.top, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'e1' }));
  body.push(edge(emb, x0, { nodes, portFrom: emb.bottom, portTo: x0.top, pulse: true, dur: 1.5, delay: 0.2, uid: UID, eid: 'e2' }));

  // decoder layer zone：trunk 在左、两个子层读写分支
  body.push(zone({ x: 90, y: 218, w: 540, h: 168, label: 'decoder layer ℓ · 同一结构重复 L 次' }));
  body.push(cap(500, 236, '式 (1)：x ← x + f(LN(x))', { mono: true, size: 9.5, anchor: 'end' }));
  // x0 → trunk 入口肘
  body.push(`<path class="dia-edge" d="M 360 198 V 208 H 140 V 236"/>`);
  body.push(pathPulse('M 360 198 V 208 H 140 V 236', { dur: 1.6, delay: 0.35 }));
  // trunk
  body.push(`<line class="dia-edge strong" x1="140" y1="236" x2="140" y2="386"/>`);
  body.push(pathPulse('M 140 236 V 386', { dur: 3.2, delay: 0.5, r: 3.4 }));
  // 子层 1：RMSNorm₁ → Attn（跨位置混合）
  const rn1 = node({ id: 'rn1', x: 190, y: 244, w: 110, h: 42, role: 'norm', title: 'RMSNorm₁', titleSize: 12.5 });
  const att = node({ id: 'att', x: 350, y: 244, w: 160, h: 42, role: 'attn', title: 'Attn_ℓ', sub: '唯一跨位置混合', mono: true, subSize: 10 });
  // 子层 2：RMSNorm₂ → FFN（逐位置变换）
  const rn2 = node({ id: 'rn2', x: 190, y: 312, w: 110, h: 42, role: 'norm', title: 'RMSNorm₂', titleSize: 12.5 });
  const ffn = node({ id: 'ffn', x: 350, y: 312, w: 160, h: 42, role: 'op', title: 'FFN_ℓ', sub: '逐位置变换', mono: true, subSize: 10 });
  [rn1, att, rn2, ffn].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([140, 265], rn1, { nodes, portTo: rn1.left, uid: UID, eid: 't1', arrow: false }));
  body.push(edge(rn1, att, { nodes, pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 'a1' }));
  body.push(edge([140, 333], rn2, { nodes, portTo: rn2.left, uid: UID, eid: 't2', arrow: false }));
  body.push(edge(rn2, ffn, { nodes, pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'f1' }));
  // 写回：子层输出 → 右路折返 → trunk 上的 ⊕
  const w1 = 'M 510 265 H 566 V 288 H 149';
  const w2 = 'M 510 333 H 566 V 368 H 149';
  body.push(`<path class="dia-edge" d="${w1}"/>`);
  body.push(`<path class="dia-edge" d="${w2}"/>`);
  body.push(pathPulse(w1, { dur: 2.0, delay: 1.0 }));
  body.push(pathPulse(w2, { dur: 2.0, delay: 1.8 }));
  body.push(addDot(140, 288), addDot(140, 368));

  // 底部输出行（横向）：final RMSNorm → W_U → q_θ
  body.push(`<path class="dia-edge" d="M 140 386 V 424 H 170"/>`);
  body.push(pathPulse('M 140 386 V 424 H 170', { dur: 1.6, delay: 2.1 }));
  const fn = node({ id: 'fn', x: 170, y: 424, w: 130, h: 44, role: 'norm', title: 'final RMSNorm', sub: 'x⁽ᶠ⁾ [B, T, d]', mono: true, titleSize: 10.5, subSize: 9 });
  const wu = node({ id: 'wu', x: 340, y: 424, w: 150, h: 44, role: 'proj', title: 'z = W_U · x', sub: 'logits [B, T, V]', mono: true, titleSize: 12, subSize: 9.5 });
  const q = node({ id: 'q', x: 516, y: 424, w: 186, h: 44, role: 'proj', title: 'q_θ(· | x_{<t})', sub: '∈ Δ^{V−1} · softmax 逐位置', mono: true, titleSize: 11.5, subSize: 9 });
  [fn, wu, q].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(fn, wu, { nodes, pulse: true, dur: 1.4, delay: 2.3, uid: UID, eid: 'o1' }));
  body.push(edge(wu, q, { nodes, pulse: true, dur: 1.4, delay: 2.5, uid: UID, eid: 'o2' }));

  body.push(cap(40, 500, '结构二分：attention 是唯一跨位置（沿 T 维）混合信息的算子；embedding、norm、FFN、输出头、残差加全部逐位置独立', { size: 11 }));
  body.push(cap(40, 522, '参数只住三处：E、每层两个子层内部的投影矩阵、W_U —— tie = false 时 E 与 W_U 是独立的两份 V×d（2.2 §9）', { size: 11 }));

  return scene(720, 540, body, { label: '一次 forward 全景：ids 经 embedding、L 层读写 residual stream、输出头到逐位置分布' }).svg();
}

/* ============================================================
 * 2.1 §7 — 三种架构形态（block 1）
 * encoder-only（M≡0 双向）/ encoder-decoder（cross-attn）/ decoder-only（因果）
 * ============================================================ */
function threeForms() {
  const UID = 'p2tf';
  const body = [];
  const nodes = [];

  body.push(cap(24, 20, '同一套 residual stream + attention 组件，按可见性 mask 与连接方式组装成三种形态（§7）', { size: 12, color: 'var(--ink-soft)' }));

  const ZX = [24, 252, 480], ZW = 216;
  body.push(zone({ x: ZX[0], y: 40, w: ZW, h: 232, label: 'encoder-only · M ≡ 0 双向' }));
  body.push(zone({ x: ZX[1], y: 40, w: ZW, h: 232, label: 'encoder-decoder' }));
  body.push(zone({ x: ZX[2], y: 40, w: ZW, h: 232, label: 'decoder-only · M = 下三角' }));

  // --- A：encoder-only：token ↕ 层 ↕ 每位置向量 ---
  const ax = [46, 88, 130, 172];
  ax.forEach((x, i) => {
    const n = node({ id: `a_t${i}`, x, y: 74, w: 34, h: 28, role: 'data', title: `x${i + 1}`, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
  });
  const encA = node({ id: 'encA', x: 52, y: 150, w: 144, h: 44, role: 'op', title: '[enc 层 ×L]', mono: true, titleSize: 12 });
  const outA = node({ id: 'outA', x: 52, y: 216, w: 144, h: 40, role: 'proj', title: '每位置向量', sub: '分类 / 检索', titleSize: 11.5, subSize: 10 });
  [encA, outA].forEach((n) => { nodes.push(n); body.push(n.svg); });
  ax.forEach((x) => body.push(darrow(x + 17, 106, 146)));
  body.push(darrow(124, 198, 212));

  // --- B：encoder-decoder：src → enc ⇢（cross-attn）dec ← tgt ---
  const src = node({ id: 'src', x: 268, y: 84, w: 56, h: 34, role: 'data', title: 'src', mono: true, titleSize: 11.5 });
  const encB = node({ id: 'encB', x: 348, y: 80, w: 104, h: 42, role: 'op', title: '[enc ×L]', sub: 'M ≡ 0', mono: true, titleSize: 11.5, subSize: 10 });
  const tgt = node({ id: 'tgt', x: 268, y: 170, w: 56, h: 34, role: 'data', title: 'tgt', mono: true, titleSize: 11.5 });
  const decB = node({ id: 'decB', x: 348, y: 166, w: 104, h: 42, role: 'op', title: '[dec ×L]', sub: '下三角 + cross', mono: true, titleSize: 11.5, subSize: 9.5 });
  [src, encB, tgt, decB].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(src, encB, { nodes, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 'b1' }));
  body.push(edge(tgt, decB, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 'b2' }));
  body.push(edge(encB, decB, { nodes, portFrom: encB.bottom, portTo: decB.top, pulse: true, dur: 1.8, delay: 0.6, uid: UID, eid: 'b3', strong: true }));
  body.push(cap(462, 128, 'cross-attn：', { size: 9, anchor: 'end' }));
  body.push(cap(462, 140, 'K/V 来自 enc 输出', { size: 9, anchor: 'end' }));
  body.push(cap(400, 236, '▼ 预测目标序列', { size: 10, anchor: 'middle' }));

  // --- C：decoder-only：x1→x2→x3→x4 → dec → 下一 token ---
  const cx = [502, 546, 590, 634];
  cx.forEach((x, i) => {
    const n = node({ id: `c_t${i}`, x, y: 74, w: 34, h: 28, role: 'data', title: `x${i + 1}`, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    if (i < 3) body.push(edge([x + 34, 88], [x + 44, 88], { nodes, pulse: true, dur: 1.4, delay: i * 0.25, uid: UID, eid: `c${i}` }));
  });
  const decC = node({ id: 'decC', x: 510, y: 150, w: 144, h: 44, role: 'op', title: '[dec 层 ×L]', mono: true, titleSize: 12 });
  nodes.push(decC); body.push(decC.svg);
  cx.forEach((x, i) => body.push(edge([x + 17, 102], [x + 17, 150], { nodes, pulse: true, dur: 1.5, delay: 0.4 + i * 0.15, uid: UID, eid: `cd${i}`, arrow: false })));
  body.push(cap(582, 226, '▼ 预测下一 token', { size: 10, anchor: 'middle' }));

  // 形态脚注
  body.push(cap(132, 296, 'BERT 家族', { size: 10.5, anchor: 'middle' }));
  body.push(cap(360, 296, 'T5 · Transformer 原文（翻译）', { size: 10.5, anchor: 'middle' }));
  body.push(cap(588, 296, 'GPT · Llama · OLMo · Qwen · DeepSeek', { size: 9.5, anchor: 'middle', mono: true }));
  body.push(cap(24, 318, '训练信号密度：CLM 对长度 T 的序列给出 T 个监督位置 —— 同样的文本预算下 decoder-only 榨出的梯度信号最多', { size: 10.5 }));

  return scene(720, 330, body, { label: '三种架构形态：encoder-only 双向、encoder-decoder 带 cross-attn、decoder-only 因果' }).svg();
}

/* ============================================================
 * 2.2 §1 + §2 — 字符串 → bytes → tokens → embedding（block 0）
 * 副板：BPE 四轮 merge 语料总长 79→47（手算例真实数字）
 * ============================================================ */
function tokenPipeline() {
  const UID = 'p2tk';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, '从字符串到向量：数据进模型的第一段链路（§1）；tokenizer 训练 = 一次性压缩问题求解（§2）', { size: 12, color: 'var(--ink-soft)' }));

  const s1 = node({ id: 's1', x: 28, y: 52, w: 104, h: 52, role: 'data', title: '"newest"', sub: '原始字符串', mono: true, titleSize: 12, subSize: 10 });
  const s2 = node({ id: 's2', x: 182, y: 52, w: 196, h: 52, role: 'data', title: '[110, 101, 119, 101, 115, 116]', sub: 'byte 序列 · 256 基元', mono: true, titleSize: 10, subSize: 9.5 });
  const s3 = node({ id: 's3', x: 428, y: 52, w: 116, h: 52, role: 'data', title: '["new", "est"]', sub: 'token → id', mono: true, titleSize: 10.5, subSize: 10 });
  const s4 = node({ id: 's4', x: 594, y: 52, w: 112, h: 52, role: 'cache', title: 'e_t = E_{id,:}', sub: 'gather · 无乘加', mono: true, titleSize: 10.5, subSize: 9.5 });
  const s5 = node({ id: 's5', x: 594, y: 140, w: 120, h: 48, role: 'proj', title: 'embedding [T, d]', sub: 'stream 起点（2.1 式 1）', mono: true, titleSize: 10, subSize: 8.5 });
  [s1, s2, s3, s4, s5].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(s1, s2, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'p1' }));
  body.push(cap(157, 44, 'UTF-8 编码', { size: 9.5, anchor: 'middle' }));
  body.push(edge(s2, s3, { nodes, pulse: true, dur: 1.6, delay: 0.25, uid: UID, eid: 'p2' }));
  body.push(cap(403, 44, 'BPE 切分', { size: 9.5, anchor: 'middle' }));
  body.push(edge(s3, s4, { nodes, pulse: true, dur: 1.6, delay: 0.5, uid: UID, eid: 'p3' }));
  body.push(cap(569, 44, '查表', { size: 9.5, anchor: 'middle' }));
  body.push(edge(s4, s5, { nodes, portFrom: s4.bottom, portTo: s5.top, pulse: true, dur: 1.4, delay: 0.75, uid: UID, eid: 'p4' }));

  // 副板：BPE 贪心压缩（四轮真实手算例：79 → 70 → 61 → 54 → 47）
  body.push(zone({ x: 28, y: 140, w: 556, h: 164, label: 'BPE 训练 = 贪心压缩（§2 手算例）' }));
  const rounds = [
    { v: 79, m: '—', tag: '轮 0' },
    { v: 70, m: 'e+s→es', tag: '轮 1 · 9 次' },
    { v: 61, m: 'es+t→est', tag: '轮 2 · 9 次' },
    { v: 54, m: 'l+o→lo', tag: '轮 3 · 7 次' },
    { v: 47, m: 'lo+w→low', tag: '轮 4 · 7 次' },
  ];
  const base = 272, scale = 1.05;
  rounds.forEach((r, i) => {
    const bx = 216 + i * 74, bh = r.v * scale;
    body.push(`<rect class="dia-bar dia-bar-soft sig-bar" style="--i:${(i * 0.35).toFixed(2)}" x="${bx}" y="${base - bh}" width="52" height="${bh}" rx="3"/>`);
    body.push(cap(bx + 26, base - bh - 6, String(r.v), { mono: true, size: 10, anchor: 'middle', color: 'var(--ink)' }));
    body.push(cap(bx + 26, base + 14, r.m, { mono: true, size: 9, anchor: 'middle' }));
    body.push(cap(bx + 26, base + 27, r.tag, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  body.push(`<line class="dia-edge" x1="204" y1="${base}" x2="570" y2="${base}"/>`);
  body.push(cap(198, 258, '语料符号总长', { size: 9, anchor: 'end' }));

  body.push(cap(28, 322, '一次加权频次 c 的 merge 把 c 处「两符号」换成「一符号」：语料总长恰减 c —— 79 → 47（−40%）；merge 可组合 → token 带层级结构', { size: 10 }));
  body.push(cap(28, 340, 'byte-level 起点（256 个基元）：任何字符串皆可编码，OOV 机制上不存在', { size: 10 }));
  body.push(cap(28, 358, 'GPT-2 词表 50257 = 256 + 50000 merge + 1 特殊 token；CJK 单字最坏 3 token（§3）', { size: 10 }));

  return scene(720, 372, body, { label: '字符串经 UTF-8、BPE 切分、embedding 查表进入模型；副板为 BPE 四轮 merge 的语料压缩账' }).svg();
}

/* ============================================================
 * 2.2 §11 — BLT：熵驱动的动态 patch（block 1）
 * bytes → 逐位熵超阈值处切开 → 变长 patch → 轻量 encoder → 大 latent transformer → 轻量 decoder
 * ============================================================ */
function bltEntropy() {
  const UID = 'p2blt';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'BLT：码本不冻结 —— patch 边界由逐位熵驱动，算力按不确定性分配（§11）', { size: 12, color: 'var(--ink-soft)' }));

  // 熵条 + byte 行（H(b_{i+1}|b_{<=i}) 超阈值处切开）
  const H = [5, 4, 5, 22, 17, 5, 4, 6];
  const bx = (i) => 52 + i * 66;
  body.push(`<line x1="40" y1="52" x2="566" y2="52" stroke="var(--seal)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>`);
  body.push(cap(574, 55, '熵阈值', { size: 9, color: 'var(--seal)' }));
  H.forEach((h, i) => {
    const hot = h > 12;
    body.push(`<rect class="dia-bar ${hot ? 'dia-bar-seal' : 'dia-bar-warm'} sig-bar" style="--i:${(i * 0.18).toFixed(2)}" x="${bx(i) + 10}" y="${64 - h}" width="32" height="${h}" rx="2"/>`);
    const n = node({ id: `b${i}`, x: bx(i), y: 74, w: 52, h: 32, role: 'data', title: `b${i + 1}`, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(596, 94, '…', { mono: true, size: 13 }));
  // 切分线：b3|b4 与 b5|b6 之间
  [243, 375].forEach((x) => {
    body.push(`<line x1="${x}" y1="68" x2="${x}" y2="116" stroke="var(--seal)" stroke-width="1.4" stroke-dasharray="3 3"/>`);
  });
  body.push(cap(243, 130, '✂', { size: 11, anchor: 'middle', color: 'var(--seal)' }));
  body.push(cap(375, 130, '✂', { size: 11, anchor: 'middle', color: 'var(--seal)' }));

  // patch 行
  const p1 = node({ id: 'p1', x: 52, y: 138, w: 184, h: 36, role: 'cache', title: '[b1 b2 b3]', mono: true, titleSize: 11 });
  const p2 = node({ id: 'p2', x: 250, y: 138, w: 118, h: 36, role: 'cache', title: '[b4 b5]', mono: true, titleSize: 11 });
  const p3 = node({ id: 'p3', x: 382, y: 138, w: 184, h: 36, role: 'cache', title: '[b6 b7 b8]', mono: true, titleSize: 11 });
  [p1, p2, p3].forEach((n) => { nodes.push(n); body.push(n.svg); });
  [[144, 1], [309, 3], [474, 5]].forEach(([x, i]) => {
    body.push(edge([x, 106], [x, 138], { nodes, pulse: true, dur: 1.4, delay: 0.3 + i * 0.2, uid: UID, eid: `g${i}`, arrow: false }));
  });
  body.push(cap(340, 192, 'patch 长度随可预测性变化：可预测区段合并、高不确定区段细切', { size: 10, anchor: 'middle' }));

  // 三段式管线
  const enc = node({ id: 'enc', x: 60, y: 216, w: 160, h: 46, role: 'op', title: 'byte encoder', sub: '轻量 · patch → 向量', mono: true, titleSize: 11.5, subSize: 9.5 });
  const lat = node({ id: 'lat', x: 300, y: 210, w: 200, h: 56, role: 'attn', title: 'latent transformer', sub: '大 · 主算力在 patch 级', mono: true, titleSize: 12, subSize: 9.5 });
  const dec = node({ id: 'dec', x: 566, y: 216, w: 140, h: 46, role: 'op', title: 'byte decoder', sub: '轻量 · 逐 byte 展开', mono: true, titleSize: 11.5, subSize: 9.5 });
  [enc, lat, dec].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([140, 174], enc, { nodes, portTo: enc.top, pulse: true, dur: 1.6, delay: 0.8, uid: UID, eid: 'e0' }));
  body.push(edge(enc, lat, { nodes, pulse: true, dur: 1.6, delay: 1.0, uid: UID, eid: 'e1' }));
  body.push(edge(lat, dec, { nodes, pulse: true, dur: 1.6, delay: 1.2, uid: UID, eid: 'e2' }));

  body.push(cap(28, 296, '对照静态码本：byte 序列 ≈ 4× token 序列长 → byte 侧只能轻量；2Vd 词表矩阵整体消失（§7 第二笔账）', { size: 10.5 }));
  body.push(cap(28, 314, 'FLOP 对齐 scaling 对照（至 8B 参数 / 4T bytes）：固定推理 FLOPs 下追平或超过 BPE 基线', { size: 10.5 }));

  return scene(720, 328, body, { label: 'BLT：逐位熵超阈值处切出变长 patch，轻量 byte 编解码夹住大 latent transformer' }).svg();
}

/* ============================================================
 * 2.3 §2 — 单头 scaled dot-product attention 数据流（block 0）
 * X →（W^Q/W^K/W^V）→ Q,K,V → S = QKᵀ/√d_k [T,T] → softmax → P → O = PV
 * ============================================================ */
function attnSingleHead() {
  const UID = 'p2a1';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'scaled dot-product attention：同一份 x 读三次、读出三个角色（式 3–4）', { size: 12, color: 'var(--ink-soft)' }));

  const X = node({ id: 'X', x: 24, y: 128, w: 120, h: 56, role: 'data', title: 'X [B, T, d]', sub: 'RMSNorm(x) 输出', mono: true, titleSize: 11.5, subSize: 9.5 });
  const Q = node({ id: 'Q', x: 210, y: 36, w: 140, h: 42, role: 'proj', title: 'Q [B, T, d_k]', mono: true, titleSize: 11.5 });
  const K = node({ id: 'K', x: 210, y: 128, w: 140, h: 42, role: 'proj', title: 'K [B, T, d_k]', mono: true, titleSize: 11.5 });
  const V = node({ id: 'V', x: 210, y: 220, w: 140, h: 42, role: 'proj', title: 'V [B, T, d_k]', mono: true, titleSize: 11.5 });
  [X, Q, K, V].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(X, Q, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'q' }));
  body.push(cap(176, 52, 'W^Q', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(X, K, { nodes, pulse: true, dur: 1.6, delay: 0.15, uid: UID, eid: 'k' }));
  body.push(cap(176, 122, 'W^K', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(X, V, { nodes, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'v' }));
  body.push(cap(176, 236, 'W^V', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(280, 100, '要找什么', { size: 9, anchor: 'middle' }));
  body.push(cap(280, 192, '广告什么', { size: 9, anchor: 'middle' }));
  body.push(cap(280, 284, '交付什么', { size: 9, anchor: 'middle' }));

  const S = node({ id: 'S', x: 420, y: 64, w: 170, h: 56, role: 'attn', title: 'S = QKᵀ/√d_k', sub: '[B, T, T] · 唯一 T² 对象', mono: true, titleSize: 12, subSize: 9.5 });
  const P = node({ id: 'P', x: 420, y: 168, w: 170, h: 46, role: 'attn', title: 'P [B, T, T]', sub: '行和为 1', mono: true, titleSize: 11.5, subSize: 10 });
  const O = node({ id: 'O', x: 420, y: 260, w: 170, h: 46, role: 'data', title: 'O = PV', sub: '[B, T, d_k]', mono: true, titleSize: 12, subSize: 10 });
  [S, P, O].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(Q, [420, 80], { nodes, pulse: true, dur: 1.6, delay: 0.45, uid: UID, eid: 'qs' }));
  body.push(edge(K, [420, 104], { nodes, pulse: true, dur: 1.6, delay: 0.55, uid: UID, eid: 'ks' }));
  body.push(edge(S, P, { nodes, pulse: true, dur: 1.4, delay: 0.8, uid: UID, eid: 'sp' }));
  body.push(cap(610, 148, 'softmax 逐行（式 2）', { size: 9.5, anchor: 'middle' }));
  body.push(edge(P, O, { nodes, pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'po' }));
  body.push(edge(V, [420, 283], { nodes, pulse: true, dur: 1.8, delay: 0.7, uid: UID, eid: 'vo', strong: true }));
  body.push(cap(392, 292, '加权和', { size: 9, anchor: 'end' }));

  body.push(cap(28, 326, 'score 经 rank-d_k 双线性透镜：⟨q_t, k_s⟩ = x_t·(W^Q W^Kᵀ)·x_s —— d = 4096、d_k = 128 时即 d 维内容的 3% 截面（§2）', { size: 10.5 }));
  body.push(cap(28, 344, '缩放 1/√d_k 不是调参产物：Var(q·k) = d_k（§3 四步推导链），幂次 1/2 唯一', { size: 10.5 }));

  return scene(720, 360, body, { label: '单头 attention：X 经三投影得 Q/K/V，打分、softmax、加权和输出' }).svg();
}

/* ============================================================
 * 2.3 §5 — multi-head：拆分、并行、concat + W^O（block 1）
 * Llama-3.1-8B 数值：d = 4096、H = 32、d_k = 128；参数 4d² 与 H 无关
 * ============================================================ */
function attnMultiHead() {
  const UID = 'p2a2';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'multi-head：H 个 rank-128 透镜并行，concat 后一次 W^O 汇合（式 11，Llama-3.1-8B 数值）', { size: 12, color: 'var(--ink-soft)' }));

  const X = node({ id: 'X', x: 20, y: 140, w: 116, h: 52, role: 'data', title: 'X [B, T, 4096]', mono: true, titleSize: 10.5 });
  const QKV = node({ id: 'QKV', x: 170, y: 128, w: 160, h: 64, role: 'proj', title: 'W_q / W_k / W_v', sub: '各 d×d 整拼成块', mono: true, titleSize: 11.5, subSize: 10 });
  [X, QKV].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(X, QKV, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 'x' }));

  body.push(cap(420, 32, 'view/transpose → [B, 32, T, 128] · 每 head 独立打分', { mono: true, size: 9 }));

  const heads = [
    { id: 'h1', y: 44, t: 'head 1', s: 'S_1 → P_1 → O_1' },
    { id: 'h2', y: 116, t: 'head 2', s: 'S_2 → P_2 → O_2' },
    { id: 'hd', y: 188, t: '⋮', s: '… 共 32 个 …' },
    { id: 'h32', y: 260, t: 'head 32', s: 'S_32 → P_32 → O_32' },
  ];
  const delays = [0.2, 0.35, 0.5, 0.65];
  heads.forEach((hh, i) => {
    const n = node({ id: hh.id, x: 430, y: hh.y, w: 150, h: 48, role: 'attn', title: hh.t, sub: hh.s, mono: true, titleSize: 11.5, subSize: 9 });
    nodes.push(n); body.push(n.svg);
    body.push(edge(QKV, n, { nodes, portFrom: QKV.right, pulse: true, dur: 1.7, delay: delays[i], uid: UID, eid: `h${i}` }));
  });

  const cc = node({ id: 'cc', x: 618, y: 140, w: 92, h: 64, role: 'proj', title: 'concat', sub: '+ W^O', mono: true, titleSize: 11.5, subSize: 10 });
  nodes.push(cc); body.push(cc.svg);
  heads.forEach((hh, i) => {
    body.push(edge(hh.id, cc, { nodes, portTo: cc.left, pulse: true, dur: 1.7, delay: 0.9 + delays[i] * 0.5, uid: UID, eid: `o${i}` }));
  });
  body.push(cap(664, 226, '→ [B, T, 4096]', { mono: true, size: 9.5, anchor: 'middle' }));

  body.push(cap(28, 330, '参数账：三块 QKV + W^O = 4d²，与 H 无关（H 加倍则 d_k 减半，H·d_k = d）—— OLMo-2-7B 恰为 4d²（真 MHA）', { size: 10.5 }));
  body.push(cap(28, 348, '唯一随 H 变的账面量：scores 内存 ∝ H；各头只在 W^O 处汇合一次 —— multi-head ≠ ensemble', { size: 10.5 }));

  return scene(720, 364, body, { label: 'multi-head attention：32 个头各自独立打分，concat 后过输出投影' }).svg();
}

/* ============================================================
 * 2.4 §4.4 — decode 步 t 单层：cache 读写（block 1）
 * q_t 一次性 / k_t,v_t append 进 cache / s_t 单行 [B,32,1,t] / 无 mask
 * ============================================================ */
function kvDecodeStep() {
  const UID = 'p2kv';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'decode 步 t（单层）：q 一次性、K/V append 进 cache、score 只有单行（§4.4 shape 走读）', { size: 12, color: 'var(--ink-soft)' }));

  const xt = node({ id: 'xt', x: 24, y: 150, w: 110, h: 50, role: 'data', title: 'x_t', sub: '[B, 1, 4096]', mono: true, titleSize: 12, subSize: 9.5 });
  const qkv = node({ id: 'qkv', x: 178, y: 144, w: 120, h: 56, role: 'proj', title: 'QKV 投影', sub: '只算当前 token', mono: true, titleSize: 11.5, subSize: 9.5 });
  const qt = node({ id: 'qt', x: 350, y: 44, w: 160, h: 46, role: 'data', title: 'q_t [B, 32, 1, 128]', sub: '一次性 · 用完即弃（§4.3）', mono: true, titleSize: 10, subSize: 9.5 });
  const kt = node({ id: 'kt', x: 350, y: 124, w: 160, h: 40, role: 'data', title: 'k_t [B, 8, 1, 128]', mono: true, titleSize: 10.5 });
  const vt = node({ id: 'vt', x: 350, y: 196, w: 160, h: 40, role: 'data', title: 'v_t [B, 8, 1, 128]', mono: true, titleSize: 10.5 });
  [xt, qkv, qt, kt, vt].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(xt, qkv, { nodes, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'x' }));
  body.push(edge(qkv, qt, { nodes, pulse: true, dur: 1.5, delay: 0.2, uid: UID, eid: 'q' }));
  body.push(edge(qkv, kt, { nodes, pulse: true, dur: 1.5, delay: 0.35, uid: UID, eid: 'k' }));
  body.push(edge(qkv, vt, { nodes, pulse: true, dur: 1.5, delay: 0.5, uid: UID, eid: 'v' }));

  // cache 区
  body.push(zone({ x: 556, y: 100, w: 148, h: 150, label: 'cache · 每层一份', tone: 'seal' }));
  const cK = node({ id: 'cK', x: 566, y: 126, w: 128, h: 42, role: 'cache', title: 'K [B, 8, t, 128]', mono: true, titleSize: 9.5 });
  const cV = node({ id: 'cV', x: 566, y: 184, w: 128, h: 42, role: 'cache', title: 'V [B, 8, t, 128]', mono: true, titleSize: 9.5 });
  [cK, cV].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(kt, cK, { nodes, pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 'ka' }));
  body.push(edge(vt, cV, { nodes, pulse: true, dur: 1.4, delay: 0.85, uid: UID, eid: 'va' }));
  body.push(cap(552, 118, 'append 沿 t 维', { size: 9, anchor: 'end' }));

  // score / 输出
  const sc = node({ id: 'sc', x: 350, y: 272, w: 160, h: 50, role: 'attn', title: 's_t = q_tKᵀ/√D', sub: '[B, 32, 1, t] · 单行', mono: true, titleSize: 10.5, subSize: 9 });
  const ot = node({ id: 'ot', x: 590, y: 272, w: 116, h: 48, role: 'data', title: 'o_t = p_tV', sub: '[B, 32, 1, 128]', mono: true, titleSize: 10, subSize: 9 });
  const wo = node({ id: 'wo', x: 590, y: 340, w: 116, h: 44, role: 'proj', title: 'concat+W^O', sub: '→ [B, 1, 4096]', mono: true, titleSize: 9.5, subSize: 9 });
  [sc, ot, wo].forEach((n) => { nodes.push(n); body.push(n.svg); });
  // q_t → score（左路下行通道，避让 k_t/v_t）
  body.push(`<path class="dia-edge" d="M 366 90 V 108 H 334 V 272 H 380"/>`);
  body.push(pathPulse('M 366 90 V 108 H 334 V 272 H 380', { dur: 2.2, delay: 1.0 }));
  // cache K/V 全量读出：K → score（顶边入），V → o_t（顶边直落，o_t = p_tV）
  body.push(`<path class="dia-edge strong" d="M 566 147 H 548 V 262 H 510 V 272"/>`);
  body.push(pathPulse('M 566 147 H 548 V 262 H 510 V 272', { dur: 2.2, delay: 1.15 }));
  body.push(`<path class="dia-edge strong" d="M 633 226 V 272"/>`);
  body.push(pathPulse('M 633 226 V 272', { dur: 2.2, delay: 1.3 }));
  body.push(cap(548, 256, '本步读出全量', { size: 9, anchor: 'end' }));
  body.push(edge(sc, ot, { nodes, pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'so' }));
  body.push(cap(550, 288, 'softmax → p_t', { size: 9, anchor: 'middle' }));
  body.push(edge(ot, wo, { nodes, portFrom: ot.bottom, portTo: wo.top, pulse: true, dur: 1.4, delay: 1.5, uid: UID, eid: 'ow' }));

  body.push(cap(28, 366, 'decode 步不需要 mask：第 t 行的可见集 = cache 的全部列（s ≤ t 是 cache 的定义域），未来列尚不存在', { size: 10.5 }));
  body.push(cap(28, 384, '本步 attention FLOPs：每层 4td（式 8）；与并行前向逐项相同 —— cache 买的是「不重算」的墙钟，不是 FLOPs', { size: 10.5 }));

  return scene(720, 400, body, { label: 'decode 单步：当前 token 投影出 q/k/v，k/v 追加进 cache，q 对全量 cache 打单行 score' }).svg();
}

/* ============================================================
 * 2.5 §4.3/§4.5 — R_m 块对角结构 + 频率轴（block 0）
 * d = 8 示意：θ_0 > θ_1 > θ_2 > θ_3；λ_j = 2π·base^(2j/d) 对数铺满
 * ============================================================ */
function ropeBlockDiag() {
  const UID = 'p2rp';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'R_m = d/2 个 2D 旋转块的块对角矩阵（d = 8 示意）· 频率几何级数（式 12/16）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：块对角矩阵
  body.push(zone({ x: 36, y: 44, w: 300, h: 224 }));
  const blk = (bx, by, th) => {
    const cells = [['c', 0, 0], ['−s', 1, 0], ['s', 0, 1], ['c', 1, 1]];
    cells.forEach(([t, j, i]) => {
      body.push(`<rect class="dia-mx-on" x="${bx + j * 30}" y="${by + i * 30}" width="26" height="26" rx="4" opacity="0.85"/>`);
      body.push(cap(bx + j * 30 + 13, by + i * 30 + 18, t, { mono: true, size: 12, anchor: 'middle', color: 'var(--paper)' }));
    });
    body.push(cap(bx + 66, by + 34, th, { mono: true, size: 11, anchor: 'start', color: 'var(--ink-soft)' }));
  };
  blk(64, 72, 'θ₀');
  blk(140, 140, 'θ₁');
  blk(216, 208, 'θ₂');
  body.push(cap(206, 196, '⋱', { mono: true, size: 14, anchor: 'middle' }));
  body.push(cap(46, 288, '每块 = 式 (9) 的 2D 旋转（c = cos mθ_j，s = sin mθ_j）', { size: 9.5 }));
  body.push(cap(46, 304, 'θ₀ > θ₁ > …；只 2d 个非零元 → 预存 cos/sin 两表', { size: 9.5 }));

  // 右：频率轴（log 刻度）
  body.push(cap(380, 52, '频率轴（log 刻度）· θ_j = base^(−2j/d)', { mono: true, size: 10.5, color: 'var(--ink-soft)' }));
  body.push(`<line class="dia-edge strong" x1="400" y1="110" x2="682" y2="110"/>`);
  body.push('<polygon points="690,110 680,105.5 680,114.5" fill="var(--ink-soft)"/>');
  for (let i = 0; i < 12; i++) {
    const x = 400 + i * 25.5;
    body.push(`<circle class="sig-cell" style="--i:${(0.2 + i * 0.14).toFixed(2)}" cx="${x.toFixed(1)}" cy="110" r="5" fill="var(--ink)"/>`);
  }
  body.push(cap(400, 132, 'j = 0 · λ₀ = 2π', { mono: true, size: 9.5 }));
  body.push(cap(400, 148, '快通道 · 局部次序', { size: 10 }));
  body.push(cap(680, 132, 'λ_max = 2π·base^(1−2/d)', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(680, 148, '慢通道 · 全局位置', { size: 10, anchor: 'end' }));
  body.push(cap(380, 178, 'λ_j = 2π·base^(2j/d)：对数刻度等距铺满 —— 位置的多尺度展开', { mono: true, size: 9.5 }));
  body.push(cap(380, 204, '锚点波长账（d_head = 128 → 64 通道）', { size: 10.5, color: 'var(--ink-soft)' }));
  body.push(cap(380, 222, 'Llama-3.1 / OLMo-2：base 5×10⁵ → λ_max ≈ 2.6×10⁶', { mono: true, size: 9.5 }));
  body.push(cap(380, 238, 'Qwen3：base 10⁶ → λ_max ≈ 5.1×10⁶', { mono: true, size: 9.5 }));
  body.push(cap(380, 254, 'DeepSeek-V3：base 10⁴、只旋 64 维 → 32 通道（§4.7）', { mono: true, size: 9.5 }));
  body.push(cap(400, 280, 'base = 把频率轴对齐到目标窗口的旋钮', { size: 10 }));
  body.push(cap(400, 296, '窗内不满圈的慢通道 = 外推失败部位（§6）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 318, body, { label: 'RoPE 的块对角旋转矩阵与频率轴：64 个频率通道从局部次序铺到全局位置' }).svg();
}

/* ============================================================
 * 2.6 §1 — 两矩阵 FFN 标准形 shape 走读（block 0）
 * x [B,T,d] → h_pre [B,T,4d] → σ → h → FFN(x) [B,T,d]；8d²/层 ≈ 2/3
 * ============================================================ */
function ffnStandard() {
  const UID = 'p2ff';
  const body = [];
  const nodes = [];

  body.push(cap(40, 20, '两矩阵 FFN：逐位置 d → 4d → d（式 1；原文工作点 d = 512、d_ff = 2048）', { size: 12, color: 'var(--ink-soft)' }));

  const x = node({ id: 'x', x: 210, y: 44, w: 180, h: 44, role: 'data', title: 'x [B, T, d]', sub: 'RMSNorm₂ 的输出', mono: true, titleSize: 12, subSize: 10 });
  const hp = node({ id: 'hp', x: 120, y: 124, w: 360, h: 52, role: 'proj', title: 'h_pre [B, T, 4d]', sub: '扩张：d → 4d', mono: true, titleSize: 12.5, subSize: 10 });
  const hh = node({ id: 'hh', x: 120, y: 216, w: 360, h: 52, role: 'op', title: 'h = σ(h_pre) [B, T, 4d]', sub: 'σ 逐元素 · shape 不变', mono: true, titleSize: 12, subSize: 10 });
  const fo = node({ id: 'fo', x: 210, y: 304, w: 180, h: 44, role: 'data', title: 'FFN(x) [B, T, d]', sub: '残差加回 stream', mono: true, titleSize: 11.5, subSize: 10 });
  [x, hp, hh, fo].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(x, hp, { nodes, portFrom: x.bottom, portTo: hp.top, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'e1' }));
  body.push(cap(316, 110, 'xW₁ + b₁ · W₁ ∈ R^{d×4d}', { mono: true, size: 10 }));
  body.push(edge(hp, hh, { nodes, portFrom: hp.bottom, portTo: hh.top, pulse: true, dur: 1.5, delay: 0.25, uid: UID, eid: 'e2' }));
  body.push(cap(284, 200, 'σ 逐元素', { size: 10, anchor: 'end' }));
  body.push(edge(hh, fo, { nodes, portFrom: hh.bottom, portTo: fo.top, pulse: true, dur: 1.5, delay: 0.5, uid: UID, eid: 'e3' }));
  body.push(cap(316, 292, 'hW₂ + b₂ · W₂ ∈ R^{4d×d}', { mono: true, size: 10 }));

  // 右列账注
  body.push(cap(524, 60, '逐位置：B·T 个位置摊平', { size: 10 }));
  body.push(cap(524, 76, '过同一组 W₁, W₂', { size: 10 }));
  body.push(cap(524, 124, '参数 2·d·4d = 8d²/层', { mono: true, size: 9.5 }));
  body.push(cap(524, 140, '≈ 2/3 层参数（2.1 §4）', { size: 10 }));
  body.push(cap(524, 180, 'per-token FLOPs', { size: 10 }));
  body.push(cap(524, 196, '= 4·d·d_ff = 16d²', { mono: true, size: 9.5 }));
  body.push(cap(524, 240, 'bias：GPT-2 为 True；', { size: 10 }));
  body.push(cap(524, 256, 'Llama-3.1 全 False', { size: 10 }));

  body.push(cap(40, 378, '现代配方：σ 换 SwiGLU、矩阵两变三（下一图）；「这 2/3 参数买了什么」→ §4 的 key-value 记忆视角', { size: 10.5 }));

  return scene(720, 394, body, { label: 'FFN 标准形：逐位置两层 MLP，先扩张到 4d 再收缩回 d' }).svg();
}

/* ============================================================
 * 2.6 §3 — SwiGLU 门控数据流（block 1）
 * x ├ xW → Swish₁ ┐⊙ → h [B,T,d_ff′] → hW₂ → FFN(x)；等参数 d_ff′ = 8d/3
 * ============================================================ */
function swigluFlow() {
  const UID = 'p2sw';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, 'SwiGLU：门与值两路独立投影，逐元素相乘（式 5–6）—— 矩阵从两个变三个', { size: 12, color: 'var(--ink-soft)' }));

  const x = node({ id: 'x', x: 24, y: 130, w: 100, h: 52, role: 'data', title: 'x [B,T,d]', mono: true, titleSize: 10.5 });
  const g = node({ id: 'g', x: 170, y: 48, w: 140, h: 46, role: 'proj', title: '门路 xW', sub: '[B, T, d_ff′]', mono: true, titleSize: 11.5, subSize: 9.5 });
  const sw = node({ id: 'sw', x: 350, y: 48, w: 130, h: 46, role: 'op', title: 'Swish₁', sub: 'u·σ(u) = SiLU', mono: true, titleSize: 12 });
  const v = node({ id: 'v', x: 170, y: 212, w: 140, h: 46, role: 'proj', title: '值路 xV', sub: '不过非线性', mono: true, titleSize: 11.5, subSize: 10 });
  [x, g, sw, v].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(x, g, { nodes, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'xg' }));
  body.push(edge(x, v, { nodes, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 'xv' }));
  body.push(edge(g, sw, { nodes, pulse: true, dur: 1.4, delay: 0.25, uid: UID, eid: 'gs' }));

  // ⊙ 汇合
  body.push(addDot(536, 140).replace('>+<', '>⊙<'));
  body.push(edge(sw, [523, 134], { nodes, pulse: true, dur: 1.5, delay: 0.5, uid: UID, eid: 'sj' }));
  body.push(edge(v, [523, 146], { nodes, pulse: true, dur: 1.5, delay: 0.5, uid: UID, eid: 'vj' }));
  body.push(cap(500, 112, '⊙ 逐元素乘', { size: 9, anchor: 'middle' }));

  const h = node({ id: 'h', x: 566, y: 112, w: 110, h: 52, role: 'data', title: 'h', sub: '[B, T, d_ff′]', mono: true, titleSize: 12.5, subSize: 9 });
  const fo = node({ id: 'fo', x: 520, y: 216, w: 170, h: 44, role: 'data', title: 'FFN(x) = h·W₂', sub: '[B, T, d] · 加回 stream', mono: true, titleSize: 11, subSize: 9.5 });
  [h, fo].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([549, 140], h, { nodes, portTo: h.left, pulse: true, dur: 1.4, delay: 0.75, uid: UID, eid: 'jh' }));
  body.push(edge(h, fo, { nodes, portFrom: h.bottom, portTo: fo.top, pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'hf' }));
  body.push(cap(636, 200, 'W₂ ∈ R^{d_ff′×d}', { mono: true, size: 9, anchor: 'middle' }));

  body.push(cap(28, 292, '双路梯度（式 7）：∂h/∂v = σ(g)、∂h/∂g = σ′(g)·v —— g = −2 时值路梯度 0.12、门路仍 0.31：门关着，门自己照样学', { size: 10 }));
  body.push(cap(28, 310, '等参数（式 8）：3·d·d_ff′ = 2·d·d_ff ⟹ d_ff′ = 8d/3 = 10922.67 → 取 256 倍数：OLMo-2 11008 = 43×256', { mono: true, size: 10 }));
  body.push(cap(28, 328, '三家对账（§3.3）：OLMo-2 11008 ≈ 2.69d（贴规则）· Qwen3 12288 = 3d · Llama-3.1 14336 = 3.5d（宽 31%）', { size: 10 }));

  return scene(720, 344, body, { label: 'SwiGLU：门路经 Swish、值路线性，逐元素相乘后过 down 投影' }).svg();
}

/* ============================================================
 * 2.7 §5 — Post-LN vs Pre-LN 结构与初始化梯度剖面（block 0）
 * 恒等通路存亡；Xiong et al. Theorem 1：O(d√ln d) vs O(d√(ln d/L))
 * ============================================================ */
function prePostLN() {
  const UID = 'p2ln';
  const body = [];
  const nodes = [];

  body.push(cap(24, 20, 'LN 摆放决定梯度命运：恒等通路的存亡（式 9/13 · Xiong et al. Theorem 1 + 扩展分析）', { size: 12, color: 'var(--ink-soft)' }));

  // ---- Post-LN ----
  body.push(zone({ x: 24, y: 44, w: 340, h: 150, label: 'Post-LN：x ← LN(x + f(x))' }));
  const xl = node({ id: 'xl', x: 40, y: 100, w: 56, h: 36, role: 'data', title: 'x_l', mono: true, titleSize: 11.5 });
  const fl = node({ id: 'fl', x: 120, y: 150, w: 72, h: 34, role: 'op', title: 'f_l', mono: true, titleSize: 11.5 });
  const lnp = node({ id: 'lnp', x: 262, y: 100, w: 56, h: 36, role: 'norm', title: 'LN', mono: true, titleSize: 11.5 });
  [xl, fl, lnp].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(xl, [228, 118], { nodes, portFrom: xl.right, uid: UID, eid: 'pd' }));
  body.push(edge(xl, fl, { nodes, portFrom: xl.bottom, portTo: fl.left, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'pf' }));
  body.push(`<path class="dia-edge" d="M 192 167 H 238 V 127"/>`);
  body.push(pathPulse('M 192 167 H 238 V 127', { dur: 1.6, delay: 0.55 }));
  body.push(addDot(238, 118));
  body.push(edge([248, 118], lnp, { nodes, portTo: lnp.left, pulse: true, dur: 1.2, delay: 0.8, uid: UID, eid: 'pl' }));
  body.push(`<path class="dia-edge arrowed" d="M 318 118 H 334"/>`);
  body.push(cap(358, 122, 'x_{l+1}', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(36, 214, 'identity 被 LN 逐层重标定（式 8 不成立）', { size: 10, color: 'var(--seal-deep)' }));

  // ---- Pre-LN ----
  body.push(zone({ x: 384, y: 44, w: 336, h: 150, label: 'Pre-LN：x ← x + f(LN(x))' }));
  const xr = node({ id: 'xr', x: 400, y: 100, w: 56, h: 36, role: 'data', title: 'x_l', mono: true, titleSize: 11.5 });
  const lnr = node({ id: 'lnr', x: 452, y: 150, w: 56, h: 34, role: 'norm', title: 'LN', mono: true, titleSize: 11.5 });
  const fr = node({ id: 'fr', x: 536, y: 150, w: 72, h: 34, role: 'op', title: 'f_l', mono: true, titleSize: 11.5 });
  [xr, lnr, fr].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(xr, [584, 118], { nodes, portFrom: xr.right, uid: UID, eid: 'qd' }));
  body.push(edge(xr, lnr, { nodes, portFrom: xr.bottom, portTo: lnr.left, pulse: true, dur: 1.6, delay: 0.3, uid: UID, eid: 'ql' }));
  body.push(edge(lnr, fr, { nodes, pulse: true, dur: 1.2, delay: 0.5, uid: UID, eid: 'qf' }));
  body.push(`<path class="dia-edge" d="M 608 167 H 664 V 118 H 614"/>`);
  body.push(pathPulse('M 608 167 H 664 V 118 H 614', { dur: 1.8, delay: 0.7 }));
  body.push(addDot(604, 118));
  body.push(`<path class="dia-edge arrowed" d="M 614 118 H 640"/>`);
  body.push(cap(644, 122, 'x_{l+1}', { mono: true, size: 10 }));
  body.push(cap(396, 214, 'identity 直达（式 8 成立）：梯度经 I 项原样到达每一层', { size: 10 }));

  // ---- 梯度剖面 ----
  body.push(cap(24, 248, '参数梯度量级（初始化；末层 FFN 上界为定理，其余各层为扩展分析）', { size: 11, color: 'var(--ink-soft)' }));
  body.push(cap(24, 282, 'Post-LN', { mono: true, size: 10.5 }));
  const ph = [30, 25, 20, 15, 11, 8];
  ph.forEach((hh, i) => {
    body.push(`<rect class="dia-bar ${i < 2 ? 'dia-bar-seal' : 'dia-bar-soft'} sig-bar" style="--i:${(i * 0.22).toFixed(2)}" x="${110 + i * 44}" y="${290 - hh}" width="36" height="${hh}" rx="3"/>`);
  });
  body.push(`<line class="dia-edge" x1="104" y1="290" x2="392" y2="290"/>`);
  body.push(cap(110, 304, '输出端大', { size: 9.5 }));
  body.push(cap(392, 304, '向输入端衰减', { size: 9.5, anchor: 'end' }));
  body.push(cap(420, 284, '末层上界 O(d·√ln d) · 与 L 无关', { mono: true, size: 10 }));

  body.push(cap(24, 344, 'Pre-LN', { mono: true, size: 10.5 }));
  for (let i = 0; i < 6; i++) {
    body.push(`<rect class="dia-bar dia-bar-ink sig-bar" style="--i:${(1.4 + i * 0.12).toFixed(2)}" x="${110 + i * 44}" y="${352 - 18}" width="36" height="18" rx="3"/>`);
  }
  body.push(`<line class="dia-edge" x1="104" y1="352" x2="392" y2="352"/>`);
  body.push(cap(110, 366, '各层大致均匀', { size: 9.5 }));
  body.push(cap(420, 340, 'O(d·√(ln d / L)) · 随 L 衰减', { mono: true, size: 10 }));
  body.push(cap(420, 356, 'd=4096：L=32 → ÷5.7 · L=128 → ÷11', { mono: true, size: 9.5 }));

  body.push(cap(24, 394, '方向：衰减朝向输入端（非「浅层爆炸」）；Post-LN 深堆叠 ⇒ warmup 是数学必然（§5.4），Pre-LN 可省', { size: 10.5 }));

  return scene(720, 410, body, { label: 'Post-LN 与 Pre-LN 的接线对照与初始化梯度剖面' }).svg();
}

/* ============================================================
 * 2.8 §1 — 一个 block 的接线图（block 0，Llama-3.1-8B shape）
 * RMSNorm₁ → QKV → RoPE(q,k) → softmax(QKᵀ/√D+M)·V → concat 32 头 → W^O → ⊕
 * RMSNorm₂ → gate/up [B,T,14336]×2 → SiLU⊙ → down → ⊕
 * ============================================================ */
function blockWiring() {
  const UID = 'p2bw';
  const body = [];
  const nodes = [];

  const xl = node({ id: 'xl', x: 6, y: 4, w: 150, h: 36, role: 'data', title: 'x⁽ˡ⁻¹⁾ [B, T, 4096]', mono: true, titleSize: 10 });
  const xo = node({ id: 'xo', x: 6, y: 358, w: 160, h: 38, role: 'data', title: 'x⁽ˡ⁾ [B, T, 4096]', mono: true, titleSize: 10.5 });
  [xl, xo].forEach((n) => { nodes.push(n); body.push(n.svg); });
  // trunk
  body.push(`<line class="dia-edge strong" x1="86" y1="40" x2="86" y2="358"/>`);
  body.push(pathPulse('M 86 40 V 358', { dur: 5.0, delay: 0, r: 3.4 }));

  // attention 子层
  body.push(zone({ x: 110, y: 22, w: 596, h: 124 }));
  body.push(cap(126, 44, 'attention 子层（式 2）', { size: 10 }));
  body.push(cap(700, 44, 'GQA：K/V 8 头广播到 32', { size: 10, anchor: 'end' }));
  const rn1 = node({ id: 'rn1', x: 126, y: 58, w: 92, h: 52, role: 'norm', title: 'RMSNorm₁', titleSize: 11.5 });
  const qk = node({ id: 'qk', x: 242, y: 58, w: 110, h: 52, role: 'proj', title: 'QKV 投影', sub: 'Q4096·K1024·V1024', mono: true, titleSize: 11.5, subSize: 8.5 });
  const rp = node({ id: 'rp', x: 376, y: 58, w: 88, h: 52, role: 'op', title: 'RoPE', sub: '只作用 q,k', mono: true, titleSize: 11.5, subSize: 9 });
  const cr = node({ id: 'cr', x: 488, y: 58, w: 118, h: 52, role: 'attn', title: 'softmax(QKᵀ/√D+M)·V', sub: 'causal', mono: true, titleSize: 9, subSize: 9 });
  const wo = node({ id: 'wo', x: 630, y: 58, w: 70, h: 52, role: 'proj', title: 'W^O', sub: 'concat 32 头', mono: true, titleSize: 11.5, subSize: 8.5 });
  [rn1, qk, rp, cr, wo].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([86, 84], rn1, { nodes, portTo: rn1.left, uid: UID, eid: 't1', arrow: false }));
  body.push(edge(rn1, qk, { nodes, pulse: true, dur: 1.3, delay: 0.2, uid: UID, eid: 'a1' }));
  body.push(edge(qk, rp, { nodes, pulse: true, dur: 1.3, delay: 0.4, uid: UID, eid: 'a2' }));
  body.push(edge(rp, cr, { nodes, pulse: true, dur: 1.3, delay: 0.6, uid: UID, eid: 'a3' }));
  body.push(edge(cr, wo, { nodes, pulse: true, dur: 1.3, delay: 0.8, uid: UID, eid: 'a4' }));
  const w1 = 'M 665 110 V 168 H 95';
  body.push(`<path class="dia-edge" d="${w1}"/>`);
  body.push(pathPulse(w1, { dur: 2.2, delay: 1.1 }));
  body.push(addDot(86, 168));

  // FFN 子层
  body.push(zone({ x: 110, y: 196, w: 596, h: 118, label: 'FFN 子层（式 3）· SwiGLU 三矩阵' }));
  const rn2 = node({ id: 'rn2', x: 126, y: 230, w: 92, h: 52, role: 'norm', title: 'RMSNorm₂', titleSize: 11.5 });
  const gu = node({ id: 'gu', x: 242, y: 230, w: 150, h: 52, role: 'proj', title: 'gate / up 投影', sub: '[B, T, 14336] ×2', mono: true, titleSize: 11.5, subSize: 9 });
  const sl = node({ id: 'sl', x: 416, y: 230, w: 90, h: 52, role: 'op', title: 'SiLU ⊙', mono: true, titleSize: 11.5 });
  const dw = node({ id: 'dw', x: 530, y: 230, w: 110, h: 52, role: 'proj', title: 'down 投影', sub: '→ [B, T, 4096]', mono: true, titleSize: 11.5, subSize: 9 });
  [rn2, gu, sl, dw].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge([86, 256], rn2, { nodes, portTo: rn2.left, uid: UID, eid: 't2', arrow: false }));
  body.push(edge(rn2, gu, { nodes, pulse: true, dur: 1.3, delay: 1.5, uid: UID, eid: 'f1' }));
  body.push(edge(gu, sl, { nodes, pulse: true, dur: 1.3, delay: 1.7, uid: UID, eid: 'f2' }));
  body.push(edge(sl, dw, { nodes, pulse: true, dur: 1.3, delay: 1.9, uid: UID, eid: 'f3' }));
  const w2 = 'M 585 282 V 330 H 95';
  body.push(`<path class="dia-edge" d="${w2}"/>`);
  body.push(pathPulse(w2, { dur: 2.2, delay: 2.2 }));
  body.push(addDot(86, 330));

  body.push(cap(24, 408, '「归一化读、加法写」（式 1）：组件可替换（RoPE↔ALiBi · SwiGLU↔GELU · GQA↔MHA），stream 层结论逐项保留', { size: 10.5 }));

  return scene(720, 422, body, { label: 'decoder block 接线：两个子层各自 RMSNorm 读出、残差写回' }).svg();
}

/* ============================================================
 * 2.8 §1/§6 — 整机栈 + 参数住处（block 1）
 * ids → E gather → x⁰ → ×L block → RMSNorm_f → W_U → logits → q_θ
 * 右侧账：式 (6)(7) 逐项，合计 8,030,261,248
 * ============================================================ */
function wholeStack() {
  const UID = 'p2ws';
  const body = [];
  const nodes = [];

  body.push(cap(28, 20, '整机栈（式 4）：参数只住五处 —— 右侧逐个入账即 §6 的个位级对账', { size: 12, color: 'var(--ink-soft)' }));

  const ids = node({ id: 'ids', x: 36, y: 28, w: 170, h: 36, role: 'data', title: 'ids [B, T]', mono: true, titleSize: 11.5 });
  const x0 = node({ id: 'x0', x: 36, y: 92, w: 170, h: 40, role: 'data', title: 'x⁽⁰⁾ [B, T, d]', sub: 'E 查表（gather）', mono: true, titleSize: 11.5, subSize: 10 });
  const bk = node({ id: 'bk', x: 36, y: 156, w: 170, h: 100, role: 'op', title: '× L 个 block', sub: '式 (1)–(3) · L = 32', mono: true, titleSize: 12.5, subSize: 10 });
  const rf = node({ id: 'rf', x: 36, y: 280, w: 170, h: 36, role: 'norm', title: 'RMSNorm_f', sub: 'γ_f', mono: true, titleSize: 11, subSize: 10 });
  const lg = node({ id: 'lg', x: 36, y: 340, w: 170, h: 40, role: 'proj', title: 'z = W_U·x', sub: 'logits [B, T, V]', mono: true, titleSize: 11.5, subSize: 10 });
  const q = node({ id: 'q', x: 252, y: 340, w: 172, h: 40, role: 'proj', title: 'q_θ(· | x_{<t})', sub: 'softmax 逐位置', mono: true, titleSize: 11, subSize: 10 });
  [ids, x0, bk, rf, lg, q].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(ids, x0, { nodes, portFrom: ids.bottom, portTo: x0.top, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'e1' }));
  body.push(cap(216, 86, 'gather · 0 FLOPs', { mono: true, size: 9 }));
  body.push(edge(x0, bk, { nodes, portFrom: x0.bottom, portTo: bk.top, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'e2' }));
  body.push(edge(bk, rf, { nodes, portFrom: bk.bottom, portTo: rf.top, pulse: true, dur: 1.4, delay: 0.4, uid: UID, eid: 'e3' }));
  body.push(edge(rf, lg, { nodes, portFrom: rf.bottom, portTo: lg.top, pulse: true, dur: 1.4, delay: 0.6, uid: UID, eid: 'e4' }));
  body.push(edge(lg, q, { nodes, pulse: true, dur: 1.4, delay: 0.8, uid: UID, eid: 'e5' }));
  body.push(cap(113, 272, 'x⁽ᴸ⁾ [B, T, d]', { mono: true, size: 9, anchor: 'end' }));

  // 右侧账板
  body.push(zone({ x: 440, y: 28, w: 268, h: 250, label: '参数住处 · 式 (6)(7)（Llama-3.1-8B）' }));
  const rows = [
    [66, 'E：V·d = 525,336,576'],
    [96, '×L：attn 2.5d² + FFN 3·d·d_ff'],
    [112, '+ γ₁,γ₂ ＝ 218,112,000 / 层'],
    [140, '32 层计 6,979,584,000'],
    [168, 'γ_f：4,096'],
    [198, 'W_U：V·d = 525,336,576（untied）'],
  ];
  rows.forEach(([y, t]) => body.push(cap(456, y, t, { mono: true, size: 9.5 })));
  body.push(`<line class="dia-edge" x1="452" y1="220" x2="696" y2="220"/>`);
  body.push(cap(456, 244, 'Σ = 8,030,261,248 ≈ 8.03×10⁹', { mono: true, size: 11.5, color: 'var(--seal-deep)', weight: '700' }));
  body.push(cap(456, 264, '「8B」名实相符 · 式 (7)', { size: 9.5 }));

  // 账板 ↔ 流向的虚线链接
  body.push(`<path class="dia-edge dash" d="M 150 80 C 260 74, 360 68, 440 62"/>`);
  body.push(`<path class="dia-edge dash" d="M 206 206 C 300 180, 380 138, 440 112"/>`);
  body.push(`<path class="dia-edge dash" d="M 206 298 C 300 272, 380 196, 440 168"/>`);
  body.push(`<path class="dia-edge dash" d="M 206 368 C 300 332, 380 236, 440 200"/>`);

  body.push(cap(440, 304, '对照 OLMo-2-7B：7,298,617,344', { mono: true, size: 9.5 }));
  body.push(cap(440, 320, '（每层 4 个 norm：QK-norm 两份增益）', { size: 9.5 }));
  body.push(cap(28, 396, 'norm 增益 (2L+1)·d = 266,240 入账才到个位（占 0.0033%）；速算 12Ld² 管量级与偏差签名、式 (6) 管个位 —— §6 的分工', { size: 10 }));

  return scene(720, 410, body, { label: '整机栈：embedding、32 层 block、final norm、输出头；右侧逐项参数账合计 8.03B' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '02-transformer-anatomy/01-overview', replacesBlock: 0, title: '一次 forward 全景：ids → stream → 逐位置分布', svg: overviewForward },
  { slug: '02-transformer-anatomy/01-overview', replacesBlock: 1, title: '三种架构形态：mask 与连接方式的差异', svg: threeForms },
  { slug: '02-transformer-anatomy/02-tokenization-and-embeddings', replacesBlock: 0, title: '字符串 → bytes → tokens → embedding', svg: tokenPipeline },
  { slug: '02-transformer-anatomy/02-tokenization-and-embeddings', replacesBlock: 1, title: 'BLT：熵驱动的动态 patch', svg: bltEntropy },
  { slug: '02-transformer-anatomy/03-attention-mechanism', replacesBlock: 0, title: '单头 attention 数据流与 T² score', svg: attnSingleHead },
  { slug: '02-transformer-anatomy/03-attention-mechanism', replacesBlock: 1, title: 'multi-head：32 个低秩透镜并行', svg: attnMultiHead },
  { slug: '02-transformer-anatomy/04-causality-and-kv-cache', replacesBlock: 1, title: 'decode 步 t 的 cache 读写', svg: kvDecodeStep },
  { slug: '02-transformer-anatomy/05-positional-encoding', replacesBlock: 0, title: 'R_m 块对角结构与频率轴', svg: ropeBlockDiag },
  { slug: '02-transformer-anatomy/06-ffn', replacesBlock: 0, title: '两矩阵 FFN：d → 4d → d', svg: ffnStandard },
  { slug: '02-transformer-anatomy/06-ffn', replacesBlock: 1, title: 'SwiGLU 门控数据流', svg: swigluFlow },
  { slug: '02-transformer-anatomy/07-norm-and-residual', replacesBlock: 0, title: 'Post-LN vs Pre-LN：恒等通路与梯度剖面', svg: prePostLN },
  { slug: '02-transformer-anatomy/08-assembly', replacesBlock: 0, title: '一个 block 的完整接线（Llama-3.1-8B）', svg: blockWiring },
  { slug: '02-transformer-anatomy/08-assembly', replacesBlock: 1, title: '整机栈与参数五住处', svg: wholeStack },
];
