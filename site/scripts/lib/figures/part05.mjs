// Part 5（05-case-studies）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs。数字与机制措辞全部来自章节原文。
// 层栈类（V3 / GLM-4.6 / K2）共用 stackFigure 模板——同一「V3 架构类」谱系读法的视觉一致。

import { scene, node, edge, zone, cap, matrix } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（直线/L 形通道用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 5.1 §5.2 — iRoPE：48 层 = 12 × 4 层周期（block 0）
 * 原文：RoPE 层 ×3 在 8192 chunk 内打分（相对位移恒 < 8192，RoPE 不出域；
 * Scout 另叠 llama3 式 scaling f16×8192）；NoPE 层 ×1 全局可见集，
 * score 乘对数温度 m(t)（式 8）补偿长程熵稀释；FFN 与层型正交。
 * ============================================================ */
function llamaIrope() {
  const UID = 'ir';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 28, 'Llama-4 文本侧 48 层 = 12 × 4 层周期（no_rope_layers 3:1）', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // 左：4 层周期（3 RoPE + 1 NoPE）
  body.push(zone({ x: 32, y: 44, w: 204, h: 204, label: '4 层周期 × 12' }));
  const ROW_Y = [68, 112, 156, 200];
  const layerCards = ROW_Y.map((y, i) => {
    const isNope = i === 3;
    const n = node({
      id: `L${i}`, x: 52, y: ROW_Y[i], w: 160, h: 40,
      role: isNope ? 'cache' : 'attn',
      title: isNope ? 'NoPE 层' : 'RoPE 层',
      sub: isNope ? '全局 · score × m(t)' : 'chunk 8192 内打分',
      titleSize: 12.5,
    });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // 右：逐层可见集（query 居右端）
  body.push(cap(288, 60, '逐层可见集（query 居右端）', { size: 10.5 }));
  const AXIS_X0 = 288, AXIS_X1 = 668;
  ROW_Y.forEach((y0, i) => {
    const Y = y0 + 20;
    body.push(`<line class="dia-edge" x1="${AXIS_X0}" y1="${Y}" x2="${AXIS_X1}" y2="${Y}"/>`);
    const isNope = i === 3;
    if (isNope) {
      body.push(`<rect class="dia-mx-hot sig-cell" style="--i:1.6" x="${AXIS_X0}" y="${Y - 5}" width="${AXIS_X1 - AXIS_X0}" height="10" rx="4"/>`);
      body.push(cap(470, Y - 10, '全部 T', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
      body.push(pathPulse(`M ${AXIS_X0} ${Y} H ${AXIS_X1}`, { dur: 2.8, delay: 0.8, uid: UID, r: 3.4 }));
    } else {
      body.push(`<rect class="dia-mx-hot sig-cell" style="--i:${(i * 0.4).toFixed(1)}" x="${AXIS_X1 - 92}" y="${Y - 5}" width="92" height="10" rx="4"/>`);
      body.push(cap(AXIS_X1 - 46, Y - 10, '8192', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
    }
    body.push(`<circle cx="${AXIS_X1}" cy="${Y}" r="3" fill="var(--seal)"/>`);
    body.push(edge(layerCards[i], [AXIS_X0, Y], { nodes, portFrom: layerCards[i].right, pulse: true, dur: 1.8, delay: i * 0.3, uid: UID, eid: i, arrow: false }));
  });
  body.push(cap(288, 106, 'RoPE 层 ×3：相对位移恒 < 8192 → RoPE 不出域', { size: 10.5 }));
  body.push(cap(288, 244, 'NoPE 层 ×1：score × m(t)（式 8 对数温度），补偿长程熵稀释 Δ ≥ ln T', { size: 10.5 }));

  // 下：FFN 侧（与层型正交）
  body.push(zone({ x: 32, y: 256, w: 664, h: 72, label: 'FFN 侧（与层型正交 · 式 (5)(6)）' }));
  const scout = node({ id: 'scout', x: 52, y: 284, w: 250, h: 40, role: 'op', title: 'Scout · 层层 MoE', sub: '16 routed + 1 shared，top-1', titleSize: 12 });
  const mav = node({ id: 'mav', x: 330, y: 284, w: 260, h: 40, role: 'op', title: 'Maverick · 隔层 MoE', sub: '128 routed + 1 shared，top-1', titleSize: 12 });
  nodes.push(scout, mav); body.push(scout.svg, mav.svg);

  body.push(cap(36, 346, 'Scout 另叠 llama3 式 scaling（factor 16 × original 8192 → 基窗 131072）；Maverick rope_scaling = null', { mono: true, size: 10 }));

  return scene(720, 358, body, { label: 'iRoPE：每 4 层周期 3 层 RoPE 只看 8192 chunk、1 层 NoPE 看全局并乘对数温度' }).svg();
}

/* ============================================================
 * 5.1 §5.3 — early fusion 多模态（block 1）
 * 原文：image [H,W,3] → patch 14（448² → 1024）→ vision tower（34 层，1408）
 * → pixel_shuffle 0.5（1024 → 256）→ projector → 与文本同一 embedding 空间
 * → 同一条 residual stream（48 层）
 * ============================================================ */
function llamaEarlyFusion() {
  const UID = 'ef';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 30, 'early fusion：vision tower 在模型内，图像 token 与文本同一条流水线', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  const img = node({ id: 'img', x: 32, y: 64, w: 100, h: 56, role: 'data', title: 'image', sub: '[H, W, 3]', mono: true });
  const tower = node({ id: 'tower', x: 188, y: 64, w: 172, h: 56, role: 'attn', title: 'vision tower', sub: '34 层 · hidden 1408', mono: true, titleSize: 13 });
  const shuffle = node({ id: 'shuffle', x: 416, y: 64, w: 146, h: 56, role: 'op', title: 'pixel shuffle', sub: 'ratio 0.5 → ×0.25', mono: true, titleSize: 13 });
  const proj = node({ id: 'proj', x: 618, y: 64, w: 88, h: 56, role: 'proj', title: 'projector', mono: true, titleSize: 11.5 });
  nodes.push(img, tower, shuffle, proj);
  body.push(img.svg, tower.svg, shuffle.svg, proj.svg);

  body.push(cap(160, 54, 'patch 14 → 1024', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(388, 54, '1024 → 256', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(590, 54, '对齐 embedding', { size: 10, anchor: 'middle' }));

  const text = node({ id: 'text', x: 32, y: 196, w: 128, h: 44, role: 'data', title: '文本 token', sub: 'text ids', mono: true });
  const stream = node({ id: 'stream', x: 188, y: 192, w: 508, h: 52, role: 'data', title: '同一条 residual stream（48 层）', sub: '视觉 token 从第 1 层起与文本互相可见（同一套 attention，含 iRoPE）', titleSize: 13 });
  nodes.push(text, stream);
  body.push(text.svg, stream.svg);

  body.push(edge(img, tower, { nodes, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(tower, shuffle, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 1 }));
  body.push(edge(shuffle, proj, { nodes, pulse: true, dur: 1.8, delay: 0.6, uid: UID, eid: 2 }));
  body.push(edge(proj, [662, 192], { nodes, portFrom: proj.bottom, pulse: true, dur: 1.8, delay: 0.9, uid: UID, eid: 3, arrow: false }));
  body.push(edge(text, stream, { nodes, portTo: stream.left, pulse: true, dur: 1.8, delay: 1.1, uid: UID, eid: 4 }));

  body.push(cap(36, 268, '448×448 图 → 1024 patch →（shuffle 0.5）→ 256 token 进流；代价 = 图像 token 全程付 48 层 FLOPs 与 cache', { size: 10.5 }));

  return scene(720, 284, body, { label: 'early fusion：图像经 patch、vision tower、pixel shuffle、projector 后与文本 token 同一条 residual stream' }).svg();
}

/* ============================================================
 * 层栈模板：dense 前导 + MoE 主体 + 训练期外挂/控制回路（zone + 账条 + 节点卡链）
 * cfg: { title, subtitle, segCounts, zones: [{label,tone}]×3, rows: [[cards]×3], extras: [svg]×3, foot, aria }
 * ============================================================ */
const ZONE_Y = [62, 156, 284];
const ZONE_H = [84, 118, 84];
const CARD_Y = [94, 192, 316];

function stackFigure(UID, cfg) {
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 30, cfg.title, { size: 12.5, color: 'var(--ink)', weight: 600 }));
  body.push(cap(36, 48, cfg.subtitle, { mono: true, size: 10.5 }));

  for (let r = 0; r < 3; r++) {
    const y = ZONE_Y[r], h = ZONE_H[r];
    // 左侧层数账条
    if (r < 2) {
      const tone = r === 0 ? 'ink' : 'soft';
      body.push(`<rect class="dia-bar dia-bar-${tone}" x="36" y="${y}" width="56" height="${h}" rx="6"/>`);
      body.push(cap(64, y + h / 2 + 4, cfg.segCounts[r], { mono: true, size: 12, anchor: 'middle', color: r === 0 ? 'var(--paper)' : 'var(--ink)' }));
    } else {
      body.push(`<rect x="36" y="${y}" width="56" height="${h}" rx="6" fill="none" stroke="rgb(168 50 40 / 0.5)" stroke-width="1.2" stroke-dasharray="5 4"/>`);
      body.push(cap(64, y + h / 2 + 4, cfg.segCounts[r], { mono: true, size: 12, anchor: 'middle', color: 'var(--seal-deep)' }));
    }
    // 分区 + 节点卡链
    body.push(zone({ x: 160, y, w: 536, h, label: cfg.zones[r].label, tone: cfg.zones[r].tone ?? 'ink' }));
    const cards = cfg.rows[r].map((c, i) => {
      const n = node({ id: `${UID}r${r}c${i}`, x: c.x, y: CARD_Y[r], w: c.w, h: 44, role: c.role, title: c.title, sub: c.sub, mono: c.mono ?? false, titleSize: c.titleSize ?? 12.5 });
      nodes.push(n); body.push(n.svg);
      return n;
    });
    for (let i = 1; i < cards.length; i++) {
      body.push(edge(cards[i - 1], cards[i], { nodes, pulse: true, dur: 1.8, delay: r * 0.5 + i * 0.25, uid: UID, eid: `c${r}${i}` }));
    }
    if (cards.length) {
      body.push(edge([92, y + h / 2], cards[0], { nodes, pulse: true, dur: 1.6, delay: r * 0.4, uid: UID, eid: `s${r}`, arrow: false }));
    }
    if (cfg.extras?.[r]) body.push(cfg.extras[r]);
  }
  if (cfg.foot) body.push(cap(36, ZONE_Y[2] + ZONE_H[2] + 20, cfg.foot, { size: 10.5 }));
  return scene(720, 398, body, { label: cfg.aria }).svg();
}

/* ============ 5.2 §3.3 — DeepSeek-V3 层栈（block 0） ============ */
function deepseekV3Stack() {
  return stackFigure('v3', {
    title: 'DeepSeek-V3 层栈：61 层 = 3 dense + 58 MoE（+ MTP ×1）',
    subtitle: 'd = 7168 · untied 词表 · 复算合计 ≈ 671.6B ≈ 671B（§3.3）',
    segCounts: ['3 层', '58 层', '×1'],
    zones: [
      { label: '层 1–3 · dense（first_k_dense_replace = 3）' },
      { label: '层 4–61 · MoE ×58（moe_layer_freq = 1）' },
      { label: '训练期外挂 · MTP ×1（D = 1）', tone: 'seal' },
    ],
    rows: [
      [
        { x: 176, w: 230, role: 'attn', title: 'MLA', sub: 'cache 576 元素/token/层' },
        { x: 446, w: 210, role: 'proj', title: 'SwiGLU FFN', sub: 'd_ff = 18432', mono: true },
      ],
      [
        { x: 176, w: 150, role: 'attn', title: 'MLA', sub: '同 dense 层' },
        { x: 356, w: 170, role: 'op', title: 'router', sub: 'sigmoid · 256 选 8', mono: true },
        { x: 556, w: 130, role: 'proj', title: '256+1 experts', sub: 'd_ff = 2048', mono: true, titleSize: 11.5 },
      ],
      [
        { x: 176, w: 180, role: 'op', title: '拼接投影', sub: '投影 14336 → 7168', mono: true },
        { x: 396, w: 180, role: 'attn', title: '一个完整 block', sub: '共享 Emb / OutHead' },
      ],
    ],
    extras: [
      null,
      matrix({ x: 176, y: 246, n: 1, m: 8, cell: 13, gap: 3, on: () => false, hot: (i, j) => j < 4, warm: (i, j) => j >= 4 }) +
        cap(312, 259, '组限定：8 组选 4 组 → 单 token 目标节点 ≤ 4', { size: 10.5 }),
      cap(592, 344, '推理可整体弃置', { size: 10.5, color: 'var(--seal-deep)' }),
    ],
    foot: 'MTP ≈ 0.69B 训练期参与、推理可弃 —— 「训练时多付、推理时折现」；词表侧 2Vd ≈ 1.85B',
    aria: 'DeepSeek-V3 层栈：3 个 dense 前导层、58 个 MoE 层与训练期 MTP 模块的构成',
  });
}

/* ============ 5.4 §4 — GLM-4.6 层栈（block 1） ============ */
function glmStack() {
  return stackFigure('g6', {
    title: 'GLM-4.6 层栈：92 层 = 3 dense + 89 MoE（+ MTP ×1）',
    subtitle: 'd = 5120 · GQA g = 12 · 官方 355B/32B 口径含 MTP、不含词表',
    segCounts: ['3 层', '89 层', '×1'],
    zones: [
      { label: '层 1–3 · dense（first_k_dense_replace = 3）' },
      { label: '层 4–92 · MoE ×89' },
      { label: '训练期外挂 · MTP ×1（整件 MoE 层，D = 1）', tone: 'seal' },
    ],
    rows: [
      [
        { x: 176, w: 240, role: 'attn', title: 'GQA 96Q/8KV × 128', sub: 'QK-Norm + QKV bias', mono: true, titleSize: 11.5 },
        { x: 456, w: 200, role: 'proj', title: 'SwiGLU FFN', sub: 'd_ff = 12288', mono: true },
      ],
      [
        { x: 176, w: 150, role: 'attn', title: 'GQA', sub: '同 dense 层' },
        { x: 356, w: 170, role: 'op', title: 'router', sub: '160 选 8 · 归一 ×2.5' },
        { x: 556, w: 130, role: 'proj', title: '160+1 experts', sub: 'd_ff = 1536', mono: true, titleSize: 11.5 },
      ],
      [
        { x: 176, w: 180, role: 'op', title: '拼接投影', sub: '投影 10240 → 5120', mono: true },
        { x: 396, w: 190, role: 'attn', title: '完整 MoE 层', sub: '161 expert + attention' },
      ],
    ],
    extras: [
      null,
      cap(176, 259, 'RoPE θ = 10⁶ · partial 0.5（旋 64 维）· rope_scaling = null —— 原生覆盖 202752', { size: 10.5 }),
      cap(602, 344, '推理可整体弃置', { size: 10.5, color: 'var(--seal-deep)' }),
    ],
    foot: '词表 151552 × 5120 × 2 ≈ 1.55B 不入官方口径；MTP 弃置后激活 32.4B → 32.0B',
    aria: 'GLM-4.6 层栈：3 个 dense 前导层、89 个 MoE 层与整件 MoE 层构成的 MTP 模块',
  });
}

/* ============ 5.5 §3/§4 — Kimi-K2 层栈（block 0） ============ */
function kimiK2Stack() {
  return stackFigure('k2', {
    title: 'Kimi-K2 层栈：61 层 = 1 dense + 60 MoE（+ MuonClip 回路）',
    subtitle: 'd = 7168 · architectures = DeepseekV3ForCausalLM —— 架构类整体复用（§2.3）',
    segCounts: ['1 层', '60 层', 'γ'],
    zones: [
      { label: '层 1 · dense（first_k_dense_replace = 1）' },
      { label: '层 2–61 · MoE ×60' },
      { label: '训练期控制回路 · MuonClip（式 4，非梯度）', tone: 'seal' },
    ],
    rows: [
      [
        { x: 176, w: 230, role: 'attn', title: 'MLA', sub: 'cache 576 元素/token/层' },
        { x: 446, w: 210, role: 'proj', title: 'SwiGLU FFN', sub: 'd_ff = 18432', mono: true },
      ],
      [
        { x: 176, w: 150, role: 'attn', title: 'MLA', sub: '同 dense 层' },
        { x: 356, w: 170, role: 'op', title: 'router', sub: 'sigmoid · 384 选 8', mono: true },
        { x: 556, w: 130, role: 'proj', title: '384+1 experts', sub: 'd_ff = 2048', mono: true, titleSize: 11.5 },
      ],
      [
        { x: 176, w: 150, role: 'attn', title: '逐头监控 S_max', sub: 'qk-logit 峰值', titleSize: 11.5 },
        { x: 356, w: 170, role: 'op', title: 'γ = min(1, τ/S_max)', sub: 'τ = 100 越阈触发', mono: true, titleSize: 11 },
        { x: 556, w: 130, role: 'proj', title: 'W_Q,W_K × √γ', sub: '步后 · 非梯度', mono: true, titleSize: 11 },
      ],
    ],
    extras: [
      null,
      cap(176, 259, 'n_group = 1：V3 的 8 组选 4 组限定取消 —— 通信拓扑账的另一个取点（§5）', { size: 10.5 }),
      null,
    ],
    foot: 'MuonClip 逐头作用于全部 MLA 层、config 无字段：稳定性技术不一定有 config 签名（§4）',
    aria: 'Kimi-K2 层栈：1 个 dense 前导层、60 个 MoE 层，外挂 MuonClip 逐头控制回路',
  });
}

/* ============================================================
 * 5.3 §4.2 — Qwen3-Next：config → layer_types → 层装配（block 0）
 * 原文：num_hidden_layers 48 + full_attention_interval 4
 * → layer_types[i] = full_attention (i+1)%4==0 否则 linear_attention
 * → 36 × [GDN → MoE] / 12 × [GatedAttn → MoE]；rope_theta 1e7 只作用于 full 层
 * ============================================================ */
function qwenAssembly() {
  const UID = 'qw';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 28, 'config 声明 → configuration 派生 layer_types → modeling 逐层装配', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  const cfg = node({ id: 'cfg', x: 36, y: 120, w: 150, h: 76, role: 'plain', title: 'config.json', sub: '48 层 · interval 4', mono: true, titleSize: 13 });
  const disp = node({ id: 'disp', x: 248, y: 120, w: 160, h: 76, role: 'op', title: 'layer_types[i]', sub: '(i+1)%4==0 → full', mono: true, titleSize: 12.5 });
  nodes.push(cfg, disp);
  body.push(cfg.svg, disp.svg);
  body.push(edge(cfg, disp, { nodes, pulse: true, dur: 1.6, delay: 0, uid: UID, eid: 0 }));

  // 层型序列：12 格 = 3 个周期（linear ×3 + full ×1）
  body.push(cap(36, 224, '层型序列（每 4 层 1 个 full）', { size: 10 }));
  for (let i = 0; i < 12; i++) {
    const full = (i + 1) % 4 === 0;
    body.push(`<rect class="${full ? 'dia-mx-hot sig-cell' : 'dia-mx-warm sig-cell'}" style="--i:${(0.3 + i * 0.12).toFixed(2)}" x="${36 + i * 15}" y="232" width="12" height="13" rx="2.5"/>`);
  }
  body.push(cap(36, 262, '36 linear + 12 full = 48', { mono: true, size: 10 }));
  body.push(cap(36, 290, 'MTP 挂在主干之上，不占 48 层预算（§2.4）', { size: 10 }));

  // 右上：linear 分支
  body.push(zone({ x: 452, y: 56, w: 244, h: 116, label: '36 × linear_attention' }));
  const gdn = node({ id: 'gdn', x: 468, y: 92, w: 104, h: 60, role: 'attn', title: 'GDN', sub: 'state 128²', mono: true, titleSize: 13 });
  const moeA = node({ id: 'moeA', x: 592, y: 92, w: 92, h: 60, role: 'proj', title: 'MoE', sub: '512 top-10', mono: true, titleSize: 13, subSize: 10.5 });
  nodes.push(gdn, moeA);
  body.push(gdn.svg, moeA.svg);
  body.push(edge(gdn, moeA, { nodes, pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 1 }));
  body.push(cap(468, 164, 'k:16×128 → v:32×128 · conv 4', { mono: true, size: 9.5 }));

  // 右下：full 分支
  body.push(zone({ x: 452, y: 196, w: 244, h: 116, label: '12 × full_attention' }));
  const gattn = node({ id: 'gattn', x: 468, y: 232, w: 104, h: 60, role: 'attn', title: 'Gated Attn', sub: '输出门 · g=8', titleSize: 11.5, subSize: 10.5 });
  const moeB = node({ id: 'moeB', x: 592, y: 232, w: 92, h: 60, role: 'proj', title: 'MoE', sub: '同上', titleSize: 13 });
  nodes.push(gattn, moeB);
  body.push(gattn.svg, moeB.svg);
  body.push(edge(gattn, moeB, { nodes, pulse: true, dur: 1.6, delay: 0.9, uid: UID, eid: 2 }));
  body.push(cap(468, 304, '16Q/2KV × 256 · RoPE 旋 64 维 · θ = 1e7', { mono: true, size: 9.5 }));

  body.push(edge(disp, gdn, { nodes, pulse: true, dur: 1.8, delay: 0.25, uid: UID, eid: 3 }));
  body.push(edge(disp, gattn, { nodes, pulse: true, dur: 1.8, delay: 0.45, uid: UID, eid: 4 }));
  body.push(cap(432, 134, '×36', { mono: true, size: 10, color: 'var(--seal-deep)' }));
  body.push(cap(440, 214, '×12', { mono: true, size: 10, color: 'var(--seal-deep)' }));

  body.push(cap(36, 330, 'rope_theta = 1e7：所有 full 层共享频率 base；GDN 层不吃 RoPE —— 四件套全部来自 P4 零件箱（§2.4）', { size: 10.5 }));

  return scene(720, 344, body, { label: 'Qwen3-Next：config 两字段派生 48 项 layer_types，装配 36 个 GDN 层与 12 个 full attention 层' }).svg();
}

/* ============================================================
 * 5.4 §2.1 — GLM 2021 最小实例：Part A / Part B（block 0）
 * 原文（0-indexed，与 3.1 §4.3 同例）：
 * Part A：x1 [M] x4 [M] x6（双向）；Part B：[s] x2 x3 [s] x5（乱序 span，span 内因果）
 * p1 = 空白地址；p2 = 空内偏移；ℓ 只落 span 内容 token（监督位）
 * ============================================================ */
function glmParts() {
  const UID = 'gm';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 28, 'GLM 最小实例：span 塌缩成 [M] 得 Part A，乱序拼接成 Part B（3.1 §4.3 同例）', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  // ---- Part A ----
  body.push(cap(48, 52, 'Part A · 全双向', { size: 11.5, color: 'var(--ink)', weight: 600 }));
  body.push(zone({ x: 48, y: 64, w: 320, h: 64 }));
  const toksA = ['x₁', '[M]', 'x₄', '[M]', 'x₆'];
  const centersA = [];
  toksA.forEach((t, i) => {
    const x = 58 + i * 66;
    centersA.push(x + 22);
    const n = node({ id: `a${i}`, x, y: 78, w: 44, h: 34, role: t === '[M]' ? 'cache' : 'data', title: t, mono: true, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
  });

  // ---- 分隔 ----
  body.push(`<line x1="384" y1="44" x2="384" y2="200" stroke="rgb(24 27 32 / 0.28)" stroke-width="1.1" stroke-dasharray="5 6"/>`);

  // ---- Part B ----
  body.push(cap(398, 52, 'Part B · 乱序 span（span 内因果）', { size: 11.5, color: 'var(--ink)', weight: 600 }));
  body.push(zone({ x: 396, y: 64, w: 188, h: 64, tone: 'seal' }));
  body.push(zone({ x: 594, y: 64, w: 122, h: 64, tone: 'seal' }));
  const toksB = ['[s]', 'x₂', 'x₃', '[s]', 'x₅'];
  const centersB = [];
  const cardsB = [];
  toksB.forEach((t, i) => {
    const x = 402 + i * 66;
    centersB.push(x + 22);
    const n = node({ id: `b${i}`, x, y: 78, w: 44, h: 34, role: t === '[s]' ? 'cache' : 'data', title: t, mono: true, titleSize: 12.5 });
    nodes.push(n); cardsB.push(n); body.push(n.svg);
  });
  // 生成序：脉冲沿链流动（跨 span 亦因果可见）
  for (let i = 0; i < 4; i++) {
    body.push(edge(cardsB[i], cardsB[i + 1], { nodes, pulse: true, dur: 1.4, delay: 0.4 + i * 0.45, uid: UID, eid: i }));
  }
  body.push(cap(490, 143, 'span 1', { size: 9.5, anchor: 'middle' }));
  body.push(cap(655, 143, 'span 2', { size: 9.5, anchor: 'middle' }));

  // ---- 位置/监督 id 行（两板对齐）----
  const rows = [
    { name: 'p1', a: ['0', '1', '2', '3', '4'], b: ['1', '1', '1', '3', '3'] },
    { name: 'p2', a: ['0', '0', '0', '0', '0'], b: ['0', '1', '2', '0', '1'] },
    { name: 'ℓ', a: ['0', '0', '0', '0', '0'], b: ['0', '1', '1', '0', '1'] },
  ];
  rows.forEach((r, ri) => {
    const y = 162 + ri * 17;
    body.push(cap(36, y, r.name, { mono: true, size: 10 }));
    r.a.forEach((v, i) => body.push(cap(centersA[i], y, v, { mono: true, size: 10.5, anchor: 'middle', color: 'var(--ink-faint)' })));
    r.b.forEach((v, i) => {
      const hot = r.name === 'ℓ' && v === '1';
      body.push(cap(centersB[i], y, v, { mono: true, size: 10.5, anchor: 'middle', color: hot ? 'var(--seal-deep)' : 'var(--ink-faint)', weight: hot ? 700 : undefined }));
    });
  });

  body.push(cap(36, 226, 'p1 = 空白地址 · p2 = 空内偏移 · ℓ = 监督位（只落 span 内容 token，3.1 §4.3）', { size: 10.5 }));
  body.push(cap(36, 246, '同一权重集：Part A 双向理解 ⊕ Part B 因果生成；代价 = 564 前向位置换 77 监督位（CLM 的 7.3×，§2.1）', { size: 10.5 }));

  return scene(720, 260, body, { label: 'GLM：Part A 双向、Part B 乱序 span 因果，p1/p2 二维寻址，ℓ 标记被监督的 span 内容' }).svg();
}

/* ============================================================
 * 5.6 §2.2 — per-token cache 斜率与 1M 窗整窗账（block 0）
 * 原文五行数字照抄；条长按斜率线性；色 = 三条路线 + GLM 对照
 * ============================================================ */
function cacheSlopes() {
  const UID = 'cb';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 30, 'per-token cache 斜率（bf16，全量驻留口径）与 1M 窗整窗账', { size: 12.5, color: 'var(--ink)', weight: 600 }));
  body.push(cap(180, 56, '斜率（bar ∝ KiB/token）', { size: 10 }));
  body.push(cap(508, 56, '整窗账', { size: 10 }));

  const rows = [
    { name: 'GLM-4.6', v: 368, tone: 'ink', main: '@1M → 368 GiB', sub: '声明窗 202752 → 71.1 GiB' },
    { name: 'Llama-4 Scout', v: 192, tone: 'soft', main: '@1M → 192 GiB', sub: '@10M 全量口径 ≈ 1.9 TiB/序列' },
    { name: 'DeepSeek-V3.2', v: 68.6, tone: 'warm', main: '@1M → 68.6 GiB', sub: '工作窗 131072 → 9.2 GB' },
    { name: 'Qwen3-Next', v: 24, tone: 'seal', main: '@1M → 24 GiB', sub: '+ 36 MiB 定长状态（GDN）' },
    { name: 'Kimi Linear', v: 7.9, tone: 'seal', main: '@1M → 7.9 GiB', sub: '+ 20 MiB 定长（4.4 §7）' },
  ];
  rows.forEach((r, i) => {
    const ym = 84 + i * 42;
    body.push(cap(36, ym - 1, r.name, { mono: true, size: 11.5, color: 'var(--ink)' }));
    body.push(cap(36, ym + 15, `${r.v} KiB/token`, { mono: true, size: 10 }));
    const w = Math.max(10, r.v * 0.82);
    body.push(`<rect class="dia-bar dia-bar-${r.tone} sig-bar" style="--i:${(0.2 + i * 0.9).toFixed(1)}" x="180" y="${ym - 8}" width="${w.toFixed(1)}" height="16" rx="4"/>`);
    body.push(cap(508, ym - 1, r.main, { mono: true, size: 11, color: 'var(--ink)' }));
    body.push(cap(508, ym + 15, r.sub, { size: 10 }));
  });

  body.push(`<line x1="36" y1="276" x2="684" y2="276" stroke="var(--rule-soft)" stroke-width="1"/>`);
  const legend = [
    { x: 36, tone: 'ink', text: '纯 GQA（不压斜率）' },
    { x: 172, tone: 'soft', text: 'chunk + NoPE（压可见窗）' },
    { x: 336, tone: 'warm', text: 'MLA + DSA（压斜率 / 压读取）' },
    { x: 530, tone: 'seal', text: 'linear + full hybrid（压层数）' },
  ];
  legend.forEach((l) => {
    body.push(`<rect class="dia-bar dia-bar-${l.tone}" x="${l.x}" y="285" width="12" height="12" rx="2.5"/>`);
    body.push(cap(l.x + 18, 295, l.text, { size: 10 }));
  });

  return scene(720, 308, body, { label: '五家模型每 token cache 斜率对比：纯 GQA 最大，hybrid 最小，条长按斜率线性' }).svg();
}

/* ============================================================
 * 5.6 §4.2 — config.json → configuration_*.py → modeling_*.py（block 1）
 * 原文三落点：*Attention.forward（投影 shape / RoPE 施加位置 / cache 读写）、
 * *MLP/*MoE block（打分 → top-k → 归一化 → 加权；shared 是否常驻）、
 * *Model/*ForCausalLM（层栈装配 / norm 拓扑 / 输出头 tying / MTP 挂点）
 * ============================================================ */
function readingPath() {
  const UID = 'rp';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 30, 'config → configuration → modeling：声明值如何成为真实计算图', { size: 12.5, color: 'var(--ink)', weight: 600 }));

  const c1 = node({ id: 'c1', x: 40, y: 64, w: 180, h: 60, role: 'data', title: 'config.json', sub: '声明值', mono: true, titleSize: 13 });
  const c2 = node({ id: 'c2', x: 280, y: 64, w: 200, h: 60, role: 'op', title: 'configuration_*.py', sub: '默认值 + 派生字段', mono: true, titleSize: 12 });
  const c3 = node({ id: 'c3', x: 540, y: 64, w: 166, h: 60, role: 'proj', title: 'modeling_*.py', sub: '真实计算图', mono: true, titleSize: 12.5 });
  nodes.push(c1, c2, c3);
  body.push(c1.svg, c2.svg, c3.svg);
  body.push(edge(c1, c2, { nodes, pulse: true, dur: 1.8, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(c2, c3, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: UID, eid: 1 }));
  body.push(cap(250, 56, '逐字段读入', { size: 10, anchor: 'middle' }));
  body.push(cap(510, 56, '派生（layer_types 等）', { size: 10, anchor: 'middle' }));

  body.push(cap(36, 158, 'modeling_*.py 的三处落点（各配本书走读实例）', { size: 10.5 }));

  const t1 = node({ id: 't1', x: 36, y: 176, w: 210, h: 80, role: 'attn', title: '*Attention.forward', sub: '投影 shape · RoPE 施加 · cache 读写', mono: true, titleSize: 11.5, subSize: 10.5 });
  const t2 = node({ id: 't2', x: 258, y: 176, w: 210, h: 80, role: 'op', title: '*MLP / *MoE block', sub: '打分 → top-k → 归一 → 加权', mono: true, titleSize: 11.5, subSize: 10.5 });
  const t3 = node({ id: 't3', x: 480, y: 176, w: 210, h: 80, role: 'plain', title: '*Model / *ForCausalLM', sub: '层栈装配 · norm 拓扑 · tying · MTP 挂点', mono: true, titleSize: 11, subSize: 10.5 });
  nodes.push(t1, t2, t3);
  body.push(t1.svg, t2.svg, t3.svg);
  [t1, t2, t3].forEach((t, i) => {
    body.push(edge(c3, t, { nodes, portFrom: c3.bottom, portTo: t.top, pulse: true, dur: 2.0, delay: 0.6 + i * 0.3, uid: UID, eid: `f${i}` }));
  });

  body.push(cap(36, 282, 'shared expert 是否常驻、归一化位置（norm_topk_prob）：config 字段各有代码落点（5.3 §4.3 示范「config 一句话 = 代码三行」）', { size: 10.5 }));

  return scene(720, 296, body, { label: 'HF 阅读路径：config 声明值经 configuration 派生，落到 modeling 的 attention、MoE、model 三处' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '05-case-studies/01-llama-lineage',
    replacesBlock: 0,
    title: 'iRoPE：12 × 4 层周期与可见集分工',
    svg: llamaIrope,
  },
  {
    slug: '05-case-studies/01-llama-lineage',
    replacesBlock: 1,
    title: 'early fusion：视觉 token 进同一条 residual stream',
    svg: llamaEarlyFusion,
  },
  {
    slug: '05-case-studies/02-deepseek',
    replacesBlock: 0,
    title: 'DeepSeek-V3 层栈：3 dense + 58 MoE + MTP',
    svg: deepseekV3Stack,
  },
  {
    slug: '05-case-studies/03-qwen',
    replacesBlock: 0,
    title: 'Qwen3-Next：config → layer_types → 层装配',
    svg: qwenAssembly,
  },
  {
    slug: '05-case-studies/04-glm',
    replacesBlock: 0,
    title: 'GLM objective：Part A / Part B 最小实例',
    svg: glmParts,
  },
  {
    slug: '05-case-studies/04-glm',
    replacesBlock: 1,
    title: 'GLM-4.6 层栈：3 dense + 89 MoE + MTP',
    svg: glmStack,
  },
  {
    slug: '05-case-studies/05-kimi-moonshot',
    replacesBlock: 0,
    title: 'Kimi-K2 层栈：1 dense + 60 MoE + MuonClip',
    svg: kimiK2Stack,
  },
  {
    slug: '05-case-studies/06-cross-lab-comparison',
    replacesBlock: 0,
    title: '五家 cache 斜率与 1M 窗整窗账',
    svg: cacheSlopes,
  },
  {
    slug: '05-case-studies/06-cross-lab-comparison',
    replacesBlock: 1,
    title: 'config → configuration → modeling 阅读路径',
    svg: readingPath,
  },
];
