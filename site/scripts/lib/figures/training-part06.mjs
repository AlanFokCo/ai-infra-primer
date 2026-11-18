// training Part 6（06-megatron-and-stacks）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；脉冲渐变用页面级 PULSE_GRAD_ID（勿自建 per-scene defs）。
// 数字与机制措辞全部来自章节原文（6.1 §6 接线图、6.2 §1 rank 网格、6.3 §1/§5）。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 6.1 §6 — TransformerLayer 的 TP 接线走读（block 1）
 * 原文：x → norm → f → linear_qkv(列切) → RoPE/core_attn(按 head，零通信)
 * → linear_proj(行切) → g① → norm → f → fc1 → act → fc2 → g② → y
 * ============================================================ */
function megatronLayer() {
  const UID = 'tr61a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 26, '一层 TransformerLayer 的 TP 走读（张量布局 [s,b,h]）：通信只发生在「回到全量」的两个 g 点', { size: 12, color: 'var(--ink)', weight: 600 }));

  const mk = (id, x, y, w, role, title, sub, mono = false, ts = 11.5) => {
    const n = node({ id, x, y, w, h: 42, role, title, sub, mono, titleSize: ts, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  };

  // 左列：attention 支路（自上而下）
  const a1 = mk('x', 60, 52, 250, 'data', 'x [s,b,h]', 'TP 组内全量；SP 时残差流 [s/t,b,h]', true);
  const a2 = mk('f1', 60, 114, 250, 'cache', 'f：前向恒等 / 反向 all-reduce', 'SP 时此处换成 all-gather（g 转换子）', true, 11);
  const a3 = mk('qkv', 60, 176, 250, 'proj', 'linear_qkv · ColumnParallel', 'gather_output=False · 每 rank 持若干 head 行块', true, 11);
  const a4 = mk('attn', 60, 238, 250, 'attn', 'RoPE + core_attention', '逐 head 逐 position · 本地 · 零通信', false, 11);
  const a5 = mk('proj', 60, 300, 250, 'proj', 'linear_proj · RowParallel', 'input_is_parallel=True · 输出部分和', true, 11);
  const a6 = mk('g1', 60, 362, 250, 'attn', 'g①：all-reduce（第 1 次集合通信）', 'SP 时换 reduce-scatter（ḡ）· +残差', true, 11);

  // 右列：MLP 支路（自上而下）
  const b1 = mk('norm2', 400, 52, 260, 'norm', 'pre_mlp_layernorm', '逐 token · 权重复制 · SP 区段', false, 11);
  const b2 = mk('f2', 400, 114, 260, 'cache', 'f：前向恒等 / 反向 all-reduce', '', true, 11);
  const b3 = mk('fc1', 400, 176, 260, 'proj', 'linear_fc1 · ColumnParallel', '[h → h_ff] · GLU 时 stride=2 交错持 [gate,up] 对', true, 11);
  const b4 = mk('act', 400, 238, 260, 'op', 'activation', '逐元素 · 本地 · 零通信', false, 11);
  const b5 = mk('fc2', 400, 300, 260, 'proj', 'linear_fc2 · RowParallel', '[h_ff → h] · input_is_parallel=True', true, 11);
  const b6 = mk('g2', 400, 362, 260, 'attn', 'g②：all-reduce（第 2 次集合通信）→ y', 'SP 时换 reduce-scatter · +残差', true, 11);

  const chain = [a1, a2, a3, a4, a5, a6];
  for (let i = 0; i < chain.length - 1; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, portFrom: chain[i].bottom, portTo: chain[i + 1].top, pulse: true, dur: 1.4, delay: i * 0.3, uid: UID, eid: `a${i}`, arrow: false }));
  }
  const chainB = [b1, b2, b3, b4, b5, b6];
  for (let i = 0; i < chainB.length - 1; i++) {
    body.push(edge(chainB[i], chainB[i + 1], { nodes, portFrom: chainB[i].bottom, portTo: chainB[i + 1].top, pulse: true, dur: 1.4, delay: 1.0 + i * 0.25, uid: UID, eid: `b${i}`, arrow: false }));
  }
  // g① → norm2：走两列之间的中央通道（下行 → 横渡 → 上行）
  const sPath = 'M 310 383 C 340 388, 356 380, 356 352 L 356 104 C 356 80, 368 73, 400 73';
  body.push(`<path class="dia-edge strong" d="${sPath}"/>`);
  body.push(pathPulse(sPath, { dur: 2.4, delay: 1.8 }));
  // 两条残差旁路（外弧，虚线）
  body.push(`<path class="dia-edge dash" d="M 168 52 C 48 60, 44 340, 60 372"/>`);
  body.push(`<path class="dia-edge dash" d="M 660 73 C 702 73, 702 383, 660 383"/>`);
  body.push(cap(52, 220, '+残差', { size: 9.5, anchor: 'start' }));
  body.push(cap(694, 220, '+残差', { size: 9.5, anchor: 'end' }));

  body.push(cap(36, 436, '两个 g 点 = 5.1「每层两次 all-reduce」的全部集合通信；中间的分片区段（逐元素激活、attention 本体）零通信', { size: 11 }));
  body.push(cap(36, 456, 'bias 不切分、只在 g 之后加一次（先加再 all-reduce 会被放大 t 倍）；GQA：KV 组数 < t 时走复制分支', { size: 10.5 }));

  return scene(720, 472, body, { label: 'Megatron TransformerLayer 的 TP 接线：f/g 转换子夹住两条列切-行切支路' }).svg();
}

/* ============================================================
 * 6.2 §1 — 16 rank 排布网格（block 0，中优先级）
 * 原文：W=16，t=2，d=2，p=4 → r = i_tp + 2·i_dp + 4·i_pp；
 * TP 组 stride 1 落节点内，PP 组 stride 4 跨节点
 * ============================================================ */
function rankGrid() {
  const UID = 'tr62a';
  const body = [];

  body.push(cap(36, 28, 'W = 16、t = 2、d = 2、p = 4：r = i_tp + 2·i_dp + 4·i_pp（最左维 stride 最小）', { size: 12, color: 'var(--ink)', weight: 600 }));

  // 网格：行 = i_dp，列组 = i_pp，组内两格 = i_tp
  const CWd = 40, GAP = 4, GROUP_GAP = 26, X0 = 130, Y0 = 96, ROW_GAP = 22;
  const cellX = (g, t) => X0 + g * (2 * (CWd + GAP) + GROUP_GAP) + t * (CWd + GAP);
  for (let g = 0; g < 4; g++) body.push(cap(cellX(g, 0) + CWd + GAP / 2, Y0 - 12, `i_pp = ${g}`, { mono: true, size: 10, anchor: 'middle' }));
  for (let d = 0; d < 2; d++) {
    body.push(cap(X0 - 14, Y0 + d * (CWd + ROW_GAP) + CWd / 2 + 4, `i_dp = ${d}`, { mono: true, size: 10, anchor: 'end' }));
    for (let g = 0; g < 4; g++) {
      for (let t = 0; t < 2; t++) {
        const r = t + 2 * d + 4 * g;
        const x = cellX(g, t), y = Y0 + d * (CWd + ROW_GAP);
        body.push(`<g class="dia-node dia-node-proj"><rect x="${x}" y="${y}" width="${CWd}" height="${CWd}" rx="8"/><text class="dia-title dia-mono" x="${x + CWd / 2}" y="${y + CWd / 2 + 5}" text-anchor="middle" font-size="13">${r}</text></g>`);
      }
      // TP 组括弧（组内两格）
      const bx1 = cellX(g, 0), bx2 = cellX(g, 1) + CWd;
      const by = Y0 + 2 * (CWd + ROW_GAP) + 4;
      body.push(`<path class="sig-cell" style="--i:${(0.3 + g * 0.2).toFixed(1)}" d="M ${bx1} ${by} V ${by + 8} H ${bx2} V ${by}" fill="none" stroke="var(--seal-deep)" stroke-width="1.4"/>`);
    }
  }
  body.push(cap(X0 - 14, Y0 + 2 * (CWd + ROW_GAP) + 16, 'TP 组', { mono: true, size: 10, anchor: 'end', color: 'var(--seal-deep)' }));

  // PP 组弧线（顶部，同一 (d,t) 跨 4 组）：以 (d=0,t=0) 为例高亮 {0,4,8,12}
  const pts = [0, 1, 2, 3].map((g) => cellX(g, 0) + CWd / 2);
  body.push(`<path class="sig-dashflow" d="M ${pts[0]} ${Y0 - 6} C ${pts[0]} ${Y0 - 46}, ${pts[3]} ${Y0 - 46}, ${pts[3]} ${Y0 - 6}" fill="none" stroke="rgb(24 27 32 / 0.45)" stroke-width="1.4"/>`);
  body.push(cap((pts[0] + pts[3]) / 2, Y0 - 52, 'PP 组 {0,4,8,12}（stride 4，跨节点）', { mono: true, size: 9.5, anchor: 'middle' }));

  // DP 组竖直连线（同列组同 t 的上下两格），以 g=0 为例
  body.push(`<path class="sig-dashflow" d="M ${cellX(0, 0) + CWd / 2} ${Y0 + CWd} V ${Y0 + CWd + ROW_GAP}" fill="none" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(cap(cellX(0, 0) + CWd / 2 + 10, Y0 + CWd + ROW_GAP / 2 + 4, 'DP 组 {0,2}（stride 2）', { mono: true, size: 9 }));

  body.push(cap(36, 252, 'TP 每层两次 all-reduce、通信最频 → stride 1，整组收进同节点 NVLink 域；PP 只传边界激活 → stride 最大、跨节点', { size: 11 }));
  body.push(cap(36, 272, 'order 参数可重排（如 "tp-pp-dp"），进程组树整体重算；EP 不进这张 dense 网格，由 expert 侧 generator 另建', { size: 10.5 }));

  return scene(720, 288, body, { label: 'Megatron 16 rank 排布：i_dp × i_pp 网格与 TP/DP/PP 三族组的 stride' }).svg();
}

/* ============================================================
 * 6.3 §1 — Megatron 家族关系（block 0，中优先级）
 * 原文：NeMo（框架层）与 Megatron-LM（参考实现）汇入 Megatron-Core；
 * Megatron-Bridge 桥接 HF 生态（双向转换）
 * ============================================================ */
function megatronFamily() {
  const UID = 'tr63a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'Megatron 家族三层分工：Core 是可组合库，NeMo / Megatron-LM 是两侧封装，Bridge 桥接 HF 生态', { size: 12, color: 'var(--ink)', weight: 600 }));

  const core = node({ id: 'core', x: 280, y: 140, w: 180, h: 64, role: 'attn', title: 'Megatron-Core', sub: '库：5D 并行 + dist-opt + dist-ckpt', titleSize: 12.5, subSize: 9.5 });
  const nemo = node({ id: 'nemo', x: 44, y: 64, w: 176, h: 56, role: 'op', title: 'NeMo', sub: '框架层：recipe/数据/评测/导出', titleSize: 13, subSize: 9.5 });
  const lm = node({ id: 'lm', x: 44, y: 196, w: 176, h: 56, role: 'op', title: 'Megatron-LM', sub: '参考实现：Core + 预配置脚本', titleSize: 12.5, subSize: 9.5 });
  const bridge = node({ id: 'bridge', x: 520, y: 56, w: 160, h: 56, role: 'proj', title: 'Megatron-Bridge', sub: '双向 checkpoint 转换', titleSize: 11.5, subSize: 9.5 });
  const hf = node({ id: 'hf', x: 520, y: 204, w: 160, h: 56, role: 'data', title: 'HF 生态', sub: '模型定义 / 权重格式', titleSize: 13, subSize: 9.5 });
  nodes.push(core, nemo, lm, bridge, hf);
  body.push(core.svg, nemo.svg, lm.svg, bridge.svg, hf.svg);

  body.push(edge(nemo, core, { nodes, pulse: true, dur: 1.8, delay: 0.2, uid: UID, eid: 0 }));
  body.push(edge(lm, core, { nodes, pulse: true, dur: 1.8, delay: 0.5, uid: UID, eid: 1 }));
  // Bridge → Core 改从 Core 顶边进（R2 复评：原 right 口入线斜穿 Core 标题文本）
  body.push(edge(bridge, core, { nodes, portFrom: bridge.left, portTo: core.top, pulse: true, dur: 1.8, delay: 0.8, uid: UID, eid: 2 }));
  // Bridge ↔ HF 双向
  body.push(edge(bridge, hf, { nodes, portFrom: bridge.bottom, portTo: hf.top, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 3, arrow: false }));
  body.push(edge(hf, bridge, { nodes, portFrom: hf.top, portTo: bridge.bottom, pulse: true, dur: 1.6, delay: 1.0, uid: UID, eid: 4, arrow: false }));
  body.push(cap(700, 168, '双向转换', { size: 9.5, anchor: 'end', color: 'var(--ink-faint)' }));

  body.push(cap(36, 296, '典型用法：HF 开放权重 → Bridge → Core 大规模继续训练/微调 → 转回 HF 发布；日常打交道最多的是 Core 库接口与 LM 参考脚本', { size: 10.5 }));
  body.push(cap(36, 316, '组织变动（2026-11 核实）：NVIDIA/NeMo 301 重定向至 NVIDIA-NeMo/Speech；Bridge 等拆为 NVIDIA-NeMo 组织下独立库', { size: 10.5 }));

  return scene(720, 332, body, { label: 'Megatron 家族：NeMo 与 Megatron-LM 汇入 Megatron-Core，Bridge 桥接 HF 生态' }).svg();
}

/* ============================================================
 * 6.3 §5 — 训练栈 ↔ 推理栈接口（block 1，中优先级）
 * 原文：prompts + 经验数据下行，权重同步（每 iteration 广播）上行；
 * rollout 引擎内部机制归 inference 维度
 * ============================================================ */
function trainInferLoop() {
  const UID = 'tr63b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 28, 'RL 场景的两栈常态组合：训练栈产出与消费 checkpoint；rollout 引擎内部机制归 inference 维度', { size: 12, color: 'var(--ink)', weight: 600 }));

  body.push(zone({ x: 36, y: 52, w: 292, h: 200, label: '训练栈（本 Part 各栈）' }));
  const actor = node({ id: 'actor', x: 60, y: 108, w: 244, h: 64, role: 'attn', title: 'actor 训练引擎', sub: 'forward / backward / optimizer', mono: true, titleSize: 13 });
  nodes.push(actor); body.push(actor.svg);
  body.push(cap(182, 200, '产出与消费 checkpoint', { size: 10, anchor: 'middle' }));
  body.push(cap(182, 218, 'Megatron-Core / TorchTitan / DeepSpeed …', { mono: true, size: 9.5, anchor: 'middle' }));

  body.push(zone({ x: 400, y: 52, w: 292, h: 200, label: '推理栈', tone: 'seal' }));
  const roll = node({ id: 'roll', x: 424, y: 108, w: 244, h: 64, role: 'proj', title: 'rollout 引擎（vLLM 等）', sub: '批量采样生成', titleSize: 12.5 });
  nodes.push(roll); body.push(roll.svg);
  body.push(cap(546, 200, 'continuous batching / PagedAttention', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(546, 218, '内部机制归 inference 维度，只链接', { size: 9.5, anchor: 'middle' }));

  // 权重同步：训练 → 推理（上通道）
  body.push(edge(actor, roll, { nodes, portFrom: actor.top, portTo: roll.top, pulse: true, dur: 2.2, delay: 0.3, uid: UID, eid: 0, strong: true }));
  body.push(cap(364, 84, '权重同步（每 iteration 广播，8.5 的账）', { size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));
  // prompts + 经验数据：推理 → 训练（下通道，向下绕弧）
  const ret = 'M 546 172 C 546 224, 182 224, 182 172';
  body.push(`<path class="dia-edge" d="${ret}"/>`);
  body.push(pathPulse(ret, { dur: 2.2, delay: 1.1 }));
  body.push(cap(364, 232, 'prompts + 经验数据（rollout 产出）', { size: 10, anchor: 'middle' }));

  body.push(cap(36, 288, '训练侧要关心的只是每 iteration 的权重广播成本；训练段与 rollout 段分开选型是后训练系统设计的常态组合', { size: 11 }));

  return scene(720, 304, body, { label: '训练栈与推理栈的接口：数据下行、权重上行成环' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { dim: 'training', slug: '06-megatron-and-stacks/01-megatron-architecture', replacesBlock: 1, title: 'TransformerLayer 的 TP 接线走读', svg: megatronLayer },
  { dim: 'training', slug: '06-megatron-and-stacks/02-megatron-runtime', replacesBlock: 0, title: '16 rank 排布：三族组的 stride', svg: rankGrid },
  { dim: 'training', slug: '06-megatron-and-stacks/03-modern-training-stacks', replacesBlock: 0, title: 'Megatron 家族：Core / NeMo / Bridge', svg: megatronFamily },
  { dim: 'training', slug: '06-megatron-and-stacks/03-modern-training-stacks', replacesBlock: 1, title: '训练栈 ↔ 推理栈：三条流成环', svg: trainInferLoop },
];
