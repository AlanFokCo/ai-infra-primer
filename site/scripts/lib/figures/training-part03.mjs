// training Part 3（03-loss-functions）场景图：替换各章无语言围栏 ASCII 块
// 规范见 STYLE.md；组件库见 figure-kit.mjs；脉冲渐变用页面级 PULSE_GRAD_DEFS 固定 id。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 3.1 §1（block 0）— CE 账流水线：shape 走读 + FLOPs / 字节账
 * 原文：H [B,T,d] → LM 头 matmul（2BTdV FLOPs）→ logits z [B,T,V]（2BTV bytes，bf16）
 * → 行内 logsumexp（式 3）→ 逐 token ℓ [B,T] → ⊙ mask m [B,T] → L 标量（式 2）
 * ============================================================ */
function cePipeline() {
  const UID = 't3a';
  const body = [];
  const nodes = [];
  const CX = 250, CW = 210, CH = 42;

  body.push(cap(36, 22, '训练侧 CE 的 shape 走读：logits 的 FLOPs 与字节两笔账都在词表维 V 上放大', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, y, title, sub, role = 'data') => {
    const n = node({ id, x: CX - CW / 2, y, w: CW, h: CH, role, title, sub, mono: true, titleSize: 12, subSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const H = mk('H', 52, 'H [B, T, d]', '隐状态');
  const LM = mk('LM', 128, 'LM 头 matmul', 'W_out [d, V]', 'proj');
  const Z = mk('Z', 204, 'logits z [B, T, V]', '未归一化（文档口径）');
  const LS = mk('LS', 280, '行内 logsumexp', '式 (3) · autocast 下 fp32', 'op');
  const EL = mk('EL', 356, '逐 token ℓ [B, T]', '−z_y + logsumexp(z)');
  const L = mk('L', 432, 'L 标量', '式 (2)：Σ m·ℓ / N_tok', 'attn');

  const chain = [[H, LM], [LM, Z], [Z, LS], [LS, EL], [EL, L]];
  chain.forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: i * 0.3, uid: UID, eid: `c${i}` }));
  });

  // 账徽标（右栏，朱砂 mono）
  body.push(cap(400, 146, '2·B·T·d·V FLOPs', { mono: true, size: 11, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(400, 164, '输出头投影：V = 128256 放大的第一站', { size: 9.5 }));
  body.push(cap(400, 222, '2·B·T·V bytes（bf16）', { mono: true, size: 11, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(400, 240, 'B·T = 32768 时 ≈ 7.8 GiB —— 不物化版本 → 1.2 §10', { size: 9.5 }));

  // mask 侧入（ℓ → L 的边上）
  const MK = node({ id: 'mk', x: 450, y: 396, w: 210, h: 42, role: 'cache', title: 'mask m [B, T]', sub: 'prompt 段不计（→ models 8.3）', mono: true, titleSize: 11.5 });
  nodes.push(MK); body.push(MK.svg);
  body.push(edge(MK, [268, 415], { nodes, portFrom: MK.left, uid: UID, eid: 'm', dash: true }));
  body.push(cap(424, 415, '⊙', { mono: true, size: 13, anchor: 'middle' }));

  body.push(cap(36, 502, '屏蔽的是监督，不是计算：被 mask 的位置照常前向、照常产生 logits，只是不进式 (2)（§1）', { size: 10.5 }));

  return scene(720, 516, body, { label: 'CE 数据流：隐状态经 LM 头投影为 logits，logsumexp 后得逐 token 损失，mask 后归约为标量' }).svg();
}

/* ============================================================
 * 3.2 §6（block 0）— DPO 数据流（TRL 口径）
 * 原文：(x,y_w,y_l) → tokenize → batch = [chosen 半批; rejected 半批]（prompt+response
 * 拼接，completion_mask prompt 0 / response 1）→ π_θ 与 π_ref 各一次前向 → fused LM head
 * （分 tile 过输出头、按 label 出逐 token logp，logits 不物化）→ response 段求和得序列级
 * logp → logratio = logp_πθ − logp_πref（各一）→ L = −log σ(β·(logratio_w − logratio_l))
 * ============================================================ */
function dpoDataflow() {
  const UID = 't3b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 22, 'DPO 一轮前向：chosen / rejected 对半拼批，双模型各跑一次，logits 全程不物化（TRL 口径）', { size: 12, color: 'var(--ink-soft)' }));

  // Row 1：样本 → tokenize → 拼批
  const T = node({ id: 'T', x: 36, y: 52, w: 140, h: 48, role: 'data', title: '(x, y_w, y_l)', sub: '偏好对样本', mono: true, titleSize: 12 });
  const TK = node({ id: 'TK', x: 226, y: 56, w: 104, h: 40, role: 'op', title: 'tokenize', mono: true, titleSize: 11.5 });
  const BA = node({ id: 'BA', x: 386, y: 44, w: 300, h: 56, role: 'data', title: 'batch = [chosen 半批 ; rejected 半批]', sub: 'completion_mask：prompt 段 0 / response 段 1', mono: true, titleSize: 11.5, subSize: 9 });
  [T, TK, BA].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(T, TK, { nodes, pulse: true, dur: 1.3, uid: UID, eid: 'a' }));
  body.push(edge(TK, BA, { nodes, pulse: true, dur: 1.3, delay: 0.25, uid: UID, eid: 'b' }));

  // Row 2：双前向
  const PT = node({ id: 'PT', x: 416, y: 140, w: 120, h: 48, role: 'proj', title: 'π_θ 前向', mono: true, titleSize: 12 });
  const PR = node({ id: 'PR', x: 566, y: 140, w: 120, h: 48, role: 'plain', title: 'π_ref 前向', sub: '冻结', mono: true, titleSize: 12 });
  nodes.push(PT, PR); body.push(PT.svg, PR.svg);
  body.push(edge([510, 100], PT, { nodes, portTo: PT.top, pulse: true, dur: 1.3, delay: 0.5, uid: UID, eid: 'c0' }));
  body.push(edge([562, 100], PR, { nodes, portTo: PR.top, pulse: true, dur: 1.3, delay: 0.65, uid: UID, eid: 'c1' }));

  // Row 3：fused LM head（logits 不物化）
  body.push(zone({ x: 390, y: 210, w: 320, h: 86, label: 'fused LM head：logits 不物化', tone: 'seal' }));
  const LT = node({ id: 'LT', x: 406, y: 238, w: 140, h: 42, role: 'proj', title: 'logp_θ（w / l）', sub: 'response 段求和', mono: true, titleSize: 11.5, subSize: 9 });
  const LR = node({ id: 'LR', x: 556, y: 238, w: 140, h: 42, role: 'plain', title: 'logp_ref（w / l）', sub: 'response 段求和', mono: true, titleSize: 11.5, subSize: 9 });
  nodes.push(LT, LR); body.push(LT.svg, LR.svg);
  body.push(edge(PT, LT, { nodes, portFrom: PT.bottom, portTo: LT.top, pulse: true, dur: 1.2, delay: 0.9, uid: UID, eid: 'd0' }));
  body.push(edge(PR, LR, { nodes, portFrom: PR.bottom, portTo: LR.top, pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'd1' }));
  body.push(cap(550, 312, '分 tile 过输出头、按 label 直接产出逐 token logp', { size: 9.5, anchor: 'middle' }));

  // Row 4：logratio 差
  const D = node({ id: 'D', x: 400, y: 336, w: 220, h: 46, role: 'proj', title: 'logratio = logp_θ − logp_ref', sub: 'chosen / rejected 各一', mono: true, titleSize: 11.5, subSize: 9 });
  nodes.push(D); body.push(D.svg);
  body.push(edge(LT, D, { nodes, portFrom: LT.bottom, portTo: [470, 336], pulse: true, dur: 1.3, delay: 1.25, uid: UID, eid: 'e0' }));
  body.push(edge(LR, D, { nodes, portFrom: LR.bottom, portTo: [550, 336], pulse: true, dur: 1.3, delay: 1.35, uid: UID, eid: 'e1' }));

  // Row 5：loss
  const LO = node({ id: 'LO', x: 360, y: 416, w: 300, h: 48, role: 'attn', title: 'L = −log σ(β·(logratio_w − logratio_l))', sub: '式 (3) 代入 DPO 重参数化，推导 → 9.3', mono: true, titleSize: 11.5, subSize: 9 });
  nodes.push(LO); body.push(LO.svg);
  body.push(edge(D, LO, { nodes, portFrom: D.bottom, portTo: LO.top, pulse: true, dur: 1.3, delay: 1.6, uid: UID, eid: 'f' }));

  // 左侧注记
  body.push(cap(36, 160, '对半拼接共享 prompt 前缀：', { size: 9.5 }));
  body.push(cap(36, 176, '一次前向同时产出两支 logp，', { size: 9.5 }));
  body.push(cap(36, 192, 'kernel 利用率按一个 batch 计', { size: 9.5 }));
  body.push(cap(36, 240, 'offline：期望取自静态偏好集，', { size: 9.5 }));
  body.push(cap(36, 256, 'π_ref 是被求值的固定函数', { size: 9.5 }));
  body.push(cap(36, 272, '（precompute 后训练期不驻留）', { size: 9.5 }));

  body.push(cap(36, 490, '序列级 logp 是求和而非平均（源码 sum(dim=1)）—— 长回答总 logp 天然更低，长度偏置由此而来；平均化正是 SimPO 的出发点（→ 9.3）', { size: 10.5 }));

  return scene(720, 504, body, { label: 'DPO 数据流：偏好对拼批后经 policy 与 ref 双前向，fused LM head 产出序列级 logp，logratio 之差进 sigmoid 损失' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'training',
    slug: '03-loss-functions/01-ce-family',
    replacesBlock: 0,
    title: 'CE 账流水线：shape 走读与两笔 V 维账',
    svg: cePipeline,
  },
  {
    dim: 'training',
    slug: '03-loss-functions/02-preference-and-contrastive-objectives',
    replacesBlock: 0,
    title: 'DPO 数据流：对半拼批 × 双前向 × fused head',
    svg: dpoDataflow,
  },
];
