// training Part 8（08-rl-math）场景图：MDP 状态链 / actor-critic 数据流 / PPO clip 双图 / PPO 循环 / GRPO 对照 / RL iteration / single-controller
// 规范见 STYLE.md；组件库见 figure-kit.mjs。脉冲统一用页面级 PULSE_GRAD_ID（Chapter.astro / fig-check 注入 defs）。
// 内容全部来自章节原文（数字、机制措辞）；replacesBlock 口径见 docs/reviews/training-figure-census.md。

import { scene, node, edge, zone, cap, PULSE_GRAD_ID } from '../figure-kit.mjs';

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// 手工路径上的脉冲（kit 的 edge 只支持自动贝塞尔；L 形/折线通道用 path 直写），缓动用 spline 变速
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 8.1 §2.2（block 0，中）— LLM-as-MDP 状态链
 * s_0=x 逐 token 确定性转移，途中 r=0，末端 r_{T-1}=R(x,y)（RM/verifier 不可微黑盒）
 * ============================================================ */
function mdpChain() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'LLM 即 MDP：状态 = prompt + 已生成前缀；奖励集中在末端，途中 r = 0（§2.2 五元组逐项落地）', { size: 12, color: 'var(--ink-soft)' }));

  const s0 = add(node({ id: 's0', x: 20, y: 60, w: 66, h: 46, role: 'data', title: 's₀ = x', mono: true, titleSize: 11.5 }));
  const s1 = add(node({ id: 's1', x: 132, y: 60, w: 108, h: 46, role: 'data', title: 's₁ = (x, y₁)', mono: true, titleSize: 11 }));
  const s2 = add(node({ id: 's2', x: 286, y: 60, w: 140, h: 46, role: 'data', title: 's₂ = (x, y₁, y₂)', mono: true, titleSize: 11 }));
  const sT = add(node({ id: 'sT', x: 520, y: 60, w: 116, h: 46, role: 'data', title: 's_T = (x, y)', mono: true, titleSize: 11.5 }));

  body.push(edge(s0, s1, { nodes, pulse: true, dur: 1.8, delay: 0, uid: 'p8a', eid: 0 }));
  body.push(edge(s1, s2, { nodes, pulse: true, dur: 1.8, delay: 0.3, uid: 'p8a', eid: 1 }));
  body.push(edge([426, 83], [470, 83], { nodes, uid: 'p8a', eid: 2 }));
  body.push(edge([494, 83], sT, { nodes, pulse: true, dur: 1.8, delay: 0.6, uid: 'p8a', eid: 3 }));
  body.push(cap(482, 88, '⋯', { mono: true, size: 14, anchor: 'middle' }));
  body.push(cap(109, 52, 'a₀ = y₁', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(263, 52, 'a₁ = y₂', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(109, 128, 'r₀ = 0', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(263, 128, 'r₁ = 0', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(cap(482, 128, 'r = 0', { mono: true, size: 9.5, anchor: 'middle' }));

  // 末端奖励：RM / verifier 黑盒
  const rm = add(node({ id: 'rm', x: 506, y: 168, w: 144, h: 52, role: 'cache', title: 'R(x, y)', mono: true, titleSize: 12.5, sub: 'RM / verifier · 不可微黑盒', subSize: 9.5 }));
  body.push(edge(sT, rm, { nodes, portFrom: sT.bottom, portTo: rm.top, pulse: true, dur: 1.6, delay: 0.9, uid: 'p8a', eid: 4, strong: true }));
  body.push(cap(586, 142, 'r_{T−1} = R(x, y)', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(20, 166, 'π_θ(·|s_t) = LM 在位置 t 的 softmax 输出', { mono: true, size: 10 }));
  body.push(cap(20, 184, '转移 P = 确定性拼接（概率 1）· γ 惯例取 1', { mono: true, size: 10 }));

  body.push(cap(24, 244, '环境无未知动力学：全部随机性由 π_θ 自己携带，「探索」= 采样温度下的 token 抽样；token 级与序列级（contextual bandit）', { size: 10.5 }));
  body.push(cap(24, 261, '两种粒度经链式分解严格等价 —— 无 per-token baseline 时两者梯度估计逐字相同（§7.1），差别只在 credit assignment 的工具粒度', { size: 10.5 }));

  return scene(720, 274, body, { label: 'LLM 作为 MDP：状态链逐 token 生长，奖励只在末端由 reward model 或 verifier 给出' }).svg();
}

/* ============================================================
 * 8.2 §6.1（block 0，中）— actor-critic 联合 loss 数据流
 * rollout → critic 估值 → δ_t → 反向递推 Â → 分流 L_V / L_pg → 一次 backward 同更 θ 与 φ
 * ============================================================ */
function actorCritic() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, '联合 loss 数据流（式 15）：δ 反向递推成 Â，分流进 L_V 与 L_pg —— 一次 backward 同更 θ 与 φ', { size: 12, color: 'var(--ink-soft)' }));

  const roll = add(node({ id: 'roll', x: 24, y: 50, w: 176, h: 60, role: 'data', title: 'rollout（当前 π_θ）', titleSize: 12.5, sub: 'N_traj 条 · s / a / r / done', subSize: 9.5 }));
  const critic = add(node({ id: 'critic', x: 24, y: 162, w: 176, h: 56, role: 'proj', title: 'critic 前向 V_φ(s_t)', titleSize: 12, sub: '逐帧估值' }));
  const dt = add(node({ id: 'dt', x: 268, y: 98, w: 208, h: 56, role: 'op', title: 'δ_t = r_t + γ·V(s_{t+1}) − V(s_t)', mono: true, titleSize: 10 }));
  const ah = add(node({ id: 'ah', x: 524, y: 98, w: 180, h: 56, role: 'attn', title: 'Â_t = Σ_l (γλ)^l δ_{t+l}', mono: true, titleSize: 11, sub: '反向递推（自末端往 t）', subSize: 9.5 }));
  const lv = add(node({ id: 'lv', x: 268, y: 230, w: 200, h: 56, role: 'norm', title: 'L_V = (V_φ(s_t) − R̂_t)²', mono: true, titleSize: 11, sub: 'R̂_t = Â_t + V(s_t)（冻结）', subSize: 9.5 }));
  const lpg = add(node({ id: 'lpg', x: 524, y: 230, w: 180, h: 56, role: 'attn', title: 'L_pg = −log π_θ·Â_t − c_e·H', mono: true, titleSize: 10, sub: '熵奖励维持探索（8.1 §3.1）', subSize: 9.5 }));

  body.push(edge(roll, critic, { nodes, portFrom: roll.bottom, portTo: critic.top, pulse: true, dur: 2.0, delay: 0.3, uid: 'p8b', eid: 0 }));
  body.push(edge(roll, dt, { nodes, portFrom: roll.right, portTo: [268, 112], pulse: true, dur: 2.0, delay: 0, uid: 'p8b', eid: 1 }));
  body.push(edge(critic, dt, { nodes, portFrom: critic.right, portTo: [268, 140], pulse: true, dur: 2.0, delay: 0.45, uid: 'p8b', eid: 2 }));
  body.push(edge(dt, ah, { nodes, pulse: true, dur: 2.0, delay: 0.7, uid: 'p8b', eid: 3, strong: true }));
  body.push(cap(496, 88, '逆序', { size: 9.5, anchor: 'middle' }));
  // Â → R̂ → L_V（绕顶）
  body.push(`<path class="dia-edge" d="M614 154 V192 H368 V228"/>`);
  body.push(pathPulse('M614 154 V192 H368 V228', { dur: 2.2, delay: 1.0, r: 3.6 }));
  body.push(cap(491, 188, 'R̂_t = Â_t + V(s_t)', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(ah, lpg, { nodes, portFrom: [640, 154], portTo: lpg.top, pulse: true, dur: 2.0, delay: 0.9, uid: 'p8b', eid: 4 }));

  // 汇流条：一次 backward 同更
  body.push(`<path class="dia-edge" d="M368 286 V304"/>`);
  body.push(`<path class="dia-edge" d="M614 286 V304"/>`);
  body.push(`<rect x="268" y="304" width="436" height="28" rx="14" fill="var(--paper)" stroke="var(--ink-soft)" stroke-width="1.2"/>`);
  body.push(cap(486, 323, '一次 backward 同更 θ 与 φ', { size: 11, anchor: 'middle', weight: 600, color: 'var(--ink)' }));
  body.push(pathPulse('M368 288 V312 H486', { dur: 1.4, delay: 1.4, r: 3.2 }));
  body.push(pathPulse('M614 288 V312 H486', { dur: 1.4, delay: 1.55, r: 3.2 }));

  return scene(720, 346, body, { label: 'actor-critic：rollout 经 critic 估值算 δ，反向递推成 Â，分流进价值损失与策略梯度损失，一次反向同更两塔' }).svg();
}

/* ============================================================
 * 8.3 §3（block 0，高）— PPO clip 目标分段函数双图（论文 Figure 1 重绘，A = ±2，ε = 0.2）
 * A>0：L = 2·min(r, 1.2)；A<0：L = −2·max(r, 0.8)。界内线性、界外平台；梯度区随 A 正负镜像
 * ============================================================ */
function ppoClip() {
  const body = [];

  body.push(cap(24, 24, 'clip 目标的分段图像（A = ±2，ε = 0.2）：界内线性、界外平台 —— 平台恰好截掉「继续走远」的全部收益', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 左：A > 0 ---- */
  body.push(zone({ x: 16, y: 40, w: 336, h: 226, label: 'A > 0：L = 2·min(r, 1.2)' }));
  const X1 = (r) => 56 + (r / 1.5) * 270;
  const Y1 = (L) => 226 - (L / 2.4) * 146;
  body.push(`<line class="dia-edge strong" x1="56" y1="226" x2="336" y2="226"/>`);
  body.push(`<line class="dia-edge" x1="56" y1="80" x2="56" y2="226"/>`);
  const lin1 = `M${X1(0)} ${Y1(0)} L${X1(0.5)} ${Y1(1.0)} L${X1(0.8)} ${Y1(1.6)} L${X1(1.0)} ${Y1(2.0)} L${X1(1.2)} ${Y1(2.4)}`;
  const flat1 = `M${X1(1.2)} ${Y1(2.4)} L${X1(1.5)} ${Y1(2.4)}`;
  body.push(`<path d="${lin1}" fill="none" stroke="var(--ink)" stroke-width="2.2" stroke-linecap="round"/>`);
  body.push(`<path d="${flat1}" fill="none" stroke="var(--seal)" stroke-width="3" stroke-linecap="round"/>`);
  body.push(pathPulse(lin1, { dur: 2.4, delay: 0.2, r: 3.6 }));
  [[0.5, 1.0], [0.8, 1.6], [1.0, 2.0]].forEach(([r, L], i) =>
    body.push(`<circle class="sig-cell" style="--i:${(0.2 + i * 0.2).toFixed(1)}" cx="${X1(r)}" cy="${Y1(L)}" r="3" fill="var(--ink)"/>`));
  body.push(`<circle class="sig-cell" style="--i:0.9" cx="${X1(1.2)}" cy="${Y1(2.4)}" r="4.5" fill="var(--seal)"/>`);
  body.push(`<line x1="${X1(1.2)}" y1="${Y1(2.4)}" x2="${X1(1.2)}" y2="226" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`);
  ['0.5', '0.8', '1.0', '1.2'].forEach((s) => body.push(cap(X1(Number(s)), 241, s, { mono: true, size: 9.5, anchor: 'middle' })));
  body.push(cap(336, 241, 'r', { mono: true, size: 10, anchor: 'middle' }));
  ['1.0', '2.0', '2.4'].forEach((s) => body.push(cap(48, Y1(Number(s)) + 3.5, s, { mono: true, size: 9.5, anchor: 'end' })));
  body.push(cap(164, 256, '◄—— 有梯度 ——►', { size: 9.5, anchor: 'middle' }));
  body.push(cap(304, 256, '梯度=0', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  /* ---- 右：A < 0 ---- */
  body.push(zone({ x: 368, y: 40, w: 336, h: 226, label: 'A < 0：L = −2·max(r, 0.8)' }));
  const X2 = (r) => 408 + (r / 1.5) * 270;
  const Y2 = (L) => 80 + (-L / 3.0) * 146;
  body.push(`<line x1="408" y1="${Y2(0)}" x2="688" y2="${Y2(0)}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`);
  body.push(`<line class="dia-edge strong" x1="408" y1="226" x2="688" y2="226"/>`);
  body.push(`<line class="dia-edge" x1="408" y1="80" x2="408" y2="226"/>`);
  const flat2 = `M${X2(0)} ${Y2(-1.6)} L${X2(0.8)} ${Y2(-1.6)}`;
  const lin2 = `M${X2(0.8)} ${Y2(-1.6)} L${X2(1.0)} ${Y2(-2.0)} L${X2(1.5)} ${Y2(-3.0)}`;
  body.push(`<path d="${flat2}" fill="none" stroke="var(--seal)" stroke-width="3" stroke-linecap="round"/>`);
  body.push(`<path d="${lin2}" fill="none" stroke="var(--ink)" stroke-width="2.2" stroke-linecap="round"/>`);
  body.push(pathPulse(lin2, { dur: 2.4, delay: 0.6, r: 3.6 }));
  body.push(`<circle class="sig-cell" style="--i:0.9" cx="${X2(0.8)}" cy="${Y2(-1.6)}" r="4.5" fill="var(--seal)"/>`);
  [[1.0, -2.0], [1.5, -3.0]].forEach(([r, L], i) =>
    body.push(`<circle class="sig-cell" style="--i:${(1.1 + i * 0.2).toFixed(1)}" cx="${X2(r)}" cy="${Y2(L)}" r="3" fill="var(--ink)"/>`));
  body.push(`<line x1="${X2(0.8)}" y1="${Y2(-1.6)}" x2="${X2(0.8)}" y2="226" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 4"/>`);
  ['0.5', '0.8', '1.0'].forEach((s) => body.push(cap(X2(Number(s)), 241, s, { mono: true, size: 9.5, anchor: 'middle' })));
  body.push(cap(688, 241, 'r', { mono: true, size: 10, anchor: 'middle' }));
  body.push(cap(400, Y2(0) + 3.5, '0', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(400, Y2(-1.6) + 3.5, '−1.6', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(400, Y2(-3.0) + 3.5, '−3.0', { mono: true, size: 9.5, anchor: 'end' }));
  body.push(cap(480, 256, '梯度=0', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(620, 256, '◄—— 有梯度 ——►', { size: 9.5, anchor: 'middle' }));

  body.push(cap(24, 288, 'A > 0 时 min 截上（r > 1+ε 出局）、A < 0 时 max 截下（r < 1−ε 出局）—— 两图的梯度区互为镜像；界内走多远由斜率 ±A 决定', { size: 10.5 }));
  body.push(cap(24, 304, '回退方向永不截断（撤销提高 / 撤销压低始终有梯度）：clip 只截单向 —— 与 KL 惩罚对双向偏离对称收费的本质区别', { size: 10.5 }));

  return scene(720, 316, body, { label: 'PPO clip 目标：A>0 时超过 1+ε 压平，A<0 时低于 1−ε 压平，平台段梯度为零' }).svg();
}

/* ============================================================
 * 8.3 §5（block 1，中）— PPO 一轮循环
 * rollout（π_old 冻结）→ GAE 冻结量 → K epoch minibatch 更新 → θ_old ← θ 回环
 * ============================================================ */
function ppoLoop() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'PPO 一轮：rollout 与 GAE 的产物全程冻结，更新阶段跑 K 个 epoch 的 minibatch，末了 θ_old ← θ 回环', { size: 12, color: 'var(--ink-soft)' }));

  const roll = add(node({ id: 'roll', x: 30, y: 56, w: 200, h: 50, role: 'data', title: '① rollout（π_old 冻结）', titleSize: 12, sub: 'N_env 并行 × T 步', subSize: 9.5 }));
  body.push(cap(40, 126, '存 (s, a, r, log π_old, V_old, done)', { mono: true, size: 9.5 }));
  const gae = add(node({ id: 'gae', x: 290, y: 56, w: 170, h: 50, role: 'op', title: '② GAE（8.2）', titleSize: 12, sub: 'δ 逆序递推 Â_t', subSize: 9.5 }));
  body.push(cap(290, 126, 'V_targ = Â + V_old', { mono: true, size: 9.5 }));
  body.push(cap(290, 141, 'Â / log π_old / V_old 全程冻结', { size: 9.5, color: 'var(--seal-deep)' }));
  const upd = add(node({ id: 'upd', x: 510, y: 56, w: 180, h: 50, role: 'attn', title: '③ 更新：K epoch', titleSize: 12, sub: '打乱切 minibatch → Adam', subSize: 9.5 }));
  body.push(cap(520, 126, 'L_CLIP + L_VF + 熵（式 10）', { mono: true, size: 9.5 }));
  body.push(cap(520, 141, '诊断 approx_kl / clipfrac', { size: 9.5 }));

  body.push(edge(roll, gae, { nodes, pulse: true, dur: 2.0, delay: 0, uid: 'p8d', eid: 0 }));
  body.push(edge(gae, upd, { nodes, pulse: true, dur: 2.0, delay: 0.3, uid: 'p8d', eid: 1 }));
  body.push(cap(230, 48, '冻结量入更新', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 回环：θ_old ← θ
  const back = 'M600 106 V204 H130 V108';
  body.push(`<path class="dia-edge strong" d="${back}"/>`);
  body.push(`<polygon points="130,100 125,110 135,110" fill="var(--ink-soft)"/>`);
  body.push(pathPulse(back, { dur: 3.2, delay: 0.8, r: 4 }));
  body.push(cap(365, 222, 'θ_old ← θ（下一轮 rollout 用新权重重采）', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--ink)' }));

  body.push(cap(24, 258, '冻结纪律：GAE 的输入（V_old、log π_old）与 Â 在 K 个 epoch 全程不变；minibatch 内 Â 做归一化', { size: 10.5 }));
  body.push(cap(24, 275, 'approx_kl > target_kl ⇒ 提前终止本轮 epoch（可选护栏）；陈旧偏差与 off-policy 校正归 §4', { size: 10.5 }));

  return scene(720, 288, body, { label: 'PPO 循环：冻结旧策略采样，GAE 算冻结优势，K 个 epoch minibatch 更新后回环' }).svg();
}

/* ============================================================
 * 8.4 §1（block 0，高）— PPO-RLHF 四组件 vs GRPO 三组件
 * critic / GAE 整组删除；组内 G 条奖励 → (r_i − mean)/std 基线；KL 从进 reward 改为进 loss（k3）
 * ============================================================ */
function grpoVsPpo() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, '组件结构对照：GRPO 把 baseline 从「学出来的 V_ψ」换成「同 prompt 组内均值」—— critic 与 GAE 整组删除', { size: 12, color: 'var(--ink-soft)' }));

  /* ---- 左：PPO-RLHF 四组件 ---- */
  body.push(zone({ x: 16, y: 44, w: 336, h: 252, label: 'PPO-RLHF · 四组件' }));
  const pOld = add(node({ id: 'pOld', x: 30, y: 72, w: 104, h: 46, role: 'data', title: 'π_old', mono: true, titleSize: 12, sub: 'actor', subSize: 10 }));
  const pCri = add(node({ id: 'pCri', x: 30, y: 152, w: 104, h: 46, role: 'proj', title: 'critic V_ψ', mono: true, titleSize: 11.5 }));
  const pRef = add(node({ id: 'pRef', x: 30, y: 232, w: 104, h: 46, role: 'cache', title: 'π_ref', mono: true, titleSize: 12, sub: '冻结', subSize: 10 }));
  const pRm = add(node({ id: 'pRm', x: 202, y: 72, w: 136, h: 46, role: 'attn', title: 'reward model', titleSize: 12 }));
  const pGae = add(node({ id: 'pGae', x: 202, y: 152, w: 136, h: 46, role: 'op', title: 'GAE → Â_t', mono: true, titleSize: 11.5 }));

  body.push(edge(pOld, pRm, { nodes, pulse: true, dur: 2.0, delay: 0, uid: 'p8e', eid: 0 }));
  body.push(cap(168, 64, 'rollout', { size: 9.5, anchor: 'middle' }));
  body.push(edge(pCri, pGae, { nodes, pulse: true, dur: 2.0, delay: 0.3, uid: 'p8e', eid: 1 }));
  body.push(cap(168, 144, 'V(s_t)', { mono: true, size: 9.5, anchor: 'middle' }));
  body.push(edge(pRm, pGae, { nodes, portFrom: pRm.bottom, portTo: pGae.top, pulse: true, dur: 1.8, delay: 0.5, uid: 'p8e', eid: 2 }));
  body.push(cap(30, 214, 'KL 逐 token 折进 reward（式 3）', { size: 9.5 }));
  body.push(cap(270, 218, 'Â_t → L_CLIP（8.3）', { mono: true, size: 9.5, anchor: 'middle' }));

  /* ---- 右：GRPO 三组件 ---- */
  body.push(zone({ x: 376, y: 44, w: 328, h: 252, label: 'GRPO · 三组件' }));
  const gOld = add(node({ id: 'gOld', x: 390, y: 72, w: 104, h: 46, role: 'data', title: 'π_old', mono: true, titleSize: 12, sub: 'actor', subSize: 10 }));
  const gRef = add(node({ id: 'gRef', x: 390, y: 152, w: 104, h: 46, role: 'cache', title: 'π_ref', mono: true, titleSize: 12, sub: '冻结', subSize: 10 }));
  const gRm = add(node({ id: 'gRm', x: 566, y: 72, w: 128, h: 46, role: 'attn', title: 'reward / 规则', titleSize: 12, sub: '验证器（RLVR）', subSize: 9.5 }));
  const gBase = add(node({ id: 'gBase', x: 546, y: 156, w: 148, h: 56, role: 'op', title: 'Â_i = (r_i − mean)/std', mono: true, titleSize: 10.5, sub: '组内 G 条共享同一值', subSize: 9.5 }));

  body.push(edge(gOld, gRm, { nodes, pulse: true, dur: 2.0, delay: 0, uid: 'p8e', eid: 3 }));
  body.push(cap(530, 64, '同 prompt 采 G 条', { size: 9.5, anchor: 'middle' }));
  // G 条奖励小柱 → 基线
  const rh = [16, 24, 10, 20, 13];
  rh.forEach((h, i) => {
    body.push(`<rect class="dia-bar dia-bar-warm sig-bar" style="--i:${(0.4 + i * 0.15).toFixed(2)};transform-origin:center bottom" x="${576 + i * 15}" y="${146 - h}" width="10" height="${h}" rx="2"/>`);
  });
  body.push(edge([630, 118], gBase, { nodes, portTo: gBase.top, pulse: true, dur: 1.8, delay: 0.5, uid: 'p8e', eid: 4 }));
  body.push(cap(652, 140, 'r_1..r_G', { mono: true, size: 9.5, anchor: 'start' }));
  body.push(edge(gRef, gBase, { nodes, portTo: [546, 186], dash: true, pulse: true, dur: 2.2, delay: 0.8, uid: 'p8e', eid: 5 }));
  body.push(cap(442, 216, 'k3 KL 进 loss（§3）', { size: 9.5, anchor: 'middle' }));

  // critic / GAE 删除的虚位
  body.push(`<rect x="390" y="232" width="150" height="46" rx="12" fill="none" stroke="var(--rule)" stroke-width="1.2" stroke-dasharray="5 4"/>`);
  body.push(cap(465, 259, 'critic + GAE 整组删除', { size: 10, anchor: 'middle', color: 'var(--ink-faint)' }));

  body.push(cap(24, 320, '显存收益是「约一个 actor 的量级」而非百分比：critic 网络、价值损失、GAE 递推一并消失（8.5 式 1 的 16N_c 项）', { size: 10.5 }));
  body.push(cap(24, 337, 'KL 换位置：进 reward 会参与 GAE / 组基线的全部下游；进 loss 与 advantage 解耦、只起锚定作用（式 2 末项）', { size: 10.5 }));
  body.push(cap(24, 354, '动机（DeepSeekMath §4.1.1）：RM 本就训在同题两两比较上 —— 组内相对与 RM 的比较本性对齐，同 prompt 内自校准', { size: 10.5 }));

  return scene(720, 366, body, { label: 'PPO 四组件对 GRPO 三组件：critic 与 GAE 删除，组内均值当基线，KL 从 reward 挪进 loss' }).svg();
}

/* ============================================================
 * 8.5 §1（block 0，中）— 一次 RL iteration 五步数据流
 * ① rollout（推理引擎 θ_old）→ ② 打分（ref logp + RM/规则）→ ③ advantage 聚合 → ④ 多 epoch 更新 → ⑤ 权重同步
 * ============================================================ */
function rlIteration() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, '一次 RL iteration：五步数据流（PPO 全配置；GRPO 砍掉 critic 一支，8.4）', { size: 12, color: 'var(--ink-soft)' }));

  const pb = add(node({ id: 'pb', x: 16, y: 38, w: 124, h: 30, role: 'plain', title: 'prompt batch', mono: true, titleSize: 10.5 }));
  const defs = [
    ['① rollout', '推理引擎 · θ_old', 'data', 'GRPO：每 prompt G 条（8.4）'],
    ['② 打分', 'ref logp + RM / 规则', 'norm', '逐 token KL(ref ‖ θ_old)'],
    ['③ advantage 聚合', 'GAE（8.2）/ 组基线（8.4）', 'op', ''],
    ['④ 多 epoch 更新', 'actor（与 critic）→ θ', 'attn', 'PPO clip / GRPO 目标'],
    ['⑤ 权重同步', 'θ → rollout 引擎', 'proj', '每轮一次（§4 账）'],
  ];
  const ns = defs.map((d, i) =>
    add(node({ id: `st${i}`, x: 16 + i * 142, y: 84, w: 128, h: 64, role: d[2], title: d[0], titleSize: 12, sub: d[1], subSize: 9 }))
  );
  defs.forEach((d, i) => {
    if (d[3]) body.push(cap(16 + i * 142 + 64 - (i === 4 ? 30 : 0), 164, d[3], { size: 9, anchor: 'middle' }));
  });
  body.push(edge(pb, ns[0], { nodes, portFrom: pb.bottom, portTo: ns[0].top, pulse: true, dur: 1.6, delay: 0, uid: 'p8f', eid: 'in' }));
  for (let i = 0; i < 4; i++) body.push(edge(ns[i], ns[i + 1], { nodes, pulse: true, dur: 2.0, delay: 0.2 + i * 0.25, uid: 'p8f', eid: i }));

  // 回环边
  const back = 'M690 148 V212 H80 V150';
  body.push(`<path class="dia-edge strong" d="${back}"/>`);
  body.push(`<polygon points="80,142 75,152 85,152" fill="var(--ink-soft)"/>`);
  body.push(pathPulse(back, { dur: 3.6, delay: 1.2, r: 4 }));
  body.push(cap(385, 230, '下一轮 iteration（θ_old ← θ）', { size: 10, anchor: 'middle', color: 'var(--ink)' }));

  body.push(cap(24, 264, '只有 ④ 是常规训练（并行与显存账归 4.x–6.x）；①⑤ 的第二套引擎与 ②③ 的多模型编排，是 P1–P7 之外的新账', { size: 10.5 }));
  body.push(cap(24, 281, 'OpenRLHF profile（LLaMA2-7B + A100，其 §3.2）：PPO 的样本生成阶段占整体训练时间 80% —— 引擎拆分（§3）的直接动机', { size: 10.5 }));

  return scene(720, 294, body, { label: 'RL iteration 五步：rollout、打分、advantage 聚合、多 epoch 更新、权重同步成环' }).svg();
}

/* ============================================================
 * 8.5 §3（block 1，高）— single-controller 星型编排
 * 编排进程居中；actor/critic 训练组、rollout 引擎组、ref/RM 推理组四向辐射；①–⑤ 编号数据流
 * ============================================================ */
function singleController() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 18, 'single-controller：编排进程居中调度四组模型，数据以 TensorDict / DataProto 形态流转（veRL / HybridFlow）', { size: 12, color: 'var(--ink-soft)' }));

  // 四组卫星
  const actor = add(node({ id: 'actor', x: 24, y: 66, w: 210, h: 64, role: 'attn', title: 'actor 训练组', titleSize: 12.5, sub: 'Megatron / FSDP · TP×PP×DP', subSize: 9.5 }));
  body.push(cap(129, 150, '训练态 16N_a', { mono: true, size: 9.5, anchor: 'middle' }));
  const critic = add(node({ id: 'critic', x: 24, y: 288, w: 210, h: 60, role: 'proj', title: 'critic 训练组', titleSize: 12.5, sub: '同栈 · PPO 才有（16N_c）', subSize: 9.5 }));
  const rollout = add(node({ id: 'rollout', x: 486, y: 66, w: 210, h: 64, role: 'data', title: 'rollout 引擎组', titleSize: 12.5, sub: 'vLLM × n 实例 · 自身 TP_g 分片', subSize: 9.5 }));
  body.push(cap(591, 150, '2N_a + KV cache', { mono: true, size: 9.5, anchor: 'middle' }));
  const refrm = add(node({ id: 'refrm', x: 486, y: 288, w: 210, h: 60, role: 'cache', title: 'ref / RM 推理组', titleSize: 12.5, sub: '2N_a / 2N_r · 前向 only', subSize: 9.5 }));

  // 中央编排
  const ctrl = add(node({ id: 'ctrl', x: 270, y: 172, w: 180, h: 68, role: 'op', title: 'single-controller', mono: true, titleSize: 12, sub: '编排进程 · 数据流顺序与搬运', subSize: 9 }));
  body.push(cap(360, 262, '③ advantage 聚合在编排侧（GAE 8.2 / 组基线 8.4）', { size: 9.5, anchor: 'middle' }));

  // ① 采样：ctrl → rollout（prompt 下发 / 经验回传）
  body.push(edge(ctrl, rollout, { nodes, portFrom: [450, 190], portTo: [516, 130], pulse: true, dur: 2.2, delay: 0, uid: 'p8g', eid: 0 }));
  body.push(cap(522, 160, '① prompt 下发 / 经验回传', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  // ② 打分：ctrl ⇄ ref/RM
  body.push(edge(ctrl, refrm, { nodes, portFrom: [450, 224], portTo: [520, 300], pulse: true, dur: 2.2, delay: 0.3, uid: 'p8g', eid: 1 }));
  body.push(cap(604, 262, '② 打分（RM 或规则验证器）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  // ④ 更新：ctrl → actor / critic
  body.push(edge(ctrl, actor, { nodes, portFrom: [270, 190], portTo: [204, 130], pulse: true, dur: 2.2, delay: 0.6, uid: 'p8g', eid: 2 }));
  body.push(cap(198, 160, '④ 更新 θ', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge(ctrl, critic, { nodes, portFrom: [270, 224], portTo: [200, 302], dash: true, pulse: true, dur: 2.2, delay: 0.75, uid: 'p8g', eid: 3 }));
  body.push(cap(196, 270, '④ 同更（PPO）', { size: 9.5, anchor: 'middle' }));
  // ⑤ 权重同步：actor → rollout（顶部大弧）
  const sync = 'M129 66 V46 H591 V64';
  body.push(`<path class="dia-edge strong" d="${sync}"/>`);
  body.push(`<polygon points="591,72 586,62 596,62" fill="var(--ink-soft)"/>`);
  body.push(pathPulse(sync, { dur: 3.0, delay: 1.0, r: 4 }));
  body.push(cap(360, 40, '⑤ 权重同步：训练侧 θ → rollout 引擎（每轮一次；resharding 归 §4）', { size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));

  body.push(cap(24, 384, '常驻账（式 1）：actor 16N_a + critic 16N_c + ref 2N_a + RM 2N_r + rollout 副本 2N_a + KV —— 同尺寸 ≈ 38N + KV', { size: 10.5 }));
  body.push(cap(24, 400, '省显存两条：训练态上 ZeRO-1（16N → 4N + 12N/d）；GRPO 去 critic、RLVR 换规则验证器 → 20N（省 47%，代价搬进 KV 项）', { size: 10.5 }));

  return scene(720, 414, body, { label: 'single-controller 架构：编排进程居中，actor/critic 训练组、rollout 引擎组、ref/RM 推理组四向辐射，五步数据流编号' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { dim: 'training', slug: '08-rl-math/01-policy-gradient', replacesBlock: 0, title: 'LLM 即 MDP：逐 token 状态链，奖励只在末端', svg: mdpChain },
  { dim: 'training', slug: '08-rl-math/02-actor-critic-and-gae', replacesBlock: 0, title: 'actor-critic 联合 loss：δ → Â → 一次 backward', svg: actorCritic },
  { dim: 'training', slug: '08-rl-math/03-ppo', replacesBlock: 0, title: 'PPO clip：界内线性、界外平台（A = ±2）', svg: ppoClip },
  { dim: 'training', slug: '08-rl-math/03-ppo', replacesBlock: 1, title: 'PPO 一轮：rollout 冻结 → GAE → K epoch 更新', svg: ppoLoop },
  { dim: 'training', slug: '08-rl-math/04-grpo-and-variants', replacesBlock: 0, title: 'PPO 四组件 vs GRPO 三组件：critic/GAE 消失', svg: grpoVsPpo },
  { dim: 'training', slug: '08-rl-math/05-llm-rl-systems', replacesBlock: 0, title: '一次 RL iteration 的五步数据流', svg: rlIteration },
  { dim: 'training', slug: '08-rl-math/05-llm-rl-systems', replacesBlock: 1, title: 'single-controller：四组模型的星型编排', svg: singleController },
];
