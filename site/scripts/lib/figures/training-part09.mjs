// training Part 9（09-post-training）场景图：多轮 mask / RLHF 三段链 / DPO 推导链 / RFT 循环 / R1 配方树
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
 * 9.1 §3（block 1，中）— 多轮样本 mask 两策略
 * [sys u1 a1 u2 a2 u3 a3]：全轮监督 0 0 1 0 1 0 1 vs 末轮监督 0 0 0 0 0 0 1
 * ============================================================ */
function sftMask() {
  const body = [];
  body.push(cap(24, 24, '多轮样本的 m_s 两种取法：prompt 段恒为 0，forward 成本相同，差别全在 loss 覆盖', { size: 12, color: 'var(--ink-soft)' }));

  const segs = ['[sys]', '[u1]', '[a1]', '[u2]', '[a2]', '[u3]', '[a3]'];
  const isA = (i) => i === 2 || i === 4 || i === 6;
  const X0 = 134, CW = 72, GAP = 8;
  body.push(cap(122, 72, 'token 段', { size: 10, anchor: 'end' }));
  segs.forEach((s, i) => {
    const x = X0 + i * (CW + GAP);
    const hot = isA(i);
    body.push(
      `<rect x="${x}" y="52" width="${CW}" height="30" rx="7" fill="${hot ? 'var(--seal-wash)' : 'var(--paper-sunk)'}" stroke="${hot ? 'var(--seal)' : 'var(--rule)'}" stroke-width="1.1"/>` +
      `<text class="dia-mono" x="${x + CW / 2}" y="71" font-size="11" fill="var(--ink)" text-anchor="middle">${esc(s)}</text>`
    );
  });

  const rows = [
    { label: '全轮监督 m_s', y: 112, mask: [0, 0, 1, 0, 1, 0, 1] },
    { label: '末轮监督 m_s', y: 168, mask: [0, 0, 0, 0, 0, 0, 1] },
  ];
  rows.forEach((row, ri) => {
    body.push(cap(122, row.y + 19, row.label, { mono: true, size: 10, anchor: 'end' }));
    row.mask.forEach((v, i) => {
      const x = X0 + i * (CW + GAP);
      if (v)
        body.push(
          `<rect class="dia-mx-hot sig-cell" style="--i:${(0.3 + ri * 0.9 + i * 0.14).toFixed(2)}" x="${x}" y="${row.y}" width="${CW}" height="28" rx="5"/>` +
          `<text class="dia-mono sig-cell-text" style="--i:${(0.3 + ri * 0.9 + i * 0.14).toFixed(2)}" x="${x + CW / 2}" y="${row.y + 19}" font-size="11" fill="var(--seal-deep)" text-anchor="middle" font-weight="700">1</text>`
        );
      else
        body.push(
          `<rect class="dia-mx-off" x="${x}" y="${row.y}" width="${CW}" height="28" rx="5"/>` +
          `<text class="dia-mono" x="${x + CW / 2}" y="${row.y + 19}" font-size="10.5" fill="var(--ink-faint)" text-anchor="middle">0</text>`
        );
    });
  });

  body.push(cap(24, 230, '全轮监督：a1 / a2 / a3 三处揭开 —— η 高，一条序列喂 3 轮监督（TRL assistant_only_loss 同型；训练分布与多轮推理同构）', { size: 10.5 }));
  body.push(cap(24, 247, '末轮监督：只揭 a3 —— η 低，监督面收窄到单一决策点（Llama-3 的 context + final response 形态，herd Table 7 表注）', { size: 10.5 }));
  body.push(cap(24, 264, 'EOS 必须被 m_s = 1 覆盖，否则模型从未学过「说完就停」（§2 纪律一）；对 u_k 计 loss = 训练模型续写用户提问，方向与部署相反', { size: 10.5 }));

  return scene(720, 278, body, { label: '多轮 SFT 的两种 mask 策略：全轮监督揭开全部 assistant 段，末轮监督只揭最后一段' }).svg();
}

/* ============================================================
 * 9.2 §1（block 0，高）— RLHF 三阶段全景
 * SFT(13k) → RM(33k，K=4~9 排序 → C(K,2) 对) → PPO(31k，无标签 rollout)；模型角色跨段继承
 * ============================================================ */
function rlhfPipeline() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 22, 'RLHF 三段链（InstructGPT Figure 2 数据流）：监督形态 示范 → 排序 → 无标注；模型角色跨段继承', { size: 12, color: 'var(--ink-soft)' }));

  // 三个阶段的列
  body.push(zone({ x: 14, y: 40, w: 222, h: 252, label: '阶段一 SFT（offline）' }));
  body.push(zone({ x: 250, y: 40, w: 222, h: 252, label: '阶段二 RM（offline）' }));
  body.push(zone({ x: 486, y: 40, w: 222, h: 252, label: '阶段三 PPO（on-policy）' }));

  body.push(cap(30, 78, '13k prompts', { size: 11, weight: 600, color: 'var(--ink)' }));
  body.push(cap(30, 96, '标注员撰写 + API', { size: 10 }));
  body.push(cap(30, 112, '标注员写示范回答', { size: 10 }));
  const sft = add(node({ id: 'sft', x: 30, y: 196, w: 190, h: 52, role: 'data', title: 'π_SFT', mono: true, titleSize: 12.5, sub: '兼作阶段三初始化与 π_ref', subSize: 9.5 }));

  body.push(cap(266, 78, '33k prompts', { size: 11, weight: 600, color: 'var(--ink)' }));
  body.push(cap(266, 96, '每 prompt K=4~9 条采样回答', { size: 10 }));
  body.push(cap(266, 112, '标注员排序 → C(K,2) 偏好对', { size: 10 }));
  const rm = add(node({ id: 'rm', x: 266, y: 196, w: 190, h: 52, role: 'proj', title: 'RM r_θ · 6B', mono: true, titleSize: 12, sub: '去 unembedding 换 scalar head', subSize: 9 }));

  body.push(cap(502, 78, '31k prompts', { size: 11, weight: 600, color: 'var(--ink)' }));
  body.push(cap(502, 96, '只含 prompt · 无人工标签', { size: 10 }));
  body.push(cap(502, 112, 'rollout：π_θ 对 prompt 采样', { size: 10 }));
  body.push(cap(502, 136, '打分 r_θ(x,y) − β·log(π_θ/π_ref)', { mono: true, size: 9 }));
  body.push(cap(502, 152, '更新：PPO clip + γ·预训练项', { size: 9.5 }));
  const actor = add(node({ id: 'actor', x: 502, y: 164, w: 190, h: 40, role: 'attn', title: 'actor π_θ', mono: true, titleSize: 11.5, sub: 'init = π_SFT', subSize: 9 }));
  const critic = add(node({ id: 'critic', x: 502, y: 216, w: 190, h: 40, role: 'proj', title: 'critic · 6B', mono: true, titleSize: 11.5, sub: 'init = RM', subSize: 9 }));

  // 数据流：π_SFT 采样回答进 RM 排序；RM 进 PPO 打分
  body.push(edge(sft, rm, { nodes, pulse: true, dur: 2.0, delay: 0, uid: 'p9b', eid: 0 }));
  body.push(cap(243, 216, '采样回答', { size: 9, anchor: 'middle' }));
  body.push(edge(rm, actor, { nodes, portFrom: rm.right, portTo: [502, 184], pulse: true, dur: 2.0, delay: 0.4, uid: 'p9b', eid: 1 }));
  body.push(cap(479, 216, '打分', { size: 9, anchor: 'middle' }));

  // 角色继承线（朱砂虚线）：π_SFT 绕底部 lane 从右侧进 actor；RM 从下方进 critic
  const inh1 = 'M125 248 V318 H700 V184 H694';
  const inh2 = 'M361 248 V282 H650 V258';
  body.push(`<path class="dia-edge dash" d="${inh1}" style="stroke:rgb(168 50 40 / 0.55)"/>`);
  body.push(`<polygon points="694,184 686,180 686,188" fill="var(--seal)"/>`);
  body.push(pathPulse(inh1, { dur: 3.4, delay: 0.7, r: 3.8 }));
  body.push(`<path class="dia-edge dash" d="${inh2}" style="stroke:rgb(168 50 40 / 0.55)"/>`);
  body.push(`<polygon points="650,255 644,263 656,263" fill="var(--seal)"/>`);
  body.push(pathPulse(inh2, { dur: 3.0, delay: 1.1, r: 3.8 }));
  body.push(cap(360, 312, '参数继承：actor ← π_SFT · π_ref = π_SFT 冻结副本', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(505, 278, 'critic ← RM（初始化）', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(24, 372, '数据量 13k → 33k → 31k（SFT→RM 扩、RM→PPO 微缩）；RM 与 critic 同为 6B，服务 1.3B–175B 全部 policy 档（App. C）', { size: 10.5 }));
  body.push(cap(24, 389, '「打分与估值用小模型、被优化对象随便大」是这条链的原始形态；四模型驻留与引擎拆分的系统账归 8.5', { size: 10.5 }));

  return scene(720, 402, body, { label: 'RLHF 三段链：SFT 教格式，RM 把成对偏好蒸成标量，PPO 以 SFT 为锚在线优化；参数沿底部继承线流转' }).svg();
}

/* ============================================================
 * 9.3 §1（block 0，高）— DPO 推导链五站
 * 式 1 KL 约束目标 → 式 3 闭式最优解 → 式 4 reward 重参数化 → 式 5 BT 相消（β·log Z(x)）→ 式 6 DPO 目标
 * ============================================================ */
function dpoChain() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'DPO 推导链：五步全是严格的代数恒等变形，没有一步近似 —— 走完 RM 与 RL 同时消失', { size: 12, color: 'var(--ink-soft)' }));

  const cards = [
    { role: 'data', t: '① KL 约束 RLHF 目标（式 1）', s: 'max_π E_{x∼D, y∼π}[ r(x,y) ] − β·KL(π ‖ π_ref)' },
    { role: 'proj', t: '② 闭式最优解（式 3）', s: 'π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)' },
    { role: 'proj', t: '③ reward 重参数化（式 4）', s: 'r(x,y) = β·log( π*(y|x)/π_ref(y|x) ) + β·log Z(x)' },
    { role: 'op', t: '④ 偏好概率只含策略（式 5）', s: '代入 Bradley-Terry：β·log Z(x) 逐 prompt 相消' },
    { role: 'attn', t: '⑤ DPO 目标（式 6）', s: 'L = −E_D log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) )' },
  ];
  const notes = ['§2 逐 prompt 变分（Gibbs 取等）', '§3 两边取 log 解出 r', '代入 Bradley-Terry（式 5）', '对 π_θ 极大似然（监督式分类损失）'];
  const ys = [46, 118, 190, 262, 334];
  const cs = cards.map((c, i) =>
    add(node({ id: `c${i}`, x: 30, y: ys[i], w: 470, h: 54, role: c.role, title: c.t, titleSize: 12, sub: c.s, mono: false, subSize: 9.5 }))
  );
  // 副行改 mono：node() 的 sub 跟随 mono 标志整卡切换，这里要 title 用 sans、sub 用 mono —— 直接补一层 mono 文本盖掉
  // （node 的 sub 已是 dia-sub sans；公式行用 mono 重绘更贴近账本感，简单起见保留 sans 小字）
  for (let i = 0; i < 4; i++) {
    body.push(edge(cs[i], cs[i + 1], { nodes, portFrom: [265, ys[i] + 54], portTo: [265, ys[i + 1]], pulse: true, dur: 1.6, delay: i * 0.3, uid: 'p9c', eid: i }));
    body.push(cap(516, ys[i] + 68, notes[i], { size: 9.5 }));
  }

  // 相消高亮：β·log Z(x) 划去（sig-diag 脉冲闪烁）
  body.push(
    `<g class="sig-diag" style="--i:0.8">` +
    `<text class="dia-mono" x="620" y="290" font-size="11.5" fill="var(--seal-deep)" text-anchor="middle" font-weight="600">β·log Z(x)</text>` +
    `<line x1="578" y1="287" x2="662" y2="287" stroke="var(--seal)" stroke-width="1.8" stroke-linecap="round"/>` +
    `</g>`
  );
  body.push(cap(620, 306, '逐 prompt 相消', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(24, 406, '③ 的 β·log Z(x) 只含 x：在 Bradley-Terry 的成对差里自行消去 —— 3.2 §3 平移不可识别伏笔的兑现', { size: 10.5 }));
  body.push(cap(24, 423, '链条走完：偏好数据直接约束策略，训练循环退化为带冻结 π_ref 的监督式分类损失（无 RM 训练、无 rollout 引擎）', { size: 10.5 }));

  return scene(720, 436, body, { label: 'DPO 推导链：KL 约束目标经闭式解、reward 重参数化、Bradley-Terry 相消，直达 DPO 损失' }).svg();
}

/* ============================================================
 * 9.4 §1（block 0，中）— RFT 一轮循环
 * 采 n 条 → 打分（规则/RM）→ top-k 筛选 D⁺ → SFT 回灌；数据采后冻结；推理栈/训练栈分界
 * ============================================================ */
function rftLoop() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, 'RFT 一轮：特异性全在数据生产侧 —— 训练阶段消费的是冻结数据集 D⁺，与 9.1 的 SFT 无形式差别', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 16, y: 44, w: 404, h: 196, label: '推理栈 · 采样后冻结' }));
  body.push(zone({ x: 436, y: 44, w: 268, h: 196, label: '训练栈' }));

  const pi0 = add(node({ id: 'pi0', x: 30, y: 112, w: 96, h: 48, role: 'data', title: 'π_θ(t)', mono: true, titleSize: 12 }));
  const samp = add(node({ id: 'samp', x: 154, y: 62, w: 122, h: 46, role: 'op', title: '采 n 条', titleSize: 12, sub: 'y_1..y_n ~ π_θ', mono: false, subSize: 9.5 }));
  const score = add(node({ id: 'score', x: 154, y: 158, w: 122, h: 46, role: 'norm', title: '打分', titleSize: 12, sub: 's_i = V(x,y_i) · 规则/RM', subSize: 9 }));
  const pick = add(node({ id: 'pick', x: 306, y: 110, w: 100, h: 52, role: 'attn', title: 'top-k 筛选', titleSize: 11.5, sub: '→ D⁺', subSize: 10 }));
  const sft = add(node({ id: 'sft', x: 466, y: 110, w: 96, h: 52, role: 'proj', title: 'SFT 回灌', titleSize: 12, sub: 'CE（9.1）', subSize: 9.5 }));
  const pi1 = add(node({ id: 'pi1', x: 592, y: 110, w: 100, h: 52, role: 'data', title: 'π_θ(t+1)', mono: true, titleSize: 11.5 }));

  // 漏斗：n 条进、k 条出
  body.push(edge(pi0, samp, { nodes, portFrom: pi0.right, portTo: samp.left, pulse: true, dur: 1.8, delay: 0, uid: 'p9d', eid: 0 }));
  body.push(edge(samp, pick, { nodes, portFrom: samp.right, portTo: [306, 122], pulse: true, dur: 1.8, delay: 0.25, uid: 'p9d', eid: 1 }));
  body.push(edge(pi0, score, { nodes, portFrom: pi0.right, portTo: score.left, pulse: true, dur: 1.8, delay: 0.15, uid: 'p9d', eid: 2 }));
  body.push(edge(score, pick, { nodes, portFrom: score.right, portTo: [306, 148], pulse: true, dur: 1.8, delay: 0.4, uid: 'p9d', eid: 3 }));
  body.push(cap(356, 100, 'n → k', { mono: true, size: 9, anchor: 'middle' }));
  body.push(edge(pick, sft, { nodes, pulse: true, dur: 1.8, delay: 0.55, uid: 'p9d', eid: 4, strong: true }));
  body.push(edge(sft, pi1, { nodes, pulse: true, dur: 1.8, delay: 0.7, uid: 'p9d', eid: 5 }));

  // 回环：下一轮
  const back = 'M642 162 V262 H78 V162';
  body.push(`<path class="dia-edge strong" d="${back}"/>`);
  body.push(`<polygon points="78,154 73,164 83,164" fill="var(--ink-soft)"/>`);
  body.push(pathPulse(back, { dur: 3.2, delay: 1.0, r: 4 }));
  body.push(cap(360, 280, '下一轮：D⁺ 的分布随策略漂移 —— 离线目标 + 滚动数据源（§3 / §7）', { size: 10, anchor: 'middle' }));

  return scene(720, 292, body, { label: 'RFT 循环：当前策略采样 n 条，验证器或 RM 打分，top-k 筛出 D⁺，SFT 回灌成下一轮策略' }).svg();
}

/* ============================================================
 * 9.5 §3（block 0，高）— DeepSeek-R1 配方全流程树
 * 主线四阶段（冷启动 SFT → 推理 RL → 拒绝采样+SFT → 全场景 RL）+ R1-Zero 对照支线 + 蒸馏支线
 * ============================================================ */
function r1Recipe() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 22, 'R1 配方：四阶段主线 + 两条支线（论文 §2.3 four stages；R1-Zero 支线兼作对照实验与数据上游）', { size: 12, color: 'var(--ink-soft)' }));

  const base = add(node({ id: 'base', x: 270, y: 36, w: 180, h: 44, role: 'data', title: 'DeepSeek-V3-Base', mono: true, titleSize: 11.5, sub: '基座' }));

  // 支线：R1-Zero（对照实验）
  body.push(zone({ x: 20, y: 108, w: 224, h: 152, label: '支线 · 对照实验', tone: 'seal' }));
  body.push(cap(36, 148, 'R1-Zero：直接 GRPO（规则 reward 式 1）', { size: 10, weight: 600, color: 'var(--ink)' }));
  body.push(cap(36, 168, 'AIME 15.6% → 71.0%（cons@64 86.7%）', { mono: true, size: 9.5 }));
  body.push(cap(36, 186, '长度随训练增长 · 反思 / 换路涌现', { size: 9.5 }));
  body.push(cap(36, 204, '缺陷：可读性差 · 语言混杂', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(36, 222, '→ 主线冷启动数据的来源之一', { size: 9.5, color: 'var(--seal-deep)' }));

  // 主线四阶段
  const st1 = add(node({ id: 'st1', x: 330, y: 102, w: 250, h: 54, role: 'proj', title: '阶段 1 · 冷启动 SFT', titleSize: 12, sub: '数千条长 CoT · 推理+summary 两段式', subSize: 9.5 }));
  body.push(cap(588, 120, '① few-shot 长 CoT 示例', { size: 9 }));
  body.push(cap(588, 133, '② 直接提示生成反思回答', { size: 9 }));
  body.push(cap(588, 146, '③ R1-Zero 输出可读化', { size: 9 }));
  const st2 = add(node({ id: 'st2', x: 330, y: 194, w: 250, h: 54, role: 'attn', title: '阶段 2 · 推理导向 RL', titleSize: 12, sub: 'GRPO · 规则正确性 + 语言一致性（式 2）', subSize: 9 }));
  body.push(cap(588, 214, '任务域：数学/代码/科学/逻辑', { size: 9 }));
  const st3 = add(node({ id: 'st3', x: 330, y: 286, w: 250, h: 54, role: 'op', title: '阶段 3 · 拒绝采样 + SFT', titleSize: 12, sub: '回 V3-Base 重训 · 600k+200k ≈ 800k · 2 epoch', subSize: 9 }));
  body.push(cap(588, 306, '滤除混杂语/长段落/代码块', { size: 9 }));
  body.push(cap(588, 320, '非推理 200k 复用 V3 管线', { size: 9 }));
  const st4 = add(node({ id: 'st4', x: 330, y: 378, w: 250, h: 54, role: 'attn', title: '阶段 4 · 全场景 RL', titleSize: 12, sub: '推理：规则 reward · 通用：RM（V3 偏好管线）', subSize: 9 }));
  body.push(cap(588, 398, 'helpfulness 只评 summary', { size: 9 }));
  body.push(cap(588, 412, 'harmlessness 评整段', { size: 9 }));
  const r1 = add(node({ id: 'r1', x: 330, y: 470, w: 250, h: 48, role: 'attn', title: 'DeepSeek-R1', titleSize: 12.5, sub: 'AIME 79.8% · MATH-500 97.3%', mono: false, subSize: 9.5 }));
  const dist = add(node({ id: 'dist', x: 606, y: 452, w: 100, h: 48, role: 'cache', title: '蒸馏支线', titleSize: 11.5, sub: '800k · 无 RL（§7）', subSize: 8.5 }));

  // checkpoint 徽章（阶段 2→3 是数据发生器）
  const ckpt = (cy, label, i) =>
    `<g class="sig-cell" style="--i:${i}"><circle cx="455" cy="${cy}" r="10" fill="var(--paper)" stroke="var(--seal)" stroke-width="1.4"/>` +
    `<text class="dia-mono" x="455" y="${cy + 3}" font-size="7.5" fill="var(--seal-deep)" text-anchor="middle" font-weight="700">ckpt</text></g>` +
    cap(475, cy + 4, label, { size: 9, color: 'var(--seal-deep)' });
  body.push(ckpt(270, '数据发生器：采 600k 推理数据', 0.9));
  body.push(ckpt(362, '', 1.2));

  // 边：基座 → 支线 / 主线；支线 → 阶段 1（虚线朱砂）；主线竖向
  body.push(edge(base, st1, { nodes, portFrom: [410, 80], portTo: st1.top, pulse: true, dur: 2.0, delay: 0, uid: 'p9e', eid: 0 }));
  body.push(edge(base, [132, 108], { nodes, portFrom: [300, 80], pulse: true, dur: 2.0, delay: 0.15, uid: 'p9e', eid: 1 }));
  const cross = 'M244 168 C280 168, 290 138, 328 131';
  body.push(`<path class="dia-edge dash" d="${cross}" style="stroke:rgb(168 50 40 / 0.55)"/>`);
  body.push(pathPulse(cross, { dur: 2.4, delay: 0.6, r: 3.4 }));
  body.push(cap(287, 184, '输出整理', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(edge(st1, st2, { nodes, portFrom: st1.bottom, portTo: st2.top, pulse: true, dur: 2.0, delay: 0.3, uid: 'p9e', eid: 2 }));
  body.push(edge(st2, st3, { nodes, portFrom: st2.bottom, portTo: st3.top, pulse: true, dur: 2.0, delay: 0.5, uid: 'p9e', eid: 3 }));
  body.push(edge(st3, st4, { nodes, portFrom: st3.bottom, portTo: st4.top, pulse: true, dur: 2.0, delay: 0.7, uid: 'p9e', eid: 4 }));
  body.push(edge(st4, r1, { nodes, portFrom: st4.bottom, portTo: r1.top, pulse: true, dur: 2.0, delay: 0.9, uid: 'p9e', eid: 5, strong: true }));
  body.push(edge(r1, dist, { nodes, pulse: true, dur: 2.0, delay: 1.2, uid: 'p9e', eid: 6, dash: true }));
  body.push(cap(593, 442, '800k 数据复用', { size: 9, anchor: 'middle' }));

  body.push(cap(24, 540, 'R1-Zero 证明「纯 RL 可涌现推理」；R1 主线回答「产物可用」：冷启动压住 RL 早期不稳 + 可读性先验，两轮 RL 之间用拒绝采样把能力转写成数据', { size: 10.5 }));

  return scene(720, 552, body, { label: 'DeepSeek-R1 配方：V3-Base 出发，冷启动 SFT、推理 RL、拒绝采样 SFT、全场景 RL 四阶段，R1-Zero 与蒸馏两条支线' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { dim: 'training', slug: '09-post-training/01-sft', replacesBlock: 1, title: '多轮 mask：全轮监督 vs 末轮监督', svg: sftMask },
  { dim: 'training', slug: '09-post-training/02-rlhf-pipeline', replacesBlock: 0, title: 'RLHF 三段链：13k → 33k → 31k 与角色继承', svg: rlhfPipeline },
  { dim: 'training', slug: '09-post-training/03-dpo-family', replacesBlock: 0, title: 'DPO 推导链：五步恒等变形，RM 与 RL 同时消失', svg: dpoChain },
  { dim: 'training', slug: '09-post-training/04-rejection-sampling-and-rft', replacesBlock: 0, title: 'RFT 一轮：采样 → 打分 → 筛选 → SFT 回灌', svg: rftLoop },
  { dim: 'training', slug: '09-post-training/05-rlvr-and-reasoning', replacesBlock: 0, title: 'R1 配方：四阶段主线 + 两条支线', svg: r1Recipe },
];
