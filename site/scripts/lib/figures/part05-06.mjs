// Part 5–6 签名动图：案例研究与视觉表征
// 规范见 STYLE.md；数字全部来自对应章节原文（config 实测 / 论文口径，照章内标注）。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='13'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

const svgOpen = (w, h, label) =>
  `<svg class="fig-svg sig" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;

/* ============ 5.1 §1 — Llama 谱系 config 演进账本 ============ */
/* 旗舰行逐代字段 diff：热格 = 相对上一代变化的字段（章内总表数字） */
function llamaLedger() {
  const w = 720, h = 322;
  const p = [svgOpen(w, h, 'Llama 一到四代旗舰 config 字段的逐代 diff 账本')];

  const fields = ['layers', 'heads Q/KV', 'vocab', 'ctx', 'rope base', 'FFN'];
  const fx = (i) => 178 + i * 90;
  fields.forEach((f, i) => {
    p.push(`<text ${MONO_S} x='${fx(i) + 84 / 2}' y='46' fill='var(--ink-faint)' text-anchor='middle'>${f}</text>`);
  });

  const rows = [
    { name: 'Llama-1 65B', note: '基准系', cells: ['80', '64/64', '32000', '2048', '1e4', 'dense'], hot: [] },
    { name: 'Llama-2 70B', note: '改 2 处', cells: ['80', '64/8', '32000', '4096', '1e4', 'dense'], hot: [1, 3] },
    { name: 'Llama-3.1 70B', note: '改 3 处', cells: ['80', '64/8', '128256', '131072', '5e5', 'dense'], hot: [2, 3, 4] },
    { name: 'Llama-4 Scout', note: '改 5 处', cells: ['48', '40/8', '202048', '10485760', '5e5·f16', '16+1 top1'], hot: [0, 1, 2, 3, 5] },
  ];
  const y0 = 58, rh = 40, gap = 6;
  rows.forEach((r, ri) => {
    const y = y0 + ri * (rh + gap);
    p.push(`<text ${MONO} x='20' y='${y + 17}' fill='var(--ink)'>${r.name}</text>`);
    p.push(`<text ${CJK_S} x='20' y='${y + 33}' fill='var(--ink-faint)'>${r.note}</text>`);
    r.cells.forEach((c, ci) => {
      const x = fx(ci);
      const isHot = r.hot.includes(ci);
      if (isHot) {
        p.push(
          `<rect class='sig-cell' style='--i:${(ri * 0.5 + ci * 0.22).toFixed(2)}' x='${x}' y='${y}' width='${84}' height='${rh}' rx='3' ` +
          `fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>` +
          `<text class='sig-cell-text' style='--i:${(ri * 0.5 + ci * 0.22).toFixed(2)}' ${MONO_S} x='${x + 84 / 2}' y='${y + 24}' fill='var(--seal-deep)' text-anchor='middle'>${c}</text>`
        );
      } else {
        p.push(
          `<rect x='${x}' y='${y}' width='${84}' height='${rh}' rx='3' fill='var(--paper-raised)' stroke='var(--rule)'/>` +
          `<text ${MONO_S} x='${x + 84 / 2}' y='${y + 24}' fill='var(--ink-soft)' text-anchor='middle'>${c}</text>`
        );
      }
    });
  });

  // 底部：不动的四件 + head_dim 恒定
  const sy = y0 + 4 * (rh + gap) + 8;
  p.push(`<rect x='20' y='${sy}' width='680' height='34' rx='3' fill='var(--paper-sunk)' stroke='var(--rule-soft)'/>`);
  p.push(`<text ${CJK_S} x='36' y='${sy + 22}' fill='var(--ink-soft)'>四代不动：RMSNorm + Pre-LN · SwiGLU · 无 bias · untied embedding ｜ head_dim 恒 128</text>`);
  p.push(`<text ${CJK_S} x='20' y='${sy + 56}' fill='var(--ink-faint)'>朱砂格 = 相对上一代变化的字段（RMSNorm 骨架不变，动的全在组件层）；5e5·f16 = base 不变 + llama3 式 scaling factor 16</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 5.2 §3.1 — aux-loss-free 偏置控制回路 ============ */
function auxLossFree() {
  const w = 720, h = 318;
  const p = [svgOpen(w, h, 'DeepSeek-V3 路由：sigmoid 亲和 + 偏置只进选择的控制回路')];
  p.push(`<text ${CJK_S} x='24' y='26' fill='var(--ink-soft)'>均衡不进优化目标：b_i 只加在选择分上、按负载误差走符号步——loss 里没有均衡项（序列级兜底损失系数极小，只兜底）</text>`);

  // 前向通路（y≈92）
  const y = 92;
  p.push(`<rect x='24' y='${y - 20}' width='64' height='40' rx='3' fill='#f1efe8' stroke='#c6c0af'/>`);
  p.push(`<text ${MONO} x='56' y='${y + 4}' fill='var(--ink)' text-anchor='middle'>x_t</text>`);

  p.push(`<rect x='118' y='${y - 26}' width='128' height='52' rx='3' fill='#e9edf3' stroke='#aebcd2'/>`);
  p.push(`<text ${MONO} x='182' y='${y - 2}' fill='var(--ink)' text-anchor='middle'>s_i = σ(x·e_i)</text>`);
  p.push(`<text ${MONO_S} x='182' y='${y + 16}' fill='var(--ink-faint)' text-anchor='middle'>i = 1..256</text>`);

  // 加性偏置节点
  p.push(`<circle cx='292' cy='${y}' r='13' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.3'/>`);
  p.push(`<text ${MONO} x='292' y='${y + 5}' fill='var(--ink)' text-anchor='middle'>+</text>`);
  p.push(`<text ${MONO_S} x='292' y='${y - 24}' fill='var(--seal)' text-anchor='middle'>b_i（只进选择）</text>`);

  p.push(`<rect x='330' y='${y - 26}' width='118' height='52' rx='3' fill='#eeeef2' stroke='#b6b6c4'/>`);
  p.push(`<text ${MONO} x='389' y='${y - 2}' fill='var(--ink)' text-anchor='middle'>top-8 选择</text>`);
  p.push(`<text ${MONO_S} x='389' y='${y + 16}' fill='var(--ink-faint)' text-anchor='middle'>按 s_i + b_i 排名</text>`);

  p.push(`<rect x='492' y='${y - 34}' width='196' height='68' rx='3' fill='#f1efe8' stroke='#c6c0af'/>`);
  p.push(`<text ${MONO} x='590' y='${y - 8}' fill='var(--ink)' text-anchor='middle'>256 routed + 1 shared</text>`);
  p.push(`<text ${MONO_S} x='590' y='${y + 10}' fill='var(--ink-faint)' text-anchor='middle'>y = Σ g_i·E_i(x) + E_s(x)</text>`);
  p.push(`<text ${MONO_S} x='590' y='${y + 28}' fill='var(--ink-faint)' text-anchor='middle'>每 token 9 个 expert</text>`);

  // 前向连线 + 流光
  const fwd = [[88, 118], [246, 279], [305, 330], [448, 492]];
  fwd.forEach(([x1, x2]) => {
    p.push(`<line x1='${x1}' y1='${y}' x2='${x2}' y2='${y}' stroke='#a9a498' stroke-width='1.2'/>`);
  });
  p.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='2.4s' repeatCount='indefinite' path='M 88 ${y} L 570 ${y}'/></circle>`
  );

  // 加权支路：g_i 用 s_i 不用 b_i（从 sigmoid 芯片下方绕过偏置）
  p.push(`<path d='M 182 ${y + 26} C 182 152, 300 152, 488 134' fill='none' stroke='var(--ink-faint)' stroke-width='1' stroke-dasharray='3 4'/>`);
  p.push(`<path d='M 480 130 L 492 134 L 481 139 Z' fill='var(--ink-faint)'/>`);
  p.push(`<text ${MONO_S} x='300' y='160' fill='var(--ink-faint)'>加权用 g_i：选中集内归一 × 2.5——用 s_i，不用 b_i</text>`);

  // 控制回路（下方）
  const ly = 236;
  p.push(`<rect x='330' y='${ly - 20}' width='118' height='40' rx='3' fill='#f3ecdc' stroke='#cbb98d'/>`);
  p.push(`<text ${MONO_S} x='389' y='${ly - 2}' fill='var(--ink)' text-anchor='middle'>负载计数 f_i</text>`);
  p.push(`<text ${MONO_S} x='389' y='${ly + 14}' fill='var(--ink-faint)' text-anchor='middle'>本批每 expert 命中数</text>`);

  p.push(`<rect x='492' y='${ly - 20}' width='96' height='40' rx='3' fill='#eeeef2' stroke='#b6b6c4'/>`);
  p.push(`<text ${MONO_S} x='540' y='${ly - 2}' fill='var(--ink)' text-anchor='middle'>err = f_i − f̄</text>`);
  p.push(`<text ${MONO_S} x='540' y='${ly + 14}' fill='var(--ink-faint)' text-anchor='middle'>对均匀目标</text>`);

  p.push(`<rect x='118' y='${ly - 20}' width='168' height='40' rx='3' fill='#eeeef2' stroke='#b6b6c4'/>`);
  p.push(`<text ${MONO_S} x='202' y='${ly - 2}' fill='var(--ink)' text-anchor='middle'>b_i ← b_i − γ·sign(err)</text>`);
  p.push(`<text ${MONO_S} x='202' y='${ly + 14}' fill='var(--ink-faint)' text-anchor='middle'>γ = 1e-3 · 只进选择不进权重</text>`);

  // 回路连线：top-8 ↓ 负载 → err → 更新 → ↑ 偏置节点（虚线流动 = 控制信号，非梯度）
  p.push(`<line x1='389' y1='${y + 26}' x2='389' y2='${ly - 20}' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<line x1='448' y1='${ly}' x2='492' y2='${ly}' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<path class='sig-dashflow' d='M 540 ${ly - 20} C 540 ${ly - 44}, 202 ${ly - 44}, 202 ${ly - 20}' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<path class='sig-dashflow' d='M 202 ${ly - 20} C 202 ${ly - 60}, 292 ${ly - 60}, 292 ${y + 14}' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<path d='M 287 ${y + 16} L 292 ${y + 5} L 297 ${y + 16} Z' fill='var(--seal)'/>`);

  p.push(`<text ${CJK_S} x='24' y='300' fill='var(--ink-faint)'>虚线回路 = 控制信号（无梯度穿过 b_i）；V2 的三路 aux loss 在 V3 被这条回路整体替换</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 6.1 §1 — patchify：conv(k=p,s=p) ≡ 单线性层 ============ */
function vitPatchify() {
  const w = 720, h = 304;
  const p = [svgOpen(w, h, 'patchify 的 shape 流与 conv-linear 等价：图像到 196 个 token')];

  // 左：图像 224×224×3（以 4×4 可见网格代表 14×14）
  const ix = 32, iy = 52, is = 116, g = is / 4;
  p.push(`<text ${MONO_S} x='${ix}' y='40' fill='var(--ink-faint)'>x [224, 224, 3]</text>`);
  p.push(`<rect x='${ix}' y='${iy}' width='${is}' height='${is}' rx='2' fill='#f1efe8' stroke='#c6c0af'/>`);
  for (let i = 1; i < 4; i++) {
    p.push(`<line x1='${ix + i * g}' y1='${iy}' x2='${ix + i * g}' y2='${iy + is}' stroke='var(--rule)' stroke-width='1'/>`);
    p.push(`<line x1='${ix}' y1='${iy + i * g}' x2='${ix + is}' y2='${iy + i * g}' stroke='var(--rule)' stroke-width='1'/>`);
  }
  // 高亮一个 patch（第 2 行第 1 列）
  p.push(`<rect x='${ix}' y='${iy + g}' width='${g}' height='${g}' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<text ${MONO_S} x='${ix}' y='${iy + is + 16}' fill='var(--ink-faint)'>14×14 网格（p = 16）</text>`);

  // 中：放大 patch → flatten 向量
  const px = 210, py = 62, ps = 54;
  p.push(`<path d='M ${ix + g} ${iy + g} L ${px} ${py} M ${ix + g} ${iy + 2 * g} L ${px} ${py + ps}' fill='none' stroke='var(--rule)' stroke-width='1'/>`);
  p.push(`<rect x='${px}' y='${py}' width='${ps}' height='${ps}' rx='2' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='${px + ps / 2}' y='${py + ps + 14}' fill='var(--ink-faint)' text-anchor='middle'>16×16×3</text>`);

  // flatten 向量（竖条）
  const vx = 310, vy = 52, vh = 96;
  p.push(`<rect x='${vx}' y='${vy}' width='24' height='${vh}' rx='2' fill='#f1efe8' stroke='#c6c0af'/>`);
  for (let i = 1; i < 6; i++) {
    p.push(`<line x1='${vx}' y1='${vy + (i * vh) / 6}' x2='${vx + 24}' y2='${vy + (i * vh) / 6}' stroke='var(--rule)' stroke-width='0.8'/>`);
  }
  p.push(`<text ${MONO_S} x='${vx + 12}' y='${vy + vh + 14}' fill='var(--ink-faint)' text-anchor='middle'>vec ∈ R^768</text>`);
  p.push(`<text ${MONO_S} x='${vx + 12}' y='${vy + vh + 28}' fill='var(--ink-faint)' text-anchor='middle'>(u,v,c) 序</text>`);

  // W 投影 → token
  p.push(`<rect x='382' y='76' width='128' height='52' rx='3' fill='#e9edf3' stroke='#aebcd2'/>`);
  p.push(`<text ${MONO} x='446' y='98' fill='var(--ink)' text-anchor='middle'>W ∈ R^768×768</text>`);
  p.push(`<text ${MONO_S} x='446' y='116' fill='var(--ink-faint)' text-anchor='middle'>+ b（单线性层）</text>`);
  p.push(`<rect x='556' y='76' width='120' height='52' rx='3' fill='#f1efe8' stroke='#c6c0af'/>`);
  p.push(`<text ${MONO} x='616' y='98' fill='var(--ink)' text-anchor='middle'>z⁽ⁱʲ⁾ ∈ R^768</text>`);
  p.push(`<text ${MONO_S} x='616' y='116' fill='var(--ink-faint)' text-anchor='middle'>196 个 token 之一</text>`);

  // 数据流连线与光点
  [[264, 310, 92], [334, 382, 102], [510, 556, 102]].forEach(([x1, x2, yy]) => {
    p.push(`<line x1='${x1}' y1='${yy}' x2='${x2}' y2='${yy}' stroke='#a9a498' stroke-width='1.2'/>`);
  });
  p.push(`<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='2.8s' repeatCount='indefinite' path='M ${ix + is} ${iy + is / 2} C 180 ${iy + is / 2}, 190 ${py + ps / 2}, ${px} ${py + ps / 2} L 264 92 L 616 102'/></circle>`);

  // 等价性框
  p.push(`<rect x='32' y='196' width='400' height='76' rx='3' fill='var(--paper-raised)' stroke='var(--rule)'/>`);
  p.push(`<text ${MONO} x='48' y='222' fill='var(--ink)'>conv(k=16, s=16, 3→768) ≡ reshape + Linear</text>`);
  p.push(`<text ${CJK_S} x='48' y='242' fill='var(--ink-soft)'>k = s ⇒ 滑窗互不重叠：每像素恰入一个输出位、恰被采样一次</text>`);
  p.push(`<text ${MONO_S} x='48' y='262' fill='var(--ink-faint)'>W[o,(u,v,c)] = K[u,v,c,o] —— 同一权重的两种排布</text>`);

  // 账框
  p.push(`<rect x='452' y='196' width='236' height='92' rx='3' fill='var(--paper-sunk)' stroke='var(--rule-soft)'/>`);
  p.push(`<text ${MONO_S} x='468' y='218' fill='var(--ink)'>N = HW/p² = 196（+CLS → 197）</text>`);
  p.push(`<text ${MONO_S} x='468' y='238' fill='var(--ink-soft)'>嵌入 FLOPs = 2HWCd，与 p 无关</text>`);
  p.push(`<text ${MONO_S} x='468' y='256' fill='var(--ink-soft)'>p 减半 → N×4，attention ×16</text>`);
  p.push(`<text ${MONO_S} x='468' y='276' fill='var(--ink-faint)'>参数 0.59M / ViT-B 86M（零头）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 6.3 §5 — RVQ 残差量化层级与 bitrate 账 ============ */
function rvqLadder() {
  const w = 720, h = 356;
  const p = [svgOpen(w, h, 'RVQ 三级残差量化：逐级细化与容量、码率两笔账')];

  // 三级：输入残差条（宽度递减）→ Q_i → e_i 选中码
  const stages = [
    { label: 'r₁ = z_e', bw: 300, q: 'Q₁', e: 'e₁ ∈ C₁' },
    { label: 'r₂ = z_e − e₁', bw: 118, q: 'Q₂', e: 'e₂ ∈ C₂' },
    { label: 'r₃ = r₂ − e₂', bw: 46, q: 'Q₃', e: 'e₃ ∈ C₃' },
  ];
  const y0 = 58, rh = 62;
  stages.forEach((s, i) => {
    const y = y0 + i * rh;
    p.push(`<text ${MONO_S} x='24' y='${y - 8}' fill='var(--ink-faint)'>${s.label}</text>`);
    p.push(`<rect class='sig-bar' style='--i:${(i * 1.2).toFixed(1)}' x='24' y='${y}' width='${s.bw}' height='22' rx='2' fill='#f1efe8' stroke='#c6c0af'/>`);
    // Q_i 量化器
    p.push(`<rect x='352' y='${y - 4}' width='86' height='30' rx='3' fill='#eeeef2' stroke='#b6b6c4'/>`);
    p.push(`<text ${MONO_S} x='395' y='${y + 15}' fill='var(--ink)' text-anchor='middle'>${s.q} 最近邻</text>`);
    // e_i 选中码
    p.push(`<rect class='sig-cell' style='--i:${(0.4 + i * 1.2).toFixed(1)}' x='470' y='${y - 4}' width='96' height='30' rx='3' fill='#f3ecdc' stroke='#cbb98d'/>`);
    p.push(`<text class='sig-cell-text' style='--i:${(0.4 + i * 1.2).toFixed(1)}' ${MONO_S} x='518' y='${y + 15}' fill='var(--ink)' text-anchor='middle'>${s.e}</text>`);
    // 连线 + 流光
    p.push(`<line x1='${24 + s.bw}' y1='${y + 11}' x2='352' y2='${y + 11}' stroke='#a9a498' stroke-width='1.1'/>`);
    p.push(`<line x1='438' y1='${y + 11}' x2='470' y2='${y + 11}' stroke='#a9a498' stroke-width='1.1'/>`);
    p.push(`<circle class='sig-flow' r='2.4' fill='var(--seal)'><animateMotion dur='2s' begin='${(i * 0.5).toFixed(1)}s' repeatCount='indefinite' path='M ${24 + s.bw} ${y + 11} L 566 ${y + 11}'/></circle>`);
  });

  // 右侧注记：每级学什么 + STE 逐级适用
  p.push(`<text ${CJK_S} x='596' y='${y0 + 6}' fill='var(--ink-soft)'>第 i 级学的是前 i−1 级</text>`);
  p.push(`<text ${CJK_S} x='596' y='${y0 + 24}' fill='var(--ink-soft)'>量化完剩下的残差分布</text>`);
  p.push(`<text ${CJK_S} x='596' y='${y0 + rh + 6}' fill='var(--ink-faint)'>级数越深、残差越集中</text>`);
  p.push(`<text ${CJK_S} x='596' y='${y0 + rh + 24}' fill='var(--ink-faint)'>STE 在每级原样拷贝梯度</text>`);

  // 第 4 行：ẑ = e₁+e₂+e₃ 堆叠，与 r₁ 同尺度对齐；缺口即误差 r₄
  const y4 = y0 + 3 * rh;
  p.push(`<text ${MONO_S} x='24' y='${y4 - 8}' fill='var(--ink-faint)'>ẑ = e₁ + e₂ + e₃（与 r₁ 同尺度）</text>`);
  const segs = [
    { w: 182, c: '#cbb98d', t: 'e₁' },
    { w: 72, c: '#b6b6c4', t: 'e₂' },
    { w: 28, c: '#cf9d92', t: 'e₃' },
  ];
  let acc = 24;
  segs.forEach((s, i) => {
    p.push(`<rect class='sig-bar' style='--i:${(0.6 + i * 1.2).toFixed(1)}' x='${acc}' y='${y4}' width='${s.w}' height='22' fill='${s.c}' fill-opacity='0.35' stroke='${s.c}'/>`);
    p.push(`<text ${MONO_S} x='${acc + s.w / 2}' y='${y4 + 15}' fill='var(--ink-soft)' text-anchor='middle'>${s.t}</text>`);
    acc += s.w;
  });
  // 误差缺口（r₄）
  p.push(`<rect x='${acc}' y='${y4}' width='18' height='22' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1' stroke-dasharray='2 2'/>`);
  p.push(`<text ${MONO_S} x='${acc + 26}' y='${y4 + 15}' fill='var(--seal)'>r₄</text>`);
  p.push(`<text ${MONO_S} x='352' y='${y4 + 15}' fill='var(--seal)'>误差 = ‖r₄‖ = ‖z_e − ẑ‖（逐级递减是经验性质，无严格单调保证）</text>`);

  // 底部两笔账
  const by = y4 + 40;
  p.push(`<rect x='24' y='${by}' width='664' height='56' rx='3' fill='var(--paper-sunk)' stroke='var(--rule-soft)'/>`);
  p.push(`<text ${MONO_S} x='40' y='${by + 22}' fill='var(--ink)'>容量：每位点 (k₁..k₈) ∈ [1024]⁸ → 有效词表 2⁸⁰；参数仅 8 张 1024×d_e 的表</text>`);
  p.push(`<text ${MONO_S} x='40' y='${by + 42}' fill='var(--ink)'>bitrate：b = n_q·log₂K·f = 8×10×75 = 6000 bps ≈ 6 kbps；只传前 m 级即低码率版</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 5.6 §2.2 — 1M 窗 KV 墙的三条出路 ============ */
function kvWallRoutes() {
  const w = 720, h = 346;
  const p = [svgOpen(w, h, '1M 上下文 KV cache 墙与压读取、压层数、压可见窗三条出路')];

  // 顶部：墙
  p.push(`<rect x='24' y='20' width='664' height='40' rx='3' fill='var(--paper-sunk)' stroke='var(--rule-soft)'/>`);
  p.push(`<text ${MONO_S} x='40' y='44' fill='var(--ink)'>M_cache = slope × T ｜ T = 1M：纯 GQA 192–368 GiB/序列——内存与算力双斜率顶墙</text>`);

  const lanes = [
    {
      tag: '① 压读取',
      who: 'V3.2 DSA',
      mech: '576 元素/token 全量驻留；每步只读 indexer 选中的行',
      num: '读 144 MB vs 9.2 GB',
      sub: '@131072 · 64×',
    },
    {
      tag: '② 压层数',
      who: 'Qwen3-Next 3:1',
      mech: '48 层只 12 层付斜率；linear 层付与 T 无关的定长状态',
      num: '24 KiB/token',
      sub: '+ 36 MiB 定长',
    },
    {
      tag: '③ 压可见窗',
      who: 'Llama-4 Scout',
      mech: 'RoPE 层打分裁进 chunk 8192；每第 4 层 NoPE + 对数温度',
      num: '192 KiB/token',
      sub: 'slope 不变，可见窗重排',
    },
  ];
  const y0 = 84;
  lanes.forEach((ln, i) => {
    const y = y0 + i * 72;
    p.push(`<text ${CJK} x='24' y='${y + 20}' fill='var(--ink)'>${ln.tag}</text>`);
    p.push(`<text ${MONO_S} x='24' y='${y + 38}' fill='var(--ink-faint)'>${ln.who}</text>`);
    p.push(`<rect x='150' y='${y}' width='360' height='52' rx='3' fill='var(--paper-raised)' stroke='var(--rule)'/>`);
    p.push(`<text ${CJK_S} x='164' y='${y + 22}' fill='var(--ink-soft)'>${ln.mech}</text>`);
    p.push(`<text ${MONO} x='545' y='${y + 22}' fill='var(--seal)'>${ln.num}</text>`);
    p.push(`<text ${MONO_S} x='545' y='${y + 40}' fill='var(--ink-faint)'>${ln.sub}</text>`);
    p.push(`<path class='sig-dashflow' d='M 158 ${y + 46} C 300 ${y + 62}, 420 ${y + 60}, 530 ${y + 34}' fill='none' stroke='var(--seal)' stroke-width='1.1'/>`);
  });

  // 压层数的小层型图（lane ② 左下角）：12 格中每 4 格 1 格 full 付斜率
  const gy = y0 + 72;
  for (let k = 0; k < 12; k++) {
    const full = k % 4 === 3;
    p.push(`<rect x='${24 + k * 9.5}' y='${gy + 46}' width='8' height='10' rx='1.5' fill='${full ? '#f6e9e5' : 'none'}' stroke='${full ? '#cf9d92' : 'var(--rule)'}' stroke-width='1'/>`);
  }

  // 对照：GLM-4.6 不答此题
  const cy = y0 + 3 * 72 + 8;
  p.push(`<line x1='24' y1='${cy}' x2='688' y2='${cy}' stroke='var(--rule)' stroke-width='1'/>`);
  p.push(`<text ${CJK_S} x='24' y='${cy + 24}' fill='var(--ink-soft)'>对照：GLM-4.6 留位——368 KiB/token 斜率照付，@202752 窗 → 71.1 GiB/序列（θ = 1e6 + partial 0.5 原生覆盖，窗长停在 200K 量级）</text>`);
  p.push('</svg>');
  return p.join('');
}

export const FIGURES = [
  {
    slug: '05-case-studies/01-llama-lineage',
    afterHeading: /^## 1\. 配置谱系总表/,
    title: 'Llama 谱系 config 账本：逐代字段 diff',
    svg: llamaLedger,
  },
  {
    slug: '05-case-studies/02-deepseek',
    afterHeading: /^### 3\.1 路由三件套/,
    title: 'aux-loss-free：均衡作为控制回路',
    svg: auxLossFree,
  },
  {
    slug: '06-vision-representations/01-vit',
    afterHeading: /^## 1\. Patchify/,
    title: 'patchify：conv ≡ linear 的 shape 流',
    svg: vitPatchify,
  },
  {
    slug: '06-vision-representations/03-autoencoders-and-tokenizers',
    afterHeading: /^## 5\. RVQ/,
    title: 'RVQ 残差层级：逐级细化与 bitrate 账',
    svg: rvqLadder,
  },
  {
    slug: '05-case-studies/06-cross-lab-comparison',
    afterHeading: /^### 2\.2 为什么四家上了 subquadratic/,
    title: '1M 窗 KV 账的三条出路',
    svg: kvWallRoutes,
  },
];
