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
  const fx = (i) => 178 + i * 90, fw = 84;
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
    // 代际箭头：上一行到本行
    if (ri > 0) {
      const yA = y0 + (ri - 1) * (rh + gap) + rh, yB = y;
      p.push(`<line x1='150' y1='${yA + 1}' x2='150' y2='${yB - 1}' stroke='#a9a498' stroke-width='1.1'/>`);
    }
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
  p.push(`<path d='M 182 ${y + 26} C 182 152, 300 152, 492 138' fill='none' stroke='var(--ink-faint)' stroke-width='1' stroke-dasharray='3 4'/>`);
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

  p.push(`<text ${CJK_S} x='24' y='300' fill='var(--ink-faint)'>虚线回路 = 控制信号（无梯度穿过 b_i）；V2 的三路 aux loss 在 V3 被这条回路整体替换</text>`);
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
];
