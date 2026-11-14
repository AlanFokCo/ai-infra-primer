// Part 1 + Part 2 签名动图：文本主干基础（形式化 / 账 / scaling law / residual stream / BPE / norm 摆放）
// 规范见 STYLE.md；机制动画只表现数据流、窗口渐长、账的累加、梯度通路。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

function svgOpen(w, h, label) {
  return `<svg class="fig-svg sig" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;
}

// 实心小三角箭头
function arrowHead(x, y, dir, fill = 'var(--ink-soft)', opacity = 1) {
  const s = 4.2;
  const pts = {
    up: `${x},${y} ${x - s},${y + s * 1.3} ${x + s},${y + s * 1.3}`,
    down: `${x},${y} ${x - s},${y - s * 1.3} ${x + s},${y - s * 1.3}`,
    left: `${x},${y} ${x + s * 1.3},${y - s} ${x + s * 1.3},${y + s}`,
    right: `${x},${y} ${x - s * 1.3},${y - s} ${x - s * 1.3},${y + s}`,
  };
  return `<polygon points="${pts[dir]}" fill="${fill}" opacity="${opacity}"/>`;
}

/* ============ 1.1 §3 — 自回归分解：链式展开与渐长的条件窗口 ============ */
function arFactorization() {
  const w = 720, h = 330;
  const p = [svgOpen(w, h, '链式法则把联合分布拆成逐位置条件分布的乘积，条件窗口随 t 渐长')];
  p.push(`<text ${CJK_S} x='32' y='24' fill='var(--ink-faint)'>链式法则逐步剥离最右变量：每套一次定义式，联合长度减 1，直到 P(x₁ | x₀)（T = 4 实例）</text>`);

  // 顶行：联合 = 四个条件因子的乘积
  p.push(`<rect x='32' y='42' width='96' height='30' rx='3' fill='#f1efe8' stroke='#c6c0af' stroke-width='1.1'/>`);
  p.push(`<text ${MONO} x='80' y='62' fill='var(--ink)' text-anchor='middle'>P(x₁:₄)</text>`);
  p.push(`<text ${MONO} x='140' y='62' fill='var(--ink-soft)'>=</text>`);
  const chips = ['P(x₁ | x₀)', 'P(x₂ | x₁)', 'P(x₃ | x₁:₂)', 'P(x₄ | x₁:₃)'];
  const chipX = [158, 262, 366, 470];
  chips.forEach((c, i) => {
    p.push(`<rect x='${chipX[i]}' y='42' width='86' height='30' rx='3' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${chipX[i] + 43}' y='61' fill='var(--seal-deep)' text-anchor='middle'>${c}</text>`);
    if (i < 3) p.push(`<text ${MONO} x='${chipX[i] + 95}' y='62' fill='var(--ink-soft)'>×</text>`);
  });
  p.push(`<text ${CJK_S} x='580' y='52' fill='var(--ink-soft)'>每个因子 = 同一 f_θ</text>`);
  p.push(`<text ${CJK_S} x='580' y='68' fill='var(--ink-faint)'>的一次调用（式 1）</text>`);

  // token 行：x₀=bos + x₁..x₄，与上方因子对齐
  const tokC = [201, 305, 409, 513];
  p.push(`<rect x='82' y='110' width='34' height='30' rx='3' fill='#f3ecdc' stroke='#cbb98d' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='99' y='129' fill='var(--ink)' text-anchor='middle'>x₀</text>`);
  p.push(`<text ${MONO_S} x='99' y='154' fill='var(--ink-faint)' text-anchor='middle'>bos</text>`);
  tokC.forEach((cx, i) => {
    p.push(`<rect x='${cx - 17}' y='110' width='34' height='30' rx='3' fill='#f1efe8' stroke='#c6c0af' stroke-width='1.1'/>`);
    p.push(`<text ${MONO} x='${cx}' y='130' fill='var(--ink)' text-anchor='middle'>x${['₁','₂','₃','₄'][i]}</text>`);
  });

  // 条件窗口条（逐 t 亮起）：bar t 覆盖 x₀..x_{t-1}
  const bars = [
    { y: 156, wdt: 20, end: 114 },
    { y: 173, wdt: 117, end: 211 },
    { y: 190, wdt: 222, end: 316 },
    { y: 207, wdt: 327, end: 421 },
  ];
  bars.forEach((b, i) => {
    p.push(`<rect class='sig-step' style='--i:${i}' x='94' y='${b.y}' width='${b.wdt}' height='13' rx='3' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
    p.push(`<text class='sig-step' style='--i:${i}' ${MONO_S} x='${b.end + 8}' y='${b.y - 4}' fill='var(--ink-faint)'>x_&lt;${i + 1}${i === 0 ? ' = bos' : ''}</text>`);
  });
  // 窗口 → 因子 chip 的连接与流光（走 token 间隙的竖直通道）
  const ports = [168, 272, 376, 480];
  const barMid = [162, 179.5, 196.5, 213.5];
  const barEnd = [114, 211, 316, 421];
  ports.forEach((px, i) => {
    const d = `M ${px} 72 V ${barMid[i]} H ${barEnd[i]}`;
    p.push(`<path d='${d}' fill='none' stroke='var(--rule)' stroke-width='1.1'/>`);
    p.push(
      `<circle class='sig-flow' r='2.4' fill='var(--seal)'>` +
      `<animateMotion dur='2s' begin='${(0.5 + i * 0.4).toFixed(1)}s' repeatCount='indefinite' path='M ${barEnd[i]} ${barMid[i]} H ${px} V 72'/>` +
      `</circle>`
    );
  });

  // 底行：列表参数化（划线否决）→ 参数共享
  p.push(`<rect x='32' y='252' width='296' height='52' rx='3' fill='var(--paper-sunk)' stroke='var(--rule)' stroke-width='1' stroke-dasharray='4 3'/>`);
  p.push(`<text ${MONO_S} x='44' y='272' fill='var(--ink-faint)'>列表参数化：V^T − 1 个自由参数</text>`);
  p.push(`<line x1='44' y1='268' x2='212' y2='268' stroke='var(--ink-faint)' stroke-width='1'/>`);
  p.push(`<text ${MONO_S} x='44' y='292' fill='var(--ink-faint)'>GPT-2：50257^1024，连写都写不下</text>`);
  p.push(`<line x1='336' y1='278' x2='382' y2='278' stroke='#a9a498' stroke-width='1.2'/>`);
  p.push(arrowHead(384, 278, 'right'));
  p.push(`<rect x='392' y='252' width='296' height='52' rx='3' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<text ${MONO_S} x='404' y='272' fill='var(--seal-deep)'>T 个条件分布，同一 f_θ 参数共享</text>`);
  p.push(`<text ${CJK_S} x='404' y='292' fill='var(--ink-soft)'>指数爆炸被参数共享压掉（式 3）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 1.2 §5 — C = 6ND：一个参数 × 一个 token 的 6 次浮点运算 ============ */
function compute6ND() {
  const w = 720, h = 320;
  const p = [svgOpen(w, h, '前向 1 次 FMA、反向对输入与权重各一份梯度：每参数每 token 6 FLOPs，乘 N 乘 D 得 C = 6ND')];
  // 左：单个 matmul 元素的机制
  p.push(`<text ${CJK_S} x='28' y='26' fill='var(--ink-soft)'>一处 matmul 元素的账（y = xW，每 token）</text>`);
  p.push(`<rect x='36' y='92' width='56' height='30' rx='3' fill='#f1efe8' stroke='#c6c0af' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='64' y='111' fill='var(--ink)' text-anchor='middle'>x_is</text>`);
  p.push(`<rect class='sig-diag' style='--i:0' x='140' y='92' width='56' height='30' rx='3' fill='#e9edf3' stroke='#aebcd2' stroke-width='1.2'/>`);
  p.push(`<text ${MONO_S} x='168' y='111' fill='var(--ink)' text-anchor='middle'>W_sj</text>`);
  p.push(`<rect x='244' y='92' width='56' height='30' rx='3' fill='#eeeef2' stroke='#b6b6c4' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='272' y='111' fill='var(--ink)' text-anchor='middle'>y_ij</text>`);
  p.push(`<line x1='92' y1='107' x2='134' y2='107' stroke='#a9a498' stroke-width='1.2'/>` + arrowHead(138, 107, 'right'));
  p.push(`<line x1='196' y1='107' x2='238' y2='107' stroke='#a9a498' stroke-width='1.2'/>` + arrowHead(242, 107, 'right'));
  p.push(`<text ${MONO_S} x='168' y='82' fill='var(--ink-soft)' text-anchor='middle'>前向：1 FMA = 2 FLOPs</text>`);
  // 上游梯度与两条反向路径
  p.push(`<text ${MONO_S} x='272' y='238' fill='var(--ink-faint)' text-anchor='middle'>上游梯度 G = ∂L/∂y</text>`);
  p.push(`<rect x='244' y='190' width='56' height='30' rx='3' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='272' y='209' fill='var(--seal-deep)' text-anchor='middle'>G_ij</text>`);
  p.push(`<path class='sig-dashflow' d='M 244 205 H 64 V 126' fill='none' stroke='var(--seal)' stroke-width='1.4'/>`);
  p.push(arrowHead(64, 124, 'up', 'var(--seal)'));
  p.push(`<text ${MONO_S} x='154' y='196' fill='var(--seal)' text-anchor='middle'>∂L/∂x = G·W^T（+2）</text>`);
  p.push(`<path class='sig-dashflow' d='M 272 190 V 154 H 168 V 126' fill='none' stroke='var(--seal)' stroke-width='1.4'/>`);
  p.push(arrowHead(168, 124, 'up', 'var(--seal)'));
  p.push(`<text ${MONO_S} x='220' y='144' fill='var(--seal)' text-anchor='middle'>∂L/∂W += x^T·G（+2）</text>`);
  p.push(`<text ${CJK_S} x='28' y='252' fill='var(--ink-soft)'>反向两份梯度：输入侧、权重侧各与前向同阶（式 3）</text>`);
  p.push(`<text ${CJK_S} x='28' y='272' fill='var(--ink-faint)'>∂L/∂W 需要前向的 x —— 激活账（§8 账二）的来源</text>`);
  p.push(`<line x1='330' y1='36' x2='330' y2='290' stroke='var(--rule-soft)' stroke-width='1'/>`);
  // 右：构成条 2N+2N+2N = 6N，× D → C
  p.push(`<text ${CJK_S} x='360' y='26' fill='var(--ink-soft)'>逐层求和（式 2 + 式 4）× D 个 token</text>`);
  const segs = [
    { x: 380, fill: '#e9edf3', stroke: '#aebcd2', label: '前向 xW' },
    { x: 470, fill: 'var(--seal-wash)', stroke: 'var(--seal)', label: '反向·输入侧' },
    { x: 560, fill: 'var(--seal-wash)', stroke: 'var(--seal)', label: '反向·权重侧' },
  ];
  segs.forEach((s, i) => {
    p.push(`<rect class='sig-bar' style='--i:${i}' x='${s.x}' y='48' width='90' height='22' fill='${s.fill}' stroke='${s.stroke}' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${s.x + 45}' y='63' fill='var(--ink)' text-anchor='middle'>2N</text>`);
    p.push(`<text ${MONO_S} x='${s.x + 45}' y='84' fill='var(--ink-faint)' text-anchor='middle'>${s.label}</text>`);
  });
  p.push(`<text ${MONO} x='515' y='108' fill='var(--ink)' text-anchor='middle'>每 token：2N + 2N + 2N = 6N</text>`);
  p.push(`<text ${MONO} x='360' y='136' fill='var(--ink)'>× D（训练 token 总数）</text>`);
  p.push(`<line x1='520' y1='130' x2='552' y2='130' stroke='#a9a498' stroke-width='1.2'/>` + arrowHead(556, 130, 'right'));
  p.push(`<rect class='sig-diag' style='--i:2' x='562' y='114' width='128' height='32' rx='3' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.4'/>`);
  p.push(`<text ${MONO} x='626' y='135' fill='var(--seal-deep)' text-anchor='middle'>C = 6ND（式 5）</text>`);
  // 锚点对账
  p.push(`<rect x='360' y='162' width='330' height='60' rx='3' fill='#f3ecdc' stroke='#cbb98d' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='372' y='184' fill='var(--ink)'>DeepSeek-V3（式 9）：6 × 37×10^9 × 14.8×10^12</text>`);
  p.push(`<text ${MONO} x='372' y='208' fill='var(--ink)'>= 3.29 × 10^24 FLOPs</text>`);
  p.push(`<text ${MONO_S} x='360' y='246' fill='var(--ink-faint)'>Llama-3.1-8B：6N + 6Vd ≈ 4.50×10^10 FLOPs / token（§9）</text>`);
  p.push(`<text ${CJK_S} x='360' y='268' fill='var(--ink-faint)'>适用：dense 前向、无重算、T ≪ N/(Ld)（§5 五条）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 1.3 §4 — log-log 下的 L*(C) frontier：同一预算、两种分配 ============ */
function scalingFrontier() {
  const w = 720, h = 340;
  const p = [svgOpen(w, h, 'log-log 坐标下 compute-optimal frontier 是幂律直线；Gopher 与 Chinchilla 预算相近，分配不同，落点不同')];
  const px = (logC) => 96 + ((logC - 21.5) / 4) * 592;
  const py = (L) => 272 - ((Math.log10(L) - 0.215) / 0.153) * 224;

  // 网格与坐标轴
  const yTicks = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3];
  yTicks.forEach((L) => {
    p.push(`<line x1='96' y1='${py(L).toFixed(1)}' x2='688' y2='${py(L).toFixed(1)}' stroke='var(--rule-soft)' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='88' y='${(py(L) + 3.5).toFixed(1)}' fill='var(--ink-faint)' text-anchor='end'>${L.toFixed(1)}</text>`);
  });
  [22, 23, 24, 25].forEach((e) => {
    p.push(`<line x1='${px(e)}' y1='48' x2='${px(e)}' y2='272' stroke='var(--rule-soft)' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='${px(e)}' y='288' fill='var(--ink-faint)' text-anchor='middle'>10^${e}</text>`);
  });
  p.push(`<line x1='96' y1='48' x2='96' y2='272' stroke='var(--ink-faint)' stroke-width='1'/>`);
  p.push(`<line x1='96' y1='272' x2='688' y2='272' stroke='var(--ink-faint)' stroke-width='1'/>`);
  p.push(`<text ${MONO_S} x='96' y='38' fill='var(--ink-soft)'>L（test CE，nats，log）· frontier L*(C)（式 7）</text>`);
  p.push(`<text ${CJK_S} x='392' y='314' fill='var(--ink-soft)' text-anchor='middle'>C（训练算力，FLOPs，log）</text>`);

  // 不可约地板 E
  p.push(`<line x1='96' y1='${py(1.69).toFixed(1)}' x2='688' y2='${py(1.69).toFixed(1)}' stroke='var(--seal)' stroke-width='1' stroke-dasharray='5 4' opacity='0.7'/>`);
  p.push(`<text ${CJK_S} x='100' y='246' fill='var(--seal)'>E = 1.69 不可约地板（语言本身的熵）</text>`);

  // frontier 折线（按式 7 数值采样）
  const samples = [21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5];
  const frontier = (logC) => 1.69 + 813.7 * Math.pow(Math.pow(10, logC) / 6, -0.154);
  const path = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${px(s).toFixed(1)} ${py(frontier(s)).toFixed(1)}`).join(' ');
  p.push(`<path d='${path}' fill='none' stroke='var(--ink)' stroke-width='1.6'/>`);
  p.push(
    `<circle class='sig-flow' r='2.8' fill='var(--seal)'>` +
    `<animateMotion dur='4s' begin='0.4s' repeatCount='indefinite' path='${path}'/>` +
    `</circle>`
  );
  p.push(`<text ${MONO_S} x='540' y='230' fill='var(--ink-soft)'>斜率 −0.154（log-log）</text>`);
  p.push(`<text ${MONO_S} x='540' y='246' fill='var(--ink-faint)'>C ×10 ⇒ (L−E) ×0.70</text>`);

  // 两个真实预算点（N、D 出自 §4；L 为式 2 估计）
  const gopher = { logC: Math.log10(5.04e23), L: 1.993 };
  const chinchilla = { logC: Math.log10(5.88e23), L: 1.937 };
  p.push(`<line x1='${px(gopher.logC).toFixed(1)}' y1='${(py(gopher.L) + 5).toFixed(1)}' x2='${px(gopher.logC).toFixed(1)}' y2='${(py(chinchilla.L) - 2).toFixed(1)}' stroke='var(--ink-faint)' stroke-width='1' stroke-dasharray='3 3'/>`);
  p.push(`<text ${MONO_S} x='446' y='160' fill='var(--ink-faint)'>ΔL ≈ 0.06</text>`);
  p.push(`<circle cx='${px(gopher.logC).toFixed(1)}' cy='${py(gopher.L).toFixed(1)}' r='4.5' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.4'/>`);
  p.push(`<text ${MONO_S} x='${(px(gopher.logC) + 12).toFixed(1)}' y='${(py(gopher.L) - 4).toFixed(1)}' fill='var(--ink)'>Gopher：280B × 0.3T，L ≈ 1.99</text>`);
  p.push(`<circle class='sig-diag' style='--i:1' cx='${px(chinchilla.logC).toFixed(1)}' cy='${py(chinchilla.L).toFixed(1)}' r='9' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<circle cx='${px(chinchilla.logC).toFixed(1)}' cy='${py(chinchilla.L).toFixed(1)}' r='4.5' fill='var(--seal)'/>`);
  p.push(`<text ${MONO_S} x='${(px(chinchilla.logC) - 12).toFixed(1)}' y='${(py(chinchilla.L) + 20).toFixed(1)}' fill='var(--seal-deep)' text-anchor='end'>Chinchilla：70B × 1.4T，L ≈ 1.94</text>`);

  p.push(`<text ${CJK_S} x='96' y='332' fill='var(--ink-faint)'>同一 C ≈ 5×10^23 FLOPs：Gopher 把预算给参数，Chinchilla 给数据——后者落在 frontier 上</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 2.1 §2 — residual stream 总线：归一化读、加法写 ============ */
function residualBus() {
  const w = 720, h = 300;
  const p = [svgOpen(w, h, 'residual stream 贯穿整机：子层经归一化读、经加法写，stream 只加不覆盖')];
  p.push(`<text ${MONO} x='36' y='28' fill='var(--ink)'>x ← x + f(LN(x))：每个子层归一化读、加法写</text>`);

  // 入口 embedding
  p.push(`<rect x='30' y='116' width='132' height='28' rx='3' fill='#f1efe8' stroke='#c6c0af' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='96' y='134' fill='var(--ink)' text-anchor='middle'>x^(0) = E_x（查表）</text>`);

  // stream 主干（虚线流动 = 残差通路）
  p.push(`<line x1='162' y1='130' x2='700' y2='130' stroke='#8b867a' stroke-width='2.4' stroke-linecap='round'/>`);
  p.push(`<line class='sig-dashflow' x1='162' y1='130' x2='700' y2='130' stroke='var(--seal)' stroke-width='1.4'/>`);
  p.push(
    `<circle class='sig-flow' r='2.6' fill='var(--seal)'>` +
    `<animateMotion dur='3.2s' begin='0s' repeatCount='indefinite' path='M 162 130 H 700'/>` +
    `</circle>`
  );

  // 两个旁挂子层（attn / ffn），各自由 tap 读、⊕ 写回
  const lobes = [
    { tap: 230, wr: 300, name: 'Attn_l', fill: '#f6e9e5', stroke: '#cf9d92', read: true },
    { tap: 420, wr: 490, name2: true, fill: '#eeeef2', stroke: '#b6b6c4', label: 'FFN_l' },
  ];
  lobes.forEach((lb, i) => {
    const name = lb.read ? 'Attn_l' : 'FFN_l';
    // 读：stream → RMSNorm → 子层
    p.push(`<line x1='${lb.tap}' y1='130' x2='${lb.tap}' y2='154' stroke='#a9a498' stroke-width='1.2'/>`);
    p.push(`<rect x='${lb.tap - 32}' y='154' width='64' height='22' rx='3' fill='#e9efe9' stroke='#a9bfa9' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${lb.tap}' y='169' fill='var(--ink)' text-anchor='middle'>RMSNorm</text>`);
    p.push(`<line x1='${lb.tap}' y1='176' x2='${lb.tap}' y2='200' stroke='#a9a498' stroke-width='1.2'/>`);
    p.push(`<rect x='${lb.tap - 37}' y='200' width='74' height='24' rx='3' fill='${lb.fill}' stroke='${lb.stroke}' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${lb.tap}' y='216' fill='var(--ink)' text-anchor='middle'>${name}</text>`);
    // 写：子层 → ⊕
    p.push(`<path d='M ${lb.tap + 37} 212 H ${lb.wr} V 141' fill='none' stroke='#a9a498' stroke-width='1.2'/>`);
    p.push(`<circle cx='${lb.wr}' cy='130' r='9' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.3'/>`);
    p.push(`<text ${MONO} x='${lb.wr}' y='135' fill='var(--ink)' text-anchor='middle'>+</text>`);
    // 机制流光：读进子层、写回 stream
    p.push(
      `<circle class='sig-flow' r='2.4' fill='var(--seal)'>` +
      `<animateMotion dur='2.4s' begin='${(0.3 + i * 0.9).toFixed(1)}s' repeatCount='indefinite' ` +
      `path='M ${lb.tap} 130 V 165 V 212 H ${lb.wr} V 130'/>` +
      `</circle>`
    );
  });
  p.push(`<text ${MONO_S} x='242' y='146' fill='var(--ink-faint)'>读：LN(x)</text>`);
  p.push(`<text ${MONO_S} x='313' y='122' fill='var(--ink-faint)'>写：+（不覆盖）</text>`);

  // layer 括框
  p.push(`<rect x='182' y='96' width='340' height='144' rx='4' fill='none' stroke='var(--rule)' stroke-width='1' stroke-dasharray='4 4'/>`);
  p.push(`<text ${MONO_S} x='512' y='110' fill='var(--ink-faint)' text-anchor='end'>decoder layer ℓ × L = 32</text>`);
  p.push(`<text ${MONO} x='538' y='135' fill='var(--ink-faint)'>…</text>`);

  // 收尾：final norm + 输出头
  p.push(`<rect x='552' y='116' width='80' height='28' rx='3' fill='#e9efe9' stroke='#a9bfa9' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='592' y='134' fill='var(--ink)' text-anchor='middle'>RMSNorm</text>`);
  p.push(arrowHead(646, 130, 'right'));
  p.push(`<rect x='650' y='117' width='58' height='26' rx='3' fill='#e9edf3' stroke='#aebcd2' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='679' y='134' fill='var(--ink)' text-anchor='middle'>W_U</text>`);
  p.push(`<text ${MONO_S} x='679' y='160' fill='var(--ink-faint)' text-anchor='middle'>→ z [B,T,V]</text>`);

  // 底行：展开式与推论
  p.push(`<text ${MONO} x='36' y='268' fill='var(--ink)'>x^(L) = x^(0) + Σ a_ℓ + Σ f_ℓ —— 2L+1 项之和（式 2）</text>`);
  p.push(`<text ${CJK_S} x='36' y='290' fill='var(--ink-soft)'>同一总线：读都过 LN、写都是加；任意中间态 ×W_U 即可读（logit lens 前提，4.7）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 2.2 §2 — BPE：逐轮 merge 与频次账本 ============ */
function bpeMerges() {
  const w = 720, h = 320;
  const p = [svgOpen(w, h, 'BPE 逐轮把最高频相邻对焊成新符号，语料总长随之缩减')];
  p.push(`<text ${CJK_S} x='32' y='26' fill='var(--ink-soft)'>语料：low ×5 · lower ×2 · newest ×6 · widest ×3（§2 手算例）——每轮取加权频次最高的相邻对焊成新符号</text>`);

  // 单元格绘制辅助：slot i 起点 x = 110 + 29*i
  const cell = (x, y, txt, cls = '', delay = 0) => {
    const st = cls ? ` class='${cls}' style='--i:${delay}'` : '';
    const seal = cls.includes('sig-cell');
    p.push(`<rect${st} x='${x}' y='${y}' width='26' height='24' rx='2' fill='${seal ? 'var(--seal-wash)' : 'var(--paper)'}' stroke='${seal ? 'var(--seal)' : 'var(--rule)'}' stroke-width='1'/>`);
  };
  const cellText = (x, y, txt, cls = '', delay = 0) => {
    const st = cls ? ` class='${cls}' style='--i:${delay}'` : '';
    p.push(`<text${st} ${MONO} x='${x + 13}' y='${y + 17}' fill='${cls ? 'var(--seal-deep)' : 'var(--ink)'}' text-anchor='middle'>${txt}</text>`);
  };
  // 合并格（跨 n 个 slot）
  const merged = (x, y, n, txt, delay) => {
    const wdt = 26 * n + 3 * (n - 1);
    p.push(`<rect class='sig-cell' style='--i:${delay}' x='${x}' y='${y}' width='${wdt}' height='24' rx='2' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>`);
    p.push(`<text class='sig-cell-text' style='--i:${delay}' ${MONO} x='${x + wdt / 2}' y='${y + 17}' fill='var(--seal-deep)' text-anchor='middle'>${txt}</text>`);
  };

  // 轨道 A：newest ×6
  p.push(`<text ${MONO_S} x='32' y='50' fill='var(--ink-faint)'>newest ×6</text>`);
  const rowsA = [
    { y: 64, label: '初始', cells: ['n', 'e', 'w', 'e', 's', 't'] },
    { y: 98, label: 'rank 1', cells: ['n', 'e', 'w', '@es@2', 't'] },
    { y: 132, label: 'rank 2', cells: ['n', 'e', 'w', '@est@3'] },
  ];
  rowsA.forEach((r, ri) => {
    p.push(`<text ${MONO_S} x='100' y='${r.y + 17}' fill='var(--ink-faint)' text-anchor='end'>${r.label}</text>`);
    let slot = 0;
    r.cells.forEach((c) => {
      const x = 110 + 29 * slot;
      if (c.startsWith('@')) {
        const [, txt, n] = c.split('@');
        merged(x, r.y, +n, txt, ri * 0.9 + 0.3);
        slot += +n;
      } else {
        cell(x, r.y, c); cellText(x, r.y, c);
        slot += 1;
      }
    });
  });
  // 待焊对标记（rank 1: e+s；rank 2: es+t）
  p.push(`<path d='M 195 92 Q 209 97 223 92' fill='none' stroke='var(--seal)' stroke-width='1.1'/>`);
  p.push(`<path d='M 195 126 Q 239 132 283 126' fill='none' stroke='var(--seal)' stroke-width='1.1'/>`);

  // 轨道 B：low ×5
  p.push(`<text ${MONO_S} x='32' y='174' fill='var(--ink-faint)'>low ×5</text>`);
  const rowsB = [
    { y: 188, label: '初始', cells: ['l', 'o', 'w'] },
    { y: 222, label: 'rank 3', cells: ['@lo@2', 'w'] },
    { y: 256, label: 'rank 4', cells: ['@low@3'] },
  ];
  rowsB.forEach((r, ri) => {
    p.push(`<text ${MONO_S} x='100' y='${r.y + 17}' fill='var(--ink-faint)' text-anchor='end'>${r.label}</text>`);
    let slot = 0;
    r.cells.forEach((c) => {
      const x = 110 + 29 * slot;
      if (c.startsWith('@')) {
        const [, txt, n] = c.split('@');
        merged(x, r.y, +n, txt, ri * 0.9 + 1.9);
        slot += +n;
      } else {
        cell(x, r.y, c); cellText(x, r.y, c);
        slot += 1;
      }
    });
  });
  p.push(`<path d='M 108 216 Q 123 221 137 216' fill='none' stroke='var(--seal)' stroke-width='1.1'/>`);
  p.push(`<path d='M 108 250 Q 152 256 195 250' fill='none' stroke='var(--seal)' stroke-width='1.1'/>`);
  p.push(`<text ${CJK_S} x='32' y='306' fill='var(--ink-faint)'>编码新词按 rank 序套用：lowest → low + est（未见词 = 已见子词的组合）</text>`);

  // 右：merge 账本
  p.push(`<text ${MONO_S} x='430' y='50' fill='var(--ink-faint)'>merge 账本：频次 = 语料缩短量</text>`);
  p.push(`<text ${MONO_S} x='432' y='68' fill='var(--ink-faint)'>#</text>`);
  p.push(`<text ${MONO_S} x='456' y='68' fill='var(--ink-faint)'>merge</text>`);
  p.push(`<text ${MONO_S} x='532' y='68' fill='var(--ink-faint)'>n(a,b)</text>`);
  p.push(`<text ${MONO_S} x='598' y='68' fill='var(--ink-faint)'>L_corpus</text>`);
  const ledger = [
    { r: '0', m: '—', n: '—', len: 79, d: null },
    { r: '1', m: 'e+s→es', n: '9', len: 70, d: 0.3 },
    { r: '2', m: 'es+t→est', n: '9', len: 61, d: 1.2 },
    { r: '3', m: 'l+o→lo', n: '7', len: 54, d: 2.1 },
    { r: '4', m: 'lo+w→low', n: '7', len: 47, d: 3.0 },
  ];
  ledger.forEach((row, i) => {
    const y = 90 + i * 24;
    const cls = i === 0 ? '' : ` class='sig-cell' style='--i:${row.d}'`;
    p.push(`<text${cls} ${MONO_S} x='432' y='${y}' fill='var(--ink)'>${row.r}</text>`);
    p.push(`<text${cls} ${MONO_S} x='456' y='${y}' fill='var(--ink)'>${row.m}</text>`);
    p.push(`<text${cls} ${MONO_S} x='532' y='${y}' fill='var(--ink)'>${row.n}</text>`);
    const bw = row.len * 1.25;
    const bcls = i === 0 ? '' : ` class='sig-bar' style='--i:${(row.d * 3).toFixed(1)}'`;
    p.push(`<rect${bcls} x='598' y='${y - 10}' width='${bw.toFixed(1)}' height='12' rx='2' fill='#f1efe8' stroke='#c6c0af' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='${598 + bw + 6}' y='${y}' fill='var(--ink-faint)'>${row.len}</text>`);
  });
  p.push(`<text ${CJK_S} x='430' y='226' fill='var(--ink-soft)'>79 → 47：四轮 merge 砍掉约 40% 符号</text>`);
  p.push(`<text ${CJK_S} x='430' y='246' fill='var(--ink-faint)'>词表每轮 +1，直到预算 V（GPT-2：256 + 50000 + 1）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 2.7 §5 — Pre-LN vs Post-LN：梯度通路对照 ============ */
function prePostLN() {
  const w = 720, h = 360;
  const p = [svgOpen(w, h, 'Post-LN 的梯度每层过 LN Jacobian 被重投影削弱；Pre-LN 的恒等通路直达输入端')];
  p.push(`<text ${MONO} x='40' y='24' fill='var(--ink)'>Post-LN：x ← LN(x + f(x))</text>`);
  p.push(`<text ${MONO} x='390' y='24' fill='var(--ink)'>Pre-LN：x ← x + f(LN(x))</text>`);
  p.push(`<line x1='365' y1='36' x2='365' y2='332' stroke='var(--rule-soft)' stroke-width='1'/>`);

  const nodeY = [60, 148, 236, 324];
  const gaps = [
    { yb: 148, op: 1.0 },
    { yb: 236, op: 0.55 },
    { yb: 324, op: 0.3 },
  ];

  /* ---- Post-LN（左，main x=120）---- */
  nodeY.forEach((y) => p.push(`<circle cx='120' cy='${y}' r='3' fill='var(--ink)'/>`));
  p.push(`<text ${MONO_S} x='132' y='56' fill='var(--ink-faint)'>输出端 · x_L</text>`);
  p.push(`<text ${MONO_S} x='132' y='316' fill='var(--ink-faint)'>输入端 · x_1</text>`);
  gaps.forEach((g, gi) => {
    const ya = g.yb - 88;
    // 前向主路：⊕ 之后过 LN
    p.push(`<line x1='120' y1='${g.yb}' x2='120' y2='${ya}' stroke='#a9a498' stroke-width='1.3'/>`);
    p.push(`<circle cx='120' cy='${g.yb - 20}' r='8' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.3'/>`);
    p.push(`<text ${MONO} x='120' y='${g.yb - 15.5}' fill='var(--ink)' text-anchor='middle'>+</text>`);
    p.push(`<rect x='97' y='${g.yb - 64}' width='46' height='22' rx='3' fill='#e9efe9' stroke='#a9bfa9' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='120' y='${g.yb - 49}' fill='var(--ink)' text-anchor='middle'>LN</text>`);
    p.push(arrowHead(120, g.yb - 76, 'up', '#a9a498'));
    // 分支 f_l
    p.push(`<path d='M 120 ${g.yb} H 205 V ${g.yb - 20} H 128' fill='none' stroke='#a9a498' stroke-width='1.2'/>`);
    p.push(`<rect x='170' y='${g.yb - 44}' width='70' height='24' rx='3' fill='#eeeef2' stroke='#b6b6c4' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='205' y='${g.yb - 28}' fill='var(--ink)' text-anchor='middle'>f_l(x)</text>`);
    // 反向梯度：逐层过 LN，opacity 向输入端衰减
    p.push(`<line class='sig-dashflow' x1='127' y1='${ya + 2}' x2='127' y2='${g.yb - 2}' stroke='var(--seal)' stroke-width='1.6' style='stroke-opacity:${g.op}'/>`);
    p.push(arrowHead(127, g.yb - 2, 'down', 'var(--seal)', g.op));
    if (gi === 0) p.push(`<text ${MONO_S} x='152' y='99' fill='var(--seal)'>×J_LN：每层重投影削弱</text>`);
  });
  p.push(`<line class='sig-dashflow' x1='127' y1='42' x2='127' y2='58' stroke='var(--seal)' stroke-width='1.6'/>`);
  p.push(`<text ${MONO_S} x='119' y='46' fill='var(--seal)' text-anchor='end'>∂L/∂x_L</text>`);
  p.push(`<text ${MONO_S} x='40' y='348' fill='var(--ink-soft)'>末层上界 O(d·√ln d) 与 L 无关；逐层不均 → warmup 必需</text>`);

  /* ---- Pre-LN（右，main x=470）---- */
  nodeY.forEach((y) => p.push(`<circle cx='470' cy='${y}' r='3' fill='var(--ink)'/>`));
  p.push(`<text ${MONO_S} x='482' y='56' fill='var(--ink-faint)'>输出端 · x_L</text>`);
  p.push(`<text ${MONO_S} x='462' y='328' fill='var(--ink-faint)' text-anchor='end'>输入端 · x_1</text>`);
  gaps.forEach((g) => {
    const ya = g.yb - 88;
    // 恒等主路 + 旁挂 LN→f
    p.push(`<line x1='470' y1='${g.yb}' x2='470' y2='${ya}' stroke='#a9a498' stroke-width='1.3'/>`);
    p.push(`<circle cx='470' cy='${g.yb - 70}' r='8' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.3'/>`);
    p.push(`<text ${MONO} x='470' y='${g.yb - 65.5}' fill='var(--ink)' text-anchor='middle'>+</text>`);
    p.push(arrowHead(470, g.yb - 44, 'up', '#a9a498'));
    p.push(`<path d='M 470 ${g.yb} H 560 V ${g.yb - 70} H 478' fill='none' stroke='#a9a498' stroke-width='1.2'/>`);
    // 分支梯度（修正项，弱；画在 chip 之下，读作「穿过」）
    p.push(`<path d='M 478 ${g.yb - 70} H 554 V ${g.yb - 4} H 477' fill='none' stroke='var(--seal)' stroke-width='1.1' stroke-dasharray='3 3' opacity='0.45'/>`);
    p.push(`<rect x='538' y='${g.yb - 30}' width='44' height='20' rx='3' fill='#e9efe9' stroke='#a9bfa9' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='560' y='${g.yb - 16}' fill='var(--ink)' text-anchor='middle'>LN</text>`);
    p.push(`<rect x='525' y='${g.yb - 58}' width='70' height='22' rx='3' fill='#eeeef2' stroke='#b6b6c4' stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='560' y='${g.yb - 43}' fill='var(--ink)' text-anchor='middle'>f_l</text>`);
    p.push(arrowHead(473, g.yb - 4, 'left', 'var(--seal)', 0.45));
  });
  // 恒等梯度高速通路：一条不断的虚线流
  p.push(`<line class='sig-dashflow' x1='463' y1='42' x2='463' y2='330' stroke='var(--seal)' stroke-width='1.8'/>`);
  [148, 236, 324].forEach((yb) => p.push(arrowHead(463, yb - 40, 'down', 'var(--seal)')));
  p.push(`<text ${MONO_S} x='455' y='50' fill='var(--seal)' text-anchor='end'>∂L/∂x_L</text>`);
  p.push(`<text ${MONO_S} x='455' y='66' fill='var(--seal)' text-anchor='end'>恒等项 I 直达（式 8）</text>`);
  p.push(`<text ${MONO_S} x='390' y='348' fill='var(--ink-soft)'>上界 O(d·√(ln d / L))：L=32 小 ≈5.7×；warmup 可省</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '01-foundations/01-formulation',
    afterHeading: /^## 3\. 链式法则到自回归分解/,
    title: '链式法则展开：条件窗口随 t 渐长',
    svg: arFactorization,
  },
  {
    slug: '01-foundations/02-compute-accounting',
    afterHeading: /^## 5\. 合并：C ≈ 6ND/,
    title: 'C = 6ND：2 + 2 + 2 的构成',
    svg: compute6ND,
  },
  {
    slug: '01-foundations/03-scaling-laws',
    afterHeading: /^## 4\. Kaplan 与 Chinchilla 的分歧/,
    title: 'frontier 上的两个真实预算',
    svg: scalingFrontier,
  },
  {
    slug: '02-transformer-anatomy/01-overview',
    afterHeading: /^## 2\. Residual stream/,
    title: 'residual stream：归一化读、加法写',
    svg: residualBus,
  },
  {
    slug: '02-transformer-anatomy/02-tokenization-and-embeddings',
    afterHeading: /^## 2\. BPE：频率驱动的 merge/,
    title: 'BPE 逐轮 merge 与频次账本',
    svg: bpeMerges,
  },
  {
    slug: '02-transformer-anatomy/07-norm-and-residual',
    afterHeading: /^## 5\. Pre-LN vs Post-LN/,
    title: '梯度通路：恒等高速 vs 逐层重投影',
    svg: prePostLN,
  },
];
