// Part 3 + Part 4 签名动图：objective 变体与现代架构（8 张）
// 规范见 STYLE.md；格式基准 figures-signature.mjs。动画只表现机制。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='13'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

const open = (w, h, label) =>
  `<svg class="fig-svg sig" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;
const aR = (x, y) => `<path d='M${x},${y} l-8,-4.5 v9 z' fill='var(--ink-soft)'/>`;
const aD = (x, y) => `<path d='M${x},${y} l-4.5,-8 h9 z' fill='var(--ink-soft)'/>`;
const aU = (x, y) => `<path d='M${x},${y} l-4.5,8 h9 z' fill='var(--ink-soft)'/>`;
const chip = (x, y, w, h, text, fill, stroke, fs = MONO_S) =>
  `<rect x='${x}' y='${y}' width='${w}' height='${h}' rx='3' fill='${fill}' stroke='${stroke}' stroke-width='1.1'/>` +
  `<text ${fs} x='${x + w / 2}' y='${y + h / 2 + 4}' fill='var(--ink)' text-anchor='middle'>${text}</text>`;
const F = { // 角色配色（与引擎一致）
  attn: ['#f6e9e5', '#cf9d92'], norm: ['#e9efe9', '#a9bfa9'], proj: ['#e9edf3', '#aebcd2'],
  data: ['#f1efe8', '#c6c0af'], cache: ['#f3ecdc', '#cbb98d'], op: ['#eeeef2', '#b6b6c4'],
  plain: ['var(--paper)', '#c9c4b8'], sunk: ['var(--paper-sunk)', 'var(--rule)'],
};

/* ============ 3.2 §7 — encoder-only vs encoder-decoder：可见性与依赖结构 ============ */
function encVsEncDec() {
  const w = 720, h = 336;
  const p = [open(w, h, 'encoder-only 与 encoder-decoder 的结构对照：差异只在 mask 与跨栈依赖')];
  p.push(`<text ${CJK} x='24' y='26' fill='var(--ink)'>encoder-only：双向读，产品是矩阵 H⁽ᴸ⁾</text>`);
  p.push(`<text ${CJK} x='384' y='26' fill='var(--ink)'>encoder-decoder：读 / 写分栈，cross-attention 接通</text>`);
  p.push(`<line x1='360' y1='40' x2='360' y2='316' stroke='var(--rule-soft)' stroke-width='1' stroke-dasharray='3 4'/>`);

  // 左：H 网格（每行都是全序列的函数）
  const gx = 30, gy = 52, c = 17, g = 3;
  p.push(`<text ${MONO_S} x='${gx}' y='${gy - 6}' fill='var(--ink-faint)'>H⁽ᴸ⁾ [T×d]</text>`);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    p.push(`<rect class='sig-cell' style='--i:${((i + j) * 0.08).toFixed(2)}' x='${gx + j * (c + g)}' y='${gy + i * (c + g)}' width='${c}' height='${c}' rx='2' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
  }
  p.push(`<text ${CJK_S} x='${gx + 92}' y='${gy + 14}' fill='var(--ink-soft)'>每行都依赖全部位置</text>`);
  p.push(`<text ${CJK_S} x='${gx + 92}' y='${gy + 32}' fill='var(--ink-faint)'>⇒ 非生成分布（§2）</text>`);
  // 读出接口
  ['[CLS] 头', 'mean pool', 'token 头'].forEach((t, k) => {
    p.push(chip(240, gy + 4 + k * 27, 92, 20, t, ...F.plain));
    p.push(`<line x1='${gx + 4 * (c + g)}' y1='${gy + 34}' x2='240' y2='${gy + 14 + k * 27}' stroke='#a9a498' stroke-width='1'/>`);
  });
  // encoder 栈 + M≡0 小网格
  p.push(chip(30, 158, 156, 34, 'L × self-attn（M ≡ 0）', ...F.attn));
  {
    const mx = 216, my = 148, mc = 10, mg = 2;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      p.push(`<rect class='sig-cell' style='--i:${(0.3 + (i + j) * 0.06).toFixed(2)}' x='${my === 0 ? 0 : mx + j * (mc + mg)}' y='${my + i * (mc + mg)}' width='${mc}' height='${mc}' rx='1.5' fill='var(--ink)'/>`);
    }
    p.push(`<text ${MONO_S} x='${mx}' y='${my + 5 * (mc + mg) + 12}' fill='var(--ink-faint)'>mask：全可见</text>`);
  }
  p.push(`<line x1='108' y1='192' x2='108' y2='${gy + 4 * (c + g)}' stroke='#a9a498' stroke-width='1.1'/>` + aU(108, gy + 4 * (c + g) + 8));
  // tokens
  for (let t = 0; t < 4; t++) {
    p.push(chip(30 + t * 40, 232, 30, 22, `x${t + 1}`, ...F.sunk, MONO));
    p.push(`<line x1='${45 + t * 40}' y1='232' x2='${70 + t * 10}' y2='196' stroke='#a9a498' stroke-width='1'/>`);
  }
  p.push(`<text ${CJK_S} x='30' y='292' fill='var(--ink-faint)'>三类接口全是「读出」——没有一个定义序列上的生成分布</text>`);

  // 右：enc 行
  p.push(chip(384, 62, 64, 24, 'x_{1:S}', ...F.sunk, MONO));
  p.push(chip(470, 58, 110, 34, 'enc 栈（M≡0）', ...F.attn));
  p.push(chip(606, 58, 90, 34, 'H^enc', ...F.cache, MONO));
  p.push(`<text ${MONO_S} x='650' y='106' fill='var(--ink-faint)' text-anchor='middle'>[S×d]，一次前向</text>`);
  p.push(`<line x1='448' y1='70' x2='466' y2='70' stroke='#a9a498' stroke-width='1.1'/>` + aR(470, 70));
  p.push(`<line x1='580' y1='70' x2='602' y2='70' stroke='#a9a498' stroke-width='1.1'/>` + aR(606, 70));
  p.push(`<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='2.2s' begin='0s' repeatCount='indefinite' path='M 448 70 L 692 70'/></circle>`);
  // cross 下引
  p.push(`<path class='sig-dashflow' d='M 651 92 C 651 130, 640 140, 620 158' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<text ${CJK_S} x='384' y='132' fill='var(--ink-soft)'>K^enc,V^enc：算一次、decode 全程复用</text>`);
  // dec 栈
  p.push(chip(384, 168, 64, 24, 'y_{&lt;t}', ...F.sunk, MONO));
  p.push(`<rect x='500' y='150' width='152' height='62' rx='4' fill='none' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  p.push(chip(508, 156, 136, 20, 'causal self-attn', ...F.attn));
  p.push(chip(508, 184, 136, 20, 'cross-attn（无 mask）', ...F.attn));
  p.push(`<line x1='448' y1='180' x2='496' y2='180' stroke='#a9a498' stroke-width='1.1'/>` + aR(500, 180));
  p.push(chip(664, 168, 44, 24, 'y_t', ...F.plain, MONO));
  p.push(`<line x1='652' y1='180' x2='660' y2='180' stroke='#a9a498' stroke-width='1.1'/>` + aR(664, 180));
  p.push(`<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='1.8s' begin='0.4s' repeatCount='indefinite' path='M 448 180 L 700 180'/></circle>`);
  // decode 步序
  for (let k = 0; k < 3; k++) {
    p.push(`<circle class='sig-step' style='--i:${k}' cx='${522 + k * 26}' cy='236' r='3' fill='var(--seal)'/>`);
    p.push(`<text ${MONO_S} x='${522 + k * 26}' y='252' fill='var(--ink-faint)' text-anchor='middle'>y${k + 1}</text>`);
  }
  p.push(`<text ${MONO_S} x='610' y='244' fill='var(--ink-faint)'>逐步生成</text>`);
  // 可见性对照小网格
  {
    const mx = 384, my = 260, mc = 9, mg = 2;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      const on = j <= i;
      p.push(`<rect class='sig-cell' style='--i:${(0.4 + (i + j) * 0.05).toFixed(2)}' x='${mx + j * (mc + mg)}' y='${my + i * (mc + mg)}' width='${mc}' height='${mc}' rx='1.5' fill='${on ? 'var(--ink)' : 'none'}' ${on ? '' : "stroke='var(--rule)' stroke-dasharray='2 2'"} stroke-width='1'/>`);
    }
    p.push(`<text ${MONO_S} x='${mx}' y='${my + 5 * (mc + mg) + 12}' fill='var(--ink-faint)'>self：s ≤ t</text>`);
  }
  {
    const mx = 520, my = 262, mc = 9, mg = 2;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 6; j++) {
      p.push(`<rect class='sig-cell' style='--i:${(0.5 + (i + j) * 0.05).toFixed(2)}' x='${mx + j * (mc + mg)}' y='${my + i * (mc + mg)}' width='${mc}' height='${mc}' rx='1.5' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1'/>`);
    }
    p.push(`<text ${MONO_S} x='${mx}' y='${my + 3 * (mc + mg) + 13}' fill='var(--ink-faint)'>cross：全部 S 恒可见</text>`);
  }
  p.push(`<text ${CJK_S} x='704' y='327' fill='var(--ink-faint)' text-anchor='end'>双向性只加在已存在的 x 上</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 3.3 §6 — induction head：prefix match + copy 两步电路 ============ */
function induction() {
  const w = 720, h = 316;
  const p = [open(w, h, 'induction head 两步电路：[A B]…[A] → B 的前缀匹配与复制')];
  // 序列行
  const tok = (cx, label, sub, seal) => {
    p.push(`<rect x='${cx - 25}' y='52' width='50' height='26' rx='3' fill='${seal ? 'var(--seal-wash)' : 'var(--paper-sunk)'}' stroke='${seal ? 'var(--seal)' : 'var(--rule)'}' stroke-width='1.1'/>`);
    p.push(`<text ${MONO} x='${cx}' y='69' fill='${seal ? 'var(--seal-deep)' : 'var(--ink)'}' text-anchor='middle'>${label}</text>`);
    p.push(`<text ${MONO_S} x='${cx}' y='46' fill='var(--ink-faint)' text-anchor='middle'>${sub}</text>`);
  };
  tok(121, '[A]', 'i', false);
  tok(201, '[B]', 'i+1', false);
  p.push(`<text ${MONO} x='300' y='70' fill='var(--ink-faint)' text-anchor='middle'>· · ·</text>`);
  tok(401, '[A]', 't（再看 A）', true);
  tok(481, '[?]', 't+1', false);
  // ① 第 1 层：previous-token copy（上方虚线弧）
  p.push(`<path class='sig-dashflow' d='M 121 52 C 150 22, 172 22, 197 52' fill='none' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  p.push(`<text ${CJK_S} x='161' y='22' fill='var(--ink-soft)' text-anchor='middle'>①</text>`);
  // ② 第 2 层：prefix match（下方弧，命中加粗脉冲；旁路细弧）
  p.push(`<path d='M 395 82 C 350 122, 280 122, 127 82' fill='none' stroke='var(--rule)' stroke-width='1'/>`);
  p.push(`<path class='sig-diag' d='M 401 82 C 370 148, 250 148, 205 82' fill='none' stroke='var(--seal)' stroke-width='2'/>`);
  p.push(`<text ${CJK_S} x='320' y='126' fill='var(--seal-deep)'>②</text>`);
  // ③ copy 弧：i+1 的 value 搬到 t
  p.push(`<path class='sig-dashflow' d='M 226 86 C 310 220, 430 220, 476 86' fill='none' stroke='var(--seal)' stroke-width='1.3'/>`);
  p.push(`<text ${CJK_S} x='351' y='200' fill='var(--ink-soft)' text-anchor='middle'>③</text>`);
  // 右：经 W_U 的 logit 抬升
  p.push(`<text ${MONO_S} x='560' y='36' fill='var(--ink-faint)'>经 W_U 读出（位置 t）</text>`);
  const bars = [['A', 19], ['B', 70], ['the', 12], ['store', 19]];
  bars.forEach(([t, bh], k) => {
    const bx = 566 + k * 34, hot = t === 'B';
    p.push(`<rect class='sig-bar' style='--i:${k}' x='${bx}' y='${180 - bh}' width='20' height='${bh}' rx='2' fill='${hot ? 'var(--seal)' : 'var(--paper-sunk)'}' stroke='${hot ? 'var(--seal-deep)' : 'var(--rule)'}' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='${bx + 10}' y='194' fill='${hot ? 'var(--seal-deep)' : 'var(--ink-faint)'}' text-anchor='middle'>${t}</text>`);
  });
  p.push(`<text ${CJK_S} x='560' y='222' fill='var(--seal-deep)'>logit(B) 被抬升 → 预测 B</text>`);
  p.push(`<line x1='540' y1='30' x2='540' y2='230' stroke='var(--rule-soft)' stroke-width='1' stroke-dasharray='3 4'/>`);
  // 底部图例（编号对应弧上标记）
  p.push(`<text ${CJK_S} x='40' y='252' fill='var(--ink-soft)'>① previous-token head（第 1 层）：A⁽¹⁾_{j+1,j} ≈ 1 —— i+1 的 stream 同时含「我是 B」与「前驱是 A」</text>`);
  p.push(`<text ${CJK_S} x='40' y='274' fill='var(--ink-soft)'>② induction head（第 2 层）：s_{t,i+1} ∝ E_{x_t}ᵀ (W_Q⁽²⁾)ᵀW_K⁽²⁾W_OV⁽¹⁾ E_{x_i} —— 跨两层的「token 相等」检测（prefix match，式 7）</text>`);
  p.push(`<text ${CJK_S} x='40' y='296' fill='var(--ink-faint)'>③ copy：value(i+1) = B 的表征写入 t；induction score = attention mass 落在偏移对角线的比例，可测（TransformerLens）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.1 §4.3 — MLA W_UK absorption：score 直接打在 latent 上 ============ */
function mlaAbsorb() {
  const w = 720, h = 316;
  const p = [open(w, h, 'MLA 吸收：decode 时上投影被离线吸进 query，score 直接打在 cached latent 上')];
  // 展开形
  p.push(`<text ${CJK} x='28' y='28' fill='var(--ink)'>展开形（训练 / prefill）：逐头物化 k^C</text>`);
  p.push(chip(30, 62, 110, 30, 'c_s^KV [512]', ...F.cache, MONO));
  p.push(chip(180, 62, 84, 30, 'W_h^UK', ...F.proj, MONO));
  p.push(chip(306, 62, 150, 30, 'k_s,h^C [128] × H 头', ...F.plain, MONO_S));
  p.push(chip(180, 108, 110, 26, 'q_t,h^C [128]', ...F.plain, MONO));
  p.push(chip(500, 78, 120, 34, '⟨q^C, k^C⟩', ...F.op, MONO));
  p.push(`<line x1='140' y1='77' x2='176' y2='77' stroke='#a9a498' stroke-width='1.1'/>` + aR(180, 77));
  p.push(`<line x1='264' y1='77' x2='302' y2='77' stroke='#a9a498' stroke-width='1.1'/>` + aR(306, 77));
  p.push(`<line x1='456' y1='88' x2='496' y2='92' stroke='#a9a498' stroke-width='1.1'/>` + aR(500, 89));
  p.push(`<line x1='290' y1='118' x2='496' y2='102' stroke='#a9a498' stroke-width='1.1'/>` + aR(500, 102));
  p.push(`<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='2.4s' begin='0s' repeatCount='indefinite' path='M 140 77 C 300 77, 420 88, 496 92'/></circle>`);
  // 同一份 cache 竖线
  p.push(`<line x1='85' y1='92' x2='85' y2='196' stroke='var(--rule)' stroke-width='1' stroke-dasharray='3 4'/>`);
  p.push(`<text ${MONO_S} x='97' y='150' fill='var(--ink-faint)'>同一份 cache（576 元素 / token / 层）</text>`);
  // 吸收形
  p.push(`<text ${CJK} x='28' y='186' fill='var(--ink)'>吸收形（decode）：score 直接打在 cached latent 上</text>`);
  p.push(chip(30, 196, 110, 30, 'c_s^KV [512]', ...F.cache, MONO));
  p.push(chip(180, 200, 100, 26, 'c_t^Q [1536]', ...F.plain, MONO_S));
  p.push(`<rect x='322' y='190' width='168' height='44' rx='3' fill='${F.proj[0]}' stroke='${F.proj[1]}' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='406' y='208' fill='var(--ink)' text-anchor='middle'>W̃_h = (W_h^UQ)ᵀW_h^UK</text>`);
  p.push(`<text ${MONO_S} x='406' y='227' fill='var(--ink-faint)' text-anchor='middle'>离线预计算一次</text>`);
  p.push(chip(530, 200, 84, 26, 'q̃ [512]', ...F.plain, MONO));
  p.push(chip(628, 198, 84, 30, '⟨q̃, c_s^KV⟩', ...F.op, MONO_S));
  p.push(`<line x1='280' y1='211' x2='318' y2='211' stroke='#a9a498' stroke-width='1.1'/>` + aR(322, 211));
  p.push(`<line x1='490' y1='211' x2='526' y2='211' stroke='#a9a498' stroke-width='1.1'/>` + aR(530, 211));
  p.push(`<line x1='614' y1='213' x2='624' y2='213' stroke='#a9a498' stroke-width='1.1'/>` + aR(628, 213));
  p.push(`<path class='sig-dashflow' d='M 140 214 C 380 250, 560 236, 624 220' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  p.push(`<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='2s' begin='0.3s' repeatCount='indefinite' path='M 280 211 L 624 213'/></circle>`);
  // 账
  p.push(`<text ${MONO_S} x='28' y='270' fill='var(--ink-soft)'>decode 每 token 每层 KV 读取：2H·d_h + d_R = 32832 → d_c + d_R = 576 元素（57×，式 12 与 §4.4 表）</text>`);
  p.push(`<text ${MONO_S} x='28' y='290' fill='var(--ink-faint)'>代价：score 内维 192 → 576（FLOPs ×3 换读取 ÷57，memory-bound 下净赚）；rank(W̃_h) ≤ 128：透镜不变</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.2 §1 — 三种可见集 pattern 网格对照 ============ */
function sparseGrids() {
  const w = 720, h = 300;
  const p = [open(w, h, 'SWA、块选择与 NSA 合成三种稀疏可见集的对照')];
  const N = 10, cell = 13, gap = 2;
  const grids = [
    { x: 46, title: 'SWA：最近 w = 3', sub: '静态 · token 级（式 1）' },
    { x: 286, title: '块选择：top-k 块', sub: '动态 · 块级（式 15）' },
    { x: 526, title: 'NSA：cmp+slc+win', sub: '三分支合成（式 8–10）' },
  ];
  // 块（块长 2）的选中集（示意内容依赖）
  const selMoBA = { 1: [0], 2: [0], 3: [1], 4: [2] };
  const selNSA = { 2: [0], 3: [1], 4: [0] };
  const cellRect = (x, y, fill, stroke, dash, i, j, gi) =>
    `<rect class='sig-cell' style='--i:${(gi * 0.6 + (i + j) * 0.05).toFixed(2)}' x='${x}' y='${y}' width='${cell}' height='${cell}' rx='1.5' fill='${fill}' ${stroke ? `stroke='${stroke}' stroke-width='1'` : ''} ${dash ? "stroke-dasharray='2 2'" : ''}/>`;
  grids.forEach((gd, gi) => {
    p.push(`<text ${CJK_S} x='${gd.x}' y='34' fill='var(--ink)'>${gd.title}</text>`);
    p.push(`<text ${MONO_S} x='${gd.x}' y='50' fill='var(--ink-faint)'>${gd.sub}</text>`);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = gd.x + j * (cell + gap), y = 62 + i * (cell + gap);
      let fill = 'none', stroke = 'var(--rule)', dash = true;
      if (gi === 0) { // SWA：带宽 3
        if (j <= i && j >= i - 2) { fill = 'var(--ink)'; stroke = null; dash = false; }
      } else if (gi === 1) { // 块选择：对角块 + 选中块
        const bi = Math.floor(i / 2), bj = Math.floor(j / 2);
        if (j <= i && (bj === bi || (selMoBA[bi] || []).includes(bj))) {
          if (bj === bi) { fill = 'var(--ink)'; stroke = null; } else { fill = 'var(--seal-wash)'; stroke = 'var(--seal)'; }
          dash = false;
        }
      } else { // NSA 合成
        if (j === 0 && i >= 2) { fill = '#f3ecdc'; stroke = '#cbb98d'; dash = false; } // 压缩 key：全员粗在场
        else if (j <= i && j >= i - 1) { fill = 'var(--ink)'; stroke = null; dash = false; } // sliding
        else {
          const bi = Math.floor(i / 2), bj = Math.floor(j / 2);
          if (j <= i && (selNSA[bi] || []).includes(bj)) { fill = 'var(--seal-wash)'; stroke = 'var(--seal)'; dash = false; }
        }
      }
      p.push(cellRect(x, y, fill, stroke, dash, i, j, gi));
    }
  });
  // 图例（两行）
  const lg = [
    ['var(--ink)', null, '静态可见（滑窗 / 对角块）', 46, 238],
    ['var(--seal-wash)', 'var(--seal)', '动态选中（top-k 块）', 260, 238],
    ['#f3ecdc', '#cbb98d', '压缩 key（NSA cmp：全局稀释在场）', 46, 260],
    ['none', 'var(--rule)', '屏蔽（−∞）', 350, 258],
  ];
  lg.forEach(([f, s, t, x, y]) => {
    p.push(`<rect x='${x}' y='${y}' width='12' height='12' rx='2' fill='${f}' ${s ? `stroke='${s}' stroke-width='1'` : ''} ${f === 'none' ? "stroke-dasharray='2 2'" : ''}/>`);
    p.push(`<text ${CJK_S} x='${x + 18}' y='${y + 10}' fill='var(--ink-soft)'>${t}</text>`);
  });
  p.push(`<text ${MONO_S} x='46' y='290' fill='var(--ink-faint)'>行 = query、列 = key；score / softmax / 投影全部不动——只改「谁参与归一化」：score 对数 T(T+1)/2 → Σ|V(i)|</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.3 §3 — linear attention ≡ 矩阵状态 RNN ============ */
function linearState() {
  const w = 720, h = 300;
  const p = [open(w, h, 'linear attention 的状态递推：S 逐 token 秩一累加，q 读出')];
  // 写入路径
  p.push(chip(30, 56, 72, 26, 'k_t [d_k]', ...F.plain, MONO_S));
  p.push(chip(30, 96, 72, 26, 'v_t [d_v]', ...F.plain, MONO_S));
  p.push(chip(130, 70, 44, 40, '⊗', ...F.op, MONO));
  p.push(`<line x1='102' y1='69' x2='126' y2='82' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<line x1='102' y1='109' x2='126' y2='98' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<text ${CJK_S} x='66' y='152' fill='var(--ink-soft)'>写入：S ← S + v_t k_tᵀ</text>`);
  // S 网格：三波叠加 = 逐 token 累加
  const gx = 210, gy = 44, cell = 20;
  p.push(`<text ${MONO_S} x='${gx}' y='32' fill='var(--ink-faint)'>S_t ∈ R^{d_v × d_k}（秩一累加，定长常驻）</text>`);
  for (let r = 0; r < 8; r++) for (let cc = 0; cc < 8; cc++) {
    const wave = (r * 3 + cc * 5) % 3;
    p.push(`<rect x='${gx + cc * cell}' y='${gy + r * cell}' width='${cell - 2}' height='${cell - 2}' rx='2' fill='var(--paper)' stroke='var(--rule)' stroke-width='1'/>`);
    p.push(`<rect class='sig-cell' style='--i:${(0.2 + wave * 0.85).toFixed(2)}' x='${gx + cc * cell}' y='${gy + r * cell}' width='${cell - 2}' height='${cell - 2}' rx='2' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1'/>`);
  }
  p.push(`<path class='sig-dashflow' d='M 174 90 C 190 90, 196 88, 206 86' fill='none' stroke='var(--seal)' stroke-width='1.2'/>`);
  // 步序点
  for (let k = 0; k < 3; k++) {
    p.push(`<circle class='sig-step' style='--i:${k}' cx='392' cy='${64 + k * 30}' r='3' fill='var(--seal)'/>`);
    p.push(`<text ${MONO_S} x='404' y='${68 + k * 30}' fill='var(--ink-faint)'>t=${k + 1}</text>`);
  }
  // 读出
  p.push(chip(226, 236, 76, 26, 'φ(q_t)', ...F.proj, MONO_S));
  p.push(`<line x1='264' y1='236' x2='264' y2='210' stroke='#a9a498' stroke-width='1.1'/>` + aU(264, 210));
  p.push(chip(330, 236, 52, 26, 'o_t', ...F.plain, MONO));
  p.push(`<line x1='302' y1='249' x2='326' y2='249' stroke='#a9a498' stroke-width='1.1'/>` + aR(330, 249));
  p.push(`<text ${MONO_S} x='392' y='253' fill='var(--ink-soft)'>= S_t·φ(q_t) / z_tᵀφ(q_t)（z_t 同理累加）</text>`);
  p.push(`<text ${CJK_S} x='210' y='284' fill='var(--ink-faint)'>训练走并行前缀和（式 3）/ chunkwise（§8）；decode 走此递推——两形态逐位置相等</text>`);
  // 右：decode 账（log2 字节条）
  p.push(`<text ${CJK_S} x='470' y='56' fill='var(--ink)'>decode 账（单头，t = 131072）</text>`);
  const bars = [
    ['S 状态 32 KiB', Math.log2(32 * 1024), 'var(--seal-wash)', 'var(--seal)'],
    ['KV cache 64 MiB', Math.log2(64 * 1024 * 1024), 'var(--paper-sunk)', 'var(--rule)'],
  ];
  bars.forEach(([t, v, f, s], k) => {
    const bw = (v / 26) * 220;
    p.push(`<text ${MONO_S} x='470' y='${74 + k * 40}' fill='var(--ink-soft)'>${t}</text>`);
    p.push(`<rect class='sig-bar' style='--i:${k * 2}' x='470' y='${80 + k * 40}' width='${bw.toFixed(1)}' height='18' rx='2' fill='${f}' stroke='${s}' stroke-width='1'/>`);
  });
  p.push(`<text ${MONO_S} x='470' y='168' fill='var(--ink-faint)'>条长 = log₂ 字节；比值 = 2t/d = 2048×</text>`);
  p.push(`<text ${CJK_S} x='470' y='196' fill='var(--ink-soft)'>同一枚硬币的另一面：任意长前缀压进定数</text>`);
  p.push(`<text ${CJK_S} x='470' y='216' fill='var(--ink-faint)'>⇒ 精确联想 ≤ d_k 对（§10 rank 界）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.4 §1 — hybrid 层栈与两笔斜率 ============ */
function hybridStack() {
  const w = 720, h = 300;
  const p = [open(w, h, 'hybrid 交错层栈与 decode 内存：固定状态加小斜率')];
  p.push(`<text ${CJK_S} x='30' y='30' fill='var(--ink)'>层栈（3:1 示意，12 层）</text>`);
  // 左：层栈
  for (let l = 0; l < 12; l++) {
    const full = (l + 1) % 4 === 0;
    const y = 42 + l * 19;
    p.push(`<rect class='sig-cell' style='--i:${(l * 0.12).toFixed(2)}' x='30' y='${y}' width='88' height='15' rx='2' fill='${full ? F.attn[0] : F.op[0]}' stroke='${full ? F.attn[1] : F.op[1]}' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='74' y='${y + 11}' fill='var(--ink)' text-anchor='middle'>${full ? 'full attn' : 'linear'}</text>`);
  }
  p.push(`<text ${MONO_S} x='130' y='${42 + 8 * 19 + 11}' fill='var(--ink-faint)'>linear 层：S = 1 MiB / 层，与 T 无关</text>`);
  p.push(`<text ${MONO_S} x='130' y='${42 + 3 * 19 + 11}' fill='var(--ink-faint)'>full 层：KV = 2 KiB / token / 层，随 T 线性</text>`);
  // 右：内存曲线（Qwen3-Next 口径，T = 262144）
  const x0 = 340, y0 = 246, x1 = 690, yTop = 46; // y: 0..24 GiB → 200px
  const Y = (v) => y0 - (v / 24) * (y0 - yTop);
  p.push(`<line x1='${x0}' y1='${y0}' x2='${x1}' y2='${y0}' stroke='var(--ink-soft)' stroke-width='1.1'/>`);
  p.push(`<line x1='${x0}' y1='${y0}' x2='${x0}' y2='${yTop}' stroke='var(--ink-soft)' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='${x0 - 8}' y='${yTop + 4}' fill='var(--ink-faint)' text-anchor='end'>24 GiB</text>`);
  p.push(`<text ${MONO_S} x='${x0 - 8}' y='${y0 + 4}' fill='var(--ink-faint)' text-anchor='end'>0</text>`);
  p.push(`<text ${MONO_S} x='${x1}' y='${y0 + 16}' fill='var(--ink-faint)' text-anchor='end'>T = 262144</text>`);
  p.push(`<text ${MONO_S} x='${x0}' y='${y0 + 16}' fill='var(--ink-faint)'>0</text>`);
  // 全 full 反事实（虚线）
  p.push(`<line x1='${x0}' y1='${y0}' x2='${x1}' y2='${Y(24)}' stroke='var(--ink-faint)' stroke-width='1.3' stroke-dasharray='5 4'/>`);
  p.push(`<circle class='sig-step' style='--i:1' cx='${x1}' cy='${Y(24)}' r='3' fill='var(--ink-soft)'/>`);
  p.push(`<text ${MONO_S} x='${x1 - 6}' y='${Y(24) + 16}' fill='var(--ink-soft)' text-anchor='end'>48 层全 full：24 GiB</text>`);
  // hybrid（实线）
  p.push(`<line x1='${x0}' y1='${Y(0.036)}' x2='${x1}' y2='${Y(6)}' stroke='var(--seal)' stroke-width='2'/>`);
  p.push(`<circle class='sig-step' style='--i:2' cx='${x1}' cy='${Y(6)}' r='3.4' fill='var(--seal)'/>`);
  p.push(`<text ${MONO_S} x='${x1 - 10}' y='${Y(6) - 10}' fill='var(--seal-deep)' text-anchor='end'>3:1 hybrid：36 MiB + 6 GiB</text>`);
  // 固定项（贴轴平线）
  p.push(`<line x1='${x0}' y1='${Y(0.036)}' x2='${x1}' y2='${Y(0.036)}' stroke='#cbb98d' stroke-width='1.2' stroke-dasharray='3 4'/>`);
  p.push(`<text ${MONO_S} x='${x0 + 150}' y='${Y(0.036) - 8}' fill='var(--ink-faint)'>linear 层固定状态 36 MiB（贴轴）</text>`);
  p.push(`<text ${CJK_S} x='30' y='284' fill='var(--ink-soft)'>交叉点 ≈ 1.5×10³ token（36 MiB ÷ 24 KiB）：hybrid 省的是斜率不是截距——窗越长兑现越足；检索职责集中在 1/(r+1) 的 full 层</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.6 §2 — RoPE 缩放族：η_j 一条曲线的三种参数化 ============ */
function yarnCurves() {
  const w = 720, h = 300;
  const p = [open(w, h, 'PI、NTK-aware、YaRN 的逐通道位置压缩率曲线对照（DeepSeek-V3 口径）')];
  const d = 64, base = 1e4, W = 4096, s = 40, A = 1, B = 32, NC = 32;
  const ntk = (j) => Math.pow(s, (2 * j) / (d - 2));
  const lam = (j) => 2 * Math.PI * Math.pow(base, (2 * j) / d);
  const yarn = (j) => { const r = W / lam(j); const g = r < A ? 0 : r > B ? 1 : (r - A) / (B - A); return s / (1 + g * (s - 1)); };
  const x0 = 64, y0 = 240, pw = 560, ph = 180;
  const px = (j) => x0 + (j / (NC - 1)) * pw;
  const py = (e) => y0 - ((e - 1) / (s - 1)) * ph;
  // 轴与刻度
  p.push(`<line x1='${x0}' y1='${y0}' x2='${x0 + pw}' y2='${y0}' stroke='var(--ink-soft)' stroke-width='1.1'/>`);
  p.push(`<line x1='${x0}' y1='${y0}' x2='${x0}' y2='${y0 - ph - 8}' stroke='var(--ink-soft)' stroke-width='1.1'/>`);
  [1, 10, 20, 30, 40].forEach((e) => {
    p.push(`<line x1='${x0}' y1='${py(e)}' x2='${x0 + pw}' y2='${py(e)}' stroke='var(--rule-soft)' stroke-width='1'/>`);
    p.push(`<text ${MONO_S} x='${x0 - 8}' y='${py(e) + 4}' fill='var(--ink-faint)' text-anchor='end'>${e}</text>`);
  });
  p.push(`<text ${MONO_S} x='${x0 - 40}' y='${y0 - ph - 2}' fill='var(--ink-faint)'>η_j</text>`);
  p.push(`<text ${MONO_S} x='${x0 + pw / 2}' y='${y0 + 34}' fill='var(--ink-soft)' text-anchor='middle'>通道 j（快 → 慢）· DeepSeek-V3：d=64，W=4096，s=40，α=1，β=32</text>`);
  p.push(`<text ${MONO_S} x='${x0 + 8}' y='${y0 + 16}' fill='var(--ink-faint)'>η ≈ 1：保局部相位分辨</text>`);
  p.push(`<text ${MONO_S} x='${x0 + pw - 8}' y='${y0 - ph - 14}' fill='var(--ink-faint)' text-anchor='end'>η ≈ s：压回训练域</text>`);
  // PI：常数虚线
  p.push(`<line x1='${x0}' y1='${py(s)}' x2='${x0 + pw}' y2='${py(s)}' stroke='var(--ink-faint)' stroke-width='1.3' stroke-dasharray='5 4'/>`);
  // NTK：指数 ramp
  let pts = [];
  for (let j = 0; j < NC; j++) pts.push(`${px(j).toFixed(1)},${py(ntk(j)).toFixed(1)}`);
  p.push(`<polyline points='${pts.join(' ')}' fill='none' stroke='var(--ink-soft)' stroke-width='1.4'/>`);
  // YaRN：三段
  pts = [];
  for (let j = 0; j < NC; j++) pts.push(`${px(j).toFixed(1)},${py(yarn(j)).toFixed(1)}`);
  p.push(`<polyline points='${pts.join(' ')}' fill='none' stroke='var(--seal)' stroke-width='2'/>`);
  [0, 10, 11, 16, 22, 23, 31].forEach((j, k) => {
    p.push(`<circle class='sig-step' style='--i:${k}' cx='${px(j)}' cy='${py(yarn(j))}' r='2.8' fill='var(--seal)'/>`);
  });
  // 分段计数注记
  p.push(`<text ${MONO_S} x='${px(5)}' y='${py(1) - 12}' fill='var(--seal-deep)' text-anchor='middle'>j ≤ 10：η=1（11 通道不动）</text>`);
  p.push(`<text ${MONO_S} x='${px(13)}' y='150' fill='var(--seal-deep)' text-anchor='middle'>11..22：ramp（12 通道）</text>`);
  p.push(`<line x1='${px(13) + 40}' y1='156' x2='${px(17)}' y2='${py(yarn(17)) - 6}' stroke='var(--seal)' stroke-width='0.8'/>`);
  p.push(`<text ${MONO_S} x='${px(27)}' y='${py(40) + 18}' fill='var(--seal-deep)' text-anchor='middle'>j ≥ 23：η = 40（9 通道全压）</text>`);
  // 图例
  const leg = [
    ['var(--ink-faint)', '5 4', 'PI：η_j = s'],
    ['var(--ink-soft)', null, 'NTK：η_j = s^{2j/(d−2)}'],
    ['var(--seal)', null, 'YaRN：按圈数分段（式 12–14）'],
  ];
  leg.forEach(([c, dash, t], k) => {
    const x = 80 + k * 190;
    p.push(`<line x1='${x}' y1='22' x2='${x + 26}' y2='22' stroke='${c}' stroke-width='2' ${dash ? `stroke-dasharray='${dash}'` : ''}/>`);
    p.push(`<text ${MONO_S} x='${x + 32}' y='26' fill='var(--ink-soft)'>${t}</text>`);
  });
  p.push('</svg>');
  return p.join('');
}

/* ============ 4.7 §2 — logit lens：同一 W_U 逐层读出 ============ */
function logitLens() {
  const w = 720, h = 310;
  const p = [open(w, h, 'logit lens：中间层 stream 经同一输出头投影，分布逐层成形')];
  p.push(`<text ${MONO_S} x='40' y='28' fill='var(--ink-soft)'>p̃_ℓ = softmax(W_U · x⁽ℓ⁾)：同一个输出头读每一层（GPT-2 small，12 层采样 5 档）</text>`);
  p.push(`<text ${CJK_S} x='40' y='48' fill='var(--ink-faint)'>每簇四条（左 → 右）：Mary（目标，朱）· John · the · store</text>`);
  const layers = [0, 3, 6, 9, 12];
  const tokens = ['Mary', 'John', 'the', 'store'];
  // 每层分布（示意：IOI 中目标名 Mary 逐层成形）
  const probs = [
    [0.05, 0.09, 0.30, 0.20],
    [0.12, 0.20, 0.24, 0.18],
    [0.24, 0.22, 0.16, 0.14],
    [0.45, 0.15, 0.10, 0.08],
    [0.72, 0.07, 0.06, 0.05],
  ];
  const baseY = 196, hMax = 130;
  layers.forEach((L, li) => {
    const cx = 100 + li * 125;
    // W_U 扇形箭头
    p.push(`<path class='sig-dashflow' d='M ${cx} 240 C ${cx} 226, ${cx} 216, ${cx} 202' fill='none' stroke='var(--ink-faint)' stroke-width='1'/>`);
    if (li === 2) p.push(`<text ${MONO_S} x='${cx + 8}' y='224' fill='var(--ink-faint)'>W_U</text>`);
    probs[li].forEach((pv, ti) => {
      const bh = Math.max(2, pv * hMax);
      const bx = cx - 46 + ti * 24;
      const hot = ti === 0;
      p.push(`<rect class='sig-bar' style='--i:${(li * 3 + ti * 0.4).toFixed(1)}' x='${bx}' y='${baseY - bh}' width='18' height='${bh.toFixed(1)}' rx='2' fill='${hot ? 'var(--seal)' : 'var(--paper-sunk)'}' stroke='${hot ? 'var(--seal-deep)' : 'var(--rule)'}' stroke-width='1'/>`);
      if (li === 4 && ti === 0) p.push(`<text ${MONO_S} x='${bx + 9}' y='${baseY + 13}' fill='var(--seal-deep)' text-anchor='middle'>Mary</text>`);
    });
    // stream 节点
    p.push(`<circle cx='${cx}' cy='252' r='4' fill='var(--paper)' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
    p.push(`<text ${MONO_S} x='${cx}' y='270' fill='var(--ink-soft)' text-anchor='middle'>ℓ=${L}${L === 12 ? '（终层）' : ''}</text>`);
    if (li < 4) {
      p.push(`<line x1='${cx + 8}' y1='252' x2='${cx + 117}' y2='252' stroke='#a9a498' stroke-width='1.1'/>`);
      p.push(`<text ${MONO_S} x='${cx + 62}' y='246' fill='var(--ink-faint)' text-anchor='middle'>+a,+f</text>`);
    }
  });
  p.push(`<text ${CJK_S} x='40' y='292' fill='var(--ink-faint)'>读数语义 = 「若后续层全为恒等，此刻输出是什么」（非中途预测）；排序可信、绝对值存疑（中间层不在 W_U 训练分布上）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    slug: '03-objectives-and-variants/02-encoder-architectures',
    afterHeading: /^## 7\. Encoder-decoder/,
    title: '两栈分工：双向读 × 因果写',
    svg: encVsEncDec,
  },
  {
    slug: '03-objectives-and-variants/03-decoder-only-and-icl',
    afterHeading: /^## 6\. 解释层 \(b\)/,
    title: 'induction head：prefix match + copy',
    svg: induction,
  },
  {
    slug: '04-modern-architectures/01-kv-compression-and-exact-attention',
    afterHeading: /^### 4\.3 W_UK absorption/,
    title: 'W_UK 吸收：score 打在 latent 上',
    svg: mlaAbsorb,
  },
  {
    slug: '04-modern-architectures/02-sparse-attention',
    afterHeading: /^## 1\. 账的位置/,
    title: '三种可见集：SWA / 块选择 / NSA',
    svg: sparseGrids,
  },
  {
    slug: '04-modern-architectures/03-linear-attention-and-ssm',
    afterHeading: /^## 3\. Recurrent 形式/,
    title: 'S_t = S_{t-1} + v_t k_tᵀ：状态累加与读出',
    svg: linearState,
  },
  {
    slug: '04-modern-architectures/04-hybrid-architectures',
    afterHeading: /^## 1\. 互补性论证/,
    title: '交错层栈与两笔斜率',
    svg: hybridStack,
  },
  {
    slug: '04-modern-architectures/06-long-context',
    afterHeading: /^## 2\. RoPE 缩放族/,
    title: 'η_j 曲线的三种参数化：PI / NTK / YaRN',
    svg: yarnCurves,
  },
  {
    slug: '04-modern-architectures/07-interpretability-primer',
    afterHeading: /^## 2\. Logit lens/,
    title: 'logit lens：逐层读出，分布成形',
    svg: logitLens,
  },
];
