// 签名动图注册表：手写 SVG + 机制动画（数据流动/填充/旋转——非入场动画）
// 设计语言与引擎输出一致：纸面板、发丝墨线、mono 标注、朱砂数据点。
// 每个导出函数返回 <svg> 字符串；build-content 按锚点插到对应章节。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='13'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

function svgOpen(w, h, label) {
  return `<svg class="fig-svg sig" viewBox="0 0 ${w} ${H_(h)}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;
}
const H_ = (h) => h;

/* ============ 2.4 §4.4 — KV cache 逐 token 填充 ============ */
export function kvCacheFill() {
  const w = 720, h = 300;
  const parts = [svgOpen(w, h, 'decode 循环中 KV cache 逐位置填充')];

  // 左：decode 时间线 t = 5..8
  parts.push(`<text ${MONO_S} x='24' y='30' fill='var(--ink-faint)'>decode 步</text>`);
  const steps = ['t=5', 't=6', 't=7', 't=8'];
  steps.forEach((s, i) => {
    const y = 56 + i * 40;
    parts.push(`<text ${MONO} x='24' y='${y + 4}' fill='var(--ink)'>${s}</text>`);
    parts.push(`<circle class='sig-step' style='--i:${i}' cx='82' cy='${y}' r='3' fill='var(--seal)'/>`);
    if (i < 3) parts.push(`<line x1='82' y1='${y + 4}' x2='82' y2='${y + 36}' stroke='var(--rule)' stroke-width='1'/>`);
  });

  // 中：cache 网格（K 带与 V 带），位置 1..8
  const gx = 130, gy = 56, cw = 34, ch = 26, gap = 4;
  parts.push(`<text ${MONO_S} x='${gx}' y='30' fill='var(--ink-faint)'>cache（每层 K,V 各一份）</text>`);
  parts.push(`<text ${CJK_S} x='${gx - 8}' y='${gy + 16}' fill='var(--ink-soft)' text-anchor='end'>K</text>`);
  parts.push(`<text ${CJK_S} x='${gx - 8}' y='${gy + 46}' fill='var(--ink-soft)' text-anchor='end'>V</text>`);
  for (let t = 0; t < 8; t++) {
    parts.push(`<text ${MONO_S} x='${gx + t * (cw + gap) + cw / 2}' y='${gy - 8}' fill='var(--ink-faint)' text-anchor='middle'>${t + 1}</text>`);
    for (let band = 0; band < 2; band++) {
      const y = gy + band * (ch + gap);
      // 已缓存旧位置：墨色淡填
      const isOld = t < 4;
      parts.push(
        `<rect x='${gx + t * (cw + gap)}' y='${y}' width='${cw}' height='${ch}' rx='2' ` +
        `fill='${isOld ? 'var(--paper-sunk)' : 'none'}' stroke='var(--rule)' stroke-width='1'/>`
      );
      if (isOld) {
        parts.push(`<text ${MONO_S} x='${gx + t * (cw + gap) + cw / 2}' y='${y + 17}' fill='var(--ink-faint)' text-anchor='middle'>k${t + 1}</text>`);
      }
    }
  }
  // 新写入列（t=5..8 依次亮起）
  for (let i = 0; i < 4; i++) {
    const t = 4 + i;
    for (let band = 0; band < 2; band++) {
      const y = gy + band * (ch + gap);
      parts.push(
        `<rect class='sig-cell' style='--i:${((i * 2 + band) * 0.35).toFixed(2)}' x='${gx + t * (cw + gap)}' y='${y}' width='${cw}' height='${ch}' rx='2' ` +
        `fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>` +
        `<text class='sig-cell-text' style='--i:${((i * 2 + band) * 0.35).toFixed(2)}' ${MONO_S} x='${gx + t * (cw + gap) + cw / 2}' y='${y + 17}' fill='var(--seal)' text-anchor='middle'>${band ? 'v' : 'k'}${t + 1}</text>`
      );
    }
  }
  // 流光：从 decode 步到对应新列
  for (let i = 0; i < 4; i++) {
    const y0 = 56 + i * 40;
    const x1 = gx + (4 + i) * (cw + gap);
    parts.push(
      `<circle class='sig-flow' style='--i:${i}' r='2.6' fill='var(--seal)'>` +
      `<animateMotion dur='2.6s' begin='${(i * 0.65).toFixed(2)}s' repeatCount='indefinite' path='M 88 ${y0} C 110 ${y0}, 110 ${gy + 13}, ${x1} ${gy + 13}'/>` +
      `</circle>`
    );
  }

  // 右：读出——全部 cache 参与当前步 attention
  const rx = 480;
  parts.push(`<text ${MONO_S} x='${rx}' y='30' fill='var(--ink-faint)'>读出：整段 cache × 当前 q_t</text>`);
  parts.push(`<rect x='${rx}' y='52' width='200' height='56' rx='3' fill='none' stroke='var(--rule)'/>`);
  parts.push(`<text ${MONO} x='${rx + 12}' y='74' fill='var(--ink)'>score_s = ⟨q_t, k_s⟩/√d_k</text>`);
  parts.push(`<text ${MONO_S} x='${rx + 12}' y='96' fill='var(--ink-faint)'>s = 1..t（无需重算历史 K/V）</text>`);
  parts.push(`<text ${CJK_S} x='${rx}' y='150' fill='var(--ink-soft)'>append-only 写：每步只加一列；</text>`);
  parts.push(`<text ${CJK_S} x='${rx}' y='170' fill='var(--ink-soft)'>读：每步全量乘——§5 的内存账由此而来</text>`);
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 2.5 §4.2 — RoPE 2D 对偶旋转 ============ */
export function ropeRotate() {
  const w = 720, h = 280;
  const parts = [svgOpen(w, h, 'RoPE：第 i 对维度按位置 m 旋转 m·θ_i')];
  parts.push(`<text ${CJK_S} x='24' y='28' fill='var(--ink-soft)'>同一个位置 m，作用在 d/2 个正交二维平面上——每个平面转 m·θ_i，频率几何递减</text>`);

  const circles = [
    { cx: 150, label: 'θ₁（最快）', dur: '3s', sub: 'i 小 · 短波长 · 局部' },
    { cx: 380, label: 'θ₂', dur: '6s', sub: '' },
    { cx: 610, label: 'θ_d/2（最慢）', dur: '12s', sub: 'i 大 · 长波长 · 远程' },
  ];
  circles.forEach((c, i) => {
    parts.push(`<circle cx='${c.cx}' cy='150' r='78' fill='none' stroke='var(--rule)' stroke-width='1'/>`);
    // 旋转臂（每平面角速度不同）
    parts.push(
      `<g class='sig-rotor' style='--dur:${c.dur}; transform-origin: ${c.cx}px 150px'>` +
      `<line x1='${c.cx}' y1='150' x2='${c.cx + 56}' y2='${150 - 56 * 0.707}' stroke='var(--ink-soft)' stroke-width='1.4'/>` +
      `<circle cx='${c.cx + 56}' cy='${150 - 56 * 0.707}' r='4' fill='var(--seal)'/>` +
      `</g>`
    );
    // 未旋转参照（虚线）
    parts.push(`<line x1='${c.cx}' y1='150' x2='${c.cx + 56}' y2='150' stroke='var(--ink-faint)' stroke-width='1' stroke-dasharray='3 4'/>`);
    parts.push(`<text ${MONO_S} x='${c.cx}' y='252' fill='var(--ink)' text-anchor='middle'>${c.label}</text>`);
    parts.push(`<text ${MONO_S} x='${c.cx}' y='268' fill='var(--ink-faint)' text-anchor='middle'>角度 m·θ_i</text>`);
  });
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 3.1 §1 — 三种 mask 网格 ============ */
export function maskGrids() {
  const w = 720, h = 272;
  const parts = [svgOpen(w, h, '三种 objective 对应的 attention mask 结构')];
  const N = 6, cell = 24, gap = 2;

  const grids = [
    { x: 60, title: 'CLM（下三角）', fill: (i, j) => j <= i },
    { x: 300, title: 'MLM（全量）', fill: () => true },
    { x: 540, title: 'prefix-LM（块）', fill: (i, j) => j <= i || (i < 3 && j < 3) ? (j <= i || j < 3) : j <= i },
  ];
  // prefix-LM：前 3 列对前 3 行双向，其余因果
  grids[2].fill = (i, j) => j <= i || j < 3;

  grids.forEach((g, gi) => {
    parts.push(`<text ${CJK_S} x='${g.x}' y='34' fill='var(--ink)'>${g.title}</text>`);
    parts.push(`<text ${MONO_S} x='${g.x}' y='50' fill='var(--ink-faint)'>行=query t · 列=key s</text>`);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const on = g.fill(i, j);
        const x = g.x + j * (cell + gap);
        const y = 64 + i * (cell + gap);
        parts.push(
          `<rect class='sig-cell' style='--i:${((i + j) * 0.07 + gi * 0.55).toFixed(2)}' x='${x}' y='${y}' width='${cell}' height='${cell}' rx='2' ` +
          `fill='${on ? 'var(--ink)' : 'none'}' ${on ? '' : "stroke='var(--rule)' stroke-dasharray='2 3'"} stroke-width='1'/>`
        );
      }
    }
  });
  parts.push(`<text ${CJK_S} x='60' y='248' fill='var(--ink-faint)'>实格 = 可见；虚格 = −∞ 屏蔽。mask 形状即 objective 的形状。</text>`);
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 4.5 §2 — MoE top-2 路由 ============ */
export function moeRouting() {
  const w = 720, h = 320;
  const parts = [svgOpen(w, h, 'MoE：token 经 router 选 top-2 expert')];
  const tokens = ['x₁', 'x₂', 'x₃'];
  const experts = ['E1', 'E2', 'E3', 'E4'];
  const picks = [[0, 2], [1, 2], [0, 3]]; // 每个 token 的 top-2
  const ty = [70, 160, 250];
  const ey = [50, 115, 180, 245];

  parts.push(`<text ${MONO_S} x='30' y='30' fill='var(--ink-faint)'>tokens</text>`);
  tokens.forEach((t, i) => {
    parts.push(`<rect x='30' y='${ty[i] - 16}' width='52' height='32' rx='3' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
    parts.push(`<text ${MONO} x='56' y='${ty[i] + 5}' fill='var(--ink)' text-anchor='middle'>${t}</text>`);
  });

  // router
  parts.push(`<rect x='170' y='130' width='96' height='60' rx='3' fill='none' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  parts.push(`<text ${MONO} x='218' y='158' fill='var(--ink)' text-anchor='middle'>router</text>`);
  parts.push(`<text ${MONO_S} x='218' y='176' fill='var(--ink-faint)' text-anchor='middle'>softmax(W·x)</text>`);
  tokens.forEach((t, i) => {
    parts.push(`<line x1='82' y1='${ty[i]}' x2='170' y2='160' stroke='var(--rule)' stroke-width='1.1'/>`);
  });

  parts.push(`<text ${MONO_S} x='340' y='30' fill='var(--ink-faint)'>experts（FFN 副本，各自独立参数）</text>`);
  const load = [2, 1, 2, 1]; // 命中数
  experts.forEach((e, j) => {
    parts.push(`<rect x='420' y='${ey[j] - 18}' width='150' height='36' rx='3' fill='var(--paper-raised)' stroke='var(--rule)'/>`);
    parts.push(`<text ${MONO} x='436' y='${ey[j] + 5}' fill='var(--ink)'>${e}</text>`);
    // 负载条
    parts.push(`<rect class='sig-bar' style='--i:${j}' x='${490}' y='${ey[j] - 8}' width='${load[j] * 24}' height='16' rx='2' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1'/>`);
    parts.push(`<text ${MONO_S} x='${580}' y='${ey[j] + 5}' fill='var(--ink-faint)'>load ${load[j]}/3</text>`);
  });

  // 选中路径 + 流光
  tokens.forEach((t, i) => {
    picks[i].forEach((j, k) => {
      const y1 = ty[i], y2 = ey[j];
      const path = `M 266 160 C 340 160, 350 ${y1}, 420 ${ey[j]}`;
      parts.push(`<path d='${path}' fill='none' stroke='var(--ink-soft)' stroke-width='1.1' opacity='0.85'/>`);
      parts.push(
        `<circle class='sig-flow' r='2.6' fill='var(--seal)'>` +
        `<animateMotion dur='${(1.8 + i * 0.3).toFixed(1)}s' begin='${(i * 0.5 + k * 0.25).toFixed(2)}s' repeatCount='indefinite' path='M 82 ${y1} C 130 ${y1}, 140 160, 266 160 S 340 160, 420 ${ey[j]}'/>` +
        `</circle>`
      );
    });
  });
  parts.push(`<text ${CJK_S} x='30' y='306' fill='var(--ink-faint)'>top-2：每 token 只付 2/4 个 expert 的 FLOPs，容量却按全量参数计——解耦即此意</text>`);
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 6.2 §2.1 — CLIP 双塔与相似度矩阵 ============ */
export function clipMatrix() {
  const w = 720, h = 300;
  const parts = [svgOpen(w, h, 'CLIP：双塔编码，batch 相似度矩阵对角即正样本')];

  // 左塔：图像 3 个 patch 块
  parts.push(`<text ${CJK_S} x='30' y='30' fill='var(--ink)'>图像塔（ViT）</text>`);
  for (let i = 0; i < 3; i++) {
    const y = 56 + i * 66;
    parts.push(`<rect x='30' y='${y}' width='56' height='44' rx='3' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
    parts.push(`<circle cx='${48 + (i % 2) * 10}' cy='${y + 16}' r='6' fill='none' stroke='var(--ink-soft)'/>`);
    parts.push(`<line x1='34' y1='${y + 34}' x2='82' y2='${y + 34}' stroke='var(--ink-soft)' stroke-width='1'/>`);
    parts.push(`<text ${MONO_S} x='94' y='${y + 26}' fill='var(--ink-faint)'>u${i + 1}</text>`);
  }
  // 右塔：文本
  parts.push(`<text ${CJK_S} x='196' y='30' fill='var(--ink)'>文本塔（transformer）</text>`);
  for (let i = 0; i < 3; i++) {
    const y = 56 + i * 66;
    parts.push(`<rect x='196' y='${y}' width='86' height='44' rx='3' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
    parts.push(`<line x1='204' y1='${y + 14}' x2='244' y2='${y + 14}' stroke='var(--ink-soft)' stroke-width='1.4'/>`);
    parts.push(`<line x1='204' y1='${y + 24}' x2='238' y2='${y + 24}' stroke='var(--ink-soft)' stroke-width='1.4'/>`);
    parts.push(`<line x1='204' y1='${y + 34}' x2='228' y2='${y + 34}' stroke='var(--ink-soft)' stroke-width='1.4'/>`);
    parts.push(`<text ${MONO_S} x='260' y='${y + 26}' fill='var(--ink-faint)'>v${i + 1}</text>`);
  }

  // 矩阵 3×3
  const mx = 390, my = 56, cell = 52, gap = 6;
  parts.push(`<text ${MONO_S} x='${mx}' y='30' fill='var(--ink-faint)'>S_ij = ⟨u_i, v_j⟩/τ（单位球面）</text>`);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const diag = i === j;
      const x = mx + j * (cell + gap), y = my + i * (cell + gap);
      if (diag) {
        parts.push(
          `<rect class='sig-diag' style='--i:${i}' x='${x}' y='${y}' width='${cell}' height='${cell}' rx='3' ` +
          `fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.4'/>` +
          `<text ${MONO} x='${x + cell / 2}' y='${y + cell / 2 + 5}' fill='var(--seal)' text-anchor='middle'>S${i + 1}${j + 1}</text>`
        );
      } else {
        parts.push(
          `<rect x='${x}' y='${y}' width='${cell}' height='${cell}' rx='3' fill='none' stroke='var(--rule)'/>` +
          `<text ${MONO_S} x='${x + cell / 2}' y='${y + cell / 2 + 4}' fill='var(--ink-faint)' text-anchor='middle'>S${i + 1}${j + 1}</text>`
        );
      }
    }
  }
  // 行/列方向标注
  parts.push(`<text ${CJK_S} x='${mx}' y='${my + 3 * (cell + gap) + 18}' fill='var(--ink-faint)'>行方向：图 i 分类到 B 个文本；列方向对称</text>`);

  // 双塔到矩阵的流线
  for (let i = 0; i < 3; i++) {
    const y = 56 + i * 66 + 22;
    parts.push(
      `<circle class='sig-flow' r='2.4' fill='var(--seal)'>` +
      `<animateMotion dur='2.4s' begin='${(i * 0.8).toFixed(1)}s' repeatCount='indefinite' path='M 112 ${y} C 300 ${y}, 320 ${my + i * (cell + gap) + 26}, ${mx + i * (cell + gap) + 26} ${my + i * (cell + gap) + 26}'/>` +
      `</circle>`
    );
  }
  parts.push('</svg>');
  return parts.join('');
}

/* ============ 注册表：slug + 锚点 → svg ============ */
export const SIGNATURE_FIGURES = [
  {
    slug: '02-transformer-anatomy/04-causality-and-kv-cache',
    afterHeading: /^### 4\.4 有 cache 的 decode/,
    title: 'KV cache 的写与读',
    svg: kvCacheFill,
  },
  {
    slug: '02-transformer-anatomy/05-positional-encoding',
    afterHeading: /^### 4\.2 2D 旋转引理/,
    title: 'RoPE 的二维对偶旋转',
    svg: ropeRotate,
  },
  {
    slug: '03-objectives-and-variants/01-objectives-and-masks',
    afterHeading: /^## 1\. 统一框架/,
    title: 'mask 形状即 objective 形状',
    svg: maskGrids,
  },
  {
    slug: '04-modern-architectures/05-moe',
    afterHeading: /^## 2\. Router 机制谱系/,
    title: 'top-k 路由与负载',
    svg: moeRouting,
  },
  {
    slug: '06-vision-representations/02-clip-and-contrastive',
    afterHeading: /^### 2\.1 相似度矩阵/,
    title: 'batch 相似度矩阵：对角即正样本',
    svg: clipMatrix,
  },
];
