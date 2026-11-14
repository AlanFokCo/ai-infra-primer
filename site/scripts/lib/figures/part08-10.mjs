// Part 8–10 签名动图：多模态接入 / 统一理解生成 / 视频 VAE 与时空 DiT / 音频表征 / CTC / 全双工延迟
// 规范见 STYLE.md：机制动画（数据流、填充、脉冲），无入场动画；viewBox 宽 720。

const MONO = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='12'";
const MONO_S = "font-family='IBM Plex Mono, ui-monospace, monospace' font-size='10.5'";
const CJK = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='13'";
const CJK_S = "font-family='Noto Sans SC, PingFang SC, sans-serif' font-size='11'";

const FILL = {
  attn: "fill='#f6e9e5' stroke='#cf9d92'",
  norm: "fill='#e9efe9' stroke='#a9bfa9'",
  proj: "fill='#e9edf3' stroke='#aebcd2'",
  data: "fill='#f1efe8' stroke='#c6c0af'",
  cache: "fill='#f3ecdc' stroke='#cbb98d'",
  op: "fill='#eeeef2' stroke='#b6b6c4'",
};

function open(h, label) {
  return `<svg class="fig-svg sig" viewBox="0 0 720 ${h}" role="img" aria-label="${label}" xmlns="http://www.w3.org/2000/svg">`;
}
function flow(path, dur, begin) {
  return `<circle class='sig-flow' r='2.6' fill='var(--seal)'><animateMotion dur='${dur}s' begin='${begin}s' repeatCount='indefinite' path='${path}'/></circle>`;
}

/* ============ 8.1 §3.4 — VLM 三个接入点并排对照 ============ */
function vlmAccessPoints() {
  const p = [open(330, 'VLM 三个接入点：input-level 拼接 / cross-attention 注入 / 混合')];
  const panels = [
    { x0: 12, tag: '(a) input-level', sub: 'LLaVA 范式', conn: 'linear / MLP', mode: 'all', strip: ['s', 's', 's', 'v', 'v', 'v', 's', 's'], note1: 'v₁..v_M 拼进输入序列，M = N 全额', note2: '融合全部 L 层 · LLM 须参训' },
    { x0: 248, tag: '(b) cross-attn-level', sub: 'Flamingo gated xattn', conn: 'resampler 64', mode: 'xattn', strip: ['s', 's', 's', 's', 's', 's', 's', 's'], note1: '视觉不占输入序列，M = 0', note2: '交互仅插入层 · LLM 冻结' },
    { x0: 484, tag: '(c) 混合', sub: 'BLIP-2 Q-Former', conn: 'Q-Former B=32', mode: 'all', strip: ['s', 's', 's', 'v', 'v', 's', 's', 's'], note1: '压缩后拼进输入，M = B = 32 常数', note2: 'token 数与分辨率解耦 · LLM 冻结' },
  ];
  for (const q of panels) {
    const x0 = q.x0;
    p.push(`<rect x='${x0}' y='10' width='224' height='306' rx='4' fill='none' stroke='var(--rule-soft)'/>`);
    p.push(`<text ${MONO} x='${x0 + 12}' y='32' fill='var(--ink)'>${q.tag}</text>`);
    p.push(`<text ${CJK_S} x='${x0 + 12}' y='48' fill='var(--ink-faint)'>${q.sub}</text>`);
    // 视觉侧：z_img → connector
    p.push(`<rect x='${x0 + 12}' y='64' width='80' height='26' rx='3' ${FILL.data} stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${x0 + 52}' y='81' fill='var(--ink)' text-anchor='middle'>z_img</text>`);
    p.push(`<rect x='${x0 + 12}' y='108' width='80' height='26' rx='3' ${FILL.proj} stroke-width='1.1'/>`);
    p.push(`<text ${MONO_S} x='${x0 + 52}' y='121' fill='var(--ink)' text-anchor='middle'>${q.conn.split(' ')[0]}</text>`);
    p.push(`<text ${MONO_S} x='${x0 + 52}' y='132' fill='var(--ink-faint)' text-anchor='middle'>${q.conn.split(' ').slice(1).join(' ') || 'φ'}</text>`);
    p.push(`<line x1='${x0 + 52}' y1='90' x2='${x0 + 52}' y2='108' stroke='#a9a498' stroke-width='1.2'/>`);
    // LLM 层叠
    const lx = x0 + 118, bw = 88;
    const kinds = q.mode === 'xattn' ? ['llm', 'xa', 'llm', 'xa', 'llm', 'xa'] : ['llm', 'llm', 'llm', 'llm', 'llm', 'llm'];
    kinds.forEach((k, i) => {
      const y = 64 + i * 29;
      if (k === 'xa') {
        p.push(`<rect x='${lx}' y='${y}' width='${bw}' height='24' rx='3' ${FILL.attn} stroke-width='1.1'/>`);
        p.push(`<text ${MONO_S} x='${lx + bw / 2}' y='${y + 16}' fill='var(--ink)' text-anchor='middle'>xattn·tanh α</text>`);
      } else {
        p.push(`<rect x='${lx}' y='${y}' width='${bw}' height='24' rx='3' ${q.mode === 'xattn' ? FILL.plain || "fill='var(--paper)' stroke='#c9c4b8'" : FILL.attn} stroke-width='1.1'/>`);
        p.push(`<text ${MONO_S} x='${lx + bw / 2}' y='${y + 16}' fill='${q.mode === 'xattn' ? 'var(--ink-faint)' : 'var(--ink)'}' text-anchor='middle'>${q.mode === 'xattn' ? 'LLM 冻结' : 'LLM'}</text>`);
      }
    });
    // 输入序列条
    q.strip.forEach((c, i) => {
      const x = lx + i * 11;
      const isV = c === 'v';
      p.push(`<rect class='sig-cell' style='--i:${(i * 0.12).toFixed(2)}' x='${x}' y='246' width='9' height='15' rx='2' fill='${isV ? 'var(--seal-wash)' : 'var(--paper-sunk)'}' stroke='${isV ? 'var(--seal)' : 'var(--rule)'}' stroke-width='1'/>`);
    });
    // 连接与流光
    if (q.mode === 'xattn') {
      [93, 151, 209].forEach((y, k) => {
        p.push(`<path d='M ${x0 + 92} 121 C ${x0 + 104} 121, ${x0 + 106} ${y + 12}, ${lx} ${y + 12}' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
        p.push(flow(`M ${x0 + 92} 121 C ${x0 + 104} 121, ${x0 + 106} ${y + 12}, ${lx} ${y + 12}`, 2.4, k * 0.5));
      });
      p.push(`<text ${MONO_S} x='${x0 + 12}' y='154' fill='var(--ink-faint)'>K/V 注入</text>`);
      p.push(`<text ${MONO_S} x='${x0 + 12}' y='168' fill='var(--ink-faint)'>α=0 ⇒ 恒等</text>`);
    } else {
      const path = `M ${x0 + 92} 121 C ${x0 + 108} 121, ${x0 + 104} 200, ${lx + 30} 244`;
      p.push(`<path d='${path}' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
      p.push(flow(path, 2.4, 0.3));
    }
    p.push(`<text ${CJK_S} x='${x0 + 12}' y='284' fill='var(--ink-soft)'>${q.note1}</text>`);
    p.push(`<text ${CJK_S} x='${x0 + 12}' y='301' fill='var(--ink-faint)'>${q.note2}</text>`);
  }
  p.push('</svg>');
  return p.join('');
}

/* ============ 8.5 §1 — 统一理解+生成的三条路线分流 ============ */
function unifiedThreeRoutes() {
  const p = [open(320, '统一理解+生成：图像进主干的三种形态（离散 token / 连续 latent / diffusion head）')];
  p.push(`<text ${CJK_S} x='24' y='26' fill='var(--ink-soft)'>同一个 θ 同时承担 p(y|x,c) 与 p(x|c)——路线按「图像以什么形态进主干」分岔，瓶颈随之搬家</text>`);
  // 左：图像源
  p.push(`<rect x='24' y='146' width='56' height='48' rx='3' ${FILL.data} stroke-width='1.1'/>`);
  p.push(`<circle cx='40' cy='162' r='6' fill='none' stroke='var(--ink-soft)'/>`);
  p.push(`<line x1='28' y1='186' x2='76' y2='186' stroke='var(--ink-soft)' stroke-width='1'/>`);
  p.push(`<text ${MONO_S} x='52' y='212' fill='var(--ink-faint)' text-anchor='middle'>x_img</text>`);
  // 主干
  p.push(`<rect x='330' y='56' width='84' height='224' rx='4' ${FILL.attn} stroke-width='1.2'/>`);
  p.push(`<text ${CJK} x='372' y='158' fill='var(--ink)' text-anchor='middle'>共享主干</text>`);
  p.push(`<text ${CJK} x='372' y='180' fill='var(--ink)' text-anchor='middle'>θ</text>`);
  // 路线 1：离散 token
  p.push(`<rect x='120' y='80' width='88' height='28' rx='3' ${FILL.op} stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='164' y='98' fill='var(--ink)' text-anchor='middle'>VQ tokenizer</text>`);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
    p.push(`<rect class='sig-cell' style='--i:${(c * 0.08 + r * 0.05).toFixed(2)}' x='${226 + c * 12}' y='${80 + r * 10}' width='10' height='8' rx='1' fill='var(--cache, #f3ecdc)' stroke='#cbb98d' stroke-width='1'/>`);
  }
  p.push(`<text ${MONO_S} x='226' y='122' fill='var(--ink-faint)'>code 网格 K=32768</text>`);
  p.push(`<path d='M 80 170 C 96 170, 100 94, 120 94' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<path d='M 208 94 L 226 94' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<path d='M 278 94 L 330 94' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(flow('M 80 170 C 100 170, 100 94, 120 94 L 330 94', 3.2, 0));
  p.push(`<text ${CJK_S} x='436' y='88' fill='var(--ink-soft)'>路线 1 · 离散 token：Emu3 / Chameleon</text>`);
  p.push(`<text ${CJK_S} x='436' y='106' fill='var(--ink-faint)'>单 softmax 逐 token CE；瓶颈 N·log₂K 硬上界</text>`);
  // 路线 2：连续 latent
  p.push(`<rect x='120' y='156' width='88' height='28' rx='3' ${FILL.op} stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='164' y='174' fill='var(--ink)' text-anchor='middle'>VAE enc</text>`);
  p.push(`<rect x='226' y='154' width='92' height='32' rx='3' ${FILL.cache} stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='272' y='168' fill='var(--ink)' text-anchor='middle'>加噪 patch x_t</text>`);
  p.push(`<text ${MONO_S} x='272' y='181' fill='var(--ink-faint)' text-anchor='middle'>BOI … EOI</text>`);
  p.push(`<path d='M 80 170 L 120 170' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<path d='M 208 170 L 226 170 M 318 170 L 330 170' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(flow('M 80 170 L 330 170', 2.8, 0.5));
  p.push(`<text ${CJK_S} x='436' y='164' fill='var(--ink-soft)'>路线 2 · 连续 latent：Transfusion</text>`);
  p.push(`<text ${CJK_S} x='436' y='182' fill='var(--ink-faint)'>主干内 T 步去噪；软瓶颈 = VAE 重建</text>`);
  // 路线 3：diffusion head（图像不进主干）
  p.push(`<path d='M 414 248 L 496 248' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(flow('M 414 248 L 496 248', 2, 1));
  p.push(`<text ${MONO_S} x='420' y='262' fill='var(--ink-faint)'>hidden → c</text>`);
  p.push(`<rect x='496' y='232' width='104' height='32' rx='3' ${FILL.proj} stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='548' y='246' fill='var(--ink)' text-anchor='middle'>小扩散头</text>`);
  p.push(`<text ${MONO_S} x='548' y='259' fill='var(--ink-faint)' text-anchor='middle'>T 步在 head 内</text>`);
  p.push(`<path d='M 600 248 L 640 248' fill='none' stroke='#a9a498' stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='646' y='252' fill='var(--ink)'>latent</text>`);
  p.push(`<text ${CJK_S} x='436' y='216' fill='var(--ink-soft)'>路线 3 · diffusion head：SEED-X 式</text>`);
  p.push(`<text ${CJK_S} x='436' y='232' fill='var(--ink-faint)' x2='0'>图像不进主干；主干可冻结</text>`);
  p.push(`<text ${CJK_S} x='436' y='292' fill='var(--ink-faint)'>瓶颈位置：码本 / VAE / head 容量——三条路线各付各的</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 9.1 §1 — video VAE 时空压缩账 ============ */
function videoVaeCompression() {
  const p = [open(300, 'Video VAE：T×H×W×3 → t×h×w×c 的时空压缩账（τ,s,s,c)=(4,8,8,16)')];
  p.push(`<text ${CJK_S} x='24' y='28' fill='var(--ink-soft)'>通用档 (τ, s, s, c) = (4, 8, 8, 16)：三个维度各付各的压缩率，token 账与体积账分开结</text>`);
  // 大立方体（前视 + 顶面 + 侧面）
  const bx = 44, by = 96, bw = 150, bh = 104, dx = 26, dy = -22;
  p.push(`<polygon points='${bx},${by} ${bx + dx},${by + dy} ${bx + bw + dx},${by + dy} ${bx + bw},${by}' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
  p.push(`<polygon points='${bx + bw},${by} ${bx + bw + dx},${by + dy} ${bx + bw + dx},${by + dy + bh} ${bx + bw},${by + bh}' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
  p.push(`<rect x='${bx}' y='${by}' width='${bw}' height='${bh}' fill='var(--paper-raised)' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  // 首帧切片
  p.push(`<rect x='${bx}' y='${by}' width='14' height='${bh}' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1'/>`);
  p.push(`<text ${MONO_S} x='${bx + 7}' y='${by + bh + 14}' fill='var(--seal)' text-anchor='middle'>首帧</text>`);
  // 正面网格线（帧）
  for (let i = 1; i < 6; i++) p.push(`<line x1='${bx + 14 + i * 22}' y1='${by}' x2='${bx + 14 + i * 22}' y2='${by + bh}' stroke='var(--rule-soft)'/>`);
  for (let i = 1; i < 4; i++) p.push(`<line x1='${bx}' y1='${by + i * 26}' x2='${bx + bw}' y2='${by + i * 26}' stroke='var(--rule-soft)'/>`);
  p.push(`<text ${MONO} x='${bx}' y='${by + bh + 34}' fill='var(--ink)'>V ∈ T×H×W×3</text>`);
  p.push(`<text ${MONO_S} x='${bx}' y='${by + bh + 52}' fill='var(--ink-soft)'>120×720×1280×3 ≈ 3.32×10⁸</text>`);
  p.push(`<text ${MONO_S} x='${bx}' y='${by + bh + 68}' fill='var(--ink-faint)'>uint8 ≈ 332 MB</text>`);
  // 三条压缩箭头
  const arrows = [
    { y: 100, label: '时间 ÷ τ=4（首帧独占 1 格）' },
    { label: '空间 ÷8 × ÷8', y: 140 },
    { label: '通道 3 → c=16', y: 180 },
  ];
  arrows.forEach((a, i) => {
    const y = a.y;
    p.push(`<path d='M ${bx + bw + dx + 8} ${y} L 462 ${y}' fill='none' stroke='#a9a498' stroke-width='1.2'/>`);
    p.push(`<polygon points='462,${y} 454,${y - 3.5} 454,${y + 3.5}' fill='var(--ink-soft)'/>`);
    p.push(`<text ${MONO_S} x='${bx + bw + dx + 14}' y='${y - 7}' fill='var(--ink-soft)'>${a.label}</text>`);
    p.push(flow(`M ${bx + bw + dx + 8} ${y} L 462 ${y}`, 2.2, i * 0.45));
  });
  // 小立方体 latent
  const sx = 470, sy = 116, sw = 64, sh = 48, sd = 14, sdy = -12;
  p.push(`<polygon points='${sx},${sy} ${sx + sd},${sy + sdy} ${sx + sw + sd},${sy + sdy} ${sx + sw},${sy}' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
  p.push(`<polygon points='${sx + sw},${sy} ${sx + sw + sd},${sy + sdy} ${sx + sw + sd},${sy + sdy + sh} ${sx + sw},${sy + sh}' fill='var(--paper-sunk)' stroke='var(--rule)'/>`);
  p.push(`<rect x='${sx}' y='${sy}' width='${sw}' height='${sh}' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.2'/>`);
  for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
    p.push(`<rect class='sig-cell' style='--i:${(c + r * 2) * 0.3}' x='${sx + 6 + c * 28}' y='${sy + 6 + r * 20}' width='24' height='16' rx='2' fill='var(--paper-raised)' stroke='var(--seal)' stroke-width='0.8'/>`);
  }
  p.push(`<text ${MONO} x='${sx}' y='${sy + sh + 34}' fill='var(--ink)'>z ∈ t×h×w×c</text>`);
  p.push(`<text ${MONO_S} x='${sx}' y='${sy + sh + 52}' fill='var(--ink-soft)'>31×90×160×16</text>`);
  p.push(`<text ${MONO_S} x='${sx}' y='${sy + sh + 68}' fill='var(--ink-faint)'>bf16 ≈ 14.3 MB</text>`);
  // 两本账
  p.push(`<text ${MONO_S} x='250' y='252' fill='var(--ink-soft)'>token 账：N = 31·90·160 = 446,400；N² ≈ 1.99×10¹¹ ≈ 399 GB bf16/层</text>`);
  p.push(`<text ${MONO_S} x='250' y='272' fill='var(--ink-soft)'>体积账：R ≈ 3τs²/c = 3·4·64/16 = 48×（首帧 +1 ⇒ 实测 46.5×）</text>`);
  p.push(`<text ${CJK_S} x='44' y='290' fill='var(--ink-faint)'>首帧只做空间压缩 → latent 帧 0 独占</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 9.2 §2 — full 3D vs factorized 注意力范围 ============ */
function attentionTopology() {
  const p = [open(330, 'full 3D attention 与 factorized attention 的可达范围对照')];
  const R = 5, C = 9, cell = 21, gap = 3, pitch = cell + 3;
  const draw = (x0, mode) => {
    const y0 = 50;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const x = x0 + c * pitch, y = y0 + r * pitch;
      const isQ = r === 2 && c === 4;
      if (mode === 'full') {
        p.push(`<rect class='sig-cell' style='--i:${((r * C + c) * 0.04).toFixed(2)}' x='${x}' y='${y}' width='${cell}' height='${cell}' rx='2' fill='${isQ ? 'var(--seal)' : 'var(--ink)'}' stroke='none'/>`);
      } else {
        const reach = r === 2 || c === 4;
        if (reach) {
          p.push(`<rect class='sig-cell' style='--i:${((Math.abs(r - 2) + Math.abs(c - 4)) * 0.18).toFixed(2)}' x='${x}' y='${y}' width='${cell}' height='${cell}' rx='2' fill='${isQ ? 'var(--seal)' : 'var(--seal-wash)'}' stroke='var(--seal)' stroke-width='1'/>`);
        } else {
          p.push(`<rect x='${x}' y='${y}' width='${cell}' height='${cell}' rx='2' fill='none' stroke='var(--rule)' stroke-dasharray='2 3'/>`);
        }
      }
    }
    // 轴标注
    p.push(`<text ${MONO_S} x='${x0 - 8}' y='${y0 + 14}' fill='var(--ink-faint)' text-anchor='end'>t0</text>`);
    p.push(`<text ${MONO_S} x='${x0 - 8}' y='${y0 + 4 * pitch + 14}' fill='var(--ink-faint)' text-anchor='end'>t4</text>`);
    p.push(`<text ${MONO_S} x='${x0 + 2}' y='${y0 - 8}' fill='var(--ink-faint)'>p0</text>`);
    p.push(`<text ${MONO_S} x='${x0 + 8 * pitch}' y='${y0 - 6}' fill='var(--ink-faint)'>p8</text>`);
    return y0;
  };
  // 左：full
  p.push(`<text ${CJK} x='40' y='30' fill='var(--ink)'>full 3D：每个 token 看全部 N 个</text>`);
  draw(40, 'full');
  p.push(`<text ${MONO_S} x='40' y='202' fill='var(--ink-soft)'>二次项 4N²d — Wan 几何 1.87 PFLOPs/层</text>`);
  p.push(`<text ${CJK_S} x='40' y='220' fill='var(--ink-faint)'>省不下，但时空耦合全部留给权重去学</text>`);
  // 右：factorized
  const fx = 400;
  p.push(`<text ${CJK} x='${fx}' y='30' fill='var(--ink)'>factorized：空间层内 + 时间层间</text>`);
  const y0 = draw(fx, 'fact');
  // 两跳中继虚线：(2,4)→(2,7)→(4,7) 与 (2,4)→(4,4)→(4,7)
  const cc = (r, c) => [fx + c * pitch + cell / 2, y0 + r * pitch + cell / 2];
  const [q1, q2] = cc(2, 4), [a1, a2] = cc(2, 7), [b1, b2] = cc(4, 4), [t1, t2] = cc(4, 7);
  p.push(`<path class='sig-dashflow' d='M ${q1} ${q2} L ${a1} ${a2} L ${t1} ${t2}' fill='none' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  p.push(`<path class='sig-dashflow' d='M ${q1} ${q2} L ${b1} ${b2} L ${t1} ${t2}' fill='none' stroke='var(--ink-soft)' stroke-width='1.2'/>`);
  p.push(`<line x1='${t1 - 5}' y1='${t2 - 5}' x2='${t1 + 5}' y2='${t2 + 5}' stroke='var(--ink-faint)' stroke-width='1.4'/>`);
  p.push(`<line x1='${t1 - 5}' y1='${t2 + 5}' x2='${t1 + 5}' y2='${t2 - 5}' stroke='var(--ink-faint)' stroke-width='1.4'/>`);
  p.push(`<text ${MONO_S} x='${fx}' y='202' fill='var(--ink-soft)'>二次项 4Nd(S+t) — 省幅 tS/(t+S) ≈ 21×（Wan 几何）</text>`);
  p.push(`<text ${CJK_S} x='${fx}' y='220' fill='var(--ink-faint)'>95.2% token 对无直接边，跨帧跨位靠两跳中继（虚线）</text>`);
  p.push(`<text ${CJK_S} x='${fx}' y='238' fill='var(--ink-faint)'>t≪S 时省幅趋近 t —— 拉长时长几乎不省</text>`);
  // 底注
  p.push(`<text ${CJK_S} x='40' y='272' fill='var(--ink-soft)'>行 = 帧 t，列 = 空间位 p；朱砂 = query。factorized 的 attention 图是 full N² 的稀疏子图：</text>`);
  p.push(`<text ${CJK_S} x='40' y='290' fill='var(--ink-faint)'>硬编码邻域省下 FLOPs，花掉的是表达力预算（§2.4）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 10.1 §2 — STFT → mel 时频面转换 ============ */
function stftToMel() {
  const p = [open(300, 'STFT 到 log-mel：波形 → 逐帧幅度谱 → mel 滤波器组加权')];
  p.push(`<text ${CJK_S} x='24' y='26' fill='var(--ink-soft)'>16 kHz · 25 ms 窗 · 10 ms hop → 100 fps；窗长一个旋钮管两个分辨率：Δt·Δf ≈ 1</text>`);
  // 波形
  let d = 'M 24 130';
  for (let x = 24; x <= 216; x += 3) {
    const t = (x - 24) / 192;
    const env = Math.exp(-Math.pow((t - 0.45) * 2.4, 2)) * 0.9 + 0.15;
    const y = 130 - 34 * env * Math.sin(t * 34) * Math.cos(t * 9);
    d += ` L ${x} ${y.toFixed(1)}`;
  }
  p.push(`<path d='${d}' fill='none' stroke='var(--ink)' stroke-width='1.3'/>`);
  p.push(`<line x1='24' y1='130' x2='216' y2='130' stroke='var(--rule)' stroke-width='0.8'/>`);
  // 三个重叠窗
  [40, 72, 104].forEach((x, i) => {
    p.push(`<rect x='${x}' y='72' width='80' height='116' rx='3' fill='none' stroke='var(--ink-faint)' stroke-width='1' stroke-dasharray='3 4'/>`);
    p.push(`<circle class='sig-step' style='--i:${i}' cx='${x + 6}' cy='66' r='3' fill='var(--seal)'/>`);
    p.push(`<text ${MONO_S} x='${x + 14}' y='66' fill='var(--ink-faint)'>w${i}</text>`);
  });
  p.push(`<text ${MONO_S} x='24' y='208' fill='var(--ink-soft)'>窗 25 ms = 400 样本（重叠）</text>`);
  p.push(`<text ${MONO_S} x='24' y='224' fill='var(--ink-faint)'>hop 10 ms = 160 → fps = 16000/160 = 100</text>`);
  // STFT 帧列（4 帧 × 10 频点，幅度=墨色浓度）
  const mags = [
    [0.9, 0.7, 0.85, 0.4, 0.55, 0.3, 0.2, 0.35, 0.15, 0.1],
    [0.85, 0.75, 0.5, 0.6, 0.3, 0.45, 0.25, 0.2, 0.3, 0.12],
    [0.7, 0.9, 0.6, 0.35, 0.5, 0.55, 0.3, 0.4, 0.2, 0.15],
    [0.8, 0.6, 0.75, 0.5, 0.4, 0.3, 0.5, 0.25, 0.3, 0.2],
  ];
  mags.forEach((col, ci) => {
    const x = 276 + ci * 40;
    col.forEach((m, bi) => {
      const y = 141 - bi * 10;
      p.push(`<rect class='sig-cell' style='--i:${(ci * 0.3 + bi * 0.03).toFixed(2)}' x='${x}' y='${y}' width='26' height='9' rx='1' fill='var(--ink)' fill-opacity='${m}'/>`);
    });
  });
  p.push(`<text ${MONO_S} x='276' y='166' fill='var(--ink-faint)'>帧 m →（逐帧 DFT）</text>`);
  p.push(`<text ${MONO_S} x='276' y='182' fill='var(--ink-faint)'>纵轴 k：|X[m,k]|，N/2+1 线性频点</text>`);
  p.push(flow('M 216 130 C 240 130, 252 100, 276 100', 2.2, 0.2));
  // mel 三角滤波器组
  p.push(`<line x1='480' y1='150' x2='700' y2='150' stroke='var(--ink-soft)' stroke-width='1'/>`);
  const tris = [
    [496, 13], [524, 15], [556, 18], [592, 22], [632, 27], [676, 33],
  ];
  tris.forEach(([cx, hw], i) => {
    p.push(`<path class='sig-diag' style='--i:${i}' d='M ${cx - hw} 150 L ${cx} 86 L ${cx + hw} 150 Z' fill='var(--seal-wash)' fill-opacity='0.55' stroke='var(--seal)' stroke-width='1'/>`);
  });
  p.push(`<text ${MONO_S} x='470' y='70' fill='var(--ink-soft)'>mel 滤波器组 M：mel 轴等距、Hz 轴低密高疏</text>`);
  p.push(`<text ${MONO_S} x='480' y='164' fill='var(--ink-faint)'>f / Hz →</text>`);
  p.push(flow('M 436 100 C 452 100, 462 118, 484 118', 2.2, 0.9));
  // 输出
  p.push(`<rect x='480' y='180' width='222' height='30' rx='3' ${FILL.data} stroke-width='1.1'/>`);
  p.push(`<text ${MONO_S} x='490' y='199' fill='var(--ink)'>S_mel[m,:] = log(M·|X[m,:]|²)</text>`);
  p.push(`<text ${CJK_S} x='480' y='226' fill='var(--ink-soft)'>80 通道 log-mel · 100 fps（Whisper 档）</text>`);
  p.push(`<text ${CJK_S} x='480' y='244' fill='var(--ink-faint)'>log：乘法声强变加法 + 压缩动态范围</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 10.2 §2.4 — CTC 对齐格点与路径族 ============ */
function ctcLattice() {
  const p = [open(340, 'CTC：扩展标签网格上的合法路径族、前向概率与 blank 坍缩')];
  p.push(`<text ${CJK} x='30' y='28' fill='var(--ink)'>合法路径族：y = ab，T = 3</text>`);
  const x0 = 70, y0 = 50, cw = 62, ch = 32, gx = 10, gy = 8;
  const cellPos = (t, s) => [x0 + (t - 1) * (cw + gx), y0 + (5 - s) * (ch + gy)];
  const sym = { 1: '∅', 2: 'a', 3: '∅', 4: 'b', 5: '∅' };
  for (let s = 5; s >= 1; s--) {
    const [, y] = cellPos(1, s);
    p.push(`<text ${MONO_S} x='${x0 - 34}' y='${y + 20}' fill='var(--ink-soft)'>s=${s}</text>`);
    p.push(`<text ${MONO_S} x='${x0 - 14}' y='${y + 20}' fill='${sym[s] === '∅' ? 'var(--ink-faint)' : 'var(--seal)'}'>${sym[s]}</text>`);
  }
  for (let t = 1; t <= 3; t++) {
    const [x] = cellPos(t, 5);
    p.push(`<text ${MONO_S} x='${x + cw / 2}' y='${y0 - 10}' fill='var(--ink-faint)' text-anchor='middle'>t=${t}</text>`);
  }
  const alpha = { '1,1': '0.30', '1,2': '0.50', '2,1': '.06', '2,2': '0.32', '2,3': '0.10', '2,4': '0.20', '3,1': '.018', '3,2': '.038', '3,3': '.126', '3,4': '0.372', '3,5': '0.06' };
  const endpoints = new Set(['1,1', '1,2', '3,4', '3,5']);
  for (let t = 1; t <= 3; t++) for (let s = 1; s <= 5; s++) {
    const [x, y] = cellPos(t, s);
    const v = alpha[`${t},${s}`];
    if (v !== undefined) {
      const hot = endpoints.has(`${t},${s}`);
      p.push(`<rect class='sig-cell' style='--i:${((t - 1) * 0.5 + s * 0.07).toFixed(2)}' x='${x}' y='${y}' width='${cw}' height='${ch}' rx='3' fill='${hot ? 'var(--seal-wash)' : 'var(--paper-raised)'}' stroke='${hot ? 'var(--seal)' : 'var(--rule)'}' stroke-width='1.1'/>`);
      p.push(`<text class='sig-cell-text' style='--i:${((t - 1) * 0.5 + s * 0.07).toFixed(2)}' ${MONO_S} x='${x + cw / 2}' y='${y + 20}' fill='${hot ? 'var(--seal-deep)' : 'var(--ink-soft)'}' text-anchor='middle'>${v}</text>`);
    } else {
      p.push(`<rect x='${x}' y='${y}' width='${cw}' height='${ch}' rx='3' fill='none' stroke='var(--rule)' stroke-dasharray='2 3'/>`);
    }
  }
  // 转移细箭头（DP 合并）
  const edge = (t1, s1, t2, s2, strong) => {
    const [a, b] = cellPos(t1, s1), [c, dd] = cellPos(t2, s2);
    const x1 = a + cw, y1 = b + ch / 2, x2 = c, y2 = dd + ch / 2;
    p.push(`<path d='M ${x1} ${y1} C ${x1 + 5} ${y1}, ${x2 - 5} ${y2}, ${x2} ${y2}' fill='none' stroke='${strong ? 'var(--seal)' : '#a9a498'}' stroke-width='${strong ? 1.6 : 0.9}' ${strong ? '' : "opacity='0.6'"}/>`);
  };
  edge(1, 1, 2, 1, false); edge(1, 1, 2, 2, true);
  edge(1, 2, 2, 2, false); edge(1, 2, 2, 3, false); edge(1, 2, 2, 4, false);
  edge(2, 2, 3, 2, false); edge(2, 2, 3, 3, false); edge(2, 2, 3, 4, true);
  edge(2, 3, 3, 4, false); edge(2, 4, 3, 4, false); edge(2, 4, 3, 5, false);
  // 主路径流光：(∅,a,b) — 停留? (1,1)→(2,2)→(3,4)
  const [p1x, p1y] = cellPos(1, 1), [p2x, p2y] = cellPos(2, 2), [p3x, p3y] = cellPos(3, 4);
  p.push(flow(`M ${p1x + cw / 2} ${p1y + ch / 2} L ${p2x + cw / 2} ${p2y + ch / 2} L ${p3x + cw / 2} ${p3y + ch / 2}`, 2.6, 0.4));
  p.push(`<text ${MONO_S} x='${x0}' y='278' fill='var(--ink)'>P(y|x) = α₃(4)+α₃(5) = 0.372+0.060</text>`);
  // 右：五条路径枚举 + B 坍缩
  const rx = 352;
  p.push(`<text ${CJK} x='${rx}' y='28' fill='var(--ink)'>枚举 5 条 → B 坍缩 → 求和</text>`);
  const paths = [
    ['a', 'a', 'b', '0.5·0.4·0.6', '0.120'],
    ['a', 'b', 'b', '0.5·0.4·0.6', '0.120'],
    ['a', 'b', '∅', '0.5·0.4·0.3', '0.060'],
    ['a', '∅', 'b', '0.5·0.2·0.6', '0.060'],
    ['∅', 'a', 'b', '0.3·0.4·0.6', '0.072'],
  ];
  paths.forEach((row, i) => {
    const y = 48 + i * 42;
    row.slice(0, 3).forEach((s, j) => {
      const x = rx + j * 36;
      const blank = s === '∅';
      p.push(`<rect class='sig-cell' style='--i:${(i * 0.35 + j * 0.08).toFixed(2)}' x='${x}' y='${y}' width='30' height='24' rx='3' fill='${blank ? 'var(--paper-sunk)' : 'var(--seal-wash)'}' stroke='${blank ? 'var(--rule)' : 'var(--seal)'}' stroke-width='1'/>`);
      p.push(`<text class='sig-cell-text' style='--i:${(i * 0.35 + j * 0.08).toFixed(2)}' ${MONO_S} x='${x + 15}' y='${y + 16}' fill='${blank ? 'var(--ink-faint)' : 'var(--seal-deep)'}' text-anchor='middle'>${s}</text>`);
    });
    p.push(`<text ${MONO_S} x='${rx + 122}' y='${y + 16}' fill='var(--ink-faint)'>${row[3]} = ${row[4]}</text>`);
    p.push(`<text ${MONO_S} x='${rx + 268}' y='${y + 16}' fill='var(--ink-soft)'>B→ ab</text>`);
  });
  p.push(`<text ${MONO_S} x='${rx}' y='278' fill='var(--ink)'>Σ = 0.432 = P(ab|x) — 与 DP 一致</text>`);
  p.push(`<text ${CJK_S} x='30' y='310' fill='var(--ink-faint)'>枚举 ~C(T,U) 不可行 → 前向 DP 在 O(TU) 精确边缘化；blank 两职：吸收多余帧、阻断重复合并（a∅a → aa）</text>`);
  p.push(`<text ${CJK_S} x='30' y='328' fill='var(--ink-faint)'>粗线 = 路径 (∅,a, b)：s→s+1 恒合法，s→s+2 仅当 y′_{s+2} ≠ y′_s（跳格过 blank）</text>`);
  p.push('</svg>');
  return p.join('');
}

/* ============ 10.4 §6.2 — 全双工 vs 级联的延迟时间轴 ============ */
function duplexLatency() {
  const p = [open(300, '全双工与级联方案的端到端延迟预算对照（时间轴）')];
  p.push(`<text ${CJK_S} x='24' y='26' fill='var(--ink-soft)'>端到端 = 用户语音帧进麦克风 → 模型声学响应出扬声器（Moshi 原文：理论 160 ms / 实测 200 ms）</text>`);
  const ms = (ms) => 130 + ms * 0.5;
  // 级联
  p.push(`<text ${CJK_S} x='24' y='62' fill='var(--ink)'>级联半双工</text>`);
  const casc = [
    [0, 500, '端点检测 300–700'],
    [500, 300, 'LLM TTFT 200–400'],
    [800, 200, 'TTS 首包 100–300'],
  ];
  casc.forEach(([s, w, label], i) => {
    p.push(`<text ${CJK_S} x='${(ms(s) + ms(s + w)) / 2}' y='70' fill='var(--ink-faint)' text-anchor='middle'>${label}</text>`);
    p.push(`<rect class='sig-bar' style='--i:${i}' x='${ms(s)}' y='78' width='${w * 0.5}' height='22' rx='3' ${FILL.op} stroke-width='1.1'/>`);
  });
  p.push(`<text ${MONO_S} x='${ms(1000) + 8}' y='94' fill='var(--ink)'>≈1 s 秒级</text>`);
  p.push(`<text ${CJK_S} x='130' y='120' fill='var(--ink-faint)'>「等整句」串行三次：ASR 等端点、LLM 等首 token、TTS 等首包——差距来自串行结构本身</text>`);
  // Moshi
  p.push(`<text ${CJK_S} x='24' y='162' fill='var(--ink)'>Moshi 全双工</text>`);
  const moshi = [
    [0, 80, '帧缓冲 80'],
    [80, 15, 'Mimi enc ~15'],
    [95, 40, 'LM 两级 9 步 ~40'],
    [135, 15, 'Mimi dec ~15'],
    [110 + 25, 20, '网络/播放 ~20'],
  ];
  moshi.forEach(([s, w], i) => {
    p.push(`<rect class='sig-bar' style='--i:${i}' x='${ms(s)}' y='170' width='${w * 0.5}' height='22' rx='3' fill='var(--seal-wash)' stroke='var(--seal)' stroke-width='1.1'/>`);
  });
  // 段标签：斜引线牵到水平散开的注记（段太窄，直接在下方放不下）
  const lab = [
    [40, 150, 208, '帧缓冲 80'], [87.5, 232, 226, 'Mimi enc ~15'], [117.5, 320, 208, 'LM 两级 9 步 ~40'], [142.5, 452, 226, 'Mimi dec ~15'], [165, 540, 208, '网络/播放 ~20'],
  ];
  lab.forEach(([cxms, tx, ty, text]) => {
    const x = ms(cxms);
    p.push(`<line x1='${x}' y1='192' x2='${tx + 4}' y2='${ty - 10}' stroke='var(--rule)' stroke-width='0.8'/>`);
    p.push(`<text ${CJK_S} x='${tx}' y='${ty}' fill='var(--ink-faint)'>${text}</text>`);
  });
  p.push(`<text ${MONO_S} x='${ms(170) + 10}' y='186' fill='var(--seal)'>≈160–200 ms</text>`);
  // 帧栅格
  p.push(`<line x1='130' y1='250' x2='630' y2='250' stroke='var(--ink-soft)' stroke-width='1'/>`);
  for (let i = 0; i <= 12; i++) {
    const x = 130 + i * 40;
    p.push(`<line x1='${x}' y1='${i === 0 ? 244 : 246}' x2='${x}' y2='254' stroke='var(--ink-soft)' stroke-width='${i % 2 === 0 ? 1.2 : 0.7}'/>`);
  }
  p.push(`<circle class='sig-step' style='--i:0' cx='130' cy='250' r='3.2' fill='var(--seal)'/>`);
  p.push(`<circle class='sig-step' style='--i:1' cx='170' cy='250' r='3.2' fill='var(--seal)'/>`);
  p.push(`<circle class='sig-step' style='--i:2' cx='210' cy='250' r='3.2' fill='var(--seal)'/>`);
  p.push(`<text ${MONO_S} x='126' y='268' fill='var(--ink-faint)'>0</text>`);
  p.push(`<text ${MONO_S} x='${210 - 40}' y='268' fill='var(--ink-faint)'>160 ms（2 帧）</text>`);
  p.push(`<text ${MONO_S} x='626' y='268' fill='var(--ink-faint)' text-anchor='end'>960 ms</text>`);
  p.push(`<text ${CJK_S} x='130' y='292' fill='var(--ink-faint)'>12.5 Hz 帧栅格：攒帧 1 帧 + 生成 1 帧 = 理论 2 帧；帧率砍半 ⇒ 攒帧一项立省一整帧</text>`);
  p.push('</svg>');
  return p.join('');
}

export const FIGURES = [
  {
    slug: '08-multimodal-llm/01-vlm-architecture-taxonomy',
    afterHeading: /^### 3\.4 三接入点对照/,
    title: '三个接入点并排：序列 / 层间 / 混合',
    svg: vlmAccessPoints,
  },
  {
    slug: '08-multimodal-llm/05-unified-understanding-generation',
    afterHeading: /^## 1\. 问题定义/,
    title: '统一模型的三条路线：瓶颈搬到哪里',
    svg: unifiedThreeRoutes,
  },
  {
    slug: '09-video-generation/01-video-vae',
    afterHeading: /^## 1\. 视频数据账/,
    title: '时空压缩账：332 MB → 14.3 MB',
    svg: videoVaeCompression,
  },
  {
    slug: '09-video-generation/02-spatiotemporal-dit',
    afterHeading: /^## 2\. Attention 拓扑账/,
    title: 'full 3D vs factorized：95.2% 的边被砍掉',
    svg: attentionTopology,
  },
  {
    slug: '10-audio-speech/01-audio-representations',
    afterHeading: /^## 2\. STFT 与 mel/,
    title: 'STFT → mel：时频面的感知重排',
    svg: stftToMel,
  },
  {
    slug: '10-audio-speech/02-asr',
    afterHeading: /^### 2\.4 前向变量/,
    title: 'CTC：五条路径的边缘化 = 0.432',
    svg: ctcLattice,
  },
  {
    slug: '10-audio-speech/04-speech-llm-and-duplex',
    afterHeading: /^### 6\.2 延迟预算账/,
    title: '延迟时间轴：秒级级联 vs 2 帧全双工',
    svg: duplexLatency,
  },
];
