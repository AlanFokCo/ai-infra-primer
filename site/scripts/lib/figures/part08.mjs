// Part 8（08-multimodal-llm）场景图：5 章 14 块 ASCII → data-on-ack 式场景插图
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 part01.mjs。
// 注意：part08-10.mjs 的两张插入式签名图（8.1 三接入点、8.5 三路线）保留不动，本文件只做 replacesBlock。

import { scene, node, edge, zone, cap } from '../figure-kit.mjs';

// 与 scene() 内部 defs 相同的朱砂径向渐变，但 id 用本图前缀，供 edge(uid) 的脉冲解析
const pulseDefs = (uid) =>
  `<defs><radialGradient id="${uid}-pg" cx="35%" cy="35%" r="80%">` +
  `<stop offset="0%" stop-color="#c4503f"/><stop offset="100%" stop-color="#a83228"/>` +
  `</radialGradient></defs>`;

// 手工路径上的脉冲（kit 只支持自动贝塞尔；弧形/回流通用 path 直写）
const pathPulse = (d, { dur = 2.4, delay = 0, uid, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${uid}-pg)">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.42 0 0.58 1" path="${d}"/>` +
  `</circle>`;

/* ============================================================
 * 8.1 §2 — 组件链：vision encoder → connector → LLM（block 0）
 * 原文：x_img [H,W,3] → z_img [N,d_v]（N = HW/p²）→ v = φ(z_img) [M,d_llm] → LLM
 * ============================================================ */
function componentChain() {
  const UID = 'c81a';
  const body = [pulseDefs(UID)];
  const nodes = [];
  const X = 120, W = 210;

  body.push(cap(36, 28, '三段链（式 2）：v = φ(z_img)——继承两份预训练资产，新增训练只发生在接缝 φ', { size: 12, color: 'var(--ink-soft)' }));

  const defs = [
    { id: 'x', y: 40, h: 40, role: 'data', title: 'x_img [H, W, 3]', mono: true, note: '输入图像 · 像素空间' },
    { id: 'enc', y: 100, h: 44, role: 'attn', title: 'vision encoder', mono: true, sub: '视觉塔 · ViT 族（6.1）', note: '冻结或微调（§5）· 对比预训练资产（6.2）' },
    { id: 'z', y: 162, h: 40, role: 'data', title: 'z_img [N, d_v]', mono: true, note: 'N = HW/p²（6.1 式 3）· 进 LLM 只取 patch tokens' },
    { id: 'phi', y: 224, h: 44, role: 'proj', title: 'connector φ', mono: true, sub: 'd_v → d_llm · N → M', note: '式 (2)：换宽度、可改 token 数；家族归 8.2' },
    { id: 'v', y: 286, h: 40, role: 'data', title: 'v [M, d_llm]', mono: true, note: 'M：每图视觉 token 数——透传 M = N、瓶颈 M = B' },
    { id: 'llm', y: 348, h: 44, role: 'attn', title: 'LLM（语言塔）', sub: 'decoder-only · P2 全套复用', note: '接入三选一（§3）；多图/视频 = 同一形式推广' },
  ];
  const ns = defs.map((d) => {
    const n = node({ id: d.id, x: X, y: d.y, w: W, h: d.h, role: d.role, title: d.title, sub: d.sub, mono: d.mono ?? false, titleSize: 13 });
    nodes.push(n); body.push(n.svg);
    return { ...d, n };
  });

  // 接缝 φ 高亮（虚线分区；标注放左侧留白，避免压节点文字）
  body.push(zone({ x: 96, y: 216, w: 258, h: 60, tone: 'seal' }));
  body.push(cap(88, 240, '接缝 φ', { size: 10.5, anchor: 'end', color: 'var(--seal-deep)' }));
  body.push(cap(88, 256, '唯一必训组件', { size: 10.5, anchor: 'end', color: 'var(--seal-deep)' }));

  // 右侧注记
  ns.forEach((d) => body.push(cap(400, d.y + d.h / 2 + 4, d.note, { size: 10.5 })));

  // 竖向主链：脉冲逐段下行
  for (let i = 0; i < ns.length - 1; i++) {
    body.push(edge(ns[i].n, ns[i + 1].n, { nodes, portFrom: ns[i].n.bottom, portTo: ns[i + 1].n.top, pulse: true, dur: 1.6, delay: i * 0.35, uid: UID, eid: i, arrow: false }));
  }

  body.push(cap(36, 420, '摊销账：视觉塔吃 4 亿图文对、LLM 吃万亿 token；端到端单塔要把两份能力重学一遍（§2）', { size: 11 }));
  return scene(720, 436, body, { label: 'VLM 三段组件链：vision encoder 到 connector 到 LLM' }).svg();
}

/* ============================================================
 * 8.1 §3.1 — input-level 序列布局（block 1，式 3）
 * [sys tokens] [user 前半] [v_1…v_M] [user 后半] → 因果 LLM
 * ============================================================ */
function inputLevelLayout() {
  const UID = 'c81b';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 30, '式 (3)：v 当作 M 个「外来 token」的 embedding 行，展开到 <image> 占位符位置', { size: 12, color: 'var(--ink-soft)' }));

  const segs = [
    { id: 'sys', x: 40, w: 104, role: 'plain', title: 'sys tokens', mono: true },
    { id: 'u1', x: 156, w: 128, role: 'data', title: 'user prompt 前半', mono: true, ts: 11.5 },
    { id: 'v', x: 291, w: 160, role: 'proj', title: 'v₁ … v_M', mono: true, sub: 'M 个视觉 embedding 行', ss: 11.5 },
    { id: 'u2', x: 465, w: 128, role: 'data', title: 'user prompt 后半', mono: true, ts: 11.5 },
  ];
  const ns = defs2nodes();
  function defs2nodes() {
    return segs.map((s) => {
      const n = node({ id: s.id, x: s.x, y: 84, w: s.w, h: 46, role: s.role, title: s.title, sub: s.sub, mono: s.mono ?? false, titleSize: s.ts ?? 13, subSize: s.ss });
      nodes.push(n); body.push(n.svg);
      return n;
    });
  }

  // 占位符展开处标注（v 段上方）
  body.push(`<line class="dia-edge" x1="371" y1="66" x2="371" y2="82"/>`);
  body.push(cap(371, 60, '<image> 占位符展开处', { size: 10.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 因果 LLM 横贯
  const llm = node({ id: 'llm', x: 150, y: 190, w: 420, h: 52, role: 'attn', title: '因果 LLM', sub: '同一份 causal mask · 位置编码 · next-token loss' });
  nodes.push(llm); body.push(llm.svg);

  const targets = [200, 290, 380, 470];
  ns.forEach((n, i) => {
    body.push(edge(n, [targets[i], 190], { nodes, portFrom: n.bottom, pulse: true, dur: 1.8, delay: i * 0.2, uid: UID, eid: i, arrow: false }));
  });

  body.push(cap(36, 276, '对 LLM 而言，图像 = 一段「不在词表里、但形状合法的 embedding 行序列」（§3.1）', { size: 11 }));
  body.push(cap(36, 296, '代价：M 个 token 全额占 context 与 KV cache（§4）；LLM 必须参与训练（§5）', { size: 11 }));
  return scene(720, 312, body, { label: 'input-level 接入：视觉 embedding 展开进输入序列' }).svg();
}

/* ============================================================
 * 8.2 §3.1 — Q-Former 结构走读（block 0）
 * z_img 作 K/V（隔层）；Q 与文本 U 共享 self-attn；Q_out 投影拼进 LLM 前缀
 * ============================================================ */
function qformerFlow() {
  const UID = 'c82a';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(30, 26, 'Q-Former（BLIP-2）：N = 576 个 patch → 恒 32 个输出——不是映射而是取舍（式 3–7）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 24, y: 110, w: 190, h: 96, label: '冻结 ViT（梯度不回传）' }));
  const zimg = node({ id: 'z', x: 40, y: 134, w: 158, h: 48, role: 'data', title: 'z_img [N × d_v]', mono: true, sub: '冻结 ViT · N = 576', titleSize: 12.5, subSize: 11.5 });
  nodes.push(zimg); body.push(zimg.svg);

  body.push(zone({ x: 246, y: 40, w: 318, h: 316 }));
  body.push(cap(262, 350, 'Q-Former（约 1.9 亿参数）', { mono: true, size: 10.5 }));
  const q = node({ id: 'q', x: 300, y: 62, w: 160, h: 44, role: 'proj', title: 'Q：32 个可学 query', sub: '[32 × 768]', mono: false, titleSize: 12 });
  const sa = node({ id: 'sa', x: 280, y: 132, w: 240, h: 48, role: 'attn', title: '共享 self-attn [Q; U]', mono: true, sub: '参数共享 · mask 按目标切换', titleSize: 12.5 });
  const xa = node({ id: 'xa', x: 280, y: 212, w: 220, h: 48, role: 'op', title: 'XAttn(LN(Q′), z_img)', mono: true, sub: '隔层插入 · K/V 来自冻结 ViT', titleSize: 11.5, subSize: 11.5 });
  const qo = node({ id: 'qo', x: 280, y: 292, w: 150, h: 44, role: 'data', title: 'Q_out [32 × 768]', mono: true, titleSize: 12 });
  [q, sa, xa, qo].forEach((n) => { nodes.push(n); body.push(n.svg); });

  const u = node({ id: 'u', x: 580, y: 62, w: 120, h: 44, role: 'data', title: 'U [T × 768]', mono: true, sub: '文本 token', titleSize: 12.5 });
  const tout = node({ id: 'to', x: 576, y: 132, w: 128, h: 48, role: 'plain', title: '文本侧输出', sub: 'ITC / ITM / ITG' });
  const wp = node({ id: 'wp', x: 470, y: 292, w: 110, h: 44, role: 'proj', title: '线性投影', sub: '到 d_llm · stage 2', titleSize: 12.5, subSize: 10.5 });
  const v = node({ id: 'v', x: 596, y: 292, w: 112, h: 44, role: 'data', title: 'v [32 × d_llm]', mono: true, sub: '拼进 LLM 前缀', titleSize: 11, subSize: 10.5 });
  [u, tout, wp, v].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(q, sa, { nodes, portFrom: q.bottom, portTo: [380, 132], pulse: true, dur: 1.6, delay: 0.2, uid: UID, eid: 0, arrow: false }));
  body.push(edge(u, [480, 132], { nodes, portFrom: u.bottom, pulse: true, dur: 1.6, delay: 0.35, uid: UID, eid: 1, arrow: false }));
  body.push(edge(zimg, xa, { nodes, portFrom: zimg.right, portTo: xa.left, pulse: true, dur: 2.0, delay: 0.6, uid: UID, eid: 2 }));
  body.push(cap(206, 200, '作为 K/V（隔层）', { size: 10, color: 'var(--ink-faint)' }));
  body.push(edge(sa, xa, { nodes, portFrom: [360, 180], portTo: [360, 212], pulse: true, dur: 1.6, delay: 0.9, uid: UID, eid: 3, arrow: false }));
  body.push(edge(sa, tout, { nodes, pulse: true, dur: 1.6, delay: 0.7, uid: UID, eid: 4 }));
  body.push(edge(xa, qo, { nodes, portFrom: [390, 260], portTo: [355, 292], pulse: true, dur: 1.6, delay: 1.2, uid: UID, eid: 5, arrow: false }));
  body.push(edge(qo, wp, { nodes, pulse: true, dur: 1.4, delay: 1.4, uid: UID, eid: 6 }));
  body.push(edge(wp, v, { nodes, pulse: true, dur: 1.4, delay: 1.55, uid: UID, eid: 7 }));

  body.push(cap(30, 384, '「文本永远看不到图像特征」：唯一通道 = 32 × 768 瓶颈；取舍函数靠三目标从 129M 对学出（§3.4）', { size: 11 }));
  return scene(720, 400, body, { label: 'Q-Former：可学 query 瓶颈与共享 self-attn' }).svg();
}

/* ============================================================
 * 8.2 §3.2 — 三目标的注意力 mask（block 1）
 * ITC：Q/T 互不可见；ITM：全双向；ITG：queries 不见文本、文本 causal
 * ============================================================ */
function qformerMasks() {
  const UID = 'c82b';
  const body = [pulseDefs(UID)];
  body.push(cap(36, 28, '同一套参数服务三目标，只换共享 self-attn 的可见性 mask（论文 Figure 2 口径）', { size: 12, color: 'var(--ink-soft)' }));

  const CELL = 34, PITCH = 40;
  const panels = [
    { x: 64, title: 'ITC（对比）', on: (i, j) => i === j, tri: null, sub: ['queries 与文本互不可见'] },
    { x: 296, title: 'ITM（匹配）', on: () => true, tri: null, sub: ['全双向、互相可见'] },
    { x: 528, title: 'ITG（生成）', on: (i, j) => !(i === 0 && j === 1), tri: [1, 1], sub: ['queries 不见文本；', '文本见全部 queries + causal'] },
  ];
  panels.forEach((p, pi) => {
    body.push(cap(p.x, 46, p.title, { mono: true, size: 13, color: 'var(--ink)', weight: 600 }));
    const x0 = p.x + 44, y0 = 76;
    ['Q', 'T'].forEach((t, j) => body.push(cap(x0 + j * PITCH + CELL / 2, y0 - 8, t, { mono: true, size: 11, anchor: 'middle', color: 'var(--ink-soft)' })));
    ['Q', 'T'].forEach((t, i) => body.push(cap(x0 - 12, y0 + i * PITCH + 22, t, { mono: true, size: 11, anchor: 'end', color: 'var(--ink-soft)' })));
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const x = x0 + j * PITCH, y = y0 + i * PITCH;
        const delay = ((i * 2 + j) * 0.25 + pi * 0.4).toFixed(2);
        if (p.tri && p.tri[0] === i && p.tri[1] === j) {
          body.push(`<rect class="dia-mx-off" x="${x}" y="${y}" width="${CELL}" height="${CELL}" rx="4"/>`);
          body.push(`<polygon class="sig-cell" style="--i:${delay}" points="${x},${y} ${x + CELL},${y} ${x},${y + CELL}" fill="var(--ink)"/>`);
        } else {
          body.push(`<rect class="${p.on(i, j) ? 'dia-mx-on' : 'dia-mx-off'} sig-cell" style="--i:${delay}" x="${x}" y="${y}" width="${CELL}" height="${CELL}" rx="4"/>`);
        }
      }
    }
    p.sub.forEach((s, k) => body.push(cap(p.x, 172 + k * 15, p.sub[k], { size: 10 })));
  });

  body.push(cap(64, 224, '■ 可见 · □ 不可见 · ◤ 下三角 causal（2.4 式 1 惯例）', { size: 10.5 }));
  body.push(cap(64, 244, '读出端：ITC 取 max_k ⟨z̄_q⁽ᵏ⁾, z̄_t⟩（式 4/5）· ITM 逐对二分类（式 6）· ITG caption LM（式 7）', { size: 10.5 }));
  return scene(720, 264, body, { label: 'Q-Former 三目标的注意力可见性 mask' }).svg();
}

/* ============================================================
 * 8.2 §5.1 — pixel shuffle：2×2 邻域 → 通道 ×4 → 投影（block 2）
 * 24×24×d_v（N=576）→ 12×12×4d_v（无损 reshape）→ 144×d_llm
 * ============================================================ */
function pixelShuffle() {
  const UID = 'c82c';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 28, '式 (9)(10)：2×2 邻域沿通道拼接再投影——拼接无损、取舍集中在 W', { size: 12, color: 'var(--ink-soft)' }));

  // 左：24×24 网格（8×8 示意，每格 = 3×3 patch）
  body.push(cap(56, 52, 'z_img：24 × 24 × d_v（N = 576）', { mono: true, size: 10.5 }));
  const G = 8, CELL = 15, PITCH = 18, GX = 56, GY = 64;
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const hot = r >= 2 && r <= 3 && c >= 2 && c <= 3;
      body.push(`<rect class="${hot ? 'dia-mx-hot sig-cell' : ''}" ${hot ? `style="--i:${((r + c) * 0.1).toFixed(1)}"` : ''} x="${GX + c * PITCH}" y="${GY + r * PITCH}" width="${CELL}" height="${CELL}" rx="2.5" ${hot ? '' : 'fill="var(--paper-sunk)" stroke="var(--rule)" stroke-width="1"'}/>`);
    }
  }
  body.push(cap(GX + 4 * PITCH, 236, 'N = 576', { mono: true, size: 10.5, anchor: 'middle' }));

  // 中：2×2 邻域 → 4 个特征 → 通道拼接
  const chips = ['z_{2i, 2j}', 'z_{2i, 2j+1}', 'z_{2i+1, 2j}', 'z_{2i+1, 2j+1}'].map((t, i) => {
    const n = node({ id: `c${i}`, x: 250, y: 56 + i * 30, w: 150, h: 24, role: 'data', title: t, mono: true, titleSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(edge([128, 118], chips[0], { nodes, portTo: chips[0].left, dash: true, arrow: false, uid: UID, eid: 'p' }));
  body.push(cap(168, 96, '2×2 邻域', { size: 10 }));

  const cat = node({ id: 'cat', x: 430, y: 84, w: 140, h: 56, role: 'op', title: '通道维拼接', sub: '12×12 × 4d_v · 无损', titleSize: 12.5, subSize: 10.5 });
  const wproj = node({ id: 'w', x: 442, y: 170, w: 116, h: 48, role: 'proj', title: 'W: 4d_v→d_llm', mono: true, sub: '唯一可训处', titleSize: 11.5 });
  [cat, wproj].forEach((n) => { nodes.push(n); body.push(n.svg); });
  chips.forEach((c, i) => body.push(edge(c, [434, 112], { nodes, portFrom: c.right, pulse: true, dur: 1.4, delay: 0.2 + i * 0.15, uid: UID, eid: `s${i}`, arrow: false })));
  body.push(edge(cat, wproj, { nodes, portFrom: cat.bottom, portTo: wproj.top, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 'cw', arrow: false }));

  // 右：12×12 网格（6×6 示意）
  body.push(cap(576, 52, 'v：12 × 12（M = 144）', { mono: true, size: 10.5 }));
  const RX = 576, RY = 64;
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      body.push(`<rect class="sig-cell" style="--i:${(1 + (r + c) * 0.12).toFixed(2)}" x="${RX + c * PITCH}" y="${RY + r * PITCH}" width="${CELL}" height="${CELL}" rx="2.5" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1"/>`);
    }
  }
  body.push(cap(RX + 3 * PITCH, 236, 'M = 576/4 = 144', { mono: true, size: 10.5, anchor: 'middle' }));
  body.push(edge(wproj, [RX, 118], { nodes, portFrom: wproj.right, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'wv', arrow: false }));

  body.push(cap(56, 272, 'token ×r²（÷4）、通道 ×4 互抵守恒；LLM 端 attention 二次项 ×1/16、逐 token 线性项 ×1/4', { size: 11 }));
  body.push(cap(56, 292, 'InternVL pixel_shuffle(0.5) · Qwen spatial_merge = 2（等效 patch 14 → 28）', { size: 10.5 }));
  return scene(720, 310, body, { label: 'pixel shuffle：token 数除以 4、通道乘 4' }).svg();
}

/* ============================================================
 * 8.2 §7 — AnyRes tiling（block 3）
 * 选网格（式 11 字典序）→ 切 k 块 + 缩略图 → (k+1)·576 tokens 进 LLM
 * ============================================================ */
function anyresTiling() {
  const UID = 'c82d';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(30, 26, 'AnyRes（LLaVA-NeXT → OneVision）：不硬 resize 整图——切块保局部全分辨率 + 缩略图保全局', { size: 12, color: 'var(--ink-soft)' }));

  const src = node({ id: 'src', x: 24, y: 96, w: 104, h: 56, role: 'data', title: '原图', sub: '1568 × 1176', mono: true, titleSize: 13, subSize: 10.5 });
  const pick = node({ id: 'pick', x: 148, y: 96, w: 140, h: 56, role: 'op', title: '选网格（式 11）', sub: '先 max e 平局 min ρ', titleSize: 12, subSize: 10.5 });
  nodes.push(src, pick); body.push(src.svg, pick.svg);

  body.push(zone({ x: 308, y: 28, w: 266, h: 232, label: 'k = 4 块 + 1 缩略图' }));
  const tilePos = [[322, 52], [446, 52], [322, 116], [446, 116]];
  const tiles = tilePos.map(([x, y], i) => {
    const n = node({ id: `t${i}`, x, y, w: 110, h: 48, role: 'data', title: `块 ${i + 1}`, sub: '336×336 → 576', mono: true, titleSize: 12.5, subSize: 10.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const thumb = node({ id: 'th', x: 322, y: 180, w: 234, h: 44, role: 'cache', title: '缩略图：整图 → 336×336', sub: '576 tokens（全局）', titleSize: 12 });
  nodes.push(thumb); body.push(thumb.svg);
  body.push(cap(322, 246, '每块独立过同一 ViT · 块内原生 24×24 PE', { size: 9.5 }));

  const vseq = node({ id: 'v', x: 584, y: 72, w: 120, h: 52, role: 'proj', title: 'v 序列', sub: '(k+1)·N = 2880', mono: true, titleSize: 13, subSize: 10.5 });
  const llm = node({ id: 'llm', x: 584, y: 168, w: 120, h: 52, role: 'attn', title: 'LLM', sub: '视觉段 2880', titleSize: 13, subSize: 10.5 });
  nodes.push(vseq, llm); body.push(vseq.svg, llm.svg);

  body.push(edge(src, pick, { nodes, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(pick, tiles[0], { nodes, portFrom: pick.right, portTo: tiles[0].left, pulse: true, dur: 1.5, delay: 0.3, uid: UID, eid: 1 }));
  body.push(cap(297, 64, '切 k 块', { size: 9.5, anchor: 'middle' }));
  // 缩略图支路：原图整图缩放（虚线，避让选网格）
  body.push(edge(src, thumb, { nodes, portFrom: src.bottom, portTo: thumb.left, dash: true, arrow: false, uid: UID, eid: 2 }));
  body.push(cap(150, 200, '整图缩放', { size: 9.5 }));
  const vy = [86, 96, 106, 116];
  tiles.forEach((t, i) => body.push(edge(t, [584, vy[i]], { nodes, portFrom: t.right, pulse: true, dur: 1.5, delay: 0.6 + i * 0.12, uid: UID, eid: `v${i}`, arrow: false })));
  body.push(edge(thumb, [584, 124], { nodes, portFrom: thumb.right, pulse: true, dur: 1.5, delay: 1.1, uid: UID, eid: 'vt', arrow: false }));
  body.push(edge(vseq, llm, { nodes, portFrom: vseq.bottom, portTo: llm.top, pulse: true, dur: 1.5, delay: 1.3, uid: UID, eid: 8, arrow: false }));

  body.push(cap(30, 288, 'M = (k+1)·N = 5 × 576 = 2880（式 12）；视觉塔前向 ×(k+1)、二次项按 2880 平方放大', { size: 11 }));
  body.push(cap(30, 308, 'raster 序拼接、行间可学 newline 向量；LLaVA-OneVision 换 SigLIP-384：5 × 729 = 3645', { size: 10.5 }));
  return scene(720, 324, body, { label: 'AnyRes：切块加缩略图，token 账 (k+1)·576' }).svg();
}

/* ============================================================
 * 8.2 §9.2 — M-RoPE 频段切分 [16, 24, 24]（block 4）
 * head 维 128 = 64 个 2D 旋转块；t 段高频、h/w 段低频长波长
 * ============================================================ */
function mropeSections() {
  const UID = 'c82e';
  const body = [pulseDefs(UID)];

  body.push(cap(40, 28, 'head 维 128 = 64 个 2D 旋转块（角度 = 轴坐标 × θ_j）；mrope_section = [16, 24, 24]（config 实测）', { size: 12, color: 'var(--ink-soft)' }));

  const X0 = 60, PITCH = 9.375, CW = 8, Y = 92, H = 36;
  const segs = [
    { j0: 0, j1: 15, cls: 'dia-mx-hot', name: 't 段 · j = 0…15（16 块）', angle: '角度 t·θ_j（最高频）', lam: 'λ：2π → ≈1.6×10²', color: 'var(--seal-deep)' },
    { j0: 16, j1: 39, cls: 'dia-mx-on', name: 'h 段 · j = 16…39（24 块）', angle: '角度 h·θ_j', lam: 'λ：≈2.0×10² → ≈2.8×10⁴', color: 'var(--ink)' },
    { j0: 40, j1: 63, cls: 'dia-mx-warm', name: 'w 段 · j = 40…63（24 块）', angle: '角度 w·θ_j（最低频）', lam: 'λ：≈3.5×10⁴ → ≈5.1×10⁶', color: 'var(--ink-soft)' },
  ];
  segs.forEach((s) => {
    const xa = X0 + s.j0 * PITCH, xb = X0 + (s.j1 + 1) * PITCH;
    // 段括号
    body.push(`<line class="dia-edge" x1="${xa}" y1="86" x2="${xb - 2}" y2="86"/>`);
    body.push(`<line class="dia-edge" x1="${xa}" y1="82" x2="${xa}" y2="90"/>`);
    body.push(`<line class="dia-edge" x1="${xb - 2}" y1="82" x2="${xb - 2}" y2="90"/>`);
    body.push(cap((xa + xb) / 2, 74, s.name, { mono: true, size: 10.5, anchor: 'middle', color: s.color }));
    body.push(cap((xa + xb) / 2, 158, s.angle, { size: 10, anchor: 'middle' }));
    body.push(cap((xa + xb) / 2, 176, s.lam, { mono: true, size: 10, anchor: 'middle', color: 'var(--ink-faint)' }));
  });
  for (let j = 0; j < 64; j++) {
    const seg = j < 16 ? segs[0] : j < 40 ? segs[1] : segs[2];
    body.push(`<rect class="${seg.cls} sig-cell" style="--i:${(j * 0.03).toFixed(2)}" x="${(X0 + j * PITCH).toFixed(1)}" y="${Y}" width="${CW}" height="${H}" rx="2"/>`);
  }
  // j 边界刻度
  [[0, '0'], [16, '16'], [40, '40'], [64, '63']].forEach(([j, label]) => {
    const x = j === 64 ? X0 + 64 * PITCH - 2 : X0 + j * PITCH;
    body.push(`<line class="dia-edge" x1="${x}" y1="${Y + H}" x2="${x}" y2="${Y + H + 5}"/>`);
    body.push(cap(x, Y + H + 16, label, { mono: true, size: 9, anchor: j === 64 ? 'end' : 'middle' }));
  });

  body.push(cap(60, 204, 'λ_j = 2π·base^{2j/d_h} = 2π·10^{3j/32}（base = 10⁶，d_h = 128）；三段共用同一条几何级数', { mono: true, size: 10.5 }));
  body.push(cap(60, 224, '读法：空间两轴拿低频长波长 = 全局坐标；t 轴 16 个高频通道承局部时间次序（§9.2）', { size: 10.5 }));
  return scene(720, 240, body, { label: 'M-RoPE：64 个旋转块按 t/h/w 三段切分' }).svg();
}

/* ============================================================
 * 8.3 §1.1 — loss masking：loss 只落在 response（block 0，式 1）
 * [system][user 前半][v_1…v_M][user 后半] = 条件段；[assistant] m_s = 1
 * ============================================================ */
function lossMasking() {
  const UID = 'c83a';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(36, 28, '式 (1)：L = −Σ_s m_s·log p(x_s | x_<s)——VLM 没有新 loss，全部新东西在 m_s 怎么取', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 32, y: 62, w: 496, h: 88, label: '条件段：建 KV、供 attend，不计 loss' }));
  body.push(zone({ x: 538, y: 62, w: 150, h: 88, label: 'response：计 loss', tone: 'seal' }));

  const segs = [
    { id: 'sys', x: 42, w: 96, role: 'plain', title: 'system', mono: true, m: 'm_s = 0', hot: false },
    { id: 'u1', x: 148, w: 110, role: 'data', title: 'user 前半', mono: true, m: 'm_s = 0', hot: false },
    { id: 'v', x: 268, w: 130, role: 'proj', title: 'v₁ … v_M', mono: true, m: 'm_s = 0', hot: false },
    { id: 'u2', x: 408, w: 110, role: 'data', title: 'user 后半', mono: true, m: 'm_s = 0', hot: false },
    { id: 'resp', x: 528, w: 150, role: 'attn', title: 'assistant 回答', m: 'm_s = 1', hot: true },
  ];
  segs.forEach((s) => {
    const n = node({ id: s.id, x: s.x, y: 96, w: s.w, h: 44, role: s.role, title: s.title, mono: s.mono ?? false, titleSize: 12.5 });
    nodes.push(n); body.push(n.svg);
    body.push(cap(s.x + s.w / 2, 162, s.m, { mono: true, size: 10.5, anchor: 'middle', color: s.hot ? 'var(--seal-deep)' : 'var(--ink-faint)', weight: s.hot ? 600 : undefined }));
  });

  // 梯度回流：mask 不挡梯度——response 的 loss 沿 attention 回传到 connector
  const back = 'M 603 96 C 603 42, 333 42, 333 92';
  body.push(`<path class="dia-edge dash" d="${back}"/>`);
  body.push(`<polygon points="333,96 329,88 337,88" fill="var(--ink-soft)"/>`);
  body.push(cap(468, 48, '梯度沿 attention 回传到 connector——mask 只作用于 loss（§1.2）', { size: 10.5, anchor: 'middle', color: 'var(--ink-soft)' }));

  body.push(cap(36, 192, 'labels 填 −100：视觉 token 无 ground truth（softmax 给不出标签）、prompt 是条件非目标', { size: 11 }));
  body.push(cap(36, 212, '被 mask 位置不省 forward：照常过全部 L 层、建 KV cache、被 response attend（§1.2）', { size: 11 }));
  body.push(cap(36, 234, 'η = Σm_s/S ≈ 150/3530 ≈ 4%（AnyRes 4 切 + 短回答）⇒ packing 与 dense caption 提 η（§1.3）', { size: 10.5, color: 'var(--seal-deep)' }));
  return scene(720, 252, body, { label: 'loss masking：只有 response 位置计 loss' }).svg();
}

/* ============================================================
 * 8.3 §2 — 三阶段配方：对齐 → 指令 → 偏好（block 1）
 * ============================================================ */
function stageRecipe() {
  const UID = 'c83b';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 26, '同一个式 (1) 跑三个阶段：换数据、换冻结开关（LLaVA 系 / Qwen2.5-VL / Molmo 的共同骨架）', { size: 12, color: 'var(--ink-soft)' }));

  const stages = [
    {
      x: 24, w: 216, label: 'stage 1 · 对齐预训练',
      chips: [{ t: 'ViT 冻', r: 'cache', w: 56 }, { t: 'LLM 冻', r: 'cache', w: 62 }, { t: 'φ 训', r: 'proj', w: 56 }],
      lines: ['只训 connector：梯度只经 v 回传', '数据：caption 对（558K 条）', '学空间对齐，不学内容补全', '产物：会看图写 caption、指令为零'],
    },
    {
      x: 264, w: 216, label: 'stage 2 · 指令微调',
      chips: [{ t: 'ViT 冻', r: 'cache', w: 56 }, { t: 'LLM 训', r: 'proj', w: 62 }, { t: 'φ 训', r: 'proj', w: 56 }],
      lines: ['解冻 LLM + connector', '数据：多任务指令混合（665K）', '必含一份纯文本——防遗忘', '模板与 special token 定型（§3.3）'],
    },
    {
      x: 504, w: 200, label: 'stage 3 · 偏好对齐',
      chips: [{ t: '参数面同 stage 2', r: 'plain', w: 130 }],
      lines: ['数据：chosen / rejected 对', '图像即锚点：核对对象存在性', '负样本：注入不存在对象/错属性', '治理「先验压过视觉证据」（§4）'],
    },
  ];
  stages.forEach((s) => {
    body.push(zone({ x: s.x, y: 40, w: s.w, h: 196, label: s.label }));
    let cx = s.x + 12;
    s.chips.forEach((c) => {
      const n = node({ id: `${s.label}${c.t}`, x: cx, y: 96, w: c.w, h: 30, role: c.r, title: c.t, titleSize: 10.5 });
      body.push(n.svg);
      cx += c.w + 6;
    });
    s.lines.forEach((l, i) => body.push(cap(s.x + 12, 150 + i * 18, l, { size: 10 })));
  });

  body.push(edge([240, 138], [264, 138], { pulse: true, dur: 1.6, delay: 0.2, uid: UID, eid: 0 }));
  body.push(edge([480, 138], [504, 138], { pulse: true, dur: 1.6, delay: 0.5, uid: UID, eid: 1 }));

  body.push(cap(36, 262, '先对齐空间 → 再学任务格式 → 最后校准行为偏好：冻结逐段放开、数据从粗到精（§2.4）', { size: 11 }));
  return scene(720, 278, body, { label: '三阶段训练配方：对齐、指令微调、偏好对齐' }).svg();
}

/* ============================================================
 * 8.4 §1/§4 — 视频管线：token 省在三个位置（block 0）
 * 视频 → 帧采样 → F 帧 → [位置一 3D patch] → ViT → [位置二 pooling] → [位置三 resampler] → LLM
 * ============================================================ */
function videoPipeline() {
  const UID = 'c84a';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(30, 26, '总账：S_vid = F · M（式 1）；1 min @ 1 fps × 256 = 15,360——压缩三位置，差别在谁的前向被省掉', { size: 12, color: 'var(--ink-soft)' }));

  const spine = [
    { id: 'vid', y: 40, h: 38, role: 'data', title: '视频 T 秒', mono: true },
    { id: 'samp', y: 100, h: 42, role: 'op', title: '帧采样（§2）', sub: 'uniform / dynamic FPS', mono: true, ts: 12.5, ss: 11.5 },
    { id: 'rgb', y: 164, h: 40, role: 'data', title: 'F 帧 RGB', mono: true, sub: 'F ≈ T·fps', ts: 12.5 },
    { id: 'vit', y: 246, h: 42, role: 'attn', title: 'ViT', mono: true, sub: '序列 (F/τ_p)·N', ts: 13 },
    { id: 'tok', y: 310, h: 40, role: 'data', title: '特征 token', sub: 'F/c_t × M 个', mono: true, ts: 12.5 },
    { id: 'llm', y: 378, h: 44, role: 'attn', title: 'LLM', mono: true, sub: 'context 硬预算 128K', ts: 13, ss: 11.5 },
  ];
  const ns = spine.map((s) => {
    const n = node({ id: s.id, x: 50, y: s.y, w: 190, h: s.h, role: s.role, title: s.title, sub: s.sub, mono: s.mono ?? false, titleSize: s.ts ?? 12.5, subSize: s.ss });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  for (let i = 0; i < ns.length - 1; i++) {
    body.push(edge(ns[i], ns[i + 1], { nodes, portFrom: ns[i].bottom, portTo: ns[i + 1].top, pulse: true, dur: 1.5, delay: i * 0.3, uid: UID, eid: i, arrow: false }));
  }

  const cards = [
    { y: 193, seg: 224, title: '位置一：输入侧 3D patch（τ_p = 2，式 6）', line: 'τ_p 帧合一的无损重排：差分进通道，ViT 序列 ÷τ_p——encoder 侧就省' },
    { y: 268, seg: 299, title: '位置二：encoder 后时空 pooling ÷c_t（式 7）', line: 'encoder 成本照付；LLM 侧 ÷c_t、attention 二次项 ÷c_t²（LLaVA-OneVision）' },
    { y: 332, seg: 363, title: '位置三：temporal queries resampler → 定长 B（§4.3）', line: '瓶颈取舍可学；视觉段长度与片长解耦——长视频最激进方案（Flamingo）' },
  ];
  cards.forEach((c, i) => {
    body.push(zone({ x: 290, y: c.y, w: 414, h: 62, tone: 'seal' }));
    body.push(cap(302, c.y + 22, c.title, { size: 11.5, color: 'var(--ink)', weight: 600 }));
    body.push(cap(302, c.y + 44, c.line, { size: 10.5 }));
    body.push(edge([240, c.seg], [290, c.seg], { dash: true, arrow: false, uid: UID, eid: `c${i}` }));
  });

  body.push(cap(30, 448, '+ (t,h,w) 三元组进 LLM（§3）：τ_f = round(t_f·ν)，ν = 2——采样与编码由「秒」咬合', { size: 11 }));
  return scene(720, 466, body, { label: '视频理解管线与三个时间压缩位置' }).svg();
}

/* ============================================================
 * 8.5 §4.1 — Janus：解耦双 encoder + 共享 AR 主干（block 0）
 * 理解：x → SigLIP → MLP → v ┐；生成：x → VQ enc → code ┘→ 共享主干
 * ============================================================ */
function janusDecoupled() {
  const UID = 'c85a';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(30, 26, '理解要不变性、生成要充分性（式 8/9）——解耦编码器、共享主干', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 20, y: 40, w: 330, h: 104, label: '理解方向（连续特征）' }));
  const x = node({ id: 'x', x: 36, y: 78, w: 70, h: 36, role: 'data', title: 'x', mono: true });
  const sig = node({ id: 'sig', x: 126, y: 74, w: 100, h: 44, role: 'attn', title: 'SigLIP 塔', sub: '连续特征', titleSize: 12.5 });
  const mlp = node({ id: 'mlp', x: 246, y: 74, w: 70, h: 44, role: 'proj', title: 'MLP', mono: true, titleSize: 12.5 });
  [x, sig, mlp].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(x, sig, { nodes, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(sig, mlp, { nodes, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 1 }));

  body.push(zone({ x: 20, y: 164, w: 330, h: 130, label: '生成方向（离散 code）' }));
  const vqe = node({ id: 'vqe', x: 36, y: 226, w: 100, h: 44, role: 'op', title: 'VQ encoder', mono: true, titleSize: 11.5 });
  const code = node({ id: 'code', x: 156, y: 226, w: 112, h: 44, role: 'cache', title: '图像 code', sub: '训练时作序列段', titleSize: 12.5, subSize: 10.5 });
  [vqe, code].forEach((n) => { nodes.push(n); body.push(n.svg); });
  body.push(edge(vqe, code, { nodes, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 2 }));
  body.push(edge(x, vqe, { nodes, portFrom: x.bottom, portTo: vqe.top, dash: true, arrow: false, uid: UID, eid: 3 }));

  const trunk = node({ id: 'trunk', x: 430, y: 110, w: 130, h: 64, role: 'attn', title: '共享 AR 主干', sub: '统一词表 · 单 softmax', titleSize: 13, subSize: 11.5 });
  const out = node({ id: 'out', x: 592, y: 110, w: 112, h: 64, role: 'data', title: '文本/图像 code', mono: true, sub: 'AR 产出', titleSize: 10.5 });
  const vqd = node({ id: 'vqd', x: 600, y: 206, w: 100, h: 44, role: 'op', title: 'VQ decoder', mono: true, titleSize: 11.5 });
  const xhat = node({ id: 'xhat', x: 600, y: 266, w: 100, h: 36, role: 'data', title: 'x̂ 像素', mono: true, titleSize: 12 });
  [trunk, out, vqd, xhat].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(edge(mlp, [455, 110], { nodes, portFrom: mlp.right, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 4, arrow: false }));
  body.push(cap(372, 86, 'v', { mono: true, size: 11, color: 'var(--ink-soft)' }));
  body.push(edge(code, [455, 174], { nodes, portFrom: code.right, pulse: true, dur: 1.6, delay: 0.4, uid: UID, eid: 5, arrow: false }));
  body.push(cap(262, 286, 'code 作序列段', { size: 9.5 }));
  body.push(edge(trunk, out, { nodes, pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 6 }));
  body.push(edge(out, vqd, { nodes, portFrom: out.bottom, portTo: vqd.top, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 7, arrow: false }));
  body.push(edge(vqd, xhat, { nodes, portFrom: vqd.bottom, portTo: xhat.top, pulse: true, dur: 1.4, delay: 1.1, uid: UID, eid: 8, arrow: false }));

  body.push(cap(30, 326, '代价：接口账部分回归——两个 encoder + 各自 connector/tokenizer；统一范围收缩到主干（Janus-1.3B / Pro-7B）', { size: 11 }));
  return scene(720, 342, body, { label: 'Janus：理解连续、生成离散，解耦编码器共享主干' }).svg();
}

/* ============================================================
 * 8.5 §6.1 — Transfusion 序列布局（block 1，式 12/14）
 * [text][BOI][x̃ 连续 patch][EOI][text]；文本位 CE、patch 位 ε 回归
 * ============================================================ */
function transfusionSequence() {
  const UID = 'c85b';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(40, 28, '式 (12)：文本 embedding 与加噪 patch latent 混排同一条序列；BOI/EOI 是普通 special token', { size: 12, color: 'var(--ink-soft)' }));

  const chips = [
    { t: 'E(s_a)', x: 46, w: 56, role: 'data' },
    { t: 'E(s)', x: 110, w: 56, role: 'data' },
    { t: 'BOI', x: 174, w: 52, role: 'cache' },
    { t: 'x̃⁽¹⁾', x: 234, w: 64, role: 'proj' },
    { t: 'x̃⁽²⁾', x: 306, w: 64, role: 'proj' },
    { t: 'x̃⁽³⁾', x: 378, w: 64, role: 'proj' },
    { t: '…', x: 450, w: 36, role: 'plain' },
    { t: 'EOI', x: 494, w: 52, role: 'cache' },
    { t: 'E(s)', x: 554, w: 56, role: 'data' },
    { t: 'E(s)', x: 618, w: 56, role: 'data' },
  ];
  chips.forEach((c, i) => {
    const n = node({ id: `c${i}`, x: c.x, y: 84, w: c.w, h: 40, role: c.role, title: c.t, mono: true, titleSize: 12 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(40, 66, '离散位置', { size: 10 }));
  body.push(cap(360, 72, '连续 patch 位置（式 13 前向加噪，同图共享一个 t）', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));

  // loss 归属括号
  const bracket = (x1, x2, y) =>
    `<line class="dia-edge strong" x1="${x1}" y1="${y}" x2="${x2}" y2="${y}"/>` +
    `<line class="dia-edge strong" x1="${x1}" y1="${y}" x2="${x1}" y2="${y - 6}"/>` +
    `<line class="dia-edge strong" x1="${x2}" y1="${y}" x2="${x2}" y2="${y - 6}"/>`;
  body.push(bracket(46, 226, 150));
  body.push(bracket(234, 486, 150));
  body.push(bracket(494, 674, 150));
  body.push(cap(136, 170, 'softmax 头 + CE', { size: 10, anchor: 'middle' }));
  body.push(cap(360, 170, '噪声头 + ε 回归', { size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(584, 170, 'CE（EOI 协议追加不计）', { size: 10, anchor: 'middle' }));

  const loss = node({ id: 'loss', x: 230, y: 204, w: 260, h: 46, role: 'attn', title: 'L = L_LM + λ·L_DDPM（式 14）', mono: true, titleSize: 12.5 });
  nodes.push(loss); body.push(loss.svg);
  [[136, 300], [360, 360], [584, 420]].forEach(([fx, tx], i) => {
    body.push(edge([fx, 156], [tx, 204], { uid: UID, eid: `l${i}`, arrow: false }));
  });

  body.push(cap(40, 282, '同一次前向：位置类型不相交——文本位走 softmax 头、patch 位走噪声头，两头不共享参数', { size: 11 }));
  body.push(cap(40, 302, '梯度只交汇于主干，λ 配平两 loss 量级（§6.2）；patch = 连续 VAE latent + t embedding', { size: 11 }));
  return scene(720, 318, body, { label: 'Transfusion 混合序列：文本 CE 与 patch 噪声回归' }).svg();
}

/* ============================================================
 * 8.5 §6.3 — Transfusion mask：因果 + 块内双向（block 2，式 15/16）
 * 8 位置实例 [w1 w2 BOI p1 p2 p3 EOI w3]
 * ============================================================ */
function transfusionMask() {
  const UID = 'c85c';
  const body = [pulseDefs(UID)];

  body.push(cap(36, 26, '式 (15)：M_ij = 1 ⟺ j ≤ i（因果）或 i,j 同属一个图像块 B_k——去噪要全图视野、LM 要因果', { size: 12, color: 'var(--ink-soft)' }));

  const labels = ['w1', 'w2', 'BOI', 'p1', 'p2', 'p3', 'EOI', 'w3'];
  const N = 8, CELL = 24, PITCH = 27, X0 = 100, Y0 = 64;
  labels.forEach((t, j) => body.push(cap(X0 + j * PITCH + CELL / 2, Y0 - 10, t, { mono: true, size: 10, anchor: 'middle', color: 'var(--ink-soft)' })));
  labels.forEach((t, i) => body.push(cap(X0 - 10, Y0 + i * PITCH + 17, t, { mono: true, size: 10, anchor: 'end', color: 'var(--ink-soft)' })));
  const inBlock = (k) => k >= 3 && k <= 5;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const causal = j <= i;
      const extra = inBlock(i) && inBlock(j) && j > i;
      if (!causal && !extra) {
        body.push(`<rect class="dia-mx-off" x="${X0 + j * PITCH}" y="${Y0 + i * PITCH}" width="${CELL}" height="${CELL}" rx="3"/>`);
      } else {
        body.push(`<rect class="${extra ? 'dia-mx-hot' : 'dia-mx-on'} sig-cell" style="--i:${((i + j) * 0.05).toFixed(2)}" x="${X0 + j * PITCH}" y="${Y0 + i * PITCH}" width="${CELL}" height="${CELL}" rx="3"/>`);
      }
    }
  }
  body.push(`<rect class="dia-zone dia-zone-seal" x="178" y="142" width="87" height="87" rx="6"/>`);

  // 右侧面板：规则与无泄漏命题
  body.push(cap(384, 64, '行 = query · 列 = key（8 位置实例）', { size: 11.5, color: 'var(--ink)', weight: 600 }));
  body.push(`<rect class="dia-mx-on" x="384" y="80" width="12" height="12" rx="3"/>`);
  body.push(cap(404, 90, '因果项：一切位置可见历史', { size: 10.5 }));
  body.push(`<rect class="dia-mx-hot" x="384" y="106" width="12" height="12" rx="3"/>`);
  body.push(cap(404, 116, '块内项新增：同图 patch 互相全见', { size: 10.5 }));
  body.push(cap(384, 140, 'BOI / EOI 按文本位置处理（只走因果项）', { size: 10.5 }));
  body.push(`<line class="dia-edge" x1="384" y1="158" x2="700" y2="158"/>`);
  body.push(cap(384, 178, '块内双向为什么不泄漏标签（式 16）', { size: 11.5, color: 'var(--ink)', weight: 600 }));
  body.push(cap(384, 198, '各 patch 噪声独立采样：ε⁽ʲ⁾ ⊥ ε⁽ⁱ⁾', { size: 10.5 }));
  body.push(cap(384, 216, 'H(ε⁽ⁱ⁾ | {x_t⁽ʲ⁾}_j≠i, prefix) = H(ε⁽ⁱ⁾)', { mono: true, size: 10.5 }));
  body.push(cap(384, 234, '可见的是加噪未来——可见内容里不含答案', { size: 10.5 }));

  body.push(cap(36, 304, 'Emu3 严格 AR：目标是干净 id，见未来 = 泄漏答案本身', { size: 10.5 }));
  body.push(cap(36, 322, 'Show-o：可见干净 id 是条件、目标是被掩位置（BERT 式合法）· Transfusion：见加噪未来、目标是噪声', { size: 10.5 }));
  return scene(720, 338, body, { label: 'Transfusion attention mask：因果加图像块内双向' }).svg();
}

/* ============================================================
 * 8.5 §6.4 — Transfusion 生图回路（block 3）
 * 文本 AR → BOI → 追加噪声 patch → T 步主干内去噪 → EOI → VAE decoder
 * ============================================================ */
function transfusionLoop() {
  const UID = 'c85d';
  const body = [pulseDefs(UID)];
  const nodes = [];

  body.push(cap(30, 26, '生图回路（§6.4）：文本前缀 KV cache 全程复用；去噪在主干内跑——无独立 U-Net 塔', { size: 12, color: 'var(--ink-soft)' }));

  const s1 = node({ id: 's1', x: 24, y: 108, w: 124, h: 52, role: 'data', title: '① 文本 AR decode', sub: '直到 BOI · KV 复用', titleSize: 11.5, subSize: 10.5 });
  const s2 = node({ id: 's2', x: 172, y: 108, w: 124, h: 52, role: 'data', title: '② 追加 n 个 patch 位', sub: '纯噪声（t = 1 端）', titleSize: 11, subSize: 10.5 });
  const s3 = node({ id: 's3', x: 330, y: 108, w: 124, h: 52, role: 'attn', title: '主干前向 → ε_θ', sub: '改写 patch · 块内双向', titleSize: 11.5, subSize: 10.5 });
  const s4 = node({ id: 's4', x: 584, y: 108, w: 112, h: 52, role: 'op', title: '④ 追加 EOI', sub: '机械追加·切回 LM', titleSize: 11.5, subSize: 10.5 });
  const s5 = node({ id: 's5', x: 580, y: 196, w: 116, h: 48, role: 'proj', title: '⑤ VAE decoder', mono: true, sub: '→ 像素；文本续 AR', titleSize: 11, subSize: 10.5 });
  [s1, s2, s3, s4, s5].forEach((n) => { nodes.push(n); body.push(n.svg); });

  body.push(zone({ x: 312, y: 72, w: 248, h: 148, label: '③ T 步去噪循环（无 cache）', tone: 'seal' }));

  body.push(edge(s1, s2, { nodes, pulse: true, dur: 1.5, delay: 0, uid: UID, eid: 0 }));
  body.push(edge(s2, s3, { nodes, pulse: true, dur: 1.5, delay: 0.3, uid: UID, eid: 1 }));
  body.push(edge(s3, s4, { nodes, pulse: true, dur: 2.2, delay: 1.4, uid: UID, eid: 2 }));
  body.push(edge(s4, s5, { nodes, portFrom: s4.bottom, portTo: s5.top, pulse: true, dur: 1.5, delay: 1.7, uid: UID, eid: 3, arrow: false }));

  // 去噪回流：每步反向更新改写 patch 输入（无 cache，全量前向）
  const loop = 'M 454 146 C 505 196, 279 196, 330 146';
  body.push(`<path class="dia-edge dash" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 2.2, delay: 0.8, uid: UID, r: 3.6 }));
  body.push(cap(392, 210, '每步全量前向 ×T（数十步档）', { size: 10, anchor: 'middle' }));

  body.push(cap(30, 254, '成本（式 17）：C_TF ≈ S₀ + T·n（深度 T、每步宽 n）vs C_AR ≈ S₀ + N（深度 N，Emu3 档 4096）', { size: 11 }));
  return scene(720, 270, body, { label: 'Transfusion 生图：T 步主干内去噪循环' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { slug: '08-multimodal-llm/01-vlm-architecture-taxonomy', replacesBlock: 0, title: '组件链：vision encoder → connector → LLM', svg: componentChain },
  { slug: '08-multimodal-llm/01-vlm-architecture-taxonomy', replacesBlock: 1, title: 'input-level 拼接：v 展开进输入序列（式 3）', svg: inputLevelLayout },
  { slug: '08-multimodal-llm/02-connectors-and-resolution', replacesBlock: 0, title: 'Q-Former：32 个可学 query 的定长瓶颈', svg: qformerFlow },
  { slug: '08-multimodal-llm/02-connectors-and-resolution', replacesBlock: 1, title: '三目标的注意力 mask：ITC / ITM / ITG', svg: qformerMasks },
  { slug: '08-multimodal-llm/02-connectors-and-resolution', replacesBlock: 2, title: 'pixel shuffle：2×2 合并，token ÷4 通道 ×4', svg: pixelShuffle },
  { slug: '08-multimodal-llm/02-connectors-and-resolution', replacesBlock: 3, title: 'AnyRes：k 块 + 缩略图，M = (k+1)·576', svg: anyresTiling },
  { slug: '08-multimodal-llm/02-connectors-and-resolution', replacesBlock: 4, title: 'M-RoPE 频段切分：(16, 24, 24)', svg: mropeSections },
  { slug: '08-multimodal-llm/03-training-objectives', replacesBlock: 0, title: 'loss masking：loss 只落在 response（式 1）', svg: lossMasking },
  { slug: '08-multimodal-llm/03-training-objectives', replacesBlock: 1, title: '三阶段配方：对齐 → 指令 → 偏好', svg: stageRecipe },
  { slug: '08-multimodal-llm/04-video-understanding', replacesBlock: 0, title: '视频管线：token 省在三个位置', svg: videoPipeline },
  { slug: '08-multimodal-llm/05-unified-understanding-generation', replacesBlock: 0, title: 'Janus：解耦双 encoder + 共享 AR 主干', svg: janusDecoupled },
  { slug: '08-multimodal-llm/05-unified-understanding-generation', replacesBlock: 1, title: 'Transfusion 序列：BOI/EOI 包夹连续 patch（式 12）', svg: transfusionSequence },
  { slug: '08-multimodal-llm/05-unified-understanding-generation', replacesBlock: 2, title: 'Transfusion mask：因果 + 块内双向（式 15）', svg: transfusionMask },
  { slug: '08-multimodal-llm/05-unified-understanding-generation', replacesBlock: 3, title: 'Transfusion 生图回路：T 步主干内去噪', svg: transfusionLoop },
];
