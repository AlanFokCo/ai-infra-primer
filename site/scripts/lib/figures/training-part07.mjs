// training Part 7（07-pretraining）场景图：数据管线漏斗 / 四家 LR 曲线 / dist-ckpt 重分片 / async save 泳道
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
 * 7.1 §1（block 0，高）— 数据管线五站漏斗
 * 来源（36T tok，FineWeb §3.3）→ 过滤（启发式/分类器/perplexity，删 ~22%）→ 去重（MinHash/LSH，20T）
 * → 配比（手工表 50/25/17/8 / DoReMi / 数据课程）→ tokenize+pack（[B, T] int32）
 * ============================================================ */
function dataPipeline() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, '五段管线：从网页快照到训练张量 —— token 量级沿站递减，每个决策靠代理规模消融背书（§7）', { size: 12, color: 'var(--ink-soft)' }));

  const xs = [16, 156, 296, 436, 576];
  const defs = [
    ['来源', '网页快照 · 书籍 · 代码', 'data'],
    ['过滤（§2）', '启发式 · 分类器 · ppl', 'op'],
    ['去重（§3）', 'MinHash/LSH · Bloom', 'op'],
    ['配比（§4）', '手工表 · DoReMi · 课程', 'proj'],
    ['tokenize+pack（§6）', '定长 packed 序列', 'cache'],
  ];
  const ns = defs.map((d, i) =>
    add(node({ id: `s${i}`, x: xs[i], y: 46, w: 124, h: 62, role: d[2], title: d[0], titleSize: 12.5, sub: d[1], subSize: 9.5 }))
  );
  for (let i = 0; i < 4; i++) body.push(edge(ns[i], ns[i + 1], { nodes, pulse: true, dur: 2.2, delay: i * 0.3, uid: 'p7a', eid: i }));

  // 漏斗：各站出口 token 量级（FineWeb 实账口径）
  const funnel = [
    { x: xs[0], w: 124, tone: 'ink', label: '36T tok', note: '（FineWeb §3.3）' },
    { w: 100, tone: 'ink', label: '删 ~22%', note: '（§3.6 三条启发式）' },
    { w: 70, tone: 'seal', label: '20T tok', note: '（逐快照 MinHash）' },
  ];
  funnel.forEach((f, i) => {
    const fx = f.x ?? xs[i] + (124 - f.w) / 2;
    body.push(`<rect class="dia-bar dia-bar-${f.tone} sig-bar" style="--i:${(0.2 + i * 0.25).toFixed(2)}" x="${fx}" y="136" width="${f.w}" height="14" rx="4"/>`);
    body.push(cap(xs[i] + 62, 168, f.label, { mono: true, size: 10.5, anchor: 'middle', color: 'var(--ink)' }));
    body.push(cap(xs[i] + 62, 182, f.note, { size: 9, anchor: 'middle' }));
  });
  // 配比站：50/25/17/8 分段条（Llama-3 最终表）
  {
    const segs = [[50, 'seal'], [25, 'ink'], [17, 'soft'], [8, 'warm']];
    let cx = xs[3];
    segs.forEach(([w, tone], i) => {
      body.push(`<rect class="dia-bar dia-bar-${tone} sig-bar" style="--i:${(0.95 + i * 0.12).toFixed(2)}" x="${cx}" y="136" width="${w}" height="14" rx="3"/>`);
      cx += w + 2;
    });
    body.push(cap(xs[3] + 62, 168, '50/25/17/8', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--ink)' }));
    body.push(cap(xs[3] + 62, 182, '（Llama-3 最终表）', { size: 9, anchor: 'middle' }));
  }
  // pack 站：int32 序列条
  for (let i = 0; i < 6; i++)
    body.push(`<rect class="dia-mx-warm sig-cell" style="--i:${(1.4 + i * 0.12).toFixed(2)}" x="${xs[4] + 10 + i * 18}" y="136" width="15" height="14" rx="3"/>`);
  body.push(cap(xs[4] + 62, 168, '[B, T] int32', { mono: true, size: 10.5, anchor: 'middle', color: 'var(--ink)' }));
  body.push(cap(xs[4] + 62, 182, '（下游唯一直接消费的形态）', { size: 9, anchor: 'middle' }));

  body.push(cap(24, 212, 'FineWeb 实账：36T（基础过滤后）→ 20T（逐快照 MinHash 去重）→ 15T（C4 式 + 三条自定义启发式），每段靠 1.71B 代理消融背书', { size: 10.5 }));
  body.push(cap(24, 229, 'GLM-4.5 在过滤段多挂一道 SemDedup：模板批量生成的近似页 MinHash 删不掉（§2 末）；DCLM 同位用 Bloom filter', { size: 10.5 }));

  return scene(720, 242, body, { label: '数据管线五站：来源、过滤、去重、配比、tokenize+pack，token 量级沿站递减成漏斗' }).svg();
}

/* ============================================================
 * 7.2（block 0，中）— 四家 LR 曲线对照
 * DSv3 恒定 10T + cosine 4.3T + 常数尾 500B；Llama-3 cosine 1.2M 步至 1%；
 * OLMo 2 cosine 按 5T 标定 4T 截断 → mid-training 线性→0（3×50B+平均）；GLM-4.5 cosine 全程
 * ============================================================ */
function lrCurves() {
  const body = [];
  body.push(cap(24, 22, '四家 decay 形态：纵轴 lr/peak、横轴 tokens（不按比例）；warmup 段全部 <1% 日程（图中略）', { size: 12, color: 'var(--ink-soft)' }));

  // 坐标系：x 60..700，y(v) = 250 − v·200（v = lr/peak）
  const Y = (v) => 250 - v * 200;
  body.push(`<line class="dia-edge strong" x1="60" y1="250" x2="700" y2="250"/>`);
  body.push(`<line class="dia-edge" x1="60" y1="46" x2="60" y2="250"/>`);
  for (const v of [1.0, 0.1]) {
    body.push(`<line x1="60" y1="${Y(v)}" x2="700" y2="${Y(v)}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 5"/>`);
    body.push(cap(52, Y(v) + 4, v.toFixed(1), { mono: true, size: 10, anchor: 'end' }));
  }
  body.push(cap(700, 268, 'tokens', { mono: true, size: 10, anchor: 'end' }));
  body.push(cap(52, 42, 'lr/peak', { mono: true, size: 10, anchor: 'end' }));

  // DeepSeek-V3（seal）：恒定 10T → cosine 4.3T → 常数尾 500B
  const dsFlat = 'M70 50 H300';
  let dsCos = 'M300 50';
  for (let i = 1; i <= 20; i++) {
    const t = i / 20;
    dsCos += ` L${(300 + t * 110).toFixed(1)} ${Y(0.5 * (1 + Math.cos(Math.PI * t)) * 0.9 + 0.1).toFixed(1)}`;
  }
  const dsTail = 'M410 ' + Y(0.1) + ' H452';
  body.push(`<path d="${dsFlat}" fill="none" stroke="var(--seal)" stroke-width="2.4" stroke-linecap="round"/>`);
  body.push(`<path d="${dsCos}" fill="none" stroke="var(--seal)" stroke-width="2.4" stroke-linecap="round"/>`);
  body.push(`<path d="${dsTail}" fill="none" stroke="var(--seal)" stroke-width="2.4" stroke-linecap="round"/>`);
  body.push(pathPulse(dsFlat + ' ' + dsCos.slice(dsCos.indexOf('L')) + ' ' + dsTail.slice(dsTail.indexOf('H') - 1), { dur: 6, delay: 0, r: 3.8 }));
  body.push(cap(185, 42, '恒定 10T', { size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(432, Y(0.1) - 8, '常数尾 500B', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // Llama-3 405B（ink）：cosine 至 1%（1.2M 步）
  let l3 = 'M70 50';
  for (let i = 1; i <= 30; i++) {
    const t = i / 30;
    l3 += ` L${(70 + t * 520).toFixed(1)} ${Y(Math.max(0.01, 0.5 * (1 + Math.cos(Math.PI * t)))).toFixed(1)}`;
  }
  body.push(`<path d="${l3}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round"/>`);
  body.push(pathPulse(l3, { dur: 7, delay: 0.8, r: 3.4 }));

  // GLM-4.5（ink-soft 虚线）：cosine 全程至 mid-training 末（峰值的 10%）
  let glm = 'M70 50';
  for (let i = 1; i <= 30; i++) {
    const t = i / 30;
    glm += ` L${(70 + t * 595).toFixed(1)} ${Y(Math.max(0.1, 0.5 * (1 + Math.cos(Math.PI * t)))).toFixed(1)}`;
  }
  body.push(`<path d="${glm}" fill="none" stroke="var(--ink-soft)" stroke-width="1.8" stroke-dasharray="6 4" stroke-linecap="round"/>`);

  // OLMo 2 7B（warm 赭）：cosine 按 5T 标定，4T 处截断 → mid-training 线性 → 0
  let ol = 'M70 50';
  for (let i = 1; i <= 22; i++) {
    const t = i / 22;
    if (t > 0.8) break;
    ol += ` L${(70 + t * 550).toFixed(1)} ${Y(0.5 * (1 + Math.cos(Math.PI * t))).toFixed(1)}`;
  }
  const olTruncX = 70 + 0.8 * 550; // 4T 截断点
  const olTruncY = Y(0.5 * (1 + Math.cos(Math.PI * 0.8)));
  const olLin = `M${olTruncX} ${olTruncY.toFixed(1)} L585 250`;
  body.push(`<path d="${ol}" fill="none" stroke="#8a6d3b" stroke-width="2" stroke-linecap="round"/>`);
  body.push(`<path d="${olLin}" fill="none" stroke="#8a6d3b" stroke-width="2" stroke-linecap="round"/>`);
  body.push(pathPulse(ol + ' ' + olLin.slice(olLin.indexOf('L')), { dur: 6.5, delay: 1.6, r: 3.4 }));
  body.push(`<line x1="${olTruncX}" y1="50" x2="${olTruncX}" y2="250" stroke="#8a6d3b" stroke-width="1" stroke-dasharray="4 4" opacity="0.55"/>`);
  body.push(`<circle cx="${olTruncX}" cy="${olTruncY.toFixed(1)}" r="4" fill="#8a6d3b"/>`);
  body.push(cap(olTruncX, 42, '4T 处截断', { size: 9.5, anchor: 'middle', color: '#8a6d3b' }));

  // 图例（右上）
  const legend = [
    ['var(--seal)', '', 'DeepSeek-V3：恒定→cosine→常数尾'],
    ['var(--ink)', '', 'Llama-3 405B：cosine 至 1%（1.2M 步）'],
    ['#8a6d3b', '', 'OLMo 2 7B：截断 → mid-training 线性→0'],
    ['var(--ink-soft)', '6 4', 'GLM-4.5：cosine 全程（至 mid-training 末）'],
  ];
  legend.forEach(([color, dash, name], i) => {
    const y = 66 + i * 20;
    body.push(`<line x1="478" y1="${y}" x2="514" y2="${y}" stroke="${color}" stroke-width="2.2"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`);
    body.push(cap(520, y + 3.5, name, { size: 9.5 }));
  });

  body.push(cap(24, 288, 'cosine 一家占五席（GPT-3 / Llama-2 / Llama-3 / OLMo 2 主段 / GLM-4.5）；DeepSeek-V3 为「恒定 + cosine + 常数尾」的 WSD 梯形变体', { size: 10.5 }));
  body.push(cap(24, 304, '下限两派：收到峰值的 10%（GPT-3、Llama-2、GLM-4.5），或收到近零（Llama-3 → 1%；OLMo 2 线性 → 0 + 模型平均）', { size: 10.5 }));

  return scene(720, 316, body, { label: '四家预训练 LR 曲线：DeepSeek-V3 梯形、Llama-3 与 GLM-4.5 全程 cosine、OLMo 2 截断后线性退火' }).svg();
}

/* ============================================================
 * 7.4 §3（block 0，高）— 分布式 checkpoint 重分片
 * 保存网格 DP=4 各写本地片 → 逻辑全量张量元数据（key + global_shape + offset）→ 加载网格 DP=2 跨边界拼装
 * ============================================================ */
function distCkpt() {
  const body = [];
  const nodes = [];
  const add = (n) => { nodes.push(n); body.push(n.svg); return n; };

  body.push(cap(24, 24, '分布式 checkpoint：逻辑张量与物理分片用元数据分开 —— 保存 / 加载网格的并行度解耦（DCP · Megatron dist-ckpt）', { size: 12, color: 'var(--ink-soft)' }));

  // 左：保存网格 DP=4
  body.push(zone({ x: 16, y: 44, w: 150, h: 238, label: '保存网格 · DP=4（每 rank 1/4）' }));
  const ranks = [0, 1, 2, 3].map((i) =>
    add(node({ id: `rk${i}`, x: 30, y: 76 + i * 48, w: 104, h: 38, role: 'data', title: `rank${i} [t${i}]`, mono: true, titleSize: 11.5 }))
  );

  // 中：逻辑全量张量元数据卡
  body.push(cap(250, 72, '逻辑全量张量 · 元数据', { size: 10.5, color: 'var(--ink-soft)' }));
  body.push(`<g class="dia-node dia-node-cache"><rect x="250" y="82" width="224" height="152" rx="12"/></g>`);
  const meta = [
    'key: "layer3.w"',
    'global_shape = [4096, 4096]',
    'shard0: offset [0:1024, :]',
    'shard1: offset [1024:2048, :]',
    'shard2 / shard3: …',
  ];
  meta.forEach((m, i) => body.push(cap(266, 110 + i * 25, m, { mono: true, size: 10.5, color: 'var(--ink)' })));

  // 右：加载网格 DP=2
  body.push(zone({ x: 524, y: 44, w: 180, h: 238, label: '加载网格 · DP=2（每 rank 1/2）' }));
  const r0 = add(node({ id: 'nr0', x: 538, y: 92, w: 152, h: 52, role: 'proj', title: "rank0′", sub: '← shard0 + shard1', mono: true, titleSize: 11.5, subSize: 9.5 }));
  const r1 = add(node({ id: 'nr1', x: 538, y: 196, w: 152, h: 52, role: 'proj', title: "rank1′", sub: '← shard2 + shard3', mono: true, titleSize: 11.5, subSize: 9.5 }));

  // 保存流：各 rank 只写本地片
  ranks.forEach((r, i) => {
    body.push(edge(r, [250, 110 + i * 25], { nodes, pulse: true, dur: 2.0, delay: i * 0.25, uid: 'p7c', eid: `w${i}` }));
  });
  // 加载流：按目标网格重新规划，跨旧分片边界拼装
  body.push(edge([474, 118], r0, { nodes, pulse: true, dur: 2.2, delay: 1.1, uid: 'p7c', eid: 'r0', strong: true }));
  body.push(edge([474, 200], r1, { nodes, pulse: true, dur: 2.2, delay: 1.35, uid: 'p7c', eid: 'r1', strong: true }));
  body.push(cap(500, 82, '按目标网格重规划', { size: 9.5, anchor: 'middle' }));

  body.push(cap(24, 308, '保存：各 rank 只写自己的本地片（SPMD）；加载：读计划按新网格声明全局区间，rank0′ 跨旧分片边界拼装 shard0 + shard1', { size: 10.5 }));
  body.push(cap(24, 325, '元数据与数据分离的附赠能力：只读元数据即可检视内容清单（不搬张量字节）；按 key 子集加载 = 部分恢复（评测加载 / 换优化器续训，§5）', { size: 10.5 }));

  return scene(720, 338, body, { label: '分布式 checkpoint：DP=4 保存网格各写本地片，元数据登记逻辑全量张量，DP=2 加载网格跨边界拼装' }).svg();
}

/* ============================================================
 * 7.4 §4（block 1，中）— async save 三级泳道
 * GPU 训练不停 → D2H staging（阻塞段）→ 后台 upload；staging_completion / upload_completion 两个 future
 * ============================================================ */
function asyncSave() {
  const body = [];

  body.push(cap(24, 24, 'async_save 两段式：GPU 可见的停顿只剩 D2H 一段，写盘整体藏进后台（DCP async_save，文档口径）', { size: 12, color: 'var(--ink-soft)' }));

  // 三条泳道
  body.push(zone({ x: 16, y: 46, w: 688, h: 64, label: 'GPU 显存' }));
  body.push(zone({ x: 16, y: 126, w: 688, h: 64, label: '主机内存（staging）' }));
  body.push(zone({ x: 16, y: 206, w: 688, h: 64, label: '持久存储（fabric）' }));

  // 泳道 1：训练 bar 持续推进
  body.push(`<rect class="dia-bar dia-bar-ink sig-bar" style="--i:0" x="170" y="72" width="470" height="16" rx="5"/>`);
  body.push(cap(405, 68, '训练 step 继续（不等写盘）', { size: 10, anchor: 'middle', color: 'var(--ink-soft)' }));
  // async_save() 调用点
  body.push(`<line x1="300" y1="66" x2="300" y2="94" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(cap(300, 60, 'async_save()', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: 600 }));

  // 泳道 2：D2H 阻塞段 + staging 驻留
  body.push(`<rect class="dia-bar dia-bar-seal sig-bar" style="--i:0.5" x="300" y="152" width="62" height="16" rx="5"/>`);
  body.push(cap(296, 164, 'D2H（阻塞段，~秒级以下）', { size: 9.5, anchor: 'end' }));
  body.push(`<rect x="362" y="152" width="180" height="16" rx="5" fill="none" stroke="var(--rule)" stroke-width="1.1" stroke-dasharray="4 4"/>`);
  body.push(cap(452, 164, 'staging 驻留', { size: 9.5, anchor: 'middle' }));
  body.push(`<circle class="sig-cell" style="--i:1.2" cx="366" cy="160" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(376, 146, 'staging_completion（future）', { mono: true, size: 9.5, color: 'var(--seal-deep)' }));

  // 泳道 3：后台 upload
  body.push(`<rect class="dia-bar dia-bar-soft sig-bar" style="--i:1.0" x="366" y="232" width="240" height="16" rx="5"/>`);
  body.push(cap(370, 226, 'upload：后台线程 / 进程（写文件 · 对象存储）', { size: 9.5 }));
  body.push(`<circle class="sig-cell" style="--i:1.8" cx="610" cy="240" r="4.5" fill="var(--seal)"/>`);
  body.push(cap(610, 264, 'upload_completion（future）', { mono: true, size: 9.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  // 连接：async_save → D2H；staging 完成 → upload
  body.push(`<path class="dia-edge" d="M300 94 V118 H330 V150"/>`);
  body.push(pathPulse('M300 94 V118 H330 V150', { dur: 1.6, delay: 0.2, r: 3.6 }));
  body.push(`<path class="dia-edge" d="M368 170 V196 H400 V230"/>`);
  body.push(pathPulse('M368 170 V196 H400 V230', { dur: 1.6, delay: 0.9, r: 3.6 }));

  body.push(cap(24, 292, 'staging_completion 之后 GPU 侧张量可安全复写 —— 正确性约束：staging 完成前训练侧不能原地改写被保存的张量', { size: 10.5 }));
  body.push(cap(24, 306, '量级：每卡 staging ≈ 0.35 GB（405B / 16384 卡）对 step ≈ 6 s；阻塞写整盘 ≈ 半个 step、2.8 s 下界的写盘全部藏进后台（§2 账）', { size: 10.5 }));

  return scene(720, 318, body, { label: '异步 checkpoint：GPU 训练不停，D2H staging 是唯一阻塞段，upload 由后台线程完成，两个 future 标记里程碑' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  { dim: 'training', slug: '07-pretraining/01-data-pipeline', replacesBlock: 0, title: '数据管线五站漏斗：36T → 20T → [B, T] int32', svg: dataPipeline },
  { dim: 'training', slug: '07-pretraining/02-recipes', replacesBlock: 0, title: '四家 LR 曲线：恒定平台 / cosine / 截断退火', svg: lrCurves },
  { dim: 'training', slug: '07-pretraining/04-checkpointing-and-resumption', replacesBlock: 0, title: '分布式 checkpoint：保存 / 加载网格解耦', svg: distCkpt },
  { dim: 'training', slug: '07-pretraining/04-checkpointing-and-resumption', replacesBlock: 1, title: 'async save：GPU 停顿只剩 D2H 一段', svg: asyncSave },
];
