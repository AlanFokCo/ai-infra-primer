// inference Part 9（09-production）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part08.mjs / inference-part07.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-24 census）：
//   9.1 唯一围栏带语言（```text 章脊全链）→ 管线逐字保留，四张全 afterHeading；
//   9.2 block 0 = TRT-LLM 执行结构；9.3 block 0 = colocate/disagg、block 1 = partial rollout；
//   9.4 block 0 = 多模态生命周期、block 1 = SO 同步/异步时序。全部 STRUCTURE/DATAFLOW，无伪代码块。
// 数值口径：9.1 教学 8B GQA bf16 κL=128 KiB/token、u=2 USD/GPU-h 教学假设；38 USD/GPU-h = 收入（off-peak）非成本；
//           9.3 sleep level 1 = 权重 offload 到 CPU 自动恢复 + 仅 KV 丢弃（S1 纠正后语义，勿画 discard-all）；
//           level 2 = 丢弃被改写集（无备份）+ preserved offload；KV 两 level 均丢弃；
//           9.4 V4.1-Flash vision_config：patch 14 / downsample 3 / n_vit 4096 → n_llm 484 → n_expand 508（每维 ceil）。

import { scene, node, edge, zone, cap as kitCap, matrix, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 覆盖 cap() 的 fill 属性 —— 显式 color 补 inline style 提权（W2 finding，不改 kit API）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

const sealDash = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="none" stroke="var(--seal)" stroke-width="1.3" stroke-dasharray="4 4"/>`;

/* ============================================================
 * 9.1 §1（afterHeading）— SLO sizing 全链（签名）
 * 原文：λ=100（chatbot 80 / agent 20）→ 三 ρ 上界取 min → ρ_P=0.72；
 *       n_P=4（D_P=1.390e15 / F_P ρ_P）；n_D=45（带宽闸 37 / 容量闸 45 取 max）；
 *       存储 4 台（写带宽闸）；98 USD/h → 0.0578 USD/M（全 token）
 * ============================================================ */
function capacityChain() {
  const UID = 'i9a';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono: true });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'SLO sizing 全链：λ=100 req/s → ρ_P=0.72 → n_P=4 / n_D=45 → 存储 4 台 → 0.0578 USD/M tokens', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 280, h: 148, label: 'workload 两段混合（§1）' }));
  const cb = mk('cb', 52, 72, 116, 52, 'data', 'chatbot 80 req/s', 'in 1024 / out 256', 8.5, 6.3);
  const ag = mk('ag', 184, 72, 116, 52, 'data', 'agent 20 req/s', 'in 16384 / out 2048', 8.5, 6.3);
  body.push(cap(52, 140, 'h：chatbot 0.5 / agent 0.9 · TTL 7d / 72h', { size: 7.5 }));
  body.push(cap(52, 158, 'SLO：TTFT 0.5 s · TPOT 30 ms · θ=0.9', { size: 7.5 }));
  body.push(cap(52, 176, '单一平均长度会同时算错三笔账', { size: 7.5 }));

  body.push(zone({ x: 368, y: 40, w: 316, h: 148, label: '三个 ρ 上界取 min（§2）· q=0.2 s' }));
  [['均值余量', 0.877, 'ink', 72, 0.2], ['Kingman 修正', 0.891, 'soft', 94, 0.4], ['attainment θ=0.9', 0.722, 'seal', 116, 0.6]].forEach(([lab, v, tone, y, di]) => {
    body.push(cap(384, y + 10, lab, { size: 7.5 }));
    const w = v * 160;
    body.push(`<g class="sig-bar" style="--i:${di}"><rect class="dia-bar dia-bar-${tone}" x="470" y="${y}" width="${w.toFixed(0)}" height="12" rx="3"/></g>`);
    body.push(cap(476 + w, y + 10, v.toFixed(3), { size: 7.5, mono: true, color: tone === 'seal' ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });
  body.push(cap(384, 152, 'attainment 上界 0.722 ≪ 均值 0.877', { size: 7.5 }));
  body.push(cap(384, 176, '取 min → ρ_P = 0.72 · percentile 是硬闸', { size: 8.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(edge([316, 114], [368, 114], { nodes, pulse: true, dur: 1.2, delay: 0.2, uid: UID, eid: 'w0' }));
  body.push(arrowHead(368, 114, 1, 0));

  body.push(zone({ x: 36, y: 204, w: 280, h: 132, label: 'prefill 算力闸（§3）' }));
  const dp = mk('dp', 52, 236, 248, 44, 'cache', 'D_P = 1.390×10¹⁵ FLOPs/s', '线性+attention · non-cached 73728 tok/s', 9, 6.3);
  const np = mk('np', 52, 288, 116, 36, 'op', 'n_P = 4', 'ceil(3.90)', 12, 6.5);
  body.push(edge(dp, np, { nodes, portFrom: dp.bottom, portTo: np.top, pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'p0' }));
  body.push(arrowHead(np.top[0], np.top[1], 0, 1));
  body.push(cap(184, 300, 'ρ=0.702 ≤ 0.722', { size: 7.5, mono: true }));
  body.push(cap(184, 314, 'attainment 91.6% ≥ 90% ✓', { size: 7.5 }));
  body.push(cap(52, 330, 'agent 贡献 54.6% FLOPs · attention 项 94.2% 来自 agent', { size: 7 }));

  body.push(zone({ x: 368, y: 204, w: 316, h: 132, label: 'decode 两闸取 max（§4）· 命中不省 decode' }));
  const cc = mk('cc', 384, 236, 120, 44, 'data', 'C = 1843.2', 'Little：Σλ·T_out·TPOT', 10, 6.3);
  const bw = mk('bw', 520, 236, 70, 44, 'cache', '带宽闸', '37 张', 9, 7);
  const cg = mk('cg', 606, 236, 62, 44, 'cache', '容量闸', '45 张', 9, 7);
  body.push(edge(cc, bw, { nodes, pulse: true, dur: 1, delay: 1.2, uid: UID, eid: 'd0' }));
  body.push(arrowHead(bw.left[0], bw.left[1], 1, 0));
  const cgPath = 'M444 280 C444 288, 637 288, 637 280';
  body.push(`<path class="dia-edge" d="${cgPath}"/>`);
  body.push(pathPulse(cgPath, { dur: 1.2, delay: 1.4, r: 3.4 }));
  body.push(arrowHead(637, 280, 0, -1));
  const nd = mk('nd', 520, 292, 148, 36, 'op', 'n_D = max(37,45) = 45', null, 9.5);
  body.push(cap(384, 300, 'agent 段 96.8% KV', { size: 7.5 }));
  body.push(cap(384, 314, '2.90 TB · 59.95 GiB/卡 ≤ 60', { size: 7 }));
  body.push(cap(384, 330, '步时 24.0 ms ≤ 30 ms TPOT ✓', { size: 7.5 }));
  body.push(edge([420, 188], [176, 204], { nodes, pulse: true, dur: 1.4, delay: 0.8, uid: UID, eid: 'r0' }));
  body.push(arrowHead(176, 204, 0, 1));
  body.push(edge([560, 188], [526, 204], { nodes, pulse: true, dur: 1.2, delay: 1.0, uid: UID, eid: 'r1' }));
  body.push(arrowHead(526, 204, 0, 1));

  body.push(zone({ x: 36, y: 352, w: 648, h: 76, label: '存储四约束取 max（§6）→ 成本（§7）· u=2 USD/GPU-h 教学假设' }));
  const st = mk('st', 52, 384, 150, 36, 'cache', '存储 4 台服务器', '写带宽闸主导', 9, 6);
  const gh = mk('gh', 232, 384, 130, 36, 'plain', '98 USD/h', '2 × (4+45) GPU 租金', 11, 6);
  const co = mk('co', 392, 384, 150, 36, 'op', '0.0578 USD/M', '全 token 口径', 11, 6.5);
  body.push(edge(st, gh, { nodes, pulse: true, dur: 1, delay: 2.0, uid: UID, eid: 's0' }));
  body.push(arrowHead(gh.left[0], gh.left[1], 1, 0));
  body.push(edge(gh, co, { nodes, pulse: true, dur: 1, delay: 2.3, uid: UID, eid: 's1' }));
  body.push(arrowHead(co.left[0], co.left[1], 1, 0));
  body.push(cap(560, 398, '÷ 1695.7 M tokens/h', { size: 7.5, mono: true }));
  body.push(cap(560, 412, '仅输出 0.443（7.67×）', { size: 7 }));
  const f1 = 'M110 336 C110 348, 297 340, 297 352';
  const f2 = 'M594 336 C594 348, 297 340, 297 352';
  body.push(`<path class="dia-edge" d="${f1}"/>`);
  body.push(`<path class="dia-edge" d="${f2}"/>`);
  body.push(pathPulse(f1, { dur: 1.4, delay: 1.6, r: 3.4 }));
  body.push(pathPulse(f2, { dur: 1.4, delay: 1.8, r: 3.4 }));
  body.push(arrowHead(297, 352, 0, 1));

  body.push(cap(36, 446, '口径锁定：8B dense bf16 @ H100 · u=2 USD/GPU-h 教学假设（非市场报价）· 成本只计 GPU 租金', { size: 9 }));
  body.push(cap(36, 462, 'DeepInfer 生产数字（155T tokens/day · 38 USD/GPU-h 收入 · 64P+64D）官方披露，不与教学算例混算', { size: 9 }));
  return scene(720, 470, body, { label: '容量规划全链：两段混合负载经三个利用率上界取最小得零点七二，分出 prefill 算力闸四张与 decode 带宽容量两闸取最大四十五张，汇入存储四约束四台与每小时九十八美元成本，折全 token 口径每百万 token 零点零五七八美元' }).svg();
}

/* ============================================================
 * 9.1 §6（afterHeading）— 存储 EC 算术与四约束（签名）
 * 原文：raw 614.6 / usable 512.2 TB；逻辑 14.4 / 物理 17.28 TB/GPU；命题 2 恒等 35.6 GPU/server；
 *       100 PiB = 112590 TB → ~183（raw）/ ~220（usable）口径歧义；
 *       教学四约束：24h 2.99→3 / 写 3.54→4 / 读 1.10→2 / 每 GPU 1.38→2 → 4 台写闸主导
 * ============================================================ */
function storageEC() {
  const UID = 'i9b';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono: true });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, '存储 EC 算术：raw / usable 两口径恒等 35.6 GPU/server —— 100 PiB 口径歧义须声明', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 324, h: 150, label: '每服务器 / 每 GPU 四口径（doc L101/L105 · EC 10+2）' }));
  const raw = mk('raw', 52, 72, 136, 44, 'data', 'raw 614.6 TB', '20 × 30.73 TB SSD', 10, 6.5);
  const usa = mk('usa', 208, 72, 136, 44, 'cache', 'usable 512.2 TB', '× 10/12（EC 冗余）', 9.5, 6.3);
  const log = mk('log', 52, 132, 136, 44, 'data', '逻辑 14.4 TB/GPU', 'doc L105 官方披露', 8.5, 6.3);
  const phy = mk('phy', 208, 132, 136, 44, 'cache', '物理 17.28 TB/GPU', '× 12/10', 8, 6.5);
  body.push(edge(raw, usa, { nodes, pulse: true, dur: 1.2, delay: 0.2, uid: UID, eid: 'e0' }));
  body.push(arrowHead(usa.left[0], usa.left[1], 1, 0));
  body.push(edge(log, phy, { nodes, pulse: true, dur: 1.2, delay: 0.5, uid: UID, eid: 'e1' }));
  body.push(arrowHead(phy.left[0], phy.left[1], 1, 0));
  body.push(cap(52, 184, '命题 2：614.6/17.28 = 512.2/14.4 = 35.6 GPU/server（EC 因子相消）', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 376, y: 40, w: 308, h: 150, label: '100 PiB 集群口径歧义（doc L187 · 约数）' }));
  body.push(cap(392, 72, '100 PiB = 112590 TB（十进制）', { size: 8.5, mono: true }));
  mk('s183', 392, 84, 128, 44, 'plain', '~183 台', 'raw 口径 ÷614.6', 10, 6.5);
  mk('s220', 540, 84, 128, 44, 'plain', '~220 台', 'usable 口径 ÷512.2', 10, 6.3);
  body.push(cap(392, 146, 'doc 未明示口径 —— 两个数都给出并声明歧义', { size: 8, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(392, 164, '反推 GPU：~6516（raw）/ ~7819（usable）', { size: 7.5, mono: true }));
  body.push(cap(392, 180, '算术隐含值，非官方披露，引用须带口径', { size: 7.5 }));

  body.push(zone({ x: 36, y: 206, w: 648, h: 140, label: '教学负载四约束取 max（式 11）· 写带宽闸主导' }));
  [['24h 保留容量', '1531 TB ×1.2 ÷ 614.6', 2.99, '3', 'ink', 240, 0.2],
   ['写带宽', '17.72 GB/s ÷ 5', 3.54, '4', 'seal', 264, 0.4],
   ['读带宽', '44.02 GB/s ÷ 40', 1.10, '2', 'ink', 288, 0.6],
   ['每 GPU 经验', '49×14.4×1.2 ÷ 614.6', 1.38, '2', 'ink', 312, 0.8]].forEach(([lab, calc, v, ceilV, tone, y, di]) => {
    body.push(cap(52, y + 10, lab, { size: 8.5 }));
    body.push(cap(200, y + 10, calc, { size: 7.5, mono: true }));
    const w = v * 40;
    body.push(`<g class="sig-bar" style="--i:${di}"><rect class="dia-bar dia-bar-${tone}" x="420" y="${y}" width="${w.toFixed(0)}" height="12" rx="3"/></g>`);
    body.push(cap(572, y + 10, `${v.toFixed(2)} → ${ceilV}`, { size: 8, mono: true, color: tone === 'seal' ? 'var(--seal-deep)' : 'var(--ink-faint)', weight: tone === 'seal' ? '650' : undefined }));
  });
  body.push(cap(52, 340, '写率 =（non-cached 73728 + 输出 61440）× 128 KiB', { size: 7 }));
  body.push(cap(420, 340, 'n_stor = 4 台 · 写闸主导（利用率 88.6%）', { size: 8.5, weight: '650', color: 'var(--seal-deep)' }));

  body.push(cap(36, 366, '教学反推 31.2 TB/GPU = 2.2× 官方 14.4：纯 KV 几何口径差（8B GQA bf16 128 KiB/token）', { size: 9 }));
  body.push(cap(36, 384, 'V4.1 量化 KV + 混合注意力小得多 —— 两口径不可互相代入；官方锚核生产一致性，24h+带宽闸给任意负载下界', { size: 9 }));
  body.push(cap(36, 402, '24h = 容量 sizing 准则 · TTL = 条目年龄上限（agent 72h / chatbot 7d）· LRU watermark = 容量压力驱逐信号 —— 同一 GC 实现', { size: 9 }));
  return scene(720, 416, body, { label: '存储 EC 算术：每服务器 raw 六百一十四点六 TB 经十比十二折 usable 五百一十二点二，每 GPU 逻辑十四点四折物理十七点二八，两口径恒等三十五点六 GPU 每服务器；一百 PiB 集群按 raw 约一百八十三台按 usable 约二百二十台须声明歧义；教学负载四约束取最大四台写带宽闸主导' }).svg();
}

/* ============================================================
 * 9.1 §7（afterHeading）— USD/M 两口径 × 收入不是成本（签名）
 * 原文：98 USD/h ÷ 1695.7 M/h = 0.0578（全 token）vs ÷ 221.2 = 0.443（仅输出），7.67×；
 *       doc L17：38 USD/GPU-h = overall revenue（off-peak）；155T/day = 6.458e6 M/h；
 *       收入/M = 5.884e-6·G，G 未披露；G=4096 教学假设 → 0.0241；三重口径缺口
 * ============================================================ */
function costCalibers() {
  const UID = 'i9c';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = true) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'USD/M tokens 两口径差 7.67× —— 38 USD/GPU-h 是收入（off-peak），不是成本', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 132, label: '通式（式 12）· 量纲 [USD/h] ÷ [M tokens/h] = USD/M tokens' }));
  mk('fm', 52, 72, 220, 48, 'op', 'USD/M = GPU 租金 ÷ (M tokens/h)', '式 (12)：(u_P n_P + u_D n_D) / (Ṫ/10⁶)', 8.5, 6.3, false);
  const c98 = mk('c98', 300, 72, 120, 48, 'cache', '98 USD/h', '2 × 49 GPU', 12, 7);
  const at = mk('at', 452, 64, 108, 40, 'data', '全 token', '1695.7 M/h', 9.5, 6.5);
  const oo = mk('oo', 452, 116, 108, 40, 'data', '仅输出', '221.2 M/h', 9.5, 6.5);
  const r1 = mk('r1', 592, 64, 76, 40, 'op', '0.0578', 'USD/M', 11, 6.5);
  const r2 = mk('r2', 592, 116, 76, 40, 'op', '0.443', 'USD/M', 11, 6.5);
  body.push(edge(c98, at, { nodes, pulse: true, dur: 1, delay: 0.3, uid: UID, eid: 'a0' }));
  body.push(arrowHead(at.left[0], at.left[1], 1, 0));
  body.push(edge(c98, oo, { nodes, portFrom: [c98.right[0], c98.right[1] + 12], pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: 'a1' }));
  body.push(arrowHead(oo.left[0], oo.left[1], 1, 0));
  body.push(edge(at, r1, { nodes, pulse: true, dur: 0.9, delay: 0.9, uid: UID, eid: 'a2' }));
  body.push(arrowHead(r1.left[0], r1.left[1], 1, 0));
  body.push(edge(oo, r2, { nodes, pulse: true, dur: 0.9, delay: 1.2, uid: UID, eid: 'a3' }));
  body.push(arrowHead(r2.left[0], r2.left[1], 1, 0));
  body.push(cap(52, 140, 'Ṫ 全 token = 471040 tok/s（输入+输出，含命中）', { size: 7.5 }));
  body.push(cap(52, 156, 'Ṫ 仅输出 = 61440 tok/s · 每请求 2.72×10⁻⁴ USD', { size: 7.5 }));
  body.push(cap(452, 168, '口径差 7.67× —— 比价先对齐口径', { size: 8, weight: '650', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 188, w: 648, h: 150, tone: 'seal', label: 'doc L17 逐字：overall revenue of US$38 per GPU-hour (at the off-peak price) —— 收入口径' }));
  const rev = mk('rev', 52, 220, 180, 48, 'attn', '38 USD/GPU-h', '收入（off-peak）≠ 成本', 11, 6.5);
  const t155 = mk('t155', 260, 220, 180, 48, 'data', '155T tokens/day', '= 6.458×10⁶ M tokens/h', 10.5, 6.5);
  const gf = mk('gf', 468, 220, 200, 48, 'proj', '收入/M = 5.884×10⁻⁶ · G', 'G = 集群 GPU 数（doc 未披露）', 8.5, 6.3);
  body.push(edge(t155, gf, { nodes, pulse: true, dur: 1, delay: 0.4, uid: UID, eid: 'b0' }));
  body.push(arrowHead(gf.left[0], gf.left[1], 1, 0));
  const bp = 'M142 268 C142 286, 568 286, 568 268';
  body.push(`<path class="dia-edge dash" d="${bp}"/>`);
  body.push(pathPulse(bp, { dur: 1.8, delay: 0.8, r: 3.4 }));
  body.push(arrowHead(568, 268, 0, -1));
  body.push(cap(52, 302, '代入须显式声明：设 G=4096（纯教学假设）→ 收入 0.0241 USD/M · 每 GPU 日产 3.78×10¹⁰ tokens', { size: 8 }));
  body.push(cap(468, 302, '38×24 = 912 USD/GPU-day', { size: 7.5, mono: true }));
  body.push(cap(52, 320, '三重口径缺口：收入非成本 · G 未披露 · 155T token 计数未定义（输入/输出/命中计否）—— 无法与他系统成本/M 直接比价', { size: 8, color: 'var(--seal-deep)' }));

  body.push(cap(36, 356, '把 38 代入式 (12) 的 u 算出的是「收入/M tokens」而非「成本/M tokens」—— off-peak 价非全时段均价，跨系统比价系统性失真', { size: 9 }));
  body.push(cap(36, 374, '教学成本用独立假设的 2 USD/GPU-hour，与 38 无任何代入关系；PD 分离两池可异价（Splitwise 异构集群优化）', { size: 9 }));
  return scene(720, 392, body, { label: '成本两口径：每小时九十八美元除以全 token 每小时一千六百九十五点七百万得零点零五七八，除以仅输出二百二十一点二百万得零点四四三，差七点六七倍；朱砂分区钉死三十八美元每 GPU 小时是低谷价收入非成本，集群 GPU 数未披露计数未定义，代入须三重声明' }).svg();
}

/* ============================================================
 * 9.1 §8（afterHeading）— 命中节省 26.9%（签名）
 * 原文：h=0 反事实 n_P0=22 vs 4，Δ18；命题 3 节省分数 18/67=26.9%；134→98 USD/h；
 *       0.0790→0.0578 USD/M；存储写 61.74→17.72 GB/s、13→4 台；
 *       h=0 agent prefill 0.639 s > TTFT 0.5 s（SLO 入场券）；TTL 二阶（chatbot 3d 反事实 0.5–1%）
 * ============================================================ */
function hitSavings() {
  const UID = 'i9d';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = true) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, '命中节省 26.9% = Δn_P/(n_P0+n_D) —— 省 prefill 算力不省 decode 并发，更是长输入 SLO 入场券', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 324, h: 168, label: 'h=0 反事实（固定 ρ_P=0.72 隔离命中变量）' }));
  const np0 = mk('np0', 52, 72, 120, 48, 'cache', 'n_P0 = 22', 'h=0 全量 prefill', 12, 6.5);
  const np = mk('np', 222, 72, 120, 48, 'op', 'n_P = 4', '命中 h：0.5 / 0.9', 12, 6.5);
  body.push(edge(np0, np, { nodes, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'h0' }));
  body.push(arrowHead(np.left[0], np.left[1], 1, 0));
  body.push(cap(197, 86, 'Δn_P = 18', { size: 8, anchor: 'middle', weight: '650', color: 'var(--seal-deep)' }));
  mk('nd', 52, 136, 290, 40, 'plain', 'n_D = 45 不变 · 命中不省 decode（KV 仍常驻全读）', null, 8.5, 6.5, false);
  body.push(cap(52, 196, '命题 3：节省分数 = 18/67 = 26.9%（全 token 分母 · 两池同价）', { size: 8, weight: '650', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 376, y: 40, w: 308, h: 168, label: '成本与存储同步省' }));
  body.push(cap(392, 70, 'h=0：67 GPU · 134 USD/h · 0.0790 USD/M', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-warm" x="392" y="76" width="88" height="14" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-ink" x="483" y="76" width="180" height="14" rx="3"/></g>`);
  body.push(cap(436, 87, 'P 22', { size: 7, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(573, 87, 'D 45', { size: 7, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(392, 110, '命中：49 GPU · 98 USD/h · 0.0578 USD/M', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-seal" x="392" y="116" width="16" height="14" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-ink" x="411" y="116" width="180" height="14" rx="3"/></g>`);
  body.push(cap(400, 127, '4', { size: 7, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(501, 127, 'D 45', { size: 7, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(598, 127, '省 18 GPU', { size: 7.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(392, 152, '存储写率 61.74 → 17.72 GB/s（÷3.5）', { size: 8 }));
  body.push(cap(392, 168, '写闸 13 → 4 台 · 命中把「重算+重写」换成「读」', { size: 8 }));
  body.push(cap(392, 192, '节省同源：prefill 算力 ↓ + 存储写 ↓', { size: 8, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 224, w: 648, h: 100, label: '命中还是 SLO 可行性问题（不只是成本）' }));
  const s0 = mk('s0', 52, 256, 190, 48, 'attn', 'h=0：agent prefill 0.639 s', '> TTFT SLO 0.5 s · 单卡算不完', 8.5, 6.3);
  const s1 = mk('s1', 282, 256, 190, 48, 'norm', '命中：压到 76.7 ms', 'TTFT SLO 单 GPU 可行', 9.5, 6.5);
  body.push(edge(s0, s1, { nodes, pulse: true, dur: 1.2, delay: 0.6, uid: UID, eid: 's0' }));
  body.push(arrowHead(s1.left[0], s1.left[1], 1, 0));
  body.push(cap(496, 272, 'h=0 须 chunked prefill', { size: 7.5 }));
  body.push(cap(496, 288, '或改并行度（4.3 / 8.1）', { size: 7.5 }));
  body.push(cap(52, 316, 'TTL 二阶：chatbot 3d TTL 损 0.5–1% 命中（反事实，非生产损失）· 生产 agent 72h / chatbot 7d 近乎无损', { size: 8 }));

  body.push(cap(36, 344, '教学二阶量级：chatbot h 降 0.5–1 个百分点 → Δn_P ≈ 0.018–0.035 张 · 成本增 0.035–0.070 USD/h —— 相对省 18 张 / 36 USD/h 是二阶小量', { size: 9 }));
  body.push(cap(36, 362, '一阶 = 有缓存 vs 无缓存（26.9%）；二阶 = TTL 在 72h/7d 基础上微调（<0.1%）—— 与官方「近乎无损」一致', { size: 9 }));
  body.push(cap(36, 380, 'chatbot 命中长尾重（少量 KV 读极多次，doc L187）：丢长尾条目的命中损失不成比例地高，故 TTL 取 7d', { size: 9 }));
  return scene(720, 398, body, { label: '命中节省账：无命中反事实 prefill 二十二张对命中四张，差十八张，节省分数十八除以六十七等于百分之二十六点九；decode 四十五张不变；存储写闸十三台降四台；无命中时 agent 单请求 prefill 零点六三九秒超 TTFT 预算，命中更是 SLO 入场券；TTL 微调是二阶小量' }).svg();
}

/* ============================================================
 * 9.2 §2（block 0）— TRT-LLM 当前 PyTorch-native 执行结构 vs legacy 已移除
 * 原文：Python LLM API / trtllm-serve → native PyTorch（_torch/models/）→
 *       PyExecutor(Worker) 每 rank 循环：Scheduler → KVCacheManager → ModelEngine → Sampler；
 *       IFB = continuous batching；Overlap Scheduler 默认开；CUDA Graph padding 至多 22%；
 *       legacy：HF checkpoint → convert_checkpoint → trtllm-build → TRT engine（removed）
 * ============================================================ */
function trtllmExecutor() {
  const UID = 'i9e';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'TRT-LLM 当前 main = PyTorch-native：PyExecutor 每 rank 循环 —— legacy 编译路线已移除', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 232, label: '当前 main（PyTorch-native · README L290「Architected on PyTorch」）' }));
  const api = mk('api', 52, 72, 150, 44, 'data', 'Python LLM API', 'trtllm-serve 入口', 9.5, 6.5);
  const mdl = mk('mdl', 232, 72, 180, 44, 'proj', 'native PyTorch 模型', 'tensorrt_llm/_torch/models/', 9.5, 6.3, true);
  body.push(edge(api, mdl, { nodes, pulse: true, dur: 1, delay: 0.2, uid: UID, eid: 't0' }));
  body.push(arrowHead(mdl.left[0], mdl.left[1], 1, 0));
  body.push(cap(440, 80, 'IFB = continuous / iteration-level', { size: 8 }));
  body.push(cap(440, 96, 'batching（非新机制 · 4.1）', { size: 8 }));
  body.push(cap(440, 112, 'Overlap Scheduler 默认开', { size: 8 }));

  body.push(zone({ x: 52, y: 132, w: 560, h: 104, label: 'PyExecutor (Worker)：每 rank 一个后台循环' }));
  const sch = mk('sch', 72, 164, 110, 40, 'op', 'Scheduler', null, 9.5);
  const kvm = mk('kvm', 212, 164, 130, 40, 'cache', 'KVCacheManager', null, 8.5);
  const men = mk('men', 372, 164, 110, 40, 'attn', 'ModelEngine', null, 9);
  const smp = mk('smp', 512, 164, 80, 40, 'norm', 'Sampler', null, 9.5);
  body.push(edge(mdl, men, { nodes, portFrom: mdl.bottom, portTo: men.top, pulse: true, dur: 1.2, delay: 0.5, uid: UID, eid: 't1' }));
  body.push(arrowHead(men.top[0], men.top[1], 0, 1));
  [[sch, kvm, 0.8], [kvm, men, 1.1], [men, smp, 1.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 0.9, delay: dl, uid: UID, eid: `t${i + 2}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  const loop = 'M552 204 C552 222, 127 222, 127 204';
  body.push(`<path class="dia-edge dash" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 1.8, delay: 1.7, r: 3.4 }));
  body.push(arrowHead(127, 204, 0, -1));
  body.push(cap(340, 232, '下一 iteration', { size: 7, anchor: 'middle' }));

  body.push(cap(52, 252, 'Overlap：', { size: 7.5 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="110" y="242" width="180" height="10" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-ink" x="110" y="256" width="180" height="10" rx="3"/>`);
  body.push(cap(200, 250, 'GPU 第 n+1 步', { size: 6, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(200, 264, 'CPU 第 n 步', { size: 6, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(300, 250, 'GPU n+1 ∥ CPU n（与 SGLang zero-overhead / vLLM 同族 · 6.2）', { size: 7.5 }));
  body.push(cap(300, 264, 'CUDA Graph padding 至多 22% e2e 吞吐（官方披露 · workload 依赖）', { size: 7.5 }));

  body.push(zone({ x: 36, y: 288, w: 648, h: 88, label: 'legacy 路线（已移除 · 文档仅留 cross-reference）' }));
  const hf = mk('hf', 52, 320, 120, 40, 'plain', 'HF checkpoint', null, 8.5);
  const cv = mk('cv', 202, 320, 140, 40, 'plain', 'convert_checkpoint', null, 7.5, 6.5, true);
  const bd = mk('bd', 372, 320, 110, 40, 'plain', 'trtllm-build', null, 8, 6.5, true);
  const eg = mk('eg', 512, 320, 150, 40, 'plain', 'TRT engine → runtime', '全图 AOT 编译产物', 8, 6);
  body.push(edge(hf, cv, { nodes, dash: true, uid: UID, eid: 'l0' }));
  body.push(edge(cv, bd, { nodes, dash: true, uid: UID, eid: 'l1' }));
  body.push(edge(bd, eg, { nodes, dash: true, uid: UID, eid: 'l2' }));
  body.push(cap(360, 372, '「The legacy TensorRT backend has been removed and is no longer supported.」', { size: 7.5, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));

  body.push(cap(36, 396, '命题 1 执行模型趋同：每 rank 后台循环 + iteration-level batching + CPU/GPU 重叠调度 —— 与 vLLM EngineCore / SGLang 事件循环三项对位', { size: 9 }));
  body.push(cap(36, 414, '分界移到生态与硬件矩阵：Dynamo / Triton 编排栈 · NVIDIA 专属（Blackwell B200/GB200/B300/GB300/DGX Spark 首发）· 新项目入口 trtllm-serve', { size: 9 }));
  return scene(720, 428, body, { label: 'TRT-LLM 当前执行结构：Python API 经 native PyTorch 模型进入 PyExecutor 每 rank 后台循环，Scheduler KVCacheManager ModelEngine Sampler 四件顺序执行并回绕下一迭代，GPU 与 CPU 步重叠；下方 legacy 编译路线虚线静止标注已移除' }).svg();
}

/* ============================================================
 * 9.2 §3（afterHeading）— LMDeploy TurboMind × W4A16
 * 原文：persistent batch = continuous batching 命名（N slots）；KV manager LRU recompute；
 *       Mem ≈ N×0.5 bytes：7B → 3.5 GB vs bf16 14 GB（4×）；V100 32 GB 装下余 28 GB；
 *       论文：至多 61% 降延迟（均值 30%）/ 156% 提吞吐（均值 58%），基线 = 混合精度框架
 * ============================================================ */
function lmdeployW4A16() {
  const UID = 'i9f';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'LMDeploy：persistent batch = continuous batching 命名 × W4A16 权重账 7B → 3.5 GB（bf16 的 1/4）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 324, h: 170, label: 'TurboMind（FasterTransformer lineage · C++）' }));
  body.push(cap(52, 70, 'persistent batch：N 个预配置 batch slots', { size: 8 }));
  const filled = [0, 1, 3, 4, 6];
  for (let i = 0; i < 8; i++) {
    const x = 52 + i * 36;
    if (filled.includes(i)) {
      const di = (0.2 + filled.indexOf(i) * 0.25).toFixed(2);
      body.push(`<rect class="dia-mx-on sig-cell" style="--i:${di}" x="${x}" y="76" width="30" height="24" rx="4"/>`);
    } else {
      body.push(`<rect class="dia-mx-off" x="${x}" y="76" width="30" height="24" rx="4"/>`);
    }
  }
  body.push(cap(52, 116, '有空槽即加入 · 生成完释放槽复用 · cache-hit 历史免重解码', { size: 7.5 }));
  body.push(cap(52, 134, 'turbomind.md 逐字：「You may recognize this', { size: 7, mono: true }));
  body.push(cap(52, 148, 'feature as continuous batching in other repos」', { size: 7, mono: true }));
  body.push(cap(52, 170, '= continuous batching 命名，非新机制（账回 4.1）', { size: 8, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 192, 'batch 自动增减以缩减计算', { size: 7.5 }));

  body.push(zone({ x: 376, y: 40, w: 308, h: 170, label: 'KV manager：memory-pool + LRU' }));
  body.push(cap(392, 72, '「cache of KV caches」', { size: 8.5, mono: true }));
  mk('ev', 392, 84, 276, 40, 'cache', '被驱逐序列 → 转存 token IDs', 'miss 时 FMHA 重解码恢复（recompute 式抢占）', 9, 6.3);
  body.push(cap(392, 140, '与 vLLM V1 recompute-only 抢占同族（成本账回 4.2 §2）', { size: 7.5 }));
  body.push(cap(392, 158, 'cutlass FMHA：Q/K 不等长 + INT8 KV', { size: 7.5, mono: true }));
  body.push(cap(392, 176, '双引擎：TurboMind + PyTorchEngine（纯 Python · Ascend）', { size: 7.5 }));
  body.push(cap(392, 194, 'DeepSeek PD disagg 经 DLSlime + Mooncake（PD 账归 8.4）', { size: 7.5 }));

  body.push(zone({ x: 36, y: 226, w: 648, h: 120, label: 'W4A16 AWQ 部署：权重 4 bit · 激活 16 bit（weight-only）' }));
  mk('fm', 52, 258, 180, 48, 'op', 'Mem ≈ N × 0.5 bytes', 'INT4 每权重元素 · A16 不计入权重账', 10, 6.3, true);
  body.push(cap(262, 256, '7B 权重显存账（理想账）', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-ink" x="262" y="262" width="196" height="16" rx="3"/></g>`);
  body.push(cap(360, 274, 'bf16 14 GB', { size: 7.5, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="262" y="288" width="49" height="16" rx="3"/></g>`);
  body.push(cap(286, 300, '3.5 GB', { size: 6.5, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(320, 300, 'W4A16 3.5 GB · group 128 scale 使实际略高（3.2）', { size: 7 }));
  body.push(cap(470, 274, '省 4×', { size: 9, weight: '650', color: 'var(--seal-deep)' }));
  mk('v1', 500, 258, 168, 48, 'norm', 'V100 32 GB 单卡装下', '仍余约 28 GB 给 KV + 激活', 9, 6.3);
  body.push(cap(52, 322, 'AWQ INT4（lmdeploy lite auto_awq）· 硬件 V100(sm70) / Turing / Ampere / Ada —— 老代际卡是差异化阵地', { size: 8 }));
  body.push(cap(52, 338, 'MXFP4 支持 NVIDIA V100+（gpt-oss H800 1.5× vLLM，README 披露口径）· InternLM 生态一等支持', { size: 8 }));

  body.push(cap(36, 366, '论文级贡献：双 hardware-aware 混合精度管线（GEMM offline packing + online acceleration · attention Q/K/V 精度组合）+ 四技术', { size: 9 }));
  body.push(cap(36, 384, '至多 61% 降延迟（均值 30%）/ 156% 提吞吐（均值 58%）—— 基线是现有混合精度框架，不是 vLLM 通用部署（200 ms → 78 / 140 ms）', { size: 9 }));
  body.push(cap(36, 402, '「× vLLM」倍率皆 workload 条件披露：internlm2-20b 1.8×（16+ RPS）· gpt-oss H800 1.5× —— 跨 workload 外推无效', { size: 9 }));
  return scene(720, 420, body, { label: 'LMDeploy 两笔账：TurboMind persistent batch 八个槽位请求有空槽即加入生成完释放，逐字等价 continuous batching；W4A16 权重账每参数零点五字节，7B 三点五 GB 对 bf16 十四 GB 省四倍，V100 三十二 GB 单卡装下仍余二十八 GB' }).svg();
}

/* ============================================================
 * 9.2 §4（afterHeading）— 五引擎选型决策表（签名）
 * 原文：选型 = 最硬约束先咬合（硬件锁定 / 模型族生态 / 特性 / 团队栈）；
 *       五行咬合场景 + PD 分离归 8.4 六轴表 + 两条边界论断
 * ============================================================ */
function engineDecision() {
  const UID = 'i9g';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9, ss = 6.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, '五引擎选型：不是「哪个最强」，而是「哪条约束先咬合」—— 最硬约束直接划掉多数候选', { size: 12, color: 'var(--ink-soft)' }));
  body.push(cap(36, 44, '约束（模型 × 硬件 × 特性 × 团队）', { size: 8, weight: '650' }));
  body.push(cap(472, 44, '咬合 → 选', { size: 8, weight: '650' }));

  const rows = [
    ['c1', 'e1', 'data', '通用开源模型 + 无专属硬件锁定', '要社区插件 / 特性广度 + 团队栈未定', 'proj', 'vLLM', 'PagedAttention + connector 插件化 PD · P5', 0.2],
    ['c2', 'e2', 'attn', '结构化程序 / 多调用共享前缀', 'agent、few-shot fork · 前缀复用率敏感', 'attn', 'SGLang', 'RadixAttention + 前端语言协同 · PD 内置 · P6', 0.5],
    ['c3', 'e3', 'cache', 'DeepSeek 族（V3 / V4.1）× SM100 / Ascend 950', '大规模 EP + 团队能自建编排', 'cache', 'DeepInfer', '专用 kernel 栈双平台 · 编排不在 release · P7', 0.8],
    ['c4', 'e4', 'proj', 'NVIDIA 专属栈 + Dynamo / Triton 编排', 'Blackwell 新代际首发（B200/GB200/B300/GB300）', 'proj', 'TRT-LLM', 'PyTorch-native + 手写 kernel 混合 · 勿按 legacy', 1.1],
    ['c5', 'e5', 'norm', 'InternLM 生态 + W4A16 混合精度', 'V100 级老卡或 Ascend（PyTorchEngine）', 'norm', 'LMDeploy', 'TurboMind + AWQ INT4 + 混合精度管线', 1.4],
  ];
  rows.forEach(([cid, eid, crole, ctitle, csub, erole, etitle, esub, dl], i) => {
    const y = 52 + i * 64;
    const c = mk(cid, 36, y, 400, 52, crole, ctitle, csub, 9, 6.5);
    const e = mk(eid, 472, y, 212, 52, erole, etitle, esub, 11, 6.3);
    body.push(edge(c, e, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(e.left[0], e.left[1], 1, 0));
  });

  body.push(cap(36, 378, 'PD 分离生产选型（直传 vs 存储中转 · 编排三型）→ 按 8.4 六轴对照表，本章不双展开', { size: 9, color: 'var(--seal-deep)' }));
  body.push(zone({ x: 36, y: 394, w: 648, h: 64, label: '两条边界论断' }));
  body.push(cap(52, 430, '硬件轴两种锁定：TRT-LLM 锁 NVIDIA 全代际（Ampere → Blackwell）；DeepInfer 锁 DeepSeek 模型族 × 双平台 —— 「NVIDIA 卡上跑 DeepSeek」两候选都合法', { size: 8 }));
  body.push(cap(52, 446, '分界在模型族专用优化与编排自建成本（7.1 / 8.4 §6）；生产选型以自有 workload 实测为准（benchmark 方法学归 1.1）', { size: 8 }));
  return scene(720, 470, body, { label: '五引擎选型决策表：五条约束场景各经脉冲连到咬合引擎——通用生态选 vLLM，结构化共享前缀选 SGLang，DeepSeek 族双平台选 DeepInfer，NVIDIA 专属 Blackwell 选 TRT-LLM，InternLM 老卡混合精度选 LMDeploy；PD 分离归八章四六轴表' }).svg();
}

/* ============================================================
 * 9.3 §2（afterHeading）— vLLM sleep/wake level 语义（签名 · S1 纠正后）
 * 原文（HEAD 6aeedab553dd 三文件互证）：
 *   level 1：discardable=∅，weights/runtime/cudagraph offload 到 host 备份，wake_up 自动拷回（无需重新注入）；
 *            kv_cache 永不备份（「KV cache and workspace scratch are never restored」）→ 仅 KV 丢弃；
 *   level 2：被改写集直接丢弃无备份（wake_up 后空显存由权重更新通道填回），preserved offload 到 CPU 恢复；KV 同样丢弃；
 *   7B 账：14 GB 权重 + 40 GB KV = 54 GB → 训练器 26 GB；sleep 后物理页全释放 → 80 GB（3.08×）
 * ============================================================ */
function sleepWakeLevels() {
  const UID = 'i9h';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'vLLM sleep/wake：level 1 权重 offload 到 CPU 自动恢复，level 2 丢弃被改写集 —— KV 两 level 均丢弃', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 324, h: 240, label: 'sleep(level 1)：权重 offload · 仅 KV 丢弃' }));
  const w1 = mk('w1', 52, 72, 130, 44, 'cache', 'weights 14 GB', '7B bf16 = 2N_a', 9.5, 6.3, true);
  const k1 = mk('k1', 212, 72, 130, 44, 'attn', 'kv_cache 40 GB', '教学部署值', 9, 6.3, true);
  const cpu1 = mk('cpu1', 52, 180, 130, 44, 'proj', 'CPU host 备份', '14 GB 权重', 9, 6.5);
  const dis1 = mk('dis1', 212, 180, 130, 44, 'plain', '丢弃 · 永不备份', '物理页释放 → 须重建', 9, 6.3);
  body.push(sealDash(212, 180, 130, 44));
  body.push(edge(w1, cpu1, { nodes, portFrom: [100, 116], portTo: [100, 180], pulse: true, dur: 1.4, delay: 0.3, uid: UID, eid: 'o1' }));
  body.push(arrowHead(100, 180, 0, 1));
  body.push(edge(cpu1, w1, { nodes, portFrom: [134, 180], portTo: [134, 116], pulse: true, dur: 1.4, delay: 1.6, uid: UID, eid: 'r1' }));
  body.push(arrowHead(134, 116, 0, -1));
  body.push(cap(96, 150, 'offload', { size: 6.5, anchor: 'end', mono: true }));
  body.push(cap(140, 150, 'wake_up 自动拷回', { size: 6.5 }));
  body.push(edge(k1, dis1, { nodes, portFrom: k1.bottom, portTo: dis1.top, pulse: true, dur: 1.2, delay: 0.6, dash: true, uid: UID, eid: 'k1' }));
  body.push(arrowHead(dis1.top[0], dis1.top[1], 0, 1));
  body.push(cap(283, 150, 'KV 丢弃', { size: 6.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 240, 'wake_up：cudaMemcpy 自动拷回原权重（L386-387）', { size: 7.5 }));
  body.push(cap(52, 256, '无需重新注入 · discardable = 空集', { size: 7.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 272, '代价：全量 host 备份 + wake_up 的 H2D 回拷', { size: 7.5 }));

  body.push(zone({ x: 376, y: 40, w: 308, h: 240, label: 'sleep(level 2)：丢被改写集 · 保留非改写集' }));
  const dw = mk('dw', 392, 72, 130, 44, 'attn', 'discardable', '被改写集（RL 全参更新）', 9, 6, true);
  const pv = mk('pv', 540, 72, 128, 44, 'cache', 'preserved', '非改写集（如 LoRA base）', 9, 6, true);
  const dis2 = mk('dis2', 392, 180, 130, 44, 'plain', '直接丢弃 · 无备份', 'wake_up 空显存 → 更新填回', 8.5, 5.8);
  body.push(sealDash(392, 180, 130, 44));
  const cpu2 = mk('cpu2', 540, 180, 128, 44, 'proj', 'CPU offload', 'wake_up 恢复', 9, 6.5);
  body.push(edge(dw, dis2, { nodes, portFrom: dw.bottom, portTo: dis2.top, pulse: true, dur: 1.2, delay: 0.4, dash: true, uid: UID, eid: 'o2' }));
  body.push(arrowHead(dis2.top[0], dis2.top[1], 0, 1));
  body.push(cap(463, 150, '丢弃', { size: 6.5, color: 'var(--seal-deep)' }));
  body.push(edge(pv, cpu2, { nodes, portFrom: [588, 116], portTo: [588, 180], pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 'o3' }));
  body.push(arrowHead(588, 180, 0, 1));
  body.push(edge(cpu2, pv, { nodes, portFrom: [620, 180], portTo: [620, 116], pulse: true, dur: 1.4, delay: 1.9, uid: UID, eid: 'r2' }));
  body.push(arrowHead(620, 116, 0, -1));
  body.push(cap(584, 150, 'offload', { size: 6.5, anchor: 'end', mono: true }));
  body.push(cap(626, 150, 'wake', { size: 6.5, mono: true }));
  body.push(cap(392, 240, 'kv_cache 同样丢弃 —— 两 level KV 永不备份（5.4 命题 4）', { size: 7.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(392, 256, 'RL 全参更新：actor 权重每轮被改写 → preserved 很小', { size: 7.5 }));
  body.push(cap(392, 272, '价值在部分参数不更新场景 · 免掉被改写集备份纯浪费', { size: 7.5 }));

  body.push(zone({ x: 36, y: 296, w: 648, h: 118, label: '显存交还账（7B · 单卡 80 GB）× wake_up(tags)' }));
  body.push(cap(52, 330, 'sleep 前：54 GB 占用 → 训练器可用 26 GB', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-warm" x="52" y="338" width="70" height="16" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.35"><rect class="dia-bar dia-bar-ink" x="125" y="338" width="200" height="16" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-soft" x="328" y="338" width="130" height="16" rx="3"/></g>`);
  body.push(cap(87, 350, '权重 14', { size: 6.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(225, 350, 'KV 40', { size: 6.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(393, 350, '训练器 26 GB', { size: 6.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(52, 368, 'sleep 后：物理页全释放（虚拟地址保留）', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.8"><rect class="dia-bar dia-bar-seal" x="52" y="374" width="406" height="16" rx="3"/></g>`);
  body.push(cap(255, 386, '训练器可用升到 80 GB 量级（80/26 ≈ 3.08×）', { size: 7, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(470, 350, '进程不杀 · 无冷启动', { size: 7.5 }));
  body.push(cap(470, 366, '免重载权重 / CUDA context / 通信组', { size: 7.5 }));
  body.push(cap(470, 382, '代价：切换时间 + KV 重建（§4 的 s 项）', { size: 7.5 }));
  body.push(cap(52, 402, 'wake_up(tags)：weights / kv_cache 是 deferrable tags，按 tag 选择性恢复 + 重建 kv_transfer group —— 粒度是 tag 池，不是全有全无', { size: 8 }));

  body.push(cap(36, 434, '三文件互证（gpu_worker L368-371 / CuMemBackend.suspend L126 / cumem L292）：level 1 不是「丢弃全部权重」—— 写反会每轮多付一次本不需要的权重同步', { size: 9 }));
  body.push(cap(36, 452, 'veRL vllm_rollout：release → sleep.remote / resume → wake_up.remote(tags)；sleep_level 由 layered_summon 判（开启 → level 1，带 overflow 警告）', { size: 9 }));
  return scene(720, 466, body, { label: 'vLLM sleep wake 两 level 语义：level 1 权重十四 GB offload 到 CPU 备份 wake_up 自动拷回无需重新注入，仅 KV 丢弃永不备份；level 2 被改写集直接丢弃无备份由权重更新通道填回，preserved 非改写集 offload 到 CPU 恢复，KV 同样丢弃；单卡八十 GB sleep 前训练器可用二十六 GB sleep 后升到八十 GB 量级' }).svg();
}

/* ============================================================
 * 9.3 §4（block 0）— colocate vs disagg rollout 时序与 GPU·时间账
 * 原文：T_co = T_r+T_t+s = 620 s；T_dis = max(T_r,T_t)+r = 305.6 s；
 *       t_sync ≥ 2N_a/BW（70B：600 GB/s 0.23 s / 25 GB/s 5.6 s）；
 *       W_co = 39,680 vs W_dis = 39,116.8 GPU·s（省 1.4%）；判据 2r < s；r=15 s 反例
 * ============================================================ */
function colocateDisagg() {
  const UID = 'i9i';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = true) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'colocate 串行 620 s vs disagg 并行 305.6 s —— 墙钟 2× 但卡数 2×，GPU·时间几乎打平', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 150, label: '每轮迭代时序（T_r = T_t = 300 s · s = 20 s · r = 5.6 s 教学构造值）' }));
  body.push(cap(52, 80, 'colocate（n 卡时分复用）', { size: 8.5, weight: '650' }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-seal" x="140" y="84" width="225" height="22" rx="4"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-ink" x="368" y="84" width="225" height="22" rx="4"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-warm" x="596" y="84" width="15" height="22" rx="3"/></g>`);
  body.push(cap(252, 99, 'rollout T_r', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(480, 99, 'train T_t', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(603, 99, 's', { size: 6.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(620, 99, '620 s', { size: 8, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(140, 120, '同一批卡交替占两相 · 迭代 = T_r + T_t + s（s = 切换 + 本地同步）', { size: 7.5 }));
  const coPath = 'M140 95 H611';
  body.push(pathPulse(coPath, { dur: 2.6, delay: 0.4, r: 3 }));

  body.push(cap(52, 142, 'disagg 训练卡', { size: 8 }));
  body.push(cap(52, 168, 'disagg rollout 卡', { size: 8 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-ink" x="140" y="136" width="225" height="18" rx="4"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.7"><rect class="dia-bar dia-bar-warm" x="368" y="136" width="12" height="18" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-seal" x="140" y="160" width="225" height="18" rx="4"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.7"><rect class="dia-bar dia-bar-warm" x="368" y="160" width="12" height="18" rx="3"/></g>`);
  body.push(cap(252, 149, 'train T_t', { size: 7, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(252, 173, 'rollout T_r', { size: 7, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(388, 150, '迭代 = max(T_r, T_t) + r = 305.6 s', { size: 8, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(388, 172, 'r = 跨集群权重广播 · 两相并行（两卡池同时忙）', { size: 7.5 }));
  body.push(pathPulse('M140 145 H365', { dur: 1.8, delay: 0.6, r: 3 }));
  body.push(pathPulse('M140 169 H365', { dur: 1.8, delay: 0.6, r: 3 }));

  body.push(zone({ x: 36, y: 206, w: 324, h: 120, label: '同步下界（式 3 · training 8.5 §4 译文）' }));
  mk('sy', 52, 238, 290, 40, 'op', 't_sync ≥ 2N_a / BW', '70B bf16：2N_a = 1.4×10¹¹ bytes', 11, 6.5);
  body.push(cap(52, 296, 'NVLink 600 GB/s → ≈ 0.23 s（节点内）', { size: 8, mono: true }));
  body.push(cap(52, 312, '跨节点 25 GB/s → ≈ 5.6 s（disagg r 起步）', { size: 8, mono: true, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 376, y: 206, w: 308, h: 120, label: 'GPU·时间账（n = m = 64 · 命题 1）' }));
  body.push(cap(392, 236, 'colocate：64×620 = 39,680 GPU·s', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-ink" x="392" y="242" width="248" height="12" rx="3"/></g>`);
  body.push(cap(392, 266, 'disagg：128×305.6 = 39,116.8 GPU·s', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="392" y="272" width="244" height="12" rx="3"/></g>`);
  body.push(cap(392, 300, 'disagg 省 563.2 GPU·s（≈1.4%）· 判据 2r < s', { size: 8, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(392, 316, '2r = 11.2 < 20 ✓ · 反例 r=15 s → 30 > 20 colocate 反超', { size: 7.5 }));

  body.push(cap(36, 346, '读账①：平衡负载 GPU·时间差由切换/同步开销项决定，不由并行度决定 ② 决策变量 = 卡数预算（colocate 省卡）vs 迭代延迟（disagg 快）', { size: 9 }));
  body.push(cap(36, 364, '③ 异步潜力：disagg rollout 卡可超前跑，colocate 不能 —— 负载不平衡（T_r ≈ 4T_t）时同步 disagg 训练卡空闲，GPU·时间反而可能更高', { size: 9 }));
  body.push(cap(36, 382, '代码对应物：veRL V1 trainer_colocate_async.py（同卡池）/ trainer_separate_async.py（独立 rollout 卡 + hybrid 池）；OpenRLHF 70B+ 默认四模型分卡', { size: 9 }));
  body.push(cap(36, 400, '放置与同步带宽是设计输入：等跑起来才发现同步走慢链路，再改放置的代价就高了（training 8.5 §4 提点）', { size: 9 }));
  return scene(720, 414, body, { label: 'colocate 与 disagg 时序对照：colocate 同一批卡 rollout 训练切换串行六百二十秒，disagg 两卡池并行三百零五点六秒；同步下界七十 B 跨节点五点六秒；GPU 时间三万九千六百八十对三万九千一百一十六点八几乎打平，判据二 r 小于 s' }).svg();
}

/* ============================================================
 * 9.3 §5（block 1）— partial rollout abort-retain-resume 数据流
 * 原文：v_i rollout → T_pre=8192 → abort 保留 tokens+logprobs（replay buffer/LB）→
 *       权重更新 v_i→v_{i+1} → LB 重试 → 前缀 KV 重建（一次 prefill ≈284 ms）→ 续 decode；
 *       重建 vs 重采 ≈144×；命题 2 轨迹内版本混合；staleness 阈值 8 版本 drop/wait
 * ============================================================ */
function partialRollout() {
  const UID = 'i9j';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'partial rollout：abort 保留 tokens+logprobs → 新版本续跑 —— 前缀重建 284 ms ≪ 重采 41 s（≈144×）', { size: 12, color: 'var(--ink-soft)' }));

  const n1 = mk('n1', 36, 60, 150, 52, 'data', 'v_i rollout', 'prompt → decode …', 10, 6.5);
  const n2 = mk('n2', 216, 60, 150, 52, 'cache', '已生成 T_pre = 8192', 'abort（迭代边界 / 资源切换）', 8.5, 6);
  const n3 = mk('n3', 396, 60, 150, 52, 'proj', '保留 tokens + logprobs', 'replay buffer / LB · 不在引擎显存', 8, 6);
  const n4 = mk('n4', 576, 60, 108, 52, 'op', '权重更新', 'v_i → v_{i+1}', 9.5, 6.5);
  [[n1, n2, 0.2], [n2, n3, 0.6], [n3, n4, 1.0]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  const n5 = mk('n5', 576, 150, 108, 52, 'attn', 'LB 重试', '前缀+续跑重新入引擎', 9.5, 5.8);
  const n6 = mk('n6', 396, 150, 150, 52, 'cache', '前缀 KV 重建', '一次 prefill ≈ 284 ms（4.2 式 1）', 9.5, 6);
  const n7 = mk('n7', 216, 150, 150, 52, 'norm', 'v_{i+1} 继续 decode', '→ EOS / 下一次 abort', 8.5, 6.5);
  body.push(edge(n4, n5, { nodes, portFrom: n4.bottom, portTo: n5.top, pulse: true, dur: 1, delay: 1.4, uid: UID, eid: 'f3' }));
  body.push(arrowHead(n5.top[0], n5.top[1], 0, 1));
  body.push(edge(n5, n6, { nodes, portFrom: n5.left, portTo: n6.right, pulse: true, dur: 1, delay: 1.8, uid: UID, eid: 'f4' }));
  body.push(arrowHead(n6.right[0], n6.right[1], -1, 0));
  body.push(edge(n6, n7, { nodes, portFrom: n6.left, portTo: n7.right, pulse: true, dur: 1, delay: 2.2, uid: UID, eid: 'f5' }));
  body.push(arrowHead(n7.right[0], n7.right[1], -1, 0));

  body.push(zone({ x: 36, y: 140, w: 160, h: 84, label: '重建 vs 重采' }));
  body.push(cap(52, 180, '重建 ≈ 284 ms', { size: 7 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="52" y="186" width="5" height="9" rx="2"/>`);
  body.push(cap(52, 206, '重采 ≈ 41 s', { size: 7 }));
  body.push(`<rect class="dia-bar dia-bar-ink" x="52" y="212" width="120" height="9" rx="2"/>`);
  body.push(cap(134, 196, '≈144×', { size: 8, weight: '650', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 240, w: 648, h: 120, label: '命题 2：轨迹内策略版本混合 × staleness 控制' }));
  const mx = mk('mx', 52, 272, 300, 48, 'attn', '前缀采样自 v_i · 后缀采样自 v_{i+k}', '整条轨迹 = 分段混合分布，不是单一策略', 9, 6.5);
  const al = mk('al', 382, 272, 286, 48, 'proj', '算法侧承接：序列级 ratio（GSPO）', '截断 IS / 拒采样（training 8.5 §5）· 本章不重推', 8.5, 6.3);
  body.push(edge(mx, al, { nodes, pulse: true, dur: 1.2, delay: 0.8, uid: UID, eid: 'm0' }));
  body.push(arrowHead(al.left[0], al.left[1], 1, 0));
  body.push(cap(52, 336, 'staleness：max_off_policy_threshold 默认 8 个模型版本 · strategy drop（丢弃重采）或 wait（等待追平）· trajectory_spans 监控', { size: 8 }));
  body.push(cap(52, 352, '边界：abort 与续跑之间权重未更新（版本不变）→ 退化为单一策略；混合取决于是否跨版本', { size: 8 }));

  body.push(cap(36, 380, 'V1 async client 行为，不是配置开关：veRL mainline = trainer/ppo/v1/（colocate_async / separate_async + replay_buffer + TransferQueue）', { size: 9 }));
  body.push(cap(36, 398, 'fully_async / one_step_off 文档现为 stub —— 已迁 verl-recipe（pinned v0.9.1，V0 trainer）；按旧文档写 veRL 当前异步形态是坑', { size: 9 }));
  body.push(cap(36, 416, '两层接口各司其职：轨迹层暂停/恢复保 tokens+logprobs（replay buffer/LB）；引擎层 sleep/wake 保虚拟地址释放物理页 —— 勿混', { size: 9 }));
  return scene(720, 430, body, { label: 'partial rollout 数据流：版本 vi rollout 生成八千一百九十二 token 后迭代边界 abort，保留 tokens 与 logprobs 在 replay buffer，权重更新到 vi 加一后 LB 重试重新入引擎，前缀 KV 一次 prefill 重建约二百八十四毫秒继续 decode，比重采四十一秒便宜约一百四十四倍；轨迹内策略版本混合归序列级 ratio 承接' }).svg();
}

/* ============================================================
 * 9.3 §7（afterHeading）— rollout 吞吐账 × 尾部下界 × 投机解码
 * 原文：每轮 512×8×4096 = 1.678e7 token，Θ=2e5 → 83.9 s；命题 3 下界 32768×5 ms = 164 s；
 *       F ≈ 2.40（α=0.8 γ=4 c=0.1）→ 35 s，分布保持前提；veRL 三引擎 vs OpenRLHF vLLM-only
 * ============================================================ */
function rolloutThroughput() {
  const UID = 'i9k';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = true) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'rollout 吞吐账：平均 83.9 s vs 尾部下界 164 s —— 投机解码 ×2.40 压到 35 s（分布保持前提）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 120, label: '平均口径（式 6）· 聚合吞吐 2×10⁵ tokens/s 教学构造值' }));
  const tk = mk('tk', 52, 72, 200, 48, 'data', '每轮 1.678×10⁷ token', 'n_prompt·G·L̄_out = 512×8×4096', 9.5, 6.3);
  const th = mk('th', 282, 72, 150, 48, 'cache', 'Θ = 2×10⁵ tok/s', '集群聚合 decode 吞吐', 10, 6.3, false);
  const t8 = mk('t8', 462, 72, 100, 48, 'op', '≈ 83.9 s', '平均口径', 12, 6.5);
  body.push(edge(tk, th, { nodes, pulse: true, dur: 1, delay: 0.3, uid: UID, eid: 'a0' }));
  body.push(arrowHead(th.left[0], th.left[1], 1, 0));
  body.push(edge(th, t8, { nodes, pulse: true, dur: 1, delay: 0.7, uid: UID, eid: 'a1' }));
  body.push(arrowHead(t8.left[0], t8.left[1], 1, 0));
  body.push(cap(580, 92, '单卡装不下全批：', { size: 7.5 }));
  body.push(cap(580, 108, '2 TiB KV ≫ 48 GiB 池', { size: 7.5, mono: true }));
  body.push(cap(52, 140, 'GRPO 每 prompt 采 G 条：组大小翻倍 → KV 与生成 token 数同步翻倍（training 8.5 §2「省下的预算搬到 KV 项」）', { size: 8 }));

  body.push(zone({ x: 36, y: 176, w: 324, h: 130, label: '命题 3：rollout 时长下界（结构结论非拟合）' }));
  mk('lb', 52, 208, 290, 44, 'attn', 'T_roll ≥ max T_i × t_step,min', '自回归串行 · decode memory-bound 单步下界', 10, 6.3);
  body.push(cap(52, 270, '32768 × 5 ms = 163.84 ≈ 164 s', { size: 9, mono: true, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 288, '164 s > 83.9 s —— 平均口径低估，尾部决定时长', { size: 8, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 376, y: 176, w: 308, h: 130, label: '投机解码 = 吞吐乘数（4.4 命题 3 recap）' }));
  mk('ff', 392, 208, 276, 44, 'proj', 'F = (1−α^(γ+1))/((1−α)(γc+1))', 'α=0.8 · γ=4 · c=0.1 → F ≈ 2.40', 8.5, 6.5);
  body.push(cap(392, 270, '83.9 / 2.40 ≈ 35 s', { size: 9, mono: true, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(392, 288, '前提：分布保持 —— rollout 采样语义不被破坏', { size: 8 }));
  body.push(cap(392, 302, 'rollout_log_probs 供训练侧 bypass / decoupled 校正', { size: 7 }));

  body.push(zone({ x: 36, y: 322, w: 648, h: 76, label: '时长对照（s）' }));
  [['平均口径', 83.9, 'soft', 352, 0.3], ['尾部下界（命题 3）', 164, 'seal', 368, 0.6], ['投机 ×2.40', 35, 'ink', 384, 0.9]].forEach(([lab, v, tone, y, di]) => {
    body.push(cap(52, y + 9, lab, { size: 7.5 }));
    const w = v * 2.4;
    body.push(`<g class="sig-bar" style="--i:${di}"><rect class="dia-bar dia-bar-${tone}" x="140" y="${y}" width="${w.toFixed(0)}" height="10" rx="3"/></g>`);
    body.push(cap(146 + w, y + 9, `${v} s`, { size: 7.5, mono: true, color: tone === 'seal' ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });

  body.push(cap(36, 416, 'partial rollout 把迭代边界与序列完成解耦 → 下界转而约束单条轨迹累计墙钟，不再约束单轮迭代（缓解尾部阻塞的机制）', { size: 9 }));
  body.push(cap(36, 434, '引擎接口选型：veRL 三引擎 backend（vllm / sglang / trtllm）= 引擎可换；OpenRLHF vLLM-only = 单引擎深耕（NCCL + IPC 双路径）', { size: 9 }));
  return scene(720, 448, body, { label: 'rollout 吞吐账：每轮一千六百七十八万 token 除以集群吞吐每秒二十万得平均八十三点九秒，命题三尾部下界最长序列三万二千七百六十八乘五毫秒得一百六十四秒超过平均，投机解码乘二点四压到三十五秒前提是分布保持' }).svg();
}

/* ============================================================
 * 9.4 §1（block 0）— 多模态请求生命周期
 * 原文：tokenize（每图 1 占位）→ overwrite_prompt 展开（CPU 预处理独立进程池）→
 *       VE LB（machine 随机 + fewest patches）→ VE 单 GPU cheaper（ViT → 3×3 downsample →
 *       MLP projector）→ embedding 合并 → LLM prefill（可 chunk）→ decode → 流式返回；
 *       TTFT = t_queue+t_pre+t_VE+t_prefill，0.5 → 0.56 s（+12% 教学假设）
 * ============================================================ */
function mmLifecycle() {
  const UID = 'i9l';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, '多模态请求生命周期：图像段插入三段 —— VE 单 GPU cheaper，TTFT 串行上界 +12%', { size: 12, color: 'var(--ink-soft)' }));

  const rq = mk('rq', 36, 56, 130, 52, 'data', '客户端请求', '文本 + 图像字节 + (w,h)', 9.5, 6);
  const tk = mk('tk', 196, 56, 120, 52, 'op', 'API tokenize', '每图一个占位 token', 9.5, 6);
  const ex = mk('ex', 346, 56, 150, 52, 'proj', '占位符展开', 'overwrite_prompt · CPU 预处理', 9, 6);
  const lb = mk('lb', 526, 56, 158, 52, 'cache', 'VE LB 路由', 'machine 随机 + fewest patches', 9, 6);
  [[rq, tk, 0.2], [tk, ex, 0.5], [ex, lb, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `a${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 124, 'estimate_num_tokens 只跑尺寸规划', { size: 7 }));
  body.push(cap(36, 138, '不碰图像本体 → LB 与 prefill 预算提前算账', { size: 7 }));
  body.push(cap(346, 124, '独立 CPU 进程池：decode / resize', { size: 7 }));
  body.push(cap(346, 138, 'min_pixels / max_image_tokens 管线', { size: 7 }));

  const ve = mk('ve', 526, 150, 158, 52, 'attn', 'VE 单 GPU · cheaper', 'ViT n_vit patches → 3×3 downsample', 9, 6);
  const pj = mk('pj', 346, 150, 150, 52, 'proj', 'MLP projector', '[n_llm, d_model] embedding', 9.5, 6);
  const mg = mk('mg', 196, 150, 120, 52, 'norm', 'embedding 合并', 'IMAGE 位置替换', 8.5, 6);
  const pf = mk('pf', 36, 150, 130, 52, 'cache', 'LLM prefill', '可 chunk · embedding 按 chunk 切片', 9, 5.8);
  body.push(edge(lb, ve, { nodes, portFrom: lb.bottom, portTo: ve.top, pulse: true, dur: 1, delay: 1.1, uid: UID, eid: 'b0' }));
  body.push(arrowHead(ve.top[0], ve.top[1], 0, 1));
  [[ve, pj, 1.4], [pj, mg, 1.7], [mg, pf, 2.0]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.left, portTo: b.right, pulse: true, dur: 1, delay: dl, uid: UID, eid: `b${i + 1}` }));
    body.push(arrowHead(b.right[0], b.right[1], -1, 0));
  });
  body.push(cap(526, 218, 'doc L51：single GPU per instance', { size: 7, mono: true }));
  body.push(cap(526, 232, 'cheaper GPUs recommended', { size: 7, mono: true }));
  body.push(cap(346, 218, '3×3 邻域 channel-concat · 首层入维 9216', { size: 7 }));
  body.push(cap(196, 218, '布局 token 走 special_embeds（不走 ViT）', { size: 7 }));

  const dc = mk('dc', 36, 240, 130, 52, 'attn', 'decode 循环', '与 1.1 完全同形 · 无多模态特例', 9.5, 5.8);
  const dt = mk('dt', 196, 240, 120, 52, 'op', 'detokenize', null, 9.5);
  const rt = mk('rt', 346, 240, 150, 52, 'data', '流式返回', 'TTFT = queue+pre+VE+prefill', 9.5, 6);
  body.push(edge(pf, dc, { nodes, portFrom: pf.bottom, portTo: dc.top, pulse: true, dur: 1, delay: 2.3, uid: UID, eid: 'c0' }));
  body.push(arrowHead(dc.top[0], dc.top[1], 0, 1));
  [[dc, dt, 2.6], [dt, rt, 2.9]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 0.9, delay: dl, uid: UID, eid: `c${i + 1}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 308, w: 648, h: 80, label: 'TTFT 串行分解（式 1）· 教学假设非实测' }));
  mk('tf', 52, 340, 240, 40, 'op', 'TTFT = t_queue + t_pre + t_VE + t_prefill', null, 8, 6.5, true);
  body.push(cap(312, 344, '纯文本基线 0.5 s → 带图 ≈ 0.56 s（+12%）', { size: 8.5, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(312, 362, 't_pre = 5 ms · t_VE = 55.7 ms（896×896 · 假设 100 TFLOPS）', { size: 7.5 }));
  body.push(cap(312, 378, 'VE 预取与排队重叠 → t_VE 部分或全部藏进前段（与 6.2 bubble 消除同构）', { size: 7.5 }));

  body.push(cap(36, 408, 'DeepInfer 预处理独立 CPU 进程池（MultimodalProcessor fork worker · pinned host 共享内存池按 rank 常驻）', { size: 9 }));
  body.push(cap(36, 426, 'vLLM 同构三件：独立预处理 + image hash 进 prefix caching + encoder cache；VE-P-D 编排层不在 release（8.4 §6）', { size: 9 }));
  body.push(cap(36, 444, '本章走读均为引擎侧 release 代码（main@a679bb86）', { size: 9 }));
  return scene(720, 458, body, { label: '多模态请求生命周期：客户端请求经 tokenize 每图一个占位 token，占位符展开由独立 CPU 进程池预处理，VE LB 按 machine 随机加 fewest patches 路由到单 GPU cheaper 的 VE，ViT 前向经三乘三降采样与 MLP projector 输出 embedding，合并进 LLM prefill 可 chunk，decode 流式返回与纯文本同形；TTFT 串行分解带图加约百分之十二' }).svg();
}

/* ============================================================
 * 9.4 §2（afterHeading）— image patch 账（签名）
 * 原文：896=64×14 → n_vit=4096；ceil(64/3)=22 每维 → n_llm=484；n_expand=22×23+2=508≤1024；
 *       坑 ceil(4096/9)=456≠484 差 28；C_ViT = 2N_VE n + 4L_VE n²d_VE（系数 4）；
 *       5.57 TFLOPs/图 attention 39.5% · N_VE≈0.411B · n*=6272；单 GPU ~18 图/s；缩减 ≈8.5×
 * ============================================================ */
function patchAccount() {
  const UID = 'i9m';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = true) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'image patch 账：896×896 → 4096 patches → 484 图像 token → 508 展开 —— 每维 ceil，不是整体 ceil', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 140, label: '主例 896×896（vision_config [W]：patch_size 14 · downsample_ratio 3）' }));
  const p1 = mk('p1', 52, 72, 140, 52, 'data', '896 = 64×14', 'patch 对齐尺寸', 11, 6.5);
  const p2 = mk('p2', 222, 72, 140, 52, 'cache', 'n_vit = 64×64', '= 4096 patches', 10, 6.5);
  const p3 = mk('p3', 392, 72, 140, 52, 'attn', 'ceil(64/3) = 22', '两维分别向上取整', 10, 6.5);
  const p4 = mk('p4', 562, 72, 106, 52, 'op', 'n_llm = 484', '22×22', 10, 6.5);
  [[p1, p2, 0.2], [p2, p3, 0.6], [p3, p4, 1.0]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(52, 140, 'n_expand = 22×(22+1)+2 = 508 ≤ max_image_tokens 1024 → 不触发缩小（其中 24 个布局 special_embeds）', { size: 8 }));
  body.push(cap(52, 158, '布局：[IMAGE_START] + ([IMAGE]×22 + [NEW_LINE])×22 + [IMAGE_END] —— 排布唯一', { size: 7.5, mono: true }));
  body.push(cap(52, 174, 'LLM prefill 只处理 508 个图像段 token；ViT 算全部 4096 个 patch 的 attention', { size: 7.5 }));

  body.push(zone({ x: 36, y: 196, w: 324, h: 110, tone: 'seal', label: '坑：整体 ceil ≠ 每维 ceil' }));
  mk('wr', 52, 228, 130, 44, 'attn', 'ceil(4096/9)', '= 456（错）', 10, 6.5);
  mk('ri', 212, 228, 130, 44, 'norm', '22×22 = 484', '每维 ceil 再乘（对）', 10, 6);
  body.push(cap(197, 254, '≠', { size: 12, anchor: 'middle', weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(52, 290, '少 28 → 占位展开与 embedding 行数错位，无法对位填入', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 376, y: 196, w: 308, h: 110, label: 'VE 算力账（式 5）· 双向全量 attention' }));
  mk('cv', 392, 228, 276, 44, 'proj', 'C = 2N_VE·n + 4L_VE·n²·d_VE', '系数 4 非 2（ViT 不打 causal mask）', 9, 6.3);
  body.push(cap(392, 290, '5.57 TFLOPs/图 · attention 39.5% · N_VE ≈ 0.411B', { size: 7.5, mono: true }));
  body.push(cap(392, 302, 'n* = 6272：超过它 attention 主导', { size: 7 }));

  body.push(zone({ x: 36, y: 322, w: 648, h: 96, label: '缩减的 serving 含义：在小模型域付算力、在大模型域省 token' }));
  body.push(cap(52, 360, 'ViT patch 数', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-ink" x="140" y="350" width="341" height="14" rx="3"/></g>`);
  body.push(cap(310, 361, '4096', { size: 7.5, anchor: 'middle', mono: true, color: 'var(--paper-raised)' }));
  body.push(cap(52, 382, 'LLM 图像 token', { size: 7.5 }));
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="140" y="372" width="40" height="14" rx="3"/></g>`);
  body.push(cap(186, 383, '484 · 缩减 ≈8.5×（含布局 508 → 8.06×）', { size: 7.5, color: 'var(--seal-deep)' }));
  mk('v1', 500, 346, 168, 48, 'norm', 'VE 单 GPU cheaper', '55.7 ms/图 · ~18 图/s', 9, 6);
  body.push(cap(500, 406, '假设 100 TFLOPS · cheaper 无具体 SKU 披露', { size: 6.5 }));

  body.push(cap(36, 438, '边界实例：448² < min_pixels 295936 → 放大 546² → 1521/169/184；1920×1080 → 10764 patches → 展开 1224 > 1024 → 缩小 8418/943/968', { size: 9 }));
  body.push(cap(36, 456, 'LLM 侧 prefill 算力、KV 容量、decode attention 读取全按 n_vit/n_llm ≈ 8.46× 缩减 —— VE 越便宜这笔交换越划算（对照 7.4 重型并行）', { size: 9 }));
  return scene(720, 470, body, { label: 'image patch 账：八百九十六乘八百九十六图 patch 对齐六十四乘十四，ViT 见四千零九十六个 patch，三乘三降采样每维向上取整六十四除三得二十二，LLM 见四百八十四个图像 token，加布局 token 展开五百零八不超一千零二十四上限；整体 ceil 得四百五十六少二十八是坑；VE 每图五点五七 TFLOPs 单 GPU cheaper 约十八图每秒' }).svg();
}

/* ============================================================
 * 9.4 §3（afterHeading）— VE LB 代理质量 × encoder cache 节省
 * 原文：命题 1 反例 1×4096 vs 4×1024：attention 2.20e12 vs 5.50e11（4×），总 FLOPs 1.42×；
 *       两层近似（LB 尺寸直算 1024 vs resize 实际 1521）；E[C]=(1−η)C：η=0.8 → 1.11 TFLOPs ≈ 11.1 ms 省 80%；
 *       每图 embedding 4.73 MiB，1 GiB ≈ 216 图；权重更新 → reset 全量失效
 * ============================================================ */
function veLbEncoderCache() {
  const UID = 'i9n';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, '两笔节省账：fewest-patches 单调代理非等价（attention 4×）× encoder cache η=0.8 省 80%', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 324, h: 180, label: '命题 1 反例：等总 patch，大图集中负载更高' }));
  mk('ga', 52, 72, 130, 48, 'attn', 'GPU A', '1×4096-patch 图', 10, 6.5);
  mk('gb', 212, 72, 130, 48, 'norm', 'GPU B', '4×1024-patch 图', 10, 6.5);
  body.push(cap(52, 136, '总 patch 数同为 4096 · 线性项相同', { size: 7.5 }));
  body.push(cap(52, 158, 'A attention', { size: 7 }));
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-seal" x="140" y="148" width="110" height="12" rx="3"/></g>`);
  body.push(cap(256, 158, '2.20×10¹² · 4×', { size: 7, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(52, 178, 'B attention', { size: 7 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-ink" x="140" y="168" width="28" height="12" rx="3"/></g>`);
  body.push(cap(174, 178, '5.50×10¹¹', { size: 7, mono: true }));
  body.push(cap(52, 196, '总 FLOPs 5.57 vs 3.92 TFLOPs（比 1.42×）', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 212, 'fewest-patches 系统性低估大图集中的 GPU', { size: 7.5 }));

  body.push(zone({ x: 376, y: 40, w: 308, h: 180, label: '两层近似 × doc 仍取 patch 代理的理由' }));
  body.push(cap(392, 72, '① LB 尺寸直算 vs resize 管线实际：', { size: 8 }));
  body.push(cap(392, 90, '448×448 → 1024（LB 估）vs 1521（实际）', { size: 8, mono: true }));
  body.push(cap(392, 112, '② 二次项不跨图合并：Σ C(n_i) ≠ C(Σ n_i)', { size: 8 }));
  body.push(cap(392, 134, 'doc 取 patch 代理的理由：轻量 —— 尺寸直算、', { size: 7.5 }));
  body.push(cap(392, 150, '无需先跑 encoder、无需知道图数分布（路由前预估）', { size: 7.5 }));
  body.push(cap(392, 172, 'fewest-patches = 两层近似下的粗均衡，', { size: 7.5 }));
  body.push(cap(392, 188, '不是 FLOPs 精确均衡；命题 1 给方向与量级', { size: 7.5 }));
  body.push(cap(392, 208, 'ViT attention 逐图 varlen 全量（vit_attn.py L16）', { size: 7 }));

  body.push(zone({ x: 36, y: 236, w: 648, h: 130, label: 'encoder cache：hash 跨请求保留 embedding，命中跳过 ViT 前向' }));
  mk('ec', 52, 276, 180, 48, 'cache', 'E[C] = (1−η)·C_ViT(n)', 'η = 命中率（按图像数加权）', 9.5, 6.3, true);
  body.push(cap(282, 268, '每图期望 VE 算力（TFLOPs）', { size: 7 }));
  [['η=0', 167, 'ink', 276, 0.3, '5.57'], ['η=0.5', 83, 'soft', 296, 0.5, '2.78'], ['η=0.8', 33, 'seal', 316, 0.7, '1.11 · 11.1 ms（省 80%）']].forEach(([lab, w, tone, y, di, val]) => {
    body.push(cap(272, y + 10, lab, { size: 7, anchor: 'end', mono: true }));
    body.push(`<g class="sig-bar" style="--i:${di}"><rect class="dia-bar dia-bar-${tone}" x="282" y="${y}" width="${w}" height="12" rx="3"/></g>`);
    body.push(cap(288 + w, y + 10, val, { size: 7, mono: true, color: tone === 'seal' ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  });
  body.push(cap(282, 344, '无 cache 55.7 ms/图（假设 100 TFLOPS）', { size: 7 }));
  mk('cp', 500, 276, 168, 48, 'proj', '4.73 MiB/图', '484×5120×2 B（bf16）', 10, 6.3, true);
  body.push(cap(500, 338, '1 GiB ≈ 216 张图驻留', { size: 7.5 }));
  body.push(cap(500, 352, 'vLLM 行数口径与此换算等价', { size: 7 }));

  body.push(cap(36, 386, '命中不是免费：embedding 须驻留显存或被取回；权重更新 → reset 全量失效（VE 输出依赖 VE 权重 —— 9.3 权重更新接口由此连到多模态 cache）', { size: 9 }));
  body.push(cap(36, 404, 'encoder cache 还是 MLLM chunked prefill 的前提（V1 blog 第 7 点）：V0 图像与文本必须同步，V1 缓存 embedding 后文本才能分 chunk 跨步', { size: 9 }));
  body.push(cap(36, 422, '三引擎落点：vLLM EncoderCacheManager（分配时驱逐 · 同请求单引用）· SGLang embedding 算一次 chunk 只切片（双键去重单次 ViT call）· DeepInfer 零占位行不进 ViT', { size: 7.5 }));
  body.push(cap(36, 440, '调度耦合：encoder 预算与 token 预算是两个旋钮 —— vLLM _try_schedule_encoder_inputs 预算内安排；SGLang chunk 切片（4.3 钳位只管 token 预算）', { size: 9 }));
  return scene(720, 454, body, { label: 'VE LB 与 encoder cache 两笔账：等总 patch 数下 GPU A 单张大图 attention 二点二乘十的十二次方是 GPU B 四张小图的四倍，总 FLOPs 比一点四二，fewest patches 是单调代理非等价；encoder cache 命中率零点八时期望算力一点一一 TFLOPs 约十一点一毫秒省百分之八十，每图 embedding 四点七三 MiB' }).svg();
}

/* ============================================================
 * 9.4 §4（block 1）— SO 同步 vs 异步编译时序
 * 原文：同步 TTFT 含全部编译 200 ms；异步专用 SO 机与前端路径 t_front=50 ms 并行，
 *       冷 miss t_wait = max(0, 200−50) = 150 ms；命中 ~1 ms 查找；摊销 200/1000 = 0.2 ms；
 *       命题 2 两段成本分离：编译只进 TTFT，每步 advance+mask，重叠 → 步时零增量
 * ============================================================ */
function soAsyncCompile() {
  const UID = 'i9o';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'SO 专用机异步编译：冷 miss 只等 max(0, t_compile − t_front) = 150 ms —— 编译一次、每步 advance', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 88, label: '同步形态：编译在引擎关键路径上' }));
  const r1 = mk('r1', 52, 72, 80, 36, 'data', '请求', null, 9);
  const c1 = mk('c1', 162, 72, 180, 36, 'attn', 'compile t_compile = 200 ms', null, 8);
  const f1 = mk('f1', 372, 72, 100, 36, 'cache', 'prefill', null, 9);
  const d1 = mk('d1', 502, 72, 166, 36, 'op', 'decode（每步 advance+mask）', null, 7.5);
  [[r1, c1, 0.2], [c1, f1, 0.6], [f1, d1, 1.0]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 0.9, delay: dl, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(162, 124, 'TTFT 含全部编译时间 · 编译占引擎 host CPU（与 overlap scheduler 抢核）', { size: 7.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 144, w: 648, h: 150, label: '异步形态（专用 SO 机 · doc L344-348）：编译与前端路径并行' }));
  const r2 = mk('r2', 52, 176, 80, 36, 'data', '请求', null, 9);
  const f2 = mk('f2', 162, 176, 180, 36, 'proj', '前端路径 t_front = 50 ms', null, 8);
  const g2 = mk('g2', 372, 176, 100, 36, 'norm', '取编译结果', null, 8.5);
  const p2 = mk('p2', 502, 176, 70, 36, 'cache', 'prefill', null, 9);
  const d2 = mk('d2', 602, 176, 66, 36, 'op', 'decode', null, 9);
  [[r2, f2, 0.3], [f2, g2, 1.1], [g2, p2, 1.5], [p2, d2, 1.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 0.9, delay: dl, uid: UID, eid: `a${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  const so = mk('so', 162, 230, 180, 36, 'attn', 'SO 机：compile 200 ms', null, 8);
  body.push(cap(352, 244, '异步 · 结果可缓存', { size: 7.5 }));
  body.push(edge(r2, so, { nodes, portFrom: r2.bottom, portTo: so.left, pulse: true, dur: 1.4, delay: 0.5, dash: true, uid: UID, eid: 'p0' }));
  body.push(arrowHead(so.left[0], so.left[1], 1, 0));
  body.push(edge(so, g2, { nodes, portFrom: so.right, portTo: g2.bottom, pulse: true, dur: 1.4, delay: 1.0, dash: true, uid: UID, eid: 'p1' }));
  body.push(arrowHead(g2.bottom[0], g2.bottom[1], 0, -1));
  body.push(cap(52, 282, '冷 miss：t_wait = max(0, 200−50) = 150 ms · 命中：只付查找（~1 ms）', { size: 8, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(400, 282, '摊销：200 ms / 1000 req = 0.2 ms/req', { size: 7.5, mono: true }));

  body.push(zone({ x: 36, y: 310, w: 648, h: 76, label: '命题 2：两段成本分离' }));
  mk('s1', 52, 342, 300, 36, 'proj', '编译段：只进 TTFT（冷 miss 且只计超出前端部分）', null, 7.5);
  mk('s2', 382, 342, 286, 36, 'norm', '运行段：每步 advance+mask · 重叠 → 步时零增量', null, 7.5);

  body.push(cap(36, 406, 'τ_mask ≤ τ_forward 且填充与前向重叠 → SO 对 decode 步时零增量（5.4 §3 bitmask 异步填充）；串行无重叠则全暴露', { size: 9 }));
  body.push(cap(36, 424, '专用机 rationale：编译是 CPU 密集突发，引擎 host CPU 已被 overlap scheduler 占满 —— 编译抢核直接制造调度延迟与 GPU bubble', { size: 9 }));
  body.push(cap(36, 442, 'release 内绑核隔离（SO_BUILDER_CPUS=4）是同一思路单机形态；grammar cache 须设字节上限（默认 10 GiB · CVE-2025-32381）+ kill switch 保留', { size: 9 }));
  return scene(720, 456, body, { label: 'SO 编译两形态时序：同步形态编译二百毫秒在引擎关键路径 TTFT 全含；异步形态专用 SO 机编译与前端路径五十毫秒并行，冷 miss 只等超出部分一百五十毫秒，命中只付查找，千请求摊销每请求零点二毫秒；命题二编译只进 TTFT 每步只 advance 加 mask 重叠时步时零增量' }).svg();
}

/* ============================================================
 * 9.4 §5（afterHeading）— embedding / reranker prefill-only 批形态
 * 原文：单次全序列前向 + pooling（CLS/Last/Mean），KV pooling 后即释；
 *       命题 3 CLS/MEAN 拒 partial prefill（RuntimeError），LAST 例外；
 *       MaxSim s = Σ_i max_j ⟨u_i,v_j⟩，2×3 例 s=2；doc 侧可预计算缓存；fp32 累加器
 * ============================================================ */
function embeddingServing() {
  const UID = 'i9p';
  const body = [];
  const nodes = [];
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 6.5, mono = false) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss, mono });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  body.push(cap(36, 24, 'embedding / reranker：prefill-only 无 decode 循环 —— CLS/MEAN 拒 partial prefill，LAST 例外', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 110, label: '批形态：单次全序列前向 + pooling，无自回归循环' }));
  const eq = mk('eq', 52, 72, 110, 48, 'data', 'embedding 请求', 'EmbeddingReqInput', 8.5, 6, true);
  const fw = mk('fw', 192, 72, 150, 48, 'attn', '单次全序列前向', '无 streaming = 单次 __anext__()', 9, 6);
  const pl = mk('pl', 372, 72, 150, 48, 'proj', 'pooling → e', 'CLS 首 / Last 末 / Mean 均值', 10, 6);
  const kv = mk('kv', 552, 72, 116, 48, 'norm', 'KV pooling 后即释', '无跨步 KV 增长', 8.5, 6);
  [[eq, fw, 0.2], [fw, pl, 0.6], [pl, kv, 1.0]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: dl, uid: UID, eid: `e${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(52, 136, '调度目标：吞吐 / 端到端延迟（无 TBT）· 非流式 TTFT 退化为 E2E · 无 decode 可保护 → 整段准入不制造 TBT 尖刺', { size: 8 }));

  body.push(zone({ x: 36, y: 166, w: 324, h: 140, label: '命题 3：hidden state 局部性' }));
  body.push(cap(52, 196, 'chunked forward 只产当前 chunk 的 hidden state', { size: 8 }));
  body.push(cap(52, 212, 'KV cache 存 K/V 投影，不是 hidden state（models 2.4）', { size: 8 }));
  mk('cm', 52, 224, 130, 40, 'attn', 'CLS / MEAN', '需非当前 chunk 行 → RuntimeError', 9.5, 5.8);
  mk('la', 212, 224, 130, 40, 'norm', 'LAST', '只需末 token 行 → 可 chunk', 9.5, 5.8);
  body.push(cap(52, 282, 'CLS/MEAN 必须整段 prefill（chunked 对其关闭）', { size: 7.5 }));
  body.push(cap(52, 298, 'CLS/MEAN ↔ 双向 encoder · LAST ↔ causal decoder（末 token 已见全序列）', { size: 7 }));

  body.push(zone({ x: 376, y: 166, w: 308, h: 140, label: 'late interaction MaxSim（式 9）' }));
  mk('ms', 392, 196, 276, 40, 'op', 's = Σ_i max_j ⟨u_i, v_j⟩', 'query / doc 独立编码 · token 级交互聚合', 10, 6.3, true);
  body.push(cap(392, 240, '2 query × 3 doc 内积矩阵（最小实例）', { size: 7 }));
  const hot = (i, j) => (i === 0 && j === 0) || (i === 1 && j === 2);
  body.push(matrix({ x: 392, y: 246, n: 2, m: 3, cell: 24, gap: 4, on: () => true, hot, warm: (i, j) => !hot(i, j) }));
  [['1', 404, 262, true], ['0.5', 432, 262, false], ['0', 460, 262, false], ['0', 404, 290, false], ['0.5', 432, 290, false], ['1', 460, 290, true]].forEach(([v, x, y, h]) => {
    body.push(cap(x, y, v, { size: 7, anchor: 'middle', mono: true, color: h ? 'var(--seal-deep)' : 'var(--ink-soft)', weight: h ? '650' : undefined }));
  });
  body.push(cap(486, 262, '逐行取 max', { size: 7 }));
  body.push(cap(486, 280, '→ 1 + 1 = 2', { size: 8, weight: '650', color: 'var(--seal-deep)' }));

  body.push(cap(36, 326, 'doc 侧编码与 query 无关 → 离线预计算 / 缓存，在线只付 query 编码 + MaxSim 聚合 —— prefill-only 与 cache 复用在此组合', { size: 9 }));
  body.push(cap(36, 344, 'cross-encoder：query 与 doc 拼接单次前向 + score 头（serving_rerank / score / classify 端点），每对都要在线前向，仍是 prefill-only', { size: 9 }));
  body.push(cap(36, 362, '实现两笔：累加器强制 fp32（bf16 直接累加丢精度）；VLLM_BATCH_INVARIANT 模式逐序列独立 sum（index_add_ is nondeterministic on GPUs）', { size: 9 }));
  body.push(cap(36, 380, 'pooler 目录锚：Pooler ABC abstract.py L16 · seqwise/methods.py CLSPool L38 / LastPool L51 / MeanPool L61（旧路径 pooler.py 已 404）', { size: 9 }));
  return scene(720, 394, body, { label: 'embedding serving 批形态：请求单次全序列前向经 pooling 输出 embedding，KV 池化后即释无 decode 循环；命题三 CLS 与 MEAN 依赖非当前 chunk 的 hidden state 拒绝 partial prefill，LAST 只需末 token 行可 chunk；late interaction MaxSim 二乘三内积矩阵逐行取 max 求和得二，doc 侧可离线预计算缓存' }).svg();
}

export const FIGURES = [
  {
    dim: 'inference',
    slug: '09-production/01-capacity-and-cost',
    afterHeading: /^## 1\. /,
    title: 'SLO sizing 全链：λ=100 → ρ_P=0.72 → n_P=4 / n_D=45 → 0.0578 USD/M',
    svg: capacityChain,
  },
  {
    dim: 'inference',
    slug: '09-production/01-capacity-and-cost',
    afterHeading: /^## 6\. /,
    title: '存储 EC 算术：两口径恒等 35.6 GPU/server × 100 PiB 歧义 183/220',
    svg: storageEC,
  },
  {
    dim: 'inference',
    slug: '09-production/01-capacity-and-cost',
    afterHeading: /^## 7\. /,
    title: 'USD/M 两口径差 7.67×：38 USD/GPU-h 是收入（off-peak）非成本',
    svg: costCalibers,
  },
  {
    dim: 'inference',
    slug: '09-production/01-capacity-and-cost',
    afterHeading: /^## 8\. /,
    title: '命中节省 26.9% = Δn_P/(n_P0+n_D)：省 prefill 不省 decode，更是 SLO 入场券',
    svg: hitSavings,
  },
  {
    dim: 'inference',
    slug: '09-production/02-other-engines',
    replacesBlock: 0,
    title: 'TRT-LLM 当前 = PyTorch-native：PyExecutor 每 rank 循环，legacy 编译已移除',
    svg: trtllmExecutor,
  },
  {
    dim: 'inference',
    slug: '09-production/02-other-engines',
    afterHeading: /^## 3\. /,
    title: 'LMDeploy：persistent batch = continuous batching 命名 × W4A16 7B→3.5 GB',
    svg: lmdeployW4A16,
  },
  {
    dim: 'inference',
    slug: '09-production/02-other-engines',
    afterHeading: /^## 4\. /,
    title: '五引擎选型：最硬约束先咬合（硬件 / 模型族 / 特性 / 团队栈）',
    svg: engineDecision,
  },
  {
    dim: 'inference',
    slug: '09-production/03-rl-rollout-serving',
    afterHeading: /^## 2\. /,
    title: 'vLLM sleep/wake：level 1 权重 offload CPU 自动恢复，level 2 丢被改写集（KV 均丢）',
    svg: sleepWakeLevels,
  },
  {
    dim: 'inference',
    slug: '09-production/03-rl-rollout-serving',
    replacesBlock: 0,
    title: 'colocate 串行 620 s vs disagg 并行 305.6 s：GPU·时间几乎打平，判据 2r < s',
    svg: colocateDisagg,
  },
  {
    dim: 'inference',
    slug: '09-production/03-rl-rollout-serving',
    replacesBlock: 1,
    title: 'partial rollout：abort 保留 tokens+logprobs → 新版本续跑，重建 284 ms ≪ 重采 41 s',
    svg: partialRollout,
  },
  {
    dim: 'inference',
    slug: '09-production/03-rl-rollout-serving',
    afterHeading: /^## 7\. /,
    title: 'rollout 吞吐账：平均 83.9 s vs 尾部下界 164 s，投机 ×2.40 → 35 s',
    svg: rolloutThroughput,
  },
  {
    dim: 'inference',
    slug: '09-production/04-multimodal-and-special-serving',
    replacesBlock: 0,
    title: '多模态生命周期：占位展开 → VE 单 GPU → embedding 合并 → LLM（TTFT +12%）',
    svg: mmLifecycle,
  },
  {
    dim: 'inference',
    slug: '09-production/04-multimodal-and-special-serving',
    afterHeading: /^## 2\. /,
    title: 'image patch 账：4096 → 484 → 508（每维 ceil，不是整体 ceil）',
    svg: patchAccount,
  },
  {
    dim: 'inference',
    slug: '09-production/04-multimodal-and-special-serving',
    afterHeading: /^## 3\. /,
    title: 'fewest-patches 单调代理非等价（attention 4×）× encoder cache η=0.8 省 80%',
    svg: veLbEncoderCache,
  },
  {
    dim: 'inference',
    slug: '09-production/04-multimodal-and-special-serving',
    replacesBlock: 1,
    title: 'SO 专用机异步编译：冷 miss 只等 max(0, t_compile−t_front) = 150 ms',
    svg: soAsyncCompile,
  },
  {
    dim: 'inference',
    slug: '09-production/04-multimodal-and-special-serving',
    afterHeading: /^## 5\. /,
    title: 'embedding/reranker：prefill-only 无 decode 循环，CLS/MEAN 拒 partial prefill',
    svg: embeddingServing,
  },
];
