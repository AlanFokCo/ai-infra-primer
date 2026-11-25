// gpu P9（09-rdma-gpudirect）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 gpu-p07.mjs / gpu-p08.mjs。
// 脉冲渐变：用页面级 PULSE_GRAD_DEFS 的固定 id（Chapter.astro 注入），不自建 per-scene defs。
// 所有 pulse delay < 3.5s（fig-check 截图时刻语义完整）；只用 dia-pulse（reduced-motion CSS 隐藏 → 静态终态）。
// 方向/口径硬规则：GDR BW = min(NIC, PCIe)（9.2 式 1，400G = NIC 线速限非 PCIe 限）；
//   bounce 串行 = 半谐波均值（并联电阻式）；DCQCN 论文 α 译 w（α 专归消息延迟）；regime 须注主机 PCIe 代际。
// census：9.1 block0 TCP/RDMA 路径 / block1 QP 状态机（§3 ```c 教学骨架为伪代码不画）；
//         9.2 block0 bounce vs GDR 路径；
//         9.3 block0 拥塞影响链 / block1 PFC pause 传播 / block2 DCQCN RP-CP-NP / block3 IB FECN-BECN-CCT。
//         签名：9.1 §3 verbs 四对象、9.2 §4 min regime 判定（本波关键）。

import { scene, node, edge, zone, cap as kitCap, matrix, bars, timeline, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性——显式传 color 时补 inline style 提权（沿 gpu-p07 先例）。
const cap = (x, y, t, o = {}) => {
  const s = kitCap(x, y, t, o);
  return o.color ? s.replace(`fill="${o.color}"`, `style="fill:${o.color}" fill="${o.color}"`) : s;
};

// 手工路径上的脉冲（L 形/回环/折线通道用 path 直写；渐变引用页面级固定 id）
const pathPulse = (d, { dur = 2.4, delay = 0, r = 4.2 } = {}) =>
  `<circle class="dia-pulse" r="${r}" fill="url(#${PULSE_GRAD_ID})">` +
  `<animateMotion dur="${dur}s" begin="${delay}s" repeatCount="indefinite" path="${d}"/>` +
  `</circle>`;

// 显式箭头三角（kit 的 marker-end 被 CSS 关闭）
const arrowHead = (x, y, dx, dy) => {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, px = -uy, py = ux;
  const bx = x - 7 * ux, by = y - 7 * uy;
  return `<path d="M${x} ${y} L${(bx + 3.5 * px).toFixed(1)} ${(by + 3.5 * py).toFixed(1)} L${(bx - 3.5 * px).toFixed(1)} ${(by - 3.5 * py).toFixed(1)} Z" fill="rgb(24 27 32 / 0.45)"/>`;
};

/* ============================================================
 * 9.1 §2（block 0）— TCP vs RDMA 数据路径 + 命题 1 软件开销账
 * ============================================================ */
function tcpVsRdma() {
  const UID = 'g9a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'RDMA 三语义各省一笔：zero-copy 免内核拷贝 / kernel bypass 免陷入内核 / CPU offload 免协议栈 + 远端 CPU', { size: 12, color: 'var(--ink-soft)' }));

  // TCP/IP 路径
  body.push(zone({ x: 36, y: 48, w: 648, h: 118, label: 'TCP/IP 发送路径（每消息）—— 接收侧对称再来一遍' }));
  const t1 = node({ id: 't1', x: 52, y: 88, w: 120, h: 48, role: 'data', title: 'app buffer', titleSize: 11 });
  const t2 = node({ id: 't2', x: 248, y: 88, w: 160, h: 48, role: 'cache', title: 'kernel socket buffer', titleSize: 10 });
  const t3 = node({ id: 't3', x: 484, y: 88, w: 90, h: 48, role: 'op', title: 'NIC DMA', titleSize: 10.5 });
  const t4 = node({ id: 't4', x: 614, y: 88, w: 54, h: 48, role: 'plain', title: 'wire', titleSize: 10.5 });
  nodes.push(t1, t2, t3, t4); body.push(t1.svg, t2.svg, t3.svg, t4.svg);
  [[t1, t2, 0], [t2, t3, 0.4], [t3, t4, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `t${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(210, 80, 'copy1 · CPU memcpy', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(446, 80, '协议栈（CPU）', { size: 8, anchor: 'middle' }));
  body.push(cap(594, 80, 'DMA', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(52, 154, '陷入内核 = 上下文切换 ×2（发送 + 接收）· 接收：wire → NIC → kernel buffer → copy2 → app buffer', { size: 9 }));

  // RDMA 路径
  body.push(zone({ x: 36, y: 186, w: 648, h: 118, tone: 'seal', label: 'RDMA 发送路径（每消息）—— 数据路径只剩提交 + 轮询' }));
  const r1 = node({ id: 'r1', x: 52, y: 226, w: 170, h: 48, role: 'attn', title: 'app buffer（MR 已注册）', titleSize: 10.5 });
  const r2 = node({ id: 'r2', x: 484, y: 226, w: 90, h: 48, role: 'op', title: 'NIC DMA', sub: '直读', titleSize: 10.5, subSize: 8 });
  const r3 = node({ id: 'r3', x: 614, y: 226, w: 54, h: 48, role: 'plain', title: 'wire', titleSize: 10.5 });
  nodes.push(r1, r2, r3); body.push(r1.svg, r2.svg, r3.svg);
  body.push(edge(r1, r2, { nodes, strong: true, pulse: true, dur: 1.4, delay: 1.0, uid: UID, eid: 'r0', arrow: false }));
  body.push(arrowHead(r2.left[0], r2.left[1], 1, 0));
  body.push(cap(353, 218, 'NIC DMA 直读 · 无内核拷贝', { size: 9, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(edge(r2, r3, { nodes, pulse: true, dur: 1.0, delay: 1.4, uid: UID, eid: 'r1', arrow: false }));
  body.push(arrowHead(r3.left[0], r3.left[1], 1, 0));
  body.push(cap(52, 292, 'post_send = 用户态 doorbell 写（不进内核）· 接收：NIC DMA 直写 app buffer（MR），远端 CPU/OS 不参与', { size: 9 }));

  // 命题 1 开销账
  body.push(cap(36, 330, '命题 1（教学假设分量）：Δ_TCP = 2t_copy + 2t_ctx + t_stack vs Δ_RDMA = t_db —— M = 4 KB 代入', { size: 10.5 }));
  const rows = [
    ['TCP 3.82 μs', 3.82, 'soft', '2×0.41 拷贝 + 2×1 切换 + 1 协议栈'],
    ['RDMA 0.2 μs', 0.2, 'seal', '1 doorbell —— 19.1×，省 3.62 μs'],
  ];
  rows.forEach(([lab, v, tone, val], i) => {
    const y = 342 + i * 24;
    body.push(cap(150, y + 11, lab, { mono: true, size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="160" y="${y}" width="${v * 60}" height="14" rx="3"/>`);
    body.push(cap(160 + v * 60 + 8, y + 11, val, { size: 9 }));
  });
  body.push(cap(36, 404, 'regime：M = 1 GB 拷贝项 2×100 ms = 200 ms 主导（大消息省带宽）；小消息固定项（切换 + 协议栈）主导（省延迟）', { size: 10 }));
  body.push(cap(36, 422, '坑：CPU offload ≠ CPU 完全不参与 —— 控制路径（QP 迁移 / MR 注册 / ibv_poll_cq）仍由 CPU 跑，只是用户态不陷入内核', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 438, body, { label: 'TCP 路径 app 经拷贝到内核缓冲经协议栈交 NIC，两次拷贝两次切换；RDMA 路径 NIC 凭 MR 直读应用缓冲，post_send 用户态 doorbell，4KB 软件开销 3.82 对 0.2 微秒 19.1 倍' }).svg();
}

/* ============================================================
 * 9.1 §3（afterHeading）— verbs 四对象：控制面预注册 / 数据面零系统调用
 * ============================================================ */
function verbsFourObjects() {
  const UID = 'g9b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'verbs 四对象：NIC 在数据路径启动前就知道「哪块内存可 DMA、发到哪、怎么判完成」—— 预注册进对象', { size: 12, color: 'var(--ink-soft)' }));

  // 初始化期（慢路径）
  body.push(zone({ x: 36, y: 48, w: 280, h: 224, label: '初始化期 · 控制面预注册（慢路径，可花 ms）' }));
  const mr = node({ id: 'mr', x: 52, y: 84, w: 248, h: 52, role: 'cache', title: 'MR · ibv_reg_mr', sub: 'pin 页 + 地址翻译 + lkey/rkey（L2692）', mono: true, titleSize: 10.5, subSize: 7.5 });
  const qp = node({ id: 'qp', x: 52, y: 152, w: 248, h: 52, role: 'proj', title: 'QP · ibv_create_qp', sub: 'SQ + RQ · QPN 标识（L3246）', mono: true, titleSize: 10.5, subSize: 7.5 });
  const cq = node({ id: 'cq', x: 52, y: 220, w: 248, h: 44, role: 'norm', title: 'CQ · ibv_create_cq', sub: '完成事件队列（L2962）', mono: true, titleSize: 10.5, subSize: 7.5 });
  nodes.push(mr, qp, cq); body.push(mr.svg, qp.svg, cq.svg);

  // 数据期（快路径）
  body.push(zone({ x: 360, y: 48, w: 324, h: 224, tone: 'seal', label: '数据期 · 零系统调用（快路径，用户态内存写）' }));
  const wr = node({ id: 'wr', x: 376, y: 84, w: 292, h: 52, role: 'attn', title: 'WR · ibv_post_send（L3662）', sub: '写 WQE + 写 UAR doorbell · static inline 无syscall', mono: true, titleSize: 10.5, subSize: 7.5 });
  const nic = node({ id: 'nic', x: 376, y: 176, w: 140, h: 52, role: 'op', title: 'NIC DMA 引擎', sub: '取 WQE · 直读写 MR', titleSize: 10.5, subSize: 7.5 });
  const cqe = node({ id: 'cqe', x: 540, y: 176, w: 128, h: 52, role: 'norm', title: 'CQE 写回 CQ', sub: 'ibv_poll_cq 轮询 L3040', mono: true, titleSize: 10, subSize: 7 });
  nodes.push(wr, nic, cqe); body.push(wr.svg, nic.svg, cqe.svg);
  body.push(`<path class="dia-edge strong" d="M 446 136 V 176"/>`);
  body.push(pathPulse('M 446 136 V 176', { dur: 1.1, delay: 0.2 }));
  body.push(arrowHead(446, 176, 0, 1));
  body.push(cap(456, 160, 'doorbell = UAR 寄存器写（t_db 来源）', { size: 8 }));
  body.push(edge(nic, cqe, { nodes, pulse: true, dur: 1.1, delay: 0.6, uid: UID, eid: 'done', arrow: false }));
  body.push(arrowHead(cqe.left[0], cqe.left[1], 1, 0));

  // 预注册信息虚线：MR→WR（sge 引 lkey，水平通道）、QP→NIC（取 SQ WQE）、CQ→CQE（底部通道）——互不交叉
  body.push(`<path class="dia-edge dash" d="M 300 110 H 376"/>`);
  body.push(arrowHead(376, 110, 1, 0));
  body.push(cap(338, 102, 'lkey 凭证', { size: 8, anchor: 'middle' }));
  body.push(`<path class="dia-edge dash" d="M 300 178 C 336 178, 340 196, 376 196"/>`);
  body.push(arrowHead(376, 196, 1, 0));
  body.push(cap(338, 168, '取 SQ WQE', { size: 8, anchor: 'middle' }));
  const cqPath = 'M 300 242 C 330 242, 336 290, 420 290 H 604 V 228';
  body.push(`<path class="dia-edge dash" d="${cqPath}"/>`);
  body.push(arrowHead(604, 228, 0, -1));
  body.push(cap(430, 284, '完成事件写进预建 CQ（轮询取 CQE）', { size: 8 }));

  body.push(cap(36, 316, 'WR = opcode + 本地 addr/len + 远端 addr/rkey；rkey 经带外通道（TCP/CM）交换给对端 —— 单边 READ/WRITE 的远端授权（§4）', { size: 9.5 }));
  body.push(cap(36, 334, '未注册内存 NIC 不能 DMA：页可能换出、翻译未建立，强行访问触发保护错误 —— pin 不可省', { size: 10 }));
  body.push(cap(36, 352, '与 8.4 NCCL proxy「建连慢、推进快」分层同构：MR/QP/CQ 初始化建好，WR 提交与 CQ 轮询数据期跑', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 368, body, { label: 'verbs 四对象：初始化期预注册 MR pin 页发 lkey rkey、QP 收发队列、CQ 完成队列；数据期 WR 写 WQE 加 UAR doorbell，NIC 取 WQE 直读写 MR，CQE 写回 CQ 用户态轮询，零系统调用' }).svg();
}

/* ============================================================
 * 9.1 §5（block 1）— QP 状态机四态迁移 + RC 全互连连接开销账
 * ============================================================ */
function qpStateMachine() {
  const UID = 'g9c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'QP 状态机：创建后逐态迁移才能收发（man ibv_modify_qp.3 L128）—— RC 须带外交换对端元数据', { size: 12, color: 'var(--ink-soft)' }));

  const s0 = node({ id: 's0', x: 36, y: 56, w: 120, h: 44, role: 'plain', title: 'RESET', sub: '创建态', mono: true, titleSize: 12, subSize: 8.5 });
  const s1 = node({ id: 's1', x: 216, y: 56, w: 120, h: 44, role: 'proj', title: 'INIT', sub: '本地属性', mono: true, titleSize: 12, subSize: 8.5 });
  const s2 = node({ id: 's2', x: 396, y: 56, w: 120, h: 44, role: 'data', title: 'RTR', sub: 'Receive Ready', mono: true, titleSize: 12, subSize: 8 });
  const s3 = node({ id: 's3', x: 576, y: 56, w: 108, h: 44, role: 'attn', title: 'RTS', sub: 'Send Ready', mono: true, titleSize: 12, subSize: 8 });
  nodes.push(s0, s1, s2, s3); body.push(s0.svg, s1.svg, s2.svg, s3.svg);
  [[s0, s1, 0], [s1, s2, 0.4], [s2, s3, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `m${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(186, 48, 'ibv_modify_qp', { mono: true, size: 8, anchor: 'middle' }));
  // 每态属性（卡下注记列）
  body.push(cap(96, 122, '设 port/pkey/', { size: 8.5, anchor: 'middle' }));
  body.push(cap(96, 136, 'access flags', { size: 8.5, anchor: 'middle' }));
  body.push(cap(276, 122, '设远端 QPN/PSN/', { size: 8.5, anchor: 'middle' }));
  body.push(cap(276, 136, 'GID/LID/PATH_MTU', { size: 8.5, anchor: 'middle' }));
  body.push(cap(456, 122, '设本地 PSN/TIMEOUT/', { size: 8.5, anchor: 'middle' }));
  body.push(cap(456, 136, 'RETRY_CNT/RNR_RETRY', { size: 8.5, anchor: 'middle' }));
  body.push(cap(630, 122, 'MAX_QP_RD_ATOMIC', { size: 8.5, anchor: 'middle' }));
  body.push(cap(630, 136, '可发送', { size: 8.5, anchor: 'middle' }));
  body.push(cap(36, 162, 'RC：RTR/RTS 需带外交换对端 QPN/PSN/GID/LID/MTU；UD：仅改 STATE（无连接，不需对端元数据）', { size: 9.5 }));

  // 连接开销账
  body.push(cap(36, 192, '连接开销账（RC 全互连）：控制面随 n² 涨', { size: 10.5, weight: '650' }));
  const c1 = node({ id: 'c1', x: 36, y: 206, w: 180, h: 52, role: 'data', title: '每 rank n−1 个 QP', sub: '全网 n(n−1) 个', titleSize: 11, subSize: 8.5 });
  const c2 = node({ id: 'c2', x: 268, y: 206, w: 190, h: 52, role: 'proj', title: 'n=1024 → ≈1.05×10⁶', sub: '每 QP 4 态迁移 + 带外交换', mono: true, titleSize: 10.5, subSize: 8 });
  const c3 = node({ id: 'c3', x: 510, y: 206, w: 174, h: 52, role: 'attn', title: '串行建立 ≈105 s', sub: '每 QP 一次 RTT 100 μs（教学假设）', mono: true, titleSize: 10.5, subSize: 7.5 });
  nodes.push(c1, c2, c3); body.push(c1.svg, c2.svg, c3.svg);
  [[c1, c2, 1.2], [c2, c3, 1.6]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `c${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  // QP 池对照条
  body.push(cap(36, 288, 'DeepEP 预建固定 QP 池复用：num_qps_per_rank = 24（v1.2.1，10.3）承载所有 token 路由', { size: 10 }));
  body.push(cap(150, 313, '全互连 O(n²)', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="160" y="302" width="440" height="14" rx="3"/>`);
  body.push(cap(608, 313, '1.05M QP · 105 s', { mono: true, size: 9 }));
  body.push(cap(150, 337, 'QP 池 O(n)', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="160" y="326" width="${24 * 1024 / 1048576 * 440}" height="14" rx="3"/>`);
  body.push(cap(180, 337, '24 QP/rank —— 建连成本压到 O(n)', { mono: true, size: 9 }));

  body.push(cap(36, 366, '连接是控制面慢路径、传输是数据面快路径（§3 提点）—— 长连接摊销建连成本，不是每消息每对端建连', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 382, body, { label: 'QP 状态机 RESET 经 INIT 设本地属性、RTR 设远端元数据到 RTS 可发送；RC 全互连 n(n-1) 个 QP，n=1024 约 1.05M 串行建立 105 秒，DeepEP 固定 24 QP 池把 O(n²) 压到 O(n)' }).svg();
}

/* ============================================================
 * 9.2 §3（block 0）— GDR vs bounce：省两跳的带宽/延迟/host 流量账
 * ============================================================ */
function gdrVsBounce() {
  const UID = 'g9d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'GDR vs host bounce：摘掉 D2H/H2D 两跳 —— 串行 1.78× / 端到端 2.56×，流水稳态持平但 host 流量 4M→0', { size: 12, color: 'var(--ink-soft)' }));

  // bounce 路径（发送侧）
  body.push(cap(36, 60, 'bounce 路径（非 GDR）：串行两跳 + 接收侧对称再两跳', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const a1 = node({ id: 'a1', x: 36, y: 72, w: 120, h: 48, role: 'cache', title: 'GPU A HBM', sub: '读 M', titleSize: 11, subSize: 8.5 });
  const hb = node({ id: 'hb', x: 240, y: 72, w: 150, h: 48, role: 'data', title: 'host DRAM buffer', sub: '写 M + 读 M = 2M', titleSize: 10.5, subSize: 8 });
  const n1 = node({ id: 'n1', x: 474, y: 72, w: 80, h: 48, role: 'proj', title: 'NIC', titleSize: 11 });
  const f1 = node({ id: 'f1', x: 614, y: 72, w: 70, h: 48, role: 'plain', title: 'fabric', titleSize: 10 });
  nodes.push(a1, hb, n1, f1); body.push(a1.svg, hb.svg, n1.svg, f1.svg);
  [[a1, hb, 0], [hb, n1, 0.4], [n1, f1, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `b${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(198, 88, 'D2H · PCIe 跳1', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(432, 88, 'DMA 读 · 跳2', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(584, 88, '线速', { size: 8, anchor: 'middle' }));
  body.push(cap(36, 138, 'CPU/copy engine 参与 D2H 中转 · 接收对称：fabric → NIC → host DRAM → H2D → GPU B HBM，又 2M —— 两侧合计 4M', { size: 9.5 }));

  // GDR 路径
  body.push(cap(36, 166, 'GDR 路径（NIC 经 PCIe 直访 HBM）：单跳', { size: 10.5, color: 'var(--seal-deep)', weight: '650' }));
  const a2 = node({ id: 'a2', x: 36, y: 178, w: 120, h: 48, role: 'cache', title: 'GPU A HBM', sub: '读 M', titleSize: 11, subSize: 8.5 });
  const n2 = node({ id: 'n2', x: 474, y: 178, w: 80, h: 48, role: 'proj', title: 'NIC', titleSize: 11 });
  const f2 = node({ id: 'f2', x: 614, y: 178, w: 70, h: 48, role: 'plain', title: 'fabric', titleSize: 10 });
  nodes.push(a2, n2, f2); body.push(a2.svg, n2.svg, f2.svg);
  body.push(edge(a2, n2, { nodes, strong: true, pulse: true, dur: 1.5, delay: 1.0, uid: UID, eid: 'g0', arrow: false }));
  body.push(arrowHead(n2.left[0], n2.left[1], 1, 0));
  body.push(cap(315, 194, 'NIC DMA 直读（经 BAR1）· PCIe 单跳 M', { mono: true, size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(315, 218, 'host DRAM 流量 = 0 · CPU 不参与数据面', { size: 9, anchor: 'middle' }));
  body.push(edge(n2, f2, { nodes, pulse: true, dur: 1.0, delay: 1.4, uid: UID, eid: 'g1', arrow: false }));
  body.push(arrowHead(f2.left[0], f2.left[1], 1, 0));

  // 带宽账（M = 1 GB，raw 峰值口径）
  body.push(cap(36, 254, 'M = 1 GB 带宽账（D2H/H2D = 64、NIC = 50 GB/s）：', { size: 10.5 }));
  const rows = [
    ['bounce 发送串行', 28.07, 'soft', '28.07 GB/s · 半谐波 (1/64+1/50)⁻¹ · GDR 1.78×'],
    ['bounce 端到端串行', 19.51, 'warm', '19.51 GB/s · 三跳 (1/64+1/50+1/64)⁻¹ · GDR 2.56×'],
    ['GDR / 流水稳态', 50, 'seal', '50 GB/s · 流水 bounce = min(64,50) = 50 持平'],
  ];
  rows.forEach(([lab, v, tone, val], i) => {
    const y = 266 + i * 24;
    body.push(cap(170, y + 11, lab, { size: 9.5, anchor: 'end' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="180" y="${y}" width="${v * 5}" height="14" rx="3"/>`);
    body.push(cap(180 + v * 5 + 8, y + 11, val, { mono: true, size: 9 }));
  });

  body.push(cap(36, 352, '延迟账：staging 每跳 15.625 ms、两跳 31.25 ms/GB GDR 全免；小消息省拷贝固定开销（cudaMemcpy 6–7 μs → gdrcopy 0.10 μs）', { size: 10 }));
  body.push(cap(36, 370, '提点：NIC 线速限 regime 下流水 bounce 也到 50 —— GDR 增益在 host 流量 4M→0、CPU 卸载、staging 延迟三项结构性收益', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 386, body, { label: 'bounce 路径 GPU A 经 D2H 到 host DRAM 再 NIC DMA 读发 fabric，两侧 host 流量 4M；GDR 单跳 NIC 经 BAR1 直读 HBM host 零流量；串行 28.07 端到端 19.51 对 GDR 50 GB/s，流水持平' }).svg();
}

/* ============================================================
 * 9.2 §4（afterHeading）— min(NIC, PCIe) regime 判定（本波关键）
 * ============================================================ */
function gdrMinRegime() {
  const UID = 'g9e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'BW_GDR = min(BW_NIC, BW_PCIe)（式 1）：两段串行取瓶颈 —— 单口 400G 是 NIC 线速限，不是 PCIe 限', { size: 12, color: 'var(--ink-soft)' }));

  // 串行两段 → min
  const p1 = node({ id: 'p1', x: 36, y: 52, w: 190, h: 52, role: 'proj', title: 'PCIe Gen5 x16 DMA 段', sub: '64 GB/s 单向（Hopper wp p.49）', mono: true, titleSize: 10.5, subSize: 7.5 });
  const p2 = node({ id: 'p2', x: 278, y: 52, w: 190, h: 52, role: 'op', title: 'NIC wire 段（400G）', sub: '50 GB/s 单向（400÷8）', mono: true, titleSize: 10.5, subSize: 7.5 });
  const p3 = node({ id: 'p3', x: 520, y: 52, w: 164, h: 52, role: 'attn', title: 'min = 50 GB/s', sub: 'NIC 线速限 · PCIe 余量 14', mono: true, titleSize: 12, subSize: 8 });
  nodes.push(p1, p2, p3); body.push(p1.svg, p2.svg, p3.svg);
  [[p1, p2, 0], [p2, p3, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `s${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(36, 124, '同一字节流必经两段串行，稳态吞吐取慢段；NIC 内 DMA 与 wire cut-through 重叠不另计跳（store-and-forward 才半谐波）', { size: 9.5 }));

  // regime 表
  body.push(cap(36, 152, '配置', { size: 9 }));
  body.push(cap(330, 152, 'NIC', { size: 9, anchor: 'middle' }));
  body.push(cap(400, 152, 'PCIe', { size: 9, anchor: 'middle' }));
  body.push(cap(470, 152, 'min', { size: 9, anchor: 'middle' }));
  body.push(cap(520, 152, '瓶颈 regime', { size: 9 }));
  const regs = [
    ['单口 CX7 NDR400 + Gen5 x16', '50', '64', '50', 'NIC 线速限（50 低于 64）', 'seal'],
    ['双口 CX7（2×NDR200）+ Gen5', '50', '64', '50', 'NIC 线速限（双口不翻倍）', 'soft'],
    ['CX8 800G + Gen5 x16 主机', '100', '64', '64', 'PCIe 限（100 高于 64）', 'warm'],
    ['CX8 800G + Gen6 x16 主机', '100', '128', '100', 'NIC 线速限（100 低于 128）', 'seal'],
  ];
  regs.forEach(([name, nic, pcie, mn, regime, tone], i) => {
    const y = 174 + i * 24;
    body.push(cap(36, y, name, { size: 9.5 }));
    body.push(cap(330, y, nic, { mono: true, size: 9.5, anchor: 'middle' }));
    body.push(cap(400, y, pcie, { mono: true, size: 9.5, anchor: 'middle' }));
    body.push(cap(470, y, mn, { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
    body.push(`<rect class="dia-bar dia-bar-${tone}" x="516" y="${y - 11}" width="4" height="14" rx="2"/>`);
    body.push(cap(528, y, regime, { size: 9 }));
  });
  body.push(cap(36, 274, 'regime 须注主机 PCIe 代际：CX8 原生主机口 Gen6 @64 GT/s x16 = 128 GB/s/方向 —— Gen6 下又回 NIC 线速限', { size: 9.5 }));

  // vs HBM
  body.push(cap(36, 302, 'vs HBM：瓶颈不在 HBM 段', { size: 10.5, weight: '650' }));
  body.push(cap(150, 327, 'GDR 路径 50', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="160" y="316" width="${50 * 0.16}" height="14" rx="3"/>`);
  body.push(cap(176, 327, '50 GB/s', { mono: true, size: 9 }));
  body.push(cap(150, 351, 'H100 HBM 3350', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="160" y="340" width="${3350 * 0.16}" height="14" rx="3"/>`);
  body.push(cap(696, 334, '3350/50 = 67×', { mono: true, size: 9.5, anchor: 'end', color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 380, '坑：见「GDR 走 PCIe」就判 PCIe 瓶颈 —— 升 PCIe 代际不动 400G 瓶颈；只有 NIC 线速超 PCIe（800G+Gen5）升 PCIe 才有收益', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 396, body, { label: 'GDR 带宽取 PCIe DMA 段与 NIC wire 段较小者：单口 CX7 50 低于 Gen5 64 为 NIC 线速限，CX8 100 高于 64 为 PCIe 限、Gen6 128 下又 NIC 限；HBM 3350 对 50 为 67 倍瓶颈不在 HBM' }).svg();
}

/* ============================================================
 * 9.3 §2（block 0）— 拥塞影响链：三路恶化 → 集合同步取 max
 * ============================================================ */
function congestionChain() {
  const UID = 'g9f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '无损 ≠ 无拥塞：拥塞 = 到达率持续超服务率 —— 排队延迟 + pause 反压 + 降速，三路恶化通信账', { size: 12, color: 'var(--ink-soft)' }));

  const root = node({ id: 'root', x: 250, y: 48, w: 220, h: 44, role: 'attn', title: '拥塞（A > C 持续）', sub: '无损指链路层不丢包，两者正交', titleSize: 11.5, subSize: 8 });
  nodes.push(root); body.push(root.svg);

  // 三分支
  const q1 = node({ id: 'q1', x: 36, y: 132, w: 200, h: 36, role: 'cache', title: '队列堆积', titleSize: 11 });
  const q2 = node({ id: 'q2', x: 36, y: 188, w: 200, h: 36, role: 'cache', title: '排队延迟涨', titleSize: 11 });
  const p1 = node({ id: 'p1', x: 260, y: 132, w: 200, h: 36, role: 'proj', title: 'PFC pause 反压', titleSize: 11 });
  const p2 = node({ id: 'p2', x: 260, y: 188, w: 200, h: 36, role: 'proj', title: '上游停发 · 有效带宽降', titleSize: 10.5 });
  const c1 = node({ id: 'c1', x: 484, y: 132, w: 200, h: 36, role: 'op', title: 'ECN/FECN 标记', titleSize: 11 });
  const c2 = node({ id: 'c2', x: 484, y: 188, w: 200, h: 36, role: 'op', title: 'DCQCN/IBTA CC 降速', titleSize: 10.5 });
  nodes.push(q1, q2, p1, p2, c1, c2); body.push(q1.svg, q2.svg, p1.svg, p2.svg, c1.svg, c2.svg);
  [['M 360 92 C 360 112, 136 112, 136 132', 0], ['M 360 92 V 132', 0.3], ['M 360 92 C 360 112, 584 112, 584 132', 0.6]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.3, delay: dl }));
  });
  body.push(arrowHead(136, 132, 0, 1)); body.push(arrowHead(360, 132, 0, 1)); body.push(arrowHead(584, 132, 0, 1));
  [[q1, q2, 0.4], [p1, p2, 0.7], [c1, c2, 1.0]].forEach(([a, b, dl], i) => {
    body.push(`<path class="dia-edge" d="M ${a.bottom[0]} ${a.bottom[1]} V ${b.top[1]}"/>`);
    body.push(pathPulse(`M ${a.bottom[0]} ${a.bottom[1]} V ${b.top[1]}`, { dur: 1.0, delay: dl }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(cap(136, 236, 'α_eff = α₀ + t_queue 涨', { mono: true, size: 9, anchor: 'middle' }));
  body.push(cap(360, 236, 'HoL / victim flow', { size: 9, anchor: 'middle' }));
  body.push(cap(584, 236, 'β_eff = 1/BW_eff 涨', { mono: true, size: 9, anchor: 'middle' }));

  // 汇聚：带宽项 → max
  const m1 = node({ id: 'm1', x: 140, y: 256, w: 440, h: 40, role: 'data', title: '集合带宽项 2(n-1)/n·M·β_eff 涨（8.2 式 2）', titleSize: 11 });
  const m2 = node({ id: 'm2', x: 140, y: 312, w: 440, h: 40, role: 'attn', title: '同步语义取 max：一条慢 rail 拖慢整个集合', titleSize: 11 });
  nodes.push(m1, m2); body.push(m1.svg, m2.svg);
  [['M 136 248 C 136 252, 200 250, 240 256', 1.2], ['M 360 248 V 256', 1.4], ['M 584 248 C 584 252, 520 250, 480 256', 1.6]].forEach(([d, dl]) => {
    body.push(`<path class="dia-edge" d="${d}"/>`);
    body.push(pathPulse(d, { dur: 1.0, delay: dl }));
  });
  body.push(arrowHead(240, 256, 40, 6)); body.push(arrowHead(360, 256, 0, 1)); body.push(arrowHead(480, 256, -40, 6));
  body.push(`<path class="dia-edge strong" d="M 360 296 V 312"/>`);
  body.push(pathPulse('M 360 296 V 312', { dur: 0.9, delay: 1.8 }));
  body.push(arrowHead(360, 312, 0, 1));

  // straggler 账
  body.push(cap(36, 372, 'straggler 账（n=16 单 rail 跨机环、M=1 GB，每 rank 发 1.875 GB）：', { size: 10 }));
  body.push(cap(170, 395, '健康 rail 50 GB/s', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="180" y="384" width="${37.5 * 4}" height="14" rx="3"/>`);
  body.push(cap(338, 395, '37.5 ms', { mono: true, size: 9.5 }));
  body.push(cap(170, 419, '一条 rail 降速 25', { size: 9.5, anchor: 'end' }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="180" y="408" width="${75 * 4}" height="14" rx="3"/>`);
  body.push(cap(488, 419, '75 ms —— 集合完成 = max，翻倍', { mono: true, size: 9.5, color: 'var(--seal-deep)', weight: '650' }));

  body.push(cap(36, 448, '通信暴露涨 → training step / inference 传输预算恶化 · 其余 15 条 rail 再快也无济于事', { size: 10 }));
  body.push(cap(36, 466, '两层兜底：PFC/credit 兜瞬时缓冲溢出（μs 级逐跳），DCQCN/IBTA CC 兜持续过载（ms 级端到端）—— 都要，参数互相让路', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 482, body, { label: '拥塞三分支：队列堆积涨 α，PFC pause 停发降带宽，ECN 标记降速涨 β；集合带宽项恶化后同步取 max，一条 rail 减半使 16 rank all-reduce 从 37.5 翻倍到 75 毫秒' }).svg();
}

/* ============================================================
 * 9.3 §3（block 1）— PFC pause 逐跳传播 + HoL/victim + headroom 账
 * ============================================================ */
function pfcPause() {
  const UID = 'g9g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'PFC pause 逐跳向上游传播：出口排不空 → 入口队列涨 → pause 上游 —— 逐类 pause 的三个代价', { size: 12, color: 'var(--ink-soft)' }));

  const src = node({ id: 'src', x: 36, y: 72, w: 130, h: 48, role: 'data', title: 'src NIC', titleSize: 11 });
  const ing = node({ id: 'ing', x: 250, y: 72, w: 150, h: 48, role: 'op', title: 'leaf 入口端口', titleSize: 11 });
  const egr = node({ id: 'egr', x: 484, y: 72, w: 200, h: 48, role: 'attn', title: 'leaf 出口（拥塞点）', sub: 'A > C 持续 · 队列涨', titleSize: 11, subSize: 8 });
  nodes.push(src, ing, egr); body.push(src.svg, ing.svg, egr.svg);
  [[src, ing, 0], [ing, egr, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `f${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(484, 64, '2:1 incast：A=100、C=50 GB/s → 队列以 50 GB/s 涨', { size: 8.5 }));

  // pause 回传链（出口 → 入口 → src，逐跳）
  const pausePath = 'M 584 120 V 156 H 101 V 120';
  body.push(`<path class="dia-edge strong" d="${pausePath}"/>`);
  body.push(pathPulse(pausePath, { dur: 1.8, delay: 0.8 }));
  body.push(arrowHead(101, 120, 0, -1));
  body.push(`<path class="dia-edge" d="M 325 156 V 120"/>`);
  body.push(arrowHead(325, 120, 0, -1));
  body.push(cap(360, 172, 'pause 帧逐跳上游：出口排不空 → 入口 Q 涨至阈值 → pause src NIC（命题 2 递归）', { size: 9, anchor: 'middle' }));

  // victim flow
  const vic = node({ id: 'vic', x: 36, y: 200, w: 200, h: 40, role: 'plain', title: 'victim flow（不经拥塞点）', titleSize: 10 });
  nodes.push(vic); body.push(vic.svg);
  body.push(`<path class="dia-edge dash" d="M 136 200 V 156"/>`);
  body.push(arrowHead(136, 156, 0, -1));
  body.push(cap(146, 188, '共享上游端口/优先级 → 一并被停（HoL）', { size: 8.5 }));

  // 触发与 headroom 账
  body.push(zone({ x: 420, y: 196, w: 264, h: 130, label: '触发与 headroom 账（教学假设）' }));
  body.push(cap(436, 232, 'Q_th = 512 KB：空队列 → pause 仅 10.24 μs', { size: 9 }));
  body.push(cap(436, 250, '比 CNP 间隔 50 μs 还短 —— PFC 必须存在', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(436, 274, 'headroom ≥ C·t_react + MTU（式 2）', { mono: true, size: 9 }));
  body.push(cap(436, 292, '= 100 KB + 4 KB = 104 KB（t_react = 2 μs）', { mono: true, size: 9 }));
  body.push(cap(436, 312, 'pause 传播 + 反应期间仍按线速注入', { size: 8.5 }));

  // storm 条件与三代价
  body.push(cap(36, 262, 'PFC storm 持续条件（命题 2）：', { size: 10.5, weight: '650' }));
  body.push(cap(36, 282, '传播链上没有端点降速让路，或依赖成环（A 等 B、B 等 A → 死锁）', { size: 9.5 }));
  body.push(cap(36, 302, '三代价：HoL blocking（整类停）/ victim flow / congestion spreading', { size: 9.5 }));
  body.push(cap(36, 322, '结构性缺陷：把「逐流问题」当「逐类问题」—— 不知道哪条流该降速', { size: 9.5 }));

  body.push(cap(36, 356, 'DCQCN 动机：端点降速替代逐跳 pause，把 PFC 压回「罕触发」兜底位 —— 无损优先级 + ECN + DCQCN 三件一起配（§8）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 372, body, { label: 'PFC pause 传播：leaf 出口拥塞队列涨，pause 帧逐跳回传入口与源 NIC，victim flow 共享优先级被牵连；512KB 阈值 10.24 微秒触发，headroom 至少 104KB，storm 条件为无端点让路或成环' }).svg();
}

/* ============================================================
 * 9.3 §4（block 2）— DCQCN RP/CP/NP 反馈环 + 速率走读
 * ============================================================ */
function dcqcnLoop() {
  const UID = 'g9h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DCQCN：交换机 ECN 标记 → 接收 NIC 回 CNP → 发送 NIC 乘性降速再恢复 —— 速率调节全在 NIC 硬件', { size: 12, color: 'var(--ink-soft)' }));

  const rp = node({ id: 'rp', x: 36, y: 64, w: 180, h: 56, role: 'attn', title: 'RP · 发送 NIC', sub: '按 RC 速率注入', titleSize: 11.5, subSize: 8.5 });
  const cp = node({ id: 'cp', x: 270, y: 64, w: 180, h: 56, role: 'op', title: 'CP · 交换机', sub: 'egress RED 式标记', titleSize: 11.5, subSize: 8.5 });
  const np = node({ id: 'np', x: 504, y: 64, w: 180, h: 56, role: 'proj', title: 'NP · 接收 NIC', sub: '收 ECN 标记包', titleSize: 11.5, subSize: 8.5 });
  nodes.push(rp, cp, np); body.push(rp.svg, cp.svg, np.svg);
  [[rp, cp, 0], [cp, np, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `d${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(243, 56, '注入', { size: 8, anchor: 'middle' }));
  body.push(cap(477, 56, 'ECN 标记', { size: 8, anchor: 'middle' }));
  // 两行注记避开 CNP 回传竖线（x126 / x594）：从 x150 起、止于 x580 前
  body.push(cap(150, 140, 'CP 标记：低于 Kmin 不标、Kmin~Kmax 概率线性升至 Pmax、', { size: 9 }));
  body.push(cap(150, 156, '超 Kmax 全标 —— 连续「拥塞程度」信号而非 0/1', { size: 9 }));

  // CNP 回传
  const cnpPath = 'M 594 120 V 176 H 126 V 120';
  body.push(`<path class="dia-edge strong" d="${cnpPath}"/>`);
  body.push(pathPulse(cnpPath, { dur: 1.8, delay: 0.8 }));
  body.push(arrowHead(126, 120, 0, -1));
  body.push(cap(360, 192, 'CNP 高优先级回传：每流每 T_CNP = 50 μs 至多一个（反馈不被拥塞本身吞掉）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // RP 四动作
  body.push(zone({ x: 36, y: 210, w: 324, h: 124, label: 'RP 四动作（论文式 (1)-(4)，α 译 w）' }));
  body.push(cap(52, 244, '收 CNP：RT=RC，RC=RC(1−w/2)，w→1（式 3）', { mono: true, size: 8.5 }));
  body.push(cap(52, 262, 'T_w=55 μs 无 CNP：w=(1−g)w 衰减（式 4）', { mono: true, size: 8.5 }));
  body.push(cap(52, 280, 'fast recovery F=5：RC=(RT+RC)/2（式 5）', { mono: true, size: 8.5 }));
  body.push(cap(52, 298, 'additive increase：RT+=RAI，RC 取中点（式 6）', { mono: true, size: 8.5 }));
  body.push(cap(52, 320, '无 slow start：流启动即线速（突发立即打满队列）', { size: 8.5, color: 'var(--seal-deep)' }));

  // 速率走读
  body.push(zone({ x: 396, y: 210, w: 288, h: 124, label: '全数字走读（RC 初值 50，w 初值 1）' }));
  body.push(cap(412, 244, 'CNP₁：RT=50，RC=25 —— 首砍恰好减半', { mono: true, size: 8.5 }));
  body.push(cap(412, 262, 'CNP₂（w 未衰减）：RT=25，RC=12.5', { mono: true, size: 8.5 }));
  body.push(cap(412, 280, 'fast recovery 向 RT=25 折半逼近：', { size: 8.5 }));
  body.push(cap(412, 298, '18.75→21.875→23.4375→24.21875→24.609', { mono: true, size: 8.5 }));
  body.push(cap(412, 320, '命题 3 闭式：RC_k = RT − (RT−RC₀)/2^k', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 356, '命题 3 代入（RC₀=25、RT=50）：k=5 → 49.21875 GB/s，差距 1.56%；压到 1% 内需 k≥6 —— F=5 已收回九成以上差距', { size: 9.5 }));
  body.push(cap(36, 374, '与 PFC 分工：CNP 回路 50 μs vs PFC 触发 10 μs —— incast 瞬间 PFC 先兜住，DCQCN 随后压端点、队列排空、pause 解除', { size: 10 }));
  body.push(cap(36, 392, '调参目标：ECN 阈值设得比 PFC 阈值低，标记先于 pause，PFC 罕触发（论文设计目标）', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 408, body, { label: 'DCQCN 三角色环：发送 NIC RP 注入，交换机 CP RED 式 ECN 标记，接收 NIC NP 每流每 50 微秒回 CNP，RP 乘性降速首砍减半 50 到 25 再到 12.5，fast recovery 折半逼近，闭式几何收敛' }).svg();
}

/* ============================================================
 * 9.3 §5（block 3）— IB IBTA CC：FECN → BECN → CCT 包间延迟
 * ============================================================ */
function ibtaCC() {
  const UID = 'g9i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'IB IBTA CC：FECN 向前标记 → BECN 回传 → CCT 表包间延迟 —— 时域旋钮（RoCE DCQCN 是频域速率）', { size: 12, color: 'var(--ink-soft)' }));

  const sh = node({ id: 'sh', x: 36, y: 64, w: 170, h: 56, role: 'attn', title: '源 HCA', sub: '③ 收 BECN → 查 CCT throttle', titleSize: 11.5, subSize: 7.5 });
  const sw = node({ id: 'sw', x: 270, y: 64, w: 180, h: 56, role: 'op', title: '① 交换机', sub: '检测拥塞 → 置 FECN bit', titleSize: 11.5, subSize: 8 });
  const dh = node({ id: 'dh', x: 514, y: 64, w: 170, h: 56, role: 'proj', title: '② 目的 HCA', sub: '收 FECN → 回 BECN', titleSize: 11.5, subSize: 8 });
  nodes.push(sh, sw, dh); body.push(sh.svg, sw.svg, dh.svg);
  [[sh, sw, 0], [sw, dh, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `f${i}`, arrow: false }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(477, 56, 'FECN 随包向前', { size: 8, anchor: 'middle' }));

  const becnPath = 'M 599 120 V 168 H 121 V 120';
  body.push(`<path class="dia-edge strong" d="${becnPath}"/>`);
  body.push(pathPulse(becnPath, { dur: 1.8, delay: 0.8 }));
  body.push(arrowHead(121, 120, 0, -1));
  body.push(cap(360, 184, 'BECN 回传：connected QP piggyback 在 ACK 上 / unconnected 专用 CN 包', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  // CCT 机制
  body.push(zone({ x: 36, y: 204, w: 324, h: 118, label: 'CCT / CCTI / CCM' }));
  body.push(cap(52, 238, '每 HCA 维护 CCT 表：每项 = 一个 IPD（包间延迟）', { size: 9 }));
  body.push(cap(52, 256, '每收一个 BECN：CCTI += CCTI_Increase', { mono: true, size: 9 }));
  body.push(cap(52, 274, '索引越深 → 延迟越大 → 注入越慢（逐包硬件执行）', { size: 9 }));
  body.push(cap(52, 292, '恢复：timer 到期无新 BECN → CCTI 递减、注入回升', { size: 9 }));
  body.push(cap(52, 310, 'CCM 设全部参数 · per-port per-VL 阈值 0-15', { size: 8.5 }));

  // IPD 注入率账
  body.push(zone({ x: 396, y: 204, w: 288, h: 118, label: '注入率账（式 8 · 4 KB 包 50 GB/s）' }));
  body.push(cap(412, 238, 't_tx = 4096/(50×10⁹) = 81.92 ns', { mono: true, size: 9 }));
  body.push(cap(412, 256, 'BW_inject = L_pkt/(t_tx + IPD)', { mono: true, size: 9 }));
  // 占空比示意：pkt + IPD 各 81.92ns → 减半
  body.push(`<rect class="dia-bar dia-bar-warm" x="412" y="268" width="60" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-soft" x="475" y="268" width="60" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="538" y="268" width="60" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-soft" x="601" y="268" width="60" height="14" rx="3"/>`);
  body.push(cap(442, 296, 'pkt', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(505, 296, 'IPD', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(568, 296, 'pkt', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(631, 296, 'IPD', { mono: true, size: 7.5, anchor: 'middle' }));
  body.push(cap(412, 314, 'IPD = t_tx → 注入恰好减半 25 GB/s；IPD=0 → 50', { size: 8.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 344, '机制同构：标记 → 反馈 → 降速 → 恢复；IB 用包间延迟（时域），RoCE 用 RC 速率（频域）—— 对 GPU 通信含义相同', { size: 10 }));
  body.push(cap(36, 362, '无损底座：IB credit-based 原生（per-VL）vs RoCE 须 PFC 配置 —— 拥塞时 β_eff 涨，账走 §7 同一条链', { size: 10, color: 'var(--seal-deep)' }));

  return scene(720, 378, body, { label: 'IBTA CC 三阶段：交换机置 FECN 随包向前，目的 HCA 回 BECN piggyback ACK，源 HCA 查 CCT 表插入包间延迟；4KB 包发送时间 81.92 纳秒，IPD 等于发送时间注入率减半到 25 GB/s' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/01-rdma-fundamentals',
    replacesBlock: 0,
    title: 'RDMA 三语义：TCP 2 拷贝+2 切换+协议栈 vs 1 doorbell（19.1×）',
    svg: tcpVsRdma,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/01-rdma-fundamentals',
    afterHeading: /^## 3\. /,
    title: 'verbs 四对象：控制面预注册（MR/QP/CQ）、数据面零系统调用（WR+doorbell）',
    svg: verbsFourObjects,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/01-rdma-fundamentals',
    replacesBlock: 1,
    title: 'QP 四态迁移 + 连接开销账：n(n-1) ≈ 1.05M QP，固定池压到 O(n)',
    svg: qpStateMachine,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/02-gpudirect-rdma',
    replacesBlock: 0,
    title: 'GDR vs bounce：单跳 50 对串行 28.07 / 端到端 19.51 GB/s，host 4M→0',
    svg: gdrVsBounce,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/02-gpudirect-rdma',
    afterHeading: /^## 4\. /,
    title: 'GDR 带宽 = min(NIC, PCIe)：400G NIC 线速限，800G+Gen5 才 PCIe 限',
    svg: gdrMinRegime,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/03-congestion-reliability',
    replacesBlock: 0,
    title: '拥塞影响链：α/β 双恶化 → 同步取 max，一条慢 rail 集合翻倍',
    svg: congestionChain,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/03-congestion-reliability',
    replacesBlock: 1,
    title: 'PFC pause 逐跳传播：HoL/victim/storm 三代价 + headroom 104 KB',
    svg: pfcPause,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/03-congestion-reliability',
    replacesBlock: 2,
    title: 'DCQCN RP/CP/NP 反馈环：首砍减半 50→25→12.5，fast recovery 折半',
    svg: dcqcnLoop,
  },
  {
    dim: 'gpu',
    slug: '09-rdma-gpudirect/03-congestion-reliability',
    replacesBlock: 3,
    title: 'IB IBTA CC：FECN→BECN→CCT 包间延迟，IPD=t_tx 注入减半',
    svg: ibtaCC,
  },
];
