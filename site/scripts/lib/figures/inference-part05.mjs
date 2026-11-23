// inference Part 5（05-vllm）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs / inference-part03.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言伪代码围栏不画（管线自动算法面板）。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

// CSS .dia-cap 的 fill 会覆盖 cap() 的 fill 属性（类选择器 > 表现属性）——
// 显式传 color 时补 inline style 提权，保证朱砂强调真正生效（不改 kit API，W2 finding）。
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

/* ============================================================
 * 5.1 §2（block 0）— V1 进程架构（tp=2, dp=4，17 进程）
 * 原文：四类进程；API↔core 多对多 ZMQ+msgpack；core→worker 共享内存
 *       MessageQueue 广播；N_proc = A + d + dpt + 1[d>1]；tp4=6
 * ============================================================ */
function processArch() {
  const UID = 'i5a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'V1 进程架构（tp=2 · dp=4 示意）：4 API + 4 EngineCore + 8 GPU worker + 1 coordinator = 17 进程', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 84, label: 'API Server 进程 ×4（默认 = DP 度）— HTTP / tokenize / mm 加载 / detokenize / stream' }));
  const apis = [52, 208, 364, 520].map((x, i) => {
    const n = node({ id: `a${i}`, x, y: 70, w: 136, h: 40, role: 'data', title: `API server ${i}`, sub: 'AsyncLLM frontend', titleSize: 11.5, subSize: 8 });
    nodes.push(n); body.push(n.svg);
    return n;
  });

  // ZMQ 带：短注记落在连线间隙（避免横穿文字）
  body.push(cap(182, 154, 'ZMQ + msgpack', { size: 9, anchor: 'middle' }));
  body.push(cap(321, 154, '多对多拓扑', { size: 9, anchor: 'middle' }));
  body.push(cap(459, 154, '任何 API → 任何 core', { size: 9, anchor: 'middle' }));

  body.push(`<rect class="dia-zone dia-zone-ink" x="36" y="176" width="470" height="96" rx="14"/>`);
  const cores = [52, 168, 284, 400].map((x, i) => {
    const n = node({ id: `c${i}`, x, y: 206, w: 104, h: 48, role: 'op', title: `EngineCore ${i}`, sub: 'v1/engine/core.py', mono: true, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  apis.forEach((a, i) => {
    body.push(edge(a, cores[i], { nodes, portFrom: a.bottom, portTo: cores[i].top, pulse: true, dur: 1.3, delay: i * 0.2, uid: UID, eid: `z${i}` }));
    body.push(arrowHead(cores[i].top[0], cores[i].top[1], 0, 1));
  });

  const co = node({ id: 'co', x: 520, y: 176, w: 164, h: 96, role: 'norm', title: 'DP Coordinator ×1', sub: '仅 DP>1 时存在', titleSize: 12, subSize: 8.5 });
  nodes.push(co); body.push(co.svg);
  body.push(cap(602, 254, 'rank 间负载均衡', { size: 8.5, anchor: 'middle' }));
  body.push(cap(602, 266, 'MoE 同步步协调', { size: 8.5, anchor: 'middle' }));
  body.push(edge(co, cores[3], { nodes, portFrom: co.left, portTo: cores[3].right, dash: true, arrow: false, uid: UID, eid: 'co' }));

  // shm 带：短注记落在广播线间隙
  body.push(cap(185, 304, 'shm 环广播', { size: 9, anchor: 'middle' }));
  body.push(cap(323, 304, '写一份', { size: 9, anchor: 'middle' }));
  body.push(cap(461, 304, '各读指针', { size: 9, anchor: 'middle' }));
  body.push(cap(617, 304, '背压', { size: 9, anchor: 'middle' }));

  body.push(`<rect class="dia-zone dia-zone-ink" x="36" y="312" width="648" height="84" rx="14"/>`);
  const ws = [52, 132, 212, 292, 372, 452, 532, 612].map((x, i) => {
    const n = node({ id: `w${i}`, x, y: 342, w: 72, h: 36, role: 'proj', title: `W${i}`, sub: 'GPU ' + i, mono: true, titleSize: 11, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  cores.forEach((c, i) => {
    [0, 1].forEach((k) => {
      const w = ws[i * 2 + k];
      body.push(edge(c, w, { nodes, portFrom: c.bottom, portTo: w.top, pulse: true, dur: 1.4, delay: 0.9 + i * 0.18 + k * 0.09, uid: UID, eid: `b${i}${k}` }));
      body.push(arrowHead(w.top[0], w.top[1], 0, 1));
    });
  });

  body.push(cap(36, 412, 'EngineCore ×4（每 DP rank 1 个）：scheduler + KV 管理 · busy loop；GPU Worker ×8（每 GPU 1 = DP×PP×TP）：Worker → GPUModelRunner → model', { size: 9.5 }));
  body.push(cap(36, 430, '共享内存 MessageQueue 广播（rpc_broadcast_mq）：写者只写一份并推进写指针，各 worker 持自己的读指针按序消费 —— 槽位复用要求读者跟上（背压）', { size: 9.5 }));
  body.push(cap(36, 448, '进程数（式 1）：N_proc = A + d + d·p·t + 1[d>1] —— tp=4 单节点 = 1+1+4 = 6 进程；tp=2 dp=4 = 4+4+8+1 = 17', { mono: true, size: 10 }));
  body.push(cap(36, 466, 'CPU 按进程数配而非 GPU 数：frontend 的 tokenize / detokenize / 媒体加载与 EngineCore 调度都是独立 CPU 消耗（配少了 t_cpu 反涨）', { size: 9.5 }));

  return scene(720, 478, body, { label: 'V1 四类进程：四个 API server 经多对多 ZMQ 连四个 EngineCore，DP coordinator 协商，共享内存环广播到八个 GPU worker，共十七进程' }).svg();
}

/* ============================================================
 * 5.1 §6（block 1）— 一次请求完整数据流：busy loop 五段（签名）
 * 原文：step() L714 五段；②non_block 发射前向 → ③grammar CPU 重叠 →
 *       ④等 Future 采样 → ⑤更新；IO 在独立 socket 线程；回程 detokenize
 * ============================================================ */
function requestFlow() {
  const UID = 'i5b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '一次请求的完整数据流：frontend ↔ EngineCore 异步 IPC × busy loop 五段（在线路径，file:line 均 main 核实）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, mono = false, ts = 11.5, ss = 8) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const http = mk('http', 52, 44, 70, 48, 'data', 'HTTP', 'chat/completions', false, 12, 7.5);
  const al = mk('al', 140, 44, 150, 48, 'op', 'AsyncLLM.add_request', 'L411 · asyncio 流式', true, 10.5);
  const ip = mk('ip', 308, 44, 160, 48, 'op', 'InputProcessor', 'tokenize + mm → EngineCoreRequest', false, 11.5, 7.5);
  [[http, al], [al, ip]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: i * 0.2, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(684, 106, 'n 输出共享 prompt：ParentRequest fork n 子请求（块共享 COW，2.1）', { size: 8.5, anchor: 'end' }));

  // frontend → EngineCore（左缘 L 形通道，避开 zone label）
  const inPath = 'M 388 92 V 112 H 24 V 198 H 52';
  body.push(`<path class="dia-edge strong" d="${inPath}"/>`);
  body.push(pathPulse(inPath, { dur: 2, delay: 0.3 }));
  body.push(arrowHead(52, 198, 1, 0));
  body.push(cap(206, 106, 'ZMQ + msgpack（AsyncMPClient.add_request_async L1374）', { size: 9, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 128, w: 648, h: 150, label: 'EngineCore 进程 · run_busy_loop L1814 — 只做 scheduler + executor；网络 IO 在独立 socket 线程' }));
  const s1 = mk('s1', 52, 166, 100, 64, 'op', '① schedule', '定组成 + 分块', false, 12, 8.5);
  const s2 = mk('s2', 168, 166, 124, 64, 'proj', '② execute_model', 'non_block=True → Future', true, 10.5, 7.5);
  const s3 = mk('s3', 308, 166, 116, 64, 'norm', '③ grammar bitmask', 'CPU · 自动机检查', false, 10.5, 8);
  const s4 = mk('s4', 440, 166, 108, 64, 'attn', '④ sample_tokens', '等前向 → 采样', true, 10.5, 8);
  const s5 = mk('s5', 564, 166, 116, 64, 'op', '⑤ update_from_output', '追加 / 判完成 / 释放块', true, 9, 7.5);
  [[s1, s2], [s2, s3], [s3, s4], [s4, s5]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1, delay: 0.5 + i * 0.25, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(347, 254, '③ CPU 藏进 ② GPU 前向（式 1）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(102, 264, 'input_queue 消化', { size: 8, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 300, w: 648, h: 96, label: 'GPU Worker ×N — 共享内存广播 SchedulerOutput（diff）· persistent batch 只应用增量' }));
  const fw = mk('fw', 240, 330, 180, 48, 'proj', 'forward', 'TP all-reduce · hidden states 暂存', false, 12, 8);
  const gs = mk('gs', 460, 330, 180, 48, 'attn', 'sample_tokens（GPU）', 'grammar mask → token ids', false, 11.5, 8);
  body.push(edge(s2, [300, 330], { nodes, portFrom: s2.bottom, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 'x1' }));
  body.push(arrowHead(300, 330, 0, 1));
  body.push(edge([360, 330], [480, 230], { pulse: true, dur: 1.4, delay: 1.7, uid: UID, eid: 'x2' }));
  body.push(arrowHead(480, 230, 0, -1));
  body.push(edge([508, 230], [520, 330], { pulse: true, dur: 1.2, delay: 2.2, uid: UID, eid: 'x3' }));
  body.push(arrowHead(520, 330, 0, 1));
  body.push(edge([580, 330], s5, { nodes, portTo: s5.bottom, pulse: true, dur: 1.2, delay: 2.7, uid: UID, eid: 'x4' }));
  body.push(arrowHead(s5.bottom[0], s5.bottom[1], 0, -1));
  body.push(cap(52, 292, '② shm 广播 diff', { size: 8.5 }));
  body.push(cap(355, 288, 'Future 回 hidden', { size: 8.5, anchor: 'middle' }));
  body.push(cap(460, 288, '④ mask+采样', { size: 8.5, anchor: 'middle' }));
  body.push(cap(640, 288, 'token ids 回', { size: 8.5, anchor: 'middle' }));

  // 回程（右缘通道 → 自右向左的输出处理链）
  const outPath = 'M 680 198 H 700 V 444 H 670';
  body.push(`<path class="dia-edge strong" d="${outPath}"/>`);
  body.push(pathPulse(outPath, { dur: 2, delay: 3.0 }));
  body.push(arrowHead(670, 444, -1, 0));
  body.push(cap(694, 318, 'Outputs · ZMQ 回', { size: 8, anchor: 'end' }));
  const oq = mk('oq', 560, 420, 110, 48, 'cache', 'output_queue', 'socket 线程发 L2305', true, 10, 7.5);
  const op = mk('op2', 380, 420, 150, 48, 'op', 'OutputProcessor', 'L554 · RequestState', true, 10.5, 7.5);
  const dt = mk('dt', 190, 420, 160, 48, 'op', 'IncrementalDetokenizer', '增量 · UTF-8 跨 chunk', false, 10.5, 7.5);
  const yd = mk('yd', 52, 420, 110, 48, 'data', 'yield token', 'generate L740 · SSE', false, 11.5, 7.5);
  [[oq, op], [op, dt], [dt, yd]].forEach(([a, b], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.left, portTo: b.right, pulse: true, dur: 1.1, delay: 3.0 + i * 0.15, uid: UID, eid: `r${i}` }));
    body.push(arrowHead(b.right[0], b.right[1], -1, 0));
  });

  body.push(cap(36, 490, '离线 LLM / LLMEngine 只换 frontend + client（SyncMP / Inproc 直调）——busy loop 与五段 step 两条路径完全一致（对称架构）', { size: 9.5 }));
  body.push(cap(36, 506, '没活时阻塞在 input_queue.get；有活但本步没跑前向则 sleep(0.001) 让出 GIL —— KV connector 后台传输不被 busy loop 饿死', { size: 9.5 }));

  return scene(720, 516, body, { label: '请求数据流：HTTP 经 AsyncLLM 与 InputProcessor 走 ZMQ 进 EngineCore，五段 step 中 grammar CPU 与前向重叠，worker 经共享内存收 diff，输出回程增量 detokenize 流式 yield' }).svg();
}

/* ============================================================
 * 5.1 §7（block 2）— vllm/v1 模块地图（分层走读）
 * 原文：engine/ core/ executor/ worker/ attention/ sample/ spec_decode/
 *       structured_output/ kv_offload/ metrics/；interface/layout 在 v1/ 根
 * ============================================================ */
function moduleMap() {
  const UID = 'i5c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'vllm/v1 模块地图（2026-11-23 main 骨架）：P1–P4 的机制按层落进这些文件', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 8) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono: true, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };

  body.push(zone({ x: 36, y: 40, w: 648, h: 78, label: 'engine/ — frontend + 引擎核心（在线 / 离线两入口同一核心）' }));
  const e1 = mk('e1', 52, 66, 148, 40, 'data', 'async_llm · llm_engine', '两入口：在线 / 离线');
  const e2 = mk('e2', 210, 66, 148, 40, 'op', 'core · core_client', 'busy loop / IPC client');
  const e3 = mk('e3', 368, 66, 148, 40, 'op', 'input/output_processor', 'detokenizer · 分词外置');
  const e4 = mk('e4', 526, 66, 148, 40, 'norm', 'coordinator · tensor_ipc', 'parallel_sampling');
  [[e1, e2, 0], [e2, e3, 0.3]].forEach(([a, b, dl]) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `e${dl}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(edge(e3, e4, { nodes, dash: true, arrow: false, uid: UID, eid: 'e34' }));

  body.push(zone({ x: 36, y: 132, w: 648, h: 78, label: 'core/ — 调度器 + KV 管理（CPU 侧 · 5.2 全量走读）' }));
  mk('k1', 52, 158, 148, 40, 'op', 'sched/', 'sched · async · queue · output', 9.5, 7.5);
  mk('k2', 210, 158, 148, 40, 'cache', 'block_pool · manager', '块池 / KV 管理接口');
  mk('k3', 368, 158, 148, 40, 'cache', 'coordinator · single_type', 'Unitary / Hybrid / 逐类型', 9, 7.5);
  mk('k4', 526, 158, 148, 40, 'cache', 'utils · planning · builder', 'encoder_cache · hash 链', 9, 7.5);

  body.push(`<rect x="36" y="222" width="648" height="30" rx="10" fill="rgb(168 50 40 / 0.03)" stroke="rgb(168 50 40 / 0.4)" stroke-width="1.1" stroke-dasharray="5 6"/>`);
  body.push(cap(52, 242, 'v1/ 根：kv_cache_interface.py · kv_cache_layout.py — spec 数据类与 stride 布局枚举（不在 core/ 下，坑注）', { size: 9.5, color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 266, w: 648, h: 78, label: 'executor/ → worker/ — 执行链（5.3 走读）' }));
  const x1 = mk('x1', 52, 292, 180, 40, 'proj', 'executor/', 'abstract · uniproc · multiproc · ray_v2', 9.5, 7);
  const x2 = mk('x2', 248, 292, 140, 40, 'proj', 'gpu_worker.py', '显存预算 / 编译预热');
  const x3 = mk('x3', 404, 292, 270, 40, 'proj', 'worker/gpu/', 'model_runner · input_batch · block_table · sample/ · mm/', 9.5, 7.5);
  [[x1, x2, 0.6], [x2, x3, 0.9]].forEach(([a, b, dl]) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `x${dl}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 358, w: 648, h: 76, label: '算子与特性模块（5.3 / 5.4）' }));
  const dcards = [
    [52, 104, 'attention/', 'selector + 实现族'],
    [172, 96, 'sample/', '只剩 ops/'],
    [284, 104, 'spec_decode/', 'metrics · dynamic'],
    [404, 128, 'structured_output/', 'grammar backend'],
    [548, 126, 'kv_offload/', '+ simple_kv_offload/'],
  ];
  dcards.forEach(([x, w, t, s], i) => {
    mk(`d${i}`, x, 384, w, 34, 'plain', t, null, 9.5);
    body.push(cap(x + w / 2, 428, s, { size: 7.5, anchor: 'middle' }));
  });

  body.push(cap(36, 452, 'Sampler 在 worker/gpu/sample/sampler.py L42（不在 v1/sample/）· v1/watermarking/ = 生成文本水印 ≠ KV watermark · metrics/ = Prometheus 导出', { size: 9.5, color: 'var(--seal-deep)' }));

  return scene(720, 464, body, { label: 'vllm v1 模块分层地图：engine 前端与核心、core 调度与 KV 管理、根目录 spec 与布局、executor 到 worker 执行链、算子与特性模块' }).svg();
}

/* ============================================================
 * 5.1 §5（afterHeading）— diff 协议控制面字节账（命题 1）
 * 原文：B=64 · T=4096 · b=64：全量 1.02 MiB/步 vs diff 2 KiB（523×）；
 *       204 MiB/s vs 0.4 MiB/s @5ms；抢占恢复按新请求重发全量
 * ============================================================ */
function diffProtocol() {
  const UID = 'i5d';
  const body = [];

  body.push(cap(36, 24, 'diff 协议控制面账（B=64 decode · T=4096 · b=64）：worker 缓存请求状态，每步只传增量', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 320, h: 136, label: 'NewRequestData — 全量（只发一次）' }));
  body.push(cap(52, 80, 'prompt_token_ids：T_p × 4 B = Θ(T_p)', { mono: true, size: 9.5 }));
  body.push(cap(52, 98, 'block_ids：⌈T_p/b⌉ 个块号', { mono: true, size: 9.5 }));
  body.push(cap(52, 116, 'mm_features · sampling_params · lora_request', { mono: true, size: 9 }));
  body.push(cap(52, 140, 'S_full(T_p) = Θ(T_p) · 抢占恢复也按新请求重发', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 160, '首次调度的请求（scheduled_new_reqs 列表）', { size: 8.5 }));

  body.push(zone({ x: 372, y: 40, w: 312, h: 136, label: 'CachedRequestData — diff（每步）' }));
  body.push(cap(388, 80, 'req_ids（定长标识）· new_block_ids', { mono: true, size: 9.5 }));
  body.push(cap(388, 98, 'num_computed_tokens · num_output_tokens', { mono: true, size: 9 }));
  body.push(cap(388, 116, 'new_token_ids 仅 PP 用 · 非 PP 为空', { mono: true, size: 9 }));
  body.push(cap(388, 140, 's_diff = O(1)：没有字段随上下文 T 增长', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(388, 160, 'decode 每步新块至多 ⌈1/b⌉ = 1 个/组', { size: 8.5 }));

  body.push(cap(36, 196, '命题 1（式 2）：bytes_step ≤ n_new · S_full(T_p) + n_old · s_diff —— 两类请求互斥且穷尽本步调度集合', { mono: true, size: 10 }));

  body.push(cap(52, 240, '全量重发', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-soft" x="110" y="228" width="512" height="16" rx="3"/></g>`);
  body.push(cap(366, 222, '64 × 16.35 KiB ≈ 1.02 MiB/步 → ~204 MiB/s 序列化 + 拷贝（CPU 税）', { size: 9, anchor: 'middle' }));
  body.push(cap(52, 270, 'diff 协议', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="110" y="258" width="2" height="16" rx="1"/></g>`);
  body.push(cap(122, 270, '64 × 32 B = 2 KiB/步 → ~0.4 MiB/s —— ≈523× 小，IPC 移出关键路径', { size: 9.5, color: 'var(--seal-deep)' }));
  body.push(cap(684, 222, '1.02 MiB', { mono: true, size: 9, anchor: 'end' }));

  body.push(cap(36, 306, '边界反例：被抢占请求恢复时按新请求重发全量（scheduler.py L1606「resumed request is re-sent in full」）—— n_new 须计入恢复者', { size: 9.5 }));
  body.push(cap(36, 324, 'worker 侧 persistent batch：InputBatch 缓存 tensor + RequestState 双向映射 / free_indices 槽复用 —— 每步增量写（Numpy），不整批重建', { size: 9.5 }));
  body.push(cap(36, 342, '口径：只算控制面（调度指令）；KV 与权重常驻 worker 显存不随步传输 —— 数据面流量账归 1.2 / 8.1', { size: 9.5 }));

  return scene(720, 356, body, { label: 'diff 协议：新请求发全量、已缓存请求只发 O(1) 增量，六十四条 decode 每步控制面从 1.02 MiB 降到 2 KiB，约五百二十三倍' }).svg();
}

/* ============================================================
 * 5.2 §5（block 0）— KV 管理三层架构
 * 原文：KVCacheManager 接口 → Coordinator（Unitary/Hybrid/NoPrefixCache）
 *       → BlockPool 全组共享 + SingleTypeManager × 组数；LCM 调度粒度
 * ============================================================ */
function kvThreeLayer() {
  const UID = 'i5e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'KV 管理三层：调度器接口 → 跨组协调 → 逐类型块逻辑（全组共享一个 BlockPool）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 12, ss = 8) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const sch = mk('sch', 250, 44, 220, 44, 'op', 'Scheduler', 'schedule() 只看到 KVCacheManager', 12.5, 8.5);
  const mgr = mk('mgr', 250, 116, 220, 56, 'proj', 'KVCacheManager（L130）', 'get_computed_blocks / allocate_slots / free / cache_blocks', 11.5, 7);
  const co = mk('co', 250, 200, 220, 56, 'norm', 'KVCacheCoordinator（L84）', 'Unitary / Hybrid / NoPrefixCache · LCM 调度粒度', 11.5, 7.5);
  [[sch, mgr, 0], [mgr, co, 0.4]].forEach(([a, b, dl]) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `v${dl}` }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(cap(482, 140, '接口层：命中查找 / 分配 / 释放 / 登记', { size: 8.5 }));
  body.push(cap(482, 224, '协调层：scheduler_block_size = 各组块大小公倍数', { size: 8.5 }));
  body.push(cap(482, 238, '（chunk 与命中边界同时落在每组块边界上）', { size: 8.5 }));

  const bp = mk('bp', 52, 296, 280, 64, 'cache', 'BlockPool（L107）×1 全组共享', 'free_block_queue + cached_block_hash_to_block + null_block', 11.5, 7);
  const st = mk('st', 372, 296, 312, 64, 'attn', 'SingleTypeKVCacheManager × 组数', 'FullAttentionManager / SlidingWindowManager / MambaManager …', 11.5, 7.5);
  body.push(edge(co, bp, { nodes, portFrom: co.bottom, portTo: bp.top, pulse: true, dur: 1.4, delay: 0.9, uid: UID, eid: 'b1' }));
  body.push(arrowHead(bp.top[0], bp.top[1], 0, 1));
  body.push(edge(co, st, { nodes, portFrom: co.bottom, portTo: st.top, pulse: true, dur: 1.4, delay: 1.1, uid: UID, eid: 'b2' }));
  body.push(arrowHead(st.top[0], st.top[1], 0, 1));

  body.push(cap(36, 384, 'FreeKVCacheBlockQueue = 零分配双向链表：O(1) 中部摘除（块自带 prev/next 指针）、全程不新建 Python 对象 → 0% 命中吞吐损失 <1%', { size: 9.5 }));
  body.push(cap(36, 400, 'free_blocks：无 hash 块 prepend 队头（LIFO 立即复用）· 有 hash 块 append 队尾（FIFO / LRU 驱逐）；free 逆序释放保前缀链', { size: 9.5 }));
  body.push(cap(36, 416, 'page = bκ：GQA 256 KiB/层/块 vs MLA 64 KiB（V4.1，单 latent d_V=0）—— hybrid 单池统一 page size 靠调 block_size 或 pad 对齐（§7）', { size: 9.5 }));

  return scene(720, 428, body, { label: 'KV 三层架构：调度器经 KVCacheManager 接口到协调器，协调器分叉到全组共享的块池与逐类型管理器，自由队列零分配链表支撑常数时间驱逐' }).svg();
}

/* ============================================================
 * 5.2 §6（afterHeading）— hybrid 命中不动点（签名）
 * 原文：full 块 0-47 cached 命中 3072；SWA 46/47 已驱逐、44/45 cached
 *       → 右→左扫命中 2944；full 向下封闭截断；单调减整数列必终止
 * ============================================================ */
function hybridFixedPoint() {
  const UID = 'i5f';
  const body = [];
  const nodes = [];
  const P = 9.5, X0 = 52; // 64 块 · cell 8.5 + gap 1

  body.push(cap(36, 24, 'hybrid 命中不动点（教学 full+SWA · b=64 · w=128 · T=4096）：full 命中 3072，SWA 缩长到 2944', { size: 12, color: 'var(--ink-soft)' }));

  const strip = (y, state) => {
    for (let i = 0; i < 64; i++) {
      const x = X0 + i * P;
      const s = state(i);
      if (s === 'cached') {
        body.push(`<g class="sig-cell" style="--i:${(i * 0.02).toFixed(2)}"><rect x="${x}" y="${y}" width="8.5" height="20" rx="2" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="0.9"/></g>`);
      } else if (s === 'evicted') {
        body.push(`<rect x="${x}" y="${y}" width="8.5" height="20" rx="2" fill="none" stroke="var(--seal)" stroke-width="0.9" stroke-dasharray="2 2"/>`);
      } else {
        body.push(`<rect x="${x}" y="${y}" width="8.5" height="20" rx="2" fill="none" stroke="var(--rule)" stroke-width="0.9" stroke-dasharray="2 3"/>`);
      }
    }
  };

  body.push(cap(52, 56, 'full 组：左→右扫，第一个 miss 即停（链式 hash：前块 miss 则后块必 miss）—— 块 0–47 cached', { size: 9.5 }));
  strip(66, (i) => (i < 48 ? 'cached' : 'miss'));
  body.push(`<line x1="${X0 + 48 * P}" y1="60" x2="${X0 + 48 * P}" y2="92" stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(cap(X0 + 48 * P, 60, '3072', { mono: true, size: 8.5, anchor: 'middle' }));
  // 收敛后 full 截断到 2944（块 46/47 交回）
  body.push(`<line x1="${X0 + 46 * P}" y1="60" x2="${X0 + 46 * P}" y2="92" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="3 3"/>`);
  body.push(`<rect x="${X0 + 46 * P}" y="66" width="18" height="20" rx="2" fill="none" stroke="var(--seal-deep)" stroke-width="1.1" stroke-dasharray="2 2"/>`);
  body.push(cap(X0 + 46 * P, 104, '2944（最终）', { mono: true, size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 104, '0', { mono: true, size: 8.5 }));
  body.push(cap(660, 104, '4096', { mono: true, size: 8.5, anchor: 'end' }));
  body.push(cap(280, 118, 'full 向下封闭：截断到 2944 后块 0–45 ⊆ cached，仍覆盖该前缀', { size: 8.5, anchor: 'middle' }));

  body.push(cap(52, 140, 'SWA 组：右→左扫，找最右连续 cached 段 ≥ ⌈(w−1)/b⌉ = 2 块 —— 块 46/47 已驱逐（虚线）、44/45 cached', { size: 9.5 }));
  strip(150, (i) => (i === 44 || i === 45 ? 'cached' : i === 46 || i === 47 ? 'evicted' : 'null'));
  const scan = `M 656 180 H ${X0 + 44 * P + 4}`;
  body.push(`<path class="dia-edge strong" d="${scan}"/>`);
  body.push(pathPulse(scan, { dur: 1.8, delay: 1.0 }));
  body.push(arrowHead(X0 + 44 * P + 4, 180, -1, 0));
  body.push(cap(560, 194, '段右边界 = 命中终点：46 × 64 = 2944 < 3072 → 触发缩长', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(52, 194, '窗外块 = null_block 占位', { size: 8.5 }));

  const mk = (id, x, w, role, title, sub) => {
    const n = node({ id, x, y: 216, w, h: 48, role, title, sub, titleSize: 11, subSize: 8 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const f1 = mk('f1', 52, 150, 'op', '候选 ℓ = 3072', 'full 组先查给上界');
  const f2 = mk('f2', 232, 170, 'attn', 'SWA 重查 → 2944', '缩长：46/47 不在缓存');
  const f3 = mk('f3', 432, 150, 'norm', 'full 截断到 2944', '向下封闭 · 不必重查');
  const f4 = mk('f4', 612, 72, 'cache', '收敛 ✓', '各组同前缀', 11, 7.5);
  [[f1, f2, 0.4], [f2, f3, 1.2], [f3, f4, 2.0]].forEach(([a, b, dl]) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `p${dl}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(360, 286, '通用不动点算法两轮；is_simple_hybrid（恰 full+X 两组、full 在前）一轮 break + 收尾截断 —— 代码实际路径', { size: 9, anchor: 'middle' }));

  body.push(cap(36, 314, '命题 1：hit_length 单调减整数列 → 至多 ⌊ℓ₀/a⌋+2 轮终止（ℓ₀ = num_tokens−1 一般不对齐，首轮缩量可 < a）；退出时无组再缩 → 各组覆盖同一前缀', { size: 9.5 }));
  body.push(cap(36, 332, 'MLA 组：page = 64 × 1024 = 64 KiB/层/块（V4.1，d_V=0 单 latent）vs GQA 256 KiB —— 协调器视其为又一个组，无特殊分支（式 1）', { size: 9.5 }));
  body.push(cap(36, 350, 'SWA 驻留帽 = ⌈(w−1+r+f)/b⌉+1（r = extra retained · f = in-flight）；稳态 w=128 · b=64 → 3 块；remove_skipped_blocks 逐 chunk 释放窗外块', { size: 9.5 }));

  return scene(720, 364, body, { label: 'hybrid 不动点：full 组左到右命中三千零七十二，SWA 组右到左只接受二千九百四十四，full 向下封闭截断后各组覆盖同一前缀收敛' }).svg();
}

/* ============================================================
 * 5.3 §1（block 0）— 执行链：executor → worker → runner（tp=4，6 进程）
 * 原文：MultiprocExecutor L150 / MessageQueue shm 环 L498（写一份各读指针）
 *       Worker L233 / GPUModelRunner L209；execute 与 sample 两方法分离
 * ============================================================ */
function execChain() {
  const UID = 'i5g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'vLLM 执行链（tp=4 · dp=1 · pp=1，6 进程）：executor 广播 → worker 环境 → runner 单步', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 11.5, ss = 8) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const api = mk('api', 52, 44, 140, 44, 'data', 'API server 进程', 'frontend（5.1）');
  const core = mk('core', 250, 44, 200, 44, 'op', 'EngineCore 进程', 'Scheduler.schedule()（5.2）');
  body.push(edge(api, core, { nodes, pulse: true, dur: 1.2, delay: 0, uid: UID, eid: 'ac' }));
  body.push(arrowHead(core.left[0], core.left[1], 1, 0));
  body.push(cap(221, 38, 'ZMQ + msgpack', { size: 8.5, anchor: 'middle' }));

  const ex = mk('ex', 250, 128, 200, 48, 'proj', 'MultiprocExecutor L150', 'collective_rpc · 只让 output_rank 回结果', 11, 7.5);
  body.push(edge(core, ex, { nodes, portFrom: core.bottom, portTo: ex.top, pulse: true, dur: 1.2, delay: 0.3, uid: UID, eid: 'ce' }));
  body.push(arrowHead(ex.top[0], ex.top[1], 0, 1));
  body.push(cap(362, 112, 'SchedulerOutput（每步 diff，5.1 命题 1）', { size: 9 }));

  // shm 环总线：写一份 → 各 worker 读指针
  body.push(`<path class="dia-edge strong" d="M 350 176 V 200"/>`);
  body.push(`<path class="dia-edge strong" d="M 122 200 H 590"/>`);
  body.push(pathPulse('M 350 176 V 200 H 590', { dur: 1.8, delay: 0.6 }));
  body.push(pathPulse('M 350 200 H 122', { dur: 1.8, delay: 0.6 }));
  body.push(cap(340, 192, 'MessageQueue（shm 环 L498）：写者只写一份', { size: 8.5, anchor: 'end' }));
  body.push(cap(362, 192, '各 worker 持自己的读指针按序消费 · 背压', { size: 8.5 }));

  body.push(zone({ x: 36, y: 224, w: 648, h: 96, label: '' }));
  body.push(cap(200, 232, '单卡环境', { size: 8.5, anchor: 'middle' }));
  body.push(cap(356, 232, '显存预算', { size: 8.5, anchor: 'middle' }));
  body.push(cap(512, 232, '编译预热', { size: 8.5, anchor: 'middle' }));
  body.push(cap(637, 232, 'Worker L233', { size: 8.5, anchor: 'middle' }));
  const ws = [52, 208, 364, 520].map((x, i) => mk(`w${i}`, x, 254, 140, 48, 'proj', `Worker ${i}`, 'GPUModelRunner L209', 11.5, 8));
  ws.forEach((w, i) => {
    const bx = 122 + i * 156;
    body.push(edge([bx, 200], w, { nodes, portTo: w.top, pulse: true, dur: 1.2, delay: 0.9 + i * 0.15, uid: UID, eid: `bw${i}` }));
    body.push(arrowHead(w.top[0], w.top[1], 0, 1));
  });

  const em = mk('em', 52, 364, 340, 48, 'attn', 'execute_model L1865', 'forward（TP 组内 all-reduce，§3）· hidden states 暂存', 11.5, 8);
  const st = mk('st', 430, 364, 254, 48, 'norm', 'sample_tokens L2279', '采样链（5.4 §7）· 动态分支留图外', 11.5, 8);
  ws.forEach((w, i) => {
    body.push(edge(w, [120 + i * 60, 364], { nodes, portFrom: w.bottom, pulse: true, dur: 1.3, delay: 1.5 + i * 0.15, uid: UID, eid: `we${i}` }));
    body.push(arrowHead(120 + i * 60, 364, 0, 1));
  });
  body.push(edge(em, st, { nodes, pulse: true, dur: 1.2, delay: 2.2, uid: UID, eid: 'es' }));
  body.push(arrowHead(st.left[0], st.left[1], 1, 0));
  body.push(cap(400, 352, 'GPUModelRunner L209 — 两方法分离 = async scheduling 接口前提', { size: 9 }));

  const ret = 'M 684 388 H 704 V 66 H 450';
  body.push(`<path class="dia-edge strong" d="${ret}"/>`);
  body.push(pathPulse(ret, { dur: 2.2, delay: 2.6 }));
  body.push(arrowHead(450, 66, -1, 0));
  body.push(cap(694, 332, '输出回 EngineCore → update_from_output（5.2）', { size: 8.5, anchor: 'end' }));

  body.push(cap(36, 432, '两条 IPC 通道不混：frontend ↔ core 走 ZMQ + msgpack（请求级）；executor ↔ worker 走 shm 环（每步高频小消息，省协议栈与一次拷贝）', { size: 9.5 }));
  body.push(cap(36, 448, 'RayExecutorV2 L239 继承 MultiprocExecutor（跨节点）；旧 ray_distributed_executor.py 已不存在', { size: 9.5 }));
  body.push(cap(36, 464, 'worker：init L1914 · load L637 · avail L724 · warmup L1068；executor 四名 abstract / uniproc / multiproc / ray_v2', { size: 9 }));

  return scene(720, 476, body, { label: '执行链三层：EngineCore 调度输出经 MultiprocExecutor 共享内存环广播四个 worker，GPUModelRunner 的 execute 与 sample 两方法分离，输出回 EngineCore 更新' }).svg();
}

/* ============================================================
 * 5.3 §5（block 1）— attention 调用链：能力筛选 → builder / impl
 * 原文：get_attn_backend L111 缓存；六项约束 + validate_configuration；
 *       block size 默认 64 / sparse_swa 256 偏好 / cpu_mla 16 硬约束
 * ============================================================ */
function attnCallChain() {
  const UID = 'i5h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'attention 调用链（引擎调用级）：选择 = 能力约束下的可行性筛选，不是 benchmark 排序（算法级归 3.1）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 11.5, ss = 8) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const n0 = mk('n0', 216, 48, 330, 44, 'data', 'config + KV spec kind（5.2）', 'get_attn_spec_kind L70', 11.5, 8);
  const n1 = mk('n1', 216, 106, 330, 44, 'op', 'selector.get_attn_backend L111', '_cached_get_attn_backend L212 · 每模型一次', 11.5, 8);
  body.push(edge(n0, n1, { nodes, portFrom: n0.bottom, portTo: n1.top, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'c0' }));
  body.push(arrowHead(n1.top[0], n1.top[1], 0, 1));

  body.push(cap(360, 160, '六项能力约束先过滤（AttentionBackend L57）→ validate_configuration L296 组合合法性兜底', { size: 9, anchor: 'middle' }));
  const filters = ['head_size', 'kv_dtype', 'block_size', 'CG 模式', 'batch_invariant', 'hisparse'];
  filters.forEach((t, i) => {
    const n = node({ id: `fl${i}`, x: 52 + i * 104, y: 170, w: 96, h: 28, role: 'plain', title: t, mono: true, titleSize: 8.5 });
    nodes.push(n); body.push(n.svg);
  });

  const n2 = mk('n2', 216, 216, 330, 44, 'proj', 'AttentionBackend（可行集按平台与偏好定一个）', null, 11);
  body.push(edge([360, 198], n2, { nodes, portTo: n2.top, pulse: true, dur: 1.1, delay: 0.5, uid: UID, eid: 'c1' }));
  body.push(arrowHead(n2.top[0], n2.top[1], 0, 1));

  const bd = mk('bd', 52, 280, 300, 56, 'norm', 'get_builder_cls → builder', 'SchedulerOutput → AttentionMetadata（block_table / seq_lens / slot_mapping）', 11, 7);
  const im = mk('im', 384, 280, 300, 56, 'attn', 'get_impl_cls → Impl.forward', 'hidden_states + metadata + kv_cache → 调 3.1 算子算法', 11, 7.5);
  body.push(edge(n2, bd, { nodes, portFrom: n2.bottom, portTo: bd.top, pulse: true, dur: 1.3, delay: 0.9, uid: UID, eid: 'c2' }));
  body.push(arrowHead(bd.top[0], bd.top[1], 0, 1));
  body.push(edge(n2, im, { nodes, portFrom: n2.bottom, portTo: im.top, pulse: true, dur: 1.3, delay: 1.1, uid: UID, eid: 'c3' }));
  body.push(arrowHead(im.top[0], im.top[1], 0, 1));
  body.push(edge(bd, im, { nodes, pulse: true, dur: 1, delay: 1.6, uid: UID, eid: 'c4' }));
  body.push(arrowHead(im.left[0], im.left[1], 1, 0));
  body.push(cap(368, 274, 'metadata', { size: 7.5, anchor: 'middle' }));

  const out = mk('out', 384, 356, 300, 44, 'data', '(O, LSE) — LSE 是标准输出（3.1 §6）', null, 11);
  body.push(edge(im, out, { nodes, portFrom: im.bottom, portTo: out.top, pulse: true, dur: 1.1, delay: 2.0, uid: UID, eid: 'c5' }));
  body.push(arrowHead(out.top[0], out.top[1], 0, 1));

  body.push(cap(204, 74, 'KVCacheSpecKind 九种', { size: 8.5, anchor: 'end' }));
  body.push(cap(204, 132, '按 AttentionSelectorConfig 缓存', { size: 8.5, anchor: 'end' }));
  body.push(cap(558, 74, '对接 5.2 spec 数据类', { size: 8.5 }));
  body.push(cap(558, 132, '不是每步一次', { size: 8.5 }));
  body.push(cap(204, 262, '单项支持 ≠ 组合合法', { size: 8.5, anchor: 'end' }));
  body.push(cap(204, 382, 'split-KV / cascade / absorbed MLA', { size: 8.5, anchor: 'end' }));

  body.push(cap(36, 416, 'block size 偏好：默认 64（config/cache.py L185）· mla/sparse_swa 返回 256 = 偏好覆盖 · cpu_mla 只接受 16 = supports_block_size 硬约束 + 平台强制', { size: 9 }));
  body.push(cap(36, 432, '最终 b = 内存侧权衡（2.1 式 7）与 kernel 访存粒度侧约束的交集，不是任一单独最优 —— 式 (6)(7) 的代码接缝', { size: 9 }));

  return scene(720, 444, body, { label: 'attention 调用链：配置与 KV spec 种类经 selector 六项能力过滤选出 backend，builder 产 metadata，impl 调算子算法输出 O 与 LSE' }).svg();
}

/* ============================================================
 * 5.3 §6（block 2）— 编译与图捕获流水线
 * 原文：support_torch_compile → Dynamo → split_graph L580（partition_rules）
 *       → 逐段 Inductor → RangeEntry（shape）/ CUDAGraphEntry（size）两索引
 * ============================================================ */
function compilePipeline() {
  const UID = 'i5i';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '编译与图捕获流水线（VLLM_COMPILE = 3 + FULL_AND_PIECEWISE，V1 默认）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono: true, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const chain = [
    ['p0', '模型类', 'decorators L88'],
    ['p1', 'Dynamo trace', 'FX graph'],
    ['p2', 'CompilerManager', 'split_graph L580'],
    ['p3', '逐段 Inductor', 'Interpreter L714'],
    ['p4', 'PiecewiseBackend', 'RangeEntry L80'],
    ['p5', 'CUDAGraphWrapper', 'CUDAGraphEntry L127'],
  ].map(([id, t, s], i) => mk(id, 52 + i * 106, 48, 96, 56, i === 3 ? 'proj' : 'op', t, s, 8.5, 7));
  for (let i = 0; i < 5; i++) {
    body.push(edge(chain[i], chain[i + 1], { nodes, pulse: true, dur: 1, delay: i * 0.3, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(chain[i + 1].left[0], chain[i + 1].left[1], 1, 0));
  }
  body.push(cap(360, 124, '切分点 = partition_rules.py（attention 算子）· passes/：fusion / collective_fusion（编译器级通信-计算重叠）', { size: 9, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 140, w: 648, h: 96, tone: 'seal', label: '运行时：replay 段 1 → attention（eager）→ replay 段 2 → … 逐层重复' }));
  const r1 = mk('r1', 52, 170, 180, 48, 'proj', 'replay 段 1', 'norm+qkv+cast 子图（静态 shape）', 10.5, 7.5);
  const at = mk('at', 268, 170, 180, 48, 'attn', 'attention（eager）', '读当前 T / block table', 10.5, 7.5);
  const r2 = mk('r2', 484, 170, 180, 48, 'proj', 'replay 段 2', 'o+FFN+norm 子图', 10.5, 7.5);
  [[r1, at, 1.2], [at, r2, 1.6]].forEach(([a, b, dl]) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `r${dl}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 252, w: 648, h: 76, label: '两个索引键不要混' }));
  mk('k1', 52, 280, 300, 36, 'cache', 'RangeEntry：shape 区间 → 跑哪份编译代码', null, 9.5);
  mk('k2', 384, 280, 300, 36, 'cache', 'CUDAGraphEntry：捕获 size → 重放哪张图', null, 9.5);

  body.push(cap(36, 348, '启动代价两笔：每个捕获 size 各捕一张图 × warmup（启动时间）· 图内存池 + 常驻 buffer 占 HBM —— 与 KV 容量互斥（5.2 规划须预留）', { size: 9.5 }));
  body.push(cap(36, 364, 'piecewise 依赖 VLLM_COMPILE 且 splitting_ops 非空；full graph 有无编译都支持 · FULL_DECODE_ONLY 给 PD 的 D 实例省图内存（8.2）', { size: 9.5 }));
  body.push(cap(36, 380, '锚点：compilation/{backends,piecewise_backend,cuda_graph,partition_rules}.py + config/compilation.py（CompilationMode L38 / CUDAGraphMode L165）', { size: 9 }));

  return scene(720, 392, body, { label: '编译流水线：装饰器声明入口，Dynamo 出 FX 图，按 attention 切段逐段 Inductor 编译，shape 区间选产物、捕获 size 选图，运行时静态段 replay 动态 attention eager' }).svg();
}

/* ============================================================
 * 5.3 §8（block 3）— DBO ping-pong：max + min/2，上界 4/3
 * 原文：C=6 · M=4 → 串行 10 / DBO 8 ms（1.25×）；C=M=5 → 7.5（4/3 取等）；
 *       前提失效三条；DBO ≠ AsyncScheduler
 * ============================================================ */
function dboPingPong() {
  const UID = 'i5j';
  const body = [];
  const X = (t) => 140 + t * 52; // 52px/ms

  body.push(cap(36, 24, 'DBO ping-pong（C=6 ms · M=4 ms，半批各 C/2=3 · M/2=2）：单步 makespan = max(C,M) + min(C,M)/2', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(52, 73, '串行', { size: 9.5 }));
  body.push(cap(296, 54, 'compute C=6', { size: 8.5, anchor: 'middle' }));
  body.push(cap(556, 54, 'comm M=4', { size: 8.5, anchor: 'middle' }));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-ink" x="${X(0)}" y="60" width="312" height="18" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-warm" x="${X(6)}" y="60" width="208" height="18" rx="3"/></g>`);
  body.push(cap(668, 73, '10 ms', { mono: true, size: 9 }));

  body.push(cap(52, 123, '半批 1', { size: 9.5 }));
  body.push(cap(52, 153, '半批 2', { size: 9.5 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="${X(0)}" y="110" width="156" height="18" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-warm" x="${X(3)}" y="110" width="104" height="18" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:0.8"><rect class="dia-bar dia-bar-seal" x="${X(3)}" y="140" width="156" height="18" rx="3"/></g>`);
  body.push(`<g class="sig-bar" style="--i:1.0"><rect class="dia-bar dia-bar-warm" x="${X(6)}" y="140" width="104" height="18" rx="3"/></g>`);
  body.push(cap(218, 106, 'C/2=3', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(348, 106, 'M/2=2', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(374, 136, 'C/2=3', { mono: true, size: 8, anchor: 'middle' }));
  body.push(cap(504, 136, 'M/2=2', { mono: true, size: 8, anchor: 'middle' }));
  body.push(`<line x1="${X(3)}" y1="104" x2="${X(3)}" y2="164" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(`<line x1="${X(6)}" y1="104" x2="${X(6)}" y2="164" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(pathPulse(`M ${X(0)} 119 H ${X(5)}`, { dur: 2, delay: 0.5 }));
  body.push(pathPulse(`M ${X(3)} 149 H ${X(8)}`, { dur: 2, delay: 1.1 }));
  body.push(cap(374, 180, '半批 2 计算与半批 1 通信重叠（ping-pong，FusedMoEModularKernel 提供 yield points）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(`<path class="dia-edge strong" d="M ${X(0)} 190 H ${X(8)}"/>`);
  body.push(`<line class="dia-edge strong" x1="${X(0)}" y1="186" x2="${X(0)}" y2="194"/>`);
  body.push(`<line class="dia-edge strong" x1="${X(8)}" y1="186" x2="${X(8)}" y2="194"/>`);
  body.push(cap(348, 206, '总长 8 ms = 6 + 2 → 加速 1.25×（C=M=5 时 7.5 ms = 4/3 上界取等）', { size: 9.5, anchor: 'middle' }));

  body.push(`<line class="dia-edge strong" x1="${X(0)}" y1="220" x2="${X(10)}" y2="220"/>`);
  [[0, '0'], [3, '3'], [5, '5'], [6, '6'], [8, '8'], [10, '10']].forEach(([t, lab]) => {
    body.push(`<line class="dia-edge" x1="${X(t)}" y1="215" x2="${X(t)}" y2="225"/>`);
    body.push(cap(X(t), 238, lab, { mono: true, size: 8.5, anchor: 'middle' }));
  });
  body.push(cap(660, 252, 't (ms)', { mono: true, size: 8.5, anchor: 'end' }));

  body.push(cap(36, 278, '命题 3：加速 = (C+M)/(max+min/2) ≤ 4/3，C=M 取等 —— 两资源对半切的上界低于 NanoFlow 三资源流水的 2×（3.4 式 4，同参数 12/6）', { size: 9.5 }));
  body.push(cap(36, 296, '前提失效三条：通信占 SM（DeepEP dispatch/combine 干扰）· 半批 kernel 效率下降（GEMM / padding）· 步间重叠把上界放宽到 (C+M)/max', { size: 9.5 }));
  body.push(cap(36, 314, 'DBO（compute↔comm，ubatching.py UBatchContext L21）≠ AsyncScheduler（CPU↔GPU，5.2 §8）；SGLang TBO/SBO 同族（6.2 §9）', { size: 9.5 }));

  return scene(720, 328, body, { label: 'DBO 时序：串行计算加通信十毫秒，两半批 ping-pong 使半批二计算与半批一通信重叠，总长八毫秒，加速上界四分之三取等于计算通信等长' }).svg();
}

/* ============================================================
 * 5.3 §3（afterHeading）— TP 通信账：位点减半 × 延迟主导（签名）
 * 原文：每层 2 处 g（f 恒等）；M_msg=2Bd；τ_TP=2L(τ_lat+2(t−1)/t·M/BW)；
 *       B=64 → 385 μs、B=1 → 321 μs（消息缩 64× 时间只降 16.7%）
 * ============================================================ */
function tpCommAccount() {
  const UID = 'i5k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'TP 通信账：serving 没有 backward —— 每层 2 处归约位点，小 batch decode 的成本几乎全是延迟', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 300, h: 200, label: '一层 transformer（只跑前向图）' }));
  const mk = (id, x, y, w, h, role, title, sub, ts = 10.5, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const hin = mk('hin', 52, 70, 120, 36, 'data', 'hidden [B, 4096]', null, 9.5);
  const ab = mk('ab', 52, 120, 120, 40, 'attn', 'attention 块', 'out 投影行切', 10.5, 7.5);
  const g1 = mk('g1', 210, 120, 100, 40, 'op', 'g：all-reduce', '位点 1', 9.5, 7.5);
  const mb = mk('mb', 52, 176, 120, 40, 'proj', 'MLP 块', 'down 投影行切', 10.5, 7.5);
  const g2 = mk('g2', 210, 176, 100, 40, 'op', 'g：all-reduce', '位点 2', 9.5, 7.5);
  body.push(edge(hin, ab, { nodes, portFrom: hin.bottom, portTo: ab.top, pulse: true, dur: 1, delay: 0, uid: UID, eid: 't0' }));
  body.push(arrowHead(ab.top[0], ab.top[1], 0, 1));
  body.push(edge(ab, g1, { nodes, pulse: true, dur: 1.1, delay: 0.3, uid: UID, eid: 't1' }));
  body.push(arrowHead(g1.left[0], g1.left[1], 1, 0));
  body.push(edge(g1, mb, { nodes, portFrom: g1.bottom, portTo: mb.top, pulse: true, dur: 1.1, delay: 0.7, uid: UID, eid: 't2' }));
  body.push(arrowHead(mb.top[0], mb.top[1], 0, 1));
  body.push(edge(mb, g2, { nodes, pulse: true, dur: 1.1, delay: 1.1, uid: UID, eid: 't3' }));
  body.push(arrowHead(g2.left[0], g2.left[1], 1, 0));
  body.push(cap(186, 232, '2 位点/层 × L=32 → 每步 64 处归约（命题 1）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(36, 268, 'f 前向恒等（归约在其反向）—— 推理图里只有 g 触发；training 每层 4 次含反向，serving 减半', { size: 8.5 }));

  body.push(cap(372, 64, '消息量（式 2）：M_msg = 2·B·d_model（bf16）', { mono: true, size: 9.5 }));
  body.push(cap(372, 88, 'decode B=64 → 512 KiB · B=1 → 8 KiB', { mono: true, size: 9.5 }));
  body.push(cap(372, 106, 'prefill T=4096 → 32 MiB（decode B=64 的 64×）', { mono: true, size: 9.5 }));
  body.push(cap(372, 134, 'τ_TP = 2L·(τ_lat + 2(t−1)/t · M/BW)（式 3）', { mono: true, size: 9.5 }));
  body.push(cap(372, 152, 't=8 · τ_lat=5 μs（教学值）· NVLink 900 GB/s', { size: 9 }));

  // 延迟/带宽分解条：0.8px/μs
  body.push(cap(372, 176, 'B=64：64 × (5 + 1.02) = 385 μs', { size: 8.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="372" y="182" width="256" height="14" rx="3"/>`);
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-seal" x="631" y="182" width="52" height="14" rx="3"/></g>`);
  body.push(cap(372, 208, '延迟 320 μs（83%）+ 带宽 65 μs', { mono: true, size: 8 }));
  body.push(cap(372, 226, 'B=1：64 × (5 + 0.016) = 321 μs', { size: 8.5 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="372" y="232" width="256" height="14" rx="3"/>`);
  body.push(`<g class="sig-bar" style="--i:0.6"><rect class="dia-bar dia-bar-seal" x="631" y="232" width="2" height="14" rx="1"/></g>`);
  body.push(cap(372, 258, '消息缩 64×，时间只降 16.7% —— 延迟项与消息量无关', { size: 9, color: 'var(--seal-deep)' }));
  body.push(cap(372, 278, 'prefill：64 × (5 + 65.2) = 4.50 ms —— 带宽项主导，与 decode 相反', { size: 9 }));

  body.push(cap(36, 300, '口径：2.55%（385 μs / 15.1 ms）与 6.67%（321 μs / 4.81 ms）的分母是单卡基线；TP 后步时本身下降、通信占比上升 —— 度数选择与 TPOT 全账归 8.1', { size: 9 }));
  body.push(cap(36, 318, 'SP 变体：归约原语换 reduce-scatter + all-gather 对 —— 原语数翻倍但位点仍是每层 2 处（命题计位点非 kernel 数）；vocab parallel embedding 另加 1 处', { size: 9 }));
  body.push(cap(36, 336, '三条归约路径（§4）：NCCL 默认回退 · CustomAllreduce L120（should_custom_ar L511）· SymmMem L29 —— decode 小消息压延迟项、prefill 大消息压带宽项', { size: 9 }));

  return scene(720, 350, body, { label: 'TP 通信账：一层前向只有 attention 与 MLP 两处 g 归约，decode 消息量随 batch 缩但延迟项不变，三百八十五微秒中三百二十是延迟' }).svg();
}

/* ============================================================
 * 5.4 §2（block 0）— 投机一轮调用链：薄壳 → 四层继承 → verify/accept
 * 原文：eagle 18 行 / mtp 63 行薄壳；propose L103 在第 4 层；
 *       RejectionSampler L152；ngram 自实现 propose 不在此链
 * ============================================================ */
function specCallChain() {
  const UID = 'i5l';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '投机一轮的调用链（EAGLE / MTP 共用）：薄壳只管权重加载，propose 语义由第 4 层决定（命题 1）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, y, h, role, title, sub, ts = 11, ss = 7.5) => {
    const n = node({ id, x: 250, y, w: 330, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const n0 = mk('n0', 48, 48, 'attn', 'target forward（verify，γ+1 个位置）', 'last_hidden_states / aux_hidden_states（4.5 §5 接口）');
  const n1 = mk('n1', 112, 56, 'plain', 'Eagle 18 行 / MTP 63 行薄壳', '只 override load_draft_model · MTP 另 4 个生命周期 hook（index 共享）', 11, 7);
  const n2 = mk('n2', 184, 56, 'proj', 'TargetDependentARSpeculator.propose L103', 'prepare_prefill_inputs L525 / update_draft_inputs L692', 10.5, 7.5);
  const n3 = mk('n3', 256, 48, 'op', 'AutoRegressiveSpeculator', '_run_multi_step_decode L167 / _fused L321 / capture L118', 10.5, 7.5);
  const n4 = mk('n4', 320, 40, 'data', 'draft tokens [num_reqs, γ]', null, 11);
  const n5 = mk('n5', 376, 48, 'attn', 'RejectionSampler L152', '接受前缀 + 残差重采样（4.4 命题 1 · 分布保持）');
  [[n0, n1, 0], [n1, n2, 0.4], [n2, n3, 0.8], [n3, n4, 1.2], [n4, n5, 1.6]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `c${i}` }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(cap(592, 140, '分派向上（MRO）', { size: 8.5 }));

  body.push(cap(238, 76, '消费 target 隐藏态', { size: 8.5, anchor: 'end' }));
  body.push(cap(238, 140, '不 override propose', { size: 8.5, anchor: 'end' }));
  body.push(cap(238, 212, 'Triton kernel L443 / L565', { size: 8.5, anchor: 'end' }));
  body.push(cap(238, 226, 'GPU 侧输入准备', { size: 8.5, anchor: 'end' }));
  body.push(cap(238, 284, 'draft 侧图捕获', { size: 8.5, anchor: 'end' }));
  body.push(cap(238, 404, 'verify / accept 共用', { size: 8.5, anchor: 'end' }));
  body.push(cap(592, 76, 'γ=4 · α=0.8 章脊', { size: 8.5 }));
  body.push(cap(592, 212, '不重跑主干', { size: 8.5 }));
  body.push(cap(592, 344, 'B(γ+1) token 预算', { size: 8.5 }));
  body.push(cap(592, 404, 'ngram 不在此链', { size: 8.5 }));

  body.push(cap(36, 448, '接受侧自适应：OnlineAcceptanceEstimator L345 → AdaptiveVerificationManager L227（动态调 γ）→ SpecDecodingStats L19 / Prom L195', { size: 9.5 }));
  body.push(cap(36, 466, '边界反例：NgramGPUSpeculator L151 自实现 propose L213（GPU n-gram 匹配，Triton _ngram_scan_kernel L21）—— 查表路线不消费 target 隐藏态', { size: 9.5 }));
  body.push(cap(36, 484, '配置三枚举正交：SpeculativeMethod L113（draft 来源）× DraftSampleMethod L123（draft 采样）× RejectionSampleMethod L122（接受规则）', { size: 9.5 }));

  return scene(720, 496, body, { label: '投机调用链：target 前向产隐藏态，薄壳分派向上到第四层 propose，自回归 draft 产 γ 个 token，拒绝采样接受前缀并残差重采样' }).svg();
}

/* ============================================================
 * 5.4 §5（block 1）— SGMV 分段 gather（签名：LoRA multi-adapter batch）
 * 原文：n_adp=3 · 8 token 分段；shrink/expand 各一次 launch；base 只读一份；
 *       naive 44800 launch = 224 ms vs SGMV 448 = 2.24 ms（100×）
 * ============================================================ */
function sgmvSegments() {
  const UID = 'i5m';
  const body = [];
  const nodes = [];
  const TONES = [
    ['var(--seal-wash)', 'var(--seal)', 'attn'],
    ['#f3ecdc', '#cbb98d', 'cache'],
    ['#ecf0f6', '#aebcd2', 'proj'],
  ];
  const ADP = [0, 0, 1, 1, 1, 2, 2, 2];

  body.push(cap(36, 24, 'SGMV 分段 gather（n_adp=3 · 一个 linear 位点）：adapter 维成为 kernel 内分段索引，而非 launch 维', { size: 12, color: 'var(--ink-soft)' }));

  // batch token 条（按 adapter 染色）
  ADP.forEach((a, i) => {
    const x = 52 + i * 72;
    const [fill, stroke] = TONES[a];
    body.push(`<rect x="${x}" y="56" width="64" height="36" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.1"/>`);
    body.push(`<text class="dia-title dia-mono" x="${x + 32}" y="72" text-anchor="middle" font-size="11">t${i + 1}</text>`);
    body.push(`<text class="dia-sub dia-mono" x="${x + 32}" y="86" text-anchor="middle" font-size="7.5">adp ${a}</text>`);
  });
  // 分段括号
  [[52, 188], [196, 404], [412, 620]].forEach(([x1, x2]) => {
    body.push(`<path class="dia-edge" d="M ${x1} 96 V 102 H ${x2} V 96"/>`);
  });

  const segCards = (y, prefix, role) => [52, 196, 412].map((x, i) => {
    const w = i === 0 ? 136 : 208;
    const n = node({ id: `${prefix}${i}`, x, y, w, h: 40, role: TONES[i][2], title: `段 ${i} @ ${prefix === 'wd' ? 'W_down' : 'W_up'}[${i}]`, mono: true, titleSize: 9.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  const wd = segCards(124, 'wd');
  const wu = segCards(190, 'wu');
  const centers = [120, 300, 516];
  centers.forEach((cx, i) => {
    body.push(edge([cx, 102], [cx, 124], { pulse: true, dur: 1, delay: 0.2 + i * 0.2, uid: UID, eid: `g${i}` }));
    body.push(arrowHead(cx, 124, 0, 1));
    body.push(edge(wd[i], wu[i], { nodes, portFrom: wd[i].bottom, portTo: wu[i].top, pulse: true, dur: 1, delay: 1.0 + i * 0.2, uid: UID, eid: `s${i}` }));
    body.push(arrowHead(cx, 190, 0, 1));
    body.push(edge([cx, 230], [cx, 252], { pulse: true, dur: 1, delay: 1.8 + i * 0.15, uid: UID, eid: `b${i}` }));
    body.push(arrowHead(cx, 252, 0, 1));
  });
  body.push(cap(684, 118, 'shrink · 一次 launch', { size: 8.5, anchor: 'end' }));
  body.push(cap(684, 184, 'expand · 一次 launch', { size: 8.5, anchor: 'end' }));
  body.push(cap(684, 246, 'addon ⊕ base 输出', { size: 8.5, anchor: 'end' }));

  const base = node({ id: 'base', x: 52, y: 252, w: 632, h: 44, role: 'proj', title: 'base GEMM：全部 8 个 token 共享 W，只读一份（命题 2：base 流量与 n_adp 无关）', sub: 'y = Wx + W_up(W_down x) —— LoRA addon 与 base GEMM 在不同 kernel（r=16 增量 FLOPs 仅 0.5–0.78%）', titleSize: 11, subSize: 8 });
  nodes.push(base); body.push(base.svg);

  body.push(cap(110, 326, 'naive 2·n_adp·n_site = 44800 次 launch × 5 μs = 224 ms', { size: 8.5 }));
  body.push(cap(52, 341, 'naive', { size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0"><rect class="dia-bar dia-bar-soft" x="110" y="332" width="500" height="14" rx="3"/></g>`);
  body.push(cap(52, 367, 'SGMV', { size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="110" y="358" width="5" height="14" rx="2"/></g>`);
  body.push(cap(124, 369, '2·n_site = 448 次 = 2.24 ms —— 差 100×（恰为 n_adp）；已占 15.1 ms 步时 14.8% → 须进 CUDA Graph 或进一步融合', { size: 9, color: 'var(--seal-deep)' }));

  body.push(cap(36, 398, '驻留账（式 4）：n_hot=8 → 0.72 GB · 100 → 9.02 GB · 1000 → 90.2 GB > 单卡 HBM 80 GB —— S-LoRA：adapter 主存常驻、按需取 + Unified Paging', { size: 9.5 }));
  body.push(cap(36, 416, '命题 3：block hash extra keys 含 lora / mm → 不同 adapter 同 prompt 也不共享 prefix cache 块 —— 命中率账须按 adapter 分组（2.2）', { size: 9.5 }));
  body.push(cap(36, 434, 'vLLM 生产 kernel = 自有 Triton shrink + expand（lora/ops/triton_ops/）；Punica master 快照 sgmv_expand 未实现 —— 机制按论文 SGMV 讲', { size: 9.5 }));
  body.push(cap(36, 452, '全模型口径（8B · r=16）：45.09 M 参数 / 90.2 MB = base 权重 0.56% —— 算力与权重流量都是小量，贵的是 kernel 数（先数 launch 再算字节）', { size: 9.5 }));

  return scene(720, 464, body, { label: 'SGMV 分段：八个 token 按三个 adapter 染色分段，shrink 与 expand 各一次 launch 按段 gather 权重，base GEMM 全批共享一份，launch 数从四万四千八降到四百四十八' }).svg();
}

/* ============================================================
 * 5.4 §6（block 2）— colocate RL rollout：sleep → 更新 → wake_up
 * 原文：CuMemAllocator tag 分池释放物理页保留虚拟地址；23.9 → 80 GB（3.35×）；
 *       命题 4：wake_up 后旧 KV 不可信，须重建或 connector 取回
 * ============================================================ */
function sleepWakeCycle() {
  const UID = 'i5n';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'colocate RL rollout 一轮：sleep 释放物理页（保留虚拟地址）→ 训练器占卡更新权重 → wake_up 按 tags 恢复', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 208, h: 130, label: 'rollout 阶段' }));
  body.push(zone({ x: 256, y: 40, w: 208, h: 130, tone: 'seal', label: '训练阶段' }));
  body.push(zone({ x: 476, y: 40, w: 208, h: 130, label: '恢复阶段' }));
  const mk = (id, x, role, title, sub) => {
    const n = node({ id, x, y: 66, w: 176, h: 44, role, title, sub, titleSize: 11.5, subSize: 8 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const p1 = mk('p1', 52, 'attn', '引擎正常 serving', '权重 + KV 占 56.1 GB');
  const p2 = mk('p2', 272, 'proj', '训练器占卡', '可用显存 23.9 → 80 GB 量级');
  const p3 = mk('p3', 492, 'norm', 'wake_up(tags)', 'weights 池按 level 恢复');
  body.push(cap(140, 132, '权重 16.1 + KV 池 40 GB', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(140, 150, '训练器可用 23.9 GB', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(368, 132, '3.35× 显存交还', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(368, 150, 'weight_transfer/ engines 更新', { size: 8 }));
  body.push(cap(584, 132, 'KV 池必重建（命题 4）', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(584, 150, 'set_weight_version / checksums', { mono: true, size: 7.5, anchor: 'middle' }));
  [[p1, p2, 0.2, 'sl'], [p2, p3, 1.2, 'wk']].forEach(([a, b, dl, eid]) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.4, delay: dl, uid: UID, eid }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  const cm = node({ id: 'cm', x: 256, y: 186, w: 208, h: 48, role: 'cache', title: 'CuMemAllocator L62', sub: 'CUDA VMM：tag 分池 weights / kv_cache', titleSize: 11, subSize: 7.5 });
  nodes.push(cm); body.push(cm.svg);
  body.push(edge(p2, cm, { nodes, portFrom: p2.bottom, portTo: cm.top, dash: true, pulse: true, dur: 1.4, delay: 0.7, uid: UID, eid: 'cm' }));
  body.push(arrowHead(cm.top[0], cm.top[1], 0, 1));
  body.push(cap(476, 206, '释放物理页 · 保留虚拟地址', { size: 8.5 }));
  body.push(cap(476, 220, 'wake_up 重新映射（内容不保证）', { size: 8.5 }));

  const loop = 'M 668 88 H 700 V 252 H 140 V 170';
  body.push(`<path class="dia-edge strong" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 2.6, delay: 2.0 }));
  body.push(arrowHead(140, 170, 0, -1));
  body.push(cap(400, 266, '下一轮 rollout —— KV 重建：重算 prefill 或经 connector 取回（§4）', { size: 9, anchor: 'middle' }));

  body.push(cap(52, 296, '显存交还账（colocate · 80 GB 单卡，教学部署值）：', { size: 9.5 }));
  body.push(cap(166, 306, '权重 16.1', { size: 8, anchor: 'middle' }));
  body.push(cap(366, 306, 'KV 池 40', { size: 8, anchor: 'middle' }));
  body.push(cap(592, 306, '训练器 23.9', { size: 8, anchor: 'middle' }));
  body.push(cap(52, 321, 'sleep 前', { size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="110" y="310" width="113" height="14" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="226" y="310" width="280" height="14" rx="3"/>`);
  body.push(`<rect x="509" y="310" width="167" height="14" rx="3" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>`);
  body.push(cap(390, 336, 'sleep 后：训练器可用 ~80 GB 量级（3.35×）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(52, 351, 'sleep 后', { size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.5"><rect class="dia-bar dia-bar-ink" x="110" y="340" width="560" height="14" rx="3"/></g>`);

  body.push(cap(36, 374, '命题 4：sleep 释放 kv_cache 池物理页 → wake_up 重映射后内容未定义 —— 沿用旧 block table 直接 decode 会静默损坏；必须重建或经 connector 取回', { size: 9.5 }));
  body.push(cap(36, 392, 'weights 池按 level 保留到 host → wake_up 可恢复原权重（内容由 host 副本保证）；KV 池无此路径（除非事先经 offload / connector 存出）', { size: 9.5 }));
  body.push(cap(36, 410, 'SGLang 对照：release / resume_memory_occupation（6.3 §7）—— 权重更新后必须 flush KV cache：权重是 cache 版本元组的分量（2.4 §8）', { size: 9.5 }));
  body.push(cap(36, 428, 'sleep 链：LLM.sleep L853 → EngineCore.sleep L1103 → Worker.sleep L351；wake_up：L897 → L1156 → L389(tags) —— CuMemAllocator 按 tag 操作', { size: 9 }));

  return scene(720, 440, body, { label: 'colocate 循环：rollout 后 sleep 按 tag 释放物理页训练器占卡更新权重，wake_up 恢复权重但 KV 池必须重建，显存交还三点三五倍' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '05-vllm/01-vllm-architecture',
    replacesBlock: 0,
    title: 'V1 进程架构：四类进程 × 17 进程（tp=2 · dp=4）',
    svg: processArch,
  },
  {
    dim: 'inference',
    slug: '05-vllm/01-vllm-architecture',
    replacesBlock: 1,
    title: '一次请求完整数据流：busy loop 五段 × grammar 重叠',
    svg: requestFlow,
  },
  {
    dim: 'inference',
    slug: '05-vllm/01-vllm-architecture',
    replacesBlock: 2,
    title: 'vllm/v1 模块地图：P1–P4 机制的文件落点',
    svg: moduleMap,
  },
  {
    dim: 'inference',
    slug: '05-vllm/01-vllm-architecture',
    afterHeading: /^## 5\. /,
    title: 'diff 协议控制面账：1.02 MiB → 2 KiB（≈523×）',
    svg: diffProtocol,
  },
  {
    dim: 'inference',
    slug: '05-vllm/02-vllm-scheduler-kv',
    replacesBlock: 0,
    title: 'KV 管理三层：接口 → 协调 → 逐类型（共享一个 BlockPool）',
    svg: kvThreeLayer,
  },
  {
    dim: 'inference',
    slug: '05-vllm/02-vllm-scheduler-kv',
    afterHeading: /^## 6\. /,
    title: 'hybrid 命中不动点：full 3072 × SWA 缩长 2944',
    svg: hybridFixedPoint,
  },
  {
    dim: 'inference',
    slug: '05-vllm/03-vllm-distributed',
    replacesBlock: 0,
    title: '执行链：executor 广播 → worker → runner 两方法分离',
    svg: execChain,
  },
  {
    dim: 'inference',
    slug: '05-vllm/03-vllm-distributed',
    replacesBlock: 1,
    title: 'attention 调用链：六项能力筛选 → builder / impl',
    svg: attnCallChain,
  },
  {
    dim: 'inference',
    slug: '05-vllm/03-vllm-distributed',
    replacesBlock: 2,
    title: '编译与图捕获：shape 区间选产物 × 捕获 size 选图',
    svg: compilePipeline,
  },
  {
    dim: 'inference',
    slug: '05-vllm/03-vllm-distributed',
    replacesBlock: 3,
    title: 'DBO ping-pong：makespan = max + min/2，上界 4/3',
    svg: dboPingPong,
  },
  {
    dim: 'inference',
    slug: '05-vllm/03-vllm-distributed',
    afterHeading: /^## 3\. /,
    title: 'TP 通信账：每层 2 位点 × 延迟项主导',
    svg: tpCommAccount,
  },
  {
    dim: 'inference',
    slug: '05-vllm/04-vllm-features',
    replacesBlock: 0,
    title: '投机调用链：薄壳 → 四层继承 → verify / accept',
    svg: specCallChain,
  },
  {
    dim: 'inference',
    slug: '05-vllm/04-vllm-features',
    replacesBlock: 1,
    title: 'SGMV 分段 gather：一份 base × n_adp 个 adapter',
    svg: sgmvSegments,
  },
  {
    dim: 'inference',
    slug: '05-vllm/04-vllm-features',
    replacesBlock: 2,
    title: 'colocate rollout：sleep → 权重更新 → wake_up（KV 必重建）',
    svg: sleepWakeCycle,
  },
];