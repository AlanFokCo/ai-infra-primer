// inference Part 6（06-sglang）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part01.mjs / inference-part03.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言伪代码围栏不画（管线自动算法面板）。

import { scene, node, edge, zone, cap as kitCap, PULSE_GRAD_ID } from '../figure-kit.mjs';

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

/* ============================================================
 * 6.1 §1（block 0）— 前端 DSL × co-designed 运行时
 * 原文：接口契约 = 前端发完整 prompt、运行时做前缀匹配复用；
 *       gen/select 描述对象异步流；fork 复制文本非 KV；Frontend Hint
 * ============================================================ */
function frontendRuntime() {
  const UID = 'i6a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '两层协同设计：前端 DSL 发完整 prompt，运行时做前缀匹配与复用（接口契约只有一条）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 260, h: 170, label: '前端 lang/（interpreter 模式）' }));
  body.push(zone({ x: 360, y: 40, w: 324, h: 170, label: '后端 srt/（三进程链 + 同进程执行体）' }));
  const mk = (id, x, y, w, h, role, title, sub, ts = 10.5, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono: true, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const it = mk('it', 52, 66, 228, 48, 'op', 'interpreter.py StreamExecutor', 'gen/select/fork → 描述对象 → 异步流', 9.5, 7.5);
  const tr = mk('tr', 52, 130, 228, 44, 'plain', 'tracer.py（图模式）', '编译成计算图 → graph executor', 9.5, 7.5);
  body.push(cap(166, 196, 'fork 复制 prompt 文本而非 KV', { size: 8.5, anchor: 'middle' }));
  const tm = mk('tm', 376, 66, 292, 36, 'data', 'TokenizerManager（tokenize）', null, 10);
  const sc = mk('sc', 376, 114, 292, 48, 'op', 'Scheduler（基数树 + 调度循环）', 'TpModelWorker → ModelRunner（同进程调用）', 10, 7.5);
  const de = mk('de', 376, 170, 292, 36, 'norm', 'DetokenizerManager（增量解码）', null, 10);
  body.push(edge(tm, sc, { nodes, portFrom: tm.bottom, portTo: sc.top, pulse: true, dur: 1.1, delay: 0.4, uid: UID, eid: 't1' }));
  body.push(arrowHead(sc.top[0], sc.top[1], 0, 1));
  body.push(cap(510, 110, 'ZMQ PUSH · TokenizedGenerateReqInput', { size: 7.5, anchor: 'end' }));
  body.push(edge(sc, de, { nodes, portFrom: sc.bottom, portTo: de.top, pulse: true, dur: 1.1, delay: 0.9, uid: UID, eid: 't2' }));
  body.push(arrowHead(de.top[0], de.top[1], 0, 1));
  body.push(cap(510, 168, 'ZMQ PUSH · BatchTokenIDOutput', { size: 7.5, anchor: 'end' }));

  body.push(edge(it, tm, { nodes, pulse: true, dur: 1.4, delay: 0, uid: UID, eid: 'c1' }));
  body.push(arrowHead(tm.left[0], tm.left[1], 1, 0));
  body.push(cap(328, 76, '完整 prompt', { size: 8, anchor: 'middle' }));
  const ret = 'M 376 188 H 328 V 104 H 280';
  body.push(`<path class="dia-edge strong" d="${ret}"/>`);
  body.push(pathPulse(ret, { dur: 1.8, delay: 1.3 }));
  body.push(arrowHead(280, 104, -1, 0));
  body.push(cap(328, 200, '流式结果', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 232, 'gen / select 返回描述对象（SglGen / SglSelect），执行发生在 interpreter 提交异步流之后 —— 取结果 s["name"] 才阻塞等待', { size: 9.5 }));
  body.push(cap(36, 250, 'fork 逐字复制 prompt 文本：size 个副本共享同一段前缀 → 运行时匹配同一棵子树；Frontend Hint 先把公共前缀插进树再发各副本', { size: 9.5 }));
  body.push(cap(36, 268, '契约：前端不需要知道 KV 缓存里有什么 / 在哪台机器 —— 命中判定完全在后端基数树（match_prefix O(T)，2.2 命题 1）', { size: 9.5 }));
  body.push(cap(36, 286, '论文口径：等价程序 2.1× 代码行数；agent / 推理 / JSON / RAG / 多轮 workload 至多 6.4× 吞吐（摘要口径，baseline 含 Guidance / vLLM / LMQL）', { size: 9.5 }));

  return scene(720, 300, body, { label: 'SGLang 两层：前端 interpreter 发完整 prompt 给后端三进程链，运行时基数树做前缀匹配复用，流式结果回前端' }).svg();
}

/* ============================================================
 * 6.1 §3（block 1）— 三进程链：一次请求完整数据流
 * 原文：TokenizerManager L608 / Scheduler L513 / DetokenizerManager L80；
 *       msgspec 四载荷；TpModelWorker 同进程；ForwardMode 八值
 * ============================================================ */
function threeProcessChain() {
  const UID = 'i6b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '一次请求的完整数据流：三进程链 ZMQ PUSH/PULL + msgspec 载荷（TpModelWorker 与 Scheduler 同进程）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10.5, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const ht = mk('ht', 52, 48, 80, 56, 'data', 'HTTP POST', '/generate', 10.5, 8);
  const tm = mk('tm', 150, 48, 150, 56, 'op', 'TokenizerManager', 'L608 · ReqState L281 · tokenize L1275', 10, 7);
  const sc = mk('sc', 318, 48, 180, 56, 'op', 'Scheduler', 'L513 · ingest → waiting · get_next_batch L4386', 10.5, 7);
  const de = mk('de', 516, 48, 168, 56, 'norm', 'DetokenizerManager', 'L80 · event_loop L164 增量', 10, 7.5);
  [[ht, tm, 0], [tm, sc, 0.3], [sc, de, 0.7]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });
  body.push(cap(234, 42, 'TokenizedGenerateReqInput', { size: 7.5, anchor: 'middle' }));
  body.push(cap(497, 42, 'BatchTokenIDOutput', { size: 7.5, anchor: 'middle' }));

  body.push(zone({ x: 318, y: 120, w: 366, h: 96, label: '同进程执行体' }));
  const tp = mk('tp', 334, 150, 160, 44, 'proj', 'TpModelWorker L283', 'forward_batch_generation L624', 9.5, 7);
  const mr = mk('mr', 510, 150, 160, 44, 'attn', 'ModelRunner L332', 'ForwardBatch L442 → GPU', 9.5, 7);
  body.push(edge(sc, tp, { nodes, portFrom: sc.bottom, portTo: tp.top, pulse: true, dur: 1.1, delay: 1.0, uid: UID, eid: 'g1' }));
  body.push(arrowHead(tp.top[0], tp.top[1], 0, 1));
  body.push(cap(452, 132, '同进程调用（无 IPC）', { size: 8 }));
  body.push(edge(tp, mr, { nodes, pulse: true, dur: 1.1, delay: 1.3, uid: UID, eid: 'g2' }));
  body.push(arrowHead(mr.left[0], mr.left[1], 1, 0));
  const back = 'M 590 194 V 232 H 330 V 104';
  body.push(`<path class="dia-edge" d="${back}"/>`);
  body.push(pathPulse(back, { dur: 1.8, delay: 1.6 }));
  body.push(arrowHead(330, 104, 0, -1));
  body.push(cap(460, 244, '结果回 Scheduler → process_batch_result L5957', { size: 8.5, anchor: 'middle' }));

  const ret = 'M 684 76 H 704 V 292 H 468';
  body.push(`<path class="dia-edge strong" d="${ret}"/>`);
  body.push(pathPulse(ret, { dur: 2, delay: 2.1 }));
  body.push(arrowHead(468, 292, -1, 0));
  body.push(cap(560, 286, 'BatchStrOutput · ZMQ PUSH', { size: 7.5, anchor: 'middle' }));
  const t2 = mk('t2', 318, 268, 150, 48, 'op', 'TokenizerManager', 'ReqState → HTTP stream', 10, 7.5);
  const h2 = mk('h2', 150, 268, 80, 48, 'data', 'HTTP stream', 'SSE', 9.5, 8);
  body.push(edge(t2, h2, { nodes, portFrom: t2.left, portTo: h2.right, pulse: true, dur: 1.1, delay: 2.5, uid: UID, eid: 'r1' }));
  body.push(arrowHead(h2.right[0], h2.right[1], -1, 0));

  body.push(cap(52, 344, 'ForwardMode 批形态（forward_mode.py L20）：', { size: 9 }));
  [['EXTEND', 52, 70], ['DECODE', 130, 70], ['MIXED', 208, 64], ['IDLE', 280, 52], ['TARGET_VERIFY', 340, 104], ['PREBUILT', 452, 76]].forEach(([t, x, w], i) => {
    const n = node({ id: `fm${i}`, x, y: 352, w, h: 26, role: 'plain', title: t, mono: true, titleSize: 8 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(684, 344, 'DRAFT_EXTEND_V2 / DLLM_EXTEND 归 4.5 / 9.4', { size: 8, anchor: 'end' }));
  body.push(cap(52, 394, 'EXTEND = prefill 剩余段 · MIXED = chunked 混批 · IDLE = DP 陪跑 collective · PREBUILT = PD KV 就绪（不进 forward）', { size: 8.5 }));

  return scene(720, 404, body, { label: 'SGLang 三进程链：HTTP 经 TokenizerManager ZMQ 到 Scheduler，同进程 TpModelWorker 与 ModelRunner 执行，结果经 DetokenizerManager 增量解码回 TokenizerManager 流式返回' }).svg();
}

/* ============================================================
 * 6.1 §5（block 2）— 两级池：索引表 → KV 本体
 * 原文：req_to_token[7,t]=4242 → k/v_buffer[layer][4242]；
 *       TreeNode.value 索引切片；48 GiB → 393216 slot；表 32 MiB
 * ============================================================ */
function twoLevelPool() {
  const UID = 'i6c';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '两级池：请求级索引表 → token 级 KV 本体；基数树只管索引的共享与驱逐', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 300, h: 150, label: 'req_to_token [rows, max_context_len] int32' }));
  const rows = [[70, '0', null], [96, '6', null], [122, '7', 2], [148, '⋯', null]];
  rows.forEach(([y, lab, hot]) => {
    body.push(cap(52, y + 15, lab, { mono: true, size: 8 }));
    for (let j = 0; j < 6; j++) {
      const x = 76 + j * 38;
      const isHot = hot === j;
      const pad = lab === '0';
      body.push(`<rect x="${x}" y="${y}" width="34" height="22" rx="4" fill="${isHot ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${isHot ? 'var(--seal)' : pad ? 'var(--rule)' : '#c6c0af'}" stroke-width="${isHot ? 1.4 : 0.9}"${pad ? ' stroke-dasharray="2 3"' : ''}/>`);
      const val = lab === '0' ? 0 : lab === '6' ? 4234 + j : 4240 + j;
      if (lab !== '⋯') body.push(cap(x + 17, y + 15, String(val), { mono: true, size: 7.5, anchor: 'middle', color: isHot ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
    }
  });
  body.push(cap(186, 182, '行 0 = CUDA Graph padded 假行（读写落此无害）', { size: 8, anchor: 'middle' }));

  body.push(zone({ x: 360, y: 40, w: 324, h: 150, label: 'token_to_kv_pool（KV 本体 · 逐层 buffer）' }));
  const kb = node({ id: 'kb', x: 376, y: 70, w: 292, h: 40, role: 'cache', title: 'k_buffer[layer][4242]', sub: '[H_kv, d_head] = [8, 128] bf16', mono: true, titleSize: 10, subSize: 7.5 });
  const vb = node({ id: 'vb', x: 376, y: 122, w: 292, h: 40, role: 'cache', title: 'v_buffer[layer][4242]', sub: 'MHA/GQA：κ = 2·H_kv·d_head·bytes', mono: true, titleSize: 10, subSize: 7.5 });
  nodes.push(kb, vb); body.push(kb.svg, vb.svg);
  body.push(edge([186, 133], kb, { nodes, portTo: kb.left, pulse: true, dur: 1.4, delay: 0.2, uid: UID, eid: 'k' }));
  body.push(arrowHead(kb.left[0], kb.left[1], 1, 0));
  body.push(edge([186, 133], vb, { nodes, portTo: vb.left, pulse: true, dur: 1.4, delay: 0.5, uid: UID, eid: 'v' }));
  body.push(arrowHead(vb.left[0], vb.left[1], 1, 0));
  body.push(cap(186, 204, 'req_to_token[7, t] = 4242', { mono: true, size: 8.5, anchor: 'middle' }));
  body.push(cap(522, 182, 'MLA 池（V3）：单 kv_buffer [rows, 1, 576] —— 无因子 2、无 H_kv 乘', { size: 8, anchor: 'middle' }));

  const tn = node({ id: 'tn', x: 36, y: 210, w: 300, h: 44, role: 'attn', title: 'TreeNode.value = [4242, 4243, …]', sub: '索引切片：可共享 / 可分裂（分裂只拷索引，KV 本体不动）', mono: true, titleSize: 9.5, subSize: 7 });
  nodes.push(tn); body.push(tn.svg);
  body.push(cap(360, 232, '基数树只持索引 —— match / split / evict', { size: 9 }));
  body.push(cap(360, 248, '都操作 value 切片，KV 本体留在池里（2.2 命题 1）', { size: 9 }));

  body.push(cap(36, 280, '容量账：KV 池 48 GiB / κL 128 KiB = 393216 slot；请求表 2048 行 × 4096 列 × 4 B = 32 MiB（占池 0.065%）', { mono: true, size: 9.5 }));
  body.push(cap(36, 298, 'page=1 无对齐尾部损失（vLLM b=64 期望损失 32 token ≈ 4 MiB）；代价是 free_pages 按 token 计、元数据操作更多（6.2 §1）', { size: 9.5 }));
  body.push(cap(36, 316, 'slot 0 ↔ 行 0 ↔ FutureMap slot 0 三处同构：CUDA Graph padded 批的假读写全落 padding（6.2 §5）', { size: 9.5 }));
  body.push(cap(36, 334, '分配器两档：token.py page=1（free_pages 前缀切）/ paged.py page>1（索引恒 page 对齐，decode 按 (seq_lens−1)%page==0 开新页）', { size: 9.5 }));

  return scene(720, 348, body, { label: '两级池映射：请求表第七行第 t 列存 slot 四千二百四十二，指向逐层 K 与 V buffer 的同 slot；基数树节点只持索引切片' }).svg();
}

/* ============================================================
 * 6.1 §4（afterHeading）— RadixAttention 树：match/split/LRU（签名）
 * 原文：R1=[s,a,b,c] R2=[s,a,b,d] R3=[s,a,e]；_split_node 只拷索引切片；
 *       evict LRU 叶优先堆；命题 1 锁保护（沿路径加锁）
 * ============================================================ */
function radixTree() {
  const UID = 'i6d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'RadixAttention 树：match / split / LRU 叶优先驱逐（R1=[s,a,b,c] · R2=[s,a,b,d] · R3=[s,a,e]）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 360, h: 284, label: '基数树（page=1 · value = KV 索引）' }));
  const mk = (id, x, y, title, role = 'cache', dash = false) => {
    const n = node({ id, x, y, w: 70, h: 32, role, title, mono: true, titleSize: 11 });
    nodes.push(n); body.push(n.svg);
    if (dash) body.push(`<rect x="${x}" y="${y}" width="70" height="32" rx="12" fill="none" stroke="var(--ink-faint)" stroke-width="1.2" stroke-dasharray="3 3"/>`);
    return n;
  };
  const rt = mk('rt', 170, 64, 'root', 'plain');
  const s = mk('s', 170, 116, 's');
  const a = mk('a', 170, 168, 'a');
  const b = mk('b', 100, 220, 'b');
  const e = mk('e', 240, 220, 'e');
  const c = mk('c', 60, 272, 'c', 'plain', true);
  const d = mk('d', 140, 272, 'd');
  [[rt, s, 0, false], [s, a, 0.3, false], [a, b, 0.6, false], [a, e, 0, true], [b, c, 0, true], [b, d, 0.9, false]].forEach(([f, t, dl, dash], i) => {
    body.push(edge(f, t, { nodes, portFrom: f.bottom, portTo: t.top, pulse: !dash, dur: 1.1, delay: dl, dash, uid: UID, eid: `e${i}` }));
    if (!dash) body.push(arrowHead(t.top[0], t.top[1], 0, 1));
  });
  body.push(cap(246, 134, 'lock_ref=1', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(246, 186, 'lock_ref=1', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(176, 238, 'lock_ref=1', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(95, 316, 'evicted · LRU 最旧', { size: 7.5, anchor: 'middle' }));
  body.push(cap(175, 316, 'lock_ref=1（R2）', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 420, y: 40, w: 264, h: 210, label: 'evict（L559）：LRU 叶优先堆' }));
  body.push(cap(436, 76, '① evictable_leaves 入堆', { size: 8.5 }));
  body.push(cap(436, 92, '   key = last_access_time', { mono: true, size: 8 }));
  body.push(cap(436, 110, '② heappop 最旧叶 c → free_segment', { size: 8.5 }));
  body.push(cap(436, 128, '③ _delete_leaf(c)：索引归池、留拓扑', { size: 8.5 }));
  body.push(cap(436, 146, '④ 父 b 仍有孩子 d → 非叶不入堆', { size: 8.5 }));
  body.push(cap(436, 164, '⑤ 父变叶且 lock_ref=0 → 入堆候选', { size: 8.5 }));
  body.push(cap(436, 190, '命题 1：lock_ref>0 不可驱逐 ——', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(436, 206, 'inc_lock_ref 沿路径锁到 root', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(436, 226, '反例：只锁叶不锁路径 → 祖先成可驱逐叶，', { size: 8 }));
  body.push(cap(436, 238, 'running 请求 KV 索引悬空', { size: 8 }));

  body.push(zone({ x: 36, y: 330, w: 648, h: 84, label: 'match_prefix（L369）与 _split_node（L684）' }));
  body.push(cap(52, 366, 'R2 到达：匹配 s,a,b 后在 c/d 分歧 → 分裂 b 节点：new_node.value = child.value[:split].clone()（只拷索引切片，KV 本体不动）', { size: 9 }));
  body.push(cap(52, 384, 'R3 到达：device_prefix_len = 2（s,a）= 2.2 式 (2) 的 δ；节省 2Nδ + 2L·d_model·δ²（匹配走 gallop + 二分，C 级切片比较）', { size: 9 }));
  body.push(cap(52, 402, 'insert 归还借用段：free_segment 只还请求自算重复段 [protected, prefix_len)；顺序命中归还 0，并发同前缀竞态归还自算段', { size: 9 }));

  body.push(cap(36, 434, 'RadixKey = token_ids + extra_key（LoRA / 隔离命名空间）+ cache_salt：token 相同而 extra_key 不同永不共享前缀节点（match_prefix docstring）', { size: 9.5 }));
  body.push(cap(36, 452, 'page_aligned 把键长截到 page_size 整数倍：page=1 原样；page>1 的尾部损失同 b 对齐账（2.2 §2）', { size: 9.5 }));
  body.push(cap(36, 470, 'evicted = value is None：索引已归池、节点只剩树拓扑；叶集入集三合取：非 evicted ∧ lock_ref=0 ∧ 所有孩子均已 evicted', { size: 9.5 }));

  return scene(720, 482, body, { label: '基数树：共享前缀 s a 后分裂 b 为 c d 两叶，R3 命中 s a；LRU 堆逐最旧叶 c 驱逐索引归池，lock_ref 沿路径保护 running 请求' }).svg();
}

/* ============================================================
 * 6.2 §2（block 0）— overlap 时序：结果处理延迟一拍（签名时间线）
 * 原文：normal GPU 每步空 c_s+c_p；overlap 用 post(j−1)+sched(j+1) 填满；
 *       零 bubble 条件 launch fwd(j+1) 早于 forward(j)+sample(j) 结束
 * ============================================================ */
function overlapTimeline() {
  const UID = 'i6e';
  const body = [];

  body.push(cap(36, 24, 'overlap 时序：结果处理延迟一拍 —— CPU 的 post(j−1)+sched(j+1) 填进 GPU forward(j) 的空隙（示意，非按比例）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(52, 76, 'normal', { size: 9.5 }));
  [['forward(j−1)+sample', 140], ['forward(j)+sample', 320], ['forward(j+1)+sample', 500]].forEach(([t, x], i) => {
    body.push(cap(x + 75, 58, t, { size: 8, anchor: 'middle' }));
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="64" width="150" height="18" rx="3"/>`);
    body.push(pathPulse(`M ${x} 73 H ${x + 150}`, { dur: 1.4, delay: i * 0.8 }));
  });
  [290, 470].forEach((x) => {
    body.push(`<rect x="${x}" y="64" width="30" height="18" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
    body.push(cap(x + 15, 96, 'idle', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  });
  body.push(cap(52, 116, 'GPU 每步空 c_s+c_p', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(305, 116, 'c_s+c_p', { size: 7.5, anchor: 'middle' }));
  body.push(cap(485, 116, 'c_s+c_p', { size: 7.5, anchor: 'middle' }));

  body.push(cap(52, 166, 'overlap', { size: 9.5 }));
  [['forward(j−1)+sample', 140], ['forward(j)+sample', 320], ['forward(j+1)+sample', 500]].forEach(([t, x]) => {
    body.push(cap(x + 90, 148, t, { size: 8, anchor: 'middle' }));
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="154" width="180" height="18" rx="3"/>`);
  });
  body.push(pathPulse('M 140 163 H 680', { dur: 3, delay: 0.3 }));
  [['post(j−2) + sched(j)', 140], ['post(j−1) + sched(j+1)', 320], ['post(j) + sched(j+2)', 500]].forEach(([t, x], i) => {
    const n = node({ id: `cpu${i}`, x, y: 186, w: 180, h: 22, role: 'norm', title: t, mono: true, titleSize: 8 });
    body.push(n.svg);
  });
  [320, 500].forEach((x, i) => {
    body.push(`<line x1="${x}" y1="150" x2="${x}" y2="212" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
    body.push(cap(x, 224, `launch fwd(j${i === 0 ? '' : '+1'})`, { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  });
  body.push(cap(400, 244, '零 bubble 条件：launch fwd(j+1) 早于 forward(j)+sample(j) 结束 —— c_cpu ≤ t_gpu（命题 1）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(52, 278, '章脊：c_cpu=1 ms · t_gpu=5.13 ms → T_normal=6.13 → T_overlap=5.13，F ≈ 1.195（式 3）', { mono: true, size: 9.5 }));
  body.push(cap(52, 296, 'F = (c_cpu+t_gpu)/max(c_cpu,t_gpu) ≤ 2，c_cpu=t_gpu 取等；CPU 超重（8 ms）F ≈ 1.64 —— 瓶颈回到 CPU，overlap 不是无条件 2×', { size: 9.5 }));
  body.push(cap(52, 314, '零开销 = GPU 零 idle，不是 CPU 零等待：pop_and_process 先 sync copy_done，CPU 被钉到 GPU 节奏（等待 ≈ t_gpu−c_cpu ≈ 4.13 ms）', { size: 9.5 }));
  body.push(cap(52, 332, 'grammar 例外：sample 延迟 launch（pop_and_process 之后）；prefill 提前处理闸门保 TTFT —— 零开销是默认路径的论断', { size: 9.5 }));

  return scene(720, 346, body, { label: 'overlap 时序对照：normal 循环 GPU 每两步之间空一段 CPU 时间，overlap 循环把上一批后处理与下一批调度填进 GPU 前向期间，GPU 连续零空闲' }).svg();
}

/* ============================================================
 * 6.2 §3（afterHeading）— FutureMap stash/publish/resolve（签名）
 * 原文：always-on pool-indexed relay；buf[5] 两拍写读；同流提交序；
 *       CPU 调度只消费数据无关量；slot 0 padding 镜像
 * ============================================================ */
function futureMapRelay() {
  const UID = 'i6f';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'FutureMap：always-on pool-indexed relay —— token 值 GPU→GPU 中继，CPU 调度全程不接触 token 值', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const f1 = mk('f1', 52, 60, 170, 44, 'proj', 'forward(j−1)', '读 buf[5] = t_{j−2}');
  const s1 = mk('s1', 52, 124, 170, 44, 'attn', 'sample(j−1)', '产出 t_{j−1}');
  const st = mk('st', 52, 196, 170, 40, 'op', 'stash scatter', 'buf[indices] = bonus_tokens', 9.5, 7);
  [[f1, s1, 0], [s1, st, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `l${i}` }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(cap(137, 252, '写端：一次 scatter · 无 host 往返', { size: 8, anchor: 'middle' }));

  body.push(zone({ x: 250, y: 140, w: 220, h: 130, label: 'FutureMap（overlap_utils L309）' }));
  body.push(cap(360, 176, 'output_tokens_buf · int64', { mono: true, size: 10, anchor: 'middle', weight: '650' }));
  body.push(cap(360, 196, 'slot 0 = padding 镜像（padded 假行）', { size: 8, anchor: 'middle' }));
  body.push(`<rect x="266" y="204" width="188" height="26" rx="6" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(cap(360, 221, '[5] = t_{j−1}', { mono: true, size: 10, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(360, 252, 'new_seq_lens_buf ← publish', { size: 8, anchor: 'middle' }));

  const sc = mk('sc', 490, 60, 194, 44, 'norm', 'CPU sched 批 j', 'seq_lens+1 · 不读 token 值');
  const ga = mk('ga', 490, 124, 194, 44, 'op', 'resolve_forward_inputs', 'decode：buf[req_pool_indices] gather', 9.5, 7);
  const f2 = mk('f2', 490, 196, 194, 44, 'proj', 'forward(j)', 'input_ids = t_{j−1}');
  body.push(edge(sc, ga, { nodes, portFrom: sc.bottom, portTo: ga.top, dash: true, arrow: false, uid: UID, eid: 'm1' }));
  body.push(cap(600, 116, '只喂 metadata', { size: 7.5 }));
  body.push(edge(ga, f2, { nodes, portFrom: ga.bottom, portTo: f2.top, pulse: true, dur: 1.1, delay: 1.6, uid: UID, eid: 'm2' }));
  body.push(arrowHead(f2.top[0], f2.top[1], 0, 1));
  body.push(cap(587, 252, 'publish(seq_lens+1) → new_seq_lens_buf', { size: 8, anchor: 'middle' }));

  body.push(edge(st, [266, 217], { nodes, pulse: true, dur: 1.2, delay: 0.8, uid: UID, eid: 'w' }));
  body.push(arrowHead(266, 217, 1, 0));
  body.push(edge([454, 217], ga, { nodes, portTo: ga.left, pulse: true, dur: 1.2, delay: 1.2, uid: UID, eid: 'r' }));
  body.push(arrowHead(ga.left[0], ga.left[1], 1, 0));
  body.push(cap(472, 120, 'gather', { size: 7.5, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 290, w: 648, h: 70, label: 'resolve 正确性四要件（命题 2）' }));
  body.push(cap(52, 326, '流序：stash(j−1) 与 gather(j) 同在 forward_stream 按提交序执行；wait_stream(schedule_stream) 把 metadata 写定序在 gather 前', { size: 9 }));
  body.push(cap(52, 344, 'WAR barrier：schedule_stream 的写（释放/重分配 KV 行）挡在 forward 共享读后 · slot 0 三处同构 · grammar CPU 顺序', { size: 9 }));

  body.push(cap(36, 380, '数据依赖分类：seq_lens 数据无关（CPU 提前算 +1）· token 值 / accept 长度数据有关（GPU relay / publish event 解析）', { size: 9.5 }));
  body.push(cap(36, 398, 'MIXED 批：prefill 部分 pinned CPU staging H2D + decode 部分 buf gather，cat 成 input_ids —— 两条来源在 forward 入口汇合', { size: 9.5 }));
  body.push(cap(36, 416, '投机 v2：accept 长度走 publish → resolve_seq_lens_cpu，私有 D2H 流 wait publish_ready event —— 不同步 schedule_stream', { size: 9.5 }));

  return scene(720, 430, body, { label: 'FutureMap 中继：上一拍 sample 产出经 stash scatter 写入请求槽位，下一拍 forward 入口 gather 同槽位作输入，CPU 调度只消费序列长等数据无关量' }).svg();
}

/* ============================================================
 * 6.3 §1（block 0）— 同 world 两套划分：attention DP × MoE EP
 * 原文：attn_tp = 8//8//1 = 1；8 独立 attention 组各持 B/8；1 EP 组切 8；
 *       M_rank = κLTB/d：17.16 → 2.14 GiB；blog 1.9×
 * ============================================================ */
function dpDualGroup() {
  const UID = 'i6g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '同一个 world 两套划分（tp_size=8 = world · attn_dp_size=8 · ep_size=8）：attention 逐 rank 独立、MoE 共享 EP 组', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(360, 56, 'attention 族：8 个独立组（attn_tp_size = 8//8//1 = 1），各持 B/8 请求', { size: 9, anchor: 'middle' }));
  const ranks = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
    const x = 52 + i * 80;
    body.push(`<rect x="${x - 3}" y="64" width="78" height="56" rx="10" fill="none" stroke="rgb(24 27 32 / 0.28)" stroke-width="1.1" stroke-dasharray="5 6"/>`);
    const n = node({ id: `r${i}`, x, y: 70, w: 72, h: 44, role: 'attn', title: `r${i}`, sub: 'B/8 请求', mono: true, titleSize: 11, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  body.push(`<rect x="48" y="130" width="652" height="40" rx="12" fill="rgb(168 50 40 / 0.03)" stroke="rgb(168 50 40 / 0.4)" stroke-width="1.1" stroke-dasharray="5 6"/>`);
  body.push(cap(360, 154, 'MoE 族：1 个 EP 组铺满 8 rank —— expert 切 8 份（ep_size=8）', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  ranks.forEach((r, i) => {
    body.push(edge(r, [r.top[0], 130], { nodes, portFrom: r.bottom, pulse: true, dur: 1, delay: 0.2 + i * 0.12, uid: UID, eid: `d${i}` }));
    body.push(arrowHead(r.top[0], 130, 0, 1));
  });

  const mk = (id, x, w, role, title, sub) => {
    const n = node({ id, x, y: 196, w, h: 44, role, title, sub, titleSize: 10.5, subSize: 7.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const ao = mk('ao', 52, 140, 'data', 'attention 输出', '逐 DP rank 本地');
  const di = mk('di', 230, 150, 'op', 'dispatch', 'EP all-to-all（3.3）');
  const mo = mk('mo', 418, 120, 'proj', 'MoE 计算', 'expert 本地');
  const co = mk('co', 576, 108, 'op', 'combine', '回各 DP rank');
  [[ao, di, 1.2], [di, mo, 1.5], [mo, co, 1.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(cap(52, 278, 'M_rank = κL·T·B/d（式 3，MLA κL=70272 B · T=4096 · B=64）：', { mono: true, size: 9.5 }));
  body.push(cap(52, 301, 'd=1', { mono: true, size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="110" y="290" width="515" height="14" rx="3"/>`);
  body.push(cap(632, 301, '17.16 GiB', { mono: true, size: 8.5 }));
  body.push(cap(52, 327, 'd=8', { mono: true, size: 9 }));
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="110" y="316" width="64" height="14" rx="3"/></g>`);
  body.push(cap(182, 327, '2.14 GiB/rank —— 同全局 batch 驻留降 1/8，腾出 HBM 换更大 batch（blog 1.9× decode 吞吐）', { size: 9, color: 'var(--seal-deep)' }));

  body.push(cap(36, 352, '派生：attn_tp_size = tp_size // attn_dp_size // attn_cp_size（式 1）；rank 布局 (dp, cp, tp) 混合进制双射（命题 1）', { size: 9.5 }));
  body.push(cap(36, 370, 'tp_size 是 world size 不是 attention TP 度；enable_dp_attention 是派生报告值（attn_dp_size>1）而非字段 —— blog flag 历史口径', { size: 9.5 }));
  body.push(cap(36, 388, 'IDLE 批：无序列 rank 陪跑 MoE 前后 collective（all-gather / redistribute）—— idle 是 DP 组同步的占位，不是空转浪费', { size: 9.5 }));

  return scene(720, 400, body, { label: 'DP attention 双族划分：八个 rank 各成一个 attention 组持八分之一请求，MoE 侧一个 EP 组铺满全 world，dispatch 与 combine 走 all-to-all，每 rank KV 驻留降八倍' }).svg();
}

/* ============================================================
 * 6.3 §4（block 1）— PD bootstrap：控制面撮合 + RDMA 直写（签名）
 * 原文：bootstrap 纯控制面无 I/O；decode 预分配 + send_metadata 目的索引；
 *       chunk RDMA 直写；KVPoll 五态；PREBUILT 不进 forward
 * ============================================================ */
function pdBootstrap() {
  const UID = 'i6h';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'PD bootstrap：纯控制面 room 撮合 + KV 数据面点对点 RDMA 直写（decode 指定位置，prefill 直写）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, w, role, title, sub) => {
    const n = node({ id, x, y: 44, w, h: 44, role, title, sub, titleSize: 10.5, subSize: 7 });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  mk('pf', 52, 150, 'proj', 'prefill 实例', 'sender（BaseKVSender L191）');
  mk('bs', 285, 150, 'norm', 'bootstrap server', '纯控制面 · 无 I/O');
  mk('dc', 518, 166, 'attn', 'decode 实例', 'receiver（BaseKVReceiver L266）');
  body.push(`<line x1="262" y1="96" x2="262" y2="340" stroke="var(--rule)" stroke-width="1" stroke-dasharray="4 5"/>`);
  body.push(`<line x1="496" y1="96" x2="496" y2="340" stroke="var(--rule)" stroke-width="1" stroke-dasharray="4 5"/>`);

  const msg = (x1, x2, y, lab, seal = false, dl = 0) => {
    const d = `M ${x1} ${y} H ${x2}`;
    body.push(`<path class="${seal ? 'dia-edge strong' : 'dia-edge'}" d="${d}"${seal ? ' style="stroke:var(--seal);stroke-width:2"' : ''}/>`);
    body.push(pathPulse(d, { dur: 1.4, delay: dl }));
    body.push(arrowHead(x2, y, x2 > x1 ? 1 : -1, 0));
    body.push(cap((x1 + x2) / 2, y - 6, lab, { size: 8, anchor: 'middle', color: seal ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
  };
  msg(202, 285, 100, '① register(room)', false, 0);
  msg(518, 435, 124, '② query(room)', false, 0.3);
  msg(435, 518, 148, '③ 撮合: room→端点', false, 0.6);
  body.push(cap(601, 172, '④ prealloc KV 槽 + metadata buf', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  msg(518, 202, 196, '⑤ send_metadata（目的索引 kv_indices）', false, 0.9);
  msg(202, 518, 228, '⑥ chunk 0 forward → send_kvcache ═ RDMA 直写', true, 1.2);
  msg(202, 518, 256, '⑦ chunk 1 …（chunk 流水，下图）', true, 1.6);
  msg(202, 518, 284, '⑧ last chunk + aux（首输出 token 元数据）', true, 2.0);
  body.push(cap(601, 308, '⑨ poll: Transferring → Success', { size: 8, anchor: 'middle' }));
  body.push(cap(127, 308, 'poll Success → 释放 KV', { size: 8, anchor: 'middle' }));
  body.push(cap(601, 330, '⑩ PREBUILT → DECODE（不进 forward）', { size: 8, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(360, 352, 'KV 数据面不经 bootstrap server：点对点 RDMA write 到预分配的目的槽位', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(52, 372, 'KVPoll 五态（base/conn.py L142）：', { size: 9 }));
  [['Failed', 52, 70], ['Bootstrapping', 130, 104], ['WaitingForInput', 242, 116], ['Transferring', 366, 96], ['Success', 470, 70]].forEach(([t, x, w], i) => {
    const n = node({ id: `kp${i}`, x, y: 380, w, h: 26, role: i === 4 ? 'norm' : i === 0 ? 'plain' : 'op', title: t, mono: true, titleSize: 8 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(684, 396, 'Failed 任意态可达（abort/超时）', { size: 8, anchor: 'end' }));

  body.push(cap(36, 424, '准入不变式：inflight + allocatable ≥ 每条 inflight 请求最大 input+output —— transfer queue 已持 KV 页不可 retract（违反 = 死锁式 OOM）', { size: 9.5 }));
  body.push(cap(36, 442, '第一个输出 token 已在 prefill 侧采样，id / logprobs 经 aux 通道传来；decode 第一次真实 forward 已是第二个 token 的 DECODE 步', { size: 9.5 }));
  body.push(cap(36, 460, 'publish_pending_metadata：对 ready 标志 all-reduce MIN —— 全 rank 一起发布或都不发布，abort 不留半等待（decode.py L3614）', { size: 9.5 }));

  return scene(720, 472, body, { label: 'PD bootstrap 时序：sender 注册 room、receiver 查询撮合，decode 预分配槽位并发目的索引，prefill 逐 chunk RDMA 直写，poll 成功后 prefill 释放 KV、decode 以 PREBUILT 进 DECODE' }).svg();
}

/* ============================================================
 * 6.3 §4（block 2）— chunk 流水：传输藏进 forward
 * 原文：4 chunk × 549 MiB = 2.14 GiB；11.5 ms/chunk 藏进下一 chunk forward；
 *       命题 2 telescoping 恰好覆盖；早发 WAR event；页对齐推迟
 * ============================================================ */
function chunkPipeline() {
  const UID = 'i6i';
  const body = [];

  body.push(cap(36, 24, 'chunk 流水：传输藏进下一 chunk 的 forward（T=32768 · c_chunk=8192 → 4 chunk × 549 MiB）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(52, 77, 'prefill SM', { size: 9 }));
  body.push(cap(52, 107, 'RDMA NIC', { size: 9 }));
  body.push(cap(52, 143, 'decode 侧', { size: 9 }));
  for (let i = 0; i < 4; i++) {
    const x = 140 + i * 120;
    body.push(cap(x + 55, 58, `chunk ${i} fwd`, { size: 8, anchor: 'middle' }));
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="64" width="110" height="18" rx="3"/>`);
    body.push(cap(x + 110, 124, `chunk ${i} KV`, { size: 8, anchor: 'middle' }));
    body.push(`<g class="sig-bar" style="--i:${(0.2 + i * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-warm" x="${x + 55}" y="94" width="110" height="18" rx="3"/></g>`);
    body.push(`<line x1="${x + 55}" y1="60" x2="${x + 55}" y2="116" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 3"/>`);
  }
  body.push(pathPulse('M 140 73 H 610', { dur: 2.6, delay: 0 }));
  body.push(pathPulse('M 195 103 H 665', { dur: 2.6, delay: 0.4 }));
  body.push(`<rect class="dia-bar dia-bar-seal" x="140" y="134" width="40" height="14" rx="3"/>`);
  body.push(cap(160, 130, 'prealloc', { size: 7.5, anchor: 'middle' }));
  body.push(`<path class="dia-edge dash" d="M 180 141 H 610"/>`);
  body.push(arrowHead(610, 141, 1, 0));
  body.push(cap(620, 145, 'Success → DECODE', { size: 8 }));
  body.push(cap(400, 168, '每 chunk 549 MiB @50 GB/s = 11.5 ms 藏进下一 chunk forward；暴露 = 末 chunk + aux + poll 间隔', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(cap(36, 196, '命题 2（telescoping）：每次发送从上一次终点起（start_send_idx = end_idx），页对齐只推迟不跳段 → 恰好覆盖 [T_hit, T)', { size: 9.5 }));
  body.push(cap(36, 214, '总字节 = κL(T−T_hit) 与 chunk 数无关：4 × 575668224 B = 2302672896 B ≈ 2.14 GiB（与整段一次发送恒等）', { mono: true, size: 9.5 }));
  body.push(cap(36, 232, 'T_hit=8192（decode radix 命中）：发送 24576 token × 70272 B（V3）≈ 1.61 GiB → 34.5 ms @50 GB/s / 276 ms @6.25 GB/s（每 GPU 份额）', { size: 9.5 }));
  body.push(cap(36, 250, '早发 WAR 陷阱：overlap 下上一步 forward 可能还在写前缀页 —— forward_stream record event，transfer worker 等 event 再 RDMA read', { size: 9.5 }));
  body.push(cap(36, 268, '页粒度：b=64 → 512 页 × 61 层（V3）= 31232 描述符（细粒度拷贝坑）；mooncake group_concurrent_contiguous 连续页聚成大块', { size: 9.5 }));
  body.push(cap(36, 286, '正确性兜底：末 chunk checksum 覆盖发送范围；P/D TP 宽度不一致 → should_verify_kv_checksum=False（跳过而非误报）', { size: 9.5 }));

  return scene(720, 300, body, { label: 'chunk 流水时序：prefill 逐 chunk 前向，RDMA 网卡错一拍发送各 chunk KV，传输藏进下一 chunk 计算，decode 预分配后 poll 成功进 DECODE' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '06-sglang/01-sglang-architecture',
    replacesBlock: 0,
    title: '前端 DSL × co-designed 运行时：完整 prompt 契约',
    svg: frontendRuntime,
  },
  {
    dim: 'inference',
    slug: '06-sglang/01-sglang-architecture',
    replacesBlock: 1,
    title: '三进程链数据流：ZMQ + msgspec × 同进程执行体',
    svg: threeProcessChain,
  },
  {
    dim: 'inference',
    slug: '06-sglang/01-sglang-architecture',
    replacesBlock: 2,
    title: '两级池：req_to_token 索引 → token_to_kv 本体',
    svg: twoLevelPool,
  },
  {
    dim: 'inference',
    slug: '06-sglang/01-sglang-architecture',
    afterHeading: /^## 4\. /,
    title: 'RadixAttention 树：match / split / LRU 叶优先 × 锁保护',
    svg: radixTree,
  },
  {
    dim: 'inference',
    slug: '06-sglang/02-sglang-overlap-scheduler',
    replacesBlock: 0,
    title: 'overlap 时序：post(j−1)+sched(j+1) 填进 forward(j)',
    svg: overlapTimeline,
  },
  {
    dim: 'inference',
    slug: '06-sglang/02-sglang-overlap-scheduler',
    afterHeading: /^## 3\. /,
    title: 'FutureMap：stash / publish / resolve 的 GPU 流中继',
    svg: futureMapRelay,
  },
  {
    dim: 'inference',
    slug: '06-sglang/03-sglang-distributed-pd',
    replacesBlock: 0,
    title: '同 world 双族划分：attention DP × MoE EP',
    svg: dpDualGroup,
  },
  {
    dim: 'inference',
    slug: '06-sglang/03-sglang-distributed-pd',
    replacesBlock: 1,
    title: 'PD bootstrap：room 撮合（控制面）× RDMA 直写（数据面）',
    svg: pdBootstrap,
  },
  {
    dim: 'inference',
    slug: '06-sglang/03-sglang-distributed-pd',
    replacesBlock: 2,
    title: 'chunk 流水：传输藏进 forward × telescoping 恰好覆盖',
    svg: chunkPipeline,
  },
];