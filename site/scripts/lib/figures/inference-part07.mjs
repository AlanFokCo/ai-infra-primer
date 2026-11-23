// inference Part 7（07-deepinfer-and-deepseek）场景图：替换各章无语言围栏 ASCII 块 + 签名插入
// 规范见 STYLE.md；组件库见 figure-kit.mjs；风格基准 inference-part05.mjs / inference-part06.mjs。
// 脉冲渐变：页面级 PULSE_GRAD_DEFS 固定 id；pulse delay < 3.5s；sig-cell --i ≤ 2.6。
// 块序号 = 该章第 N 个无语言围栏（0 计，2026-11-23 census）；带语言伪代码围栏不画（管线自动算法面板）。
// 数值口径：V4.1 MLA = 512 元素（448 latent + 64 rope，X-1 errata 裁定），NOT 576（V3 几何）；
//           StateBlock state_size_to_disk = 1,415,040 B 仅 stored region；moe_lb 1.074/1.045/1.090。

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

// release 外组件的朱砂虚线描边（叠加在节点卡上）
const sealDash = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="none" stroke="var(--seal)" stroke-width="1.3" stroke-dasharray="4 4"/>`;

/* ============================================================
 * 7.1 §2（block 0）— 完整 serving 系统 × release 边界（签名）
 * 原文：doc L39 架构图重绘；Entrypoint CHECK 匹配 → 三 DP LB → VE/PE/CE；
 *       3FS kvcache-store；命题 1 实例边界；EPLB 跨 rank 不跨实例；~80% offline
 * ============================================================ */
function servingBoundary() {
  const UID = 'i7a';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '完整 serving 系统（doc L39 架构图重绘，官方披露）：朱砂虚线 = release 外的跨实例编排层（仅 doc 披露）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10.5, ss = 7.5) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const rq = mk('rq', 52, 40, 88, 44, 'data', '请求', null, 11);
  const rc = mk('rc', 176, 40, 150, 44, 'norm', 'Recipe 协议层', '多协议 → 统一对话（Rust）', 10.5, 7.5);
  const ep = mk('ep', 362, 40, 210, 48, 'op', 'Entrypoint', 'CHECK hash-chain 前缀匹配 → 命中长度 → 路由', 10.5, 7);
  body.push(sealDash(362, 40, 210, 48));
  body.push(edge(rq, rc, { nodes, pulse: true, dur: 1.1, delay: 0, uid: UID, eid: 'f0' }));
  body.push(arrowHead(rc.left[0], rc.left[1], 1, 0));
  body.push(edge(rc, ep, { nodes, pulse: true, dur: 1.1, delay: 0.3, uid: UID, eid: 'f1' }));
  body.push(arrowHead(ep.left[0], ep.left[1], 1, 0));
  body.push(cap(251, 98, '另开源 deepseek-recipe ≠ 仅 doc', { size: 7.5, anchor: 'middle' }));
  body.push(cap(684, 104, 'in-memory directory（跨机 SWA 定位）', { size: 7.5, anchor: 'end' }));
  body.push(cap(580, 58, 'release 外', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(580, 72, 'doc L191 / L223', { size: 7.5 }));

  body.push(zone({ x: 36, y: 112, w: 648, h: 90, label: 'release 外', tone: 'seal' }));
  body.push(cap(684, 126, '跨实例编排：决策范围跨引擎实例', { size: 8, anchor: 'end' }));
  const vl = mk('vl', 52, 142, 150, 48, 'op', 'VE LB', 'machine 随机 + fewest patches', 10.5, 7.5);
  const pl = mk('pl', 230, 142, 170, 48, 'op', 'Prefill LB', '4 长度档 + compute-quota pull', 10.5, 7.5);
  const dl = mk('dl', 428, 142, 170, 48, 'op', 'Decode LB', 'seq count + total KV length 双维', 10.5, 7.5);
  [vl, pl, dl].forEach((n) => body.push(sealDash(n.x, n.y, n.w, n.h)));
  body.push(`<rect x="612" y="150" width="20" height="14" rx="4" fill="none" stroke="var(--seal)" stroke-width="1.3" stroke-dasharray="4 4"/>`);
  body.push(cap(640, 161, '仅 doc', { size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(612, 180, 'L151 / L137-143', { size: 7 }));
  body.push(cap(612, 192, '/ L147 · L161', { size: 7 }));
  [[vl, 0.6], [pl, 0.8], [dl, 1.0]].forEach(([n, dlx], i) => {
    body.push(edge(ep, n, { nodes, portFrom: ep.bottom, portTo: n.top, pulse: true, dur: 1.2, delay: dlx, uid: UID, eid: `b${i}` }));
    body.push(arrowHead(n.top[0], n.top[1], 0, 1));
  });

  const ve = mk('ve', 52, 232, 150, 56, 'attn', 'VE ×m（引擎侧）', '1 GPU · MultimodalProcessor L29', 10.5, 7);
  const pe = mk('pe', 230, 232, 160, 56, 'proj', 'PE ×n（PROMPT=1）', 'EP2–8 / DP2–8 · 只做 prefill', 10, 7.5);
  const ce = mk('ce', 440, 232, 160, 56, 'attn', 'CE ×n（COMPLETION=2）', 'EP64–128 / DP64–128 · 只 decode', 9.5, 7);
  [[vl, ve, 1.2], [pl, pe, 1.4], [dl, ce, 1.6]].forEach(([a, b, dlx], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.1, delay: dlx, uid: UID, eid: `e${i}` }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(edge(pe, ce, { nodes, pulse: true, dur: 1.1, delay: 1.9, uid: UID, eid: 'kv' }));
  body.push(arrowHead(ce.left[0], ce.left[1], 1, 0));
  body.push(cap(415, 252, 'KV 传输', { size: 7.5, anchor: 'middle' }));
  body.push(cap(360, 386, 'release 内：单引擎实例（含实例内 EP/DP 通信域）—— engine_type 角色钩子 / StateBlock / EPLB / DSpark / SWA replay / SO / batch-invariant', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  const st = mk('st', 230, 326, 368, 48, 'cache', '3FS kvcache-store（专用存储集群）', '任何机命中任何前缀 · 存储编排不在 release（3FS 文件系统本身另开源）', 10.5, 7.5);
  body.push(sealDash(230, 326, 368, 48));
  body.push(edge(pe, [310, 326], { nodes, portFrom: pe.bottom, pulse: true, dur: 1.1, delay: 2.2, uid: UID, eid: 's0' }));
  body.push(arrowHead(310, 326, 0, 1));
  body.push(edge(ce, [520, 326], { nodes, portFrom: ce.bottom, pulse: true, dur: 1.1, delay: 2.4, uid: UID, eid: 's1' }));
  body.push(arrowHead(520, 326, 0, 1));
  body.push(cap(296, 318, '落盘 / 回载', { size: 7.5, anchor: 'end' }));
  body.push(cap(534, 318, '跨机 hit 400G VLAN pull-back', { size: 7.5 }));

  body.push(cap(36, 404, '命题 1（实例边界）：决策范围跨引擎实例的组件 = 编排层（路由 / LB / 跨机存储编排 / 弹性），repo 无实现、仅 doc；实例内运行时组件 release 均有实现', { size: 9.5 }));
  body.push(cap(36, 422, '边界案例：EPLB 跨 rank（EP 组内 all_gather 统计 + 权重 shuffle）但 EP 组 = 实例本身 → 在 release；把边界收紧到「单进程」会错误划出 EPLB 与 DP attention', { size: 9.5 }));
  body.push(cap(36, 440, '代码级验证：全 repo grep compute_quota / quadratic 为空（2026-11-23）—— Prefill LB 二次成本模型（doc L143）与弹性层（L249-265）仅 doc', { size: 9.5 }));
  body.push(cap(36, 458, '缺口代价：在线负载均衡 ≈ 完全均衡离线 benchmark 的 80%（doc L155 官方披露）—— 约两成吞吐差距就是编排层的生产价值（7.4 §7）', { size: 9.5 }));

  return scene(720, 472, body, { label: '完整 serving 系统：请求经 Recipe 到 Entrypoint 前缀匹配路由，三个 DP 负载均衡器分发到 VE、PE、CE 引擎，KV 落盘到 3FS 存储集群；朱砂虚线标出 release 外的跨实例编排层' }).svg();
}

/* ============================================================
 * 7.1 §3（block 1）— deepinfer/ 模块地图：C++ 热路径 × Python 簿记
 * 原文：csrc/ 33 文件 scheduler.hpp 1329 行；_C/ 4 个 .pyi stub（坑）；
 *       LLM L23 offline only；LLMEngine step L860；StateBlock L221；vendor 六 submodule
 * ============================================================ */
function moduleMap() {
  const UID = 'i7b';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '模块地图（main@a679bb86 核实）：C++ 调度热路径在根 csrc/，Python 侧只做簿记与发射 —— _C/ 只是类型 stub', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 7) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono: true, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };

  body.push(zone({ x: 36, y: 40, w: 320, h: 190, label: 'csrc/ —— C++ 热路径（33 文件）' }));
  const sch = mk('sch', 52, 70, 288, 44, 'op', 'engine/scheduler.hpp · 1329 行', 'EngineType L39 / ALL-ANY 投票 L299 / process_MTPInfo L1196', 9.5, 6.8);
  const mv = mk('mv', 52, 122, 140, 40, 'op', 'mtp_varlen.hpp L133', 'DSpark 总预算选择（7.2）', 8.5, 6.8);
  const fe = mk('fe', 200, 122, 140, 40, 'op', 'flops_estimator L28', 'FP8 等效 FLOPs / MFU', 8.5, 6.8);
  body.push(cap(52, 182, 'block_manager.hpp / sequence.hpp / zos.cpp/hpp', { mono: true, size: 7.5 }));
  body.push(cap(52, 196, 'input_metadata/prepare_tensor_input_v4.cpp', { mono: true, size: 7.5 }));
  body.push(cap(52, 210, 'KEEP_TOKENS ≥ window+backtracking 约束 L355', { size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 224, 'pybind.cpp → deepinfer._C；_create_scheduler L307 构造 native', { mono: true, size: 7.5 }));

  body.push(zone({ x: 372, y: 40, w: 312, h: 300, label: 'deepinfer/ —— Python 簿记与发射' }));
  const llm = mk('llm', 388, 66, 280, 40, 'data', 'entrypoints/llm.py · LLM L23', 'offline only · generate L90 / _run_engine L168', 9, 6.8);
  const eng = mk('eng', 388, 112, 280, 40, 'op', 'engine/ · LLMEngine L130', 'step L860 / post_process L984 / engine_type.py', 9, 6.8);
  const wk = mk('wk', 388, 158, 280, 40, 'proj', 'worker/ · Worker L51 / ModelRunner L53', 'cache_engine StateBlock L221 · dflash graph runner', 8.5, 6.8);
  const me = mk('me', 388, 204, 280, 52, 'attn', 'model_executor/ · eplb/ L42', 'models/deepseek.py · v4_modules · mtp/dflash.py DFlash L82', 8.5, 6.8);
  [[llm, eng, 0], [eng, wk, 0.4], [wk, me, 0.8]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, portFrom: a.bottom, portTo: b.top, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `p${i}` }));
    body.push(arrowHead(b.top[0], b.top[1], 0, 1));
  });
  body.push(edge(sch, eng, { nodes, pulse: true, dur: 1.4, delay: 1.2, uid: UID, eid: 'native' }));
  body.push(arrowHead(eng.left[0], eng.left[1], 1, 0));
  body.push(cap(388, 272, 'layers/ Sampler L59 + tilelang_ops/（cuda/ + asc/）', { mono: true, size: 7.5 }));
  body.push(cap(388, 286, 'parallel_utils/ EP·DP·CP 进程组 · input_metadata/', { mono: true, size: 7.5 }));
  body.push(cap(388, 300, 'distributed/shm_broadcast ShmRingBuffer L33（vLLM 同名 lineage）', { size: 7.5 }));
  body.push(cap(388, 314, 'platform.py GPU/NPU · structure_output/ XGrammar · ve/ · npu_ops/', { size: 7.5 }));
  body.push(cap(388, 328, 'hugging_face/config.py HF→engine 映射 · environ.py 59 个 DEEPINFER_*', { size: 7.5 }));

  body.push(zone({ x: 36, y: 246, w: 320, h: 76, label: 'deepinfer/_C/ —— 类型 stub（非 C++ 源码）' }));
  body.push(cap(52, 282, 'engine.pyi / cuda_utils.pyi / graph_timer.pyi / __init__.pyi', { mono: true, size: 7.5 }));
  body.push(cap(52, 298, '坑：按「_C 是 native 模块」直觉找实现会落空 —— 本体在根 csrc/', { size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(52, 312, 'from deepinfer._C.engine import Scheduler（llm_engine.py L19 逐字）', { size: 7.5 }));

  body.push(zone({ x: 36, y: 338, w: 320, h: 100, label: 'vendor/ · deep_cluster/ · examples/' }));
  body.push(cap(52, 372, 'vendor/ 六 submodule：FlashMLA / DeepGEMM / DeepEP /', { size: 8 }));
  body.push(cap(52, 386, 'DeepSelect + DeepGEMM-Ascend / DeepEP-Ascend（7.3 分列）', { size: 8 }));
  body.push(cap(52, 402, 'deep_cluster/ topology bind + hardware/{nvidia,ascend}', { mono: true, size: 7.5 }));
  body.push(cap(52, 418, 'examples/benchmark.py 588 行 · sweep/run_sweep.sh · yoco/ 等价验证', { size: 7.5 }));

  body.push(zone({ x: 372, y: 356, w: 312, h: 82, label: '执行链与 IPC 关键事实' }));
  body.push(cap(388, 388, 'LLM._run_engine → LLMEngine.step → Worker.execute_model_step L89', { size: 7.8 }));
  body.push(cap(388, 402, '→ ModelRunner：execute_main_model L308 / execute_sample L337', { size: 7.8 }));
  body.push(cap(388, 416, 'shm MessageQueue L162：每步高频小消息走共享内存环不走网络', { size: 7.8 }));
  body.push(cap(388, 430, '双平台：get_platform L223 读 DEEPINFER_ACCELERATOR（nvidia/ascend）', { size: 7.8 }));

  body.push(cap(36, 458, '入口唯一：entrypoints/ 只有 llm.py，无 OpenAI 兼容 server —— 在线协议层 = Recipe + Entrypoint（另开源或 release 外，§2 命题 1）', { size: 9.5 }));

  return scene(720, 472, body, { label: 'DeepInfer 模块地图：根 csrc 的 C++ 调度器热路径被 Python 引擎层调用，入口 LLM 离线唯一，worker 与 model_executor 承载执行，_C 目录只是四个类型 stub，vendor 钉六个 kernel submodule' }).svg();
}

/* ============================================================
 * 7.1 §4（block 2）— ZOS 重叠时序（doc L334 口径）
 * 原文：CPU schedule(N+1)/launch/post-process(N−1) 全藏进 GPU forward(N)/sample(N)；
 *       lineage 仿 SGLang/NanoFlow；C++ 压低 CPU 工作本身；update_mtp_runtime 测量闭环
 * ============================================================ */
function zosTimeline() {
  const UID = 'i7c';
  const body = [];

  body.push(cap(36, 24, 'ZOS 重叠时序（doc L334 口径）：CPU 的 schedule(N+1) 与 post-process(N−1) 全部藏进 GPU forward(N)/sample(N) 时间内', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(52, 78, 'GPU', { size: 9.5, weight: '650' }));
  [['forward(N)', 120, 250], ['sample(N)', 370, 110], ['forward(N+1)', 480, 140], ['sample(N+1)', 620, 60]].forEach(([t, x, w]) => {
    body.push(cap(x + w / 2, 58, t, { size: 8, anchor: 'middle', mono: true }));
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="64" width="${w}" height="20" rx="3"/>`);
  });
  body.push(pathPulse('M 120 74 H 680', { dur: 3, delay: 0.2 }));

  body.push(cap(52, 136, 'CPU', { size: 9.5, weight: '650' }));
  const mkc = (id, x, w, t) => {
    const n = node({ id, x, y: 120, w, h: 22, role: 'norm', title: t, mono: true, titleSize: 7.8 });
    body.push(n.svg);
    return n;
  };
  const c1 = mkc('c1', 120, 90, 'schedule(N+1)');
  const c2 = mkc('c2', 216, 96, 'launch_fwd(N+1)');
  const c3 = mkc('c3', 318, 84, 'post(N−1)');
  const c4 = mkc('c4', 408, 72, 'launch_spl');
  [[c1, c2], [c2, c3], [c3, c4]].forEach(([a, b], i) => {
    body.push(`<path class="dia-edge" d="M ${a.x + a.w} 131 H ${b.x}"/>`);
    body.push(arrowHead(b.x, 131, 1, 0));
  });
  body.push(pathPulse('M 120 131 H 480', { dur: 2.2, delay: 0.6 }));
  body.push(cap(488, 135, 'launch_sample(N+1)', { size: 7, mono: true }));

  // launch → GPU 执行 的虚线因果
  body.push(`<path class="dia-edge dash" d="M 264 120 C 264 100, 550 104, 550 84"/>`);
  body.push(arrowHead(550, 84, 0, -1));
  body.push(`<path class="dia-edge dash" d="M 444 120 C 444 102, 650 104, 650 84"/>`);
  body.push(arrowHead(650, 84, 0, -1));
  body.push(`<path d="M 120 150 V 156 H 480 V 150" fill="none" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(cap(300, 172, '全部藏进 GPU forward(N)/sample(N) 时间内 —— GPU 全程无 idle', { size: 9, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(36, 192, '虚线 = launch 因果：launch_fwd(N+1) / launch_sample(N+1) 在 N 步内发射，GPU 到 N 步结束后才执行', { size: 8.5 }));

  body.push(cap(36, 212, 'lineage：SGLang overlap + FutureMap 是同一问题的 Python 侧解（6.2）；idea 自认源自 NanoFlow [A]（arXiv:2408.12757）', { size: 9.5 }));
  body.push(cap(36, 230, 'DeepInfer 增量：把调度热路径本身搬进 C++（scheduler.hpp 1329 行）—— overlap 藏残余 CPU 工作，C++ 压低 CPU 工作本身，两手法叠加', { size: 9.5 }));
  body.push(cap(36, 248, '测量闭环：post_process L984 → update_mtp_runtime L997 用 CUDA event elapsed 喂 MTPVarlen —— DSpark 吞吐曲线在线更新（7.2）', { size: 9.5 }));
  body.push(cap(36, 266, '同步基线（doc L330）：串行 schedule→forward→sample→post 时 GPU 在 post(N) 与 schedule(N+1) 期间空转；ZOS 默认 NORMAL/CE 开、PE 关', { size: 9.5 }));

  return scene(720, 284, body, { label: 'ZOS 重叠时序：GPU 连续执行 forward 与 sample，CPU 的下一步调度、发射与上一步后处理全部藏进当前步 GPU 时间内，GPU 全程无空闲' }).svg();
}

/* ============================================================
 * 7.1 §4（afterHeading）— step() 七段流水线 × 引擎角色（签名）
 * 原文：step L860 七段；worker 四段 docstring；ZOS decode 返回上一步 outputs；
 *       NORMAL/PROMPT/COMPLETION；ALL 70 / ANY 100 投票；flush_inflight_reqs
 * ============================================================ */
function stepPipeline() {
  const UID = 'i7d';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'step() 七段（llm_engine.py L860）：C++ Scheduler 定步组成，Python 只做簿记与发射 —— ZOS decode 返回上一步输出', { size: 12, color: 'var(--ink-soft)' }));

  const stages = [
    ['① schedule', 'C++ 定步组成', 'op'],
    ['② clone/zero', 'states+blocks', 'plain'],
    ['③ post_schedule', '调度后处理', 'plain'],
    ['④ SO 簿记', 'so_status 迁移', 'norm'],
    ['⑤ copy_state', 'decode KV 落盘', 'cache'],
    ['⑥ mm fetch', '多模态取数', 'data'],
    ['⑦ async 发射', '非阻塞 worker', 'proj'],
  ];
  const st = stages.map(([t, s, role], i) => {
    const n = node({ id: `s${i}`, x: 36 + i * 94, y: 48, w: 84, h: 52, role, title: t, sub: s, titleSize: 8.5, subSize: 6.5 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  st.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(st[i - 1], n, { nodes, pulse: true, dur: 0.9, delay: i * 0.25, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });
  body.push(cap(36, 118, '① 传入 block_manager / state_manager / FlopsEstimator / DP 进程组（ZOS 用 cpu_data_parallel_group）；兼释放 finished 组', { size: 8 }));
  body.push(cap(36, 134, '⑤ decode 不做 offload、不用 Hi-Sparse（不受 cache 容量约束），但产出 KV 仍写盘提命中率（doc L183）；⑦ 的 ZOS 语义在 worker 侧', { size: 8 }));

  body.push(zone({ x: 36, y: 148, w: 648, h: 94, label: 'Worker.execute_model_step 四段（worker.py L96-102 docstring 逐字）' }));
  const ph = [['forward 当前步', 'GPU', 'proj'], ['post-process 上一步', '与 forward 重叠', 'norm'], ['sample 当前步', 'GPU', 'attn'], ['post-process 当前步', 'ZOS 推迟到下一步', 'norm']].map(([t, s, role], i) => {
    const n = node({ id: `w${i}`, x: 52 + i * 156, y: 176, w: 140, h: 36, role, title: t, sub: s, titleSize: 9, subSize: 6.8 });
    nodes.push(n); body.push(n.svg);
    return n;
  });
  ph.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(ph[i - 1], n, { nodes, pulse: true, dur: 0.9, delay: 1.0 + i * 0.2, uid: UID, eid: `w${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });
  const loop = 'M 590 212 V 226 H 122 V 212';
  body.push(`<path class="dia-edge strong dash" d="${loop}"/>`);
  body.push(pathPulse(loop, { dur: 2, delay: 1.8 }));
  body.push(arrowHead(122, 212, 0, -1));
  body.push(cap(360, 256, 'ZOS decode：step() 返回上一步 outputs —— 尾部在途请求由 flush_inflight_reqs L977 drain（_run_engine while True + step/flush 三元 + break）', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 36, y: 270, w: 648, h: 88, label: 'engine_type 三角色（engine_type.py ↔ csrc enum 对齐）—— 步型决策与角色耦合' }));
  [['NORMAL = 0（混合）', 'ZOS 默认开 · ALL/ANY 投票定步型', 'plain'], ['PROMPT = 1（PE）', 'ZOS 默认关 · 每步断言 prompt_run L947', 'proj'], ['COMPLETION = 2（CE）', 'ZOS 默认开 · 禁抢占重算 TORCH_CHECK', 'attn']].forEach(([t, s, role], i) => {
    const n = node({ id: `et${i}`, x: 52 + i * 212, y: 298, w: 196, h: 44, role, title: t, sub: s, titleSize: 10, subSize: 7 });
    nodes.push(n); body.push(n.svg);
  });

  body.push(cap(36, 376, 'NORMAL 步型投票（scheduler.hpp L299）：DP 组每步同型 —— ALL 点每 70 decode step（全员有 prefill 活），ANY 点每 100 步（任一有活）', { size: 9.5 }));
  body.push(cap(36, 394, '投票前另有 GPU_CACHE_PREFILL_WATERMARK 水位门；attention DP 下各 rank 步形状必须对齐，否则先完成的 rank 在同步点空等', { size: 9.5 }));
  body.push(cap(36, 412, '这是 #DP = #EP 耦合（doc L33）在调度器的落地（8.1 详展）；输出生命周期：RequestOutput 下一步 step() 自动释放，要保留必须 clone（L181）', { size: 9.5 }));
  body.push(cap(36, 430, 'PE/CE 拆分 = VE/P/D 分离的引擎侧地基：角色在 release、编排不在（命题 1）；CE 禁抢占重算 → stop_preempted_request=true', { size: 9.5 }));

  return scene(720, 444, body, { label: 'step 七段流水线：C++ 调度、状态克隆、调度后处理、结构化输出簿记、decode KV 落盘、多模态取数、非阻塞发射 worker；worker 四段中 ZOS 把后处理推迟到下一步，引擎三角色决定步型与抢占策略' }).svg();
}

/* ============================================================
 * 7.2 §4（block 0）— DSpark 一步数据流（签名）
 * 原文：draft 块并行 + ConfidenceHead sigmoid → r；CPU cumprod → 全局降序 →
 *       逐前缀扫描 best_bsz；GPU remap+cumprod+rank counting → pinned D2H；verify
 *       算例：top-6 = seq0 前 4 + seq1 前 2，seqlens_q=[5,3]，峰值 1.2501 tok/ms
 * ============================================================ */
function dsparkDataflow() {
  const UID = 'i7e';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'DSpark 一步数据流（B 个请求，draft 5）：ConfidenceHead → cumprod → 期望吞吐 argmax → top-m 前缀选择', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 7) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const chain = (arr, delays, eid) => arr.forEach((n, i) => {
    if (i === 0) return;
    body.push(edge(arr[i - 1], n, { nodes, pulse: true, dur: 1, delay: delays[i - 1], uid: UID, eid: `${eid}${i}` }));
    body.push(arrowHead(n.left[0], n.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 40, w: 648, h: 84, label: 'draft 侧 —— DFlash 3 层 nextn（models/mtp/dflash.py L82，引擎实现名 dflash）' }));
  const d1 = mk('d1', 52, 66, 150, 44, 'data', 'golden + 4×noise', 'noise_token_id 128799');
  const d2 = mk('d2', 218, 66, 150, 44, 'proj', '块并行 forward', '一次前向算 5 位置 hidden');
  const d3 = mk('d3', 384, 66, 140, 44, 'attn', '逐 step 采样', 'Markov rank256 修正');
  const d4 = mk('d4', 540, 66, 128, 44, 'norm', 'ConfidenceHead L46', 'sigmoid → r[i,1..5]', 9);
  chain([d1, d2, d3, d4], [0, 0.3, 0.6], 'd');
  body.push(cap(360, 120, 'r = 逐位置独立接受概率估计（尚未 cumprod）；MTPPredictInfo(token_ids, confidence_rates) → mtp_info_pool.store L339', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 136, w: 648, h: 84, label: 'CPU schedule(N+1) —— ZOS 下 confidence(N) 未回：只定总预算（N−2 轮 a）' }));
  const c1 = mk('c1', 52, 162, 130, 44, 'op', 'cumprod r → a', 'process_MTPInfo L1196', 9.5, 6.8);
  const c2 = mk('c2', 198, 162, 130, 44, 'op', '全局降序', '同值 flat index 升序');
  const c3 = mk('c3', 344, 162, 170, 44, 'op', '逐前缀扫描 E[out]/cost', 'MTPVarlen::schedule L150', 9, 6.8);
  const c4 = mk('c4', 530, 162, 138, 44, 'cache', 'best_bsz = total_q', 'CPU 定死 · 不可改', 9.5);
  chain([c1, c2, c3, c4], [0.9, 1.1, 1.3], 'c');
  body.push(cap(360, 216, 'cost(Q) = baseline(g(Q)) + slope×g(Q) + sampler（式 4）；g(Q) = ≥Q 的最小已捕获 graph size；E[out] = B + Σ a（式 2）', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 232, w: 648, h: 84, label: 'GPU 修正（ZOS）—— TileLang kernel 用 confidence(N) 重算 per-request 长度（§5）' }));
  const g1 = mk('g1', 52, 258, 150, 44, 'proj', 'remap gather + cumprod', 'Phase 1 · fresh seq remap=−1', 9, 6.8);
  const g2 = mk('g2', 218, 258, 150, 44, 'attn', 'rank counting top-m', 'Phase 2 · 不物化排序', 9);
  const g3 = mk('g3', 384, 258, 140, 44, 'op', 'seqlens_q = 1 + wins', 'Σ q_i = Q 恒成立', 9);
  const g4 = mk('g4', 540, 258, 128, 44, 'data', 'D2H pinned buffer', 'd2h_evt 定序', 9);
  chain([g1, g2, g3, g4], [1.6, 1.8, 2.0], 'g');
  body.push(cap(360, 312, '消费端读 pinned 前 synchronize(d2h_evt)（worker.py L148）；CPU 值是滞后一拍占位 —— per-request 长度以 GPU 修正后为准', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 328, w: 648, h: 76, label: 'verify —— target 主干（prefill 形态）' }));
  const v1 = mk('v1', 52, 352, 210, 40, 'proj', '每请求打 seqlens_q[i] 个 token', '接受停在第一个拒绝处', 9.5);
  const v2 = mk('v2', 290, 352, 170, 40, 'attn', '接受前缀', 'E[N_out] = B + Σ_选中 a', 10);
  const v3 = mk('v3', 488, 352, 180, 40, 'norm', '产出 N_out token/步', 'golden 位恒验（B 个）', 9.5);
  chain([v1, v2, v3], [2.3, 2.5], 'v');

  // 跨层数据流：r → CPU cumprod；total_q → GPU extra_budget；seqlens_q → verify
  const p1 = 'M 604 110 V 130 H 117 V 136';
  body.push(`<path class="dia-edge strong" d="${p1}"/>`);
  body.push(pathPulse(p1, { dur: 1.6, delay: 0.8 }));
  body.push(arrowHead(117, 136, 0, 1));
  const p2 = 'M 599 206 V 226 H 293 V 232';
  body.push(`<path class="dia-edge strong" d="${p2}"/>`);
  body.push(pathPulse(p2, { dur: 1.6, delay: 1.5 }));
  body.push(arrowHead(293, 232, 0, 1));
  body.push(cap(606, 224, 'total_q', { size: 7, mono: true }));
  const p3 = 'M 604 302 V 322 H 157 V 328';
  body.push(`<path class="dia-edge strong" d="${p3}"/>`);
  body.push(pathPulse(p3, { dur: 1.6, delay: 2.2 }));
  body.push(arrowHead(157, 328, 0, 1));
  body.push(cap(606, 320, 'seqlens_q', { size: 7, mono: true }));

  body.push(zone({ x: 36, y: 416, w: 648, h: 104, label: '算例（教学 B=2，nextn 5）：top-6 = seq0 前 4 + seq1 前 2 → seqlens_q = [5, 3]', tone: 'seal' }));
  const rows = [
    ['seq0 a', 444, ['0.9', '0.765', '0.612', '0.4284', '0.25704'], [1, 2, 4, 5, 0]],
    ['seq1 a', 474, ['0.7', '0.42', '0.21', '0.084', '0.0252'], [3, 6, 0, 0, 0]],
  ];
  rows.forEach(([lab, y, vals, ranks]) => {
    body.push(cap(52, y + 16, lab, { mono: true, size: 8.5 }));
    vals.forEach((v, j) => {
      const x = 110 + j * 100;
      const sel = ranks[j] > 0;
      body.push(`<rect x="${x}" y="${y}" width="92" height="24" rx="5" fill="${sel ? 'var(--seal-wash)' : 'var(--paper-raised)'}" stroke="${sel ? 'var(--seal)' : '#c6c0af'}" stroke-width="${sel ? 1.3 : 0.9}"${sel ? '' : ' stroke-dasharray="2 3"'}/>`);
      body.push(cap(x + 46, y + 10, v, { mono: true, size: 8.5, anchor: 'middle', color: sel ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
      body.push(cap(x + 46, y + 20, sel ? `rank ${ranks[j]}` : '砍', { mono: true, size: 6.5, anchor: 'middle', color: sel ? 'var(--seal-deep)' : 'var(--ink-faint)' }));
    });
  });
  body.push(cap(612, 452, 'best_bsz=8', { mono: true, size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(612, 466, 'E[out]=5.8254', { mono: true, size: 7.5 }));
  body.push(cap(612, 480, '峰值 1.2501', { mono: true, size: 7.5 }));
  body.push(cap(612, 494, 'tok/ms (m=6)', { mono: true, size: 7 }));
  body.push(cap(360, 512, '被砍 = seq0 位 5（0.25704）+ seq1 位 3-5 —— 全是低概率尾部：accepted 几乎不损，成本省一档 graph（4.66 vs 6.42 ms）', { size: 8, anchor: 'middle' }));

  return scene(720, 532, body, { label: 'DSpark 一步数据流四层：draft 侧块并行前向与 ConfidenceHead 产出逐位置概率，CPU 累乘排序扫描定总预算，GPU 用最新置信重算每请求长度写回 pinned，verify 按前缀接受产出；算例 top-6 选中 seq0 前四与 seq1 前二' }).svg();
}

/* ============================================================
 * 7.2 §4（afterHeading）— 期望吞吐扫描：非单调 × argmax 停 graph 边界（签名）
 * 原文：全数字算例表 m=0..10；吞吐峰值 1.2501@m=6；m=2→3 与 6→7 跨 graph
 *       边界吞吐反降；N−2/N−1 配对；cost 三项式 (4)
 * ============================================================ */
function throughputScan() {
  const UID = 'i7f';
  const body = [];

  body.push(cap(36, 24, '期望吞吐扫描（命题 2）：吞吐非单调 —— argmax 必须穷举 m，最优恰停在 graph 边界 8（教学成本曲线，非实测）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 250, h: 120, label: 'cumprod：r → a（式 1）' }));
  body.push(cap(52, 72, 'seq0 r = [0.90, 0.85, 0.80, 0.70, 0.60]', { mono: true, size: 7.5 }));
  body.push(cap(52, 88, '→ a = [0.9, 0.765, 0.612, 0.4284, 0.25704]', { mono: true, size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 112, 'seq1 r = [0.70, 0.60, 0.50, 0.40, 0.30]', { mono: true, size: 7.5 }));
  body.push(cap(52, 128, '→ a = [0.7, 0.42, 0.21, 0.084, 0.0252]', { mono: true, size: 7.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 150, 'a 沿位置单调不增 → top-m 自动前缀封闭（命题 1）', { size: 7.5 }));

  body.push(zone({ x: 302, y: 40, w: 382, h: 120, label: 'cost(Q) = baseline(g(Q)) + slope×g(Q) + sampler（式 4）' }));
  body.push(cap(318, 72, 'graph sizes [2, 4, 8, 16] · baseline [3.0, 3.4, 4.2, 5.8] ms', { mono: true, size: 8 }));
  body.push(cap(318, 90, 'slope 0.02 ms/token · sampler 0.3 ms', { mono: true, size: 8 }));
  body.push(cap(318, 108, 'Q=8: g(8)=8 → cost = 4.2 + 0.16 + 0.3 = 4.66 ms', { mono: true, size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(318, 130, 'g(Q) = 不小于 Q 的最小已捕获 CUDA Graph size（lower_bound）', { size: 8 }));
  body.push(cap(318, 146, '启动 profile_decode_perf L562 + 在线 CUDA event 更新（7.1 §4）', { size: 8 }));

  body.push(zone({ x: 36, y: 176, w: 648, h: 178, label: 'throughput(m) = (B + Σ_top-m a) / cost(B+m) —— 逐 m 扫描 0…10' }));
  const T = [0.5988, 0.7672, 0.9696, 0.9367, 1.0680, 1.1600, 1.2501, 0.9474, 0.9801, 0.9932, 0.9971];
  const Q = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const px = (m) => 80 + m * 56;
  const py = (t) => 316 - (t - 0.5) * 120;
  [0.6, 0.9, 1.2].forEach((t) => {
    body.push(`<line x1="70" y1="${py(t)}" x2="660" y2="${py(t)}" stroke="var(--rule)" stroke-width="0.8" stroke-dasharray="2 4"/>`);
    body.push(cap(64, py(t) + 3, t.toFixed(1), { mono: true, size: 7, anchor: 'end' }));
  });
  body.push(`<line class="dia-edge" x1="70" y1="316" x2="660" y2="316"/>`);
  // graph 边界虚线（m=2→3 跨 4→8；m=6→7 跨 8→16）
  [[220, 'g: 4→8'], [444, 'g: 8→16']].forEach(([x, lab]) => {
    body.push(`<line x1="${x}" y1="212" x2="${x}" y2="316" stroke="var(--seal)" stroke-width="1" stroke-dasharray="3 4"/>`);
    body.push(cap(x, 206, lab, { size: 7, anchor: 'middle', color: 'var(--seal-deep)', mono: true }));
  });
  const pts = T.map((t, m) => `${px(m)},${py(t).toFixed(1)}`).join(' ');
  body.push(`<polyline class="dia-edge strong" points="${pts}"/>`);
  T.forEach((t, m) => {
    const hot = m === 6;
    body.push(`<g class="sig-cell" style="--i:${(m * 0.2).toFixed(1)}"><circle cx="${px(m)}" cy="${py(t).toFixed(1)}" r="${hot ? 5 : 3}" fill="${hot ? 'var(--seal)' : 'var(--ink)'}"${hot ? ' stroke="var(--seal-wash)" stroke-width="2"' : ''}/></g>`);
    body.push(cap(px(m), 330, String(m), { mono: true, size: 7.5, anchor: 'middle' }));
    body.push(cap(px(m), 344, String(Q[m]), { mono: true, size: 6.5, anchor: 'middle' }));
  });
  body.push(cap(668, 330, 'm', { mono: true, size: 8 }));
  body.push(cap(64, 344, 'Q', { mono: true, size: 6.5, anchor: 'end' }));
  body.push(cap(408, 220, '峰值 1.2501（m=6, Q=8）', { size: 8.5, anchor: 'end', color: 'var(--seal-deep)', weight: '650' }));
  body.push(pathPulse(`M ${pts.split(' ').join(' L ')}`, { dur: 3.2, delay: 0.4, r: 3.4 }));

  body.push(cap(36, 372, '结构事实一：m=2→3 跨 graph 边界 4→8，成本 3.78→4.66，吞吐反降（0.9696→0.9367）—— 必须穷举扫描，不能贪心加到边际为负', { size: 9.5 }));
  body.push(cap(36, 390, '结构事实二：m=6→7 跨 8→16，成本 4.66→6.42，吞吐从峰值 1.2501 跌到 0.9474 —— 最优预算恰停 graph 边界 8；被砍全是低概率尾部', { size: 9.5 }));
  body.push(cap(36, 408, 'N−2/N−1 配对：best_bsz 永远用 N−2 轮 a（CPU 定死不可改）；per-request seqlens_q 用 N−1 轮（non-ZOS）或 GPU 以 confidence(N) 重算（ZOS，§5）', { size: 9.5 }));

  return scene(720, 422, body, { label: '期望吞吐随预算 m 的扫描曲线：跨 CUDA Graph 边界处成本跳档吞吐反降，峰值一点二五零一出现在 m 等于六总预算八，最优预算恰停在图边界上' }).svg();
}

/* ============================================================
 * 7.2 §5（block 1）— ZOS 时序：总预算 CPU × per-request 长度 GPU 重算
 * 原文：sched(N+1) 时 confidence(N) 未回 → CPU 只定 total_q（N−2 轮 a）；
 *       GPU TileLang cumprod via remap + rank counting → seqlens_q → pinned D2H（d2h_evt）
 * ============================================================ */
function zosGpuCorrect() {
  const UID = 'i7g';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'ZOS 时序：CPU 只定总预算 total_q（N−2 轮 a）—— per-request 长度由 GPU 用 confidence(N) 重算（示意）', { size: 12, color: 'var(--ink-soft)' }));

  // beat 分隔
  [265, 465].forEach((x) => body.push(`<line x1="${x}" y1="44" x2="${x}" y2="162" stroke="var(--rule)" stroke-width="1" stroke-dasharray="4 5"/>`));
  [['N−1', 164], ['N', 365], ['N+1', 565]].forEach(([t, x]) => body.push(cap(x, 52, `拍 ${t}`, { size: 9, anchor: 'middle', mono: true, weight: '650' })));

  // GPU lane
  body.push(cap(40, 84, 'GPU', { size: 9, anchor: 'end', weight: '650' }));
  [['forward(N−1)+sample(N−1)', 70, 188], ['forward(N)+sample(N)', 272, 186], ['forward(N+1)+sample(N+1)', 472, 186]].forEach(([t, x, w], i) => {
    body.push(`<rect class="dia-bar dia-bar-ink" x="${x}" y="66" width="${w}" height="28" rx="4"/>`);
    body.push(cap(x + w / 2, 84, t, { size: 7.6, anchor: 'middle', color: 'var(--paper-raised)', mono: true }));
  });
  body.push(pathPulse('M 70 80 H 658', { dur: 3, delay: 0.2 }));

  // CPU lane
  body.push(cap(40, 138, 'CPU', { size: 9, anchor: 'end', weight: '650' }));
  const mkc = (id, x, w, t, hot = false) => {
    const n = node({ id, x, y: 122, w, h: 26, role: hot ? 'attn' : 'norm', title: t, mono: true, titleSize: 7.6 });
    nodes.push(n); body.push(n.svg);
    if (hot) body.push(`<rect x="${x}" y="122" width="${w}" height="26" rx="12" fill="none" stroke="var(--seal)" stroke-width="1.4"/>`);
    return n;
  };
  const c0 = mkc('c0', 70, 62, 'sched(N)');
  const c1 = mkc('c1', 138, 76, 'launch fwd(N)');
  const c2 = mkc('c2', 220, 62, 'post(N−1)');
  const c3 = mkc('c3', 288, 70, 'sched(N+1)', true);
  const c4 = mkc('c4', 364, 84, 'launch fwd(N+1)');
  const c5 = mkc('c5', 454, 56, 'post(N)');
  [[c0, c1], [c1, c2], [c2, c3], [c3, c4], [c4, c5]].forEach(([a, b], i) => {
    body.push(`<path class="dia-edge" d="M ${a.x + a.w} 135 H ${b.x}"/>`);
    body.push(arrowHead(b.x, 135, 1, 0));
  });
  body.push(pathPulse('M 70 135 H 510', { dur: 2.6, delay: 0.5 }));
  body.push(cap(365, 160, 'confidence(N) 还在 GPU → CPU 只定 total_q', { size: 7.6, anchor: 'middle', color: 'var(--seal-deep)' }));

  // GPU 重算流
  body.push(zone({ x: 36, y: 182, w: 648, h: 96, label: 'GPU TileLang kernel 重算 per-request 长度（_correct_seqlens_qk L191）', tone: 'seal' }));
  const mk = (id, x, w, role, title, sub, ts = 9, ss = 6.8) => {
    const n = node({ id, x, y: 210, w, h: 44, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const g0 = mk('g0', 52, 120, 'data', 'confidence(N)', '已在 GPU · 未回 CPU');
  const g1 = mk('g1', 196, 150, 'proj', 'Phase 1 cumprod', 'via remap · fresh=−1 全 0');
  const g2 = mk('g2', 370, 150, 'attn', 'Phase 2 rank counting', 'rank<m → wins · 不物化排序');
  const g3 = mk('g3', 544, 124, 'op', 'seqlens_q=1+wins', 'Σ q_i = Q 恒成立');
  [[g0, g1, 0.4], [g1, g2, 0.8], [g2, g3, 1.2]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `k${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  // pinned 回写 + 消费（y=300 行，手绘节点卡）
  body.push(`<g class="dia-node dia-node-cache" data-node="pn"><rect x="300" y="300" width="150" height="44" rx="12"/><text class="dia-title" x="375" y="318" text-anchor="middle" font-size="9.5">pinned host buffer</text><text class="dia-sub dia-mono" x="375" y="334" text-anchor="middle" font-size="7">D2H non_blocking · d2h_evt</text></g>`);
  body.push(`<g class="dia-node dia-node-norm" data-node="cs"><rect x="500" y="300" width="168" height="44" rx="12"/><text class="dia-title" x="584" y="318" text-anchor="middle" font-size="9.5">消费端读 pinned</text><text class="dia-sub dia-mono" x="584" y="334" text-anchor="middle" font-size="7">synchronize(d2h_evt) 后才读</text></g>`);
  const dp = 'M 606 254 V 288 H 375 V 300';
  body.push(`<path class="dia-edge strong" d="${dp}"/>`);
  body.push(pathPulse(dp, { dur: 1.6, delay: 1.6 }));
  body.push(arrowHead(375, 300, 0, 1));
  body.push(edge([450, 322], [500, 322], { nodes, pulse: true, dur: 1.1, delay: 2.0, uid: UID, eid: 'pc' }));
  body.push(arrowHead(500, 322, 1, 0));
  body.push(cap(52, 322, 'CPU 不被 D2H 拖住 —— event 把同步推迟到真正读长度那一刻', { size: 8.5 }));

  body.push(cap(36, 366, '为什么预算用更旧的 N−2：总预算 CPU 定死（决定 graph size 与 KV 分配）不可改，须用保证已返回的概率；per-request 可在 GPU 修正，用最新可见', { size: 9.5 }));
  body.push(cap(36, 384, 'CPU 的 seqlens_q 是滞后一拍占位（坑）：GPU 用 step N 实际 confidence 重写 pinned 源 —— 只有 total_q 是 CPU 定死、GPU 不改', { size: 9.5 }));

  return scene(720, 398, body, { label: 'ZOS 时序：GPU 逐拍前向采样，CPU 提前调度下一步时置信还在 GPU 只定总预算，GPU TileLang kernel 累乘与排名计数重算每请求长度写回 pinned 缓冲，消费端事件同步后才读' }).svg();
}

/* ============================================================
 * 7.2 §9（block 2）— StateBlock slot：stored / dropped 分区 × 落盘账
 * 原文：stored 前 20 层 SWA 每层 134×528=70,752 B，state_size_to_disk=1,415,040 B；
 *       dropped consumer SWA + ca/ica_state；ring 134=128+6，窗口外 +6 吸收 overshoot；
 *       528=512 数据+16 scale；to disk 只取前 state_size_to_disk 字节
 * ============================================================ */
function stateBlockSlot() {
  const UID = 'i7h';
  const body = [];

  body.push(cap(36, 24, 'StateBlock slot（cache_engine.py L221）：需落盘 state 全放前面 stored region，不需落盘的放后面 dropped region', { size: 12, color: 'var(--ink-soft)' }));

  // 主 slot 条：stored | dropped
  const sx = 50, sw = 300, dx = 350, dw = 320, sy = 64, sh = 96;
  body.push(`<rect x="${sx}" y="${sy}" width="${sw}" height="${sh}" rx="10" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.4"/>`);
  body.push(`<rect x="${dx}" y="${sy}" width="${dw}" height="${sh}" rx="10" fill="#f3f1ea" stroke="#c6c0af" stroke-width="1.2" stroke-dasharray="4 4"/>`);
  body.push(cap(sx + sw / 2, sy + 22, 'stored region（落盘）', { size: 11, anchor: 'middle', weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(sx + sw / 2, sy + 42, 'SWA · 前 20 层（stored layers）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(sx + sw / 2, sy + 58, '每层 134 token × 528 B = 70,752 B', { size: 8, anchor: 'middle', mono: true }));
  body.push(cap(sx + sw / 2, sy + 76, 'state_size_to_disk = 20 × 70,752', { size: 8, anchor: 'middle', mono: true }));
  body.push(cap(sx + sw / 2, sy + 90, '= 1,415,040 B ≈ 1.35 MiB', { size: 8.5, anchor: 'middle', mono: true, weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(dx + dw / 2, sy + 22, 'dropped region（不落盘）', { size: 11, anchor: 'middle', weight: '650' }));
  body.push(cap(dx + dw / 2, sy + 42, 'YOCO consumer SWA（后 20 层）', { size: 8.5, anchor: 'middle' }));
  body.push(cap(dx + dw / 2, sy + 58, '+ ca_state / ica_state（C2 压缩边界累加器）', { size: 8, anchor: 'middle' }));
  body.push(cap(dx + dw / 2, sy + 74, 'fp32 [num_state_seqs, ring, 2, dim]', { size: 7.6, anchor: 'middle', mono: true }));
  body.push(cap(dx + dw / 2, sy + 90, '恢复靠 prefill 重算 / 压缩窗 128 对齐无跨边界依赖', { size: 7.6, anchor: 'middle' }));
  body.push(cap(sx, sy - 8, '低地址', { size: 7.5, mono: true }));
  body.push(cap(dx + dw, sy - 8, '高地址', { size: 7.5, anchor: 'end', mono: true }));

  // to disk（只取 stored 前缀）与 PE→CE（两部分）
  const td = 'M 200 160 V 196';
  body.push(`<path class="dia-edge strong" d="${td}" style="stroke:var(--seal);stroke-width:2"/>`);
  body.push(pathPulse(td, { dur: 1.2, delay: 0.3 }));
  body.push(arrowHead(200, 196, 0, 1));
  body.push(`<rect x="90" y="196" width="220" height="34" rx="10" fill="var(--paper-raised)" stroke="var(--seal)" stroke-width="1.2"/>`);
  body.push(cap(200, 210, 'to disk：只取前 state_size_to_disk 字节', { size: 8.5, anchor: 'middle', weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(200, 224, '长度截断即可排除整个 dropped region', { size: 7.5, anchor: 'middle' }));
  const pc = 'M 510 160 V 196';
  body.push(`<path class="dia-edge strong" d="${pc}"/>`);
  body.push(pathPulse(pc, { dur: 1.2, delay: 0.7 }));
  body.push(arrowHead(510, 196, 0, 1));
  body.push(`<rect x="400" y="196" width="220" height="34" rx="10" fill="var(--paper-raised)" stroke="rgb(24 27 32 / 0.3)" stroke-width="1.2"/>`);
  body.push(cap(510, 210, 'PE→CE 传输：两部分都传', { size: 8.5, anchor: 'middle', weight: '650' }));
  body.push(cap(510, 224, 'total_bytes_inference（含 dropped + ring 余量）', { size: 7.5, anchor: 'middle' }));

  // ring 细节条：128 窗 + 6 回溯 = 134 inference，窗口外 +6 overshoot
  body.push(cap(50, 258, 'ring（每层）：134 = 128 窗 + 6 回溯；窗外 +6 吸收 overshoot ≤5', { size: 8.5 }));
  const ry = 268, u = 1.7;
  body.push(`<rect class="dia-bar dia-bar-ink" x="50" y="${ry}" width="${128 * u}" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="${50 + 128 * u + 3}" y="${ry}" width="${6 * u}" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-warm" x="${50 + 134 * u + 6}" y="${ry}" width="${6 * u}" height="18" rx="3"/>`);
  body.push(cap(50 + 64 * u, ry + 13, '128 窗', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(50 + 128 * u + 3 + 3 * u, ry + 32, '6 回溯', { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(50 + 134 * u + 6 + 3 * u, ry + 44, '+6 overshoot', { size: 7, anchor: 'middle' }));
  body.push(cap(50 + 140 * u + 16, ry + 13, '← slot ring', { size: 7.5 }));

  // 528 B/token 构成条
  body.push(cap(400, 258, '528 B/token = 512 数据 + 16 scale', { size: 8.5 }));
  body.push(`<rect class="dia-bar dia-bar-ink" x="400" y="${ry}" width="${512 * 0.42}" height="18" rx="3"/>`);
  body.push(`<rect class="dia-bar dia-bar-seal" x="${400 + 512 * 0.42 + 3}" y="${ry}" width="${16 * 0.42}" height="18" rx="3"/>`);
  body.push(cap(400 + 256 * 0.42, ry + 13, '512 B FP8 数据（448 latent + 64 rope）', { size: 7, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(400 + 512 * 0.42 + 3 + 8 * 0.42, ry + 32, '16 B scale', { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(400, ry + 46, '528 = 512×1 + (512/32)×1（式 9）· per-32 UE8M0', { size: 7.5, mono: true }));

  body.push(cap(36, 330, '仅 stored 落盘（W0 B-6）：1,415,040 = state_size_to_disk 非 slot 总尺寸 —— slot 另含 dropped 与 ring 余量，total_bytes_inference 更大', { size: 9.5 }));
  body.push(cap(36, 348, 'stored layers = 前 20（_is_dropped_swa_layer L113 判 layer ≥ n_prefill_layer=20 为 dropped）；70,752 = 32×2,211 整除（32B 对齐无 pad）', { size: 9.5 }));
  body.push(cap(36, 366, '128 粒度落盘：prefill 末留 1 token 向下取整到 128 倍数，chunked 中途不落只末 chunk 落；decode 跨 128 边界 copy 当前窗进 replica，越界 ≤5 由 ring +6 吸收', { size: 9.5 }));
  body.push(cap(36, 384, 'kvstore ring：extra-state 开启且 compact 140 时 = 134+128+140 = 402 token/层；默认关闭 ring=134。slot 数 = int(64×1.2)+1 = 77', { size: 9.5 }));

  return scene(720, 398, body, { label: 'StateBlock slot 分区：前面 stored region 是前二十层滑窗态共一百四十一万五千零四十字节落盘，后面 dropped region 是消费者滑窗与压缩累加器不落盘；落盘只取 stored 前缀，PE 到 CE 传两部分；ring 一百三十四加六吸收越界' }).svg();
}

/* ============================================================
 * 7.2 §7（afterHeading）— SWA bounded replay：截断窗钉死 128（签名）
 * 原文：keys(i)=[max(s,i−127),i]（式 6）；精确重建依赖锥 20×128=2,560；
 *       encoder replay（block KV 原样复用）/ decoder replay（只供 decode 不入 cache）；
 *       近乎减半 (20T+20w)/(40T)→1/2，T=32768 省 49.8%；compact vs mask
 * ============================================================ */
function swaBoundedReplay() {
  const UID = 'i7i';
  const body = [];

  body.push(cap(36, 24, 'SWA bounded replay：截断注意力窗到重算段内 —— 输入 token 数钉死末 128，与上下文长 T 和层数无关（命题 3）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 210, h: 180, label: '精确重建：依赖锥' }));
  [36, 74, 112, 150, 188].forEach((w, i) => {
    body.push(`<rect class="dia-bar ${i === 0 ? 'dia-bar-ink' : 'dia-bar-soft'}" x="${141 - w / 2}" y="${78 + i * 22}" width="${w}" height="16" rx="3"/>`);
  });
  body.push(cap(141, 72, '第 20 层位置 i', { size: 7.5, anchor: 'middle' }));
  body.push(cap(141, 196, '20×128 = 2,560 token', { size: 8.5, anchor: 'middle', mono: true, color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(141, 210, '每层向过去展宽一个窗（doc L241）', { size: 7.5, anchor: 'middle' }));

  body.push(zone({ x: 262, y: 40, w: 210, h: 180, label: 'bounded replay：段闭合', tone: 'seal' }));
  body.push(`<rect x="282" y="70" width="170" height="104" rx="8" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.3"/>`);
  [104, 118, 132, 146, 160].forEach((y) => body.push(`<line x1="282" y1="${y}" x2="452" y2="${y}" stroke="rgb(168 50 40 / 0.25)" stroke-width="0.9"/>`));
  body.push(cap(367, 90, '重算段 [s, s+128)', { size: 8.5, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(367, 186, '输入恒 128 token', { size: 8.5, anchor: 'middle', color: 'var(--seal-deep)', weight: '650' }));
  body.push(cap(367, 200, '每层 ≤ w² = 16,384 对，与 T 无关', { size: 7.5, anchor: 'middle' }));
  const cut = 'M 236 122 H 282';
  body.push(`<path class="dia-edge strong dash" d="${cut}"/>`);
  body.push(arrowHead(282, 122, 1, 0));
  body.push(cap(259, 114, '截断', { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));

  body.push(zone({ x: 488, y: 40, w: 196, h: 180, label: '截断窗（式 6）' }));
  body.push(cap(504, 72, 'keys(i) = [max(s, i−127), i]', { mono: true, size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(504, 92, '代入 s = 9856：', { size: 8 }));
  body.push(cap(504, 108, 'i=9999 → [9872, 9999]', { mono: true, size: 7.5 }));
  body.push(cap(504, 122, '完整 128 窗', { size: 7.5 }));
  body.push(cap(504, 140, 'i=9856 → [9856, 9856]', { mono: true, size: 7.5 }));
  body.push(cap(504, 154, '段首窗截到 1 个位置 → 近似', { size: 7.5 }));
  body.push(cap(504, 178, '近似是声明过的代价', { size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(504, 192, '（doc 口径 "approximate"）', { size: 7.5 }));

  body.push(zone({ x: 36, y: 236, w: 648, h: 100, label: '两处使用（doc L239 / L241，官方披露）' }));
  const enc = node({ id: 'enc', x: 52, y: 264, w: 300, h: 56, role: 'proj', title: 'encoder replay（L239）', sub: 'SWA miss → 重算命中前缀末 128 + 非缓存 suffix', titleSize: 10, subSize: 7.5 });
  const dec = node({ id: 'dec', x: 380, y: 264, w: 288, h: 56, role: 'attn', title: 'decoder replay（L241）', sub: '每 prefill 只重算 prompt 末 128（consumer 20 层）', titleSize: 10, subSize: 7.5 });
  body.push(enc.svg, dec.svg);
  body.push(cap(202, 330, '命中长只由 block KV 定 · block KV 原样复用（不重算不覆盖）', { size: 7.5, anchor: 'middle' }));
  body.push(cap(524, 330, '结果只供 decode、永不入 prefix cache（128 段短于锥，近似态）', { size: 7.5, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 352, w: 648, h: 64, label: '近乎减半账（式 7）：prefill 线性项计算比' }));
  body.push(cap(52, 384, '(L_enc·T + L_dec·w) / (L·T) → L_enc/L = 1/2（T ≫ w）', { mono: true, size: 9 }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="52" y="392" width="300" height="12" rx="3"/>`);
  body.push(`<g class="sig-bar" style="--i:0.4"><rect class="dia-bar dia-bar-seal" x="52" y="392" width="150.6" height="12" rx="3"/></g>`);
  body.push(cap(362, 402, 'T=32768：657920/1310720 = 0.50195 → 省 49.8%（python 复核）', { size: 8.5, color: 'var(--seal-deep)' }));

  body.push(cap(36, 436, '口径：「与层数无关」= 输入 token 数不随层数涨；总 replay 计算仍随重算层数线性（L_dec×O(w²+w·d)）—— 恒定的是段长 w 与每层成本对 T 独立', { size: 9.5 }));
  body.push(cap(36, 454, 'YOCO 截断实现：compact 真移除截断区 query（生产，README L25）；mask 保留全部 query 只收窄 SWA 可见域（等价验证 / 消融）—— mask 不省计算（坑）', { size: 9.5 }));
  body.push(cap(36, 472, 'T=4096 省 48.4% · T=1048576 → 0.50006；KEEP_TOKENS=140 得 0.50214 结论不变；SWA 注意力项 decoder 每层 T×w 对 → w² 对', { size: 9.5 }));

  return scene(720, 486, body, { label: 'SWA bounded replay：精确重建依赖锥二十层展宽到两千五百六十 token，截断窗把每层闭合在一百二十八段内输入恒定；encoder replay 只重建滑窗态 block KV 原样复用，decoder replay 只供 decode 不入前缀缓存，prefill 计算近乎减半' }).svg();
}

/* ============================================================
 * 7.2 §8（afterHeading）— hit policy：H ≤ T−134 × 回退一块（签名）
 * 原文：T=10000 算例 cap 9866 / H₁ 9856；case A suffix 144 / case B H 9728 suffix 272；
 *       窗覆盖 9999→[9872,9999]、回溯 9993→[9866,9993]；反例 H=9900 缺 34 key；
 *       134=128+6，KEEP_TOKENS 140 双重 enforce
 * ============================================================ */
function hitPolicy() {
  const UID = 'i7j';
  const body = [];

  body.push(cap(36, 24, 'hit policy（命题 4）：H ≤ T−134 —— block KV 定 H₁，SWA 定是否回退一块；suffix ≥ 134 保证回溯 6 的窗无洞', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 160, label: '全数字算例 T=10000：cap = 10000−134 = 9866，block 命中 H₁ = 9856（77×128，≤ cap）' }));
  body.push(`<rect class="dia-bar dia-bar-soft" x="60" y="88" width="170" height="32" rx="6"/>`);
  body.push(cap(145, 108, '0 … 9600（前缀）', { size: 8, anchor: 'middle' }));
  body.push(`<path class="dia-edge dash" d="M 230 104 H 240"/>`);
  body.push(`<line class="dia-edge strong" x1="240" y1="104" x2="660" y2="104"/>`);
  body.push(cap(320, 92, 'block KV 复用边界恒 H₁', { size: 7.5, anchor: 'middle' }));
  body.push(`<line x1="430" y1="96" x2="430" y2="112" stroke="var(--seal)" stroke-width="1.6"/>`);
  body.push(cap(430, 92, 'H₁=9856', { size: 7.5, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  [[240, '9600'], [330, '9728'], [430, '9856'], [470, '9866 cap'], [510, '9900'], [600, '9993'], [630, '9999'], [660, '10000']].forEach(([x, lab]) => {
    body.push(`<line class="dia-edge" x1="${x}" y1="100" x2="${x}" y2="108"/>`);
    body.push(cap(x, 122, lab, { size: 7, anchor: 'middle', mono: true, color: lab.includes('cap') ? 'var(--seal-deep)' : undefined }));
  });
  // case A
  body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-warm" x="330" y="138" width="100" height="12" rx="3"/><rect class="dia-bar dia-bar-seal" x="430" y="138" width="230" height="12" rx="3"/></g>`);
  body.push(cap(545, 134, 'case A：SWA hit → H=9856，suffix=144 ≥ 134', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(380, 160, '存储 SWA [9728, 9856] ⊇ 首 suffix query 窗', { size: 7, anchor: 'middle' }));
  // case B
  body.push(`<g class="sig-bar" style="--i:0.6"><rect x="330" y="172" width="100" height="12" rx="3" fill="var(--seal-wash)" stroke="var(--seal)" stroke-width="1.2"/><rect class="dia-bar dia-bar-seal" x="430" y="172" width="230" height="12" rx="3"/></g>`);
  body.push(cap(495, 168, 'case B：SWA miss → H=9728（回退一块），suffix=272', { size: 7.5, anchor: 'middle' }));
  body.push(cap(380, 196, 'replay 段 128：重建 SWA（block KV 不动）', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 212, w: 648, h: 100, label: '窗覆盖检查（decode 起点 9999，最大回溯 δ=6 → 位置 9993；示意）' }));
  body.push(`<line x1="122" y1="240" x2="122" y2="284" stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>`);
  body.push(cap(110, 262, 'H=9856', { size: 7, anchor: 'end', mono: true }));
  body.push(cap(52, 250, 'p=9999', { size: 8, mono: true }));
  body.push(`<g class="sig-bar" style="--i:0.9"><rect class="dia-bar dia-bar-ink" x="130" y="240" width="300" height="12" rx="3"/></g>`);
  body.push(cap(445, 250, '窗起点 9872 ≥ H=9856 ✓（余量 16）', { size: 8.5 }));
  body.push(cap(52, 278, 'p=9993', { size: 8, mono: true }));
  body.push(`<g class="sig-bar" style="--i:1.1"><rect class="dia-bar dia-bar-ink" x="130" y="268" width="300" height="12" rx="3"/></g>`);
  body.push(cap(445, 278, '窗起点 9866 ≥ H=9856 ✓（余量 10）', { size: 8.5 }));
  body.push(cap(52, 306, '反例 H=9900', { size: 8, mono: true, color: 'var(--seal-deep)' }));
  body.push(`<rect x="130" y="296" width="60" height="12" rx="3" fill="var(--paper-raised)" stroke="var(--seal)" stroke-width="1.2" stroke-dasharray="3 3"/>`);
  body.push(`<rect class="dia-bar dia-bar-ink" x="190" y="296" width="240" height="12" rx="3" opacity="0.45"/>`);
  body.push(`<line x1="190" y1="290" x2="190" y2="308" stroke="var(--seal)" stroke-width="1.4"/>`);
  body.push(cap(190, 288, 'H=9900', { size: 7, anchor: 'middle', mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(445, 306, '窗起点 9866 < 9900 → 缺 34 key（9866–9899），窗洞', { size: 8, color: 'var(--seal-deep)' }));

  body.push(cap(36, 334, '134 = 128 窗 + 6 回溯：δ = max(v, nextn+1) = 6（environ L139）；inference SWA 窗 get_inference_swa_block_size L88 = 134', { size: 9.5 }));
  body.push(cap(36, 352, 'KEEP_TOKENS 双重 enforce：prepare_tensor_input_v4.cpp L355 + model_runner.py L108；README 默认 140 ≥ 134 余量 6', { size: 9.5 }));
  body.push(cap(36, 370, '命题 4：suffix T−H ≥ 134（case A 144 / case B 272）；回溯 δ 内任意位置窗无洞；block KV 复用与存储态逐字一致（bit-exact 另需 batch-invariant 前提）', { size: 9.5 }));
  body.push(cap(36, 388, 'block KV 复用边界恒 H₁ 非 H：SWA miss 时 H=H₁−128 与 H₁ 差的 128 正是 replay 段 —— 重算命中前缀末一块重建 SWA，block KV 仍复用到 H₁', { size: 9.5 }));

  return scene(720, 402, body, { label: 'hit policy 算例：输入长一万命中上限九千八百六十六，block 命中九千八百五十六，滑窗命中则命中长不变否则回退一块；decode 起点与回溯六的窗起点都不小于命中长无洞，反例命中九千九百则缺三十四个 key 成窗洞' }).svg();
}

/* ============================================================
 * 7.3 §1（block 0）— kernel 栈映射：三条计算路径 × 编译底座 × 双平台
 * 原文：attention（DeepGEMM mqa_logits → candidate_blocks → DeepSelect K=512 →
 *       FlashMLA sparse/fused）；MoE（TileKernels routing → DeepEP EPBuffer →
 *       DeepGEMM grouped / Mega MoE）；采样（Sampler → DeepSelect fp32）；TileLang 底座
 * ============================================================ */
function kernelStackMap() {
  const UID = 'i7k';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'kernel 栈映射：V4.1-Flash forward 三条计算路径 × 编译底座 × 双平台（README L23 六件一手链接 + custom TileLang ops）', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 7) => {
    const n = node({ id, x, y, w, h, role, title, sub, mono: true, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const fw = mk('fw', 200, 44, 320, 40, 'data', 'V4.1-Flash forward', 'models/deepseek.py DeepseekForCausalLM L251', 10.5, 7.5);

  body.push(zone({ x: 36, y: 108, w: 214, h: 220, label: 'attention 路径' }));
  const a1 = mk('a1', 52, 136, 182, 44, 'op', 'indexer 打分', 'DeepGEMM fp8_fp4_mqa_logits', 9.5, 6.8);
  const a2 = mk('a2', 52, 192, 182, 44, 'op', '块预选 → 选择', 'candidate_blocks → DeepSelect K=512', 8.5, 6.5);
  const a3 = mk('a3', 52, 248, 182, 56, 'attn', 'FlashMLA sparse prefill/decode', 'FP8/FP4 KV · fused norm-RoPE-attn-RoPE-cast', 8, 6.2);
  body.push(zone({ x: 266, y: 108, w: 214, h: 220, label: 'MoE 路径' }));
  const b1 = mk('b1', 282, 136, 182, 44, 'op', 'routing top-k/scoring', 'TileKernels · TopKGate L40', 9, 7);
  const b2 = mk('b2', 282, 192, 182, 44, 'proj', 'dispatch / combine', 'DeepEP EPBuffer（V2.5）', 9.5, 7);
  const b3 = mk('b3', 282, 248, 182, 56, 'attn', 'expert GEMM / Mega MoE', 'DeepGEMM grouped contiguous/masked + mega', 8, 6.2);
  body.push(zone({ x: 496, y: 108, w: 188, h: 220, label: '采样路径' }));
  const c1 = mk('c1', 512, 136, 156, 44, 'norm', 'Sampler L59', 'layers/sampler.py', 9.5, 7);
  const c2 = mk('c2', 512, 192, 156, 44, 'op', 'DeepSelect topk fp32', 'vocab ~128K · 2–20× torch.topk', 8.5, 6.5);
  const c3 = mk('c3', 512, 248, 156, 56, 'plain', 'fused_sample', 'Ascend 仅 bf16（fp32 仅 CUDA）', 9, 6.8);

  [[a1, a2, 0.4], [a2, a3, 0.7], [b1, b2, 0.5], [b2, b3, 0.8], [c1, c2, 0.6], [c2, c3, 0.9]].forEach(([f, t, dl], i) => {
    body.push(edge(f, t, { nodes, portFrom: f.bottom, portTo: t.top, pulse: true, dur: 1, delay: dl, uid: UID, eid: `v${i}` }));
    body.push(arrowHead(t.top[0], t.top[1], 0, 1));
  });
  [[a1, 0], [b1, 0.2], [c1, 0.4]].forEach(([t, dl], i) => {
    body.push(edge(fw, t, { nodes, portFrom: fw.bottom, portTo: t.top, pulse: true, dur: 1.2, delay: dl, uid: UID, eid: `f${i}` }));
    body.push(arrowHead(t.top[0], t.top[1], 0, 1));
  });
  body.push(cap(143, 318, 'DSA：indices 页地址已 bake · block_table 传 None', { size: 7, anchor: 'middle' }));
  body.push(cap(373, 318, 'Mega MoE：dispatch+linear1/2+SwiGLU+combine', { size: 7, anchor: 'middle' }));
  body.push(cap(590, 318, '有效内存带宽度量（TopK 无 FLOPs）', { size: 7, anchor: 'middle' }));

  body.push(zone({ x: 36, y: 344, w: 648, h: 64, label: '编译底座' }));
  [['TileLang DSL', 52, 140], ['TileKernels 七族', 208, 130], ['DeepJIT 运行时编译', 354, 122], ['tilelang_ops（cuda/+asc/）', 492, 176]].forEach(([t, x, w], i) => {
    const n = node({ id: `base${i}`, x, y: 370, w, h: 28, role: 'plain', title: t, mono: true, titleSize: 8.5 });
    nodes.push(n); body.push(n.svg);
  });
  body.push(cap(360, 424, '全部 kernel 经 DeepJIT 运行时编译（shape 特化 + 磁盘缓存 $HOME/.dj）· TileLang 把调度空间与数据流解耦（3.4 §5）', { size: 8.5, anchor: 'middle' }));

  body.push(cap(36, 446, '硬件：NVIDIA SM100 / SM103 + Ascend 950 —— vendor/ 六 submodule（FlashMLA / DeepGEMM / DeepEP / DeepSelect + 各 Ascend 变体）钉进引擎树', { size: 9.5 }));
  body.push(cap(36, 464, 'Ascend 分列：DeepGEMM-Ascend / DeepEP-Ascend 另立 repo（同 API）；FlashMLA / TileKernels / DeepSelect 的 Ascend 支持在主 repo（双后端选择）', { size: 9.5 }));

  return scene(720, 478, body, { label: 'kernel 栈映射：forward 分 attention、MoE、采样三条路径，attention 走 DeepGEMM 打分、块预选、DeepSelect 与 FlashMLA，MoE 走 TileKernels 路由、DeepEP 收发、DeepGEMM 专家 GEMM 与 Mega MoE，采样走 DeepSelect，底座是 TileLang 与 DeepJIT，硬件双平台' }).svg();
}

/* ============================================================
 * 7.3 §4（afterHeading）— 三格式字节账：528 / 288 / 68（签名）
 * 原文：512 元素 = 448 latent + 64 rope（X-1 errata，NOT 576）；
 *       528=512+16（per-32 E8M0）；288=256+32（per-16 E4M3）；68=64+4（128 元素）；
 *       四方互证；288 仅作 extra_k_cache；packed UE8M0 两层
 * ============================================================ */
function formatBytes() {
  const UID = 'i7l';
  const body = [];

  body.push(cap(36, 24, '三格式字节账（2.3 式 (4)）：每 token 字节 = 数据 + scale —— 512 元素 = 448 latent + 64 rope（X-1 errata：NOT 576）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 92, label: 'V4.1 共享 KV 向量几何（引擎读法为一手权威）' }));
  body.push(cap(52, 78, 'config.py L289：kv_lora_rank = head_dim − qk_rope_head_dim = 512 − 64 = 448（config 无此字段，引擎导出）', { size: 8 }));
  body.push(`<g class="sig-bar" style="--i:0.2"><rect class="dia-bar dia-bar-ink" x="52" y="84" width="403" height="20" rx="3"/><rect class="dia-bar dia-bar-seal" x="458" y="84" width="58" height="20" rx="3"/></g>`);
  body.push(cap(253, 98, 'kv_lora_rank = 448（latent）', { size: 8, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(487, 98, 'rope 64', { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
  body.push(cap(530, 98, '= 512 元素（head_dim 512 含 rope）', { size: 8.5, color: 'var(--seal-deep)' }));
  body.push(cap(52, 120, 'V41_FORMAT.quantized_elements = 512（cache_common.py L109）· 64 RoPE 维全量化、无 bf16 部分', { size: 7.5 }));

  body.push(zone({ x: 36, y: 148, w: 648, h: 150, label: '每 token 字节分解（数据条 + scale 条，比例 0.5 px/B）' }));
  const rows = [
    ['主 KV MXFP8', 180, 256, 8, 'ink', 'seal', '512 B E4M3 数据', '16 B', '= 528 B · 1.03125 B/元素', 'per-32 E8M0 scale ×16 · SWA 落盘格式（doc L223）'],
    ['主 KV per-16 FP4', 222, 128, 16, 'ink', 'warm', '256 B E2M1', '32 B', '= 288 B · 0.5625 B/元素', 'per-16 E4M3 scale ×32 · 类 NVFP4 无 global scale'],
    ['indexer KV MXFP4', 264, 32, 2, 'ink', 'warm', '64 B', '4 B', '= 68 B · 0.53125 B/元素', 'per-32 E8M0 scale ×4 · 128 元素（index_head_dim）'],
  ];
  rows.forEach(([lab, y, dw, sw, dt, st, dlab, slab, total, note], ri) => {
    body.push(cap(52, y + 14, lab, { size: 9, weight: '650' }));
    body.push(`<g class="sig-bar" style="--i:${(0.4 + ri * 0.3).toFixed(1)}"><rect class="dia-bar dia-bar-${dt}" x="170" y="${y}" width="${dw}" height="20" rx="3"/><rect class="dia-bar dia-bar-${st}" x="${170 + dw + 3}" y="${y}" width="${Math.max(sw, 2)}" height="20" rx="2"/></g>`);
    if (dw >= 100) body.push(cap(170 + dw / 2, y + 14, dlab, { size: 7.5, anchor: 'middle', color: 'var(--paper-raised)' }));
    else body.push(cap(170 + dw / 2, y - 6, dlab, { size: 7, anchor: 'middle' }));
    body.push(cap(170 + dw + 3 + sw / 2, y - 6, slab, { size: 7, anchor: 'middle', color: 'var(--seal-deep)' }));
    body.push(cap(470, y + 14, total, { size: 8.5, mono: true, color: 'var(--seal-deep)', weight: '650' }));
    body.push(cap(470, y + 28, note, { size: 7.5 }));
  });

  body.push(cap(36, 318, '四方一手互证：config.py L289 引擎导出 + V41_FORMAT L109（512 元素）+ FlashMLA main README（528 覆盖全 512 维）+ doc L223 官方 528 B/token', { size: 9.5 }));
  body.push(cap(36, 336, 'X-1 errata：旧「576 元素 / 主 KV 324 B」是事实错误（head_dim 512 误当 kv_lora_rank 再加 rope 64，重复计 rope）—— 576 是 V3 几何，非 V4.1 合法口径', { size: 9.5 }));
  body.push(cap(36, 354, '接口口径：FlashMLA 288 格式仅作 extra_k_cache（k_cache 为 528 时）；引擎两种格式都支持（doc L271 列主 KV FP4 选项口径，三源各按其口径读）', { size: 9.5 }));
  body.push(cap(36, 372, 'kernel 落点：KV cast/dequant → TileKernels；GEMM 侧 SM100 消费 packed UE8M0（4 个 pack 进一个 int）—— 存储与计算格式两层', { size: 9.5 }));
  body.push(cap(36, 390, 'routed expert 权重 per-32 ue8m0 MXFP4（doc L275）↔ config expert_dtype fp4 / weight_block_size [32,32]；非 expert 权重 FP8', { size: 9.5 }));

  return scene(720, 404, body, { label: '三格式字节账：共享 KV 向量五百一十二元素等于四百四十八 latent 加六十四 rope；主 KV MXFP8 五百二十八字节、per-16 FP4 二百八十八字节、indexer 六十八字节，各为数据加 scale；五百七十六是 V3 几何误套' }).svg();
}

/* ============================================================
 * 7.4 §1（block 0）— 四个 LB × release 边界：EPLB 在引擎内，三 DP LB 仅 doc
 * 原文：Request→Recipe→Entrypoint；VE LB/Prefill LB/Decode LB → VE/PE/CE；
 *       每引擎内置 EPLB（release 内）；KVCache Storage（release 外）
 * ============================================================ */
function fourLB() {
  const UID = 'i7m';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, '四个 LB × release 边界：Entrypoint / 三 DP LB / kvcache-store 仅 doc；EPLB 内置 P/D 在 release', { size: 12, color: 'var(--ink-soft)' }));

  const mk = (id, x, y, w, h, role, title, sub, ts = 10, ss = 7) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const rq = mk('rq', 52, 48, 90, 40, 'data', 'Request', null, 10.5);
  const rc = mk('rc', 160, 48, 130, 40, 'norm', 'Recipe', '多协议 → 统一对话（§9）', 10, 7);
  const ep = mk('ep', 310, 48, 210, 44, 'op', 'Entrypoint', 'CHECK hash-chain 前缀匹配（release 外）', 10, 7);
  body.push(sealDash(310, 48, 210, 44));
  body.push(edge(rq, rc, { nodes, pulse: true, dur: 1, delay: 0, uid: UID, eid: 'f0' }));
  body.push(arrowHead(rc.left[0], rc.left[1], 1, 0));
  body.push(edge(rc, ep, { nodes, pulse: true, dur: 1, delay: 0.3, uid: UID, eid: 'f1' }));
  body.push(arrowHead(ep.left[0], ep.left[1], 1, 0));
  body.push(cap(560, 64, '另开源', { size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(560, 78, 'deepseek-recipe', { size: 7.5 }));

  const vl = mk('vl', 52, 140, 150, 48, 'op', 'VE LB', 'machine 随机 + fewest patches', 10, 7);
  const pl = mk('pl', 240, 140, 170, 48, 'op', 'Prefill LB', '4 长度档 + compute-quota pull', 10, 7);
  const dl = mk('dl', 450, 140, 190, 48, 'op', 'Decode LB', 'seq count + total KV length 双维', 10, 7);
  [vl, pl, dl].forEach((n) => body.push(sealDash(n.x, n.y, n.w, n.h)));
  [[vl, 0.6, '图像'], [pl, 0.8, 'prefill'], [dl, 1.0, 'decode']].forEach(([n, dlx, lab], i) => {
    body.push(edge(ep, n, { nodes, portFrom: ep.bottom, portTo: n.top, pulse: true, dur: 1.2, delay: dlx, uid: UID, eid: `b${i}` }));
    body.push(arrowHead(n.top[0], n.top[1], 0, 1));
    body.push(cap(n.x + 8, 132, lab, { size: 7.5 }));
  });

  body.push(zone({ x: 36, y: 218, w: 648, h: 92, label: 'release 内：每引擎内置 EPLB（冗余副本 + 统计窗 + 重分配 + symm_mem shuffle，§3/§4）' }));
  const ve = mk('ve', 52, 244, 150, 52, 'attn', 'VE（单 GPU）', '推荐更便宜 GPU（L51）', 10, 7);
  const pe = mk('pe', 240, 244, 170, 52, 'proj', 'Prefill 引擎（PE）', 'EP2–8 / DP2–8（最小 HBM 允许）', 10, 7);
  const ce = mk('ce', 450, 244, 190, 52, 'attn', 'Decode 引擎（CE）', 'EP64–128 Wide-EP（最大 GPU 允许）', 10, 7);
  [[vl, ve, 1.3], [pl, pe, 1.5], [dl, ce, 1.7]].forEach(([a, b, dlx], i) => {
    body.push(edge(a, [b.top[0], 218], { nodes, portFrom: a.bottom, pulse: true, dur: 1.1, delay: dlx, uid: UID, eid: `e${i}` }));
    body.push(arrowHead(b.top[0], 218, 0, 1));
  });

  const st = mk('st', 240, 340, 240, 48, 'cache', 'KVCache Storage', '3FS kvcache-store 专用集群（release 外）', 10, 7);
  body.push(sealDash(240, 340, 240, 48));
  body.push(cap(490, 356, '任何机命中任何前缀', { size: 8 }));
  body.push(cap(490, 370, '消除 prefix-affinity → pull 可行（2.4）', { size: 7.5 }));
  const sp = 'M 325 296 V 320 H 360 V 340';
  body.push(`<path class="dia-edge strong" d="${sp}"/>`);
  body.push(pathPulse(sp, { dur: 1.6, delay: 2.0 }));
  body.push(arrowHead(360, 340, 0, 1));
  body.push(cap(200, 328, 'KV 落盘 / 跨机 fetch', { size: 8, anchor: 'middle' }));

  body.push(cap(36, 410, 'prefill 取最小 EP/DP（doc L57）：每 expert batch 已够大 + 小组改善均衡 + 减 NVLink 流量；decode 取最大：靠 Wide-EP 减每卡权重流量 + 耦合扩 DP 提强度', { size: 9.5 }));
  body.push(cap(36, 428, '#DP = #EP（doc L33，沿 V3/R1 口径）；TP 与 CP 暂不用（L53）；四 LB + KVCache Storage 配套把在线推向离线（~80%，§7）', { size: 9.5 }));

  return scene(720, 442, body, { label: '四个负载均衡器架构：请求经 Recipe 到 Entrypoint 前缀匹配，图像走 VE LB 到单 GPU 视觉编码器，prefill 走 Prefill LB 到小 EP 引擎，decode 走 Decode LB 到 Wide-EP 引擎，每引擎内置 EPLB，KV 落盘到专用存储集群；朱砂虚线标出 release 外组件' }).svg();
}

/* ============================================================
 * 7.4 §2（afterHeading）— Wide-EP 账：每卡权重流量 ∝ 1/e（签名）
 * 原文：W_exp 18.8 MB；EP8 52/卡 977.6 MB 127.0 μs vs EP64 7/卡 131.6 MB 17.1 μs；
 *       流量比 7.43×；B_exp 32 vs 256；每卡计算 B_local·k=1536 不变；τ(e) 拐点
 * ============================================================ */
function wideEpTraffic() {
  const UID = 'i7n';
  const body = [];

  body.push(cap(36, 24, 'Wide-EP 账兑现：每卡 MoE 权重流量 ∝ E/e —— EP8→EP64 降 7.43×，但每卡计算不变（提强度靠耦合扩 DP）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(cap(36, 48, '单 expert 权重：W_exp = 3×5120×2304 = 35,389,440 参数 × 0.53125 B（MXFP4）= 18,800,640 B ≈ 18.8 MB/expert/层', { size: 9, mono: true }));

  // 两配置卡（宽 280，间隙 56 放比率徽章）
  const card = (x, title, sub, experts, traffic, tw, readt, bexp, hot) => {
    body.push(`<rect x="${x}" y="66" width="280" height="190" rx="12" fill="var(--paper-raised)" stroke="${hot ? 'var(--seal)' : 'rgb(24 27 32 / 0.25)'}" stroke-width="${hot ? 1.5 : 1.2}"/>`);
    body.push(cap(x + 16, 90, title, { size: 12, weight: '650', color: hot ? 'var(--seal-deep)' : 'var(--ink)' }));
    body.push(cap(x + 16, 106, sub, { size: 8 }));
    body.push(cap(x + 16, 130, `物理 expert/卡：${experts}`, { size: 9.5, mono: true }));
    const cols = experts > 20 ? 13 : 7;
    const rowsN = Math.ceil(experts / cols);
    for (let i = 0; i < experts; i++) {
      const cx = x + 16 + (i % cols) * 10, cy = 138 + Math.floor(i / cols) * 10;
      body.push(`<rect x="${cx}" y="${cy}" width="8" height="8" rx="2" fill="${hot ? 'var(--seal)' : 'var(--ink)'}" opacity="0.75"/>`);
    }
    const by = 138 + rowsN * 10 + 6;
    body.push(cap(x + 16, by + 10, '每卡每层权重流量', { size: 8 }));
    body.push(`<g class="sig-bar" style="--i:${hot ? 0.6 : 0.2}"><rect class="dia-bar dia-bar-${hot ? 'seal' : 'soft'}" x="${x + 16}" y="${by + 16}" width="${tw}" height="16" rx="3"/></g>`);
    body.push(cap(x + 16 + tw + 8, by + 28, traffic, { size: 9, mono: true, weight: '650', color: hot ? 'var(--seal-deep)' : 'var(--ink)' }));
    body.push(cap(x + 16, by + 46, `读时间下界 ${readt}（B200 7.7 TB/s）`, { size: 8, mono: true }));
    body.push(cap(x + 16, by + 62, `每 expert batch B_exp = ${bexp}`, { size: 8.5, mono: true }));
  };
  card(52, 'EP8 DP8（prefill）', '冗余 4/卡 → 384+32=416，每卡 52', 52, '977.6 MB', 200, '127.0 μs', '32（B=2048）', false);
  card(388, 'EP64 DP64（decode）', '冗余 1/卡 → 384+64=448，每卡 7', 7, '131.6 MB', 27, '17.1 μs', '256（B=16384）', true);

  // 7.43× 比率（两卡间隙）
  body.push(cap(360, 150, '7.43×', { size: 14, anchor: 'middle', weight: '650', color: 'var(--seal-deep)' }));
  body.push(cap(360, 166, '52/7', { size: 8, anchor: 'middle', mono: true }));
  body.push(`<path class="dia-edge strong" d="M 336 176 H 384" style="stroke:var(--seal)"/>`);
  body.push(arrowHead(384, 176, 1, 0));
  body.push(arrowHead(336, 176, -1, 0));

  body.push(zone({ x: 36, y: 262, w: 648, h: 78, label: '关键：Wide-EP 减的是每卡权重流量，不是每卡计算' }));
  body.push(cap(52, 298, '每卡收到分配对数 Bk/e = B_local·k = 256×6 = 1536，两配置相同（耦合扩张下每卡计算不变，3.3 命题 3 的 F 常数前提）', { size: 9 }));
  body.push(cap(52, 316, '提强度靠耦合扩 DP 增全局 B：EP64 B=16384 → B_exp=256 vs EP8 B=2048 → B_exp=32（doc L295「Increasing EP raises arithmetic intensity」）', { size: 9 }));
  body.push(cap(52, 332, '步时 τ(e) = max(A/e, F) + g(e)，拐点 e*=A/F；过拐点后通信/同步主导（3.3 命题 3 recap）', { size: 8.5, mono: true }));

  body.push(cap(36, 356, 'EP>32 递减（doc L293-297）：带宽不再瓶颈后，更大 EP 引入额外通信/同步开销 + 超节点多 job 共享 NVLink 互相干扰 —— 整超节点作单 EP 组把干扰内部化', { size: 9.5 }));
  body.push(cap(36, 374, 'wide_ep 图读近似（非逐字）：~180–200 tok/s/user 处 EP64 ≈ 22.8k vs EP8 ≈ 12.7k tok/s/GPU ≈ 1.8×，EP64 曲线全程压制 EP8（context 131072）', { size: 9.5 }));
  body.push(cap(36, 392, '图副标题 τ=3.25 自有口径（1+3×0.75）≠ 4.4 几何级数 2.734 —— 引图照抄图口径勿混算（坑）；Mega MoE 把 g(e) 重叠进计算、拐点外推但有限', { size: 9.5 }));

  return scene(720, 406, body, { label: 'Wide-EP 流量账：EP8 每卡五十二个 expert 权重流量九百七十七点六兆读时间一百二十七微秒，EP64 每卡七个 expert 一百三十一点六兆十七点一微秒，流量比七点四三倍；但每卡计算一千五百三十六分配对不变，提强度靠耦合扩 DP' }).svg();
}

/* ============================================================
 * 7.4 §3（afterHeading）— EPLB：冗余副本摊热 expert × 统计窗重分配（签名）
 * 原文：last N steps 统计 → every M steps 重分配 → symm_mem shuffle；
 *       命题 1 ℓ=[8,4] c=[1,1] 复制 argmax M 8→4；生产参数 prefill 4/100/1000、
 *       decode 1/1000/10000；moe_lb 1.074/1.045/1.090（残差偏斜）；CV ∝ √(e/E)
 * ============================================================ */
function eplbRebalance() {
  const UID = 'i7o';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'EPLB：热 expert 复制多份摊到不同卡 —— 统计窗 → 重分配 → symm_mem shuffle；moe_lb 是 EPLB 已生效后的残差偏斜', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 80, label: '机制（doc L125 官方披露）× release 引擎（ExpertLoadBalancer L42）' }));
  const mk = (id, x, w, role, title, sub, ts = 9.5, ss = 6.8) => {
    const n = node({ id, x, y: 66, w, h: 40, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const m1 = mk('m1', 52, 180, 'op', '收集 last N steps 负载', '统计窗 ring [window,layers,experts]', 9, 6.5);
  const m2 = mk('m2', 250, 180, 'op', 'every M steps 重分配', '复制热 expert + 放置贪心');
  const m3 = mk('m3', 448, 204, 'proj', 'NVLink symm_mem shuffle', 'intranode 镜像 / internode NCCL P2P', 9, 6.8);
  [[m1, m2, 0], [m2, m3, 0.4]].forEach(([a, b, dl], i) => {
    body.push(edge(a, b, { nodes, pulse: true, dur: 1.1, delay: dl, uid: UID, eid: `m${i}` }));
    body.push(arrowHead(b.left[0], b.left[1], 1, 0));
  });

  body.push(zone({ x: 36, y: 136, w: 320, h: 150, label: '命题 1：一步贪心复制 argmax（ℓ=[8,4]，c=[1,1]）' }));
  const col = (cx, lab, bars, vals, clab, mlab, hot) => {
    body.push(cap(cx, 172, lab, { size: 7.5, anchor: 'middle' }));
    bars.forEach(([w, tone], i) => {
      body.push(`<rect class="dia-bar dia-bar-${tone}" x="${cx - 32}" y="${180 + i * 18}" width="${w}" height="12" rx="3"/>`);
      body.push(cap(cx - 32 + w + 4, 190 + i * 18, vals[i], { size: 7.5, mono: true }));
    });
    body.push(cap(cx, 226, mlab, { size: 8.5, anchor: 'middle', mono: true, weight: '650', color: hot ? 'var(--seal-deep)' : 'var(--ink)' }));
    body.push(cap(cx, 240, clab, { size: 7, anchor: 'middle', mono: true }));
  };
  col(100, '复制前', [[64, 'ink'], [32, 'soft']], ['8', '4'], 'c=[1,1]', 'M=8', false);
  col(205, '复制 j0=argmax', [[32, 'seal'], [32, 'seal']], ['4', '4'], 'c=[2,1]', 'M=4 ✓', true);
  col(300, '复制 j1', [[64, 'ink'], [16, 'soft']], ['8', '2'], 'c=[1,2]', 'M=8 不降', false);
  body.push(cap(196, 258, '复制任意 expert 不增 M；复制 argmax 使复制后 M 最小', { size: 8, anchor: 'middle' }));
  body.push(cap(196, 274, 'release policy.py L73：逐 step 窗口向量贪心（目标更细）', { size: 7.5, anchor: 'middle' }));

  body.push(zone({ x: 372, y: 136, w: 312, h: 150, label: '生产参数（doc L127/L131）× moe_lb' }));
  body.push(cap(388, 172, 'prefill：冗余 4/卡 · 窗 100 步 · 周期 1000 步', { size: 8.5, weight: '650' }));
  body.push(cap(388, 190, 'moe_lb 1.074（EP4DP4）/ 1.045（EP2DP2）', { size: 8, mono: true }));
  body.push(cap(388, 210, 'decode：冗余 1/卡 · 窗 1000 步 · 周期 10000 步', { size: 8.5, weight: '650' }));
  body.push(cap(388, 228, 'moe_lb 1.090（EP64）→ 1/1.090 = 91.7%', { size: 8, mono: true, color: 'var(--seal-deep)' }));
  body.push(cap(388, 250, 'moe_lb = slowest rank / mean（EPLB 生效后残差偏斜）', { size: 8 }));
  body.push(cap(388, 266, '≠ §7 ~80% 端到端在线/离线比（两笔账勿混）', { size: 8 }));
  body.push(cap(388, 282, '首次 rearrange 提前到 1/4 周期（冷启动纠正，L121）', { size: 7.5 }));

  body.push(zone({ x: 36, y: 302, w: 648, h: 64, label: '命题 2：负载粒度 CV = (σ/μ)√(e/E) —— EP 越大每卡 expert 越少、相对偏斜越大（σ/μ=1 示意）' }));
  [['EP2', 52, 90, 21.6, '0.072', 'soft'], ['EP4', 180, 218, 30.6, '0.102', 'soft'], ['EP64', 316, 360, 122.4, '0.408', 'seal']].forEach(([lab, lx, bx, bw, v, tone]) => {
    body.push(cap(lx, 344, lab, { size: 8.5, mono: true }));
    body.push(`<g class="sig-bar" style="--i:0.3"><rect class="dia-bar dia-bar-${tone}" x="${bx}" y="334" width="${bw}" height="12" rx="3"/></g>`);
    body.push(cap(bx + bw + 6, 344, v, { size: 7.5, mono: true }));
  });
  body.push(cap(560, 344, 'moe_lb 单调升', { size: 8, color: 'var(--seal-deep)' }));
  body.push(cap(560, 358, '1.045 < 1.074 < 1.090', { size: 7.5, mono: true }));
  body.push(cap(52, 358, 'iid 理想化给方向不给定量拟合；moe_lb 是 max/mean 非 CV —— 解释 prefill 取小 EP、decode 大 EP 必须配 EPLB', { size: 7.5 }));

  body.push(cap(36, 386, 'rounding（L21）：padding = −(384+min_extra) % ep_size —— decode min_extra 64 → 448、7/卡；prefill EP8 min_extra 32 → 416、52/卡', { size: 9.5 }));
  body.push(cap(36, 404, 'shuffle：intranode symm_mem 镜像直读对端权重（L150）；NPU 无 symm_mem → internode NCCL P2P（L65）；DeepEP lb_prefetch_weights 原语层', { size: 9.5 }));
  body.push(cap(36, 422, 'EPLB repo 计划算法 ≠ release 引擎算法：eplb.py weight/logcnt 迭代复制；release policy.py L73 逐 step 窗口贪心；SGLang 镜像 repo 算法（三处分标）', { size: 9.5 }));

  return scene(720, 436, body, { label: 'EPLB 机制：统计窗收集负载、周期性重分配、对称内存_shuffle 权重；贪心复制负载最大 expert 使每副本最大值从八降到四，复制非最大者不降；生产参数 prefill 四冗余百步窗千步周期、decode 一冗余千步窗万步周期，残差偏斜一点零九零' }).svg();
}

/* ============================================================
 * 7.4 §8（afterHeading）— Engram ENP：vocab 分片 × one-sided RDMA read（签名）
 * 原文：ENP8 prefill / decode ENP=DP/EP；384,006,168/8=48,000,771 条/rank；
 *       host memory RDMA 注册，forward 开始 kernel 发 one-sided read，NIC 后台搬；
 *       scales 留 GPU 经 NVLink；NPU 无 offload 全留 NPU memory
 * ============================================================ */
function engramEnp() {
  const UID = 'i7p';
  const body = [];
  const nodes = [];

  body.push(cap(36, 24, 'Engram ENP：沿 vocabulary 维分片 × one-sided RDMA read —— 取数与计算重叠，不占对端 CPU（doc L83-87 官方披露）', { size: 12, color: 'var(--ink-soft)' }));

  body.push(zone({ x: 36, y: 40, w: 648, h: 60, label: 'config [W] × 分片算术' }));
  body.push(cap(52, 70, 'engram_vocab_size 16,000,000 · num_embeddings [384006168, 384016682] · layer_ids [1,14] · n_heads 8 · head_dim 256', { size: 8.5, mono: true }));
  body.push(cap(52, 88, 'prefill ENP8：384,006,168/8 = 48,000,771 条/rank（整除）；decode ENP = DP/EP size（如 64，按 1/ENP 分片）', { size: 8.5 }));

  body.push(zone({ x: 36, y: 116, w: 400, h: 180, label: '每 rank（ENP shard）：host memory → GPU' }));
  const mk = (id, x, y, w, h, role, title, sub, ts = 9.5, ss = 7) => {
    const n = node({ id, x, y, w, h, role, title, sub, titleSize: ts, subSize: ss });
    nodes.push(n); body.push(n.svg);
    return n;
  };
  const hm = mk('hm', 52, 146, 150, 56, 'cache', 'local host memory', 'vocab shard 1/ENP · RDMA 注册', 9.5, 7);
  const gb = mk('gb', 280, 146, 140, 56, 'proj', 'GPU HBM', 'KV cache + 权重 + scales', 10, 7);
  const rd = 'M 202 174 H 280';
  body.push(`<path class="dia-edge strong" d="${rd}" style="stroke:var(--seal);stroke-width:2"/>`);
  body.push(pathPulse(rd, { dur: 1.4, delay: 0.4 }));
  body.push(arrowHead(280, 174, 1, 0));
  body.push(cap(241, 166, 'one-sided RDMA read', { size: 7.5, anchor: 'middle', color: 'var(--seal-deep)' }));
  body.push(cap(241, 190, 'NIC 后台搬运', { size: 6.8, anchor: 'middle' }));
  const kn = mk('kn', 52, 226, 368, 40, 'op', 'forward 开始：一个 kernel 发起 read', '发起后不阻塞 · 取数与计算重叠（与 2.4 GDR swap-in 同思路，方向/源不同）', 9.5, 7);
  body.push(`<path class="dia-edge dash" d="M 127 226 V 202"/>`);
  body.push(arrowHead(127, 202, 0, -1));
  body.push(cap(140, 218, '发起', { size: 7 }));

  body.push(zone({ x: 452, y: 116, w: 232, h: 180, label: 'scales 路径 × NPU' }));
  body.push(cap(468, 152, 'scales 小 → 留 GPU memory', { size: 8.5, weight: '650' }));
  body.push(cap(468, 170, '经 NVLink 传（域内延迟低）', { size: 8 }));
  body.push(cap(468, 188, '避免每次取数都走 host→GPU', { size: 8 }));
  body.push(cap(468, 214, 'NPU：CPU offload 尚不支持', { size: 8.5, weight: '650' }));
  body.push(cap(468, 232, '全部 Engram 数据留 NPU memory', { size: 8 }));
  body.push(cap(468, 250, 'ENGRAM_FORCE_GPU_STORAGE L154', { size: 7.5, mono: true }));
  body.push(cap(468, 266, 'Ascend 强制（与 doc L87 一致）', { size: 7.5 }));
  body.push(cap(468, 286, 'EngramBuffer.fetch(indices) 每层 hook', { size: 7.5 }));

  body.push(cap(36, 316, '为什么 host+RDMA：表条数亿级（384M）全留 HBM 挤占 KV 与权重；host 容量大、RDMA 注册后 NIC 直取，one-sided read 不占对端 CPU', { size: 9.5 }));
  body.push(cap(36, 334, '库层：DeepEP EngramBuffer（experimental，RDMA，多层存储 + 多层 fetch，fetch(indices) 每层返回 completion hook，用前须调 hook，7.3 §6）', { size: 9.5 }));
  body.push(cap(36, 352, '引擎侧：v4_modules/engram.py（CompressedTokenizer L35 / NgramEmbedding L69 / EngramRuntime L164）+ gather_engram_tokens_zos.py', { size: 9.5 }));
  body.push(cap(36, 370, '三笔账：host+RDMA 省 HBM · scales 留 GPU 经 NVLink 低延迟 · forward 开始即发 read 重叠取数与计算 —— RDMA 硬件层归 cluster 维度', { size: 9.5 }));

  return scene(720, 384, body, { label: 'Engram ENP：词表沿 vocab 维分片到各 rank 的 host memory 并注册 RDMA，forward 开始时一个 kernel 发起单边读，网卡后台把数据搬进 GPU 显存与计算重叠；scales 小留 GPU 经 NVLink 传，NPU 不支持 offload 全留 NPU 内存' }).svg();
}

/* ============ 注册表 ============ */
export const FIGURES = [
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/01-deepinfer-architecture',
    replacesBlock: 0,
    title: '完整 serving 系统 × release 边界：实例内全栈，跨实例编排仅 doc',
    svg: servingBoundary,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/01-deepinfer-architecture',
    replacesBlock: 1,
    title: 'deepinfer/ 模块地图：C++ 热路径 × Python 簿记 × 双平台',
    svg: moduleMap,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/01-deepinfer-architecture',
    replacesBlock: 2,
    title: 'ZOS 重叠时序：CPU 工作藏进 GPU forward/sample',
    svg: zosTimeline,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/01-deepinfer-architecture',
    afterHeading: /^## 4\. /,
    title: 'step() 七段流水线 × 引擎角色：ZOS decode 返回上一步输出',
    svg: stepPipeline,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    replacesBlock: 0,
    title: 'DSpark 一步数据流：ConfidenceHead → cumprod → argmax → top-m 前缀',
    svg: dsparkDataflow,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    afterHeading: /^## 4\. /,
    title: '期望吞吐扫描：非单调，argmax 停在 graph 边界 8',
    svg: throughputScan,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    replacesBlock: 1,
    title: 'ZOS 分工：CPU 定 total_q（N−2）× GPU 重算 per-request（N）',
    svg: zosGpuCorrect,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    replacesBlock: 2,
    title: 'StateBlock slot：stored region 落盘 1,415,040 B × dropped 重算恢复',
    svg: stateBlockSlot,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    afterHeading: /^## 7\. /,
    title: 'SWA bounded replay：截断窗钉死末 128 token 重建',
    svg: swaBoundedReplay,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/02-dspark-and-swa-replay',
    afterHeading: /^## 8\. /,
    title: 'hit policy：H ≤ T−134，SWA miss 回退一块',
    svg: hitPolicy,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/03-deepseek-kernel-stack',
    replacesBlock: 0,
    title: 'kernel 栈映射：三条计算路径 × 编译底座 × 双平台',
    svg: kernelStackMap,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/03-deepseek-kernel-stack',
    afterHeading: /^## 4\. /,
    title: '三格式字节账：528 / 288 / 68 = 数据 + scale（512 元素 = 448+64）',
    svg: formatBytes,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/04-deepseek-large-scale-ep',
    replacesBlock: 0,
    title: '四个 LB × release 边界：EPLB 引擎内，三 DP LB 仅 doc',
    svg: fourLB,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/04-deepseek-large-scale-ep',
    afterHeading: /^## 2\. /,
    title: 'Wide-EP 账：每卡权重流量 ∝ 1/e，EP8→EP64 降 7.43×',
    svg: wideEpTraffic,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/04-deepseek-large-scale-ep',
    afterHeading: /^## 3\. /,
    title: 'EPLB：冗余副本摊热 expert × 统计窗重分配 × symm_mem shuffle',
    svg: eplbRebalance,
  },
  {
    dim: 'inference',
    slug: '07-deepinfer-and-deepseek/04-deepseek-large-scale-ep',
    afterHeading: /^## 8\. /,
    title: 'Engram ENP：vocab 分片 × one-sided RDMA read',
    svg: engramEnp,
  },
];