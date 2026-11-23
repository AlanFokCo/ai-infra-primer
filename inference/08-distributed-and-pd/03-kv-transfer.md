# 8.3 KV 传输与互联（KV Transfer and Interconnect）

> 一条 32768-token prompt 的 KV 有 4 GiB，从 prefill 卡搬到 decode 卡——什么时候传、谁发起、传多久才算不亏？本章把传输拆成粒度 / 方向 / 时序三个自由度，推导逐层流水的暴露时间账，走读 NIXL 与 Mooncake transfer engine 两个传输底座的 C++ 接口，再把带宽账对到 TTFT 预算与跨 DC 生产口径 | 前置：1.2（ $\kappa$ 与 prefill/decode 账）、2.4（KV offload 带宽需求式 (3)、GDR 与拷贝路径、Mooncake 架构）、6.3（SGLang PD bootstrap 与 chunk 流水、命题 2）、8.2（PD goodput、传输 vs 重算带宽阈值 $\mathrm{BW}^{\ast}$ ）；5.3 / 5.4（vLLM connector）与 7.4（DeepInfer 部署）只 recap | 锚点：NIXL（[NVIDIA, ai-dynamo/nixl](../references.md)，教学参照 + 官方实现，cb66ce83）、Mooncake transfer engine（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]，继承 kvcache-ai/Mooncake 2cdbaeeb）、DeepInfer doc（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）、Splitwise（[Patel et al., 2024](../references.md)，arXiv:2311.18677，[B]）、Beyond the Buzz（[Mitra et al., 2025](../references.md)，arXiv:2506.05508，[B]）、DistServe（[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]）、vLLM / SGLang repo [A]；代码锚点 2026-11-23 核实

## 1. 传输语义：粒度、方向、时序

4 GiB KV 在 400 Gbps 网络上要传 85.9 ms——若这笔时间全部计入 TTFT，2 s 的 SLO 预算一下吃掉 4.3%。能不能把它藏进 prefill 计算里？答案取决于传输的三个自由度，本节先把它们定义清楚。

**本章口径**（R2-2 章脊）：主口径 = Llama-3.1-8B GQA（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16，与 2.4 章脊一致）， $\kappa$ = 4096 B/token/层、 $\kappa L$ = 131072 B/token = 128 KiB；硬件 H100 SXM（bf16 dense 989.5 TFLOPS、HBM 3.35 TB/s，[NVIDIA, 官方 datasheet](../references.md)，[C] 官方规格披露）， $F_{\mathrm{eff}}$ = 50% × 989.5 = 494.75 TFLOPS（教学假设，沿 2.4 §2）；prompt $T$ = 32768（与 6.3 传输口径一致）。传输口径：400 Gbps = 50 GB/s（单 NIC， $10^9$ 口径）；跨 DC 每 8 GPU 共享 400 Gbps → 每 GPU 6.25 GB/s（DeepInfer doc L97 官方披露）。生产披露口径 = DeepInfer V4.1-Flash（ $L$ = 40、主 KV per-16 FP4 $\kappa$ = 288 B/token/层，[W] + doc，7.3 §4 格式账 recap），换口径原因：跨 DC 生产数字只存在于该口径。MLA 对照口径 = DeepSeek-V3 系（FP8 KV $\kappa$ = 576 B/token/层、 $L$ = 61、 $N_{\mathrm{act}}$ = 37B，8.2 数值锚 recap），仅用于 $\mathrm{BW}^{\ast}$ 对照表。

所谓 P2D 传输（prefill-to-decode KV transfer），是指 prefill 实例算出的 KV cache 经网络送到 decode 实例、供其继续生成的数据面搬运。它有三个正交自由度：

- **粒度**：full 一次性（整条请求 KV 算完后一次传）vs layer-wise 流式（每层算完即传该层 KV，chunk-wise 是其粗粒度变体，6.3 §4 的 `send_kv_chunk`）。
- **方向**：push（prefill 侧发起 WRITE，把 KV 写到 decode 的远端地址）vs pull（decode 侧发起 READ，从 prefill 的远端地址读回）。方向是 **initiator 视角**——谁 post 传输请求谁是 initiator，与数据流向无关（NIXL `NIXL_WRITE` / `NIXL_READ` 即此口径，§3）。
- **时序**：serialized（传输与计算串行，prefill 完 → 传 → decode 开始）vs overlapped（传输与 prefill 后续层计算重叠）。

粒度与时序不独立：layer-wise 是 overlapped 的实现前提——只有按层切分，才能让第 $i$ 层的传输与第 $i+1$ 层的计算并行。full 一次性只能 serialized（KV 全部算完才存在）。

```
时间 →
serialized（full 一次性）:
prefill SM   [================ 全部 L 层 ================]
RDMA NIC                                                  [==== 全部 KV ====]
decode                                                                    [DECODE...

layer-wise（逐层流式）:
prefill SM   [L0][L1][L2][L3]........................[L31]
RDMA NIC         [L0][L1][L2][L3]................[L31]
decode                                                  ↑ 末层到齐 → DECODE
             传输藏进计算，暴露的只有末层尾巴
```

Splitwise 对这三种组合给了生产级裁定（[Patel et al., 2024](../references.md)，arXiv:2311.18677，[B]，§IV-C / §VI-A）：serialized 传输「starts only after the prompt phase has finished and the first token is generated」且须在下一个输出 token 前完成——直接顶高 max TBT 与端到端延迟；layer-wise 在每层末尾触发异步传输、prompt 计算继续下一层，附带两个收益：token 相更早启动、prompt 机 KV 内存更早释放。代价是逐层细粒度同步，小 prompt 上同步开销可能反增 TTFT——所以 Splitwise 按 prompt 大小切换：H100 上 < 512 token 用 serialized，更大用 layer-wise（token 数启动时已知）。实测账：layer-wise 把第二 token 延迟开销从 serialized 的 64% 压到 16.5%（3.9× 改善），端到端开销从 3% 压到 0.8%；非重叠尾巴近似常数（A100 ~8 ms / H100 ~5 ms，200 vs 400 Gbps）。

> **提点**：「暴露」（口径：未被计算藏住、计入 TTFT 的传输时间）是本章的核心量。layer-wise 的目标不是让传输变快，而是让传输与计算抢同一段时间——NIC 与 SM 是两种资源，重叠后传输的边际成本趋近末层尾巴。

P2D 时序还有一个容易漏的事实：**第一个输出 token 在 prefill 侧采样**，其 id / logprobs 经 aux 通道随 KV 传到 decode（6.3 §4-5 recap）；decode 侧 KV 到齐后置 `ForwardMode.PREBUILT`，只做 metadata 填充、不跑 prefill forward，第一次真实 forward 已是第二个 token 的 DECODE 步。所以要分清首 token 的**计算**与**交付**：计算在 prefill 侧早完成（不等待 KV 传输），但交付须等首 token 经 aux 通道随末 chunk 到达 decode、再流给用户——用户感知的 TTFT 因此含暴露传输项（式 (4) 的 $t_{\mathrm{exposed}}$ ，与 6.3 §4「暴露＝计入 TTFT 的传输时间」同口径）；被重叠藏住的部分不计入，只有暴露尾巴计入。「第二 token」的说法限于 decode 侧第一次真实 forward（PREBUILT 之后的 DECODE 步），不是 TTFT 的交付时刻。

## 2. 逐层流水的暴露时间账

暴露时间到底是多少？先给结论的量级：本章口径下 full serialized 暴露 85.9 ms，layer-wise 只暴露 2.68 ms——差 32 倍，恰等于层数 $L$ 。下面推导这个 32× 的结构来源。

路线：把逐层传输写成两阶段流水（计算阶段 + 传输阶段），用归纳法求末层到达时刻，再减去纯计算时间得暴露量——关键一步是认出流水的瓶颈阶段决定 makespan 的形式。

传输总时间 recap 6.3 式 (5)（ $\kappa$ recap 1.2 式 (3)）：

$$
t_{\mathrm{xfer}} = \frac{\kappa L (T - T_{\mathrm{hit}})}{\mathrm{BW}} \qquad (1)
$$

符号表： $\kappa$ = 每 token 每层 KV 字节数； $L$ = 层数； $T$ = prompt 长； $T_{\mathrm{hit}}$ = decode 侧已有前缀长（bootstrap 时承诺，命中部分不传，2.2 recap）； $\mathrm{BW}$ = 有效传输带宽（GB/s， $10^9$ 口径）。代入章脊（ $T_{\mathrm{hit}}$ = 0）： $t_{\mathrm{xfer}}$ = 4.295 GB / 50 GB/s = 85.9 ms。

设每层计算时间 $\tau_c = t_P / L$ 、每层传输时间 $\tau_x = t_{\mathrm{xfer}} / L$ （均匀层口径）， $t_P$ 为 prefill 纯计算时间。第 $i$ 层计算在 $i \tau_c$ 时刻完成，其传输须等计算完成且 NIC 空闲（单通道）。记第 $i$ 层传输完成时刻为 $r_i$ ：

$$
r_i = \max(i \tau_c,\ r_{i-1}) + \tau_x, \qquad r_0 = 0 \qquad (2)
$$

**命题 1**（逐层流水末层到达时刻）：式 (2) 的解为 $r_L = \tau_c + \tau_x + (L-1) \max(\tau_c, \tau_x)$ ；相对纯计算 $t_P = L \tau_c$ 的暴露时间为 $\max(\tau_x,\ t_{\mathrm{xfer}} - (L-1)\tau_c)$ 。

**证明**（路线：对 $i$ 归纳，分 $\tau_c \ge \tau_x$ 与 $\tau_c \lt \tau_x$ 两情形——关键一步是比较 $r_{i-1}$ 与 $i\tau_c$ 决定 $\max$ 取谁）。基例： $r_1 = \max(\tau_c, 0) + \tau_x = \tau_c + \tau_x$ ，与公式 $i$ = 1 一致。保持：设 $r_{i-1} = \tau_c + \tau_x + (i-2)\max(\tau_c,\tau_x)$ 。情形一 $\tau_c \ge \tau_x$ ： $r_{i-1} = (i-1)\tau_c + \tau_x \le (i-1)\tau_c + \tau_c = i\tau_c$ ，故 $\max(i\tau_c, r_{i-1}) = i\tau_c$ ， $r_i = i\tau_c + \tau_x = \tau_c + \tau_x + (i-1)\tau_c$ ，与公式（ $\max = \tau_c$ ）一致。情形二 $\tau_x \gt \tau_c$ ： $r_{i-1} = \tau_c + (i-1)\tau_x \ge \tau_c + (i-1)\tau_c = i\tau_c$ ，故 $\max(i\tau_c, r_{i-1}) = r_{i-1}$ ， $r_i = r_{i-1} + \tau_x = \tau_c + i\tau_x$ ，与公式（ $\max = \tau_x$ ）一致。终止： $i = L$ 得 $r_L = \tau_c + \tau_x + (L-1)\max(\tau_c,\tau_x)$ 。暴露量 = $r_L - L\tau_c$ ：情形一为 $\tau_x$ ；情形二为 $\tau_c + L\tau_x - L\tau_c = t_{\mathrm{xfer}} - (L-1)\tau_c$ 。两情形合并即 $\max(\tau_x,\ t_{\mathrm{xfer}} - (L-1)\tau_c)$ 。证毕。

正例（计算主导， $\tau_c \ge \tau_x$ ）：章脊 $t_P = 2 N T / F_{\mathrm{eff}} = 2 \times 8 \times 10^9 \times 32768 / (4.9475 \times 10^{14}) = 1.060$ s（线性项口径，attention 二次项使 $t_P$ 更长、结论同向）， $\tau_c$ = 33.1 ms、 $\tau_x$ = 85.9/32 = 2.68 ms。暴露 = $\tau_x$ = 2.68 ms，是 full serialized 85.9 ms 的 1/32。边界反例（传输主导， $\tau_x \gt \tau_c$ ）：若 $\mathrm{BW}$ = 1 GB/s， $t_{\mathrm{xfer}}$ = 4.295 s、 $\tau_x$ = 134.2 ms $\gt \tau_c$ = 33.1 ms，暴露 = 4.295 − 31 × 0.0331 = 3.27 s——此时 layer-wise 只藏住 31 层的计算，暴露量随 $T$ 线性涨，不再是被 $L$ 除的小尾巴。

> **坑**：把命题 1 的 $\tau_x$ 当作实测暴露时间的拟合值。触发条件：用 Splitwise 的常数尾巴（A100 ~8 ms / H100 ~5 ms）反推每层传输时间；现象：模型给 $\tau_x = \kappa T / \mathrm{BW}$ 随 $T$ 涨，实测却近似常数。原因：实测尾巴含逐层同步、kernel 启动、末块对齐等固定开销，命题 1 给的是带宽项的结构上界，不拟合该常数。规避：结构账用命题 1，固定开销另列延迟项（2.4 §2 的 $\ell_i$ 口径）。

命题 1 的两情形分界是 $\tau_c = \tau_x$ ，即 $t_P = t_{\mathrm{xfer}}$ ——这恰好是 8.2 传输 vs 重算的临界条件。§6 命题 2 把它写成带宽阈值并证明 egress 需求恒等于 $\mathrm{BW}^{\ast}$ 。

## 3. NIXL：agent、descriptor 与插件抽象

引擎为什么不直接调 RDMA verbs，而要隔一层 NIXL？因为传输后端不止一种（UCX / GDS / Mooncake / hf3fs / ...），内存类型也不止一种（DRAM / VRAM / 文件 / 对象），引擎不该为每种组合写一套搬运代码。NIXL（NVIDIA Inference Xfer Library）把这层抽象成统一接口（[NVIDIA, ai-dynamo/nixl](../references.md)，教学参照 + 官方实现，cb66ce83，2026-11-23 tarball 核实）。

定位（README + docs/nixl.md）：AI 推理（Dynamo）的点对点通信加速库，**memory（CPU/GPU）+ storage（file/block/object）统一抽象、模块化插件架构**；核心 backend = UCX + GDS（GPUDirect Storage）；数据面覆盖 RoCE / InfiniBand / GPUDirect RDMA / NVMe-oF / TCP / NVLink。conductor 进程管理 agent（每 agent 可管多 GPU，全局唯一 name）。

Transfer Agent 三抽象（docs/nixl.md Design 节）：

| 抽象 | 职责 | 关键语义 |
| --- | --- | --- |
| Memory Section | 注册 DRAM / VRAM / NVMe-oF / Object / File 段 | 注册描述符列表 `nixl_reg_dlist_t` |
| Transfer Backend Interface | 按 src/dst 内存类型 + 双方共有 backend 选最优路径 | 插件化，引擎不感知具体 transport |
| Metadata Handler | side-channel 或 etcd/Redis 中央交换元数据 | **控制路径，非数据路径**；缓存；加 agent = 提供 metadata，删 agent = invalidate + disconnect |

```
        控制面（metadata，小字节）              数据面（KV，大字节）
prefill agent ──getLocalMD──► blob ──► decode agent      prefill VRAM ══RDMA══► decode VRAM
     │                          loadRemoteMD                │   NIXL_WRITE/READ      │
     └── registerMem(reg_dlist) ──┘                         └── postXferReq(xfer_dlist) ──┘
              元数据走 side-channel / etcd，KV 不走这条通道
```

C++ agent API（src/api/cpp/nixl.h，file:line 锚）：

| API | 行 | 语义 |
| --- | --- | --- |
| `nixlAgent` | L35 | 主传输对象，构造传 name + config |
| `registerMem` / `deregisterMem` | L118 / L131 | 注册 / 注销内存，入参 `nixl_reg_dlist_t` |
| `prepXferDlist` | L184 | 准备描述符列表句柄（本地 / 远端两 overload） |
| `makeXferReq` | L257 | 用已准备句柄 + 索引构造传输请求 |
| `createXferReq` | L306 | 一次性构造（直接传 local/remote descs + agent name） |
| `postXferReq` | L345 | 异步提交，返回 `NIXL_IN_PROG` 或 `NIXL_SUCCESS` |
| `getXferStatus` | L355 | 异步 poll 完成状态 |
| `getLocalMD` / `loadRemoteMD` / `invalidateRemoteMD` | L512 / L543 / L555 | 元数据导出 / 加载 / 失效（控制面） |

descriptor 三型（src/api/cpp/nixl_descriptors.h）：`nixlBasicDesc` L33（单连续区，`addr` uintptr_t + `len` size_t + `devId` uint64_t）/ `nixlStrideDesc` L168（跨步：count 个等长块、块间距 stride，stride == len 即连续）/ `nixlBlobDesc` L226（带元数据 blob，继承 BasicDesc）。列表 typedef：`nixl_xfer_dlist_t` L547 = `nixlDescList<nixlBasicDesc>`（**传输**列表）/ `nixl_reg_dlist_t` L559 = `nixlDescList<nixlBlobDesc>`（**注册**列表）。操作枚举 `nixl_xfer_op_t {NIXL_READ, NIXL_WRITE}`（nixl_types.h L48），initiator 视角。

> **坑**：混用 `nixl_xfer_dlist_t` 与 `nixl_reg_dlist_t`。触发条件：把注册列表直接传给 `createXferReq`；现象：类型不匹配（reg 是 BlobDesc、xfer 是 BasicDesc），编译期或运行期报错。规避：注册用 reg_dlist（带 blob 元数据），传输用 xfer_dlist（纯 addr+len+devId）；两者元素类型不同，不可互代。

教学代码片段（教学用，非优化实现；签名按 repo examples/cpp/nixl_example.cpp + nixl.h，省略错误处理与 extra_params）：

```cpp
// decode 侧 agent，push 语义由 prefill 侧发起；此处演示 prefill agent 写 decode
nixlAgentConfig cfg;
nixlAgent A("prefill-0", cfg);
nixl_reg_dlist_t reg(VRAM_SEG);
nixlBlobDesc buf; buf.addr = (uintptr_t)kv_ptr; buf.len = kv_bytes; buf.devId = 0;
reg.addDesc(buf);
A.registerMem(reg);                       // 注册本地 KV buffer
std::string meta; A.getLocalMD(meta);     // 导出元数据 → 经控制面交给 decode
// decode 侧：A.loadRemoteMD(meta_from_prefill) 后才能访问远端
nixl_xfer_dlist_t src(VRAM_SEG), dst(VRAM_SEG);
nixlBasicDesc s; s.addr = (uintptr_t)kv_ptr; s.len = chunk_bytes; s.devId = 0;
nixlBasicDesc d; d.addr = remote_kv_addr;   d.len = chunk_bytes; d.devId = 0;
src.addDesc(s); dst.addDesc(d);
nixlXferReqH* req;
A.createXferReq(NIXL_WRITE, src, dst, "decode-0", req);  // push = initiator WRITE
A.postXferReq(req);                                      // 异步
while (A.getXferStatus(req) != NIXL_SUCCESS) { /* poll */ }
```

插件抽象（src/plugins/，18 个插件目录，2026-11-23 tarball 逐目录核实）：ucx / cuda_gds / **mooncake** / hf3fs / obj / posix / libfabric / gpunetio / gpudirect_tcpxo / doca_memos / azure_blob / redis / uccl / rocm_ais / infinia / gusli / telemetry / tracing。**NIXL 经 mooncake 插件可用 Mooncake TE 作 backend**（§4 对照）——两个传输底座不是互斥竞争，而是可组合的上下层。

> **坑**：把 GDR 当成 NIXL 的独立 backend。触发条件：按「UCX / GDR / RDMA」并列读旧文档；现象：在插件列表里找 GDR 找不到。事实：GDR（GPUDirect RDMA）是 UCX backend 下的 GPU 直写路径，不是独立插件；核心 backend 是 UCX + GDS（GPUDirect Storage，面向 NVMe）。规避：backend 看 src/plugins/ 目录，GDR 看 UCX 的 GPU 内存注册路径（§5）。

引擎集成（W3 核维持，recap 5.3 / 6.3）：vLLM `NixlPullConnector` L374（READ）/ `NixlPushConnector` L415（WRITE）；SGLang `disaggregation/nixl/conn.py` `NixlKVManager` L531。push/pull 双模即 `NIXL_WRITE` / `NIXL_READ` 的引擎侧封装（§7 映射表）。Python 绑定一句注：W3 已核 Python `_api.py`，本章以 C++ agent/descriptor 语义为主。

## 4. Mooncake transfer engine：Segment、BatchTransfer 与拓扑感知

单张 NIC 50 GB/s，一台服务器 8 张 GPU 各要传 KV——聚合带宽从哪来？Mooncake transfer engine 的答案是多网卡池化 + 拓扑感知路径选择（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]；repo kvcache-ai/Mooncake 2cdbaeeb，docs/source/design/transfer-engine/index.md 678 行 + C++ 头文件，2026-11-23 contents API 核实）。架构级 recap 2.4 §5（KVCache-centric、闲置 DRAM/SSD 池化），本节走传输底座接口。

两个核心抽象（index.md）：

- **Segment**：可远程读写的连续地址空间。RAM Segment（DRAM/VRAM）每进程**一个**，名 = `local_hostname`，逻辑覆盖全地址空间；内部划多个 **Buffer**（连续、同设备、各自 rkey 权限 + NIC 亲和）。远程 `openSegment(name)` 取引用。NVMeof Segment 另型：文件 → DRAM/VRAM 零拷贝、不经 CPU。
- **BatchTransfer**：READ/WRITE 请求数组（op / length / local addr / remote addr），异步 `getTransferStatus` 监控完成；design doc 称之为「async flexible AllScatter/AllGather」——比集合通信更灵活，因为每个请求的源/目的/长度独立。

> **坑**：把 Segment 当 Buffer。触发条件：以为每次注册内存创建一个 Segment；现象：Segment 数爆炸、metadata 膨胀。事实：每进程一个 RAM Segment（名 = hostname），Buffer 才是注册粒度（`registerLocalMemory` 加一个 Buffer）。规避：Segment 是进程级地址空间容器，Buffer 是其中的连续区。

C++ API 源码锚（mooncake-transfer-engine/include/）：

| 文件 | 符号 | 行 | 语义 |
| --- | --- | --- | --- |
| transfer_engine.h | `TransferEngine` | L90 | 主类 |
| transfer_engine.h | `openSegment` | L156 | 按名取远端 Segment 句柄 |
| transfer_engine.h | `registerLocalMemory` | L187 | 注册本地 Buffer（addr+len+location） |
| transfer_engine.h | `submitTransfer` | L205 | 提交 BatchTransfer（请求数组） |
| transfer_engine.h | `allocateBatchID` | L287 | 分配批 ID |
| transfer_engine.h | `getTransferStatus` | L305 | 异步 poll 单请求状态 |
| transfer_engine.h | `getTaskCongestionState` | L308 | 拥塞状态（HEAD 新增 API） |
| topology.h | `TopologyEntry` / `Topology` | L38 / L64 | 拓扑条目（preferred_hca L40）/ 拓扑类 |
| topology.h | `discover` / `selectDevice` / `getMatrix` | L74 / L109 / L126 | 发现 NIC / 按 storage_type+retry 选设备 / 取矩阵 |
| transport.h | `TransferRequest` / `OpCode {READ, WRITE}` | L66 / L67 | 请求结构 / 操作枚举 |

Transport 类族：Tcp / Rdma / Efa / NVMeoF / Nvlink / IntraNodeNvlink / Hip / Shm——本地目标走 memcpy / cudaMemcpy，同 host DRAM 走 SHM，跨机走 RDMA / TCP，GPU 直传走 Nvlink，NVMe 走 cuFile GDS。

拓扑感知路径选择（index.md L59-73）：每服务器生成 **topology matrix** 广播集群；按内存类型把 NIC 分 **preferred / secondary** 两列表（preferred = 本地 NUMA / 本地 PCIe switch 的 GDR 路径）；正常用 preferred，故障时两列表皆可用。**单请求 > 64 KB 切多 slice，各 slice 可走不同路径 → 全 NIC 协同**。

```
本地节点（topology matrix）                    远端节点
cpu:0 ── preferred: mlx5_0, mlx5_1            cpu:1 ── preferred: mlx5_2, mlx5_3
cpu:1 ── preferred: mlx5_2, mlx5_3
gpu:0 ── preferred: mlx5_4（本地 PCIe switch）
        请求 > 64 KB → 切 slice：
        [slice0 → mlx5_0] [slice1 → mlx5_1] [slice2 → mlx5_4] ...
        聚合带宽 = 各 NIC 带宽之和（多网卡池化）
```

聚合带宽账：

$$
\mathrm{BW}_{\mathrm{pool}} = \sum_{j=1}^{n_{\mathrm{NIC}}} \mathrm{BW}_j \qquad (3)
$$

符号表： $`\mathrm{BW}_j`$ = 第 $j$ 张 NIC 的有效带宽； $n_{\mathrm{NIC}}$ = 参与池化的 NIC 数。前提：请求被切成足够多 slice 分散到各 NIC（> 64 KB 阈值），且各路径无共同瓶颈（PCIe switch / UPI 不超限——拓扑感知正是为避开这些瓶颈）。代入：4 × 400 Gbps NIC → $`\mathrm{BW}_{\mathrm{pool}}`$ = 200 GB/s。章脊每层 KV = $\kappa T$ = 4096 × 32768 = 134.2 MB，按 64 KB 切 = 2048 个 slice——足够分散到 4 张 NIC，每层传输时间从单 NIC 2.68 ms 降到 0.67 ms。

endpoint 管理（index.md L75-88）：按需建连、endpoint pooling（cap 活跃连接数）、**SIEVE 驱逐**、`waiting_list_` 异步回收（~1 Hz `monitorWorker` heartbeat）。fault handling（L90-98）：NIC 暂不可用 → 自动选替代路径重提交；检测 RDMA context / CQ 问题暂避至恢复。

> **提点**：版本漂移注（2026-11-23 HEAD）：transfer engine 新增 congestion API（`getTaskCongestionState`）/ Ascend Direct Transport / `disconnect_all_peers`；index.md 已扩至 678 行。核心 Segment / BatchTransfer / topology 语义不变，本章锚核心接口，新增件一句登记。

NIXL × Mooncake 对照：NIXL 的 mooncake 插件把 Mooncake TE 接成一个 backend——上层引擎调 NIXL 统一接口，底层可走 UCX 也可走 Mooncake TE。选型不是「NIXL 还是 Mooncake」，而是「引擎接口层用谁、传输底座用谁」两层决策（8.4 对照表）。

## 5. GDR 与拷贝路径：谁来付搬运成本

同样搬 KV，用 SM kernel、CUDA copy engine 还是 RDMA NIC 直写，差别不只是带宽，还有**谁付成本**——SM 付了计算就少了，copy engine 付了细粒度拷贝就慢了。本节 recap 2.4 §4 的三条路径，补 P2D 语境。

所谓 GDR（GPUDirect RDMA），是指 RDMA NIC 绕过 host memory 与 CPU 拷贝栈、直接读写 GPU HBM 的 PCIe 直连数据路径（硬件协议层归 cluster 维度）。DeepInfer swap-in 口径（官方披露，doc L191 逐字）：

> 「Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.」

三条拷贝路径 recap（2.4 §4 已交付坑注与机制，此处只登记对照）：

| 路径 | 占 SM | 细粒度块拷贝 | P2D 语境 |
| --- | --- | --- | --- |
| SM kernel 拷贝 | 是 | 灵活但与计算争资源 | vLLM `swap_blocks_triton.py` 机制例证（2.4） |
| CUDA copy engine（DMA） | 否 | 差（逐描述符开销主导） | 小块 gather 成瓶颈（2.4 §4 坑注） |
| GDR（RDMA NIC 直写 HBM） | 否 | 好（NIC 饱和 PCIe） | VRAM→VRAM 直传、swap-in 逐层 H2D |

P2D 语境下 GDR 的价值：Dynamo 的 NIXL KV 传输是 **VRAM→VRAM 直传**（prefill GPU HBM → decode GPU HBM，不经 host），GDR 让 NIC 直接读写两端 HBM、不占 SM（8.4 §4）。DeepInfer 的 swap-in 是 host shm → HBM 的 H2D，同样用 GDR 逐层重叠 forward（2.4 §4 三段流水 recap）。

小片段问题与 bounce buffer：当 ctx 与 gen 的 attention 切分不同（ctx TP ≠ gen TP / attention DP），一次 KV write 由大量小 fragment 组成，每个 fragment 一个 NIXL descriptor——描述符开销主导。TRT-LLM 的对策是 bounce buffer：sender 把小 fragment gather 进一块连续缓冲、按 chunk 发送，receiver 再 scatter 回 KV cache（默认 ≥ 1024 descriptor 且平均 ≤ 16 KiB 才启用，8.4 §5 详）。这是「把小块聚合成大拷贝」的 P2D 版本，与 2.4 §4 坑注的规避手段同源。

**边界声明**：RDMA 协议（RC/UC/QP）、NIC 硬件、PCIe 拓扑、NVLink 域、交换机与 NCCL 实现归 cluster 维度。本章只讲传输语义（push/pull、layer-wise、Segment/descriptor）、带宽账与系统设计；GDR 的 PCIe 直连机制、RoCE vs InfiniBand 的协议差异不展开。

## 6. 带宽账：TTFT 预算、egress/ingress 与跨 DC

传输要多快才不亏？预算是 TTFT，不是容量。本节先给 P2D 的预算分解，再走 Beyond the Buzz 的 egress/ingress 公式，最后对到跨 DC 生产口径。

### 6.1 TTFT 预算分解

PD 分离下 TTFT 的构成（口径：首 token 在 prefill 侧采样，§1）：

$$
\mathrm{TTFT} = t_{\mathrm{queue},P} + t_P + t_{\mathrm{exposed}} + t_{\mathrm{aux}} + t_{\mathrm{admit},D} \qquad (4)
$$

符号表： $t_{\mathrm{queue},P}$ = prefill 排队； $t_P$ = prefill 计算； $t_{\mathrm{exposed}}$ = 暴露传输（命题 1）； $t_{\mathrm{aux}}$ = aux 通道（首 token 元数据，~14.4 KB，6.3 §5）； $t_{\mathrm{admit},D}$ = decode 侧准入 + poll 间隔。把 TTFT SLO 分给传输的份额记 $t_{\mathrm{budget}}$ ，需求带宽 recap 2.4 式 (3) 的同形：

$$
\mathrm{BW}_{\mathrm{req}} = \frac{\kappa L T}{t_{\mathrm{budget}}} \qquad (5)
$$

代入章脊：TTFT SLO 2 s、份额 10% → $t_{\mathrm{budget}}$ = 0.2 s， $`\mathrm{BW}_{\mathrm{req}}`$ = 4.295 GB / 0.2 s = 21.5 GB/s。400 Gbps 单 NIC（50 GB/s）满足（余量 2.3×）；跨 DC 每 GPU 6.25 GB/s 不满足 full serialized——但 layer-wise 把暴露压到 2.68 ms（命题 1），远小于 0.2 s 预算，所以**预算约束的是暴露时间而非裸传输时间**，这是 2.4 §3 offload 账与 P2D 账的关键差异：offload 取回没有计算可重叠（取回前无法开算），P2D 传输可以藏进 prefill 后续层。

### 6.2 egress / ingress：实例级摊销账

Beyond the Buzz 给了 per-GPU 带宽需求的解析式（[Mitra et al., 2025](../references.md)，arXiv:2506.05508，[B]，§5.1 式 (1)(2)；无官方 repo，讲论文不推测代码）。原式用 MHA/GQA 几何 $d_{\mathrm{head}} \times H_{kv} \times \mathrm{bytes}$ ；本书统一以 $\kappa$ （每 token 每层全部 KV 字节，含 K/V 两份，1.2 式 (3)）代入，MLA 以 latent 几何计 $\kappa$ ——两种写法的几何因子口径不同，勿混（论文 `bytes_element` 定义句为「KV cache bytes per token」，与逐元素字节数有歧义，本书以 $\kappa$ 为准）。术语对照：FTL（First Token Latency）在本节带宽公式取论文 §5.1 口径 = prefill 计算完成时间（不含排队），TTL（Token-to-Token Latency）= TPOT。口径注：FTL 在论文 Table 1 另定义为「跑 prefill 并生成首 token 的延迟」≈ TTFT，8.2 §4 取该义——两义不同，代入式 (6) 用纯计算口径，跨章对照见 8.2 术语对照。

egress（prefill 侧发出，完全重叠所需）：

$$
\mathrm{BW}_{\mathrm{egress}} = \frac{\kappa L \cdot B_P \cdot \mathrm{ISL}}{\mathrm{FTL} \cdot n_{\mathrm{GPU},P}} \qquad (6)
$$

ingress（decode 侧接收，按生成时长摊销）：

$$
\mathrm{BW}_{\mathrm{ingress}} = \frac{\kappa L \cdot B_D \cdot \mathrm{ISL}}{\mathrm{TTL} \cdot \mathrm{OSL} \cdot n_{\mathrm{GPU},D}} \qquad (7)
$$

符号表： $B_P / B_D$ = prefill / decode 实例 batch（请求数）； $\mathrm{ISL} / \mathrm{OSL}$ = 输入 / 输出序列长； $n_{\mathrm{GPU},P} / n_{\mathrm{GPU},D}$ = 实例内**实际分片 KV 的 GPU 数**（口径：TP 域超 KV head 数时 KV 复制而非分片，复制的 GPU 不计入归一——DP attention 下每 DP rank 独立持 KV，归一只计实际分片者，论文 §5.1 逐字）。

**命题 2**（egress 需求恒等于带宽阈值）：线性 prefill 口径下（ $\mathrm{FTL} = 2 N B_P \mathrm{ISL} / (n_{\mathrm{GPU},P} F_{\mathrm{eff}})$ ），式 (6) 的 per-GPU egress 需求恒等于 $\mathrm{BW}^{\ast} = \kappa L F_{\mathrm{eff}} / (2N)$ ，与 $B_P$ 、 $\mathrm{ISL}$ 、 $n_{\mathrm{GPU},P}$ 无关。

**证明**（路线：代入 FTL 的线性表达式，约去 batch 与长度——关键一步是认出分子分母同含 $B_P \mathrm{ISL}$ ）。 $`\mathrm{BW}_{\mathrm{egress}} = \kappa L B_P \mathrm{ISL} / (\mathrm{FTL} \cdot n_{\mathrm{GPU},P})`$ ，代入 $\mathrm{FTL}$ ：分母 = $`[2 N B_P \mathrm{ISL} / (n_{\mathrm{GPU},P} F_{\mathrm{eff}})] \cdot n_{\mathrm{GPU},P} = 2 N B_P \mathrm{ISL} / F_{\mathrm{eff}}`$ 。故 $`\mathrm{BW}_{\mathrm{egress}} = \kappa L B_P \mathrm{ISL} F_{\mathrm{eff}} / (2 N B_P \mathrm{ISL}) = \kappa L F_{\mathrm{eff}} / (2N) = \mathrm{BW}^{\ast}`$ 。证毕。推论：命题 1 的计算主导条件 $\tau_c \ge \tau_x$ 等价于 $\mathrm{BW} \ge \mathrm{BW}^{\ast}$ 。layer-wise 全重叠条件 = $\mathrm{BW}^{\ast}$ 已由 8.2 命题 6 立（原理层，证明留 8.2 练习 4）；本章增量是把它对到 Beyond the Buzz 的 per-GPU egress 公式（式 (6)）并给代数恒等式——8.2 的传输 vs 重算阈值、本章的流水主导区、egress 完全隐藏需求，三者是同一个不等式的三种读法。

数值实例（章脊，TP8 prefill 实例， $B_P$ = 8、 $\mathrm{ISL}$ = 4096）：总 token = 32768， $\mathrm{FTL} = 2 \times 8 \times 10^9 \times 32768 / (8 \times 4.9475 \times 10^{14}) = 132.5$ ms；总 KV = 4.295 GB；聚合 egress = 4.295 GB / 0.1325 s = 32.4 GB/s；per-GPU = 32.4/8 = 4.05 GB/s = $\mathrm{BW}^{\ast}$ （2.4 式 (2) 章脊值）。跨 DC 每 GPU 6.25 GB/s $\gt$ 4.05 → egress 完全隐藏可行，余量 1.54×。边界：attention 二次项使 FTL 超线性涨（ $\propto \mathrm{ISL}^2$ 项），长 prompt 下实际 egress 需求**低于**线性 $\mathrm{BW}^{\ast}$ ——命题 2 的线性口径是保守上界。

**命题 3**（ingress 摊销标度）：在 decode 步时由 KV 读主导的口径下（ $\mathrm{TTL} \propto \mathrm{ISL}$ ，1.2 roofline recap），式 (7) 的 ingress 需求与 $\mathrm{ISL}$ 无关、 $\propto 1/\mathrm{OSL}$ ；若 TTL 由权重读主导（短上下文），则 ingress $\propto \mathrm{ISL}$ 。

**证明**（路线：把 TTL 对 ISL 的依赖代入式 (7)，看 ISL 是否约去）。KV 读主导时 $`\mathrm{TTL} \approx B_D \kappa L \bar{T} / (n_{\mathrm{GPU},D} \mathrm{BW}_{\mathrm{HBM}})`$ ， $\bar{T} \approx \mathrm{ISL} + \mathrm{OSL}/2 \approx \mathrm{ISL}$ （ISL ≫ OSL），故 $\mathrm{TTL} = c \cdot \mathrm{ISL}$ 。代入式 (7)： $`\mathrm{BW}_{\mathrm{ingress}} = \kappa L B_D \mathrm{ISL} / (c \cdot \mathrm{ISL} \cdot \mathrm{OSL} \cdot n_{\mathrm{GPU},D}) = \kappa L B_D / (c \cdot \mathrm{OSL} \cdot n_{\mathrm{GPU},D})`$ ，ISL 约去， $\propto 1/\mathrm{OSL}$ 。权重主导时 TTL ≈ 常数（与 ISL 无关），式 (7) 分子 $\propto \mathrm{ISL}$ → ingress $\propto \mathrm{ISL}$ 。证毕。边界反例：OSL 极短（分类任务只出几个 token）时 ingress 飙升——摊销窗口短，KV 到达集中在生成初期。

数值实例（章脊，decode 实例 TP8， $B_D$ = 64、 $\mathrm{ISL}$ = 4096、 $\mathrm{OSL}$ = 1024、TTL = 2 ms 教学值）：roofline 校验 TTL——每 GPU 每步流量 = 权重 16 GB/8 + KV 读 $64 \times 131072 \times 4608 / 8$ = 2 + 4.83 = 6.83 GB，@3.35 TB/s = 2.04 ms，与 2 ms 教学值一致（KV 读主导，2.4× 权重）。式 (7)：ingress = $131072 \times 64 \times 4096 / (0.002 \times 1024 \times 8)$ = 2.10 GB/s per GPU。若 OSL = 64：ingress × 16 = 33.6 GB/s $\gt$ 跨 DC 6.25 GB/s——短输出 workload 的 ingress 才是跨 DC 的真约束。ISL 翻倍（KV 读主导）：TTL 翻倍、分子翻倍 → ingress 不变（命题 3）。

DistServe 的 sanity check（[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]，§3.3）：OPT-66B 单条 512-token 请求 KV ≈ 1.13 GB，10 rps → 11.3 GB/s = 90.4 Gbps 使开销不可见；InfiniBand 800 Gbps 或节点内 NVLink（A100 600 GB/s）下 negligible。这是实例级聚合账（非 per-GPU），与式 (6)(7) 的 per-GPU 口径互补：聚合需求 = per-GPU × 分片 GPU 数。

### 6.3 跨 DC P/D：压缩使可行，可靠性付代价

DeepInfer 官方披露（doc L97 逐字）：

> 「Because V4.1 Flash compresses the KV cache substantially, splitting P and D across data centers is technically feasible (at some cost in reliability, e.g., the fiber running between data centers is easily severed by construction work). Budget roughly 400 Gbps of interconnect bandwidth per 8 GPUs. In practice, DeepSeek's inference nodes are deployed in different but adjacent data centers.」

压缩是带宽需求的直接除数（2.4 §3 recap）。V4.1 主 KV per-16 FP4： $\kappa$ = 288 B/token/层（256 B E2M1 数据 + 32 B E4M3 scale，7.3 §4 格式账）， $L$ = 40 → $\kappa L$ = 11520 B/token。bf16 反事实： $\kappa$ = 1024 B（512 元素 × 2）、 $\kappa L$ = 40960 B/token。压缩比 = 40960/11520 = 3.56×。代入 $T$ = 32768：

| 口径 | $\kappa L$ （B/token） | KV 字节（ $T$ = 32768） | @6.25 GB/s 每 GPU | TTFT 预算 0.2 s |
| --- | --- | --- | --- | --- |
| V4.1 per-16 FP4（生产） | 11520 | 377.5 MB | 60.4 ms | 达标，余量 3.31× |
| V4.1 bf16（反事实） | 40960 | 1.342 GB | 214.7 ms | 超预算 |
| V3 系 FP8（对照） | 35136 | 1.151 GB | 184.2 ms | 超预算（full） |
| 8B GQA bf16（章脊） | 131072 | 4.295 GB | 687.2 ms | 超预算（full） |

压缩把跨 DC 传输从「超预算」翻成「达标且余量 3.3×」——这就是 doc「compresses substantially → technically feasible」的账。再对 $\mathrm{BW}^{\ast}$ ：V3 系 MLA FP8 的 $\mathrm{BW}^{\ast} = \kappa L F_{\mathrm{eff}} / (2 N_{\mathrm{act}}) = 35136 \times (1979 \times 10^{12} \times 0.4) / (2 \times 37 \times 10^9) = 0.376$ GB/s ≈ 3 Gbps（8.2 数值锚 recap，H100 FP8）；跨 DC 每 GPU 6.25 GB/s = 50 Gbps ≫ 3 Gbps（16.6×）→ 传输远优于重算，与 doc L97 自洽。V4.1 $\kappa L$ 更小（11520 vs 35136，3.05×）， $\mathrm{BW}^{\ast}$ 同向更低、余量更大。

> **坑**：把「跨 DC 可行」读成「跨 DC 无代价」。触发条件：只引带宽账忽略 doc 同句的可靠性括注；现象：以为压缩解决了全部问题。事实：doc 明说可靠性代价——DC 间光纤易被施工挖断，所以 DeepSeek 部署在**不同但相邻**的 DC（缩短光纤路径、降低单点故障概率），且 layer-wise 重叠 + 压缩共同压低暴露时间。规避：带宽账与可靠性账并列，跨 DC 选型须同时评估（8.5 容错）。

```
DC-A（prefill）                相邻 DC 间光纤                DC-B（decode）
┌──────────────┐         ~400 Gbps / 8 GPU            ┌──────────────┐
│ P 实例 ×n_P  │ ═══════ 压缩 KV（FP4）═══════════════ │ D 实例 ×n_D  │
│ V4.1 L=40    │         每 GPU 6.25 GB/s              │ Wide-EP      │
└──────────────┘         可靠性代价：光纤易断           └──────────────┘
```

## 7. 引擎接口：connector 与 disaggregation 的传输语义映射

同一套传输语义（§1），vLLM 与 SGLang 的接口长什么样？本节只交付「接口 ↔ 语义」映射——模块全量走读归 5.3 / 5.4 / 6.3（两层纪律，已走读片段不重复展开）。

vLLM（recap 5.3 §9 / 5.4）：`distributed/kv_transfer/kv_connector/v1/base.py` `KVConnectorBase_V1` L182 + `KVConnectorRole` L140（scheduler / worker 角色分件）；connector 族：nixl/（`NixlPullConnector` L374 = READ / `NixlPushConnector` L415 = WRITE）、mooncake/（`MooncakeConnector` L798）、mooncake_store/、lmcache_connector.py、hf3fs/、multi_connector.py（`MultiConnector` L142，组合多后端）、hisparse/、offloading_connector.py。实例编排经 `kv_role` + `kv_transfer_config`。Dynamo 视角：vLLM 用 `kv_transfer_params`（block IDs + remote worker 连接信息），prefill 同步、decode 等 prefill 完（8.4 §4）。

SGLang（recap 6.3 §4-6）：`disaggregation/base/conn.py` 四抽象——`KVArgs` L62（注册描述符）/ `KVPoll` L142（五态状态机）/ `BaseKVManager` L150 / `BaseKVSender` L191（send L209 / poll L240）/ `BaseKVReceiver` L266 / `BaseKVBootstrapServer` L339；backend mooncake / nixl 同构。bootstrap 时序 recap：decode 注册 → bootstrap 撮合 room → prefill `send_kvcache` RDMA → decode poll → PREBUILT forward（6.3 §4 图，不重绘）。

传输语义 ↔ 接口映射表（本章增量）：

| 传输语义（§1） | NIXL | vLLM connector | SGLang disaggregation |
| --- | --- | --- | --- |
| push（initiator WRITE） | `NIXL_WRITE` | `NixlPushConnector` L415 | `send_kvcache`（prefill 直写） |
| pull（initiator READ） | `NIXL_READ` | `NixlPullConnector` L374 | receiver 侧 READ 路径 |
| 地址协商 | metadata 控制面 | `kv_transfer_params` | `send_metadata`（decode 指定位置） |
| layer-wise / chunk | 逐层 xfer_dlist | chunk 提交 | `send_kv_chunk`（6.3 命题 2） |
| 注册 | `nixl_reg_dlist_t` | buffer 注册 | `KVArgs`（ptr/len/item_len/layer） |
| 完成 poll | `getXferStatus` | connector poll | `KVPoll` 五态 |

一条关键语义（6.3 §5 recap）：SGLang 的地址协商是 **pull 式询问、push 式数据**——decode 经 `send_metadata` 把预分配槽位索引发给 prefill，prefill 的 `send` 随后 RDMA WRITE 到那些远端地址。所以「push vs pull」要分控制面与数据面两层看：控制面谁问地址，数据面谁 post WRITE/READ。vLLM 的 NixlPush/Pull 则是数据面 initiator 的双模封装。

## 总结

| 论断 | 账 / 证据 | 回指 |
| --- | --- | --- |
| 传输三自由度：粒度（full/layer-wise）、方向（push/pull，initiator 视角）、时序（serialized/overlapped）；layer-wise 是 overlapped 的前提 | Splitwise §IV-C：<512 serialized / 大 prompt layer-wise；第二 token 开销 64%→16.5% | §1 |
| 逐层流水暴露时间 = $\max(\tau_x, t_{\mathrm{xfer}}-(L-1)\tau_c)$ ；计算主导时 = $t_{\mathrm{xfer}}/L$ | 命题 1；章脊 85.9 ms→2.68 ms（32× = $L$ ） | §2、式 (2)、练习 2 |
| egress per-GPU 完全隐藏需求恒等于 $\mathrm{BW}^{\ast}=\kappa L F_{\mathrm{eff}}/(2N)$ （线性口径），与 batch/ISL/GPU 数无关 | 命题 2；章脊 4.05 GB/s；计算主导 ⟺ BW ≥ BW* | §6.2、式 (6)、8.2 |
| ingress 在 KV 读主导 decode 下与 ISL 无关、∝ 1/OSL；短输出 workload ingress 飙升 | 命题 3；OSL 1024→64 时 2.10→33.6 GB/s | §6.2、式 (7)、练习 3 |
| NIXL = 插件化传输抽象（memory+storage 统一），核心 UCX+GDS，18 插件含 mooncake；GDR 是 UCX 下 GPU 直写路径非独立 backend | nixl.h agent API / descriptor 三型 / xfer vs reg dlist | §3 |
| Mooncake TE = Segment（进程级）+ Buffer（注册级）+ BatchTransfer；>64 KB 切 slice 多 NIC 池化，聚合 BW = Σ NIC BW | 4×400G = 200 GB/s；每层 134 MB → 2048 slice | §4、式 (3) |
| GDR：RDMA NIC 直写 HBM、不占 SM / copy engine；细粒度块拷贝 copy engine 差 | doc L191 逐字；bounce buffer 聚合小 fragment | §5、2.4 §4 |
| 跨 DC P/D：压缩使可行（V4.1 FP4 60.4 ms 达标余量 3.31×），可靠性付代价（光纤易断、相邻 DC） | doc L97；每 8 GPU 400 Gbps = 6.25 GB/s/GPU ≫ BW* 0.376 GB/s | §6.3、练习 4 |
| 预算约束的是暴露时间而非裸传输时间——P2D 可重叠，offload 取回不可 | 式 (5) BW_req 21.5 GB/s vs layer-wise 暴露 2.68 ms | §6.1、2.4 §3 |
| 硬件层（RDMA 协议 / NIC / PCIe 拓扑 / NVLink / NCCL）归 cluster 维度 | 边界声明 | §5 |

## 练习

1.（计算 · [10]）KV 字节与传输时间代入（章脊 8B GQA， $T$ = 32768、 $T_{\mathrm{hit}}$ = 0）：(a) 由 $\kappa = 2 H_{kv} d_{\mathrm{head}} \times 2$ 求 $\kappa$ 与 $\kappa L$ ；(b) 求 KV 总字节（GB 与 GiB）；(c) 求 400 Gbps（50 GB/s）下的 $t_{\mathrm{xfer}}$ 。校验：(a) 4096 B / 131072 B = 128 KiB；(b) 4.295 GB = 4 GiB；(c) 85.9 ms，与 §1/§2 一致； $\kappa$ 公式 recap 1.2 式 (3)。
2.（推导 · [20]）命题 1 证明闭环：(a) 补全归纳保持步的两情形（ $\tau_c \ge \tau_x$ 与 $\tau_x \gt \tau_c$ ），说明 $\max$ 项如何吸收为常数；(b) 代入 $\tau_c$ = 33.1 ms、 $\tau_x$ = 2.68 ms 求暴露时间；(c) 给边界反例： $\mathrm{BW}$ = 1 GB/s 时暴露时间是多少，为什么不再是被 $L$ 除的小尾巴。校验：(b) 2.68 ms；(c) 3.27 s，与式 (2) 命题 1 的 $\max$ 第二项一致；先预测 (c) 的量级再算。
3.（计算 · [30]）egress / ingress 全账（章脊，TP8）：(a) $B_P$ = 8、ISL = 4096，求 FTL 与 per-GPU egress，验证等于 $\mathrm{BW}^{\ast}$ ；(b) $B_D$ = 64、ISL = 4096、OSL = 1024、TTL = 2 ms，求 per-GPU ingress；(c) OSL = 64 时 ingress 是多少，是否超跨 DC 每 GPU 6.25 GB/s；(d) 先预测 ISL 翻倍（KV 读主导）时 ingress 的变化方向再代入验证。校验：(a) FTL 132.5 ms、egress 4.05 GB/s = BW*；(b) 2.10 GB/s；(c) 33.6 GB/s $\gt$ 6.25；(d) 不变（命题 3）；TTL roofline 校验见 §6.2。
4.（计算 · [30]）跨 DC 压缩账（V4.1 生产口径， $\kappa$ = 288 B、 $L$ = 40、 $T$ = 32768）：(a) 求 KV 字节；(b) 求每 GPU 6.25 GB/s 下的传输时间；(c) bf16 反事实（ $\kappa$ = 1024 B）的传输时间与压缩比；(d) TTFT 传输预算 0.2 s 时 FP4 是否达标、余量多少。校验：(a) 377.5 MB；(b) 60.4 ms；(c) 214.7 ms、3.56×；(d) 达标、余量 3.31×，与 §6.3 表及 doc L97 一致；格式账 recap 7.3 §4。

## 参考

- [NVIDIA, ai-dynamo/nixl](../references.md)：key: nixl，教学参照 + 官方实现（cb66ce83，2026-11-23 tarball C++ 深核）——docs/nixl.md Transfer Agent 三抽象；src/api/cpp/nixl.h nixlAgent L35 / registerMem L118 / prepXferDlist L184 / makeXferReq L257 / createXferReq L306 / postXferReq L345 / getXferStatus L355 / getLocalMD L512 / loadRemoteMD L543；nixl_descriptors.h nixlBasicDesc L33 / nixlStrideDesc L168 / nixlBlobDesc L226 / nixl_xfer_dlist_t L547 / nixl_reg_dlist_t L559；nixl_types.h NIXL_READ/WRITE L48；src/plugins/ 18 插件目录（含 mooncake）；examples/cpp/nixl_example.cpp 调用序列
- [Qin et al., 2024](../references.md)：Mooncake，arXiv:2407.00079，[A]（key: mooncake，继承 kvcache-ai/Mooncake 2cdbaeeb）——transfer engine Segment / BatchTransfer / 拓扑感知多 NIC 池化；index.md 678 行（>64 KB slice / SIEVE endpoint / fault handling）；transfer_engine.h TransferEngine L90 / openSegment L156 / registerLocalMemory L187 / submitTransfer L205 / getTransferStatus L305 / getTaskCongestionState L308；topology.h Topology L64 / selectDevice L109；transport.h TransferRequest OpCode L67；2026-11-23 contents API 核实
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——L97 跨 DC 压缩 KV 可行 + ~400 Gbps/8 GPU + 相邻 DC 可靠性代价；L191 GDR swap-in 逐字（copy engine 细粒度差、NIC 不占 SM）+ Entrypoint CHECK + 三段流水
- [Patel et al., 2024](../references.md)：Splitwise，arXiv:2311.18677，[B]（key: splitwise，仅作者模拟器）——§IV-C serialized vs layer-wise、<512 切换阈值、逐层异步重叠；§VI-A 第二 token 64%→16.5%、E2E 3%→0.8%、常数尾巴 A100 ~8 ms / H100 ~5 ms；InfiniBand 25-50 GBps/GPU pair
- [Mitra et al., 2025](../references.md)：Beyond the Buzz，arXiv:2506.05508，[B]（key: beyond-buzz，无官方 repo）——§5.1 egress/ingress 带宽公式（MHA/GQA 几何，本书 κ 代入）、per-GPU 归一只计实际分片 KV 的 GPU、FTL/TTL 术语、egress 随 ISL 降 / ingress ∝ 1/OSL、现有 DC 带宽足以支撑
- [Zhong et al., 2024](../references.md)：DistServe，arXiv:2401.09670，[A]（key: distserve）——§3.3 OPT-66B 512-token KV ≈1.13 GB、10 rps → 90 Gbps 使开销不可见、InfiniBand 800 Gbps / NVLink A100 600 GB/s negligible
- [Kwon et al., 2023](../references.md)：vLLM，arXiv:2309.06180，[A]（key: vllm-repo）——kv_transfer/kv_connector/v1/ KVConnectorBase_V1 L182 / NixlPullConnector L374 / NixlPushConnector L415 / MooncakeConnector L798 / MultiConnector L142；W3 同日 tarball 2026-11-23 再确认
- [Zheng et al., 2023](../references.md)：SGLang，arXiv:2312.07104，[A]（key: sglang-repo）——disaggregation/base/conn.py KVArgs L62 / KVPoll L142 / BaseKVManager L150 / BaseKVSender L191 / BaseKVReceiver L266 / BaseKVBootstrapServer L339；mooncake/nixl backend；W3 同日 tarball 2026-11-23 再确认
- [NVIDIA, 官方 datasheet](../references.md)：key: nvidia-datasheet，[C]（官方规格披露、非闭源）——H100 SXM bf16 dense 989.5 TFLOPS / FP8 1979 TFLOPS / HBM 3.35 TB/s；本章 F_eff 与 TTL roofline 代入
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：key: v41-flash-config，[W]——L=40、MLA 512 元素（448+64）；主 KV per-16 FP4 288 B 格式账 recap 7.3 §4；模型侧形式化归 models（V4.1 待补）
