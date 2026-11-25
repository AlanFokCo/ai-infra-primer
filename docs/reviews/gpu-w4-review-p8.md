# gpu W4 Part 8  adversarial review（T6 · fresh-context evaluator）

日期：2026-11-25。对象：gpu/08-nccl-collectives/ 8.1-8.4（NCCL 集合通信）。
输入：4 章 + spec §5 Part 8 / §4.3 符号 / §6.3 强制重推导 + training 4.1（recap 源）+ STYLE_GUIDE + rubric + R2 style sheet + gpu/references.md P8 + docs/research/gpu-w4-sources.md。
代码锚一手源：NCCL 2.32.7-1 fresh tarball commit 351f33fd087e0c4d0cfb99f1c61dc13da7996c28，解压 /tmp/gpu-w4-nvidia_nccl/nccl-351f33fd.../（version.mk L9-11 MAJOR 2/MINOR 32/PATCH 7 核实）。
方法：evaluator 自行 grep tarball file:line 全量核锚 + python3 -I 独立重算 + curl 一手 blog/docs/Crossref 对照。不继承写作上下文，未派发子代理。

## 0. 结论摘要

- **S1 = 0，S2 = 0，S3 = 8**（全部 minor：5 处代码锚行号偏移 + 1 处数学白名单 \div + 1 处措辞歧义 + 1 处 prose 裸 < 可选）。
- **强制重推导 8.2 ring 带宽最优性下界：PASS**（证明逻辑严密、单阶段界与总量 2× 严格分离、数值逐数闭合）。
- **强制重推导 8.3 ring/tree 交叉 M\*：PASS**（闭式正确、Δβ=0 无交叉、n=8 → 1.83 MB / 36.6 μs 独立重算闭合）。
- **recap 边界：PASS**（ring 消息量归纳 + α-β 均 recap training 4.1 不重推；下界证明 8.2 / 交叉 8.3 为 spec 强制增量重推导，未越界重推归纳）。
- **NCCL 全量锚点 pass：PASS**（8.4 ~75 锚 + 8.1/8.2/8.3 锚全量 grep 核实，内容/函数/数值全部正确；仅 5 处行号 minor 偏移，无错锚/错事实）。
- 一手来源（NCCL 2.4 blog / NCCL docs env / Crossref 三 DOI）curl 逐字核实 PASS。
- lint 4 章全 LINT OK；无 process-leak / AI 味 / 松散词；数学公式内 < 一律 \lt（裸 < 命中均在 code span，安全）。

**总体裁定：P8 四章事实与推导全部成立，无 S1/S2 阻断项；8 条 S3 累计下波清零（行号偏移建议本波顺手修）。**

## 1. Findings 表

| # | severity | 位置 | 断言（原文） | 证据（一手） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S3-1 | S3 | 8.1 §4 L85 | 「换优化器状态显存 `$\div n$`」 | STYLE_GUIDE §3.1 白名单「其他」行收 `\times/\setminus/\oplus/\oslash`，**未收 `\div`**；`\div` 是 `\`+字母不受反转义陷阱、GitHub 渲染正常，但属白名单外命令 | 改 `$M/n$` 式写法或「除以 $n$」；或在 STYLE_GUIDE 白名单补 `\div`（后者属规范变更非章改）。单处、渲染无碍，minor |
| S3-2 | S3 | 8.4 §6 L110 + 练习 3 L235 | 「`static_assert(sizeof == 16)`（L25）」「校验：static_assert sizeof==16（ll__types.h L25）」 | tarball src/include/nccl_device/impl/ll__types.h：union L14、注释 L15-16、字段 L17-22、`};` L25、**static_assert 实在 L26**；union 范围 L14-26 正确 | 两处 static_assert 行号 L25→**L26**（union 范围 L14-26 不动） |
| S3-3 | S3 | 8.4 §7 L165（tree.cc bullet） | 「busBw ×0.92（L34）」 | tarball src/tuning/tree.cc：`if (c==ncclFuncAllReduce) busBw = std::min(busBw*.92,...)` 实在 **L33**（L34 是 `if (proto==NCCL_PROTO_LL){`） | L34→**L33** |
| S3-4 | S3 | 8.4 §7 L165（tree.cc bullet） | 「LL ×1/3.8（L36）」 | tarball src/tuning/tree.cc：`busBw = std::min(busBw*1.0/3.8, llMaxBw)` 实在 **L35**（L36 是 `}`） | L36→**L35** |
| S3-5 | S3 | 8.4 §1 L9（版本锚） | 「makefiles/version.mk L9-12 MAJOR 2 / MINOR 32 / PATCH 7」 | tarball makefiles/version.mk：MAJOR L9 / MINOR L10 / PATCH L11 / **SUFFIX L12（空）/ PKG_REVISION:=1 L13**；三版本号在 L9-11，「2.32.7-1」的「-1」源自 L13 PKG_REVISION | 改「L9-11 MAJOR 2/MINOR 32/PATCH 7（+L13 PKG_REVISION 1）」；L9-12 含空 SUFFIX 行且漏 -1 出处，minor |
| S3-6 | S3 | 8.3 §6 L99 | 「NCCL_ALGO 枚举，plugin/nccl_tuner.h L28-34」（含 PAT） | tarball src/include/plugin/nccl_tuner.h：TREE0 L29…NVLS_TREE5 L34、**PAT6 L35**；L28-34 漏 PAT（L35）。8.4 §7/参考已正确写 L28-35 | 8.3 该处 L28-34→**L28-35**（与 8.4 对齐） |
| S3-7 | S3 | 8.3 §3 L41 / 8.4 §3 L50 | 「第二棵偶数 rank 用 mirror…奇数 rank 用 shift」 | tarball src/graph/trees.cc L94 `if (nranks % 2 == 1){shift}else{mirror}`——分支按 **nranks 奇偶**（rank 总数），非单个 rank 编号奇偶；「偶数 rank」字面易误读为「编号偶的 rank」。前文 blog 逐字「if nranks is even」可恢复语义 | 措辞改「nranks 偶用 mirror、nranks 奇用 shift」消歧；非事实错（代码语义正确），minor |
| S3-8 | S3（可选） | 8.4 §7 L164 | 「≥sm80 取 1.4、<sm80 取 1.9」 | prose 裸 `<sm80`：GFM 对非识别标签渲染为字面文本（安全，非数学公式内 <，不触发 P26 tag-swallow）；数学公式内 < 已全用 \lt | 可选美化：prose 改「低于 sm80」或全角＜；不修亦无渲染碍 |

**已核正、无 finding 的「已知偏移」项**：task 列出的 graph.h L119-147 偏移——8.4 §2/参考现写 **L120-148**，tarball PATH_LOC L120…PATH_SYS L148 逐值核实**正确**（writer 已修，无需再动）。

## 2. Verification log（强制）

### 2.1 NCCL 全量锚点 pass — **PASS**（fresh tarball 351f33fd，grep file:line 全量）

8.4 ~75 锚 + 8.1/8.2/8.3 锚逐项 grep 核实。函数/内容/数值全部正确；行号偏移见 §1（S3-2…S3-6）。

**init.cc（8.4 §2）**：ncclCommInitRank L3071✓ / ncclCommInitRankDev L2975✓ / ncclCommInitRankFunc L2229✓ / ncclCommInitRankAsyncJob L2132✓ / bootstrapInit L2314✓ / initTransportsRank L1219（调用 L2320）✓ / ncclTopoGetSystem L1447（亦 L1451）✓ / ncclTopoComputePaths L1453✓ / ringGraph ncclTopoCompute L1488（maxChannels=MAXCHANNELS/2 L1487）✓ / treeGraph L1496（channels=ring nChannels L1494-1495）✓ / collNetChainGraph L1513 / collNetDirectGraph L1515✓ / nvlsGraph L1525✓ / ncclTransportP2pSetup L1957✓ / devCommSetup L2076·L714（ncclCudaMemcpyAsync L847）✓ / DEFAULT_LL_BUFFSIZE L1008·LL128 L1010·DEFAULT_BUFFSIZE(1<<22) L1011✓。

**graph/（8.4 §2/§3，8.2 §4，8.3 §3）**：topo.cc ncclTopoGetSystem L1994 / GetSystemFromXml L1072 / nvlBw=ncclTopoNVLinkBw L944✓；topo.h SM90_NVLINK_BW 20.6 L24 / SM100 40.1 L26 / ncclTopoNVLinkBw 分派 L315-323✓；paths.cc PATH_PHB L104 / PATH_PXN L243 / p2pLevel=PATH_NVL L341✓；search.cc ncclTopoSearchRec L331 / SearchRecGpu L357 / ncclTopoCompute L1245✓；graph.h PATH 梯子 L120-148（LOC0/NVL1/NVB2/C2C3/PIX4/PXB5/P2C6/PXN7/PHB8/SYS9 逐值✓）/ ncclTopoGraph L173（pattern/minChannels/nChannels/bwIntra/bwInter/latencyInter/typeIntra 字段齐✓）；rings.cc ncclBuildRings L29 / loopback 校验 L51 / 全 rank 位图校验 L55-60✓；trees.cc ncclGetBtree L32 / ncclGetDtree L90 / mirror-odd-shift 注释+12/13-rank ASCII 图 L69-88（章内 12-rank 图逐字匹配✓）/ mirror=nranks-1-rank·shift=(rank-1+nranks)%nranks✓；connect.cc ncclTopoPreset L22 / ncclTopoPostset L395✓。

**transport（8.4 §4/§5，8.1 §6）**：transport.cc ncclTransports[]={p2p,shm,net,collNet} L15-20（NVLS 不在表✓）；p2p.cc p2pCanConnect L130 / p2pMap L358 / cudaIpcOpenMemHandle(cudaIpcMemLazyEnablePeerAccess) L331 / p2pSendSetup L403 / useRead L410（useMemcpy→read=0 L412）/ p2pSendProxySetup L697✓；shm.cc shmCanConnect L61 / shmSendSetup L88 / shmRecvSetup L121✓；net.cc sendSetup L296 / recvSetup L356 / sendProxyProgress L1327（posted/transmitted/done L1337·1349-1351，maxDepth=min(NCCL_STEPS,NCCL_SHARED_STEPS/nsubs) L1346，NCCL_SHARED_STEPS=16 L666）/ recvProxyProgress L1496 / GDRDMA 日志 L325 / GDRCOPY_SYNC_ENABLE 默认1 L338 / FLUSH_ENABLE 默认0 L340✓；coll_net.cc canConnect L143 / sharedBuffersInit L416✓；nvls.cc ncclNvlsInit L246 / cuMulticastCreate L270 / MULTICAST_SUPPORTED L271 / 「NVLS multicast support is %savailable」L277 / ncclNvlsTreeConnect L283 / ncclNvlsSetup L494✓。

**proxy.cc（8.4 §5）**：ncclProxyPost L481 / progressOps L801 / ncclProxyProgress(void*) L956✓。

**device/（8.4 §6/§9）**：ll__types.h union L14 / 注释「Flags follow data…」L15-16 / 字段 L17-22 / **static_assert sizeof==16 L26**（章标 L25→S3-2）/ NCCL_LL_LINES_PER_THREAD 8 L29✓；ll128__types.h LINESIZE128 L13 / LINEELEMS16 L14 / DATAELEMS15 L15 / MAX_NTHREADS640 L17 / ELEMS_PER_THREAD120 L18 / SHMEM_ELEMS_PER_THREAD8 L20✓；prims_ll.h readLL L102 / ld.volatile.global.v4.u32 L108 / spin flag1==flag&&flag2==flag L113✓；prims_ll128.h FLAGTHREAD=LINEELEMS-1 L10✓；prims_simple.h Primitives<ProtoSimple> 特化 L20 / waitPeer L104 / genericOp L179✓；device.h NCCL_STEPS8 L26 / MAXCHANNELS64 L80 / NCCL_MAX_TREE_ARITY3 L163 / NCCL_MAX_NVLS_ARITY32 L192 / ncclProtoGrainSize L304-310（LL16/LL128=WARP_SIZE×8/16×15×8=1920/Simple512✓）；all_reduce.h runRing L14（FanSymmetric<1> L31，directRecvReduceCopyDirectSend postOp=true L64）/ runTreeUpDown L86（归约 FanAsymmetric<3,1> L97，广播 FanAsymmetric<1,3> L121）/ runTreeSplit L146（70%/30% 注释 L161-162）/ NVLS RunWorkColl L390（scatter FanAsymmetric<0,32> L421）/ ProtoSimple dispatch L229-244（tree 多数 CUDA 走 split：#if CUDART 11020-11040&&sm80→updown #else→split✓）/ LL·LL128 dispatch L762-788（ring+treeSplit，无 treeUpDown✓）；common.cu ncclDevKernel_Generic L23（__global__ args4K）/ common.h ncclDevFuncTable L29-31 / RunWorkBatch L303 / ncclKernelMain L420✓；generate.py all_protos L16 / all_algos 七种 L17✓。

**tuning/（8.4 §7，8.3 §4/§6，8.2 §5）**：tuning_general.cc ncclTuningGetTime L58（time=lat×latCount×precision_ratio + nBytes/(1000×bw)×precision_ratio L70-71 逐字✓；latCount ring=numPipeOps·tree=DIVUP L59；FP8 ring&&nRanks>8→1024 L65-67，注释「Relegate fp8 reduction trees…」L61-62✓）；tuning.cc ncclTuningSelectBestTuning L153（注释「lowest simulated time/cost…best」L151✓）；cost_model.cc isLL128Enabled L125 / baseLatencies Ring{6.6,14.0,8.4} L152（Simple8.4−LL6.6=1.8✓）/ hwLatencies NVLink Ring{0.6,1.9,3.4} L164·PCI{1.0,2.5,5.7} L174·NET{2.7,4.0,14.0} L184（3.4/5.7/14.0✓，14/3.4=4.1×·5.7/3.4=1.7×✓）/ NCCL_PROTO·ALGO env parseList L365-379✓；ring.cc LL min(llMaxBw,busBw×0.5) L40 / LL128 busBw×0.92(注释120/128) L43 / generalBandwidths=busBw×nRanks/nSteps L47（n=8 ×8/14=0.571✓）/ 延迟项(nSteps−nInterSteps)intraLat+nInterSteps·interLat L86-87 / ringSimplePlateau L109-139（多节点AR或Blackwell单节点NVLink AR/AG/RS，plateauFactor <80?1.9:1.4 L119✓）；tree.cc 仅AllReduce hard-disable L23-28 / **busBw×0.92 L33**（章标 L34→S3-3）/ **LL×1/3.8 L35**（章标 L36→S3-4）/ LL128 单节点7/9·多节点120/128 L38 / generalBandwidths=busBw×0.5 L44 / 延迟项2×((ppn−1)intraLat+log2i(nNodes)interLat) L57-58 / treeCorrectionFactor[proto][logSize] L81（logSize=log2i(nBytes>>6) L77）✓；plugin/nccl_tuner.h 算法 TREE0…PAT6 L29-35（8.4 L28-35✓，8.3 L28-34 漏 PAT→S3-6）/ 协议 LL0/LL1281/SIMPLE2 L39-41✓。

**enqueue/（8.4 §7）**：enqueue.cc LL trafficBytes×4 L554（亦 L759 trafficPerByte×4）✓；task_posttuning.cc task->algorithm/protocol L547-548✓；task_pretuning.cc fillCollTuningInput L24（调用 L248）✓。

**nccl.h.in（8.1 §7）**：ncclReduce L561 / ncclBroadcast L589 / ncclAllReduce L602 / ncclReduceScatter L618 / ncclAllGather L635 / **ncclAlltoAll L648（小写 t 逐字✓，非 ncclAllToAll）** / ncclGather L663 / ncclScatter L678 / ncclSend L739 / ncclRecv L758 / ncclGroupStart L873 / ncclGroupEnd L883✓。doc comment 逐字：「called separately for each communicator」L540-541 / 「return when operations have been enqueued on the CUDA stream」L543 / in-place AR sendbuff==recvbuff L600·RS recvbuff==sendbuff+rank×recvcount L616·AG sendbuff==recvbuff+rank×sendcount L633 / group「need to progress concurrently…fused within ncclGroupStart」L736 / groupEnd「only guarantees…enqueued…not…effectively done」L854-856✓。count 约定：AlltoAll「sends count values to all other devices…sendbuff+j*count」L643-646（count=每 peer✓）/ RS「sendcount=nranks*recvcount」L613 / AG「recvcount=nranks*sendcount」L630✓。

**缓冲尺寸数值（8.4 §6）**：LL=8×512×8×16=524288=512 KiB✓ / LL128=120×640×8×8=4915200=4.6875 MiB✓ / Simple=1<<22=4 MiB✓（常数 NCCL_LL_MAX_NTHREADS512 device.h L87 / LL128_MAX_NTHREADS640 / ELEMS_PER_THREAD120 核实）。

### 2.2 强制重推导 8.2 ring 带宽最优性下界 — **PASS**

**独立证明审查**（命题 1：Σ_r S_r ≥ 2(n-1)M）：
- 切 n chunk（各 M/n），固定 chunk c，r_c = 首个持齐全和的 rank（良定义）。
- 归约树界：每个 r≠r_c 的贡献须离开 r 至少一次；取首次离开传输 t_r，源互异（一传输一源）→ n-1 次不同传输；每次携带 chunk c 部分和 ≥ M/n（不可压缩）→ 归约类 ≥ (n-1)M/n。**审查关键**：t_r 必发生在 r_c 持齐之前（r_c 持齐意味着已含 r 的贡献，故 r 的贡献必先离开），故全部 t_r 落在「全和存在前」时间窗，论证严密✓。
- 广播树界：r_c 持齐后其余 n-1 rank 各须 ≥ M/n 全和流入 → 广播类 ≥ (n-1)M/n✓。
- 不相交：归约类（全和存在前，携部分和）与广播类（全和存在后，携全和）按时间互斥；同一 chunk 的一次传输不能既携部分和又携全和 → 字节可加，每 chunk ≥ 2(n-1)M/n✓。
- 求和 n chunk → 总流量 ≥ 2(n-1)M✓。
- 推论（负载均衡）：T_comm ≥ max_r S_r·β，max_r S_r ≥ (1/n)Σ S_r ≥ 2(n-1)/n·M；ring 每步每 rank 等量（天然均衡）达界✓。
- **单阶段界单独归属**：归约树界单独 = reduce-scatter 总流量下界 (n-1)M，广播树界单独 = all-gather (n-1)M；负载均衡每 rank (n-1)/n·M。章 §3 口径注（硬规则）+ 坑块明确「单阶段 (n-1)/n·M ≠ all-reduce 总量 2(n-1)/n·M，混用少算一半」——**单阶段与总量 2× 严格分离，未混✓**（task 重点核查项）。
- 弱界 S_r ≥ M 漏转发流量，n=8 差 1.75×，不足闭合✓。

**独立数值**（python3 -I，n=8/M=1 GB）：总下界 2×7×1=**14 GB**（章 14✓）；每 chunk 2×7/8=**1.75 GB**（章 1.75✓）；每 rank 下界 **1.75 GB** = ring 实发✓；单阶段 7/8=**0.875 GB**（章 0.875✓）；弱界比 1.75/1=**1.75×**✓。n=4 最小例：归约 3M/4 + 广播 3M/4 = 6M/4，4 chunk 求和 6M=2(4-1)M✓。

### 2.3 强制重推导 8.3 ring/tree 交叉 M\* — **PASS**

**独立代数**：T_ring=2(n-1)α+2(n-1)/n·M·β_ring,eff；T_tree=2⌈log2 n⌉α+2(n-1)/n·M·β_tree,eff。n 为 2 幂时 ⌈log2 n⌉=log2 n。令相等：2(n-1-log2 n)α = 2(n-1)/n·M·Δβ → **M\* = n(n-1-log2 n)α/((n-1)Δβ)**（章式 (2) 逐项一致✓）。
**理想模型 Δβ=0**：分母零、M\* 无定义（无交叉）；T_tree−T_ring=2(log2 n−(n-1))α<0（n>2），tree 延迟恒赢、带宽恒等，无 ring 反超有限 M\*✓（章「禁从纯 α-β 凭空解出」口径正确）。

**独立数值**（n=8/α=2 μs/β_ring=1/370.8 GB/s/β_tree=1/130 GB/s 教学假设）：
- Δβ = 7.6923e-12 − 2.6969e-12 = **4.9954e-12 s/B**（章 4.995e-12✓）。
- M\* = 8×(7-3)×2e-6/(7×4.9954e-12) = **1.8302e6 B ≈ 1.83 MB**（章 1.83✓）。
- 闭合验证 M=M\*：T_ring = 28 + 1.75×1.8302e6×2.6969e-12×1e6 = **36.64 μs**；T_tree = 12 + 1.75×1.8302e6×7.6923e-12×1e6 = **36.64 μs**（章 ≈36.6✓，两式逐数闭合）。
- 两侧判定：M=0.5 MB → T_ring 30.4 / T_tree 18.7 μs（tree 赢✓）；M=10 MB → T_ring 75.2 / T_tree 146.6 μs（ring 赢✓）。

### 2.4 Recap 边界 — **PASS**

- 8.1 §2 / 8.2 §2：ring 每 rank 消息量式 (1) = recap training 4.1 式 (2)（d→n 翻译），α-β 式 (2) = recap training 4.1 式 (3)；两章均显式声明「归纳证明已交付 training 4.1 §2.2（scatter-reduce 不变式：第 s 步累加 s+1 份贡献），本章不重推」。正文只给两阶段计数摘要（n-1 步×M/n），**未复制基例/保持/终止三段归纳证明**✓。
- 8.3 §2：recursive halving/doubling 步数 = recap training 4.1 练习 3，「步数推导已交付…本章不重推，只取结论」✓。
- **增量重推导（spec §6.3 强制，非 recap）**：8.2 §3 带宽最优性下界（命题 1+推论+单阶段界）、8.3 §4 交叉 M\*（命题 2）——按本书命题-证明骨架独立重推，经典（Patarasuk-Yuan/Sanders）只作出处✓。
- 对照 training 4.1 原文：式 (2) C_rank=2(d-1)/d·M、式 (3) T_ring=2(d-1)α+2(d-1)/d·Mβ、§2.2 归纳命题、§2.1 星型 2(d-1)M、§2.2「大消息带宽项相同树形赢延迟」、练习 3——recap 源逐项存在，翻译 d→n 附口径注✓。**无越界重推归纳（若重推 = S2，未触发）**。

### 2.5 一手来源事实核查 — **PASS**（curl 逐字）

- **NCCL 2.4 blog**（developer.nvidia.com，curl 200，237123 B）逐字核：「half or less ranks in a binary tree are nodes and half (or more) ranks are leaves」✓ /「There might be one rank which is a leaf on both trees but **no rank is a node on both trees**」✓ /「each rank will at most receive half of the data twice and send half of the data twice, which is **as optimal as rings in terms of data sent/received**」✓ /「NCCL **automatically switches back to rings when that pattern results in greater bandwidth**」✓ /「bandwidth degrades a bit when we **cross L3 switches** in the InfiniBand fabric」✓ /「**up to 180x improvement at 24k GPUs**」✓ / 2D ring「**performs a reduce-scatter operation inside the node, then multiple all-reduce operations between nodes, then an all-gather operation inside the node again**」✓（8.2 §7 / 8.3 §2/§3/§4 引句全部逐字匹配）。
- **NCCL docs env.html**（curl 200，236288 B，版本 2.32.3）：「enabling LL128 on platforms that don't support it can lead to data corruption」✓（8.4 §6）/「NVLink SHARP is available in **third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures**」✓（8.4 §8）。
- **Crossref 三 DOI**（api.crossref.org 200）：Patarasuk-Yuan 10.1016/j.jpdc.2008.09.002 = JPDC **69(2):117-124**「Bandwidth optimal all-reduce algorithms for clusters of workstations」✓ / Rabenseifner 10.1007/978-3-540-24685-5_1 = LNCS **pp 1-9**「Optimization of Collective Reduction Operations」✓ / Sanders 10.1016/j.parco.2009.09.001 = Parallel Computing **35(12):581-594**「Two-tree algorithms for full bandwidth broadcast, reduction and scan」✓。

### 2.6 其余数值独立重算 — **PASS**

- 8.1 M_c=nα/β=8×2e-6×370.8e9=**5.9328 MB**（章 5.93✓）；M/M_c=1e9/5.9328e6=**168.55**（章 168.6✓）；训练 13 GiB/M_c=**2352.8**（章 2353✓）；decode 614400/M_c=**0.1036**（章 0.104✓）；ring 带宽项 **4.7195 ms**/延迟 **0.028 ms**/合计 **4.7475 ms**（章 4.72/0.028/4.75✓）。decode M=2×60×5120=614400 B✓。
- 8.2 多 ring 式 (6)：k=4 带宽 4.72/4=**1.18 ms**✓ / k=8 **0.59 ms**✓ / 延迟恒 0.028✓；每字节延迟摊薄 28e-6/1e9=**2.8e-14 s/B**，/β=**1/96.3**（章 1/96✓）；n=10000 系数 2×9999/10000=**1.9998**✓ / 延迟 2×9999×2e-6=**39.996 ms≈40**✓。
- 8.3 双树命题 1：单树内部节点 3D（上 D+下 2D）/叶子 D；互补 3D+D=4D=2M✓；n=8 双树≈2 GB vs ring 1.75 GB 差 2M/n=**0.25 GB**✓。延迟 n=8 tree12/ring28 μs（3/7✓）；n=1024 40/4092（**102.3×**✓）；n=24576 ⌈log2⌉=**15**✓ / tree60 μs / ring98300 μs≈98 ms / 理想比 **1638.3×**✓（blog 180× 端到端口径分离✓）。
- 8.4 命题 1 效率：η_LL=8/16=**0.5**✓ / η_LL128=120/128=**0.9375**✓ / Simple→1✓；C=1.75 GB 带宽项 Simple **4.72**/LL128 4.72/0.9375=**5.03**/LL 4.72/0.5=**9.44 ms**✓。命题 2 交叉：C\*=1.8e-6×370.8e9×0.5/0.5=**667440 B=667.4 KB**✓ / M\*=C\*×8/14=**381.4 KB**✓ / 闭合 T_LL=6.6+3.6=**10.2**=T_Simple=8.4+1.8✓；M=64 KB→LL 赢 / 1 MB→Simple 赢✓。LL128 warp grain 32×8/16×15×8=**1920 B**✓。
- 8.4 §7 单位对账：nBytes/(1000×bw)，bw GB/s → nBytes/bw=ns，/1000=μs✓；1.75e9/(1000×370.8)=**4719.5 μs≈4.72 ms**✓。

### 2.7 教学审查 — **PASS**

- 符号 §4.3：4 章章首均声明 n=rank/GPU 数（与 N 参数量/training d 区分，d→n 翻译注）、α=消息延迟（非 inference 接受率）、β=每字节时间、M=消息字节、T_ring/T_comm 下标✓。N/B 不占用（recap 数值以文字「参数量 7B」「batch 60」给出）✓。
- a2a 每 rank vs 每 peer：8.1 命题 1 严格区分 (n-1)/n·M（每 rank）vs M/n（每 peer），自环排除 + 边界反例（含自环多计 M/n=125 MB）✓。
- 三临界消歧：C\*（8.4 协议间 667.4 KB/M\*≈381.4 KB）≠ M\*（8.3 算法间 1.83 MB）≠ M_c（8.1 ring 内部 5.93 MB）——8.1 §5 / 8.3 §4 / 8.4 §6 三处口径注互链✓。
- 推导路线句（R2-5）：8.2 命题 1「切 chunk→每 chunk 归约树界+广播树界→不相交相加→求和，四步」✓ / 8.3 命题 2「令两式相等→延迟项差=带宽项差→解 M，三步」✓ / 8.4 命题 1/2 路线句✓。命题-证明骨架（R2-4）+ 定义即配例+边界反例（R2-6，8.1 命题 1 自环反例）✓。
- 前置缺口：无（recap 源 training 4.1/inference 8.1/7.x 均纯文字指引，下游 8.4/9.2/10.x 指针清晰）；越界检查：无 serving 调度/K8S 内容（fabric 工程归 10.2、GDR 归 9.2 两层纪律声明✓）。

### 2.8 规范审查 — **PASS**（含 1 处白名单 S3）

- lint：4 章全 **LINT OK**（禁令词/模板小节/metadata/arXiv/证据等级/$$ 配平）。
- 篇幅（CJK）：8.1=4333 / 8.2=5807（【重】≥5500 软目标✓）/ 8.3=3616 / 8.4=4287，均 ≥3500✓。
- 练习：16 题全 `（类型 · [难度]）` 格式，难度非递减（[10]/[20]/[30]/[30]），≥1 [10]/[20]✓，≤1 [40]✓，每计算/代码题带校验锚✓，每章 ≥1 题证正文命题✓，先预测后算（8.1 ex2/8.2 ex2/8.3 ex2/8.4 ex2）✓。
- 总结表回指列（R2-12）：4 章总结表每行含式 N/§N/练习 N 回指✓。
- 数学渲染：公式内 < 一律 \lt（8.1/8.3/8.4 核实✓）；反转义陷阱（\, \; \! \{ \} \% \#）无数学内命中（grep 命中均在 C 代码 fence/span）✓；display 用 \ast 代 *（8.2/8.3 M^{\ast}、8.4 C^{\ast}）✓；表格内无裸 | 切列✓。**\div 白名单外**（8.1 L85，S3-1，渲染无碍）。
- prose 裸 <：仅 8.4 §7 L164「<sm80」（GFM 字面渲染安全，S3-8 可选）；其余裸 < 均在 code span（Primitives<…>/FanAsymmetric<…> 等，安全）。
- process-leak：无 TODO/FIXME/待核/writer/evaluator/T6 泄漏✓。AI 味禁令词/松散词/营销形容词：无✓。历史叙事：无（经典只作 citation 出处，年代不进正文）✓。
- 证据分级：NCCL repo 教学参照（官方实现）/ NCCL docs [C] 官方规格披露·一手权威 / blog [C] 开放技术博客 / 经典 [B] 老经典无 arXiv——标注齐✓（spec §4.5 豁免句覆盖，[C] 官方规格不判弱证据）。

## 3. 覆盖率与残留

- NCCL 锚点覆盖率：8.4 参考清单 ~75 锚 + 正文锚全量 grep（init/graph/transport/proxy/tuning/device/enqueue/nccl.h.in 全覆盖）；8.1 API 12 锚 + doc comment、8.2 rings/tuning_general/device.h、8.3 trees/tree/tuning/nccl_tuner 全核。
- 一手来源覆盖：blog 7 引句 + docs 2 引句 + Crossref 3 DOI 全 curl 逐字。
- 重推导覆盖：spec §6.3 W4 子集 8.2 下界 + 8.3 M\* 全覆盖（独立证明审查 + python3 -I 数值）；附加 8.1 M_c / 8.4 协议效率与交叉 / 双树 / 延迟数值全重算。
- 残留（非本波 scope，沿 research）：PCI-SIG 403 / NVLink Switch per-chip 拆分 / B200 SM 数 / JEDEC 403——与 P8 无关，不影响本章事实。
- 未独立核（沿用 research 已核记录）：7.1/7.2/7.3 硬件规格（370.8=18×20.6、450/50=9×、NVLS 每 rank M）为 P7 已审锚，本波作 recap 口径引用，数值内部自洽✓。

## 4. 修复建议优先级

S3-2/S3-3/S3-4/S3-5/S3-6（5 处行号偏移）建议本波顺手修（纯锚点精度，改动小、无技术含义变化）；S3-1（\div）改写法或规范补白名单二选一；S3-7（mirror/shift 措辞）/S3-8（prose <）可选。全部 S3 累计下波清零（spec §6.4）。无 S1/S2，P8 可进 commit gate（图 QA 另行）。