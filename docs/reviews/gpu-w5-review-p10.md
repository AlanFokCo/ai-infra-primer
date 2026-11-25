# gpu W5 adversarial review — Part 10（机间 fabric 与大规模通信）

- **Evaluator**: fresh-context adversarial（不继承写作上下文，未派 subagents），read-only（仅写本报告）。
- **Date**: 2026-11-25。
- **Scope**: `gpu/10-fabric-and-scale-communication/01-fabric-architecture-topology.md`（10.1）、`02-topology-aware-collectives-sharp.md`（10.2）、`03-ep-alltoall-implementation.md`（10.3，DeepEP 双锚模块走读）、`04-point-to-point-sequence-parallel.md`（10.4）。
- **Authority**: spec `docs/superpowers/specs/2026-11-24-gpu-dimension-design.md` §5 P10 / §4.3 符号表 / §6.3 强制重推导；T1 `docs/research/gpu-w5-sources.md` §5-8；`gpu/references.md` P10；recap 源 training 5.2/5.4、inference 3.3/7.4、models 4.6、gpu 7.1/7.2/8.1/8.2/8.4/9.2；fresh tarballs `/tmp/gpu-w5-{deepep-v121,deepep-main,nccl}`。
- **Method**: 全部关键数值独立 Python（`-I`）重算（172 项 + 构造性路由 120 trial + zigzag 恒等 5 度数 + 双缓冲仿真 3 regime，见 verification log）；DeepEP 双锚 + NCCL 代码锚 file:line 逐一对 fresh tarball（~60 锚）；recap 边界逐句对兄弟源；3 项官方逐字 live WebFetch 再核；`lint-chapter.sh` + `lint-math-boundary.py` 机械门 + display/prose/leak/CJK 独立扫描；refs citedBy 程序化对账。
- **Diff package**: `.superpowers/sdd/2026-11-25-gpu-w5-rdma-fabric/review-a2ef471..2af2606.diff`（P10 hunks；含 T3/T5 fix round 1：10.1 交叉引 7.1§5→§4 + Clos/torus 符号局部声明、10.4 练习3(d) 每步max锚对齐）。

## 结论速览

| 项 | 结果 |
| --- | --- |
| S1（fatal） | **0** |
| S2（important） | **0** |
| S3（minor） | **7**（2 锚偏移 / 1 容差措辞 / 1 citedBy 回写组 / 1 术语 / 2 可选 tightening） |
| 强制重推导（10.1 fat-tree / 10.3 a2a 拥塞 / 10.4 overlap / 10.2 hier+SHARP） | **全 PASS（172/172 + 构造性扩展）** |
| Recap 边界（10.2←8.4；10.3←training 5.4+inference 3.3/7.4；10.4←models 4.6+training 5.2） | **PASS（不重推）** |
| DeepEP 双锚（机制 v1.2.1 / 接口 main V2.5） | **PASS（NVSHMEM 全移除 + IBGDA 仅注释 live grep 核实，无 V1-当-current 混读）** |
| CP=5.2 指针 | **PASS（training 5.5 误引 = 0；唯一 "5.5" 命中为数值 5.50 ms）** |
| 代码锚抽查 | **~60 锚命中，2 处 off-by-one（S3-1/S3-2）** |
| lint（chapter + math-boundary） | **4/4 OK** |
| CJK（≥3500） | 10.1=5146 / 10.2=4814 / 10.3=4583 / 10.4=5180 **PASS** |
| display 9 禁形 / prose \lt / bare <= / process-leak | **0 / 0 / 0 / 0** |
| 练习 3(d) 锚一致性（brief 专项） | **PASS（option-b：每步max锚已删，保留总量 9u'，与 §4 推论自洽）** |

**总体**：四章数值脊柱全部独立重推闭合，fat-tree 收敛比/规模账、a2a incast、Ring Attention overlap 阈值与三模式量账无一 FAIL；recap 边界干净（增量落点与 spec §5 关键推导逐一对应，8.2 §7 明确把分层精确账让渡 10.2）；DeepEP 双锚分离严格、版本漂移硬注到位。无 S1/S2。7 项 S3 均为锚偏移/措辞/回写类小项，T7 修复成本低。

---

## 1. 强制重推导 verification log（独立 Python，`python3 -I`）

脚本 `/tmp/gpu-w5-p10-recompute.py`（172 项 check + 构造性路由/仿真扩展）。每项 = 独立重算值 vs 章内值，容差 1%（另注者除外）。**不信任章内数字，全部从口径重新推。**

### 10.1 fabric 收敛比 / 规模 / rail（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| ρ=k_down/k_up；1:1 k144 72/72；3:1 k64 48/16 | ρ=1 / ρ=3 | 同 | PASS |
| 命题 1 必要性：leaf 对齐割 cap/demand=k_up/k_down | 1（1:1）/ 1/3（3:1） | 同 | PASS |
| 命题 1 充分性：König 边着色构造路由（k=4/8/16，各 30 随机排列） | 每 leaf 每色 ≤1 出流、每 (spine,dst-leaf) 每色 ≤1 入流，Δ≤k/2 色数够用 | 「可重排无阻塞」 | PASS（构造性） |
| 扩展：非 leaf 对齐均衡割（interleaved even/odd NIC 划分，k=8/16 各 30 trial） | 着色路由仍全链路 ≤1 流；3:1 interleaved 割 cap/demand=0.667，leaf 对齐割=1/3 为 min（二分定义取 min） | 全二分 iff k_up≥k_down | PASS |
| 命题 2 规模 k²/2：k144→10368、k64→2048；可达构造（k leaf×k/2 spine，端口计数闭合） | 10368 / 2048 | 同 | PASS |
| 三级 pod 计数 k³/4：k64→65536；core (k/2)²=1024；/两级=32× | 65536 / 1024 / 32 | 同 | PASS |
| pattern k^t/2^(t-1)：t=2→k²/2、t=3→k³/4 | 闭合 | 同 | PASS |
| 超售 3:1 k64：16 spine×64 口→64 leaf，NIC=64×48=3072（+50%），二分 16/48=1/3 | 3072 / 1.5 / 0.333 | 同 | PASS |
| 严格无阻塞 Clos 中平面 ≥2k_down−1=143（vs 1:1 的 72） | 143 | 同 | PASS |
| rail 聚合 8×50=400；3600/400=9× | 400 / 9 | 同 | PASS |
| 式(7) n16 M1GB：每节点发 2×15/16=1.875 GB；单 NIC 37.5 ms；8 rail 4.6875 ms（÷8） | 1.875 / 37.5 / 4.6875 / 8 | 同 | PASS |
| ECMP 8 流 8 路径全错开 8!/8⁸ | 0.002403 | ≈0.0024 | PASS |
| torus 2D k32 二分 2k=64 vs fat-tree 512（8×）；每节点 2/k=1/16 vs 1/2 | 64 / 512 / 8 / 0.0625 | 同 | PASS |
| NVL72：900/100=9；10368/72=144 机架；72×1.8=129.6≈130 TB/s | 9 / 144 / 129.6 | 同 | PASS |
| Quantum 口径：XDR 144×800G=115.2 Tb/s 单向；Q2 64×400G×2=51.2 Tb/s 双向 | 115.2 / 51.2 | 同 | PASS |

### 10.2 分层 ring vs 平环 + SHARP（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| recap ring n8：bw 1.75/370.8=4.72 ms + lat 28 μs = 4.75 ms | 4.7475 | 4.75 | PASS |
| 命题1(a) 分层每 NIC 2(k−1)/k·M/8=218.75 MB；平环 2(n−1)/n·M=1.96875 GB；比 9.0× | 218.75 / 1.96875 / 9.0 | 同 | PASS |
| 命题1(b) T_lat=14×2+2×7×14=224 μs；T_bw=1.75/370.8+0.21875/50=4.72+4.375 ms；T_hier=9.32 ms | 224 / 9.3185 | 同 | PASS |
| 平环单通道 1.96875/50=39.375 ms；加速 39.375/9.32=4.2× | 39.375 / 4.225 | 同 | PASS（注：加速比用 T_hier 全量 vs 平环纯带宽项，偏保守，方向无误） |
| 多 channel 平环每 NIC ≈1.96875/8=0.246 GB（与分层 0.219 同量级） | 0.24609 | 0.246 | PASS |
| k=1 边界机间项=0 | 0 | 同 | PASS |
| SHARP 命题2：n64 出口 M=1GB→20 ms vs ring 39.375 ms，比 1.96875→2 | 20 / 39.375 / 1.96875 | 同 | PASS |
| 万卡 n10240：ring 2(n−1)α=286.7 ms；tree 2⌈log2 n⌉α=2×14×14=392 μs；比 731× | 286.692 / 392 / 731.36 | 同 | PASS |
| SAT max 2³⁰−32=1073741792 B；threshold 16384 B | 1073741792 | 同 | PASS |
| busbw AllReduce 因子 n64=2×63/64=1.96875；370.8=18×20.6 | 1.96875 / 370.8 | 同 | PASS |

### 10.3 a2a 消息量 / EP dispatch / incast（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| a2a n64 M1GB：每 rank 63/64=0.984375 GB、每 peer 1/64=0.015625 GB=15.625 MB | 同 | 同 | PASS |
| FP8 每元素 1+4/128=1.03125 B | 1.03125 | 同 | PASS |
| 式(1) dispatch 4096×8×7168×1.03125=242.22 MB（逐副本） | 242.22 | 242.2 | PASS |
| 节点去重 4096×3.6×7168×1.03125=109.0 MB；比 8/3.6=2.22× | 109.0 / 2.222 | 同 | PASS |
| 命题1 节省比 D_card/D_node=7.6/3.6=2.11× | 2.111 | 2.11 | PASS |
| recap E[D] 算术自洽：卡 64(1−(63/64)⁸)=7.62、节点 8(1−(7/8)⁸)=5.25、组限定 4(1−(3/4)⁸)=3.60 | 7.6/5.3/3.6 圆整 | 同 | PASS（recap 值，非重推） |
| LL 每副本 7168+7168/128×4+16=7408 B；dispatch 128×8×7408=7.586 MB；combine 128×8×14336=14.68 MB | 7408 / 7.586 / 14.68 | 同 | PASS（与 test_low_latency.py L138 公式逐字一致） |
| LL 带宽重算 EP8/64/256 dispatch 98.52/43.85/39.10、combine 128.77/46.75/40.78 GB/s | 同 | 表列 98.5/43.8/39.1、128.8/46.8/40.8 | PASS（重算值与章表一致；对 README 偏差见 S3-3） |
| 命题2 ΔT=α_NVL+mβ_NVL：decode 30720B→0.192 μs→2.19 μs；prefill 1MB→6.25→8.25 μs | 2.192 / 8.25 | 同 | PASS |
| V4.1 decode recap：64×6×5120=1.966 MB≈1.97；/64=30.72 KB；/50GB/s=0.614 μs | 1.97 / 30.72 / 0.61 | 同 | PASS |
| 命题3 incast：均衡入口 (n−1)·C/(n−1)=C；σC σ=1.09/2/4→54.5/100/200；超额 4.5/50/150；填 64MB 14.22/1.28/0.427 ms | 同 | 同 | PASS |
| 跨机占比 1−8/e：e8/16/32/64/128/256→0/50/75/87.5/93.75/96.875% | 同 | 同 | PASS |
| H800 160/50=3.2×；H100 450/50=9×；1/1.090=91.7%；153/51≈3× | 3.2 / 9 / 0.9174 / 3.0 | 同 | PASS |

### 10.4 Ring Attention overlap / 三模式 / causal（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| 式(1) M_blk=2×8192×64×128×2=268435456 B=256 MiB；式(2) F_blk=4×8192²×8192=2⁴¹ | 256 MiB / 2.199e12 | 同 | PASS |
| 式(3) t_comm=0.2684/50=5.369 ms（α 14 μs 占 0.26%）；t_compute=2⁴¹/400e12=5.498 ms；比 0.977（余量 2.3%） | 5.3687 / 5.4976 / 0.9766 | 同 | PASS |
| 命题1 阈值 bytes·F_eff/(2g·BW)=2×400e12/(2×50e9)=8000 token；8192≥8000 勉强成立 | 8000 | 同 | PASS |
| 反例 T_blk=4096：比 1.953，暴露 (1−1/1.953)=48.8%≈半 | 1.95 / 0.488 | 同 | PASS |
| 式(5) I_blk=2gT_blk/bytes=8192；R_comm=F_eff/BW=8000（roofline 同构） | 8192 / 8000 | 同 | PASS |
| 命题2 t_ring=t_comp+(n−1)max=16×5.498=87.96 ms；串行 168.49 ms；省 47.8% | 87.96 / 168.49 / 47.8 | 同 | PASS |
| 双缓冲依赖链仿真（compute 等 recv、group 等上一步 compute WAR + group 串行）：compute-bound/comm-bound/equal 三 regime t_ring 与闭式逐点相等 | 8/15/8（n8 单位时间） | 闭式 | PASS（仿真） |
| 式(7) M_KV=2×131072×8192×2=2³² B=4 GiB | 4 GiB | 同 | PASS |
| 命题3 AG=ring=(n−1)/n·M_KV：n16=3.75 GiB=4.027 GB | 4.0265 | 4.03 | PASS |
| 三模式 n8/16/32：ring 3.758/4.027/4.161 GB；Ulysses 8BTh(n−1)/n²=0.940/0.503/0.260 GB；比=n/2 | 同 | 3.76/4.03/4.16、0.94/0.50/0.26 | PASS |
| Ulysses 每 peer 每张量 n16/32/64=8/2/0.5 MiB | 8/2/512KiB | 同 | PASS |
| 驻留：ring 常驻 256+双缓冲 512=768 MiB；Ulysses 全序列×head 分片 256 MiB；AG 全量 4 GiB | 768 / 256 | 同 | PASS |
| AG 时序 15×0.2684/50=80.5 ms + attention 87.96=168.5 ms（=无重叠串行基线） | 80.53 / 168.49 | 同 | PASS |
| n8 对比行：T_blk16384、M_blk512MiB、t_comm1.448 ms、t_compute21.99 ms、比 0.066 | 同 | 同 | PASS |
| 阈值表：NVL 370.8→1079 token；CX8 Gen5 64→6250；比 0.763；8000/1079=7.4≈370.8/50 | 1078.75 / 6250 / 0.7629 / 7.414 | 同 | PASS |
| F_eff 400/989.5=40.4%≈40% | 40.4 | 40 | PASS |
| 练习1(c) sweep T_blk 2048/8192/32768→比 3.906/0.977/0.244 | 同 | 0.977@8192 | PASS |
| **causal 头块引理**：连续满块轮转第 k 轮 rank k 持 KV0（(k−k)%n=0），Q_k 整体晚于 KV0→满块对；仿真每步 max：step0=0.5（对角）、k≥1=1.0（满块对），overlap 条件不变 | 仿真确认 | 同 | PASS |
| **zigzag 总量恒等（练习3d 锚）**：rank r 持 {r,2n−1−r}，每 query 半块见 ≤自己的 KV 半块各一次（调度无关），总数 (r+1)+(2n−r)=2n+1；n=2/3/4/8/16 全验 =2n+1（n4=9u'） | 9u'（n4） | 9u' | PASS |
| **zigzag 每步 max（§4 推论）**：满块对轮转枚举 n4 四轮 max∈{2,3}u'⊂[2u',4u']（半..满块对）；头半块（含位置0）持有者每轮 ≥2u' 下界成立；充分条件阈值翻倍 16000 | [2,4]u' / 16000 | 同 | PASS |
| zigzag 总量 (2n+1)/(4n)→1/2（n16=33/64=0.516） | 0.516 | ≈半 | PASS |
| 练习4 环轮转仿真 n8：n−1 步后每 rank 见全 n 块、每 rank 发 (n−1)M_blk | 仿真确认 | 同 | PASS |

**重推导结论：172/172 PASS，构造性路由 120 trial PASS，仿真全 regime PASS。无 FAIL。**

---

## 2. Recap 边界判定（逐源核对，PASS）

| 章 | recap 源 | 核对结果 | 增量落点（spec §5 对应） |
| --- | --- | --- | --- |
| 10.2 | 8.4 NCCL 机制（init/channel/transport/proxy/协议/cost model/NVLS） | §2 recap 表逐行指针回 8.4 各节，不重推；α_NET=14.0 μs 对 8.4 §4 L82 逐字（cost_model.cc hwLatencies NET）；MAXCHANNELS 64/ringGraph 32 对 8.4 §3 | 拓扑检测部署口径 / rail 映射 / **分层 ring 精确账（命题1）** / SHARP 部署 / 调优 / 多作业 / 万卡测量 ✓ |
| 10.3 | training 5.4（P 置换/2kd/E[D] 7.6/5.3/3.6/木桶/20SM）+ inference 3.3（τ(e)/e*=32/dispatch 式(6)/1.97MB/30.7KB/双锚）+ 7.4（Wide-EP/EPLB/moe_lb 1.090/CV√(e/E)/18.8MB/doc L295-297）+ 8.1（a2a 每rank/每peer） | §2 recap 表逐行指针；E[D] 值直接 recap 不重算期望公式；τ(e)/e*=32 recap 不重推；命题1 节省比是 **transport 级副本粒度**（rdma_idx inplace_unique 代码锚），非 training 5.4 期望账重推 | DeepEP 双锚走读 / normal 两阶段 WarpRole / LL hook+IBGDA / V2.5 EPBuffer+Gin / **incast 命题3** / EP>32 transport 四成分 ✓ |
| 10.4 | models 4.6 §3.2（分块+两不变量，L212-227）+ training 5.2 §3-5（式(2)(3)(4)(5)，L63-109） | §2 recap 式(2)(5) 逐字翻译 c→n/s→T/b→B/a_kv→H_kv，无证明重推；式(3)(4) causal/zigzag recap 指针；models 4.6 原边界句「通信调度…归 cluster」L227 逐字存在，重判声明兑现 spec §2 L48 | send/recv transport / **overlap 命题1/2** / **AG=ring 命题3（增量）** / 双缓冲 / 拓扑映射 ✓ |
| 10.1 | 9.2 命题2（多 NIC 聚合 PCIe 前提）/ 8.2（ring 消息量+带宽项）/ 9.3（拥塞）/ 7.1/7.2（域几何/400/9×） | §1 边界声明 + 式(6) 标 recap 9.2 式(2)；ring 项 recap 8.2；跨域字节 M/8→M/72 对 8.2 §7 L177 逐字 | **fat-tree 命题1/2 + 三级 pod + rail×NIC（spec 关键推导）** ✓ |

**关键让渡核对**：8.2 §7 L179 逐字「分层的精确时间账与 rail 对齐、拥塞交互归 10.2，本章只立『压跨域字节 + 压跨域跳数』两条动机」——10.2 命题1 正是该让渡的兑现，**非越界重推**。8.2 §7 已交付的 M/8 动机账在 10.1/10.2 均标 recap。

**越界检查**：K8S/容器编排仅边界指针（10.1 §1「平台编排…归 cluster」、10.2 §7「作业级编排…归 cluster」），无实质内容；serving 调度/引擎编排仅 recap 指针（10.3 EPLB 只走读 DeepEP lb_* 原语层，引擎 shuffle 归 inference 7.4）。**PASS，无 S2 越界。**

---

## 3. DeepEP 双锚判定（PASS）

- **机制锚 v1.2.1**（9af0e0d0）：§4-5 讲 normal 两阶段 WarpRole + LL 纯 RDMA hook + IBGDA，全部标 v1.2.1 锚。代码锚逐一命中（见 §5 表）。
- **接口锚 main V2.5**（f6aa38fe）：§6 讲统一 EPBuffer（hybrid/direct）+ NCCL Gin + do_cpu_sync + lb_* + Buffer 家族拆分，全部标 main 锚。
- **版本漂移硬注**（§1 L11）：「spec 大纲…是 V1 语义；当前 main V2.5 已统一 EPBuffer + NCCL Gin + NVSHMEM 全移除——把 V1 机制当当前 main 接口写 = 事实错误」✓ 到位。
- **NVSHMEM 全移除 live grep**：`grep -rin nvshmem` 全 repo（py/cpp/hpp/cu/cuh/md/h/txt/toml）**仅 README L18** 一句 ✓；`ibgda_device.cuh / internode_ll.cu / buffer.py` 三文件 ls 确认 No such file ✓。
- **IBGDA 仅注释 live grep**：`grep -rin ibgda` **仅 hybrid_dispatch.cuh L441/L444 + hybrid_combine.cuh L482** 三处注释 ✓；紧邻代码为 `gin.put<ncclTeamTagRail>`（L445-451）✓；§6 坑注「V2.5 IBGDA 仅存注释实为 Gin GDAKI」与代码一致 ✓。
- **V1 vs V2.5 SM 口径分层**（§6 L127）：V1 LL 接收侧零 SM（README L7 逐字）vs V2.5 dispatch/combine 占 SM（README L39 逐字，章标 L38 见 S3-1），明确「勿混读成简单退化」✓。
- **Gin 后端探测**：context.cpp L126 `(allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE` 逐字 ✓；NCCL gin_host.cc L28/L36 + net_ib/gdaki/doca-gpunetio 目录存在 ✓。

**判定：双锚分离严格，无 V1-当-current 混读，无锚虚构。PASS。**

---

## 4. CP=5.2 指针判定（PASS）

- training 首行一手核：`02-sequence-and-context-parallelism.md` = 「# 5.2 SP / CP：序列维的切分」✓；`05-combining-strategies.md` = 5.5（组合策略，非 CP）。
- P10 四章 `grep -c "5\.5"`：10.1/10.2/10.3 = 0；10.4 = 1，命中行 L227 为「≈5.50 ms」**数值**，非章节指针。
- 10.4 全部 CP recap 指针均「training 5.2」（§1/§2/§5/参考）✓。**无 CP=5.5 误引。PASS。**

---

## 5. 代码锚抽查表（fresh tarball，~60 锚）

### DeepEP v1.2.1（机制锚）

| 章引 file:line | tarball 实测 | 判定 |
| --- | --- | --- |
| buffer.py L13 class Buffer / L30 num_sms=20 / L33-37 ctor（num_qps_per_rank=24 等） | 逐行命中 | ✓ |
| buffer.py L305 dispatch / L387 combine / L435 internode_dispatch / L530-535 low_latency_dispatch / L561-562 hook 逐字 / L96-99 NVSHMEM_DISABLE_P2P | 逐行命中（hook 逐字一致） | ✓ |
| deep_ep.cpp L306 intranode_dispatch / L654 internode_dispatch / L935 internode_combine / L1091 low_latency_dispatch | 逐行命中 | ✓ |
| configs.cuh L3 NUM_MAX_NVL_PEERS 8 / L20-21 SEND_PHASE 1 RECV_PHASE 2 | 逐字 | ✓ |
| internode.cu L74-75 translate_dst_rdma_rank / L100 rdma_rank·nvl_rank / L355-358 template / L370-376 WarpRole 五角色 | 逐字（normal=节点/LL=GPU 分支一致） | ✓ |
| internode.cu L462 kRDMASender / L690-691「RDMA consumers and NVL producers」/ L738「Forward tokens from RDMA buffer」/ L149·L672 put_nbi_warp / L1008 sender warps 7 | 逐行命中 | ✓ |
| internode_ll.cu L40-41 dispatch / L555-556 combine（__launch_bounds__ 1024,1）/ L87 SEND 相 / L248 RECV 相 / L162-166 p2p_ptr 探 NVLink / L164 put / L231 amo | 逐行命中（「Sending phase」注释 L86、gate L87；「Receiving phase」L247、gate L248） | ✓ |
| ibgda_device.cuh L63 mlx5_wqe_ctrl_seg typedef / L339 put_nbi_warp / L495 quiet | 逐行命中 | ✓ |
| setup.py L19-25 NVSHMEM_DIR / nvidia.nvshmem wheel | 逐行命中 | ✓ |
| README L5 normal / L7 LL hook「without occupying any SM resource」/ L19 intranode 153·158 / L20-22 internode EP16 43·43 EP32 58·57 EP64 51·50 / L32 LL EP8 77·114 / L37 EP256 194·360 / L39 News#173 / L82 NVSHMEM_DIR / L270 hook 逐字 | 逐行命中 | ✓ |
| test_internode.py L50-52 rdma_idx inplace_unique / L179 rdma_send_bytes / L194 fp8_factor / L297 num_tokens 4096 | 逐行命中 | ✓ |
| test_low_latency.py L138 每副本字节公式 | `(hidden + hidden/128*4 + 16), hidden*2` 逐字 = 7408/14336 | ✓ |

### DeepEP main V2.5（接口锚）

| 章引 file:line | tarball 实测 | 判定 |
| --- | --- | --- |
| ep.py L208 EPBuffer(BufferBase) / L217-219 hybrid docstring / L562 dispatch / L770 combine / L854 lb_prefetch_weights / L886 lb_reduce_grads / L27 EPHandle | 逐行命中（hybrid docstring 逐字） | ✓ |
| ep.py L553-557 QP 解析 | direct `min(num_sms, 8+1)` L554 ✓；**hybrid `num_sms*16+1` 实在 L558**（章引范围 L553-557 差一行） | **S3-2** |
| base.py L4 BufferBase(ABC) / allocator.py L10 BufferAllocator / ep.hpp L23 EPBuffer / L183 get_dispatch_buffer_size / L192 Direct / L199 Hybrid | 逐行命中 | ✓ |
| handle.cuh L21 ncclGin gin / L31 ctor / L113 VASignalAdd / L123-125 wait | 逐行命中 | ✓ |
| dispatch.cuh L31-32 dispatch_impl / L155-156 gin.put_value<team_t> | 逐行命中 | ✓ |
| hybrid_dispatch.cuh L34-35 hybrid_dispatch_impl / L179-183 gin.put<ncclTeamTagRail> / L441·L444 IBGDA 注释 / L445-451 token put | 逐行命中 | ✓ |
| hybrid_combine.cuh L482「Issue IBGDA」 | 逐字 | ✓ |
| context.cpp L126 Gin 探测 | 逐字 | ✓ |
| README L18 NVSHMEM 移除 / L25-32 features / L107 单 EPBuffer / L160-162 do_cpu_sync / L406 EP_DISABLE_GIN / L428 EP_GIN_GDAKI_DEBUG / L45-56 Requirements / L11-15 V2.5 release | 逐行命中 | ✓ |
| README L38 Notes「EP dispatch…zero-SM…not supported」 | **实在 L39**（L37=### Notes，L38 空行） | **S3-1** |

### NCCL 351f33fd（10.2/10.4 共用）

| 章引 file:line | tarball 实测 | 判定 |
| --- | --- | --- |
| topo.cc L1994 ncclTopoGetSystem / L2006-2013 NCCL_TOPO_FILE（L2012 默认路径）/ L2170 DUMP | 逐行命中（默认 /var/run/nvidia-topologyd/virtualTopology.xml 逐字） | ✓ |
| topo.h L24 SM90_NVLINK_BW 20.6 / L26 SM100 40.1 / L47-51 GPU0/PCI1/NVS2/CPU3/NIC4 | 逐字 | ✓ |
| paths.cc L243 PATH_PXN / L319「intermediate rank」注释 / L320-321 ncclTopoCheckP2p intermediateRank 出参 / L494 ncclTopoCheckGdr / L542 NCCL_NET_GDR_LEVEL | 逐行命中 | ✓ |
| graph.h（src/include/）PATH_NVL 1 / PATH_PXN 7 / PATH_PHB 8 / PATH_NVB 2 | L123/142/145/126 | ✓ |
| connect.cc L105 connectRings / L462 调用 | 逐行命中 | ✓ |
| coll_net.cc L416 sharedBuffersInit / L1613 COLLNET_CHAIN / L1648 COLLNET_DIRECT | 逐行命中 | ✓ |
| init.cc L67 NCCL_PARAM(CollnetEnable,…) | 逐行命中 | ✓ |
| net_ib/init.cc L12 IB_ADAPTIVE_ROUTING -2 / L464 IB_HCA / L594-596 IB 开 RoCE 关 | 逐行命中（L596 link_layer==IB?1:0） | ✓ |
| net_ib/connect.cc L26-31 IB_TIMEOUT 20（4.096μs×2^20≈4.3s 逐字）/ L32 RETRY_CNT 7 / L106 QPS_PER_CONNECTION 1 / L989 AR merge AND | 逐行命中 | ✓ |
| net_ib/gdr.cc L18 nv_mem sysfs / L24 ncclIbGdrSupport | 逐行命中 | ✓ |
| gin_host.cc L28 ncclGetGinType / L36 ncclGetRailedGinType；net_ib/gdaki/doca-gpunetio 目录 | 命中 | ✓ |
| nccl.h.in L739 ncclSend / L758 ncclRecv / L873 GroupStart / L883 GroupEnd；send/recv 配对 + 「blocking for the GPU」+ group 逐字（L735-737） | 逐行命中 | ✓ |
| transport.cc L15-20 p2p/shm/net/collNet / p2p.cc L358 p2pMap / L410 useRead / net.cc L1327 sendProxyProgress | 逐行命中 | ✓ |
| cost_model.cc L125 isLL128Enabled | 命中 | ✓ |

**抽查结论：~60 锚命中，2 处 off-by-one（S3-1 README L38→L39；S3-2 ep.py L557→L558），无虚构锚。**

---

## 6. 官方逐字 live WebFetch 再核（3 项）

| 来源 | 章引逐字 | live 抓取 | 判定 |
| --- | --- | --- | --- |
| XDR Switch HW Manual（networking-docs.nvidia.com/xdrswitcheshw/1.6/introduction） | 「a two-level fat tree topology can connect up to 10,368 NICs at lowest latency」+ 144×800G/72 OSFP + Q3400-RA 115.2 Tb/s + 200Gb/s/lane SerDes + Quantum-3 | 全文命中（10,368 逐字、115.2 Tb/s 规格表、144x 800 Gb/s 72 OSFP） | ✓ |
| NCCL User Guide env.html（docs.nvidia.com） | 「Enabled (1) by default on IB networks. Disabled (0) by default on RoCE」+ NCCL_IB_RAIL_POLICY | 逐字命中 | ✓ |
| IB Cluster Bring-Up（networking-docs.nvidia.com/ibclusterbringupproc/setting-the-infiniband-cluster-topology） | 「Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized」/「Rail-optimized design means…each GPU "rail"…onto a different first level (LEAF) switch」/「NVSwitch path…one switch hop away」 | 三句逐字命中 | ✓ |

其余官方/论文锚（SHARP v3.17.0 SAT、Quantum-2、NCCL 2.12/2.4 blog、Al-Fares Crossref、Ring Attention arXiv+repo、DCQCN）沿 T1 research 2026-11-25 一手记录（§5-8），本轮未逐一重抓（research 已逐字记录 + 本轮 3 项抽样全命中，证据链可信）。Al-Fares PDF 不可达残留（元数据 Crossref 已核、公式独立重推）口径正确。

---

## 7. Findings（S1=0 / S2=0 / S3=7）

| ID | 级 | 文件:行 | 问题 | 建议修复 |
| --- | --- | --- | --- | --- |
| S3-1 | S3 | 10.3 L127 / L190 / L204（+ research §5.2） | main README「EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported」实在 **L39**（L37=### Notes、L38 空行），章三处标 L38 | L38→L39（三处 + research 同步） |
| S3-2 | S3 | 10.3 L127 / L204 | QP 解析引「ep.py L553-557」，hybrid 式 `num_qps = num_sms * 16 + 1` 实在 **L558**（L553-557 仅含 direct 式 + hybrid 注释/if） | 范围改 L553-558 |
| S3-3 | S3 | 10.3 L115 | LL 带宽表「逐数闭合（容差 ~1%，README 圆整）」——独立重算对 README 偏差最大 **1.97%**（EP64 dispatch 43.85 vs 43；EP256 combine 40.78 vs 40；EP8 combine 128.77 vs 127），超「~1%」。重算值本身正确且表内双列透明 | 「~1%」→「~2%」，或注「README 延迟/带宽各自圆整，重算按显示延迟反推故偏差至 ~2%」 |
| S3-4 | S3 | gpu/references.md（9 条目） | citedBy 回写缺口：nccl-all2all-blog+=10.1；connectx7+=10.1,10.2,10.3；nccl+=10.1；nccl-docs+=10.1；nccl-tree-blog+=10.2；ib-cluster-topology+=10.2；dcqcn+=10.2,10.3；ib-cc-whitepaper+=10.2；nvidia-datasheet+=10.4 | 程序化回写（条目均存在，「先收录后引用」满足，仅 citedBy 字段漏） |
| S3-5 | S3 | 10.3 L147（命题3 陈述） | 「均衡 a2a 是排列」术语不准——均衡 a2a 是均匀 all-to-all（双随机矩阵），非排列（10.1 命题1 已把「排列流量」精确定义为每 NIC 唯一目的）。入口=C 的定量与证明不受影响（证明只用均分，不用排列性质） | 「是排列」→「是均匀流量（每 receiver 入口恰 C）」；§3「NVSwitch 非阻塞对均衡排列」语境可保留 |
| S3-6 | S3 | 10.1 L142（torus 段） | 临时借用保留符号 n（=维数）与 k（=arity），虽有显式「局部借用注」且限于结构对照段，仍偏离 §4.3「n 专归 rank 数」硬规则 | 可选 tightening：用字面「二维 k×k」/ d_dim 表维数，避免借 n（k 已章首局部声明，借用注可保留）；T7 可裁定接受（显式注已消歧） |
| S3-7 | S3 | 10.2 L76-78（命题2 + 口径注） | SHARP 延迟模型「树深 ⌈log2 n⌉ 量级 / 2⌈log2 n⌉ 跳」是二叉树教学抽象；实际 SHARP 聚合树深由 fabric 层级数界定（万卡 2-3 层 fabric 下 ~常数跳，远小于 14）。已标「机制级教学推导」+ 731× 标「纯延迟项理论比」vs Summit 180× 实测，结论方向不变 | 可选：补一句「实际 SHARP 树深随 fabric 层级（常数），二叉树 log n 模型为保守上界」；T7 可裁定接受（口径注已披露） |

**无 S1/S2。** S3-1/2/3/4 建议修复（成本低）；S3-5 建议修复（术语）；S3-6/7 为可选 tightening，T7 可接受/驳回逐条裁定。

---

## 8. 规范 / 模板 / 证据分级审查

- **lint-chapter.sh + lint-math-boundary.py**：4/4 LINT OK。
- **独立扫描**：display 9 禁形（`\, \! \; \: \{ \} \| \% \#`）= 0；prose `\lt` = 0；bare `<=` = 0；process-leak（T1-T8/brief/writer/evaluator/reviewer/fix round）= 0（10.2 命中 INT8/NVS2 为技术子串误报）；禁用宏（operatorname/align/equation/newcommand/boxed/medspace）= 0。
- **CJK**（正文汉字，不含标点/公式/代码）：10.1=5146 / 10.2=4814 / 10.3=4583 / 10.4=5180，均 ≥3500 ✓（10.3 模块走读章、10.1/10.4 推导章均超标准章线）。
- **模板 R2**：metadata 块（L3）✓；§1 三问+章脊口径锁定（R2-2，10.1 多 SKU 对比口径章首声明）✓；节首 crux/设问/数值开场（R2-1）✓；公式后全数字实例（R2-3）✓；命题-证明骨架（R2-4）+ 推导路线句（R2-5）✓；定义即配例+边界反例（R2-6）✓；口径注（R2-7）✓；术语判断句「所谓 X 是指」（R2-8）✓；提点/坑注块引用（R2-10/11）✓；总结表回指列（R2-12）✓。
- **练习**：四章均 4 题，分级 [10]/[20]/[30]/[30] 非递减、≥1 题 [10]/[20]、≤1 题 [40]（无 [40]）、每计算/代码题带校验锚、≥1 题证正文命题（10.1 练3 证命题2 / 10.2 练3 证命题1b / 10.3 练3 证命题2 / 10.4 练3 证命题1）✓。**(a)(b)(c)(d) 脚手架齐全。**
- **练习 3(d) 锚一致性（brief 专项）**：10.4 练3(d) 现仅要求 zigzag n=4 **总量恒等 9u'**（调度无关块对计数），每步max锚已按 option-b 删除（commit 2af2606）；与 §4 推论「每步 max 介于半..满块对、充分阈值翻倍」自洽（独立枚举 n4 四轮 max∈{2,3}u' 验证）。**PASS。**
- **证据分级**：[A] Ring Attention（2310.01889 + haoliuhl/ringattention 官方 repo）✓；[B] Al-Fares（Crossref pp 63-74，PDF 残留口径正确）/ DCQCN ✓；[C] quantum-switch / ib-cluster-topology / nccl-all2all-blog（开放技术博客）/ sharp-docs / connectx7/8 / gb200-nvl72 / ibta-spec / nvidia-datasheet ✓；教学参照 nccl / deepep / nccl-tests / nccl-rdma-sharp-plugins ✓。无推测当事实。
- **符号 §4.3**：N 不表 rank、B 专 batch、T 专序列（10.4）/时间下标（10.2/10.3）、α/β 章首通信语义声明、bytes 不占字母、n 章首声明 ✓。局部符号（10.1 k/r/n_tier、10.2 k/C、10.3 e/k/C/σ、10.4 R_comm/u'）均章首/节首声明 ✓（torus 借 n 见 S3-6）。
- **refs**：P10 八条目 + 跨 Part 共享锚均存在，arXiv ID（2310.01889）在 references.md ✓；citedBy 缺口见 S3-4。

---

## 9. 总体判定

**Part 10 四章通过对抗审查（S1=0 / S2=0 / S3=7）。** 强制重推导 172/172 PASS（含 fat-tree König 构造性路由 120 trial、Ring Attention 双缓冲仿真全 regime、zigzag 恒等 5 度数），数值脊柱可靠；recap 边界干净、增量落点与 spec §5 逐一对应；DeepEP 双锚分离严格（NVSHMEM 移除/IBGDA 仅注释 live grep 核实）；CP=5.2 无误引；代码锚 ~60 命中（2 off-by-one）；3 项官方逐字 live 再核全命中。7 项 S3 交 T7 修复/裁定（S3-1/2/3/4/5 建议修，S3-6/7 可选）。**无阻塞项，P10 可进 T7 fix + 图 QA + commit gate。**
