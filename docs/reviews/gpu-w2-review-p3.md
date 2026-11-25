# gpu W2 T6 adversarial review — Part 3 CUDA 编程模型（3.1–3.4）

日期：2026-11-25。评审：fresh-context adversarial evaluator（T6，不继承写作上下文，无 subagent）。
对象：`gpu/03-cuda-programming-model/{01-thread-hierarchy-simt,02-memory-model-address-spaces,03-stream-event-graph,04-occupancy-resource-constraints}.md`。
标准：STYLE_GUIDE + gpu spec §5 Part 3 / §4.3 符号 / §6.3 重推导清单 + R2 style sheet（R2-1…12）+ rubric + `docs/research/gpu-w2-sources.md` + `gpu/references.md` P3。
方法：一手来源 curl 自取对照（PG 13.5 PDF + 子页 / archive 13.0.0 / PTX 9.5 / BP 13.5 / NCU / libcu++ / Fermi wp / CUDA Graphs blog）+ Python 独立重算全部数值账 + recap 边界核对 + 机械规范扫。

## 0. 结论摘要

| 项 | 结果 |
| --- | --- |
| **总体状态** | **PASS**（事实与推导全部干净；S1=0）；S2×1 为 refs 被引回写簿记缺口（须修），S3×2 minor |
| S1（事实致命） | **0** |
| S2（实质问题） | **1**（references.md 被引回写不完整，7 条 source / 8 处；非读者面） |
| S3（minor） | **2**（引文版本归属精度；3.3 §1 「3.2 ms」口径混用） |
| **重推导 3.4 occupancy + Little's law** | **PASS**（register 悬崖 64→65 / 隐藏反解 O_min→7 blocks→36 regs / compute-bound 16 warps ILP4→6.25% / K2 / 官方算例 / shared 悬崖 / CC7.0 37 regs 全部独立重算对上） |
| **RECAP 边界** | **PASS**（1.2 B_max 四约束 min = recap 不重推；1.4 Little's law 通式 + 208 sector = recap；3.4 增量 = 分配口径 b_act + API + register/ILP + 反解 + compute-bound 命题；3.3 recap inference 3.4 launch 账，增量 = 摊销盈亏平衡） |
| 数值账独立重算覆盖率 | 100%（3.1/3.2/3.3/3.4 全部 display 公式与案例数值） |
| 一手事实核查覆盖率 | ~90%+ 载重断言逐字对照（见 §3 verification log） |
| 机械规范（rescape/process-leak/AI 味/裸 symbol/练习/总结） | 全过 |

## 1. Findings 表

| # | Sev | 位置 | 断言 | 证据 | 建议 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | `gpu/references.md` P1 共享锚点被引字段（3.3/3.4 引用回写） | 7 条 source 条目的「被引章节」缺 3.3/3.4（共 8 处），与章末参考实际引用不符；research pass §5 误称已含 | 逐条 grep 被引字段：`cuda-best-practices`=1.2,1.4,2.2,2.3,3.2,3.4,4.1,4.2,4.3（**缺 3.3**，3.3 §4 引 BP overlap/copy-engine/tE+tT/nStreams）；`hopper-whitepaper`=1.1…2.4,4.4（**缺 3.3、3.4**，3.3 §4 + 3.4 §4 引 132SM/HBM3352/FP32 66.9）；`pcie-spec`=2.4,3.2,7.2（**缺 3.3**，3.3 §4 引 PCIe 64GB/s）；`hopper-tuning-guide`=1.1,1.2,2.3,4.4（**缺 3.4**，3.4 参考引 CC9.0 64warps/228KB）；`blackwell-tuning-guide`=1.1,1.2,2.1,2.2,2.3（**缺 3.4**，3.4 §2 引 1KB 预留逐字）；`hopper-microbench`=1.3,1.4,2.2,2.3,2.4,3.2（**缺 3.4**，3.4 参考引 HBM 478.8）；`nvidia-datasheet`=1.1,1.3,2.1,2.4,6.1,7.1,7.2,11.1（**缺 3.4**，3.4 参考引 FP32 67/HBM3.35）。`gpu-w2-sources.md` §5 称「hopper-whitepaper / hopper-tuning-guide 被引已含 3.1-3.4…不变」——实际两条均无任何 3.x，声明不实 | 回写 8 处被引（cbp+=3.3；hw+=3.3,3.4；pcie+=3.3；htg+=3.4；btg+=3.4；hm+=3.4；nd+=3.4）；订正 research pass §5 表述。属簿记/协议缺口（spec §6.1 step4 七字段含被引），不影响读者与章内事实；lint check4（先收录后引用）已过（key 均存在） |
| F2 | **S3** | 3.2 §2/§3 + 参考、3.3 §5 参考（多处「PG 13.5 逐字」引文） | 部分标「13.5 逐字」的引文实为 **archive 13.0.0 逐字**；13.5 PDF/子页已改写措辞（语义全同），章已引 13.0 互证，故引文真实、事实无误，仅版本归属精度问题 | 自取 PG 13.5 PDF（12784076B）+ 子页 + archive 13.0.0（3921458B）逐字比对：①volatile 章引「Reads and writes to volatile qualified objects are not atomic, and are compiled to one or more .volatile instructions which do NOT guarantee…」= 13.0 line12401-12402 逐字；13.5 PDF line32323-32330 作「Reading and writing to volatile-qualified objects are not atomic and are compiled into one or more volatile instructions that do not guarantee…」②fence 章引「The memory fence functions differ in the scope in which the orderings are enforced but they are independent of the accessed memory space (shared memory,…)」= 13.0 line5195 逐字；13.5 PDF line34897 / 子页 sub-lang line932 作「These functions differ in the thread scope…including shared memory…」③syncthreads/atomic relaxed/local backing/cudaMallocAsync/UM page fault 引文均 = 13.0 逐字（line5283/5608-5609/4890/10817/16023），13.5 改写。**反例（确为 13.5 逐字）**：SIMT divergence、graph update「Major changes to graph structure (such as topology or node types)…」（13.5 line13048-13050）、conditional nodes、launch overhead、default stream、carveout、launch_bounds、occupancy 算例 | 将这几处引文标注为「13.0 逐字 / 13.5 同义」或注明 13.5 改写；不必改语义（语义两版全同，事实正确）。属可选精度增补 |
| F3 | **S3** | 3.3 metadata + §1 首段 | 「拷 192 MiB 数据要 3.2 ms」/「PCIe 上要走 3.2 ms」把 **τ_seq（拷贝+计算串行总时间 3.206 ms）** 当作纯 PCIe 拷贝时间；纯拷贝实为 3.146 ms | 独立重算：192 MiB=201326592 B ÷ 64e9 = **3.1457 ms**（纯 H2D 128MiB+D2H 64MiB）；τ_seq=3.206 ms 含 kernel 60.062 μs（§4 整块对账 2097.152+60.062+1048.576）。§1「计算只占端到端的 1.9%」用端到端 3.206 ms 正确（60.062/3205.79=1.87%），但同段「PCIe 上要走 3.2 ms」把 3.2 ms 归给 PCIe  alone，混入 60 μs kernel | 改「端到端串行 3.2 ms（PCIe 拷贝 3.15 ms + kernel 60 μs）」或「拷 192 MiB 约 3.15 ms」；§4 分解已正确，仅头条口径需对齐。minor |

> 无 S1。所有载重事实经一手核实正确，全部数值账独立重算对上，无虚构数值/张冠李戴引用/不存在 repo。⟨verify⟩ 残留（PCI-SIG 精确 URL 留 W4）按任务口径**不计 finding**。

## 2. 重推导 3.4 + Little's law（MANDATORY，独立 Python 重算，PASS）

脚本 `/tmp/gpu-w2-t6-eval/scripts/rederive.py`（从第一性原理重算，非抄章）。硬件库存独立 restated：R_file=65536、W_slots=64、B_slot=32、S_total=233472 B。

### 2.1 register 悬崖 64→65 regs × 512 threads（CC 6.x）— PASS
- 64 regs：2×512×64 = 65536 = R_file 恰满 → 2 blocks、32 warps、O=32/64=**0.50**。
- 65 regs：2×512×65 = 66560 > 65536 → 1 block、16 warps、O=16/64=**0.25**。
- register 需求涨 65/64−1 = **1.5625% ≈ 1.6%**（章「1.6%」对）。
- 官方锚 PG archive/13.0.0 line4725 逐字对上（「two blocks (i.e., 32 warps)…2x512x64…one more register…one block (i.e., 16 warps)…2x512x65」）。

### 2.2 隐藏反解 O_min 0.81 → 7 blocks → 36 regs — PASS
- 独立重算 Little's law 输入：λ_SM = 3352e9/(132×32×1.83e9) = **0.4336 sectors/cycle**；L_SM = 0.4336×478.8 = **207.63 ≈ 208 sectors**；warp load = 207.63/4 = **51.91 ≈ 52**。
- O_min(ILP1) = 207.63/(4×1×64) = **0.8110**（章/1.4 round 0.81）；ILP2→0.4055（0.41）、ILP4→0.2028（0.20）。
- n_b_req = ceil(0.8110×64/8) = ceil(6.488) = **7 blocks**（56 warps、O=0.875）；R_th_max = floor(65536/(7×256)) = floor(36.571) = **36 regs**。
- CC 7.0 粒度（256/warp）收紧：36 regs→ceil(1152/256)×256=1280/warp，7×8×1280=**71680 > 65536 装不下**；33/34/35 同 1280/warp 均 fail；**32 regs→1024/warp，56×1024=57344 ≤ 65536 OK**。章「降到 32」对（32 是 7 blocks 下可装的最大值）。
- 注（非 finding）：章「0.81×64 = 52 warps」中 0.81×64 字面 = 51.84，52 来自 L_SM/s_warp=208/4 的整 warp 口径（1.4 已立 52/64≈0.81 约定）；用精确 O_min=0.8110 或 round 0.81 反解均得 7 blocks/36 regs，答案不变，属继承 1.4 的舍入表述。

### 2.3 compute-bound 16 warps / ILP4 → 6.25%（命题 2）— PASS
- 每 scheduler 在飞 = 1 issue/cycle × 4 cycle FMA 延迟 = 4；4 scheduler = **16 条在飞 warp FMA**。
- ILP1：16/1 = **16 warps**，O=16/64=**0.25**；ILP4：16/4 = **4 warps**，O=4/64=**0.0625=6.25%**。
- 官方锚 PG archive/13.0.0 line4720 逐字「16 active warps per multiprocessor (4 cycles, 4 warp schedulers)…fewer warps…ILP…back to back」对上。

### 2.4 K2 判定 + 官方算例 + shared 悬崖 + CC7.0 37 regs — PASS
- K2（256/96 regs/32KB）：四项 min(32, 8, floor(65536/24576)=2, floor(233472/33792)=6) = **2 blocks、16 warps、O=0.25**，register 绑定（96×32=3072=12×256 无取整损耗）；ILP1 16<52 藏不住，min ILP=ceil(52/16)=**4**（16×3=48<52、16×4=64≥52）；launch_bounds(256,4)→L_reg=floor(65536/1024)=**64**→4 blocks/32 warps/O=0.5，ILP2 32×2=64≥52；carveout 100KB→shared 项 floor(102400/33792)=3 仍宽于 register 项 2，**occupancy 不变**。全部对上。
- 官方算例（PG 13.5 line4654-4667 逐字）：768 threads→2 blocks/48 warps/**75%**；32 threads→32 blocks/32 warps/**50%**（block-slot 绑定）；100KB shared→2 blocks（3×102400=307200>233472；含 1KB 预留 floor(233472/103424)=2 同值）。对上。
- shared 悬崖表（S_alloc=S_sh+1024）：32KB→7/6、48→4/4、64→3/3、96→2/2、112→2/2（233472/115712=2.018）、116→1/1、227→1/1；carveout 64KB：228 档 floor(233472/66560)=3、100 档 floor(102400/66560)=1。全对上。
- CC7.0 37 regs（BP line848 逐字）：128 threads→1280/warp、5120/block、floor(65536/5120)=**12 blocks/75%**（官方对上）；320 threads→1280/warp、12800/block、warp 粒度手算 floor(65536/12800)=**5（78%）**，**官方 4 blocks/63%**——章正确标注手算给上界、API 给精确值、架构特定细节官方未逐字披露，方向 4≤5 与 1.2 命题 3 一致。**诚实处理，PASS**。
- ridge：R=66.9/3.352=**19.958≈20** FLOPs/B；elementwise I=1/12=0.0833≪R→memory-bound。对上。

### 2.5 其余章数值账独立重算（全 PASS）
- **3.1**：D1/D2 η=0.5、D3=1、D4=0.1875、D4'=0.8125（差 4.33×）、D5=0.25；概率 p=0.5 E[slot]=2/η=0.5，p=0.99 E[slot]=1.275/η_agg=0.784/E[η]=0.8625（聚合 vs 逐 warp 口径区分正确）；波次 65536/1056=62.06→62 满波+64 尾（6.06%≈6.1%）、grid-stride ceil(16777216/270336)=63、132×2048=270336、62 倍。全对上。
- **3.2**：fp32 (1e8−1e8)+1=1 对 1e8+(−1e8+1)=0（struct.pack 实算）、ulp@1e8=2^(26−23)=8、顺序累加 1 停 2^24=16777216；延迟比 478.8/29=16.51、263/40.7=6.46、263/29=9.07、263/1.83=143.72ns、29/1.83=15.85ns；constant 65536/4=16384；histogram ρ=16777216/270336=62.06；迁移 2097152/64e9=32.768μs、1073741824/64e9=16.777ms（512 页×32.768 对上）。全对上。
- **3.3**：t_H2D=262.144/t_D2H=131.072/t_comp=7.508μs、Σ=400.724、τ_seq=3.206ms、τ_pipe=Σ+7×262.144=2.236ms、加速 1.4339、渐近 Σ/max=1.5286、省 970.06μs(30.3%)、H2D 占 Σ 65.4%；n_chunk 4→1.3502、16→1.4797；单引擎拷贝忙 3145.73μs、省≤60.06μs(1.9%)、双/单 16.15×；整块 2097.152+60.062+1048.576 对账；graph blog 暴露 6.7/0.9/0.5、Δ0.4、break-even 400/0.4=1000、20000 摊 0.02、多 kernel 400/(1650−5)=0.243<1、1.65/4.81=34.3%、1.65/15.1=10.9%、省 1.645ms。全对上（唯 §1 头条 3.2ms 口径见 F3）。

## 3. RECAP 边界核对（MANDATORY，PASS）

| recap 接口 | 边界判定 | 证据 |
| --- | --- | --- |
| **3.4 ← 1.2 occupancy 物理上界** | **PASS（recap 不重推）** | 3.4 §2 写 B_max 四约束 min 式即标「1.2 式(3)-(5) recap，不重推」，引 1.2 命题 3 定方向；**未**重做 1.2 §5 的必要条件表/量纲对账/min-vs-sum/取整论证。增量 = 分配口径 b_act（R_th·threads→R_blk warp 取整、S_sh→S_alloc 128B+1KB）+ occupancy API 语义 + 官方算例。正是 spec §5 3.4 增量 |
| **3.4 ← 1.4 Little's law** | **PASS（recap 不重推）** | 3.4 §4 标「recap 1.4，不重推」，L_SM=208 引 1.4 式(3)、隐藏条件引式(4)、O_min 引式(5)、服务率坑注引 1.4；**未**重推 λ_SM=BW/(n_SM·32·f)=0.434 与 L_SM=0.434×478.8。增量 = 反解资源预算式(5)（n_b_req/R_th_max，1.4 无）+ compute-bound 命题 2（spec §5 3.4 明列「compute-bound 低 occupancy 满吞吐」；命题 2 引「1.4 §4 热身例」作 per-scheduler 步，增量 = ILP→warp 数折算 + occupancy% + 官方锚）。spec §6.3 明列「3.4 occupancy 计算 + Little's law latency hiding（数值重算抽查）」为本章重推导靶 → 命题 2 是预期增量非越界 |
| **3.3 ← inference 3.4 图捕获** | **PASS（recap + 指引）** | 3.3 §5「先 recap 引擎侧的账（inference 3.4 §3，纯文字指引，不重推）」：330 kernel/5μs/1.65ms/34.3%/10.9%/省 1.645ms/静态三要求/full vs piecewise 均忠实 recap（与 inference 3.4 line35-53 逐数对上）；式(5)=inference 式(2) 符号译名（n_ker→n_kernel，附翻译注）；引擎配置面（CompilationMode/CUDAGraphMode/splitting_ops）+ piecewise 正确性命题明确归 inference 3.4 不重复。增量 = GPU 侧 t_launch 机制 + 三阶段路径 + 摊销式(6)(7) + 盈亏平衡命题 2（inference 3.4 无 t_inst/n_rep 摊销）|
| **3.2 ← inference 1.4 batch-invariance** | **PASS（纯文字指引）** | 3.2 §5 recap「concurrency+floating point 只是部分解释 / batch 组成改变 reduction 分区顺序 / batch-invariant 固定 order」与 inference 1.4 §6 line154-156 逐字一致；引擎模式入口与 kernel 实现归 inference 1.4/3.4；3.2 自给 fp32 atomic 例（1e8/−1e8/1，与 inference 1.4 命题 6 的 2^24 例不同）作本章增量，不重推 inference 命题 |
| 3.1/3.2/3.3 ← 1.2/1.4/2.2/2.3/2.4 | PASS | 库存数（65536/64/32/233472/228-227-256）、延迟实测口径（29/40.7/263/478.8 [B] H800 非 H100 官方）、PCIe 64GB/s、roofline 入口均忠实 recap 带指针，不重推 |

**越界检查**：无 serving 调度/引擎编排（归 inference）、无 K8S/容器（归 cluster）内容；3.3 §5 引擎配置面明确外指。PASS。

## 4. Verification log（MANDATORY — 抽查项 × 来源 × 结论 × 覆盖率）

### 4.1 一手来源自取（curl，全部 200，字节数与 research pass 记录逐一吻合）

| 来源 | URL | HTTP/字节 | 用途 |
| --- | --- | --- | --- |
| PG 13.5 PDF | docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf | 200 / 12784076B | 主 corpus（SIMT/fence/volatile/occupancy/launch_bounds/carveout/graph/stream） |
| PG 13.5 子页 | …/05-appendices/cpp-language-extensions.html ; 02-basics/writing-cuda-kernels.html ; 03-advanced/advanced-kernel-programming.html | 200 / 569418 ; 214943 ; 145518B | 子页措辞核对（fence 改写确认） |
| PG archive 13.0.0 | docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html | 200 / 3921458B | 互证 corpus（register 悬崖/16warps/API/copy-engine/priority/queue） |
| PTX ISA 9.5 | docs.nvidia.com/cuda/parallel-thread-execution/index.html | 200 / 3965886B | fence.sc/membar、morally strong、cache ops、divergence .uni、atom |
| BP 13.5 | docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 200 / 433269B | occupancy/37regs/256per-warp/ILP/copy-engine/overlap/估算 |
| Nsight Compute | docs.nvidia.com/nsight-compute/NsightCompute/index.html | 200 / 347513B | Occupancy Calculator replaces spreadsheet + Utilization 三态 |
| libcu++ memory model | nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html | 200 / 93352B | thread_scope 五级 enum + device scope + pre-SM90 widened |
| Fermi whitepaper | www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf | 200 / 876380B | GigaThread 逐字 + two-level scheduler |
| CUDA Graphs blog | developer.nvidia.com/blog/cuda-graphs/ | 200 / 237053B | V100/CUDA10.1 launch 开销 2.9/9.6/3.8/3.4/400/0.02/33% |

### 4.2 逐字断言核对（抽查项 × 来源引句 × 结论）

| 抽查项 | 来源（自取行号） | 结论 |
| --- | --- | --- |
| SIMT divergence「executes each branch path taken…disabling…Branch divergence occurs only within a warp」 | PG13.5 PDF 8822-8826 | ✅ 逐字对上（3.1 §3） |
| SIMT vs SIMD「SIMD…expose the SIMD width…SIMT instructions specify…single thread」 | PG13.5 8829 | ✅ |
| Volta 独立 PC「execution state per thread…program counter and call stack」「schedule optimizer」「sub-warp granularity」 | PG13.5 8848-8858 | ✅ |
| atomic 同址「they are all serialized, but the order…undefined」/ 非原子「which thread performs the final write is undefined」 | PG13.5 8880-8883 | ✅（3.2 §5 命题 2 锚） |
| block 不可跨 SM「expected to reside on the same streaming multiprocessor core」 | PG13.0 976 | ✅ |
| block 调度「enumerated and distributed…available execution capacity」「vacated multiprocessors」+ 无 ordering 保证 | PG13.0 4670 / PG13.5 4586 | ✅ |
| tid 公式 x+y·Dx+z·Dx·Dy | PG13.0 956 | ✅（3.1 式1） |
| GigaThread「The GigaThread global scheduler distributes thread blocks to SM thread schedulers」+ two-level + 12288 | Fermi wp 236-237 / 675-680 | ✅（3.1 §5；PG 8.0-13.5 无此词，legacy 注正确） |
| fence 三级 ≡ seq_cst block/device/system + 「only affect the ordering…do not, by themselves…visible」 | PG13.0 5195-5227 | ✅ 语义对；**措辞 = 13.0 逐字，13.5 改写**（F2） |
| membar ≡ fence.sc（sm_70+，cta/gl/sys ≡ cta/gpu/sys） | PTX9.5 13481 | ✅ 逐字（3.2 §2） |
| morally strong 三条件（program order/strong+scope 互含；same proxy；完全 overlap）+ data race | PTX9.5 3163-3170 | ✅（3.2 §2 restatement 准确） |
| L1 非一致「coherent at the L2…multiple L1 caches are not coherent」+ driver invalidate + performance hints only | PTX9.5 8399/8406 | ✅（3.2 §3） |
| texture/ld.global.nc 不适用内存模型 | PTX9.5 3059 | ✅（3.2 §1） |
| divergence .uni / optimizing code generator reconvergence | PTX9.5 4144-4145 | ✅（3.1 §3） |
| atom .sem={relaxed,acquire,release,acq_rel} .scope={cta,cluster,gpu,sys}；默认 relaxed/gpu | PTX9.5 13607-13620 | ✅（3.2 §5） |
| volatile 新口径（不原子/不保序/NOT suitable for Inter-Thread Sync/tile 无效） | PG13.5 32323-32340 / PG13.0 12401 | ✅ 语义对；**章引文 = 13.0 逐字**（F2） |
| syncthreads 双语义 / atomic relaxed+scope / local backing / cudaMallocAsync / UM page fault ∝ page size | PG13.0 5283/5608/4890/10817/16023 | ✅ 语义对；**= 13.0 逐字，13.5 改写**（F2） |
| thread_scope 五级 enum + device scope（same device+sync domain）+ pre-SM90 widened + 成本随距离 | libcu++ 446-461/439 | ✅ 逐字（3.2 §2） |
| occupancy CC10.0 算例 768→75% / 32→50% / 100KB→2 blocks（300KB>233472） | PG13.5 4654-4667 | ✅ 逐字（3.4 §2） |
| register 悬崖 CC6.x 64/65×512 | PG13.0 4725 | ✅ 逐字（3.4 §3） |
| 16 active warps（4 cycles,4 schedulers）+ ILP fewer warps | PG13.0 4720 | ✅ 逐字（3.4 命题 2 锚） |
| launch_bounds L 推导 / 超 L 压 / 双参抬至 L / 超 maxThreads launch fail / 架构分档 | PG13.5 34519-34554 | ✅（3.4 §3 命题 1） |
| occupancy API blocks/SM + ×warps÷maxwarps 换算 | PG13.0 4731-4732 | ✅ 逐字（3.4 §2） |
| carveout「next larger capacity」+ driver hint | PG13.5 9704/9717 | ✅（3.4 §5） |
| BP「Higher occupancy does not always…low occupancy always interferes」 | BP13.5 844 | ✅ 逐字（3.4 §4） |
| BP CC7.0 37regs 128→12blocks75% / 320→4blocks63% + 256/warp | BP13.5 848 | ✅ 逐字（3.4 §3，320 手算 5≠官方 4 诚实标注） |
| BP ILP「fully cover latency with a low occupancy」 | BP13.5 864 | ✅ 逐字（3.4 §3） |
| BP 单/双 copy engine + overlap 三条件（pinned+不同非默认 stream）+ blocking 不可 overlap + tE+tT/nStreams 估算 | BP13.5 392/413/420 | ✅ 逐字（3.3 §4） |
| NCU「replaces the previously provided…spreadsheet」+ Utilization 三态 + .ncu-occ | NCU 1230/1240 | ✅ 逐字（3.4 §6） |
| asyncEngineCount >0 / =2 双向 + page-locked | PG13.0 1860/1863 | ✅（3.3 §4） |
| CUDA_DEVICE_MAX_CONNECTIONS 1-32 默认 8 / CC3.5+；COPY_CONNECTIONS CC8.0+ | PG13.0 15319-15324 | ✅ 逐字（3.3 §2） |
| stream in-order / implicit sync NULL stream / default stream synchronize blocking | PG13.5 6990/7080/7015 | ✅（3.3 §2） |
| event timestamp / non-recorded always success / false dependence workqueue | PG13.5 6653/8346/16984 | ✅（3.3 §2/§3） |
| stream priority hints 不抢占运行中 / Compute Preemption 指令级 context 开销 | PG13.0 1935/4661 | ✅ 逐字（3.3 §6） |
| graph 定义/capture 不入队/rejoin origin/Major changes re-instantiation/Graph Update/conditional device eval/Physical memory mapping+cudaGraphUpload/launch overhead driver preparation | PG13.5 12361/12532/12565/13048/13052/13249/14227/12369 | ✅（「Major changes」确为 13.5 逐字；graph 定义「new model」= 13.0，F2） |
| blog 2.9/9.6/3.8/3.4μs + 400μs instantiate + 0.02μs(1000×20) + 33% first launch + pre-initialises work descriptors + V100/CUDA10.1/500000/512 | blog 58/72/88/112/122 | ✅ 逐字（3.3 §5） |

### 4.3 机械规范扫（全过）

| 检查 | 结果 |
| --- | --- |
| display 反转义 9 形（`\, \! \; \: \{ \} \| \% \#`，fixed-string） | 0 命中 ✅ |
| 数学白名单（`\operatorname`/align/equation/gather/boxed/newcommand/color） | 0 命中 ✅ |
| process-leak（T1-T7/evaluator/writer/builder/QA 内部 ID） | 0 命中 ✅（「波次」为合法技术词非泄漏） |
| AI 味禁令词 + rubric A2/A3/B1（强大的/革命性/其实/我们不妨/一些/很多/显著…） | 0 命中 ✅ |
| 公式内 `<`→`\lt`（P26） | math 用 `\lt`×4 正确；裸 `<` 仅 4 处均在 C++ for 循环代码块（豁免）✅ |
| 裸 reserved symbol（$n$/$N$/$B$/$T$ math） | 0 命中；$M$=传输字节（spec 语义）正确；T_M/T_N tile 维；R/R_th + L/L_reg 混场声明 ✅ |
| 练习格式（四档难度/非递减/≥1[10][20]/≤1[40]/校验锚/≥1 证命题/先预测） | 4 章全过：3.1[10,20,30,30]、3.2[10,20,30,30]、3.3[10,20,20,30]、3.4[10,20,30,30]；ex3/3/2/3 各证命题 2/1/1/2；校验锚齐；先预测齐 ✅ |
| 总结表回指列 + 模板节（总结/练习/参考） | 4 章均「论断|公式/数值|回指」+ 3/3 节 ✅ |
| R2-1 节首（问句/数值/事实，禁「本节介绍」） | §1-§7 全合规 ✅ |
| R2-8 术语判断句定义（所谓 X（English）） | 全合规 ✅ |
| CJK 篇幅（标准章 ≥3500） | 3.1=4791 / 3.2=5560 / 3.3=5152 / 3.4=4175 ✅ |
| ⟨verify⟩ 残留 | 章内 0（PCI-SIG 精确 URL 留 W4，按口径不计 finding）✅ |
| ASCII 结构图 | 每章 ≥1 ✅ |

### 4.4 覆盖率声明
- 数值账：100% 独立 Python 重算（3.1/3.2/3.3/3.4 全部 display 公式 + 案例表 + 练习校验锚），全部对上。
- 一手事实：~90%+ 载重逐字断言自取一手来源对照（§4.2 共 30+ 项），全部语义正确；唯一系统精度问题 = 引文版本归属（F2，S3）。
- recap 边界：100%（1.2/1.4/inference 3.4/inference 1.4 四接口 + 越界检查），全 PASS。
- 机械规范：100% 扫（rescape/whitelist/process-leak/AI 味/symbol/练习/总结/R2-1/R2-8/篇幅），全过。
- 未覆盖（明示）：cuda-samples repo 教学例路径未本轮重取（research pass §2.9 已 GitHub API master 树路径级核实，本轮信其记录）；PCI-SIG 精确 URL（W4 残留）；BP/NCU 个别非载重 prose 未逐字。

## 5. 修复建议优先级
1. **F1（S2，须修）**：references.md 回写 8 处被引 + 订正 gpu-w2-sources.md §5 不实表述。簿记，不影响章内事实。
2. **F2（S3，可选）**：volatile/fence/syncthreads/atomic/local/mallocAsync/UM/graph-definition 引文标「13.0 逐字 / 13.5 同义」。语义已正确。
3. **F3（S3，可选）**：3.3 头条「3.2 ms」改「端到端串行 3.2 ms（拷贝 3.15 + kernel 0.06）」对齐 §4 分解。

**评审结论**：P3 四章事实与推导干净（S1=0），重推导 3.4 + recap 边界全 PASS，规范全过。唯一须修为 refs 被引回写簿记（F1/S2）；两条 S3 为精度增补。建议 F1 修复后本章组通过。
