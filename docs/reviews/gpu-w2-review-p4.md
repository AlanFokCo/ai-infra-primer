# gpu W2 Part 4 对抗审查档案（T6 · fresh-context evaluator · kernel 访存优化 4.1–4.4）

日期：2026-11-25。审查对象：`gpu/04-kernel-memory-optimization/` 四章（4.1 访存合并与向量化 / 4.2 Shared memory 与 bank conflict / 4.3 Tiling 与数据复用【重】/ 4.4 异步拷贝与流水线），分支 gpu-w2，HEAD 3f9794e（T5 review fix round 1 后）。
审查身份：独立 fresh-context evaluator，不继承写作上下文，未派子代理；一手来源自行 curl 取得（WebFetch 沿 W2 记录未重试），关键数学独立 Python 重算（脚本 `/tmp/gpu-w2-t6-scripts/rederive.py`，216 项检查）。
输入：四章成品 + STYLE_GUIDE.md + gpu spec §5 Part 4 / §4.3 符号表 / §6.3 强制重推导清单 + R2 style sheet + rubric + `docs/research/gpu-w2-sources.md` + `gpu/references.md` P4 及跨 Part 共享锚点 + recap 源章（1.2/1.3/1.4/2.1/2.2/2.3/2.4/3.1/3.2/3.4）。

---

## 0. 结果摘要

| 项 | 结果 |
| --- | --- |
| S1（致命事实/公式错误） | **0** |
| S2（实质问题/局部事实错误） | **2**（4.3 bf16 装载 sector 数误用 fp32 口径；4.4 TMA 每 tile 发射条数 1 应为 2，含练习校验锚传播） |
| S3（minor） | **6**（4.4 swizzle 档位 recap 漏 96B；4.4 PipelineState 字段数 2 应为 3；4.4 producer_commit 骨架单参与实际签名不符；4.3 代码注释「转置存」自相矛盾；references.md 六条被引回写缺口；4.3 裸 r 简写未声明） |
| 强制重推导 4.3 tiling 强度 O(1)→O(T) | **PASS**（独立重算逐数吻合，见 §2.1） |
| 4.1 coalescing HARD CALIBER（W0 纠正方向） | **PASS**（distinct=32 真值 / ceil(span/32)=63 高估，方向正确，无回归，见 §2.2） |
| 4.4 multi-stage 重推导 + wait_prior 末 tile 语义 | **PASS**（见 §2.3/§2.4） |
| 一手事实核查（BP 13.5 / PTX 9.5 / Hopper wp V1.04 / CUTLASS 4.9.0 / NCU / PG 13.5+archive 13.0.0 / cuda-samples / Hopper tuning guide） | 40 组引句/数值全部核实；**2 处章节数值与一手不符即 S2-1/S2-2**；其余 PASS |
| recap 忠实性（2.2/2.3/2.4/3.1/3.4/1.3/1.4/2.1） | PASS（逐锚核对，无重推越界） |
| 两层走读纪律（4.3↔5.2 / 4.4↔5.2 / 4.2↔5.2） | PASS（机制片段级声明齐全，L2/rasterization/split-K/CUTLASS 全量走读均显式归 5.2，无预支） |
| 规范（lint / R2 / rubric / 模板 / 练习 / 数学白名单 / rescape / flanking / process-leak） | lint 4/4 OK；rescape/flanking/禁令词/process-leak 零命中；练习四档合规；CJK 篇幅达标（4.3=5703≥5500） |
| 符号 §4.3 | tile 一律 T_M/T_N/T_K/T_tile，无裸 T 表 tile（O(T) 为渐近记号，spec §6.3 自身用法）；n 无 rank 误占；bytes 不占字母；M/N/K 章首局部声明齐全；S↔n_stage 跨章翻译注齐全；裸 r 简写见 S3-6 |

**特别注（evaluator 独立核实推翻研究笔记的一处）**：4.2 §4「TMA swizzle 32/64/96/128 B 多档、128 B 档含 16-byte atomicity 子模式」**正确**——PTX ISA 9.5 §5.5.7 逐字含「96 byte swizzle mode」（ptx.txt L2312）与「16-byte atomicity sub-mode: In this sub-mode, the 16-byte of data is kept intact while swizzling」（L2331-2332），Table 14 列 96B Swizzling Mode/16B atomicity 合法组合。W2 research notes §2.2 该条只列 32/64/128 + 32-byte atomicity，**不完整**；章节比研究笔记更准确，不构成 finding。反向地，4.4 的「32/64/128」recap 漏 96B 才是缺口（S3-1）。

---

## 1. Findings 表

| # | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S2-1 | S2 | 4.3 §7 装载侧判定段（L242） | 「tiled 版 A tile 装载 k 最快、B tile 装载 n 最快，warp 内连续地址 → **各 4 sector/128 B**，η=1」——bf16 口径下 warp 32 lane × 2 B = 64 B 连续 = **2 sector/64 B**；4 sector/128 B 是 fp32 口径误代入 | 独立重算：32 个连续 bf16 字节地址 {2i} 的 distinct floor(a/32) = 2（rederive.py「bf16 warp load sectors (TRUTH)」PASS=2）；§7 代码装载为标量 `As[m][k] = A[...]`（__nv_bfloat16，每线程每迭代 1 元素）；章脊 bf16 bytes=2。基址对齐（K=4096 行距 8192 B、k0 步长 64 B）不改变 sector 数。η=1 结论不受影响，仅 sector/字节数错 | 改「各 2 sector/64 B，η=1」；naive 版 B 读同口径（如需补数亦 2 sector）。总结表与练习未引用此数，无需他改 |
| S2-2 | S2 | 4.4 §6 发射侧账（L151）+ §6 对照表「发射」行（L157）+ 总结表（L178）+ 练习 1(b) 校验（L191）+ §3 语义骨架（L56-66） | 「TMA 需 **1 条**指令（elected 线程）」「TMA 恒 1」「cp.async 1024 条对 TMA 1 条」——本章 16 KB tile = A tile 8 KB + B tile 8 KB，**两个独立张量、两个 tensor-map**（§3 自账「A/B 两个 tensor-map 共 256 B」），每 tile 需 **2 条** cp.async.bulk.tensor（A/B 各一）；「恒 1」应为「恒 2（与 tile 字节无关）」 | PTX §9.7.10.28.5：tensorMap 操作数描述单个张量，块坐标给单张量 tile；一条指令搬一个张量的一块。CUTLASS mainloop 对 A/B 分别发 TMA load（sm90_pipeline.hpp producer 路径 + efficient_gemm.md 层级）。§3 骨架单条 `cp.async.bulk.tensor.2d` 配 `expect_tx(mbar, 16384)`：单条 2D op 只完成 8192 B，tx-count 余 8192 不归零、barrier 永不放行（骨架虽标「语义级」，仍强化「1 条」误解）。练习 1(b) 校验「(b) 1」把错误传播进习题 | 数值改 2（A/B 各一条），「恒 2」；§3 骨架拆两条 TMA（A/B 各一，同一 mbarrier expect_tx 16384 或分 barrier 各 8192）；练习 1(b) 校验改 2。定性结论（cp.async O(bytes) 对 TMA O(1)、发射槽/地址 register 两笔省）不变 |
| S3-1 | S3 | 4.4 §3（L54）+ §6 对照表 swizzle 行（L161） | 「tensor-map 带 **32/64/128 B** swizzle 模式（4.2 §4 recap）」漏 96B，与 4.2 §4 的四档枚举（32/64/96/128）不一致，recap 不忠实 | PTX 9.5 §5.5.7 逐字含 96 byte swizzle mode（L2312）+ Table 14「96B Swizzling Mode / 16B」合法（L2437-2438）；96B 档限制（swizzle_atomicity 16B、Box-Size[0]≤96B 等，L11103-11108）不影响档位枚举。4.2 §4 与总结表已列 96B | 补「32/64/96/128 B」与 4.2 对齐；若刻意只列 sm_90 常用档须加口径注（PTX 该节未对 96B 作 sm_90 排除） |
| S3-2 | S3 | 4.4 §5（L123） | 「`PipelineState`（**index/phase 两个字段**，stage 索引循环、phase 翻转）」——4.9.0 实际**三字段** index_/phase_/count_ | sm90_pipeline.hpp L175-177（fresh main，48142 B）：`int index_ = 0; uint32_t phase_ = 0; uint32_t count_ = 0;`；count_ 供 PipelineTmaStore::producer_commit(state) 用（L699-700 state.count()）。W2 research notes 亦记「index/phase」，同漏 | 改「index/phase/count 三字段（count 辅助计数），stage 索引循环、phase 翻转」 |
| S3-3 | S3 | 4.4 §5 骨架（L133） | `pipe.producer_commit(state);` 单参调用与 PipelineTmaAsync 实际签名不符 | sm90_pipeline.hpp PipelineTmaAsync 仅有两个重载：`producer_commit(PipelineState, uint32_t bytes)`（L453）与 `producer_commit(PipelineState, UserDefinedArriveOp&&)`（L459）；单参 `producer_commit(PipelineState)` 属 **PipelineTmaStore**（L699）。No-Op 语义本身正确（L578「NOP for TMA based mainloop」+ pipeline.md 逐字已核） | 骨架改 `pipe.producer_commit(state, m_tile);` 或注「单参为示意简化，实际签名带 bytes/arrive-op」；避免与 PipelineTmaStore 混读 |
| S3-4 | S3 | 4.3 §7 代码注释（L201） | 「// A tile **转置存**：行 m 列 k，装载 k 最快」自相矛盾——`As[TM][TK]` 按 [m][k] 索引与 global A（M×K 行主序）**同向**，并非转置存储；转置的是计算侧访问模式（固定 k 跨 m 读列），不是存储 | 代码 `As[m][k] = A[(blockIdx.y*TM+m)*K + k0+k]`：global 行 m 内 k 连续 → shared 行 m 内 k 连续，同向。若真转置存应为 As[TK][TM] 按 [k][m] | 注释改「A tile 行 m 列 k（与 global A 同向），k 最快存、装载合并」 |
| S3-5 | S3 | gpu/references.md 被引章节字段（6 条） | 被引回写缺口：cuda-programming-guide 缺 **4.1/4.2**；cuda-best-practices 缺 **4.4**；nsight-compute 缺 **4.2**；hopper-microbench 缺 **4.2/4.3/4.4**；cuda-samples 缺 **4.4**；cutlass-docs 缺 **4.4** | 逐条对照四章 ## 参考 节：4.1 引 PG（archive 13.0.0 Device Memory Accesses）；4.2 引 PG（shared/同址写）+ NCU（§2.11.1 wavefronts）+ Luo；4.3 引 Luo；4.4 引 BP（§10.2.3.4）+ NCU 无、Luo（t_lat 口径）+ cuda-samples（globalToShmemAsyncCopy）+ cutlass-docs（pipeline.md）。references.md 现行被引列：PG「…3.4, 4.4, 7.3」/ BP「…4.1, 4.2, 4.3」/ NCU「…3.4, 4.1, 11.2」/ microbench「1.3, 1.4, 2.2, 2.3, 2.4, 3.2」/ samples「3.1, 3.2, 3.3, 4.1, 4.3, 5.1, 5.3, 7.3」/ cutlass-docs「4.3」。W2 research notes §5 声称「被引已含 4.1-4.4 对应章不变」与事实不符。lint 检查项 4 只验条目存在，故 LINT OK 不覆盖此字段 | 按章回写六条被引列（spec §6.1.4 七字段 schema）；parse-check 复核 groups/entries 不变 |
| S3-6 | S3 | 4.3 §4 表题（L95）/§5（L129）/总结（L259/261）/练习 3(c)/4(d) | 裸 $r$ 简写（r=8、r=16）系统使用但未声明——符号表只声明 $r_M, r_N$；方阵时 $r = r_M = r_N$ 的简写缺一句定义 | §4「$T_M=T_N=128$、$r=8$、256 线程」首现即裸 r；§5「保 256 线程则 $r=16$」「保 $r=8$」；总结与练习同。spec §4.3 未禁 r，但「未定义符号」属教学审查项 | §4 首现处加「方阵 register tile 简写 $r = r_M = r_N$」一句（或符号表补） |

无 S1。无越界（serving 调度/引擎编排/K8S 内容零命中）。无已交付公式重推越界（models 2.4 roofline 入口、1.3 MMA 几何、3.4 occupancy 式 (1)、1.4 Little's law 均 recap + 指引，增量合规）。

---

## 2. Verification log（强制）

### 2.1 强制重推导 4.3 tiling 强度 O(1)→O(T) —— **PASS**

独立 Python 重算（不依赖章节推导路径，从计数第一性原理出发），脚本 `/tmp/gpu-w2-t6-scripts/rederive.py`：

| 量 | 独立重算 | 章节值 | 判定 |
| --- | --- | --- | --- |
| naive 强度 I = 2K/(2K·bytes) = 1/bytes（bf16） | 0.5 FLOP/byte | 0.5 | PASS |
| naive 流量 2MNK·bytes | 274.88 GB | 274.88 GB（metadata 274.9 四舍五入一致） | PASS |
| naive 带宽口径时间 | 82.00 ms | 82.00 ms | PASS |
| FLOPs 2×4096³ | 137.4 GFLOP | 137.4 | PASS |
| 算力上限时间 @989.5 TFLOPS | 138.9 μs | 138.9 | PASS |
| naive/算力比 | 590.4 ≈ R/I = 295.2/0.5 | 约 590 | PASS |
| **I_blk = 2T_M·T_N/((T_M+T_N)·bytes)**（per-K-step 计数：装载 (T_M+T_N)T_K 元素、FLOPs 2T_M·T_N·T_K，T_K 约去） | 2×128×128/(256×2) = **64** | 64 | PASS |
| 方阵化简 T_tile/bytes | 128/2 = 64 | 64 | PASS |
| 复用因子 A×T_N / B×T_M | 128 / 128 | 128 / 128 | PASS |
| **traffic = MNK(1/T_M+1/T_N)·bytes**（block 数 (M/T_M)(N/T_N) × 每 block (T_M+T_N)K 元素） | 4096³×(2/128)×2 = **2.147 GB** | 2.147 GB | PASS |
| tiled 带宽口径时间 | 640.7 μs | 640.7 | PASS |
| 流量比 naive/tiled | 128 | 128× | PASS |
| C 写回 MN·bytes | 33.55 MB，占 1.56% | 33.55 MB / 1.56% | PASS |
| 加 C 写时间 | 650.7 μs | 650.7 | PASS |
| tile 扫描表 T=32/64/128/256 | I=16/32/64/128；8.590/4.295/2.147/1.074 GB；2562.6/1281.3/640.7/320.3 μs | 逐行同 | PASS |
| ridge 阈值 T≥2R（bf16） | 精确 2R=590.39；章用 R≈295 口径给 590 | 590 | PASS（≈口径内取整，非错误；精确值 591 才严格达 ridge，差 0.4 在 R≈295 声明容差内） |
| register tiling I_reg = 2r_M·r_N/((r_M+r_N)bytes) | 8/2 = 4 | 4 | PASS |
| 线程闭合 256×64 = 128×128 | 16384 | 16384 | PASS |
| 三级账每单 k 步：HBM 512 B / shared 8192 B / FLOPs 32768 | I=64 / 4；放大 16× | 同 | PASS |
| shared ridge 7496/253 / register ridge 7496/4055 | 29.6 / 1.85 | 29.6 / 1.85 | PASS |
| warp distinct 流量 (2×8+16×8)×2 | 288 B；请求口径 1024 B；省 3.56×；有效强度 4096/288=14.2 | 同 | PASS |
| 精度不变性 I/R（fp8 bytes=1、峰值 2×） | (128/590)/(64/295)=1 | 不变 | PASS |
| S_sh = n_stage(T_M+T_N)T_K·bytes | 单 16 KB / 双 32 KB（T_K=32）；32/64 KB（T_K=64） | 同 | PASS |
| occupancy 回代（S_alloc=ceil(S/128)×128+1024，总 233472） | 双 T_K=32：33792→shared 6 / reg 2 / warp 8 / slots 32 → min 2，W=16，O=0.25；双 T_K=64：66560→shared 3，min 2；单 T_K=32：17408→shared 13 | 逐数同 | PASS |
| 命题 3 反例 128×256 对同周长方阵 192 | 85.33 < 96 | 85.33 / 96 | PASS |
| T=256 register 悬崖 | r=16→256 regs>255；r=8→1024 线程、131072>65536 | 同 | PASS |
| **double buffering τ=max(t_comp, t_lat+t_load)** | t_comp=139.9 / t_load=645.2 / t_lat=261.6 ns → τ_db=906.8、τ_sync=1046.7、加速 1.154；T_K=64：279.8/1290.4/1552.0/1831.8、1.180 | 逐数同 | PASS |
| 完全隐藏 iff t_comp≥t_lat+t_load；必要条件 t_comp≥t_load ⟺ I_blk≥R | 代数等价验证（两边代式 (6)，F_SM/BW_SM = F_peak/BW = R） | 同 | PASS |
| 加速比上界 2（max(a,b)≥(a+b)/2） | 成立 | 同 | PASS |
| BDP 每 SM | 25.394 GB/s × 261.64 ns = 6644 B = 6.64 十进制 kB ≈ 6.6（1.4 逐字「6644 B ≈ 6.6 KB」且芯片级注明十进制）；二进制 6.49 KiB | 6.6 KB（recap 1.4） | PASS（recap 忠实；16 KB>BDP 结论两口径均 robust） |
| T_K 延迟账 | 139.9<261.6（盖不住）/ 279.8>261.6（盖住） | 同 | PASS |
| 教学 GEMM 冲突账（逐 (m,k)/(k,n) 穷举模拟） | As 读全 2 路（ty 两行字距 128≡0 mod 32）；Bs 读全 2 路（bank 4tx mod 32，tx 与 tx+8 同 bank）；练习 4(e) 字号 0/128 同 bank 0 | 同 | PASS |

**结论：4.3 强制重推导 PASS**（强度公式、流量公式、register 再升、预算约束、double buffering 条件与数值全部独立吻合；唯一数值偏差在 §7 装载侧 sector 数，属 4.1 口径误代入，另立 S2-1，不影响强度/流量主账）。

### 2.2 4.1 coalescing HARD CALIBER（W0 纠正方向，MUST NOT regress）—— **PASS**

独立重算 stride-16 fp32（字节 stride 64，基址对齐）：
- lane i 地址 64i → sector 号 floor(64i/32) = **2i**（偶数序列 0,2,...,62）→ **distinct = 32**（真值）。
- 跨度 = 64×31+4 = **1988 B** → ceil(1988/32) = ceil(62.125) = **63**（高估 31 个 sector，近 2 倍）。
- 方向判定：**ceil(span/32)=63 高估，distinct=32 是真值**。

章节逐处核对（方向全部正确，无回归）：
| 位置 | 章节表述 | 判定 |
| --- | --- | --- |
| §2 反例·高估 | 「真值 32，跨度公式给 63，高估 31 个 sector（近 2 倍）」 | PASS |
| §2 ASCII 图 | 「s=16 → 32 sector，流量 1024 B，η=1/8；跨度 1988 B，ceil(跨度/32)=63≠32——跨度公式高估」 | PASS |
| §2 反例·低估 | misaligned 连续 128 B 偏移 4 B：[4,132) 跨 sector 0–4 共 5，ceil(128/32)=4 漏首尾残 sector | PASS（独立重算 5/4 吻合） |
| §2 坑注 | 「大 stride 高估（stride-16 给 63 对真值 32）、misaligned 低估（4 对 5）」 | PASS |
| 命题 1 式 (1) | n_sec = |{floor(a_i/32)}|（distinct 集合基数） | PASS |
| 命题 2 式 (2) | η=1/min(s,8)，s≥8 饱和 1/8（32 sector 地板来自 lane 数） | PASS（s=1..16 全表独立重算吻合；floor(31s/8)+1=4s 恒等式 s=1..8 逐值验证、s=9 反例成立） |
| 总结表 / 练习 2(a) | 「63 对真值 32（高估）」 | PASS |
| §6 带宽账 | s=16 与 s=8 同 320.33 μs（饱和）；无界 1/s 估 209.5 GB/s 对地板 419 GB/s 低估 2 倍 | PASS |

另核：float4 对齐 16 sector、偏移 16 B 17 sector η=16/17、fp64 s=4 饱和、AoS s=4 η=1/4 838 GB/s、V100 790×4/5=632、t_copy 全表 40.04/80.08/160.16/320.33 μs、n_v4=4194304、launch 16384/65536 blocks、buffer s=16 每侧 1 GiB——独立重算全部吻合。

### 2.3 4.4 multi-stage 重推导 —— **PASS**

- τ_S = max(t_comp, t_load, (t_lat+t_load)/(S−1))：S=2/3/4 代入 T_K=32 得 906.8/645.2/645.2 ns、T_K=64 得 1552.0/1290.4/1290.4 ns——逐数吻合。
- 模型自洽性独立核查：稳态 τ=t_load 地板时发射率=服务率，相邻装载服务段首尾相接（FCFS）、延迟段重叠，三项 max 成立；「S−1 份均分带宽」替代模型（τ≥t_load+t_lat/(S−1)）仅在服务段重叠时适用，与稳态地板条件不相容，章节口径正确。
- S_min = 1+ceil((t_lat+t_load)/max(t_comp,t_load))：T_K=32 → 1+ceil(1.4055)=3；T_K=64 → 1+ceil(1.2027)=3。不等待条件 (S−1)τ_S≥t_lat+t_load：2×645.2=1290.4≥906.8 ✓。
- S=2 退化 4.3 命题 4 ✓；S→∞ 极限 max(t_comp,t_load)=645.2 ✓；对同步加速 1046.7/645.2=1.622、1831.8/1290.4=1.420 ✓。
- 容量：S=3 T_K=32 → 48 KB、S_alloc=50176、shared 项 floor(233472/50176)=4；T_K=64 → 96 KB、shared 项 2；register 项 2 更紧、occupancy 0.25 不变 ✓。
- 延迟主导推论：S−1≥261.6/139.9=1.87→S≥3；261.6/279.8=0.93（章节用显示值舍入，独立重算 0.9349）→S≥2 ✓。
- 发射/记账：16384/16=1024 条 cp.async、4 条/线程；expect_tx A 8192 + B 8192 = 16384；descriptor 2×128=256 B、占流量 256/2.147e9=1.19e-7≈1.2e-7 ✓。**TMA 条数 1 应为 2 → S2-2**。

### 2.4 4.4 wait_prior 末 tile 语义 —— **PASS**

独立模拟（group pending 模型，wait_prior(N) 完成除最新 N 组外全部）：
- 骨架 `wait_prior(k0+TK<K ? 1 : 0)`：非末 tile 留 1 组（预取）在飞、等当前 tile 组完成 ✓；末 tile 无预取、仅 1 组在飞，wait_prior(0) 等其完成 ✓。模拟 128 步全程读已落地 shared（read_unlanded=False）。
- 反例模拟：末 tile 误用 wait_prior(1) → 「至多 1 组未完成」已满足（1≤1）立即返回、计算读未落地 shared（read_unlanded=True）——与章节坑注「按语义因 1 不大于 1 立即返回、根本不等完成」逐字一致 ✓。
- 双 __syncthreads() 位置（wait 后跨线程可见 + 读毕防覆写）与 PG async-copies 逐字「if you use LDGSTS to prefetch some data that will be shared with other threads, a __syncthreads() is necessary after synchronizing with the LDGSTS completion mechanism」同向 ✓。

### 2.5 一手事实核查（curl 主通道，全部 200）

来源版本确认：BP **v13.5**（bp.html 433269 B）；PTX ISA **9.5**（ptx.html 3965886 B，§1.3 Version 9.5）；Hopper whitepaper **V1.04**（dam-cdn PDF 7880278 B）；PG **13.5 官方 PDF**（12784076 B）+ **archive/13.0.0** 单页（1178393 B txt）；CUTLASS **main @ 4.9.0（2026-11-13）**（CHANGELOG 逐字）；cuda-samples master（README 支持 CUDA 13.5）。

| # | 核查项 | 来源/位置 | 结论 |
| --- | --- | --- | --- |
| 1 | coalescing CC6.0+「number of 32-byte transactions necessary to service all of the threads of the warp」 | BP §10.2.1（bp.txt L512） | PASS 逐字 |
| 2 | access unit 32-byte 与 L1 caching 无关 | BP L515 | PASS |
| 3 | adjacent 4-byte words → four 32B transactions；subset 仍搬整 segment；permuted 仍 four | BP L520/524 | PASS |
| 4 | misaligned sequential → five 32-byte segments | BP §10.2.1.2 L526 | PASS |
| 5 | V100 offset 非 8 words 倍数 → 790 GB/s、4/5 吞吐 | BP §10.2.1.3 L541 | PASS |
| 6 | strideCopy kernel 同形 + stride 2 → 50% + 「32 32-byte segments」饱和 | BP §10.2.1.4 L547/555 | PASS |
| 7 | 32 banks × 32 bits/cycle、successive words → successive banks、warp 32 = banks 32（CC5.x+） | BP §10.2.3.1 L617 | PASS |
| 8 | conflict 串行化 factor = separate requests 数 + broadcast/multicast 例外 | BP L615 | PASS |
| 9 | 「Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses」 | BP L619 | PASS |
| 10 | padding w+1=33 ≡ unit stride + transposedTile[TILE_DIM][TILE_DIM+1] + 列写冲突「every thread of the warp hits the same bank」 | BP §10.2.3.3 L727-729 | PASS |
| 11 | V100 C=AAT 三级 12.8/140.2/199.4 GB/s | BP L706/726/735-739 | PASS |
| 12 | V100 C=AB 三级 119.9/144.4/195.5 GB/s + 「not due to improved coalescing ... avoiding redundant transfers」+ cache line 逐出逐字 + shared broadcast 逐字 | BP §10.2.3.2 L638/660/679/642 | PASS |
| 13 | async copy 免 intermediate register 逐字 + 16 B bypass L1 逐字 + __pipeline_wait_prior 官方例 | BP §10.2.3.4 L786/779 | PASS |
| 14 | 有效带宽 (Br+Bw)/1e9/time；legacy Global Memory Load/Store Efficiency 逐字 | BP §9.2.2 L350-353 / §9.2.3 L367 | PASS（章节节号正确；research notes「§9.1.1 附近」不确，章节为准） |
| 15 | BP 节号 10.2.1.1-4 / 10.2.3.1-4 / 9.2.2-3 | bp.txt 目录 L307-828 | PASS |
| 16 | ld .v2/.v4 ≤128-bit（.v4 .f64 not allowed）+ ld.v4.b32 16 B + 自然对齐违反 UB（掩低位/fault） | PTX §9.7.10.8 L1829/2497-2498 | PASS |
| 17 | cp.async cp-size{4,8,16}、PTX 7.0、**sm_80+**、.ca/.cg、non-blocking、完成只认 async-group/mbarrier 逐字、组间无序逐字 | PTX §9.7.10.28.3.1 L10476-10484 + Target ISA Notes | PASS |
| 18 | commit_group per-thread batch 逐字 / wait_group N「wait till only N or fewer of the most recent ... pending」逐字 / wait_all≡commit+wait 0 | PTX §9.7.10.28.3.2/3 L10521/10548 | PASS |
| 19 | cp.async.bulk.tensor .1d-.5d、PTX 8.0、**sm_90+**、.tile/.im2col、global→shared .mbarrier::complete_tx::bytes / shared→global .bulk_group、.multicast::cluster | PTX §9.7.10.28.5.3 L11177+ | PASS |
| 20 | **tensor-map 128-byte opaque object**（.const/.param/.global） | PTX §5.5.8 L2467 | PASS |
| 21 | swizzle 模式 32/64/**96**/128 B + 128 B 档 **16B**/32B/32B+8flip/64B atomicity 子模式 + Table 14 | PTX §5.5.7 L2238-2443 | PASS（4.2 正确；research notes 不完整；4.4 漏 96B → S3-1） |
| 22 | mbarrier .b64 8B-aligned .shared + phase/pending arrivals/tx-count + user-defined 数量仅受 shared 容量限 | PTX §9.7.15.16 L14320-14359 | PASS |
| 23 | ldmatrix warp collective load + 八地址八线程 + 每地址行首 + 「Consecutive instances of row need not be stored contiguously」 | PTX §9.7.16.5.15 L17473/17524 | PASS |
| 24 | TMA copy descriptor 逐字 + 1D-5D + single-threaded elected（cuda::memcpy_async / cuda::barrier）+ A100 LoadGlobalStoreShared vs Hopper 硬件接管 stride/offset/boundary | Hopper wp V1.04 L1085-1119（p.32-34） | PASS |
| 25 | Async Transaction Barrier「it also counts transactions ... The transaction count is essentially a byte count」+ sum reaches expected value + Hopper wait sleep vs 前代 spin | wp L1149-1159 | PASS |
| 26 | CUTLASS swizzle.hpp「AA = ZZ xor YY」+ Swizzle<BBits,MBase,SShift> + apply「offset ^ shiftr(offset & yyy_msk{}, msk_sft{})」+ static_assert(abs(num_shft)>=num_bits) | GitHub contents API main，16120 B，L52/57-63/67-68/78 | PASS（Swizzle<3,4,3> 掩码 bits7-9→bits4-6 与 4.2 式 (4) 逐偏移等价，0..4095 穷举） |
| 27 | sm90_pipeline.hpp PipelineState/PipelineTmaAsync<Stages,ClusterTransactionBarrier>/full_barrier_[Stages]/empty_barrier_[Stages]/producer_acquire/commit/expect_transaction/consumer_wait/release | 48142 B，L171/289/299-300/443/453/480/498/508 | PASS（字段数/commit 签名两处 → S3-2/S3-3） |
| 28 | PipelineTmaUmmaAsync 不在当前 main（4.4 坑注） | grep Umma/UMMA = 0 命中 | PASS |
| 29 | PipelineTmaStore/PipelineTransactionAsync/PipelineAsync 同文件 | L673/784/1033 | PASS |
| 30 | pipeline.md「In order to write a performant GEMM Kernel, software pipelining is critical...」+ producer/consumer 角色定义 + producer_commit TMA No-Op 逐字 | 8954 B，L53-66/107-109 | PASS |
| 31 | efficient_gemm.md「A larger threadblock means fewer fetches ... DRAM bandwidth does not become a bottleneck」+ ThreadblockShape::{kM,kN,kK} + 大 tile 形状失配/split-K 段；gemm_tutorial.md 不存在 | 20550 B，L66-81；目录 grep=0 | PASS |
| 32 | NCU「Each instruction generates exactly one request for LSU traffic」+ Sectors/Req 最优 32-bit:4/64-bit:8/128-bit:16 + higher numbers uncoalesced + Bytes=sectors×32B + shared wavefronts 串行 | ProfilingGuide §2.11.1/2 L1722/1744/1753/1758 | PASS |
| 33 | NCU achieved/ideal/excessive 定义（ideal=每非 predicate-off 线程给定宽度、excessive=盈余） | NsightCompute 主文档 L975 | PASS |
| 34 | PG 32/64/128B naturally-aligned transactions + 「throughput is divided by 8」+ cudaMalloc ≥256B 逐字 + cudaMallocPitch pad 逐字 + float4 对齐 16（Table 7） | **archive/13.0.0 §8.3.2** L4855-4877/1296/15056（13.5 PDF 已无 Device Memory Accesses 节——章节引 archive 正确） | PASS |
| 35 | PG 13.5 async-copies：cuTensorMapEncodeTiled driver API host 创建 + __grid_constant__ const 参数 + LDGSTS CC8.0+ 4/8/16B + 16B L1 bypass + wait 后 __syncthreads 两步 | pg-async.txt L535/846/97-102 | PASS |
| 36 | Hopper tuning guide TMA 1D-5D global↔shared + cluster DSMEM + 单线程发射 | §1.4.1.2 L60-65 | PASS |
| 37 | cuda-samples cpp/6_Performance/alignedTypes、cpp/0_Introduction/matrixMul、cpp/3_CUDA_Features/globalToShmemAsyncCopy 存在 + README「implements matrix multiplication ... compute capability 8.0 or higher. Also demonstrates arrive-wait barrier」逐字 | GitHub contents API master | PASS |
| 38 | H100 脊柱 3352 GB/s / 989.5 TFLOPS dense / 132 SM / 1.83 GHz TC-boost / 1.98 GHz FP32-boost / 228 KB shared / 65536 regs / 255 per-thread | recap 源章 2.1/2.4/2.3/3.4 对 W1 已核 wp Table 3/4 + datasheet | PASS |
| 39 | Luo et al. 2402.13499 shared 29.0 / HBM 478.8 cycles [B] H800 口径声明 | references.md hopper-microbench 条 + 各章实测口径声明 | PASS |
| 40 | 4.3 §7 bf16 装载 sector 数 / 4.4 TMA 条数 | 独立重算（§2.1/§2.3） | **FAIL → S2-1 / S2-2** |

### 2.6 recap 忠实性核对（逐锚）

| recap 接口 | 源章锚点 | 判定 |
| --- | --- | --- |
| 4.1 ← 2.2 §3 line 128B/sector 32B、命题 2 流量=distinct×32B、式 (2) η 带宽口径、式 (3) BW_eff=η·BW | 2.2 L89-101 逐字同；2.2 的 s=distinct sector 数 → 4.1 译 n_sec 翻译注 | PASS |
| 4.1 ← 3.1 n_vec=16777216 / 65536×256 launch | 3.1 章脊 L9 同 | PASS |
| 4.1 ← 2.1 式 (1) 3352 GB/s | 2.1 式 (1) 栈几何 | PASS |
| 4.2 ← 2.3 §3 32 banks×4B / stride-32 7.3 GB/s / 29.0 cycles / 双 clock 1.83/1.98 / 三旋钮 / 48KB / 227KB / 255 / 同址写 undefined / 放置原则 §2 / 4KB 副本例 | 2.3 L47/56/58-62/9/41 逐一对上 | PASS |
| 4.2 ← 1.3 wgmma B shared 描述符直喂 | 1.3 §5 L125/119 | PASS |
| 4.3 ← 2.4 ridge≈295 / BW_SM 25.4 / F_SM 7.496 / 160:10:1 / register 4055 需求上界 / shared 253 / 式 (4) 归一化 / 式 (8) I_ach / §7 几何解读 | 2.4 L95-125/222/286 逐一对上 | PASS |
| 4.3 ← 3.4 式 (1) 四约束 / S_alloc 128B+1KB / 233472 / 命题 2 FP32 16 warps / §5「只有绑定约束松紧改变 occupancy」 | 3.4 L21-35/110/136 逐字同 | PASS |
| 4.3 ← 1.3 §4 A×n/B×m 复用 / 式 (4) 累加器 N/2 / §5 wgmma | 1.3 L103/135-140/125 | PASS |
| 4.3 ← 1.4 6.6 KB BDP | 1.4 L114「6644 B ≈ 6.6 KB」（十进制口径） | PASS |
| 4.4 ← 4.3 式 (5)(6)-(8) / 命题 4 / 时间账 139.9/645.2/261.6 | 4.3 §5/§6 逐数同 | PASS |
| 4.4 ← 3.3 stream 级 overlap 对照 | 3.3 存在、copy engine/H2D 口径 | PASS |
| 已交付公式不重推（models 2.4 roofline 入口 / 1.2 occupancy 上界 / 1.4 Little's law） | 各章均「recap + 指引，不重推」声明 + 只代值 | PASS |

### 2.7 规范审查（rubric + R2 + 模板 + 数学）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| lint-chapter.sh 六项 | bash 逐章 | 4/4 LINT OK |
| rescape 禁写（\, \! \; \: \{ \} \% \# \|） | raw grep -F 全章 | 零命中 |
| 非白名单命令（\operatorname/\begin{align}/\boxed/\newcommand/\medspace/\thickspace） | grep -F | 零命中 |
| 公式内裸 < 紧跟字母 | Python display+inline 扫描 | 零命中（\lt 三处用法正确：4.1 练习 3、4.3 §3/总结） |
| 开口/闭口 flanking + 内侧空格 | Python 配对扫描（剔除 code/display，backtick 豁免） | 0 issues |
| rubric A 禁令词（值得注意的是/综上所述/让我们/其实/呢/吧/嘛 等 20 词）+ 营销形容词 | grep -E | 零命中 |
| 空洞过渡/评论开场节 | grep + 逐节首句 | 零命中；全部 ## 节首两句内问题/数值开场（R2-1） |
| process-leak（writer/evaluator/T5/review/W2/波次/brief/lint） | grep -E | 零命中 |
| R2-2 章脊口径 | 逐章 | 四章均「本章口径：H100 SXM CC 9.0」+ 精度/running example/双 clock/实测 [B] 声明；换口径（T_K=64、V100 时代锚）均有原因句 |
| R2-3 公式后 300 字内全数字实例 | 逐 display 块 | 全部落地（式 (1)-(8) 各章逐式核对，代入到带单位数值） |
| R2-4 命题-证明骨架 | 逐命题 | 4.1 命题 1/2、4.2 命题 1/2/3+推论、4.3 命题 1/2/3/4、4.4 命题 1 均骨架齐全；降级「留作练习 N」的练习均存在（4.1 练习 3、4.2 练习 3、4.3 练习 3(a)、4.4 练习 3） |
| R2-5 路线句 | 逐推导段 | ≥4 步推导均有路线句（技巧+步数+关键一步） |
| R2-6 定义配例+边界反例 | 逐定义 | coalescing/tiling/复用因子/swizzle/软件流水/AoS-SoA 均判断句定义+正例；全称论断配反例（broadcast 边界、padding p=2、跨度双向、缓存不改阶） |
| R2-7 口径注 | 逐窄口径词 | η 两口径、服务周期 vs 延迟、per-block vs 整 kernel、「免费」、稳态、请求 vs 服务、时代口径均注 |
| R2-10/11 提点/坑块引用 | 逐章 | 4.1 坑×3、4.2 坑+提点、4.3 坑×2+提点、4.4 坑×2，均含触发条件/现象/规避，可删性满足 |
| R2-12 总结表回指列 | 逐行 | 四章总结表每行含式/§/练习回指，论断可判定 |
| 练习分级 | 逐章 | 四章均 [10]/[20]/[30]/[30] 非递减；≥1 [10]/[20]；无 [40]；每计算/代码题带校验锚；≥1 题证正文命题；先预测后算句齐全 |
| 模板 | lint + 目视 | metadata 定位|前置|锚点行、总结/练习/参考齐全 |
| 篇幅（正文 CJK，剔除代码/公式） | Python 计数 | 4.1=4039 / 4.2=4241 / **4.3=5703（≥5500 软目标达）** / 4.4=3865，均 ≥3500 |
| ASCII 结构图 | 逐章 | 4.1 sector 图、4.2 bank 映射图、4.3 tiling 数据流图、4.4 流水时序图各 ≥1 |
| 符号 §4.3 | grep + 目视 | 无裸 T 表 tile（O(T) 渐近记号除外，spec §6.3 自身用法）；无裸 n 占 rank；bytes 不占字母；M/N/K 章首声明；B 不表 tile；S/n_stage、m_tile/M、τ_S/τ 跨章翻译注齐全；裸 r → S3-6 |
| 两层走读纪律 | 逐边界声明 | 4.3/4.4/4.2 均声明机制片段级 + 5.2 全量回链；L2/rasterization/split-K/wgmma tile 选型/CUTLASS 模块走读显式归 5.2，无预支 |
| 越界（serving/K8S） | grep | 零命中 |

### 2.8 抽查覆盖率

- 四章全部 display 公式（4.1 三式 / 4.2 四式 / 4.3 八式 / 4.4 两式）100% 独立重算或代数验证。
- 四章全部数值表（4.1 stride 表+copy 表+向量化表 / 4.2 冲突表+transpose 账 / 4.3 扫描表+三级表+双缓冲表 / 4.4 stage 表+对照表）100% 重算。
- 四章全部官方逐字引句（约 40 组）100% 对一手原文（§2.5）。
- 练习校验锚 100% 重算（含 4.2 练习 4(e) 逐 lane bank 模拟、4.3 练习 4(e) 冲突穷举）。
- recap 接口 12 条逐源核对（§2.6）。
- Python 检查 216 项：210 直接 PASS；6 项 FAIL 经复核全部为脚本口径/舍入问题而非章节错误（3 项 BW 单位 B/s vs GB/s 脚本 bug；ridge 阈值 590 vs 591 属 R≈295 取整口径；BDP 6.5 vs 6.6 属十进制/二进制 KB，1.4 明示十进制；0.93 vs 0.94 属显示值舍入顺序）——章节数值在声明口径内全部成立。

---

## 3. 判定与移交

- **S1=0 / S2=2 / S3=6**。S2 两条必须修复（S2-1 单句数值；S2-2 数值+骨架+练习锚联动）；S3 六条按修复轮逐条判定（修复成本低，建议全修）。
- 强制重推导 4.3 **PASS**、coalescing HARD CALIBER **PASS**（W0 纠正方向无回归）、4.4 multi-stage 与 wait_prior 语义 **PASS**。
- 修复后复审范围：4.3 §7（S2-1/S3-4/S3-6）、4.4 §3/§5/§6/总结/练习 1（S2-2/S3-1/S3-2/S3-3）、references.md 被引列（S3-5）。
- evaluator 独立核实纠正研究笔记一处：PTX 9.5 swizzle 含 96B 档与 16-byte atomicity 子模式，4.2 表述正确（research notes §2.2 该条不完整，建议修复轮顺带补全笔记，非章节 finding）。
- 本档案仅记录，未改任何章节或 references。
