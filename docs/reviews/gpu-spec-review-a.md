# gpu 维度 Spec 对抗审查 · 路线 A：P1–P4 GPU 微架构 / 内存层级与带宽账 / CUDA 编程模型 / kernel 访存优化

- 日期：2026-11-24
- 审查对象：`docs/superpowers/specs/2026-11-24-gpu-dimension-design.md`（§5 Part 1–4 大纲 L178–281、§4.3 符号表 L118–136、§6.3 强制重推导清单 1.2/2.4/3.4/4.3 条 L471–474、§2 边界 L31–54、§4.1 证据分级、§4.2 recap 纪律、附录预核实 + builder-check 修复传播）
- 审查人：fresh-context adversarial evaluator（路线 A，不继承写作上下文；未写本 spec；未派发子代理）
- 标准：STYLE_GUIDE.md §6 / §10（S1 = 事实致命 / 错误公式 / 不存在 repo；S2 = 实质：约定冲突 / 推导断裂 / 局部事实错误；S3 = minor；verification log 强制）+ R2 style sheet
- recap 对照源（全文读 §6）：`models/02-transformer-anatomy/04-causality-and-kv-cache.md`（式 (9) 容量 / 式 (10)-(12) 强度 / roofline 入口 prose）；定点核实 training 4.1 §2.2（d/α/β/式 (2)(3)）、inference 1.2（R = ridge）、inference 1.3（datasheet 表 + SKU 纪律）、inference 1.4 §6（batch invariance）
- 站点代码核实：`site/scripts/build-content.mjs` L41–90（DIMENSIONS/asserts）、L110–116（Part 目录 try/catch）、L210（references 硬读）、L216–221（断言抛错）
- 任务：事实核查（一手来源）+ 独立数学重推导（1.2/2.4/3.4/4.3）+ 边界纪律 + 符号表一致性 + builder-check 修复传播

## 0. 核查通道与约束（影响覆盖率，先声明）

| 通道 | 状态 | 用途 |
| --- | --- | --- |
| WebFetch（全家族域） | 本机被挡（"Unable to verify if domain is safe"，docs.nvidia.com 两次失败） | 改 curl 一手抓取 + 本地 grep/python 提取（spec risk 8 fallback 通道） |
| WebSearch | 可用 | Hopper whitepaper 镜像定位、B200 SM 数 / SKU 多方互证、JEDEC HBM3 几何 |
| docs.nvidia.com（curl） | 可用 | Programming Guide（新路径 cuda-programming-guide/）、PTX ISA 9.5 全文 3.97MB、Best Practices 13.5 全文 3.97MB、Blackwell Tuning Guide v13.5、Nsight Compute Profiling Guide、compute-capabilities / device-callable-apis / l2-cache-control 子页 |
| developer.nvidia.com（curl） | 可用 | Hopper Architecture In-Depth blog 全文（425KB，逐字引句） |
| nvidia.com 产品页（curl） | 可用 | H100 / H200 / HGX 规格表 |
| Hopper whitepaper PDF 镜像 | hpctech / modal-cdn 两镜像 curl 超时（120s/90s），modal-cdn 取得 1.49MB 截断件（无 %%EOF） | SM scheduler 发射宽度改以 Best Practices Table 5（FP32 128 results/cycle/SM）+ spec 1.4 自述互证；whitepaper 正文未读，相关断言降级标注 |
| arxiv.org/html（curl） | 可用 | 2501.12084v1（Dissecting Hopper）延迟表互证（学术 [B]，非官方规格，仅 corroborate ⟨verify⟩ 延迟项） |
| GitHub API / repo | 本路未用（P1–P4 锚点 repo 路径核实留 W1/W2；B/C 路已覆盖 CUTLASS/NCCL/DeepEP/FlashMLA） | — |

抓取件存 `/tmp/gpu-review-a/`（审查输入，非仓库交付物）；重算脚本 `/tmp/gpu-review-a-scripts/rederive.py`（inline Python，全部数值见 §2）。

**总判定：S1 = 1，S2 = 6，S3 = 7。**

- **2.4 roofline recap 纪律：PASS**——§2 / §4.2 / §6.3 / 2.4 大纲四处均声明 I/ridge recap models 2.4 不作新推导；增量（层级带宽表 + HBM 微架构来源 + 测量 + latency-bound 精确化 + 实测 roofline）界定正确，与 models 2.4 §6 末句「微架构来源与测量方法归 gpu 维度」seam 逐字吻合（§3.1）。
- **S1 唯一项**：1.2 occupancy 物理上界公式三项单位不一致（warps/threads/blocks），register 约束对所有合法 R_th ≤ 255 恒不绑定——「三约束」实为死约束 + 单位错（§2.1；D 路 D-S2-1 同址判 S2，A 路按错误公式升 S1，去重合并由 controller 裁定）。
- S2 集中在：2.1/§4.3 HBM 带宽公式 bus width 口径（代入章内自定义 1024bit/stack 即 16× 高估）、4.1 coalescing 两条定量断言（ceil(span/32B) 通用式 + 1/s 衰减律）与官方规则不符、1.2「warp scheduler 双发射」与 Hopper 单发射事实及 1.4 自述矛盾、§4.3 GEMM 问题维 M/N/K/T 撞硬规则（与 D-S2-3/B-3 重叠）、§2/§4.2 models 2.4 式号/节号误引（容量式实为 §5 式 (9)）。
- 全部 ⟨verify⟩ 延迟项按任务书**不计 findings**；本路顺带预核实的列 §5 供 W1/W2 research pass 引用。两个 verify-tagged **口径标注错误**（228KB「combined」、B200 SKU）沿 C-8 先例（「方向约定是已断言错误，datasheet 核数不会自动纠正」）以 S3 入表，数值本身仍留 verify。

## 1. Findings 表

| # | severity | 位置（spec 节/行） | 断言 | 证据（来源 URL / 原文引句 / 重算） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| A-01 | **S1** | §5 1.2 关键推导 L188 + 内容行 L187；§6.3 清单 1.2 条 L471 | occupancy 物理上界 $\mathcal{O}_{\max}=\min(\text{warp slots}, \lfloor R_{\mathrm{file}}/R_{\mathrm{th}}\rfloor, \lfloor S_{\mathrm{total}}/S_{\mathrm{sh}}\rfloor)$ 三项单位不一致：warp slots = warps、$R_{\mathrm{file}}/R_{\mathrm{th}}$ = **threads**（registers ÷ registers/thread）、$S_{\mathrm{total}}/S_{\mathrm{sh}}$ = **blocks**（bytes ÷ bytes/block），不能直接 min 得「理论 active warps」。且 register 项恒不绑定：$R_{\mathrm{th}}\le 255$ 时 $65536/R_{\mathrm{th}}\ge 257 > 64$，「三约束」实为两约束 + 死约束；缺 max blocks/SM（CC 9.0 = 32）第四约束。与 3.4 大纲正确流程（block size → active blocks/SM → active warps）内部矛盾 | §4.3 L130 定义 $R_{\mathrm{th}}$ = registers/thread、$S_{\mathrm{sh}}$ = shared bytes/block；CC 9.0 表（compute-capabilities.html）64 warps / 32 blocks / 64K regs / 228KB shared。独立重算（§2.1）：R_th=64 bs=256 → spec 64「warps」(O=1.0) vs 正确 32 warps (O=0.5)，2× 错；S_sh=48KB bs=256 → spec 4 vs 正确 32，8× 错；**R_th=255 bs=1024 → spec 64 (O=1.0) vs 正确 0（kernel 无法 launch）**。STYLE_GUIDE §10「错误公式」= S1 | 改 warp 单位一致形：$\text{warps}_{\max}=\min(64,\ 32\cdot w_b,\ \lfloor R_{\mathrm{file}}/(32 R_{\mathrm{th}})\rfloor_{\text{gran}},\ \lfloor S_{\mathrm{total}}/S_{\mathrm{sh}}\rfloor\cdot w_b)$，$w_b$ = warps/block，注明 register/shared 按 block 粒度取整（CC 9.0 register 分配粒度 warp 级、shared 128B + 每 block 1KB reserved）；$\mathcal{O}_{\max}=\text{warps}_{\max}/64$。与 3.4 流程对齐。**跨路注：D-S2-1 同址判 S2（理由：草式 + 章级强制重推导兜底）；A 路按「register 约束全输入域死 + 不可 launch kernel 给 100% occupancy」升 S1，controller 去重裁定** |
| A-02 | **S2** | §5 2.1 关键推导 L207–208 + §4.3 HBM 几何行 L131 | $\mathrm{BW}=n_{\mathrm{stack}}\cdot n_{\mathrm{ch}}\cdot f_{\mathrm{pin}}\cdot\text{width}/8$ 中 width 口径未定义；2.1 内容行唯一给出的宽度是「bus width 1024bit/**stack**」。writer 以章内自定义量 1024 代入含 $n_{\mathrm{ch}}$ 因子的公式 → 通道数双计，**16× 高估** | JEDEC HBM3（JESD238）/ Synopsys：1024bit/stack = 16 ch × 64bit（32 pseudo-ch × 32bit）；Hopper blog「5 HBM3 stacks, 10 512-bit memory controllers」（5×1024=5120=10×512 ✓）。重算（§2.2）：H100 5×16×5.234G×**1024**/8 = 53.6 TB/s ≠ 3.35；正确 5×16×5.234G×**64**/8 = 3.35 TB/s 或 5×1024×5.234G/8 = 3.35 TB/s | §4.3 公式行注「width = **每通道**位宽（HBM3 = 64bit）；1024bit/stack = $n_{\mathrm{ch}}\times$ width」；或改 $\mathrm{BW}=n_{\mathrm{stack}}\times 1024\times f_{\mathrm{pin}}/8$（每栈口径，免 $n_{\mathrm{ch}}$）。2.1「5 stack × 16 ch × …」逐项 ⟨verify⟩ 保留。**跨路注：D-S3-5 同址判 S3；A 路按「章内自定义量代入即 16× 错」升 S2，修复合并** |
| A-03 | **S2** | §5 4.1 关键推导 L260 | 「coalescing transaction 数 = $\lceil\text{warp 访问跨度}/32\text{B}\rceil$」作为通用式不成立。官方规则 = 服务 warp 全部线程所需的 32B transaction 数（= 触及的**不同 sector 数**），非跨度取整 | Best Practices 13.5 §10.2.1 逐字：「the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to **the number of 32-byte transactions necessary to service all of the threads of the warp**」。反例（§2.5，fp32 对齐）：stride 16 元素 → ceil(span/32)=63 vs 实际 32 sectors；stride 32 → 125 vs 32；两簇 scatter（0..60 ∪ 1024..1084）→ 34 vs 4。连续访问（核心 coalescing 场景）与 stride ≤ 8 元素时公式恰成立 | 通用式改 $\lvert\lbrace\lfloor\text{addr}_i/32\rfloor\rvert\rvert$（不同 sector 计数）；注明 ceil(span/32B) 仅适用于连续/小 stride 稠密访问。与 2.2 sector 粒度账回链 |
| A-04 | **S2** | §5 4.1 内容行 L259 | 「stride $s$ → 有效带宽降 $1/s$ 上界」对 $s>8$（fp32）不成立：利用率降至 1/8 后**饱和**（每线程独占一个 32B sector），不再随 $s$ 降 | Best Practices §10.2.1.4 逐字：「A stride of 2 results in a 50% of load/store efficiency … **As the stride increases, the effective bandwidth decreases until the point where 32 32-byte segments are loaded for the 32 threads in a warp**」。重算（§2.5）：fp32 利用率 s=1→1, 2→1/2, 4→1/4, 8→1/8, 16→**1/8**, 32→**1/8**；闭式 = $1/\min(s,8)$（= elem_bytes / min(stride_bytes, 32B)），非 1/s | 改「stride $s$（元素计）利用率 = $1/\min(s, 32/\text{elem\_bytes})$，fp32 即 $1/\min(s,8)$：$s\le 8$ 按 1/s 降，$s\ge 8$ 饱和于 1/8（sector 粒度地板）」；与 2.2「sector 粒度对有效带宽的影响」回链 |
| A-05 | **S2** | §5 1.2 内容行 L187 | 「warp scheduler **双发射**与指令派发」——Hopper 每 scheduler 每周期发射 **1** 条 warp 指令（单发射 ×4 scheduler），双发射是 Kepler 时代术语；与 1.4 自述「throughput 1/cycle/SM partition ⟨verify⟩」内部矛盾 | Best Practices Table 5（Throughput of Native Arithmetic Instructions, results/cycle/SM）：CC 9.0 FP32 add/mul/fma = **128** = 4 warp-inst × 32 lane = 每 scheduler 1 warp-inst/cycle（若双发射应 256）；CC 9.0 INT32 = 64（半率）。Hopper blog SM 框图每 processing block = 1 Warp Scheduler + 1 Dispatch Unit。spec 1.4 L199 自述 1/cycle/partition | 改「4 个 warp scheduler，各每周期单发射 1 条 warp 指令到对应 dispatch unit/执行管线（FP32 128 results/cycle/SM）」；删「双发射」或注明 Kepler 历史对照（§2 禁史规则下建议直接删） |
| A-06 | **S2** | §4.3 消歧硬规则 L136 / 表 L124/126/129 vs §5 5.2 关键推导 L292 | GEMM 问题维 $M/N/K$ 未入符号表且撞三条硬规则：5.2「HBM traffic = $O(MNK/T)$」用裸 $M$（GEMM 行维，撞「$M$ = 消息总字节」）、裸 $N$（GEMM 列维，撞「$N$ 专归参数量 / gpu 不占用」）、裸 $T$（tile 尺寸，撞「$T$ = 序列长；tile 用 $T_M/T_N/T_K$」）。P4–P6 kernel 章与 8.x/10.x 通信章跨波同字母异义 | §4.3 L124/126/129/136 逐字；§5 5.2 L292 $O(MNK/T)$；4.3 大纲用 $T_M/T_N$（合规）vs 5.2 裸 $T$（违规）。builder-check 附录 L590「符号表内部一致」只查表内、未查表 vs 大纲 | §4.3 加章域句 + GEMM 问题维行：「P4–P6 kernel 章 $M,N,K$ = GEMM 矩阵维（章首声明，非消息字节/参数量）；tile 尺寸一律 $T_M/T_N/T_K$ 或 $T_{\mathrm{tile}}$，禁裸 $T$；通信章 8.x/10.x 按表内通信语义」。**跨路注：与 D-S2-3 / B-3 重叠（独立确认）；注意力几何 $d_{\mathrm{head}}/d_{\mathrm{model}}/d_{\mathrm{latent}}$ 缺行 + IO 定理 $M$ 翻译见 D-S2-4，本路不重复** |
| A-07 | **S2** | §2 L36；§4.2 recap 行 L109 | models 2.4 式号/节号误引：(1) §2「KV cache **容量公式**与 prefill/decode 算术强度 / roofline 入口（models 2.4 **§6 式 (10)-(12)**）」——容量公式实为 **§5 式 (9)** $\mathrm{Mem}_{\mathrm{KV}}=2LH_{kv}d_{\mathrm{head}}TBb$；§6 式 (10)-(12) 是 prefill/decode **强度**式。(2) §4.2 把「$I$ = FLOPs/bytes、吞吐上限 $\min$、ridge = 峰值÷带宽」归于「式 (10)-(12)」——三者是 §6 **prose**（无编号），式 (10)-(12) 不含它们；且 recap 清单漏 prefill $I\approx T_p$（式 (10)） | models 2.4 一手核实：L188 式 (9)（§5 cache size）、L221/229/239 式 (10)/(11)/(12)（§6 强度）、L216 roofline min/ridge prose（无编号）、L244「本章交付的是式 (10)–(12)」。先例：inference D-S2-1 式号误引（式 (5)→(9)）判 S2（D 路附录引） | §2 改「KV cache 容量公式（models 2.4 §5 式 (9)）与 prefill/decode 算术强度 / roofline 入口（§6 式 (10)-(12) + 入口 prose）」；§4.2 recap 行改「models 2.4 §6（式 (10)-(12) 强度 + roofline 入口 prose：I/min/ridge）」，recap 清单补 prefill $I\approx T_p$ 或注明只 recap decode 侧 |
| A-08 | S3 | §5 2.1 L207（B200 8 TB/s / 192GB ⟨verify⟩）+ 1.1 L181；§4.2 L114（2.4 引 inference 1.3 同表） | B200 SKU / raw-vs-usable 口径规则缺失：大纲「B200 8 TB/s / 192GB」与 2.4 必须引用的 inference 1.3 已核实表（**HGX B200 = 180GB / 7.7 TB/s**；GB200 = 186GB / 8 TB/s）冲突；192GB = 8×24GB **raw** 栈容量，非任何出货 SKU usable。数值本身 ⟨verify⟩ 延迟，但缺 SKU 口径规则会使 2.1 与 2.4「引用同表」自相矛盾 | inference 1.3 L31 逐字「B200 指 HGX B200（180 GB / 7.7 TB/s），与 GB200（186 GB / 8 TB/s）是不同 SKU」（2026-11-23 PDF 一手核实）；HGX 产品页 Total Memory 1.4 TB（=8×180）；Lenovo lp2226 / primeline B200 datasheet 镜像（186/8 vs 180/7.7 两列）；WebSearch 多方互证。重算：8 TB/s ⇔ 8 stack×1024bit×7.8125Gbps/8；7.7 ⇔ ~7.5Gbps | §4.3 显存行或 2.1 加 SKU + raw/usable 口径注：「B200 分数值须标 SKU（HGX B200 180GB/7.7 vs GB200 186GB/8）与口径（datasheet usable vs whitepaper raw 192GB）；2.4 对账以 inference 1.3 同表（HGX B200）为准」。数值留 W1 ⟨verify⟩ |
| A-09 | S3 | §5 1.2 内容行 L187 | 「可配比例 H100 **228KB combined** ⟨verify⟩」口径混淆：228KB 是 **shared carveout 上限**（per SM），L1+shared **combined = 256KB**。数值 228 本身正确，「combined」标注是已断言错误（datasheet 核 228 不会自动纠正 combined 字样，沿 C-8 先例入表） | Hopper blog 逐字「**256 KB of combined shared memory and L1 data cache**, 1.33x larger than A100」+ 表「Shared Memory Size / SM **Configurable up to 228 KB**」；CC 9.0 表 max shared/SM = 228KB、max shared/block = 227KB；Blackwell Tuning Guide「combined L1 cache, texture cache, and shared memory of **256 KB**」（B200 同） | 改「shared/L1 统一物理 combined **256KB**，shared 可配至 **228KB**（每 block opt-in 上限 227KB，1KB reserved）」。⟨verify⟩ 保留 |
| A-10 | S3 | §5 3.2 关键推导 L240 | 「命题-证明骨架：__threadfence 保证 **grid 内**可见」scope 不紧：官方语义 = **device** 级（membar.gl），grid 是其子集（命题真但不紧）；且 fence 单独不建立跨线程 happens-before，须与同步操作（atomic flag / barrier）配对才构成可见性保证 | PTX ISA 9.5 membar 逐字「The **level qualifier specifies the set of threads that may observe the ordering effect**」；.gl = GPU/device 级、.cta = block、.sys = system（__threadfence_block/system 对应）。CUDA Programming Guide memory fence functions 同口径（device scope） | 命题改「__threadfence() 保证调用线程 prior writes 对 **device 内所有线程**可见（membar.gl；grid ⊂ device）；与 release/acquire 或 atomic flag 配对才构成跨线程 happens-before」；证明骨架注明配对条件 |
| A-11 | S3 | §5 4.3 内容行/关键推导 L271–272 | tiling 强度 $I=O(T_MT_N/(T_M+T_N))$ 公式本身正确（§2.3 验证），但数值实例须声明「**per-threadblock、无跨 block L2 复用**」假设：T=128 bf16 → I=64 < H100 BF16 ridge 295，per-block 账 memory-bound；真实 GEMM 经 L2 跨 block 复用进入 compute-bound（4096³ 例：I 64 → 2048）。不声明假设，实例与「GEMM compute-bound」直觉矛盾 | 重算（§2.3）：4096³ bf16 T=128，无 L2 traffic 2.15GB → I=64；A 行 tile/B 列 tile 各读一遍（L2 捕获）traffic 67MB → I=2048。H100 L2 50MB（spec 2.2）。跨路注：B-6 已为 5.2 $O(MNK/T)$ 补同族假设注（S3），本条为 4.3 侧 | 4.3 大纲补「强度式 = per-threadblock shared-tile 账（假设每 tile 从 HBM 读、无跨 block L2 复用）；L2 复用与 rasterization 修正入 5.2 走读对账」；数值实例同注 |
| A-12 | S3 | §5 1.4 关键推导 L200 + 3.4 L252 | Little's law 数值代入须锁定 throughput = **瓶颈资源服务率**（memory pipeline sectors/cycle），非 warp 发射率；否则所需并发高估 ~10×。$n_{\mathrm{warps}}\cdot\text{ILP}\ge\text{latency}\times\text{throughput}$ 结构正确，但「latency × throughput」用发射率（4/cycle/SM）× 600 = 2400 warp-inst ≫ 64 warp slots，会推出「occupancy 不可能藏住 HBM 延迟」的错误结论 | 重算（§2.4）：H100 per-SM HBM 份额 3.35e12/132/1.83e9 = 13.9 B/cyc = 0.433 sectors/cyc；L = 0.433×600 ≈ 260 sectors = 8.1 KB in-flight/SM；64 warps × 1 outstanding 128B load（4 sectors）= 256 sectors ≈ 刚好。正确归一化下 64 warps 量级自洽 | 1.4/3.4 章首口径注：「Little's law throughput 取瓶颈资源服务率（HBM 延迟隐藏 = per-SM 带宽份额 ÷ 32B sector），非发射率；per-SM 归一化」；数值实例用 §2.4 账 |
| A-13 | S3 | §5 2.3 关键推导 L220 | 「register spill **带宽代价**（spill bytes × HBM **延迟**）」量纲错：bytes × cycles 不是带宽代价（时间）。带宽代价 = spill bytes ÷ BW（时间）；延迟暴露另计（spill 访问数 × latency ÷ MLP） | 量纲分析：[bytes]×[cycles] ≠ [time]；带宽代价 $t_{\mathrm{spill}}=\text{spill\_bytes}/\mathrm{BW}_{\mathrm{HBM}}$（或 ×β）。local memory HBM backing 正确（2.3 内容行） | 改「spill 带宽代价 = spill bytes / BW_HBM（时间）；延迟暴露 = spill 访问数 × latency / MLP（1.4 回链）」 |
| A-14 | S3 | 附录 builder-check 关闭状态 L609 vs L596 | 「本附录 **6 项** self-decidable 修复 flag 给四路复审防传播」与正文「**9 项**自决修复」计数不一致 | 附录 L596 列 9 项（1 MLA / 2 refsMin / 3 P5 波次 / 4 ring-tree 归属 / 5 C 路扩 P11 / 6 scope 声明 / 7 GPU 验证标准 / 8 P9 波次 / 9 ring recap 拆分）；L609 写 6 项 | L609 改「9 项」或显式列哪 6 项 flag 四路（本路核实 9 项传播状态见 §4） |

## 2. 独立数学重推导（§6.3 强制清单 1.2 / 2.4 / 3.4 / 4.3 + 2.1 / 4.1）

脚本 `/tmp/gpu-review-a-scripts/rederive.py`，H100 参数取 CC 9.0 表 + 官方 datasheet（64 warps / 32 blocks / 65536 regs / 228KB shared / 132 SM / 3.35 TB/s / BF16 dense 989.5 TFLOPS）。

### 2.1 occupancy 物理上界（1.2）——FAIL（A-01）

正确推导（warp 单位一致）：设 $w_b=\lceil\text{blockDim}/32\rceil$，resident blocks $b$ 受四约束：$b\le 32$（block slots）、$b\cdot w_b\le 64$（warp slots）、$b\cdot w_b\cdot 32 R_{\mathrm{th}}\le 65536$（register，粒度取整略）、$b\cdot S_{\mathrm{sh}}\le S_{\mathrm{total}}$（shared，128B 粒度 + 1KB/block reserved 略）。active warps = $b\cdot w_b$，$\mathcal{O}=b w_b/64$。

| 例（R_th, S_sh, blockDim） | spec 式结果 | 正确 warps | 误差 |
| --- | --- | --- | --- |
| (64, 0, 256) | 64（O=1.0） | 32（O=0.5） | 2× |
| (128, 0, 256) | 64（O=1.0） | 16（O=0.25） | 4× |
| (32, 48KB, 256) | 4（O=0.062） | 32（O=0.5） | 8× |
| (40, 96KB, 512) | 2（O=0.031） | 32（O=0.5） | 16× |
| (255, 0, 1024) | 64（O=1.0） | **0（不可 launch）** | 致命 |

register 项 $65536/R_{\mathrm{th}}\ge 257>64$ 对所有 $R_{\mathrm{th}}\le 255$ 恒不绑定 → 死约束。shared 项给 blocks 非 warps。**判定：spec 草式错误公式（S1）。**

### 2.2 HBM 带宽栈几何（2.1）——公式口径 FAIL（A-02），数值 PASS

- H100：5 stack × 1024 bit/stack × 5.234 Gbps/pin ÷ 8 = **3.35 TB/s** ✓（每栈口径）；等价 5 × 16 ch × 64 bit × 5.234G ÷ 8 ✓（每通道口径）。3.35 TB/s 反推 pin rate = 5.234 Gbps。
- H200：6 stack × 1024 × 6.25G ÷ 8 = **4.8 TB/s** ✓（141GB = 6×24GB raw − spare）。
- **错误代入**：5 × 16 × 5.234G × **1024** ÷ 8 = 53.6 TB/s（16×）——章内自定义「1024bit/stack」代入含 $n_{\mathrm{ch}}$ 公式即此错。
- B200：8 stack × 1024 × 7.8125G ÷ 8 = 8 TB/s（GB200/DGX bin）；HGX B200 7.7 TB/s ⇔ ~7.5 Gbps（A-08 SKU）。

### 2.3 GEMM tiling 强度 O(1)→O(T)（4.3）——公式 PASS，假设注缺（A-11）

- naive：每输出元素 2K FLOPs、读 2K 元素 → $I=1/\mathrm{bytes}=0.5$ FLOP/B（bf16）= O(1) ✓。
- shared tile $T_M\times T_N$，K  loop 步长 $T_K$：每步 FLOPs $2T_MT_NT_K$，HBM bytes $(T_M+T_N)T_K\cdot\mathrm{bytes}$ → $I=2T_MT_N/((T_M+T_N)\mathrm{bytes})$ ✓；方阵 $T$：$I=T/\mathrm{bytes}$ = O(T) ✓。T=32/64/128/256 → I=16/32/64/128 FLOP/B。
- 与 5.2 traffic 式一致：traffic = $MNK(1/T_M+1/T_N)\cdot\mathrm{bytes}$（B-6 已验证）。
- **假设陷阱**：T=128 bf16 I=64 < H100 ridge 295（per-block memory-bound）；4096³ 例无 L2 traffic 2.15GB（I=64）vs L2 复用 67MB（I=2048，compute-bound）。须声明 per-block / 无 L2 复用假设（A-11）。

### 2.4 Little's law latency hiding（3.4 / 1.4）——结构 PASS，归一化注缺（A-12）

$L=\lambda W$。H100 per-SM HBM 份额 = 3.35e12/132/1.83e9 = 13.9 B/cyc = 0.433 sectors/cyc；W≈600 cyc → L≈260 sectors = 8.1 KB in-flight/SM。64 warps × 1 outstanding 128B load（4 sectors）= 256 sectors ≈ 自洽。**错误归一化**：W × 发射率 = 600×4 = 2400 warp-inst ≫ 64 slots → 须用 memory 服务率非发射率。

### 2.5 coalescing 定量（4.1）——FAIL（A-03 / A-04）

fp32、32 threads、对齐，sector 计数 = 不同 $\lfloor\text{addr}/32\rfloor$ 数：

| stride（元素） | ceil(span/32) | 实际 sectors | 利用率 | 1/min(s,8) |
| --- | --- | --- | --- | --- |
| 1 | 4 | 4 | 1.0 | 1.0 |
| 2 | 8 | 8 | 0.5 | 0.5 |
| 4 | 16 | 16 | 0.25 | 0.25 |
| 8 | 32 | 32 | 0.125 | 0.125 |
| 16 | **63** | **32** | **0.125** | 0.125 |
| 32 | **125** | **32** | **0.125** | 0.125 |

两簇 scatter：ceil(span/32)=34 vs 实际 4。官方 BP 逐字「until … 32 32-byte segments … for the 32 threads」证饱和。**ceil(span/32) 仅连续/stride≤8 成立；1/s 仅 s≤8 成立，s≥8 地板 1/8。**

### 2.6 ridge  cross-check（2.4 recap 数值）——PASS

H100 BF16 989.5/3.35=295.4、H200 989.5/4.8=206.1、B200 HGX BF16 2250/7.7=292.2、FP4 9000/7.7=1168.8 —— 与 inference 1.3 表逐字一致 ✓。MMA FLOPs/指令：mma.sync m16n8k16 = 4096、wgmma m64n256k16 = 524288（= m·n·k·2 ✓）。

## 3. 边界与 recap 纪律核查

### 3.1 2.4 roofline recap scope（任务 2/4 最高优先）——PASS

- models 2.4 §6 已交付：I = FLOPs/bytes（prose）、吞吐上限 min(峰值, I×带宽)（prose）、ridge = 峰值÷带宽（prose）、式 (10) prefill I≈T_p、式 (11) decode I(t)、式 (12) decode I(t,B) + I≈B + 饱和上界（4.9 算例）。末句 seam：「roofline 的逐卡 datasheet 参数表归 inference 1.3，微架构来源与测量方法归 gpu 维度」。
- gpu spec 四处声明 recap 不重推：§2 L36（out-of-scope）、§4.2 L109（recap 入口 + 增量 = 层级带宽账 + HBM 微架构来源 + 测量）、§6.3 L472（「I/ridge recap models 2.4，不作新推导」）、2.4 大纲 L225（recap + 指引 → 增量）。**增量界定正确**：层级带宽-延迟-容量全表 / HBM 栈几何来源（2.1 recap）/ bound 三区精确化（latency-bound 低 occupancy，models 未交付，合法新增）/ nvbandwidth + Nsight 测量 / 实测 roofline 绘制。
- 2.4 关键推导「roofline 拐点 R 代入 H100/B200（recap models 式）」= 数值实例化，非重推导 ✓。「latency-bound 判定条件」= 新增 ✓。
- inference 1.3 datasheet 表引用 seam ✓（§4.2 L114 + 2.4 L225，inference W1 errata 分工一致）。
- 唯一瑕疵 = 式号/节号误引（A-07，S2），不影响 recap scope 正确性。

### 3.2 serving / cluster 越界扫描（P1–P4 全 16 章）——PASS

- 无 serving 调度 / 引擎编排内容：3.3 inference 3.4 图捕获 = recap + 硬件机制增量 ✓；3.2 inference 1.4 batch-invariance = recap 回链 ✓（inference 1.4 §6 一手核实存在）；2.4 inference 1.3 = datasheet 引用 ✓。
- 无 K8S / 容器 / 编排内容 ✓。MIG/vGPU 硬件分区在 11.1（P11，C 路），P1–P4 不涉及。
- 两层走读纪律：4.4「CUTLASS mainloop pipeline 走读**片段**（5.2 全量回链）」✓ 显式片段级；1.3/4.2 CUTLASS 代码 = 机制例证锚 ✓；P1–P4 余章默认机制片段级（§4.2 L115 默认规则覆盖）。

### 3.3 符号表 §4.3 一致性——部分 FAIL

- n=rank / N=params / B=batch / bytes 不占字母 / R=ridge（沿 inference 1.2 ✓ 一手核实）/ α-β 通信语义 / T_M/T_N/T_K tile ✓ 表内自洽。
- recap 翻译规则 ✓：training d→n（training 4.1 §2.2 d=DP 度一手核实）、models b→bytes（models 2.4 式 (11) b 一手核实）、models d(hidden)→d_model ✓。
- **缺口**：GEMM 问题维 M/N/K/T 撞硬规则（A-06，S2，与 D-S2-3/B-3 重叠）；注意力几何 d_head/d_model/d_latent 缺行 + IO 定理 M 翻译（D-S2-4 已覆盖，本路不重复）；5.1 n=元素数双重赋值（D-S2-2/B-3 已覆盖）；R_file/S_total 未入表（D-S3-8 已覆盖）。

## 4. builder-check 修复传播核查（9 项 + 站点断言）

| fix | 内容 | 传播状态 |
| --- | --- | --- |
| 1 | 6.4 MLA 几何 V4.1 512=448+64 / V3 576 区分 + 坑注 | ✓ L329–330 落地（B 路最高优先核查 PASS 互证）；§4.1 L99「V4.1 MLA 512 元素」一致 |
| 2 | §8 refsMin ≥50 种入（空骨架不可过） | ✓ L523 与代码 L220–221（refCount<refsMin 抛错）一致；L210 references.md 硬读无 try/catch ✓「须先于条目存在」正确；L110–116 Part 目录 try/catch 跳过 ✓ W0 {0,0} 可过 |
| 3 | Part 5 标题 W2→W3 | ✓ L282 W3；§7 表 W3=P5+P6 ✓ |
| 4 | §6.3 ring/tree 归属 8.2/8.3 拆分 | ✓ L478（8.2 下界 / 8.3 临界 M*，W4 覆盖 8.2+8.3）；§4.2 L111 同步 ✓ |
| 5 | §10 C 路扩 P7–P11 | ✓ L550 |
| 6 | §4.2 每章大纲声明代码 scope | ✓ L115 规则 + 默认机制片段级；4.4 显式片段 ✓；6.2/6.4 显式标记缺（B-8/D-S3-6 已 flag，P6 非本路） |
| 7 | §6.2 GPU 验证标准（教学片段 + Python 重算 + 实测语义） | ✓ L461 |
| 8 | P9 标题 W4→W5 | ✓ L380 W5 |
| 9 | §4.2 ring recap 增量 8.2/8.3/8.4 拆分 | ✓ L111 |

站点断言滚动核实：W1={8,2}（P1 4+P2 4）/ W2={16,4}（+P3 4+P4 4）与 §7 波次表、累计章/Part 数精确吻合 ✓；依赖图 14 边逐边计数 ✓（P1→P2/P2→P3/P3→P4/P3→P5/P1→P5/P4→P6/P5→P6/P2→P7/P7→P8/P7→P9/P8→P10/P9→P10/P10→P11/P6→P11）；8 重章（1.2/2.4/4.3/5.1/5.2/6.3/8.2/9.2）全在 §6.3 清单 ✓。附录「6 项」计数误（A-14）。

## 5. ⟨verify⟩ 延迟项预核实 heads-up（不计 findings，供 W1/W2 research pass）

| 项 | spec 位置 | 预核实结论 | 一手来源 |
| --- | --- | --- | --- |
| H100 132 SM / B200 148 SM | 1.1 | **均正确**。H100 SXM 132（144 full die）；B200 148 active（160 physical，74/die×2） | Hopper blog「132 SMs per GPU」；WebSearch 互证 glennklockwood / chipsandcheese / TechPowerUp 18944 cores=148×128 |
| register file 64K×32bit/SM | 1.2/2.3 | 正确（CC 9.0/10.0 同） | CC 表「Number of 32-bit registers per SM 64 K」；Blackwell Tuning Guide 同 |
| warp slots 64/SM | 1.2 | 正确（CC 9.0/10.0 = 64；12.0 = 48） | CC 表 + Blackwell Tuning Guide |
| H100 228KB combined | 1.2 | **口径错**（228=shared max，combined=256KB）→ A-09 | Hopper blog + CC 表 + Blackwell Tuning Guide |
| HBM 1024bit/stack、5 stack×16 ch | 2.1 | 正确（HBM3 = 16×64bit） | JEDEC JESD238 / Synopsys / Hopper blog 10×512-bit controllers |
| H100 3.35 TB/s / 80GB、H200 4.8 / 141GB | 2.1 | 正确 | H100 / H200 产品页逐字 |
| B200 8 TB/s / 192GB | 2.1 | **SKU 待定**（HGX 180/7.7；GB200 186/8；192=raw）→ A-08 | inference 1.3 已核实表 + HGX 页 + datasheet 镜像 |
| L2 hit ~200 cycle | 2.2 | H100 偏低：near hit ~258 / far ~414（A100 ~203）；建议 ~250-270 near + 分区注 | arXiv 2501.12084 Table III/IV（H800） |
| FMA ~4 cyc / shared ~20-30 / L1 ~30 / HBM ~400-800 | 1.4/2.2 | 量级正确（H800 实测 L1 32 / shared 29 / global 656；FMA 4 沿 Volta 微基准） | arXiv 2501.12084 + Jia et al. 微基准传统 |
| mma.sync m16n8k16 / wgmma m64nNk16 / tcgen05 | 1.3 | 存在 ✓；**几何随 dtype 变**：wgmma k16=f16/bf16、k8=tf32、k32=fp8/int8、k256=b1；mma.sync 同理；tcgen05 cta_group::1/::2 + block_scale ✓ | PTX ISA 9.5 全文 grep |
| CUDA Programming Guide URL 404 | 附录 | **404 属实**；现行路径 = `cuda-programming-guide/`（200）；旧路径加 `index.html` 亦 200 | curl 三 URL 实测 |
| H100 50MB L2 两分区 | 2.2 | 正确 | Hopper blog + arXiv 2501.12084「50MB partitioned L2」 |
| MIG 最多 7 实例 | 11.1（C 路） | 正确（H100 页「Up to 7 MIGS @ 10GB each」） | H100 产品页 |

## 6. Verification log（强制）

| # | 抽查项 | 来源 URL / 原文引句 | 结论 | 覆盖 |
| --- | --- | --- | --- | --- |
| V1 | Programming Guide URL 404 + 现行路径 | curl：`cuda-c-programming-guide/`→404；`cuda-programming-guide/`→200；`cuda-c-programming-guide/index.html`→200 | 附录 404 记录正确；现行路径已定位 | ✓ |
| V2 | CC 9.0 per-SM 资源 | docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html：「Maximum number of resident warps per SM … 64」「resident blocks … 32」「Number of 32-bit registers per SM 64 K」「registers per thread 255」「Maximum amount of shared memory per SM … 228 KB」「per thread block … 227 KB」「Warp size 32」「KB … 1024 bytes」 | 64 warps / 32 blocks / 64K regs / 255 / 228KB shared / 227KB block ✓；228 是 shared-max 非 combined | ✓ |
| V3 | Hopper SM / cache / HBM 几何 | developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/：「8 GPCs, 66 TPCs, 2 SMs/TPC, **132 SMs per GPU** … 80 GB HBM3, **5 HBM3 stacks, 10 512-bit memory controllers** 50 MB L2」「**256 KB of combined shared memory and L1 data cache**, 1.33x larger than A100」「Shared Memory Size / SM Configurable up to **228 KB**」「Register File Size / SM 256 KB」 | 132 SM / 256KB combined / 228KB shared-max / 50MB L2 / 5 stack ✓ | ✓ |
| V4 | NVLink4 方向带宽 | Hopper blog：「a single link delivering **25 GB/sec effective bandwidth in each direction** … H100 includes **18** fourth-generation NVLink links to provide **900 GB/sec total bandwidth**」 | 25 GB/s 单向、900 双向（C-8 同证据独立确认；P7 非本路 findings） | ✓ |
| V5 | H100 datasheet 带宽/容量 | nvidia.com/en-us/data-center/h100/：「GPU Memory **80GB** … GPU Memory Bandwidth **3.35TB/s**」「BFLOAT16 Tensor Core 1,979（sparse 脚注）」「Up to 7 MIGS @ 10GB each」 | 3.35/80 ✓；dense 989.5 ✓；MIG 7 ✓ | ✓ |
| V6 | H200 带宽/容量 | nvidia.com/en-us/data-center/h200/：「**141** gigabytes (GB) of HBM3e memory at **4.8** terabytes per second」「H200 NVL … Up to 600W」 | 4.8/141 ✓；NVL 变体 note | ✓ |
| V7 | B200 SM 数 + SKU | WebSearch 互证：glennklockwood.com/garden/processors/b200（160 physical/148 active）、chipsandcheese B200（74/die）、TechPowerUp B200 SXM6（18944=148×128）；HGX 页 Total Memory 1.4TB；Lenovo lp2226（180GB/7.7/1000W）；primeline datasheet 镜像（186/8 vs 180/7.7）；inference 1.3 L31 SKU 纪律 | 148 SM ✓；HGX 180/7.7 vs GB200 186/8 vs raw 192 → A-08 | ✓ |
| V8 | Blackwell SM 资源 | docs.nvidia.com/cuda/blackwell-tuning-guide/index.html：「concurrent warps per SM is **64** for … 10.0」「register file … **64K** 32-bit」「registers per thread … **255**」「thread blocks per SM … **32**」「shared memory capacity per SM is **228 KB**」「maximum shared memory per thread block … **227 KB**」「combined L1 cache, texture cache, and shared memory of **256 KB**」 | B200 资源与 H100 同 ✓ | ✓ |
| V9 | cache line / sector | docs.nvidia.com/nsight-compute/ProfilingGuide/index.html：「sector Aligned **32 byte**-chunk … An L1 or L2 cache line is **four sectors, i.e. 128 bytes**」「The L1 and L2 both have **128 byte cache lines**」 | 128B line / 32B sector ✓（spec 2.2/4.1） | ✓ |
| V10 | coalescing 官方规则 | docs.nvidia.com/cuda/cuda-c-best-practices-guide/ §10.2.1：「coalesce into a number of transactions equal to **the number of 32-byte transactions necessary to service all of the threads of the warp**」「CC 6.0+ … data access unit is **32-byte**」 | 通用式 = 不同 sector 计数 → A-03 | ✓ |
| V11 | strided 饱和 | BP §10.2.1.4：「stride of 2 results in a **50%** … efficiency … decreases **until the point where 32 32-byte segments are loaded for the 32 threads**」 | 1/s 至 32 segments 饱和 → A-04 | ✓ |
| V12 | shared banks | BP §10.2.3.1：「each bank has a bandwidth of **32 bits** every clock cycle, and **successive 32-bit words are assigned to successive banks** … number of banks is also **32**」；broadcast/multicast 例外；padding「TILE_DIM+1 … stride … 33 … equivalent to a unit stride」 | 32 banks×4B / bank=(addr/4)%32 / k-way→k / broadcast / +1 padding ✓（spec 2.3/4.2） | ✓ |
| V13 | 常量内存 / occupancy API | BP §10.2.6：「There is a total of **64 KB** constant memory on a device」；§11.1：「**cudaOccupancyMaxActiveBlocksPerMultiprocessor**」 | 64KB ✓；API 真实 ✓（spec 2.3/3.4） | ✓ |
| V14 | FP32 发射宽度 | BP Table 5（results/cycle/SM）：CC 9.0「32-bit floating-point add, multiply, multiply-add … **128**」 | 128=4 warp-inst → 单发射/scheduler → A-05 | ✓ |
| V15 | MMA 几何 | docs.nvidia.com/cuda/parallel-thread-execution/（PTX ISA 9.5，3.97MB）：`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`；`wgmma.mma_async.m64nNk16/k8/k32/k256` fragments；`tcgen05.mma.cta_group.kind`（.cta_group::1/::2，.kind::f16/tf32/f8f6f4，block_scale）；`tcgen05.alloc` | m16n8k16 / m64nNk16 / tcgen05 2-SM ✓；k 随 dtype 变（§5 heads-up） | ✓ |
| V16 | 异步拷贝 / ldmatrix | PTX：`cp.async.bulk`（TMA）、`cp.async.commit_group`/`wait_group`、`ldmatrix`、`stmatrix` | cp.async/TMA/ldmatrix 真实 ✓（spec 4.2/4.4） | ✓ |
| V17 | fence scope | PTX membar：「The **level qualifier specifies the set of threads that may observe the ordering effect**」；.gl/.cta/.sys | __threadfence=device scope → A-10 | ✓ |
| V18 | L2 persistence | cuda-programming-guide/04-special-topics/l2-cache-control.html：「accessPolicyWindow … persistence … CC 8.0 and above」 | cudaAccessPolicyWindow ✓（spec 2.2） | ✓ |
| V19 | HBM3 通道几何 | JEDEC jedec.org/standards-documents/docs/jesd238b01 + synopsys.com/glossary/what-is-high-bandwitdth-memory-3.html（WebSearch  mediated）：1024-bit/stack = 16×64-bit ch（32 pseudo-ch） | 16 ch×64bit ✓；width 口径 → A-02 | ✓ |
| V20 | 延迟数互证 | arxiv.org/html/2501.12084v1 Table III/IV（H800）：L1 32.0 / shared 29.0 / L2 264.5–502（near 258 / far 414）/ global 656 cycles；「50MB partitioned L2」 | 量级 ✓；L2 ~200 偏低（§5 heads-up） | ✓ |
| V21 | models 2.4 recap 源 | models/02-transformer-anatomy/04-causality-and-kv-cache.md：L188 式(9) 容量（§5）、L221/229/239 式(10)-(12) 强度（§6）、L216 min/ridge prose、L244 seam 句 | recap scope PASS；式号误引 → A-07 | ✓ |
| V22 | training 4.1 符号/式 | training/04-data-parallelism/01-dp-and-ddp.md §2.2：d=DP 度、式(2) 2(d-1)/d·M、式(3) 2(d-1)α+2(d-1)/d·Mβ、α/β 定义 | d→n 翻译 ✓； recap 式一致 ✓ | ✓ |
| V23 | inference R / datasheet / batch-invariance | inference 1.2 L19「R = ridge point = 峰值算力÷带宽」；inference 1.3 L19–31 dense 表 + SKU 纪律；inference 1.4 §6 batch invariance | R 沿用 ✓；datasheet 表 ✓；1.4 回链 ✓ | ✓ |
| V24 | 站点断言代码 | site/scripts/build-content.mjs L41–90 DIMENSIONS/asserts、L110–116 try/catch、L210 refs 硬读、L216–221 抛错 | refsMin/try-catch/硬读 claims 全 ✓；W1/W2 断言吻合 | ✓ |
| V25 | 指针面 | grep models/training/inference「归/属 gpu/cluster、gpu/cluster 维度」raw 150 hits（未排除）；training 4.1 L73「硬件侧归 cluster 维度」errata 目标存在 | 大 errata 面 ✓（73 句计数口径不同，builder 已复现）；D-S2-6 README 漏扫已覆盖 | ✓ |

**覆盖率**：任务书事实核查清单 10/10 项一手核实（SM 数 / register file / warp slots / shared-L1 / HBM 几何→带宽×3 / MMA 几何×3 / cache line-sector / shared banks）；P1–P4 全 16 章大纲逐章扫描（边界 + 符号 + 推导）；§6.3 强制重推导 4/4 条独立重算（1.2 FAIL / 2.4 PASS / 3.4 PASS+注 / 4.3 PASS+注）；builder-check 9/9 修复传播核实；站点断言代码核实。未覆盖：Hopper whitepaper PDF 正文（镜像超时，SM scheduler 以 BP Table 5 互证）；P1–P4 repo 路径锚（CUTLASS swizzle/pipeline 留 W2，B/C 路已部分覆盖）；JEDEC 规范原文 PDF（WebSearch mediated，未直取）。

## 7. 汇总与跨路去重

| severity | 数 | 项 |
| --- | --- | --- |
| S1 | 1 | A-01（1.2 occupancy 错误公式；D-S2-1 同址 S2，controller 裁定） |
| S2 | 6 | A-02（HBM width 口径；D-S3-5 同址 S3）/ A-03（ceil(span/32) 通用式）/ A-04（1/s 饱和）/ A-05（双发射）/ A-06（GEMM M/N/K/T；= D-S2-3/B-3 独立确认）/ A-07（models 2.4 式号误引） |
| S3 | 7 | A-08（B200 SKU 口径）/ A-09（228KB combined 口径，C-8 先例）/ A-10（threadfence scope）/ A-11（4.3 假设注，B-6 同族）/ A-12（Little's law 归一化）/ A-13（spill 量纲）/ A-14（附录 6 vs 9） |

**跨路重叠**：A-01↔D-S2-1（severity 分歧 S1/S2）、A-02↔D-S3-5（S2/S3）、A-06↔D-S2-3/B-3（同 S2）。其余 A-03/04/05/07/08/09/10/11/12/13/14 为本路新增（B/C/D 未覆盖）。D-S2-2（5.1 n=元素数）/ D-S2-4（注意力几何符号 + IO 定理 M 翻译）/ D-S3-8（R_file/S_total 入表）属符号表同族但 P5/P6 侧，本路不重复入表，§3.3 已指针。

**修复优先级建议**：A-01（S1，W1 写作前必修，1.2/3.4/§6.3 三处同步）→ A-02/A-03/A-04/A-05（W1/W2 写作前修，公式级）→ A-06/A-07（符号表 + recap 引用，W0/W1）→ S3 累计 W2 清零。全部 ⟨verify⟩ 数值项留各波 research pass（§5 heads-up 供引用）。
