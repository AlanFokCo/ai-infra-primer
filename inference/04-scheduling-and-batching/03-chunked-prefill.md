# 4.3 Chunked Prefill 与混合批（Chunked Prefill and Mixed Batching）

> 一条 32768 token 的 prefill 要算约 0.78 s，同批 64 条 decode 序列的下一个 token 就得等 0.78 s——TBT 尖刺 52 倍。怎样把长 prefill 切成 chunk、与 decode 混批，让 TBT 有界而 prefill 吞吐不塌？切多小才够、切多小又太碎？本章给出 stall 账、chunk size 权衡区间与 DeepInfer compute-quota 的二次求解 | 前置：1.1（TTFT / TPOT / TBT 口径、命题 1 步时与 TBT）、1.2（式 (1)(2)(3) 算力与带宽账、章脊口径）、1.3（roofline 与 ridge）、2.1（KV 分页）、4.1（迭代级调度与 selective batching）；models 2.4 §6（算术强度，recap 不重推） | 锚点：Sarathi（[Agrawal et al., 2023](../references.md)，arXiv:2308.16369，[A]）、Sarathi-Serve（[Agrawal et al., 2024](../references.md)，arXiv:2403.02310，[A]，OSDI'24）、Orca（[Yu et al., 2022](../references.md)，USENIX OSDI'22，无 arXiv，[B]）、vLLM repo（[vLLM](../references.md)，[A]）、SGLang repo（[SGLang](../references.md)，[A]）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）

## 1. stall：长 prefill 独占一个 iteration 的代价

32768 token 的 prompt 在 8B 口径下 prefill 计算下界约 781 ms，而 64 条序列的 decode iteration 只要 15.1 ms（1.2 §3 算例）。不开 chunked prefill 时，这条 prefill 占满一个完整 iteration：即使 64 条 decode 序列同批在跑，它们的下一个 token 也要等 781 ms 才产出，TBT 从 15.1 ms 跳到 781 ms，尖刺约 52 倍。这就是本章要解决的 stall 问题——长 prefill 不能劫持 decode 的延迟，prefill 的吞吐也不能因此塌掉。

所谓 stall，是指 decode 序列因其所在 iteration 混入非 decode 工作（整条 prefill、抢占重算）而等待、TBT 远超正常 decode 步时的现象（口径：按逐间隔 TBT 样本度量，不是请求级 TPOT 均值——均值会被几百个正常间隔稀释，尖刺看不见，1.1 §2 坑注）。机制是迭代级调度的直接推论：因为 batch 是执行的原子单位，iteration 时长由批内总工作量决定，所以批内所有序列共享这个时长（1.1 命题 1：每条序列的 TBT 恰为 $\tau_{\mathrm{step}}$ ）。整条 prefill 不可分——它的 FLOPs 是 $2N_{\mathrm{act}} T + 2LT^{2} d_{\mathrm{model}}$ （1.2 式 (1)）， $T$ = 32768 时两笔分别为 $4.92 \times 10^{14}$ 与 $2.81 \times 10^{14}$ ，合计 $7.73 \times 10^{14}$ FLOPs，按 H100 BF16 dense 峰值 989.5 TFLOPS 折算 781 ms（峰值下界口径：忽略访存与 launch 开销，与 1.2 一致；实际时间按 MFU 放大，MFU = 0.5 时约 1.56 s、尖刺约 104 倍）。中等长度也逃不掉： $T$ = 4096 的 prefill 66.5 ms，已是 decode 步时的 4.4 倍——stall 不是极端长度的专利。

```
不开 chunked prefill（T = 32768 prompt 到达，B_D = 64 条 decode 在跑）
时间 ─► |◄──────── prefill 独占 781 ms ────────►|◄ decode 15 ms ►|◄ decode 15 ms ►|
64 条 decode 的 TBT:        781 ms（尖刺 ≈52×）        15 ms            15 ms

开 chunked prefill（c_chunk = 512，64 个 chunk，每 iteration piggyback 全部 64 条 decode）
时间 ─► |◄ 混合₁ 20 ms ►|◄ 混合₂ 20 ms ►| ... |◄ 混合₆₄ 21 ms ►|◄ decode 15 ms ►|
TBT:         20 ms           20 ms                21 ms            15 ms
        （每个 iteration 都产出 1 token，TBT = iteration 时长，无尖刺）
```

解法有两手：把 prefill 切成 chunk（chunked prefill），再把每个 chunk 与全部待跑的 decode 合成一个批（混合批，mixed batch）。前者给单 iteration 的非 decode 工作量封顶，后者保证 decode 每个 iteration 都产出 token，两者合起来构成 stall-free 调度。另一条路是抢占（4.2）：把长 prefill 中断、让 decode 先跑，但抢占要付重算或 swap 的成本，且被抢的 prefill 自身 TTFT 恶化；chunking 是非抢占的替代——不中断任何工作，只改变工作的分批粒度。两条路的成本账对照归 4.2，本章只展开 chunking。本章围绕三问展开：混合批怎么组成、piggyback 为什么近乎免费（§2）；chunk size 在 stall 与效率之间的权衡怎么算（§3）；生产系统如何把权衡变成运行时求解器（§4–§6）。

本章口径（章脊锁定，全章例题复用）：8B dense @H100 SXM bf16，沿用 1.2 §1—— $N = 7.50 \times 10^{9}$ （matmul 口径）、权重 $W = 1.61 \times 10^{10}$ 字节、 $L$ = 32、 $d_{\mathrm{model}}$ = 4096、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $\kappa$ = 4096 字节/token/层；硬件 BF16 dense 峰值 $F$ = 989.5 TFLOPS、HBM 带宽 $\mathrm{BW}$ = 3.35 TB/s、ridge $R \approx 295$ （NVIDIA 官方 datasheet，[C]，出处与完整参数见 1.2 §1 及其参考）。场景参数：长 prompt $T$ = 32768、decode 上下文 $\bar{T}$ = 4096、decode 序列数 $B_D$ = 64（沿用 1.2 §3 decode 算例），换长度随文声明。两个教学假设先声明：每 iteration 固定开销 $\tau_{\mathrm{oh}}$ = 1 ms（组批、kernel launch、采样等与 token 数无关的成本；CPU 部分可被 overlap scheduler 隐藏，归 6.2）；§4 的时间预算算例取 MFU = 0.5 的有效算力——预算必须反映真实时间，峰值下界会超装。

记号（spec §4.3 章首局部声明）： $c_{\mathrm{chunk}}$ = chunk size（token 数）； $c$ = 已算前缀长（token 数）——**本章 $c$ 是前缀长，不是 CP 度**（CP 度不在本章出现）；§4 的 DeepInfer $\mathrm{cost}(q, c)$ 沿 doc 原记号， $q$ = chunk token 数（即 $c_{\mathrm{chunk}}$ ）， $c$ 语义不变。 $B_D$ = 混合批内 decode 序列数； $\bar{T}$ = decode 序列平均上下文长； $\tau_{\mathrm{step}}$ = iteration 步时（1.1 命题 1 记号）。

## 2. Sarathi：chunked prefill 与 decode piggyback

切 chunk 为什么能让 decode 免费搭车？先回答一个更基本的问题：一个纯 decode iteration 到底在付什么钱。权重读一遍 $W/\mathrm{BW}$ = 4.8 ms，加 $B_D$ 份 KV 读 $B_D \kappa L \bar{T}/\mathrm{BW}$ = 10.3 ms——GEMM 是 memory-bound 的 GEMV，算力几乎闲着。若某个 iteration 本来就要做 compute-bound 的 prefill GEMM，权重反正要读，把 64 行 decode token 并进同一个 GEMM 只增加 $2N_{\mathrm{act}} \times 64/F \approx 1.0$ ms 计算，却省掉了一次独立的 4.8 ms 权重读。这就是 piggyback（捎带）的算术核心。

所谓 chunked prefill，是指把 $T$ token 的 prompt 切成 $m = T/c_{\mathrm{chunk}}$ 个 chunk、逐 iteration 执行，第 $i$ 个 chunk 对已算前缀的 KV 做 attention（读 cache，不重算前缀）。正例： $T$ = 32768、 $c_{\mathrm{chunk}}$ = 512 切成 64 个 chunk。边界： $c_{\mathrm{chunk}} = T$ 退化为不切（stall 回归）； $c_{\mathrm{chunk}}$ = 1 退化为逐 token prefill，attention 强度只有 4 FLOP/byte（§3 式 (5)），彻底 memory-bound——两个极端都不可取，权衡在中间。所谓混合批（mixed batch），是指一个 iteration 的 batch 同时含一个 prefill chunk 的 $c_{\mathrm{chunk}}$ 个 token 与 $B_D$ 条 decode 序列的各 1 个 token。Sarathi 把这两手与 decode-maximal batching 组合：每个 prefill chunk piggyback 尽可能多的 decode、把 decode batch 填到内存上限，从而将 memory-bound 的 decode 转成 compute-bound 的批 GEMM（[Agrawal et al., 2023](../references.md)，[A]；官方实现是 microsoft/sarathi-serve 内的 sarathi scheduler provider，原文无自链 repo，等级依据见 references）。chunking 还有第二个用途：统一的 chunk 工作单元让流水各段负载均匀、减 PP bubble（serving 侧 PP 账归 8.1，此处不展开）。

混合批的组成沿用 Orca selective batching（4.1 recap：attention 逐序列算、其余算子按 token 合批，[Yu et al., 2022](../references.md)，[B]）。为什么 attention 必须拆开？因为批内各序列的 attention 形状异构：chunk 是 $c_{\mathrm{chunk}}$ 行 query 对 $c + c_{\mathrm{chunk}}$ 长 KV 的 causal 块，decode 是 1 行 query 对各自 $\bar{T}$ 长 KV 的满读——若 padding 到统一形状再合批，短序列要为长序列的空位白算，浪费随长度方差涨。拆开后各序列走 varlen / paged attention kernel（接口与 kernel 算法归 3.1），GEMM 侧则完全同构（每行都是 $d_{\mathrm{model}}$ 维 token），合批无浪费。shape 走读（8B 口径， $c_{\mathrm{chunk}}$ = 512、 $B_D$ = 64）：

```
混合批：512 个 prefill token + 64 个 decode token
──────────────────────────────────────────────────────────────
token 矩阵 X: [576, 4096]（512 + 64 行，每行一个 token）
   │
   ├─► QKV / O 投影 + FFN GEMM：576 行全部合批
   │     权重读一遍 W = 16.1 GB；计算 2N×576 FLOPs
   │     decode 的 64 行搭同一次权重读 —— piggyback
   │
   └─► attention：逐序列拆开（selective batching）
         prefill chunk: Q [512, 32, 128] 对 前缀+chunk KV [c+512, 8, 128]
                        causal mask，varlen / paged kernel（3.1）
         decode 序列 i: Q [1, 32, 128] 对 自己的 KV [T̄_i, 8, 128]，共 64 份
                        （各读各的 KV，不可共享）
──────────────────────────────────────────────────────────────
GEMM 相:  max(2N×576/F, W/BW) = max(8.7, 4.8) = 8.7 ms（compute-bound）
attention 相: decode KV 读 64×0.537 GB = 34.4 GB → 10.3 ms（memory-bound）
              chunk 自 attention 计算 0.07 ms + decode attention 计算 0.14 ms，可忽略
iteration ≈ 8.7 + 10.3 + 1(τ_oh) = 20.0 ms
```

代入章脊口径：GEMM 相 $\max(2N \times 576/F,\ W/\mathrm{BW})$ = max(8.7, 4.8) = 8.7 ms，compute-bound；attention 相里 decode KV 读 34.4 GB 折 10.3 ms，远超 chunk 自身的 attention 计算（0.07 ms）与 decode attention 计算（0.14 ms），memory-bound。iteration 时长 ≈ 8.7 + 10.3 + 1 = 20.0 ms——64 条 decode 每条产出 1 token，TBT = 20.0 ms，对比不开 chunk 的 781 ms 尖刺。

piggyback 的边际账（路线：混合 iteration 减纯 prefill iteration 得 decode 的边际成本，再与独立 decode iteration 对比得节省，关键一步是权重读只付一次）。 $c_{\mathrm{chunk}}$ = 512、 $c$ = 0 时纯 prefill iteration ≈ 8.8 ms，混合 iteration 20.0 ms，decode 边际成本 11.2 ms；独立 decode iteration 15.1 ms。（口径：边际账两边均略固定开销 $\tau_{\mathrm{oh}}$ 。边际 11.2 ms 是两个含 oh 的 iteration 之差，oh 相消；独立基线 15.1 ms 取 1.2 带宽下界，亦免 oh。若计入独立 iteration 自身的开销，节省为 4.9 ms——piggyback 还多摊掉一份 iteration 开销；正文取保守口径，下文反例同口径。）**节省 ≈ 3.9 ms/iteration**，主项是被摊掉的权重读 $W/\mathrm{BW} - 2N_{\mathrm{act}} B_D/F$ = 4.8 − 1.0 = 3.8 ms（0.1 ms 差是 decode attention 计算与相 regime 细节）。节省为正的条件： $B_D \lt WF/(2N_{\mathrm{act}} \mathrm{BW}) \approx 317$ （8B 峰值口径）——超过它，decode 的边际 GEMM 计算超过它独立跑时付的权重读，GEMM 相的节省转负。边界反例： $B_D$ = 512 时边际成本反而比独立 decode 步贵 2.9 ms。但此时 stall-free 的延迟收益仍在——decode 照样每 iteration 产出 token，TBT 仍等于 iteration 时长。**两笔收益分开记：吞吐节省有条件，延迟有界无条件。**

> **提点**：piggyback 的「免费」是窄口径（口径：省的是独立权重读 $W/\mathrm{BW}$ ，decode 的 KV 读 $B_D \kappa L \bar{T}/\mathrm{BW}$ 两种形态都全额付）。方向还能反过来：chunk 小到 GEMM 计算 < $W/\mathrm{BW}$ （ $c_{\mathrm{chunk}} + B_D \lt 317$ ）时，是 chunk 搭 decode 权重读的便车——同一个 max() 代数的两种读数，§3 式 (2) 统一处理。

decode batch 的上限由什么定？Sarathi 原文给三个决定因素：GPU 内存、参数内存、最大序列长。容量界写成 $B_D \le (M_{\mathrm{GPU}} - W - M_{\mathrm{act}})/(\kappa L T_{\max})$ （ $M_{\mathrm{act}}$ 为激活与预留，教学上界略去）：H100 80 GB、 $T_{\max}$ = 4096 代入得 (80 − 16.1) GB ÷ 0.537 GB ≈ 119 条。decode-maximal batching 就是填到这个上限；真实可用量由 KV 分页与 watermark 决定（2.1）。

但「填到上限」不是无代价的： $B_D$ 越大，每个混合 iteration 的 attention 相要读的 decode KV 越多，TBT 的地板随之抬高。代入章脊口径： $B_D$ = 64 时 decode KV 读 34.4 GB、折 10.3 ms；填到 $B_D$ = 119 时读 63.9 GB、折 19.1 ms——attention 相地板几乎翻倍，GEMM 相还要多算 55 个 token。所以 decode-maximal batching 自身也有一笔吞吐-延迟账：更多序列同时推进（decode 总吞吐升），但每条序列的 TBT 变长（1.2 §3「凑 batch 收益递减」在混合批里的同形读数）。操作点取多大 $B_D$ ，由 TBT SLO 反推——式 (2) 对 $B_D$ 同样单调，§3 的区间代数原样适用。

## 3. stall 账：chunk size、TPOT 与 prefill 效率

chunk 切多小才够？切多小又太碎？本节从同一个 iteration 时间模型推出两个界：上界来自 TPOT SLO（切太大，stall 回归），下界来自 prefill 效率（切太小，attention 与固定开销吃掉吞吐）。路线：先给 chunk attention 的 FLOPs 式与 iteration 时间模型，再证 chunking 下 attention FLOPs 守恒（命题 1）——把效率损失定位到访存而非计算，然后给 KV 重读与强度式，最后把上下界合成可行区间。

**chunk attention FLOPs**。第 $i$ 个 chunk 有 $c_{\mathrm{chunk}}$ 个新 token，全局位置 $c+1$ 到 $c+c_{\mathrm{chunk}}$ ，第 $j$ 个 token 可见 $c+j$ 个位置；可见位置总数 $c_{\mathrm{chunk}} c + c_{\mathrm{chunk}}(c_{\mathrm{chunk}}+1)/2$ ，每个位置对 $QK^{\top}$ 与 $PV$ 各贡献 $2d_{\mathrm{model}}$ FLOPs、共 $L$ 层：

$$
C_{\mathrm{attn}}(c_{\mathrm{chunk}}, c) = 4 L d_{\mathrm{model}} \left( c_{\mathrm{chunk}} c + \frac{c_{\mathrm{chunk}}(c_{\mathrm{chunk}}+1)}{2} \right) \qquad (1)
$$

**符号**： $C_{\mathrm{attn}}$ ：单个 chunk 的 attention FLOPs（不含 decode）； $c$ ：已算前缀长；其余同章脊。与 1.2 式 (1) 对齐：令 $c$ = 0、 $c_{\mathrm{chunk}} = T$ 得 $2LT(T+1)d_{\mathrm{model}}$ ，即 1.2 二次项 $2LT^{2} d_{\mathrm{model}}$ 的精确形（差 $1/T$ 相对量级）。**1.2 的系数 2 已含 causal 平均可见长 $T/2$ （W1 裁定），本章任何地方不再另打对折。**全数字实例： $c_{\mathrm{chunk}}$ = 512、 $c$ = 8192 代入得 $4 \times 32 \times 4096 \times (512 \times 8192 + 131328) = 2.27 \times 10^{12}$ FLOPs，峰值折算 2.29 ms。

> **坑**：causal 双重折扣。触发条件：见「causal 减半」口诀后把式 (1) 再乘 1/2，或把 chunk 自 attention 写成 $q^{2}/2$ 后再对 $4Ld_{\mathrm{model}}$ 折半；现象：attention 账系统性低估 2 倍，chunk 下界与 §4 quota 求解同步偏小、SLO 被超装；规避：式 (1) 的 $c_{\mathrm{chunk}}(c_{\mathrm{chunk}}+1)/2$ 就是 causal 精确和， $4Ld_{\mathrm{model}}$ 是每位置对的完整系数，两处都不再折扣。

**iteration 时间模型**。混合 iteration 的工作分两相：GEMM 相（投影 + FFN，全部 token 合批）与 attention 相（chunk attention + $B_D$ 条 decode attention）。各相取自己的 roofline 界、两相相加（口径：教学两相集中模型，无 intra-device 重叠；真实执行逐层交替但总量同形，kernel 级重叠归 3.4 / 6.2）：

$$
\tau_{\mathrm{step}}(c_{\mathrm{chunk}}, c) \approx \max\left( \frac{2N_{\mathrm{act}}(c_{\mathrm{chunk}} + B_D)}{F}, \frac{W}{\mathrm{BW}} \right) + \max\left( \frac{C_{\mathrm{attn}} + B_D \cdot 4L\bar{T}d_{\mathrm{model}}}{F}, \frac{\kappa L (c + c_{\mathrm{chunk}}) + B_D \kappa L \bar{T}}{\mathrm{BW}} \right) + \tau_{\mathrm{oh}} \qquad (2)
$$

**符号**：第一 max 是 GEMM 相（计算 vs 权重读）；第二 max 是 attention 相（计算 vs KV 读：前缀 + chunk 一份、decode $B_D$ 份）； $\tau_{\mathrm{oh}}$ = 1 ms 教学假设。KV 写每 chunk 仅 $\kappa L c_{\mathrm{chunk}}$ 字节，比读小 $c/c_{\mathrm{chunk}}$ 倍量级，略去（同 1.2 §3 边界注）。实例：§2 的 $c_{\mathrm{chunk}}$ = 512、 $c$ = 0 给 20.0 ms；同一 chunk size 到长 prompt 后段 $c$ = 24576 时给 21.0 ms——前缀变长，步时漂移，§6 处理。

**命题 1**（chunking 的 attention FLOPs 守恒）。对 $T$ 的任意正整数划分 $T = \sum_i q_i$ （ $c_i = \sum_{j \lt i} q_j$ 为第 $i$ 个 chunk 的前缀），总 attention FLOPs 与划分无关：

$$
\sum_i 4Ld_{\mathrm{model}} \left( q_i c_i + \frac{q_i(q_i+1)}{2} \right) = 2LT(T+1)d_{\mathrm{model}} \qquad (3)
$$

**证明**（路线：把每 chunk 的可见位置对数写成累计长度平方差，裂项相消，三步）。其一，记 $P_i = c_i + q_i$ 为第 $i$ 个 chunk 结束时的累计长度，则 $q_i c_i + q_i^{2}/2 = (P_i^{2} - P_{i-1}^{2})/2$ （展开 $P_i^{2} - P_{i-1}^{2} = 2c_i q_i + q_i^{2}$ 即得）；其二，补上 causal 精确和的 $+q_i/2$ 项，每 chunk 对数 $= (P_i^{2} - P_{i-1}^{2})/2 + q_i/2$ ；其三，对 $i$ 求和，平方差裂项相消剩 $P_m^{2}/2 = T^{2}/2$ ， $\sum q_i/2 = T/2$ ，合计 $T(T+1)/2$ ，乘 $4Ld_{\mathrm{model}}$ 即式 (3)。∎

正例（等长 chunk）： $T$ = 4096、 $c_{\mathrm{chunk}}$ = 512、 $m$ = 8，逐 chunk 对数求和 = 8390656，与整条 prefill 的 $T(T+1)/2$ = 8390656 逐位相等，FLOPs 同为 $4.40 \times 10^{12}$ （1.2 式 (1) 首项 $4.398 \times 10^{12}$ ，差 0.024%）。命题对不等长 chunk 同样成立（证明未用等长假设）。边界反例：若实现把前缀 KV 重算而非读 cache，守恒即破——那是实现错误，不是 chunking 的固有成本。

> **提点**：FLOPs 守恒、字节不守恒。chunking 的效率损失不在「多算了」，而在前缀 KV 被反复读、固定开销被反复付——优化对象是访存流量与 iteration 数，不是 FLOPs。下面两式把这两笔定量化。

**前缀 KV 重读**。等长 chunk 下第 $i$ 个 chunk 读前缀 $c_i = (i-1)c_{\mathrm{chunk}}$ 个 token 的 KV，求和（路线：等差数列计数，两步）： $\sum_i c_i = c_{\mathrm{chunk}} m(m-1)/2 = T(m-1)/2$ ，乘 $\kappa L$ 得

$$
M_{\mathrm{reread}} = \kappa L \cdot \frac{T(m-1)}{2} \approx \frac{\kappa L T^{2}}{2 c_{\mathrm{chunk}}} \qquad (4)
$$

**符号**： $M_{\mathrm{reread}}$ ：整条 prefill 期间前缀 KV 的累计重读字节； $m = T/c_{\mathrm{chunk}}$ 。重读流量 $\propto 1/c_{\mathrm{chunk}}$ ：chunk 减半，重读翻倍。实例（ $T$ = 32768）： $c_{\mathrm{chunk}}$ = 512 时 $m$ = 64，重读 135 GB、带宽折算 40.4 ms； $c_{\mathrm{chunk}}$ = 64 时 $m$ = 512，重读 1097 GB、折 328 ms——而 attention 计算总量由命题 1 锁死在 284.5 ms（峰值），小 chunk 让访存反超计算。

**attention 强度**。chunk attention 的计算量 $\approx 4Ld_{\mathrm{model}} c_{\mathrm{chunk}} c$ （ $c \gg c_{\mathrm{chunk}}$ 时），KV 读 $\approx \kappa L c$ ，相除：

$$
I_{\mathrm{attn}} = \frac{4 c_{\mathrm{chunk}} d_{\mathrm{model}}}{\kappa} = \frac{2 c_{\mathrm{chunk}}}{\mathrm{bytes}} \cdot \frac{H_q}{H_{kv}} \qquad (5)
$$

**符号**： $H_q$ ：query head 数（8B 为 32）； $\mathrm{bytes}$ ：每元素字节数（bf16 = 2）；第二等号代入 GQA 的 $\kappa = 2H_{kv}d_{\mathrm{head}} \cdot \mathrm{bytes}$ 。强度与 $c$ 无关、随 $c_{\mathrm{chunk}}$ 线性涨——每读一份前缀 KV，只摊到 $c_{\mathrm{chunk}}$ 个 query 的计算。8B bf16 GQA-8 代入得 $I_{\mathrm{attn}} = 4 c_{\mathrm{chunk}}$ ： $c_{\mathrm{chunk}}$ = 512 时 2048 ≫ $R$ = 295（compute-bound）； $c_{\mathrm{chunk}}$ = 64 时 256 < 295（memory-bound）。compute-bound 的硬下界： $c_{\mathrm{chunk}} \ge R\kappa/(4d_{\mathrm{model}})$ = 295/4 ≈ 74 token。这就是「chunk 太小 attention 二次项效率低」的机制：二次项的活没少（命题 1），但每字节 KV 读摊到的计算变少，相 regime 从算力封顶滑到带宽封顶。

**stall-free 与 TBT 上界**。所谓 stall-free batching，是指每个 iteration 都带上全部在跑的 decode 序列、没有任何 decode 需要等待整条 prefill 的调度（Sarathi-Serve，[Agrawal et al., 2024](../references.md)，[A]，OSDI'24）。

**命题 2**（stall-free 的 TBT 上界）。迭代级调度下，若每个 iteration 都包含全部在跑 decode 序列且各产出 1 token，则任一 decode 序列的 TBT 等于其所在 iteration 的步时；若进一步每个 iteration 步时 $\le \tau_{\max}$ ，则 TBT $\le \tau_{\max}$ 。

**证明**（路线：对 iteration 序号归纳，不变式三段——基例：首个含该序列的 iteration 产出第 1 个 token；保持：第 $k$ 个 iteration 后序列恰产出 $k$ 个 token、相邻 token 间隔即该 iteration 步时；终止：任意时刻 TBT 样本都是某个 $\tau_{\mathrm{step}}$ ，上界逐样本成立。形式化归纳与边界反例留作练习 2）。∎

实例对照：不开 chunk 时 prefill iteration 的 781 ms 就是一个 TBT 样本（命题 2 的前提被破坏——decode 虽在批内，步时却由 prefill 独占决定）；开 chunk 后最大 TBT 样本 = 最大 iteration 步时， $c_{\mathrm{chunk}}$ = 512 时约 21 ms，尖刺从 52 倍压到 1.4 倍。命题 2 把「stall 账」化简为一件事：**控制每个 iteration 的步时上界**。式 (2) 里随 $c_{\mathrm{chunk}}$ 单调涨的是 GEMM 计算项与 chunk attention 项，于是 SLO 给 chunk size 一个上界。取 GEMM compute-bound、attention 相近似常数 $\tau_{\mathrm{attn}}$ 的早期 chunk（ $c$ = 0）线性化：

$$
c_{\mathrm{hi}} \approx \frac{(\tau_{\mathrm{SLO}} - \tau_{\mathrm{oh}} - \tau_{\mathrm{attn}}) F}{2N_{\mathrm{act}}} - B_D \qquad (6)
$$

**符号**： $c_{\mathrm{hi}}$ ：chunk size 上界； $\tau_{\mathrm{SLO}}$ ：TBT 的 SLO 上限（即命题 2 的 $\tau_{\max}$ 取 SLO 值）； $\tau_{\mathrm{attn}}$ ：attention 相时长（ $B_D$ = 64、 $\bar{T}$ = 4096 时 ≈ 10.3 ms）。代入 $\tau_{\mathrm{SLO}}$ = 50 ms：(50 − 1 − 10.3) ms × 989.5 TFLOPS ÷ 15.0 GFLOPs/token − 64 ≈ 2553 − 64 = 2489；数值精确解 2485（差来自 attention 相对 $c_{\mathrm{chunk}}$ 的弱依赖）。

**权衡表**（ $T$ = 32768；效率列取纯 prefill $B_D$ = 0 口径以隔离 chunk size 效应；TPOT 列取混合批 $c$ = 0、 $B_D$ = 64； $\tau_{\mathrm{oh}}$ = 1 ms）：

| $c_{\mathrm{chunk}}$ | $m$ | attention regime（ $I_{\mathrm{attn}}$ ） | prefill 总时长 | 对比不切 781 ms | 混合 iteration TPOT |
| --- | --- | --- | --- | --- | --- |
| 64 | 512 | memory-bound（256） | 1338 ms | +71% | 16.1 ms |
| 128 | 256 | compute-bound（512） | 1037 ms | +33% | 16.1 ms |
| 512 | 64 | compute-bound（2048） | 845 ms | +8.2% | 20.0 ms |
| 2048 | 16 | compute-bound（8192） | 797 ms | +2.0% | 43.4 ms |
| 8192 | 4 | compute-bound（32768） | 785 ms | +0.5% | 144.1 ms |

读表：TPOT 列随 $c_{\mathrm{chunk}}$ 涨（stall 界），总时长列随 $c_{\mathrm{chunk}}$ 降（效率）。 $c_{\mathrm{chunk}}$ = 64 / 128 的 TPOT 同为 16.1 ms，因为 GEMM 还在权重读地板下（ $c_{\mathrm{chunk}} + B_D \lt 317$ ），iteration ≈ decode 步时 + 开销。把总时长列拆成三项（GEMM + attention 相 + 开销），膨胀的来源一目了然：GEMM 项恒为 497 ms（参数项 FLOPs 与划分无关），attention 计算项由命题 1 锁死在 284 ms，所以 $c_{\mathrm{chunk}}$ = 512 的 +8.2% 几乎全来自 64 ms 开销； $c_{\mathrm{chunk}}$ = 64 的 +71% 里，开销占 512 ms、attention 相因滑入 memory-bound 再涨 45 ms（329 vs 284 ms）——小 chunk 的惩罚以开销为主、访存 regime 恶化为辅。两列口径不同不可直接互比：混合形态下 prefill 窗口内 decode 产出了 $m$ 轮 token，纯 prefill 形态一个没产——比较必须在 SLO 约束下比 goodput（1.1 式 (3)），不是比裸时长。

TTFT 侧也要记一笔：chunking 把 prefill 完成时间从 781 ms 拉到 845 ms（ $c_{\mathrm{chunk}}$ = 512、纯 prefill 口径），TTFT 的计算成分涨 8%——这是给 TBT 上界付的保费。若 TTFT SLO 同样紧，chunk size 就不能只按 TBT 选，要在两个 SLO 之间联立（排队成分在高负载下主导 TTFT，归 1.3）；Sarathi-Serve 的 throughput-latency 权衡正是这组联立约束下的操作点搜索（[Agrawal et al., 2024](../references.md)，[A]）。

下界侧：attention 硬下界 74 token；固定开销摊销与 GEMM 的 M 维效率把实用下界推到数百 token 量级（开销占比要求 $\tau_{\mathrm{step}} \ge 10 \tau_{\mathrm{oh}}$ ，仅解 GEMM 项得 $c_{\mathrm{chunk}} \ge 594$ ——口径：略去 chunk 自 attention 的近似，含自 attention 的完整式 (2) 解 ≈ 588，差约 1%，量级结论不变；GEMM tile 效率的量化归 3.3 / gpu 4.3 / 5.2）。本章取 512 为教学工作点（口径：综合下界的教学取值，非推导硬界——此时纯 prefill 开销占比 11%、混合 5%， $I_{\mathrm{attn}}$ = 2048）。于是章脊场景的可行区间约 [512, 2485]。**区间可以为空**： $\tau_{\mathrm{SLO}}$ 收紧到 20 ms 时式 (6) 给 $c_{\mathrm{hi}} \approx 510 \lt 512$ ——没有 chunk size 同时满足效率与 SLO，chunked prefill 独力无解，需要 PD 分离（8.2）或抢占 / 优先级（4.2）。Sarathi-Serve 的 throughput-latency 权衡就是这个区间内的操作点选择：负载高、SLO 松时偏上界取吞吐，SLO 紧时偏下界取延迟；chunk size 调优即区间搜索（[Agrawal et al., 2024](../references.md)，[A]）。其 repo 含 orca / vllm / faster_transformer 对照 scheduler（sarathi/core/scheduler/），可在同一 workload 下复现这条权衡曲线。

## 4. DeepInfer compute-quota：时间预算解出 chunk size

§3 的可行区间是静态分析；DeepInfer 的生产调度器把它变成每步运行时求解：给定纳秒级时间预算，反解恰好装得下的最大 chunk token 数——同样 50 ms 预算，前缀 8192 时解出 chunk 1260 token，前缀 65536 时只解出 499。官方 doc 逐字（L143，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露，2026-11-23 核实）：

> Each prefill step is given a time budget (`compute_quota_limit`, in nanoseconds), and a fitted quadratic cost model `cost(q, c)` predicts how long the step will take (`q` = number of sequence tokens, `c` = the prefix length already computed for that sequence). The scheduler solves that quadratic for `q`, taking the largest `q` that stays within the 32,768-token budget while exactly consuming the remaining time budget. The budget itself is further reduced by an all-reduce MIN within the DP group.

机制三件：预算以纳秒计（`compute_quota_limit`）； $\mathrm{cost}(q, c)$ 是拟合的二次成本模型；解二次方程取最大 $q$ ，受时间预算与 32768 token 上限双约束，预算再经 DP 组 all-reduce MIN 缩减。doc 未披露拟合系数——下面的二次形由本章账反推（教学推导），算例系数是教学回算值，不是生产数值。

**二次形**。路线：chunk 的时间成本三来源——参数 GEMM（ $q$ 线性）、前缀 attention（ $q \times c$ 双线性）、chunk 自 attention（ $q$ 二次）——合并同类项即标准二次式。取有效算力 $F_{\mathrm{eff}}$ = MFU × $F$ （时间预算须反映真实时间；MFU = 0.5 教学假设），式 (1) 的 FLOPs 除以 $F_{\mathrm{eff}}$ ：

$$
\mathrm{cost}(q, c) = u q^{2} + (v_0 + v_1 c) q \qquad (7)
$$

**符号**： $u = 2Ld_{\mathrm{model}}/F_{\mathrm{eff}}$ ：自 attention 系数（ns/token²）； $v_0 = 2N_{\mathrm{act}}/F_{\mathrm{eff}}$ ：参数 GEMM（ns/token）； $v_1 = 4Ld_{\mathrm{model}}/F_{\mathrm{eff}}$ ：前缀 attention（ns/token/前缀 token）；常数项取 0（固定开销另计入预算，教学简化）。8B 代入： $u \approx 0.53$ ns、 $v_0 \approx 30300$ ns、 $v_1 \approx 1.06$ ns（四舍五入的教学值）。二次项系数来自 causal 自 attention 的 $q^{2}/2$ ，已含 1/2，与式 (1) 同口径、无双重折扣。为什么 doc 强调「fitted」？因为二次形是结构的（FLOPs 账决定项数与幂次），系数却是经验的——MFU、kernel 效率、launch 开销都吸进系数里，随模型与硬件变，必须实测拟合。本章用账反推系数只为把机制算到底；生产系数未披露，不可把教学值当 DeepInfer 参数引用。

**求解**。令 $\mathrm{cost}(q, c)$ = quota（剩余时间预算），标准形 $u q^{2} + (v_0 + v_1 c)q - \mathrm{quota} = 0$ ，取正根（路线：判别式 → 正根 → 向下取整 → 双 cap，四步）：

$$
q^{\ast} = \frac{-(v_0 + v_1 c) + \sqrt{(v_0 + v_1 c)^{2} + 4u \cdot \mathrm{quota}}}{2u} \qquad (8)
$$

负根舍去（token 数非负）；向下取整；最后 $q = \min(q^{\ast}, 32768)$ ，再按剩余 prompt token 数截断。全数字算例（quota = 50 ms = $5 \times 10^{7}$ ns，教学系数）： $c$ = 8192 时 $v_0 + v_1 c$ = 38983.5 ns，判别式开方 ≈ 40320.1 ns， $q^{\ast}$ ≈ (40320.1 − 38983.5)/1.06 ≈ 1260.9，取整 **1260**；回代 $\mathrm{cost}(1260, 8192)$ = 49.96 ms（取整损失 0.08%）。 $c$ = 65536 时同预算解出 **499**（回代 49.92 ms）。读数：同一 50 ms 预算，前缀从 8192 涨到 65536，chunk 从 1260 缩到 499—— $v_1 c$ 项吃掉预算。这正是 §6 动态机制的生产实例：quota 固定， $q$ 随 $c$ 自适应。32768 token 上限是 doc 明示的单步 token 帽，与时间预算双约束、谁紧谁生效（本例两个解都远低于帽，起作用的是时间约束）。

**DP all-reduce MIN**。doc 只给操作，机制解读如下（结构论证）：DP attention 形态下各 rank 本地算自己序列的 attention，FFN/MoE 部分走 EP 集合通信、每步全组同步（8.1）；同步步的墙钟由最慢 rank 决定。若各 rank 按自己的预算解 $q$ ，预算宽的 rank 取更大 chunk、步时超出紧预算 rank 的容许，全组等它。MIN 把组预算缩到最紧 rank 的预算，各 rank 在同一（保守）预算下解各自的 $q$ ——没有 rank 超自己的容许，步时有界。代价是有余裕的 rank 利用不足；生产缓解靠负载均衡：doc L137 把 prefill 分四长度档（<16K / 16–64K / 64–256K / ≥256K），各档设 compute-quota cap、引擎按上轮需求份额动态重分配；L139 引擎按剩余 quota pull work、无 prefix-affinity 路由（官方披露，2026-11-23 核实）。分档让同档序列的 $c$ 接近，MIN 的保守性损失变小；完整负载均衡账归 7.4。

```
DP 组（d 个 rank；attention 各 rank 本地算，FFN/MoE 集合通信每步同步）
rank 1: 本地 quota 50 ms ─┐
rank 2: 本地 quota 40 ms ─┴─► all-reduce MIN = 40 ms（组预算）
两 rank 各按 40 ms 解式 (8)：c = 8192 → q = 1012；c = 65536 → q = 400
同步步时 = max(各 rank 实际步时) ≤ 预算（保守界）
```

数值：两 rank 预算 50 / 40 ms，MIN 后 40 ms； $c$ = 8192 解出 $q$ = 1012， $c$ = 65536 解出 $q$ = 400——预算相同、前缀不同，chunk 各异。MIN 缩的是预算，不是直接缩 $q$ ； $q$ 的缩减由式 (8) 逐 rank 完成。

「恰好装满剩余预算」这句还有一层调度语义：doc L139 的引擎按剩余 quota pull work（官方披露），quota 是每步的消耗品——解出的 $q$ 让这一步的时间成本贴着预算走，而不是「先估一个 chunk、跑完再看超没超」。对照 token 预算引擎（§5）：固定 token 数、步时随前缀浮动，SLO 只能事后验证；固定时间、token 数浮动，SLO 在组批时就进入约束。两种参数化的控制论差异是：前者控输入、后者控输出，而 TBT SLO 约束的恰是输出（步时）。

## 5. 引擎实现：vLLM、SGLang、Sarathi-Serve

三个引擎的 chunked prefill 配置面长什么样？本节走读机制例证片段（模块全量走读归 5.2 / 6.2，两层纪律），路径锚定 2026-11-23 tarball 核实。

**vLLM**。三个字段（config/scheduler.py，[vLLM](../references.md)，[A]）：`max_num_batched_tokens`（L47）是每 iteration 的 token 预算——prefill chunk token 与 decode token 共用（decode 每序列每步 1 token）；`long_prefill_token_threshold`（L104）设定触发 chunking 的 prompt 长度阈值；`long_prefill_token_threshold_adaptive`（L112）按 token 预算与请求数自适应调整阈值。预算分配的教学伪码（教学重写，非官方代码；官方实现逐行走读归 5.2）：

```python
# 教学重写：每 iteration token 预算分配（机制示意，非官方代码；可运行）
max_num_batched_tokens = 2048
running_decode = [f"dec{i}" for i in range(64)]          # 64 条 decode 序列
waiting_prefill = [{"id": "req0", "remaining_prompt": 32768}]
budget = max_num_batched_tokens
batch = []
for req in running_decode:                 # decode 优先，每序列 1 token
    batch.append((req, 1)); budget -= 1
for req in waiting_prefill:                # prefill 用剩余预算装 chunk
    if budget <= 0: break
    q = min(req["remaining_prompt"], budget)   # 长 prompt 自然拆成多个 chunk
    batch.append((req["id"], q)); budget -= q
assert sum(n for _, n in batch) <= max_num_batched_tokens
# batch 即混合批：64×1 + 1×1984 = 2048 token
```

decode 排在 prefill 前面不是随意的：decode token 是延迟敏感工作（每序列只差 1 token 就产出），prefill 是吞吐型工作、可以等下一轮预算——先保延迟、再填吞吐，与命题 2 的 stall-free 目标同向。运行片段得 chunk = 1984 token（2048 − 64），长 prompt 的其余部分在后续 iteration 继续装填。

**SGLang**。`chunked_prefill_size` 已从 server_args.py 迁出——当前 main 经 runtime_context.py 的 `get_schedule()`（L1351）读取，scheduler 侧经 `init_chunked_prefill`（scheduler.py L1468）初始化，动态 chunk size 实现在 scheduler_components/dynamic_chunk_sizer.py（[SGLang](../references.md)，[A]，2026-11-23 核实；函数级语义归 6.2）。

> **坑**：引用旧路径 server_args.chunked_prefill_size。触发条件：照 2026 年之前的教程或旧版代码写配置、做走读锚；现象：当前 main 该字段不在 server_args（已迁 runtime_context），配置读取与走读锚点双双落空；规避：锚 runtime_context.py 的 `get_schedule()` + scheduler.py 的 `init_chunked_prefill`，版本日期 2026-11-23。

**Sarathi-Serve**。repo 的 sarathi/core/scheduler/ 提供 sarathi / orca / vllm / faster_transformer 四个 scheduler provider，可在同一 workload 下对照复现 §3 权衡（[Agrawal et al., 2024](../references.md)，[A]）；benchmark README 逐字引 arXiv:2308.16369，暴露 `chunk_size` 与 `enable_dynamic_chunking_schedule` 配置——后者是 Sarathi 的动态 chunking 开关（机制见 §6）。Sarathi 的 [A] 等级依据即此 repo 的 sarathi scheduler provider：原文无自链 repo，也不存在独立的 microsoft/sarathi repo（references 条目注）。

配置面对照（DeepInfer 一并列入）：

| 系统 | chunk 配置 | 预算语义 | 动态机制 |
| --- | --- | --- | --- |
| vLLM | max_num_batched_tokens / long_prefill_token_threshold（+adaptive） | 每 iteration token 预算，prefill + decode 共用 | 阈值按预算 / 请求数自适应 |
| SGLang | chunked_prefill_size（runtime_context get_schedule()） | chunk token 帽 | dynamic_chunk_sizer.py（6.2） |
| Sarathi-Serve | chunk_size / enable_dynamic_chunking_schedule | chunk token 帽 | 动态 chunking schedule 开关 |
| DeepInfer | compute_quota_limit（ns） | 每步时间预算 + 32768 token 双约束 | 二次求解 + DP MIN（§4） |

读表：token 预算与时间预算是同一约束的两种参数化。token 预算固定工作量，步时随前缀 $c$ 浮动（式 (2)）；时间预算固定步时，token 数随 $c$ 浮动（式 (8)）。后者直接对准 TBT SLO——步时就是 TBT（命题 2），前者经由工作量间接控制。这是 §6 动态化的两条路线。

还有一个执行层后果要预告：混合批的 token 总数逐步变（chunk 余量、decode 进出），GEMM 的 M 维是动态 shape——整图 CUDA Graph 捕获困难。vLLM 的 CUDAGraphMode 提供 FULL_DECODE_ONLY（只给纯 decode 批捕整图）与 PIECEWISE 选项，SGLang 的 piecewise 编译把动态 shape 的 attention 排除在图外（机制 3.4 已交付，CompilationMode / CUDAGraphMode 新名口径见 3.4）。混合批的动态 shape 是这类设计的动因之一：调度的灵活性与图捕获的静态性天然冲突，分段捕获是折中。

## 6. 动态 chunk size：从静态区间到运行时求解

静态 $c_{\mathrm{chunk}}$ 为什么次优？因为每 token 的时间成本随前缀 $c$ 涨：参数项 $2N_{\mathrm{act}}/F \approx 15.2$ μs/token 不变，前缀 attention 项 $4Ld_{\mathrm{model}} c/F$ 随 $c$ 线性涨—— $c$ = 32768 时达 17.4 μs/token，合计 32.5 μs，是首 chunk 的 2.1 倍。固定 $c_{\mathrm{chunk}}$ = 2048 的读数（式 (2)，章脊口径）：首 iteration 43.4 ms， $c$ = 24576 时 60.9 ms，末 chunk（ $c$ = 30720）67.6 ms——开头合规的 chunk size 在 prompt 中段漂出 50 ms SLO。可行区间的上界本身是 $c$ 的函数： $c_{\mathrm{hi}}$ 在 $c$ = 0 时 2485， $c$ = 24576 时降到 1673（式 (2) 数值解）。

动态化的原则：固定目标步时 $\tau_{\mathrm{target}}$ ，把 chunk token 数解成 $c$ 的函数。线性化读数（忽略 $q^{2}$ 自 attention 项与 decode 项；口径： $q \ll c$ 且 $B_D$ 小）：

$$
\frac{q(c)}{q(0)} \approx \frac{2N_{\mathrm{act}}}{2N_{\mathrm{act}} + 4 L d_{\mathrm{model}} c} \qquad (9)
$$

$c$ = 32768 代入得比值 ≈ 0.47——chunk 应缩到约一半。DeepInfer compute-quota（§4）是这个原则的精确版：不线性化、解完整二次式，时间预算直接对准 TBT。vLLM 的 adaptive threshold 是另一条路线：按 token 预算与请求数调 chunking 阈值（config/scheduler.py L112），间接适应负载；Sarathi 的 `enable_dynamic_chunking_schedule` 是研究侧的动态 chunking 开关（sarathi-serve benchmark README）。三条路线的共性：chunk size 是运行时变量，不是编译期常量。

> **提点**：quota 是策略（给多少时间），二次求解是机制（装多少 token）。两者分离的好处：调 SLO 只动 quota 数值，求解器与 kernel 栈不动——策略变更不触发机制重验证。

三个交互。prefix caching（2.2）：命中缩减有效 $c$ —— $\mathrm{cost}(q, c)$ 的 $c$ 是「已算前缀」，命中部分不必重算，同预算解出更大 chunk。代入 §4 教学系数：前缀 8192 命中一半（有效 $c$ = 4096），50 ms 预算的解从 1260 涨到 1412（+12%）——前缀复用直接放大每步的 prefill 进度（命中率账归 2.2）。投机解码（4.4）：verify 是 prefill 形态（ $\gamma + 1$ 个 token 一次前向），其成本与 chunking 账同形，draft 长度选择与混合批组成交互（4.4 / 7.2）。PD 分离（8.2）：可行区间为空时（§3 的 20 ms SLO 算例），把 prefill 与 decode 拆到不同实例从根上消除干扰，代价是 KV 传输——Sarathi-Serve 与 DistServe 是同一权衡谱的两端（8.2 对照）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 整条 prefill 独占 iteration → TBT 尖刺 = prefill 步时；8B @32768 为 781 ms vs 15.1 ms（52×） | 1.2 式 (1)、1.1 命题 1 | §1、练习 1 |
| 混合批 = prefill chunk + 全部 decode；attention 逐序列、GEMM 合批（selective batching） | §2 shape 走读 | §2、4.1 |
| piggyback 省独立权重读 ≈ $W/\mathrm{BW} - 2N_{\mathrm{act}}B_D/F$（8B $B_D$ = 64 时 3.8 ms）； $B_D > 317$ 转负，延迟收益仍在 | §2 边际账 | §2、练习 3 |
| chunk attention FLOPs = $4Ld_{\mathrm{model}}(c_{\mathrm{chunk}} c + c_{\mathrm{chunk}}(c_{\mathrm{chunk}}+1)/2)$；与 1.2 式 (1) 同口径，无双重折扣 | 式 (1)、坑注 | §3、练习 4 |
| chunking 下 attention FLOPs 精确守恒（任意划分）；效率损失在 KV 重读与开销，不在计算 | 命题 1、式 (3) | §3、练习 2 |
| 前缀 KV 重读 $\approx \kappa L T^{2}/(2c_{\mathrm{chunk}})$；attention 强度 $4c_{\mathrm{chunk}}d_{\mathrm{model}}/\kappa$，compute-bound 硬下界 ≈ 74 token | 式 (4)(5) | §3、练习 3 |
| stall-free → TBT = iteration 步时；SLO 给 chunk 上界 $c_{\mathrm{hi}}$（50 ms → 2485），与效率下界合成可行区间；区间空 → PD 分离 / 抢占 | 命题 2、式 (2)(6)、权衡表 | §3、练习 2、3 |
| DeepInfer：compute_quota_limit ns + 拟合二次 cost(q,c)，解正根取最大 q ≤ 32768；DP all-reduce MIN 缩预算 | 式 (7)(8)，doc L143 官方披露 | §4、练习 4 |
| 同 50 ms 预算：c = 8192 → q = 1260，c = 65536 → q = 499；quota 固定、q 随 c 自适应 | §4 算例 | §4、§6 |
| vLLM token 预算 vs DeepInfer 时间预算 = 同一约束两种参数化；SGLang chunked_prefill_size 在 runtime_context（非 server_args） | §5 对照表、坑注 | §5 |
| 静态 chunk 随 c 漂出 SLO（2048：43.4 → 67.6 ms）；动态化 = 固定步时解 q(c)，比值 ≈ 0.47 @c = 32768 | 式 (9) | §6、练习 3 |

## 练习

1.（计算 · [10]）stall 尖刺（8B 章脊口径）：(a) 用 1.2 式 (1) 算 $T$ = 32768 的 prefill FLOPs 与峰值下界时间；(b) 对比 $B_D$ = 64、 $\bar{T}$ = 4096 的 decode 步时 15.1 ms，求尖刺倍数；(c) 先预测 MFU = 0.5 时尖刺倍数的变化方向再计算（提示：decode 步时是带宽下界，不随 MFU 变）。校验：(a) 与 §1 的 781 ms 一致；(b) ≈ 52×；(c) 倍数应翻倍。
2.（推导 · [20]）命题 2 的形式化证明与边界：(a) 对 iteration 序号归纳，按基例 / 保持 / 终止三段写出「stall-free 调度下任一 decode 序列的 TBT 等于其所在 iteration 步时」的证明；(b) 边界反例：若第 $k$ 个 iteration 漏掉某条 decode 序列（其余 iteration 都含），该序列在第 $k$ 段的 TBT 是多少？(c) 由此说明命题 2 的「每个 iteration 都包含全部」前提为何不可弱化。校验：(b) 的答案应等于相邻两个 iteration 步时之和（命题 2 等式的特例叠加）。
3.（计算 · [30]）chunk size 权衡（ $T$ = 32768， $\tau_{\mathrm{oh}}$ = 1 ms）：(a) 纯 prefill（ $B_D$ = 0）口径，对 $c_{\mathrm{chunk}} \in \lbrace 64, 512, 2048 \rbrace$ 先判定 attention regime（式 (5)），再算 prefill 总时长（GEMM 项 + 逐 chunk attention 相 max + $m \tau_{\mathrm{oh}}$ ）；(b) 混合批（ $B_D$ = 64、 $\bar{T}$ = 4096、 $c$ = 0）口径，用式 (6) 估 $\tau_{\mathrm{SLO}}$ = 50 ms 的 $c_{\mathrm{hi}}$ ，再数值解式 (2) 精确值；(c) 固定 $c_{\mathrm{chunk}}$ = 2048，算 $c$ = 24576 的 iteration 步时，判定是否漂出 50 ms SLO。校验：(a) 与 §3 权衡表 1338 / 845 / 797 ms 一致（容差 1%）；(b) 估值 ≈ 2489、精确解 2485；(c) 与 §6 的 60.9 ms 一致。
4.（推导 + 计算 · [30]）compute-quota 二次模型：(a) 由式 (1) 与参数项 $2N_{\mathrm{act}} q$ 出发，除以 $F_{\mathrm{eff}}$ 推出式 (7)，给出 $u, v_0, v_1$ 的表达式；(b) 推导式 (8) 的正根（判别式 → 求根公式 → 舍负根）；(c) 用教学系数 $u$ = 0.53 ns、 $v_0$ = 30300 ns、 $v_1$ = 1.06 ns，算 quota = 50 ms 时 $c$ = 8192 与 $c$ = 65536 的 $q$ （向下取整、cap 32768），并回代验证 cost ≈ quota；(d) 两 DP rank 预算 50 / 40 ms，MIN 后组预算多少？ $c$ = 8192 的 rank 解出 $q$ 多少？校验：(c) 与 §4 的 1260 / 499 一致，回代残差 < 1%（取整损失）；(d) 40 ms、1012；(a) 的 $u$ 应恰为 $v_1/2$ （causal 1/2 的代数痕迹）。

## 参考

- [Agrawal et al., 2023](../references.md)：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills，arXiv:2308.16369，[A]（key: sarathi；官方实现 = microsoft/sarathi-serve 的 sarathi scheduler provider，原文无自链 repo）——chunked-prefills + decode-maximal batching、decode batch 上限三决定因素、统一工作单元减 PP bubble
- [Agrawal et al., 2024](../references.md)：Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve，arXiv:2403.02310，[A]（key: sarathi-serve；OSDI'24，repo README 自链）——stall-free batching、throughput-latency 权衡、chunk size 调优；repo 含 orca / vllm / faster_transformer 对照 scheduler 与 chunk_size / enable_dynamic_chunking_schedule 配置
- [Yu et al., 2022](../references.md)：Orca: A Distributed Serving System for Transformer-Based Generative Models，USENIX OSDI'22，无 arXiv，[B]（key: orca）——iteration-level scheduling 与 selective batching（attention 逐序列、其余 token-wise 合批），混合批组成的机制源头
- [vLLM](../references.md)：vllm-project/vllm repo，[A]（key: vllm-repo）——config/scheduler.py max_num_batched_tokens（L47）/ long_prefill_token_threshold（L104）/ long_prefill_token_threshold_adaptive（L112），2026-11-23 tarball 核实；模块全量走读归 5.2
- [SGLang](../references.md)：sgl-project/sglang repo，[A]（key: sglang-repo）——chunked_prefill_size 经 runtime_context.py get_schedule()（L1351）读取、scheduler.init_chunked_prefill（L1468）、scheduler_components/dynamic_chunk_sizer.py，2026-11-23 tarball 核实；函数级语义归 6.2
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——compute-quota 机制（L143 逐字：compute_quota_limit ns / cost(q,c) 二次 / 32768 token 帽 / DP all-reduce MIN）、四长度档与 quota cap（L137）、按剩余 quota pull（L139），2026-11-23 核实；生产负载均衡账归 7.4
