# 6.4 注意力变体 kernel（Attention Variant Kernels）

> MLA 吸收后 score 内维从 192 涨到 576，V4.1 又变成 512——这两个数各是哪两截拼的，混了会怎样？64×512 的输出矩阵占掉半个寄存器堆，FA3 的 pingpong 直接放不下，FlashMLA 用什么调度顶上？decode 每步 query 只有 1 行，split-KV 切出来的 partial 状态在 GPU 上怎么一步步拼回完整输出？本章 recap models 4.1 的 absorption 等价与 inference 3.1 的变体策略（不重推），交付 GPU 侧片段级走读：FlashMLA 双版本锚（V3 Hopper seesaw + V4.1 Blackwell FP8/FP4）、GQA/MQA head 共享与 PackGQA（命题 1）、paged block table gather、DSA indexer 的 top-k gather、FlashInfer split-KV reduction kernel 与 scheduler，收尾给变体选择决策表 | 前置：6.3（FA 走读、命题 1 tile 预算、命题 2 流量账与 $\eta$ 修正）、models 4.1 §4（MLA absorption recap 源）、inference 3.1（split-KV/cascade/FlashMLA 几何/DSA 策略 recap 源）、inference 2.1（块表式 (3)）、5.1（warp/block 归约）、4.1（coalescing）、6.1（FP8/FP4 格式账）；下游 11.2（实测 drill-down） | 锚点：FlashMLA（[DeepSeek-AI, FlashMLA](../references.md)，教学参照 + 官方实现，双版本锚 main 395cb3afbb33 + ba89a3466e9470ad08ab39738d4e7bb66989e1e7）、FlashInfer（[Ye et al., 2025](../references.md)，arXiv:2501.01005，[A]，0.7.6 commit 5228ea6be8ec）、vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、DeepGEMM（[DeepSeek-AI, DeepGEMM](../references.md)，教学参照 + 官方实现）——代码锚 2026-11-25 fresh tarball / contents API 一手核实

## 1. 几何决定 kernel：本章口径与代码 scope

6.3 的命题 1 留了一个预告： $d_{\mathrm{head}}$ = 512 时输出累加器 64×512 fp32 = 32768 寄存器，恰为 $R_{\mathrm{file}}$ = 65536 的一半——FA3 pingpong 需要两份输出交替，预算上当场死亡。变体 kernel 不是「换个模板参数」，是几何把调度逼到另一条路上。本章回答四个问题：MLA 两代几何（576 与 512）各怎么拆，FlashMLA 两版代码各锚在哪（§2）？GQA 的 head 共享在 kernel 里省的是哪一笔读（命题 1，§3）？paged 与 DSA 的 gather 访存长什么样（§4/§5）？split-KV 的 reduction kernel 逐指令做什么（§6）？

**本章口径**（章脊锁定，多 SKU 对比章，换口径随段声明）：

| 口径 | 几何 | 硬件与用途 | 来源 |
| --- | --- | --- | --- |
| V3 MLA（§2 走读） | $H_q$ = 128、 $d_{\mathrm{latent}}$ = 512、 $d_{\mathrm{rope}}$ = 64、bf16 | H800 SXM5（deep-dive 原始 benchmark 口径：3.35 TB/s、throttled 峰值约 865 TFlops） | [DeepSeek-AI, 2024](../references.md)，[A]；FlashMLA ba89a34 |
| V4.1 MLA（§2 对照） | 共享 KV 向量 512 元素 = 448 latent + 64 rope、FP8/FP4 KV | B200（CUDA 13.3，README 报告 1460/950 TFlops） | [DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]；FlashMLA main |
| 8B GQA（§3/§6 数值） | $T$ = 4096、 $d_{\mathrm{head}}$ = 128、 $H_q$ = 32、 $H_{kv}$ = 8、 $g$ = 4、bf16 | H100 SXM（沿 6.3 章脊，3352 GB/s、ridge 295） | 6.3 章脊表 |

符号声明（spec §4.3）： $T$ = KV 序列长， $B$ = batch， $s_q$ = 每请求 query token 数（decode 无投机时 = 1，沿 inference 3.1）； $d_{\mathrm{latent}}/d_{\mathrm{rope}}$ = MLA latent/rope 维（models $d_c/d_R$ 翻译）， $d_{\mathrm{qk}} = d_{\mathrm{latent}} + d_{\mathrm{rope}}$ = score 内维， $d_v$ = value 侧维度（V3 kernel 口径 512，§2）； $b$ = page size（token 数，沿 inference 2.1）； $K_{\mathrm{top}}$ = DSA 选择行数（inference 3.1 的 $K$ 翻译，避与 GEMM 维冲突）； $S$ = split 数， $g = H_q/H_{kv}$； $\kappa$ = 每 token 每层 KV 字节（inference 1.2 式 (3)）； $R_{\mathrm{file}}$ = 65536； $\mathrm{bytes}$ 不占字母； $I$ = 算术强度。B/KB/MB 单位豁免。

**recap 边界**（spec §4.2，不重推）：MLA 低秩构造、RoPE×低秩冲突、W_UK/W_UV absorption 等价（models 4.1 §4 式 (11)-(14)）只 recap 关键步；split-KV/cascade 归约正确性（inference 3.1 命题 1/2）、DSA 选择链与期望界（inference 3.1 §7）、FlashMLA 吸收形几何与 regime 翻转（inference 3.1 §5）只 recap 结论与指针；块表地址翻译（inference 2.1 式 (3)）recap。**代码 scope：片段级走读**（spec §4.2）——FlashMLA 取 V4.1 mainloop 片段锚 + ba89a34 Hopper absorbed 机制片段（config/traits/splitkv/deep-dive），FlashInfer 取 MergeState/scheduler/plan-run 片段；FA 模块全量走读归 6.3，本章不重复。**与 6.3 的 split-KV 分工**：6.3 = 流量账修正（ $\eta$），本章 = reduction kernel 指令走读。

**版本锚**：FlashMLA **双锚**——main commit 395cb3afbb33（2026-10-27，仅 SM100/SM103，V4.1）+ commit ba89a3466e9470ad08ab39738d4e7bb66989e1e7（V3 Hopper，main README L3 链接全 SHA）；main 于 2026.09.30 移除 Hopper 与 V3/V3.2/V4.0 支持并改 FP8/FP4 KV 格式（README L3 breaking change 逐字）。FlashInfer **0.7.6**（commit 5228ea6be8ec，contents API 核实——repo 372 MB，tarball 截断规避）。

> **坑**：把 FlashMLA 当前 main 当 Hopper MLA decode 读。触发条件：2026.09.30 之后 clone main 找 V3 的 bf16 kernel；现象：`csrc/cuda_kernels/` 下只有 sm100/smxx，sm90 目录整个不存在，按旧文档对代码全部落空；规避：Hopper absorbed 机制锚 ba89a34（本章 §2 口径），V4.1 内容锚 main，两版勿混。

## 2. MLA kernel：两代几何与 FlashMLA 双锚走读

吸收之后 score 打在什么对象上？先 recap 吸收形等价（models 4.1 §4.3-4.4，不重推）：K 侧把上投影吸进 query、V 侧把加权和提到上投影之前：

$$
s_{t,s,h} = \frac{\left\langle \tilde{q}_{t,h}, c_s^{KV} \right\rangle + \left\langle q_{t,h}^{R}, k_s^{R} \right\rangle}{\sqrt{d_h + d_{\mathrm{rope}}}}, \qquad o_t = \sum_{h} \widehat{W}_h \sum_s p_{t,s,h} \thinspace c_s^{KV} \qquad (1)
$$

**符号与翻译注**：式 (1) 即 models 4.1 式 (12)(13)(14) 的合并形（$`\tilde{W}_h = (W_h^{UQ})^{\top} W_h^{UK}`$、$`\widehat{W}_h = W_h^{O} W_h^{UV}`$ 离线预计算）； $c_s^{KV}$ = 缓存 latent， $q^R/k^R$ = decoupled RoPE 通道（不吸收）。等价性证明（结合律 + 线性性）见 models 4.1，serving 含义见 inference 3.1 命题 3，本章不复述。kernel 侧读法：**score 直接打在缓存 latent 上，K/V 同源**——deep-dive 逐字「in MLA, K and V are the same with different names」，QK 的 B 操作数与 PV 的 B 操作数是同一份数据的两个视图，KV 读一遍服务两个 GEMM。

**两代几何，两个数，勿混**（本章硬口径）：

| 代际 | $d_{\mathrm{latent}}$ | $d_{\mathrm{rope}}$ | $d_{\mathrm{qk}}$ | KV 存储 | 代码锚 |
| --- | --- | --- | --- | --- | --- |
| V3/V3.1/V3.2 | 512 | 64 | **576** | bf16，1152 B/token | ba89a34 `config.h` L7-11：HEAD_DIM_K=576 / HEAD_DIM_V=512 |
| V4/V4.1 | 448 | 64 | **512** | FP8 528 B / FP4 288 B（全量化，无 bf16 部分） | main README L140-143：head_dim=512 |

V3 的 576 = 512 latent + 64 rope（config 字段 kv_lora_rank=512、qk_rope_head_dim=64，[DeepSeek-AI, 2024](../references.md)，[A]）；V4.1 的 512 = 448 latent + 64 rope（config 无 kv_lora_rank 字段、引擎导出 448，[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]，模型侧形式化归 models）。value 侧维度口径：V3 的 $d_v$ = 512（HEAD_DIM_V，config.h L11，seesaw 的 64×512 输出由此而来）；ba89a34 README 脚注逐字两代 MQA 模式均「head_dim_v = 512」（kernel 口径）——V4.1 的 448 latent 是模型侧切分，kernel V 维与 latent 的映射关系归 models 形式化，本章不推定。ba89a34 的 `params.h` L6-9 把两代写成枚举：V32（d_qk=576）/ V4（d_qk=512）/ V41（d_qk=512，RoPE fp8，quant tile 32）/ V41_FP4（fp4 e2m1，quant tile 16，e4m3 scales）——**576 是 V3 几何，512 是 V4.1 几何**，把 512+64=576 写成 V4.1 是两代混淆的典型错。

> **坑**：V4.1 几何写 512+64=576。触发条件：沿用 V3 的 kv_lora_rank=512 推 V4.1；现象：kernel 几何、KV 字节账、TMA 份数（576/64=9 对 512/64=8）全错；规避：V4.1 共享 KV 向量共 512 元素（448 latent + 64 rope），64 维 rope 已含在 512 内且一并量化（README 逐字「The 64 RoPE dimensions are quantized as well, so there is no bfloat16 part」）。

**V3 Hopper absorbed kernel 走读**（ba89a34，`csrc/kernels/sm90/decode/dense/`）。几何：BLOCK_SIZE_M=64、PAGE_BLOCK_SIZE=64（config.h L7-8）——Q tile 64 行（decode $s_q$ = 1 时 M 维铺 head， $H_q$ = 128 分两块），KV 按 64-token 页读。寄存器预算（6.3 命题 1 的 MLA 实例）：输出 64×512 fp32 = 32768 寄存器，deep-dive L21 逐字「Each 64×512 output matrix occupies 32,768 32-bit registers. With only 65,536 32-bit registers per SM, we can store only one output matrix per SM」——**单输出矩阵/SM，双输出 pingpong 不可行**。解法分两步：

其一，**输出竖切**。`traits.h` 的 GMMA op selector 用 `Shape<BLOCK_M, HEAD_DIM_V/2, PAGE>`（L39/44）——O 切成 O_L/O_R 各 64×256 = 16384 寄存器，两个 warpgroup（NUM_THREADS=256，L24）各持一半；V 同步切成 V0L/V0R/V1L/V1R。其二，**seesaw 调度**：每轮取两个 KV block（K0/K1），两个 WG 各算一份 QK（p0/p1），共享一个 running max，交错更新本半区并交叉更新对方半区（deep-dive 步骤 0-11，L27-38）；deep-dive L42 逐字「This schedule can be viewed as a 'ping-pong' variant using one output matrix—we call it 'seesaw' scheduling. It's mathematically equivalent to FlashAttention's online softmax algorithm」。半区更新的 rescale 因子是「自上次更新以来全部 max 位移的累积因子」（O_R 的因子是两个 scale 之积）——正确性就是 online softmax 不变量按半区维护（models 4.1 式 (16)，recap 不重证）：两个半区的贡献最终都落在同一全局 max 基准上。与 6.3 §7 的三代对照接上：FA3 双输出 pingpong（寄存器够）、FlashMLA 单输出 seesaw（寄存器减半）、FA4 TMEM（约束移除）。

```
seesaw 调度（ba89a34 deep-dive 步骤 0-11，L27-38；[0]/[1] = warpgroup 编号）
──────────────────────────────────────────────────────────────────────
共享：running max m；输出 O_L（64×256，WG0 regs）| O_R（64×256，WG1 regs）
装载：K0、K1 各 9 份 TMA（64×576，576/64=9）逐份落地即启动 GEMM；
      V0/V1 = 同一份数据的 latent 512 列视图（K/V 同源，不另装载），
      V0L/V0R/V1L/V1R 是视图竖切（各 64×256）

[0] p0 = q·K0ᵀ/qk_scale        [1] p1 = q·K1ᵀ/qk_scale   ← 两 WG 各算一份 QK
[0] mp0 = max(p0)；m_new0 = max(m, mp0)；scale0；m ← m_new0
[0] p0 ← exp(p0 − m_new0)                               ← softmax
[0] O_L ← O_L·rescale(scale0) + p0·V0L                  ← 本半区更新
[1] mp1 = max(p1)；m_new1 = max(m, mp1)；scale1；m ← m_new1
[1] p1 ← exp(p1 − m_new1)
[1] O_R ← O_R·rescale(scale0·scale1) + p1·V1R           ← 本半区，累积因子
[0] p0 ← p0·rescale(scale1)                             ← 跨 WG 准备：p0 换基准
[1] O_R ← O_R + p0·V0R                                  ← 跨 WG 更新：WG1 用 WG0 的 p0
[0] O_L ← O_L·rescale(scale1) + p1·V1L                  ← 跨 WG 更新：WG0 用 WG1 的 p1

rescale = 换基准因子（≤1，exp(m_old − m_new) 方向）；deep-dive 原文 scale
记号为其倒数（inference 3.1 §5 译注口径）。重叠：WG0 做 softmax/rescale
（CUDA/SFU）时 WG1 占 TC，反之亦然；TMA 在数据不再需要后立即发射。
```

**fine-grained TMA-GEMM 流水**：576 = 9×64，一个 64×576 K block 拆 9 次 64×64 TMA copy——`traits.h` L82-83 的 `TMABarrier barriers_K0[HEAD_DIM_K/64]`（= 9 个 barrier，K1 同），每份落地即启动对应 GEMM 段；deep-dive L54 逐字「For a 64×576 K block, we launch 9 TMA copies (each moving a 64×64 block). GEMM operations begin as soon as each TMA copy completes」。对照 FA3 的整 tile TMA（6.3 §5：一个 K tile 一条 op、32768 B）：MLA 的 K block 72 KB（64×576×2 B），整块等完再算的首字节延迟太大，切 9 份把「等整块」变「等首份」——4.4 延迟隐藏条件在 head 维上的细化。cache hint 取 **EVICT_FIRST**（`splitkv_mla.cuh` L47）：decode splitkv 下每个 KV block 只被一个 CTA 消费、无跨 CTA 复用，流式数据标「先逐出」保护 L2 里的其他数据——与 FA3 的 EVICT_LAST（KV 有跨 CTA 重读，6.3 §5）方向相反，**hint 跟着复用结构走，不是固定配方**。

指令级细节一枚：`get_AorC_row_idx`（splitkv_mla.cuh L20-23）把 fragment 局部行号映回实际行：`row_idx = (idx/32)*16 + local*8 + (idx%32/4)`——wgmma 的 A/C fragment 里每线程持两行、行号非连续（PTX wgmma 布局），而 rescale 与 mask 都按实际行操作，这个反解是「数学按行、硬件按 fragment」的接缝（6.3 §4 FA2 mask 行索引公式同构）。paged KV：块表 + cache_seqlens 输入，每 token 576 元素连续存放，kernel 按 64×576 tile 读取——地址翻译与 inference 2.1 式 (3) 同形（§4）。性能（deep-dive/README 口径）：至多 660 TFlops（H800，throttled 峰值 865 的 76%）、memory-bound 设置 3 TB/s。regime 背景 recap（inference 3.1 §5，不重推）： $I_{\mathrm{MLA}} = H_q s_q (d_{\mathrm{qk}} + d_v)/d_{\mathrm{qk}}$，V3 代入 128×1088/576 = 241.8 FLOP/byte，是 throttled ridge 258.2 的 94%——compute-bound 边缘，kernel 按 compute-bound 设计，所以杠杆在 TC 利用率（seesaw）而非带宽。

**V4.1 Blackwell 对照**（main 395cb3af）。仅 SM100/SM103（README L79 逐字「NVIDIA SM100 / SM103 GPU」）；kernel 形态换成 CUTLASS 风格 warp-specialized collective（`csrc/cuda_kernels/sm100/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp`，6.3 §5 同构）+ token-level sparse prefill/decode + fused Q-norm/Q-RoPE/attn/O-RoPE/cast kernel（README L7；fused kernel 报告 1460 TFlops prefill / 950 TFlops decode，B200 CUDA 13.3，L41）。KV 格式账（README L140-143，6.1 格式账的实例化）：

$$
\text{FP8: } 528 = 512 + 512/32 \qquad \text{FP4: } 288 = 512/2 + 512/16 \qquad (2)
$$

**符号**：FP8 形每 token 512 个 float8_e4m3 值（512 B）+ 每 32 值一个 float8_e8m0 scale（16 B）；FP4 形 512 个 e2m1 值两值一字节（256 B）+ 每 16 值一个 float8_e4m3 scale（32 B）。代入读法：528 B 对 V3 bf16 1152 B 省 2.18 倍，288 B 省 4 倍——**MLA 的 57× 读取削减（对展开形，models 4.1 recap）之上再叠量化倍数**。scale 粒度 per-32 E8M0 是 MX 族口径、per-16 是 NVFP4 粒度口径（位布局与 scale 编码账归 6.1；FlashMLA 的 per-16 scale 用 e4m3，README 口径，与 NVFP4 规范的 UE4M3 编码细节以 6.1 为准）。decode 只支持 FP8/FP4 KV、不支持 bf16（README L138 逐字）——量化从可选变必选，这是 V4.1 kernel 与 V3 的又一结构差。

## 3. GQA/MQA kernel：head 共享省的是哪一笔读

GQA 的省账 models 4.1 §3 已交付（cache 按 $H_{kv}$ 份存、广播在视图层不产生新字节）——kernel 侧的问题更细：不打包时，同一份 KV 的 HBM 流量要付几遍？答案是 $g$ 遍—— $g$ 个 query head 的 CTA 各读一遍。FA2 的 head 映射逐字可查：`gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), ...)`（flash_fwd_kernel.h L157）——query head bidh 整除 $g$ 得 KV head， $g$ 个相邻 bidh 的 CTA 读同一段 K/V。省回这笔读有两条路（6.3 §6 的 $\sigma$ 机制，此处走读代码实体）：

**命题 1**（GQA head 共享带宽节省）。设 $g$ 个 query head 共享一份 KV，完美共享口径下 KV 的 HBM 流量为无共享的 $1/g$；decode（ $s_q$ = 1、无 causal 结构）下 PackGQA 精确达到该界。**证明**（路线：计数 + 构造，两步）。计数：无共享时 $H_q$ 个 CTA 列各读全量 KV，流量 $H_q \cdot T \cdot 2 d_{\mathrm{head}} \cdot \mathrm{bytes}$；完美共享下每份 KV 只从 HBM 读一次，流量 $H_{kv} \cdot T \cdot 2 d_{\mathrm{head}} \cdot \mathrm{bytes}$，比值 $H_{kv}/H_q = 1/g$。构造（decode PackGQA）： $g$ 个 head 的 query（各 1 行）打包成 $g$ 行 Q tile，一个 CTA 读一次 KV 服务 $g$ 行——KV 装载次数从 $H_q$ 降到 $H_{kv}$，界精确成立。causal prefill 的打包计数与 $\sigma$ 的成立条件留作练习 3。∎

**PackGQA 代码实体**（FA3，hopper/）：模板参数 `PackGQA_`（mainloop L33）启用后 `Use_TMA_Q = !PackGQA`（L56）——Q 改 cp.async 装载（打包后 Q 的行来自不同 head，物理不连续，TMA 的连续 box 假设失效）；tile scheduler 把 M 维长度乘 $g$（`tile_scheduler.hpp` L108/L605：`seqlen *= qhead_per_khead`），grid 的 head 维从 $H_q$ 缩到 $H_{kv}$（launch template L148-152：`qhead_per_khead = ceil_div(params.h, params.h_k)`）。代入 6.3 章脊口径的 decode（ $T$ = 4096）：不打包 32 个 CTA 各读 2.10 MB KV，合 67.1 MB；打包后 8 个 CTA 各读一遍，合 16.8 MB——流量恰为 1/4，界精确成立。MQA 是 $g = H_q$ 的极端（ $H_{kv}$ = 1），同一机制不再赘述。

强度对照（decode，bf16，recap inference 1.2 口径）：GQA attention 的 FLOPs $4 T H_q d_{\mathrm{head}}$、KV 读 $2 T H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}$，故

$$
I_{\mathrm{GQA}} = \frac{4 T H_q d_{\mathrm{head}}}{2 T H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}} = \frac{2g}{\mathrm{bytes}} \qquad (3)
$$

代入 $g$ = 4、bf16： $I$ = 4 FLOP/byte，远低于 ridge 295——memory-bound，优化杠杆是带宽（共享、量化、split-KV 铺满 SM）。对照 §2 的 MLA： $I$ = 241.8 贴 ridge——compute-bound 边缘，杠杆是 TC 利用率。**同一笔「head 共享」，GQA 共享 4 个头把强度抬到 4，MLA 共享 128 个头（宽头 MQA）再叠 K/V 同源把强度抬到 242**——模型几何决定 kernel 的优化方向（inference 3.1 §5 提点的 kernel 侧兑现）。

## 4. paged attention kernel：block table gather

块表在 inference 2.1 是内存管理语义，到 kernel 侧是一个 gather 的 indices 数组： $\mathrm{slot}(t) = \mathrm{bt}[\lfloor t/b \rfloor] \cdot b + r(t)$（ $r(t)$ = $t$ 除以 $b$ 的余数；inference 2.1 式 (3) recap，不重推）。逻辑连续不变（causal mask 同形），物理散落经块表翻译。FA2 splitkv 路径的翻译逐字可查（flash_fwd_kernel.h L597-598）：`block_table_idx = (n_block_max-1)*kBlockN / page_block_size`、`block_table_offset = ... - block_table_idx*page_block_size`——先算 KV 块落在哪一页、页内偏移多少，再取页地址。

访存模式按 4.1 coalescing 判定拆两层：**页内**连续（ $b$ 个 token × $d_{\mathrm{head}}$ 元素行主序），16 B 向量化装载满 sector 利用；**页间**跳转（块表给出的页地址任意散落），每次跳转是新的一段连续流——gather 的代价不在页内而在页表读取与跳转粒度。 $b$ = 64、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16 时一页 128 KiB（NHD 全头口径），单 head 页内 slice 16 KiB 连续（HND layout，inference 3.1 §4 recap：decode 逐 head 读，HND 合并访存占优）。块表本身很小（ $T/b$ 个 int32， $T$ = 4096、 $b$ = 64 时 256 B），驻 L2 无压力。

TMA 与 paged 的兼容分两路（FA3，6.3 §5 的补充）：页尺寸对齐时走 **TMA per-page**——`paged_kv_manager.get_indices_for_K_TMA()` 给出页号，每页一条 TMA copy（mainloop L752-754），tensor-map 的 box 落在单页内；页尺寸任意（PagedKVNonTMA）时退 **cp.async gather**（L757 `load_K<Seqlenk_mask>`），pipeline 换 `PipelineAsync`（L284-286 条件分支，无 TMA 的 tx-count 语义，改线程 arrive 计数）。代价：FP8 headdim 大于 192 的 paged 配置禁用 IntraWGOverlap（tile_size.h L51 注释逐字「PagedKV uses more registers so we disabled IntraWGOverlap」）——gather 的地址状态吃寄存器，6.3 命题 1 的预算在 paged 路径上更紧。

vLLM v1 的 backend 族（inference 3.1 §6 recap，接口级不重走读）：`vllm/v1/attention/backends/` 按 kernel 族 × 硬件 × KV 几何一个 backend，`flash_attn.py` 的 build 产出 block_table + scheduler_metadata + max_num_splits 传给 FA varlen 接口。GPU 侧增量一句：max_num_splits 三态（0 启发式 / CUDA Graph 固定上界 / batch-invariant = 1）对应 §6 scheduler 的切分决策——引擎接口与 kernel 决策是同一件事的两端。

## 5. DSA indexer kernel：选择链的 GPU 侧

稀疏化后主路只读 0.39% 的行，带宽瓶颈搬去了哪？先 recap 选择链（inference 3.1 §7，不重推）：indexer 对全历史打分（ReLU 加权内积、小维 $d_I$ = 128、头间共享）→ candidate block max 池化预选（选 $K_b$ 块，精确性条件 recap inference 3.1 命题 4）→ DeepSelect token top-k（ $K_{\mathrm{top}}$ = 512）→ 主路只 gather 选中行。GPU 侧增量三件：

其一，**打分 kernel 是 MQA logits GEMV（general matrix-vector multiply，矩阵-向量乘）**：indexer query 对全量 $T$ 行 indexer K 做加权内积，形态是 query 多头共享一份 indexer K 的窄 GEMM——DeepGEMM 的 mqa_logits kernel（lightning indexer 组件，[DeepSeek-AI, DeepGEMM](../references.md)，教学参照；GEMM mainloop 走读归 5.2/6.2）。其二，**top-k gather 的访存**：选中 indices（[B, s_q, K_top] int32，页地址已 bake，inference 3.1 §7 接口口径）驱动主路按行 gather——每行 512 元素（V4.1）连续、行间任意跳转，与 §4 paged gather 同模式但粒度更细（token 级对页级）； $K_{\mathrm{top}}$ = 512 行 × 528 B = 264 KiB/请求/层，对全量 $T$ = 131072 行的 0.39%。其三，**瓶颈搬家账**（recap inference 3.1 式 (10)，代入 V4.1 生产格式）：主路读 $K_{\mathrm{top}} \kappa L$，bf16 教学反事实 20 MiB/步，**FP8 528 B 生产口径 10.3 MiB/步**（512×528×40 层）；indexer 读全量 $`T d_I \cdot \mathrm{bytes}_I \cdot L`$，bf16 口径 1.25 GiB/步 = 主路的 64 倍（MXFP4 indexer 省 3.76 倍后仍 17 倍）——稀疏化把带宽瓶颈从主 KV 搬到 indexer K，indexer 精度成为新杠杆（格式账归 6.1，量化 kernel 归 6.2）。DeepSelect 的 scan/filter/compact 与期望界（inference 3.1 式 (9)）recap 不重推；其 PTX 级优化（比较/位操作、浮点加法代整数加法）属 5.1 reduction 原语的变体，本章只立接口。

## 6. decode split-KV：reduction kernel 指令走读（FlashInfer）

切出来的 partial 状态由哪个 kernel、哪些指令拼回去？先 recap 策略与正确性（inference 3.1 §2，不重推）：KV 切 $S$ 段、各段独立算 partial attention、产出 (O_i, LSE_i) 状态、段间用 LSE 加权归约（命题 1：结合交换、与全量逐点相等）。流量代价 $\eta$ 归 6.3 §6（本章口径 2.76%）。本节走读归约在 GPU 上的指令实现（FlashInfer 0.7.6，`include/flashinfer/attention/cascade.cuh`）。

**MergeStateKernel**（L45）：grid 按 (pos, head) 铺——`tx = threadIdx.x` 切 head_dim 向量、`ty = threadIdx.y` 映 head、`blockIdx.x` 映 token 位置。每 (pos, head) 的归约序列：

```cpp
// include/flashinfer/attention/cascade.cuh L53-72（机制摘录）
float s_max = max(s_a_val, s_b_val);                  // 合并基准
s_a_val = math::ptx_exp2(max(s_a_val - s_max, -math::inf));  // w_a = 2^{LSE_a - m}
s_b_val = math::ptx_exp2(max(s_b_val - s_max, -math::inf));  // w_b
float d_sum = s_a_val + s_b_val;
float a_scale = (d_sum > 0.f) ? s_a_val / d_sum : 0.f;       // 空态守卫
// vec_t<float, vec_size> 向量 load v_a/v_b →
//   v_merged[i] = a_scale * v_a[i] + b_scale * v_b[i]       // d_head 次 FMA
// vec 化 store；tx==0 写 s_merged = ptx_log2(d_sum) + s_max
```

三个指令级要点。其一，**底数是 2**：注释逐字（L42）「Both s_a and s_b are logsumexp values with base 2」——LSE 存 $\log_2$ 形，归约用 `ptx_exp2`/`ptx_log2`（MUFU.EX2/LG2 原生指令），与 FA2 的 exp2 trick 同构（6.3 §4）：换底数对齐 SFU，全链路无自然 exp。合并数学（inference 3.1 式 (3)）翻译成 base-2 形，即本章式 (4)：

$$
w_i = 2^{\mathrm{LSE}_i - m}, \quad O = \frac{w_i O_i + w_j O_j}{w_i + w_j}, \quad \mathrm{LSE} = \log_2(w_i + w_j) + m \qquad (4)
$$

**翻译注**：inference 3.1 式 (3) 为自然对数形，式 (4) 是同一恒等式的 base-2 实现形（$`m = \max(\mathrm{LSE}_i, \mathrm{LSE}_j)`$）；正确性证明归 inference 3.1 命题 1，本章不重证。数值核对沿 inference 3.1 §2 的例（ $x = (1, 2, 5, -1)$ 切两段）：base-2 形与自然对数形的合并输出同为 $O = (1.8853, 0.9783)$——换底不改数学对象。其二，**空态哨兵**：注释逐字（L56-57）「max() drops the NaN from merging two empty states ((-inf) - (-inf)); the d_sum guard keeps their merged output zero instead of 0/0 = NaN」——空 split 的 LSE = $-\infty$，直接相减得 NaN；实现先把差 clamp 到 $-\infty$（exp2 得 0）、再用 d_sum 守卫把 0/0 兜成 0，无特判分支（inference 3.1 §2 边界例的代码兑现）。其三，**向量化**：`vec_t<float, vec_size>` 的 cast_load/cast_store 按 16 B 粒度搬 partial O（4.1 向量化），归约本体是 $d_{\mathrm{head}}$ 次 FMA + 每 (pos, head) 3 次 SFU op（2 exp2 + 1 log2）——归约是带宽小头（ $\eta$），指令数也小头。

多状态归约（ $S$ 段）：`MergeStatesKernel`（L224）把两两合并铺成归约树，`warp_sync_state`（L147）/ `threadblock_sync_state`（L125）做 warp/block 级状态归约——5.1 的 warp shuffle / shared 两级归约结构，只是归约对象从标量换成 (O, LSE) 状态、算子换成 ⊕（结合交换保证任意树形等价，inference 3.1 命题 1 recap）。变长批用 `PersistentVariableLengthMergeStatesKernel`（L378）persistent 化。

**scheduler：切多少段是算出来的**（`scheduler.cuh` L74-98，`PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`）：二分搜索最小 pages/batch（段长下界），目标函数

$$
n_{\mathrm{new}} = \sum_{i} \lceil T_i / (b \cdot c) \rceil, \qquad n_{\mathrm{new}} \cdot H_{kv} \le G_{\max} \qquad (5)
$$

**符号**： $n_{\mathrm{new}}$ = 切分后的等效 batch（L86/96 逐字 `new_batch_size += ceil_div(elem, mid)`）， $T_i$ = 请求 $i$ 的 KV 长度（token）， $c$ = 段长（页为单位）， $b$ = page size（§1；代码按页计数，elem = num_pages， $T_i$ 为 $b$ 整倍时与 token 式等价）， $G_{\max}$ = `max_grid_size` = occupancy API 实测每 SM block 数 × SM 数（inference 3.1 §2 recap：并行度上限是 occupancy 事实非名义 SM 数）；代码的 grid y 维 gdy = $H_{kv}$。代入 6.3 章脊口径（ $B$ = 1、 $H_{kv}$ = 8、 $T$ = 4096、 $b$ = 64 即 64 页；教学口径设 $G_{\max}$ = 112）：二分得最小段长 $c$ = 5 页—— $\lceil 4096/(64 \times 5) \rceil = 13$ 段、 $13 \times 8 = 104 \le 112$，而 $c$ = 4 页给 $\lceil 4096/256 \rceil = 16$ 段、 $16 \times 8 = 128$ 超限—— $S$ = 13、grid 104 个 CTA。prefill 侧同构函数 `PrefillBinarySearchKVChunkSize`（L101-123）把 qo_chunk_size 也计入。plan/run 两阶段（`decode.py` BatchDecodeWithPagedKVCacheWrapper L1338、plan L2179、run L3237）：plan 在 CPU 侧吃序列长度、产出切分决策与工作队列写 workspace，run 启动 kernel 消费——plan 跨同 step 全部层复用（接口级 recap inference 3.1 §4）。cascade 两级（`cascade.py` MultiLevelCascadeAttentionWrapper L226、plan L401、run L537）：共享前缀状态 + 独立后缀状态各算一遍，MergeState 合并——前缀 KV 读取从 $B$ 次降到每 query tile 一次（inference 3.1 命题 2 recap），GPU 侧实体就是大行块 BSR（block-sparse row，inference 3.1 §4）+ 本节 merge kernel。JIT 按架构实例化（`jit/core.py` L186-202：sm90a/sm100a/sm103a 等 nvcc arch flags）——同一模板跨代编译，与 FA4 的 CuTeDSL 跨代是两条路线（6.3 §7）。

## 7. 变体选择决策表：几何 × batch × 硬件

怎么选 kernel？选 kernel 不是选名字，是选「几何决定的 bound + 预算决定的调度」。决策表（每行可判定，回指正文）：

| 场景 | 几何特征 | bound（口径） | kernel 族 | 关键旋钮 | 硬件 |
| --- | --- | --- | --- | --- | --- |
| dense prefill，长 seq | $T \gg d_{\mathrm{head}}$，Q 块铺满 SM | compute（6.3 §6： $I$ = 399） | FA2/FA3/FA4 | tile 预算（命题 1）、pingpong、L2 swizzle | sm80/sm90/sm100 |
| GQA decode，小 batch | $s_q$ = 1，grid = $B H_{kv}$ 铺不满 | memory（ $I$ = 2g/bytes = 4） | split-KV + PackGQA | $S$ 由式 (5) 反解、 $\eta$ 代价 | 通用 |
| GQA decode，大 batch | grid 已铺满 | memory | 不切（ $\eta$ 纯开销） | disable_split_kv | 通用 |
| MLA V3 decode | $d_{\mathrm{qk}}$ = 576， $H_q$ = 128 宽头 | compute 边缘（ $I$ = 241.8） | FlashMLA ba89a34 seesaw | 单输出竖切、9 TMA、EVICT_FIRST | Hopper |
| MLA V4.1 decode | 512 = 448+64，FP8/FP4 KV | compute（TC 利用率杠杆） | FlashMLA main sparse/fused | 528/288 B 格式、fused norm-RoPE | SM100/103 |
| DSA 稀疏 decode | $K_{\mathrm{top}}$ = 512 行 gather | indexer 带宽主导（64×主路） | indexer GEMV + DeepSelect + sparse MLA | indexer 精度（MXFP4）、 $K_b$ 预选 | 通用 |
| 共享前缀批 | 前缀（长 $T_p$）跨请求相同 | memory（前缀重复读） | cascade 两级 | 行块 $r_q$（query tile 行数）、merge ⊕ | 通用 |

两条读法。其一，**bound 由几何定，不由 kernel 名定**：同为 decode，GQA-4 的 $I$ = 4 与 MLA 的 $I$ = 242 差 60 倍，优化杠杆一个在带宽一个在 TC——先算强度再选 kernel（2.4 roofline 口径）。其二，**预算由几何 × 硬件定**： $d_{\mathrm{head}}$ = 128 在 Hopper 上 pingpong 可行、512 在 Hopper 上只能 seesaw、Blackwell TMEM 解除约束（6.3 §7 三解）——同一几何换硬件，调度换代。

## 8. 边界：本章不做什么

变体走读收口，剩下的归谁？本章交付 MLA 两代几何与 FlashMLA 双锚片段（§2）、GQA 共享命题与 PackGQA（§3）、paged gather（§4）、DSA GPU 侧三件（§5）、FlashInfer reduction/scheduler 走读（§6）、决策表（§7）；以下各有专章。FA 模块全量走读与 tile 预算/流量账归 6.3；MLA 模型侧形式化（V4.1 低秩构造）归 models；DSA 选择链算法与期望界、cascade 正确性归 inference 3.1/4.x；vLLM 引擎模块全量走读归 inference 5.3；格式位布局归 6.1；量化 kernel 归 6.2；实测 drill-down 归 11.2。代码 scope：片段级走读（FlashMLA 双锚 + FlashInfer merge/scheduler + FA paged/PackGQA 片段）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| MLA 吸收形等价 recap models 4.1（不重推）；K/V 同源，KV 读一遍服务两个 GEMM | 式 (1) | §2 |
| 两代几何勿混：V3 576 = 512 latent + 64 rope（bf16 1152 B）；V4.1 512 = 448 + 64（FP8 528 B / FP4 288 B，rope 一并量化） | §2 表、式 (2) | §2、练习 1 |
| 64×512 输出 = 32768 regs = $R_{\mathrm{file}}$/2 → 单输出 seesaw：O_L/O_R 竖切（GMMA HEAD_DIM_V/2）+ 双 KV block 交错，半区 online softmax 不变量 | §2 走读 | §2、练习 2 |
| 576 = 9×64 → 9 次 TMA copy 逐份启动 GEMM；EVICT_FIRST（无复用流式）对 FA3 EVICT_LAST（有复用） | traits.h L82-83、splitkv L47 | §2、练习 2 |
| 命题 1：GQA 完美共享 KV 流量为无共享的 $1/g$；decode PackGQA 精确达界（67.1 → 16.8 MB）；causal prefill 打包计数近似守恒 | 命题 1 计数式 | §3、练习 3 |
| $I_{\mathrm{GQA}}$ = 2g/bytes = 4（memory）对 $I_{\mathrm{MLA}}$ = 241.8（compute 边缘）——几何定杠杆 | 式 (3) | §3、§7 |
| paged gather：页内连续满 sector、页间跳转；TMA per-page 对 cp.async gather 两路；PagedKV 禁 IntraWGOverlap（寄存器） | §4 | §4 |
| DSA：主路 FP8 生产口径 10.3 MiB/步，indexer bf16 1.25 GiB = 64× 主路——瓶颈搬家，indexer 精度是新杠杆 | §5 | §5 |
| MergeState：base-2 LSE + ptx_exp2/log2、空态 clamp + d_sum 守卫、vec 化 FMA；多状态走 warp/block 归约树（⊕ 结合交换 recap） | 式 (4)、§6 摘录 | §6、练习 4 |
| scheduler 二分： $n_{\mathrm{new}} \cdot H_{kv} \le G_{\max}$（occupancy 实测）；plan/run 分离、cascade = 大行块 + merge | 式 (5) | §6、练习 4 |
| 决策表：bound 由几何定、预算由几何×硬件定；FA3 pingpong / FlashMLA seesaw / FA4 TMEM 三解 | §7 表 | §7 |

## 练习

1.（计算 · [10]）两代几何与格式账：(a) V3 与 V4.1 的 $d_{\mathrm{qk}}$ 各是多少、各由哪两截拼成；(b) V4.1 FP8 形 528 B 与 FP4 形 288 B 各拆成数据 + scale 两项（式 (2)）；(c) 528 B 对 V3 bf16 1152 B 的倍数；(d) V3 一个 64×576 K block 的 TMA 份数（每份 64×64）与 V4.1 512 维的对应份数。校验：(a) 576 = 512+64 / 512 = 448+64（§2 表，勿混）；(b) 512+16 / 256+32；(c) 2.18×；(d) 9 对 8（576/64 与 512/64）。
2.（计算 · [20]）seesaw 预算账：(a) 64×512 fp32 输出的寄存器数与 $R_{\mathrm{file}}$ 的比值，先预测「双输出 pingpong 是否可行」再算；(b) O_L/O_R 各占多少寄存器，两个 WG 各持一半后 block 内输出总量；(c) 9 份 TMA 每份字节（bf16）与整块字节；(d) 对照 6.3 命题 1： $d_{\mathrm{head}}$ = 128 的 FA3 双输出为何可行（每 WG 64 regs acc_o）。校验：(a) 32768 = 65536/2，不可行（deep-dive L21 逐字）；(b) 16384、32768；(c) 8192 B、73728 B = 72 KB；(d) 6.3 §3/§7。
3.（推导 · [30]）补全命题 1 并对照 MLA：(a) 写出无共享/完美共享的 KV 流量计数式，推出 ÷g；(b) 证明 decode PackGQA 精确达界（构造：g 行 Q tile、KV 装载次数 $H_q \to H_{kv}$）；(c) causal prefill 打包计数： $g$ = 4、 $T_M$ = 128、 $T_N$ = 64、 $T$ = 4096 时打包每 KV head 装载 $\sum_{i=0}^{127} \lceil (i+1)/2 \rceil = 4160$ 对不打包 $4 \times 1056 = 4224$，解释近似守恒与 decode 侧 ÷g 的差异（提示：causal 下每 CTA 的 KV 范围随 token 位置变，打包细化 token 粒度）；(d) 用式 (3) 与 $I_{\mathrm{MLA}}$ 式说明 GQA-4 与 MLA-128 的杠杆差异。校验：(a)(b) 命题 1 路线；(c) 4160/4224 = 0.985（6.3 练习 3(d) 同数）；(d) 4 对 241.8（§3）。
4.（代码 · [30]）FlashInfer 走读（版本锚 5228ea6be8ec）：先写预测再对照——(a) MergeStateKernel 的 LSE 是什么底数？用哪两条 SFU 指令？(b) 两个空态合并会出什么 NaN，代码哪两处守卫？(c) scheduler 二分的目标函数与约束是什么，max_grid_size 怎么来的？(d) plan 与 run 各在哪一侧执行、plan 为什么能跨层复用？校验：(a) base 2（cascade.cuh L42 注释逐字）、ptx_exp2/ptx_log2；(b) (-inf)-(-inf) 与 0/0，clamp + d_sum 守卫（L56-57 注释）；(c) 式 (5)，occupancy API 实测（scheduler.cuh L86/96）；(d) CPU/GPU，同 step 序列长度相同（decode.py L2179/L3237，inference 3.1 §4 recap）。

## 参考

- [DeepSeek-AI, FlashMLA](../references.md)：deepseek-ai/FlashMLA repo（key: flashmla），教学参照 + 官方实现，双版本锚（2026-11-25 fresh tarball 核实）——**main 395cb3afbb33**：README L3（2026.09.30 移除 Hopper/V3 系 breaking change 逐字）、L7（V4.1 sparse/fused）、L25（2026.09.10 release）、L41（1460/950 TFlops B200 CUDA 13.3 逐字）、L79（SM100/SM103）、L138（decode 仅 FP8/FP4）、L140-143（528/288 B 格式逐字）；csrc/cuda_kernels/sm100/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp。**ba89a3466e9470ad08ab39738d4e7bb66989e1e7**：README L81（head_dim_k 576/512、head_dim_v 512）；csrc/kernels/params.h L6-9（ModelType 枚举）；sm90/decode/dense/config.h L7-11（BLOCK_SIZE_M/PAGE_BLOCK_SIZE/HEAD_DIM_K=576/HEAD_DIM_V=512）；traits.h L24（NUM_THREADS=256）/L39/44（GMMA HEAD_DIM_V/2）/L82-83（barriers_K0[9]）；splitkv_mla.cuh L20-23（get_AorC_row_idx）/L36-49（launch_kv_tiles_copy_tma、EVICT_FIRST L47）；docs/20250422-new-kernel-deep-dive.md L21（32768/65536 逐字）/L25（O_L/O_R）/L27-38（步骤 0-11，seesaw 图依据）/L42（seesaw 逐字）/L54（9 TMA 逐字）
- [Ye et al., 2025](../references.md)：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving，arXiv:2501.01005，[A]（key: flashinfer），0.7.6 commit 5228ea6be8ec（2026-11-25 contents API 核实）——include/flashinfer/attention/cascade.cuh（MergeStateKernel L45、base-2 注释 L42、空态守卫注释 L56-57、MergeStateInPlaceKernel L90、threadblock_sync_state L125、warp_sync_state L147、MergeStatesKernel L224、PersistentVariableLengthMergeStatesKernel L378、host MergeState L583）；scheduler.cuh（PartitionPagedKVCacheBinarySearchMinNumPagePerBatch L74-98、new_batch_size L86/96、PrefillBinarySearchKVChunkSize L101-123）；flashinfer/decode.py（BatchDecodeWithPagedKVCacheWrapper L1338、plan L2179、run L3237）；cascade.py（merge_state L42、MultiLevelCascadeAttentionWrapper L226、plan L401、run L537）；jit/core.py（common_nvcc_flags L186、arch flags L194-202）
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——paged KV 与块表语义源头（inference 2.1 recap）；vLLM v1 backend 族接口 recap inference 3.1 §6（本章不重走读）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——V3 MLA 几何（kv_lora_rank 512 + qk_rope 64 = 576、H_q 128）
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：[W]（key: v41-flash-config）——head_dim 512 = 448 latent + 64 rope（config 无 kv_lora_rank 字段、引擎导出 448）、40 层；模型侧形式化归 models
- [DeepSeek-AI, DeepGEMM](../references.md)：deepseek-ai/DeepGEMM repo（key: deepgemm），教学参照 + 官方实现，commit 5081f8513af5——mqa_logits（lightning indexer）kernel，DSA 打分 GEMV（§5）；mainloop 走读归 5.2/6.2
- [Dao et al., 2022](../references.md)：FlashAttention，arXiv:2205.14135，[A]（key: flashattention）——repo b038b4974f26：flash_fwd_kernel.h L157（bidh/h_h_k_ratio head 映射）、L597-598（paged block_table 翻译）；hopper/ PackGQA 与 PagedKVNonTMA 锚见 6.3 参考（tile_scheduler.hpp L108/L605、mainloop L56/L284-286/L752-757、tile_size.h L51）
- 跨维度指引（纯文字）：models 4.1 §4（MLA 构造与 absorption 式 (11)-(14)）、§3（GQA 广播视图）；inference 3.1 §2（split-KV 命题 1 与式 (4)）、§4（BSR/plan-run/cascade 命题 2）、§5（FlashMLA 几何与 $I_{\mathrm{MLA}}$ 式 (8)）、§6（vLLM backend 接口）、§7（DSA 选择链与式 (9)(10)）；inference 2.1（块表式 (3)）、1.2（ $\kappa$ 式 (3)）；gpu 6.3（命题 1/2、§5 EVICT_LAST、§7 三代 overlap）、6.1（MX/NVFP4 格式账）、5.1（归约原语）、4.1（coalescing）、2.4（roofline）
