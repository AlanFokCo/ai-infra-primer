# 8.1 Serving 侧并行策略与账（Serving-Side Parallelism: Strategies and Accounting）

> training 5.1 的账是每层 4 次 all-reduce——serving 只跑前向，这 4 次变成几次、消息缩到多大、为什么延迟项反而成了主项？MLA 只有一个 KV head，TP 切不动它，DP attention 凭什么消掉复制？LoongServe 的弹性序列并行、DeepInfer release 的 balanced CP、训练侧的 ring CP，三个「CP」各是什么？本章把 training P5 的五维并行账翻译到 serving 侧，重推 TP decode 延迟账与 DP attention 通信账，落到一张模型规模 × 上下文 × SLO × 拓扑的组合决策表 | 前置：1.1（TTFT / TPOT / goodput 口径）、1.2（ $\kappa$ 、decode 强度与 Wide-EP 伏笔）、1.3（scale-up domain 与硬件参数表）、3.3（EP dispatch/combine 语义与字节账）、4.3（chunked prefill）、5.3 / 6.3（vLLM / SGLang 分布式执行结构）、7.4（Wide-EP 生产口径与 EPLB）；training 5.1 / 5.2 / 5.3 / 5.4 / 5.5 只 recap + 指引，不重推训练账 | 锚点：LoongServe（[Wu et al., 2024](../references.md)，arXiv:2404.09526，[A]，LoongServe/LoongServe 官方实现）、DeepInfer doc + repo（[DeepSeek-AI, inference_deployment_qa.md](../references.md)、[DeepSeek-AI, DeepInfer](../references.md)，教学参照 + 官方披露）、SGLang v0.4 blog（[SGLang Team, 2024](../references.md)，[C] 开放技术博客、非闭源）、vLLM / SGLang repo（[A]）、V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、NVIDIA datasheet（[NVIDIA, H100/H200/B200 datasheet](../references.md)，[C] 官方规格披露）

## 1. 从训练账到 serving 账：无 backward 改变了什么

先回答一个更基本的问题：serving 为什么不能直接搬训练侧的并行配置？因为推理只执行前向图——没有 loss、没有 autograd 反向遍历，训练账里所有「反向对称翻倍」的项整笔消失，而消息量从训练的 micro-batch 序列缩到 decode 的每序列 1 token。通信次数减半、消息缩几个量级，这两件事叠加，把训练侧「带宽主导」的通信账翻成 serving 侧「延迟主导」的账。并行度的取舍随之改变：训练里 TP 压关键路径换显存，serving 里 TP 的每层归约可能根本不值得付。

training P5 五维的 recap 与本章增量对照如下（训练账一律 recap + 纯文字指引，不重推导）：

| 维度 | training P5 口径（recap） | serving 增量（本章） | 指引 |
| --- | --- | --- | --- |
| TP | 每层 4 次 all-reduce（前向 g×2 + 反向 f×2），消息 $2bsh$ bytes，通信-计算比 $\propto (t-1)$ | 无 backward → 每层 2 次前向 × $L$ ；decode 消息 $2B d_{\mathrm{model}}$ ，延迟项主导 | training 5.1 §6 式 (9)-(11)；本章 §2 |
| PP | GPipe / 1F1B 气泡占比 $(p-1)/(m+p-1)$ ， $t_b \approx 2t_f$ ，峰值激活 $\le p$ 份 | 前向-only 气泡同形、绝对时长减；decode token 级流水，TPOT 加 $(p-1)\tau_{\mathrm{p2p}}$ | training 5.3 §2-3；本章 §3 |
| EP | dispatch/combine 每 token 每层 $2k d_{\mathrm{model}}$ 元素上界，木桶 $\max_j R_j$ | decode 小消息延迟敏感；Wide-EP 每步 $\propto 1/e$ + EPLB 兜底 | training 5.4 §1-2/§4；3.3 §6、7.4；本章 §4 |
| CP | ring attention 训练账，固定度，含反向 | 三义区分：LoongServe ESP ≠ 训练 CP ≠ DeepInfer balanced CP | training 5.2；本章 §6 |
| 组合 | world size 乘积约束 + group 树，TP 进节点、PP/DP/EP 可跨节点 | 模型规模 × 上下文 × SLO × 拓扑 → $t/p/e/d$ 决策表 | training 5.5 §3/§6；本章 §7 |

**符号翻译口径注**（spec §4.3 硬规则）：training 5.1 通信账的 $b$ 是 micro-batch 序列数、 $s$ 是序列长、 $h$ 是 hidden 宽；recap 到本维度一律译作 $B$ （batch 内 token 数，decode 口径每序列 1 token，故 $B$ = 序列数）、 $s \to 1$ 、 $h \to d_{\mathrm{model}}$ 。training 4.1 式 (3) 的 $\alpha$ （每跳延迟）译作 $\tau_{\mathrm{hop}}$ 、 $\beta$ （每字节时间）译作 $1/\mathrm{BW}$ ——本维度 $\alpha$ 专归投机接受率、 $\gamma$ 专归 draft 长度，不沿用兄弟维度字母的兄弟语义。models 4.5 / training 5.4 的 $d$ （hidden 维）译作 $d_{\mathrm{model}}$ （本维度 $d$ 专归 DP 度，与 $d_{\mathrm{model}}$ 严格区分）；training 5.1 的 $a_{kv}$ （KV head 数）译作 $H_{kv}$ 。bytes 不占数学字母，一律文字「字节」或 $\mathrm{bytes}$ ； $b$ 在本维度专归 block size（2.1），本章不用。

**本章口径**（章脊锁定，全章例题复用，换口径随文声明）：模型主锚 V4.1-Flash（[W]： $L$ = 40、 $d_{\mathrm{model}}$ = 5120、 $H_{kv}$ = 1、共享 KV 向量 512 元素 = $d_{\mathrm{latent}}$ 448 + $d_{\mathrm{rope}}$ 64、 $E$ = 384 routed + 1 shared、 $k$ = 6）；硬件主锚 8 卡 NVLink 节点，NVLink 900 GB/s（H100 / H200 datasheet 口径，1.3 参数表；B200 为 1.8 TB/s，换卡处随文注）；激活 bf16（每元素 2 字节），KV 精度随文注。TP 延迟账取 $t$ = 8、 $B$ = 60；DP attention 与 PP 例题取 $B$ = 64（原因：DP 按请求均分需整除，60 不整除 8，padding 到 64，§5 声明）；长上下文演示取 $T$ = 131072（原因：展示 KV 主导区，§5 声明）。生产数字标「官方披露」+ doc 行号 + 核实日期 2026-11-23。

边界声明：本章讲 serving 侧并行的传输语义、带宽/延迟账与组合决策。RDMA / NVLink 硬件、NCCL 实现、交换机拓扑归 cluster 维度（纯文字指引，不放死链）；PD 分离的原理账归 8.2、KV 传输归 8.3、系统对照归 8.4、弹性容错归 8.5；引擎侧执行结构（通信组、归约路径、配置面）已在 5.3 / 6.3 / 7.4 走读，本章 recap + 指针不重展开。V4.1-Flash 模型侧形式化（CED / YOCO / Engram / DSpark）归 models 维度（V4.1 待补），本章只引 config 字段做 serving 账。

## 2. TP serving：每层 2 次前向归约与 decode 延迟账

training 5.1 §6 的每层 4 次 all-reduce，到 serving 剩几次？答案是 2 次——不是 1 次，也不是 4 次。这一节把次数、消息量、每步时间三笔账重推到数（spec §6.3 强制重推导），再落到 TPOT 占比与 NVLink 域边界。5.3 §3 已在 vLLM 引擎级交付「次数减半 + 延迟项主导」两条结构结论与代码落点；本节是它显式留给 8.1 的全账（度数选择、TPOT 占比、域边界），推导自包含。

recap training 5.1 的 f/g 共轭（不重推）：f = 前向恒等 / 反向 all-reduce，g = 前向 all-reduce / 反向恒等。一层 transformer 的 attention 块与 MLP 块各贴一对 f/g：前向通信在两处 g（attention 的 out 投影、MLP 的 down 投影，都是行切），反向通信在两处 f。训练前向反向都跑，每层 4 次；消息量 $2bsh$ bytes（training 5.1 式 (9)，其 $b$ = micro-batch 序列数）。serving 前向-only 的位置图（每卡视角，AR = all-reduce，decode 每序列 1 token）：

```
        复制区段（各卡全量：残差、RMSNorm —— serving 无 dropout）
            │
 x [B,d_model] ─┼─ f ─► QKV 列切 ─► 逐 head attention ─► out proj 行切 ─► g ─► (+残差)
            │      ▲ 前向恒等（无反向）                       ▲ 前向 AR ①
            ├─ f ─► W1 列切 ─► φ 逐位置 ─► W2/MoE down 行切 ─► g ─► (+残差)
            │                                               ▲ 前向 AR ②
     前向通信：g × 2（① attention out-proj + ② MLP/MoE down-proj）
     反向通信：无（推理不执行反向图）—— 每层 2 次，每步 2L 次
```

**命题 1**（serving 归约位点）：推理前向中，TP 组内每层的集合归约位点恰为 2 处（attention 块与 MLP/MoE 块各一处），每 decode step 共 $2L$ 次 all-reduce；不存在反向归约。

**证明**（路线：推理图的算子集合 → f/g 触发条件 → 逐层计数，三步）。第一步，推理只执行前向图，反向图不存在（无 loss、无 autograd 反向遍历），故 f 的反向归约永不触发。第二步，f 的归约在其反向、g 的归约在其前向（training 5.1 式 (5)(6) 共轭定义）——前向图里只有 g 触发，f 是恒等。第三步，每层两个行切位点（attention out 投影、MLP/MoE down 投影）各贴一个 g，故每层恰 2 处， $L$ 层共 $2L$ 处。∎ 正例：V4.1 $L$ = 40 → 每步 80 处归约。边界两条：其一，开 sequence parallelism 时归约原语换成 reduce-scatter + all-gather 对，原语数翻倍但归约位点仍每层 2 处（命题计位点不计 kernel 数，training 5.2 §2 恒等）；其二，vocab parallel embedding 另有一处前向归约（training 5.1 §9），按整模型计须加 1，本命题只计 transformer 层。

> **坑**：把 serving TP 通信写成「每层 1 次 all-reduce × $L$ 」，低估 2×（四路审查 C-S3-1）。触发条件：沿「× $L$ 」字面把每层归约数读成 1，或把训练的 4 次「减半」再误减一次；现象：每步归约数算成 $L$ = 40 而非 $2L$ = 80，延迟账整笔虚低一半；规避：训练 4 次 = 前向 2 + 反向 2，去掉 backward 剩前向的 2（attention out-proj + MLP down-proj 各 1），不是 1。每 decode step 是 $2L$ 次。

消息量。decode 每序列 1 token，training 的 $2bsh$ 翻译到本维度（ $b \to B$ 、 $s \to 1$ 、 $h \to d_{\mathrm{model}}$ ）即：

$$
M_{\mathrm{msg}} = 2 \thinspace B \thinspace d_{\mathrm{model}} \qquad (1)
$$

**符号**： $M_{\mathrm{msg}}$ = 单次 all-reduce 的消息字节数；系数 2 = bf16 每元素字节数（字节量已编码在系数里，bytes 不占字母）； $B$ = batch 内序列数（decode 口径，每序列 1 token，故亦为 batch token 数）； $d_{\mathrm{model}}$ = hidden 宽。代入章脊口径 $B$ = 60、 $d_{\mathrm{model}}$ = 5120： $M_{\mathrm{msg}} = 2 \times 60 \times 5120 = 614{,}400$ 字节 ≈ 0.61 MB。 $B$ = 1 时 10,240 字节 = 10 KiB——亚 MB 级小消息。prefill 口径（ $T$ 个 token 一次前向，换口径原因：展示消息随 token 数线性涨）： $T$ = 4096 → 41.9 MB，是 decode $B$ = 60 的 68 倍。

每步时间。recap training 4.1 式 (2)(3) 的 ring 账（每卡收发字节乘 $2(t-1)/t$ 、延迟项 $2(t-1)$ 跳），符号翻译后，每步 $2L$ 次归约的总时间：

$$
\tau_{\mathrm{TP}} = 2L \left( \tau_{\mathrm{lat}} + \frac{2(t-1)}{t} \cdot \frac{M_{\mathrm{msg}}}{\mathrm{BW}} \right) \qquad (2)
$$

**符号**： $\tau_{\mathrm{TP}}$ = 每步 TP 通信总时间； $2L$ = 每步归约次数（命题 1）； $\tau_{\mathrm{lat}}$ = 每次归约的固定延迟（含 launch、组同步与跳数；ring 口径下 $\approx 2(t-1)\tau_{\mathrm{hop}}$ ，custom allreduce / multimem 可压跳数，5.3 §4；教学值 10–20 μs，本章取 15 μs——5.3 取 5 μs，两者都是量级示意非实测，实测依硬件与实现归 cluster 维度）； $\mathrm{BW}$ = 链路带宽（NVLink 900 GB/s，datasheet 口径 [C]）； $t$ = TP 度。代入 $t$ = 8、 $L$ = 40、 $B$ = 60：ring 系数 $2 \times 7/8 = 1.75$ ，带宽项 $= 1.75 \times 614400 / (9 \times 10^{11}) = 1.19$ μs，每次归约 $15 + 1.19 = 16.19$ μs，每步 $\tau_{\mathrm{TP}} = 80 \times 16.19 = 1295.6$ μs ≈ 1.30 ms（纯延迟项 $80 \times 15 = 1.2$ ms）。带宽项只占 7.4%——**延迟项主导**。

两个极限读数。其一， $B$ = 1：消息缩到 1/60，带宽项降到 0.02 μs，每步 $\tau_{\mathrm{TP}} = 1201.6$ μs——消息量缩 60 倍，通信时间只降 7.25%。延迟项与消息量无关，小 batch decode 的 TP 成本几乎全是它，这是 latency-bound（口径：通信时间由固定延迟而非带宽决定）的定量含义。其二，prefill $T$ = 4096：带宽项 81.6 μs ≫ $\tau_{\mathrm{lat}}$ 15 μs，每步 7.72 ms，**带宽项主导，与 decode 相反**（5.3 §3 已交付此对照，recap）。同一个 TP 组，两个阶段的最优归约路径不同——decode 压延迟（custom kernel / symmetric memory），prefill 压带宽（ring 近最优）。

TPOT 占比与关键路径。1.30 ms 对生产 decode 操作点是什么份额？7.4 §2 的 wide_ep pareto 图（图读近似）约 100–200 tok/s/user，对应 TPOT 5–10 ms—— $\tau_{\mathrm{TP}}$ 占 13%–26%；对 1.1 教学口径 TPOT 30 ms 占 4.3%。且这笔时间是**暴露的**：all-reduce 卡在两段 GEMM 之间的关键路径上，同一 batch 的下一段计算必须等归约完成（training 5.1 §6「难 overlap」），不像 EP dispatch/combine 有 kernel 级融合重叠（§4）。度数为什么不能无限加？training 5.1 式 (11) 的通信-计算比 $\propto (t-1)$ 在 serving 侧同形成立：每卡计算时间随 $t$ 反比缩（权重读 $W/t$ ， $W$ = 每步权重读字节，1.2 式 (4)），而延迟项 $2L \cdot \tau_{\mathrm{lat}}$ 近乎常数（ring 口径 $\tau_{\mathrm{lat}}$ 还随 $t$ 线性涨），比值至少线性恶化。

NVLink 域内边界。把带宽换成跨节点口径（training 5.1 §6 的 IB 每卡 50 GB/s）：带宽项 $= 1.75 \times 614400 / (5 \times 10^{10}) = 21.5$ μs，已超延迟项 15 μs，每步下界 2.92 ms——且跨节点跳的固定延迟高于 NVLink 跳（RDMA 往返与协议开销，硬件测量归 cluster 维度），实际更差。decode 小消息在域内靠延迟项撑住，跨域后带宽项与延迟项双涨。**TP 留在 scale-up domain 内、 $t$ 不超过域内卡数**（1.3 §3：H100/H200 典型 8 卡 HGX，GB200 NVL72 为 72 卡单域），是这笔除法的结果，不是经验习惯。

DeepInfer 不用 TP 的官方 rationale（doc L69 逐字，官方披露，2026-11-23 核实）：「Tensor parallelism (TP) has high communication overhead. We do not use it when the model fits in memory and the latency SLA can be met without it.」拆成两笔：其一，TP 的训练侧动机是显存——把参数/激活切到 $t$ 卡装下大模型；V4.1 的参数大头在 expert（§5：routed expert 权重约 289 GB MXFP4），已被 EP 切走，MLA 又把 KV 压到每 token 每层 512 元素，非 expert 参数与 KV 在单卡预算内可安置——TP 的显存动机不存在。其二，TP 的通信成本（本节 1.30 ms/step）是纯损失：模型装得下、SLA 可达，就不付这笔钱。这与 training 5.1 §8「V3 不用 TP」的论文口径同构（MLA 可切份额小 + 参数大头已被 EP 切走 + 通信随 $t$ 线性涨），serving 侧再叠加 decode 延迟项主导这一笔。

> **提点**：TP 买的是层内并行（同一步的计算切到 $t$ 卡，降单步延迟），代价是每层 2 次关键路径归约。当显存动机消失（EP 已切走大头、MLA 已压小 KV），这笔代价就没有可换的对象——「不用 TP」不是保守，是账面上无收益。

引擎落点 recap（不重展开）：vLLM 三条归约路径（NCCL / CustomAllreduce / SymmMemCommunicator）与门函数见 5.3 §4；SGLang 原语目录（custom_all_reduce / torch_symm_mem / pynccl 等）见 6.3 §2；配置面 vLLM 显式字段 tensor_parallel_size 见 5.3 §2、SGLang 派生面见 6.3 §1。

## 3. PP serving：prefill 流水气泡与 decode token 级流水

PP 把模型按层切成 $p$ 段放到 $p$ 组设备上。训练侧的气泡账（training 5.3）serving 侧怎么变？先回答一个更基本的问题：没有 backward 之后，流水线里还剩什么要调度？只剩前向——micro-batch 沿 stage 正序推进，没有反向回传，GPipe 与 1F1B 的区分消失（1F1B 的全部设计是「早反向早释放激活」，无 backward 则无此问题）。但气泡并没有消失，只是换了结构；decode 侧更是把「micro-batch 流水」换成了「token 级流水」。

recap training 5.3（不重推）：GPipe 气泡占比 $(p-1)/(m+p-1)$ （式 (5)， $m$ = micro-batch 数），1F1B 气泡不变、峰值激活从 $m$ 份降到 $\le p$ 份， $t_b \approx 2t_f$ 。

**prefill PP**。前向-only 流水的 makespan 为 $(m+p-1)\thinspace t_f$ （ $t_f$ = 单 micro-batch 单 stage 前向耗时），气泡 $(p-1)\thinspace t_f$ ，占比 $(p-1)/(m+p-1)$ ——**与训练 GPipe 式 (5) 同形**（分子分母同含 $t_f$ ，约掉）。两处 serving 差异：其一，绝对气泡时长少了 $t_b$ 项（训练气泡 $(p-1)(t_f+t_b)$ ， $t_b \approx 2t_f$ ），且无 backward 激活驻留——前向算完即释放， $m$ 不再受训练侧「峰值激活 $\le p$ 份」的显存约束，可以开大。其二，延迟侧的代价：首 token 要等本 micro-batch 走完 $p$ 段，TTFT 的关键路径 $= p \cdot t_f$ ；增大 $m$ 消气泡，但后续 micro-batch 排队更久，tail TTFT 涨。**气泡 vs 延迟是同一枚硬币**： $m$ 大 → 气泡小、排队 TTFT 大； $m$ 小 → 反之。代入 $p$ = 4： $m$ = 8 气泡占比 27.3%， $m$ = 16 降到 15.8%——消气泡靠 $m$ 对 $p$ 过采样，与训练同构。chunked prefill（4.3）天然提供 micro-batch：长 prompt 切成的 chunk 就是流水的进料，stall-free batching 与 PP 流水可组合。

**decode PP**。decode 每步每序列只产 1 token，流水的进料不是「micro-batch 的前向+反向」而是「token 的 hidden 沿 stage 正序走一遍」。设每 stage 持 $L/p$ 层、单 micro-batch 单 stage 步时 $t_s$ ，相邻 stage 间点对点送中间 tensor 耗时 $\tau_{\mathrm{p2p}}$ 。每 token 的关键路径：

$$
\mathrm{TPOT}_{\mathrm{PP}} = p \cdot t_s + (p-1)\cdot \tau_{\mathrm{p2p}} \qquad (3)
$$

**符号**： $p$ = PP 度（stage 数）； $t_s$ = 单 stage 单 micro-batch 步时； $\tau_{\mathrm{p2p}}$ = 一次 stage 边界点对点传输时间 $= \tau_{\mathrm{hop}} + 2B d_{\mathrm{model}}/\mathrm{BW}$ （消息与 TP 单次归约同尺寸，但只 1 跳、无 ring 的 $2(t-1)$ 跳）。口径注：式 (3) 是满流水（在飞 micro-batch 数 $m \ge p$ ）下每 token 的关键路径下界；严格满流水需 $m \cdot t_s$ 不小于关键路径长， $m$ 略大于 $p$ ，循环间隙气泡量级 $(p-1)\tau_{\mathrm{p2p}}/(p \cdot t_s)$ ——远小于训练 GPipe 气泡，因为 decode 的 $\tau_{\mathrm{p2p}}$ （μs 级）≪ $t_s$ （ms 级）。代入 $p$ = 2、 $B$ = 64、 $d_{\mathrm{model}}$ = 5120、bf16：消息 $2 \times 64 \times 5120 = 655{,}360$ 字节，带宽项 0.73 μs，取 $\tau_{\mathrm{p2p}} \approx 5$ μs（教学值），TPOT 开销 $= (2-1) \times 5 = 5$ μs。对照 §2 的 TP 1.30 ms/step：**PP 每步只在 $p-1$ 个层段边界送一次，TP 每步在层内归约 $2L$ = 80 次**——这是「层间通信」与「层内通信」的结构差，PP 的延迟开销比 TP 小两个量级。

token 级流水的时序（ $p$ = 2，在飞 micro-batch mb1/mb2，每格 $t_s$ ，→ 为 stage 边界 P2P）：

```
 时刻      t0      t1      t2      t3      t4
 stage1  [mb1]   [mb2]   [mb1]   [mb2]   [mb1]   ← 每 t_s 进料一个 micro-batch
            └→      └→      └→      └→
 stage2          [mb1]   [mb2]   [mb1]   [mb2]   ← 滞后 1 格，满流水两 stage 同忙
 mb1 每 token 关键路径 = 2·t_s + 1·τ_p2p（遍历 p 段 + p-1 次边界送）
```

那 PP 买的是什么？容量，不是单请求延迟。每 token 仍要串行遍历全部 $L$ 层， $p \cdot t_s$ 不减；PP 把权重与 KV 切到 $p$ 段，每卡只持 $1/p$ ，装得下更大的模型与 batch。吞吐侧，满流水让 $p$ 个 stage 同时读各自的 KV、算各自的层，聚合 HBM 带宽 $p \cdot \mathrm{BW}$ 被用满。这落到一条可判定的访存账：

**命题 2**（KV 读的切分与复制）：decode PP 下每 stage 每 micro-batch 的 KV 读字节为 $(B/m)\cdot \kappa \cdot (L/p)\cdot T$ ；TP 下若 $H_{kv} \lt t$ （MLA 恒有 $H_{kv}$ = 1），每 rank 每步 KV 读字节为 $B \cdot \kappa L T$ ，与 $t$ 无关。即 PP 沿层切 KV 读、DP 沿请求切 KV 读，而 TP 在 MLA 上复制 KV 读。

**证明**（路线：KV 的物理驻留位置 → 每步谁读哪几行，两步）。第一步，PP 的 stage $i$ 持层 $[(i-1)L/p,\ iL/p)$ 的权重与这些层的 KV——各 stage 的 KV 按层不相交；一个 micro-batch（ $B/m$ 条请求）经过 stage $i$ 时只读该 stage 那 $L/p$ 层的 KV，故每 stage 每 micro-batch 读 $(B/m)\kappa(L/p)T$ 。第二步，TP 按 head 切 Q 侧，但 MLA 的全部 head 共享一个低秩 latent（training 5.1 §8：latent 投影无处可切、每 rank 复制），每个 rank 的 Q head 都要对全部缓存位置读完整 latent——KV 读不随 $t$ 分片，每 rank 每步读 $B\kappa L T$ 。∎ 正例：V4.1 bf16 $\kappa$ = 1024 B/token/层、 $L$ = 40、 $T$ = 4096、 $B$ = 64：每请求 KV $= 1024 \times 40 \times 4096 = 160$ MiB，全 batch 10 GiB。TP $t$ = 8 每 rank 读全量 10.74 GB（3.35 TB/s 下 3.21 ms）；PP $p$ = 2、 $m$ = 2 每 stage 读 $(64/2) \times 1024 \times 20 \times 4096 = 2.5$ GiB——TP 的 1/4。边界：GQA 且 $H_{kv} \ge t$ 时 KV head 可切，TP 每 rank KV 读降为 $B\kappa L T/t$ ，复制命题不成立——这正是 §5 DP attention 对 MLA 必要、对宽 GQA 不那么必要的原因。

读这条命题的吞吐含义：长上下文 decode 的 KV 读主导步时（1.2 式 (4) 饱和区），TP 的每卡 KV 读不随 $t$ 缩 → 加卡不省带宽，吞吐撞墙；PP 与 DP 的每卡 KV 读随度数缩 → 聚合带宽被用满。容量侧同理：把 $T$ 拉到 131072（换口径原因：展示 KV 主导区），全量口径 KV 总驻留 320 GiB——TP 每卡复制全量 320 GiB，超单卡 HBM，容量上不可行；DP $d$ = 8 每卡持 $320/8$ = 40 GiB 可容（§5）；PP 每段持 $320/p$ GiB（全部请求、本段层），需 $p \ge 8$ 才压到 40 GiB， $p$ = 2 每段 160 GiB 仍超。口径注：V4.1 混合注意力（SWA / CED / DSA top-k）使实际每步 KV 读与驻留远小于全量 $\kappa L T$ 上界（7.2），此处全量口径是结构演示，「TP 复制 vs PP/DP 切分」的结论不变。

引擎落点 recap：vLLM 的 AsyncIntermediateTensors 异步承接层间中间结果（decode PP token 级流水的前提）见 5.3 §2；SGLang 的 scheduler_pp_mixin + pp_sync_window 见 6.3 §1。PP 与 PD 分离是两件事：PP 切层、同一请求遍历各段；PD 切阶段、prefill 与 decode 在不同实例（8.2）。DeepInfer 生产用 PD 分离而非 PP（doc L31/L47-53，§7）。

## 4. EP serving：dispatch/combine 延迟账与 Wide-EP recap

MoE 的 expert 切到 $e$ 张卡后，token 按路由流动——serving 侧的 dispatch/combine 与训练侧有什么不同？答案是消息尺寸：训练一个 micro-batch 几千 token，decode 每序列 1 token，all-to-all 的逐 peer 消息缩到 KB 级，瓶颈从带宽翻成延迟。字节账与语义已在 3.3 §6 交付，本节 recap + 指针，增量落在每步暴露延迟、木桶命题的 serving 读法与 Wide-EP 生产口径。

recap 三处（不重推）：training 5.4 §1-2——dispatch/combine 两次 all-to-all，无归约语义（combine 的加权求和在接收后本地），每 token 每层 $2k d_{\mathrm{model}}$ 元素上界（bf16 合 $4k d_{\mathrm{model}}$ bytes，models 4.5 §7 原式 $2kd$ 的 $d$ = hidden，译作 $d_{\mathrm{model}}$）；3.3 §6 式 (6)——serving 字节账 $`\mathrm{bytes}_{\mathrm{dispatch}} = Bk\thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}}`$ （FP8 dispatch 每元素 1 字节，scale 另计），decode 逐 peer KB 级 → 延迟主导 → low-latency kernel 存在的定量依据；7.4 §2——Wide-EP（大 $e$ 的 EP 部署，decode EP64–128 量级，定义 recap 1.2 §6 / 7.4 §1）每卡权重流量 $\propto E/e$ 、每步 $\propto 1/e$ 至通信/同步主导（EP>32 递减，doc L293-297）。

每步暴露延迟。每 MoE 层 dispatch + combine 两次 all-to-all， $L_{\mathrm{moe}}$ 个 MoE 层每步共 $2L_{\mathrm{moe}}$ 次。每次 all-to-all 的时间同为「固定延迟 + 带宽项」结构，带宽项按**每 rank 共享出口链路**口径（一 rank 的全部发送字节过自己的 NIC / NVLink 端口；逐 peer 交换在交换 fabric 上并行，聚合时间不串行累加 $e-1$ 份）： $`\tau_{\mathrm{a2a}} + \frac{e-1}{e}\cdot \frac{(B/d)\thinspace k\thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}}}{\mathrm{BW}}`$ （每 rank 发送 $(B/d)k\thinspace d_{\mathrm{model}}$ 个元素、其中 $(e-1)/e$ 发往他卡； $\tau_{\mathrm{a2a}}$ 为 all-to-all 固定延迟，教学值与 $\tau_{\mathrm{lat}}$ 同量级 10–20 μs，launch + 组同步开销，实测归 cluster）。代入 $B$ = 64、 $d$ = $e$ = 8、 $k$ = 6，FP8 dispatch（换口径原因：对回 3.3 §6 的 FP8 / IB 锚）：每 rank 每层 dispatch 发送 $8 \times 6 \times 5120 \times 1 = 245{,}760$ 字节，共享出口带宽项 $= 0.875 \times 245760 / (5 \times 10^{10}) = 4.30$ μs，延迟项 $\tau_{\mathrm{a2a}}$ 15 μs 仍主导（带宽占 22%）。另一笔是**逐 peer 消息尺寸**：均衡路由摊到 $e$ = 8 个 rank，逐 peer $245760/8 = 30{,}720$ 字节 = 30 KiB，单份 chunk 过链路 0.61 μs——3.3 §6 的 $B_{\mathrm{local}}$ = 64 / EP64 锚（每卡每层 1.97 MB、逐 peer 30.7 KB、0.61 μs）正是此逐 peer 读数。**口径注（两种链路读数勿混）**：0.61 μs 是逐 peer 单份 chunk 的链路时间（消息尺寸读数），4.30 μs 是每 rank 全部字节过共享出口的聚合带宽项（本节公式所用）——聚合 = 逐 peer × $(e-1)$ 份，但 fabric 上各 peer 并行，聚合时间不另乘 $e-1$ 进关键路径；§5 上界的 0.48 μs 同用共享出口模型（bf16 / NVLink）。V4.1 config 无 first_k_dense_replace 字段，本章按每层含 MoE 的上界 $L_{\mathrm{moe}} \le L$ = 40 计；每步暴露延迟上界 $2 \times 40 \times (15 + 0.48) \approx 1.24$ ms（bf16 激活、NVLink 900 GB/s，共享出口带宽项 $0.875 \times 491520 / (9 \times 10^{11}) = 0.48$ μs，与 §5 同模型）。

关键路径差异。TP 的 all-reduce 是层内依赖屏障（§2），EP 的 dispatch/combine 却有成熟的重叠手段：Mega MoE 单 mega-kernel 融合 dispatch + GEMM1/2 + SwiGLU + combine、NVLink 通信与 tensor core 计算重叠（3.3 §5），TBO/SBO 双批重叠把 dispatch 藏进另一批的计算（6.2）。所以 EP 的 1.24 ms 是**未重叠上界**，暴露部分可压到「通信时间减去可重叠计算时间」与 0 的较大者；TP 的 1.30 ms 则基本全暴露。这是 §5 DP attention 选 EP 而非 MoE TP 的账面理由之一。

木桶命题的 serving 读法。training 5.4 §4 的木桶命题（一层 MoE 墙上时间 $\propto \max_j R_j$ ，偏斜乘性进层时长）在 serving 侧直接落成 tail TPOT：热 expert 所在卡拖住整层，其余 $e-1$ 卡空等，每步 TPOT 被乘性抬高。训练侧靠路由控制（aux-loss-free 偏置控制器）兜底，推理侧靠容量复制——EPLB（expert parallelism load balancer，按运行时负载统计重分布 expert 副本以均衡各卡计算的机制）。recap 7.4 §3（不重推）：每 GPU 另持冗余 expert 承载热副本，收集 last N steps 负载、every M steps 重分配，权重经 NVLink symm_mem shuffle；生产参数 prefill 冗余 4/卡、统计窗 100 步、重分配 1000 步，decode 冗余 1/卡（EP64 → 7 expert/GPU）、窗 1000 步、重分配 10000 步；实测 moe_lb（slowest rank / mean，EPLB 已生效后的残差偏斜）prefill 1.074（EP4DP4）/ 1.045（EP2DP2）、decode 1.090。归属口径（spec §2 裁定，W4 S3-11 收窄）：serving 侧 EPLB 部署与算法归 7.4（本节 recap），硬件/网络拓扑归 cluster。

拓扑约束。all-to-all 的域直径就是链路选择：EP 域收进 scale-up domain 内走 NVLink（900 GB/s / 1.8 TB/s）， $e$ 受域内卡数封顶；Wide-EP decode $e$ = 64–128 跨超节点（doc L49），整超节点作单 EP 组避免多 job 共享 NVLink 互相干扰（doc L297，7.4 §2）。跨域 dispatch/combine 上 RDMA，延迟项更高——low-latency kernel 纯 RDMA 点对点绕过两阶段转发正是为此（3.3 §6，V1 语义锚 tag v1.2.1；当前 DeepEP main 为 V2.5 统一 EPBuffer，版本漂移注见 3.3）。

## 5. DP attention for MLA：消 KV 复制的通信账

MLA 只有一个共享 KV head，TP 按 head 切时 latent 无处可切、每 rank 复制全量 KV（命题 2）——容量浪费 $t$ 倍、KV 读带宽浪费 $t$ 倍。有没有一种并行，让 attention 不复制 KV、又不付 TP 的层内归约？这就是 DP attention。本节重推它的通信账（spec §6.3 强制重推导）：attention 侧免了什么、MoE 侧付了什么、边界为什么对齐。

所谓 DP attention，是指 attention 按请求切给各 DP rank 独立算、MoE/FFN 仍在全 world 上切分的混合并行（定义 recap 6.3 §1，不重展开配置面）。recap 问题源头（不重推）：training 5.1 §8——GQA 当 $H_{kv} \lt t$ 时 KV 侧无 head 可切、每 rank 持完整 KV 投影与激活；MLA 的 latent 投影被全部 head 共享、无处可切，复制的 latent 激活还随序列长线性涨。6.3 §1 式 (3) 已给容量账 $M_{\mathrm{rank}} = \kappa L T \cdot B/d$ （V3 口径数值）：纯 TP（ $d$ = 1）每 rank 持全量 $B$ 条请求的 KV， $d$ = $t$ 时复制因子降到 1。命题 2 把同一件事落到带宽账：TP 每 rank KV 读 $B\kappa L T$ 与 $t$ 无关，DP 每 rank $(B/d)\kappa L T$ 。

通信账的路线：先追踪每 rank 在 attention 输出端持有什么 layout，再看 MoE 侧需要什么 layout，两者不一致处就是要付的边界通信——关键一步是「token 的 home rank = dispatch 源 rank = combine 目的 rank」是否成立。

**命题 3**（DP↔EP 边界对齐）：设 attention DP 度 $d$ 、attention TP 度 $a$ = 1、MoE EP 度 $e$ 。若 $d = e$ （#DP = #EP），则每层集合通信恰为：attention 块 0 次、MoE 块 dispatch + combine 2 次 all-to-all，DP↔EP 边界无额外 all-gather / reduce-scatter。若 MoE 改用 TP（度 $d$ ），则每 MoE 层边界需 1 次 all-gather + 1 次 reduce-scatter（合计字节与延迟等价 1 次 all-reduce），attention 块仍 0 次。

**证明**（EP 形式，路线：逐段 layout 追踪 → 边界 mismatch 计数，三步）。第一步， $a$ = 1 时每 rank 持完整 attention 权重、独立算自己 $B/d$ 条请求的 attention，输出 layout 为 $[B/d,\ d_{\mathrm{model}}]$ ，无跨 rank 归约（attention 块 0 次）。第二步， $d = e$ 时 EP 组与 DP 组是同一组 rank：dispatch 把每条 token 的 hidden 发往其 $k$ 个 expert 所在 rank，源 rank 恰是 token 的 home rank；combine 把 expert 输出发回 home rank——而 home rank 正是下一层 attention 需要该 token 的 rank。layout 全程一致，边界 mismatch 为零，无额外 all-gather / reduce-scatter。第三步，MoE 块内只有 dispatch + combine 两次 all-to-all，故每层合计 2 次。∎ 正例： $d$ = $e$ = 8、 $B$ = 64 → 每 rank 8 条请求，dispatch 每 rank 每层发送 $8 \times 6 \times 5120 \times 2 = 491{,}520$ 字节（bf16）。边界反例：若 $a \gt 1$ （attention 仍 TP），attention 块内 1 次 all-reduce 不可免—— $d = e$ 只消边界通信，不消 attention TP 自身的归约；若 MoE 用 TP 而非 EP，边界 all-gather + reduce-scatter 不可免（字节恒等证明留作练习 3）。

边界对齐的数据流（world = 8， $d$ = $e$ = 8、 $a$ = 1，#DP = #EP）：

```
attention（DP，各 rank 独立算 B/8 条请求，KV 本地不复制）—— 0 次集合通信
  [r0] [r1] [r2] [r3] [r4] [r5] [r6] [r7]
   │    │    │    │    │    │    │    │
   └────┴────┴────┴── dispatch（EP all-to-all，每 token 发往 k=6 个 expert 所在 rank）
                      ▼
MoE（EP，expert 切 8 份，每 rank 持 E/e 个 + EPLB 冗余副本）
  [r0] [r1] [r2] [r3] [r4] [r5] [r6] [r7]
   ┌────┬────┬────┬── combine（EP all-to-all，输出发回 token 的 home rank）
   │    │    │    │    │    │    │    │
  下一层 attention：home rank = dispatch 源 = combine 目的 → 边界 0 次额外通信
```

MoE TP 形式的边界通信为什么等价 1 次 all-reduce？all-gather 把各 rank 的 $[B/d,\ d_{\mathrm{model}}]$ 聚成 $[B,\ d_{\mathrm{model}}]$ （TP FFN 的列切需要每 rank 持全量 token），reduce-scatter 把 down 投影的部分和归约并切回 $[B/d,\ d_{\mathrm{model}}]$ 。两者每 rank 字节各 $\frac{d-1}{d}\cdot B\thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}$ ，合计 $\frac{2(d-1)}{d}\cdot B\thinspace d_{\mathrm{model}} \cdot \mathrm{bytes}$ ——正是 training 4.1 式 (2) 一次 all-reduce 的每 rank 流量，也是 training 5.2 §2 的 SP 恒等（all-reduce = reduce-scatter + all-gather）。SGLang v0.4 blog 的原始形态即此：「MoE 前 all-gather、MoE 后 redistribute」（[SGLang Team, 2024](../references.md)，[C]）。

两种 MoE 形式的每 rank 每步发送字节（V4.1， $B$ = 64、 $d$ = $e$ = $t$ = 8、bf16 激活、 $L_{\mathrm{moe}}$ = 40 上界）：

$$
C_{\mathrm{rank}}^{\mathrm{EP}} = 2 \cdot \frac{B}{d} \cdot k \cdot d_{\mathrm{model}} \cdot \mathrm{bytes}_{\mathrm{act}} \cdot L_{\mathrm{moe}} \qquad (4)
$$

$$
C_{\mathrm{rank}}^{\mathrm{MoE\text{-}TP}} = \frac{2(d-1)}{d} \cdot B \cdot d_{\mathrm{model}} \cdot \mathrm{bytes} \cdot L_{\mathrm{moe}} \qquad (5)
$$

**符号**： $C_{\mathrm{rank}}$ = 每 rank 每步发送字节（MoE 部分，attention 块 0 次不计）；式 (4) 因子 2 = dispatch + combine 两程； $`\mathrm{bytes}_{\mathrm{act}}`$ = 激活每元素字节（bf16 = 2，FP8 dispatch = 1）；式 (5) 为 all-gather + reduce-scatter 合计（1 次 all-reduce 等价）。代入：式 (4) $= 2 \times 8 \times 6 \times 5120 \times 2 \times 40 = 39{,}321{,}600$ 字节 ≈ 39.3 MB；式 (5) $= 1.75 \times 64 \times 5120 \times 2 \times 40 = 45{,}875{,}200$ 字节 ≈ 45.9 MB。对照纯 TP（attention TP + MoE TP，每层 2 次 all-reduce）： $2 \times 1.75 \times 655360 \times 40 = 91{,}750{,}400$ 字节 ≈ 91.8 MB。三形式字节比 91.8 : 45.9 : 39.3——DP attention 把 attention 部分归零，总字节减半（MoE TP 形）到减 57%（EP 形）；MoE 部分本身在 MoE TP 形与纯 TP 的 MoE 部分相同（各 45.9 MB），EP 形再减 14%（39.3 vs 45.9）。

但 decode 的瓶颈是延迟不是字节（§2），诚实的暴露延迟读数须并列（本段 $B$ = 64、NVLink 900 GB/s、 $\tau_{\mathrm{lat}}$ = $\tau_{\mathrm{a2a}}$ = 15 μs）：纯 TP 每步 80 次归约 × 16.27 μs = 1.30 ms（全暴露）；DP + MoE TP 每步 40 次 all-reduce 等价 × 16.27 μs = 0.65 ms（attention 归约消失，减半）；DP + EP 每步 80 次 all-to-all × 15.48 μs = 1.24 ms（**未重叠上界**，次数与纯 TP 同——每 MoE 层 2 次）。EP 形式的次数不减，赢在三处：其一，字节更小且逐 peer 消息 KB 级（§4）；其二，dispatch/combine 可融合重叠（Mega MoE / TBO，§4），暴露部分可压向零，而 all-reduce 是层内屏障；其三，也是决定性的一笔——KV 读与容量。

KV 读与容量（命题 2 兑现）。V4.1 bf16 $\kappa$ = 1024 B/token/层、 $L$ = 40、 $T$ = 4096、 $B$ = 64：全 batch KV 10 GiB。TP $t$ = 8 每 rank 读全量 10.74 GB（3.35 TB/s 下 3.21 ms/step）；DP $d$ = 8 每 rank 读 $10/8 = 1.34$ GB（0.40 ms/step）——**每卡 KV 读降 8 倍**。容量同形：每 rank KV 驻留从 10 GiB 降到 1.25 GiB，腾出的 HBM 换更大 batch（6.3 §1 式 (3)）。长上下文 $T$ = 131072 时全量口径 KV 320 GiB：TP 每卡复制 320 GiB 超单卡 HBM、容量上不可行，DP 每卡 40 GiB 可容（V4.1 混合注意力实际更小，7.2；结构结论不变）。attention 权重在 DP（ $a$ = 1）下复制，但被复制的只是 attention 侧——参数大头 routed expert（单 expert 每层 18.8 MB × 384 × 40 ≈ 289 GB MXFP4，7.4 §2）仍按 $e$ 切分，复制代价有界。SGLang v0.4 blog 的实测回报：8×H100 DeepSeek-Coder-V2-Instruct-FP8 decode 吞吐 1.9×（[C]，v0.4 vs v0.3）——账面三笔（KV 容量、KV 读带宽、可重叠通信）合起来的经验兑现。

#DP = #EP 的生产口径（doc L33 逐字，官方披露）：「attention uses data parallelism (#DP = #EP)」。DeepInfer prefill EP2–8 / DP2–8（HBM 允许的最小）、decode EP64–128 / DP64–128（GPU 允许的最大，Wide-EP），doc L47-49（recap 7.4 §1）。耦合扩张的账面： $B = d \cdot B_{\mathrm{local}}$ ，每 expert batch $Bk/E$ 随全局 $B$ 涨（1.2 §5），每卡计算 $B_{\mathrm{local}} k$ 不变（7.4 §2）——Wide-EP 减每卡权重流量、耦合扩 DP 提强度，两件事同时做。

> **坑**：把 SGLang 的 `--enable-dp-attention` 当当前字段写（版本漂移）。触发条件：照 v0.4 blog 或旧文档锚定配置；现象：当前 main 无此布尔字段——它由 `attn_dp_size > 1` 派生（`server_args.py` L345/L355），且 `tp_size` = world size、`attn_tp_size = tp_size // attn_dp_size // attn_cp_size`（`runtime_context.py` derive_attn_tp_size L73）；规避：配置读 `attn_dp_size`，派生关系与 rank 布局走读见 6.3 §1（命题 1 双射），本坑 recap 该处不重展开。

> **坑**：把 DP attention 的 $d$ 当整模型 DP 副本数。触发条件：混淆 SGLang `dp_size`（整模型副本，每副本一套独立引擎）与 `attn_dp_size`（TP 组内部的 attention DP 宽度）；现象：以为每 rank 持完整模型副本、按整模型 DP 算显存；规避：DP attention 是 attention DP + MoE EP 的混合——attention 权重复制（ $a$ = 1）、expert 权重仍切 $e$ 份，不是整模型复制（6.3 §1 两字段定义）。

引擎落点 recap：SGLang 配置面派生与两套通信组（attention 族 / MoE 族）见 6.3 §1；vLLM 每 DP rank 一个独立 EngineCore + scheduler（DPEngineCoreProc / DPCoordinator）与跨 rank CUDA graph 协商见 5.3 §2；DeepInfer 的 CPManager（parallel_utils/cp.py L9）与 balanced CP 见 §6。

## 6. CP serving：三义区分与弹性序列并行

「CP」在 serving 文献里至少指三个不同的东西——把它们混为一谈，会同时读错 LoongServe 的弹性机制与 DeepInfer 的生产取舍。先立区分，再逐个给账。

> **坑**：CP 三义混用（必加）。触发条件：把训练 CP、LoongServe ESP、DeepInfer balanced CP 当同一机制引用；现象：用训练 ring attention 的固定度账去读 LoongServe 的每 iteration 动态 DoP，或用 DeepInfer release 支持 balanced CP 推断其生产用了 CP；规避：三者分清——训练 CP（training 5.2，ring attention，固定度，含反向）≠ LoongServe ESP（serving 弹性序列并行，DoP 每 iteration 变，[A]）≠ DeepInfer balanced CP（README L12 release 支持 vs doc L67 生产不用，两口径）。CP 度记 $c$ （spec §4.3），仅 LoongServe 对照、不入默认决策元组 $t/p/e/d$ 。

**训练 CP**（recap + 指引，不重推）：training 5.2 的 ring attention 沿序列维切 KV、环状流通，causal 负载均衡用 striped / zigzag 修法，Ulysses 用 all-to-all 换维；度数固定、含反向通信账。serving 侧不直接搬——推理无 backward，且 serving 的序列长与资源需求逐请求、逐阶段动态变化，固定度 CP 无法适配。

**LoongServe ESP**（elastic sequence parallelism，弹性序列并行；[Wu et al., 2024](../references.md)，arXiv:2404.09526，[A]，LoongServe/LoongServe 官方实现，2026-11-23 核实）。所谓 ESP，是指每 iteration 动态决定请求的并行度（Degree of Parallelism, DoP）的 serving 序列并行：prefill 阶段把 DoP 设到全集群 GPU 数加速长 prompt，转入轻量的 decode 阶段降 DoP 减通信、释放资源给其他请求（论文 §3）。执行单元是 elastic instance——持完整模型副本、内部用统一模型并行的独立单元；ESP 组跨 instance 组建，DoP = 组内 instance 数。为什么弹性走 SP 维而非 TP 维？改 TP 度要重启整个推理 runtime，分钟级甚至更久（论文 §2.4），latency-sensitive serving 不可接受；SP 维改 DoP 不需重切模型参数。

两个弹性机制各消一笔迁移成本：

- **scale-down = proactive migration**（论文 §4.1）。prefill 的 SP ring 本就在组内流通 KV——缩组时不等 prefill 结束再搬 KV（reactive migration），而是在流通过程中按新组 $R'$ 的放置计划选择性留存 KV，prefill 算完时 KV 已在新组各 instance 的 pool 里，零额外迁移通信。buffer 只需暂存单层 KV（论文 $O(bsh/d)$ 口径，其 $b/s/h$ = batch/序列长/hidden、 $d$ = DoP；译到本维度为 $B \cdot T \cdot d_{\mathrm{model}}/c$ 元素量级、单层），比纯模型并行的全层 buffer 小。对照 reactive migration 的两笔代价：1M-token 7B 模型 KV 超 488 GB，即便 NVLink/InfiniBand 高带宽，单请求迁移仍数秒（远超一个 decode step）；且碎片约束要求每 instance 有 $O(blsh/d)$ 空闲显存（论文记号， $l$ = 层数）——600K 请求 DoP=3、三 instance 空闲 100k/200k/400k slot 时，总量够但首 instance OOM。
- **scale-up = multi-master distributed decoding**（论文 §4.2）。加 instance 不迁现有 KV：组内多个 master instance 各持自己负责请求的 KV，attention 层 master 算 Q/KV、存本地 KV、把 Q 发给组内其余 instance 并行算 local attention 回传；FFN 跨 master 并行，Q 交换与 local attention 重叠。单 master 即可让整 batch 跨 instance 算（只要一个 instance 装得下新增 KV），多 master 进一步破碎片、提计算效率。

调度侧（论文 §5）：global manager + dispatcher + elasticity controller + 统一 KV cache pool。SIB（scaling information base）profiling + 解析模型估 prefill 迭代时间——常数项 + 线性项（FFN）+ 二次项（attention）三项，系数由少量 profiling 最小二乘拟合（论文式 (7) 的三项系数为论文局部记号，本维度 $\alpha$ / $\gamma$ 另有专归，故以文字 recap 不沿用其字母）；DP batching 用 prefix sum 在多项式时间求最优分批与 DoP；decode DoP 取启动时最小 best DoP（模型并行度），运行时只缩到 KV 装得下的最小 DoP。评估（论文 §7）：吞吐至多 3.85× vs chunked prefill（LightLLM SplitFuse）、5.81× vs DistServe（4P+4D，8.2）、4.64× vs vLLM。口径注：ESP 是学术方案（长上下文 serving），与 DeepInfer 生产取舍是不同口径；LoongServe 论文 v2（2024-10）/ repo 2024-11 稳定，机制无漂移（W5 核实）。

**DeepInfer balanced CP：release 支持 vs 生产不用**（两口径，必分清）。README L12 逐字：「Parallelism: Attention DP / balanced CP, MoE EP + EPLB」——release 支持 balanced CP（代码侧 parallel_utils/cp.py CPManager L9，7.1 走读）。但 doc L53「TP and CP are not used for now」、L67 给生产不用的 rationale（官方披露，2026-11-23 核实）：

- **主因**（doc L67 逐字）：「The main reason we do not currently use it is that V4.1 is fast enough to keep tail TTFT acceptable even on a complete cache miss.」——CP 的收益是切长序列降 tail TTFT，而 V4.1 足够快、完全 cache miss 下 tail TTFT 已可接受，收益不必要。
- **代价面**（非决策首因）：e2e 至多慢 5%（doc 称「reasonable tradeoff」——慢 5% 换掉原 DP load-balancing 方案的需要）；CP 虽改善 DP 均衡，却 tends to worsen expert-parallel load balance；另加 kernel、communication、CPU 开销，仍在优化。

教学结论落在「CP 取舍是模型/负载依赖的」：长上下文、TTFT 敏感、prefill 慢的模型，CP 降 tail TTFT 的收益可能压倒代价（LoongServe 的适用域）；V4.1 足够快、EP 均衡优先，CP 收益不必要、代价反而突出——所以不用。**不是「CP 慢 5% 所以不用」**（5% 是 doc 称合理交换的代价面，主因是收益不必要，四路审查 C-S2-3）。

引擎配置面 recap：vLLM 的 CP 度分 prefill_context_parallel_size / decode_context_parallel_size 两字段（两阶段可不同度，5.3 §2）；SGLang 的 attn_cp_size 与派生整除断言见 6.3 §1。本章 $c$ 仅 LoongServe 对照用，默认决策元组 $t/p/e/d$ 不含 $c$ （§7）。

## 7. 组合决策：模型规模 × 上下文 × SLO × 拓扑 → t/p/e/d

五个度数怎么联合取值？training 5.5 的组合规则（world size 乘积约束 + group 树 + TP 进节点/PP DP EP 跨节点）serving 侧怎么对照？先 recap 约束，再给决策表。

recap training 5.5 §3（不重推）：dense 路径 world $= t \cdot c \cdot p \cdot d$ ，MoE 路径 world $= t' \cdot p \cdot e \cdot d_e$ ，相容条件 $tcd = t'ed_e$ ；每 rank 同时属每层一个组，通信只在组内。serving 侧的两条修正：其一，无 backward → TP 通信账从带宽主导翻成延迟主导（§2），TP 的度数上限不止显存/整除，还有 TPOT 预算；其二，MLA 模型 attention 侧用 DP 而非 TP（§5），SGLang 的派生面 $\mathrm{attn\_tp\_size} = \mathrm{tp\_size}/(\mathrm{attn\_dp\_size}\cdot\mathrm{attn\_cp\_size})$ （6.3 §1 式 (1)）正是「world 固定、attention 与 MoE 两套划分」的落地——attention 族 $d \cdot c \cdot a$ （ $a$ = attention TP 度）、MoE 族 $e \cdot d_{\mathrm{moe}}$ （ $d_{\mathrm{moe}}$ = MoE 侧 DP 度，即 moe_dp_size）各铺满 world。PD 分离再引入 $n_P/n_D$ 实例配比维度（8.2），不在本节元组内。

组合决策表（每行一条可判定论断 + 回指； $t/p/e/d$ = TP/PP/EP/DP 度， $c$ 仅 LoongServe 对照）：

| 场景（模型 × 上下文 × SLO × 拓扑） | $t$ | $p$ | $e$ | $d$ | 论断依据 | 回指 |
| --- | --- | --- | --- | --- | --- | --- |
| 小 dense（≤8B），单节点，chat SLO 宽松 | 1 | 1 | — | 8 | 单卡装得下，纯 DP 副本免层内归约；TP 延迟账无收益 | §2、命题 1 |
| 小 dense，TPOT 紧（单卡步时超预算） | 2–4 | 1 | — | $8/t$ | TP 切层内计算降步时，付 $2L$ 次归约；域内、整除 head 数 | §2、training 5.1 §7 |
| 大 dense（70B+），单节点 HBM 不足 | 8 | 1 | — | 1 | TP 域内切权重/激活； $t$ ≤ 域内卡数 | §2、NVLink 边界 |
| 大 dense，跨节点 | 8 | >1 | — | 跨节点 DP | TP 收节点内，PP 切层跨节点；TPOT 加 $(p-1)\tau_{\mathrm{p2p}}$ | §3、式 (3)、training 5.5 §2 |
| MoE MLA，prefill | 1 | 1 | 2–8 | $=e$ | 每 expert batch 已够大，最小 EP/DP 改善均衡 + 减 NVLink 流量 | §4、doc L47/L57、7.4 §1 |
| MoE MLA，decode（Wide-EP） | 1 | 1 | 64–128 | $=e$ | 每步每序列 1 token，最大 EP/DP 减每卡权重流量 + 耦合扩 DP 提强度 | §4-5、doc L49、7.4 §2 |
| 长上下文（128K+），TTFT 敏感，prefill 慢 | 1 | 1 | 按 MoE | 按上 | CP/ESP 切序列降 tail TTFT（ $c$ 仅 LoongServe）；或 chunked prefill 替代 | §6、4.3 |
| 长上下文，V4.1 类足够快 | 1 | 1 | 64–128 | $=e$ | $c$ = 1：CP 收益不必要（tail TTFT 已可接受）+ worsens EP balance | §6、doc L67 |
| GQA 宽 KV head（ $H_{kv} \ge t$ ） | 可选 >1 | 按容量 | 按 MoE | 按上 | KV head 可切、TP 不复制 KV，DP attention 必要性降 | 命题 2 边界、training 5.1 §8 |

DeepInfer 生产组合是这张表的一个实例： $t$ = 1、 $c$ = 1、 $p$ = 1（用 PD 分离而非 PP 切阶段，doc L31/L47-53），prefill $e$ = $d$ = 2–8、decode $e$ = $d$ = 64–128。选择原则（doc L73 逐字，官方披露）：「maximize utilization while meeting reasonable TTFT and TPS targets」；过程（L75）：分析各策略对 kernel 性能的影响、核算负载均衡代价、离线 benchmark 验证。这与 training 5.5 §6 的三元权衡（显存把度数往大推、通信往小拉）同构，serving 侧再叠加 SLO 约束与 PD 配比。

三引擎配置面对照（回链 5.3 / 6.3，D-S3-1）：vLLM 显式字段（tensor_parallel_size / pipeline_parallel_size / data_parallel_size / enable_expert_parallel + CP 两字段，5.3 §2）；SGLang 派生面（tp_size = world size，attn_dp_size / attn_cp_size / ep_size 派生 attn_tp_size 与 moe 组，6.3 §1）；DeepInfer doc 口径（EP/DP 取值 + TP/CP 暂不用，7.4 §1）。三套配置面不可互译（5.3 §2 坑注）。

通信硬件与拓扑的边界（纯文字指引）：本章只讲传输语义（all-reduce / all-to-all / all-gather / reduce-scatter / P2P 的归约与投递语义）、带宽/延迟账与组合决策。NVLink / 灵衢 / RDMA 的硬件实现、NCCL 集合通信库的算法选择与拓扑探测、交换机与网络协议，归 cluster 维度；scale-up domain 的 serving 侧口径已在 1.3 §3 立（域内高带宽低延迟、域外 RDMA 低一个量级），本章只用其结论做度数约束，不展开硬件层。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| serving 无 backward → TP 每层 2 次前向归约 × $L$ ，每步 $2L$ 次（非 $L$ 、非 4） | 命题 1； $L$ = 40 → 80 次 | §2、练习 1 |
| decode TP 消息 $2B d_{\mathrm{model}}$ ，延迟项主导； $B$ = 60 每步 1.30 ms，带宽项仅 7.4% | 式 (1)(2)；614,400 B、1.19 μs | §2、练习 1 |
| TP 留 scale-up domain 内；跨节点带宽项 21.5 μs 超延迟项；DeepInfer 装得下 + SLA 可达不付 TP | 式 (2)；doc L69 | §2 |
| prefill PP 气泡占比 $(p-1)/(m+p-1)$ 与训练同形；无 backward 激活驻留， $m$ 不受训练显存约束；气泡 vs tail TTFT 同硬币 | training 5.3 式 (5) recap； $p$ = 4 $m$ = 8 → 27.3% | §3 |
| decode PP TPOT 加 $(p-1)\tau_{\mathrm{p2p}}$ （层间 1 跳 vs TP 层内 $2L$ 次）；PP 买容量非单请求延迟 | 式 (3)； $p$ = 2 开销 5 μs vs TP 1.30 ms | §3、练习 4 |
| PP 沿层切 KV 读、DP 沿请求切、TP（MLA $H_{kv} \lt t$ ）复制 KV 读与 $t$ 无关 | 命题 2；TP 每卡 10.74 GB vs DP 1.34 GB | §3、§5、练习 4 |
| EP decode 逐 peer KB 级延迟主导；每步 $2L_{\mathrm{moe}}$ 次 a2a 未重叠上界 1.24 ms，可融合重叠 | 3.3 式 (6) recap；30 KiB、0.61 μs | §4 |
| 木桶 $\max_j R_j$ 落成 tail TPOT；EPLB 冗余副本 + 统计窗 + 重分配兜底（moe_lb 残差偏斜） | training 5.4 recap；7.4 §3 | §4 |
| DP attention： $d=e$ 边界对齐 → attention 0 次 + MoE 2 次 a2a；MoE TP 形式边界 AG+RS = 1 AR 等价 | 命题 3；式 (4)(5) | §5、练习 3 |
| 三形式每 rank 每步字节 91.8 : 45.9 : 39.3 MB；EP 赢在 KV 读/容量 8× + 可重叠，非次数 | 式 (4)(5)；KV 3.21 → 0.40 ms | §5、练习 2 |
| CP 三义：训练 ring CP ≠ LoongServe ESP（每 iteration DoP）≠ DeepInfer balanced CP（release 支持 vs 生产不用） | §6 坑注；arXiv:2404.09526 | §6 |
| DeepInfer 不用 CP 主因 V4.1 足够快（tail TTFT 可接受），非「慢 5%」；CP 取舍模型/负载依赖 | doc L67 逐字 | §6 |
| 组合：world 乘积约束 + TP 域内 + MLA attention DP；PD 引入 $n_P/n_D$ 归 8.2 | training 5.5 recap；决策表 | §7、8.2 |

## 练习

1.（计算 · [10]）TP decode 消息与延迟项代入（章脊口径 $L$ = 40、 $d_{\mathrm{model}}$ = 5120、 $t$ = 8、NVLink 900 GB/s、 $\tau_{\mathrm{lat}}$ = 15 μs）：(a) 按式 (1) 算 $B$ = 60 与 $B$ = 1 的单次归约消息字节；(b) 按式 (2) 算两档的带宽项与每步 $\tau_{\mathrm{TP}}$ ；(c) 先预测 $B$ 从 60 缩到 1 时 $\tau_{\mathrm{TP}}$ 降百分之几，再计算对照。校验：(a) 614,400 / 10,240 字节；(b) 带宽项 1.19 / 0.02 μs、每步 1.30 / 1.20 ms；(c) 降幅约 7%（延迟项与消息无关）——与 §2「消息缩 60 倍时间只降 7.25%」一致。
2.（计算 · [20]）DP attention 三形式账（V4.1， $B$ = 64、 $d$ = $e$ = $t$ = 8、bf16 激活、 $L_{\mathrm{moe}}$ = 40 上界、 $\kappa$ = 1024 B/token/层、 $T$ = 4096）：(a) 按式 (4)(5) 与纯 TP 口径算三形式每 rank 每步 MoE 发送字节，验证比 91.8 : 45.9 : 39.3；(b) 算 TP 与 DP 的每 rank 每步 KV 读字节及 3.35 TB/s 下时间，验证 8×；(c) 把 $T$ 换 131072 重算 KV 总驻留，说明 TP 为何容量上不可行（先预测再算）。校验：(a) 91.75 / 45.88 / 39.32 MB；(b) 10.74 / 1.34 GB、3.21 / 0.40 ms；(c) 320 GiB，TP 每卡复制超 80 GB HBM、DP 每卡 40 GiB（全量口径上界，V4.1 混合注意力实际更小，7.2）。
3.（推导 · [30]）命题 3 闭环（DP↔EP 边界）：(a) 补全 MoE TP 形式的字节恒等证明——all-gather 每 rank 接收 $\frac{d-1}{d} B d_{\mathrm{model}} \cdot \mathrm{bytes}$ 、reduce-scatter 每 rank 发送同量，合计等于 training 4.1 式 (2) 一次 all-reduce 的每 rank 流量，并指出延迟上为何等价（training 5.2 §2 的 AR = RS + AG 分解）；(b) 给边界反例： $a \gt 1$ 时为什么 attention 块内归约不可免， $d = e$ 消的是哪一段通信；(c) 若 $d \ne e$ （如 attention DP 8、MoE EP 4），追踪 token home rank 与 EP rank 的 layout mismatch，说明边界需要什么额外原语。校验：(a) 合计 $\frac{2(d-1)}{d} B d_{\mathrm{model}} \cdot \mathrm{bytes}$ ，与式 (5) 一致；(b) 对回 §5 证明第二步「home rank = dispatch 源 = combine 目的」；(c) 需 all-gather 或带复制的 a2a 重对齐（对照 6.3 §1 两套划分）。
4.（推导 · [30]）decode PP 账与组合决策：(a) 按式 (3) 算 $p$ = 2 / $p$ = 4 的 TPOT 开销（ $\tau_{\mathrm{p2p}}$ = 5 μs），与 §2 TP 每步 1.30 ms 对照，说明「层间 $p-1$ 次 vs 层内 $2L$ 次」的结构差；(b) 用命题 2 算 PP $p$ = 2、 $m$ = 2 每 stage 每 micro-batch KV 读，与 TP 每 rank 对照（ $B$ = 64、 $T$ = 4096）；(c) 给「MoE MLA、decode、长上下文、TPOT 紧、单超节点 64 卡」场景选 $t/p/e/d$ 并逐维说明依据，指出为什么 $c$ = 1（V4.1 类）而 $t$ = 1。校验：(a) 5 / 15 μs ≪ 1.30 ms；(b) 2.5 GiB vs 10 GiB（1/4）；(c) $t$ = 1（doc L69 + 命题 2 KV 复制）、 $p$ = 1（PD 分离替代）、 $e$ = $d$ = 64（doc L49 Wide-EP + #DP=#EP）、 $c$ = 1（doc L67），对回 §7 决策表。

## 参考

- [Wu et al., 2024](../references.md)：LoongServe，arXiv:2404.09526，[A]（key: loongserve，LoongServe/LoongServe 官方实现，论文自链，2026-11-23 arXiv v2 API + repo tarball f6e8fc15 核实）——ESP 每 iteration 动态 DoP（§3）、elastic instance、proactive migration（§4.1，零额外迁移 + 单层 buffer + 1M-token 7B KV 488 GB 数秒 + 600K DoP=3 碎片例）、multi-master distributed decoding（§4.2）、global manager + SIB 解析模型（§5 式 (7) 常数+线性+二次）+ DP batching prefix sum、改 TP 度重启分钟级（§2.4）、3.85× / 5.81× / 4.64×（§7）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——#DP=#EP（L33）、prefill EP2-8/DP2-8 与 decode EP64-128/DP64-128（L47-49）、TP/CP 暂不用（L53）、prefill 最小 rationale（L57）、CP 主因 V4.1 足够快 + 代价面 ≤5% + worsens EP balance（L67）、TP 装得下+SLA 可达不付通信（L69）、选择原则 maximize utilization（L73-75）、EP>32 递减 + 整超节点单 EP 组（L293-297）、PD 分离 TTFT/TBT 独立优化（L31）
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——README L12「Parallelism: Attention DP / balanced CP, MoE EP + EPLB」（release 支持 balanced CP，与 doc L67 生产不用两口径）、parallel_utils/cp.py CPManager L9；main@a679bb86 tarball 2026-11-23（W4 同日复用）
- [SGLang Team, 2024](../references.md)：SGLang v0.4 blog，key: sglang-v04-blog，[C]（开放技术博客、非闭源，2026-11-23 全文核实）——DP attention for DeepSeek MLA（单 KV head TP8 重复 KV → attention DP，MoE 前 all-gather / 后 redistribute，8×H100 DeepSeek-Coder-V2-Instruct-FP8 1.9× decode 吞吐）；--enable-dp-attention 为历史 flag，当前由 attn_dp_size>1 派生
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：key: v41-flash-config，[W]（hf-mirror 传输核实，2026-11-23）——L 40 / d_model 5120 / H_kv 1 / head_dim 512（含 rope 64，kv_lora_rank 448 引擎导出）/ E 384+1 / k 6 / moe_intermediate_size 2304；模型侧形式化归 models（V4.1 待补）
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：key: nvidia-datasheet，[C]（官方规格披露，2026-11-23 PDF 一手核实）——NVLink 900 GB/s（H100/H200）/ 1.8 TB/s（B200）、HBM 3.35 TB/s（H100）；参数表出处见 1.3，本章只取带宽值
- vLLM / SGLang repo（key: vllm-repo / sglang-repo，[A]，2026-11-23 同日 tarball 复用）：分布式执行结构与配置面走读归 5.3 / 6.3，本章 recap + 指针——vLLM parallel_state.py GroupCoordinator L427 / 三条归约路径 / DPEngineCoreProc / AsyncIntermediateTensors / CP 两字段；SGLang runtime_context.py derive_attn_tp_size L73 / 两套通信族 / scheduler_pp_mixin / attn_dp_size 派生
- training 5.1 / 5.2 / 5.3 / 5.4 / 5.5（纯文字指引，不挂链接）：TP f/g 共轭与通信账（5.1 §6 式 (9)-(11)、§8 GQA/MLA 复制）、SP 恒等 AR=RS+AG（5.2 §2）、PP 气泡（5.3 §2 式 (5)）、EP all-to-all 与木桶（5.4 §1-2/§4）、组合乘积约束与决策流程（5.5 §3/§6）——训练账 recap 不重推
