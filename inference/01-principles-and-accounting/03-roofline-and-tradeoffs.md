# 1.3 Roofline 与延迟-吞吐权衡（Roofline and the Latency-Throughput Tradeoff）

> 同一张卡，为什么凑 batch 能提吞吐却必然抬高每 token 延迟，而排队延迟又在利用率逼近 1 时爆炸？本章把 models 2.4 §6 的 roofline 入口接上官方 datasheet 参数，推出 batch 的三方权衡曲线与 SLO 约束下的操作点，再补排队论这块容量规划的数学地基 | 前置：1.1（TTFT / TPOT / goodput 定义）、1.2（prefill / decode 算力带宽账、临界 batch）、models 2.4 §6（算术强度式 (10)-(12)）、models 1.2 §10（强度 $\propto B$）；排队论直觉章内建立 | 锚点：NVIDIA H100 / H200 / B200 官方 datasheet [C]（官方规格披露）、Ascend 950 官方 keynote / 产品页 [C]（官方披露）、DistServe [A]（arXiv:2401.09670）

## 1. roofline 入口 recap 与本章口径

decode 凑 batch 提吞吐，天花板在哪里？models 2.4 §6 已交付答案的骨架：算术强度 $I$ = FLOPs ÷ 访存字节数，roofline 给出的可达吞吐上限是 $\min(F_{\mathrm{peak}},\ I \cdot \beta)$ ，分界 ridge point $R = F_{\mathrm{peak}} / \beta$ ；prefill 强度 $I \approx T_p$ 远在 ridge 右侧（compute-bound），decode 强度 $I \approx B$ 且被长上下文 KV 读封顶于 $(2N + 4LTd_{\mathrm{model}})/(\kappa LT)$ ，全窗口落在 ridge 左侧（memory-bound）。这三条（models 2.4 式 (10)-(12)）本章只 recap、不重推；符号按本维度约定翻译——models 的 $d$（hidden 宽）译作 $d_{\mathrm{model}}$ 、models 的 $b$（每元素字节数）译作文字「字节」、models 的 $t$（已缓存长度）译作 $T$ 。强度 $\propto B$ 与饱和上界的完整推导见 models 1.2 §10 与 1.2，本章不复述。

本章增量有四块，全部是 models 入口之外的 serving 侧事实与推导：其一，**datasheet 参数表**——ridge 与吞吐上限里的 $F_{\mathrm{peak}}$ 和 $\beta$ 取官方规格值（微架构来源与测量方法归 gpu 2.1 / 2.4 / 11.2，本章只引规格事实）；其二，**batch 三方权衡**——把「decode memory-bound」翻译成吞吐、TPOT、TTFT 排队三条随 $B$ 变化的曲线；其三，**排队论**——利用率 $\rho$ 与延迟爆炸的定量关系；其四，**SLO 操作点**——goodput 最大化与吞吐最大化为什么不是同一个选择。

**本章口径**（R2-2 章脊锁定）：模型主锚为 8B dense（Llama-3.1-8B 几何： $L = 32$ 、 $d_{\mathrm{model}} = 4096$ 、 $H_{kv} = 8$ 、 $d_{\mathrm{head}} = 128$ 、 $N = N_{\mathrm{full}} \approx 8.03 \times 10^{9}$ ），权重 bf16（每参数字节数 2），上下文 $T = 4096$ 、输入 $T_{\mathrm{in}} = 4096$ 、输出 $T_{\mathrm{out}} = 512$ ；主硬件 H100 SXM，H200 / B200 作 datasheet 对照（换卡是本章参数表的目的，模型口径不变）。**$N$ 口径**：本章 $N = N_{\mathrm{full}} = 8.03 \times 10^{9}$ 为总参数量 / 存储口径（含输入 embedding，权重字节用），与 1.2 FLOPs 用的 matmul 口径 $N = 7.50 \times 10^{9}$（层参数 + 输出头）区分——两者差约 7%，跨章 prefill 近似的一致性见 §4 式 (5) 口径注。 $B$ = decode batch 内并发序列数（每序列每步产 1 token，故 batch token 数亦为 $B$）。**本章 $\lambda$ = 请求到达率（req/s），非 training GAE 的 $\lambda$ 参数**； $\mu$ = 服务率， $\rho = \lambda/\mu$ = 利用率。TTFT / TPOT 用文字记法 $\mathrm{TTFT}$ / $\mathrm{TPOT}$ ，goodput 记 $\mathcal{G}$ 。TPOT SLO 上限记 $\tau$（§4；**与 1.4 采样温度 $\tau$ 不同章不同义**）。bytes 不占数学字母，一律文字「字节」或 $\mathrm{bytes}$ 。

先把本章反复用到的两个字节量算死。权重字节 $W = 2N = 1.606 \times 10^{10}$ 字节（约 16.06 GB）。每序列每步 KV 读字节 $K = \kappa L T$ （ $K$ 即单序列 KV cache 字节量，recap models 2.4 式 (9) 的 $B = 1$ 口径，本章用作 decode 每步读字节；容量公式不重推），其中 GQA 每 token 每层 KV 字节 $\kappa = 2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes} = 2 \times 8 \times 128 \times 2 = 4096$ 字节（K、V 两份，bf16），故 $K = 4096 \times 32 \times 4096 = 5.37 \times 10^{8}$ 字节 = 0.5 GiB。这两个数（ $W \approx 16.06$ GB、 $K = 0.5$ GiB）是后续全部操作点算例的输入。

## 2. 硬件参数表：datasheet 事实与 ridge point

ridge point 到底是多少？答案不在论文里，在官方 datasheet 里。下表全部取 NVIDIA 官方 datasheet 的 **dense** 值（[NVIDIA H100/H200/B200 datasheet](../references.md)，[C] 官方规格披露，2026-11-23 PDF 一手核实）。官方规格表以 sparse 口径印刷，脚注原文 "Specifications in sparse. Dense is one-half of the sparse spec shown."——dense 恰为表值之半，本章一律用 dense（serving 的 GEMM 不依赖结构化稀疏，dense 才是可达算力口径）。

| 参数（dense） | H100 SXM | H200 SXM | B200（HGX，SM100） |
| --- | --- | --- | --- |
| BF16 / FP16 Tensor Core | 989.5 TFLOPS | 989.5 TFLOPS | 2250 TFLOPS |
| FP8 Tensor Core | 1979 TFLOPS | 1979 TFLOPS | 4500 TFLOPS |
| FP4 Tensor Core | 无（Hopper 不支持） | 无 | 9000 TFLOPS |
| TF32 Tensor Core | 494.5 TFLOPS | 494.5 TFLOPS | 1100 TFLOPS |
| FP32 | 67 TFLOPS | 67 TFLOPS | 75 TFLOPS |
| 显存 | 80 GB HBM3 | 141 GB HBM3e | 180 GB HBM3E |
| 显存带宽 $\beta$ | 3.35 TB/s | 4.8 TB/s | 7.7 TB/s |
| NVLink | 900 GB/s（NVLink4） | 900 GB/s | 1.8 TB/s（NVLink5） |
| TDP | ≤700 W | ≤700 W | ≤1000 W 可配 |

**SKU 与代际纪律**（口径注，违反即张冠李戴）：B200 指 HGX B200（180 GB / 7.7 TB/s），与 GB200（Grace 配对、186 GB / 8 TB/s、频率更高、FP4 dense 10 PFLOPS）是**不同 SKU**，本章参数表只用 B200 列，GB200 仅作对照注、不入表混算；B300（Blackwell Ultra，288 GB）与 Rubin（SM107）是**后续代际**，勿与 B200 混；H200 NVL 变体（BF16 1671 / FP8 3341 TFLOPS sparse、141 GB、4.8 TB/s、TDP up to 600 W configurable，H200 官方 datasheet NVL 列）勿与 H200 SXM（700 W）混。Hopper（H100 / H200）无 FP4 硬件支持，FP4 行只对 Blackwell 起成立。带宽单位 TB/s 按 $10^{12}$ 字节/s、显存 GB 按 $10^{9}$ 字节（datasheet 十进制口径）；GiB 按 $2^{30}$ ，混用处逐一注明。

ridge point 由表直接算。单位对账：TFLOPS ÷ (TB/s) = $(10^{12}\ \mathrm{FLOP/s}) / (10^{12}\ \mathrm{字节/s})$ = FLOP/字节，量纲恰好消去：

$$
R = \frac{F_{\mathrm{peak}}}{\beta} \qquad (1)
$$

**符号**： $F_{\mathrm{peak}}$ = 该精度 dense 峰值算力（FLOP/s）； $\beta$ = 显存带宽（字节/s）； $R$ = ridge point（FLOP/字节）。代入式 (1)：

| ridge（FLOP/字节） | H100 SXM | H200 SXM | B200 |
| --- | --- | --- | --- |
| BF16 | $989.5/3.35 \approx 295$ | $989.5/4.8 \approx 206$ | $2250/7.7 \approx 292$ |
| FP8 | $1979/3.35 \approx 591$ | $1979/4.8 \approx 412$ | $4500/7.7 \approx 584$ |
| FP4 | — | — | $9000/7.7 \approx 1169$ |

这张表读出三条 serving 结论。第一，**H200 对 H100 是纯带宽升级**：算力同为 989.5 TFLOPS，带宽 3.35→4.8 TB/s，ridge 从 295 降到 206——compute-bound 的 prefill 上限不变，memory-bound 的 decode 上限随带宽线性抬升。第二，**B200 对 H100 算力与带宽近乎同比放大**（BF16 ×2.27、带宽 ×2.3），ridge 几乎不动（295→292）：换 B200 同时买到了 prefill 算力与 decode 带宽。第三，**精度越低 ridge 越高**：FP8 ridge 约为 BF16 的 2 倍、FP4 约 4 倍，因为峰值算力翻倍而带宽不变。结合 1.2 的 decode 饱和上界（8B 满窗约 4.9、本章 $T = 4096$ 口径约 34），在本章及更长上下文口径下，decode 强度饱和上界远低于任何卡的 ridge（最低 H200 BF16 约 206）——**decode memory-bound，换卡对 decode 的收益几乎全来自带宽项**。须注意这是长上下文结论： $T$ 很小时饱和上界 $(2N + 4LTd_{\mathrm{model}})/(\kappa LT)$ 随 $T$ 减小而抬高，大 $B$ 下可能转入 compute-bound（临界 $B^{\ast}$ 的存在条件归 1.2）。量化（FP8 / FP4）降的是权重字节 $W$ ，从而缩短 decode 步时（格式账归 3.2），不是把 decode 推过 ridge。

> **坑**：拿官方 sparse 表值当 dense 算力代入 roofline，会把 ridge 与 prefill 上限系统性高估一倍——现象是「实测 prefill 吞吐只有理论值一半」的假矛盾。规避：认准脚注 "Dense is one-half of the sparse spec shown."，serving 账一律取 dense。

roofline 的形状与两阶段定位如下图（示意，非按数值比例）：

```
 可达吞吐上限
   ▲
   │              ┌──────────────────  F_peak（compute-bound 平台）
   │             /│
   │            / │
   │           /  │  斜率 = β（memory-bound 斜线）
   │          /   │
   │         /    │
   │        /     │
   │       /      │
   │      /       │
   │     /        │
   │    /         │
   │   /          │
   │  /           │
   │ /            │
   │/             │
   └──────────────┼────────────────────────►  I（FLOP/字节）
                  R = F_peak / β
   decode ●（I≈1–34，左斜线，带宽决定）
   prefill ●（I≈T_p≫R，右平台，算力决定）
```

## 3. Ascend 950 与 scale-up domain

非 NVIDIA 硬件的规格怎么引？原则只有一条：**官方披露逐字引，口径未注明就不推定**。Ascend 950 系（Huawei Connect 2025-09 keynote，[Huawei Connect 2025 keynote / hiascend 产品页](../references.md)，[C] 官方披露，2026-11-23 核实）的芯片级官方数如下，注意每一处的口径限定。

keynote 逐字要点：950PR 与 950DT 同 Ascend 950 Die；算力 "deliver 1 PFLOPS in FP8, MXFP8, and HiF8, and 2 PFLOPS in MXFP4"；互联 "2 TB/s interconnect bandwidth, which is 2.5 times higher than the Ascend 910C"；SIMD+SIMT 混合架构；访存粒度 512B→128B；HiF8 为华为自有格式（精度近 FP16、效率比肩 FP8）。**keynote 未注明 1 PF FP8 / 2 PF FP4 的 dense/sparse 口径**——本章只引 keynote 口径，不推定它是 dense 还是 sparse。

| 变体 | 定位 | 显存 / 带宽 | 上市 | 出处与口径 |
| --- | --- | --- | --- | --- |
| 950PR | prefill + 推荐 | 最大 128 GB、1.6 TB/s（HiBL 1.0 自有低成本 HBM） | 2026Q1 | hiascend 官方页逐字「最大配备128GB、1.6TB/s高速片上内存」 |
| 950DT | decode + 训练 | 144 GB、4 TB/s（HiZQ 2.0） | 2026Q4 | keynote 逐字 "delivering 144 GB of memory and 4 TB/s memory access bandwidth" |

hiascend 官方产品页（950PR tab）另有一句「支持多种精度格式，最大支持1784TFLOPS」——**精度格式未注明，只可逐字引用、不可解释**为某种精度的 dense 值；同页「支持灵衢2.0互联协议，最大支持2TB/s双向互联带宽」。Atlas 950 SuperPoD = 8192 × 950DT、8 EFLOPS FP8 / 16 EFLOPS FP4、16 PB/s 互联（keynote 口径，2026Q4）。

> **坑**：把 Ascend 950 的「1 PF FP8」与 NVIDIA B200 的「4500 TFLOPS FP8 dense」直接同表比大小，会得出错误结论——前者 dense/sparse 口径官方未注明，后者明确 dense，两者不可通约。规避：Ascend 单独成表、标注「口径未注明」，不与 NVIDIA dense 数并列排名；同理 hiascend「1784TFLOPS」精度未知，不得代入式 (1) 算 ridge。

正因算力口径未知，**本章不给 Ascend 的 ridge 数值**：式 (1) 的分子 $F_{\mathrm{peak}}$ 口径不明，算出的任何 ridge 都是推定。带宽侧 950PR 1.6 TB/s、950DT 4 TB/s 是官方明确的，可作 decode 带宽账的输入；算力侧待官方补全 dense/sparse 口径后再算。另须区分**芯片级与卡级**：Atlas 350 是基于 950PR 的加速卡（2026-03 上市），媒体广泛报道的卡级数（1.56 PF FP4 / 112 GB / 1.4 TB/s / 600 W）与芯片级数不同，且本轮未取得华为官方卡级 datasheet——按 spec §2 闭源与生产披露处理条款，**媒体数一律不作事实来源**，本章只用芯片级官方数。

**scale-up domain**（口径：一组经高带宽紧耦合互联、可近似视为单一超大加速器的 GPU/NPU 集合，与跨机 scale-out 网络相对）。所谓 scale-up domain，是指域内任意两卡间走 NVLink / 灵衢这类高带宽低延迟互联，域外走 RDMA 网络（带宽通常低一个量级）。datasheet 给出的域参数：H100 / H200 NVLink4 900 GB/s（典型 8 卡 HGX 域）；B200 NVLink5 1.8 TB/s；GB200 NVL72 = 72 GPU 单 NVLink 域（datasheet Total NVLink Bandwidth 130 TB/s）；Atlas 950 SuperPoD = 8192 NPU（UnifiedBus / 灵衢，16 PB/s 互联）。这个概念为 Wide-EP 立口径：MoE 的 all-to-all dispatch/combine 在域内走 NVLink、跨域走 RDMA，故 EP 度 $e$ 的经济上限受 scale-up domain 大小约束——域越大，Wide-EP 越能在高带宽域内摊开 expert（每卡权重流量账见 1.2，系统展开归 7.4 / 8.1）。互联硬件与拓扑细节归 gpu（NVLink 域 → 7.1，机间 → 10.1；灵衢 / UnifiedBus 非 NVIDIA，沿本章 [C] keynote 口径只引规格事实，不做微架构展开），本章只立 serving 侧口径。

## 4. batch 三方权衡：吞吐、TPOT、TTFT 排队

把 $B$ 从 32 加到 64，吞吐、每 token 延迟、首 token 排队各怎么变？本节把 §1 的「decode memory-bound」翻译成三条可算的曲线。先立 decode 步时模型：memory-bound 区每步耗时由访存字节除以带宽决定（roofline 斜线区），权重读一遍、KV 逐序列读 $B$ 份：

$$
t_{\mathrm{step}}(B) = \frac{W + B \cdot K}{\beta} \qquad (2)
$$

**符号**： $W$ = 权重字节（§1，16.06 GB）； $K$ = 每序列每步 KV 读字节（§1，0.5 GiB）； $\beta$ = 显存带宽； $B$ = 并发序列数。口径注：式 (2) 是 roofline 理想下界（取峰值带宽、忽略 kernel launch 与调度开销），实测步时在其上，操作点须按实测校准（9.1）。代入 H100（ $\beta = 3.35$ TB/s）： $B = 32$ 时 $t_{\mathrm{step}} = (16.06 + 32 \times 0.537)/3350$ ms $= 33.24/3350$ ms $\approx 9.92$ ms； $B = 64$ 时 $\approx 15.05$ ms。

每序列每步产 1 token，故每输出 token 延迟就是步时；吞吐是 batch 内全部序列的产 token 速率：

$$
\mathrm{TPOT}(B) = t_{\mathrm{step}}(B) = \frac{W + BK}{\beta} \qquad (3)
$$

$$
X(B) = \frac{B}{t_{\mathrm{step}}(B)} = \frac{\beta B}{W + BK} \qquad (4)
$$

**符号**： $\mathrm{TPOT}$ = 每输出 token 延迟（s）； $X$ = decode 吞吐（token/s）。式 (3) 是 $B$ 的仿射增函数——**凑 batch 必然抬高 TPOT**，斜率 $K/\beta$ 由 KV 几何与带宽定死。式 (4) 的行为用命题 1 概括。

**命题 1**（decode 吞吐饱和）： $X(B) = \beta B/(W + BK)$ 在 $B \ge 0$ 上单调增、上凸，且 $\lim_{B \to \infty} X(B) = \beta/K$ 。**证明**（极限部分）：分子分母同除 $B$ ， $X = \beta/(W/B + K) \to \beta/K$ ；单调性与上凸性的导数验证留作练习 2。代入 H100：饱和吞吐 $\beta/K = 3.35 \times 10^{12}/5.37 \times 10^{8} \approx 6240$ token/s； $X(32) \approx 3225$ 、 $X(64) \approx 4252$ token/s——从 32 到 64， $B$ 翻倍但吞吐只涨 32%，边际收益递减（练习 2 量化）。这就是「凑 batch 提吞吐」的天花板：KV 读随 $B$ 线性增长，最终把带宽吃满，吞吐饱和于 $\beta/K$ 。

第三条曲线是 TTFT 的排队分量，它依赖调度形态，须分口径。**static batching 基线**（口径：请求按批处理，一批内所有请求同进同出，新请求须等当前批结束）：批周期 = prefill 时间 + decode 时间，

$$
C(B) = \underbrace{\frac{2N T_{\mathrm{in}}}{F_{\mathrm{peak}}} \cdot B}_{\text{prefill，compute-bound}} + \underbrace{T_{\mathrm{out}} \cdot \frac{W + BK}{\beta}}_{\text{decode}} = a + bB \qquad (5)
$$

**符号**： $T_{\mathrm{in}}$ / $T_{\mathrm{out}}$ = 输入 / 输出 token 数； $F_{\mathrm{peak}}$ = dense 峰值算力； $a = T_{\mathrm{out}} W/\beta$ 为 $B$ 无关项， $b = 2NT_{\mathrm{in}}/F_{\mathrm{peak}} + T_{\mathrm{out}} K/\beta$ 为每请求边际周期。**口径注（prefill 项，跨章 $N$ 一致性）**：式 (5) prefill 取参数项近似 $2N T_{\mathrm{in}}/F_{\mathrm{peak}}$ ， $N$ 用总参数量 $N_{\mathrm{full}} = 8.03 \times 10^{9}$（存储口径）且略去 attention 二次项——较 1.2 的 matmul 口径（ $N = 7.50 \times 10^{9}$，层参数 + 输出头）高估约 7%（embedding 查表本不计 FLOPs），略二次项在 $T_{\mathrm{in}} = 4096$ 时低估约 6.7%，两误差相消，故 66.5 ms 与 1.2 含二次项的 matmul 账跨章一致（1.2 §1 有同注）；权重字节 $W$ 两章同取 $1.606 \times 10^{10}$ 字节。均匀到达假设下，新请求平均等待当前批结束约 $C(B)/2$ ，此即 TTFT 的排队分量。代入 8B / H100（ $T_{\mathrm{in}} = 4096$ 、 $T_{\mathrm{out}} = 512$ 、 $F_{\mathrm{peak}} = 989.5$ TFLOPS）：每请求 prefill $2NT_{\mathrm{in}}/F_{\mathrm{peak}} \approx 66.5$ ms， $a \approx 2.455$ s， $b \approx 0.1486$ s。于是 $B = 32$ ： $C \approx 7.21$ s、请求吞吐 $B/C \approx 4.44$ req/s、TTFT 排队 $\approx 3.6$ s、TPOT $\approx 9.92$ ms； $B = 64$ ： $C \approx 11.96$ s、请求吞吐 $\approx 5.35$ req/s、TTFT 排队 $\approx 5.98$ s、TPOT $\approx 15.05$ ms。**三项同时单调**： $B$ ↑ → 吞吐 ↑、TPOT ↑、TTFT 排队 ↑。机制一句话：大 batch 摊薄了权重读与 prefill 的固定成本（吞吐 ↑），但每步 KV 读更多（TPOT ↑）、批周期更长（head-of-line 排队 ↑）。

continuous batching（4.1）正是对第三条曲线的攻击：迭代级调度让请求随时 join/leave，新请求不必等整批结束，head-of-line 排队被消去；chunked prefill（4.3）再把长 prefill 切块与 decode 混批，消去 prefill 独占 GPU 造成的 decode stall。两者之后，TPOT-吞吐张力（式 (3)(4)）不变，TTFT 排队改由利用率 $\rho$ 支配（§5）——这是「为什么现代引擎不用 static batching」的账面答案。

操作点由约束夹出。TPOT SLO 给 $\mathrm{TPOT}(B) \le \tau$ ，由式 (3) 解出 batch 上限：

$$
B_{\max}(\tau) = \frac{\beta \tau - W}{K} \qquad (6)
$$

**符号**： $\tau$ = TPOT SLO 上限（s）。式 (6) 为正要求 $\beta \tau > W$ ——SLO 太紧（ $\tau < W/\beta$ ，单权重读就超时）则无可行 batch。代入 H100、 $\tau = 10$ ms： $B_{\max} = (3.35 \times 10^{12} \times 0.01 - 1.606 \times 10^{10})/5.37 \times 10^{8} \approx 32.5$ ，取 $B = 32$ ，对应吞吐 $\approx 3225$ token/s。另一条独立约束是 KV 显存容量： $B \cdot K + W \le$ 显存，H100 给 $B \le (80 - 16.06)\times 10^{9}/5.37 \times 10^{8} \approx 119$ （理想口径，未计激活与碎片，精确账归 2.1 / 9.1）。 $\tau = 10$ ms 时 TPOT 约束（32）先于容量约束（119）绑定——**操作点是 SLO 与容量两条上限的较小者**，再叠加 goodput 考量（§6）。三卡对照（ $\tau = 10$ ms，理想 roofline 口径）：

| 卡 | $\beta$ | $B_{\max}$（TPOT 约束） | $X(B_{\max})$ | KV 容量上限 |
| --- | --- | --- | --- | --- |
| H100 SXM | 3.35 TB/s | ≈32 | ≈3225 token/s | ≈119 |
| H200 SXM | 4.8 TB/s | ≈59 | ≈5949 token/s | ≈232 |
| B200 | 7.7 TB/s | ≈113 | ≈11351 token/s | ≈305 |

换带宽更大的卡，同一 TPOT SLO 下能塞更多序列、吞吐更高——带宽升级对 decode 的收益在这里定量兑现。

## 5. 排队论基础：利用率与延迟爆炸

利用率 0.8 和 0.95，排队延迟差多少？不是差 15%，是差数倍。本节建立 SLO 容量规划的数学地基。先回答一个更基本的问题：为什么需要排队论？因为 §4 的操作点只算了「系统能服务多快」（ $\mu$），没算「请求来得多快」（ $\lambda$）时队列会怎样——而 TTFT 的排队分量、goodput 的 attainment 约束，全是 $\lambda$ 与 $\mu$ 之比的函数。

**M/M/1 队列**（口径：到达为 Poisson 过程率 $\lambda$ ，服务时间独立同分布指数率 $\mu$ ，单服务台，无限队列，FCFS）。所谓 M/M/1，是指到达间隔与服务时间皆指数分布（Markov 性，第一个 M）、单服务台（1）的最简排队模型。它是可精确求解的基线，真实 serving 的偏离在本节末讨论。

**推导路线**（4 步）：由 birth-death 平衡求稳态队长分布 → 求期望队长 $L$ → 用 Little 定律 $L = \lambda W_{\mathrm{soj}}$ 反解逗留时间 $W_{\mathrm{soj}}$ → 减去服务时间得纯排队等待 $W_q$ 。关键一步是 Little 定律把「队长」与「时间」两个口径接通。

稳态下 birth-death 平衡给 $\lambda p_n = \mu p_{n+1}$ ，故 $p_n = \rho^n p_0$ ， $\rho = \lambda/\mu$ 。归一化 $\sum_{n \ge 0} p_n = 1$ 要求 $\rho < 1$ ，得 $p_0 = 1 - \rho$ ：

$$
p_n = (1 - \rho)\rho^{n}, \qquad \rho = \frac{\lambda}{\mu} < 1 \qquad (7)
$$

**符号**： $p_n$ = 系统内恰有 $n$ 个请求的稳态概率； $\rho$ = 利用率（ $\rho \ge 1$ 时队列无稳态、无限增长）。期望队长 $L = \sum_n n p_n = \rho/(1-\rho)$（几何级数求导）。Little 定律 $L = \lambda W_{\mathrm{soj}}$ 给逗留时间（sojourn time，排队 + 服务）：

$$
W_{\mathrm{soj}} = \frac{L}{\lambda} = \frac{1}{\mu - \lambda} \qquad (8)
$$

纯排队等待 $W_q = W_{\mathrm{soj}} - 1/\mu$（减去均值服务时间 $1/\mu$）：

$$
W_q = \frac{1}{\mu - \lambda} - \frac{1}{\mu} = \frac{\lambda}{\mu(\mu - \lambda)} = \frac{\rho}{\mu - \lambda} \qquad (9)
$$

**符号**： $W_{\mathrm{soj}}$ = 逗留时间（排队 + 服务）； $W_q$ = 纯排队等待； $1/\mu$ = 均值服务时间。**记号消歧**（W1 fix round F3）：本节 $W_{\mathrm{soj}}$ 为逗留时间，与 §1–4 的权重字节 $W$ 不同义，故加 soj 下标避免 $W$ 重载。**命名口径**（四路审查 A-S3-8）： $W_{\mathrm{soj}}$ 是 sojourn time 而非纯等待，Kingman VUT 近似的是 $W_q$ ，两者勿混。代入示意口径 $\mu = 10$ req/s、 $\lambda = 8$ req/s（换抽象服务率口径，只为隔离排队效应，不与 §4 的 8B decode 数混算）： $\rho = 0.8$ ， $W_{\mathrm{soj}} = 1/(10-8) = 0.5$ s， $W_q = 0.8/2 = 0.4$ s，服务 $1/\mu = 0.1$ s， $W_{\mathrm{soj}} = W_q + 1/\mu$ 自洽。把 $\lambda$ 推到 9.5： $\rho = 0.95$ ， $W_{\mathrm{soj}} = 1/0.5 = 2$ s、 $W_q = 1.9$ s——利用率从 0.8 到 0.95，逗留时间翻 4 倍。 $\lambda \to \mu$ 时 $W_{\mathrm{soj}} \to \infty$ ：**延迟爆炸是 $1/(1-\rho)$ 的极点，不是线性恶化**。这就是 SLO 容量规划必须留利用率余量的数学原因。

指数服务时间是强假设。真实请求长度分布重尾（agent 与 chatbot 长度差异大，1.1），服务时间方差远超指数分布。**Kingman VUT 近似**（G/G/1 重流量近似）把方差补进来：

$$
W_q \approx \frac{\rho}{1 - \rho} \cdot \frac{c_a^{2} + c_s^{2}}{2} \cdot \mathrm{E}[S] \qquad (10)
$$

**符号**： $c_a^{2}$ / $c_s^{2}$ = 到达间隔 / 服务时间的变异系数平方（方差 ÷ 均值平方）； $\mathrm{E}[S] = 1/\mu$ = 均值服务时间；VUT = Variability × Utilization × Time。一致性核对：M/M/1 的 $c_a^{2} = c_s^{2} = 1$ ，式 (10) 退化为 $\frac{\rho}{1-\rho} \cdot 1 \cdot \frac{1}{\mu} = \frac{\rho}{\mu - \lambda}$ ，恰为式 (9)——Kingman 是 M/M/1 的方差推广。代入 $\rho = 0.8$ 、 $\mathrm{E}[S] = 0.1$ s、 $c_a^{2} = 1$ 、 $c_s^{2} = 4$（服务时间高方差）： $W_q \approx (0.8/0.2) \times (5/2) \times 0.1 = 1.0$ s，是 M/M/1 同 $\rho$ 下 0.4 s 的 2.5 倍——**方差因子 $(c_a^{2}+c_s^{2})/2$ 直接乘在排队延迟上**，重尾长度分布的代价定量可见。

把爆炸反过来用，就得到容量规划的余量规则。给定排队延迟预算 $q$ （TTFT 排队分量的 SLO），由 M/M/1 式 (9) 写 $W_q = \mathrm{E}[S] \cdot \rho/(1-\rho) \le q$ ，解出利用率上限：

$$
\rho \le \frac{q}{q + \mathrm{E}[S]} \qquad (11)
$$

**符号**： $q$ = 排队延迟预算（s）； $\mathrm{E}[S] = 1/\mu$ = 均值服务时间。代入 $\mathrm{E}[S] = 0.1$ s、 $q = 0.4$ s： $\rho \le 0.4/0.5 = 0.8$ ——要把排队压在 0.4 s 内，利用率最多 0.8，必须留 20% 余量；若预算收紧到 $q = 0.1$ s，则 $\rho \le 0.5$ ，余量翻倍。预算越紧、服务时间越长，可跑的利用率越低——这就是「SLO 越严、单卡能服务的请求率越低、成本越高」的排队论根源，9.1 的容量与成本全账以此为地基。方差修正下把式 (11) 的 $q$ 换成 $q / [(c_a^2+c_s^2)/2]$ 即得 Kingman 口径的余量（更保守）。

> **提点**：M/M/1 把 GPU 当单服务台、 $\mu$ 当常数，但 continuous batching 的 $\mu$ 是状态依赖的——队列越长、batch $B$ 越大、聚合 token 率越高（式 (4)），直到饱和。所以 M/M/1 是「 $\rho \to 1$ 爆炸」的直觉基线，不是精确 serving 模型；生产容量规划用实测吞吐-延迟曲线 + Kingman 方差修正（9.1），不套常数 $\mu$ 的解析式。

## 6. SLO 操作点：goodput 最大化 ≠ 吞吐最大化

吞吐最大的 batch，为什么不是该选的 batch？因为吞吐不计延迟达标率。goodput 的定义（1.1 已立，DistServe 原文口径，[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]）是：**每 GPU 在 SLO attainment 目标（如 90% / 99%）约束下可服务的最大请求率**——per-GPU 归一 + attainment 目标是定义要件，不是「达标请求的裸吞吐」。记 goodput 为 $\mathcal{G}$ 。吞吐最大化只管 $X(B)$ 最大（式 (4)， $B$ 越大越好直到饱和）；goodput 最大化要同时满足 $\mathrm{TPOT} \le \tau_T$ 与 $\mathrm{TTFT} \le \tau_F$ ，把 $B$ 压在式 (6) 的 $B_{\max}$ 之下、把 $\rho$ 压在延迟爆炸极点之下。两者最优解通常不重合：吞吐最优把 $B$ 推到饱和附近，TPOT 早已越限，attainment 崩塌， $\mathcal{G}$ 反而下降。

操作点选择因此是一个约束问题：在 $\mathrm{TPOT}(B) \le \tau_T$（式 (6)）、 $B \cdot K + W \le$ 显存（§4）、 $\rho = \lambda/\mu(B) < 1$ 且留余量（§5）三条约束下，选 $B$ 与实例数使 $\mathcal{G}$ 最大。 $\mu(B) = X(B)/T_{\mathrm{out}}$ 是 batch 依赖的请求服务率——把 §4 的 token 吞吐除以每请求输出长度即得，这正是 §5 提点里「 $\mu$ 状态依赖」的来源。

DistServe 的实测把这条账兑现成数字（[Zhong et al., 2024](../references.md)，[A]，§1，13B 模型 A100、90% attainment 口径）：colocate 单卡 goodput ≈1.6 rps；prefill / decode 分离后 prefill 实例 5.6 rps、decode 实例 10 rps，2P+1D 配置整体 10 rps = 3.3 rps/GPU，为 colocate 的 2.1×。机制：colocate 时 prefill 与 decode 互相干扰（prefill 独占 GPU 抬 decode TPOT、decode 占槽延 prefill TTFT），两阶段 SLO 难以同时达标；分离后各自按 §4 的操作点独立优化，goodput 提升。这是 PD 分离的账面动机（系统展开归 8.2），也是「goodput 而非吞吐才是 serving 目标函数」的实证。本章交付操作点的数学（式 (6) + §5 排队约束），容量与成本的全账算例归 9.1。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| ridge point = dense 峰值算力 ÷ 带宽；H100 BF16 ≈295、H200 ≈206、B200 ≈292、B200 FP4 ≈1169 FLOP/字节 | 式 (1)；datasheet dense 表 | §2、练习 1 |
| decode 强度全窗口 ≪ ridge，换卡收益几乎全来自带宽；H200 纯带宽升级、B200 算力带宽同比放大 | §2 三条结论 | §2、1.2 |
| Ascend 950 算力 dense/sparse 口径官方未注明，只逐字引、不与 NVIDIA dense 同表比；Atlas 350 卡级媒体数不作一手 | §3 口径注 | §3 |
| TPOT(B) 仿射增、X(B) 单调增上凸饱和于 β/K（H100 ≈6240 token/s） | 式 (3)(4)；命题 1 | §4、练习 2 |
| static batching 下 B↑ → 吞吐↑、TPOT↑、TTFT 排队↑ 三项同时单调；continuous batching 消去 head-of-line | 式 (5)；8B/H100 算例 | §4、4.1 / 4.3 |
| TPOT SLO 给 B_max=(βτ−W)/K，操作点取 SLO 与 KV 容量两上限较小者 | 式 (6)；三卡对照表 | §4、练习 3 |
| M/M/1 逗留 W_soj=1/(μ−λ)、纯排队 W_q=ρ/(μ−λ)；ρ→1 延迟按 1/(1−ρ) 爆炸 | 式 (7)(8)(9) | §5、练习 4 |
| Kingman VUT 近似 W_q，方差因子 (c_a²+c_s²)/2 直乘；c_a=c_s=1 退化 M/M/1 | 式 (10) | §5、练习 4 |
| 排队预算 q 给利用率上限 ρ≤q/(q+E[S])；SLO 越紧余量越大、单卡请求率越低 | 式 (11) | §5、9.1 |
| goodput = per-GPU + attainment 约束下最大请求率；吞吐最优 ≠ goodput 最优；DistServe 2P+1D 2.1× | §6；DistServe §1 | §6、1.1 / 8.2 / 9.1 |

## 练习

1.（计算 · [10]）用 §2 datasheet 表与式 (1) 算 H200 SXM 的 BF16 与 FP8 ridge point，并解释为何 H200 的 BF16 ridge 低于 H100。校验：BF16 ≈206、FP8 ≈412 FLOP/字节，与 §2 ridge 表一致；H200 与 H100 算力同为 989.5 TFLOPS。
2.（推导 · [20]）完成 §4 命题 1（decode 吞吐饱和）留作的导数验证：对 $X(B) = \beta B/(W + BK)$ 求 $X'(B)$ 与 $X''(B)$ ，证明 $X' > 0$ 、 $X'' < 0$（单调增、上凸），并确认 $\lim_{B\to\infty} X = \beta/K$ 。再代入 8B/H100 口径算 $B = 32$ 处的边际增益 $X'(32)$（token/s 每增一序列），与 $B = 64$ 处对比说明递减。校验： $X'(B) = \beta W/(W+BK)^2$ 恒正； $X'(32) > X'(64)$；饱和值 ≈6240 token/s 与正文一致。
3.（计算 · [30]）8B dense、 $T = 4096$ 、TPOT SLO $\tau = 10$ ms：(a) 用式 (6) 求 H100 的 $B_{\max}$ 与对应吞吐；(b) 算 KV 容量上限并判断哪条约束先绑定；(c) 换 H200 重算 (a)(b)，给出同 SLO 下吞吐提升倍数；(d) 若把 $\tau$ 收紧到 5 ms，H100 上式 (6) 是否仍有正解？为什么？校验：(a) $B_{\max} \approx 32$ 、吞吐 ≈3225 token/s；(c) H200 吞吐 ≈5949 token/s；(d) $\beta\tau = 1.675 \times 10^{10} > W = 1.606 \times 10^{10}$ 仍有正解但 $B_{\max}$ 极小——与 §4「 $\beta\tau > W$ 才有可行 batch」一致。
4.（推导 · [30]）(a) 由式 (7) 的稳态分布 $p_n = (1-\rho)\rho^n$ 推期望队长 $L = \rho/(1-\rho)$，再用 Little 定律导出式 (8)(9)；(b) 验证 Kingman 式 (10) 在 $c_a^2 = c_s^2 = 1$ 时退化为式 (9)；(c) 设 $\rho = 0.8$ 、 $\mathrm{E}[S] = 0.1$ s、 $c_a^2 = 1$ 、 $c_s^2 = 4$，算 $W_q$ 并给出相对 M/M/1 的倍数。校验：(b) 退化后恰为 $\rho/(\mu-\lambda)$；(c) $W_q = 1.0$ s、倍数 2.5，与 §5 数值一致。

## 参考

- [NVIDIA H100/H200/B200 datasheet](../references.md)：NVIDIA 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露——H100 SXM / H200 SXM / B200（HGX）dense 算力与带宽、GB200 对照值、sparse 脚注、NVLink / SuperPoD 域参数（§2 / §3）
- [Huawei Connect 2025 keynote / hiascend 产品页](../references.md)：Ascend 950 系规格（key: ascend-950-keynote），[C] 官方披露——950PR / 950DT 芯片级显存带宽、1 PF FP8 / 2 PF FP4（dense/sparse 未注明）、1784TFLOPS（精度未注明）、互联 2 TB/s、Atlas 950 SuperPoD（§3）
- [Zhong et al., 2024](../references.md)：DistServe，arXiv:2401.09670，[A]（key: distserve）——goodput 定义（per-GPU + attainment）、13B/A100/90% attainment 的 colocate 1.6 rps vs 2P+1D 3.3 rps/GPU（2.1×）实测（§6）
