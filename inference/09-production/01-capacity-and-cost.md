# 9.1 容量规划与成本账（Capacity Planning and Cost Accounting）

> 给定到达率与长度分布，SLO 要多少张 GPU、多少台存储服务器、每百万 token 多少美元？本章把排队余量、prefill 算力、decode 带宽与容量、存储 EC、命中节省串成一条可复算的全账，并钉死「收入不是成本」的口径 | 前置：1.1（goodput 定义式 (3)）、1.2（prefill / decode 算力带宽账与 $\kappa$ 几何）、1.3（M/M/1 与 Kingman、余量规则式 (8)-(11)）、2.2（命中率 → prefill 节省命题 2）、8.2（xPyD 配比式 (5)-(8)）、8.5（伸缩触发账与冗余 quota）；7.4（Wide-EP 最小 decode 域）、2.4（3FS 分层）为指针 | 锚点：DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露，doc L17 / L91 / L101 / L105 / L111 / L187 / L211 / L251，核实 2026-11-24）、DistServe（[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]）、NVIDIA datasheet（[NVIDIA datasheet](../references.md)，[C] 官方规格披露）、3FS（[DeepSeek-AI, 3FS](../references.md)，教学参照）

## 1. 问题与章脊口径：一条负载要多少硬件

λ=100 req/s 的两段混合负载，TTFT 要压在 0.5 s、TPOT 压在 30 ms——需要几张 prefill GPU、几张 decode GPU、几台 KV 存储服务器，每百万 token 成本多少美元？这四个数不是拍脑袋的余量叠加，而是从到达率与长度分布出发、经排队论与 roofline 逐步算出的约束解。本章把这条链算到底，并在每一步钉死口径：哪些是官方披露、哪些是教学假设、哪些口径不可互相代入。

**本章口径**（章脊锁定，R2-2）：模型与硬件沿 1.2 / 1.3 / 8.2 / 8.5 的 8B dense bf16 @ H100 SXM—— $L = 32$ 、 $d_{\mathrm{model}} = 4096$ 、 $H_{kv} = 8$ 、 $d_{\mathrm{head}} = 128$ 、matmul 口径 $N = 7.50 \times 10^{9}$ 、 $\kappa = 4096$ B/token/层（GQA-8，1.2 式 (3)）、 $\kappa L = 131072$ B/token（128 KiB）、权重字节 $W = 1.606 \times 10^{10}$ B、BF16 dense 峰值 989.5 TFLOPS、HBM 带宽 $\beta = 3.35$ TB/s、每 GPU 可供 KV 的 HBM 预算 $M_{\mathrm{KV}} = 60$ GiB（8.5 教学值）（[NVIDIA datasheet](../references.md)，[C]）。prefill MFU 取 0.5，故每 GPU 有效 prefill 算力 $F_P = 494.75$ TFLOPS（沿 8.2）。**workload 换两段混合口径**（原因：8.2 / 8.5 用单段负载，本章要演示长度分布与命中形态如何分别主导各笔账，故取 chatbot + agent 两组）：

| 段 | 到达率（req/s） | $T_{\mathrm{in}}$ | $T_{\mathrm{out}}$ | 命中率 $h$ | TTL（生产口径） |
| --- | --- | --- | --- | --- | --- |
| chatbot | 80 | 1024 | 256 | 0.5 | 7 天 |
| agent | 20 | 16384 | 2048 | 0.9 | 72 小时 |

SLO：TTFT 0.5 s、TPOT 30 ms（沿 1.1 / 8.2 教学值）、attainment 目标 $\theta = 0.9$ （即 90%，DistServe 口径，1.1 式 (3)）。排队延迟预算 $q = 0.2$ s（口径：TTFT 0.5 s 扣除 prefill 计算与 KV 取回 / 传输暴露份额 0.3 s 的教学分配，保守覆盖最长 agent prefill 76.7 ms，§3）。decode 利用率余量 $\rho_D = 0.8$ （教学假设，沿 8.2）。GPU 时价 $u_P = u_D = 2$ USD/GPU-hour（**教学假设，非市场报价、非官方披露**，只为把成本算到底；真实时价随云厂商 / 代际 / 租约变动）。**本章 $\lambda$ = 到达率（req/s），非 training GAE 的 $\lambda$ 参数**； $n_P / n_D$ = P / D 两池 GPU 数（实例数 = GPU 数 ÷ 每实例 GPU 数，本章教学实例取单 GPU，禁裸 P / D / G 当计数）；bytes 不占数学字母，一律文字「字节」或 $\mathrm{bytes}$ ；并发用 $C$ ，本章不用字母 $B$ ；货币一律 USD。DeepInfer 生产数字（155T tokens/day、38 USD/GPU-hour、14.4 TB/GPU、64P+64D 等）按「官方披露」+ doc 行号单独标注，**不与 8B 教学算例混算**，换口径处一句话声明。

```text
容量规划全链（章脊 λ=100 req/s，chatbot 80 / agent 20）

  到达率 λ + 长度分布 + 命中率 h
        |
        v
  [§2] 排队余量：三个 ρ 上界取 min --------> ρ_P = 0.72
        |            （均值 / Kingman / P90 attainment）
        +-------------------+
        |                   |
        v                   v
  prefill 算力需求      decode 并发 C（Little 定律）
  （FLOPs/s）               |
        |                   +------------+
        v                   v            v
  [§3] n_P = 4        [§4] 带宽闸 37   容量闸 45
        |                   取 max -> n_D = 45
        +---------+---------+
                  v
        [§6] 存储：四约束取 max -> 4 台服务器
                  v
        [§7] 成本：98 USD/h -> 0.0578 USD/M tokens（全 token 口径）
```

所谓容量规划（capacity planning），是指在 SLO 约束下把负载需求映射为硬件数量的过程；所谓成本账，是把硬件数量折成单位 token 货币成本的过程。两者的交点是 goodput：1.1 式 (3) 定义的 per-GPU goodput 是「attainment 目标下每 GPU 可服务的最大请求率」，它直接把「每 GPU 能服务多少请求」与「每请求成本」接成反比——goodput 越高，单位成本越低。本章的 sizing 就是求满足 goodput 约束的最小硬件数。

**workload mix 为什么是两段而不是平均值。** agent 与 chatbot 的长度分布与命中形态截然不同（1.1 recap）：chatbot 输入短（千 token 量级）、输出短、多轮历史复用带来中等命中率，命中长尾重——少量 KV 被读极多次（doc L187 官方披露）；agent 输入长（万 token 量级，工具输出与文件内容逐步累积）、输出长（推理链）、前缀逐步追加带来高命中率，TTL 72 小时恰跨周末（doc L211）。公开 trace 的归属也分开：Azure LLM inference trace 归 Splitwise（两个 Azure 服务 code/conv，2023-11-11），ShareGPT 是 vLLM 系 benchmark 常用数据集，两者形态不同（1.1 已标注，本章不重述）。用单一平均长度做 sizing 会抹掉「谁主导哪笔账」的结构——§3 / §4 将看到 prefill 算力、decode KV、存储写带宽分别由不同段主导，平均值会同时算错三笔账。

## 2. 利用率余量：三个 ρ 上界取最小

利用率 ρ 能推到多高？1.3 已给出 M/M/1 的均值余量规则，但 SLO 是百分位约束不是均值约束——本节补上排队尾分布，导出 attainment 上界，再与均值、Kingman 两个上界并排取最小。路线（4 步）：先 recap M/M/1 均值结果（1.3 式 (8)-(11)，不重推）；再推纯排队等待的尾分布（命题 1，Erlang 混合）；然后把 attainment 目标翻译成 ρ 上界；最后代入两段混合的服务时间分布算三个上界。关键一步是命题 1：把「队长几何分布 × Erlang 等待」混合成指数尾，才有闭式的 percentile 约束。

**recap（1.3，指引不重推）**：M/M/1 逗留时间 $W_{\mathrm{soj}} = 1/(\mu - \lambda)$ 、纯排队等待 $W_q = \rho/(\mu - \lambda)$ （1.3 式 (8)(9)）； $\rho \to 1$ 时延迟按 $1/(1-\rho)$ 爆炸。Kingman VUT 近似 $W_q \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \mathrm{E}[S]$ （1.3 式 (10)），方差因子直乘。均值余量规则 $\rho \le q/(q + \mathrm{E}[S])$ （1.3 式 (11)）。1.3 提点已声明：continuous batching 的 $\mu$ 状态依赖，M/M/1 是直觉基线非精确 serving 模型，生产用实测吞吐-延迟曲线 + Kingman 修正。本章沿用该基线做独立重推导，口径注见本节末。

**命题 1**（M/M/1 纯排队尾分布）：M/M/1 FCFS 队列，利用率 $\rho \lt 1$ ，则纯排队等待 $W_q$ 满足

$$
\Pr(W_q \gt t) = \rho \thinspace e^{-(\mu - \lambda) t} \qquad (1)
$$

**证明**（路线：PASTA 看到几何队长 → 条件等待是 Erlang → 对几何分布求和交换次序 → 指数闭式，4 步）。第一步，PASTA 性质使到达请求看到的系统内请求数 $N$ 服从稳态几何分布 $\Pr(N = n) = (1-\rho)\rho^n$ （1.3 式 (7)）；请求需要等待当且仅当 $N \ge 1$ ，概率 $\rho$ 。第二步，给定 $N = n \ge 1$ ，等待时间 = 在服请求的剩余服务（指数率 $\mu$ ，无记忆性）+ 队列中 $n-1$ 个请求的完整服务，合为 $\mathrm{Erlang}(n, \mu)$ ，其尾 $\Pr(\mathrm{Erlang}(n,\mu) \gt t) = \sum_{k=0}^{n-1} e^{-\mu t}(\mu t)^k / k!$ 。第三步，对 $N$ 求和并交换求和次序：

$$
\Pr(W_q \gt t) = \sum_{n=1}^{\infty} (1-\rho)\rho^n \sum_{k=0}^{n-1} \frac{e^{-\mu t}(\mu t)^k}{k!} = \sum_{k=0}^{\infty} \frac{e^{-\mu t}(\mu t)^k}{k!} (1-\rho)\sum_{n=k+1}^{\infty} \rho^n
$$

内层几何和 $(1-\rho)\sum_{n=k+1}^{\infty}\rho^n = \rho^{k+1}$ ，代入得 $\rho \thinspace e^{-\mu t} \sum_k (\rho\mu t)^k/k! = \rho \thinspace e^{-\mu t} e^{\rho\mu t} = \rho \thinspace e^{-\mu(1-\rho)t}$ 。第四步， $\mu(1-\rho) = \mu - \lambda$ ，即式 (1)。证毕。一致性核对： $t=0$ 给 $\Pr(W_q \gt 0) = \rho$ （需要等待的概率）；对 $t$ 积分 $\int_0^\infty \rho e^{-(\mu-\lambda)t} dt = \rho/(\mu-\lambda) = W_q$ ，恰为 1.3 式 (9)——尾分布与均值自洽。∎

**推论**（attainment 上界唯一）：式 (1) 对 $\rho$ 严格增（导数 $e^{-\mu(1-\rho)q}(1 + \rho\mu q) \gt 0$ ），故 attainment 约束有唯一 $\rho$ 上界。把 TTFT 排队分量预算 $q$ 与 attainment 目标 $\theta$ 代入式 (1)：

$$
\rho \thinspace e^{-\mu(1-\rho) q} \le 1 - \theta \qquad (2)
$$

**符号**： $\mu$ = 每 GPU 服务率（req/s）； $\rho$ = 利用率； $q$ = 排队延迟预算（s）； $\theta$ = attainment 目标。口径：本章把每张 prefill GPU 建模为独立 M/M/1（随机 dispatch），式 (2) 是 per-queue 尾约束；真实是 M/M/n + 状态依赖 $\mu$ ，此为教学基线（1.3 提点）。代入章脊 $\mu = 35.6$ req/s/GPU（§3 算出）、 $q = 0.2$ s、 $\theta = 0.9$ ： $\mu q = 7.12$ ，解 $\rho e^{-7.12(1-\rho)} = 0.1$ 。二分： $\rho = 0.7$ 给尾 $0.0827 \le 0.1$ ， $\rho = 0.75$ 给 $0.1265 \gt 0.1$ ，收敛到 $\rho \le 0.722$ 。**attainment 上界 0.722**。

**均值与 Kingman 上界。** 两段混合的服务时间（§3 表）：chatbot $s_c = 15.9$ ms、agent $s_a = 76.7$ ms，按 80/20 加权得 $\mathrm{E}[S] = 28.1$ ms、 $\mathrm{E}[S^2]$ 进而 $c_s^2 = 0.748$ （两点分布方差低于指数分布的 1）。Poisson 到达 $c_a^2 = 1$ ，方差因子 $V = (1 + 0.748)/2 = 0.874$ 。均值余量（1.3 式 (11)）： $\rho \le q/(q+\mathrm{E}[S]) = 0.2/0.2281 = 0.877$ 。Kingman 修正把 $q$ 换 $q/V$ ：

$$
\rho \le \frac{q}{q + V \thinspace \mathrm{E}[S]} \qquad (3)
$$

代入得 $\rho \le 0.2/(0.2 + 0.874 \times 0.0281) = 0.891$ 。**三个上界并排**：

| 上界 | 口径 | 数值 |
| --- | --- | --- |
| 均值余量（1.3 式 (11)） | $\mathrm{E}[W_q] \le q$ | 0.877 |
| Kingman 修正（式 (3)） | 方差修正均值 | 0.891 |
| attainment（式 (2)） | $\Pr(W_q \gt q) \le 1-\theta$ | 0.722 |

取最小得 $\rho_P = 0.72$ （向下取整到两位小数，保守）。**关键读数：percentile 约束比均值约束紧**——attainment 上界 0.722 远低于均值上界 0.877，因为指数尾在 P90 处比均值更早撞预算。只用均值余量做 sizing 会系统性高估单卡产能、低估 GPU 数。这就是「goodput 约束下的冗余」的数学来源：attainment 目标 $\theta$ 直接压低了可用利用率，多出的 GPU 是为尾延迟买的单。

> **提点**：本章两点混合的 $c_s^2 = 0.748 \lt 1$ ，Kingman 修正反而放松（0.891 > 0.877）；真实长度分布重尾（1.1）， $c_s^2 \gt 1$ 时 Kingman 收紧。方向取决于方差，不取决于「修正一定更保守」。无论方差如何，attainment 上界（式 (2)）在本章口径下都是最紧的——percentile 约束才是 sizing 的硬闸。

> **坑**：把 M/M/1 解析式当精确 serving 模型。触发条件：continuous batching 下 $\mu$ 随 batch 状态变化（1.3 提点），且 prefill 是 chunked、多请求共卡；现象：解析 $\rho$ 上界与实测吞吐-延迟曲线偏离，重尾负载下尤甚；规避：本章解析式只作独立重推导的教学基线，生产 sizing 用实测曲线定 $\rho$ （8.2 的 DistServe simulator 即干这件事），解析式给方向与量级。

## 3. Prefill sizing：长度分布 → 算力需求 → n_P

prefill 要几张 GPU？路线（3 步）：先把到达率与长度分布折成「需真正计算的 token 率」（扣命中），再算每请求 FLOPs（线性项 + attention 二次项，命中折减），最后除以有效算力与余量之积、向上取整。关键一步是 attention 项：命中只省前缀内部的 causal 对，剩余 token 仍 attend 复用前缀，省长尾比线性项慢（2.2 命题 2 recap）。

**non-cached token 率（recap 8.5 式 (1) 的混合推广）**：8.5 单段式 $R_{\mathrm{nc}} = \lambda(1-h)T_{\mathrm{in}}$ 对每段求和：

$$
R_{\mathrm{nc}} = \sum_i \lambda_i \thinspace (1 - h_i) \thinspace T_{\mathrm{in},i} \qquad (4)
$$

**符号**： $\lambda_i$ = 段 $i$ 到达率； $h_i$ = 段命中率（token 加权，2.2 口径）； $T_{\mathrm{in},i}$ = 段输入长。代入章脊：chatbot $80 \times 0.5 \times 1024 = 40960$ 、agent $20 \times 0.1 \times 16384 = 32768$ ，合 $R_{\mathrm{nc}} = 73728$ token/s。命中部分不耗 prefill 算力——这是用 non-cached 而非 total 的全部理由（8.5）。

**每请求 prefill FLOPs（recap 2.2 命题 2  corollary）**：命中前缀 $\delta_i = h_i T_{\mathrm{in},i}$ ，剩余 token 的线性项 + attention 项为

$$
f_{P,i} = 2N \thinspace (T_{\mathrm{in},i} - \delta_i) + 2L \thinspace d_{\mathrm{model}} \thinspace (T_{\mathrm{in},i}^2 - \delta_i^2) \qquad (5)
$$

**符号**： $N$ = matmul 参数量； $L$ 、 $d_{\mathrm{model}}$ = 层数、hidden 宽； $\delta_i$ = 命中前缀长。系数口径沿 2.2：乘加计 2、causal 减半、score 与 PV 两个 GEMM，每层 attention $\approx 2 d_{\mathrm{model}} T^2$ 。线性项省 $\delta$ 个 token 的权重 GEMM；attention 项只省前缀内部 causal 对（ $\delta^2/2$ 对），剩余 token attend 复用前缀的部分不省（2.2 命题 2 证明 recap，不重推）。代入章脊（attention 系数 $2L d_{\mathrm{model}} = 262144$ ）：

| 段 | 线性项（FLOPs） | attention 项（FLOPs） | $f_{P,i}$（FLOPs） | 服务时间 $s_i = f_{P,i}/F_P$ |
| --- | --- | --- | --- | --- |
| chatbot | $7.68 \times 10^{12}$ | $2.06 \times 10^{11}$ | $7.886 \times 10^{12}$ | 15.9 ms |
| agent | $2.458 \times 10^{13}$ | $1.337 \times 10^{13}$ | $3.795 \times 10^{13}$ | 76.7 ms |

agent 的 attention 项占自身 FLOPs 的 35%——长输入下二次项不可略。加权均值 $\mathrm{E}[S] = (80 \times 15.9 + 20 \times 76.7)/100 = 28.1$ ms，每 GPU 服务率 $\mu = 1/\mathrm{E}[S] = 35.6$ req/s（§2 已用）。

**n_P（recap 8.2 式 (5) 的混合推广）**：8.2 单段式 $n_P \ge \lambda \cdot 2N T_{\mathrm{in}}/(F_P \rho_P)$ 把分子换成混合 FLOPs 需求率：

$$
n_P \ge \frac{\sum_i \lambda_i f_{P,i}}{F_P \thinspace \rho_P} \qquad (6)
$$

**符号**： $F_P$ = 每 GPU 有效 prefill 算力（峰值 × MFU）； $\rho_P$ = §2 定的利用率上界。代入：需求率 $D_P = 80 \times 7.886 \times 10^{12} + 20 \times 3.795 \times 10^{13} = 1.390 \times 10^{15}$ FLOPs/s； $F_P \rho_P = 4.9475 \times 10^{14} \times 0.72 = 3.562 \times 10^{14}$ ； $n_P \ge 3.90$ ，向上取整 **$n_P = 4$** 。若只算线性项（漏 attention），需求率 $1.106 \times 10^{15}$ 、 $n_P \ge 3.10$ ，取整同为 4 但表观利用率从 0.702 虚低到 0.559——attention 项吃掉了 14 个百分点的余量，负载再涨 10% 就会撞穿 attainment 上界。**谁主导 prefill**：agent 段贡献 54.6% 的 FLOPs（其中 attention 项的 94.2% 来自 agent），chatbot 段贡献线性项的 55.6%——长输入段主导二次项，短输入段主导线性项总量，平均长度会同时抹掉这两个结构。

**goodput 交叉核对（recap 1.1 式 (3) / 8.2 式 (2)）**：4 张 GPU 实际利用率 $\rho = D_P/(4 F_P) = 0.702 \le 0.722$ ，代入式 (1) 得尾 $0.0843$ 、attainment $`91.6\% \ge 90\%`$ ✓。per-GPU prefill 请求率 $g_P = \mu \rho = 25.0$ req/s， $n_P g_P = 100 = \lambda$ ✓（8.2 式 (2) 的 min 平衡）。系统 per-GPU goodput $\mathcal{G} = \lambda/(n_P + n_D) = 100/49 = 2.04$ req/s/GPU（ $n_D$ 见 §4）——这是 attainment 约束下的操作点，不是裸吞吐。

> **坑**：用离线 benchmark 吞吐直接承诺在线产能。触发条件：离线测法固定 batch、无到达过程、无 SLO（1.1）；现象：DeepInfer 官方披露在线只达完全均衡离线基准的约 80%（doc L155「we reach roughly 80% of a perfectly balanced offline benchmark」，官方披露，核实 2026-11-24）；规避：本章 MFU 0.5 是已折入在线损失的教学假设，**不再与 0.8 因子叠乘**（避免双重打折）；若从离线实测出发 sizing，须先除 0.8 再定 $\rho$ 。

## 4. Decode sizing：带宽闸与容量闸取最大

decode 要几张 GPU？路线（3 步）：先用 Little 定律把到达率折成稳态并发，再分别算 KV 带宽闸与 KV 容量闸，取最大、向上取整，最后用步时预算交叉核对。关键一步是两道闸的物理含义不同——带宽闸管「跑不跑得动」（每步全读 KV），容量闸管「装不装得下」（KV 常驻），8.5 只算了容量闸并把带宽账留给本章，此处补齐。

**并发（recap 8.2 式 (6) 的混合推广）**：每请求占 decode 槽 $T_{\mathrm{out}}$ 步、每步 TPOT，Little 定律给稳态并发

$$
C = \sum_i \lambda_i \thinspace T_{\mathrm{out},i} \thinspace \mathrm{TPOT} \qquad (7)
$$

代入：chatbot $80 \times 256 \times 0.03 = 614.4$ 、agent $20 \times 2048 \times 0.03 = 1228.8$ ，合 $C = 1843.2$ 请求。平均上下文 $`\bar{T}_i = T_{\mathrm{in},i} + T_{\mathrm{out},i}/2`$ （生成均匀假设，8.2 口径）：chatbot 1152、agent 17408。**命中不省 decode**：命中前缀的 KV 仍常驻、每步仍全读——命中省的是 prefill 算力（§3），不是 decode 并发或 KV 流量（8.5 命题 1 边界 recap）。口径：逐序列独立读 KV（1.2），共享前缀的 cascade 复用是例外（2.2 / 3.1 指针），本章按独立读保守算。

**带宽闸（recap 8.2 式 (7) 的混合推广）**：每步全集群读 KV $`\kappa L \sum_i C_i \bar{T}_i`$ 字节，每秒 $1/\mathrm{TPOT}$ 步：

$$
n_D \ge \frac{\kappa L \sum_i C_i \bar{T}_i}{\mathrm{TPOT} \thinspace \beta_D \thinspace \rho_D} \qquad (8)
$$

**符号**： $\beta_D$ = decode 池每 GPU HBM 带宽（沿 8.2 符号）； $\rho_D$ = 利用率余量。代入：KV token 和 $`\sum C_i \bar{T}_i = 614.4 \times 1152 + 1228.8 \times 17408 = 2.210 \times 10^7`$ token，KV 总量 $\kappa L \times 2.210 \times 10^7 = 2.897 \times 10^{12}$ B（2.90 TB）；带宽需求 $2.897 \times 10^{12}/0.03 = 9.655 \times 10^{13}$ B/s = 96.55 TB/s； $\beta_D \rho_D = 3.35 \times 10^{12} \times 0.8 = 2.68 \times 10^{12}$ ； $n_D \ge 36.0$ ，取整 37。**agent 段贡献 96.8% 的 KV**——长输出 × 长上下文是 decode 池的最贵客户（8.2 §4 读数 recap）。

**容量闸（recap 8.5 式 (4)）**：KV 常驻总量除以每 GPU 预算：

$$
n_D \ge \frac{\kappa L \sum_i C_i \bar{T}_i}{M_{\mathrm{KV}}} \qquad (9)
$$

代入： $2.897 \times 10^{12} / 6.442 \times 10^{10} = 44.96$ ，取整 45。两道闸取最大： $\max(37, 45) = 45$ ， **$n_D = 45$** （容量闸主导，与 8.5 教学负载同结构）。

**步时交叉核对**：45 张 GPU 每 GPU 摊 KV $2.897 \times 10^{12}/45 = 6.437 \times 10^{10}$ B = 59.95 GiB $\le 60$ ✓；每步 KV 读时 $6.437 \times 10^{10}/3.35 \times 10^{12} = 19.2$ ms，权重读时 $W/\beta = 1.606 \times 10^{10}/3.35 \times 10^{12} = 4.8$ ms（口径：权重每步读一遍、与并发无关，由并行度与 batch 决策承担，8.1 / 7.4），合 24.0 ms $\le 30$ ms TPOT ✓。每 GPU 并发 $1843.2/45 = 40.96$ 请求。账自洽：容量闸恰好把步时压在预算内，带宽闸留的 20% 余量被容量闸的更紧约束覆盖。

**与弹性伸缩的接口（recap 8.5，指引不重推）**：本章算出的两个量恰是 DeepInfer autoscaling 的触发信号——doc L251 逐字（官方披露，核实 2026-11-24）：「Prefill determines the number of machines it needs from the number of non-cached tokens; decode adjusts based on the number of running requests.」prefill 看 non-cached token 率（式 (4)），decode 看 running request 数（并发 $C$ ，式 (7)）。本章给的是静态 sizing 基线；rate-limit、上下界、grace 迁移与冗余 quota 的动态账归 8.5，此处不重推。

> **提点**：两道闸的相对位置随 $\kappa$ 与上下文移动。MLA / KV 量化把 $\kappa$ 压低（2.3 / 7.2），容量闸与带宽闸同因子松——这是「KV 压缩直接折成 decode GPU 数」的账面依据（8.2 §4 读数）。本章 8B GQA bf16 的 $\kappa L = 128$ KiB/token 是偏大口径，生产 V4.1 量化 KV + 混合注意力小得多（§6 口径对照），故教学 decode 池偏大是口径使然，非生产形态。

## 5. P/D 配比与最小高效单元

教学负载算出 $n_P : n_D = 4 : 45 = 1 : 11.25$ ，DeepInfer 生产却披露 ~1:1——矛盾吗？不矛盾：配比是 workload 与硬件的函数，不是常数。本节 recap 8.2 的配比结论与官方披露，再讲最小高效单元的量子化代价。

**recap（8.2 式 (8) 与命题 4，指引不重推）**：KV 流量口径下配比 $n_P : n_D = \frac{2N_{\mathrm{act}} T_{\mathrm{in}}}{F_P \rho_P} : \frac{\kappa L \thinspace T_{\mathrm{out}} \bar{T}}{\beta_D \rho_D}$ ，与到达率 $\lambda$ 无关、与 TPOT 目标无关（命题 4 双重无关性），只随输入输出长度比与硬件 $F/\beta$ 比移动。本章 agent 段长输出 × 长上下文把 decode 右项推到 prefill 左项的 11 倍，故 1:11.25 是该混合负载的结构结果，不是普适常数。

**官方披露（doc L91 逐字，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，核实 2026-11-24）**：「1P + 1D is deployable. In practice the GPU ratio between P and D is roughly 1:1. Since the recommended decode configuration is EP64 or above, the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.」三个信息：~1:1 是**生产流量 mix 的经验配比**（doc「In practice」），不是理论最优定值——它对应 DeepInfer 实际负载的长度比与硬件比，换负载即移动（8.2 式 (8)）；64P+64D 是**最小高效单元**，量子化来自 decode 推荐 EP64 以上（Wide-EP 最小 decode 域，7.4 recap：每卡 MoE 权重流量 $\propto E/e$ ， $e$ 须足够大才摊薄权重读，EP64 是官方推荐下限）；1P+1D 可部署（小规模，不追求高效）。

**量子化代价。** 需求侧配比（式 (8)）与部署侧配比之间隔着实例 / 通信域粒度的向上取整。本章教学负载若按 DeepInfer 生产形态部署，decode 须凑足 EP64 域：需求 45 张 → 向上取整到 64 张（一个最小 decode 域），prefill 同步凑 64 张，合 128 张最小高效单元——相对需求 49 张，量子化引入约 2.6 倍冗余。这不是浪费，而是 Wide-EP 的通信域下限：小于 64 的 decode 域每卡权重流量过高、步时撞 SLO（7.4）。**最优配比在实践中是阶梯函数，不是光滑曲线**（8.2 §4 结论 recap）：小负载付量子化代价，大负载才逼近连续最优。教学算例用单 GPU 实例避开量子化，只为隔离 sizing 账；生产形态的量子化冗余另计（8.5 冗余 quota 是 fleet 层再上一层，本章不叠算）。

## 6. 存储 sizing：EC 算术与四约束

KV 存储要几台服务器？路线（4 步）：先列官方披露的硬件与容量锚，再推 EC 冗余下的 raw / usable / 物理 / 逻辑四口径算术（命题 2），然后把教学负载的写 / 读 / 24h 保留 / 每 GPU 经验四个约束并排取最大，最后声明 100 PiB 集群的口径歧义。关键一步是命题 2：raw 与 usable 两口径算出的「每服务器 GPU 数」恒等，EC 因子相消。

**官方披露锚（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，核实 2026-11-24）**：
- doc L101 逐字：「We recommend 20 × 30.73 TB SSDs per KV cache storage server, with each server providing at least 40 GB/s read and 5 GB/s write throughput and enough total capacity to retain 24 hours of data. Scale out the number of storage servers with the actual size of the inference deployment.」——每服务器 20 × 30.73 TB SSD、≥40 GB/s 读 / 5 GB/s 写、保留 24h 数据、随部署规模 scale out。
- doc L105 逐字：「In practice, about 14.4 TB per GPU measured as logical size; the physical capacity required is larger than that.」——每 GPU 约 14.4 TB 逻辑容量（物理更大）。
- doc L111：3FS **EC 10+2** ，每 EC 组容两节点同时故障；doc L113：读向组内全部 11 节点发请求、any 10 响应即可解码返回。
- doc L187：3FS 提供约 **100 PiB** SSD 给 KV cache。

> **坑**：doc L113「11 节点 any 10」与 L111「10+2 = 12 节点」存在内部张力（W1 / W4 裁定维持）；且开源 3FS repo 是 CRAQ 链式复制、无 EC 实现，EC 10+2 属生产官方披露（[DeepSeek-AI, 3FS](../references.md)，教学参照；2.4 recap）。引用时区分「生产 EC 口径」与「开源 repo 口径」，本章 sizing 用生产 EC 10+2 → 物理 ×12/10。

**EC 四口径算术（本轮独立重算）**。TB 取十进制 $10^{12}$ B、PiB 取二进制 $2^{50}$ B（spec §4.3 换算注明）：

$$
\begin{aligned}
\text{raw/server} &= 20 \times 30.73 = 614.6 \thinspace \text{TB} \\
\text{usable/server} &= 614.6 \times 10/12 = 512.2 \thinspace \text{TB} \\
\text{物理/GPU} &= 14.4 \times 12/10 = 17.28 \thinspace \text{TB}
\end{aligned} \qquad (10)
$$

**命题 2**（存储两口径等价）：每服务器可支撑的 GPU 数，用 raw ÷ 物理/GPU 与用 usable ÷ 逻辑/GPU 计算，结果恒等。**证明留作练习 1(c)**（路线提示：两式同乘 EC 因子 12/10，相消）。代入： $614.6/17.28 = 512.2/14.4 = 35.6$ GPU/server——两口径同值，约 **35.6 GPUs/server**。这个恒等不是巧合，是 EC 冗余在分子分母同侧出现的结构结果。

**100 PiB 集群的口径歧义（须声明）**： $100 \thinspace \text{PiB} = 100 \times 2^{50} \thinspace \text{B} = 1.126 \times 10^{17}$ B = 112590 TB（十进制）。若 100 PiB 指 **raw** 容量，服务器数 $\approx 112590/614.6 = 183$ ；若指 **usable / 逻辑** 容量， $\approx 112590/512.2 = 220$ 。doc 未明示口径，**两个数都给出并声明歧义**（向上取整则 184 / 220；~100 PiB 本身是约数，整数精度无意义）。反推 GPU 规模同理：raw 口径 $112590/17.28 \approx 6516$ GPU，usable 口径 $112590/14.4 \approx 7819$ GPU——这是算术隐含值，非官方披露，引用须带口径。

**教学负载的四约束（算到底）**。先算存储流量。写盘 = 新生成 KV（prefill non-cached + decode 输出，口径：decode 不做 offload 但产出 KV 仍写盘，doc L183）：non-cached $73728$ + 输出 $61440 = 135168$ token/s，乘 $\kappa L = 131072$ B 得写率 **17.72 GB/s**。读盘 = 命中取回（口径：专用存储集群形态，全部前缀命中从 3FS 取回，2.2 / 2.4；本地 HBM 命中不计存储读）：命中 token $80 \times 512 + 20 \times 14745.6 = 335872$ /s，乘 $\kappa L$ 得读率 **44.02 GB/s**。24h 保留逻辑容量 = $17.72 \times 86400 / 1000 \approx 1531$ TB。四个约束并排：

| 约束 | 计算 | 需求服务器数 |
| --- | --- | --- |
| 24h 保留容量 | $1531 \times 1.2 / 614.6$ | 2.99 → 3 |
| 写带宽 | $17.72 / 5$ | 3.54 → 4 |
| 读带宽 | $44.02 / 40$ | 1.10 → 2 |
| 每 GPU 经验（14.4 TB 逻辑） | $49 \times 14.4 \times 1.2 / 614.6$ | 1.38 → 2 |

$$
n_{\mathrm{stor}} \ge \max\left( \frac{24\text{h 逻辑} \times 1.2}{614.6},\ \frac{\text{写率}}{5},\ \frac{\text{读率}}{40},\ \frac{n_{\mathrm{GPU}} \times 14.4 \times 1.2}{614.6} \right) \qquad (11)
$$

取最大、向上取整： **$n_{\mathrm{stor}} = 4$** ，**写带宽闸主导**（4 台写容量 20 GB/s，利用率 88.6%，接近官方最低规格上限；生产会再留 quota 余量，8.5 recap，教学按官方最低规格取整不另加）。

**口径对照：为什么教学 24h 闸超过每 GPU 经验闸。** 官方 14.4 TB/GPU 是 V4.1 生产口径——量化 KV（主 KV 与 indexer KV 的低精度格式，2.3 / 7.3）+ 混合注意力（只部分层缓存全 KV，7.2）使每 token KV 远小于本章 8B GQA bf16 的 128 KiB/token。教学负载反推逻辑容量 $1531/49 = 31.2$ TB/GPU，是官方 14.4 的 2.2 倍，纯因 KV 几何口径不同。**两口径不可互相代入**：官方经验锚用于生产 V4.1 部署的一致性核对，24h 保留 + 带宽闸用于任意负载的 sizing 下界，取最大即同时满足。

**24h、TTL、LRU 三者不矛盾。** 24h 是容量 sizing 准则（doc L101「enough total capacity to retain 24 hours of data」）；TTL 是条目年龄上限（agent 72h / chatbot 7d，doc L211）；LRU watermark 是容量压力下的驱逐信号（doc L115 GC Worker）。关系：容量决定「实际能留多久」，TTL 决定「最晚必须删」，LRU 决定「容量不够时先逐谁」。容量充足时条目活到 TTL；容量紧张时 LRU 在 TTL 之前驱逐——TTL 缩短的命中损失正是 §8 的反事实账。kvcache-store 侧无过期逻辑、回收全在 3FS GC（doc L187），三者由同一 GC 实现。

## 7. 成本模型：USD/M tokens 与「收入不是成本」

每百万 token 多少美元？路线（3 步）：先做量纲对齐推出通式，再代入教学负载算两种 token 计数口径，最后把 DeepInfer 的 38 USD/GPU-hour 钉死为收入口径、声明不可代入成本。关键一步是量纲：小时价 × GPU 数 = USD/h，分母必须是 tokens/h 速率才能约成 USD/M tokens。

**通式推导**。设两池 GPU 时价 $u_P, u_D$ （USD/GPU-hour）、GPU 数 $n_P, n_D$ ，token 速率 $\dot{T}$ （tokens/h）。每小时成本 $u_P n_P + u_D n_D$ （USD/h），除以每小时百万 token 数 $\dot{T}/10^6$ （M tokens/h）：

$$
\text{USD/M tokens} = \frac{u_P n_P + u_D n_D}{\dot{T} / 10^6} \qquad (12)
$$

**符号**： $u_P, u_D$ = 两池 GPU 时价（本章 $u_P = u_D = 2$ USD/GPU-hour，教学假设；与 8.5 quota 利用率 $u$ 不同章不同义）； $\dot{T}$ = token 计数速率（口径见下）。量纲核对：[USD/h] ÷ [M tokens/h] = USD/M tokens ✓。PD 分离时 $n_{\mathrm{GPU}} = n_P + n_D$ 、两池可异价（Splitwise 异构集群优化，8.2 recap）；同价时退化为 $u \thinspace n_{\mathrm{GPU}} / (\dot{T}/10^6)$ 。口径：成本只计 GPU 租金，存储 / 网络 / 运维另项不计（教学口径，真实 TCO 须加）。

**教学负载代入（两种 token 计数口径）**。成本 $2 \times 49 = 98$ USD/h。分母口径一：**全 token**（输入 + 输出，含命中） $= 80 \times 1280 + 20 \times 18432 = 471040$ token/s $= 1695.7$ M tokens/h，得 $98/1695.7 = \mathbf{0.0578}$ **USD/M**。口径二：**仅输出** $= 61440$ token/s $= 221.2$ M tokens/h，得 $98/221.2 = \mathbf{0.443}$ **USD/M**。两口径差 **7.67 倍**——token 计数口径直接改变单位成本量级，跨系统比价前必须先对齐口径。每请求成本 $98/(100 \times 3600) = 2.72 \times 10^{-4}$ USD/req。

**DeepInfer 38 USD/GPU-hour 是收入，不是成本（doc L17 逐字，官方披露，核实 2026-11-24）**：「A single cluster currently serves 155 trillion tokens per day, with overall revenue of US\$38 per GPU-hour (at the off-peak price).」逐字关键词是 **overall revenue**（总收入）与 **off-peak price**（低谷价）——这是每 GPU 小时的收入口径，不是成本。把它代入式 (12) 的 $u$ 会算出「收入/M tokens」而非「成本/M tokens」，两者不可混用。

**155T tokens/day 的口径与隐含收入**。 $155 \times 10^{12} / 24 = 6.458 \times 10^{12}$ tokens/h $= 6.458 \times 10^6$ M tokens/h。设集群 GPU 数 $n_{\mathrm{GPU}}$ （**doc 未披露**），收入口径：

$$
\text{收入 USD/M tokens} = \frac{38 \thinspace n_{\mathrm{GPU}}}{6.458 \times 10^6} = 5.884 \times 10^{-6} \thinspace n_{\mathrm{GPU}} \qquad (13)
$$

**符号**： $n_{\mathrm{GPU}}$ = DeepInfer 单集群 GPU 数（doc 未披露，代入须显式声明假设；与教学负载的 $n_{\mathrm{GPU}} = n_P + n_D = 49$ 是不同部署，勿混）。式 (13) 对 $n_{\mathrm{GPU}}$ 线性，无 $n_{\mathrm{GPU}}$ 则只能给每 GPU 收入强度： $38 \times 24 = 912$ USD/GPU-day（低谷价口径）。**代入算例须显式声明**：设 $n_{\mathrm{GPU}} = 4096$ （纯教学假设，非披露），收入 $= 5.884 \times 10^{-6} \times 4096 = 0.0241$ USD/M tokens，每 GPU 日产 token $155 \times 10^{12}/4096 = 3.78 \times 10^{10}$ 。**且 155T 的 token 计数口径 doc 未定义**（输入 / 输出 / cache 命中计否不明），故即便给定 $n_{\mathrm{GPU}}$ ，该收入/M 也无法与他系统成本/M 直接比价——三重口径缺口（收入非成本、 $n_{\mathrm{GPU}}$ 未披露、计数未定义）必须同时声明。

> **坑**：把 38 USD/GPU-hour 当成本代入式 (12)。触发条件：doc L17 的「revenue」被略读为「cost」；现象：算出的「成本/M tokens」实为收入口径，且 off-peak 价非全时段均价，跨系统比价系统性失真；规避：引用必带「收入（off-peak）非成本」限定 + 声明 $n_{\mathrm{GPU}}$ 假设 + 声明 token 计数口径未定义。本章教学成本用独立假设的 2 USD/GPU-hour，与 38 无任何代入关系。

## 8. 命中 → 成本节省与 workload mix

命中率省下多少美元？路线（3 步）：先做 $h = 0$ 反事实重算 prefill GPU 与成本（命题 3 给节省分数闭式），再算 TTL 缩短的二阶命中损失，最后把 workload mix 的命中形态与成本杠杆汇总。关键一步是命题 3：全 token 分母与同价口径下，成本节省分数恰等于 prefill GPU 节省占总 GPU 的比例——decode 与存储不随命中省。

**$h = 0$ 反事实（固定 $\rho_P = 0.72$ 隔离命中变量）**。令两段 $h = 0$ ，重算式 (5)(6)：chatbot 全量 prefill $f = 2N \times 1024 + 262144 \times 1024^2 = 1.563 \times 10^{13}$ 、agent $f = 2N \times 16384 + 262144 \times 16384^2 = 3.161 \times 10^{14}$ FLOPs。需求率 $D_P = 80 \times 1.563 \times 10^{13} + 20 \times 3.161 \times 10^{14} = 7.573 \times 10^{15}$ FLOPs/s， $n_P \ge 7.573 \times 10^{15}/3.562 \times 10^{14} = 21.3$ ，取整 **22**（vs 命中时 4）。decode 不变（ $n_D = 45$ ，命中不省 decode，§4）。

**命题 3**（命中成本节省分数）：全 token 分母口径、两池同价 $u$ 下，命中相对 $h=0$ 的 USD/M 节省分数为 $\Delta n_P / (n_{P,0} + n_D)$ ，其中 $\Delta n_P = n_{P,0} - n_P$ 。**证明**（要点：式 (12) 直接比）。分母 $\dot{T}$ 不变（全 token 计数与命中无关），同价 $u$ 提出，USD/M $\propto n_P + n_D$ ；故节省分数 $= 1 - (n_P + n_D)/(n_{P,0} + n_D) = (n_{P,0} - n_P)/(n_{P,0} + n_D)$ 。∎ 代入： $\Delta n_P = 22 - 4 = 18$ ， $n_{P,0} + n_D = 67$ ，节省分数 $`18/67 = 26.9\%`$ 。成本从 $2 \times 67 = 134$ USD/h 降到 98 USD/h，USD/M 从 0.0790 降到 0.0578。**命中省的是 prefill 算力，不省 decode 并发**——这是命题 3 分母只含 $\Delta n_P$ 的结构原因（8.5 命题 1 边界 recap）。

**命中还是 SLO 可行性问题，不只是成本**。 $h = 0$ 时 agent 单请求 prefill 服务时间 $3.161 \times 10^{14}/4.9475 \times 10^{14} = 0.639$ s $\gt$ TTFT SLO 0.5 s——单 GPU 根本算不完，须 chunked prefill 或改并行度（4.3 / 8.1 指针，本反事实不展开）。命中把 agent prefill 压到 76.7 ms，才使 TTFT SLO 在单 GPU 上可行。所以命中不仅是 26.9% 的成本节省，更是长输入负载的 SLO 入场券。

**存储也随命中省**。 $h = 0$ 时写率升到全 token $\times \kappa L = 471040 \times 131072 = 61.74$ GB/s（命中取回变成重算 + 重写），写带宽闸 $61.74/5 = 12.3$ → 13 台、24h 闸 10.4 → 11 台，取最大 13 台（vs 命中时 4 台）。命中把「重算 + 重写」换成「读」，写带宽需求降 3.5 倍——存储节省与 prefill 节省同源。

**TTL 缩短是二阶效应（chatbot 反事实，非生产损失）**。官方披露（doc L211，核实 2026-11-24）：「a TTL of 72 hours for agent traffic and 7 days for chatbot traffic captures almost all reusable cache entries. The 72-hour TTL for agent traffic also spans a weekend, allowing requests on subsequent workdays to reuse the cache.」——生产 agent 72h / chatbot 7d **近乎无损**。反事实（doc L187 逐字）：「a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate」——**chatbot 流量缩至 3d TTL 损失 0.5–1% 命中率，是反事实口径，非生产损失**（2.2 回链，四路审查 B-4 / C-S2-4 限定）。

> **坑**：把「0.5–1% 命中损失」写成 agent 72h 的生产损失。触发条件：略读 doc L187 的流量类型与 TTL 条件；现象：损失主体是 chatbot、条件是 TTL 缩到 3 天，与 agent 72h 生产口径的流量类型和 TTL 值都不同；规避：引用必带「chatbot 反事实（TTL 3d）」限定。生产 agent 72h / chatbot 7d 近乎捕获全部可复用条目。

代入教学负载算二阶量级：chatbot $h$ 从 0.5 降 0.5–1 个百分点（口径：doc 未区分相对 / 绝对，教学按绝对百分点保守算），non-cached token 增 $80 \times 1024 \times (0.005 \sim 0.01) = 410 \sim 819$ /s，FLOPs 增约 $6.25 \times 10^{12} \sim 1.25 \times 10^{13}$ /s（含 attention 修正），折 GPU $\Delta n_P \approx 0.018 \sim 0.035$ 张、成本增 $0.035 \sim 0.070$ USD/h——相对命中整体省下的 18 张 GPU / 36 USD/h，TTL 微调是**二阶小量**。结论：有缓存 vs 无缓存是一阶（26.9%），TTL 在 72h/7d 基础上微调是二阶（<0.1%）；官方「近乎无损」与此一致。chatbot 命中长尾重（少量 KV 读极多次，doc L187），丢长尾条目的命中损失不成比例地高，故 chatbot TTL 取 7d 而非更短。

**workload mix 与成本杠杆汇总**。两段负载的命中形态决定各自主导的账：

| 账 | 主导段 | 占比 | 杠杆 |
| --- | --- | --- | --- |
| prefill 线性项 | chatbot | 55.6% | 命中率 $h$ （式 (4)） |
| prefill attention 项 | agent | 94.2% | 命中率 + 上下文压缩（2.3） |
| decode KV | agent | 96.8% | $\kappa$ 压缩 / MLA（2.3 / 7.2） |
| 存储写带宽 | 全段（non-cached + 输出） | — | 命中率（§8） |

成本杠杆方向表（每行回指公式与章）：

| 杠杆 | 方向 | 公式 / 指针 |
| --- | --- | --- |
| 命中率 $h$ ↑ | $n_P$ ↓、存储写 ↓ | 式 (4)(6)、命题 3、§8 |
| KV 量化 / MLA（ $\kappa$ ↓） | $n_D$ ↓（两闸同松） | 式 (8)(9)、2.3 / 7.2 |
| PD 分离（goodput ↑） | $n_{\mathrm{GPU}}$ ↓ | 1.1 式 (3)、8.2 |
| Wide-EP（ $e$ ↑） | decode 每卡权重流量 ↓ | 7.4、式 (8) 硬件侧 |
| TTL ↑（至 72h/7d） | $h$ ↑（二阶） | §8、doc L211 |
| 投机解码（有效 TPOT ↓） | 同 SLO 下并发容量 ↑ | 4.4 / 7.2 指针 |

**全账汇总（章脊口径，教学假设已声明）**：

| 量 | 值 | 来源 |
| --- | --- | --- |
| $n_P$ | 4 GPU | 式 (6)， $\rho_P = 0.72$ |
| $n_D$ | 45 GPU | 式 (8)(9) 取最大 |
| $n_{\mathrm{GPU}}$ | 49 | $n_P + n_D$ |
| $n_{\mathrm{stor}}$ | 4 服务器 | 式 (11)，写带宽闸 |
| 成本 | 98 USD/h | 式 (12)， $u = 2$ 教学假设 |
| USD/M（全 token） | 0.0578 | 式 (12) |
| USD/M（仅输出） | 0.443 | 式 (12)，口径差 7.67× |
| 命中节省 | 26.9%（18 GPU） | 命题 3 |
| DeepInfer 收入 | 38 USD/GPU-h（off-peak，非成本） | doc L17 |

## 总结

| 论断 | 依据 | 回指 |
| --- | --- | --- |
| M/M/1 纯排队尾 $\Pr(W_q \gt t) = \rho e^{-(\mu-\lambda)t}$ ；attainment 上界比均值余量紧（0.722 vs 0.877） | 命题 1、式 (1)(2) | §2、练习 2 |
| 三个 $\rho$ 上界（均值 / Kingman / attainment）取最小定 $\rho_P$ ；percentile 约束是 sizing 硬闸 | 式 (2)(3) | §2 |
| prefill 需求 = 混合 non-cached FLOPs 率（线性 + attention，命中折减）； $n_P$ = 需求 /（ $F_P \rho_P$ ）取整 | 式 (4)(5)(6) | §3、练习 3 |
| decode 两闸取最大：带宽闸（每步全读 KV）与容量闸（KV 常驻）；命中不省 decode | 式 (7)(8)(9) | §4、练习 3 |
| 配比是 workload / 硬件函数非常数；~1:1 是生产经验值；64P+64D 是 EP64 最小 decode 域量子化 | doc L91、8.2 式 (8) recap | §5 |
| 存储四口径：raw 614.6 / usable 512.2 / 物理 17.28 TB/GPU；两口径 GPU/server 恒等 35.6 | 命题 2、式 (10) | §6、练习 1 |
| 100 PiB → ~183（raw）/ ~220（usable）服务器，口径歧义须声明 | 式 (10) 算术 | §6 |
| 存储 sizing 四约束取最大；教学负载写带宽闸主导（4 台） | 式 (11) | §6、练习 3 |
| USD/M =（ $u_P n_P + u_D n_D$ ）/（tokens/h ÷ 1e6）；token 计数口径差 7.67× | 式 (12) | §7 |
| DeepInfer 38 USD/GPU-h = 收入（off-peak）非成本；155T 计数未定义、 $n_{\mathrm{GPU}}$ 未披露 → 代入须三重声明 | doc L17、式 (13) | §7、练习 4 |
| 命中节省分数 = $\Delta n_P/(n_{P,0}+n_D)$ ；命中省 prefill 不省 decode；TTL 微调二阶 | 命题 3 | §8、练习 3 |
| chatbot 3d TTL 损 0.5–1% 命中是反事实非生产损失；生产 agent 72h / chatbot 7d 近乎无损 | doc L187 / L211 | §8、2.2 回链 |

## 练习

1.（计算 · [10]）存储 EC 算术与两口径等价（章脊 §6 锚）。(a) 由 20 × 30.73 TB 算 raw/server；(b) 由 14.4 TB 逻辑/GPU 与 EC 10+2 算物理/GPU；(c) **证明命题 2**（raw ÷ 物理/GPU = usable ÷ 逻辑/GPU），并代入验证两口径同给 35.6 GPU/server。校验：(a) 614.6 TB、(b) 17.28 TB、(c) 两式同乘 12/10 相消，与 §6 式 (10) 表一致。

2.（推导 · [20]）attainment 上界（命题 1 应用）。(a) 验证式 (1) 在 $t=0$ 给 $\Pr(W_q \gt 0)=\rho$ 、对 $t$ 积分给 $W_q = \rho/(\mu-\lambda)$（与 1.3 式 (9) 一致）；(b) 设 $\mu = 35.6$ req/s、 $q = 0.2$ s、 $\theta = 0.9$ ，用二分两步定位 $\rho$ 上界（先算 $\rho=0.7$ 与 $0.75$ 的尾概率）；(c) 先预测 $q$ 减半到 0.1 s 时 $\rho$ 上界的变化方向，再计算对照。校验：(a) 积分恰为 $\rho/(\mu-\lambda)$ ；(b) $\rho \le 0.722$ ，与 §2 一致；(c) 上界应**下降**（预算更紧、余量更大）， $\rho \le 0.531$ 。

3.（计算 · [30]）纯 chatbot 负载全账重算（口径切换）。把章脊混合负载换成纯 chatbot： $\lambda = 100$ req/s、 $T_{\mathrm{in}} = 1024$ 、 $T_{\mathrm{out}} = 256$ 、 $h = 0.5$ ，其余硬件 / SLO / 价格不变， $\rho_P$ 固定 0.72（口径：隔离负载变量，不重解 §2）。(a) 用式 (4)(5)(6) 算 $n_P$；(b) 用式 (7)(8)(9) 算 $n_D$（两闸取最大）；(c) 用式 (11) 算存储服务器数（先算写 / 读率与 24h 容量）；(d) 用式 (12) 算全 token 口径 USD/M，并与混合负载的 0.0578 对照，说明哪段主导哪笔账。校验：(a) $n_P = 3$（需求率 $7.886 \times 10^{14}$ FLOPs/s）；(b) $n_D = 2$（容量闸 1.80 主导）；(c) 3 台（写带宽闸 2.01 主导）；(d) 分母 460.8 M tokens/h、USD/M = 0.0217，混合负载是其 2.66 倍——agent 段主导 decode KV 与存储。

4.（计算 · [30]）DeepInfer 收入口径对账（doc L17）。(a) 把 155T tokens/day 折成 tokens/h 与 M tokens/h；(b) 用式 (13) 写收入 USD/M 关于 $n_{\mathrm{GPU}}$ 的表达式，代入 $n_{\mathrm{GPU}} = 4096$ 与 $n_{\mathrm{GPU}} = 8192$（**显式声明这是教学假设、doc 未披露 $n_{\mathrm{GPU}}$**）；(c) 说明为什么 38 不能代入式 (12) 的成本 $u$，以及要把收入换成成本还需哪些假设（毛利率、token 计数口径、off-peak 与全时段均价差）。校验：(a) $6.458 \times 10^{12}$ tokens/h；(b) 系数 $5.884 \times 10^{-6} \times n_{\mathrm{GPU}}$，对 $n_{\mathrm{GPU}}$ 线性；(c) 38 是「overall revenue（off-peak）」逐字口径，非成本，且 155T 计数口径未定义。

## 参考

- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-24）——155T tokens/day 与 38 USD/GPU-hour 收入（off-peak）口径（L17）、~1:1 经验配比与 64P+64D 最小高效单元（L91）、存储 20×30.73 TB / ≥40 GB/s 读 / 5 GB/s 写 / 24h 保留（L101）、14.4 TB/GPU 逻辑（L105）、EC 10+2 与读路径张力（L111 / L113）、GC LRU watermark（L115）、~80% 离线基准（L155）、100 PiB 与 chatbot 3d 反事实 0.5–1%（L187）、TTL agent 72h / chatbot 7d 近乎无损（L211）、autoscaling 信号（L251）、decode 产出 KV 仍写盘（L183）
- [Zhong et al., 2024](../references.md)：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，arXiv:2401.09670，[A]（key: distserve）——goodput 定义（per-GPU 归一 + attainment 目标，1.1 式 (3) 沿用）、sizing 与 placement 优化、simulator 定 $\rho$
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 SXM BF16 989.5 TFLOPS dense、HBM 3.35 TB/s、80 GB
- [DeepSeek-AI, 3FS](../references.md)：教学参照（key: 3fs）——开源 repo 为 CRAQ 链式复制无 EC，EC 10+2 属生产官方披露，引用区分口径
- 纯文字指引（不挂链接）：1.1（goodput 定义式 (3)、Azure trace / ShareGPT 归属、benchmark 陷阱）、1.2（prefill / decode 算力带宽账、 $\kappa$ 几何式 (3)）、1.3（M/M/1 式 (8)(9)、Kingman 式 (10)、余量规则式 (11)、 $\mu$ 状态依赖提点）、2.2（命中率 → prefill 节省命题 2、TTL 反事实限定）、2.3（KV 量化）、2.4（3FS 分层与 EC 口径）、4.3（chunked prefill）、7.2（V4.1 混合注意力与量化 KV）、7.3（kernel 栈与 KV 量化格式）、7.4（Wide-EP 最小 decode 域、EPLB）、8.1（权重读项与并行度）、8.2（xPyD 配比式 (5)-(8)、命题 4 双重无关性、DistServe simulator）、8.5（伸缩触发账式 (1)-(4)、冗余 quota、容量闸）
