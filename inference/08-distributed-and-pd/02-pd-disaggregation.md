# 8.2 PD 分离原理（Prefill-Decode Disaggregation）

> prefill 与 decode 同卡时，一条 32768 token 的 prompt 能把同批 decode 的 TBT 打到 52 倍尖刺，TTFT 与 TPOT 两个 SLO 还被锁在同一份并行度决策上——分离之后 per-GPU goodput 能提多少、两池 GPU 配比怎么定、KV 传输比重算便宜的条件是什么？本章推导 PD 分离的原理账：干扰与 goodput 分解（DistServe）、phase splitting 与异构池（Splitwise）、xPyD 配比、传输 vs 重算的带宽阈值、何时分离的决策边界 | 前置：1.1（TTFT / TPOT / goodput 定义式 (3)）、1.2（prefill / decode 算力带宽账与 $\kappa$ 几何）、1.3（datasheet 参数与 M/M/1 排队）、4.3（stall 账与 chunked prefill）、2.4（KV offload 与 Mooncake）；8.1（serving 并行）、8.3（KV 传输）、8.4（系统实现对照）为同波兄弟章，本章只交付原理账、实现给指针 | 锚点：DistServe（[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]，LLMServe/DistServe）、Splitwise（[Patel et al., 2024](../references.md)，arXiv:2311.18677，[B]，仅作者模拟器）、Mooncake（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]）、Sarathi-Serve（[Agrawal et al., 2024](../references.md)，arXiv:2403.02310，[A]，对照）、Beyond-the-Buzz（[Mitra et al., 2025](../references.md)，arXiv:2506.05508，[B]，proprietary 模拟器）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）、NVIDIA datasheet（[NVIDIA datasheet](../references.md)，[C]，官方规格披露）

**本章口径**（章脊锁定）：主数值口径沿 1.2 / 1.3 / 4.3 的 8B dense bf16 @ H100 SXM—— $L = 32$ 、 $d_{\mathrm{model}} = 4096$ 、 $H_{kv} = 8$ 、 $d_{\mathrm{head}} = 128$ 、 $N = 7.50 \times 10^{9}$ （matmul 口径）、 $\kappa = 4096$ B/token/层（GQA-8，1.2 式 (3)）、权重字节 $W = 1.606 \times 10^{10}$ B、BF16 dense 峰值 989.5 TFLOPS、HBM 3.35 TB/s（[NVIDIA datasheet](../references.md)，[C]）；workload $\lambda = 10$ req/s、 $T_{\mathrm{in}} = 4096$ 、 $T_{\mathrm{out}} = 512$ 、TTFT SLO 0.5 s、TPOT SLO 30 ms（沿 1.1 教学数值）；prefill MFU（Model FLOPs Utilization，有效算力占峰值比）取 0.5、两池利用率余量 $\rho_P = \rho_D = 0.8$ 为教学假设（实际值由 profiling 定，§2 的 simulator 即干这件事）。两处换口径均声明原因：§2 的 goodput 数值用 DistServe 论文的 13B / A100 口径（沿 1.1 §3 先例，为对账论文原文）；§5 的传输 vs 重算换 V3 MLA FP8 KV @ H100 FP8（原因：MLA + 量化 KV 是跨 DC 传输的生产形态，且与 DeepInfer 官方披露直接可比）。**本章 $\lambda$ = 到达率（req/s），非 training GAE 的 $\lambda$ 参数**； $n_P / n_D$ = P / D 两池的 GPU 数（全章统一口径；实例数 = GPU 数 ÷ 每实例 GPU 数，§2 声明 $a_P / a_D$ ）； $\beta$ = 每 GPU HBM 带宽（沿 1.3 符号）；bytes 不占字母，一律文字「字节」或 $\mathrm{bytes}$ ；batch 用文字或并发数 $C$ 表达，本章不用字母 $B$ 。

## 1. 干扰：为什么一张卡伺候不了两个 SLO

32768 token 的 prefill 计算下界约 781 ms，而同批 64 条序列的 decode iteration 只要 15.1 ms——混进同一个 iteration，TBT 尖刺约 52 倍（4.3 §1 的账 recap：8B 口径、H100 BF16 峰值下界）。中等长度也逃不掉：4096 token 的 prefill 66.5 ms，已是 decode 步时的 4.4 倍。机制是迭代级调度的直接推论：batch 是执行的原子单位，批内所有序列共享 iteration 时长（1.1 命题 1），所以一条长 prefill 就能劫持全批 decode 序列的下一 token 延迟。所谓 stall，是指 decode 序列因所在 iteration 混入非 decode 工作而等待、TBT 远超正常步时的现象（口径：按逐间隔 TBT 样本度量，不是请求级 TPOT 均值——均值会被几百个正常间隔稀释，4.3 §1 同注）。

尖刺只是第一层问题。第二层是**决策耦合**：colocated（同卡混部）形态下 prefill 与 decode 共享同一份并行度配置与 batch 决策，而两相的偏好相反——prefill compute-bound，要 intra-op 并行（TP，算子内切分）压单请求执行时间换 TTFT；decode memory-bound，要大 batch 加 replication / inter-op 并行（PP，层间流水）扩容量换 TPOT 与吞吐（DistServe §2.3 / §3，[Zhong et al., 2024](../references.md)，[A]）。耦合之下配置只能按更严的那个 SLO 选，另一相被迫 over-provision（过度配置）。优先级调度也救不了：prefill 优先则 decode 排队等待（queuing delay 涨），decode 优先则 prefill 等待，纯 decode 批又让算力闲置（decode memory-bound，算力不是瓶颈）——两种优先序都逃不出耦合（DistServe §2.3「Ineffective scheduling」）。

**命题 1**（colocated goodput 受更严 SLO 约束）。记 $`\mathcal{G}_{\mathrm{TTFT}}`$ 、 $`\mathcal{G}_{\mathrm{TPOT}}`$ 分别为只满足 TTFT SLO、只满足 TPOT SLO 时的最大请求率（attainment（达标率）目标 $\theta$ 相同），则 colocated 的 per-GPU goodput（1.1 式 (3) 口径）为

$$\mathcal{G}_{\mathrm{co}} = \min\left( \mathcal{G}_{\mathrm{TTFT}},\ \mathcal{G}_{\mathrm{TPOT}} \right) \qquad (1)$$

**证明**（路线：单调性 → 可行集是区间 → 交集的最大值 = 端点的 min，三步）。其一，attainment 对 $\lambda$ 非增：到达率越高，排队越长、干扰越密，达标占比只降不升（1.3 §5 的排队机制）。其二，故每个 SLO 的可行集 $\lbrace \lambda : \mathrm{Attain}(\lambda) \ge \theta \rbrace$ 是区间 $`[0, \mathcal{G}_X]`$ 。其三，colocated 可行集是两区间之交，交集的最大值就是两端点的 min。∎

正例（DistServe Figure 1，13B / A100 / 输入 512 / 输出 64 / $\theta$ = 90%）：colocate 单卡 goodput ≈ 1.6 rps，论文明说该值「constrained by the more stringent one of TTFT and TPOT requirements」——式 (1) 的直接读数。边界反例：把 TPOT SLO 放宽到人类阅读速度（chatbot 场景，约 250 words/min），约束退化为 TTFT 单项、min 变单项——但决策耦合仍在，over-provision 不随单项 SLO 放宽而消失。

所谓 **PD 分离（PD disaggregation）**，是指 prefill 与 decode 跑在不同实例上的部署形态：prefill 实例只处理 prompt、产出第一个 token，把 KV cache（连同首 token 元数据）传给 decode 实例，由后者完成后续生成。实例（instance）沿 DistServe 的定义：持有一份完整模型权重的资源单元，模型并行时可以跨多张 GPU。分离同时攻击两层问题：物理隔离两相，消除干扰（代价是 KV 传输，§5 算）；两相各自独立选并行度与 batch，解除决策耦合。时序对照如下（8B 口径，4.3 §1 数值）：

```
colocated：prefill 与 decode 共享同一组 GPU（迭代级调度）
GPU 时间轴 ──[decode 15.1ms][── 新请求整条 prefill 781ms ──][decode 15.1ms]──→
批内 r1 的 TBT：  15.1 ms             781 ms（尖刺 52×）            15.1 ms

PD 分离：两池各自迭代
P 池 ──[prefill r2][prefill r3][prefill r4]──→       compute-bound，小 batch（§2）
             │ KV(r2) + 首 token 元数据（传输账 §5）
D 池 ──[decode 混合批 r1,r2 15.1ms][15.1ms][15.1ms]──→  memory-bound，大 batch
r1 的 TBT：   15.1 ms       15.1 ms      15.1 ms     无尖刺
```

本章其余部分围绕四笔账展开：分离后 goodput 能提多少（§2）、两池 GPU 配比怎么定（§4）、KV 传输与重算哪个便宜（§5）、净收益何时为正（§6）。§3 先补两个生产级变体——Splitwise 的异构池与 Mooncake 的 KVCache-centric 架构。

## 2. DistServe：goodput 最大化的分解

分离后 per-GPU goodput 从 1.6 rps 提到 3.3 rps（2.1 倍）——这是 DistServe 的 13B / A100 / 90% attainment 答案（论文 §1）；摘要的全评估口径是「7.4× more requests or 12.6× tighter SLO」（>90% 请求满足延迟约束）。本节把这个数从两相各自的速率重推导出来，再沿推导结构讲论文的 placement 算法（placement 的精确定义见下文推导路线段）。本节数值用论文 13B / A100 口径（章脊已声明）。

**goodput 定义 recap**（1.1 式 (3)，符号逐字沿用，不重推）：

$$\mathcal{G} = \frac{1}{n_{\mathrm{GPU}}} \max \lbrace \lambda : \mathrm{Attain}(\lambda) \ge \theta \rbrace$$

定义的两个要件缺一不可：per-GPU 归一（分母是配置的 GPU 总数），以及 attainment 目标约束下的**最大可达请求率**（优化语义）。所谓 goodput，是指在给定 SLO 达标率约束下每 GPU 可服务的最大请求率——不是「达标请求的裸吞吐」。

> **坑**：把 goodput 写成「达标请求的吞吐」。触发条件：丢掉了 max 的优化语义或 per-GPU 归一；现象：两个配置达标吞吐相同但一个还能继续加压、另一个已经到顶，裸吞吐读数无法区分，容量规划（9.1）与配比优化（§4）全部失去目标函数；规避：一律沿 1.1 式 (3) 的 $\max$ + attainment + 归一三件套。

推导路线：先写两池串联的稳态瓶颈（式 (2)），再证最优配比平衡两池、得调和形容量（式 (3)），最后处理整数 replication 的取整损失——关键一步是平衡条件把双变量优化约成单变量。所谓 placement，是指并行度配置 + 每类实例数 + 物理放置三者的联合方案（论文 §4 定义），goodput 最大化就是对 placement 的优化。

分离后每相只对自己的 SLO 负责。记 $g_P$ = prefill 池在 TTFT SLO 约束下的 per-GPU goodput（rps/GPU）， $g_D$ = decode 池在 TPOT SLO 约束下的 per-GPU goodput； $a_P / a_D$ = 每实例 GPU 数，每实例 goodput 分别为 $a_P g_P$ （P）与 $a_D g_D$ （D）。论文 Figure 1 的实测：13B / A100 / 90% 下 $g_P$ = 5.6 rps/GPU、 $g_D$ = 10 rps/GPU（单卡实例）。两相速率能各自独立优化，前提是相间的 KV 传输不进入任何一相的 SLO——这个前提的定量版本在本节末（90 Gbps 账）。

**命题 2**（稳态瓶颈与平衡）。 $n_P$ 张 prefill GPU、 $n_D$ 张 decode GPU 串联服务同一请求流（每请求先 prefill 后 decode），稳态、work-conserving（有活就不闲着）调度、传输非阻塞，则系统请求率与 per-GPU goodput 为

$$R_{\mathrm{sys}} = \min\left( n_P g_P,\ n_D g_D \right), \qquad \mathcal{G} = \frac{R_{\mathrm{sys}}}{n_P + n_D} \qquad (2)$$

连续松弛下（GPU 数可取分数），固定总 GPU 数时 $\mathcal{G}$ 的最优解满足平衡条件 $n_P g_P = n_D g_D$ 。**证明**（要点：流量守恒 + 交换论证）。可达性：两相是串联管道，prefill 池每秒最多产出 $n_P g_P$ 个完成 prefill 的请求，decode 池每秒最多接纳 $n_D g_D$ 个，稳态流量守恒取 min；最短队列 dispatch（论文 §4.3 的 FCFS，先到先服务）使瓶颈池饱和、另一池有空闲，min 可达。最优性：若 $n_P g_P \lt n_D g_D$ ，把 $\Delta$ 张 GPU 从 D 池挪到 P 池，min 的前项升、后项降，但只要不平衡仍在，前项仍低于后项，min 严格升——与最优矛盾；故最优必平衡（若 GPU 数另受约束则可能取边界解，此处不展开）。∎ 代入论文数值（单卡实例， $g_P$ = 5.6、 $g_D$ = 10）：2P+1D 的 $R_{\mathrm{sys}} = \min(2 \times 5.6,\ 10) = 10$ rps，瓶颈在 decode 侧——式 (2) 的直接读数。

平衡时代入 $n_P = R/g_P$ 、 $n_D = R/g_D$ （ $R$ 为系统率），总 GPU 数 $R(1/g_P + 1/g_D)$ ，得

$$\mathcal{G}^{\ast} = \frac{g_P g_D}{g_P + g_D} = \frac{1}{\dfrac{1}{g_P} + \frac{1}{g_D}} \qquad (3)$$

**符号**： $\mathcal{G}^{\ast}$ = 平衡配比下的 per-GPU goodput 上界（连续松弛）； $g_P / g_D$ 见式 (2) 前声明。式 (3) 是调和形：两相串联的有效容量像并联电阻的倒数和，永远小于 $\min(g_P, g_D)$ 、不低于其一半（ $g_P = g_D$ 时恰等于一半，边界取等）。代入论文数值： $g_P$ = 5.6、 $g_D$ = 10 得 $\mathcal{G}^{\ast}$ = 56/15.6 ≈ 3.59 rps/GPU。

> **提点**：调和形就是「分离的代价」的账面形态——两池都得配 GPU，per-GPU 容量按倒数和折损。它仍远高于 colocate 的 1.6 rps，因为干扰消除把 $g_P$ 、 $g_D$ 各自抬到了单 SLO 的容量极限（命题 1 的 min 约束被解除）。

**命题 3**（单调性 → 两级分解）。 $\mathcal{G}^{\ast}$ 对 $g_P$ 、 $g_D$ 各自严格单调增。**证明**（要点）：固定 $g_D$ ， $\mathcal{G}^{\ast} = g_D / (1 + g_D/g_P)$ ， $g_P$ 增则 $g_D/g_P$ 减、分母减、整体严格增；对 $g_D$ 对称。∎ 推论：只要两相的可行配置集互相独立（高带宽 placement 下传输时间可略， $g_P$ 只依赖 prefill 配置、 $g_D$ 只依赖 decode 配置），**各自独立最大化 per-GPU goodput 就得到全局最优**——不必联合搜索。这正是论文 Algorithm 1 的两级结构：第一级对每相枚举可行并行配置（intra-op × inter-op，受单卡显存约束），用 simulator 找 per-GPU goodput（论文记法 config.goodput / config.num_gpus）最大的配置；第二级 replication 凑总率。

整数部署把连续最优打了折扣。论文 Algorithm 1 末行按实例粒度向上取整：目标总率 $R$ 下，prefill 实例数 $\lceil R/(a_P g_P) \rceil$ 、decode 实例数 $\lceil R/(a_D g_D) \rceil$ 。代入 $R$ = 10 rps、单卡实例： $\lceil 10/5.6 \rceil = 2$ 个 prefill 实例 + $\lceil 10/10 \rceil = 1$ 个 decode 实例，即 2P+1D， $R_{\mathrm{sys}}$ = 10 rps（式 (2)），摊到 3 张 GPU 得 3.33 rps/GPU——比连续最优 3.59 低约 7%（取整损失），仍是 colocate 1.6 的 2.1 倍。反向配比 1P+2D： $R_{\mathrm{sys}} = \min(5.6,\ 20) = 5.6$ rps，per-GPU 掉到 1.87——瓶颈切到 prefill 侧。**配比是优化变量不是常数**，这是 §4 xPyD 推导的入口（1.1 练习 3 同账）。

simulator 与二分搜索。attainment 没有解析闭式：真实 workload 的输入输出长度分布不齐、到达不规则，给定配置的达标率只能仿真估计。DistServe 的做法（§4.1 + Appendix A）：从历史 trace 拟合到达与长度分布、重采样新 trace 喂 simulator；simulator 用 FLOPs + 内存访问分析 + latency model 估执行时间（§6.4 验证与真实 profiling 吻合）；对每个配置，在 $\lambda$ 上**二分搜索**满足 attainment 目标的最大率——谓词「 $\mathrm{Attain}(\lambda) \ge \theta$ 」对 $\lambda$ 单调（命题 1 证明的第一步），二分合法，每步一次仿真。搜索复杂度 $O(N M^{2})$ （ $N$ = 每实例节点上限、 $M$ = 每节点 GPU 数，通常 8），最大设置求解不到 1.3 分钟（§6.5）——快到可以随 workload 漂移周期性 replan（§4.3）。

并行度偏好可以从排队论读出形状。prefill 单请求即饱和 GPU（13B 512-token on A100）、长度均匀时 FCFS 无 batching，是 M/D/1 队列（1.3 §5 M/M/1 的定长服务版）：执行时间 $D$ 恒定、Poisson 到达率 $\lambda$ 、 $\lambda D \lt 1$ ，平均 TTFT 为

$$\mathrm{Avg\_TTFT} = D + \frac{\lambda D^{2}}{2(1 - \lambda D)} \qquad (4)$$

**符号**： $D$ = 单请求 prefill 执行时间；第一项执行、第二项排队（论文 Eq. 1，符号译为本章 $\lambda$ ）。代入抽象口径 $D$ = 0.1 s： $\lambda$ = 5 时 TTFT = 0.15 s， $\lambda$ = 9 时 0.55 s——利用率 0.5 → 0.9，排队项翻 9 倍（1.3 的 $1/(1-\rho)$ 极点同形）。两相并行度偏好由此分化：2 路 inter-op 把每 stage 服务时间减半、容量翻倍，高率下排队项主导故 inter 占优；2 路 intra-op 把执行时间压到 $D/K$ （ $1 \lt K \lt 2$ ，通信折损）但容量不变，低率下执行项主导故 intra 占优（论文 §3.1 Eq. 2/3 对比）。decode 侧类似：TPOT SLO 紧时要 intra 压步时，batch 凑大后 inter / replication 线性扩吞吐（§3.2）。colocate 时这份偏好表只能二选一，分离后各取所需——决策解耦的第二笔收益。

placement 算法按集群带宽分两版。**Algorithm 1（高节点亲和）**：跨节点有 InfiniBand 级高带宽时，KV 传输可略、两相解耦，命题 3 的两级分解成立，P / D 实例可放任意两节点。**Algorithm 2（低节点亲和）**：跨节点带宽受限、只有节点内 NVLink 高带宽时，把同一 inter-op stage 的 prefill 段与 decode 段 colocate 进同一节点（instance segment），强制 KV 传输只走 NVLink；枚举 inter-op 度 1 到 $N$ + 节点内配置（get_intra_node_configs），约束同节点内 P 段与 D 段的 GPU 数之和不超过 $M$ （每节点 GPU 数，通常 8），联合 simulate 选 per-GPU goodput 最优再 replication。此时命题 3 的分解前提失效——不是传输时间不可略，而是 placement 约束把两相的可行集耦合了（P 段选多大，D 段就只剩多少），必须联合枚举。大模型还有显存约束：175B（350 GB）一对 P+D 实例要 700 GB，8 × 80 GB = 640 GB 的单节点装不下（论文 §4.2 例），约束一并进搜索。

传输开销的定量前提（论文 §3.3）：OPT-66B 单条 512-token 请求的 KV ≈ 1.13 GB；10 rps 到达率需要 11.3 GB/s = 90 Gbps 才能使传输开销不可见。InfiniBand 800 Gbps 或节点内 NVLink（A100 600 GB/s）都远高于此——「传输可略」不是假设，是带宽账的结论；低带宽集群达不到，才有 Algorithm 2 的 placement 约束。运行时还有两个配套设计（§4.3）：KV 用 **pull** 语义——decode 实例按需取 KV，prefill 实例的 HBM 当排队缓冲，burst 到达时不会压垮 decode 侧显存（push/pull 的传输语义与实现归 8.3）；workload 漂移触发 replan，重跑 placement 算法秒级、权重重载分钟级，远快于小时级的负载变化。

边界声明：DistServe 有官方实现 LLMServe/DistServe（[A]，SwiftTransformer C++ backend，论文脚注自链）；本章只讲论文算法与账，不做引擎代码走读。各引擎的 PD 实现对照（vLLM KV connector / SGLang disaggregation / Dynamo / TRT-LLM / DeepInfer VE-P-D）归 8.4；SGLang 的 bootstrap 与 KV 传输实现已在 6.3 走读、DeepInfer 生产形态已在 7.4 走读，本章一律指针不重复。

## 3. Splitwise 与 Mooncake：phase splitting 与 KVCache-centric

decode 相能不能用上一代 GPU 跑？Splitwise 的答案是能：token generation 是 memory-bound，H100 相对 A100 的 3.43 倍算力优势用不上，功耗却高 1.75 倍（论文 Table I）；memory-to-compute 比反而 A100 占优。这是 phase splitting 的硬件侧论据——两相拆到独立机器后，可以按相独立 provision（资源配置）、配异构硬件（[Patel et al., 2024](../references.md)，[B]，ISCA 2024）。证据等级注：Splitwise 官方代码只有作者模拟器 mutinifni/splitwise-sim 与一个 vLLM PR，代码不全——本节按论文讲机制与账，不推测生产代码形态。

负载证据来自 Azure 生产 trace：两个 Azure LLM inference 服务（coding 与 conversation）2023-11-11 的 trace，释放子集为 Azure/AzurePublicDataset 的 AzureLLMInferenceDataset2023（CC-BY，只给尺寸无 prompt 内容；schema 与归属 recap 1.1 §4，[Microsoft Azure, AzureLLMInferenceDataset2023](../references.md)，教学参照）。characterization（负载画像）的关键读数：coding 服务输出 token 极少，token 相超过 20% 的时间在跑单 token batch——batching 极差。phase splitting 让 token 池专职攒 batch，prompt 池专职算力，两相各自的利用率都上来。

集群调度是三池结构（§IV-A）：prompt pool / token pool / **mixed pool**。机器按需动态进出 mixed pool：队列超阈值时从对侧池借机（减碎片 + 高负载保 SLO），mixed 机保留原身份、对侧任务队列清空即回原池；re-purposing 是粗粒度、不频繁的操作。CLS（cluster-level scheduler）给请求同时指派 prompt 机与 token 机，让 KV 传输与 prompt 计算重叠。KV 传输本身分两档（§IV-C）：serialized（prompt 算完、首 token 产出后整块传，传完才能出下一 token）直接顶高 max TBT 与端到端延迟；layer-wise（每层算完即异步传该层 KV，prompt 继续下一层）把传输藏进计算，还让 token 相早启动、prompt 机 KV 显存早释放。代价是逐层细粒度同步，小 prompt 可能干扰反增 TTFT——但小 prompt 的 KV 本来就小，不需要 layer-wise。所以 Splitwise 按启动时已知的 token 数选档：小 prompt serialized、大 prompt layer-wise（H100 上阈值 512 token）。实测（§VI-A）：layer-wise 的非重叠残余近似常数——A100（200 Gbps）约 8 ms、H100（400 Gbps）约 5 ms；coding trace 无 batch 口径下第二 token 延迟开销 16.5%，serialized 是 64%。集群级结果（摘要口径）：1.4 倍吞吐 @ 低 20% 成本，或 2.35 倍吞吐 @ 同成本同功耗。

Mooncake 走的是另一条杠杆（[Qin et al., 2024](../references.md)，[A]；架构与存储账 recap 2.4 §5，此处不重推）：KVCache-centric——prefill / decode 集群分离之外，把 GPU 集群**闲置的 CPU / DRAM / SSD** 池化成 disaggregated KVCache cache，调度中心从 GPU 换成 KV；KVCache-centric scheduler 最大化有效吞吐并保 SLO，过载用 prediction-based early rejection（预测将违约的请求提前拒绝，保已接请求的达标率）。摘要账：模拟场景至多 525% 吞吐提升、真实 workload 多服务 75% 请求（Kimi 生产）。transfer engine 的多网卡池化与拓扑感知归 8.3，分层命中成本归 2.4。

三个系统对照一句话：DistServe 优化 placement（goodput 目标），Splitwise 优化硬件构成（成本 / 功耗目标），Mooncake 优化 KV 复用（有效吞吐目标）——同一分离形态下的三个正交杠杆，生产系统通常三个都要。

## 4. xPyD：两池 GPU 配比怎么定

prefill 配多少 GPU、decode 配多少？所谓 xPyD，是指 x 份 prefill 配 y 份 decode 的部署配比记法（文字记法，如 2P1D、64P+64D；份的粒度是实例或 GPU，随文声明）。答案不是 1:1 这种常数，而是 workload 形状与硬件参数的函数——本节从头推导。路线四步：先由 prefill 算力需求定 $n_P$ ；再用 Little 定律得 decode 并发、折成 KV 带宽需求定 $n_D$ ；然后相比，观察两个相消；最后代数字、讨论边界。关键一步是第三步： $\lambda$ 与 TPOT 在配比里精确相消，剩下的只有长度分布与硬件参数。

**P 侧：算力需求**。每请求 prefill 约 $2 N_{\mathrm{act}} T_{\mathrm{in}}$ FLOPs（1.2 式 (1) 参数项；二次项在 $T_{\mathrm{in}}$ = 4096 时占 6.7%，折进 MFU 与利用率余量，全账见 1.2），到达率 $\lambda$ 下需求率为 $\lambda \cdot 2 N_{\mathrm{act}} T_{\mathrm{in}}$ 。每 GPU 有效供给 $F_P \rho_P$ ，故

$$n_P \ge \frac{\lambda \cdot 2 N_{\mathrm{act}} T_{\mathrm{in}}}{F_P\thinspace  \rho_P} \qquad (5)$$

**符号**： $F_P$ = 每 GPU 有效 prefill 算力 = 峰值 × MFU（本章 989.5 × 0.5 = 494.75 TFLOPS）； $\rho_P$ = 利用率余量； $N_{\mathrm{act}}$ = 激活参数量（dense 模型 $N_{\mathrm{act}} = N$ ）。口径注：prefix cache 命中会折减有效 $T_{\mathrm{in}}$ （2.2 命中账），本节按全 miss 算。等价读法： $\mu_P = F_P / (2 N_{\mathrm{act}} T_{\mathrm{in}})$ 是每 GPU 服务率，式 (5) 就是 $\lambda / (n_P \mu_P) \le \rho_P$ ——1.3 §5 的 M/M/1 利用率约束： $\rho \to 1$ 时排队延迟爆炸，余量必须留。代入章脊：每请求 $6.144 \times 10^{13}$ FLOPs，需求率 $6.144 \times 10^{14}$ FLOPs/s， $F_P \rho_P = 3.958 \times 10^{14}$ ，得 $n_P \ge 1.55$ ； $\mu_P$ = 8.05 req/s/GPU。

**D 侧：并发 → 带宽需求**。稳态并发由 Little 定律（recap 1.1 / 1.3）：每请求占据一个 decode 槽位 $T_{\mathrm{out}}$ 步、每步 TPOT，故

$$C = \lambda\thinspace  T_{\mathrm{out}}\thinspace  \mathrm{TPOT} \qquad (6)$$

**符号**： $C$ = decode 并发序列数；TPOT 取 SLO 上限（保守口径：步时越慢并发越高、所需 GPU 越多，实际操作点快于 SLO 时此账是上界）。代入： $C = 10 \times 512 \times 0.03 = 153.6$ 条。每条序列每步读 KV $\kappa L \bar{T}$ 字节（1.2 §3 recap）， $\bar{T} = T_{\mathrm{in}} + T_{\mathrm{out}}/2$ 为 decode 期间平均上下文长（从 $T_{\mathrm{in}}$ 线性涨到 $T_{\mathrm{in}} + T_{\mathrm{out}}$ ）； $C$ 条每 TPOT 读一遍，带宽需求除以每 GPU 有效带宽得

$$n_D \ge \frac{C\thinspace  \kappa L\thinspace  \bar{T}}{\mathrm{TPOT}\ \beta_D\thinspace  \rho_D} = \frac{\lambda\thinspace  T_{\mathrm{out}}\thinspace  \kappa L\thinspace  \bar{T}}{\beta_D\thinspace  \rho_D} \qquad (7)$$

**符号**： $\beta_D$ = decode 池每 GPU HBM 带宽； $\rho_D$ = 利用率余量； $\kappa$ 、 $L$ 见 1.2 式 (3)。代入章脊： $\kappa L$ = 131072 B/token， $\bar{T}$ = 4352，每序列每步 $5.704 \times 10^{8}$ B，带宽需求 $2.92 \times 10^{12}$ B/s = 2.92 TB/s； $\beta_D \rho_D = 2.68$ TB/s，得 $n_D \ge 1.09$ 。步时交叉核对（口径注：KV 流量账不含权重读项——权重每步读一遍、与并发无关，由并行度与 batch 决策承担，8.1 / 7.4）：每 GPU 摊 141 条并发，KV 项 24.0 ms + 权重读 4.8 ms = 28.8 ms ≤ 30 ms 预算，账自洽。若权重项主导（小 batch），sizing 改由步时预算支配——命题 4 的边界。

**配比与两个相消**。式 (5) 比式 (7)：

$$n_P : n_D = \frac{2 N_{\mathrm{act}} T_{\mathrm{in}}}{F_P\thinspace  \rho_P} : \frac{\kappa L\thinspace  T_{\mathrm{out}}\thinspace  \bar{T}}{\beta_D\thinspace  \rho_D} \qquad (8)$$

代入章脊得 $n_P : n_D$ = 1.55 : 1.09 ≈ 1.4 : 1，向上取整到 2 prefill + 2 decode（1:1）。教学口径的 chatbot 型 workload 给出接近 1:1 的配比——与 DeepInfer 生产经验值同量级，但这是口径巧合，不是普适常数。

**命题 4**（配比的双重无关性）。KV 流量口径下，配比 $n_P / n_D$ 与到达率 $\lambda$ 无关、与 TPOT 目标无关。**证明**（路线：代入 → 约分，两步）。 $\lambda$ ：两池需求同 $\propto \lambda$ （式 (5) 显含、式 (7) 经 $C$ 隐含），相比约去——负载水平只缩放两池的绝对规模，不改形状。TPOT：式 (7) 中并发 $C \propto \mathrm{TPOT}$ 而读取频率 $\propto 1/\mathrm{TPOT}$ ，两者精确相消（式 (7) 第二个等号已显式约掉）。∎ 边界反例：权重读项主导步时时，TPOT 通过步时预算约束并行度与 batch， $n_D$ 不再由 KV 流量账给出；KV 容量约束 $C \kappa L T_{\max} \le n_D M_{\mathrm{KV}}$ （ $M_{\mathrm{KV}}$ = 每 GPU 可用 KV 显存）引入 $T_{\max}$ ；若 $\rho$ 余量随 $\lambda$ 变（排队 SLO 收紧）， $\lambda$ 也间接回到配比。命题的「无关」是 KV 流量主导区的结构结论，不是全工况断言。

扫一遍 workload 与硬件（其余参数不变；MLA 两行是几何反事实：只把 $\kappa L$ 换成 V3 的 35136 B/token，隔离 $\kappa$ 因子）：

| 口径 | $n_P$ | $n_D$ | 读数 |
| --- | --- | --- | --- |
| chatbot 4096/512（章脊） | 1.55 | 1.09 | 约 1.4:1，取整后 1:1 |
| agent 4096/8192 | 1.55 | 32.8 | 1:21，长输出使 D 池贵 |
| MLA 几何，chatbot | 1.55 | 0.29 | 小 $\kappa$ 使 D 池便宜 |
| MLA 几何，agent | 1.55 | 8.8 | 两因子相乘 |
| H200（ $\beta$ = 4.8 TB/s），chatbot | 1.55 | 0.76 | 带宽换卡使 D 池便宜 |

三个读数。其一， $T_{\mathrm{out}}$ 翻 16 倍（512 → 8192）， $n_D$ 涨约 30 倍——因为 $T_{\mathrm{out}}$ 与 $\bar{T}$ 双因子同涨（式 (8) 右项），长输出 agent 是 decode 池的最贵客户。其二，MLA 把 $\kappa L$ 压到 GQA-8 的 0.268 倍， $n_D$ 同因子缩——KV 压缩直接折成 decode GPU 数（1.2 几何表的带宽侧读数）。其三， $F_P / \beta_D$ 比是硬件侧杠杆：换 H200（算力不变、带宽 1.43 倍） $n_D$ 缩到 0.70 倍。两池可以配不同的卡——Splitwise §IV-D 的异构集群优化正是此账；Ascend 950PR（prefill / 推荐，1.6 TB/s）与 950DT（decode / 训练，4 TB/s）的产品分化是厂商侧的呼应（1.3 datasheet recap，[Huawei Connect 2025 keynote / hiascend 产品页](../references.md)，[C] 官方披露）。

DeepInfer 的生产口径（官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，2026-11-23 核实，doc L91 逐字）：「1P + 1D is deployable. In practice the GPU ratio between P and D is roughly 1:1. Since the recommended decode configuration is EP64 or above, the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.」三个信息：~1:1 是**生产流量 mix 的经验配比**（doc「In practice」），不是理论常数——式 (8) 说明它随输入输出长度比与硬件 $F/\beta$ 比移动；64P+64D 是**最小高效单元**，量子化来自 decode 推荐 EP64 以上（Wide-EP 最小 decode 域，7.4 recap）；1P+1D 可部署（小规模）。需求侧配比（式 (8)）与部署侧配比之间隔着实例 / 通信域粒度的向上取整——「最优配比」在实践中是阶梯函数，不是光滑曲线。

Beyond-the-Buzz 把配比做成了求解器（[Mitra et al., 2025](../references.md)，[B]，NVIDIA；proprietary 高保真模拟器、无官方代码，按论文讲不推测代码）。所谓 rate matching，是指求平衡两相吞吐的 P:D 实例比；§3.2 的做法是对每对候选 prefill / decode 并行映射，用 **integer solver** 求 ctx:gen（context:generation = prefill:decode）实例比（Appendix B：TTL 约束下最小化总 GPU 数）。术语对照：NVIDIA 用 FTL（First Token Latency）/ TTL（Token-to-Token Latency），即 TTFT / TPOT 同义异名（论文 Table 1 口径）；其 §5.1 带宽公式中 FTL 取 prefill 计算完成时间口径（不含排队），两义区分见 8.3 §6.2。§4.3 的动态性：最优 ctx:gen 比随模型与目标延迟显著变化——ratio 3.5 利 relaxed latency、0.5 利 tight latency，固定比在两端都退化——所以必须动态 rate matching。这与命题 4 一致：配比是 workload 与 SLO 的函数，SLO 移动（式 (8) 经 $\rho$ 余量与步时预算边界）最优比就移动。§4.2 的适用域结论：**disaggregation 最有效 = prefill-heavy（ISL $\gg$ OSL；ISL / OSL = input / output sequence length，即本章 $T_{\mathrm{in}} / T_{\mathrm{out}}$ ）+ 大模型（>10B）**——式 (8) 的读法： $T_{\mathrm{in}} / T_{\mathrm{out}}$ 比大则 P 池相对贵，colocate 的干扰也最重（prefill 主导 iteration），分离的两笔收益同时最大。Dynamo 则把配比做成运行时变量：worker 运行时加入 / drain（排空在途请求后）移除 + discovery 自动发现（disagg-serving.md，[NVIDIA, Dynamo disagg-serving.md](../references.md)，教学参照），编排实现归 8.4、弹性归 8.5。

## 5. 传输 vs 重算：带宽阈值，不是临界长度

decode 侧不收 KV、改为自己重算 prompt，划不划算？路线：把传输与重算都写成 per-token 账 → 发现两者同 $\propto T$ → $T$ 相消 → 临界量是带宽阈值；然后把长度依赖的三个真实来源加回来。关键一步是相消——它决定这笔账的临界**不是**「临界上下文长度」。同形阈值已在 2.4 命题 1 / 式 (2) 于 offload 取回语境（HBM / DRAM / SSD 层间）推过；本节按 PD 跨节点传输语境重推导—— $\mathrm{BW}$ 取网络每 GPU 份额，重算侧还可能占用 decode 资源（下文第三条渠道），两笔增量都是 PD 特有的。

传输 $T$ token 的 KV 需时（ $\mathrm{BW}$ = 每 GPU 有效网络带宽，GB/s 为 $10^9$ 口径）：

$$t_{\mathrm{xfer}} = \frac{\kappa L T}{\mathrm{BW}} \qquad (9)$$

重算同一 prompt 需时（1.2 式 (1) 除以有效算力， $F_{\mathrm{eff}}$ = 重算 GPU 的有效算力 = 峰值 × MFU）：

$$t_{\mathrm{recomp}} = \frac{2 N_{\mathrm{act}} T + 2 L T^{2} d_{\mathrm{model}}}{F_{\mathrm{eff}}} \approx \frac{2 N_{\mathrm{act}} T}{F_{\mathrm{eff}}} \qquad (10)$$

**符号**： $\kappa$ = 每 token 每层 KV 字节数（1.2 式 (3)）；近似取前向线性项（attention 二次项下面单独加回）。per-token 读数（V3 口径，章脊已声明）：传输每 token $\kappa L$ = 35136 字节，重算每 token $2 N_{\mathrm{act}} = 7.4 \times 10^{10}$ FLOPs——两笔都线性于 $T$ ，系数就是阈值的全部内容。传输更便宜的条件（口径：「便宜」比墙钟时间，非货币成本，2.4 §2 同注；干扰与预算下面加回），三步代数（路线：列不等式 → 两边同除 $T$ → 解出 $\mathrm{BW}$ ）：

$$\frac{\kappa L T}{\mathrm{BW}} \lt \frac{2 N_{\mathrm{act}} T}{F_{\mathrm{eff}}} \quad\Longleftrightarrow\quad \frac{\kappa L}{\mathrm{BW}} \lt \frac{2 N_{\mathrm{act}}}{F_{\mathrm{eff}}} \quad\Longleftrightarrow\quad \mathrm{BW} > \frac{\kappa L\thinspace  F_{\mathrm{eff}}}{2 N_{\mathrm{act}}}$$

**命题 5**（带宽阈值）。线性项口径下，传输比重算便宜当且仅当 $\mathrm{BW} > \mathrm{BW}^{\ast}$ ，其中

$$\mathrm{BW}^{\ast} = \frac{\kappa L\thinspace  F_{\mathrm{eff}}}{2 N_{\mathrm{act}}} \qquad (11)$$

且该条件与 $T$ 无关。**证明**：传输字节 $\kappa L T$ 与重算 FLOPs $2 N_{\mathrm{act}} T$ 同 $\propto T$ ， $T > 0$ 时两边同除 $T$ 不等号方向不变，临界只剩带宽。∎

> **坑**：把临界写成「临界上下文长度」（C-S3-2）。触发条件：见传输字节随 $T$ 涨就断言「长上下文传输赢、短上下文重算赢」；现象：前向线性项下两边同 $\propto T$ 、 $T$ 相消，长度根本不在临界条件里——按「临界长度」做的决策在所有长度上同向错误；规避：临界是带宽阈值 $`\mathrm{BW}^{\ast}`$ ，长度依赖只从下面三条渠道进入。

代入 V3 口径（章脊已声明换口径：MLA + FP8 KV 是跨 DC 传输的生产形态）： $\kappa$ = 576 B（V3 MLA 576 元素 × FP8 每元素 1 字节；scale 开销约 1/32 略去，格式账归 2.3）， $L$ = 61， $\kappa L$ = 35136 B/token； $F_{\mathrm{eff}}$ = 1979 TFLOPS × MFU 0.4 = 791.6 TFLOPS（H100 FP8 dense，[NVIDIA datasheet](../references.md)，[C]）； $2 N_{\mathrm{act}} = 7.4 \times 10^{10}$ FLOPs/token（V3 激活 37B，[DeepSeek-AI, 2024](../references.md)，[A]）。得每 token 重算 93.5 μs， $`\mathrm{BW}^{\ast}`$ = 35136 字节 ÷ 93.5 μs ≈ 0.376 GB/s ≈ 3 Gbps。对照 DeepInfer 跨 DC 官方披露（doc L97，官方披露，2026-11-23 核实）：~400 Gbps / 8 GPU，每 GPU 份额 6.25 GB/s = 阈值的 16.6 倍——**传输远优于重算**，与 doc「压缩 KV 使跨 DC P/D 技术可行」的论断自洽（可靠性代价如光纤易被施工挖断，是另一笔账）。全量核对（ $T$ = 4096）：KV 143.9 MB，传输 23.0 ms vs 重算 382.9 ms，比值同为 16.6。

V4.1 方向更强：MLA 几何是 **512 元素**（448 + 64，X-1 跨波裁定；V3 的 576 元素不适用 V4.1——1.2 §3 坑注 recap，[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]），主 KV FP4 格式 288 B/token/层（2.3 §4 官方口径）， $L$ = 40， $\kappa L$ = 11520 B/token = V3 的 0.328 倍—— $`\mathrm{BW}^{\ast}`$ 同比例下降，传输优势更大。bf16 口径则 1024 B × 40 = 40960 B/token（1.2 几何表 recap）。

长度依赖的三个真实来源。其一，**attention 二次项**：把式 (10) 的完整分子代回不等式，阈值变成

$$\mathrm{BW}^{\ast}(T) = \frac{\kappa L\thinspace  F_{\mathrm{eff}}}{2 N_{\mathrm{act}} + 2 L T d_{\mathrm{model}}} \qquad (12)$$

分母随 $T$ 增、阈值单调降——上下文越长重算越贵，传输越容易赢。代入 V3 $T$ = 32768：分母 $1.027 \times 10^{11}$ ， $`\mathrm{BW}^{\ast}`$ = 0.271 GB/s，降到线性项值的 72%。其二，**TTFT 预算**：per-token 比较与 $T$ 无关，但绝对传输时间 $\propto T$ ，必须装进 TTFT SLO 的传输份额（2.4 式 (3) 的 $`\mathrm{BW}_{\mathrm{req}}`$ 账 recap）——长度从预算侧进入。其三，**重算的干扰**：重算若发生在 decode 池，等于把 compute-bound 工作重新塞回 decode 池——正是 §1 消除的干扰，TPOT 尖刺回归。所以「重算便宜」只在专注重算资源的隔离口径下成立；生产中重算是传输失败的 fallback，不是常态路径（口径注）。小 $T$ 另有一个边界：每次传输的固定延迟（启动 / 同步）与 $T$ 无关，KV 小时它主导传输时间，per-token 账须加回（2.4 §2 边界注）——这也是 Splitwise「小 prompt 用 serialized」选择的另一面（§3）：KV 小时传输本身够快，layer-wise 的逐层同步开销反而不值。

layer-wise 重叠进一步压低传输的边际成本。**命题 6**（全重叠阈值 = $`\mathrm{BW}^{\ast}`$ ）：线性项口径下（同命题 5），逐层流式传输（每层 KV 算完即传、与后续层计算重叠）能完全藏进 prefill 计算，当且仅当 $\mathrm{BW} \ge \mathrm{BW}^{\ast}$ ；attention 二次项增加可藏的计算量，阈值换式 (12) 的 $`\mathrm{BW}^{\ast}`$ (T)、条件更宽。证明留作练习 4(a)。直觉预告：每层 KV $\kappa T$ 字节 vs 每层计算时间 $2 N_{\mathrm{act}} T / (L F_{\mathrm{eff}})$ ，相比又是同一笔 per-token 账。代入 V3 $T$ = 4096：每层 KV 2.36 MB，6.25 GB/s 下 0.38 ms，远小于每层计算 6.28 ms——完全藏住。暴露时间的一般形：

$$t_{\mathrm{exposed}} \approx \max\left( 0,\ t_{\mathrm{xfer}} - t_{\mathrm{hideable}} \right) \qquad (13)$$

**符号**： $t_{\mathrm{hideable}}$ = 可重叠的 prefill 计算时间。serialized 传输 $t_{\mathrm{hideable}} = 0$ ，全量进 TTFT / TBT（Splitwise §IV-C：serialized 直接顶 max TBT）；layer-wise 把暴露项压到末层 KV / $\mathrm{BW}$ + 逐层同步开销。Splitwise 实测（§VI-A）：非重叠残余近似常数，A100（200 Gbps）约 8 ms、H100（400 Gbps）约 5 ms。Beyond-the-Buzz §5.1 的带宽需求公式即假设逐层即传；egress / ingress（发送侧 / 接收侧）带宽的全账（含 ISL / OSL 缩放方向）归 8.3。

> **提点**： $`\mathrm{BW}^{\ast}`$ 有双重身份——传输 vs 重算的盈亏平衡点，与 layer-wise 全重叠条件，是同一个数。因为两者都在比「KV 字节的产出速率」与「token 的计算速率」。这解释了为什么传输 vs 重算的决策在生产中如此稳健：每 GPU 网络份额只要超过 3 Gbps（V3 口径）传输就赢，而现代互联通常高出阈值一到两个数量级。

## 6. 何时分离：收益账与替代方案

分离净收益 = 干扰消除 − KV 传输成本 − 池碎片。何时为正？本节把三笔摆齐，再对照 chunked prefill 替代方案，给决策表。

收益侧两笔。其一，命题 1 的 min 约束解除：每相逼近自己 SLO 的容量极限， $g_P$ 、 $g_D$ 各自抬高（§2 的 5.6 / 10 vs colocate 1.6）。其二，decode 大 batch 不再与 prefill 冲突：DistServe §3.2 的读法——colocate 时加大 decode batch 会挤占 prefill、恶化 TTFT，分离后多个 P 实例喂一个 D 实例，decode 池专职攒 batch 提利用率。

成本侧三笔。其一，传输带宽 + TTFT 份额（§5：现代互联下通常小，跨 DC / 低带宽集群必须按式 (11)(12) 与 2.4 预算账核）。其二，两池碎片：每池各留各的余量，低并发时 aggregated（同卡混部）反而省传输与池碎片开销（Dynamo disagg-serving.md「Capacity effects」，[NVIDIA, Dynamo disagg-serving.md](../references.md)，教学参照）。其三，P 池的 KV 占着 HBM 却不用于 decode——decode KV 容量是瓶颈时，加 prefill 副本反而降低总服务容量，除非延迟收益补偿被挪走的 GPU（同源）。持续负载下两相争抢时，分池的延迟隔离收益才压过这三笔。

替代方案：chunked prefill（Sarathi-Serve，[Agrawal et al., 2024](../references.md)，[A]；机制与账 recap 4.3，不重推）。stall-free 给 TBT 上界（4.3 命题 2），但三笔残留：TTFT 仍付（chunk 串行 + piggyback 竞争），前缀 KV 重读 $\propto T^{2}/c_{\mathrm{chunk}}$ （4.3 式 (4)），并行度决策仍耦合。DistServe §2.3 的判定：chunking 是「trades TTFT for TPOT」，不能消除干扰；chunk 太小 GPU 饱和不了、太大 piggyback 槽位没了，两头堵。Beyond-the-Buzz §4.1 补一个机制敏感点：MLA 每个 prefill chunk 有冗余的 down/up projection 计算（DeepSeek-R1 例，可缓存 up-projected KV 缓解），chunking 收益对 attention 机制（MLA vs GQA）敏感；其结论是 colocate + chunking 最利 relaxed latency + generation-heavy workload——与分离的适用域（prefill-heavy + 大模型，§4）正好互补。

决策表（每行回指依据）：

| workload / 约束 | 建议 | 依据 |
| --- | --- | --- |
| prefill-heavy（ISL $\gg$ OSL）+ 大模型（>10B）+ 双 SLO 紧 | PD 分离 | 式 (8)；Beyond-the-Buzz §4.2 |
| generation-heavy + relaxed latency | colocate + chunked prefill | §6；Beyond-the-Buzz §4.1；4.3 |
| 低并发 / 小规模 | colocate（aggregated） | §6 碎片账；Dynamo doc |
| 跨节点带宽受限 | 分离 + 同节点 placement（Algorithm 2） | §2；DistServe §4.2 |
| decode KV 容量瓶颈 | 先扩 D 池 / KV 分层，加 P 副本谨慎 | §6；2.4 |

生产形态指针（纯文字）：DeepInfer 的 VE/P/D 三角色分离 + 专用存储集群 recap 7.4；autoscaling 按 non-cached token 数定 P 池机数、按 running requests 调 D 池（8.5）；引擎实现对照（vLLM KV connector / SGLang disaggregation / Dynamo / TRT-LLM）归 8.4，SGLang 实现已在 6.3 走读。本章交付原理账，不走读代码。

## 总结

| 论断 | 依据 | 回指 |
| --- | --- | --- |
| colocated goodput = 两个单 SLO goodput 的 min；决策耦合导致 over-provision | 式 (1)，命题 1 | §1 |
| PD 分离 = 物理隔离消干扰 + 两相独立并行度 / batch；实例 = 一份完整权重的资源单元 | §1 定义 | §1、§2 |
| 稳态系统率 = 两池容量 min；连续最优平衡两池，per-GPU goodput 调和形 | 式 (2)(3)，命题 2 | §2 |
| 5.6 / 10 rps/GPU → 连续最优 3.59；整数 replication 2P+1D → 3.33（取整损失约 7%），colocate 1.6 的 2.1 倍 | 式 (3)，论文 §1 | §2、练习 1 |
| 相独立优化合法 ⟺ 相解耦（高带宽 placement）；低节点亲和 placement 约束耦合可行集，须联合枚举 | 命题 3，Algorithm 1/2 | §2 |
| attainment 无闭式 → trace 拟合 simulator + 二分；M/D/1 给并行度偏好：低率 intra、高率 inter | 式 (4)，论文 §4.1 | §2 |
| $n_P$ = 算力需求 /（有效算力 × 余量）； $n_D$ = KV 带宽需求 /（HBM × 余量），并发由 Little 定律 | 式 (5)(6)(7) | §4、练习 2 |
| 配比与 $\lambda$ 、TPOT 无关（KV 流量口径），只依赖 workload 形状 + 硬件；权重项 / 容量约束是边界 | 式 (8)，命题 4 | §4 |
| DeepInfer ~1:1 = 生产流量经验配比；64P+64D = EP64 最小 decode 域量子化；1P+1D 可部署 | doc L91 官方披露 | §4 |
| 传输 vs 重算临界 = 带宽阈值 $\mathrm{BW}^{\ast} = \kappa L F_{\mathrm{eff}} / (2 N_{\mathrm{act}})$ ，非临界长度（ $T$ 相消） | 式 (11)，命题 5 | §5、练习 3 |
| V3 FP8 口径 $\mathrm{BW}^{\ast}$ ≈ 0.376 GB/s ≈ 3 Gbps；DeepInfer 跨 DC 每 GPU 份额 6.25 GB/s = 16.6 倍阈值 | §5 数值 | §5 |
| layer-wise 全重叠条件恰为 $\mathrm{BW}^{\ast}$ （线性项口径；二次项下换式 (12)）；暴露项压到末层 KV / BW + 同步 | 命题 6，式 (13) | §5、练习 4 |
| 分离净收益 = 干扰消除 − 传输成本 − 池碎片；prefill-heavy + 大模型最利，generation-heavy + relaxed latency 利 chunking | §6 三笔账 | §6 |

## 练习

1.（计算 · [10]）调和容量与整数 replication：prefill 实例 2 GPU、每实例 goodput 8 rps；decode 实例 1 GPU、每实例 goodput 12 rps。(a) 求 $g_P$ 、 $g_D$ （per-GPU）；(b) 用式 (3) 求连续最优 $\mathcal{G}^{\ast}$ 与平衡配比 $n_P : n_D$ ；(c) 验证 3 prefill 实例 + 2 decode 实例的部署达到 $\mathcal{G}^{\ast}$ 。校验：(b) 与式 (3) 调和形一致；(c) $\min(3 \times 8,\ 2 \times 12) = 24$ rps、8 GPU，per-GPU 恰为 (b) 的值。

2.（计算 · [20]）xPyD 的 agent 口径：章脊不变（8B bf16 @ H100， $\lambda$ = 10、 $T_{\mathrm{in}}$ = 4096、TPOT SLO 30 ms、 $\rho$ = 0.8），把 $T_{\mathrm{out}}$ 换成 8192。(a) 求并发 $C$ 与平均上下文 $\bar{T}$ ；(b) 求 KV 带宽需求与 $n_D$ ；(c) 与 chatbot 口径（ $T_{\mathrm{out}}$ = 512）的配比对照，说明变化由式 (8) 的哪几个因子驱动；(d) 先预测换 MLA 几何（ $\kappa L$ = 35136 B/token）后 $n_D$ 的变化方向与倍数，再代入。校验：(b) 与 §4 表的 32.8 一致；(d) 的倍数应等于 $\kappa L$ 之比 0.268（式 (8) 对 $\kappa L$ 线性）。

3.（推导 · [30]）带宽阈值全推导（命题 5 补全 + 二次项扩展）：(a) 从式 (9)(10) 出发完整写出 $t_{\mathrm{xfer}} \lt t_{\mathrm{recomp}} \Longleftrightarrow \mathrm{BW} > \mathrm{BW}^{\ast}$ 的三步代数，注明每步的合法性条件；(b) 把 attention 二次项加回，推出式 (12) 的 $`\mathrm{BW}^{\ast}`$ (T) 并证其对 $T$ 单调降；(c) 代入 V3 口径（ $\kappa L$ = 35136 B/token、 $F_{\mathrm{eff}}$ = 791.6 TFLOPS、 $N_{\mathrm{act}}$ = 37B、 $L$ = 61、 $d_{\mathrm{model}}$ = 7168）算 $T$ = 32768 的阈值，与线性项值 0.376 GB/s 对照。校验：(c) ≈ 0.271 GB/s、约为线性项的 72%；(b) 的单调性由分母随 $T$ 严格增直接读出。

4.（推导 · [40]）layer-wise 重叠阈值（命题 6 的证明 + 暴露时间账）：(a) 证明逐层全重叠条件恰为式 (11) 的 $\mathrm{BW}^{\ast}$ ——写出每层 KV 字节与每层计算时间（参数项按层均摊），相比并说明与命题 5 同构；(b) 推导 layer-wise 下暴露传输时间 ≈ 末层 KV / $\mathrm{BW}$ （+ 逐层同步开销），代入 V3 $T$ = 4096、每 GPU 份额 6.25 GB/s 算暴露值，并与 serialized 全量 23.0 ms 对照；(c) 读 Beyond-the-Buzz §5.1 的 egress 公式（论文 Eq. 1），把论文的 d_head × N_kv_heads（MHA/GQA 几何）翻译成本维度 $\kappa$ 记号，解释为什么 egress 需求随 ISL 增大反而下降（提示：FTL 超线性）、ingress 为什么与 ISL 无关而与 OSL 成反比。校验：(a) 与式 (11) 逐字一致；(b) ≈ 0.38 ms，与 §5 数值一致；(c) 与论文 §5.1 公式结构一致（MLA 以 $\kappa$ 替换 MHA/GQA 几何，归一只计实际分片 KV 的 GPU）。

## 参考

- [Zhong et al., 2024](../references.md)：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，arXiv:2401.09670，[A]（key: distserve）——goodput 定义与 13B / A100 数值（§1）、M/D/1 与并行度偏好（§3.1）、KV 传输 90 Gbps 账（§3.3）、Algorithm 1/2 与 simulator 二分（§4）、pull 与 replan（§4.3）；官方实现 LLMServe/DistServe（SwiftTransformer backend）
- [Patel et al., 2024](../references.md)：Splitwise: Efficient generative LLM inference using phase splitting，ISCA 2024，arXiv:2311.18677，[B]（key: splitwise）——phase splitting 与异构硬件（Table I：H100 3.43× 算力 / 1.75× 功耗）、Azure trace characterization、三池调度（§IV-A）、layer-wise / serialized 传输（§IV-C、§VI-A：残余 8/5 ms、16.5% vs 64%）、1.4× / 2.35× 集群账；仅作者模拟器 mutinifni/splitwise-sim + vLLM PR，代码不全
- [Qin et al., 2024](../references.md)：Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving，arXiv:2407.00079，[A]（key: mooncake）——KVCache-centric 架构与 early rejection（recap 2.4 §5）、525% / 75% 摘要口径
- [Agrawal et al., 2024](../references.md)：Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve，arXiv:2403.02310，[A]（key: sarathi-serve）——chunked prefill 替代方案对照（机制与账 recap 4.3）
- [Mitra et al., 2025](../references.md)：Beyond the Buzz: A Pragmatic Take on Inference Disaggregation，arXiv:2506.05508，[B]（key: beyond-buzz）——rate matching integer solver（§3.2 / Appendix B）、动态配比（§4.3：3.5 / 0.5）、prefill-heavy + 大模型最有效（§4.2）、MLA chunking 冗余计算（§4.1）、FTL / TTL 术语；proprietary 模拟器、无官方代码
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23）——~1:1 经验配比与 64P+64D 最小高效单元（L91）、跨 DC ~400 Gbps / 8 GPU 与压缩 KV 可行性（L97）
- [Microsoft Azure, AzureLLMInferenceDataset2023](../references.md)：Azure/AzurePublicDataset，教学参照（key: azure-llm-trace）——Splitwise Azure trace 释放子集（schema 与归属 recap 1.1 §4）
- [NVIDIA datasheet](../references.md)：[C]（key: nvidia-datasheet，官方规格披露、非闭源）——H100 BF16 989.5 / FP8 1979 TFLOPS dense、HBM 3.35 TB/s；H200 4.8 TB/s
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——V3 几何：61 层、37B 激活、MLA 576 元素（§5 口径）
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：[W]（key: v41-flash-config）——V4.1 MLA 512 元素（448 + 64）、40 层（§5 方向注；hf-mirror 传输核实，2026-11-23）
- [NVIDIA, Dynamo disagg-serving.md](../references.md)：教学参照（key: dynamo）——Capacity effects（低并发 aggregated / 持续负载隔离 / decode 容量瓶颈）、runtime-reconfigurable xPyD
