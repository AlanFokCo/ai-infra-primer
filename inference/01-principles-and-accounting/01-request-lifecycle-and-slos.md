# 1.1 请求生命周期与 SLO 指标（Request Lifecycle and SLO Metrics）

> 一条请求从 HTTP 到达到最后一个 token 流出，墙钟时间花在哪几段、每段该用什么指标度量、为什么「吞吐最高的配置」往往不是「能服务最多用户的配置」？本章把请求全链路形式化，定义 TTFT / TPOT / TBT / ITL 与 goodput 的精确口径，并给出 benchmark 的方法学与陷阱 | 前置：models 2.4 §3（prefill / decode 两阶段伪码）、models 1.2 §10（算术强度入口，本章只用结论）；排队论直觉在 1.3 建立，本章只用到 Little 定律 | 锚点：vLLM bench（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]；vllm-project/vllm repo）、DistServe goodput 定义（[Zhong et al., 2024](../references.md)，arXiv:2401.09670，[A]）、Orca（[Yu et al., 2022](../references.md)，USENIX OSDI'22，无 arXiv，[B]）、Splitwise Azure trace（[Patel et al., 2024](../references.md)，arXiv:2311.18677，[B]）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）

## 1. 请求全链路：一条请求的墙钟时间花在哪

一条 prompt 1024 token、输出 512 token 的请求，端到端十几秒的墙钟时间里，GPU 真正在算的时间占多少？答案取决于链路里每一段的成本结构。先把链路画全（口径：单引擎 colocated 部署，PD 分离形态见 8.2）：

```
客户端 HTTP 请求（prompt 文本 + 采样参数）
  │
  ▼
API 协议层（OpenAI-compatible：/v1/chat/completions 等）
  │  解析、鉴权、参数校验
  ▼
tokenize（文本 → token ids；长 prompt 可达 ms 量级，1.4 流式工程回链）
  │
  ▼
排队（waiting queue；调度器按策略选入 running batch，4.1 / 4.2）
  │
  ▼
prefill（一次并行前向打全部 prompt token，产出首 token + 全量 KV cache）
  │  ← TTFT 的主要成分
  ▼
decode 循环（每 iteration 全 batch 各出 1 token；迭代级调度，Orca）
  │  ← 每步间隔 = TBT 的主要成分
  ▼
detokenize（token ids → 文本片段；增量式，UTF-8 跨 chunk 处理归 1.4）
  │
  ▼
流式返回（SSE chunk 逐 token 推送）→ 客户端
```

所谓 prefill，是指对 prompt 的全部 $T_p$ 个 token 做一次并行前向、同时初始化 KV cache 的阶段；所谓 decode，是指此后每步只输入 1 个新 token、读取全量 cache 产出下一个 token 的串行循环（两阶段的数学与伪码在 models 2.4 §3，此处只取链路位置）。两阶段的成本结构截然不同：prefill 是大 matmul、compute-bound，decode 每步读全量权重与 cache、memory-bound（models 2.4 §6 的结论，定量账在 1.2）。这个非对称是后续全部调度设计的根源——prefill 独占 GPU 时会把 decode 的步间隔拖出尖刺（chunked prefill 的动机，4.3），两类负载混跑时互相干扰的形态不同（PD 分离的动机，8.2）。

链路上还有三段常被忽略的成本。其一，排队：请求到达率超过引擎瞬时服务能力时，请求在 waiting queue 里等待的时间可以远超计算时间本身——高负载下 TTFT 的大头是排队而非 prefill（1.3 的排队论把这件事定量化）。其二，tokenize / detokenize：CPU 侧操作，长 prompt 下不可忽略，DeepInfer 把编解码交给 DeepSeek Recipe 组件以压低长输入延迟并防 special-token 注入（官方 doc 口径，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，2026-11-23 核实；实现走读归 7.4）。其三，调度开销：每个 iteration 的组批、KV 块分配、采样后处理都在 CPU 上发生，overlap scheduler 把它藏进 GPU 计算的时间里（6.2）。

decode 循环的调度粒度是 iteration 而非 request：调度器每个 iteration 重新组 batch，请求随时加入、完成即离开——这是 Orca 的 iteration-level scheduling（原文：「schedules execution at the granularity of iteration (instead of request)」，[Yu et al., 2022](../references.md)，[B]，摘要）。机制展开与吞吐账归 4.1，本章只需要这个事实：batch 的组成每步都在变，所以「batch size」不是一个请求级常量，而是一个逐 iteration 的运行时量——§5 的 concurrency ≠ batch 陷阱由此而来。

本章口径：数值实例统一取「prompt 1024 token、输出 512 token、TTFT 0.5 s、TPOT 30 ms」这组教学数值（非实测，只为把定义算到底）；goodput 实例取 DistServe 论文的 13B 模型 A100 口径（§3 随文声明）；硬件与模型的性能账不在本章，归 1.2 / 1.3。本章 $\lambda$ = 到达率（requests/s），非 GAE 参数； $B$ = 一个 iteration 内 decode 的序列数； $T_{\mathrm{out}}$ = 输出 token 数。

## 2. 延迟指标：TTFT、TPOT / TBT、ITL 与端到端

「延迟」这一个词背后藏着几个不同的量？至少四个：TTFT、TPOT / TBT、ITL、端到端——统计口径各不相同，混用是 SLO 纠纷的第一来源。首 token 延迟（TTFT，time to first token）定义为请求到达到第一个输出 token 可见的时间： $\mathrm{TTFT} = t_1 - t_{\mathrm{arrival}}$。它覆盖 §1 链路里 tokenize、排队、prefill 三段（流式形态下 detokenize 首片段的时间也计入）；非流式请求没有「首 token 可见」时刻，TTFT 退化为端到端延迟，两种形态的指标不可混比。

每输出 token 延迟（TPOT，time per output token）与 token 间延迟（TBT，time between tokens）是同一物理量的两种统计口径。逐 token 看，第 $i$ 个输出 token 的 TBT 是相邻两个 token 的时间差 $`\mathrm{TBT}_i = t_i - t_{i-1}`$（ $i \ge 2$）；逐请求看，TPOT 是这些间隔的均值：

$$
\mathrm{TPOT} = \frac{t_{\mathrm{last}} - t_1}{T_{\mathrm{out}} - 1} = \frac{1}{T_{\mathrm{out}} - 1} \sum_{i=2}^{T_{\mathrm{out}}} \mathrm{TBT}_i \qquad (1)
$$

**符号**： $t_i$：第 $i$ 个输出 token 的可见时刻； $t_{\mathrm{last}}$：末 token 时刻； $T_{\mathrm{out}}$：输出 token 数（首 token 由 prefill 产出，故分母是 $T_{\mathrm{out}} - 1$）。最小实例： $t_{\mathrm{last}} - t_1 = 15.33$ s、 $T_{\mathrm{out}}$ = 512，TPOT = 15.33/511 = 30 ms。两个词的用法分野是文献事实：DistServe 用 TPOT（原文「TTFT and TPOT requirements」，[Zhong et al., 2024](../references.md)，[A]），DeepInfer 用 TBT（原文「so TTFT and TBT can be controlled and optimized independently」，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，doc L31，2026-11-23 核实）。本书两者并存时以 TPOT 指请求级均值、TBT 指逐间隔样本。ITL（inter-token latency）与 TBT 同义，vLLM / SGLang 的 benchmark 输出用 ITL 字段名（§5）。统计口径还有一层分野：对「全部请求的全部间隔」取百分位，与「先逐请求取均值、再对请求取百分位」是两个不同的分布——前者抓单步尖刺，后者抓请求级体验，报告必须写明是哪一种（本书 SLO 实例默认前者）。

端到端延迟是各段之和。把式 (1) 代入即得分解式：

$$
\mathrm{E2E} = \mathrm{TTFT} + \sum_{i=2}^{T_{\mathrm{out}}} \mathrm{TBT}_i \approx \mathrm{TTFT} + (T_{\mathrm{out}} - 1) \cdot \mathrm{TPOT} \qquad (2)
$$

代入本章口径：TTFT = 0.5 s、TPOT = 30 ms、 $T_{\mathrm{out}}$ = 512，得 E2E = 0.5 + 511 × 0.03 = 15.83 s。其中 TTFT 只贡献 3%，decode 循环贡献 97%——长输出请求的延迟优化主战场在 decode 步时，而不在 prefill。反过来，短输出请求（ $T_{\mathrm{out}}$ = 8）的 E2E = 0.5 + 7 × 0.03 = 0.71 s，TTFT 占 70%——agent 类负载里「长 prompt、短输出」的调用正是这个形态，TTFT 是它们的 SLO 主项（workload 形态见 §4）。

**均值与百分位**。式 (2) 是逐请求的恒等式，但服务级指标是对请求分布的统计。P50 / P90 / P99 分别是延迟分布的 50% / 90% / 99% 分位数：P99 = 5 s 意味着 99% 的请求快于 5 s。均值陷阱在于：长尾请求把均值拉向尾部，而均值对尾部形状不敏感。一个 100 请求的最小实例：90 条 TTFT = 0.2 s、10 条 TTFT = 5 s（长 prompt 或排队尖刺），均值 = 0.9 × 0.2 + 0.1 × 5 = 0.68 s，P50 = P90 = 0.2 s，P99 = 5 s。均值 0.68 s 是 P90 的 3.4 倍，却比 P99 低一个量级——只看均值会同时误判「典型体验」（实际 90% 的请求是 0.2 s）与「尾部风险」（实际 1% 的请求要等 5 s）。SLO 必须写在百分位上（如「P99 TTFT ≤ 2 s」），写在均值上的 SLO 可以被长尾任意违反而均值不动。

> **坑**：TPOT 均值掩盖 TBT 尖刺。触发条件：decode 循环里混入长 prefill（未开 chunked prefill）或发生抢占重算；现象：请求级 TPOT 只涨几个百分点（尖刺被 511 个间隔平均掉），但个别 TBT 从 30 ms 跳到数百 ms，流式输出肉眼可见地卡顿。规避：SLO 与监控同时看 TBT 的高百分位（P99 ITL），不只看 TPOT 均值；stall 的定量账归 4.3。

## 3. 吞吐与 goodput：为什么最大吞吐不是目标

吞吐（throughput）有两个单位口径：输出 tokens/s（decode 侧产能）与 requests/s（服务侧产能），两者经平均输出长度换算，混用是 benchmark 报告的第一大歧义源（§5）。稳态 decode 的 token 吞吐有一个精确的守恒式：

**命题 1**（稳态 decode 吞吐）。设每个 iteration 的步时为 $\tau_{\mathrm{step}}$、batch 内恰有 $B$ 条序列各产出 1 个 token（无投机解码、无抢占），则输出 token 吞吐 $\Theta = B / \tau_{\mathrm{step}}$，且每条序列的 TBT 恰为 $\tau_{\mathrm{step}}$。

**证明**（计数）：一个 iteration 产出的 token 数 = batch 内序列数 $B$（每序列每步恰 1 个，这是 decode 的定义）；iteration 的墙钟时长 = $\tau_{\mathrm{step}}$。吞吐 = 产出 ÷ 时间 = $B / \tau_{\mathrm{step}}$。对单条序列，相邻两个 token 之间恰好隔一个 iteration，故 TBT = $\tau_{\mathrm{step}}$。∎

代入： $B$ = 32、 $\tau_{\mathrm{step}}$ = 30 ms，得 $\Theta$ = 32 / 0.03 ≈ 1067 tokens/s。这个式子的系统含义：提高吞吐只有两条路——增大 $B$ 或压短 $\tau_{\mathrm{step}}$。而 1.2 将证明 $\tau_{\mathrm{step}}$ 本身随 $B$ 增长（每步要读 $B$ 份 KV cache），所以「增大 batch」的收益被带宽账封顶；同时 $B$ 增大使 $\tau_{\mathrm{step}}$ 变长，由命题 1 的第二句，每条请求的 TBT 同步变长——吞吐与延迟在同一个变量 $B$ 上反向。这就是吞吐-延迟权衡的算术根源（三方权衡曲线的推导归 1.3）。

goodput 把 SLO 约束写进吞吐的定义。DistServe 的原文定义（[Zhong et al., 2024](../references.md)，[A]，§1，逐字）：

> "maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query."

两个定义要件缺一不可：**per-GPU 归一**（除以配置的 GPU 数，使不同并行配置可比）与 **SLO attainment 目标**（达标率约束，不是裸吞吐）。写成定义式：

$$
\mathcal{G} = \frac{1}{n_{\mathrm{GPU}}} \max \lbrace \lambda : \mathrm{Attain}(\lambda) \ge \theta \rbrace \qquad (3)
$$

**符号**： $\mathcal{G}$：per-GPU goodput（requests/s/GPU）； $n_{\mathrm{GPU}}$：配置的 GPU 总数； $\lambda$：请求到达率（本章口径：到达率，非 GAE 参数）； $\mathrm{Attain}(\lambda)$：到达率 $\lambda$ 下同时满足 TTFT 与 TPOT 两项 SLO 的请求占比（DistServe 口径）； $\theta$：attainment 目标（如 90% 或 99%）。所谓 goodput，是指在给定 SLO 达标率约束下每 GPU 可服务的最大请求率——「达标请求的裸吞吐」不是 goodput：它丢掉了「最大可达率」的优化语义，也丢掉了 per-GPU 归一。8.2 的 PD 配比推导与 9.1 的容量规划显式沿用式 (3)。

全数字实例（DistServe 论文 §1 口径：13B 模型、A100、 $\theta$ = 90%，[Zhong et al., 2024](../references.md)，[A]）：colocate 单卡 goodput ≈ 1.6 rps；PD 分离后 prefill 实例 5.6 rps、decode 实例 10 rps，2P+1D 配置的整体受限于 decode 侧 $\min(2 \times 5.6,\ 10) = 10$ rps，摊到 3 张 GPU 得 $\mathcal{G}$ = 10/3 ≈ 3.3 rps/GPU，是 colocate 的 2.1 倍。注意这个 2.1× 不是「吞吐提高了 2.1 倍」——它是「同样 90% 达标率约束下，每 GPU 能多服务 2.1 倍请求」，换算成每请求成本即下降为 1/2.1（定义引文的最后一句）。瓶颈取 min 的配比逻辑是 8.2 xPyD 推导的入口。

**吞吐-延迟-goodput 三角**。三个量的关系可以一句话判定：吞吐是产能，延迟是体验，goodput 是「体验约束下的有效产能」。增大 $B$ 提吞吐（命题 1），但 $\tau_{\mathrm{step}}$ 随之变长、TBT 变差，固定 $\lambda$ 下 attainment 下降；当 attainment 跌破 $\theta$，式 (3) 的可行集收缩，goodput 反而下降。所以吞吐最大化的操作点（离线 benchmark 的测法）与 goodput 最大化的操作点（在线 serving 的目标）一般不重合。在线 / 离线 serving 的分界正在于此：在线 serving 受 SLO 约束、优化目标是式 (3)；离线批处理（评测跑分、数据合成）无延迟约束、优化目标是裸吞吐。DeepInfer 对两者差距的生产披露：「we reach roughly 80% of a perfectly balanced offline benchmark」（官方 doc L155，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）——在线生产吞吐约为完全均衡离线基准的 80%，差距来自负载不均、排队与 SLO 约束下的操作点保守化。拿离线 benchmark 数字直接承诺在线产能，会系统性高估约两成。

## 4. 到达过程与 workload 形态

请求什么时候来、带多长的 prompt、要生成多少 token？这三件事决定排队形态与 prefill/decode 负载比。到达过程的标准起点是 Poisson 假设：单位时间到达数服从参数 $\lambda$ 的 Poisson 分布、间隔独立指数分布。真实流量偏离 Poisson 的形态是 burst（突发）：agent 系统里一个用户动作触发一串并发工具调用，到达在秒级聚簇。排队论把到达过程偏离 Poisson 的程度写进等待时间公式：间隔的变异系数平方作为乘性因子进入 Kingman 近似（M/M/1 与 Kingman 归 1.3），burst traffic 按该因子放大等待——所以 benchmark 用哪种到达过程必须声明——固定 request rate 的开环压测与固定 concurrency 的闭环压测测的不是同一个量（§5）。

长度分布的两个公开事实源，归属必须分开标注：

**Azure LLM inference trace**。Splitwise（ISCA 2024）释放了两个 Azure LLM inference 服务（coding 与 conversation）2023-11-11 的生产 trace（论文原文：「We use production traces taken from two Azure LLM inference services on November 11th 2023」，[Patel et al., 2024](../references.md)，[B]，§3）。数据集在 Azure/AzurePublicDataset repo，描述文件是 AzureLLMInferenceDataset2023.md（是 .md 文件不是目录），CSV 为 data/AzureLLMInferenceTrace_code.csv 与 data/AzureLLMInferenceTrace_conv.csv，schema 三字段：TIMESTAMP / ContextTokens / GeneratedTokens（[Microsoft Azure, AzureLLMInferenceDataset2023](../references.md)，教学参照，2026-11-23 核实）。CC-BY 许可；因 GDPR 只给尺寸、无 prompt 内容。论文的 characterization 用 20 分钟子集。这份 trace 的价值：到达时间 + 输入输出长度三元组足以重建到达过程与长度分布、驱动排队仿真，是 1.3 排队论与 8.2 PD 配比的标准负载输入。

**ShareGPT**。社区共享的 ChatGPT 对话数据集，常用 benchmark 版本是 anon8231489123/ShareGPT_Vicuna_unfiltered 中的 ShareGPT_V3_unfiltered_cleaned_split.json（hf-mirror 传输核实存在，huggingface.co 本机不可达，2026-11-23；[社区, ShareGPT_Vicuna_unfiltered](../references.md)，教学参照）。vLLM bench 以 `--dataset-name sharegpt --dataset-path <本地路径或 HF ID>` 使用（datasets.py 的 ShareGPTDataset，无硬编码默认 URL），SGLang bench serving 亦支持 sharegpt 数据集（[vLLM](../references.md)、[SGLang](../references.md)，repo 路径 2026-11-23 核实）。它提供真实对话文本（可测 tokenize 与 prefix 命中），但**非 Azure 生产 trace、非 Splitwise 释放**——到达时间字段缺失，不能直接驱动到达过程仿真。两份数据的用途互补：Azure trace 给「何时来、多长」，ShareGPT 给「说什么」。

workload 形态对系统设计的传导路径有两条。其一，prompt/输出长度比决定 prefill/decode 负载比：chatbot 多轮对话是「中 prompt、中输出、高前缀复用」，agent 是「长 prompt、短输出、工具调用 burst」——前者 decode 主导、后者 prefill 与 TTFT 主导，PD 配比因此不同（8.2）。其二，前缀复用率决定 prefix cache 的收益：DeepInfer 生产对 agent 流量取 72 h TTL、chatbot 取 7 d TTL 即可捕获几乎全部可复用条目（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实；命中率账与 TTL 权衡归 2.2）。

## 5. benchmark 方法学与陷阱

测一个 serving 引擎，工具链当前有三个入口（版本锚定 2026-11-23 main/dev）：

| 工具 | 入口 | 负载形态 | 指标 | 状态 |
| --- | --- | --- | --- | --- |
| vLLM bench | `vllm bench serve / latency / throughput`（vllm/benchmarks/{serve,latency,throughput}.py + CLI vllm/entrypoints/cli/benchmark/） | serve：request rate 或 concurrency 打 OpenAI-compatible 端点；latency / throughput：离线批 | TTFT / TPOT / ITL / 吞吐 | 活跃（[vLLM](../references.md)，[A]，路径核实 2026-11-23） |
| SGLang bench | `python -m sglang.benchmark.serving`（python/sglang/benchmark/serving.py） | 多 backend（sglang / vllm / lmdeploy / trt 等 OpenAI-compatible 端点） | throughput / TTFT / ITL / per-request e2e | 活跃；**旧 `python -m sglang.bench_serving` 路径在当前 main 已不存在**（[SGLang](../references.md)，[A]，2026-11-23 核实） |
| genai-perf | triton-inference-server/perf_analyzer 下 genai-perf/ | concurrency 或 request rate | output token throughput / TTFT / time to second token / ITL / request throughput | **官方 README 声明停止新功能开发（phased out），后继 ai-dynamo/aiperf**（[NVIDIA, genai-perf](../references.md)，教学参照，2026-11-23 核实） |

> **坑**：照旧教程敲 `python -m sglang.bench_serving` 会直接报模块不存在。触发条件：SGLang 当前 main（2026-11-23 核实）已把 bench 迁到 `python -m sglang.benchmark.serving`；现象：ModuleNotFoundError；规避：按上表新路径锚定，官方指南在 docs/docs/developer_guide/bench_serving.mdx（文件名未随模块迁移）。genai-perf 同理——新项目应直接用 aiperf，genai-perf 只作存量对照。

五个方法学陷阱，每个都给触发条件与后果：

**concurrency ≠ batch**。闭环压测固定 concurrency $C$（在途请求数），但 $C$ 里包含排队中、prefill 中、decode 中的全部请求；命题 1 的 $B$ 只是当前 iteration 在 decode 的序列数。Little 定律给出两者的关系： $C = \lambda \cdot \mathrm{E}[\mathrm{E2E}]$（稳态下在途请求数 = 到达率 × 平均逗留时间）。代入： $\lambda$ = 2 rps、平均 E2E = 8 s，得 $C$ = 16——这 16 条里可能只有 8 条在 decode batch 里，其余在排队或 prefill。所以「concurrency 16 的 benchmark」不等于「batch 16 的吞吐」，报告必须写明压测形态与实测 batch 分布。

**token 计数口径**。tokens/s 必须声明是输出 token 还是输入+输出合计：prefill 处理输入 token 的速率与 decode 产出输出 token 的速率差一个量级（1.2 的账），合计口径会把 prefill 产能混进 decode 指标。requests/s 与 tokens/s 的换算依赖平均输出长度，跨 workload 不可比。

**warmup**。CUDA Graph 捕获、JIT 编译、KV pool 预热都发生在头几个 iteration；不含 warmup 的测量把一次性初始化成本摊进稳态指标。规范做法：丢弃前若干请求或前若干秒的样本，并声明丢弃量。可复现报告的最低声明集：引擎版本 / commit、硬件 SKU、数据集与版本、到达过程（request rate 或 concurrency）、warmup 丢弃量、prefix caching 开关——缺任何一项，数字都不可跨报告比较（本书各章的实测引用一律带核实日期与版本锚定，同此纪律）。

**prefix cache 命中污染**。重复发送相同 prompt（或共享长系统前缀）时，第二次起命中 prefix cache，prefill 只剩未命中后缀——TTFT 骤降、吞吐虚高。测「冷启动产能」必须禁用 prefix caching 或保证 prompt 互不共享前缀；测「生产形态产能」则必须声明命中率。命中率的账归 2.2。

**离线 benchmark ≠ 在线 goodput**。离线测法（固定 batch、无到达过程、无 SLO 约束）测的是产能上界；在线 goodput 受排队、负载不均与 attainment 约束（§3 的 DeepInfer 80% 披露）。用离线数字做容量规划必须打折，折扣系数取决于 workload 形态与 SLO 紧度（9.1 的容量账）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 请求链路 = API → tokenize → 排队 → prefill → decode 循环 → detokenize → 流式返回；排队在高负载下是 TTFT 主项 | §1 链路图 | §1、1.3 |
| TPOT 是请求级均值、TBT / ITL 是逐间隔样本；DistServe 用 TPOT、DeepInfer 用 TBT | 式 (1) | §2、练习 1 |
| E2E = TTFT + (T_out − 1)·TPOT；512 输出 @30 ms 口径下 decode 占 97% | 式 (2)，15.83 s | §2、练习 1 |
| SLO 写在百分位上；均值对长尾不敏感（0.68 s 均值 vs P99 = 5 s 实例） | §2 百分位段 | §2、练习 4 |
| 稳态 decode 吞吐 = B/τ_step，单请求 TBT = τ_step；吞吐与延迟在 B 上反向 | 命题 1 | §3、练习 2 |
| goodput = per-GPU 归一 + attainment 目标下的最大请求率，非达标裸吞吐 | 式 (3) | §3、练习 3 |
| DistServe 13B A100 @90%：colocate 1.6 → 2P+1D 3.3 rps/GPU，2.1× | §3 实例 | §3、练习 3、8.2 |
| 在线生产吞吐 ≈ 完全均衡离线的 80%（DeepInfer 官方披露） | §3 引文 | §3、9.1 |
| Azure trace（Splitwise 释放，TIMESTAMP/ContextTokens/GeneratedTokens）与 ShareGPT（对话文本）归属分开、用途互补 | §4 | §4、1.3 |
| concurrency = λ·E[E2E] ≠ decode batch；benchmark 五陷阱：concurrency、token 口径、warmup、prefix 命中、离线≠在线 | §5 | §5、练习 2 |

## 练习

1.（计算 · [10]）本章口径（TTFT = 0.5 s、TPOT = 30 ms）下：(a) $T_{\mathrm{out}}$ = 512 的 E2E；(b) $T_{\mathrm{out}}$ = 8 的 E2E 与 TTFT 占比；(c) 若 SLO 要求 E2E ≤ 5 s，最大输出长度是多少（TPOT 不变）？校验：(a) 与式 (2) 的正文实例一致；(c) 的答案向下取整后代回式 (2) 应恰不超 5 s。
2.（推导 · [20]）证明命题 1（稳态 decode 吞吐 = $B / \tau_{\mathrm{step}}$、单请求 TBT = $\tau_{\mathrm{step}}$），并指出证明用到的两个前提（每序列每步恰 1 token、步时恒定）各在哪类系统机制下失效（提示：投机解码 4.4、抢占 4.2）。再用 Little 定律解释：concurrency = 16、 $\lambda$ = 2 rps 时平均 E2E 是多少，为什么这 16 条不等于 batch 16。校验： $B$ = 1 时命题 1 退化为单流 token 速率 = $1 / \tau_{\mathrm{step}}$；Little 定律结果与 §5 实例一致。
3.（计算 · [30]）goodput 口径对账（DistServe 论文 §1 数值：13B、A100、 $\theta$ = 90%，colocate 单卡 1.6 rps；分离后 prefill 实例 5.6 rps、decode 实例 10 rps）。(a) 2P+1D 的整体请求率为什么是 $\min(2 \times 5.6, 10)$ 而不是两者之和或均值？(b) 算 per-GPU goodput 与相对 colocate 的倍数；(c) 若把配置改成 1P+2D，整体请求率与 per-GPU goodput 各变多少？由此说明配比为什么是优化变量（8.2 的入口）。校验：(b) 与论文给出的 3.3 rps/GPU、2.1× 一致；(c) 的瓶颈应切换到 prefill 侧。
4.（代码 · [30]）百分位陷阱仿真（教学用，纯 Python 即可，无需 GPU）：构造 1000 个 TTFT 样本——90% 取自 0.2 s 附近的窄分布、10% 取自 5 s 附近的窄分布（加小抖动避免分位数落在平台边界）。(a) 先写出你对均值、P50、P90、P99 四个数的大小排序预测及理由；(b) 计算四个统计量，验证均值 > P90 且 P99 落在尾部；(c) 把尾部占比从 10% 扫到 1%，均值与 P99 各怎么动？由此说明「均值达标」为什么不能推出「P99 达标」。校验：P50 ≤ P90 ≤ P99 恒成立；尾部占比 10% 时均值 ≈ 0.68 s（与 §2 实例同口径）。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——vLLM 系 benchmark 的引擎背景
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——bench 入口 vllm/benchmarks/{serve,latency,throughput}.py + CLI vllm/entrypoints/cli/benchmark/（2026-11-23 核实）
- [Zhong et al., 2024](../references.md)：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving，arXiv:2401.09670，[A]（key: distserve）——goodput 定义原文与 13B A100 数值实例（§1）；TPOT 用法
- [Yu et al., 2022](../references.md)：Orca: A Distributed Serving System for Transformer-Based Generative Models，USENIX OSDI'22，无 arXiv，[B]（key: orca）——iteration-level scheduling 原文出处
- [Patel et al., 2024](../references.md)：Splitwise: Efficient generative LLM inference using phase splitting，ISCA 2024，arXiv:2311.18677，[B]（key: splitwise）——Azure trace 释放与 characterization 口径（§3）
- [Microsoft Azure, AzureLLMInferenceDataset2023](../references.md)：Azure/AzurePublicDataset，教学参照（key: azure-llm-trace）——trace schema 与许可
- [社区, ShareGPT_Vicuna_unfiltered](../references.md)：hf-mirror 传输核实，教学参照（key: sharegpt）——benchmark 对话数据集
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——bench serving 新路径与版本漂移（2026-11-23 核实）
- [NVIDIA, genai-perf](../references.md)：triton-inference-server/perf_analyzer，教学参照（key: genai-perf）——指标集与 phased out 状态、后继 ai-dynamo/aiperf
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：deepseek-ai/DeepInfer doc，教学参照 + 官方披露（key: deepinfer-doc）——TBT 用法（L31）、「80% of perfectly balanced offline」（L155），2026-11-23 核实
