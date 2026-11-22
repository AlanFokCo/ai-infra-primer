# Inference 维度 Spec 对抗审查报告 — Evaluator C（P8-P9：分布式推理与 PD 分离 / 生产全流程）

- **审查对象**：`docs/superpowers/specs/2026-11-22-inference-dimension-design.md` §5 Part 8（8.1-8.5）/ Part 9（9.1-9.4）、§4.3 符号表、§6.3 强制重推导清单（8.1 / 8.2 / 9.1 子集）、§2 边界与 errata 通道、预核实附录
- **审查人**：evaluator C（fresh context，未继承任何写作上下文；spec 非本人所写）
- **日期**：2026-11-22
- **方法**：全部一手核实——arXiv API 批量核对 7 个 P8/P9 arXiv ID↔标题（7/7）+ LMDeploy 论文补查；GitHub API 核 11 个 repo（含任务指定 5 个）与文件级 API 源码；DeepInfer `doc/inference_deployment_qa.md`（34,526 B / 373 行）与 README 全文一手读取（raw.githubusercontent.com 被 SSL reset 拦截 → 改用 GitHub contents API `Accept: application/vnd.github.raw+json` 取得，渠道已标注）；DistServe / Splitwise arXiv HTML 全文抓取 grep 关键论断；LoongServe / MemServe / Llumnix / Mooncake / HybridFlow 摘要核对；NIXL Python API 源码、Mooncake transfer engine 设计文档、Dynamo / TRT-LLM / LMDeploy README 核对；数学独立重推导 6 组（Wide-EP、KV 传输 vs 重算、TP decode 延迟、容量成本账、xPyD 配比、goodput 优化）含数值代入；跨维度指针目标逐章核对（training 5.1/5.4/5.5/7.3/8.4/8.5、models 2.4/4.1/4.2/4.5）
- **severity 口径**：STYLE_GUIDE §10——S1 事实性致命；S2 实质问题（约定冲突 / 推导断裂 / 协议缺口 / 局部事实错误）；S3 minor

## Findings 表

### S1（事实性致命）

无。7/7 arXiv ID↔标题全中；任务指定 5 repo 全部存在；P8/P9 引用的 DeepInfer 生产数字与 QA doc 逐字核对无虚构；$38/GPU-hour 在 spec 9.1 中**正确标注为收入（off-peak 价）非成本**（任务重点核查项，通过）；大纲级公式草图无错误断言。

### S2（实质问题，须修复）

| # | 位置 | 断言 | 证据（一手来源） | 建议修正 |
|---|---|---|---|---|
| S2-1 | §4.1 [B] 例 list；1.1 / 8.2 / 9.1 锚点「DistServe [A/B ⟨verify code⟩]」；附录延迟核实 list | DistServe 列为 [B]（「有 paper、无官方代码或代码不全」）例，官方代码开放程度 ⟨verify⟩ | DistServe 论文（2401.09670v3 HTML 全文）脚注 2 自链官方实现「we build DistServe … https://github.com/LLMServe/DistServe」；GitHub API 核实 `LLMServe/DistServe` 存在（847 stars，desc "Disaggregated serving system for Large Language Models (LLMs)"，pushed 2025-04-06）。按本 spec §4.1 自己的口径 [A] = 正式 paper + official code，DistServe 满足 [A] | ⟨verify⟩ 就地销项：DistServe 从 [B] 例 list 移入 [A] 例 list（2401.09670 + LLMServe/DistServe）；1.1 / 8.2 / 9.1 锚点改 [A]；references 骨架收录该 repo。若 W5 research pass 发现代码不全再降 [B] 并注明，但现有证据支持 [A] |
| S2-2 | §4.1「教学参照 + 官方实现 = 无独立论文…」list 含 LMDeploy；9.2 锚点「TensorRT-LLM / LMDeploy [教学参照 + 官方实现]」 | LMDeploy 归「无独立论文」档 | InternLM/lmdeploy README（GitHub contents API 一手）news 行：「[2026/08] Our paper, "LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind," has been accepted to EuroSys 2027」；arXiv API 核实 **2508.15601**（v2）标题逐字对上。正式 paper + official repo → 按 §4.1 口径与 STYLE_GUIDE §6 回升判定应为 [A] | LMDeploy 移出「无独立论文」list；9.2 锚点 LMDeploy 改 [A]（2508.15601 + InternLM/lmdeploy），Sources 加该论文；TensorRT-LLM 无正式 paper，维持教学参照 + 官方实现不变 |
| S2-3 | 8.1 内容行 | 「DeepInfer 不用 CP 的官方 rationale：端到端慢 ≤5%、worsens EP balance、kernel/comm/CPU 开销」 | QA doc L67 原文：CP「reducing tail TTFT … up to 5% slower than DP end to end … making it a **reasonable tradeoff**.**The main reason we do not currently use it is that V4.1 is fast enough to keep tail TTFT acceptable even on a complete cache miss**. Although CP improves data-parallel load balance … tends to worsen expert-parallel load balance. It also adds kernel, communication, and CPU overhead」。spec 所列三条均为 doc 原文事实，但**遗漏 doc 明示的主因**（模型够快 → 全 miss 下尾 TTFT 也可接受 → CP 的收益不必要），且把 doc 称为「合理代价」的 ≤5% 慢摆成决策理由首位——官方口径的重心失真 | 8.1 括注补主因：「主因：V4.1 足够快、完全 cache miss 下 tail TTFT 仍可接受（CP 收益不必要）；代价面：e2e 至多慢 5%（doc 称合理 tradeoff）、worsens EP balance、kernel/comm/CPU 开销」。教学结论应落在「CP 取舍是模型/负载依赖的」而非「CP 慢 5% 所以不用」 |
| S2-4 | 9.1 内容行「TTL vs 命中率：72h agent / 7d chatbot、72h 损失 0.5-1%」；同文见 2.2 与预核实附录（跨路去重） | 「72h 损失 0.5-1%」未带 traffic 限定，易被读成「生产 72h agent TTL 损失 0.5-1% 命中率」 | QA doc L187 原文：「a TTL of 3 days for agent traffic and 7 days for chatbot traffic … Empirically, a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate」；L211：「a TTL of 72 hours for agent traffic and 7 days for chatbot traffic **captures almost all reusable cache entries**」。即 0.5-1% 损失是「chatbot 若缩短到 3d TTL」的反事实代价，生产配置（agent 72h / chatbot 7d）近乎无损——spec 压缩句恰好丢掉关键限定，方向可被读反 | 9.1（并建议 2.2 / 附录同步）改为「chatbot  traffic 若用 72h TTL（而非 7d）损失 0.5-1% 命中率；生产 TTL agent 72h / chatbot 7d 近乎捕获全部可复用条目——官方披露」 |
| S2-5 | §2 errata 通道例 list；7.4 / 8.1 EPLB 覆盖与 training 已发布文本的接口 | spec §2 errata 通道只描述「泛指向 inference 维度」的指针收窄，例 list（models 2.4 §6 / 4.1 §6 / 4.5 §7、training 8.5 / 6.3）未含 training 5.4；而 training 5.4 把 EPLB 指向了**错误维度** | training/05-model-parallelism/04-expert-parallelism.md L79（已发布文本）：「训练侧靠路由控制、推理侧靠容量复制，**EPLB 一类的系统实现归 cluster 维度**」。本 spec 7.4 深度走读 EPLB（冗余副本 / 统计窗 / 重分配 / symm_mem shuffle / moe_lb 数字 / EPLB repo 锚点）、8.1 recap——EPLB 是 serving 侧负载均衡算法，按本 spec §2 边界（cluster 只管通信硬件）应归 inference。W0 盘点若只按「泛指向 inference」模式扫描会漏掉这条指向 cluster 的 stale 指针；cluster 维度开工时还会二次冲突 | §2 errata 例 list 显式加 training 5.4 §4；W0 盘点把「指向错误维度的 serving 指针」纳入扫描模式；errata 最小修订收窄为「serving 侧 EPLB 归 inference 7.4、通信硬件归 cluster」（先例：training/README.md 2026-11-16 对 models 4.5 §7「全部归 cluster 维度」的同型收窄，机制已验证可行） |

### S3（建议性 / 一致性，累计下波清零）

| # | 位置 | 断言 | 证据 | 建议修正 |
|---|---|---|---|---|
| S3-1 | 8.1 内容行「每层 all-reduce 延迟 × $L$」 | 每层 all-reduce 计数未显式 | training 5.1（已发布）L3/L17 口径：「每层 2 次前向 all-reduce … 前向每层 4bsh、反向对称翻倍」「每层 4 次 all-reduce」（含 backward）。serving 无 backward → 每 decode step **2L** 次 all-reduce（attn out-proj + MLP down-proj 各 1），消息量 $B d_{\mathrm{model}} \cdot \mathrm{bytes}$，decode 小消息延迟项主导（数值见 log D-M3） | 8.1 草图写显式「每层 2 次前向 all-reduce × L（training 5.1 的 4 次含 backward，serving 减半）」，防 writer 沿「× L」字面推出 L 次的 2× 低估 |
| S3-2 | 8.2 内容行「传输字节 / 带宽 vs 重算 FLOPs / FLOPS 的临界推导」 | 临界推导草图未提示结构陷阱 | 独立重推导（log D-M2）：前向线性项下传输字节 $\kappa L T$ 与重算 FLOPs $2NT$ **同 ∝ T，T 相消**——临界点是带宽阈值 $\mathrm{BW}^{\ast} = \kappa L F_{\mathrm{eff}}/(2N)$ 而非「临界上下文长度」；长度依赖只能来自 attention 二次项 / TTFT 预算 / 重算对 decode 的干扰；layer-wise 重叠（8.3） further 压低传输边际成本 | 8.2 关键推导处加注「线性项 T 相消 → 临界为带宽阈值；长度依赖由二次项与 TTFT 预算引入；重算代价须计干扰项」，防 naive 推导得出「不存在临界长度」或凭空造出临界长度 |
| S3-3 | §4.3 符号表 | 三处 notation 缺口：(a) P/D 实例/GPU 数无数学符号（P/D 只占文字、$G$ 归 GRPO、$\mathcal{G}$ 归 goodput），但 8.2 xPyD 推导与 9.1 sizing 必需计数量；(b) $\lambda$ 到达率声明规则只写「排队章」，9.1 sizing 同样用 $\lambda$；(c) 9.1 成本文本密集出现裸货币 $（「$38/GPU-hour」「$/M tokens」），按 STYLE_GUIDE §3 flanking 实测规则，同段两个裸 $ 会被配对成行内公式吞掉中间文本 | §4.3 现表无 P/D 计数符号；9.1 大纲原文即含「$/GPU-hour × GPU 数 / tokens → $/M tokens」双 $ 模式；STYLE_GUIDE [G1] 提供 \$ 转义但 spec 未提示 | §4.3 增补：P/D GPU 数用 $n_P / n_D$（或同型下标量，禁裸 P/D/G）；$\lambda$ 声明规则扩为「任何用 $\lambda$ 表到达率的章（1.3 / 9.1）」；货币一律 \$ 转义或写 USD/美元（9.1 尤甚），大纲行文同步示范 |
| S3-4 | 9.2 内容行「TensorRT-LLM（… inflight batching、disagg、Blackwell 支持 …）」 | 与 8.4「TensorRT-LLM disagg ⟨verify 实现形态⟩」潜在双展开 | 8.4 已列 TRT-LLM disagg 为对照轴之一；9.2 再列 disagg 无 recap+指针纪律约束（§4.2 精神） | 9.2 的 disagg 词后加「recap 8.4 + 指引」，9.2 只留选型定位一句 |
| S3-5 | 附录延迟核实 list；§9 risk 1 组织更名 list（现仅 outlines / Mooncake） | P8/P9 四个 ⟨verify code⟩ 本轮可销项 + 新增更名案例 | GitHub API：Llumnix 官方代码存在，`AlibabaPAI/llumnix` **301 → llumnix-project/llumnix-ray**（README 自链 2406.03243 与 OSDI'24 citation）→ [A]；`LoongServe/LoongServe` 存在且论文 HTML 自链 → [A]；MemServe 论文无代码链接、GitHub search 无官方 repo（同名 repo 均为无关项目）→ [B] 确认；Splitwise 仅作者模拟器 `mutinifni/splitwise-sim`（论文 ref [20]）+ vLLM PR #2809（ref [1]），无完整官方实现 → [B]（代码不全）确认。另 `MoonshotAI/Mooncake` 旧路径 API 与 web 均 **404 无重定向**——risk 1「用重定向跟到现名」策略对 Mooncake 失效，须直接用已核实新名 | 附录销四项 ⟨verify⟩ 并记 Llumnix 新路径；risk 1 更名 list 加 Llumnix（AlibabaPAI → llumnix-project，repo 名 llumnix → llumnix-ray），并注明 Mooncake 旧路径无重定向、以 kvcache-ai/Mooncake 为准 |
| S3-6 | §6.3 强制重推导清单 vs §9 risk 5 | risk 5 称「强制重推导清单覆盖全部算账重章」，但 8.3（P2D 带宽需求 / 跨 DC 可行性账）与 8.5（伸缩触发账 / grace 迁移成本）有「关键推导」却不在清单（清单只覆盖 12 个【重】章） | §4.4 的 12 重章 ↔ §6.3 清单一一对应；8.3 / 8.5 非【重】章但含数值推导，且 8.3 带宽账与 8.2 传输临界强耦合 | 二选一：W5 强制子集扩列「8.3 P2D 带宽账 + 跨 DC 可行性账、8.5 伸缩触发账」；或把 risk 5 措辞改为「覆盖全部【重】标记算账章」消除内部不一致 |
| S3-7 | 9.1 / 8.2 生产数字口径 | (a)「~1:1」未注明是 DeepInfer 流量下的经验值；(b) 存储算例缺 doc 给出的两个 sizing 输入；(c) 155T tokens/day 未要求声明 token 计数口径 | QA doc L91：「**In practice** the GPU ratio between P and D is roughly 1:1」（经验口径，非普适最优；xPyD 推导本身给出负载依赖配比，log D-M5）；L101 每存储服务器「enough total capacity to retain **24 hours of data**」；L105「14.4 TB per GPU measured as **logical** size; the physical capacity required is larger」（EC 10+2 → 物理 ×12/10）；doc 未说明 155T 含输入/输出/cache 命中与否；doc 混用 TB（十进制）与 PiB（二进制），§4.3 要求注明口径 | 8.2 / 9.1 注「~1:1 为 DeepInfer 生产流量经验配比，最优配比随输入/输出长度比与硬件 F/BW 比变化」；9.1 算例输入补 24h 保留准则与 EC 物理开销；155T 代入时声明计数口径；TB/PiB 换算按 §4.3 注明 |
| S3-8 | 1.1 内容行「ShareGPT / Azure trace ⟨verify 出处归 Splitwise⟩」（跨路 A，verify 销项记录在此） | 该行把 ShareGPT 与 Azure trace 并排挂在 Splitwise 核实标签下，易误读为两者同源 | Splitwise 全文（2311.18677v2）：「We use production traces taken from **two Azure LLM inference services on November 11th 2023** … coding and conversation … released a subset … at https://github.com/Azure/AzurePublicDataset」——Azure trace 归属 Splitwise ✓；ShareGPT 非 Splitwise 数据集（vLLM 系 benchmark 常用，另归出处） | ⟨verify⟩ 销项：Azure trace = Splitwise（Azure/AzurePublicDataset，AzureLLMInferenceDataset2023）；1.1 行文把 ShareGPT 与 Azure trace 分开归属 |
| S3-9 | 8.2 / 9.1 goodput 用法（承接 1.1 定义） | 1.1 摘要式定义「SLO 达标请求率」偏松，8.2/9.1 若沿此口径会丢 per-GPU 归一与 SLO attainment 目标 | DistServe 原文（log C-1 逐字引句）：goodput = 「the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) **for each GPU provisioned**」——per-GPU 归一 + attainment 目标（90%/99%）是定义要件，直接决定 9.1 的冗余 sizing 与成本换算 | 1.1 定义章按原文口径交付（per-GPU goodput + attainment 目标）；8.2 recap 与 9.1 sizing 显式沿用，不得把 goodput 写成「达标请求的裸吞吐」 |

## Verification log（抽查项 × 来源 URL/引句 × 结论 × 覆盖率）

### A. arXiv ID ↔ 标题（arXiv API `id_list` 批量，2026-11-22，https://export.arxiv.org/api/query?id_list=2401.09670,2311.18677,2407.00079,2404.09526,2406.17565,2406.03243,2409.19256）

**P8/P9 全部 7 个唯一 arXiv ID，7/7 命中（覆盖率 100%）**：

| ID | spec 期望 | API 返回标题（逐字） | 结论 |
|---|---|---|---|
| 2401.09670（v3） | DistServe | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（comment: OSDI 2024） | ✓ |
| 2311.18677（v2） | Splitwise | Splitwise: Efficient generative LLM inference using phase splitting | ✓ |
| 2407.00079（v5） | Mooncake | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | ✓ |
| 2404.09526（v2） | LoongServe | LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism | ✓ |
| 2406.17565（v3） | MemServe | MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool | ✓ |
| 2406.03243（v1） | Llumnix | Llumnix: Dynamic Scheduling for Large Language Model Serving（comment: To appear at OSDI '24） | ✓ |
| 2409.19256（v2） | HybridFlow（9.3 指引） | HybridFlow: A Flexible and Efficient RLHF Framework | ✓ |
| 2508.15601（v2，补查） | LMDeploy 论文（S2-2 证据） | LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind | ✓ 存在 |

### B. GitHub repo / 文件路径（GitHub API，2026-11-22）

| 抽查项 | 来源 | 结论 |
|---|---|---|
| ai-dynamo/nixl（8.3） | api.github.com/repos/ai-dynamo/nixl | ✓ desc "NVIDIA Inference Xfer Library (NIXL)" |
| ai-dynamo/dynamo（8.4） | api.github.com/repos/ai-dynamo/dynamo | ✓ desc "A Datacenter Scale Distributed Inference Serving Framework" |
| kvcache-ai/Mooncake（8.2/8.3/6.3） | api.github.com/repos/kvcache-ai/Mooncake | ✓ desc "Mooncake is the serving platform for Kimi … Moonshot AI"；**旧路径 MoonshotAI/Mooncake API 与 web 均 404、无重定向**（spec 用新名正确；risk 1「跟重定向」策略对此例失效，见 S3-5） |
| NVIDIA/TensorRT-LLM（9.2/8.4） | api.github.com/repos/NVIDIA/TensorRT-LLM | ✓ desc 含 "easy-to-use Python API to define Large Language Models" |
| InternLM/lmdeploy（9.2） | api.github.com/repos/InternLM/lmdeploy | ✓ desc "toolkit for compressing, deploying, and serving LLMs" |
| deepseek-ai/DeepInfer（P8/P9 官方披露源） | api.github.com/repos/deepseek-ai/DeepInfer | ✓ Apache-2.0、default branch main、desc "High-performance, large-scale inference framework for DeepSeek models" |
| LLMServe/DistServe（S2-1） | api.github.com/repos/LLMServe/DistServe + 论文脚注 2 | ✓ 存在，847 stars，pushed 2025-04-06；论文自链 → 官方代码 |
| LoongServe/LoongServe（8.1 ⟨verify code⟩） | api.github.com/repos/LoongServe/LoongServe + 2404.09526v2 HTML 链接 | ✓ 存在且论文自链 → [A] |
| Llumnix 官方代码（8.5/4.2 ⟨verify code⟩） | api.github.com/repos/AlibabaPAI/llumnix → **301** → llumnix-project/llumnix-ray；其 README 自链 arxiv.org/abs/2406.03243 + OSDI'24 citation | ✓ 官方代码存在，现名 llumnix-project/llumnix-ray → [A]（新更名案例，S3-5） |
| Splitwise 官方代码（8.2 ⟨verify⟩） | 论文 ref [20] SplitwiseSim（github.com/Mutinifni/splitwise-sim，API 核实 mutinifni/splitwise-sim 存在，170 stars）+ ref [1] vLLM PR #2809；GitHub search "splitwise" 前 8 命中无完整官方 serving 实现（其余为同名记账 app SDK） | ✓ [B]（代码不全：仅模拟器 + vLLM PR）确认 |
| MemServe 官方代码（8.4 ⟨verify code⟩） | 2406.17565v3 HTML 全文 github 链接仅 TensorRT-LLM（related work）；GitHub search "memserve in:name" 命中均无关（Xiaoxiao-Liu/MemServe 无描述且作者名不在论文作者 list；HeatherHong/MemServe 为 GPU-DRAM-SSD 分层原型，异项目） | ✓ 无官方代码 → [B] 确认 |
| NIXL agent API / descriptor 语义（8.3） | contents API：README.md、docs/python_api.md、**src/api/python/_api.py**（54 KB 源码） | ✓ `nixl_agent` / `nixl_agent_config`、`nixl_prepped_dlist_handle`（"prepared transfer descriptor list"）、`nixlRegDList` / `get_reg_descs` / register_memory；UCX 主 backend + GDRCopy；「UCX / GDR / RDMA 抽象、agent API、descriptor 语义」成立 |
| Mooncake transfer engine multi-NIC pooling / topology-aware（8.3） | git trees API 定位 + blob API 取 docs/…/design/transfer-engine/index.md（57 KB） | ✓ 原文「It supports multi-network card pooling and retry functions」「Transfer Engine implements a topology-aware path selection algorithm」（topology matrix、preferred/secondary NIC、>64KB 切片多路径） |
| Dynamo planner / router / KV-aware / disagg（8.4） | contents API：README.md | ✓ 特性表「Disaggregated Serving / KV-Aware Routing / SLA-Based Planner」；「Dynamo is the orchestration layer above inference engines」定位与 8.4 一致 |
| TRT-LLM disagg / Blackwell / Python API（9.2/8.4） | contents API：README.md | ✓ news「[06/19] Disaggregated Serving in TensorRT LLM」官方博客链接、DeepSeek-V4 on Blackwell 博客、「pythonic framework」；inflight batching 一词未在 README 出现（docs 级术语，W5 research pass 补文档出处） |
| LMDeploy TurboMind / persistent batch / W4A16（9.2） | contents API：README.md | ✓ 「persistent batch(a.k.a. continuous batching)」「W4A16 inference for sm_75」、TurboMind 多处；另发现论文（S2-2） |
| vLLM v1 KV connector（NIXL backend）文件级路径（8.4 ⟨verify 路径⟩） | GitHub API rate limit（unauthenticated 60/h 耗尽）+ github.com HTML 超时，**本轮未独立取到目录 listing**；旁证：Mooncake README news（2025-12-19）「Mooncake Transfer Engine has been directly integrated into vLLM v1 as a KV Connector in PD-disaggregated setups」（docs.vllm.ai 链接）；spec 附录称 2026-11-22 API 已核 `vllm/v1/` 树 | 存在性有官方旁证 ✓；具体 nixl_connector 文件名留 W5 research pass（spec 本就标 ⟨verify 路径⟩，非缺口） |
| SGLang srt/disaggregation 目录（8.4 ↔ 6.3） | 同上 rate limit；旁证：Mooncake README「TE is used in production inference stacks such as SGLang, vLLM, TensorRT-LLM …」+ spec 附录 API 预核实记录 | 存在性有旁证 ✓；listing 留 W5 复核（6.3 主责在 route B） |

### C. 论文内容断言（全文/摘要 grep + 逐字引句）

| # | 抽查项 | 来源与引句 | 结论 |
|---|---|---|---|
| C-1 | DistServe goodput 定义（8.2 recap 1.1、9.1 sizing 基础） | 2401.09670v3 HTML 全文：「maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query」 | ✓ 定义成立；per-GPU 归一 + attainment 目标是要件（S3-9） |
| C-2 | DistServe 优化变量（8.2「P/D 实例数与并行度联合优化 + placement」） | 同上：「The optimization ensures maximizing the per-GPU goodput and may assign different numbers of GPUs and parallelism strategies to each phase … DistServe then scales this allocation to multiple instances via replication until meeting the user-required traffic」；摘要：「places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation」 | ✓ 实例数（GPU 数）+ 并行度 + placement 三变量与 spec 草图一致 |
| C-3 | DistServe 用 TPOT 术语（1.1「DistServe 用 TPOT、DeepInfer 用 TBT」） | DistServe 摘要「time per output token (TPOT)」；DeepInfer QA doc L31「TTFT and TBT can be controlled and optimized independently」 | ✓ 两词归属正确 |
| C-4 | Splitwise Azure trace 归属（1.1 ⟨verify⟩ / 8.2） | 2311.18677v2 HTML 全文：「We use production traces taken from two Azure LLM inference services on November 11th 2023 … coding and conversation … released a subset of our traces at https://github.com/Azure/AzurePublicDataset」；ref [4] AzureLLMInferenceDataset2023 | ✓ 归属 Splitwise 确认（S3-8 销项） |
| C-5 | Splitwise mixed pool（8.2） | 同上全文：「mixed pool, expands and contracts as needed by the workload」「mixed pool use mixed continuous batching」「mixed pool to reduce fragmentation and meet SLOs at higher loads」 | ✓ mixed pool 为原文机制 |
| C-6 | Splitwise phase splitting / 传输（8.2） | 摘要：「compute-intensive prompt computation … memory-intensive token generation」「splitting the two phases … onto separate machines」「state transfer … using the fast back-plane interconnects」 | ✓ |
| C-7 | LoongServe ESP（8.1） | 2404.09526v2 摘要：「we propose a new parallelism paradigm, elastic sequence parallelism (ESP) … elastically adjusting the degree of parallelism in real-time … reducing key-value cache migration overhead」 | ✓ 「elastic sequence parallelism」与 spec 一致 |
| C-8 | Llumnix live migration（8.5 recap 4.2） | 2406.03243v1 摘要：「reschedules requests … efficient and scalable live migration mechanism for requests and their in-memory states」 | ✓ |
| C-9 | MemServe elastic memory pool / context caching（8.4） | 2406.17565v3 摘要：「MemServe introduces MemPool, an elastic memory pool managing distributed memory and KV caches across serving instances … combines context caching with disaggregated inference … global prompt tree-based locality-aware policy」 | ✓ |
| C-10 | Mooncake KVCache-centric 分离（8.2 recap 2.4） | 2407.00079v5 摘要：「KVCache-centric disaggregated architecture that separates the prefill and decoding clusters … KVCache-centric scheduler … balances maximizing overall effective throughput while meeting … SLOs」 | ✓ |
| C-11 | HybridFlow 定位（9.3 指引） | 2409.19256v2 摘要：RLHF dataflow、hybrid single/multi-controller；training 8.5 已锚定该论文 + verl-project/verl | ✓ 作训练侧指引正确，9.3 不展开训练侧编排 |

### D. DeepInfer 生产披露逐字核对（doc/inference_deployment_qa.md，GitHub contents API 全文 373 行读取；行号为本次抓取副本行号）

| # | spec 断言（位置） | doc 引句 | 结论 |
|---|---|---|---|
| D-1 | **$38/GPU-hour 是收入（off-peak）非成本**（9.1，任务重点） | L17：「A single cluster currently serves 155 trillion tokens per day, with overall **revenue** of US$38 per GPU-hour (at the off-peak price).」 | ✓ spec 标注正确，不可混用警告与 doc 一致 |
| D-2 | 155T tokens/day 单集群（9.1） | L17 同上「A single cluster currently serves 155 trillion tokens per day」 | ✓ 单集群口径正确 |
| D-3 | autoscaling：prefill 按 non-cached tokens / decode 按 running reqs + rate-limit 上下界（8.5，任务项） | L251：「Prefill determines the number of machines it needs from the number of non-cached tokens; decode adjusts based on the number of running requests. Adjustments are rate-limited and bounded above and below.」 | ✓ 逐字对上 |
| D-4 | grace 1 min + 未完成请求/KV 一次性转发其他 CE（8.5，任务项） | L253：「a decode engine (CE) enters a grace period (1 min) after receiving a shutdown notice; any requests still unfinished, along with their KV cache, are forwarded in one shot to other CEs to continue decoding.」 | ✓ |
| D-5 | quota >0.8 禁 scale-in + ready 机池 + 在线实例 > 需求（8.5，任务项） | L257：「ready machines held unoccupied in the pool as scale-out inventory, and always keeping more instances online than the computed demand — concretely, scale-in is forbidden whenever quota utilization is above 0.8.」 | ✓ |
| D-6 | EP failover：单通信域单 GPU 故障整实例不可用、重启从健康 GPU 重定 EP size（8.5，任务项） | L265：「all GPUs of an instance form a single communication domain, so a single failed GPU makes the whole instance unusable … the platform re-determines the EP size from all currently healthy GPUs, and the instance takes the number of GPUs it actually receives as its EP size.」 | ✓ |
| D-7 | ~1:1 P/D + 最小高效单元 64P+64D（8.2/9.1，任务项） | L91：「1P + 1D is deployable. In practice the GPU ratio between P and D is roughly 1:1. Since the recommended decode configuration is EP64 or above, the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.」 | ✓ 数字与「高效单元」措辞正确；「In practice」经验限定建议显式化（S3-7）；9.1「Wide-EP 最小 decode 域约束」正确捕获 EP64  rationale |
| D-8 | 重启分钟级 + quota 冗余/rerouting 容错哲学（8.5） | L261：「Restarts are on the order of minutes. Fault tolerance comes from PE / CE quota redundancy plus request-level rerouting, rather than from shortening the restart time of a single engine.」 | ✓ |
| D-9 | 训推集群共调度（8.5） | L119：「Training and inference clusters are managed together, and the resource allocation between them can be adjusted dynamically.」 | ✓ serving 侧资源视角，未越入训练侧编排 |
| D-10 | TP 不用 rationale：装得下 + SLA 可达不付通信（8.1） | L69：「Tensor parallelism (TP) has high communication overhead. We do not use it when the model fits in memory and the latency SLA can be met without it.」 | ✓ |
| D-11 | CP 不用 rationale（8.1） | L67 全文见 S2-3 引句 | ⚠ 三条代价属实但**主因遗漏** → S2-3 |
| D-12 | #DP = #EP attention DP（8.1） | L33：「attention uses data parallelism (#DP = #EP)」 | ✓ |
| D-13 | prefill 最小 EP/DP（HBM 允许）/ decode 最大 EP/DP（GPU 允许）（8.1 recap 7.4 / 附录） | L47-49：「Prefill: EP2–8 / DP2–8; use the smallest EP/DP that HBM allows. Decode: EP64–128 / DP64–128 (Wide-EP); use the largest EP/DP that the available GPUs allow.」 | ✓ |
| D-14 | ~400 Gbps/8 GPU 跨 DC + 压缩 KV 可行 + 可靠性代价（8.3） | L97：「Because V4.1 Flash compresses the KV cache substantially, splitting P and D across data centers is technically feasible (at some cost in reliability …) Budget roughly 400 Gbps of interconnect bandwidth per 8 GPUs.」 | ✓ |
| D-15 | GDR swap-in：RDMA NIC 直写不占 SM、copy engine 细粒度差（8.3 recap 2.4） | L191：「Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.」 | ✓ |
| D-16 | Entrypoint CHECK hash-chain 前缀匹配（8.4） | L191：「Matching is done by the Entrypoint against the local kvcache-store in the CHECK role, as a hash-chain prefix match over metadata only」 | ✓ |
| D-17 | VE 单 GPU + cheaper GPU 推荐（8.4/9.4） | L51：「VE: a single GPU per instance is generally enough, and cheaper GPUs are recommended.」 | ✓ |
| D-18 | VE LB fewest patches (w/patch)(h/patch)（9.4 ↔ 7.4） | L151：「the GPU currently processing the fewest image patches — computed directly from the dimensions as (width/patch_size) × (height/patch_size)」 | ✓ |
| D-19 | SO 专用机异步编译 + 入引擎前取结果 + 每步 automaton advance + mask（9.4） | L344-348：「SO compilation runs asynchronously, and a request retrieves the compiled result before it enters the engine … on a dedicated SO machine … at each decode step, the automaton is advanced … a mask is generated and applied」 | ✓ |
| D-20 | 存储 sizing：20×30.73 TB / ≥40 GB/s 读 / 5 GB/s 写 / scale out / 14.4 TB/GPU 逻辑（9.1） | L101：「20 × 30.73 TB SSDs per KV cache storage server … at least 40 GB/s read and 5 GB/s write throughput and enough total capacity to retain 24 hours of data. Scale out …」；L105：「about 14.4 TB per GPU measured as logical size; the physical capacity required is larger than that.」 | ✓ 数字全对；24h 准则与物理开销建议补入算例输入（S3-7） |
| D-21 | TTL 72h agent / 7d chatbot、0.5-1% 损失（9.1/2.2） | L187：「a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate」；L211：「72 hours for agent … 7 days for chatbot … captures almost all reusable cache entries」 | ⚠ 损失归属 chatbot 反事实缩短，spec 压缩句丢限定 → S2-4 |
| D-22 | ~80% of perfectly balanced offline（8.5/7.4 口径、1.1 benchmark 陷阱） | L155：「we reach roughly 80% of a perfectly balanced offline benchmark」 | ✓ |
| D-23 | EPLB decode EP64 → 7 expert/GPU（7.4 数字，8.1 recap 接口一致性抽查） | L131：「at EP64 … 7 experts per GPU after rounding, excluding the shared expert」；config 384 routed（附录 [W]）：384/64=6 +1 冗余=7 | ✓ 与 M1 重推导互洽 |
| D-24 | release 边界：KV 系统 / VE/P/D 编排 / DP LB 未含，引擎侧功能已大部分实现（8.4） | README L38-42：「Required for a complete inference service (not included in this release): A KV cache system; VE/Prefill/Decode disaggregation (most required engine functionality is already implemented); Load balancing across data-parallel replicas …」 | ✓ 8.4 边界口径与 README 逐字一致 |
| D-25 | balanced CP（8.1 引 README） | README L12：「Parallelism: Attention DP / balanced CP, MoE EP + EPLB」 | ✓ 「balanced CP」为 README 原文词；与 QA doc「CP 暂不用（生产）」不矛盾——release 支持 vs 生产不用，spec 两处出处分别标注正确 |
| D-26 | DeepInfer 定位：V4.1-Flash only / SM100 + Ascend 950 / Apache-2.0 / 源自 vLLM + SGLang/TRT-LLM 启发（8.4 背景、7.1 主责） | README L7、L35-36、L91、L95 | ✓（route B 主核，本轮顺带确认） |

### E. 数学独立重推导（evaluator 自行推导 + 数值代入，不依赖 spec 草图）

**M-1 Wide-EP 每步时间 ∝ 1/e（7.4 交付 / 8.1 recap / 1.2 伏笔；§6.3 清单项）**
设 E 个 routed expert、top-k、decode batch B（每 seq 1 token）、EP 度 e、均匀路由：
- 每 expert 期望 batch = Bk/E，**与 e 无关** ✓（1.2 论断成立；EPLB 冗余副本摊薄的是物理副本负载，逻辑 expert 口径不变）。
- 每 rank compute = (E/e)·(Bk/E)·2P_expert = 2P_expert·Bk/e ∝ Bk/e ✓。
- 每 rank 权重流量 = (E/e)·P_expert·bytes（前提：本地 expert 全部被激活，即大 B 情形；小 B 有未激活 expert 时流量更低）∝ E/e ✓。
- 每 rank all-to-all dispatch 量 ≈ B_local·k·d_model·bytes·(e−1)/e，随 e **饱和**；同步/延迟项随 e 增 → 通信地板。故 step time ∝ 1/e 直至通信/同步主导 ✓，与 doc「EP>32 递减」（L293-297：带宽不再瓶颈 → 通信/同步 + NVLink 多 job 干扰）一致。
- 数值（V4.1-Flash 几何取 spec 附录 [W]：E=384、k=6、d=5120；EP64/DP64、60 seq/rank）：B=3840、Bk=23040、Bk/E=**60**、Bk/e=**360**、E/e=6（+1 冗余=**7** ✓ doc D-23）；EP8→EP64 权重流量比 = **8×** 下降；每 rank dispatch ≈ **3.63 MB**（bf16，饱和量）。
- 边界 caveat（writer 应声明）：非 MoE 部分（DP attention KV、shared expert、dense 层）不随 e 缩 → Amdahl 地板；moe_lb（slowest/mean）乘子修正均匀路由假设。spec 草图把账限定在「MoE 权重流量/compute」✓ 正确scoped。

**M-2 KV 传输 vs 重算临界（8.2；§6.3 清单项）**
- 传输时间 t_xfer = κLT/BW_net；重算时间 t_recomp ≈ (2NT + attn 二次项)/F_eff。
- 线性项下两边同 ∝ T → **T 相消**，临界为带宽阈值 BW* = κL·F_eff/(2N)，不是临界长度（S3-2）。长度依赖来自 attention O(LT²d) 项、TTFT 预算、以及重算占用 decode 机的干扰成本；layer-wise 重叠使传输边际成本 = max(0, t_xfer − 可重叠 prefill 计算)。
- 数值（illustrative，V3 MLA 几何：κ=576 B/token/层 FP8、L=61、N_act=37B；H100 FP8 峰值 1979 TFLOPS × MFU 0.4）：KV=35,136 B/token；t_recomp=93.5 μs/token；**BW* ≈ 0.376 GB/s ≈ 3 Gbps**。DeepInfer 跨 DC 预算 400 Gbps/8 GPU = 6.25 GB/s/GPU ≫ BW* → 传输远优于重算，与 doc「压缩 KV 使跨 DC P/D 可行」（D-14）自洽 ✓。V4.1 κ 更小（per-16 e4m3 主 KV + L=40），结论同向更强。

**M-3 TP decode 延迟账（8.1；§6.3 清单项）**
- Megatron 型 TP：每层 forward **2 次** all-reduce（attn out-proj 行切 + MLP down-proj 行切；training 5.1 已发布口径「每层 2 次前向 … 每层 4 次含 backward」）。serving 无 backward → 每 decode step 2L 次，消息 m = B·d_model·bytes。
- 小消息延迟主导：ring 带宽项 2(t−1)/t·m/BW_link 数值（L=40、d=5120、t=8、B=60、bf16、NVLink 900 GB/s）：m=614,400 B、带宽项 ≈ **1.19 μs** ≪ 延迟项 ~10-20 μs → 每步通信 ≈ 2L×15 μs = **1.2 ms**，占 TPOT 预算可观份额 → 解释 doc「装得下 + SLA 可达就不付 TP 通信」（D-10）✓。
- spec 草图「每层 all-reduce 延迟 × L」计数歧义 → S3-1（正确为 2L）。

**M-4 容量/成本账（9.1；§6.3 清单项「算例重算」）**
- 量纲：$/M tokens = (c_$/GPU·h × G) / (tokens/h) × 10⁶ ✓（分母必须是**速率** tokens/h 与小时价对齐；PD 分离时 G = n_P + n_D，两池可异价）。
- 数值示例（教学口径）：c=2 $/GPU·h、G=128（64P+64D）、decode 吞吐取 doc DSpark 表 @60 conc 的 14,466 tok/s/rank × 64 rank = 925,824 tok/s → 3.333×10⁹ tok/h → 成本 **0.0768 $/M output tokens**；同 G 下收入侧 38 $/GPU·h → 1.459 $/M——**两者不可混用**，spec 9.1 的「收入非成本」警告正确且必要 ✓（D-1）。
- 存储链：14.4 TB/GPU（逻辑）× G ÷ 每服务器可用容量；20×30.73=614.6 TB raw，EC 10+2 → ×10/12 = **512.2 TB** 可用；G=128 → 1843 TB → **4 台**（未计 ≥40 GB/s 读 / 5 GB/s 写吞吐约束与 24h 保留准则，S3-7 补）。
- 互洽性抽查：100 PiB = 112,590 TB（十进制）÷ 14.4 TB/GPU ≈ **7,819 GPU**——doc 的存储集群容量与 per-GPU 配比在 ~7-8k GPU 集群规模下互洽；155T/day ÷ 24h = 6.458T tok/h，÷7.8k GPU ≈ 230 tok/s/GPU 均摊，量级合理 ✓。TB/PiB 混用须按 §4.3 注明口径。
- token 计数口径（输入+输出？cache 命中计否？）doc 未定义 → 算例必须显式声明（S3-7）。

**M-5 xPyD 配比 + DistServe goodput 优化（8.2；§6.3 清单项）**
- 设到达率 λ、输入长 T_in、输出长 T_out：P 侧需求 λ·2N·T_in FLOPs/s → n_P ≈ λ·2N·T_in/(F_P·ρ_P)（ρ_P 为排队论利用率上限，1.3 M/M/1/Kingman recap）；D 侧并发 C ≈ λ·T_out·TPOT（Little's law），带宽需求 C·κ·L·T̄/TPOT → n_D ≈ C·κ·L·T̄/(TPOT·BW_D·ρ_D)（T̄ ≈ T_in+T_out/2）。
- 配比 n_P:n_D = [2N·T_in/(F_P·ρ_P)] : [T_out·κ·L·T̄/(BW_D·ρ_D)]——**负载与硬件依赖**（输入/输出比、κ、F/BW 比）；MLA 小 κ 使 D 侧便宜、长输出 agent 负载使 D 侧贵。DeepInfer ~1:1 是其流量+硬件下的经验值（D-7「In practice」），且 n_D 受 EP64 最小域量子化 → 64P+64D 最小高效单元 ✓。spec 草图「prefill 算力需求 vs decode 带宽需求 → 最优配比」方向正确。
- DistServe 形式化：max per-GPU goodput（C-1 定义）over（prefill 并行度、decode 并行度、每实例 GPU 数、placement），profiling+simulation 搜索，再 replication 到目标流量（C-2）——与 spec「SLO 约束下 P/D 实例数与并行度联合优化 + placement 优化」一致 ✓。

**M-6 goodput 定义核对（1.1 → 8.2 → 9.1 链）**：见 C-1/S3-9；「SLO 达标请求率」摘要式定义需补 per-GPU 归一与 attainment 目标，否则 9.1 冗余 sizing 失去数学基础。

### F. 边界纪律与符号一致性（P8/P9）

| 检查项 | 结论 |
|---|---|
| 8.1 vs training P5（recap only） | ✓ 8.1 明示「training P5 recap + 指引 → serving 侧增量」；增量（无 backward、decode 小消息延迟敏感、DP attention for MLA、CP serving、组合决策表 serving 对照）均为 serving 侧；training 5.1/5.4/5.5 目标章存在且主题对上（01-tensor-parallelism / 04-expert-parallelism / 05-combining-strategies） |
| 通信硬件 → cluster | ✓ 8.3 明示「硬件 / 拓扑 / RDMA 协议归 cluster 维度边界声明」；NIXL/GDR/TE 只讲接口与账；8.1「NVLink 域内边界」是 serving 约束非硬件展开（1.3 立 scale-up domain 口径） |
| RL rollout 训练侧 → training 8.5/8.4 | ✓ 9.3 明示 recap（四模型显存/引擎拆分/权重同步训练侧）+ 引擎侧增量；off-policy 校正 → training 8.4（目标章 04-grpo-and-variants.md 含 GSPO off-policy 校正内容，核实）；HybridFlow 只作指引 ✓；training 8.5（05-llm-rl-systems.md）确含四模型/两引擎/权重同步账与 veRL 走读，接口吻合 |
| SDC → training 7.3 | ✓ 03-training-stability.md §2.4「硬件触发：SDC 与故障统计口径」存在，指针有效 |
| EPLB 归属 | ⚠ training 5.4 已发布文本指向 cluster，与本 spec 7.4/8.1 冲突 → S2-5 |
| 8.4 TRT-LLM disagg vs 9.2 | 潜在双展开 → S3-4；其余 8.4 各系统（vLLM/SGLang/Dynamo/MemServe/DeepInfer）与 2.4（Mooncake/LMCache/3FS）、2.2（prefix）无重复展开，MemServe 为 8.4 新引入 ✓ |
| 章间前瞻引用 | P8/P9 无未声明前瞻使用：8.2 Sarathi-Serve（4.3 已定义）、8.3 GDR（2.4）、8.5 Llumnix（4.2）、9.3 sleep mode（5.4）、9.4 dllm（6.3 展望）均回链；5.3/6.2/6.3 的「↔ 8.1」为纯文字指针非提前使用，5.3「无 backward」一句与 8.1 重叠属单句事实非推导重复，可接受 |
| §4.3 符号 | $t,p,e,d$（8.1「t/p/e/d 选择」✓）、$L$ ✓、goodput $\mathcal{G}$ 不占 $G$（training 8.4 GSPO 式确以 $G$ 为组大小，跨维一致 ✓）、P/D 文字 ✓、bytes 文字（8.3「KV 字节」、9.1「TB/GPU」）✓、混场章 8.1/9.1 无 B-bytes 用法 ✓；缺口：P/D 计数符号、$\lambda$ 声明范围、货币 $ 转义 → S3-3 |
| §6.3 清单 ↔ §4.4 重章 | 8.1/8.2/9.1 在清单且为【重】✓；8.3/8.5 关键推导不在清单 vs risk 5 措辞 → S3-6 |

### G. 覆盖率与限制

- **章节覆盖**：P8 5/5 + P9 4/4 = 9/9 章大纲级断言全查（100%）；§4.3 / §6.3（C 路子集）/ §2 边界 / 附录相关行全查。
- **arXiv**：7/7 ID↔标题（100%）+ 摘要级内容核对 7 篇 + DistServe/Splitwise 全文级 grep + LMDeploy 论文补查 1。
- **Repo**：任务指定 5/5 + 附加 6（DeepInfer、LLMServe/DistServe、LoongServe、llumnix 重定向链、splitwise-sim、MemServe search）；文件级源码/文档核实 4（NIXL _api.py、Mooncake TE 设计文档、Dynamo README、DeepInfer README+QA）。
- **DeepInfer 披露**：P8/P9 相关 doc 断言 26 项逐字核对（D-1…D-26），QA doc 全文读取（100%）、README 全文读取（100%）。
- **数学**：任务指定 5 组全部独立重推导（M-1…M-5）+ goodput 定义核对（M-6），均含数值代入与量纲检查。
- **限制（诚实声明）**：(1) vLLM nixl_connector 文件级路径与 SGLang disaggregation 目录 listing 因 GitHub API unauthenticated rate limit + github.com HTML 超时未独立取得——存在性有 Mooncake README 官方旁证与 spec 附录预核实记录，且 spec 对前者本就标 ⟨verify 路径⟩，留 W5 research pass；(2) raw.githubusercontent.com 本机 SSL reset → 全程改用 GitHub contents/blob API（raw accept header），渠道已标注；(3) V4.1-Flash config 字段未重取（route B 主责，hf-mirror 渠道），M-1 数值代入使用 spec 附录已录 [W] 字段（E=384/k=6/d=5120/L=40）并如此标注；(4) TRT-LLM「inflight batching」术语在 README 未出现，属 docs 级术语，W5 补文档出处（不构成 finding，feature 本身无争议）。

## 结论

- **S1 = 0，S2 = 5，S3 = 9**。
- 任务重点核查项全部通过：7 个 arXiv ID↔标题全中；5 个指定 repo 全部存在（kvcache-ai/Mooncake 新名正确）；**$38/GPU-hour 在 9.1 正确标注为收入（off-peak）非成本**；autoscaling / grace 1min / quota>0.8 / EP failover / ~1:1 五项生产披露与 QA doc 逐字一致；Wide-EP、传输 vs 重算、TP decode、容量成本四组账独立重推导成立（含数值互洽）。
- 5 条 S2 均为 spec 文本可修复项：两处证据分级错误（DistServe 应 [A]、LMDeploy 应 [A]）、两处官方披露口径失真（CP 主因遗漏、TTL 损失缺 chatbot 限定）、一处跨维度指针冲突（training 5.4 EPLB→cluster 需入 errata 通道）。无阻塞性设计缺陷；S2 修复后 P8/P9 大纲可锁定开工。
- 9 条 S3 以 notation 补全（P/D 计数符号、货币 $ 转义）、推导陷阱提示（TP 2L 计数、传输临界 T 相消）、⟨verify⟩ 销项记录（Llumnix/LoongServe/MemServe/Splitwise 四项本轮已解）与算例输入补全为主，按 §6.4 累计下波清零。
