# Inference W5 审查报告 — P8 分布式推理与 PD 分离（8.1–8.5）

日期：2026-11-23。Evaluator：fresh-context 对抗审查（不继承写作上下文）。
协议：`.superpowers/sdd/inference-w1/evaluator-brief.md` 三项任务 + spec §6.3 强制重推导 + W5 dispatch 特定项。
范围：`inference/08-distributed-and-pd/{01-serving-parallelism,02-pd-disaggregation,03-kv-transfer,04-pd-systems,05-elasticity-and-fault-tolerance}.md`。
基准：spec §5 P8 大纲 + §2 边界 + §4.1/4.2/4.3/6.3；STYLE_GUIDE；R2 style sheet；rubric；`inference/references.md`；`docs/research/inference-w5-sources.md`；training P5（recap 核对）。
方法：全部公式独立重推导（python 复核）；一手来源自行 fetch（arXiv abs/HTML、GitHub contents API/codeload、hf-mirror、lmsys blog）；不信任 writer 报告与 research notes 的数值，逐条对照原文。

## 0. 结论摘要

- **S1 = 0；S2 = 6；S3 = 9。**
- **强制重推导：8.1 PASS（TP decode 延迟账 + DP attention 通信账）；8.2 PASS（DistServe goodput 调和分解 + xPyD 配比 + 传输 vs 重算带宽阈值）。** 全部关键数值独立重算命中。8.1 §4 EP 例有一处公式-代入不一致（S2-3，不在两条强制项本体，结论稳健）。
- 一手来源核实覆盖：8/8 arXiv ID↔标题↔第一作者（abs/HTML 通道，export API 本轮限流）；DistServe/Splitwise/LoongServe/MemServe/Llumnix/Beyond-the-Buzz/Mooncake 全文/摘要关键句逐字；NIXL fresh tarball cb66ce83（18 插件 + C++ API 行号全中）；TRT-LLM disagg-serving.md + blog05 fresh；Dynamo disagg/routing/planner/README fresh；Mooncake 三头文件 + index.md fresh；DeepInfer doc 372 行 fresh（25 组行号/逐字全中）；V4.1 config hf-mirror fresh；vLLM/SGLang/llumnix-ray/LoongServe/DistServe repo 锚点（W3/W4/W5 同日 tarball + vLLM live API spot-check）。
- 机械门：lint 5/5 OK；math-boundary 5/5 exit 0；CJK 8.1=7668 / 8.2=7361 / 8.3=4697 / 8.4=3879 / 8.5=5759（8.1/8.2 ≥5500 达标，余 ≥3500）；rubric A1/A2/A3 禁令词扫描 0 命中。
- **lint 门禁缺口（advisory）**：`scripts/lint-math-boundary.py` 不检查 STYLE_GUIDE §3 反斜杠反转义陷阱（`\,` `\!` 等），故 S2-1/S2-2 漏网。建议追加反转义序列检查。

## 1. Findings 表

### S2（实质，必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 8.2 式 (5)(6)(7)(8)(11)(12) + L155 推导行（L103/109/113/119/155/159/171，共 17 处 `\,`） | display 公式用 `\,` 细间距，违反 STYLE_GUIDE §3 反转义硬规则 | GitHub Markdown 管线先把 `\`+ASCII 标点反转义：`\,`→`,`（STYLE_GUIDE §3 实测记录 + 白名单「`\,`…因反转义陷阱禁用」）。渲染后 `F_P\, \rho_P` 变 `F_P, \rho_P`——乘积被插入可见逗号，读成两个并列量，xPyD/BW* 核心公式语义受损。lint-math-boundary.py 未覆盖此项 | 17 处 `\,` 全改 `\thinspace`（或 `\ ` 控制空格）；与 8.1/8.3/8.5 已用 `\thinspace` 对齐 |
| S2-2 | 8.5 式 (5) L67 | `\mathrm{clip}\!\left(` 用 `\!` 负间距，违反 §3 反转义硬规则 | `\!`→`!`（STYLE_GUIDE §3），渲染成 `clip!(…)` 带可见感叹号 | 删 `\!` 或改 `\negthinspace`（白名单替代） |
| S2-3 | 8.1 §4 L118（EP dispatch 带宽项） | 公式与代入数不一致：式给 `(e−1)/e·(B/d)k·d_model·bytes_act/BW`，代入 B=64/d=e=8/k=6/FP8/IB 50 GB/s 应得 **4.30 μs**，正文却写 **0.61 μs** | python 复核：per-rank 245,760 B，shared-link 项 0.875×245760/5e10=4.30 μs；0.61 μs = 逐 peer 30,720 B/5e10（借自 3.3 §6 的 per-peer 锚）。同节 bf16/NVLink 上界 0.48 μs 又用 shared-link 式（0.875×491520/9e11=0.478）——两种链路模型混用无口径注。结论「延迟主导」两模型下均成立（4.3≪15、0.61≪15），1.24 ms 上界用 0.48 自洽 | 统一链路模型：要么按所写 shared-NIC 式把 0.61 改 4.3，要么把式改写为逐 peer 并行链路口径（τ_a2a + 逐peer字节/BW_link）并注与 §5 0.48 的模型差异；对回 3.3 §6 锚时声明口径 |
| S2-4 | 8.3 §5 L208（bounce buffer 条件） | 「默认 ≥1024 descriptor 且**均** ≤16 KiB 才启用」与一手来源及 8.4 矛盾 | TRT-LLM disagg-serving.md L182（fresh contents API）逐字：「only writes with at least 1024 descriptors **averaging at most 16 KiB** use the bounce buffer」；参数名 `max_average_descriptor_size`。8.4 §5 L109 正确写「平均 ≤16 KiB」。refs `tensorrt-llm` 条目亦误写「均 ≤16KiB」 | 8.3「均」→「平均」；refs tensorrt-llm 条目同步收窄；research notes 语义纠正 1 的「均」亦应更正 |
| S2-5 | 8.5 §5 命题 4 L124 + 总结 L175 + 练习 3(a) L186 | 命题不等式方向误：「利用率不超过 u_max ⟹ 可吸收故障比例 f 满足 f ≤ 1−u_max」对 u<u_max 不成立 | 由 (1−f)n_online≥n_demand 得 f ≤ 1−u（实际利用率）；u≤u_max ⟹ 1−u ≥ 1−u_max，故可吸收上界是 1−u（≥1−u_max），不是 ≤1−u_max。练习 3(a) 要求「由…与式 (7) 推出 f≤1−u_max」在 u<u_max 时不可证（不等号反向）。证明提示与数值例（u=0.8 边界）本身正确 | 改陈述为「保证**至少**可吸收 1−u_max 比例故障；最大可吸收比例 = 1−u ≥ 1−u_max，最坏情形 u=u_max 时为 1−u_max」；练习 3(a) 目标改为推 f≤1−u 并讨论 u=u_max 边界 |
| S2-6 | 8.3 §1 L37 vs §6.1 式 (4) L221 | TTFT 口径内部张力：§1 称「TTFT 的首 token 本身不等待 KV 到齐」，但式 (4) 把 t_exposed 计入 TTFT，且 §1 提点定义「暴露＝计入 TTFT 的传输时间」、6.3 §4 同口径 | 6.3 §4 L107 逐字「暴露指未被计算藏住、计入 TTFT 的传输时间」；§5 首 token 经 aux 随末 chunk 到 decode、client 从 decode 取流 ⟹ 用户感知 TTFT 含暴露传输。§1 把「首 token 在 prefill 侧**计算**早」误读成「TTFT 不等待传输」 | §1 句区分「首 token 计算（prefill 侧早完成）」与「首 token 交付（经 aux/末 chunk 到 decode 后才流给用户）」；明确用户感知 TTFT 含暴露项，与式 (4)/6.3 对齐；「第二 token」说法限于 decode 首次真实 forward |

### S3（minor，下波清零）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S3-1 | 8.2 §2 L63 | 「永远小于 min(g_P,g_D)、**大于**其一半」边界缺等号 | g_P=g_D=g 时 G*=g/2=min/2，恰等于一半（调和形 ab/(a+b)） | 「大于」→「不低于/至少」其一半（R2-6 边界反例配齐） |
| S3-2 | 8.2 §5 命题 6 L175 | 「当且仅当 BW≥BW*」缺命题 5 的「线性项口径」限定 | attention 二次项增加可藏计算量，全重叠阈值实为 BW*(T)<BW*；末层尾巴已在式 (13) 讨论承认 | 命题 6 陈述加「线性项口径下」，与命题 5 一致 |
| S3-3 | 8.1 §5 L165 | 「DP attention 把 **MoE 部分**字节减半（MoE TP）…attention 部分归零」措辞误导 | MoE-TP 形 MoE 字节 45.9 MB = 纯 TP 的 MoE 部分（91.8 的一半）；减半来自 attention 归零，非 MoE 部分本身减半；EP 39.3 较 MoE-TP 45.9 仅减 14% | 改「总字节减半（MoE TP 形）到减 57%（EP 形），因 attention 部分归零；MoE 部分与纯 TP 的 MoE 部分相同，EP 再减 14%」 |
| S3-4 | 8.3 §1 章脊 L9 / §6.3 L270 / 练习 4 L332 | 「7.3 **§5** 格式账」指针错（×3） | 7.3 节标题：§4「低精度落地：528/288/68 三格式字节账」，§5 是「DeepGEMM…Mega MoE」 | 三处「7.3 §5」→「7.3 §4」 |
| S3-5 | 8.3 §6.2 L254 | 「= BW*（**式 (4)** 章脊值，2.4 §2 一致）」式号错 | 2.4 的 BW* 是式 (2)（命题 1），式 (3) 是 BW_req；2.4 无式 (4) 给 4.05 | 「式 (4)」→「2.4 式 (2)」 |
| S3-6 | 8.5 §2 L60 | 「这是教学负载（**长输出**）的配比」标签误 | T_out=512 与 8.2 chatbot 同形（非长输出，8.2 agent 行才是 8192）；1:1.5 来自容量闸 sizing + τ=20ms，非长输出 | 删「长输出」或改「容量闸主导的教学负载」 |
| S3-7 | references.md 被引章节字段 | W5 回写不全（18 对）：sglang-v04-blog+8.1、v41-flash-config+8.1/8.2/8.3、nvidia-datasheet+8.1/8.2/8.3、deepseekv3+8.2、dynamo+8.2/8.5、splitwise+8.3、distserve+8.3/8.4、mooncake+8.4、memserve+8.5、sglang-repo+8.5、vllm-repo+8.5、ascend-950-keynote+8.2 | 逐章参考节 key 扫描 vs 条目被引字段（python 比对）；W5 精炼说明已声明「P8 被引由 closeout 回写」，尚未完成 | closeout 按实际引用补全 18 对 |
| S3-8 | references.md `nixl` 条目 | 「src/plugins/ **19** 插件」计数错 | fresh codeload tarball cb66ce83 + contents API 双核：src/plugins/ 下 **18** 个目录（azure_blob/cuda_gds/doca_memos/gpudirect_tcpxo/gpunetio/gusli/hf3fs/infinia/libfabric/mooncake/obj/posix/redis/rocm_ais/telemetry/tracing/uccl/ucx）+ meson.build。research notes 自列亦 18 名。8.3 正文正确写 18 | refs「19」→「18」；research notes 总口径「19」同步更正 |
| S3-9 | 8.2 §4 L139 vs 8.3 §6.2 L234 | FTL 术语两章取义不同未互链 | 论文 Table 1：FTL=「Latency to run prefill and generate the first token」（≈TTFT）；§5.1：FTL=「time required for prefill computation to complete」（纯计算）。8.2 取 Table 1「TTFT 同义异名」，8.3 取 §5.1「计算完成时间（不含排队）」——均有所本但读者跨章见两义 | 8.2 术语对照加一句「带宽公式中 FTL 取 prefill 计算时间口径，见 8.3 §6.2」 |

## 2. Writer 上报待核项裁定（独立复核，不照单全收）

| 项 | 裁定 | 证据 |
| --- | --- | --- |
| **NIXL 插件数 18 vs 19** | **18 正确**。8.3 正文「18 个插件目录」对；refs/research notes 的「19」是计数错（自列 18 名）。 | fresh tarball cb66ce83 解压 `ls src/plugins/` = 18 目录 + meson.build；contents API `src/plugins?ref=cb66ce836f82` 同 18。见 S3-8（refs 收窄） |
| **τ_lat 15 μs（8.1）vs 5 μs（5.3）** | **接受为已声明教学值，不强制统一**。8.1 显式注「教学值 10–20 μs，本章取 15 μs——5.3 取 5 μs，两者都是量级示意非实测，实测依硬件与实现归 cluster」；5.3 章脊亦声明「教学值 5 μs」。两处均 R2-7 口径注到位、跨章差异已点破。15 μs 近 NCCL ring 小消息（2(t−1) 跳），5 μs 近 custom allreduce，各自合理。 | 8.1 L62 / 5.3 L55/L101。无 finding |
| **Beyond-the-Buzz bytes_element 歧义 + MHA/GQA 缺 K/V 系数 → κ 统一替换** | **忠实**。论文 §5.1 逐字「bytes_element indicates the number of KV cache bytes per token」——与公式中 d_head×N_kv_heads×bytes_element 的逐元素读法确有歧义（且 MHA/GQA 形缺 K/V 两份系数）。8.3 §6.2 逐字引该定义句 + 显式声明「本书以 κ 为准」「两种写法的几何因子口径不同，勿混」——是声明式口径替换，非篡改论文论断。归一「只计实际分片 KV 的 GPU」逐字命中（「GPU count…that uniquely shards the KV cache」+ TP 超 KV head 复制注）。 | buzz-fresh.txt §5.1 Eq.(1)(2) 定义段。无 finding（FTL 取义见 S3-9） |
| **TRT-LLM disagg 漂移（当前 Python V2 + NIXL，C++ 1.4 弃用）** | **8.4 用当前形态，正确**。L169 弃用引文逐字命中；backend 无默认/unset 拒 disagg（L162）、bounce 默认 0、timeout 60000、max_chunk 32 MiB、snowflake [1<<40,2^63)、coordinator fleet SO_REUSEPORT /select /finish /renew、ridge/ridge_gbm shorter-first、overlap、cache layout transformation TP2→PP2 全 fresh 核中。 | trtllm-disagg-fresh.md L162/166/169/174/176/182/184/103/305-333；blog05 fresh L210/221/241/243/263/186-189。无 finding（bounce「均/平均」见 S2-4） |
| **LoongServe O(bsh/d) 原记号 + 翻译注（§4.3 兄弟字母风险）** | **接受**。§4.3 翻译硬规则约束的是 models/training 兄弟维度字母；LoongServe 是外部论文，8.1 以「论文 O(bsh/d) 口径，其 b/s/h=batch/序列长/hidden、d=DoP；译到本维度为 B·T·d_model/c 元素量级」显式标注原记号 + 即时翻译，O(blsh/d) 同（l=层数）。与 training 5.4 metadata 翻译先例同构，d=DoP 局部声明，未沿用兄弟语义。SIB 式 (7) 的 α/γ 亦以文字 recap 避占用。 | loong-fresh.txt §4.1/§4.2/§5；8.1 L191/L194。无 finding |
| **8.4 CJK 3879 近 3500 下限（对照章两层纪律）** | **接受**。≥3500 达标；内容完整（六轴对照总表 + 选型论断表 + Dynamo 成本模型命题 1/2 + TRT-LLM 漂移坑 + MemServe 机制 + release 边界论断）；6.3/7.4/5.3/5.4 已走读内容一律 recap+指针，增量限定对照表与选型论断，两层纪律守住。 | cjk2.py 计数 3879；8.4 §1/§2/§3/§6 recap 结构。无 finding |

## 3. 强制重推导（独立重推，python 复核）

### 8.1 TP decode 延迟账 — PASS
- 命题 1（每层 2 次前向归约 ×L，非 1 非 4）：推理无反向图 ⟹ f 反向归约不触发，前向仅 g×2（attention out-proj + MLP/MoE down-proj）⟹ 每层 2、每步 2L。V4.1 L=40→80。边界（SP 换原语不换位点、vocab embedding 另 +1）与 training 5.2 §2 / 5.1 §9 一致。✓
- 式 (1) M_msg=2·B·d_model：B=60→614,400 B；B=1→10,240 B=10 KiB；prefill T=4096→41.9 MB（68×）。✓
- 式 (2) τ_TP=2L(τ_lat+2(t−1)/t·M/BW)：t=8 ring 系数 1.75，带宽项 1.75×614400/9e11=1.19 μs，每次 16.19 μs，每步 80×16.19=1295.6 μs≈1.30 ms；带宽占比 7.4%。B=1：0.02 μs、1201.6 μs、降幅 7.25%。prefill T=4096：81.6 μs、7.72 ms（带宽主导）。跨节点 IB 50 GB/s：21.5 μs、下界 2.92 ms。TPOT 占比 13–26%（5–10 ms）/4.3%（30 ms）。✓ 全部命中。
- τ_lat=15 μs 教学值（见 §2 裁定）。

### 8.1 DP attention 通信账 — PASS
- 命题 3（d=e 边界对齐）：a=1 每 rank 独立算 B/d 请求、attention 0 次；d=e ⟹ home rank=dispatch 源=combine 目的 ⟹ 边界 0 次额外通信；MoE 块 dispatch+combine 2 次 a2a。MoE-TP 形边界 AG+RS=1 AR 等价（training 4.1 式 (2)/5.2 §2 SP 恒等）。✓
- 式 (4) C_EP=2(B/d)k·d_model·bytes_act·L_moe=2×8×6×5120×2×40=39,321,600≈39.3 MB；式 (5) C_MoE-TP=2(d−1)/d·B·d_model·bytes·L_moe=1.75×64×5120×2×40=45,875,200≈45.9 MB；纯 TP=2×1.75×655360×40=91,750,400≈91.8 MB。比 91.8:45.9:39.3。✓
- 暴露延迟：纯 TP 80×16.27=1.30 ms；DP+MoE-TP 40×16.27=0.65 ms；DP+EP 80×15.48=1.24 ms（未重叠上界，次数同纯 TP）。EP 赢在字节小 + 可重叠 + KV 读/容量，非次数。✓
- 命题 2 KV 读：TP 每 rank BκLT=64×1024×40×4096=10.74 GB（3.21 ms@3.35 TB/s）；DP 每 rank /8=1.34 GB（0.40 ms）→8×；PP p=2/m=2 每 stage (B/m)κ(L/p)T=2.5 GiB（TP 1/4）。T=131072 全量 320 GiB，TP 复制超 HBM、DP 40 GiB 可容。✓
- **相邻 S2-3**：§4 EP a2a 带宽项公式-代入不一致（4.3 vs 0.61 μs），不在本两条强制项本体，延迟主导结论稳健。

### 8.2 DistServe goodput 调和分解 — PASS
- goodput 定义 recap 1.1 式 (3)：per-GPU 归一 + attainment 约束下最大请求率（非达标裸吞吐），坑注到位。✓
- 命题 2 稳态 R_sys=min(n_P g_P, n_D g_D)、G=R_sys/(n_P+n_D)；连续最优平衡 n_P g_P=n_D g_D（交换论证）。✓
- 式 (3) G*=g_P g_D/(g_P+g_D)：5.6×10/15.6=3.59 rps/GPU。整数 replication ⌈10/5.6⌉=2P+⌈10/10⌉=1D→R=10、10/3=3.33（取整损失 7%）、3.33/1.6=2.1×。1P2D→min(5.6,20)=5.6、1.87。✓ 全部命中。
- 命题 3 单调性→两级分解（相独立优化合法 ⟺ 高带宽 placement 解耦；Algorithm 2 低节点亲和耦合可行集须联合枚举）。✓ 教学扩展，非论文论断（论文只给数值 + Algorithm 1 ceil 行），代数与边界正确。
- 式 (4) M/D/1 Avg_TTFT=D+λD²/(2(1−λD))（论文 Eq.1 R→λ 翻译）：D=0.1、λ=5→0.15 s、λ=9→0.55 s、排队项 9×。✓
- Algorithm 1/2、二分搜索、O(NM²)、<1.3 min、175B/350GB/640GB、§3.3 1.13GB/90Gbps/800Gbps/600GB/s、pull/replan 全 fresh 逐字核中。✓

### 8.2 xPyD 配比 — PASS
- 式 (5) n_P≥λ·2N_act·T_in/(F_P ρ_P)：6.144e13 FLOPs/req、6.144e14 需求、F_P ρ_P=3.958e14→n_P≥1.55；μ_P=8.05 req/s/GPU。✓
- 式 (6) C=λ T_out TPOT=153.6；式 (7) n_D≥λ T_out κL T̄/(β_D ρ_D)：κL=131072、T̄=4352、5.704e8 B/seq/step、2.92 TB/s 需求、β_D ρ_D=2.68→n_D≥1.09。步时交叉核对 141 并发×24.0 ms KV+4.8 ms 权重=28.8≤30。✓
- 式 (8) 配比 1.55:1.09≈1.4:1→取整 2:2。命题 4 双重无关性（λ 约去、TPOT 经 C∝TPOT 与频率∝1/TPOT 精确相消）+ 三边界反例（权重项/容量约束/ρ 随 λ）。✓ 教学扩展，代数正确。
- 表：agent 32.8（1:21）、MLA chatbot 0.29、MLA agent 8.8、H200 0.76（0.70×）；T_out 16×→n_D 30×（双因子）；MLA κL 0.268×。✓ 全部命中。

### 8.2 传输 vs 重算带宽阈值 — PASS
- 式 (9)(10)(11)：t_xfer=κLT/BW、t_recomp≈2N_act T/F_eff，T 相消 ⟹ BW*=κL F_eff/(2N_act)，**非临界长度**（C-S3-2 坑注到位）。✓
- V3 FP8：κ=576、L=61、κL=35136；F_eff=1979×0.4=791.6 TFLOPS；2N_act=7.4e10；每 token 重算 93.5 μs；BW*=35136/93.5e-6=0.376 GB/s≈3 Gbps。跨 DC 6.25 GB/s=16.6×。T=4096 全量：KV 143.9 MB、传输 23.0 ms vs 重算 382.9 ms、比 16.6。✓ **0.376 GB/s≈3 Gbps 命中**。
- 式 (12) 二次项 BW*(T)=κL F_eff/(2N_act+2LTd_model)：V3 T=32768 分母 1.027e11→0.271 GB/s（72%）。✓
- V4.1：512 元素（448+64，X-1）、FP4 288 B、κL=11520（V3 0.328×）、bf16 40960。✓
- 命题 6 全重叠阈值=BW*（每层 κT vs 2N_act T/(L F_eff)）：每层 KV 2.36 MB、0.38 ms vs 每层计算 6.28 ms。✓（线性口径限定见 S3-2）
- 8.3 命题 2 egress 恒等 BW*：代入 FTL=2N B_P ISL/(n_GPU,P F_eff) 约去 B_P ISL ⟹ κL F_eff/(2N)。章脊 4.05 GB/s（=2.4 式 (2)）；FTL 132.5 ms、聚合 32.4、per-GPU 4.05、6.25/4.05=1.54×。命题 3 ingress ∝1/OSL、ISL 无关（KV 主导 TTL∝ISL）；TTL roofline 2.04 ms、ingress 2.10 GB/s、OSL=64→33.6>6.25。✓ 全部命中。

## 4. 跨章一致性核查

| 项 | 结论 |
| --- | --- |
| BW* 8.2↔8.3↔2.4 同形 | ✓ 三处均 κL F_eff/(2N_act)、T 相消；8.2 式 (11)/8.3 命题 2/2.4 式 (2) 同形；8.3 recap 指针防双展开（8.2 命题 6 立原理、8.3 对 egress 公式）。8.3「式 (4)」式号误→S3-5 |
| V4.1 MLA=512 元素（X-1，非 576） | ✓ 8.1 章脊 448+64=512、κ bf16 1024；8.2 §5 显式「512 元素（448+64，X-1 跨波裁定；V3 576 不适用）」；8.3 FP4 288/bf16 1024。config fresh 核：head_dim 512 含 rope、kv_lora_rank 字段不存在（引擎导出 448）。V3 行 576 不动 |
| serving TP 2× 非 4× | ✓ 8.1 命题 1 + 坑注（C-S3-1）；training 5.1 §6 每层 4 次（前向 g×2+反向 f×2）recap 忠实；5.3 §3「次数减半」结构结论 recap + 全账留 8.1 |
| CP 三义区分 | ✓ 8.1 §6 坑注：训练 ring CP（5.2）≠ LoongServe ESP（每 iteration DoP）≠ DeepInfer balanced CP（README L12 release 支持 vs doc L67 生产不用）；c 仅 LoongServe 对照不入 t/p/e/d |
| 符号 §4.3 | ✓ B 专 batch、b 专 block（8.1 声明不用）、bytes 不占字母、d_model/d_latent/d_rope、goodput 𝒢、n_P/n_D（8.2/8.4/8.5 禁裸 P/D）、λ 到达率声明（8.2/8.5）、α/γ 专归（8.1 译 τ_hop/τ_lat、LoongServe SIB 以文字 recap）。8.5 τ=TPOT SLO 局部声明（与 1.4 采样温度 τ 区分）。LoongServe 原记号见 §2 裁定 |
| 命题逐章编号 | ✓ 各章独立 1..N，无跨章混号 |
| recap 指针 | ✓ 1.1 式 (3)/1.2 式 (1)(3)(4)/1.3 §3/§5/2.4 式 (2)(3)/3.3 式 (6)/4.2 式 (4)/4.3 §1/5.3 §2/§3/§4/§9/6.3 §1 式 (1)(3)/§4 式 (5)/命题 2/§5/7.4 §1/§2/§3 全命中；7.3 §5→§4（S3-4）、2.4 式 (4)→(2)（S3-5）两处式号误 |
| refs 无竖线 / 无混合档 | ✓ P8 五条目 7 字段解析通过（83 条目全 7 列），无 [A/B] 混合档，日期合法 |
| [B] 不推测代码 | ✓ 8.2 Splitwise/Beyond-buzz「按论文讲机制不推测生产代码」、8.4 MemServe「讲论文机制不推测代码」、8.5 MemServe 同；Splitwise 仅模拟器、MemServe 无 repo、Beyond-buzz proprietary 均显式注 |
| 两层走读纪律（8.4） | ✓ 6.3/7.4/5.3/5.4 已走读 recap+指针，增量=对照表+选型论断+Dynamo 新件；9.2 防双展开注（C-S3-4） |
| 长度 | ✓ 8.1/8.2 ≥5500（7668/7361）；8.3/8.4/8.5 ≥3500（4697/3879/5759） |

## 5. Verification log（抽查项 × 来源 × 结论 × 覆盖率）

一手来源全部 evaluator 自行 fetch（/tmp/eval-p8-evidence/）。export.arxiv.org API 本轮限流（Rate exceeded），改用 abs/HTML 通道逐 ID 核实（等效）。

### 5.1 arXiv ID↔标题↔第一作者（8/8）
| ID | 通道 | 标题/作者命中 |
| --- | --- | --- |
| 2401.09670v3 | HTML fresh | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving / Zhong ✓ |
| 2311.18677v2 | HTML fresh | Splitwise: Efficient Generative LLM Inference Using Phase Splitting / Patel ✓ |
| 2407.00079 | abs fresh | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving / Qin ✓ |
| 2404.09526v2 | HTML fresh | LoongServe: Efficiently Serving Long-Context LLMs with Elastic Sequence Parallelism / Wu ✓ |
| 2406.17565v3 | HTML+abs fresh | abs 官方标题 MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool / Hu ✓（HTML 内题「Flexible Mem Pool…」异，refs 用 abs 官方题正确） |
| 2406.03243v1 | HTML fresh | Llumnix: Dynamic Scheduling for LLM Serving / Sun ✓ |
| 2403.02310 | abs fresh | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve / Agrawal ✓ |
| 2506.05508v1 | HTML fresh | Beyond the Buzz: A Pragmatic Take on Inference Disaggregation / Mitra ✓ |

### 5.2 DistServe（8.2 §1/§2/§6）— fresh HTML 逐字
- goodput 定义「maximize per-GPU goodput, defined as the maximum request rate…adhering to the SLO attainment goal (say, 90%) for each GPU provisioned」✓
- 1.6 rps colocate「constrained by the more stringent one of TTFT and TPOT requirements」✓；5.6 prefill/10 decode；2P1D→10 rps=3.3/GPU=2.1× ✓；摘要 7.4×/12.6× >90% ✓
- Eq.1 M/D/1 D+RD²/(2(1−RD)) ✓；Eq.2 inter D+RD²/(4(2−RD)) ✓；Eq.3 intra D/K+RD²/(2K(K−RD))、1<K<2 ✓
- Algorithm 1：config.goodput/config.num_gpus 目标、simu_prefill/decode 二分、n,m=⌈R/config_p.goodput⌉,⌈R/config_d.goodput⌉ ✓；Algorithm 2：inter_op∈{1..N}、get_intra_node_configs、P_p.num_gpus+P_d.num_gpus≤M ✓
- 复杂度 O(NM²)（N=node limit per instance、M=8）✓；<1.3 min（§6.5）✓；simulator trace 拟合+重采样（§4.1）✓；§6.4 profiling 吻合 ✓
- §3.3 OPT-66B 512-token 1.13GB、10 rps→11.3GB/s=90Gbps、IB 800Gbps、A100 NVLink 600GB/s ✓；§4.2 175B(350GB)、8×80G=640G<350×2 ✓
- §4.3 FCFS+shortest-queue dispatch、pull（decode fetch、prefill HBM 当排队缓冲）、replan 秒级/权重分钟级 ✓；§2.3「Ineffective scheduling」「trades TTFT for TPOT and cannot eliminate the interference」✓；250 words/min ✓
- 官方 code LLMServe/DistServe README=论文摘要 + SwiftTransformer C++ backend ✓（[A] 维持）

### 5.3 Splitwise（8.2 §3 / 8.3 §1）— fresh HTML 逐字
- Table I H100 3.43× compute/1.75× power vs A100 ✓；摘要 1.4×@20% 低成本/2.35×@同 cost+power ✓
- §VI-A 第二 token 16.5% vs serialized 64% ✓；E2E 3%→0.8% ✓；常数尾巴 A100~8ms/H100~5ms（200 vs 400 Gbps）✓；<512 token（H100）serialized/大 layer-wise ✓
- §IV-A 三池 prompt/token/mixed、mixed 机保留身份队列清空回原池、粗粒度 re-purposing 不频繁、CLS 同时指派重叠 KV 传输与 prompt 计算 ✓
- §IV-C serialized「starts only after the prompt phase has finished and the first token is generated」+ 逐层异步 ✓；§II InfiniBand 25-50 GBps per GPU pair ✓
- Azure trace 2023-11-11 两服务 code/conv、AzurePublicDataset 释放子集 ✓；coding 单 token batch >20% 时间 ✓
- 官方 code 仅 mutinifni/splitwise-sim 模拟器 + vLLM PR（[B] 维持，8.2/8.3 不推测代码）✓

### 5.4 LoongServe（8.1 §6）— fresh HTML 逐字
- ESP 每 iteration 动态 DoP、elastic instance=模型副本+统一模型并行、ESP 组跨 instance ✓
- §4.1 proactive migration：prefill SP ring 流通 KV 选择性留新组、零额外迁移、buffer O(bsh/d) 单层（<纯模型并行 O(bsh)）✓；reactive O(blsh/d) 碎片约束（b/l/s/h=batch/层/序列/hidden）✓；1M-token 7B KV>488GB 数秒迁移 ✓；600K DoP=3 三 instance 100k/200k/400k slot 首 instance OOM ✓
- §4.2 multi-master distributed decoding：master 持本请求 KV+发 Q 并行 local attention、FFN 跨 master、Q 交换重叠 ✓
- §2.4 改 TP 度重启 runtime 分钟级 ✓；§5 SIB α_p+β_p 线性+γ_p 二次最小二乘、DP batching prefix sum 多项式 ✓；decode DoP=启动最小 best DoP（模型并行度）、运行时缩到 KV 装得下最小 ✓
- §7 3.85× vs chunked prefill（LightLLM SplitFuse/DeepSpeed-MII）、5.81× vs DistServe（4 prefill GPU+4 decode GPU、DoP 4）、4.64× vs vLLM ✓
- 官方 code LoongServe/LoongServe README 自认论文实现+artifact-eval ✓（[A] 维持）

### 5.5 MemServe（8.4 §7 / 8.5 §7）— fresh HTML 逐字
- §7「MemPool and global scheduler from scratch, 5.6K SLOC in Python and **1.6K SLOC in C++**」✓（**chapter 的 1.6K C++ 在论文中，research notes/refs 漏列**）；改 vLLM 200 Python+400 CUDA C++ ✓；NCCL send/recv GPU-GPU + socket（任一侧 DRAM）✓
- §8 ShareGPT disagg JCT 至多 42%（P99）/+caching 29% ✓；LooGLE 10.8%/26.9% ✓
- §4 MemPool 三 API（memory/indexing/distributed transfer）、fixed-size allocator、transfer 三步 allocation→transmission→insertion、指定 dst addr 跳 allocation→layer-by-layer、transfer_with_insert ✓；Table 1 alloc_mem（type HBM/DRAM/mixed、@id、addrList 编码 instance ID、transfer flag）✓
- §5 context caching+disagg 首次结合四步（含 decode-to-prefill）、huge pages 共优 ✓；§6 global prompt tree locality-aware best-effort ✓；§4.4 CM heartbeat+释放故障 instance memory block ✓
- 无官方 repo（[B] 维持，8.4/8.5 不推测代码）✓

### 5.6 Llumnix（8.5 §4/§7）— fresh HTML 逐字
- §6.2 downtime 近常数 20-30 ms（短于单 decode step）、baseline 至多 111×、8k LLaMA-30B recompute 3.5s≈54 decode steps ✓
- §5 Gloo Send/Recv（NCCL 并发 unsafe）、failed actors 由 Ray 自动重启、故障 instance 请求 abort ✓
- §4.4 virtual usage 统一四场景（load balancing/de-fragmentation/prioritization/drain-saturate auto-scaling）、freeness 可负 ✓；§4.3 global scheduler+llumlet ✓
- §6.5 freeness 阈值 [10,60]、同激进度省 16%、同 P99 prefill latency（~5s）省 36% ✓；摘要尾延迟一个数量级/高优先级 1.5× ✓
- 官方 code llumnix-ray v0 README 自链 2406.03243+OSDI'24 ✓；代码锚 MigrationCoordinator L228/migrate_out L251/MigrationStatus L42、RayRpcMigrationBackend L62/RayColMigrationBackend L220/gloo L260/pygloo bf16→glooFloat16 L212-215、Scaler L84 ✓（[A] 维持，v0 锚）

### 5.7 Beyond-the-Buzz（8.2 §4/§6 / 8.3 §6.2）— fresh HTML 逐字
- §5.1 Eq.(1) BW_egress=(N_layers×BS_prefill×ISL×d_head×N_kv_heads×bytes_element)/(FTL×NumGPU_prefill)、Eq.(2) ingress 同分子/(TTL×OSL×NumGPU_decode) ✓
- bytes_element 定义「indicates the number of KV cache bytes per token」✓（歧义实存，8.3 κ 替换声明式）；FTL「time required for prefill computation to complete」✓；NumGPU「GPU count…that uniquely shards the KV cache」+ TP 超 KV head 复制注 ✓
- egress 随 ISL 降（FTL 超线性）/ingress KV 与 TTL 同∝ISL 相消、∝1/OSL ✓；大模型 MLA egress 可能更低 ✓；现有 DC 带宽足以支撑 ✓；逐层即传重叠假设 ✓
- §3.2 rate matching integer solver（Appendix B、TTL 约束+总 GPU 最小化）✓；§4.3 ratio 3.5 relaxed/0.5 tight、固定比退化 ✓；§4.2 prefill-heavy（ISL≫OSL）+>10B 最有效 ✓；§4.1 MLA down/up projection 冗余、缓存 up-projected KV 缓解、chunking 利 relaxed+generation-heavy ✓
- Table 1 FTL=「Latency to run prefill and generate the first token」/TTL=「Latency to generate each new token」✓（FTL 两义见 S3-9）
- 无官方 repo（proprietary simulator，[B] 维持）✓

### 5.8 Mooncake（8.2 §3 / 8.3 §4）— fresh abs + contents API
- 摘要 525% 模拟吞吐/75% more requests（Kimi 生产）、KVCache-centric、闲置 CPU/DRAM/SSD、prediction-based early rejection ✓
- transfer_engine.h：TransferEngine L90/openSegment L156/registerLocalMemory L187/submitTransfer L205/submitTransferWithNotify L258/allocateBatchID L287/getTransferStatus L305/getTaskCongestionState L308/getBatchTransferStatus L314 ✓
- topology.h：TopologyEntry L38（preferred_hca L40）/Topology L64/discover L74/selectDevice L109（storage_type+retry）/getMatrix L126/matrix_ L134 ✓
- transport.h：TransferRequest L66/OpCode{READ,WRITE} L67 ✓；Transport 类 Tcp/Rdma/Efa/NVMeoF/Nvlink/IntraNodeNvlink/Hip/Shm 全在树 ✓
- index.md 678 行：每进程一个 RAM Segment（名 local_hostname）+Buffer（rkey+NIC 亲和）、BatchTransfer「async flexible AllScatter/AllGather」、topology matrix 广播+preferred/secondary、>64KB 切 slice 多路径全 NIC 协同、SIEVE 驱逐+waiting_list_ ~1Hz monitorWorker、fault handling 替代路径重提交 ✓

### 5.9 NIXL（8.3 §3）— fresh codeload tarball cb66ce83 + contents API
- **src/plugins/ = 18 目录**（双通道核）✓（S3-8）
- nixl.h：nixlAgent L35/registerMem L118/deregisterMem L131/prepXferDlist L184/makeXferReq L257/createXferReq L306/postXferReq L345/getXferStatus L355/getLocalMD L512/loadRemoteMD L543/invalidateRemoteMD L555 ✓
- nixl_descriptors.h：nixlBasicDesc L33（addr uintptr_t+len size_t+devId uint64_t）/nixlStrideDesc L168（@class，struct L174，count 等长块 stride==len 连续）/nixlBlobDesc L226（继承 BasicDesc+metaInfo）/nixlDescList L343/nixl_xfer_dlist_t L547=DescList<BasicDesc>/nixl_reg_dlist_t L559=DescList<BlobDesc> ✓
- nixl_types.h：nixl_xfer_op_t{NIXL_READ,NIXL_WRITE} L48、nixl_blob_t=std::string L112、VRAM_SEG ✓
- createXferReq(operation,local_descs,remote_descs,remote_agent,req_hndl,extra_params) 签名 ✓；examples/cpp/nixl_example.cpp 调用序列 ✓（教学片段签名匹配）
- docs/nixl.md：conductor 进程、agent 管多 GPU+全局唯一 name、Metadata Handler side-channel/etcd/Redis 控制路径、加 agent=交换 metadata/删=disconnect+purge、数据面 RoCE/IB/GDR/NVMe-oF/TCP/NVLink、backend 按内存类型+共有 backend 选 ✓

### 5.10 TRT-LLM（8.4 §5）— fresh contents API
- disagg-serving.md L169 C++ 1.4（2026-11）弃用+3 月迁移期+Python V2+NIXL+KV manager V2 逐字 ✓；L162 backend 无默认/unset 拒 disagg ✓；L166 timeout 60000 ✓；L174 V2 须 use_kv_cache_manager_v2:true/V1 报错 ✓；L176 bounce 默认 0 ✓；L182「averaging at most 16 KiB」（S2-4）；L184 max_chunk 32 MiB ✓；L103 snowflake [1<<40,2^63)/[0,1<<40) ✓；L65-67 overlap ✓；L76-78 cache layout transformation TP2→PP2 ✓；L305-333 coordinator fleet SO_REUSEPORT /select /finish /renew ZMQ ridge/ridge_gbm shorter-first ✓
- blog05：MTP off 1.4-1.8×（低并发）/MTP on 1.6-2.5×（平均高 20-30%）/GEN4 1.73×/GEN8 2×/50 tok/s/user 1.7×/E2E vs SOL 0-25%（分数比+无 KV 传输假设）/rate-matching 四步 L186-189 ✓
- IFB doc paged-attention-ifb-scheduler.md（9.2 出处，本轮登记）✓

### 5.11 Dynamo（8.4 §4 / 8.5 §3）— fresh contents API
- README L36「the orchestration layer above inference engines — it doesn't replace SGLang, TensorRT-LLM, or vLLM…」逐字 ✓
- disagg-serving.md：PrefillRouter 四步、NIXL VRAM→VRAM 直传最优 transport、backend 元数据 SGLang bootstrap_info（host/port/room_id，prefill background/decode 立即+KV 并行）/vLLM kv_transfer_params（block IDs+remote，prefill 同步 decode 等）/TRTLLM opaque_state ✓；runtime xPyD（add register+RuntimeConfig KV capacity/remove drain+deregister/discovery 自动）✓；Capacity effects 三句（低并发 aggregated/持续负载隔离/decode KV 容量瓶颈加 P 副本反降容量）✓
- routing-concepts.md L29-35 成本模型逐字：raw=active_prefill+incoming_prompt、adjusted=max(0,raw−overlap_credit)、potential_decode=active_decode+incoming_active、active_request=weight×active_requests、cost=prefill_load_scale×adjusted+potential_decode+active_request ✓；weight 默认 0 ✓；overlap device/host/disk/shared-cache+credit_decay ✓；conditional-disagg 例外式 ✓
- planner README：SLA-driven、throughput-based（traffic prediction+perf model→副本下界保 TTFT/ITL、bootstrap self-benchmark/AI Configurator/profiler/live FPM regression）+load-based（FPM event plane、optimization_target:sla online regression、attention-DP per-rank）、throughput 下界+load 实时 ✓

### 5.12 DeepInfer doc（8.1/8.3/8.4/8.5）— fresh contents API 372 行
- L33 #DP=#EP ✓；L47 prefill EP2-8/DP2-8 最小 ✓；L49 decode EP64-128/DP64-128 Wide-EP 最大 ✓；L53 TP/CP 暂不用 ✓；L57 prefill 最小 rationale ✓；L67 CP 主因 V4.1 足够快+≤5% reasonable tradeoff+worsens EP balance+kernel/comm/CPU 逐字 ✓；L69 TP 装得下+SLA 可达不付通信逐字 ✓；L73 maximize utilization ✓；L75 过程 ✓
- L91 1P1D deployable/In practice ~1:1/64P+64D 最小高效单元逐字 ✓；L97 跨 DC 压缩 KV 可行+可靠性光纤易断+~400Gbps/8GPU+相邻 DC 逐字 ✓；L119 训推共调度逐字 ✓
- L127/L131 EPLB prefill 4/100/1000 moe_lb 1.074(EP4DP4)/1.045(EP2DP2)、decode 1/1000/10000 moe_lb 1.090、7 experts/GPU ✓；L191 Entrypoint CHECK hash-chain+async fetch+GDR swap-in（copy engine 细粒度差/NIC 不占 SM）逐字 ✓
- L251 autoscaling non-cached tokens/running reqs+rate-limited bounded 逐字 ✓；L253 grace 1min 一次性转发逐字 ✓；L257 冗余 quota ready 机池+>0.8 禁 scale-in 逐字 ✓；L261 重启分钟级+quota 冗余+rerouting 逐字 ✓；L265 EP 单通信域单 GPU 故障整实例不可用+重定 EP size 逐字 ✓
- L51 VE 单 GPU cheaper ✓；L293-297 EP>32 递减+整超节点单 EP 组 ✓；L31 PD TTFT/TBT 独立 ✓
- repo：README L12 balanced CP ✓、L38-42 release 边界三项 ✓、CPManager cp.py L9 ✓、engine_type.py PROMPT/COMPLETION L12-13 ✓、MultimodalProcessor L29 ✓

### 5.13 V4.1-Flash config — hf-mirror fresh
- L=40/d_model=5120/H_kv=1/head_dim=512/qk_rope=64/kv_lora_rank 字段不存在/E=384+1 shared/k=6/moe_intermediate 2304/index_topk 512/index_n_heads 32/index_head_dim 128/sliding_window 128/max_position 1048576/quant fp8 block[32,32] ue8m0 expert fp4 ✓；first_k_dense_replace 不存在（8.1 L_moe≤40 上界口径注忠实）✓

### 5.14 vLLM / SGLang 引擎锚点 — W3 同日 tarball + vLLM live API spot-check
- vLLM：KVConnectorBase_V1 L182/KVConnectorRole L140/NixlBaseConnector L79/NixlPullConnector L374/NixlPushConnector L415/MooncakeConnector L798/MultiConnector L142 ✓（live contents API 复核 nixl/connector.py L79/374/415 与 tarball 一致）；connector 目录 nixl/mooncake/mooncake_store/lmcache_connector/hf3fs/multi_connector/hisparse/offloading_connector 全在 ✓；elastic_ep/（elastic_execute/elastic_state/standby_state）✓
- SGLang：base/conn.py KVArgs L62/KVPoll L142/BaseKVManager L150/BaseKVSender L191（send L209/poll L240）/BaseKVReceiver L266（send_metadata L299）/BaseKVBootstrapServer L339 ✓；MooncakeKVManager L364/NixlKVManager L531 ✓；PrefillBootstrapQueue L270/DecodePreallocQueue L471/DecodeTransferQueue L3303 ✓；elastic_ep ScaleCohort L30/update_dp_attention_for_elastic_ep L77/ElasticEPStateManager L132+expert_backup_manager.py ✓；runtime_context derive_attn_tp_size L73 ✓；server_args L355 enable_dp_attention 由 attn_dp_size>1 派生 ✓
- SGLang v0.4 blog（lmsys fresh）：单 KV head TP8 重复 KV→attention DP、MoE 前 all-gather/后 redistribute、8×H100 DeepSeek-Coder-V2-Instruct-FP8 1.9× decode（v0.4 vs v0.3）、--enable-dp-attention 历史 flag ✓

### 5.15 training P5 recap 锚（8.1）
- training 5.1 §6 每层 4 次（前向 g×2+反向 f×2）、式 (9) 2bsh、式 (10) 8bshL·2(t−1)/t、式 (11) ∝(t−1)、IB 50 GB/s、NVLink 900、「难 overlap」、§8 GQA a_kv<t 复制+MLA latent 无处可切、§9 vocab +1、V3 不用 TP ✓
- training 4.1 式 (2) 2(d−1)/d·M、式 (3) 2(d−1)α+2(d−1)/d·Mβ（α/β→τ_hop/1/BW 翻译注）✓
- training 5.2 §2 SP 恒等 AR=RS+AG ✓；5.3 式 (5) (p−1)/(m+p−1) ✓；5.5 式 (3)(4) world=t·c·p·d / t'·p·e·d_e、tcd=t'ed_e ✓；5.4 木桶 max_j R_j ✓

### 5.16 inference 兄弟章 recap 锚
- 1.1 式 (3) goodput+§3 DistServe 数值+练习 3 xPyD ✓；1.2 式 (1)(3)(4)+§3 几何表 V4.1 512/V3 576+§5/§6 ✓；1.3 §3 scale-up（900GB/s/1.8TB/s/NVL72）+§5 M/M/1+Little+Ascend 950PR 1.6/950DT 4 TB/s ✓
- 2.4 式 (2) BW* 4.05+式 (3) BW_req 21.5+§4 GDR 逐字+§5 Mooncake 525/75+§9 三路线+N=8e9 圆整声明 ✓；2.3 §4 288B/3.56×/11520 ✓
- 3.3 §6 式 (6) bytes_dispatch+B_local=64/EP64 1.97MB/30.7KB/0.61μs（per-peer 锚）✓；4.2 式 (4) 10.7+2.7=13.4ms/recompute 133ms ✓；4.3 §1 781/15.1/52×/66.5/4.4×+式 (4) T²/(2c_chunk) ✓
- 5.3 §2 CP 两字段+AsyncIntermediateTensors、§3 τ_lat 5μs+次数减半+延迟主导（全账留 8.1）、§4 三归约路径、§9 connector 基线 ✓；6.3 §1 式 (1)(3)+scheduler_pp_mixin/pp_sync_window、§4 式 (5)+命题 2+暴露计入 TTFT、§5 send_metadata pull 式询问/push 式数据+aux 14.4KB+PREBUILT 不进 forward ✓
- 7.4 §2 18.8MB/expert/层+289GB+pareto 100-200 tok/s/user、§3 EPLB 参数+moe_lb ✓；7.3 §4 528/288/68 格式账（§5 是 DeepGEMM，S3-4）✓；training 7.3 §2.4 SDC ✓

### 5.17 机械检查
- lint-chapter.sh：5/5 LINT OK ✓
- lint-math-boundary.py：5/5 exit 0 ✓（**不覆盖反转义陷阱，S2-1/S2-2 漏网——advisory 补检查**）
- CJK（cjk2.py，strip code/math 后汉字数）：8.1=7668 / 8.2=7361 / 8.3=4697 / 8.4=3879 / 8.5=5759 ✓
- rubric A1/A2/A3 禁令词（由此可见/不难发现/总而言之/接下来我们/本节将/其实/我们不妨/强大的/革命性的/…/呢吧嘛）：0 命中 ✓
- 反转义序列扫描：`\,` 8.2×17、`\!` 8.5×1（S2-1/S2-2）；`\;` `\:` `\{` `\}` `\%` `\#` 0；表外 `\|` 0 ✓
- refs 解析：83 条目全 7 字段、无竖线破损、无混合档、日期合法 ✓；被引回写 18 对缺口（S3-7）

## 6. 覆盖率与未决项

- 强制重推导两条（8.1/8.2）全项独立重推 + 数值 python 复核：覆盖 100%。
- 一手来源：8 论文 + 7 repo/doc 锚点簇 + config + blog，关键论断逐字核：覆盖 spec §5 P8 全部 Sources。
- 教学审查：命题-证明骨架（8.1×3/8.2×6/8.3×3/8.4×2/8.5×5）路线句/边界反例/练习闭环逐查；R2-1 节首（全 ## 节设问/数值/元问题开场）/R2-2 章脊/R2-3 公式后实例/R2-12 总结回指逐查通过；习题难度分布（[10]/[20]/[30]/[40] 非递减、≤1 [40] 带脚手架、≥1 命题目标题、校验锚）5/5 合规。
- 规范审查：STYLE_GUIDE §1-§11 + R2-1…12 + rubric A/B 逐条；历史叙事 0、术语首现定义 ✓、证据等级显式 ✓。
- **未决（非本批 finding）**：W5 场景图尚未产（progress.md 管线「修复→产图入 QA」在审查后），commit gate 前须补图 + 图 QA（spec §6.6）；refs 回写（S3-7）与 nixl 计数（S3-8）closeout 执行。
- 含 S1 章须复审——本批 S1=0，无强制复审；S2 修复后建议 fix-review 复核 S2-1/2（渲染）、S2-3（链路模型）、S2-5（命题方向）、S2-6（TTFT 口径）联动。

## 7. 计数

**S1 = 0；S2 = 6（S2-1…S2-6）；S3 = 9（S3-1…S3-9）。**
强制重推导：8.1 PASS / 8.2 PASS。
Writer 待核项裁定：NIXL=18（refs 收窄）/ τ_lat 接受 / Beyond-buzz κ 替换忠实 / TRT-LLM 当前形态正确 / LoongServe 原记号接受 / 8.4 篇幅接受。