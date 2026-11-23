# inference W2 fix round 复审（fresh-context re-review）

日期：2026-11-23。审查人：fresh-context re-reviewer（不继承写作与首轮审查上下文）。只读审查——未改章节、未 commit、未派 subagent；本报告为唯一写入文件。
输入：docs/reviews/inference-w2-review-p3.md（S2=9/S3=14）+ inference-w2-review-p4.md（S2=4/S3=7）；.superpowers/sdd/inference-w2/writer-{A,B,C,D,E,F}-report.md fix round 节；9 章修后全文；spec §4.3/§4.4/§6.3/§6.4；inference/references.md（W2 精炼版）；STYLE_GUIDE §3/§3.1；/tmp/iw2-evidence + /tmp/iw1-evidence + /tmp/eval-p3-deepep-v1 存档。
方法：13 条 S2 逐条 verdict + 改动数学全部独立重推（python3 -I，/tmp/fixreview-w2-scripts/{check,math,math2}.py）；21 条 S3 抽查；9 章 lint + 反转义/flanking 机械扫描 + 禁令词 + 篇幅 + 命题/式编号 + 旧值残留 grep；跨章符号/指针一致性；改动涉及的一手来源逐字复核。

## 0. 结论速览

- **S2 = 13/13 ADDRESSED，0 NOT ADDRESSED**。S3 = 21/21 抽查全部已修。
- **强制重推导（改动数学）全 PASS**：3.1 命题 4 两情形上界 + 448 反例（30000 次随机穷举 + 构造反例）；3.1 式 (4) 量纲与 451.5 KiB/0.086%；3.2 对称 /8 约定（0.6→0.625 err 0.025）+ 非对称 −z/clamp（z=8 全值域）+ SmoothQuant 2.56→25.6→256；3.3 分布无关 w=1−μ/M_blk（5 组分布穷举）+ 64.5%/75%/11.0%；4.2 Llumnix 133/13.4=9.9≈10×（+49× 尾段口径）；4.4 存在量词 iff（19×19 网格 + 固定 γ 判据网格）+ 反例 0.885/1.06 + MSS 采样假设（200k 蒙特卡洛保分布、确定性反例 P(c)=0）；4.5 EAGLE-1 倒数第二层对论文摘要逐字。
- **新破坏：无 S1/S2**。1 处 fix 引入的 S3 级外观瑕疵（3.1 两处英中紧贴空格缺失）+ 3 处既有 minor 观察项（均首轮已 PASS、本轮未触，非 fix 破坏），见 §3。
- lint 9/9 LINT OK；反转义扫描（`\!` `\_` `\,` `\;` `\{` `\|` `\%` `\#` `\gtrsim`）9 章全 0；abs_max 残留 0；旧值（0.143/0.571/0.029/66.5%/12×/末层 feature/批不变）全 0 命中；篇幅 4 重章 ≥5500、5 标准章 ≥3500 全达标；命题编号逐章重启不变、式编号不变、练习难度分布不变。
- 含 S2 章（3.1/3.2/3.3/4.2/4.4/4.5）scoped 复审通过；按 spec §6.4 可进 commit gate（图 QA 另按 §6.6）。

## 1. S2 逐条 verdict（13 条，全 ADDRESSED）

| # | 章 | verdict | 修后位置 | 独立重推 / 核验输出 |
| --- | --- | --- | --- | --- |
| S2-1 | 3.2 | **ADDRESSED** | 02-weight-quantization.md L37（式 (1)）、L40（约定声明）、L46（数值例）、L272（LLM.int8 变体注）、L351（练习 2） | 保留 AWQ 约定 Δ=absmax/2^{q-1}（awq.txt L334 逐字 Δ=max(\|w\|)/2^{N−1} 核实）；格点声明 [-2^{q-1},2^{q-1}-1]（INT8 [-128,127]/INT4 [-8,7]）；例 Δ=1.0/8=0.125、0.6/0.125=4.8→Round 5→0.625、err 0.025 ✓；/7 混用消除（全文件 grep 仅剩 L272 声明的 2^{q-1}-1 变体与 L319「表 6/7」表号）；练习 2 INT3 Δ=absmax/4 同约定，WX=5/Q(W)X=6/scaled 4.75/Δ'=Δ 复算 ✓；§6 L272 注「Δ=absmax/127，2^{q-1}-1 约定，与式 (1) 差一格」✓ |
| S2-2 | 3.2 | **ADDRESSED** | L43（式 (2)）、L46（z/clamp 符号表） | Q(x)=Δ·(clamp(Round(x/Δ)+z,0,2^q−1)−z)、z=Round(−x_min/Δ) ✓；与 GPTQ repo quant.py L9-10 `clamp(round(x/scale)+zero,0,maxq); scale*(q-zero)` 同构核实；z=8 例独立重推（x_min=−1,x_max=1,q=4）：Δ=2/15、z=Round(7.5)=8；x=0→0、x=0.6→4Δ=0.5333（round-half-even）、x=1→clamp 15→7Δ=0.9333、x=−1→clamp 0→−8Δ；值域 [−8Δ,7Δ]，端点 z 舍入溢出一格内（标准行为）✓；旧式 Δz 偏移错误消除 |
| S2-3 | 3.2 | **ADDRESSED** | L254（级数账 + 跨通道辨析）、L355（练习 4(c) 锚）、L342（总结行） | 正常通道 m_X=m_W=1 → s=1、迁移后 max 仍 1；outlier 100→10；per-tensor max 100→10；级数 256×1/100=2.56 → 256×1/10=25.6（10×）→ α=1 全压平 max 1 满级 256 ✓ 逐档重推；「统一到 10 不成立」辨析与命题 4 (m_X m_W)^{1−α} 一致；原文 smoothquant.txt L307 只声称对应通道共享相似 max（逐字核），α=1 全压平对应原文 s_j=max\|X_j\| 直选 ✓；练习锚 25.6 同步 ✓ |
| S2-4 | 3.2 | **ADDRESSED** | 式 (1)(2)(10) L37/43/195 | 全文件 `\!` 计数 0（/usr/bin/grep -F）；三处均 \mathrm{Round}\left( ✓；lint 含反转义检查 LINT OK |
| S2-5 | 3.2 | **ADDRESSED** | 全章 absmax 记法 | 全文件 `\_` 计数 0、abs_max 计数 0；式 (1) 符号表注改「absmax(x)=组内最大绝对值（即 max\vert x\vert）」语义清晰 ✓；9 章连带扫描全 0 |
| S2-6 | 3.1 | **ADDRESSED** | 01-decode-attention-kernels.md L289（命题）、L291（证明）、L293（实现口径注）、L295（坑注）、L321（总结）、L285（式 (9) 路线句） | 命题加「设 token 分数两两不同（同分时 top-K 集依赖 tie-breaking，结论另议）」——前提必要性独立构造验证（K=2、分数 5/3/3/1 四块、τ=3、K_b=2 时并列块 tie-break 可挤掉 top-2 token）；第二 clause「恒 ≤ K−K_b 且存在输入恰达」✓；两情形上界证明独立重推：情形 1（≥τ 块 <K_b）含 top-K 块全入选→损失 0；情形 2（≥τ 块 ≥K_b）入选 K_b 块各含 ≥1 top-K token→损失 ≤K−K_b；K_b≥K 时 ≥τ 块 ≤K≤K_b 全收→精确。**30000 次随机穷举（互异分数、bc∈{2,4}）：K_b≥K 精确 0 违例、K_b<K 上界 0 违例、取等可达**；反例 K=512 分散 512 块（T≥8192）、K_b=64 → 恰 448=512−64 ✓（构造实例复算）；强制名额 K_b≥K+1 收紧对 candidate_blocks.py L20 docstring「The last valid block gets +inf」逐字核 ✓；式 (9) 路线句「doc 分析设元素两两不同，同分不恶化界」对 DeepSelect-deep-dive L55 逐字 ✓ |
| S2-7 | 3.3 | **ADDRESSED** | 03-quantized-gemm-and-moe-kernels.md L91-93（命题 2(b) 扩陈述 + 分布无关证明）、L95（μ=32 实例）、L192（总结）、L203（练习 2） | μ=32 行删 66.5%，改「每段非空且 ≤M_blk ⇒ 恰占一整块，w=1−32/128=75% 精确、与余数分布无关」；命题 2(b) 扩为逐段陈述 + 证明（ceil=1 → Σpad/Σ128=1−μ/128）独立重推 ✓；**5 组不同分布穷举（[32×4]/[1,1,1,125]/[32,17,128,5]/[128,1,1,1]/[100,2,50,30]）w=1−μ/128 全成立**；layout 图四段 [32,17,128,5] μ=45.5 → 330/512=64.45%≈64.5% 与图一致 ✓；均匀余数均值 63.5>32 不相容声明 ✓；均值式限定 μ≳M_blk；μ=512 → 63.5/575.5=11.03%≈11.0% 保留 ✓；空 expert（Bk<E）pad=0 另计 ✓；练习 2(b) 加适用条件 + 不适用小问、锚改 μ=512 ✓ |
| S2-8 | 3.1 | **ADDRESSED** | L81（式 (4)）、L84（符号表）、L314（总结） | 删尾部 ·bytes；量纲独立核：2(写+读)×S×H_q×(4 B/元素×d_head + 4 B LSE) = 字节，系数 4 编码 fp32 ✓；2×14×32×516=462336 B=451.5 KiB ✓、η=462336/536870912=0.0861% ✓、S/(4T)=0.003% ✓ 数值不变；符号表注「字节量已编码在系数里，式尾不再乘 bytes 因子」✓ |
| S2-9 | 3.2 | **ADDRESSED** | L174（§4 开场标图 2 口径 + 指针）、L208（口径注块）、L360（参考条双口径） | 口径注：图 2 RTN=43.2 与表 1/2 RTN=23.54 原文即不一致（表 1 act-1%=11.39），各引出处、不混算，「差异来源原文未解释」不推测 ✓；对 awq.txt L192（图 2 caption 43.2/13.0/INT3-g128 逐字）与 L329/357（表 2 s=1: 23.54=RTN）核实，注内容准确。残留 minor：总结表 AWQ 行（L326）未标「图 2 口径」（首轮建议的次要项；口径注 + 开场 + 参考三处已覆盖，不混淆，见 §3 观察项） |
| F1 | 4.4 | **ADDRESSED** | 04-speculative-decoding.md L131（命题 3）、L141（证明结论）、L143（边界辨析 + 反例）、L244（总结） | 「存在 γ 使 F>1 iff α>c」✓ 对 Leviathan Cor 3.9 原文（leviathan.txt L317「If α>c, there exists γ for which we'll get an improvement…at least (1+α)/(1+c)」）逐字核；证明双向独立重推：α>c → k=1 项 α−c>0 → γ=1 满足；α≤c → α^k≤α → 逐项 ≤0 → 任意 γ F≤1 ✓；**19×19 (α,c) 网格 × γ∈[1,60) 存在性 iff 0 违例（tol 1e-9）；固定 γ 判据式 (9) Σ(α^k−c)>0 ⟺ F>1 网格 0 违例**；反例 α=.8/c=.7/γ=4：0.67232/0.76=0.8846<1 ✓、γ=1：1.8/1.7=1.0588>1 ✓、式 (9) 和 −0.4384<0 一致 ✓ |
| F2 | 4.4 | **ADDRESSED** | L195（命题 5 题面）、L197（证明）、L199（边界辨析 + 确定性反例） | 题面显式「第 s 个候选 token 按 x_s~q_s 独立采样（与命题 1 同一采样假设，缺它则结论不成立）」✓；注原文 Algorithm 2 随机序 s~rand(H)（specinfer.txt L508 逐字核）+ 本命题对任意固定序亦成立 ✓；确定性 top-k 反例独立重推：p=(.5,.3,.2)/q=(.4,.4,.2)/候选{a,b}，a 的 p/q=1.25>1 恒接受 → P(a)=1、残差 (0.1,0,0) 不含 c → P(c)=0≠0.2 ✓；**采样版 MSS 200000 次蒙特卡洛（k=2、固定序）经验分布 (0.500,0.301,0.199) ≈ p，保分布 ✓**；greedy 走 VerifyGreedy 声明 ✓；证明代数未动（采样读法下原正确，首轮已核） |
| F3 | 4.5 | **ADDRESSED** | 05-speculative-systems.md L91（§5）、L84（propose 代码注释）；§1 L7 一致 | 「EAGLE-1 用倒数第二层 feature（second-to-top-layer，经 last_hidden_states 传入，参数名泛称勿按字面读成末层）、EAGLE-3 用多层融合（aux_hidden_states）」✓；**对 EAGLE-1 论文摘要逐字核（arxiv-p3p4-batch.xml 2401.15077v3：「autoregression at the feature (second-to-top-layer) level」出现 3 次）+ repo README L34「second-top-layer contextual feature vectors」**；代码注释改「target 隐藏态（EAGLE-1 取倒数第二层）」✓；全 9 章「末层 feature」残留 grep 0 |
| F4 | 4.2 | **ADDRESSED** | 02-scheduler-and-preemption.md L114（§4）、L189（总结） | 「约 10×」+ 显式口径（全量传输 + 停拷 13.4 ms）+ 第二口径（只暴露尾段 2.7 ms → 约 49×）✓；分量独立重推：bulk=512 MiB/50 GB/s=10.74 ms、last=16×64×128 KiB/50 GB/s=2.68 ms、合 13.42 ms；recompute=(2NT+2LT²d)/F_eff=133.1 ms；**133.1/13.42=9.91≈10× ✓、133.1/2.68=49.6≈49× ✓**；旧 12×≈133/10.7（漏停拷）读法消除，全 9 章「12×」残留 0；式 (4) 分量未动 ✓ |

## 2. S3 抽查（21 条，全部已修）

| # | 章 | 位置 | 核验 |
| --- | --- | --- | --- |
| S3-1 | 3.1 | L163 | 「论文 §4.4 parallel-generation 评测口径」✓（flashinfer-paper.txt L75/539 节标题核） |
| S3-2 | 3.2 | L236/244 | s_j 引「原文式 (4)」、迁移等价引「原文式 (3)」✓（smoothquant.txt (3)/(4) 编号逐字核） |
| S3-3 | 3.2 | L232 | 「原文 §3、Fig 3；per-channel 保精度的模拟验证见表 1」✓ |
| S3-4 | 3.1 | L9 | O(T²d_head²/M) + 注「models 4.1 式 (17) 的 d=head 维，按 spec §4.3 译 d_head」✓（models 4.1 L276 d=64、L291 式 (17) 核） |
| S3-5 | 3.1 | L210 | 标「本章账，非 models 4.1 §4.4 原文账」；参数 786432/262144=3×（独立复算 ✓）、decode 逐步投影 2×1536×512 / 2×1536×128=4×（k^C 已缓存口径）✓、score 逐对 ×3 归 models recap ✓ |
| S3-6 | 3.1 | L127 | 「FA2 论文的 seq 维并行沿 query 维；decode q=1 时同构策略是 KV split——FlashDecoding 独立贡献，FA≥2.2 集成，博客口径」✓ |
| S3-7 | 3.4 | L9/41/49/187 | 「15.1/66.5 recap 1.2；4.81 为本章权重-only 下界算例」✓；独立重推 16.1e9/3.35e12=4.806 ms、+KV 0.537 GB→4.966 ms、KV +3.33% ✓；34.3%（权重-only）/≈33%（完整 4.97）双读数 ✓；replay 3.2 ms 与练习 3 锚同步 ✓ |
| S3-8 | 3.3 | L138 | v1.2.1 NVLink 版本注 ✓（DeepEP-1.2.1/README.md L39 News 2025.06.05「leverage NVLink as much as possible」逐字核；单跳直连、不改两阶段对比语义） |
| S3-9 | 3.3 | L138/152 | hook pull 语义「RDMA 后台接收不占 SM、计算侧主动调 hook() 完成接收」✓（README L270-271「the actual tensor will not be received only if you call hook()」「without any SM occupation」逐字核）；SM 账口径分层注（V1 接收侧 vs V2.5 操作整体）✓ |
| S3-10 | 3.1/3.3 | 3.1 L11；3.3 L59 | CTA（cooperative thread array，即 thread block）/grid 首现定义 ✓；3.3 launch 首现「（CPU 提交 kernel 给 GPU，定义见 3.4 §1）」✓（3.4 L7 定义存在） |
| S3-11 | 3.1 | 全章 | 「批不变」残留 0，统一 batch invariance/batch-invariant（术语表 L196「正文保留英文」口径）✓；两处英中紧贴外观瑕疵见 §3 |
| S3-12 | 3.3 | L162/198 | 「以 5 遍为例（教学假设；1.4 的链为 penalty×3 + ban mask，遍数随配置变）」✓（1.4 L103 链核）；49.4/39.5 μs 数值不变 ✓ |
| S3-13 | 3.2 | L224/286 | 式 (13) 最小代入 s_X=4/α=0.5→s=2、α=0→1、α=1→4 ✓；式 (16)(17) 2 特征分解算例 C=10.5（outlier fp16 10 + normal int8 0.5，vector-wise 各自 absmax 落格点精确）✓；不隔离对照 Δ=20/127≈0.157、1→6Δ=0.945、err 0.055 ✓ 独立复算 |
| S3-14 | 3.2 | L164 | 块内更新 W_{:,j:(i+B)} / [H^{-1}]_{j,j:(i+B)} + 注「i=块首列，j:(i+B)=块内剩余列，块局部切片」✓（gptq.txt Algorithm 1 同形） |
| F5 | 4.3 | L65 | 「1.2 §3」✓（短语实在 1.2 L91，属 §3 decode 算例段，§4 为临界 batch——节标题核） |
| F6 | 4.2 | L66 | 「2.4 §2」✓（2.4 §2=「分层命中成本账：什么时候取比重算便宜」，§3=传输带宽需求——节标题核）；因子 2 论断不动 ✓ |
| F7 | 4.2 | L144 | 推论限定「不能立即插队；U 更小者 aging 更快可反超，但反超者被服务后即离队；稳定负载下等待有界、不永久饥饿」✓（按首轮建议；命题 3 (i)(ii)(iii) 与证明未动） |
| F8 | 4.3 | L145 | 「仅解 GEMM 项得 594——略去 chunk 自 attention 的近似，完整式 (2) 解 ≈588，差约 1%」✓；独立重推 9/0.015159=593.7→594；完整二次 2.6493e-7c²+0.015159c−9=0 正根 587.7→588、τ(588)=10.005 ms ✓ |
| F9 | 4.4 | L173/175 | 命题 4 题面补「节点位置编码取路径深度（与独立序列一致，RoPE 相对位置由此确定）」、证明基例引用之 ✓ |
| F10 | 4.1 | L94/122/128/218 | 命题 3 + 式 (4) 口径行 + §4 开场 + 总结四处「纯 decode 口径（忽略新准入 prefill 步；prefill 混入时增益低于 1/η）」✓；证明与数值未动（η=23.4%、4.27×、5.13 ms 复算 ✓） |
| F11 | 4.3 | L59 | 口径注「边际账两边均略 τ_oh；边际 11.2 为两含 oh iteration 之差（oh 相消）、独立基线 15.1 免 oh；含 oh 读数节省 4.9 ms（16.1−11.2）；正文取保守口径、反例同口径」✓；20.0−8.8=11.2、15.1−11.2=3.9、16.1−11.2=4.9 复算 ✓ |

## 3. 新破坏与观察项

**新破坏（fix 引入）：无 S1/S2。** 仅 1 处 S3 级外观瑕疵：

| # | severity | 位置 | 断言 | 处置建议 |
| --- | --- | --- | --- | --- |
| N1 | S3（外观） | 3.1 L3「1.4（batch invariance概念）」、L149「这是batch invariance（1.4）」 | S3-11 术语英文化时就地替换未补空格，英中紧贴与全书「英文术语两侧留空格」排版惯例不一致（无 $ 定界符相邻，渲染无影响，lint 过） | 下波顺手补 2 处空格（「batch invariance 概念」「这是 batch invariance」） |

**既有 minor 观察项（首轮已 PASS、本轮未触，非 fix 破坏，不阻塞）：**

1. **4.4 γ 表 γ=8 格**（L159）：表值 2.41，精确 F=(1−0.8^9)/(0.2×1.8)=2.40495→两位小数应为 2.40（疑双重舍入 2.405→2.41；相对差 0.2%，最优 γ=6 结论不受影响）。P4 逐格重算时已放行；下波可顺手改 2.40。
2. **4.5 L41「EAGLE-1/2…只用 top-layer feature」**：与 §1/§5「倒数第二层」存在表面张力。核来源：此句逐字转述 EAGLE repo README L52（「Considering that top-layer features are limited to next-token prediction, EAGLE-3 replaces them with a fusion…」，章已标「repo README 口径」），而 EAGLE-1 论文摘要用 second-to-top-layer——张力在原文两处措辞本身，章各引出处忠实。可选：下波加一句 reconciling 注（README 的 top-layer 泛指 EAGLE-1/2 的倒数第二层 feature）。
3. **3.2 对称式 (1) 边界**：Δ=absmax/2^{q-1} 约定下 x=absmax → Round=2^{q-1}，码字超出声明格点 [-2^{q-1},2^{q-1}-1] 正侧一格（实际 kernel clamp）。此为 AWQ 原文式 (1) 自身的理想化写法（awq.txt L334 逐字），且首轮 S2-1 建议修正即指定「格点 [-8,7]/[-128,127] + 例 /8」，章忠实执行；练习 2 锚（Q(W)X=6）同此约定、首轮已复算放行。不另立 finding；如求完备可加半句「端点码字越界时实现 clamp」注。
4. **3.3 ≳ 走 prose**（L95/192/203）：\gtrsim 不在白名单，writer 自查改 prose Unicode ≳（首轮 S2-7 建议原文即用 μ≳M_blk）。已发布章无 ≳ 先例但 prose Unicode 算符（→/≈/×）为成例，渲染无风险，flanking 合规。可接受，无需动作。
5. **3.2 总结表 AWQ 行**（L326）：未标「图 2 口径」（S2-9 建议的次要项）；口径注（L208）+ 开场（L174）+ 参考（L360）三处已覆盖，读者对账无混淆。可选下波补标。

## 4. 跨章一致性

- **符号 §4.3**：α（3.2 局部缩放指数声明与 4.4 接受率消歧 ✓）、c（4.3 前缀长章首声明 ✓、4.4 c_d/c_v 无量纲 ✓）、γ/α（4.2 UNIBOOST 译 g/u ✓）、B/b/bytes 不占字母 ✓、E/e/k 大小写 ✓、c_chunk ✓。
- **术语**：batch invariance/batch-invariant 3.1/3.3/3.4 统一英文（术语表口径）✓；「批不变」全 9 章 0 残留。
- **命题/式编号**：逐章重启不变（3.1:1-4 / 3.2:1-4 / 3.3:1-3 / 3.4:1-3 / 4.1:1-3 / 4.2:1-3 / 4.3:1-2 / 4.4:1-5 / 4.5:1-2），式编号不变——跨章式号/命题号指针全部仍有效。
- **量词修复的跨章涟漪**：4.5 §4 L69 与练习 1 引「4.4 命题 3 的 α>c 条件」在存在量词读法下仍自洽（α=.7>c=.5、γ=3 时 F=1.01>1 但近 1，「c 大压低加速」论断不变）✓；4.4 §7 L232 EAGLE「second-to-top-layer feature」与 4.5 §1/§5 修后一致 ✓。
- **双向指针**：3.3 §2 坑 ↔ 3.4 §6 Marlin 绕开（docstring L1201 逐字核）✓；3.3 §4 ↔ 3.4 §3 padding 同构 ✓；3.3 launch → 3.4 §1 定义 ✓；4.3 F5 → 1.2 §3、4.2 F6 → 2.4 §2 节标题实核 ✓；3.1 recap → models 4.1 式 (17) d=head 维实核 ✓。
- **等级**：marlin/specinfer/fastserve [A]、smoothquant 官方 repo（references W2 裁定）使用正确 ✓；DeepEP V1 锚 tag v1.2.1 / V2.5 漂移注分离 ✓。

## 5. 规范 / 篇幅 / 习题

- lint：9/9 LINT OK（scripts/lint-chapter.sh，含 lint-math-boundary.py）。
- 机械扫描：`\!` `\_` `\,` `\;` `\{` `\|` `\%` `\#` `\gtrsim` 全 9 章 0 命中；abs_max 0；旧值 0.143/0.571/0.029/66.5%/12×/末层 feature/批不变 0 命中。
- 禁令词 / 松散语气词：程序扫描 0 命中（含 lint 词表外扩展词表）。
- 篇幅（body CJK，剥围栏/display/行内公式/code span）：3.1=7247、3.2=7603、4.3=5894、4.4=6441（重章 ≥5500 ✓）；3.3=5715、3.4=4634、4.1=4490、4.2=5377、4.5=3823（标准 ≥3500 ✓）。与 writer 报告一致。
- 习题：9 章难度非递减（4.4 [10][20][30][40] 带脚手架，余 [10][20][30][30]）；改动锚（3.2 练 4(c) 25.6、3.3 练 2 μ=512、3.4 练 3 §1 步时、4.3 练 3/4）独立复算可校验 ✓；命题-练习闭环未破坏（3.3 练 2 仍证扩陈述后的命题 2(b)）✓。

## 6. Verification log（抽查项 × 来源 × 结论）

覆盖率：13 条 S2 修后文本逐行读 + 改动数学 100% 独立重推；21 条 S3 逐条定位核验；9 章全文通读；改动涉及的一手来源逐字复核；未改段数值按首轮 PASS 记录抽查复算（~60 项）。估计覆盖率：S2 断言 100%、改动数值 100%、S3 100%、一手来源（改动相关）100%。

### 6.1 独立重推（python3 -I，/tmp/fixreview-w2-scripts/）

| 项 | 输出 | 结论 |
| --- | --- | --- |
| 3.1 命题 4 穷举 | 30000 随机 trial（互异分数）：K_b≥K 精确 0 违例；K_b<K loss≤K−K_b 0 违例；取等可达 | PASS |
| 3.1 命题 4 反例 | 512 top token 分散 512 块、K_b=64 → loss=448=K−K_b；513×16=8208 | PASS |
| 3.1 式 (4) | 462336 B=451.5 KiB、η=0.0861%、S/(4T)=0.003% | PASS |
| 3.2 对称 /8 | Δ=0.125、0.6→0.625、err 0.025；练习 2 WX=5/Q(W)X=6/scaled 4.75/Δ'=Δ | PASS |
| 3.2 非对称 z=8 | Δ=2/15、z=8；x=0→0、0.6→0.5333、1→0.9333（clamp）、−1→−1.0667（clamp）；值域 [−8Δ,7Δ] | PASS |
| 3.2 SmoothQuant | 2.56→25.6→256；s=10/31.6/100；正常通道 s=1 max 1 | PASS |
| 3.2 格式账 / GPTQ toy（未改段抽查） | 0.515625/8.25 MiB/3.88×、8.5/3.76×、3.125%/6.25%、MX 三格式、V4.1 1.00098/0.098%/4×、MSE 0.1832→0.1032/43.7%/δ₂=0.05 | PASS |
| 3.2 LLM.int8 例（S3-13） | C=10.5、Δ=20/127=0.157、1→0.945、err 0.055 | PASS |
| 3.3 padding | 5 组分布 w=1−μ/128 恒等；[32,17,128,5] 330/512=64.45%；μ=32→75%；63.5>32 不相容；μ=512→11.03%；渐近 12.5%；pad 513→127/512→0 | PASS |
| 3.3 未改段抽查 | 4.125 bit/3.88×、B_max 76、A=195.2/e*=32、EP 表 [24.54,12.5,6.72,7.36,8.64] | PASS |
| 4.2 Llumnix | bulk 10.74/last 2.68/total 13.42 ms；133.1/13.42=9.91；133.1/2.68=49.6；旧 12×=133/10.7 读法确认消除 | PASS |
| 4.2 未改段抽查 | β* 8.64/8.07/5.50；HRRN 9.77/2.44/0.98；UNIBOOST Φ_A=−0.073/Φ_B=−0.008、b(100/1000/10000)=0.278/0.075/1.7e-4；FastServe κL=4.72e6/2.42e9/504 步/26 步 | PASS |
| 4.4 量词 | 反例 0.8846/1.0588；式 (9) 和 −0.4384；存在性 iff 19×19 网格 0 违例；固定 γ 判据网格 0 违例；γ 表 7/8 格吻合（γ=8 格 2.40495→2.40 vs 表 2.41，见 §3 观察 1）；最优 γ=6；c=.3→γ3/1.55、α=.5→γ2/1.46；n-gram 1.248；E=3.3616/E[K]=2.3616；Thm3.11 3.79 | PASS（1 既有舍入观察） |
| 4.4 MSS | 确定性反例残差 (0.1,0,0)、P(c)=0；采样版 200k 蒙特卡洛 (0.500,0.301,0.199)≈p | PASS |
| 4.4 §6 抽查 | f(256)=1.51e10、verify ~9.7 ms、带宽 6.09 ms、E=1.4251/182 tok/71.5% | PASS |
| 4.3 F8/F11 | 593.7→594；完整二次根 587.7→588、τ(588)=10.005；11.2/3.9/4.9；B_D 阈 317 | PASS |
| 4.3 未改段抽查 | 权衡表 1338/1037/845/797/785 逐行重算；TPOT 16.1/16.1/20.0/43.4/144.1；漂移 60.9/67.6；守恒 8390656；c_hi 2489；重读 135/1097 GB；284.5 ms；I=4c/74；式 (9) 0.47；quota 1260/499/1012/400/1412 + 回代 49.96/49.92 | PASS |
| 3.4 S3-7 + 未改段 | 4.806/4.966 ms、+3.33%、34.3%/33.2%；0.5 MiB/1.05 MB/0.31 μs/20 μs/0.13%/320 μs/2.1%/16×；33.6/67.1/1.28 ms/1.9%；SiLU 1.84/3.67/1.10/35.1、117.4/234.9/70.1/2.24/3.4%；1.65/10.9%/1.645/3.2；32/24%/128/97%；55/26=2.1、42/26=1.6 | PASS |
| 4.1 未改段抽查 | η 23.4%/4.27×、t_step 5.13、2.62 s/366、0.62 s/~1561、pad 10752/65.6%/0.33 s、池 192/768、QKV I=7 | PASS |
| 4.5 F 值 | EAGLE 2.99、MTP 2.75（2.952/1.075）、D&V 1.01（2.533/2.5）、α.9/c.5 1.38；confidence .30/.24/.21/.06 | PASS |
| 3.1 S3-5 | 786432/262144=3、投影比 4、I_MLA 241.8/ridge 258.2/94% | PASS |

### 6.2 一手来源（改动相关，存档逐字）

| 项 | 来源 | 结论 |
| --- | --- | --- |
| AWQ Δ 约定 | /tmp/iw2-evidence/awq.txt L334 | Q(w)=Δ·Round(w/Δ)、Δ=max(\|w\|)/2^{N−1} 逐字 ✓（S2-1 保留约定忠实原文） |
| AWQ 基线冲突 | awq.txt L192（图 2 caption）/L329/L357（表 2） | 43.2→13.0 INT3-g128 与 s=1: 23.54（RTN）原文即不一致 ✓（S2-9 注准确、不推测） |
| SmoothQuant 式号 / α 语义 | smoothquant.txt (3)/(4) 编号、L307 well-balanced、s_j=max\|X_j\| 直选段 | 式 (3)=迁移、式 (4)=s_j ✓；原文只声称对应通道共享相似 max ✓（S2-3/S3-2） |
| GPTQ 非对称 quantize | /tmp/iw2-evidence/gptq/quant.py L6-10 | clamp(round(x/scale)+zero,0,maxq) → scale*(q−zero) 同构 ✓（S2-2） |
| Leviathan Cor 3.9 | /tmp/iw2-evidence/leviathan.txt L317 | 「If α>c, there exists γ…at least (1+α)/(1+c)」逐字 ✓（F1 存在量词） |
| SpecInfer Algorithm 2 | /tmp/iw2-evidence/specinfer.txt L508 | 「s∼rand(H), r∼U(0,1), x_s=H[s]」逐字 ✓（F2 随机序注） |
| EAGLE-1 feature 层 | arxiv-p3p4-batch.xml 2401.15077v3 摘要 + EAGLE/README.md L34 | 「feature (second-to-top-layer) level」×3 + 「second-top-layer contextual feature vectors」✓；2.7x-3.5x/吞吐翻倍/分布保持逐字 ✓（F3） |
| EAGLE-3 README 措辞 | EAGLE/README.md L52 | 「top-layer features are limited to next-token prediction…fusion of low-, mid-, and high-level」✓（§3 观察 2 来源核实） |
| DeepSelect 互异前提 | DeepSelect/docs/DeepSelect-deep-dive.md L55 | 「assume that all elements are distinct. Equal-valued elements do not worsen the performance bound」逐字 ✓（S2-6 式 (9) 路线句） |
| DeepInfer 强制末块 | /tmp/iw1-evidence/DeepInfer/DeepInfer-main/.../candidate_blocks.py L20/L38 | 「The last valid block gets +inf so…」+ assert block_size in (8,16) ✓（S2-6 实现口径注） |
| DeepEP v1.2.1 | /tmp/eval-p3-deepep-v1/DeepEP-1.2.1/README.md L7/L39/L270-271 | pure RDMA + hook 零 SM；News 2025.06.05 NVLink；「will not be received only if you call hook()」「without any SM occupation」逐字 ✓（S3-8/9） |
| Llumnix README | /tmp/iw2-evidence/llumnix-ray/README.md L95 | TTFT mean/P99 4.6x/9.1x、P99 TBT 15% 逐字 ✓（F4 节实测引文） |
| vLLM Marlin bypass | /tmp/iw1-evidence/vllm/vllm-main/.../batch_invariant.py L1201 | 「the Marlin GEMM picks its tiling and K-split…」逐字 ✓（3.3/3.4 双向钉死） |
| SGLang 分段 matmul | /tmp/iw1-evidence/sglang/sglang-main/.../batch_invariant_ops.py L267/273 | _SEGMENT_COUNT=4 + 注释逐字 ✓（3.4 §6） |
| FlashInfer 节号 | flashinfer-paper.txt L75/L539 | §4.4 Parallel-Generation Performance ✓（S3-1） |
| 1.2 / 2.4 / models 4.1 指针目标 | inference 1.2 L37/91/93、2.4 L42/76、models 4.1 L276/291 | F5 短语在 §3 ✓、F6 §2=分层命中成本账 ✓、S3-4 d=head 维（d=64）✓ |
| references 等级 / 术语 | inference/references.md L43/89/93/105/111/196 | marlin/specinfer/fastserve [A]、smoothquant 官方 repo、batch invariance「正文保留英文」✓ |

### 6.3 机械检查

| 项 | 命令 | 结论 |
| --- | --- | --- |
| lint 9 章 | bash scripts/lint-chapter.sh <file> | 9/9 LINT OK |
| 反转义 / 白名单外命令 | /usr/bin/grep -cF '\!' '\_' '\,' '\;' '\{' '\|' '\%' '\#' '\gtrsim' 逐章 | 全 0 |
| abs_max / 旧值残留 | grep -F abs_max / 0.143 / 0.571 / 0.029 / 66.5% / 12× / 末层 feature / 批不变 | 全 0 命中 |
| 篇幅 / 编号 / 禁令词 | python3 -I check.py（body CJK 剥围栏/display/公式/code span） | 4 重章 ≥5500、5 标准 ≥3500；命题逐章重启、式编号连续、练习难度合规；禁令词/语气词 0 |

## 7. 处置建议

- 13 条 S2 全 ADDRESSED、21 条 S3 全修、无新 S1/S2——W2 fix round 通过 scoped 复审。
- N1（3.1 两处空格）与 §3 观察项 1/2/5（γ 表 γ=8 格、EAGLE README 措辞 reconciling 注、总结表 AWQ 行口径标）均为可选 minor，建议随下波顺手清，不阻塞本波 commit gate。
- 按 spec §6.4 结论记入 inference/README.md 审查记录区；§6.6 图 QA 另行核对后提交。