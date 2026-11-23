# inference W2 对抗审查 — P3 推理算子（3.1 / 3.2 / 3.3 / 3.4）

日期：2026-11-23。evaluator：fresh-context 独立审查（不继承写作上下文），只读审查，不改章节。
协议：`.superpowers/sdd/inference-w1/evaluator-brief.md` + inference spec §6.3（三项任务 + 强制重推导 + findings 表 + verification log 强制）。
输入：spec §5 P3 大纲 / §4.1 / §4.3 / §6.3；`inference/references.md`（W2 精炼版，**marlin / specinfer / fastserve 已升 [A]，按 refs 新 grade 审**）；`docs/research/inference-w2-sources.md`；STYLE_GUIDE / R2 style sheet / rubric；四章全文；/tmp/iw2-evidence + /tmp/iw1-evidence 存档；本轮 live 补取 DeepEP tag v1.2.1 tarball（codeload）+ FlashMLA 旧 commit README（contents API）+ DeepEP tags API。

## 0. 结论速览

- **S1 = 0，S2 = 9，S3 = 14**。lint 4/4 OK；flanking / 反转义机械扫描 0 命中（`\!` 与 `\_` 两处源码白名单违规另计，见 S2-4/5）；篇幅达标（3.1 = 7020、3.2 = 7143 CJK ≥5500；3.3 = 5413、3.4 = 4594 ≥3500）。
- **强制重推导**：3.1 split-KV / MLA absorbed / cascade 全 PASS；3.2 GPTQ 补偿（含 toy MSE）/ AWQ 等价 PASS，SmoothQuant 命题 4 PASS 但**数值例有效级数账 FAIL**（S2-3）；3.3 W4A16 / B_max PASS，padding 最坏情形 75% PASS 但**均值情形 FAIL**（S2-7），EP 拐点 e*=A/F PASS。
- **writer concerns 裁定**：FlashMLA deep-dive scale 译注**正确**（doc 原文 exp(m_new−m) 与重缩放因子互为倒数，章按 exp(m_old−m_new) 读是对的）；命题 4 强制块 +inf → K_b≥K+1 机制**属实**（candidate_blocks.py docstring 逐字），但命题缺「分数两两不同」前提（S2-6）；241.8 vs ridge 258.2（94%）边缘口径**属实且诚实**；硬件三口径分离**合格**；GPTQ toy 0.1832→0.1032 数学正确且已标教学；FP8 recap 对 training 1.3 §1/§6 **忠实**（含 E5M6 例外）；DeepEP V1 锚 v1.2.1 **核实**（tag 存在且为唯一 tag，V1 语义从本轮 fetched tarball 逐字核）；教学参数 F=6.1ms / t_launch=5μs / n_ker=330 / SM132 **均已声明**；SGLang `_SEGMENT_COUNT=4`、vLLM capture-pad docstring、Marlin bypass docstring **逐字核实**。
- 算子边界（算法/IO 在本维度、warp/SASS 指令级归 gpu、RDMA 硬件归 cluster）：四章均显式声明且无越界展开。等级升级（marlin [A] / specinfer [A] / smoothquant 官方 repo）使用正确。
- S2 集中于 3.2（6 条：量化定义式两处、SmoothQuant 数值例、两处源码白名单违规、AWQ 基线口径）与 3.1/3.3 各一余；全部可局部修复，不动章节骨架。含 S2 章修复后按 §6.4 复审。

## 1. Findings 表

### S2（必须修复）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S2-1 | 3.2 §2 式 (1) vs 同节正文/例（L37-46） | 对称量化步长内部口径冲突：式 (1) Δ=abs_max/2^{q−1}（INT4 → /8），正文却写「INT8 对称格点 [-127,127]（2^7−1=127）、INT4 对称格点 [-7,7]」，数值例用 Δ=1.0/7≈0.143、0.6→0.571、误差 0.029；练习 2 又用 INT3 Δ=absmax/4（从式 (1)）。/8 与 /7 两套约定混用 | AWQ 原文（awq.txt L334）：Δ=max(\|w\|)/2^{N−1}，式 (1) 忠实 AWQ；但格点 [-7,7] 与例 1.0/7 属 2^{q−1}−1 约定，与式 (1) 矛盾（/8 时 0.6→0.625、误差 0.025） | 二选一并声明：保留式 (1) 的 2^{q−1}（AWQ 口径）则格点改 [-8,7] / [-128,127]、例改 /8；或式 (1) 改 2^{q−1}−1 并注「AWQ 原文用 2^{q−1}，本书取 qmax=2^{q−1}−1 约定」。练习 2 同步 |
| S2-2 | 3.2 §2 式 (2)（L43） | 非对称量化重构式错误：Q(x)=Δ·(Round(x/Δ)+z) 缺「−z」与 clamp。按符号表 z=整数偏移，该式输出偏离真值 Δz，值域落 [0, x_max−x_min] 而非声称的 [x_min,x_max]；与式 (1) 的 Q=重构值语义（0.6→0.571）不一致 | 标准非对称量化（GPTQ repo quantize(w, scale, zero, maxq) 同构）：q=clamp(Round(x/Δ)+z, 0, 2^q−1)，x̂=Δ(q−z)；2.3 §2 只给 min-max 步长未给此式，无上游先例可沿 | 改 Q(x)=Δ·(clamp(Round(x/Δ)+z, 0, 2^q−1) − z)，z=Round(−x_min/Δ)（或等价约定并声明）；字节账（式 (3)(4)）不受影响 |
| S2-3 | 3.2 §5.1 全数字实例（L252）+ 练习 4(c) 校验锚 | SmoothQuant 有效级数账错误：「α=0.5 迁移后所有通道最大幅度统一到 10，正常通道 256×10/10=256 级——从 2-3 级回到满级」。toy 参数 outlier 通道 m_X=100、平坦权重 m_W=1、正常通道 m_i=1：正常通道 s=1^0.5/1^0.5=1，迁移后 max 仍 1；per-tensor max 由 100 降 10，正常通道有效级数 = 256×1/10 = **25.6**（10× 改善），非 256。「所有通道统一到 10」仅当各通道 m_X·m_W 乘积相等或 α=1（全压平）才成立 | 独立重推命题 4：迁移后通道 max=(m_X m_W)^{1−α}；正常通道乘积 1 → max 1；outlier 通道乘积 100 → max 10。命题 4 本身正确，错在数值例把「通道内权重/激活均分」误写成「跨通道激活均平」。原文（smoothquant.txt L307）只声称对应通道权重与激活共享相似 max，未声称跨通道统一 | 数值例改「per-tensor max 100→10，正常通道 2.56→25.6 级（10×）；α=1 才全压平到满级 256（难度全归权重）」；练习 4(c) 校验锚同步改 25.6 |
| S2-4 | 3.2 式 (1)(2)(10)（L37/43/195） | `\mathrm{Round}\!\left(` 使用 `\!`——STYLE_GUIDE §3 反转义陷阱明令禁止（`\!`→`!`），display 渲染为可见「Round!(…)」；源码白名单不含 `\!` | STYLE_GUIDE §3「反斜杠反转义陷阱（实测）」逐字列 `\!`→`!`；§3.1 白名单间距行只收 `\thinspace`/`\negthinspace`/`\hspace` | 三处删 `\!`，写 `\mathrm{Round}\left(` |
| S2-5 | 3.2 全章 10 处 `\mathrm{abs\_max}`（式 (1)(15) 及各符号表） | `\_` 不在源码白名单，且落入 §3 总规则「`\`+ASCII 标点先反转义」（`_` 为 ASCII 标点、CommonMark 可转义）：反转义后 `\mathrm{abs_max}` 的 `_` 在 \mathrm 内成下标，渲染为 absₘax 而非 abs_max。90 章已发布章无 `\_` 先例，docs/math-render-probe.md 无 `\_` 豁免探测记录 | STYLE_GUIDE §3 总规则 + §3.1 白名单（无 `\_` 行）；grep models/training/inference W1 零先例；probe 文件零记录 | 改 `\mathrm{absmax}`（或 `\max\vert x\vert` 记法）全章统一；如坚持下划线写法则先做 GitHub 渲染探测、入 probe 记录后再收编。连带修式 (1) 符号表注「（表格与正文记法，公式内写 $\vert x \vert$）」——abs_max≠\|x\|，该注语义不清，重写为「组内最大绝对值，公式记 \max\vert x\vert」 |
| S2-6 | 3.1 §7 命题 4 及证明（L289-291） | 命题缺前提：证明步「分数 ≥ τ 的 token 至多 K 个」仅在分数两两不同时成立；bf16 indexer 分数同分常见，同分时含 top-K token 的块可多于 K 个（并列块分数=τ），K_b≥K 不保证对任意 tie-breaking 精确。所译 DeepSelect doc 显式假设「all elements are distinct」，章未移植该前提。第二clause「存在输入使**至多** K−K_b 个 top-K token 落选」量词弱（反例实示**恰** K−K_b 可达；且任意输入损失 ≤ K−K_b） | DeepSelect-deep-dive.md L40「For simplicity, assume that all elements are distinct」；独立构造反例：K=2，分数 5/3/3/1 分落 4 块，τ=3，K_b=2 时并列块 tie-break 可挤掉一个 top-2 token。强制块口径 K_b≥K+1 的收紧本身正确（candidate_blocks.py docstring 核实） | 命题加「设分数两两不同（同分并列时结论按 tie-breaking 另议）」；第二 clause 改「损失可达 K−K_b（且恒 ≤ K−K_b）」；练习/坑注口径随之 |
| S2-7 | 3.3 §4 全数字实例（L95）+ 练习 2 校验锚 | padding 均值账在 μ=32 处与定义不相容：「设余数均匀（M_j mod M_blk 均匀分布），平均 pad=63.5，w≈63.5/(32+63.5)=66.5%」。余数均匀 {0..127} 均值 63.5，与均值 μ=32 不能同时成立（非空 expert 余数均匀 ⇒ 均值 ≥63.5）；且 μ<M_blk、全 expert 非空时每 expert 恰占一整块，w=1−μ/M_blk=75% **对任意余数分布精确成立**，66.5% 读数不可达。均值-余数模型只在 μ≳M_blk 时适用（μ=512 行与渐近式正确） | 独立重推：pad=128⌈M_j/128⌉−M_j；M_j∈(0,128] ⇒ pad=128−M_j，w=Σpad/Σ128=1−μ/128 恒等。命题 2(b)（全 M_j=μ）本身正确；错在把大 μ 近似式代入 μ=32 | μ=32 行改「全非空时 w=75% 精确（与分布无关）；含空 expert 时按激活数另计」；均值公式限定 μ≳M_blk 并注明近似条件；练习 2 校验锚「代入 μ=32 与 §4 实例一致」同步改 |
| S2-8 | 3.1 §2 式 (4)（L81） | M_red ≈ 2S H_q(4d_head+4)·bytes 尾部「·bytes」冗余且未定义：符号表已声明「系数 4 = fp32 每元素字节数」，括号内已是字节量；数值代入（451.5 KiB）未乘 bytes。按 1.2 约定 bytes=每元素字节数（bf16=2）逐字代入会翻倍（903 KiB、η=0.17%） | 章内符号表 + 数值例自证；1.2 式 (3) 的 ·bytes 是变量因子，本式无对应变量定义 | 删「·bytes」（4 已编码 fp32 字节），或重写 2S H_q(d_head+1)·4 并注 4=fp32 字节 |
| S2-9 | 3.2 §4 开场（L174）vs §4.2（L206） | AWQ 两套 RTN 基线未注口径差：§4 引图 2「OPT-6.7B INT3-g128，1% salient 留 fp16，PPL 43.2→13.0」；§4.2 引表 2「s=1 时 PPL 23.54（即 RTN）」。原文图 2 与表 1/2 基线本就不一致（表 1 INT3-g128 RTN=23.54、act-1%=11.39；图 2 写 43.2/13.0），章把两者当同一设置并列且无口径注，读者对账必困惑 | awq.txt L192（图 2 caption 逐字 43.2/13.0/INT3-g128）、表 1 OPT-6.7B 行（10.86/23.54/11.58/11.39/11.36/…）、表 2（s=1: 23.54）；两处引用各自忠实，缺的是 reconciling 注 | 加口径注：「图 2 与表 1/2 的 RTN 基线在原文即不一致（43.2 vs 23.54），本书各引出处、不混算」；总结表 AWQ 行标「图 2 口径」 |

### S3（minor，下波清零）

| # | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- |
| S3-1 | 3.1 §4（L163） | FlashInfer e2e 数引「§5.2 口径」误——论文 parallel-generation 评测在 **§4.4**；§5.2 是 Related Work 的 Sparse Optimizations on GPUs | flashinfer-paper.txt 目录（L75「4.4 Parallel-Generation Performance」/L81「5.2 Sparse Optimizations on GPUs」）；数值本身（13.73/17.42/16.41/22.86、n=4、rate 16、MLC-Engine、Llama-3.1 8B/70B、ShareGPT）逐字核实 | 改「§4.4 口径」 |
| S3-2 | 3.2 §5.1（L242-245） | s_j 公式引「原文式 (3)」误——SmoothQuant 式 (3) 是迁移等价 Y=(X diag(s)⁻¹)(diag(s) W)，s_j=max\|X_j\|^α/max\|W_j\|^{1−α} 是**式 (4)** | smoothquant.txt L283-305（(3)/(4) 编号逐字） | 改「原文式 (4)」 |
| S3-3 | 3.2 §5（L230） | 「per-channel 激活量化硬件不可行（原文表 1）」指针偏——表 1 是 simulated per-channel 精度验证；硬件不可行论证在 §3 正文 + Fig 3（外维 scale、低吞吐指令插不进 MMA 序列） | smoothquant.txt L268-276 | 改「原文 §3、Fig 3；精度模拟见表 1」 |
| S3-4 | 3.1 §1（L9） | recap IO 账 O(T²d²/M) 裸 d 未翻译——此 d 是 head 维（models 4.1 §5.4 口径 d=64），按本章记号应 d_head；spec §4.3 recap 翻译硬规则 | models 4.1 式 (17) 上下文（d=head 维）；3.1 符号约定已声明 d_head | 写 O(T²d_head²/M) 并注「models 4.1 式 (17) 的 d 译 d_head」 |
| S3-5 | 3.1 §5（L210） | 「Q 侧合并矩阵每 head 1536×512，参数与 FLOPs 较展开形两因子（128×1536+128×512）放大 3 倍——（models 4.1 §4.4 代价账，recap）」指针超载：models §4.4 只有 score 逐对 FLOPs ×3；合并矩阵参数 3×（786432/262144）是本章自算（数学正确），且「FLOPs 3×」按离线预计算口径成立（decode 逐步投影比是 4×，展开形 k^C 已缓存） | models 4.1 §4.4 全文（代价三笔无合并矩阵参数账）；独立算 1536×512=786432、128×1536+128×512=262144、比 3.0 | 标「本章账」+ 口径注（参数/离线预计算 FLOPs 3×；decode 逐步投影 4×） |
| S3-6 | 3.1 §3（L127） | FA2 技术 (2)「decode 侧的实例正是 §2 的 KV split」易误读——FA2 论文的 seq 维并行指 query seq；KV split 是 FlashDecoding 的独立贡献（FA≥2.2 集成），非 FA2 论文技术 (2) 本身 | FA2 摘要（parallelize … across thread blocks，论文 §3.2 沿 query 维）；FlashDecoding 博客独立署名 | 改「decode q=1 时该思路的同构策略是 KV split（FlashDecoding，FA≥2.2 集成，博客口径）」 |
| S3-7 | 3.4 §1（L9） | 「三个步时全部 recap 1.2 §2–3 的算例」不准确——4.81 ms（B=1 权重-only 下界）非 1.2 算例（grep 1.2/1.3 无 4.81），是本章新算；且 B=1、T=4096 完整下界 ≈4.97 ms（KV 读 +3.3%） | 1.2 全文 grep；16.1e9/3.35e12=4.806 ms | 改「15.1/66.5 recap 1.2；4.81 为本章权重-only 下界算例（B=1 完整下界 ≈4.97，KV 项 +3.3%）」 |
| S3-8 | 3.3 §6（L138） | low-latency「纯 RDMA 点对点」对锚定 tag 略过强——v1.2.1 README News（2025.06.05，PR #173）「low-latency kernels now leverage NVLink as much as possible」（节点内单跳 NVLink，仍绕两阶段转发） | 本轮 fetched DeepEP-1.2.1/README.md L39；节标题仍写 pure RDMA（L26） | 加一句版本注：「纯 RDMA 为原始设计口径；v1.2.1 已含节点内 NVLink 直连优化（单跳，不改两阶段对比语义）」 |
| S3-9 | 3.3 §6（L138） | hook 机制转述「计算侧注册完成 hook、数据到达即触发」方向偏——V1 语义：dispatch 返回 hook，RDMA 后台接收**不占 SM**，计算侧在需要时**主动调用 hook()** 完成接收（pull 式，double-batch overlap 用） | DeepEP-1.2.1/README.md L266-290（「the actual tensor will not be received only if you call hook()」「without any SM occupation」「RDMA network traffic is happening in the background」） | 改「RDMA 后台接收（不占 SM）、计算侧主动调 hook 完成接收，免 CPU 同步往返」 |
| S3-10 | 3.1 §1/§2/§4；3.3 §4 | 术语首现未定义：3.1 CTA（L11 起多处，未展开 cooperative thread array / 线程块）、grid（未定义）；3.3 launch（§4 首现，定义在 3.4 §1，无前向指针） | STYLE_GUIDE §1 首现规则 / R2-8；3.4 §1 有 launch 定义 | CTA/grid 首现处一行定义（CTA=thread block）；3.3 launch 加「（CPU 提交 kernel，定义见 3.4 §1）」 |
| S3-11 | 3.1 全章（L3/76/104/149/241/327/329/350） | 「批不变性 / 批不变模式」中文写法 vs 术语表口径「正文保留英文」及 3.3/3.4 的 batch-invariant 写法跨章漂移 | references.md 术语表 batch invariance 行；1.4 §6 正文用英文 | 统一 batch invariance / batch-invariant 模式（或术语表补「修饰语可用批不变」例外） |
| S3-12 | 3.3 §7（L162）+ 总结表 | 「5 个 processor 各走一遍」教学假设未声明（同章 F=6.1ms、3.4 t_launch/n_ker 均显式声明）；1.4 链为 penalty×3 + ban mask | 3.3 §5 声明先例；1.4 §3 | 标「以 5 遍为例（教学假设）」或对齐 1.4 链遍数 |
| S3-13 | 3.2 §4.3 式 (13)、§6 式 (16)(17) | R2-3 缺口：三式后 300 字内无全数字代入实例（节有量化锚但无公式代入） | R2-3 逐 display 检查 | 各补一个最小代入（如 s_X=4、α=0.5→s=2；LLM.int8 给一个 2×2 分解算例或声明「结构式，数值锚见 §6 内存账」） |
| S3-14 | 3.2 §3.3 Step 3（L164） | Algorithm 1 块内更新写 W_{:,j:}（全局切片记法）vs 论文 W_{:,j:(i+B)}（块局部）；「块内更新」限定词 + ASCII 图已澄清，切片记法仍易歧义 | gptq.txt Algorithm 1（W_{:,j:(i+B)}）；gptq.py L137（W1 为块局部 clone） | 切片加块上界 j:(i+B) 或注「j: 指块内剩余列」 |

## 2. 强制重推导（独立重推，不信任 writer）

### 3.1 split-KV 并行度 + reduction 账 — PASS（式 (4) 记法缺陷另计 S2-8）
- grid n_CTA=B·H_kv·S：B=1、H_kv=8、S=1 → 8；S=14 → 112≥108 ✓；段长 ⌈131072/14⌉=9363 ✓；S 上限 131072/128=1024 ✓（scheduler.cuh L205 下界 max(128/page_size,1) 逐字核实）。
- 合并式 (3) 独立重推：w_iO_i=e^{LSE_i−m}·(Σe^{x_s−m_i}v_s)/ℓ_i=e^{−m}Σe^{x_s}v_s（用 e^{LSE_i}=ℓ_i e^{m_i}）；分母同构；O/LSE 基准无关 ✓。结合/交换继承不相交并 ✓。空段 LSE=−∞→w=0 ✓。
- 数值例独立复算：段 1 m=2、ℓ=1.3679、O=(0.2689,0.7311)、LSE=2.3133；段 2 m=5、ℓ=1.0025、O=(1.9951,0.9951)、LSE=5.0025；合并 m=5.0025、w_1=0.0679、O=(1.8853,0.9783)、LSE=5.0682；全行直算 O=(1.88526,0.97826)、LSE=ln(158.888)=5.0682 ✓ 逐位吻合。
- reduction 账：2×14×32×516=462336 B=451.5 KiB ✓；η=462336/536870912=0.0861% ✓；FLOPs 比 S/(4T)=14/524288≈0.003% ✓。Amdahl 折扣：KV 512 MiB/层→160.3 μs×32=5.13 ms；权重 16.1 GB→4.81 ms；(5.13+4.81)/4.81=2.07≈2.1× ✓。博客 8×/50× 为 34B 长窗口口径，分离正确 ✓。

### 3.1 MLA absorbed 等价 — PASS
- K 侧：(W^UQ c^Q)^T(W^UK c^KV)=(c^Q)^T(W^UQ)^T W^UK c^KV=⟨q̃,c^KV⟩，W̃∈R^{1536×512} ✓（models 4.1 式 (11) 同构，符号翻译 d_c→d_latent 等已声明）。V 侧线性性 Σp·W^UV c=W^UV Σp·c ✓，Ô=W^O W^UV ✓（式 (13)(14)）。RoPE 不吸收（decoupled 前提）✓。
- 几何账：192→576 ×3 ✓；32832=2×128×128+64 → 576，57× ✓（models 4.1 §4.3 逐字）；合并矩阵参数 786432/262144=3.0 ✓（指针问题 S3-5）。
- I_MLA=2H_q s_q T(d_k+d_v)/(2T d_k)=H_q s_q(d_k+d_v)/d_k：128×1088/576=241.78 ✓；H800 throttled ridge 865/3.35=258.2 ✓；241.8/258.2=93.6%≈94% ✓；doc 阈值 (1/2)(865/3.35)=129.1→doc 自写「=128」，章忠实转述并给精确边缘读数 ✓。寄存器 64×512=32768 / 65536 ✓；O_L/O_R 64×256=16384 ✓；660/865=76.3% ✓；9 次 64×64 TMA=64×576 ✓。

### 3.1 cascade 两级正确性 — PASS
- V_i=P⊔U_i 不相交并 → 命题 1 直接给合并=全集 ✓；前缀状态逐请求合法依赖前缀不可变（models 2.4 命题 L112 核实）✓；物理共享经 2.1 ref_cnt/COW + 2.2 ✓。IO：M_naive=32×2304×128 KiB=9 GiB ✓；M_cas=256 MiB+1 GiB=1.25 GiB ✓；省 86.1% ✓；前缀读 B→⌈B/r_q⌉ ✓。B=1/无前缀退化为纯开销、多级递归边界 ✓。论文 e2e 数（n=4 峰值、ITL/TTFT 四数、rate 16、MLC-Engine/Llama-3.1/ShareGPT）逐字核实（节号误引 S3-1）。

### 3.2 GPTQ 误差补偿（约束二次极小 → Schur 块逆 → 逆 Hessian 形）— PASS
- 行目标 ‖e^T X‖²=e^T XX^T e=½e^T He，H=2XX^T ✓（原文逐字）。固定 e_q：H_Fq e_q+H_FF e_F=0 → e_F=−H_FF⁻¹H_Fq e_q，δ_F=−e_F ✓。Schur S=H_qq−H_qF H_FF⁻¹H_Fq，[H⁻¹]_qq=S⁻¹、[H⁻¹]_{F,q}=−H_FF⁻¹H_Fq S⁻¹ ✓ → δ_F=−(e_q/[H⁻¹]_qq)[H⁻¹]_{F,q} ✓ 与原文式 (2) 一致。
- 降阶式 (8)：H⁻¹−col·row/[H⁻¹]_qq 删行列 ✓（原文式 (3)）；toy 验证 [[0,0],[0,1/4]]→[1/4]=H_FF⁻¹ ✓。
- toy 独立复算：X=[[1,0,1],[0,1,1]]、H=[[4,2],[2,4]]、H⁻¹=[[1/3,−1/6],[−1/6,1/3]] ✓；RTN ŵ=(0.5,0.0)、e=(0.1,0.24)、½e^THe=0.1832 ✓；δ_2=−(0.1/(1/3))(−1/6)=0.05、w_2→0.29→quant 0.5、e=(0.1,−0.26)、½e^THe=0.1032 ✓；降 43.7% ✓；约束最优 e_2=−H_21 e_1/H_22=−0.05 与 δ_2 一致 ✓。**toy 已标教学**（§1 口径「2×2/2×3 小矩阵（教学 toy，随文声明）」+ §3.2「toy」）✓。
- 复杂度 O(d_out·d_in³)→O(max{d_out·d_in²,d_in³})、降 min{d_out,d_in} 倍：独立计数（逆更新 d_in 次×O(d_in²)、权重更新 O(d_out·d_in²)）✓，原文 L243 逐字同式；4096×4096 降 4096 倍 ✓。lazy batch B=128 / Cholesky(H⁻¹)^T / dampening 1% 平均对角 / 行 q 从对角起 / sqrt 差别 / indefinite「几 B 以上至少几层」全部原文核实 ✓。act-order=argsort(diag(H)) 代码核实（gptq.py L90）✓。

### 3.2 AWQ per-channel scaling 等价 — PASS
- (W diag(s))(diag(s)⁻¹X)=W(diag(s)diag(s)⁻¹)X=WX ✓；s_j=0 不可逆反例 ✓；量化打破等式 ✓。误差比 (Δ'/Δ)(1/s) ✓（原文式 (2) 上下文）；s=2→≈0.5 ✓。表 2 全数（23.54/11.92/12.36、8.2%/21.2%、1.038/1.213、0.519）逐字核实 ✓。s=s_X^α、α∈[0,1] grid 20、weight clipping、group 128、INT4/INT3、Pile 校准 ✓。基线口径冲突另计 S2-9。

### 3.2 SmoothQuant 迁移 + α 难度分配 — 命题 PASS / 数值例 FAIL（S2-3）
- Y=(X diag(s)⁻¹)(diag(s)W) 与本章约定 (W diag(s))(diag(s)⁻¹X)=WX 同一恒等式 ✓（原文式 (3) 逐字，行向量约定翻译已声明）。
- 命题 4 独立重推：m_X/s_j=m_X·m_W^{1−α}/m_X^α=(m_X m_W)^{1−α} ✓；s_j·m_W=(m_X m_W)^α ✓；α=0.5 两者=几何平均 ✓；α→1/0 极限 ✓。toy s 值（10/31.6/100）与通道 max（10/3.16/31.6/1/100）✓。
- **FAIL 点**：有效级数账「正常通道 256 级 / 所有通道统一到 10」——正常通道 (m_X=1,m_W=1) 迁移后 max=1，per-tensor max=10，级数=25.6 非 256（详见 S2-3）。

### 3.3 W4A16 带宽节省（4.125 bit → 3.88×、B_max）— PASS
- bytes_weight=q/8+bytes_scale/g=0.5+2/128=0.515625 → 4.125 bit ✓；S_ideal=2/0.515625=16/4.125=3.879≈3.88 ✓（README「optimal 3.87x」+0.125 bit 自注逐字核实，截断口径声明 ✓）。g=32→3.56×、g=256→3.94× ✓。
- 命题 1：I=2B/bytes_weight≤R → B_max=R·bytes_weight/2 ✓；H100 R=989.5/3.35=295.4 → B_max=76.2≈76 ✓。数值例：B=16 bf16 33.6 MB→10.0 μs、INT4 8.65 MB→2.58 μs、FLOPs 5.37e8→0.54 μs ✓；B=128 INT4 计算 4.34 μs>2.58（I=496>295 翻转）、bf16 仍 10.0 μs（I=128<295）、加速 2.31× ✓。MAC 口径 25–50（FLOP:byte 100–200、0.5 B/权重）✓ README 逐字。FP8 B_max=R_FP8/2≈295 跨精度对消 ✓（1.2 式 (6) recap 忠实）。

### 3.3 grouped GEMM padding（μ=32/M_blk=128 → 75%）— 最坏 PASS / 均值 FAIL（S2-7）
- pad_j=⌈M_j/M_blk⌉M_blk−M_j≤M_blk−1 ✓；全 M_j=μ<M_blk → w=1−μ/M_blk=75% ✓（命题 2 证明正确）。M_j=513→pad 127 触界 ✓、512→0 ✓。
- μ=512 均值 63.5/575.5=11.0% ✓、渐近 M_blk/(2μ) ✓（大 μ 下成立）。**μ=32 均值 66.5% 与均匀余数假设不相容**（S2-7）。
- masked buffer (E/e)·C·K·bytes=4×1024×7168=29.4 MB ✓；有效 4×32×7168=0.92 MB、32× ✓。contiguous/masked 语义、对齐常数 kLegacyMKAlignmentForContiguousLayout=128（runtime.hpp L12）✓。

### 3.3 EP>32 拐点 e*=A/F — PASS
- τ(e)=max(A/e,F)+g(e)：A/e 严格减、与 F 交于 e*=A/F；e≥e* 时 τ=F+g(e) 单调不减 ✓。g≡0 持平边界 ✓。#DP=#EP 耦合下每卡计算不变（Bk/e=B_local·k）前提 ✓（1.2 §5 reconciling 注忠实）；固定全局 batch 下计算并入 A/e 的结构注 ✓。
- A=654 GB/3.35 TB/s=195.2 ms ✓（654 GB 与 1.2 §5/§6 一致）；e*=195.2/6.1=32.0 ✓；表五行（24.5/12.5/6.72/7.36/8.64）逐行复算 ✓；最小值 e=32 对该 g(e)=0.02(e−1) 成立（e=31→6.90、e=33→6.74 均 >6.72）✓。F/g 教学参数显式声明 ✓。doc L293 问句「beyond 32」+ L295/L297 逐字核实 ✓。Mega MoE 重叠式 max(A/e,F,g(e)) 与练习 3(c)（e=128: 6.1<8.64）✓。

## 3. Writer concerns 裁定（独立核一手来源）

| concern | 裁定 | 证据 |
| --- | --- | --- |
| 3.1 deep-dive scale 定义 exp(m_new−m) 疑误、译注按 exp(m_old−m_new) 读 | **译注正确**。doc L30 逐字 scale_0=exp(m_new_0−m)，m_new_0=max(m,mp_0)≥m ⇒ scale≥1；而 L32/35/36/38 将其乘入旧累积——online softmax 换基准（max 增大）的正确因子是 exp(m_old−m_new)≤1，doc 的定义与用法互为倒数、按字面执行会放大旧贡献（与 doc 自称「mathematically equivalent to FlashAttention's online softmax」矛盾）。章未改引用原文、以译注按数学正确方向读，且 O_R 累积因子=两因子之积=exp(m_old−m_new) 的重构正确 | FlashMLA docs/20250422-new-kernel-deep-dive.md L27-42 逐字；独立重推 seesaw 各步基准平移 |
| 3.1 命题 4 DeepInfer 强制末块 +inf 占预选名额 → K_b≥K+1 | **机制属实**：candidate_blocks.py docstring「The last valid block gets +inf so that candidate selection always includes it」+ assert block_size in (8,16) + 1024B 行对齐逐字核实；+inf 占块级 TopK 一名额 ⇒ 精确条件 K_b−1≥K。但命题本身缺分数互异前提（S2-6） | /tmp/iw1-evidence/DeepInfer candidate_blocks.py L12-50 |
| 3.1 MLA compute-bound 边缘 241.8 vs H800 ridge 258.2（94%）并列呈现 | **属实且诚实**：doc 公式/阈值/865 throttled/3.35 TB/s 逐字核实；章给 doc 近似口径（≈2H_q s_q、阈值 128）+ 精确读数 94% 边缘，双口径并列不混 | deep-dive L11-15；独立复算 241.78/258.2 |
| 3.1 硬件口径分离（A100 108SM 博客[C] / H100 3.35TB/s datasheet[C] / H800 doc） | **合格**：章脊表逐行标来源 +「两笔账各自出处，不混算同一台机器」；occupancy 用博客 A100、带宽-时间用 H100 datasheet、MLA regime 用 H800 doc，无混算 | 章脊表 L17-19；flashdecoding.txt L54；refs nvidia-datasheet；deep-dive L13 |
| 3.2 GPTQ toy MSE 0.1832→0.1032 为 writer 构造教学例 | **数学正确 + 已标教学**（§1 口径声明 + §3.2「toy」）；非论文事实、未冒充原文 | 独立复算见 §2；原文无此 toy |
| 3.2 FP8 recap 对 training 1.3 §1/§6 忠实性 | **忠实**：E4M3 max 448/18 binade/无 inf、E5M2 57344/inf+NaN、1×128 激活/128×128 权重、s_g=absmax/448、N_C=128 提升 CUDA Core fp32（~14bit TC 累加）、E5M6 例外、loss 偏差 <0.25%、权重块 scale 开销 0.024%、ue8m0 幂次 scale 精确——逐条对 training 1.3 §1 表/§6.2-6.5 核实（E5M6 为 training 1.3 已交付事实，非 typo） | training/01-autograd-and-backprop/03-mixed-precision.md 全文 |
| 3.2 论文事实对 /tmp/iw2-evidence | GPTQ/AWQ/SmoothQuant/LLM.int8 全部数值与机制逐字核（见 verification log）；发现 AWQ 图 2/表 1 基线原文不一致未注（S2-9）、SmoothQuant s_j 式号误引（S3-2） | gptq.txt/awq.txt/smoothquant.txt/llm-int8.txt + gptq repo |
| 3.3 DeepEP V1 语义锚 tag v1.2.1（本地 tarball 为 V2.5） | **核实**：tags API 返回唯一 tag v1.2.1；本轮 codeload 取 v1.2.1 tarball，normal=NVLink+RDMA 两阶段转发（训练/prefill、SM 数可控）、low-latency=纯 RDMA+hook 逐字核实；V2.5 漂移注（EPBuffer/NCCL Gin/DeepJIT/do_cpu_sync/psum_num_recv/零 SM Notes/lb_*）对当前 README 逐字核实。两处 nuance 见 S3-8/9 | api.github.com/tags；/tmp/eval-p3-deepep-v1/DeepEP-1.2.1/README.md；/tmp/iw2-evidence/DeepEP/README.md |
| 3.3 教学参数（F=6.1ms、t_launch=5μs、n_ker=330、SM132）声明非官方 | **均已显式声明**（3.3 §5「两者为声明的教学参数…非官方数值」；3.4 §1 t_launch「声明教学值 5 μs」、§3 n_ker「声明教学口径」、§6 SM132「教学口径」）；「5 个 processor」未声明（S3-12） | 各章原文 |
| 3.3 新 tarball 代码事实（SGLang _SEGMENT_COUNT=4、vLLM capture-pad docstring、Marlin bypass） | **逐字核实**：SGLang L273 _SEGMENT_COUNT=4 + L267 注释「adds the segment sums in order, at every M」；vLLM decode_cudagraph_capture_shapes docstring「A batch pads to the smallest graph that holds it」；batch_invariant.py L1201「the Marlin GEMM picks its tiling and K-split from the number of tokens」+ dequantize_gptq_weight | /tmp/iw1-evidence sglang/vllm tarball（2026-11-23 同日） |

## 4. 跨章一致性

- **符号 §4.3**：B（batch，3.3 §2-3 GEMM 行数 / §4 起全局 batch 双口径显式声明）、b（block size）、κ、d_model/d_latent/d_rope、E/e/k 大小写、bytes 不占字母（per-element 因子一律文字/\mathrm{bytes}；命名流量量 W/M_red/S 沿 1.2 W 先例）——全 batch 一致。局部记号均章首声明（3.2 α/k/q/g/n、3.3 M_blk/C/V/g/S_ideal、3.4 S/n/n_ker/t_launch、3.1 S/H_q/d_k/d_v/m_{t,h} 消歧注）。唯一翻译漏项：3.1 recap IO 式裸 d（S3-4）。
- **章脊口径**：3.1 三口径（8B GQA κ=4096 / V3 MLA 128 头 512+64 61 层 / V4.1 index 字段 40 层）与 1.2 章脊表、V4.1 config 逐项一致；3.3 三口径（8B / V3 E=256+1 k=8 7168/2048 58 MoE 层 W_exp 44.0M / V4.1 384+1 k=6 5120/2304 词表 129280）与 1.2/config 一致；3.4 8B 口径与 1.2 一致（4.81 ms 指针问题 S3-7）。
- **recap 指针**：models 4.1 式 (11)-(17)/§4.3-4.4/§5.2 logit 例 (1,2,5,−1)、models 2.4 前缀不可变命题、models 4.2 §6 indexer（ReLU 加权内积/头间共享/K=2048）/§8 召回、models 2.6 SwiGLU、training 1.3 §1/§6、training 5.4 §6（IB 50/NVLink 160 GB/s、两阶段同序号转发、20 SM）、inference 1.2 式 (2)(3)(4)(6)(11)(12)(13)/§5 reconciling、1.3 scale-up domain、1.4 §6 模式入口表 +「3.4 量化」承诺、2.1 式 (3)/§5 beam search、2.3 式 (3)(4)/§4 格式账——全部核实到位。误引/超载：S3-1/2/3/5/7。
- **算子边界**：四章均显式声明算法/tiling 策略/IO/接口在本章，warp 内寄存器分配细节、tensor core 指令形状、SASS、TMA/WGMMA 指令语义、lop3/bfe、PTX、Triton codegen 归 gpu，RDMA/NVLink 硬件归 cluster；FlashMLA SASS 差距只一句 + 7.3 指针（doc L289 官方披露口径正确）。寄存器预算/seesaw/padded TC 均为策略级资源账，无指令级越界。
- **等级升级**：marlin [A]（3.3 metadata/参考标「W2 升 [A]」+ arXiv:2408.11743）、specinfer [A]（3.1 tree mask 引 [Miao et al., 2023][A]）、smoothquant 官方 repo（3.2 §5 末句）——与 refs W2 裁定一致；fastserve 不在本 batch。FlashDecoding [C] 开放博客、FlashMLA/DeepSelect/DeepGEMM/DeepEP/TileKernels/DeepInfer 教学参照、config [W]、OCP/NVFP4/datasheet [C]、DeepInfer doc 官方披露口径注——全部正确标注。
- **两层走读纪律**：机制章代码均为「机制摘录」标注（scheduler.cuh / flash_attn.py），全量走读指针 5.3/6.x/7.3 到位。

## 5. 规范审查（STYLE_GUIDE / R2 / rubric）

- lint：4/4 LINT OK（禁令词/模板/metadata/arXiv 收录/证据等级/$$ 配平）。
- 机械扫描：flanking 开口/闭口/内侧空格 0 命中；`\,`/`\;`/`\{`/`\|`/`\%`/`\#` 0 命中；**`\!` 3 处（S2-4）、`\_` 10 处（S2-5）**；行内 `<`+字母 0 命中；矩阵环境均在 display ✓。
- rubric A：禁令词/营销词/松散语气词/历史叙事 0 命中（OBS→OBQ→GPTQ 与 FA 代际表为算法谱系/技术代际，非年代叙事）。「显著」3 处均带数值或原文引用豁免。
- rubric B：含糊量词 0 命中；术语漂移仅「批不变性」跨章写法（S3-11）；节开场全部事实/数值/设问（R2-1 逐节核过）。
- R2 逐条：R2-1 ✓（四章全部 ## 节首两句内问题/数值/元问题）；R2-2 ✓（章脊表 + 换口径原因声明）；R2-3 大体 ✓（3.2 三式缺口 S3-13）；R2-4 ✓（3.1 命题 1-4 / 3.2 命题 1-4 / 3.3 命题 1-3 / 3.4 命题 1-3，证明路线句齐全，「留作练习」闭环均存在）；R2-5 ✓（≥4 步推导首句路线句）；R2-6 ✓（定义配正例 + 边界反例：s_j=0、fan-out、grid 依赖 T、种子/硬件版本）；R2-7 ✓（占用率/免费/近理想/开销/暴露等窄口径词首现注）；R2-8 大体 ✓（CTA/grid/launch 首现缺口 S3-10）；R2-9 ✓；R2-10/11 ✓（提点/坑块引用齐全、坑含触发-现象-规避）；R2-12 ✓（总结表逐行回指）。
- 习题：四章各 4 题，难度 [10]/[20]/[30]/[30] 非递减 ✓；≥1 题 [10]/[20] ✓；无 [40] ✓；计算/代码题全带校验锚 ✓；每章 ≥1 题以正文命题为目标（3.1 练 2→命题 1、3.2 练 3→命题 2、3.3 练 2→命题 2、3.4 练 2→命题 2）✓；代码题先预测后验证 ✓。校验锚错误随 S2-3/S2-7 修正。
- 篇幅：3.1 7020 / 3.2 7143（重章 ≥5500 ✓）、3.3 5413 / 3.4 4594（标准 ≥3500 ✓）；ASCII 结构图 3.1×3 / 3.2×3 / 3.3×1 / 3.4×2 ✓。
- 模板：标题中英双语、metadata 定位|前置|锚点、总结表、练习、参考（../references.md 链接 + key 标注）✓。

## 6. Verification log（抽查项 × 来源 × 结论）

覆盖率：四章全文逐行读；全部 display 公式独立重推或数值复算；全部 repo 行号锚点对 tarball 核实；全部逐字引文对一手来源核实；全部数值事实对论文/README/doc/config 核实。估计覆盖率：数值断言 ~100%、repo 锚点 100%、逐字引文 100%、命题 100%。

### 6.1 一手来源（本轮 live 取）
| 项 | 来源 | 结论 |
| --- | --- | --- |
| DeepEP tag 列表 | api.github.com/repos/deepseek-ai/DeepEP/tags | 唯一 tag v1.2.1 ✓（V1 最后版本成立） |
| DeepEP V1 语义 | codeload v1.2.1 tarball → /tmp/eval-p3-deepep-v1/DeepEP-1.2.1/README.md L5/7/13-39/266-290 | normal 两阶段转发 + low-latency 纯 RDMA + hook 不占 SM ✓；News 2025.06.05 NVLink 优化（S3-8）；hook pull 语义（S3-9） |
| FlashMLA Hopper 接口 | contents API README.md?ref=ba89a34… L98-153 | get_mla_metadata(cache_seqlens,…)→(tile_scheduler_metadata,num_splits)；flash_mla_with_kvcache(q,kvcache,block_table,cache_seqlens,dv,…)→(out,lse) ✓ 3.1 §6 表行准确 |

### 6.2 存档论文 / 博客（/tmp/iw2-evidence、/tmp/iw1-evidence）
| 项 | 来源引句/位置 | 结论 |
| --- | --- | --- |
| FlashDecoding occupancy/三步/数值 | flashdecoding.txt L54（108 SM、<1% 逐字）、L70-76（LSE scalar、views 逐字）、L92-102（8×/50×/32k 恒定）、L108-110（FA≥2.2/xFormers≥0.0.22） | 3.1 §1/§2 全部逐字吻合 ✓ |
| FA2 摘要 | arxiv-p3p4-batch.xml 2307.08691：25-40%、三技术、~2×、50-73%、225 TFLOPs/s（72% MFU） | 3.1 §3 ✓ |
| FA3 摘要 | 2407.08608：FA2 H100 35%、warp-specialization/interleave/block quant+incoherent、1.5-2.0×、740（75%）、~1.2 PFLOPs、2.6× lower error | 3.1 §3 ✓；740/989.5=74.8% 一致性核对 ✓ |
| FlashInfer 论文 | flashinfer-paper.txt：无 cascade 一词（grep 0 命中）✓；BSR/ragged（L287）、Fig 2 (Br=4,Bc=1)（L283）、Fig 3 (3,1)+(1,1)（L289）、composable 无数据移动（L295）、attention state ⊕ canonical/结合交换/GEMM 类比（L248-263，§2.2）、Stream-K 去 atomic/确定归约序（L367）、短 KV 直写 final buffer（L1462 附录）、FA2≤sm89/FA3 Hopper（L299）、JIT functor（L325-331）、plan CPU→async workspace/persistent kernel/跨层复用（L429/502）、CG 固定 grid/offset（L433）、e2e n=4 峰值四数+MLC-Engine/Llama-3.1/ShareGPT/rate 16（L548-550，**§4.4**） | 3.1 §4 全 ✓；节号误引 S3-1 |
| GPTQ 论文 | gptq.txt：H_F=2X_F X_F^T（L215 逐字）、OBS 更新式 (2)、降阶式 (3)、O(d_row·d_col³)（L230）、ResNet-50≈1h（L229）、任意序动机（Step 1）、lazy batch 访存瓶颈+数量级加速（L249/263）、indefinite/几 B 几层（L268-270）、dampening 1%（L272）、Cholesky 行 q/sqrt 差（L274-275）、Algorithm 1（L281-300）、复杂度 max/min 式（L243 逐字）、128×2048 C4（L321）、4 GPU hours（L131/155）、单 A100-80GB（L320）、表 3 OPT-66B full 9.34/RTN4 110/GPTQ4 9.55、RTN3 1.3e3-1.3e4、「very stringent accuracy metric」（L155） | 3.2 §3/§8 全 ✓ |
| GPTQ repo | gptq/README.md L23/40：act-order 9.55→9.34、3bit 14.16→9.95、LLaMA-7B 7.15→6.09（act-order+true-sequential）、static-groups；gptq.py L90 argsort(diag(H))、L61 blocksize=128 percdamp=.01、L131 quantize(scale,zero,maxq)、L137 块内更新 | 3.2 §3.3/§3.4 ✓（切片记法 S3-14） |
| AWQ 论文 | awq.txt：Δ=max(\|w\|)/2^{N−1}（L334）、式 (2) Q(ws)(x/s)（L340）、RoundErr~U[0,0.5] 均值 0.25/Δ'≈Δ/fp16 无量化误差（L344-345）、误差比 (Δ'/Δ)(1/s)（L355）、表 2 全数（L253-327）、图 2 43.2→13.0 INT3-g128（L192）、表 1 RTN 23.54/act-1% 11.39（L236-260）、式 (4) 目标/式 (5) s=s_X^α（L408-424）、grid [0,1] size 20+clipping（L426/597）、group 128/INT4-INT3/Pile（L597）、GPTQ 过拟合/reorder 评述（related work）、图 3 W4A16 强度 4×（L404） | 3.2 §1/§4 ✓；S2-1（Δ 约定混用）、S2-9（基线冲突） |
| SmoothQuant 论文 | smoothquant.txt：outlier ~100×（L223）、>70 通道/通道内方差小/权重平坦（L213）、2^8·m_i/m 与 2-3 级（L225）、per-channel 硬件论证外维 T/Co（L270-276）、式 (3) 迁移/式 (4) s_j（L283-305）、α=0.5 well-balanced（L307）、GLM-130B ~30% α=0.75（L309/523）、Fig 10 消融 0.4-0.6（L932-934）、530B 单 8-GPU 节点/OPT-175B 半 GPU 更快（L186）、表 6/7 lossless W8A8 LLaMA/Llama-2/Falcon/Mistral/Mixtral（L596/620）、GLM MMLU/MNLI/QNLI/LAMBADA 因训练集（L511）、官方 repo 摘要自链（refs 已录） | 3.2 §5/§8 ✓；S2-3（数值例）、S3-2/3（指针） |
| LLM.int8 论文 | llm-int8.txt：6.7B 相变/全层/75% 序列维/20×（L134）、150k outliers per 2048-token seq **13B**（L329 逐字，章引 §4 口径正确）、6 维集中/|O|≤7（L134/228）、置零 top-1 降 >20%/PPL 600-1000%/随机 ≤0.3%/~0.1%（L134/374）、判据 ≥6.0/≥25% 层/≥6% 序列维（L334）、式 (7) vector-wise 外积反量化/式 (8) 分解（L212/224 逐字）、BLOOM-176B 1.96×（L202）、125M-175B 保 16-bit（摘要）、唯一良好 scaling（L247） | 3.2 §6 全 ✓ |
| Marlin | marlin/README.md L7-14（16-32 vs 1-2、FLOP:byte 100-200、25-50 MAC）、L41-44（optimal 3.87x+0.125 bit）、L65（≥8.0 Ampere/Ada、Hopper 未优化）、L85（marlin.Layer.pack）；arxiv-marlin.xml 摘要（64-128 gradually decreasing but still significant、vLLM e2e up to 2.8×） | 3.3 §2 全 ✓ |
| NanoFlow | arxiv-p3p4-batch.xml 2408.12757v2 摘要：「end-to-end LLM serving is compute bound for most common workloads and LLMs」逐字、nano-batch+复制算子、自动搜索 number/size/ordering/allocation + interference 逐字、1.91×、「achieving 50% to 72% of optimal throughput across popular models」逐字；Nanoflow/README.md L79（68.5% of optimal、59%-72% across ported models）；repo gemv/ pipeline/ 目录存在 | 3.4 §7 全 ✓（摘要为准 + README 另注口径正确） |
| TileLang | 2504.17577v2 摘要：tile DRAM↔SRAM 数据流、scheduling space（thread binding/layout/tensorize/pipeline）与 dataflow 解耦、annotations+primitives | 3.4 §5 ✓ |
| batch-invariance 博客 | iw1 batch-invariance.txt：三类 reduction 算子（L93）、RMSNorm data-parallel/split 破不变（L108-109）、matmul 2D tile/split-K/stream-k 破 position-invariance（L113-117）、padded TC（L121）、固定 split-size（L139-140）、Performance 表 Qwen-3-8B/1000 seq/90-110/26-55-42 s（L151-152）、true on-policy RL（L154-156） | 3.4 §6 全 ✓；55/26=2.1×、42/26=1.6× ✓ |

### 6.3 repo / doc / config 锚点（tarball，2026-11-23 同日）
| 项 | 位置 | 结论 |
| --- | --- | --- |
| FlashInfer cascade.py | merge_state L42 / .to(float32) L90-91 / merge_states L170 / MultiLevelCascadeAttentionWrapper L226 / plan L401 / run L537 | 3.1 引全部行号 ✓ |
| FlashInfer scheduler.cuh | gdy=num_kv_heads L172、occupancy API L185、max_grid_size L187、batch*gdy≥max→false L188-189、二分下界 max(128/page_size,1) L205、new==batch&&!CG→false / CG 恒切 L206-211（注释逐字） | 3.1 §2 代码摘录忠实 ✓ |
| FlashInfer decode.py/utils.py/jit/core.py/prefill.py | Wrapper L1338 / plan L2179 / run L3237；NHD=0/HND=1 L57-58；gen_jit_spec L760；maybe_packed_custom_mask 存在 | ✓ |
| FlashMLA deep-dive + README | deep-dive L11-15（I 公式/865/3.35/128/无 TP）、L21（32768/65536）、L25-42（seesaw/scale 定义/K=V 逐字/mathematically equivalent）、L54（9 TMA 64×64）、L57（80% TC/3 TB/s/~2%）、L60-61（PDL/tile scheduler）、L3（660）；README L3（ba89a34 自链/2026.09.30 移除）、L41（fused saves overhead/1460/950 B200 CUDA 13.3）、L151-153（indices shape/bake/None/−1） | 3.1 §5/§7、3.4 §2 ✓；scale 译注裁定见 §3 |
| DeepSelect README + deep-dive | 2~20×/topk≤4096 主优化 512/bf16 indexer/fp32 sampling vocab 129280/Ascend 仅 bf16/有效带宽逐字/stride requirement/return_value=False ~10%（L91）；deep-dive 三步 scan-filter-compact/随机块序/阈值不降/每元素恰读一遍/O(k+B+B2)/L=k+B+B2 定义/E[A_i]≤L/i/W=A+kR/RB2≤A/定理 (1+k/B2)LH_m/O(k log(N/k))/k=512 推荐 B=B2=1024 | 3.1 §7 式 (9) 与证明骨架、3.3 §7 ✓；数值 1.5×2560×5.433=20863≈16%T ✓ |
| DeepInfer doc | L269-275（Where is FP4 used? 主 KV per-16 e4m3 类 NVFP4 无 global/indexer per-32 ue8m0 MXFP4/routed-expert per-32 ue8m0 MXFP4 逐字）、L285（Mega MoE scale-up 通用解逐字）、L289（SASS 5%-10% 逐字）+L291（e2e at most 5%）、L293（beyond 32 问句）、L295/L297（EP 强度/NVLink 干扰逐字）、L334（SGLang/NanoFlow 异步逐字） | 3.1 §5/§7、3.2 §7、3.3 §3/§5/§6、3.4 §7 引文全部逐字 ✓ |
| DeepInfer candidate_blocks.py / attn.py / tilelang_ops | docstring +inf 强制末块、block_size∈(8,16)、1024B 对齐、restore_candidate_indices_inplace、block_starts=candidate_blocks*block_size；DEEPINFER_BATCH_INVARIANT attn.py L89/666/1002 消费；tilelang_ops/ 含 candidate_blocks/compress/norm/fused_sample + cuda/asc 双后端 dispatcher | 3.1 §7、3.4 §5/§6 ✓ |
| DeepGEMM README + runtime.hpp | DeepJIT 安装期无 CUDA 编译、D=C+A@B/NT（SM90 仅 NT/SM100 全）、LHS SF TMA-aligned transposed、SM90 FP32 SF/SM100 packed UE8M0 4/int、transpose/cast 用户自理或融前置、M 轴 group/contiguous 对齐 get_mk_alignment/masked CPU 不知数+DeepEP low-latency 输入、Mega MoE mega-kernel dispatch+linear1/2(FP8xFP4/FP8xFP8)+SwiGLU+combine/symm memory/PyTorch≥2.9/fp8_fp4_mega_moe/transform_weights/get_symm_buffer、weighted ReLU MQA logits non-paged/paged、$HOME/.dj+DG_JIT_CACHE_DIR（L172）；runtime.hpp L12 kLegacyMKAlignmentForContiguousLayout=128 | 3.3 §3/§4/§5、3.2 §7 ✓ |
| DeepEP V2.5 README | V1/NVSHMEM/旧文档全移除、EPBuffer 统一、NCCL Gin、DeepJIT、SM/QP 解析免 auto-tuning、hybrid/direct、do_cpu_sync True/False 语义+handle.psum_num_recv_tokens_per_expert（L162 逐字）、零 SM Notes（L39 逐字）、lb_prefetch_weights/lb_reduce_grads NVLink 权重+scale/FP32 梯度/MoonEP/UltraEP、FP8 dispatch | 3.3 §4/§6 ✓ |
| vLLM flash_attn.py | build L909、block_table L1063、max_num_splits L952（注释「0 means use FA3's heuristics, not CG compatible」逐字）/L962 固定上界/L964-965 批不变=1、buffer 注释 [num_splits,num_heads,num_tokens,head_size] L959-960 逐字、forward L1284、varlen 调用 L1583 起 return_softmax_lse=True L1599 | 3.1 §6 ✓ |
| vLLM backends 目录 | flash_attn/flashinfer/flex_attention/triton_attn/cpu_attn + mla/{flashmla,flashinfer_mla,cutlass_mla,indexer,sparse_swa} 全在 | 3.1 §6 ✓ |
| vLLM config/compilation.py | CompilationMode L38（NONE=0/STOCK_TORCH_COMPILE=1/DYNAMO_TRACE_ONCE=2/VLLM_COMPILE=3 + 四件 docstring 逐字）、CUDAGraphMode L165（FULL_DECODE_ONLY=(FULL,NONE)/FULL_AND_PIECEWISE=(FULL,PIECEWISE)）、v1 default FULL_AND_PIECEWISE（L744）、FULL_DECODE_ONLY P/D 省内存 docstring、piecewise 依赖 VLLM_COMPILE+非空 splitting_ops/full 有无编译均可（docstring 逐字）、decode_cudagraph_capture_shapes L54 pad docstring 逐字、set_splitting_ops_for_v1 L1344 默认 _attention_ops、cudagraph_copy_inputs、compile_sizes L704、pass_config.fuse_attn_quant L244 | 3.4 §3/§4 ✓ |
| vLLM determinism/ | batch_invariant.py + batch_invariant_configs.py（无 batch_invariant_ops 目录）、Marlin bypass docstring L1201 逐字、dequantize_gptq_weight L1156、CUBLAS_WORKSPACE_CONFIG=":16:8" L1423/CUBLASLT_WORKSPACE_SIZE="1" L1424、禁 TF32 L1531；v1/sample/+ops/ | 3.3 §2 坑、3.4 §6 表 ✓ |
| SGLang | batch_invariant_ops.py _SEGMENT_COUNT=4 L273 + 注释 L267 逐字、rms_norm/fused_add_rms_norm_batch_invariant、get_batch_invariant_attention_block_size L1473；srt/compilation/{cuda_piecewise_backend,backend,compilation_config,compile_phase,npu_piecewise_backend}.py 全在 | 3.4 §3/§6 ✓ |
| TileKernels README | dozens/近硬件极限/内部训推已用、MoE routing top-k/scoring、per-token/block/channel FP8/FP4 cast/dequant+fused SwiGLU+quant、Engram gating fused RMSNorm fwd/bwd、mHC Sinkhorn、RoPE（L17）、2026-09-30 Ascend 双后端运行时自动选择 | 3.4 §2/§5 ✓ |
| V4.1 config（v41-config-w2.json，hf-mirror 传输） | quantization_config={fp8,dynamic,[32,32],ue8m0,fp4} 逐字；text_config：40 层/5120/384 routed/top-6/2304/129280/index_topk 512/index_n_heads 32/index_head_dim 128/head_dim 512/qk_rope 64/nextn 3/max_position 1048576 | 3.1 §7、3.2 §2/§7、3.3 章脊/§6/§7 ✓ |

### 6.4 跨维度 recap 对照（本书已交付章）
| 项 | 来源 | 结论 |
| --- | --- | --- |
| models 4.1 式 (11)-(17)、§5.2 logit 例、32832/57×/192→576、§4.4 代价三笔 | models/04-modern-architectures/01-kv-compression-and-exact-attention.md L169-214/227-250/284-299 | 3.1 recap 忠实 ✓（裸 d S3-4、合并矩阵指针 S3-5） |
| models 2.4 前缀不可变命题 | models/02-transformer-anatomy/04-causality-and-kv-cache.md L112 | 3.1 命题 2 依据 ✓ |
| models 4.2 §6 indexer（式 (12) ReLU 加权/头间共享/K=2048/144 MB vs 9.2 GB）/§8 召回 | models/04-modern-architectures/02-sparse-attention.md L193-237/262 | 3.1 §7 recap ✓ |
| models 2.6 SwiGLU | models/02-transformer-anatomy/06-ffn.md | 3.4 前置指针 ✓ |
| training 1.3 §1 表/§6（含 E5M6） | training/01-autograd-and-backprop/03-mixed-precision.md 全文 | 3.2 FP8 recap 忠实 ✓ |
| training 5.4 §6（IB 50/NVLink 160/两阶段同序号/20 SM/重叠） | training/05-model-parallelism/04-expert-parallelism.md L93 | 3.3 §6 recap ✓ |
| inference 1.2（κ 式 (3)/f 式 (2)/强度式 (4)/B_w* 式 (6)/MoE 式 (10)-(13)/654 GB/16.1 GB/15.1 ms/66.5 ms/reconciling 注） | inference/01-principles-and-accounting/02-prefill-decode-accounting.md 全文 | P3 各章脊与 recap 一致 ✓（4.81 非 1.2 算例，S3-7） |
| inference 1.3 scale-up domain / 1.4 §6 模式表 +「3.4 量化」/ 2.1 式 (3)+§5 / 2.3 式 (3)(4)+§4 | 各章 grep 定位 | 指针与口径 ✓ |

### 6.5 独立数值复算清单（全部通过，除已列 S2）
3.1：8/108=7.4%、112 CTA、9363 段长、451.5 KiB/0.086%/0.003%、160 μs×32=5.13 ms、2.1× 上界、数值例逐位、9 GiB→1.25 GiB/86.1%、NHD stride 2048 B/HND 16 KiB/页 128 KiB、mask 32→4 KiB、241.8/258.2/94%/76%、22.5 MiB/5.625 GiB/0.39%/1.25 GiB/56.9×/340 MiB/15×/3.76×/256×、20863/16%/H_128=5.433、8208=513×16。
3.2：16,777,216 参数/32 MiB、8.25/8.5 MiB、3.88/3.76×、3.125%/6.25%、MXFP8 1.03125/16.5 MiB/1.94×、MXFP4 0.53125/8.5 MiB/3.76×、NVFP4 0.5625/9 MiB/3.56×、V4.1 块 1.00098/16.02 MiB、0.024%/0.098%/4×、expert 1.00024→0.53125、toy H/H⁻¹/0.1832/0.1032/43.7%/[1/4]、复杂度 4096×、练习 2（WX=5/Q(W)X=6/scaled 4.75）、命题 4 全代数、s=10/31.6/100 与 10/3.16/31.6（**级数账 FAIL→S2-3**）。
3.3：4.125 bit/3.88/3.56/3.94、B_max 76、10.0/2.58/0.54/4.34 μs/2.31×、FP8 5.01/8.68/17.4 μs/295 对消、scale 4096 B/0.024%、pad 127/0、75%（**66.5% FAIL→S2-7**）/11.0%/M_blk/(2μ)、29.4/0.92 MB/32×、A=195.2 ms/e*=32/表五行、58.7 MB/6.81 GB/0.92 MB/1.02 μs、1.97 MB/30.7 KB/0.61 μs、33.1 MB/9.88/49.4/39.5 μs。
3.4：0.5 MiB/1.05 MB/0.31 μs/20 μs/0.13%/320 μs/2.1%/16×、33.6/67.1 MB/20 μs/1.28 ms/1.9%、1.84/3.67 MB/1.10 μs/35.1 μs、117.4/234.9 MB/70.1 μs/2.24 ms/3.4%、residual 5S→4S/0.16 μs、330×5=1.65 ms/10.9%/34.3%/1.645 ms/3.2 ms、32 block/24%/128 block/97%、26/55/42→2.1×/1.6×、12/6/2× 流水。

## 7. 修复指引（给 fix round）

- 3.2 优先：S2-1/2（量化定义式统一约定）、S2-3（级数账 + 练习 4 锚）、S2-4/5（`\!` 删除、abs_max 记法统一，改后重跑 lint + flanking 扫描）、S2-9（AWQ 基线口径注）。
- 3.1：S2-6（命题 4 加互异前提 + 第二 clause 重写）、S2-8（式 (4) 删 ·bytes）。
- 3.3：S2-7（μ=32 均值账改精确 75% 读数 + 均值式限定大 μ + 练习 2 锚）。
- S3 十四条按 §6.4 下波清零；S3-1/2/3/5/7 为指针类可随手修。
- 含 S2 章（3.1/3.2/3.3）修复后由 evaluator 复审对应条目；3.4 无 S2，S3 随下波。
