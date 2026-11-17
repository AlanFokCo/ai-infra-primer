# training W4 审查报告 · 批 2（7.4 / 7.5 逐章 + 全波 5 章跨章 sweep）

日期：2026-11-17。审查人：独立 fresh-context evaluator（批 2，与写作 agent、批 1 evaluator 无共享上下文）。
对象：`training/07-pretraining/04-checkpointing-and-resumption.md`（7.4）、`05-long-context-and-mid-training.md`（7.5）；跨章 sweep 覆盖 7.1–7.5 全 5 章。
立场：假设内容有错，直到一手来源证明无误。本批回源文本：arXiv:2407.21783v3（herd 全文 HTML）、2412.19437v1（DSv3 全文 HTML）、2501.15383v1（Qwen2.5-1M 全文 HTML）、2508.06471v1（GLM-4.5 全文 HTML）、PyTorch 2.9/2.15 官方文档（DCP、RNG API）、GitHub API（repo 路径实测）、arXiv abs（2309.00071）。

## 0. 结论

- **S1 = 0，S2 = 1，S3 = 5。**
- **7.4 放行**（无 S1/S2；两条 S3 为措辞级，按协议累积处理）。
- **7.5 有条件放行**：S2-1（§2 判据 3 的 tokens 量级带与本章自家锚点冲突）属协议「必须修复」项，修一行措辞即关闭；其余全部回源核实通过。
- 波末 sweep 通过：四组跨章指针（1.5↔7.2、2.x↔7.2、6.2↔7.4、5.2↔7.5）内容一致性全部核实；批 1 审查报告在 docs/reviews/ 尚不存在（无 p7a 文件），无继承 S3 待清零项，本报告 S3 即全波汇总。

## 1. Findings 表

| # | 级别 | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S2-1 | S2 | 7.5 §2 判据 3 + §2 ASCII 表 mid-training 行 | 「mid-training 以十亿至千亿 tokens 计」（ASCII 表作「量级 10^9-10^11 tokens」）——量级带上沿与本章自家锚点冲突 | 本章 §1 与 §2 末段把 Llama-3 的 §3.4.2 长上下文段（约 800B tokens = 8×10^11，herd §3.4.2 原文 "approximately 800B training tokens"，已回源）与 DSv3 扩窗（约 126B = 1.26×10^11）明确列为 mid-training 的同构对应物，两者均超「千亿 = 10^11」上沿；判据的区分功能（对 SFT 百万样本级）不受影响，但数字带失真 | 改为「以十亿至数千亿 tokens 计（10^9–10^12 下沿；本书实例：DSv3 约 126B、Llama-3 约 800B）」，ASCII 表同步 |
| S3-1 | S3 | 7.4 §2 变体注记 (ii) | 「fp32 master 与工作参数同时在场时二者必存其一可推另一个」 | 方向不对称：master→bf16 的 round-to-nearest 是幂等的（bf16 本就是每步从 master 舍入得到，1.3 式 (2)），精确可推；bf16→master 则低位不可恢复，不精确。注记结论（生产栈通常都存）正确，论证措辞过强 | 收窄为「存 master 即精确推出工作参数（1.3 式 (2) 的舍入幂等）；反向不可精确恢复——生产栈通常都存」 |
| S3-2 | S3 | 7.4 §6 式 (3) 前 | 「故障率 λ = 1/MTBF」定义后全文未再用 λ | 式 (3) 直接写 MTBF；死符号 | 删 λ 定义或在式 (3) 用 λ 表示 |
| S3-3 | S3 | 7.5 §3 式 (1) 符号注 | 「该章记作 d，本章按 training 约定改写为 h」 | training 维度的 h 约定实为 hidden（5.1/5.2/1.4 的 h = hidden）；本章 h = H_q·d_head 是 models 4.6 的 d 的换名。MHA 下二者数值相等（如 8B：32×128=4096），不产生数值后果，但「training 约定」措辞不确 | 改为「该章记作 d；本章改用记号 h 以免与窗长符号混排（MHA 形态下与 hidden 同值）」之类 |
| S3-4 | S3 | 5.2 全章（跨章 sweep 项） | 5.2 无指向 7.5 的前瞻指针 | 7.5 §3 的「CP 启用判据（单卡激活账 / 单步时延账）+ tokens/step 恒定下的序列数反比缩」是 5.2 未覆盖的增量，5.2 §6 只到「CP 服务超长上下文（CP16 @ 128K）」。spec §5 对 7.5 只要求「回链 5.2」单向，故不构成违规，属可选增补 | （可选）5.2 §6 边界段补一句「CP 启用时机与扩窗成本账归 7.5」；判定：接受不修亦可 |
| S3-5 | S3 | 7.3 §2.4 口径注记（波末 sweep 裁定，供 controller 判定） | herd 58.7% vs Table 5 计数和 64% 的注记可再补一刀根源 | 本次回源核定：Table 5 的 % 列 GPU 六行求和恰为 58.7%（30.1+17.2+4.5+4.1+1.4+1.4），与正文自洽；但 % 列首行 148→30.1% 隐含分母 ≈491，其余行均以 419 为分母（72/419=17.2% ✓ 等）——原表首行分母异常是差异根源 | （可选）注记补「58.7% = 原表 % 列 GPU 行之和；首行 % 的分母口径异于其余行」一句 |

无 S1。无历史叙事 / 营销词 / 模板句（rubric A 类全 5 章 grep 零命中，含 lint 词表外清单）。

## 2. 强制独立重推导记录

### 7.4-A：checkpoint 字节账与最优间隔

- 式 (1) 重推：bf16 混合精度 + AdamW 的续训态 = bf16 参数 2N + fp32 master 4N + fp32 m/v 各 4N = 14N bytes ✓（梯度 2N 为暂态不入档，与 4.2 的 16N 常驻账差恰为梯度项，口径相容）。变体 (iii) DSv3 BF16 矩估计版 2N+4N+2N+2N = 10N ✓（DSv3 §3.3.3 "Low-Precision Optimizer States" 回源核实）。
- 代入复算：405B → 5.67×10^12 B ≈ 5.7 TB ✓；16384 卡分片 3.46×10^8 B ≈ 0.35 GB ✓、8192 卡 6.92×10^8 ≈ 0.69 GB ✓（均落 herd「1 MB to 4 GB per GPU」带内，原文 "Checkpointing saves each GPU's model state, ranging from 1 MB to 4 GB per GPU" 回源）；5.67 TB / 2 TB/s = 2.84 s ✓；step 账 6×405e9×16e6 = 3.89×10^19 FLOPs ÷（16384×400e12）= 5.93 s ✓（Table 4 行回源：16384/TP8/CP1/PP16/DP128/16M tokens/400 TFLOPS/MFU 41%）；2.8/5.9 ≈ 半步 ✓。
- MTBF：54×86400/466 = 10,012 s ≈ 1.0×10^4 s ≈ 2.78 h ✓（466 = 47 + 419 回源）。
- 式 (3)(4) 重推：C(T) = δ/T + (T/2+R)/MTBF；C′ = −δ/T² + 1/(2·MTBF) = 0 ⇒ T\* = √(2δ·MTBF) ✓；C″ = 2δ/T³ > 0 极小 ✓；R 为可加常数不进最优 ✓；最优点浪费率 C(T\*) = 2·√(δ/(2·MTBF)) = √(2δ/MTBF) ✓。代入（MTBF=10^4）：δ=2.8 → T\*≈237 s、浪费率 2.37% ≈ 2.4% ✓；δ=0.3 → T\*≈77 s、0.77% ≈ 0.8% ✓。与经典 Young/Daly 一阶模型同形；章内标「本书自含推导」合规。

### 7.5-A：长上下文扩展 training 侧成本账

- 式 (1) 重推：φ_attn(W) = 2W²h/W = 2Wh（每 token 每层 fwd）✓，与 models 4.6 式 (1)（2T²d，d = H_q·d_head）同式换名 ✓。
- 式 (2) 重推：C_k = B_k·(2N_act + 2hLW_k) ✓（参数关联 matmul 含 QKVO 投影并入 2N_act，score 加权部分 2hLW 单列——分解正确）。fwd+bwd ×3 对两项同乘，推论不变 ✓。
- 交叉点：2N_act = 2hLW ⇒ W\* = N_act/(hL)；8B 形态 8e9/(4096×32) ≈ 6.1×10^4 ✓（与 models 4.6 式 (2) 一致）。
- DSv3 账复算：1920×32768 = 62,914,560 = 480×131072 ✓（tokens/step 恒等，原文 §4.3 回源：「sequence length 32K, batch size 1920」「128K, batch size reduced to 480」「each comprising 1000 steps」）；扩窗 tokens = 62.9M×1000×2 ≈ 1.258×10^11 ≈ 126B ✓；tokens 份额 126B/14.93T ≈ 0.84% ≈ 0.8% ✓；卡时份额 119K/2788K = 4.27% ≈ 4.3% ✓（Table 1：2664K/119K/5K/2788K 回源）；单价倍数 (119K/0.126T)/(180K/1T) = 945.9/180 ≈ 5.25 ≈ 「约 5 倍」✓。
- 练习 2 引导的下界复算（MHA 口径，h=128×128、L=61、N_act=37e9）：4K 每 token fwd = 82.2e9（attn 10%）、32K = 139.5e9（47%）、128K = 336e9（78%）；扩窗均值 / 4K = 2.89×——正文「单价差约 5 倍」高于此前缀 FLOPs 下界，缺口归因（MLA score 结构 + 长序列 kernel 效率）声明为不复算项，口径诚实 ✓。

## 3. 存疑项裁定

### task-22（7.4 相关 4 项）

1. **herd 58.7% vs Table 5 ≈64%**：回源裁定——v3 原文内部不自洽成立：正文 "GPU issues are the largest category, accounting for 58.7% of all unexpected issues" 逐字在案；Table 5 GPU 六行计数和 268、268/419 = 64.0%；但原表 % 列 GPU 行之和恰为 58.7%，首行（148 → 30.1%）分母与其余行（均以 419 为分母）不一致。**7.3 §2.4 的「按原文各自照录、不做调和」注记处置正确**，本批维持；增补建议见 S3-5。7.4 只引 466/47/419、3 次人工干预、>90% 有效训练时间，全部与原文一致 ✓。
2. **异步 staging 暴露 0.3 s（本书估计口径）**：标注合规；量级合理（0.35 GB D2H 原始拷贝为十毫秒级，0.3 s 为含 staging 开销的保守上限，且只用于展示 T\* 对 δ 的敏感性）✓。
3. **最优间隔式 (4) 无出处**：本批已独立重推（§2 上），公式正确；「本书自含推导、经典结论形式」的标注符合规范（不虚构出处）✓。
4. **RNG 行为随实现而变**：正文已带「审计自己的栈」纪律注记；`torch.get_rng_state()` "for the default generator on CPU only"、`torch.cuda.get_rng_state_all()` "Return a list of ByteTensor representing the random number states of all devices" 均回 PyTorch 2.15 官方文档逐字核实 ✓。

### task-23（7.5 全部 6 项）

1. **DSv3 扩窗 ≈126B 为换算口径**：算术复算通过（§2 下），文中已标「换算口径」✓。
2. **单价差约 5 倍**：复算 5.25× ✓；非 FLOPs 因素的缺口已声明不复算并配练习 2 下界（2.89×）✓。
3. **三家 CP 配置未披露**：本批对三份 TR 全文 grep 独立复核——GLM-4.5 与 Qwen2.5-1M 全文零命中 context/sequence parallelism；DSv3 唯一 "Sequence Parallelism" 命中在 §3.4.1 **推理部署** prefilling（TP4+SP），非训练侧——「训练侧 CP 配置未披露」成立 ✓。
4. **Qwen2.5-1M §3 为 run-in 无编号标题**：回源确认（"3 Pre-training / Natural and Synthetic Data. / Training Strategy." 均为 run-in 头），「§3 Training Strategy 段」写法准确 ✓。
5. **herd 40M（§3.4.3）与 40B（§3.1.3）是两数**：回源确认——§3.4.3 "final 40M tokens... linearly annealed the learning rate to 0, maintaining a context length of 128K... Polyak (1991) averaging"；§3.1.3 "annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens... 30% weight to the new dataset" ✓。「CP16 @ 128K」与 5.2 的登记（Table 4 第三行 TP8×CP16×PP16×DP8 @ 131072）一致，本批复核 herd Table 4 原文 ✓。
6. **五段自述 vs 四档口径**：回源确认——"progressive context length expansion strategy, which includes five stages. The first two stages are similar to those of other Qwen2.5 models... initially trained with a context length of 4096 tokens, and then... 32768"；后三段 65,536/131,072/262,144、ABF base 1,000,000 / 5,000,000 / 10,000,000 ✓。7.5 四档表 + 五段注记准确 ✓。

## 4. 波末 sweep（全 5 章跨章一致性）

### 4.1 指针双向性矩阵

| 指针对 | 正向 | 反向 | 内容一致性 | 判定 |
| --- | --- | --- | --- | --- |
| 1.5 ↔ 7.2（tokens 口径） | 1.5 式 (5) 处与总结表均前瞻 7.2 ✓ | 7.2 §2/§3 回链 1.5 式 (5) ✓ | B_tok = b·m·d·T 两章同式 ✓ | 通过 |
| 2.x ↔ 7.2 | 2.2 §7（「全字段对照表归 7.2」）、2.3 §6（节题即「配方规律与 7.2 预告」）、2.5 §8（「配方对照归 7.2」）✓ | 7.2 回链 2.2 式 (4)/§2/§6/§7、2.3 式 (7)/§3/§5/§6、2.5 §4 ✓ | wd=0.1×lr、β 未披露纪律、Moonlight 非 µP 纪律三处口径逐字相符 ✓ | 通过 |
| 6.2 ↔ 7.4 | 6.2 §5 节题带「前瞻 7.4」、metadata 带「衔接 7.4」✓ | 7.4 metadata/§3 回链 6.2 §5 ✓ | dist-ckpt 的 ShardedTensor 元数据、保存/加载网格解耦、只读元数据接口（load_sharded_metadata）两侧描述一致；7.4 不重复机制走读 ✓ | 通过 |
| 5.2 ↔ 7.5 | 无 5.2→7.5 前瞻指针（S3-4，可选） | 7.5 回链 5.2 §1/§3-§6 ✓ | zigzag 即 herd §3.3.2 的「i-th 与 (2·CP−1−i)-th 两块」配对（本次回源逐字确认，7.5 的同一性断言成立）；CP16 @ 128K 与 herd Table 4 第三行一致 ✓ | 通过（带 S3-4 可选项） |
| 7.3 ↔ 7.4（附加） | 7.3 §2.4/§5.2/§5.3 前瞻 7.4 ✓ | 7.4 metadata/§1/§5 回链 7.3 §2.4/§3/§5.3 ✓ | 中断统计同一组数（466/419/47、78%、58.7%、3 次、>90%）两章一致 ✓ | 通过 |
| 7.2 ↔ 7.5（附加） | 7.2 §3 末「长上下文的成本账归 7.5」✓ | 7.5 §4 回链 7.2（Qwen3 不披露先例）✓ | GLM-4.5 的 23T、cosine 2.5e-4→2.5e-5 两章一致 ✓ | 通过 |

跨维度指针（纯文字指引纪律）：7.5 → models 4.6 四处（§1.1 式 (1)(2)、§2.3 YaRN 公式本体、§2.5 DCA、§4 配方本体、§5 claimed vs effective）逐一核对存在且内容相符（YaRN 温度式 √(1/t)=0.1·ln s+1 在 4.6 式 (14)、DSv3 §4.3 原文同式，三方一致）✓。1.4（checkpoint RNG 语义）、4.2（16N 账）、4.3（FSDP 全分片语义）、3.2（completion masking）、models 1.2 式 (8)（优化器态 (4+2c)N）全部回读确认 ✓。9.1 / P8-P9 为未写章的合规前瞻（Part 级指针，README 已登记计划）。

### 4.2 术语与符号一致性

- spike/突刺、tokens 口径、annealing、mid-training、CP/TP/PP/DP 度数符号（c/t/p/d）全波一致 ✓。
- 符号局部复用两处已登记：7.5 的 h = H_q·d_head 与 5.x 的 h = hidden（S3-3）；7.4 的 T = 存档间隔（秒）与 1.5/7.1 的 T = 序列长——均局部定义、无跨章混排，不另立案。
- 「未披露」入表纪律：7.2（Qwen3 整行）、7.5（三格 CP 未披露）一致执行 ✓。

### 4.3 S3 清零汇总

批 1（p7a）报告文件在 docs/reviews/ 尚不存在，无继承 S3；本批 S3 共 5 条（§1 表 S3-1 至 S3-5），按协议累积至下一波统一判定。其中 S3-4、S3-5 为「接受不修亦可」的可选增补，S3-1/S3-2/S3-3 为措辞级微调。

### 4.4 规范审查（rubric A/B/C，全 5 章）

- A 类（模板句/营销词/松散词/历史叙事）：lint 词表 + rubric A.1 扩展词表全 5 章 grep 零命中 ✓。
- B 类：「显著」5 处命中逐条人工核——7.3/7.4 的「显著人工干预仅 3 次」为 herd 原文 "significant manual intervention" 的直译、7.1 两处带消融语境、7.1 一处为 FineWeb 基准选取判据——均有数值/出处支撑，不立案。超长句抽查未见三层嵌套失控。术语漂移无（除已登记的 h/T 局部符号复用）。
- 每节以事实/公式/问题开场：7.4 §1 定义句+表格、§2 公式、§3 问题（网格不同）、§4 账、§5 标准动作、§6 统计数；7.5 §1 两个实测数、§2 原文标题、§3 公式、§4 对照事实——合规 ✓。
- 模板齐全（metadata/总结表/练习/参考）与篇幅：7.4 = 3584 CJK 字（≥3500 标准章 ✓）、7.5 = 2743（短篇 ≥2500 ✓）；ASCII 图 7.4 两张、7.5 一张 ✓。

## 5. Verification log（抽查项 × 来源 × 结论）

来源代码：H = herd arXiv:2407.21783v3 全文 HTML（2026-11-17 curl）；D = DSv3 arXiv:2412.19437v1 全文 HTML；Q = Qwen2.5-1M arXiv:2501.15383v1 全文 HTML；G = GLM-4.5 arXiv:2508.06471v1 全文 HTML；P = PyTorch 官方文档（docs.pytorch.org 2.9/2.15）；GH = GitHub API；A = arXiv abs 页。

| 抽查项 | 章内位置 | 来源 × 原文要点 | 结论 |
| --- | --- | --- | --- |
| 存储 fabric 240PB/7500 SSD 服务器/2TB/s 可持续/7TB/s 峰值 | 7.4 §2 | H §3.3.1："offers 240 PB of storage out of 7,500 servers equipped with SSDs... sustainable throughput of 2 TB/s and a peak throughput of 7 TB/s" | 一致 ✓ |
| 每 GPU checkpoint 1MB–4GB、"highly bursty"、"minimize GPU pause time" | 7.4 §2/§4/metadata | H §3.3.1："ranging from 1 MB to 4 GB per GPU"、"highly bursty checkpoint writes"、"minimize GPU pause time... increase checkpoint frequency" | 逐字一致 ✓ |
| Table 4：16384 卡 TP8/CP1/PP16/DP128、16M tokens、400 TFLOPS、MFU 41%；CP16 行 | 7.4 §2；7.5 §3；5.2 | H Table 4 三行原表（另两行 8192 卡与 16384/CP16/DP8 @ 131072/380 TFLOPS/38%） | 一致 ✓ |
| 466=47+419、约 78% 硬件、GPU 58.7%、3 次人工、>90% 有效训练时间、单 GPU 故障可重启全作业 | 7.4 §6；7.3 §2.4 | H §3.3.4 正文逐句 | 一致 ✓（58.7% 口径差裁定见 §3 task-22-1） |
| Table 5 行项（148/72/54/35/32/19/17/6/6 及 % 列） | 7.3 §2.4 表 | H Table 5 全 18 行提取 | 行项逐行一致；% 列首行分母异常（S3-5）✓ |
| flight recorder（元数据+stack trace 入 ring buffer、watchdog 超时自动 dump）、straggler 拖慢数千 GPU、数十 MW 功率摆动、昼夜 1-2% | 7.4 §6 | H §3.3.4："captures collective metadata and stack traces into a ring buffer"、"automatically dump tracing data on NCCLX watchdog or heartbeat timeout"、"a single straggler can slow down thousands of other GPUs"、"tens of megawatts"、"diurnal 1-2% throughput variation" | 一致 ✓ |
| 六阶段 8K→128K 约 800B、quadratic 动机句 | 7.5 §1/§2/§4 | H §3.4.2："increased context length gradually in six stages... approximately 800B training tokens"；"we do not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length" | 逐字一致 ✓ |
| annealing 末 40M tokens、LR 线性归 0、保持 128K、Polyak 平均 | 7.5 §2/总结 | H §3.4.3 逐句 | 一致 ✓ |
| §3.1.3 数据试纸：50% 8B + 40B tokens + 30% 新数据 | 7.5 §2 | H §3.1.3 逐句 | 一致 ✓ |
| CP 机制：2×CP 块、rank i 取 i 与 2CP−1−i、"up to 128K in length" | 7.5 §3 | H §3.3.2 逐句——即 zigzag 配对 | 一致（7.5 的 zigzag 同一性断言成立）✓ |
| 主阶段 15.6T 全程 8K 窗 | 7.5 §1 | H §2："pre-train a model with 405B parameters on 15.6T tokens using a context window of 8K tokens" | 一致 ✓ |
| DCP save SPMD / ShardedTensor / DTensor / 每 rank 存本地分片 / Stateful 钩子 | 7.4 §3 | P DCP 文档 save 签名段逐句 | 一致 ✓ |
| load-time resharding（"saving in one cluster topology and loading into another"） | 7.4 §3 | P DCP 文档首段 | 一致 ✓ |
| async_save 两段（de-stage 默认 CPU + 独立线程）、AsyncSaveResponse 双 future 语义、AsyncCheckpointerType THREAD/PROCESS、AsyncStager、experimental | 7.4 §4 | P DCP 文档 async_save 段与类文档逐句 | 全部一致 ✓ |
| SavePlan/WriteItem/LoadPlan/ReadItem 类名 | 7.4 §3 | P DCP 文档（94/12/99/5 次命中） | 存在 ✓ |
| repo 路径：torch/distributed/checkpoint/state_dict_saver.py、staging.py、megatron/core/dist_checkpointing/ | 7.4 §3/§4/参考 | GH contents API | 存在（19603 B / 19001 B / 13 项）✓ |
| RNG API：get_rng_state 仅 CPU 默认 generator；cuda.get_rng_state_all 逐设备 list | 7.4 §1 | P 2.15 函数文档逐句 | 一致 ✓ |
| Megatron consumed_samples 机制 | 7.4 §1 | GH：megatron/training/training.py 5 处 consumed_samples 命中 | 存在 ✓ |
| DSv3 Table 1：2664K/119K/5K/2788K H800 小时、180K/万亿 tokens | 7.5 §1/§2/§4 | D Table 1 与 §1 逐句 | 一致 ✓ |
| DSv3 §4.3 两阶段各 1000 步、1920×32K → 480×128K、LR 7.3e-6 "matching the final learning rate"、YaRN s=40/α=1/β=32/√t=0.1 ln s+1 | 7.5 §2/§3/§4 | D §4.3 整段提取 | 逐字一致 ✓ |
| DSv3 BF16 矩估计（§3.3.3 Low-Precision Optimizer States） | 7.4 §2 注 (iii) | D §3.3.3 节号结构 + "We adopt the BF16 data format instead of FP32 to track the first and second moments... without incurring observable performance degradation" | 一致 ✓ |
| DSv3 训练侧 CP 未披露 | 7.5 §3/§4 | D 全文 grep：唯一 "Sequence Parallelism" 命中在 §3.4.1 推理 prefilling | 成立 ✓ |
| GLM-4.5 §2.3 标题与定义引文、三段结构（repo 4K→32K/合成推理/长上下文+agent 32K→128K）、packing 换 best-fit、RoPE base 1e4→1e6、23T | 7.5 §2/§4 | G §2.3/§2.4/摘要逐句（"Mid-Training: Boost Reasoning & Agentic Capacity"、"we denote these training stages as mid-training"、"adjusted RoPE's base frequency from 10,000 to 1,000,000" 等） | 逐字一致 ✓ |
| GLM-4.5 LR cosine 2.5e-4→2.5e-5 "until the end of mid-training"；预训练两段上采样 §2.2 | 7.5 §2；7.2 表 | G §2.4/§2.2 逐句 | 一致 ✓ |
| Qwen2.5-1M 四档 32768/65536/131072/262144、ABF 1e6/1e6/5e6/1e7、75%/25%、三类合成数据（FIM/关键词与位置检索/段落重排）、五段自述 | 7.5 §3/§4 | Q §3 Training Strategy 段与 Table 2 逐句 | 一致 ✓ |
| Qwen2.5-1M 两阶段 SFT（≤32,768 短 → ≤262,144 混合、防遗忘平衡）+ DPO 类离线 RL | 7.5 §2 | Q §4 "Two-stage Supervised Fine-tuning." / "Reinforcement Learning." 段逐句 | 一致 ✓ |
| DCA+YaRN 为推理侧技术（"an inference technique"，§5.1 在 Inference and Deployment 下）；32K 训练 + DCA passkey @1M 超 80%；"training on longer sequences (up to 256k tokens) substantially improves..." | 7.5 §4 | Q §5.1 逐句 | 逐字一致 ✓ |
| YaRN 出处（2309.00071，Peng et al.）与温度缩放 | 7.5 §4/参考；models 4.6 链 | A abs 页 + 4.6 式 (14) 与 DSv3 §4.3 三方互证 | 一致 ✓ |
| 1.5 式 (5) B_tok、1.4 RNG 语义、2.2 式 (4) bias correction、2.3 §2 warmup 两论证与 RAdam ρ_∞=2/(1−β₂)−1、4.2 16N、4.3 FULL_SHARD、3.2 completion mask、models 1.2 式 (8)、models 4.6 §1.1/§2.3/§2.5/§4/§5 | 7.4/7.5 回链处 | 本书各章回读 | 全部存在且内容相符 ✓ |

**覆盖率**：7.4 的外部事实主张（数值/引文/API/repo 路径）24 项、7.5 的 32 项全部回一手来源核实，覆盖率 100%（无「确认无误但无来源」项）；强制重推导 2/2 完成（§2）；练习题的引导数值（7.5 练习 2 的 2.9× 下界、7.4 练习 1/3 的算术）抽查复算一致。

## 6. 机械检查

```
bash scripts/lint-chapter.sh training/07-pretraining/01-data-pipeline.md            → LINT OK
bash scripts/lint-chapter.sh training/07-pretraining/02-recipes.md                  → LINT OK
bash scripts/lint-chapter.sh training/07-pretraining/03-training-stability.md       → LINT OK
bash scripts/lint-chapter.sh training/07-pretraining/04-checkpointing-and-resumption.md → LINT OK
bash scripts/lint-chapter.sh training/07-pretraining/05-long-context-and-mid-training.md  → LINT OK
```

（2026-11-17 本批逐章实跑，退出码全 0。）

## 7. 交接事项（非章节缺陷，供 controller）

1. references.md「被引章节」补登（task-22 已报备、简报禁止 writer 修改）：pytorch-repo +7.4、megatron-lm +7.4、deepseekv3 +7.3、kimi-k2 +7.3、moonlight +7.3。
2. training/README.md Part 7 进度表五行仍为「未开始」，待 W4 关闭时统一更新。
3. S2-1 修复后本批即可关闭；S3×5 按协议入下一波清零清单。
