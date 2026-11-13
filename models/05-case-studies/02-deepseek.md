# 5.2 DeepSeek 谱系（The DeepSeek Lineage）

> DeepSeek 谱系是本书 Part 4 三条省算主线的共同落地现场：MLA（4.1）、DSA（4.2）、细粒度 MoE 与 aux-loss-free 路由及 MTP（4.5）——本章不做新推导，把 V2 → V3 → V3.1-Terminus → V3.2 的逐代架构增量对到账上，逐字段精读 DeepSeek-V3 config.json，并给成本数字一个经得起转述的口径 | 前置：1.2（ $6ND$ 与成本口径）、2.5（RoPE）、2.8（config 逐字段读法）、4.1（MLA）、4.2（DSA）、4.5（MoE 与 MTP）、4.6（YaRN） | 锚点：DeepSeek-V2 论文 [A]（arXiv:2405.04434）、DeepSeek-V3 论文 [A]（arXiv:2412.19437）与 HF config [A]（deepseek-ai/DeepSeek-V3）、DeepSeek-V3.2 论文与官方 repo deepseek-ai/DeepSeek-V3.2-Exp [A]（arXiv:2512.02556）

## 1. 谱系表：同一条架构轴上的四次增量

谱系起点取 V2——MLA 与 DeepSeekMoE 的首秀代。字段口径按 P5 约定（config 实测优先；total/active 照抄官方口径；ctx 列写 max_position_embeddings）：

| 模型 | total/active | layers | hidden | heads Q/KV | head_dim | attention | MoE | norm | vocab | ctx（max_position_embeddings） | rope_theta | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V2 | 236B/21B | 60 | 5120 | 128/128（名义） | 128+64 | MLA | 160+2 top-6，首 1 层 dense | RMSNorm（pre-norm） | 102400（HF config 口径） | 128K（摘要口径） | $10^{4}$ | [A] deepseekv2 |
| DeepSeek-V3 | 671B/37B | 61 | 7168 | 128/128（名义） | 128+64 | MLA | 256+1 top-8，首 3 层 dense | RMSNorm（pre-norm） | 129280 | 163840（YaRN f40，工作窗 128K） | $10^{4}$ | [A] deepseekv3 + deepseek-v3-config |
| DeepSeek-V3.1-Terminus | 同 V3 | 61 | 7168 | 同 | 同 | MLA | 同 V3 | 同 | 同 | 128K（续训起点） | $10^{4}$ | [A] deepseek-v3.2（起点口径） |
| DeepSeek-V3.2-Exp | 同 V3 | 61 | 7168 | 同（+indexer） | 同 | MLA + DSA | 同 V3 | 同 | 同 | 128K | $10^{4}$ | [A] deepseek-v3.2 |

脚注三条。其一，MLA 行的 heads Q/KV 写「名义」：MLA 不逐头存 KV，cache 是一份共享 latent（4.1 §4），config 里 num_key_value_heads = 128 只是与 num_attention_heads 同值的占位。其二，head_dim 列写 score 内维 128 + 64 = 192（qk_nope + qk_rope），v 侧 128 另计——2.8 §8「head_dim $\neq$ hidden/heads」的反例本尊：7168 / 128 = 56，与 192 彻底脱钩。其三，V2 行数字分两级：236B/21B、60 层、5120、160+2 top-6、128K、8.1T token 出自 V2 论文摘要与架构节（[A]）；vocab 102400 为 HF config 精度口径（论文 §3.1.1 只写 100K 量级）——V2 config 未纳入本书实测序列，该字段按 config 口径单独标注；V3.1-Terminus 与 V3.2-Exp 行的「同 V3」断言出自 V3.2 论文所述的续训起点关系（[A]），非独立 config 实测。

读表三件事。（1）主线是沿同一架构轴加宽、加稀疏、加长窗，不是换架构：V2 → V3 层数 60 → 61 几乎不动，参数约 2.8 倍的增量来自 hidden（5120 → 7168）与 MoE 规模（160+2 → 256+1，MoE 层数 59 → 58 基本持平）的乘积。（2）V3.1-Terminus 与 V3.2-Exp 在架构字段上与 V3 同构——V3.2 的全部架构增量是一个 lightning indexer 模块加一套续训配方，这是「稀疏化可以 retro-fit 到存量 dense checkpoint」的谱系级证据（§4）。（3）激活比单调走低：V2 的 21 / 236 ≈ 8.9%、V3 的 37 / 671 ≈ 5.5%，MoE 层内口径 9 / 257 ≈ 3.5%——与 4.5 §1 的稀疏度谱同向。

## 2. V2：两个结构性首秀的坐标登记

V2（[DeepSeek-AI, 2024](../references.md)，arXiv:2405.04434，[A]）一次推出两个后续被反复复用的结构，本章只登记坐标，推导归原章。

**MLA（multi-head latent attention）首秀**。完整推导在 4.1 §4：RoPE 与低秩 KV 压缩的数学冲突（ $R_{n-t}$ 卡在 $W^{UK}$ 与 $c$ 之间、交换子不相容）→ decoupled 解法（内容部分走低秩 latent、位置部分走独立 64 维 $k^{R}$ 旋转后缓存）→ cache 每 token 每层 $d_c + d_R$ = 512 + 64 = 576 元素。V2 定下的维度家族——kv_lora_rank 512、qk_nope_head_dim 128、qk_rope_head_dim 64、q_lora_rank 1536、v_head_dim 128——原样进入 V3（§6 逐字段对账）。质量口径按论文：V2 报告 MLA 与同规模 MHA 相当或更强（[A]）——「cache 压缩必损检索分辨率」的直觉不成立，机制解释见 4.1 §4.2 末（512 维共享 latent 对 $H$ 份高相关 KV 副本的显式低秩利用）。

**DeepSeekMoE 的旗舰线首秀**。结构出自专论（[Dai et al., 2024](../references.md)，arXiv:2401.06066，[A]），V2 是它首次进入旗舰权重：细粒度切分 + 常驻 shared expert（公式为 4.5 式 (14)(15)）：expert 数乘 $m$、中间维除 $m$、top-k 乘 $m$，参数与 per-token FLOPs 两笔账不变，收益全部来自组合自由度 $\binom{m N_e}{m k} \gg \binom{N_e}{k}$。V2 的落点：160 routed + 2 shared、top-6，外加组限定路由（V2 论文口径 M = 3，即 8 组中限发 3 组对应节点；V3 放宽为 4 组，4.5 §3.2 只覆盖 V3 侧的 n_group = 8 / topk_group = 4）。数据预算 8.1T token（论文摘要口径）——14.8T 是 V3 代的数字，两代口径不要混。

**路由均衡的代际更替**也登记在此。V2 走常规辅助损失路线——expert 级、device 级、communication 级三路平衡损失叠加在主损失上、系数调低以护主任务（V2 论文口径，[A]）；V3 把它们整体替换为 §3.1 的偏置控制器。这条替换路线的理论注脚在 4.5 §2.2 的诚实边界：aux loss 的全局最小本就不在均匀点，可用性来自均值回复的梯度方向——既然起作用的是方向而不是损失值，同向的控制回路就能做到不进损失函数，把「均衡」从优化目标里彻底拿掉。

## 3. V3：架构决策的公式级对账

先给代际 diff，再逐项对账。V2 侧数字出自论文架构节（[A]，config 未实测），V3 侧为 config 实测：

| 字段 | V2 | V3 | 改动的机制读法 |
| --- | --- | --- | --- |
| layers | 60（1 dense + 59 MoE） | 61（3 dense + 58 MoE） | MoE 层数基本持平，dense 前导 1 → 3（§6 的机制读法） |
| hidden | 5120 | 7168 | 容量轴加宽，全部投影矩阵随之放大 |
| MoE | 160+2 top-6 | 256+1 top-8 | 组合自由度再上台阶（4.5 式 (15)），shared 2 → 1 |
| expert 中间维 | 1536 | 2048 | 单 expert 变宽，top-k 与 expert 数同步上调 |
| 路由打分 | softmax | sigmoid | 归一化竞争推迟到 top-k 之后（4.5 式 (16)） |
| 负载均衡 | 三路 aux loss | aux-loss-free 偏置 | §2 末的代际更替 |
| MTP | 无 | $D = 1$ | 监督加密 + 推理草稿（4.5 §6） |
| vocab | 102400（HF config 口径） | 129280 | 词表加宽（tokenizer 侧账归 2.2） |
| 数据 | 8.1T token | 14.8T token | 约 1.8 倍（摘要口径） |
| ctx | 128K（摘要口径） | 163840（YaRN f40 on 4096） | 扩窗配方见 §3.6 |

### 3.1 路由三件套（recap → 4.5 §3.2）

V3 相对 V2 的路由改动共三处，公式与机制推导均在 4.5：

1. **sigmoid 亲和度**（4.5 式 (16)）： $s_{i,t} = \sigma(x_t^{\top} e_i)$，把 softmax 的全 expert 归一化竞争推迟到 top-k 之后、只在选中集内进行；config 落点 `scoring_func = "sigmoid"`。机制上多一句：softmax 的逐 token 归一让任一 expert 的分数变化牵动全部 expert 的概率分配，sigmoid 下各 expert 分数相互独立，均衡回路读到的负载信号更干净——这与第 2 条的偏置控制器是配套设计。
2. **aux-loss-free 偏置控制器**（4.5 式 (17)(18)）：持久偏置 $b_i$ 只进选择、不进权重，按负载误差走符号步（ $\gamma = 10^{-3}$ 为训练大部分时间的工作点），把负载均衡从优化目标降级为控制回路；config 落点 `topk_method = "noaux_tc"`。外加一个系数极小的序列级兜底损失（论文口径：extremely small，只兜底不主导）。
3. **组限定分发沿用**：n_group = 8、topk_group = 4，每 token 的分发目标节点数 $\le 4$——动机不在模型侧，在 all-to-all 通信账（4.5 §7）。

激活口径：256 个 routed 选 8、加 1 个常驻 shared，每 token 共 9 个 expert。

### 3.2 MTP（recap → 4.5 §6）

config 的 `num_nextn_predict_layers = 1` 对应 4.5 式 (19)–(21) 的 $D = 1$ 链式模块：主模型表示与真实下一 token 的嵌入拼接、投影、过一个完整 transformer block、共享 embedding 与输出头，预测 $t + 2$。训练侧是监督信号加密（每个位置多一个 horizon 的 CE）；推理侧翻作 self-speculative 草稿，接受率 $\alpha = 1 - D_{\mathrm{TV}}(p, q)$（4.5 式 (22)(23)）——V3 报告第二 token 接受率 85%–90%、端到端约 1.8× TPS（[A]）。该模块训练后可整体丢弃，不进推理的必要权重集。成本侧登记一笔：MTP 模块约 0.69B 参数（§3.3 表）参与训练期的前向与反向，是「训练时多付、推理时折现」结构的典型案例；链式条件（模块吃真实 token 嵌入）对并行头方案的优势对照见 4.5 §6.3 的 Medusa 分析。

### 3.3 671B：从 config 字段复算总参数

671B 不是 config 字段，是摘要口径——但 config 字段足以把它复算到 0.5% 以内。逐块对账（字段值见 §6；单 expert 参数 $3 \times 7168 \times 2048 \approx 44.0$ M，attention 单层约 187M 见 4.1 §4.2 的逐项账）：

```
DeepSeek-V3 层栈（61 层，d = 7168，untied 词表）
┌─ 层 1–3：dense —— MLA（cache 576 元素/token/层）+ SwiGLU FFN（d_ff = 18432）
├─ 层 4–61：MoE —— MLA + router（sigmoid，256 选 8）+ 257 个 expert（d_ff = 2048）
│                  └─ 组限定：8 组选 4 组，单 token 目标节点 ≤ 4
└─ 训练期外挂：MTP × 1（拼接投影 + 一个完整 block；共享 Emb/OutHead，推理可弃）
```

| 块 | 算术 | 参数 |
| --- | --- | --- |
| 58 个 MoE 层 | $58 \times 257 \times 44.0$ M | ≈ 656.5 B |
| attention × 61 | $61 \times 187$ M | ≈ 11.4 B |
| 前 3 层 dense FFN | $3 \times (3 \times 7168 \times 18432)$ | ≈ 1.19 B |
| embedding + 输出头 | $2 \times 129280 \times 7168$ | ≈ 1.85 B |
| MTP 模块（ $D = 1$） | $7168 \times 14336 + (187 + 396)$ M | ≈ 0.69 B |
| 合计 | | ≈ 671.6 B ≈ 671B ✓ |

激活侧（每 token）：MoE 层 9 个 expert $9 \times 44.0$ M × 58 ≈ 23.0 B，加 attention 11.4 B 与 dense FFN 1.19 B，不含词表侧的逐层激活账 ≈ 35.6 B。官方口径「37B activated」出自论文摘要（[A]）——差额即官方口径对词表与 MTP 的计法；按 P5 约定照抄官方数字，不自行换算，口径差登记在此。

### 3.4 成本口径：2.788M H800 GPU-hours 的三个切分

这个数是 V3 被转述最多的数字，也是被转述错最多的。三个必须分开的口径：

1. **全训练口径**：2.788M H800 GPU-hours 是论文 Table 1 的全训练合计——pre-training 2664K + context extension 119K + post-training 5K = 2788K（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）。它不是 pre-training 单项的数字。
2. **对账口径**：与 14.8T token 对 $6ND$ 账时只能取 pre-training 的 2664K——1.2 §9 的对账（ $6 \times 37\mathrm{B} \times 14.8\mathrm{T}$ FLOPs 对 $2.664 \times 10^{6} \times 3600$ GPU-秒，得每卡平均 $3.43 \times 10^{14}$ FLOPS）用的正是这个子口径；拿 2.788M 去除得 $3.27 \times 10^{14}$，只差 5%，但口径已经混了（extension 与 post 的 12.4 万卡时对应另一批 token 与 FLOPs）。
3. **量纲口径**：GPU-hours 是算力 × 时间，不是货币总成本——折货币需要 GPU 时单价假设，且该数字不含数据生产、人力、失败实验与推理基建。H800 本身的峰值口径（FP8 / BF16 各档 FLOPS）与 MFU 反推演归 gpu 维度（1.2 练习 3 已布置）。

凡遇「V3 训练只花 X」式转述，先问三句：全训练还是 pre-training？GPU-hours 还是货币？含不含失败实验与数据成本？三句都答不上来的转述不可用。

### 3.5 FP8 训练：模型侧要点

V3 论文完整披露了 671B 规模的 FP8 混精训练配方（[A]，论文 §3.3）。模型侧四个要点：

1. **格式**：全部 FP8 张量用 E4M3（1 符号 + 4 指数 + 3 尾数），不用 E5M2——细粒度 scaling 把动态范围问题局域化之后，尾数精度优先（论文口径）。
2. **粒度**：activation 按 $1 \times 128$ tile、weight 按 $128 \times 128$ block 各自配 scale。per-tensor 单一 scale 会被 outlier 绑架——数值走一遍：某 weight 张量绝大多数元素幅值在 $10^{-2}$ 量级、个别 outlier 到 $6$，per-tensor 以 $6$ 为 amax 定 scale 后， $10^{-4}$ 量级的小元素被压进 E4M3 的 denormal 区乃至直接归零，相对误差从百分之几恶化到接近 100%； $128 \times 128$ block 各自取 amax 后，outlier 只抬高自己所在 block 的 scale，其余 block 的有效精度不受牵连。activation 的 $1 \times 128$ tile 同理，把 outlier 通道的影响局域在 tile 内。
3. **online quantization**：每个 tile / block 的 amax 即时计算，不用历史滑动最大值（delayed scaling）——分布漂移下 scale 不失效。
4. **敏感模块豁免**：embedding、输出头、MoE gate、norm 算子、attention 算子保持原始精度（BF16 / FP32，论文清单）；Tensor Core 的部分和定期提升到 CUDA core 做 FP32 累加，补回累加精度。

工程侧——DualPipe 流水、all-to-all 通信的低精度化、kernel 实现——归 training / cluster / gpu 维度。对模型侧的接口只有一条：量化方案决定了哪些张量可以 1 字节化，这就是「671B 权重以 FP8 驻留约 671 GB」这笔部署账（4.1 练习 3 的设定）的出处。

### 3.6 YaRN 128K（recap → 4.6 §2.3）

config 的 rope_scaling 是 YaRN 字段的标准实例（W1 实测）：factor 40、original_max_position_embeddings 4096， $4096 \times 40 = 163840$ 恰为 max_position_embeddings；beta_fast 32 / beta_slow 1 的三段 ramp 在 32 个旋转通道上数出「11 个最快通道不动、12 个走 ramp、9 个最慢通道全压 40 倍」（4.6 §2.3 的逐通道走读）；mscale 字段对应 YaRN 温度修正（4.6 式 (14)， $s = 40$ 时 logits 放大 $(0.1 \ln 40 + 1)^{2} \approx 1.87$ 倍）。训练侧为两阶段扩窗（4K → 32K → 128K，论文口径），扩窗占 119K H800 GPU-hours——即 Table 1 全训练口径里的 extension 项，与 §3.4 的切分互为印证。预算比例顺手一记：119K 对 pre-training 2664K ≈ 4.5%，这就是 4.6 §4「staged extension 买的是逐段可校验的过程控制」在 V3 口径下的成本量级。

## 4. V3.1-Terminus → V3.2：DSA 与 sparse-from-dense（recap → 4.2 §6）

V3.2-Exp（[DeepSeek-AI, 2025](../references.md)，arXiv:2512.02556，[A]；官方 repo deepseek-ai/DeepSeek-V3.2-Exp）是稀疏化 retro-fit 的落地样本：架构字段与 V3 同构，增量只有一个模块与一套配方。

**结构（DSA）**。每 query 由 lightning indexer（小维度头、ReLU 打分、可 FP8）对全部历史逐 token 打分，取 top-$K$ = 2048 个位置，MLA 主路只在选中集上算（公式 4.2 式 (12)(13)）。cache 格式不变（576 元素 / token / 层全量驻留），变的是每步读哪几行——decode 每步主路读取 $61 \times 2048 \times 576 \times 2$ B ≈ 144 MB，全量对照 $61 \times 131072 \times 576 \times 2$ B ≈ 9.2 GB（ $t$ = 131072，4.2 §6 的账）。减的是带宽与 FLOPs，不是 cache 容量。

**配方（sparse-from-dense）**。从 V3.1-Terminus 的 128K dense checkpoint 出发——起点选型本身是设计：V3.1-Terminus 已是 128K 就绪的 dense MLA 模型，sparse 化只动「读哪些」、不动「怎么存」。两阶段：dense warm-up 冻结主模型、只训 indexer，目标是主模型跨头聚合 attention 分布的 KL 蒸馏（4.2 式 (14)）；随后解冻全模型 sparse 续训（lr 7.3e-6、15000 步、每步 480 条 128K 序列 ≈ $9.4 \times 10^{11}$ token——「免训练稀疏」在这条路线上不成立，4.2 §6 的规模账）。

**勘误登记**。官方 repo README 公告（2025-11-17）：indexer 的 RoPE 输入用 non-interleaved 布局，MLA 主路用 interleaved——2.5 的两种维度配对约定在同一模型里并存，写 kernel 混用即错。

**质量证据面**按 4.2 §8 的口径登记：V3.2-Exp 报告与 V3.1-Terminus 各基准持平（[A]，论文口径）——稀疏化的质量账是「持平」而不是「提升」，换到的是 §4 开头那笔 144 MB 对 9.2 GB 的读取账。

## 5. R1：一句话与去向

R1 是 V3-Base 之上的大规模强化学习推理模型（GRPO 族配方），架构字段与谱系表 V3 行逐项相同——差异全部在训练侧，归 training 维度，本章不展开。同一边界适用于 V3.2 线：V3.2 正式版相对 V3.2-Exp 的增量是 reasoning 能力的整合（发布侧口径），其机制与 R1 同属训练侧，一并归 training 维度。

## 6. 旗舰 config 精读：deepseek-ai/DeepSeek-V3 config.json 逐字段

按 2.8 §2 的方法读：每个字段在公式里点名位置。关键字段（教学用摘录，非完整文件）：

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "deepseek_v3",
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "intermediate_size": 18432,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,
  "first_k_dense_replace": 3,
  "moe_layer_freq": 1,
  "n_group": 8,
  "topk_group": 4,
  "scoring_func": "sigmoid",
  "topk_method": "noaux_tc",
  "norm_topk_prob": true,
  "routed_scaling_factor": 2.5,
  "num_nextn_predict_layers": 1,
  "vocab_size": 129280,
  "max_position_embeddings": 163840,
  "rope_theta": 10000,
  "rope_scaling": {"type": "yarn", "factor": 40.0, "beta_fast": 32, "beta_slow": 1,
                   "mscale": 1.0, "mscale_all_dim": 1.0,
                   "original_max_position_embeddings": 4096},
  "rms_norm_eps": 1e-06,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}
```

**骨架字段**。hidden_size 7168 / num_hidden_layers 61 / vocab_size 129280——61 层的内部结构 = 3 个 dense 前导层 + 58 个 MoE 层（由 MoE 块字段决定，见下）。tie_word_embeddings = false：输入 embedding 与输出头独立，词表侧参数 $2 V d$（§3.3 的 1.85B）。attention_bias = false、rms_norm_eps = $10^{-6}$：当代标准取法（对照 2.8 §8 的 bias 分野表）。torch_dtype = bfloat16：config 声明的默认计算 dtype。

**MLA 块**。q_lora_rank 1536 / kv_lora_rank 512 / qk_nope_head_dim 128 / qk_rope_head_dim 64 / v_head_dim 128——逐字段对账 4.1 式 (5)–(9)： $c^{KV} = W^{DKV} u \in \mathbb{R}^{512}$（kv_lora_rank）；query 侧同样低秩到 1536（q_lora_rank）；score 内维 $128 + 64 = 192$（nope + rope，2.8 §8 反例的数字出处）；cache 每 token 每层 $512 + 64 = 576$ 元素。num_key_value_heads = 128 是占位同值（MLA 无逐头 KV，§1 脚注一）。

**MoE 块**。n_routed_experts 256 + n_shared_experts 1 = 257 个 expert / 层；num_experts_per_tok 8，每 token 加 shared 共 9 个。**first_k_dense_replace = 3：前 3 层是 dense**——这是全 config 里最容易被扫过去、机制含义最重的一个字段。读法两条：其一，浅层表示尚未分化、token 间差异小，路由信号噪声最大，dense 前导层给全部 token 统一的基底变换；其二，浅层 MoE 的负载均衡在训练早期最不稳（router 未收敛），dense 起步把均衡回路的早期振荡隔离在表示基底之外。3 是 DeepSeek 选的工作点（GLM-4.5/4.6 同取 3，K2 取 1——5.4 / 5.5 的对照素材），公开消融缺失，列入 §7。moe_layer_freq = 1：自第 4 层起层层 MoE（对照 Llama-4 Maverick 的隔层 MoE，5.1 表）。moe_intermediate_size 2048 是 expert 的 SwiGLU 中间维（单 expert $3 \times 7168 \times 2048 \approx 44.0$ M）；intermediate_size 18432 只服务前 3 个 dense 层。scoring_func = "sigmoid" + norm_topk_prob = true + routed_scaling_factor = 2.5：4.5 式 (16)(17) 的 config 落点——sigmoid 亲和、选中集内归一、再乘 2.5 缩放。机制读法（本书注，非论文断言）：归一化使门控权重和恒为 1，2.5 把 MoE 层的输出幅度抬回与 dense FFN 分支可比的水平。topk_method = "noaux_tc"：aux-loss-free 的 config 级签名。n_group 8 / topk_group 4：组限定分发（通信动机，4.5 §7）。

**位置块**。rope_theta = $10^{4}$（32 个旋转通道的最慢波长 $2\pi \times 10^{4}$，2.5 §6 的波长账）+ rope_scaling（YaRN：factor 40 / beta_fast 32 / beta_slow 1 / mscale 1.0 / mscale_all_dim 1.0 / original 4096）→ max_position_embeddings 163840。逐通道走读与温度修正在 4.6 §2.3。

**MTP 块**。num_nextn_predict_layers 1，即 $D = 1$（4.5 §6）。

**拓扑注记**。model_type = deepseek_v3 的 DeepseekV3ForCausalLM 之后被 Kimi K2 直接复用（5.5 的 architectures 字段证据）——MLA + DeepSeekMoE 的组合已成跨实验室的架构基座。

## 7. 技术开放问题

1. **first_k_dense_replace 的最优性**：3（V3、GLM-4.5）与 1（K2）之间无公开消融；dense 前导层数与早期路由稳定性的定量关系开放。
2. **MLA 的秩下界**：4.1 §4.2 给出「512 > 单头 KV 两倍的 256」的定性论证，更小 $d_c$ 的失真—节省曲线无公开扫描。
3. **aux-loss-free 的长期动力学**： $\gamma = 10^{-3}$ 符号步的极限环幅度与训练长度的关系、偏置漂移对路由分布的慢变量效应，只有经验工作点，没有理论刻画（4.5 §3.2 的收敛分析是步级而非训练全程的）。
4. **MTP 的 $D$ 扩展**： $D = 1 \to D > 1$ 的接受率递减与训练开销曲线，4.5 §6 框架下的实证缺口。
5. **DSA 的分布外召回**：indexer 在训练分布外的静默退化风险（4.2 §8 已登记风险形状）；indexer 跨 checkpoint 迁移的可行性无公开证据。
6. **FP8 的收缩边界**：豁免清单（embedding / 输出头 / gate / norm / attention）在更低精度（FP4）下是否收缩，决定下一档权重量化的模型侧可行性。

## 总结

| 项 | 数字 / 公式 | 出处与去向 |
| --- | --- | --- |
| 谱系主线 | 同架构轴加宽加稀疏：60 → 61 层，hidden 5120 → 7168，160+2 → 256+1，8.1T → 14.8T | 谱系表；V2 [A]、V3 [A] |
| MLA 坐标 | kv_lora_rank 512、qk_rope 64、cache 576 元素 / token / 层 | 推导 4.1 §4 |
| DeepSeekMoE | 细粒度等参等算力（4.5 式 (14)(15)），组合自由度 | 推导 4.5 §3.1 |
| V3 路由 | sigmoid + 偏置只进选择（ $\gamma = 10^{-3}$）+ 组限定 8 选 4 | 推导 4.5 §3.2；config 签名 noaux_tc |
| MTP | $D = 1$，接受率 85%–90%，约 1.8× TPS | 推导 4.5 §6 |
| 671B 复算 | MoE 656.5 + attn 11.4 + dense 1.19 + 词表 1.85 + MTP 0.69 ≈ 671.6B | §3.3 |
| 成本口径 | 2.788M = 2664K + 119K + 5K（GPU-hours ≠ 货币）；对账取 2664K | §3.4；1.2 §9 |
| FP8 模型侧 | E4M3 全量； $1 \times 128$ / $128 \times 128$ 粒度；online；五类豁免 | §3.5；工程归 training |
| YaRN | factor 40 × 4096 = 163840；两阶段扩窗 119K GPU-hours | 推导 4.6 §2.3 |
| DSA | indexer top-2048；decode 读取 144 MB vs 9.2 GB；sparse-from-dense ≈ 0.94T token | 推导 4.2 §6 |

## 练习

1.（计算）参数复算。(a) 由 §6 字段重算 §3.3 的 671B 分解，逐项验证到 0.1B 精度；(b) 重算无词表侧激活账 ≈ 35.6B，说明与官方 37B 的口径差可能落在哪些项上；(c) 若把 first_k_dense_replace 从 3 改为 1（两层改 MoE、参数按同形 MoE 层计），总参数与激活参数各变多少？
2.（计算）成本口径。(a) 验证 2664 + 119 + 5 = 2788K；(b) 用 1.2 式 (9) 以 pre-training 2664K 对账 $6 \times 37\mathrm{B} \times 14.8\mathrm{T}$，推出每卡 $3.43 \times 10^{14}$ FLOPS；(c) 演示误用 2.788M 得 $3.27 \times 10^{14}$，并说明这 5% 偏差为什么仍算口径错误。
3.（计算）DSA decode 账。 $t$ = 131072、 $K$ = 2048、61 层、576 元素 / token / 层、bf16：(a) 验证主路每步读取 144 MB 对全量 9.2 GB；(b) 解释 cache 总量为何不变、这笔账省的是 2.4 式 (11) 分母里的哪一项；(c) 若 $K$ 减半到 1024，质量风险由哪个机制兜底（4.2 §8 的框架）？
4.（代码）aux-loss-free 教学仿真（教学用，非优化实现）：256 个 expert、top-8，初始负载取 Zipf 形 $f_i \propto i^{-1.2}$，按 4.5 式 (18) 的符号步（ $\gamma = 10^{-3}$）更新偏置，每步重路由并统计负载变异系数；(a) 画变异系数收敛曲线，标出极限环；(b) 扫 $\gamma \in \lbrace10^{-4}, 10^{-3}, 10^{-2}\rbrace$，验证「 $\gamma$ 小则波纹小、响应慢」的折中（4.5 §3.2 (i)）。

## 参考

- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，arXiv:2405.04434，[A]（key: deepseekv2）——MLA 与 DeepSeekMoE 首秀、组限定路由；谱系表 V2 行数字
- [Dai et al., 2024](../references.md)：DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models，arXiv:2401.06066，[A]（key: deepseekmoe）——细粒度切分与 shared expert 的公式出处
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——671B/37B/14.8T、Table 1 成本口径（2664K+119K+5K）、路由三件套、MTP、FP8 配方、两阶段 YaRN 扩窗
- [DeepSeek-AI, DeepSeek-V3 config.json（HF）](../references.md)：[A]（key: deepseek-v3-config）——§6 逐字段精读对象
- [DeepSeek-AI, 2025](../references.md)：DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models，arXiv:2512.02556，[A]（key: deepseek-v3.2）——DSA 结构与 sparse-from-dense 配方；官方 repo deepseek-ai/DeepSeek-V3.2-Exp（RoPE 布局勘误公告 2025-11-17）
