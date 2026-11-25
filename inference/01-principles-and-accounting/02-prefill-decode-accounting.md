# 1.2 Prefill / Decode 的算力与带宽账（Prefill / Decode Compute and Bandwidth Accounting）

> decode 强度 $I \approx B$ 已在 models 维度推导——那为什么把 batch 凑到几百，decode 还是到不了 compute-bound？本章给出临界 batch $B^{\ast}$ 的存在条件（长上下文 KV 读把强度封顶）、MoE 每 expert batch 与 EP 度无关的论证、以及 Wide-EP 降低每卡权重流量的账 | 前置：models 1.2 §10 与 2.4 §6（算术强度与 roofline 入口，本章 recap 不重推）、models 2.4 §5（KV cache 容量式 (9)）、models 4.1 §4（MLA 低秩）、models 4.5（MoE 路由）、models 5.2（DeepSeek-V3 config 账）；1.1（命题 1 吞吐守恒、TTFT / TPOT 口径） | 锚点：DeepSeek-V3（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]）、DeepSeek-V4.1-Flash config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]）、NVIDIA 官方 datasheet（[NVIDIA, H100/H200/B200 datasheet](../references.md)，[C] 官方规格披露）、DeepInfer 官方部署 QA（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照 + 官方披露）

## 1. recap：算术强度与两阶段分野，以及本章的增量

decode 每生成一个 token 要算约 $2N$ FLOPs、同时把全部权重从显存读一遍（bf16 合 $2N$ 字节），两者相除得算术强度 $I \approx 1$ FLOP/byte；把 $B$ 条序列合成一个 batch，FLOPs 变 $B$ 倍而权重仍只读一遍，强度升到 $I \approx B$ ——这是 models 1.2 §10 已交付的结论，本章原样沿用，不重复推导。prefill 侧同理： $T_p$ 个 token 一次前向，强度 $\approx T_p$ （models 2.4 §6 式 (10)），上千 token 即进入 compute-bound 区。decode 侧带上 KV cache 与 batch 的完整强度式是 models 2.4 §6 式 (12)，本章 §3 把它翻译成本维度符号后作为式 (4) 使用；该式还给出一个关键的饱和现象： $B$ 大到 KV 读主导分母后，强度不再随 $B$ 涨，饱和于 $(2N + 4LTd_{\mathrm{model}})/(\kappa L T)$ ——8B 满窗时这个上界约 4.9，离 ridge point（峰值算力 ÷ 带宽，精确定义见 §4）的 $10^{2}$ 量级差近两个数量级（models 2.4 §6 的算例）。读这条式的两个极限是本章的入口：短上下文端 $T \to 0$ 时 $I \to 1$ （纯权重 GEMV），长上下文端 $I$ 趋于饱和上界——batch 只能把强度在两个端点之间移动，不能突破上界。所谓 compute-bound，是指吞吐被峰值算力封顶的 regime（强度高于 ridge）；memory-bound 则被带宽封顶（强度低于 ridge）。models 维度交付了公式与「decode memory-bound」的结论；serving 侧还要三个更精确的答案：batch 究竟要多大才能跨过 ridge（或证明临界不存在）、MoE 把权重切到 expert 后账怎么变、EP 度扩张时每卡流量怎么缩。这三问即本章增量。三笔账的共同用途是把 serving 决策落到可复算的数上：临界存在性决定「凑 batch 还是压 KV」的优化方向，每 expert batch 决定 MoE 的全局 batch 下限，每卡流量决定 Wide-EP 的步时与 GPU 配比——最终都汇入 1.1 式 (3) 的 goodput 与 9.1 的成本账。

**符号翻译口径注**（spec §4.3 硬规则）：recap 时 models 的 $d$ （hidden 宽）一律译为本维度的 $d_{\mathrm{model}}$ ；models 的 $b$ （每元素字节数）译为文字「字节」或 $\mathrm{bytes}$ ，不占字母——本维度的 $b$ 专归 block size（token 数，2.1 定义，本章不用）；models 的 $t$ （已缓存长度）译为 $T$ 。KV cache 容量公式 recap 一句： $`\mathrm{Mem}_{\mathrm{KV}} = \kappa L T B`$ （GQA 口径 $\kappa = 2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}$ ），即 models 2.4 式 (9) 的翻译形（原式 $2 L H_{kv} d_{\mathrm{head}} T B b$ ，其 $b$ = 每元素字节数）。

本章口径（章脊锁定，全章例题复用，换口径随文声明）：

| 口径 | 模型与几何 | 精度与硬件 | 来源 |
| --- | --- | --- | --- |
| 8B dense | Llama-3.1-8B： $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、窗口 131072 | bf16 基线；H100 SXM（BF16 dense 989.5 TFLOPS、HBM 3.35 TB/s） | models 1.2 §9 / 2.4 §5 逐项账（config [A]）；硬件 [NVIDIA datasheet](../references.md)，[C] |
| V3 MoE | DeepSeek-V3：671B 总 / 37B 激活、61 层（3 dense + 58 MoE）、 $d_{\mathrm{model}}$ = 7168、 $E$ = 256 routed + 1 shared、 $k$ = 8、expert 中间维 2048、MLA 512 + 64、窗口 163840 | 权重 fp8；带宽换算沿用 H100（教学口径，随文注） | [DeepSeek-AI, 2024](../references.md)，[A]；models 5.2 config 账 |
| V4.1-Flash | 40 层、 $d_{\mathrm{model}}$ = 5120、 $H_{kv}$ = 1 / head_dim 512 / qk_rope 64（MLA）、 $E$ = 384 routed + 1 shared、 $k$ = 6、index_topk 512 / index_n_heads 32 / index_head_dim 128、sliding_window 128、nextn 3 | config 声明 fp8 权重 / expert fp4；KV 精度随文注 | [DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]（hf-mirror 传输核实，2026-11-23） |

8B 口径的两个参数量沿用 models 1.2 §9 的声明： $N = 7.50 \times 10^{9}$ 为 matmul 口径（层参数 + 输出头，FLOPs 用）， $N_{\mathrm{full}} = 8.03 \times 10^{9}$ 为存储口径（含输入 embedding，权重字节用），两者差约 7%——embedding 查表不计 FLOPs 但计存储。跨章口径注：本章 FLOPs 用 matmul 口径 $N = 7.50 \times 10^{9}$ ，1.3 的 prefill 近似用存储口径 $N_{\mathrm{full}} = 8.03 \times 10^{9}$ （总参数量）且略去二次项（matmul 口径 N=7.50e9 vs 存储口径 N_full=8.03e9）——前者高估约 7%、后者在 $T$ = 4096 时低估约 6.7%，两误差相消，prefill 66.5 ms 的结果跨章一致（见 1.3 §4）；权重字节 $W$ 两章同取 $1.61 \times 10^{10}$ B。硬件只取本章需要的两个 H100 数（完整逐卡参数表归 1.3，微架构来源归 gpu 2.1 / 2.4）。 $B$ = batch 内序列数（MoE 节起 = 全局 batch，DP 组合计，§5 声明）； $T$ = 当前序列长（prompt + 已生成）； $\kappa$ = 每 token 每层 KV 字节数； $R$ = ridge point = 峰值算力 ÷ 带宽。

本章增量有三件，全部是 serving 侧精确化而非重推：其一，临界 batch $B^{\ast}$ 的闭式与**存在条件**（§4）——饱和上界低于 ridge 时根本不存在 compute-bound 临界；其二，MoE 每 expert batch $Bk/E$ 与 EP 度无关的论证及强度稀释（§5）；其三，Wide-EP 每卡权重流量 $\propto 1/e$ 的账（§6）。V4.1-Flash 的模型侧形式化（CED / YOCO 混合注意力、Engram、DSpark draft 结构）归 models 维度（V4.1 待补），本章只引 config 字段做 serving 账。

## 2. prefill FLOPs：参数项 + 二次项

prefill 一次前向打 $T$ 个 prompt token，FLOPs 由哪两笔构成？参数项每 token $2N_{\mathrm{act}}$ （models 1.2 式 (2)，MoE 用激活口径），attention 打分与加权和随 $T$ 平方涨（models 1.2 式 (6)）。两笔相加、符号翻译到本维度即得：

$$
C_{\mathrm{prefill}} \approx 2N_{\mathrm{act}} T + 2 L T^{2} d_{\mathrm{model}} \qquad (1)
$$

**符号**： $N_{\mathrm{act}}$ ：每 token 激活参数量（dense 模型 $N_{\mathrm{act}} = N$ ，MoE 为激活口径）； $T$ ：prompt 长度； $L$ ：层数； $d_{\mathrm{model}}$ ：hidden 宽。二次项系数 2 来自 causal mask 的平均可见长 $T/2$ （models 1.2 §6 的 $\sum_t t = T(T+1)/2$ ），不打 causal mask 的双向场景系数回到 4。这是 recap 翻译，不是新推导；FlashAttention 不减少这笔 FLOPs、只省 score 矩阵的显存与访存（models 1.2 §6 / 4.1 已论证）。prefill 强度 $\approx T$ 的理由也一并 recap：分子随 $T$ 线性涨、分母的权重读与 $T$ 无关，激活读写虽随 $T$ 线性但在长 prompt 下被 matmul 摊销（models 2.4 §6 式 (10)）——所以 prefill 是「大 matmul」，serving 系统把它当训练类负载调度。

二次项什么时候反超参数项？令每 token 两笔相等解出 $T^{\ast} = N/(L d_{\mathrm{model}})$ （models 1.2 §6 式，层参数口径），8B 代入得 $T^{\ast} \approx 5.3 \times 10^{4}$ 。三个长度下二次项占前向总量的比例（recap models 1.2 §6 表，口径 = 二次项 ÷ 总前向）： $T$ = 4096 时 6.7%、32768 时 36%、131072 时 70%——服务窗口拉满时，attention 才是 prefill 的算力主项。

全数字实例（8B 口径， $T$ = 4096）：参数项 $1.50 \times 10^{10} \times 4096 = 6.14 \times 10^{13}$ FLOPs，二次项 $2 \times 32 \times 4096^{2} \times 4096 = 4.40 \times 10^{12}$ FLOPs，合计 $6.58 \times 10^{13}$ FLOPs。按 H100 BF16 dense 峰值 989.5 TFLOPS（[NVIDIA datasheet](../references.md)，[C]）折算约 66.5 ms——这是 prefill 计算时间的峰值下界（口径：忽略访存与 launch 开销；prefill 强度 $\approx T = 4096 \gg R \approx 295$ ，compute-bound，下界接近实际）。对照 1.1 的 TTFT 分解：这 66.5 ms 只是 TTFT 的计算成分，排队与 tokenize 另计。MoE 口径换 $N_{\mathrm{act}}$ ：V3 每 token 前向 $2 \times 37 \times 10^{9} = 7.4 \times 10^{10}$ FLOPs（models 1.2 §9），二次项的 $d_{\mathrm{model}}$ = 7168、 $L$ = 61 同式代入。全数字实例（V3 口径， $T$ = 4096）：参数项 $7.4 \times 10^{10} \times 4096 = 3.03 \times 10^{14}$ FLOPs，二次项 $2 \times 61 \times 4096^{2} \times 7168 = 1.47 \times 10^{13}$ FLOPs，合计 $3.18 \times 10^{14}$ ——二次项占比 4.6%。MoE 的大 $N_{\mathrm{act}}$ 把交叉点 $T^{\ast} = N_{\mathrm{act}}/(L d_{\mathrm{model}})$ 推到约 $8.4 \times 10^{4}$ （V3 口径），比 8B 的 $5.3 \times 10^{4}$ 更远：激活参数越多，参数项越晚被二次项反超。serving 侧的一个修正预告：chunked prefill 把 $T$ 切成 chunk 分批前向，单次前向的强度从 $T$ 降到 chunk 长，二次项效率损失与 decode stall 的权衡账归 4.3。

## 3. decode 每步：FLOPs、KV 读字节与几何对照

decode 每步只算 1 个 token，为什么访存反而比 prefill 更紧张？因为每步必须把全量 KV cache 读一遍。先给 FLOPs 与字节两笔账，再逐几何代入。第 $T$ 步（已缓存长度 $T$ ）单条序列的 FLOPs 是式 (1) 的被加项按单 token 取值：参数项 $2N_{\mathrm{act}}$ ，attention 项每层 $4 T d_{\mathrm{model}}$ （ $QK^{\top}$ 与 $PV$ 各 $2Td_{\mathrm{model}}$ ，对全部 $T$ 个缓存位置），合计：

$$
f(T) = 2N_{\mathrm{act}} + 4 L T d_{\mathrm{model}} \qquad (2)
$$

**符号**： $f(T)$ ：单序列单步 decode FLOPs；其余同式 (1)。这是 models 2.4 式 (8) 被加项的翻译形（recap）。代入 8B 口径 $T$ = 4096： $f = 1.50 \times 10^{10} + 4 \times 32 \times 4096 \times 4096 = 1.71 \times 10^{10}$ FLOPs——attention 线性项只占 12%，参数项主导； $T$ = 131072 时线性项涨到 $6.87 \times 10^{10}$、反超参数项 4.6 倍，长上下文 decode 的 FLOPs 主项易主。访存侧，每步必须把该序列的全量 KV cache 读一遍——每 token 每层 $\kappa$ 字节、 $L$ 层、 $T$ 个 token，合 $\kappa L T$ 字节；权重则全 batch 共享、每步只读一遍 $W$ 字节。 $\kappa$ 按注意力几何分形：

$$
\kappa = \begin{cases} 2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}, & \mathrm{MHA / GQA / MQA} \\ (d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}, & \mathrm{MLA} \end{cases} \qquad (3)
$$

**符号**： $H_{kv}$ ：KV head 数（MHA 取 $H_{kv} = H_q$ ，GQA 取 $H_{kv} < H_q$ ，MQA 取 $H_{kv} = 1$ ）； $d_{\mathrm{head}}$ ：head 维； $\mathrm{bytes}$ ：每元素字节数（bf16 = 2，fp8 = 1）； $d_{\mathrm{latent}}$ = kv_lora_rank（MLA 压缩 latent 维）； $d_{\mathrm{rope}}$ = qk_rope_head_dim（decoupled RoPE 维）。MHA/GQA 形的因子 2 是 K、V 两份；MLA 形**无因子 2、无 $H_{kv}$ 乘**——cache 只存一份共享压缩 latent 加一份旋转 key（低秩推导归 models 4.1 §4，此处只取字节账）。DSA 在 MLA 形之上另加 indexer K（每 token 每层一份小维 key，头间共享，models 4.2 §6 口径），字节数按 indexer 精度另计。

> **坑**： $2 H_{kv} d_{\mathrm{head}}$ 形勿代入 MLA。触发条件：V4.1-Flash config 的 num_key_value_heads = 1、head_dim = 512 是共享 KV 向量维（已含 rope 64），不是逐头 KV 几何；现象：代入得 $2 \times 1 \times 512 = 1024$ 元素/层，较真值 $448 + 64 = 512$ 高估 2.0×，整条带宽账与容量账同步虚高；规避：MLA 一律走式 (3) 第二行。V3 config 有 kv_lora_rank / qk_rope_head_dim 语义字段；V4.1 config **无 kv_lora_rank 字段**——head_dim 512 已含 rope 64，kv_lora_rank = 512 − 64 = 448 由引擎导出（DeepInfer config.py 读法，见 7.2 §9，[W]）。V4.1 的 512 元素与 V3 的 576 元素（512 latent + 64 rope）是两代几何，勿混。

几何对照表（bf16 口径；V3 / V4.1 的 MLA 行 $\kappa$ 同形）：

| 几何 | $\kappa$ 形 | 代入 | 每 token 每层 | 整模型每 token | 满窗单序列 |
| --- | --- | --- | --- | --- | --- |
| MHA（8B 反事实， $H_{kv}$ = 32） | $2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}$ | $2 \times 32 \times 128 \times 2$ | 16384 B | 512 KiB（32 层） | 64 GiB @131072 |
| GQA（8B， $H_{kv}$ = 8） | 同上 | $2 \times 8 \times 128 \times 2$ | 4096 B | 128 KiB | 16 GiB @131072 |
| MQA（ $H_{kv}$ = 1） | 同上 | $2 \times 1 \times 128 \times 2$ | 512 B | 16 KiB | 2 GiB @131072 |
| MLA（V3，512 + 64） | $(d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}$ | $576 \times 2$ | 1152 B | 68.6 KiB（61 层） | 10.7 GiB @163840 |
| MLA（V4.1-Flash，448 + 64）[W] | 同上 | $512 \times 2$ | 1024 B | 40 KiB（40 层） | 随混合注意力几何，7.2 |
| DSA（V3.2 / V4.1） | MLA 形 + indexer K | indexer 128 维另计 | +128 B（fp8 论文口径） | 主路读取按 top-k 削减 | 容量不变 |

读这张表的方式：同一台 8B 机器，MHA 反事实的每 token 增量是 GQA 的 4 倍（head 分组 4:1 在 cache 上是精确 4 倍削减，recap models 2.4 §5）；MLA（V3 口径）的 576 元素介于 GQA-4 的 2048 与 MQA 的 256 之间（recap models 4.1 §4.2 表）——它不是把 KV 压到比单头还小，而是把 $H_q$ 份高相关副本换回一份共享 latent，检索结构一个没少（低秩论证归 models 4.1 §4）。表内 8B 三行与 V3 行 recap models 2.4 §5 / 4.1 §4.2 的实例账（config [A]）；V4.1 行代入 config 字段 [W]：head_dim 512 是共享 KV 向量维、已含 rope 64（config 无 kv_lora_rank 字段，引擎导出 kv_lora_rank = 512 − 64 = 448，读法见 7.2 §9）， $\kappa = (448 + 64) \cdot \mathrm{bytes} = 512 \cdot \mathrm{bytes}$ ，bf16 合 1024 B/token/层，40 层合 40960 B = 40 KiB/token——与 V3 的 576 元素区分，两代几何勿混。indexer K 的维度取 index_head_dim = 128 [W]，每 token 每层一份（头间共享）；V3.2 论文口径 FP8 合 128 B，DeepInfer 生产 indexer KV 用 per-32 ue8m0 MXFP4（官方披露，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，2026-11-23 核实），格式账归 2.3 / 3.2。DSA 的容量不变、变的是每步读哪几行：V3.2 主路 top-2048 使每步读取从全量 9.2 GB 降到 144 MB（ $T$ = 131072，recap models 4.2 §6 / 5.2 §4）；V4.1-Flash 的 index_topk = 512 [W]，读取账同形（top-k × $\kappa$ × $L$ ），混合注意力层（SWA / CED）的窗口与压缩修正归 7.2，本章不展开模型侧。KV 的分页管理（block size $b$ 、碎片）归 2.1（[Kwon et al., 2023](../references.md)，[A]）。

$B$ 条序列合批：FLOPs 变 $B f(T)$ ，KV 读变 $B \kappa L T$ （逐序列独立、不可共享），权重仍只读一遍 $W$ 。强度式即 models 2.4 式 (12) 的翻译 + MoE 推广：

$$
I_{\mathrm{decode}}(B, T) = \frac{B f(T)}{W + B \kappa L T} \qquad (4)
$$

**符号**： $W$ ：每步权重读字节（dense 为 $N_{\mathrm{full}} \cdot \mathrm{bytes}$ ；MoE 随 batch 激活的 expert 数变，§5 式 (10)）；其余同前。式 (4) 是 recap 忠实性核对的落点： $B$ 小、权重项主导分母时 $I \approx B f / W \approx B$ （bf16 dense），即 models 的「强度 $\propto B$ 」； $B$ 大、KV 项主导时饱和。最小实例： $B$ = 1、 $T$ = 4096 时 $I = 1.71 \times 10^{10}/(1.61 \times 10^{10} + 5.37 \times 10^{8}) \approx 1.03$ FLOP/byte——单序列 decode 强度贴地，与 models 1.2 §10 的「 $\approx 1$」读数一致。每步访存的结构画成走读图（8B dense， $B$ 条序列各长 $T$ ）：

```
decode 一个 iteration 的 HBM 流量（bf16，8B 口径）
─────────────────────────────────────────────────────────
权重 W = 16.1 GB ────────────────► 读一遍，全 batch 共享
                                   （GEMM：[B,4096]×[4096,·]）
KV cache：
  序列 1  κLT = 4096 B × 32 × T ─► 读全量 ┐
  序列 2  κLT ───────────────────► 读全量 ├ B 份，逐序列独立
  ...                                    │（attention 逐序列算）
  序列 B  κLT ───────────────────► 读全量 ┘
─────────────────────────────────────────────────────────
合计字节 = W + B·κLT；FLOPs = B·f(T)
B=64、T=4096：16.1 GB + 64×0.537 GB = 50.4 GB
```

全数字实例（8B bf16， $B$ = 64、 $T$ = 4096）： $W = 1.61 \times 10^{10}$ B， $\kappa L T = 4096 \times 32 \times 4096 = 5.37 \times 10^{8}$ B，合计 $5.04 \times 10^{10}$ B。按 H100 HBM 3.35 TB/s（[NVIDIA datasheet](../references.md)，[C]）折算步时下界约 15.1 ms（口径：带宽下界，memory-bound 下接近实际）；强度 $I = 64 \times 1.71 \times 10^{10} / 5.04 \times 10^{10} \approx 21.8$ FLOP/byte，远低于 ridge ≈ 295。由 1.1 命题 1，吞吐 $\approx 64 / 0.0151 \approx 4240$ tokens/s。把 $B$ 翻到 128：KV 项翻倍、权重项不动，步时约 25.3 ms、吞吐约 5060 tokens/s——batch 翻倍只换来 19% 吞吐，因为分母里 KV 已经占大头。这就是「凑 batch 收益递减」的逐步读数。两笔边界注。其一，KV 写：每步每序列只 append $\kappa L$ 字节（8B 口径 128 KiB），比读 $\kappa L T$（ $T$ = 4096 时 0.537 GB）小约 $T$ 倍——逾三个数量级，流量账由读主导；写的影响在容量与碎片，归 2.1。其二，KV 读不能跨 batch 共享：逐序列内容不同， $B$ 条就是 $B$ 份——唯一例外是共享前缀，其复用账归 2.2（prefix caching）。这就是式 (4) 分母里 KV 项带 $B$ 而权重项不带的原因。

## 4. 临界 batch $B^{\ast}$：存在条件与闭式

多大的 batch 才能让 decode 转为 compute-bound？路线分三步：先取纯权重 GEMM 的强度（recap），与 ridge 对齐得权重口径临界 $B_w^{\ast}$ ；再把 KV 读加回分母解完整临界；关键一步是第三步——强度有饱和上界，上界低于 ridge 时方程无解，临界根本不存在。

纯权重 GEMM 口径（忽略 KV 与 attention 项）：每 token 参数 FLOPs $2N$ 、权重字节 $`N \cdot \mathrm{bytes}_{\mathrm{param}}`$ ， $B$ 条序列合批得强度（models 1.2 §10 已交付，recap）：

$$
I_w = \frac{2B}{\mathrm{bytes}_{\mathrm{param}}} \qquad (5)
$$

**符号**： $`\mathrm{bytes}_{\mathrm{param}}`$ ：每参数字节数（bf16 = 2、fp8 = 1、fp4 = 0.5）。「bf16 GEMV 强度 $\approx 1$ 」是式 (5) 在 $B$ = 1、 $`\mathrm{bytes}_{\mathrm{param}}`$ = 2 的取值；FP8 同 batch 强度翻倍、FP4 翻四倍。ridge point $R$ = 峰值算力 ÷ 带宽（models 2.4 §6 roofline 入口；逐卡参数表归 1.3），H100 SXM 代入：BF16 $R = 989.5/3.35 \approx 295$ 、FP8 $R = 1979/3.35 \approx 591$ FLOP/byte（[NVIDIA datasheet](../references.md)，[C]，dense 口径）。令 $I_w = R$ 解出权重口径临界：

$$
B_w^{\ast} = \frac{R \cdot \mathrm{bytes}_{\mathrm{param}}}{2} \qquad (6)
$$

代入：H100 bf16 约 295、FP8 约 295——精度升级使峰值翻倍、字节减半，两者在式 (6) 里恰好对消， $B_w^{\ast}$ 在同一硬件上大致不变（B200 FP4 验证： $R = 9000/7.7 \approx 1169$ ， $B_w^{\ast} = 1169 \times 0.5/2 \approx 292$ ，[NVIDIA datasheet](../references.md)，[C]）。所以「换低精度就能让 decode 进 compute-bound」不成立：真正改变临界存在性的是 KV 项。

把 KV 读加回，对式 (4) 解 $I(B^{\ast}, T) = R$ 。路线：交叉相乘 → 移项合并 $B^{\ast}$ → 除以系数，三步代数： $B^{\ast} f = R W + R B^{\ast} \kappa L T$ ，故 $B^{\ast}(f - R \kappa L T) = RW$ ，得

$$
B^{\ast} = \frac{R W}{f(T) - R \kappa L T} \qquad (7)
$$

**符号**： $f(T)$ 见式 (2)， $W$ 、 $\kappa$ 见式 (4)。式 (7) 的分母可能非正——这不是代数瑕疵，是物理事实的入口。记饱和上界（式 (4) 令 $B \to \infty$ ）：

$$
I_{\mathrm{sat}}(T) = \frac{f(T)}{\kappa L T} \qquad (8)
$$

**命题 1**（ $B^{\ast}$ 存在条件）。 $I_{\mathrm{decode}}(B, T)$ 对 $B$ 严格单调递增、以 $I_{\mathrm{sat}}(T)$ 为上确界；compute-bound 临界 $B^{\ast}$ 存在当且仅当 $I_{\mathrm{sat}}(T) > R$ ，存在时由式 (7) 给出。

**证明**（路线：改写 → 单调 → 取极限 → 存在性；单调性细节留作练习 3）。把式 (4) 分子分母同除 $B$ ： $I = f/(W/B + \kappa L T)$ 。 $B$ 增则 $W/B$ 严格减、分母严格减、 $I$ 严格增（练习 3(a) 补全）。 $B \to \infty$ 时 $W/B \to 0$ ， $I \to f/(\kappa L T) = I_{\mathrm{sat}}$ 。连续严格增函数取到 $R$ 当且仅当 $R$ 小于上确界，即 $I_{\mathrm{sat}} > R$ ；此时式 (7) 分母 $f - R\kappa LT > 0$ ，解唯一。边界注：若 $R \le I(1)$ 则交叉点小于 1，意为单序列 decode 已 compute-bound；真实硬件 ridge 比 $I(1) \approx 1$ 高约两个数量级，该分支不出现，命题的「存在」按「batch ≥ 1 的交叉点存在」读。∎

全数字实例（8B bf16，H100 $R \approx 295$ ， $W = 1.61 \times 10^{10}$ B）：

| $T$ | $f(T)$（FLOPs） | $\kappa LT$（B） | $I_{\mathrm{sat}}$ | $B^{\ast}$ |
| --- | --- | --- | --- | --- |
| 131072（满窗） | $8.37 \times 10^{10}$ | $1.72 \times 10^{10}$ | 4.9 | 不存在（4.9 ≪ 295） |
| 4096 | $1.71 \times 10^{10}$ | $5.37 \times 10^{8}$ | 31.9 | 不存在（31.9 < 295） |
| 256 | $1.51 \times 10^{10}$ | $3.36 \times 10^{7}$ | 451 | ≈ 908 |
| 128 | $1.51 \times 10^{10}$ | $1.68 \times 10^{7}$ | 898 | ≈ 469 |

满窗行 recap models 2.4 §6 的 4.9 算例。存在条件的边界可以解出闭式：令 $f(T) = R \kappa L T$ ，即 $2N + 4LTd_{\mathrm{model}} = R \kappa L T$ ，移项得

$$
T_c = \frac{2N}{L (R \kappa - 4 d_{\mathrm{model}})} \qquad (9)
$$

**符号**： $T_c$ ：存在 compute-bound 临界的上下文长度上限（ $T < T_c$ 时存在）。代入 8B bf16 H100： $R\kappa = 295 \times 4096 \approx 1.21 \times 10^{6}$ ，减 $4 d_{\mathrm{model}} = 16384$ 、乘 $L$ = 32 得 $3.82 \times 10^{7}$ ， $T_c = 1.50 \times 10^{10} / 3.82 \times 10^{7} \approx 393$ 。结论：**上下文超过约 400 token，8B bf16 @H100 的 decode 就不存在 compute-bound 临界**——真实 serving 的全部长度区间里，decode 都是 memory-bound，「凑 batch」只能把强度从 ≈1 抬向饱和上界（吞吐仍受益），永远到不了 ridge。这是 models 2.4 §6「memory-bound 结论在整个窗口内成立」的精确化：不是「batch 不够大」，是「临界不存在」。 $T_c$ 随硬件移动：换 H200 SXM（同算力、带宽 4.8 TB/s，[NVIDIA datasheet](../references.md)，[C]）， $R = 989.5/4.8 \approx 206$ 下降，代入式 (9) 得 $T_c \approx 567$——带宽越大、ridge 越低，compute-bound 临界存在的上下文区间越宽。方向与直觉一致：decode 的 memory-bound 程度是硬件参数与上下文长度的联合性质，不是模型的固有标签。

临界不存在不等于凑 batch 无收益——吞吐仍有上界，且上界与 $B$ 无关。由 1.1 命题 1， $\Theta = B/\tau_{\mathrm{step}}$；取带宽下界 $\tau_{\mathrm{step}} \approx (W + B\kappa LT)/\mathrm{BW}$（ $\mathrm{BW}$ 为 HBM 带宽）， $B$ 增大时权重项被摊掉， $\Theta \to \mathrm{BW}/(\kappa L T)$。代入 8B：满窗 $T$ = 131072 的上界 $\approx 3.35 \times 10^{12}/1.72 \times 10^{10} \approx 195$ tokens/s， $T$ = 4096 时 $\approx 6240$ tokens/s——每卡 decode 吞吐上界由「带宽 ÷ 每 token KV 字节」决定，与模型 FLOPs 无关。这是「长上下文 serving 贵」的带宽侧读数：上下文翻倍，上界减半；容量规划把这条上界换算成 GPU 数，归 9.1。把 $T$ = 256 的 $B^{\ast} \approx 908$ 对容量账交叉核对：KV 总量 $908 \times 3.36 \times 10^{7}$ B ≈ 30.5 GB，加权重 16 GB 合 46.5 GB，H100 80 GB 装得下——该临界批不被容量排除。同一个乘积 $\kappa L T B$ 有两笔用途：容量账数驻留（每序列一份、常驻），带宽账数流量（每步全读一遍）；前者决定「装不装得下」，后者决定「跑不跑得动」，2.1 的分页与本章的强度式分别管这两笔。

> **提点**： $B^{\ast}$ 是带宽代数的临界，不是调度推荐值。实际操作点由 SLO 约束下的 goodput 最大化决定（1.3 的三方权衡与排队论）——即使 $B^{\ast}$ 存在，把 batch 开到 $B^{\ast}$ 也会把 TBT 拖出 SLO。

精度口径注：权重与 KV 同降 FP8 时， $\kappa$ 减半使 $I_{\mathrm{sat}}$ 翻倍、 $R$ 也翻倍，存在条件 $I_{\mathrm{sat}} > R$ 两边同乘 2、结论不变；KV 单独量化（权重不动）才真正抬高 $I_{\mathrm{sat}}$ 相对 $R$ 的位置——KV 量化的带宽账归 2.3。

## 5. MoE 账：每 expert batch 与 EP 无关

MoE 每 token 只激活 37B 参数（V3 口径），为什么权重流量不是 37B 量级？因为「激活多少 FLOPs」与「读多少权重」在 MoE 里是两笔账：FLOPs 按 $N_{\mathrm{act}}$ 计（recap models 1.2 §9），权重读多少取决于 batch 激活了多少个**不同的** expert。所谓专家并行（expert parallelism，EP），是指把 $E$ 个 routed expert 切到 $e$ 张卡上、每卡持有 $E/e$ 个、token 按路由结果跨卡流动的部署方式（训练侧 dispatch/combine 通信账归 training 5.4，serving 侧增量归 8.1；路由与 top- $k$ 的模型侧数学归 models 4.5）。本节起 $B$ 的口径升级为**全局 batch**——全部 DP 组合计的序列数（口径注：单 DP 组本地 batch 记 $B_{\mathrm{local}}$ ， $B = d \cdot B_{\mathrm{local}}$ ， $d$ 为 DP 度；前提声明是命题 2 的要件）。

先给权重流量的两 regime 账。设单 expert 参数量 $W_{\mathrm{exp}}$ （V3：SwiGLU 三矩阵 $3 \times 7168 \times 2048 \approx 44.0$ M，models 5.2 §3.3），dense 部分（attention + 前导 dense 层 + shared expert）权重 $W_{\mathrm{dense}}$ 每步必读，routed 部分按激活数计：

$$
W_{\mathrm{MoE}}(B) = W_{\mathrm{dense}} + \min(Bk, E) \cdot W_{\mathrm{exp}} \cdot \mathrm{bytes}_{\mathrm{param}} \cdot L_{\mathrm{moe}} \qquad (10)
$$

**符号**： $k$ ：top-k 路由的 k； $L_{\mathrm{moe}}$ ：MoE 层数（V3 = 58）； $\min$ 取的是「每 token 贡献 $k$ 个 (token, expert) 对、至多激活 $Bk$ 个不同 expert」的上界（均衡路由理想化；随机路由下期望激活数略低于此上界，量级不变）。两个 regime： $Bk < E$ 时每 expert 平均不到 1 个 token，权重流量随 $B$ 线性涨、强度停在 dense GEMV 量级； $Bk \ge E$ 时全部 expert 激活，权重流量封顶于全量 expert 权重——V3 routed 合计 $256 \times 44.0 \mathrm{M} \times 58 \approx 654$ GB（fp8；models 5.2 §3.3 的 MoE 块账 656.5B 含 shared expert，口径差即此一项）。**FLOPs 按 37B 计、权重按 654 GB 读**，这个不对称就是 MoE decode 需要更大全局 batch 的算术根源。代入小 batch 看第一 regime（V3 fp8， $B$ = 8）： $Bk$ = 64 < $E$ = 256，至多 64 个 expert 激活，routed 权重流量 $\approx 64 \times 44.0\ \mathrm{MB} \times 58 \approx 163$ GB，只有全量 654 GB 的四分之一；每个激活 expert 平均约 1 个 token，GEMM 退化为 GEMV、强度停在 ≈ 2 FLOP/byte（fp8）。 $B$ = 32（ $Bk = E$）起流量封顶全量——「MoE 权重流量随 batch 涨」只在小 batch 区间成立，大 batch 区间涨的是 FLOPs、流量不动。

每 expert 拿到多少 token？在均衡路由下这是一个与 EP 度无关的量：

**命题 2**（每 expert batch）。设 $B$ 为全局 batch（DP 组合计）、路由均衡（每个 routed expert 被选中的概率相同），则每个 routed expert 每步收到的 token 数为

$$
B_{\mathrm{exp}} = \frac{Bk}{E} \qquad (11)
$$

与 EP 度 $e$ 无关。

**证明**（计数，三步）：其一， $B$ 个 token 各选 $k$ 个 routed expert，每步共产生 $Bk$ 个 (token, expert) 分配对；其二，均衡路由把分配对均摊到 $E$ 个 expert，每 expert 得 $Bk/E$ （均衡前提的模型侧来源：V3 的 aux-loss-free 偏置控制器把负载均摊写进路由决策，recap models 4.5 §3 / 5.2 §6；serving 时分布仍会漂移，系统侧兜底是 EPLB，7.4）；其三，EP 只决定 expert 的物理放置（每卡 $E/e$ 个），不改变分配对的产生与摊分——放置是存储侧自由度，路由是计算侧事实，两者正交。∎

边界反例两条。其一，若把 $B$ 误读为单 DP 组本地 batch，结论差 $d$ 倍：全局分配对是 $d B_{\mathrm{local}} k$ 而非 $B_{\mathrm{local}} k$ 。其二，路由不均时 $Bk/E$ 只是均值，热 expert 实收可远超均值——尾延迟由最慢卡决定（training 5.4 §4 的 $\max_j R_j$ 论证 serving 侧同形），均衡的系统手段（EPLB 冗余副本与重分配）归 7.4。正例代入：V3 $k$ = 8、 $E$ = 256， $B_{\mathrm{exp}} = B/32$ ； $B$ = 1024 时每 expert 32 个 token。V4.1-Flash $k$ = 6、 $E$ = 384 [W]， $B_{\mathrm{exp}} = B/64$ ； $B$ = 2048 时同样 32 个——expert 数涨 1.5 倍、k 降 25%，稀释因子从 1/32 变 1/64，凑同样的每 expert batch 需要翻倍的全局 batch。

每 expert 的 GEMM 强度随之确定： $B_{\mathrm{exp}}$ 个 token 过 $W_{\mathrm{exp}}$ 参数，FLOPs $2 B_{\mathrm{exp}} W_{\mathrm{exp}}$ 、权重字节 $`W_{\mathrm{exp}} \cdot \mathrm{bytes}_{\mathrm{param}}`$ ，相除得

$$
I_{\mathrm{exp}} = \frac{2 B_{\mathrm{exp}}}{\mathrm{bytes}_{\mathrm{param}}} = \frac{2Bk}{E \cdot \mathrm{bytes}_{\mathrm{param}}} \qquad (12)
$$

**符号**： $I_{\mathrm{exp}}$ ：单 expert GEMM 的算术强度。对照 dense 的式 (5)：MoE 把强度稀释了 $k/E$ 倍（V3 为 1/32、V4.1 为 1/64）。全数字实例（V3 fp8， $`\mathrm{bytes}_{\mathrm{param}}`$ = 1）： $B$ = 1024 时 $I_{\mathrm{exp}} = 2 \times 1024 \times 8/256 = 64$ FLOP/byte，远低于 H100 FP8 ridge ≈ 591；令 $I_{\mathrm{exp}} = R$ 解出 $B = RE/(2k) \approx 591 \times 256/16 \approx 9452$ ——MoE decode 要让 expert GEMM 进 compute-bound，需要近万的全局 batch。这个量级解释了为什么 MoE serving 必须把 DP 组开大、把全局 batch 凑足（§6 的 Wide-EP 与 reconciling 注）。

**reconciling 注**（命题 2 与生产口径的对账）。DeepInfer 生产部署 attention 侧 #DP = #EP（官方 doc L33，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）：扩 $e$ 的同时 DP 组数同步扩，若每组本地 batch 不变，全局 $B = d B_{\mathrm{local}}$ 随 $e$ 线性增，由式 (11) 每 expert batch 增、由式 (12) 强度升。这正是 doc「Increasing EP raises arithmetic intensity」论断（doc L295，逐字）的机制——**EP 本身不改变强度（命题 2），改变强度的是与 EP 耦合扩张的全局 batch**。两个口径不矛盾，前提不同：命题 2 固定全局 $B$ 谈 $e$ 的无关性，生产口径固定本地 batch 谈 $e$ 的耦合效应。DeepInfer 的取值因此分阶段：prefill 取 EP2–8 / DP2–8（HBM 允许的最小；理由：prefill 每 expert batch 已足够大，小 DP/EP 组改善负载均衡并减 NVLink 流量，doc L47 / L55-57）；decode 取 EP64–128 / DP64–128 的 Wide-EP（GPU 允许的最大，doc L49）。所谓 Wide-EP，是指大 $e$ 的 EP 部署（decode 阶段 EP64–128 量级），其每卡流量账见 §6。

## 6. Wide-EP 每卡流量账： $\propto 1/e$ 的权重读

同一个全局 batch，把 EP 从 8 扩到 128，每卡每步要读的 expert 权重变多少？路线：先数每卡持有的 expert 数，再证均衡且 $B_{\mathrm{exp}} \ge 1$ 时本地 expert 全部激活、读取数就是持有数，最后对 $e$ 取比——关键一步是「全激活」条件，它把流量从「激活 expert 数」简化成「持有 expert 数」。布局走读（ $e$ = 4、 $E$ = 8、top-2 示意；DP 组与 EP rank 一一对应，#DP = #EP）：

```
全局 batch B（DP 组 1..4 各 B/4 条序列，attention 逐组本地算）
   │ 每 token 选 k=2 个 expert → 共 Bk 个分配对
   ▼ all-to-all dispatch（按路由投递，training 5.4 语义）
┌──────────┬──────────┬──────────┬──────────┐
│ rank 1   │ rank 2   │ rank 3   │ rank 4   │
│ expert 1,2│ expert 3,4│ expert 5,6│ expert 7,8│  ← 每卡 E/e = 2 个
│ 各收 Bk/E │ 各收 Bk/E │ 各收 Bk/E │ 各收 Bk/E │  ← 命题 2，与 e 无关
│ 读 2 份权重│ 读 2 份权重│ 读 2 份权重│ 读 2 份权重│  ← 式 (13)
└──────────┴──────────┴──────────┴──────────┘
   ▼ all-to-all combine（输出发回原 DP 组加权归约）
每卡流量 = (E/e)·W_exp·bytes·L_moe；e 卡合计 = E·W_exp·bytes·L_moe（不变）
```

**命题 3**（每卡流量与强度）。命题 2 前提下，若 $B_{\mathrm{exp}} = Bk/E \ge 1$ （全部 expert 激活），则每卡每步 MoE 权重流量为

$$
M_{\mathrm{rank}} = \frac{E}{e} \cdot W_{\mathrm{exp}} \cdot \mathrm{bytes}_{\mathrm{param}} \cdot L_{\mathrm{moe}} \qquad (13)
$$

对 $e$ 反比；每卡 MoE GEMM 强度 $`= 2Bk/(E \cdot \mathrm{bytes}_{\mathrm{param}})`$ ，与 $e$ 无关。

**证明**（计数）：每卡持有 $E/e$ 个 expert； $B_{\mathrm{exp}} \ge 1$ 保证每个本地 expert 至少收到 1 个 token、权重必须读，故流量 = 持有数 × 单 expert 字节 × MoE 层数，即式 (13)。每卡处理的分配对数 = $(E/e) \cdot (Bk/E) = Bk/e$ ，FLOPs = $2(Bk/e) W_{\mathrm{exp}} L_{\mathrm{moe}}$ ，除以式 (13) 得强度 $`2Bk/(E \cdot \mathrm{bytes}_{\mathrm{param}})`$ ， $e$ 对消。 $e$ 卡合计流量 = $`e \cdot M_{\mathrm{rank}} = E W_{\mathrm{exp}} \mathrm{bytes}_{\mathrm{param}} L_{\mathrm{moe}}`$ ，与 $e$ 无关——Wide-EP 不减总流量，减的是**每卡**流量；各卡并行读，步时墙钟由每卡流量决定。∎

全数字实例（V3 fp8，routed expert 合计 654 GB，带宽换算沿用 H100 3.35 TB/s——教学口径；生产部署的硬件带宽不同，账同形换数，硬件形态走读归 7.1）：

| $e$ | 每卡 expert 权重 | 每步读时间下界 | 每卡强度（ $B$ = 2048，fp8） |
| --- | --- | --- | --- |
| 8 | 81.7 GB | 24.4 ms | 128 FLOP/byte |
| 64 | 10.2 GB | 3.05 ms | 128 FLOP/byte |
| 128 | 5.11 GB | 1.52 ms | 128 FLOP/byte |

流量与读时间严格 $\propto 1/e$ ，强度列不动——两个量各管一件事。DeepInfer 官方披露的论断正是此账：「Wide-EP substantially reduces per-step memory traffic in the MoE expert part and, with good EPLB support, effectively raises TPS」（doc L61，[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，2026-11-23 核实）。引文的两个限定词对应命题 3 的两个前提：「good EPLB support」= 均衡路由（否则热卡流量与计算双超均值，步时被尾卡拖住），「per-step memory traffic」= 每卡口径（总流量不变）。均衡前提的反向读法：若路由偏斜使某卡收到的分配对为均值的 $\alpha$ 倍（ $\alpha > 1$），该卡流量与计算同乘 $\alpha$，步时被最慢卡拖住——引文的「effectively raises TPS」按 $\alpha$ 折损。所以 Wide-EP 必须与 EPLB 配套部署：式 (13) 给的是均值账，尾延迟由偏斜决定（负载不均的系统形态与 training 5.4 §4 的 $\max_j R_j$ 论证同形；EPLB 的冗余副本与重分配算法归 7.4）。

> **提点**：强度与流量是两笔账，别混。强度（FLOPs ÷ 字节）决定 compute-bound 还是 memory-bound 的 regime；每卡流量决定步时墙钟。Wide-EP 在固定全局 batch 下不改强度、只缩步时（命题 3）；耦合扩 DP 才改强度（§5 reconciling 注）。读 doc 的「raises TPS」时先问它动的是哪个量。

代价面与边界，各一句伏笔。通信侧：dispatch/combine 两次 all-to-all 的延迟随 $e$ 与跨节点比例上升，decode 小消息延迟敏感（训练侧账 recap training 5.4，serving 侧增量归 8.1）。 $e$ = 64–128 必然跨出单机 NVLink 域——scale-up domain（NVLink 域 / 超节点）的概念与逐卡互联参数归 1.3，RDMA 硬件层归 gpu 9.1 / 9.2，本章只记「通信代价随域直径上升」这一笔； $e$ 继续增大时通信与同步开销逐渐主导、Wide-EP 边际收益递减（DeepInfer 官方 doc L295 口径，正式账归 3.3 / 7.4）。激活条件侧： $B_{\mathrm{exp}} \ge 1$ 要求 $B \ge E/k$ （V3 = 32）；低于此值只有部分 expert 激活，式 (13) 改为按激活数计、流量更低但 GEMM 更窄（grouped GEMM 的 padding 浪费账归 3.3）。负载均衡的部署与算法（EPLB）归 7.4，本节只交付流量账本身。

## 7. 三口径汇总与账的边界

三口径在同一张账上差多少？把全章的账按章脊三口径并排（bf16 KV 口径；V3 权重 fp8、V4.1 权重 config 声明 fp8 / expert fp4 [W]）：

| 账 | 8B dense @H100 | V3 MoE | V4.1-Flash |
| --- | --- | --- | --- |
| FLOPs 口径 | $N = 7.50 \times 10^{9}$（matmul） | $N_{\mathrm{act}} = 37 \times 10^{9}$ | 激活口径随 config，模型侧归 models |
| $\kappa$（每 token 每层） | 4096 B（GQA-8） | 1152 B（MLA 576 元素） | 1024 B（MLA 512 元素）+ indexer K [W] |
| 整模型每 token KV | 128 KiB | 68.6 KiB | 40 KiB（40 层） |
| 满窗单序列 KV | 16 GiB @131072 | 10.7 GiB @163840 | 混合几何，7.2 |
| decode 强度上界 | $I_{\mathrm{sat}}$：4.9（满窗）→ 451（ $T$ = 256） | 每 expert 口径 $I_{\mathrm{exp}} = B/16$（fp8） | $I_{\mathrm{exp}} = B/32$（fp8， $k/E$ = 1/64） |
| 临界 batch 读数 | $T \gtrsim 393$ 无 compute-bound 临界 | expert GEMM 进 ridge 需 $B \approx 9452$（fp8） | 同形，稀释更深 |
| Wide-EP 每卡流量 | 不适用（dense） | 654 GB / $e$（routed fp8） | expert fp4 再减半，格式账归 3.2 |

账的边界声明（本维度分工，纯文字指引）：硬件逐卡参数表与排队论归 1.3；KV 的系统手段——分页与碎片（2.1）、前缀复用（2.2）、量化与驱逐（2.3）、offload 与分层（2.4）——全部以本章的 $\kappa L T B$ 为输入；decode attention 与量化 GEMM 的 kernel 算法归 P3；组批与 chunked prefill 归 P4；EPLB 部署与 Wide-EP 生产形态归 7.4，serving 侧并行组合与 PD 分离归 8.1 / 8.2；V4.1-Flash 模型侧形式化归 models 维度（V4.1 待补）。V4.1 行的 expert fp4 使每卡流量在 V3 fp8 基础上再减半（每参数 0.5 字节，scale 开销归格式账 3.2）——量化与 Wide-EP 两个杠杆在流量账上相乘： $`M_{\mathrm{rank}} \propto \mathrm{bytes}_{\mathrm{param}}/e`$ ，这是 7.4 生产部署同时动两个旋钮的账面依据。本章不做的事也声明清楚：不推导 kernel 内的 tiling 与指令级实现（归 P3 与 gpu 4.3 / 5.2 / 6.3）、不建立排队论延迟模型（归 1.3）、不展开模型侧架构数学（归 models）。本章交付的是这些章共用的三笔底账：prefill FLOPs（式 (1)）、decode 强度与临界（式 (4)(7)(8)）、MoE 每 expert batch 与每卡流量（式 (11)(13)）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| prefill FLOPs = 参数项 + 二次项；8B 满窗二次项占 70%， $T^{\ast} \approx 5.3 \times 10^{4}$ | 式 (1)，6.58e13 FLOPs @4096 | §2、练习 2 |
| decode 单步单序列 FLOPs = $2N_{\mathrm{act}} + 4LTd_{\mathrm{model}}$；KV 读 $\kappa LT$ 字节 | 式 (2)(3) | §3 |
| $\kappa$：MHA/GQA 取 $2H_{kv}d_{\mathrm{head}}\cdot\mathrm{bytes}$，MLA 取 $(d_{\mathrm{latent}}+d_{\mathrm{rope}})\cdot\mathrm{bytes}$（无因子 2、无 $H_{kv}$）；DSA 另加 indexer K | 式 (3) 表 | §3、练习 1 |
| MLA 误代 $2H_{kv}d_{\mathrm{head}}$ 形高估 2.0×（V4.1：1024 vs 512） | §3 坑注 | §3、练习 1 |
| decode 强度 $Bf/(W+B\kappa LT)$：小 $B$ 时 $\approx B$（recap models），大 $B$ 饱和于 $f/(\kappa LT)$ | 式 (4)(8) | §1、§3 |
| 权重口径临界 $B_w^{\ast} = R\cdot\mathrm{bytes}/2$，同硬件跨精度大致不变（H100 ≈ 295） | 式 (5)(6) | §4 |
| $B^{\ast}$ 存在当且仅当 $I_{\mathrm{sat}} > R$；8B bf16 @H100 的 $T_c \approx 393$，真实上下文区间无 compute-bound 临界 | 命题 1、式 (7)(9) | §4、练习 2、3 |
| MoE 权重流量两 regime： $Bk \lt E$ 线性涨、 $Bk\ge E$ 封顶全量（V3 routed 654 GB fp8） | 式 (10) | §5 |
| 每 expert batch = $Bk/E$，与 EP 度无关（前提：全局 batch + 均衡路由）；强度稀释 $k/E$ | 命题 2、式 (11)(12) | §5、练习 4 |
| 「Increasing EP raises arithmetic intensity」的机制是 #DP=#EP 耦合扩全局 batch，非 EP 本身 | §5 reconciling 注 | §5、7.4、8.1 |
| Wide-EP 每卡流量 $\propto 1/e$、每卡强度与 $e$ 无关；总流量不变、步时由每卡流量定 | 命题 3、式 (13) | §6、练习 4 |

## 练习

1.（计算 · [10]）V4.1-Flash 口径 [W]：(a) 由 config 字段（head_dim 512、qk_rope_head_dim 64、40 层；注意 head_dim 是共享 KV 向量维、已含 rope，见 §3 坑注）算 bf16 的 $\kappa$ 与整模型每 token KV 增量（KiB）；(b) 若误代 $2 H_{kv} d_{\mathrm{head}}$ 形（num_key_value_heads = 1），每 token 每层元素数高估多少倍？校验：(a) 与 §3 几何对照表 V4.1 行一致；(b) 与坑注的 2.0× 一致。
2.（计算 · [20]）8B bf16 @H100（ $R \approx 295$ ， $W = 1.61 \times 10^{10}$ B）：(a) 先预测 $T$ = 4096 与 $T$ = 256 哪个存在 compute-bound 临界及理由，再用式 (8) 算 $I_{\mathrm{sat}}$ 判定；(b) 存在者算 $B^{\ast}$；(c) 用式 (1) 算 $T$ = 4096 的 prefill FLOPs 与峰值下界时间。校验：满窗 $I_{\mathrm{sat}} \approx 4.9$ 与 models 2.4 §6 算例一致；(b) 的 $B^{\ast}$ 代回式 (4) 应给 $I \approx R$（容差 1%）。
3.（推导 · [30]）(a) 补全命题 1 证明的单调性步：由 $I = f/(W/B + \kappa LT)$ 严格论证 $I$ 对 $B$ 单调增（ $W, \kappa, L, T, f > 0$ ）；(b) 由存在条件边界 $f(T) = R\kappa LT$ 解出 $T_c$ 闭式（式 (9)），代入 8B bf16 H100 口径验证 ≈ 393；(c) 权重与 KV 同降 FP8（ $`\mathrm{bytes}_{\mathrm{param}}`$ 与 $\kappa$ 减半、 $R$ 翻倍）时 $T_c$ 怎么变？先预测方向再代入。校验：(b) 与正文一致；(c) 的 $T_c$ 变化应能由式 (9) 的 $R\kappa$ 乘积不变直接读出。
4.（计算 · [30]）Wide-EP 账（V3 fp8， $B$ = 2048 全局 batch， $e \in \lbrace 8, 64, 128 \rbrace$ ，带宽 3.35 TB/s）：(a) 每 expert batch 与每卡收到的分配对数；(b) 每卡每步 expert 权重流量与读时间下界，验证 $\propto 1/e$ ；(c) 每卡 MoE GEMM 强度，验证与 $e$ 无关；(d) 若 #DP = #EP 耦合扩张、每 DP 组本地 batch 固定 32， $e$ 从 8 到 64 时全局 $B$ 与 $I_{\mathrm{exp}}$ 各变多少？校验：(b) 与 §6 表一致；(c) 与式 (12) 一致；(d) 强度应 $\propto e$ （reconciling 注的定量形式）。

## 参考

- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——671B / 37B 激活口径、MoE 与 MLA 架构事实；config 几何经 models 5.2 逐项账
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：[W]（key: v41-flash-config）——MLA 几何（head_dim 512 / qk_rope 64）、384+1 top-6、indexer 字段、quantization_config（hf-mirror 传输核实，2026-11-23）
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——KV cache 系统背景，分页管理归 2.1
- [NVIDIA, H100/H200/B200 datasheet](../references.md)：[C] 官方规格披露（key: nvidia-datasheet）——H100 SXM BF16 989.5 / FP8 1979 TFLOPS dense、3.35 TB/s；B200 FP4 9000 TFLOPS dense、7.7 TB/s（完整参数表归 1.3）
- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：教学参照 + 官方披露（key: deepinfer-doc）——#DP=#EP（L33）、prefill EP2–8 / decode EP64–128（L47 / L49 / L55-57）、Wide-EP 流量论断（L61）、EP-强度论断（L295）、indexer KV MXFP4，2026-11-23 核实
