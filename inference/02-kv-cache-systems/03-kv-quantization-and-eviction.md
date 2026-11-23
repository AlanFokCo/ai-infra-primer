# 2.3 KV 量化、驱逐与压缩（KV Quantization, Eviction, and Compression）

> KV cache 放不下、读不动时，砍字节数的三个因子——每元素字节、token 数、KV 几何——各砍哪一个，代价各是什么形状？ | 前置：1.2（ $\kappa$ 几何与 decode 带宽账）、2.1（块管理与 block size $b$ ）、2.2（prefix caching）、models 2.4（cache 容量式 (9)）、models 4.1（MLA 低秩）、models 4.2 §8（稀疏召回风险） | 锚点：KIVI（arXiv:2402.02750）[A]、KVQuant（arXiv:2401.18079）[A]、H2O（arXiv:2306.14048）/ StreamingLLM（arXiv:2309.17453）/ SnapKV（arXiv:2404.14469）[A]、OCP MX v1.0 规范 [C]（官方格式规范、开放标准、非闭源）、NVIDIA NVFP4 官方文档 [C]（开放文档、非闭源）、DeepInfer doc（教学参照 + 官方实现，生产数字为官方披露）

## 1. 三笔账：KV 字节数砍哪个因子

一条请求的 KV cache 占多少字节？recap models 2.4 式 (9)：容量 $= 2 L H_{kv} d_{\mathrm{head}} T B \cdot \mathrm{bytes}$ （models 的 $b$ = 每元素字节数，按本维度翻译规则译为 $\mathrm{bytes}$ ；本章 $b$ 专归 block size，见 2.1）。把它整理成「每 token 每层系数 × 三个计数」：

$$
\mathrm{Mem}_{\mathrm{KV}} = \kappa \cdot L \cdot T \cdot B \qquad (1)
$$

符号表： $\kappa$ = 每 token 每层 KV 字节数（MHA / GQA 取 $2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}$ ，MLA 取 $(d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}$ ，几何对照表在 1.2，MLA 低秩推导在 models 4.1）； $L$ = 层数； $T$ = 序列长； $B$ = batch 内序列数。式 (1) 与 models 2.4 式 (9) 是同一笔账的两种写法，增量只在 $\kappa$ 的几何分列。

**本章口径**：Llama-3.1-8B GQA（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、bf16），全章例题复用；§4 的低精度格式事实换 V4.1-Flash MLA 口径（40 层、共享 KV 向量 512 元素 = $d_{\mathrm{latent}}$ 448 + $d_{\mathrm{rope}}$ 64；config head_dim 512 已含 rope、kv_lora_rank 448 由引擎导出，[W]，hf-mirror 传输核实），换口径原因：DeepInfer 的格式披露只存在于 V4.1 生产口径。代入章脊： $\kappa$ = 2 × 8 × 128 × 2 = 4096 B/token/层， $\kappa L$ = 131072 B = 128 KiB/token； $T$ = 131072、 $B$ = 1 时总量 16 GiB——与 models 2.4 §5 表的 8B 行一致。

式 (1) 的三个可砍因子对应三条路线，本章逐一算账：

```
KV 总字节 = κ × L × T × B
            │         │
            │         └── 驱逐（eviction）：T → ηT，只保留部分 token
            └── 量化（KV quantization）：每元素 16 bit → q bit
κ 的几何本身 ── 模型侧压缩：2 H_kv·d_head → d_latent + d_rope（MLA）
```

三条路线的精度损失形态不同：量化是有界数值扰动（token 全在、值不准），驱逐是无界信息丢失（值准、token 没了），模型侧压缩是训练期决定的低秩近似。§6 把三笔账并成一张对照表。

## 2. 量化误差界：per-channel 为什么压过 per-token

2 bit 量化把 KV 字节砍到 1/8，但 KIVI 对 K 用 per-channel、对 V 用 per-token，而不是统一粒度——为什么？答案在误差界与 K/V 的分布结构里。

所谓均匀量化（uniform quantization），是指把取值区间等分成 $2^{q}$ 个格点、每个值就近落入格点的映射。设量化对象的最小 / 最大值为 $x_{\min}, x_{\max}$ ，极差 $R = x_{\max} - x_{\min}$ （ $R$ 为本章极差局部记号，非 1.2 / 1.3 的 ridge point），则步长与误差界为：

$$
\Delta = \frac{R}{2^{q} - 1}, \qquad \vert x - Q(x) \vert \le \frac{\Delta}{2} \qquad (2)
$$

符号表： $q$ = 量化位宽（本章局部记号）； $Q(x)$ = 量化映射； $\Delta$ = 相邻格点间距；误差界对区间内任意 $x$ 成立（就近舍入的直接推论）。代入 $q$ = 2、 $R$ = 20： $\Delta$ = 20/3 ≈ 6.67，误差界 3.33——一个取值范围只有 2 的信号，量化后误差可达其范围的 1.67 倍，信息尽毁。误差界由极差 $R$ 单独决定，这就是粒度选择的全部杠杆。

粒度是「哪些元素共享一个 $R$ 」的划分。**per-token**：一个 token 向量的 $d_{\mathrm{head}}$ 个通道共享一个 scale；**per-channel**：一个通道跨一组 $g$ 个 token 共享一个 scale（ $g$ = 量化组大小，实现参数）；另引入对照粒度 **per-tensor**：整组 token × 全部通道共享一个 scale。注意前两种划分是**横截的**——per-token 按行切、per-channel 按列切，互不细化，误差界比较必须逐对做，不能传递。

**命题 1**（per-channel 误差界占优 per-tensor）：同一 $q$ 、同一组 token 上，per-channel 量化的最大误差界不超过 per-tensor（整组单 scale）的最大误差界。**证明**：记通道 $c$ 的极差为 $R_c$ ，整组极差为 $R_{\mathrm{group}}$ 。通道的取值集合是整组取值集合的子集，故对每个 $c$ 有 $R_c \le R_{\mathrm{group}}$ ；由式 (2)， $\Delta$ 随 $R$ 单调，所以 $\max_c \Delta_c / 2 \le \Delta_{\mathrm{group}} / 2$ 。等号成立当且仅当某通道同时取到全局最大与全局最小。证毕。正例：全通道共享同一区间且各通道都取到两端，两粒度界一致。等号例：单个 outlier 通道含全局两极、其余通道极差窄——该通道极差等于整组极差，最大界**相等**（通道 1 = $\lbrace 0, 10 \rbrace$ 、通道 2 = $\lbrace 4, 6 \rbrace$ ， $q$ = 2：两界皆 10/6 ≈ 1.667），此时 per-channel 的收益在正常通道，属命题 2 论域。严格占优条件：全局两极**分落不同通道**、各通道极差严格小于整组极差（通道 1 = $\lbrace 0, 4 \rbrace$ 、通道 2 = $\lbrace 6, 10 \rbrace$ ：per-channel 界 4/6 ≈ 0.667 < 1.667）。

命题 1 **不能推广到 per-token**——子集论证只对整组极差成立，对单个 token 的极差不成立。反例（ $g$ = 2、2 个通道、 $q$ = 2）：token 1 = (0, 1)、token 2 = (2, 3)。per-token：每 token 极差 1，最大误差界 1/6 ≈ 0.167；per-channel：每通道极差 2，最大误差界 1/3 ≈ 0.333 > 0.167——per-channel 的最坏界反而更差。所以「per-channel 更好」不是普适界定理；KIVI 的粒度选择依据的是 K cache 的分布事实，把它立成条件命题。

**命题 2**（outlier 隔离，条件命题）：设存在 outlier 通道 $c^{\ast}$ ，每个 token $t$ 都有 $x_{t,c^{\ast}} \ge M$ ，其余通道满足 $\vert x_{t,c} \vert \le m$ ，且 $M \ge 3m$ （持久大 outlier，KIVI 实测的分布前提）。则 per-token 量化下正常通道元素的误差界至少为 $(M-m)/(2(2^{q}-1))$ ，而 per-channel 下正常通道的误差界至多为 $m/(2^{q}-1)$ ，前者不小于后者。**证明路线**：用 outlier 给 per-token 极差下界、用 $2m$ 给 per-channel 正常通道极差上界，夹出两条界的不等式链——关键一步是 $M \ge 3m$ 使下界超过上界。**证明**：token $t$ 内最大值 $\ge M$ （outlier 通道）、最小值 $\le m$ （正常通道取值 $\le m$ ），故 $R_t \ge M - m$ ，per-token 对该 token 内任意元素（含正常通道）的误差界 $\Delta_t/2 \ge (M-m)/(2(2^{q}-1))$ ；per-channel 正常通道跨 token 极差 $R_c \le 2m$ ，误差界 $\le 2m/(2(2^{q}-1)) = m/(2^{q}-1)$ ；由 $M \ge 3m$ 得 $(M-m)/2 \ge m$ ，串联两个不等式即得。证毕。口径注：这是分布前提下的条件命题，不是普适界——前提不成立时 per-channel 可以输给 per-token（上方反例）。KIVI 原文正是分布性论断：「The persistence of outliers within each channel means that per-channel quantization can confine the quantization error to each individual channel without impacting the other normal channels」（§3）。

代入章脊感受差距：设 $d_{\mathrm{head}}$ = 128 个通道中 127 个正常通道取值 $[-1, 1]$ （ $m$ = 1），1 个 outlier 通道持久取值 $[8, 10]$ （ $M$ = 8 ≥ 3m，满足命题 2 前提）， $q$ = 2。per-token：每 token 极差 $R_t \ge M - m$ = 7，误差界 ≥ 7/6 ≈ 1.17——正常通道极差才 2，相对界已 58%；最坏 token（outlier = 10、正常通道取到 −1） $R$ = 11， $\Delta$ ≈ 3.67，格点 −1、2.67、6.33、10，正常通道取值塌到 −1 / 2.67 两个格点、以格点中点 0.833 为界粗分两档——幅度信息基本尽失，符号也不严格保留（0.833 以下的正值同样落 −1；相对界 92%）。per-channel：正常通道 $R$ = 2， $\Delta$ ≈ 0.667，误差界 0.333（相对极差 16.7%）；outlier 通道自身跨 token 极差也只有 2（持久 → 集中），同样拿到细 scale——这正是「把误差限制在每个通道内部」。

KIVI 的分布事实（[Liu et al., 2024](../references.md)，arXiv:2402.02750，[A]）：K cache 存在**持续性 outlier 通道**——少数通道跨 token 一致地大，per-channel 把 outlier 隔离进自己的 scale；V cache 无此通道结构，per-token 即够。方案为非对称 2 bit、tuning-free（无需校准微调）。per-channel 需要一组 token 才能估计每通道极差，不足一组的尾部 token 暂留全精度——论文原文「the residual key cache and value cache are kept in full precision」。摘要口径的收益：峰值内存降 2.6×（含权重）、batch 扩 4×、吞吐 2.35–3.47×。

误差界约束的是单个元素，而 attention 消费的是量化后 K/V 与 query 的内积：score 扰动沿内积维累积，再经 softmax 指数放大或抑制——所以端到端影响必须以 perplexity / task benchmark 实测，不能从逐元素界推出「无损」（§6 的口径注由此而来）。

> **坑**：对 K cache 用 per-token scale——outlier 通道绑架极差，正常通道塌到两个格点、幅度信息尽失，现象是 perplexity 爆炸而非显存报错。规避：K per-channel / V per-token 的非对称粒度（KIVI），或 outlier 逐向量隔离（§3 KVQuant）。

节省账（章脊）：bf16 的 16 bit → 2 bit，理想压缩 8×：128 KiB/token → 16 KiB/token， $T$ = 131072 时 16 GiB → 2 GiB。理想口径不含 scale 开销；scale 本身也要存。per-channel 每组 $g$ 个 token、每通道一个 fp16 scale 时，数据占 $g q$ bit、scale 占 16 bit，开销占比：

$$
\text{overhead} = \frac{16}{g \thinspace q} \qquad (3)
$$

符号表： $g$ = 量化组 token 数； $q$ = 位宽；口径为对称量化、只计 scale 不计 zero-point（非对称量化另需 zero-point，开销翻倍）。代入 $g$ = 32、 $q$ = 2：开销 25%； $g$ = 128：6.25%（教学示例值， $g$ 为实现参数）。组越大开销越小，但尾部全精度残差窗口也越大——这是 $g$ 的权衡两端。

## 3. KVQuant：把 3 bit 推到可用的四项技术

3 bit 量化、perplexity 退化小于 0.1（Wikitext-2 / C4）、LLaMA-7B 1M 上下文单张 A100-80GB、10M 上下文 8 卡——这是 KVQuant 的摘要账（[Hooper et al., 2024](../references.md)，arXiv:2401.18079，[A]）。3 bit 比 2 bit 多 50% 的字节，换来的是四项技术各修一个分布事实：

1. **Per-Channel Key Quantization**：机制同 §2 命题 2（outlier 隔离，条件命题）。
2. **Pre-RoPE Key Quantization**：在 RoPE 之前量化 K。机制：RoPE 对通道对做位置相关的旋转（models 2.5 已推导），旋转把两个通道的值按位置混合——RoPE 之后通道极差随位置变化，per-channel 在一组 token 上估出的 scale 不再逐位置贴合；RoPE 之前 K 的通道分布跨位置稳定，极差估计才有效。
3. **Non-Uniform Quantization（NUQ）**：所谓 NUQ，是指格点间距不均匀、按取值分布配置的量化——值密集处格点密、稀疏处格点疏，按层敏感度加权配置。式 (2) 的均匀网格是它的特例；分布越偏离均匀，NUQ 的同位宽误差越小。
4. **Per-Vector Dense-and-Sparse**：outlier 逐向量隔离——每个向量里少数 outlier 元素留 fp16（sparse 部分），其余走低 bit（dense 部分）。与权重量化的 outlier 分解同构（先例在 3.2）。

四项技术并非独立：NUQ 的非均匀格点拟合「正常值」的分布，拟合不进格点的 outlier 交给 dense-and-sparse 隔离——格点拟合与 outlier 隔离各管分布的一端。定制 CUDA kernel 带来约 1.7× 加速（摘要口径）。节省账（章脊）：bf16 → 3 bit 理想压缩 16/3 ≈ 5.3×：128 KiB/token → 24 KiB/token， $T$ = 131072 时 16 GiB → 3 GiB（理想口径，不含 outlier fp16 部分与 scale 开销）。对照 §2：KIVI 2 bit 省得更多但误差界更大，KVQuant 3 bit 用 50% 的额外字节买回精度——位宽选择是误差界与字节数的直接权衡，没有普适最优点，按任务的 perplexity / benchmark 容差定。

## 4. 低精度格式账：MXFP8 / NVFP4 与 DeepInfer 生产口径

量化后一个元素实际占多少字节？不是 $q/8$ 就完——scale 也要存。格式账回答「多少个元素共享一个 scale、scale 是什么类型」；格式账的正式交付在 3.2（该章回链本节），本节只算 KV 侧需要的开销账。

所谓 MX（Microscaling）格式，是指由块大小 $k$ + 元素类型 + scale 类型三元组定义的块格式（ $k$ 沿 OCP 规范原记号，本节局部，与 MoE top-k 的 $k$ 无关；OCP MX v1.0 规范，2023-09，[OCP, 2023](../references.md)，[C] 官方格式规范、开放标准、非闭源）。规范事实：MXFP8 / MXFP6 / MXFP4 / MXINT8 一律 $k$ = 32、scale 为 E8M0——8 bit 无符号纯指数，规范原文「an unsigned representation of a conventional biased Float32 exponent」；MXFP8 元素可取 E4M3 或 E5M2（不同 encoding 算不同格式），MXFP4 元素为 FP4 E2M1。每元素字节数：

$$
\mathrm{bytes}_{\mathrm{elem}} = \frac{k \cdot \mathrm{bytes}_{\mathrm{data}} + \mathrm{bytes}_{\mathrm{scale}}}{k} \qquad (4)
$$

符号表： $k$ = 块内元素数； $`\mathrm{bytes}_{\mathrm{data}}`$ = 单元素数据字节； $`\mathrm{bytes}_{\mathrm{scale}}`$ = 每块 scale 字节。代入 MXFP8（E4M3 元素 1 字节 + E8M0 scale 1 字节）：(32 × 1 + 1)/32 = 1.03125 B/元素，开销 3.125%，对 bf16 压缩 2/1.03125 ≈ 1.94×。MXFP4（E2M1 元素 0.5 字节）：(32 × 0.5 + 1)/32 = 0.53125 B/元素，对 bf16 ≈ 3.76×。

NVFP4 是 NVIDIA 的块格式（[NVIDIA, cuBLAS / TensorRT-LLM 官方文档](../references.md)，[C] 开放文档、非闭源；blog 正文反爬不可达，格式事实以这两份官方文档为据）：E2M1 元素 + **每 16 元素一个 UE4M3 block scale** + **tensor-wide FP32 global scale**，要求 Blackwell（compute capability 10.0+，Hopper 无 FP4）。cuBLAS 原文：「a dedicated scaling factor stored as an 8-bit CUDA_R_8F_UE4M3 value for each 16-element block in the innermost dimension」（CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3）；全局 scale 对应 CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F。两级 scale 的分工：per-16 block scale 管局部极差，FP32 global scale 管 tensor 级动态范围（UE4M3 可表示范围有限，单级 scale 罩不住整个 tensor）。代入式 (4)：(16 × 0.5 + 1)/16 = 0.5625 B/元素（global scale 每 tensor 4 字节，长 tensor 摊薄可忽略），对 bf16 ≈ 3.56×。TensorRT-LLM 的 KV cache 支持 `dtype='nvfp4'`（需 ModelOpt 离线量化）。

DeepInfer 生产口径（官方披露，doc 逐字，核实 2026-11-23）：doc 在「Where is FP4 used?」标题下列出主 KV——「Main KV: per-16, e4m3 — like NVFP4 but without the global scale.」「Indexer KV: per-32, ue8m0, MXFP4.」格式读法：元素是 FP4 E2M1（0.5 字节，与 NVFP4 相同；cuBLAS 类型表 CUDA_R_4F_E2M1 为 4-bit 浮点），doc 的「e4m3」指 **scale 类型**——每 16 元素一个 e4m3 scale；与 NVFP4 的差别正是省掉 tensor 级 FP32 global scale 一项。代入式 (4)：(16 × 0.5 + 1)/16 = 0.5625 B/元素——字节账与 NVFP4 一致（global scale 每 tensor 4 字节摊薄可忽略），差别在动态范围处理，不在字节数。换 V4.1-Flash 口径（原因见 §1）：MLA 每 token 每层 512 元素（ $d_{\mathrm{latent}} + d_{\mathrm{rope}}$ = 448 + 64；config head_dim 512 已含 rope，kv_lora_rank = 512 − 64 = 448 由引擎导出，读法见 7.2 §9——V3 的 576 元素（512 + 64）不适用 V4.1），512/16 = 32 个块、32 字节 scale，数据 512 × 0.5 = 256 字节，故主 KV $\kappa$ = 256 + 32 = 288 B/token/层（推导口径；与 2.4 §8 StateBlock 的 MXFP8 528 B = 512 + 16 同一 512 元素几何）；bf16 为 1024 B，压缩 1024/288 ≈ 3.56×。40 层合计 11520 B/token = 11.25 KiB；满窗 $T$ = 1048576（config `max_position_embeddings` [W]）单序列主 KV = 11.25 GiB，bf16 口径则 40 GiB（上界口径：按 40 层全量驻留计，SWA 层实际只留窗口内 KV，state 账见 2.4 §8；indexer K 另计，其 MXFP4 格式同式 (4) 的 0.53125 B/元素）。

> **坑**：把「Main KV: per-16, e4m3」读成元素 = e4m3（1 字节）——该行在「Where is FP4 used?」标题下，元素是 FP4 E2M1（0.5 字节），e4m3 是 scale 类型；误读使 $\kappa$ 虚高 1.89×（544 vs 288 B）。规避：格式读法对标题上下文 + 官方类型表（cuBLAS CUDA_R_4F_E2M1）双重核对。

> **坑**：把 NVFP4_PLUS 当成 NVFP4——后者是 Rubin（SM107）的另一格式：per-32 UE5M3 scale、无 FP32 global scale，块大小与 scale 类型都不同。触发条件：跨代际文档混读；后果：式 (4) 代入错误的 $k$ 与 scale 字节，开销账全错。规避：格式名绑定代际（NVFP4 = SM100，NVFP4_PLUS = SM107），逐字核对官方文档。

## 5. 驱逐：heavy hitter、sink 与观察窗

保留 20% 的 token 就是 5× 节省——驱逐（eviction）砍的是 token 数：量化保留全部 token、扰动值，驱逐相反，保留部分的值原封不动，但永久删除其余 token。节省账是保留比例 $\eta$ （本章局部记号：保留 token 数 / $T$ ），代价账是召回。三家方案各用一个统计量决定「谁值得留」。

**H2O**（[Zhang et al., 2023](../references.md)，arXiv:2306.14048，[A]）：所谓 heavy hitter，是指累积 attention score 排名靠前、贡献了大部分 attention value 的 token（论文口径）。H2O 以累积 score 为重要性统计量，heavy hitter 与 recent token 动态平衡——新 token 进来时按累积 score 驱逐最弱者；驱逐被形式化为 dynamic submodular 问题并带理论保证。定量锚：保留 20% heavy hitter 预算时，吞吐对 DeepSpeed Zero-Inference / HF Accelerate / FlexGen 至多 29× / 29× / 3×（OPT-6.7B / 30B），同 batch 延迟至多降低 1.9×（摘要口径）。

**StreamingLLM**（[Xiao et al., 2023](../references.md)，arXiv:2309.17453，[A]）：attention sink 现象——初始 token 即使语义不重要也获得强 attention score。机制是 softmax 归一化压力的出口（recap models 4.2 §4：attention 行权重和恒为 1，无相关内容可看时概率质量倾泻到初始位置；删掉 sink，整行权重重分配，PPL 爆炸）。处方：保留初始 sink token 的 KV + recent window，有限窗口模型即可泛化到 4M+ token 流式输入（Llama-2 / MPT / Falcon / Pythia）；预训练时加专用 sink token 进一步改善。对 sliding window recomputation 至多 22.2× 加速（摘要口径）。

**SnapKV**（[Li et al., 2024](../references.md)，arXiv:2404.14469，[A]）：每个 head 在生成中稳定关注特定的 prompt 特征，且可从 **prompt 末端 observation window** 观察得到——用 prompt 末端观察窗的 attention 投票，per-head 聚类选出重要 KV 位置压缩。16K 输入下 3.6× 生成速度、8.2× 内存效率；单张 A100-80GB 支撑 380K 上下文（HF 实现小改）；NIAH 精度损失可忽略（论文口径）。

三家统计量的偏向不同：H2O 的累积 score 偏向历史上已被频繁关注的 token，SnapKV 的观察窗偏向 prompt 末端重要的特征，StreamingLLM 不打分、只留位置结构。统计量的选择决定哪类 needle 被系统性漏掉——选择偏差就是召回风险的具体形状。三家的代价账同形：**召回风险**。设 needle 位置在 $[1, T]$ 均匀分布，静态窗口策略保留 $w$ 个 recent 位置加 $n_{\mathrm{sink}}$ 个 sink 位置：

$$
\mathrm{recall} = \frac{w + n_{\mathrm{sink}}}{T} \qquad (5)
$$

符号表： $w$ = recent window 宽度； $n_{\mathrm{sink}}$ = sink token 数；口径为 $T > w + n_{\mathrm{sink}}$ 、needle 单位置、均匀分布。**命题 3**（静态窗口召回）：式 (5) 成立。**证明留作练习 4**（路线：needle 被召回当且仅当其位置落在保留集内，均匀分布下概率 = 保留集大小 / $T$ ）。代入 $T$ = 32768、 $w$ = 4096、 $n_{\mathrm{sink}}$ = 4：recall = 4100/32768 ≈ 12.5%——窗口外 87.5% 的位置上的 needle 永久不可达。

动态打分方案（H2O / SnapKV）的召回上限 = 打分器的召回率：needle 若尚未被任何 query 关注过，累积 score 低、被驱逐，未来需要它的 query 永久失败——与 models 4.2 §8 的失败模式 2（打分器漏选）同构，风险账直接平移。两点系统含义：其一，驱逐不可逆，被删 KV 只能重算 prefill 找回（recompute vs swap 的成本账在 4.2）；其二，驱逐截断前缀的 hash chain / radix 路径，后续请求的 prefix 命中止于驱逐点（2.2 recap）——驱逐策略与 prefix caching 的命中率账耦合，不是孤立的显存操作。

## 6. 质量风险与三路线对照

三条路线的节省账能直接相加吗？字节可以相乘，精度损失形态却不可互换：量化是有界扰动，驱逐是无界丢失，模型侧压缩是训练期决策。models 4.2 §8 的结论——召回是稀疏的软肋——在这里逐字适用：驱逐就是「改可见集」，三类失败模式（可见集外驱逐 / 打分器漏选 / 保底结构缺失）原样搬来；量化不改可见集，token 全在、误差有式 (2) 的界，但误差经 attention 加权求和与层间传播累积，端到端影响以 perplexity / task benchmark 实测为准（口径：不写普适「无损」，KIVI / KVQuant 的收益数字均为论文自报口径）。

| 路线 | 砍的因子 | 精度损失形态 | 节省账（章脊） | 适用场景 |
| --- | --- | --- | --- | --- |
| 量化 | 每元素字节（16 bit → $q$ bit） | 有界数值扰动 $\Delta/2$ ，token 全在 | 2 bit 理想 8×：16 GiB → 2 GiB | 通用、tuning-free（KIVI）；超长上下文（KVQuant） |
| 驱逐 | token 数（ $T \to \eta T$ ） | 信息丢失，召回风险（静默、无界） | $\eta$ = 20%：5× | 流式对话（sink）、冗余 prompt（SnapKV）、长尾生成（H2O） |
| 模型侧压缩 | $\kappa$ 几何 | 低秩近似，质量由训练保证 | MLA 576 元素 vs MHA 反事实 32768（V3 口径 56.9×，models 4.1 recap） | 需架构支持，不可 retrofit |

模型侧压缩的系统含义： $\kappa$ 降低 → decode 每步 KV 读字节 $\kappa L T B$ 同比例下降 → 带宽账改善（1.2 的 decode 带宽账 recap，推导不重复）。MLA 低秩推导归 models 4.1，此处只登记 serving 侧后果。

> **提点**：三条路线对应三个动词——量化改「每个元素存几字节」，驱逐改「存几个元素」，模型侧压缩改「元素是什么」。前两个是运行时决策、可随 workload 切换，第三个是训练期决策、上线后不可改。这是 MLA 不能 retrofit 而量化 / 驱逐可以引擎开关的账面根源。

**组合：先驱逐后量化**。顺序有两个理由：其一，驱逐先减 token 数，量化只作用于幸存者，省掉对被驱逐 token 的量化计算与 scale 存储；其二，若先量化，驱逐打分读到的是量化后的 score，重要性统计量额外引入一层噪声。节省账相乘： $\eta \times q/16$ 。代入 $\eta$ = 0.2、 $q$ = 2：1/40，章脊 16 GiB → 0.4 GiB（口径：乘法账假设两机制的精度损失独立叠加，实际端到端影响须实测，论文未给组合口径）。组合的召回风险仍由驱逐侧决定——量化不改变可见集，式 (5) 的账原样成立。

## 总结

| 论断 | 账 | 回指 |
| --- | --- | --- |
| KV 字节 = $\kappa L T B$ ，三因子对应三路线 | 章脊 $\kappa L$ = 128 KiB/token，满窗 16 GiB | 式 (1)、§1 |
| 均匀量化误差界 $\Delta/2$ ， $\Delta = R/(2^{q}-1)$ ，由极差单独决定 | $q$ = 2、 $R$ = 20 → 界 3.33 | 式 (2)、§2 |
| per-channel 误差界 ≤ per-tensor 误差界；对 per-token 无普适占优（横截划分） | 反例 g=2：per-channel 界 1/3 > per-token 界 1/6 | 命题 1、§2、练习 2 |
| 持久 outlier 前提（M ≥ 3m）下，per-channel 把正常通道误差限回自身极差 | 正常通道界 0.333 vs per-token ≥ 1.17（M=8、m=1、q=2） | 命题 2、§2 |
| scale 开销 = $16/(g q)$ （fp16 scale、对称口径） | $g$ = 32、 $q$ = 2 → 25%； $g$ = 128 → 6.25% | 式 (3)、§2 |
| MXFP8 1.03125 / NVFP4 0.5625 / MXFP4 0.53125 B/元素 | 对 bf16 1.94× / 3.56× / 3.76× | 式 (4)、§4 |
| DeepInfer 主 KV = FP4 E2M1 元素 + per-16 e4m3 scale（类 NVFP4 无 global scale）→ 288 B/token/层 | 对 bf16 3.56×；满窗单序列 11.25 GiB | §4 官方披露 |
| 静态窗口召回 = $(w + n_{\mathrm{sink}})/T$ ；动态打分召回上限 = 打分器召回率 | 12.5% @ $T$ = 32768、 $w$ = 4096 | 式 (5)、命题 3、练习 4 |
| 先驱逐后量化，节省账 $\eta \times q/16$ 相乘 | 0.2 × 2/16 = 1/40 → 0.4 GiB | §6 |

## 练习

1.（计算 · [10]）章脊节省账：8B GQA bf16、 $T$ = 131072、 $B$ = 1。(a) cache 总量（GiB）；(b) 理想 2 bit（不计 scale 与残差窗口）总量；(c) 计入 $g$ = 128 的 per-channel fp16 scale 开销（式 (3)）后的总量。校验：(a) 与 models 2.4 §5 表 8B 行的 16 GiB 一致；(c) = (b) × (1 + 16/(128 × 2))。
2.（推导 · [20]）重证命题 1（per-channel 对 per-tensor 误差界占优），写出等号成立的充要条件；再代入 §2 反例（ $g$ = 2、2 个通道、 $q$ = 2，token (0, 1) 与 (2, 3)）分别算 per-token 与 per-channel 的最大误差界，说明为何该反例否证「per-channel 普适占优 per-token」、而 KIVI 仍对 K 选 per-channel（指向命题 2 的分布前提）。校验：反例中 per-channel 界 1/3 > per-token 界 1/6；命题 1 等号条件 = 某通道同时取到全局最大与全局最小。
3.（计算 · [20]）格式开销账：按 cuBLAS 口径（每 16 元素一个 8-bit UE4M3 scale + tensor-wide FP32 global scale）推导 NVFP4 的每元素字节数，global scale 按 4096 元素 tensor 摊薄；算对 bf16 的压缩比，并与 MXFP4（per-32 E8M0）对照。校验：块内口径与式 (4) 代入 $k$ = 16 一致；global 摊薄项 = 4/4096 B/元素。
4.（推导 · [30]）召回账：needle 位置在 $[1, T]$ 均匀分布。(a) 证明命题 3（式 (5)）；(b) 代入 $T$ = 32768、 $w$ = 4096、 $n_{\mathrm{sink}}$ = 4 算召回率；(c) 换 H2O $\eta$ = 20% 预算，若 needle 从未被关注过（累积 score 为 0、必被驱逐），召回率是多少？先写出 (c) 的预测再作答，并说明 (c) 与式 (5) 的结构差异。校验：(b) = 4100/32768；(c) 的结果与预算 $\eta$ 无关、由打分器的 attention 历史决定——对照式 (5) 的结构差异与 models 4.2 §8 失败模式 2。

## 参考

- [Liu et al., 2024](../references.md)：KIVI，arXiv:2402.02750，[A]（key: kivi）——K per-channel / V per-token 非对称 2 bit、residual 全精度窗口、2.6× / 4× / 2.35–3.47× 摘要口径
- [Hooper et al., 2024](../references.md)：KVQuant，arXiv:2401.18079，[A]（key: kvquant）——四项技术、3 bit <0.1 perplexity、1M / 10M 上下文、~1.7× kernel
- [Zhang et al., 2023](../references.md)：H2O，arXiv:2306.14048，[A]（key: h2o）——heavy hitter + recent 动态平衡、dynamic submodular 保证、20% 预算收益
- [Xiao et al., 2023](../references.md)：StreamingLLM，arXiv:2309.17453，[A]（key: streamingllm）——attention sink、sink + recent window、4M+ token、22.2×
- [Li et al., 2024](../references.md)：SnapKV，arXiv:2404.14469，[A]（key: snapkv）——observation window、per-head 聚类选择、3.6× / 8.2× / 380K
- [OCP, 2023](../references.md)：OCP Microscaling Formats (MX) Specification v1.0，[C]（key: ocp-mx，官方格式规范、开放标准、非闭源）——block 32 + E8M0、MXFP4 元素 E2M1
- [NVIDIA, cuBLAS / TensorRT-LLM 官方文档](../references.md)：NVFP4 格式，[C]（key: nvidia-nvfp4，开放文档、非闭源；blog 正文反爬，格式事实以官方文档为据）——per-16 UE4M3 + tensor-wide FP32、SM100、KV cache nvfp4
- [DeepSeek-AI, DeepInfer doc](../references.md)：doc/inference_deployment_qa.md（key: deepinfer-doc，教学参照 + 官方实现，官方披露，核实 2026-11-23）——主 KV per-16 e4m3 / indexer per-32 ue8m0 MXFP4 逐字口径
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json（key: v41-flash-config，[W]，hf-mirror 传输核实）——MLA 几何 head_dim 512 含 rope 64（kv_lora_rank 448，引擎导出）、40 层、max_position_embeddings 1048576
