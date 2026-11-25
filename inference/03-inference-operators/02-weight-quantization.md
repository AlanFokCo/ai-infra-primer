# 3.2 权重量化算法（Weight Quantization Algorithms）

> 把权重从 16 bit 压到 4 bit，为什么直接就近舍入（RTN）会让 OPT-66B 的 perplexity 从 9.34 爆到 110，而 GPTQ 能压回 9.55、加 act-order 后追平 9.34？本章拆三族算法——二阶误差补偿（GPTQ）、激活感知缩放（AWQ）、outlier 迁移（SmoothQuant / LLM.int8）——各自怎么改量化映射，并交付 INT4 / FP8 / MXFP4 / NVFP4 的格式字节账 | 前置：1.2（decode 带宽账与算术强度）、2.3 §2（均匀量化误差界）、2.3 §4（KV 侧格式账，本章回链）、training 1.3（FP8 E4M3/E5M2 与 V3 blockwise） | 锚点：GPTQ（arXiv:2210.17323）[A]、AWQ（arXiv:2306.00978）[A]、SmoothQuant（arXiv:2211.10438）[A]、LLM.int8（arXiv:2208.07339）[A]、OCP MX v1.0 规范 [C]（官方格式规范、开放标准、非闭源）、NVIDIA NVFP4 官方文档 [C]（开放文档、非闭源）、DeepSeek-V3（arXiv:2412.19437）[A]、V4.1-Flash config [W]、DeepInfer doc（教学参照 + 官方实现，生产数字为官方披露）

## 1. 为什么压权重：decode 带宽下界与三族算法

权重占 decode 每步访存的大头，压位宽能直接砍带宽下界——可为什么不能把每个权重就近舍入到 4 bit 格点就完事？先回答这个更基本的问题：量化省的是什么、代价是什么形状。

recap 1.2 式 (4)：decode 每步把全量权重读一遍，bf16 下合 $2N$ 字节（ $N$ = 参数量），权重项主导分母时算术强度 $I \approx B$ 。把权重压到 4 bit（每参数 0.5 字节），权重读字节砍到 1/4，同 batch 强度升到 $4B$ ——AWQ 原文的「W4A16 把算术强度提 4×」正是这笔账（[Lin et al., 2023](../references.md)，arXiv:2306.00978，[A]，图 3）。所谓 weight-only 量化（W4A16），是指只压权重存储位宽、激活仍走 16 bit 计算：省的是权重读带宽，不是算力。decode 是 memory-bound（1.2 §4），所以这笔节省直接落到步时下界上。

代价是数值扰动。量化把连续权重映到有限格点，每个权重引入舍入误差；误差经矩阵乘累积、再经层间传播放大，端到端表现为 perplexity 上升或 task 精度下降。三族算法的区别不在「压不压」，而在**怎么改量化映射让输出误差最小**：

```
共同目标：min ||WX - ŴX||²  （层重建误差，Ŵ = 量化后权重）
        │
        ├── 改舍入决策（二阶补偿）── GPTQ：量化一列，把误差按逆 Hessian
        │                            补偿到未量化列，再量化下一列
        ├── 改量化格点（逐通道缩放）── AWQ：放大 salient 通道权重、缩小对应
        │                            激活，等价变换下相对舍入误差下降
        └── 改分布形状（outlier 处理）── SmoothQuant：把激活 outlier 迁移到权重
                                     ── LLM.int8：outlier 维隔离走 fp16
```

三族并非互斥：GPTQ 与 AWQ 都作用在 weight-only 路线，SmoothQuant / LLM.int8 作用在 W8A8（权重激活同压 INT8）路线。格式字节账（§2）是所有路线共用的底座——先算清一个元素实际占几字节，再谈算法怎么压。

**本章口径**：字节账脊柱取单个线性层权重 $W \in \mathbb{R}^{4096 \times 4096}$ （16,777,216 个参数，bf16 = 32 MiB），全章格式开销与压缩比复用；算法推导的数字实例用 $2 \times 2$ / $2 \times 3$ 小矩阵（教学 toy，随文声明）；质量（perplexity）实例引 GPTQ / AWQ / SmoothQuant / LLM.int8 论文自报的 OPT / BLOOM / GLM 数字（换口径原因：论文实验在这些模型族上，PPL 只能引原文，不在字节脊柱模型上）；§7 的 FP8/FP4 部署对照换 V3 / V4.1-Flash 口径（原因：格式披露只存在于这两者的 config / doc）。

**符号约定**： $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ = 权重矩阵， $X \in \mathbb{R}^{d_{\mathrm{in}} \times n}$ = 校准激活（ $n$ = 校准 token 数），层输出 $Y = WX$ ； $q$ = 量化位宽、 $g$ = 量化组大小（元素数）、 $k$ = 块内元素数（沿 OCP 规范记号，本章局部，与 MoE top-k 的 $k$ 无关）； $\Delta$ = 量化步长； $\alpha$ = 缩放指数 / 迁移强度（**本章局部记号，沿 AWQ / SmoothQuant 原文，与 4.4 投机解码接受率 $\alpha$ 无关**）；bytes 一律文字「字节」或 $\mathrm{bytes}$ ，不占字母（ $b$ 专归 KV block size，2.1）。GPTQ 原文的 $d_{\mathrm{row}} / d_{\mathrm{col}}$ 即本章 $d_{\mathrm{out}} / d_{\mathrm{in}}$ 。

## 2. 量化格式账：粒度、scale 与每元素字节

一个 INT4 权重实际占多少字节？不是 0.5 就完——scale 也要存，粒度决定 scale 的数量。2.3 §4 已交付 KV 侧的格式账（该章式 (4)），本节把同一笔账推广到权重侧并补齐 INT 对称 / 非对称与二维块，正式回链 2.3 的前瞻引用。

所谓均匀量化（uniform quantization），是指把取值区间等分成格点、每个值就近落入格点的映射（2.3 §2 已定义）。**对称量化**以零为中心，只需一个 scale：

$$
Q(x) = \Delta \cdot \mathrm{Round}\left( \frac{x}{\Delta} \right), \qquad \Delta = \frac{\mathrm{absmax}(x)}{2^{q-1}} \qquad (1)
$$

符号表： $q$ = 位宽； $\Delta$ = 相邻格点间距； $\mathrm{absmax}(x)$ = 量化组内最大绝对值（即 $\max \vert x \vert$ ）； $\mathrm{Round}$ = 就近取整； $Q(x)$ = 重构值。步长取 AWQ 原文约定 $\mathrm{absmax}/2^{q-1}$ ，对称格点 $[-2^{q-1}, 2^{q-1}-1]$ ：INT8 为 $[-128, 127]$ 、INT4 为 $[-8, 7]$ （负侧多一格）。另一常见约定取 $\mathrm{absmax}/(2^{q-1}-1)$ 、格点 $[-127, 127]$ （§6 LLM.int8 用），两者差一格、大格点数下相对误差同阶——本章式 (1) 统一走 AWQ 约定。**非对称量化**（min-max 网格）另需一个 zero-point $z$ 把区间平移到 $[x_{\min}, x_{\max}]$ ：

$$
Q(x) = \Delta \cdot \left( \mathrm{clamp}\left( \mathrm{Round}\left( \frac{x}{\Delta} \right) + z, 0, 2^{q}-1 \right) - z \right), \qquad \Delta = \frac{x_{\max} - x_{\min}}{2^{q} - 1} \qquad (2)
$$

符号表： $z = \mathrm{Round}(-x_{\min}/\Delta)$ = zero-point（整数偏移）； $\mathrm{clamp}(u, 0, 2^{q}-1)$ = 把整数代码截到 $[0, 2^{q}-1]$ ； $Q(x)$ = 重构值（与式 (1) 同语义，先量化成代码再减 $z$ 乘 $\Delta$ 还原）。其余同式 (1)。GPTQ 用非对称 min-max（§3），AWQ 用对称 absmax（§4）。非对称比对称多存一个 zero-point，开销翻倍——这是式 (4) 的口径来源。代入脊柱 $q$ = 4、组内 $\mathrm{absmax}$ = 1.0：对称 $\Delta$ = 1.0/8 = 0.125，格点间距 0.125；一个取值 0.6 的权重落格点 0.625（ $5 \times 0.125$ ），舍入误差 0.025。

粒度是「哪些元素共享一个 $\Delta$ 」的划分，四档由粗到细：**per-tensor**（整矩阵一个 scale）、**per-channel / per-row**（每输出通道一个 scale）、**group**（每 $g$ 个连续输入通道一个 scale，AWQ / GPTQ 用 $g$ = 128）、**block**（二维 $r \times c$ 块一个 scale，V3 / V4.1 权重块）。粒度越细，outlier 只抬高自己所在组的 scale，其余组的小元素保住——这正是 training 1.3 §6 论证的 blockwise 精度收益，此处不重推。每元素字节数（推广 2.3 式 (4) 到二维块 + zero-point）：

$$
\mathrm{bytes}_{\mathrm{elem}} = \frac{k \cdot \mathrm{bytes}_{\mathrm{data}} + \mathrm{bytes}_{\mathrm{scale}} + \mathrm{bytes}_{\mathrm{zp}}}{k} \qquad (3)
$$

符号表： $k$ = 块（组）内元素数； $`\mathrm{bytes}_{\mathrm{data}}`$ = 单元素数据字节（INT4 = 0.5、INT8/FP8 = 1、FP4 = 0.5）； $`\mathrm{bytes}_{\mathrm{scale}}`$ = 每块 scale 字节； $`\mathrm{bytes}_{\mathrm{zp}}`$ = 每块 zero-point 字节（对称 = 0）。group 量化的 scale 开销占比（对称、fp16 scale）即 2.3 式 (3)：

$$
\text{overhead} = \frac{16}{g \thinspace q} \qquad (4)
$$

代入脊柱 INT4-g128 对称： $g$ = 128、 $q$ = 4，开销 16/512 = 3.125%；非对称加 zero-point 翻倍到 6.25%。脊柱总账：数据 16,777,216 × 0.5 = 8,388,608 字节 = 8 MiB，组数 16,777,216/128 = 131,072，对称 scale 131,072 × 2 = 262,144 字节 = 256 KiB，合计 8.25 MiB，对 bf16（32 MiB）压缩 32/8.25 ≈ 3.88×；非对称合计 8.5 MiB、压缩 ≈ 3.76×。

低精度浮点格式把 scale 收进格式本身。FP8 E4M3 / E5M2 的位布局与范围 recap training 1.3 §1 表（E4M3 最大 448、18 个 binade、无 inf；E5M2 最大 57344、inf/NaN 齐全），分工与 per-tensor scaling 归 training 1.3 §5，本章不重推。块格式（MX / NVFP4）的规范事实如下，字节账按式 (3) 代入脊柱：

| 格式 | 元素 | 块大小 $k$ | scale 类型 | 每元素字节 | 脊柱总量 | 对 bf16 压缩 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 per-tensor 对称 | INT8 | 整矩阵 | fp32 ×1 | ≈1.0 | 16 MiB | 2× | 式 (3) |
| INT4-g128 对称 | INT4 | 128 | fp16 ×1 | 0.515625 | 8.25 MiB | 3.88× | 式 (4) |
| MXFP8 | E4M3 | 32 | E8M0 ×1 | 1.03125 | 16.5 MiB | 1.94× | OCP MX v1.0 |
| MXFP4 | E2M1 | 32 | E8M0 ×1 | 0.53125 | 8.5 MiB | 3.76× | OCP MX v1.0 |
| NVFP4 | E2M1 | 16 | UE4M3 ×1 + FP32 global | 0.5625 | 9 MiB | 3.56× | NVIDIA 官方文档 |
| V4.1 权重块 | FP8 E4M3 | 32×32 | ue8m0 ×1 | ≈1.00098 | ≈16.02 MiB | ≈2× | config [W] |

表内每元素字节按式 (3)：MXFP8 (32×1+1)/32 = 1.03125；MXFP4 (32×0.5+1)/32 = 0.53125；NVFP4 (16×0.5+1)/16 = 0.5625（global scale 每 tensor 4 字节，长 tensor 摊薄可忽略）；V4.1 块 32×32 = 1024 个 FP8 元素配 1 字节 ue8m0 scale，(1024+1)/1024 ≈ 1.00098。INT4-g128 对称每元素 0.5 × (1+3.125%) = 0.515625。

规范事实逐条带出处。**OCP MX v1.0**（[OCP, 2023](../references.md)，[C] 官方格式规范、开放标准、非闭源）：MXFP8 / MXFP6 / MXFP4 / MXINT8 一律块大小 32 + E8M0 scale——8 bit 无符号纯指数，规范原文「an unsigned representation of a conventional biased Float32 exponent」；MXFP4 元素为 FP4 E2M1。**NVFP4**（[NVIDIA, cuBLAS / TensorRT-LLM 官方文档](../references.md)，[C] 开放文档、非闭源；blog 正文反爬不可达，格式事实以这两份官方文档为据）：E2M1 元素 + 每 16 元素一个 UE4M3 block scale + tensor-wide FP32 global scale，要求 Blackwell（SM100，Hopper 无 FP4）。cuBLAS 记法 CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3（每 16 元素一个 8-bit UE4M3）+ SCALAR_32F（全局）；TensorRT-LLM KV cache 支持 `dtype='nvfp4'`。两级 scale 分工：per-16 block scale 管局部极差，FP32 global scale 管 tensor 级动态范围（UE4M3 可表范围有限，单级罩不住整 tensor）。

V4.1-Flash config 的量化字段（[DeepSeek-AI, V4.1-Flash config](../references.md)，[W]，hf-mirror 传输核实，2026-11-23）：`quant_method` fp8、`activation_scheme` dynamic、`weight_block_size` [32,32]、`scale_fmt` ue8m0、`expert_dtype` fp4。读法：非 expert 权重走 FP8 + 32×32 块 + ue8m0 scale，routed expert 走 FP4。ue8m0 是 8-bit 无符号纯指数（与 OCP E8M0 同构），scale 取 2 的整数幂——乘除是精确的指数加减，不引入额外量化误差（training 1.3 §6 的 round-scaled 同理）。expert FP4 的块格式与 DeepInfer 生产披露对照见 §7。

> **坑**：把 NVFP4_PLUS 当成 NVFP4——后者是 Rubin（SM107）的另一格式：per-32 UE5M3 scale、无 FP32 global scale，块大小与 scale 类型都不同。触发条件：跨代际文档混读；后果：式 (3) 代入错误的 $k$ 与 scale 字节，开销账全错。规避：格式名绑定代际（NVFP4 = SM100，NVFP4_PLUS = SM107），逐字核对官方文档（2.3 §4 同坑）。

> **坑**：把 scale 类型读成元素类型——V4.1 `scale_fmt` ue8m0 说的是 scale 的编码，元素仍是 FP8 E4M3（1 字节）；DeepInfer 主 KV「per-16, e4m3」的 e4m3 也是 scale 类型、元素是 FP4 E2M1（0.5 字节，2.3 §4 已辨析）。误读使每元素字节虚高，整条压缩比账同步错。规避：格式三元组（元素类型、块大小、scale 类型）分开读。

## 3. GPTQ：二阶误差补偿

OPT-66B 在 WikiText-2 上，全精度 perplexity 9.34，直接 4 bit 就近舍入（RTN）爆到 110，GPTQ 压回 9.55、加 act-order 后追平 9.34（[Frantar et al., 2022](../references.md)，arXiv:2210.17323，[A]，表 3 与 repo README）。同一个位宽，差距为什么这么大？因为 RTN 逐权重独立舍入、不管误差之间的相关，GPTQ 用二阶信息把已量化权重的误差补偿到未量化权重上。

GPTQ 的算法谱系是 OBS（二阶权重剪枝）→ OBQ（逐权重量化 + 误差补偿）→ GPTQ（三步松弛），每一步的动机都是复杂度与数值稳定性，谱系描述出自 GPTQ 原文 §2-3（OBS / OBQ 作为命名框架引用，不另立条目）。OBQ 逐行处理、按贪心序量化单个权重并补偿其余，复杂度 $O(d_{\mathrm{out}} \cdot d_{\mathrm{in}}^{3})$ ——立方依赖让它在大模型上极贵（ResNet-50 25M 参数约 1 GPU 小时）。GPTQ 的三步把复杂度降下来且保住精度。

### 3.1 层重建目标与 Hessian

路线：把层重建目标写成逐行二次型 → 固定被量化列的误差、对其余全精度列求约束极小 → 用分块逆矩阵恒等式把解写成逆 Hessian 形式（OBS 更新式）——关键一步是约束极小解与逆 Hessian 列的换算。下面先立目标与 Hessian，补偿的三步推导见 §3.2。

GPTQ 逐层求解重建问题：找量化权重 $\hat{W}$ 最小化层输出的平方误差（原文式 (1)）：

$$
\hat{W} = \mathrm{argmin}_{\hat{W}} \lVert WX - \hat{W}X \rVert_F^2 \qquad (5)
$$

符号表： $W$ = 全精度权重 $[d_{\mathrm{out}}, d_{\mathrm{in}}]$ ； $X$ = 该层校准输入 $[d_{\mathrm{in}}, n]$ （GPTQ 用 128 段 × 2048 token 的 C4 随机片段）； $\lVert \cdot \rVert_F$ = Frobenius 范数（全元素平方和）。式 (5) 可按 $W$ 的行拆成独立子问题：记一行权重为列向量 $w \in \mathbb{R}^{d_{\mathrm{in}}}$ ，误差 $e = w - \hat{w}$ ，则该行输出误差 $\lVert e^{\top} X \rVert^2 = e^{\top} X X^{\top} e$ 。这是关于 $e$ 的二次型，Hessian 为（原文逐字）：

$$
H = 2 X X^{\top} \qquad (6)
$$

符号表： $H \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{in}}}$ = 逐行共享的 Hessian（只依赖输入 $X$ ，不依赖权重——这是 GPTQ 所有行共用一套逆 Hessian 的根据）。行目标即 $\frac{1}{2} e^{\top} H e$ 。代入教学 toy（ $d_{\mathrm{in}}$ = 2、 $n$ = 3）：

$$
X = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}, \quad X X^{\top} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad H = \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix}, \quad H^{-1} = \frac{1}{12}\begin{bmatrix} 4 & -2 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 1/3 & -1/6 \\ -1/6 & 1/3 \end{bmatrix}
$$

对角元 $H_{jj} = 2 \sum_i X_{j,i}^2$ 正比于通道 $j$ 的激活能量——act-order（§3.4）按它降序排列有直接依据。

### 3.2 误差补偿的推导（命题 1）

**命题 1**（GPTQ 最优误差补偿）：量化列 $q$ （固定 $`\hat{w}_q = \mathrm{quant}(w_q)`$ ，误差 $e_q = w_q - \mathrm{quant}(w_q)$ ），其余全精度列集合 $F$ 的最优补偿为

$$
\delta_F = -\frac{e_q}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{F,q} \qquad (7)
$$

即未量化权重更新 $w_F \leftarrow w_F + \delta_F$ 。**证明**（路线：约束二次极小 → 分块逆换算，三步）：

第一步，约束极小。把 $e$ 分块为 $(e_q, e_F)$ ， $H$ 对应分块。固定 $e_q$ 、对 $e_F$ 求 $\frac{1}{2} e^{\top} H e$ 的极小，梯度为零给出 $H_{Fq} e_q + H_{FF} e_F = 0$ ，故 $e_F = -H_{FF}^{-1} H_{Fq} e_q$ 。补偿量是 $`\hat{w}_F`$ 相对 $w_F$ 的改变： $\delta_F = -e_F = H_{FF}^{-1} H_{Fq} e_q$ 。

第二步，分块逆恒等式。记 Schur 补 $S = H_{qq} - H_{qF} H_{FF}^{-1} H_{Fq}$ ，分块逆矩阵给出 $[H^{-1}]_{qq} = S^{-1}$ 与 $[H^{-1}]_{F,q} = -H_{FF}^{-1} H_{Fq} S^{-1} = -H_{FF}^{-1} H_{Fq} [H^{-1}]_{qq}$ 。

第三步，换算。由第二步解出 $H_{FF}^{-1} H_{Fq} = -[H^{-1}]_{F,q} / [H^{-1}]_{qq}$ ，代入第一步的 $\delta_F$ 即得式 (7)。证毕。

式 (7) 就是 OBS / OBQ 的更新式：误差 $e_q$ 按逆 Hessian 第 $q$ 列的方向分摊到其余权重，归一化因子是对角元 $[H^{-1}]_{qq}$ 。正例：toy 中 $w = (0.6, 0.24)$ 、格点 $\lbrace 0, 0.5, 1.0 \rbrace$ ，先量化列 1： $e_1 = 0.6 - 0.5 = 0.1$ ， $[H^{-1}]_{11} = 1/3$ 、 $[H^{-1}]_{2,1} = -1/6$ ， $\delta_2 = -(0.1/(1/3)) \times (-1/6) = 0.05$ ，列 2 更新为 $0.24 + 0.05 = 0.29$ 、再量化落 0.5。边界反例：若 $H$ 对角占优到非对角全零（ $X$ 各通道正交）， $[H^{-1}]_{F,q} = 0$ ，补偿为零——GPTQ 退化为 RTN，二阶信息无相关可利用。

全数字对账（toy，式 (5) 目标）：RTN 直接舍入 $\hat{w} = (0.5, 0.0)$ ， $e = (0.1, 0.24)$ ， $\frac{1}{2} e^{\top} H e = 0.1832$ ；GPTQ 补偿后 $\hat{w} = (0.5, 0.5)$ ， $e = (0.1, -0.26)$ ， $\frac{1}{2} e^{\top} H e = 0.1032$ ——输出 MSE 降 43.7%。注意列 2 的逐权重误差反而从 0.24 涨到 0.26，但输出误差下降：补偿利用的是误差经 $X$ 的相关结构，不是逐权重误差最小。这正是「二阶」相对 RTN 的全部增量。

### 3.3 逆 Hessian 更新与三步松弛

量化列 $q$ 后把它移出 $F$ ，逆 Hessian 要随之降阶。OBQ 用一步高斯消元直接更新逆（原文式 (3)）：

$$
H_{-q}^{-1} = \left( H^{-1} - \frac{[H^{-1}]_{:,q} \thinspace [H^{-1}]_{q,:}}{[H^{-1}]_{qq}} \right)_{-q} \qquad (8)
$$

符号表： $H_{-q}^{-1}$ = 删去第 $q$ 行列后的逆 Hessian；下标 $-q$ = 删去对应行列。**命题 2**：式 (8) 成立。**证明（要点）**：对 $H^{-1}$ 做分块高斯消元——秩一修正 $[H^{-1}]_{:,q} [H^{-1}]_{q,:} / [H^{-1}]_{qq}$ 把第 $q$ 行列消成零，再删去即得降阶逆；完整证明留作练习 3。toy 验证： $H^{-1}$ 删列 1 后应得 $[1/4]$ ，式 (8) 代入 $[H^{-1}]_{11} = 1/3$ ：

$$
H^{-1} - \frac{[H^{-1}]_{:,1} \thinspace [H^{-1}]_{1,:}}{[H^{-1}]_{11}} = \begin{bmatrix} 0 & 0 \\ 0 & 1/4 \end{bmatrix}
$$

删行列 1 得 $[1/4]$ ，与 $H_{FF} = [4]$ 的逆一致。

GPTQ 的三步松弛（原文 §3，Fig 2）：

**Step 1 任意序量化**。OBQ 按贪心序（每步选当前误差最小的权重）量化，GPTQ 发现固定任意序在大层上精度损失很小——因为贪心序把大误差权重留到末尾、届时可补偿的未量化权重已少，两者大致抵消。于是所有行按同一列序量化， $F$ 与 $H_F^{-1}$ 对所有行相同。关键收益： $H_F$ 只依赖 $X_F$ （所有行共享），式 (8) 的逆更新只需做 $d_{\mathrm{in}}$ 次（每列一次）而非 $d_{\mathrm{out}} \cdot d_{\mathrm{in}}$ 次（每权重一次），复杂度从 $O(d_{\mathrm{out}} \cdot d_{\mathrm{in}}^{3})$ 降到 $O(\max\lbrace d_{\mathrm{out}} \cdot d_{\mathrm{in}}^{2}, d_{\mathrm{in}}^{3}\rbrace)$ ，降 $\min\lbrace d_{\mathrm{out}}, d_{\mathrm{in}}\rbrace$ 倍。脊柱 $d_{\mathrm{out}} = d_{\mathrm{in}} = 4096$ ：降 4096 倍。

**Step 2 Lazy Batch-Updates**。直接实现式 (8) 每步只算几 FLOPs 却要更新整个大矩阵，计算 / 访存比低、被带宽卡死。观察：列 $i$ 的最终舍入决策只受落在该列上的更新影响，对后续列的更新此刻无关。于是把 $B$ = 128 列分成一块，块内逐列量化、更新只限制在本块与对应的 $B \times B$ 逆 Hessian 子块；块处理完才用多权重版式 (7)(8) 全局更新剩余权重。脊柱 $d_{\mathrm{in}}$ = 4096 分 32 块。这一步不减理论计算量，但解决访存瓶颈，大模型上带来一个数量级的实际加速。

```
GPTQ 块更新（B=128，Fig 2 重绘）
列:  [ 块 0 (128列) ][ 块 1 (128列) ] ... [ 块 31 ]
      └─逐列量化─┘
      量化列 j → 误差 E[:,j] → 只更新本块内 j 之后的列
      块末 ──────────────────► 用累积误差 E 全局更新所有剩余列
逆 Hessian：预计算 Cholesky 因子，块内取对应行（不迭代降阶）
```

**Step 3 Cholesky Reformulation**。式 (8) 反复迭代累积数值误差，尤其额外的矩阵求逆会让 $H_F^{-1}$ 变不定（indefinite），导致补偿朝错误方向激进更新、整层量化任意坏——模型越大越几乎必然出现（几 B 参数以上至少几层中招）。两个对策：其一 dampening，给 $H$ 对角加小常数 $\lambda$ （取平均对角值的 1%），即 $H^{-1} = (2XX^{\top} + \lambda I)^{-1}$ ；toy 平均对角 4、 $\lambda$ = 0.04。其二 Cholesky 预计算：量化列 $q$ 时只需 $H_{F_q}^{-1}$ 的第 $q$ 行（从对角元起），而式 (8) 的逐行消去本质就是对 $H^{-1}$ 做 Cholesky 分解（差别仅在 Cholesky 把第 $q$ 行除以 $\sqrt{[H^{-1}]_{qq}}$ ）。于是用稳定的 Cholesky kernel 一次性预计算全部所需行： $H^{-1} \leftarrow \mathrm{Cholesky}(H^{-1})^{\top}$ （上三角），配合 mild dampening 即可在大模型上稳健运行，还顺带提速。算法 1 的块内更新因此写成：量化列 $j$ 、误差 $E_{:,j} = (W_{:,j} - Q_{:,j}) / [H^{-1}]_{jj}$ 、块内更新 $W_{:,j:(i+B)} \leftarrow W_{:,j:(i+B)} - E_{:,j} \cdot [H^{-1}]_{j,j:(i+B)}$ （ $i$ = 块首列， $j:(i+B)$ 指块内第 $j$ 列到块末的剩余列，是块局部切片而非全局），其中 $[H^{-1}]$ 已是 Cholesky 因子。

### 3.4 act-order 与工程开关

repo README 的三个开关（[Frantar et al., 2022](../references.md)，[A]，IST-DASLab/gptq）：`--act-order` 按激活幅度降序量化列（即按 $H_{jj} = 2\sum_i X_{j,i}^2$ 降序，先量化激活能量大的通道）；`--static-groups` 预先定死 group 网格，使 act-order 无需推理侧改动（否则列重排会拖慢推理）；`--true-sequential` 在单个 Transformer block 内也顺序量化。act-order 对 outlier 模型收益显著：OPT-66B 4 bit Wiki2 PPL 9.55 → 9.34、3 bit 14.16 → 9.95；LLaMA-7B 7.15 → 6.09（act-order + true-sequential）。机制：大激活通道对输出误差贡献大（式 (6) 的 $H_{jj}$ ），先量化它们时其余列尚未量化、可供补偿；留到末尾则补偿空间枯竭。GPTQ 量化 OPT-175B / BLOOM-176B 约 4 GPU 小时（单 A100-80GB），校准集仅 128 × 2048 token 的 C4 片段、不见任务数据（zero-shot）。

> **提点**：GPTQ 与 AWQ 的取舍是「重建 vs 泛化」——GPTQ 在校准集上最小化层重建误差，AWQ 原文指出这可能过拟合校准集、且对 LLaMA-7B / OPT-66B 需要 reorder 技巧才 work；AWQ 只测逐通道激活平均幅度、对校准集依赖极小。两者不是谁取代谁，是校准数据敏感度与硬件友好度的不同权衡。

## 4. AWQ：激活感知逐通道缩放

OPT-6.7B INT3-g128 量化，把 1% 的 salient 权重通道留 fp16，perplexity 从 43.2 降到 13.0（[Lin et al., 2023](../references.md)，arXiv:2306.00978，[A]，图 2 口径；与表 1/2 的 RTN 基线差见 §4.2 口径注）——但 mixed-precision 硬件不友好。能不能不留 fp16、只靠缩放就保护这 1%？AWQ 的答案是：能，而且缩放是一个数学等价变换。

关键观察：salient 通道由**激活分布**定，不是权重分布。按权重 L2 范数选 1% 留 fp16，收益与随机选相近；按激活幅度选，0.1%-1% 的通道就能把 PPL 从 43.2 压到 13.0。机制：激活幅度大的输入通道处理更重要的特征，对应权重通道更 salient。所谓 salient weights，是指对应输入通道激活幅度大、对层输出误差贡献集中的那 0.1%-1% 权重通道（AWQ 原文口径）。

### 4.1 缩放等价性（命题 3）

**命题 3**（逐通道缩放等价）：对任意可逆逐通道缩放 $\mathrm{diag}(s)$ （ $s \in \mathbb{R}^{d_{\mathrm{in}}}$ ，各分量非零），全精度下

$$
(W \thinspace \mathrm{diag}(s)) \thinspace (\mathrm{diag}(s)^{-1} X) = WX \qquad (9)
$$

**证明**（路线：矩阵结合律 + 对角互逆，两步）：由结合律， $(W \mathrm{diag}(s))(\mathrm{diag}(s)^{-1} X) = W (\mathrm{diag}(s) \mathrm{diag}(s)^{-1}) X$ ；而 $\mathrm{diag}(s) \mathrm{diag}(s)^{-1} = I$ ，故等于 $WX$ 。证毕。正例： $W = [0.6]$ 、 $X = [2]$ 、 $s = [2]$ ， $(0.6 \times 2) \times (2/2) = 1.2 \times 1 = 1.2 = 0.6 \times 2$ 。边界： $s_j = 0$ 时 $\mathrm{diag}(s)$ 不可逆、等价不成立——所以缩放因子必须非零。

式 (9) 在全精度下严格成立，但量化打破等式： $Q(W \mathrm{diag}(s))(\mathrm{diag}(s)^{-1} X) \neq WX$ 。AWQ 的全部设计就是选 $s$ 让这个打破后的误差最小。 $\mathrm{diag}(s)^{-1} X$ 通常融进前一个算子（LayerNorm 等），不额外增加 kernel 调用。

### 4.2 为什么放大 salient 通道降误差

路线：写出对称 absmax 量化的误差表达式 → 比较缩放前后的误差比 → 指出 $\Delta' \approx \Delta$ 时比值由 $1/s$ 主导。考虑一组权重 $\mathbf{w}$ ，量化函数 $Q(\mathbf{w}) = \Delta \cdot \mathrm{Round}(\mathbf{w}/\Delta)$ ， $\Delta = \mathrm{absmax}(\mathbf{w}) / 2^{q-1}$ （ $q$ = 位宽，同式 (1)）。把单个 salient 元素 $w$ 乘 $s > 1$ 、对应激活 $x$ 除 $s$ ，量化输出为（原文式 (2)）：

$$
Q(w \cdot s) \cdot \frac{x}{s} = \Delta' \cdot \mathrm{Round}\left( \frac{ws}{\Delta'} \right) \cdot x \cdot \frac{1}{s} \qquad (10)
$$

符号表： $\Delta'$ = 缩放后的新步长。误差来自 $\mathrm{Round}$ （ $\Delta$ 与 $x$ 是 fp16、无量化误差）。原文三条经验事实：(1) $\mathrm{Round}$ 误差近似 $[0, 0.5]$ 均匀分布、均值 0.25，缩放不改变其分布；(2) 放大单个元素通常不改变组内最大值，故 $\Delta' \approx \Delta$ ；(3) 于是缩放前后误差比为

$$
\frac{\mathrm{Err}(Q(ws)(x/s))}{\mathrm{Err}(Q(w)x)} = \frac{\Delta'}{\Delta} \cdot \frac{1}{s} \qquad (11)
$$

代入 $s$ = 2、 $\Delta'/\Delta \approx 1$ ：比值 ≈ 0.5，salient 通道相对误差减半。这就是「放大 salient 通道保护它」的机制——不是让该权重误差绝对值变小，而是让它的**相对**误差（除以 $s$ 后）变小。

但 $s$ 不能无限大：放大 salient 元素可能让它变成组内最大值，抬高 $\Delta'$ ，非 salient 通道的误差被 $\Delta'/\Delta$ 放大。原文表 2（OPT-6.7B，1% salient 通道乘 $s$ ）给出权衡的全数字： $s$ = 1 时 PPL 23.54（即 RTN）、 $\Delta'/\Delta$ = 1； $s$ = 2 时 PPL 11.92、8.2% 的组 $\Delta$ 改变、平均 $\Delta'/\Delta$ = 1.038、误差比 0.519； $s$ = 4 时 PPL 反弹到 12.36、21.2% 的组 $\Delta$ 改变、 $\Delta'/\Delta$ = 1.213。最优在 $s$ = 2：再大则非 salient 通道误差反超收益。

> **口径注**（AWQ 两套 RTN 基线）：§4 开场引图 2 的「RTN 43.2 → 1% salient fp16 13.0」与本节表 2 的「s=1（RTN）23.54」在原文即不一致——图 2 caption 写 INT3-g128 RTN=43.2，表 1/2 同设置 RTN=23.54（表 1 act-1%=11.39）。两处各引出处（图 2 / 表 1/2）、不混算同一基线；本书引图 2 标「图 2 口径」、引表 2 标「表 2 口径」。差异来源原文未解释，读者对账时以各自出处为准。

### 4.3 grid search α

为同时顾及 salient 与非 salient，AWQ 不手选 $s$ ，而是搜最优逐通道缩放。目标（原文式 (4)）：

$$
s^{\ast} = \mathrm{argmin}_{s} \thinspace \mathcal{L}(s), \qquad \mathcal{L}(s) = \lVert Q(W \mathrm{diag}(s))(\mathrm{diag}(s)^{-1} X) - WX \rVert \qquad (12)
$$

量化函数不可微，直接反传不稳定。AWQ 把搜索空间收成一个超参（原文式 (5)）：

$$
s = s_X^{\alpha}, \qquad \alpha^{\ast} = \mathrm{argmin}_{\alpha} \thinspace \mathcal{L}(s_X^{\alpha}) \qquad (13)
$$

符号表： $s_X$ = 逐通道激活平均幅度（ $\alpha$ 本章局部 = 缩放指数，见 §1）； $\alpha \in [0,1]$ ，0 = 不缩放、1 = 搜索空间内最激进。最小代入：某通道 $s_X$ = 4，取 $\alpha$ = 0.5 得 $s = 4^{0.5} = 2$ （该通道权重放大 2 倍、对应激活除 2）； $\alpha$ = 0 得 $s$ = 1（不缩放）、 $\alpha$ = 1 得 $s$ = 4（最激进）。 $\alpha$ 用 grid search（grid size 20）在 $[0,1]$ 上搜，再叠加 weight clipping 最小化 MSE。group size 128、INT4/INT3 weight-only、校准集取 Pile 的小子集（避免过拟合特定任务）。机制：salient 由激活幅度定（「activation-aware」）， $s_X^{\alpha}$ 把「保护强度」压成单旋钮， $\alpha$ 大则 salient 通道放大更多、非 salient 风险更大——式 (11) 的权衡由 grid search 自动定标。AWQ 不依赖回归或反传，对校准集依赖极小（只测逐通道平均幅度），保住校准分布外的泛化能力。

> **坑**：把式 (9) 的等价当成量化后也等价——触发条件：以为「等价变换」意味着 $Q(W\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}X) = WX$ ；现象：忽略 $\Delta'$ 变化，误判任意大 $s$ 都无损；后果：非 salient 通道误差被 $\Delta'/\Delta$ 放大、PPL 反弹（表 2 的 $s$ = 4）。规避：等价只在全精度成立，量化后误差比按式 (11) 算、 $s$ 由 grid search 定。

## 5. SmoothQuant：把激活 outlier 迁移到权重

W8A8 把权重和激活都压到 INT8，权重好压、激活难压——激活 outlier 幅度可达正常值的 ~100×（[Xiao et al., 2022](../references.md)，arXiv:2211.10438，[A]）。per-channel 激活量化能保精度，却和 INT8 GEMM kernel 不兼容。怎么办？SmoothQuant 的答案：不在运行时做 per-channel 激活量化，而是离线把激活的量化难度**迁移**到权重上，让两者都变好压。

三个分布事实（原文 §2）：(1) 少数激活通道幅度极大（大于 70）；(2) 同一通道跨 token 的方差小（outlier 通道持续地大）；(3) 原始权重分布平坦均匀。outlier 持续落在固定通道，所以可以离线统计、离线迁移。per-tensor INT8 下，通道 $i$ 的有效量化级数为 $2^{8} \cdot m_i / m$ （ $m_i$ = 通道 $i$ 最大幅度， $m$ = 整矩阵最大幅度）：outlier 把 $m$ 抬到 100、正常通道 $m_i$ = 1，有效级数只剩 256/100 ≈ 2.56（2-3 级），信息基本尽毁。per-channel 激活量化能救，但 INT8 GEMM 的 scale 只能沿外维（激活的 token 维、权重的输出通道维）施加、在内维（输入通道维）插不进低吞吐指令——所以 per-channel 激活量化硬件不可行（原文 §3、Fig 3；per-channel 保精度的模拟验证见表 1）。

### 5.1 迁移等价与 α 权衡

SmoothQuant 原文用行向量约定写迁移（原文式 (3)）： $Y = (X \mathrm{diag}(s)^{-1})(\mathrm{diag}(s) W)$ ， $X \in \mathbb{R}^{n \times d_{\mathrm{in}}}$ 、 $W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$ 。在本章统一约定（§1， $W [d_{\mathrm{out}}, d_{\mathrm{in}}]$ 、 $X [d_{\mathrm{in}}, n]$ 、 $Y = WX$ ）下是同一个恒等式：

$$
Y = (W \thinspace \mathrm{diag}(s)) \thinspace (\mathrm{diag}(s)^{-1} X) = WX \qquad (14)
$$

即命题 3 的逐通道缩放等价——**SmoothQuant 与 AWQ 共用同一个等价变换**，差别只在 $s$ 的选法与量化目标（AWQ weight-only 保护 salient，SmoothQuant W8A8 平滑 outlier）。等价性证明同命题 3（结合律 + 对角互逆），迁移在全精度下严格不改输出。 $\mathrm{diag}(s)^{-1}$ 融进前一层参数（离线），运行时激活已平滑、无需额外 scaling。

$s$ 怎么选？两个极端都坏： $s_j = \mathrm{absmax}(X_j)$ 把激活全压平（所有通道同最大值），但难度全推给权重，权重 outlier 化、误差大； $s_j = 1/\mathrm{absmax}(W_j)$ 把难度全推给激活，激活误差大。引入迁移强度 $\alpha$ 折中（原文式 (4)）：

$$
s_j = \frac{\mathrm{absmax}(X_j)^{\alpha}}{\mathrm{absmax}(W_j)^{1-\alpha}} \qquad (15)
$$

符号表： $\mathrm{absmax}(X_j)$ = 输入通道 $j$ 的激活最大幅度（跨 token）， $\mathrm{absmax}(W_j)$ = 通道 $j$ 的权重最大幅度（跨输出通道）； $\alpha$ = 迁移强度（本章局部）。**命题 4**（α 难度分配）：记 $m_{X,j} = \mathrm{absmax}(X_j)$ 、 $m_{W,j} = \mathrm{absmax}(W_j)$ ，迁移后平滑激活通道 $j$ 的最大幅度为 $(m_{X,j} m_{W,j})^{1-\alpha}$ ，调整后权重通道 $j$ 的最大幅度为 $(m_{X,j} m_{W,j})^{\alpha}$ 。**证明**（路线：代入式 (15) 到两个最大幅度定义，两步）：平滑激活 $\hat{X}$ 通道 $j$ 除以 $s_j$ ，最大幅度 $m_{X,j}/s_j = m_{X,j} \cdot m_{W,j}^{1-\alpha} / m_{X,j}^{\alpha} = (m_{X,j} m_{W,j})^{1-\alpha}$ ；调整后权重 $\hat{W}$ 通道 $j$ 乘 $s_j$ ，最大幅度 $s_j \cdot m_{W,j} = m_{X,j}^{\alpha} \cdot m_{W,j}^{\alpha} = (m_{X,j} m_{W,j})^{\alpha}$ 。证毕。

推论（ $\alpha$ = 0.5）：两者都等于几何平均 $\sqrt{m_{X,j} m_{W,j}}$ ——权重与激活在对应通道共享相同的最大幅度，量化难度均分。这正是原文「 $\alpha$ = 0.5 是 OPT/BLOOM 的 well-balanced point」的数学根据。 $\alpha \to 1$ 激活最大幅度 $\to 1$（全压平）、权重 $\to m_X m_W$（难度全归权重）； $\alpha \to 0$ 相反。

全数字实例（教学 toy，outlier 通道 $m_X$ = 100、平坦权重 $m_W$ = 1，积 = 100）： $\alpha$ = 0.5 时 $s$ = 10，平滑激活最大 10、调整后权重最大 10（几何平均 $\sqrt{100}$ = 10，均分）； $\alpha$ = 0.75 时 $s$ = 100^{0.75} ≈ 31.6，激活最大 100^{0.25} ≈ 3.16、权重最大 100^{0.75} ≈ 31.6； $\alpha$ = 1 时激活最大 1、权重最大 100。有效量化级数账（另取正常通道 $m_X$ = 1、 $m_W$ = 1，与上方 outlier 通道对照）：迁移前 per-tensor max 被 outlier 抬到 100，正常通道 256 × 1/100 ≈ 2.56 级； $\alpha$ = 0.5 迁移后 outlier 通道激活 max 100 → 10、正常通道 $s$ = 1 故 max 仍 1，per-tensor max 降到 10，正常通道 256 × 1/10 = 25.6 级（10× 改善）； $\alpha$ = 1 才把激活全压平到 max 1、正常通道满级 256（难度全归权重）。注意「迁移后各通道激活 max 统一到 10」不成立：命题 4 给的是通道 max = $(m_X m_W)^{1-\alpha}$ ，正常通道积 1 → max 1、outlier 通道积 100 → max 10，两者不等——均分的是**同一通道内**权重与激活的难度，不是跨通道激活幅度。原文口径： $\alpha$ = 0.5 对 OPT/BLOOM 是 sweet spot（原文图 10 消融区间 0.4-0.6，过小激活难压、过大权重难压）；GLM-130B 激活 outlier 占比更高（~30%），取 $\alpha$ = 0.75 把更多难度迁给权重。SmoothQuant 使 OPT-175B / BLOOM-176B / GLM-130B / MT-NLG 530B 的 W8A8 保住精度，530B 单节点（8 GPU）serving、OPT-175B 用 FP16 一半的 GPU 数且更快。官方实现 mit-han-lab/smoothquant（论文摘要逐字自链）。

```
SmoothQuant 迁移（α=0.5，outlier 通道）
激活 X:  通道 j 最大 100 ──÷s=10──► X̂ 通道 j 最大 10  ┐
                                                      ├─ 都=10，难度均分
权重 W:  通道 j 最大 1   ──×s=10──► Ŵ 通道 j 最大 10  ┘
全精度等价：(W diag(s))(diag(s)⁻¹ X) = WX；离线融合，运行时零额外 scaling
```

> **提点**：AWQ 与 SmoothQuant 是同一个等价变换的两种用法——AWQ 的 $s = s_X^{\alpha}$ 放大 salient 权重通道以保护它（weight-only），SmoothQuant 的 $s_j = m_X^{\alpha}/m_W^{1-\alpha}$ 把激活 outlier 摊到权重（W8A8）。前者 $\alpha$ 调「保护多少 salient」，后者 $\alpha$ 调「难度迁多少给权重」；数学骨架都是命题 3，差别在 $s$ 的物理意图。

## 6. LLM.int8()：outlier 维 fp16 分解

6.7B 是一个相变点：超过它，outlier 出现在所有层、75% 的序列维，幅度达正常值 20×（[Dettmers et al., 2022](../references.md)，arXiv:2208.07339，[A]）。13B 模型每 2048 token 序列有多达 150,000 个 outlier，却集中在 ≤6-7 个特征维（~0.1% 特征）。这个稀疏结构怎么利用？LLM.int8() 的答案：把 outlier 特征维隔离出来走 fp16，其余 >99.9% 走 int8。

outlier 为什么致命：把 outlier 特征维置零，attention softmax 的 top-1 概率质量降 >20%、验证 perplexity 恶化 600-1000%；对照置零同等数量的随机特征，概率只降 ≤0.3%、PPL 恶化 ~0.1%。outlier 不是噪声，是模型预测性能的承载者——所以不能简单 clip 或忽略，必须高精度保留。outlier 判据（原文 §3）：幅度 ≥ 6.0、影响 ≥25% 的层、影响 ≥6% 的序列维（此 6.0 是幅度阈值，与本章 $\alpha$ 记号无关）。

LLM.int8() 两部分。**第一部分 vector-wise 量化**：把矩阵乘看作一串独立内积，给 $X$ 的每行、 $W$ 的每列各一个归一化常数（absmax 缩放到 $[-127,127]$ ，即 $\Delta=\mathrm{absmax}/127$ ，取 $2^{q-1}-1$ 约定、与式 (1) 的 AWQ $2^{q-1}$ 约定差一格，见 §2），Int32 累加输出，再用行列常数的外积反量化（原文式 (7)）：

$$
C_{\mathrm{f16}} \approx \frac{1}{c_x \otimes c_w} \thinspace C_{\mathrm{i32}} = S \cdot Q(X_{\mathrm{f16}}) \thinspace Q(W_{\mathrm{f16}}) \qquad (16)
$$

符号表： $c_x$ = 逐行（token）归一化常数向量， $c_w$ = 逐列（输出通道）常数向量； $\otimes$ = 外积； $S = 1/(c_x \otimes c_w)$ = 反量化因子； $C_{\mathrm{i32}}$ = Int32 矩阵乘输出。每个内积独立归一化，比 per-tensor 单常数精度高——但 vector-wise 对 outlier 特征维仍无效（outlier 是列方向、贯穿所有行，行归一化压不住）。

**第二部分 mixed-precision decomposition**：把 outlier 特征维集合 $O$ （幅度超阈值的维）隔离走 fp16 矩阵乘，其余走 int8（原文式 (8)）：

$$
C_{\mathrm{f16}} \approx \sum_{h \in O} X_{\mathrm{f16}}^{h} W_{\mathrm{f16}}^{h} + S_{\mathrm{f16}} \cdot \sum_{h \notin O} X_{\mathrm{i8}}^{h} W_{\mathrm{i8}}^{h} \qquad (17)
$$

符号表： $O$ = outlier 特征维集合（ $\vert O \vert \le 7$ 对 ≤13B 模型）；上标 $h$ = 第 $h$ 个特征维的子矩阵；两部分输出在 fp16 累加。最小代入（结构例， $n$ = 1 token、 $h$ = 2 特征、 $o$ = 1 输出）： $X=[20, 1]$ 、 $W=[0.5, 0.5]^{\top}$ ，特征 0 为 outlier（ $O=\lbrace 0\rbrace$ ）。真值 $C=20×0.5+1×0.5=10.5$ ；式 (17) 拆两部分——outlier 部 fp16 $20×0.5=10$ （精确）、normal 部 int8 $1×0.5=0.5$ （本例落格点），和 10.5 与真值一致。对照不隔离 outlier、整行 int8（行 absmax=20、 $\Delta$ =20/127≈0.157）：正常值 1 落格点 0.945（ $6×0.157$ ）、误差 0.055——outlier 抬高步长、正常值精度被牺牲；分解后 normal 部用自己的 absmax=1 归一，该误差消失。因 $\vert O \vert \le 7$ ，分解只多占 ~0.1% 内存，>99.9% 的值仍走 int8——BLOOM-176B 内存降 1.96×。LLM.int8() 在 125M 到 175B 全程保住 16-bit perplexity 与 zero-shot 精度，是唯一有良好 scaling 趋势的 8-bit 方法（普通 absmax / row-wise / zeropoint 在 6.7B 后崩坏）。

与 SmoothQuant 的对照是理解 W8A8 两条路线的钥匙：LLM.int8() 用 mixed-precision 分解（outlier 留 fp16），精度稳但分解在硬件加速器上难高效实现；SmoothQuant 用等价迁移把 outlier 摊平、全矩阵统一 INT8，硬件友好（标准 INT8 GEMM）。前者保精度靠「隔离」，后者靠「摊平」——隔离要 mixed-precision kernel，摊平只要离线改权重。

> **坑**：把 LLM.int8() 的「无退化」外推到 4 bit——触发条件：看到「175B 无性能退化」就以为低 bit 都无损；现象：忽略那是 8-bit（INT8）且 outlier 走 fp16 的口径；后果：误判 4 bit weight-only 也能无损。规避：LLM.int8() 的无损口径限定 8-bit + outlier fp16 分解；4 bit 的精度账归 GPTQ/AWQ（§3/§4），且依赖模型与任务（§8）。

## 7. FP8 / FP4 部署：从 V3 blockwise 到 V4.1 MXFP4 expert

V3 用 128×128 权重块 + fp32 scale，V4.1-Flash config 写的是 `weight_block_size` [32,32]、`scale_fmt` ue8m0、`expert_dtype` fp4（[W]）。粒度细了 16 倍、scale 从 4 字节变 1 字节、expert 从 FP8 变 FP4——这笔账怎么算？

V3 的 FP8 方案 recap training 1.3 §6（不重推）：全 E4M3、激活 $1 \times 128$ tile / 权重 $128 \times 128$ 块、在线量化（ $s_g = \mathrm{absmax}/448$ ）、每 128 个元素把部分和提升到 CUDA Core fp32 累加（ $N_C$ = 128，对抗 Tensor Core ~14 bit 累加精度）、E5M6 例外、相对 BF16 基线 loss 偏差 <0.25%（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§3.3）。权重块 scale 开销 4/16384 ≈ 0.024%（training 1.3 §6.3）。

V4.1-Flash 把粒度推到 32×32 + ue8m0。字节账（式 (3)）：32×32 = 1024 个 FP8 元素（1024 字节）配 1 字节 ue8m0 scale，每元素 (1024+1)/1024 ≈ 1.00098 字节，scale 开销 1/1024 ≈ 0.098%——约是 V3 的 4 倍（块细 16 倍但 scale 从 4 字节缩到 1 字节，净 4 倍）。代价换收益：更细的块把 outlier 限制在更小范围，精度收益沿 training 1.3 §6 的 blockwise 论证；ue8m0 纯指数 scale 的乘除是精确指数加减，不引入额外量化误差，且对 Blackwell 硬件友好——DeepGEMM 在 SM100 用 packed UE8M0（4 个 pack 进一个 int），SM90 用 FP32 scale（repo README，归 3.3 走读）。

expert 侧换 FP4。DeepInfer 生产披露（官方披露，doc 逐字，核实 2026-11-23，[DeepSeek-AI, DeepInfer doc](../references.md)，教学参照 + 官方实现）「Where is FP4 used?」标题下三条：主 KV per-16 e4m3「like NVFP4 but without the global scale」、indexer KV per-32 ue8m0 MXFP4、**MoE routed-expert weights per-32 ue8m0 MXFP4**。第三条与 config `expert_dtype` fp4 互证：routed expert 权重走 MXFP4（E2M1 元素 + per-32 E8M0 scale），每元素 0.53125 字节（式 (3)），对 FP8 权重再砍半。主 KV / indexer KV 的格式账已在 2.3 §4 交付（回链），本章只登记权重侧。

| 维度 | V3 | V4.1-Flash | 出处 |
| --- | --- | --- | --- |
| 非 expert 权重元素 | FP8 E4M3 | FP8 E4M3 | training 1.3 / config [W] |
| 权重块大小 | 128×128 | 32×32 | training 1.3 / config [W] |
| scale 类型 | fp32（4 字节） | ue8m0（1 字节） | training 1.3 / config [W] |
| 权重块 scale 开销 | ≈0.024% | ≈0.098% | 式 (3) |
| routed expert | FP8 | FP4（MXFP4 per-32 ue8m0） | config [W] + doc 官方披露 |
| expert 每元素字节 | ≈1.00024 | 0.53125 | 式 (3) |

对照读法：V4.1 非 expert 权重仍 FP8（字节数与 V3 几乎相同，差别在粒度与 scale 编码），真正的位宽下探在 expert——FP8 → FP4 砍半。这与 MoE 的结构一致：routed expert 占参数大头（V4.1 384 routed + 1 shared，1.2 §5），压 expert 收益最大；attention / 非 expert 部分对精度更敏感，保 FP8。模型侧形式化（为何 expert 可压更狠）归 models 维度（V4.1 待补），本章只引 config / doc 做 serving 字节账。量化 GEMM 与 fused MoE kernel 怎么消费这些格式（DeepGEMM fine-grained scaling / Marlin W4A16 / Mega MoE FP8×FP4）归 3.3，指令级归 gpu 6.2。

## 8. 质量评估口径：perplexity / task，没有普适「无损」

GPTQ 4 bit 在 OPT-66B 上追平全精度（9.34），LLM.int8() 8 bit 在 175B 上「无退化」——能不能据此说量化「无损」？不能。这些是**特定模型 × 特定位宽 × 特定粒度 × 特定任务**下的论文自报结果，不是普适定理。

评估口径有两层。**perplexity**（语言建模，WikiText-2 / C4）是严格指标：对数似然的指数，量化误差直接抬高它。GPTQ 原文称 perplexity 是「very stringent accuracy metric」。**task benchmark**（zero-shot：LAMBADA / HellaSwag / PIQA / WinoGrande / MMLU 等）测下游能力：perplexity 接近不代表 task 精度保持，两者要分别报。SmoothQuant 对 GLM-130B 换用 MMLU / MNLI / QNLI / LAMBADA（因部分 benchmark 在其训练集里），正是 task 口径随模型变的例子。

为什么没有普适无损：量化误差经层间传播累积，影响取决于模型规模、outlier 结构、位宽、粒度、校准集与任务。同一方法在不同模型上表现可以相反——GPTQ 4 bit 在 OPT-66B 追平全精度，但 3 bit RTN 崩坏（PPL 上千）；LLM.int8() 的「无退化」限定 8-bit + outlier fp16 分解，不外推到 4 bit（§6 坑注）；SmoothQuant 原文表 6/7 用「lossless W8A8」描述 LLaMA / Llama-2 / Falcon / Mistral / Mixtral 的特定设置，那是条件论断不是普适保证。本章口径：**不写普适「无损」**，所有收益数字标论文自报口径，端到端影响以 perplexity / task benchmark 实测为准。

校准集是另一处口径。GPTQ 用 128 × 2048 token 的 C4 随机片段（generic text、不见任务数据，结果保持 zero-shot）；AWQ 用 Pile 小子集、只测逐通道平均幅度（对校准集依赖极小，防过拟合）；SmoothQuant 用校准样本离线统计 $\mathrm{absmax}$ 。校准集分布与部署分布的偏移会改变量化质量——AWQ 原文指出 GPTQ 的重建可能过拟合校准集、损失校准分布外的泛化能力（§3 提点）。所以「无损」还隐含「校准与部署同分布」的前提。

| 算法 | 路线 | 关键机制 | 论文自报质量锚 | 校准依赖 |
| --- | --- | --- | --- | --- |
| GPTQ | weight-only W4A16 | 二阶误差补偿 + Cholesky + act-order | OPT-66B 4bit PPL 9.55→9.34（追平 full） | 128×2048 C4，重建可能过拟合 |
| AWQ | weight-only W4A16 | 激活感知逐通道缩放 + grid search α | OPT-6.7B INT3-g128 近 1% FP16（13.0，图 2 口径） | Pile 小子集，依赖极小 |
| SmoothQuant | W8A8 | outlier 迁移 α=0.5/0.75 | OPT-175B/BLOOM-176B/GLM-130B/530B 保精度 | 校准样本统计 absmax |
| LLM.int8() | W8A8 + fp16 分解 | vector-wise + outlier 维隔离 | 125M-175B 保 16-bit PPL | 无需校准（运行时分解） |

四算法的系统含义汇到一处：weight-only（GPTQ/AWQ）省权重读带宽、利好 memory-bound decode（§1），但 dequant 要融进 GEMM mainloop（W4A16 混合精度，kernel 归 3.3 Marlin）；W8A8（SmoothQuant/LLM.int8）权重激活同精度、dequant 可放 epilogue，标准 INT8 GEMM 即可，但 LLM.int8() 的 mixed-precision 分解硬件难高效。格式与算法的选择最终落到 3.3 的算子决策表（精度 × 硬件 × batch 形态）。

## 总结

| 论断 | 账 | 回指 |
| --- | --- | --- |
| weight-only 量化砍权重读字节，decode 强度 $`\propto 1/\mathrm{bytes}_{\mathrm{param}}`$ | bf16→4bit 强度 ×4，带宽下界 ÷4 | §1、1.2 式 (4) |
| 每元素字节 = (数据 + scale + zero-point)/块元素数；group 对称开销 $16/(gq)$ | INT4-g128 对称 3.125%、非对称 6.25% | 式 (3)(4)、§2 |
| MXFP8 1.03125 / MXFP4 0.53125 / NVFP4 0.5625 B/元素 | 对 bf16 1.94× / 3.76× / 3.56× | 式 (3)、§2、2.3 §4 |
| GPTQ 最优补偿 $\delta_F = -(e_q/[H^{-1}]_{qq})[H^{-1}]_{F,q}$ | toy 输出 MSE 0.1832→0.1032（降 43.7%） | 命题 1、式 (7)、§3.2 |
| 逆 Hessian 降阶 = 秩一修正 + 删行列；GPTQ 以 Cholesky 预计算替代迭代 | 复杂度 $O(d_{\mathrm{out}} d_{\mathrm{in}}^3)\to O(\max\lbrace d_{\mathrm{out}} d_{\mathrm{in}}^2, d_{\mathrm{in}}^3\rbrace)$ | 命题 2、式 (8)、练习 3 |
| 逐通道缩放全精度等价 $(W\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}X)=WX$；量化打破等式 | AWQ 误差比 $(\Delta'/\Delta)(1/s)$， $s$=2 时 ≈0.5 | 命题 3、式 (9)(11)、练习 2 |
| SmoothQuant 迁移后激活/权重通道最大幅度 = $(m_X m_W)^{1-\alpha}$ / $(m_X m_W)^{\alpha}$ | $\alpha$=0.5 两者 = 几何平均，同通道权重/激活难度均分（非跨通道均平） | 命题 4、式 (15)、§5.1 |
| LLM.int8() outlier 维 fp16 分解，>99.9% 值 int8，outlier 维数 ≤7 多占 ~0.1% 内存 | BLOOM-176B 内存 1.96×，保 16-bit PPL | 式 (17)、§6 |
| V4.1 非 expert FP8 32×32 ue8m0、expert MXFP4 per-32 | 权重块 scale 开销 0.098% vs V3 0.024%；expert 字节砍半 | §7、config [W] + doc |
| 质量无普适「无损」，按模型×位宽×粒度×任务实测 | OPT-66B GPTQ 4bit 追平 full 是条件论断 | §8 |

## 练习

1.（计算 · [10]）格式字节账（脊柱 $W$ 4096×4096，bf16 = 32 MiB）：(a) INT4-g128 对称的每元素字节、scale 总字节、压缩比；(b) 非对称（加 fp16 zero-point）的压缩比；(c) MXFP4（per-32 E8M0）与 NVFP4（per-16 UE4M3 + FP32 global，global 按整矩阵 4 字节摊薄）的每元素字节与压缩比。校验：(a) 开销 = 16/(128×4) = 3.125%，与式 (4) 一致；(c) 块内口径与式 (3) 代入 $k$=32 / $k$=16 一致，且与 2.3 §4 的 MXFP4 0.53125 / NVFP4 0.5625 相同。

2.（推导 · [20]）重证命题 3（逐通道缩放等价，式 (9)），写出等式成立的充要条件（ $\mathrm{diag}(s)$ 可逆）；再构造一个量化打破等式的反例：取 $W=[0.4, 1.0]$ 、 $X=[10, 1]^{\top}$ 、per-tensor 对称 INT3（ $\Delta=\mathrm{absmax}(W)/4$ ），算 $WX$ 与 $Q(W)X$ ，验证两者不等；再取 $s=[2, 1]$ 算 $Q(W\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}X)$ ，说明它也不等于 $WX$ 、且与 $Q(W)X$ 不同。校验：全精度 $WX=5$ ； $s=[2,1]$ 时 $Ws=[0.8,1.0]$ 组内 absmax 不变、 $\Delta'=\Delta$ （单元素组则 absmax 随 $s$ 变、量化恒精确，故反例须用多元素组）。

3.（推导 · [30]）GPTQ 逆 Hessian 降阶与补偿验证：(a) 证明命题 2（式 (8)）——对 $H^{-1}$ 做分块高斯消元，说明秩一修正 $[H^{-1}]_{:,q}[H^{-1}]_{q,:}/[H^{-1}]_{qq}$ 把第 $q$ 行列消零、删去即得降阶逆；(b) 用 §3.1 toy（ $H$ 对角元 4、非对角元 2）验证删列 1 后式 (8) 给出 $[1/4]$ ；(c) 用 §3.2 toy（ $w=(0.6,0.24)$ 、格点 $\lbrace 0,0.5,1.0\rbrace$ ）按命题 1 算补偿、再量化，验证输出 MSE 从 RTN 的 0.1832 降到 0.1032。校验：(b) 与 $H_{FF}=[4]$ 的逆一致；(c) $\frac{1}{2}e^{\top}He$ 两值与正文一致。

4.（计算 · [30]）SmoothQuant α 难度分配（命题 4）：取 outlier 通道 $m_X$=100、平坦权重 $m_W$=1。(a) 代入式 (15) 算 $\alpha$=0.5 / 0.75 / 1 的 $s_j$ 、平滑激活最大幅度、调整后权重最大幅度；(b) 验证 $\alpha$=0.5 时两者相等且等于几何平均；(c) 另取正常通道 $m_X$ = $m_W$ = 1（迁移后 $s$=1、max 仍 1），算迁移前后 per-tensor INT8 该通道的有效量化级数 $2^8 m_i/m$ （ $m$ = 全矩阵激活 max）；(d) 先写出你对「GLM-130B（~30% outlier）该取更大还是更小 $\alpha$ 」的预测再用命题 4 解释。校验：(a) $\alpha$=0.5 两者 = 10；(c) 迁移前 ≈2.56 级、 $\alpha$=0.5 后 25.6 级（per-tensor max 100→10）、 $\alpha$=1 才满级 256；(d) 与原文 GLM-130B $\alpha$=0.75 一致。

## 参考

- [Frantar et al., 2022](../references.md)：GPTQ，arXiv:2210.17323，[A]（key: gptq）——OBS→OBQ→GPTQ 谱系、层重建目标式 (5)、Hessian $H=2XX^{\top}$、误差补偿式 (7)、逆 Hessian 降阶式 (8)、三步松弛（任意序 / lazy batch B=128 / Cholesky + dampening λ=1% 平均对角）、act-order / static-groups / true-sequential、OPT-66B 4bit PPL 9.55→9.34、175B 约 4 GPU 小时
- [Lin et al., 2023](../references.md)：AWQ，arXiv:2306.00978，[A]（key: awq）——0.1%-1% salient 由激活幅度定、缩放等价式 (9)、误差比式 (11)、目标式 (12)、grid search $s=s_X^{\alpha}$（grid 20）+ weight clipping、group 128、INT4/INT3 weight-only、Pile 校准、OPT-6.7B INT3-g128 PPL 43.2→13.0（1% FP16，图 2 口径）、表 2 s=2 最优 11.92（表 2 口径，RTN 基线 23.54 与图 2 的 43.2 原文即不一致，见 §4.2 口径注）
- [Xiao et al., 2022](../references.md)：SmoothQuant，arXiv:2211.10438，[A]（key: smoothquant，官方 repo mit-han-lab/smoothquant）——W8A8、迁移等价式 (14)、s_j = absmax(X_j)^α / absmax(W_j)^(1-α) 式 (15)、α=0.5 OPT/BLOOM sweet spot（0.4-0.6）、GLM-130B α=0.75、有效量化级数 2^8 · m_i/m、530B 单节点、OPT-175B 半 GPU
- [Dettmers et al., 2022](../references.md)：LLM.int8()，arXiv:2208.07339，[A]（key: llm-int8，repo TimDettmers/bitsandbytes）——vector-wise 量化式 (16) + mixed-precision 分解式 (17)、outlier 6.7B 相变（所有层 / 75% 序列维 / 幅度 20× / 150,000 outliers 集中 ≤6-7 维）、置零 outlier top-1 概率降 >20% / PPL 恶化 600-1000%、outlier 维数 ≤7 多占 ~0.1% 内存、BLOOM-176B 1.96×、125M-175B 保 16-bit
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——§3.3 FP8 方案 recap training 1.3：全 E4M3、128×128 权重块 / 1×128 激活 tile、在线量化、 $N_C$ = 128 提升累加、loss 偏差小于 0.25%
- [OCP, 2023](../references.md)：OCP Microscaling Formats (MX) Specification v1.0，[C]（key: ocp-mx，官方格式规范、开放标准、非闭源）——MXFP8/MXFP6/MXFP4/MXINT8 块 32 + E8M0 scale、MXFP4 元素 E2M1
- [NVIDIA, cuBLAS / TensorRT-LLM 官方文档](../references.md)：NVFP4 格式，[C]（key: nvidia-nvfp4，开放文档、非闭源；blog 正文反爬，格式事实以官方文档为据）——E2M1 + per-16 UE4M3 + tensor-wide FP32 global、SM100、VEC16_UE4M3 + SCALAR_32F、KV cache nvfp4
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json（key: v41-flash-config，[W]，hf-mirror 传输核实，2026-11-23）——quant_method fp8 / activation_scheme dynamic / weight_block_size [32,32] / scale_fmt ue8m0 / expert_dtype fp4
- [DeepSeek-AI, DeepInfer doc](../references.md)：doc/inference_deployment_qa.md（key: deepinfer-doc，教学参照 + 官方实现，官方披露，核实 2026-11-23）——「Where is FP4 used?」主 KV per-16 e4m3（类 NVFP4 无 global scale）/ indexer KV per-32 ue8m0 MXFP4 / MoE routed-expert weights per-32 ue8m0 MXFP4