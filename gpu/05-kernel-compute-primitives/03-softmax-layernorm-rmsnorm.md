# 5.3 Softmax / LayerNorm / RMSNorm kernel（Softmax, LayerNorm and RMSNorm Kernels）

> 同一个 $4096 \times 4096$ 的 fp32 矩阵做逐行 softmax，行驻留 kernel 的 HBM 下界是 134.22 MB、40.04 μs，三趟朴素实现要搬 268.44 MB、80.08 μs——差的 2 倍全部来自「输入从 HBM 读了几遍」；而 exp 的 SFU 算力地板只要 4.34 μs，远低于带宽地板，memory-bound 钉死。本章交付 softmax 的 two-pass 对 online single-pass 访存账（命题 1：归约阶段 2 读对 1 读，行驻留达 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$ 下界）、Welford 在线方差不变式（命题 2）与 Chan 并行合并公式（命题 3，fp32 朴素方差对 $10^{4}$ 偏移数据算出 $-8$ 而 Welford 给 1.25）、LayerNorm/RMSNorm 带宽下界（命题 4）与 TransformerEngine norm kernel 走读片段 | 前置：5.1（命题 1 读下界、命题 3 shuffle 免 shared、warp/block 归约原型）、2.4（roofline 与 ridge、带宽库存）、3.4（occupancy 四约束）、4.1（向量化加载）、1.2（SFU 吞吐 16 results/cycle/SM）；下游 5.4（norm+residual+quant 融合带宽账）、6.3（FA online softmax 指令级走读、exp2 trick 与 quad 归约） | 锚点：TransformerEngine normalization（[NVIDIA, TransformerEngine](../references.md)，教学参照 + 官方实现，main pushed 2026-11-25）、FA2 softmax.h（[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]，repo b038b4974f26）、CUDA Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）、CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C]）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、cuda-samples（[NVIDIA, cuda-samples](../references.md)，教学参照）、NVIDIA datasheet（[NVIDIA, datasheet](../references.md)，[C]）——代码锚 2026-11-25 fresh tarball / contents API 一手核实

## 1. 读几遍决定快慢：三个问题与本章口径

逐行 softmax 的 FLOPs 少得可怜——每元素一次减 max、一次 exp、一次除 sum，约 3 到 5 FLOP；可它要把整行读进来、把整行写出去。一个 $4096 \times 4096$ 的 fp32 矩阵，行驻留 kernel 读一遍写一遍是 134.22 MB，带宽口径 40.04 μs；若 max 一趟、sum 一趟、normalize 再一趟，输入读三遍，流量翻到 268.44 MB、80.08 μs。FLOPs 一个没多，时间差 2 倍——差的全是搬运。本章回答三个问题：softmax 的归约结构怎么决定 HBM 读次数，two-pass 与 online single-pass 各搬多少（命题 1）？online softmax 的正确性 models 4.1 已证，standalone kernel 侧的增量是什么？方差为什么不能拿 $E[x^{2}] - E[x]^{2}$ 硬算，Welford 与 Chan 合并凭什么数值稳定（命题 2/3），norm kernel 的带宽下界又是什么形状（命题 4）？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0**（ $n_{\mathrm{SM}} = 132$，沿 5.1/5.2）；running example 为 $4096 \times 4096$ 矩阵（ $n_{\mathrm{el}} = 2^{24} = 16777216$ 个元素，规模沿 5.1），逐行处理、行长 $n_{\mathrm{row}} = 4096$。 $\mathrm{BW} = 3352$ GB/s（2.4 recap，datasheet 口径）。**精度分段声明**：§2 softmax 教学 kernel 取 **fp32**（ $\mathrm{bytes} = 4$），为的是把数值稳定与访存账讲清；§3/§4 的 norm 生产路径取 **bf16 输入、fp32 累加**（ $\mathrm{bytes} = 2$ 计 HBM 流量，compute_t 为 fp32，TransformerEngine 口径），换口径原因：norm 的实际部署精度是 bf16，而累加必须 fp32（§3 坑注）。fp32 ridge $R \approx 20$ FLOP/byte（ $66.9/3.352$）、bf16 ridge $R \approx 295$（2.4 recap）；SFU 吞吐 16 results/cycle/SM、FP32 128 results/cycle/SM（1.2 §2 Table 5 recap，[NVIDIA, Best Practices](../references.md)，[C]）；周期↔时间取 TC-boost 1.83 GHz。片上 KB = 1024 B，带宽 GB/s 按 $10^{9}$ B/s。

符号声明（spec §4.3）：**本章 $n_{\mathrm{el}}$ = 总元素数**（5.1 recap， $n$ 专归集合通信 rank 数，不被元素数占用）； $n_{\mathrm{row}}$ = 行长（每行元素数，norm 行即 $d_{\mathrm{model}}$）；行数 = $n_{\mathrm{el}}/n_{\mathrm{row}}$，不另占字母； $d_{\mathrm{model}}$ = hidden 宽（models $d$ 翻译，§4.3）； $\mathrm{bytes}$ = 每元素字节（fp32 取 4、bf16 取 2，bytes 不占字母， $B$ 专归 batch、 $T$ 专归序列长）； $m$ = running max、 $\ell$ = running 指数和（online softmax，沿 models 4.1 记号）； $x_i$ = 输入元素、 $y_i$ = 输出元素；Welford 状态 $\mu$ = 均值、 $M_2$ = 中心化平方和、 $c$ = 已见元素数（**本章局部计数符号，非 rank $n$**）； $\varepsilon$ = norm 稳定项、 $\gamma/\beta$ = norm 缩放/平移； $u_{\mathrm{fp32}} = 2^{-24}$ = fp32 单位舍入（5.1 recap）； $I$ = 算术强度、 $R$ = ridge。B/KB/MB 是单位符号豁免。

**边界声明**：**online softmax 的正确性（running max/sum 增量更新 + rescale 恒等）models 4.1 §5.2 式 (15)(16) 已用归纳法证明，本章 recap + 翻译、不重证**（§4.2 强制 recap 纪律）；本章增量 = standalone softmax kernel 的 two-pass 对 online 访存账 + warp/block 归约映射（5.1 recap）。FA 的 online softmax 指令级走读（exp2 trick、quad 归约、O 累加器 rescale）归 6.3，本章只引 FA2 softmax.h 的 exp2 片段作 compute 侧锚。norm+residual+quant 的融合带宽节省账归 5.4，本章只立融合形态与回链。代码 scope：机制片段级（§2 教学 softmax kernel）+ TransformerEngine norm kernel 走读片段（§4，路径级 + 关键行锚，非模块全量走读）。

## 2. softmax kernel：读几遍的访存账

先回答一个更基本的问题：softmax 为什么必须减 max？因为 $\exp$ 增长太快，fp32 的指数自变量超过约 88.7 就 overflow（ $e^{88} \approx 1.65 \times 10^{38}$ 尚在上限 $3.4 \times 10^{38}$ 内， $e^{89} \approx 4.49 \times 10^{38}$ 即超限）。数值稳定的 softmax 把每个 logit 减去行最大值再取指数：

$$
y_i = \frac{\exp(x_i - m)}{\sum_{j} \exp(x_j - m)}, \qquad m = \max_j x_j \qquad (1)
$$

**符号**： $m$ = 行 max， $\ell = \sum_j \exp(x_j - m)$ = 行指数和。减 max 不改结果（分子分母同乘 $e^{-m}$），却把所有指数自变量压到不超过 0， $\exp(\cdot) \le 1$，overflow 在构造上不可能——这就是 log-sum-exp 技巧的核心。式 (1) 暴露了 softmax 的归约结构：**要先知道整行的 max 与 sum，才能写任何一个输出**。max 与 sum 都是行级 reduction（5.1 原语），而「先归约再逐元素」的依赖关系决定了输入要被读几遍。

所谓 two-pass，是指 max 与 sum 各走一趟归约；所谓 online single-pass，是指把 max 与 sum 融进一趟、边读边更新两个 running 统计量。online 的增量更新公式 models 4.1 §5.2 式 (15) 已交付（本章 recap，不重证）：

$$
m^{(j)} = \max\left(m^{(j-1)}, \max_{s \in B_j} x_s\right), \qquad \ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_{s \in B_j} e^{x_s - m^{(j)}} \qquad (2)
$$

**recap 翻译注**：式 (2) 即 models 4.1 式 (15)，记号沿用（ $B_j$ = 第 $j$ 块 logit， $U_j$ = 前 $j$ 块并集）；该章已证不变量「 $m^{(j)}$ 是已见 logit 全局 max、 $\ell^{(j)} = \sum_{s \in U_j} e^{x_s - m^{(j)}}$」，归纳证明见 models 4.1 §5.2，本章不复述。重缩放因子 $e^{m_{\mathrm{old}} - m_{\mathrm{new}}} \le 1$ 只换基准、是恒等变换，不引入近似。式 (2) 的 kernel 侧意义：**max 与 sum 可以在同一趟读入里一起算完**——每读一块，更新 $m$ 并把旧的 $\ell$ 重缩放后加上新块的指数和，不必先读一遍求 max、再读一遍求 sum。

**命题 1**（softmax HBM 流量下界与变体账）。任何正确 softmax 的 HBM 流量至少 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$（读输入一遍 + 写输出一遍）；三种实现的流量为：two-pass（max、sum 各一趟归约 + normalize 一趟） $4 n_{\mathrm{el}} \cdot \mathrm{bytes}$（3 读 1 写），online single-pass（max+sum 融合一趟 + normalize 一趟） $3 n_{\mathrm{el}} \cdot \mathrm{bytes}$（2 读 1 写），行驻留 single-pass（整行片上、归约不落 HBM） $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$（1 读 1 写）达下界。**证明**（路线：下界反证 + 逐变体读写计数，三步）。下界：输入每元素至少读一次（否则改该元素输出不变，5.1 命题 1 同法），输出 $n_{\mathrm{el}}$ 个元素各写一次，故流量 $\ge 2 n_{\mathrm{el}} \cdot \mathrm{bytes}$。计数：two-pass 的 max 归约读一遍、sum 归约读一遍、normalize 读一遍写一遍，合 3 读 1 写 = $4 n_{\mathrm{el}} \cdot \mathrm{bytes}$；online 把 max+sum 融成一趟（式 (2)），归约阶段只读一遍，加 normalize 的 1 读 1 写 = $3 n_{\mathrm{el}} \cdot \mathrm{bytes}$——**归约阶段 2 读对 1 读**正是 two-pass 与 online 的差；行驻留把整行 load 进 register/shared 后，max/sum/normalize 全在片上归约（5.1 warp/block reduction），HBM 只剩 load 的 1 读与 store 的 1 写 = $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$，达下界。∎

全数字代入（本章口径 fp32， $n_{\mathrm{el}} \cdot \mathrm{bytes} = 67.11$ MB）：

| 变体 | 归约阶段读 | 总读写 | 流量 | 带宽口径时间 |
| --- | --- | --- | --- | --- |
| two-pass（max、sum 分趟） | 2 | 3 读 1 写 | 268.44 MB | 80.08 μs |
| online single-pass（融合） | 1 | 2 读 1 写 | 201.33 MB | 60.06 μs |
| 行驻留 single-pass | 0（片上） | 1 读 1 写 | 134.22 MB | 40.04 μs |

行驻留的成立条件是整行装得进片上： $n_{\mathrm{row}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}}$。代入 $n_{\mathrm{row}} = 4096$、fp32：每行 16384 B = 16 KB，远低于每 block 227 KB opt-in 上限（3.4 recap）；即便 bf16 的 $d_{\mathrm{model}} = 8192$ 也只 16 KB。**常见 hidden 宽与序列长的 softmax/norm 行都驻留得下**，所以生产 kernel 默认走行驻留 single-pass，达 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$ 下界。只有超长行（ $n_{\mathrm{row}} \cdot \mathrm{bytes}$ 超 shared 预算，如 fp32 行超过 58112 元素）才退到 online streaming 的 2 读 1 写——此时式 (2) 的融合归约把三趟压成两趟，省的就是归约阶段那一遍读。

```
softmax 三变体的 HBM 读写（每行 n_row 元素，↓读 ↑写）：
two-pass:        ↓x(max趟)  ↓x(sum趟)  ↓x(normalize) ↑y     = 3读1写
online stream:   ↓x(max+sum融合趟)     ↓x(normalize) ↑y     = 2读1写
行驻留:          ↓x(load进register/shared)  [片上max→sum→norm] ↑y = 1读1写
                 └─ 归约全在片上（5.1 warp shuffle + block partial）─┘
```

warp/block 归约映射（5.1 §3/§4 recap，不重推）：行驻留 kernel 每 block 负责一行，256 线程各持 $n_{\mathrm{row}}/256 = 16$ 个元素于 register。求 max：每线程先对自己 16 个元素串行取 max，再 warp shuffle 5 步归约（`__shfl_down_sync`，5.1 命题 3 免 shared），8 个 warp 的 lane0 各写 1 个 partial 进 shared（32 B），warp0 读回再 shuffle 3 步——block 级 shared 流量 64 B（5.1 命题 3）。求 sum 同构，只是把 max 换成 add。online 变体的 per-thread 状态是 $(m, \ell)$ 对，warp 内合并用 softmax 版的 Chan 公式： $m = \max(m_a, m_b)$、 $\ell = \ell_a e^{m_a - m} + \ell_b e^{m_b - m}$——与 §3 的方差 Chan 合并同构（都是「换基准 + 合并」），只是把平方和换成指数和。FA2 的 warp 内 4-thread quad 行归约（`quad_allreduce_`，softmax.h L39）是这条映射在 MMA fragment 布局下的实例，指令级走读归 6.3。

compute 侧一笔账钉死 memory-bound。exp 走 SFU（special function unit），吞吐 16 results/cycle/SM（1.2 §2，[NVIDIA, Best Practices](../references.md) Table 5，[C]）——只有 FP32 管线（128）的 1/8。全矩阵 $n_{\mathrm{el}} = 2^{24}$ 次 exp 的 SFU 地板：

$$
t_{\mathrm{exp}} = \frac{n_{\mathrm{el}}}{n_{\mathrm{SM}} \times 16 \times f} = \frac{16777216}{132 \times 16 \times 1.83 \times 10^{9}} = 4.34 \ \mathrm{\mu s} \qquad (3)
$$

代入得 4.34 μs，远低于行驻留带宽地板 40.04 μs（约 9.2 倍裕量）——**即便 exp 是 SFU 慢指令，softmax 仍稳稳落在 memory-bound 区**。强度口径：每元素约 3 FLOP（减 max、exp、除 sum），单 pass 流量 8 B/元素， $I \approx 0.375$ FLOP/byte，远低于 fp32 ridge 20（2.4 recap）。所以 softmax 优化的全部杠杆在「少读几遍」，不在「算快一点」。

exp2 trick（FA2 代码锚，compute 侧再省一笔）：FA2 不算 $\exp(x - m)$ 而算 $\exp_2(x \log_2 e - m \log_2 e)$，softmax.h L79 注释逐字「Instead of computing exp(x - max), we compute exp2(x * log_2(e) - ...」，L86 `exp2f(__fmul_rn(tensor, scale) - max_scaled)`（[Dao et al., 2022](../references.md)，repo b038b4974f26）。机制：SFU 的原生指令是 `ex2.approx`（MUFU.EX2）， $\exp_2$ 直接映射； $\exp$ 则要先乘 $\log_2 e$ 再 ex2，多一条乘法。把 $\log_2 e$ 折进 softmax scale（ $1/\sqrt{d_{\mathrm{head}}}$ 与 $\log_2 e$ 离线合并），运行期每元素省一次乘法——rescale 因子也同步改 $\exp_2$（softmax.h L157 `exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2)`）。这是「换底数对齐硬件原生指令」的实例，FA 全量走读归 6.3。

教学 kernel（机制片段级，fp32 行驻留，每 block 一行；warp/block 归约用 5.1 原型）：

```cpp
// 教学片段，非优化实现：行驻留 softmax（fp32，n_row=4096，256 线程/block）
__global__ void softmax_row(const float* x, float* y, int n_row) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * n_row;
    float* yr = y + (size_t)row * n_row;
    float v[16];                                  // 每线程 16 元素驻留 register
    float m = -INFINITY;
    for (int k = 0; k < 16; k++) {                // load 一遍（HBM 1 读）
        v[k] = xr[threadIdx.x + k * 256];
        m = fmaxf(m, v[k]);
    }
    m = block_reduce_max(m);                      // 5.1 warp shuffle + block partial
    float s = 0.f;
    for (int k = 0; k < 16; k++) { v[k] = expf(v[k] - m); s += v[k]; }
    s = block_reduce_sum(s);                      // 片上归约，不再读 HBM
    float inv = 1.f / s;
    for (int k = 0; k < 16; k++)                  // store 一遍（HBM 1 写）
        yr[threadIdx.x + k * 256] = v[k] * inv;
}
```

> **坑**：bf16 直接累加求 sum。触发条件：用 bf16 存 $\ell$ 并累加 $n_{\mathrm{row}} = 4096$ 个 $\exp$ 值；现象：bf16 尾数 8 bit，累加到 $2^{8}$ 量级后小项被整个吞掉（5.1 §4 精度账同构，ulp 随量级涨），sum 静默偏小、输出概率和不为 1；规避：累加器一律 fp32（TransformerEngine compute_t = float，§4），输入 bf16 在 load 时转 fp32 再算。另一条：减 max 前若 max 本身用 bf16 归约，max 的舍入会让 $\exp(x - m)$ 的自变量微微为正，极端下 overflow——max 归约也要 fp32。

## 3. Welford 在线方差：数值稳定的命题与证明

方差能不能拿 $E[x^{2}] - E[x]^{2}$ 直接算？数学上恒等，浮点上灾难。所谓 catastrophic cancellation（灾难性消去），是指两个相近的大数相减、有效数字几乎全部丢失的现象。先看一个 fp32 实例：数据 $x = [10001, 10002, 10003, 10004]$（均值 10002.5，真方差 1.25）。 $E[x^{2}] \approx 1.0005 \times 10^{8}$，fp32 在该量级的 ulp 是 8（ $2^{26} \le 1.0005 \times 10^{8} \lt 2^{27}$，ulp $= 2^{26-23} = 8$）； $E[x]^{2} = 100050006.25$ 同样落在 ulp 8 的网格上。两个大数各自舍入到 8 的倍数再相减，真差 1.25 远小于 ulp 8——Python fp32 实算朴素公式得 **$-8.0$**（负方差！），而 Welford 给 1.25（练习 2 复算）。方差非负是定义性质，算出负数说明朴素公式在浮点下根本不可用。

Welford 的出路是**不显式算 $E[x^{2}]$ 与 $E[x]^{2}$ 再相减**，而是增量维护中心化平方和 $M_2 = \sum_i (x_i - \mu)^{2}$。每来一个元素更新三个状态（计数 $c$、均值 $\mu$、 $M_2$）：

$$
c \leftarrow c + 1, \qquad \delta = x - \mu, \qquad \mu \leftarrow \mu + \frac{\delta}{c}, \qquad M_2 \leftarrow M_2 + \delta \cdot (x - \mu_{\mathrm{new}}) \qquad (4)
$$

**命题 2**（Welford 不变式）。处理完前 $c$ 个元素后， $\mu = \frac{1}{c}\sum_{i=1}^{c} x_i$ 且 $M_2 = \sum_{i=1}^{c}(x_i - \mu)^{2}$——均值与中心化平方和恰为已见数据的精确统计量（实数意义下）。**证明**（对 $c$ 归纳，三步）。基例 $c=1$： $\mu = x_1$、 $M_2 = 0$，成立。保持：设 $c-1$ 步后 $\mu_{\mathrm{old}} = \frac{1}{c-1}\sum_{i \lt c} x_i$、 $M_{2,\mathrm{old}} = \sum_{i \lt c}(x_i - \mu_{\mathrm{old}})^{2}$。第 $c$ 步均值更新 $\mu_{\mathrm{new}} = \mu_{\mathrm{old}} + (x_c - \mu_{\mathrm{old}})/c = \frac{1}{c}\sum_{i \le c} x_i$，均值不变式保持。对 $M_2$，关键恒等式： $\sum_{i \le c}(x_i - \mu_{\mathrm{new}})^{2} = \sum_{i \lt c}(x_i - \mu_{\mathrm{old}})^{2} + (x_c - \mu_{\mathrm{old}})(x_c - \mu_{\mathrm{new}})$——把前 $c-1$ 项的基准从 $\mu_{\mathrm{old}}$ 平移到 $\mu_{\mathrm{new}}$，交叉项恰好消去（展开 $(x_i - \mu_{\mathrm{new}})^{2} = (x_i - \mu_{\mathrm{old}})^{2} - 2(x_i - \mu_{\mathrm{old}})(\mu_{\mathrm{new}} - \mu_{\mathrm{old}}) + (\mu_{\mathrm{new}} - \mu_{\mathrm{old}})^{2}$，对 $i \lt c$ 求和，一次项 $\sum(x_i - \mu_{\mathrm{old}}) = 0$，剩 $(c-1)(\mu_{\mathrm{new}} - \mu_{\mathrm{old}})^{2}$，与第 $c$ 项合并即得 $\delta \cdot (x_c - \mu_{\mathrm{new}})$）。故式 (4) 的 $M_2$ 更新保持不变式。终止：任意 $c$ 步后成立，方差 $= M_2/c$。∎

数值稳定的机制一句话： $M_2$ 累加的是**中心化后的平方** $(x_i - \mu)^{2}$，量级是方差本身（小），不是 $E[x^{2}]$（大）；全程没有「两个大数相减」这一步，cancellation 无从发生。误差仍随 $c$ 累积（每次加法一份舍入），但是一阶小量累加，不是有效数字整体丢失——这与 5.1 §4 的 pairwise 求和精度账同构（浅依赖链给小舍入）。

GPU 上数据分属多线程，还要把各自的 $(c, \mu, M_2)$ 合并。这就是 **Chan 并行合并公式**（TransformerEngine utils.cuh L660 注释逐字「parallel Welford algorithm ... Chan's update formulae」，[NVIDIA, TransformerEngine](../references.md)）。两组统计量 $(c_a, \mu_a, M_{2,a})$、 $(c_b, \mu_b, M_{2,b})$ 合并：

$$
c = c_a + c_b, \qquad \mu = \frac{c_a \mu_a + c_b \mu_b}{c}, \qquad M_2 = M_{2,a} + M_{2,b} + (\mu_a - \mu_b)^{2} \frac{c_a c_b}{c} \qquad (5)
$$

**命题 3**（Chan 合并正确性）。式 (5) 合并后的 $(c, \mu, M_2)$ 恰为两组数据并集的精确统计量。**证明**（路线：均值加权 + 平方和分解，两步）。均值：并集总和 $= c_a \mu_a + c_b \mu_b$，除 $c$ 即式 (5) 第二式。 $M_2$：并集中心化平方和 $\sum_{i \in A \cup B}(x_i - \mu)^{2}$ 按组拆成 $\sum_{A}(x_i - \mu)^{2} + \sum_{B}(x_i - \mu)^{2}$；对 A 组把基准从 $\mu$ 换回 $\mu_a$， $\sum_{A}(x_i - \mu)^{2} = M_{2,a} + c_a(\mu_a - \mu)^{2}$（交叉项消去，命题 2 同法），B 组同理。代入 $\mu_a - \mu = \frac{c_b}{c}(\mu_a - \mu_b)$、 $\mu_b - \mu = \frac{c_a}{c}(\mu_b - \mu_a)$，两项合并得 $c_a \frac{c_b^{2}}{c^{2}}(\mu_a - \mu_b)^{2} + c_b \frac{c_a^{2}}{c^{2}}(\mu_a - \mu_b)^{2} = \frac{c_a c_b}{c}(\mu_a - \mu_b)^{2}$，即式 (5) 第三式。完整逐行展开留作练习 3。∎

8 元素核对（练习 3 同例）： $A = [1,2,3,4]$（ $c_a=4$、 $\mu_a=2.5$、 $M_{2,a}=5$）， $B = [5,6,7,8]$（ $c_b=4$、 $\mu_b=6.5$、 $M_{2,b}=5$）。合并 $\mu = 4.5$， $M_2 = 5 + 5 + (2.5-6.5)^{2} \times 4 \times 4/8 = 10 + 16 \times 2 = 42$——与 $[1..8]$ 直接算的 $\sum(x-4.5)^{2} = 42$ 逐位相同。

TransformerEngine 的归约层级正是 Chan 公式的实例（utils.cuh，[NVIDIA, TransformerEngine](../references.md)）：`warp_chan_upd_dynamic`（L672-701）用 warp shuffle 蝶形逐步合并——L687 `n_ab = n_a + n_b`、L690 `delta = m_a - m_b`、L691 `m2_ab = m2_a + m2_b + delta*delta*n_a*n_b*rn_ab`、L692 `m_ab = (n_a*m_a + n_b*m_b)*rn_ab`，逐行对应式 (5)。层级是：每 warp 内部先算局部统计量（`Stats<T,1,WARPS_M,1>::compute`，L844-864，先 allreduce 求 mean 再求中心化平方和），warp 间用 Chan 合并（L819），跨 CTA 再经 workspace + barrier 合并（L758）。**注意一个细节**：TE 的 per-warp 基例其实是 two-pass（先 mean 后 $M_2$，L848-861），Chan 合并用在 warp 间与 CTA 间——「Welford」在 TE 里特指并行合并层，不是每个线程都跑式 (4) 的逐元素增量。这给「在线算法」一个工程读法：基例可以用任何稳定方法，合并层用 Chan 公式保证结合律与数值稳定。

> **坑**：拿 $E[x^{2}] - E[x]^{2}$ 当「优化」（少一趟归约）。触发条件：为省一次 reduction 用朴素公式，数据带大偏移（如 LayerNorm 前的激活均值远离 0）；现象：fp32 算出负方差或 0，rsqrt 得 NaN/Inf，loss 突然跳变；规避：用 Welford/Chan 维护 $M_2$（式 (4)(5)），或至少 two-pass 先求 mean 再求中心化平方和（TE general kernel 即此，§4）。省一趟归约的收益远小于数值崩溃的代价。

## 4. LayerNorm / RMSNorm kernel：归约结构与带宽下界

LayerNorm 与 RMSNorm 各要归约什么，带宽下界为什么与 softmax 同形？两者与 softmax 共享「先归约再逐元素」的结构，差别在归约的统计量。LayerNorm 对每行（一个 token 的 $d_{\mathrm{model}}$ 维激活）算均值与方差再标准化：

$$
\mathrm{LN}(x)_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^{2} + \varepsilon}} + \beta_i, \qquad \mu = \frac{1}{d_{\mathrm{model}}}\sum_j x_j, \quad \sigma^{2} = \frac{1}{d_{\mathrm{model}}}\sum_j (x_j - \mu)^{2} \qquad (6)
$$

RMSNorm 砍掉均值中心化与平移，只除均方根：

$$
\mathrm{RMS}(x)_i = \gamma_i \frac{x_i}{\sqrt{\frac{1}{d_{\mathrm{model}}}\sum_j x_j^{2} + \varepsilon}} \qquad (7)
$$

**符号**： $\gamma/\beta$ = 逐通道缩放/平移（可学习参数）， $\varepsilon$ = 防除零稳定项。归约结构差异：LN 要 mean + var 两个统计量（Welford 一趟或 two-pass 两趟），RMSNorm 只要 mean-square 一个（一趟 $\sum x^{2}$）。RMSNorm 少一个归约、少一次减法、少一个 $\beta$——这是它比 LN 省 compute 的来源（模型侧取舍归 models，本章只立 kernel 结构）。

**命题 4**（norm 带宽下界与 memory-bound）。行驻留 norm kernel 的 HBM 流量下界为 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$（读输入 + 写输出），且 norm 是 memory-bound。**证明**（下界 + 强度两步）。下界：输入每元素至少读一次、输出每元素写一次（命题 1 同法）， $\ge 2 n_{\mathrm{el}} \cdot \mathrm{bytes}$；行驻留（ $d_{\mathrm{model}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}}$）下归约全在片上，达下界。强度：LN 每元素约 6 FLOP（减 $\mu$、乘 rs、乘 $\gamma$、加 $\beta$ 各 1，加归约摊约 2），bf16 读+写 4 B/元素， $I \approx 1.5$ FLOP/byte；RMSNorm 约 3 FLOP/元素， $I \approx 0.75$——都远低于 bf16 ridge 295（2.4 recap），memory-bound。∎

全数字代入（norm 口径 bf16， $n_{\mathrm{el}} = 2^{24}$、 $\mathrm{bytes} = 2$）：读 33.55 MB + 写 33.55 MB = 67.11 MB，带宽口径 $67.11 \times 10^{6}/(3352 \times 10^{9}) = 20.02$ μs。每行 $d_{\mathrm{model}} = 4096$ bf16 = 8192 B = 8 KB，行驻留轻松（register 每线程 16 个 fp32 = 64 B，或 shared 8 KB）。**norm 的优化杠杆与 softmax 同：少读几遍 + 融合下游算子**（§5），不在算力。

TransformerEngine norm kernel 走读片段（机制片段级，路径 `transformer_engine/common/normalization/`，main pushed 2026-11-25，[NVIDIA, TransformerEngine](../references.md)；**路径漂移注**：旧 `common/layer_norm/`、`rms_norm/` 已 404，现行 `normalization/{layernorm,rmsnorm}/`）。LayerNorm tuned kernel（`layernorm/ln_fwd_kernels.cuh` `ln_fwd_tuned_kernel` L26）的执行体五段：

```
ln_fwd_tuned_kernel（每 CTA 多行，grid-stride over rows，L85）：
 ① 向量化 load + 转 fp32：Ivec x[LDGS] → compute_t xf[]（L86-98，bf16→fp32）
 ② Welford 统计：stats.compute(xf, rn) → (mu, m2)（L100，Chan 合并 §3）
 ③ rs = rsqrtf(rn*m2 + eps)（L109）；mu/rs 写回供反向（L105-113）
 ④ 逐元素 normalize：y = rs*(xf - mu)（L121）→ gamma*y + beta（L127）
    → zero_centered_gamma +1（L123）→ fp8 scale（L140）→ amax 追踪（L129）
 ⑤ 向量化 store：Ovec z → params.z（L144-146）
 gamma/beta 在行循环外 load 一次、register 驻留复用（L66-74）
```

三个读法要点。其一，**输入 bf16、累加 fp32**：load 时 `compute_t(x[it].data.elt[jt])` 把 bf16 转 fp32（L94），整个归约与 normalize 在 fp32 跑——§3 坑注的工程兑现。其二，**Welford 路径**：tuned kernel 走 `stats.compute` 得 $(\mu, M_2)$（Chan 合并），rs 由 $M_2/n$ 开方（L109）；对照 general kernel（`ln_fwd_general_kernel` L169）是显式 two-pass——先 allreduce 求 mean（L246-255）再求中心化平方和（L258-270），两条路径数值都稳定，tuned 用 Chan 合并省归约趟数。其三，**$\gamma/\beta$ register 驻留**：行循环外 load 一次（L66-74），多行复用——norm 权重只有 $d_{\mathrm{model}}$ 个元素，驻留 register 后每行不再重读，这是「小权重 + 大激活」结构的复用账（4.3 复用因子同法）。

RMSNorm 侧一个有趣的细节（`rmsnorm/rmsnorm_fwd_kernels.cuh`）：tuned kernel（L25）同样走 Welford 得 $(\mu, M_2)$，但 RMS 不用均值，于是 rs 写成 `rsqrtf(rn*m2 + mu*mu + eps)`（L102）——用 $E[x^{2}] = M_2/n + \mu^{2}$ 从 Welford 状态**重建均方**（注释 L100-101 逐字「we could optimize here to count mean square directly」）。核对： $[1..8]$ 的 $\mu=4.5$、 $M_2=42$、 $n=8$， $42/8 + 4.5^{2} = 5.25 + 20.25 = 25.5 = E[x^{2}]$（练习 2）。general kernel（L161）则直接 $\sum x^{2}$（L236-248）、`rsqrtf(sqsigma*rn + eps)`（L249），一趟归约——RMSNorm 本不需要 mean，general 路径更直接。tuned 路径复用 LN 的 Welford 基础设施（一套 Stats 代码服务两种 norm），多算一个 mean 换代码复用，是工程取舍不是数学必需。

norm kernel 还顺手做了 FP8 量化的准备：amax 追踪（L129-139，逐元素取 $|y|$ 的 max）+ block 归约（L152-157 `reduce_max` + `atomicMaxFloat`）+ scale 乘（L140-142）。这把「norm 输出 → 量化成 FP8 喂 GEMM」的 cast 融进 norm kernel——正是 §5 与 5.4 的 norm+quant 融合，TE 另有 `rmsnorm/rmsnorm_fwd_mxfp8.cu` 专做 RMSNorm + MXFP8 cast（5.4 回链）。

## 5. norm + residual + quant 融合与带宽账汇总

norm 算完，输出还要加残差、再量化喂下一层 GEMM——这些能不能不落地 HBM？能，而且 TE kernel 已经在做（§4 的 amax/scale 即 quant 融合的一半）。所谓 norm+residual+quant 融合，是指把「残差加 + norm + 量化 cast」三个算子并进一个 kernel：读残差与输入、片上算 norm、量化、写 FP8 输出与更新后的残差。融合省的是中间 tensor 的 HBM 往返——完整带宽节省账（ $k$ 个逐元素算子省 $2(k-1) n_{\mathrm{el}} \cdot \mathrm{bytes}$）归 5.4 命题 1，本章只立形态与回链。

融合为什么在 norm 上特别自然？因为 norm 是行驻留的（§4）：整行已在 register/shared，下游的残差加（逐元素）、 $\gamma/\beta$（逐元素）、量化 cast（逐元素）都能就地接在 normalize 之后，中间值不落地。这正是 5.4 三类融合策略里「reduction + 逐元素」一类的实例（norm 归约 + 逐元素链）。TE 的 `rmsnorm_fwd_mxfp8.cu`（RMSNorm + MXFP8 cast）与 `ln_fwd_kernels.cuh` 的 fp8_out 分支是代码实体；inference 3.4 §2 的 RMSNorm+quant 融合对（算法级 IO 账）在此落到 kernel 层。

带宽账汇总（本章口径，行驻留达下界）：

| 算子 | 精度 | 归约 | HBM 下界 | 带宽口径时间 | 强度 / bound |
| --- | --- | --- | --- | --- | --- |
| softmax（行驻留） | fp32 | max + sum | $2 n_{\mathrm{el}} \cdot 4$ = 134.22 MB | 40.04 μs | $I \approx 0.375$ / memory |
| softmax（online stream） | fp32 | max+sum 融合 | $3 n_{\mathrm{el}} \cdot 4$ = 201.33 MB | 60.06 μs | memory |
| LayerNorm | bf16 | mean + var | $2 n_{\mathrm{el}} \cdot 2$ = 67.11 MB | 20.02 μs | $I \approx 1.5$ / memory |
| RMSNorm | bf16 | mean-square | $2 n_{\mathrm{el}} \cdot 2$ = 67.11 MB | 20.02 μs | $I \approx 0.75$ / memory |

三个算子同一结论：**memory-bound，下界由「读一遍写一遍」钉死，优化杠杆是少读几遍（行驻留 / online 融合）与融合下游算子（5.4）**。compute 侧即便 exp 走慢速 SFU（4.34 μs 地板）也远未触顶——这与 GEMM（5.2，compute-bound、靠 L2 复用进平台区）形成对照：归约类算子的 roofline 点天生在斜线区左端，tiling 提强度对它们意义不大（每元素只被消费常数次，无 GEMM 的 $O(T)$ 复用），真正的手段是 fusion 与减少 pass。

## 6. 边界：本章不做什么

下界与数值稳定交付完，剩下的归谁？本章交付 softmax 访存账（命题 1 + §2）、Welford 不变式与 Chan 合并（命题 2/3 + §3）、norm 带宽下界与 TE 走读片段（命题 4 + §4）、融合形态与带宽汇总（§5）；以下各有专章。online softmax 正确性证明归 models 4.1 §5.2（本章 recap 不重证）；FA 的 online softmax 指令级走读（exp2 trick 全量、quad 归约、O rescale、tiling）归 6.3；norm+residual+quant 的融合带宽节省账与预算约束归 5.4；GEMM epilogue fusion 代码实体归 5.2 §7；多 GPU 归约消息量归 8.1/8.2；实测 drill-down 归 11.2。代码 scope：机制片段级（§2 教学 softmax）+ TE norm 走读片段（§4，非模块全量）。

本章交付清单（下游 recap 接口）：命题 1 + 三变体流量表（§2）、exp2 trick 锚（§2）、命题 2/3 + Welford/Chan（§3）、命题 4 + TE norm 五段走读（§4）、融合形态（§5）。5.4 recap §4/§5 作「reduction + 逐元素」融合策略的 norm 实例与带宽下界；6.3 recap §2 的 online 访存账与 exp2 锚作 FA softmax 走读的入口（正确性仍指向 models 4.1）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| softmax HBM 下界 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$；two-pass 4×（3 读 1 写）/ online 3×（2 读 1 写）/ 行驻留 2×（1 读 1 写）；归约阶段 2 读对 1 读 | 命题 1 + §2 表 | §2、练习 1 |
| 本章口径 fp32：268.44 / 201.33 / 134.22 MB，80.08 / 60.06 / 40.04 μs；行驻留条件 $n_{\mathrm{row}} \cdot \mathrm{bytes} \le S_{\mathrm{sh}}$ | §2 代入 | §2、练习 1 |
| online softmax 正确性 recap models 4.1 §5.2 式 (15)(16)，不重证；增量 = 访存账 + 归约映射 | 式 (2) recap | §2、练习 4 |
| exp SFU 地板 4.34 μs（16 results/cycle/SM）远低于带宽地板 40.04 μs； $I \approx 0.375$ 低于 fp32 ridge 20 → memory-bound | 式 (3) | §2、练习 2 |
| exp2 trick： $\exp_2$ 对齐 SFU 原生 ex2， $\log_2 e$ 折进 scale（FA2 softmax.h L79/L86） | §2 代码锚 | §2、6.3 |
| 朴素方差 $E[x^{2}] - E[x]^{2}$ 灾难消去：fp32 对 $10^{4}$ 偏移数据算出 $-8$（真值 1.25） | §3 实例 | §3、练习 2 |
| Welford 不变式 $M_2 = \sum(x-\mu)^{2}$（命题 2）；Chan 合并 $M_2 = M_{2,a} + M_{2,b} + (\mu_a-\mu_b)^{2} c_a c_b/c$（命题 3） | 式 (4)(5) | §3、练习 3 |
| TE Chan 合并 utils.cuh warp_chan_upd_dynamic L672-701；per-warp 基例 two-pass、warp/CTA 间 Chan | §3 代码锚 | §3、练习 4 |
| norm 带宽下界 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$、memory-bound；bf16 67.11 MB / 20.02 μs；LN $I \approx 1.5$、RMS $I \approx 0.75$ 低于 ridge 295 | 命题 4 + §4 | §4、练习 2 |
| LN 需 mean+var、RMS 只需 mean-square；TE tuned 走 Welford、general 走 two-pass；RMS tuned 用 $M_2/n + \mu^{2}$ 重建均方 | 式 (6)(7) + §4 | §4、练习 4 |
| norm+residual+quant 融合 = reduction+逐元素（5.4）；TE fp8_out 分支 + rmsnorm_fwd_mxfp8.cu | §5 | §5、5.4 |

## 练习

1.（计算 · [10]）softmax 访存代入（本章口径 $4096 \times 4096$ fp32、 $\mathrm{BW} = 3352$ GB/s）：(a) two-pass / online single-pass / 行驻留三变体的 HBM 流量与带宽口径时间；(b) 行驻留的 shared 预算条件， $n_{\mathrm{row}} = 4096$ fp32 每行字节是否满足；(c) 归约阶段 two-pass 对 online 各读几遍。校验：(a) 268.44 / 201.33 / 134.22 MB、80.08 / 60.06 / 40.04 μs（§2 表）；(b) 16 KB 低于 227 KB（§2）；(c) 2 对 1（命题 1）。
2.（计算 · [20]）数值稳定与带宽账：(a) 用 fp32 复算 $x = [10001, 10002, 10003, 10004]$ 的朴素方差 $E[x^{2}] - E[x]^{2}$ 与 Welford 方差，先预测朴素公式的符号再算；(b) 验证 RMSNorm tuned 的 $M_2/n + \mu^{2} = E[x^{2}]$（用 $[1..8]$）；(c) norm bf16 $4096 \times 4096$ 的读+写字节与带宽口径时间；(d) exp SFU 地板（式 (3)）对行驻留带宽地板的比值，判 bound。校验：(a) 朴素 $-8.0$、Welford 1.25（§3）；(b) 25.5（§4）；(c) 67.11 MB、20.02 μs（命题 4）；(d) 4.34 / 40.04 ≈ 0.108，memory-bound（§2）。
3.（推导 · [30]）补全命题 3 的证明（Chan 合并）：(a) 写出并集均值的加权推导；(b) 把 $\sum_{A \cup B}(x-\mu)^{2}$ 按组拆，对 A 组做基准平移 $\sum_A(x-\mu)^{2} = M_{2,a} + c_a(\mu_a-\mu)^{2}$，逐句说明交叉项为何消去；(c) 代入 $\mu_a - \mu = \frac{c_b}{c}(\mu_a-\mu_b)$ 推出合并项 $\frac{c_a c_b}{c}(\mu_a-\mu_b)^{2}$，用 §3 的 8 元素例（ $M_2 = 42$）逐位核对；(d) 边界： $c_b = 0$ 时式 (5) 退化为恒等，说明合并公式对空组成立。校验：(c) 与 §3 数值一致（ $\mu = 4.5$、 $M_2 = 42$）；(d) $c_b=0$ 给 $\mu = \mu_a$、 $M_2 = M_{2,a}$（代入锚）。
4.（代码 · [30]）softmax 教学实现与 TE norm 走读：(a) 用 §2 教学 kernel 结构写 fp32 行驻留 softmax（PyTorch 教学版，非优化实现），对 `torch.softmax(x, dim=-1)` 断言 allclose（容差 $10^{-5}$），先预测输出是否随 block 尺寸改变及理由；(b) 在 TE `ln_fwd_kernels.cuh` 找出 tuned（Welford，L100）与 general（two-pass，L246-270）两条路径，说明 rs 公式差异；(c) 指出 `rmsnorm_fwd_kernels.cuh` L102 的 `rn*m2 + mu*mu` 为何等于均方（命题 3 推论）；(d) 找出 fp8_out 分支的 amax/scale 行，说明它对应 §5 哪种融合。校验：(a) allclose 容差 $10^{-5}$、输出不随 block 尺寸变（精确性）；(b) L109 对 L271（§4）；(c) $E[x^{2}] = M_2/n + \mu^{2}$（§4）；(d) norm+quant 融合（§5、5.4）。

## 参考

- [NVIDIA, TransformerEngine](../references.md)：NVIDIA/TransformerEngine repo（key: transformerengine），教学参照 + 官方实现，main pushed 2026-11-25 contents API 一手核实——normalization/layernorm/ln_fwd_kernels.cuh（ln_fwd_tuned_kernel L26：向量化 load+fp32 L86-98、stats.compute Welford L100、rs=rsqrtf(rn*m2+eps) L109、normalize L121/gamma+beta L127/fp8 scale L140/amax L129/store L144、gamma/beta register 驻留 L66-74；ln_fwd_general_kernel L169：two-pass mean L246-255/var L258-270/rs L271）；normalization/rmsnorm/rmsnorm_fwd_kernels.cuh（tuned L25：rs=rsqrtf(rn*m2+mu*mu+eps) L102「we could optimize here to count mean square directly」L100-101；general L161：直接 Σx² L236-248/rs L249）+ rmsnorm_fwd_mxfp8.cu（norm+MXFP8 cast，5.4 回链）；utils.cuh（parallel Welford / Chan's update formulae 注释 L660、warp_chan_upd_dynamic L672-701：n_ab L687/delta L690/m2_ab L691/m_ab L692、Stats compute L727-761/per-warp two-pass 基例 L844-864）；**路径漂移注**：旧 common/layer_norm·rms_norm 404 → normalization/{layernorm,rmsnorm}
- [Dao et al., 2022](../references.md)：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（key: flashattention），arXiv:2205.14135，[A]，repo b038b4974f26（2026-11-25）——csrc/flash_attn/src/softmax.h：scale_apply_exp2 L67（exp2 trick 注释 L79「Instead of computing exp(x - max), we compute exp2(x * log_2(e) - ...」+ exp2f L86）、softmax_rescale_o L137（scores_scale=exp2f((max_prev-max_cur)*scale_log2) L157/row_sum*=L158/acc_o*=L160）、quad_allreduce_ L39（warp 4-thread 行归约）；本章只引 exp2/quad 片段作 compute 侧锚，FA 全量指令级走读归 6.3；online softmax 正确性证明归 models 4.1 §5.2（recap 不重证）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide 13.5（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Table 5 CC 9.0 SFU 16 / FP32 128 results/cycle/SM（§2 exp 算力地板）；Register Pressure / spill 与 occupancy（§4 融合预算，5.4 回链）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide 13.5（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——Warp Shuffle Functions（__shfl_down_sync 归约，5.1 recap 源）；shared memory 容量与 opt-in（行驻留条件）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——ex2.approx（SFU 原生 exp2，§2 exp2 trick 硬件 backing）；shfl.sync（warp 归约）；redux.sync（5.1 recap）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/2_Concepts_and_Techniques/reduction（warp/block 归约原型，5.1 recap 源，本章 softmax/norm 归约映射用）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100/H200/B200 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM 3.35 TB/s / BF16 989.5 TFLOPS / FP32 67 TFLOPS（§1 带宽与 ridge 口径）
