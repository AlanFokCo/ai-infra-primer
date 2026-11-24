# 1.4 指令流水线与延迟隐藏（Instruction Pipeline and Latency Hiding）

> HBM 一次访存要等约 479 个周期，一个 SM 需要多少在飞请求、多少 active warp 才能把这段等待藏住？本章交付指令延迟/吞吐表（官方 + 实测两口径）、warp 调度与 scoreboard 依赖追踪、Little's law 延迟隐藏推导（并发 = 延迟 × 吞吐 → 藏住 HBM 延迟需约 52 个在飞 warp load）、ILP 与 TLP 的乘积替代关系、stall 四分类与 Nsight Compute 读法 | 前置：1.2（命题 1 单发射、式 (3)(4) occupancy 上界与 $W_{\mathrm{active}}$、§3 寄存器权衡）、1.3（wgmma 异步、实测延迟 128 cycles）、1.1（逐精度 clock）；下游 3.4（latency hiding → 所需 occupancy recap）、11.2（stall reason 实测 drill-down）、2.4（latency-bound 区）、4.3（tiling 降低暴露延迟） | 锚点：CUDA Best Practices（[NVIDIA, Best Practices](../references.md)，[C] 官方规格披露·一手权威）、Nsight Compute ProfilingGuide（[NVIDIA, Nsight Compute](../references.md)，[C]）、Hopper 微基准（[Luo et al., 2024](../references.md)，arXiv:2402.13499，[B] 实测口径）、Hopper whitepaper（[NVIDIA, Hopper whitepaper](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 479 个周期的空等：延迟隐藏问题

一次 HBM 访存实测要约 479 个周期才回数据（[Luo et al., 2024](../references.md)，arXiv:2402.13499，H800 微基准，[B]）。这 479 个周期里，发射这条 load 的 warp 干等——但 SM 不该干等：它还有别的 warp 可以发射。本章回答三个问题：延迟与吞吐为什么是两个口径，流水线怎么让两者解耦？藏住 HBM 延迟需要多少在飞并发，怎么从 Little's law 推出来？这个并发量怎么落到 occupancy（1.2）与 ILP 上，Nsight 又怎么读出哪里没藏住？

**本章口径**（章脊锁定）：主口径 **H100 SXM**（ $n_{\mathrm{SM}} = 132$、HBM 3352 GB/s，沿 1.1）；周期 ↔ 时间换算取 **TC-boost 1830 MHz**（BF16 kernel 口径，沿 1.1；1.98 GHz 灵敏度见练习 2）。**实测口径声明**：精确延迟 cycles（shared 29 / L2 263 / HBM 479）来自 H800 微基准 [B]，非 H100 官方规格——官方 Best Practices 只给定性「hundreds of clock cycles」；本章按 H100 clock 换算做量级账，cycles 数值标实测口径，不冒充官方披露。

符号声明（spec §4.3）： $L$ = 在飞并发量（请求数/sector 数）； $\lambda$ = 服务率（请求/cycle 或 /s）； $t_{\mathrm{lat}}$ = 延迟（cycle 或 s）； $\mathrm{ILP}$ = 每 warp 在飞独立指令数； $s_{\mathrm{warp}}$ = 一条 warp load 触及的 sector 数（coalesced fp32 = 4）； $W_{\mathrm{active}}$ = active warps（recap 1.2 式 (4)）； $\mathcal{O}$ = occupancy； $f$ = clock。带宽与 sector： $`\mathrm{BW}_{\mathrm{SM}}`$ = 每 SM HBM 带宽份额；sector = 32 B（L2 访存粒度，[NVIDIA, Nsight Compute](../references.md) memory chart 口径，[C]；层级 line/sector 全账归 2.2）。

**边界声明**：本章交付延迟隐藏的**机制与并发量级推导**。occupancy 的编程模型侧精确计算归 3.4（recap 本章所需并发为输入）；stall reason 的实测 profiling 全流程与 roofline drill-down 归 11.2；latency-bound 区的 roofline 判定归 2.4；tiling 怎么把 HBM 访存换成 shared 访存从而降低暴露延迟归 4.3。本章不重复这些，只把「需要多少并发」这一硬件来源交付到位。

## 2. 指令延迟与吞吐：两个口径

先回答一个更基本的问题：FMA 延迟 4 个周期，为什么吞吐还能是每周期 128 个 result？因为延迟和吞吐是流水线的两个不同口径——延迟是一条指令从发射到写回的时间，吞吐是流水线每周期能接受多少条新指令。两者靠流水线解耦。

执行管线时序（单 scheduler，单发射，1.2 命题 1）：

```
周期:            1    2    3    4    5    6    7
scheduler 0:    FMA0 FMA1 FMA2 FMA3 ...        ← 每周期发射 1 条（吞吐）
FMA0 流水线:    [E1][E2][E3][E4]→写回@4        ← 4 周期延迟
FMA1 流水线:         [E1][E2][E3][E4]→写回@5
FMA2 流水线:              [E1][E2][E3][E4]→@6
依赖链:  FMA1 读 FMA0 结果 → 须等 @4 → 每 4 周期才 1 条（吞吐塌到 1/4）
独立链:  FMA0..FMA3 互不依赖 → 每周期 1 条（吞吐满）
```

延迟/吞吐表（H100 SXM；官方口径 [C] 与实测口径 [B] 分列）：

| 指令 / 访存 | 延迟 | 吞吐 | 口径 / 出处 |
| --- | --- | --- | --- |
| FP32 FMA | ~4 cycles | 128 results/cycle/SM（= 4×32，单发射） | 官方 prose（CC 7.0 措辞）[C] + 1.2 命题 1 |
| shared load | 29.0 cycles | 128 B/cycle/SM（32 banks×4 B，无冲突） | 实测 [B] Table IV + PG [C] |
| L1 命中 | 40.7 cycles | — | 实测 [B] Table IV |
| L2 命中 | 263.0 cycles | — | 实测 [B] Table IV |
| HBM（global，L2 miss） | 478.8 cycles | 每 SM 份额 25.4 GB/s | 实测 [B] + wp [C] |
| wgmma m64n128k16 | 128 cycles（dense） | 稳态 64 cycles/指令（反推） | 实测 [B] + 1.3 式 (2) |

**口径注**（三层）：其一，FMA「~4 cycles」是 Best Practices 逐字「The latency of most arithmetic instructions is typically 4 cycles on devices of compute capability 7.0」（[NVIDIA, Best Practices](../references.md)，[C]）——CC 7.0 措辞沿用，Hopper 算术延迟量级同；吞吐 128 是 CC 9.0 Table 5 + 1.2 命题 1 单发射互证。其二，shared/L1/L2/HBM 的精确 cycles 是 H800 微基准实测 [B]，官方只给定性「When accessing uncached local or global memory, there are hundreds of clock cycles of memory latency」；479 落在「hundreds」量级内，两口径相容。其三，wgmma 延迟 128 与稳态间隔 64 的差是流水线填充（1.3 §3 已核），异步语义把两者解耦。

这张表的关键读法：**延迟逐层涨（shared 29 → L2 263 → HBM 479），吞吐逐层跌**。藏住哪一层的延迟，就要按哪一层的延迟 × 服务率备并发——这是 §4 Little's law 的输入。

## 3. warp 调度与 scoreboard：依赖追踪

一个 scheduler 每周期从哪些 warp 里挑、挑的条件是什么？1.2 命题 1 锁了发射宽度（每 scheduler 单发射 1 条），本节补发射的**前提**：warp 必须 eligible（源操作数就绪、执行管线可用、同步点已过）。

所谓 scoreboard（记分板），是指硬件对每个 warp 的寄存器依赖追踪机制——一条指令的目的寄存器在被写回前标记为「pending」，后续指令若读这个寄存器就须等 pending 清除。这是机制语义层；具体电路实现属闭源，本书不推定，只引其可观测投影（Nsight stall reason，§7）。长延迟操作（load）的结果异步返回，挂的是 long scoreboard；固定延迟操作（FMA 4 周期）的依赖挂 wait 状态。

warp 调度与依赖追踪（单 processing block）：

```
warp scheduler（每 processing block 1 个，单发射，1.2 命题 1）
  resident warps（64/4 = 16 slots/scheduler，均分口径反推）
        │ 逐 warp 判 eligible：
        ├─ 源寄存器就绪？（scoreboard：pending 写回已清除）
        ├─ 执行管线可用？（FP32/TC/LSU/SFU 不忙）
        └─ 同步点已过？（barrier/fence/wait_group）
        ▼
  eligible warps ──挑 1 个──► dispatch ──► 执行管线
        │
        └─ 不 eligible 的 warp → stall，按原因分类（Nsight，§7）：
             long_scoreboard（等 global/local 结果）
             wait（等固定延迟执行依赖）
             barrier（等 __syncthreads）
             not_selected（eligible 但没被挑中）
```

stall 四分类（spec 口径，映射到 Nsight ProfilingGuide §2.4.7 Warp Stall Reasons，[NVIDIA, Nsight Compute](../references.md)，[C]）：

| 类 | 机制 | Nsight reason | 含义 |
| --- | --- | --- | --- |
| memory dependency | 等访存结果返回 | `long_scoreboard` | L1TEX（local/global/surface/texture）依赖，memory-latency-bound |
| execution dependency | 等前序指令写回 | `wait` | 固定延迟执行依赖（如 FMA 4 周期链） |
| sync | 等同步点 | `barrier` / `membar` | `__syncthreads` / fence |
| not selected | eligible 但调度器挑了别的 warp | `not_selected` | 健康信号：可调度 warp 充足 |

其余 reason 归「其他」桶：`mio_throttle`（MIO/shared 队列满）、`lg_throttle`（LSU 队列满）、`math_pipe_throttle`（数学管线满）、`no_instructions`（取指空）、`drain`（kernel 退出排空）、`dispatch_stall`、`branch_resolving`、`imc_miss`（立即数常量缓存 miss）。这些是队列/管线背压或边界状态，不是延迟隐藏的主矛盾，诊断时次要（§7）。

异步指令的依赖追踪走另一条路。mma.sync 的结果写回寄存器，后续指令读它须等寄存器 scoreboard 清除（表现为 `wait` stall）；wgmma 异步发射后结果不立即写回，warp-group 用 `wgmma.wait_group N` 显式等至多 N 个 group 完成（1.3 §5）——同步点从「逐寄存器 pending」变成「逐 group 计数」。所以 wgmma 把「等 MMA 结果」从每条指令的寄存器依赖里挪出来、集中到一个可控制的同步点；`wait_group` 之前 warp-group 仍可发射独立指令（地址计算/TMA/softmax），这就是 §6 第三条轴的机制来源。scoreboard 的具体电路实现属闭源，本书只引其可观测语义（stall reason）与 PTX 披露的同步原语，不推定内部连线。

**not_selected 为什么是健康信号**：它说明这个 warp 已经 eligible（依赖都清了），只是调度器这周期挑了另一个 warp——意味着可调度 warp 供过于求，发射槽不缺候选。反过来，若 stall 全是 `long_scoreboard` 而 `not_selected` 接近零，说明 warp 都卡在等访存、没有富余候选，延迟没藏住——这正是 §4 要算的并发不足。

## 4. Little's law：藏住延迟需要多少在飞并发

藏住 479 周期的 HBM 延迟，需要多少在飞请求？推导路线分三步：先写 Little's law（在飞并发 = 服务率 × 延迟）并把它重排成「吞吐 = 并发/延迟」，再代入 HBM 的每 SM 服务率与实测延迟算所需并发，最后把并发折成 warp 数。关键一步是**服务率取瓶颈资源（HBM）的、且 per-SM 归一化**——取错服务率（如 warp 发射率）会把所需并发高估约一个量级（§4 末坑注）。

Little's law（稳态排队守恒）：

$$
L = \lambda \cdot t_{\mathrm{lat}} \qquad (1)
$$

**符号**： $L$ = 系统内平均在飞请求数； $\lambda$ = 请求完成速率（服务率）； $t_{\mathrm{lat}}$ = 每请求在系统内停留时间（延迟）。量纲： $[\lambda] = \mathrm{请求}/\mathrm{cycle}$， $[t_{\mathrm{lat}}] = \mathrm{cycle}$，积为请求数。

先用 FMA 热身（小数字）：单 scheduler 每周期发射 1 条 warp FMA（ $\lambda = 1$ warp 指令/cycle，1.2 单发射），FMA 延迟 4 周期（ $t_{\mathrm{lat}} = 4$），故 $L = 1 \times 4 = 4$ 条 warp FMA 在飞。要喂满这个 scheduler，须有 4 条独立 FMA 链在飞——可以是 1 个 warp 的 ILP 4（4 条独立链），或 4 个 warp 各跑依赖链（TLP 4）。全 SM 4 个 scheduler 共 16 条在飞。这就是延迟隐藏的微观图景：**延迟越长、吞吐越高，需要的在飞并发越多**。

**命题 1**（延迟隐藏条件）。设瓶颈资源满服务率为 $\lambda_{\max}$、延迟为 $t_{\mathrm{lat}}$，则维持满吞吐所需在飞并发为 $L_{\mathrm{req}} = \lambda_{\max} \cdot t_{\mathrm{lat}}$；若实际提供并发 $C \lt L_{\mathrm{req}}$，则资源利用率 $= C/L_{\mathrm{req}} \lt 1$。**证明**（Little's law 重排 + 利用率）：由式 (1)，稳态吞吐 $\lambda = L/t_{\mathrm{lat}}$——每个在飞请求占用服务器 $t_{\mathrm{lat}}$ 时间，故 $C$ 个在飞请求的完成率是 $C/t_{\mathrm{lat}}$。要达到满服务率 $\lambda_{\max}$，须 $C/t_{\mathrm{lat}} = \lambda_{\max}$，即 $C = \lambda_{\max} t_{\mathrm{lat}} = L_{\mathrm{req}}$。若 $C \lt L_{\mathrm{req}}$，则实际吞吐 $C/t_{\mathrm{lat}} \lt \lambda_{\max}$，利用率 $= (C/t_{\mathrm{lat}})/\lambda_{\max} = C/L_{\mathrm{req}} \lt 1$，资源空闲。故隐藏延迟（让瓶颈资源不空闲）当且仅当 $C \ge L_{\mathrm{req}}$。∎

代入 HBM（H100 SXM 主口径）。每 SM 带宽份额与 sector 服务率：

$$
\lambda_{\mathrm{SM}} = \frac{\mathrm{BW}}{n_{\mathrm{SM}} \cdot 32 \cdot f} \qquad (2)
$$

**符号**： $\lambda_{\mathrm{SM}}$ = 每 SM 的 HBM sector 服务率（sectors/cycle）； $\mathrm{BW} = 3352$ GB/s； $n_{\mathrm{SM}} = 132$；32 = sector 字节； $f = 1.83$ GHz。代入：每 SM 份额 $3352/132 = 25.4$ GB/s， $\lambda_{\mathrm{SM}} = 25.4 \times 10^{9}/(32 \times 1.83 \times 10^{9}) = 0.434$ sectors/cycle。所需在飞并发（式 (1)， $t_{\mathrm{HBM}} = 478.8$ cycles 实测 [B]）：

$$
L_{\mathrm{SM}} = \lambda_{\mathrm{SM}} \cdot t_{\mathrm{HBM}} = 0.434 \times 478.8 \approx 208 \ \text{sectors} \qquad (3)
$$

全数字落点：每 SM 须有约 **208 个 sector（≈6.6 KB）在飞**才能喂满它的 HBM 份额。换算时间口径核对： $t_{\mathrm{HBM}} = 478.8/1.83 = 261.6$ ns， $25.4 \times 10^{9} \times 261.6 \times 10^{-9} = 6644$ B ≈ 6.6 KB，与 sector 口径一致。全芯片在飞 $3352 \times 10^{9} \times 261.6 \times 10^{-9} \approx 877$ KB（十进制）——这是 HBM 带宽延迟积（bandwidth-delay product）的芯片级量级。

把 sector 折成 warp load：一条 coalesced 的 32 线程 × 4 B load 触及 $128/32 = 4$ 个 sector（ $s_{\mathrm{warp}} = 4$），故所需在飞 warp load = $208/4 = 52$ 条。若每 warp 只有 1 条在飞 load（ILP 1），就需要 **52 个 active warp**， $\mathcal{O} = 52/64 = 0.81$。

> **坑**：服务率取 warp 发射率而非 HBM 服务率。触发条件：把「每 SM 每周期 1 条 warp load」（LSU 发射率，32 LD/ST ÷ 32 线程）当 $\lambda$ 代式 (1)，配 HBM 延迟 479；现象： $L = 1 \times 479 = 479$ 条 warp load → 需 479 个 warp，远超 64 warp slots，得出「HBM 延迟根本藏不住」的错误结论，高估约 9.2×（479/52）；规避：Little's law 的 $\lambda$ 与 $t_{\mathrm{lat}}$ 须指**同一个排队系统**——藏 HBM 延迟，系统是 HBM 通道， $\lambda$ 取每 SM 带宽份额 ÷ 32 B sector（式 (2)），不是 LSU 发射率。发射率配 HBM 延迟是混了两个系统（spec §5 1.4 口径注：per-SM 归一化，否则高估约 10×）。

## 5. 隐藏条件与 occupancy 连接

52 个在飞 warp load 怎么落到 occupancy 与 ILP 上？把命题 1 落到 warp 数。提供的在飞 sector 并发 = active warps × 每 warp 在飞独立 load 数 × 每条 load 的 sector 数，故隐藏条件：

$$
W_{\mathrm{active}} \cdot \mathrm{ILP} \cdot s_{\mathrm{warp}} \ge L_{\mathrm{SM}} \qquad (4)
$$

**符号**： $W_{\mathrm{active}}$ = active warps（1.2 式 (4) 上界）； $\mathrm{ILP}$ = 每 warp 在飞独立 load 指令数； $s_{\mathrm{warp}}$ = 每条 warp load 的 sector 数（coalesced fp32 = 4）； $L_{\mathrm{SM}}$ = 式 (3) 所需 sector 并发（208）。两边同除 $s_{\mathrm{warp}}$ 得 warp-load 口径： $W_{\mathrm{active}} \cdot \mathrm{ILP} \ge 52$。这正是 spec 的「 $n_{\mathrm{warps}} \cdot \mathrm{ILP} \ge \mathrm{latency} \times \mathrm{throughput}$」——右边 throughput 取瓶颈资源服务率（折 warp load 为 $0.434/4 = 0.108$ 条/cycle）， $478.8 \times 0.108 = 52$。

最小 occupancy（式 (4) 解出 $\mathcal{O}$）：

$$
\mathcal{O}_{\min} = \frac{L_{\mathrm{SM}}}{s_{\mathrm{warp}} \cdot \mathrm{ILP} \cdot W_{\mathrm{slots}}} \qquad (5)
$$

代入 $W_{\mathrm{slots}} = 64$：ILP 1 → $`\mathcal{O}_{\min} = 208/(4 \times 1 \times 64) = 0.81`$；ILP 2 → 0.41；ILP 4 → 0.20。**ILP 翻倍，所需 occupancy 减半**——这是 ILP 与 TLP 替代关系的定量形式。

连接 1.2 occupancy 案例（recap 式 (3)(4)，不重推）：

- 1.2 案例 C1（ $R_{\mathrm{th}} = 64$、256 线程）→ $W_{\mathrm{active}} = 32$。ILP 1 时 $32 \lt 52$，藏不住 HBM；ILP 2 时 $32 \times 2 = 64 \ge 52$，藏住。所以 C1 的 50% occupancy 配 ILP 2 即可喂满 HBM。
- 1.2 案例 C2（ $R_{\mathrm{th}} = 128$）→ $W_{\mathrm{active}} = 16$。须 ILP 4（ $16 \times 4 = 64 \ge 52$）才藏住。

这解释了 1.2 §6 那句「occupancy 高 ≠ 延迟藏住」的反面：**occupancy 低也未必藏不住，只要 ILP 补上乘积**。occupancy 提供 $W_{\mathrm{active}}$ 上界，ILP 提供每 warp 的在飞独立指令，两者相乘才是隐藏并发。

**命题 2**（ILP-TLP 乘积替代）。在 $s_{\mathrm{warp}}$ 固定（coalescing 不变）时，隐藏条件式 (4) 只依赖乘积 $W_{\mathrm{active}} \cdot \mathrm{ILP}$，故 ILP 与 TLP（active warps）沿双曲线可互相替代：低 occupancy 高 ILP 与高 occupancy 低 ILP 可达同一隐藏并发。**证明（要点）**：式 (4) 左边是乘积形式，约束只约束乘积、不约束各因子，故任意满足乘积 $\ge L_{\mathrm{SM}}/s_{\mathrm{warp}}$ 的 $(W_{\mathrm{active}}, \mathrm{ILP})$ 组合等价。完整证明（含 ILP 上界由独立指令数与寄存器预算 $R_{\mathrm{th}}$ 设定、非 coalesced 时 $s_{\mathrm{warp}}$ 变化的边界）留作练习 3。∎

命题 2 的边界须钉一句：替代不是免费的。ILP 提高要求每 warp 有更多独立指令在飞，每条独立 load 要自己的地址寄存器与目的寄存器—— $R_{\mathrm{th}}$ 上升，反过来压低 $W_{\mathrm{active}}$（1.2 §3 寄存器权衡）。所以 ILP 与 TLP 竞争同一个寄存器堆，乘积替代受寄存器预算约束（§6）。

## 6. ILP vs TLP：寄存器预算下的权衡

ILP 和 TLP 都能补隐藏并发，为什么不能无限拉高其中一个？因为两者都要寄存器，而寄存器堆只有 256 KB（1.2 §3）。

所谓 ILP（instruction-level parallelism，指令级并行），是指单个 warp 内在飞的独立指令数——独立 load（不同地址、互不依赖）、展开的独立 FMA 链。所谓 TLP（thread-level parallelism，线程级并行），是指可供调度器选择的 active warp 数，即 occupancy 提供的库存（1.2）。两者都增加在飞并发，但路径不同：ILP 让一个 warp 同时等多条 load，TLP 让更多 warp 轮流发射。

寄存器预算权衡（recap 1.2 §3）： $R_{\mathrm{th}}$ 大，单线程能驻留的独立地址/中间变量多、ILP 高，但每 block 占用多、能装的 block 少、 $W_{\mathrm{active}}$ 低； $R_{\mathrm{th}}$ 小则反之。1.2 案例 C1（ $R_{\mathrm{th}} = 64$、 $W_{\mathrm{active}} = 32$）与 C2（ $R_{\mathrm{th}} = 128$、 $W_{\mathrm{active}} = 16$）正是这条权衡的两端。代入式 (4)：C1 配 ILP 2、C2 配 ILP 4，乘积都是 64，都藏住 HBM——但 C2 的 ILP 4 要求每 warp 4 条独立 load 在飞，地址寄存器压力更高。最优 $R_{\mathrm{th}}$ 取决于 kernel 的延迟结构与独立指令数，编程模型侧的 launch_bounds 控制归 3.4。

异步 MMA 是第三条轴（recap 1.3 §5）：wgmma 异步发射后 warp-group 可继续做 CUDA Core 工作（softmax/地址/TMA），Tensor Core 在后台算——这是**跨管线**的重叠，不占 warp 的 load-ILP，也不直接占 occupancy。它把「等 MMA 结果」的 execution dependency 从 warp 的关键路径上挪走，等效于增加了可用并发。FlashAttention-3 的 warp specialization（一组 warp 专做 TMA、一组专做 MMA、一组专做 softmax）就是这条轴的 kernel 级运用（指令级走读归 6.3）。

> **提点**：延迟隐藏的本质是「用并发换等待」——硬件不能缩短 HBM 的物理延迟（479 周期是访存路径的结构性固有延迟），但可以用足够多的在飞请求让瓶颈资源在每个周期都有活干。Little's law 只是把这个「并发换等待」的守恒关系写成公式。删掉这条类比，式 (1)-(5) 的推导链依然完整。

## 7. Nsight Compute stall reason 读法

理论算出需 52 个在飞 warp load，实测怎么知道 kernel 藏没藏住？读 Nsight Compute 的 Warp Scheduler States 与 stall reason 分布（[NVIDIA, Nsight Compute](../references.md) ProfilingGuide §2.4.7，[C]）。

Warp Scheduler States 三态：eligible（依赖清、可发射）、issued（这周期被挑中发射）、stalled（卡住）。issue-slot 利用率 = 发射槽被用的比例。stall reason 按 warp-cycle 加权统计——哪个 reason 占比高，哪里就是主矛盾。诊断决策树（四分类 → 动作）：

| 主导 stall | 诊断 | 动作（指针） |
| --- | --- | --- |
| `long_scoreboard` 高、`not_selected` 低 | memory-latency-bound：在飞并发不足 | 提 occupancy 或 ILP（式 (4)）；或 tiling 把 HBM 换 shared（4.3）；或改善 coalescing 降 $s_{\mathrm{warp}}$ 浪费（4.1） |
| `wait` 高 | execution dependency：依赖链太串 | 展开独立链提 ILP；重排指令 |
| `barrier` 高 | sync：`__syncthreads` 等待 / 负载不均 | 减少同步点 / 均衡 warp 工作量（3.x） |
| `not_selected` 高 | 健康：可调度 warp 充足 | 发射槽不缺候选，瓶颈在别处（看吞吐/带宽利用率） |
| `mio_throttle` / `lg_throttle` 高 | 队列背压：shared/LSU 压力 | bank conflict（4.2）/ 访存指令过密 |

读法要点：`long_scoreboard` 主导且 `not_selected` 接近零，是延迟没藏住的典型签名——warp 都卡等访存、没有富余候选，对应式 (4) 左边 $\lt L_{\mathrm{SM}}$。此时再对照 occupancy 实测值（Nsight occupancy metrics，3.4 回链）与式 (5) 的 $`\mathcal{O}_{\min}`$：若实测 occupancy 低于 $`\mathcal{O}_{\min}`$ 且 ILP 补不上，就是并发不足。若 occupancy 已高但仍 `long_scoreboard` 主导，则可能是 coalescing 差（ $s_{\mathrm{warp}}$ 浪费，有效带宽减半，练习 4(c)）或带宽本身打满（memory-bound，2.4 roofline 判定）。

**测量方法边界**：本章给 stall reason 的语义与诊断映射；完整的 profiling 流程（Nsight Systems timeline → Nsight Compute kernel metrics → roofline 实测绘制 → 逐层 drill-down）归 11.2，届时 recap 本章四分类为 stall 读法起点。latency-bound 区（低 occupancy 时吞吐既不到算力峰值也不到带宽峰值）的 roofline 判定归 2.4——本章的 $`\mathcal{O}_{\min}`$ 正是 latency-bound 与 memory-bound 的分界输入。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 延迟 ≠ 吞吐：FMA 延迟 ~4 cycles [C] 但吞吐 128 results/cycle/SM（流水线解耦） | §2 时序图 + 表 | §2、练习 1 |
| 延迟逐层涨 shared 29 / L2 263 / HBM 479 cycles（实测 [B]），官方只给定性 hundreds | §2 表 + 口径注 | §2 |
| scoreboard 追踪寄存器依赖；eligible = 源就绪 + 管线可用 + 同步过；单发射挑 1（1.2） | §3 框图 | §3、练习 1 |
| stall 四分类：memory（long_scoreboard）/ execution（wait）/ sync（barrier）/ not selected | §3 表 | §3、§7 |
| Little's law L = λ·t_lat；吞吐 = 并发/延迟，利用率 = C/L_req | 式 (1)，命题 1 | §4、练习 1 |
| 藏 HBM 延迟需每 SM ≈208 sector（6.6 KB）在飞 = 52 warp load；ILP 1 需 52 warp（O 0.81） | 式 (2)(3) | §4、练习 1/2 |
| 服务率取 HBM per-SM 份额 ÷ 32B，非发射率；误用发射率高估 9.2×（479 warp） | §4 坑注 | §4、练习 2 |
| 隐藏条件 W_active·ILP·s_warp ≥ L_SM；O_min = L_SM/(s_warp·ILP·64)，ILP 翻倍 O_min 减半 | 式 (4)(5)，命题 2 | §5、练习 3 |
| 1.2 C1（32 warp）配 ILP 2、C2（16 warp）配 ILP 4 均藏住；occupancy 低未必藏不住 | §5 连接 | §5、练习 3 |
| ILP/TLP 竞争同一寄存器堆（R_th 权衡，1.2 §3）；异步 MMA 是跨管线第三轴（1.3 §5） | §6 | §6 |
| Nsight：long_scoreboard 高 + not_selected 低 = 延迟没藏住；诊断映射 → 11.2 全流程 | §7 决策树 | §7 |

## 练习

1.（计算 · [10]）H100 SXM 主口径（ $f = 1.83$ GHz）：(a) 用式 (2) 算每 SM 的 HBM sector 服务率 $\lambda_{\mathrm{SM}}$（先算带宽份额 3352/132）；(b) 用式 (3) 算藏住 HBM 延迟（478.8 cycles）所需在飞 sector 数 $L_{\mathrm{SM}}$；(c) 折成 warp load（ $s_{\mathrm{warp}} = 4$）与 ILP 1 所需 warp 数、 $\mathcal{O}$。校验：(a) ≈0.434 sectors/cycle（份额 25.4 GB/s）；(b) ≈208 sectors（≈6.6 KB）；(c) ≈52 warp load / 52 warp、 $\mathcal{O} \approx 0.81$，与 §4 逐数一致（容差 1%）。
2.（计算 · [20]）口径坑与灵敏度：(a) 若误把 warp 发射率（1 条 load/cycle/SM）当 $\lambda$ 配 HBM 延迟 479，得多少所需 warp、较正确值 52 高估几倍？(b) 改用 $f = 1.98$ GHz（FP32 类 clock）重算 $L_{\mathrm{SM}}$，变多少？(c) 藏 shared 延迟（29 cycles）：shared 服务率 128 B/cycle = 4 sectors/cycle，求所需在飞 sector 与 warp load。先预测 (a) 的方向再算。校验：(a) ≈479 warp、高估 ≈9.2×（与 §4 坑注一致）；(b) ≈192 sectors（clock 升、sectors/cycle 降）；(c) 116 sectors = 29 warp load。
3.（推导 · [20]）补全命题 2 的证明（ILP-TLP 乘积替代）：(a) 由式 (4) 说明约束只依赖乘积 $W_{\mathrm{active}} \cdot \mathrm{ILP}$（ $s_{\mathrm{warp}}$ 固定），故两因子沿双曲线替代；(b) 论证 ILP 上界由独立指令数与寄存器预算 $R_{\mathrm{th}}$ 设定（每条独立 load 需地址/目的寄存器），故替代受 1.2 §3 寄存器权衡约束；(c) 代入 1.2 C1（ $W_{\mathrm{active}} = 32$）与 C2（ $W_{\mathrm{active}} = 16$），求各自藏住 HBM 所需最小 ILP。校验：(c) C1 需 ILP 2（ $32 \times 2 = 64 \ge 52$）、C2 需 ILP 4（ $16 \times 4 = 64 \ge 52$），与 §5 连接段一致；(b) 方向与 1.2 命题 3「资源占用只增不减」（实际 occupancy ≤ 上界）相容。
4.（计算 · [30]）综合账（CC 9.0 库存，carveout 228 KB）：(a) kernel 每 block 256 线程、 $R_{\mathrm{th}} = 96$、 $S_{\mathrm{sh}} = 32$ KB，用 1.2 式 (3) 求 $W_{\mathrm{active}}$；ILP 1 能否藏住 HBM（52 warp load）？若不能，最小 ILP 是多少？(b) 若访存非 coalesced，每条 warp load 触及 8 sector（有用字节仍 128 B），所需在飞 sector 与 warp load 各变多少？(c) 由 (b) 说明带宽利用率代价。先预测 (a) 的绑定约束再算。校验：(a) $B_{\max} = 2$、 $W_{\mathrm{active}} = 16$（register 绑定），ILP 1 藏不住、最小 ILP 4（ $16 \times 4 = 64 \ge 52$， $16 \times 3 = 48 \lt 52$）；(b) sector 流量翻倍 → $L_{\mathrm{SM}} \approx 416$ sectors、warp load 仍 ≈52（ $416/8$）；(c) 带宽利用率 50%（128 有用 / 256 触及），与 4.1 coalescing 账一致。

## 参考

- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Table 5 CC 9.0 FP32 128 results/cycle/SM（吞吐口径）、算术延迟 prose「typically 4 cycles」（CC 7.0 措辞）、global「hundreds of clock cycles」定性（§2 两口径）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute docs（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-25——ProfilingGuide §2.4.7 Warp Stall Reasons（long_scoreboard / wait / barrier / not_selected / mio_throttle / lg_throttle / math_pipe_throttle / no_instructions / drain / dispatch_stall / branch_resolving / imc_miss / membar）、Warp Scheduler States（eligible/issued/stalled、issue-slot）、memory chart sector 口径（§3/§7）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——H800 微基准 Table IV 延迟 cycles：L1 40.7 / shared 29.0 / L2 263.0 / global 478.8（实测口径，§2 表与 §4 代入）；wgmma m64n128k16 dense 128 cycles（1.3 回链）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——132 SM、HBM3 3352 GB/s、TC-boost 1830 MHz / FP32-boost 1980 MHz（§4 带宽份额与 clock 换算）
