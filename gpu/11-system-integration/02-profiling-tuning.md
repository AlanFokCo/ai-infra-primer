# 11.2 Profiling 与调优方法论（Profiling and Tuning Methodology）

> 每个 kernel 的带宽利用率都有 85%，端到端步时却比 kernel 时间总和多出 33%——损失在 kernel 内、kernel 间隙，还是 CPU 侧？本章交付 profiling 全流程与调优方法论：Nsight Systems 系统级 timeline（kernel launch / copy / API trace、CPU-GPU 重叠、NVTX 标注）先把端到端时间拆成三段并归因，Nsight Compute 四面板（SpeedOfLight / roofline / stall reason / memory chart / occupancy）再读 kernel 内部，bottleneck 三区判定（recap 2.4）接 stall 签名逐层 drill-down，调优决策树（coalescing 4.1 → bank conflict 4.2 → tiling 4.3 → occupancy 3.4 → fusion 5.4 → 精度 6.1）选手段，调优前后带宽/吞吐对账（命题 1 四因子分解 + 命题 2 带宽饱和加速上界）确认收益，achieved roofline 逐 kernel 绘制与端到端 MFU 对账，profiling 开销与采样偏差的口径纪律，nvbandwidth / nccl-tests 通信基准（recap 2.4/10.2）与 DCGM 机群监控 | 前置：models 2.4 §6（roofline 入口，经 2.4 recap，不重推）、2.4（三区判定式 (6)(7) / 测量口径式 (8)(9) / 实测点上界式 (10)(11)）、1.4（Little's law 与 stall reason 四分类）、3.4（occupancy 精确计算与 NCU 读法）、3.3（stream / graph 与 launch 摊销）、4.1–4.3 / 5.4 / 6.1（调优手段机制章）、7.2 / 10.2（通信带宽账与 nccl-tests 方法）；下游 11.3（DCGM diag 健康检查与故障归因）、inference 9.1（serving 操作点校准） | 锚点：Nsight Systems UserGuide v2026.6（[NVIDIA, Nsight Systems](../references.md)，[C] 官方规格披露·一手权威）、Nsight Compute ProfilingGuide（[NVIDIA, Nsight Compute](../references.md)，[C]）、nvbandwidth（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现）、nccl-tests（[NVIDIA, nccl-tests](../references.md)，教学参照 + 官方实现）、DCGM（[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现）、datasheet（[NVIDIA, datasheet](../references.md)，[C]）、Best Practices（[NVIDIA, Best Practices](../references.md)，[C]）、NCCL repo 与 2.4 blog（[NVIDIA, NCCL](../references.md) / [NVIDIA, NCCL 2.4 blog](../references.md)，教学参照 / [C]）——官方文档 2026-11-26 一手核实

## 1. 33% 的时间不属于任何 kernel：三个问题与本章口径

一个推理 step 有 330 个 kernel（recap inference 3.4 / 3.3 §5 的 8B decode kernel 数口径），GPU 上执行时间合计 3.96 ms，墙钟却走了 5.93 ms——多出的 1.97 ms 不属于任何 kernel，Nsight Compute 看不见它，因为它是 kernel 之间的间隙。把这段间隙归因清楚，再谈 kernel 内优化，否则优化方向从一开始就错。本章回答三个问题：端到端时间怎么拆成 kernel / 间隙 / CPU 三段并归因（§2–§3）？kernel 的瓶颈落在 compute / memory / latency 哪一区、该选哪个调优手段（§4–§5）？调优前后的带宽/吞吐怎么对账、实测 roofline 怎么画、测量本身的开销与偏差怎么控制（§6–§9）？

**本章口径**（章脊锁定；本章为方法章，数值口径一次声明，表外不再换）：

| 口径 | SKU 与参数 | 来源 |
| --- | --- | --- |
| 主口径 | H100 SXM：BF16 dense 989.5 TFLOPS / HBM 3.35 TB/s（乘法链值 3352 GB/s）/ 132 SM | recap 2.4 §1/§3 同表（[NVIDIA, datasheet](../references.md) + [NVIDIA, Hopper whitepaper](../references.md)，[C]） |
| 通信口径 | HGX 8-GPU NVLink4：raw 450 GB/s 单向 / NCCL 模型有效 370.8 GB/s；跨机 per-rail CX7 线速 50 GB/s | recap 7.2 式 (2) / 10.2 §1（[NVIDIA, NCCL](../references.md)，教学参照） |
| 精度口径 | roofline 主精度 BF16 dense；copy 与 coalescing 例 fp32（沿 4.1）；精度对比行 FP8 | recap 2.4 §1 / 4.1 §1 |

**实测口径声明**（spec §6.2）：本机无 NVIDIA GPU，本章「实测」项交付的是**测量方法 + 官方披露/已发表数值**，不是新跑 benchmark。凡未引官方披露/已发表基准的自算数值，一律标**教学假设口径**。工具行为口径：Nsight Systems v2026.6 UserGuide、Nsight Compute ProfilingGuide（2026-11-26 一手核实）；profiler 选项与默认行为随版本漂移，命令以安装版本文档为准。

**混场声明**（spec §4.3，本章同用 roofline 与 occupancy）： $R$ = ridge， $R_{\mathrm{th}}$ = registers/thread，两者勿混。符号声明： $I$ = 算术强度（FLOP/byte）、 $R$ = ridge、 $F_{\mathrm{peak}}$ = 峰值算力、 $\mathrm{BW}$ = 带宽、 $u_B$ / $u_F$ = 带宽/算力利用率、 $F_{\mathrm{ach}}$ / $I_{\mathrm{ach}}$ = 实测算力/强度、 $\mathrm{loss}$ = 利用率损失、 $C$ = 提供的在飞并发（sector 口径）、 $L_{\mathrm{req}}$ = 所需并发（以上 recap 2.4）； $n_{\mathrm{SM}}$ = SM 数、 $\mathcal{O}$ = occupancy、 $W_{\mathrm{active}}$ = active warps、 $\mathrm{ILP}$ = 每 warp 在飞独立指令数、 $s_{\mathrm{warp}}$ = 每条 warp load 的 sector 数、 $L_{\mathrm{SM}}$ = 每 SM 所需在飞 sector 并发（recap 1.4/3.4）； $n_{\mathrm{kernel}}$ = 每 step kernel 数、 $t_{\mathrm{launch}}$ = 每 kernel CPU 提交开销（recap 3.3）；本章新增 $t_{\mathrm{busy}}$ / $t_{\mathrm{wall}}$ / $t_{\mathrm{gap}}$ = step 的 kernel 忙碌/墙钟/间隙时间、 $t_{\mathrm{dispatch}}$ = CPU 侧每 kernel 提交间隔、 $\mathrm{duty}$ = GPU 忙碌占比、 $\eta$ = coalescing 效率（有用字节 ÷ 实际 sector 流量，recap 4.1）、 $k$ = 融合链算子数（5.4 口径；4.2 的 bank 冲突路数也记 $k$，本章引其结论不重算，两场不同节不混式）。通信基准节（§8）： $n$ = rank/GPU 数、 $M$ = 消息总字节、 $\alpha$ = 每消息固定延迟、 $\beta$ = 每字节传输时间（本章 $\alpha$ = 消息延迟，非 inference 接受率）。裸字母 $n$ 专归 rank 数、 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 $t/\tau$ 带下标，不用裸 $T$（序列长专用）。

**边界声明**： $I$/吞吐上限/ridge 的推导归 models 2.4 §6（经 2.4 recap，**不重推**）；三区判定与 latency-bound 上界归 2.4 §5（本章 recap 判定流程，增量 = 把 NCU 读数接进判定）；stall reason 语义归 1.4 §7、occupancy 计算归 3.4（本章 recap 读法）；coalescing / bank conflict / tiling / fusion / 精度的机制与推导各归 4.1 / 4.2 / 4.3 / 5.4 / 6.1，本章只交付**选择与对账**；通信消息量账归 8.1/10.2（recap）；ECC / XID / 故障诊断归 11.3；serving 调度与引擎编排归 inference；K8S 编排归 cluster。代码 scope：机制片段级（命令片段 + NVTX 标注片段），不做模块走读。

## 2. Nsight Systems：先把端到端时间拆成三段

1.97 ms 的间隙不属于任何 kernel——它该从 timeline 的哪一行读出来？先回答一个更基本的问题：为什么不能直接上 Nsight Compute？因为 NCU 只测 kernel 内部，kernel 之间的间隙、CPU 提交节奏、拷贝与计算的重叠它一概不见；系统级问题必须先由系统级工具回答。所谓 Nsight Systems（nsys），是指 NVIDIA 的系统级 tracer：把 CPU 线程的 CUDA API 调用、GPU 侧的 kernel / memcpy / memset 执行、NVTX 标注记录在同一条时间轴上。它单 pass 采集、不 replay kernel、不 flush cache、不锁时钟（ProfilingGuide 工具对比节逐字口径，§7），所以它看到的是**接近生产态**的时间线。

timeline 的三行结构与读法：

```
CPU 线程行   │ API│  API│  API│  API│  API│  API│   ← CUDA API trace（launch/memcpy/sync）
             │launch│launch│launch│launch│launch│launch│
NVTX 行      │◄──────── step N ────────►│◄── step N+1 ──►│
GPU kernel 行     │K1│    │K2│    │K3│    │K4│         ← kernel 执行（含间隙）
GPU copy 行  │H2D│              │D2H│                   ← memcpy（CE 引擎）
                  ▲                ▲
                  └─ 间隙：GPU 等 CPU 下一次提交 ─┘
```

- **kernel launch / copy / API trace**：CPU 行的每次 launch API 与 GPU 行的 kernel 执行一一相关（correlation），两者的时间差 = 队列等待 + 提交开销。CUDA API trace 默认跳过非关键 API，`--cuda-trace-all-apis` 开全量——官方注「This feature may cause significant runtime overhead」（[NVIDIA, Nsight Systems](../references.md)，[C]）。memcpy 行区分 H2D / D2H / D2D 与 copy engine；CUDA graph 的 trace 粒度三档 graph / node / host-only，node 档官方注「may cause significant runtime overhead」，graph 档只录整图不录节点、开销最低。
- **CPU-GPU 重叠分析**：拷贝与计算的重叠收益上界归 3.3 命题 1（makespan 账，recap 不重推）；本节交付的是**间隙归因**——GPU 行出现空隙时，对照 CPU 行与 sync API 分三类：CPU 行持续忙碌且无 sync → CPU-bound（提交喂不饱 GPU）；有 `cudaStreamSynchronize` / event wait → 同步等待；CPU 行空闲而 GPU 行等前序 kernel/copy → GPU 侧依赖。三类的修法不同（§5 决策树系统级分支）。
- **NVTX 标注**：所谓 NVTX（NVIDIA Tools Extension），是指 NVIDIA 的工具扩展标注 API——在 CPU 代码里 push/pop 命名 range，给 timeline 加语义层。官方逐字「The Timeline View also projects these ranges onto the GPU timeline」（[NVIDIA, Nsight Systems](../references.md)，[C]）：CPU 侧标的 range 会投影到 GPU 时间轴，于是「step N 的间隙」可以直接归因到语义区间，不必对着裸 kernel 名猜。配合 `cudaProfilerStart()` / `cudaProfilerStop()`（`cuda_profiler_api.h`）可把采集限定在指定区间，跳过 warmup。

教学片段（NVTX 标注 + 采集区间，教学用，非优化实现）：

```cpp
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>
cudaProfilerStart();                  // 只采这段
for (int step = 0; step < 10; ++step) {
    nvtxRangePushA("decode_step");    // range 投影到 GPU timeline
    forward_one_step();
    nvtxRangePop();
}
cudaProfilerStop();
```

命令片段（教学参照，选项以安装版 UserGuide 为准）：

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph -o step python app.py
```

全数字实例（**教学假设口径**；kernel 数 recap inference 3.4，时长与间隔为教学假设，非 8B 权重字节账——权重-only 理想步时 4.79 ms 见 2.4 §7，两套口径勿混）：330 个 kernel、每个 GPU 时长 12 μs、CPU 侧提交间隔 18 μs（含框架 dispatch，其中 driver 提交约 5 μs 沿 inference 3.4 教学值）。间隔大于 kernel 时长，CPU 喂不饱 GPU，属 CPU-bound。三段账：

$$
t_{\mathrm{busy}} = n_{\mathrm{kernel}} \cdot t_{\mathrm{kernel}} = 330 \times 12 = 3960 \ \mathrm{\mu s} \qquad (1)
$$

$$
t_{\mathrm{wall}} = (n_{\mathrm{kernel}} - 1) \cdot t_{\mathrm{dispatch}} + t_{\mathrm{kernel}} = 329 \times 18 + 12 = 5934 \ \mathrm{\mu s} \qquad (2)
$$

$$
\mathrm{duty} = \frac{t_{\mathrm{busy}}}{t_{\mathrm{wall}}} = \frac{3960}{5934} = 0.667 \qquad (3)
$$

**符号**： $t_{\mathrm{kernel}}$ = 单 kernel GPU 时长； $t_{\mathrm{dispatch}}$ = CPU 侧相邻两次提交的间隔。**口径注**：式 (2) 取 CPU-bound、无同步、等长 kernel 口径——第 $i$ 个 kernel 在 $(i-1) t_{\mathrm{dispatch}}$ 时刻开始、末尾 kernel 的尾巴再加一个 $t_{\mathrm{kernel}}$；若 GPU-bound（间隔小于时长），间隙被队列填平， $t_{\mathrm{wall}} \approx t_{\mathrm{busy}}$，式 (2) 不适用。间隙 $t_{\mathrm{gap}} = 5934 - 3960 = 1974$ μs，占墙钟 33.3%。signature 核对：GPU 行有隙、CPU 行连轴转、无 sync API——CPU-bound 确诊。修法是 CUDA Graph（recap 3.3 §5：整段录图一次重放，抹掉逐 kernel 的 CPU dispatch）：重放后 kernel 背靠背， $t_{\mathrm{wall}} \approx 3960 + 5 = 3965$ μs（graph launch 约 5 μs，教学假设、与单 kernel launch 同量级），加速 $5934/3965 = 1.50$。**这 1.50 倍不来自任何 kernel 变快**——kernel 一个没动，动的是提交路径。drill-down 顺序由此定：先修占比最大的损失段（此处间隙 33.3%），再进 kernel 内部。

## 3. Nsight Compute：kernel 内部的四组面板

间隙归因之后，剩下的 3.96 ms 全是 kernel 时间——哪个 kernel 值得钻？按时间贡献排序取头部（§6 给逐 kernel 分解）。所谓 Nsight Compute（NCU），是指 NVIDIA 的 kernel 级 profiler：对单个 kernel 采集硬件计数器与 SASS 级指标，代价是 kernel replay 与测量口径受控（§7）。四组面板各回答一个问题，机制细节各有 recap 源：

- **SpeedOfLight（SOL）——「哪个资源先到顶」**：逐单元报 compute（SM）与 memory（DRAM）两个利用率百分比，官方逐字「the throughput reports the achieved percentage of utilization with respect to the theoretical maximum」（[NVIDIA, Nsight Compute](../references.md)，[C]）——即 $u_F$ 与 $u_B$ 的直接读数（recap 2.4 §6 式 (9)，不重推）。读法：哪个高，瓶颈方向就在哪边；两个都低，问题在并发（latency-bound，§4）。
- **Roofline chart——「点落在哪条 roof 下」**：achieved 点坐标 $(I_{\mathrm{ach}}, F_{\mathrm{ach}})$，点到 roof 的距离 = 改进机会（2.4 §7 命题 2 recap，Nsight 逐字「The distance from the achieved value to the respective roofline boundary … represents the opportunity for performance improvement.」）。层级版分 L1/L2/DRAM 三条 ceiling（逐字「Hierarchical roofline overlays can separate L1, L2, and DRAM bandwidth ceilings.」）；2026-11-26 重核的 section 名含 SpeedOfLight_Hierarchical{Double,Half,Single,Tensor}RooflineChart——Tensor Core kernel 须用 tensor 版并交叉核对 SM frequency 与 SASS 操作计数（2.4 §7 recap）。本章增量是逐 kernel 绘制工作流与 baseline 追踪（§6）。
- **Warp Scheduler States / stall reason——「warp 卡在哪」**：eligible / issued / stalled 三态与 issue-slot 利用率，stall 按 warp-cycle 加权；四分类诊断映射 recap 1.4 §7（`long_scoreboard` = memory 依赖、`wait` = 执行依赖、`barrier` = 同步、`not_selected` = 候选富余、`mio_throttle` / `lg_throttle` = 队列背压），不重推。本章增量是 stall 签名与 SOL 读数的**组合判定**（§4）。
- **Memory Workload Analysis / memory chart——「字节搬得值不值」**：memory chart 以图形化逻辑图给各存储子单元的 transfer size / hit rate / request 数（ProfilingGuide §2.10）；关键指标 Sectors/Req 的 achieved / ideal / excessive 三口径 recap 4.1 §6（fp32 全活跃最优 4，「大于 4 即 uncoalesced」是误读——float4 最优 16）；Mem Busy / 带宽利用率 / Mem Pipes Busy 三口径分开读（recap 2.4 §6：单元忙不等于带宽满）。
- **Occupancy——「驻留库存够不够」**：理论 occupancy（calculator/API 预测）与 achieved occupancy（实测时间平均）两口径，achieved ≤ 理论，差来自尾波量化与负载不均（recap 3.4 §6）；Utilization 堆叠条三态读绑定约束（哪种资源的 unallocated 段最大，哪种就是绑定约束），Graphs 锯齿对应粒度悬崖。

全数字实例（**教学假设口径**，kernel 与流量账 recap 4.1 §6 strideCopy）：stride-16 的 fp32 copy kernel（ $n_{\mathrm{vec}} = 16777216$，有用流量 128 MiB），NCU 读数：时长 320.33 μs、DRAM 字节 1024 MiB、Sectors/Req = 32（ideal 4、excessive 28）、 $u_B = 100\text{％}$、 $u_F \approx 0$。读法：带宽**打满**了，但 7/8 的 sector 是浪费——这是 coalescing 问题，不是并发问题，也不是带宽不够。若只看 SOL 的 $u_B = 100\text{％}$ 就判「已到顶、无优化空间」，恰好漏掉 8 倍的收益（§5 对账）。**面板之间要交叉读**：SOL 给方向，memory chart 给浪费定位，stall 给并发证据，occupancy 给库存约束。

## 4. Bottleneck 定位：三区判定接上 NCU 读数

同样是「慢」，三个区的损失在不同侧——怎么从实测读数判区？判定框架 recap 2.4 §5（不重推）： $I \ge R$ 且算力管线并发充足 → compute-bound（上限 $F_{\mathrm{peak}}$）； $I \lt R$ 且 $C \ge L_{\mathrm{req}}$ → memory-bound（上限 $I \cdot \mathrm{BW}$）； $C \lt L_{\mathrm{req}}$ → latency-bound（点在斜线下方， $F_{\mathrm{ach}} = I \cdot \mathrm{BW} \cdot \min(1, C/L_{\mathrm{req}})$）。其中 $L_{\mathrm{req}} \approx 208$ sectors = 52 条 coalesced warp load（recap 1.4 式 (3)，HBM 478.8 cycles 实测 [B] 口径）。本章增量是把**NCU 读数接进判定的每个输入**：

```
nsys 三段分解 ──► duty 低？──是──► 系统级分支（§5：graph / 融合 / stream）
      │否（时间都在 kernel 里）
      ▼
按时间贡献取 top kernel ──► NCU 采集
      │
      ├─ I_ach = FLOPs_ach / bytes_ach（DRAM 口径，2.4 式 (8)）
      ├─ u_B / u_F ← SOL；C ← achieved occupancy × ILP × s_warp
      ▼
  I_ach ≥ R？──是──► compute-bound：看 u_F 与指令 mix（5.2 / 6.1）
      │否
  C ≥ L_req？──是──► memory-bound：看 Sectors/Req 与流量（4.1 / 4.3 / 5.4）
      │否
      └─► latency-bound：stall 签名交叉验证 ──► 3.4 / 1.4
          （long_scoreboard 主导 + not_selected ≈ 0）
```

stall 签名是判定的交叉验证而非替代：`long_scoreboard` 主导且 `not_selected` 接近零 = 延迟没藏住（recap 1.4 §7），与 $C \lt L_{\mathrm{req}}$ 是同一件事的两个投影；若 occupancy 已高仍 `long_scoreboard` 主导，则查 coalescing（ $s_{\mathrm{warp}}$ 浪费）或带宽本身打满（memory-bound）。

端到端侧还需要一笔分解账，把 kernel 级读数与 step 级表现连起来。**命题 1**（端到端带宽四因子分解）。设 step 的有用 DRAM 字节为 $W$，kernel 忙碌总时长 $t_{\mathrm{busy}}$、墙钟 $t_{\mathrm{wall}}$，定义端到端有效带宽 $`\mathrm{BW}_{\mathrm{e2e}} = W/t_{\mathrm{wall}}`$。在 HBM 单资源瓶颈、纯访存流口径下：

$$
\mathrm{BW}_{\mathrm{e2e}} = \mathrm{BW} \cdot u_B \cdot \eta \cdot \mathrm{duty} \qquad (4)
$$

**符号**： $u_B$ = 忙碌加权 DRAM 带宽利用率（$`\sum \mathrm{bytes}_i / (\mathrm{BW} \cdot t_{\mathrm{busy}})`$）； $\eta$ = 有用字节占 DRAM 实际流量之比（coalescing 与流量浪费的总折扣）； $\mathrm{duty}$ = 式 (3)。**证明**（要点，定义链乘三步）：$`\mathrm{BW}_{\mathrm{e2e}} = W/t_{\mathrm{wall}} = (W/t_{\mathrm{busy}}) \cdot (t_{\mathrm{busy}}/t_{\mathrm{wall}})`$；前一因子 $`= (\mathrm{bytes}_{\mathrm{DRAM}}/t_{\mathrm{busy}}) \cdot (W/\mathrm{bytes}_{\mathrm{DRAM}}) = \mathrm{BW} \cdot u_B \cdot \eta`$；后一因子 $= \mathrm{duty}$。完整证明（含 compute-bound 口径把 $\mathrm{BW}$ 换 $F_{\mathrm{peak}}$、非纯访存流的边界）留作练习 3。∎

全数字代入（**教学假设口径**）：§2 的 step，设忙碌加权 $u_B = 0.85$、无 sector 浪费（ $\eta = 1$）、 $\mathrm{duty} = 0.667$：$`\mathrm{BW}_{\mathrm{e2e}} = 3352 \times 0.85 \times 1 \times 0.667 = 1901`$ GB/s，仅峰值的 56.7%。graph 修复后 $\mathrm{duty} \to 0.999$，$`\mathrm{BW}_{\mathrm{e2e}} \to 2846`$ GB/s（84.9%）。对照 kernel 侧天花板：把 $u_B$ 从 0.85 拉到 1 只有 $1/0.85 = 1.18$ 倍，而修间隙有 1.50 倍——**四个因子哪个离 1 最远，先修哪个**。这就是 drill-down 顺序的定量依据。正例：strideCopy 单 kernel（ $\mathrm{duty} = 1$、 $u_B = 1$、 $\eta = 1/8$）代入得 419 GB/s，与 §5 对账一致。边界反例：compute-bound step 的分解把 $\mathrm{BW}$ 换 $F_{\mathrm{peak}}$、 $\eta$ 换 MMA 指令占比口径，结构不变但因子语义换（练习 3(d)）。

latency-bound 的 drill-down 实例（recap 2.4 §5 表第三行 + 3.4 K2 案例，**教学假设口径**）：某 decode 类 kernel $I = 2.5$，achieved occupancy 0.25（16 warps）、ILP 1，提供并发 $C = 16 \times 1 \times 4 = 64$ sectors，低于 $L_{\mathrm{req}} = 208$。NCU 签名：SOL 双低（ $u_B = 30.8\text{％}$、 $u_F$ 更低）、`long_scoreboard` 主导、`not_selected` 近零、occupancy 面板 register 绑定。判区：latency-bound。代入 2.4 式 (7)： $F_{\mathrm{ach}} = 2.5 \times 3352 \times 64/208 = 2.58$ TFLOPS，loss 69.2%。修法沿 1.4 式 (4)：ILP 提到 4（ $16 \times 4 \times 4 = 256 \ge 208$），或提 occupancy 到 0.81（52 warps）；前者不动寄存器预算的分配结构、后者要过 3.4 的 register 悬崖账。ILP 4 后 $F_{\mathrm{ach}} = 8.38$ TFLOPS，点从 roof 下方拉回斜线（§6 绘制），加速 3.25 倍。

## 5. 调优决策树：从区到手段，前后对账

判区之后选哪个手段？六类手段各有机制章，本节交付选择树与对账方法。决策树（系统级分支承接 §2，kernel 级分支承接 §4）：

```
duty 低（系统级）
  ├─ CPU-bound ────────► CUDA Graph（3.3）/ fusion 减 kernel 数（5.4）
  └─ sync/依赖 ────────► stream/event 重组（3.3）/ copy-compute 重叠（3.3/4.4）
memory-bound（I < R，u_B 高）
  ├─ Sectors/Req > 最优 ─► coalescing / 向量化 / SoA 布局（4.1）
  ├─ mio_throttle + shared k 路冲突 ─► padding / swizzle（4.2）
  ├─ 复用低、HBM 流量大 ─► tiling（4.3）/ fusion 免中间 tensor（5.4）
  └─ 字节本身大 ────────► 低精度格式 / 量化（6.1/6.2）
latency-bound（C < L_req，long_scoreboard + not_selected≈0）
  ├─ occupancy 绑定 ───► register/shared 预算、launch_bounds（3.4）
  └─ ILP 不足 ─────────► 展开独立链（1.4）/ 异步搬运（4.4）
compute-bound（I ≥ R，u_F 高）
  ├─ MMA 占比低 ───────► 指令 mix / tile 几何（5.2/6.1）
  └─ 精度未用满 ───────► FP8/FP4 路径（6.1）
```

六类手段的机制 recap 与收益锚（机制推导各归其章，本表只给选择依据；锚点分已发表与教学两档）：

| 手段 | 机制一句话（recap） | 适用签名 | 收益锚 |
| --- | --- | --- | --- |
| coalescing（4.1） | $\eta = 1/\min(s, 8)$，浪费 sector 随 stride 涨至饱和 | Sectors/Req 超最优、excessive 大于 0 | strideCopy 8×（教学，下例）；BP stride-2 效率 50%（[NVIDIA, Best Practices](../references.md)，[C]） |
| bank conflict（4.2） | $k$ 路冲突串行 $k$ 周期，padding/swizzle 消除 | `mio_throttle`、shared 访问 $k$ 路 | BP V100 transpose 12.8 → 140.2 → 199.4 GB/s（合并 10.95× + padding 1.42×，已发表） |
| tiling（4.3） | 复用 $O(1) \to O(T_{\mathrm{tile}})$，强度 0.5 → 64（128 tile bf16） | 复用低、HBM 流量冗余 | BP V100 119.9 → 144.4 → 195.5 GB/s（已发表）；教学 GEMM 82.00 ms → 640.7 μs |
| occupancy（3.4） | $`\mathcal{O}_{\min}`$ 从隐藏账反解，ILP 翻倍所需减半 | register/shared 绑定、 $C \lt L_{\mathrm{req}}$ | K2： $R_{\mathrm{th}}$ 96 → 25%，launch_bounds(256,4) → 50%（教学） |
| fusion（5.4） | 省 $2(k-1)$ 个中间 tensor 读写 + 每 kernel 一次 launch | 小 tensor launch 主导 / 大 tensor IO 主导 | decode $k=4$ 省 15.94 μs、prefill 省 60.06 μs（教学，5.4 表） |
| 精度（6.1） | bytes 减半（memory-bound）或吞吐翻倍（compute-bound） | 字节主导 / MMA 精度未用满 | FP8/BF16 = 2×（k32/k16 几何比）；decode 理想步时 4.79 → 2.40 ms（recap 2.4 §7，教学推论） |

**口径注**：表中 V100 锚点是 Best Practices 官方已发表口径（锚点原测硬件，非本章主口径 H100），机制跨代同形、绝对值随硬件漂移；Summit 锚点（§8）是 NCCL blog 官方披露口径。教学档数值均为本章或源章的教学假设口径。

对账方法是本章的关键推导：调优前后必须同口径（同 SKU / 时钟 / cache 状态 / replay 设置，纪律见 §7），然后对三个数——有用字节 $W$（算法不变则不变）、DRAM 流量 $V$、时长 $t$。**命题 2**（带宽饱和调优的加速上界）。设调优前后 kernel 均 HBM 带宽饱和（ $u_B = 1$），有用字节 $W$ 不变，DRAM 流量分别为 $V_{\mathrm{before}}$、 $V_{\mathrm{after}}$，则

$$
\frac{t_{\mathrm{after}}}{t_{\mathrm{before}}} = \frac{V_{\mathrm{after}}}{V_{\mathrm{before}}}, \qquad \mathrm{speedup} = \frac{V_{\mathrm{before}}}{V_{\mathrm{after}}} \qquad (5)
$$

**证明**（一步比值）：带宽饱和口径 $t = V/\mathrm{BW}$，前后 $\mathrm{BW}$ 相同，取比值即得。∎ 正例：coalescing 修复把 $\eta$ 从 1/8 提到 1，流量比 8，加速上界 8。**边界反例**：若调优后不再带宽饱和（例如流量降下来后暴露了延迟瓶颈， $u_B$ 掉到 0.9），实际加速低于上界；若算法本身变了（fusion 减少有用字节）， $W$ 不变前提失效，须另对有用字节账。

全数字对账（strideCopy，recap 4.1 §6 式 (3)，**教学假设口径**）：

| 口径 | 调优前（ $s = 16$） | 调优后（ $s = 1$） |
| --- | --- | --- |
| $\eta$ / Sectors/Req | 1/8 / 32（ideal 4） | 1 / 4 |
| DRAM 流量 $V$ | 1024 MiB | 128 MiB |
| 时长 $t = V/\mathrm{BW}$ | 320.33 μs | 40.04 μs |
| $u_B$ | 100% | 100% |
| 有用带宽 $W/t$ | 419 GB/s（12.5%） | 3352 GB/s（100%） |

加速 $320.33/40.04 = 8.0$，恰为流量比（式 (5)）。注意表中最反直觉的一行：**$u_B$ 前后都是 100%**——带宽利用率没变，变的是搬的字节里有多少是有用的。所以「 $u_B$ 到顶」不等于「优化到顶」，coalescing 浪费要靠 Sectors/Req 与 excessive 读数定位（§3），这正是命题 1 里 $\eta$ 因子独立存在的意义。灵敏度：若修复后 $u_B$ 只有 0.9，时长 $128 \times 2^{20}/(0.9 \times 3352 \times 10^{9}) \times 10^{6} = 44.49$ μs，加速 7.20，低于上界 8（命题 2 边界反例的数值化）。fusion 的对账另一笔（recap 5.4 §2，教学假设口径）：decode $B = 64$ 的 $k = 4$ 逐元素链，IO 省 3.15 MB 折 0.94 μs，launch 省 3 次 × 5 μs = 15 μs，合计约 15.94 μs——小 tensor 上 launch 项是 IO 项的 16 倍；prefill $T = 4096$ 同链 IO 省 201.33 MB 折 60.06 μs，IO 主导。同一手段在不同口径下收益结构不同，对账先问 tensor 多大。

> **提点**：profiling 的顺序格言是「先 kernel 外、后 kernel 内；先判区、后选手段」——timeline 决定看哪个 kernel，区决定用哪类手段，对账决定收益是否真实。删掉这条格言，式 (4)(5) 与决策树的推导链依然完整。

## 6. Achieved roofline 绘制：逐 kernel 分解与端到端对账

一个 step 的几百个 kernel 怎么画进同一张 roofline 图、又怎么与端到端 MFU 对账？2.4 §7 交付了单 kernel 的实测点读法，本节交付整 step 的绘制工作流与对账。方法五步（测量口径沿 2.4 §6，不重推）：

1. **nsys 取 kernel 清单与时长**：按时间贡献排序，记录每类 kernel 的个数与单时长（§2 的 trace 输出）。
2. **NCU 逐 kernel 取坐标**：$`I_{\mathrm{ach}} = \mathrm{FLOPs}_{\mathrm{ach}}/\mathrm{bytes}_{\mathrm{ach}}`$、$`F_{\mathrm{ach}} = \mathrm{FLOPs}_{\mathrm{ach}}/t_{\mathrm{kernel}}`$（2.4 式 (8) recap；FLOPs 用 SASS 操作计数器，FFMA 计 2，DRAM 字节读+写）。
3. **画 theoretical roof**：平台 $F_{\mathrm{peak}}$ 与斜率 $\mathrm{BW}$ 取 datasheet dense 值（2.4 §3 两条乘法链）；层级 ceiling 用 NCU 层级 roofline 的 L1/L2/DRAM 实测带宽。log-log 坐标，斜线斜率 1。
4. **逐点标注**：kernel 名 + 时间占比；Tensor Core kernel 用 tensor roofline section 并核对 SM frequency（§3）。NCU 的 baseline 功能可把调优前后两次采集叠在同一图上追踪点位移动——官方逐字「Using the baseline feature in combination with roofline charts, is a good way to track optimization progress over a number of kernel executions.」（[NVIDIA, Nsight Compute](../references.md)，[C]）。
5. **端到端对账**：step 的 MFU 是各 kernel $u_F$ 的时间加权平均再乘 duty（2.4 §7 MFU 段 recap），量化为

$$
\mathrm{MFU}_{\mathrm{step}} = \frac{\sum_i \mathrm{FLOPs}_i}{F_{\mathrm{peak}} \cdot t_{\mathrm{wall}}} = \mathrm{duty} \cdot \frac{\sum_i \mathrm{FLOPs}_i}{F_{\mathrm{peak}} \cdot t_{\mathrm{busy}}} \qquad (6)
$$

全数字实例（**教学假设口径**；§2 的 330-kernel step 拆两类，GEMV 类读数 recap 2.4 §6 点 B）：

| 类 | 个数 | 单时长 | 时间占比 | $I_{\mathrm{ach}}$ | $F_{\mathrm{ach}}$ | $u_B$ | 区 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEMV 类 | 30 | 100 μs | 75.8% | 2.5 | 7.125 TFLOPS | 85.1% | memory-bound |
| elementwise 类 | 300 | 3.2 μs | 24.2% | 0.083 | 0.238 TFLOPS | 85.1% | memory-bound |

对账：总 FLOPs $2.16 \times 10^{10}$、总 DRAM 字节 11.29 GB；忙碌加权 $u_F = 0.551\text{％}$，乘 duty 0.667 得 $`\mathrm{MFU}_{\mathrm{step}} = 0.368\text{％}`$。读法：decode 类负载 MFU 天然低（强度 2.5 离 ridge 295 差两个数量级，models 2.4 §6 结论 recap），**优化目标是 $u_B$ 与 duty，不是 MFU**——MFU 是对账读数，不是目标读数。两类 kernel 都贴斜线（ $u_B$ 85.1%），剩余 14.9% 的带宽损失去向由 memory chart 分解（§3）。

实测 roofline 示意（H100 BF16 主口径，log-log，示意非按比例；点位 recap 2.4 §7 并加本章调优轨迹）：

```
 F_ach (TFLOPS, log)
 ▲
 │                 ┌─────────────────────── F_peak = 989.5
 │                /│
 │               / │   斜线 = I × BW（BW = 3.35 TB/s）
 │              /  │
 │             /   │          ● A (400, 700) compute-bound
 │            /    │
 │           /     │
 │          /      │
 │         /       │
 │        /        │
 │       / ● B (2.5, 7.125) memory-bound，u_B 85.1%
 │      /  ▲
 │     /   │ ILP 1→4：C 64→256 sectors，点 C→B（§4）
 │    /  ○ C (2.5, 2.58) latency-bound，loss 69.2%
 │   / ● E (0.083, 0.238) elementwise
 │  /
 │ /
 └─────────────────┼──────────────────────► I_ach (FLOP/byte, log)
                   R ≈ 295
```

三条轨迹的读法：C → B 是**垂直上移**（强度不变、并发补齐，latency-bound 修复）；B 沿斜线向 roof 贴近是 $u_B$ 收窄（coalescing / 流量优化， $\eta$ 因子）；若 tiling 提复用，点所在的斜线换到更高的 L2/L1 ceiling、强度不变点上移（2.4 §7 层级 roofline recap，4.3 的几何解释）。A 点（ $u_F$ 70.7%）的损失在算力侧，与 B/C 不同侧——三区判定（§4）决定看哪条 roof。

## 7. Profiling 开销与采样偏差：测量本身改变被测量

profiler 报的时长为什么和生产态不一样？差异不是单一「开销」，是四个口径差加采样偏差。NCU 的采集机制是 kernel replay：官方逐字「all metrics requested for a specific kernel instance … are grouped into one or more passes」（[NVIDIA, Nsight Compute](../references.md)，[C]）——单 pass 装不下的指标分多 pass 采，pass 之间保存并恢复 kernel 写过的显存，保证每 pass 看到相同内存内容。指标越多 pass 越多，开销越大；software-patched 指标开销最高（改 kernel 指令），且官方明确该 pass 不采时长；硬件计数器开销很低。

四个口径差（ProfilingGuide「Workload Durations」节逐字，跨工具对账前必读）：

| 口径 | Nsight Compute | Nsight Systems |
| --- | --- | --- |
| 时钟 | 「locks SM clocks by default」（锁到 boost 值，消除 kernel 位置与 pass 序号的时钟态差异） | 「does not lock clocks」 |
| cache | 「flushes all GPU caches by default between replay passes」 | 单 pass 采集，「does not flush any caches」 |
| 串行化 | 「serializes kernel launches, unless a dedicated replay mode is used」 | 「does not serialize launches」 |
| 时长测法 | `gpu__time_duration`（semaphore 时间戳，单 kernel） | CUPTI 测 workload 时长（含间隙，2.4 §6 recap） |

后果与对策：NCU 默认测的是**冷 cache + 锁频 + 串行**的 kernel，不是生产态——cache 敏感负载（工作集被 L2 吸收的 decode 小 tensor）用 `--replay-mode application --cache-control none` 让应用自己暖 cache；跨工具比时长先统一时钟（官方建议 nvidia-smi 外部锁频 + `--clock-control none`）；并发 kernel 的时长差异用 range replay（整段 API + kernel 捕获重放，不串行化）。两条硬边界：热节流由硬件/驱动主导，工具锁频压不住它（逐字「Thermal throttling … always overrides any selected options」，诊断归 11.3）；MIG 分区上 `--clock-control` 会失败或被静默忽略（11.1 的分区口径在此有测量后果）。另逐字：NCU profiling 期间**不能**用 host 计时或 CUDA event 推 kernel 时长——那会把工具开销计进去。

> **坑**：拿 NCU 时长直接对 nsys 时长算「profiling 开销」。触发条件：同一 kernel 两个工具各报一个时长，相减当开销；现象：差值里混着时钟态、冷 cache、串行化三个口径差，不是纯开销，可能为负也可能数倍；规避：先按上表统一口径再比，或干脆各报各的口径、不做跨工具减法。

采样偏差三源。其一，**短 kernel 的多 pass 误差**：hit rate 类指标若分子分母在不同 pass 采集、kernel 又未达稳态，误差显著——官方给的稳态量级是「generally > 20 µs」（逐字），§6 表里 3.2 μs 的 elementwise 类正落在误差敏感区，对策是增大被测 workload 或减少同采指标数让计数器落进单 pass。其二，**异步引擎干扰**：copy engine 等与 SM 共享 L2/DRAM/PCIe/NVLink，小 kernel 的 DRAM 读数易被污染（逐字口径），对策是空闲 GPU 上 profiling。其三，**时间采样混叠**：nsys 的 GPU metrics 按固定频率采样（`--gpu-metrics-frequency`，默认 10000 Hz）、CPU IP sampling 是统计采样——周期与采样周期合拍的 workload 会系统性偏读，短函数在 IP 采样里欠代表；DCGM 的 field 采样同理（§9）。nsys 侧开销旋钮：`--cuda-trace-all-apis` / backtrace / graph node 档均注「may cause significant runtime overhead」；`--cuda-flush-interval` 控制 buffer 落盘节奏，官方逐字「The CUDA data buffer saves may cause profiler overhead」，间隔调大省落盘开销但涨内存占用。纪律收尾：采集区间用 `cudaProfilerStart/Stop` + NVTX capture-range 限定；对比实验保持采集设置逐次相同（官方逐字「Keep capture settings identical across runs」）；从小 workload 起步、下结论前恢复代表性 workload。

## 8. 通信基准：nvbandwidth 与 nccl-tests

kernel 调完，all-reduce 还是慢——怎么区分硬件天花板与库配置？两个工具、两个口径（recap 2.4 §6 的框架，不重述全量）：**nvbandwidth 测拷贝天花板**（硬件库存口径），**nccl-tests 测集合通信达成值**（负载口径）。流程位置：先跑 nvbandwidth 确认设备健康与天花板分母可靠，再跑 kernel/集合 profiling——否则利用率可能对着错误的天花板算。

nvbandwidth（[NVIDIA, nvbandwidth](../references.md)，教学参照 + 官方实现，README 2026-11-26 重核）要点 recap 2.4 §6：CE / SM 两类拷贝（CE 走 memcpy API 不占 SM，SM 走 kernel 占算力）、spin kernel + CUDA events 排除 enqueue 开销、默认 3 采样取中位数、pointer chasing 测延迟。两条对账口径守卫（2.4 §6 坑注 recap）：D2D 同卡拷贝报值对 HBM 峰值须 ×2（读+写双流量）；bidirectional 报值是并发反向流量下的**每方向**带宽、非双向总和（NVLink 900 GB/s 是双向总和口径）。P2P 带宽矩阵的逐对读法归 7.2。

nccl-tests（[NVIDIA, nccl-tests](../references.md)，教学参照 + 官方实现，README + doc/PERFORMANCE.md 重核）要点 recap 10.2 §8：两个带宽口径——algbw = $M/t$（消息字节 ÷ 实测时间），busbw = algbw × 修正因子，因子表 AllReduce $2(n-1)/n$、ReduceScatter / AllGather / AlltoAll $(n-1)/n$、Broadcast / Reduce 1。官方逐字「The bus bandwidth should reflect the speed of the hardware bottleneck: NVLink, PCI, QPI, or network」——busbw 对的是硬件瓶颈段，独立于 rank 数，可与峰值直比。命令与读法：

```bash
all_reduce_perf -b 8 -e 8G -f 2 -g 1   # 8 B 到 8 GB、倍率 2、每进程 1 GPU
```

读法三条（recap 10.2 §8）：小消息段延迟项主导、busbw 低；大消息段 busbw 趋硬件瓶颈——NVLink 域内趋有效带宽 370.8 GB/s（7.2 模型口径），rail-optimized 跨机趋 per-rail NIC 线速 50 GB/s；busbw 远低于预期查 rail 映射 / 多 channel / GDR / 拥塞（10.2 §6–§7）。ring/tree 交叉临界 $M^{\ast}$ 的实测须引入 $\beta_{\mathrm{eff}}$ 差（recap 8.3，理想 α-β 模型解不出，不重推）。

全数字对账（**教学假设口径**，时间取 7.2 α-β 模型值非新跑）： $n = 8$、 $M = 1$ GB、 $t = 4.75$ ms（recap 7.2 §3：带宽项 4.72 ms + 延迟项 0.028 ms）：

$$
\mathrm{algbw} = \frac{M}{t} = \frac{10^{9}}{4.75 \times 10^{-3}} = 210.5 \ \mathrm{GB/s}, \qquad \mathrm{busbw} = \mathrm{algbw} \times \frac{2(n-1)}{n} = 210.5 \times 1.75 = 368.4 \ \mathrm{GB/s} \qquad (7)
$$

busbw 368.4 对 NCCL 模型 370.8 GB/s 差 0.6%——差来自延迟项与 $t$ 的舍入（精确模型时 4.7475 ms 给 busbw 368.6，仍差 0.6%，即 $\alpha$ 项的确定性折扣），大消息极限下 busbw 收敛到有效带宽。**基准读数与机制模型闭环互证**：busbw 显著低于 370.8 才是配置问题，0.6% 不是。已发表大规模锚（官方披露，非新跑）：NCCL 2.4 blog（[NVIDIA, NCCL 2.4 blog](../references.md)，[C]）Summit 24,576 GPUs 双树对 ring 延迟最高 180× 改善（recap 10.2 §8；180× 是实测系统级、含带宽项与重叠，与纯延迟项理论比 731× 两口径勿混）。

## 9. DCGM：从单机 profiling 到机群监控

机群里哪张卡不正常——这个问题和「这个 kernel 为什么慢」差在哪？前者要常驻、低开销、覆盖全部卡，后者要深、可以高开销、只盯一个 kernel；监控与 profiling 是两层工具，不能互相替代。所谓 DCGM（Data Center GPU Manager），是指 NVIDIA 的数据中心 GPU 管理工具集（[NVIDIA, DCGM](../references.md)，教学参照 + 官方实现，repo 与 docs 2026-11-26 重核），三类能力：Field Value Watches（field 周期采样）、Active Health Checks（子系统健康检查）、Policy Notification（越限策略通知）。它与 profiling 的口径分工：

| 工具 | 作用域 | 时间口径 | 开销 | 归因粒度 |
| --- | --- | --- | --- | --- |
| DCGM | 机群常驻 | 采样间隔的时间平均 | 低 | 设备 / field |
| nsys | 单进程 | 单 pass trace | 低–中（选项相关，§7） | API / kernel / NVTX 区间 |
| NCU | 单 kernel | replay 多 pass | 高（§7） | kernel / SASS |

field 按族描述（语义据官方 field-ids 页 2026-11-26 核实转录；**field ID 是版本化常量、名称随版本漂移**——如功率 field 现行页印 BOARD_POWER_WATTS，脚本以部署版本的 field-ids 页为准，勿硬编码旧名）。profiling 相关的是 PROF 利用率族与设备状态族：

| field（名） | ID | 语义（官方页口径） |
| --- | --- | --- |
| DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO | 1001 | graphics/compute engine 活跃时间比 |
| DCGM_FI_PROF_SM_UTIL_RATIO | 1002 | SM 至少驻留 1 条 warp 的周期占比 |
| DCGM_FI_PROF_SM_OCCUPANCY_RATIO | 1003 | 驻留 warp 数对理论最大 warp 数的时间平均比 |
| DCGM_FI_PROF_TENSOR_UTIL_RATIO | 1004 | tensor pipe 活跃周期占比（对 peak sustained elapsed） |
| DCGM_FI_PROF_DRAM_UTIL_RATIO | 1005 | 显存接口收发数据的活跃周期占比 |
| DCGM_FI_PROF_FP32_UTIL_RATIO | 1007 | FP32 pipe 活跃周期占比 |
| DCGM_FI_PROF_FP16_UTIL_RATIO | 1008 | FP16 pipe 活跃周期占比（不含 HMMA） |
| DCGM_FI_PROF_PCIE_TX_BYTES | 1009 | PCIe 活跃发送字节（含 header + payload，GPU 视角：DtoH 计 tx） |
| DCGM_FI_PROF_PCIE_RX_BYTES | 1010 | PCIe 活跃接收字节（HtoD 计 rx） |
| DCGM_FI_PROF_NVLINK_TX_BYTES | 1011 | NVLink 活跃发送字节（含 header + payload） |
| DCGM_FI_PROF_NVLINK_RX_BYTES | 1012 | NVLink 活跃接收字节 |
| DCGM_FI_DEV_SM_CLOCK | 100 | SM 时钟 |
| DCGM_FI_DEV_GPU_TEMP_CELSIUS | 150 | GPU 温度 |
| DCGM_FI_DEV_MEMORY_TEMP_CELSIUS | 140 | 显存温度 |
| DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION | 156 | 能耗（mJ，自 driver reload） |

PROF 族依赖 GPM（GPU Performance Monitoring）硬件支持（field 表有 DCGM_FI_DEV_GPM_SUPPORT 查询项，ID 94）。ECC 计数、row remap、NVLink CRC / replay / recovery 错误计数等 field 族同样可监控，其语义与故障归因归 11.3，本章只留监控入口。

> **坑**：把 DCGM DRAM active ratio 当 $u_B$。触发条件：拿 field 1005 的读数直接除以峰值带宽当利用率；现象：active 周期占比 ≠ 字节 ÷ 峰值——空闲周期与突发流量都使两者偏离，busy 但低效的访存（如 stride 浪费）照样报高 active；规避： $u_B$ 的权威读数在 NCU SOL（字节口径），DCGM 给的是机群筛查信号，发现异常再下钻 profiling。同理 SM occupancy ratio（1003）是时间平均驻留比，不是 3.4 的理论 occupancy；FP16 pipe 不含 HMMA，BF16 GEMM 负载应读 tensor ratio（1004）而非 FP16。

诊断侧：`dcgmi diag` 按 run level 分级（官方逐字表，高级含低级）：

| level | 时长口径 | 测试成员（增量） |
| --- | --- | --- |
| r1 Short | 小于 2.5 s | software |
| r2 Medium | 4-GPU 小于 2.5 min / 8-GPU 小于 10.5 min | + pcie（PCIe + NVLink）/ memory / memory_bandwidth |
| r3 Long | 4-GPU 小于 10 min / 8-GPU 小于 35 min | + diagnostic / targeted_stress / targeted_power / **nvbandwidth / nccl_tests** |
| r4 Extra Long | 4-GPU 小于 45 min / 8-GPU 小于 2.25 h | + memtest / pulse（Input EDPp） |

r3 把 §8 的两个基准工具内嵌为 plugin——作业前健康检查与周期性巡检直接复用同一套测量；`dcgmi diag -r "pcie,diagnostic"` 可指名测试、`-r 3 -p diagnostic.test_duration=300.0` 传参。用法分工：DCGM 常驻采样做机群筛查与趋势（温度/功率/错误计数），异常触发后 nsys → NCU 下钻归因，故障确诊与恢复动作归 11.3。采样间隔口径：低于间隔的瞬时尖峰不可见（与 §7 时间采样混叠同源），报「监控没报」不等于「没发生过」。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| nsys 系统级单 pass（不 replay / 不 flush / 不锁频）；间隙归因三类：CPU-bound / sync / GPU 依赖；NVTX range 投影 GPU timeline（逐字） | §2 三行 timeline + 归因 | §2、练习 1 |
| 三段账：busy 3960 μs / wall 5934 μs / gap 33.3%（330 kernel × 12 μs、间隔 18 μs，教学假设）；graph 修复加速 1.50，kernel 一个没动 | 式 (1)(2)(3) | §2、练习 1 |
| NCU 四面板交叉读：SOL 给方向（ $u_B$/ $u_F$ 逐字）、roofline 给距离、stall 给并发证据、memory chart 给浪费定位、occupancy 给库存 | §3（recap 2.4/1.4/4.1/3.4） | §3 |
| $u_B = 100\text{％}$ 不等于优化到顶：strideCopy 浪费 7/8 sector，Sectors/Req 32 对 ideal 4 | §3 实例（recap 4.1） | §3、练习 2 |
| 三区判定接 NCU 读数： $I_{\mathrm{ach}}$ 对 $R$、 $C$ 对 $L_{\mathrm{req}}$（208 sectors）、stall 签名交叉验证 | §4 流程（recap 2.4 §5/1.4 §7） | §4、练习 4 |
| 端到端四因子分解 $`\mathrm{BW}_{\mathrm{e2e}} = \mathrm{BW} \cdot u_B \cdot \eta \cdot \mathrm{duty}`$；0.85/1/0.667 → 1901 GB/s（56.7%）；哪个因子离 1 最远先修哪个 | 式 (4)，命题 1 | §4、练习 3 |
| latency-bound 实例：O 0.25 ILP 1 → C 64、F_ach 2.58 TFLOPS、loss 69.2%；ILP 4 → 8.38 TFLOPS、3.25×（recap 2.4 §5） | §4 实例 | §4、练习 4 |
| 带宽饱和调优加速 = 流量比（命题 2）；strideCopy 1024 → 128 MiB、320.33 → 40.04 μs、8×； $u_B$ 前后均 100%，有用带宽 419 → 3352 GB/s | 式 (5) + 对账表 | §5、练习 2 |
| 决策树：duty 低修系统级；memory-bound 按 Sectors/Req / 冲突 / 复用 / 字节分流；latency-bound 修 occupancy/ILP；compute-bound 修指令 mix/精度 | §5 树 + 六手段表 | §5 |
| 收益锚：BP V100 transpose 12.8 → 140.2 → 199.4 GB/s、tiling 119.9 → 144.4 → 195.5（已发表）；fusion decode 15.94 μs / prefill 60.06 μs（教学） | §5 表 | §5 |
| roofline 绘制五步 + baseline 追踪（逐字）；MFU_step = duty × 忙碌加权 u_F = 0.368%（decode 类 MFU 天然低，目标是 u_B 与 duty） | 式 (6) + 两类分解表 | §6、练习 4 |
| NCU 四口径差：锁频 / flush cache / 串行化 / 时长测法（逐字）；短 kernel 多 pass 误差（稳态 generally > 20 µs）；热节流压不住、MIG 上锁频失效 | §7 表 + 偏差三源 | §7 |
| nccl-tests algbw = M/t、busbw × 2(n-1)/n；n=8 M=1GB t=4.75ms → 210.5 / 368.4 GB/s，对模型 370.8 差 0.6%（α 项 + 舍入）；Summit 24576 GPU 180×（已发表） | 式 (7) | §8、练习 4 |
| DCGM 三能力 + PROF field 族（active 周期占比 ≠ u_B）；diag r1–r4，r3 内嵌 nvbandwidth/nccl_tests；field ID 版本化、名称漂移 | §9 两表 + 坑注 | §9 |

## 练习

1.（计算 · [10]）三段分解与 duty（式 (1)-(3)，教学假设口径）：330 kernel、单时长 12 μs、CPU 提交间隔 18 μs。(a) $t_{\mathrm{busy}}$；(b) $t_{\mathrm{wall}}$（CPU-bound 口径）；(c) duty 与间隙占比；(d) graph 修复后墙钟（busy + 5 μs）与加速比。先预测 (c) 的间隙占比量级再算。校验：(a) 3960 μs；(b) 5934 μs；(c) 0.667 / 33.3%；(d) 3965 μs / 1.50，与 §2 实例逐数一致（容差 1%）；kernel 数口径 recap inference 3.4。
2.（计算 · [20]）strideCopy 对账（命题 2，recap 4.1 §6）： $n_{\mathrm{vec}} = 16777216$ fp32、 $s = 16$。(a) $\eta$ 与 Sectors/Req；(b) DRAM 流量与 $t_{\mathrm{copy}}$；(c) 修复为 $s = 1$ 后的流量与时长；(d) 前后有用带宽与加速比，核对命题 2 上界；(e) 若修复后 $u_B = 0.9$，实际时长与加速比。先预测 (d) 的加速上界再算。校验：(a) 1/8 / 32（ideal 4）；(b) 1024 MiB / 320.33 μs；(c) 128 MiB / 40.04 μs（4.1 §6 表公开数值锚）；(d) 419 → 3352 GB/s、8×；(e) 44.49 μs、7.20×（低于上界，命题 2 边界反例方向）。
3.（推导 · [30]）补全命题 1 的证明（端到端四因子分解）：(a) 由 $`\mathrm{BW}_{\mathrm{e2e}} = W/t_{\mathrm{wall}}`$ 出发，插入 $t_{\mathrm{busy}}$ 拆成两因子，推出式 (4)；(b) 代入 strideCopy 单 kernel（duty 1、 $u_B$ 1、 $\eta$ 1/8）与 §2 step（ $u_B$ 0.85、 $\eta$ 1、duty 0.667）数值验证；(c) 解释为什么 kernel 侧 $u_B$ 0.85 → 1 至多 1.18× 而修间隙有 1.50×，由此给出 drill-down 顺序规则；(d) 边界：compute-bound step 把 $\mathrm{BW}$ 换 $F_{\mathrm{peak}}$ 时四因子各换什么语义，非纯访存流（kernel 间有 L2 复用）时 $\eta$ 口径怎么变。校验：(b) 419 GB/s / 1901 GB/s 与 §4 逐数一致（容差 1%）；(c) $1/0.85 = 1.176$、 $5934/3965 = 1.497$；(d) 与 §4 边界反例及 2.4 §6 $I_{\mathrm{ach}}$ 分母口径注一致。
4.（计算 · [30]）通信读数与 roofline 点综合账（式 (7)，recap 2.4 §5/§7）：(a) $n = 8$、 $M = 1$ GB、 $t = 4.75$ ms 的 algbw 与 busbw；(b) busbw 对 370.8 GB/s 模型的相对差，并说明差的来源；(c) $n = 64$ 的 AllReduce 因子；(d) $I = 2.5$ kernel、achieved occupancy 0.25、ILP 1：提供并发 $C$、 $F_{\mathrm{ach}}$（2.4 式 (7)）、loss；ILP 提到 4 后的 $F_{\mathrm{ach}}$ 与加速比；(e) 写出 (d) 前后两点的 roofline 坐标与各自 roof。先预测 (b) 的差是否算「配置问题」再算。校验：(a) 210.5 / 368.4 GB/s；(b) ≈0.6%，α 项 + 舍入、非配置问题（§8）；(c) 1.96875（10.2 §8 因子表）；(d) 64 sectors / 2.58 TFLOPS / loss 69.2% → 8.38 TFLOPS / 3.25×（2.4 §5 表公开数值锚，容差 1%）；(e) (2.5, 2.58) 与 (2.5, 8.38)，roof = 8.375 TFLOPS（3.35 TB/s 口径）。

## 参考

- [NVIDIA, Nsight Systems](../references.md)：Nsight Systems UserGuide v2026.6（key: nsight-systems），[C] 官方规格披露·一手权威，2026-11-26——NVTX range 投影 GPU timeline 逐字、cudaProfilerStart/Stop（cuda_profiler_api.h）、CUDA API trace 默认跳过非关键 + --cuda-trace-all-apis overhead 注、CUDA graph trace 三档（node overhead 注）、--gpu-metrics-frequency 默认 10000 Hz、--cuda-flush-interval buffer 落盘开销逐字（§2/§7）
- [NVIDIA, Nsight Compute](../references.md)：Nsight Compute ProfilingGuide（key: nsight-compute），[C] 官方规格披露·一手权威，2026-11-26——SOL「achieved percentage … theoretical maximum」逐字、SpeedOfLight_Hierarchical{Double,Half,Single,Tensor}RooflineChart、roofline 点到 roof 距离逐字 + baseline 追踪逐字 + 层级 ceiling 逐字、Kernel/Application/Range Replay、Workload Durations 四口径差逐字（clock/cache/serialization/duration）、稳态 generally > 20 µs、异步引擎干扰、热节流 override、MIG 锁频失效、Memory Chart §2.10（§3/§6/§7）
- [NVIDIA, nvbandwidth](../references.md)：NVIDIA/nvbandwidth repo（key: nvbandwidth），教学参照 + 官方实现，2026-11-26 重核——CE/SM 两类、spin kernel + events、3 采样中位数、D2D ×2 与双向 per-direction 口径（recap 2.4 §6）、DCGM diag r3 内嵌 plugin 互证（§8）
- [NVIDIA, nccl-tests](../references.md)：NVIDIA/nccl-tests repo（key: nccl-tests），教学参照 + 官方实现，2026-11-26 重核——algbw = S/t、busbw 因子 AllReduce 2(n-1)/n / RS/AG/AlltoAll (n-1)/n / Bcast/Reduce 1、「hardware bottleneck: NVLink, PCI, QPI, or network」逐字（recap 10.2 §8）、all_reduce_perf 命令（§8）
- [NVIDIA, DCGM](../references.md)：NVIDIA/DCGM repo + docs（key: dcgm），教学参照 + 官方实现，2026-11-26——Field Value Watches / Active Health Checks / Policy Notification、diag r1–r4 成员与时长逐字表、r3 内嵌 nvbandwidth/nccl_tests、field-ids 页 PROF 族语义转录（1001–1012）+ 设备 field（100/140/150/156）+ GPM_SUPPORT 94、名称版本漂移注（§9）
- [NVIDIA, datasheet](../references.md)：NVIDIA H100 官方 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威，2026-11-25——H100 SXM BF16 dense 989.5 TFLOPS / 3.35 TB/s（§1 主口径，recap 2.4 同表）
- [NVIDIA, Hopper whitepaper](../references.md)：NVIDIA H100 Tensor Core GPU Architecture Whitepaper V1.04（key: hopper-whitepaper），[C] 官方规格披露·一手权威，2026-11-25——HBM 乘法链 3352 GB/s / 132 SM（§1 口径表，recap 2.4 §3）
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——V100 transpose 12.8 → 140.2 → 199.4 GB/s 与 tiling 119.9 → 144.4 → 195.5 GB/s 三级锚（§5 表，recap 4.2/4.3）、stride-2 效率 50%（§5 表）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1 / 351f33fd——SM90 per-link 20.6 GB/s 模型 → 18 × 20.6 = 370.8 GB/s（§1 通信口径 / §8 对账，recap 7.2 式 (2)）
- [NVIDIA, NCCL 2.4 blog](../references.md)：NCCL 2.4 double binary tree blog（key: nccl-tree-blog），[C] 开放技术博客，2026-11-25——Summit 24,576 GPUs 双树对 ring 延迟最高 180× 改善（§8 已发表锚，recap 10.2 §8）
- [Luo et al., 2024](../references.md)：Benchmarking and Dissecting the Nvidia Hopper GPU Architecture（key: hopper-microbench），arXiv:2402.13499，[B] 有 paper 无官方代码，2026-11-25——HBM 478.8 cycles 延迟实测口径（§4 $L_{\mathrm{req}}$ recap 1.4 的输入，非 H100 官方规格）
