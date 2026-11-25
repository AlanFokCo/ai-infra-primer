# 2.4 KV Offload 与分层存储（KV Offload and Tiered Storage）

> KV cache 超出 HBM 容量时，把它挪到 DRAM 或 SSD——什么时候取回比重算便宜、传输需要多少带宽、传输怎么藏进计算里？ | 前置：1.2（prefill FLOPs 与 $\kappa$ ）、1.3（datasheet 参数表）、2.1（块管理）、2.2（prefix caching 与 TTL）、2.3（KV 量化格式账）、models 2.4（式 (9)） | 锚点：Mooncake（arXiv:2407.00079）[A]、LMCache / 3FS（教学参照 + 官方实现）、vLLM / SGLang repo [A]、DeepInfer doc（教学参照 + 官方实现，生产数字为官方披露）、NVIDIA datasheet [C]（官方规格披露、非闭源）

## 1. 三层存储账：带宽、延迟、容量

从 host DRAM 取 KV 比读 HBM 慢多少？从 SSD 取又慢多少？先把三层账锁定——本章后面所有决策都是这张表的代入。所谓 KV offload，是指把 KV cache 从 GPU HBM 挪到更低层存储、需要时再取回的系统手段；所谓分层存储（tiered storage），是指按带宽 / 延迟 / 容量把存储介质排成层级、按访问热度放置数据的结构。

**本章口径**：主口径 = Llama-3.1-8B GQA（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16，与 2.3 章脊一致），硬件 H100 SXM（80 GB HBM3、3.35 TB/s、PCIe Gen5，[NVIDIA, 官方 datasheet](../references.md)，[C] 官方规格披露）；生产披露口径 = DeepInfer V4.1-Flash（40 层 MLA）+ 专用 3FS 集群，换口径原因：生产数字只存在于该口径。KV 容量 recap models 2.4 式 (9)（models 的 $b$ = 每元素字节数，本维度译为 $\mathrm{bytes}$ ；本章 $b$ 专归 block size）：章脊 $\kappa L$ = 131072 B/token = 128 KiB， $T$ = 32768 时单序列 4 GiB；V4.1 口径 $\kappa L$ = 40960 B/token = 40 KiB（bf16，512 元素 = 448 latent + 64 rope，几何见 1.2 §3），同 $T$ 单序列 1.25 GiB。

三层账如下。**延迟列是 GPU 侧访问口径**（含 PCIe / 网络 / 文件系统栈），不是介质本征延迟：

| 层 | 带宽（GPU 侧） | 延迟（GPU 侧，量级） | 容量（量级） | 出处 |
| --- | --- | --- | --- | --- |
| HBM | 3.35 TB/s（H100 SXM）/ 7.7 TB/s（B200） | 百 ns | 百 GB（80 / 180 GB） | NVIDIA datasheet [C] |
| host DRAM | 介质百 GB/s 量级；GPU 侧受 PCIe Gen5 单向 ~64 GB/s 封顶 | μs 量级（PCIe 往返 + 拷贝栈） | TB 量级（服务器配置） | datasheet + 教学量级 |
| SSD（3FS 集群） | 单服务器 ≥40 GB/s 读 / 5 GB/s 写；单盘 GB/s 量级 | ms 量级（网络 + 文件系统栈） | 20 × 30.73 TB/服务器；集群 ~100 PiB | DeepInfer doc 官方披露 |

口径注（两条，不混用）：其一，介质本征延迟另有一笔账——DRAM ~50-100 ns、NVMe SSD ~10-100 μs（教学量级口径），GPU 侧延迟在其上叠加 PCIe / 网络 / 软件栈，两列数字不可互相代入。其二，DRAM 介质带宽是百 GB/s 量级（DDR5 多通道服务器口径），但 GPU 访问 host DRAM 必须过 PCIe，**本章 DRAM 层带宽一律取 PCIe 单向 ~64 GB/s**：datasheet 的「PCIe Gen5 128 GB/s」是双向合计，D2H 或 H2D 单方向拷贝取半（Gen5 x16：32 GT/s × 16 lane ÷ 8 = 64 GB/s，128b/130b 编码再折约 1.5%，教学口径）。SSD 层的 40 GB/s 读 / 5 GB/s 写是 DeepInfer 存储服务器官方披露（每服务器 20 × 30.73 TB SSD、保留 24h 数据，核实 2026-11-23）；读写不对称 8×，写路径的账在 §4 的 write-back。

```
              带宽（GPU 侧）        延迟（GPU 侧）     容量
  ┌─────────┐
  │  HBM    │  TB/s 量级            百 ns             百 GB
  └────┬────┘
       │  PCIe Gen5 单向 ~64 GB/s（D2H / H2D）
  ┌────┴────┐
  │  host   │  介质百 GB/s 量级     μs 量级           TB 量级
  │  DRAM   │  （GPU 侧受 PCIe 封顶）
  └────┬────┘
       │  RDMA 网络 + 3FS 栈（单服务器 ≥40 GB/s 读）
  ┌────┴────┐
  │  SSD    │  GB/s 量级            ms 量级           百 TB 量级
  │ (3FS)   │
  └─────────┘
```

DeepInfer 的放置策略不是按热度统一的 L1/L2/L3，而是**按 KV 类型分列**（官方披露，doc 三层表）：in-flight block KV + state KV → 推理 GPU HBM；SWA KV → 本地 host memory（单副本、不上 3FS，跨机命中走 400G VLAN 拉取）；paged block KV → 专用 3FS 集群（无应用层副本，靠 3FS 自身冗余）。逻辑容量 ~14.4 TB/GPU（物理更大；口径：逻辑 = 扣除冗余与预留后可用于 KV 的容量）。集群规模 ~100 PiB SSD（1 PiB = $2^{50}$ bytes，约合 $1.13 \times 10^{17}$ bytes）。SWA KV 为何不进持久缓存，§8 的 StateBlock 给账。

为什么 offload 是必选项而非可选项？容量账：章脊 8B 满窗（ $T$ = 131072）单序列 16 GiB，权重 bf16 约 16 GB，H100 的 80 GB 装一条满窗请求就去了近半； $B$ = 32 并发满窗 cache 达 512 GiB（约合 550 GB），是单卡 HBM 80 GB 的 6.9×——不挪出去，batch 根本凑不起来（models 2.4 §5 的账，此处只代入）。offload 把容量约束从 HBM 解绑到 DRAM / SSD，batch 上限随之从「HBM 装得下几条」变成「带宽取回来得及几条」——后者正是 §2 的不等式。

## 2. 分层命中成本账：什么时候取比重算便宜

一个 $T$ = 32768 的前缀命中请求：取回 4 GiB KV，还是重算 32768 token 的 prefill？答案是带宽不等式，不是容量不等式——容量决定放不放得下，带宽决定取回值不值（口径：「便宜」指取回墙钟时间小于重算墙钟时间，不是货币成本）。

设前缀 $T$ token，KV 字节 = $\kappa L T$ 。各层命中率（口径：命中 = 前缀 KV 已驻留该层、无需重算）： $h_1$ = HBM 驻留（即 2.2 的 prefix cache 命中）、 $h_2$ = host DRAM、 $h_3$ = SSD，全 miss 概率 $h_0 = 1 - h_1 - h_2 - h_3$ 。取回时间 $`t_i = \kappa L T / \mathrm{BW}_i + \ell_i`$ （带宽项 + 固定延迟项， $\ell_i$ 取 §1 表该层 GPU 侧延迟量级），HBM 命中无传输、 $t_1 = 0$ 。期望成本：

$$
E[t_{\mathrm{KV}}] = h_1 t_1 + h_2 t_2 + h_3 t_3 + h_0 \thinspace t_{\mathrm{recompute}} \qquad (1)
$$

符号表： $E[t_{\mathrm{KV}}]$ = 「拿到前缀 KV」这一步的期望墙钟时间（口径：不含 suffix prefill、排队与首 token 计算，TTFT 全账在 1.1 / 9.1）； $t_{\mathrm{recompute}}$ = 全 miss 时重算前缀的 prefill 时间。重算账 recap 1.2 式 (1)：prefill FLOPs $\approx 2 N_{\mathrm{act}} T + 2 L T^{2} d_{\mathrm{model}}$ ——二次项系数 2 **已含 causal 减半**（平均可见长 $T/2$ ；不打 causal mask 的双向场景系数才是 4，见 1.2 §2 口径注 / models 2.4 §6），不得再取半；除以有效算力 $F_{\mathrm{eff}}$ 。

代入章脊（ $T$ = 32768， $F_{\mathrm{eff}}$ = 50% × 989.5 TFLOPS = 494.75 TFLOPS，教学假设口径）：KV = 4.295 GB； $t_2$ = 4.295/64 ≈ 67 ms（PCIe）， $t_3$ = 4.295/40 ≈ 107 ms（单存储服务器读），400 Gbps 网络（= 50 GB/s）≈ 86 ms。重算：权重项 $2 \times 8 \times 10^{9} \times 32768$ = 5.24 × $10^{14}$ FLOPs（ $N$ = 8 × $10^{9}$ 为圆整数，1.2 用 matmul 口径 $N$ = 7.50 × $10^{9}$ ，差约 7%，本章例题统一用圆整数）；attention 项 $2 L T^{2} d_{\mathrm{model}}$ = 2 × 32 × $32768^{2}$ × 4096 = 2.81 × $10^{14}$ （已含 causal 口径）；合计 8.06 × $10^{14}$ ， $t_{\mathrm{recompute}}$ ≈ 1.63 s（ $F_{\mathrm{eff}}$ 取峰值 50% 是 prefill compute-bound 下的 MFU 教学假设，实测按引擎 profile 代入）。**取回比重算便宜 15–24×**——三层里最慢的 SSD 层也便宜一个量级。

为什么便宜这么多？把两笔成本都写成 $T$ 的函数，比值的结构就出来了。

**命题 1**（临界带宽）：权重项主导 regime（attention 二次项可忽略），取回比重算便宜当且仅当该层有效带宽 $\mathrm{BW} > \mathrm{BW}^{\ast}$ ，其中

$$
\mathrm{BW}^{\ast} = \frac{\kappa L \thinspace F_{\mathrm{eff}}}{2 N_{\mathrm{act}}} \qquad (2)
$$

且 $\mathrm{BW}^{\ast}$ 与 $T$ 无关。**证明路线**：两笔成本各写成 $T$ 的函数、作比、约去 $T$ ——关键一步是 KV 字节与重算 FLOPs 都线性于 $T$ 。**证明**： $t_{\mathrm{fetch}} = \kappa L T / \mathrm{BW}$ ， $t_{\mathrm{recompute}} = 2 N_{\mathrm{act}} T / F_{\mathrm{eff}}$ ； $t_{\mathrm{fetch}} \lt t_{\mathrm{recompute}}$ 两边同除 $T$ （ $T > 0$ ）再整理，得 $\mathrm{BW} > \kappa L F_{\mathrm{eff}} / (2 N_{\mathrm{act}})$ ，与 $T$ 无关。证毕。

代入章脊： $\mathrm{BW}^{\ast}$ = 131072 × 4.9475 × $10^{14}$ / (1.6 × $10^{10}$) ≈ 4.05 GB/s。含义：PCIe 64 GB/s、400G 网络 50 GB/s、单存储服务器 40 GB/s 全部高于临界值一个量级——8B @ H100 从任何一层取回都比重算便宜；单块 NVMe SSD（GB/s 量级）在临界值附近，是「取回 vs 重算」的分界介质。两个边界（命题前提的失效方向）：其一，attention 二次项使重算成本超线性于 $T$ ，长 $T$ 下 $\mathrm{BW}^{\ast}$ 进一步下降——上例含 attention 项的实际临界值 = 4.295 GB / 1.63 s ≈ 2.64 GB/s，长上下文更偏向取回；其二，小 KV 时固定延迟项主导，不等式须加回延迟项，「与 $T$ 无关」不再成立。

> **提点**：「取回比重算便宜」是带宽不等式，不是容量不等式。容量账决定 offload 能不能做（放不下才挪），带宽账决定 offload 值不值（取回快于重算才赚）——两笔账独立，临界带宽与剩余容量无关。

数值例忽略固定延迟项 $\ell_i$ （大 KV、带宽项主导口径）；小 $T$ 时须加回——如 $T$ = 1024、KV = 128 MiB，SSD 层 ms 量级延迟与传输项（≈ 3.4 ms）同量级，式 (1) 的 $t_3$ 不能只算带宽项。

命中率加权实例（教学示例值，命中率本身是 workload 属性，2.2 的 TTL / 命中率账）： $h_1$ = 0.5、 $h_2$ = 0.2、 $h_3$ = 0.2、 $h_0$ = 0.1，代入式 (1)： $E[t_{\mathrm{KV}}]$ = 0.2 × 67 + 0.2 × 107 + 0.1 × 1630 ≈ 198 ms，对比全 miss 的 1630 ms 改善 8.2×。注意期望值被 miss 项主导：0.1 的 miss 概率贡献了 163 ms（约 82%）。主导条件可直接写出： $h_0 t_{\mathrm{recompute}} > h_2 t_2 + h_3 t_3$ ，当 $t_{\mathrm{recompute}}$ 比各层取回时间大一个量级以上时，只要 $h_0$ 不趋于零就成立——**提升命中率的边际收益集中在消灭 miss，而非加快已命中层的传输**。

最后一笔：重叠。TTFT 的增量不是 $t_{\mathrm{fetch}}$ 而是 $\max(0, t_{\mathrm{fetch}} - t_{\mathrm{overlap}})$ ，重叠窗口 = 排队等待 + 可并行的计算（逐层传输，§4）。DeepInfer 的读路径已与排队 / 计算重叠（官方披露）——所以更快介质对 TTFT 的边际收益有限，这笔账 §10 的 CXL 裁定直接复用。

## 3. 传输带宽需求：KV 字节除以 TTFT 预算

§2 回答了「取回值不值」，本节回答「取回要多快」——预算是 TTFT，不是容量。把 TTFT SLO 分给 KV 传输的份额记为 $t_{\mathrm{budget}}$ ，需求带宽：

$$
\mathrm{BW}_{\mathrm{req}} = \frac{\kappa L T}{t_{\mathrm{budget}}} \qquad (3)
$$

符号表： $`\mathrm{BW}_{\mathrm{req}}`$ = 传输预算内的最低有效带宽； $t_{\mathrm{budget}}$ = TTFT SLO × 传输份额（份额是调度决策，本章取 10% 作教学示例）。TTFT 的分解 = 排队 + KV 获取 + suffix prefill + 首 token 计算：命中前缀越长，suffix prefill 越短，可分给传输的份额越大——10% 是保守示例，不是上限。代入章脊：TTFT SLO 2 s、份额 10% → $t_{\mathrm{budget}}$ = 0.2 s， $`\mathrm{BW}_{\mathrm{req}}`$ = 4.295 GB / 0.2 s ≈ 21.5 GB/s。对照 §1 表：PCIe 64 GB/s 满足（余量 3×）、400G 网络 50 GB/s 满足、单存储服务器 40 GB/s 满足（余量 1.9×）、单盘 GB/s 量级不满足——**单盘不够就池化**：多盘并行 / 多网卡聚合（§5 Mooncake 的 transfer engine 正是这个实现基础）。

压缩是带宽需求的直接除数。CacheGen 压缩 3.5–4.3×（§6）→ KV 字节除以 3.5–4.3 → $`\mathrm{BW}_{\mathrm{req}}`$ 降到 5.0–6.1 GB/s，≥6.1 GB/s 级单盘（Gen4 旗舰 / Gen5）变得可行；4 GB/s 级单盘即便压到 4.3× 上限仍不达标（练习 4(c) 算所需压缩比）。跨 DC 一句伏笔：DeepInfer 跨 DC P/D 口径 ~400 Gbps / 8 GPU（官方披露），每 GPU 份额 50/8 = 6.25 GB/s——高于 8B 的临界带宽 4.05 GB/s 但余量有限，压缩使跨 DC 传输从容可行（详细账在 8.3，此处只登记量级）。

逐层流式传输把暴露时间再砍一刀：KV 按层传输（每层 $\kappa T$ 字节），第一层到达即可开算，暴露的传输时间从全量降为「首层 + 尾部」。代入章脊：每层 KV = 4096 × 32768 = 1.34 × $10^{8}$ bytes = 128 MiB，PCIe 传输 2.1 ms；每层 prefill 计算 ≈ 1.63 s / 32 ≈ 51 ms——传输完全藏进计算。CachedAttention 的 layer-wise pre-loading + asynchronous saving 是这条流水线的多轮对话形态（2.2 recap，推导不重复）。

## 4. Offload 的重叠与拷贝路径：copy engine、GDR 与逐层流水

同样 4 GiB KV，用 CUDA copy engine 搬与用 RDMA NIC 直写，差别不只是带宽，还有谁来付成本。DeepInfer 的生产选择是后者。

时机：swap-out（D2H）发生在驱逐 / 抢占（4.2 的 recompute vs swap 账）与 write-back（prefix cache 持久化，2.2）；swap-in（H2D）发生在 prefix 命中后的调度。write-back 是 best-effort：DeepInfer 写失败不上报引擎，读时严格校验保正确（官方披露）——**写路径可以丢，读路径是正确性边界**，丢了写只是命中率损失，不是正确性损失。写带宽也要算账：SSD 层写 5 GB/s、读 40 GB/s，不对称 8×；章脊每 token 写 $\kappa L$ = 128 KiB，单服务器写容量 ≈ 5 × $10^{9}$ / 131072 ≈ 3.8 × $10^{4}$ token/s，由多实例共享——write-back 必须批量、限速、可丢，否则写路径挤占读路径的带宽预算。

拷贝路径三条：(1) SM kernel 拷贝——vLLM `kv_offload/cpu/swap_blocks_triton.py` 的机制例证（Triton kernel 搬块），占 SM、与计算争资源；(2) CUDA copy engine（DMA）——不占 SM，但细粒度块拷贝性能差（下方坑注）；(3) GDR——所谓 GDR（GPUDirect RDMA），是指 RDMA NIC 绕过 host memory 与 CPU 拷贝栈、直接读写 GPU HBM 的 PCIe 直连数据路径（硬件协议层归 gpu 9.2）。DeepInfer swap-in 口径（官方披露，doc 逐字）：

> 「Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.」

> **坑**：用 CUDA copy engine 做 paged KV 的细粒度 swap-in——paged 布局下 KV 物理不连续，swap-in 是块粒度的非连续 gather（章脊 4 GiB KV、 $b$ = 64、每层每块 256 KiB，共 16384 次小拷贝），copy engine 的逐描述符开销在小拷贝上主导，吞吐远低于 PCIe 标称。触发条件：块级非连续拷贝；后果：swap-in 成为 TTFT 瓶颈。规避：GDR（NIC 直写 HBM、不占 SM），或把小块聚合成大拷贝（vLLM swap_blocks 的批处理机制，走读归 5.2）。

DeepInfer 的三段流水（官方披露，doc 重叠机制）：匹配 = Entrypoint CHECK 角色对 local kvcache-store 做 hash-chain 前缀匹配（metadata only，hit length 路由前算出、不可隐藏——2.2 recap）；取 KV = 独立进程异步 3FS → host shared memory；swap-in = 引擎进程内线程逐层 H2D 与 forward 重叠。三段各占一种资源，互不抢占：

```
时间 →
3FS 读      [L0][L1][L2][L3]...        独立进程：3FS → host shared memory（异步）
H2D (GDR)       [L0][L1][L2][L3]...    引擎线程：逐层传输，NIC 直写 HBM、不占 SM
forward             [L0][L1][L2][L3]...  SM：第 i 层计算与第 i+1 层传输重叠
```

存储读（网络 + SSD）、H2D（PCIe）、forward（SM + HBM）三条资源轴并行——§2 的重叠窗口 $t_{\mathrm{overlap}}$ 就是这条流水线的产物。vLLM 侧机制例证：`simple_kv_offload/cuda_mem_ops.py` + `copy_backend.py`（拷贝后端抽象），模块全量走读归 5.2 / 5.3（两层纪律）。

## 5. Mooncake：KVCache-centric 架构与 transfer engine

模拟场景至多 525% 吞吐提升、真实 workload 多服务 75% 请求——Mooncake 的摘要账（[Qin et al., 2024](../references.md)，arXiv:2407.00079，[A]）。架构赌注一句话：以 KVCache 而非 GPU 为调度中心。

Mooncake 是 Kimi 的生产 serving 平台：prefill / decode 集群分离，利用 GPU 集群**闲置的 CPU / DRAM / SSD** 做 disaggregated KVCache cache——KV 池化在集群级，而非 2.2 的实例内 prefix 复用；KVCache-centric scheduler 平衡有效吞吐与 SLO；过载场景用 prediction-based early rejection（预测将违反 SLO 的请求提前拒绝，保住已接请求的达标率）。资源账的关键在「闲置」二字：GPU 节点的 host DRAM / SSD 本已随机器配置存在，KV 池化用的是 stranded capacity，成本是机会成本而非新增硬件——这与 §8 DeepInfer 专用存储集群（新增硬件、换故障域与弹性）是两种不同的成本结构。

transfer engine 是传输底座（repo design doc 口径，docs/source/design/transfer-engine/）：零拷贝传输库，两个核心抽象——**Segment**：RAM Segment = 可远程读写的 DRAM/VRAM 地址空间，每进程一个，内部划 Buffer 注册 (GPUDirect) RDMA；NVMeof Segment = NVMe 文件偏移直传 DRAM/VRAM，零拷贝。**BatchTransfer**：异步 Read/Write，同步非连续空间，design doc 称之为异步且更灵活的 AllScatter/AllGather。接口为 TransferEngine 类（mooncake-transfer-engine/include/transfer_engine.h）；Transport 当前列表：Tcp / Rdma / Efa / NVMeoF / Nvlink / IntraNodeNvlink / Hip / Shm。

拓扑感知路径选择：每服务器生成 topology matrix 广播集群；NIC 按内存类型分 preferred / secondary 列表，正常走 preferred（本地 NUMA / 本地 PCIe switch 的 GDR 路径），故障时两列表皆可用；多网卡池化 + 重试；本地目标走 memcpy / cudaMemcpy。带宽账：池化聚合带宽 = 各 NIC 带宽之和——§3「单盘不够就池化」的实现落点。硬件 / 拓扑 / RDMA 协议细节归 gpu 7.2 / 9.1 / 9.2 / 10.1；PD 分离的编排与 goodput 账在 8.2 / 8.3。

## 6. LMCache 与 CacheGen：压缩换带宽

带宽是预算时（§3），压缩是直接除数：CacheGen 把 KV 尺寸压缩 3.5–4.3×、取回 + 处理总延迟降低 3.2–3.7×、质量影响可忽略（[Liu et al., 2023](../references.md)，arXiv:2310.07240，[A]，论文口径）。

CacheGen 机制：量化 + **逐层逐通道算术编码**——KV 量化成 delta tensor，lossless 算术编码器把它编成 bitstream，利用 KV 分布的局部性与层敏感度差异；压缩级随可用带宽自适应（带宽紧则提高压缩比）。所谓算术编码，是指把整条消息映射为 $[0, 1)$ 内一个区间、码长逼近源熵的无损编码（信息论口径；此处只用结论：分布越集中，码长越短）。delta tensor 分布集中 → 压缩率高；逐层逐通道独立编码 → 层间敏感度差异可以被分别对待。两级压缩的性质要分清：量化是有损的（2.3 的误差界账适用），算术编码是无损的——编码本身不引入任何额外误差，压缩率来自分布冗余而非丢弃信息。流式形态：编码后的 bitstream 边传边解，取回与解码重叠——「compression and streaming」的标题账。

代入章脊：4 GiB KV 压缩后 4.295/4.3 ≈ 1.0 GB 到 4.295/3.5 ≈ 1.23 GB；400 Gbps（50 GB/s）传输 20–25 ms；§3 的 $`\mathrm{BW}_{\mathrm{req}}`$ 从 21.5 GB/s 降到 5.0–6.1 GB/s。压缩的代价是编解码计算与量化误差（2.3 的误差界账适用）——带宽受限场景（跨 DC、单盘）赚，带宽富余场景（PCIe 本地）未必值得付编解码延迟。

LMCache 集成（[LMCache, LMCache repo](../references.md)，教学参照 + 官方实现，dev 分支，核实 2026-11-23）：CacheGen 压缩 kernels 在 repo（csrc/cuda/cachegen_kernels.cuh + lmcache/v1/platform/torch_ops/cachegen_kernels.py，另有 musa / sycl 变体）；**官方 doc（docs/source/mp/cachegen.rst）注明 multiprocess 模式 CacheGen 尚未实现（coming soon），原始实现维护于 LMCache/LMCache_inprocess（doc 自链）**——「LMCache 已全模式集成 CacheGen」的说法不成立，引用其集成能力时须区分 inprocess 与 multiprocess 两模式。CacheBlend 集成在 lmcache/v1/multiprocess/modules/blend/（跨文档 KV 融合，2.2 recap）；vLLM / SGLang connector 在 lmcache/integration/{vllm,sglang}/。版本锚定：dev 分支（main 分支不存在，research 核实口径）。

## 7. 3FS 存储集群：CRAQ、USRBIO 与 FUSE 瓶颈

FUSE 路径实测上限 ~400K 4KiB reads/s——换算带宽 400000 × 4096 ≈ 1.64 GB/s，距存储服务器 ≥40 GB/s 的读能力差 24×。3FS 的答案是 USRBIO：绕过 FUSE。

架构（[DeepSeek-AI, 3FS repo](../references.md)，教学参照 + 官方实现，核实 2026-11-23）：cluster manager / metadata service（无状态，FoundationDB）/ storage service / client（FUSE + native）四组件；RDMA 网络（IB / RoCE）；文件切 chunk 跨 replication chain 条带化。存储侧冗余是 **CRAQ 链式复制**：所谓 CRAQ，是指写沿链传播、全节点确认后提交（write-all），读可落任意节点（read-any）的链式复制变体——强一致，读压力分摊到全链。**开源 repo 的冗余机制只有 CRAQ，没有 erasure coding**（全 repo 无 EC 实现 / 文档，research 核实口径）——EC 是另一口径，§8 分离。CRAQ 的读写账：write-all 使写延迟沿链逐跳累积（链长 × 每跳延迟），read-any 使读吞吐随链长线性扩展（每个节点都可服务读）——读多写少的 KV workload 恰好贴合这个不对称；对照 §8 EC 的 any-10 读，两者都用冗余换读性能，但 CRAQ 付全副本容量、EC 付 20% 冗余。

USRBIO（User Space Ring Based IO，src/lib/api/UsrbIo.md）：应用直接向 FUSE 进程内的 3FS I/O 队列提交请求，绕过 FUSE 限制（如最大单 I/O 尺寸）。两个共享内存抽象：**Iov** = 用户与 FUSE 进程间的大段零拷贝区（IB 内存注册由 FUSE 进程管理，读入 Iov、写先入 Iov）；**Ior** = io_uring 式小共享环（io_depth 控制批处理，多环推荐多线程）。fd 须注册后方可用 USRBIO；fio engine 在 benchmarks/fio_usrbio/。

FUSE 瓶颈的机制（docs/design_notes.md）：kernel-user 数据拷贝耗内存带宽、增延迟；共享队列自旋锁 → 并发不扩展，实测 ~400K 4KiB reads/s 上限；Linux 5.x FUSE 不支持同文件并发写。

> **坑**：用 FUSE 路径喂 KV 读——400K 4KiB reads/s ≈ 1.64 GB/s 的天花板比 PCIe 单向 64 GB/s 低一个量级还多，现象是 SSD 与 NIC 都闲着、吞吐卡在文件系统层。规避：USRBIO（Iov 零拷贝 + Ior 批量提交），让瓶颈回到 NIC / SSD 本身。

## 8. 专用集群 vs 分层存储：EC、GC 与放置决策

开源 3FS repo 只有 CRAQ，DeepInfer 生产 kvcache-store 用 10+2 EC——两句话不矛盾，是两个口径，必须分开写。

**EC 口径分离**：DeepInfer doc（官方披露，L111-113，核实 2026-11-23）逐字：「10+2 erasure coding」「each EC group tolerates two simultaneous node failures」「A read issues requests to all 11 storage nodes in the group simultaneously; any 10 responses are enough to decode and return」。

> **坑**：把 10+2 EC 归给开源 3FS repo——repo 只有 CRAQ（§7），EC 是 DeepInfer 生产 kvcache-store 的官方披露，开源侧无实现可对照。引用时两口径分列，不得混写。

张力注（不自行修复 doc 数字）：10+2 蕴含 12 节点组（10 数据 + 2 校验），而 doc 字面说「all 11 storage nodes in the group」——12 节点组读 11 取 10 是可成立的 quorum 读策略，但两处表述存在内部张力。本书逐字引 doc + 标官方披露 + 注明张力，以 doc 原文为准。

EC 账（按 10+2 标准语义推导）：存储开销 12/10 = 1.2×（20% 冗余），可用分数 10/12 ≈ 83.3%——对照三副本 3× 开销，容量效率 2.5×。读尾延迟：any-10 意味着不等最慢的 1–2 个节点，尾延迟由第 10 快的响应决定（顺序统计量口径，教学推导）——EC 不仅省容量，还买尾延迟容忍。GC Worker 做 LRU 回收（每存储节点一实例，水位以下触发，doc L115 官方披露）；TTL recap 2.2（agent 72h / chatbot 7d 近乎无损，官方披露）。

**专用集群 vs 分层权衡**（关键推导，doc L175 官方披露）：DeepInfer 选专用 3FS 集群而非计算节点本地分层，三条理由——(1) 任何 prefill 机命中任何前缀（「Since any engine can hit any prefix, there is no prefix-affinity routing」），避免 prefix affinity 与负载均衡路由的冲突（2.2 recap）：KV 若留在计算节点，命中就把请求绑死在特定机器，负载均衡失效；(2) 存储与计算 autoscaling 解耦——缓存容量按命中率需求扩缩（TTL × 到达率 × 复用距离决定值得留多少 KV），计算按 QPS 扩缩，两条曲线形状不同：流量翻倍时计算要翻倍，缓存未必（复用距离不变则命中率不变），互不绑架；(3) 计算节点故障率比存储节点高一个量级以上（「more than an order of magnitude higher」）——KV 放计算节点，节点故障即缓存损失，专用存储集群隔离故障域。代价面：专用集群多一跳网络（§1 的 ms 量级延迟）与独立存储硬件。权衡 = 命中率收益 + 弹性收益 + 故障域收益 vs 网络延迟 + 硬件成本；DeepInfer 的生产选择是专用集群，前提是 §2 的带宽账成立（网络取回仍远快于重算）。

StateBlock（SWA state cache，doc L217-223 官方披露）：每序列一个固定 slot，state_size_to_disk = 20 层 × 134 token × 528 B = 1,415,040 B/seq（MXFP8 SWA KV 格式，仅 stored region 落盘口径），约合 1.35 MiB。528 B 的构成（推导口径，research 侧算术观察）：512 维共享 KV 向量（448 latent + 64 rope）× 1 字节 + 512/32 = 16 个 E8M0 scale 字节 = 528 B——与 MXFP8 per-32 格式一致（2.3 §4 式 (4)）。128 token 粒度存储、ring buffer 窗口外 +6 位置（copy 位置至多越界 5 token）。SWA KV 不进持久 cache、留生产机内存，Entrypoint 内存目录（逻辑统一、物理按 key 分区）记录持有机——这解释了 §1 放置表的 SWA 行：窗口态复用距离短、跨机命中走 400G VLAN 即够，不值得付持久化的写放大。

write-back 与版本（doc L195 / L199 官方披露）：写失败不上报引擎、读时严格校验（§4 已引）；版本元组 (deployment configuration, model weights, user group)——权重 / block 字节布局 / tokenizer / 改 KV 值的框架变更均使旧 cache 失效；生产未做过 cache 格式迁移。系统含义：**KV 量化格式变更（2.3）同样使旧 cache 失效**——版本元组是量化 / 驱逐策略变更的正确性边界，格式升级要么全量失效重建，要么做迁移（DeepInfer 从未做过后者）。

## 9. 引擎实现对照：vLLM、SGLang 与 DeepInfer 的三条路线

三个引擎三条 offload 路线：vLLM 通用分层框架，SGLang 三层 HiCache 加 DSA 专用 HiSparse，DeepInfer decode 不开 offload。分歧不是实现偏好，是 decode 是否被容量约束。

**vLLM**（[vLLM, vllm-project/vllm repo](../references.md)，[A]，main，核实 2026-11-23，两模块均在 vllm/v1/ 下）：`kv_offload/` = 通用 offload / 分层框架——base.py（「Core abstractions for KV cache offloading in vLLM v1」）、config / factory / file_mapper / pointer_utils；cpu/（manager、gpu_worker、swap_blocks_triton、host_register、shared_offload_region、policies/{lru,arc}）；tiering/（manager、async_lookup、backpressure、metrics + fs/ obj/ p2p/ kvcr/ 子后端，p2p = nixl 数据面 + zmq 控制面 + session）。`simple_kv_offload/` = SimpleCPUOffloadConnector 简化 CPU offload connector（manager.py docstring「Scheduler-side manager for SimpleCPUOffloadConnector」；worker / host_buffer / copy_backend / disk_backend / cuda_mem_ops / sizing / metadata / metrics）。分工：前者多后端分层框架，后者单一 CPU offload connector；模块全量走读归 5.2 / 5.3。vLLM 侧 hisparse/ 与 kv_hints/ 目录存在但语义未核实，留 5.3（W3），本章不展开。

**SGLang HiCache**（[SGLang, sgl-project/sglang repo](../references.md)，[A]，main，核实 2026-11-23）：L1 GPU HBM / L2 host DRAM / L3 distributed storage；**L1/L2 每实例私有，L3 是否跨实例共享取决于 backend 配置**（hicache_design.mdx tier sharing 表）；L3 backends：file / mooncake / hf3fs（3FS）/ nixl / aibrix / lmcache / eic / flexkv / seaweedfs 等（mem_cache/storage/）；HiRadixTree 元数据 + prefetch 策略 + write-back 策略 + zero-copy 传输：prefetch 决定何时把 L3 拉向 L2 / L1（调度前预取，对应 §4 的 swap-in 时机），write-back 决定何时持久化（对应 §4 的 write-back 时机）——两个策略就是 §2 命中率账的执行器。路径（python/sglang/srt/ 下）：mem_cache/hicache_storage.py（HiCacheStorage ABC）、hybrid_cache/hybrid_cache_controller.py、memory_pool_host.py、hicache_auto_size.py。

> **坑**：按旧资料找 HiRadixCache 类——当前 main 已不存在，重构为 UnifiedRadixCache（unified_radix_cache.py）+ hybrid_cache/（HiCache 用）。触发条件：版本漂移（2026-11-23 main 核实）；规避：按新路径锚定，2.2 的 radix 机制本身不受影响。

**SGLang HiSparse**（hisparse_guide.mdx 口径）：DSA 系模型 decode 侧 GPU 只留小 hot buffer（如 4K token），全量 KV 在 CPU pinned memory；CUDA kernel 按 attention score **on-demand top-k swap-in**（短序列 fast path / 长序列 hit detection → LRU reordering → miss handling）；eager backup 异步回写上一 token KV。前提三条：DSA 系模型（DeepSeek-V3.2、GLM-5.1、Kimi-K3.1 DSA 层、DeepSeek V4、MiniMax M3）+ PD 分离模式 + 仅 decode 实例。PD 集成 direct-to-host：prefill GPU 经 RDMA 直写 decode host pool（绕过 decode GPU，消除 KV 传输的瞬时显存尖峰）；DeepSeek V4 direct-to-host 只写 C4 KV，c4_indexer 与 C128 KV 仍 D2D。路径：python/sglang/srt/mem_cache/hisparse_memory_pool.py、python/sglang/srt/managers/hisparse_coordinator.py、python/sglang/kernels/ops/kvcache/hisparse.py（注意最后一个在 kernels/ 下、不在 srt/ 下）。机制对照：HiCache 是容量驱动分层（全量 KV 都可能换入），HiSparse 是 attention 驱动分层（只换入 top-k 位置）——后者传输量由 index_topk 决定、与 $T$ 无关（DSA 链路 recap models 4.2 §6，推导不重复）。

**DeepInfer 对照**（[DeepSeek-AI, DeepInfer doc](../references.md)，官方披露，doc L183）：decode 不开 KV offload、也不用 Hi-Sparse——「Decode is not bound by cache capacity」；但 decode 产生的 KV 仍写盘提命中率。账面原因：V4.1 MLA $\kappa$ 小（bf16 40 KiB/token，主 KV 低精度 11.25 KiB/token，2.3 §4 推导）+ Wide-EP decode 大显存池（1.2 recap）→ decode 不被容量约束；GQA 大 $\kappa$ 模型（章脊 128 KiB/token，3.2× 于 V4.1 bf16、11.4× 于其主 KV 低精度）则相反。结论：offload 与否取决于 $\kappa T B$ vs HBM 容量的账，不是引擎风尚——§2 的临界带宽回答「值不值」，本节的容量账回答「要不要」。决策链三步：先算容量账（ $\kappa T B$ 是否超 HBM，决定要不要挪），再算临界带宽（取回是否快于重算，决定挪了值不值），最后选介质与拷贝路径（PCIe / 网络 / GDR，决定怎么挪）——§1 至 §4 就是这条链的三步，引擎差异只在第三步的实现与第一步的阈值。

## 10. CXL L2 展望：为什么 DeepInfer 当前不用

CXL 附加 DRAM 作 HBM 与 SSD 之间的 L2 层——DeepInfer QA 的回答是收益有限、当前不用。三条理由全是账。

CXL（Compute Express Link）一句机制：字节寻址的远程内存池化，host 经 PCIe 物理层扩展 DRAM 容量（CXL 硬件协议细节非本书覆盖范围）。L2 = 插在 HBM 与 SSD 之间的 CXL DRAM 层，带宽 / 延迟介于两者之间。

DeepInfer QA（官方披露，doc L203-207）：(1) V4.1 KV 小（MLA + 低精度，§9 账），容量压力不在 DRAM 层；(2) 读已与排队 / 计算重叠（§2 / §4），更快介质的 TTFT 边际收益有限；(3) SSD 远比 DRAM 便宜（每字节成本口径），容量扩展优先 SSD。开放问题：CXL 减 SSD 写（延寿）——写密集 workload 的 SSD 寿命账，当前不用。教学结论：一层介质的收益 = 容量缓解 × 延迟边际收益 − 成本；KV 小且读被重叠藏住时两项都缩水，CXL 层不划算。介质选型跟着 workload 的账走，不跟着层级图的对称性走。

## 总结

| 论断 | 账 | 回指 |
| --- | --- | --- |
| 三层账：HBM TB/s / 百 ns / 百 GB；DRAM GPU 侧受 PCIe 单向 ~64 GB/s 封顶 / μs / TB；SSD 40 GB/s 读 / ms / 百 TB | 延迟列为 GPU 侧口径，介质本征另注 | §1 表 |
| $E[t_{\mathrm{KV}}] = \sum_i h_i t_i + h_0 t_{\mathrm{recompute}}$ ，期望被 miss 项主导 | h = (0.5, 0.2, 0.2, 0.1) → 198 ms vs 全 miss 1630 ms | 式 (1)、§2 |
| 临界带宽 $\mathrm{BW}^{\ast} = \kappa L F_{\mathrm{eff}} / (2 N_{\mathrm{act}})$ ，权重项 regime 与 $T$ 无关 | 8B @ H100 ≈ 4.05 GB/s；含 attention 项 2.64 GB/s | 式 (2)、命题 1、练习 3 |
| $`\mathrm{BW}_{\mathrm{req}} = \kappa L T / t_{\mathrm{budget}}`$ ，压缩是直接除数 | 4 GiB / 0.2 s ≈ 21.5 GB/s；CacheGen 后 5.0–6.1 GB/s | 式 (3)、§3 |
| GDR swap-in 不占 SM、饱和 PCIe；copy engine 细粒度块拷贝性能差 | 三段流水：存储读 / H2D / forward 各占一轴 | §4 官方披露 |
| Mooncake：KVCache-centric + Segment / BatchTransfer + 拓扑感知多 NIC 池化 | 模拟至多 525% 吞吐提升、真实多服务 75% 请求 | §5 |
| CacheGen 3.5–4.3× 压缩；LMCache multiprocess 模式尚未实现 | 4 GiB → 1.0–1.23 GB，400G 传输 20–25 ms | §6 |
| FUSE ~400K 4KiB reads/s ≈ 1.64 GB/s → USRBIO（Iov / Ior）绕过 | 距 40 GB/s 读能力差 24× | §7 |
| EC 10+2 = DeepInfer 生产披露，开源 repo 只有 CRAQ；「11 节点 any 10」张力注 | 开销 1.2×、可用 83.3%、any-10 买尾延迟 | §8 |
| 专用集群三理由：任何机命中任何前缀 / autoscaling 解耦 / 故障域隔离 | 计算节点故障率高一个量级以上 | §8 官方披露 |
| decode offload 与否取决于容量约束：DeepInfer 不 offload，SGLang HiSparse 按 top-k 换入 | V4.1 主 KV 11.25 KiB/token vs 8B 128 KiB/token | §9 |
| CXL L2 收益有限：KV 小 + 读已重叠 + SSD 更便宜 | 收益 = 容量缓解 × 延迟边际收益 − 成本 | §10 官方披露 |

## 练习

1.（计算 · [10]）StateBlock 账：验证 20 × 134 × 528 = 1,415,040 B 并换算 MiB；按 MXFP8 per-32 格式（512 维共享 KV 向量 = 448 latent + 64 rope、E8M0 scale）推导 528 B 的构成。校验：与 doc 披露 state_size_to_disk = 1,415,040 B 一致；528 = 512 + 512/32。
2.（计算 · [20]）临界带宽代入：8B 章脊、H100 bf16 峰值 989.5 TFLOPS、 $F_{\mathrm{eff}}$ = 50%。(a) $T$ = 32768 的重算时间（含 attention 二次项 $2 L T^{2} d_{\mathrm{model}}$ ，已含 causal 口径，1.2 式 (1)）；(b) $\mathrm{BW}^{\ast}$ = KV 字节 / $t_{\mathrm{recompute}}$ ；(c) 与式 (2) 权重项口径对照并解释差异方向。先写出 (b) 的数量级预测再计算。校验：(c) 权重项口径 ≈ 4.05 GB/s，(b) 应更低（attention 项使重算更贵）。
3.（推导 · [30]）命题 1 证明闭环：(a) 推导式 (2) 并证明权重项 regime 下 $\mathrm{BW}^{\ast}$ 与 $T$ 无关；(b) 给出命题前提失效的边界（attention 项主导 / 小 KV 延迟项主导）及 $\mathrm{BW}^{\ast}$ 随 $T$ 的变化方向；(c) 代入 $h_1$ = 0.5、 $h_2$ = 0.2、 $h_3$ = 0.2、 $h_0$ = 0.1 与 §2 数值算 $E[t_{\mathrm{KV}}]$ ，对比全 miss。校验：(a) $T$ = 1024 与 $T$ = 32768 的权重项 $\mathrm{BW}^{\ast}$ 相同；(c) ≈ 198 ms。
4.（计算 · [30]）压缩与跨 DC：4 GiB KV、CacheGen 压缩 3.5–4.3×、跨 DC 400 Gbps（= 50 GB/s）。(a) 压缩后传输时间区间；(b) 8 GPU 份额口径的每 GPU 带宽，是否高于 8B 临界带宽 4.05 GB/s；(c) TTFT 传输预算 0.2 s 时，单块 NVMe SSD（设 4 GB/s，教学口径）要达标所需的最小压缩比。先写出 (c) 的预测再计算。校验：(a) 20–25 ms；(b) 50/8 = 6.25 GB/s > 4.05 GB/s；(c) 需求带宽 21.5 GB/s 除以 4 GB/s。

## 参考

- [Qin et al., 2024](../references.md)：Mooncake，arXiv:2407.00079，[A]（key: mooncake）——KVCache-centric 分离架构、transfer engine（Segment / BatchTransfer / 拓扑感知）、525% / 75% 摘要口径
- [Liu et al., 2023](../references.md)：CacheGen，arXiv:2310.07240，[A]（key: cachegen）——量化 + 逐层逐通道算术编码、带宽自适应压缩级、3.5–4.3× / 3.2–3.7×
- [Gao et al., 2024](../references.md)：CachedAttention，arXiv:2403.19708，[B]（key: cachedattention，无官方 repo）——layer-wise pre-loading + asynchronous saving
- [Yao et al., 2024](../references.md)：CacheBlend，arXiv:2405.16444，[A]（key: cacheblend）——跨文档 KV 融合（LMCache blend 模块集成）
- [LMCache, LMCache repo](../references.md)：dev 分支（key: lmcache，教学参照 + 官方实现）——CacheGen kernels 路径、multiprocess 未实现限定（官方 doc）、CacheBlend / connector 路径
- [DeepSeek-AI, 3FS repo](../references.md)：key: 3fs，教学参照 + 官方实现——CRAQ、USRBIO（Iov / Ior）、FUSE ~400K 4KiB reads/s、开源侧无 EC
- [DeepSeek-AI, DeepInfer doc](../references.md)：doc/inference_deployment_qa.md（key: deepinfer-doc，教学参照 + 官方实现，官方披露，核实 2026-11-23）——三层放置、存储 sizing、EC 10+2 与「11 节点 any 10」、GDR swap-in、GC LRU、StateBlock、版本元组、CXL 裁定、decode 不 offload
- [vLLM, vllm-project/vllm repo](../references.md)：key: vllm-repo，[A]——kv_offload/ 与 simple_kv_offload/ 模块树（main，2026-11-23 核实）
- [SGLang, sgl-project/sglang repo](../references.md)：key: sglang-repo，[A]——HiCache L1/L2/L3 与 HiSparse 路径、UnifiedRadixCache 版本漂移（main，2026-11-23 核实）
- [NVIDIA, 官方 datasheet](../references.md)：key: nvidia-datasheet，[C]（官方规格披露、非闭源）——H100 SXM 80 GB / 3.35 TB/s、B200 180 GB / 7.7 TB/s、PCIe Gen5 128 GB/s 双向
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json（key: v41-flash-config，[W]，hf-mirror 传输核实）——40 层 MLA 几何（head_dim 512 含 rope 64，kv_lora_rank 448）
