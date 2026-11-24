# gpu 维度设计 Spec 对抗审查 · C 路（P7-P11 互连/NCCL/RDMA/fabric/系统集成）

- 日期：2026-11-24
- 审查对象：`docs/superpowers/specs/2026-11-24-gpu-dimension-design.md`（§5 Part 7-11、§6.3 条目 8.2/8.3/9.2/10.1/10.3、§2 seam、§4.3 符号）
- 审查者：fresh-context adversarial evaluator（route C），不继承写作上下文，未派子代理，全部工作自做
- 方法：独立数学重推导（8.2/8.3/9.2/10.1）+ 一手来源事实核查（curl/WebSearch；WebFetch 对 nvidia.com/docs.nvidia.com 被网络策略拦截，按 spec 风险 8 fallback 走 curl + contents API + WebSearch）+ 边界/指针核对（已发布 models/training/inference 文本逐句 grep）
- 输入对照：STYLE_GUIDE.md、training 4.1 §2.2（式 2/3）、training 5.2/5.4/5.5、inference 3.3/7.4/8.3、inference/references.md deepep 条、models 2.8/4.5/4.6/2.4/1.2/4.1/1.3 指针句
- 口径：⟨verify⟩ 延迟项不计 finding（留各波 research pass）；本档案只记 spec 已断言的事实/数学/指针错误与协议缺口

---

## 1. Findings 表

severity：S1 = 事实性致命；S2 = 实质问题（推导断裂/局部事实错误/约定冲突/协议缺口）；S3 = minor（措辞/格式/可选增补）。

| # | sev | 位置（spec 行） | 断言 | 证据（URL/引句） | 建议修正 |
|---|---|---|---|---|---|
| C-1 | S2 | §5 8.2 L363-364；§6.3 L478 | 命题「all-reduce 每 rank 至少发送 $(n-1)/n\cdot M$ 且接收 $(n-1)/n\cdot M$ → ring 达下界，带宽最优」。ring all-reduce 每 rank 发送/接收各 $2(n-1)/n\cdot M$（spec 自身 L363 recap），是所述下界的 **2 倍**，「ring 达下界」不成立；$(n-1)/n\cdot M$ 实为 **单阶段**（reduce-scatter / all-gather）紧下界。证明草图「交出自有份、收齐全体和」给出的是 $S_r\ge M$ 弱界，不足以闭合 | 独立重推导（§2.1）；Patarasuk-Yuan 2009 JPDC 69(2):117-124 DOI 10.1016/j.jpdc.2008.09.002（cs.fsu.edu/~xyuan/paper/09jpdc.pdf）；Rabenseifner 2004 LNCS 3036（fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf）：recursive halving/doubling allreduce = $2(p-1)/p\cdot M$/process | 命题改述为「all-reduce 总流量 $\ge 2(n-1)M$；负载均衡（最小化完成时间）下每 rank 发送 $\ge 2(n-1)/n\cdot M$；ring 达界」，用 chunk 归约树+广播树论证（§2.1 给出完整证明）；或显式按 reduce-scatter/all-gather 两阶段各 $(n-1)/n\cdot M$ 分解。$(n-1)/n\cdot M$ 标注为单阶段界，勿与 all-reduce 总量混 |
| C-2 | S2 | §5 8.3 L369-370；§6.3 L478 | 「大消息 ring 赢带宽利用率，临界 $M^{\ast}$（延迟项 = 带宽项差求解）」。理想 α-β 模型下 recursive halving/doubling 与 double binary tree 的带宽项与 ring **相同**（均 $2(n-1)/n\cdot M\beta$），带宽项差 = 0，$M^{\ast}$ 无解；training 4.1 §2.2 已交付「大消息两者带宽项相同，树形赢的只是延迟项」。交叉只能来自**实测有效带宽差** $\beta_{\mathrm{tree,eff}}>\beta_{\mathrm{ring,eff}}$，spec 未引入该机制 | Rabenseifner 2004（带宽最优 $2(p-1)/p\cdot M$、$2\log_2 p$ 步）；NCCL 2.4 blog（developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/，Jeaugey 2019）逐字「double binary trees that provide **full bandwidth** and logarithmic latency」；training 4.1 L75 | 显式引入实测有效带宽参数（NCCL tuning 表/nccl-tests：大消息 ring 实测带宽高于 tree），推导 $M^{\ast}=\frac{2(n-1-\log_2 n)\alpha}{2\frac{n-1}{n}(\beta_{\mathrm{tree,eff}}-\beta_{\mathrm{ring,eff}})}$（§2.2）；说明理想模型无交叉、交叉是工程实测现象，勿从纯 α-β 凭空解出 |
| C-3 | S2 | §5 10.3 L415-418 | 大纲以「normal 模式（NVLink 域内 dispatch + 机间 RDMA 两阶段）/ low-latency 模式（纯 RDMA hook、IBGDA）」为走读对象——这是 **V1 语义**，当前 main（V2.5）已全移除：统一 EPBuffer（hybrid/direct）、NCCL Gin 后端、NVSHMEM/IBGDA 删除。spec 无版本锚注，writer 照大纲走读 fresh tarball 会全部对不上；IBGDA ⟨verify⟩ 仅存于 V1 | DeepEP README（current main，api.github.com/repos/deepseek-ai/DeepEP/contents/README.md）V2.5 release notes 逐字「Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation. NVSHMEM is no longer a dependency」「High-throughput and low-latency APIs unified into a single EPBuffer interface」；tags API 最新 tag v1.2.1；inference/references.md L104 deepep 条已记「spec 大纲 normal/low-latency 为 V1 语义，锚 tag v1.2.1」；inference 3.3 §6 坑注 | 继承 inference 3.3 双锚口径：机制讲解锚 tag v1.2.1（normal 两阶段/low-latency 纯 RDMA），接口走读锚 current main V2.5（EPBuffer/do_cpu_sync/lb_prefetch_weights/lb_reduce_grads/EngramBuffer/PPBuffer）；IBGDA 标 V1-only；fresh tarball 双锚（main + v1.2.1）file:line |
| C-4 | S2 | §2 L37；§4.2 L112；§5 10.4 L421/423/424 | 「CP 账（training 5.5）」「training 5.1/5.4/5.5 … TP/EP/CP 通信账」「training 5.5 CP 账 recap」。CP（Ring/Ulysses 通信账）在 **training 5.2**（SP/CP）；training 5.5 是组合策略与总账。errata 通道与 10.4 recap 会写入错误章号 | training/05-model-parallelism/02-sequence-and-context-parallelism.md 首行「# 5.2 SP / CP：序列维的切分」，§3 Ring Attention、§5 Ulysses、§6 通信账总表；05-combining-strategies.md 首行「# 5.5 组合策略与总账」 | 全部 CP 指针 5.5 → 5.2（L37/L112/L421/L423/L424）；10.4 recap 对象改「training 5.2 CP 账」 |
| C-5 | S2 | §2 L37 | 「PP 气泡（training 5.2）」。PP 在 **training 5.3**；5.2 是 SP/CP | training/05-model-parallelism/03-pipeline-parallelism.md 首行「# 5.3 PP：流水线气泡与 Schedule」 | PP 指针 5.2 → 5.3 |
| C-6 | S2 | §5 11.3 L441/443/444 | 「inference 8.5 / training 7.3 弹性容错 recap」「training 7.3 checkpoint」。checkpoint 与续训在 **training 7.4**；7.3 是训练稳定性（Loss Spike） | training/07-pretraining/04-checkpointing-and-resumption.md 首行「# 7.4 Checkpoint 与续训」；03-training-stability.md 首行「# 7.3 训练稳定性」；training 5.5 L177「checkpoint 的并行度无关保存与弹性续训，归 7.4 讲」 | checkpoint/弹性指针 7.3 → 7.4（L441/443/444） |
| C-7 | S2 | §5 8.1 L358 | 「all-to-all $(n-1)/n\cdot M$ **每 peer**」。all-to-all 每 peer 发送 $M/n$；$(n-1)/n\cdot M$ 是**每 rank 总发送**（10.3 L416 正确写「每 rank」）。per-peer 与 per-rank 混，量级差 $(n-1)$ 倍 | 独立重推导（§2.3）；spec 内部矛盾：10.3 L416「a2a 每 rank 消息量 ($(n-1)/n\cdot M$，recap 8.1)」 | 改「all-to-all 每 rank 总发送 $(n-1)/n\cdot M$（每 peer $M/n$，不含自环）」 |
| C-8 | S2 | §5 7.1 L337 | 「H100 18 links × 25 GB/s **双向** = 450 GB/s 每 GPU」。25 GB/s 是**单向**（per-direction），每 link 双向 50 GB/s；官方 H100 NVLink = **900 GB/s 双向**（18×50）。450 GB/s 是单向聚合，被误标「双向」，与官方头条差 2×；与 §4.3「PCIe gen5 x16 = 64 GB/s **单向**」（正确标注）口径不一致 | NVIDIA H100 产品页（nvidia.com/en-us/data-center/h100/，curl）逐字「NVLink: 900GB/s」「NVLink, which offers 900 gigabytes per second (GB/s) of GPU-to-GPU interconnect」；Hopper blog（developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/）「NVLink with 900 GB/sec total bandwidth … 7x the bandwidth of PCIe Gen 5」（900/128≈7，证 900 为双向） | 改「18 links × 25 GB/s 单向 = 450 GB/s 单向（= 900 GB/s 双向）」或直接用官方 900 GB/s 双向头条；§4.3 互连行补 NVLink 单向/双向口径注。注：本项有 ⟨verify datasheet⟩，数值留 W4，但**方向约定**是已断言错误，datasheet 核数不会自动纠正「25 GB/s 双向」标注 |
| C-9 | S3 | §5 9.2 L389 | 「NIC Gen5 x16 = 64 GB/s 单向 vs HBM 3.35 TB/s → GDR 带宽受 PCIe 限」。min(BW_NIC, BW_PCIe) 公式（L390）正确，但单口 400G ConnectX-7 = 50 GB/s < PCIe 64 GB/s，**NIC 线速**先卡住；PCIe 限仅在双口 400G/800G NIC（聚合 >64）时成立。prose 把 PCIe 当唯一瓶颈易误导读者以为 GDR 达 64 GB/s | ConnectX-7 datasheet（nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf）：400 Gb/s NDR = 50 GB/s，PCIe Gen5 x16 ≈ 64 GB/s（~63 usable），「64 > 50 → PCIe not a bottleneck」 | prose 补 regime：单口 400G NIC（50 GB/s）NIC 限；双口/800G（>64 GB/s）PCIe 限；可选增补 GDR 读/写不对称（write posted 快于 read non-posted） |
| C-10 | S3 | §5 10.4 L423 | Ring Attention 论文标 **[B]**，但 training/references.md L114 同论文（arXiv:2310.01889）已核 **[A]**（2026-11-17）。跨维度证据等级不一致 | training/references.md L114「ring-attention … arXiv:2310.01889 … [A] … 2026-11-17 … 5.2」 | W5 research pass 核官方 repo 存在性后与 training 对齐（[A] 或注明 [B] 理由）；spec §4.1 已声明沿兄弟维度已核实记录，宜直接沿 training [A] |
| C-11 | S3 | §2 L44-52 预盘点 | 「重点重判句（预盘点）」8 条全为 models/training，**无 inference 条目**；但 inference 有多处通信→cluster 指针须按新 seam 重判到 gpu P9/P10：inference 8.3 §5 L194 GDR「硬件协议层归 cluster 维度」、inference 3.3 L7「RDMA / NVLink 硬件与拓扑归 cluster 维度」、inference/references.md L104 deepep「硬件归 cluster」。W0 全量 sweep 会覆盖，但预盘点漏列 route-C 直接相关目标 | inference/08-distributed-and-pd/03-kv-transfer.md L194；inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md L7；inference/references.md L104 | 预盘点补 inference 三条：8.3 GDR → gpu 9.2；3.3 RDMA/NVLink 硬件拓扑 → gpu P7/P9/P10；refs deepep 硬件 → gpu 10.3 |
| C-12 | S3 | §5 10.3 L415 | 「EP > 32 递减 recap inference 7.4」。EP>32 拐点账**首交付**在 inference 3.3 §5（命题 3，$e^{\ast}=A/F$）；7.4 是 recap。10.3 内容行已引 3.3，此处单独挂 7.4 略偏 | inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md §5 标题「Fused MoE：Mega MoE 与 EP > 32 的拐点账」；inference 7.4 §2 recap 3.3 命题 3 | 改「EP > 32 递减 recap inference 3.3 §5 命题 3（7.4 兑现）」；非阻塞 |

---

## 2. 独立数学重推导（强制清单 8.2/8.3/9.2/10.1）

### 2.1 【8.2】ring all-reduce 带宽最优性下界（C-1 闭合证明）

**设定**：$n$ rank，各持 $M$ bytes 向量 $x_r$（元素取自大域、不可压缩），all-reduce 要求每 rank 输出 $y=\sum_r x_r$（逐元素）。$\beta$ = 每字节传输时间。

**单阶段紧下界（$(n-1)/n\cdot M$ 的正确归属）**：
- reduce-scatter：每 rank 终持一个 chunk（$M/n$）的全和。rank $r$ 对**非己所有**的 $n-1$ 个 chunk 的自有贡献必须离开 $r$（每 chunk $\ge M/n$）→ $S_r\ge (n-1)/n\cdot M$。ring reduce-scatter 每 rank 发 $(n-1)\cdot M/n$，**达界**。
- all-gather：每 rank 始持一个 chunk，须把自己的 chunk 送达其余 $n-1$ rank → $S_r\ge (n-1)/n\cdot M$。ring all-gather 达界。
- 故 $(n-1)/n\cdot M$ 是**单阶段**每方向紧下界，ring 逐阶段达界。

**all-reduce 总流量下界（ring 达界的正确命题）**：把向量切成 $n$ 个 chunk（各 $M/n$）。对 chunk $c$：
1. 设 $r^{\ast}$ 为**首个**持有 chunk $c$ 全和的 rank。在此之前，$n$ 份贡献须经归约树汇聚到 $r^{\ast}$：$r^{\ast}$ 须接收 $\ge (n-1)\cdot M/n$ 的 chunk-$c$ 部分和数据（归约树边携带**未全**部分和）。
2. 其余 $n-1$ 个 rank 各须获得 chunk $c$ 全和：要么接收全和（$\ge M/n$），要么本地独立归约（须接收全部 $n$ 份贡献 $\ge (n-1)M/n$，更多）。无论哪种，每 rank $\ge M/n$ 的 chunk-$c$ 全和信息流入（广播树边携带**已全**和）。
3. 归约边（未全）与广播边（已全）是**不相交**的传输集合 → chunk $c$ 总流量 $\ge 2(n-1)\cdot M/n$。

对 $n$ 个 chunk 求和：**总流量 $\ge 2(n-1)M$**。每 rank 平均 $\ge 2(n-1)/n\cdot M$；完成时间 $T\ge \max_r S_r\cdot\beta \ge \text{平均}\cdot\beta = 2(n-1)/n\cdot M\beta$。ring all-reduce 每 rank 发送/接收各 $2(n-1)/n\cdot M$、全双工流水、负载均衡，$T_{\mathrm{ring}}$ 带宽项 $=2(n-1)/n\cdot M\beta$，**达界** → 带宽最优。$\blacksquare$

**信息论弱界（spec 草图实际给出的）**：固定 rank $r$，其余 rank 输出 $y[c]$ 须含 $x_r[c]$ 信息（每 chunk $M/n$），该信息只能源自 $r$ 的发送 → $S_r\ge M$；对称 $R_r\ge M$。此界 $M < 2(n-1)/n\cdot M$（$n>2$），**弱于**总流量界，不能证 ring 最优。spec「交出自有份、收齐全体和」即此弱界，不足以闭合「ring 达 $(n-1)/n\cdot M$ 界」。

**反例校验（证「每 rank 发送 $\ge 2(n-1)/n\cdot M$」非普适）**：reduce-broadcast（星型）叶 rank 只发 $M < 2(n-1)/n\cdot M$（$n>2$）；但 root 发 $(n-1)M$，$\max_r S_r$ 仍 $\ge 2(n-1)/n\cdot M$，完成时间由 root 主导 ≫ ring。故紧命题须落在**总流量/均衡/max-rank/时间**口径，而非「每 rank 普适发送 $\ge 2(n-1)/n\cdot M$」。

**结论**：spec 命题把单阶段界 $(n-1)/n\cdot M$ 误挂 all-reduce 并断言 ring 达之，差 2×（C-1）。修正命题与证明如上。recap 忠实性：training 4.1 式 (2) $C_{\mathrm{rank}}=2\frac{d-1}{d}M$、式 (3) $T_{\mathrm{ring}}=2(d-1)\alpha+2\frac{d-1}{d}M\beta$（含归纳证明）已交付，spec 8.1/8.2 以 $d\to n$ 翻译 recap、**不重推消息量** ✓ 纪律正确。

### 2.2 【8.3】tree 延迟与 ring/tree 交叉（C-2 闭合推导）

**延迟项**（power-of-2 $n$，Rabenseifner recursive halving/doubling）：reduce-scatter 递归折半 $\log_2 n$ 步 + all-gather 递归倍增 $\log_2 n$ 步 = $2\log_2 n$ 步 → $2\log_2 n\cdot\alpha$。ring $2(n-1)$ 步 → $2(n-1)\alpha$。$n=8$：tree $6\alpha$ vs ring $14\alpha$ ✓。

**带宽项**：recursive halving/doubling 每 rank 发送 $M(1/2+1/4+\cdots+1/n)\times 2 = 2(n-1)/n\cdot M$，**与 ring 同**（Rabenseifner 2004 证带宽最优）。double binary tree（NCCL 2.4）每 rank ~$2M$、blog 称「full bandwidth」。故理想 α-β 模型：

$$T_{\mathrm{tree}}=2\log_2 n\cdot\alpha+2\tfrac{n-1}{n}M\beta,\quad T_{\mathrm{ring}}=2(n-1)\alpha+2\tfrac{n-1}{n}M\beta$$

带宽项差 = 0 → tree 对**所有** $M$ 占优（同带宽、低延迟），**理想模型无交叉**。training 4.1 L75 已交付此结论。

**真实交叉（工程实测）**：NCCL 大消息选 ring，因 ring 实测有效带宽高于 tree（tree reduce/broadcast 阶段依赖致利用率降）。引入 $\beta_{\mathrm{tree,eff}}>\beta_{\mathrm{ring,eff}}$（实测参数），令 $T_{\mathrm{ring}}=T_{\mathrm{tree}}$：

$$2(n-1-\log_2 n)\alpha = 2\tfrac{n-1}{n}M(\beta_{\mathrm{tree,eff}}-\beta_{\mathrm{ring,eff}})\;\Rightarrow\; M^{\ast}=\frac{n(n-1-\log_2 n)\,\alpha}{(n-1)(\beta_{\mathrm{tree,eff}}-\beta_{\mathrm{ring,eff}})}$$

$M<M^{\ast}$ tree 赢（延迟主导），$M>M^{\ast}$ ring 赢（实测带宽主导）。**spec 须显式引入 $\beta_{\mathrm{eff}}$ 差并引 NCCL tuning/nccl-tests 数值**，否则「延迟项=带宽项差求解」无定义（C-2）。数值例（$n=8,\alpha=1\,\mu s,\beta_{\mathrm{ring}}=1/400\,\mathrm{GB/s},\beta_{\mathrm{tree}}=1/200\,\mathrm{GB/s}$）：$M^{\ast}=\frac{8(7-3)\times10^{-6}}{7(1/200-1/400)\times10^{9}}\approx 1.83$ MB（量级示意，实测参数留 W4）。

### 2.3 【8.1】原语消息量（C-7）与【9.2】GDR 带宽账

**原语每 rank 发送**（$M$ = 被传输总字节）：all-gather $(n-1)/n\cdot M$ ✓；reduce-scatter $(n-1)/n\cdot M$ ✓；all-reduce $2(n-1)/n\cdot M$ ✓；**all-to-all 每 rank 总发送 $(n-1)/n\cdot M$（每 peer $M/n$，不含自环）**——spec 8.1「$(n-1)/n\cdot M$ 每 peer」误（C-7），10.3「每 rank」正确。

**【9.2】GDR NIC→HBM**：$\mathrm{BW}_{\mathrm{GDR}}=\min(\mathrm{BW}_{\mathrm{NIC}},\mathrm{BW}_{\mathrm{PCIe}})$（公式正确）。PCIe Gen5 x16：$32\,\mathrm{GT/s}\times16\times\frac{128}{130}/8=63.0$ GB/s 单向（名义 64）✓。单口 ConnectX-7 400G NDR = 50 GB/s < 63 → **NIC 限**（$\min=50$）；双口 400G/800G（100 GB/s）> 63 → **PCIe 限**（$\min=63$）。多 NIC 聚合（rail-optimized 每 GPU 独立 NIC）：$k$ NIC 各走独立 x16 → $\min(k\cdot\mathrm{BW}_{\mathrm{NIC}},k\cdot\mathrm{BW}_{\mathrm{PCIe}})$，受 HBM/PCIe switch uplink 上限约束。HBM 3.35 TB/s ≫ PCIe，GDR 路径瓶颈在 PCIe/NIC 而非 HBM ✓。spec prose「受 PCIe 限」须补 NIC 限 regime（C-9）。BAR1 映射容量约束 ✓ 已列。

### 2.4 【10.1】fat-tree 收敛比与 rail-optimized（验证正确，无 finding）

**fat-tree（Al-Fares 2008）**：$k$-port 交换机，$k$ pod（每 pod $k/2$ edge + $k/2$ agg），core $(k/2)^2$ 台，总主机 $k^3/4$（论文逐字「a fat-tree built with k-port switches supports k³/4 hosts」✓）。edge 下行 $k/2$ host 口 = 上行 $k/2$ agg 口 → **收敛比 1:1 无阻塞** ✓。oversubscription $r$:1 = 上行 = 下行/$r$。

**rail-optimized**：每节点 GPU $i$ 接 leaf $i$（rail $i$），跨节点同 rail GPU 一跳直达。ring 对齐使 $g$ 个环各走独立 rail → 节点间聚合带宽 $=g\times\mathrm{BW}_{\mathrm{NIC}}$（8 GPU×8 NIC×50 GB/s = 400 GB/s/节点）✓「all-reduce 带宽 × NIC 数」成立（rail 对齐前提）。spec 10.1 推导正确。

---

## 3. Verification log（强制：抽查项 × 来源 × 结论 × 覆盖率）

WebFetch 对 nvidia.com/docs.nvidia.com 被拦截（「Unable to verify domain」），按 spec 风险 8 fallback：curl 一手 HTML/PDF + GitHub contents API + WebSearch（一手 URL 命中）。

| 抽查项 | 来源（URL/引句） | 结论 | 覆盖 |
|---|---|---|---|
| H100 NVLink 带宽/方向 | nvidia.com/en-us/data-center/h100/（curl）「NVLink: 900GB/s」「900 gigabytes per second … GPU-to-GPU」；Hopper blog「900 GB/sec total … 7x PCIe Gen 5」 | 900 GB/s **双向**，18 links×50 GB/s 双向（25 单向）；spec「25 GB/s 双向=450」误标 → C-8 | ✓ 一手 |
| H100 SM 数/NVSwitch | Hopper blog「132 SM」「NVSwitch … 64 ports of fourth-generation NVLink」「NVIDIA SHARP in-network reductions」 | 132 SM ✓、4 NVSwitch（HGX）✓、NVLS=NVLink SHARP ✓ | ✓ 一手 |
| B200 NVLink/GB200 NVL72 | WebSearch 命中 nvidia.com/en-us/data-center/gb200-nvl72/ + B200 datasheet：NVLink 5 = 1.8 TB/s 双向/GPU；NVL72 = 72 Blackwell GPU+36 Grace，130 TB/s 聚合（72×1.8=129.6） | spec 1.8 TB/s / 72 GPU 域 ✓；NVLink 6（Rubin）3.6 TB/s 勿混（spec 风险 1 已警） | ✓ 一手命中 |
| NCCL 算法/协议 | docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html（curl）NCCL_ALGO: Ring/Tree/CollnetChain/CollnetDirect/NVLS/NVLSTree/PAT；NCCL_PROTO: LL/LL128/Simple；「NVLS and NVLSTree enable NVLink SHARP offload」「enabling LL128 on platforms that don't support it can lead to data corruption」 | spec ring/tree/NVLS/CollNet + LL/LL128/Simple ✓；自动按拓扑+消息大小选择 ✓ | ✓ 一手 |
| double binary tree | developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/（curl，Jeaugey 2019-02-04）「double binary trees that provide full bandwidth and logarithmic latency」「introduced in MPI in 2009」 | full bandwidth + log latency ✓ → 理想模型与 ring 同带宽项，交叉须实测（C-2） | ✓ 一手 |
| ring 下界（8.2） | Patarasuk-Yuan 2009 JPDC 69(2):117-124 DOI 10.1016/j.jpdc.2008.09.002（cs.fsu.edu/~xyuan/paper/09jpdc.pdf，PDF 下载截断，WebSearch 摘要确认每方向 $(n-1)/n\cdot X$、ring 总 $2(n-1)/n\cdot X$）+ 独立重推导 §2.1 | 单阶段界 $(n-1)/n$、all-reduce 总流量界 $2(n-1)M$；spec 命题差 2× → C-1 | ✓ 一手命中+独立推导 |
| Rabenseifner（8.3） | fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf（WebSearch 命中）ICCS 2004 LNCS 3036 pp.1-9；recursive halving/doubling = $2(p-1)/p\cdot M$、$2\log_2 p$ 步 | 带宽项与 ring 同 ✓ → C-2；spec LNCS 3036 引用 ✓ | ✓ 一手命中 |
| GDR 内核锚 | api.github.com/repos/NVIDIA/open-gpu-kernel-modules/contents/kernel-open/nvidia-peermem → nvidia-peermem.c/nv-p2p.h/peer_mem.h；NVIDIA/peer-memory + Mellanox/peer-memory 均 404 | nvidia-peermem 路径 ✓、peer-memory 404 改锚 ✓ | ✓ 一手 API |
| repo 存在性 | api.github.com/repos/{NVIDIA/nccl, open-gpu-kernel-modules, gdrcopy, deepseek-ai/DeepEP, linux-rdma/rdma-core, nccl-tests, DCGM, nvbandwidth} 全 200 | spec 预核实 12 repo 200 ✓（抽查 8） | ✓ 8/8 |
| DeepEP 版本 | api.github.com/repos/deepseek-ai/DeepEP/contents/README.md（current main）V2.5「Fully remove V1 … NVSHMEM no longer a dependency」「unified into a single EPBuffer」「NCCL Gin backend」；tags 最新 v1.2.1 | V1 normal/low-latency/IBGDA 已移除 → C-3；inference refs L104 已记漂移 | ✓ 一手 API |
| ConnectX-7/PCIe | WebSearch 命中 nvidia.com ConnectX-7 datasheet：400 Gb/s NDR=50 GB/s，PCIe Gen5 x16≈64 GB/s（~63 usable），「64>50 → PCIe not a bottleneck」 | 单口 NIC 限 50 GB/s；spec「受 PCIe 限」prose 偏 → C-9；min 公式 ✓ | ✓ 一手命中 |
| MIG 7 实例 | WebSearch 命中 nvidia.com MIG 页 + MIG User Guide PDF（docs.nvidia.com/datacenter/tesla/pdf/MIG_User_Guide.pdf）：A100/H100 max 7；B100/B200/GB200 支持；A30 max 4 | spec「A100/H100/B200 … 最多 7 实例」✓；MIG User Guide PDF URL 可供 W6 | ✓ 一手命中 |
| XID 31/43/48/63/74/79 | WebSearch 命中 docs.nvidia.com/deploy/xid-errors/latest/analyzing-xid-catalog.html + A100 mem-error-mgmt：31=page fault、43=stopped processing（app）、48=DBE、63=row remap/page retirement 记录、74=NVLink error、79=fallen off bus | spec 六码语义 ✓；现行 URL /latest/（旧 /index.html 重定向）供 W6 | ✓ 一手命中 |
| DCQCN | WebSearch 命中 conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf：Zhu/Eran/Firestone/Guo … SIGCOMM 2015 pp.523-536；RP/CP/NP、ECN→CNP→rate decrease、无 CNP→rate increase | spec 9.3「ECN 标记+CNP 反馈+rate adjust」✓、[B] 无 arXiv ✓ | ✓ 一手命中 |
| Al-Fares fat-tree | WebSearch 命中 web.stanford.edu/class/cs244/papers/al-fares-sigcomm08.pdf：SIGCOMM 2008「k-port switches supports k³/4 hosts」、k pod、$(k/2)^2$ core | spec 10.1 ✓（⟨verify 精确出处 W5⟩ 可销）；$k^3/4$ 独立复核 ✓ | ✓ 一手命中 |
| GPUDirect RDMA docs | docs.nvidia.com/cuda/gpudirect-rdma/ → 200 | spec 预核实 200 ✓ | ✓ |
| NCCL docs / Blackwell WP | docs.nvidia.com/deeplearning/nccl/user-guide/docs/ → 200；resources.nvidia.com/en-us-blackwell-architecture → 200 | spec 预核实 ✓ | ✓ |
| SHARP docs URL | 猜测路径 docs.nvidia.com/networking/display/SHARPv300/SHARP+Overview → 404 | ⟨verify W5⟩ 延迟项，**不计 finding**；现行 URL 留 W5 research pass | 延迟 |
| training 4.1 式 2/3 recap | training/04-data-parallelism/01-dp-and-ddp.md L62/70：$C_{\mathrm{rank}}=2\frac{d-1}{d}M$ (2)、$T_{\mathrm{ring}}=2(d-1)\alpha+2\frac{d-1}{d}M\beta$ (3)，归纳证明 L55 | spec 8.1/8.2 $d\to n$ recap 忠实、不重推消息量 ✓ | ✓ 本地 |
| 指针句存在性 | models 2.8 §4 L367「硬件互联归 cluster」、4.5 §7 L327「all-to-all 带宽与拓扑归 cluster」、4.6 L227/287「通信实现归 cluster」、2.4 §6 L216/244 + 1.2 §10 L245「微架构来源与测量归 gpu」、4.1 L298/302/316「指令级 kernel 归 gpu」、1.3 L192「集群侧属 cluster」；training 4.1 L73/75「硬件侧/选择逻辑归 cluster」、5.4 §6 L95「kernel 实现归 gpu/cluster」 | spec §2 预盘点 8 条目标句**全部存在**且章节号 ✓（重判目标 gpu 章号合理） | ✓ 本地逐句 |
| training 章号 | training P5 首行：5.1 TP/5.2 SP-CP/5.3 PP/5.4 EP/5.5 组合；P7：7.3 稳定性/7.4 checkpoint | spec CP 5.5（应 5.2）/PP 5.2（应 5.3）/checkpoint 7.3（应 7.4）误 → C-4/5/6 | ✓ 本地 |
| inference recap 目标 | inference 3.3 §6 DeepEP 语义+版本坑注、§5 EP>32 拐点；7.4 EPLB/Wide-EP；8.3 §5 GDR 拷贝路径；1.4 §6 batch invariance；9.1 容量成本；2.1 paged；7.3 kernel 栈 | spec recap 目标存在 ✓；EP>32 首交付 3.3 §5（C-12 minor） | ✓ 本地 |
| Ring Attention 等级 | training/references.md L114 [A]（2026-11-17）vs spec 10.4 [B] | 跨维度等级不一致 → C-10 | ✓ 本地 |

**覆盖率**：route C 关键硬件事实（NVLink/NVSwitch/GB200、NCCL 算法协议、GDR 内核锚、ConnectX-7/PCIe、MIG、XID、DCQCN、fat-tree）全部一手或一手命中核查；4 个强制重推导章（8.2/8.3/9.2/10.1）独立重算；§2 预盘点 8 条指针句逐句本地核对；repo 存在性抽查 8/8。⟨verify⟩ 延迟项（SHARP/IBTA 精确 URL、B200 datasheet 逐项、NCCL 当前版本、DeepEP fresh tarball file:line、Ring Attention arXiv 再核）留各波 research pass，不计 finding。

---

## 4. 边界检查（gpu↔cluster 新 seam）

- **P7-P11 通信内容归 gpu** ✓：spec §2 seam「硬件机制与通信全栈（NVLink→NCCL→RDMA→机间 fabric→大规模通信工程）归 gpu；平台编排（K8S/容器/Sandbox/作业调度/device plugin）归 cluster」正确。P7-P11 无 K8S/容器/作业编排越界。
- **MIG/vGPU/SR-IOV 硬件分区归 gpu 11.1，device plugin/调度归 cluster** ✓：11.1「K8S device plugin / 调度 → cluster 指针（新 seam）」正确；11.3「集群级故障编排 → cluster 指针」✓。
- **10.2「集合调度与多作业干扰」** 框定为 fabric 带宽共享/拥塞交互（transport 级），非作业编排 → 归 gpu 合理，不触发越界 ✓。
- **已发布指针重判目标** ✓：training 4.1 NVLink/IB 硬件 → gpu 7.2/9.x、NCCL 选择 → gpu 8.4/10.2；models 2.8 TP 互联 → gpu P7/P9-P10；models 4.5 a2a → gpu 8.1/10.3 + 拓扑 10.1；models 4.6 Ring Attention → gpu 10.4；training 5.4 DeepEP → gpu 10.3。目标句全部存在、重判章号合理。
- **预盘点缺口**（C-11）：inference 通信→cluster 指针（8.3 GDR、3.3 RDMA/NVLink、refs deepep）未列入预盘点，W0 全量 sweep 须覆盖。
- **cluster/gpu 占位 README** 现为旧 scope（cluster=通信、gpu=kernel），spec W0 计划重写 ✓（非 finding，W0 待执行）。
- **10.3 DeepEP「CP 对照」**：DeepEP V2.5 README 已含 experimental CP/PP/DP 原语 → 10.4 Sources「DeepEP（CP 对照）」实际成立 ✓（非 finding）。

---

## 5. Summary counts

- **S1 = 0**
- **S2 = 8**（C-1 8.2 下界差 2×；C-2 8.3 交叉无定义；C-3 10.3 DeepEP V1/V2.5 版本锚缺；C-4 CP 5.5→5.2；C-5 PP 5.2→5.3；C-6 checkpoint 7.3→7.4；C-7 8.1 a2a per-peer；C-8 7.1 NVLink 方向误标）
- **S3 = 4**（C-9 9.2 GDR 瓶颈 prose；C-10 Ring Attention 等级；C-11 预盘点漏 inference 指针；C-12 EP>32 首交付归属）
- 合计 12 findings。
- 强制重推导：8.2/8.3 发现命题缺陷（C-1/C-2，附闭合证明）；9.2 公式正确 prose 偏（C-9）；10.1 验证正确无 finding。
- recap 纪律 ✓：8.1/8.2 忠实 recap training 4.1 式 2/3（$d\to n$），不重推消息量；10.3/10.4 recap 目标存在（10.3 须补版本锚 C-3，10.4 须改章号 C-4）。
- 边界 ✓：P7-P11 通信归 gpu、MIG 硬件归 gpu 11.1、编排归 cluster，无越界；预盘点补 inference 指针（C-11）。
- 修复优先级：C-1/C-2（数学命题，【重】章）> C-3（DeepEP 版本，全走读对不上）> C-4/5/6（指针，errata 传播面）> C-7/C-8 > S3。