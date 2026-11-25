# gpu W4 Part 7 对抗审查（NVLink/NVSwitch，7.1–7.3）— T6 fresh-context

日期：2026-11-25。审查员：独立 fresh-context adversarial evaluator（不继承写作上下文，未派子代理，全部工作自行完成）。
对象：`gpu/07-nvlink-nvswitch/01-nvlink-nvswitch-architecture.md`（7.1）、`02-node-topology-bandwidth.md`（7.2）、`03-gpudirect-p2p.md`（7.3）。
量规：STYLE_GUIDE + rubric（`docs/reviews/2026-11-15-style-audit/rubric.md`）+ R2 style sheet（`docs/reviews/2026-11-20-textbook-style-sheet.md`，R2-1…12 + 习题分级 §4）+ gpu spec §5 Part 7 / §4.3 符号 / §6.3 强制重推导。
来源：`docs/research/gpu-w4-sources.md` + `gpu/references.md` P7；一手来源本轮自行 curl/pdftotext/contents API 复核（下载缓存 `/tmp/gpu-w4-review-p7/`，重算脚本 `/tmp/p7-review-scripts/rederive.py`，`python3 -I` 隔离）。

## 0. 结论摘要

- **S1 = 0 / S2 = 1 / S3 = 4**。
- **NVLink 方向口径：PASS**（H100 18×25 = 450 单向 ×2 = 900 双向；B200 18×50 = 900 单向 ×2 = 1.8 TB/s 双向；Hopper wp p.47 + Blackwell brief p.10 逐字一手核实）。
- **强制重推导：全 PASS**（P2P 每对上界 / 域聚合恒等式 / bounce 两跳节省 / NVLS 流量账；独立 Python 重算 **72/72 数值检查 PASS**）。
- 唯一 S2：7.3 把串行端到端带宽 $(1/\mathrm{BW}_1+1/\mathrm{BW}_2)^{-1}$ 误称「谐波均值」（应为谐波均值之半），公式与全部数值正确、仅术语错，全章 9 处（含 metadata/总结/练习）。
- 三章 lint 全 OK；CJK 4435/3949/4351 均 ≥3500；R2-1…12 / rubric / 模板 / 习题 / 数学白名单 / rescape / 裸 `<` / process-leak 全过；recap 边界与符号 §4.3 基本合规（裸 T 一处 S3）。
- 全部 21 个引用 key 在 `gpu/references.md` 存在；一手来源（五份 whitepaper / 两 blog / CUDA PG / GPUDirect 家族 / GDR / GDS cuFile / nvbandwidth README / NCCL tarball / cuda-samples / 产品页 / 四份 datasheet）逐字核实，无虚构数值、无张冠李戴。

## 1. Findings 表

| # | severity | 位置 | 断言（原文） | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| P7-01 | **S2** | 7.3 metadata L3、§4 命题 1 符号 L92、证明 L94（×2）、代入 L96、§6 L163、总结 L176、练习 1(d) L184、练习 3(a) L186（共 9 处） | 「$\mathrm{BW}_{\mathrm{serial}}$ …为两跳带宽的**谐波均值**」；证明路线「时间相加 → **谐波均值** → 流水瓶颈」；「端到端带宽按定义 $M/T$ 即得**谐波均值**」 | 谐波均值定义 $H(a,b)=2/(1/a+1/b)=2ab/(a+b)$。式 (1) 给的是 $(1/a+1/b)^{-1}=ab/(a+b)=H/2$，**不是谐波均值而是其一半**（并联电阻式「谐波和/parallel sum」）。独立重算：$H(64,64)=64$，而式 (1) $=32$；$H(14,64)=22.97$，式 (1) $=11.49$。等跳 64/64 时「谐波均值 = 32」与定义直接矛盾（谐波均值应为 64）。公式 (1) 本身与全部数值（32 / 24.62 / 11.49 / 31.25 ms / 14.06×）**均正确**，错在术语命名，会误导熟悉谐波均值定义的读者。 | 全 9 处把「谐波均值」改为「谐波均值之半」或「倒数和的倒数（串联合成带宽，形如并联电阻）」；证明路线句改「时间相加 → 倒数和取倒数 → 流水瓶颈」；练习 3(a)「端到端带宽 = 谐波均值的推导」改「= 倒数和取倒数（半谐波均值）」。公式 (1) 与数值不动。 |
| P7-02 | S3 | 7.3 §4 证明 L94 | 「串行两跳时间相加， $`T = M/\mathrm{BW}_1 + M/\mathrm{BW}_2`$，端到端带宽按定义 $M/T$」用**裸 $T$** 表时间 | §1 符号声明（L11）只声明 $n/M/\alpha/\beta/\mathrm{BW}$（uni/serial/pipe），**未声明 $T$**；spec §4.3 全书 $T$ 专归序列长，通信章时间应带下标（7.2 式 (3) 正确用 $T_{\mathrm{ring}}$）。裸 $T$ 在本章局部无歧义，但与全书约定及 7.2 的 $T_{\mathrm{ring}}$ 纪律不一致。 | 证明中改用 $T_{\mathrm{serial}}$（或 $T_{\mathrm{end}}$）并在 §1 符号声明补一行；与 7.2 $T_{\mathrm{ring}}$ 下标纪律对齐。 |
| P7-03 | S3 | 7.1 §3 口径注 L63 | 「13.6 Tbit/s 含编码开销的**单向信令口径**，与 3.2 TB/s 有效双向不是同一笔账」——把 wp「13.6 Tbits/sec」断言为含编码开销的单向信令口径 | wp 仅逐字「Total switch throughput increases to 13.6 Tbits/sec from 7.2 Tbits/sec」，**未逐字披露 13.6 的方向/编码 basis**。research §2.2 的「单向信令含开销」是反推拟合（$64\times2\times100\,\mathrm{G}\times1.0625\approx13.6\,\mathrm{T}$，1.0625 为凑数因子）。且与本章式 (1)「H100 per-link raw = 有效 = 25 GB/s（无编码落差）」存在张力：若 per-link 无开销，64 端口单向 = $64\times25=1600$ GB/s = 12.8 Tbit/s ≠ 13.6。主账用 blog 25.6 Tbit/s（=3.2 TB/s 双向）正确，本句仅为旁注。 | 与本书他处认识论纪律（NV-HBI 方向「按原值引、不做方向推定」、per-chip 拆分「不做推定」）对齐：把该句降为推定口径，如「wp 13.6 Tbits/sec 为 switch throughput 口径，与 blog 双向有效口径不同；按 $64\times2\times100\mathrm{G}\times1.0625$ 反推约为单向信令含开销，basis 官方未逐字披露」，或直接删去「含编码开销的单向信令」具体 characterization、只保留「两数不混、本书用 blog 口径」。 |
| P7-04 | S3 | 7.1 §6 全数字实例 L154 | 「单向有效带宽取 **370.8 GB/s**——7.2 §3 的 NCCL 模型口径，教学推导」 | 370.8 = 18×20.6（NCCL SM90 per-link 模型值）在 **7.2 §3** 才引入，7.1 为上游章（metadata 列 7.2 为下游）。7.1 读者只见 370.8 数值与指针、未见 $18\times20.6$ 来源，自包含性略弱（前向依赖下游章的模型值）。算术本身正确（4.72 / 2.70 ms）。 | 二选一：(a) 7.1 实例改用 raw 450 GB/s（本章式 (2) 已交付），比值 1.75 不变；(b) 保留 370.8 但inline 补「$=18\times20.6$，NCCL per-link 模型值，推导见 7.2 §3」一句，使数值可在本章内核验。 |
| P7-05 | S3 | 7.3 §4 坑注 L106 | 「触发条件：peer access 未启用，或被 **IOMMU/ACS** 挡掉（§5）而 API 仍返回成功；现象：路径实际 **staged through host**，带宽只有串行 ~32 GB/s 量级」 | §5 正确区分两道门后果：IOMMU bare-metal 不支持 P2P → 退化 host 中转（~32 GB/s 串行）；**ACS 是把 P2P 事务重定向经 root complex（仍 device-to-device，降 bisection），并非 staged through host**。坑注把 IOMMU 与 ACS 并列归因到「staged through host / ~32 GB/s」，对 ACS 不精确（ACS 绕行带宽 ≠ host 中转 32 GB/s）。 | 坑注触发条件拆分：「peer access 未启用或被 IOMMU 挡掉 → staged through host（~32 GB/s）；ACS 开 → 绕行 root complex（bisection 降，§5）」，与 §5 判定表口径一致。 |

## 2. Verification log（强制）

### 2.1 NVLink 方向口径（任务指定焦点）— **PASS**

| 抽查项 | 一手来源（本轮自行 fetch） | 原文引句 | 结论 |
| --- | --- | --- | --- |
| H100 18 links × 25 GB/s 单向 = 450 单向 = 900 双向 | Hopper wp V1.04 PDF（dam-cdn 705n6ur5…，7880278 B）p.47，pdftotext L1572-1576 | 「fourth-generation NVLink uses only two high-speed differential pairs in each direction to form a single link, also delivering 25 GB/sec effective bandwidth in each direction. H100 includes 18 fourth-generation NVLink links to provide 900 GB/sec total bandwidth」 | **PASS**：25 单向 ×18 = 450 单向，头条 900 = 双向总和；章式 (2) 与硬规则正确 |
| B200 18 × 50 = 900 单向 = 1.8 TB/s 双向 | Blackwell brief PDF（dam-cdn gl2l4l48…，1268171 B）p.10，L332-333 | 「Blackwell and Blackwell Ultra GPUs include 18 fifth-generation NVLink links to provide 1.8 TB/sec total bandwidth, 900 GB/sec in each direction」 | **PASS**：「900 GB/sec in each direction」逐字 = 单向 900；1.8 = 双向 |
| 方向硬规则（头条双向 / per-link 单向 / per-GPU 单向 = 头条÷2） | spec §4.3 + 上两源 | 900/1800 双向、25/50 单向、450/900 单向 | **PASS**：7.1 §1/§2、7.2 §1/§3、7.3 §1 一致声明，全文先注方向再代数字 |
| NVLink Switch 命名（非 NVSwitch 4.0） | Blackwell brief L322/338「Fifth-Generation NVLink and NVLink Switch」+ NVLink 产品页 nvlink.md L60「NVLink 5 Switch」 | 官方名 NVLink Switch / NVLink 5 Switch（第四代 NVSwitch chip） | **PASS**：7.1 §3 用官方名 + 订正注，「NVSwitch 4.0」标非官方 |
| 三层互连 NV-HBI / C2C / NVLink5 | brief L204-205（NV-HBI）+ GB200 blog（C2C）+ brief L332（NVLink5） | NV-HBI「single 10 terabyte-per-second (TB/s) chip-to-chip … one fully coherent chip」；C2C「900 GB/s of bidirectional bandwidth … coherent access to a unified memory space」；NVLink5 1.8 TB/s | **PASS**：10 TB/s die-to-die（方向未披露按原值引）/ 900 GB/s 双向 CPU-GPU / 1.8 TB/s 双向 GPU-GPU，三层三对象正确，7.1 §5 表 + 坑注与 1.1 §5 recap 一致 |

### 2.2 强制重推导（独立 Python，`python3 -I`，72/72 PASS）— **全 PASS**

| 重推导项 | 章/式 | 独立重算 | 结论 |
| --- | --- | --- | --- |
| **P2P 每对上界（min 注入）** | 7.2 命题 1 式 (1) | A↔B 双向流量 $x+y$，A 注入 $=x+y\le\mathrm{BW}_{A,\mathrm{bidir}}$，B 同理 → $\le\min$；直连 $900/7=128.57\approx128$（blog Table 1 取整）；switch 单对 900；反例 $7\times900=6300>900$ | **PASS** |
| **域聚合恒等式** | 7.1 命题 1 式 (4) | $2\times8\times18\times25=7200$ GB/s = 7.2 TB/s；$8\times1.8=14.4$；$72\times1.8=129.6\approx130$；$576\times1.8=1036.8$ TB/s = 1.0368 PB/s > 1 PB/s；端点计数（每 link 恰一 GPU 端点）无重无漏 | **PASS** |
| **bounce 两跳节省** | 7.3 命题 1 式 (1)(2) | 串行 $(1/64+1/64)^{-1}=32$ → 31.25 ms；流水 $\min(64,64)=64$ → 15.625 ms；NVLink 450 → 2.22 ms；$450/32=14.06$、$450/64=7.03$；不等跳 $(1/64+1/40)^{-1}=24.62$、$\min=40$；host 流量 $2M$ vs 0 | **PASS**（公式/数值全对；「谐波均值」术语错另见 P7-01） |
| **NVLS 流量账** | 7.1 §6 | ring 每 rank 发送 $2(n-1)/n\cdot M$，NVLS 发送 $M$，比值 $2(n-1)/n$：$n=8\to1.75$、$n\to\infty\to2$；$1.75/370.8=4.72$ ms、$1/370.8=2.70$ ms，比值 1.75 闭合 | **PASS** |
| NVSwitch chip / 系统端口账 | 7.1 式 (3) | $64\times2\times25=3200$ GB/s = 3.2 TB/s；$25.6/8=3.2$；$4\times3.2=12.8$；$8\times18=144$ links；$4\times64=256$ ports；$144/4=36$ | **PASS** |
| NVL72 tray 端口账 | 7.1 §3 | $144\times100=14400$ GB/s = 14.4 TB/s；$9\times144=1296=72\times18$；$9\times14.4=129.6\approx130$ | **PASS** |
| NCCL 有效带宽模型 | 7.2 式 (2) | $18\times20.6=370.8$；$\eta=20.6/25=82.4\%$；$18\times40.1=721.8$（topo.h SM90=20.6 / SM100=40.1 一手核） | **PASS** |
| ring α-β 时间账 | 7.2 式 (3) | BW 项 $2(7/8)\times10^9/370.8\mathrm{e}9=4.72$ ms；延迟项 $2\times7\times2\,\mu\mathrm{s}=0.028$ ms；合计 4.75 ms；比值 168.6≈169 | **PASS** |
| GDS min 账 | 7.3 式 (3) | $\min(14,64)=14$；bounce $(1/14+1/64)^{-1}=11.49$；增益 1.22；71.43 vs 87.05 ms | **PASS** |
| NUMA min / NIC 数量级 / vs PCIe | 7.2 式 (4)/§7、7.1 式 (5) | $\min(64,40)=40=62.5\%$；$400/8=50$；$450/50=9$、$900/50=18$；$8\times450=3600$ vs $8\times50=400=9\times$；$900/128=7.03$、$1800/128=14.06$、$160/32=5$ | **PASS** |
| 五代 per-link raw / per-GPU | 7.1 式 (1)(2) | P100 8×20/8=20、V100 8×25.78/8=25.78（开销 3.0%）、A100 4×50/8=25、H100 2×100/8=25、B200 2×200/8=50；per-GPU uni/bidi 五代全对 | **PASS** |
| 8-peer 上限 / HBM 延迟 recap | 7.3 §3、7.2 §3 | $n-1\le8\Rightarrow n\le9$，10 GPU 需 9 peer 超限；$478.8/1.83=261.6$ ns ≈ 0.26 μs | **PASS** |

### 2.3 一手来源逐字核实（覆盖率：P7 全部数值/引文断言）

| 来源 | 本轮通道 | 核实断言 | 结论 |
| --- | --- | --- | --- |
| Hopper wp V1.04 | dam-cdn PDF + pdftotext | NVLink4 18×25=900 / 1.5× / 7× PCIe（L1566-1576）；NVSwitch 64 ports / 13.6 Tbits（L1597/1607）；PCIe Gen5 128 total/64 each（L1661）；NVLS multicast/SHARP/write broadcast/reduce_scatter/up to 2x/load on SMs（L1611-1615）；OSFP 8-channels 100G PAM4（L1655）；256 GPU/32 节点 2:1 fat tree 57.6 TB/s + 「not available with H100 systems now」（L382/267/1648） | 全 PASS |
| Blackwell brief | dam-cdn PDF + pdftotext | NVLink5 doubles / 2 lanes / 50 GB/sec / 18 links / 1.8 TB/sec / 900 each / 14X / 576（L325-335）；NV-HBI 10 TB/s fully coherent（L204）；SHARP FP8 4X bandwidth efficiency（L339）；Table 2 NVL72 18 compute/9 switch/72\|36/130 TB/s Bidirectional（L420-464）；HGX B200 Aggregate NVLink 14.4（L857） | 全 PASS |
| NVLink/NVSwitch blog | developer.nvidia.com HTML | four third-generation NVSwitch chips / 25.6 terabits bidirectional / non-blocking 逐字 / 900 split seven 128 GB/s（Table 1） | 全 PASS |
| GB200 blog | developer.nvidia.com HTML | 144 NVLink ports at 100 GB / nine switches fully connect 72×18 / C2C 900 GB/s bidirectional + coherent unified / 576 GPUs >1 PB/s / introduced in 2014（宣布口径） | 全 PASS |
| CUDA PG 13.5 multi-gpu-systems | docs.nvidia.com HTML + python 去标签 | §3.4.2.1 staged through host；§3.4.2.2 Depending on system properties / 8 peer connections / UVA same pointer **from** both devices / operates globally / multiplicative overhead Note；§3.4.2.3 thread_scope_system + atomic single GPU；§3.4.2.5 IOMMU bare-metal 不支持 + silent corruption + VM VFIO + Windows 例外 + ACS redirects bisection；§3.4.2.6 reserved and mapped + endpoint id/offset。**8-peer 上限实在 §3.4.2.2**（章正确；research §3.1 表误置 .1，非章错） | 全 PASS |
| GPUDirect 家族页 | developer.nvidia.com/gpudirect | family of technologies part of Magnum IO / 四成员 / P2P GPU-to-GPU copies loads stores over PCIe NVLink + CUDA Driver natively / GDS direct path DMA without burdening CPU / Video on Windows only | 全 PASS |
| GDR docs | docs.nvidia.com/cuda/gpudirect-rdma | same upstream PCI Express root complex / translation other than 1:1 must be disabled or pass-through / BAR1 16GB or larger | 全 PASS |
| GDS cuFile API v1.20 | docs.nvidia.com api-reference-guide（landing + functional-spec） | cuFileDriverOpen/HandleRegister(§3.2.1 required + memoized)/BufRegister/Read/Write/ReadAsync；null stream「do not have the semantic of being ordered…」逐字；O_DIRECT compat「CUDA toolkit 12.2 (GDS version 1.7.x) supports non O_DIRECT」 | 全 PASS |
| nvbandwidth README | GitHub contents API raw | 「CE copies use memcpy APIs. SM copies use kernels.」/ spin kernel + events / 3 samples median / 512 threadsPerBlock × deviceSMCount 截断公式 / pointer chasing 防 prefetch / 「The reported bidirectional GB/s represents per-direction bandwidth … not an aggregated sum.」/ 测试名 read_sm/write_ce/latency_sm/host_to_device_memcpy_ce 全在 | 全 PASS |
| NCCL repo | codeload tarball 351f33fd（4724874 B） | version.mk 2.32.7；topo.h SM80=20.0/SM90=20.6/SM100=40.1（L23-26）；p2p.cc cudaIpcOpenMemHandle L331（cudaIpcMemLazyEnablePeerAccess）/ p2pMap L358 / useRead L410 | 全 PASS |
| cuda-samples simpleP2P | GitHub contents API raw | cudaDeviceCanAccessPeer L99 / EnablePeerAccess(gpuid[1]) L130 / (gpuid[0]) L132 / UVA 注释 L159 / cudaMemcpyDefault L163/166 / 带宽打印 L173 | 全 PASS |
| NVLink 产品页 | nvidia.com nvlink.md | NVLink 4 Switch 900 GB/s / Total Aggregate 7.2 TB/s；NVLink 5 Switch 1800 / 130 TB/s（NVL72）；Rubin 列非 scope（章正确排除） | 全 PASS |
| datasheet ×4 | dam-cdn PDF + pdftotext | H100（u5hh6fv4，411990 B）SXM NVLink 900 / **PCIe 版 600** / PCIe Gen5 128；H200（b2kt2580，642800 B）900 + bridge；HGX（44wa，312368 B）NVSwitch 4-GPU **N/A** / 8-GPU Third generation / 900GB/s；Blackwell（01dyp27s，1156321 B）Total NVLink **130/7.2/14.4 TB/s**（NVL72/NVL4/HGX B200） | 全 PASS |

### 2.4 recap 边界 + 符号 §4.3 + 规范 — PASS（除 P7-02 裸 T）

- **recap 忠实**：training 4.1 §2.2 式 (2) $C_{\mathrm{rank}}=2(d-1)/d\cdot M$、式 (3) $T_{\mathrm{ring}}=2(d-1)\alpha+2\frac{d-1}{d}M\beta$ 一手核（`training/04-data-parallelism/01-dp-and-ddp.md` L62/70）；7.1 §6 / 7.2 §3 recap 均 $d\to n$ 翻译不重推 ✓。errata 句「节点内 NVLink 与跨节点 IB 的有效带宽差一个量级（硬件侧归 cluster 维度）」逐字匹配 training 4.1 L73，7.2 §1 重判到本章 ✓。1.1 §5 NV-HBI/C2C 坑注（L158）+ 10>7.7 对比账（L156）、2.4 §4 HBM 478.8 cycles@1.83 GHz（L99/132）、2.4 §6 nvbandwidth 口径 + 双向报值坑（L204）、3.2 thread_scope_system + page fault、3.3 NULL stream 均一手核，recap 忠实 ✓。
- **跨章前瞻指针**：7.2 §4 ring 映射 → 8.2（下界证明）/8.4（rings.cc）✓；7.3 §7 → 9.2（RDMA）/8.1/8.4 ✓；7.1 §6 NVLS → 8.4/10.2 ✓。无越界重推（ring 带宽最优性归 8.2、GDR 归 9.2，本 Part 只引结论）✓。
- **符号 §4.3**：$n$ 专归 rank（7.2/7.3 章首声明）✓；7.1 GPU 数用 $n_{\mathrm{GPU}}$、$n$ 不占用 ✓；$N$/$B$ 不占用 ✓；$\alpha$ 章首声明消息延迟非接受率 ✓；7.2 式 (3) $T_{\mathrm{ring}}$ 下标 ✓；**7.3 §4 证明裸 $T$ 未声明 → P7-02（S3）**。
- **R2-1…12**：节首 crux/设问/数值开场全 21 节合规 ✓；章脊口径锁定（H100 SXM / HGX 8-GPU，对比 SKU 声明）✓；公式后 300 字内全数字实例 ✓；命题-证明骨架（三章各 1 命题 1 证明，路线句齐）✓；定义即配正例 + 边界反例（NVSwitch/domain/P2P/GPUDirect P2P）✓；口径注密集 ✓；术语判断句「所谓 X，是指」✓；逻辑关节词 ✓；提点/坑块引用（7.1 提点 1 坑 2 / 7.2 坑 1 / 7.3 坑 4）✓；总结表回指列全 ✓。
- **rubric A/B/C**：AI 味禁令词 grep 零命中 ✓；无历史叙事（出货年/历史注为口径注非叙事，spec 允许）✓；含糊量词「显著」2 处均带数值或 wp 逐字归属 ✓；无营销形容词 ✓。
- **模板/习题**：metadata + 总结/练习/参考齐 ✓；习题 [10]/[20]/[30]/[30] 难度非递减、≥1 入场券、≤1 [40]（实 0）、每计算题校验锚、≥1 题证正文命题（各章练习 3）、先预测要求（7.1 ex2c / 7.2 ex2 / 7.3 ex2）✓；7.1 证明题取定量论断（域聚合恒等式）合 spec §4.6 事实密集章口径 ✓。
- **数学白名单/rescape/裸 `<`/process-leak**：`\operatorname`/禁止环境零命中 ✓；rescape 禁序列（`\,`/`\;`/`\:`/`\!`/`\{`/`\}`/`\%`/`\#`）fixed-string grep 零命中 ✓；全文无 `<` 字符（用 `\le`）✓；process-leak 词零命中 ✓；display 用 `\thinspace`/`\hspace{5mu}` 合规 ✓。
- **lint**：三章 `bash scripts/lint-chapter.sh` 全 **LINT OK** ✓。**CJK**：7.1=4435 / 7.2=3949 / 7.3=4351，均 ≥3500（标准章）✓。**refs**：21 个 key 全在 `gpu/references.md`（各 1 条）✓。

## 3. 修复优先级

1. **P7-01（S2，必修）**：7.3「谐波均值」→「谐波均值之半/倒数和的倒数」，9 处全改（metadata L3 / L92 / L94×2 / L96 / L163 / L176 / L184 / L186）。公式 (1) 与数值不动。
2. **P7-02…05（S3，累计下波清零）**：裸 T 改 $T_{\mathrm{serial}}$；13.6 口径注降推定；7.1 §6 实例 370.8 补来源或改 raw 450；7.3 §4 坑注拆分 IOMMU/ACS 后果。

## 4. 审查结论

P7 三章事实脊柱（NVLink 方向口径、五代几何、NVSwitch/NVL72 端口账、三层互连、GPUDirect P2P/GDS 一手引文、NCCL/cuda-samples 代码锚）**全部一手核实通过**，强制重推导 72/72 PASS，recap 边界与符号约定基本合规，规范全绿。唯一实质问题为 7.3「谐波均值」术语误用（S2，公式数值正确、命名错，9 处）；4 条 S3 为符号/口径注/坑注精度的 minor 改进。**无 S1**。建议：修 P7-01 后即可过审，S3 累计下波清零。