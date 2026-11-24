# gpu W1 T6 对抗审查 — Part 2（内存层级与带宽账，2.1–2.4）

日期：2026-11-25。审查人：fresh-context adversarial evaluator（不继承写作上下文、未派子代理、全部工作自做）。
范围：`gpu/02-memory-hierarchy/01-hbm-and-memory-subsystem.md`、`02-cache-hierarchy.md`、`03-onchip-storage-registers.md`、`04-bandwidth-accounting-roofline.md`。
输入：spec `docs/superpowers/specs/2026-11-24-gpu-dimension-design.md` §5 Part 2 / §4.2 / §4.3 / §4.6 / §6.2 / §6.3；recap 源 `models/02-transformer-anatomy/04-causality-and-kv-cache.md` §6 式 (10)-(12)；`STYLE_GUIDE.md`；R2 `docs/reviews/2026-11-20-textbook-style-sheet.md`；rubric `docs/reviews/2026-11-15-style-audit/rubric.md`；`docs/research/gpu-w1-sources.md`；`gpu/references.md` P1/P2；兄弟章 1.1–1.4 与 inference 1.3（recap 对账）。
方法：一手来源自行 curl/dam-cdn PDF + GitHub API + arXiv HTML（WebFetch 通道沿 T1 记录不可用，全 curl）；全部数值账独立 Python 重算（112 项）；recap 边界逐式比对；R2-1…12 + rubric + 模板 + 练习分级 + 数学白名单 + display 反转义 + flanking 逐条机械/判定审查。
下载隔离：每源独立空目录 `/tmp/t6-p2/<src>/`，脚本在 `/tmp/t6-p2-scripts/`（`-I` 运行）。

## 0. 结论摘要

- **S1 = 0，S2 = 0，S3 = 7**（全部为 minor：2 项 T5 遗留未修 + 5 项口径/标注/元数据精化）。
- **强制重推导 2.4：PASS**（层级带宽全表 + 归一化 + ridge 代入 + 三区判定 + achieved roofline，112 项独立重算全过，见 §2）。
- **recap 边界：PASS**（I / 吞吐上限 / ridge / decode 饱和上界均为 models 2.4 §6 式 (10)-(12) 与入口 prose 的翻译 recap，标「不重推」；增量 = 层级全表 + HBM/峰值微架构来源对账 + 测量方法 + 三区精确化 + 实测 roofline，见 §3）。
- 事实核查：45 项抽查全部一手核实闭合（含 3 篇 arXiv 微基准逐字、4 份 datasheet 中 3 份逐表、wp Table 3/4 + Fig 6/7 视觉读图、ECC p.38 逐字、nvbandwidth README 逐字、Nsight ProfilingGuide §2.9 逐字、PG 现行四子页 + legacy 13.0.0 逐字、三 vendor 页逐字）；⟨verify⟩ 残留（JEDEC 精确 URL / PCI-SIG 精确 URL / B200 SM 数）章内已标延迟，**不计 finding**。
- 机械门：lint 4/4 OK；display 反转义陷阱 0；白名单违禁 0；flanking 真实违例 0（6 处扫描命中均为 `` $`...`$ `` 保护形，开口豁免）；rubric A 禁令词 0；练习四档/校验锚/命题闭环 4/4 合规；CJK 4003 / 3577 / 4642 / 5697（2.4 ≥5500 软目标过）。
- **放行建议**：P2 四章无 S1/S2，可进 fix 轮清 S3 后随波 commit；S3-1/S3-2 为 T5 明示遗留，本轮判定「仍未修、维持 minor」。

## 1. Findings 表

| # | severity | 位置 | 断言（问题） | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| S3-1 | S3 | 2.4 §1 符号声明 vs §5 首段（T5 遗留①，未修） | $L$ 双语义未消歧：§1 声明「$L$ = 层数」（models 语义），§5 recap 行「Little's law $L = \lambda \cdot t_{\mathrm{lat}}$」裸 $L$ 取 1.4 的在飞并发语义；同章两义。§1 混场声明只覆盖 $R$/$R_{\mathrm{th}}$，未覆盖 $L$ | 2.4 L19「$L$ = 层数」；L138「recap 1.4（不重推）：Little's law $L = \lambda \cdot t_{\mathrm{lat}}$」；1.4 L11 声明 $L$ = 在飞并发量。spec §4.3 消歧硬规则未列 $L$，故非硬违规；recap 行有标签且本章并发一律 $C$/$L_{\mathrm{req}}$，混淆面仅一行 | §1 混场声明补一句：「1.4 recap 行的 $L$ = 在飞并发（1.4 局部符号），本章 $L$ = 层数；本章并发一律 $C$ / $L_{\mathrm{req}}$」；或 §5 该行改写为 $C = \lambda \cdot t_{\mathrm{lat}}$ 口径注 |
| S3-2 | S3 | 2.4 §7 实测 roofline ASCII（T5 遗留②，未修） | (a) log-log 轴下标注「斜率 = BW = 3.35 TB/s」不精确：log-log 下 memory 斜线斜率为 1，BW 是乘性系数（定线位置/截距）；(b) 几何缺陷：knee 在 col 20（┌ 与上段竖线），下段竖线与轴 ┼（R≈295）在 col 9——ridge 垂线折弯、R tick 不在 knee 正下方；(c) B(col 8)/C(col 9) 同 $I=2.5$ 未垂直对齐，且 C 落在 R tick 上（log 轴上 2.5 应远在 295 左侧） | 逐列解析：上段竖线 col 20、下段 col 9、┼ col 9、● B col 8、○ C col 9、● A col 34（knee 右 ✓）。Nsight ProfilingGuide §2.9 自身措辞「this slope is determined entirely by the memory transfer rate」+ 双轴 logarithmic——标注有来源沿袭但数学不精确；图已标「示意非按数值比例」豁免比例、不豁免拓扑 | 图子循环（spec §6.6 roofline 为签名场景）产图时修正：knee 直落 R tick；B/C 垂直对齐且严格在 ridge 左；标注改「斜线 = $I \times \mathrm{BW}$（log-log 斜率 1，BW 定线位置）」。ASCII 若保留则同改 |
| S3-3 | S3 | 2.4 §1 符号声明 / §2 / 总结 | $T_p$ 未在本章声明即使用（§2「$T_p$ 个 token 一次前向」、总结「prefill $I \approx T_p$」）；§1 只声明 $T$ = 序列长。$T_p$ 为 models 2.4 §6 符号（prompt 长度），recap 继承但章内无定义句 | models 2.4 L218「$T_p$ 个 token 一次前向」；2.4 L19 符号表无 $T_p$ | §1 符号声明补「$T_p$ = prefill prompt 长度（models 2.4 §6 沿用）」 |
| S3-4 | S3 | gpu/references.md P1/P2 条目元数据 | 被引章节回写滞后 + nvbandwidth 核实日期滞后：hopper-whitepaper 被引「1.1, 1.2, 1.4, 2.1, 2.2, 4.4」缺 2.3/2.4；blackwell-whitepaper「1.1, 1.3」缺 2.1/2.2/2.4；hopper-microbench「1.4, 2.2」缺 2.3/2.4；cuda-programming-guide 缺 2.4；nvbandwidth 核实日期 2026-11-24 而 2.4 章首/参考声称「README 2026-11-25 经 GitHub API 一手核实」（本轮 evaluator 已独立复核 README 全文逐字相符，内容无误，仅元数据未回写） | references.md L54/55/57/64/72；2.4 L3/L187/L317；spec §6.1 步骤 4「被引章节回写」 | 回写被引章节 + nvbandwidth 条目补 README 核实描述与日期 2026-11-25 |
| S3-5 | S3 | 2.1 参考 [Zand et al., 2026] 条 | 「vs datasheet usable 180 / 270 GB」归属不精确：B300 usable 270 GB 出自 Blackwell **brief** 对照表（brief txt L892「270 GB HBM3e \| 7.7 TB/s … Up to 192 GB HBM3e」），Blackwell datasheet PDF 无 B300/270（本轮 grep 0 命中）；正文 §6 未误归属，仅参考条措辞 | brief.pdf pdftotext L892；ds-bw/bw.txt grep「270\|B300\|GB300」0 命中 | 参考条改「vs datasheet/brief usable 180 / 270 GB」 |
| S3-6 | S3 | 2.3 §1 实测口径声明 vs §3 例 vs §7 表 | shared 带宽换算双 clock：§7 表「128 B/cycle/SM = 253 GB/s @1.98 GHz」（recap 1.2）与 §3 stride-32 例「4 B/cycle @1.83 GHz 合 7.3 GB/s」并存；§1 声明只把 1.98 归「register operand 带宽账」，未注 shared 峰值换算沿 1.2。两处均有点标、结论（7.3 或 7.9 均 < 25.4）不随 clock 变，故 minor | 2.3 L9/L56/L146；1.2 L112（shared 253 @1.98） | §1 补一句口径注：shared 峰值 GB/s 换算沿 1.2 的 1.98 GHz，最坏例与延迟换算同取 1.83 GHz，两 clock 下结论同向 |
| S3-7 | S3 | 2.4 §7 层级 roofline 段末句 | 「Tensor Core kernel 须用 tensor roofline section，并交叉核对 profiler 报的 SM frequency 与 SASS 操作计数（Nsight §2.9 逐字口径）」——该句为忠实中文压缩而非逐字； guide 原文为「Ensure that the enabled roofline section matches the data path being analyzed, including tensor roofline sections for tensor-core kernels. Cross-check the SM frequency reported by the profiler, and use operation counters together with source or SASS inspection to validate the amount of math that actually executed.」同段另两条英文引句（distance / hierarchical）确为逐字 | ProfilingGuide §2.9 本轮 curl 逐字（见 §2 表 V-20/V-21） | 「逐字口径」改「口径」或补英文逐字引句 |

无 S1/S2。判定纪律：拿不准的不报；上表 S3 均带原文位置与一手证据。

## 2. Verification log（强制）

### 2.1 事实核查抽查表（抽查项 × 来源 × 结论）

| # | 抽查项 | 来源（本轮一手通道） | 结论 |
| --- | --- | --- | --- |
| V-01 | HBM BW = 5×16×5.238e9×64/8 = 3352.32 GB/s；w_ch=64bit/ch；1024 误代 16× | wp Table 3（dam-cdn PDF）「5 HBM3 stacks / 5120-bit HBM3 / 2619 MHz DDR / 3352 GB/sec」+ Micron 16ch/1024 IO + 独立重算 | ✓ 逐字+重算闭合；53637.12 GB/s = 16× ✓ |
| V-02 | ECC SECDED + Sideband + L2/L1/RF SECDED + Row Remapping（p.38） | wp PDF p.38 逐字三段 + remap 段（本轮 pdftotext L1267-1283） | ✓ 三引句逐字相符；Sideband→数据带宽不变为章内推论（wp 只披露放置口径），章已标口径注 ✓ |
| V-03 | Hamming 界 d=64 → r=7+1=8、12.5%；Inline 64/72≈88.9% | 独立重算（2^6=64<71；2^7=128≥72） | ✓ 教学实例化标签在位 |
| V-04 | L2 50 MB 两分区 + partitioned crossbar prose | wp Table 3「L2 50 MB」+ Fig 6 视觉（左右两 L2 Cache 块+中间双向箭头，本轮 pdfimages p.19 读图）+ prose 逐字 + Jarmusch 互证 | ✓ |
| V-05 | cache line 128 B / sector 32 B | Nsight ProfilingGuide glossary 逐字「Aligned 32 byte-chunk … An L1 or L2 cache line is four sectors, i.e. 128 bytes」 | ✓ |
| V-06 | register file 256 KB/SM、65536×32-bit、4 分区 16384×32-bit、33792 KB/GPU、255/thread | wp Table 3 + Fig 7 视觉（本轮 p.21 读图：4×「Register File (16,384 x 32-bit)」）+ PG Table 34 | ✓ |
| V-07 | constant 64 KB + 每 SM 常量缓存工作集 8 KB（「同表逐字」） | PG compute-capabilities Table 34 行「Constant memory size 64 KB」「Cache working set per SM for constant memory 8 KB」（legacy 同表亦有） | ✓ 章归属正确 |
| V-08 | 延迟 shared 29.0 / L1 40.7 / L2 263.0 / HBM 478.8（H800 [B]）+ DSMEM 180 | arXiv 2402.13499v1 HTML Table IV 逐字（H800 列）+ 「SM-to-SM network latency is 180 cycles」 | ✓ 实测口径声明在位（非 H100 官方） |
| V-09 | TC-boost 1830 / FP32-boost 1980；BF16 wp 989.4 | wp Table 3 逐字两 clock 行 + 989.4/1978.9 | ✓ |
| V-10 | datasheet H100 3.35 TB/s / 80 GB / BF16 1979* / NVL 94 GB 3.9 TB/s 勿混 | dam-cdn H100 datasheet PDF 逐表 | ✓ |
| V-11 | H200 141 GB / 4.8 TB/s / ≤700W(SXM) ≤600W(NVL) | dam-cdn H200 datasheet PDF | ✓ |
| V-12 | HGX B200 180 GB / 7.7 / 2250 / 4500 / 9000 dense / 1000W / PCIe Gen5 128 GB/s；GB200 186 / 8 / 2500 / 1200W；sparse 脚注逐字 | dam-cdn Blackwell datasheet PDF 逐表 + 脚注 2 逐字 | ✓ ridge 表分子分母全部对账 |
| V-13 | raw 192 = 8×24（brief「Up to 192 GB HBM3e」）；B300 270 | dam-cdn Blackwell brief PDF L892 | ✓（270 归属见 S3-5） |
| V-14 | nvidia-smi 179 GiB / 269 GiB ≈ raw 192/288 | arXiv 2610.41919v1 逐字「Driver-reported HBM3e is 179 GiB per B200 and 269 GiB per B300 …」+ Table III 183,359/275,040 MiB + 独立换算 192.2/288.8 GB | ✓ |
| V-15 | Micron 16 independent channels / 32 pseudo / 1024 IO / 24GB 8-high H200 / >9.2 Gbps | micron.com HBM3E 页逐字 | ✓ |
| V-16 | Samsung 6.4 Gbps→819 GB/s、9.2→1180、12H,8H；SK hynix 9.6→1.23 TB/s、8,12Hi | semiconductor.samsung.com + product.skhynix.com 逐字 | ✓ 每栈公式三 vendor 闭环（819.2/1177.6/1228.8 重算 ✓） |
| V-17 | nvbandwidth CE/SM 逐字、测试名族、spin kernel+events、3 采样中位数、512×SM 截断公式、pointer chasing/TLB warm、双向报值 per-direction 逐字 | GitHub API README 全文（本轮一手） | ✓ 七项逐字/公式相符 |
| V-18 | D2D 同卡 ×2 读+写账 | README 带宽公式（payload/time）+ 物理账独立推论；章标「账口径…工具输出口径以其 README 为准」 | ✓ 非 README 逐字、章已声明口径 |
| V-19 | Nsight SpeedOfLight「achieved percentage … theoretical maximum」、.pct_of_peak_sustained_elapsed 定义、Mem Busy/Max Bandwidth/Mem Pipes Busy、gpu__time_duration semaphore、「use different methods to measure the duration of a workload」 | ProfilingGuide 本轮 curl 逐字 | ✓ 五项相符 |
| V-20 | §2.9 编号 + distance 引句 + hierarchical L1/L2/DRAM 三句引句 | ProfilingGuide「2.9. Roofline Charts」+ 逐字 | ✓ 编号与引句相符 |
| V-21 | tensor roofline + SM frequency/SASS 交叉核对 | ProfilingGuide §2.9 原文（压缩转述，见 S3-7） | ✓ 内容相符、标签过强（S3） |
| V-22 | FADD/FMUL=1、FFMA=2 FLOPs 计数 | 定义式算术（FMA=2 FLOPs）；guide 未列权重、章已声明「计数器名以 Metrics Reference 为准」 | ✓ 口径声明在位 |
| V-23 | PG 现行：spill 定义/local「logical scope」/同延迟带宽/coalesced 排布/三触发/texture「no longer … benefit」/DSMEM §2.3.3.8 | writing-cuda-kernels.html 逐字 | ✓ 六引句相符 |
| V-24 | PG >48 KB opt-in 逐字、carveout「next larger capacity」、96 KB=98304 示例、hint 非保证 | advanced-kernel-programming.html 逐字 | ✓ |
| V-25 | L2 set-aside cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)、hitRatio 概率性、thrashing 例 | l2-cache-control.html 逐字 | ✓ |
| V-26 | PG Table 33/34/35：64K regs/255/32 banks/local 512 KB/CC9.0=10.0=228 KB（colspan=3 核实）/227 opt-in/unified 256/carveout 十档/KB=1024 注 | compute-capabilities.html 表格 colspan 解析 | ✓ CC10.7=328 与章「B300 系非 W1 口径」相容 |
| V-27 | legacy：broadcast「split into as many separate requests」、texture 2D locality+constant latency、low-precision、[0,1-1/N]、surface 字节寻址、Wrap/Mirror 归一化、__maxnreg__ | archive/13.0.0 单页逐字 | ✓ 七项相符 |
| V-28 | shared broadcast 写「which thread performs the write is undefined」 | writing-cuda-kernels.html 逐字 | ✓ |
| V-29 | PG Figure 15/17 = strided / 32×32 column bank 结构（2.3 练习 1 校验锚） | writing-cuda-kernels.html Figure 15/16/17 caption | ✓（legacy 为 39/40，章锚现行 PG 正确） |
| V-30 | Blackwell tuning guide：GB200 L2 126 MB、227 KB+1 KB reserved 逐字、B200 180 GB、static 48 KB | blackwell-tuning-guide 本轮 curl 逐字 | ✓ |
| V-31 | Best Practices「typically 4 cycles」(CC7.0 措辞) +「hundreds of clock cycles」 | cuda-c-best-practices-guide 本轮 curl | ✓ |
| V-32 | NV-HBI 10 TB/s「providing one fully coherent chip」、NVLink5「18 fifth-generation … 1.8」 | brief PDF 逐字 | ✓ |
| V-33 | NVLink4 900 GB/s 双向总和 | wp prose「900 GB/sec total bandwidth for multi-GPU IO」+ Fig 6 18 links（视觉）→ 450 单向 / 25 per-link 算术闭环 | ✓（NVLink 产品页本轮未重 fetch，沿 refs 2026-11-24 核实 + W4 残留标签；wp 一手已足） |
| V-34 | PCIe Gen5 x16 = 64 单向 / 128 双向 | spec §4.3 钉值 + Blackwell datasheet「PCIe Gen5: 128 GB/s」互证；32 GT/s×16/8=64 算术 | ✓ 理论口径标签在位；PCI-SIG 原文 403 残留已标 |
| V-35 | Jarmusch「In the GH100, the L2 cache is divided into two independent partitions」+ GB203 monolithic | arXiv 2507.10789v1 HTML 逐字 | ✓ |
| V-36 | Nsight metrics：derived__shared_spilling_requests、l1tex__data_bank_conflicts_pipe_lsu.sum、DCC/IDC/IMC、long_scoreboard/not_selected | ProfilingGuide 本轮 grep（2/11/3/8 命中） | ✓ |
| V-37 | 1.2 §5 MFU 指针、1.4 式 (1)-(5)/208/0.81/116、1.3 §4 0.25 B/FLOP、1.1 §3 舍入漂移 | 兄弟章本轮 grep 对账 | ✓ 跨章 recap 数值逐一对上 |
| V-38 | inference 1.3 ridge 表 295/206/292/591/412/584/1169 + W=16.06 GB + 式 (2) 步时 | inference 1.3 本轮 grep + 独立重算 | ✓ 逐行一致 |
| V-39 | models 2.4 §6 式 (10)-(12) 原文与翻译映射 | models 2.4 L220-242 逐字比对 | ✓ t→T、d→d_model、b→bytes 翻译正确（§3 边界判定） |
| V-40 | wp Table 3「(Not Finalized for H100)」残留脚注 | 本轮 pdftotext L1347 | 信息注：V1.04 表内残留标记与封面 final 声明并存；章取 Table 3 final 3352 与 1.1 漂移注口径一致，**非 finding** |
| V-41 | research notes §2.1「132×4096×1.83 = 989.46」 | 独立重算 = 989.42976 | 信息注：**research notes 笔误**（989.46），章节值 989.43 正确；非章节 finding，建议 research 档案下轮勘误 |
| V-42 | HGX H100/H200 datasheet（refs 第四份） | 本轮未重 fetch | 覆盖注：章用数值（NVL 94/3.9 等）已由 H100 datasheet 一手覆盖；HGX 份无独用数值，留 W4 随 refs 复核 |
| V-43 | JEDEC JESD238A 原文 / PCI-SIG 原文 / B200 SM 数 | 403 / 未披露 | ⟨verify⟩ 残留，章内已标延迟通道（W4/W3/W6），**不计 finding** |
| V-44 | 2.1 §2 顶视图 ch0-7→MC0 分组 | wp 只披露 10×512-bit MC；512/64=8 ch/MC 为算术必然，具体分组为示意 | 信息注：机制示意、非逐字披露；§5 已声明调度闭源；不构成错误事实 |
| V-45 | 2.4 §2「min of 两条线」一句机制解释是否越 recap 边界 | models §6 prose 无编号入口 + 章标「只 recap 与翻译…不重推」+ 无新编号/无证明骨架 | 边界判定：recap latitude 内（一句直觉复述+指针），**非 S2**（见 §3） |

覆盖率：P2 四章全部披露数值/逐字引句/公式代入项 45/45 抽查闭合（V-40…V-44 为信息注/覆盖注，非未决事实）；未决项仅章内已标 ⟨verify⟩ 残留 3 项。

### 2.2 强制重推导 2.4（独立 Python，112 项）

脚本 `/tmp/t6-p2-scripts/rederive.py`（覆盖 2.1/2.2/2.3 全部数值账 + 2.4 全部）。**112/112 PASS**。2 项初报 FAIL 经复核为审查侧容差伪影：40.7/1.83=22.2404→章 22.2 为一位小数正确舍入；(1−0.8)(1−0.5)=0.1 浮点尾差。关键独立值：

| 项 | 独立重算 | 章值 | 判定 |
| --- | --- | --- | --- |
| 层级全表 register | 2048 B/cyc×1.98 GHz=4055.04 GB/s/SM；×132=535.26 TB/s | 4055 / ≈535 | ✓ |
| shared | 128×1.98=253.44 GB/s/SM；×132=33.396 TB/s | 253 / ≈33.4 | ✓ |
| HBM 份额 / 片上比 | 3352/132=25.394；4055:253:25.4=159.7:9.96:1 | 25.4 / 160:10:1 | ✓ |
| 片外比 | 3352:450:64=52.4:7.03:1；3352/900=3.724 | 52:7:1 / 3.72 | ✓ |
| ridge（BF16/FP8/FP4 ×3 SKU） | 295.37/206.15/292.21；590.75/412.29/584.42；1168.83 | 295/206/292；591/412/584；1169 | ✓ |
| F_peak 分子链 | 132×4096×1.83e9=989.42976 TF | 989.43 | ✓ |
| 分母链（recap 2.1） | 3352.32 GB/s | 3352.32 | ✓ |
| Little recap | λ=25.394e9/(32×1.83e9)=0.43364；L_req=207.63 | 0.434 / ≈208 | ✓ |
| BDP | 3352e9×261.6e-9=876.88 KB；/132=6.643 KB；shared 4×29=116 | 877 / 6.6 / 116 | ✓ |
| 三区/latency 上界 | I·BW=8.38 TF；C=104→4.19；C=64→u=0.3077、BW=1031.4、F=2.578 | 8.38/4.19/0.308/1032/2.58 | ✓（1032 为 u=0.308 舍入链，表内自洽） |
| achieved 实例 | u_B=0.8507、u_F=0.0072、I_ach=2.5；A u_F=0.7074；B roof 8.375；C u=0.5003 | 85.1%/0.72%/2.5/70.7%/8.375/50% | ✓ |
| 练习 4 | 5.7e9/2ms=2.85 TB/s；1.425e10/2ms=7.125 TF；payload 1425 GB/s | 同 | ✓ |
| serving 落点 | 16.06e9/3.35e12=4.794 ms；/0.85=5.640 ms | 4.79/5.64 | ✓ |
| 2.1/2.2/2.3 全账 | 误代 16×/53.6 TB/s、Hamming、camping 41.904=1/80、179GiB=192.2 GB、AMAT 五例+ns、sector η=0.125→419、spill 32.768 MB→1.29/0.262/1.05 ms、练习 4 65.536 MB→2.58/2.09 ms 等 80 项 | 同 | ✓ |

**重推导 2.4 判定：PASS**（层级带宽账 + HBM 微架构来源 + 测量/三区/achieved 全部独立复现；无跳步、无量纲错）。

### 2.3 recap 边界判定：PASS

- models 式 (10)：gpu 2.4 §2 译写 $I_{\mathrm{prefill}} \approx T_p$（models 原式含 $2NT_p/2N$ 中间步，recap 省略中间步但标「已交付公式，不重推」+ 指针）✓。
- models 式 (11)/(12)：逐符号比对 $t \to T$、$d \to d_{\mathrm{model}}$、$b \to \mathrm{bytes}$，分子分母结构逐字同构 ✓；翻译注在 §1 声明 ✓（spec §4.2 硬规则）。
- 入口 prose 三关系（I=FLOPs/bytes、min 上限、ridge=峰值÷带宽）：§2 以无编号 prose recap ✓；§3 式 (1) 编号明标「models prose / inference 1.3 式 (1) recap，本章编号供代入引用」✓ 非新推导。
- decode 饱和上界 4.9 / 满窗 2.5 / 短窗 1：§2 标「数值均为 models 2.4 §6 已交付口径」✓ 无重推。
- ridge 数值表：标「与 inference 1.3 §2 ridge 表逐行一致」✓（V-38 逐行对上）。
- 增量核对（spec §6.3 清单）：层级全表+归一化式 (4)(5) ✓、HBM/峰值微架构乘法链对账（recap 2.1 式 (1)(2) + 1.1 式 (2)/1.3 §3，均标不重推）✓、测量方法（nvbandwidth/Nsight/式 (8)(9)）✓、三区精确化（式 (6)(7)+命题 1，1.4 命题 1 recap 乘 I）✓、实测 roofline（式 (10)(11)+命题 2，测量增量非 models 上限重推）✓。
- 边界灰区判定（V-45）：§2 首段「两个独立资源封顶…取小」一句为入口 prose 的直觉复述（无新编号、无证明骨架、紧跟「推导见 models 2.4 §6，本章不复述」），属 recap latitude；§7 命题 2 是 measured-point 上界（$I_{\mathrm{ach}}$ 口径）的形式化，属 spec 明列增量「点落 roofline 下方=利用率损失」，**不构成 S2 越界**。
- 越界检查：无 serving 调度/引擎编排、无 K8S 内容；§7 serving 落点为 recap inference 式 (2) + u_B 代入（标「这不是新推导」+ 操作点归 inference 9.1）✓；2.2 §5 persisting 决策归 inference/4.x 边界声明 ✓。

### 2.4 教学审查（推导断步/未定义符号/循环解释/前置缺口）

- 推导路线句（R2-5）：2.1 §3/§4、2.2 §2/§3、2.3 §6、2.4 §3/§4/§7 均在位；命题-证明骨架（R2-4）2.1×3 / 2.2×2 / 2.3×1 / 2.4×2，降级「留作练习 N」全部有对应练习 ✓。
- 符号：四章 §1 声明齐备；bytes 不占字母 ✓；$R$/$R_{\mathrm{th}}$ 混场声明 ✓；未定义仅 $T_p$（S3-3）与 $L$ 双语义（S3-1）；$W_{\mathrm{active}}$/$\mathcal{O}$/$s_{\mathrm{warp}}$ 经前置 1.2/1.4 recap 链引入 ✓。
- 全数字实例（R2-3）：逐 display 公式后 300 字内落带单位数值 ✓（含 2.4 式 (4)(5)/(6)/(7)/(8)(9)/(10)(11)）。
- 前置缺口：无（P1→P2 依赖边全部 recap 接口在位；2.3→2.4 前向指针「2.4 §4」为同波次前向、可接受）。
- 循环解释：无（spill 双代价、AMAT、sector 账各给独立机制理由）。

### 2.5 规范审查（R2-1…12 + rubric + 模板 + 练习 + 白名单 + 反转义 + 反 AI 味）

- R2-1 节首设问/数值/元问题：四章逐节过（含「先回答一个更基本的问题」式）✓；R2-2 章脊口径：四章 §1 锁定，2.1/2.4 多 SKU 一次声明 ✓；R2-6 定义配例+反例：camping/sector/bank/broadcast/spill 均有反例 ✓；R2-7 口径注密集在位 ✓；R2-8 判断句定义 ✓；R2-9 关节词抽查过 ✓；R2-10/11 坑注 2.1×1/2.2×1/2.3×1/2.4×2 含触发+现象+规避 ✓（P2 无类比混主线，提点非必需）；R2-12 总结表回指列 4/4 全行含 §/式/练习 ✓。
- 练习：4 章各 4 题 [10]/[20]/[30]/[30] 非递减、≥1 入场券、无裸 [40]、计算题全带校验锚、≥1 题证正文命题、先预测要求在位 ✓。
- 模板：标题/定位块/总结/练习/参考 齐全；每章 ≥1 ASCII 图 ✓；lint 4/4 LINT OK。
- 数学：白名单违禁 0（无 operatorname/align/boxed/color/newcommand/medspace）；反转义陷阱 `\, \! \; \: \{ \} \% \#` 0、表格 `\|` 0；flanking 真实违例 0（6 命中均 `` $`...`$ `` 豁免形）；公式内 `<` 一律 `\lt` ✓。
- rubric A/B：禁令词 0；无历史叙事；无空洞评论收尾；术语首现中英+统一 ✓。
- 篇幅：CJK 4003/3577/4642/5697，标准章 ≥3500 全过、2.4 ≥5500 软目标过。

### 2.6 T5 遗留 minors 裁定

| 遗留项 | 本轮状态 | 裁定 |
| --- | --- | --- |
| 2.4 L 双语义消歧 | 未修（§1 混场声明仍只覆盖 R/R_th） | 维持 minor → S3-1，fix 轮一句声明关闭 |
| roofline ASCII slope log-log 标注 | 未修（「斜率 = BW」保留 + 几何折弯/同 I 不对齐，见 S3-2 证据） | 维持 minor → S3-2，图子循环产图时一并关闭 |

## 3. 汇总

- findings：**S1×0 / S2×0 / S3×7**（S3-1…S3-7，见 §1）。
- 重推导 2.4：**PASS**（112/112 独立重算）。recap 边界：**PASS**。
- 事实抽查 45/45 闭合；⟨verify⟩ 残留 3 项章内已标延迟，不计 finding。
- 信息注 2 条（非 finding）：research notes 989.46 笔误（章值 989.43 正确）；wp Table 3「Not Finalized」残留脚注（章取 final 口径正确）。
- 处置建议：S3 累计下波清零通道或本波 fix 轮逐条判定；S3-1/S3-2 建议本波 fix（T5 已遗留一轮）；S3-4 元数据回写随 fix commit；其余可接受/修复由 controller 裁定。P2 无 S1/S2，**审查放行进入 fix + 图子循环**。
