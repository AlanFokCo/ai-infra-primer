# inference W1 对抗审查报告 — P1（1.1 / 1.2 / 1.3 / 1.4）

- 审查角色：fresh-context adversarial evaluator（不继承写作上下文）
- 审查日期：2026-11-23
- 批次：P1 = 1.1 请求生命周期与 SLO、1.2 Prefill/Decode 算力带宽账【重】、1.3 Roofline 与延迟-吞吐权衡、1.4 采样与解码策略
- 依据：evaluator-brief.md、inference spec §4.1/§4.2/§4.3/§5(Part1)/§6.3、STYLE_GUIDE、R2 style sheet、rubric、inference/references.md、docs/research/inference-w1-sources.md
- 方法：全部公式与数值独立重算；强制重推导独立重推；一手来源自行 fetch（arXiv API/HTML、USENIX PDF、GitHub contents API、hf-mirror、NVIDIA 官方 datasheet PDF、Huawei keynote/hiascend、thinkingmachines.ai）；lint + 篇幅 + flanking + 禁令词 + R2 逐条 + 习题分级机械/判定核对。**只读审查，未改章节、未 commit、未派 subagent。**

## 结论

**Status：PASS with findings（S1=0 / S2=1 / S3=5）。强制重推导 4/4 PASS。**

章节整体质量高：全部公式与数值实例独立重算无误；8 个 arXiv ID↔标题↔第一作者全对；DistServe goodput 定义逐字命中、13B/A100/90% 数值（1.6/5.6/10/3.3/2.1×）全对；Orca/Splitwise/Azure/batch-invariance/top-p/min-p/XGrammar/Outlines 引文与机制全对；V4.1-Flash config 全字段命中；NVIDIA B200/GB200/H200 dense 规格与 sparse 脚注逐字命中；Ascend 950 口径纪律完全忠实；DeepInfer doc 全部被引行号（L31/33/35/47/49/57/61/155/211/271-275/295）逐字命中；引擎 repo 路径全 200（含 SGLang 旧路径 404、vLLM 无 batch_invariant_ops 目录的否定性断言）。唯一实质事实错为 H200 NVL 变体 TDP 注（S2，低影响）；其余为符号/措辞级 minor。无 S1，无需 S1 复审。

## Findings 表

| # | severity | 位置 | 断言（原文） | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | 1.3 §2 L31（SKU 与代际纪律注） | 「H200 NVL 变体（1671 / 3341 TFLOPS sparse、**350-400 W**）勿与 H200 SXM 混」 | NVIDIA 官方 H200 datasheet PDF（resources.nvidia.com hpc-datasheet-sc23 落地页内嵌，dam-cdn b2kt2580…pdf）Technical Specifications 表 H200 NVL 列：「Max Thermal Design Power (TDP)：**Up to 600W (configurable)**」。1671/3341 sparse 值正确，但 350-400 W 与官方 datasheet 不符（350-400 W 系 H100 NVL 口径，疑似混淆）。research notes L94 亦误植 350-400 W（错误源头在 research pass）。 | 改为「up to 600 W（configurable）」或删去该变体瓦数（H200 NVL 不入本章任何算账，仅「勿混」注）。低影响：变体不参与计算，但 SKU 纪律注本身以精确为务，宜纠正。 |
| F2 | S3 | 1.2 §1 L19 vs 1.3 §1 L11/13 + §4 式(5) L123/126 | 1.2「$N=7.50\times10^9$ matmul 口径（FLOPs）、$N_{\mathrm{full}}=8.03\times10^9$ 存储口径」；1.3「$N\approx8.03\times10^9$」并以 $2NT_{\mathrm{in}}/F_{\mathrm{peak}}$ 算 prefill | spec §4.3 符号表 $N$（总）。1.3 用 $N$=总(8.03e9) 合规；1.2 用 $N$=matmul(7.50e9) 偏离「总」语义（沿 models 1.2 §9 FLOPs 口径，已声明）。1.3 式(5) prefill 项用存储 $N$（embedding 查表本不计 FLOPs，高估约 7%）且略去二次项（$T_{\mathrm{in}}$=4096 时约 −6.7%），两误差相消 → 66.5 ms 恰与 1.2 含二次项的 matmul 账一致。**数值结果正确，$W$=1.606e10 跨章一致**，纯记号/口径清晰度问题。 | 1.3 §4 式(5) 加一句口径注（prefill 取参数项近似、$N$ 用总参数量、略二次项，与 1.2 matmul+二次项口径互为近似）；或全 Part 统一 $N$ 语义。**P2 须知**：1.2 FLOPs 用 $N$=7.50e9(matmul)、字节用 $N_{\mathrm{full}}$=8.03e9(storage)；1.3 用 $N$=8.03e9。 |
| F3 | S3 | 1.3 §1 L13 / §4 / §6 L196 vs §5 L160-172 | $W$ 在 §1/§4/§6 = 权重字节（16.06 GB），在 §5 = 逗留时间 sojourn（0.5 s） | 章内 $W$ 重载。§5 符号表声明「$W$=逗留时间」并有命名口径注（区分 $W$ vs $W_q$，A-S3-8），但**未声明与 §1-4 权重字节 $W$ 不同义**；§6「$B\cdot K+W\le$显存（§4）」又回到权重 $W$。$W$ 非 §4.3 保留字母，§5 自标抽象口径、§6 带（§4）回指，歧义被缓解但未消除。 | §5 加一句局部记号注（「本节 $W$ 为逗留时间，与 §1–4 权重字节 $W$ 不同义」）或改用 $W_{\mathrm{soj}}$ 之类避免重载。 |
| F4 | S3 | 1.3 §4 L136 vs 1.4 §1 L9/17 | 1.3 $\tau$=TPOT SLO 上限；1.4 $\tau$=温度 | 跨章 $\tau$ 复用，两章均章首局部声明（1.4「温度记 $\tau$（非序列长 $T$）」）。$\tau$ 非 §4.3 保留字母；温度 $\tau$ 为采样文献通用记号。属可接受的章局部约定，仅提示跨章阅读 momentarily 歧义。 | 可选：1.4 口径注补一句「$\tau$ 为温度，与 1.3 TPOT SLO 上限 $\tau$ 不同章不同义」；或不改（均已声明）。 |
| F5 | S3 | 1.2 §3 L91（KV 写边界注） | 「每步每序列只 append $\kappa L$ 字节（128 KiB），比读 $\kappa L T$（$T$=4096 时 0.537 GB）**小两个数量级**」 | 读/写 = $\kappa LT/(\kappa L)=T=4096$≈$4\times10^3$（约 3.6 个数量级），非「两个数量级」（$10^2$）。结论「流量账由读主导」正确，仅量级措辞低估。 | 改「小约 $T$ 倍（$T$=4096 时逾三个数量级）」或「小三个数量级以上」。 |
| F6 | S3 | 1.4 §7 L176（DeepInfer doc 引文） | 引号内「DeepSeek Recipe handles this encoding and decoding efficiently and accurately... **preventing** special-token injection while reducing tokenization latency for long inputs」 | doc L35 原文：「…**DeepSeek Recipe** handles this encoding and decoding efficiently and accurately; by segmenting on special tokens and encoding the segments in parallel, it **prevents** special-token injection while reducing tokenization latency for long inputs.」省略号合法，但省略后动词由 prevents 改为 preventing，非严格逐字（research notes L111 同误）。语义全保。 | 引文恢复「it prevents」或改为非逐字转述（去引号）。极 minor。 |

>  severity 裁定说明：F1 为可一手证伪的局部事实错（brief S2=局部事实错），虽处「勿混」旁注、变体不入算账，仍按 S2 计；F2–F6 均为记号/措辞/口径清晰度级（结果正确或已声明），按 S3。F2 的 $N$ 口径虽涉 §4.3 保留字母，但两章均显式声明、$W$ 一致、数值结果正确，故按 S3（清晰度）而非 S2（实质冲突）；fix round 如按「保留字母跨章一致」从严可升 S2，附此裁定通道。

## 强制重推导（独立重推，不信任 writer）— 4/4 PASS

### R1 临界 $B^{\ast}$ 存在条件与临界值（1.2 §4）— PASS
独立重推：$I(B,T)=\dfrac{Bf}{W+B\kappa LT}=\dfrac{f}{W/B+\kappa LT}$。$W,\kappa,L,T,f>0$ 时 $B\uparrow\Rightarrow W/B\downarrow\Rightarrow$ 分母$\downarrow\Rightarrow I$ 严格$\uparrow$；$B\to\infty$ 时 $I\to f/(\kappa LT)=I_{\mathrm{sat}}$（上确界，有限 $B$ 不达）。连续严格增函数取值 $R$ 当且仅当 $I(1)\le R<I_{\mathrm{sat}}$；真实硬件 $R\gg I(1)\approx1$，故存在性按 $I_{\mathrm{sat}}>R$ 读（命题 2 边界注正确）。解 $I(B^\ast)=R$：$B^\ast f=RW+RB^\ast\kappa LT\Rightarrow B^\ast(f-R\kappa LT)=RW\Rightarrow B^\ast=RW/(f-R\kappa LT)$（式7），分母$>0\Leftrightarrow I_{\mathrm{sat}}>R$，自洽。边界 $f(T)=R\kappa LT\Rightarrow 2N=LT(R\kappa-4d_{\mathrm{model}})\Rightarrow T_c=2N/[L(R\kappa-4d_{\mathrm{model}})]$（式9）。
数值独立重算（8B bf16 H100，$R$=989.5/3.35=295.37，$\kappa$=4096，$2N$=1.50e10，$W$=1.61e10）：
- $T_c$：$R\kappa$=1.2098e6，$-4d_{\mathrm{model}}$(16384)=1.1935e6，$\times L$(32)=3.819e7，$T_c$=1.50e10/3.819e7=**392.7≈393** ✓。
- 表：满窗 $f$=8.37e10、$\kappa LT$=1.72e10、$I_{\mathrm{sat}}$=**4.87≈4.9** ✓；$T$=4096 $I_{\mathrm{sat}}$=1.71e10/5.37e8=**31.8≈31.9** ✓；$T$=256 $I_{\mathrm{sat}}$=**451**、$B^\ast$=295×1.61e10/(1.513e10−295×3.355e7)=**908** ✓；$T$=128 $I_{\mathrm{sat}}$=**898**、$B^\ast$=**469** ✓。
- H200 $R$=206.15 → $T_c$=**566≈567** ✓。吞吐上界 BW/($\kappa LT$)：满窗 **195**、$T$=4096 **6240** tokens/s ✓。容量交叉核对 $B^\ast$=908×3.36e7=30.5 GB+16 GB=46.5 GB<80 GB ✓。
- 精度口径注：权重+KV 同降 FP8 → $\kappa$减半($I_{\mathrm{sat}}$×2)、$R$×2，存在条件两边×2 不变 ✓；$T_c$ 的 $R\kappa$ 乘积不变 → $T_c$ 不变（练习3c 锚正确）✓。

### R2 MoE 每 expert batch $Bk/E$ 与 EP 度无关（1.2 §5 命题3）— PASS
独立重推：$B$ 全局 token 各选 $k$ routed expert → 每步 $Bk$ 个 (token,expert) 分配对；均衡路由按对称性均摊 $E$ 个 expert → 每 expert $Bk/E$。EP 度 $e$ 只决定物理放置（每卡 $E/e$ 个），不改路由分配对的产生与摊分 → $B_{\mathrm{exp}}=Bk/E$ 与 $e$ 无关 ✓。边界：误读 $B$ 为本地 batch 差 $d$ 倍 ✓。$I_{\mathrm{exp}}=2B_{\mathrm{exp}}/\mathrm{bytes}=2Bk/(E\cdot\mathrm{bytes})$（式12），稀释 $k/E$ ✓。
数值：V3 $k$=8/$E$=256 → $B_{\mathrm{exp}}=B/32$，$B$=1024→32 ✓；$I_{\mathrm{exp}}$(fp8,$B$=1024)=2×1024×8/256=**64** ✓；进 ridge $B=RE/(2k)$=590.75×256/16=**9452** ✓。V4.1 $k$=6/$E$=384 → $B/64$，$B$=2048→32 ✓。两 regime 式(10) $\min(Bk,E)$：$B$=8→$Bk$=64<256，routed≈64×44.0MB×58=**163 GB**≈654/4 ✓，每激活 expert≈1 token、$I\approx2$(fp8) ✓。

### R3 Wide-EP 每卡 MoE 流量 $\propto1/e$（1.2 §6 命题4）— PASS
独立重推：每卡持 $E/e$ expert；$B_{\mathrm{exp}}\ge1$ 保证本地 expert 全激活 → 流量=持有数×单 expert 字节×MoE 层数=$(E/e)W_{\mathrm{exp}}\mathrm{bytes}L_{\mathrm{moe}}$（式13），$\propto1/e$ ✓。每卡分配对=$(E/e)(Bk/E)=Bk/e$，FLOPs=$2(Bk/e)W_{\mathrm{exp}}L_{\mathrm{moe}}$，除以式(13) → 强度=$2Bk/(E\cdot\mathrm{bytes})$，$e$ 对消 ✓。$e$ 卡合计=$EW_{\mathrm{exp}}\mathrm{bytes}L_{\mathrm{moe}}$ 与 $e$ 无关（总流量不变、减每卡）✓。
数值：routed 654 GB（=256×44.04M×58×1B，与 models 5.2 §3.3 的 656.5B 含 shared 差一项 ✓）；$e$=8/64/128 → **81.7/10.2/5.11 GB**，读时间 **24.4/3.05/1.52 ms**，强度($B$=2048,fp8)=**128** 不动 ✓。激活条件 $B\ge E/k$=32 ✓。

### R4 强度$\propto B$ recap 忠实性 + 饱和 caveat（对 models 1.2 §10 / 2.4 §6）— PASS
读 models 原文核对：models 1.2 §10「把 $B$ 条序列…每步算 $2NB$ FLOPs、权重仍只读一遍，算术强度升到 $\approx B$」；models 2.4 §6 式(12) $I_{\mathrm{decode}}(t,B)=B(2N+4Ltd)/(2N+B\cdot2LH_{kv}d_{\mathrm{head}}tb)$，「$B$ 小…$I\approx B$；$B$ 大…饱和于 $(2N+4Ltd)/(2LH_{kv}d_{\mathrm{head}}tb)$，8B 满窗≈4.9」。1.2 §1 recap「$I\approx B$…原样沿用不重推」+ 饱和上界 $(2N+4LTd_{\mathrm{model}})/(\kappa LT)$≈4.9 + 「batch 只能在两端点间移动、不能突破上界」caveat **忠实** ✓。符号翻译（$d\to d_{\mathrm{model}}$、$t\to T$、$b\to\mathrm{bytes}$）合规；KV 容量 recap 指针 **models 2.4 式(9)**（非式5）正确 ✓。数值落点：分子 1.50e10、权重 1.61e10、满窗 KV 1.72e10 与 models 算例一致 ✓。

## writer 上报待核项裁定（独立，不照单全收）

- **W-A（「Increasing EP raises arithmetic intensity」归 DeepInfer doc）：成立，引文逐字命中。** 自行 fetch doc（contents API，main，372 行）：**L295 逐字**「Increasing EP raises arithmetic intensity to avoid being memory-bound. Once memory bandwidth is no longer the bottleneck, a larger EP size introduces additional communication and synchronization overhead.」（位于「Why might Mega MoE performance stop improving as EP grows beyond 32 during decode?」答）。research notes 未逐字收此句，但 doc 确有此逐字句 → attribution 成立。1.2 §5 reconciling 注的机制解释（#DP=#EP 耦合扩全局 $B$ → 强度升，EP 本身不改强度）与 doc L29（EP increases batch size）+ L33（#DP=#EP）+ L295 一致；§6「$e$ 增大通信/同步主导、边际递减」对应 L295 第二句 ✓。**裁定：PASS（quote verbatim + mechanism faithful）。** 建议（可选）：reconciling 注补 L295 行号锚。
- **W-B（三方权衡 TTFT↑ 按 static-batch head-of-line 基线，continuous batching 消去后 $\rho$ 支配）：合 spec 1.3 意图。** spec 1.3 要求「$B\uparrow$→吞吐↑、TPOT↑、TTFT排队↑ 的推导与 Operating point」，未限定 batching 基线。1.3 §4 在**显式声明的 static batching 口径**下推 $C(B)=a+bB$、TTFT排队≈$C(B)/2$（三项同时单调），随即注明 continuous batching（4.1）消去 head-of-line、chunked prefill（4.3）消去 decode stall、TTFT排队改由 $\rho$ 支配（§5）。口径注（R2-7）齐全，且给出「为什么现代引擎不用 static batching」的账面答案。**裁定：PASS（忠实且比朴素「TTFT 恒↑」更严谨）。**
- **B200 SKU 纪律：PASS。** Blackwell architecture datasheet PDF（dam-cdn 01dyp27s…）Individual Blackwell GPU Specifications 表第三列（HGX B200）：FP4 18 PF sparse→**9000 dense**、FP8 9 PF→**4500**、BF16 4.5 PF→**2250**、TF32 2.2 PF→**1100**、FP32 **75**、**180 GB HBM3E / 7.7 TB/s**、NVLink5 **1.8 TB/s**、TDP **Configurable up to 1,000 W**；脚注逐字「Specifications in sparse. Dense is one-half of the sparse spec shown.」✓。GB200 列 186 GB/8 TB/s、FP4 20 PF sparse→10 PF dense（对照注，不入表混算）✓。Hopper（H100/H200）datasheet 无 FP4 行 ✓。B300/Rubin 代际分离 ✓。**唯一偏差见 F1（H200 NVL 瓦数）。**
- **Ascend 950 精度口径披露：PASS。** keynote（huawei.com hc-xu-keynote-speech）逐字「deliver 1 PFLOPS in FP8, MXFP8, and HiF8, and 2 PFLOPS in MXFP4」「2 TB/s interconnect bandwidth, which is 2.5 times higher than the Ascend 910C」「HiZQ 2.0…delivering 144 GB of memory and 4 TB/s」「granularity of memory access from 512 bytes to 128 bytes」「950PR…prefill and recommendation…first quarter of 2026」「950DT…decode and training…fourth quarter of 2026」「8,192 Ascend 950DT chips…8 EFLOPS in FP8 and 16 EFLOPS in FP4…16 PB/s」——**均无 dense/sparse 标注** ✓。hiascend 产品页逐字「最大配备128GB、1.6TB/s高速片上内存」「支持多种精度格式，最大支持1784TFLOPS」（精度未注明）「支持灵衢2.0互联协议，最大支持2TB/s双向互联带宽」✓。章节「只逐字引、不推定口径、不算 Ascend ridge、芯片级 vs 卡级分离、媒体数不作一手」**完全忠实** ✓。
- **M/M/1 $W$=sojourn：PASS。** 独立重推：$p_n=(1-\rho)\rho^n$、$L=\rho/(1-\rho)$、Little $W=L/\lambda=1/(\mu-\lambda)$（sojourn=排队+服务）、$W_q=W-1/\mu=\rho/(\mu-\lambda)$ ✓。数值 $\mu$=10/$\lambda$=8：$W$=0.5/$W_q$=0.4/服务0.1 自洽 ✓；$\lambda$=9.5：$W$=2（翻4倍）✓。Kingman $W_q\approx\frac{\rho}{1-\rho}\frac{c_a^2+c_s^2}{2}E[S]$，$c_a=c_s=1$ 退化式(9) ✓；$\rho$=0.8/$c_s^2$=4 → $W_q$=1.0 s（2.5×）✓。余量 $\rho\le q/(q+E[S])$ ✓。命名口径（$W$ sojourn vs $W_q$ wait，Kingman 近似 $W_q$）合 spec A-S3-8 ✓。

## 跨章一致性 + P2 对齐须知

- **§4.3 保留字母全 batch 一致**：$B$=batch（1.2 §5 起升级为全局 batch，显式声明）、$b$=block size（1.2 声明本章不用、bytes 不占字母）、$\mathrm{bytes}$ 文字记法、$d_{\mathrm{model}}/d_{\mathrm{latent}}/d_{\mathrm{rope}}$、goodput $\mathcal{G}$、$\lambda$=到达率（1.1/1.3 均声明「非 GAE」）、$k$=MoE top-k（1.4 top-k 用 $k_{\mathrm{top}}$ 区分）、$\alpha_{\mathrm{freq/pres/len}}$ 下标区分投机 $\alpha$ —— **均一致** ✓。$\gamma$（draft）P1 未用。
- **attention 系数（P2 2.2 对照锚）**：1.2 prefill 二次项 **$2LT^2d_{\mathrm{model}}$**（每层 $2d_{\mathrm{model}}T^2$，causal 平均可见长 $T/2$；双向系数回到 4）；decode attention **$4LTd_{\mathrm{model}}$**（每层 $4Td_{\mathrm{model}}$，$QK^\top$+$PV$ 各 $2Td_{\mathrm{model}}$，对全 $T$ 缓存）。**P2 2.2「每层 $2\cdot d_{\mathrm{model}}\cdot T^2$」与 1.2 prefill 每层系数 2 同阶同系数，对齐** ✓。
- **$N$ 口径（P2 须知，见 F2）**：1.2 FLOPs 用 $N$=7.50e9(matmul)、字节用 $N_{\mathrm{full}}$=8.03e9(storage)；1.3 用 $N$=8.03e9(总)。$W$=1.606e10 跨章一致。P2 recap models 2.4 式(9) 用 $\kappa LTB$ 时注意此口径差。
- **$\kappa$ 几何**：1.2 式(3) MHA/GQA $2H_{kv}d_{\mathrm{head}}\mathrm{bytes}$、MLA $(d_{\mathrm{latent}}+d_{\mathrm{rope}})\mathrm{bytes}$（无因子2、无 $H_{kv}$）、DSA +indexer K —— 与 models 2.4 §5 / 4.1 §4.2 / 5.2 §3.3/§4 recap 一致（576 元素、656.5B、144MB/9.2GB 全核对）✓。P2 2.1/2.3 沿用 $\kappa LT$ 输入。
- **recap 指针**：models 2.4 式(9)（KV 容量，非式5）、式(10)-(12)（强度）、models 1.2 §10（$\propto B$）、models 5.2 §3.3（MoE 块）—— 指针全正确 ✓。
- **章脊口径跨章**：1.1 教学数值（1024/512/TTFT0.5/TPOT30ms）、1.2 三口径（8B/V3/V4.1）、1.3（8B@H100 $T$=4096）、1.4（5-token 玩具分布）各章锁定且换口径声明 ✓。$W$/$\tau$ 重载见 F3/F4（非保留字母，局部声明）。
- **跨维度指针 errata（观察，非 P1 finding）**：models 2.4 §6 末句仍「roofline 的逐卡参数属 gpu 维度」，spec §2 三层裁定已重判归 inference 1.3——errata 通道待收窄（W0 盘点项，不阻塞 P1）。

## 教学 / 规范审查小结

- **教学**：推导无断步（≥4 步推导均有路线句 R2-5）；命题-证明骨架完整（1.1 命题1、1.2 命题2/3/4、1.3 吞吐饱和命题、1.4 nucleus最小性/penalty不可交换/softmax平移不变/Gumbel-max等价/约束保持/浮点非结合，留作练习者对应练习均存在）；符号表齐全；定义配正例+边界反例（R2-6）；无循环解释；前置缺口无（1.1 用 Little 定律为声明的给定工具，1.3 补排队论地基，依赖序合规）。习题校验锚均可校验（正文数值/config 字段/守恒关系）。
- **习题分级**：四章均 [10]/[20]/[30]/[30] 非递减、≥1 题 [10]/[20]、无 [40]、每题校验锚、≥1 题以正文命题为证明目标（1.1 练2→命题1、1.2 练3→命题2、1.3 练2→饱和命题/练4→式8-9、1.4 练2→平移不变）✓。
- **规范**：lint 四章全 LINT OK；篇幅 1.1≈4030 / 1.2≈5749（【重】≥5500 软目标达）/ 1.3≈4102 / 1.4≈4296 CJK（≥3500）✓；rubric A 禁令词 0 命中、B 实质问题未见；无历史叙事（论文仅 citation）；术语首现「中文（English）」+ 判断句定义（R2-8）；证据等级标注齐全；flanking/反转义/`<`in-math 合规（lint + 手工核 backtick 保护行，1.3/1.4 plain 形 0 违例）；总结表回指列齐全（R2-12）；节首 crux/设问/数值开场（R2-1）逐节合规；坑注/提点块引用（R2-10/11）齐全。内部过程引用（spec §4.3 / R2-2 / 四路审查 A-S3-8）有 training 先例（training/02-optimizers/05），属本书 provenance 语域，不判违规。

## Verification Log（抽查项 × 来源 × 结论 × 覆盖率）

| 抽查项 | 来源（自行 fetch）| 原文引句 / 数值 | 结论 |
| --- | --- | --- | --- |
| 8 arXiv ID↔标题↔一作 | export.arxiv.org/api/query?id_list=1904.09751,2407.01082,2411.15100,2307.09702,2412.19437,2309.06180,2311.18677,2401.09670 | 2411.15100 XGrammar/Dong；2309.06180 PagedAttention/Kwon；2311.18677 Splitwise/Patel；1904.09751 Curious Case/Holtzman；2412.19437 V3/DeepSeek-AI；2401.09670 DistServe/Zhong；2407.01082 Min-p/Nguyen；2307.09702 Guided Generation/Willard(2 auth) | 8/8 ✓ |
| DistServe goodput 定义逐字 | arxiv.org/html/2401.09670v3 | "maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query." | 逐字 ✓ |
| DistServe 13B/A100/90% 数值 | 同上 | "about 1.6 requests per second"；"5.6 rps for the prefill phase and 10 rps for decoding"；"2 GPUs for prefill and 1 GPU for decoding…overall goodput of 10 rps, or equally 3.3 rps per GPU, which is 2.1x higher"；"Under the SLO attainment of 90%"；Fig1 "13B…one NVIDIA 80GB A100" | 1.6/5.6/10/3.3/2.1×/90%/13B/A100 全 ✓；min(2×5.6,10)=10、1P+2D→5.6 prefill-bound ✓ |
| Orca iteration-level 引文 | usenix.org/system/files/osdi22-yu.pdf（pdftotext）| "schedules execution at the granularity of iteration (instead of request)"；selective batching；36.9× | 逐字 ✓ |
| Splitwise Azure trace | arxiv.org/html/2311.18677v2 | "production traces taken from two Azure LLM inference services on November 11th 2023"；"coding and conversation"；"released a subset…Azure/AzurePublicDataset"；"characterization are 20 minutes long…arrival time, input size…output size" | 逐字 ✓ |
| Azure schema/许可/GDPR | api.github.com Azure/AzurePublicDataset/contents/AzureLLMInferenceDataset2023.md?ref=master | "ISCA 2024 paper 'Splitwise…'"；"CC-BY Attribution License"；TIMESTAMP/ContextTokens/GeneratedTokens；data/AzureLLMInferenceTrace_{code,conv}.csv；"Due to…GDPR…do not have visibility into the content of the prompts" | 全 ✓（.md 文件非目录）|
| batch-invariance 博客 | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | "The original sin: floating-point non-associativity"；"concurrency + floating point hypothesis"；"output to depend on the batch size"；"reduction order…fixed regardless of the batch-size"；"Horace He…Sep 10, 2025"；batch-invariant RMSNorm/matmul/attention | 全 ✓ |
| V4.1-Flash config | hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json | 40L/hidden5120/kv_heads1/head_dim512/qk_rope64/q_lora1280/o_lora1024/o_groups8/384routed+1shared/top6/sw128/index_topk512/index_n_heads32/index_head_dim128/nextn3/vocab129280/max_pos1048576；quant fp8/block[32,32]/ue8m0/expert_dtype fp4；kv_lora_rank absent（head_dim512 占位）| 全字段 ✓；$\kappa$=576·bytes、误代 1024 高估 1.78× ✓ |
| NVIDIA B200/GB200 dense + 脚注 | dam-cdn.nvd.orangelogic.com/AssetLink/01dyp27s6aj4e0483cs5goo15354m438.pdf（pdftotext -layout）| Individual Blackwell GPU Spec：FP4 18PF*/FP8 9PF*/BF16 4.5PF*/TF32 2.2PF*/FP32 75/180GB HBM3E/7.7TB/s/NVLink5 1.8TB/s/TDP up to 1000W；GB200 186GB/8TB/s/FP4 20PF*；"Dense is one-half of the sparse spec shown."；NVLink domain 130 TB/s | dense=半 sparse 全 ✓；SKU 分离 ✓ |
| NVIDIA H200 SXM/NVL | dam-cdn b2kt2580e5b6424if22lg65223m6ghjb.pdf（H200 datasheet）| H200 SXM：BF16 1979*/FP8 3958*/TF32 989*/FP32 67/141GB/4.8TB/s/700W/NVLink900；H200 NVL：BF16 1671*/FP8 3341*/**TDP up to 600W**；"With sparsity." | SXM dense 989.5/1979/494.5 ✓；**NVL 600W ≠ 章节 350-400W → F1** |
| NVIDIA H100 SXM | H200 datasheet 对照（"nearly double the capacity of H100"=141≈2×80、"1.4X more memory bandwidth"=4.8/3.35）+ research notes W0/W1 官方 PDF | 989.5 BF16 dense/1979 FP8 dense/3.35TB/s/80GB（H200 同算力）| 间接证实 ✓（未单独取 H100 PDF；值业界标准且 H200 datasheet 佐证）|
| Ascend 950 keynote | huawei.com/en/news/2025/9/hc-xu-keynote-speech | "1 PFLOPS in FP8, MXFP8, and HiF8, and 2 PFLOPS in MXFP4"；"2 TB/s…2.5 times…910C"；"144 GB…4 TB/s"（HiZQ2.0）；"512 bytes to 128 bytes"；950PR prefill+recommendation 2026Q1 / 950DT decode+training 2026Q4；"8,192…8 EFLOPS FP8…16 EFLOPS FP4…16 PB/s"；无 dense/sparse 标注 | 逐字 ✓；口径未注明 ✓ |
| Ascend 950PR hiascend | hiascend.com/hardware/processor | "最大配备128GB、1.6TB/s高速片上内存"；"支持多种精度格式，最大支持1784TFLOPS"；"支持灵衢2.0互联协议，最大支持2TB/s双向互联带宽" | 逐字 ✓；精度未注明 ✓ |
| top-p 定义 | arxiv.org/html/1904.09751v2 | "top-p vocabulary V^(p)…as the smallest set such that Σ P(x)≥p"（式2）| ≥p + smallest set 逐字 ✓；nucleus 最小性命题 ✓ |
| min-p 定义 | arxiv.org/html/2407.01082v8 | "base probability threshold p_base∈(0,1]…p_scaled=p_base×p_max"（式1）| $\eta$=p_base、$\ge\eta p_{\max}$ ✓ |
| XGrammar / Outlines 机制 | arXiv API summary | XGrammar "dividing the vocabulary into context-independent tokens that can be prechecked and context-dependent tokens…interpreted [runtime]"；Outlines "transitions between…finite-state machine…index over…vocabulary…little overhead" | 自适应 mask / FSM index ✓ |
| DeepInfer doc 被引行 | api.github.com deepseek-ai/DeepInfer/contents/doc/inference_deployment_qa.md?ref=main | L31 TBT；L33 #DP=#EP；L35 Recipe；L47 prefill EP2–8/DP2–8；L49 decode EP64–128；L57 prefill 理由；L61 Wide-EP traffic；L155 "roughly 80% of a perfectly balanced offline benchmark"；L211 TTL 72h/7d；L271/273/275 主KV per-16 e4m3 / indexer per-32 ue8m0 MXFP4；**L295 "Increasing EP raises arithmetic intensity…"** | 全部逐字 ✓（W-A 成立）|
| 引擎 repo 路径 | api.github.com contents（main）| vllm model_executor/determinism/{batch_invariant,batch_invariant_configs}.py 200；vllm/v1/sample 200；vllm/benchmarks/{serve,latency,throughput}.py 200；vllm/entrypoints/cli/benchmark 200；sglang python/sglang/benchmark/serving.py 200、srt/layers/sampler.py 200、srt/batch_invariant_ops/batch_invariant_ops.py 200；**sglang bench_serving.py 404**、**vllm determinism/batch_invariant_ops 404**；DeepInfer v4_modules/attn.py 200（envs.DEEPINFER_BATCH_INVARIANT L89/666/1002）| 全 ✓（含否定性断言）|
| vLLM ShareGPTDataset | api.github.com vllm/benchmarks/datasets/datasets.py | "class ShareGPTDataset"；"if self.dataset_path is None: raise ValueError('dataset_path must be provided')" | 无硬编码 URL ✓（datasets.py 现处 datasets/ 子目录，章节「datasets.py」指代仍成立）|
| genai-perf phased out | api.github.com triton-inference-server/perf_analyzer/contents/genai-perf/README.md | "GenAI-Perf is being phased out…please use AIPerf instead"（ai-dynamo/aiperf）| 逐字 ✓ |
| ShareGPT 数据集存在 | hf-mirror.com/api/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered | HTTP 200 | ✓ |
| deepseek-recipe repo | api.github.com deepseek-ai/deepseek-recipe + README.md + docs/streaming.md | "collection of Rust libraries and Python bindings"；Streaming response；Chat Completions/Responses/Messages；special tokens | repo/Rust/多协议/流式/special-token ✓；**增量 KMP 算法具体未 source 级核实（7.4 走读项，核心机制经 doc L35 证实）** |
| models recap（2.4 §5/§6、1.2 §6/§9/§10、5.2 §3.3/§4）| 本地 models 章原文 | 2.4 式(9) $2LH_{kv}d_{\mathrm{head}}TBb$、式(12) 饱和 4.9；1.2 §10 $\approx B$、§9 N=7.50e9(FLOPs)/8.03e9(总)/6.98e9(层)、$T^\ast$=5.3e4、6.7%/36%/70%；5.2 §3.3 单expert 3×7168×2048≈44.0M、58×257×44.0M≈656.5B、§4 DSA 144MB/9.2GB | recap 忠实 ✓（R4）|
| 全部公式数值独立重算 | 本地重算（python/手算）| 1.1 TPOT/E2E/百分位/命题1/Little；1.2 式1-13 全部数值（6.58e13/66.5ms/1.71e10/几何表/4.9/31.9/451/898/908/469/393/567/195/6240/654GB/163GB/64/9452/81.7-10.2-5.11GB/24.4-3.05-1.52ms/128）；1.3 ridge 表/t_step/X/C(B)/B_max/三卡表/M-M-1/Kingman/余量；1.4 softmax/top-p/min-p/renorm/penalty反例/log_softmax/Gumbel积分/beam/fp32反例 | 全 ✓（除 F5 量级措辞）|
| lint / 篇幅 / flanking / 禁令词 / 习题分级 | scripts/lint-chapter.sh + python 扫描 + 手工核 | 四章 LINT OK；CJK 4030/5749/4102/4296；rubric A 词 0；flanking 手工核 backtick 行无违例；习题 [10]/[20]/[30]/[30] 非递减+校验锚+命题闭环 | 全 ✓ |

**覆盖率**：公式与数值实例 100% 独立重算；强制重推导 4/4 独立重推；被引一手来源 ~95% 自行 fetch 逐字核对（H100 精确 PDF 未单独取，经 H200 datasheet 对照 + research notes 官方 PDF 记录佐证；recipe 增量 KMP 算法具体未 source 级核实，属 7.4 走读项，核心经 doc L35 证实）。8/8 arXiv ID、全部被引 doc 行号、全部引擎 repo 路径（含否定性断言）核实。无「无 log 的确认无误」。

## 修复建议优先级
- S2 F1（H200 NVL 瓦数）：必修复（改 600W 或删瓦数）。
- S3 F2–F6：累计下波清零；F2（$N$ 口径）/F3（$W$ 重载）建议本波顺手加口径注（ foundational 章，清晰度收益高）；F4/F5/F6 可选。
- 含 S1=0，无需 S1 复审。writer 反驳通道：附证据复核（尤其 F2 severity 裁定）。
