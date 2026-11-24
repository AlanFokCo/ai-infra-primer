# gpu W1 · Part 1 深度对抗审查（T6）— GPU 微架构 1.1–1.4

- 日期：2026-11-25
- 审查人：fresh-context adversarial evaluator（不继承写作上下文；未派子代理；全部一手来源自行 curl/fetch + 独立 Python 重推导）
- 范围：`gpu/01-gpu-microarchitecture/01-chip-organization-and-generations.md`（1.1）、`02-sm-internals.md`（1.2【重】）、`03-tensor-core-mma.md`（1.3）、`04-instruction-pipeline-latency-hiding.md`（1.4）
- 输入：STYLE_GUIDE.md、gpu spec §4.3/§5 P1/§6.3、R2 style sheet（R2-1…12 + 习题分级）、rubric A/B、`docs/research/gpu-w1-sources.md`、`gpu/references.md` P1 条目
- 方法：① 硬件事实逐条对一手来源（dam-cdn 官方 PDF + docs.nvidia.com + arXiv + GitHub API）；② 强制重推导 1.2 occupancy 物理上界（独立 Python，全五案例 + 练习 + register-binding + 粒度方向）+ 1.1/1.3/1.4 全部编号公式与 worked case 数值重算；③ 教学审查（断步/未定义符号/循环/前置/越界/跨章 recap）；④ 规范审查（R2 逐条 + rubric A/B + 模板 + 习题分级 + 数学白名单 + display rescape + 去 AI 味 + 仓库 lint 双脚本）。
- 网络通道：WebFetch 全域名不可用（沿 W1 research 记录）→ 全部 curl；dam-cdn 稳定 200；docs.nvidia.com 重试后 200；arXiv pdf 200（html v2 404 → 改 pdf）。大 PDF 核对 Content-Length 防截断（hopper-wp 7880278 B 全量）。

## 0. 结论摘要

**S1 = 0 ／ S2 = 0 ／ S3 = 6。强制重推导 1.2 = PASS。**

Part 1 四章硬件事实与公式全部经一手来源/独立重算核实，无致命或实质错误；规范（R2-1…12、rubric、模板、习题分级、数学白名单、rescape、lint）全过。6 条 S3 均为措辞/口径注/归属精度/跨章符号消歧类 minor，不阻塞 commit；其中 S3-5 为 T5 遗留 minor 的确认（修复点在 2.4，属 P2 范围，需与 P2 reviewer 协调）。

## 1. Findings 表

severity：S1=致命事实/错误公式；S2=实质约定/断裂推导/局部事实；S3=minor。

| # | sev | 位置 | 断言（问题） | 证据（URL/引句/重算） | 建议 |
| --- | --- | --- | --- | --- | --- |
| S3-1 | S3 | 1.1 §6 陷阱表 row1 + §参考 nvidia-datasheet；1.3 §参考 nvidia-datasheet | 把「Specifications in sparse. Dense is one-half of the sparse spec shown.」当作通用 datasheet 脚注逐字引用；该逐字句**仅见于 Blackwell datasheet**（footnote 2）。H100/H200/HGX datasheet 脚注实为「*With sparsity」/「2. With sparsity.」/「* With sparsity.」，只标 sparse、未逐字印折半规则。读者开 H100 datasheet 找不到所引句。 | H100 ds PDF L151「*With sparsity」；H200 ds L140「2. With sparsity.」；HGX ds L205「* With sparsity.」；Blackwell ds L537「2. Specifications in sparse. Dense is one-half of the sparse spec shown.」（均 dam-cdn 一手）。折半约定本身正确（wp dense 989.4 ≈ 1979/2 互证）。 | 注明逐字折半脚注出自 Blackwell datasheet；H100/H200/HGX 脚注为「*With sparsity」（同约定），折半由 whitepaper dense 值互证。非错误事实，仅归属精度。 |
| S3-2 | S3 | 1.2 §5 末段（L210）；1.3 §1 代际表注（L13）+ §6（L176） | 读者面文本泄漏内部 QA/research 任务标识：1.2「也是 evaluator 重推导的核对点」；1.3「代际映射沿 T1 一手核实」「T1 已核实 cta_group::1/::2 存在与 TMEM 几何」。已收官 inference/training 正文无 evaluator/T1 先例（「spec §4.3」符号引用有先例、可保留）。 | grep inference/0*、training/0* 正文：无「evaluator」、无任务号 T1/T2；inference 1.2/1.4 有「spec §4.3」先例。 | 删/改自足表述：1.2 改「也是重推导核对点」或删；1.3 改「经一手核实」/「PTX 逐字核实」。保留 spec §4.3 符号引用。 |
| S3-3 | S3 | 1.1 §4 逐因子读表（L123） | 「5 栈 80 GB…换成 6 栈 141 GB/4.8 TB/s」把栈数与 usable 容量直接配对；6 栈×24 GB=144 GB raw，141 为 usable。与本章自身 B200 raw192/usable180 纪律不齐。 | H200 ds：141 GB/4.8 TB/s（usable）；GH100 full 6 栈（wp L618）；24 GB/栈（Micron H200 8-high，research §2.8）。6×24=144≠141。 | 改「6 栈（raw 144 GB）usable 141 GB / 4.8 TB/s」，或注「141 为 usable，栈几何 raw 账归 2.1」。 |
| S3-4 | S3 | 1.3 §4 wgmma 段（L119） | 「wgmma shared operand 流 96 B/cycle 恰好落在 shared 带宽预算 128 内、留 25% 余量」隐含假设：TC 描述符读与 LSU shared 访问共享同一 bank 带宽预算且可线性相加。该仲裁细节闭源未披露；128 B/cycle 是 bank 几何峰值（32 banks×4 B）。 | 1.2 §4 bank 几何（PG Table 34：32 banks）；PTX wgmma B 走 shared-desc（逐字）；TC/LSU 仲裁无官方披露。一阶预算检查合理但假设未注。 | 加一句口径注：「128 B/cycle 为 bank 几何峰值；TC 读与 LSU 访问的仲裁属闭源，此处按共享同一预算的一阶上界检查」。 |
| S3-5 | S3 | 2.4 §5 recap（L138）× 2.4 §1 符号声明（L19）【跨章；T5 遗留 minor 确认】 | 2.4 章首声明 $L$=层数，但 §5 recap 写「Little's law $L=\lambda t_{\mathrm{lat}}$」裸 $L$（=1.4 在飞并发口径），未局部重声明 → 2.4 章内 $L$ 双义。1.4 自身 §1 已局部声明 $L$=在飞并发（正确，非 1.4 缺陷）。 | 2.4 L19「$L$ = 层数」；2.4 L138「Little's law $L = \lambda \cdot t_{\mathrm{lat}}$」；2.4 全章无其他裸 $L$ 使用（层数 $L$ 声明后未用）。1.4 L11 局部声明正确。 | 修复点在 2.4（P2 范围）：recap 处改 $L_{\mathrm{req}}$ 或加「$L$ 此处为 1.4 在飞并发口径，本章层数 $L$ 不用」一句。P1 侧无需改；与 P2 reviewer 协调销项。 |
| S3-6 | S3 | 1.3 §4 首句（L119） | 「wgmma 把复用推到极致」用 rubric A2 清单词「极致」作 intensifier。属灰区：该段有数值支撑（96 B/cycle、75%），按 A2「无数值支撑」要件不构成硬违规，但词面命中清单。 | rubric A2 列「极致」；本段随后给 6144 B/64 cycles=96、96/128=75% 数值支撑。 | 可选改「wgmma 把复用进一步推高：B（乃至 A）走 shared 描述符」。保留亦可（有数值支撑）。 |

无 S1/S2。⟨verify W3/W6⟩（B200 SM 数 148 反推、CUTLASS 当前版本、tcgen05 细形状）为 spec 允许的 deferred 项，**不作 finding**（已核实其「反推/官方未披露」标注齐全）。

## 2. Verification log（强制；无 log 视为未审查）

### 2.1 强制重推导 1.2 occupancy 物理上界 — **PASS**

独立 Python（/tmp/t6-p1-review/rederive.py），库存取 CC 9.0 = slot 32 / warp 64 / reg 65536 / shared 233472 B（已另对 PG Table 33/34/35 + wp Table 4 一手核实，见 2.2）。
$B_{\max}=\min(32,\lfloor 64/w_b\rfloor,\lfloor 65536/(R_{th}\cdot threads)\rfloor,\lfloor 233472/S_{sh}\rfloor)$，$W_{active}=B_{\max}w_b$，$\mathcal{O}=W_{active}/64$。

| 案例 | threads/w_b/R_th/S_sh | 四项 (slot,warp,reg,shared) | B_max | W_active | O | 章值 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 256/8/64/0 | (32,8,4,∞) | 4 | 32 | 0.50 | 32/0.50 | ✓ |
| C2 | 256/8/128/0 | (32,8,2,∞) | 2 | 16 | 0.25 | 16/0.25 | ✓ |
| C3 | 256/8/32/48KB | (32,8,8,4) | 4 | 32 | 0.50 | 32/0.50 | ✓ |
| C4 | 32/1/16/0 | (32,64,128,∞) | 32 | 32 | 0.50 | 32/0.50 | ✓ |
| C5 | 1024/32/255/0 | (32,2,0,∞) | 0 | 0 | 0 | 0/不可launch | ✓ |

- active warps 序列 **32/16/32/32/0** 与章 §5 表逐字一致 → PASS。
- register-binding：255×1024=261120 > 65536（3.984×≈3.98）→ reg 项 0 → B_max=0 不可 launch ✓；1024/512/256 线程 R_th 上限 64/128/255（255 封顶），255×256=65280≤65536 ✓。
- 单位一致：四项皆 blocks/SM（slot=blocks；warp=(warps/SM)/(warps/block)；reg=(regs/SM)/(regs/block)；shared=(B/SM)/(B/block)）→ min 合法 ✓；式(4) blocks×warps/block=warps；式(5) 无量纲 ✓。
- 命题 2（必要性+取整+充分性[理想口径]）/命题 3（粒度上界性，实际≤B_max）结构 sound；C3 含 1KB 预留 floor(233472/50176)=4 与理想同值 ✓；§6 例 32KB 理想 7 / 含预留 6（7.125→7、6.909→6）方向正确 ✓。
- 练习 4 重算：(a) 512th/R48/S64KB → (32,4,2,3)=2、W=32、O=0.5 register 绑定 ✓；(b) carveout100KB → shared 1 → B=1、O=0.25 ✓；(c) 128th → (32,16,10,3)=3、W=12 shared 绑定 ✓。C1 灵敏度 R32 → B=8、O=1.0 ✓。
- §3 端口账：bytes_FMA=32×4×4=512 ✓；BW_RF=4×512×1.98e9=4.055e12≈4.06 TB/s/SM ✓；HBM/SM=3352/132=25.39 ✓；shared=128×1.98e9=253.4 GB/s ✓；比 159.7:10.0:1≈160:10:1 ✓；全芯片 132×4.055=535 TB/s、/3.35=160 ✓。

### 2.2 硬件事实抽查 × 一手来源 × 结论（覆盖率：P1 全部披露数值 ~100%）

| 抽查项 | 来源（一手） | 结论 |
| --- | --- | --- |
| H100 SXM5 132 SM/66 TPC；PCIe 114/57；full 144=8GPC×9TPC×2SM | hopper-wp PDF Table 3 + Fig 6（视觉读图） | ✓ 132/114/66/57；Fig6 8 GPC、两 L2 分区、6 栈/12 控制器、18 NVLink |
| 逐精度 boost：TC 类 1830/1620；FP64TC+FP32/FP64 非TC 1980/1755 | wp Table 3 逐字 | ✓ 频率依精度类而异成立 |
| BF16 dense 989.4/sparse 1978.9；FP8 1978.9/3957.8；TF32 494.7/989.4；FP64TC 66.9；FP32 66.9；FP64 33.5；INT8 1978.9/3957.8 | wp Table 3 | ✓（章 INT8 行 1979 取 datasheet dense，表头已声明双源） |
| 舍入路径：dense 真值 132×4096×1.83=989.42976→989.4；sparse 1978.85952→1979→/2=989.5 | 独立重算 + wp/datasheet | ✓ 章 989.43 正确（research notes 的 989.46 为笔误，章无误） |
| 5120-bit HBM3/5 栈/10 控制器/2619 DDR/3352 GB/s/L2 50MB/RF256KB/33792KB/814mm²/80B/4N/700W | wp Table 3 + L618/627 | ✓ |
| CC9.0：64 warps/32 blocks/2048 threads/65536 regs/255/thread/1024/block/228KB | wp Table 4 + PG Table 33/34（colspan 解析） | ✓ |
| CC9.0/10.0 库存同形；carveout {0,8,16,32,64,100,132,164,196,228}；unified 256KB；opt-in 227；banks32；KB=1024；>48KB dynamic opt-in | PG compute-capabilities 子页 Table 33/34/35 逐字 | ✓ 9.0/10.0 逐行同；227/228/256 三口径成立 |
| 单发射：Fig7 每 processing block 1 Warp Scheduler(32 thread/clk)+1 Dispatch Unit+RF16384×32bit+32FP32/16INT32/16FP64/1TC/8LDST/SFU；SM 级 TMA+256KB L1/shared+4Tex | wp Fig 7（pdftoppm 视觉读图） | ✓ 4×1×32=128；双发射不成立 |
| BP Table 5 CC9.0：FP32 128/FP64 64/INT32 add 128/shfl 32/SFU(lg2) 16/DPX 64；列 7.5…12.x | BP HTML Table 5（colspan 解析） | ✓ 与 1.2 吞吐表逐字；INT32 128>64 单元章已注闭源 |
| BP prose「typically 4 cycles…CC 7.0」「hundreds of clock cycles」 | BP HTML 逐字 | ✓ 1.4 两口径声明诚实 |
| 227 opt-in + 1KB reserved 逐字；CC10.0 64warps/32blocks/64K/228/256 | blackwell-tuning-guide 逐字「CUDA reserves 1 KB…up to 227 KB」 | ✓ |
| +39% vs A100 164；Hopper 227 | hopper-tuning-guide 逐字「228 KB, a 39% increase…164 KB」「227 KB」 | ✓ 228/164=1.390 |
| Blackwell 208B/4NP/双 reticle die/NV-HBI 10TB/s 单一致/5th TC FP4+OCP/18 NVLink5 1.8TB/s(双向)/GB200 8TB/s | blackwell-brief PDF 逐字（L188-205/230/332/361） | ✓ NV-HBI≠NVLink-C2C 坑注正确 |
| datasheet：H100 989.5(1979*/2)/3.35/700W/NVL94GB3.9；H200 141/4.8/700W(NVL600)；HGX B200 2250/4500/9000/180/7.7/1000W/FP32 75；GB200 2500/5000/10000/186/8/1200W/FP32 80 | 四份 dam-cdn datasheet PDF | ✓ SKU 三口径（180/186/raw192）成立 |
| PTX mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32；fragment A4/B2/C4/D4 regs；wgmma m64nNk16 N∈{8..256}步8/m64nNk8 tf32/m64nNk32 f8f6f4；A reg-or-shared、B shared-only；commit/wait_group；fence.proxy.async；tcgen05.mma/.sp/.ws + cta_group::1/::2 + kind::mxf4nvf4 | PTX ISA HTML 逐字 | ✓ 1.3 fragment 表/操作数路径/异步三件套全对 |
| TMEM「512 columns and 128 rows per CTA, each cell 32-bits」=256KB/CTA | PTX ISA 逐字（sm_100a/f） | ✓ 章引「rows」逐字正确（references 条目写 lanes 为 paraphrase） |
| Nsight stall：long_scoreboard/wait/barrier/not_selected/mio_throttle/lg_throttle/math_pipe_throttle/no_instructions/drain/dispatch_stall/branch_resolving/imc_miss/membar；Warp Scheduler States eligible/issued 单 warp | Nsight ProfilingGuide HTML | ✓ 1.4 四分类+其他桶全对 |
| Hopper blog 早期「over 3 TB/sec」+「Memory Data Rate (Not finalized for H100)」 | developer.nvidia.com blog HTML | ✓ 版本漂移注成立（章引为压缩形，可接受） |
| 微基准 H800 Table IV：L1 40.7/shared 29.0/L2 263.0/global 478.8；wgmma dense 128/sparse SS 144 | arXiv:2402.13499 PDF Table IV/VIII/IX + 正文 | ✓ 列=RTX4090/A100/H800，章取 H800 列正确 |
| sparse 净加速 1.78：2×128/144=1.778；实测吞吐比 sp SS 1308/dense 729.3=1.793 | 重算 + Table VIII/IX | ✓ 1.78 与实测吞吐比一致（ latency-ratio 捷径在此成立） |
| CUTLASS include/cute/arch/mma*.hpp 存在 | GitHub contents API 200 | ✓ mma_sm70/75/80/100/… 在列 |

### 2.3 其余编号公式/数值独立重算（覆盖率 100%）

- 1.1：c_BF16=989.4e12/(132×1.83e9)=4095.9→4096；c_FP32=256；TF32 2048；FP8 8192；16×0.924=14.79=989.4/66.9 ✓；clock-mix 1070.53/989.5=+8.19%≈8.2 ✓；harvest 144→1079.4/+9.1% ✓；PCIe 756.45/比 0.7645 ✓；B200 反推 FP32 clock 1.98/BF16 1.8558/比 1.067 vs Hopper 1.082/FP4 c 32764≈32768/208SM→1.32 ✓；能效 1.41/2.25/2.08、+59% ✓。
- 1.3：2mnk=4096/262144/524288 ✓；t_issue 4/64、SM 闭环 4096 ✓；latency-as-interval 2048→494.7（低估一半）✓；fragment 字节 512/256/512/512、1792/4096=0.4375、标量 8、比 18.3 ✓；MMA 流 448<512、SM 1792<2048 ✓；wgmma shared 6144/64=96=75%×128 ✓；reg 0.25 B/FLOP ✓；R_acc=N/2、N128→8blocks/32warps/0.5、N256→4/16/0.25 ✓；TMEM 262144B=256KB ✓。
- 1.4：λ_SM=25.39e9/(32×1.83e9)=0.4336 ✓；L_SM=0.4336×478.8=207.6≈208 sectors=6644B≈6.6KB ✓；t=261.6ns、芯片 877KB ✓；52 warp load、O=0.81 ✓；坑 479/52=9.2× ✓；O_min ILP1/2/4=0.812/0.406/0.203 ✓；ex2b f1.98→192 ✓；ex2c shared 116 sectors=29 warp load ✓；ex4a B=2/W=16/minILP4 ✓；ex4b 416/8=52、util 50% ✓。

### 2.4 教学/规范审查结论

- 推导断步/未定义符号/循环解释：无。各章符号表齐、路线句齐、命题-证明骨架齐（留练习者该练习存在）。
- 前置/越界：无 serving/K8S 内容；跨章 recap（1.3←1.2 occupancy、1.4←1.2 C1/C2 + 1.3 wgmma、1.4 Little）数值与源章一致。
- 符号 §4.3：n_b 非裸 B ✓；N=参数量保留、1.3 N=指令字面显式消歧 ✓；bytes 不占字母 ✓；B=batch 不表 tile ✓；1.4 L 局部声明 ✓（跨章冲突见 S3-5，点在 2.4）。
- R2-1…12 逐条：节首问/数/元问题齐；章脊口径齐（1.1 多 SKU 声明）；公式后 300 字全数字实例齐；命题骨架齐；路线句齐；定义配例+反例齐；口径注齐；判断句定义齐；关节词齐；提点/坑块引用齐；总结表回指列齐。
- 习题分级：四章均 [10]/[20]/[20或30]/[30] 非递减、≥1 低档、≤1 [40]、≥1 证命题、每计算题校验锚、先预测句在位。
- 数学白名单 + display rescape（9 禁形）+ flanking + 裸<：`scripts/lint-math-boundary.py` 四章 exit=0；raw grep 无 `\, \! \; \: \{ \} \| \% \#`（数学内）、无 escaped pipe、无裸 `<字母`。
- 去 AI 味 + rubric A/B：lint 词表零命中；「极致」一处灰区见 S3-6（有数值支撑）；无历史叙事/营销形容词/空洞过渡。
- 模板/篇幅：四章 总结/练习/参考 齐、metadata 齐、ASCII 图 ≥1；CJK 正文 1.1=4038、1.2=5791（【重】≥5500 ✓）、1.3=4157、1.4=3801（均 ≥3500）。
- 仓库 lint：`lint-chapter.sh` 四章 LINT OK。

## 3. T5 遗留 minor 裁定

S3-5 确认成立：2.4 章首 $L$=层数 与 §5 recap 裸 $L$（在飞并发）章内双义；1.4 侧局部声明正确、无需改。修复点在 2.4（P2），建议 recap 改 $L_{\mathrm{req}}$ 或加一句口径消歧；与 P2 reviewer 协调销项。P1 不因此阻塞。

## 4. 汇总

S1=0，S2=0，S3=6（S3-1 脚注归属、S3-2 过程泄漏、S3-3 H200 raw/usable、S3-4 wgmma shared 仲裁口径注、S3-5 跨章 L 消歧[2.4]、S3-6 极致灰区）。强制重推导 1.2=PASS（32/16/32/32/0）。Part 1 审查通过，S3 累计下波清零。
