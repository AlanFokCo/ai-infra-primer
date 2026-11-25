# gpu W7 全书一致性审计 + errata/残留/图执行清单（T1，fresh whole-book auditor）

日期：2026-11-26 · 分支 gpu-w7 · BASE ca166b1（W6 收官，41 章/11 Part 满）
审计范围：全 41 章 + 兄弟维 models 51 / training 39 / inference 37 + gpu/README 盘点表 + spec §2/§6.3/§8。
方法：read-only 审计（独立 Python 重算 + /usr/bin/grep -F/-rn + node parse-check）；不派 subagents。
权威映射：gpu/README.md W0 盘点表 L246-340（93 行）；errata 通道 spec §2 L42-56（仅限指针句最小修订，不动公式数值）。

## 0. 汇总

| 项 | 结果 |
| --- | --- |
| S1（致命事实） | **0** |
| S2（实质问题） | **0** |
| S3（minor/口径） | **6**（8.1:98 排列 / 10.3:49 排列 / 9.2:122 typo / 10.4:199 表头 / 1.3:176 verify 残留 / refs deepseekv3 notes stale） |
| 数值脊柱 | **PASS**（H100/B200/NVLink/GDR/MIG/fat-tree/三 critical 全部独立重算闭合，989.4/989.5 双值有显式口径注） |
| recap 链边界 | **PASS**（5 条链全部指向正确源章/节，无重推导，无 CP=5.5/ECC=1.2 误指） |
| refs | **PASS**（parse-check 11 groups/88 entries；独立 forward+reverse cross-check 0 缺口/0 unknown，292 citedBy pairs） |
| glossary | **PASS**（190 术语；「首次实质引入章」列语义声明完整） |
| R2 文风机械抽查 | **PASS**（禁令词全 41 章 0 命中） |
| T2 errata 执行清单 | **86 行仍泛指向未收窄**（82 章级可收窄/重判 + 4 references 级）；另 2 行措辞待裁定 + 1 历史口径句；3 README 边界句 W1 已执行 |
| T3 gpu 残留清单 | **7 项**（1.1 verify×6 行 + 2512.02189 回写 / 排列族×2 / 9.2 typo / 10.4 表头 / 1.3 verify / 根 README 翻转 / deepseekv3 notes） |
| T4 图排列清单 | **2 图 / 6 处**（gpu-p08:144；gpu-p10:375,382,384,385,438） |
| 「README 18→19」re-identify | **未证实为残留**（见 §6.5） |

---

## 1. 审计 findings（S1/S2/S3）

S1 = 0，S2 = 0。以下 S3 全部为口径/措辞/标记残留，数值与公式均正确，不阻塞收官；其中前 4 项与 brief 候选重合（已核实确存在），后 2 项为本审计新发现。

| # | severity | 位置 | 问题 | 建议修 |
| --- | --- | --- | --- | --- |
| S3-1 | S3 | gpu/08-nccl-collectives/01-collective-primitives.md:98 | 「均衡 a2a 是一个排列（每 rank 收发均 (n-1)/n·M）」——均衡 a2a 是**均匀流量（双随机矩阵）**，非排列。与 W5 已修的 10.3 命题 3（L147「均匀流量是双随机矩阵、非排列」）矛盾。W5 同类修定 S3，沿先例。 | 「均衡 a2a 是均匀流量（双随机矩阵，非排列；每 rank 收发均 (n-1)/n·M）」。后半「NVSwitch 非阻塞对任意排列线速」是 crossbar 义，**保留**。 |
| S3-2 | S3 | gpu/10-fabric-and-scale-communication/03-ep-alltoall-implementation.md:49 | 「NVSwitch 非阻塞对均衡排列线速……偏斜路由打破排列均衡」——「均衡排列」措辞与命题 3（L147）不一致。 | 「均衡（均匀）流量」/「流量均衡」；crossbar「任意排列线速」义保留。 |
| S3-3 | S3 | gpu/09-rdma-gpudirect/02-gpudirect-rdma.md:122 | 「省 D2H+D2H 两跳」typo——紧接括注已正确写「两跳 = 发送 D2H + 接收 H2D」，引号内短语误重 D2H。 | 「省 D2H+H2D 两跳」。 |
| S3-4 | S3 | gpu/10-fabric-and-scale-communication/04-point-to-point-sequence-parallel.md:199 | §6 transport 选择表表头「章脊 T_blk = 8192 读法」对 row 1（n=8）不成立：row 1 比值 0.066 = 1079/16384，用的是 n=8 对比行 T_blk=16384（L205 全账、L3 章脊「同一几何」同口径）；rows 2-3（n=16）才用 8192（0.977=8000/8192、0.763=6250/8192）。数值全对，表头口径误标。 | 表头改「章脊读法（n=16 T_blk=8192；n=8 对比行 T_blk=16384）」或 row 1 比值格附注。**勿改 row 1 数值**（与 L205/L3 闭合）。 |
| S3-5 | S3 | gpu/01-gpu-microarchitecture/03-tensor-core-mma.md:176 | ⟨verify W3 PTX/CUTLASS sm100 atom⟩ 标记未销项。标记期望「更细 operand 选项与形状表留 W3 走读销项」，但 W3 明确 scope-out（5.2:15「Blackwell tcgen05 全量走读不在本章……聚焦 SM90」、5.2:188「归 6.3（FA4）与后续版本」）；6.3 §7 只到 TMEM 几何/CuTeDSL kind 分派机制级，未给 cta_group::2 精确形状表。brief 未列此条，属 verify 标记族新发现。 | W7 二选一：(a) 对 PTX ISA 9.5 核实 cta_group::2 operand/形状后销项；(b) 改写为显式 scope 注（「精确形状表非本书交付，沿 PTX ISA tcgen05 节」），去 ⟨verify⟩ 标记。建议 (b)（全书已收官，不新增走读）。 |
| S3-6 | S3 | gpu/references.md:130（deepseekv3 条 notes） | notes 写「6.4 MLA kernel 几何 + 10.3 EP a2a 锚」，但 W5 被引回写已撤 10.3（「deepseekv3 -= 10.3，V3 几何经 training 5.4 recap 非直引」），citedBy 列现仅 6.4。notes 与 citedBy 不一致（stale）。另「EP DualPath」疑为 DualPipe 之误（V3 论文术语）。 | notes 删「+ 10.3 EP a2a 锚」或改「10.3 EP a2a 经 training 5.4 recap 间接」；「DualPath」→「DualPipe」。交 T5 refs 终 sweep（plan T5 Step 2 已列 deepseekv3 notes stale）。 |

**非 finding 的到期残留**（spec 允许延迟项，W7 到期，入 T3 清单不计 finding，沿 W1-W6 先例）：1.1 ⟨verify W3/W6⟩×6 行（B200 SM 148 已经 W6 [B] 销项，正文标记待回写）。

---

## 2. 数值脊柱 verdict：PASS

独立 Python 重算（python3 -I）与全书 grep 交叉核，全部闭合：

| 脊柱项 | 全书口径 | 独立重算 | verdict |
| --- | --- | --- | --- |
| H100 SM | 132（SXM5；GH100 full 144） | 132×4096×1.83e9 = 989.43 TF | ✓ |
| H100 BF16 dense | 989.5（datasheet）/ 989.4（wp Table 3） | 989.43 → 两印刷值 | ✓ 双值有显式口径注（2.4:295「wp 989.4 / datasheet 989.5 为舍入路径漂移」） |
| H100 HBM | 3.35 TB/s（datasheet）/ 3352 GB/s（wp 推导） | 5×16×5.238e9×64/8 = 3352.32 GB/s | ✓ |
| ridge | ≈295（BF16）；fp32 ≈20 | 989.5/3.35=295.37；989.4/3.352=295.17；66.9/3.352=19.96 | ✓（5.1:19/5.3:9 fp32 ridge 口径正确区分） |
| B200 SM | 148 [B]（2 die×74，物理 160） | 75e12/(148×256)=1.98 GHz；2250e12/(148×8192)=1.856 GHz；FP4 c=32768 | ✓ 11.1:227 [B] 口径注完整；1.1 反推自洽（待 T3 回写引 2512.02189） |
| NVLink4 | 900 双向 = 450 单向 = 18×25 | 18×25=450，×2=900 | ✓ 7.1:33/42 方向硬规则 |
| NVLink5 | 1.8 TB/s 双向 = 900 单向 = 18×50 | 18×50=900，×2=1800 | ✓ 7.1:43 |
| 域聚合 | HGX H100 7.2 TB/s；NVL72 130 TB/s | 8×0.9=7.2；72×1.8=129.6≈130 | ✓ |
| GDR min | min(NIC,PCIe)：CX7 50<64 NIC 限；CX8 100>64 PCIe 限 | min(50,64)=50；min(100,64)=64；bounce 串行 (1/64+1/50)^-1=28.07 | ✓ 9.2 式(1)(3) |
| MIG | SM 1/7、mem 1/8（容量+带宽+L2） | H100 1g：132/7=18.9、80/8=10、3350/8=418.75、50/8=6.25；B200 1g：148/7=21.1、180/8=22.5、7700/8=962.5 | ✓ 11.1:191（L2 随 memory slice 1/8，坑注 1g.20gb 反推 2/8 错） |
| fat-tree | 两级 k²/2：144→10368、64→2048；三级 k³/4：64→65536 | 144²/2=10368；10368/72=144 机架 | ✓ 10.1 命题 2，XDR 官方逐字闭合 |
| 三 critical | 8.1 M_c=5.93 MB（ring 内部）≠ 8.3 M*=1.83 MB（ring/tree 算法间）≠ 8.4 C*=667.4 KB（LL/Simple 协议间，对应 M*≈381.4 KB） | M_c=8×2e-6×370.8e9=5.93 MB；M*=8×4×2e-6/(7×4.995e-12)=1.83 MB；C*=1.8e-6×370.8e9=667.4 KB；T_LL=T_Simple=10.2 μs | ✓ 8.4:143 显式「三个临界勿混」口径注；8.1:118/8.3:77 回链 |
| 370.8 有效带宽 | 18×20.6（NCCL SM90_NVLINK_BW） | 18×20.6=370.8 | ✓ 9 处同链 |

跨章不一致：**未发现**。989.4/989.5、3352/3.35 双值均为官方不同文档印刷值，书内显式口径注消歧，非错误。

## 3. recap 链边界 verdict：PASS

| 链 | 核实 | verdict |
| --- | --- | --- |
| 9.2 ← inference 8.3 | 9.2:3 章脊「inference 8.3 KV 传输语义 recap（不重推）」+ 9.2:21 边界重定界声明（语义层↔硬件层分工，共享 4.295 GB/50 GB/s 数值对接）；inference 8.3:194 GDR 源存在 | ✓ 无重推 |
| 10.3 ← training 5.4 + inference 3.3/7.4 | 10.3:15 recap 边界声明逐源列（training 5.4 置换 P/2kd/E[D]/木桶/20SM；inference 3.3 τ(e)/e*=32；7.4 Wide-EP/EPLB/moe_lb；8.1 共享出口）「均已交付……不重推」；10.3:25-29 recap 表逐行指针 | ✓ 无重推 |
| 10.4 ← models 4.6 + training 5.2 | 10.4:13「models 4.6 §3.2 Ring Attention 分块 + training 5.2 §3–§5 CP 账……只 recap + 符号翻译（c→n、s→T、b→B），不重推」；models 4.6:212 §3.2 存在；**CP=training 5.2 非 5.5**（10.4 全章无 5.5 误指） | ✓ 无重推 |
| 11.2 ← models 2.4 | 11.2:21「I/吞吐上限/ridge 推导归 models 2.4 §6（经 2.4 recap，不重推）」；11.2:203 ridge 295 recap | ✓ 无重推 |
| 11.3 ← 2.1 §4 + inference 8.5 + training 7.4 | 11.3:3/21/52/335 全部「ECC SECDED recap **2.1 §4**」（非 1.2）；inference 8.5 冗余 quota/rerouting、training 7.4 checkpoint/MTBF recap + 指引不重推 | ✓ 无重推、无误指 |

符号翻译硬规则（spec §4.3）抽查：8.1:11 d→n、10.4:11 c→n/s→T/b→B、6.3 d_head/M_SRAM 章首声明均在位。

## 4. refs / glossary verdict：PASS

- **parse-check**（site/scripts/lib/references.mjs）：`11 groups / 88 entries` ✓（命令见 §8）。
- **独立 forward+reverse cross-check**（python3 -I，/tmp/w7_refs_crosscheck.py）：88 entries / 41 chapters / 292 citedBy pairs → **forward gaps 0、reverse gaps 0、unknown keys 0**。（初跑 3 处 reverse 假阳为脚本 regex 未匹配「，key: xxx）逗号后标记，修正后归零；非书内缺口。）
- **待 T5 回写**：1.1 引 arXiv:2512.02189 后 blackwell-b200-microbench citedBy += 1.1（现仅 11.1；W6 research note 明示「留 W7 sweep 回写 1.1 正文」）。回写后须重跑 cross-check 保 0 缺口。
- **glossary**：190 术语（192 行 − 表头 − 分隔）；列名「首次实质引入章」+ 列语义声明完整（references.md:207「首次进入机制解释/数值账，纯前瞻指针不计，与正式定义章不同者注后者」）；W1-W6 各波归位记录在头注。✓
- **deepseekv3 notes stale**：见 S3-6（交 T5）。

## 5. R2 文风机械抽查：PASS

禁令词全 41 章扫描（在当今/值得注意的是/综上所述/总的来说/众所周知/让我们一起/其实/随着…快速发展）：**0 命中**。STYLE_GUIDE:144-145 禁令词表 + rubric A.3「其实」（W3 5.3 先例）均清。8.2:7「换句话说」不在禁令表，合法。

---

## 6. T2 errata 执行清单（兄弟维指针收窄/重判）

**状态核实**：W0 扫描模式重跑（/usr/bin/grep -rn '归 gpu\|属 gpu\|归 cluster\|属 cluster\|gpu 维度\|cluster 维度' models/ training/ inference/ --include='*.md'，文件名前缀排除 references/README）→ **86 章级命中，行号与盘点表（HEAD 22f904b）零漂移**。逐条核对：**86 行全部仍泛指向**（无一行已收窄到 gpu 章号；「归 gpu/cluster」混合句亦未收窄）。inference W4 closeout 已执行的是 serving 侧收窄（training 5.4:79 / models 4.5:327 的「EPLB serving 侧归 inference 7.4」部分），**gpu/cluster 归属部分仍泛**，须本次重判。

**计数**：93 盘点行 = 86 章级 + 3 README + 4 references。
- 3 README 边界句：**W1 已执行**（models/training/inference README L3 现文已含「通信全栈归 gpu、平台编排归 cluster」新 seam，逐字核实 ✓；errata 表 L236-238 记录在案）。
- 86 章级 = 41 重判 + 41 收窄 + 4 保留。保留 4 行中 2 行（models 1.2:228 / models 5.2:90 MFU 训练侧账）**不动**；2 行（models 4.8:66 MoD / inference 2.4:181 CXL）**措辞待裁定**。
- **可执行 = 82 章级 + 4 references = 86 行**（下表）。W1-W6 全 live → 全部目标章可收窄。

**执行规则**（spec §2）：仅限指针句最小修订，不动公式数值；逐条记入 gpu/README errata 表（T5 独占）。多目标跨波行按目标次序列各章。非 NVIDIA 句（inference 1.3:94 / 7.3:69 / 8.1:229）只把 NVIDIA 部分落章号，非 NVIDIA 沿 inference 1.3 [C] keynote 口径加一句注。

### 6.1 models（19 行；17 可执行 + 2 待裁定/保留）

| file:line | 现指针（gist） | 目标 gpu 章 | 波次 | 类型 |
| --- | --- | --- | --- | --- |
| models/01-foundations/02-compute-accounting.md:90 | 「elementwise、通信、IO 不计；后两者归 cluster 维度」 | 通信→8.1/10.1；**IO 项待裁定**（访存 IO→2.4，数据加载 IO→training 7.1，均非 cluster） | W4/W5 | 重判（IO 待裁定） |
| models/01-foundations/02-compute-accounting.md:228 | H800 峰值口径与 MFU 反推「属 gpu 与 training 维度」 | 保留（MFU 训练侧账；峰值口径可收窄 1.1/6.1 非必须） | — | 保留·不动 |
| models/01-foundations/02-compute-accounting.md:245 | 「微架构来源与测量方法归 gpu 维度」 | 2.1/2.4/11.2 | W1/W6 | 收窄 |
| models/01-foundations/03-scaling-laws.md:192 | 「expert 并行通信（……集群侧属 cluster 维度）」 | 通信硬件/transport→8.1/10.3；作业编排→cluster（本句不涉及） | W4/W5 | 重判 |
| models/02-transformer-anatomy/04-causality-and-kv-cache.md:88 | 「逐卡测量方法归 gpu 维度」 | 2.4/11.2 | W1/W6 | 收窄 |
| models/02-transformer-anatomy/04-causality-and-kv-cache.md:216 | 「微架构来源与测量方法归 gpu 维度」 | 2.1/2.4/11.2 | W1/W6 | 收窄 |
| models/02-transformer-anatomy/04-causality-and-kv-cache.md:244 | 「微架构来源与测量方法归 gpu 维度」 | 2.1/2.4/11.2 | W1/W6 | 收窄 |
| models/02-transformer-anatomy/08-assembly.md:367 | 「硬件互联归 cluster 维度」（TP 系统展开） | NVLink→7.1/7.2；机间→9.1/10.1 | W4/W5 | 重判 |
| models/04-modern-architectures/01-kv-compression-and-exact-attention.md:296 | 「实测吞吐数字归 gpu 维度的 benchmark 口径」 | 2.4/11.2 | W1/W6 | 收窄 |
| models/04-modern-architectures/01-kv-compression-and-exact-attention.md:298 | 「指令级 kernel 实现归 gpu 维度」（FA2/FA3） | 6.3 | W3 | 收窄 |
| models/04-modern-architectures/01-kv-compression-and-exact-attention.md:302 | 「指令级 kernel 实现归 gpu 维度」（FA IO 入口） | 6.3 | W3 | 收窄 |
| models/04-modern-architectures/01-kv-compression-and-exact-attention.md:316 | 「指令级归 gpu」（FA2/FA3 总结表行） | 6.3 | W3 | 收窄 |
| models/04-modern-architectures/05-moe.md:327 | 「硬件与网络侧（all-to-all 带宽与拓扑）归 cluster 维度」 | 带宽账→8.1/10.3；拓扑→10.1 | W4/W5 | 重判 |
| models/04-modern-architectures/05-moe.md:345 | 「拓扑归 cluster 维度」（EP 总结表行） | 10.1 | W5 | 重判 |
| models/04-modern-architectures/06-long-context.md:227 | 「通信调度、overlap 策略与拓扑映射的全部细节归 cluster 维度」+ cluster/README 活链接 | Ring Attention 调度/overlap→10.4；拓扑映射→10.2；活链接随新 seam 改写 | W5 | 重判 |
| models/04-modern-architectures/06-long-context.md:287 | 「通信实现归 cluster 维度」（Ring Attention 总结表行） | 10.4 | W5 | 重判 |
| models/04-modern-architectures/08-frontier-outlook.md:66 | MoD/early-exit「kernel 与流水线的处理归 inference 与 cluster 维度」 | **待裁定**（gpu 无专章；cluster 新 scope 不含 kernel/流水线） | — | 措辞待裁定 |
| models/05-case-studies/02-deepseek.md:90 | H800 峰值口径与 MFU 反推「归 gpu 维度」 | 保留（MFU 训练侧账；峰值可收窄 1.1/6.1 非必须） | — | 保留·不动 |
| models/05-case-studies/02-deepseek.md:103 | 「工程侧——DualPipe、a2a 低精度化、kernel——归 training/cluster/gpu 维度」 | a2a→10.3（低精度格式→6.1）；kernel→6.2/6.3；DualPipe 仍 training 5.3；cluster 不含 | W3/W5 | 重判 |

### 6.2 training（14 行，全可执行）

| file:line | 现指针（gist） | 目标 gpu 章 | 波次 | 类型 |
| --- | --- | --- | --- | --- |
| training/01-autograd-and-backprop/01-computation-graph-and-autograd.md:168 | 「其机制归 gpu 维度（kernel 与图捕获）」 | 3.3/5.4 | W2/W3 | 收窄 |
| training/01-autograd-and-backprop/01-computation-graph-and-autograd.md:187 | 「有效吞吐的量级口径归 gpu 维度」 | 2.4/11.2 | W1/W6 | 收窄 |
| training/01-autograd-and-backprop/02-operator-backward.md:46 | 「（kernel 层归 gpu 维度）」（GEMM 家族） | 5.2 | W3 | 收窄 |
| training/01-autograd-and-backprop/02-operator-backward.md:218 | 「块调度与 SRAM 驻留的 kernel 级实现归 gpu 维度」（FA 反向） | 6.3 | W3 | 收窄 |
| training/04-data-parallelism/01-dp-and-ddp.md:73 | 「节点内 NVLink 与跨节点 IB……硬件侧归 cluster 维度」 | 节点内→7.2；机间→9.1/9.2 | W4/W5 | 重判 |
| training/04-data-parallelism/01-dp-and-ddp.md:75 | 「NCCL……选择逻辑归 cluster 维度」 | 8.3/8.4/10.2 | W4/W5 | 重判 |
| training/04-data-parallelism/01-dp-and-ddp.md:130 | 「变体选择归 cluster」（树形对照总结表行） | 8.3/8.4/10.2 | W4/W5 | 重判 |
| training/05-model-parallelism/01-tensor-parallelism.md:158 | 「kernel 归 gpu 维度」（按 head 切分 FA） | 6.3 | W3 | 收窄 |
| training/05-model-parallelism/02-sequence-and-context-parallelism.md:71 | 「块调度与 SRAM 驻留的 kernel 细节归 gpu 维度」（CP 分块） | 6.3（通信调度→10.4） | W3/W5 | 收窄 |
| training/05-model-parallelism/04-expert-parallelism.md:71 | 「kernel 层的实现归 gpu 维度」（dropless grouped GEMM） | 6.2 | W3 | 收窄 |
| training/05-model-parallelism/04-expert-parallelism.md:79 | 「硬件/网络拓扑归 cluster 维度」（EPLB；serving 侧已归 inference 7.4） | 拓扑→10.1；EPLB 通信影响→10.3 | W5 | 重判 |
| training/05-model-parallelism/04-expert-parallelism.md:93 | 「细节归 gpu/cluster 维度」（DeepEP 两级网络/warp spec/通信通道） | 10.3（transport 级） | W5 | 重判 |
| training/05-model-parallelism/04-expert-parallelism.md:95 | 「kernel 实现归 gpu/cluster 维度，本书不展开」（DeepEP） | 10.3（transport 级） | W5 | 重判 |
| training/05-model-parallelism/04-expert-parallelism.md:134 | 「dropless 归 gpu」（总结表行） | 6.2 | W3 | 收窄 |

### 6.3 inference（53 行；51 可执行 + 1 待裁定 + 1 历史口径句）

| file:line | 现指针（gist） | 目标 gpu 章 | 波次 | 类型 |
| --- | --- | --- | --- | --- |
| inference/01-principles-and-accounting/02-prefill-decode-accounting.md:19 | 「微架构来源归 gpu 维度」 | 2.1/2.4 | W1 | 收窄 |
| inference/01-principles-and-accounting/02-prefill-decode-accounting.md:226 | 「RDMA 硬件层归 cluster 维度」 | 9.1/9.2 | W5 | 重判 |
| inference/01-principles-and-accounting/02-prefill-decode-accounting.md:242 | 「不推导 kernel 内 tiling 与指令级实现（归 P3 与 gpu 维度）」 | 4.3/5.2/6.3 | W2/W3 | 收窄 |
| inference/01-principles-and-accounting/03-roofline-and-tradeoffs.md:9 | 「微架构来源与测量方法归 gpu 维度」 | 2.1/2.4/11.2 | W1/W6 | 收窄 |
| inference/01-principles-and-accounting/03-roofline-and-tradeoffs.md:94 | 「互联硬件与拓扑细节归 cluster 维度」（scale-up domain） | NVLink 域→7.1；机间→10.1（非 NVIDIA 沿 1.3 [C] 口径加注） | W4/W5 | 重判 |
| inference/02-kv-cache-systems/04-kv-offload-and-tiering.md:96 | 「GDR……硬件协议层归 cluster 维度」 | 9.2 | W5 | 重判 |
| inference/02-kv-cache-systems/04-kv-offload-and-tiering.md:121 | 「硬件/拓扑/RDMA 协议细节归 cluster 维度」（Mooncake） | 7.2/9.1/9.2/10.1 | W4/W5 | 重判 |
| inference/02-kv-cache-systems/04-kv-offload-and-tiering.md:181 | 「CXL……硬件协议细节归 cluster 维度」 | **待裁定**（gpu 无 CXL 专章；cluster 新 scope 不含硬件协议） | — | 措辞待裁定 |
| inference/03-inference-operators/01-decode-attention-kernels.md:11 | 「warp 级占用率精确定义与硬件来源归 gpu 维度」 | 1.2/3.4 | W1/W2 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:23 | 「warp 内寄存器分配、tensor core 指令形状、SASS 调度归 gpu 维度」 | 1.3/2.3/6.3 | W1/W3 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:125 | 「指令级归 gpu 维度」（FA1/2/3） | 6.3 | W3 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:129 | 「TMA/WGMMA 指令语义归 gpu」（FA3） | 1.3/4.4 | W1/W2 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:210 | 「TMA 指令语义归 gpu」（FlashMLA copy-GEMM） | 4.4 | W2 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:285 | 「指令级优化（PTX 比较/位操作等）归 gpu 维度」（DSA radix-select） | 6.4 | W3 | 收窄 |
| inference/03-inference-operators/01-decode-attention-kernels.md:350 | 「warp/tensor core/SASS 指令级归 gpu 维度」（参考） | 6.3/6.4 | W3 | 收窄 |
| inference/03-inference-operators/02-weight-quantization.md:311 | 「指令级归 gpu 维度」（量化格式消费） | 6.2 | W3 | 收窄 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:7 | 「指令级实现归 gpu 维度，RDMA/NVLink 硬件与拓扑归 cluster 维度」 | 收窄 5.2/6.2；重判 7.1/9.1/10.1 | W3/W4/W5 | 混合 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:43 | 「指令级归 gpu 维度」×2 +「多 warp 细节归 gpu 维度」（Marlin） | 6.2 | W3 | 收窄 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:53 | 「TMA……指令细节归 gpu 维度」（DeepGEMM SF） | 4.4 | W2 | 收窄 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:55 | 「warp specialization/TMA/SASS 级优化归 gpu 维度」（DeepGEMM） | 4.4/5.2 | W2/W3 | 收窄 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:95 | 「tile 粒度推论，指令细节归 gpu 维度」（grouped GEMM padding） | 4.3/6.2 | W2/W3 | 收窄 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:146 | 「硬件测量归 cluster 维度」（RDMA 单跳延迟） | 9.1/11.2 | W5/W6 | 重判 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:150 | 「QP（……硬件归 cluster 维度）」 | 9.1 | W5 | 重判 |
| inference/03-inference-operators/03-quantized-gemm-and-moe-kernels.md:181 | 「指令级实现归 gpu 维度，通信硬件归 cluster 维度」 | 收窄 5.2/6.2；重判 7.1/8.4/9.1/10.1 | W3/W4/W5 | 混合 |
| inference/03-inference-operators/04-fusion-and-graph-capture.md:7 | 「warp/tensor core/SASS 指令级、Triton codegen 内部归 gpu 维度」 | 5.4/6.3 | W3 | 收窄 |
| inference/03-inference-operators/04-fusion-and-graph-capture.md:9 | 「测量与微架构来源归 gpu 维度」（t_launch） | 3.3/11.2 | W2/W6 | 收窄 |
| inference/03-inference-operators/04-fusion-and-graph-capture.md:81 | 「codegen 内部归 gpu 维度」（Inductor→Triton） | 5.4 | W3 | 收窄 |
| inference/03-inference-operators/04-fusion-and-graph-capture.md:99 | 「Inductor/Triton 内部归 gpu 维度」 | 5.4 | W3 | 收窄 |
| inference/03-inference-operators/04-fusion-and-graph-capture.md:103 | 「tile 到 thread/warp 映射与指令选择归 gpu 维度」（TileLang） | 4.3/5.4 | W2/W3 | 收窄 |
| inference/04-scheduling-and-batching/03-chunked-prefill.md:145 | 「GEMM tile 效率的量化归 3.3 / gpu 维度」 | 4.3/5.2 | W2/W3 | 收窄 |
| inference/05-vllm/03-vllm-distributed.md:125 | 「DeepEP 模式与硬件层归 cluster 维度」 | 10.3 | W5 | 重判 |
| inference/05-vllm/03-vllm-distributed.md:405 | 「silicon 与通信硬件层归 gpu / cluster 维度」（参考） | silicon→gpu P1–P6；通信硬件→gpu P7–P10；cluster 仅平台编排 | W1–W5 | 混合 |
| inference/06-sglang/03-sglang-distributed-pd.md:67 | 「RDMA/NVLink 硬件层归 cluster 维度」 | 7.1/9.1 | W4/W5 | 重判 |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:28 | 「指令级实现归 gpu 维度，RDMA/NVLink 硬件层归 cluster 维度」 | 收窄 6.3/6.4；重判 7.1/9.1 | W3/W4/W5 | 混合 |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:48 | 「接口/文件级锚，不是指令级实现——后者归 gpu」 | 5.2/6.3/6.4 | W3 | 收窄 |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:69 | 「硬件参数表归 1.3，微架构归 gpu」（Ascend 950 峰值反推） | NVIDIA 微架构→1.1/2.1；Ascend gpu 不覆盖，**句需口径注** | W1 | 收窄（带注） |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:110 | 「warp specialization/TMA/SASS 级优化归 gpu 维度」 | 4.4/6.3 | W2/W3 | 收窄 |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:120 | 「硬件层归 cluster 维度」（DeepEP SM 账/NCCL Gin） | 8.4/10.3 | W4/W5 | 重判 |
| inference/07-deepinfer-and-deepseek/03-deepseek-kernel-stack.md:136 | 「silicon 归 gpu」（FlashMLA 手调 SASS 差距） | 6.4 | W3 | 收窄 |
| inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md:22 | 「RDMA/NVLink 硬件与网络拓扑归 cluster 维度」 | 7.1/9.1/10.1 | W4/W5 | 重判 |
| inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md:74 | 「硬件/网络拓扑归 cluster」（EPLB 重判记录句） | 10.1/10.3；**句内「原文尚未改」为 W4 历史口径，收窄时一并修订为历史注** | W5 | 重判+历史注 |
| inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md:133 | 「one-sided read 不占对端 CPU（……语义归 cluster）」 | 9.1 | W5 | 重判 |
| inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md:135 | 「RDMA 硬件层归 cluster 维度」（Engram） | 9.1/9.2 | W5 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:23 | 「RDMA/NVLink 硬件、NCCL 实现、交换机拓扑归 cluster 维度」 | 7.1/7.2/8.4/9.1/10.1 | W4/W5 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:62 | 「实测依硬件与实现归 cluster 维度」（τ_lat 教学值） | 7.2/8.4/11.2 | W4/W6 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:68 | 「RDMA 往返与协议开销，硬件测量归 cluster 维度」 | 7.2/9.1 | W4/W5 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:118 | 「τ_a2a……实测归 cluster」 | 8.1/10.3 | W4/W5 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:122 | 「硬件/网络拓扑归 cluster」（EPLB 归属口径句） | 10.1/10.3 | W5 | 重判 |
| inference/08-distributed-and-pd/01-serving-parallelism.md:229 | 「NVLink/灵衢/RDMA 硬件实现、NCCL 算法选择与拓扑探测、交换机与网络协议归 cluster 维度」 | 7.1/8.4/9.1/10.1/10.2（灵衢非 NVIDIA 加注） | W4/W5 | 重判 |
| inference/08-distributed-and-pd/03-kv-transfer.md:194 | 「GDR……硬件协议层归 cluster 维度」 | 9.2 | W5 | 重判 |
| inference/08-distributed-and-pd/03-kv-transfer.md:210 | 「RDMA 协议/NIC/PCIe 拓扑/NVLink 域/交换机与 NCCL 实现归 cluster 维度」（边界声明） | 9.1（RDMA/NIC）/7.2（PCIe）/7.1（NVLink 域）/10.1（交换机）/8.4（NCCL） | W4/W5 | 重判 |
| inference/08-distributed-and-pd/03-kv-transfer.md:325 | 「硬件层（RDMA/NIC/PCIe/NVLink/NCCL）归 cluster 维度」（总结表） | 7.1/7.2/8.4/9.1/10.1 | W4/W5 | 重判 |
| inference/08-distributed-and-pd/05-elasticity-and-fault-tolerance.md:9 | 「RDMA/NVLink 硬件与网络拓扑归 cluster 维度」（章脊） | 7.1/9.1/10.1 | W4/W5 | 重判 |

### 6.4 references 级（4 行，inference/references.md，全可执行）

| file:line | 现指针 | 目标 | 波次 |
| --- | --- | --- | --- |
| inference/references.md:93（fa2 条） | 「指令级归 gpu」 | 6.3 | W3 |
| inference/references.md:94（fa3 条） | 「指令级归 gpu」 | 6.3 | W3 |
| inference/references.md:101（marlin 条） | 「指令级 lop3/bfe 归 gpu」 | 6.2 | W3 |
| inference/references.md:104（deepep 条） | 「硬件归 cluster」 | 10.3 | W5 |

### 6.5 待裁定项 / 历史口径句 / 「README 18→19」re-identify

**待裁定（3 项，gpu 无专章、cluster 新 scope 亦不含，按内容裁定措辞，不强行塞章号）**：
1. models/01-foundations/02-compute-accounting.md:90 **IO 项**（通信部分可收窄 8.1/10.1；IO 项访存→2.4 / 数据加载→training 7.1，建议拆句裁定）。
2. models/04-modern-architectures/08-frontier-outlook.md:66 **MoD/early-exit kernel 与流水线**（建议改「kernel 与流水线处理归 inference（算子策略）与 gpu（kernel 实现）」泛指向，或删 cluster）。
3. inference/02-kv-cache-systems/04-kv-offload-and-tiering.md:181 **CXL 硬件协议**（建议改「CXL 硬件协议细节非本书覆盖范围」或泛指向 gpu 内存层级 2.x 不展开，去 cluster）。

**历史口径句（1 项）**：inference/07-deepinfer-and-deepseek/04-deepseek-large-scale-ep.md:74「截至本章核实，training 5.4 §4 / models 4.5 §7 原文尚未改」——两处 serving 侧 errata 已于 inference W4 closeout 执行（现文已含「归 inference 7.4」），此括注为 W4 写作时点记录，收窄 gpu 部分时一并修订为历史注（避免误读为当前状态）。

**「README 18→19」re-identify 结论：未证实为残留，无需修**。排查：gpu/README.md 全部「18」出现均为图 QUALITY 计数（18/18，与实图数吻合：p01+p02=10+8=18、p03+p04=18、p05+p06=22、p07+p08=20、p09+p10=9+9=18、p11=13，总 109）或盘点表行号，无 18→19 计数错；根 README/cluster README 无相关计数。最可能候选 = DeepEP main README NVSHMEM 锚 **L18**（10.3:119/141/189/200 + refs deepep 条），W5 已对 pinned commit f6aa38fe fresh tarball live grep 核实「全 repo 仅 README L18 一句」正确（同波 L38→L39 是另一锚的订正）。书锚 pinned commit，L18 正确，**不动**；若 T3 重核当前 main 发现漂移再按 pinned 口径处理。

---

## 7. T3 gpu 残留清单

| # | 位置 | 残留 | 修 |
| --- | --- | --- | --- |
| T3-1 | gpu/01-gpu-microarchitecture/01-chip-organization-and-generations.md:15,110,125,187,198,203（6 行） | ⟨verify W3/W6⟩ B200 SM 148 标记（W6 已经 [B] arXiv:2512.02189 销项，正文未回写） | 去 ⟨verify⟩ 标记；148 措辞「反推自洽、官方未披露」→「[B] 微基准核实（arXiv:2512.02189）+ 反推自洽」；L125 反推段引 2512.02189 外部确认；L203 Blackwell brief 参考条「B200 SM 数未披露（⟨verify W3/W6⟩）」→「SM 数 datasheet 未印，148 经 blackwell-b200-microbench [B]」。**refs citedBy += 1.1 交 T5**（现 blackwell-b200-microbench 仅 11.1）。练习 4(d) L198「须留 ⟨verify⟩」措辞同步改（反推不唯一性教学点保留，结论改「148 已 [B] 核实，反推自洽获外部确认」）。 |
| T3-2 | gpu/08-nccl-collectives/01-collective-primitives.md:98 | 排列族（S3-1） | 「均衡 a2a 是一个排列」→「均衡 a2a 是均匀流量（双随机矩阵、非排列）」；crossbar「任意排列线速」保留 |
| T3-3 | gpu/10-fabric-and-scale-communication/03-ep-alltoall-implementation.md:49 | 排列族（S3-2） | 「均衡排列」→「均衡（均匀）流量」；与命题 3 L147 同口径 |
| T3-4 | gpu/09-rdma-gpudirect/02-gpudirect-rdma.md:122 | D2H+D2H typo（S3-3） | →「D2H+H2D」 |
| T3-5 | gpu/10-fabric-and-scale-communication/04-point-to-point-sequence-parallel.md:199 | §6 表头口径（S3-4） | 表头注 n=8 行 T_blk=16384（勿改数值） |
| T3-6 | gpu/01-gpu-microarchitecture/03-tensor-core-mma.md:176 | ⟨verify W3⟩ cta_group 形状表未销项（S3-5，新发现） | 建议改显式 scope 注去标记（不新增走读） |
| T3-7 | README.md:14,38（+34,40） | 根 README gpu 占位 | L14 gpu 行「占位（W1 上线）……41 章规划」→「已关闭（W1–W7 全部审结）……11 Part/41 章」；L38 项目状态「gpu / cluster 占位」→「gpu 41 章已关闭（W1–W7）」「cluster 占位」+「站点上线（models+training+inference+gpu）」；**L34 文献汇总列补 gpu/references.md、L40 逐章进度列补 gpu/README.md**（brief 仅列 14/38，34/40 同属翻转面）。og.png 已存在 site/public/og.png（更新交 T3/T5 执行）。 |

**交 T5（非 T3）**：deepseekv3 notes stale（S3-6）+ 1.1 citedBy 回写后 cross-check + 断言锁死 {41,11,50}（build-content.mjs:106 现值已 {41,11,50}，W6 已锁，T5 确认保持）。

## 8. T4 图排列清单

权威口径 = 10.3 命题 3（gpu/10-fabric-and-scale-communication/03-ep-alltoall-implementation.md:147）：「均衡 a2a 是均匀流量……均匀流量是双随机矩阵、非排列——排列流量精确定义见 10.1 命题 1」。

| 图文件 | 行 | 现文本 | 修 |
| --- | --- | --- | --- |
| site/scripts/lib/figures/gpu-p08.mjs | 144 | cap「均衡 a2a 是一个排列（每 rank 收发 (n-1)/n·M）：NVSwitch 非阻塞对任意排列线速，域内总注入不变」 | 首句→「均衡 a2a 是均匀流量（双随机矩阵、非排列，每 rank 收发 (n-1)/n·M）」；后半 crossbar 义保留 |
| site/scripts/lib/figures/gpu-p10.mjs | 375 | 注释「a2a incast：均衡排列 vs 偏斜 σC」 | 「均衡（均匀）流量 vs 偏斜 σC」 |
| site/scripts/lib/figures/gpu-p10.mjs | 382 | cap「a2a incast：均衡排列入口恰线速 C……」 | 「均衡（均匀）流量入口恰线速 C」 |
| site/scripts/lib/figures/gpu-p10.mjs | 384 | 注释「// 均衡排列」 | 「// 均衡（均匀）流量」 |
| site/scripts/lib/figures/gpu-p10.mjs | 385 | zone label「均衡排列（σ=1）：入口恰线速」 | 「均衡流量（σ=1）：入口恰线速」 |
| site/scripts/lib/figures/gpu-p10.mjs | 438 | scene label「a2a incast 对照：均衡排列每 sender 出口均分 n-1 份……」 | 「均衡（均匀）流量每 sender 出口均分 n-1 份」 |

**保留不动（crossbar 义正确）**：gpu-p10.mjs:69（König 边着色「排列流量全部线速」）；gpu-p08.mjs:310/323（「环排列任意（非阻塞）」）；7.1:55 / 7.2:73/75 章内 crossbar「任意排列」义。σ=1 入口=C 数值账正确，仅术语修。图修后须重跑 fig-check + QUALITY + Playwright（T4/T5 QA）。

---

## 9. Verification log（spec §6.3 强制）

| 检查 | 命令/方法 | 结果 |
| --- | --- | --- |
| 兄弟维指针扫描 | /usr/bin/grep -rn '归 gpu\|属 gpu\|归 cluster\|属 cluster\|gpu 维度\|cluster 维度' models/ training/ inference/ --include='*.md'，文件名前缀排除 | 86 章级命中，行号与盘点表零漂移，全泛指向 |
| 已收窄探测 | grep 'gpu 维度 [0-9]\|gpu [0-9]\.[0-9]\|gpu P[0-9]\|归 gpu/' | 0 真收窄（3 假阳：models 5.2:90「1.2 练习 3」指 models、training 5.4:93/95「归 gpu/cluster」仍泛） |
| README 边界句 | sed -n '3p' models/training/inference README | 三处新 seam 已在位（W1 执行 ✓） |
| 数值脊柱 | python3 -I 独立重算（§2 表） | 全闭合 |
| refs parse-check | cd site && node --input-type=module -e "import {readFileSync} from 'node:fs'; import {parseReferences} from './scripts/lib/references.mjs'; const r=parseReferences(readFileSync('../gpu/references.md','utf8')); console.log(r.groups.length, r.groups.reduce((a,g)=>a+g.entries.length,0));" | 11 88 |
| refs forward+reverse | python3 -I /tmp/w7_refs_crosscheck.py（key 标记 regex 含全角/逗号后形式） | forward 0 / reverse 0 / unknown 0 / 292 pairs |
| glossary 计数 | sed -n '/## 术语表/,$p' gpu/references.md \| grep -c '^\| ' | 192 行 − 2 = 190 |
| 禁令词 | grep -rn '在当今\|值得注意的是\|综上所述\|总的来说\|众所周知\|让我们一起\|其实\|随着.*快速发展' gpu/ | 0 命中 |
| verify 标记 | grep -rn '⟨verify' gpu/ 章文件 | 1.1×6 行 + 1.3:176（1.3 为新发现） |
| 图注册数 | grep -c 'replacesBlock:\|afterHeading:' gpu-p*.mjs | p01-p11 = 10/8/10/8/10/12/7/13/9/9/13 = 109，与 README QUALITY 记录吻合 |
| 站点断言 | grep build-content.mjs:106 | gpu {chapters:41, parts:11, refsMin:50}（W6 已锁） |
| recap 源存在 | grep models 4.6 §3.2 / inference 8.3:194 / training 5.4:79 | 源章/节均在位 |

**审计结论**：全书数值脊柱/recap/refs/glossary/R2 机械面全 PASS，S1=S2=0，S3=6（均口径/措辞/标记，不阻塞）。T2 执行清单 86 行（+3 待裁定 +1 历史注），T3 残留 7 项，T4 图 2/6 处。W7 收官条件满足，执行 T2/T3/T4 后 T5 终 sweep 锁死 {41,11}。