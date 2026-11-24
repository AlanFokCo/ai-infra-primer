# gpu 维度设计 Spec 四路对抗审查 · D 路（设计合理性 + 边界纪律 + 站点/lint 方案）

- 日期：2026-11-24
- 审查对象：`docs/superpowers/specs/2026-11-24-gpu-dimension-design.md`（609 行全读，§1-§10 + 预核实附录 + builder-check 附录）
- 审查者：fresh-context adversarial evaluator（route D），不继承写作上下文，未派发 subagent，全部工作自行执行
- 对照材料：inference spec（§4/§6/§7/§8 + builder-check/四路附录）、STYLE_GUIDE.md（180 行全读）、R2 style sheet（304 行全读）、站点代码（build-content.mjs / dimensions.mjs / references.mjs / links.mjs / lint-chapter.sh / lint-math-boundary.py / [dim]/index.astro / index.astro / DependencyGraph.astro / qa-shots.mjs / package.json / figures 模块）、inference README（进度表/依赖图/指针盘点/errata 范式）、gpu+cluster 占位 README、已发布 models/training/inference 章节（指针句与公式一手核对）
- 结论：**S1×0 / S2×6 / S3×8**。设计主体高度可执行：依赖图 14 边无环且零边违反波次序、41 章/6 波/滚动断言算术全对、站点 §8 计划与代码逐条吻合（含 refsMin 地板、三处硬读、FORTHCOMING 死代码、live 翻转机制）、builder-check 9 项修复全部落地、指针 errata 通道 ~73 计数精确复现。S2 集中在符号表内部一致性（n/GEMM M-N-K/注意力几何三处约定缺口 + 1.2 occupancy 草式单位不一致）、11.3 一处跨维度章号误引、W0 指针盘点漏兄弟 README 边界句。无阻塞性设计缺陷；S2 修复后 W0 可放行。

---

## 1. Findings 表

severity 口径沿 STYLE_GUIDE §10：S1 = 事实性致命；S2 = 实质问题（约定冲突、推导断裂、协议缺口、局部事实错误）；S3 = minor（措辞、格式、可选增补）。

| # | severity | 位置 | 断言 | 证据（code line / spec section / 已发布文本） | 建议 |
| --- | --- | --- | --- | --- | --- |
| D-S2-1 | S2 | spec §5 1.2 关键推导（L188）+ 内容行（L187） | occupancy 物理上界草式 $\mathcal{O}_{\max} = \min(\text{warp slots}, \lfloor R_{\mathrm{file}}/R_{\mathrm{th}}\rfloor, \lfloor S_{\mathrm{total}}/S_{\mathrm{sh}}\rfloor)$ 三项单位不一致：warp slots 是 warps、$R_{\mathrm{file}}/R_{\mathrm{th}}$ 是 threads（registers ÷ registers/thread）、$S_{\mathrm{total}}/S_{\mathrm{sh}}$ 是 blocks（bytes ÷ bytes/block），不能直接取 min 得「理论 active warps」；且缺 block 粒度取整（register/shared 按 block 分配）。与 3.4 大纲正确流程（block size → active blocks/SM → active warps）内部矛盾 | §4.3 L130 定义 $R_{\mathrm{th}}$ = registers/thread、$S_{\mathrm{sh}}$ = shared bytes/block；§5 3.4 L251-252 正确流程；1.2 内容行自称结果「理论 active warps」。writer 照草式直写必错，evaluator 强制重推导（§6.3 L471）会在章级抓出，但 spec 契约本身应先修 | 草式改 warp 单位一致形：$\min(\text{warp slots}, \lfloor R_{\mathrm{file}}/(32 R_{\mathrm{th}})\rfloor, \lfloor S_{\mathrm{total}}/S_{\mathrm{sh}}\rfloor \cdot w_{\mathrm{block}})$ 并注明 block 粒度取整（或写「三约束各自换算到 warp 单位后取 min，换算含 block 粒度」）；可顺带声明 max blocks/SM 第四约束或注明简化。与 3.4 流程对齐。（跨路注：A 路 P1 数学核查可能同 flag，去重合并） |
| D-S2-2 | S2 | spec §4.3 消歧硬规则（L136）/ rank 行（L125）vs §5 5.1（L285-286）、§6.3（L475） | $n$ 双重赋值：硬规则「$n$ 专归 rank 数（章首声明）」，但 5.1 大纲「$n$ 元素求和」「Blelloch scan work $O(n)$/depth $O(\log n)$」与 §6.3 强制清单同用 $n$ = 归约/扫描**元素数**（非 rank）。5.1 是 intra-GPU kernel 章，无 rank 语义。evaluator 约定一致性检查（风险 7）会系统性判 S2 | §4.3 L125「$n$ = rank/GPU 数」、L136「$n$ 专归 rank 数」；§5 5.1 L285「$n$ 元素求和」、L286「$\Omega(n)$ 下界」；§6.3 L475。builder-check 附录 L590 声称「符号表内部一致（无 inference 初稿 $b$ 双重赋值类问题）」——此处正是该类问题，builder-check 漏检 | 沿 inference $c_{\mathrm{chunk}}$ 局部声明先例：5.1 元素数改 $n_{\mathrm{elem}}$（或硬规则加显式例外「5.1 reduction/scan 章首声明『本章 $n$ = 元素数，非 rank 数』」），§6.3 5.1 行同步记号 |
| D-S2-3 | S2 | spec §4.3（L124/126/129/136）vs §5 5.2 关键推导（L292） | GEMM 问题维 $M/N/K$ 未入符号表且撞硬规则：5.2「HBM traffic = $O(MNK/T)$」用裸 $M$（GEMM 行维）、$N$（GEMM 列维）、$T$（tile 尺寸），与「$N$ 专归参数量 / gpu 不占用」硬规则、$M$ = 消息总字节表定义、$T$ = 序列长表定义（tile 应用 $T_M/T_N/T_K$）三重冲突。P4-P6 kernel 章与 8.x/10.x 通信章跨波同字母异义 | §4.3 L124「$N$…gpu 不占用」、L126 $M$ = 消息字节、L129「$T$ = 序列长…tile 维用 $T_M,T_N,T_K$」、L136「$N$ 专归参数量」；§5 5.2 L292 $O(MNK/T)$。inference 先例：$c$ 双义经四路裁定 + $c_{\mathrm{chunk}}$ 局部声明闭合（inference spec §4.3 二次修订注） | §4.3 加行或硬规则例外：「kernel 章（P4-P6）GEMM 问题维 $M/N/K$ 章首局部声明（本章 = GEMM 矩阵维，非消息字节/参数量/序列长）；通信章（8.x/10.x）$M$ = 消息字节、禁 $N$ 表 rank 不变」；5.2 草式 $T$ 改 $T_{\mathrm{tile}}$ 或声明 |
| D-S2-4 | S2 | spec §4.3 表（L122-134）/ §4.2 recap 行（L110/116）vs §5 6.3（L323-324）、6.4（L329-330） | 注意力几何符号缺行 + IO 定理 recap 翻译缺口：(1) 6.3 用裸 $d$（head 维，「$T_M \times d$ Q tile」）未入表；(2) §4.2 翻译规则引用 $d_{\mathrm{model}}$ 但表无此行；(3) 6.4 用 $d_{\mathrm{latent}}/d_{\mathrm{rope}}$ 未入表（inference 表有）；(4) models 4.1 IO 定理 $O(T^2 d^2/M)$ 的 $M$ = **可用 SRAM 元素数**、$d$ = head 维，与 gpu $M$ = 消息字节冲突，§4.2 recap 行原样保留 $O(T^2 d^2/M)$ 未翻译，违反自设「recap 兄弟公式把符号译为 gpu 约定」硬规则 | models 4.1 L276 逐字「$M$ 为可用 SRAM、元素计」、L288 $d$ = 64 head 维口径（一手核实）；spec §4.2 L116 翻译硬规则、L110 recap 行；§5 6.3 L323 裸 $d$。先例：inference fix round 1 裸 $d$ 判 Important、译 $d_{\mathrm{model}}$ 入表（inference spec L603） | §4.3 加「注意力/模型几何」行：$d_{\mathrm{head}}$（recap models 4.1 FA 的 $d$ 翻译）、$d_{\mathrm{model}}$（models $d$ hidden 翻译）、$d_{\mathrm{latent}}/d_{\mathrm{rope}}$（沿 inference 口径）；§4.2 IO 定理 recap 行 $M$ → $S_{\mathrm{total}}$（或局部声明「IO 定理 $M$ = SRAM 容量，非消息字节」）；6.3/6.4 大纲裸 $d$ 同步 |
| D-S2-5 | S2 | spec §5 11.3（L441 内容 / L443 锚点 / L444 Sources，共 4 处「training 7.3」） | 「training 7.3 checkpoint」章号误引：checkpoint 与续训是 **training 7.4**（`04-checkpointing-and-resumption.md`）；7.3 = training stability（`03-training-stability.md`），其 checkpoint 指针全部指向 7.4。W6 writer 照引会 recap 错章，evaluator 事实核查判局部事实错误 | training/07-pretraining/ 目录一手核实：7.3 = 03-training-stability.md、7.4 = 04-checkpointing-and-resumption.md（标题「7.4 Checkpoint 与续训」）；7.3 L81「checkpoint 与自动恢复……归 7.4」、L165「完整内容语义归 7.4 §1」。先例：inference D-S2-1 式号误引（式 (5)→(9)）同判 S2 | 11.3 四处改「training 7.4（checkpoint 与续训；故障分类见 7.3）」；§2 out-of-scope 若补训练侧容错指针同改 |
| D-S2-6 | S2 | spec §2 扫描模式（L44）/ §8 根 README 条（L529） | W0 指针盘点漏兄弟维度 README 边界句：models/README L3「集群通信……归 `cluster/`」、training/README L3「网络与拓扑归 `cluster/`」、inference/README L3「跨机通信硬件归 `cluster/`」三处**活的范围声明**被新 seam 推翻，但 §2 扫描模式（归 gpu/属 gpu/归 cluster/属 cluster/gpu 维度/cluster 维度）匹配不到这些句式（反引号路径、词序不同），73 计数亦排除 README；§8 只覆盖根 README。W0 后三处 stale 边界句将留在已发布 README 里与新 seam 矛盾 | models/README.md L3、training/README.md L3、inference/README.md L3 一手核实（逐字见验证 log V49）；spec §2 L44 扫描模式与「约 73 处」计数口径（章节 only）。先例：inference 四路 D-S2-4 已发布指针归属冲突入 errata 同判 S2 | §2 或 §8 显式列三处 README 边界句为 W0 errata 目标（W0 盘点时一并改写为「通信全栈归 gpu、平台编排归 cluster」）；扫描模式补 `cluster/` 范围句变体（如「归 \`cluster/\`」「集群通信」）；盘点表加 README 级行 |

| # | severity | 位置 | 断言 | 证据 | 建议 |
| --- | --- | --- | --- | --- | --- |
| D-S3-1 | S3 | spec §5 依赖图 prose（L176） | 波内依赖枚举不全：prose 只列 P5→P6/P7→P8/P9→P10 三条，实际波内边 5 条（缺 P1→P2、P3→P4）；builder-check 附录（L592）列全 5 条 | mermaid 14 边逐边波次映射（V3/V4）；附录 L592 全 5 条 | prose 补全 5 条波内边，与附录一致 |
| D-S3-2 | S3 | spec §5 章级前瞻引用 prose（L176） | 前瞻引用清单不全且混同 recap 接口：所列 4 项中 6.1←1.3 / 6.3←4.3 / 10.2←8.4 的定义章均**早于**使用章（W1<W3、W2<W3、W4<W5），实为 recap 接口（真正前瞻方向是 1.3→6.1 等）；跨波前瞻指针另有 1.2→3.4、1.4→11.2、2.2→7.x、2.3→4.2、3.1→5.1、4.3→5.2、4.4→5.2、7.1→10.2、7.3→9.2 未列 | 全 41 章大纲交叉扫描（V5）；inference 先例 D-S3-1 前瞻计数补全（inference spec L177「四处章级前瞻引用」） | 改题「recap 接口与章级前瞻指针」并全量枚举（或声明「前瞻指针以大纲回链为准，此处列典型」），波内章级互引（9.2↔10.3、5.4↔6.3 等）注明由固定大纲覆盖 |
| D-S3-3 | S3 | spec §2（L44「通信类约 25-35 处」/ L45-52 预盘点） | 重判估计略低 + 预盘点无 inference 例：实测 cluster 归属句 38 处（章节 only），其中 ~36 处含通信/硬件关键词（新 seam 下几乎全部重判到 gpu）；预盘点 8 句全为 models/training，inference 侧 cluster 边界声明密集（8.1 L23、8.3 L210+L325、5.3 L125、3.3 L181、2.4 L181 CXL、8.3 L194 GDR 等）虽被全量 sweep 覆盖，但预盘点缺代表性示例 | grep 实测（V43/V44/V50）；inference 章节一手核实 | W0 盘点按 ~35-40 校准工作量；预盘点补 1-2 条 inference 例（如 8.3 L210 边界声明 → gpu P7-P10 章号） |
| D-S3-4 | S3 | spec §4.6（L150）R2-2 条 | R2-2 章脊口径未对 gpu 实例化：R2 原文口径为模型规模（「grep 全章模型名/参数量集合，>2 且无声明即违规」）；gpu 章脊是硬件 SKU/世代 + 精度，1.1 世代对比表 / 2.4 H100-B200 roofline / 11.1 服务器形态**有意**多 SKU，逐字读 R2 的 evaluator 可能误判 | R2 style sheet §3 R2-2 行（L239）；spec §5 1.1/2.4/11.1 大纲 | §4.6 加一句：gpu 章脊口径 = 硬件 SKU/世代 + 精度（「本章口径：H100 SXM」式），对比章章首声明多 SKU 对比口径 |
| D-S3-5 | S3 | spec §4.3 HBM 几何行（L131）vs §5 2.1（L207） | bus width 口径歧义：$\mathrm{BW} = n_{\mathrm{stack}} n_{\mathrm{ch}} f_{\mathrm{pin}} \cdot \text{bus width}/8$ 中 bus width 须为**每通道**位宽（64bit）才与 $n_{\mathrm{ch}}$ 因子自洽；2.1 内容行引「bus width 1024bit/stack」（= $n_{\mathrm{ch}} \times$ 每通道位宽），若 writer 以 1024 代入公式将 16× 高估 | §4.3 L131 公式；§5 2.1 L207「bus width 1024bit/stack」+「5 stack × 16 ch × … → 3.35 TB/s ⟨verify 逐项⟩」；H100 3.35 TB/s = 5×1024×5.2Gbps/8（每栈口径）或 5×16×64×5.2Gbps/8（每通道口径） | §4.3 公式行注「bus width = 每通道位宽；1024bit/stack = $n_{\mathrm{ch}} \times$ bus width」；2.1 ⟨verify 逐项⟩ 保留 |
| D-S3-6 | S3 | spec §5 6.2（L317）/ 6.4（L329）大纲 vs §4.2（L115，builder fix 6） | 代码 scope 声明隐式：§4.2 要求「6.2 Marlin / 6.4 FlashMLA 等含 repo 走读的章在大纲声明其片段范围」，但两章大纲仅以内容列表隐式定界（mixed-input tiling/dequant 指令；MLA 几何/paged gather），无显式「片段级走读（scope: …）」标记 | §4.2 L115 规则；§5 6.2 L317「repo 走读」、6.4 L329「FlashMLA repo 走读」 | 两章大纲显式标「片段级走读（scope: …）」，示范 §4.2 规则；writer brief 照抄 |
| D-S3-7 | S3 | spec §1 成功判据（L27）/ §4.6（L150） | 事实密集章 proof-exercise 命题口径未显式：1.1/7.1/9.1/11.1/11.3 事实密集推导少，每章 ≥1 题证命题的要求需命题落在可判定定量论断（各章关键推导行已提供：芯片级 FLOPS 反推/NVLink 聚合带宽/RDMA 延迟账/MIG 分区账/链路降级带宽账），但 spec 未点明训练 6.3 模式 | inference builder-check 附录「对照/总览章的 proof-exercise 要求（training 6.3 模式：命题落在可判定定量论断、非选型判断）」（inference spec L566） | §4.6 补一句沿 training 6.3 模式：事实密集章证明题命题取关键推导行的定量论断，不取选型判断 |
| D-S3-8 | S3 | spec §4.3 SM/occupancy 行（L130）/ §5 1.2（L188） | $R_{\mathrm{file}}/S_{\mathrm{total}}$（1.2 草式所用）未入表；$R$（ridge）与 $R_{\mathrm{th}}$（registers/thread）靠下标区分可成立，但混场章（3.4/11.2 同用 roofline 与 occupancy）宜章首声明 | §4.3 表仅列 $R_{\mathrm{th}}/S_{\mathrm{sh}}$；§5 1.2 L188 用 $R_{\mathrm{file}}/S_{\mathrm{total}}$ | 表补 $R_{\mathrm{file}}$（每 SM register file 容量）/$S_{\mathrm{total}}$（每 SM shared 总量）；混场章章首声明「$R$ = ridge，$R_{\mathrm{th}}$ = registers/thread」 |

---

## 2. Verification log（强制）

格式：检查项 × 证据 × 结论 × 覆盖率。证据为 code line / spec section / 已发布文本一手引句 / 实测命令输出。

### Task 1 · 依赖图（§5 mermaid）

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V1 | mermaid 边数 = 14 | spec L159-174 逐边清点：P1→P2, P2→P3, P3→P4, P3→P5, P1→P5, P4→P6, P5→P6, P2→P7, P7→P8, P7→P9, P8→P10, P9→P10, P10→P11, P6→P11 | ✓ 14 边，与 prose「14 边」及 §8 DependencyGraph 条一致 | 14/14 |
| V2 | Kahn 拓扑排序无环 | 入度表：P1=0, P2=1, P3=1, P4=1, P5=2, P6=2, P7=1, P8=1, P9=1, P10=2, P11=2；出队序 P1→P2→{P3,P7}→P4→P5→P6→{P8,P9}→P10→P11，11/11 节点出队 | ✓ 无环 | 11/11 Part |
| V3 | 零边违反波次序 | 逐边 wave(src)≤wave(dst)：W1={P1,P2}, W2={P3,P4}, W3={P5,P6}, W4={P7,P8}, W5={P9,P10}, W6={P11}；14 边全部前向或波内，无后向边 | ✓ 0 违反 | 14/14 |
| V4 | 波内边集合 | 波内边 = {P1→P2, P3→P4, P5→P6, P7→P8, P9→P10} 共 5 条；固定大纲并行写作先例（inference W1 P1→P2，inference spec L566）覆盖；§5 prose 只列 3 条 → D-S3-1；builder-check 附录 L592 列全 5 条 | ✓ 机制覆盖，prose 枚举不全（S3） | 5/5 |
| V5 | 章级前瞻引用完整性 | 全 41 章大纲扫描：prose 列 4 项（6.1←1.3, 6.3←4.3, 10.2←8.4, 10.3←9.2）；前三项定义章早于使用章（recap 接口），10.3←9.2 波内；另有跨波前瞻 1.2→3.4, 1.4→11.2, 2.2→7.x, 2.3→4.2, 3.1→5.1, 4.3→5.2, 4.4→5.2, 7.1→10.2, 7.3→9.2 未列 → D-S3-2；全部为指针+回链式，无「先用后定义」强依赖 | ⚠ 清单不全（S3），无阻塞 | 41/41 章大纲 |

### Task 2 · 算术

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V6 | 41 = 4+4+4+4+4+4+3+4+3+4+3 | §3 目录注释 + §5 各 Part 大纲逐章清点：P1-P6 各 4、P7=3、P8=4、P9=3、P10=4、P11=3；和 = 41 | ✓ | 11/11 Part |
| V7 | 波次 8+8+8+7+7+3 = 41 | §7 表：W1=P1+P2=8, W2=P3+P4=8, W3=P5+P6=8, W4=P7+P8=3+4=7, W5=P9+P10=3+4=7, W6=P11=3 | ✓ | 6/6 波 |
| V8 | 滚动断言 {8,2}/{16,4}/{24,6}/{31,8}/{38,10}/{41,11} | 累计章/Part：W1=8/2, W2=16/4, W3=24/6, W4=31/8, W5=38/10, W6=41/11；与 §8 L523 序列逐项吻合；W7 锁死 {41,11,refsMin:50} 沿 inference 先例（refsMin 不随实际 refs 数收紧，inference W7 锁 50 实际 83） | ✓ | 6/6 断言点 |
| V9 | W0 {0,0,refsMin:50} 可过 | build-content.mjs L110-116 Part 目录缺失 try/catch 跳过、L117 files.length===0 continue → 0 章/0 Part 可过；L220-221 refCount<refsMin 抛错 → refs 须 ≥50 | ✓（refs 地板见 V27/V28） | 代码路径全核 |
| V10 | 8 重章 ↔【重】标记 ↔ §6.3 清单 | grep【重】= 8 个章标题（1.2 L186 / 2.4 L224 / 4.3 L270 / 5.1 L284 / 5.2 L290 / 6.3 L322 / 8.2 L362 / 9.2 L388）+ §4.4 列表同 8 章；§6.3 强制清单含全 8 章 + 3.4/8.3/10.1/10.3 数值抽查（与 §4.4/风险 5「非【重】章仍抽查」一致） | ✓ 8/8/8 | 8/8 重章 |
| V11 | 章节文件名匹配站点 regex | build-content L112 `/^\d{2}-[\da-z-]+\.md$/`；§5 全部 41 个文件名（01-chip-organization-and-generations.md … 03-faults-reliability.md）逐一比对 | ✓ 41/41 匹配 | 41/41 |

### Task 3 · 符号表 §4.3

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V12 | 表内无双重赋值 | 逐行扫描：N/N_act、n、M/bytes、α/β、I/R/F_peak/BW、B/T/T_M-T_N-T_K、n_SM/O/R_th/S_sh、n_stack/n_ch/f_pin、BW_NVLink/n_link 各行内部自洽；**但** n 在 5.1 大纲被赋元素数（D-S2-2）、GEMM M/N/K 未入表撞硬规则（D-S2-3）、注意力 d/d_model/d_latent/d_rope 缺行（D-S2-4） | ⚠ 表内自洽，表↔大纲三处冲突（S2×3） | 13/13 行 |
| V13 | 跨维度翻译规则 | §4.2 L116：training d→n ✓（training 4.1 L73 α/β 符号、式 (2)(3) 用 d 一手核实）、models b→bytes ✓（models 2.4 式 (11)(12) 分母 b = 每元素字节一手核实）、inference R ridge 沿用 ✓（inference 1.2 L19「R = ridge point = 峰值算力 ÷ 带宽」逐字）、models d(hidden)→d_model ✓ 规则在但 d_model 未入表（D-S2-4） | ✓ 规则一致（登记缺口另计） | 4/4 翻译规则 |
| V14 | 消歧硬规则 | L136：N/n/B 专归、tile T_M/T_N/T_K、bytes 不占字母、α/β 章首声明、R=ridge（I* 同物注）、货币 \$ 转义、混场章禁 N 表 rank——与 inference §4.3 硬规则同构；「n 专归」「N 专归」与 5.1/5.2 大纲冲突（D-S2-2/3） | ⚠ 两处硬规则被自设大纲违反（S2） | 9/9 条 |
| V15 | 1.2 O_max 草式单位 | §4.3 R_th=registers/thread、S_sh=bytes/block → ⌊R_file/R_th⌋=threads、⌊S_total/S_sh⌋=blocks、warp slots=warps；min 三项单位不齐；3.4 大纲流程正确（block size→blocks/SM→warps） | ⚠ D-S2-1 | 草式全项 |
| V16 | HBM BW 公式 | §4.3 L131 公式含 n_ch 因子 → bus width 须每通道位宽；2.1 L207「1024bit/stack」为每栈口径；H100 3.35 TB/s 两种代入自洽（5×1024×5.2G/8 = 5×16×64×5.2G/8） | ⚠ 口径歧义（D-S3-5），⟨verify 逐项⟩ 兜底 | 公式+用例 |
| V17 | R vs R_th / 未登记符号 | R（ridge）与 R_th 下标区分可成立；R_file/S_total（1.2 草式）未入表；混场章 3.4/11.2 宜声明 | ⚠ D-S3-8（minor） | 表全行 |

### Task 4 · 证据分级 §4.1

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V18 | [A] = paper+code（模型锚点另需 weights）与 STYLE_GUIDE §6 三者齐的死结 | gpu §4.1 L96 沿 inference §4.1 先例；§4.5 L146 显式计划序言豁免句扩至 gpu（「证据分级对硬件官方规格/无 paper repo 锚点的口径……以本 spec §4 为准」）——与 inference D-S2-7 修复同构，死结已预决 | ✓ | [A] 全例 |
| V19 | [C] 官方规格披露子类 = 硬件最高档 | §4.1 L98：[C] 含「官方规格披露·一手权威」子类，沿 inference [C]「开放技术博客、非闭源」子类先例（inference §4.1 L91/L94）；「最高证据档」限定于**硬件事实**（无 paper 可存在的域），非全局 [C]>[A]；媒体转述只作线索不作事实 ✓；§1 成功判据同口径 | ✓ 语义一致 | [C] 全子类 |
| V20 | 教学参照 + 官方实现（无 paper repo） | NCCL/CUTLASS/cuBLASLt/cuda-samples/nvbandwidth/gdrcopy/nccl-tests/DCGM/open-gpu-kernel-modules/DeepEP/FlashMLA/DeepGEMM/TileKernels——沿 inference §4.1 L93 先例；lint check-5（lint-chapter.sh L25）grep「教学参照」子串可过；豁免句防 evaluator 判 S2 ✓ | ✓ | 13/13 锚点 |
| V21 | [B] 老经典无 arXiv 子类 | Blelloch 1990 CMU-CS-90-190 / Rabenseifner 2004 LNCS 3036 / Patarasuk-Yuan 2009 JPDC 69(2) / DCQCN SIGCOMM 2015 / Al-Fares 2008——有 paper 无官方 repo → [B] 语义符合；「标原始出处 + 注明无 arXiv」沿 inference Orca 先例（inference §4.1 L94） | ✓ | 5/5 经典 |
| V22 | [W] 硬件 N/A + 五单值禁混合档 | §4.1 L99 [W] 硬件不适用、模型 config 例沿 inference [W]+指针 ✓；L101「等级只取五个单值，禁 [A/B] 混合档」沿 inference references 写入规则（inference W2 精炼说明「写入规则补『无混合档』约定」一手核实）；lint check-5 兼容 | ✓ | 全条 |
| V23 | Marlin [A] W2 升级 | inference/references.md L101 逐字「[A]（W2 核实论文 + 官方 repo，教学参照升 [A]）」+ arXiv:2408.11743；gpu §4.1/6.2 引用一致 | ✓ | 1/1 |
| V24 | 骨架 [A/W] 拒绝记录 | §4.1 L102「骨架初稿 1.1 曾写『官方 whitepaper [A/W]』——不成立……whitepaper 归 [C]」裁定记录在案 | ✓ | 1/1 |

### Task 5 · 站点计划 §8 vs 实际代码

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V25 | DIMENSIONS 条目结构 | build-content.mjs L41-90：每维度 key/parts/asserts 三字段；**references 不是条目字段** ✓ spec §8 L523 表述准确 | ✓ | 3/3 现存维度 |
| V26 | references 路径派生 + 硬读 | L210 `readFileSync(path.join(SRC,'references.md'))` 无 try/catch；SRC 由 key 派生（L101）→ gpu/references.md 须先于条目存在 ✓ spec 表述准确 | ✓ | 代码路径全核 |
| V27 | refsMin 地板 | L220-221 `if (refCount < dim.asserts.refsMin) throw` → W0 refsMin:50 要求种入 ≥50 条；spec fix 2（L598）与 §8 L523 表述与代码一致；inference W0 种 77 条先例（inference spec 波次记录 L519 一手核实） | ✓ | 断言全组 |
| V28 | W0 种入 ≥50 可行性 | 预核实附录可种条目清点：arXiv 10 + 老经典 5 + 已核 repo 12 + 待核 repo 4 + 已核官方文档 6 + 待核官方文档 ~18 ≈ 55 ≥ 50 | ✓ 可行（余量小，research pass 扩充） | 附录全条目 |
| V29 | UI 对象字段 | dimensions.mjs L91-148 inference 先例：key/zh/orderLabel/scope/lede/ledgerNote/depNoteShort/depNote/prereqs/prereqNote/firstChapter/graph；spec §8 L524 字段列全（key 由「字段照 inference 先例」覆盖）；orderLabel 'DIMENSION 04 / 05' 与 01/02/03 序列一致 | ✓ | 12/12 字段 |
| V30 | 三处硬读无兜底 → W1 TypeError | [dim]/index.astro L15 `const cfg = DIMENSIONS[dim]` 无兜底（L21/24/26/37/46 用 cfg.scope/orderLabel/zh/lede/depNote/prereqs）；index.astro L134 `DIMENSIONS[d.key].depNoteShort`（liveDims only）；DependencyGraph.astro L10 `DIMENSIONS[dim].graph` 无兜底。[dim]/index.astro L7-12 getStaticPaths 按 manifest chapters>0 过滤 → W0（0 章）不生成 gpu 页、W1（8 章）生成 → 缺 UI 对象即 W1 TypeError ✓ spec 表述精确 | ✓ | 3/3 硬读点 |
| V31 | FORTHCOMING.gpu 死代码 | index.astro L29 `DIMENSIONS[key] ?? FORTHCOMING[key]…` 兜底仅在 DIMENSIONS 缺条目时用；DIMENSIONS.gpu 建立后 FORTHCOMING.gpu 不可达 → 移除正确 ✓ | ✓ | 兜底逻辑全核 |
| V32 | live = chapterCount>0 | index.astro L26-28：chapterCount 从 manifest 读、live 自动翻转；代码无 live 字段 ✓ spec 表述准确 | ✓ | 1/1 |
| V33 | DIMENSION_ORDER | dimensions.mjs L151 已含 'gpu','cluster' ✓ 无需改 | ✓ | 1/1 |
| V34 | FORTHCOMING.cluster 文案 | dimensions.mjs L156 现文「EP、all-to-all、Ring Attention、网络拓扑」= spec §8 L529 指为过时的引文逐字吻合；W0 更新为 K8S/容器/Sandbox 计划 ✓ | ✓ | 1/1 |
| V35 | links.mjs 已 dim 参数化 | links.mjs L40-59 三规则：../references.md→/<dim>/references/（L45-46）、跨维度 README→GitHub（L48-51）、章链→/<dim>/…（L53-58）均 dim 参数化 ✓ 「W0 仅覆盖核对，无需扩展」准确 | ✓ | 3/3 规则 |
| V36 | figure-kit dim:'gpu' 路由 | build-content L13-31：模块自动加载（readdirSync *.mjs）、dim 缺省 'models'（L29）、sceneKey=dim/slug#idx（L27）、insertSignatureFigures 按 dim 过滤（L352）→ dim:'gpu' 仅为路由键 ✓。观察：spec 写「figures/gpu-pNN.mjs」，实际存量约定为 inference-partNN.mjs（9 模块核实）——注册表按内容不按文件名，命名偏差无害（inference spec 同型偏差先例） | ✓ | 注册表全路径 |
| V37 | QA 命令链 | site/package.json L8/10/12：build:content / build / qa（= build:content && build && pagefind && check-links && qa-shots）与 spec §8 L528 命令链一致；check-links.mjs / qa-shots.mjs / fig-check.mjs 存在 | ✓ | 5/5 命令 |
| V38 | qa-shots 可扩展 | qa-shots.mjs PAGES 平铺数组（inference 逐波追加先例 L26-49）；gpu 样章 2.4（W1）/6.3（W3）/8.2（W4）随波落地可行 | ✓ | 1/1 |
| V39 | DependencyGraph 建设期渲染 | DependencyGraph.astro L48-55：manifest 无对应 Part 的节点渲染为 pending 不可点 → W0 全 11 节点图 + 逐波 Part 落地兼容；14 边几何入 graph 计划与 inference 18 边先例同构 | ✓ | 1/1 |
| V40 | 根 README 行 | README.md L14 gpu 占位行 / L15 cluster 行「集群与通信：EP、all-to-all、Ring Attention、网络拓扑」（spec 引为过时逐字吻合）/ L17 边界句「跨机通信在 cluster」→ spec §8 L529 flip + 更新计划覆盖 | ✓ | 3/3 行 |
| V41 | [dim] 路由参数化 | site/src/pages/[dim]/{index,references,[...slug]}.astro 三页存在 → gpu 自动生效 ✓ | ✓ | 3/3 页 |

### Task 6 · 指针 errata §2

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V42 | 新 seam vs cluster 占位 | cluster/README.md L7-13 现 scope：EP a2a 通信实现 / Ring Attention 通信细节 / 集合通信与网络拓扑 / NVLink-IB 层级——与 spec §2 L42「全部重判到 gpu 具体章号」清单逐项对应；gpu 占位 README 旧 scope（内存层级/roofline/FA kernel）被新 P1-P6 覆盖 | ✓ | 4/4 scope 项 |
| V43 | ~73 归属句计数复现 | `/usr/bin/grep -rn -E '归 gpu\|属 gpu\|归 cluster\|属 cluster\|gpu 维度\|cluster 维度' models/ training/ inference/ --include='*.md'` 排除 README/references → **恰 73**（含 README/references 为 116）；命中行抽查无代码路径/公式标签假阳性 | ✓ 精确复现 | 全语料 sweep |
| V44 | 通信类重判估计 | cluster 指向句 38 处（章节 only），其中 36 处含通信/硬件关键词 → spec「约 25-35 处」略低（D-S3-3）；gpu 指向句 35 处主要需收窄章号 | ⚠ 估计校准（S3） | 73/73 分类 |
| V45 | W0 只盘点 / W1+ 收窄 | spec §2 L44「W0 盘点全部指针……W1 起随章号生效逐步收窄」+「仅限指针句的最小修订，不动任何公式与数值」+「逐条记入 gpu/README.md errata 表」——与 inference README L290 范式（W0 只盘点记录，不改文本）同构 | ✓ | 通道全条款 |
| V46 | 8 条重点重判句一手核实 | ① training 4.1 §2.2 L73「硬件侧归 cluster 维度」（NVLink/IB）+ L75「选择逻辑归 cluster 维度」（NCCL）✓；② models 2.8 §4 提点 L367「硬件互联归 cluster 维度」✓（inference W5 errata 后状态，gpu spec 引文与现文一致）；③ models 4.5 §7 L327「硬件与网络侧（all-to-all 的带宽与拓扑）归 cluster 维度」+ 总结 L345 ✓；④ models 4.6 总结 L287「通信实现归 cluster 维度」✓（另 §3 L227 含活链接 cluster/README.md，sweep 覆盖）；⑤ training 5.4 §6 L95「kernel 实现归 gpu/cluster 维度」✓（§6 = DeepSeek-V3 的 EP 部署，节号核实）；⑥ models 2.4 §6 L216/L244 + 1.2 §10 L245「微架构来源与测量方法归 gpu 维度」✓（inference W1 已重判 datasheet→inference 1.3，gpu 只接微架构来源与测量——分工链一致）；⑦ models 4.1 §5.4 L298 / §6 L302 / 总结 L316「指令级 kernel 实现归 gpu 维度」✓（节号逐一核实）；⑧ models 1.3 L192「集群侧属 cluster 维度」✓ | ✓ 8/8 位置与节号准确 | 8/8 |
| V47 | inference-kept 句再重判 | inference README L328「扫描后保留、无需收窄」列 models 4.6 总结 / models 1.3 L192——旧 seam 下保留，gpu spec §2 显式拾起重判 ✓ 无遗漏冲突 | ✓ | 2/2 |
| V48 | 跨维度公式引用 | training 4.1 式 (2) C_rank=2(d-1)/d·M（L62）/ 式 (3) T_ring=2(d-1)α+2(d-1)/d·Mβ（L70）+ 归纳证明命题 ✓；models 2.4 §6 式 (10)-(12) ✓（roofline 上限/ridge 为 §6 prose 非编号式，spec 节级引用正确，观察项）；inference 1.2 R=ridge ✓；training 4.1 练习 3 = 星型 vs ring + recursive halving（8.2/8.3 recap 引用准确）✓ | ✓ | 4/4 接口 |
| V49 | 兄弟 README 边界句 | models/README L3「训练算法、serving 系统、GPU kernel、集群通信分别归 training/、inference/、gpu/、cluster/」；training/README L3「网络与拓扑归 `cluster/`」；inference/README L3「跨机通信硬件归 `cluster/`」——三处活范围声明被新 seam 推翻，扫描模式匹配不到 → D-S2-6 | ⚠ S2 | 3/3 README |
| V50 | inference 侧 cluster 指针 | inference 8.3 L194（GDR 硬件协议层归 cluster）/ L210 边界声明（RDMA 协议/NIC/PCIe 拓扑/NVLink 域/交换机/NCCL 归 cluster）/ L325 总结行、8.1 L23、5.3 L125、3.3 L181、2.4 L181（CXL）等——全量 sweep 覆盖 ✓；预盘点无 inference 例（D-S3-3） | ✓ 覆盖（示例缺，S3） | sweep 全命中 |

### Task 7 · builder-check 附录 9 项修复

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V51 | 9 项修复记录在案 | 附录 L596-605 逐条：1 MLA 几何 / 2 refsMin 种子 / 3 Part5 W3 / 4 ring-tree 8.2/8.3 拆分 / 5 C 路 P11 / 6 走读 scope / 7 GPU 验证标准 / 8 P9 W5 / 9 §4.2 ring recap 拆分 | ✓ 9/9 | 9/9 |
| V52 | 修复落地到正文 | fix1→§5 6.4 L329-330「512 = 448 latent + 64 rope」+ 坑注「勿写 512+64=576，那是 V3 几何」+ 关键推导行同步 ✓（与 inference W4 errata 口径一致，inference 波次记录 L523 一手核实）；fix2→§8 L523 refsMin 地板 + W0 种 ≥50 ✓（与 build-content L220-221 一致）；fix3→P5 标题 L282「W3」✓；fix4→§6.3 L478「8.2 下界证明；8.3 临界 M*（W4 覆盖 8.2+8.3）」+ §4.2 L111 拆分 ✓；fix5→§10 L550「C：P7-P11」✓；fix6→§4.2 L115 每章大纲声明 scope ✓（6.2/6.4 大纲显式标记缺 → D-S3-6）；fix7→§6.2 L461 教学片段验证标准 +「实测」语义 ✓；fix8→P9 标题 L380「W5」✓；fix9→§4.2 L111「下界证明（8.2）+ 延迟交叉（8.3）+ NCCL 环构建（8.4）」✓ | ✓ 9/9 落地 | 9/9 |

### Task 8 · 波次 §7 / 质量门 §6

| # | 检查项 | 证据 | 结论 | 覆盖率 |
| --- | --- | --- | --- | --- |
| V53 | 每波图子循环 | §6.6 census→产图→图 QA（fig-check + QUALITY 五维 + Playwright）→W7 census sweep，与 inference §6.6（用户裁定 (b)）逐条同构；§6.5 commit gate 含图 QA ✓；§7 W1-W6 备注「含本波场景图」✓；签名场景 14 项为示例、实际按波 census | ✓ | 4/4 步骤 |
| V54 | R2 自始 | §4.6 写作输入 = STYLE_GUIDE + rubric + R2；12 规则逐条列名与 R2 style sheet §3 一致；练习四档/配额/校验锚/证命题与 R2 §4.3 一致；rewrite-gate 不适用（retrofit 门禁）✓。R2-2 硬件口径实例化缺（D-S3-4）；事实密集章证命题口径（D-S3-7） | ✓（两处 S3 润色） | 12/12 规则 + 4/4 配额 |
| V55 | 强制重推导重定界 | §6.3 清单：2.4「I/ridge recap models 2.4，不作新推导」、6.3「IO 定理 recap models 4.1，不作新推导」、8.2「消息量 recap training 4.1 式 2，不作新推导」——recap-not-re-derive 口径与 §4.2/§1 成功判据一致；越界检查（重推已交付公式即 S2）✓ | ✓ | 12/12 清单项 |
| V56 | commit gate | §6.5：审查+图 QA 才 commit ✓；README 进度表随提交 ✓；git status 逐目录核对（11 Part 含两位数 10/11，training glob 0* 教训）✓；不 commit 4 个 models/10-audio-speech 修改文件（与当前 git status 快照逐项吻合）✓；不 commit gitignored spec（git check-ignore 实测命中）✓；Co-Authored-By 尾注 ✓；每波 merge→main+push（main-based 长期授权）✓ | ✓ | 7/7 条款 |
| V57 | 连续执行授权 | §7「一路推进……rulings not stalls……仅 4 种停止条件」与 inference 管线 memory 口径一致 | ✓ | 1/1 |
| V58 | writer 负载 | §6.2 writer ≤2 章/人；W3 最密 8 章 → ≥4 writers 可行；brief 含修正后大纲 + 引用 key 白名单 + lint 门禁 + R2 ✓ | ✓ | 6/6 波 |
| V59 | lint 分支计划 | lint-chapter.sh check-4 case L12-17 现 models/training/inference 三支；加 `gpu/*) refs="gpu/references.md"` 可行；usage L2 同步 ✓；127 章回归（51+39+37=127 ✓）；lint-math-boundary.py rescape 规则已全局生效（L15/121/178 核实）且维度无关 →「不变」正确 | ✓ | 检查项 4/8 全核 |
| V60 | STYLE_GUIDE 泛化四项 | §4.5 计划：标题/适用范围行（L1/L5）、§10 指针（L152）、§11 usage（L168）、序言豁免句（L3）扩至 gpu——与 inference §4.5 四项同构；豁免句防逐字 evaluator 把 [C] 官方规格/教学参照判 S2 ✓ | ✓ | 4/4 项 |
| V61 | 跨维度章号指针抽查 | inference 1.4 §6 batch invariance ✓（L148 标题逐字）、inference 3.4 图捕获 ✓、inference 8.3 KV 传输 ✓、inference 7.3 FlashMLA ✓、inference 7.4 EP>32 递减 ✓、inference 9.1 成本 ✓、inference 1.3 datasheet 表 ✓、models 4.6 Ring Attention ✓、training 5.5 CP ✓；**training 7.3 checkpoint ✗（应为 7.4，D-S2-5）** | ⚠ 1 处误引（S2），余 ✓ | 10/10 抽查 |
| V62 | MLA 几何跨波一致 | 6.4 V4.1 512=448+64 / V3 576=512+64 两代区分 ✓；inference W4 errata（576→512，config 无 kv_lora_rank、引擎导出 448）口径沿用 ✓；models 4.1 L302 的 576 为 V3 几何（models 覆盖 V3）不冲突 ✓ | ✓ | 2/2 代际 |
| V63 | arXiv ID / repo 预核实继承 | 10 个 arXiv ID（FA/FA2/FA3/Marlin/NanoFlow/TileLang/FlashInfer/vLLM/DeepSeek-V3/Ring Attention）与 inference 附录已核实记录一致（Ring Attention 2310.01889 为 Liu et al. 2023 正确 ID，⟨verify W5⟩ 保留）；12 repo 200 清单计数 ✓；peer-memory 404 改锚 open-gpu-kernel-modules+gdrcopy 与 9.2 大纲一致 ✓；CUDA Programming Guide 404 显式留 W1 ⟨verify⟩ ✓ | ✓（深核留 A/B/C 路与各波 research pass） | 10 ID + 12 repo |
| V64 | 教学 kernel 验证标准（fix7） | §6.2 L461：macOS 无 NVIDIA GPU、片段标教学用、验证 = evaluator 审查 + Python 独立重算 + 编译非硬门；「实测」= 讲测量方法 + 引官方披露/已发表 nccl-tests 数值——与前三维 127 章无 GPU 执行先例一致，可规划 | ✓ | 1/1 |

### 观察项（不计 findings）

- O1：图模块命名 spec 写 `gpu-pNN.mjs`、存量约定 `inference-partNN.mjs`——注册表按内容加载（build-content L15-19），命名偏差无害，inference spec 同型先例。
- O2：Playwright 图 QA harness（fig-review-inference.mjs 先例）gpu 版需 builder 新建，§6.6 未点名——先例可迁移，builder 职责。
- O3：models 2.4 §6 的 roofline 上限/ridge 为 prose 非编号式，spec「式 (10)-(12)」为节级 lumped 引用（inference spec 同型），节号正确。
- O4：站点落地页 creed 行「证据分四级：[A] 论文+代码+权重齐……」为存量通用文案，非 gpu 引入的矛盾，超出本 spec 范围（inference 已共存）。
- O5：gpu/cluster 占位 README 现链接 gitignored spec 路径（GitHub 死链）为存量状态；spec §3 注明「finish 手动 cp 同步回 main 树」，W0 重写 README 时 builder 宜避免新增 gitignored 链接。
- O6：STYLE_GUIDE §7「走读锚点优先 2024-2026 模型」对 gpu repo 锚点（NCCL/CUTLASS 等非模型）的适用性由 §4.1 口径细化 + 序言豁免覆盖（inference 引擎走读同型先例，37 章无误判）；老经典只作公式源头不作走读对象 ✓ 与 Blelloch 等 [B] 定位一致。

---

## 3. Summary

| severity | 计数 | 编号 |
| --- | --- | --- |
| S1 | 0 | — |
| S2 | 6 | D-S2-1（1.2 occupancy 草式单位）/ D-S2-2（n 双重赋值）/ D-S2-3（GEMM M/N/K 未登记）/ D-S2-4（注意力几何符号缺行 + IO 定理 M 翻译）/ D-S2-5（11.3 training 7.3→7.4 误引）/ D-S2-6（W0 盘点漏兄弟 README 边界句） |
| S3 | 8 | D-S3-1（波内边枚举 3/5）/ D-S3-2（前瞻清单不全）/ D-S3-3（重判估计校准 + 预盘点补 inference 例）/ D-S3-4（R2-2 硬件口径）/ D-S3-5（HBM bus width 口径）/ D-S3-6（6.2/6.4 scope 显式标记）/ D-S3-7（事实密集章证命题口径）/ D-S3-8（R_file/S_total 入表） |
| 合计 | 14 | 跨路去重注：D-S2-1 与 A 路 P1 数学核查可能重叠，合并处理 |

**放行判定**：S1=0，无含 S1 章节复审触发。S2×6 全部为 spec 文本级修复（符号表三处登记/例外、一处草式重写、一处章号更正、一处盘点范围扩充），不动章节大纲结构与波次计划，修复成本低。D 路放行条件 = S2 全修复或附证据驳回；修复后 W0 可锁定大纲，writer brief 以修订后 spec 为输入。设计主体（依赖图/算术/站点计划/证据分级/errata 通道/波次质量门/builder-check 闭环）验证全过，可规划性成立。

**审查执行声明**：本审查全量读取 spec 609 行、STYLE_GUIDE 180 行、R2 style sheet 304 行、inference spec 相关节与附录；站点代码 11 文件逐行核对；已发布语料 grep sweep 全量（73 命中逐条分类）+ 15 处章节一手引句核对；全部结论附证据与覆盖率（V1-V64），无未附证据的「确认无误」。未修改 spec 或任何审查对象文件；本档案为唯一产出。
