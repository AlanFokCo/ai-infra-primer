# W4 fix-review · P7 S2/S3 修复复核 + X-1 跨波 errata 复核 + 5.2 残留裁定

日期：2026-11-23。re-reviewer：fresh-context（不继承 writer / errata agent 上下文）。范围：docs/reviews/inference-w4-review-p7.md 的 X-1（S1 跨波）+ S2×4 + S3×12 修复验证、mla-errata-report.md §8.1  flagged 残留（5.2 §6 L130 + 练习 2(c)）裁定、跨波 MLA 口径簇一致性、新破坏检查。只读审查（本报告除外），不改章节、不 commit、不派 subagent。

## 0. 方法与独立性

- **config 本轮自取**：hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json → HTTP 200 / 3,311 B（与 evaluator / errata agent 存档同尺寸），md5 e1e4964ca281289b9bc74de57a7ebcf5，存档 /tmp/w4-fixreview-evidence/v41-flash-config.json。python3 -I 全字段核：`head_dim=512`、`qk_rope_head_dim=64`、`num_key_value_heads=1`、`num_hidden_layers=40`、`index_topk=512`、`index_head_dim=128`、`max_position_embeddings=1048576`、`sliding_window=128`、`compress_ratios` 43 项首个 1 在下标 20、`kv_source_layer_ids=[2,8,14,20]`；**`kv_lora_rank` / `qk_nope_head_dim` / `v_head_dim` 字段不存在**（top-level 与 text_config 均无）。
- **DeepInfer tarball 本轮自取**：codeload @a679bb86ce36e883a98a97733492b686e5b9a08e → 200 / 2,331,722 B（与 evaluator 同尺寸），解压 /tmp/w4-fixreview-evidence/deepinfer-x/。关键锚逐条重核（§5 log）。FlashMLA main / ba89a34 双 README 用 errata agent 同日 fresh 解压（/tmp/mla-errata-evidence/flashmla-x / flashmla-old-x）抽查逐字。
- **独立重算**：/tmp/w4-fixreview-scripts/recompute.py（读自取 config，**93 项断言全 PASS**，不抄 evaluator / errata agent 值）；CJK 复核 /tmp/w4-fixreview-scripts/cjk.py。
- lint：11 个受影响章（1.2 / 2.1 / 2.3 / 2.4 / 3.1 / 5.2 / 5.3 / 7.1-7.4）`bash scripts/lint-chapter.sh` **全部 LINT OK**。
- 图模块：node 渲染 inference-part03.mjs FIGURES **12/12 过**，DSA 图 stale 值缺席 / 新值在场断言全过；part01-06 stale 模式 sweep 仅坐标误报。
- stale sweep：/usr/bin/grep -F 全 inference 维度 + 六图模块，模式 = 1.78 / 46080 / 12960 / 324 B / 12.7 / 22.5 / 56.9 / 45 KiB / 45 GiB / 45 MiB / 47.2 / 6.04 / 3.02 / 386.5 / 0.37 / 5.625 / 612 vs / 1.41 GiB / 1152 / 576（排除 README errata 记录与 7.2 L264 裁定引注——两者按设计引旧值）。

## 1. X-1 跨波数值 errata：**ADDRESSED**

裁定真值独立确认：V4.1-Flash 每 token 每层 **512 元素 = kv_lora_rank 448 + rope 64**（head_dim 512 已含 rope；config 无 kv_lora_rank 字段，引擎 config.py L289 导出 448）。四方一手互证本轮全部重核成立（§5.1-5.4）。

逐章复核（行号为修后定位，全部与 mla-errata-report §3 声明一致）：

| 章 | 位置 | 复核结果 |
| --- | --- | --- |
| 1.2 | L53 坑注 | ✓ 「head_dim 512 是共享 KV 向量维（已含 rope 64）」「较真值 448+64=512 高估 2.0×」「V4.1 config 无 kv_lora_rank 字段…引擎导出…见 7.2 §9」+ 两代勿混注；旧「config 读 kv_lora_rank」句已删 |
| 1.2 | L63 表行 | ✓ 「MLA（V4.1-Flash，448 + 64）[W]｜512×2｜1024 B｜40 KiB（40 层）」 |
| 1.2 | L66 | ✓ 「MLA（V3 口径）的 576 元素」代际标注；κ=(448+64)·bytes=512·bytes、1024 B/token/层、40960 B=40 KiB/token |
| 1.2 | L235-236 / L251 / L262 | ✓ 汇总表 V4.1 列 1024 B（MLA 512 元素）/ 40 KiB；总结 2.0×（1024 vs 512）；练习 1 (a) head_dim 含 rope 注 + (b) 校验 2.0× |
| 2.1 | L240 / L295（X-1 清单外同族） | ✓ 「共享 KV 向量 512 元素 = d_latent 448 + d_rope 64；head_dim 512 已含 rope，kv_lora_rank 448 由引擎导出，见 1.2 §3」；参考节同步 |
| 2.3 | L15 / L90 / L92 / L142 / L163 | ✓ 章口径 512 元素；§4 推导 512/16=32 块、256+32=**288 B**、bf16 1024、3.56× 不变、11520 B=**11.25 KiB**、满窗 **11.25 GiB** / bf16 40 GiB；坑注 **544 vs 288**（1.89× 不变）；总结 / 参考同步 |
| 2.4 | L9 / L159 / L175 / L199 / L204 / L221 | ✓ 章脊 **40960 B=40 KiB** / 1.25 GiB；§8 「512 维共享 KV 向量（448 latent + 64 rope）」（528 / 1,415,040 / 1.35 MiB 不动）；§9 40 / 11.25 KiB + **3.2× / 11.4×**；总结 / 练习 1 / 参考同步 |
| 3.1 | L274 / L277 / L303 式 (10) / L321 | ✓ 走读图 20 MiB；「128 对主路 512」；κ=512×2=**1024**、M_main=512×1024×40=**20 MiB**（全量 5 GiB、0.39%）、M_idx 1.25 GiB=**64×**、MXFP4 340 MiB 仍 **17×**（3.76× 不变）；总结 64× |
| 3.1 图 | inference-part03.mjs dsaSelectChain | ✓ L160/L193/L200/L201 同步（512×1024×40=20 MiB / 5 GiB / 64× / 17×）；渲染 12/12 过、stale 缺席新值在场 |
| 5.3 | L266 / L291 / L293 / L380 / L391 | ✓ §7 开场 20 MiB；式 (4) κ=1024 + 512 元素注、hot **41.9 MB（40 MiB）** / 全量 5.37 GB / B=64 **2.68 / 343.6 GB**；换入 **20 MiB / 0.33 ms**；总结 + 练习 3(d) 校验 0.78% / 2.68 GB / 0.33 ms |

**独立重算**（93 断言全 PASS，脚本读自取 config）：1.2 κ=1024 / κL=40960=40 KiB / 误代 2.0×；2.3 FP4=256+32=288 / 3.5556≈3.56× / 11520=11.25 KiB / 满窗 11.25 GiB / bf16 40 GiB / 544÷288=1.8889≈1.89× / MXFP8 528；2.4 1.25 GiB@32768 / 128÷40=3.2× / 128÷11.25=11.3778≈11.4× / 20×134×528=1,415,040=1.3495 MiB；3.1 M_main=20 MiB / 全量 5 GiB / idx 1.25 GiB=64× / 0.390625% / MXFP4 340 MiB=17× / 2÷0.53125=3.7647；5.3 hot 41.94304 MB / 5.36870912 GB / 2.68435456 / 343.59738368 GB / 换入 0.32768 ms / f_hot 0.78125%。比值类（3.56× / 1.89× / 3.76× / 0.78% / 0.39%）线性缩放不变，与裁定一致。

**V3 行 UNCHANGED 确认**：1.2 L62（576×2=1152 B / 68.6 KiB 61 层 / 10.7 GiB@163840，重算 68.625 / 10.7227 与原 rounding 一致）；2.3 L124（576 vs 32768、56.9×=56.889）；3.1 §5 吸收形（576 / 64×576 / 9 次 TMA / 241.8=128×1088÷576 / 57×=32832÷576 / 656=512+128+16，V3 口径 [A] 声明在 §5 首）；6.3 L48/L153（明锁「recap 1.2 §3 几何表 V3 行」61 层 / 1152 / 70272 / 页 72 KiB，**git diff 空**）；5.3 L151 supports_head_size 指针「3.1 §5（MLA 512 + 64）」V3 吸收形不动；图 part02 L334 / part05 L392 / part06 L174/L503 V3 口径标签未动（part05 见 §3 裁定）。

**Stale grep clean**：全维度 + 六图模块零 V4.1-attributed 旧值。唯一命中类：① README errata 记录与 7.2 L264 裁定引注（按设计引旧值）；② 2.3 L124 V3 口径 56.9×（裁定保留）；③ 坐标 / arXiv ID / 1048576 / 11520 子串误报；④ **5.2 L130 / L219 残留（§3 裁定）**。六章残留 576/1152 逐条审计：全部 V3 行 / V3 recap / 「V3 的 576 不适用 V4.1」区分注 / 误报。

**README 登记**：errata 表 6 行（L252-257，逐章 old→new 全值 + V3 不动声明）+ 「W4 跨波数值 errata（非指针 errata，X-1 裁定执行）」专段（L265，四方证据链 + 执行范围含清单外同族声明 + 重算记录 + 观察项）✓。git diff --stat 确认 errata 仅改六章 + part03.mjs + README（references.md diff 为 W4 research pass 精炼、models 10-* / training 8.5 为无关先在改动）。

## 2. P7 S2 逐条 verdict：**4/4 ADDRESSED**

| # | verdict | 复核 |
| --- | --- | --- |
| S2-1（7.1 κ 错口径） | **ADDRESSED** | L223 「κ=512·bytes（head_dim 512 含 rope 64，引擎 kv_lora_rank=448，读法见 7.2 §9）」；「1.2 回链」已删改指 7.2 §9。配套：L244 第 5 处读法 bullet（config.py L288-289 注释逐字 + 448 导出 + V41_FORMAT 512）、L246 新坑注（V3 几何套 V4.1 重复计 rope，四方互证，576 非合法替代口径）、L292 总结行（512 元素=448+64）。remnant：(512+64) 全章 0 命中 |
| S2-2（7.3 d_latent 512 错） | **ADDRESSED** | §1 L28 「共享 KV 向量 512 维 = kv_lora_rank 448 + rope 64（head_dim 512 含 rope，引擎导出 448，读法见 §3）」；§3 L56 「V4.1 head_dim_k 降为 512（=448+64），d_rope 保持 64；旧 README 576 为 V3 系（kv_lora_rank 512 + rope 64）」+ 引擎读法四方证据 + 明示「§4 的 528 B = 512 元素 + 16 scale 而非 576 元素」——与 §4 表（512 数据+16 scale / 256+32 / 64+4）自洽，「几何随代际变」论断保留且由此才自洽。§3 Hopper recap 576/656 显式 V3 系不动 |
| S2-3（7.2 口径注） | **ADDRESSED** | L264 重写：引擎读法 512 元素为一手权威（config.py L289 注释逐字 + V41_FORMAT L109 + FlashMLA main「covering all 512 dimensions」+ doc L223 四方）；1.2/2.3 的 576/324 B 标**事实错误**（head_dim 误当 kv_lora_rank、重复计 rope、V3 误套）非合法口径，指 X-1 / README 审查记录区；「归 7.3 对账」悬空指针已删，改「kernel 栈侧格式细节见 7.3」（7.3 §3/§4 现有真实对账内容，指针落地）。注：errata 已执行后该注读作裁定记录（引旧值按设计，与 README errata 同型），非悬空 |
| S2-4（7.1 B_max 违 §4.3） | **ADDRESSED** | 式 (1) L212 / 符号表 L215 / 正文（nextn7→δ=8→136）/ 练习 2 L299（校验式 128+δ≤140）全章 B_max→δ，符号表注「最大回溯深度，与 7.2 同符号」；B_max / B_{\max} remnant 0；与 7.2 §1 δ=6 对齐 |

**S3 逐条复核（12/12 修复成立或按裁定留 closeout）**：S3-1 L51 锚 **L1239-1240** ✓（fresh tarball：L1239 `prev_accept_rates = std::move(accept_rates)` / L1240 clear / L1243 `float accept_rate = 1.0f`，原锚偏确认）；S3-2 7.1 L97 三行独立 import 逐字 ✓（_C/__init__.pyi L2-4 核）+ 7.2 L276 docstring 恢复「因为」「（落盘粒度）」逐字 ✓（cache_engine.py L104 核）；S3-3 L276 EXTRA_STATE 前提 ✓（environ.py L170 `EnvVar.bool()` + L77 默认 False + cache_engine L108-110 条件式核；开启且 compact 140→402、默认关闭→134）；S3-4 7.3 L75 改引 **2.3 式 (4)** + 「FP 格式 zp=0；3.2 式 (3) 是含 zp 推广」✓（3.2 L51 式 (3) 确含 bytes_zp、2.3 L83 式 (4) 无 zp；映射表 L39 / 表 L79-81 / L83 / 总结 L161 指针同步 3.2 §2 格式表）；S3-5 7.4 L70 μ,σ 消歧 ✓；S3-6 7.2 L67 τ 局部阈值消歧 + 7.3 L138 δ=kernel 慢比例消歧 ✓；S3-7 7.4 L52 「完全均衡（perfectly balanced）」与 L125 / 1.1 统一 ✓（完美均衡 grep 0）；S3-8 式 (6) 后 s=9856/i=9999→[9872,9999] + 段首 [9856,9856]、式 (8) 后 T=10000→cap 9866/H_1 9856 一行算 ✓（R2-3 节奏补齐）；S3-9 命题 4(3) 陈述 L228 + 证明 L230 「与存储态逐字一致无条件；与全量重算 bit-exact 另需 batch-invariant（README L27 / 3.4 §6）；数学相等由前缀不可变」✓；S3-10 7.1 L141/L175/L300 「while True 内 step/flush 三元 + break」✓（llm.py L176-186 句法核）；S3-11 7.4 L74 措辞改「spec §2 已裁定重判…留 W4 closeout errata 执行（截至本章核实原文尚未改）」✓（training 5.4 L79 / models 4.5 L327/L345 原文确认仍在，git 未动——closeout 条件项未触发升级）；S3-12 refs 被引回写七处确认仍缺（v41-flash-config 缺 7.3 / tilelang 缺 7.2 / flashmla 缺 7.2 / nanoflow 缺 7.1 / vllm-repo 缺 7.1 / sglang-repo 缺 7.1 / 3fs 缺 7.1）——closeout 项，writer 按 brief 未动，正确。

## 3. 关键残留裁定（5.2 §6 L130 + 练习 2(c)）：**fix-to-V4.1**

**裁定：两处均为 X-1 事实错误的清单外传播（V4.1-attributed / 未标代际而指针解析到 V4.1 源），非合法 V3 例。修到 V4.1 口径（512 元素 / κ=1024 B / page 64 KiB），不走 keep-V3 标注路线。**

证据与推理：
1. **L130 未标代际且指针现已矛盾**：原文「代入 MLA 几何（d_latent=512、d_rope=64，**1.2 / 2.1 口径**，缓存维 d_K=576）」。errata 后 2.1 §6 明写 V4.1-Flash「共享 KV 向量 512 元素 = d_latent 448 + d_rope 64」——指针落点与例中 576 直接矛盾；1.2 §3 则两行并存（V3 576 / V4.1 512），指针歧义。
2. **provenance 确认（git diff 旧文）**：errata 前 2.1 §6 原文 =「V4.1-Flash …MLA 几何（d_latent=512、d_rope=64）」、1.2 V4.1 行原文 =「MLA（V4.1-Flash，512+64）｜576×2｜1152 B｜45 KiB（40 层）」。5.2 两处数值与**旧 V4.1 行逐字同源**——W3 写作时两章 V4.1 口径尚带 X-1 错，5.2 是照抄错误 V4.1 口径，不是有意取 V3。数值恰与 V3 重合（V3 rank 512）是误代巧合，不构成 V3 归属。
3. **练习 2(c) 的 40 层是 V4.1 层数**（V3=61 层）：「κ=1152 B/token/层、40 层」= 旧 1.2 V4.1 行逐字，混口径即 X-1 传播；校验锚「与 1.2 MLA 带宽账方向一致」现解析到 1.2 V4.1 行（1024 B）矛盾。
4. **keep-V3 路线代价更大且违 provenance**：须 (a) L130 补「V3 口径」+ 指针改「1.2 §3 V3 行」（2.1 §6 已无 576），(b) 练习 2(c) 40 层→61 层并重算（分母 64×61×1152=4,497,408 B=4.2890625 MiB、num_blocks=11459，本轮重算），改动面大于 κ 修正，且把本章 MLA 例留在与 2.1 §6 / 5.3 生产口径（V4.1）不同的代际。按任务判定规则（V4.1-attributed / generic-unlabeled → 同一事实错 → 修 512/1024）裁定 fix。

**精确修正（留 fix agent / closeout 执行，本轮只读未动）**：
1. **5.2 L130**：「代入 MLA 几何（ $d_{\mathrm{latent}}$ = 512、 $d_{\mathrm{rope}}$ = 64，1.2 / 2.1 口径，缓存维 $d_K$ = 576）： $H_{kv}$ = 1、state_content = 576×2 = 1152 B = $\kappa_{\mathrm{MLA}}$，page = 64×1152 = **72 KiB**/层/块」→「代入 MLA 几何（V4.1-Flash：共享 KV 向量 512 元素 = $d_{\mathrm{latent}}$ 448 + $d_{\mathrm{rope}}$ 64，1.2 §3 / 2.1 §6 口径，缓存维 $d_K$ = 512）： $H_{kv}$ = 1、state_content = 512×2 = 1024 B = $\kappa_{\mathrm{MLA}}$，page = 64×1024 = **64 KiB**/层/块」。同句「MLA 块几何由 latent 维定」宜改「由共享 KV 向量维（latent+rope）定」（d_K 含 rope，原措辞两代均松）。代码忠实性不受影响：MLAAttentionSpec 的 head_size 来自 config，例为教学代入非代码常量。
2. **5.2 L209 总结行**：「GQA 256 KiB / MLA 72 KiB 每层每块」→「GQA 256 KiB / MLA 64 KiB 每层每块」。
3. **5.2 L219 练习 2(c)**：「κ = 1152 B/token/层、40 层」→「κ = 1024 B/token/层、40 层」；校验「(c) 分母 = 64×40×1152 ≈ 2.81 MiB，num_blocks ≈ 17476」→「(c) 分母 = 64×40×1024 = 2.5 MiB，num_blocks = 19660（48 GiB ÷ 2.5 MiB = 19660.8 向下取整）」。方向论断不变（19660 > 6144，MLA κ 小→同显存切更多块），「与 1.2 MLA 带宽账方向一致」落地到 V4.1 行。本轮重算：64×40×1024=2,621,440 B=2.5 MiB 精确、51,539,607,552÷2,621,440=19660.8 ✓。
4. **site/scripts/lib/figures/inference-part05.mjs**：L323 「MLA 72 KiB」→「MLA 64 KiB」；L392 「page = 64 × 1152 = 72 KiB/层/块」→「page = 64 × 1024 = 64 KiB/层/块」（5.2 两图同步）。
5. **README errata 登记**：X-1 同族行（5.2 §6 + 总结 + 练习 2(c) + part05 图，old→new 全值，注明裁定依据本 fix-review §3）。
6. **同簇可选标签卫生（非阻塞）**：part02 L334 「几何：MLA 576 元素」补「V3」tag（内容对应 2.3 §6 V3 recap 行，正确）；part06 L174 「kv_buffer [rows, 1, 576]」为 SGLang V3 系池口径（6.1 正文符号形、P6 锁 V3），可补「V3」tag。两处是 V3 内容缺代际标签，非事实错。

## 4. 跨波 MLA 口径簇一致性：**clean（除 §3 已裁定残留）**

- 全维度 V4.1-Flash MLA 引用一律 **512 元素（448+64）**：1.2 / 2.1 / 2.3 / 2.4 / 3.1 §7 / 5.3 / 7.1 §5 / 7.2 §9 / 7.3 §1/§3/§4 九处口径簇互相一致，派生值（1024 / 528 / 288 / 40 KiB / 11.25 KiB / 20 MiB / 64× / 0.33 ms）全链闭合；7.1/7.3 的「待 errata」指针经本次 errata 全部落地。
- V3 引用一律 **576（512+64）且显式代际**：1.2 L62/L66/L235 V3 列、2.3 L124、3.1 §5（V3 口径 [A]）、6.3（明锁 V3 行）、7.3 §3 Hopper recap / 总结 / refs、5.3 L151 指针。6.3 / 5.2 git diff 空（V3 内容未误伤）。
- 图：part03 DSA 已同步（§1）；part02 L334 = V3 行内容正确（可选 tag）；part05 L323/L392 随 §3 修正同步；part06 V3 口径不动。part01/04 无 MLA 几何值。
- 无混合 / 未标代际的 V4.1 口径残留（§3 两处除外，已裁定）。models 4.1 的 576 为 V3 模型侧推导，正确不动。

## 5. Verification log（抽查项 × 来源 × 结论）

### 5.1 config（本轮自取 hf-mirror，200 / 3,311 B）
全字段核 §0 所列；kv_lora_rank 不存在 ✓；compress_ratios 43 项首 1 在 20 ✓；kv_source [2,8,14,20] ✓。与 evaluator / errata agent 存档同尺寸（md5 本轮 e1e4964c…）。

### 5.2 DeepInfer fresh tarball（本轮自取 codeload @a679bb86，200 / 2,331,722 B）
- hugging_face/config.py **L288-289**：注释逐字 "HF's shared KV vector includes RoPE; the internal rank excludes it." + `kv_lora_rank=text["head_dim"] - text["qk_rope_head_dim"]` ✓（X-1 / S2-1/2/3 权威锚）。
- tilelang_ops/cache_common.py **V41_FORMAT**：quantized_elements=512 / elements_per_sf=32 / bf16_elements=0 / e4m3+ue8m0 ✓；**V41_FP4_FORMAT**：e2m1 / 512 / per-16 / e4m3 sf ✓（528 / 288 格式常量）。
- worker/cache_engine.py **L54** assert `kv_lora_rank + rope_head_dim == layout.num_elements` ✓；**get_kvstore_swa_block_size L101-110** docstring 逐字「因为 kvstore 落盘只在 128（落盘粒度）的整倍数落，prefill 时可能不会触发落盘，多存一段才能保证多轮命中。」+ EXTRA_STATE 条件式 ✓（S3-2/S3-3）。
- csrc/engine/scheduler.hpp **L1239-1240** prev shift / clear ✓，L1243 `float accept_rate = 1.0f` ✓（S3-1 原锚偏确认）。
- environ.py **L77** bool 默认 False / **L139** MAX_BACKTRACKING=max(v,nextn+1) / **L170** PREFILL_STORE_EXTRA_STATE=EnvVar.bool() ✓（S3-3 / 式 (1)）。
- entrypoints/llm.py **L176-186** while True + unfinished 三元 + break ✓（S3-10）；_C/__init__.pyi **L2-4** 三行独立 import ✓（S3-2）。
- doc/inference_deployment_qa.md **L223** 逐字「20 layers × 134 tokens × 528 B = 1,415,040 B … The 528 B/token is the MXFP8 SWA KV format」✓（X-1 证据④）。

### 5.3 FlashMLA 双 README（errata agent 同日 fresh 解压抽查）
main：「For DeepSeek V4.1 (head_dim = 512) … 528 (V4.1) or 288 (V4.1 fp4) … the first 512 bytes are float8_e4m3 values covering all 512 dimensions. The 64 RoPE dimensions are quantized as well, so there is no bfloat16 part」+ 288=256 B（512 e2m1）+32 B per-16 e4m3 ✓。旧 ba89a34：「head_dim_k = 576 (for DeepSeek V3/V3.1/V3.2) or 512 (for DeepSeek V4/V4.1)」✓（576=V3 几何的代际证据）。

### 5.4 数值 / lint / 图 / 篇幅
- recompute.py **93 断言全 PASS**（§1 列关键值；另含 7.1 43=40+3 / nextn7→136 / v10→138、7.2 528/288/68/70752=32×2211/1415040/ca_state 28672/ica 7168/402/77/lcm 1056/w² 16384/锥 2560/减半 0.50195、7.3 1.03125/0.5625/0.53125/431.6/433.7/f≤0.5/700/190、7.4 moe_lb 91.7/93.1/95.7/CV 0.072/0.408、5.2 残留两分支全值）。
- lint 11 章全 LINT OK；CJK（剥围栏/display/行内/code span）7.1=4817 / 7.2=7210 / 7.3=4179 / 7.4=5673，均达标（7.2 ≥5500）。
- part03 FIGURES 渲染 12/12；DSA stale（22.5/56.9/5.625/1152/主路 576/仍 15×）全缺席、fresh（20 MiB/64×/5 GiB/17×/主路 512/512×1024×40）全在场。
- stale sweep 全维度 + 六图模块：零 V4.1 旧值（例外类见 §1）；B_max / (512+64) / while-else / 完美均衡 remnant 0。

## 6. 新破坏检查：**无**

- 公式 / 证明：3.1 式 (10)、5.3 式 (4)、2.3 式 (4) 代入、1.2 式 (3) 表、2.4 StateBlock 账在新数值下全链自洽（93 断言）；7.2 四命题证明未受 errata 触碰（数值锚 528/288/134/402 等本就正确）；7.3 命题 1 / 7.4 命题 1-3 数值复核全中。
- 练习 / 校验锚：1.2 练 1（2.0×）、2.4 练 1（528=512+512/32）、5.3 练 3(d)（0.78%/2.68/0.33）、7.1 练 2（128+δ≤140）、7.2 练 3（cap 反例）、7.3 练 1（528/288/68）全一致；3.1 练 3 用 8B κL=128 KiB 不受影响。
- V3 行 / 6.3 / 5.2 git-intact；README / references 无误伤（references diff = research pass 精炼）。
- **非阻塞观察项（不立 finding）**：① 7.2 总结 L295 / 练习 3(c) 沿用「block KV 复用 bit-exact」简写，未带 S3-9 新增的「与存储态」限定——按存储态读法无条件成立、与修后命题一致，可选补「与存储态」三字求全精确；② 7.2 L264 口径注以现在时引 1.2/2.3 旧值作裁定记录，errata 后读作历史记录（按设计，指针 README 落地），可选改「errata 前」措辞；③ part02/part06 V3 标签可选代际 tag（§3.6）。

## 7. 结论与 closeout 清单

- **X-1：ADDRESSED**（六章 + 2.1 同族 + part03 图 + README 登记，93 断言独立重算全过，V3 行不动，stale clean）。
- **P7 S2：4/4 ADDRESSED**；S3 12/12 修复成立或按裁定留 closeout（S3-11/12）。
- **5.2 残留：RULING = fix-to-V4.1**（§3 精确修正六项，含 part05 图同步 + README 登记；执行前 5.2 L130/L219 为全维度唯一未闭合 X-1 同族位置）。
- **跨波一致性：clean**（除已裁定残留）；**新破坏：无**。
- closeout 必做：① 执行 §3 修正 + README 登记；② S3-11 三处指针 errata（training 5.4 L79 / models 4.5 L327/L345，原文确认未改，7.4 措辞已正确声明）；③ S3-12 refs 被引回写七处（缺口确认仍在）；④ 站点全量 QA（build + check-links + qa-shots）+ 统一 commit。