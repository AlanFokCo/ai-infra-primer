# W6 Fix 轮复审报告 · P9 生产全流程（9.2 / 9.3 / 9.4）

日期：2026-11-24。Re-reviewer：fresh-context 独立复审（不继承写作上下文）。只读复审：未改章节、未 commit、未派 subagent；本报告为唯一写入。
触发：P9 审查（docs/reviews/inference-w6-review-p9.md）S1=1（9.3 §2 vLLM sleep level 1 权重处置写反）→ 修复后强制复审 9.3；同时复核 S3=6 修复与机械项、查新破坏。
一手来源：本轮自行 fresh fetch（不信任 writer 报告与既有证据目录）——vLLM codeload tarball @6aeedab553dd + contents API raw 四文件（gpu_worker / cumem / sleep_mode_backend / llm）+ 当前 main 1d9f8058bf8b 对照；veRL @7b6c9cc0e19d、SGLang @ac0243723b0d、TRT-LLM @a41224a3dee2、DeepInfer @a679bb86 contents API raw。证据 /tmp/iw6-fixreview-evidence/，脚本 /tmp/iw6-fixreview-scripts/。

## 0. 总判定

- **Status：PASS。S1-1 ADDRESSED（fresh 四文件互证 + 当前 main 字节级稳定）；S3 6/6 ADDRESSED；无新破坏。**
- level-2 draft.* concern 裁定：**无需展开**（章已给集合补实现语义，覆盖该情形；可选一句注，非阻断）。
- 「level 2 是 RL colocate 自然选择」：已声明机制层读法、非 veRL 默认值断言，**接受**。
- Closeout 残留（非本轮 finding，见 §5）：研究笔记语义纠正 4 / L115 level 1 误口径须 errata；references.md tensorrt-llm W6 注 overlap「CPU n-1」措辞与 overview.md 路径精度；9.2 被引回写。

## 1. S1-1 验证（CRITICAL，本轮 fresh fetch）

### 1.1 Fresh 证据（file:line × 引句 × 结论）

vLLM @6aeedab553dd（= 章节锚定 HEAD；当前 main = 1d9f8058bf8b3354214b0f4a4dafdac25bfb4be1，2026-11-23T16:24:47Z，gpu_worker.py 两 commit **diff 0 行**——语义与行号在今日 HEAD 亦稳定）：

| 文件:行 | 引句 / 事实 | 结论 |
| --- | --- | --- |
| vllm/v1/worker/gpu_worker.py L361 | `def sleep(self, level: int = 1)` | sleep 入口行号 ✓ |
| 同上 L367-372 | 注释「Level 2 keeps everything the weight update doesn't rewrite.」；`discardable = (self._discardable_parameters(...) if level == 2 else set())` | **level 1 → discardable = ∅** ✓ |
| 同上 L338-339 | `_discardable_parameters` docstring 逐字「Storage of the parameters the weight update rewrites.」 | 章引句逐字 ✓ |
| 同上 L344-349 / L359 | preserved 集来自 `sleep_preserve_parameter_names` fnmatch；`return set(storage.values()) - preserved` | **discardable = 全参数 storage − preserved**（被改写集丢弃、非改写集保留）✓ |
| 同上 L350-358 | draft.* 不在 preserve 名单 → warning_once「Level 2 sleep discards draft model weights; reload_weights restores only the target…set --sleep-preserve-parameter-names 'draft.*' to keep them.」 | concern 裁定输入（§3） |
| 同上 L381 / L402 / L409-410 | `self.sleep_mode_backend.suspend(discardable)`；`def wake_up(tags)`；wake_kv_cache 时 `get_kv_transfer_group().restore_kv_caches()` | suspend 传 discardable ✓；wake_up 重建 kv_transfer group ✓ |
| vllm/device_allocator/sleep_mode_backend.py L56-60 | 基类 suspend docstring 逐字「…the parameters not selected for preservation at level 2, **nothing at level 1**. A backend doesn't need to know the level. **KV cache and workspace scratch are never restored.**」 | level 1 无丢弃集 + KV 永不恢复，逐字 ✓ |
| 同上 L126 / L130-132 | `CuMemBackend.suspend` → `get_mem_allocator_instance().sleep(("weights", "runtime", "cudagraph"), discardable)` | 三 tag 池 offload、kv_cache 不在列表 ✓ |
| vllm/device_allocator/cumem.py L62 / L261 | `class CuMemAllocator` / `def sleep(offload_tags, discardable)`；docstring「All data in the memory allocation with the specified tag will be offloaded to CPU memory, and others will be discarded.」 | 类 / sleep 行号 ✓ |
| 同上 L292 | `offload = data.tag in offload_tags and ptr not in discarded` | 章引 L292 判据逐字 ✓ |
| 同上 L322-325 / L330 | offload → `cpu_backup_tensor = self._create_cpu_backup(...)` + cudaMemcpy D2H；两路均 `_unmap_and_release` | 权重 offload = host 备份 + GPU 物理页释放 ✓ |
| 同上 L364 / L366-367 / L385-387 | `def wake_up(tags)`；docstring「All data that is previously offloaded will be loaded back to GPU memory, and the rest will have empty memory」；`if data.cpu_backup_tensor is not None: …cudaMemcpy(ptr, cpu_ptr, …)` | **wake_up 自动从 host 备份恢复；无备份则空显存** ✓ |
| 同上 L370-372 / L381 / L394 | deferrable tags = (weights, kv_cache)，其余 tag 总是恢复；`use_memory_pool` L394 | wake_up(tags) 选择性恢复 ✓ |
| vllm/entrypoints/llm.py L853 / L897 / L587 | sleep / wake_up / collective_rpc 行号未漂移 | ✓ |
| 同上 L864-868（第四证） | Level 1 docstring 逐字「**Offload model weights to CPU, discard KV cache.** …Good for sleeping and waking up the engine to run the same model again. Please make sure there's enough CPU memory to store the model weights.」 | 公开 API 口径与修复后章节完全一致；host 备份代价注有官方对应 |
| 同上 L869-874 | Level 2 docstring「Discard model parameters and KV cache, and offload the rest of the model's GPU state to CPU. Good for …update the model's weights, where the previous weights are not needed. It reduces CPU memory pressure.」 | level 2 = 丢弃（被改写）参数 + KV、其余 offload ✓ |

**语义结论（fresh fetch 确认）**：level 1 = 权重 offload 到 CPU（host 备份，wake_up 自动恢复，无需重新注入）+ 仅 KV 丢弃；level 2 = discardable（被改写集 = 全参数 − preserved）丢弃无备份（wake_up 后空显存、由权重更新填回）、preserved（非改写集）offload 到 host 恢复；KV 两 level 均丢弃。**修复后 9.3 §2 与此完全一致；原 S1「level 1 丢弃全部权重 / 须重新注入」确认为误，已清除。**

路径注：dispatch 提示「v1/worker/gpu/cumem.py」与实际树不符——本轮 contents listing + tarball 全树 listing 确认 cumem.py / sleep_mode_backend.py 在 **device_allocator/** 下（v1/worker/gpu/ 目录无 cumem.py）；章节路径正确。

**codeload tarball 交叉核（主通道，已完成）**：首轮 480s 超时（24.1MB 已收）、range 续传不支持（exit 33），重启全量下载成功——**53,630,685 B，md5 d19ff4d2581bc7dfbc073530afc2c075**，根目录 vllm-6aeedab553dd/。tarball 抽出四文件与 contents API raw 逐一 **cmp 字节级相同**（cumem.py / sleep_mode_backend.py / gpu_worker.py / llm.py）；tarball 副本关键行再核：sleep L361 / `else set()` L371 / wake_up L402 / 「nothing at level 1」L58 / offload 判据 L292 / 「Offload model weights to CPU」L864 全中；weight_transfer/ 六 engine（含 sparse_nccl_engine.py）在树。**两独立通道互证，S1 证据链闭合。**

### 1.2 章节 echoes 逐项核对（9.3）

| 位置 | 修复后内容（要点） | 判定 |
| --- | --- | --- |
| §2 L30 level 1 条 | 权重 offload 到 CPU、只丢 KV；discardable=∅（L368-371）；suspend 三 tag 池 host 备份（cumem L292）；wake_up L386-387 cudaMemcpy 自动拷回、无需重新注入；kv_cache 永不备份（docstring 逐字） | ✓ 与代码逐句对应 |
| §2 L31 level 2 条 | 丢弃被改写集、保留非改写集；discardable = docstring 引句 + 实现「全参数减去 sleep_preserve_parameter_names 声明的 preserved 集」；被改写集无 host 备份、wake_up 后空显存由权重更新填回；preserved offload 恢复；KV 同样丢弃 | ✓（S3-1 术语同修，见 §2） |
| §2 L32 wake_up(tags) | weights/kv_cache deferrable、其余总恢复；kv_cache 重建 kv_transfer group | ✓ |
| §2 L34 新坑注 | 触发（二手摘要 / 研究笔记语义纠正 4 初版「level 1 discard all」）/ 现象（每轮多付一次同步）/ 事实（三文件互证、与早期公开文档 offload-to-CPU 一致）/ 规避（以 HEAD 代码为准）——R2-11 四件齐全，纠正留痕 | ✓ |
| §2 L36 备份策略权衡 | 两 level 差别只在权重备份策略；level 1 付全量 host 备份 + H2D 回拷；level 2 免被改写集备份；「level 2 是自然选择（机制层读法）」声明 | ✓ 与 llm.py docstring（reduces CPU memory pressure）一致 |
| §2 L38 显存账 | 14+40=54 → 26 → 80/26≈3.08× 数值不变；补 host 侧 14 GB 备份 + H2D 回拷代价 | ✓ 重算吻合；与 llm.py「enough CPU memory」注对应 |
| §3 对照表 L67 显存释放行 | 「level 定权重备份策略（1 权重 offload 到 host / 2 丢弃被改写集、保留非改写）」 | ✓（原「1 丢全部」已清） |
| 总结 L194 | level 1 offload 自动恢复 / 仅 KV 丢弃；level 2 丢弃被改写集、非改写 offload；三文件互证锚 | ✓ |
| 练习 1(b) L207 | 校验锚「权重按 level 或 offload 到 host 或丢弃被改写集，KV 两 level 均必失效」 | ✓ |
| 参考 L214 | 三文件 level 锚全量 + 「研究笔记语义纠正 4 初版口径被本轮推翻」留痕 | ✓ 行号逐一核实 |
| recap 源 5.4 §6 一致性 | 5.4 L230 命题 4 边界「weights 池若按 level 处置为保留到 host，则 wake_up 可恢复原权重」+「权重按 level 可恢复」（条件式接口边界，逐 level 细节归 9.3）——修复后 9.3 与之一致，5.4 无需改动 | ✓ |
| veRL 注（§2 L40 + §7 表） | vllm_rollout layered_summon → level 1（L125-129 overflow 警告）/ 否则 VLLM_SLEEP_LEVEL；LoRA-level-1-只释-KV 归 sglang_rollout L206 | ✓ fresh veRL 代码逐字（§2 S3-2） |

**S1-1 判定：ADDRESSED。**

## 2. S3 判定（6/6，fresh 证据）

| # | 位置 | 修复核对 | fresh 证据 | 判定 |
| --- | --- | --- | --- | --- |
| S3-1 | 9.3 §2 L31 术语 | 误标「(discardable parameters)」已删；现文明确 discardable = 被改写集（丢弃）、preserved = 非改写集（保留，sleep_preserve_parameter_names 声明） | gpu_worker L339 docstring + L344-349 preserved + L359 集合补 | **ADDRESSED** |
| S3-2 | 9.3 §2 L40 + §7 表 L173 | LoRA-level-1-只释-KV 归 sglang_rollout（L206 判据 + §7 表）；vllm_rollout 改述 layered_summon 判据（L125-129 + overflow 警告 + VLLM_SLEEP_LEVEL 常量），两 backend 勿混声明 | veRL @7b6c9cc0e19d：sglang_rollout L206 `sleep_level = 1 if lora_served_as_adapter(...) else 2` + L202-205 注释 + release L288 docstring L291-293 + tags 分支 L299-302；vllm_rollout L125-129 `if config.layered_summon: warning("…memory overflow."); sleep_level=1 else VLLM_SLEEP_LEVEL`（常量 import 自 verl.third_party.vllm L39） | **ADDRESSED** |
| S3-3 | 9.3 §3 表 L57 + 对照表 L70 + 参考 L215 | record_weight_version → **record_weight_version_after_update** L179，三处同修；另补全 destroy_weights_update_group L215 | sgl weight_updater.py @ac0243723b0d：L179 函数名逐字、L215 destroy 全名；表内其余锚（L131/L164/L182/L210/L232/L287）逐一吻合 | **ADDRESSED** |
| S3-4 | 9.2 §2 L31 + L55 + 参考 L133 | 路径写全 **docs/source/developer-guide/overview.md**；参考加同名易混注（docs/source/overview.md 产品页 / torch/arch_overview.md）；overlap 表述改「GPU 第 n+1 步与 CPU 第 n 步重叠」 | TRT-LLM @a41224a3dee2 developer-guide/overview.md：L17 PyExecutor(Worker) 每 rank / L21 后台循环 / L23-31 四组件逐字 / L61「up to a 22% end-to-end throughput increase」/ L63-67 overlap prose「launch the GPU's work for the next step (n+1)…without waiting for the CPU to finish processing the results of the current step (n)」/ L82「enabled by default」。章从 prose 口径（n），与 dispatch 预期一致；doc 自身代码注释 L77 用 n-1 标号（索引原点不同，同一管线），见 §5 advisory | **ADDRESSED** |
| S3-5 | 9.4 §4 L131 引句 | 补全「…, forcing the output to conform to the target structure.」全句逐字 | DeepInfer doc @a679bb86 L348 全句逐字比对吻合（L344/L346 两段亦吻合） | **ADDRESSED** |
| S3-6 | 9.4 §1 L28 + 参考 L238 | 「_encode_images（L47，docstring L57）」双锚；参考同步 | vl_aligner.py @a679bb86：L47 `def _encode_images(`、L57 docstring「ViT -> downsample -> aligner，输出与 IMAGE token 逐一对应的 embeddings。」；L10 类 / L45 special_embeds 亦吻合 | **ADDRESSED** |

## 3. Writer concern 裁定（fix 轮三项）

| concern | 裁定 | 证据 / 理由 |
| --- | --- | --- |
| level 2 draft.* 警告（gpu_worker L350-358）未展开 | **无需展开，章描述已充分**（可选一句注，非阻断） | 代码事实：level 2 下 draft.* 若不在 preserve 名单则被丢弃，warning 提示 reload_weights 只恢复 target、建议 `--sleep-preserve-parameter-names 'draft.*'`。章 level 2 条同时给出 docstring 意图与**集合补实现**（「全参数减去 preserved 集」）——实现语义已覆盖「未声明保留的 draft 权重会被丢弃」情形，无错误陈述；draft 权重属投机解码栈部署细节，在本章 RL rollout（actor 全参更新）范围声明之外。closeout 若愿补一句「level 2 下未列入 preserve 的 draft 权重同样被丢弃」为可选增益 |
| 「level 2 是 RL colocate 自然选择」是否 veRL 默认断言 | **接受（已声明机制层读法）** | §2 L36 显式「（机制层读法）」；章未断言 veRL 默认值。fresh veRL 代码：非 layered_summon 时 sleep_level = VLLM_SLEEP_LEVEL 常量（verl.third_party.vllm，本轮未取值，章亦未引值）；layered_summon 判据表述与代码逐字一致 |
| 研究笔记语义纠正 4 与 dispatch 预期同误，需 closeout errata | **确认，closeout 必须登记** | 本轮 grep docs/research/inference-w6-sources.md：L33 语义纠正 4 仍为「level 1 丢弃全部（weights + KV）」、L115 仍为「level 1 丢全部」，且 L33 把 LoRA 判据误挂 vllm_rollout（S3-2 同源）。writer 按纪律未动笔记本体；章内坑注 L34 + 参考 L214 已留推翻痕迹。errata 前研究笔记不可作 level 语义事实源 |

## 4. 机械 / 规范检查（3 章）

| 项 | 命令 / 方法 | 结果 |
| --- | --- | --- |
| lint | bash scripts/lint-chapter.sh（仓库根，含 lint-math-boundary.py） | **3/3 LINT OK** |
| display 反转义 raw grep | /usr/bin/grep -F 全文件 ×8 形（\, \! \; \: \{ \} \% \#） | **3 章全 0**（含 display 块） |
| CJK 篇幅 | python3 -I re [一-鿿]（全文件口径） | 9.2 = **2767** ≥2500 ✓ / 9.3 = **4904** ≥3500 ✓ / 9.4 = **5339** ≥3500 ✓（writer 报 2738/4774 为正文口径，差值在 metadata/参考节，两口径均达标） |
| 参考节竖线 | awk 参考节 × grep -F '\|' 与裸 '\|' | 3 章均 0 ✓ |
| training-only arXiv | grep 2405.11143 / 2501.12948 / 2503.14476 / 2402.03300 | 0 ✓（OpenRLHF 纯文字指引维持） |
| arXiv 白名单 | 章内 ID（2309.06180 / 2312.07104 / 2508.15601 / 2409.19256 / 2411.15100）× inference/references.md | 全在 ✓（lint 第 4 项同过） |
| 错误口径残留 | grep 丢全部/丢弃全部/须重新注入/discardable parameters/discard all | 正文 **0 残留**；仅 9.3 坑注 L34 与参考 L214 引误口径作被纠正对象（预期留痕）；L30「无需重新注入」为修复后正确表述。9.1 无 level 语义 echoes（grep 0） |

## 5. 新破坏与 advisory

**新破坏：无。** fix 轮全部改动（9.3 level 语义重写 + 坑注 + host 代价 + veRL 映射 + 函数全名；9.2 路径 + overlap n；9.4 引句补全 + 双锚）逐项对 fresh 一手来源核实，无新事实错误；数值账未动（3.08× / 563.2 GPU·s / 284 ms / 144× / 164 s / 83.9 s / 35 s 与 P9 重推导一致，本轮抽核吻合）；章内 level 术语（offload / preserved / discardable / 被改写 / 非改写）跨 §2 / §3 表 / 总结 / 练习 / 参考五处一致。

Advisory（非阻断，归 closeout）：

1. **研究笔记 errata**（必须）：docs/research/inference-w6-sources.md L33 语义纠正 4 与 L115 的 level 1 误口径 + L33 LoRA 判据误挂 vllm_rollout，按 writer-B fix 报告登记入 errata。
2. **references.md tensorrt-llm W6 注**：仍写「GPU n+1 与 CPU n-1 重叠」与「overview.md」未写全路径——章已按 doc prose（L67，CPU 处理 step n 结果）与 developer-guide 全路径修复；doc 自身 prose（n）与代码注释（n-1）索引原点不同，closeout 回写时建议与章同口径（或注两标号）。references.md 本轮纪律不动。
3. **9.2 被引回写**：vllm-repo / sglang-repo / deepinfer-repo / deepinfer-doc 被引字段 +9.2（writer-B concern 5 维持）。
4. 可选：9.3 §2 level 2 条补一句 draft.* 未保留即丢弃（§3 裁定，非必须）。

## 6. Verification log

### 6.1 本轮 fresh fetch（全部自行取，不信任 writer / evaluator 既有证据）

| 来源 | 通道 / ref | 文件 | 用途 |
| --- | --- | --- | --- |
| vllm-project/vllm | codeload tarball @6aeedab553dd（首轮 480s 超时收 24.1MB；range 续传 exit 33 不支持；重启全量成功：53,630,685 B，md5 d19ff4d2581bc7dfbc073530afc2c075） | 全树；抽出四文件与 API fetch cmp 字节级相同 | S1 主通道 + 路径核实（§1.1 交叉核段） |
| vllm-project/vllm | contents API raw @6aeedab553dd | vllm/v1/worker/gpu_worker.py（1961 行）/ vllm/device_allocator/cumem.py（485 行）/ vllm/device_allocator/sleep_mode_backend.py（209 行）/ vllm/entrypoints/llm.py | S1 四文件互证（§1.1 全行号） |
| vllm-project/vllm | contents API raw @1d9f8058bf8b（当前 main） | gpu_worker.py | 与 @6aeedab553dd diff = 0 行 → 语义今日 HEAD 稳定 |
| vllm-project/vllm | contents API listing @6aeedab553dd | vllm/device_allocator/（cumem.py / sleep_mode_backend.py / xpumem.py / alloc_conf.py）；vllm/v1/worker/gpu/（无 cumem.py）；vllm/distributed/weight_transfer/（六 engine + base/clients/factory/packed_tensor） | 路径裁定 + 六引擎核实（sparse_nccl_engine.py 在树） |
| verl-project/verl | contents API raw @7b6c9cc0e19d | verl/workers/rollout/vllm_rollout/vllm_rollout.py（341 行）/ sglang_rollout/sglang_rollout.py（480 行） | S3-2 + veRL 注 + §7 表（ServerAdapter L49/L128、resume L195/L276、release L204/L288、update_weights L210/L305、layered_summon L125-129、LoRA L206、slime L321） |
| sgl-project/sglang | contents API raw @ac0243723b0d | python/sglang/srt/managers/scheduler_components/weight_updater.py（686 行） | S3-3 + §3 表全锚 |
| NVIDIA/TensorRT-LLM | contents API raw @a41224a3dee2 | docs/source/developer-guide/overview.md（209 行） | S3-4（L17/L21/L23-31/L61/L63-67/L82） |
| deepseek-ai/DeepInfer | contents API raw @a679bb86 | doc/inference_deployment_qa.md（372 行）/ deepinfer/model_executor/models/vision_encoder/vl_aligner.py（141 行） | S3-5（L344/346/348）+ S3-6（L10/L45/L47/L57） |

注：raw.githubusercontent.com 本机 SSL reset，contents API（api.github.com）通道正常——与 P9 同通道。veRL / DeepInfer 路径须带包前缀（verl/…、deepinfer/…），首轮不带前缀 404 后修正。

### 6.2 机械命令

- lint：`bash scripts/lint-chapter.sh inference/09-production/{02-other-engines,03-rl-rollout-serving,04-multimodal-and-special-serving}.md` → 3/3 LINT OK。
- 反转义：/usr/bin/grep -nF 各禁用形 ×3 章全文件 → 0 命中。
- CJK：python3 -I `len(re.findall(r"[一-鿿]", text))` → 2767 / 4904 / 5339。
- 残留 / 白名单 / 竖线：/usr/bin/grep -nE / -F（§4 表）。
- 5.4 recap 对照：inference/05-vllm/04-vllm-features.md L212-245（§6 命题 4 / 边界 / 3.35× 账）逐句读核。
- 研究笔记残留确认：docs/research/inference-w6-sources.md L33 / L115 grep。
- tarball 交叉核：bash /tmp/iw6-fixreview-scripts/tarball-crosscheck.sh（tar 抽出四文件 → cmp × API fetch 全 BYTE-IDENTICAL；关键行 grep 再核；weight_transfer 六 engine listing；v1/worker/gpu/ 无 cumem.py 确认）。

### 6.3 覆盖率

- S1：codeload tarball + contents API 双通道字节级互证四文件 + 当前 main 对照，level 1/2 语义全链（discardable 构造 → suspend offload 判据 → CPU 备份 → wake_up 恢复 / 空显存）逐行核实；章节 12 处 echoes 逐项核对。
- S3：6/6 fresh 来源逐字 / 逐行核实。
- concern：3/3 裁定（§3）。
- 机械：lint / 反转义 / 篇幅 / 竖线 / 白名单 / 残留全覆盖。
- 未重复项（P9 已全量核实且 fix 轮未动）：9.1 全账重推导、9.4 patch 账复算、OpenRLHF tarball 行号、veRL async / checkpoint_engine 目录、SGLang engine.py 漂移行号——本轮抽核未动部分与 P9 记录一致。