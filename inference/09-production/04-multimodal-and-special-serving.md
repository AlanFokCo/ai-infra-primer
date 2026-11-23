# 9.4 多模态与特殊 Serving（Multimodal and Special Serving）

> 一张 896×896 的图在 VE 里是 4096 个 patch，LLM 却只见到 484 个图像 token——这 8.5× 的缩减从哪来、VE 为它付了多少算力？结构化输出的 grammar 编译为什么能搬去专用机、embedding serving 为什么没有 decode 循环？本章交付多模态 serving 的全账：image patch → VE 算力 → encoder cache 复用，SO 专用机异步编译的生产形态，以及 embedding / reranker 的 prefill-only 批形态 | 前置：1.1（请求生命周期与 TTFT 分解）、1.2（prefill FLOPs 式 (1) 与 $\kappa$ 几何）、1.4 §5（grammar → token mask 原理与命题 5）、2.2（命中率 → 节省账）、4.3（chunked prefill）、5.2 §8（EncoderCacheManager 调度接口）、5.4 §3（bitmask 异步填充）、6.1 §6（SGLang constrained backend）、7.4 §7（VE LB 口径）、8.4 §6（VE/P/D 架构与 release 边界）；models 4.8 §1（diffusion LLM）只 recap 不重推 | 锚点：DeepInfer doc + repo（[DeepSeek-AI, inference_deployment_qa.md](../references.md)、[DeepSeek-AI, DeepInfer](../references.md)，教学参照 + 官方披露，main@a679bb86）、V4.1-Flash vision_config（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]）、vLLM（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]；vllm-repo `encoder_cache_manager.py` / `pooler/`）、SGLang（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]；sglang-repo `mm_schedule.py` / `serving_embedding.py` / `dllm/`）、XGrammar（[Dong et al., 2024](../references.md)，arXiv:2411.15100，[A]）、vLLM V1 blog（[vLLM Team, 2025](../references.md)，[C] 开放技术博客、非闭源）

## 1. 多模态请求生命周期：图像从哪一段进入链路

一条带图请求的 TTFT 比纯文本请求多出哪几段？1.1 §1 的链路（tokenize → 排队 → prefill → decode）在图像输入下插入三段：占位符展开与 CPU 预处理、VE（vision encoder，视觉编码器）前向、embedding 合并。先把扩展链路画全（口径：DeepInfer 引擎侧 release 代码 + doc 披露的 VE 分离架构；VE-P-D 编排层不在 release，recap 8.4 §6，本章走读均为引擎侧）：

```
客户端请求（文本 prompt + 图像字节 + 声明尺寸 (w, h)）
  │
  ▼
API 协议层 → tokenize（文本 → token ids；每图一个占位 token）
  │
  ▼
占位符展开 overwrite_prompt（1 占位 → n_expand 个图像段 token）
  │  ← CPU 预处理 worker：decode / resize（min_pixels / max_image_tokens 管线，§2）
  ▼
VE LB（machine 随机 + fewest patches，7.4 §7 recap）→ VE（单 GPU、cheaper，doc L51）
  │  ViT 前向（n_vit patches）→ 3×3 downsample → MLP projector → [n_llm, d_model]
  ▼
embedding 合并（IMAGE 位置替换为图像 embedding）→ LLM prefill（可 chunk，embedding 按 chunk 切片）
  │
  ▼
decode 循环 → detokenize → 流式返回（与 1.1 完全同形，无多模态特例）
```

DeepInfer 的预处理是独立 CPU 进程池：`ve/multimodal_processor.py` MultimodalProcessor（L29）fork 预处理 worker 子进程（`_start_workers` L93），pinned host 共享内存池按 rank 常驻（L49-56）；`overwrite_prompt`（L105）把每个图像占位 token 按分辨率展开为 `mm_image_token + N×mm_image_pad_token`，N 由 §2 的 layout 管线算出。调度器不必解码图像就能知道每图占多少 token：`estimate_num_tokens`（image_processor.py L308）只跑尺寸规划、不碰图像本体——这是 VE LB 与 prefill 预算能提前算账的前提。ViT 之后的路径在 `vision_encoder/vl_aligner.py`：VisionLanguageInstructTuner（L10）的 `_encode_images`（L47，docstring L57）逐字「ViT -> downsample -> aligner，输出与 IMAGE token 逐一对应的 embeddings」；downsample（空间降采样）是 3×3 邻域 channel-concat（`mlp_projector.py` L16-17：首层入维 = input_dim × downsample_ratio² = 1024 × 9 = 9216），再经 MLP projector 投到 LLM hidden 宽；NEW_LINE / START / END 等布局 token 不走 ViT，用学习到的 special_embeds（vl_aligner.py L45）。vLLM 侧同构三件（预处理独立进程 + image hash 进 prefix caching + encoder cache）已在 5.4 §8 登记（[vLLM Team, 2025](../references.md)，[C]），SGLang 侧的 embedding 调度归 §3 走读。

串行口径下，多模态请求的 TTFT 分解为：

$$
\mathrm{TTFT} = t_{\mathrm{queue}} + t_{\mathrm{pre}} + t_{\mathrm{VE}} + t_{\mathrm{prefill}} \qquad (1)
$$

**符号**： $t_{\mathrm{queue}}$ ：排队时间（1.3）； $t_{\mathrm{pre}}$ ：CPU 预处理（decode / resize / 占位展开）； $t_{\mathrm{VE}}$ ：VE 前向（§2 式 (5) 折算）； $t_{\mathrm{prefill}}$ ：LLM prefill（含图像 token）。这是串行上界口径——VE 与排队 / tokenize 可重叠（异步预取），重叠后 $t_{\mathrm{VE}}$ 部分或全部藏进前段，与 6.2 的 bubble 消除账同构。全数字实例（教学假设，非实测）：纯文本基线 TTFT = 0.5 s（1.1 章脊口径）， $t_{\mathrm{pre}}$ = 5 ms、 $t_{\mathrm{VE}}$ = 55.7 ms（§2 按 896×896 图与假设有效算力折算），串行相加 TTFT ≈ 0.56 s——图像段给 TTFT 加约 12%。若 VE 预取与排队完全重叠，增量只剩 $t_{\mathrm{pre}}$ 与合并开销。

本章口径锁定：vision 侧一律取 V4.1-Flash vision_config [W]（§2 字段表），主例图 896×896；LLM 侧只引 $d_{\mathrm{model}}$ = 5120 / 40 层 [W] 做容量换算，不做模型侧推导（V4.1 形式化归 models 维度，待补）；延迟、命中率与有效算力数值均为教学假设、随文声明，非生产实测；硬件逐卡参数表归 1.3。 $B$ = batch 内请求数； $T$ = 序列长（token 数）；bytes 一律文字； $\eta$ = encoder cache 命中率（按图像数加权，与 2.2 的 token 加权前缀命中率 $h$ 不同口径不同符号，勿混）。

## 2. image patch 账：从像素到 LLM token

896×896 的图，ViT 见 4096 个 patch（图像分块）、LLM 见 484 个图像 token——两个数各服从什么取整规则？先钉 config 字段（[DeepSeek-AI, V4.1-Flash config.json](../references.md)，[W]，2026-11-24 hf-mirror 第五次重取 byte-identical）：

| vision_config 字段 | 值 | serving 含义 |
| --- | --- | --- |
| patch_size | 14 | ViT patch 边长（像素） |
| downsample_ratio | 3 | ViT → LLM 的空间缩减比（每维） |
| num_hidden_layers / hidden_size | 32 / 1024 | VE 层数 / hidden 宽（算力账输入） |
| num_attention_heads / intermediate_size | 16 / 2816 | VE attention 头数 / SwiGLU 中间维 |
| max_image_tokens | 1024 | 单图展开后 token 数上限 |
| min_pixels | 295936 | 像素数下限（= 544²，小图放大） |

模型侧形式化归 models 维度（V4.1 待补），本节只引字段做 serving 账。尺寸规划管线在 `ve/processor/image_processor.py` compute_image_layout（L155）：先按 min_pixels 等比放大（L170-173），再把宽高向上取整到 patch_size 倍数（L178-179），最后 safe_resize（L131）检查展开 token 数是否超 max_image_tokens、超则 solve_resize_ratio（L95）等比缩小。设规划后的 patch 对齐尺寸为 $w', h'$，则 ViT patch 数：

$$
n_{\mathrm{vit}} = \frac{w'}{14} \cdot \frac{h'}{14} \qquad (2)
$$

**符号**： $w', h'$ ：patch 对齐后的宽 / 高（best_width / best_height，14 的整数倍）；14 = patch_size [W]。主例代入：896 = 64×14， $n_{\mathrm{vit}}$ = 64×64 = 4096。ViT 输出经 3×3 downsample 后，LLM 侧图像 token 数是**两维分别向上取整再相乘**（downsampled_seqlens，image_processor.py L81，代码形 `((n_vit_w+r-1)//r)*((n_vit_h+r-1)//r)`，r = downsample_ratio）：

$$
n_{\mathrm{llm}} = \left\lceil \frac{n_{\mathrm{vit},w}}{3} \right\rceil \cdot \left\lceil \frac{n_{\mathrm{vit},h}}{3} \right\rceil \qquad (3)
$$

**符号**： $n_{\mathrm{vit},w} = w'/14$ 、 $n_{\mathrm{vit},h} = h'/14$ ：ViT patch 网格宽 / 高；3 = downsample_ratio [W]。主例代入：ceil(64/3) = 22， $n_{\mathrm{llm}}$ = 22×22 = 484。LLM 序列里图像段还要加布局 token——排布唯一（文件头 docstring 逐字）：`[IMAGE_START] + ([IMAGE]*n_llm_w + [IMAGE_NEW_LINE])*n_llm_h + [IMAGE_END]`，展开总数（expanded_image_token_len，L76）：

$$
n_{\mathrm{expand}} = n_{\mathrm{llm},h} \cdot (n_{\mathrm{llm},w} + 1) + 2 \qquad (4)
$$

**符号**： $n_{\mathrm{llm},w}, n_{\mathrm{llm},h}$ ：式 (3) 的两个向上取整因子；+1 是每行 NEW_LINE，+2 是首尾边界 token。max_image_tokens = 1024 约束的是 $n_{\mathrm{expand}}$（safe_resize L142 比较的是 num_tokens），不是 $n_{\mathrm{llm}}$。

主例代入（896×896，python 复核）：896 = 64×14， $n_{\mathrm{vit}}$ = 64×64 = 4096；ceil(64/3) = 22， $n_{\mathrm{llm}}$ = 22×22 = 484； $n_{\mathrm{expand}}$ = 22×23+2 = 508 ≤ 1024，不触发缩小。三个数各就各位：ViT 算 4096 个 patch 的 attention，LLM prefill 只处理 508 个图像段 token（其中 484 个来自 ViT、24 个是布局 special_embeds）。

> **坑**：把式 (3) 算成「总 patch 数除以 9 再取整」。触发条件：用 ceil(n_vit/9) 估图像 token 数；现象：ceil(4096/9) = 456，比真值 484 少 28——占位符展开数与 embedding 行数错位，图像 embedding 无法对位填入（overwrite_prompt 按 layout 展开，两者必须同源）。规避：两维分别 ceil 再相乘（代码 L81 逐字），64/3 = 21.33 → 22 的向上取整在每维各付一次。

两个边界实例把管线两端钉死。小图端：448×448 = 200704 像素 < min_pixels 295936，等比放大到 544×544、patch 对齐到 546×546（39×14），实际 $n_{\mathrm{vit}}$ = 39² = 1521、 $n_{\mathrm{llm}}$ = 13² = 169、 $n_{\mathrm{expand}}$ = 184。注意 7.4 §7 的 VE LB 例算「448×448 → 1024 patches」用的是 doc L151 的尺寸直算式（LB 预估口径），与这里 resize 管线后的实际 patch 数（1521）是两层口径——LB 在路由前估负载，不跑 resize 管线（§3 命题 1 再谈这层近似）。大图端：1920×1080 patch 对齐后 1932×1092 → 138×78 = 10764 patches → 46×26 = 1196 图像 token → 展开 1224 > 1024，触发 solve_resize_ratio 等比缩小到 1708×966 → 122×69 = 8418 patches → 41×23 = 943 图像 token → 展开 968 ≤ 1024（可按 solve_resize_ratio L95 逐步复现）。

缩减的 serving 含义用 1.2 的账读：LLM prefill 参数项 $\propto$ token 数（1.2 式 (1) recap），每图 KV 占用 = $n_{\mathrm{llm}} \kappa L$（1.2 §3 recap， $\kappa$ 为每 token 每层 KV 字节）——downsample 把 LLM 侧的 prefill 算力、KV 容量与 decode attention 读取全部按 $n_{\mathrm{vit}}/n_{\mathrm{llm}}$ ≈ 8.46×（4096/484；含布局 token 为 4096/508 ≈ 8.06×）缩减，代价是 VE 侧仍要算全部 4096 个 patch。VE 算力账由此登场。路线：两笔——参数项每 patch 2N_VE（与 1.2 式 (1) 同形），attention 项双向全量 4n²d_VE（1.2 式 (1) 的系数 2 来自 causal mask 平均可见长减半；ViT 不打 causal mask，系数回到 4，代码锚 vit.py Attention forward `causal=False` L116 + npu_ops/vit_attn.py L16 docstring 逐字「非 causal，每段内做全量注意力」）：

$$
C_{\mathrm{ViT}}(n) = 2 N_{\mathrm{VE}} \thinspace n + 4 L_{\mathrm{VE}} \thinspace n^{2} \thinspace d_{\mathrm{VE}} \qquad (5)
$$

**符号**： $n$ ：单图 patch 数（ $n_{\mathrm{vit}}$）； $N_{\mathrm{VE}}$ ：VE matmul 口径参数量； $L_{\mathrm{VE}}$ = 32、 $d_{\mathrm{VE}}$ = 1024（VE hidden 宽， $d_{\mathrm{model}}$ 语义的 VE 侧局部记号）[W]。口径：只计 ViT 前向；downsample / projector 是 per-token MLP（无 attention 二次项，首层入维 9216 已锚），CPU 预处理另计。全数字代入（n = 4096， $N_{\mathrm{VE}}$ = 411,041,792 ≈ 0.411B，计数见下）：参数项 2 × 4.11×10⁸ × 4096 = 3.37×10¹² FLOPs，attention 项 4 × 32 × 4096² × 1024 = 2.20×10¹² FLOPs，合计 5.57 TFLOPs/图，attention 占 39.5%。

$N_{\mathrm{VE}}$ 按代码结构数：Attention 的 qkv + proj 共 $4d_{\mathrm{VE}}^{2}$（vit.py L96-97），SwiGLU 三矩阵共 $3 \times 1024 \times 2816$（L130-131；中间维 = 128·ceil(1024×2.7/128) = 2816，与 config intermediate_size 互证），每层 12,845,056，32 层合 411,041,792（matmul 口径，略 norm 与 patch_embed；结构常量 L27-32 与 config [W] 一致）。这是 serving 算力账的参数计数，不是模型侧架构形式化。

二次项何时反超参数项？令两笔相等解出 $n^{\ast} = N_{\mathrm{VE}}/(2 L_{\mathrm{VE}} d_{\mathrm{VE}})$ = 6272 patches——超过它 attention 主导（1920×1080 缩小前 10764 patches 时 attention 占 63.2%）。VE 单 GPU 部署的生产口径（doc L51 逐字，官方披露，2026-11-24 再核）：「VE: a single GPU per instance is generally enough, and cheaper GPUs are recommended.」账面依据：0.411B 小模型 + 每图 5.57 TFLOPs，假设有效算力 100 TFLOPS（教学假设，非特定 SKU 披露；「cheaper」是 doc 推荐口径，无具体型号），折算 55.7 ms/图、单 GPU 约 18 图/s 的算力上界（口径：忽略访存与预处理）——VE 池机数随图像到达率线性扩，与 prefill EP2–8 / decode EP64–128 的重型并行形成对照（7.4 §1 recap）。

> **提点**：downsample 的设计本质是「在小模型域付算力、在大模型域省 token」——VE 一次性付 4096 patch 的账，换来 LLM 的 prefill、KV 容量、decode attention 三本账全部按 ~8.5× 缩减。VE 越便宜（单 GPU、cheaper 卡），这笔交换越划算。

## 3. VE LB 与 encoder cache：两笔节省账

fewest-patches 用 patch 数当负载代理——这个代理与真实 FLOPs 负载差多少？doc L151（官方披露，7.4 §7 已登记）：图像按 machine 粒度路由，随机选一台「每 GPU 一个 vision encoder」的机器，机内交给当前处理 patch 最少的 GPU，patch 数由尺寸直算 `(width/patch_size) × (height/patch_size)`。先钉成本模型：ViT attention 是逐图 varlen 全量（vit_attn.py L16「每段内做全量注意力」），一个 GPU 批内多图的负载是**逐图成本之和** $\sum_i C_{\mathrm{ViT}}(n_i)$，不是 $C_{\mathrm{ViT}}(\sum_i n_i)$——二次项不跨图合并。

**命题 1**（单调代理非等价）。设逐图成本 $C(n) = an + bn^{2}$（ $a = 2N_{\mathrm{VE}}$ 、 $b = 4L_{\mathrm{VE}}d_{\mathrm{VE}}$ ，均正），则总 patch 数 $\sum_i n_i$ 是总 FLOPs 负载的单调代理（对每个 $n_i$ 单调：任一图 patch 数增加，两个量同增）但非等价代理：总 patch 数相同的两种分配，总 FLOPs 可以不同；线性项完全等价，差异全部来自二次项，且 $\sum_i n_i^{2} \le (\sum_i n_i)^{2}$。

**证明**（路线：拆两项，线性项求和交换，二次项展开交叉项）。线性项 $\sum_i a n_i = a \sum_i n_i$，只依赖总 patch 数——这部分代理精确。二次项 $b \sum_i n_i^{2}$：展开 $(\sum_i n_i)^{2} = \sum_i n_i^{2} + 2\sum_{i \lt j} n_i n_j \ge \sum_i n_i^{2}$，交叉项非负，等号成立当且仅当至多一个 $n_i$ 非零。故总 patch 数相同的两种分配，图越碎（多图小图）二次项越小、总 FLOPs 越低；单图独占时代理退化为单调函数、精确成立。∎

边界反例（全数字，python 复核）：GPU A 处理 1 张 4096-patch 图，GPU B 处理 4 张 1024-patch 图，总 patch 数同为 4096。线性项相同（a×4096）；attention 项 A = 2.20×10¹² FLOPs、B = 4 × 1.37×10¹¹ = 5.50×10¹¹ FLOPs，**A 是 B 的 4×**；总 FLOPs A = 5.57 TFLOPs、B = 3.92 TFLOPs，比 1.42×。fewest-patches 把两者视为等负载——它系统性低估大图集中的 GPU。doc 仍取 patch 代理的理由是轻量：尺寸直算、无需先跑 encoder、无需知道图数分布（路由前预估）；再加上 §2 的第二层近似（LB 尺寸直算 vs resize 管线实际 patch 数，448×448 例 1024 vs 1521），fewest-patches 是两层近似下的粗均衡，不是 FLOPs 精确均衡。命题 1 给出了这笔近似的方向与量级。

第二笔节省账不改负载代理，而是让部分图根本不进 VE。所谓 encoder cache，是指把 VE 输出的图像 embedding 按输入特征 hash 跨请求保留，同图再现时跳过 ViT 前向。三引擎各有落点：

**vLLM EncoderCacheManager**（`v1/core/encoder_cache_manager.py`，5.2 §8 已走读调度接口，此处补复用账；行号与 W3 同，2026-11-24 contents 再核）。容量以 **embedding 行数**计（docstring 口径：布局 / 文本 token 不计入 cache_size），mm_hash 跨请求共享（L36-40）。生命周期四步：`check_and_update_cache`（L111）查命中并登记请求引用；`can_allocate`（L148）双门——先查 encoder compute budget、再查容量，容量不足时从 freeable（无请求引用的旧条目，OrderedDict）**分配时驱逐**、最旧优先（L203-208）；`allocate`（L211）扣 free slots；`free_encoder_input`（L259）解引用，引用集空才进 freeable、并不立即物理释放。一个容易漏的语义（L286-293 注释口径）：同一请求内同图多次出现（跨对话轮携带）只算**一个引用**，最后一次出现释放前条目不可驱逐——否则请求还需要它时就被逐出、encoder 重算已有之物。权重更新时 `reset`（L96）全量失效（docstring：invalidate stale embeddings）——VE 输出依赖 VE 权重，9.3 的权重更新接口由此连到多模态 cache。预算侧 `compute_mm_encoder_budget`（L329）返回 (compute budget, cache size) 双预算，各取 max(配置值, 单 item 最大 token 数) 保证最大图装得下；`disable_chunked_mm_input` 且单 item 超 `max_num_batched_tokens` 直接 ValueError（L358-367）。EncoderDecoderCacheManager（L392）的 check 恒 False——enc-dec 模型不做跨请求共享，只用作调度记账。

**SGLang mm_schedule.py**（`managers/`，W3 核实函数级语义、本轮行号全同）。`init_mm_embedding_cache`（L28）建 MultiModalStaticCache；核心是 **embedding 算一次、chunk 只切片**：`get_embedding_chunk`（L33）把当前 chunk 的序列区间 [extend_prefix_len, extend_prefix_len + extend_seq_len) 经 items_offset 映射成 embedding 行区间，从整图 embedding 里切出本 chunk 需要的行——ViT 不因 chunked prefill 重跑。`_get_chunked_prefill_embedding`（L593）三相位：Phase 0 把请求分成 per-image 路径与 full/EVS 路径（video 等 multi-span item 走后者，L628-634）；Phase 1 `_batch_encode_per_image_misses`（L389）收集**所有** per-image 请求的 cache miss，按 (cache_key, expected_token_count) 双键去重后**一次 ViT call** 批编码、回填 cache（L449-481）——双键含 token count 是防同 hash 不同占位跨度被错误去重（docstring 逐字：「prevents two colliding hashes with different placeholder spans from being deduplicated within the same batch」）；cache 命中但 token count 不符的条目丢弃重算（`_discard_mismatched_cached_embedding` L221）。Phase 2 按原请求序组装各 chunk（L673-681）。预计算接口：item 可携带 precomputed_embeddings（可能在 CPU，用前经 host_offload_event 同步，L90-95）；PrefetchedEmbeddingFunc（L165）让 per-image 路径以持有 embedding 替代 ViT 前向——这是「VE 计算在引擎外 / 提前完成」的接口形态（架构观察：接口能力，非 SGLang 已附带分离 VE 部署的声明）。

**DeepInfer** 的命中路径在 `ve/prepare_tensor_input_vl.py`：零占位行（缓存图 / 空图）不产 raw、不进 ViT（L99 注释逐字），全 batch 命中时 seqlens_vit 为空、下游跳过 vl_aligner（L125）。

命中省多少？设命中率 $\eta$（按图像数加权），每图期望 VE 算力：

$$
E[C] = (1 - \eta) \thinspace C_{\mathrm{ViT}}(n) \qquad (6)
$$

**符号**： $\eta$ ：encoder cache 命中率（命中图数 / 总图数；命中 = 复用已有 embedding、跳过 ViT 前向）； $C_{\mathrm{ViT}}$ 见式 (5)； $E[\cdot]$ ：对命中 / 未命中取期望。命中不是免费：embedding 须驻留显存或被取回，容量账见下。代入主例（n = 4096，C = 5.57 TFLOPs）： $\eta$ = 0.5 时期望 2.78 TFLOPs/图； $\eta$ = 0.8 时期望 1.11 TFLOPs/图，按假设 100 TFLOPS 折算 11.1 ms——相对无 cache 的 55.7 ms 省 80%。cache 容量账（bf16 教学口径，projector 后 embedding 进 LLM hidden 宽）：每图 embedding 字节 = $n_{\mathrm{llm}} \times d_{\mathrm{model}} \times 2$ = 484 × 5120 × 2 = 4,956,160 B ≈ 4.73 MiB，1 GiB 容量约驻 216 张这样的图。vLLM 的行数口径与此换算等价（cache_size 行 × 每行字节）。

encoder cache 还是 MLLM chunked prefill 的前提（[vLLM Team, 2025](../references.md)，[C]，V1 blog 第 7 点）：V0 图像与文本必须同步处理（vision embedding 步后即弃），V1 缓存 embedding 后文本才能分 chunk 跨步。调度耦合 recap 5.2 §8：vLLM `_try_schedule_encoder_inputs`（scheduler.py L2111）在 encoder_compute_budget 内安排 encoder 计算，预算或缓存不足时把请求本步 num_new_tokens 缩到该 encoder 输入之前；SGLang 侧即上面的 chunk 切片——两引擎都把「encoder 预算」与「token 预算」分成两个旋钮（4.3 的 chunk 钳位只管后者）。

## 4. 结构化输出生产形态：SO 专用机与异步编译

grammar 编译是 CPU 工作、decode mask 是每步工作——生产形态为什么把前者搬去专用机？先 recap 原理链（各一句，不重推）：1.4 §5 交付 grammar → 自动机状态 → token mask 的原理与命题 5（约束保持：每步 mask 到合法集且按转移走，输出必被接受）；5.4 §3 走读 vLLM bitmask 异步填充（GPU 跑前向时 CPU 并行填 mask，sample 前就绪）；6.1 §6 走读 SGLang GrammarManager 与三 backend（XGrammar 默认）；7.1 §3 登记 DeepInfer `structure_output/` 引擎侧模块（在 release）。本节增量是 doc 披露的**生产部署形态**与编译时序账。

DeepInfer 官方部署 QA 三阶段（[DeepSeek-AI, inference_deployment_qa.md](../references.md)，教学参照，官方披露，doc L344-348 逐字，2026-11-24 再核）：

> 「Compilation and inference are decoupled: SO compilation runs asynchronously, and a request retrieves the compiled result before it enters the engine.」（L344）
>
> 「Compilation stage (on a dedicated SO machine): the format constraint is compiled into a grammar automaton.」（L346）
>
> 「Runtime: at each decode step, the automaton is advanced, the set of legal tokens is determined, and a mask is generated and applied to the sampling logic, forcing the output to conform to the target structure.」（L348）

即：SO（structured output，结构化输出）编译在**专用机**上异步跑，产物是 grammar automaton（语法自动机，1.4 §5 的 FSM / PDA 统称），请求入引擎前取编译结果；运行时每步只做 advance + mask。时序画成对照：

```
同步形态（编译在引擎关键路径上）
请求 ──► [compile t_compile] ──► [prefill] ──► decode（每步 advance+mask）
              ▲ TTFT 含全部编译时间，且编译占引擎 host CPU

异步形态（专用 SO 机，doc L344-348）
请求 ──► tokenize / 排队 / 预处理（t_front）
   │              SO 机：[compile t_compile]（异步、结果可缓存）
   └──► 取编译结果 ──► [prefill] ──► decode（每步 advance+mask）
         ▲ 命中：只付查找；冷 miss：只等超出前端路径的部分
```

冷 miss 的编译等待可以压到多少？路线：编译与前端路径（tokenize / 排队 / 预处理）并行，请求入引擎时编译已完成则零等待，否则只等剩余部分——关键一步是取 max 而非相加：

$$
t_{\mathrm{wait}} = \max(0, \thinspace t_{\mathrm{compile}} - t_{\mathrm{front}}) \qquad (7)
$$

**符号**： $t_{\mathrm{compile}}$ ：schema 编译时间； $t_{\mathrm{front}}$ ：请求到达至入引擎的前端路径时间（与编译并行）。全数字实例（教学假设，非官方披露）：复杂 JSON schema 编译 $t_{\mathrm{compile}}$ = 200 ms、前端路径 $t_{\mathrm{front}}$ = 50 ms——同步形态 TTFT 加 200 ms；异步冷 miss 只加 150 ms；schema 命中 cache 时只加查找时间（~1 ms 量级，假设）。schema 复用还有摊销账： $N_{\mathrm{req}}$ 个请求共享同一 schema（ $N_{\mathrm{req}}$ = 请求数，局部记号，与参数量 $N$ 区分），编译只付一次，每请求摊 $t_{\mathrm{compile}}/N_{\mathrm{req}}$ = 200 ms / 1000 = 0.2 ms——与 2.2 前缀复用的「一次算、多次用」同构，只是复用对象从 KV 换成编译产物。

**命题 2**（两段成本分离）。SO 的编译成本只进入 TTFT（冷 miss 且只计超出前端路径的部分，式 (7)）；每步运行时成本只有 automaton advance + mask 生成，与 $t_{\mathrm{compile}}$ 无关；当 mask 填充与前向重叠且 $\tau_{\mathrm{mask}} \le \tau_{\mathrm{forward}}$ 时，SO 对 decode 步时的增量为零。

**证明**（要点，两段）。编译段：产物是自动机的静态结构（状态集 + 转移 + 各状态合法 token 集），编译完成后不再参与运行时计算；请求入引擎前取结果（doc L344），故编译时间只可能出现在 TTFT 里，且被前端路径重叠掉 $t_{\mathrm{front}}$ 部分（式 (7)）。运行段：每步只做状态转移 $\delta(s, \text{token})$ 与按当前状态取 mask（1.4 §5 recap），成本只依赖状态与词表、不依赖编译耗时；mask 填充若与 GPU 前向重叠（5.4 §3 的 bitmask 异步填充正是此形态），则 $\tau_{\mathrm{mask}} \le \tau_{\mathrm{forward}}$ 时完全隐藏——与 6.2 bubble 消除账同构。∎ 边界：若 mask 填充串行在前向之后（无重叠），步时增量 $\tau_{\mathrm{mask}}$ 全暴露；命题的零增量结论以重叠为前提。

引擎侧代码落点（release 内，main@a679bb86）：`structure_output/compiler.py` XGrammarCompiler（L14）的 `dispatch_and_compile`（L54）按 schema 类型三分支——json 走 compile_json_schema（L57）、regex 走 compile_regex（L59）、ebnf 走 compile_grammar（L61）；`matcher.py` XGrammarMatcherFuture（L261）把编译 / matcher 构建包成 Future，timeout 默认 20（environ.py L175，秒口径，Future.result 语义）——冷 miss 等待有上界，编译卡死不拖死请求。CPU 隔离：`DEEPINFER_SO_BUILDER_CPUS` 默认 4（environ.py L178），`utils.py` get_idle_cpus（L154）把 SO builder 绑到每 GPU CPU 配额的前 4 个核——编译线程不与调度热路径抢核。

> **坑**：grammar cache 无上限运行。触发条件：cache 开启且 `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` ≤ 0；现象：cache 随不同 schema 数无界增长直至 OOM（compiler.py L23-29 警告逐字点名 DoS vector，CVE-2025-32381）。规避：生产设正字节上限（默认 10 GiB，environ.py L177）；kill switch `DEEPINFER_XGRAMMAR_DISABLE_CACHE` 保留（L17-18 注释逐字：「xgrammar <= 0.1.x had production incidents with cache enabled」）——上游库的 cache 行为变更后，引擎侧要能一键退回。

为什么专用机（结构 rationale，非 doc 逐字）：编译是 CPU 密集突发负载，而引擎 host 的 CPU 已被 overlap scheduler 占满——6.2 的账要求 CPU schedule / launch 跟得上 GPU 步时（ms 级），编译抢核会直接制造调度延迟与 GPU bubble。专用 SO 机把编译整个移出引擎 host；release 内的绑核隔离（get_idle_cpus）是同一思路在单机尺度的形态，doc 的专用机是集群尺度的形态。

> **提点**：「编译一次、每步 advance」是 1.4 §5 三系统对照里「FSM 离线索引 / XGrammar 自适应 mask」思想的生产化——schema 级的一次性成本与 token 级的每步成本分离，前者异步化 + 缓存 + 专用机，后者重叠进前向。两段各自优化，互不拖累。

## 5. embedding / reranker serving：prefill-only 的批形态

一个 embedding 请求没有 decode 循环——它的 batch 长什么样、调度器在优化什么？所谓 prefill-only serving，是指请求只做一次全序列前向 + pooling（池化，把 token 表示聚合为序列表示），无自回归循环、无跨步 KV 增长。SGLang 的端点四件（`entrypoints/openai/serving_{embedding,rerank,classify,score}.py`，2026-11-24 contents 核实）中，serving_embedding.py 的 `_handle_non_streaming_request`（L248）对 `EmbeddingReqInput` 只取一次 `__anext__()`——无 streaming 即单次前向的代码形态。vLLM 侧入口是 AsyncLLM.encode，pooling 层已模块化。

> **坑**：按旧路径找 pooler。触发条件：引用 `model_executor/layers/pooler.py`；现象：该文件已不存在（404），实现拆成 `pooler/` 目录（abstract.py / seqwise/ / tokwise/ / special.py）。规避：按目录锚——Pooler ABC 在 abstract.py L16，序列级三方法在 seqwise/methods.py（2026-11-24 contents 核实）。

三个 pooling 方法各取什么（seqwise/methods.py）：CLSPool（L38）取首 token hidden state，LastPool（L51）取末 token，MeanPool（L61）取全 token 均值：

$$
e = \frac{1}{T} \sum_{t=1}^{T} z_t \qquad (8)
$$

**符号**： $z_t$ ：第 $t$ 个 token 的末层 hidden state（ $d_{\mathrm{model}}$ 维）； $T$ ：序列长； $e$ ：序列 embedding。最小实例：3 token 的 hidden [[1,2],[3,4],[5,6]] → e = [3,4]。实现细节两笔：累加器强制 fp32（L94-113，bf16 直接累加会丢精度）；`VLLM_BATCH_INVARIANT` 模式下改逐序列独立 sum（L82-86，注释逐字：「index_add_ is nondeterministic on GPUs」）——1.4 §6 的 batch invariance 账在 pooling 层的落点：归约顺序随 batch 组成变，逐段求和把顺序钉死。CLS / MEAN 对应双向 encoder 家族（BERT 式）的惯例，LAST 对应 causal decoder embedder（末 token 经 causal attention 已见全序列）；方法选择与模型注意力形态配对，不是自由旋钮。

**命题 3**（pooling 的 hidden state 局部性）。chunked prefill 下，pooler 可消费的 hidden state 仅限当前 chunk 的 token；凡依赖非当前 chunk token hidden state 的 pooling 方法，不能在 partial prefill 上执行。

**证明**（要点）。chunked forward 只产出当前 chunk token 的 hidden state——KV cache 存的是各层 K/V 投影，不是 hidden state（models 2.4 recap），前序 chunk 的 $z_t$ 不被保留。MEAN 需要全部 $T$ 行、CLS 需要首 token 行，所需行不在当前 chunk 时无法取齐，故代码对 partial prefill 直接 RuntimeError（CLSPool L45-46 逐字「partial prefill is not supported with CLS pooling」，MeanPool L68-69 同形）。∎ 边界（例外）：LAST 只需末 token 行，含末 token 的 chunk 必产出它——LastPool（L51-58）无 partial prefill 检查，可 chunk。

调度含义：CLS / MEAN 模型必须整段 prefill（chunked prefill 对其关闭或整段准入）。但这不产生 4.3 的 stall 问题——stall 账的保护对象是 decode 步时，prefill-only 批里没有 decode 可保护，大 chunk / 整段准入不制造 TBT 尖刺。批形态对照：

| 轴 | decode serving（1.1 / 4.1） | embedding / reranker serving |
| --- | --- | --- |
| 循环形态 | prefill + 逐 token decode 循环 | 单次前向 + pooling，无循环 |
| KV 生命周期 | 跨步增长至请求结束 | 前向内存在，pooling 后即释 |
| 调度目标 | decode 步时与 TTFT 双 SLO | 吞吐 / 端到端延迟（无 TBT） |
| chunked prefill | 保 decode 的核心旋钮（4.3） | CLS/MEAN 禁用（命题 3），LAST 可用 |
| 指标口径 | TTFT / TPOT / TBT 分立 | 非流式 TTFT 退化为 E2E（1.1 §2 recap） |

reranker（重排序器）有两种形态。cross-encoder：query 与 doc 拼接单次前向 + score 头（serving_rerank / score / classify 端点），仍是 prefill-only。late interaction：query 与 doc **独立编码**，token 级交互聚合，代表打分 MaxSim（vLLM 落点 `v1/pool/late_interaction.py`，2026-11-24 目录核实）：

$$
s = \sum_{i=1}^{T_q} \max_{1 \le j \le T_d} \langle u_i, v_j \rangle \qquad (9)
$$

**符号**： $u_i$ ：query 第 $i$ 个 token 的向量； $v_j$ ：doc 第 $j$ 个 token 的向量； $T_q, T_d$ ：query / doc 长度； $\langle \cdot, \cdot \rangle$ ：内积。最小实例：2 query token × 3 doc token，内积矩阵 [[1, 0.5, 0], [0, 0.5, 1]]，逐行取 max 得 1 与 1，s = 2。serving 含义（结构论证）：doc 侧编码与 query 无关，可离线预计算 / 缓存，在线只付 query 编码 + MaxSim 聚合——prefill-only 与 §3 的 cache 复用在此组合；cross-encoder 则每对 (query, doc) 都要在线前向，无此分离。

## 6. diffusion LLM serving 展望

SGLang `dllm/algorithm/` 已有 DllmAlgorithm（base.py L30，step L83 / run L98 / _run_fdfo L149，FDFO = first-done-first-out）与 DllmConfig（config.py L8：block_size / mask_id / first_done_first_out_mode）——masked diffusion 的并行去掩码相对自回归 decode 的 serving 账（多步全序列前向、无 cache 复用、attention 项付 2× 步数倍，账见 models 4.8 §1）只留 recap 指针，本章不展开（spec §2 边界；展望口径 [C]：研究原型、无工业负载验证）。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 多模态链路插入三段：占位展开 + CPU 预处理、VE 前向、embedding 合并；TTFT 串行分解 | 式 (1)；0.5 s → 0.56 s（教学假设） | §1、1.1 |
| patch 账三数分立：n_vit = (w'/14)(h'/14)、n_llm 两维分别 ceil、展开加布局 token | 式 (2)(3)(4)；896² → 4096 / 484 / 508 | §2、练习 1 |
| 每维 ceil ≠ 整体 ceil：ceil(4096/9) = 456 ≠ 484，差 28 token 致对位错位 | downsampled_seqlens L81 逐字 | §2 坑注 |
| min_pixels 放大 / max_image_tokens 缩小是管线两端；LB 尺寸直算是另一层口径 | 448² → 1521 实际 vs 1024 LB 估；1920×1080 → 8418 | §2、7.4 §7 |
| VE 算力 = 参数项 + 双向 attention 二次项（系数 4 非 2）；N_VE ≈ 0.411B | 式 (5)；5.57 TFLOPs/图、attention 39.5%、 $n^{\ast}$ = 6272 | §2、练习 2 |
| VE 单 GPU + cheaper 是官方推荐；~18 图/s/GPU 算力上界（假设 100 TFLOPS） | doc L51 逐字 | §2、7.4 / 8.4 recap |
| fewest-patches 是单调代理非等价：等总 patch 下大图集中负载更高（1×4096 vs 4×1024 attention 4×） | 命题 1；总 FLOPs 比 1.42× | §3、练习 3 |
| encoder cache 命中跳过 ViT；期望算力 (1−η)C；每图 embedding ≈ 4.73 MiB（bf16） | 式 (6)；η=0.8 → 1.11 TFLOPs | §3、5.2 §8 |
| embedding 算一次、chunk 只切片；跨请求 miss 去重单次 ViT call | get_embedding_chunk L33 / _batch_encode_per_image_misses L389 | §3、练习 4 |
| SO 编译异步 + 专用机 + 入引擎前取结果；冷 miss 等待 max(0, t_compile − t_front) | 式 (7)；doc L344-348；200/150 ms 例 | §4 |
| SO 两段成本分离：编译只进 TTFT，每步只 advance+mask；重叠且 τ_mask ≤ τ_forward 时步时零增量 | 命题 2 | §4、5.4 §3 / 6.2 |
| grammar cache 须设字节上限（CVE-2025-32381）；kill switch 保留；SO builder 绑核 4 | compiler.py L17-29；environ L177-178 | §4 坑注 |
| pooling 三法 CLS/Last/Mean；CLS/MEAN 拒绝 partial prefill，LAST 例外 | 命题 3；式 (8)；RuntimeError L45-46 / L68-69 | §5 |
| prefill-only 批无 decode 可保护，整段准入不制造 TBT 尖刺；非流式 TTFT = E2E | 批形态对照表 | §5、4.3 / 1.1 |
| late interaction doc 侧可预计算缓存；MaxSim 逐行 max 求和 | 式 (9)；2×3 例 s = 2 | §5 |
| dllm 展望一句：DllmAlgorithm step/run/FDFO + models 4.8 §1 recap，不展开 | [C] 研究原型口径 | §6 |

## 练习

1.（计算 · [10]）patch 账代入（vision_config [W]，式 (2)(3)(4)）：(a) 672×672 图求 n_vit / n_llm / n_expand；(b) 224×224 图先判断 min_pixels 是否触发，再求三数；(c) 896×448 图求三数。校验：(a) 2304 / 256 / 274；(b) 224² = 50176 < 295936 触发放大，546×546 → 1521 / 169 / 184；(c) 2048 / 242 / 255——与 compute_image_layout（image_processor.py L155）python 复现一致，布局公式对 expanded_image_token_len L76。
2.（计算 · [20]）VE 算力与 cache 节省账（式 (5)(6)）：(a) 按 vit.py 代码结构（Attention 四矩阵 $4d_{\mathrm{VE}}^{2}$ + SwiGLU 三矩阵 $3 d_{\mathrm{VE}} \times 2816$，L96-97 / L130-131）数 $N_{\mathrm{VE}}$；(b) n = 4096 时参数项 / attention 项 / 合计 FLOPs 与 attention 占比；(c) 求二次项反超的 $n^{\ast}$；(d) η = 0.8 时每图期望 VE 算力，按假设有效算力 100 TFLOPS（教学假设）折算毫秒。校验：(a) 411,041,792 ≈ 0.411B；(b) 3.37×10¹² / 2.20×10¹² / 5.57 TFLOPs、39.5%；(c) $n^{\ast} = N_{\mathrm{VE}}/(2 L_{\mathrm{VE}} d_{\mathrm{VE}})$ = 6272；(d) 1.11 TFLOPs ≈ 11.1 ms——与 §2 / §3 正文数值一致。
3.（推导 · [30]）命题 1 闭环（fewest-patches 代理质量）：(a) 补全证明两步（线性项求和交换、二次项交叉项展开），写明等号成立条件；(b) 全数字算反例 1×4096 vs 4×1024 的 attention 项比与总 FLOPs 比；(c) 说明 doc 仍取 patch 代理的两条理由（轻量预估、无需 encoder 运行），并指出 LB 尺寸直算与 resize 管线的第二层近似（用 448×448 例）。校验：(b) attention 项 4×、总 FLOPs 5.57 vs 3.92 TFLOPs（比 1.42×），与 §3 python 复核一致；(c) 1024（LB 估）vs 1521（实际），对回 §2 与 7.4 §7。
4.（代码 · [30]）mm_schedule chunk 切片走读（`managers/mm_schedule.py`，2026-11-24 contents）：(a) 设图像占序列位置 10..493（484 token，items_offset = [(10, 493)]），chunk size 256：手算 get_embedding_chunk（L33）在 chunk 1（prefix 0）与 chunk 2（prefix 256）各取 embedding 哪段行区间，验证两段合计 484 行；(b) 解释为什么 ViT 只跑一次（embedding 整图算好、chunk 只切片）；(c) _batch_encode_per_image_misses（L389）的去重键为什么是 (cache_key, expected_token_count) 双键而非单 hash（读 docstring）；(d) cache 命中但 token count 不符时走哪条路径（L221）。校验：(a) [0:246] 与 [246:484]，246+238 = 484；(c) 防同 hash 不同占位跨度错误去重（docstring 逐字）；(d) 丢弃重算——file:line 锚对研究笔记（W3 核实、本轮行号全同）。

## 参考

- [DeepSeek-AI, inference_deployment_qa.md](../references.md)：key: deepinfer-doc，教学参照 + 官方实现（官方披露，核实 2026-11-23，2026-11-24 byte-identical 再核）——L51 VE 单 GPU cheaper；L151 VE LB machine 随机 + fewest patches 尺寸直算；L344-348 SO 专用机异步编译三阶段逐字
- [DeepSeek-AI, DeepInfer](../references.md)：key: deepinfer-repo，教学参照 + 官方实现——ve/multimodal_processor.py（MultimodalProcessor L29 / overwrite_prompt L105）、ve/processor/image_processor.py（downsampled_seqlens L81 / _layout_from_pixels L87 / expanded_image_token_len L76 / compute_image_layout L155 / safe_resize L131 / solve_resize_ratio L95 / estimate_num_tokens L308）、ve/prepare_tensor_input_vl.py（cache-hit 过滤 L99 / L125）、model_executor/models/vision_encoder/{vit.py（常量 L27-32 / Attention L90 causal=False L116 / SwiGLU L126 / VisionTransformer L168）, vl_aligner.py（L10 / _encode_images L47，docstring L57 / special_embeds L45）, mlp_projector.py（L10 / L16-17）}、npu_ops/vit_attn.py L16、structure_output/{compiler.py（XGrammarCompiler L14 / dispatch_and_compile L54 / compile_grammar L61 / kill switch L17-18 / CVE 警告 L23-29）, matcher.py（XGrammarMatcherFuture L261）}、utils.py get_idle_cpus L154、environ.py L175-178；main@a679bb86（W4 tarball byte-identical 复用）；VE-P-D 编排不在 release（8.4 §6）
- [DeepSeek-AI, V4.1-Flash config.json](../references.md)：key: v41-flash-config，[W]（hf-mirror 传输核实，2026-11-24 第五次重取 byte-identical）——vision_config：patch_size 14 / downsample_ratio 3 / 32L / 1024h / 16 heads / 2816 inter / max_image_tokens 1024 / min_pixels 295936；LLM 侧 d_model 5120 / 40L；模型侧形式化归 models（V4.1 待补）
- [Kwon et al., 2023](../references.md)：vLLM / PagedAttention，arXiv:2309.06180，[A]（key: vllm）——V1 架构与 encoder cache 设计背景
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——v1/core/encoder_cache_manager.py（EncoderCacheManager L18 / check_and_update_cache L111 / can_allocate L148 / allocate L211 / free_encoder_input L259 / reset L96 / compute_mm_encoder_budget L329 / EncoderDecoderCacheManager L392；行号与 W3 同）、v1/core/sched/scheduler.py _try_schedule_encoder_inputs L2111、model_executor/layers/pooler/（abstract.py Pooler L16 / seqwise/methods.py CLSPool L38 / LastPool L51 / MeanPool L61 / seqwise/poolers.py SequencePooler L44）、v1/pool/late_interaction.py（MaxSim）、AsyncLLM.encode；2026-11-24 contents API 核实（HEAD 6aeedab553dd）
- [Zheng et al., 2023](../references.md)：SGLang / RadixAttention，arXiv:2312.07104，[A]（key: sglang）——srt 运行时与约束解码背景
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——managers/mm_schedule.py（init_mm_embedding_cache L28 / get_embedding_chunk L33 / PrefetchedEmbeddingFunc L165 / PerImageRequestInfo L378 / _batch_encode_per_image_misses L389 / _get_chunked_prefill_embedding L593；行号与 W3 同）、entrypoints/openai/serving_{embedding,rerank,classify,score}.py（serving_embedding.py OpenAIServingEmbedding L30 / _handle_non_streaming_request L248）、dllm/algorithm/{base.py（DllmAlgorithm L30 / step L83 / run L98 / _run_fdfo L149）, config.py（DllmConfig L8）}；2026-11-24 contents API 核实（HEAD ac0243723b0d）
- [Dong et al., 2024](../references.md)：XGrammar，arXiv:2411.15100，[A]（key: xgrammar）——grammar automaton 编译与自适应 mask（1.4 §5 原理源头，本章 SO 生产形态的引擎库）
- [vLLM Team, 2025](../references.md)：vLLM V1 blog，[C]（key: vllm-v1-blog，开放技术博客、非闭源）——第 7 点 MLLM first-class：预处理独立进程 + image hash prefix caching + encoder cache 使 MLLM 可 chunked prefill
- 跨维度指引（纯文字）：models 2.4（KV cache 存 K/V 投影非 hidden state）、models 4.8 §1（masked diffusion LLM 与 2N× attention 账，本章 §6 只指针）；inference 1.1（生命周期与 TTFT）、1.2（prefill FLOPs 式 (1) / κ）、1.4 §5-6（mask 原理 / batch invariance）、2.2（命中率节省账）、4.3（chunked prefill 与 stall）、5.2 §8 / 5.4 §3 / §8（encoder cache 接口 / bitmask 异步填充 / 多模态落点）、6.1 §6 / 6.2（constrained backend / bubble 消除账）、7.4 §7（VE LB）、8.4 §6（VE/P/D 与 release 边界）、9.3（权重更新 → encoder cache reset）
