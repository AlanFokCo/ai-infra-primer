# models 维度图品质基线评审（存量评分轮）

评审对象：`site/scripts/lib/figures/*.mjs` + `figures-signature.mjs` 注册、且 dim=models 的全部 **136 张**图（5 张核心签名图 + 模块场景图/锚点图）。量规：`site/scripts/lib/figures/QUALITY.md` 五维 1–5 分制，全维 ≥4 过线。

## 评审方法与证据链（全部实拍，无源码目测）

1. **注册表枚举**：按 `build-content.mjs` 同序同去重（CORE_FIVE → figures/*.mjs 字母序）得 136 键；跨维度无 models 外泄漏。
2. **在站挂载核验**：`npm run build:content && npm run build && npm run preview` 后逐章抓 `data-fig`，**136/136 全部挂载成功**（block 替换与 afterHeading 插入均生效；抽查 `02-transformer-anatomy/04-causality-and-kv-cache` 确认签名图紧跟 `### 4.4 有 cache 的 decode` 锚点标题，`qa/fig-review/insitu-kvcache.png`）。
3. **逐张渲染实拍**：`site/scripts/fig-review.mjs`（本轮新增评审工具，不改图代码）把 136 张注册图以真实 `tokens.css + figure.css + PULSE_GRAD_DEFS`、`.fig.in` 常亮渲染，Playwright（channel: chrome, dsf=2）逐张四帧：正常 t=0.6s / t=6s，reduced-motion t=3s / t=4.5s → `site/qa/fig-review/fNNN-*.png`。疑似出框图（f015/f019/f088）另以 dsf=3 复查（`f0NN-zoom.png`）。
4. **动效度量**：同图早晚帧像素差证运动存在（126/136 有运动，10 张入场后静止或全静态，均为对照/账本类，合宜）；reduced 两帧 canvas 像素差（`qa/fig-review/reduced-diff.json`）测降级失效——阈值 >200px 变色斑块判「仍在动」。
5. **出框度量**：渲染态 `getBBox` 逐 text 对 viewBox 越界检查（5 张命中，其中 f054 为旋转文本的度量误报，视觉复查无出框）。

## 系统性问题归纳

1. **【动效自然度 · 最弱维度 · 25 张失守】reduced-motion 降级被 CSS 特异性打败**：`figure.css:264–272` 的 `@media (prefers-reduced-motion: reduce)` 复位写作 `.fig-sig .sig-*`（特异性 0-2-0），输给常亮规则 `.fig.in .sig-*`（0-3-0），`sig-rotor / sig-diag / sig-dashflow` 的**无限循环动画在降级态照跑**（实测 `matchMedia` 已为 reduce 但 computed `animation-name: sig-spin` 仍在）；聚合模块（part01-02 等）自写的裸 SMIL `<animateMotion>`（非 `.dia-pulse` 包裹）同样不受降级覆盖。受影响 25 张（reduced 两帧 >300px 实质运动）：#1,4,5,6,7,8,12,26,27,28,33,54,56,58,59,88,89,91,92,93,94,95,96,97,98。**一行修复**（复位选择器提权至 `.fig.in .sig-*` 或加 `!important`）可整体解救。另 2 张边界残留（#30 @213px、#57 @221px，单点小面积微动），不封顶但应同查。
2. **【细节完成度】viewBox 出框/截断 4 张**：#15 页脚行 734px>720「…最坏 3 toke」截断且 `embedding` 节点 sub 贴左框；#88 页脚「（§2.3 的」整句截断 + 图例压曲线端点；#19 `concat+W^O` 节点贴 viewBox 底角（底 396/400）且入边回绕节点右侧；#27 注记②出右界 8px。度量另命中 #38 题注 2px 出界（视觉不可见，记注不扣分）。
3. **【动效自然度 · 轻度】10 张入场后即静止**（#20,29,31,44,67,103,107,111,112,120）：均为对照表/账本/频段切分类声明性内容，静止合宜，不扣分；但 kit `pulse` 为 paced 匀速 SMIL，全站无缓动流点，追求「呼吸感循环」需在 kit 层加 keyPoints/keyTimes 配速（评注，不扣现状分）。

## 全量评分表

维度序：动效自然度 / 场景可读性 / 视觉层级 / 细节完成度 / 正文咬合。总评 = min（过线口径）；证据列 = 实拍截图 + 度量。截图均在 `site/qa/fig-review/`（稳态帧 `fNNN-norm-t6000.png`）。

| # | 图（slug # 锚） | 标题 | 动 | 读 | 层 | 节 | 合 | 总评 | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 02-transformer-anatomy/04-causality-and-kv-cache @4.4 | KV cache 的写与读 | 4 | 5 | 5 | 5 | 5 | **4** | f000；reduced 帧字节一致（pulse 正确隐藏）；append 边入 cache 虚线区 |
| 1 | 02-transformer-anatomy/05-positional-encoding @4.2 | RoPE 的成对 2D 旋转 | 3 | 5 | 4 | 5 | 5 | **3** | f001；reduced-diff 6137px：sig-rotor 三转子降级照转（特异性 bug） |
| 2 | 03-objectives-and-variants/01-objectives-and-masks @§1 | mask 形状即 objective 形状 | 5 | 5 | 4 | 5 | 5 | **4** | f002；sig-cell 波次入场（--i 随 i+j 递增）后静止，reduced 静态终态正确 |
| 3 | 04-modern-architectures/05-moe @§2 | top-k 路由与负载 | 4 | 5 | 4 | 5 | 5 | **4** | f003；6 条路由边负载账 2/3,1/3,2/3,1/3 自洽；reduced 静态 |
| 4 | 06-vision-representations/02-clip-and-contrastive @2.1 | batch 相似度矩阵：对角即正样本 | 3 | 4 | 4 | 4 | 5 | **3** | f004；reduced-diff 3525px：sig-diag 对角环降级仍脉冲 |
| 5 | 01-foundations/01-formulation @§3 | 链式法则展开：条件窗口随 t 渐长 | 3 | 4 | 4 | 4 | 5 | **3** | f005；reduced-diff 875px：裸 SMIL 光点（非 .dia-pulse）降级照跑 |
| 6 | 01-foundations/02-compute-accounting @§5 | C = 6ND：2 + 2 + 2 的构成 | 3 | 4 | 4 | 4 | 5 | **3** | f006；reduced-diff 4244px：sig-dashflow 虚线回路 + diag 降级照跑 |
| 7 | 01-foundations/03-scaling-laws @§4 | frontier 上的两个真实预算 | 3 | 5 | 5 | 5 | 5 | **3** | f007；reduced-diff 610px：Chinchilla 描点脉冲环降级不停 |
| 8 | 02-transformer-anatomy/01-overview @§2 | residual stream：归一化读、加法写 | 3 | 5 | 5 | 4 | 5 | **3** | f008；reduced-diff 2124px：总线 dashflow 降级照流；总线脉搏冲点瞬态压 FFN_l 文本 |
| 9 | 02-transformer-anatomy/02-tokenization-and-embeddings @§2 | BPE 逐轮 merge 与频次账本 | 4 | 5 | 5 | 5 | 5 | **4** | f009；全静态账本（声明性内容合宜）；merge 括号与账本条对齐精确 |
| 10 | 01-foundations/01-formulation #block0 | teacher forcing：一次前向，T 个位置并行 | 4 | 5 | 5 | 5 | 5 | **4** | f010；五行位流对齐，loss 汇流排收口式 (4)；reduced 静态 |
| 11 | 01-foundations/02-compute-accounting #block0 | 单层 FLOPs 走读：两个子层与残差写回 | 4 | 4 | 5 | 4 | 5 | **4** | f011；双色分区 + 底部构成条（FFN 81% 主链路最重） |
| 12 | 01-foundations/03-scaling-laws #block0 | log-log 下的幂律切片与 IsoFLOP 剖面 | 3 | 4 | 4 | 4 | 5 | **3** | f012；reduced-diff 359px；「E + A/N^α（D 充分大）」标签压灰曲线 |
| 13 | 02-transformer-anatomy/01-overview #block0 | 一次 forward 全景：ids → stream → 逐位置分布 | 4 | 5 | 4 | 4 | 5 | **4** | f013；720×540 最高图版，层区与头部三段清楚 |
| 14 | 02-transformer-anatomy/01-overview #block1 | 三种架构形态：mask 与连接方式的差异 | 4 | 4 | 5 | 4 | 5 | **4** | f014；decoder-only 区 token 间红点语义略含糊（像相邻流） |
| 15 | 02-transformer-anatomy/02-tokenization-and-embeddings #block0 | 字符串 → bytes → tokens → embedding | 4 | 4 | 4 | 2 | 5 | **2** | f015/f015-zoom：页脚行 734px>720 截断（「…最坏 3 toke」）；embedding 节点 sub 贴左框 |
| 16 | 02-transformer-anatomy/02-tokenization-and-embeddings #block1 | BLT：熵驱动的动态 patch | 4 | 5 | 5 | 4 | 5 | **4** | f016；熵阈值虚线 + 剪刀断点 + patch 合并三层读法顺 |
| 17 | 02-transformer-anatomy/03-attention-mechanism #block0 | 单头 attention 数据流与 T² score | 4 | 4 | 4 | 4 | 5 | **4** | f017；Q/K/V 三角色中文注记（要找/广告/交付）是亮点 |
| 18 | 02-transformer-anatomy/03-attention-mechanism #block1 | multi-head：32 个低秩透镜并行 | 4 | 5 | 4 | 5 | 5 | **4** | f018；扇出 1/2/…/32 省略得体，concat 一次汇合 |
| 19 | 02-transformer-anatomy/04-causality-and-kv-cache #block1 | decode 步 t 的 cache 读写 | 4 | 4 | 4 | 3 | 5 | **3** | f019/f019-zoom：concat+W^O 节点底 396/400 贴底角，入边回绕节点右侧出画感 |
| 20 | 02-transformer-anatomy/05-positional-encoding #block0 | R_m 块对角结构与频率轴 | 4 | 4 | 4 | 4 | 5 | **4** | f020；入场后静态；右栏锚点波长账三段偏文本化 |
| 21 | 02-transformer-anatomy/06-ffn #block0 | 两矩阵 FFN：d → 4d → d | 4 | 5 | 5 | 5 | 5 | **4** | f021；纵链 + 右侧账注，层级干净 |
| 22 | 02-transformer-anatomy/06-ffn #block1 | SwiGLU 门控数据流 | 4 | 4 | 4 | 4 | 5 | **4** | f022；⊙ 汇合点小但标准；Swish 节点标题字号偏大 |
| 23 | 02-transformer-anatomy/07-norm-and-residual #block0 | Post-LN vs Pre-LN：恒等通路与梯度剖面 | 4 | 5 | 5 | 4 | 5 | **4** | f023；上接线 + 下梯度条双联，衰减方向注记精确 |
| 24 | 02-transformer-anatomy/08-assembly #block0 | 一个 block 的完整接线（Llama-3.1-8B） | 4 | 4 | 4 | 4 | 5 | **4** | f024；残差主干左置 + 双子层区，GQA 广播注记在位 |
| 25 | 02-transformer-anatomy/08-assembly #block1 | 整机栈与参数五住处 | 4 | 5 | 4 | 4 | 5 | **4** | f025；栈—账本虚线牵引对个别位级账闭环 |
| 26 | 03-objectives-and-variants/02-encoder-architectures @§7 | 两栈分工：双向读 × 因果写 | 3 | 4 | 4 | 4 | 5 | **3** | f026；reduced-diff 534px：cross-attn dashflow 降级照流 |
| 27 | 03-objectives-and-variants/03-decoder-only-and-icl @§6(b) | induction head：prefix match + copy | 3 | 4 | 4 | 3 | 5 | **3** | f027；reduced-diff 2932px；注记②「…式 7)」出 viewBox 右界 8px（728>720） |
| 28 | 04-modern-architectures/01-kv-compression-and-exact-attention @4.3 | W_UK 吸收：score 打在 latent 上 | 3 | 4 | 4 | 4 | 5 | **3** | f028；reduced-diff 1291px：c_s^KV 长横扫 dashflow 降级照流 |
| 29 | 04-modern-architectures/02-sparse-attention @§1 | 三种可见集：SWA / 块选择 / NSA | 4 | 5 | 4 | 5 | 5 | **4** | f029；全静态三联 mask + 四格图例，声明性内容静止合宜 |
| 30 | 04-modern-architectures/03-linear-attention-and-ssm @§3 | S_t = S_{t-1} + v_t k_tᵀ：状态累加与读出 | 4 | 4 | 4 | 4 | 5 | **4** | f030；reduced 微动残留 213px（单点小面积，不封顶，随特异性修复同查） |
| 31 | 04-modern-architectures/04-hybrid-architectures @§1 | 交错层栈与两笔斜率 | 4 | 5 | 5 | 4 | 5 | **4** | f031；层栈—斜率图双联，交叉点 ≈1.5×10³ 注记点题 |
| 32 | 04-modern-architectures/06-long-context @§2 | η_j 曲线的三种参数化：PI / NTK / YaRN | 4 | 5 | 5 | 5 | 5 | **4** | f032；分段注记（j≤10 / 11..22 / ≥23）落在正确区段 |
| 33 | 04-modern-architectures/07-interpretability-primer @§2 | logit lens：逐层读出，分布成形 | 3 | 5 | 5 | 4 | 5 | **3** | f033；reduced-diff 1013px：+a,+f 总线 dashflow 降级照流 |
| 34 | 03-objectives-and-variants/01-objectives-and-masks #block0 | 五种 objective：同序列 (x′, ℓ, M) 对照 | 4 | 4 | 4 | 4 | 5 | **4** | f034；8 token × 5 目标大对照 + mask 小图，密度高但图例兜底 |
| 35 | 03-objectives-and-variants/01-objectives-and-masks #block1 | GLM：span 塌缩与二维位置 (p⁽¹⁾, p⁽²⁾) | 4 | 5 | 5 | 5 | 5 | **4** | f035；Part A/B 分区与 p1/p2/ℓ 三行寻址示范精确 |
| 36 | 03-objectives-and-variants/01-objectives-and-masks #block2 | T5 span corruption：sentinel 替换 | 4 | 4 | 4 | 4 | 5 | **4** | f036；sentinel→target 弯边略野但有向可读 |
| 37 | 03-objectives-and-variants/02-encoder-architectures #block0 | enc-dec 数据流：cross K/V 一次计算、全程复用 | 4 | 4 | 4 | 4 | 5 | **4** | f037；三区（enc/cache/dec）账注完整 |
| 38 | 03-objectives-and-variants/02-encoder-architectures #block1 | ModernBERT-base 层栈：local/global 交替 | 4 | 4 | 4 | 4 | 5 | **4** | f038；题注 722px 出界 2px（视觉不可见，记注） |
| 39 | 03-objectives-and-variants/03-decoder-only-and-icl #block0 | s_K 布局：prefill 一次付清 | 4 | 4 | 4 | 4 | 5 | **4** | f039；τ 三段拼接 + prefill 虚线区，预测盒右下略孤立 |
| 40 | 03-objectives-and-variants/03-decoder-only-and-icl #block1 | induction head：prefix match + copy 布局 | 4 | 4 | 4 | 4 | 5 | **4** | f040；两层虚线区 + 中文角色注（前驱是 A/我是 A）达意 |
| 41 | 04-modern-architectures/01-kv-compression-and-exact-attention #block0 | MHA / GQA / MQA：谁和谁共享 K/V | 4 | 5 | 5 | 4 | 5 | **4** | f041；三联共享拓扑 + KiB/token 账，对照一目了然 |
| 42 | 04-modern-architectures/01-kv-compression-and-exact-attention #block1 | MLA：低秩 latent + decoupled RoPE 数据流 | 4 | 4 | 4 | 4 | 5 | **4** | f042；cache 区只存两份的点题构型；W^KR 标签压边 |
| 43 | 04-modern-architectures/01-kv-compression-and-exact-attention #block2 | FlashAttention tiling：scores 永不落 HBM | 4 | 4 | 5 | 4 | 5 | **4** | f043；HBM/SRAM 双区 + 编号边 ①②，片上语义清楚 |
| 44 | 04-modern-architectures/02-sparse-attention #block0 | SWA 窗口 pattern 与逐层感受野 | 4 | 5 | 5 | 5 | 5 | **4** | f044；mask + 逐层扩野双联，ρ_ℓ 递推注记精确 |
| 45 | 04-modern-architectures/02-sparse-attention #block1 | NSA：三分支可见集与门控合并 | 4 | 4 | 4 | 4 | 5 | **4** | f045；三分支区 + g 门控汇流，traceable |
| 46 | 04-modern-architectures/02-sparse-attention #block2 | DSA：lightning indexer → top-2048 → MLA | 4 | 4 | 4 | 4 | 5 | **4** | f046；打分条—TopK—主路三段，gather 虚线达意 |
| 47 | 04-modern-architectures/03-linear-attention-and-ssm #block0 | 同一状态方程的三种算图 | 4 | 4 | 4 | 4 | 5 | **4** | f047；parallel/recurrent/chunkwise 三区并列 |
| 48 | 04-modern-architectures/04-hybrid-architectures #block0 | Qwen3-Next：3 GDN + 1 gated attention 周期 | 4 | 5 | 4 | 4 | 5 | **4** | f048；周期链 ×12 读法直接 |
| 49 | 04-modern-architectures/05-moe #block0 | MTP 模块流（D = 1） | 4 | 5 | 4 | 4 | 5 | **4** | f049；双 RMSNorm 入拼接，共享 OutHead 收口 |
| 50 | 04-modern-architectures/06-long-context #block0 | Ring Attention：KV 环流、Q 驻留 | 4 | 4 | 4 | 4 | 5 | **4** | f050；2×2 设备环 + 中心累算盒；环向箭头缺位靠 pulse 补 |
| 51 | 04-modern-architectures/07-interpretability-primer #block0 | residual stream 总线：可读、可删、可换 | 4 | 5 | 4 | 4 | 5 | **4** | f051；读出上挂、写入下挂的总线语义清楚 |
| 52 | 04-modern-architectures/07-interpretability-primer #block1 | SAE：h → 稀疏特征 z → 重构 ĥ | 4 | 5 | 4 | 5 | 5 | **4** | f052；稀疏条（3/13 实格）+ ‖h−ĥ‖² 虚线回扣 |
| 53 | 04-modern-architectures/07-interpretability-primer #block2 | IOI circuit：三类 head 的接力 | 4 | 4 | 4 | 4 | 5 | **4** | f053；早/中/晚三盒接力，「拷贝名字表征」长虚线可追 |
| 54 | 04-modern-architectures/08-frontier-outlook #block0 | masked diffusion：前向腐蚀 × 反向去掩码 | 3 | 5 | 4 | 5 | 5 | **3** | f054；reduced-diff 653px：双向轴 dashflow 降级照流（旋转文本出框为度量误报，视觉无出框） |
| 55 | 05-case-studies/01-llama-lineage @§1 | Llama 谱系 config 账本：逐代字段 diff | 4 | 5 | 5 | 5 | 5 | **4** | f055；朱砂格 = 逐代 diff 的编码直接有效 |
| 56 | 05-case-studies/02-deepseek @3.1 | aux-loss-free：均衡作为控制回路 | 3 | 4 | 4 | 4 | 5 | **3** | f056；reduced-diff 2450px：b_i 控制回路 dashflow 降级照流 |
| 57 | 06-vision-representations/01-vit @§1 | patchify：conv ≡ linear 的 shape 流 | 4 | 5 | 4 | 5 | 5 | **4** | f057；reduced 微动残留 221px（不封顶，同查）；网格→patch→vec→W 读法顺 |
| 58 | 06-vision-representations/03-autoencoders-and-tokenizers @§5 | RVQ 残差层级：逐级细化与 bitrate 账 | 3 | 5 | 5 | 5 | 5 | **3** | f058；reduced-diff 595px：裸 SMIL 光点降级照跑 |
| 59 | 05-case-studies/06-cross-lab-comparison @2.2 | 1M 窗 KV 账的三条出路 | 3 | 4 | 5 | 4 | 5 | **3** | f059；reduced-diff 6363px：三条斜率 dashflow 降级照流（全场最大残动之一） |
| 60 | 05-case-studies/01-llama-lineage #block0 | iRoPE：12 × 4 层周期与可见集分工 | 4 | 4 | 4 | 4 | 5 | **4** | f060；层周期—可见窗双联；NoPE 行 sub 贴框 |
| 61 | 05-case-studies/01-llama-lineage #block1 | early fusion：视觉 token 进同一条 residual stream | 4 | 5 | 4 | 4 | 5 | **4** | f061；视觉链入流折返达意 |
| 62 | 05-case-studies/02-deepseek #block0 | DeepSeek-V3 层栈：3 dense + 58 MoE + MTP | 4 | 4 | 5 | 4 | 5 | **4** | f062；左色块计数 + 三区装配 |
| 63 | 05-case-studies/03-qwen #block0 | Qwen3-Next：config → layer_types → 层装配 | 4 | 5 | 5 | 5 | 5 | **4** | f063；声明→派生→装配三段式干净 |
| 64 | 05-case-studies/04-glm #block0 | GLM objective：Part A / Part B 最小实例 | 4 | 5 | 5 | 5 | 5 | **4** | f064；最小实例 + p1/p2/ℓ 行，与 3.1 图互证 |
| 65 | 05-case-studies/04-glm #block1 | GLM-4.6 层栈：3 dense + 89 MoE + MTP | 4 | 4 | 5 | 4 | 5 | **4** | f065；同模板层栈，RoPE 注记行精确 |
| 66 | 05-case-studies/05-kimi-moonshot #block0 | Kimi-K2 层栈：1 dense + 60 MoE + MuonClip | 4 | 4 | 4 | 4 | 5 | **4** | f066；MuonClip 回路段与层栈并列清楚 |
| 67 | 05-case-studies/06-cross-lab-comparison #block0 | 五家 cache 斜率与 1M 窗整窗账 | 4 | 5 | 5 | 5 | 5 | **4** | f067；条长∝斜率 + 右列整窗账 + 家族色图例，总结图范本 |
| 68 | 05-case-studies/06-cross-lab-comparison #block1 | config → configuration → modeling 阅读路径 | 4 | 4 | 4 | 4 | 5 | **4** | f068；顶链 + 三落点扇出，落点注记有代码实证 |
| 69 | 06-vision-representations/01-vit #block0 | ViT 数据流：patchify → encoder → 读出 | 4 | 5 | 5 | 5 | 5 | **4** | f069；逐步 shape + 右侧式注对位 |
| 70 | 06-vision-representations/02-clip-and-contrastive #block0 | CLIP 一次训练步：双塔汇合与双向 CE | 4 | 5 | 5 | 4 | 5 | **4** | f070；双塔—S—双向 CE—L 汇流分区干净 |
| 71 | 06-vision-representations/03-autoencoders-and-tokenizers #block0 | VAE 数据流：重参数化与两项 loss | 4 | 4 | 4 | 4 | 5 | **4** | f071；encode/decode 纵链；右侧 KL 区红点孤立感 |
| 72 | 06-vision-representations/03-autoencoders-and-tokenizers #block1 | VQ 前向硬量化 / 反向 STE 梯度拷贝 | 4 | 4 | 4 | 4 | 5 | **4** | f072；前向链 + 反向虚线拷贝环，codebook 区高亮 e₃ |
| 73 | 06-vision-representations/03-autoencoders-and-tokenizers #block2 | RVQ 残差级联：逐级细化与求和 | 4 | 5 | 4 | 4 | 5 | **4** | f073；Q 链 + e_i 下落汇入求和线 |
| 74 | 07-diffusion-image-generation/01-generative-model-map #block0 | Part 7 章节地图：推导链与收口 | 4 | 4 | 4 | 4 | 5 | **4** | f074；章节节点 + 依赖边，7.6→7.7 长弯边可追 |
| 75 | 07-diffusion-image-generation/02-ddpm #block0 | 前向 q 链 / 反向 p_θ 链：跳步直达 | 4 | 5 | 5 | 5 | 5 | **4** | f075；上 q 下 p_θ 双轨 + 跳步直达虚线弧，章法漂亮 |
| 76 | 07-diffusion-image-generation/02-ddpm #block1 | 训练循环 vs 采样循环 | 4 | 4 | 4 | 4 | 5 | **4** | f076；左区一条入边穿 ε_θ 盒底（轻交叉） |
| 77 | 07-diffusion-image-generation/02-ddpm #block2 | ε_θ 的 I/O 契约：U-Net 下-上采样 + skip | 4 | 5 | 5 | 5 | 5 | **4** | f077；下/上采样镜像 + skip 虚线直达，对称美 |
| 78 | 07-diffusion-image-generation/03-score-sde #block0 | 前向 SDE 与两条反向动力学 | 4 | 4 | 4 | 4 | 5 | **4** | f078；正向弧 + 双反盒，左端入边略交叠 |
| 79 | 07-diffusion-image-generation/04-sampling-and-guidance #block0 | ancestral 采样环：T 步 = T 次 NFE | 4 | 5 | 5 | 5 | 5 | **4** | f079；环语义一眼可读，Tweedie 旁注「不进环」精确 |
| 80 | 07-diffusion-image-generation/04-sampling-and-guidance #block1 | 步数-质量曲线：solver 阶数与蒸馏 | 4 | 4 | 4 | 4 | 5 | **4** | f080；多 solver 曲线 + 蒸馏学生红段，图例序需对照 |
| 81 | 07-diffusion-image-generation/05-flow-matching #block0 | flow map：密度的 pushforward | 4 | 5 | 4 | 4 | 5 | **4** | f081；三轨密度变形，红黑点半瞬态语义弱 |
| 82 | 07-diffusion-image-generation/05-flow-matching #block1 | 条件直线 vs 边际弯曲 | 4 | 5 | 4 | 4 | 5 | **4** | f082；左直右弯对照 + 绕开交叉区，论证图范本 |
| 83 | 07-diffusion-image-generation/06-latent-diffusion-and-dit #block0 | 像素 → latent：扩散的两阶段分工 | 4 | 5 | 5 | 4 | 5 | **4** | f083；两空间分区 + K 步循环弧 |
| 84 | 07-diffusion-image-generation/06-latent-diffusion-and-dit #block1 | MMDiT 一个 block 的双流数据账 | 4 | 4 | 5 | 4 | 5 | **4** | f084；双流对称 + 拼接中段 + 2×2 分块小图 |
| 85 | 07-diffusion-image-generation/07-t2i-systems-and-ar-image #block0 | SD3 整机数据流 | 4 | 4 | 5 | 4 | 5 | **4** | f085；三区（文本条件/采样环/解码）整机关联完整 |
| 86 | 07-diffusion-image-generation/07-t2i-systems-and-ar-image #block1 | VAR：next-scale 的尺度链 | 4 | 5 | 5 | 4 | 5 | **4** | f086；1×1→16×16 渐密格链 + 前缀回扣虚线 |
| 87 | 07-diffusion-image-generation/07-t2i-systems-and-ar-image #block2 | MaskGIT：置信度 remask 的迭代并行解码 | 4 | 5 | 4 | 4 | 5 | **4** | f087；t=0→T 四帧兑现过程，n=⌈γN⌉ 注记对位 |
| 88 | 07-diffusion-image-generation/02-ddpm @2.3 | 噪声日程：信号逐层淹没与 SNR | 3 | 4 | 4 | 2 | 5 | **2** | f088/f088-zoom：页脚「…（§2.3 的」整句截断出界；图例压曲线端点；reduced-diff 353px 降级微动 |
| 89 | 07-diffusion-image-generation/03-score-sde @6.2 | 同一边缘族：逆时 SDE vs PF-ODE 轨迹 | 3 | 4 | 4 | 4 | 5 | **3** | f089；reduced-diff 860px：轨迹光点降级照跑 |
| 90 | 07-diffusion-image-generation/04-sampling-and-guidance @4.2 | CFG 的向量合成：ε̃ = ε_u + w·Δ | 4 | 4 | 4 | 4 | 5 | **4** | f090；w=0/1/2/3 射线族 + 分布语义 + 范数账三层 |
| 91 | 07-diffusion-image-generation/05-flow-matching @§7 | 路径几何：VP 圆弧 vs rectified flow 弦 | 3 | 5 | 5 | 5 | 5 | **3** | f091；reduced-diff 408px：路径光点降级照跑；弧长/弦长注记精确 |
| 92 | 07-diffusion-image-generation/06-latent-diffusion-and-dit @7.1 | MMDiT 双流与联合 attention 矩阵 | 3 | 5 | 5 | 4 | 5 | **3** | f092；reduced-diff 940px：S_yz 朱砂脉冲（文读图）降级不停 |
| 93 | 08-multimodal-llm/01-vlm-architecture-taxonomy @3.4 | 三个接入点并排：序列 / 层间 / 混合 | 3 | 5 | 5 | 4 | 5 | **3** | f093；reduced-diff 1010px；三栏并排对照本身优秀 |
| 94 | 08-multimodal-llm/05-unified-understanding-generation @§1 | 统一模型的三条路线：瓶颈搬到哪里 | 3 | 4 | 4 | 4 | 5 | **3** | f094；reduced-diff 553px；路线 3 底部支路略孤立 |
| 95 | 09-video-generation/01-video-vae @§1 | 时空压缩账：332 MB → 14.3 MB | 3 | 5 | 5 | 4 | 5 | **3** | f095；reduced-diff 656px：三条压缩箭头光点降级照跑；等距方块与账注优秀 |
| 96 | 09-video-generation/02-spatiotemporal-dit @§2 | full 3D vs factorized：95.2% 的边被砍掉 | 3 | 5 | 5 | 5 | 5 | **3** | f096；reduced-diff 839px：两跳中继 dashflow 降级照流 |
| 97 | 10-audio-speech/01-audio-representations @§2 | STFT → mel：时频面的感知重排 | 3 | 5 | 5 | 5 | 5 | **3** | f097；reduced-diff 3766px：6 处 sig-diag 窗框降级仍脉冲 |
| 98 | 10-audio-speech/02-asr @2.4 | CTC：五条路径的边缘化 = 0.432 | 3 | 4 | 5 | 4 | 5 | **3** | f098；reduced-diff 1285px：DP 网格高亮脉冲降级不停；与正文 0.432 账一致 |
| 99 | 10-audio-speech/04-speech-llm-and-duplex @6.2 | 延迟时间轴：秒级级联 vs 2 帧全双工 | 4 | 5 | 5 | 5 | 5 | **4** | f099；级联/全双工双条 + ms 刻度尺，reduced 静态正确 |
| 100 | 08-multimodal-llm/01-vlm-architecture-taxonomy #block0 | 组件链：vision encoder → connector → LLM | 4 | 5 | 4 | 4 | 5 | **4** | f100；接缝 φ 虚线区点题；节点 sub 贴底框 |
| 101 | 08-multimodal-llm/01-vlm-architecture-taxonomy #block1 | input-level 拼接：v 展开进输入序列（式 3） | 4 | 5 | 5 | 4 | 5 | **4** | f101；占位符展开位 + 下弯边入 LLM，读法直接 |
| 102 | 08-multimodal-llm/02-connectors-and-resolution #block0 | Q-Former：32 个可学 query 的定长瓶颈 | 4 | 4 | 4 | 4 | 5 | **4** | f102；冻结 ViT 区 + 共享 self-attn，瓶颈语义到位 |
| 103 | 08-multimodal-llm/02-connectors-and-resolution #block1 | 三目标的注意力 mask：ITC / ITM / ITG | 4 | 5 | 4 | 5 | 5 | **4** | f103；2×2 块 mask 三联 + 图例行，入场后静止合宜 |
| 104 | 08-multimodal-llm/02-connectors-and-resolution #block2 | pixel shuffle：2×2 合并，token ÷4 通道 ×4 | 4 | 5 | 5 | 5 | 5 | **4** | f104；2×2 邻域高亮→四路→拼接→v 网格，守恒注记精确 |
| 105 | 08-multimodal-llm/02-connectors-and-resolution #block3 | AnyRes：k 块 + 缩略图，M = (k+1)·576 | 4 | 5 | 5 | 4 | 5 | **4** | f105；块/缩略图双路入 v 序列，M 账收口 |
| 106 | 08-multimodal-llm/02-connectors-and-resolution #block4 | M-RoPE 频段切分：(16, 24, 24) | 4 | 5 | 5 | 5 | 5 | **4** | f106；64 块三段色谱 + λ 范围注，静态合宜 |
| 107 | 08-multimodal-llm/03-training-objectives #block0 | loss masking：loss 只落在 response（式 1） | 4 | 5 | 5 | 5 | 5 | **4** | f107；m_s 行 + response 红区，声明性布局静止合宜 |
| 108 | 08-multimodal-llm/03-training-objectives #block1 | 三阶段配方：对齐 → 指令 → 偏好 | 4 | 5 | 4 | 4 | 5 | **4** | f108；三 stage 区冻/训芯片对照 |
| 109 | 08-multimodal-llm/04-video-understanding #block0 | 视频管线：token 省在三个位置 | 4 | 5 | 5 | 4 | 5 | **4** | f109；纵管线 + 右侧三位置虚线挂接 |
| 110 | 08-multimodal-llm/05-unified-understanding-generation #block0 | Janus：解耦双 encoder + 共享 AR 主干 | 4 | 5 | 4 | 4 | 5 | **4** | f110；理解/生成双区汇入共享主干 |
| 111 | 08-multimodal-llm/05-unified-understanding-generation #block1 | Transfusion 序列：BOI/EOI 包夹连续 patch（式 12） | 4 | 5 | 5 | 5 | 5 | **4** | f111；序列条 + 下括号映射双 loss，静态合宜 |
| 112 | 08-multimodal-llm/05-unified-understanding-generation #block2 | Transfusion mask：因果 + 块内双向（式 15） | 4 | 5 | 5 | 5 | 5 | **4** | f112；8×8 mask 块内红格 + 不泄漏论证侧栏 |
| 113 | 08-multimodal-llm/05-unified-understanding-generation #block3 | Transfusion 生图回路：T 步主干内去噪 | 4 | 5 | 5 | 5 | 5 | **4** | f113；①–⑤ 编号链 + 去噪循环虚线区 |
| 114 | 09-video-generation/01-video-vae #block0 | 像素 ⇄ latent 全链：V → μ,σ → z → V̂ | 4 | 5 | 4 | 4 | 5 | **4** | f114；上下双链镜像，体积账随链走 |
| 115 | 09-video-generation/01-video-vae #block1 | k_t = 3 时间卷积：对称 padding vs 因果 padding | 4 | 5 | 4 | 4 | 5 | **4** | f115；对称/因果双区，「未来泄漏 ✗」红虚线达意 |
| 116 | 09-video-generation/02-spatiotemporal-dit #block0 | spacetime patchify：z → X [N×d] → L×DiT → v_θ | 4 | 5 | 5 | 4 | 5 | **4** | f116；N² 靶子账注红字点题 |
| 117 | 09-video-generation/02-spatiotemporal-dit #block1 | dual-stream 段 → single-stream 段（HunyuanVideo） | 4 | 5 | 5 | 4 | 5 | **4** | f117；前后段双区，出口只取视频流注记精确 |
| 118 | 09-video-generation/02-spatiotemporal-dit #block2 | Wan DiT block ×40：self / cross / FFN | 4 | 4 | 5 | 4 | 5 | **4** | f118；残差脊 + 右侧 91%/1.7%/4.2% 占比账 |
| 119 | 09-video-generation/02-spatiotemporal-dit #block3 | SNR-split MoE：t_moe 阈值切换，全程一次 | 4 | 5 | 5 | 5 | 5 | **4** | f119；SNR 曲线阈值线 + 双色 expert 条 |
| 120 | 09-video-generation/02-spatiotemporal-dit #block4 | 3D RoPE 频段切分：64 旋转对的三家分配 | 4 | 5 | 5 | 5 | 5 | **4** | f120；三家 t/h/w 段条对照，高频端注记点题；静态合宜 |
| 121 | 09-video-generation/03-training-and-acceleration #block0 | CausVid 非对称蒸馏：ODE 预训 + DMD 主阶段 | 4 | 4 | 4 | 4 | 5 | **4** | f121；教师/学生四环，score 差边达意 |
| 122 | 09-video-generation/03-training-and-acceleration #block1 | 流式双层循环：chunk 自回归 × K 步退火 | 4 | 4 | 4 | 4 | 5 | **4** | f122；三区（prefill/cache/外层）+ 回写弯边 |
| 123 | 09-video-generation/03-training-and-acceleration #block2 | Diffusion Forcing：逐 token 噪声水平配置空间 | 4 | 4 | 4 | 4 | 5 | **4** | f123；配置空间三代表点 + 底轴语义 |
| 124 | 09-video-generation/04-control-and-editing #block0 | I2V 输入拼接：x_t ; z_c ; m = 36 通道 | 4 | 5 | 5 | 4 | 5 | **4** | f124；三路通道拼接，36=16+16+4 账闭环 |
| 125 | 09-video-generation/04-control-and-editing #block1 | keyframe 链：段间一致性只经接口帧 | 4 | 5 | 4 | 4 | 5 | **4** | f125；段链 + 接口帧节点，式 12 对齐 |
| 126 | 10-audio-speech/01-audio-representations #block0 | neural codec 骨架：Enc → RVQ → Dec | 4 | 5 | 4 | 4 | 5 | **4** | f126；RVQ 区残差递推 + 判别器训练期注记 |
| 127 | 10-audio-speech/01-audio-representations #block1 | Mimi split-RVQ：语义/声学并联求和 | 4 | 5 | 4 | 4 | 5 | **4** | f127；双支路求和，蒸馏只挂语义支路达意 |
| 128 | 10-audio-speech/02-asr #block0 | CTC 状态网格与三类步进 | 4 | 5 | 5 | 4 | 5 | **4** | f128；红路径 + 三类步进注记，可达带收口 |
| 129 | 10-audio-speech/02-asr #block1 | RNN-T 二维格点：右移 / 上移 | 4 | 5 | 5 | 5 | 5 | **4** | f129；格点红路径 + 右列迁移规则注记 |
| 130 | 10-audio-speech/03-tts #block0 | 级联管线：duration + LR 复制展开 | 4 | 4 | 4 | 4 | 5 | **4** | f130；LR 复制扇出边略交叉，整体可读 |
| 131 | 10-audio-speech/03-tts #block1 | VALL-E：AR 第一层 + NAR 逐层精修 | 4 | 4 | 4 | 4 | 5 | **4** | f131；条件区→AR 链→NAR 网格三区 |
| 132 | 10-audio-speech/03-tts #block2 | CosyVoice：LLM + FM + HiFi-GAN | 4 | 5 | 4 | 4 | 5 | **4** | f132；上下双链汇 mel，token 来源注记精确 |
| 133 | 10-audio-speech/03-tts #block3 | F5-TTS：infilling 与 filler 对齐 | 4 | 5 | 4 | 4 | 5 | **4** | f133；mel 帧轴干净/挖空双段 + 文本 filler 行 |
| 134 | 10-audio-speech/04-speech-llm-and-duplex #block0 | 12.5 Hz 栅格上的三流并行 | 4 | 4 | 4 | 4 | 5 | **4** | f134；三流栅格 + c_s 滞后语义，斜虚线格稍密 |
| 135 | 10-audio-speech/04-speech-llm-and-duplex #block1 | RQ-Transformer：一帧的 9 步深度生成 | 4 | 4 | 4 | 4 | 5 | **4** | f135；帧内 9 步纵链 + 帧末回写区 |

## weakest-first 清单（总评 <4，共 28 张）

按最低维排序；同档内按影响面（修复杠杆）排。

### 档 1：min = 2（硬渲染缺陷，必须重画/改版）

| 排名 | # | 图 | 五维 | 缺陷 |
|---|---|---|---|---|
| 1 | 88 | 07-diffusion…/02-ddpm @2.3 噪声日程 | 3/4/4/**2**/5 | 页脚结论句「（§2.3 的…」整句截断出 viewBox（723>720，f088-zoom 实证）；reduced 353px 微动同患 |
| 2 | 15 | 02-transformer-anatomy/02-tokenization-and-embeddings #block0 | 4/4/4/**2**/5 | 页脚行 734px>720 截断（「3 toke」）；embedding 节点 sub 贴左框（f015-zoom 实证） |

### 档 2：min = 3 · 细节失守

| 排名 | # | 图 | 五维 | 缺陷 |
|---|---|---|---|---|
| 3 | 19 | 02-transformer-anatomy/04-causality-and-kv-cache #block1 | 4/4/4/**3**/5 | concat+W^O 节点贴 viewBox 底角（396/400），入边回绕节点右侧（f019-zoom） |
| 4 | 27 | 03-objectives-and-variants/03-decoder-only-and-icl @§6(b) | 3/4/4/**3**/5 | 注记②出右界 8px；叠加 reduced-motion 失守 |

### 档 3：min = 3 · reduced-motion 降级失守（系统性 CSS 特异性 bug，一行修复可解救全部）

| # | 图 | reduced 残动 | 备注 |
|---|---|---|---|
| 1 | 02-…/05-positional-encoding @4.2 RoPE 旋转 | 6137px | sig-rotor ×3 |
| 59 | 05-case-studies/06-cross-lab-comparison @2.2 三条出路 | 6363px | dashflow ×3 |
| 6 | 01-foundations/02-compute-accounting @§5 C = 6ND | 4244px | dashflow+diag |
| 97 | 10-audio-speech/01-audio-representations @§2 STFT→mel | 3766px | sig-diag ×6 |
| 4 | 06-…/02-clip-and-contrastive @2.1 相似度矩阵 | 3525px | sig-diag ×3 |
| 27 | 03-…/03-decoder-only-and-icl @§6(b) induction head | 2932px | dashflow ×2 |
| 56 | 05-case-studies/02-deepseek @3.1 aux-loss-free | 2450px | 控制回路 dashflow |
| 8 | 02-…/01-overview @§2 residual stream | 2124px | 总线 dashflow |
| 98 | 10-audio-speech/02-asr @2.4 CTC 边缘化 | 1285px | DP 高亮脉冲 |
| 28 | 04-…/01-kv-compression @4.3 W_UK 吸收 | 1291px | dashflow |
| 33 | 04-…/07-interpretability @§2 logit lens | 1013px | dashflow ×5 |
| 93 | 08-…/01-vlm-architecture-taxonomy @3.4 三接入点 | 1010px | — |
| 92 | 07-…/06-latent-diffusion-and-dit @7.1 MMDiT 矩阵 | 940px | S_yz 脉冲 |
| 89 | 07-…/03-score-sde @6.2 SDE vs PF-ODE | 860px | 轨迹光点 |
| 96 | 09-…/02-spatiotemporal-dit @§2 full3D vs factorized | 839px | dashflow ×2 |
| 5 | 01-foundations/01-formulation @§3 链式法则 | 875px | 裸 SMIL 光点 |
| 7 | 01-foundations/03-scaling-laws @§4 frontier 预算 | 610px | 描点脉冲环 |
| 95 | 09-…/01-video-vae @§1 时空压缩账 | 656px | 箭头光点 |
| 54 | 04-…/08-frontier-outlook #block0 masked diffusion | 653px | dashflow ×2 |
| 58 | 06-…/03-autoencoders @§5 RVQ 层级 | 595px | 裸 SMIL |
| 94 | 08-…/05-unified-understanding-generation @§1 三条路线 | 553px | — |
| 26 | 03-…/02-encoder-architectures @§7 两栈分工 | 534px | dashflow |
| 91 | 07-…/05-flow-matching @§7 路径几何 | 408px | 路径光点 |
| 12 | 01-foundations/03-scaling-laws #block0 幂律切片 | 359px | diag |
| 88 | 07-…/02-ddpm @2.3 噪声日程 | 353px | 叠加档 1 截断 |

（#30 @213px、#57 @221px 低于 300px 阈值，记为残留观察项，未封顶。）

## 维度均分与结论

- 动效自然度 **3.82**（25 张被降级 bug 封顶 3；其余 pace 匀速流点为基线 4）——最弱维。
- 细节完成度 **4.25**（2 张截断 = 2 分，2 张贴界 = 3 分）——次弱维。
- 视觉层级 ≈4.3 / 场景可读性 ≈4.5 / 正文咬合 5.0（136/136 挂载核验 + 逐张目视：每图均引本章式号/§号/实测数值，fig-tag 全部点题）。
- **过线 108/136（79%）**。修复杠杆排序：① figure.css reduced-motion 复位提权（一行 → 解救 25 张）；② #88/#15 页脚出框改排（2 张）；③ #19/#27 边界回卷（2 张）；④ kit pulse 配速缓动化（全局提升动效维上限）。
