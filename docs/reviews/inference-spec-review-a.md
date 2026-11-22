# Inference Spec 对抗审查 · Route A — P1–P4 事实与数学核查

- 日期：2026-11-22（W0 spec 批准后四路审查之一）
- 审查对象：`docs/superpowers/specs/2026-11-22-inference-dimension-design.md` §5 Part 1–4 大纲、§4.1 证据分级、§4.3 符号表、§6.3 强制重推导清单（P1–P4 子集）、附录预核实记录
- 标准：`STYLE_GUIDE.md`（§6 证据分级、§10 evaluator 协议、§3 公式）+ 本 spec §4 维度细化
- 审查者：fresh-context adversarial evaluator，未继承写作上下文；全部一手来源自行 fetch（arXiv API/abs/html、GitHub contents API / codeload / tree HTML、hf-mirror.com、官方 blog/datasheet）；关键数学独立重推导，不信任 spec 文本
- 环境注记：huggingface.co 本机不可达 → 用 hf-mirror.com（已标注）；raw.githubusercontent.com 本机不可达（curl 000）→ 改用 GitHub contents API / codeload tarball / github tree HTML；GitHub core API 审查中途触发共享 IP 限流（60/hr）→ 后半程改用 codeload + tree HTML + search API（独立配额）
- 工具注记：本机 bare grep 为 shim 损坏，全程 `/usr/bin/grep`

## 0. 结论摘要

| 项 | 结果 |
| --- | --- |
| S1（事实致命） | **0** |
| S2（实质） | **2** |
| S3（minor） | **13** |
| arXiv ID↔标题 | P1–P4 全部 41 个 distinct ID 经 arXiv API 逐一核对，**41/41 正确**；另核 9 个 P5+ / 附录 ID 亦正确（含附录纠错项 2402.07055 确为天线阵列论文）。spec 全文 50 个 distinct ID 全覆盖 |
| repo / 模型存在性 | 30 个被引 repo GET 200 全部存在；V4.1-Flash-1017 与 V3 config 经 hf-mirror 取得且字段与附录逐项吻合；DeepInfer doc 全文读取，P1–P4 相关生产数字逐项吻合 |
| 关键数学 | 8 组强制重推导全部独立重推（§2）；7 组与 spec 一致，1 组（投机解码加速比）发现归一化缺口 → A-S2-1 |
| ⟨verify⟩ 解决 | P1–P4 范围内可解决项基本全部锁定（§3）：官方 code 存在性、博客/规范 URL、Azure trace 归属、genai-perf、硬件规格、config 字段、SGLang 策略名、DeepInfer env var 等 |
| 总体 | 大纲事实底座扎实，无致命错误；2 个 S2 须在 write pass 前修复（加速比公式归一化、vLLM 默认 block size 版本漂移），13 个 S3 为口径/精度/附录计数类，建议随修复闭环逐条判定 |

**覆盖率声明**：P1–P4 大纲中出现的每一个 arXiv ID、每一个被点名的 repo/文件路径、每一个生产数字与 config 字段、每一组强制重推导公式均已核查或重推（逐项见 §4 verification log）。未独立 fetch 的少数次要项（vLLM batch-invariant 具体文件名、B200 datasheet PDF 精确数值、Ascend 950 变体选型）已在 log 中显式标注为「部分核实 / 留 research pass」，不计入「确认无误」。

---

## 1. Findings 表

severity：S1=事实致命（错误公式/张冠李戴/不存在 repo/虚构数字）；S2=实质（约定冲突/推导断裂/局部事实错误）；S3=minor（措辞/口径/格式/可选增补）。

| ID | Sev | 位置 | 断言（spec 原文摘要） | 证据（一手来源，详见 §4 log） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| A-S2-1 | S2 | §5 4.4 关键推导「加速比账」；§4.3 $c_d/c_v$ | walltime improvement factor 写为 $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c_d+c_v))$，称「Leviathan 定理口径」 | Leviathan 2211.17192v2 HTML Theorem 3.8 原文：improvement factor $=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$，其中 $c$ 为 draft/verify **成本比**（无量纲，$c<0.05$）；证明：每轮成本 $Tc\gamma+T$、产 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ token、基线每 token 成本 $T$。spec 式以 §4.3 的绝对成本 $c_d,c_v$ 代入，量纲为 1/time（=tokens/s 吞吐），**不是无量纲 improvement factor**，缺 $\times c_v$ 归一化 | 二选一：(a) 定义 $c:=c_d/c_v$、$c_v$ 归一为 1，用 Leviathan 原式；(b) 保留绝对成本，写 $F=\frac{(1-\alpha^{\gamma+1})\,c_v}{(1-\alpha)(\gamma c_d+c_v)}$。并在 §4.3 注明 $c_d/c_v$ 是否归一。重推导清单 4.4「加速比定理」按修正后口径执行 |
| A-S2-2 | S2 | §5 2.1 内容「block size $b$ 的权衡…默认 16 的出处」 | vLLM 默认 block size = 16 | 当前 main（2026-11-22 fetch）`vllm/config/cache.py` L185：`DEFAULT_BLOCK_SIZE: ClassVar[int] = 64`；L483–484 `block_size is None → DEFAULT_BLOCK_SIZE`；`vllm/platforms/interface.py` `_preferred_block_size_for_backends` 注释明言默认 64、SWA 偏好 256、CPU_MLA 取 16。16 是 V0/早期 V1 与 PagedAttention 论文实验值，非当前引擎默认 | 改「默认 16」为「当前 main 默认 64（`CacheConfig.DEFAULT_BLOCK_SIZE`），按 attention backend 偏好可调（SWA 256 / CPU_MLA 16）；PagedAttention 论文实验用 16」。权衡推导本身正确，仅默认值与出处需版本锚定 |
| A-S3-1 | S3 | §5 4.4「期望接受长度 $E[\alpha]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$」 | 用 $E[\alpha]$ 记期望接受长度 | Leviathan Eq.(1) 原文：$E(\#\text{generated tokens})=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$（capped geometric，cap $\gamma+1$）。$\alpha$ 是接受率**参数**（常量），$E[\alpha]=\alpha$；该式实为「每轮生成 token 数（含拒绝重采样/全接受 bonus 的第 $\gamma+1$ 个）」 | 改记 $E[N_{\text{tok}}]$ 或 $E[\text{len}]$；加口径注：「接受长度」= 每 verify 步产出 token 数（含 bonus），非「被接受的 draft token 数」（后者 $=\frac{\alpha-\alpha^{\gamma+1}}{1-\alpha}$）。与 DSpark doc「accepted length」口径对齐 |
| A-S3-2 | S3 | §5 1.2「decode 每步…KV 读字节 $2LH_{kv}d_{\mathrm{head}}T\cdot\mathrm{bytes}$」+「V4.1 代入 $H_{kv}$=1、$d_{\mathrm{head}}$=512」 | 以 $2LH_{kv}d_{\mathrm{head}}T\,\mathrm{bytes}$ 作为 decode KV 读通式 | 该式仅适用 MHA/GQA（K、V 两份）。MLA 存单份压缩 latent：$\kappa=(d_{\text{latent}}+d_{\text{rope}})\cdot\mathrm{bytes}$，**无因子 2、无 $H_{kv}$ 乘**。V3 config `kv_lora_rank=512`+`qk_rope_head_dim=64`→576；V4.1 `head_dim=512`+`qk_rope=64`→576。若把 V4.1 $H_{kv}$=1/$d_{\mathrm{head}}$=512 代入通式得 $2\times512=1024$/层，较真值 576 高估 1.78×。DeepInfer doc StateBlock 528 B=512(MXFP8)+16(scale) 亦佐证 latent=512 | 通式写 $\kappa L T$（batch 则 $\kappa L T B$），$\kappa$ 逐几何给：MHA/GQA $2H_{kv}d_{\mathrm{head}}\mathrm{bytes}$；MLA $(d_{\text{latent}}+d_{\text{rope}})\mathrm{bytes}$；DSA 另加 indexer K。明确「$2H_{kv}d_{\mathrm{head}}$ 形勿代入 MLA」。1.2 为【重】章，此口径须先锁 |
| A-S3-3 | S3 | §5 1.2「prefill FLOPs $\approx2NT$」「decode 每步 FLOPs $\approx2N$」「GEMV 强度 $\approx1$」 | 以总参数量 $N$ 与 bf16 强度 1 作通式 | MoE 前向只激活 $N_{\mathrm{act}}$：prefill $\approx2N_{\mathrm{act}}T$、decode $\approx2N_{\mathrm{act}}$（§4.3 已区分 $N/N_{\mathrm{act}}$，V3 config 256 routed/top-8 佐证）。GEMV 强度 $=2/\mathrm{bytes}_{\text{param}}$：bf16→1、FP8→2、FP4→4；临界 $B^{\ast}$ 随 dtype 变 | headline 公式注明「dense 用 $N$、MoE 用 $N_{\mathrm{act}}$」；「强度≈1」注明 bf16 基线，FP8/FP4 另算。属口径注，非公式错误 |
| A-S3-4 | S3 | §5 1.2「每 expert batch $=Bk/E$（与 EP 度无关的论证）」 | 无条件断言与 EP 无关 | 独立重推（§2.2）：$Bk/E$ 与 $e$ 无关**当且仅当 $B$=全局 batch（DP 组合计）且路由均衡**。生产 Wide-EP 中 #DP=#EP，若 $B$ 取每 rank 固定值，则全局 $B\propto e$、每 expert batch $=Bek/E$ 随 $e$ 增——这正是 DeepInfer doc L295「Increasing EP raises arithmetic intensity」的机制。两说法在「固定全局 B」vs「固定每 rank B + DP=EP」下各自成立 | 1.2 推导须声明 $B$=全局 batch + 均衡路由前提；并加一句 reconciling 注：DP=EP 耦合扩 $e$ 时全局 B 增→每 expert batch 增→强度升（与 7.4/8.1 衔接）。§4.3 已要求逐章声明 B 口径，落实即可 |
| A-S3-5 | S3 | §5 2.1「vLLM 论文实测浪费率与 <4% 口径 ⟨verify 精确数字⟩」 | 把「<4%」当作论文口径 | arXiv 2309.06180 仅 v1（abs 页 submission history 确认）。v1 HTML 正文口径为「vLLM limits all the memory wastes for a request **within one block**」+「near-zero waste」；全文 grep 无独立「<4%」文字（仅 20.4%–38.2% 为旧系统有效率）。<4% 系 Fig.2 读图/典型长度下 $b/T$ 的经验结果，非论文文字 stated bound，亦非理论上界 | 理论上界写「每请求浪费 $\le1$ block（$b$ token）→ 相对浪费 $<(b)/T$」；「<4%」若引用须注明来自 Fig.2 实测均值/典型长度，随 workload 变，勿当普适上界。⟨verify⟩ 据此关闭 |
| A-S3-6 | S3 | §5 2.2 / 9.1 / 附录「72h TTL 损失 0.5-1% 命中率」 | 未指明损失主体 | DeepInfer doc L187 原文：「a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate」；L211：72h agent / 7d chatbot「captures almost all reusable cache entries」（agent 72h 近乎无损） | 补主体：「**chatbot** 流量在 3d TTL 下损失 0.5–1% 命中；agent 72h 几乎无损（且跨周末）」。避免读者误以为 agent 72h 损失 0.5–1% |
| A-S3-7 | S3 | §5 2.4「三层存储账…DRAM：百 GB/s、μs、TB；SSD：GB/s、ms、百 TB——官方规格出处」 | 延迟列未声明视角 | 介质本征延迟：DRAM ~50–100 ns、NVMe SSD ~10–100 μs；「DRAM μs / SSD ms」只在 **GPU 访问视角**（H2D over PCIe ~μs；经网络+3FS 取 SSD ~百 μs–ms）成立。doc 未给介质延迟表 | 表头注明「GPU 侧访问延迟（含 PCIe/网络栈）」，或分列「介质本征 / GPU 访问」两口径；「官方规格出处」对应到具体 datasheet/3FS 文档，勿以介质规格标 μs/ms |
| A-S3-8 | S3 | §5 1.3「M/M/1 等待时间 $W=1/(\mu-\lambda)$」 | 称 $W$ 为「等待时间」 | 排队论标准：$W=1/(\mu-\lambda)$ 是**逗留时间**（排队+服务）；纯排队等待 $W_q=\rho/(\mu-\lambda)=\lambda/(\mu(\mu-\lambda))$。Kingman VUT 近似的是 $W_q$ | 注明 $W$=逗留时间（sojourn），$W_q$=排队等待；1.3/9.1 用 Kingman 时对象为 $W_q$。公式本身正确，仅命名口径 |
| A-S3-9 | S3 | §4.3 符号表 | 表未收 $E$（expert 数）、$k$（top-k）、$c$（4.3 chunk size） | 1.2/7.4 用 $Bk/E$、4.3 用 chunk size $c$；§4.3 表只有 $N/N_{\mathrm{act}},B,T,L,H_{kv},d_{\mathrm{head}},\kappa,b,t,p,e,d,\alpha,\gamma,c_d/c_v,\lambda,\mu,\rho,\mathcal{G}$。$c$ 与 $c_d/c_v$ 易混 | 表补 $E$=routed expert 数、$k$=top-k、$c$=chunk size（4.3 专用，与 $c_d/c_v$ 区分）；或在对应章首声明。消歧硬规则已覆盖 B/b/bytes/γ/λ，补此三项即闭合 |
| A-S3-10 | S3 | 附录「arXiv ID 核实（48 查 47 中）」「全部候选 arXiv ID」 | 预核实清单漏 2412.19437 | spec 正文 1.2/3.2/3.3 共 4 处引 [arXiv:2412.19437]（DeepSeek-V3），但附录 48 查清单（14+15+16+2+1）无此 ID。arXiv API 核 2412.19437=「DeepSeek-V3 Technical Report」✓ 正确，仅附录覆盖漏登。（另 9.3 的 2409.19256 HybridFlow 亦不在清单——属 route C，附带提示） | 附录补登 2412.19437（及提示 route C 补 2409.19256）；「全部候选」措辞改为与实际清单一致。ID 本身无误，不升 S2 |
| A-S3-11 | S3 | 附录 / §5 引言「25 个关键 repo 经 GitHub API 预核实」 | 计数 25 vs 实列 24 | 附录 L528 逐项枚举：vllm,sglang,DeepInfer,TRT-LLM,lmdeploy,nixl,dynamo,LMCache,Mooncake(9) + deepseek-ai{FlashMLA,DeepGEMM,DeepEP,EPLB,3FS,DeepSelect,TileKernels,deepseek-recipe}(8) + tilelang,flashinfer,flash-attention,xgrammar,outlines,guidance,marlin(7) = **24**。§4.1 另列 gptq/llm-awq/EAGLE 等未计入该 24 | 改「25」为「24」或补列第 25 个（如 IST-DASLab/gptq / mit-han-lab/llm-awq / SafeAILab/EAGLE，本次均 GET 200 存在）。计数对齐 |
| A-S3-12 | S3 | 附录 L528 / §5 5.2「`v1/core/{…kv_cache_interface.py,kv_cache_layout.py}`」 | 把两文件归于 v1/core/ | GitHub contents API：`vllm/v1/core` 子项 = {__init__,block_pool,encoder_cache_manager,kv_cache_config_builder,kv_cache_coordinator,kv_cache_manager,kv_cache_metrics,kv_cache_planning,kv_cache_utils,sched/,single_type_kv_cache_manager}，**无** kv_cache_interface/layout；二者实在 `vllm/v1/` 根（API `vllm/v1` 子项含 kv_cache_interface.py,kv_cache_layout.py） | 附录与 5.2 把 kv_cache_interface.py / kv_cache_layout.py 路径改为 `vllm/v1/`（非 v1/core/）。2.1 所列 v1/core 四文件均存在✓，不受影响。（5.2 属 route B，附带提示） |
| A-S3-13 | S3 | §5 4.5 / 5.4「vLLM `v1/spec_decode/` eagle + ngram（+mtp）⟨verify 文件名⟩」 | 假定 spec_decode 下有平铺 eagle.py/ngram.py/mtp.py | GitHub contents API `vllm/v1/spec_decode` = {__init__.py, **dynamic/**, metrics.py}；tree HTML `spec_decode/dynamic` = {__init__.py, utils.py}；该页全文 grep「eagle」=0。当前 main 已重构，平铺 eagle/ngram/mtp 文件名不在 spec_decode/ 下（实现迁往他处/dynamic 重组） | ⟨verify 文件名⟩ 据此关闭为「目录存在、平铺文件名已漂移」；W2/W3 research pass 须定位当前 EAGLE/ngram/MTP proposer 实际文件并锚定（含函数级），勿沿用旧平铺名。（5.4 属 route B，附带提示） |

** cross-route 观察（不计入 route A findings，供 B/C/D 参考）**：
- 附录/7.2「528 B / 1,415,040 B/seq」以 `B` 作 bytes 单位后缀，与 §4.3「bytes 不占字母」字面冲突；若视为单位（如 GB）而非数学字母可豁免，建议 §4.3 加一句单位豁免注或统一写「字节」。doc L223 原文即用「528 B」。
- DeepGEMM README（codeload main）确认 **Mega MoE / lightning-indexer MQA scoring / HyperConnection 均为 DeepGEMM 组件**；7.3 kernel 栈映射宜注明 Mega MoE ⊂ DeepGEMM（spec 3.3/7.3 经 doc 口径引 Mega MoE，无误，仅归属可更精确）。
- 9.3 的 2409.19256（HybridFlow）不在附录预核实清单（route C）；本次 API 核 = 「HybridFlow: A Flexible and Efficient RLHF Framework」✓。

---

## 2. 独立重推导（不信任 spec，逐组重推）

### 2.1 Batching 算术强度 ∝ B 与临界 B*（1.2）
decode 权重 GEMM：batch $B$ token，权重 $N$ 参数。FLOPs $=2NB$；权重字节 $=N\cdot w$（$w$=每参数字节，batch 共享，不随 $B$ 增）。忽略 activation/KV（权重 GEMM 不含 KV），强度 $I=2NB/(Nw)=2B/w$。bf16 $w=2$→$I=B$（GEMV $B=1$ 时 $I=1$）✓ **∝ B 成立**（FP8 $w=1$→$I=2B$，FP4→$4B$，见 A-S3-3）。compute-bound 临界对齐 ridge point：$I=B^{\ast}=\text{peak FLOPS}/\text{BW}$。H100 SXM bf16：$989\times10^{12}/3.35\times10^{12}\approx295$；H200 同算力、BW 4.8 TB/s→$\approx206$。**与 spec 一致**。注意：此 ∝B 仅指权重 GEMM；decode attention 的 KV 读随 $B$ 线性增，其强度不随 $B$ 涨（长 $T$ 时饱和 $\approx2N/(\kappa LT)$），1.2 须分开账（spec 已分列 KV 读✓）。

### 2.2 MoE 每 expert batch = Bk/E 与 EP 无关（1.2）
$E$ routed experts、top-$k$、全局 batch $B$ token→token-expert 指派总数 $Bk$；均衡路由下每 expert 平均 $Bk/E$。EP 度 $e$ 只改变 expert 在 GPU 间的划分（每卡 $E/e$ 个），不改变每 expert 收到的 token 数→**$Bk/E$ 与 $e$ 无关**✓。每卡指派 $=(E/e)(Bk/E)=Bk/e$；每卡权重流量 $\propto(E/e)\cdot P_{\text{exp}}\cdot w$；每卡 FLOPs $\propto(Bk/e)\cdot2P_{\text{exp}}$；每卡强度 $=\frac{(Bk/e)2P_{\text{exp}}}{(E/e)P_{\text{exp}}w}=\frac{2Bk}{Ew}$（$e$ 约掉，**强度本身与 $e$ 无关**）。Wide-EP 加速机制是每卡**绝对**字节/算力 $\propto1/e$ 下降→memory-bound 时间 $\propto1/e$，直到 all-to-all 通信/同步时间主导（7.4「每步时间∝1/e」✓）。**前提**：$B$=全局 batch + 均衡。若 $B$=每 rank 且 #DP=#EP=$e$，全局 $B=B_{\text{rank}}e$→每 expert $=B_{\text{rank}}ek/E$ 随 $e$ 增（doc L295「raises intensity」即此耦合场景）。spec 断言在固定全局 B 下正确，须声明口径（A-S3-4）。

### 2.3 PagedAttention 碎片上界 + block-size 权衡（2.1）
块从左到右填、满才 alloc 新块→每请求内部碎片 $\le b-1$ token（末块未满）；统一块大小→无外部碎片；V1 不预 reserve max-len→无 reservation 浪费（parallel sampling/beam 经 COW 共享）。**理论上界：每请求浪费 $<1$ block $=\kappa L b$ 字节；相对浪费 $<b/T$**（论文原文「within one block」✓）。$b=16$、典型 $T\sim400$→$<4\%$（经验，非普适上界，A-S3-5）。权衡：小 $b$→内部碎片 $\approx(b-1)/2$ 小、prefix-hash 粒度细（命中率高），但 block table 项 $\lceil T/b\rceil$ 多→元数据/CPU/kernel gather 间接开销大；大 $b$→表开销小、locality 好，但碎片大、prefix 共享粒度粗。当前 main 默认 64、backend 偏好可调（A-S2-2）。**推导与 spec 一致**（默认值需更新）。

### 2.4 GPTQ 误差补偿（3.2）
layer-wise 目标 $\lVert WX-\hat WX\rVert_F^2$；单行 $w$ 的 Hessian $H=2XX^{\top}$（各行共享）✓。逐列量化第 $q$ 列：误差 $e=(w_q-\hat w_q)/[H^{-1}]_{qq}$，对剩余列 $F$ 做 OBS 补偿 $W_{:,F}\mathrel{-}=e\cdot H^{-1}_{q,F}$（最小化条件 MSE）。GPTQ 用 $H^{-1}$ 的 Cholesky 上三角保数值稳定、lazy batch（128 列块）更新、act-order（按 $H$ 对角降序=激活幅度）✓。**与 spec「逆 Cholesky 逐列量化+误差补偿、lazy batch、act-order」一致**。

### 2.5 AWQ scaling 等价（3.2）
per-channel $s$：$(W\,\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}X)=WX$✓（逐通道 $(W\cdot s)(X/s)=WX$）。salient ~1% 通道由**激活幅度**定（非权重幅度）；$s=s_X^{\alpha}$ 归一后 grid search $\alpha\in[0,1]$ 最小化 MSE；放大 salient 权重通道降低相对量化误差、同步缩小激活✓。**与 spec 一致**。

### 2.6 SmoothQuant 迁移（3.2）
$s_j=\max|X_j|^{\alpha}/\max|W_j|^{1-\alpha}$；$\hat X=X\,\mathrm{diag}(s)^{-1}$、$\hat W=\mathrm{diag}(s)W$；$Y=\hat X\hat W=XW$✓。激活 outlier 除以 $s$ 变平滑、权重乘 $s$ 变难量化（但权重本易量化）；$\alpha$ 权衡迁移难度（默认 0.5）✓。**与 spec 公式一致**。

### 2.7 投机解码：分布保持 + 期望长度 + 加速比（4.4）
**分布保持**：接受 prob $\min(1,p(x)/q(x))$；拒绝从 $p'=\mathrm{norm}(\max(0,p-q))$ 重采样。设 $Z=\sum_x\max(0,p-q)=\sum_x\max(0,q-p)=1-\beta$（$\beta=\sum\min(p,q)$）。则 $P(\text{out}=x)=q(x)\min(1,p/q)+Z\cdot\frac{\max(0,p(x)-q(x))}{Z}=\min(p,q)+\max(0,p-q)=p(x)$✓ **严格等于目标 $p$**（与 Leviathan A.1 证明逐项吻合）。
**期望长度**：接受数 $N$ 为 capped geometric（$P(N\ge k)=\alpha^k$，cap $\gamma$）；每轮产出 $N+1$（拒绝位重采样 1 个 / 全接受 bonus 1 个）。$E[N+1]=1+\sum_{k=1}^{\gamma}\alpha^k=1+\frac{\alpha(1-\alpha^{\gamma})}{1-\alpha}=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$✓ **公式正确**（Leviathan Eq.1）。但 spec 记 $E[\alpha]$ 误（$\alpha$ 是参数），且「接受长度」含 bonus 须声明（A-S3-1）。
**加速比**：每轮成本 $\gamma c_d+c_v$，产 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ token；基线每 token 成本 $c_v$。improvement factor $F=\frac{E[\text{tok}]\cdot c_v}{\gamma c_d+c_v}=\frac{(1-\alpha^{\gamma+1})c_v}{(1-\alpha)(\gamma c_d+c_v)}$。令 $c=c_d/c_v$ 即 Leviathan Thm 3.8 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$✓。**spec 式缺 $\times c_v$（或 $c_v=1$ 归一），量纲为吞吐而非 factor**→A-S2-1。

### 2.8 Chunked-prefill stall 账（4.3）
无 chunk：长 prefill $P$ token 独占 GPU $t_P\approx aP+bP^2$，期间所有 decode 停→TPOT 尖刺 $=t_P$。chunk $c$：prefill 切 $\lceil P/c\rceil$ 块，每迭代混 1 chunk + piggyback decode；decode 每迭代 stall $\le t_{\text{chunk}}(c)\approx ac+bc\cdot(\text{已算前缀})$。权衡：$c$ 太小→prefill GEMM 强度 $\propto c$ 落 memory-bound、attention 逐 chunk 重读前缀 KV、launch 开销→prefill 效率损失；$c$ 太大→$t_{\text{chunk}}$ 大→decode stall 长。最优 $c$ 在 TPOT SLO 约束下最大化 prefill 吞吐（Sarathi-Serve stall-free）✓。DeepInfer compute-quota：时间预算 `compute_quota_limit`(ns)、拟合 $\mathrm{cost}(q,c)$ 二次（$q$=chunk token、$c$=已算前缀）、解二次取最大 $q\le32768$ 恰耗尽时间预算、DP 组 all-reduce MIN（doc L143 逐字✓）。**与 spec 一致**。

### 2.9 附：M/M/1（1.3）
$W=1/(\mu-\lambda)$（逗留）、$W_q=\rho/(\mu-\lambda)$、$L=\lambda/(\mu-\lambda)$、$\rho=\lambda/\mu$；Kingman $W_q\approx\frac{\rho}{1-\rho}\cdot\frac{c_a^2+c_s^2}{2}\cdot\frac1\mu$。公式正确，命名口径见 A-S3-8。

---

## 3. ⟨verify⟩ 项解决表（P1–P4 范围）

| ⟨verify⟩ 项 | 解决结果 | 一手证据 |
| --- | --- | --- |
| DistServe code | **存在**，LLMServe/DistServe（官方完整实现，SwiftTransformer 后端）→ 可升 [A] | GET 200 + codeload README |
| Splitwise code | **部分**：Mutinifni/splitwise-sim（官方模拟器，非完整生产栈）→ [B]（代码不全） | GET 200 |
| Sarathi / Sarathi-Serve code | **存在** microsoft/sarathi-serve → [A] | GET 200 |
| FastServe code | 未发现官方 repo（search 仅无关同名）→ [B] | GitHub search |
| Llumnix code | **存在** AlibabaPAI/llumnix → [A] | GET 200 |
| CachedAttention code | 未发现官方 repo（search 空）→ [B] | GitHub search |
| CacheBlend code | **存在** YaoJiayi/CacheBlend（207★；UChi-JCL/CacheBlend 为 404）→ [A] | search + GET 404 对照 |
| Preble code | **存在** WukLab/preble（111★「Stateful LLM Serving」）→ [A] | GitHub search |
| KIVI / KVQuant code | jy-yuan/KIVI、SqueezeAILab/KVQuant → [A] | GET 200 |
| H2O / StreamingLLM / SnapKV code | FMInference/H2O、mit-han-lab/streaming-llm、FasterDecoding/SnapKV → [A] | GET 200 |
| CacheGen code | UChi-JCL/CacheGen（176★）→ [A] | search |
| SpecInfer code | goliaro/specinfer-ae（AE）+ flexflow/FlexFlow 集成 → [A/B]（实现散落/AE） | search |
| REST / Medusa code | FasterDecoding/REST（221★ NAACL24）、FasterDecoding/Medusa → [A] | search / GET 200 |
| Draft & Verify code | dilab-zju/self-speculative-decoding（231★，论文明示）→ [A] | search |
| EAGLE repo | SafeAILab/EAGLE → [A] | GET 200 |
| NanoFlow code | efeslab/Nanoflow（978★）→ [A] | search |
| LLM.int8 code | TimDettmers/bitsandbytes → [A] | GET 200 |
| Marlin ~4× | README「close to ideal (4x) speedups up to batchsizes 16-32」「max 3.87x」✓ | codeload README |
| batch-invariance 博客 URL | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ✓（标题 Defeating Nondeterminism in LLM Inference） | GET 200 + <title> |
| OCP MX 规范 URL | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ✓（MX Spec v1.0, Sep 2023；§5.4 E8M0 scale） | GET 200 + PDF p1-2 |
| NVFP4 / MXFP8 粒度 | NVFP4=per-16 E4M3+FP32 global；MXFP8/MXFP4=per-32 E8M0 ✓ | NVIDIA blog/TE docs（search）+ OCP PDF + DeepGEMM README |
| Azure trace 归属 | AzureLLMInferenceDataset2023.md 明示「data described and analyzed in ISCA 2024 Splitwise」，cite Patel et al. ✓ 归 Splitwise 正确 | codeload md L4/L16 |
| genai-perf | triton-inference-server/perf_analyzer 下 `genai-perf/` ✓ | tree HTML path |
| SGLang schedule_policy 策略名 | LPM="lpm"、DFS_WEIGHT="dfs-weight"、FCFS="fcfs"、LOF="lof"、RANDOM="random" ✓（spec 三名均在） | contents API schedule_policy.py L191-203 |
| SGLang speculative 路径 | python/sglang/srt/speculative/ ✓（eagle_*.py、dspark_*、cpp_ngram/ 等） | tree HTML |
| vLLM spec_decode 文件名 | 平铺 eagle/ngram/mtp **不在**当前 spec_decode/（={__init__,dynamic/,metrics}）→ 漂移，须重锚（A-S3-13） | contents API + tree HTML |
| DeepInfer tilelang_ops 路径 | deepinfer/model_executor/layers/tilelang_ops ✓ | tree HTML path |
| DeepInfer DEEPINFER_BATCH_INVARIANT | environ.py L204 `DEEPINFER_BATCH_INVARIANT = EnvVar.bool()` ✓ | codeload grep |
| DeepInfer compute-quota / TTL / KV 格式 / EP>32 / DSpark 表 / StateBlock / GDR / 80% | 全部与 doc 逐字吻合（§4 log） | contents API doc 全文 |
| V4.1-Flash config 字段 | 40L/64heads/kv_heads1/head_dim512/qk_rope64/q_lora1280/o_lora1024/o_groups8/384+1 top6/sqrtsoftplus/noaux_tc/swa128/compress_ratios43/index_topk512/index_n_heads32/index_head_dim128/nextn3/vocab129280/hidden5120/maxpos1048576 YaRN×16 from65536/quant fp8 block[32,32] ue8m0 expert fp4/vision 32L1024 patch14 down3/image_token129264/dspark_*/engram_*/hc_*/kv_source_layer_ids[2,8,14,20] ✓ 全吻合 | hf-mirror config.json |
| V3 nextn=1 / MoE 几何 | num_nextn_predict_layers=1、256 routed top8、61L、kv_lora512+rope64 ✓ | hf-mirror V3 config |
| H100/H200 规格 | H200 页：BF16 989 TFLOPS dense/1979 sparse、FP8 1979/3958、HBM3e 4.8 TB/s、141GB ✓；H100 同算力、3.35 TB/s、80GB | nvidia.com H200 GET |
| B200 精确数值 | **部分**：产品页现主推 Rubin NVL8（400 PF NVFP4/130 TB/s 为 Rubin 非 B200）；B200 官方 datasheet PDF 为稳定源，精确值留 W1 research pass 锁定（spec 未断言具体 B200 数，非错误） | b200.html GET（Rubin 头条） |
| Ascend 950 规格 | 官方 Huawei Connect 2025-09 keynote：950 系 1 PF FP8/2 PF FP4、互联 2 TB/s；950PR HiBL1.0 128GB 1.6TB/s（Q1'26）、950DT HiZQ2.0 144GB 4TB/s（Q4'26）。注意 Atlas 350 卡级数（1.56PF FP4/112GB）≠ 芯片级，章须区分变体 | huawei.com keynote（search 汇总） |
| vLLM <4% 浪费 | 论文文字无 <4%；上界=1 block/请求；<4% 为 Fig.2/典型长度经验（A-S3-5） | arXiv v1 HTML + abs |
| vLLM 默认 block 16 | 当前 main=64（A-S2-2） | cache.py/interface.py |

---

## 4. Verification log（抽查项 × 来源 × 结论 × 覆盖）

格式：项 | 来源 URL / 原文引句 | 结论。所有 fetch 于 2026-11-22 本机执行。

### 4.1 arXiv ID↔标题（export.arxiv.org API，id_list 批量，3 批 50 ID 全返回）
来源：`https://export.arxiv.org/api/query?id_list=…`（b1/b2/b3.xml，Atom entry id+title 逐条解析）。
- 2309.06180 → Efficient Memory Management for LLM Serving with PagedAttention ✓（1.2/2.1/2.2/4.1/5.x）
- 2412.19437 → DeepSeek-V3 Technical Report ✓（1.2/3.2/3.3；**附录漏登** A-S3-10）
- 2401.09670 → DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving ✓（1.1/1.3/8.2/9.1）
- 2311.18677 → Splitwise: Efficient generative LLM inference using phase splitting ✓（1.1/8.2）
- 2411.15100 → XGrammar: Flexible and Efficient Structured Generation Engine for LLMs ✓（1.4）
- 2307.09702 → Efficient Guided Generation for LLMs（Outlines）✓（1.4）
- 1904.09751 → The Curious Case of Neural Text Degeneration（top-p）✓（1.4）
- 2407.01082 → Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs ✓（1.4）
- 2312.07104 → SGLang: Efficient Execution of Structured Language Model Programs ✓（2.2/4.1/6.x）
- 2403.19708 → Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention ✓（2.2）
- 2405.16444 → CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion ✓（2.2）
- 2407.00023 → Preble: Efficient Distributed Prompt Scheduling for LLM Serving ✓（2.2）
- 2402.02750 → KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache ✓（2.3）
- 2401.18079 → KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization ✓（2.3，10M 口径吻合）
- 2306.14048 → H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs ✓（2.3）
- 2309.17453 → Efficient Streaming Language Models with Attention Sinks ✓（2.3）
- 2404.14469 → SnapKV: LLM Knows What You are Looking for Before Generation ✓（2.3）
- 2407.00079 → Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving ✓（2.4/8.2）
- 2310.07240 → CacheGen: KV Cache Compression and Streaming for Fast LLM Serving ✓（2.4）
- 2307.08691 → FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning ✓（3.1）
- 2407.08608 → FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision ✓（3.1）
- 2501.01005 → FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving ✓（3.1）
- 2210.17323 → GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers ✓（3.2）
- 2306.00978 → AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration ✓（3.2）
- 2211.10438 → SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs ✓（3.2）
- 2208.07339 → LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale ✓（3.2）
- 2504.17577 → TileLang: A Composable Tiled Programming Model for AI Systems ✓（3.4/7.3）
- 2408.12757 → NanoFlow: Towards Optimal Large Language Model Serving Throughput ✓（3.4/6.2）
- 2305.05920 → Fast Distributed Inference Serving for LLMs（FastServe）✓（4.2）
- 2406.03243 → Llumnix: Dynamic Scheduling for LLM Serving ✓（4.2/8.5）
- 2308.16369 → SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills ✓（4.3）
- 2403.02310 → Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve ✓（4.3/8.2）
- 2211.17192 → Fast Inference from Transformers via Speculative Decoding（Leviathan）✓（4.4）
- 2302.01318 → Accelerating LLM Decoding with Speculative Sampling（Chen）✓（4.4）
- 2305.09781 → SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification ✓（4.4）
- 2311.08252 → REST: Retrieval-Based Speculative Decoding ✓（4.4/4.5）
- 2401.15077 → EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty ✓（4.5）
- 2406.16858 → EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees ✓（4.5）
- 2503.01840 → EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test ✓（4.5）
- 2401.10774 → Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads ✓（4.5）
- 2309.08168 → Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding ✓（4.5；附录纠错正确）
- 2402.07055 → Optimization of Super-Directive Linear Arrays with Differential Evolution for High Realized Gain（天线阵列）✓ 附录「误记→纠正」属实
- 附（P5+/附录）：2405.05254 YOCO ✓、2502.11089 NSA ✓、2409.19606 Hyper-Connections ✓、2409.19256 HybridFlow ✓、2311.03285 S-LoRA ✓、2310.18547 Punica ✓、2404.09526 LoongServe ✓、2406.17565 MemServe ✓
**覆盖：spec 全文 50 distinct ID 全核（P1–P4 41/41 正确）；无张冠李戴。**

### 4.2 论文数学原文（arxiv.org/html）
- Leviathan 2211.17192v2 `https://arxiv.org/html/2211.17192v2`：
  - §2.3「keeping it if q(x)≤p(x)…reject with probability 1−p(x)/q(x)…adjusted distribution p′(x)=norm(max(0,p(x)−q(x)))」→ 接受 prob min(1,p/q) ✓
  - A.1「P(x)=min(p,q)+(p−min(p,q))=p」→ 分布保持 ✓（与 §2.7 重推一致）
  - Eq.(1)「E(#generated tokens)=(1−α^{γ+1})/(1−α)」→ 期望长度公式 ✓、记号应为 #tokens 非 E[α]（A-S3-1）
  - Thm 3.8「expected improvement factor in total walltime…(1−α^{γ+1})/((1−α)(γc+1))」+ 证明「each run costs Tcγ+T…baseline T」→ c 为成本比；spec 绝对成本式缺归一（A-S2-1）
- vLLM 2309.06180v1 `https://arxiv.org/html/2309.06180v1` + abs：
  - 「vLLM limits all the memory wastes for a request within one block」→ 碎片上界=1 block ✓
  - 「near-zero waste」；旧系统有效内存 20.4%–38.2%（Fig.2）
  - 全文无独立「<4%」文字；abs submission history 仅 [v1] Tue 12 Sep 2023 → <4% 非文字 stated bound（A-S3-5）

### 4.3 repo / 路径存在性（github.com GET 200 / contents API / codeload / tree HTML）
- 30 repo GET 200：vllm-project/vllm、sgl-project/sglang、deepseek-ai/{DeepInfer,DeepSelect,TileKernels,deepseek-recipe,FlashMLA,DeepGEMM,DeepEP,EPLB,3FS}、kvcache-ai/Mooncake、LMCache/LMCache、ai-dynamo/{nixl,dynamo}、tile-ai/tilelang、flashinfer-ai/flashinfer、IST-DASLab/{marlin,gptq}、mit-han-lab/{llm-awq,smoothquant}、SafeAILab/EAGLE、mlc-ai/xgrammar、dottxt-ai/outlines、guidance-ai/guidance、NVIDIA/TensorRT-LLM、InternLM/lmdeploy、Dao-AILab/flash-attention ✓（组织更名 outlines/Mooncake 现名正确）
- 官方 code 补核（GET/search）：LLMServe/DistServe✓、Mutinifni/splitwise-sim✓、microsoft/sarathi-serve✓、AlibabaPAI/llumnix✓、jy-yuan/KIVI✓、SqueezeAILab/KVQuant✓、FMInference/H2O✓、mit-han-lab/streaming-llm✓、FasterDecoding/{SnapKV,Medusa,REST}✓、punica-ai/punica✓、S-LoRA/S-LoRA✓、TimDettmers/bitsandbytes✓、triton-inference-server/perf_analyzer✓、Azure/AzurePublicDataset✓；search：YaoJiayi/CacheBlend✓、WukLab/preble✓、UChi-JCL/CacheGen✓、efeslab/Nanoflow✓、dilab-zju/self-speculative-decoding✓、goliaro/specinfer-ae✓；UChi-JCL/CacheBlend=404（正确 repo 为 YaoJiayi）；CachedAttention/FastServe 无官方 repo
- vLLM 路径（contents API）：`vllm/v1`={core,engine,executor,attention,sample,spec_decode,structured_output,kv_offload,simple_kv_offload,hisparse,kv_hints,metrics,worker,kv_cache_interface.py,kv_cache_layout.py,…}✓；`vllm/v1/core`={block_pool.py,kv_cache_manager.py,kv_cache_coordinator.py,single_type_kv_cache_manager.py,sched/,kv_cache_config_builder.py,kv_cache_planning.py,encoder_cache_manager.py,…}✓（2.1 四文件全在）；**kv_cache_interface/layout 在 v1 根非 core**（A-S3-12）；`vllm/v1/spec_decode`={__init__.py,dynamic/,metrics.py}、dynamic/={__init__.py,utils.py}（A-S3-13）
- vLLM 默认 block（contents API raw）：cache.py L185 `DEFAULT_BLOCK_SIZE=64`、L483 None→64；interface.py `_preferred_block_size_for_backends` 注释「default…64…SWA preference of 256…CPU_MLA takes 16」（A-S2-2）
- SGLang 路径（contents API + tree HTML）：srt={managers,mem_cache,disaggregation,eplb,elastic_ep,batch_overlap,batch_invariant_ops,constrained,connector,checkpoint_engine,dllm,compilation,speculative,…}✓；managers={scheduler.py,schedule_policy.py,schedule_batch.py,data_parallel_controller.py,overlap_utils.py,prefill_delayer.py,shortest_prefill_load.py,hisparse_coordinator.py,mm_schedule.py,…}✓；schedule_policy.py L191-203 LPM/DFS_WEIGHT/FCFS/LOF/RANDOM ✓；speculative/={eagle_*.py,dspark_*,cpp_ngram/,draft_*.py,…}✓
- DeepInfer 路径（tree HTML + codeload）：deepinfer/model_executor/layers/tilelang_ops ✓；environ.py L204 DEEPINFER_BATCH_INVARIANT ✓；layers={attention_backend,layernorm,linear,mtp_info_pool,rotary_embedding,sampler,tilelang_ops,vocab_parallel_embedding,…}
- Marlin README（codeload master）：「close to ideal (4x) speedups up to batchsizes of 16-32」「near ideal 4x speedup over FP16 weights」「maximum possible 3.87x」✓（3.3 ~4× 口径）
- DeepGEMM README（codeload main）：「SM90/SM100」「SM100 requires…packed UE8M0」「SM100 accepts MXFP4/MXFP8…UE8M0 scales」「fused MoE…(Mega MoE)」「MQA scoring for the lightning indexer」「HyperConnection (HC)」「DeepJIT」✓（3.3 UE8M0-for-Blackwell/JIT；cross-route：Mega MoE⊂DeepGEMM）

### 4.4 DeepInfer doc（contents API raw，doc/inference_deployment_qa.md 全文 34.5KB）
- L31「TTFT and TBT」→ 1.1「DeepInfer 用 TBT」✓
- L155「roughly 80% of a perfectly balanced offline benchmark」→ 1.1 ✓
- L143「time budget (compute_quota_limit, in nanoseconds)…fitted quadratic cost model cost(q,c)…q=number of sequence tokens, c=prefix length already computed…largest q…within the 32,768-token budget…all-reduce MIN within the DP group」→ 4.3 逐字 ✓
- L187「TTL of 3 days for agent traffic and 7 days for chatbot…a TTL of around 3 days costs **chatbot traffic** 0.5%–1% of KV cache hit rate」+ L211「72 hours for agent…7 days for chatbot captures almost all」→ 2.2 TTL ✓、损失主体=chatbot（A-S3-6）
- L271/273/275「Main KV: per-16, e4m3 — like NVFP4 but without the global scale」「Indexer KV: per-32, ue8m0, MXFP4」「MoE routed-expert weights: per-32, ue8m0, MXFP4」→ 2.3/3.2/7.3 ✓
- L293/295/297「Mega MoE…stop improving as EP grows beyond 32 during decode」「Once memory bandwidth is no longer the bottleneck…additional communication and synchronization overhead」「Different jobs sharing NVLink…interfere」→ 3.3 EP>32 ✓
- L285「Mega MoE is the general solution within a scale-up domain」→ 3.3 ✓
- L289/291「FlashMLA lacks one hand-tuned SASS optimization…decoding kernel 5%–10% slower」「End-to-end…at most 5% apart」→ 3.1/7.3 ✓
- L303/307 + 表 L309-316「draft model generates 5 tokens…Confidence Head…sorted by expected acceptance rate…prefix subset that maximizes effective throughput」「5.00 → 3.2–4.4…3.29 vs. 3.20…+4–7%…~60 concurrent sequences per rank」；表 conc60:+5.4%、conc72:+6.7% → 4.4/7.2 DSpark 数字逐字 ✓（注：表内 dynamic accepted 逐行略低于 fixed，增益来自 verify 成本降而非 accepted 增；7.2 目标 accepted/cost 正确）
- L324「schedule(N+1)…CPU scheduling only fixes the total budget; per-request lengths recomputed on GPU by a TileLang kernel (cumulative product, global sort for top-k, fused remap scatter)…pinned buffer」→ 7.2 ✓
- L217/219/223「stored layers are the first 20」「ring buffer…6 extra positions…overshoot at most 5」「state_size_to_disk = 20 layers × 134 tokens × 528 B = 1,415,040 B…528 B/token is the MXFP8 SWA KV format」→ 7.2/2.4 ✓；算术 20×134×528=1,415,040 ✓；528=512+512/32 ✓ 内部自洽
- L245「maximum hit length…at most L − 134 (128-token window + 6 tokens of speculative backtracking)…hit length L1…otherwise L1 − 128」→ 7.2 hit policy ✓
- L191「Entrypoint…CHECK role…hash-chain prefix match over metadata…CUDA copy engine performs poorly on fine-grained copies…swap-in implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs」→ 2.2/2.4 ✓
- L187/111「roughly 100 PiB of SSD…10+2 erasure coding」→ 2.4 ✓
- L17「155 trillion tokens per day…overall revenue of US$38 per GPU-hour (at the off-peak price)」→ 9.1 收入口径 ✓
- L97/101/105「400 Gbps…per 8 GPUs」「20 × 30.73 TB SSDs…at least 40 GB/s read and 5 GB/s write」「about 14.4 TB per GPU…logical」→ 8.3/9.1 ✓
- L253/257「grace period (1 min)…forwarded in one shot」「scale-in forbidden whenever quota utilization above 0.8」→ 8.5 ✓
- L127「4 redundant experts…100-step statistics window…re-assignment every 1,000 steps…moe_lb…1.074 (EP4DP4)/1.045 (EP2DP2)」→ 7.4 ✓
- L368/370「state machine…incremental KMP matching for delimiter markers」→ 1.4/7.4 ✓
**覆盖：附录 DeepInfer doc 关键事实段 + P1–P4 所引 doc 口径逐项核对，全部吻合（仅 TTL 损失主体需补，A-S3-6）。**

### 4.5 config（hf-mirror.com，huggingface.co 不可达已标注）
- `https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/config.json`（3311B）：architectures DeepseekV41ForCausalLM；text 40L/64heads/kv_heads1/head_dim512/qk_rope64/q_lora1280/o_lora1024/o_groups8/384routed+1shared/top6/sqrtsoftplus/noaux_tc/swa128/compress_ratios(43)/index_topk512/index_n_heads32/index_head_dim128/nextn3/vocab129280/hidden5120/maxpos1048576/rope yarn factor16 orig65536；quant fp8 weight_block_size[32,32] scale_fmt ue8m0 expert_dtype fp4；vision 32L/1024h/patch14/downsample3；image_token_id 129264；dspark_block_size5/markov_rank256/target_layer_ids[37,38,39]/n_routed128/top3；engram_*、hc_*、kv_source_layer_ids[2,8,14,20] → **与附录逐项吻合** ✓（1.2/3.2/4.5/7.x config 口径）
- `https://hf-mirror.com/deepseek-ai/DeepSeek-V3/resolve/main/config.json`：nextn1、256routed/top8/1shared、61L、kv_lora512/qk_rope64 → 4.5「V3 nextn1」✓、1.2 MoE 例几何 ✓

### 4.6 非 arXiv 一手来源
- Orca `https://www.usenix.org/conference/osdi22/presentation/yu` GET 200，<title>「Orca: A Distributed Serving System for Transformer-Based Generative Models | USENIX」→ OSDI'22、无 arXiv ✓（4.1）
- FlashDecoding `https://crfm.stanford.edu/2023/10/12/flashdecoding.html` GET 200，正文「Flash-Decoding for long-context inference…load the keys and values in parallel…separately rescale and combine」→ split-KV 源头、CRFM Stanford ✓（3.1）
- batch-invariance `https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/` GET 200，<title>「Defeating Nondeterminism in LLM Inference - Thinking Machines Lab」→ URL 解决 ✓（1.4/3.4）
- OCP MX `https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf` GET 200，PDF p1「OCP Microscaling Formats (MX) Specification Version 1.0…Sep 2023」、TOC §5.4 Scale Data Types/E8M0 → URL+格式 ✓（2.3/3.2）
- NVFP4：NVIDIA developer blog「Introducing NVFP4…」+ Transformer Engine docs（search 汇总）：E2M1 元素、per-16 E4M3 scale、FP32 per-tensor global；MXFP4 per-32 E8M0 → 3.2 粒度 ✓
- genai-perf：perf_analyzer tree HTML `"path":"genai-perf"` ✓（1.1）
- Azure trace：codeload AzurePublicDataset `AzureLLMInferenceDataset2023.md` L4「data described and analyzed in the ISCA 2024 paper 'Splitwise…'」、L16 cite Patel et al. Splitwise ISCA24 → 归属 Splitwise ✓（1.1）
- H200 `https://www.nvidia.com/en-us/data-center/h200/` GET 200：989/1979/3958 TFLOPS、4.8TB/s、141GB ✓（1.3）
- B200 `https://www.nvidia.com/en-us/data-center/b200/` GET 200：现头条为 Rubin NVL8（400PF NVFP4/130TB/s），B200 精确值须 datasheet PDF（部分核实，留 W1）
- Ascend 950：Huawei Connect 2025-09 keynote（huawei.com/en/news/2025/9/hc-xu-keynote-speech，search 汇总）：1PF FP8/2PF FP4/2TB/s 互联；950PR 128GB1.6TB/s、950DT 144GB4TB/s；Atlas350 卡级数另计（1.3，变体须区分）

### 4.7 符号 / 分级 / 重推导清单一致性（P1–P4）
- §4.3：P1–P4 大纲 bytes 均用 $\mathrm{bytes}$/文字（1.2 ✓），无 B 表 bytes；$b$=block（2.1 ✓）；goodput 用文字/𝒢 无 G 占用 ✓；γ/α 专归投机 ✓；λ 到达率须声明（spec 已注 ✓）。缺 E/k/c 入表（A-S3-9）；附录/7.2「528 B」单位豁免待定（cross-route）
- §4.1 分级：Orca[B]（paper 无 code）✓；Leviathan/Chen[B]（无官方开源）✓；DistServe/Splitwise/Sarathi/Llumnix/KIVI/KVQuant/CacheBlend/NanoFlow/Medusa/REST/Draft&Verify/EAGLE 等 ⟨verify code⟩ 经 §3 解决后多可升 [A]（Splitwise 模拟器→[B]、SpecInfer AE→[A/B]、CachedAttention/FastServe 无 code→[B]）；DeepInfer/TRT-LLM/LMDeploy/FlashMLA/DeepGEMM/DeepEP/EPLB/3FS/DeepSelect/TileKernels/Marlin「教学参照+官方实现」✓（均无独立 paper、repo 存在）；DeepInfer 生产数字「官方披露」不升[A]不降[C] ✓
- §6.3 重推导清单 P1–P4 子集（1.2/2.1/2.4/3.1/3.2/4.3/4.4）与 §4.4【重】标记一致 ✓；本报告 §2 已全重推

---

## 5. 修复优先级建议

1. **S2 必修（write pass 前）**：A-S2-1（4.4 加速比归一化，连带 §4.3 $c_d/c_v$ 口径）、A-S2-2（2.1 默认 block 64 + 版本锚定）。
2. **S3 随闭环逐条判定**：口径类（A-S3-1/2/3/4/7/8）建议在 1.2/1.3/2.4/4.4 章首口径注落实；附录类（A-S3-10/11/12）直接改附录；漂移类（A-S3-13）转 W2/W3 research pass 重锚；A-S3-5/6/9 改对应大纲措辞。
3. 无 S1，无需含 S1 章节复审触发；S2 修复后建议对 4.4 / 2.1 两章大纲做定点复核。

（route A 审查完毕；B/C/D 报告独立并行，cross-route 观察仅供 triage 参考。）
