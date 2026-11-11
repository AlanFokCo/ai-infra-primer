# W2 波末独立对抗审查报告（P3 + P4，11 章）

- 审查日期：2026-11-12；evaluator：独立 fresh-context 对抗审查（不继承写作上下文）。
- 审查对象：models/03-objectives-and-variants/01–03、models/04-modern-architectures/01–08，共 11 章成品。
- 方法：全文通读 + 一手来源抽查（arXiv API / ar5iv 全文 / hf-mirror config / GitHub API 与 repo 文件）+ 三项强制独立重推导 + STYLE_GUIDE 逐条对照 + lint。
- 机械检查：11/11 章 `scripts/lint-chapter.sh` 全部 LINT OK；公式白名单扫描（`\boxed`/`\color`/align 系环境/`\newcommand`/math 围栏/`\operatorname*`/`\text{}` 内中文）全部干净；AI 味禁令词零命中。

## 一、Findings 表

| # | severity | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | 3.2 总结表（02-encoder-architectures.md L236）；4.2 §3（02-sparse-attention.md L106） | 3.2 总结表写「3:1 local…/global」；4.2 §3 写「encoder 侧的 ModernBERT 3:1」并进入「5:1 vs 3:1 vs 3:1」混合比并列 | 3.2 §10 正文与 ASCII 图均正确（`global_attn_every_n_layers=3` ⇒ 每 3 层 1 global + 2 local = **2:1** local:global；论文原文 "every third layer employs global attention"，config 实测同）；总结表与 4.2 的引用与正文及 config 矛盾 | 两处「3:1」改为「2:1」（或「每 3 层 1 个 global」）；4.2 §3 的并列改为「5:1 vs 2:1 vs 4.4 的 3:1」 |
| F2 | **S2** | models/README.md 进度表（L144–159） | 3.1 标「已写」，3.2–4.8 共 10 章仍标「未开始」 | 11 章成品均在磁盘且 lint 通过；README 自称「全维度唯一进度事实源」，spec §6.4/§6.5 要求随波更新 | 将 11 章状态统一改为当前真实状态（建议「审查中」），W2 复审通过后转「已关闭」 |
| F3 | S3 | 3.1 §4.2 式 (8)（01-objectives-and-masks.md L167–170） | 符号注写「损失按 span 总 token 数归一」，但式 (8) 显示的是未归一的双重求和 | GLM 论文（arXiv:2103.10360）原目标函数亦无归一化；本章式 (2)(4)(7)(12) 均带 1/· 归一，式 (1)(8) 不带——约定不一致 | 二选一：式 (8) 加归一因子并在符号注写明「论文为未归一形式」；或删掉「归一」字样并在 §1 说明归一约定 |
| F4 | S3 | 4.6 §1.2 表与正文、§3.3 表（06-long-context.md L33、L40、L231） | 「GQA-8（Llama-3.1-8B）T=10^6 单序列 cache 128 GiB」「每步要读 128 GiB」 | 独立重算：2×32×8×128×10^6×2 B = 1.31072×10^11 B = **122.1 GiB**（或 131 GB）；同表 MLA 行 65.4 GiB 正确 | 改 122 GiB（或「约 120 GiB」）；「40 ms @ 3 TB/s」结论不变 |
| F5 | S3 | models/references.md「被引章节」字段 | 至少 16 个 key 未回写 W2 新引用章：gpt2（缺 3.3、4.7）、deepseekv3（缺 4.2、4.6）、deepseek-v3-config（缺 4.5）、llama-3.1-8b-config（缺 4.1、4.6）、llama-3.1-70b-config（缺 4.1）、llama-modeling（缺 4.1）、gemma3（缺 4.6）、deepseek-v3.2（缺 4.6、4.8）、kimi-linear（缺 4.6）、kimi-linear-config（缺 4.6）、minimax01（缺 4.6）、qwen3-next（缺 4.5、4.6）、kimi-k3（缺 4.8）、nope（缺 4.4）、chinchilla（缺 4.5）、llama3（缺 4.6） | 逐章 grep `key: …` 实引集合与 references.md 字段 diff（本报告 verification log V14） | 按「写入规则」回写被引章节 |
| F6 | S3 | models/references.md 术语表 | 术语表止于 2.8，P3/P4 新术语（blank infilling、span corruption、sentinel、RTD、attention sink、aux-loss-free、MTP、chunkwise/WY、logit lens、SAE、dual-chunk、NTK-aware 等）未登记 | STYLE_GUIDE §1：全书术语表由 references.md 附带、各波次追加；章节内首现「中文（English）」格式已合规 | 补登 W2 术语行 |
| F7 | S3 | references.md `moba` 条目及 4.2 锚点行 | MoBA 标 [A]，但 MoonshotAI/MoBA repo 仅有代码（目录：moba/ examples/ tests/），无开放权重——按 STYLE_GUIDE §6 三齐定义应为 [B] | GitHub API contents 实测无权重；章节对 MoBA 的使用全部为论文公式级（路由式 (15)(16)、免训练声明），无权重派生断言，内容不受影响 | 条目降 [B]（或补注「代码官方、无权重」）；4.2 锚点行同步 |
| F8 | S3 | 4.1 §3（01-kv-compression-and-exact-attention.md L60） | 「论文的速度-质量对照（Fig 6）据此把 GQA-8 作为中间点选择」 | GQA 论文（ar5iv:2305.13245）Fig 6 仅为 speed vs GQA groups；质量侧数据在 Table 1 / Fig 5；「We selected 8 groups as a favorable middle ground」一句在 Fig 5/6 讨论段 | 改为「Fig 5/Table 1 质量 + Fig 6 速度」或「论文据此」 |
| F9 | S3 | 3.1 §3.5 第 3 条（01-objectives-and-masks.md L118） | 把官方 `create_masked_lm_predictions` 的执行序描述为单次掷签「<0.8 → [MASK]；<0.9 → 随机词；否则保留」 | 官方代码（google-research/bert，已取原文核对）为两次掷签：`< 0.8` → [MASK]，否则第二次 `< 0.5` → 保留原 token、否则随机词。净分布同为 80/10/10，但分支次序与「官方执行序」表述不符（章内教学复刻代码用的是章节次序且已声明差异范围不含此项） | 描述改为「<0.8 → [MASK]；余下再掷一次均分保留/随机」 |
| F10 | S3 | 4.4 §5 vs 4.5 §1 | Qwen3-Next 稀疏度两处口径不一：4.4 §5 写「约 2.0%」（10/512，只算 routed），4.5 §1 写「ρ ≈ 2.1%」（(10+1)/513，含 shared，与 4.5 自身 ρ=(k+K_s)/N_e 定义一致） | 两值各自内部一致；config 实测 512 experts + top-10 + 1 shared | 4.4 §5 加注「不含 shared」或统一用含 shared 口径 |
| F11 | S3 | 全部 11 章篇幅 | 实测总字符数 10.0K–29.5K（3.1=23.6K、4.3=29.5K、4.8=10.0K 等），超 STYLE_GUIDE §8 篇幅约定（标准章 4000–7000、推导重章 6000–9000、4.8 短篇 2500–4000）2–4 倍 | `wc -m` 逐章实测；W1 波同形态且 W1 报告记为「均在标准章区间内」——W1 的口径判断与本指南条款本身存在系统性张力 | 下一波统一清零时裁决：或修指南口径、或压篇幅 |

severity 汇总：S1 = 0；S2 = 2（F1、F2）；S3 = 9（F3–F11）。

## 二、Verification log（抽查项 × 来源 × 结论）

### A. 强制独立重推导（三项，全部 PASS）

**A1. 4.1 MLA（RoPE×低秩冲突 → decoupled → absorption → 576 账）— PASS**
- 冲突：cache 只存 $c_n$ 时 $s_{t,n}=\langle R_tq_t, R_nW^{UK}c_n\rangle=\langle q_t, R_{n-t}W^{UK}c_n\rangle$（用 2.5 式 (13)，已核 05-positional-encoding.md L183 原文）——$R_{n-t}$ 卡中间 ✓。三条出路的封闭论证中第三条我独立验证：与全部块对角旋转 $\{R_\Delta\}$（各块角速度 $\theta_j$ 互异）交换的矩阵必为逐块 scaled rotation（对 $\Delta$ 在 0 处求导得 $M$ 与 $\mathrm{diag}(\theta_j J)$ 交换 ⇒ 非对角块满足 $\theta_i J M_{ij} = \theta_j M_{ij}J$，$\theta_i\neq\theta_j$ 时 $M_{ij}=0$）——与全混合压缩矩阵不相容 ✓，与 V2 论文 §2.1.3 原文（"a RoPE matrix … will lie between W^Q and W^UK"）一致（ar5iv:2405.04434）✓。
- Absorption：$\langle q^C,k^C\rangle=(c^Q)^\top(W^{UQ})^\top W^{UK}c^{KV}$，shape $d_c'\times d_h$ × $d_h\times d_c$ = $d_c'\times d_c$ ✓；$\mathrm{rank}\,\tilde W_h\le d_h=128$ ✓。V 侧 $o_t=\sum_h W_h^O W_h^{UV} m_{t,h}$，$\widehat W_h\in\mathbb{R}^{d\times d_c}$ ✓ 对称成立。
- 账：576 = 512+64 ✓（DeepSeek-V3 config 实测 kv_lora_rank=512、qk_rope_head_dim=64、q_lora_rank=1536、v_head_dim=128、num_attention_heads=128、61 层、max_position_embeddings=163840——hf-mirror 逐字段命中）；参数账逐项复算 11.0+25.2+12.6+3.7+8.4+8.4+0.5+117.4 ≈ 187M ✓，MHA 反事实 4×117.4 ≈ 470M ✓，61 层 11.4B vs 28.7B ✓；decode 读取 32832/576 = 57.0 ✓；吸收形 score FLOPs 比 576/192 = 3 ✓；V3 满窗 70272 B × 163840 = 10.7 GiB ✓，反事实 MHA 610 GiB ✓（56.9×）。
- V2 论文「MLA stronger than MHA」（§2.1.3 末句原文）与 absorption（"W^UK can be absorbed into W^Q, and W^UV into W^O" 原文）✓。

**A2. 4.3 DeltaNet→GDN→KDA 状态更新链 — PASS**
- DeltaNet：$\mathcal L_t=\frac12\|Sk_t-v_t\|^2$，$d\mathcal L=(Sk-v)^\top(dS\,k)=\mathrm{tr}(k(Sk-v)^\top dS)$ ⇒ $\nabla_S=(Sk-v)k^\top$ ✓；一步 GD 重排即式 (13) ✓。$\beta=1,\|k\|=1$：$S_tk_t=v_t$ ✓；$u\perp k_t$：$S_tu=S_{t-1}u$ ✓。
- GDN 式 (14) 对照论文（ar5iv:2412.06464 式 (10)：$S_t=S_{t-1}(\alpha_t(I-\beta_tk_tk_t^\top))+\beta_tv_tk_t^\top$，同布局 $S\in\mathbb{R}^{d_v\times d_k}$）逐符号一致 ✓。数值例 $0.9\times(1-0.5)=0.45$ ✓。退化链 α=1→DeltaNet ✓、β=0 写入同归于尽 ✓（与 GDN-2 解耦动机一致）。
- GDN-2 式 (15) 对照论文式 (10)（ar5iv:2605.22791，原文 $S_t=(I-k_t(b_t\odot k_t)^\top)D_tS_{t-1}+k_t(w_t\odot v_t)^\top$，布局 $d_k\times d_v$）转置后逐项一致 ✓；负特征值变体 $b_t\in[0,2]^{d_k}$（论文 kernel 节原文"If negative eigenvalues are enabled, only the erase gate is scaled by 2"）✓；RetNet 档为本书推导（全文无 RetNet 字样，约定页口径属实）✓；实证 1.3B/100B FineWeb-Edu、multi-key 检索优势最大（摘要原文）✓。
- KDA 式 (16) 对照 Kimi Linear 论文式 (1)（ar5iv:2510.26692：$S_t=(I-\beta_tk_tk_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top\in\mathbb{R}^{d_k\times d_v}$）转置一致 ✓；逐列展开 $(S_t)_{:,j}=\alpha_j(S_{t-1})_{:,j}+\beta k_j(v-S\mathrm{diag}(\alpha)k)$ 独立复算一致 ✓；DPLR 特化（摘要原文 "specialized variant of the Diagonal-Plus-Low-Rank transition matrices"）✓。
- WY：C=2 展开系数对位 $W_{11}=\beta_1,W_{22}=\beta_2,W_{12}=-\beta_1\beta_2k_1^\top k_2,W_{21}=0$ ✓；递推式 $(I-KWK^\top)(I-\beta_jk_jk_j^\top)$ 展开验证 $W_{jj}=\beta_j$、$W_{1:j-1,j}=-\beta_jW(K^\top k_j)$ ✓。
- 回忆上界命题两方向证明独立重做 ✓；SNR $d_k/(n-1)$：$d_k=128,n=1024$ 得 0.125（−9.03 dB）✓；半衰期 $\ln2/\ln(1/\gamma)$：0.99→69.3、0.999→693 ✓；FLOPs 交叉点 $T=2d$ ✓；decode 账 64 MiB vs 32 KiB = 2048× = 2t/d ✓。
- 实现布局注记（HF `torch_recurrent_gated_delta_rule` 状态 $(d_k,d_v)$）经 modeling_qwen3_next.py L566–570 原文核实（`k_t.unsqueeze(-1) * delta.unsqueeze(-2)`）✓。

**A3. 4.5 MoE routing — PASS**
- 协方差分解：$f_iP_i=(f_i-1/N)(P_i-1/N)+(f_i+P_i)/N-1/N^2$ 求和、交叉项为零 ⇒ $\sum f_iP_i=\sum(\cdot)(\cdot)+1/N$ ⇒ $L_{aux}=\alpha(1+N^2\widehat{\mathrm{Cov}})$ ✓ 独立复算一致。
- 对角最小性：Cauchy–Schwarz $(\sum f_i^2)N\ge(\sum f_i)^2=1$、等号 iff 均匀 ✓。
- 梯度：$\partial L_{aux}/\partial p_i(x_t)=\alpha N_e f_i/T$ ✓；$f$ 分段常数 ⇒ 梯度只走 $P$ ✓；诚实边界（乘积单纯形上双线性形无约束最小在反相关顶点）独立构造验证（$f=e_1,P=e_2$ 时 $\sum f_iP_i=0<1/N$）✓。
- aux-loss-free：V3 论文原文 "decrease the bias term by γ if … overloaded, and increase it by γ if … underloaded"、γ=0.001（前 14.3T tokens，余 500B 置 0）、seq-wise loss α=0.0001 "extremely small"、sigmoid 亲和、"bias only used for routing, gating value derived from original affinity score"（ar5iv:2412.19437）全部命中 ✓；官方 inference/model.py `Gate.forward` 逐行核对（sigmoid、bias 只加在 choice、`original_scores.gather`、topk(2).sum 组分数、×route_scale、bias 仅 dim==7168 时存在）✓。
- MTP：式 (19)(20) 与论文式 (21)(22) 逐符号一致（M_k∈R^{d×2d}、共享 Emb/OutHead、$P_{i+k+1}$）✓；接受率 85–90% 与 1.8× TPS（论文 5.4.3 节原文）✓；$\alpha=1-D_{TV}$ 与 $\mathbb E=(1-\alpha^{\gamma+1})/(1-\alpha)$ 独立复算 ✓。
- 实数账：V3 expert 3×7168×2048≈44.0M ✓、257×44.0M≈11.3B ✓、58 层≈656B ✓、激活 9×44.0M≈396M ✓、28.6× ✓；EP 通信 2×8×7168×2≈229KB ✓；Mixtral 46.7B/12.9B、Qwen3-30B-A3B 128 选 8、ρ 序列 25%→6.25%→3.5% ✓。

### B. arXiv ID 全量核对（67/67 命中）

正文 grep 出 67 个唯一 arXiv ID，分 7 批经 arXiv API（`export.arxiv.org/api/query?id_list=…&max_results=N`，https）核对 ID↔标题↔一作——全部一致，含三个「未来」ID：2512.02556 DeepSeek-V3.2（v1，2025-12-02）、2605.22791 Gated DeltaNet-2（v2，2026-05-21 发、2026-11-06 更）、2607.24653 Kimi K3（v2，2026-07-27 发）。抽查覆盖率 100%。

### B+. 波末时效另查

- Qwen3-Next 仍**无 arXiv 论文**（arXiv 全文搜索 5 条均为他文引用）→ [W] 评级继续成立 ✓。
- DeepSeek-V3.2-Exp repo 存在（api.github.com 200，2025-09-29 建）；README 2025-11-17 RoPE non-interleaved/interleaved 勘误公告原文命中（与 4.2 §6 所引一致）✓。
- Kimi Linear v2、Kimi K3 v2、GDN-2 v2 均为当前最新版；正文引用的数字在最新版原文中仍在 ✓。
- spec 中 W2 全部 ⟨verify⟩ 项（GDN-2、MiniMax-01、Jamba、Qwen3-Next、Kimi K3、Qwen2.5-1M、MoBA、differential-transformer、LLaDA、SEDD、Gemma Scope 等）本轮均已一手核实，见下表。

### C. HF config 逐字段实测（hf-mirror，2026-11-12）

| config | 核对结果 |
| --- | --- |
| answerdotai/ModernBERT-base | 22 层 / 768 / 12 heads / intermediate 1152 / vocab 50368 / max_pos 8192 / global_attn_every_n_layers=3 / local_attention=128 / global_rope_theta=1.6e5 / local_rope_theta=1e4 / tie=true / attention_bias=false / position_embedding_type="absolute"（残留字段断言成立）——全命中 ✓ |
| answerdotai/ModernBERT-large | 28 层 / 1024 / 16 heads / intermediate 2624，余同 base ✓ |
| Qwen/Qwen3-Next-80B-A3B-Instruct | 48 层 / full_attention_interval=4 / 512 experts top-10 / shared_expert_intermediate_size=512 / hidden 2048 / head_dim 256 / 16Q+2KV / partial_rotary_factor=0.25 / linear 32v×128×128、16k×128、conv 4 / rope_theta=1e7 / max_pos 262144 / vocab 151936——全命中 ✓ |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 27 层；linear_attn_config.kda_layers（20 个）/full_attn_layers=[4,8,12,16,20,24,27]；mla_use_nope=true；kv_lora_rank=512/qk_rope 64/qk_nope 128/v 128；num_experts 256、num_experts_per_token=8、num_shared_experts=1、sigmoid router、routed_scaling_factor=2.446、first_k_dense_replace=1；hidden 2304、head_dim 72、intermediate 9216、moe_intermediate 1024、rope_theta 1e4、model_max_length=1048576、vocab 163840、tie=false——全命中 ✓ |
| MiniMaxAI/MiniMax-Text-01 | 80 层；attn_type_list = 70 linear + 10 full（full 位于 1-indexed 8,16,…,80 ⇒ 7:1）；hidden 6144 / 64Q+8KV / head_dim 128 / rotary_dim=64 / rope_theta 1e7 / max_pos 10240000 / vocab 200064 / num_local_experts=32 / num_experts_per_tok=2 / postnorm=true / layernorm_*_alpha≈3.5566、beta=1.0——全命中 ✓；MiniMaxAI/MiniMax-01 不存在（404）✓ |
| deepseek-ai/DeepSeek-V3 | 61 层 / hidden 7168 / 128 heads / kv_lora_rank=512 / qk_rope 64 / qk_nope 128 / v 128 / q_lora 1536 / max_pos 163840 / first_k_dense_replace=3 / n_routed_experts=256 / num_experts_per_tok=8 / routed_scaling_factor=2.5 / n_group=8 / topk_group=4 / n_shared_experts=1 / moe_intermediate 2048 / scoring_func=sigmoid / rope_scaling={type:yarn, factor:40, beta_fast:32, beta_slow:1, mscale:1.0, mscale_all_dim:1.0, original:4096}——全命中 ✓（4.6 §2.3「9 通道全压/11 不动/12 通道 ramp」计数独立复算：λ_j=2π·10^{j/8}，j≥23 九条、j≤10 十一条 ✓；√(1/t)=0.1 ln40+1=1.369 ⇒ ×1.87 ✓） |
| mistralai/Mistral-7B-v0.1 | sliding_window=4096 / 32 层 / 8 KV heads / d_head=128 ✓（4.2 §2 账：R=32×4095+1=131041 ✓；cache 512 MiB ✓；2w/T=25%@32K、6.25%@131K ✓） |

### D. ar5iv 全文取证（20 篇）与 GitHub 文件

- DeepSeek-V3.2（2512.02556）：indexer 打分式、"ReLU as the activation function for throughput consideration"、FP8、K=2048、warm-up 冻结主模型+跨头求和+L1 归一化+KL、sparse 阶段 lr 7.3e-6 / 15000 步 / 480×128K = 943.7B token、V3.1-Terminus 续训、O(L²)→O(Lk)——全部与 4.2 §6 一致 ✓。4.2 §6 账：61×2048×576×2 B ≈ 144 MB、全量 9.2 GB、1.5625% 复算 ✓。
- Kimi Linear（2510.26692）：摘要 "for the first time, outperforms full attention under fair comparisons"、"up to 75%" cache、"up to 6×"（摘要）vs 6.3×（Fig 1(b) TPOT 1.84ms vs 11.48ms @1M）、1.4T 公平对照 vs 5.7T（Appendix D、对齐 Moonlight）、"uniform 3:1 ratio"、NoPE 节原文（"delegates the entire responsibility for encoding positional information … to the KDA layers"）✓。4.4 的 −75%=1−7/27 与 6.3× 归属无误 ✓。
- GDN-2（2605.22791）：见 A2 ✓。
- Kimi K3（2607.24653）：摘要 "2.8T … 104 billion activated … native vision … 1-million-token"、KDA + Attention Residuals + Stable LatentMoE 16/896 ✓（4.4 §5 一致；激活率 104/2800≈3.7% ✓、16/896≈1.8% ✓）。
- DeepSeek-V3（2412.19437）：见 A3；另核 Table 1（2664K+119K+5K=2788K）与 4.6 §4「扩窗 119K H800 GPU-hours」✓。
- DeepSeek-V2（2405.04434）：见 A1（absorption 与 MLA>MHA 原文）✓；group-limited 路由 V2 出处 ✓。
- GLM（2103.10360）：Poisson λ=3、"until at least 15%"、2D 位置（Part A p2=0、Part B 空内偏移）、document-level 50%–100% 单 span ✓。3.1 §4.3 的 p1/p2 实例表与论文 Figure 2 约定一致 ✓。
- UL2（2205.05131）："a mixture of 7 denoisers"（R: μ=3/8 r=0.15；S: μ=L/4 r=0.25；X: μ=3/8/64×2 r=0.5/0.15）✓；R/S/X 三族与 mode-switch token ✓。
- RoBERTa（1907.11692）：Table（with NSP）segment-pair 90.4 / sentence-pair 88.7，（without NSP）full-sentences 90.4 / doc-sentences 90.6——3.2 §4 的转述（账面提升主要来自输入格式）与数据逐格一致 ✓。
- ELECTRA（2003.10555）：λ=50（Appendix A 原文）、Table 6 base 口径 D hidden 768、generator 1/3×=256、共享 embedding ✓。
- FlashAttention（2205.14135）：A100 HBM 40–80GB / 1.5–2.0 TB/s、192KB SRAM/SM、~19 TB/s ✓；外循环 K/V 内循环 Q（Fig 1）✓；B_c=⌈M/4d⌉ ✓。
- GQA（2305.13245）：5% uptraining、mean pooling 三臂最优（Fig 4）、Fig 6 速度与「8 groups middle ground」✓（另见 F8）。
- NSA（2502.11089）：l=32、d=16、l'=64、n=16（含 1 初始块+2 局部块）、w=512 ✓；4.2 §5 的 9.7K/7.4% 复算 ✓。
- StreamingLLM（2309.17453）："four initial tokens generally suffices"、4M token 稳定、专职 sink token 训练侧替代 ✓。
- YaRN（2309.00071）：ramp 式 (18) 与温度式 (22) √(1/t)=0.1 ln s+1（"found by fitting"）✓。
- Qwen2.5-1M（2501.15383）：DCA（An et al. 2024a 出处标注一致）+ "we always use attention scaling in YaRN together with DCA" ✓；训练 256K → 推理 1M 免训外推 ✓。
- Llama-3 herd（2407.21783）："six stages, starting from the original 8K … ending in the final 128K … approximately 800B training tokens"（405B 口径）✓。
- Gemma 3（2503.19786）：5:1 local:global、local span 1024、global RoPE base 10k→1M、local 10k ✓。
- Gemma Scope（2408.05147）：JumpReLU、宽度 2^14≈16.4K 至 2^19≈524K 主套件 + 1M 档、每层三挂载点（attention 输出 / MLP 输出 / post-MLP residual stream）✓。
- Jamba（2403.19887）：4 个 Jamba block × l=8 层、a:m=1:7、MoE 每 e=2 层、n=16 top-2、52B/12B、256K ✓。
- MiniMax-01（2501.08313）摘要：456B/45.9B、32 experts、"1 million tokens during training and extrapolate to 4 million tokens during inference" ✓。
- IOI（2211.00593）摘要：26 heads / 7 类、GPT-2 small ✓。LLaDA（2502.09992）摘要：8B 从头训练掩码扩散 ✓。
- GitHub：30 个 repo 探测全部 200（含 THUDM/GLM、NVlabs/GatedDeltaNet(-2)、fla-org/flash-linear-attention、MoonshotAI/Kimi-Linear 与 Kimi-K3、MiniMax-AI/MiniMax-01、ML-GSAI/LLaDA、decoderesearch/SAELens、NVIDIA/RULER 等）；deepseek-ai/NSA 与 zai-org/GLM 按本书断言为 404 ✓；hsiehjackson/RULER 301 → 新址（references 用 NVIDIA/RULER ✓）。文件路径：DeepSeek-V3/inference/model.py、modeling_mixtral.py（MixtralSparseMoeBlock L113）、modeling_llama.py、modeling_modernbert.py（`sliding_window = local_attention // 2` 注释行）、modeling_qwen3_next.py（zero-centered norm 的 torch.zeros 初始化与 `output*(1.0+weight)`、gate 从 q_proj 切出并 `attn_output*sigmoid(gate)`、`use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule","fla")`）、microsoft/unilm/Diff-Transformer 目录——全部存在且关键行命中 ✓。
- BERT `create_pretraining_data.py` 原文核对：`num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))` ✓、[CLS]/[SEP] 排除 ✓、洗牌 ✓、WWM 的 `##` 归组与同组内逐 token 独立 80/10/10 ✓（分支次序差异见 F9）。
- T5 `preprocessors.py` 原文核对：num_noise_tokens=round(T·r) 并 clip 到 [1,T−1] ✓、num_noise_spans=max(1,round(m/μ)) ✓、`_random_segmentation` 均匀切分 ✓、交替拼接 ✓——3.1 式 (13)(14) 与教学复刻一致；T=512 账 m=77、n_s=26、enc 461、target 104 复算 ✓；几何极限比 (m−k−n_s+2)/(m−k)→1−1/μ ✓。

### E. 跨章引用与约定一致性抽查

- 2.4 式 (5)(8)(9)(11)(12)、2.5 式 (13)(16)(17)、2.3 式 (4)(10)(12)、1.3 式 (8)、1.1 式 (8) 均按 W2 各章引用语义逐一打开原章核对 ✓。
- W2 数学约定页（w2-sources.md）与三章正文逐项对照：状态布局 $S\in\mathbb{R}^{d_v\times d_k}$、右乘擦除、KDA 的 $\alpha_t\in\mathbb{R}^{d_k}$（key 侧）、GDN-2 转置式——4.3/4.4 全部一致 ✓；HF kernel 转置布局注记存在且正确 ✓。
- 3.3 §2 的 2.4 账（naive (2/3)LT³d vs cache 2LT²d、T/3≈4.4×10^4 @131K）复算 ✓；3.2 §8 参数账（12×12d²+12×16d²=336d²、decode 0.57×、cross cache 1.8e7 B=18 MiB）✓；ALBERT 账（Vd 23.4M→Ve+ed 4.0M、层共享 84.9M→7.1M、总 110M→~12M）✓。
- 4.4 §7 内存统一账逐项复算：Qwen3-Next 固定 36 MiB / 斜率 24 KiB → 262144 处 6.0 GiB、全 full 24.0 GiB ✓；Kimi Linear 20 MiB / 7.9 KiB → 1M 处 7.87 GiB、全 MLA 30.4 GiB ✓；MiniMax 40 KiB → 4M 处 160 GiB、全 full 1.25 TiB ✓；交叉点 910 token / 2600 token ✓；decode FLOPs 2.1M vs 4.3G ≈ 两千倍 ✓。4.6 §1.1（2.6×10^17 FLOPs、T*≈6.1×10^4、16×、260 s、128 TB 物化）✓；§1.2 KDA hybrid 7.5 GiB + 20 MiB ✓（GQA 行见 F4）。
- 4.1 §5.2 online softmax 数值走查（m、ℓ 两行与直接基准 1.0706）逐步复算一致 ✓；B_c≈390、Q 重读 42 趟 ✓；FA/naive IO 比 d²/M≈1/25 ✓。
- 教学审查：11 章每节均以事实/公式/问题开场；公式均带符号表；前置声明与 README 依赖图（P1→P2→P3→P4）一致；唯一前向引用（3.2 §10 引 4.2 的 SWA 感受野）为旁注性质，不构成功能依赖。
- 规范审查：metadata 块 / 总结（表格或公式清单）/ 练习（2–4 题，分推导/计算/代码型）/ 参考 四件齐全 ✓；每章 ≥1 张 ASCII 图 ✓；无历史叙事段（仅 3.2 §10「2018 年的零件」为版本指称，非时间线叙事）✓；证据等级标注齐全且与 references.md 一致（MoBA 一处见 F7）；[B]/[W] 条目的降级措辞（NSA、UL2、Qwen3-Next、NoPE 等）到位 ✓。

## 三、一句话总结

11 章的数学主线（MLA / delta-rule 族 / MoE 路由）、全部 67 个 arXiv ID、6 个 HF config、30 个 repo 与 20 篇全文级取证均经一手来源复核通过；未发现 S1；两处 S2（ModernBERT 2:1 被误写为 3:1 且跨章传播；README 进度表未随波更新）需修复，九处 S3 记入下波清零清单。

## 四、放行结论

**放行**（S1=0）。条件：F1、F2 修复后无需全章复审（小编辑）；F3–F11 按 S3 流程累计、下一波复审统一清零。W2 的三项强制重推导全部 PASS，波末时效性锚点（Qwen3-Next / Kimi K3 / Kimi Linear / DeepSeek-V3.2 / GDN-2）在 2026-11-12 口径下仍准确。
