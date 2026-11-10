# W1 波次独立对抗审查报告（models 维度：1.1–1.3 + 2.1–2.8，共 11 章）

- 审查日期：2026-11-11。审查人：W1 独立 evaluator（fresh context，不继承写作上下文）。
- 立场：假设内容有错，直到一手来源证明无误。所有数值/引用抽查均由本 evaluator 用 curl 自行 fetch 一手来源（arXiv API、ar5iv 全文、hf-mirror config、GitHub API/contents），或独立重算/重推导/数值模拟。
- 机械检查：`bash scripts/lint-chapter.sh` 对 11 章全部 LINT OK（禁令词、模板小节、metadata、arXiv ID 回链、证据等级、`$$` 配平）。

## Findings 表

| # | severity | 位置 | 断言（正文写法） | 证据（一手来源） | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S1** | 2.4 §5 cache 实例表（`04-causality-and-kv-cache.md`「OLMo-2-7B（MHA 锚点）」行） | OLMo-2-7B 满窗（T=4096）cache 总量 **4 GiB** | 按同章式 (9) 逐项乘：2(K,V)×32(L)×32(H_kv)×128(d_head)×4096(T)×1(B)×2(bf16) B = 2^31 B = **2 GiB**；且同行自报的每 token 增量 512 KiB × 4096 = 2 GiB，行内自相矛盾。OLMo-2 config（hf-mirror 实测：H_kv=32、L=32、max_position_embeddings=4096）无歧义 | 把 4 GiB 改为 **2 GiB** |
| F2 | **S1** | 1.2 §1 与 §9（`02-compute-accounting.md`）；同误亦写入 `docs/research/w1-sources.md` 第 20 行 | 「预训练耗 2.788M H800 GPU-hours——这个口径**只含 pre-training 本身**，不含前期研究与 post-training」「paper Table 1 的 pre-training 口径」 | DeepSeek-V3 论文（ar5iv:2412.19437）Table 1 原文行：Pre-Training **2664K** + Context Extension **119K** + Post-Training **5K** = Total **2788K** H800 GPU hours；正文：「Combined with 119K GPU hours for the context length extension and 5K GPU hours for post-training, DeepSeek-V3 costs only 2.788M GPU hours for its **full training**」。2.788M 是**全训练总量**（仍不含前期研究，该半句对），纯 pre-training 是 2.664M。另外 1.2 §1「Table 1 给出四个数：671B / 37B / 14.8T / 2.788M」不准：Table 1 只有成本行，671B/37B/14.8T 出自摘要与架构节 | 口径改为「2.788M = pre-training(2.664M) + 上下文扩展(119K) + post-training(5K) 的全训练口径；纯 pre-training 为 2.664M」。§9 对账改用 2.664M 小时：每卡持续 3.29e24/(2.664e6×3600) ≈ **3.4e14** FLOPS（现值 3.3e14 偏低 4.5%，第 5 节集群算例与 2.4 §3「7 天」换算沿用同一派生量，需同步）；w1-sources.md 同步更正 |
| F3 | **S2** | 2.7 §7（`07-norm-and-residual.md`） | 「QK-Norm（Henry et al.）……**原文用 LayerNorm**；OLMo 2 用 RMSNorm」；式 (15) 保留 1/√d_k 缩放 | Henry et al. 2020（ar5iv:2010.04245）原文：「we apply **ℓ2 normalization** along the head dimension of each query and key matrix prior to multiplying them and then **scale up by a learnable parameter instead of dividing by the square root** of the embedding dimension」，即式 (5) `softmax(g·Q̂K̂ᵀ)V`——原文是**逐 head ℓ2 归一化 + 可学标量 g 替换 1/√d_k**，不是 LayerNorm | 改为「原文用逐 head ℓ2 归一化并以可学标量 g 替换 1/√d_k；OLMo 2 / Gemma 3 的现代实践换成 RMSNorm 且保留 1/√d_k」。式 (15) 标注为现代变体而非原文形式 |
| F4 | **S2** | 1.3 §5 实账表（`03-scaling-laws.md`） | Llama-3.1-8B「推理每 token 1.6e10」（= 2×8.03e9，把 embedding 参数也算进 FLOPs） | 与 1.2 式 (2) 的口径冲突：embedding 查表是 gather 无乘加，前向 = 2N_full = **1.50e10**（1.2 §9 逐项加出并反复强调该口径）；同表 35B 行用的却是 2×N_opt（非 embedding 口径 6.9e10），两行口径不一 | 8B 行改 1.5e10（或两行统一口径并注明）。Q* 随之从 3e13 微调至 ≈2.9e13，结论不变 |
| F5 | S3 | 1.1 §7 符号表、1.2 §3、2.3 §1、2.4 §1、2.5 §3.2/§4.3/§5 | 「注意力权重」「均匀注意力下」等把 attention 译成中文 | STYLE_GUIDE §1 把 attention 列入「保持英文」清单；2.3 §1 虽以「注意力权重（attention weights）」格式首现，但保护清单本意是该术语不译 | 全书统一裁决：要么白名单例外登记「注意力权重」，要么回改为 attention weights |
| F6 | S3 | 2.2 §7 三条 | `$V \uparrow \Rightarrow T \downarrow$` 等使用 `\uparrow` / `\downarrow` | STYLE_GUIDE §3.1 白名单（已锁定）未收 `\uparrow`/`\downarrow`（箭头类只收 \to/\Rightarrow 等）。MathJax [M2] 虽支持，但锁定的白名单是唯一标准 | 改用文字（升/降）或待 math-render-probe 复核后补入白名单 |
| F7 | S3 | 2.8 §3 末对照表 | 「nanoGPT / GPT-2：FFN 两矩阵 GELU，d_ff=4d（**gelu_new = tanh 近似**）」 | nanoGPT `model.py`（GitHub contents API 实测）：`self.gelu = nn.GELU()`——nanoGPT 用的是精确 erf GELU；gelu_new（tanh 近似）只是 GPT-2 的 HF config 字段（openai-community/gpt2 config 实测 activation_function=gelu_new） | 该行括号改为「GPT-2 为 gelu_new = tanh 近似；nanoGPT 用精确 GELU」 |
| F8 | S3 | 1.3 §4 | 「最优模型只有其**约四分之一**大——……区间即 40B–67B」 | Chinchilla 论文（ar5iv:2203.15556）「4 times smaller」对应 70B/67B 端（280/67≈4.2）；区间下端 40B 是 280 的 1/7 | 改为「约四分之一到七分之一（40B–67B）」 |
| F9 | S3 | 2.1 §3 shape 表、2.2 式 (8) | 2.1 步 6/7 用 $P$（softmax 后概率矩阵）早于 2.3 定义；2.2 式 (8) `o_t^T g_t^T` 的形状注记 `[V,1]×[1,d]` 要求 g_t 取列向量约定，与 1.2 式 (3) 的行向量约定相反 | 章内均可读通（各有局部注记），但跨章约定不自洽 | 2.1 步 6 加注「$P$ 的正式定义见 2.3 式 (4)」；统一梯度向量行列约定 |

## 教学审查结论

- 推导断步：未发现断步级问题。1.3 的 Lagrangian 链、2.3 的 √d_k 四段链、2.4 的可缓存性归纳、2.5 的 RoPE 引理链、2.7 的 LN Jacobian 引理均为逐步可验（本人逐步重推，见 verification log A）。1.1 §4 的 softmax-CE 复合导数「q−δ」只声明未推导，但标注为「入口事实」且在 2.2 §9 复用前有足够铺垫，可接受。
- 未定义符号：除 F9 的两处 S3 外未发现。
- 循环解释：未发现。
- 前置知识：各章 metadata 前置声明与 models/README.md 依赖图一致（P1→P2 顺序成立；对 3.x/4.x/5.x 的引用全部是前瞻指针而非依赖）。1.3 用 Lagrange 乘子法、2.7 用谱范数，均在 README「线性代数/概率论」射程内。

## 规范审查结论（STYLE_GUIDE 逐条）

- §1 语言：除 F5（attention 译文）外合规；术语表登记与正文一致（抽查 RoPE/ALiBi/SwiGLU/RMSNorm 等）。
- §2 禁历史叙事：全 11 章未发现时间线叙事；「活化石」「骨架七年未动」等为结构对照措辞，合规。章末总结全部为表格，无空洞评论段。
- §3 公式：除 F6（\uparrow/\downarrow）外全部命中白名单（全章命令清单已枚举核对：\mathrm/\mathbb/\mathcal/\operatorname/aligned/cases/bmatrix/underbrace 等均在内）；公式符号表齐全；shape 记法 [B,T,H,D] 一致。
- §4 代码：教学片段均标「教学用，非优化实现」；官方实现引用到 repo+文件+函数/类名，并逐一实测核对（见 log D）。
- §6 引用与证据分级：正文引用格式统一链到 references.md；证据等级标注逐章存在（lint 项⑤）；[W]/[B] 级的谨慎措辞到位（如 WordPiece 公式的社区重述标注、Gemma 数值挂论文口径、Llama rms_norm_eps 明示未独立核实）。**例外即 F2**（数值口径标注错误）。
- §7 锚点：每章 metadata 块含锚点模型+论文+证据等级，锚点以 2024-2026 模型为主（Llama-3.1/OLMo-2/Qwen3/DeepSeek-V3），老经典只作公式源头，合规。
- §8 模板：11 章均有 总结/练习/参考 三节，练习 2-4 题且分推导/计算/代码型，每章至少 1 张 ASCII 图。篇幅均在标准章区间内（推导重章 2.3/2.5/2.7 更长）。
- §9 去 AI 味：禁令词零命中（lint 项① + 人工复核），无排比堆砌。

---

## Verification log

### A. 强制独立重推导清单（W1 子集，含关键中间步）

**A1. 1.3 Chinchilla compute-optimal Lagrangian — PASS（与论文 eq (4) 形式一致）**
- 重推：∂/∂N: −αA·N^(−α−1) + 6λD = 0；∂/∂D: −βB·D^(−β−1) + 6λN = 0。两式相除消 6λ：αA·N^(−α−1)·N = βB·D^(−β−1)·D ⟹ αA·N^(−α) = βB·D^(−β) ⟹ D = (βB/αA)^{1/β}·N^{α/β}。代入 6ND=C：N^{(α+β)/β} = (C/6)(αA/βB)^{1/β} ⟹ **N_opt = G(C/6)^a，a = β/(α+β)，G = (αA/βB)^{1/(α+β)}**；对称得 D_opt = G^{−1}(C/6)^b，b = α/(α+β)。a+b=1 使约束自闭合 ✓。
- 最小性：g(N)=L(N,C/6N)，g' = N^{−α−1}[−αA + βB(6/C)^β·N^{α+β}]，括号严格增、单零点、由负变正 ⟹ 唯一全局最小；β<1 时数据项二阶导为负、g 非全局凸（章内该注记正确）✓。
- 数值：a = 0.28/0.62 = 0.4516 ✓；G = (138.176/114.996)^{1/0.62} = 1.20157^1.6129 = e^{0.1836×1.6129} = e^{0.296} = 1.345 ≈ 1.34 ✓；L*−E 指数 αβ/(α+β) = 0.0952/0.62 = 0.1535 ≈ 0.154 ✓；10^{−0.154} = 0.70 ✓。
- 一手来源对照：ar5iv:2203.15556 原文 eq (4)（「a=β/(α+β), b=α/(α+β)」与 G 的形式逐字符一致）、Eq. (10) 常数 E=1.69/A=406.4/α=0.34/B=410.7/β=0.28、Huber δ=10^{−3} 拟合 log L、Table 2 三法 a∈[0.46,0.50]/b∈[0.50,0.54]、Table 3 Gopher 预算行 67B/1.5T、§3.3 正文「we project the optimal model size given the Gopher FLOP budget to be 40B」、§4「70B × 1.4T tokens」。全部命中 ✓。

**A2. 2.3 attention √d_k — PASS**
- 重推：A1（q,k 分量独立零均值单位方差）下 E[s]=ΣE[q_i]E[k_i]=0；E[s²] 的双重求和中 i≠j 项因四因子独立零均值归零，只剩 d_k 个对角项各为 1 ⟹ Var(s)=d_k ✓。softmax Jacobian：∂p_i/∂z_j = (δ_ij e^{z_i}S − e^{z_i}e^{z_j})/S² = p_i(δ_ij − p_j) ✓。梯度：∂L/∂z_j = Σ_i g_i p_i(δ_ij − p_j) = p_j(g_j − ḡ)，ḡ = Σp_ig_i ✓。饱和界：|∂L/∂z_j| ≤ 2p_j‖g‖_∞，p_m = 1−ε 时最大项 ≤ 2ε(1−ε)‖g‖_∞ ⟹ 全行 ≤ 2ε‖g‖_∞ ✓。幂次唯一性：z/d_k^α 的方差 = d_k^{1−2α}，仅 α=1/2 与 d_k 无关 ✓。
- 数值复核（本人 Monte Carlo，T=4096、d_k=128、2000 行 N(0,128) 采样）：行内最大 score 中位 40.5 nat（章：≈40 ✓）；p_max 中位 0.840（章：0.84 ✓）；非最大质量中位 16%（章：15–20% ✓）；权重 <1e−15 的 key 占比中位 70.2%（章：约七成 ✓）；恒定间距近似 p_max ≈ 0.94（r = e^{−2.8} = 0.061，r/(1−r) = 6.5% ✓）。缩放后章内估计 p_max ≈ 0.009（plug-in 粗估；本人模拟中位 0.0053——同量级，章内结论「接近均匀起步」成立，粗估性质已标明，不报 finding）。

**A3. 2.4 KV cache — PASS（公式与推导正确；实例表一处 S1 数值错，即 F1）**
- 可缓存性：对层归纳——x_s^(ℓ) 只依赖 (x_1..x_s)：基例 embedding 查表 ✓；归纳步用式 (7) 求和上限 s（causal）+ 逐位置算子 ✓；换成 M≡0 归纳即断 ✓。「列被未来读、行只被自己读」的 Q 不缓存论证：q_s 只出现在 score 第 s 行（j≤s），该行仅第 s 步被消费 ✓。
- cache size：每层 K、V 各 [B,H_kv,T,D] 元素 ⟹ bytes = 2·L·H_kv·d_head·T·B·b ✓。实例重算：Llama-3.1-8B 2×32×8×128×131072×2 = 2^34 B = 16 GiB ✓；70B（L=80）= 40 GiB ✓（unsloth mirror config：L=80、H_kv=8 ✓）；8B-MHA 反事实 64 GiB ✓；MQA 2 GiB ✓；**OLMo-2-7B 行 4 GiB ✗（正确值 2 GiB，F1）**。
- FLOPs 恒等：朴素 decode Σ_t(2Nt+2Lt²d) = NT(T+1) + (Ld/3)T(T+1)(2T+1) ✓（Σt²=T(T+1)(2T+1)/6）；有 cache = 2NT+2Ld·T(T+1) ≈ 2NT+2LT²d ✓，与 teacher-forced 并行前向逐项相同 ✓；比值 T/3 = 4.4e4 @131072 ✓。
- 算术强度：decode 满窗分子 1.50e10+4Ltd(=4×32×131072×4096=6.87e10)=8.4e10 ✓；分母 16 GB+16 GiB=3.3e10 ✓；I≈2.5 ✓；batch 饱和上界 8.4e10/1.72e10≈4.9 ✓。

**A4. 2.5 RoPE — PASS**
- 引理 (i)：R(−θ) = [cosθ, sinθ; −sinθ, cosθ] = R(θ)^T ✓；R^TR=I（cos²+sin²=1，非对角相消）✓。引理 (ii)：乘积左上角 cosαcosβ−sinαsinβ=cos(α+β) 等，归两角和 ✓。引理 (iii)：⟨R(a)q, R(b)k⟩ = q^T R(−a)R(b) k = q^T R(b−a)k ✓——只含 b−a。
- 块对角主命题：⟨R_m q, R_n k⟩ = q^T R_m^T R_n k = q^T R_{n−m} k；逐 2D 块代入 q^TR(φ)k = (q_1k_1+q_2k_2)cosφ + (q_2k_1−q_1k_2)sinφ 得式 (13) ✓。复数形式：q̃_j k̃_j^* = A_j+iB_j，Re[(A_j+iB_j)e^{i(m−n)θ_j}] = A_j cos((n−m)θ_j) + B_j sin((n−m)θ_j)，逐项等于式 (13) 第 j 块 ✓。
- 数值例 q=(1,2),k=(3,1),a=1.5,b=3.5：本人重算左侧 ⟨(−1.924,1.139),(−2.459,−1.989)⟩ = 2.466；右侧 (1,2)·(−2.158,2.312) = 2.466；⟨q,k⟩=5 ✓（章内 2.47 ✓）。
- 波长账：λ_max = 2π·base^{1−2/d}：Llama/OLMo（base 5e5, d=128）2π·e^{13.122×0.984} = 2.56e6 ≈ 2.6e6 ✓；Qwen3（1e6）5.06e6 ≈ 5.1e6 ✓；DeepSeek-V3（base 1e4，旋 64 维 → 32 通道）2π·10^{3.875} = 4.71e4 ✓。周期覆盖：base=1e4、W=4096：λ_j>4096 ⟺ j≥46（2π·10^{j/16}>4096 ⟺ j>45.03）⟹ 18/64 = 28% ✓；θ_63 = 10^{−3.9375} = 1.15e−4，弧长 0.47 rad ≈ 27° ✓；OLMo（base 5e5, W=4096）j≥32 即一半 ✓；Llama θ_63 ≈ 2.45e−6、弧长 0.32 rad ✓。
- 一手来源：ar5iv:2104.09864 §3.4.3 Abel 变换上界 ✓、θ_i = 10000^{−2i/d} ✓。

**A5. 2.7 Pre/Post-LN（Xiong Theorem 1）— PASS（定理表述逐字命中原文；同章另有 QK-Norm 归属错，即 F3）**
- 原文逐字对照（ar5iv:2002.04745）：「Theorem 1 (Gradients of the last layer in the Transformer). Assume that ‖x^{post,5}_{L,i}‖² and ‖x^{pre}_{L+1,i}‖² are (ε,δ)-bounded for all i… with probability at least 0.99−δ−ε/(0.9+ε), for the Post-LN Transformer with L layers, ‖∂L̃/∂W^{2,L}‖_F ≤ O(d√(ln d))… for the Pre-LN… ≤ O(d√(ln d/L))」——章内式 (13) 与概率界、有界性假设逐字一致 ✓；δ = exp(−nε²/8)（原文 eq (108)）✓；扩展分析方向「the gradient norm in the Post-LN Transformer is large for the parameters near the output and will be likely to decay as the layer index l decreases」——章内「近输出端大、向输入端衰减」✓，方向正确（非浅层爆炸）。
- LN Jacobian 引理本人重推：ŷ 视角 y = Px，J = (1/σ)(I − x̂x̂^T/d)P = (1/σ)(P − x̂x̂^T/d)（用 Px̂=x̂）✓；J·1_d=0、J·x̂=0（x̂^Tx̂=d）、正交补空间上 = (1/σ)I ⟹ rank d−2、‖J‖=1/σ=√d/‖x−μ‖ ✓ 与式 (10)(11) 一致。
- §5.2 范数行为与式 (12) Θ(Ld) 随机游走口径一致 ✓；Xiong 原文 Lemma 2（Post-LN E‖x‖²=3d/2）佐证 Post-LN 端 Θ(√d) ✓。比值 1/√L：L=32 → 5.66≈5.7 ✓，L=128 → 11.3 ✓。
- DeepNorm（ar5iv:2203.00555）：TL;DR 表 decoder-only α=(2M)^{1/4}、β=(8M)^{−1/4} ✓；encoder-decoder decoder 侧 (3M)^{1/4}/(12M)^{−1/4} ✓；更新界条件「2N(v²+w²)/α²=1」（原文 eq (17) 前后）✓，代入 v=w=β 得 α²=4Lβ²、α/β=2√L ✓；αβ=(1/4)^{1/4}=1/√2 ✓；L=1000 → α≈6.69、β≈0.106 ✓。

### B. arXiv ID ↔ 标题全量核对（32/32 一致）

方法：`curl -s "https://export.arxiv.org/api/query?id_list=…"`（https，一次查 32 个）。正文+references.md 中 W1 引用的全部 arXiv ID：2001.08361=Kaplan ✓、2005.14165=GPT-3 ✓、2203.15556=Chinchilla ✓、2206.07682=Wei emergence ✓、2304.15004=Schaeffer ✓、2401.06066=DeepSeekMoE ✓、2412.19437=DeepSeek-V3 ✓、2304.01373=Pythia ✓、2407.21783=Llama 3 ✓、1706.03762=Transformer ✓、2501.00656=OLMo 2 ✓、1508.07909=BPE/Sennrich ✓、1804.10959=Kudo Unigram ✓、1609.08144=GNMT/WordPiece ✓、1608.05859=Press&Wolf ✓、1711.03953=softmax bottleneck ✓、2412.09871=BLT ✓、1911.02150=MQA ✓、2104.09864=RoFormer ✓、2108.12409=ALiBi ✓、1910.10683=T5 ✓、2305.19466=NoPE ✓、2002.05202=GLU variants ✓、1606.08415=GELU ✓、2012.14913=FFN-KV ✓、1607.06450=LayerNorm ✓、1910.07467=RMSNorm ✓、2002.04745=Xiong ✓、2203.00555=DeepNet ✓、2408.00118=Gemma 2 ✓、2503.19786=Gemma 3 ✓、2010.04245=QKNorm ✓。覆盖率：W1 正文出现的 arXiv ID 的 100%。

### C. HF config 抽查（hf-mirror 实测；Llama 官方 gated 仓库 401，用 unsloth 镜像仓 config 替代核实）

- Qwen/Qwen3-8B：vocab_size=151936、hidden=4096、L=36、heads 32/8、head_dim=128（显式字段 ✓ 2.8 误解一）、intermediate=12288、rope_theta=1e6、max_position_embeddings=40960、rms_norm_eps=1e−6、attention_bias=false、tie=false、initializer_range=0.02、bos=151643/eos=151645 —— 与 1.1/2.2/2.4/2.5/2.6/2.7/2.8 全部引用一致 ✓。tokenizer_config：Qwen2Tokenizer ✓、model_max_length=131072 ✓、added_tokens 恰 26 个、id 151643–151668、含 <|endoftext|>/<|im_start|>/<|im_end|>/<think>/</think> ✓。vocab.json 实测 151,643 条（max id 151642）✓。
- allenai/OLMo-2-1124-7B：vocab=100352、hidden=4096、L=32、heads 32/32（纯 MHA）✓、intermediate=11008 ✓、max_position_embeddings=4096 ✓、rope_theta=500000 ✓、rms_norm_eps=1e−6 ✓、attention_bias=false ✓、tie=false ✓。
- deepseek-ai/DeepSeek-V3：L=61 ✓、vocab=129280 ✓、qk_nope_head_dim=128 + qk_rope_head_dim=64（=192 ≠ hidden/heads ✓）、rope_theta=10000 ✓、rope_scaling={type:yarn, factor:40, beta_fast:32, beta_slow:1, mscale:1.0, mscale_all_dim:1.0, original_max_position_embeddings:4096} ✓、max_position_embeddings=163840（=4096×40 ✓）、n_routed_experts=256、n_shared_experts=1、num_experts_per_tok=8、first_k_dense_replace=3 ✓。
- openai-community/gpt2：vocab=50257 ✓、n_ctx=n_positions=1024 ✓、n_layer=12/n_head=12/n_embd=768 ✓、activation=gelu_new ✓、initializer_range=0.02 ✓、layer_norm_epsilon=1e−5 ✓、bos=eos=50256 ✓。
- Qwen/Qwen2.5-7B-Instruct-1M：rope_theta=1e7、max_position_embeddings=1010000、无 rope_scaling 字段 —— 2.5 §6 对照实例 ✓。
- meta-llama/Llama-3.1-8B / 70B：官方 gated（401）。unsloth/Meta-Llama-3.1-8B config：4096/32/32/8/14336/rope_theta 5e5/vocab 128256/max_pos 131072/tie=false/attention_bias=false/mlp_bias=false ✓；rms_norm_eps=1e−5（2.8 §2 诚实标注「未独立核实」，现由本审查补证：实际值 1e−5，可回填）。70B：hidden 8192/L=80/heads 64/8/intermediate 28672 ✓（2.4 §5 表 L=80、H_kv=8 命中）。

### D. repo 与代码走读核对（GitHub API）

- repo 存在性 5/5：huggingface/transformers ✓、karpathy/nanoGPT ✓、facebookresearch/blt ✓、zihangdai/mos ✓、google/sentencepiece ✓。
- `src/transformers/models/llama/modeling_llama.py`（contents API 拉取逐行核对）：`LlamaRMSNorm.forward` 五行与 2.8 §4 摘录逐字一致（fp32 归约 + rsqrt + weight）✓；`head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)` ✓；`num_key_value_groups = heads // kv_heads` ✓；`self.scaling = self.head_dim**-0.5` ✓；`repeat_kv`/`rotate_half`/`apply_rotary_pos_emb` 存在 ✓；`eager_attention_forward` 中 `nn.functional.softmax(..., dim=-1, dtype=torch.float32).to(query.dtype)` ✓；`LlamaDecoderLayer.forward` 的 residual→input_layernorm→attn→add→post_attention_layernorm→mlp→add 与 2.8 §4 摘录逐行一致 ✓；`create_causal_mask` 模型级调用 ✓；`LlamaRotaryEmbedding` 只存 inv_freq buffer、per-forward 由 position_ids 现算 ✓；`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` ✓。
- `modeling_olmo2.py`：`q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim)`（4096 全维变体 ✓ 2.7 §7）；q_norm/k_norm 在投影后、RoPE 前 ✓；`Olmo2MLP` 三投影硬编码 bias=False ✓（2.8 §8 误解二）；`Olmo2DecoderLayer.forward` = x + post_norm(attn(x)) 的 reordered 接线 ✓（2.1/2.7/2.8 三处声明）。
- `modeling_qwen3.py`：`Qwen3MLP` 三投影 bias=False 硬编码 ✓。
- nanoGPT `model.py`：`bias: bool = True`（注释「like GPT-2」）——2.8 §8「GPT-2 与 nanoGPT 全为 true」✓；Block 为 `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))` + ln_f ✓；**`self.gelu = nn.GELU()`（精确 GELU，非 tanh 近似）——F7**。

### E. 论文原文关键数值/论断抽查（ar5iv 全文 grep + 原文引句）

- Chinchilla（2203.15556）：Eq.(10) 常数 ✓、Table 2 区间 ✓、Table 3 67B ✓、§3.3 40B ✓、「over 400 language models ranging from 70M to over 16B」✓（1.3 §2「70M 到 16B、约 400 组」✓）。
- Kaplan（2001.08361）：α_N=0.076/N_c=8.8e13 ✓、α_D=0.095/D_c=5.4e13 ✓、α_Cmin=0.050/C_cmin=3.1e8 PF-days ✓、raw-C 律 α_C=0.057（Table 4）✓、N_opt∝C_min^0.73 且 D=B·S∝C^0.27 ✓（1.3 §1/§4 全部命中）。
- DeepSeek-V3（2412.19437）：摘要「671B total…37B activated」✓；14.8T tokens ✓；**Table 1：Pre-Training 2664K + Context Extension 119K + Post-Training 5K = 2788K——F2**。
- Xiong（2002.04745）：Theorem 1 逐字 ✓（见 A5）。
- DeepNorm（2203.00555）：常数与条件式 ✓（见 A5）。
- Llama 3（2407.21783）：「about 15T multilingual tokens」✓（1.3 §5 的 15T）；「improves compression rates…from 3.17 to 3.94 characters per token」✓（2.2 §7/§12；3.94/3.17 = +24% ✓）；128K vocab ✓。
- Gemma 2（2408.00118）：「soft_cap parameter to 50.0 for the self-attention layers and to 30.0 for the final layer」✓；Vocab 256128 ✓；Table 1 Pre-norm/Post-norm 均 yes ✓；全文 0 处 QK-norm（2.7「Gemma 2 无 QK-norm」✓）。
- Gemma 3（2503.19786）：「we replace the soft-capping of Gemma 2 with QK-norm」✓；「post-norm and pre-norm with RMSNorm」✓；词表正文为「262k entries」（tokenizer 节；Table 1 注「256k」，论文内部两种写法并存——2.2 写 262144 挂论文口径，方向正确但精确值实际出自 config，Gemma HF gated；记录为口径注记，不报 finding）。
- OLMo 2（2501.00656）：§3.3.2「Reordered norm and QK-norm」节名与「applying the layer normalization to the outputs of the MLP and attention blocks instead of the inputs」✓；「apply another normalization, also RMSNorm, to the queries and keys」✓。
- Schaeffer（2304.15004）：「>92% of emergent abilities appear under one of two metrics: Multiple Choice Grade and Exact String Match」✓；Token Edit Distance 换尺对照 ✓；vision autoencoder 人为制造涌现 ✓（1.3 §7）。
- Pythia（2304.01373）：「16 LLMs…70M to 12B…154 checkpoints for each one of the 16 models…public data seen in the exact same order」✓。
- DeepSeekMoE（2401.06066）：摘要「scale up DeepSeekMoE to 16B…comparable performance with LLaMA2 7B, with only about 40% of computations」✓；16.4B 总/2.8B 激活 ✓（1.3 §6）。
- BLT（2412.09871）：「first flop controlled scaling study of byte-level models up to 8B parameters and 4T training bytes」✓；entropy-driven patching ✓。
- ALiBi（2108.12409）：「for n heads, our set of slopes is the geometric sequence that starts at 2^{−8/n}…」⟹ m_h = 2^{−8h/H} ✓；train 1024 → extrapolate 2048 同 PPL ✓。
- QK-Norm（2010.04245）：「apply ℓ2 normalization along the head dimension…scale up by a learnable parameter instead of dividing by the square root」——**原文 = 逐 head ℓ2 归一化 + 可学 g 替换 1/√d_k，非 LayerNorm——F3**。
- T5（1910.10683）：「we use 32 embeddings…ranges increase logarithmically up to an offset of 128 beyond which…same embedding」✓（2.5 §3.1）；跨层共享 ✓。
- Vaswani（1706.03762）：Table 1 四行（self-attn O(n²d)/O(1)/O(1)；recurrent O(nd²)/O(n)/O(n)；conv O(knd²)/O(1)/O(log_k n)；restricted O(rnd)/O(1)/O(n/r)）✓；§3.2.1 缩放动机 ✓；§3.2.3「masking out (setting to −∞)」✓；§3.3 d_ff=2048 ✓；§3.4「share the same weight matrix…multiply those weights by √d_model」✓；h=8、d_k=64 ✓。
- NoPE（2305.19466）：摘要「NoPE outperforms other explicit positional encoding methods…on length generalization」——2.5 §5 的保守转述（「相当」「不落下风」）方向正确且有 [B] 限定 ✓。
- Wei（2206.07682）：两类分类 = §3 few-shot prompted + §4 augmented prompting strategies ✓。

### F. 数值重算清单（本人独立重乘，全部命中除非另报 finding）

1.1：输出头参数 128256×4096=525,336,576 ✓、151936×4096=622,329,856 ✓；0.1 nats=0.144 bits ✓；uniform 基线 PPL=V ✓；rank 上界 4098/151936=2.7%<3% ✓。
1.2：FFN 352.3M ✓、QKV 50.3M ✓、层合计 4.362e8×32=1.396e10=2N ✓；N=6,979,321,856 ✓、总 8,029,995,008 ✓；T*=5.3e4 ✓；attention 三档 6.7%/36%/70%（含自身口径）✓；激活 256KB/层/token@4096 → 8MB/token → B=8 时 256GiB ✓；T=131072 时 138MB/token、P 占 97% ✓；KV cache 16 GiB ✓；优化器态 128GB/80GB 表 ✓；DeepSeek 3.29e24 ✓、1474 tok/s/GPU ✓（分母口径问题见 F2）；MoE 84×/4.9× ✓；1024 卡 30 天 8.8e23 ✓。
1.3：a/b/G/0.154 ✓（A1）；L(8B,15T)=1.948 ✓、L(35B,3.3T)≈1.925 ✓、8B@48T ✓、Q*≈3e13 ✓（F4 的口径注记除外）；100^0.73≈29 ✓；0.8^8=0.168/0.95^8=0.663 ✓；过渡宽 1.34/k dex ✓；α_D/(α_N+α_D)=0.56 ✓（练习 1 的命题本人验证成立）。
2.1：12Ld²=6.44e9 ✓、13Ld²=6.98e9 ✓、OLMo 12.06d²/层→7.30e9 ✓；2.5d²=41,943,040 ✓；11008=43×256 ✓；t≈8400 交叉 ✓。
2.2：BPE 手算例四轮（9/9/7/7 与语料长 79→70→61→54→47）本人重算全中 ✓；GPT-2 50257=256+50000+1 ✓；GPT-2 参数 84.9M+38.6M=123.5M ✓；std(z)=0.02×64=1.28 ✓；ln(151936)=11.93 ✓；GELU 驻点 x*≈−0.75（Φ(−0.75)+(−0.75)φ(−0.75)=0.0008）✓。
2.3：scores 内存表（1/2/64 GiB…1/2/64 TiB）✓；交叉点 T=4d/2d ✓；本体 vs 投影 20%@4096、89%@131072 ✓。
2.4：除 F1 外实例表全中 ✓；「Q 一份 = 式 (9) 的 2 倍、总量 3 倍」✓；B=32 → 512 GiB ✓。
2.5：全部波长/分桶/角度数值 ✓（A4）；T5 桶边界 11.3/16/22.6/32/45.3/64/90.5/128 与整数区间逐端点核对 ✓；ALiBi H=32 斜率 2^{−1/4}→2^{−8}、跨 215× ✓；4 nat/16 nat/e^{−12} ✓。
2.6：三家 FFN 账（8.0625d²/9d²/10.5d²，占比 66.8%/78.3%/80.8%）✓；整机 FFN 占比 70.2%/59.3% ✓；σ'(2)×3≈0.31 ✓；458,752 条记忆 ✓。
2.7：√(ln 4096)=2.88≈2.9 ✓；DeepNorm 数值 ✓；soft-capping 梯度 0.42/0.07/0.01 @ c/2c/3c ✓。
2.8：N_total=8,030,261,248 逐项重加 ✓；OLMo 7,298,617,344 ✓（含 q/k_norm 的 norm 参数 528,384 ✓）；冒烟测试断言的期望参数式本人重算 = 3,344,640 与式 (6) 一致 ✓；§5 九行 FLOPs 表逐项重算 ✓（低阶项 3.9e5 = 0.09% ✓）；并行块 GEMM 6144+28672=34816 ✓。

### G. 抽查覆盖率声明

- arXiv ID↔标题：100%（32/32）。
- 章节引用的一手论文：W1 引用的 19 篇论文全部拉取全文并 grep 关键论断（上表 E），19/19 有直接命中证据或注明结论。
- config 数值：6 个 HF config 全字段核对（Qwen3-8B、OLMo-2-1124-7B、DeepSeek-V3、GPT-2、Qwen2.5-1M、Llama-3.1-8B/70B 经镜像）。
- 代码走读声明：4 个 modeling 文件（llama/olmo2/qwen3/nanoGPT）逐行对照。
- 正文数值事实：约 80 处可计算断言中重算约 70 处（其余为「见练习」类）；发现 1 处硬错误（F1）+ 1 处口径错误（F2）+ 1 处口径混用（F4）。
- 强制重推导 5/5 完成（A1–A5，含中间步）；另做 1 组 Monte Carlo 模拟验证 2.3 §3.2 的统计断言。
- 未覆盖项：GFM 数学渲染实测（仓库无远端，白名单锁定条款允许推迟）；Gemma HF config（gated，正文已正确改挂论文口径）；Llama 官方 gated config（以 unsloth 镜像仓替代核实）。

## 总结与结论

一句话：11 章的推导主干（Chinchilla Lagrangian、√d_k 链、KV cache 论证、RoPE 证明、Xiong 定理）全部独立重推通过、引用网络 100% 真实命中，但存在 2 处 S1 级事实错误（OLMo-2 cache 4 GiB 应为 2 GiB；DeepSeek-V3 2.788M 小时被错标为 pre-training 口径，实际含 post-training、纯 pre-training 为 2.664M）。

**结论：不放行。** 修复 F1–F4（S1×2、S2×2）后需复审 1.2 与 2.4 两章；F5–F9（S3×5）记入累计、下一波复审时统一清零。

---

## 复审记录（2026-11-11，修复后轮次 2）

修复提交：d1f9d8c（1.2）、e9194d5（2.4）、3bbf48f（2.7）、55f5d43（1.3）。复审方式：重读当前全章状态（非 diff），关键数值全部重算，新增论断回一手来源核对。四章 lint 全部 LINT OK。

### F2（1.2，S1：DSV3 训练时长口径）——关闭，放行 1.2

- §1 现为「全训练耗 2.788M…拆分为 pre-training 2664K、context extension 119K、post-training 5K」，与论文 Table 1 逐字一致 ✓。
- §9 三步对账改用 2.664e6 GPU-hours：×3600 = 9.59e9 GPU-秒 ✓；3.29e24/9.59e9 = 3.43e14 FLOPS ✓；交叉验证 14.8e12/2.664e6 = 5.6e6 token/卡时、1543 token/s、×2.22e11 = 3.43e14 ✓；并显式登记了误用 2.788M 的 3.27e14 对照（「只差 5%，但口径已经混了」）✓。
- 派生链一致性：§5 集群算例改为 3.4e14 FLOPS/卡 → 1024 卡 30 天 = 9.0e23 FLOPs（重算 1024×3.4e14×2.592e6 s = 9.02e23 ✓）→ 8B 训 22T（9.0e23/4.19e10 = 2.15e13 ✓）、70B 训 2.1T ✓；练习 2/3 与参考节注记同步 ✓；w1-sources.md 第 20/41 行已同步并留更正记录 ✓；全 repo 无残留 3.3e14 ✓。
- 残留（降为 S3，累计）：§1「Table 1 给出四个数」仍不准确——Table 1 只有成本行（2664K/119K/5K/2788K 与美元行），671B/37B/14.8T 出自摘要与架构节。建议改为「论文摘要与 Table 1 给出四个数」。

### F1（2.4，S1：OLMo-2-7B cache 4→2 GiB）——关闭，放行 2.4

- §5 实例表 OLMo-2-7B 行现为 2 GiB，与式 (9)（2×32×32×128×4096×2 B = 2^31 B）及同行每 token 增量 512 KiB × 4096 = 2 GiB 自洽 ✓。
- 波及面无残留：总结表未含 OLMo 行；MQA 2 GiB 行不动且仍正确；全 repo grep 无残留裸「4 GiB」；其他章节无引用该数值 ✓。

### F3（2.7，S2：QK-Norm 原文形式）——原断言不再成立，关闭

- 新文本正确区分：原文 = 逐 head ℓ2 归一化 + 可学标量 g **替换** 1/√d_k（余弦相似度，硬上界 |g|）；现代变体（Gemma 3 / OLMo 2）= RMSNorm + 保留 1/√d_k（式 (15a) 新增）✓。
- 新增论断回源核对：g₀ = log₂(L²−L)、L 为训练语料序列长度 97.5 分位数（原文 eq (3) ✓）；「Table 5 消融该分位数选择」（原文 Table 5 即 75th–99th 分位数扫描 ✓）；「去掉 g 掉分最多」（原文 Table 6：Without g 24.53 BLEU，其余消融 ≥31.5 ✓）。逐 head 粒度归属（Henry 原文与 Gemma 3）不变且正确 ✓。

### F4（1.3，S2：推理 FLOPs 口径）——原断言不再成立，关闭

- 实账表一、三行改为 1.5e10（= 1.2 §9 全量口径 2N_full），并新增口径注记说明第二行 6.9e10 为式 (2) 非 embedding 口径、输出头修正 <2% ✓。
- 打平式同步更新：2.2e24 + 1.5e10·Q = 6.8e23 + 6.9e10·Q ⟹ Q* = 1.52e24/5.4e10 ≈ 2.8e13 ≈ 3e13 ✓；「贵 4.3 倍」改 4.6 倍（6.9/1.5 ✓）；§5 引言与总结表口径一致 ✓。

### 复审结论

四章 S1×2 + S2×2 全部关闭，**W1 整体放行**（1.2、2.4 两章复审通过，2.7、1.3 的 S2 修复确认消除原断言）。未关闭项：S3 累计 6 条（原 F5–F9 + 本轮新增 1 条「Table 1 给出四个数」措辞），下一波复审时统一清零。建议将本轮结论（轮次 2、S1/S2 全关闭、W1 放行）回填 models/README.md 审查记录表。
