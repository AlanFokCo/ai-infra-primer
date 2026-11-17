# 7.1 数据管线与配比（Data Pipeline and Mixing）

> 本章把预训练数据侧拆成五段可独立对账的管线——来源 → 过滤 → 去重 → 配比 → tokenize + pack——逐段给机制与数值：FineWeb 用 1.71B 代理模型跑 70 余次消融定下的五段管线（36T → 20T → 15T tokens 的逐段账）、MinHash/LSH 相似度估计的完整推导（含 FineWeb 112 个哈希、14 桶 × 8 哈希对应的 75% 阈值验算）、配比的两条路线（Llama-3 的手工配比表 50/25/17/8 与 DoReMi 的 group DRO 学习配比）、annealing 数据课程的逐家实证，以及 packing 的效率账与跨文档 attention 污染的两种处置 | 前置：1.5（tokens 口径）、2.3 §5（噪声尺度与 batch 联动——ramp 章 7.2 用）；models 2.2（tokenizer 与 fertility，纯文字指引）、models 1.3（compute-optimal 数据量，纯文字指引） | 锚点：FineWeb（[Penedo et al., 2024](../references.md)，arXiv:2406.17557，[A]）、DCLM（[Li et al., 2024](../references.md)，arXiv:2406.11794，[A]）、DoReMi（[Xie et al., 2023](../references.md)，arXiv:2305.10429，[B]）、Llama-3 herd [A]、DeepSeek-V3 [A]、MiniCPM [A]

## 1. 五段管线：从网页到训练张量

FineWeb 的最终管线是五段的典型实例（[Penedo et al., 2024](../references.md)，arXiv:2406.17557，[A]，§3.7）：对 96 个 Common Crawl 快照，先做文本抽取（WARC 原始 HTML 经 trafilatura 抽正文，不用 WET 的粗抽结果），再过基础过滤（URL 黑名单 + fastText 语言分类 + MassiveText 规则），剩约 36T tokens；然后逐快照独立做 MinHash 去重，剩 20T tokens；再过一组 C4 式规则过滤与三条自定义启发式过滤器，最终 15T tokens。每一段都由小模型消融实验证明有净收益（§7 的方法学）。把五段与各家对应物画在一起：

```
来源      过滤（§2）            去重（§3）           配比（§4）         tokenize + pack（§6）
────────► 启发式规则          ──► MinHash / LSH     ──► 手工配比表      ──► 定长 packed 序列
网页快照   质量分类器           （近似重复）           学习配比(DoReMi)     （文档拼接 + 边界处理）
书籍代码   perplexity 过滤      Bloom filter（DCLM）  数据课程（§5）       vocab / fertility
│          │                   │                    │                  │
36T tok    删 ~22% tok         20T tok              50/25/17/8         [B, T] int32
(FineWeb   (FineWeb 三条       (FineWeb 逐快照)      (Llama-3 最终表)     （下游唯一直接消费的形态）
 §3.3)      §3.6)
```

五段的共同设计原则是**决策全部用消融证据驱动**：任何一条过滤规则、一个去重阈值、一种配比，都要先在代理规模上训出对照模型、跑基准，再决定是否进管线。Llama-3 的数据侧是同一结构的企业版（§4、§5 给它的口径）；GLM-4.5 在过滤段另加一道语义级去重（SemDedup——模板批量生成的网页 MinHash 删不掉，§2 末）。本章余下各节按段展开；去重（§3）与配比（§4）是两处推导密集区，packing（§6）带一张效率账。

## 2. 过滤：启发式规则、质量分类器与 perplexity

过滤段的三类手段按单文档成本从低到高排：启发式规则（计数与阈值）、小模型分类器（fastText 级）、LM 打分（perplexity 或 LLM 标注蒸馏）。三者在真实管线里串联使用，各自拦不同形态的低质。

**启发式规则**是逐文档统计量加阈值。FineWeb 的三条自定义过滤器（§3.6，全部有 28B tokens 消融背书）：行末带标点的行占比 $\le 0.12$ 则删（删 10.14% tokens；C4 原过滤器同指标的删法对应 30%）；重复行字符占比 $\ge 0.1$ 则删（删 12.47%；MassiveText 原阈值为 0.2）；短于 30 字符的行占比 $\ge 0.67$ 则删（删 3.73%）。三条并集共删约 22% tokens，聚合基准分反而升约 1 个百分点。Llama-3 的启发式清单（[Grattafiori et al., 2024](../references.md)，arXiv:2407.21783，[A]，§3.1.1）形态相同：重复 n-gram 覆盖率（删日志/报错类长行）、脏词计数（补域名黑名单漏掉的成人站）、token 分布与全语料的 KL 散度（删离群 token 过多的文档）。

**质量分类器**把「像不像高质量来源」做成二分类。DCLM 在 1B-1x 规模上横评了七种方案（[Li et al., 2024](../references.md)，arXiv:2406.11794，[A]，Table 4）：fastText 二分类器（正例 = OpenHermes-2.5 指令数据 + ELI5 高分帖，负例 = RefinedWeb 复现版的随机网页抽样，训练集约 40 万文档、两类各半）Core 均分 30.2，居首；其后依次是 top-k 平均 logit 打分 29.2、perplexity 过滤 29.0、AskLLM 28.6、BGE 嵌入线性分类器 27.2、SemDedup 27.1、PageRank 前 20% 26.1。阈值上保留前 10% 优于 15% 与 20%（Table 5）。Llama-3 用的是同族方案的两个变体：fastText 分类器学「是否会被 Wikipedia 引用」，以及 DistilRoberta 分类器学 Llama 2 对清洗后网页的质量判定（§3.1.1）；代码与数学另有域专用管线（DistilRoberta 学 Llama 2 的域标注）。分类器方案的纪律问题在正例选择——DCLM 的消融显示换正例集（Wikipedia / OpenWebText2 / GPT-3 式混合）Core 分在 34.7–37.5 之间摆动，比换算法本身的影响大。

**perplexity 过滤**用一个小 LM 给文档打分、保留低 perplexity 段（CCNet 的原始口径）。DCLM 的实现细节值得登记：打分模型是 154M 参数的 causal Transformer，训练数据为英文 Wikipedia + RedPajama-v1 books + peS2o 的混合。它在横评中输给 fastText 一个百分点以上（29.0 对 30.2），成本却高得多——这是 perplexity 过滤在 2024 年后的开放管线里逐渐退到次要位置的直接证据。

分类器删不掉的是「模板批量生成、彼此近似但逐字不重复」的网页——MinHash 的 n-gram 重合对它们不灵敏。GLM-4.5 的处置是在 MinHash 之外再加 SemDedup（按文档嵌入的语义相似度去重），原话是这类页面「cannot be removed by MinHash deduplication」（[GLM Team, 2025](../references.md)，arXiv:2508.06471，[A]，§2.2）。

## 3. 去重：MinHash / LSH 的机制与近似重复率账

**MinHash 的相似度估计**。把文档 $D$ 表示为词级 n-gram 集合 $S$（FineWeb 取 5-gram，英文分词器切词）。两文档的相似度用 Jaccard 系数：

$$
J(S, T) = \frac{|S \cap T|}{|S \cup T|} \qquad (1)
$$

**符号**： $S, T$：两文档的 n-gram 集合。MinHash 的技巧是用哈希置换把集合比较压成签名比较。取一个把全体 n-gram 随机打乱的置换哈希 $h$，记 $h(S) = \min_{x \in S} h(x)$（集合在置换下的最小像）。关键恒等式：

$$
\Pr[h(S) = h(T)] = J(S, T) \qquad (2)
$$

证明一行： $S \cup T$ 在随机置换下的最小元均匀分布在该并集上， $h(S) = h(T)$ 当且仅当这个最小元落在交集中，概率即 $|S \cap T| / |S \cup T|$。取 $m$ 个独立置换哈希得到长度 $m$ 的签名向量，逐位一致率

$$
\hat{J} = \frac{1}{m} \sum_{j=1}^{m} \mathbb{1}\left[ h_j(S) = h_j(T) \right], \qquad \mathbb{E}[\hat{J}] = J, \qquad \mathrm{std}(\hat{J}) = \sqrt{\frac{J(1-J)}{m}} \qquad (3)
$$

**符号**： $m$：签名长度。 $m = 112$ 时， $J = 0.75$ 处的估计标准差 $\approx 0.041$——逐对估计足够用，但 $O(M^{2})$ 的两两比较在亿级文档上不可行，需要 LSH 把比较压成桶查找。

**LSH banding：阈值是可调的**。把 $m$ 个哈希切成 $b$ 个桶（band）、每桶 $r$ 个（ $m = b \cdot r$ ）；两文档只要有一个桶的 $r$ 位签名全同，就判为候选重复对。候选概率是逐带独立的并事件：

$$
P_{\mathrm{cand}}(s) = 1 - \left( 1 - s^{r} \right)^{b} \qquad (4)
$$

**符号**： $s = J(S, T)$：真实相似度。这是一条 S 形曲线： $s$ 低于阈值时 $s^{r}$ 快速压零，高于阈值时 $1 - s^{r}$ 快速压零，拐点约在 $s^{\ast} = (1/b)^{1/r}$。代入 FineWeb 的参数 $b = 14$ 、 $r = 8$（[Penedo et al., 2024](../references.md)，[A]，§3.4：112 个哈希分 14 桶，目标 75% 相似度）： $s = 0.75$ 时 $s^{8} \approx 0.100$ ， $P_{\mathrm{cand}} \approx 1 - 0.900^{14} \approx 0.771$； $s = 0.5$ 时 $s^{8} \approx 0.0039$ ， $P_{\mathrm{cand}} \approx 0.053$； $s = 0.9$ 时 $\approx 0.9995$。即真阳性对在 0.75 处约四分之三被捞出、0.9 处几乎不漏，而半相似的文档九成五不会被误判为候选——阈值附近的漏检率与误检率由 $b, r$ 显式控制，这是 MinHash 相对精确去重（后缀数组、Bloom filter 精确匹配）的核心卖点： fuzzy 且可调。判重之后 FineWeb 再做传递聚类（A 与 C 重复、B 与 C 重复 ⇒ A、B、C 同簇），每簇随机留一篇。

**FineWeb 的对照实验给出去重策略的一个反直觉结论**。直觉做法是全局去重（96 个快照合在一起迭代去重）：最老的快照被其余 95 个反复「对账」，最多删掉 90%，全局剩 4T tokens。但 350B tokens 消融显示全局去重对非去重基线几乎无提升，且低于 RefinedWeb 对照（§3.4 Figure 3/4 口径；原文为图形读数、无点值）。定位实验：把 2013-48 这个老快照单独切开，「全局去重后留下的约 31B tokens」训出的模型，不如「被全局去重删掉的那 460B 再单独去重得到的 171B tokens」训出的模型——留下的反而是更差的 10%（人工抽查：留下的更多是广告与关键词堆砌页）。机制读法（论文 §3.4 的假设）：去重的主要收益来自删掉「几十万篇级」的大重复簇（它们在所有快照里都存在），而对每簇不足百篇的长尾重复继续动刀反而误伤。最终方案是**逐快照独立去重**，产 20T tokens，打平 RefinedWeb。这个教训的普适形式：去重强度不是单调有益的，「删掉的」与「留下的」要用同一消融协议对照。

**工程替代**也值得记：DCLM 横评 MinHash（挂在后缀数组管线里）与改造的 Bloom filter 近重方案，两者下游表现在 7B-2x 规模差距在 0.2 个 Core 百分点以内，但 Bloom filter 在 10TB 以上数据规模更好扩展——DCLM-baseline 选 Bloom filter，其余实验用 MinHash（[Li et al., 2024](../references.md)，[A]，§3.3 与附录 L）。Llama-3 则是三级去重叠加（§3.1.1）：URL 级（同 URL 保留最新版）、文档级（全局 MinHash）、行级（仿 ccNet，在 30M 文档桶内出现超 6 次的行删除——人工抽查确认它同时删掉了导航栏模板与高频优质文本，但实证净收益为正）。

## 4. 配比：手工配比表与学习配比

去重后的语料是多域混合（网页 / 代码 / 数学 / 多语 / 书籍），配比（domain mixing）决定每个域被采样的权重。两条路线并行存在。

**手工配比表**（主流做法）的代表是 Llama-3 的最终配比：约 50% 通用知识、25% 数学与推理、17% 代码、8% 多语（[Grattafiori et al., 2024](../references.md)，[A]，§3.1.2 的 Data mix summary）。这张表不是拍出来的：流程是先用知识分类器给网页数据分域、对过度代表的类目（如娱乐）降采样，然后对候选配比做小模型 scaling-law 实验——同一规模下训若干配比的模型、外推预测大模型在各基准上的表现，筛出的候选再训一个更大的模型验证（§3.1.2）。DCLM 的 mixing track 补了一条边界证据：把 Wikipedia / StackExchange / arXiv 等高质量源按 67%/33% 混入 Common Crawl 系数据，对过滤较弱的子集（C4、RedPajama-CC、RefinedWeb）有提升，对已充分过滤的 DCLM-baseline 反而平均掉分（[Li et al., 2024](../references.md)，[A]，Table 6）——配比手段补救不了过滤段没做完的功课。

**学习配比**把配比本身变成优化变量。DoReMi（[Xie et al., 2023](../references.md)，arXiv:2305.10429，[B]）分三步：第一步训一个参考模型（实验用 280M 参数）提供各样本的基线难度；第二步训一个同规模代理模型，用 group DRO 学出域权重；第三步按学到的权重重采样数据、训主模型。第二步的目标是最小化最坏域上的超额损失（excess loss）：

$$
\min_{\theta} \max_{\alpha \in \Delta^{k}} \hspace{5mu} L(\theta, \alpha) = \sum_{i=1}^{k} \alpha_i \cdot \left[ \frac{1}{\sum_{x \in D_i} |x|} \sum_{x \in D_i} \left( \ell_{\theta}(x) - \ell_{\mathrm{ref}}(x) \right) \right] \qquad (5)
$$

**符号**： $k$：域数； $\Delta^{k}$： $k$ 维概率单纯形； $\alpha_i$：域 $i$ 的权重； $\ell_{\theta}(x) = -\log p_{\theta}(x)$：代理模型在样本 $x$ 上的逐 token 负对数似然； $\ell_{\mathrm{ref}}$：参考模型的同一量； $|x|$：样本 token 数。超额损失 $\ell_{\theta}(x) - \ell_{\mathrm{ref}}(x)$ 度量的是「参考模型已经能学好、代理模型还差多少」的可改进余量——高熵难样本（两者 loss 都高）与已学会样本（两者都低）都被减掉，剩下的是「学得会但还没学会」的部分。内层最大化把权重压向当前最差域，外层最小化追着压它——minimax 结构。

优化器是指数上升与梯度下降的交错（论文 Algorithm 1）：每步从均匀域分布采 minibatch，逐域算平均超额损失（逐 token 的 $\max\lbrace \ell_{\theta}(x) - \ell_{\mathrm{ref}}(x), 0 \rbrace$ 再按 token 数归一），然后

$$
\alpha^{\prime}_t = \alpha_{t-1} \odot \exp\left( \eta \thinspace \lambda_t \right), \qquad \alpha_t = (1 - c) \thinspace \frac{\alpha^{\prime}_t}{\sum_{i=1}^{k} \alpha^{\prime}_t[i]} + c \thinspace u \qquad (6)
$$

**符号**： $\lambda_t \in \mathbb{R}^{k}$：当步各域的平均超额损失（截断到非负）； $\eta$：域权重的步长； $c$：平滑系数（实现取 $10^{-3}$ ）； $u = (1/k) \cdot \mathbf{1}$：均匀分布。第一行是指数梯度上升（exponentiated gradient，乘性更新保证权重非负），第二行归一化并向均匀分布掺 $c$——防止任何域权重塌缩到零。轨迹平均 $\bar{\alpha} = (1/T) \sum_t \alpha_t$ 作为输出（Nemirovski 式平均，论文口径）。实证：280M 代理学出的权重直接用于 8B 主模型（30 倍参数差），The Pile 上平均 few-shot 准确率比默认配比基线高 6.5 个百分点，达到基线同等水平只需 1/2.6 的步数（代理与主模型同规模时加速比为 4 倍）。注意 DoReMi 的产出是**权重**不是过滤规则：它回答「每个域吃多少」，不回答「域内留谁」——与 §2 的过滤段正交。

## 5. 数据课程：annealing 与后期上采样

配比不必全程恒定。**数据课程**（data curriculum）在训练后期集中喂高质量数据，是 2024 年后各家公开配方里反复出现的结构，四个实例：

- **Llama-3 的 annealing**（[Grattafiori et al., 2024](../references.md)，[A]，§3.1.3）：对小量高质量代码与数学数据上采样做退火，8B 模型 GSM8k 与 MATH 验证集分别提升 24.0% 与 6.4%；同法用于 405B 几乎无提升——旗舰模型的推理能力不靠退火期的域内样本。同节还有一个把 annealing 当测量工具的用法：取训到 50% 的 8B 模型、lr 线性降到 0、跑 40B tokens、其中 30% 权重给待评估的新数据集（70% 给默认配比）——退火终点性能即该数据集的质量分，比每个小数据集各跑一轮 scaling-law 实验便宜。
- **DeepSeek-V3**：预训练语料相对 V2 提高数学与编程样本占比、扩展英中之外的多语覆盖（[DeepSeek-AI, 2024](../references.md)，arXiv:2412.19437，[A]，§4.1）——配比调整放在数据构造阶段，与课程正交但同向。
- **MiniCPM**：WSD 的 decay 段混入高质量知识数据与 SFT 数据（[Hu et al., 2024](../references.md)，arXiv:2404.06395，[A]，§5）。消融对照（A-1 vs A-2、B-1/B-2/B-3 系列）显示退火期混入 SFT 分布的数据优于「纯预训练数据退火 + 训后等量 SFT」；论文的机制解释是 decay 段本来就有显著 loss 下降，在此阶段对准 SFT 分布能拿到比训后单独 SFT 更彻底的适配。
- **GLM-4.5**：两阶段预训练——第一阶段以通用网页为主，第二阶段上采样 GitHub 代码与编码/数学/科学相关网页（[GLM Team, 2025](../references.md)，[A]，§2.2）；其后还有独立的 mid-training 段（repo 级代码、合成推理、长上下文 + agent 轨迹，§2.3）。

课程的共同形态：**稳定段吃粗数据、收尾段吃精数据**。它与 lr schedule 的耦合不是巧合——decay/annealing 段的有效学习率低、参数移动小，此时喂入的数据分布被「写进」参数的程度最深（MiniCPM 论文假设的表述），这也是 7.2 把 decay 形状与数据课程放在同一张配方表里读的原因。

## 6. Tokenize 与 pack：定长序列的效率账与跨文档污染

**tokenizer 的训练侧接口**只有两条：词表大小 $V$ 决定 embedding 参数与输出投影的账（models 2.2 §7 的三方账，纯文字指引）；fertility（每词平均 token 数）决定同样文本量折合多少训练 tokens——DeepSeek-V3 的 BBPE 词表扩到 128K 并为多语压缩效率改了预分词器（[DeepSeek-AI, 2024](../references.md)，[A]，§4.1）。tokenizer 在训练前冻结，之后不再改。

**packing 的效率账**。训练框架消费的是定长 $[B, T]$ 张量（ $T$ 为序列长），文档长度 $L$ 服从某个分布。两种端点方案：

- **不 pack（每序列一文档、尾部补 pad）**：pad 位置被 attention mask 与 loss mask 同时屏蔽（3.1 的 mask 语义），计算被浪费。有效 token 占比

$$
\eta_{\mathrm{pad}} = \frac{\mathbb{E}[L]}{T} \qquad (7)
$$

- **顺序拼接 pack（文档逐条填入、跨界截断或续入下一条）**：每个序列至多浪费跨界处的一段文档残尾。设文档长度均值 $\mathbb{E}[L]$ 、切割点在文档内近似均匀，每条序列平均浪费 $\approx \mathbb{E}[L]/2$ 个 token（一个序列只含一个上边界），

$$
\eta_{\mathrm{pack}} \approx 1 - \frac{\mathbb{E}[L]}{2T} \qquad (8)
$$

**符号**： $L$：文档 token 长度； $T$：训练序列长。代入 $\mathbb{E}[L] = 2000$ 、 $T = 4096$ ：不 pack 的 $\eta_{\mathrm{pad}} \approx 0.49$ ，顺序 pack 的 $\eta_{\mathrm{pack}} \approx 0.76$——同样的 GPU 小时多喂一半以上有效 token。 $\mathbb{E}[L] \ll T$ 时 $\eta_{\mathrm{pack}} \to 1$ ，浪费只剩边界残尾项。更精细的 best-fit packing（从候选文档里选恰好填满缝隙的组合，把截断降到接近零）在 ICML 2024 的 Ding et al.（「Fewer Truncations Improve Language Modeling」，经 GLM-4.5 与 DeepSeek-V3 转引）中形式化；GLM-4.5 的取舍恰好说明它不是免费午餐——预训练阶段**不用** best-fit packing，理由是随机截断本身是预训练文档的数据增强；mid-training 阶段改用，理由是推理链与 repo 级代码不能被截断（[GLM Team, 2025](../references.md)，[A]，§2.3 末段）。

**跨文档 attention 污染**。pack 之后一条序列里有多篇文档，causal attention 默认让后文 token 看到前文所有 token——包括不相关的前序文档。两家的处置恰好是两个极端：GPT-3 把多篇文档 pack 进 2048 上下文、仅以 end-of-text token 分隔、不做任何额外 mask（[Brown et al., 2020](../references.md)，arXiv:2005.14165，[B]，App. B）；DeepSeek-V3 用文档级 packing 保数据完整性，但训练时**不**加跨样本 attention mask（[DeepSeek-AI, 2024](../references.md)，[A]，§4.1）。不 mask 的代价是模型学到虚构的跨文档依赖，收益是注意力核不必分块、吞吐最高。第三条路线（现代开放栈的默认）是把每条序列内部的文档边界显式传给 kernel：`position_ids` 在每篇文档起点重置为 0（RoPE 的位置由此不与文档外前文耦合），attention mask 取块对角形（文档内 causal、跨文档全零）；FlashAttention 的 varlen 接口（`flash_attn_varlen_func`，以 `cu_seqlens` 传入各文档的累计边界）原生支持这种变长块对角布局，避免物化 $[B, T, T]$ 的 mask（[Dao et al., 2022](../references.md)，arXiv:2205.14135，[A]，实现接口口径）。三种方案的取舍一句话：不 mask 换吞吐、块对角换语义干净，position 重置是两者的正交补充。

## 7. 数据质量研究的方法学：代理模型消融

FineWeb 的管线决策协议（[Penedo et al., 2024](../references.md)，[A]，§3.1）是本章所有「某过滤器有效」断言的证据形式，值得单独登记：代理模型 1.71B 参数（含 embedding）、Llama 架构、序列长 2048、全局 batch 约 2M tokens、GPT-2 tokenizer；过滤类消融训约 28B tokens（该规模的大致 Chinchilla 最优量），去重类与累计收益验证训 350B tokens；每个数据版本训两个模型（不同数据子抽样与初始化种子）取平均；基准选取有三条显式判据——同一数据集不同抽样下分数方差小、随训练单调（或近单调）上升、该规模下高于随机基线（原文判据为 above random baseline，定性口径）——最终用 CommonSenseQA / HellaSwag / OpenBookQA / PIQA / SIQA / WinoGrande / ARC / MMLU 八个基准（大基准截断到 1000 题）。全部消融共训 70 余个模型、约 8 万 H100 小时。DCLM 把同一协议产品化：240T tokens 的原始池（DCLM-Pool）、412M 到 7B 的五档规模阶梯（7B-1x = 138B tokens、7B-2x = 276B tokens）、固定训练代码与逐规模超参，416 组基线实验（[Li et al., 2024](../references.md)，[A]，§3.2）——其头条结果 7B 模型 2.6T tokens 训到 64% MMLU 5-shot，是「数据侧收益可量化」的标尺数字。方法学结论只有一条：数据决策的证据等级可以做到与模型决策相同，代价是每个决策都要在代理规模上先付一次训练成本。

## 总结

| 段 | 机制 | 关键数值与出处 |
| --- | --- | --- |
| 来源与抽取 | WARC + trafilatura 抽正文（WET 锅炉板太多） | FineWeb 96 快照 → 36T tokens [A] §3.2-3.3 |
| 过滤 | 启发式规则 + fastText 分类器 + perplexity 三层 | FineWeb 三规则删 22% tok、+1pp；DCLM：fastText 30.2 > perplexity 29.0（1B-1x Core）[A] |
| 去重 | MinHash： $\Pr[h(S)=h(T)] = J$ 式 (2)；banding $1-(1-s^{r})^{b}$ 式 (4) | FineWeb $b$=14, $r$=8 → 0.75 阈值、逐快照 20T（全局去重 4T 反伤）[A] §3.4；DCLM Bloom filter 差 <0.2pp [A] |
| 配比 | 手工表（scaling-law 实验筛选）vs DoReMi（式 (5)(6)） | Llama-3 50/25/17/8 [A] §3.1.2；DoReMi 280M→8B、+6.5pp、2.6× [B] |
| 数据课程 | 稳定段粗数据、decay/annealing 段精数据 | Llama-3 8B GSM8k +24.0%（405B 无感）；MiniCPM decay 混 SFT 数据；GLM-4.5 两阶段上采样 |
| tokenize + pack | $\eta_{\mathrm{pad}} = \mathbb{E}[L]/T$ 式 (7) vs $\eta_{\mathrm{pack}} \approx 1 - \mathbb{E}[L]/(2T)$ 式 (8) | GPT-3 / DSv3 不 mask 跨文档；块对角 + position 重置走 FA varlen [A] |
| 方法学 | 1.71B 代理 + 双种子 + 八基准 | FineWeb 70+ 模型、8 万 H100 时 [A] §3.1；DCLM 416 组基线 [A] |

## 练习

1.（推导）MinHash 签名方差分析：(a) 从式 (2) 出发证明式 (3) 的期望与标准差；(b) FineWeb 参数 $m = 112$ 下， $J = 0.75$ 与 $J = 0.9$ 两点的估计标准差各是多少；(c) 若要求 $J = 0.75$ 处 95% 置信半宽 $\le 0.02$ ， $m$ 至少取多少？注意 LSH 分桶只复用这 $m$ 个哈希，不增加独立估计量——估计精度与判重阈值是同一预算的两个出口。

2.（计算）banding 曲线：对 $(b, r) \in \lbrace (14, 8), (28, 4), (7, 16) \rbrace$ 三组（ $m$ 分别为 112 / 112 / 112），逐点计算 $s \in \lbrace 0.5, 0.6, 0.7, 0.75, 0.8, 0.9 \rbrace$ 处的 $P_{\mathrm{cand}}(s)$（式 (4)），画出三条 S 曲线并回答：固定 $m$ 时，把 $r$ 调大（桶变瘦）对「阈值 0.75、宁漏勿错」与「宁错勿漏」两种诉求分别意味着什么？

3.（计算）packing 账：某语料文档长度对数正态、 $\mathbb{E}[L] = 1500$ tokens，训练序列长 $T = 8192$ 。(a) 按式 (7)(8) 算 $\eta_{\mathrm{pad}}$ 与 $\eta_{\mathrm{pack}}$；(b) 若引入 best-fit packing 把边界残尾消掉九成，有效吞吐再升多少；(c) 同语料把 $T$ 降到 4096，重算 (a)——序列长与文档长分布的什么关系决定 pack 的收益？

4.（代码 + 分析）用教学版伪码实现 DoReMi 式 (6) 的域权重更新（三域玩具问题：三个域的「真实超额损失」固定为 0.1 / 0.3 / 0.5，每步加小噪声），跑 200 步观察 $\alpha_t$ 轨迹与 $\bar{\alpha}$；再把截断 $\max\lbrace \cdot, 0 \rbrace$ 去掉重跑，解释轨迹为何不同（提示：负超额损失域的权重会被乘性更新压向零还是抬升？与式 (5) 的 minimax 目标对照）。

## 参考

- [Penedo et al., 2024](../references.md)：The FineWeb Datasets，arXiv:2406.17557，[A]（key: fineweb）——五段管线（§3.7）、基础过滤与 36T（§3.3）、MinHash 参数与全局/逐快照对照（§3.4）、三条自定义过滤器（§3.6）、消融协议（§3.1）
- [Li et al., 2024](../references.md)：DataComp-LM，arXiv:2406.11794，[A]（key: dclm）——七方案质量过滤横评（Table 4）、fastText 消融（Table 5）、Bloom filter vs MinHash（§3.2-3.3 与附录 L）、mixing track（Table 6）、规模阶梯与 416 组基线（§3.2）、7B/2.6T → 64% MMLU
- [Xie et al., 2023](../references.md)：DoReMi，arXiv:2305.10429，[B]（key: doremi）——式 (5) minimax 目标、Algorithm 1 与式 (6)、280M→8B 与 2.6× / 4× 加速
- [Grattafiori et al., 2024](../references.md)：The Llama 3 Herd of Models，arXiv:2407.21783，[A]（key: llama3）——三级去重与启发式/分类器清单（§3.1.1）、配比确定流程与 50/25/17/8（§3.1.2）、annealing 与数据质量测量（§3.1.3）
- [DeepSeek-AI, 2024](../references.md)：DeepSeek-V3 Technical Report，arXiv:2412.19437，[A]（key: deepseekv3）——14.8T 语料与配比调整、文档 packing 不加跨样本 mask、FIM 0.1、BBPE 128K（§4.1）
- [Hu et al., 2024](../references.md)：MiniCPM，arXiv:2404.06395，[A]（key: minicpm）——decay 段混入高质量 + SFT 数据的消融（§5）
- [GLM Team, 2025](../references.md)：GLM-4.5，arXiv:2508.06471，[A]（key: glm-4.5）——质量分桶上采样、SemDedup 补 MinHash、两阶段预训练上采样（§2.2）、best-fit packing 取舍（§2.3）
- [Brown et al., 2020](../references.md)：Language Models are Few-Shot Learners，arXiv:2005.14165，[B]（key: gpt3）——pack 不 mask 的原始口径（App. B）、Spark MinHashLSH 模糊去重（App. A）
- [Dao et al., 2022](../references.md)：FlashAttention，arXiv:2205.14135，[A]（key: flashattention）——varlen 块对角接口的实现口径
