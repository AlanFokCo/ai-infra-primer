# 8.4 视频理解（Video Understanding）

> 视频理解的第一笔账是乘法：$F$ 帧 × 每帧 $M$ 个 token（1 min @ 1 fps × 256 = 15,360 级），直接压进 LLM 的 context 硬预算，4.6 的长上下文四堵墙原样适用；全章按「帧从哪来（uniform vs dynamic FPS 采样）、token 在哪省（时间压缩的三个位置：3D patch / 时空 pooling / temporal queries resampler）、时间怎么读（Qwen2.5-VL 绝对时间编码进 M-RoPE $t$ 轴）、答案怎么写（时间戳分桶方案）」展开，收口于视频-图像混合训练的机制账与视频评测方法论（静态捷径检查、时长分档） | 前置：2.4（KV cache 式 (9)）、2.5（RoPE）、4.6（长上下文四笔账、claimed vs effective 方法论）、6.1（ViT patchify $N = HW/p^{2}$）、8.1（token 预算账）、8.2（connector 四族、pixel shuffle 式 (9)(10)、M-RoPE 式 (14)-(17) 与 `get_rope_index`） | 锚点：Qwen2.5-VL（dynamic FPS + 绝对时间编码官方实现）[A]、Qwen2.5-VL-7B config（ctx 128000 实测）[A]、QwenLM/Qwen3-VL repo [A]、transformers Qwen2.5-VL modeling [A]、LLaVA-OneVision（单图→多图→视频课程）[A]、Apollo（视频 LMM 设计空间 ablation）[B——代码与权重已下架，只引论文结论]、Flamingo（视频 resampler）[B]

## 1. 视频 = 图像 × 时间：总预算账

「视频理解」在 token 层面只增加一个事实：采样器把视频切成 $F$ 帧，每帧走 8.1 的图像管线产出 $M$ 个 token，LLM 看到的视觉段长度是

$$
S_{vid} = F \cdot M \qquad (1)
$$

**符号**：$F$：采样帧数（由时长与采样规则决定，§2）；$M$：每帧进 LLM 的视觉 token 数（connector 出口口径——8.2 的 pixel shuffle $M = N r^{2}$、Q-Former $M = B$、spatial merge 后的 $(H/28)(W/28)$ 全部沿用）。8.1 的账里单图是 576、AnyRes 4 切是 2880；式 (1) 把时间轴乘上去：

| 场景 | 时长 $T$ | fps | $F$ | 每帧 $M$ | $S_{vid}$ |
| --- | --- | --- | --- | --- | --- |
| 短片精看 | 10 s | 2 | 20 | 576 | 11,520 |
| 一分钟基线 | 60 s | 1 | 60 | 256 | **15,360**（15K 级） |
| 一分钟高帧率高 token | 60 s | 2 | 120 | 576 | 69,120 |
| 十分钟省 token 档 | 600 s | 0.5 | 300 | 144 | 43,200 |
| 十分钟常规档 | 600 s | 1 | 600 | 144 | 86,400 |
| 一小时低速档 | 3600 s | 0.2 | 720 | 144 | 103,680 |

「15K 级」的来源即表中第二行：256 tokens/帧 × 60 帧 = 15,360（约定页口径的公式推导值）。这笔账撞上的是硬预算：Qwen2.5-VL-7B 的 context 是 128,000（[Qwen2.5-VL-7B config](../references.md)，[A]）——表中后三行任何一行都吃掉预算的大半，而视觉段只是输入的一部分，prompt 与生成还要从同一预算里扣。把 4.6 的四笔账代进视频档数字，三笔立刻有感：

- 墙二（KV cache，2.4 式 (9)）：按 Qwen2.5-VL-7B 的 LLM 形态（28 层、4 KV 头、$d_{head}$ = 128、bf16，[Qwen2.5-VL-7B config](../references.md)，[A]）每 token 占 $2 \times 28 \times 4 \times 128 \times 2$ B = 57,344 B；15,360 个视觉 token ≈ 0.88 GB，86,400 个 ≈ 4.95 GB——单请求的 KV 预算被视觉段独占一头。
- 墙一（prefill 的 $O(T^{2})$）：$T$ = 86,400、$d$ = 3,584 代入单层 $2T^{2}d \approx 5.4 \times 10^{13}$ FLOPs，28 层合计 $\approx 1.5 \times 10^{15}$——已是旗舰卡秒级的理想下界，还没算视觉塔本身的前向。
- 墙三（softmax 熵稀释）：15K 级视觉 token 摊在同一行归一化里，目标位置的 logit 优势预算随 $\ln T$ 涨（4.6 式 (4) 口径）——长视频「看得见但注意不到」与长文本同机制。

视频因此不是一个新模态问题，而是 4.6 长上下文问题的一个快速实例：它是把 $T$ 抬进 $10^{4}$–$10^{5}$ 档最便宜的途径——一秒钟视频就能产生数百上千个 token。全章的技术地图就是对式 (1) 两个因子的操作：§2 动 $F$（帧从哪来），§4 动 $M$ 与 $F \to$ token 的换算（token 在哪省），§3 解决动 $F$ 之后位置编码的一致性，§5-§8 处理极端 $T$ 与输出、数据、评测。

```
视频 T 秒
   │ §2 帧采样：uniform（固定 F）/ dynamic FPS（固定 fps，F ≈ T·fps）
   ▼
F 帧 RGB
   │ 位置一：3D patch，τ_p 帧合一（§4.1）——encoder 侧就省
   ▼
F/τ_p 组 × N 个 patch ──ViT──► F/τ_p × N 特征 token
   │ 位置二：时空 pooling ÷c_t（§4.2）——encoder 成本照付，LLM 侧省
   ▼
   │ 位置三：temporal queries resampler → 定长 B（§4.3）——瓶颈取舍
   ▼
视觉 token + (t,h,w) 三元组（§3 绝对时间）──► LLM（context 硬预算）
```

## 2. 帧采样：uniform 与 dynamic FPS

帧采样器（frame sampler）决定式 (1) 的 $F$，两条规则各有一个结构性缺陷。

**uniform frame-count（均匀取 $F$ 帧）**：无论片长，均匀取 $F$ 帧，采样间隔 $\Delta t = T / F$ 随片长线性变稀。混叠账：要分辨时长 $\delta$ 的事件，相邻采样间隔必须 $\Delta t \lesssim \delta$，即

$$
F \;\ge\; \frac{T}{\delta} \qquad (2)
$$

一分钟视频里定位一秒级事件 ⇒ $F \ge 60$；十分钟 ⇒ $F \ge 600$。$F$ 固定意味着**时间分辨率随片长退化**——这是固定帧数方案的结构性上限，与实现无关。

**dynamic FPS（固定 fps）**：$F \approx T \cdot fps$（实现向下取整到帧），时间分辨率与片长解耦，代价是 $F$ 随 $T$ 线性涨，直接撞式 (1) 的预算。预算约束下的可行 fps：

$$
fps^{*} = \min\!\left( fps_{max},\; \frac{S_{budget} - S_{text}}{T \cdot M} \right) \qquad (3)
$$

**符号**：$S_{budget}$：context 总预算；$S_{text}$：文本侧预留；$M$：每帧 token 数。式 (3) 引出一个连锁问题：$fps$ 变成片长 $T$ 的函数——同一模型对 10 秒片用 2 fps、对 10 分钟片被迫降到 0.2 fps。若位置编码按帧序号计数，「相邻两个采样帧」在 10 秒片里对应 0.5 秒物理时间、在 10 分钟片里对应 5 秒：同一位置间隔对应相差 10 倍的物理间隔。模型要从位置差读出「过了多久」，就得先知道本片的 fps——而这个信息不在任何 token 里。这是帧序号位置编码与 dynamic FPS 的组合内在地监督不一致，也是 §3 绝对时间编码要消掉的东西。

Qwen2.5-VL 走 dynamic FPS 路线：采样率按片长与预算动态调整（[Bai et al., 2025](../references.md)，arXiv:2502.13923，[A]；官方实现见 [QwenLM/Qwen3-VL repo](../references.md)，[A]——Qwen2.5-VL 系列 repo 已归并），并用下一节的绝对时间编码把位置语义从「帧序号」换成「秒」。

## 3. 绝对时间编码：把秒写进 M-RoPE 的 $t$ 轴

8.2 式 (14) 的视频支给每个视频 patch 一个三元组 $(t_0 + \tau_f,\; t_0 + i,\; t_0 + j)$，全部时间信息集中在 $\tau_f$ 这一个函数上。两代 Qwen-VL 的差别就在这个函数的定义：

- 帧序号规则（Qwen2-VL，[Wang et al., 2024](../references.md)，arXiv:2409.12191，[A]）：$\tau_f$ 按（temporal merge 后的）帧序递增，相邻采样帧的 $t$-id 差是常数，物理间隔完全不出现在位置编码里。
- 绝对时间规则（Qwen2.5-VL，[Bai et al., 2025](../references.md)，[A]）：$t$-id 与帧的真实秒数对齐，

$$
\tau_f = \operatorname{round}\left( t_f \cdot \nu \right) \qquad (4)
$$

**符号**：$t_f$：第 $f$ 个采样帧的绝对时刻（秒）；$\nu$：$t$-id 速率，即每秒多少个位置 id（换算常数以 tech report 原文为准，本书只按机制呈现）。

式 (4) 的直接后果是**间隔不变性**：两个事件位于绝对时刻 $t_a < t_b$，无论采样率取多少，

$$
\tau_b - \tau_a = \operatorname{round}(t_b\, \nu) - \operatorname{round}(t_a\, \nu) \approx \nu\,(t_b - t_a) \qquad (5)
$$

误差至多 1 个 id（两次取整各贡献至多 1/2），与 fps 无关。对照帧序号规则：同一对事件的 id 差 $= (t_b - t_a) \cdot fps$——正是式 (3) 里随片长漂移的那个 fps。RoPE 的 score 只含相对位移（8.2 式 (17) 的 $t$ 段项 $B_j(q, k; (t_n - t_m)\,\theta_j)$），所以「模型从 attention 相位读到的时间差」在两条规则下分别正比于帧数差与秒数差。dynamic FPS 下，前者是一把随片长漂移的尺子，后者是固定尺子——绝对时间编码的存在理由就是把采样率从位置语义里消掉。

对事件定位（temporal grounding，时序定位）的意义具体有三层：

1. **监督一致**：定位数据的标注以秒为单位（§6 的输出格式），文本侧的时间数字与位置侧的 $t$-id 共用同一时钟——标注「12.0 s」对应的相位差恒为 $12\nu$。若用帧序号 + dynamic FPS，同一标注在不同片长下对应不同相位差，定位监督自带噪声。
2. **分辨率摆对了位置**：8.2 式 (15) 的频段切分把最高频的 16 个通道分给 $t$ 轴（mrope_section [16,24,24]，[Qwen2.5-VL-7B config](../references.md)，[A]）——高频通道对小的 $\Delta t$ 最敏感，局部时间次序的分辨率最高，恰好是「事件 A 在事件 B 前半秒」这类判断需要的粒度。
3. **节奏可读**：相邻采样帧的 id 差 $= \nu / fps$，帧率本身成了可推断量——快放、慢放、变速片段在位置编码里留下与物理节奏一致的痕迹，而不是被均匀化成无差别的「相邻帧」。

实现落点：三元组构造在 `get_rope_index` 的视频分支（8.2 §9.4 伪码的 `tau(f)` 即式 (4)；走读锚点 [transformers Qwen2.5-VL modeling](../references.md)，[A]），采样器输出的帧时刻 $t_f$ 直接进位置 id——采样与编码由「秒」这个公共单位咬合。

## 4. 时间压缩：token 省在三个位置

式 (1) 的预算要省，压缩可以发生在管线的三个位置——差别在谁的前向被省掉。

### 4.1 位置一：输入侧 3D patch（temporal patch）

patchify 时对 $\tau_p$ 个相邻帧取一个时空 patch（tubelet）：

$$
z_{f,i,j} = W_e\,\mathrm{vec}\!\Big( x\big[\, \tau_p f : \tau_p f + \tau_p,\; ip : (i{+}1)p,\; jp : (j{+}1)p \,\big] \Big) + b_e \qquad (6)
$$

**符号**：$x$：输入视频体 $[F, H, W, 3]$；$\tau_p$：时间 patch 长（Qwen2-VL / 2.5-VL 为 2，即相邻两帧合并，tech report 与 config 口径，[Bai et al., 2025](../references.md)，[A]）；$p$ = 14：空间 patch 边长；$W_e$：patch embedding 线性层。token 数从 $F \cdot N$ 变为 $(F/\tau_p) \cdot N$。

性质与 8.2 §5.1 的 pixel shuffle 同构：合并发生在任何线性变换之前，是无损重排，取舍集中在 $W_e$ 上——$\tau_p$ 帧内的差分（运动的一阶信息）被排进通道维，由 $W_e$ 学读取。与 pixel shuffle 的差别在收益位置：合并发生在视觉塔入口，ViT 序列直接短 $\tau_p$ 倍，encoder 侧 FLOPs 同步省掉（自注意力还赚一个平方项）。代价同样写在输入处：时间分辨率在 patchify 之后就定为 $\tau_p / fps$ 秒一格，更细的事件边界对 ViT 不可见，只能靠通道里保留的差分间接恢复。M-RoPE 的 $t$ 轴挂在 merge 后的帧组上（8.2 §9.4），与式 (4) 的绝对时间对齐在同一粒度上生效。

### 4.2 位置二：encoder 后时空 pooling

每帧独立过完 ViT，再在特征序列上跨帧合并：

$$
v'_g = \mathrm{Pool}\!\left( v_{c_t g},\; \ldots,\; v_{c_t g + c_t - 1} \right), \qquad S_{vid} = \frac{F}{c_t} \cdot M \qquad (7)
$$

**符号**：$c_t$：时间压缩比；Pool：mean / max / concat+linear 皆可（空间版即 8.2 式 (9) 的 pixel shuffle，时间版同族；$c_t$ = 2 常与空间 2×2 叠加成 ÷8 总压缩）。LLM 侧视觉段 ÷$c_t$、attention 二次项 ÷$c_t^{2}$；但 $F$ 帧的 ViT 前向分文未省——长视频场景的吞吐瓶颈若在视觉塔，这个位置不解决问题。LLaVA-OneVision 的视频管线即此族：均匀采样帧各自过视觉塔、空间下采样后按序拼进 LLM（[Li et al., 2024](../references.md)，arXiv:2408.03326，[A]；每帧 token 数与采样帧上限引论文/repo 原文，[LLaVA-NeXT repo](../references.md)，[A]）。位置一与位置二是同一枚硬币的两面：分辨率在 encoder 前丢（便宜）还是在 encoder 后丢（保住 ViT 全程高分辨率）——静态细节多的内容（屏幕文字、文档画面）倾向后者，预算紧的倾向前者。

### 4.3 位置三：connector 侧 temporal queries（resampler 定长瓶颈）

可学 $B$ 个 query 对全部 $F \cdot M$ 个视觉 token 做 cross-attention，输出定长——8.2 式 (8) 的视频用法：Flamingo 即把 $F$ 帧特征拼上时间位置编码后一起当 K/V（[Alayrac et al., 2022](../references.md)，arXiv:2204.14198，[B]，DeepMind 未开源，公式引论文）。成本对照：resampler 每层 $O(B \cdot F \cdot M \cdot d)$，把二次项挡在 LLM 之外；输出 $B$ 个（或按时间段各 $B$ 个）token，视觉段长度与片长解耦——长视频预算的最激进方案。代价沿用 8.2 §3.4 的容量-数据账：取舍函数要从零学，且视频版多了一维自由度——「选哪些瞬间」与「瞬间里选哪些区域」耦合，训练信号比图像瓶颈更稀。Apollo 的设计空间 ablation 覆盖了时间压缩方式与每帧 token 预算的权衡（[Zohar et al., 2024](../references.md)，arXiv:2412.10360，[B]），方向性结论：固定 token 预算下，用每帧 token 换时间覆盖（帧数）是划算方向——具体档位数字引论文原表。

三位置对照：

| 压缩位置 | token 公式 | encoder 侧省 | LLM 侧省 | 取舍可学处 | 代表 |
| --- | --- | --- | --- | --- | --- |
| 输入侧 3D patch | $(F/\tau_p) \cdot N$ | 是（ViT 序列 ÷$\tau_p$） | 是 | $W_e$ 一层 | Qwen2-VL / 2.5-VL（$\tau_p$ = 2）[A] |
| encoder 后 pooling | $(F/c_t) \cdot M$ | 否 | 是（二次项 ÷$c_t^{2}$） | 规则 Pool 或一层线性 | LLaVA-OneVision [A] |
| connector 侧 resampler | 定长 $B$（或 段数 × $B$） | 否 | 最多（与片长解耦） | 整个 resampler 塔 | Flamingo [B] |

## 5. 流式与长视频：chunk + 循环 memory（机制描述）

把 4.6 的墙推到极端：直播、监控、小时级视频里 $T$ 无界，任何「全量帧进 context」的方案都有顶。通用结构是 chunk + 循环 memory：视频切成固定时长 chunk $C_k$ 顺序处理，模型维护一个大小与 $k$ 无关的 memory 状态 $S_k$——读侧，问题时刻的 query 对 $S_K$ 与当前 chunk 做 attention；写侧，每个 chunk 处理完把增量信息压入状态 $S_{k+1} \leftarrow \mathrm{Update}(S_k,\, E(C_k))$。

与本书既有机制的对应是直接的：$S_k$ 就是 4.3 线性注意力「固定大小状态」的语义版——把 $O(T)$ 的记忆换成 $O(1)$ 的有损压缩；chunking 则把 KV cache（墙二）的上界钉在 chunk 长度上，与 4.6 §3 的分块思想同源。三个机制难点没有便宜解：(a) 写入目标决定记忆内容——状态容量有限，在线取舍比 §4.3 的离线瓶颈更苛刻（写入时看不到未来）；(b) 状态漂移——压缩误差在 Update 链上累积，长程依赖随链长退化；(c) 评测缺位——流式能力的基准建设落后于离线视频基准，§8 的分档与对照检查同构适用。本节无收录条目（本轮研究未找到合用的开放论文锚点），按 w5-sources 降级口径只给机制描述，不引具体系统与数字。

## 6. 输出格式：temporal grounding 与时间戳

视频特有的输出类型是时间：给定 query（「进球发生在什么时候」），模型要产出时间段 $[t_s, t_e]$。连续时间挤进离散文本通道有两套方案。

**方案一：十进制数字文本**。直接生成「12.5 秒」。零词表改造、量程无限；代价是数字被 BPE 切碎（「12.5」可能占 3-4 个 token，切分边界随 tokenizer 合并表漂移），逐位生成是隐式的自回归回归——数值误差在 token 间累积，监督信号被拆散到多个位置上。

**方案二：分桶 special token（special token 化）**。把 $[0, T_{max}]$ 量化进词表，新 token $\langle t_k \rangle$ 对应时刻

$$
t_k = k \cdot \Delta, \qquad k(t) = \operatorname{round}(t / \Delta), \qquad \text{量化误差} \le \Delta / 2 \qquad (8)
$$

**符号**：$\Delta$：桶宽；词表增加 $\lceil T_{max}/\Delta \rceil$ 行 embedding。生成从「逐位回归」变成「桶分类」：每个时间戳一个 token、一次 softmax 决策；代价是精度天花板 $\Delta/2$ 与量程上限 $T_{max}$ 都写死在词表里。

与 §3 的咬合是设计要点：无论哪套文本方案，监督一致性都来自**文本时钟与位置时钟同源**。Qwen2.5-VL 把绝对时间编码与时戳监督配套——输入侧 $t$-id 按秒推进（式 (4)），训练数据含带时间标注的定位与时戳描述任务（[Bai et al., 2025](../references.md)，[A]，具体数据格式引 tech report 原文）；若位置侧仍用帧序号，文本里同一个「12.5 s」在不同 fps 下对应不同的位置相位差，输出侧方案再好也补不上输入侧的尺子漂移。

## 7. 视频-图像训练数据混合

结构事实先摆出来：在 $(t, h, w)$ 三元组下，**图像就是 $T = 1$ 的视频**——8.2 式 (14) 的图像支是视频支 $\tau_f$ 恒定的退化。同一套管线、同一套位置系统，图像 batch 与视频 batch 可以在一个训练流里混合，无需任何架构改造。LLaVA-OneVision 把这种同构直接用作课程：单图 → 多图 → 视频三阶段（[Li et al., 2024](../references.md)，[A]；阶段划分与数据配比引论文原文）——多图与视频在序列结构上只差「帧间是否有时间语义」，视频阶段因此是在已训好的多图能力上叠加时间监督，而不是从零学视觉。

混合的机制账有两面。视频侧：相邻帧近重复，单位 token 的信息多样性低于图像——视频 token 便宜地提供时间监督（变化、因果、计数），但静态感知监督被稀释；图像侧：每张图独立采样，静态多样性高，是低成本的高密度视觉监督。两边的 ablation 方向一致：LLaVA-OneVision 的课程设计（图像先行打底）与 Apollo 的数据配比实验（[Zohar et al., 2024](../references.md)，[B]）都报告图像-视频混合优于纯视频训练；Apollo 同时给出固定 token 预算下帧数与每帧 token 的权衡方向（§4.3 已引）。定性规则：图像数据保住静态能力与分辨率敏感任务，视频数据提供时间技能，视频占比过高时静态基准回退（Apollo 论文口径的方向性结论，具体配比数字引原表）。

## 8. 视频评测方法论

视频评测比图像评测多一个必须主动防守的失效模式：静态捷径——模型不看时序也能答对。任务类型学先行：

- **整体理解**（MCQA / open-ended QA）：多选题的局限与 8.3 同源（选项偏置、猜测基线），视频版额外叠加静态捷径。
- **时序定位**（temporal grounding）：输出时间段，指标 mIoU 与 Recall@IoU=$\theta$：

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}, \qquad \mathrm{mIoU} = \frac{1}{K}\sum_{k=1}^{K} \mathrm{IoU}(A_k, B_k) \qquad (9)
$$

**符号**：$A, B$：预测与标注的时间段，$|\cdot|$ 为区间长度。
- **事件计数**（event counting）：数「某动作发生了几次」，MAE 计——对 §2 式 (2) 的混叠上限最敏感的探针：采样间隔漏掉的事件，任何下游都数不出来。
- **长视频检索**：needle 型「找瞬间」任务，4.6 §5 长上下文检索方法向时间轴的平移。

静态捷径用两个对照实验防守，应作为任何「时序理解」结论的前置检查：(a) **单帧对照**——只给中间一帧（或任意一帧）回答，若分数不降，题目测的是静态感知而非时间理解；(b) **乱序对照**——保持帧集合不变、打乱时序，若分数不降，模型没有使用时序信息。两个对照不过，「视频理解能力」的声称不成立——与 8.3「claimed vs 真实能力」同一方法论，视频版只是把检查的维度从模态换到时间。

第二道防线是时长分档：claimed vs effective 的视频版（4.6 §5 的 RULER 方法论平移）。模型卡片声明支持 $F_{max}$ 帧 / $T_{max}$ 分钟，不等于各时长档行为达标——按视频时长分档报告性能（Apollo 的评测部分即按分档组织，[Zohar et al., 2024](../references.md)，[B]，论文口径），曲线在哪个档崩坏，effective 视频长度就到哪。评测读数因此永远要配两个脚注：帧是怎么采样的（§2）、每帧多少 token（§1）——同一 benchmark 在不同采样档位下的分数不可比。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 总预算账 | 式 (1) $S_{vid} = F \cdot M$；1min@1fps×256 = 15,360；KV 57,344 B/token（0.88 GB @ 15K） | 视频是把 $T$ 抬进 $10^{4}$-$10^{5}$ 最快的途径；4.6 四堵墙原样适用 |
| 帧采样 | uniform：式 (2) $F \ge T/\delta$；dynamic：式 (3) $fps^{*}$ | 固定 $F$ ⇒ 分辨率随片长退化；固定 fps ⇒ fps 成片长函数，帧序号编码尺子漂移 |
| 绝对时间编码 | 式 (4) $\tau_f = \operatorname{round}(t_f \nu)$、式 (5) 间隔不变性（误差 ≤1 id） | 采样率从位置语义消掉；定位监督一致；$t$ 段高频通道承细粒度时差；节奏可读（Qwen2.5-VL [A]） |
| 时间压缩 | 式 (6) 3D patch（$\tau_p$=2，encoder 也省）；式 (7) 时空 pooling（只省 LLM 侧）；resampler 定长 $B$ | 三位置「谁付钱」对照表（§4 末） |
| 流式长视频 | chunk + 循环 memory（无收录条目，机制描述） | $S_k$ = 4.3 固定状态的语义版；三难点：在线写入取舍 / 漂移 / 评测缺位 |
| 输出格式 | 式 (8) 分桶 special token（误差 ≤ $\Delta/2$）vs 数字文本 | 文本时钟与位置时钟同源是监督一致的根 |
| 数据混合 | 图像 = $T{=}1$ 视频（8.2 式 (14) 退化） | 课程：单图→多图→视频 [A]；混合优于纯视频（Apollo [B] 方向性） |
| 评测 | 式 (9) mIoU / R@IoU；单帧与乱序对照；时长分档 | 静态捷径检查是时序结论的前置；claimed ≠ effective（4.6 平移） |

## 练习

1.（计算）帧预算。context 预算 128,000、文本预留 4,096，视觉段可用 123,904 token。求下列配置下可进 context 的最长视频时长：(a) $M$ = 256、无时间压缩、2 fps；(b) 同 (a) 但输入侧 3D patch $\tau_p$ = 2（每帧 token 减半）；(c) $M$ = 144、0.5 fps。另算 (a) 档下视觉段的 KV cache 占用（Qwen2.5-VL-7B 形态：28 层、4 KV 头、$d_{head}$ = 128、bf16，2.4 式 (9) 口径）。
2.（推导）式 (5) 的误差界与对照。(a) 由 $|\operatorname{round}(x) - x| \le 1/2$ 证明 $|\Delta\tau - \nu\,\Delta t| \le 1$；(b) 写出帧序号规则下同一对事件的 id 差，说明它如何随式 (3) 的 $fps^{*}$ 变化；(c) 设 $\nu\,\Delta t = 3$，构造取整导致 id 差为 2 与为 4 的 $(t_a, t_b)$ 实例各一。
3.（计算）压缩位置的成本账。$F$ = 600 帧、每帧 ViT 前向成本 $E$、LLM 视觉段 attention 成本按二次项主导记 $q(T) = 2T^{2}d$（每层），$M$ = 256、$d$ = 3,584、28 层。比较三方案的视觉相关总成本（encoder 侧 + LLM 视觉段 attention）：(a) 3D patch $\tau_p$ = 2；(b) encoder 后 pooling $c_t$ = 2；(c) 不压缩。指出「encoder 成本 ≫ LLM 视觉段成本」与相反情形各自的最优压缩位置。
4.（代码）教学实现 dynamic FPS 采样器 + 时间戳分桶（教学用，非优化实现）：给定片长 $T$（秒）、$S_{budget}$、$S_{text}$、$M$、$fps_{max}$，按式 (3) 求 $fps^{*}$ 并输出采样帧时刻列表与对应 $t$-id（式 (4)，$\nu$ 为参数）；断言三条：(i) 相邻 id 差 ≈ $\nu / fps^{*}$；(ii) 同一物理事件处在两段不同片长视频里（不同 $fps^{*}$）的 id 差相同（误差 ≤ 1）；(iii) 式 (8) 分桶 bin/unbin 往返误差 ≤ $\Delta/2$。

## 参考

- [Bai et al., 2025](../references.md)：Qwen2.5-VL Technical Report，arXiv:2502.13923，[A]（key: qwen2.5-vl）——§2 dynamic FPS、§3 绝对时间编码、§4.1 temporal patch $\tau_p$ = 2、§6 时戳监督数据口径
- [Wang et al., 2024](../references.md)：Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution，arXiv:2409.12191，[A]（key: qwen2-vl）——§3 帧序号规则对照、M-RoPE 原始出处
- [Li et al., 2024](../references.md)：LLaVA-OneVision: Easy Visual Task Transfer，arXiv:2408.03326，[A]（key: llava-onevision）——§4.2 encoder 后下采样视频管线、§7 单图→多图→视频课程
- [Zohar et al., 2024](../references.md)：Apollo: An Exploration of Video Understanding in Large Multimodal Models，arXiv:2412.10360，[B]（key: apollo）——§4.3/§7 设计空间 ablation 方向性结论、§8 时长分档评测（代码与权重已下架，只引论文结论）
- [Alayrac et al., 2022](../references.md)：Flamingo: a Visual Language Model for Few-Shot Learning，arXiv:2204.14198，[B]（key: flamingo）——§4.3 视频 resampler（$F$ 帧 + 时间位置编码作 K/V）
- [Qwen, Qwen2.5-VL-7B-Instruct config.json](../references.md)：[A]（key: qwen25vl-config）——ctx 128000、28L/4KV/$d_{head}$ 128、mrope_section [16,24,24]、spatial_merge 2
- [Qwen, QwenLM/Qwen3-VL repo](../references.md)：[A]（key: qwen-vl-repo）——dynamic FPS 官方实现（原 Qwen2.5-VL repo 已 301 归并）
- [HuggingFace, transformers Qwen2.5-VL modeling](../references.md)：[A]（key: qwen25vl-modeling）——`get_rope_index` 视频分支走读锚点
- [LLaVA team, LLaVA-NeXT repo](../references.md)：[A]（key: llava-next-repo）——LLaVA-OneVision 视频管线实现口径
