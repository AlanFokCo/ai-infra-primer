# 10.4 Speech LLM 与全双工（Speech LLMs and Full-Duplex Dialogue）

> 前三章把语音的「读、识、写」拆成了独立部件：codec 把波形变成可进 LM 的 token（10.1），CTC/RNN-T/AED 解决对齐（10.2），LM/FM 两条路线解决合成（10.3）。本章把部件装回一个系统：speech LLM 要同时回答「怎么听进去」（encoder + adapter，同构 8.1）、「怎么说出来」（Thinker-Talker 双流、Moshi 的 multi-stream + RQ-Transformer 两级分解 + Inner Monologue——本章深拆的三个机制），以及「对话延迟从哪来」（帧率 × 级数 × vocoder 的逐项预算账，~200 ms 档与秒级档的分界） | 前置：10.1（Mimi split-RVQ 式 (11)(12)、1.1 kbps 账式 (13)、fps vs token rate 约定）、10.2（LLM-based ASR 式 (23)、流式 chunk attention 式 (22)）、10.3（VALL-E 两阶段、FM decoder、BigVGAN）、8.1（connector 三自由度）、8.2（M-RoPE 三元组式 (14)-(17)）、2.4（KV cache 账式 (9)）、4.5（MTP）、7.5（FM） | 锚点：kyutai-labs/moshi（官方全栈，含 Mimi 与 RQ-Transformer 实现）[A]、QwenLM/Qwen2.5-Omni [A]、QwenLM/Qwen3-Omni [A]、zai-org/GLM-4-Voice（原 THUDM）[A]、MoonshotAI/Kimi-Audio [A]、stepfun-ai/Step-Audio [A]、ictnlp/LLaMA-Omni [A]；Moshi [A]、Qwen2.5-Omni [A]、Qwen3-Omni [A]、GLM-4-Voice [A]、AudioLM [B]、Kimi-Audio [A]、Step-Audio [A]、LLaMA-Omni [A]、Qwen2-Audio [A]、GPT-4o voice [C]

记号（沿用本 Part 约定页并本章专用）：时间步 $s = 1..S$ 固定在 12.5 Hz 帧栅格上（帧长 $\Delta_f = 80$ ms，Mimi 口径）； $K = 8$ 为每帧音频 codebook 数；流（stream）用上标： $a^{u}_{s,k}$ = 用户流第 $s$ 帧第 $k$ 个码本、 $a^{m}_{s,k}$ = 模型流、 $w_s$ = 模型文本流（Inner Monologue）；一帧的全流 token 组记 $\mathbf{x}_s$； $\mathbf{x}_{<s} = (\mathbf{x}_1, \ldots, \mathbf{x}_{s-1})$。

## 1. Speech 理解 LLM：encoder + adapter + LLM 的回收

语音理解进 LLM 的组件链在 10.2 §9 已经就位：audio encoder → projector → LLM，与 8.1 的 VLM connector 逐槽同构，识别变成条件文本生成（10.2 式 (23)）。本章只补两笔系统侧的读法。

第一笔是**表征由谁备好**。projector 只做坐标变换（8.1 的 connector 读法原样成立），声学内容必须在 encoder 里备齐——Qwen2-Audio（[Chu et al., 2024](../references.md)，arXiv:2407.10759，[A]）即以 audio encoder + adapter + LLM 的理解端范式确立这个分工，它也正是 Qwen2.5-Omni Thinker 理解端的前代参照；FireRedASR-LLM（10.2 §8）把自家 AED encoder 直接搬进 LLM 款，是同一个赌注的另一家产品形态。

第二笔是**流式从哪一级开始**。理解侧要流式，encoder 的 attention 必须先因果/分块化（10.2 §7 的 chunk attention 账，逐帧延迟落在 $[L, L+C]$ hop 区间内）；LLM 侧的消费天然是自回归流式。也就是说，理解链路的流式瓶颈不在 LLM 而在 encoder 的右上下文——这个判断在 §2 的 Qwen2.5-Omni 音频编码器（2 秒块 block-wise attention）上会看到实例。

## 2. 时间对齐：interleaved chunk 与 TMRoPE

理解（听）与生成（说）装进同一个模型之后，多模态序列的组织从「图像段 + 文本段」变成「按物理时间交错的多条流」。这一节先给输入侧的时间对齐形式化，再给位置编码侧的正式解。

**Interleaved chunk**。设音频帧率 $r_a$、视频帧率 $r_v$（文本在理解侧通常以 transcript 段落参与）。把物理时间轴切成区间 $I_k = [k\Delta,\ (k+1)\Delta)$，序列按区间分块交错排列：

$$\mathrm{seq} = \big[\ \underbrace{a_{k\Delta r_a : (k+1)\Delta r_a}}_{\text{audio}},\ \underbrace{v_{k\Delta r_v : (k+1)\Delta r_v}}_{\text{video}},\ \underbrace{t_{I_k}}_{\text{text}}\ \big]_{k=0,1,\ldots} \qquad (1)$$

**符号**： $a_{i:j}$：第 $i$ 到 $j$ 个音频帧 token（audio 块）； $v$、 $t$：区间 $I_k$ 内的视频帧与文本段； $\Delta$：交错区间长度（实现超参，本章不取具体数值）。式 (1) 的机制含义：同一时间窗内的多模态证据在序列中相邻，跨模态对齐从「全局软注意力里学」变成「局部邻接里读」。

但序列邻接只是叙事顺序，模型感知「同时」要靠位置编码。8.2 §9 的 M-RoPE 把 position id 扩成三元组 $(t, h, w)$：文本三值同值、图像 $t$ 恒定 + $(h, w)$ 网格、视频 $t$ 按帧序递增，rotary 频段按 [16, 24, 24] 切三段各挂一轴（8.2 式 (14)-(17)，score 只含三轴相对位移的分段加权和）。Qwen2.5-Omni 的 **TMRoPE（Time-aligned Multimodal RoPE）** 是其直接扩展（原文：「a novel position embedding approach, named TMRoPE (Time-aligned Multimodal RoPE)」；Qwen3-Omni 沿用，原文：「extends M-RoPE by incorporating absolute temporal information」，[Xu et al., 2025a/2025b](../references.md)，arXiv:2503.20215 / arXiv:2509.17765，均 [A]）。差分要收窄：相对 Qwen2-VL 的 M-RoPE（视频 $t$ 按帧序递增，8.2 §9.1），TMRoPE 把三元组的 $t$ 分量改写为**绝对时间坐标**；相对 Qwen2.5-VL（其视频 $t$ 轴已与绝对秒数对齐，8.2 §9.1），增量在**音频流入轴、音/视共享同一 $t$ 坐标**：

$$t = \left\lfloor \tau / \Delta_0 \right\rfloor \qquad (2)$$

**符号**： $\tau$：该帧的物理时刻； $\Delta_0$：公共时间量子。音频帧与视频帧只要覆盖同一物理时段，就拿到相同的 $t$ 坐标； $(h, w)$ 两轴对音频/文本退化为随 $t$ 同步推进（8.2 的文本一致性推论原样成立：纯文本段 TMRoPE 与 1D RoPE 逐点相同）。

由 8.2 式 (17) 的主命题直接读出推论：两个不同模态的 token（物理时刻 $\tau_1, \tau_2$）在 $t$ 轴频段上的相对旋转角正比于 $\lfloor \tau_1/\Delta_0 \rfloor - \lfloor \tau_2/\Delta_0 \rfloor$——**跨模态相对位置编码的是物理时间差，而不是序列里隔着多少个别的模态的 token**。M-RoPE 解决「一个位置系统装下三种模态」，TMRoPE 再进一步解决「这个位置系统感知同步」；式 (1) 的块交错与式 (2) 的绝对时间轴是同一个设计的序列侧与位置侧两半。

## 3. Qwen2.5-Omni / Qwen3-Omni：Thinker-Talker

### 3.1 Talker 接 hidden：把句子级等待改成 chunk 级等待

级联 speech 对话的延迟结构是硬串行：speech → ASR → LLM → TTS → waveform，后一级等前一级出齐（或至少出端点）。Thinker-Talker（Qwen2.5-Omni，[Xu et al., 2025a](../references.md)，arXiv:2503.20215，[A]；repo [QwenLM/Qwen2.5-Omni](../references.md)，[A]）把中间两级的串行拆开：**Thinker** 是理解 + 文本生成的 LLM（音频输入经 2 秒块 block-wise attention 的音频编码器流式进入，式 (1)(2) 的对齐与位置系统即为此服务）；**Talker** 是语音生成端，原文定义为「a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens」。写成公式：Talker 在第 $s$ 个输出帧的输入是双轨拼接

$$u_s = \big[\thinspace e(w_{\pi(s)}) \thinspace;\hspace{5mu} h^{\text{Thinker}}_{\pi(s)} \thinspace\big], \qquad P\big( z_s \mid z_{<s},\thinspace u_s \big) \qquad (3)$$

**符号**： $w_i$：Thinker 正在生成的第 $i$ 个文本 token； $h^{\text{Thinker}}_i$：其对应的隐层表征； $e(\cdot)$：token embedding； $\pi(s)$：输出帧 $s$ 对齐到的文本位置（chunk 映射）； $z_s$：第 $s$ 帧的 codec code。

双轨的机制理由两条。其一，**延迟**：Talker 消费的是 Thinker 逐 token 产出的 hidden，不等整句——首音帧的等待从「 $U$ 个文本 token 全部解码」降到「首个 chunk 的若干 token + TTFT」（2.4 的 prefill/decode 口径），句子级等待变成 chunk 级等待。其二，**信息**：codec code 要承载韵律与音色走向，纯文本 token 序列里这些信息已被压掉（说「什么」留下了，「怎么说」不在词面）；hidden 是 Thinker 的内部表征，语义与语用上下文未经词表瓶颈——Talker 直读 hidden 等价于把 TTS 的条件从「转写文本」升级成「生成该文本时的完整内部状态」。这正是对级联管线「文本即瓶颈」的结构性回应。

### 3.2 vocoder 对照：block-wise DiT（2.5）vs Code2Wav（3）——一代之隔的 S1 陷阱

两代 Omni 的输出链在 codec code 之后完全分叉，混淆即事实错误，逐字按原文锁定：

- **Qwen2.5-Omni**：codec code →「transformed into a mel-spectrogram using **Flow-Matching**」（**block-wise DiT**，图 4 的 sliding window block attention）→「followed by a modified **BigVGAN**」回波形。两级：FM 负责 code → mel（条件流匹配，机制 recap 7.5），BigVGAN 负责 mel → waveform（snake 抗混叠 vocoder，10.3 §BigVGAN）。流式能力来自 DiT 的块注意力：attention 被限制在滑窗块内，右上下文有界，可以按块出 mel 帧——代价是 FM 的多步采样与两级串联各压一段首包延迟。
- **Qwen3-Omni**（[Xu et al., 2025b](../references.md)，arXiv:2509.17765，[A]；repo [QwenLM/Qwen3-Omni](../references.md)，[A]）：原文「the waveform stage (**Code2Wav**) **replaces block-wise DiT with a lightweight convolutional network (ConvNet)**」——codec code → 轻量因果 ConvNet 直接出波形，mel 中间态与 BigVGAN 第二级一并取消。配套两处：音频 code rate 收到 **12.5 Hz**（「the input and output audio code rates are reduced to 12.5 Hz」），「the output codec enabling **single-frame, immediate** speech synthesis」——因果 ConvNet 无右上下文，单个 80 ms 帧到达即可出声；Thinker-Talker 也升级为 MoE 形态（文本 119 语言）。

一句话锁死对照：**2.5 = block-wise DiT（FM → mel）+ modified BigVGAN（→ waveform）两级；3 = Code2Wav 因果 ConvNet 一级直达**。Qwen3-Omni 的 Talker 生成侧另有一处结构变化：AR 主链逐帧产出 codec 帧，**MTP 模块并行产出其余残差码本**——一帧多码的层间依赖不再逐级自回归，而是用多 token 预测头的并行草稿一次铺出（MTP 机制 recap 4.5；与 §4.3 Moshi 用 depth Transformer 解同一个问题的路线对照，§4.6 收拢）。

### 3.3 参数与延迟的机制注记

Thinker-Talker 相对级联方案的参数账：省掉独立 ASR 与独立 TTS 两份参数，代价是 Thinker 与 Talker 联合训练、Talker 词表扩出音频 code 区。延迟账的定性结论在 §3.1 已给（chunk 级等待），定量要到 §6.2 的预算表里与 Moshi 档逐项对照。

## 4. Moshi 深拆：multi-stream、RQ-Transformer、Inner Monologue

Moshi（[Défossez et al., 2024](../references.md)，arXiv:2410.00037，[A]；官方全栈 [kyutai-labs/moshi](../references.md)，[A]）是本章的深拆范例：它是第一个把「全双工对话」写成分布级假设的开放系统——原文定位「the first multi-stream audio language model, i.e. a model that explicitly processes the input and output audio streams jointly into two autoregressive token streams. This altogether removes the concept of speak[er turns]」，并给出「**theoretical latency of 160ms, 200ms in practice**」的一手数字。三个机制逐个拆：联合分布（§4.2）、两级分解（§4.3）、文本前缀（§4.4）。地基全部已在 10.1 备好：Mimi 12.5 Hz 帧率、split-RVQ $K = 8$（1 语义 VQ + 7 声学 RVQ）、1.1 kbps / 100 tok/s（10.1 式 (11)-(13)）。

### 4.1 设定：12.5 Hz 栅格上的三条流

对话被离散化成以 $\Delta_f = 80$ ms 为格长的同步网格。每个栅格 $s$ 上并行挂着三条流：

- **用户流** $a^{u}_{s, 1:K}$：麦克风音频经 Mimi encoder 实时编码的 $K = 8$ 个码本——对模型是**观测量**（每帧到达即得），不是生成目标；
- **模型音频流** $a^{m}_{s, 1:K}$：模型自己的输出语音， $K = 8$ 个码本，**生成目标**；
- **模型文本流** $w_s$：与模型语音时间对齐的文本 token（Inner Monologue，§4.4），**生成目标**。

记一帧的全流 token 组 $\mathbf{x}_s = \big( a^{u}_{s,1}, \ldots, a^{u}_{s,K},\thinspace w_s,\thinspace a^{m}_{s,1}, \ldots, a^{m}_{s,K} \big)$，共 $Q = 2K + 1 = 17$ 个 token。注意设定里没有任何「轮到谁说」的变量：两条音频流各自在每一帧都必须取某个值（模型侧的「不说话」是一种显式输出状态，§6.1），这就是「removes the concept of speaker turns」在设定层的含义。

```
12.5 Hz 栅格（每格 80 ms）上的三流并行（↓= 该帧生成，● = 输入观测）

  帧 s:      s-2        s-1         s          s+1
 用户流 aᵘ:  ●─────────●──────────●──────────●───▶  （Mimi encoder 实时供给）
 模型文本 w: ··········↓ w_{s-1}──↓ w_s··········   （Inner Monologue，depth 第 0 步）
 模型音频 aᵐ: ··········↓ 8 码本──↓ 8 码本·······   （depth 第 1..8 步）
                          ▲          ▲
                c_{s-1} ──┘    c_s ──┘   c_s = Temporal( x_{<s} )：只含 ≤ s-1 帧
                                           的全流 token（同帧用户码本不进本帧生成）
```

### 4.2 Multi-stream 联合分布

整段对话的联合分布按帧自回归分解。帧内的写法需要先立一条物理约束：**同帧的三条流互为「同时」**——在 80 ms 栅格内，用户帧 $s$ 与模型帧 $s$ 在同一物理时段产生，实时因果下谁也不可能条件于对方的同帧内容。因此帧内取如下分解：用户流因子条件于历史与自身帧内前缀，模型流（文本 → 音频的条件链）只条件于历史：

$$P\big( \mathbf{x}_{1:S} \big) = \prod_{s=1}^{S}\ \underbrace{\prod_{k=1}^{K} P\big( a^{u}_{s,k} \mid \mathbf{x}_{<s},\thinspace a^{u}_{s,<k} \big)}_{\text{user stream (observed)}}\ \cdot\ \underbrace{P\big( w_s \mid \mathbf{x}_{<s} \big)\ \prod_{k=1}^{K} P\big( a^{m}_{s,k} \mid \mathbf{x}_{<s},\thinspace w_s,\thinspace a^{m}_{s,<k} \big)}_{\text{model stream: text prefix, then audio}} \qquad (4)$$

**符号**： $\mathbf{x}_{<s}$： $s$ 帧之前全部流的历史；各因子的具体形式化由 §4.3 的两级分解给出。式 (4) 的帧内结构 = 链式法则 + 同帧跨流独立（物理同时性，非近似妥协）：模型帧 $s$ 的响应最早反映用户帧 $\le s-1$ 的内容，这个一帧反应滞后正是 §6.2 预算表「2 帧 = 160 ms」的结构来源；流间 delay（§4.5）整体平移各流栅格后，有限的同物理时刻跨流可见性会被恢复——式 (4) 是零 delay 基准形态。口径注记：原文 Eq (6) 的深度序把用户流排在模型流之后（训练口径，配 delay $\tau$）；式 (4) 取部署相关的条件链重排——模型流不读同帧用户流的结论在两口径下相同。

三个读法。其一，**对话 = 一个联合分布**：turn-taking、抢话、backchannel（边听边「嗯」）、重叠发言全部是这个分布里的事件模式，不是外加的控制逻辑——与半双工的形式对照在 §6.1 展开。其二，**部署时用户流因子整块退掉**： $a^{u}$ 由 Mimi encoder 供给，需要的只是条件部分 $P( w_s, a^{m}_{s,1:K} \mid \cdot)$；训练时保留用户流因子（双流同建，原文「jointly into two autoregressive token streams」）是表征塑形的免费监督，且让预训练可以退化为单流音频 LM（§4.5）。其三，**帧内序 $w$ 在 $a^m$ 前不是随意选择**——它就是 Inner Monologue 的条件链（§4.4）。

### 4.3 RQ-Transformer：时间 × 深度的两级分解（本章核心推导）

式 (4) 若朴素实现，要把每帧 $Q$ 个 token 展平成一条长 $Q \cdot S$ 的序列过标准 causal Transformer——长上下文成本与 cache 都乘上 $Q$ 倍（账在下面）。RQ-Transformer（Moshi 论文 §3.4.1；谱系：RQ-Transformer 出自 Lee et al. 2022 的二维 token 网格生成，层次化序列分解思想同族于 MegaByte（Yu et al. 2024）——Moshi 论文转引）把二维网格（时间 $s$ × 深度 $k$）拆成两级模型。原文定义：「breaks down a flattened sequence of length $K \cdot S$ into $S$ timesteps for a large **Temporal Transformer** which produces a context embedding used to condition a smaller **Depth Transformer** over $K$ steps」（引句的 $K$ 为全流子序列总数 = 本章 $Q$ = 17，记号映射见式 (6) 后的计数口径说明）。

**第一级：Temporal Transformer**（大模型，Helium 7B 文本 LLM 初始化，2.1T tokens 文本预训练）。帧 $s$ 的输入是该帧已齐 token 的 embedding 求和：

$$v_s = \sum_{k=1}^{K} E^{u}_{k}\big[ a^{u}_{s,k} \big] + E^{w}\big[ w_s \big] + \sum_{k=1}^{K} E^{m}_{k}\big[ a^{m}_{s,k} \big], \qquad h_s = f_{\text{temp}}\big( v_{\le s} \big)_s, \qquad c_{s+1} = W_o\thinspace h_s \qquad (5)$$

**符号**： $E^{u}_k, E^{m}_k \in \mathbb{R}^{2048 \times d}$：用户/模型流第 $k$ 个码本的 embedding 表（码本大小 2048，Mimi 口径）； $E^w$：文本词表 embedding； $f_{\text{temp}}$：帧级因果 Transformer； $c_{s+1}$：供第 $s+1$ 帧深度生成的上下文向量。求和嵌入的机制含义：一帧之内各流各码本的信息先被压成一个帧级向量——**帧内结构不进时间轴的 attention**。

**第二级：Depth Transformer**（小模型，原文口径 smaller）。帧 $s$ 内自回归走 $Q' = K + 1 = 9$ 步，条件于 $c_s$：

$$g_{s,0} = c_s, \qquad g_{s,j} = f_{\text{depth}}\big( c_s,\thinspace e_{s,0}, \ldots, e_{s,j-1} \big), \qquad P\big( x_{s,j} \mid \cdot \big) = \mathrm{softmax}\big( W_j\thinspace g_{s,j} \big) \qquad (6)$$

**符号**：深度序固定为 $x_{s,0} = w_s$（文本第 0 步）、 $x_{s,j} = a^{m}_{s,j}$（ $j = 1..K$）； $e_{s,j}$：深度第 $j$ 步已生成 token 的 embedding； $W_j$：第 $j$ 类 token 的输出头（文本头与各码本头分开）。

计数口径说明（防与原文「over $K$ steps」对不上）：原文的 $K$ 是**全流子序列总数**——§3.4.2 Eq (6) 明确「a total number of $K = 2Q + 1$ streams, with $Q = 8$」即 $K = 17$（其 $Q$ = 每流码本数，对应本章 $K$ = 8；含用户流 8 条——训练时用户流也建模，部署时只喂不采），「the number of steps in the Depth Transformer is at most $K$」按 17 计。本章的 8 步口径对应**官方实现**（kyutai-labs/moshi `lm.py` 实测：`dep_q = 8`、`n_q = 16`，文本由 `text_linear` 从 temporal 输出直出、用户流推理只喂不采）：Depth Transformer 本体只跑 Moshi 侧 8 个码本步，文本步是 $c_s$ 上的线性头——本章 $Q' = 9$（1 文本 + 8 码本）由此而来。

把式 (5)(6) 代回式 (4)，模型流的条件因子全部换成「经 $c_s$ 瓶装」的形式：

$$P\big( w_s, a^{m}_{s,1:K} \mid \mathbf{x}_{<s} \big) \hspace{5mu}\longrightarrow\hspace{5mu} P\big( w_s \mid c_s \big) \prod_{k=1}^{K} P\big( a^{m}_{s,k} \mid c_s,\thinspace w_s,\thinspace a^{m}_{s,<k} \big) \qquad (7)$$

**推导的两半要点**。帧内一半（式 (6) 右端）就是链式法则对固定深度序的逐字展开，**无近似**；跨帧的一半（ $\mathbf{x}_{<s} \to c_s$）才是近似：全部历史对当前帧的影响被强制流过单个向量 $c_s$——这是用「历史压缩成瓶颈向量」换「时间轴上每帧只走一步」的核心交易（练习 1 给信息论口径的表述）。

```
RQ-Transformer 一帧的生成流（帧 s；c_s 由 §4.3 式 (5) 在帧 s-1 末备好）

 c_s ──▶ Depth step 0:  P(w_s | c_s)                    ── 文本头
          │  e(w_s) ↘
          ├─▶ Depth step 1:  P(aᵐ_{s,1} | c_s, w_s)     ── 码本 1 头
          │  e(aᵐ_{s,1}) ↘
          ├─▶ ……逐级自回归……
          └─▶ Depth step 8:  P(aᵐ_{s,8} | c_s, w_s, aᵐ_{s,<8})
                          │
 帧末：v_s = Σ 全流 embedding（含本帧用户码 aᵘ 与刚生成的 w_s, aᵐ）
        h_s = Temporal(v_{≤s})，c_{s+1} = W_o h_s ──▶ 喂给下一帧
```

**效率账（推导重点）**。比较对象：展平基线 = 同参数量的标准 causal Transformer 直接建模长度 $Q S$ 的展平序列。逐帧稳态 decode 成本（attention 内积项， $L$ 层、时/深宽度 $d_t / d_d$ 分别记；FLOPs ∝ 内积次数 × 宽度，score 与加权和同阶、常数因子略）：

| 项 | 展平基线 | RQ 两级 |
| --- | --- | --- |
| 每帧生成步数 | $Q = 17$ 个 token | Temporal 1 步 + Depth $Q' = 9$ 步 |
| 每步 attention 读的 cache | 全部历史 $\sim Q S$ 条 | Temporal $\sim S$ 条；Depth $\le Q'$ 条 |
| 每帧 attention FLOPs 量级（内积 × 宽度） | $\propto L \cdot Q \cdot (Q S) \cdot d = L Q^2 S\thinspace d$ | $\propto L \big( S\thinspace d_t + Q'^2 d_d \big)$ |
| 整段 $S$ 帧累计 | $O\big( L Q^2 S^2 d \big)$ | $O\big( L S^2 d_t + L S Q'^2 d_d \big)$ |
| KV cache 增长（2.4 式 (9) 账） | 每帧 $+Q$ 条 | Temporal 每帧 $+1$ 条；Depth cache 帧内 $\le Q'$ 条、帧末作废 |

两点结论。**其一，长上下文因子从 $Q^2$ 降回 $1$**：展平基线的随 $S$ 增长项携带 $Q^2 = 289$ 倍乘子，RQ 把它拆成「时间轴 $S$ 项 × 1 倍」加「帧内 $Q'^2$ 的常数项」——后者不随对话长度增长。代入具体数：10 分钟对话 $S = 7500$ 帧，展平基线每帧读约 $17 \times 7500 \approx 1.3 \times 10^5$ 条 cache，RQ 时间轴读 $7.5 \times 10^3$ 条——长上下文读取量 $17 \times$ 差，且 cache 总增长也慢 $17 \times$。**其二，帧内结构成本被移进小模型**：Depth Transformer 承担每帧 9 步，但它是「smaller」档，9 步浅层前向在 80 ms 的帧预算内是廉价的；大模型的宽度只按帧率（12.5 Hz）计费，不按 token rate（100 tok/s + 文本）计费。这就是「80 ms 一帧出 17 个 token 还能实时」的算术来源。

**参数账（结构比例）**。三块：(i) Temporal——Helium 7B 初始化，是绝对大头（原文口径；7B 级 LLM 的典型宽度 $d_t$ 在 4096 档）；(ii) Depth——原文给全规模（§3.4.1 正文：「As detailed in Table 1, our Depth Transformer has 6 layers, a dimension of 1024, and 16 attention heads」；Table 1 同载 MLP 4096）：按每层 $4 d^2 + 2 d\thinspace d_{\text{mlp}}$（attention 四矩阵 + MLP 上下投影）估， $6 \times (4 \times 1024^2 + 2 \times 1024 \times 4096) \approx 7.5 \times 10^7$——约 75M，恰为 Temporal 7B 的约 1%，「smaller」是百倍级的小，效率表里的 $d_d$ 由此钉在 1024；(iii) embedding + 输出头表：16 个音频码本（双流 × $K = 8$）× 2048 entry 的输入表、文本词表的输入/输出表、各码本输出头——音频表合计约 $16 \times 2048 \times d_t \approx 3.3 \times 10^4 \times d_t$ 参数， $d_t = 4096$ 档约 1.3 亿，相对 7B 主干是百分位级。结构判断：RQ 的账本质是「把 token rate 的税从宽度上转移到深度（层数极少的小模型）上」——75M 对 7B 的 1% 配比把这笔记账直接坐实：codebook 越多、帧率越低，这笔转移越划算；反方向（ $K = 1$ 单码本）则根本不需要 Depth 级，§5.2 的 GLM-4-Voice 正是那个极端点。

### 4.4 Inner Monologue：时间对齐的文本前缀

式 (4)(7) 里 $w_s$ 在帧内深度序的第 0 位，这不是书写习惯而是条件结构：模型音频流的每一个码本都以同帧文本 token 为条件。原文：「first predict **time-aligned text tokens as a prefix** to audio tokens. Not only this 'Inner Monologue' method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide **streaming speech recognition and text-to-speech**」。

训练数据的构造是把转写文本强制对齐上 12.5 Hz 栅格：第 $i$ 个词（或子词）按对齐时间戳落入帧 $\phi(i)$，无词帧填 pad：

$$w_s = \begin{cases} y_i, & s = \phi(i) \\ \varnothing, & \text{otherwise} \end{cases} \qquad (8)$$

**符号**： $y_i$：文本序列第 $i$ 个 token； $\phi$：词到帧的对齐映射（ $s = \phi(i)$ 即词 $i$ 的对齐帧，其余帧取 pad）； $\varnothing$：pad 符（沿用全书空位记号）。训练目标就是式 (4) 各因子的逐帧交叉熵之和——文本位置与音频位置同为离散词表上的 CE（1.1 的机制原样生效），联合目标无需任何新组件。

机制上文本前缀为什么提升语音质量：没有 $w_s$ 时， $P(a^{m}_{s,k} \mid c_s, a^{m}_{s,<k})$ 必须隐式地同时做「内容规划」（这 80 ms 说哪个音）与「声学实现」（这个音的码本怎么取）；文本前缀把内容规划变成条件链上的显式变量——训练时它是观测量（式 (8)），推理时它由文本头先采样、音频步再以其为条件。两笔收益同出一个结构：(i) 音频码本分布的条件熵下降——声学实现只需在「内容已定」的子空间里建模；(ii) 文本头直接继承 Helium 7B 的文本预训练先验（2.1T tokens）——语言质量的瓶颈从「音频 token 分布里自学语言学」换成「文本 LLM 的现成先验」。与 10.3 的对照：VALL-E 用音素序列做条件、CosyVoice 用监督语义 token——Inner Monologue 是同一个「内容先行」思想在全双工设定里的版本，差别只在条件变量换成了模型自己生成的文本流。

### 4.5 Delay、流式副产品与训练阶段

**层级/流间 delay**（Moshi 论文表 1 的 Acoustic delay / Text delay 超参；延迟交错的并行预测模式转引自 Copet et al. 2023）。机制：流 $j$ 相对实时栅格整体平移整数帧 $d_j$——模型在绝对帧 $s$ 实际预测的是该流的第 $s - d_j$ 帧。两个用途。(i) 并行化：多码本逐层错开 delay 后，同一绝对时刻各层可并行采样（层 $k$ 在帧 $s$ 预测时，层 $<k$ 的「未来」帧已在更早的绝对帧生成）——这是没有 Depth Transformer 时解「一帧多码」的替代方案，与 §4.3 的深度自回归构成同一问题的两种答案。(ii) 读法翻转：文本流相对音频流的 delay 方向决定这个联合模型的「用法」——文本超前（先出文本、音频跟随）即流式 TTS 读法；文本滞后（音频先行、文本跟随产出）即流式 ASR 读法。原文把这两种读法作为 Inner Monologue 的直接副产品演示——同一个联合分布、同一套参数，只是把哪条流当「输入」哪条当「输出」。

**训练阶段**（原文口径）：Helium 7B（2.1T tokens 文本预训练）为基座，随后四阶段——无监督音频预训练（单流音频 LM）→ simulated multi-stream（单声道素材人工构造双流）→ Fisher 语料全双工微调 → 后续阶段。这条链的结构读法：前两阶段解决「音频 token 上的 LM 能力」与「双流格式」，Fisher 阶段才把「对话动力学」（重叠、backchannel、话轮交接的统计模式）灌进联合分布——§6.1 的「turn 作为分布事件」就是在这里学出来的。

### 4.6 与 Qwen3-Omni MTP 路线的对照

「一帧 $K$ 码」的两个解至此可以并排：Moshi 用 Depth Transformer 在帧内自回归（式 (6)，9 步小模型）；Qwen3-Omni 用 MTP 头并行铺出残差码本（§3.2，主链 1 步 + 并行草稿）。前者的帧内依赖是精确的链式分解、代价是 $Q'$ 步串行；后者用一步并行换层间依赖的近似（MTP 头对远层码本的条件更弱，4.5 的接受率账同族）。共同前提都是低帧率：12.5 Hz 把「每帧内部结构」的预算放宽到 80 ms——帧率选择（10.1 的 fps vs token rate 账）在这里兑现为生成架构的自由度。

## 5. 分层与单流：AudioLM、GLM-4-Voice 与一段式系统

### 5.1 AudioLM：semantic → coarse → fine 的分层生成

AudioLM（[Borsos et al., 2022](../references.md)，arXiv:2209.03143，[B]——Google 未开源，按论文拆）把音频生成分成三级自回归，原文：「first construct coarse **semantic tokens** from a model pre-trained with a self-supervised masked language modeling objective」（w2v-BERT 系，10.1 §3.2），再接 SoundStream 系声学 token 的 coarse → fine 两级：

$$P = \underbrace{\prod_t P\big( s_t \mid s_{<t} \big)}_{\text{semantic}}\ \times\ \underbrace{\prod_t P\big( c_t \mid c_{<t},\thinspace s \big)}_{\text{coarse}}\ \times\ \underbrace{\prod_t P\big( f_t \mid f_{<t},\thinspace c,\thinspace s \big)}_{\text{fine}} \qquad (9)$$

**符号**： $s_t$：semantic token（语义层）； $c_t, f_t$：SoundStream 系 coarse / fine acoustic token（粗声学层 = 前几个码本、细声学层 = 其余码本）。

分层的机制读法与 §4.4 同源：长程一致性（不漂移的「内容骨架」）由低帧率、分布近文本的语义层承担——纯声学 token 上的无约束 AR 容易出无意义哼鸣，语义条件把声学层的建模压回「实现」；semantic vs acoustic 的能力分工表见 10.1 §6.2。与 Moshi 的关系是问题结构的两次回答：AudioLM 把「一帧多码」的难点用**帧外分层**（级间条件）解决，代价是三次串行生成、无流式对话能力；Moshi 用**帧内深度**（RQ）解决，保住 12.5 Hz 的实时栅格。

### 5.2 GLM-4-Voice： $K = 1$ 的删繁路线

GLM-4-Voice（[Zeng et al., 2024](../references.md)，arXiv:2412.02612，[A]；repo [zai-org/GLM-4-Voice](../references.md)，[A]——THUDM 已 301 重定向至此）把 10.1 的 token 三旋钮拧到极简：单码本、12.5 Hz、175 bps 的**监督**语音 tokenizer（原文：「ultra-low bitrate (175bps), single-codebook speech tokenizer with 12.5Hz frame rate derived from an ASR model by incorporating a vector-quantized bottleneck into the encoder」）。bitrate 账按 10.1 式 (7) 复算： $175 / 12.5 = 14$ bit/帧 → $K = 2^{14} = 16384$， $1 \times 14 \times 12.5 = 175$ bps ✓；token rate = 12.5 tok/s。

结构推论直接且重要：**$K = 1$ 时「一帧多码」问题整个消失**——没有层级依赖，RQ 的 Depth 级与 delay 交错都不需要，音频 token 流与文本 token 按块交错（文本/语音交错 token 的流式形态）即可交给标准 decoder-only 逐 token 建模。交错的具体形态是式 (1) 在 $K = 1$、 $r_a = 12.5$ Hz 的实例化——文本块与单码本音频 token 块按时间窗 $I_k$ 交错排布：

$$\mathrm{seq} = \big[\thinspace t_{I_k},\ a_{k\Delta r_a : (k+1)\Delta r_a} \thinspace\big]_{k=0,1,\ldots}, \qquad r_a = 12.5\ \text{Hz} \qquad (10)$$

**符号**： $t_{I_k}$：落在时间窗 $I_k$ 内的文本 token 段； $a_{i:j}$：同窗内的音频 token 段（每 80 ms 一个 token，每窗 $\Delta \times 12.5$ 个）。流式读法：音频块逐窗到达即消费、生成侧按窗交错产出文本与音频——块长 $\Delta$ 就是延迟粒度旋钮。生成端回波形用 flow-matching decoder（FM recap 7.5）。代价写在 10.1 的账里：单码本 14 bit 要同时装内容与声学细节，重建保真的冗余度低于多码本方案——tokenizer 的 ASR 蒸馏出身（监督语义路线，同 10.1 §3 与 10.3 S3Tokenizer 一脉）正是用「内容优先」对冲这个瓶颈。与 Moshi 并排：同帧率（12.5 Hz）、同联合建模目标，结构差异全部来自 $K = 1$ vs $K = 8$ 这一个旋钮。

### 5.3 一段式三例：Kimi-Audio / Step-Audio / LLaMA-Omni

**Kimi-Audio**（[KimiTeam, 2025](../references.md)，arXiv:2504.18425，[A]；repo [MoonshotAI/Kimi-Audio](../references.md)，[A]）：开源音频基座，理解 + 生成 + 对话一体；预训练音频数据超 **13 million hours**；从预训练文本 LLM 初始化（与 Moshi 的 Helium 基座同一策略——音频能力嫁接在文本先验上）；生成端回波形用 flow-matching detokenizer。

**Step-Audio**（[Huang et al., 2025](../references.md)，arXiv:2502.11946，[A]；repo [stepfun-ai/Step-Audio](../references.md)，[A]）：**130B** 参数的统一语音-文本多模态模型（开源款为 Step-Audio-Chat），配生成式语音数据引擎——统一理解/生成的「大基座 + 数据工程」路线代表。

**LLaMA-Omni**（[Fang et al., 2024](../references.md)，arXiv:2409.06666，[A]；repo [ictnlp/LLaMA-Omni](../references.md)，[A]）：低延迟 speech→speech 直连的最小链——原文「integrates a pretrained **speech encoder, a speech adaptor, an LLM, and a streaming speech decoder**. It **eliminates the need for speech transcription**, and can simultaneously generate text and speech responses directly from speech instructions」。机制注记：省掉转写中间态意味着理解侧不经过文本瓶颈（§3.1 的「文本即瓶颈」在输入侧的镜像），生成侧用 streaming speech decoder 直接消费 LLM 隐状态——与 Thinker-Talker 的「直读 hidden」是同族判断的独立实例。

## 6. Full-duplex 数学与延迟预算

### 6.1 形式化：turn-taking 假设的有与无

**半双工（级联/turn-based）**。响应是对「已完成用户话语」的一次性条件生成：

$$P\big( a^{m} \mid a^{u}_{\text{completed}} \big) \qquad (11)$$

**符号**： $a^{u}_{\text{completed}}$：已完成的整段用户话语。「轮到谁」由系统逻辑裁决——生成由端点检测器的外部信号触发，模型本身不建模重叠与抢话；对话动力学全部在分布之外。

**全双工（Moshi 式）**。输出分布必须每帧取值： $w_s$ 与 $a^{m}_{s,1:K}$ 的词表里，「这一帧不说话」是一个显式状态（文本 pad $\varnothing$、音频的静音帧——Mimi 对静音段的常规编码），不是「不输出」。always-listen = 用户流每帧照常进（式 (5) 的 $v_s$ 含 $a^{u}$ 项）；always-talk = 模型流每帧照常生成，多数帧采到静音。于是话轮交接、backchannel、打断全部成为式 (4) 联合分布里的事件统计——没有 turn 变量，turn 结构是学出来的边缘行为。训练侧的两个支点已在 §4.5：静音/pad 状态把「不说」变成可监督的类别；Fisher 全双工微调把真实对话的重叠统计灌进分布。

### 6.2 延迟预算账：~200 ms 逐项

端到端延迟的定义取「用户语音某帧进麦克风 → 模型对该帧的声学响应出扬声器」。Moshi 原文一手数字：「theoretical latency of 160ms, 200ms in practice」——理论值恰为 2 个 80 ms 帧，预算表逐项拆如下（帧率/单帧口径为原文事实；其余环节数值为工程常识区间的**教学估算**，非 Moshi 测量值）：

| 环节 | 预算 | 口径 |
| --- | --- | --- |
| 采集与帧缓冲 | 80 ms | 1 帧时长 $\Delta_f$（12.5 Hz，Mimi 帧口径，原文） |
| Mimi encoder | ~10–20 ms | 流式卷积 + RVQ，教学估算 |
| Temporal + Depth（9 步） | ~30–50 ms | 1 帧级步 + $Q'$ 步小模型，教学估算 |
| Mimi decoder | ~10–20 ms | 流式反卷积，教学估算 |
| 网络/调度/播放缓冲 | ~10–30 ms | 教学估算 |
| **合计** | **≈160 ms（理论）/ 200 ms（实测）** | 合计锚定原文数字 |

理论 160 ms 的结构读法：第一帧（80 ms）用来攒齐一帧用户音频并完成编码，第二帧（80 ms）内完成 LM 两级推理与解码出声——2 帧 = 帧率选型的直接后果。这张表同时解释了为什么各家拼命压帧率：帧率砍半（如 25 Hz → 12.5 Hz），「攒帧」一项立省一整个帧长；而帧率一旦选定，LM 每帧的步数预算（RQ 的 $Q'$ 步、Qwen3 的单帧即时）就被 80 ms 钉死。

半双工级联的对照账（同为教学估算）：端点检测（等用户说完的静音判据）300–700 ms + LLM TTFT 200–400 ms + TTS 首包 100–300 ms，合计**秒级**。逐段对照可见差距的来源不是某个环节更慢，而是串行结构本身：级联把「等整句」重复了三次（ASR 等端点、TTS 等文本、播放等首包），全双工把等待压到一个帧栅格。工程含义反过来也成立：帧栅格之外的开销（网络、调度、缓冲）在 200 ms 预算里已占可感比例——继续压帧率的边际收益会被这些固定项吃掉。

## 7. 离散 token vs 连续特征进 LLM：adapter 选择分析

10.1 §6.3 埋的形态选择问题，在本章各系统上收成全表：

| 维度 | 离散 codec token 进 LLM | 连续特征 + adapter 进 LLM |
| --- | --- | --- |
| 输入接口 | 扩词表，复用 embedding/softmax/KV cache 全套（1.1、2.4 基建原样生效） | projector 投影进 embedding 空间（8.1 connector 同构，式 (2)） |
| 信息上限 | 硬性： $n_q \log_2 K \cdot fps$（10.1 式 (7) 与 8.5 同账） | 无硬上限，取决于 encoder 与 adapter 训练 |
| 对齐机制 | 无需跨模态对齐（token 即词表成员）；代价是 codec 瓶颈损失不可恢复 | 连续-离散空间对齐由训练学出；多一道速率转换（帧堆叠，10.2 §9） |
| 生成方向 | 原生：输出头直接吐 code → codec decoder | 不可原生：softmax 吐不出连续向量，仍要回到离散 codec 或扩散头（7.5） |
| 流式延迟结构 | 帧率定栅格，生成架构（RQ/MTP/单码本）定帧内步数 | 理解侧看 encoder 块长与 adapter 下采样；生成侧仍归左列 |
| 代表取点 | Moshi（双向全离散）[A]、GLM-4-Voice（双向全离散， $K = 1$）[A] | Qwen2.5/3-Omni（连续进、离散出）[A]、LLaMA-Omni（连续进）[A]、Kimi-Audio（FM detokenizer 出；输入侧形态 sources 未收）[A] |

读表一句话：当代系统的收敛解几乎都是**混合**——理解侧走连续（保信息、复用强 encoder），生成侧走离散（复用 softmax 生成链、帧率可控）；全离散（Moshi、GLM-4-Voice）用 tokenizer 质量换结构统一，全双工与低帧率的收益已在 §4、§6 结清。adapter 的选择因此不是「哪种更好」，而是「信息瓶颈放在 tokenizer 里还是 alignment 训练里」——前者损失在编码时一次性付清且可量化（bitrate 账），后者损失在训练里分期支付且难审计。

## 8. GPT-4o voice：闭源参照 [C]

GPT-4o 的语音交互为闭源系统，官方公开披露极少。按证据等级 [C] 的纪律：只陈述已公开定位（原生语音输入输出的端到端实时交互形态），不引未披露的数字、不推测架构细节为事实。分析上可把它放回本章坐标系：本章建立的三个轴——理解侧接口（连续/离散）、一帧多码的解（RQ/MTP/单码本）、延迟预算（帧率 × 级数 × vocoder）——是任何同形态系统的必要设计点；闭源系统在这三个轴上的取点未知，但轴本身与开放系统共享。它存在的参照价值即在于此：§6.2 的预算表与 §7 的对照表给出了检验任何「实时语音对话」主张的算术框架——声称的延迟数字必须能在帧栅格与级数账上复算，复算不出的部分即披露缺口。

## 总结

| 主题 | 公式 / 数值 | 要点 |
| --- | --- | --- |
| 时间对齐 | 式 (1) 块交错；式 (2) TMRoPE 绝对时间轴 $t = \lfloor \tau/\Delta_0 \rfloor$ | 8.2 M-RoPE 三元组的 $t$ 分量改为物理时间；跨模态相对位置 = 时间差 |
| Thinker-Talker | 式 (3) 双轨 $u_s = [e(w_{\pi(s)}); h^{\text{Thinker}}_{\pi(s)}]$ | Talker 直读 hidden：句子级等待 → chunk 级等待；绕过文本瓶颈 |
| Qwen vocoder 对照 | **2.5 = block-wise DiT（FM→mel）+ modified BigVGAN；3 = Code2Wav 因果 ConvNet 一级直达** | 12.5 Hz code rate + 单帧即时合成（3 代）；混淆即 S1 |
| Multi-stream 联合分布 | 式 (4)：帧级 AR × 帧内「用户码→文本→模型码」固定序 | 无 turn 变量；部署时用户流因子退化为观测条件 |
| RQ-Transformer | 式 (5) 求和嵌入 + 式 (6) 深度 9 步 + 式 (7) 瓶颈条件 | 长上下文因子 $Q^2 \to 1$（10 min： $1.3 \times 10^5 \to 7.5 \times 10^3$ 条 cache 读）；cache 增长慢 $17\times$ |
| Inner Monologue | 式 (7) 文本第 0 位 + 式 (8) 强制对齐铺 pad | 内容规划显式化；文本头继承 Helium 7B 先验；delay 翻转得流式 ASR/TTS |
| AudioLM 分层 | 式 (9)：semantic → coarse → fine | 帧外分层解「一帧多码」；[B] 无官方代码 |
| GLM-4-Voice | 175 bps = $1 \times 14 \times 12.5$， $K = 16384$；式 (10) 交错实例化 | $K = 1$ 删去 Depth/delay 问题；FM decoder 回波形 |
| Full-duplex | 式 (11) 半双工对照；静音为显式输出状态 | 预算表：2 帧 = 160 ms 理论 / 200 ms 实测（原文）；级联秒级（教学估算） |
| 离散 vs 连续 | §7 表 | 混合为收敛解：连续进、离散出；瓶颈在 tokenizer 还是 alignment |

## 练习

1.（推导）Multi-stream 联合分布与瓶颈。从展平序列 $(x_{1,1}, \ldots, x_{S,Q})$ 的链式法则出发：(a) 不重排 token 顺序，把 $\prod_{j} P(x_{s,j} \mid x_{<s,j})$ 按帧分组、再按 §4.2 的帧内固定序展开，在 §4.2 的同帧跨流独立前提下证明式 (4) 由展平链式法则精确推出，并指出该前提在哪一步进入（哪两个因子的条件集各被删去哪一项）；(b) 设 $\mathcal{H}_s$ 为「 $\mathbf{x}_{<s}$ 中且与 $x_{s,j}$ 相关」的全部变量，说明式 (7) 等价于引入瓶颈 $c_s$ 使 $\mathcal{H}_s \to c_s \to x_{s,j}$ 成马尔可夫链，并用互信息给出这个近似的上界表述（提示： $I(x_{s,j}; \mathcal{H}_s \mid c_s) = 0$ 当且仅当 $c_s$ 是充分统计量）；(c) 若帧内文本步移到音频步之后（深度序反转），联合分布的哪几个因子改变？这个改动在生成侧的语义后果是什么（结合 §4.4 回答）。

2.（计算）延迟预算账。(a) Moshi 帧率若从 12.5 Hz 翻倍到 25 Hz，只考虑「攒帧 + 单帧生成」两项，理论端到端延迟变为多少？(b) 某级联系统：ASR chunk 320 ms（块首上界口径，10.2 §7）、LLM TTFT 250 ms、TTS 首包 150 ms、网络 50 ms，逐项给出端到端预算并与 Moshi 档对照；指出哪一项无法靠加 GPU 消掉、为什么。(c) Qwen3-Omni 档（12.5 Hz code、单帧即时合成）：只计入「首帧到达 + 首帧生成」的结构下界是多少 ms？与 (a) 的 25 Hz 档比较并解释差异来源。

3.（计算 + 伪码）RQ-Transformer 账。 $S = 7500$ 帧（10 min）、 $Q = 2K + 1 = 17$、 $Q' = K + 1 = 9$：(a) 展平基线与 RQ 每帧 attention 读取的 cache 条数各是多少（用 §4.3 表内公式），整段对话的 cache 峰值各多少条？(b) 写出单帧生成的教学伪码（temporal 一步、depth $Q'$ 步、帧末求和嵌入回写），标出每一步的因果依赖必须与 §4.3 一致；(c) 若码本数 $K$ 从 8 减到 4（其余不变），两项账如何变？ $K = 1$ 时结构退化为什么（联系 §5.2）？

4.（分析）TMRoPE 与 interleave。给定音频 $r_a = 25$ fps、视频 $r_v = 2$ fps、区间 $\Delta = 2$ s：(a) 按式 (1) 写出 $k = 0$ 区间的 token 排列；(b) 取 $\Delta_0 = 0.04$ s（音频帧长），给出该区间内音频第 3 帧与视频第 1 帧的 TMRoPE $t$ 坐标（式 (2)），并说明在 M-RoPE（视频 $t$ 按帧序递增）下两者的 $t$ 坐标会是什么——哪种编码保留了「音画同步」的几何事实？(c) 若 $\Delta_0$ 改取 0.5 s（视频帧长量级），同步粒度发生什么变化？给出选型建议。

## 参考

- [Défossez et al., 2024](../references.md)：Moshi: a speech-text foundation model for real-time dialogue，arXiv:2410.00037，[A]（key: moshi）——式 (4)-(8)：multi-stream 联合分布、RQ-Transformer 两级分解（§3.4.1 原文口径）、Inner Monologue、Helium 7B 基座（2.1T tokens）、四阶段训练、Acoustic/Text delay（表 1）、160 ms 理论 / 200 ms 实测；Mimi 部分 recap 10.1
- [Xu et al., 2025a](../references.md)：Qwen2.5-Omni Technical Report，arXiv:2503.20215，[A]（key: qwen2.5-omni）——TMRoPE、Thinker-Talker 双轨（式 (3)）、block-wise DiT（FM→mel）+ modified BigVGAN、音频编码器 2 秒块
- [Xu et al., 2025b](../references.md)：Qwen3-Omni Technical Report，arXiv:2509.17765，[A]（key: qwen3-omni）——Thinker-Talker MoE、MTP 产残差码本、Code2Wav 因果 ConvNet、12.5 Hz code rate 与单帧即时合成、119 语言
- [Zeng et al., 2024](../references.md)：GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot，arXiv:2412.02612，[A]（key: glm-4-voice）——175 bps 单码本 12.5 Hz 监督 tokenizer、FM decoder、文本/语音交错流式
- [Borsos et al., 2022](../references.md)：AudioLM: a Language Modeling Approach to Audio Generation，arXiv:2209.03143，[B]（key: audiolm）——semantic → coarse → fine 分层生成（式 (9)）；Google 未开源
- [KimiTeam, 2025](../references.md)：Kimi-Audio Technical Report，arXiv:2504.18425，[A]（key: kimi-audio）——13M 小时预训、FM detokenizer、文本 LLM 初始化
- [Huang et al., 2025](../references.md)：Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction，arXiv:2502.11946，[A]（key: step-audio）——130B 统一语音-文本模型、Step-Audio-Chat
- [Fang et al., 2024](../references.md)：LLaMA-Omni: Seamless Speech Interaction with Large Language Models，arXiv:2409.06666，[A]（key: llama-omni）——speech encoder + adaptor + LLM + streaming speech decoder、免转写直连
- [Chu et al., 2024](../references.md)：Qwen2-Audio Technical Report，arXiv:2407.10759，[A]（key: qwen2-audio）——audio encoder + adapter + LLM 理解端范式
- 锚点 repo：[kyutai-labs/moshi](../references.md)（key: moshi-repo，[A]）；[QwenLM/Qwen2.5-Omni](../references.md)（key: qwen2.5-omni-repo，[A]）；[QwenLM/Qwen3-Omni](../references.md)（key: qwen3-omni-repo，[A]）；[zai-org/GLM-4-Voice](../references.md)（key: glm4voice-repo，[A]，THUDM 301 重定向链）；[MoonshotAI/Kimi-Audio](../references.md)（key: kimi-audio-repo，[A]）；[stepfun-ai/Step-Audio](../references.md)（key: step-audio-repo，[A]）；[ictnlp/LLaMA-Omni](../references.md)（llama-omni 条目第二 URL，[A]）
