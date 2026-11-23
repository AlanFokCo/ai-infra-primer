# 2.1 PagedAttention 与块管理（PagedAttention and Block Management）

> 几十条并发请求的 KV cache 如何装进一块固定显存，且每请求的浪费不超过一个块？本章从连续分配的三种浪费推出碎片上界与 block size 权衡账，给出 COW 正确性的命题与证明，并用 vLLM V1 的块池 / 协调器机制片段把这些语义对到代码 | 前置：models 2.4（KV cache 容量式 (9)、前缀不可变命题）、inference 1.2（每 token KV 字节数 κ 的几何表与 prefill / decode 账） | 锚点：vLLM / PagedAttention [Kwon et al., 2023] [A]（arXiv:2309.06180 + vllm-project/vllm）；机制片段 `vllm/v1/core/` 四文件与 `config/cache.py`（2026-11-23 main 核实，模块全量走读见 5.2）

## 1. 连续分配的三种浪费：从容量账到碎片

vLLM 论文的实测数字：既有 serving 系统的 KV cache 显存中，只有 20.4%–38.2% 真正用于存储 token 状态（[Kwon et al., 2023](../references.md)，arXiv:2309.06180，[A]，Fig.2 / §2；口径：「有效」指实际存放 token KV 的字节占比）。其余六成到八成被三种浪费占去。先 recap 容量账，再逐项拆浪费。

KV cache 总字节由 models 2.4 式 (9) 给出，此处按本维度符号重述（翻译注：models 的 $b$ = 每元素字节数，按 spec §4.3 译为文字「字节」；本章 $b$ 专归块大小（block size，token 数）， $B$ 专归 batch）：

$$
\mathrm{Mem}_{\mathrm{KV}} = \kappa \thinspace L \thinspace T \thinspace B, \qquad \kappa = 2 \thinspace H_{kv} \thinspace d_{\mathrm{head}} \cdot \mathrm{bytes} \qquad (1)
$$

**符号**： $\kappa$ = 每 token 每层 KV 字节数（MHA / GQA 形，K、V 两份合计）； $L$ = 层数； $T$ = 序列长； $B$ = batch 内请求数（本章口径）； $H_{kv}$ = KV head 数； $d_{\mathrm{head}}$ = head 维； $\mathrm{bytes}$ = 每元素字节数（bf16 = 2）。MLA 形无因子 2、无 $H_{kv}$ 乘： $\kappa = (d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}$，低秩推导归 models 4.1、几何账见 1.2。

**本章数值口径**（全章例题复用）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、bf16）， $\kappa$ = 4096 字节 = 4 KiB， $\kappa L$ = 128 KiB/token；block size $b$ 取 16（论文实验值）与 64（当前引擎默认值）两档，版本差异见 §3；KV 池 48 GiB（教学口径：H100 SXM 80 GB × 0.9 预算 − 权重 16 GB − 激活预留，取整；H100 规格见 1.3 datasheet 表）。§6 换教学反事实层分布，原因在该节声明。

代入式 (1)： $T$ = 131072（128K 满窗）、 $B$ = 1 时 $`\mathrm{Mem}_{\mathrm{KV}}`$ = 128 KiB × 131072 = 16 GiB——一条长上下文请求的 cache 与整个模型权重同量级（models 2.4 §5 同账）。 $B$ = 32 时 512 GiB，远超单卡显存。容量账说明 cache 是显存主项；但「装不下」只是问题的一半，另一半是「装得下却用不满」——碎片。

连续分配按请求的最大可能长度 $M$ 预留整段显存。设请求实际长度 $T$，预留浪费（口径：已划给该请求、尚未写入 KV 的 slot；slot = 存储位，一个 token 的 KV 位置）为：

$$
\mathrm{waste}_{\mathrm{res}} = (M - T) \thinspace \kappa L, \qquad \frac{\mathrm{waste}_{\mathrm{res}}}{M \thinspace \kappa L} = \frac{M - T}{M} \qquad (2)
$$

**符号**： $M$ = 预留的最大序列长； $T$ = 当前实际长度； $\kappa L$ = 每 token 全层 KV 字节。代入 $M$ = 2048、 $T$ = 256：浪费 1792 slot = 224 MiB，相对浪费 87.5%——这段显存整个请求生命周期都占着，别的请求用不了。

论文把浪费分成三种（§2 / Fig.3 口径），定义与例如下：

- **预留浪费（reserved）**：为未来 token 预留的 slot。它最终可能被用上，但在用上之前一直阻塞其他请求——式 (2) 量的就是它。
- **内部碎片（internal fragmentation）**：所谓内部碎片，是指已分配给请求、但直到请求结束也不会被写入的 slot。例：按 $M$ = 2048 预留而请求在 $T$ = 256 终止，多出的 1792 slot 在终止那一刻从「预留」转为内部碎片；分配器按 2 的幂向上取整（buddy allocator）也会制造内部碎片——需要 1000 slot 给了 1024，多出 24 slot。
- **外部碎片（external fragmentation）**：所谓外部碎片，是指分配器留下的、介于已分配块之间的空隙。例：两个 2048-slot 请求之间剩一段 1500 slot 的空隙，任何 $M$ = 2048 的新请求都装不进——空隙总量可能不小，却谁也用不上。

边界辨析：预留与内部碎片的区别是时间性的——请求存活期间叫预留（还可能被用），终止后才坐实为内部碎片；外部碎片则在服务开始前就已知不可用（论文 §2 原文口径）。分页机制对三者各个击破：按需分配消灭预留，定长块消灭外部碎片，内部碎片压到每请求不超过一个块——这是 §3 的命题。分页的单位是块（block，KV block）：一段定长的连续存储区，装 $b$ 个 token 的 K 与 V（ $b$ 即 block size，取值权衡见 §3）。

先排除一个直觉方案：为什么不保持连续、但按需动态扩容？因为 KV tensor 必须连续才能被 kernel 消费，每次扩容都要把已有 KV 整段搬进新区域。逐 token 扩容时（几何级预分配只是把预留浪费请回来），第 $t$ 步搬 $t \thinspace \kappa L$ 字节，全程累计 $\sum_{t=1}^{T} t \thinspace \kappa L \approx T^{2} \kappa L / 2$。代入 $T$ = 4096：约 $4096^{2}/2 \times 131072$ ≈ 1.1 TB 的累计拷贝，按 H100 HBM 3.35 TB/s 折合约 0.33 s； $T$ = 131072 满窗时达约 1125 TB、折合 336 s（口径：按单向流量折算；memcpy 含读 + 写两份流量，实际时间约 2×）——二次增长使连续动态扩容在长上下文下不可行。分页把每步成本降为块表追加一项 + 至多分配一块，搬运只在 COW 时发生且至多一块（§5）。连续与分页的差别不是「省一点内存」，是每步 $O(T)$ 拷贝与 $O(1)$ 登记的差别。

碎片为什么是 throughput 问题而不只是内存问题？因为浪费的显存直接压低了并发 batch。设 workload 为 $M$ = 2048 预留、实际平均长度 $T$ = 500（教学 workload 口径）， $B$ = 64 条请求：连续预留需 64 × 2048 × 128 KiB = 16 GiB，实际只用 4 GiB；分页（ $b$ = 16）下每请求占 $\lceil 500/16 \rceil$ = 32 块 = 512 slot，共 64 × 512 × 128 KiB = 4 GiB。同一个 48 GiB 池，连续预留装得下 192 条请求，分页装得下 768 条——4 倍并发。而 decode 的权重 GEMM 算术强度 $I \approx B$（models 2.4 §6 / inference 1.2 已推导，recap 一句；总强度另受 KV 读制约，饱和上界见 models 2.4 式 (12)）：并发翻 4 倍意味着权重 GEMM 的强度同翻 4 倍，decode 吞吐随之上升。所以碎片账的终点是吞吐账：省下的每一个块都换成 batch 里的一个位置。

```
连续分配（按最大长度 M 预留，请求间留空隙）：
请求 A [████░░░░░░░░░░░░░░░░]  M=2048, 实际 T=256
请求 B [██████░░░░░░░░░░░░░░]  M=2048, 实际 T=192
        ↑已用 ↑预留/内部碎片
空闲区 [   1500 slot 空隙   ] ← 外部碎片：装不下任何 M=2048

分页分配（b=4 示意，物理块定长、无需连续）：
A: 逻辑块 0→物理 7, 逻辑块 1→物理 1, ...
B: 逻辑块 0→物理 2, 逻辑块 1→物理 5, ...
物理池 [7][2][1][5][4][3][6][0]...  块间无空隙，按需取用
```

## 2. 块表：逻辑块与物理块

把显存切成定长块之后，请求的 KV 不再连续——attention kernel 怎么找到位置 $t$ 的 KV？答案是块表（block table）：所谓块表，是指每请求一张的逻辑块号 → 物理块号映射表，表项同时记录该块的已填充 slot 数（论文 §4.2 原文：「Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions」）。

所谓 PagedAttention，是指以块表为输入、按块 gather KV 的 attention 计算组织方式：逻辑视图仍是连续序列，物理视图是散落的定长块，kernel 经块表把逻辑位置翻译成物理 slot 再读取（[Kwon et al., 2023](../references.md)，§4.1）。翻译是纯算术。设 token 位置 $t$ 从 0 计，block size 为 $b$（token 数）：

$$
j(t) = \lfloor t / b \rfloor, \qquad r(t) = t - j(t) \thinspace b, \qquad \mathrm{slot}(t) = \mathrm{bt}[j(t)] \cdot b + r(t) \qquad (3)
$$

**符号**： $j(t)$ = 逻辑块号； $r(t)$ = 块内偏移（ $t$ 除以 $b$ 的余数）； $\mathrm{bt}[j]$ = 块表中逻辑块 $j$ 对应的物理块号； $\mathrm{slot}(t)$ = 该 token KV 在整个物理池中的 slot 下标。代入论文 Fig.6 口径（ $b$ = 4；此处从章脊的 16/64 换到 4，只为复用论文 7-token 走读）： $t$ = 5 时 $j$ = 1、 $r$ = 1，若逻辑块 1 映射物理块 1，则 $\mathrm{slot}$ = 1 × 4 + 1 = 5。

Fig.6 的完整走读（ASCII 重绘，出处：论文 Fig.6）：

```
prompt = 7 token, b = 4
逻辑块:  [0: t0 t1 t2 t3] [1: t4 t5 t6 _]
块表:    0 → 物理 7 (#filled=4)
         1 → 物理 1 (#filled=3)
物理池:  ... [1: t4 t5 t6 _] ... [7: t0 t1 t2 t3] ...

decode 第 1 步: t7 → 逻辑 1 偏移 3 → 写入物理 1 的空 slot（不分配新块）
decode 第 2 步: t8 → 逻辑 1 已满 → 新逻辑 2 → 分配物理 3，块表追加一项
```

两个结构事实值得钉住。其一，块表是 append-only 的：已分配的映射不改，只在尾部追加（代码锚点：`block_pool.py` `BlockHashToBlockMap` NOTE #1——不去重缓存块，正是为了保证块号不变、块表 append-only）。其二，分块不改变 attention 的数学：kernel 按块 gather 后，可见集仍由 causal mask 决定，式 (3) 只是地址翻译。kernel 侧的输入是 query、块表、KV 池三件；paged gather 与 split-KV 的算法实现归 3.1，本章只钉接口。

分布式执行下块表语义不变（论文 §4.6）：KV 管理器集中在 scheduler 一侧，TP（tensor parallelism，张量并行）组内各 worker 共享同一份逻辑块 → 物理块映射，每步开头随控制消息广播，执行中不再同步内存管理。差别只在物理内容：attention 按 head 维切分后，每个 worker 只存自己那部分 head 的 KV——块号相同，块内字节是切分后的份额。所以块表是全局一份，KV 池是每 worker 一份；TP 的通信账归 8.1，此处只钉「映射共享、内容分片」这一结构。

表项里的 #filled 不是冗余字段：最后一块通常未满，未写入的 slot 里是垃圾值或上一个请求的残留 KV，kernel 必须知道有效范围才能正确 mask。块表给出「物理块号 + 已填充数」，kernel 据此把读取限制在已写入的 slot 内——这是分页不改变 attention 语义的另一半保证（地址翻译管「读哪里」，#filled 管「读多少」）。

$b \gt 1$ 的动机在 kernel 侧：一个块覆盖多个位置，kernel 得以在块内并行处理更多位置（论文 §4.3 原文：「Storing multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel」）。 $b$ = 1 退化为逐 token 分页——SGLang 论文的 page = 1 token 即此形（2.2 §3 对照）。

## 3. 碎片上界与 block size 权衡

分页留下多少浪费？答案是结构上界而非经验值：每请求浪费不超过一个块。这是论文的核心论断（§4.3 原文：「vLLM limits all the memory wastes for a request within one block」），下面给出命题与证明。

**命题 1**（碎片上界）：在「从左到右填充、前块满才分配新块」的策略下，任意时刻序列长为 $T$ 的请求占用 $\lceil T/b \rceil$ 个块，浪费 slot 数满足

$$
\mathrm{waste}(T, b) = \lceil T/b \rceil \thinspace b - T \le b - 1, \qquad \frac{\mathrm{waste}(T,b)}{T} \lt \frac{b}{T} \qquad (4)
$$

**符号**： $\mathrm{waste}(T,b)$ = 已分配但未写入的 slot 数； $\lceil \cdot \rceil$ = 向上取整；其余同前。代入章脊口径（ $\kappa L$ = 128 KiB）： $T$ = 1000 时， $b$ = 16 占 $\lceil 1000/16 \rceil$ = 63 块 = 1008 slot，浪费 8 slot = 1 MiB，相对浪费 0.8% $\lt$ 1.6% = $b/T$； $b$ = 64 占 16 块 = 1024 slot，浪费 24 slot = 3 MiB，相对浪费 2.4% $\lt$ 6.4%。同一请求， $b$ 翻 4 倍，浪费翻 3 倍。

**证明**（路线：先立一条分配不变式——「除最后一块外所有块皆满」，对分配事件归纳；上界由不变式一步得出）。基例：prefill 结束时块从左到右填充，只有最后一块可能未满，不变式成立。保持：每步 decode 写入一个 token——若最后一块未满则写入其中，其余块不变；若已满则分配新块并写入其首 slot，旧的最后一块转为满块，新的最后一块成为唯一可能未满者。两种情形都不破坏不变式。终止：任意 $T$ 下只有最后一块有浪费，浪费 = $b - r(T)$（ $r(T) \gt 0$ 时）或 0（ $r(T) = 0$ 时），故 $\mathrm{waste} \le b - 1 \lt b$，相对浪费 $\lt b/T$。证毕。

边界例： $T$ 恰被 $b$ 整除时浪费 = 0（界不取到）； $r(T) = 1$ 时浪费 = $b-1$（界取到）。命题对任意 $T$ 成立，与 workload 无关。

> **坑**：论文里的「<4%」不是普适上界——那是 Fig.2 典型长度 workload 下的经验均值，随 workload 变化；论文文字 stated 的界只有「within one block」。把「<4%」写成保证上界是口径错误：相对上界是 $b/T$，短请求可以远超 4%（ $T$ = 100、 $b$ = 64 时上界 64%）。

**block size 权衡**。论文 §7.2 给出两侧代价（原文：「If the block size is too small, vLLM may not fully utilize the GPU's parallelism... If the block size is too large, internal fragmentation increases and the probability of sharing decreases」）：

- $b$ 小：碎片上界 $b/T$ 低；但块表项数 $\lceil T/b \rceil$ 涨、kernel 每块一次 gather 的并行度不足、swap 时小块拷贝多而 PCIe 效率低（§7.3 实验：小 $b$ 下 swap 开销大）。
- $b$ 大：碎片涨、共享概率降——共享要求整块内容相同，块越大越难恰好相同（前缀缓存的块粒度，2.2 §2）。

「共享概率降」可以量化。设两条请求的前缀在位置 $D$ 处分叉（前 $D$ 个 token 相同），块粒度下可共享的满块数 = $\lfloor D/b \rfloor$：分叉点所在的那个块整块作废，哪怕块内只有最后一个 token 不同。分叉点均匀落在块内时，平均作废 $b/2$ 个 token 的共享机会—— $b$ = 16 平均作废 8 token， $b$ = 64 平均作废 32 token。块越大，单点分叉连坐的字节越多，这是论文 §7.2「probability of sharing decreases」的账面内容。

把两侧 overhead 写成 $b$ 的函数可以解出内存账最优值。路线：三步——先求期望碎片（均匀尾部假设），再叠块表字节，求导找极小点，最后核对哪个约束真正起作用。设余数 $r(T) = T - \lfloor T/b \rfloor b$ 在 $\lbrace 0, \ldots, b-1 \rbrace$ 上均匀分布（workload 长度足够分散时近似成立），期望浪费为：

$$
E[\mathrm{waste}] = \frac{b-1}{2} \text{ slot} \qquad (5)
$$

**符号**： $E[\mathrm{waste}]$ = 期望浪费 slot 数。代入 $b$ = 16：期望浪费 7.5 slot ≈ 960 KiB； $b$ = 64：31.5 slot ≈ 3.94 MiB。总 overhead = 期望碎片字节 + 块表字节：

$$
W(b) = \frac{b-1}{2} \thinspace \kappa L + \lceil T/b \rceil \thinspace s_{\mathrm{table}} \approx \frac{b \thinspace \kappa L}{2} + \frac{T \thinspace s_{\mathrm{table}}}{b} \qquad (6)
$$

**符号**： $s_{\mathrm{table}}$ = 每块表项字节数（教学口径 8 字节：物理块号 int32 + 填充计数 int32）；近似取 $\lceil T/b \rceil \approx T/b$、 $(b-1)/2 \approx b/2$。对 $b$ 求导置零： $\kappa L / 2 - T s_{\mathrm{table}} / b^{2} = 0$，解得

$$
b^{\ast} = \sqrt{\frac{2 \thinspace T \thinspace s_{\mathrm{table}}}{\kappa L}} \qquad (7)
$$

**符号**： $b^{\ast}$ = 内存账最优 block size（token 数）。代入章脊口径： $T$ = 4096 时 $b^{\ast}$ = $\sqrt{2 \times 4096 \times 8 / 131072}$ = $\sqrt{0.5}$ ≈ 0.71； $T$ = 131072 满窗时 $b^{\ast}$ = $\sqrt{16}$ = 4。两档都低于论文实验下界 16——结论：8B GQA 口径下内存账最优被压到 1 以下， $b$ 的真实下界来自 kernel 并行度与 swap 效率，而非块表内存。这定量解释了论文 §7.2 的结论「16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation」：16 是 kernel 侧下界，不是内存侧最优。

> **提点**：块表自身的内存在账上可忽略—— $T$ = 128K、 $b$ = 16 时每请求 8192 项 × 8 字节 = 64 KiB， $B$ = 256 也才 16 MiB，占 48 GiB 池的 0.03%。「小 $b$ 高表开销」的代价主要在 kernel 侧（gather 次数、分支数），不在 CPU 内存侧。

**默认值的两个口径**（版本锚定，2026-11-23 main 核实，[A] vllm-repo）：

- 论文实验默认 16（§7.2 原文：「vLLM sets its default block size as 16」；ShareGPT trace（请求轨迹数据集）上 16–128 最优，Alpaca trace 上 16/32 合适、更大显著变差——序列比块还短，Fig.18(b) 实验口径）。
- 当前 main 默认 64：`CacheConfig.DEFAULT_BLOCK_SIZE = 64`（`vllm/config/cache.py` L185）。
- backend 偏好可调：SWA（sliding window attention，每位置只 attend 最近 $w$ 个 token 的 attention 形，模型侧归 models 4.2）backend 偏好 256（`v1/attention/backends/mla/sparse_swa.py` `get_preferred_block_size`，L166-167）——SWA 驻留量受窗口封顶（§6），大 $b$ 的碎片代价被窗口限住；CPU MLA 强制 16（`v1/attention/backends/mla/cpu_mla.py`，`get_supported_kernel_block_sizes` 返回 [16]）；偏好协商在 `vllm/platforms/interface.py` `_preferred_block_size_for_backends`（L711）。

Alpaca trace 上大 $b$ 变差的机制值得写出来：Alpaca 指令短，序列长小于 $b$ 时整个请求只占一个块，浪费占比趋近 $(b-T)/b$—— $b$ = 128 下一条 20 token 的指令浪费 108 slot = 84%。这正是命题 1 的相对上界 $b/T$ 在短请求上失去约束力的实例：上界本身高达 640%，碎片不再是「零头」而是主项。短请求 workload 选小 $b$，长请求 workload 容忍大 $b$；当前引擎把 $b$ 暴露为 backend 可协商参数（上方偏好列表），把这个自由度留给部署侧。

> **坑**：把论文的 16 当当前引擎默认是版本漂移错误——两个口径差 4 倍，碎片上界 $b/T$ 同差 4 倍。写 benchmark 或容量账前先查引擎版本的 `CacheConfig`；引用论文实验数字时注明「论文实验口径」。

## 4. 块池：分配、回收与 watermark

块被释放后内容就失效了吗？vLLM 的答案是否——`ref_cnt` = 0 的块只是驱逐候选，内容仍可能被前缀缓存命中（2.2）。这一节把块池的三个操作对到代码（机制例证片段级；`block_pool.py` 等四文件的模块全量走读见 5.2）。

`BlockPool`（`vllm/v1/core/block_pool.py`，2026-11-23 main 核实）持三件结构：`blocks` = 全部物理块的列表；`free_block_queue` = 空闲块双向链表，按驱逐顺序排列；`cached_block_hash_to_block` = 块 hash → 已缓存块的映射（前缀缓存的查找基础，§4 末与 2.2 展开）。所谓引用计数 `ref_cnt`，是指正在使用该块的请求数——它是共享（§5）与驱逐候选判定的唯一依据。

三个操作的语义（函数级锚点）：

- **分配** `get_new_blocks`（L681）：从 free 队列头取块；若块带缓存 hash，先逐出缓存登记（`_maybe_evict_cached_block`）再置 `ref_cnt` = 1。取块即覆盖内容，所以缓存登记必须先清。
- **命中复活** `touch`（L774）：前缀命中时 `ref_cnt` += 1；若原为 0（在 free 队列中），先把块从队列摘除。
- **释放** `free_blocks`（L798）：`ref_cnt` −= 1；归零时按有无缓存 hash 分流——无 hash 块插队列头（LIFO 立即复用），有 hash 块 append 队列尾（FIFO；LRU = least recently used，最久未访问者先驱逐）。

```python
# vllm/v1/core/block_pool.py（main 2026-11-23，机制摘录，非全文）
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0:      # 驱逐候选 → 复活
            self.free_block_queue.remove(block)
        block.ref_cnt += 1

# free_blocks 的分流逻辑（摘录）
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)   # LIFO：无缓存价值，立即复用
else:
    blocks_to_evict_last.append(block)    # FIFO：缓存块按 LRU 顺序驱逐
```

状态走读：

```
分配中 (ref_cnt≥1) ──请求结束/共享解除──→ ref_cnt=0
  ├─ 有 hash → free 队列尾：驱逐候选，内容仍有效，touch 可复活
  └─ 无 hash → free 队列头：LIFO 立即复用
驱逐候选 ──get_new_blocks──→ 清 hash 登记，ref_cnt=1，内容将被覆盖
```

> **坑**：把 `ref_cnt` = 0 当「内容已失效」会提前清掉 hash 登记——块还在池里、KV 还正确，却再也无法被前缀命中。正确语义是「无人引用 = 可驱逐」，不是「不可用」；驱逐发生在被重新分配的那一刻（`get_new_blocks`），不是释放的那一刻。

**watermark**（预留水位）。所谓 watermark，是指 admission（准入，把等待请求接进 running 队列）时额外保留的空闲块数：`KVCacheManager` 构造时算 `watermark_blocks = int(watermark × num_blocks)`（`kv_cache_manager.py` L200），`allocate_slots` 只对 WAITING / PREEMPTED 状态的请求把 watermark 加进需求（L529-536）——running 请求的每步分配不受 watermark 限制。语义：新请求准入不能把空闲块吃到零，否则下一步 running 请求无块可分就触发抢占。分配失败的处理（摘录，L595-599）：

```python
# vllm/v1/core/kv_cache_manager.py allocate_slots（机制摘录）
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None    # 分配失败 → 调度器排队或抢占
```

代入章脊口径看量级： $b$ = 64 时池含 6144 块，若配 watermark = 0.01，则 `watermark_blocks` = $\lfloor 0.01 \times 6144 \rfloor$ = 61 块 = 488 MiB——新请求准入时这 488 MiB 必须空着，留给 running 请求下一步的增量。当前 main 的 `SchedulerConfig.watermark` 默认 0.0（`config/scheduler.py` L232，2026-11-23 核实），即默认不做额外预留，机制保留为可配置的 admission 门。抢占策略本身是论文 §4.5 口径：FCFS（first-come-first-served，先到先服务）；all-or-nothing 驱逐——一条序列的块要么全逐要么全留，因为 attention 每步读它的全部块，逐一半等于全逐；最晚到达的请求先被抢占；beam（beam search，束搜索，每步保留 top-k 候选）候选组（sequence group）成组调度（gang-scheduling，同抢占同恢复），共享块的序列必须同进同退。恢复有 recompute（重算 KV）与 swap（换出到 CPU 内存）两条路，成本账归 4.2；论文 §7.3 实验口径：swap 按块拷贝，小 $b$ 意味着同样字节数拆成更多次小拷贝、PCIe 效率低，recompute 开销不高于 swap 延迟的 20%（论文实验 workload 内）。

块写满时还有一步登记：`cache_full_blocks`（`block_pool.py` L203）为满块计算 hash 并插入 `cached_block_hash_to_block`——这是前缀缓存的块 hash 基础，hash 链的构造与命中判定在 2.2 §2 展开。另有一个机制细节：池初始化时取出 0 号块作 null block（`block_pool.py` L161-165），作为「此处无需 KV」的占位——SWA 窗外位置、Mamba（状态空间模型）层的跳过块都映射到它，`ref_cnt` 不维护、永不释放。占位块让块表保持满长度，kernel 不必为缺块写分支。

## 5. 写时复制：parallel sampling 与 beam search 的块共享

$n$ 个采样输出共享同一个 prompt，prompt 的 KV 该存几份？朴素答案是 $n$ 份；PagedAttention 的答案是 1 份，外加每个输出至多一次单块拷贝。

共享场景有三（论文 §4.4）：parallel sampling（ $n$ 个输出共享 prompt）、beam search（候选共享前缀，且共享模式随解码演化——候选 3 可能转而继承候选 2 的前缀）、共享 system prompt（provider 预缓存公共前缀，用户请求直接映射过去，最后一块标 COW——这是前缀缓存的先声，2.2 展开）。

所谓写时复制（copy-on-write，COW），是指多个序列的逻辑块映射到同一物理块（`ref_cnt` $\gt$ 1）、仅在某个序列要写入共享块时才复制该块的协议。触发条件很窄：论文 §4.4 原文「The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block... This involves only copying one block of data」。正确性可以写成命题。

**命题 2**（COW 正确性）：在块粒度 COW 协议下，任意时刻每条序列的块表都把自身前缀映射到正确的 KV 内容，且任何序列的写入都不修改 `ref_cnt` $\gt$ 1 的共享块内容。

**证明**（路线：以「共享块内容不可变」为不变式，对 decode 步数归纳；关键一步是保持——新 token 的写入目标要么是独占块、要么先触发复制，两种情形都不写共享块。公共前缀的内容正确性由前缀不可变性保证： $k_s, v_s$ 只依赖 $x_{\le s}$，models 2.4 §4 命题，recap 一句）。基例：fork（一条请求分裂出多条子序列）时刻，共享块 = 公共前缀的满块与可能的尾部半满块，其内容由公共前缀 token 序列唯一确定（前缀不可变），对所有共享者相同。保持：第 $t+1$ 步写入新 token 时分两种情形——新 token 落在块边界（公共前缀长度恰为 $b$ 的整数倍）则分配新独占块，无拷贝；落在共享半满块内则先复制该块到独占块（`ref_cnt` = 1）、写入拷贝、原块 `ref_cnt` −= 1 而内容不变。此步的完整论证留作练习 3。终止：归纳得任意步各序列读到自身前缀的 KV，共享块从未被写。证毕（保持步见练习 3）。

边界反例：「COW 每次 fork 都拷贝」是错的——公共前缀恰在块边界结束时，全程零拷贝；拷贝只发生在尾部半满共享块被写入时。

beam search 的共享模式会演化，块表机制照样接得住（论文 §4.4 / Fig.9 语义，ASCII 重绘）：

```
beam 宽度 2，prompt 块 P0 P1 已共享（ref_cnt=2）
第 t 步:  候选 A → P0 P1 [a1]      候选 B → P0 P1 [b1]
第 t+1 步: 候选 B 分裂出候选 C（B 的前缀更优）
  C 的块表 = B 的块表逐项复制映射（P0 P1 [b1] 各 ref_cnt +1）
  仅当 C 要写入 [b1] 所在共享块时触发一次单块 COW
第 t+2 步: 候选 A 被淘汰 → 其独占块 ref_cnt 归零回池，
  共享块 P0 P1 的 ref_cnt 只减 1，内容不动
```

关键在「继承前缀 = 复制块表映射 + 引用计数加一」，不搬 KV 字节；淘汰候选也只减引用计数。论文 §6.3 的 55% 内存节省上界就来自这种随解码演化的共享——若每次分裂都整段拷贝前缀，节省无从谈起。

拷贝与节省的账。单次拷贝字节 = 一个块； $n$ 输出共享 prompt 的节省（口径：prompt 部分 KV 字节，不含各输出自己的尾块）：

$$
\mathrm{copy} = b \thinspace \kappa L, \qquad \mathrm{save} = (n-1) \thinspace T_p \thinspace \kappa L \qquad (8)
$$

**符号**： $T_p$ = prompt 长度； $n$ = 共享同一 prompt 的输出数。代入章脊口径： $b$ = 16 时单次拷贝 2 MiB，按 H100 HBM 3.35 TB/s（[NVIDIA datasheet](../references.md)，[C] 官方规格披露；1.3 参数表）device 侧拷贝约 0.63 μs；对照整段 prompt 拷贝（ $T_p$ = 2048）256 MiB ≈ 80 μs，差 128 倍 = $T_p / b$（读 + 写 2× 流量口径同 §1，两值同乘 2、比值不变）。 $n$ = 4、 $T_p$ = 2048 时节省 3 × 2048 × 128 KiB = 768 MiB，即 prompt KV 的 75%。论文 §6.3 口径：beam search 共享至多省 55% 内存（workload 依赖，非普适上界）。

V1 的实现形态（机制级，2026-11-23 main 核实）：parallel sampling 在 engine 层 fork 出 $n$ 个子请求（`v1/engine/parallel_sampling.py` `ParentRequest`），共享经前缀 hash + `ref_cnt` + `touch`（§4）实现；共享半满尾块走 COW redirect——`single_type_kv_cache_manager.py` `_apply_cow` 分配私有块并登记（源块, 目标块）拷贝对，`take_pending_cow_copies` 取出交 worker 执行；beam search 编排在 entrypoints 层（`entrypoints/generate/beam_search/`）。四文件的模块全量走读归 5.2，此处只钉机制语义。

共享块同时是前缀缓存候选：`ref_cnt` $\gt$ 1 的块不在 free 队列、不可驱逐；待全部共享者释放后，它带 hash 进入驱逐候选态，仍可被后续请求命中（§4）。共享与前缀缓存是同一套 `ref_cnt` 机制的两条复用路径——前者在请求组内（本章），后者跨请求（2.2）。

> **提点**：COW 是 OS fork 语义的直译——页表换成块表、页换成 KV 块、「写」换成「追加新 token 的 KV」。差别在于：OS 页写是修改已有内容，KV 追加只写新 slot，所以仅当公共前缀的尾块未满时才触发拷贝。

## 6. 混合 KV 几何：协调器与逐类型管理器

同一模型里同时有 full attention、SWA、MLA 层时，各层类型的 $\kappa$ 与驻留量不同——一个块池如何服务不同块几何？V4.1-Flash 的 serving 含义正在此：config 给出 `sliding_window` = 128 与 MLA 几何（共享 KV 向量 512 元素 = $d_{\mathrm{latent}}$ 448 + $d_{\mathrm{rope}}$ 64；head_dim 512 已含 rope，kv_lora_rank 448 由引擎导出，见 1.2 §3）（[V4.1-Flash config](../references.md)，[W]，hf-mirror 传输核实），模型侧形式化归 models 维度（V4.1 待补）；serving 侧的事实是不同层类型的 KV 生命周期不同——SWA 窗外 KV 可丢，full attention 全量保留。

V1 的答案是分组协调（机制片段，2026-11-23 main 核实）：同几何的层划为一个 KVCacheGroup，每组一个 `SingleTypeKVCacheManager`（`single_type_kv_cache_manager.py`：`FullAttentionManager` / `SlidingWindowManager` / `MambaManager` 等类）；`KVCacheCoordinator`（`kv_cache_coordinator.py`）持 `single_type_managers` 元组，调度粒度 `scheduler_block_size` = 各组块大小的最小公倍数，分配需求逐组求和：

```python
# vllm/v1/core/kv_cache_coordinator.py（机制摘录）
# 调度粒度 = 各组块大小的 LCM，须为每组块大小的整数倍
assert scheduler_block_size % hash_block_size == 0 and all(
    scheduler_block_size % g.block_size == 0
    for g in kv_cache_config.kv_cache_groups
)
self.single_type_managers = tuple(
    create_single_type_manager(group, ...)   # 每组一个逐类型管理器
    for group in kv_cache_config.kv_cache_groups
)
```

调度粒度为什么取最小公倍数？因为调度器按 token 数切 chunk（chunked prefill，4.3）与判定命中边界，这些边界必须同时落在每一组的块边界上——否则某个组会出现「半个块」的分配需求。`scheduler_block_size` 是各组块大小的公倍数就保证了这一点；块 hash 的基础粒度 `hash_block_size` 可以更小，组的实际块大小是它的整数倍（`BlockPool` docstring 口径，2026-11-23 核实）。

SWA 组的关键操作是 `remove_skipped_blocks`：`allocate_slots` 内先释放窗外块再分配新块（`kv_cache_manager.py` L575），驻留量由窗口而非 $T$ 封顶。每请求 SWA 组驻留块数的上界：

$$
\mathrm{blocks}_{\mathrm{SWA}} \le \lceil w/b \rceil + 1 \qquad (9)
$$

**符号**： $w$ = 窗口长度（token 数）；+1 来自窗口跨块边界时多压一个块。代入 $w$ = 128、 $b$ = 64：至多 3 块 = 192 slot，与 $T$ 无关。

数值例（教学反事实层分布；换口径原因：V4.1 实际层分布本轮未核实，用反事实演示账的结构）：40 层 = 30 full + 10 SWA，各层均 GQA $\kappa$ = 4 KiB， $w$ = 128、 $b$ = 64、 $T$ = 4096。full 组每请求 64 块 × 64 token × 30 层 × 4 KiB = 480 MiB；SWA 组至多 3 块 × 64 × 10 层 × 4 KiB = 7.5 MiB；合计 ≈ 487.5 MiB，对照全 full 的 640 MiB 省 23.8%。 $B$ = 32 时 15.2 GiB 对 20 GiB——混合架构的 KV 容量账必须逐组算，不能拿单一 $\kappa$ 乘总层数。MLA 组同理另立： $\kappa = (d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}$（1.2 口径），块几何由 latent 维决定，协调器视其为又一个组，无特殊分支。

## 总结

| 论断 | 回指 |
| --- | --- |
| 连续分配三浪费：reserved / internal / external；既有系统实测仅 20.4%–38.2% 显存有效 | §1、式 (2) |
| 分页按需分配 + 定长块：每请求浪费 ≤ 1 块，相对浪费 $\lt b/T$；「<4%」是经验均值非上界 | 命题 1、式 (4)、§3 |
| 地址翻译 $\mathrm{slot}(t) = \mathrm{bt}[\lfloor t/b \rfloor] b + r(t)$；分块不改 attention 可见集 | 式 (3)、§2 |
| 内存账最优 $b^{\ast} = \sqrt{2 T s_{\mathrm{table}} / (\kappa L)}$，8B 口径下 $\lt 1$； $b$ 下界由 kernel 定 | 式 (5)-(7)、§3、练习 2 |
| 论文实验默认 $b$ = 16；当前 main 默认 64，SWA 偏好 256 / CPU MLA 16（2026-11-23 核实） | §3 版本锚定 |
| `ref_cnt` = 0 ≠ 内容失效：缓存块是驱逐候选，`touch` 可复活；驱逐发生在重新分配时 | §4、练习 4 |
| watermark 只约束 WAITING / PREEMPTED 准入；当前 main 默认 0.0 | §4 |
| COW：共享块不可变，拷贝 ≤ 1 块 = $b \kappa L$； $n$ 输出共享 prompt 节省 $(n-1) T_p \kappa L$ | 命题 2、式 (8)、练习 3 |
| 混合 KV：逐类型管理器 + 协调器，调度粒度 = 块大小 LCM；SWA 驻留 ≤ $\lceil w/b \rceil + 1$ 块 | §6、式 (9) |

## 练习

1.（计算 · [10]）章脊口径（ $\kappa L$ = 128 KiB，KV 池 48 GiB）：(a) $b$ = 16 与 $b$ = 64 时池各含多少块，token 容量各多少？(b) $T$ = 1000 的请求在两种 $b$ 下各占多少块、浪费多少字节？校验：(a) 两种 $b$ 的 token 容量相等（块数 × $b$）；(b) 浪费 slot 数严格小于 $b$，且与式 (4) 的 $\lceil T/b \rceil b - T$ 一致。
2.（推导 · [20]）从头推导 block size 权衡账：(a) 在余数 $r(T)$ 均匀分布假设下求 $E[\mathrm{waste}]$；(b) 写出总 overhead $W(b)$（期望碎片字节 + 块表字节， $s_{\mathrm{table}}$ = 8 字节）并解出 $b^{\ast}$；(c) 代入 8B 口径的 $T$ = 4096 与 $T$ = 131072，解释为何实际默认 $b$ 远大于 $b^{\ast}$。校验：(c) 两档 $b^{\ast}$ 分别 ≈ 0.71 与 4，与式 (7) 一致；结论方向与论文 §7.2「16 is large enough... small enough」一致（下界来自 kernel 侧而非内存侧）。
3.（证明 · [30]）补全命题 2（COW 正确性）的保持步：设第 $t$ 步后不变式成立（凡 `ref_cnt` $\gt$ 1 的块内容不可变、由公共前缀确定），证第 $t+1$ 步后仍成立。要求分两种情形：(a) 新 token 落在块边界；(b) 新 token 落在共享半满块内。指明哪一步用到前缀不可变性（models 2.4），并说明拷贝目标块 `ref_cnt` = 1 为何保证后续写入不影响其他共享者。校验：你的证明应蕴含「公共前缀长度为 $b$ 的整数倍时全程零拷贝」这一特例。
4.（代码 · [30]）读 `vllm/v1/core/block_pool.py` 的 `free_blocks` 与 `get_new_blocks`（main 2026-11-23）：先写出预测——请求结束释放块时，有 `block_hash` 的块与无 hash 的块各进 free 队列哪一端、谁先被驱逐？再对照源码 `prepend_n` / `append_n` 验证；并解释无 hash 块用 LIFO 的理由（源码注释给了一条）。校验：源码注释「LIFO reuse of non-cached blocks for better GPU locality」与「FIFO reuse of cached blocks for LRU eviction behavior」；预测应与这两条的队列端语义一致。

## 参考

- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——碎片上界 §4.3、块表 §4.2、COW §4.4、调度与抢占 §4.5、block size 实验 §7.2、recompute / swap §7.3
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——机制片段 `v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager}.py`、`config/cache.py` L185、`config/scheduler.py` L232、`v1/engine/parallel_sampling.py`、`platforms/interface.py` L711、`v1/attention/backends/mla/{sparse_swa,cpu_mla}.py`（均 2026-11-23 main 核实）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——80 GB HBM3、3.35 TB/s（拷贝成本例）
- [DeepSeek-AI, V4.1-Flash config](../references.md)：config.json（key: v41-flash-config，[W]，hf-mirror 传输核实）——sliding_window 128、MLA 几何 head_dim 512 含 rope 64（kv_lora_rank 448，§6 混合 KV 几何）
- 跨维度指引（纯文字）：models 2.4 式 (9)（KV 容量）与 §4 前缀不可变命题；models 4.1（MLA 低秩）；models 4.2（SWA 模型侧）；inference 1.2（ $\kappa$ 几何表）、1.3（datasheet 参数表）、2.2（块 hash 链）、3.1（paged kernel 算法）、4.2（recompute vs swap 成本账）、5.2（四文件模块全量走读）
