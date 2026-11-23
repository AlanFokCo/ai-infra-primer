# 2.2 Prefix Caching 与 RadixAttention（Prefix Caching and RadixAttention）

> 命中一段前缀能省下多少 prefill，缓存又凭什么不被死条目占满？本章给出前缀复用的两条实现路径——hash 链与基数树，推导命中率 → prefill 节省账，并用 DeepInfer 生产披露校准 TTL 与命中率的权衡 | 前置：2.1（块表、ref_cnt、块池）、models 2.4（前缀不可变命题）、inference 1.2（prefill FLOPs 账） | 锚点：SGLang / RadixAttention [Zheng et al., 2023] [A]（arXiv:2312.07104 + sgl-project/sglang）、vLLM [A]、CachedAttention [B]（arXiv:2403.19708，无官方 repo）、CacheBlend [A]（arXiv:2405.16444）、Preble [A]（arXiv:2407.00023）、DeepInfer doc（教学参照 + 官方实现，官方披露口径）

## 1. 前缀复用问题：两条实现路径

多轮对话里，第 $k+1$ 轮的 prompt = 前 $k$ 轮历史 + 新消息，前缀逐字重复。重算这段前缀是纯浪费：8B 口径下 3072 token 前缀的 prefill 线性项 FLOPs 为 $2 \times 7.5 \times 10^{9} \times 3072$ ≈ 46.1 TFLOP（§4 给完整账）。SGLang 论文报告多调用 workload 上至多 6.4× 的吞吐提升（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]，摘要口径，workload 依赖），KV cache 复用（RadixAttention）是其中的核心机制之一。

所谓前缀缓存（prefix caching），是指把已完成请求的 KV cache 按 token 前缀组织成可查找的结构保留下来，让后续同前缀请求直接复用 KV、跳过对应位置的 prefill 计算。正确性依据是前缀不可变性： $k_s, v_s$ 只依赖 $x_{\le s}$（models 2.4 §4 命题，recap 一句）——同一前缀在任何请求里的 KV 都相同，复用不改变输出。命中（口径：新请求复用了已有 KV、跳过对应 prefill）的计量用命中率 $h$ = 复用前缀 token 数 / prefill 总 token 数（token 加权，非请求数加权；请求数加权会被短请求稀释，两种口径不可混比）。

实现分两条路径，差别在查找结构与命中粒度：

| 路径 | 查找结构 | 命中粒度 | 代表 |
| --- | --- | --- | --- |
| hash 链 | 块 hash → 物理块的字典 | $b$ 的整数倍 | vLLM automatic prefix caching、DeepInfer Entrypoint CHECK |
| 基数树 | 压缩 trie，边标 = token 子序列 | token（page = 1）或 page | SGLang RadixAttention |

复用范围则分四档：请求内（parallel sampling / beam 共享，2.1 §5 的 COW）、跨请求同机（本章 §2-§3）、跨机（§6 与 2.4 的分层存储）、非前缀复用——检索增强生成（RAG, retrieval-augmented generation）的文档拼接属此档（§6 的 CacheBlend）。四档的正确性依据都是前缀不可变或其受控近似，差别只在 KV 放在哪、查找走什么结构。

**本章数值口径**（沿用 2.1 章脊）：Llama-3.1-8B 几何（ $\kappa L$ = 128 KiB/token、 $L$ = 32、 $d_{\mathrm{model}}$ = 4096）， $N = 7.50 \times 10^{9}$ 沿 1.2 matmul 口径（层参数 + 输出头，FLOPs 用；存储口径 $N_{\mathrm{full}} = 8.03 \times 10^{9}$ 不入本章 FLOPs 账）， $b$ = 64（当前 vLLM 默认，2.1 §3 版本锚定）；workload 例取多轮对话、agent、RAG 三形态。attention FLOPs 系数口径：乘加计 2、causal 减半、score 与 PV 两个 GEMM（§4 钉死，与 1.2 式 (1) 一致）。

## 2. hash 链：vLLM block hash 与 DeepInfer CHECK

怎么用每块 O(1) 的查找判定两条请求的前缀相同？把前缀内容哈希成块的 key——并把哈希链起来。vLLM 的块 hash 递推（`vllm/v1/core/kv_cache_utils.py` `hash_block_tokens`，L697，2026-11-23 main 核实）：

$$
h_i = H(h_{i-1}, \thinspace \tau_i, \thinspace e_i), \qquad h_{-1} = \mathrm{NONE} \qquad (1)
$$

**符号**： $h_i$ = 第 $i$ 块的 hash； $H$ = 哈希函数（默认 sha256，`config/cache.py` L269，可选 sha256_cbor / xxhash / xxhash_cbor，2026-11-23 核实）； $\tau_i$ = 第 $i$ 块的 token id 序列（代码 `curr_block_token_ids`，覆盖位置 $ib$ 到 $(i+1)b-1$）； $e_i$ = 隔离键（代码 `extra_keys`：LoRA adapter id、多模态特征 hash、cache_salt 等）； $\mathrm{NONE}$ = 链首常量（代码 `NONE_HASH`）； $b$ = block size。链式递推的含义： $h_i$ 依赖 $h_{i-1}$，所以 $h_i$ 覆盖的是整个前缀 $[0, (i+1)b)$，不只是本块 token——前缀相同当且仅当链上每一步的输入都相同。

```python
# vllm/v1/core/kv_cache_utils.py hash_block_tokens（main 2026-11-23，机制摘录）
def hash_block_tokens(hash_function, parent_block_hash,
                      curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(hash_function(
        (parent_block_hash, tuple(curr_block_token_ids), extra_keys)))
```

代入章脊口径： $T_p$ = 4096 的 prompt 在 $b$ = 64 下产生 64 个块 hash；新请求前 48 块逐块命中，则复用 48 × 64 = 3072 token，其余 1024 token 走 prefill。哈希计算总量 $O(T_p)$（每块 $O(b)$），查找每块 $O(1)$（字典）。命中判定 = 从 $h_0$ 起的最长连续命中：`get_computed_blocks`（`kv_cache_manager.py` L278）经协调器 `find_longest_cache_hit` 逐组查，命中长度 = 连续命中块数 × $b$（机制级；判定与分配的完整走读见 5.2）。

块粒度有尾部损失：命中长度按 $b$ 对齐，尾部不足一块的部分即使内容相同也不命中。余数均匀假设下期望损失 $b/2$ = 32 token ≈ 4 MiB KV，对应 prefill 线性项 $2 \times 7.5 \times 10^{9} \times 32$ = 0.48 TFLOP，按 H100 bf16 dense 989.5 TFLOPS（[NVIDIA datasheet](../references.md)，[C] 官方规格披露）理想折算约 0.49 ms。SGLang 论文 page = 1 token 无此损失（page 是其 KV 分页单位，§3）；page 可配置后同样有对齐损失。

> **坑**：hash 碰撞在 sha256 下工程上可忽略，但 `extra_keys` 漏配是真实的正确性风险：token 相同而 LoRA adapter 不同的两条请求，KV 值不同，不隔离就共享会产出错误输出。vLLM 把 LoRA id 等放进 `extra_keys`；SGLang 用 `RadixKey.extra_key` 做同一件事（`radix_cache.py` `match_prefix` docstring：不同 `extra_key` 的条目永不共享前缀节点，2026-11-23 核实）。

DeepInfer 的匹配同属 hash 链路径，但放在路由之前（官方披露，[DeepInfer doc](../references.md)，教学参照 + 官方实现，2026-11-23 核实）：Entrypoint（请求入口组件）的 CHECK 角色对 local kvcache-store（KV 缓存存储服务）做 hash-chain prefix match，**只查 metadata（元数据，hash / 长度等描述数据）、不读 KV 字节**；hit length 必须在路由前算出、不可隐藏（doc L191 原文口径）。缓存兼容性由元组 (deployment configuration, model weights, user group) 标识——权重、块字节布局、tokenizer 变更或改 KV 值的框架变更都使旧 cache 失效，生产未做过 cache 格式迁移（doc L199）。跨机命中如何取 KV、存储集群如何组织，§6 与 2.4 展开。

## 3. RadixAttention：基数树、LRU 与缓存感知调度

hash 链把命中粒度钉死在 $b$ 的整数倍；基数树把命中粒度还给了 token——代价是查找从每块 $O(1)$ 变成树行走。所谓基数树（radix tree），是指边标为 token 子序列（而非单个 token）的压缩 trie（前缀树）：单孩子路径合并成一条边，节点数随之压缩（[Zheng et al., 2023](../references.md)，§3）。树管理 token 序列 → KV 索引的映射，KV 本体在 paged 布局的池里；论文口径 page = 1 token（§3.1 原文：「the size of each page is equivalent to one token」）。

```
root
 ├─[system prompt A]─┬─[第 1 轮问]─[第 1 轮答]     （会话 1）
 │                   └─[第 2 轮问]─...
 ├─[few-shot 示例]───┬─[问题 1]─[答案 1]
 │                   └─[问题 2]─...
 └─[system prompt B]─[第 1 轮问]─...               （会话 2）

边标 = token 子序列；节点 value = KV 池索引（不是 KV 拷贝）
会话 1 的第 2 轮命中 [system prompt A]+[第 1 轮问]+[第 1 轮答] 全路径
```

（ASCII 重绘，出处：论文 Fig.3 的九时点演化简化为静态结构。）

匹配机制（`radix_cache.py` `_match_prefix_helper`，L658，2026-11-23 main 核实）：从 root 出发，以查询首 page 为 child key 查 children 字典 → 沿边标逐 token 比较 → 整段匹配则继续下行；部分匹配则 `_split_node`（L684）在匹配边界把节点一分为二。分裂只拷贝索引 tensor（`value[:split_len].clone()`），KV 本体留在池里不动。插入走同一套行走。复杂度可以写成命题。

**命题 1**（基数树匹配复杂度）：`match_prefix` 的 token 比较次数为 $O(T)$（ $T$ = 查询长度），字典查找次数与路径节点数同阶；树的节点总数与缓存 token 总数同阶，分裂不增加 KV 占用。

**证明**（路线：两个计数事实——每轮循环至少消耗查询的一个 page，每个节点的边标非空且互不重叠；前者给比较上界，后者给节点数上界）。每轮迭代中 child key 命中保证首 page 相同，故消耗 $\ge 1$ 个 page，迭代数 $\le T$，token 比较总数 $O(T)$。每个节点边标 $\ge 1$ token，不同节点的边标对应缓存序列的不同位置，节点数 $\le$ 缓存 token 总数。分裂把一条边拆成两条、索引 tensor 切片拷贝，KV 池占用不变。证毕。

代入量级： $T$ = 4096 的查询、树深 5 节点，匹配 = 5 次字典查找 + 至多 4096 次 token 比较；对照 hash 链的 64 次查找 + 4096 token 的哈希计算——同阶，常数不同（比较 vs 哈希）。驱逐的堆操作另计：每次叶弹出 / 父入堆 $O(\log n)$（ $n$ = 可驱逐叶数）。

**LRU（least recently used，最久未用）叶优先驱逐**。论文 §3.1 原文：「evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves」。为什么叶优先：内部节点是多条请求的共享祖先，逐它等于让所有后代的命中一起失效；叶只服务一条路径，逐它损害最小。running batch 正在用的节点不可逐——每节点维护引用计数 `lock_ref`，`inc_lock_ref` 把从该节点到 root 的整条路径加锁（`radix_cache.py` L589，2026-11-23 核实）。驱逐实现（L559，机制摘录）：

```python
# sglang/srt/mem_cache/radix_cache.py evict（main 2026-11-23，机制摘录）
eviction_heap = [(strategy.get_priority(node), node)   # LRU: last_access_time
                 for node in self.evictable_leaves]
heapq.heapify(eviction_heap)
while num_evicted < num_tokens and eviction_heap:
    _, x = heapq.heappop(eviction_heap)      # 最久未用的叶
    self.token_to_kv_pool_allocator.free_segment(x.value)  # 归还 KV 索引
    self._delete_leaf(x)
    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
        heapq.heappush(eviction_heap, ...)   # 父变成叶 → 入堆候选
```

缓存与 running 请求共享同一个内存池，不预留固定 cache 配额（论文 §3.1 原文口径）：内存紧张时逐掉全部缓存换更大 batch。这与 vLLM 的「驱逐候选」同构（2.1 §4）——缓存不是额外分配，而是空闲块的延迟回收；两边都用引用计数区分「正在用」与「可驱逐」。

**缓存感知调度（cache-aware scheduling）**。所谓缓存感知调度，是指按前缀匹配长度而非到达顺序给等待请求排序：longest-shared-prefix-first（LPM）优先调度命中更长的请求（论文 §3）。论文结论：cache 容量 $\ge$ 最长请求时，按基数树的 DFS（depth-first search，深度优先搜索）序访问请求可达最优命中率，而 LPM 序等价于 DFS 序（§3 + 附录 A.3 口径）。当前实现为 `CacheAwarePolicy.LPM` / `DFS_WEIGHT`（`schedule_policy.py` L188-192，2026-11-23 核实；策略全量走读归 4.2）。代价论文自己写明：贪心 LPM 可能饿死短前缀请求，与公平调度的集成留作 future work——饥饿与公平的账归 4.2。

版本漂移注（2026-11-23 main 核实）：`HiRadixCache` 类在当前 main 已不存在，重构为 `UnifiedRadixCache`（`unified_radix_cache.py` L224）+ `hybrid_cache/`（HiCache 用，见 2.4）；`RadixCache`（`radix_cache.py` L282）仍在。论文的 page = 1 token 是实验口径，当前实现 `page_size` 可配置，大于 1 时命中按 page 对齐（`match_prefix` 的 `page_aligned`）——§2 的尾部损失账对 page 大于 1 同样成立。本章机制级引论文 + `radix_cache.py`，实现全量走读归 6.1。

## 4. 命中率 → prefill 节省账

命中率 $h$ 省下多少 prefill 时间？线性项的答案是 $h$，attention 项的答案是 $h^{2}$——两项的节省比例不同，长上下文下差别不可忽略。路线：三步——先 recap prefill FLOPs 账（1.2），再数命中跳过的计算（线性项跳过前缀的权重 GEMM，attention 项只跳过前缀内部的 causal 对），最后求和代入。关键一步在第二步：剩余 token 仍要 attend 复用的前缀 KV，这部分 attention 不省。

recap prefill 账（1.2）： $\mathrm{FLOPs} \approx 2N T_p$ + attention 二次项。本章系数口径：乘加计 2、causal 减半、score 与 PV 两个 GEMM，每层 attention $\approx 2 \thinspace d_{\mathrm{model}} T_p^{2}$，全层 $2L \thinspace d_{\mathrm{model}} T_p^{2}$（与 1.2 的 $O(L T^{2} d_{\mathrm{model}})$ 同阶，系数在此钉死以便做节省账；MoE 把 $N$ 换 $N_{\mathrm{act}}$，1.2 口径）。

**命题 2**（前缀命中节省）：prompt 长 $T_p$、命中前缀长 $\delta$（ $\delta \le T_p$）时，节省的 prefill FLOPs 为

$$
\Delta\mathrm{FLOPs} = 2N\delta + 2L \thinspace d_{\mathrm{model}} \thinspace \delta^{2} \qquad (2)
$$

**符号**： $N$ = 参数量（dense）； $\delta$ = 命中前缀长度（token 数）； $L$、 $d_{\mathrm{model}}$ = 层数、hidden 宽。

**证明**（路线：线性项直接计数，attention 项数 causal 对之差；对数细节留作练习 2）。线性项：前缀 $\delta$ 个 token 的权重 GEMM 整体跳过，省 $2N\delta$。attention 项：全量 prefill 的 causal 对数 $\approx T_p^{2}/2$；命中后剩余 $T_p - \delta$ 个 token（位置 $\delta+1$ 到 $T_p$）仍 attend 各自的全部前驱，对数 $\approx (T_p^{2} - \delta^{2})/2$；省下的对数 = $\delta^{2}/2$，恰为前缀内部的 causal 对，与 $T_p$ 无关。每对对应 score 内积 + PV 累加共 $4 \thinspace d_{\mathrm{model}}$ FLOPs，全层 $L$，故省 $4 \thinspace d_{\mathrm{model}} \times \delta^{2}/2 \times L = 2L \thinspace d_{\mathrm{model}} \delta^{2}$。证毕（对数计数见练习 2）。

代入章脊口径： $T_p$ = 4096、 $\delta$ = 3072（ $h$ = 75%）：线性项 $2 \times 7.5 \times 10^{9} \times 3072$ = 46.1 TFLOP；attention 项 $2 \times 32 \times 4096 \times 3072^{2}$ ≈ 2.47 TFLOP；合计 ≈ 48.6 TFLOP。全量 prefill ≈ $61.4 + 4.4$ = 65.8 TFLOP（与 1.2 实例同值），节省比例 73.7%；按 H100 bf16 dense 989.5 TFLOPS 理想折算省约 49 ms（compute-bound 上界口径，1.3；排队与取 KV 开销另计）。两个边界： $\delta = T_p$（全命中）时式 (2) 退化为全量 prefill FLOPs； $\delta = 0$ 时节省为 0——公式在两端连续，中间单调。

节省比例的一般式。令 $h = \delta / T_p$，式 (2) 除以全量 prefill：

$$
\frac{\Delta\mathrm{FLOPs}}{\mathrm{FLOPs}_{\mathrm{full}}} = h \cdot \frac{N + L \thinspace d_{\mathrm{model}} \thinspace h \thinspace T_p}{N + L \thinspace d_{\mathrm{model}} \thinspace T_p} \qquad (3)
$$

**符号**： $h$ = 命中率（token 加权，§1 口径）；其余同前。两个极限： $L \thinspace d_{\mathrm{model}} T_p \ll N$（短上下文、线性项主导）时比例 $\approx h$； $L \thinspace d_{\mathrm{model}} T_p \gg N$（长上下文、attention 主导）时比例 $\approx h^{2}$。代入 $T_p$ = 4096： $L \thinspace d_{\mathrm{model}} T_p$ = $5.4 \times 10^{8}$ ≪ $N$，比例 73.7% ≈ $h$； $T_p$ = 131072 满窗： $L \thinspace d_{\mathrm{model}} T_p$ = $1.7 \times 10^{10}$ $\gt$ $N$，同 $h$ = 75% 下比例降到 61.9%——长上下文里「剩余 token attend 长前缀」的 attention 省不掉，命中率兑换节省的效率下降。

命中不等于免费：KV 必须在 HBM 里（同机）或传过来（跨机），TTFT 节省 = 计算节省 − 取 KV 开销；跨机带宽账归 2.4 / 8.3。命中率的测量口径也要钉：vLLM 的 prefix cache stats 按 queries / hits 的 token 数统计（`kv_cache_manager.py` `make_prefix_cache_stats`，L228）；benchmark 的 warmup 请求会污染命中率——测稳态命中须排除 warmup 或重置缓存（1.1 benchmark 方法学 recap）。

## 5. TTL 与驱逐：让缓存活多久

DeepInfer 的生产披露：agent 流量 TTL 72 小时、chatbot 流量 7 天，「captures almost all reusable cache entries」——72 小时恰跨周末，后续工作日的请求仍能复用（[DeepInfer doc](../references.md)，教学参照 + 官方实现，官方披露，2026-11-23 核实，doc L211 原文口径）。TTL 取多长，是命中率与容量的权衡。

所谓 TTL（time-to-live），是指缓存条目的最长保留时间，到期由回收机制清除。TTL 长则命中率高、容量占用久；短则反之。边际收益由复用间隔分布决定：教学模型下，命中损失 $\approx P(\Delta t \gt \mathrm{TTL})$（口径： $\Delta t$ = 条目的复用间隔，即相邻两次命中的时间差；每条目的下次复用是随机事件，间隔超过 TTL 的份额即损失；分布形态由 workload 决定）。agent 的 72h 恰跨周末（doc 口径），工作日产生的条目在后续工作日仍存活；chatbot 的命中长尾重——少量 KV 被读极多次，丢失它们损失的命中率不成比例地高（doc L187 口径）。

**两个口径必须分开**（官方披露，2026-11-23 核实）：

- 生产口径：agent 72h / chatbot 7d，近乎无损（doc L211，「almost all reusable cache entries」）。
- 反事实口径：chatbot 流量缩至 3d TTL，损失 0.5%–1% 命中率（doc L187 原文：「a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate」）。

> **坑**：把「0.5%–1% 命中损失」写成 agent 72h TTL 的生产损失是口径混用——损失主体是 chatbot 流量，条件是 TTL 缩到 3 天；两个口径的流量类型与 TTL 值都不同。引用时必须带「chatbot 反事实」限定。

LRU 与 TTL 是互补的两个信号：LRU 按访问新近度回答「容量不够时逐谁」（SGLang 叶优先，§3；vLLM free 队列，2.1 §4），TTL 按条目年龄回答「没有容量压力时留多久」（DeepInfer 3FS GC（garbage collection，垃圾回收）的 LRU + TTL，2.4 展开）。只有 LRU 没有 TTL，冷条目会一直占着容量直到被挤掉；只有 TTL 没有 LRU，容量压力下无法定向腾挪。

多轮对话的跨请求保留之外，还有跨轮次的 KV 搬运问题。CachedAttention（[Gao et al., 2024](../references.md)，arXiv:2403.19708，[B]，无官方 repo——W1 复核维持）把 session（会话，一次多轮对话）KV 放在分层存储（cost-effective 介质），用四个机制把搬运藏进计算：layer-wise pre-loading（逐层 H2D 传输与 forward 重叠；H2D = host-to-device，内存 → 显存）、asynchronous saving（保存与计算重叠）、scheduler-aware fetching / eviction（按调度器 hint 决定取哪层、逐哪层）、位置编码解耦 + KV 截断（防 context window overflow 使旧 KV 失效）。重叠条件可以写成不等式：

$$
\frac{\kappa \thinspace T_p}{\mathrm{BW}} \le t_{\mathrm{layer}} \qquad (4)
$$

**符号**： $\kappa T_p$ = 每层 KV 字节（ $\kappa$ 为每 token 每层字节，2.1 式 (1)）； $\mathrm{BW}$ = H2D 带宽； $t_{\mathrm{layer}}$ = 每层 forward 计算时间。代入章脊口径： $\kappa T_p$ = 4096 × 4096 = 16 MiB/层，PCIe Gen5 128 GB/s（[NVIDIA datasheet](../references.md)，[C]）下 H2D 约 131 μs；每层计算 $2 \times (7.5 \times 10^{9}/32) \times 4096 / 989.5 \times 10^{12}$ ≈ 1.94 ms——131 μs ≪ 1.94 ms，重叠裕量 14.8 倍，逐层搬运完全藏得住。分层存储的正式成本账（各层延迟 × 命中率）归 2.4。

> **口径注**：本例把 PCIe Gen5 128 GB/s 当作单向 H2D 带宽代入（乐观口径）；2.4 §1 钉死同一 datasheet 数为双向合计，单向 D2H / H2D 取半（~64 GB/s）。按 2.4 口径，H2D 约 262 μs、重叠裕量约 7.4 倍——逐层搬运完全藏得住的结论不变。

## 6. 非前缀复用与分布式前缀调度

RAG 把多篇检索文档拼进 prompt——文档相互独立，谁也不是谁的前缀。它们的 KV 还能复用吗？直接拼接不行：各文档的 KV 独立预计算时没见过其他文档，跨文档 attention 缺失。CacheBlend（[Yao et al., 2024](../references.md)，arXiv:2405.16444，[A]，YaoJiayi/CacheBlend）的解法是选择性重算：拼接预计算 KV 后，挑跨文档 attention 偏差大的 token 重算，把融合误差压回可接受范围（重算份额 workload 依赖，论文口径）。质量口径：论文 benchmark 上测试任务损失可忽略——不外推为普适无损。这是前缀不可变性的受控近似：复用不再精确，用选择性重算买回精度。LMCache 集成与传输账归 2.4。

跨机复用引出路由问题。Preble（[Srivatsa et al., 2024](../references.md)，arXiv:2407.00023，[A]，WukLab/preble）做分布式 prompt 调度，核心是 prefix affinity——把请求路由到持有其前缀 KV 的 GPU，命中率优先。冲突随之而来：热前缀的请求集中到少数机器，与负载均衡路由打架（练习 4 给量化例）。

DeepInfer 用专用存储集群解开这个冲突（官方披露，2026-11-23 核实）：block KV 放专用 3FS（DeepSeek Fire-Flyer File System，2.4 走读）存储集群而非本地 host memory，任何 prefill 机可命中任何前缀——「Since any engine can hit any prefix, there is no prefix-affinity routing」（doc L139 原文）。代价是 KV 取回走网络（CHECK 匹配在路由前，§2；取回异步、swap-in 与 forward 重叠，机制与 §5 的不等式同构，正式账归 2.4）。收益除解冲突外还有两条（doc L175 口径）：存储与计算的弹性伸缩（autoscaling）解耦；计算节点故障率比存储节点高一个量级以上（原文「more than an order of magnitude higher」），KV 放存储集群不受计算节点故障牵连。

三条路线对照：

| 方案 | 命中率 | 负载均衡 | 额外代价 |
| --- | --- | --- | --- |
| 本地缓存 + affinity 路由 | 高（前缀局部性） | 差（热前缀集中） | 无传输；热点机过载 |
| 本地缓存 + 负载均衡 | 中（冷机 miss 或重复缓存） | 好 | 重算或 KV 重复占容量 |
| 共享存储 + 负载均衡 | 高（任何机命中任何前缀） | 好 | KV 取回带宽（2.4 账） |

选型不是「共享存储一定好」：取回带宽与 TTFT 预算的账（2.4）、PD 分离下的路由（8.2）决定第三条路线何时划算。DeepInfer 选它的生产理由如上；单机或小集群里本地缓存 + LPM 调度（§3）往往就够了。

## 总结

| 论断 | 回指 |
| --- | --- |
| 前缀缓存的正确性依据 = 前缀不可变；复用跳过 prefill、不跳过 decode | §1、models 2.4 §4 |
| hash 链 $h_i = H(h_{i-1}, \tau_i, e_i)$：每块 $O(1)$ 查找，命中粒度 = $b$ 整数倍，期望尾部损失 $b/2$ | 式 (1)、§2 |
| `extra_keys` / `extra_key` 是隔离键：LoRA / 多模态 / salt 不同则 KV 不可共享 | §2 坑注 |
| 基数树：token 级命中，匹配 $O(T)$，LRU 叶优先 + `lock_ref` 保护，缓存与 running 共池 | 命题 1、§3 |
| LPM ≈ DFS 序：cache $\ge$ 最长请求时最优命中（论文口径）；贪心有饥饿风险 | §3、4.2 指针 |
| 命中节省 = $2N\delta + 2L \thinspace d_{\mathrm{model}} \delta^{2}$：线性项节省比例 $h$，attention 项 $h^{2}$ | 命题 2、式 (2)(3)、练习 2 |
| TTL 边际收益由复用间隔分布定：agent 72h / chatbot 7d 近乎无损；chatbot 3d 反事实损失 0.5%–1%（官方披露） | §5 |
| 逐层重叠条件 $\kappa T_p / \mathrm{BW} \le t_{\mathrm{layer}}$：8B 口径裕量 14.8 倍 | 式 (4)、§5 |
| prefix affinity 与负载均衡冲突；共享存储以取回带宽为代价解冲突 | §6、练习 4 |

## 练习

1.（计算 · [20]）章脊口径（ $N$ = $7.5 \times 10^{9}$、 $L$ = 32、 $d_{\mathrm{model}}$ = 4096）， $T_p$ = 4096、 $\delta$ = 3072：(a) 线性项与 attention 项节省 FLOPs 各多少？(b) 总节省比例？(c) 按 H100 bf16 dense 989.5 TFLOPS 理想折算省多少 ms？校验：(b) 介于 $h^{2}$ = 56.25% 与 $h$ = 75% 之间；(c) 的分子 = (a) 两项之和。
2.（推导 · [20]）补全命题 2 证明中的 attention 对数计数：(a) 写出长 $T_p$ 的全量 prefill causal 对数；(b) 写出命中 $\delta$ 前缀后剩余 $T_p - \delta$ 个 token 的对数（注意剩余 token 仍 attend 复用的前缀）；(c) 作差得省下对数 $\delta^{2}/2$，并解释为何与 $T_p$ 无关。校验： $\delta = T_p$（全命中）时式 (2) 退化为全量 prefill FLOPs（与 1.2 口径一致）； $\delta = 0$ 时节省为 0。
3.（推导 · [30]）基数树操作走读：树初始为空，三条请求依次到达并完成——R1 = [s, a, b, c]（s 为 system prompt），R2 = [s, a, b, d]，R3 = [s, a, e]。(a) 画出每次 insert 后的树（边标与节点边界）；(b) 给出 R2、R3 的 `match_prefix` 命中长度与分裂次数；(c) 设容量只装得下 6 个 token 且三请求均已结束（`lock_ref` = 0），给出 LRU 叶优先的驱逐顺序。校验：(a) 树中 token 总数 = 6（s、a、b、c、d、e 各计一次，边标不重叠）；(b) 每条请求的命中长度 + 新插入长度 = 该请求 token 数（R1/R2 = 4，R3 = 3），与 `_match_prefix_helper` 的部分匹配分裂语义一致；(c) 顺序满足「任一节点在其全部孩子被逐之前不可被逐」的叶优先不变式，且父节点入堆时机与 `evict` 源码一致。
4.（计算 · [30]）prefix affinity 与负载均衡的量化冲突（教学口径）：2 张 GPU，各只缓存得下一个前缀；80% 请求共享前缀 P，20% 共享前缀 Q，稳态 LRU 保留更频繁的前缀。(a) affinity 路由（P→GPU-1，Q→GPU-2）：命中率与负载分布；(b) round-robin 负载均衡：稳态命中率与负载分布；(c) 共享存储（任何机命中任何前缀，取回成本另计）：命中率与负载分布。对照 §6 三路线表，说明 DeepInfer 选 (c) 的理由。校验：三行结果应与 §6 表的定性排序一致——本地缓存下命中率与负载均衡不可兼得；(c) 的命中率论断与 doc L139「any engine can hit any prefix」及 Preble 动机节一致。

## 参考

- [Zheng et al., 2023](../references.md)：SGLang: Efficient Execution of Structured Language Model Programs，arXiv:2312.07104，[A]（key: sglang）——RadixAttention §3、LRU 叶优先 §3.1、cache-aware 调度与 DFS 等价 §3 / 附录 A.3
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——共享前缀 §4.4（prefix caching 先声）
- [Gao et al., 2024](../references.md)：Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention，arXiv:2403.19708，[B]（key: cachedattention，无官方 repo，W1 复核维持）
- [Yao et al., 2024](../references.md)：CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion，arXiv:2405.16444，[A]（key: cacheblend）
- [Srivatsa et al., 2024](../references.md)：Preble: Efficient Distributed Prompt Scheduling for LLM Serving，arXiv:2407.00023，[A]（key: preble）
- [vLLM repo](../references.md)：vllm-project/vllm，[A]（key: vllm-repo）——`v1/core/kv_cache_utils.py` `hash_block_tokens`、`config/cache.py` L269、`kv_cache_manager.py`（2026-11-23 main 核实）
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——`mem_cache/radix_cache.py`、`unified_radix_cache.py`、`managers/schedule_policy.py`（2026-11-23 main 核实；HiRadixCache → UnifiedRadixCache 漂移注见 §3）
- [DeepInfer doc](../references.md)：doc/inference_deployment_qa.md，教学参照 + 官方实现（key: deepinfer-doc，官方披露口径注）——TTL L187 / L211、prefix affinity L139 / L175、CHECK L191、版本元组 L199（2026-11-23 核实）
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——bf16 dense 989.5 TFLOPS、PCIe Gen5 128 GB/s
- 跨维度指引（纯文字）：models 2.4 §4 前缀不可变命题；inference 1.1（benchmark 命中污染）、1.2（prefill FLOPs 账）、1.3（datasheet 参数表）、2.1（块池与 ref_cnt）、2.4（分层存储与取回带宽账）、4.2（调度策略与公平）、5.2 / 6.1（引擎走读）、8.2（PD 路由）
