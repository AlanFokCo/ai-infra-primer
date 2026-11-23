# 6.1 SGLang 架构与 RadixAttention 实现（SGLang Architecture and RadixAttention Implementation）

> 一个含多次 gen 调用的 Python 程序如何落到 GPU 上执行，前一次调用的 KV 又凭什么自动成为后一次调用的缓存？本章把 SGLang 的前端语言与 srt 运行时逐文件走读，RadixAttention 的基数树与两级内存池对回 2.2 的公式，并按调度 / KV / IPC 三轴对照 vLLM 的架构选择 | 前置：2.2（RadixAttention 机制与命中节省账）、2.1（块表与块池）、4.1（迭代级调度循环）、1.4（结构化输出原理）、models 2.4（前缀不可变命题） | 锚点：SGLang [Zheng et al., 2023] [A]（arXiv:2312.07104 + sgl-project/sglang）；sglang-repo [A]（`python/sglang/{lang,srt}`，2026-11-23 main fresh tarball file:line 核实）；XGrammar [Dong et al., 2024] [A]、Outlines [Willard & Louf, 2023] [A]、guidance（教学参照 + 官方实现）；SGLang v0.4 blog [C]（开放技术博客、非闭源）

## 1. 两层协同设计：前端语言与运行时

多调用 LLM 程序——agent 控制、few-shot 评测、branch-solve-merge prompting——的每一步都要拼接字符串、发起调用、等待结果、再拼接。用 OpenAI 式裸 API 写这类程序，字符串操纵与并行控制全部手工，论文报告等价程序需要 2.1× 的代码行数（[Zheng et al., 2023](../references.md)，arXiv:2312.07104，[A]，§2 口径）。SGLang 的回答是把程序抽象成一层嵌入式 DSL（domain-specific language，领域特定语言），再为这层 DSL 协同设计一个运行时：论文摘要报告在 agent / 推理 / JSON 解码 / RAG / 多轮对话等 workload 上至多 6.4× 吞吐提升（摘要口径，workload 依赖，baseline 含 Guidance / vLLM / LMQL）。

所谓 SGLang 系统，是指前端语言（frontend language）与 SGLang Runtime（SRT）后端两层的组合：前端提供 gen / select / fork 等原语操纵 prompt 状态，后端负责调度、KV 复用与约束解码。论文 §2 把它归入 low-level 系统（与 LMQL / Guidance 同族，区别于 LangChain / DSPy 等 high-level 系统），差异点在「comes with its own co-designed runtime」——前端知道运行时能做什么复用，运行时知道前端的程序结构，两层互相喂信息。

接口契约只有一条：**前端发送完整 prompt，运行时做前缀匹配与复用**。论文 §3.1 原文：「The frontend interpreter sends full prompts to the runtime, and the runtime performs prefix matching and reuse.」前端不需要知道 KV 缓存里有什么、在哪台机器；命中判定完全在后端的基数树里完成（机制见 2.2 §3，本章 §4 走读实现）。这条契约是「自动跨调用复用」的全部含义——多调用程序的每一次 gen 都带全量前缀到达，重复前缀的 KV 由运行时跳过重算。

**本章数值口径**（沿用 2.2 / 4.1 章脊，全章例题复用）：Llama-3.1-8B 几何（ $L$ = 32、 $H_{kv}$ = 8、 $d_{\mathrm{head}}$ = 128、 $d_{\mathrm{model}}$ = 4096、bf16）， $\kappa L$ = 128 KiB/token（每 token 全层 KV 字节）； $N = 7.50 \times 10^{9}$（matmul 口径，1.2）；硬件 H100 SXM（bf16 dense 989.5 TFLOPS、HBM 3.35 TB/s，[NVIDIA datasheet](../references.md)，[C] 官方规格披露）。代码锚点一律 sgl-project/sglang main 2026-11-23 fresh tarball（路径相对 `python/sglang/`），与论文 / blog 口径的差异逐处给漂移注。

```
前端 lang/                          后端 srt/
┌─────────────────────┐   完整 prompt   ┌──────────────────────────────┐
│ interpreter.py      │ ──────────────→ │ TokenizerManager（tokenize）  │
│  StreamExecutor     │                 │        │ ZMQ PUSH             │
│  gen/select/fork    │                 │ Scheduler（基数树 + 调度循环） │
│ tracer.py（图模式）  │ ←────────────── │        │ 同进程调用            │
└─────────────────────┘   流式结果      │ TpModelWorker → ModelRunner   │
                                        │        │ ZMQ PUSH             │
                                        │ DetokenizerManager（增量解码） │
                                        └──────────────────────────────┘
```

## 2. 前端语言：gen / select / fork

一个原语集合能覆盖多少多调用模式？SGLang 的答案是六类：生成（gen）、选择（select）、prompt 状态操纵（`+=` / `[name]`）、并行（fork / join）、多模态输入（image / video）、约束（regex / json_schema 参数）。论文 §2 的 running example 是一个 essay judge：先 select 判断文章与图片是否相关，再 fork 出三个副本并行评分，merge 后 gen 总结，最后按 regex 约束输出 JSON 字母等级。

代码落点（`lang/api.py`，2026-11-23 核实）：

| 原语 | 锚点 | 语义 |
| --- | --- | --- |
| gen | `api.py` L75 | 调模型生成，结果存入命名变量；支持 regex / json_schema / choices / stop / n 等参数 |
| select | `api.py` L236 | 从候选列表选概率最高项（gen 的 choices 参数在 L102 直接转成 SglSelect） |
| fork | `lang/interpreter.py` StreamExecutor.fork L367（程序级 wrapper L885 委托；tracer 模式 `tracer.py` L108；backend 接口 `base_backend.py` fork_program L38） | 复制 prompt 状态成 size 个并行副本 |
| image / video | `api.py` L228 起 | 多模态输入挂到 prompt 状态 |

gen 的签名值得走读一段（机制摘录，省略部分采样参数）：

```python
# sglang/lang/api.py gen（main 2026-11-23，机制摘录）
def gen(name=None, max_tokens=None, n=None, stop=None,
        temperature=None, top_p=None, ...,
        choices=None, regex=None, json_schema=None):
    if choices:
        return SglSelect(name, choices, ...)   # L102 choices → select
    if regex is not None:
        re.compile(regex)                      # L111 前端先验证 regex 合法
    return SglGen(name, max_tokens, ..., regex, json_schema)
```

原语返回的是**描述对象**（SglGen / SglSelect），不是立即执行——执行发生在 interpreter 把描述对象提交到异步流之后。论文 §2 的执行模式口径：interpreter 模式下每个 prompt 由一个后台线程的 StreamExecutor 管理，extend / gen / select 是非阻塞提交，「This is similar to launching CUDA kernels asynchronously」；取结果（`s["name"]`）才阻塞等待。程序内并行由此而来：fork 出的多个副本各自持有流，互不等待。另一条执行路径是 tracer 把程序编译成计算图交 graph executor（论文 Appendix D），本章只走读 interpreter 模式。

fork 的复制语义（`interpreter.py` L367，机制摘录）：

```python
# sglang/lang/interpreter.py StreamExecutor.fork（main 2026-11-23，摘录）
def fork(self, size=1, position_ids_offset=None):
    if size > 1 and str(self.text_):
        self.submit(SglCommitLazy())     # 先把已提交前缀落盘
    self.sync()
    exes = [StreamExecutor(...) for _ in range(size)]
    for i in range(size):
        exes[i].variables = dict(self.variables)
        exes[i].text_ = str(self.text_)          # 前缀逐字复制
        exes[i].fork_start_text_pos = len(self.text_)
    return exes
```

差异点（教学摘录 vs 官方）：官方还复制 messages_ / images_ / cur_role 等状态字段，摘录只留 KV 复用相关的主干。复制的是 prompt **文本**，不是 KV——KV 复用发生在后端：size 个副本的 prompt 共享同一段前缀，运行时匹配到同一棵子树。论文 §3.1 的 Frontend Hint 机制更进一步：fork 时前端先把公共前缀作为 hint 发送，确保它被插入树中，再发各副本的剩余 prompt——前端用程序结构知识帮运行时提前建缓存，这是 co-design 的具体含义。

多调用复用的账由 2.2 命题 2 给出（recap 一句：命中 $\delta$ 前缀省 $2N\delta + 2L \thinspace d_{\mathrm{model}} \delta^{2}$ FLOPs，推导见 2.2 §4，不重复）。代入本章口径的一个最小实例：character description 程序 `s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`（论文 §5 例），context 1000 token、name 输出 20 token，则第二次 gen 的 prompt 前 1020 token 命中第一次调用留下的 KV，只有 `"job:"` 几个 token 走 prefill——线性项省 $2 \times 7.5 \times 10^{9} \times 1020$ ≈ 15.3 TFLOP。

## 3. 后端 srt：三进程链与 ZMQ IPC

一条 HTTP 请求从进入到流式返回，经过哪几个进程？SGLang 的答案是三个管理进程加一个执行体：TokenizerManager、Scheduler、DetokenizerManager 各占一个进程，TpModelWorker 与 Scheduler 同进程（§8 对照 vLLM 的关键差异）。入口是 `entrypoints/engine.py` 的 Engine（L266，launch_server / launch_engine 入口族；权重更新与显存释放接口 update_weights_from_ipc L1790 / release_memory_occupation L1899 / resume_memory_occupation L1905 归 6.3 / 9.3）。HTTP 层是 FastAPI（`entrypoints/http_server.py`），offline 用法直接构造 Engine。

进程间通道是 ZMQ PUSH/PULL（`managers/scheduler_components/ipc_channels.py` SchedulerIpcChannels，L17，2026-11-23 核实）：

```python
# sglang/srt/managers/scheduler_components/ipc_channels.py（main 2026-11-23，摘录）
recv_from_tokenizer = get_zmq_socket(
    context, zmq.PULL, port_args.scheduler_input_ipc_name, False)   # L36
send_to_tokenizer = get_zmq_socket(
    context, zmq.PUSH, port_args.tokenizer_ipc_name, False)         # L48
send_to_detokenizer = get_zmq_socket(
    context, zmq.PUSH, port_args.detokenizer_ipc_name, False)       # L59
```

通道地址族在 `server_args.py` PortArgs（L826）：单机用 `ipc://` + tempfile 路径，多机用 tcp 地址（L897-899 / L979 起）。消息体是 msgspec 结构（`managers/io_struct.py`，3043 行），四个关键载荷：

| 结构 | 锚点 | 方向与内容 |
| --- | --- | --- |
| GenerateReqInput | io_struct.py L220 | HTTP → TokenizerManager：text / input_ids / image_data / 采样参数 / rid |
| TokenizedGenerateReqInput | L1205 | TokenizerManager → Scheduler：token id 序列 + 采样参数 + 多模态特征 |
| BatchTokenIDOutput | L1832 | Scheduler → DetokenizerManager：output_ids_flat + offsets（按行切分）、finished_reasons、logprobs |
| BatchStrOutput | L1942 | DetokenizerManager → TokenizerManager：增量解码后的文本 |

一次请求的完整数据流（ASCII，锚点均 2026-11-23 main 核实）：

```
HTTP POST /generate
  → TokenizerManager（tokenizer_manager.py L608；ReqState L281 挂请求状态；
                      _tokenize_one_request L1275 做 tokenize）
  → ZMQ PUSH（TokenizedGenerateReqInput）
  → Scheduler（scheduler.py L513，7322 行；ingest_requests 收进 waiting）
      每步循环：get_next_batch_to_run L4386 → run_batch L5455
  → TpModelWorker（tp_worker.py L283；forward_batch_generation L624）
  → ModelRunner（model_executor/model_runner.py L332）→ ForwardBatch L442 → GPU
  → BatchTokenIDOutput → ZMQ PUSH
  → DetokenizerManager（detokenizer_manager.py L80；event_loop L164；
                        增量 detokenization，机制见 1.4 §7）
  → BatchStrOutput → ZMQ PUSH → TokenizerManager → HTTP stream
```

Scheduler 是单进程多 mixin 的组织（L513-519）：SchedulerDisaggregationDecodeMixin / SchedulerDisaggregationPrefillMixin（PD 分离，6.3）/ SchedulerPPMixin（流水并行）/ SchedulerDllmMixin（diffusion LLM，9.4 展望）/ SchedulerMlxOverlapMixin（MLX 平台）。调度循环与 overlap 变体的完整走读归 6.2；本章只登记执行侧的批形态枚举 ForwardMode（`model_executor/forward_mode.py` L20）：

| ForwardMode | 语义 | 机制章 |
| --- | --- | --- |
| EXTEND | prefill（含命中前缀后的剩余段） | 2.2 / 4.1 |
| DECODE | 逐 token 一步 | 4.1 |
| MIXED | chunked prefill 与 decode 混批 | 4.3 |
| IDLE | DP attention 下无序列 worker 的空转批（陪跑 collective） | 6.2 §8 |
| TARGET_VERIFY / DRAFT_EXTEND_V2 | 投机解码 verify / draft 形态 | 4.5 |
| PREBUILT | PD 分离 decode 侧 KV 就绪批 | 6.3 |
| DLLM_EXTEND | diffusion LLM 扩展 | 9.4 展望 |

执行链上 TpModelWorker（BaseTpWorker L83 的实现）持有 ModelRunner，forward_batch_generation（L624）把 ScheduleBatch 转成 ForwardBatch（forward_batch_info.py L442）再调 forward + sample——forward 与 sample 的分离接口是 overlap 调度的接线点，6.2 §2 走读。

## 4. RadixAttention 实现走读：radix_cache.py ↔ 2.2

2.2 §3 已交付机制：基数树存 token 序列 → KV 索引映射、LRU 叶优先驱逐、lock_ref 保护、cache-aware 调度，匹配复杂度 $O(T)$（2.2 命题 1）。本节走读实现本体 `mem_cache/radix_cache.py`（839 行），增量是数据结构字段级语义与代码 ↔ 公式对照，不重复推导。

**RadixKey**（L60）：树的查找键 = token_ids 序列 + extra_key（LoRA adapter / 隔离命名空间）+ cache_salt（采样盐 / 缓存版本隔离），另有 is_bigram（EAGLE 投机的 bigram 视图）与 limit（O(1) 截断）两个扩展字段。extra_key 的隔离语义与 2.2 §2 坑注一致：token 相同而 extra_key 不同的条目永不共享前缀节点（match_prefix docstring 逐字：「Entries that share identical leading token ids but have different extra_key values are intentionally kept disjoint」）。page_aligned（L148）把键长截到 page_size 整数倍——page = 1 时原样返回。

**TreeNode**（L251）：children（defaultdict，child key = 首 page 的 token 元组）/ parent / key / value / lock_ref / last_access_time / hit_count / priority。关键字段是 value——**KV 池索引 tensor，不是 KV 本体**；evicted 属性（L275）定义为 value is None，即索引已归还池、节点只剩树拓扑。

**RadixCache**（L282，BasePrefixCache 实现）持有 req_to_token_pool 与 token_to_kv_pool_allocator 两个池引用（§5）加 page_size / eviction_policy。四个公共操作：

match_prefix（L369）走 _match_prefix_helper（L658）：

```python
# sglang/srt/mem_cache/radix_cache.py _match_prefix_helper（main 2026-11-23，摘录）
child_key = key.child_key(self.page_size)        # 首 page 作字典键
while len(key) > 0 and child_key in node.children.keys():
    child = node.children[child_key]
    prefix_len = child.key.match(key, page_size=self.page_size)
    if prefix_len < len(child.key):              # 部分匹配 → 分裂
        new_node = self._split_node(child.key, child, prefix_len)
        value.append(new_node.value); node = new_node; break
    value.append(child.value); node = child
    key = key[prefix_len:]                       # 整段匹配 → 下行
    if len(key): child_key = key.child_key(self.page_size)
```

返回 MatchResult.device_prefix_len = 各段 value 长度之和——这就是 2.2 式 (2) 的 $\delta$（命中前缀长度）。逐边比较走 RadixKey.match（L179）：指数搜索（gallop）加二分定位首个分歧 token，C 级切片比较、无逐 token Python 循环，结果向下取整到 page_size——2.2 命题 1 的 $O(T)$ 比较上界不变，常数落在 C 层。_split_node（L684）在匹配边界把节点一分为二：`new_node.value = child.value[:split_len].clone()`，只拷贝索引 tensor 切片，KV 本体留在池里不动（2.2 命题 1 的「分裂不增加 KV 占用」在代码里就是这一行）。

insert（L433）走 _insert_helper；请求侧入口 _insert_cache（L455）先从 req_to_token 行取本请求的 KV 索引，插入后调 `free_segment(kv_indices[cache_protected_len : result.prefix_len])`（L509-512）。释放段要读对：匹配前缀在请求行里是**树自有索引的借用**——match 时 protect_prefix 把 [0, cache_protected_len) 标为树有（mem_cache/common.py L230-231；schedule_batch.py L282-284 注释逐字：「The request's own KV is [cache_protected_len, kv_allocated_len) … Tree cache owns [0, here) (matched or inserted)」），「The protected prefix is not this req's to free」（common.py L420-421）——释放借用段会腐蚀树。free 段非空仅当 prefix_len > cache_protected_len，即请求**自算**的 KV 与**执行期间新入树**的内容重复：并发同前缀竞态下两请求都 miss、各算一份相同前缀，后完成者插入时树已拥有该前缀，其自算段 [protected, prefix_len) 被回收归池。这是「缓存与 running 共池」（论文 §3.1）的回收半边，也是 6.2 §6 in-batch dedup「避免重复 prefill」的代码落点。

evict（L559，完整机制摘录）：

```python
# sglang/srt/mem_cache/radix_cache.py evict（main 2026-11-23，摘录）
eviction_heap = [(self.eviction_strategy.get_priority(node), node)
                 for node in self.evictable_leaves]     # LRU: last_access_time
heapq.heapify(eviction_heap)
while num_evicted < num_tokens and len(eviction_heap):
    _priority, x = heapq.heappop(eviction_heap)
    self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
    num_evicted += len(x.value)
    self._delete_leaf(x)
    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
        heapq.heappush(eviction_heap, ...)   # 父变成叶且无锁 → 入堆候选
```

与 2.2 §3 机制摘录的差异：当前实现把优先级抽象成 eviction_strategy.get_priority（LRU 之外可配 priority-aware 策略），并维护 KV 事件记录（kv_events.record_remove）；主干语义不变。

lock_ref 的维护是 inc_lock_ref（L589）/ dec_lock_ref（L604）：从节点沿 parent 走到 root，逐节点 ±1；0→1 时把 len(node.key) 从 evictable_size_ 转入 protected_size_，1→0 转回。可驱逐叶集合由 _update_leaf_status（L797）维护，入集条件三个合取：节点非 evicted、lock_ref == 0、所有孩子均已 evicted。由此得到锁保护命题。

**命题 1**（锁保护）：lock_ref > 0 的节点不会被 evict 驱逐。**证明**（路线：evict 的候选集不变式——evictable_leaves 只含 lock_ref == 0 的节点，且增删与分裂操作都保持它）。基例：reset 后集合为空，空虚成立。保持：inc_lock_ref 对路径上每个节点 +1 后调 _update_leaf_status，lock_ref > 0 的节点被移出集合，dec_lock_ref 对称转回；_split_node 让 new_node 继承 child.lock_ref——new_node 新建、从未入集，child 的 lock_ref 与 children 均不变，集合无需改动即保持；_insert_helper 新增叶时对父与新节点调 _update_leaf_status（L762-763，新叶 lock_ref = 0 入集、父因有孩子出集）；_delete_leaf 只对已弹出（lock_ref == 0）的叶操作，其父入堆前显式检查 lock_ref == 0。终止：evict 只从该集合弹节点，故任何 lock_ref > 0 的节点始终不在候选中。证毕。正例：running 请求匹配到的整条路径被 inc_lock_ref 加锁，容量压力下逐掉的是其他会话的叶。边界反例：若 inc_lock_ref 只锁叶不锁路径，祖先节点 lock_ref 仍为 0，孩子被逐后祖先成为可驱逐叶——running 请求的 KV 索引悬空，命题失效；这正是「沿路径加锁」不可省略的原因。

代码 ↔ 2.2 公式对照：

| 代码量 | 公式 / 机制对应 | 出处 |
| --- | --- | --- |
| MatchResult.device_prefix_len | 命中长度 $\delta$，节省 $2N\delta + 2L d_{\mathrm{model}} \delta^{2}$ | 2.2 式 (2) |
| key.page_aligned(page_size) | 命中按 page 对齐；page > 1 的尾部损失同 $b$ 对齐账 | 2.2 §2 |
| lock_ref / protected_size_ | 引用计数保护（vLLM ref_cnt 同构） | 2.1 §4 |
| free_segment | 请求自算重复段归还（借用前缀受 protect_prefix 保护，不在释放之列） | 2.1 §4、§4 |
| eviction_strategy.get_priority | LRU 叶优先（论文 §3.1 原文口径） | 2.2 §3 |

## 5. 两级池：req_to_token_pool 与 token_to_kv_pool

KV 索引存在哪，KV 字节又存在哪？SGLang 把两者拆成两级池：请求级映射表（req_to_token_pool）与 token 级 KV 本体（token_to_kv_pool），基数树只管索引的共享与驱逐。

**ReqToTokenPool**（`mem_cache/memory_pool.py` L382，docstring：「A memory pool that maps a request to its token locations」）：核心是一张二维表 `req_to_token = torch.zeros((size+1, max_context_len), dtype=torch.int32)`——行 = 请求槽位（req_pool_idx），列 = token 位置，值 = KV 池的 slot 索引。第 0 行是 padding 行（源码注释逐字：「cuda-graph padded batches default req_pool_indices to 0, so dummy reads/writes land here harmlessly」）——CUDA Graph 捕获的批大小固定，运行时批不足则补 req_pool_idx = 0 的假行，读写全落在 padding 行上，真实请求不受影响（6.2 §5 的 FutureMap slot 0 与此同构）。free_slots 从尾部弹出分配（O(need_size)），req_generation 逐行单调递增，供读者检测行复用。

**KVCache 本体**（KVCache ABC L1949）按注意力几何分实现：MHATokenToKVPool（L2282）的 k_buffer / v_buffer 是逐层 tensor 列表，NHD 布局 shape 为 $[\text{size}+\text{page},\ H_{kv},\ d_{\mathrm{head}}]$（_kv_buffer_shapes L2666；HND / vectorized_5d 布局选项服务特定 backend，一句注）；MLATokenToKVPool（L4875）只有一份 kv_buffer，shape $[\text{rows},\ 1,\ d_{\mathrm{latent}}+d_{\mathrm{rope}}]$——单 latent、无 K/V 两份。对照 1.2 的 $\kappa$ 口径：MHA/GQA 每 token 每层 $2 H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes}$，MLA $(d_{\mathrm{latent}} + d_{\mathrm{rope}}) \cdot \mathrm{bytes}$（无因子 2、无 $H_{kv}$ 乘）——池的 buffer 形状就是 $\kappa$ 公式的直接编码。DSATokenToKVPool（L5467）另加 indexer K，FP4 / MXFP8 量化池（L3648 / L3874）与 MambaPool（L521）各按几何扩展，目录已扩展（2026-11-23 核实），不展开。

**分配器**分 page 粒度两档：`allocator/token.py` TokenToKVPoolAllocator（L28）page = 1 token——free_pages = arange(1, size+1)，slot 0 留作 padding（注释：「The padded slot 0 is used for writing dummy outputs from padded tokens」），alloc 直接切 free_pages 前缀；`allocator/paged.py` PagedTokenToKVPoolAllocator（L132）page > 1——发给请求的索引恒 page 对齐，decode 续写时按 `(seq_lens-1) % page_size == 0` 判断是否开新 page（Triton kernel 加速）。论文口径 page = 1 token（§3.1 原文：「the size of each page is equivalent to one token」）；当前实现 page_size 可配置，大于 1 时命中与分配都按 page 对齐——2.2 §2 的尾部损失账（期望 $b/2$）对 page 同样成立。

两级映射的 shape 走读：

```
req_pool_idx = 7 的请求，位置 t 的 KV：
  req_to_token[7, t] = 4242            （请求级表，int32 索引）
  k_buffer[layer][4242] → [H_kv, d_head]   （token 级池，KV 字节）
  v_buffer[layer][4242] → [H_kv, d_head]
基数树 TreeNode.value = [4242, 4243, ...]  （索引切片，可共享、可分裂）
```

代入章脊口径算容量账：KV 池 48 GiB（2.1 §1 同池）， $\kappa L$ = 128 KiB/token，则 slot 数 = $48 \times 2^{30} / 131072$ = 393216 token。请求级表自身的开销：max_running_requests = 2048 行 × max_context_len = 4096 列 × 4 B = 32 MiB，占 KV 池的 0.065%——索引表用 int32 换紧凑，代价是 max_context_len 上限固定（超长上下文要调表宽）。page = 1 对 vLLM $b$ = 64 的命中粒度优势 recap 2.2 §2：64 对齐的期望尾部损失 32 token ≈ 4 MiB KV，page = 1 无此损失；代价是 free_pages 管理按 token 计、元数据操作次数更多（6.2 §1 的 CPU 开销来源之一）。

> **坑**：旧教程与早期笔记里的 `HiRadixCache` 类在当前 main 已不存在——重构为 UnifiedRadixCache（`unified_radix_cache.py` L224）+ `hybrid_cache/`（HybridCacheController L325，HiCache L1 GPU / L2 host / L3 storage 三层控制器）。触发条件：照旧类名写走读锚或 grep 代码；现象：类名落空。规避：纯 GPU 缓存锚 RadixCache（本章），分层缓存锚 UnifiedRadixCache + hybrid_cache/，三层成本账 recap 2.4 §9 不重复（版本日期 2026-11-23）。storage/ 后端已扩至 17+（file / hf3fs / mooncake_store / nixl / lmcache 等），swa_radix_cache / pure_swa_radix_cache 服务 SWA 几何，均一句注不展开。

## 6. compressed FSM 结构化输出：论文机制与当前 backend

一段确定的 JSON 前缀 `{"summary": "` 为什么要花 5、6 次 forward 才能解码完？因为逐 token 的 FSM 运行时每步只走一条边，即使下一 token 唯一确定也要跑一次完整的模型前向。论文的 compressed FSM（§4 + Appendix B）就是对这个浪费动手。1.4 §5 已交付 grammar → token mask 的原理（recap 一句：约束解码 = 把语法自动机的当前状态翻译成 vocab 上的 0/1 mask，乘进 logits），本节走读 SGLang 的论文机制与当前实现两个口径。

论文机制分四步。其一，**FSM 建在字符 / 串级而非 token 级**（Appendix B 逐字：「we build the original FSM on characters/strings instead of on tokens」）。其二，**singular transition edge** 定义：源节点只有一个后继、且边上只有一个可接受字符 / 串的边；**compressed edge** = 把连续相邻的 singular 边递归合并进前一条边，边文本 = 各段拼接。其三，**Jump Forward**：解码走到压缩边时，边上的长确定串可以提前预判（anticipate）、跳过若干 token 一次 forward 完成——「multiple tokens on the compressed transition edge can be decoded in one forward pass」（§4 逐字）。其四，**retokenize 保对齐**：跳过的串要转回 token 喂后续解码，直接切分会改变 tokenization 含义（论文例：`{"summary": "` 只能切成 `{"` / `summary` / `":` / `_"`，不能切成 `{"` / `summa` / `ry`），所以用原 tokenizer 对全部前文 + 压缩边文本重新 tokenize，开销很小（Appendix B 逐字：「only brings minor retokenization overhead」）。

数值实例（教学假设口径）：设该确定串对应 $k$ = 6 个 token（实际 $k$ 由 tokenizer 决定），逐 token 解码需 6 次 forward，jump forward 需 1 次，省 5 次 forward；按章脊步时 5.13 ms（4.1 式 (3)， $B$ = 8 批内口径）教学折算约 25.7 ms——对 JSON 重 workload（每个请求多段固定键名）这是逐请求的常数节省（口径：批内其他请求仍占步时，单请求 walltime 节省视批组成而定，此处只计 forward 次数账）。

论文自认的开放问题也要登记：**概率分布偏斜**。压缩边改变了模型「看到」的选项空间——`"[ABCD][+-]?"` 例中，若选项换成 Excellent / Above Average 等长词，模型可能把 A 误映射到 Above Average，因为它在压缩边上看不见完整的选项集合；精确计算每个选项的概率要对所有导向它的 token 序列求和，开销大。论文的 workaround 是把 choices / regex 直接放进 prefill prompt 让模型感知选项，但「doesn't solve the underlying issue of distorted probabilities」（Appendix B 逐字，open problem）。

当前实现是另一个口径（2026-11-23 main 核实）：`constrained/grammar_manager.py` GrammarManager（L37）管 grammar 队列与 backend 创建；backend 族三件——XGrammarGrammar（xgrammar_backend.py L78）/ XGrammarGrammarBackend（L201，**默认 backend**，[Dong et al., 2024](../references.md)，arXiv:2411.15100，[A]）、GuidanceBackend（llguidance_backend.py L224，经 llguidance Rust 移植集成，[guidance](../references.md)，教学参照 + 官方实现）、OutlinesGrammarBackend（outlines_backend.py L88，[Willard & Louf, 2023](../references.md)，arXiv:2307.09702，[A]）。v0.4 blog 报告 XGrammar 集成使 JSON decoding 至多 10× faster（[SGLang v0.4 blog](../references.md)，[C] 开放技术博客、非闭源，--grammar-backend xgrammar）。

> **坑**：把论文的 compressed FSM / jump-forward 当成当前默认 runtime 的行为。触发条件：照论文 §4 描述预测当前 SGLang 的约束解码路径；现象：当前 main 的 jump-forward 符号仅存于 `constrained/inkling2_grammar.py` try_jump_forward（L229，返回 None 的 stub，jump_and_retokenize 直接 raise），默认 XGrammar backend 走 mask 路线、无 jump。规避：论文机制按原始 runtime 口径引用，当前实现锚 GrammarManager + 三 backend（版本日期 2026-11-23）；两个口径都给，不混写。

## 7. API speculative execution：调用级的投机

黑盒 API 模型没有 KV cache 可复用，多调用程序的 input token 费怎么省？论文 §5 的 API speculative execution 把投机从 token 级挪到调用级：第一个 gen 调用**忽略 stop condition 续生成若干 token**，interpreter 保留额外输出，与后续 primitive 的期望前缀匹配复用；匹配成功则后续调用免发，失败则正常重发。

代码落点（`lang/backend/openai.py`，2026-11-23 核实）：L100 注释「API speculative execution」；_prepare_spec_execution（L109）把采样参数的 stop 剔除、max_tokens 覆写为 num_api_spec_tokens（L116），并断言多次调用的采样参数一致（L132，不一致则投机匹配无意义）；开关是装饰器参数 `@function(num_api_spec_tokens=128)`（L153 错误提示逐字给出用例）；重试上限 spec_max_num_tries = 3（L104）。

论文结果（§6 逐字）：GPT-3.5 三字段抽取 prompt，few-shot 下匹配准确率高，「reduces the cost of input tokens by about threefold due to the extraction of three fields」。代入教学算例：context 1000 token、三字段各输出 20 token，naive 三次调用的 input ≈ 3 × 1000 + 已生成字段回填 + 模板 ≈ 3060 token；投机一次调用的 input 只有 context + 首字段模板 ≈ 1002 token（续生成的 128 token 上限覆盖三字段输出，output 费两种方案相近），input 费比值 ≈ 3.05×，与论文 ~3× 口径一致（口径：只计 input token 费；匹配失败重试时成本回升，few-shot 准确率是前提）。

与投机解码（4.4 / 4.5）的区别必须钉死：API speculative execution 省的是**调用次数与 input 费**，投机对象是「后续调用的输出前缀」，不改变单次调用内部的逐 token 生成，也没有 draft-verify 的分布保持证明义务——输出仍是 API 模型自己的生成，只是 stop 时机被提前预判。token 级投机解码（draft 模型 + verify + 拒绝采样）是另一机制，两者名字相近、层次不同。

## 8. 与 vLLM 架构对照：调度 / KV / IPC 三轴

两个引擎都有 continuous batching、prefix caching、chunked prefill、投机解码与结构化输出，差异在哪？不在机制有无，而在组织粒度。三轴对照（vLLM 侧事实 recap 2.1 / 2.2 已交付内容，模块全量走读见 5.1 / 5.2，纯文字指引）：

| 轴 | SGLang（本章锚点） | vLLM V1（5.1 / 5.2 走读） |
| --- | --- | --- |
| 调度 | Scheduler 单进程多 mixin（scheduler.py L513），循环与 TP worker 同进程；prefill / decode 两批对象步边界合并（4.1 §6） | EngineCore 独立进程，Scheduler 与 worker 分进程；统一 token 视图（无 phase 标签），worker 侧缓存请求状态、每步只传 diff |
| KV | 基数树 token 级命中（page = 1 可配）+ 两级池（req_to_token / token_to_kv，§5）；LRU 叶优先 + lock_ref | block pool + hash 链（ $b$ = 64 默认，2.1 §3 版本锚定）；FreeKVCacheBlockQueue LRU + ref_cnt |
| IPC | ZMQ PUSH/PULL 三进程链（TokenizerManager / Scheduler / DetokenizerManager），msgspec 载荷（io_struct） | ZMQ + msgpack frontend ↔ EngineCore；executor ↔ worker 走共享内存 MessageQueue（shm_broadcast） |

读表三个论断。其一，调度轴的差异是**进程边界画在哪**：SGLang 把调度与执行放同进程省一次 IPC，代价是 scheduler 进程兼做 GPU launch（6.2 的 overlap 正是把这份 CPU 工作藏进 GPU 时间）；vLLM 把 EngineCore 隔离成纯 busy loop，tokenize / detokenize / 流式全部外置重叠。其二，KV 轴的差异是**命中粒度与查找结构**：token 级 radix 对多调用 / fork workload 命中更细（§2 的 Frontend Hint 只有 radix 路线能用），block hash 链对大块顺序前缀查找常数更小（2.2 §2 对照）。其三，IPC 轴上两者同选 ZMQ 做管理面，差异在数据面：vLLM 的 executor ↔ worker 用共享内存环（每步高频小消息），SGLang 的 scheduler ↔ worker 同进程直接调用——进程拓扑决定通道选型，不是偏好问题。

选型论断（9.2 展开）：结构化多调用程序、fork 密集 / 前缀细碎复用的 workload，SGLang 的前端-运行时 co-design 是结构性优势；通用模型生态与硬件覆盖的广度是 vLLM 的组织优势。三主引擎的完整对照（含 DeepInfer）归 8.4 / 9.2，本章只立 P6 侧基线。

## 总结

| 论断 | 回指 |
| --- | --- |
| SGLang = 前端 DSL + co-designed 运行时；接口契约 = 前端发完整 prompt、运行时做匹配复用 | §1、论文 §2 / §3.1 |
| gen / select 返回描述对象，interpreter 异步流执行；fork 复制 prompt 文本而非 KV，Frontend Hint 先插前缀 | §2、api.py L75 / L236、interpreter.py L367 |
| 三进程链 TokenizerManager → Scheduler → DetokenizerManager，ZMQ PUSH/PULL + msgspec；TpModelWorker 与 Scheduler 同进程 | §3、ipc_channels.py L17 |
| ForwardMode 八值枚举批形态：EXTEND / DECODE / MIXED / IDLE / TARGET_VERIFY / DRAFT_EXTEND_V2 / PREBUILT / DLLM_EXTEND | §3、forward_mode.py L20 |
| TreeNode.value = KV 索引而非本体；分裂只拷贝索引切片；device_prefix_len = 2.2 式 (2) 的 $\delta$ | §4、radix_cache.py L251 / L684 / L369 |
| 匹配前缀 = 树自有索引借用（protect_prefix 保护）；free_segment 只归还请求自算的重复段，顺序命中情形归还 0 | §4、common.py L420-421、练习 4 |
| 锁保护命题：lock_ref > 0 的节点不可驱逐；inc_lock_ref 沿路径加锁不可省 | 命题 1、§4、练习 2 |
| 两级池：req_to_token [rows, max_context_len] int32 + 逐层 KV buffer；slot 0 / 行 0 为 CUDA Graph padding；MLA 单 latent 无因子 2 | §5、memory_pool.py L382 / L2282 / L4875 |
| page = 1 无对齐尾部损失（对 vLLM $b$ = 64 期望损失 32 token）；page 可配后损失同账 | §5、2.2 §2 |
| compressed FSM = 字符级 FSM + singular 边合并 + jump forward + retokenize；分布偏斜是论文自认 open problem；当前默认 XGrammar mask 路线、无 jump | §6、论文 Appendix B、inkling2 L229 |
| API speculative execution 省调用级 input 费（~3×，GPT-3.5 三字段口径），非 token 级 draft-verify | §7、openai.py L109、练习 3 |
| 三轴对照：调度 = 进程边界位置；KV = 命中粒度与查找结构；IPC = 进程拓扑决定通道 | §8、5.1 / 5.2 指针 |

## 练习

1.（计算 · [10]）章脊口径（ $\kappa L$ = 128 KiB/token）：(a) KV 池 48 GiB 装得下多少 token slot？(b) req_to_token 表 2048 行 × 4096 列 × int32 占多少字节，占池的百分比？(c) 若 max_context_len 扩到 131072（128K），表开销变为多少？校验：(a) slot 数 × $\kappa L$ = 48 GiB；(b) 与 §5 的 32 MiB / 0.065% 一致；(c) 按列数线性放大。
2.（证明 · [20]）把命题 1（锁保护）的证明独立写全：(a) 写出 evictable_leaves 的入集条件（_update_leaf_status L797 的三个合取）；(b) 逐一论证 inc_lock_ref / _split_node / _delete_leaf 三个操作保持「lock_ref > 0 的节点不在集合中」；(c) 构造反例：若 inc_lock_ref 只锁叶不锁路径，给出一个两请求共享前缀的树状态与驱逐序列，使 running 请求的 KV 索引悬空。校验：(c) 的反例必须触发 evict 源码 L580 的父入堆条件（children 空且 lock_ref == 0）；(b) 与命题 1 证明的保持步一致。
3.（计算 · [20]）API speculative execution 成本账（§7 教学算例扩展）：context 2000 token、四字段各输出 30 token、字段间模板 5 token，num_api_spec_tokens = 256。(a) naive 四次调用的 input token 总量；(b) 投机一次调用的 input token 量（续生成覆盖四字段）；(c) input 费比值；(d) 若匹配失败率 20%（失败则后续调用正常重发），期望 input 费比值变为多少？校验：(c) 应接近论文 §6 的 ~3× 口径方向（字段数越多省越多）；(d) 的失败分支成本 = naive 成本，按全概率公式加权。
4.（代码 · [30]）读 `mem_cache/radix_cache.py` 的 match_prefix / _split_node / _insert_cache（main 2026-11-23），先写出预测再对照源码：树初始为空，page = 1，三请求依次到达并完成——R1 = [s, a, b, c]，R2 = [s, a, b, d]，R3 = [s, a, e]（s 为 system prompt）。(a) R2 / R3 的 device_prefix_len 与分裂次数各是多少？(b) R2 插入时 free_segment 归还多少 token 的索引？若 R2 与 R1 并发到达（树空、两者都 miss、各自自算全前缀），R2 后完成时归还量又是多少？(c) 若 page_size = 2，R2 的命中长度变为多少（page_aligned 截断），对照 page = 1 的结果说明对齐损失？校验：(a) 命中 + 新插入 = 请求 token 数（与 2.2 练习 3 的树演化一致）；(b) 归还量 = max(0, prefix_len − cache_protected_len)（_insert_cache L509-512）：顺序情形 R2 两者相等（均 3），归还 0——命中段是借用树索引，无重复可还；并发变体 R2 的 protected = 0、prefix_len = 3，归还 3；(c) 命中按 page 对齐向下取整（L148-152），损失 = page=1 命中 − page=2 命中。

## 参考

- [Zheng et al., 2023](../references.md)：SGLang: Efficient Execution of Structured Language Model Programs，arXiv:2312.07104，[A]（key: sglang）——前端原语与执行模式 §2、RadixAttention 与 Frontend Hint §3.1、compressed FSM §4 + Appendix B、API speculative execution §5 / §6、6.4× 摘要口径
- [SGLang repo](../references.md)：sgl-project/sglang，[A]（key: sglang-repo）——`lang/{api,interpreter}.py`、`lang/backend/openai.py`、`srt/entrypoints/engine.py`、`srt/managers/{tokenizer_manager,scheduler,detokenizer_manager,tp_worker,io_struct}.py`、`srt/managers/scheduler_components/ipc_channels.py`、`srt/mem_cache/{radix_cache,memory_pool,unified_radix_cache,common}.py` + `allocator/{token,paged}.py`、`srt/managers/schedule_batch.py`（L282-284 KV 所有权注释）、`srt/model_executor/{forward_mode,forward_batch_info,model_runner}.py`、`srt/constrained/`（均 2026-11-23 main fresh tarball file:line 核实）
- [SGLang v0.4 blog](../references.md)：SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs，LMSYS，2024-12-04，[C]（key: sglang-v04-blog，开放技术博客、非闭源）——XGrammar JSON decoding 至多 10× faster 口径
- [Dong et al., 2024](../references.md)：XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models，arXiv:2411.15100，[A]（key: xgrammar）——当前默认 grammar backend
- [Willard & Louf, 2023](../references.md)：Efficient Guided Generation for Large Language Models，arXiv:2307.09702，[A]（key: outlines）——outlines_backend
- [guidance](../references.md)：guidance-ai/guidance + llguidance，教学参照 + 官方实现（key: guidance）——llguidance_backend 集成
- [Kwon et al., 2023](../references.md)：Efficient Memory Management for Large Language Model Serving with PagedAttention，arXiv:2309.06180，[A]（key: vllm）——§8 对照的 paged 路线论文锚点
- [NVIDIA datasheet](../references.md)：H100 SXM 官方 datasheet，[C]（key: nvidia-datasheet，官方规格披露、开放产品页）——bf16 dense 989.5 TFLOPS、HBM 3.35 TB/s
- 跨维度指引（纯文字）：models 2.4（前缀不可变命题）；inference 1.2（ $\kappa$ 几何口径）、1.4 §5 / §7（grammar mask 原理、增量 detokenization）、2.1（块池 / ref_cnt / 容量账）、2.2（RadixAttention 机制与命中节省账）、2.4 §9（HiCache 三层）、4.1 §6（调度循环机制级）、4.3（MIXED 批）、4.5（投机形态）、5.1 / 5.2（vLLM 架构与模块全量走读）、6.2（overlap 调度与 ForwardMode.IDLE）、6.3（PD / 权重更新）、8.4 / 9.2（引擎对照与选型）、9.4（dllm 展望）
