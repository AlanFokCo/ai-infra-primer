# 10.4 点对点与序列并行通信（Point-to-Point and Sequence-Parallel Communication）

> Ring Attention 声称「fully overlapping the communication of key-value blocks with the computation」——这个「完全」在什么条件下成立？块长 8192 token、跨机 50 GB/s 时通信/计算比 0.977，勉强藏住；换到 NVLink 域内同一几何比值降到 0.066——差在哪？本章 recap models 4.6 Ring Attention 分块机制与 training 5.2 CP 账（不重推），交付四块 GPU 侧增量：send/recv 点对点原语与 transport 底座（8.4 recap）、每步 overlap 条件与 ring 总时间账（命题 1/2）、CP 三模式通信量对照（命题 3 all-gather = ring 恒等）、transport 选择与逻辑环到物理 fabric 的映射 | 前置：models 4.6 §3.2（Ring Attention 分块，recap 源）、training 5.2 §3–§5（CP 账，recap 源）、8.1（原语语义与 API）、8.4（p2p/net transport 与 proxy）、7.2（有效带宽 370.8 GB/s）、7.3（peer 映射）、9.2（GDR 与 NIC 带宽账）、2.4（roofline）、6.3（FA kernel 与 head 符号）；下游 10.2（拓扑感知环构建）、10.3（a2a 拥塞对照）、11.2（实测 overlap timeline） | 锚点：Ring Attention 论文（[Liu et al., 2023](../references.md)，[A]，arXiv:2310.01889，官方 repo haoliuhl/ringattention，JAX GPU/TPU 实现）、NCCL repo（[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / 351f33fd，nccl.h.in send/recv + transport）、H100 datasheet（[NVIDIA, nvidia-datasheet](../references.md)，[C]，BF16 dense 989.5 TFLOPS）、DeepEP（[DeepSeek-AI, deepep](../references.md)，教学参照，双锚 v1.2.1 / main V2.5，CP buffer 对照）——代码锚 2026-11-25 fresh tarball 核实

## 1. 三个问题与本章口径

「通信与计算完全重叠」是定理还是条件？Ring Attention 论文 abstract 写「fully overlapping the communication of key-value blocks with the computation」（[Liu et al., 2023](../references.md)，arXiv:2310.01889，[A]）——本章把这个「完全」变成一个可判定的不等式：每步通信时间不超过计算时间，并解出块长阈值。三个问题：环轮转的 transport 底座是什么，为什么用 send/recv 而不是集合（§3）？每步通信在什么条件下被计算完全藏住，ring 总时间省多少（§4）？all-gather KV / ring pass / Ulysses all-to-all 三种 CP 通信模式的量、驻留、时序各差在哪（§5），逻辑环怎么摆到物理 fabric 上（§6）？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU 节点 × 2**（CP 组 $n = 16$，节点内 NVLink4 + 机间单口 ConnectX-7 NDR 400G）；序列几何取 **training 5.2 练习 3 合成几何**（ $h = d_{\mathrm{model}} = 8192$、MHA $H_q = H_{kv} = 64$、 $d_{\mathrm{head}} = 128$、 $T = 131072$、 $B = 1$、bf16）——该几何是合成口径（training 5.2 题面声明），非任何模型的官方窗口；Llama-3 长上下文阶段 CP16 @ 131072 配置（training 5.2 §6 recap，herd Table 4 口径）是其工程邻近锚。有效带宽：域内 **370.8 GB/s 单向**（7.2 式 (2)，NCCL 模型值 18 × 20.6）、机间 **50 GB/s**（NIC 线速，9.2）；有效算力 $F_{\mathrm{eff}} = 400$ TFLOPS bf16（**教学假设**，沿 training 5.2 练习 3 口径，约 H100 BF16 dense 峰值 989.5 TFLOPS 的 40%，[NVIDIA, nvidia-datasheet](../references.md)，[C]）；域内 $\alpha = 2$ μs（**教学假设**，7.2 §3），机间延迟项取 NCCL 校准 NET 14 μs 作参照（8.4 §4，内置校准值、非实测）。贯穿实例为每步 KV 块消息 $M_{\mathrm{blk}}$（§4）。对比行：同 $T$ 把度数缩到 $n = 8$（单节点 NVLink 域，§4/§6）——只为对照域内 vs 跨机，不换其他口径。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，即 CP 度数；training 5.2 用 $c$、models 4.6 用 $P$，recap 统一译 $n$，附口径注）； $T$ = 序列长（training 5.2 $s \to T$ 翻译）； $B$ = batch（training $b \to B$）； $T_{\mathrm{blk}} = T/n$ = 每 rank 序列块长（序列维分片，非 GEMM tile）； $M$ = 消息字节（本章 $M_{\mathrm{blk}}$ = 每步 KV 块消息、 $M_{\mathrm{KV}}$ = 全序列 K+V 总字节，§5 口径注）； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $H_q, H_{kv}$ = query/KV head 数， $g = H_q/H_{kv}$（沿 6.3）； $d_{\mathrm{head}}$ = head 维， $h = d_{\mathrm{model}} = H_q \cdot d_{\mathrm{head}}$； $\mathrm{bytes}$ = 每元素字节数（bf16 取 2，bytes 不占字母）； $F_{\mathrm{eff}}$ = 有效算力（教学假设）； $t_{\mathrm{comm}}, t_{\mathrm{compute}}, t_{\mathrm{ring}}$ = 每步通信/计算时间与 ring 总时间（时间一律小写 $t$ 下标， $T$ 专归序列长）； $R_{\mathrm{comm}}$ = 通信 ridge（本章局部符号，§4）； $u^{\prime}$ = 半块对工作量单位 = 满块对之 1/4（training 5.2 §4 口径，recap）。 $N$ 专归参数量，本章不占用。

**recap 边界声明**（强制，spec §4.2）：models 4.6 §3.2 Ring Attention 分块机制（Q 常驻 + KV 轮转 + online softmax 逐块合并，数学逐点不变与 FLOPs 总量不变两个不变量）与 training 5.2 §3–§5 CP 账（ring 通信量式 (2)、causal 负载均衡式 (3)(4)、Ulysses 换维式 (5)）**已交付，本章只 recap + 符号翻译（ $c \to n$、 $s \to T$、 $b \to B$、 $a_{kv} \to H_{kv}$），不重推**；online softmax 引理归 models 4.1，FA kernel 的分块摄入与 SRAM 驻留细节归 6.3。本章增量 = **跨设备通信实现**：send/recv transport 底座、每步 overlap 条件推导、三模式对照（all-gather KV 账为本章增量）、双缓冲调度、拓扑映射。**重判声明**：models 4.6 §3.2 原边界「通信调度、overlap 策略与拓扑映射的全部细节归 cluster 维度」——gpu 维度重定界后归本章，本章即该指针的兑现。**代码 scope 声明**：本章 = 机制片段级（NCCL API 签名 + transport 片段锚；模块全量走读归 8.4，DeepEP 全量走读归 10.3）。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

**序列并行口径注**：本章「序列并行」以 CP 族（attention 序列维切分）为主体；training 5.2 §2 的狭义 SP（TP 复制区段激活回收，all-gather / reduce-scatter 对）账已交付该处，transport 底座即 8.1 集合原语，本章不展开。

## 2. Recap：Ring Attention 分块与 CP 账

每 rank 每层前向发多少字节、轮转几步？training 5.2 §3 已把账算到数，本节 recap 并翻译符号，不重推——增量从 §3 的「怎么发」开始。

**分块机制 recap**（models 4.6 §3.2，指针不重推）：序列维切 $n$ 段，设备 $r$ 常驻第 $r$ 段 Q 块（ $T/n$ 行）与第 $r$ 段 KV 块；第 $k$ 步用 online softmax 把本地 Q 块对当前持有的 KV 块累算进 running $(m, \ell, O)$，同时把该 KV 块发给设备 $r+1$、接收来自 $r-1$ 的下一块； $n$ 步轮转后每个 Q 块见完全部 KV 块。两个不变量已在 models 4.6 交付：数学逐点不变（online softmax 分块归一化归纳不变量与 models 4.1 §5 同一个，跨设备只是分块边界移动）与 FLOPs 总量不变（ $O(T^{2})$ 拆 $n$ 份并行）。论文的 blockwise 还覆盖 feedforward（逐位置独立、天然可切）；本章聚焦 attention 的 KV 轮转——唯一有跨位置耦合的部分。官方实现为 JAX（GPU/TPU，shard_map 分片，官方 repo haoliuhl/ringattention，[A]），非 PyTorch/NCCL——本章 transport 级增量用 NCCL send/recv 做底座（§3），调度机制引论文 + JAX repo 佐证。

**ring 通信量 recap**（training 5.2 式 (2)， $c \to n$ 翻译，bf16 MHA 口径）：每步每卡发一个 K 块加一个 V 块， $n-1$ 步合计每卡发送

$$
C_{\mathrm{ring}} = 4 \thinspace B \thinspace T \thinspace H_{kv} \thinspace d_{\mathrm{head}} \cdot \frac{n-1}{n} \hspace{5mu} \approx \hspace{5mu} 4 \thinspace B \thinspace T \thinspace h \qquad (\text{training 5.2 式 (2) recap})
$$

**符号**：系数 4 = 2 个张量（K、V）× 2 bytes（bf16）； $H_{kv}$ = KV head 数（training $a_{kv}$ 翻译）；MHA 口径 $H_{kv} = H_q$、 $h = H_q \cdot d_{\mathrm{head}}$。两个结构性质（training 5.2 已交付，本章 §4 把它变成定量条件）：量系数 $(n-1)/n$ 饱和到 1，**与 $n$ 几乎无关**；每步块计算 ∝ $(T/n)^{2}$、传输 ∝ $T/n$——块越大算术强度越高，长序列恰是 overlap 最友好的区间。代入章脊几何 $n = 16$： $C_{\mathrm{ring}} = 4 \times 131072 \times 8192 \times 15/16 = 3.75$ GiB ≈ **4.03 GB** / 层 / 前向（training 5.2「约 4 GB」口径对账）；反向 dK/dV 沿环反向再轮转一圈，通信量对称翻倍（training 5.2 §3 已交付结论）。

**causal 负载均衡 recap**（training 5.2 §4，不重推）：causal mask 下连续切分 rank $r$ 可见块对数 $r+1$，最大/均值比 $2n/(n+1) \to 2$（ $n = 16$ 时 ≈1.88）；两种修法——striped（位置对 $n$ 取余的等距条带，近似均衡）与 zigzag（ $2n$ 个半块首末配对，rank $r$ 持 $\lbrace r, 2n-1-r \rbrace$，每 rank 工作量恒等 $(2n+1)u^{\prime}$（ $u^{\prime}$ = 半块对工作量单位，满块对之 1/4），驻留与式 (2) 通信账不变）。**归属边界**：两种修法都是后续工作、不在 Ring Attention 原文（training 5.2 §4 已声明，勿张冠李戴）。本章 §4 的调度读法在此基础上：ring 步时间由最重 rank 决定，全 mask 步零计算但 send/recv 不能跳——整块跳过会垮掉环的同步节奏（training 5.2 §4 坑注 recap）。

**Ulysses recap**（training 5.2 §5，不重推）：不轮转 KV，attention 前后各一次 all-to-all 把「沿序列切」换成「沿 head 切」——换维后每卡持完整序列、 $H_q/n$ 个 head，attention 本地精确计算。每层前向每卡发送

$$
C_{\mathrm{ulysses}} = 8 \thinspace B \thinspace T \thinspace h \cdot \frac{n-1}{n^{2}} \hspace{5mu} \approx \hspace{5mu} \frac{8 B T h}{n} \qquad (\text{training 5.2 式 (5) recap})
$$

度数约束 $n \le H_q$（GQA 下收紧到 $H_{kv}$，head 是换维原子）；单对消息 ∝ $1/n^{2}$，有延迟地板。代入 $n = 16$： $C_{\mathrm{ulysses}} = 8 \times 131072 \times 8192 \times 15/256 = 0.46875$ GiB ≈ **0.503 GB**，约为 ring 的 1/8（training 5.2「约 500 MB」口径对账）。两模式的结构差异全在 $n$ 的指数上：ring 饱和、Ulysses 随 $1/n$ 下降——§5 把三模式摆进一张表。

## 3. send/recv 点对点原语：轮转的 transport 底座

轮转的「发当前块给下一 rank、收上一 rank 的块」对应哪个 NCCL 原语？为什么不用集合？

**定义**（点对点原语）：所谓点对点（point-to-point）原语，是指单个发送者与单个接收者按约定的 count 与 datatype 完成数据搬运的通信操作——NCCL 的对应 API 是 ncclSend / ncclRecv（src/nccl.h.in L739 / L758，[NVIDIA, nccl](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd）：

```c
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
```

count = 消息元素数（非字节），peer = 对端 rank 号。doc comment 逐字：「Rank peer needs to call ncclRecv with the same datatype and the same count from this rank」——send/recv 必须配对，count 与 datatype 逐项一致。代入章脊口径：KV 块 $M_{\mathrm{blk}} = 256$ MiB（§4 式 (1)），bf16 count = $256 \times 2^{20} / 2 = 134217728$ 元素；K/V 可打包成一个连续缓冲一次发送，也可在同一 group 内各发一次。**stream 语义 recap**（8.1 §7）：API 返回 = 入队完成，不是传输完成；doc 逐字「This operation is blocking for the GPU」指 GPU 侧 kernel 等待传输完成，不是 CPU API 阻塞——两个口径勿混。

**group 语义**：ring 一步的 send 与 recv 必须融成一组——doc comment 逐字：「If multiple ncclSend and ncclRecv operations need to progress concurrently to complete, they must be fused within a ncclGroupStart/ncclGroupEnd section」（ncclGroupStart L873 / ncclGroupEnd L883；8.1 §7 recap）。教学片段（非优化实现）：

```c
/* Ring Attention 一步：发当前块给 next、收下一块自 prev（教学片段，非优化实现） */
ncclGroupStart();
ncclSend(kv_cur, count, ncclBfloat16, next_rank, comm, comm_stream);
ncclRecv(kv_nxt, count, ncclBfloat16, prev_rank, comm, comm_stream);
ncclGroupEnd();
/* compute stream: block_attn(Q_local, kv_cur) 与上述 send/recv 重叠（§4 时序） */
```

> **坑**：send/recv 不 group 的死锁。触发条件：两 rank 各自先调 ncclSend 再调 ncclRecv、未用 group 融合；现象：「This operation is blocking for the GPU」，send 等对端 recv 入队、两边互等；规避：send+recv 融进 ncclGroupStart/ncclGroupEnd，让两个操作并发推进（doc 逐字要求）。

**为什么不用集合**：NCCL 没有「轮转 + 逐块计算」的集合语义——all-to-all 一次搬全部块，没有逐步 overlap 窗口；ring all-gather 也是阻塞集合（§5）。send/recv group 给出显式的逐步控制：哪一步发、收进哪个缓冲、何时与计算重叠，都由调用方决定。PP 层间激活传递同样是 send/recv（training 5.3 / 8.1 §2 recap）——同一原语、不同流量模式：PP 是流水依赖链（stage $i$ → stage $i+1$ 单向），CP ring 是循环轮转（每 rank 既发又收、邻居双向）。

**transport 底座 recap**（8.4 §4，不重推）：NCCL transport 注册表只有四项 p2p / shm / net / collNet（src/transport.cc L15-20）。环轮转的两条邻居边各落其一：**域内 p2p**——p2pMap（p2p.cc L358）把 peer HBM 映射进本进程地址空间，GPU kernel 直接 load/store，数据路径不经 CPU 拷贝（7.3 peer 映射 recap）；useRead（L410）决定 read/write 语义。**跨机 net**——NIC 的 verbs 接口是 host 侧 API，proxy 线程替 GPU 提交请求、轮询完成（sendProxyProgress，net.cc L1327，8.4 §5 recap）；GDR 让 NIC 直访 GPU HBM、省 host bounce（9.2）。流量模式：ring 轮转是邻居排列——每 rank 只与 prev/next 两个邻居通信，邻居链路承载持续均匀负载；对照 a2a 的完全图突发与 incast（8.1 §4 recap，拥塞账归 10.3）。这个均匀负载正是 ring 能在大规模下稳定重叠的结构原因（§6）。

## 4. 每步 overlap：通信被完全藏住的条件【关键推导】

「完全藏住」是无条件的吗？本节把 abstract 的论断变成可判定不等式并解出块长阈值——结果恰是 roofline 把带宽换成互连。

**每步账**。轮转单位 = 一个 KV 块（K+V， $T_{\mathrm{blk}} = T/n$ 行）。每步消息与块对计算（非 causal 满块对口径）：

$$
M_{\mathrm{blk}} = 2 \thinspace B \thinspace T_{\mathrm{blk}} \thinspace H_{kv} \thinspace d_{\mathrm{head}} \cdot \mathrm{bytes} \qquad (1)
$$

$$
F_{\mathrm{blk}} = 4 \thinspace B \thinspace T_{\mathrm{blk}}^{2} \thinspace H_q \thinspace d_{\mathrm{head}} \qquad (2)
$$

**符号**：式 (1) 系数 2 = K、V 两个张量；式 (2) 系数 4 = 两个 GEMM（score $QK^{\top}$ 与 probs·V）× 每乘加 2 FLOPs。**口径注**：式 (2) 为非 causal 满块对口径；causal 对角块三角 mask 减半、全 mask 块为零——causal 步拍读法见下面推论。代入章脊几何（ $T_{\mathrm{blk}} = 8192$ ）： $M_{\mathrm{blk}} = 2 \times 8192 \times 64 \times 128 \times 2 = 268435456$ B = **256 MiB** ≈ 0.268 GB； $F_{\mathrm{blk}} = 4 \times 8192^{2} \times 8192 = 2^{41} \approx 2.199 \times 10^{12}$ FLOPs。

每步时间（α-β 模型，结构 recap training 4.1 式 (3)）：

$$
t_{\mathrm{comm}} = \alpha + M_{\mathrm{blk}} \thinspace \beta, \qquad t_{\mathrm{compute}} = \frac{F_{\mathrm{blk}}}{F_{\mathrm{eff}}} \qquad (3)
$$

代入机间 50 GB/s（ $\beta = 1/(50 \times 10^{9})$ s/B，延迟项取 NCCL 校准 NET 14 μs 参照）：带宽项 $0.268 \times 10^{9} / (50 \times 10^{9}) \approx 5.369$ ms，延迟项 0.014 ms 占 0.26%——**带宽主导**； $t_{\mathrm{compute}} = 2.199 \times 10^{12} / (400 \times 10^{12}) \approx 5.498$ ms。比值 $t_{\mathrm{comm}} / t_{\mathrm{compute}} \approx 0.977$——勉强藏住，余量仅 2.3%。

**命题 1**（每步 overlap 条件）。带宽主导口径（略 α 项，数值验证见上：占 0.26%），每步通信被计算完全藏住当且仅当

$$
t_{\mathrm{comm}} \le t_{\mathrm{compute}} \quad \Longleftrightarrow \quad T_{\mathrm{blk}} \ge \frac{\mathrm{bytes} \cdot F_{\mathrm{eff}}}{2 \thinspace g \cdot \mathrm{BW}} \qquad (4)
$$

**证明**（路线：代入两时间式 → 约公因子 → 解 $T_{\mathrm{blk}}$，三步）。其一， $M_{\mathrm{blk}} \beta \le F_{\mathrm{blk}} / F_{\mathrm{eff}}$，代入式 (1)(2)： $2 B T_{\mathrm{blk}} H_{kv} d_{\mathrm{head}} \cdot \mathrm{bytes} / \mathrm{BW} \le 4 B T_{\mathrm{blk}}^{2} H_q d_{\mathrm{head}} / F_{\mathrm{eff}}$。其二，两边约去正公因子 $2 B T_{\mathrm{blk}} H_{kv} d_{\mathrm{head}}$，得 $\mathrm{bytes} / \mathrm{BW} \le 2 T_{\mathrm{blk}} g / F_{\mathrm{eff}}$。其三，解出 $T_{\mathrm{blk}} \ge \mathrm{bytes} \cdot F_{\mathrm{eff}} / (2 g \mathrm{BW})$。∎ 正例：章脊几何 $T_{\mathrm{blk}} = 8192$，阈值 $2 \times 400 \times 10^{12} / (2 \times 1 \times 50 \times 10^{9}) = 8000$ token——8192 不低于 8000，勉强成立。边界反例： $T_{\mathrm{blk}} = 4096$（度数翻倍到 $n = 32$）低于阈值，比值 1.95，通信暴露约一半——「长序列 overlap 友好」有定量边界，不是无条件。

**roofline 同构**。定义块对算术强度与通信 ridge：

$$
I_{\mathrm{blk}} = \frac{F_{\mathrm{blk}}}{M_{\mathrm{blk}}} = \frac{2 \thinspace g \thinspace T_{\mathrm{blk}}}{\mathrm{bytes}}, \qquad R_{\mathrm{comm}} = \frac{F_{\mathrm{eff}}}{\mathrm{BW}} \qquad (5)
$$

式 (4) 即 $I_{\mathrm{blk}} \ge R_{\mathrm{comm}}$——**overlap 条件与 2.4 roofline 同构**：把 HBM 带宽换成互连有效带宽、峰值算力换成有效算力，块对就是那个「kernel」。代入： $I_{\mathrm{blk}} = 2 \times 1 \times 8192 / 2 = 8192$ FLOPs/byte， $R_{\mathrm{comm}} = 400 \times 10^{12} / (50 \times 10^{9}) = 8000$ FLOPs/byte——8192 不低于 8000，compute-bound（勉强）。 $I_{\mathrm{blk}}$ 随 $T_{\mathrm{blk}}$ 线性涨：这是 training 5.2「块越大算术强度越高」结构种子的定量兑现。

> **提点**：roofline 的 ridge 在这里不是硬件常数而是配置量—— $F_{\mathrm{eff}}$ 随 MFU 变、BW 随互连代际变， $R_{\mathrm{comm}}$ 跟着动。overlap 条件的工程读法因此有两条路：有效算力更低（MFU 低、 $t_{\mathrm{compute}}$ 更大）更容易藏，或带宽更高（ $t_{\mathrm{comm}}$ 更小）——前者不是优化目标，后者是硬件路径（§6）。

**命题 2**（ring 总时间）。非 causal 均匀口径（每 rank 每步计算相同），同步 ring 步时间 = $\max(t_{\mathrm{comm}}, t_{\mathrm{compute}})$，总时间

$$
t_{\mathrm{ring}} = t_{\mathrm{compute}} + (n-1) \max(t_{\mathrm{comm}},\ t_{\mathrm{compute}}) \qquad (6)
$$

完全隐藏当且仅当 $t_{\mathrm{comm}} \le t_{\mathrm{compute}}$，此时 $t_{\mathrm{ring}} = n \thinspace t_{\mathrm{compute}}$（通信零成本）；否则 $t_{\mathrm{ring}} = t_{\mathrm{compute}} + (n-1) t_{\mathrm{comm}}$（通信主导）。

**证明**（路线：写步依赖链 → 归纳步起点 → 求和，三步）。其一，双缓冲调度的依赖链：compute $k$ 等 recv $k$ 完成（数据依赖），group $k$ 等 compute $k-1$ 完成（缓冲写后读，见下坑）；step 0 算常驻本地块、不受通信门控。其二，归纳：若 $t_{\mathrm{comm}} \le t_{\mathrm{compute}}$，步 $k$ 起点 $k \thinspace t_{\mathrm{compute}}$（recv 已在上一步计算期间完成）；否则步 $k$ 起点 $k \thinspace t_{\mathrm{comm}}$（计算等通信）。其三， $n$ 个计算步、 $n-1$ 个收发步，求和即式 (6)。∎ 代入章脊几何： $t_{\mathrm{ring}} = 16 \times 5.498 = 87.96$ ms；串行基线（无重叠） $t_{\mathrm{serial}} = 16 \times 5.498 + 15 \times 5.369 = 168.49$ ms——**重叠省 47.8%**。口径注：每层 attention 前向、单 microbatch；「完全隐藏」指通信时间被计算完全覆盖、步时间 = 计算时间，不是通信流量为零。

```
双缓冲轮转时序（rank r，缓冲 A/B 交替，非 causal 口径）：

step k    compute 流: [ attn(Q_r, A) ]              [ attn(Q_r, B) ]
          comm 流:    [ send A→r+1 | recv B←r-1 ]   [ send B→r+1 | recv A←r-1 ]
                      └── 一个 ncclGroup ──┘        └── 一个 ncclGroup ──┘
step k+1  依赖: compute 等 recv 完成; group 等上一步 compute 完成 (WAR)
          t_comm ≤ t_compute: comm 段被 compute 段完全覆盖 → 步时间 = t_compute
```

> **坑**：双缓冲写后读（write-after-read, WAR）竞态。触发条件：recv 覆写缓冲 A 时 compute 仍在读 A（上一步的块），两者无 event 依赖；现象：compute 读到被覆写一半的 KV，attention 输出静默错误（无 NaN 哨兵）；规避：comm 流的 group 等 compute 流上一步 kernel 完成 event 再入队 recv——compute-bound 时这个等待免费（通信本来就要等），comm-bound 时它就是步拍本身。

**推论**（causal 步拍）。causal mask 下，含位置 0 的 KV 块对所有晚于它的 query 可见。连续满块轮转下，第 $k$ 轮（ $k \ge 1$ ）rank $k$ 持 KV 块 0，其常驻 $Q_k$ 整体晚于块 0——该步有效计算 = 满块对，每步最重 rank 计算与非 causal 同构，**overlap 条件不变**。证明（路线）：头块可见性 → rank $k$ 的 $`Q_k \times \mathrm{KV}_0`$ 满块对 → 每步 max = 满块对。∎ 代入章脊几何：8192 不低于 8000，causal 连续口径仍勉强藏住。三个配套读法：其一，causal 总计算约减半（zigzag 总量 $(2n+1)u^{\prime}$ ≈ 非 causal $4n u^{\prime}$ 之半，training 5.2 式 (4) recap），通信量不变 → **平均**通信/计算比约翻倍——但步拍由最重 rank 决定，总量节省表现为非最重 rank 的 slack，不是更短的步拍；其二，全 mask 步零计算但 send/recv 不能跳（training 5.2 §4 坑注 recap），其通信被步拍吸收；其三，zigzag 负载均衡下（保量、轮转单位仍为满块），每步最重 rank 计算介于满块对之半与满块对之间——头半块（含位置 0）对全 query 可见，每轮持有它的 rank 的两个 query 半块都看得见它（不低于 2 个可见半块对，即满块对之半），故完全隐藏的充分条件是阈值翻倍（16000 token @50 GB/s）；章脊几何 8192 低于这条保守线，不能保证完全隐藏（精确每步 max 随轮次变化，翻倍是充分非必要）。负载均衡（training 5.2 §4）与 overlap 余量（本章）是轮转设计的两个权衡维度。

**反向一句**：dK/dV 块沿环反向再轮转一圈、dQ 本地累积，通信量对称翻倍（training 5.2 §3 已交付结论 recap）；反向块对计算约 2× 前向（training 1.1 结构口径），overlap 余量更宽——反向比前向容易藏，通信瓶颈读法在前向侧。

## 5. CP 三种通信模式：all-gather KV / ring pass / Ulysses

all-gather KV、ring pass、Ulysses all-to-all 三模式，哪个通信最少？答案可能反直觉：前两者量严格相等，差在驻留与时序。

**all-gather KV 模式**。all-gather 原语（8.1）与 CP 分片（training 5.2）的直接组合：每 rank 常驻 Q 分片（ $T/n$ 行）与本地 KV 分片，attention 前沿 CP 组 all-gather K/V 到全序列，再本地算 Q 分片 × 全量 KV。令全序列 K+V 总字节

$$
M_{\mathrm{KV}} = 2 \thinspace B \thinspace T \thinspace H_{kv} \thinspace d_{\mathrm{head}} \cdot \mathrm{bytes} \qquad (7)
$$

代入章脊几何： $M_{\mathrm{KV}} = 2 \times 131072 \times 8192 \times 2 = 2^{32}$ B = **4 GiB**。

**命题 3**（all-gather = ring 量恒等）。all-gather KV 与 ring pass 每 rank 每层前向发送量严格相等：

$$
C_{\mathrm{AG}} = C_{\mathrm{ring}} = \frac{n-1}{n} \cdot M_{\mathrm{KV}} \qquad (8)
$$

**证明**（路线：各模式数发送步数 × 每步字节，两步）。其一，ring all-gather 每 rank 发 $(n-1)/n \cdot M_{\mathrm{KV}}$（8.1 原语账 recap： $n-1$ 步 × 每步 $M_{\mathrm{KV}}/n$）。其二，ring 轮转每 rank 发 $n-1$ 个 KV 块、每块 $M_{\mathrm{blk}} = M_{\mathrm{KV}}/n$（式 (1)），合计 $(n-1)/n \cdot M_{\mathrm{KV}}$。两者逐项相等。∎ 代入 $n = 16$：均 3.75 GiB ≈ 4.03 GB。正例： $n = 2$ 两者均发 $M_{\mathrm{KV}}/2$；边界反例：把自环（本地 KV 分片）也计发送得 $M_{\mathrm{KV}}$，多计 $M_{\mathrm{KV}}/n = 256$ MiB——自环留本地不发（8.1 命题 1 自环排除口径）。

> **提点**：两模式的差异不在流量而在驻留与时序——all-gather 要全量 KV 驻留（4 GiB），ring 只要常驻分片 + 双工作缓冲（768 MiB，常驻 1 + 工作 2）；all-gather 是 attention 前的阻塞集合（可与 QKV 投影计算重叠、不能与 attention 本身重叠），ring 是逐块 P2P 流水（§4）。同样的字节，不同的关键路径位置。

**三模式对照表**（每 rank 每层前向发送，bf16 MHA，章脊几何 $T = 131072$、 $h = 8192$；反向各对称翻倍，training 5.2 口径）：

| 模式 | 每 rank 发送 | $n=8$ | $n=16$ | $n=32$ | KV 驻留 | 时序 / overlap | 度数约束 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all-gather KV | $(n-1)/n \cdot M_{\mathrm{KV}}$ | 3.76 GB | 4.03 GB | 4.16 GB | 全量 $M_{\mathrm{KV}}$（4 GiB） | 阻塞集合，attention 前 | 无 head 约束 |
| ring pass | $(n-1)/n \cdot M_{\mathrm{KV}}$ | 3.76 GB | 4.03 GB | 4.16 GB | $O(T/n)$ + 双缓冲（768 MiB） | 逐块 P2P 流水，可重叠 | 无 head 约束；causal 需均衡 |
| Ulysses a2a | $8BTh(n-1)/n^{2}$ | 0.94 GB | 0.50 GB | 0.26 GB | 全序列 × head 分片（256 MiB） | 2× a2a，attention 前后 | $n \le H_q$（GQA $H_{kv}$）；消息 ∝ $1/n^{2}$ |

读法：ring/AG 量系数饱和到 1（与 $n$ 几乎无关），Ulysses 随 $1/n$ 下降——ring/Ulysses 比值恰为 $n/2$（ $n = 16$ 时 8，training 5.2「约 1/8」口径对账）。Ulysses 的低量换两个约束：度数不能超 head 数（章脊几何 $H_q = 64$， $n = 32$ 仍合法、 $n = 128$ 非法），单对消息随 $1/n^{2}$ 缩（ $n = 16$ 每 peer 每张量 8 MiB、 $n = 32$ 时 2 MiB、 $n = 64$ 时 512 KiB）——小消息 regime α 项接管（8.1 $M_c$ 读法 recap），延迟地板是其结构边界。ring 无 head 约束， $n$ 可开到序列本身的极限——超长上下文是 ring 的 regime（training 5.2 §6 边界 recap）。

**all-gather 时序账**（增量）：ring all-gather 时间 ≈ $n-1$ 步 × 每步 $M_{\mathrm{KV}}/n$ 过最慢边——代入跨机 50 GB/s： $15 \times 0.268 \times 10^{9} / (50 \times 10^{9}) \approx 80.5$ ms；attention 本体 $n \times t_{\mathrm{compute}} = 87.96$ ms（FLOPs 总量不变，models 4.6 不变量 recap）。两者串行 ≈ 168.5 ms——**恰是 ring 的无重叠串行基线**（§4 命题 2）。ring overlap 把这 168.5 ms 压到 87.96 ms：这是「ring pass 不是量的优化、是时序的优化」的定量读法。all-gather 可与前段 QKV 投影计算重叠部分隐藏，但 attention 需要全量 KV、gather 本身不能与 attention 重叠——关键路径位置与 ring 结构不同。

**buffer 预注册对照**（DeepEP，[DeepSeek-AI, deepep](../references.md)，教学参照，双锚 v1.2.1 / main V2.5）：ring 轮转的工作缓冲须预注册（NCCL buffer 注册、跨机 GDR MR 注册，9.2 recap）——与 EP dispatch/combine 的预注册 EPBuffer/Bucket（V2.5 接口锚，10.3 走读）同一模式：buffer 预注册 + transport 直写，省逐步注册开销。差异在流量模式：ring 是邻居均匀负载（§3），EP a2a 是完全图突发 + 热 expert incast（10.3）——同一预注册模式、两种流量下的拥塞行为不同，拥塞控制归 9.3/10.3。

## 6. transport 选择与拓扑映射：逻辑环到物理 fabric

同一个逻辑环，摆在 NVLink 域内 vs 跨机，overlap 阈值差 7.4×——环的步拍由哪条边决定？

**ring 映射物理拓扑**（10.2 recap 指针）：逻辑环的 $n$ 个 rank 摆到 2 节点 × 8 GPU，每条边落两种物理路径之一——节点内 NVLink（370.8 GB/s 模型单向）或机间 IB（50 GB/s NIC 线速）。**步拍由最慢边决定**：同步 ring 每步所有 rank 同时收发，步时间 = 最慢 rank 的步时间 = $`M_{\mathrm{blk}} / \min(\mathrm{BW}_{\mathrm{intra}}, \mathrm{BW}_{\mathrm{inter}})`$——跨机环的步拍是机间段的 5.369 ms，节点内 14 跳的 370.8 GB/s 只决定非瓶颈 rank 的 slack。

```
逻辑环 (n=16) → 物理 fabric (2 节点 × 8 GPU, rail-optimized):

  朴素 rank 序:  r0→r1→…→r7 → r8→r9→…→r15 → r0
                 机间边 r7→r8, r15→r0 跨 rail (经 spine, 多跳)

  rail 对齐:     r0→r1→…→r7 → r15→r14→…→r8 → r0
                 机间边 r7→r15 (rail 7), r8→r0 (rail 0) — 同 rail 一跳 leaf
  节点内: NVLink 域 370.8 GB/s/跳;  机间: IB 50 GB/s/跳
  步拍 = 最慢边 = 机间 M_blk / 50 GB/s = 5.369 ms
```

rail 对齐环把节点 1 的遍历顺序反转，让两条机间边都落在同 rail 的 GPU 对上（每 GPU 的 NIC 接本 rail 的 leaf，10.1 rail-optimized recap）——避免跨 rail 的 spine 转发。环构建的图搜索与 NCCL_TOPO_FILE 工程归 10.2；本章只登记映射对 overlap 账的影响：**阈值公式的 BW 取环的最慢边带宽**。

**transport 选择账**（域内 P2P vs 跨机 RDMA）。把两种带宽代入式 (4) 阈值：

| 环放置 | 最慢边 BW | 阈值（bf16 MHA， $F_{\mathrm{eff}} = 400$ TFLOPS） | 章脊 $T_{\mathrm{blk}} = 8192$ 读法 |
| --- | --- | --- | --- |
| 单节点 NVLink 域（ $n = 8$ ） | 370.8 GB/s | 1079 token | 比值 0.066，轻松藏住 |
| 跨机 CX7 NDR400（ $n = 16$ ） | 50 GB/s | 8000 token | 比值 0.977，勉强藏住 |
| 跨机 CX8 + Gen5 主机（ $n = 16$ ） | 64 GB/s（PCIe 限，9.2 命题 1 recap） | 6250 token | 比值 0.763，余量放宽 |

$n = 8$ 对比行全账： $T_{\mathrm{blk}} = 16384$、 $M_{\mathrm{blk}} = 512$ MiB、 $t_{\mathrm{comm}} = 0.537 \times 10^{9} / (370.8 \times 10^{9}) \approx 1.448$ ms、 $t_{\mathrm{compute}} = 21.99$ ms——比值 0.066，通信只占计算的 6.6%。raw 9× 带宽差（450/50，7.2 §7 recap）在模型有效口径是 7.4×（370.8/50）——阈值比与带宽比同比例（式 (4) 反比）：8000/1079 ≈ 7.4。这是「CP 尽量留 NVLink 域内」的定量读法。三条工程路径：其一，度数放置——CP 组留 NVLink 域内（ $n \le 8$ ）或 GB200 NVL72 超节点（域扩到 72 GPU，10.1 recap），跨机跳消失；其二，带宽升级——CX8 800G 把跨机有效带宽抬到 PCIe 限 64 GB/s（9.2 regime recap），阈值降到 6250；其三，多 NIC 聚合——rail-optimized 每 GPU 多 NIC 抬每 rank 出口带宽（9.2 命题 2 recap），步拍随之放宽。causal 保守口径（§4 推论 zigzag 充分条件）把跨机阈值翻倍到 16000 token，章脊几何低于线——causal 超长序列对跨机带宽更敏感，是 CP 偏好 NVLink 域的账侧理由。

## 7. 边界：本章不做什么

调度收口，剩下的归谁？本章交付：send/recv 原语与 transport 底座（§3）、每步 overlap 条件与 ring 总时间（命题 1/2，§4）、CP 三模式对照（命题 3，§5）、transport 选择与拓扑映射（§6）。以下各有专章：**online softmax 引理与分块合并正确性**归 models 4.1；**FA kernel 分块摄入、SRAM 驻留、合并实现**归 6.3；**CP 组配置与并行状态走读**归 training 6.2；**NCCL 模块全量走读**（环构建图搜索、proxy 状态机、协议）归 8.4；**fabric 工程**（拓扑感知环构建、rail 对齐、SHARP、多 NIC 调优）归 10.2；**a2a 拥塞与 DeepEP 全量走读**归 10.3；**GDR 硬件机制**归 9.2；**实测 overlap timeline drill-down**（Nsight Systems comm/compute 重叠分析）归 11.2。代码 scope：机制片段级。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 环轮转 = 邻居 P2P 流水；send/recv 配对（count/datatype）、并发推进须 group；API 返回 = 入队；transport 域内 p2p / 跨机 net | nccl.h.in L739/L758/L873/L883；transport.cc L15-20 | §3、8.1/8.4 |
| 每步消息 $M_{\mathrm{blk}} = 2 B T_{\mathrm{blk}} H_{kv} d_{\mathrm{head}} \mathrm{bytes}$；块对 FLOPs $F_{\mathrm{blk}} = 4 B T_{\mathrm{blk}}^{2} H_q d_{\mathrm{head}}$（非 causal） | 式 (1)(2)；256 MiB / $2.199 \times 10^{12}$ FLOPs | §4、练习 1 |
| overlap 条件 $T_{\mathrm{blk}} \ge \mathrm{bytes} F_{\mathrm{eff}} / (2 g \mathrm{BW})$ ⟺ $I_{\mathrm{blk}} \ge R_{\mathrm{comm}}$（roofline 同构）；跨机 8000 / 域内 1079 token | 式 (4)(5)，命题 1 | §4、练习 3 |
| ring 总时间 $t_{\mathrm{ring}} = t_{\mathrm{compute}} + (n-1) \max(t_{\mathrm{comm}}, t_{\mathrm{compute}})$；完全隐藏 ⟺ $t_{\mathrm{comm}} \le t_{\mathrm{compute}}$；章脊 87.96 vs 串行 168.49 ms，省 47.8% | 式 (6)，命题 2 | §4、练习 4 |
| causal 步拍：头块对全 query 可见 → 连续满块轮转每步最重 rank = 满块对、条件同构；总量减半表现为 slack；zigzag 每步 max 不低于满块对之半 → 充分条件阈值翻倍（16000） | §4 推论 | §4、练习 3 |
| 双缓冲 WAR：recv 覆写等 compute 读完；send/recv 不 group 死锁 | §3/§4 坑注 | §3/§4 |
| AG = ring 量恒等 $(n-1)/n \cdot M_{\mathrm{KV}}$；差在驻留（4 GiB vs 768 MiB）与时序（阻塞 vs 流水） | 式 (7)(8)，命题 3 | §5、练习 2 |
| 三模式：ring/AG 饱和 ≈ $4BTh$、Ulysses $8BTh/n$ 随 $1/n$ 降；比值 $n/2$；Ulysses $n \le H_q$ + 消息 ∝ $1/n^{2}$ 延迟地板 | §5 表；4.03 / 0.50 GB @ $n=16$ | §5、练习 2 |
| 步拍 = 最慢边；rail 对齐环机间边同 rail；阈值 BW 取最慢边；raw 9× / 模型 7.4× 带宽差 → 阈值同比例 | §6 表 | §6、10.1/10.2 |

## 练习

1.（计算 · [10]）每步消息与时间代入（章脊口径，式 (1)(2)(3)）：(a) $M_{\mathrm{blk}}$ 多少 MiB、多少 GB；(b) 跨机 50 GB/s 的 $t_{\mathrm{comm}}$ 带宽项多少 ms；(c) $F_{\mathrm{eff}} = 400$ TFLOPS 的 $t_{\mathrm{compute}}$ 多少 ms；(d) 比值与隐藏判定。校验：(a) 256 MiB ≈ 0.268 GB；(b) ≈5.37 ms；(c) ≈5.50 ms；(d) ≈0.977 勉强藏住，与 §4 逐数一致（容差 1%）。
2.（计算 · [20]）三模式通信量对比（式 (7)(8) 与 training 5.2 式 (5)，章脊几何， $n \in \lbrace 8, 16, 32 \rbrace$）：(a) ring/AG 每 rank 发送各多少 GB；(b) Ulysses 各多少 GB；(c) ring/Ulysses 比值；(d) Ulysses $n = 32$ 每 peer 每张量消息多少 MiB，对照 α 项说明延迟地板。先预测哪列随度数饱和、哪列下降，再算。校验：ring 列 ≈3.76/4.03/4.16 GB 几乎不变，Ulysses ≈0.94/0.50/0.26 GB 近似减半；比值 = $n/2$；与 training 5.2 练习 3 口径（c=16 约 4 GB / 约 500 MB）一致。
3.（推导 · [30]）证明命题 1 与 causal 读法：(a) 从 $t_{\mathrm{comm}} \le t_{\mathrm{compute}}$ 推出式 (4) 阈值；(b) 证明 $I_{\mathrm{blk}} = 2 g T_{\mathrm{blk}} / \mathrm{bytes}$ 且条件同构于 $I_{\mathrm{blk}} \ge R_{\mathrm{comm}}$；(c) 证明 §4 推论的头块论证（连续满块轮转第 $k$ 轮 rank $k$ 算 $`Q_k \times \mathrm{KV}_0`$ 满块对），并说明全 mask 步为什么不能跳 send/recv；(d) zigzag $n = 4$ 工作量恒等（块对计数，对角计 1 对，training 5.2 §4 口径）：rank $r$ 持半块 $\lbrace r, 2n-1-r \rbrace$，每个 query 半块恰好见每个不晚于自己的 KV 半块一次（与轮转调度无关），数出每 rank 可见块对总数，验证恒等 $(2n+1)u^{\prime} = 9u^{\prime}$。校验：(a)(b) 与式 (4)(5) 一致；(c) rank $k$ 的 $Q_k$ 整体晚于 $`\mathrm{KV}_0`$；(d) 总量 $9u^{\prime}$ 与 training 5.2 式 (4) $(2c+1)u^{\prime}$ 在 $c=4$ 一致（rank $r$ 总数 $(r+1)+(2n-r) = 2n+1$）。
4.（代码 · [30]）教学实现（非优化实现）：纯 Python 模拟 $n = 8$ 环轮转与双缓冲 overlap 时序：(a) 模拟 KV 块轮转，断言 $n-1$ 步后每 rank 见完全部 $n$ 块、每 rank 发送 $(n-1) \times M_{\mathrm{blk}}$；(b) 模拟步依赖链（compute 等 recv、group 等上一步 compute），断言 $t_{\mathrm{ring}} = t_{\mathrm{compute}} + (n-1) \max(t_{\mathrm{comm}}, t_{\mathrm{compute}})$，compute-bound / comm-bound 两组参数各验一次；(c) 扫 $T_{\mathrm{blk}} \in \lbrace 2048, 8192, 32768 \rbrace$ 输出比值与隐藏判定（50 GB/s、 $F_{\mathrm{eff}} = 400$ TFLOPS）。先写预测再跑。校验：(a) $(n-1)/n \cdot M_{\mathrm{KV}}$ 与式 (8) 一致；(b) 与命题 2 闭式一致；(c) $T_{\mathrm{blk}} = 8192$ 比值 ≈0.977 与 §4 一致。

## 参考

- [Liu et al., 2023](../references.md)：Ring Attention with Blockwise Transformers for Near-Infinite Context，arXiv:2310.01889，[A]（key: ring-attention）——blockwise 机制与「fully overlapping the communication of key-value blocks with the computation」abstract 逐字；官方 repo haoliuhl/ringattention（原 lhao499/llm_large_context 301 redirect，JAX GPU/TPU 实现，ringattention_jax.py / shard_map）；分块机制 recap 源 models 4.6、CP 账 recap 源 training 5.2（§2/§4）
- [NVIDIA, nccl](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e）fresh tarball 核实——src/nccl.h.in ncclSend L739 / ncclRecv L758（count/datatype 配对 + 「This operation is blocking for the GPU」+ group 逐字）/ ncclGroupStart L873 / ncclGroupEnd L883；src/transport.cc L15-20 p2p/shm/net/collNet；p2p.cc p2pMap L358 / useRead L410；net.cc sendProxyProgress L1327（模块全量走读归 8.4）（§3/§6）
- [NVIDIA, nvidia-datasheet](../references.md)：NVIDIA H100 datasheet（key: nvidia-datasheet），[C] 官方规格披露·一手权威——BF16 dense 989.5 TFLOPS（ $F_{\mathrm{eff}}$ 教学假设的峰值参照）（§1）
- [DeepSeek-AI, deepep](../references.md)：deepseek-ai/DeepEP repo（key: deepep），教学参照（官方实现），双锚 tag v1.2.1（机制）/ main V2.5（接口）——EPBuffer/Bucket 预注册对照、a2a incast 归 10.3（§5）
- models 4.6 / 4.1（纯文字指引，不挂链接）：Ring Attention 分块机制与两个不变量（4.6 §3.2）、online softmax 引理（4.1 §5）——recap 不重推（§2/§4）
- training 5.2 / 5.3 / 4.1 / 1.1（纯文字指引，不挂链接）：CP ring 通信量式 (2)、causal 均衡式 (3)(4)（striped/zigzag 归属）、Ulysses 式 (5) 与度数约束（5.2 §3–§5）、PP send/recv（5.3）、α-β 模型（4.1 式 (3)）、反向 2× 结构口径（1.1）——recap 不重推（§2/§3/§4/§5）
- 2.4 / 6.3 / 7.2 / 7.3 / 8.1 / 8.4 / 9.2 / 9.3 / 10.1 / 10.2 / 10.3 / 11.2（纯文字指引，不挂链接）：roofline ridge（2.4）、FA kernel 与 head 符号（6.3）、有效带宽 370.8 GB/s 与 α 教学假设（7.2）、peer 映射（7.3）、原语语义 API 与 $M_c$（8.1）、transport/proxy 模块走读（8.4）、GDR 与 NIC 带宽 regime（9.2）、拥塞控制（9.3）、rail-optimized 与 NVL72（10.1）、拓扑感知环构建（10.2）、a2a 拥塞 DeepEP（10.3）、实测 overlap timeline（11.2）——recap 与下游（§1/§3/§4/§5/§6/§7）
