# 8.3 Tree / 双二叉树算法（Tree and Double-Binary-Tree Algorithms）

> ring 的延迟项 $2(n-1)\alpha$ 随 $n$ 线性涨，万卡下小消息怎么撑？tree 把步数压到 $2\lceil \log_2 n \rceil$，可单棵二叉树带宽不最优——NCCL 双二叉树怎么既带宽最优（每 rank $\approx 2M$）又延迟 $O(\log n)$？ring/tree 交叉临界 $M^{\ast}$ 为什么在理想 α-β 模型里解不出、必须引入实测有效带宽差 $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$？本章 recap training 4.1 练习 3（recursive halving/doubling）与式 (2)(3)，交付五块增量：tree 延迟推导（ $2\lceil \log_2 n \rceil$ 步 vs ring $2(n-1)$ 步）、double binary tree 带宽最优性（命题 1，Sanders 2009 学术出处 + NCCL blog）、ring/tree 交叉 $M^{\ast}$ 闭式（命题 2，实测 $\beta_{\mathrm{eff}}$ 差）、binomial/k-ary 变体与 NVLink-aware 分层 tree、NCCL 算法选择逻辑（ring/tree/NVLS 按消息 + 拓扑，8.4 回链） | 前置：training 4.1 §2.2 / 练习 3（recap 源）、8.2（ring 带宽最优性下界、消息量与 α-β）、8.1（regime 临界 $M_c$）、7.1 §6（NVLS）、7.2（有效带宽 370.8 GB/s）；下游 8.4（NCCL 算法选择与 cost model 模块走读）、10.2（分层 tree 与 fabric） | 锚点：NCCL 2.4 double binary tree blog（[NVIDIA, NCCL 2.4 blog](../references.md)，[C] 开放技术博客）、Sanders-Speck-Träff 2009（[Sanders et al., 2009](../references.md)，[B] 老经典无 arXiv，Parallel Computing 35(12):581-594，双树学术出处）、Rabenseifner 2004（[Rabenseifner, 2004](../references.md)，[B]）、NCCL repo（[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1 / commit 351f33fd，trees.cc ncclGetDtree + tree.cc 延迟项）、nccl-tests（[NVIDIA, nccl-tests](../references.md)，教学参照）——blog 2026-11-25 逐字核、经典 Crossref 再核、代码 fresh tarball 核实

## 1. 三个问题与本章口径

ring 带宽最优（8.2 命题 1），可它的延迟项 $2(n-1)\alpha$ 随 $n$ 线性涨—— $n=10000$ 时约 40 ms（8.2 §7），小消息下成为瓶颈。tree 把消息步数压到 $2\lceil \log_2 n \rceil$，延迟项降到 $O(\log n)$；但单棵二叉树做 all-reduce 每 rank 流量可达 $3M$、root 还是热点，带宽不最优。NCCL 的双二叉树怎么同时拿到「带宽最优」与「对数延迟」两个好处？ring 与 tree 的交叉临界 $M^{\ast}$ 怎么解——为什么理想 α-β 模型里两者带宽项相同、根本解不出交叉，必须靠实测有效带宽差？本章回答三个问题：tree 的延迟项怎么从步数推出来、双二叉树怎么补上带宽最优？ring/tree 交叉 $M^{\ast}$ 的闭式怎么立、实测 $\beta_{\mathrm{eff}}$ 差从哪来？binomial/k-ary 变体与分层 tree 怎么权衡、NCCL 按什么选算法？

**本章口径**（章脊锁定）：主口径 **HGX H100 8-GPU**（ $n = 8$，NVLink4）；有效带宽 **NCCL 模型 370.8 GB/s 单向**（7.2 式 (2)）； $\alpha = 2$ μs **教学假设口径**（7.2 §3）；贯穿实例 $M = 1$ GB；bf16。交叉临界段引入 tree 有效带宽 $\beta_{\mathrm{tree,eff}}^{-1} = 130$ GB/s 为**教学假设口径**（低于 ring 的 370.8，示意 tree 实测带宽折扣，非实测报告；真实值以 nccl-tests 实测为准）。万卡延迟对比段换 $n = 1024 / 24576$ 只为展示对数 vs 线性，除该段外不再换。

符号声明（spec §4.3）： $n$ = rank/GPU 数（**本章 $n$ 专归 rank 数**，recap training 4.1 时 $d \to n$ 翻译）； $M$ = 被归约向量总字节； $\alpha$ = 每消息固定延迟（**本章 $\alpha$ = 消息延迟，非 inference 接受率**）； $\beta$ = 每字节传输时间 = 有效带宽倒数； $\beta_{\mathrm{ring,eff}}$ / $\beta_{\mathrm{tree,eff}}$ = ring / tree 的**实测有效**每字节时间（本章局部符号，§4）； $M^{\ast}$ = ring/tree 交叉临界消息尺寸。 $N$ 专归参数量、 $B$ 专归 batch，本章不占用；时间一律 $T_{\mathrm{ring}}$ / $T_{\mathrm{tree}}$ 下标。

**recap 边界声明**（强制，spec §4.2）：training 4.1 §2.2 式 (2)(3)（ring 消息量 + α-β）与练习 3（recursive halving/doubling 步数）**已交付，本章只 recap + $d \to n$ 翻译，不重推归纳**；training 4.1 §2.2「大消息下两者带宽项相同、树形赢的只是延迟项」结论本章 recap 不重推。本章增量 = tree 延迟项定量对比 + double binary tree 带宽最优性证明 + ring/tree 交叉 $M^{\ast}$（spec §6.3 强制重推导，实测 $\beta_{\mathrm{eff}}$ 差口径）+ 变体 + NCCL 选择逻辑。NCCL 算法选择与 cost model 模块全量走读归 8.4，机间 fabric 归 10.2。**代码 scope 声明**：本章 = 机制片段级（双树构建锚 trees.cc ncclGetDtree + 延迟项锚 tree.cc；模块全量走读归 8.4）。**测量口径**沿 7.2：本机无 NVIDIA GPU，数值账为 Python 独立重算 + 官方披露口径，非新跑 benchmark。

## 2. tree 延迟推导：对数步数 vs 线性步数

tree 凭什么把延迟项压下来？先 recap training 4.1 练习 3 的 recursive halving/doubling 步数，再增量推出延迟项对比。

**recursive halving/doubling 步数**（recap training 4.1 练习 3，Rabenseifner 2004，[Rabenseifner, 2004](../references.md)，[B]）： $n = 2^k$ 时，recursive halving（归约）每步与距离 $2^j$ 的 peer 交换半块、走 $\lceil \log_2 n \rceil$ 步，recursive doubling（广播）对称再走 $\lceil \log_2 n \rceil$ 步，合计 $2\lceil \log_2 n \rceil$ 步；每 rank 字节数仍约 $2M$ 量级（带宽项与 ring 同量级）。步数推导已交付 training 4.1 练习 3，本章不重推，只取结论。

**增量：延迟项对比**。每步付一次固定延迟 $\alpha$，故 tree 延迟项 $2\lceil \log_2 n \rceil \alpha$、ring 延迟项 $2(n-1)\alpha$。tree 的时间模型（带宽项系数与 ring 同为 $2(n-1)/n$，理想模型口径，§4 展开）：

$$
T_{\mathrm{tree}} = 2\lceil \log_2 n \rceil \thinspace \alpha + 2 \thinspace \frac{n-1}{n} \cdot M \cdot \beta \qquad (1)
$$

**符号**： $\lceil \log_2 n \rceil$ = 树深（向上取整）；其余沿 8.2 式 (2)。代入 $n=8$、 $\alpha=2$ μs：tree 延迟项 $2 \times 3 \times 2 = 12$ μs，ring 延迟项 $2 \times 7 \times 2 = 28$ μs——tree 延迟项是 ring 的 $3/7$。差距随 $n$ 拉大：代入 $n=1024$，tree 延迟项 $2 \times 10 \times 2 = 40$ μs，ring $2 \times 1023 \times 2 = 4092$ μs，比值约 102×； $n=24576$ 时 tree 树深 $\lceil \log_2 24576 \rceil = 15$、延迟项 $60$ μs，ring $2 \times 24575 \times 2 = 98300$ μs ≈ 98 ms。**对数 vs 线性**是 tree 在大规模小消息下的全部优势来源。NCCL 2.4 blog 披露 Summit 24576 GPU 实测延迟最高 180× 改善（[NVIDIA, NCCL 2.4 blog](../references.md)，[C]，逐字「up to 180x improvement at 24k GPUs」）——这是端到端实测口径（含带宽项与 fabric 交互），不是纯延迟项比值（理想延迟比值约 1638×），两者口径不同勿混。

## 3. double binary tree：带宽最优 + 对数延迟

单棵二叉树为什么带宽不最优、双树怎么补上？先算单树的账，再证双树达界。

**单棵二叉树的带宽缺陷**。二叉树 all-reduce = 归约上行 + 广播下行。设每 rank 持 $M$ 字节：归约上行时每个内部节点从两个孩子各收 $M$、合并后发 $M$ 给父亲；广播下行时从父亲收 $M$、发给两个孩子各 $M$。于是一个有 2 个孩子的内部节点发送 $M$（上行）+ $2M$（下行）= $3M$，root 收 $2M$ 是热点。每 rank 流量可达 $3M$，远高于 ring 的 $2(n-1)/n \cdot M \approx 2M$——单树延迟虽 $O(\log n)$，带宽却不合格。

**双互补树**。所谓双二叉树（double binary tree），是指两棵内部节点/叶子位置互补的二叉树，各处理一半数据，使每个 rank 至多在一棵树里承担内部节点的转发负载——从而把单树的带宽缺陷补掉、同时保留对数延迟。其结构依据 NCCL 2.4 blog 逐字（[NVIDIA, NCCL 2.4 blog](../references.md)，[C]）：「half or less ranks in a binary tree are nodes and half (or more) ranks are leaves. Therefore, we can build a second tree using leaves as nodes and vice-versa... There might be one rank which is a leaf on both trees but **no rank is a node on both trees**」。关键结构：二叉树里内部节点至多占一半、叶子至少占一半，故可建第二棵树把第一棵的叶子当内部节点、内部节点当叶子——**没有 rank 在两棵树里都是内部节点**。两棵树各处理一半数据（ $D = M/2$），叠加后每 rank 至多在一棵树里当内部节点、另一棵当叶子。

**命题 1**（double binary tree 带宽最优）。双互补树各处理 $M/2$ 数据，负载均衡下每 rank 发送 $\approx 2M$，与 ring 的 $2(n-1)/n \cdot M$ 同 leading term（带宽最优，blog 逐字「as optimal as rings in terms of data sent/received」）。

**证明**（路线：单树内部节点/叶子分别计数 → 互补性叠加 → 与 ring 同 leading term，三步）。其一，单树处理 $D = M/2$：内部节点（2 孩子）发送 $3D$（上行 $D$ + 下行 $2D$），叶子发送 $D$（仅上行）。其二，互补性：每 rank 至多在一棵树是内部节点（发 $3D$）、另一棵是叶子（发 $D$），合计 $3D + D = 4D = 4 \times M/2 = 2M$；至多一个 rank 在两棵都是叶子（发 $2D = M$，更省）。其三， $2M$ 与 ring 的 $2(n-1)/n \cdot M = 2M - 2M/n$ 同 leading term（差 $2M/n$， $n \to \infty$ 时相对差趋 0），故带宽最优。∎ 代入 $n=8$、 $M=1$ GB：双树每 rank $\approx 2$ GB，ring 1.75 GB，差 0.25 GB（ $2M/n$）；大 $n$ 下两者都趋 $2M$。学术出处 Sanders-Speck-Träff 2009（[Sanders et al., 2009](../references.md)，[B]，「Two-tree algorithms for full bandwidth broadcast, reduction and scan」，Parallel Computing 35(12):581-594，NCCL blog ref [6] 自引）；延迟侧双树深 $\lceil \log_2 n \rceil$、上行 + 下行 $2\lceil \log_2 n \rceil \alpha$，仍 $O(\log n)$。**所以双树同时拿到带宽最优与对数延迟**——这是它取代单树的原因。

NCCL 的双树构建代码是 `ncclGetDtree`（src/graph/trees.cc L90，[NVIDIA, NCCL](../references.md)，教学参照 + 官方实现，版本锚 2.32.7-1，fresh tarball 核实）：第一棵用 `ncclGetBtree`（L32，power-of-two 位模式最大化局部性），第二棵偶数 rank 用 mirror（`nranks-1-rank` 翻转）、奇数 rank 用 shift（移一位），注释逐字「mirror tree (if nranks is even)... shift it by one rank (if nranks is odd)」（L69-88）。代码自带的双树图（12-rank 偶数 mirror 例，trees.cc L69-88 注释逐字；左 = 第一棵 btree、右 = mirror 第二棵，各处理 $M/2$）：

```
0---------------8                   3----------------11
         ______/ \                 / \______
        4         \               /         7
      /   \        \             /        /   \
    2       6       10         1        5      9
   / \     / \     /  \       / \      / \    / \
  1   3   5   7   9   11     0   2    4   6  8   10
```

读法：第一棵的内部节点（如 0、4、2、6、10）在第二棵里落到叶子位置，第二棵的内部节点（如 3、7、1、5、9）在第一棵里是叶子——**没有 rank 在两棵都是内部节点**（blog 逐字）。叠加后每 rank 至多在一棵树承担内部节点的 $3D$ 发送、另一棵只承担叶子的 $D$，这正是命题 1 互补性计数的结构来源。mirror（偶数 rank）与 shift（奇数 rank）两种构造都保证这条互补性，差异只在第二棵的根位置。

## 4. ring/tree 交叉临界 M*：实测带宽差才有交叉

tree 延迟赢、ring 实测带宽赢，交叉点 $M^{\ast}$ 怎么解？关键：理想 α-β 模型里两者带宽项相同，**解不出交叉**；真实交叉来自实测有效带宽差。

**理想模型无交叉**（recap training 4.1 §2.2 结论，不重推）。理想 α-β 模型里 ring 与 tree 带宽项都是 $2(n-1)/n \cdot M\beta$（同一个 $\beta$），差异只在延迟项。此时 $T_{\mathrm{tree}} - T_{\mathrm{ring}} = 2(\lceil \log_2 n \rceil - (n-1))\alpha \lt 0$ 对所有 $M$ 成立——tree 延迟项恒小、带宽项恒等，tree 永远不输，**不存在 ring 反超的有限 $M^{\ast}$**。若强行令两式相等解 $M$，带宽项相消、方程退化为「延迟项相等」，与 $M$ 无关，解不出临界。这就是「禁从纯 α-β 凭空解出 $M^{\ast}$」的含义。

**真实交叉来自实测 $\beta_{\mathrm{eff}}$ 差**。实测中 tree 的有效带宽低于 ring（ $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$）：tree 的归约上行在多孩子节点处有 incast（多对一同时涌入）、流量不如 ring 邻居链路平滑，且跨 L3 IB switch 时与路由交互降带宽（blog 逐字「bandwidth degrades a bit when we cross L3 switches」）。NCCL 因此「automatically switches back to rings when that pattern results in greater bandwidth」（blog 逐字）。把两个有效带宽分开记，ring/tree 时间模型各为：

$$
T_{\mathrm{ring}} = 2(n-1)\alpha + 2 \thinspace \frac{n-1}{n} M \beta_{\mathrm{ring,eff}}, \qquad T_{\mathrm{tree}} = 2\lceil \log_2 n \rceil \alpha + 2 \thinspace \frac{n-1}{n} M \beta_{\mathrm{tree,eff}}
$$

**命题 2**（ring/tree 交叉临界）。设 $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$，令 $T_{\mathrm{ring}} = T_{\mathrm{tree}}$ 解得临界（ $n$ 为 2 的幂时树深恰 $\log_2 n$）：

$$
M^{\ast} = \frac{n(n-1-\log_2 n)\thinspace \alpha}{(n-1)(\beta_{\mathrm{tree,eff}} - \beta_{\mathrm{ring,eff}})} \qquad (2)
$$

$M \lt M^{\ast}$ 时 tree 赢（延迟项主导）、 $M \gt M^{\ast}$ 时 ring 赢（实测带宽项主导）。

**证明**（路线：令两式相等 → 延迟项差 = 带宽项差 → 解 $M$，三步）。其一，令 $T_{\mathrm{ring}} = T_{\mathrm{tree}}$，带宽项系数同 $2(n-1)/n$，移项得 $2(n-1)\alpha - 2\log_2 n \cdot \alpha = 2(n-1)/n \cdot M (\beta_{\mathrm{tree,eff}} - \beta_{\mathrm{ring,eff}})$。其二，左边提 $2\alpha$ 得 $2(n-1-\log_2 n)\alpha$，右边记 $\Delta\beta = \beta_{\mathrm{tree,eff}} - \beta_{\mathrm{ring,eff}} \gt 0$。其三，解出 $M^{\ast} = n(n-1-\log_2 n)\alpha / ((n-1)\Delta\beta)$。∎

**口径注**（硬规则）：其一， $\Delta\beta \gt 0$ 是交叉存在的**前提**——理想模型 $\Delta\beta = 0$ 时分母为零、 $M^{\ast}$ 无定义（无交叉），所以 $M^{\ast}$ 必须用实测 $\beta_{\mathrm{eff}}$ 差，不能从纯理想 α-β 解出。其二，一般 $n$（非 2 的幂）以 $\lceil \log_2 n \rceil$ 替 $\log_2 n$（双树深仍 $O(\log n)$），本章数值例 $n=8$ 两者同值。其三， $M^{\ast}$ 是 ring/tree **算法间**交叉，与 8.1 §5 的 $M_c$（ring **内部**延迟项/带宽项临界）不同——两个临界勿混（8.1 §5 口径注回链）。

全数字代入（ $n=8$、 $\alpha=2$ μs、 $\beta_{\mathrm{ring,eff}} = 1/370.8$ GB/s、 $\beta_{\mathrm{tree,eff}} = 1/130$ GB/s 教学假设）： $\Delta\beta = 7.69 \times 10^{-12} - 2.70 \times 10^{-12} = 4.995 \times 10^{-12}$ s/B； $M^{\ast} = 8 \times (7-3) \times 2 \times 10^{-6} / (7 \times 4.995 \times 10^{-12}) = 1.83 \times 10^{6}$ B ≈ **1.83 MB**。即小消息（低于 1.83 MB）tree 赢延迟、大消息（高于 1.83 MB）ring 赢实测带宽。代入验证： $M = M^{\ast}$ 时 $T_{\mathrm{ring}} = T_{\mathrm{tree}} \approx 36.6$ μs，两式逐数闭合。这个 1.83 MB 与 8.1 的 $M_c = 5.93$ MB 同量级但不同物—— $M_c$ 是 ring 内部两项之比， $M^{\ast}$ 是 ring/tree 实测带宽差驱动的算法切换点。

> **坑**：从纯理想 α-β 模型解 $M^{\ast}$。触发条件：把 ring/tree 带宽项都用同一个 $\beta$、令两式相等解 $M$；现象：带宽项相消、方程与 $M$ 无关，或分母 $\Delta\beta = 0$ 除零，得出「无交叉」或伪临界；规避：交叉必须引入实测 $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$（nccl-tests 实测或 NCCL tuning 表），理想模型只给「tree 延迟恒赢、带宽恒等」的结论（training 4.1 已交付）。

NCCL 把这套账落进 cost model：tree 的算法带宽取 bus 带宽之半（`generalBandwidths = busBw * 0.5`，src/tuning/tree.cc L44，对应 all-reduce 的 $2M$ 流量），延迟项实化 `2 * ((ppn-1)*intraLat + log2i(nNodes)*interLat)`（tree.cc L58，**log2 树深延迟项的代码形态**），再按消息尺寸查 `treeCorrectionFactor[proto][logSize]` 修正（L81）；最终 `ncclTuningGetTime`（tuning_general.cc L58）算 `time = lat*latCount + nBytes/(1000*bw)`——正是 α-β 模型的代码形态（8.2 §5 回链）。这些 per-algo cost model 的模块全量走读归 8.4。

## 5. binomial / k-ary 变体与 NVLink-aware 分层 tree

双二叉树之外还有哪些树形变体、各权衡什么？节点内 + 机间怎么分层？

**binomial tree**：root 有 $\log_2 n$ 个孩子、各发部分和 $M$，reduce 时 root 收 $M \log_2 n$（ $n=8$ 即 $3M$，8.1 §3 教学计数口径 recap）。binomial tree 延迟 $O(\log n)$，但 root 负载随 $\log n$ 涨、且每 rank 流量不最优——它是 reduce/broadcast 的常用结构，all-reduce 则用双树补带宽。

**k-ary tree**：每个内部节点 $k$ 个孩子，树深 $\lceil \log_k n \rceil$，延迟项 $2\lceil \log_k n \rceil \alpha$ 随 $k$ 增大而降；但归约上行时内部节点从 $k$ 个孩子各收 $D$（incast $k$ 路）、带宽项随 $k$ 增大而升。**权衡**： $k$ 大 → 延迟低、带宽差； $k$ 小 → 延迟高、带宽好。双二叉树是 $k=2$ 的带宽最优构造（命题 1）； $k \gt 2$ 用延迟换带宽，适合更小消息或更高延迟链路。NCCL 的 btree 模式本质是 $k=2$ 的位模式实现（trees.cc ncclGetBtree）。

**NVLink-aware 分层 tree**：节点内 tree + 机间 tree 分层。tree.cc 的延迟项 `2*((ppn-1)*intraLat + log2i(nNodes)*interLat)`（L58）正是分层结构：节点内每节点 ppn 个 rank 走 intraLat（链内段），机间 $n_{\mathrm{node}}$ 个节点代表走 $\log_2(n_{\mathrm{node}})$ 层 interLat。算术依据沿 8.2 §7：节点内 NVLink 快（370.8 GB/s）、机间 NIC 慢（50 GB/s，9×），分层把慢链路的使用压到节点代表之间。分层 tree 的 fabric 工程（rail 对齐、跨机树排列、SHARP/CollNet 网内归约）归 10.2，NVLS_TREE 多节点 NVLS 变体归 8.4。

## 6. NCCL 算法选择：ring / tree / NVLS 按消息 + 拓扑

NCCL 怎么在 ring/tree/NVLS 之间选？选择逻辑归 8.4 模块走读，本章只立「按消息尺寸 + 拓扑选最小时间」的口径与本章账的接口。

NCCL 对每个 (算法, 协议) 组合用 cost model 算时间，取最小者（`ncclTuningSelectBestTuning`，src/tuning/tuning.cc L153，版本锚 2.32.7-1）；时间即 `ncclTuningGetTime` 的 α-β 形态（§4）。候选算法含 Ring / Tree / NVLS / NVLS_TREE / CollNet / PAT（NCCL_ALGO 枚举，plugin/nccl_tuner.h L28-34；docs NCCL_ALGO 节，[NVIDIA, NCCL docs](../references.md)，[C]）。选择规律（本章账的定性接口）：**小消息**延迟主导 → tree（ $O(\log n)$ 延迟）或 NVLS（约 2 跳，7.1 §6）；**大消息**带宽主导 → ring（实测带宽最优）或 NVLS（每 rank 发送 $M$，域内）；**NVLink 域内**有 NVLS 时优先 NVLS（网内归约 bypass），跨机则 ring/tree 分层。环境变量 NCCL_ALGO / NCCL_PROTO 可强制指定（cost_model.cc L365-379 解析）。training 4.1 §2.2「选择逻辑归 cluster 维度」按 gpu 维度边界裁定重判到 gpu 8.4 / 10.2——算法选择的代码机制归 8.4，fabric 工程归 10.2。本章交付的 $M^{\ast}$（命题 2）正是 ring/tree 切换的定量依据，cost model 的 per-algo 表与 kernel dispatch 全量走读归 8.4。

> **提点**：理想模型与实测模型各管一段——理想 α-β 给「tree 延迟恒赢、带宽恒等」的结构结论（谁也不会错），实测 $\beta_{\mathrm{eff}}$ 差才给「大消息 ring 反超」的交叉点（工程切换的依据）。把这两段分开，就不会犯「从纯理想模型凭空解出 $M^{\ast}$」的错，也不会把 blog 的端到端 180× 实测改善误当纯延迟项比值。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| tree 延迟项 $2\lceil \log_2 n \rceil \alpha$ vs ring $2(n-1)\alpha$（recap training 4.1 练习 3 步数）； $n=8$ 12 vs 28 μs、 $n=1024$ 40 vs 4092 μs（约 102×）；blog Summit 24576 实测 180×（端到端口径，非纯延迟比） | 式 (1) | §2、练习 1 |
| 单棵二叉树内部节点发 $3M$、root 热点带宽不最优；双互补树「no rank is a node on both trees」、各处理 $M/2$，每 rank $\approx 2M$ 带宽最优 + $O(\log n)$ 延迟 | 命题 1 | §3、练习 3 |
| 双树学术出处 Sanders 2009 + NCCL blog；ncclGetDtree mirror（偶）/shift（奇）；内部节点 $3D$ + 叶子 $D = 2M$ | trees.cc L90/L32 | §3、8.4 |
| 理想 α-β 带宽项相同无交叉（recap training 4.1）；真实交叉来自 $\beta_{\mathrm{tree,eff}} \gt \beta_{\mathrm{ring,eff}}$； $M^{\ast} = n(n-1-\log_2 n)\alpha/((n-1)\Delta\beta)$， $n=8$ → 1.83 MB | 式 (2)，命题 2 | §4、练习 2/4 |
| $M^{\ast}$（算法间，实测 $\beta_{\mathrm{eff}}$ 差）≠ $M_c$（ring 内部两项临界）； $\Delta\beta=0$ 除零无交叉 | §4 口径注 + 坑 | §4、8.1 §5 |
| binomial root 收 $M\log_2 n$；k-ary 延迟 $2\lceil \log_k n \rceil\alpha$ 换带宽；分层 tree 延迟项 $2((ppn-1)\mathrm{intraLat} + \log_2(n_{\mathrm{node}})\mathrm{interLat})$ | tree.cc L58 | §5、8.1 §3 |
| NCCL 取最小 timeUs 选 (算法,协议)；小消息 tree/NVLS、大消息 ring/NVLS、域内 NVLS 优先；training 4.1「选择归 cluster」重判 gpu 8.4/10.2 | tuning.cc L153 | §6、8.4 |

## 练习

1.（计算 · [10]）tree 延迟代入（式 (1)， $\alpha=2$ μs 教学假设）：(a) $n=8$ tree 与 ring 延迟项各多少 μs；(b) $n=1024$ 各多少 μs、比值几倍；(c) $n=8$、 $M=1$ GB 时 tree 带宽项多少 ms（ $\beta=1/370.8$ GB/s，理想口径）。校验：(a) 12 / 28 μs；(b) 40 / 4092 μs、约 102×；(c) 4.72 ms（带宽项与 ring 同，理想模型），与 §2 逐数一致。
2.（计算 · [20]）ring/tree 交叉 $M^{\ast}$（式 (2)， $n=8$、 $\alpha=2$ μs、 $\beta_{\mathrm{ring,eff}}=1/370.8$、 $\beta_{\mathrm{tree,eff}}=1/130$ GB/s 教学假设）：(a) $\Delta\beta$ 多少 s/B；(b) $M^{\ast}$ 多少 MB；(c) $M=0.5$ MB 与 $M=10$ MB 各谁赢？先预测 (c) 两侧各落哪个算法再算。校验：(a) $4.995 \times 10^{-12}$ s/B；(b) ≈1.83 MB；(c) 0.5 MB tree 赢、10 MB ring 赢，与 §4 逐数一致（容差 1%）。
3.（推导 · [30]）证明命题 1（双树带宽最优）：(a) 算单棵二叉树内部节点（2 孩子）归约上行 + 广播下行的发送量 $3D$、叶子 $D$；(b) 用互补性（每 rank 至多在一棵树是内部节点）叠加得 $3D + D = 2M$；(c) 说明与 ring $2(n-1)/n \cdot M$ 同 leading term、差 $2M/n$；(d) 说明为什么单树带宽不最优而双树最优。校验：(b) $D=M/2$ 时 $4D=2M$；(c) $n=8$、 $M=1$ GB 双树 2 GB vs ring 1.75 GB、差 0.25 GB，与 §3 逐数一致。
4.（推导 · [30]）证明命题 2（式 (2) 交叉临界）并辨析理想模型：(a) 令 $T_{\mathrm{ring}} = T_{\mathrm{tree}}$（各用实测 $\beta_{\mathrm{eff}}$），移项解出 $M^{\ast}$；(b) 验证 $M=M^{\ast}$ 时两式相等（ $n=8$ 参数下 ≈36.6 μs）；(c) 说明理想模型 $\Delta\beta=0$ 时为什么解不出交叉；(d) 辨析 $M^{\ast}$ 与 8.1 $M_c$ 的区别。校验：(a) $n(n-1-\log_2 n)\alpha/((n-1)\Delta\beta)$；(b) ≈36.6 μs；(c) 带宽项相消、分母除零；(d) $M^{\ast}$ 算法间实测带宽差、 $M_c$ ring 内部两项临界，与 §4 口径注一致。

## 参考

- [NVIDIA, NCCL 2.4 blog](../references.md)：Massively Scale Deep Learning Training with NCCL 2.4（key: nccl-tree-blog），[C] 开放技术博客、非闭源，2026-11-25 逐字核——双互补树「half or less ranks... nodes... no rank is a node on both trees」+「as optimal as rings in terms of data sent/received」+「automatically switches back to rings when that pattern results in greater bandwidth」+ Summit 24576 GPU「up to 180x improvement」+ 跨 L3 switch 带宽降 + 2D hierarchical ring 逐字；ref [6]=Sanders 2009（§2/§3/§4）
- [Sanders et al., 2009](../references.md)：Two-tree algorithms for full bandwidth broadcast, reduction and scan（key: sanders-two-tree），[B] 老经典无 arXiv，Parallel Computing 35(12):581-594，2009-12，DOI 10.1016/j.parco.2009.09.001，Crossref 2026-11-25 核实——double binary tree 学术出处（前驱版 EuroPVM/MPI 2007 LNCS 4757）；命题 1 按本书骨架独立重推，经典只作出处（§3）
- [Rabenseifner, 2004](../references.md)：Optimization of Collective Reduction Operations（key: rabenseifner），[B] 老经典无 arXiv，Springer LNCS 3036 ICCS 2004 pp 1-9，Crossref 2026-11-25 再核——recursive halving/doubling 步数对照（§2，recap training 4.1 练习 3）
- [NVIDIA, NCCL](../references.md)：NVIDIA/nccl repo（key: nccl），教学参照 + 官方实现，版本锚 2.32.7-1（commit 351f33fd087e，master 2026-11-24）fresh tarball 核实——src/graph/trees.cc ncclGetBtree L32 / ncclGetDtree L90（mirror 偶 / shift 奇，注释 L69-88 双树图）；src/tuning/tree.cc generalBandwidths=busBw×0.5 L44 / 延迟项 2×((ppn-1)intraLat + log2i(nNodes)interLat) L58 / treeCorrectionFactor L81；src/tuning/tuning_general.cc ncclTuningGetTime L58；src/tuning/tuning.cc ncclTuningSelectBestTuning L153；plugin/nccl_tuner.h NCCL_ALGO 枚举 L28-34（§3/§4/§5/§6，模块全量走读归 8.4）
- [NVIDIA, NCCL docs](../references.md)：NCCL User Guide（key: nccl-docs），[C] 官方规格披露·一手权威，2.32.3（滞后 repo 2.32.7 注）——NCCL_ALGO（Ring/Tree/NVLS/NVLSTree/PAT）/ NCCL_PROTO 环境变量口径（§6）
- [NVIDIA, nccl-tests](../references.md)：NVIDIA/nccl-tests repo（key: nccl-tests），教学参照 + 官方实现——all-reduce 带宽-延迟基准，实测 $\beta_{\mathrm{eff}}$ 差的测量工具（§4，真实 $M^{\ast}$ 以实测为准）
- training 4.1 / 8.1 / 8.2 / 7.1 / 7.2（纯文字指引，不挂链接）：ring 消息量式 (2) + α-β 式 (3) + recursive halving/doubling 练习 3 + 「大消息带宽项相同树形赢延迟」结论（4.1 §2.2，recap 不重推）、regime 临界 $M_c$（8.1 §5）、ring 带宽最优性下界与万卡延迟（8.2 §3/§7）、NVLS 每 rank 发送 $M$（7.1 §6）、有效带宽 370.8 GB/s（7.2 §3）——recap 不重推（§1/§2/§4/§6）
