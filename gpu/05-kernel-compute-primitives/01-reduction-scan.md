# 5.1 Reduction 与 Scan（Reduction and Scan）

> 1677 万个 fp32 数求和，HBM 只读一遍的下界是 20.02 μs——reduction tree 每元素恰读一次，达到 $\Omega(n_{\mathrm{el}})$ 下界；warp shuffle 把块内 shared 流量从 4084 B 压到 64 B（63.8 倍），bank conflict 随之归零；Blelloch scan 用 2 倍 work 换 $O(\log n_{\mathrm{el}})$ depth，up-sweep/down-sweep 各 $n_{\mathrm{el}}-1$ 次加法。本章交付 reduction 带宽最优性下界（命题 1）与 work/depth 账（命题 2）、warp shuffle 免 shared 的流量与冲突账（命题 3）、block/grid 归约三方案与确定性、Blelloch scan work $O(n_{\mathrm{el}})$/depth $O(\log n_{\mathrm{el}})$ 证明（命题 4）与正确性不变式（命题 5）、scan 应用（compaction/radix sort/varlen 前缀）与片上消息量账 | 前置：3.1（thread/warp/block/grid 层级、cooperative groups 共驻留、Volta 独立 PC 坑）、3.2（fence/atomic/last-block 模式、atomic 非确定性命题 2、fp32 非结合）、4.1（coalescing）、4.2（bank conflict 判定）、2.4（带宽库存与 roofline 入口）、2.2（延迟表）；下游 5.3（softmax/norm 的 reduction 映射）、6.3（FA warp 内 quad 归约）、6.4（split-KV combine）、8.1/8.2（多 GPU 归约消息量，另一本账） | 锚点：Blelloch 1990（[Blelloch, 1990](../references.md)，CMU-CS-90-190，[B] 老经典无 arXiv）、CUDA Programming Guide 13.5（[NVIDIA, CUDA Programming Guide](../references.md)，[C] 官方规格披露·一手权威）、PTX ISA 9.5（[NVIDIA, PTX ISA](../references.md)，[C]）、cuda-samples reduction/shfl_scan/scan/threadFenceReduction（[NVIDIA, cuda-samples](../references.md)，教学参照 + 官方实现）、CUB/cccl（[NVIDIA, CUB](../references.md)，教学参照 + 官方实现）、Best Practices 13.5（[NVIDIA, Best Practices](../references.md)，[C]）——官方规格 2026-11-25 一手核实

## 1. 20.02 μs 的下界：三个问题与本章口径

1677 万个 fp32 数求和，最快能多快？答案不是由加法器数量决定的，而是由搬运决定的：每个数至少要从 HBM 读一次， $16777216 \times 4 = 67.11$ MB，除以 3352 GB/s 得 20.02 μs——任何算法都绕不过这个数。本章回答三个问题：reduction 的带宽下界是什么形状、tree 结构凭什么达到它（work 与 depth 各付多少）？warp 内归约怎么做到完全不碰 shared、块级和 grid 级又各走哪条路？scan 比 reduction 难在哪——Blelloch 怎么用两趟 sweep 同时拿 $O(n_{\mathrm{el}})$ work 与 $O(\log n_{\mathrm{el}})$ depth？

**本章口径**（章脊锁定）：主口径 **H100 SXM，Compute Capability 9.0，fp32 元素**（ $\mathrm{bytes} = 4$）；running example 为 $n_{\mathrm{el}} = 2^{24} = 16777216$ 个元素的求和与扫描（规模沿 3.1 的 1677 万元素向量，该章记 $n_{\mathrm{vec}}$，本章归约/扫描口径记 $n_{\mathrm{el}}$）， $\mathrm{BW} = 3352$ GB/s（2.4 recap，datasheet 口径）。block 配置 256 threads（ $w_b = 8$ warps），grid-stride 每线程 64 元素， $n_{\mathrm{blk}} = 1024$ blocks；§5 的 8 元素走读是教学最小例，换口径原因：展示 up/down-sweep 的逐层数组状态。片上 KB = 1024 B，带宽 GB/s 按 $10^{9}$ B/s（2.1/2.4 口径）。

符号声明（spec §4.3）：**本章 $n_{\mathrm{el}}$ = 归约/扫描元素数**（ $n$ 专归集合通信 rank 数，不被元素数占用，8.x 章首声明）； $n_{\mathrm{blk}}$ = grid 的 block 数； $w_b$ = warps/block（1.2 口径）； $W$ = work（运算次数）、 $D$ = depth（最长依赖链步数）——本章局部符号； $\mathrm{bytes}$ = 每元素字节（fp32 取 4，bytes 不占字母， $B$ 专归 batch）； $\ell$ = warp 内 lane 编号（0..31）； $x_i$ = 输入元素、 $y_i$ = exclusive scan 输出、 $p_i$ = compaction 谓词； $d$ = sweep 层号（§5 局部）； $u_{\mathrm{fp32}} = 2^{-24}$ = fp32 单位舍入（§4 局部，别于 2.4 利用率 $u$）； $K$ = GEMM 归约维、 $MN$ = GEMM 输出元素数（仅 §1 关系段局部借用，GEMM 问题维口径见 5.2 章首声明）。B/KB/MB 是单位符号豁免。

与 GEMM 的关系先钉一句：GEMM 的 $K$ 维本质是归约——每个输出元素是一条长 $K$ 的内积，wgmma 沿 k 方向的累加就是硬件实现的归约（1.3 recap）；差别在 GEMM 的归约是「每输出元素私有」的（ $MN$ 条独立内积，并行性来自输出维），本章归约是「全局共享」的（并行性来自树结构）。5.2 的 CUTLASS 走读会回到前者，本章的 work/depth 与片上流量账是它的原语侧基础。

**边界声明**：本章代码 scope 为机制片段级（教学 kernel），CUB/CCCL 的 BlockReduce/BlockScan 模块走读不展开（教学参照指针）；多 GPU 归约的 rank 间消息量账归 8.1/8.2（ $n$ = rank 数、 $M$ = 消息字节，与本章片上流量是两本账）；softmax/norm 的 reduction 应用映射归 5.3；FA 的 warp 内 quad 归约与 split-KV combine 归 6.3/6.4。**消息量口径**：本章「消息量」指单 GPU kernel 内层级间的数据搬运量（HBM/shared/register），不是 rank 间通信消息——后者见 8.x。

## 2. reduction 下界：Ω(n_el) reads 与 tree 达界

先回答一个更基本的问题：求和为什么是 memory-bound 而不是 compute-bound？ $n_{\mathrm{el}}$ 个元素只做 $n_{\mathrm{el}}-1$ 次加法（加法计 1 FLOP，约 1 FLOP/元素——别于 GEMM 乘加计 2 的口径，4.3），却要从 HBM 搬 $\mathrm{bytes}$ 字节/元素——强度 $I \approx 1/\mathrm{bytes} = 0.25$ FLOP/byte。归约是 fp32 CUDA-core 加法，适用的 ridge 是 FP32 峰值对带宽之比 $66.9/3.352 \approx 20$（1.2 §2 / 5.3 §1 口径），离 fp32 ridge ≈ 20 差约 80 倍——2.4 recap 的 295 是 BF16 Tensor Core ridge，不适用于 fp32 执行单元（对 BF16 TC ridge 的上界比较是 1180 倍，非本工作负载口径）。所以优化目标不是省加法，是**省搬运**：每个元素从 HBM 读几次、中间结果落在哪一层。

**命题 1**（读下界与 tree 达界）。任何正确求和算法的 HBM 读流量至少 $n_{\mathrm{el}} \cdot \mathrm{bytes}$（ $\Omega(n_{\mathrm{el}})$）；grid-stride 装载 + 层级 tree 归约的 kernel 每元素恰读一次、部分和驻留片上，HBM 总流量 $(n_{\mathrm{el}} + n_{\mathrm{blk}}) \cdot \mathrm{bytes} + O(1)$，渐近达到下界。**证明**（路线：反证下界 + 构造达界，两步）。下界：若某元素 $x_j$ 未被读取，则把 $x_j$ 改成 $x_j + \delta$（ $\delta \neq 0$）后算法输出不变，而真实和改变 $\delta$——矛盾；故每个元素至少读一次，读流量 $\ge n_{\mathrm{el}} \cdot \mathrm{bytes}$。达界：每线程 grid-stride 读 64 个元素在 register 内串行累加（每元素恰一次 HBM 读），warp/block 部分和经 shuffle 与 shared 归约（§3/§4，片上），每 block 只写 1 个 partial 到 HBM（ $n_{\mathrm{blk}}$ 个），末趟归约写 1 个结果——总流量 $(n_{\mathrm{el}} + n_{\mathrm{blk}}) \cdot \mathrm{bytes} + O(1)$，对 $n_{\mathrm{el}}$ 的比值趋于 1。∎

全数字代入（本章口径）：读下界 $67108864$ B = 67.11 MB，带宽口径时间 $67.11 \times 10^{6}/(3352 \times 10^{9}) = 20.02$ μs；partial 写回 $1024 \times 4 = 4096$ B = 4 KB，占读流量 0.0061%——**partial 是低阶项，tree 归约的 HBM 账就是「读一遍」**。对照 naive 口径：若每线程把部分和反复写回 HBM 再读（无片上归约），流量按中间结果层数翻倍，下界之上白付倍数。

work 与 depth 是另一本账（并行复杂度，不直接是时间）。所谓 work，是指算法执行的运算总次数；所谓 depth，是指依赖图最长链的步数（每步内无依赖的运算可并行）。

**命题 2**（二叉归约的 work/depth）。 $n_{\mathrm{el}}$ 个元素经二叉结合归约：任何算法的 work 下界 $n_{\mathrm{el}}-1$ 次结合运算；平衡 tree 达到 $W = n_{\mathrm{el}}-1$、 $D = \lceil \log_2 n_{\mathrm{el}} \rceil$；串行链 work 同为 $n_{\mathrm{el}}-1$、depth $n_{\mathrm{el}}-1$。**证明**（计数，三步）。其一，每次二元结合把部分和个数减 1，从 $n_{\mathrm{el}}$ 个到 1 个至少 $n_{\mathrm{el}}-1$ 次——work 下界与结构无关；平衡 tree 每内节点恰一次结合、内节点数 $n_{\mathrm{el}}-1$，达界。其二，tree 第 $d$ 层有 $n_{\mathrm{el}}/2^{d+1}$ 次互不依赖的结合，层数 $\lceil \log_2 n_{\mathrm{el}} \rceil$ 即 depth；下界侧，一步结合至多把输出的依赖扇入翻倍， $D$ 步后至多覆盖 $2^{D}$ 个输入，故 $D \ge \lceil \log_2 n_{\mathrm{el}} \rceil$。其三，串行链每步依赖前一步结果，depth = 运算数 = $n_{\mathrm{el}}-1$。∎

代入 $n_{\mathrm{el}} = 2^{24}$： $W = 16777215$、 $D = 24$；串行 depth 16777215——**work 相同、depth 差 70 万倍**，这就是 tree 的全部收益来源。本章 kernel 的实际 depth 是层级拼接：每线程 64 元素串行链（63 步）+ warp shuffle 5 步 + block 级 3 步 + grid 级 $\log_2 1024 = 10$ 步，合计约 81 步，仍远低于串行。

```
8 元素 reduction tree（W = 7 次加法，D = 3 层）：
  x0 x1 x2 x3 x4 x5 x6 x7
   \ /   \ /   \ /   \ /        层 0：4 次并行加法
    a     b     c     d
     \   /       \   /          层 1：2 次并行加法
      ab          cd
        \        /              层 2：1 次加法
         abcd  ← 结果
层级映射（本章配置）：
  register：每线程 64 元素串行链（grid-stride 装载，HBM 每元素读一次）
  warp    ：32 lane → shuffle 5 步（§3，免 shared）
  block   ：8 warp partial → shared 8 字 → warp0 shuffle 3 步（§4）
  grid    ：1024 partial → atomic / two-pass / cooperative（§4）
```

每线程先串行累加 64 个元素再进 tree，这不是随手优化，是 Brent 定理的调度： $p$ 个处理器上 work $W$、depth $D$ 的树可在约 $W/p + D$ 步内完成——把树的叶子层摊给线程串行做，处理器数用在真正并行的上层。cuda-samples reduction 例 reduce6 的注释逐字点名这条原理（[NVIDIA, cuda-samples](../references.md)，教学参照）：「This version adds multiple elements per thread sequentially. This reduces the overall cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n). (Brent's Theorem optimization)」——口径注：sample 注释的 n 指元素数（本章 $n_{\mathrm{el}}$），O(n)/O(log n) 即命题 2 的 $W$/ $D$。

代入 Brent 界： $p = 262144$ 线程， $W/p = 16777215/262144 \approx 64$， $D = 24$，理想调度约 88 步——与层级拼接的 81 步同量级（差值来自每线程串行段与树上层并非完美流水，界是理想调度口径）。这条界也解释「多开线程」为什么收益递减：若 $p$ 超过 $W/D$， $D$ 项主导，再加处理器压不动依赖链——depth 下界由结构钉死（命题 2），不由线程数决定。

> **提点**：reduction 的三本账各管一件事——HBM 流量账（命题 1，决定带宽口径时间）、work/depth 账（命题 2，决定并行结构与延迟）、片上流量账（§3，决定 shared 瓶颈与 bank conflict）。删掉这条分类，上面两个命题的推导链依然完整。

## 3. warp shuffle reduction：免 shared 的寄存器交换

warp 内 32 个 lane 各持一个部分和，怎么不碰 shared 就归约成一个？CUDA 的 warp shuffle 原语直接在寄存器之间交换数据。PG 逐字（[NVIDIA, CUDA Programming Guide](../references.md) Warp Shuffle Functions 节，[C]）：「Warp shuffle functions exchange a value between non-exited threads within a warp without the use of shared memory.」——**免 shared 是官方语义，不是实现巧合**。

两个原语的语义逐字（同节）：`__shfl_down_sync(mask, var, delta, width)`「calculates a source lane ID by adding delta to the caller's lane ID」，且「the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will effectively remain unchanged」——源 lane 越界时**不环绕**，高 delta 个 lane 拿回自己的值；`__shfl_xor_sync(mask, var, laneMask, width)` 按 lane 编号异或取源，「This mode implements a butterfly addressing pattern, which is used in tree reduction and broadcast」。down 形适合「结果聚到 lane 0」的归约，xor 形（butterfly）适合「每个 lane 都要结果」的全归约——两者步数同为 $\log_2 32 = 5$。

```
__shfl_down_sync 5 步归约（8 lane 最小例，offset = 4,2,1 三步；下表为 shuffle 源值语义）：
  lane:    0    1    2    3    4    5    6    7
  初始:   x0   x1   x2   x3   x4   x5   x6   x7
  off=4:  x0+x4 x1+x5 x2+x6 x3+x7  x4   x5   x6   x7   ← lane 4-7 源越界，返回自身
  off=2:  x0+x4+x2+x6  ...  x2+x6  x3+x7  ...                ← 只有 lane 0-3 有效
  off=1:  Σx0..x7  ...                                            ← lane 0 持全和
  32 lane 全形：offset = 16,8,4,2,1，共 5 步，lane 0 持 warp 和
  注：表列每步 shuffle 返回的源值。代码 v += shfl 下高位 lane 实持翻倍旧值
  （off=4 后 lane4-7 = 2·x_i），仅 lane 0 的累加轨迹干净（x=[1..8] 时 6→16→36），
  结果只取 lane 0（坑注见下）。
```

**命题 3**（shuffle 的 shared 流量与 bank conflict）。warp shuffle 归约的 shared 流量为 0，bank conflict 数为 0；block 级「warp shuffle → shared partial → warp0 shuffle」方案（ $w_b = 8$）的 shared 流量为 $2 w_b \cdot \mathrm{bytes} = 64$ B，对比 256 线程朴素 shared tree 的 4084 B，省 63.8 倍。**证明**（计数）。warp 级：shuffle 走寄存器交换网络，不产生 shared 访问；bank conflict 是 shared 访问的串行化现象（4.2 判定 recap），shared 流量为 0 则冲突数为 0。block 级 shuffle 方案：8 个 warp 的 lane 0 各写 1 个 partial（ $8 \times 4 = 32$ B），warp0 读回这 8 个字（32 B）再 shuffle 3 步——合计 64 B。朴素 shared tree：初始 256 线程各写 1 字（1024 B），随后 $s = 128, 64, \ldots, 1$ 每层 $s$ 个活跃线程各读 2 字写 1 字（ $3s \times 4$ B），tree 部分 $3 \times (128+64+\cdots+1) \times 4 = 3060$ B，总计 4084 B。比值 $4084/64 = 63.8$。∎

代入吞吐口径：shuffle 指令吞吐 32 results/cycle/SM（BP Table 5，CC 9.0，[NVIDIA, Best Practices](../references.md)，[C]）——一条 warp shuffle 每周期可发射一条（SM 级），5 步 warp 归约的发射账约 5 cycles；朴素 shared tree 8 层各带 load/add/store 与一次 `__syncthreads()`，barrier 数 8 对 1（shuffle 方案只在 partial 交接处同步一次）。shared 流量、conflict、barrier 三笔同向省。

> **坑**：假设 shuffle 源 lane 环绕。触发条件：把 `__shfl_down_sync` 当循环移位用，期望 lane 31 加 offset 后绕回 lane 0；现象：PG 逐字「will not wrap around」——越界 lane 拿回自身值，tree 的高位 lane 一直在加自己的旧值，结果静默偏大；规避：down 形归约只取 lane 0 的结果（其余 lane 的值按语义是部分和，不是全和）；要全 lane 拿全和用 xor butterfly。另一条：目标 lane 必须参与——「If the target thread is inactive, the retrieved value is undefined」（PG 逐字），mask 与分支须保证源 lane 活跃（3.1 Volta 独立 PC 坑的 shuffle 侧实例）。

两形的差别一句话钉死：down 形结束后只有 lane 0 持全和，其余 lane 持各自子树的部分和（「upper delta lanes unchanged」的语义兑现）；xor butterfly 结束后**每个 lane 都持全和**——第 $d$ 步 lane $\ell$ 与 lane $\ell \oplus 2^{d}$ 交换相加，5 步后每个 lane 的依赖扇入覆盖全部 32 lane。只要写一个结果用 down（reduction），后续计算要每 lane 都拿全和用 xor（broadcast 式全归约）。`width` 参数再把 warp 切成幂 2 子段、各子段按从 0 起的逻辑 lane 编号独立运作（PG 口径）——FA2 softmax 的 4-thread quad 行归约就是 4 lane 子组内的 butterfly（指令级走读归 6.3）。寄存器侧一笔：shuffle 不占 shared 容量，但交换网络按 warp 指令发射，操作数与结果都在寄存器——它把 shared 带宽压力转成发射槽压力，本章配置下 5 步 shuffle 的发射账远低于省下的 8 层 shared 访问（吞吐口径见上段）。

硬件归约指令一笔：PTX `redux.sync`（PTX 7.0，sm_80+）把 warp 归约做成单条指令，逐字「redux.sync performs a reduction operation .op of the 32 bit source register src across all non-exited threads in the membermask」（[NVIDIA, PTX ISA](../references.md)，[C]）；但语法逐字限定 `.op = {.add, .min, .max} .type = {.u32, .s32}`——**加法只支持 32 位整数**，`.f32` 形仅 min/max（PTX 8.6 引入）。所以 fp32 求和仍走 shuffle tree，整数求和/最值才有单指令路径；cuda-samples reduction README 逐字把 `__reduce_add_sync` 与 `__shfl_down_sync` 并列为演示策略（[NVIDIA, cuda-samples](../references.md)）。shuffle 不隐含内存屏障（PG 逐字「These intrinsics do not imply a memory barrier.」）——它只交换寄存器值，跨 warp 可见性仍靠 barrier/fence（3.2 recap）。

教学 kernel（机制片段级，fp32；warp + block 两级）：

```cpp
// 教学片段，非优化实现：warp shuffle 归约 + block 归约（256 线程 = 8 warp）
__device__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)      // 5 步：16,8,4,2,1
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;                                    // lane 0 持 warp 和
}
__device__ float block_reduce_sum(float v) {
    __shared__ float partial[8];                 // w_b = 8 个 partial，32 B
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) partial[wid] = v;             // 8 次写，32 B
    __syncthreads();                             // 唯一一次 block barrier
    if (wid == 0) {                              // warp0 读回 8 个 partial
        v = (lane < 8) ? partial[lane] : 0.f;    // 8 次读，32 B
        v = warp_reduce_sum(v);                  // 3 步有效（off=4,2,1）
    }
    return v;                                    // block 和只在 warp0 lane0 有效
}
```

## 4. block 与 grid 归约：层级、确定性与三方案

block 的和怎么汇成 grid 的和？block 之间没有 shared、没有 block 级 barrier（3.1 recap：block 驻留单 SM，跨 SM 两者都失去物理载体），所以 grid 归约必须走 global 内存 + 显式同步。三条路：atomic 单趟、two-pass 双 kernel、cooperative groups 单 kernel grid sync；另有 3.2 已交付的 last-block 变体。先钉一笔所有方案共有的账：**确定性**。fp32 加法非结合（3.2 §5 recap： $10^{8}$ 量级 ulp 为 8，加数 1 可被整个吞掉；顺序累加 1 到 $2^{24} = 16777216$ 后停滞——本章 $n_{\mathrm{el}}$ 恰在这个临界规模），求和顺序不同则舍入不同、比特级结果不同。

**atomic 单趟**：每 block 把 partial `atomicAdd` 到同一 global 地址。 $n_{\mathrm{blk}} = 1024$ 次同址 RMW 在 L2 串行化（3.2 同址 atomic 口径 recap），且完成顺序官方未定义——**结果比特级不可复现**（3.2 命题 2 recap，不重证）。流量侧 partial 不经 HBM 数组（L2 内 RMW），但串行化把 1024 次更新排成一条队。适用：只要数值容差、不要比特复现的场景（梯度累加类）；要确定性则排除。

**two-pass 双 kernel**：pass 1 每 block 写 partial 到 HBM 数组（1024 字，4 KB），pass 2 单 block 归约该数组（256 线程各读 4 个，串行 + shuffle，depth 约 11 步）。pass 2 按固定索引顺序归约，**结果确定**；代价是额外一次 kernel launch（~μs 量级，3.3 recap）与 partial 的写+读 8 KB（占读流量 0.012%）。pass 2 复用 §3 的 warp/block kernel 原样即可——1024 元素恰是 4 元素/线程的 grid-stride 加三层树。两 kernel 不能重叠（pass 2 依赖 pass 1 全部 partial），launch 与同步开销是固定支付：小 $n_{\mathrm{el}}$ 时占比显著，大规模下被 20 μs 级的读账摊薄。cuda-samples reduction 例的主结构即此（block partial → 末趟归约）。

**cooperative groups 单 kernel**：`this_grid().sync()` 把两阶段并进一个 kernel（3.1 §6 recap），前提是 grid 共驻留——本章配置核验：1024 blocks 对 $132 \times 32 = 4224$ block slots、8192 warps 对 $132 \times 64 = 8448$ warp slots，每 SM 约 8 blocks × 8 warps = 64 warps 恰满 warp slots，register/shared 宽松，`cudaLaunchCooperativeKernel` 可发射。grid sync 后由 block 0 归约 partial（或各 block 冗余归约换取广播）。确定、单 kernel，代价是共驻留约束限制 grid 规模。

**last-block 变体**（3.2 §3 已交付，recap 不重推）：每 block 写 partial 后 `__threadfence()` + `atomicInc` 取票，末票 block 归约全部 partial——单 kernel、无共驻留要求；计数虽经 atomic，末 block 按固定索引序读 partial，结果仍确定。官方教学例 cuda-samples `cpp/2_Concepts_and_Techniques/threadFenceReduction`（[NVIDIA, cuda-samples](../references.md)，教学参照）。fence 只管 ordering、可见性须配对的原语边界见 3.2 命题 1。

| 方案 | kernel 数 | 额外 HBM 流量 | 确定性 | 约束 |
| --- | --- | --- | --- | --- |
| atomic 单趟 | 1 | 1024 次 L2 RMW（无 partial 数组） | 否（3.2 命题 2） | 同址串行化 |
| two-pass | 2 | 写+读 8 KB | 是（固定索引序） | launch 开销 ~μs |
| cooperative | 1 | 写+读 8 KB（grid sync 前写） | 是 | grid 共驻留 |
| last-block | 1 | 写+读 8 KB + 计数 | 是（末 block 固定序） | fence+atomic 配对 |

确定性之外还有一笔**精度账**：tree 归约不仅快，还更准。一阶浮点模型 $\mathrm{fl}(a+b) = (a+b)(1+\delta)$、 $|\delta| \le u_{\mathrm{fp32}} = 2^{-24}$（教学口径）下，每个输入的贡献经过几次加法就累积几份舍入：串行链中 $x_0$ 经过 $n_{\mathrm{el}}-1$ 次加法，误差系数约 $(n_{\mathrm{el}}-1) u_{\mathrm{fp32}}$；平衡 tree 中每个输入只经过 $D = \log_2 n_{\mathrm{el}}$ 次，系数约 $\log_2 n_{\mathrm{el}} \cdot u_{\mathrm{fp32}}$。代入 $n_{\mathrm{el}} = 2^{24}$：串行系数约 1（一阶界已无意义），tree 系数 $24 \times 2^{-24} \approx 1.43 \times 10^{-6}$——**pairwise 求和的误差界随规模对数增长，串行线性增长**（经典数值分析结论的一阶计数版）。所以「确定性归约」与「高精度归约」在 tree 结构上是同一件事的两面：固定顺序给比特复现，浅依赖链给小舍入。

具体例（3.2 §5 口径）：求和 $2^{24}+2$ 个 1。串行累加到 $2^{24}$ 后停滞——该量级 ulp 变 2，加 1 舍入回原值，最后两个 1 被整个吞掉，结果 16777216；pairwise tree 先把 1 两两合成 2、4、……、 $2^{24}$，末两个 1 合成 2 再并入，中间值全为可表示整数，结果 16777218 = 真值。**同一份输入，两种结构差 2 个整单位**——不是「小误差」，是加数级别的丢失（3.2 口径）。这也给「要不要为确定性付 two-pass 的 launch 成本」一个量化答案：要比特复现、或累加规模越过 $2^{24}$ 量级时，tree + 固定序不是可选项，是正确性需求。

## 5. Blelloch scan：2 倍 work 换 O(log n_el) depth

reduction 输出 1 个数，scan 输出 $n_{\mathrm{el}}$ 个数——难在哪？所谓扫描（scan，前缀和），是指对序列 $x_0, \ldots, x_{n_{\mathrm{el}}-1}$ 输出每个位置的前缀累积：**exclusive scan** $y_i = \sum_{j=0}^{i-1} x_j$（ $y_0 = 0$，不含自身），**inclusive scan** $z_i = \sum_{j=0}^{i} x_j$（含自身）。正例： $x = [1,2,3,4]$ 的 exclusive 为 $[0,1,3,6]$、inclusive 为 $[1,3,6,10]$。边界反例：compaction 用 exclusive 扫描谓词得写地址，若误用 inclusive，首个被选元素写到位置 1、位置 0 留空——整体错位一项（§6）。

naive 做法对每个前缀各跑一次 reduction：work $O(n_{\mathrm{el}}^{2})$；串行一趟累加：work $O(n_{\mathrm{el}})$ 但 depth $O(n_{\mathrm{el}})$。能不能同时拿 $O(n_{\mathrm{el}})$ work 与 $O(\log n_{\mathrm{el}})$ depth？Blelloch 的答案是两趟 sweep（[Blelloch, 1990](../references.md)，CMU-CS-90-190，[B] 老经典无 arXiv）：**up-sweep 沿 tree 向上归约出子块和，down-sweep 沿 tree 向下把前缀分发回每个位置**。推导路线分三步：先写两趟的逐层操作，再按层计数 work 与 depth（命题 4），最后用不变式证正确性（命题 5）；关键一步是「down-sweep 的 swap 把父节点的前缀送给左子、把左子的子块和加到父前缀上送给右子」——一次交换同时完成前缀的下传与右子前缀的构造。

```
8 元素 Blelloch 走读（x = [1,2,3,4,5,6,7,8]，数组下标 0..7）：
up-sweep（右节点 += 左节点，层 d 的步长 2^{d+1}）：
  d=0: [1, 3, 3, 7, 5,11, 7,15]   x[1]+=x[0], x[3]+=x[2], x[5]+=x[4], x[7]+=x[6]
  d=1: [1, 3, 3,10, 5,11, 7,26]   x[3]+=x[1], x[7]+=x[5]
  d=2: [1, 3, 3,10, 5,11, 7,36]   x[7]+=x[3]  ← 根 = 总和 36
置根为 0: [1, 3, 3,10, 5,11, 7, 0]
down-sweep（左节点 ↔ 右节点：左=右旧值，右=左旧值+右旧值）：
  d=2: [1, 3, 3, 0, 5,11, 7,10]   x[3]=0, x[7]=10
  d=1: [1, 0, 3, 3, 5,10, 7,21]   x[1]=0,x[3]=3; x[5]=10,x[7]=21
  d=0: [0, 1, 3, 6,10,15,21,28]   逐对交换 ← exclusive scan 结果
核对：y = [0,1,3,6,10,15,21,28]，总和 36 = 末项 28 + x[7]=8 ✓
```

**命题 4**（Blelloch work/depth）。 $n_{\mathrm{el}} = 2^{L}$ 时，up-sweep 加法 $n_{\mathrm{el}}-1$ 次，down-sweep 加法 $n_{\mathrm{el}}-1$ 次、交换 $n_{\mathrm{el}}-1$ 次；总 work $W = 2(n_{\mathrm{el}}-1)$ 次加法 + $(n_{\mathrm{el}}-1)$ 次交换 = $O(n_{\mathrm{el}})$，depth $D = 2L = 2\log_2 n_{\mathrm{el}} = O(\log n_{\mathrm{el}})$。**证明**（按层计数 + 等比求和）。up-sweep 第 $d$ 层（ $d = 0, \ldots, L-1$）对 $n_{\mathrm{el}}/2^{d+1}$ 个不相交数对各做一次加法，层内无依赖、一步完成；加法总数 $\sum_{d=0}^{L-1} n_{\mathrm{el}}/2^{d+1} = n_{\mathrm{el}}(1 - 2^{-L}) = n_{\mathrm{el}}-1$。down-sweep 第 $d$ 层（ $d = L-1, \ldots, 0$）同样 $n_{\mathrm{el}}/2^{d+1}$ 个数对，每对一次交换（左=右旧值）加一次加法（右=左旧值+右旧值），加法与交换各 $n_{\mathrm{el}}-1$。两趟层数各 $L$，每层一步，depth $2L$。∎

代入 $n_{\mathrm{el}} = 2^{24}$：加法 $2 \times 16777215 = 33554430$ 次、交换 16777215 次，depth 48——**对串行 scan（work 16777215、depth 16777215）付 2 倍 work、depth 从 1677 万压到 48**。work 翻倍是结构代价：up-sweep 的子块和 down-sweep 要再用一遍，没有免费的前缀分发。

拿到 $O(\log n_{\mathrm{el}})$ depth 的还有另一条结构：offset-doubling scan——第 $d$ 步每个位置加上 $2^{d}$ 个位置之前的前缀， $\log_2 n_{\mathrm{el}}$ 步完成，但每步做 $n_{\mathrm{el}}$ 次加法（越界位置加 0），work $O(n_{\mathrm{el}} \log n_{\mathrm{el}})$。三种结构并排：

| 结构 | work（加法） | depth | $2^{24}$ 代入 work |
| --- | --- | --- | --- |
| 串行一趟 | $n_{\mathrm{el}}-1$ | $n_{\mathrm{el}}-1$ | 16777215 |
| offset-doubling | 约 $n_{\mathrm{el}} \log_2 n_{\mathrm{el}}$ | $\log_2 n_{\mathrm{el}}$ | 约 4.03 亿 |
| Blelloch 两趟 | $2(n_{\mathrm{el}}-1)$ | $2\log_2 n_{\mathrm{el}}$ | 33554430 |

Blelloch 的取舍清楚：对 offset-doubling 付 2 倍 depth（两趟），work 从约 4.03 亿降到 3355 万次加法（12 倍）。指令发射与寄存器/shared 流量是实付成本时，work-efficient 结构胜出——库实现的默认取向即 Blelloch 族（CUB BlockScan 的 raking 布局属此族，教学参照指针）。

把 swap 读深一层：down-sweep 第 $d$ 层开始时，每对的右节点持该对块左边界的 exclusive 前缀（归纳假设），左节点持 up-sweep 留下的左半块和。swap 第一赋值把右节点的前缀送给左节点——左半块的边界前缀恰等于整块边界前缀；第二赋值把左节点旧值（左半块和）加到前缀上送给右节点——右半块边界前缀 = 整块边界前缀 + 左半块和。一次交换、两个前缀各就各位，这就是 down-sweep「沿树分发前缀」的全部机制。

**命题 5**（正确性不变式）。up-sweep 第 $d$ 层完成后，对所有 $i \equiv 0 \pmod{2^{d+1}}$， $x[i+2^{d+1}-1]$ 等于原数组第 $i$ 起 $2^{d+1}$ 个元素之和；down-sweep 第 $d$ 层开始前， $x[i+2^{d+1}-1]$ 等于原数组前 $i$ 个元素之和（块边界的 exclusive 前缀）；down-sweep 结束后 $x[i] = y_i$。**证明**（要点）：up-sweep 对 $d$ 归纳——基例 $d=0$ 是相邻两元素相加；归纳步把两个相邻 $2^{d}$ 块和加成 $2^{d+1}$ 块和。down-sweep 对 $d$ 递减归纳——根置 0 是空前缀；每层的交换把父节点持有的块边界前缀送给左子（左子边界 = 父边界），右子边界 = 左子旧值（左子块和）+ 父边界，恰为右子块起点的前缀。终止于 $d=0$ 时每个位置持有自己的 exclusive 前缀。完整证明（含归纳假设的逐层代入与 8 元素例核对）留作练习 3。∎

带宽侧推论（命题 1 同法）：scan 每元素读一次、写一次，HBM 流量 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$ 达 $\Omega(n_{\mathrm{el}})$ 下界——代入得 134.22 MB、40.04 μs，恰为 reduction 的 2 倍（多写一遍输出）。exclusive 与 inclusive 互转只付逐元素一次加法： $z_i = y_i + x_i$，work $n_{\mathrm{el}}$、depth 1。

教学 kernel（机制片段级；block 内 shared 版，幂 2 长度，grid 扩展见 §6 指针）：

```cpp
// 教学片段，非优化实现：block 级 Blelloch exclusive scan（n = 1024，幂 2）
__global__ void blelloch_scan(const float* x, float* y) {
    __shared__ float tmp[1024];
    int tid = threadIdx.x;
    tmp[tid] = x[blockIdx.x * 1024 + tid];
    __syncthreads();
    for (int d = 1; d < 1024; d <<= 1) {          // up-sweep：d = 2^层号
        int i = tid * (d << 1);
        if (i + (d << 1) - 1 < 1024) tmp[i + (d << 1) - 1] += tmp[i + d - 1];
        __syncthreads();
    }
    if (tid == 0) tmp[1023] = 0.f;                // 置根为 identity
    __syncthreads();
    for (int d = 512; d >= 1; d >>= 1) {          // down-sweep
        int i = tid * (d << 1);
        if (i + (d << 1) - 1 < 1024) {
            float t = tmp[i + d - 1];
            tmp[i + d - 1] = tmp[i + (d << 1) - 1];
            tmp[i + (d << 1) - 1] += t;
        }
        __syncthreads();
    }
    y[blockIdx.x * 1024 + tid] = tmp[tid];
}
```

> **坑**：非幂 2 长度直接跑 Blelloch。触发条件： $n_{\mathrm{el}}$ 不是 2 的幂时照搬步长 $2^{d+1}$ 的索引；现象：越界读写或漏加元素，结果静默错误；规避：补 identity 元（加法补 0）到幂 2 再扫，输出截回原长——补零不改前缀值。另一条：shared 版深层步长访问带 bank conflict（4.2 判定），教学版未消；免冲突路径是 padding 或 CUB BlockScan 的 raking 布局（教学参照指针），warp 内 scan 可用 `__shfl_up_sync` 做（cuda-samples shfl_scan 例，README 逐字「demonstrates how to use the shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block」）。

## 6. scan 应用与消息量账

scan 的前缀地址语义撑起了三类结构。

**stream compaction（流压缩）**：谓词 $p_i \in \lbrace 0,1 \rbrace$ 标记保留元素，对 $p$ 做 exclusive scan 得每个被选元素的写地址，再 scatter。8 元素例： $p = [1,0,1,1,0,0,1,0]$，scan 得 $[0,1,1,2,3,3,3,4]$，被选元素 $x_0, x_2, x_3, x_6$ 分别写到位置 0,1,2,3，输出长度 4 = scan 末项 + 末位谓词。work/depth 继承 scan： $O(n_{\mathrm{el}})$/ $O(\log n_{\mathrm{el}})$。HBM 账（教学口径，谓词扫描融合在片上； $n_{\mathrm{sel}}$ = 被选元素数）：读 $n_{\mathrm{el}} \cdot \mathrm{bytes}$ + 写 $n_{\mathrm{sel}} \cdot \mathrm{bytes}$；代入选择率 25%：67.11 + 16.78 = 83.89 MB、25.03 μs——比「先 scan 写回再 scatter 读回」的三趟结构省一趟中间数组往返。谓词若按 1 bit/元素打包，读侧再降到 $n_{\mathrm{el}}/8$ 字节，但 unpack 要付位操作指令（lop3/prmt 族，6.2 指针）——流量省换计算增，值不值由 roofline 判（2.4 口径）。

**radix sort**：每趟按一个 digit 桶计数（histogram，reduction 侧），对桶计数做 exclusive scan 得各桶的写偏移，再按偏移 scatter——scan 把「每桶多少个」翻译成「每桶从哪写」。代入 8-bit digit 一趟：256 个桶，histogram 得计数 $c_0, \ldots, c_{255}$（ $\sum_j c_j = n_{\mathrm{el}}$），exclusive scan 得偏移 $o_j = \sum_{k \lt j} c_k$，元素 $i$ 的写地址 = $o_{\mathrm{digit}(i)}$ + 桶内序号。这里的 scan 只有 256 个元素——单 warp 即可完成，成本可略；真正的流量在 histogram 与 scatter 两趟 $n_{\mathrm{el}}$ 元素读写。CUB `DeviceRadixSort` 即此结构（[NVIDIA, CUB](../references.md)，教学参照，NVIDIA/cub 已并入 NVIDIA/cccl）；多趟按 digit 重复、趟数 = 位宽 ÷ digit 位宽（算法级口径，完整推导不展开），本章只立 scan 在其中的地址语义。

**attention 的 varlen 前缀**：变长序列批打包成一个 tensor 时，`cu_seqlens[i]` = 第 $i$ 条序列的起始偏移 = 序列长度数组的 exclusive scan——FlashAttention varlen 接口的前缀结构（模型侧数学归 models 4.1，指令级走读归 6.3）；causal mask 的行边界同理是前缀结构，tile 级跳过判定用块边界前缀（6.3）。softmax 的行 max/行 sum 是 reduction 不是 scan（5.3 映射），两原语勿混。

消息量账汇总（本章口径 $n_{\mathrm{el}} = 2^{24}$、fp32；「消息」= 层级间搬运，口径见 §1）：

| 账目 | reduction（two-pass） | scan（Blelloch） |
| --- | --- | --- |
| HBM 读 | $n_{\mathrm{el}} \cdot \mathrm{bytes}$ = 67.11 MB | 67.11 MB |
| HBM 写 | $n_{\mathrm{blk}} \cdot \mathrm{bytes}$ = 4 KB（+结果） | $n_{\mathrm{el}} \cdot \mathrm{bytes}$ = 67.11 MB |
| 带宽口径时间 | 20.02 μs | 40.04 μs |
| work（加法） | $n_{\mathrm{el}}-1$ = 16777215 | $2(n_{\mathrm{el}}-1)$ = 33554430 |
| depth | $\log_2 n_{\mathrm{el}}$ = 24（tree 侧） | $2\log_2 n_{\mathrm{el}}$ = 48 |
| block 内 shared 流量 | 64 B（shuffle 方案，命题 3） | $O(n_{\mathrm{el}} \cdot \mathrm{bytes})$（每层读写 tmp） |

两原语的 HBM 账都达 $\Omega(n_{\mathrm{el}})$ 下界（命题 1 及其推论），差别在写侧：reduction 输出 1 个数、scan 输出 $n_{\mathrm{el}}$ 个数。**多 GPU 侧是另一本账**：rank 间归约的消息量与带宽最优性下界归 8.1/8.2（ $n$ = rank 数、 $M$ = 消息字节），NCCL 的 ring/tree 算法是本章 reduction tree 的通信侧对应物（8.3/8.4）——本章的「每元素读一次」在通信侧对应「每 rank 的消息收发次数下界」，结构同构、量纲不同。

## 7. 边界：本章不做什么

下界与两原语交付完，剩下的归谁？本章交付读下界与达界（命题 1 + §2）、work/depth（命题 2）、shuffle 流量与冲突账（命题 3 + §3）、grid 三方案与确定性/精度账（§4）、Blelloch work/depth 与正确性（命题 4/5 + §5）、应用与消息量账（§6）；以下各有专章。softmax/norm 的 reduction 映射与 two-pass/online 访存账归 5.3；FA 的 warp 内 quad 归约（FA2 softmax 行归约）与 split-KV combine 的 reduction kernel 走读归 6.3/6.4；算子融合的带宽节省账归 5.4；多 GPU 归约消息量与下界证明归 8.1/8.2；CUB/CCCL 模块走读不作（教学参照指针）；实测 drill-down 归 11.2。代码 scope：机制片段级（§3/§5 教学 kernel），不做模块走读。

本章交付清单（下游 recap 接口）：命题 1/2（§2）、命题 3 + warp/block 教学 kernel（§3）、grid 三方案表与确定性口径（§4）、命题 4/5 + scan 教学 kernel（§5）、消息量账表（§6）。5.3 recap 命题 3 与 §3 kernel 作 softmax/norm 的 warp/block 归约原型；6.4 recap §4 grid 归约结构作 split-KV combine 的机制侧；8.2 recap 命题 1 的下界方法作 rank 侧带宽最优性证明的原型。

## 总结

| 论断 | 公式 / 数值 | 回指 |
| --- | --- | --- |
| 任何求和算法 HBM 读流量 $\ge n_{\mathrm{el}} \cdot \mathrm{bytes}$（ $\Omega(n_{\mathrm{el}})$）；grid-stride + tree 每元素恰读一次达界 | 命题 1 | §2、练习 1/2 |
| 二叉归约 work 下界 $n_{\mathrm{el}}-1$；平衡 tree $W = n_{\mathrm{el}}-1$、 $D = \lceil \log_2 n_{\mathrm{el}} \rceil$；串行 depth $n_{\mathrm{el}}-1$ | 命题 2 | §2、练习 1 |
| 本章配置： $2^{24}$ fp32 读 67.11 MB、20.02 μs；partial 4 KB 占 0.0061%；实际 depth 约 81（63+5+3+10） | §2 代入 | §2、练习 1/2 |
| 每线程串行累加 + 上层 tree = Brent 调度（ $W/p + D$）；sample reduce6 注释逐字点名 | §2 Brent 段 | §2、练习 4 |
| warp shuffle 免 shared（PG 逐字）；down 不环绕、xor butterfly；5 步归约 lane0 持和 | §3 PG 逐字 | §3、练习 4 |
| shuffle 方案 block shared 流量 64 B 对朴素 tree 4084 B（63.8×）；conflict 0；barrier 1 对 8 | 命题 3 | §3、练习 2 |
| redux.sync 加法仅 u32/s32（f32 仅 min/max，PTX 8.6）；fp32 求和走 shuffle tree | §3 PTX 逐字 | §3 |
| grid 归约：atomic 非确定（3.2 命题 2）/ two-pass 确定 / cooperative 共驻留 / last-block fence+票（3.2 recap） | §4 表 | §4、练习 2 |
| pairwise 误差系数 $\log_2 n_{\mathrm{el}} \cdot u_{\mathrm{fp32}}$ 对串行 $(n_{\mathrm{el}}-1) u_{\mathrm{fp32}}$； $2^{24}$ 给 $1.43 \times 10^{-6}$ 对约 1 | §4 精度账 | §4 |
| Blelloch：up/down 各 $n_{\mathrm{el}}-1$ 加法， $W = 2(n_{\mathrm{el}}-1)$、 $D = 2\log_2 n_{\mathrm{el}}$； $2^{24}$ 给 33554430 加法/depth 48 | 命题 4 | §5、练习 1/3 |
| 正确性不变式：up 存子块和、down 传块边界前缀；exclusive/inclusive 互转逐元素加 $x_i$ | 命题 5 | §5、练习 3 |
| scan HBM 读+写 $2 n_{\mathrm{el}} \cdot \mathrm{bytes}$ = 134.22 MB、40.04 μs，达下界；compaction/radix/varlen 用前缀地址语义 | §5 推论 + §6 | §6、练习 2 |

## 练习

1.（计算 · [10]）下界与复杂度代入（本章口径 $n_{\mathrm{el}} = 2^{24}$、fp32、 $\mathrm{BW} = 3352$ GB/s）：(a) reduction 的 HBM 读下界字节与带宽口径时间；(b) 平衡 tree 的 work 与 depth，串行链的 depth；(c) warp shuffle 归约步数与 256 线程 block 的 shared partial 字数。校验：(a) 67.11 MB、20.02 μs（§2）；(b) 16777215、24、16777215（命题 2）；(c) 5、8（§3）。
2.（计算 · [20]）流量与方案账：(a) 朴素 shared tree 与 shuffle 方案的 block 级 shared 流量各多少字节、比值多少；(b) two-pass 的额外 HBM 流量与占读流量百分比；(c) scan 的 HBM 流量与带宽口径时间，对 reduction 的比值；(d) 先预测 atomic 单趟对 two-pass 的 HBM 流量方向（多/少/同）再按 §4 口径核对，并指出 atomic 方案用流量换掉了什么。校验：(a) 4084 B 对 64 B、63.8（命题 3）；(b) 8 KB、0.012%（§4）；(c) 134.22 MB、40.04 μs、2（§5）；(d) 少（L2 RMW 无 partial 数组）但非确定（§4 表）。
3.（推导 · [30]）补全命题 5 的证明（Blelloch 正确性）：(a) 写出 up-sweep 不变式并对 $d$ 归纳（基例 $d=0$、归纳步：两个 $2^{d}$ 块和相加）；(b) 写出 down-sweep 不变式并对 $d$ 递减归纳，逐句说明 swap 的两个赋值各把什么前缀送到哪个子节点；(c) 终止条件给出 $x[i] = y_i$，用 §5 的 8 元素走读逐层核对；(d) 推导 exclusive→inclusive 转换的 work 与 depth。校验：(c) 与 §5 数组状态逐位一致（末项 28、总和 36）；(d) $n_{\mathrm{el}}$ 次加法、depth 1（§5）。
4.（代码 · [30]）cuda-samples reduction 走读（公开 repo `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`，教学参照）：(a) 对照 reduce0/reduce1/reduce2 的注释逐字，说明三步各消除什么问题（modulo 与 warp 不完整活跃 → bank conflict → sequential addressing）；(b) reduce6 的「multiple elements per thread」对应本章哪条调度原理，为什么 work 仍 $O(n_{\mathrm{el}})$；(c) 找出 reduce7 的 `warpReduceSum`（`__shfl_down_sync`）循环，说明它对应 §3 的几步树、结果在哪个 lane（字面 `tile32.shfl_down` 循环在 reduce6，勿混）；(d) 先预测 reduce1 对 reduce2 的性能方向再对照注释验证。校验：(a) reduce2 注释「no divergence or bank conflicts」逐字（§3/§4 口径）；(b) Brent 定理（§2 逐字引文）；(c) 5 步、lane 0（命题 3）；(d) reduce2 更快（conflict 消除，4.2 判定）。

## 参考

- [Blelloch, 1990](../references.md)：Prefix Sums and Their Applications（key: blelloch-scan），CMU-CS-90-190，[B] 老经典无 arXiv，2026-11-25——up-sweep/down-sweep 两趟结构、work $O(n_{\mathrm{el}})$/depth $O(\log n_{\mathrm{el}})$（命题 4/5 的公式源头；报告页 cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html + Ble93.pdf）
- [NVIDIA, CUDA Programming Guide](../references.md)：CUDA C++ Programming Guide 13.5（key: cuda-programming-guide），[C] 官方规格披露·一手权威，2026-11-25——Warp Shuffle Functions 节（05-appendices/cpp-language-extensions.html）：「Warp shuffle functions exchange a value between non-exited threads within a warp without the use of shared memory.」逐字 + __shfl_down_sync 源 lane 加 delta、「will not wrap around the value of width and so the upper delta lanes will effectively remain unchanged」逐字 + __shfl_xor_sync「butterfly addressing pattern, which is used in tree reduction and broadcast」逐字 + 「If the target thread is inactive, the retrieved value is undefined」逐字 + 「These intrinsics do not imply a memory barrier.」逐字；cooperative groups grid.sync 共驻留（3.1 recap 源）；atomic relaxed 序与同址串行（3.2 recap 源）
- [NVIDIA, PTX ISA](../references.md)：NVIDIA PTX ISA 9.5（key: ptx-isa），[C] 官方规格披露·一手权威，2026-11-25——shfl.sync（warp shuffle 指令 backing）；redux.sync（PTX 7.0 sm_80+，「performs a reduction operation .op of the 32 bit source register src across all non-exited threads in the membermask」逐字 + .add/.min/.max 仅 .u32/.s32 + .f32 仅 min/max（PTX 8.6））；atom 语义（3.2 recap 源）
- [NVIDIA, cuda-samples](../references.md)：NVIDIA/cuda-samples repo（key: cuda-samples），教学参照 + 官方实现，2026-11-25——cpp/2_Concepts_and_Techniques/reduction（reduce0–reduce7 + cg_reduce/multi_warp_cg_reduce；reduce6 注释「(Brent's Theorem optimization)」逐字 + reduce2「no divergence or bank conflicts」逐字；README「using shared memory, __shfl_down_sync, __reduce_add_sync and cooperative_groups reduce」逐字）、shfl_scan（README「demonstrates how to use the shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block」逐字）、scan、threadFenceReduction（last-block 官方教学例）、reductionMultiBlockCG
- [NVIDIA, CUB](../references.md)：CUB cooperative primitives（key: cub），教学参照 + 官方实现，2026-11-24——NVIDIA/cub 已 archived 并入 NVIDIA/cccl；BlockReduce/BlockScan（raking 免冲突布局指针）/DeviceRadixSort（histogram + scan + scatter 结构指针），本章不作模块走读
- [NVIDIA, Best Practices](../references.md)：CUDA C++ Best Practices Guide 13.5（key: cuda-best-practices），[C] 官方规格披露·一手权威，2026-11-25——Table 5 CC 9.0 shfl 32 results/cycle/SM（§3 shuffle 吞吐口径）
