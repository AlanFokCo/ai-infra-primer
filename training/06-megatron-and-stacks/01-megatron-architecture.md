# 6.1 Megatron 的层栈并行原语（Megatron's Parallel Primitives for the Layer Stack）

> 5.1 把张量并行推成了公式：列切与行切的分法、f/g 恒等算子、每层两次 all-reduce 的位置。本章把这套数学逐项对到 NVIDIA/Megatron-LM 的实现上：`megatron/core` 模块树导览；f/g 的 autograd.Function 定义（`tensor_parallel/mappings.py` 的 `_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion`，每个只有 forward/backward 两行）与 5.1 公式逐行对照；`ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding`（`tensor_parallel/layers.py`）的构造与 forward 调用点；attention 与 MLP 的组合接线（`transformer/attention.py` / `transformer/mlp.py`）；并行下权重初始化的按分片 RNG 处理（`tensor_parallel/random.py`）；RMSNorm 与 RoPE 在序列并行区段的归属 | 前置：5.1（TP 推导，本章引用其结论不重复）、1.1（autograd 与自定义 Function）、4.1（集合通信语义）；5.2（SP 的通信恒等改写） | 锚点：Megatron-LM 论文（[Shoeybi et al., 2019](../references.md)，arXiv:1909.08053，[A]）+ NVIDIA/Megatron-LM repo（[A]，文件与类名核实于 2026-11-17，main 分支）、Megatron-SP（[Korthikanti et al., 2022](../references.md)，arXiv:2205.05198，[A]）

## 1. `megatron/core` 模块树导览

Megatron-Core 是 Megatron-LM 的可复用库层。与本章及前后章相关的模块树（目录清单核实于 2026-11-17）：

```
megatron/core/
├── parallel_state.py        # 进程组树：initialize_model_parallel（5.5 式 (5) 的实现；走读归 6.2）
├── tensor_parallel/         # 本章主场：单算子怎么切
│   ├── layers.py            # ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding
│   ├── mappings.py          # f/g 与 scatter/gather/SP 区段算子（autograd.Function）
│   ├── random.py            # 按分片的 RNG tracker（§7）
│   ├── cross_entropy.py     # 词表并行交叉熵（§5 一句）
│   └── utils.py             # divide / VocabUtility 等
├── transformer/             # 算子怎么接成层
│   ├── attention.py         # Attention / SelfAttention：linear_qkv + core_attention + linear_proj
│   ├── mlp.py               # MLP：linear_fc1 + activation + linear_fc2
│   ├── transformer_layer.py # TransformerLayer：norm 位置与残差接线
│   └── transformer_config.py# TransformerConfig（sequence_parallel 等开关）
├── pipeline_parallel/       # schedules.py：1F1B / interleaved（5.3 的实现；走读归 6.2）
├── optimizer/               # distributed optimizer（ZeRO-1/2 同族；走读归 6.2）
├── dist_checkpointing/      # 并行度无关 checkpoint（衔接 7.4）
├── models/                  # GPT 等模型定义
└── datasets/                # 数据管线（衔接 7.1）
```

三层分工：tensor_parallel/ 回答「单个矩阵乘怎么切」，transformer/ 回答「切好的算子怎么接成一个 Transformer 层」，parallel_state.py 回答「rank 怎么分组」（5.5 的 group 树，走读归 6.2）。5.1 的公式落在第一层，本章的接线在第二层。

## 2. f/g 的 autograd 实现：mappings.py

5.1 的推导结论（本章只登记不重复）：列切线性层前向不需要通信、反向要对各 rank 算出的输入梯度求和；行切前向要对各 rank 的部分积求和、反向直通。两个通信点被封装成一对互为镜像的算子：

$$
f: \quad \text{前向 } f(x) = x \thinspace ; \qquad \text{反向 } \overline{x} = \mathrm{AllReduce}(\overline{y}) \qquad (1)
$$

$$
g: \quad \text{前向 } g(x) = \mathrm{AllReduce}(x) \thinspace ; \qquad \text{反向 } \overline{x} = \overline{y} \qquad (2)
$$

**符号**： $x$、 $y$ 为前向输入输出，上划线记对应的上游梯度（ $\overline{y} = \partial L / \partial y$ ）；all-reduce 在 TP 组内做。这两个算子的 forward 与 backward 不对称——恒等映射的默认 backward 是直通、all-reduce 不是逐点可微算子——框架推不出来，必须手写 `torch.autograd.Function`（1.1 的扩展机制）。代码逐字对应（`megatron/core/tensor_parallel/mappings.py`，行级事实核实于 2026-11-17）：

```python
class _CopyToModelParallelRegion(torch.autograd.Function):      # 式 (1) 的 f
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_                                           # 前向恒等
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None            # 反向 all-reduce

class _ReduceFromModelParallelRegion(torch.autograd.Function):  # 式 (2) 的 g
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)                           # 前向 all-reduce
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None                                # 反向恒等
```

`_reduce` 是对 `torch.distributed.all_reduce` 的薄封装（group size 为 1 时短路直通）；对外 API 是 `copy_to_tensor_model_parallel_region` 与 `reduce_from_tensor_model_parallel_region` 两个函数包装。代码与 5.1 公式的逐行对照：

| 5.1 公式 | mappings.py 代码 | 语义 |
| --- | --- | --- |
| $f(x) = x$ | `forward: return input_` | 列切层的输入在组内本来就全量，直通 |
| $`\partial L / \partial x = \sum_{i=1}^{t} \overline{y}_i A_i^{\top}`$ | `backward: _reduce(grad_output, ...)` | 输入全量 ⇒ 各 rank 的输入梯度是部分和，须求和 |
| $g(x) = \sum_{i=1}^{t} x_i$ | `forward: _reduce(input_, ...)` | 行切的部分积求和 |
| $\partial L / \partial x_i = \partial L / \partial y$ | `backward: return grad_output` | 和对每个加项的梯度相同，直通 |

f 与 g 互为镜像的代数根源是 all-reduce（求和算子）的反向仍是求和—— $`\overline{x}_i = \overline{y}`$ 对 $y = \sum_i x_i$ 逐行成立（练习 1 把两步推导补全）。

同文件还有一族变体算子，覆盖切分/拼接与 SP（sequence parallelism）边界；下表为主要选编（类名与行为核实于 2026-11-17；`_ReduceScatterToTensorParallelRegion`、异步版 `_GatherFromSequenceParallelRegionAsync` 等与本章主线关系较远的变体从略）：

| autograd.Function | 前向 | 反向 | 用途 |
| --- | --- | --- | --- |
| `_ScatterToModelParallelRegion` | 沿最后一维切、留本 rank 片 | all-gather 拼回 | 行切层输入未分片时补切 |
| `_GatherFromModelParallelRegion` | all-gather 后沿最后一维拼接 | 切分留本 rank 片 | 列切层 `gather_output=True` |
| `_ScatterToSequenceParallelRegion` / `_GatherFromSequenceParallelRegion` | 沿第一维（序列维）切 / 拼 | 互逆 | SP 区段进出 |
| `_ReduceScatterToSequenceParallelRegion` / `_AllGatherFromTensorParallelRegion` | reduce-scatter 到序列维 / all-gather 回全量 | 互逆 | SP 下 g / f 的替代对 |
| `_AllToAll` | 按组重排分片 | 互逆 | EP dispatch/combine 的底层（5.4） |

SP 的改写规则（推导归 5.2）：TP 区段两端的 all-reduce 换成 all-gather + reduce-scatter 对，字节数不变；megatron-sp 论文记这对转换算子为 g / ḡ——g 前向 all-gather、反向 reduce-scatter，ḡ 前向 reduce-scatter、反向 all-gather（[Korthikanti et al., 2022](../references.md)，[A]，§4.2.2 及 Figure 5）。上表中间两行就是这对算子的实现。

一处容易被略读的实测注记值得登记：megatron-sp 论文明确报告「reduce-scatter + all-gather 合起来的执行时间比单次 all-reduce 慢」（其 §6.2：序列并行让单层前向从 7.7 ms 降到 7.2 ms，收益被这一项吃掉一部分）——字节数恒等不代表墙钟时间恒等，SP 的收益来自激活显存侧（norm 与 dropout 的激活从全量复制变成按 $t$ 分片：其逐算子账里 $10sbh$ 的复制项由此可除 $t$（其 §4.2.1 对该项的归属），账归 1.4 与 5.2），通信侧甚至是小幅倒退。这解释了 SP 为什么是一个显式开关（`TransformerConfig.sequence_parallel`）而不是默认行为：它是拿通信效率换激活显存的权衡，不是免费的改写。

## 3. `ColumnParallelLinear`：列切

数学（5.1）： $Y = XA$， $A \in \mathbb{R}^{h \times h'}$ 按列切成 $A = [A_1, \ldots, A_t]$，rank $i$ 持有 $A_i \in \mathbb{R}^{h \times h'/t}$：

$$
Y_i = X A_i \thinspace , \qquad Y = [Y_1, \ldots, Y_t] \qquad (3)
$$

前向无通信（输入 $X$ 全量、各 rank 独立算自己那块输出列），反向经 f 对 $\partial L / \partial X$ 做 all-reduce。逐元素算子（GELU 等）作用在分片 $Y_i$ 上不需通信——这是「列切 + 激活函数 + 行切」三段式省掉中间通信的原因（5.1 的论证）。

代码（`tensor_parallel/layers.py`，类 `ColumnParallelLinear`）：

- **构造**：本地权重形状 `[output_size_per_partition, input_size]`——`torch.nn.functional.linear` 计算的是 $XA^{\top}$，故分片维（输出维）存在权重第 0 维；`divide(output_size, world_size)` 执行整除；初始化走 §7 的 RNG 路径；`set_tensor_model_parallel_attributes` 在参数上标记分片维，供 optimizer 与 checkpoint 识别。
- **forward**（三个调用点按序）：
  1. f 的位置：`input_parallel = copy_to_tensor_model_parallel_region(input_)`。默认生产配置实际走**融合分支**：构造期 `allreduce_dgrad = (world_size > 1 and not sequence_parallel and not disable_grad_reduce)` 为真时跳过独立 f——同一次输入梯度 all-reduce 折进 GEMM autograd.Function 的 backward 里异步发出（`LinearWithGradAccumulationAndAsyncCommunication.backward` 的 `all_reduce(grad_input, async_op=True)`，源码注释说明依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1` 保证它在权重梯度计算前排入流）；`sequence_parallel=True` 时同样跳过——f 的 SP 变体（前向 all-gather 输入、反向 reduce-scatter 输入梯度）也折进同一个 Function 内（其 forward 里的 `dist_all_gather_func(all_gather_buffer, input, ...)`）。数学与式 (1)(3) 逐项相同，实现位置不同：参考形态是 mappings.py 的独立算子，生产形态是融合进 GEMM。
  2. `self._forward_impl(...)`：本地 GEMM（可选梯度累加融合 `main_grad` 直写等工程分支）。
  3. 可选 `gather_tensor_parallel_output`（`gather_output=True` 时沿最后一维 all-gather 拼回全量 $Y$）。Transformer 层内恒为 `False`——输出保持分片、直接喂给下一个行切层；只在需要全量结果的场合（如某些 logits 用法）打开。

## 4. `RowParallelLinear`：行切

数学： $Y = XB$， $B$ 按行切成 $B = [B_1^{\top}, \ldots, B_t^{\top}]^{\top}$，输入相应按列切 $X = [X_1, \ldots, X_t]$：

$$
Y = \sum_{i=1}^{t} X_i B_i \qquad (4)
$$

各 rank 本地算部分积 $X_iB_i$，求和即 g 的前向 all-reduce。

代码（同类文件，类 `RowParallelLinear`）：

- **构造**：`input_is_parallel` 标记输入是否已沿最后一维分好——Transformer 接线里恒为 `True`（上游列切层的输出天然是分片的），免一次 scatter；独立使用时 forward 先调 `scatter_to_tensor_model_parallel_region` 补切。
- **forward**：本地 GEMM 之后两分支——默认 `reduce_from_tensor_model_parallel_region(output_parallel)`（g，前向 all-reduce）；`sequence_parallel=True` 时换 `reduce_scatter_to_sequence_parallel_region(...)`（g 的 SP 变体 ḡ：求和并沿序列维散开，输出直接落进 $[s/t, b, h]$ 的序列分片布局）。
- **bias 的位置**：bias 不切分，且只在 g 之后加一次（forward 末尾的 `output = (output_ + self.bias)`——若先加再 all-reduce，bias 会被求和放大 $t$ 倍）；`skip_bias_add=True` 时 bias 不进层内、作为返回值交给调用方与后续逐元素算子融合。这与 5.1 公式里「行切层输出 = 部分和 + bias」的书写一一对应。

对照 §3 的列切层可以看出 Megatron 的接线美学：f 永远紧贴列切层输入、g 永远紧贴行切层输出，中间的分片区段（逐元素激活、attention 本体）零通信——5.1 的「每层两次 all-reduce」在代码层面就是每层两个 g 点（§6 的接线图）。

## 5. `VocabParallelEmbedding`：词表切分

词表 $V$ 均分成 $t$ 段，rank $i$ 持有行区间 $[v_i, v_i + V/t)$ 的 embedding 行（`VocabUtility.vocab_range_from_global_vocab_size` 给出本 rank 的起止索引）。token $x$ 的查表恰好只在一个 rank 上有定义：

$$
\mathrm{Emb}(x) = \sum_{i=1}^{t} \mathrm{Emb}_i(x - v_i) \cdot \mathbb{1}[v_i \le x \lt v_i + V/t] \qquad (5)
$$

**符号**： $`\mathrm{Emb}_i`$ 为本地分片查表； $\mathbb{1}[\cdot]$ 为指示函数。指示函数保证每 token 只被一个 rank 贡献、其余 rank 出 0，于是求和（all-reduce）精确等于查表——不是近似。梯度侧对称：只有拥有该 token 行的 rank 收到非零行梯度，分片所有权与梯度归属天然一致，embedding 的稀疏梯度模式（2.2 §1 的动机之一）在分片下不需要额外处理。

代码（同类文件，类 `VocabParallelEmbedding`，forward 逐行）：

1. `input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)`，`masked_input = input_ - self.vocab_start_index`、越界位置置 0——式 (5) 的指示函数与本地索引；
2. `F.embedding(masked_input, weight)` 本地查表；
3. `output_parallel[input_mask, :] = 0.0`——非己贡献清零；
4. `reduce_from_tensor_model_parallel_region(output_parallel)`——all-reduce 求和（g）；`reduce_scatter_embeddings=True`（SP 配置）时换 `reduce_scatter_to_sequence_parallel_region`，输出直接进序列分片布局。

两个工程注记。其一，查表有两条路径：`deterministic_mode` 下走 `weight[masked_input]` 张量索引（其反向为确定性的索引累加），源码注释说明默认的 `F.embedding` 反向本身非确定性，追求逐位复现的实验要显式打开这个开关。其二，`EmbeddingWithGradAccumulationFusion` 分支把行梯度直接累加进 `main_grad` 缓冲——与 §3 的梯度累加融合同族，省一次中间梯度物化。

配套一句：输出侧 logits 投影用同一个 `ColumnParallelLinear`（`gather_output=False`，logits 保持词表分片），交叉熵在分片 logits 上直接算——`tensor_parallel/cross_entropy.py` 的词表并行 CE 避免物化全量 $[b, s, V]$；CE 的数值稳定数学归 3.1，本章只登记入口。

## 6. attention / MLP 的组合接线

把 §3-§5 的原语接成一个 Transformer 层（`transformer/mlp.py` 与 `transformer/attention.py`，行级事实核实于 2026-11-17）。张量布局为 $[s, b, h]$：

```
        x [s, b, h]（TP 组内全量；SP 时整个残差流是 [s/t, b, h]）
 ┌───────┴────────┐
 │ input_layernorm│（逐 token，权重复制；§8）
 └───────┬────────┘
      f：恒等（反向 all-reduce）——SP 时此处是 all-gather（g 转换子）
 ┌───────┴────────┐
 │ linear_qkv     │ ColumnParallelLinear, gather_output=False
 │                │ 每 rank 持有若干 head 的 Q/K/V 行块
 └───────┬────────┘
      RoPE（逐 head 逐 position，本地；§8）
 ┌───────┴────────┐
 │ core_attention │ 每 rank 只算自己的 head 分片，零通信
 └───────┬────────┘
 ┌───────┴────────┐
 │ linear_proj    │ RowParallelLinear, input_is_parallel=True
 └───────┬────────┘
      g：all-reduce ◄── 每层第 1 次集合通信（SP 时为 reduce-scatter，ḡ）
 ┌───────┴────────┐
 │ pre_mlp_layernorm │
 └───────┬────────┘
      f
 ┌───────┴────────┐
 │ linear_fc1     │ ColumnParallelLinear, gather_output=False
 │ [h → h_ff]     │ GLU/SwiGLU 时 stride=2（每 rank 交错持 [gate, up] 对）
 └───────┬────────┘
      activation（逐元素，本地，零通信）
 ┌───────┴────────┐
 │ linear_fc2     │ RowParallelLinear, input_is_parallel=True
 │ [h_ff → h]     │
 └───────┬────────┘
      g ◄── 每层第 2 次集合通信
        y [s, b, h]
```

核实点：`MLP.__init__`（mlp.py）以 `gather_output=False` 构造 `linear_fc1`、以 `input_is_parallel=True` 构造 `linear_fc2`；GLU 族激活时 `ffn_hidden_size` 翻倍且 `stride=2`——源码注释说明这是为了让每个 TP rank 持有交错的 [gate, up] 对、保证跨 TP 度数 resharding 的正确性。`Attention.__init__`（attention.py）以 `input_is_parallel=True` 构造 `linear_proj`；`SelfAttention` 以 `gather_output=False` 构造 `linear_qkv`。两个 g 点即 5.1 「每层两次 all-reduce」的全部集合通信。

把 shape 沿接线图走一遍（ $t$ 为 TP 度数， $a$ = head 数， $d_h$ = head dim， $h_{ff}$ = FFN 中间维；张量布局 $[s, b, \cdot]$）。attention 支路： $x$ 为 $[s, b, h]$ 全量； `linear_qkv` 输出 $[s, b, (a + 2 a_{kv}) d_h / t]$（ $a_{kv}$ = KV 组数，列切后每 rank 只持自己 head 的行块）；reshape 成 head 维后 RoPE 与 core attention 都在 $[s, b, a/t, d_h]$ 上本地算； `linear_proj` 输入 $[s, b, a d_h / t]$（行切的右因子已按列分好）、输出是 $[s, b, h]$ 的部分和，g 点 all-reduce 成全量。MLP 支路同构： `linear_fc1` 输出 $[s, b, h_{ff} / t]$（GLU 时 gate/up 交错），逐元素激活本地算， `linear_fc2` 输出 $[s, b, h]$ 部分和再过 g。两个支路之间、以及层与层之间，残差流始终保持 $[s, b, h]$ 全量——通信只发生在「回到全量」的两个点上，这正是 5.1 位置论的代码形状。

把接线图写成可跑的教学实现（教学用，非优化实现；单进程模拟 $t$ 个 rank，通信用求和模拟；逐行与式 (3)(4) 对应。注意梯度方向在单进程模拟里由 autograd 自动正确——「求和」就是 g 的前向、其反向自动是广播，f 的恒等前向、其反向自动累加各分片对 $x$ 的梯度，这两个性质在多进程实现里才需要 §2 的手写 Function）：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
t, h, hff, s = 2, 8, 32, 5
x = torch.randn(s, h, dtype=torch.float64, requires_grad=True)
W1 = (torch.randn(h, hff, dtype=torch.float64) / h**0.5).requires_grad_(True)
W2 = (torch.randn(hff, h, dtype=torch.float64) * 0.1).requires_grad_(True)

# 参考：单机全量
ref = F.gelu(x @ W1) @ W2
ref.sum().backward()
gx_ref, g1_ref, g2_ref = x.grad.clone(), W1.grad.clone(), W2.grad.clone()

# TP 版：W1 列切、W2 行切
W1s = [w.detach().clone().requires_grad_(True) for w in W1.detach().chunk(t, dim=1)]
W2s = [w.detach().clone().requires_grad_(True) for w in W2.detach().chunk(t, dim=0)]
x2 = x.detach().clone().requires_grad_(True)
out = sum(F.gelu(x2 @ w1) @ w2 for w1, w2 in zip(W1s, W2s))   # 末位 sum 即 g
out.sum().backward()
assert torch.allclose(out, ref, atol=1e-12)
assert torch.allclose(x2.grad, gx_ref, atol=1e-12)            # f 的反向求和被 autograd 自动完成
assert torch.allclose(torch.cat([w.grad for w in W1s], 1), g1_ref, atol=1e-12)
assert torch.allclose(torch.cat([w.grad for w in W2s], 0), g2_ref, atol=1e-12)
```

与官方实现的差异清单：官方版每个分片是独立 rank 上的独立张量，通信是真集合通信且反向手写（§2）；教学版在同一进程内用 `sum` 模拟 g、靠 autograd 免费得到正确的梯度语义。对拍通过说明 5.1 的公式与 §2 的手写 backward 在同一件事上收敛——练习 2 把它推广到多进程。

GQA 分支（attention.py 构造期）：`num_query_groups >= t` 时 K/V 按组整除（每 rank `num_query_groups / t` 组 QKV 行块）；`num_query_groups < t` 时走复制分支——每 rank 持有 1 个 KV 组加 `num_heads / num_query_groups` 个 Q head，KV 权重在组内复制。KV head 数小于 TP 度数的复制代价（权重与激活不除 $t$）推导归 5.1，这里是它的实现形态。

## 7. 并行下权重初始化的一致性

分片初始化要同时满足三个不变量：(i) 同一分片的不同 DP 副本取值必须一致（否则 DP 的梯度平均语义被破坏，4.1）；(ii) 同层不同 TP 分片取值必须不同（否则列切的多个分片初始化成同一份、对称性不破）；(iii) 初始化不得污染默认 RNG 流（dropout 等的可复现性）。

实现（`tensor_parallel/random.py` 与 layers.py，行级事实核实于 2026-11-17）：`model_parallel_cuda_manual_seed` 为每个 rank 注册多条 RNG 流，seed 公式为

$$
s_{\mathrm{dp}} = s_0 \thinspace , \qquad s_{\mathrm{tp}} = s_0 + 2718 + i_{\mathrm{tp}} \thinspace , \qquad s_{\mathrm{ep}} = s_0 + 1024 + 100 \thinspace i_{\mathrm{ep}} + i_{\mathrm{etp}} \qquad (6)
$$

**符号**： $s_0$ 为用户 seed； $i_{\mathrm{tp}}$、 $i_{\mathrm{ep}}$、 $i_{\mathrm{etp}}$ 为 TP / EP / expert-TP 组内 rank；2718 与 1024 是让三条流错开的任意常数（源码注释原话 "2718 is just for fun"）。不变量的落点：(i) 公式不含 $i_{\mathrm{dp}}$——同一 TP 分片在所有 DP 副本上同 seed；(ii) 含 $i_{\mathrm{tp}}$——同层不同分片不同 seed；(iii) 由 `CudaRNGStatesTracker.fork()` 的 save/set/restore 语义保证：`layers.py` 的 `_initialize_affine_weight_gpu` 在 `get_cuda_rng_tracker().fork()` 上下文里调 `init_method(weight)`，初始化在 model-parallel 流上抽数、退出即恢复默认流。两条流的另一个用途是 dropout：TP 区段内张量各 rank 持不同分片、掩码本应各自独立（model-parallel 流），残差侧的全量张量在组内是复制的、掩码必须组内一致（default 流）——这是代码结构的直接读法。

GPU 路径之外还有一条 CPU 路径：`_initialize_affine_weight_cpu` 先在 CPU 上把全量 master weight 完整初始化（同一 $s_0$、默认流——所有 rank 造出同一份全量权重），再 `torch.split` 后按 `weight_list[rank::world_size]` 取本 rank 的切片。stride 的含义在这里可见：stride 1 时该表达式取连续块；GLU 的 stride 2 时（§6 的 [gate, up] 交错约定），`torch.split` 先把每 rank 份额切成两半、交错取样后每 rank 拿到的恰是一对对齐的 gate/up 块——切片规则与 §6 的 stride=2 存储约定互为表里。两条路径的差异值得记清：CPU 路径拼合出的全量权重与 $t$ 无关（先造全量再切），GPU 路径各分片独立抽数（每个分片的分布正确，但拼合矩阵随 $t$ 变）——要求「与并行度无关的初始化」时走 CPU 路径（`use_cpu_initialization` 配置开关）。

## 8. RMSNorm 与 RoPE 在 SP 区段的归属

SP 把一个 Transformer 层切成两类区段：TP 区段（线性层包围的中间部分，张量按 hidden/head 维分片）与 SP 区段（norm、dropout、残差，张量按序列维分片 $[s/t, b, h]$）——切分规则与通信恒等的推导归 5.2，本节登记边界上的实现事实。

**Norm 归 SP 区段**。RMSNorm / LayerNorm 是逐 token 算子：归一化沿 hidden 维计算，参数是逐 hidden 维的增益向量。SP 区段的张量序列维分片、hidden 维全量，norm 在本地分片上直接算，零通信；增益权重在 TP 组内复制持有——增益只有 $h$ 个元素，相对两侧线性层的 $h \times h_{ff}$ 级矩阵在参数量上低两个量级，复制几乎免费（2.2 §8 的参数账同口径）。其梯度因为「同一份复制权重收到组内多个分片的梯度」而需要组内归约——这一笔折进 DP 的梯度归约里统一处理（实现归 6.2 的 DDP 走读）。实现注记：Megatron 的 torch 原生 norm 路径（`ops/norm/torch_norm.py` 的 `WrappedTorchNorm`）对 `sequence_parallel` 配置直接 assert 拒绝（"sequence parallel not supported by torch LayerNorm"）——SP 配置走 Transformer Engine 的 TP-aware norm；RMSNorm 选项要求 torch ≥ 2.4（同文件断言）。

**RoPE 归 TP 区段，但不产生 TP 通信**。RoPE 是逐 (position, head) 的旋转：既不跨 head 归约、也不跨位置归约（公式归 models 2.5）。调用点（attention.py，行级事实）：`apply_rotary_pos_emb` 在 `linear_qkv`（列切投影）之后、`core_attention` 之前施加——每个 rank 只旋转自己持有的 head 分片；位置表按本地序列片的全局偏移索引，函数签名带 `cp_group` 参数处理 CP 下的位置偏移（5.2 的 CP 区段由此接入）。于是 §6 接线图里的 RoPE 节点标「本地」是准确的：它在分片数据上算，却一次集合通信都不发。

## 总结

| 主题 | 结论 | 出处 |
| --- | --- | --- |
| f/g 定义 | 式 (1)(2)：f 前向恒等/反向 all-reduce，g 前向 all-reduce/反向恒等；互为镜像的根源是求和算子的反向仍是求和 | mappings.py；5.1 |
| 列切 | `ColumnParallelLinear`：权重按输出维分片，forward 依次 f → 本地 GEMM → 可选 gather；生产配置把 f 折进 GEMM 的 backward（异步 all-reduce） | layers.py，式 (3) |
| 行切 | `RowParallelLinear`：`input_is_parallel=True` 免 scatter；输出端 g（SP 时换 reduce-scatter） | layers.py，式 (4) |
| 词表切分 | 区间掩码 + 本地查表 + 清零 + all-reduce，求和精确等于查表 | layers.py，式 (5) |
| 层接线 | qkv/proj、fc1/fc2 两对列切-行切；每层两个 g 点；GQA 在 KV 组数 < $t$ 时走复制分支 | mlp.py / attention.py，§6 图 |
| 初始化 | 三条 RNG 流（式 (6)）：DP 副本一致、TP 分片互异、默认流不被污染；CPU 路径与 $t$ 无关 | random.py / layers.py |
| SP 归属 | norm 在 SP 区段（逐 token、权重复制、零通信）；RoPE 在 TP 区段但逐 head 逐 position、零通信；torch 原生 norm 不支持 SP | torch_norm.py / attention.py |

## 练习

1.（推导）证明 all-reduce 作为前向算子时其反向仍是 all-reduce：设 $y = \sum_{i=1}^{t} x_i$，逐行写出 $\partial L / \partial x_i$；再对 f 推一遍反向（列切层 $Y_i = XA_i$、 $X$ 全量），说明两算子为何互为镜像。这条性质如果把 all-reduce 换成 reduce-scatter 还成立吗？（提示：reduce-scatter 的反向是什么算子——对照 mappings.py 里 `_ReduceScatterToSequenceParallelRegion` 的 backward。）

2.（代码）用 mappings.py 的两个公开 API（`copy_to_tensor_model_parallel_region` / `reduce_from_tensor_model_parallel_region`）写一个 $t = 2$ 的「列切 + GELU + 行切」MLP 教学实现（教学用，非优化实现），与单机参考实现对拍前向输出与输入/权重梯度；列出对拍容差与不一致时的排查顺序。

3.（分析）式 (6) 的三条流：若把 $s_{\mathrm{tp}}$ 也设成 $s_0$（TP 组内同 seed），列切层的多个分片初始化成什么样、对 5.1 的 MLP 分解有什么后果？若把 $s_{\mathrm{dp}}$ 设成 $s_0 + i_{\mathrm{dp}}$，DP 的梯度平均语义在哪一步被破坏（对照 4.1 式 (1) 的等价性条件）？

4.（分析）GQA 复制分支：head 数 64、KV 组数 8、 $t = 8$ 时，attention.py 的两条分支各给每个 rank 分到多少 Q/K/V 行块？KV 权重在 TP 组内的复制因子是多少，相对 MHA（64 KV 组）多出的每 rank KV 权重字节数怎么算（设 head dim 128、bf16）？

## 参考

- [Shoeybi et al., 2019](../references.md)：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，arXiv:1909.08053，[A]（key: megatron-lm）——f/g 算子与列切/行切的公式源头（论文 §3）；NVIDIA/Megatron-LM repo 走读对象：`megatron/core/tensor_parallel/mappings.py`（`_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion` / SP 与 all-to-all 变体）、`layers.py`（`ColumnParallelLinear` / `RowParallelLinear` / `VocabParallelEmbedding` / `_initialize_affine_weight_gpu` / `_initialize_affine_weight_cpu` / `LinearWithGradAccumulationAndAsyncCommunication`）、`random.py`（`model_parallel_cuda_manual_seed` 与式 (6)）、`transformer/mlp.py`、`transformer/attention.py`、`ops/norm/torch_norm.py`——路径与类名核实于 2026-11-17（main 分支）
- [Korthikanti et al., 2022](../references.md)：Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，[A]（key: megatron-sp）——SP 区段划分与 g/ḡ 转换算子对（§4.2.2、Figure 5）；「reduce-scatter + all-gather 比单次 all-reduce 慢」的实测注记（§6.2）——通信量恒等但实现更慢，是 §2 变体表的工程注脚
- [Narayanan et al., 2021](../references.md)：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM，arXiv:2104.04473，[A]（key: megatron-cluster）——TP/PP 组合的规模分析先例（本章 §6 接线图对应的系统侧走读归 6.2）
- [PyTorch, pytorch/pytorch repo](../references.md)（key: pytorch-repo，教学参照）——`torch.autograd.Function` 机制（1.1）与 `torch.distributed` 集合通信（4.1）的官方实现
