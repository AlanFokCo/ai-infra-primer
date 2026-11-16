// 维度 UI 配置：落地页账本行、维度索引页文案、Part 依赖图几何。
// 章节与文献数据一律从 generated/manifest.json / references.json 读；
// 本文件只放「维度级静态文案与图形几何」这类不进 manifest 的东西。
//
// 图的几何是手工调定的 SVG 坐标（models 图 2026-11-13 上线版原样抽离）；
// edges: {line: [x1,y1,x2,y2]} 或 {path: '<svg d>'}，可选 label: {x, y, text}。

export const DIMENSIONS = {
  models: {
    key: 'models',
    zh: '模型',
    orderLabel: 'DIMENSION 01 / 05',
    scope: '生成式模型的数学：文本主干、视觉、扩散、多模态、视频、语音',
    lede: '生成式模型的数学：文本 LLM 主干、视觉表征、扩散生图、多模态 LLM、视频生成、语音音频。关键公式完整推导不跳步、逐公式附符号表、与官方实现代码对照；数值事实全部带出处并标注证据等级。',
    ledgerNote: '全部已关闭',
    depNoteShort: '十个 Part，五十一章。主干 P1→P5 按序读，支线从 P2 分出。',
    depNote:
      '主干 P1→P2→P3→P4→P5 是文本 LLM 线，按序读——P2 各章是后续一切的公共前置（attention、KV cache、RoPE、RMSNorm 的推导只在那里写一遍）。支线从 P2 分出：P6 讲图像如何变成 latent / token，P7 在其上建扩散数学，P8 复用 P6 的 ViT / CLIP 与 P7 的扩散目标，P9 把 P7 的 DiT 扩到时空维，P10 直接复用 P6.3 的 RVQ 公式建音频 codec。',
    prereqs: [
      '<strong>线性代数</strong>——矩阵乘法与 rank、内积与投影、特征分解；张量 shape 全书统一 [B, T, H, D] 记法',
      '<strong>概率论</strong>——条件分布与边缘化、期望、KL 散度、高斯分布、Bayes 规则；7.3 / 7.5 另需 ODE 直觉',
      '<strong>Python / PyTorch</strong>——能读官方 modeling 代码；代码走读锚点是 transformers、flash-attention、diffusers 等 repo 的具体文件与函数',
    ],
    prereqNote: '章节自包含：每个公式附符号表；跨章前置概念给链接，不重复展开。',
    firstChapter: { slug: '01-foundations/01-formulation', label: '从 1.1 开始读' },
    graph: {
      viewBox: '0 0 960 400',
      aria: 'Part 依赖图：主干 P1 到 P5，支线 P6 到 P10',
      nodes: [
        { x: 80, y: 44, n: 1 }, { x: 260, y: 44, n: 2 }, { x: 440, y: 44, n: 3 },
        { x: 620, y: 44, n: 4 }, { x: 800, y: 44, n: 5 },
        { x: 170, y: 184, n: 6 }, { x: 440, y: 184, n: 7 },
        { x: 620, y: 184, n: 8 }, { x: 800, y: 184, n: 9 },
        { x: 300, y: 322, n: 10 },
      ],
      edges: [
        { line: [172, 70, 254, 70] }, { line: [352, 70, 434, 70] },
        { line: [532, 70, 614, 70] }, { line: [712, 70, 794, 70] },
        { path: 'M285 96 C 285 150, 225 150, 222 178' },
        { line: [262, 210, 434, 210] },
        { line: [532, 210, 614, 210] }, { line: [712, 210, 794, 210] },
        { path: 'M190 236 C 160 300, 260 296, 292 318', label: { x: 196, y: 292, text: 'RVQ codec' } },
      ],
    },
  },

  training: {
    key: 'training',
    zh: '训练',
    orderLabel: 'DIMENSION 02 / 05',
    scope: 'autograd 与 optimizer、并行全家桶（DP/TP/PP/EP/CP）、Megatron、预训练与后训练全流程、RL 数学',
    lede: '训练动力学与训练系统：autograd 与算子反向、optimizer 与 LR 动力学、混合精度与 FP8、数据/张量/流水/专家并行、Megatron-LM 系统实现、预训练配方与稳定性、RL 数学（PPO/GRPO）与后训练全流程（SFT/RLHF/DPO/RLVR）。公式级推导 + 生产训练栈文件级走读 + 显存/通信/吞吐的账。',
    ledgerNote: null, // 建设期由各章状态决定，占位为 null 时前台显示章数即可
    depNoteShort: '九个 Part，三十九章。优化线 P1→P3、并行线 P4→P6，P7 合流，P8→P9 收官。',
    depNote:
      '主干 P1→P2→P3 是优化线：反向机制 → optimizer 与 LR → loss 函数族。P4→P5→P6 是并行与系统线：数据并行（ZeRO/FSDP）→ 模型并行（TP/PP/EP/CP）→ Megatron 实现。P7 预训练全流程合流两线；P8 RL 数学详解 → P9 后训练全流程。',
    prereqs: [
      '<strong>models 维度 Part 1-2</strong>——语言建模形式化、参数/计算/内存的账、Transformer 各组件推导（attention、KV cache、RoPE、RMSNorm、SwiGLU），或同等知识',
      '<strong>线性代数与概率论</strong>——同 models 维度；P8 另需 MDP 与随机过程的直觉（章内建立）',
      '<strong>Python / PyTorch</strong>——能读训练栈源码；代码走读锚点是 pytorch、Megatron-LM、DeepSpeed、TRL、veRL 等 repo 的具体文件与函数',
    ],
    prereqNote: '章节自包含：每个公式附符号表；跨章前置概念给纯文字指引（如「models 2.3 §3.4」），不重复展开。',
    firstChapter: { slug: '01-autograd-and-backprop/01-computation-graph-and-autograd', label: '从 1.1 开始读' },
    graph: {
      viewBox: '0 0 960 400',
      aria: 'Part 依赖图：优化线 P1 到 P3，并行线 P4 到 P6，应用线 P7 到 P9',
      nodes: [
        { x: 80, y: 44, n: 1 }, { x: 300, y: 44, n: 2 }, { x: 520, y: 44, n: 3 },
        { x: 80, y: 184, n: 4 }, { x: 300, y: 184, n: 5 }, { x: 520, y: 184, n: 6 },
        { x: 520, y: 322, n: 7 }, { x: 740, y: 44, n: 8 }, { x: 740, y: 322, n: 9 },
      ],
      edges: [
        // 优化线 P1→P2→P3
        { line: [172, 70, 294, 70] }, { line: [392, 70, 514, 70] },
        // P1→P4、P2→P4（并行线的前置）
        { line: [126, 96, 126, 178] }, { path: 'M346 96 C 346 140, 200 150, 168 178' },
        // 并行线 P4→P5→P6
        { line: [172, 210, 294, 210] }, { line: [392, 210, 514, 210] },
        // P6→P7（infra → 预训练）、P2→P7（配方理论）
        { line: [566, 236, 566, 316] },
        { path: 'M346 96 C 380 250, 470 290, 514 322', label: { x: 420, y: 240, text: 'optimizer 配方' } },
        // P3→P9（偏好目标接口）
        { path: 'M566 96 C 640 200, 700 280, 782 318' },
        // P8→P9、P7→P9
        { path: 'M786 96 C 800 200, 800 280, 786 316', label: { x: 806, y: 200, text: 'RL 数学' } },
        { line: [612, 348, 734, 348] },
      ],
    },
  },
};

export const DIMENSION_ORDER = ['models', 'training', 'inference', 'gpu', 'cluster'];

// 未上线维度的落地页占位文案（live=false 行）
export const FORTHCOMING = {
  inference: { zh: '推理', scope: 'continuous batching、PagedAttention、speculative decoding' },
  gpu: { zh: '硬件', scope: '内存层级、roofline、FlashAttention 等算子的实现层' },
  cluster: { zh: '集群', scope: 'EP、all-to-all、Ring Attention、网络拓扑' },
};
