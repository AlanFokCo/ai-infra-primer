// 内容管线：models/**/*.md → src/generated/（chapters + manifest.json + references.json）
// 源文件只读，一个字符不改；所有 web 适配在此处完成。
import { readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync, copyFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { transformMath, countBacktickMath } from './lib/math-transform.mjs';
import { parseChapter } from './lib/metadata.mjs';
import { rewriteLinks } from './lib/links.mjs';
import { parseReferences } from './lib/references.mjs';
import { renderFigure, renderGridFigure, renderAlgo, classifyBlock } from './lib/ascii-figure.mjs';
import { SIGNATURE_FIGURES as CORE_FIVE } from './lib/figures-signature.mjs';

// 汇总签名图：核心五张 + figures/ 目录下全部模块
const figuresDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'lib/figures');
const SIG_MODULES = readdirSync(figuresDir).filter((f) => f.endsWith('.mjs'));
const MODULE_FIGS = [];
for (const f of SIG_MODULES) {
  const mod = await import(`./lib/figures/${f}`);
  if (Array.isArray(mod.FIGURES)) MODULE_FIGS.push(...mod.FIGURES);
}
const SIGNATURE_FIGURES = [...CORE_FIVE, ...MODULE_FIGS];

// 场景替换注册表：dim/slug#blockIndex → figure
// figures 模块的 slug 形如 '02-transformer-anatomy/03-attention-mechanism'（models 存量约定）；
// dim 字段缺省视为 'models'（101 张存量注册零改动），training 图模块显式写 dim: 'training'。
const sceneByBlock = new Map();
const sceneKey = (dim, slug, idx) => `${dim}/${slug}#${idx}`;
for (const f of SIGNATURE_FIGURES) {
  const dim = f.dim ?? 'models';
  if (typeof f.replacesBlock === 'number') sceneByBlock.set(sceneKey(dim, f.slug, f.replacesBlock), f);
}

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const REPO = path.resolve(SITE, '..');
const OUT = path.join(SITE, 'src/generated');
const BASE = '/ai-infra-primer';

// 维度层（training spec §8）：每维度 = key / 源目录 / PARTS 表 / references 文件 / 断言组。
// Part 元数据事实源：各维度 README.md 目录节；tagline 为该 Part 章节描述的浓缩。
// 断言口径：models 收官锁定 51/10；training 建设期 = 当波应发布章数（W1 起滚动收紧，W7 锁死 39/9）。
const DIMENSIONS = [
  {
    key: 'models',
    parts: [
      { num: 1, dir: '01-foundations', titleEn: 'Foundations', titleZh: '基础', tagline: '语言建模的数学形式化、参数/计算/内存的账、scaling laws', wave: 'W1' },
      { num: 2, dir: '02-transformer-anatomy', titleEn: 'Transformer Anatomy', titleZh: 'Transformer 解剖', tagline: '从 tokenization 到组装完整 decoder block，每个组件的推导', wave: 'W1' },
      { num: 3, dir: '03-objectives-and-variants', titleEn: 'Objectives & Variants', titleZh: '目标与变体', tagline: 'CLM / MLM 统一形式化、encoder 家族、in-context learning', wave: 'W2' },
      { num: 4, dir: '04-modern-architectures', titleEn: 'Modern Architectures', titleZh: '现代架构', tagline: 'KV 压缩、sparse / linear attention、MoE、long context、可解释性', wave: 'W2' },
      { num: 5, dir: '05-case-studies', titleEn: 'Case Studies', titleZh: '案例研究', tagline: 'Llama / DeepSeek / Qwen / GLM / Kimi 五家谱系逐代拆解', wave: 'W3' },
      { num: 6, dir: '06-vision-representations', titleEn: 'Vision Representations', titleZh: '视觉表征', tagline: 'ViT、CLIP 对比学习、visual tokenizer', wave: 'W3' },
      { num: 7, dir: '07-diffusion-image-generation', titleEn: 'Diffusion & Image Generation', titleZh: '扩散与图像生成', tagline: 'DDPM、SDE/ODE、guidance、flow matching、DiT', wave: 'W4' },
      { num: 8, dir: '08-multimodal-llm', titleEn: 'Multimodal LLMs', titleZh: '多模态 LLM', tagline: 'VLM 架构分类学、connector、训练目标、视频理解、统一生成', wave: 'W5' },
      { num: 9, dir: '09-video-generation', titleEn: 'Video Generation', titleZh: '视频生成', tagline: 'video VAE、时空 DiT、训练加速、控制与编辑', wave: 'W6' },
      { num: 10, dir: '10-audio-speech', titleEn: 'Audio & Speech', titleZh: '音频与语音', tagline: '音频表征与 codec、ASR、TTS、speech LLM 与全双工', wave: 'W7' },
    ],
    asserts: { chapters: 51, parts: 10, refsMin: 100 },
  },
  {
    key: 'training',
    parts: [
      { num: 1, dir: '01-autograd-and-backprop', titleEn: 'Autograd & Backprop', titleZh: 'Autograd 与反向机制', tagline: '计算图与 vjp、算子反向推导、混合精度与 FP8、checkpointing、accumulation', wave: 'W1' },
      { num: 2, dir: '02-optimizers', titleEn: 'Optimizers & LR', titleZh: 'Optimizer 与 LR 动力学', tagline: 'SGD/momentum → Adam → AdamW、LR schedule、梯度裁剪、Muon 与谱方法', wave: 'W1' },
      { num: 3, dir: '03-loss-functions', titleEn: 'Loss Functions', titleZh: 'Loss 函数族', tagline: 'CE 数值稳定变体、z-loss、偏好与对比目标的接口', wave: 'W2' },
      { num: 4, dir: '04-data-parallelism', titleEn: 'Data Parallelism', titleZh: '数据并行与显存切分', tagline: 'DDP、ZeRO 各阶段、FSDP/FSDP2、显存与通信的统一账', wave: 'W2' },
      { num: 5, dir: '05-model-parallelism', titleEn: 'Model Parallelism', titleZh: '模型并行全家桶', tagline: 'TP 列/行切分、SP/CP、PP 气泡与 schedule、EP 与 all-to-all、组合策略', wave: 'W3' },
      { num: 6, dir: '06-megatron-and-stacks', titleEn: 'Megatron & Training Stacks', titleZh: 'Megatron 与现代训练栈', tagline: '层栈并行原语走读、运行时与 distributed optimizer、框架选型', wave: 'W3' },
      { num: 7, dir: '07-pretraining', titleEn: 'Pretraining', titleZh: '预训练全流程', tagline: '数据管线与配比、配方、训练稳定性、checkpoint 与续训、长上下文扩展', wave: 'W4' },
      { num: 8, dir: '08-rl-math', titleEn: 'RL Mathematics', titleZh: 'RL 数学详解', tagline: '策略梯度定理、GAE、PPO、GRPO 与变体、LLM-RL 系统实现', wave: 'W5' },
      { num: 9, dir: '09-post-training', titleEn: 'Post-Training', titleZh: '后训练全流程', tagline: 'SFT、RLHF 全链、DPO 家族、RFT 与拒绝采样、RLVR 与推理模型', wave: 'W6' },
    ],
    // 建设期断言 = 当波应发布章数（滚动收紧）：W1=10 / W2=16 / W3=24 / W4=29 / W5=34 / W6=39
    asserts: { chapters: 39, parts: 9, refsMin: 87 },
  },
  {
    key: 'inference',
    parts: [
      { num: 1, dir: '01-principles-and-accounting', titleEn: 'Principles & Accounting', titleZh: '推理原理与性能账', tagline: '请求生命周期与 SLO、prefill/decode 算力带宽账、roofline 与排队论、采样与结构化输出', wave: 'W1' },
      { num: 2, dir: '02-kv-cache-systems', titleEn: 'KV Cache Systems', titleZh: 'KV cache 系统', tagline: 'PagedAttention 块管理、prefix caching/RadixAttention、KV 量化与驱逐、offload 与分层存储', wave: 'W1' },
      { num: 3, dir: '03-inference-operators', titleEn: 'Inference Operators', titleZh: '推理算子', tagline: 'decode attention 族、权重量化算法、量化 GEMM 与 fused MoE、融合与图捕获', wave: 'W2' },
      { num: 4, dir: '04-scheduling-and-batching', titleEn: 'Scheduling & Batching', titleZh: '调度与 batching', tagline: 'continuous batching、调度器与抢占、chunked prefill、投机解码原理与系统', wave: 'W2' },
      { num: 5, dir: '05-vllm', titleEn: 'vLLM', titleZh: 'vLLM 走读', tagline: 'V1 架构、scheduler 与 KV manager、分布式执行、高级特性', wave: 'W3' },
      { num: 6, dir: '06-sglang', titleEn: 'SGLang', titleZh: 'SGLang 走读', tagline: '架构与 RadixAttention、overlap scheduler、分布式与 PD', wave: 'W3' },
      { num: 7, dir: '07-deepinfer-and-deepseek', titleEn: 'DeepInfer & DeepSeek Stack', titleZh: 'DeepInfer 与 DeepSeek 栈', tagline: '架构走读、DSpark 与 SWA bounded replay、kernel 栈、大规模 EP 与负载均衡', wave: 'W4' },
      { num: 8, dir: '08-distributed-and-pd', titleEn: 'Distributed Inference & PD', titleZh: '分布式推理与 PD 分离', tagline: 'serving 并行账、PD 原理 goodput、KV 传输、PD 系统对照、弹性容错', wave: 'W5' },
      { num: 9, dir: '09-production', titleEn: 'Production', titleZh: '生产全流程', tagline: '容量与成本账、引擎选型、RL rollout serving、多模态与特殊 serving', wave: 'W6' },
    ],
    // 建设期断言滚动：W0=0/0 → W1=8/2 → W2=17/4 → W3=24/6 → W4=28/7 → W5=33/8 → W6=37/9 → W7 锁死
    asserts: { chapters: 37, parts: 9, refsMin: 50 },
  },
  {
    key: 'gpu',
    parts: [
      { num: 1, dir: '01-gpu-microarchitecture', titleEn: 'GPU Microarchitecture', titleZh: 'GPU 微架构拆解', tagline: '芯片组织与世代、SM 内部与 occupancy 物理上界、Tensor Core MMA、指令流水线与延迟隐藏', wave: 'W1' },
      { num: 2, dir: '02-memory-hierarchy', titleEn: 'Memory Hierarchy & Bandwidth', titleZh: '内存层级与带宽账', tagline: 'HBM 带宽来源、缓存与片上存储层级、register spill 账、roofline 微架构来源与测量', wave: 'W1' },
      { num: 3, dir: '03-cuda-programming-model', titleEn: 'CUDA Programming Model', titleZh: 'CUDA 编程模型', tagline: 'SIMT 与 divergence 账、内存模型与地址空间、stream/event/graph 异步、occupancy 计算', wave: 'W2' },
      { num: 4, dir: '04-kernel-memory-optimization', titleEn: 'Kernel Memory Optimization', titleZh: 'Kernel 访存优化', tagline: 'coalescing 与向量化、bank conflict 消除、tiling 强度推导、cp.async/TMA 流水线', wave: 'W2' },
      { num: 5, dir: '05-kernel-compute-primitives', titleEn: 'Kernel Compute & Primitives', titleZh: 'Kernel 计算优化与并行原语', tagline: 'reduction/scan 带宽最优性、CUTLASS GEMM 走读、softmax/norm kernel、算子融合账', wave: 'W3' },
      { num: 6, dir: '06-low-precision-attention-kernels', titleEn: 'Low-Precision & Attention Kernels', titleZh: '低精度与注意力 kernel', tagline: '低精度格式与 MMA 吞吐、Marlin/DeepGEMM 量化 kernel、FlashAttention 指令级走读、MLA/GQA/paged 变体', wave: 'W3' },
      { num: 7, dir: '07-nvlink-nvswitch', titleEn: 'NVLink & NVSwitch', titleZh: '节点内互连 NVLink/NVSwitch', tagline: 'NVLink/NVSwitch 架构与世代、节点内拓扑带宽账、GPUDirect P2P', wave: 'W4' },
      { num: 8, dir: '08-nccl-collectives', titleEn: 'NCCL Collectives', titleZh: 'NCCL 集合通信', tagline: '集合原语消息量账、ring 带宽最优性证明、tree 延迟交叉、NCCL 实现走读', wave: 'W4' },
      { num: 9, dir: '09-rdma-gpudirect', titleEn: 'RDMA & GPUDirect', titleZh: 'RDMA 与 GPUDirect', tagline: 'RDMA verbs 与 IB/RoCE、GPUDirect RDMA 带宽账、DCQCN/PFC 拥塞与可靠性', wave: 'W5' },
      { num: 10, dir: '10-fabric-and-scale-communication', titleEn: 'Fabric & Scale Communication', titleZh: '机间 fabric 与大规模通信', tagline: 'fabric 拓扑与收敛比账、SHARP 网内聚合、DeepEP a2a 实现、Ring Attention 通信', wave: 'W5' },
      { num: 11, dir: '11-system-integration', titleEn: 'System Integration & Production', titleZh: '系统集成与生产', tagline: '服务器形态与 MIG 分区、Nsight profiling 与调优、ECC/XID 故障诊断', wave: 'W6' },
    ],
    // 建设期断言滚动：W0={0,0} → W1={8,2} → W2={16,4} → W3={24,6} → W4={31,8} → W5={38,10}（当前）→ W6={41,11} → W7 锁死
    asserts: { chapters: 38, parts: 10, refsMin: 50 },
  },
];

function main() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(path.join(OUT, 'chapters'), { recursive: true });

  const manifest = { generatedAt: new Date().toISOString().slice(0, 10), dimensions: {} };
  const references = {};
  const grand = { chapters: 0, parts: 0, totalCjk: 0, formulas: 0, refs: 0 };

  for (const dim of DIMENSIONS) {
    const SRC = path.join(REPO, dim.key);
    const dimOut = { key: dim.key, parts: [] };
    let chapterCount = 0;
    let totalCjk = 0;
    let totalFormulas = 0;

    for (const part of dim.parts) {
      const partPath = path.join(SRC, part.dir);
      let files = [];
      try {
        files = readdirSync(partPath)
          .filter((f) => /^\d{2}-[\da-z-]+\.md$/.test(f))
          .sort();
      } catch {
        // 建设期维度允许 Part 目录尚不存在（training W0）：跳过，不计入 manifest
      }
      if (files.length === 0) continue;
      const chapters = [];

      for (const f of files) {
        const relPath = `${dim.key}/${part.dir}/${f}`;
        const source = readFileSync(path.join(SRC, part.dir, f), 'utf8');
        const meta = parseChapter(source, relPath);

        // 断言：转换前后 $`...`$ 计数守恒（转换后必须为 0）
        const before = countBacktickMath(source);
        let body = transformMath(source);
        const after = countBacktickMath(body);
        if (after !== 0) throw new Error(`${relPath}: $'...'$ 转换残留 ${after}/${before}`);
        if (/\\thinspace\b|\\negthinspace\b|\\hspace\{[45]mu\}/.test(body)) {
          throw new Error(`${relPath}: 间距命令回映残留`);
        }

        body = rewriteLinks(body, { base: BASE, partDir: part.dir, dim: dim.key });
        if (body.includes('](../references.md')) {
          throw new Error(`${relPath}: references 相对链接重写残留`);
        }

        // ASCII 结构图 → 矢量动图（SVG + 数据流光点）；伪代码/表格类围栏保持深色图版
        body = replaceDiagrams(body, dim.key, `${part.dir}/${meta.file}`);
        // 签名动图：按锚点插入（新增描述图，不动源文件）
        body = insertSignatureFigures(body, dim.key, `${part.dir}/${meta.file}`);

        // 生成文件：frontmatter + 正文（去掉源 H1 与 metadata 块——两者由章节布局
        // 从 frontmatter 以设计化版面呈现，meta 内的 $...$ 由布局侧 KaTeX 渲染）
        const bodyStripped = stripHeaderBlock(body);
        const chapterFormulas = countFormulas(body);
        totalFormulas += chapterFormulas;
        const frontmatter = toFrontmatter({
          dim: dim.key,
          num: meta.num,
          titleZh: meta.titleZh,
          titleEn: meta.titleEn,
          partDir: part.dir,
          partNum: part.num,
          partTitleEn: part.titleEn,
          partTitleZh: part.titleZh,
          partWave: part.wave,
          file: meta.file,
          cjkChars: meta.cjkChars,
          formulas: chapterFormulas,
          evidenceLevels: meta.evidenceLevels,
          headings: meta.headings,
          positioning: transformMath(meta.metaLine.positioning),
          prerequisites: transformMath(meta.metaLine.prerequisites),
          anchors: transformMath(meta.metaLine.anchors),
        });
        const outDir = path.join(OUT, 'chapters', dim.key, part.dir);
        mkdirSync(outDir, { recursive: true });
        writeFileSync(path.join(outDir, f), frontmatter + bodyStripped);

        chapters.push({
          num: meta.num,
          file: meta.file,
          titleZh: meta.titleZh,
          titleEn: meta.titleEn,
          slug: `${part.dir}/${meta.file}`,
          cjkChars: meta.cjkChars,
          formulas: chapterFormulas,
          evidenceLevels: meta.evidenceLevels,
        });
        chapterCount += 1;
        totalCjk += meta.cjkChars;
      }

      dimOut.parts.push({
        num: part.num,
        dim: dim.key,
        dir: part.dir,
        titleEn: part.titleEn,
        titleZh: part.titleZh,
        tagline: part.tagline,
        chapters,
      });
    }

    // 维度内阅读顺序的 prev/next（Part 升序、章号升序）
    const flat = dimOut.parts.flatMap((p) =>
      p.chapters.map((c) => ({ ...c, dim: dim.key, partDir: p.dir, partNum: p.num, partTitleEn: p.titleEn, partTitleZh: p.titleZh }))
    );
    flat.forEach((ch, i) => {
      ch.prev = i > 0 ? { slug: flat[i - 1].slug, num: flat[i - 1].num, titleZh: flat[i - 1].titleZh } : null;
      ch.next = i < flat.length - 1 ? { slug: flat[i + 1].slug, num: flat[i + 1].num, titleZh: flat[i + 1].titleZh } : null;
    });
    dimOut.flat = flat;
    dimOut.stats = { chapters: chapterCount, parts: dimOut.parts.length, totalCjk, formulas: totalFormulas };
    manifest.dimensions[dim.key] = dimOut;

    // references.md
    const refSource = readFileSync(path.join(SRC, 'references.md'), 'utf8');
    const refs = parseReferences(refSource);
    const refCount = refs.groups.reduce((n, g) => n + g.entries.length, 0);
    references[dim.key] = refs;

    // 维度断言组
    if (chapterCount !== dim.asserts.chapters)
      throw new Error(`${dim.key} 章节数 ${chapterCount} !== ${dim.asserts.chapters}`);
    if (dimOut.parts.length !== dim.asserts.parts)
      throw new Error(`${dim.key} Part 数 ${dimOut.parts.length} !== ${dim.asserts.parts}`);
    if (refCount < dim.asserts.refsMin)
      throw new Error(`${dim.key} references 条目数 ${refCount} < ${dim.asserts.refsMin}`);

    grand.chapters += chapterCount;
    grand.parts += dimOut.parts.length;
    grand.totalCjk += totalCjk;
    grand.formulas += totalFormulas;
    grand.refs += refCount;
    console.log(
      `[build-content] ${dim.key}: ${chapterCount} 章 / ${dimOut.parts.length} Part / ${refCount} 条文献 / ${(totalCjk / 10000).toFixed(1)} 万 CJK 字`
    );
  }

  manifest.stats = grand;
  writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));
  writeFileSync(path.join(OUT, 'references.json'), JSON.stringify(references, null, 2));

  // KaTeX CSS 补丁版：注入 font-display: swap（原 css 未声明，公式字体 FOIT 会阻塞首屏 LCP），
  // 字体只收 woff2（现代浏览器全覆盖），重写到同目录 fonts/ 供 Vite 打包
  {
    const katexDir = path.join(SITE, 'node_modules/katex/dist');
    let css = readFileSync(path.join(katexDir, 'katex.min.css'), 'utf8');
    css = css.replace(/@font-face\{/g, '@font-face{font-display:swap;');
    // 去掉 woff/ttf 回退引用（精确匹配 format 串，不越界；KaTeX minified 无尾分号）
    css = css.replace(
      /,url\([^)]+\.woff\)\s*format\(["']woff["']\),url\([^)]+\.ttf\)\s*format\(["']truetype["']\)/g,
      ''
    );
    // 结构自检：@font-face 数与 } 配平必须与注入量一致
    const faces = (css.match(/@font-face\{/g) || []).length;
    const woff2Srcs = (css.match(/\.woff2\)/g) || []).length;
    const braceOpen = (css.match(/\{/g) || []).length;
    const braceClose = (css.match(/\}/g) || []).length;
    if (faces !== woff2Srcs || faces < 20 || braceOpen !== braceClose) {
      throw new Error(
        `katex css 补丁异常：@font-face ${faces} / woff2 src ${woff2Srcs} / braces ${braceOpen}≠${braceClose}`
      );
    }
    const fontsOut = path.join(OUT, 'fonts');
    mkdirSync(fontsOut, { recursive: true });
    for (const f of readdirSync(path.join(katexDir, 'fonts'))) {
      if (f.endsWith('.woff2')) copyFileSync(path.join(katexDir, 'fonts', f), path.join(fontsOut, f));
    }
    writeFileSync(path.join(OUT, 'katex.min.css'), css);
  }
}

/** 无语言围栏 → 场景图（有注册场景时）/结构图/网格图/算法面板；带语言围栏逐字保留 */
function replaceDiagrams(body, dim, slug) {
  const lines = body.split('\n');
  const out = [];
  let i = 0;
  let figNo = 0;
  let blockIdx = 0;
  let algoNo = 0;
  while (i < lines.length) {
    const open = lines[i].match(/^```(\w*)\s*$/);
    if (!open) {
      out.push(lines[i]);
      i += 1;
      continue;
    }
    // 收齐围栏到闭合行
    const buf = [];
    let j = i + 1;
    while (j < lines.length && !/^```\s*$/.test(lines[j])) {
      buf.push(lines[j]);
      j += 1;
    }
    if (open[1] !== '' || j >= lines.length) {
      // 带语言围栏或未闭合：逐字保留
      out.push(lines[i], ...buf, lines[j] ?? '```');
      i = j + 1;
      continue;
    }
    const myBlock = blockIdx++;
    const registered = sceneByBlock.get(sceneKey(dim, slug, myBlock));
    if (registered) {
      figNo += 1;
      out.push(
        '',
        `<figure class="fig fig-sig" data-fig="${slug}#block-${myBlock}">`,
        `<span class="fig-tag">${registered.title}</span>`,
        `<div class="fig-body">`,
        registered.svg(),
        `</div>`,
        `</figure>`,
        ''
      );
      i = j + 1;
      continue;
    }
    const kind = classifyBlock(buf);
    if (kind === 'diagram') {
      figNo += 1;
      const svg = renderFigure(buf);
      const wide = /viewBox="0 0 (\d+)/.exec(svg);
      const w = wide ? Number(wide[1]) : 0;
      const wideAttrs = w > 760 ? ` data-wide style="--fig-w:${w}px"` : '';
      out.push(
        '',
        `<figure class="fig"${wideAttrs} data-fig="${slug}#${figNo - 1}">`,
        `<span class="fig-tag">FIG ${figNo}</span>`,
        `<div class="fig-body">`,
        svg,
        `</div>`,
        `</figure>`,
        ''
      );
    } else if (kind === 'table') {
      figNo += 1;
      out.push(
        '',
        `<figure class="fig fig-grid" data-fig="${slug}#${figNo - 1}">`,
        `<span class="fig-tag">FIG ${figNo}</span>`,
        `<div class="fig-body">`,
        renderGridFigure(buf),
        `</div>`,
        `</figure>`,
        ''
      );
    } else {
      algoNo += 1;
      out.push('', renderAlgo(buf), '');
    }
    i = j + 1;
  }
  return out.join('\n');
}

/** 签名动图插入：匹配到锚点标题行后紧跟插入 */
function insertSignatureFigures(body, dim, slug) {
  const targets = SIGNATURE_FIGURES.filter((f) => (f.dim ?? 'models') === dim && f.slug === slug);
  if (targets.length === 0) return body;
  const lines = body.split('\n');
  const out = [];
  const done = new Set();
  for (const line of lines) {
    out.push(line);
    for (const t of targets) {
      if (t.afterHeading && !done.has(t.title) && t.afterHeading.test(line)) {
        done.add(t.title);
        out.push(
          '',
          `<figure class="fig fig-sig" data-fig="${slug}#sig-${t.title}">`,
          `<span class="fig-tag">${t.title}</span>`,
          `<div class="fig-body">`,
          t.svg(),
          `</div>`,
          `</figure>`,
          ''
        );
      }
    }
  }
  return out.join('\n');
}

/** display $$...$$ + 行内 $...$ 计数（跳过代码围栏） */
function countFormulas(body) {
  const lines = body.split('\n');
  let inFence = false;
  const prose = [];
  for (const line of lines) {
    if (/^```/.test(line)) inFence = !inFence;
    else if (!inFence) prose.push(line);
  }
  const text = prose.join('\n');
  const display = text.match(/\$\$[\s\S]+?\$\$/g) || [];
  const withoutDisplay = text.replace(/\$\$[\s\S]+?\$\$/g, '');
  const inline = withoutDisplay.match(/\$[^$\n]+?\$/g) || [];
  return display.length + inline.length;
}

/** 去掉源文件 H1 行与紧随的 metadata 引用块（页面布局从 frontmatter 呈现二者） */
function stripHeaderBlock(body) {
  const lines = body.split('\n');
  const out = [];
  let h1Skipped = false;
  let metaSkipped = false;
  for (const line of lines) {
    if (!h1Skipped && /^#\s+/.test(line)) {
      h1Skipped = true;
      continue;
    }
    if (h1Skipped && !metaSkipped) {
      if (line.trim() === '') continue; // H1 与引用块之间的空行
      if (/^>\s?/.test(line)) {
        metaSkipped = true;
        continue;
      }
    }
    out.push(line);
  }
  return out.join('\n').replace(/^\n+/, '');
}

function toFrontmatter(obj) {
  const lines = ['---'];
  for (const [k, v] of Object.entries(obj)) {
    lines.push(`${k}: ${JSON.stringify(v)}`);
  }
  lines.push('---', '');
  return lines.join('\n');
}

main();
