// 内容管线：models/**/*.md → src/generated/（chapters + manifest.json + references.json）
// 源文件只读，一个字符不改；所有 web 适配在此处完成。
import { readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { transformMath, countBacktickMath } from './lib/math-transform.mjs';
import { parseChapter } from './lib/metadata.mjs';
import { rewriteLinks } from './lib/links.mjs';
import { parseReferences } from './lib/references.mjs';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const REPO = path.resolve(SITE, '..');
const MODELS = path.join(REPO, 'models');
const OUT = path.join(SITE, 'src/generated');
const BASE = '/ai-infra-primer';

// Part 元数据（事实源：models/README.md 目录节；tagline 为该 Part 章节描述的浓缩）
const PARTS = [
  { num: 1, dir: '01-foundations', titleEn: 'Foundations', titleZh: '基础', tagline: '语言建模的数学形式化、参数/计算/内存的账、scaling laws' },
  { num: 2, dir: '02-transformer-anatomy', titleEn: 'Transformer Anatomy', titleZh: 'Transformer 解剖', tagline: '从 tokenization 到组装完整 decoder block，每个组件的推导' },
  { num: 3, dir: '03-objectives-and-variants', titleEn: 'Objectives & Variants', titleZh: '目标与变体', tagline: 'CLM / MLM 统一形式化、encoder 家族、in-context learning' },
  { num: 4, dir: '04-modern-architectures', titleEn: 'Modern Architectures', titleZh: '现代架构', tagline: 'KV 压缩、sparse / linear attention、MoE、long context、可解释性' },
  { num: 5, dir: '05-case-studies', titleEn: 'Case Studies', titleZh: '案例研究', tagline: 'Llama / DeepSeek / Qwen / GLM / Kimi 五家谱系逐代拆解' },
  { num: 6, dir: '06-vision-representations', titleEn: 'Vision Representations', titleZh: '视觉表征', tagline: 'ViT、CLIP 对比学习、visual tokenizer' },
  { num: 7, dir: '07-diffusion-image-generation', titleEn: 'Diffusion & Image Generation', titleZh: '扩散与图像生成', tagline: 'DDPM、SDE/ODE、guidance、flow matching、DiT' },
  { num: 8, dir: '08-multimodal-llm', titleEn: 'Multimodal LLMs', titleZh: '多模态 LLM', tagline: 'VLM 架构分类学、connector、训练目标、视频理解、统一生成' },
  { num: 9, dir: '09-video-generation', titleEn: 'Video Generation', titleZh: '视频生成', tagline: 'video VAE、时空 DiT、训练加速、控制与编辑' },
  { num: 10, dir: '10-audio-speech', titleEn: 'Audio & Speech', titleZh: '音频与语音', tagline: '音频表征与 codec、ASR、TTS、speech LLM 与全双工' },
];

function main() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(path.join(OUT, 'chapters'), { recursive: true });

  const manifest = { generatedAt: new Date().toISOString().slice(0, 10), parts: [] };
  let chapterCount = 0;
  let totalCjk = 0;
  let totalFormulas = 0;

  for (const part of PARTS) {
    const partPath = path.join(MODELS, part.dir);
    const files = readdirSync(partPath)
      .filter((f) => /^\d{2}-[\da-z-]+\.md$/.test(f))
      .sort();
    const chapters = [];

    for (const f of files) {
      const relPath = `models/${part.dir}/${f}`;
      const source = readFileSync(path.join(MODELS, part.dir, f), 'utf8');
      const meta = parseChapter(source, relPath);

      // 断言：转换前后 $`...`$ 计数守恒（转换后必须为 0）
      const before = countBacktickMath(source);
      let body = transformMath(source);
      const after = countBacktickMath(body);
      if (after !== 0) throw new Error(`${relPath}: $'...'$ 转换残留 ${after}/${before}`);
      if (/\\thinspace\b|\\negthinspace\b|\\hspace\{[45]mu\}/.test(body)) {
        throw new Error(`${relPath}: 间距命令回映残留`);
      }

      // 公式计数（转换后口径：display 块 + 行内）——在下方计入 frontmatter

      body = rewriteLinks(body, { base: BASE, partDir: part.dir });
      if (body.includes('](../references.md')) {
        throw new Error(`${relPath}: references 相对链接重写残留`);
      }

      // 生成文件：frontmatter + 正文（去掉源 H1 与 metadata 块——两者由章节布局
      // 从 frontmatter 以设计化版面呈现，meta 内的 $...$ 由布局侧 KaTeX 渲染）
      const bodyStripped = stripHeaderBlock(body);
      const chapterFormulas = countFormulas(body);
      totalFormulas += chapterFormulas;
      const frontmatter = toFrontmatter({
        num: meta.num,
        titleZh: meta.titleZh,
        titleEn: meta.titleEn,
        partDir: part.dir,
        partNum: part.num,
        partTitleEn: part.titleEn,
        partTitleZh: part.titleZh,
        file: meta.file,
        cjkChars: meta.cjkChars,
        formulas: chapterFormulas,
        evidenceLevels: meta.evidenceLevels,
        headings: meta.headings,
        positioning: transformMath(meta.metaLine.positioning),
        prerequisites: transformMath(meta.metaLine.prerequisites),
        anchors: transformMath(meta.metaLine.anchors),
      });
      const outDir = path.join(OUT, 'chapters', part.dir);
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

    manifest.parts.push({
      num: part.num,
      dir: part.dir,
      titleEn: part.titleEn,
      titleZh: part.titleZh,
      tagline: part.tagline,
      chapters,
    });
  }

  // 全书阅读顺序的 prev/next（Part 升序、章号升序）
  const flat = manifest.parts.flatMap((p) =>
    p.chapters.map((c) => ({ ...c, partDir: p.dir, partNum: p.num, partTitleEn: p.titleEn, partTitleZh: p.titleZh }))
  );
  flat.forEach((ch, i) => {
    ch.prev = i > 0 ? { slug: flat[i - 1].slug, num: flat[i - 1].num, titleZh: flat[i - 1].titleZh } : null;
    ch.next = i < flat.length - 1 ? { slug: flat[i + 1].slug, num: flat[i + 1].num, titleZh: flat[i + 1].titleZh } : null;
  });
  manifest.flat = flat;
  manifest.stats = { chapters: chapterCount, parts: PARTS.length, totalCjk, formulas: totalFormulas };

  writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));

  // references.md
  const refSource = readFileSync(path.join(MODELS, 'references.md'), 'utf8');
  const refs = parseReferences(refSource);
  const refCount = refs.groups.reduce((n, g) => n + g.entries.length, 0);
  writeFileSync(path.join(OUT, 'references.json'), JSON.stringify(refs, null, 2));

  // 全局断言
  if (chapterCount !== 51) throw new Error(`章节数 ${chapterCount} !== 51`);
  if (manifest.parts.length !== 10) throw new Error(`Part 数 ${manifest.parts.length} !== 10`);
  if (refCount < 100) throw new Error(`references 条目数 ${refCount} < 100`);

  console.log(
    `[build-content] ${chapterCount} 章 / ${PARTS.length} Part / ${refCount} 条文献 / ${(totalCjk / 10000).toFixed(1)} 万 CJK 字`
  );
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
