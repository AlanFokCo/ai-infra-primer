// 章节元数据抽取：H1 拆编号/中英标题、首个 > 块拆定位/前置/锚点、
// CJK 字数（阅读时长）、证据等级集合。TOC 由 Astro 的 headings 导出提供，此处不重复造 slugger。

/**
 * @param {string} source 章节 markdown 原文
 * @param {string} relPath 形如 models/02-transformer-anatomy/03-attention-mechanism.md
 */
export function parseChapter(source, relPath) {
  const parts = relPath.split('/');
  const file = parts[parts.length - 1].replace(/\.md$/, '');
  const partDir = parts[parts.length - 2];

  const h1Line = source.match(/^#\s+(.+)$/m);
  if (!h1Line) throw new Error(`${relPath}: 缺 H1`);
  // 「2.3 Attention 机制解剖（Anatomy of the Attention Mechanism）」
  const h1 = h1Line[1].trim().match(/^(\d+\.\d+)\s+(.+?)（([^（）]+)）\s*$/);
  if (!h1) throw new Error(`${relPath}: H1 不符合「编号 中文（English）」格式: ${h1Line[1]}`);

  const metaMatch = source.match(/^>\s+(.+)$/m);
  const metaRaw = metaMatch ? metaMatch[1] : '';
  const [positioning = '', prerequisites = '', anchors = ''] = metaRaw
    .split('|')
    .map((s) => s.trim());

  const prose = stripFences(source);
  const cjkChars = countCjk(prose);

  const evidenceLevels = [
    ...new Set(
      [...prose.matchAll(/\[([ABCW])\]|教学参照/g)].map((m) => m[1] ?? '教学参照')
    ),
  ].sort();

  // TOC：h2/h3 按文档序编 sec-N（rehype-heading-ids 插件在 HTML 侧按同序写 id）
  const headings = [];
  {
    let inFence = false;
    for (const line of source.split('\n')) {
      if (/^```/.test(line)) {
        inFence = !inFence;
        continue;
      }
      if (inFence) continue;
      const h = line.match(/^(#{2,3})\s+(.+)$/);
      if (h) {
        headings.push({ depth: h[1].length, text: h[2].trim(), id: `sec-${headings.length}` });
      }
    }
  }

  return {
    partDir,
    file,
    num: h1[1],
    titleZh: h1[2].trim(),
    titleEn: h1[3].trim(),
    metaLine: { raw: metaRaw, positioning, prerequisites, anchors },
    cjkChars,
    evidenceLevels,
    headings,
  };
}

function stripFences(source) {
  const lines = source.split('\n');
  let inFence = false;
  const out = [];
  for (const line of lines) {
    if (/^```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (!inFence) out.push(line);
  }
  return out.join('\n');
}

function countCjk(text) {
  const m = text.match(/[一-鿿]/g);
  return m ? m.length : 0;
}
