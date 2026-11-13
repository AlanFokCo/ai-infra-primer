// GitHub 适配写法 → web 标准 LaTeX 的回映。
// 源内容受 STYLE_GUIDE §3 白名单约束（为 GitHub math-renderer 让路的写法），
// 站点用 KaTeX 构建期渲染，能力面更宽，回映后输出更规范的 LaTeX。
// 铁律：代码围栏与行内 code span 内的内容一个字不动。

/** 按 ``` 围栏把源码分片；inFence 行原样保留 */
function mapOutsideFences(source, fn) {
  const lines = source.split('\n');
  let inFence = false;
  const out = [];
  for (const line of lines) {
    if (/^```/.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    out.push(inFence ? line : fn(line));
  }
  return out.join('\n');
}

/** 对单行（围栏外）做回映；先收编 $`...`$，再跳过其余 code span */
function transformLine(line) {
  // $`...`$ → $...$（GitHub 的 backtick 保护形式；内容单行、不含反引号）
  let s = line.replace(/\$`([^`\n]+)`\$/g, (_, math) => `$${math}$`);
  // 剩余 code span 分段，span 内不动
  return s
    .split(/(`[^`\n]+`)/)
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // code span
      return seg
        .replace(/\\thinspace\b/g, '\\,')
        .replace(/\\negthinspace\b/g, '\\!')
        .replace(/\\hspace\{5mu\}/g, '\\;')
        .replace(/\\hspace\{4mu\}/g, '\\:')
        .replace(/\\hspace\{(\d+)mu\}/g, '\\mkern$1mu'); // 兜底（语料当前只有 5mu）
    })
    .join('');
}

/** @param {string} source 章节 markdown 原文 */
export function transformMath(source) {
  return mapOutsideFences(source, transformLine);
}

/** 统计 $`...`$ 出现次数（出现级，非行级），供转换前后断言 */
export function countBacktickMath(source) {
  const m = source.match(/\$`[^`\n]+`\$/g);
  return m ? m.length : 0;
}
