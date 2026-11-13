// 站内链接重写：把仓库相对路径映射到站点路由。
// 语料实测：章节内只出现 ../references.md（1229 处）与外部 http 链接；
// bare 章节文件链接当前为零，但保留处理能力（防御性）。
// 铁律：代码围栏与行内 code span 内的内容不动。

/**
 * @param {string} md markdown 原文
 * @param {{ base: string, partDir: string }} ctx base 形如 '/ai-infra-primer'
 */
export function rewriteLinks(md, ctx) {
  const { base } = ctx;
  const lines = md.split('\n');
  let inFence = false;
  const out = [];
  for (const line of lines) {
    if (/^```/.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    out.push(inFence ? line : rewriteLine(line, base));
  }
  return out.join('\n');
}

function rewriteLine(line, base) {
  return line
    .split(/(`[^`\n]+`)/)
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // code span 不动
      return seg.replace(
        /\]\(([^)\s]+)\)/g,
        (whole, target) => `](${rewriteTarget(target, base)})`
      );
    })
    .join('');
}

function rewriteTarget(target, base) {
  if (/^https?:\/\//.test(target) || target.startsWith('#') || target.startsWith('mailto:')) {
    return target;
  }
  // ../references.md 或 ../references.md#anchor
  const refMatch = target.match(/^\.\.\/references\.md(#.*)?$/);
  if (refMatch) return `${base}/models/references/${refMatch[1] ?? ''}`;
  // 跨维度链接（../../inference/README.md 等）：维度未上线，指向 GitHub 仓库源
  const dimMatch = target.match(/^\.\.\/\.\.\/([\da-z-]+)\/(README\.md)$/);
  if (dimMatch) {
    return `https://github.com/AlanFokCo/ai-infra-primer/blob/main/${dimMatch[1]}/${dimMatch[2]}`;
  }
  // 章节文件链接：bare 同目录（02-x.md）或 ../NN-dir/MM-y.md
  const chapMatch = target.match(/^(\.\.\/)?((\d{2}-[\da-z-]+)\/)?(\d{2}-[\da-z-]+)\.md(#.*)?$/);
  if (chapMatch) {
    // 本站路由：/models/<partDir>/<file>/（无 partDir 前缀时无法定位，保持原样）
    if (!chapMatch[3]) return target;
    return `${base}/models/${chapMatch[3]}/${chapMatch[4]}/${chapMatch[5] ?? ''}`;
  }
  return target;
}
