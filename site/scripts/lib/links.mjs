// 站内链接重写：把仓库相对路径映射到站点路由。
// 语料实测：章节内只出现 ../references.md（1229 处）与外部 http 链接；
// bare 章节文件链接当前为零，但保留处理能力（防御性）。
// 跨维度约定（training spec §4.2）：章间引用一律纯文字不挂链接，故此处无跨维度章链规则。
// 铁律：代码围栏与行内 code span 内的内容不动。

/**
 * @param {string} md markdown 原文
 * @param {{ base: string, partDir: string, dim: string }} ctx base 形如 '/ai-infra-primer'，dim 为维度 key
 */
export function rewriteLinks(md, ctx) {
  const { base, dim } = ctx;
  const lines = md.split('\n');
  let inFence = false;
  const out = [];
  for (const line of lines) {
    if (/^```/.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    out.push(inFence ? line : rewriteLine(line, base, dim));
  }
  return out.join('\n');
}

function rewriteLine(line, base, dim) {
  return line
    .split(/(`[^`\n]+`)/)
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // code span 不动
      return seg.replace(
        /\]\(([^)\s]+)\)/g,
        (whole, target) => `](${rewriteTarget(target, base, dim)})`
      );
    })
    .join('');
}

function rewriteTarget(target, base, dim) {
  if (/^https?:\/\//.test(target) || target.startsWith('#') || target.startsWith('mailto:')) {
    return target;
  }
  // ../references.md 或 ../references.md#anchor（章深度下解析到本维度 references）
  const refMatch = target.match(/^\.\.\/references\.md(#.*)?$/);
  if (refMatch) return `${base}/${dim}/references/${refMatch[1] ?? ''}`;
  // 跨维度链接（../../inference/README.md 等）：README 级链接，指向 GitHub 仓库源
  const dimMatch = target.match(/^\.\.\/\.\.\/([\da-z-]+)\/(README\.md)$/);
  if (dimMatch) {
    return `https://github.com/AlanFokCo/ai-infra-primer/blob/main/${dimMatch[1]}/${dimMatch[2]}`;
  }
  // 章节文件链接：bare 同目录（02-x.md）或 ../NN-dir/MM-y.md（同维度内）
  const chapMatch = target.match(/^(\.\.\/)?((\d{2}-[\da-z-]+)\/)?(\d{2}-[\da-z-]+)\.md(#.*)?$/);
  if (chapMatch) {
    // 本站路由：/<dim>/<partDir>/<file>/（无 partDir 前缀时无法定位，保持原样）
    if (!chapMatch[3]) return target;
    return `${base}/${dim}/${chapMatch[3]}/${chapMatch[4]}/${chapMatch[5] ?? ''}`;
  }
  return target;
}
