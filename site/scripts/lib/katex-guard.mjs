// KaTeX 渲染守卫：rehype-katex 以 throwOnError:false 运行以便一次构建收集全部错误，
// 本模块的 rehype 插件逐文件统计 .katex-error 节点，astro:build:done 钩子在发现任何
// 错误时 throw —— 公式渲染失败的站点绝不允许构建成功。
import { visit } from 'unist-util-visit';

const errors = [];

// KaTeX 失败签名有两类：硬解析失败给 class="katex-error"；throwOnError:false 下
// 未知命令渲染为红字（MathML <mstyle mathcolor="#cc0000"> / HTML span style color:#cc0000）。
// 白名单禁用 \color，语料中不存在合法 #cc0000，红色即错误。
const ERROR_COLOR = '#cc0000';

export function katexGuard() {
  return (tree, file) => {
    let count = 0;
    let firstMessage = '';
    visit(tree, 'element', (node) => {
      const cls = node.properties?.className;
      const isHardError = Array.isArray(cls) && cls.includes('katex-error');
      const isRedText =
        (node.tagName === 'mstyle' && node.properties?.mathcolor === ERROR_COLOR) ||
        (typeof node.properties?.style === 'string' &&
          node.properties.style.includes(ERROR_COLOR));
      if (isHardError || isRedText) {
        count += 1;
        if (!firstMessage) {
          firstMessage =
            (node.properties?.title || '') +
            ' :: ' +
            extractText(node).slice(0, 120);
        }
      }
      if (isRedText) node.properties.dataKatexError = true; // 供下游排查
    });
    if (count > 0) {
      const path = file?.history?.[0] || file?.path || '(unknown)';
      errors.push({ path, count, firstMessage });
      console.error(`[katex-guard] ${path}: ${count} 个公式渲染失败 — ${firstMessage}`);
    }
  };
}

function extractText(node) {
  if (node.type === 'text') return node.value || '';
  return (node.children || []).map(extractText).join('');
}

export function katexGuardIntegration() {
  return {
    name: 'katex-guard',
    hooks: {
      'astro:build:done': () => {
        if (errors.length > 0) {
          const total = errors.reduce((n, e) => n + e.count, 0);
          const detail = errors.map((e) => `  ${e.path}: ${e.count}`).join('\n');
          throw new Error(
            `[katex-guard] 构建中止：${errors.length} 个文件共 ${total} 个公式渲染失败。\n${detail}`
          );
        }
      },
    },
  };
}
