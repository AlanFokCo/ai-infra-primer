// 给 h2/h3 写确定性 id：sec-0..N，与 build-content 抽取 TOC 的文档序一致。
// （Astro 默认 slugger 在数学标题上会产生 mangled 文本与丑陋 id，统一接管。）
import { visit } from 'unist-util-visit';

export function rehypeHeadingIds() {
  return (tree) => {
    let n = 0;
    visit(tree, 'element', (node) => {
      if (node.tagName === 'h2' || node.tagName === 'h3') {
        node.properties = node.properties || {};
        node.properties.id = `sec-${n}`;
        n += 1;
      }
      // KaTeX 的 MathML 层（含 LaTeX 源码 annotation）不进搜索索引
      const cls = node.properties?.className;
      if (Array.isArray(cls) && cls.includes('katex-mathml')) {
        node.properties['data-pagefind-ignore'] = true;
      }
    });
  };
}
