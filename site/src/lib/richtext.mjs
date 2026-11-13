// 富文本小件：frontmatter/heading 等纯文本字段 → 安全 HTML
// 顺序铁律：先按 $...$ 分段（LaTeX 原样喂 KaTeX），纯文本段才做 HTML 转义。
import katex from 'katex';

export function richText(text) {
  if (!text) return '';
  return text
    .split(/(\$[^$]+\$)/g)
    .map((seg, i) => {
      if (i % 2 === 1) {
        const tex = seg.slice(1, -1);
        try {
          return katex.renderToString(tex, { throwOnError: true });
        } catch (e) {
          throw new Error(`[richtext] KaTeX 失败：「${tex}」 in 「${text.slice(0, 100)}…」 — ${e.message}`);
        }
      }
      return seg.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    })
    .join('');
}

export function richWithBadges(text) {
  return richText(text).replace(
    /\[([ABCW])\]/g,
    (_, g) => `<span class="badge badge-${g.toLowerCase()}" title="证据等级 ${g}">${g}</span>`
  );
}
