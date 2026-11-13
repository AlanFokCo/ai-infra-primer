// references.md 解析：「## 条目」下按 ### Part N 分组的七字段表 → 结构化 JSON。
// 行格式：| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |

/**
 * @param {string} source references.md 原文
 * @returns {{ groups: [{ part: string, title: string, entries: RefEntry[] }], glossary: string }}
 * @typedef {{ key: string, title: string, authors: string, idOrUrl: string, level: string, verified: string, citedBy: string }} RefEntry
 */
export function parseReferences(source) {
  const lines = source.split('\n');
  const groups = [];
  let current = null;
  let inEntries = false;
  let glossary = '';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^## 条目/.test(line)) {
      inEntries = true;
      continue;
    }
    if (/^## 术语表/.test(line)) {
      inEntries = false;
      glossary = lines.slice(i + 1).join('\n').trim();
      break;
    }
    if (!inEntries) continue;

    const h3 = line.match(/^###\s+(Part\s+\d+)\s*·\s*(.+)$/);
    if (h3) {
      current = { part: h3[1], title: h3[2].trim(), entries: [] };
      groups.push(current);
      continue;
    }
    if (!current) continue;
    if (!line.startsWith('|')) continue;
    const cells = line.split('|').slice(1, -1).map((c) => c.trim());
    if (cells.length !== 7) continue;
    if (cells[0] === 'key' || /^-{3,}$/.test(cells[0])) continue; // 表头/分隔行
    const [key, title, authors, idOrUrl, level, verified, citedBy] = cells;
    current.entries.push({ key, title, authors, idOrUrl, level, verified, citedBy });
  }
  return { groups, glossary };
}
