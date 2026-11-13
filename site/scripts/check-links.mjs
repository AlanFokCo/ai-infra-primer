// dist 链接完整性：爬全部 html，站内 href/src 必须有对应文件；base 路径必须正确。
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const DIST = path.join(SITE, 'dist');
const BASE = '/ai-infra-primer';

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (name.endsWith('.html')) yield p;
  }
}

let pages = 0;
let checked = 0;
const missing = [];

for (const file of walk(DIST)) {
  pages += 1;
  const html = readFileSync(file, 'utf8')
    .replace(/<script[\s\S]*?<\/script>/g, '') // 内联脚本里的模板字符串不是链接
    .replace(/<style[\s\S]*?<\/style>/g, '');
  const attrs = html.matchAll(/(?:href|src)="([^"]+)"/g);
  for (const [, url] of attrs) {
    if (/^(https?:|mailto:|#|data:|javascript:)/.test(url)) continue;
    checked += 1;
    const clean = url.split('#')[0].split('?')[0];
    if (!clean) continue;
    if (clean.startsWith(BASE + '/')) {
      // 站内绝对路径 → dist 相对
      const rel = clean.slice(BASE.length + 1);
      const target = path.join(DIST, rel);
      const ok =
        existsSync(target) ||
        existsSync(path.join(target, 'index.html')) ||
        existsSync(target + '.html');
      if (!ok) missing.push(`${path.relative(DIST, file)} → ${url}`);
    } else if (clean.startsWith('/')) {
      missing.push(`${path.relative(DIST, file)} → ${url}（站内绝对路径未带 base）`);
    } else {
      // 相对路径
      const target = path.resolve(path.dirname(file), clean);
      const ok =
        existsSync(target) ||
        existsSync(path.join(target, 'index.html')) ||
        existsSync(target + '.html');
      if (!ok) missing.push(`${path.relative(DIST, file)} → ${url}`);
    }
  }
}

console.log(`[check-links] ${pages} 页 / ${checked} 个站内引用`);
if (missing.length > 0) {
  console.error(`[check-links] ${missing.length} 个断链:`);
  for (const m of missing.slice(0, 50)) console.error('  ' + m);
  process.exit(1);
}
console.log('[check-links] OK');
