// 签名图自检工具：渲染某 figures 模块的全部图为预览页并逐张截图
// 用法：node scripts/fig-check.mjs <模块文件名不含扩展名，如 part01>
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const name = process.argv[2];
if (!name) {
  console.error('用法: node scripts/fig-check.mjs <partNN>');
  process.exit(1);
}
const mod = await import(`./lib/figures/${name}.mjs`);
const FIGURES = mod.FIGURES;
if (!Array.isArray(FIGURES) || FIGURES.length === 0) {
  console.error(`模块 ${name} 未导出非空 FIGURES 数组`);
  process.exit(1);
}

const tokens = readFileSync(path.join(SITE, 'src/styles/tokens.css'), 'utf8');
const figcss = readFileSync(path.join(SITE, 'src/styles/figure.css'), 'utf8');

const rows = FIGURES.map(
  (f, i) =>
    `<h3>[${i}] ${f.slug} — ${f.title} — 锚点 ${f.afterHeading}</h3>` +
    `<figure class="fig fig-sig in"><span class="fig-tag">${f.title}</span><div class="fig-body">${f.svg()}</div></figure>`
);
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>${tokens}\n${figcss}\nbody{background:var(--paper);padding:32px;max-width:760px;margin:0 auto}h3{font:12px monospace;color:#888;margin:36px 0 8px}</style></head><body>${rows.join('\n')}</body></html>`;

const outDir = path.join(SITE, 'qa', `figs-${name}`);
mkdirSync(outDir, { recursive: true });
const htmlPath = path.join(outDir, 'index.html');
writeFileSync(htmlPath, html);

const browser = await chromium.launch({ channel: 'chrome' });
const page = await (await browser.newContext({ viewport: { width: 820, height: 1000 }, deviceScaleFactor: 2 })).newPage();
await page.goto('file://' + htmlPath);
await page.waitForTimeout(3500); // 等动画完成/进入稳态
const figs = page.locator('.fig');
const n = await figs.count();
for (let i = 0; i < n; i++) {
  await figs.nth(i).screenshot({ path: path.join(outDir, `fig-${i}.png`) });
}
await browser.close();
console.log(`[fig-check] ${n} 张图 → ${outDir}/fig-*.png`);
