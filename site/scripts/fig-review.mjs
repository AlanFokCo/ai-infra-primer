// 图品质评审工具（只读评审，不改图代码）：
// 汇总全部注册图（与 build-content.mjs 同序同去重），渲染成独立预览页，
// Playwright 逐张截图（早帧/稳态/reduced-motion）+ 结构度量（出框文本/动画计数），
// 产出 qa/fig-review/index.json 供评分引用。
import { mkdirSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';
import { PULSE_GRAD_DEFS } from './lib/figure-kit.mjs';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const OUT = path.join(SITE, 'qa', 'fig-review');
mkdirSync(OUT, { recursive: true });

// ---- 注册表（与 build-content.mjs 同序：CORE_FIVE → figures/*.mjs 字母序，后者覆盖前者） ----
const sig = await import('./lib/figures-signature.mjs');
const figuresDir = path.join(SITE, 'scripts/lib/figures');
const files = readdirSync(figuresDir).filter((f) => f.endsWith('.mjs'));
const all = sig.SIGNATURE_FIGURES.map((f) => ({ ...f, _src: 'figures-signature.mjs' }));
for (const f of files) {
  const mod = await import(`./lib/figures/${f}`);
  if (Array.isArray(mod.FIGURES)) for (const fig of mod.FIGURES) all.push({ ...fig, _src: `figures/${f}` });
}
const registry = new Map();
for (const fig of all) {
  const dim = fig.dim ?? 'models';
  if (dim !== 'models') continue;
  const key =
    typeof fig.replacesBlock === 'number'
      ? `${fig.slug}#block${fig.replacesBlock}`
      : `${fig.slug}@after`;
  registry.set(key, fig); // 同 key 后者胜（afterHeading 一 slug 至多一张）
}
const figs = [...registry.entries()].map(([key, f], i) => ({
  i,
  key,
  slug: f.slug,
  title: f.title,
  kind: typeof f.replacesBlock === 'number' ? `block${f.replacesBlock}` : 'afterHeading',
  src: f._src,
  svgFn: f.svg,
}));
console.log(`[fig-review] 注册图 ${figs.length} 张`);

// ---- 预览页：真实 tokens.css + figure.css，.in 常亮（等价入视口） ----
const tokens = readFileSync(path.join(SITE, 'src/styles/tokens.css'), 'utf8');
const figcss = readFileSync(path.join(SITE, 'src/styles/figure.css'), 'utf8');
const rows = figs.map(
  (f) =>
    `<h3 id="h-${f.i}">[${f.i}] ${f.key} — ${f.title}（${f.src}）</h3>` +
    `<figure class="fig fig-sig in" data-i="${f.i}"><span class="fig-tag">${f.title}</span><div class="fig-body">${f.svgFn()}</div></figure>`
);
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>${tokens}\n${figcss}\nbody{background:var(--paper);padding:32px;max-width:780px;margin:0 auto}h3{font:12px monospace;color:#999;margin:40px 0 8px;word-break:break-all}</style></head><body>${PULSE_GRAD_DEFS}${rows.join('\n')}</body></html>`;
writeFileSync(path.join(OUT, 'index.html'), html);

// ---- 结构度量（解析态，无需渲染） ----
const metrics = figs.map((f) => {
  const svg = f.svgFn();
  const vb = /viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/.exec(svg);
  const count = (re) => (svg.match(re) || []).length;
  return {
    i: f.i,
    key: f.key,
    title: f.title,
    src: f.src,
    viewBox: vb ? [Number(vb[1]), Number(vb[2])] : null,
    smilMotion: count(/<animateMotion/g),
    smilAnimate: count(/<animate(?!Motion)/g),
    pulses: count(/class="dia-pulse"/g),
    sigCell: count(/sig-cell[ "-]/g),
    sigBar: count(/sig-bar[ "-]/g),
    sigRotor: count(/sig-rotor/g),
    sigDiag: count(/sig-diag/g),
    sigStep: count(/sig-step/g),
    sigDashflow: count(/sig-dashflow/g),
    textEls: count(/<text/g),
    nodes: count(/dia-node[ "]/g) / 2, // class 出现在 g 与 css 选择各一次→粗估
  };
});

// ---- 截图 ----
const browser = await chromium.launch({ channel: 'chrome' });

async function shootAll(tag, { reduced = false, times }) {
  const ctx = await browser.newContext({
    viewport: { width: 820, height: 1000 },
    deviceScaleFactor: 2,
    reducedMotion: tag === 'reduced' ? 'reduce' : 'no-preference',
  });
  const page = await ctx.newPage();
  await page.goto('file://' + path.join(OUT, 'index.html'));
  const loc = page.locator('.fig');
  let prev = 0;
  for (const t of times) {
    await page.waitForTimeout(t - prev);
    prev = t;
    for (let i = 0; i < figs.length; i++) {
      await loc.nth(i).screenshot({ path: path.join(OUT, `f${String(i).padStart(3, '0')}-${tag}-t${t}.png`) });
    }
  }
  await ctx.close();
}

await shootAll('norm', { times: [600, 6000] });
await shootAll('reduced', { reduced: true, times: [3000, 4500] });

// ---- 渲染态度量：文本出 viewBox / 元素溢出 ----
{
  const ctx = await browser.newContext({ viewport: { width: 820, height: 1000 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto('file://' + path.join(OUT, 'index.html'));
  await page.waitForTimeout(6500);
  const render = await page.evaluate(() => {
    const out = [];
    document.querySelectorAll('.fig').forEach((fig) => {
      const i = Number(fig.dataset.i);
      const svg = fig.querySelector('svg');
      const vb = svg.viewBox.baseVal;
      const overs = [];
      svg.querySelectorAll('text').forEach((t) => {
        const b = t.getBBox();
        if (b.x < -2 || b.y < -2 || b.x + b.width > vb.width + 2 || b.y + b.height > vb.height + 2) {
          overs.push(`${t.textContent.slice(0, 24)}@(${Math.round(b.x)},${Math.round(b.y)},${Math.round(b.width)}x${Math.round(b.height)})`);
        }
      });
      const fr = fig.getBoundingClientRect();
      const sr = svg.getBoundingClientRect();
      out.push({ i, vbW: vb.width, vbH: vb.height, textOverflow: overs, renderW: Math.round(sr.width), figW: Math.round(fr.width) });
    });
    return out;
  });
  for (const r of render) Object.assign(metrics[r.i], r);
  await ctx.close();
}

await browser.close();

// 去掉内部的 svgFn（不可序列化）
for (const f of figs) delete f.svgFn;
writeFileSync(path.join(OUT, 'index.json'), JSON.stringify({ figs, metrics }, null, 2));
console.log(`[fig-review] 截图与度量 → ${OUT}/`);
