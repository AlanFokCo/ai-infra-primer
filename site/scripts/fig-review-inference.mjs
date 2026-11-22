// 图品质评审工具 · inference 维度（只读评审，不改图代码）：
// 与 fig-review-training.mjs 同口径，过滤 dim === 'inference'，输出 qa/fig-review-inference/。
// 截图（早帧/稳态/reduced-motion）+ 文本出 viewBox 度量 + reduced 两帧像素差 + reduced DOM 核验。
import { mkdirSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';
import { PULSE_GRAD_DEFS } from './lib/figure-kit.mjs';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const OUT = path.join(SITE, 'qa', 'fig-review-inference');
mkdirSync(OUT, { recursive: true });

const figuresDir = path.join(SITE, 'scripts/lib/figures');
const files = readdirSync(figuresDir).filter((f) => f.endsWith('.mjs'));
const all = [];
for (const f of files) {
  const mod = await import(`./lib/figures/${f}`);
  if (Array.isArray(mod.FIGURES)) for (const fig of mod.FIGURES) all.push({ ...fig, _src: `figures/${f}` });
}
const registry = new Map();
for (const fig of all) {
  if ((fig.dim ?? 'models') !== 'inference') continue;
  const key =
    typeof fig.replacesBlock === 'number'
      ? `${fig.slug}#block${fig.replacesBlock}`
      : `${fig.slug}@after`;
  registry.set(key, fig);
}
const figs = [...registry.entries()].map(([key, f], i) => ({
  i, key, slug: f.slug, title: f.title,
  kind: typeof f.replacesBlock === 'number' ? `block${f.replacesBlock}` : 'afterHeading',
  src: f._src, svgFn: f.svg,
}));
console.log(`[fig-review-inference] 注册图 ${figs.length} 张`);

const tokens = readFileSync(path.join(SITE, 'src/styles/tokens.css'), 'utf8');
const figcss = readFileSync(path.join(SITE, 'src/styles/figure.css'), 'utf8');
const rows = figs.map(
  (f) =>
    `<h3 id="h-${f.i}">[${f.i}] ${f.key} — ${f.title}（${f.src}）</h3>` +
    `<figure class="fig fig-sig in" data-i="${f.i}"><span class="fig-tag">${f.title}</span><div class="fig-body">${f.svgFn()}</div></figure>`
);
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>${tokens}\n${figcss}\nbody{background:var(--paper);padding:32px;max-width:780px;margin:0 auto}h3{font:12px monospace;color:#999;margin:40px 0 8px;word-break:break-all}</style></head><body>${PULSE_GRAD_DEFS}${rows.join('\n')}</body></html>`;
writeFileSync(path.join(OUT, 'index.html'), html);

const metrics = figs.map((f) => {
  const svg = f.svgFn();
  const vb = /viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/.exec(svg);
  const count = (re) => (svg.match(re) || []).length;
  return {
    i: f.i, key: f.key, title: f.title, src: f.src,
    viewBox: vb ? [Number(vb[1]), Number(vb[2])] : null,
    smilMotion: count(/<animateMotion/g),
    pulses: count(/class="dia-pulse"/g),
    sigCell: count(/sig-cell[ "-]/g),
    sigBar: count(/sig-bar[ "-]/g),
    sigDiag: count(/sig-diag/g),
    sigDashflow: count(/sig-dashflow/g),
    textEls: count(/<text/g),
  };
});

const browser = await chromium.launch({ channel: 'chrome' });

async function shootAll(tag, { times }) {
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
      await loc.nth(i).screenshot({ path: path.join(OUT, `t${String(i).padStart(3, '0')}-${tag}-t${t}.png`) });
    }
  }
  await ctx.close();
}

await shootAll('norm', { times: [600, 3500, 6000] });
await shootAll('reduced', { reduced: true, times: [3000, 4500] });

// 渲染态度量：文本出 viewBox
{
  const ctx = await browser.newContext({ viewport: { width: 820, height: 1000 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto('file://' + path.join(OUT, 'index.html'));
  await page.waitForTimeout(3500);
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
      out.push({ i, vbW: vb.width, vbH: vb.height, textOverflow: overs });
    });
    return out;
  });
  for (const r of render) Object.assign(metrics[r.i], r);
  await ctx.close();
}

// reduced 两帧像素差（>200px 判仍在动）
{
  const page = await (await browser.newContext()).newPage();
  await page.goto('about:blank');
  const pad = (i) => String(i).padStart(3, '0');
  const results = [];
  for (const f of figs) {
    const a = readFileSync(path.join(OUT, `t${pad(f.i)}-reduced-t3000.png`)).toString('base64');
    const b = readFileSync(path.join(OUT, `t${pad(f.i)}-reduced-t4500.png`)).toString('base64');
    const r = await page.evaluate(async ({ a, b }) => {
      const load = (d) => new Promise((res, rej) => { const img = new Image(); img.onload = () => res(img); img.onerror = rej; img.src = 'data:image/png;base64,' + d; });
      const [ia, ib] = await Promise.all([load(a), load(b)]);
      const w = Math.min(ia.width, ib.width), h = Math.min(ia.height, ib.height);
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      const ctx = c.getContext('2d');
      ctx.drawImage(ia, 0, 0);
      const da = ctx.getImageData(0, 0, w, h).data;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(ib, 0, 0);
      const db = ctx.getImageData(0, 0, w, h).data;
      let px = 0, sum = 0;
      for (let p = 0; p < da.length; p += 4) {
        const d = Math.max(Math.abs(da[p] - db[p]), Math.abs(da[p + 1] - db[p + 1]), Math.abs(da[p + 2] - db[p + 2]));
        sum += d;
        if (d > 16) px++;
      }
      return { px, mean: +(sum / (da.length / 4)).toFixed(3) };
    }, { a, b });
    results.push({ i: f.i, px: r.px, mean: r.mean });
  }
  writeFileSync(path.join(OUT, 'reduced-diff.json'), JSON.stringify(results));
  const bad = results.filter((r) => r.px > 200);
  console.log(`[reduced-diff] ${results.length} 张；>200px 仍动：${bad.length} 张${bad.length ? ' → ' + bad.map((r) => `#${r.i}@${r.px}px`).join(', ') : ''}`);
  await page.context().close();
}

// reduced DOM 级核验
{
  const ctx = await browser.newContext({ viewport: { width: 820, height: 1000 }, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  await page.goto('file://' + path.join(OUT, 'index.html'));
  await page.waitForTimeout(800);
  const dom = await page.evaluate(() => {
    const q = (sel) => document.querySelector(sel);
    const dash = q('.fig.in .sig-dashflow');
    const cell = q('.fig.in .sig-cell');
    const pulse = q('.dia-pulse');
    return {
      mediaReduce: matchMedia('(prefers-reduced-motion: reduce)').matches,
      dashAnimationName: dash ? getComputedStyle(dash).animationName : '(no dashflow)',
      cellAnimationName: cell ? getComputedStyle(cell).animationName : '(no sig-cell)',
      cellOpacity: cell ? getComputedStyle(cell).opacity : '(no sig-cell)',
      diaPulseDisplay: pulse ? getComputedStyle(pulse).display : '(no dia-pulse)',
    };
  });
  console.log('[reduced-dom]', JSON.stringify(dom));
  await ctx.close();
}

await browser.close();
for (const f of figs) delete f.svgFn;
writeFileSync(path.join(OUT, 'index.json'), JSON.stringify({ figs, metrics }, null, 2));
const over = metrics.filter((m) => m.textOverflow && m.textOverflow.length);
console.log(`[overflow] ${over.length} 张出框${over.length ? ' → ' + over.map((m) => `#${m.i}: ${m.textOverflow.join(' | ')}`).join(' ;; ') : ''}`);
console.log(`[fig-review-inference] 截图与度量 → ${OUT}/`);
