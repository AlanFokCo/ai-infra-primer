// 降级动效判定工具（配合 fig-review.mjs 的截图产出）：
// 1) 对每张注册图比较 reduced 两帧（t3000 vs t4500）像素差 → qa/fig-review/reduced-diff.json
//    口径：像素 = max(R,G,B) 通道绝对差 > 16 记为变色（>200px 判「仍在动」，与基线评审同口径）；
//    mean = 全图像素 max 通道差均值（0-255 制）。
// 2) DOM 级降级核验：reduced 上下文里 sig-rotor 的 computed animation-name 应为 none、
//    sig-flow / dia-pulse 应 display:none（SMIL 无法被 CSS 停止，只能隐藏）。
// 3) pulse 配速探针：kit edge 脉冲在直线通道上采样 t=10%/50% 处位移，
//    验证 calcMode=spline 缓动生效（10% 处进度应远小于线性 0.10）。
// 用法：node scripts/fig-reduced-diff.mjs   （需先跑过 scripts/fig-review.mjs）
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';
import { PULSE_GRAD_DEFS, scene, edge } from './lib/figure-kit.mjs';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const OUT = path.join(SITE, 'qa', 'fig-review');
const idx = JSON.parse(readFileSync(path.join(OUT, 'index.json'), 'utf8'));
const pad = (i) => String(i).padStart(3, '0');

const browser = await chromium.launch({ channel: 'chrome' });

/* ---- 1) reduced 两帧像素差 ---- */
{
  const page = await (await browser.newContext()).newPage();
  await page.goto('about:blank');
  const results = [];
  for (const f of idx.figs) {
    const a = readFileSync(path.join(OUT, `f${pad(f.i)}-reduced-t3000.png`)).toString('base64');
    const b = readFileSync(path.join(OUT, `f${pad(f.i)}-reduced-t4500.png`)).toString('base64');
    const r = await page.evaluate(async ({ a, b }) => {
      const load = (d) =>
        new Promise((res, rej) => {
          const img = new Image();
          img.onload = () => res(img);
          img.onerror = rej;
          img.src = 'data:image/png;base64,' + d;
        });
      const [ia, ib] = await Promise.all([load(a), load(b)]);
      const w = Math.min(ia.width, ib.width);
      const h = Math.min(ia.height, ib.height);
      const c = document.createElement('canvas');
      c.width = w;
      c.height = h;
      const ctx = c.getContext('2d');
      ctx.drawImage(ia, 0, 0);
      const da = ctx.getImageData(0, 0, w, h).data;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(ib, 0, 0);
      const db = ctx.getImageData(0, 0, w, h).data;
      let px = 0;
      let sum = 0;
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

/* ---- 2) DOM 级降级核验 ---- */
{
  const ctx = await browser.newContext({ viewport: { width: 820, height: 1000 }, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  await page.goto('file://' + path.join(OUT, 'index.html'));
  await page.waitForTimeout(800);
  const dom = await page.evaluate(() => {
    const q = (sel) => document.querySelector(sel);
    const rotor = q('.fig.in .sig-rotor');
    const dash = q('.fig.in .sig-dashflow');
    const flow = q('.sig-flow');
    const pulse = q('.dia-pulse');
    return {
      mediaReduce: matchMedia('(prefers-reduced-motion: reduce)').matches,
      rotorAnimationName: rotor ? getComputedStyle(rotor).animationName : '(no rotor)',
      dashAnimationName: dash ? getComputedStyle(dash).animationName : '(no dashflow)',
      sigFlowDisplay: flow ? getComputedStyle(flow).display : '(no sig-flow)',
      diaPulseDisplay: pulse ? getComputedStyle(pulse).display : '(no dia-pulse)',
    };
  });
  console.log('[reduced-dom]', JSON.stringify(dom));
  await ctx.close();
}

/* ---- 3) pulse 配速探针（正常动效模式） ---- */
{
  const sc = scene(720, 120, [
    `<path id="probe-path" d="M 40 60 H 680" fill="none" stroke="#ccc"/>`,
    edge([40, 60], [680, 60], { pulse: true, dur: 4, delay: 0, arrow: false, uid: 'probe', eid: 0 }),
  ]);
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0}</style></head><body>${PULSE_GRAD_DEFS}${sc.svg()}</body></html>`;
  const ctx = await browser.newContext({ viewport: { width: 760, height: 160 } });
  const page = await ctx.newPage();
  await page.goto('about:blank');
  await page.setContent(html);
  const dot = page.locator('.dia-pulse').first();
  const sample = async () => {
    const b = await dot.boundingBox();
    const p = b ? (b.x + b.width / 2 - 40) / 640 : null; // 40→680 通道
    return p === null ? null : +p.toFixed(3);
  };
  // SMIL begin 从 setContent 渲染起算；首采样对齐 t≈10%（400ms/4s）
  await page.waitForTimeout(400);
  const at10 = await sample();
  await page.waitForTimeout(1600); // t≈50%
  const at50 = await sample();
  console.log(`[pulse-pace] 直线通道位移进度：t=10% → ${at10}（线性应为 0.10），t=50% → ${at50}（应为 ≈0.50）`);
  await ctx.close();
}

await browser.close();
