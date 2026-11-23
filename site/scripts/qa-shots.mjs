// Playwright 截图套件：关键页面 × 桌面/移动 → qa/ 目录，供人眼审阅迭代
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const OUT = path.join(SITE, 'qa');
const BASE_URL = process.env.QA_URL || 'http://localhost:4321/ai-infra-primer';

const PAGES = [
  ['home', '/'],
  ['models', '/models/'],
  ['references', '/models/references/'],
  ['part-p2', '/models/02-transformer-anatomy/'],
  ['ch-attention', '/models/02-transformer-anatomy/03-attention-mechanism/'], // 公式密
  ['ch-moe', '/models/04-modern-architectures/05-moe/'], // 表格密
  ['ch-ddpm', '/models/07-diffusion-image-generation/02-ddpm/'], // 推导重章
  ['ch-assembly', '/models/02-transformer-anatomy/08-assembly/'], // 代码+ASCII
  // training 维度（W1 起；spec §8：公式密 / 表格密 / 推导重章各一）
  ['training', '/training/'],
  ['tr-references', '/training/references/'],
  ['tr-ch-mixed-precision', '/training/01-autograd-and-backprop/03-mixed-precision/'], // 公式密
  ['tr-ch-adamw', '/training/02-optimizers/02-adam-and-adamw/'], // 表格密（超参对照）
  ['tr-ch-muon', '/training/02-optimizers/05-muon-and-spectral/'], // 推导重章
  // inference 维度（W1 起；spec §8：公式密 / 表格图密 / 推导重章各一，取本波已落地代表章）
  ['inference', '/inference/'],
  ['inf-references', '/inference/references/'],
  ['inf-ch-prefill-decode', '/inference/01-principles-and-accounting/02-prefill-decode-accounting/'], // 公式密（1.2【重】）
  ['inf-ch-paged-attention', '/inference/02-kv-cache-systems/01-paged-attention/'], // 表格/图密（2.1【重】，块表场景图）
  ['inf-ch-kv-offload', '/inference/02-kv-cache-systems/04-kv-offload-and-tiering/'], // 推导重章（2.4【重】，分层命中成本账）
  // inference W2 样章（P3/P4 落地代表章：公式密 / 表格图密 / 推导重章各一）
  ['inf-ch-weight-quant', '/inference/03-inference-operators/02-weight-quantization/'], // 公式密（3.2【重】，量化格点 / GPTQ 补偿 / AWQ 等价）
  ['inf-ch-gemm-moe', '/inference/03-inference-operators/03-quantized-gemm-and-moe-kernels/'], // 表格/图密（3.3，W4A16 mainloop / MoE dispatch 场景图）
  ['inf-ch-speculative', '/inference/04-scheduling-and-batching/04-speculative-decoding/'], // 推导重章（4.4【重】，分布保持 / 加速比定理 / tree verify）
];

const VIEWPORTS = [
  ['desktop', { width: 1440, height: 900 }],
  ['mobile', { width: 390, height: 844 }],
];

mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome' });
let failures = 0;

for (const [vpName, viewport] of VIEWPORTS) {
  const ctx = await browser.newContext({
    viewport,
    deviceScaleFactor: vpName === 'mobile' ? 2 : 1.5,
    reducedMotion: 'reduce', // 截图取静态终态，动效存在性单独验证
  });
  const page = await ctx.newPage();
  const consoleErrors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });
  page.on('pageerror', (e) => consoleErrors.push(String(e)));

  for (const [name, route] of PAGES) {
    const url = BASE_URL + route;
    const resp = await page.goto(url, { waitUntil: 'networkidle' });
    if (!resp || resp.status() !== 200) {
      console.error(`[qa] ${name} ${url} → HTTP ${resp?.status()}`);
      failures += 1;
      continue;
    }
    await page.screenshot({ path: path.join(OUT, `${vpName}-${name}.png`), fullPage: false });
    // 章节页追加一张中部（公式区）与末尾（pager）
    if (name.startsWith('ch-')) {
      await page.evaluate(() => scrollTo(0, document.body.scrollHeight * 0.4));
      await page.waitForTimeout(200);
      await page.screenshot({ path: path.join(OUT, `${vpName}-${name}-mid.png`) });
      await page.evaluate(() => scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(200);
      await page.screenshot({ path: path.join(OUT, `${vpName}-${name}-end.png`) });
    }
    // 横向溢出检查：以 document.scrollWidth 为准；定位未被任何 overflow 祖先裁剪的肇事元素
    const overflow = await page.evaluate(() => {
      const doc = document.documentElement;
      const over = doc.scrollWidth - doc.clientWidth;
      if (over <= 1) return { over, culprits: [] };
      const clipped = (el) => {
        let cur = el.parentElement;
        while (cur) {
          const ox = getComputedStyle(cur).overflowX;
          if (ox !== 'visible') {
            const cr = cur.getBoundingClientRect();
            const er = el.getBoundingClientRect();
            if (er.right <= cr.right + 1) return true; // 被祖先裁住
          }
          cur = cur.parentElement;
        }
        return false;
      };
      const culprits = [];
      document.querySelectorAll('body *').forEach((el) => {
        if (el.closest('svg') && el.tagName !== 'svg') return; // SVG 内部坐标系不参评
        const r = el.getBoundingClientRect();
        if (r.right > doc.clientWidth + 1 && !clipped(el)) {
          culprits.push(
            `${el.tagName}.${String(el.className).split(' ')[0]} right=${Math.round(r.right)}`
          );
        }
      });
      return { over, culprits: culprits.slice(0, 5) };
    });
    if (overflow.over > 1) {
      console.error(`[qa] ${vpName}/${name} 横向溢出 ${overflow.over}px — ${overflow.culprits.join(' | ')}`);
      failures += 1;
    }
    // 章节页桌面端：正文与右栏 TOC 不得交叠
    if (name.startsWith('ch-') && vpName === 'desktop') {
      const overlap = await page.evaluate(() => {
        const art = document.querySelector('article.prose');
        const toc = document.querySelector('.toc-rail');
        const nav = document.querySelector('.part-nav');
        if (!art) return null;
        const a = art.getBoundingClientRect();
        const bad = [];
        for (const [label, el] of [['toc', toc], ['nav', nav]]) {
          if (!el) continue;
          const r = el.getBoundingClientRect();
          if (r.width > 0 && a.right > r.left + 1 && a.left < r.right - 1) bad.push(`${label}(相交 ${Math.round(a.right - r.left)}px)`);
        }
        return bad.length ? bad.join(' ') : null;
      });
      if (overlap) {
        console.error(`[qa] desktop/${name} 栏间交叠 — ${overlap}`);
        failures += 1;
      }
    }
    console.log(`[qa] ${vpName}/${name} ok`);
  }
  if (consoleErrors.length > 0) {
    console.error(`[qa] ${vpName} 控制台错误 ${consoleErrors.length} 条:`);
    consoleErrors.slice(0, 5).forEach((e) => console.error('  ' + e));
    failures += consoleErrors.length;
  }
  await ctx.close();
}

// OG 卡片：重新生成 public/og.png
{
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 630 } });
  const page = await ctx.newPage();
  const resp = await page.goto(`${BASE_URL}/og-card/`, { waitUntil: 'networkidle' });
  if (resp?.ok()) {
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(SITE, 'public', 'og.png') });
    console.log('[qa] og.png 已生成');
  }
  await ctx.close();
}

await browser.close();
console.log(failures === 0 ? '[qa] 全部通过' : `[qa] ${failures} 处问题`);
process.exit(failures === 0 ? 0 : 1);
