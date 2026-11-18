// P7-P9 场景图实拍抽查：在真实章节页里截各注册图（fig-sig 容器），Chrome 通道
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const OUT = path.join(SITE, 'qa', 'figs-p789-insitu');
mkdirSync(OUT, { recursive: true });

const BASE = 'http://localhost:4331/ai-infra-primer/training';
const PAGES = [
  ['p7-data-pipeline', '/07-pretraining/01-data-pipeline/'],
  ['p7-checkpoint', '/07-pretraining/04-checkpointing-and-resumption/'],
  ['p8-ppo', '/08-rl-math/03-ppo/'],
  ['p8-grpo', '/08-rl-math/04-grpo-and-variants/'],
  ['p8-systems', '/08-rl-math/05-llm-rl-systems/'],
  ['p9-rlhf', '/09-post-training/02-rlhf-pipeline/'],
  ['p9-dpo', '/09-post-training/03-dpo-family/'],
  ['p9-r1', '/09-post-training/05-rlvr-and-reasoning/'],
];

const browser = await chromium.launch({ channel: 'chrome' });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1.5 });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

for (const [name, route] of PAGES) {
  const resp = await page.goto(BASE + route, { waitUntil: 'networkidle' });
  if (!resp || resp.status() !== 200) { console.error(`[spot] ${name} HTTP ${resp?.status()}`); continue; }
  const figs = page.locator('.fig-sig');
  const n = await figs.count();
  for (let i = 0; i < n; i++) {
    await figs.nth(i).evaluate((el) => el.scrollIntoView({ block: 'center' }));
    await page.waitForTimeout(1400);
    await figs.nth(i).screenshot({ path: path.join(OUT, `${name}-fig${i}.png`) });
  }
  console.log(`[spot] ${name}: ${n} 张场景图`);
}
if (errors.length) { console.error('[spot] 页面错误:', errors.slice(0, 5)); process.exit(1); }
await browser.close();
console.log('[spot] 完成 →', OUT);
