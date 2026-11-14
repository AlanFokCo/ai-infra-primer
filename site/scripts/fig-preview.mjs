// 图引擎预览：把样本块渲染成独立 HTML，供截图迭代
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { renderFigure, isDiagram } from './lib/ascii-figure.mjs';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/scripts$/, '');
const REPO = path.resolve(SITE, '..');

const SAMPLES = [
  'models/02-transformer-anatomy/03-attention-mechanism.md#0',
  'models/02-transformer-anatomy/08-assembly.md#0',
  'models/04-modern-architectures/01-kv-compression-and-exact-attention.md#1',
  'models/07-diffusion-image-generation/02-ddpm.md#0',
  'models/04-modern-architectures/05-moe.md#0',
  'models/03-objectives-and-variants/01-objectives-and-masks.md#1',
  'models/08-multimodal-llm/02-connectors-and-resolution.md#0',
  'models/10-audio-speech/03-tts.md#0',
];

function getBlock(file, idx) {
  const src = readFileSync(path.join(REPO, file), 'utf8');
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let m;
  const out = [];
  while ((m = re.exec(src))) if (m[1] === '') out.push(m[2].split('\n'));
  return out[idx];
}

const figures = SAMPLES.map((s) => {
  const [file, idx] = s.split('#');
  const block = getBlock(file, Number(idx));
  const dia = isDiagram(block);
  return `<h3>${s} ${dia ? '' : '（非结构图→保持代码板）'}</h3>` +
    (dia
      ? `<figure class="fig in"><div class="fig-body">${renderFigure(block)}</div></figure>`
      : `<pre style="background:#181b20;color:#e8e6df;padding:16px">${block.join('\n').replace(/</g, '&lt;')}</pre>`);
});

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
:root { --paper:#f7f6f2; --paper-raised:#fdfcf9; --paper-sunk:#efede6; --ink:#181b20; --ink-soft:#454b57; --ink-faint:#636a78; --rule:#d9d6cc; --rule-soft:#e7e5db; --seal:#a83228; --font-mono:'IBM Plex Mono',ui-monospace,monospace; --text-2xs:11px; --space-1:4px; --space-2:8px; --space-3:12px; }
body { background: var(--paper); padding: 40px; max-width: 780px; margin: 0 auto; font-family: sans-serif; }
h3 { font: 12px monospace; color: #888; margin: 40px 0 8px; }
</style>
<link rel="stylesheet" href="../src/styles/figure.css">
</head><body class="">${figures.join('\n')}</body></html>`;

writeFileSync(path.join(SITE, 'qa/fig-preview.html'), html);
console.log('written qa/fig-preview.html');
