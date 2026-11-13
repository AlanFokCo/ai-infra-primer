import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { transformMath, countBacktickMath } from '../scripts/lib/math-transform.mjs';

const REPO = fileURLToPath(new URL('../../', import.meta.url));
const chapter = (p) => readFileSync(new URL(`../../models/${p}`, import.meta.url), 'utf8');

test('$`...`$ 回映为 $...$', () => {
  const src = '按公比 $`r = e^{-2.8} \\approx 6\\%`$ 几何衰减';
  const out = transformMath(src);
  assert.equal(out, '按公比 $r = e^{-2.8} \\approx 6\\%$ 几何衰减');
  assert.ok(!out.includes('$`'));
});

test('\\thinspace / \\negthinspace / \\hspace{5mu} 回映', () => {
  const src = '$\\mathrm{softmax}(QK^{\\top}/\\sqrt{d_k})\\thinspace V$ 与 $\\mathrm{arg\\thinspace max}_{s} \\hspace{5mu} \\langle q, k_s \\rangle$ 与 $a \\negthinspace b$';
  const out = transformMath(src);
  assert.ok(!out.includes('\\thinspace'));
  assert.ok(!out.includes('\\hspace{5mu}'));
  assert.ok(out.includes('\\, V'));
  assert.ok(out.includes('\\; \\langle'));
  assert.ok(out.includes('\\! b'));
});

test('代码围栏内的 $`...`$ 与 \\thinspace 不动', () => {
  const src = '前文 $`x_i`$ 转换\n```\n$`y_j`$ 与 \\thinspace 都在代码里\n```\n正文 \\thinspace 转';
  const out = transformMath(src);
  const lines = out.split('\n');
  assert.equal(lines[0], '前文 $x_i$ 转换');
  assert.equal(lines[2], '$`y_j`$ 与 \\thinspace 都在代码里');
  assert.equal(lines[4], '正文 \\, 转');
});

test('行内 code span 内的写法不动', () => {
  const src = '写 `\\thinspace` 是源格式，而 $\\alpha\\thinspace\\beta$ 要转';
  const out = transformMath(src);
  assert.ok(out.includes('`\\thinspace`'));
  assert.ok(out.includes('$\\alpha\\,\\beta$'));
});

test('真实语料：转换后无 $` 残留、无 \\thinspace、计数守恒', () => {
  const files = [
    '02-transformer-anatomy/03-attention-mechanism.md',
    '07-diffusion-image-generation/02-ddpm.md',
    '10-audio-speech/02-asr.md',
  ];
  for (const f of files) {
    const src = chapter(f);
    const out = transformMath(src);
    assert.ok(countBacktickMath(out) === 0, `${f}: 残留 $'`);
    assert.ok(!/\\thinspace\b/.test(out), `${f}: 残留 \\thinspace`);
    assert.ok(!/\\negthinspace\b/.test(out), `${f}: 残留 \\negthinspace`);
    assert.ok(!/\\hspace\{[45]mu\}/.test(out), `${f}: 残留 \\hspace`);
  }
});
