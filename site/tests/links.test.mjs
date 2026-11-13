import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { rewriteLinks } from '../scripts/lib/links.mjs';

const chapter = (p) => readFileSync(new URL(`../../models/${p}`, import.meta.url), 'utf8');
const ctx = { base: '/ai-infra-primer', partDir: '02-transformer-anatomy' };

test('../references.md 重写到站内文献页', () => {
  const src = '（[Vaswani et al., 2017](../references.md)，arXiv:1706.03762）';
  const out = rewriteLinks(src, ctx);
  assert.ok(out.includes('](/ai-infra-primer/models/references/)'));
  assert.ok(!out.includes('../references.md'));
});

test('外部链接与锚点不动', () => {
  const src = '[code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) 与 [本地](#sec)';
  const out = rewriteLinks(src, ctx);
  assert.ok(out.includes('](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)'));
  assert.ok(out.includes('](#sec)'));
});

test('代码围栏内的类链接文本不动', () => {
  const src = '正文 [a](../references.md)\n```\n[b](../references.md)\n```';
  const out = rewriteLinks(src, ctx);
  const lines = out.split('\n');
  assert.ok(lines[0].includes('/ai-infra-primer/models/references/'));
  assert.equal(lines[2], '[b](../references.md)');
});

test('跨维度 README 链接指向 GitHub 源', () => {
  const src = '见 [inference 维度](../../inference/README.md) 与 [cluster](../../cluster/README.md)';
  const out = rewriteLinks(src, ctx);
  assert.ok(out.includes('](https://github.com/AlanFokCo/ai-infra-primer/blob/main/inference/README.md)'));
  assert.ok(out.includes('](https://github.com/AlanFokCo/ai-infra-primer/blob/main/cluster/README.md)'));
});

test('真实语料：references.md 相对链接清零', () => {
  const src = chapter('02-transformer-anatomy/03-attention-mechanism.md');
  const out = rewriteLinks(src, ctx);
  assert.ok(!out.includes('](../references.md)'));
  assert.ok(out.includes('](/ai-infra-primer/models/references/)'));
});
