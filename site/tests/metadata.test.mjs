import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseChapter } from '../scripts/lib/metadata.mjs';

const chapter = (p) => readFileSync(new URL(`../../models/${p}`, import.meta.url), 'utf8');

test('H1 拆编号/中英标题', () => {
  const src = chapter('02-transformer-anatomy/03-attention-mechanism.md');
  const meta = parseChapter(src, 'models/02-transformer-anatomy/03-attention-mechanism.md');
  assert.equal(meta.num, '2.3');
  assert.equal(meta.titleZh, 'Attention 机制解剖');
  assert.equal(meta.titleEn, 'Anatomy of the Attention Mechanism');
  assert.equal(meta.partDir, '02-transformer-anatomy');
  assert.equal(meta.file, '03-attention-mechanism');
});

test('metadata 块三段切分', () => {
  const src = chapter('02-transformer-anatomy/03-attention-mechanism.md');
  const meta = parseChapter(src, 'models/02-transformer-anatomy/03-attention-mechanism.md');
  assert.ok(meta.metaLine.positioning.startsWith('本章把一个公式拆到底'));
  assert.ok(meta.metaLine.prerequisites.startsWith('前置：'));
  assert.ok(meta.metaLine.anchors.startsWith('锚点：'));
});

test('CJK 字数在合理区间（≥2500 字下限）', () => {
  const src = chapter('02-transformer-anatomy/03-attention-mechanism.md');
  const meta = parseChapter(src, 'models/02-transformer-anatomy/03-attention-mechanism.md');
  assert.ok(meta.cjkChars > 2500, `cjkChars=${meta.cjkChars}`);
});

test('证据等级集合', () => {
  const src = chapter('02-transformer-anatomy/03-attention-mechanism.md');
  const meta = parseChapter(src, 'models/02-transformer-anatomy/03-attention-mechanism.md');
  assert.ok(meta.evidenceLevels.includes('A'));
});

test('H1 不合规抛错', () => {
  assert.throws(() => parseChapter('# 没有编号的标题\n', 'models/x/y.md'), /H1 不符合/);
});
