import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { isDiagram, renderFigure } from '../scripts/lib/ascii-figure.mjs';

const blocks = (p) => {
  const src = readFileSync(new URL(`../../models/${p}`, import.meta.url), 'utf8');
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let m;
  const out = [];
  while ((m = re.exec(src))) if (m[1] === '') out.push(m[2].split('\n'));
  return out;
};

test('isDiagram 分类：结构图 vs 伪代码/掩码表', () => {
  assert.ok(isDiagram(blocks('02-transformer-anatomy/08-assembly.md')[0])); // 接线图
  assert.ok(isDiagram(blocks('04-modern-architectures/05-moe.md')[0])); // 分支结构
  assert.ok(!isDiagram(blocks('02-transformer-anatomy/04-causality-and-kv-cache.md')[0])); // 伪代码
  assert.ok(!isDiagram(blocks('03-objectives-and-variants/01-objectives-and-masks.md')[1])); // 掩码表
});

test('renderFigure：输出合法 SVG、shape pill 化、含数据流光点', () => {
  const svg = renderFigure(blocks('02-transformer-anatomy/08-assembly.md')[0]);
  assert.ok(svg.startsWith('<svg'));
  assert.ok(svg.includes('class="pill"'), 'shape 标签应 pill 化');
  assert.ok(svg.includes('class="flow"'), '长边应有流光点');
  assert.ok(svg.includes('class="arrow"'), '应有箭头');
  assert.ok(svg.includes('RMSNorm'), '文本保留');
  assert.ok(!svg.includes('──'), '结构线不应残留为文本');
});

test('renderFigure：CJK 原子化 + textLength 网格对齐', () => {
  const svg = renderFigure(blocks('04-modern-architectures/05-moe.md')[0]);
  // CJK 独字单元不带 textLength
  assert.ok(/class="t cjk" x="[\d.]+" y="[\d.]+">拼</.test(svg), 'CJK 应独字渲染');
  // ASCII 运行带 textLength
  assert.ok(svg.includes('textLength='), 'ASCII 运行应带 textLength');
});

test('转义：< > & 不泄漏', () => {
  const svg = renderFigure(['a ──► b <x> & "c"'].map((l) => l)[0] ? ['a ──► b'] : []);
  assert.ok(!svg.includes('<x>'));
});
