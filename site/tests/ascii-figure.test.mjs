import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { classifyBlock, renderFigure, renderGridFigure } from '../scripts/lib/ascii-figure.mjs';

const blocks = (p) => {
  const src = readFileSync(new URL(`../../models/${p}`, import.meta.url), 'utf8');
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let m;
  const out = [];
  while ((m = re.exec(src))) if (m[1] === '') out.push(m[2].split('\n'));
  return out;
};

test('classifyBlock：结构图 / 网格表 / 伪代码', () => {
  assert.equal(classifyBlock(blocks('02-transformer-anatomy/08-assembly.md')[0]), 'diagram');
  assert.equal(classifyBlock(blocks('03-objectives-and-variants/01-objectives-and-masks.md')[1]), 'table');
  assert.equal(classifyBlock(blocks('02-transformer-anatomy/04-causality-and-kv-cache.md')[0]), 'code');
});

test('renderFigure：芯片 + 角色 + 流光 + 箭头', () => {
  const svg = renderFigure(blocks('02-transformer-anatomy/08-assembly.md')[0]);
  assert.ok(svg.includes('class="chip role-'), '应有角色芯片');
  assert.ok(svg.includes('role-norm'), 'RMSNorm 应染 norm 色');
  assert.ok(svg.includes('role-add'), '(+) 应是加和圆圈');
  assert.ok(svg.includes('class="flow"'), '长边应有流光');
  assert.ok(svg.includes('class="arrow"'), '应有箭头');
  assert.ok(svg.includes('RMSNorm'), '标签保留');
  assert.ok(!svg.includes('──'), '结构线不残留为文本');
});

test('renderFigure：末尾 [shape] 收进芯片第二行；行中 [shape] 留在标签内', () => {
  const svg = renderFigure(blocks('02-transformer-anatomy/03-attention-mechanism.md')[0]);
  assert.ok(svg.includes('class="dim"'), '维度注存在');
  // Q [B, T, d_k] → 芯片内含 [B, T, d_k] 维度行
  assert.ok(/>Q</.test(svg) && svg.includes('[B, T, d_k]'), 'Q 芯片与维度注');
});

test('renderGridFigure：掩码 token 染色 + 分隔线', () => {
  const svg = renderGridFigure(blocks('03-objectives-and-variants/01-objectives-and-masks.md')[1]);
  assert.ok(svg.includes('cell-hot'), '[M]/[s] 应染色');
  assert.ok(svg.includes('stroke-dasharray'), '分隔虚线');
  assert.ok(svg.includes('token'), '行标签保留');
});

test('转义：< > & 不泄漏', () => {
  const svg = renderFigure(['a ──► b <x> & "c"']);
  assert.ok(!svg.includes('<x>'));
});
