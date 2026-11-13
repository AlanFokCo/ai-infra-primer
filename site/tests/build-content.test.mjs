import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SITE = path.dirname(fileURLToPath(import.meta.url)).replace(/\/tests$/, '');
const OUT = path.join(SITE, 'src/generated');

test('管线全量构建：51 章 / 10 Part / 文献 >100 / 无残留', () => {
  execFileSync('node', [path.join(SITE, 'scripts/build-content.mjs')], { stdio: 'pipe' });

  const manifest = JSON.parse(readFileSync(path.join(OUT, 'manifest.json'), 'utf8'));
  assert.equal(manifest.parts.length, 10);
  const totalChapters = manifest.parts.reduce((n, p) => n + p.chapters.length, 0);
  assert.equal(totalChapters, 51);
  assert.equal(manifest.flat.length, 51);
  assert.equal(manifest.flat[0].num, '1.1');
  assert.equal(manifest.flat[50].num, '10.4');
  assert.equal(manifest.flat[0].prev, null);
  assert.equal(manifest.flat[0].next.num, '1.2');

  // 生成章节文件计数与内容检查
  let files = 0;
  for (const p of manifest.parts) {
    const dir = path.join(OUT, 'chapters', p.dir);
    for (const f of readdirSync(dir)) {
      files += 1;
      const body = readFileSync(path.join(dir, f), 'utf8');
      assert.ok(!body.includes('$`'), `${p.dir}/${f}: $' 残留`);
      assert.ok(!/\\thinspace\b/.test(body), `${p.dir}/${f}: \\thinspace 残留`);
      assert.ok(!body.includes('](../references.md'), `${p.dir}/${f}: 相对链接残留`);
      assert.ok(body.startsWith('---\n'), `${p.dir}/${f}: 缺 frontmatter`);
    }
  }
  assert.equal(files, 51);

  const refs = JSON.parse(readFileSync(path.join(OUT, 'references.json'), 'utf8'));
  const refCount = refs.groups.reduce((n, g) => n + g.entries.length, 0);
  assert.ok(refCount > 100, `refCount=${refCount}`);
  assert.equal(refs.groups.length, 10);
  assert.ok(refs.glossary.length > 100, '术语表缺失');
});
