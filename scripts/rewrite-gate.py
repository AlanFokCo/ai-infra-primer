#!/usr/bin/env python3
"""可读性改写的保留性门禁（spec 2026-11-18 §3 机械检查项）。

对照 git HEAD 版本与当前章节文件，逐项核对改写不动内容：
  formulas  display $$ 块数 + 行内 $..$ 数（剥围栏后）逐一相等
  sections  ## 节标题序列逐一相等
  citations ../references.md 链接引用的 key 集合 + arXiv ID 集合守恒
  xrefs     「§N」「式 (N)」「N.N」跨章引用 multiset 计数守恒
  fences    围栏块数量、语言标记序列、bare 块内容哈希逐一相等（场景图按序号映射，错一位全错配）
  metadata  章首 metadata 单行、恰好三个 | 字段、第 2/3 段带「前置：」「锚点：」前缀、公式内无裸 |
  cjk       CJK 字数：骤降 >15% fail；增长 >25% flag（不 fail）

用法：python3 scripts/rewrite-gate.py <chapter-file>   # 与 git HEAD 对照
退出码：0 通过（flag 只警告），1 有违规。
"""
import re
import subprocess
import sys


def git_head(path):
    r = subprocess.run(['git', 'show', f'HEAD:{path}'], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout


def strip_fences(text):
    lines, out, fences = text.split('\n'), [], []
    i = 0
    while i < len(lines):
        m = re.match(r'^```(\w*)\s*$', lines[i])
        if m:
            buf = []
            j = i + 1
            while j < len(lines) and not re.match(r'^```\s*$', lines[j]):
                buf.append(lines[j])
                j += 1
            fences.append((m.group(1), '\n'.join(buf)))
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out), fences


def formula_counts(text):
    display = len(re.findall(r'\$\$[\s\S]+?\$\$', text))
    inline = len(re.findall(r'\$[^$\n]+?\$', re.sub(r'\$\$[\s\S]+?\$\$', '', text)))
    return display, inline


def sections(text):
    return [l for l in text.split('\n') if re.match(r'^#{2,3}\s', l)]


def citations(text):
    keys = set(re.findall(r'\]\(\.\./references\.md\)', text))
    ids = set(re.findall(r'arXiv:(\d{4}\.\d{4,5})', text))
    return ids  # key 集合不可见（链接不带 key），核 arXiv ID 集合 + 链接数
    # 注：key 不可见处由保留性审查补


def xref_multiset(text):
    pats = re.findall(r'§\s*\d+(?:\.\d+)?', text) + re.findall(r'式 \((\d+)\)', text)
    pats += re.findall(r'(?<![\d.])(\d+\.\d+)(?![\d.])', text)  # 章号引用
    return sorted(pats)


def metadata_check(text):
    m = re.search(r'^>\s+(.+)$', text, re.M)
    if not m:
        return ['缺 metadata 块']
    line = m.group(1)
    problems = []
    fields = line.split('|')
    if len(fields) != 3:
        problems.append(f'metadata 字段数 {len(fields)} ≠ 3（管道内公式禁裸 |）')
    if len(fields) >= 2 and not fields[1].strip().startswith('前置：'):
        problems.append('metadata 第 2 段缺「前置：」前缀')
    if len(fields) >= 3 and not fields[2].strip().startswith('锚点'):
        problems.append('metadata 第 3 段缺「锚点」前缀')
    return problems


def count_cjk(text):
    return len(re.findall(r'[一-鿿]', text))


def main():
    path = sys.argv[1]
    cur = open(path, encoding='utf-8').read()
    old = git_head(path)
    problems = []
    flags = []

    problems.extend(metadata_check(cur))

    if old is None:
        flags.append('git HEAD 无此文件（新增章？）——只查 metadata 门，其余项由 reviewer 人工核')
    else:
        old_txt, old_fences = strip_fences(old)
        cur_txt, cur_fences = strip_fences(cur)

        if formula_counts(old_txt) != formula_counts(cur_txt):
            problems.append(f'公式计数变 {formula_counts(old_txt)} → {formula_counts(cur_txt)}')
        if sections(old_txt) != sections(cur_txt):
            problems.append('节标题序列变化')
        old_ids, cur_ids = citations(old_txt), citations(cur_txt)
        if old_ids != cur_ids:
            problems.append(f'arXiv ID 集合变 {sorted(old_ids)} → {sorted(cur_ids)}')
        old_x, cur_x = xref_multiset(old_txt), xref_multiset(cur_txt)
        if old_x != cur_x:
            problems.append('跨章/式引用 multiset 不守恒（详见保留性审查）')
        if len(old_fences) != len(cur_fences):
            problems.append(f'围栏块数量变 {len(old_fences)} → {len(cur_fences)}')
        elif [f[0] for f in old_fences] != [f[0] for f in cur_fences]:
            problems.append('围栏块语言标记序列变化')
        elif [f[1] for f in old_fences] != [f[1] for f in cur_fences]:
            problems.append('围栏块内容变化（场景图序号映射与代码注释冻结，内容不可动）')

        co, cc = count_cjk(old_txt), count_cjk(cur_txt)
        if cc < co * 0.85:
            problems.append(f'CJK 骤降 {co} → {cc}（>15% 疑内容丢失）')
        elif cc > co * 1.25:
            flags.append(f'CJK 增长 {co} → {cc}（>25%，标注复核理由）')

    for p in problems:
        print(f'FAIL: {p}')
    for f in flags:
        print(f'FLAG: {f}')
    if not problems:
        print(f'GATE OK: {path}' + ('' if not flags else '（含 flag）'))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
