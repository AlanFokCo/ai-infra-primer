#!/usr/bin/env python3
"""GitHub 行内公式边界检查（STYLE_GUIDE §3，2026-11-13 两轮渲染实测规则）。

逐行把章节切成 code / btmath($`...`$) / display($$..$$ 同行) / math($..$) / text 段，
对普通行内公式检查：
  open   开口 $ 的前字符只允许 空白/行首/(/元素边界（**、code、另一公式之后）
  close  闭口 $ 后不允许紧跟 ASCII 字母或数字
  space  公式内容首尾不允许是空白
  matrix 行内公式不允许含 \\begin{...} 矩阵环境（用 display）
  pipe   表格行（| 开头）内公式不允许含裸 |（写 \\|）
  range  $A$–$B$ / $A$-$B$ 区间写法（第二段开口紧贴 dash 不解析，应合并为 $A\\text{–}B$）
代码围栏与多行 display 块内部不检查。
退出码：0 通过，1 有违规。
"""
import re
import sys


def scan_line(line):
    segs = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == '$' and i > 0 and line[i - 1] == '\\':
            if segs and segs[-1][0] == 'text':
                segs[-1] = ('text', segs[-1][1] + '$')
            else:
                segs.append(('text', '$'))
            i += 1
            continue
        if c == '$' and line[i:i + 2] == '$`':
            j = line.find('`$', i + 2)
            if j == -1:
                segs.append(('text', '$'))
                i += 1
                continue
            segs.append(('btmath', line[i:j + 2]))
            i = j + 2
            continue
        if c == '$' and line[i:i + 2] == '$$':
            j = line.find('$$', i + 2)
            if j == -1:
                segs.append(('text', '$$'))
                i += 2
                continue
            segs.append(('display', line[i:j + 2]))
            i = j + 2
            continue
        if c == '$':
            j = i + 1
            while True:
                j = line.find('$', j)
                if j == -1:
                    break
                if line[j:j + 2] == '$$':
                    j += 2
                    continue
                if line[j - 1] == '\\':
                    j += 1
                    continue
                break
            if j == -1:
                segs.append(('text', '$'))
                i += 1
                continue
            segs.append(('math', line[i:j + 1]))
            i = j + 1
            continue
        if c == '`':
            m = re.match(r'`+', line[i:])
            ticks = m.group(0)
            j = line.find(ticks, i + len(ticks))
            if j == -1:
                segs.append(('text', c))
                i += 1
                continue
            segs.append(('code', line[i:j + len(ticks)]))
            i = j + len(ticks)
            continue
        j = i
        while j < n and line[j] not in '$`':
            j += 1
        segs.append(('text', line[i:j]))
        i = j
    return segs


def check_line(line, fname, lineno, problems):
    segs = scan_line(line)
    in_table = line.lstrip().startswith('|')
    for idx, (kind, text) in enumerate(segs):
        if kind != 'math':
            continue
        inner = text[1:-1]
        if inner != inner.strip():
            problems.append((fname, lineno, 'space', text[:50]))
        if '\\begin{' in inner:
            problems.append((fname, lineno, 'matrix', text[:50]))
        if in_table and re.search(r'(?<!\\)\|', inner):
            problems.append((fname, lineno, 'pipe', text[:50]))
        prev = ''
        prevkind = ''
        for k in range(idx - 1, -1, -1):
            if segs[k][1]:
                prev = segs[k][1][-1]
                prevkind = segs[k][0]
                break
        safe = (not prev) or prev.isspace() or prev == '(' \
            or (prevkind == 'text' and segs[idx - 1][1].endswith('**')) \
            or prevkind in ('math', 'btmath', 'code')
        if not safe:
            problems.append((fname, lineno, 'open', segs[idx - 1][1][-15:] + '|' + text[:35]))
        nxt = segs[idx + 1] if idx + 1 < len(segs) else None
        if nxt and nxt[0] == 'text' and re.match(r'[A-Za-z0-9]', nxt[1]):
            problems.append((fname, lineno, 'close', text[:40] + '|' + nxt[1][:12]))
        if prevkind == 'text' and re.fullmatch(r'[–-]', segs[idx - 1][1].strip()) \
                and idx >= 2 and segs[idx - 2][0] == 'math':
            problems.append((fname, lineno, 'range',
                             segs[idx - 2][1] + segs[idx - 1][1] + text[:30]))


def check_file(path):
    problems = []
    in_fence = False
    in_display = False
    for lineno, line in enumerate(open(path, encoding='utf-8').read().split('\n'), 1):
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.strip() == '$$':
            in_display = not in_display
            continue
        if in_display:
            continue
        check_line(line, path, lineno, problems)
    return problems


if __name__ == '__main__':
    all_problems = []
    for f in sys.argv[1:]:
        all_problems.extend(check_file(f))
    for f, ln, rule, ctx in all_problems:
        print(f'FAIL[{rule}] {f}:{ln}: {ctx}')
    sys.exit(1 if all_problems else 0)
