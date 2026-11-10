#!/usr/bin/env bash
# usage: bash scripts/lint-chapter.sh models/02-transformer-anatomy/03-attention-mechanism.md
set -u
f="$1"; fail=0
# 1. AI 味禁令词
if grep -nE "在当今|值得注意的是|综上所述|总的来说|众所周知|让我们一|随着.*快速发展" "$f"; then echo "FAIL: 禁令措辞"; fail=1; fi
# 2. 模板小节齐全
for s in "## 总结" "## 练习" "## 参考"; do grep -qF "$s" "$f" || { echo "FAIL: 缺 $s"; fail=1; }; done
# 3. metadata 块（前 8 行内存在 > 引用行）
head -8 "$f" | grep -q "^> " || { echo "FAIL: 缺 metadata 块"; fail=1; }
# 4. 正文出现的 arXiv ID 必须在 references.md 有条目
for id in $(grep -oE "arXiv:[0-9]{4}\.[0-9]{4,5}" "$f" | sed 's/arXiv://' | sort -u); do
  grep -q "$id" models/references.md || { echo "FAIL: references.md 缺 $id"; fail=1; }
done
# 5. 证据等级标注存在
grep -qE "\[(A|B|C|W)\]|教学参照" "$f" || { echo "FAIL: 无证据等级标注"; fail=1; }
# 6. \$\$ 定界符配平
n=$(grep -o '\$\$' "$f" | wc -l | tr -d ' '); [ $((n % 2)) -eq 0 ] || { echo "FAIL: \$\$ 不配平"; fail=1; }
[ "$fail" -eq 0 ] && echo "LINT OK: $f"
exit $fail
