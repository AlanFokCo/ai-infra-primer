# references.md — training 维度文献索引与术语表

本文件是 training 维度全部章节的引用唯一事实源。核心规则：**先收录后引用**——条目由各波 research pass 核实后写入本文件，章节正文才能引用；`scripts/lint-chapter.sh` 检查项 4 强制执行：正文出现的 arXiv ID 在本文件 grep 不到即 lint FAIL。写作 agent 不新增条目，只引用已有条目。

## 写入规则

- 条目只由 research pass 写入；「核实日期」= 当次核实 ID / URL 的日期（YYYY-MM-DD）。spec 中 `⟨verify⟩` 标记的条目必须经 WebFetch 复核（arXiv abs 页 / HF config / repo 路径）后才收录。
- 非 arXiv 来源（blog、官方文档、repo 文件、无 arXiv 的经典文献）在「arXiv ID / URL」字段写完整 URL 或标准出处（如「Polyak 1964, USSR Computational Mathematics」）。lint 只机械核对 arXiv ID，URL 条目的真实性由 research pass 与 evaluator 把关。
- 「证据等级」五档：[A] / [B] / [C] / [W] / 教学参照，定义见 STYLE_GUIDE §6；training 维度对方法/框架锚点的口径细化（paper + official code 即 [A]、无 paper 的官方库为教学参照）见 training spec §4.1。[W] 级找到正式 paper 后的回升判定同 STYLE_GUIDE §6。
- key 为小写 slug，全文件唯一，取论文或资源的通用短名（如 `zero`、`megatron-lm`）。
- 章节写完（或引用关系变化）后回写「被引章节」；一条目被多章引用时逗号分隔累加（如 `4.1, 4.2`）。
- 正文引用格式 `[Rafailov et al., 2023]`，链接到本文件对应条目（spec §4.6 / STYLE_GUIDE §6）。

## 条目 schema

每条目七个字段：

| 字段 | 含义 |
| --- | --- |
| key | 条目唯一 slug（小写） |
| 标题 | 论文 / 资源标题 |
| 作者 | 第一作者 et al.，或机构名 |
| arXiv ID / URL | arXiv 编号；非 arXiv 来源写完整 URL |
| 证据等级 | [A] / [B] / [C] / [W] / 教学参照 |
| 核实日期 | research pass 核实日期（YYYY-MM-DD） |
| 被引章节 | 引用本条目的章节号，逗号分隔 |

条目写法（表头 + 分隔行 + 数据行；下行示例同时是 lint grep 的格式基准）：

```
| key | 标题 | 作者 | arXiv ID / URL | 证据等级 | 核实日期 | 被引章节 |
| --- | --- | --- | --- | --- | --- | --- |
```

## 条目

按 Part 分组；research pass 向对应 Part 的表追加，chapter → sources 映射体现在「被引章节」字段。

（骨架：各 Part 表由对应波次的 research pass 建立——W1 建 P1/P2 表，W2 建 P3/P4，W3 建 P5/P6，W4 建 P7，W5 建 P8，W6 建 P9。）

## 术语表

（各波 writing pass 累积：章节首次出现的「中文（English Term）」对在此登记，格式与 models/references.md 术语表一致。）
