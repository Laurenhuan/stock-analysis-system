# D2 提交 rebase 对账表（原哈希 → rebase 后哈希）

> 维护人：Role 3 · 25151405 胡梦帆
> 背景：2026-09-02 组长对 `feat/eda-visualization` 做过 rebase + force-update，同一批提交因此出现「原哈希」（备份分支 `backup/eda-local-584717a`）与「rebase 后哈希」（已合入 main）两套，机械层看到的成对提交需靠本表解释。

## 重复提交映射（同内容、不同哈希）

| 原哈希（backup/eda-local-584717a） | rebase 后哈希（main） | 提交内容 |
|---|---|---|
| 8cc3707 | c8d5207 | feat(eda)：6 个 EDA 分析函数 |
| 4906034 | f6ad7b1 | feat(charts)：6 个 Plotly 图表 |
| 34accd5 | 64ef7b2 | test：EDA + 图表单测 39 个 |
| 846a1f5 | 3ae5e41 | docs：demo 脚本、D2 过程记录与 prompts 导出 |
| 578c4f4 | 92a7331 | docs：精简提示词为仅提示、统一措辞为演示脚本、忽略 demo 生成图 |

rebase 后新增（无原哈希对应）：

- 145d447 fix(eda,charts)：按 PR 评审意见修正统计口径与图表校验
- 24692a3 docs：补充 D2 日报评审修复进展并导出剩余 4 个会话提示词

## 备份分支指向

`backup/eda-local-584717a` → 584717a「按老师规则将过程记录归入学号子目录」，其上还有 a837120（重命名提示词为 role3_claude.txt）等共 7 个提交，是 rebase 前的原始链，仅作对账备份，不再推进。

## 课程日编号变更（D2 → D1 → D2）

- 67df399 docs：按课程日历将过程记录从 D2 更正为 D1
- d022072 docs：更新 D1 日报提交哈希为 rebase 后实际值并同步明日计划
- d578858 docs：更正过程记录课程日为 D2 并补齐 D1 记录

三者均发生在 09-02，对应 D1 日报/提示词为回补记录（已在文首标注「补记于 09-02」）。

## 用途

- 评审/机械层遇到成对提交（如 8cc3707 与 c8d5207）时，按本表认作同一提交的两个哈希版本。
- 若组长再次 force-update，本表作为新一轮 rebase 对账的起点。
