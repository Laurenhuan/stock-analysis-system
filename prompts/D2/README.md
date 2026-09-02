# prompts/D2 提示词导出

本目录存放 Day 2 的 AI 提示词 / 对话记录导出文件。

## 怎么导出（Claude Code）

在 Claude Code 里输入：

```
/export
```

它会生成当前会话的导出文件（Markdown / HTML）。把它保存到本目录，例如：

- `D2-prompt.md`（对话文字）
- `D2-session.html`（可选，含界面截图）

## 要求

- 一个文件对应一天的对话，命名清晰（`D2-xxx`）。
- 不要提交 Token、密码或任何秘密（导出前确认里面没有 `.env` 内容）。
- 如果 `/export` 不可用，也可以把 `~/.claude/projects/` 下当天会话的 `.jsonl` 转存一份。
