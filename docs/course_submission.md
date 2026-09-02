# 课程过程记录与双远程协作

## 1. 仓库定位

- Gitee `origin`：课程主仓库、功能分支、Pull Request、Review、每日扫描与期末提交。
- GitHub `github`：只作为备份镜像，默认不从 GitHub 拉取后继续开发。
- 不在两个平台分别修改两套代码；所有开发从 Gitee 最新 `main` 创建分支。

## 2. 身份检查

每位成员首次提交前在当前仓库执行：

```bash
git config user.name "报名时填写的名字"
git config user.email "本人已绑定的邮箱"
git config user.name
git config user.email
```

禁止为了补算工作量修改旧 commit 日期或重写已共享历史。

## 3. 每次提交

```bash
git status
git add <本次任务文件>
git commit -m "一句话说明本次改动"
git push origin <当前分支>
```

功能未完成也应形成真实、小范围、可解释的阶段提交，并在 24:00 前推送当天分支。禁止提交不能运行的占位代码、秘密或缓存。

## 4. PR 合并与 GitHub 镜像

功能分支在 Gitee 创建 Pull Request 合入 `main`。Role 1 合并并测试后执行：

```bash
git switch main
git pull --ff-only origin main
git push github main
```

GitHub 只同步已经在 Gitee 审核过的 `main`。若两个平台出现不同的独立提交，先停止并核对，不直接强推覆盖。

## 5. 每日 22:00 检查

- `daily/Dn.md` 已写真实目标、进展、问题、关键 commit 和明日最低目标。
- `todos.md` 已按验收结果更新复选框。
- 六位成员的 `prompts/Dn/*.txt` 已提交且无敏感信息。
- 每位成员当天分支已推送到 Gitee。
- `git status` 无意外文件，测试结果已记录。

## 6. 六人 Gitee 账号

| Role | Gitee 用户名 | 课程模块 |
|---|---|---|
| Role 1 | `Laurenhuan` | 架构、应用集成、PR 与课程仓库 |
| Role 2 | `juanjuan-he` | 数据获取、清洗与公共指标 |
| Role 3 | `aaaSpringaaa` | EDA 与 Plotly 可视化 |
| Role 4 | `zhou_learn_hahaha` | Decision Tree 分类 |
| Role 5 | `a3556115538qqcom` | Stock Profile 与 K-Means |
| Role 6 | `lu2160` | Linear Regression 与算法 Review |
