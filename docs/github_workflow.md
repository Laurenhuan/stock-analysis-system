# Gitee Primary and GitHub Mirror Workflow

本项目从课程 D1 起以 Gitee 为唯一开发主仓库，GitHub 仅保留为备份镜像。文件名沿用 `github_workflow.md`，避免旧链接失效。

## Repository topology

```text
Gitee: sp1-2026/25151407 (primary)
    feature branch -> Gitee Pull Request -> main
                                      |
                                      +-> Role 1 pushes tested main to GitHub
GitHub: Laurenhuan/stock-analysis-system (mirror)
```

六位成员都是 Gitee 中央仓库开发者，不再要求为课程开发维护个人 GitHub fork。不得在 Gitee 和 GitHub 分别产生两套独立提交。

## First-time setup

新成员优先直接克隆 Gitee：

```bash
git clone https://gitee.com/sp1-2026/25151407.git
cd 25151407
git remote add github https://github.com/Laurenhuan/stock-analysis-system.git
git remote -v
```

如果仓库原本以 GitHub 为 `origin`，转换一次：

```bash
git remote rename origin github
git remote rename gitee origin
git remote -v
```

预期结果是 `origin` 指向 Gitee，`github` 指向 GitHub。不要把账号密码或 Token 写进 URL、脚本或仓库。

## Start each task

```bash
git switch main
git fetch origin
git merge --ff-only origin/main
git switch -c <assigned-feature-branch>
```

使用 `docs/role_boundaries.md` 记录的模块分支。若本地修改阻止切换或快进，先检查和提交自己的工作，不使用 `reset --hard` 丢弃文件。

## Commit and push before 24:00

```bash
git status
git diff
git add <specific-files>
git commit -m "feat(module): description"
git push -u origin <assigned-feature-branch>
```

当天未完成的模块也应形成真实、可解释的阶段提交并推送分支。每位成员同时提交自己的 `prompts/Dn/Gitee用户名-工具.txt`；Role 1 协调 `daily/Dn.md` 和 `todos.md`。

## Gitee Pull Request

```text
feature branch -> Gitee Pull Request -> path owner Review -> fixes on same branch
-> module tests -> complete tests -> merge into main
```

Review 必须核对 Ownership、Contract、输入输出、时间序列切分和测试结果。无冲突不等于可以合并。Review 修复继续推送原分支，不重复创建 PR，不强推已评审历史。

## Mirror reviewed main to GitHub

只有 Role 1 在 Gitee PR 合并并通过完整测试后执行：

```bash
git switch main
git pull --ff-only origin main
git push github main
```

如果 GitHub 拒绝快进，停止并检查两个平台是否出现独立提交；禁止直接使用 `--force` 覆盖。

## Prohibited operations

- 不直接开发或提交到 `main`。
- 不执行 `git push --force`。
- 不在两个平台分别修改同一任务。
- 不越过 Owner 修改其他 Role 的业务模块或公共 Contract。
- 不提交 `.env`、Token、密码、Cookie、缓存和大体积原始数据。
- 不替其他成员编造日报、提示词记录、测试结果或算法结果。
