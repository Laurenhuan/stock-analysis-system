# AI Usage Record

## Architecture Bootstrap

- Role: Role 1
- Agent: Codex / GPT Work Agent
- Task: Repository Bootstrap and Architecture Skeleton
- Human Responsibility: Review architecture, inspect generated files, run application, review Git diff before commit.

AI 用于生成初始工程骨架、文档草案和验证建议。任何 Contract 与金融含义仍需团队人工评审。

## Contract Freeze v0.2

- Role: Role 1
- Agent: Codex / GPT Work Agent
- Task: Convert approved engineering decisions into Review Candidate v0.2 documentation, importable schemas, exceptions, and Contract tests.
- Human Responsibility: Review Contract semantics and diff, approve the Pull Request, and coordinate any interface or data-policy changes.

## GitHub Collaboration Baseline

- Role: Role 1
- Agent: GPT Work / Codex
- Task: GitHub collaboration baseline and parallel development launch
- Agent responsibilities: Remote verification, PR workflow, GitHub templates, role ownership documentation, Phase 1 issue preparation.
- Human responsibilities: Review remote destination, verify PR, approve merge, invite collaborators, confirm GitHub usernames.

## Six-Role Ownership Revision

- Role: Role 1
- Agent: Codex
- Task: Convert the approved six-person assignment into repository-visible Agent instructions, path ownership, task boundaries, fork workflow and Review rules.
- Agent responsibilities: Add the root `AGENTS.md`, add the detailed six-role boundary document, split classification/regression ownership, update collaboration documentation and run repository checks.
- Human responsibilities: Review the diff, verify real GitHub usernames, manually push the branch, create the Pull Request and approve the final merge.

## Data Layer Implementation (Role 2)

- Role: Role 2
- Agent: Claude Code
- Task: 实现数据层 fetch/clean/features 三个模块、Sample Data Fallback、单元测试，并建立个人 fork 工作流。
- Human Responsibility: 审查 diff、理解并现场解释代码、运行测试、确认 PR 提交。

AI 用于生成数据层代码草稿与测试建议。字段语义、单位与复权口径均以 docs/contracts/market_data.md 为准，需团队人工 Review。
