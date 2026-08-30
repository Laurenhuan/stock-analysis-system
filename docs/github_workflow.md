# GitHub Collaboration Workflow

This is the minimum collaboration process for the five-person team. It deliberately avoids complex DevOps.

## Repository setup status

The local repository is ready, but the GitHub remote must be created or connected before the first team clone. The repository must be **Private** unless the project owner explicitly decides otherwise.

## Maintainer: first remote setup

Preferred path after GitHub CLI is installed:

```bash
gh auth login
cd D:\Codex工作空间\stock-analysis-system
gh repo create stock-analysis-system --private --source=. --remote=origin
git push -u origin main
git push -u origin docs/contracts-v02
git push -u origin chore/github-collaboration
```

Do not add `--public`, do not use `--force`, and verify the destination account before creating the repository.

Without GitHub CLI, create an empty Private repository named `stock-analysis-system` in the GitHub UI, then run:

```bash
git remote add origin <private-repository-url>
git push -u origin main
git push -u origin docs/contracts-v02
git push -u origin chore/github-collaboration
```

## Required Pull Request order

1. Open `docs/contracts-v02` → `main` using `docs/pull_requests/contracts-v02.md`.
2. Review and merge it with a normal merge after checks pass.
3. Open `chore/github-collaboration` → `main` using `docs/pull_requests/github-collaboration.md`.
4. Review and merge it.

The collaboration branch is stacked on the approved Contract commit. Merge the Contract PR first so the second PR contains only collaboration infrastructure.

## Every member: first time

```bash
git clone <private-repository-url>
cd stock-analysis-system
```

## Start each task

```bash
git checkout main
git pull origin main
git checkout -b feat/xxx
```

Use the branch recorded in the relevant Issue when one is provided.

## Finish development

```bash
git status
git diff
git add <specific-files>
git commit -m "feat(module): description"
git push -u origin feat/xxx
```

Then follow:

```text
Pull Request → Human Review → Merge
```

Complete the PR template, include exact test results, and request Review from the path owner listed in `docs/code_ownership.md`.

## Main branch protection

After the Private repository exists, a repository administrator should open:

```text
GitHub repository → Settings → Branches or Rules → Add rule for main
```

Enable only the baseline rule:

```text
Require a Pull Request before merging
```

Do not enable broad enterprise restrictions unless the team later needs them. Verify the rule in GitHub UI; never report protection as active until GitHub confirms it.

## Prohibited operations

- Do not develop directly on `main`.
- Do not run `git push --force`.
- Do not run `git reset --hard` when the impact is not fully understood.
- Do not silently modify another Role's Contract or business logic.
- Do not commit `.env`, tokens, credentials, caches, or raw/processed datasets.
