# GitHub Collaboration Workflow

This is the minimum collaboration process for the six-person team. It deliberately avoids complex DevOps.

## Repository topology

The central repository is owned and integrated by Role 1. Role 2-6 develop in personal forks and open Pull Requests back to central `main`.

```text
central: Laurenhuan/stock-analysis-system
    ↑ Pull Request
personal fork / feature branch
```

Role 1 may use the central repository as `origin` but still develops on a feature branch and opens a same-repository PR. Nobody develops directly on central `main`.

## Role 2-6: first-time fork setup

```bash
git clone <personal-fork-url>
cd stock-analysis-system
git remote add upstream <central-repository-url>
git remote -v
```

`origin` must be the contributor's fork. `upstream` must be the central repository. Do not guess either URL; verify both before pushing.

## Start each task

```bash
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git checkout -b <assigned-feature-branch>
```

Use the branch recorded in `docs/role_boundaries.md` and the relevant Issue. If local work prevents a fast-forward update, stop and inspect instead of resetting or discarding changes.

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

The PR base is central `main`; the compare branch is the contributor fork's assigned feature branch. Review fixes are pushed as normal new commits to the same branch and automatically update the existing PR. Do not open a duplicate PR and do not force-push reviewed history.

Before approval, Reviewers must inspect `Files changed`, verify Ownership and Contract boundaries, and confirm the complete test command. GitHub's `Ready to merge` status means only that no merge conflict was detected.

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
- Do not implement Streamlit or Service code outside Role 1 unless a cross-role change is explicitly coordinated.
- Do not modify another algorithm owner's module merely because Role 6 provided a Review.
- Do not commit `.env`, tokens, credentials, caches, or raw/processed datasets.
