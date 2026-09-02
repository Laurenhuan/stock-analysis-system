# Pull Request Draft — GitHub Collaboration Baseline

> Historical record from the former five-role baseline. For current assignments, read `AGENTS.md` and `docs/role_boundaries.md`.

- Base: `main`
- Head: `chore/github-collaboration`
- Title: `chore(github): establish collaboration workflow`

## Purpose

Establish the lightweight GitHub workflow required for five Roles to develop in parallel after Contract v0.2 approval.

## Includes

- Pull Request template
- Feature/Role Task and Bug Issue templates
- CODEOWNERS placeholder without invented usernames
- Code ownership documentation
- Branch, Review, and merge workflow
- Branch protection recommendation
- Phase 1 Issue drafts for Role 2–5

## Validation

- Python compile passed
- pytest passed
- Streamlit smoke test passed
- `git diff --check` passed
- Secret scan passed
- No Role 2–5 business files changed

## Dependency

Merge `docs/contracts-v02` into `main` before reviewing this stacked Pull Request.
