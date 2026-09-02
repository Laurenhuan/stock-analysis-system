# Agent Working Agreement

Every coding agent must read this file, `docs/role_boundaries.md` and
`docs/team_conventions.md` before changing code in this repository.

## Architecture direction

```text
Streamlit (`app.py`, `pages/`)
    -> Service Layer (`src/services/`)
    -> Domain Modules (`src/data/`, `src/analysis/`, `src/models/`,
       `src/visualization/`)
```

Domain modules must not import Streamlit. Pages call the Service Layer instead
of depending on domain-module internals.

## Six-role ownership

| Role | Primary responsibility and paths |
| --- | --- |
| Role 1 | Architecture and application integration: `app.py`, `pages/`, `src/services/`, `src/contracts/`, `src/utils/`, `tests/integration/`, repository governance and final integration |
| Role 2 | Financial data engineering: `src/data/`, data unit tests and coordinated Sample Data |
| Role 3 | Financial analysis and visualization: `src/analysis/`, `src/visualization/` and matching unit tests |
| Role 4 | Decision Tree classification: `src/models/supervised/classification.py` and classification/leakage tests |
| Role 5 | Stock Profile and K-Means: `src/models/unsupervised/clustering.py` and profile/clustering/reproducibility tests |
| Role 6 | Linear Regression and algorithm review: `src/models/supervised/regression.py` and regression/leakage tests |

Role 2-6 deliver importable, tested domain functions. Role 1 owns all
Streamlit and Service integration. Ownership is not permission for
uncoordinated cross-role edits.

## Shared-file rule

Role 1 coordinates changes to shared files, including package `__init__.py`
exports, `requirements.txt`, shared fixtures, repository configuration,
`src/contracts/` and `docs/contracts/`. A contributor who needs a shared-file
change must list it in the PR and request every affected owner's Review.

Do not silently change a Contract. Contract or data-policy changes require an
approved `INTERFACE / DATA POLICY CHANGE REQUEST` before implementation.

## P0 algorithm boundaries

- Role 4 implements `DecisionTreeClassifier` only.
- Role 6 implements `LinearRegression` only.
- Role 5 implements `KMeans(k=3)` only.
- Do not add Random Forest, XGBoost, SVM, LSTM, PCA, DBSCAN or other P1
  algorithms without an approved scope change.
- Supervised learning uses `X(t) -> y(t+1)`, earliest 80% training data and
  latest 20% test data, without shuffle.
- Role 6 may review leakage, temporal splitting and sklearn usage in other
  algorithm PRs, but the owning Role makes the code changes.

## Git and review rules

- Gitee `origin` (`sp1-2026/25151407`) is the primary course repository.
- GitHub `github` is a backup mirror; do not develop independent commits on
  both platforms.
- Start from Gitee's current `main` and push the assigned feature branch to the
  central Gitee repository.
- Never commit directly to `main` and never force-push reviewed history.
- Do not reset, overwrite or delete unrelated work.
- Run module tests and the complete test suite before requesting Review.
- Push Review fixes to the existing PR branch; do not open a duplicate PR.
- `Ready to merge` means only that Gitee found no conflict. Human Review,
  Contract/Ownership checks and exact tests are still required.

## Course process records

- Before 24:00, every contributor pushes that day's commits and feature branch
  to Gitee.
- Root `README.md` and `todos.md` are shared by the whole team and coordinated
  by Role 1; progress must match actual commits and tests.
- Every contributor writes only their own daily report at
  `daily/<student_id>/Dn.md`. Never guess another member's student ID or write
  another member's report.
- Every contributor adds their own plain-text AI record at
  `prompts/<student_id>/Dn/<tool>.txt`. Do not fabricate another member's
  record.
- Never commit passwords, tokens, cookies or personal identity data in prompt
  exports.
- Each contributor's course reports belong at
  `docs/<student_id>/立项报告.md`, `docs/<student_id>/调研报告.md` and
  `docs/<student_id>/项目报告.md`; images belong in `docs/assets/`.

Detailed boundaries, branch names, handoff requirements and the Review matrix
are authoritative in `docs/role_boundaries.md`. Daily repository operations
are defined in `docs/course_submission.md` and `docs/github_workflow.md`.
