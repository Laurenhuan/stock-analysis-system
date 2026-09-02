# Six-Role Development Boundaries

## Status

This document is the authoritative implementation boundary for the current
six-person team. It supersedes the former five-role split for all new work and
PR Review. Existing approved commits remain valid.

The team labels below identify the current assignment but are not necessarily
GitHub usernames. Do not activate `.github/CODEOWNERS` until every real GitHub
handle is verified.

## Integration model

```text
Role 2 data --------------------------+
Role 3 analysis / figures ------------+--> Role 1 Service Layer --> Streamlit
Role 4 classification ----------------+
Role 5 clustering --------------------+
Role 6 regression --------------------+
```

Role 2-6 produce importable, tested domain functions. Role 1 coordinates
interfaces, calls those functions through `src/services/`, maps domain errors
to user-facing states and integrates results into `pages/`.

## Ownership matrix

| Role | Team label | Scope | Primary implementation paths | Branch |
| --- | --- | --- | --- | --- |
| Role 1 — Architecture and Application Integration | Project owner | Streamlit, Service Layer, interface coordination, central repository, PRs and final integration | `app.py`, `pages/`, `src/services/`, `src/contracts/`, `src/utils/`, `tests/integration/`, architecture and governance docs | Task-specific Role 1 branch |
| Role 2 — Financial Data Engineering | 嗯嗯 | A-share acquisition, cleaning, standard DataFrame, common indicators and approved Sample fallback | `src/data/`, data unit tests and coordinated `data/sample/` files | `feat/data-foundation` |
| Role 3 — Financial Analysis and Visualization | 默念 | EDA, descriptive statistics, reusable Plotly figures and multi-stock comparison | `src/analysis/`, `src/visualization/` and matching unit tests | `feat/eda-visualization` |
| Role 4 — Supervised Classification | 周毅谦 | Decision Tree, next-day up/non-up classification, Accuracy and Confusion Matrix | `src/models/supervised/classification.py` and classification/leakage tests | `feat/decision-tree-classification` |
| Role 5 — Unsupervised Learning and Stock Profiling | 茨佰 | Stock Profile, StandardScaler, K-Means, cluster outputs and cross-stock positioning | `src/models/unsupervised/clustering.py` and profile/clustering/reproducibility tests | `feat/clustering` |
| Role 6 — Supervised Regression and Algorithm Review | 2160p180 | Linear Regression, next-day return, MAE, R², Actual-vs-Predicted data and algorithm Review | `src/models/supervised/regression.py` and regression/leakage tests | `feat/linear-regression` |

## Role 1 — Architecture and application integration

Role 1 owns page inputs and layout, Service orchestration, exception-to-UI
mapping, shared Contracts, integration tests, repository governance and the
final merge. Role 1 maintains shared package exports such as
`src/models/supervised/__init__.py` to prevent classification/regression merge
conflicts.

Role 1 must not duplicate Role 2-6 business algorithms. Integration calls the
public functions delivered by the responsible Role.

## Role 2 — Financial data engineering

Role 2 delivers `fetch_market_data`, `clean_market_data` and
`build_common_features` as Contract-compliant DataFrame functions.

Role 2 may change `src/data/`, matching unit tests and coordinated Sample Data.
Role 2 must not implement Streamlit, EDA, Plotly figures or models. Token values
must never enter code or Git. Data-source fallback must be explicit and visible
to the Service Layer; provider, validation and programming errors must not be
silently hidden by Sample Data.

## Role 3 — Financial analysis and visualization

Role 3 accepts Contract-compliant market data and returns reusable analysis
tables or Plotly `Figure` objects. Functions in `src/visualization/` must not
call Streamlit. Role 3 owns descriptive statistics, return/risk/trend analysis,
correlation and multi-stock comparison.

Role 3 must not acquire/clean source data, train models or implement pages.
Role 1 renders returned outputs in the appropriate Streamlit page.

## Role 4 — Decision Tree classification

Role 4 owns classification only:

- `DecisionTreeClassifier`;
- `X(t) -> direction(t+1)`;
- `next_return > 0` maps to 1; otherwise 0;
- earliest 80% train / latest 20% test, without shuffle;
- `accuracy` and fixed-label-order `[0, 1]` 2x2 `confusion_matrix`;
- `ClassificationResult` predictions containing `trade_date`, `y_true` and
  `y_pred`.

Role 4 must not modify `regression.py`, clustering, pages, Service, Contracts or
data acquisition. Role 3 owns reusable Confusion Matrix figures; Role 1 owns
page integration.

## Role 5 — Stock Profile and K-Means

Role 5 owns Stock Profile construction and `KMeans(k=3)` using exactly:

- `mean_return`;
- `volatility` with `ddof=1`, not annualized;
- `max_drawdown`.

Role 5 fits `StandardScaler` before K-Means, records reproducibility parameters
and returns original-scale profiles and cluster centers through
`ClusteringResult`. Cluster numbers have no fixed good/bad meaning. Role 5 must
not write `cluster` into the shared Market DataFrame or implement pages, data
acquisition, EDA, classification or regression.

## Role 6 — Linear Regression and algorithm Review

Role 6 owns regression only:

- `LinearRegression`;
- `X(t) -> return(t+1)`;
- earliest 80% train / latest 20% test, without shuffle;
- `mae` and `r2`;
- `RegressionResult` and date-aligned Actual-vs-Predicted data containing
  `trade_date`, `y_true` and `y_pred`.

Role 3 owns reusable Actual-vs-Predicted Plotly figures and Role 1 owns page
integration. Role 6 reviews time-series splitting, target alignment, leakage,
NaN handling, scaler fitting, sklearn parameters and reproducibility across
algorithm PRs. Review means comments and requested changes; the module owner
implements corrections unless that owner explicitly coordinates a cross-role
edit.

## Shared and forbidden paths

Unless Role 1 coordinates a cross-role change, Role 2-6 must not modify:

- `app.py` or `pages/`;
- `src/services/`;
- `src/contracts/` or `docs/contracts/`;
- `src/utils/`;
- `tests/integration/`;
- another Role's primary module.

Tests follow the module under test. `requirements.txt`, package `__init__.py`
files, shared fixtures, Sample Data and repository configuration are shared
paths and must be called out explicitly in the PR.

## Handoff requirements

Every Role 2-6 PR must report:

1. files changed;
2. public function signatures;
3. required input columns and types;
4. output schema and a minimal example;
5. documented domain exceptions;
6. module and complete-suite test results;
7. Contract changes (normally `None`);
8. temporary logic and known limitations;
9. commit list and pushed branch;
10. a Role 1 integration guide.

## Review matrix

| Change | Required Review focus |
| --- | --- |
| Role 2 market data | Role 1 integration; Role 3 analysis fields; Role 4/6 temporal-model input; Role 5 multi-stock/profile input |
| Role 3 EDA/figures | Role 1 integration and Role 2 field semantics |
| Role 4 classification | Role 1 output integration and Role 6 algorithm/leakage Review |
| Role 5 clustering | Role 1 output integration, Role 2/3 feature semantics and Role 6 algorithm/reproducibility Review |
| Role 6 regression | Role 1 output integration and Role 4 temporal-split consistency Review |
| Contract/data policy | Every affected Role plus Role 1 coordination |

## Pull Request workflow

```text
central main -> personal fork feature branch -> commits -> PR -> human Review
-> fixes on the same branch -> complete tests -> merge
```

`Ready to merge` only means GitHub found no merge conflict. A PR is mergeable
only after required human Review, exact tests and Contract/Ownership checks.
Do not open duplicate PRs for Review fixes; push normal follow-up commits to the
existing feature branch. Do not force-push reviewed history unless the team
explicitly coordinates it.
