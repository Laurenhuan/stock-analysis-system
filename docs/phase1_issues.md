# Phase 1 Issue Drafts

These Issue bodies are ready to copy into GitHub after the Private remote is available. Do not assign them until the real member usernames are confirmed.

## Role 2 — Financial Data Foundation P0

- Role: Role 2
- Goal: Implement the Contract-compliant financial data foundation.
- Scope: `fetch_market_data`, `clean_market_data`, `build_common_features`, Sample Data fallback, and data unit tests.
- Owner: Role 2; GitHub assignee pending.
- Contract dependencies: Market Data Contract v0.2 and shared exceptions.
- Branch: `feat/data-foundation`

Acceptance criteria:

- Market Data Contract v0.2 compliant.
- Tests pass.
- No model, EDA, visualization, or Web implementation.
- No public Contract change without an approved Change Request.

## Role 3 — EDA & Visualization Foundation P0

- Role: Role 3
- Goal: Build reusable EDA and Figure functions on Contract-compliant data.
- Scope: Descriptive statistics, returns analysis, risk analysis, trend analysis, correlation, and independent Figure functions.
- Owner: Role 3; GitHub assignee pending.
- Contract dependencies: Market Data Contract v0.2.
- Branch: `feat/eda-visualization`

Acceptance criteria:

- Functions accept documented Market Data fields and return reusable analysis/Figure outputs.
- Tests pass.
- No data acquisition, model training, or Streamlit page implementation.
- No public Contract change without an approved Change Request.

## Role 4 — Supervised Learning Foundation P0

- Role: Role 4
- Goal: Implement the approved supervised-learning baseline without time leakage.
- Scope: Decision Tree Classification, Linear Regression, time-based train/test split, classification metrics, regression metrics, and leakage tests.
- Owner: Role 4; GitHub assignee pending.
- Contract dependencies: Market Data and Supervised Learning Contract v0.2.
- Branch: `feat/supervised-learning`

Acceptance criteria:

- Uses `X(t) → y(t+1)` and the earliest 80% / latest 20% split without shuffle.
- Returns the approved shared output Schema and P0 metrics.
- Leakage and Contract tests pass.
- No unapproved model or Web implementation.

## Role 5 — Clustering & Stock Profiling Foundation P0

- Role: Role 5
- Goal: Implement the approved stock profile and clustering baseline.
- Scope: Stock Profile Table, StandardScaler, K-Means with `k=3`, cluster outputs, and reproducibility tests.
- Owner: Role 5; GitHub assignee pending.
- Contract dependencies: Market Data and Clustering Contract v0.2.
- Branch: `feat/clustering`

Acceptance criteria:

- Uses only `mean_return`, `volatility`, and `max_drawdown` for P0 clustering.
- Returns the approved profiles, cluster centers, features, and k fields.
- Tests pass and reproducibility parameters are recorded.
- No unapproved algorithm, EDA, or Web implementation.
