# Phase 1 Six-Role Issue Boundaries

These task boundaries apply to the current six-person team. Assign Issues only to verified GitHub usernames. Every Agent must also read `AGENTS.md` and `docs/role_boundaries.md`.

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
- Scope: Descriptive statistics, returns analysis, risk analysis, trend analysis, correlation, multi-stock comparison, and independent Plotly Figure functions.
- Owner: Role 3; GitHub assignee pending.
- Contract dependencies: Market Data Contract v0.2.
- Branch: `feat/eda-visualization`

Acceptance criteria:

- Functions accept documented Market Data fields and return reusable analysis/Figure outputs.
- Tests pass.
- No data acquisition, model training, or Streamlit page implementation.
- No public Contract change without an approved Change Request.

## Role 4 — Decision Tree Classification P0

- Role: Role 4
- Goal: Implement next-trading-day up/non-up classification without time leakage.
- Scope: `DecisionTreeClassifier`, time-based train/test split, Accuracy, fixed-order 2x2 Confusion Matrix, predictions, and classification leakage tests.
- Owner: Role 4; GitHub assignee pending.
- Contract dependencies: Market Data and Supervised Learning Contract v0.2.
- Branch: `feat/decision-tree-classification`

Acceptance criteria:

- Uses `X(t) → direction(t+1)` and the earliest 80% / latest 20% split without shuffle.
- Returns the approved `ClassificationResult`, Accuracy and Confusion Matrix.
- Classification leakage and Contract tests pass.
- No regression, unapproved classifier or Web implementation.

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

## Role 6 — Linear Regression & Algorithm Review P0

- Role: Role 6
- Goal: Implement next-trading-day return regression without time leakage and Review algorithm correctness across model PRs.
- Scope: `LinearRegression`, time-based train/test split, MAE, R², date-aligned Actual-vs-Predicted data, regression leakage tests, and algorithm Review comments.
- Owner: Role 6; GitHub assignee pending.
- Contract dependencies: Market Data and Supervised Learning Contract v0.2.
- Branch: `feat/linear-regression`

Acceptance criteria:

- Uses `X(t) → return(t+1)` and the earliest 80% / latest 20% split without shuffle.
- Returns the approved `RegressionResult`, MAE, R² and date-aligned predictions.
- Regression leakage and Contract tests pass.
- Algorithm Review checks temporal splitting, target alignment, NaN handling, scaler fitting, sklearn parameters and reproducibility.
- Review comments do not authorize Role 6 to modify another owner's module without explicit coordination.
- No classification, clustering, unapproved regressor or Web implementation.
