# Code Ownership

Ownership identifies the primary reviewer and coordinator for a path. It is not an absolute prohibition on contributions from other Roles. Cross-role changes require coordination and Review from every affected owner.

## Path ownership

| Role | Primary paths |
| --- | --- |
| Role 1 | `app.py`, `pages/`, `src/contracts/`, `src/services/`, `src/utils/`, `tests/integration/`, `docs/architecture.md`, `docs/contracts/` |
| Role 2 | `src/data/` |
| Role 3 | `src/analysis/`, `src/visualization/` |
| Role 4 | `src/models/supervised/` |
| Role 5 | `src/models/unsupervised/` |

Tests primarily follow the module they verify. Shared integration tests are coordinated by Role 1.

## Review rules

- A change inside one Role's paths needs that Role's human Review.
- A cross-role Pull Request lists all affected Roles and obtains their Review.
- Shared Contract or data-policy changes require an `INTERFACE / DATA POLICY CHANGE REQUEST` before implementation.
- Ownership does not authorize direct commits to `main`; all work follows Branch → Commit → Pull Request → Review → Merge.

## GitHub usernames

Real GitHub usernames are not yet recorded. Until the team confirms all handles, `.github/CODEOWNERS` remains an explicit, inactive template rather than using invented accounts.

After confirmation, Role 1 should replace the comments with active path patterns and verified `@username` values, then submit the change through a Pull Request.
