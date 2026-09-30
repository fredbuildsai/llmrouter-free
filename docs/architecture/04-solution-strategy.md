# 4 Solution strategy

| Goal | Approach |
|---|---|
| Reliability | Ordered failover with an explicit **outcome taxonomy** (`ok`, `invalid_output`, `rate_limited`, `quota`, `error`, `auth`, `unavailable`); each outcome has its own recovery (retry same / cool down group / cool down one / disable). |
| Respect limits proactively | Local **sliding-window rpm** and ledger-derived **rpd** per `rate_group`, checked *before* the request; provider 429s are only the backstop. |
| Correct accounting under threads | One lock around selection + **atomic slot reservation**; network I/O and DB writes outside the lock. |
| Bad output is failure | Empty content and `validate=` errors are `invalid_output`; only validated answers are written as cache-eligible `ok`. |
| Persistence | The ledger *is* the cache and the quota counter: one table, no separate cache layer to keep consistent. |
| Testability | Time (`clock`, `sleep`) and I/O (`completion_fn`) are injected; no test needs network. |
| Low friction | Missing key = deployment skipped. Default ledger is a private temp file when no engine is given. |

Key decisions are recorded in [section 9](09-architecture-decisions.md).
