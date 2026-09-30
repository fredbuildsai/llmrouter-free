# 10 Quality requirements

## Quality tree
- **Reliability**: failover, retry policy, exhaustion semantics.
- **Correctness**: atomic slot reservation, lossless ledger writes, no invalid output cached.
- **Operability**: `status`, per-attempt ledger, warnings with outcome.
- **Usability**: one config file, missing keys skipped, annotated starter.
- **Maintainability**: pure functions for arithmetic, injected time and I/O, tested public contract.

## Scenarios
| ID | Scenario | Measure | Verified by |
|---|---|---|---|
| Q1 | The first deployment 429s | The next is used in the same `complete()` call; the group is cooled for 60 s | `test_router.py` failover tests |
| Q2 | A deployment hits its daily quota | It is excluded for 24 h; subsequent calls do not attempt it | `test_router.py` quota tests |
| Q3 | 8 threads write to the ledger at once | 0 lost rows | `test_store.py` concurrency tests |
| Q4 | A provider returns garbled JSON | Treated as `invalid_output`, never cached | `test_validation.py` |
| Q5 | All deployments are down | `AllDeploymentsExhausted` in bounded time (<= `max_wait_seconds`) | `test_router.py` exhaustion tests |
| Q6 | README examples drift from code | Test failure | `test_readme_examples.py` |
| Q7 | A key is absent | The deployment is skipped, no error | `test_router.py`, `test_cli.py` |
