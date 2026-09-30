# 8 Cross-cutting concepts

## 8.1 Ledger as cache and as quota counter
A single table, `llm_calls`, records every attempt. Its `ok` rows are the response cache (keyed by a SHA-256 of
route + messages + parameters); its rows since UTC midnight, grouped by deployment and summed across a
`rate_group`, are the daily usage. One source of truth means the cache and the quota cannot disagree.

## 8.2 Concurrency
- One `threading.Lock` guards `_cooldown_until`, `_disabled` and the per-group rpm windows.
- The rpm slot is appended to the window *inside* the selection critical section (reservation is atomic with the choice).
- Network calls and DB writes run outside the lock; each DB write opens its own session.
- SQLite engines must use WAL + busy timeout; the default temp-file engine does. (An in-memory `StaticPool` engine was rejected: it shares one connection across threads and lost ~1 in 60 concurrent bursts of writes.)

## 8.3 Error handling
Exceptions from the provider are never propagated to the caller; they are classified into outcomes and recovered
from. The only exception type callers see is `AllDeploymentsExhausted`. Analytics-write failures are logged and
swallowed so observability can never break a real call.

## 8.4 Validation of model output
Output is validated *before* it is accepted and cached. `json_validator` additionally rejects non-empty objects
sharing no field with the schema, because Pydantic accepts them when all fields default.

## 8.5 Logging
Standard `logging` under `llmrouter_free.router`: WARNING for each failed attempt (with outcome, route,
deployment), WARNING when switching from cloud to local, ERROR on exhaustion. No handlers are installed by the library.

## 8.6 Testability
`completion_fn`, `clock`, `sleep` are constructor arguments. The suite never touches the network and never sleeps.

## 8.7 Security
Only the *name* of an environment variable is configured. Keys are read at call time and passed to LiteLLM as
the `api_key` argument, never logged or stored in the ledger. The ledger stores prompts' hashes and responses;
treat the database as sensitive if prompts are.
