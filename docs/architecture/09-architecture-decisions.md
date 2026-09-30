# 9 Architecture decisions

## ADR-1: Extract the router into its own project
- **Status**: accepted (2026-09-30).
- **Context**: the router was built inside `batterygemma` but has no domain knowledge; two more projects need it.
- **Decision**: separate repository and package; the host injects its database engine.
- **Consequences**: narrow public API to maintain; hosts pin a tag; router tests travel with it.

## ADR-2: The ledger doubles as cache and quota counter
- **Context**: separate cache and usage tables can disagree after crashes or manual edits.
- **Decision**: one `llm_calls` table.
- **Consequences**: deleting rows changes both cache and quota (a feature: deleting a poisoned row un-poisons the cache).

## ADR-3: Reserve the rate-limit slot during selection
- **Context**: check-then-act under threads let two callers both take the last slot.
- **Decision**: append to the rpm window inside the locked selection step.
- **Consequences**: a slot is consumed even if the call then fails (correct: the provider counts it too).

## ADR-4: Retry the same deployment before falling back
- **Context**: with free tiers, the first deployment is usually far more reliable than the fallbacks, which 429 constantly. Falling back on its first hiccup wasted attempts on weaker models.
- **Decision**: `max_attempts_per_deployment`; on retry the per-deployment error cooldown is cleared.
- **Consequences**: one extra provider call per transient error; measured to keep most work on the strong deployment.

## ADR-5: Do not cap cloud attempts when the route has no local fallback
- **Context**: `max_cloud_attempts` was meant to jump to a local model sooner. On an all-cloud route it excluded the remaining cloud deployments after N attempts, so the last fallback was never tried.
- **Decision**: apply the cap only if the route contains a local deployment.

## ADR-6: Default ledger is a temp file, not in-memory SQLite
- **Context**: `sqlite://` + `StaticPool` shares a single connection between threads; concurrent sessions interleave transactions and silently drop writes.
- **Decision**: private temp-file database with WAL, removed at exit.
- **Consequences**: a deterministic regression test asserts concurrent checkouts get distinct connections.

## ADR-7: Validation guard for garbled JSON objects
- **Context**: a provider returned `{"": "results"}`; with all-default fields Pydantic accepted it; the cache replayed it forever, stalling a pipeline for hours.
- **Decision**: reject non-empty objects sharing no field with the model.
