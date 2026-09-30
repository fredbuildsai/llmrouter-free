# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/),
versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-09-30

First standalone release, extracted from the `batterygemma` project where it was developed and used at scale
(tens of thousands of real calls against free-tier providers).

### Added
- `LLMRouter`: ordered per-route failover across deployments with per-deployment and per-`rate_group`
  requests-per-minute / requests-per-day limits, 429 and quota cooldowns, response cache and quota ledger
  (`llm_calls`), family exclusion for independent judging, paid-tier budget guard, and a local-model fallback cap.
- Retry-the-same-deployment on transient `error` / `invalid_output` before moving down the route
  (`max_attempts_per_deployment`).
- Thread-safe selection and rate-limit accounting: the rpm slot is reserved atomically with the selection.
- Optional analytics table (`llm_call_metrics`) via `metrics_engine=`.
- `json_validator` / `json_schema_response_format` for schema-constrained structured output, including a guard
  against garbled objects that share no fields with the schema.
- Context-window sizing for local Ollama (`recommend_num_ctx`, `apply_global_num_ctx`) with batch-aware sizing.
- `llmrouter-free` CLI: `init`, `status`, `test`.

### Fixed (during extraction)
- The default (no `engine=`) ledger is now a private temp-file SQLite database. An in-memory SQLite behind
  `StaticPool` shared one connection between threads and silently lost concurrent writes.
