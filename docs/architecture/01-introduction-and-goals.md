# 1 Introduction and goals

## 1.1 Requirements overview
`llmrouter-free` is a Python library that turns a prioritized list of LLM deployments - mostly free-tier
APIs - into a single dependable `complete(route, messages)` call.

| ID | Requirement |
|---|---|
| R1 | Route a request to the first *usable* deployment of an ordered list and fail over on error. |
| R2 | Never exceed a provider's stated per-minute / per-day limits from our side; back off correctly when the provider pushes back anyway. |
| R3 | Treat empty or schema-invalid output as a failure (fail over), never as a success. |
| R4 | Remember answers (cache) and quota usage (ledger) across process restarts. |
| R5 | Be safe to call concurrently from many threads. |
| R6 | Support independent judging (exclude a model family) and a hard paid-spend guard. |
| R7 | Work with a local Ollama model as a last-resort fallback. |
| R8 | Be testable without network access. |

## 1.2 Quality goals
| Priority | Goal | Scenario |
|---|---|---|
| 1 | **Reliability** | A 10,000-call batch keeps progressing while individual providers rate-limit, error out and go down. |
| 2 | **Correctness of accounting** | Under 8 concurrent threads no rate-limit slot is double-booked and no ledger row is lost. |
| 3 | **Transparency** | Every attempt and its outcome is recorded and queryable; `status` shows why a deployment is not being used. |
| 4 | **Low friction** | One YAML file plus environment keys; deployments with no key are skipped rather than erroring. |

## 1.3 Stakeholders
| Role | Expectation |
|---|---|
| Application developer (primary user) | A small, stable API; clear failure modes; no surprises in retry behaviour. |
| Project owner | Independence from any single provider; predictable cost (zero unless opted in). |
| Sibling projects (`corpusforge`, `batterygemma`) | A dependency with a narrow, versioned public API. |
