# 12 Glossary

| Term | Meaning |
|---|---|
| **Route** | Named, ordered list of deployments used for one kind of task. |
| **Deployment** | One concrete way to reach a model: provider + model id + key variable + limits. |
| **Rate group** | Deployments that draw on one provider budget; they share rpm/rpd windows and 429 cooldowns. |
| **Family** | Model lineage (e.g. `nemotron`, `gpt-oss`); used to keep a judge independent of its generator. |
| **Outcome** | Classification of one attempt: `ok`, `invalid_output`, `rate_limited`, `quota`, `error`, `auth`, `unavailable`. |
| **Cooldown** | Period during which a deployment (or its rate group) is not selected. |
| **Ledger** | The `llm_calls` table: every attempt, doubling as cache and daily-usage counter. |
| **rpm / rpd** | Requests per minute / per day. |
| **Slot** | One unit of a deployment's per-minute allowance; reserved atomically at selection. |
| **Local deployment** | A model served by Ollama (`ollama_chat/...`). |
| **`num_ctx`** | Ollama's context-window size; changing it between calls forces a model reload. |
| **Relaxed family** | A judge result produced by a same-family (different model) fallback, flagged `relaxed_family=True`. |
