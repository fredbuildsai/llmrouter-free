# 11 Risks and technical debt

| # | Item | Impact | Mitigation / plan |
|---|---|---|---|
| 1 | Free-tier providers change limits, model names or retire models without notice | Configs go stale | Keep provider specifics in YAML, not code; `status` and the ledger show what is failing |
| 2 | Cooldown and disabled state are process memory | A restart forgets a 24 h quota cooldown and re-probes once | Ledger already persists usage; persisting cooldowns is on the roadmap |
| 3 | Daily counting assumes this router is the key's only consumer | Undercount if the key is shared | Document; provider 429 still triggers `quota` |
| 4 | Synchronous only | Cannot exploit async event loops | Roadmap: `acomplete` |
| 5 | Depends on LiteLLM's exception shapes for classification | A LiteLLM change could misclassify | `_classify_error` is small and unit-tested; falls back to `error` |
| 6 | Name `llmrouter-free` may already exist on PyPI | Blocks a future PyPI publish | Check availability before publishing; GitHub install unaffected |
| 7 | Cost tracking only for deployments that declare prices | Paid guard blind for undeclared models | Require prices on every `tier: paid` deployment (documented) |
