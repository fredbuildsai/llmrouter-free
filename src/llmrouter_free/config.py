"""Loading a router config (`llm_routes.yaml`) and constructing a ready-to-use `LLMRouter` from it."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Engine

from llmrouter_free.context_budget import TaskBudget, TokenCounter, apply_global_num_ctx, recommend_num_ctx
from llmrouter_free.router import LLMRouter

TEMPLATE_PATH = Path(__file__).parent / "templates" / "llm_routes.yaml"


def load_routes(path: str | Path) -> dict[str, Any]:
    """Read a router config file. See `templates/llm_routes.yaml` for every supported key."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_request_timeout(config: Mapping[str, Any], factor: int, *, default: float = 150) -> dict[str, Any]:
    """Copy of `config` with `request_timeout_seconds` multiplied by `factor`.

    A call that bundles `factor` items (a batched extraction) legitimately takes about `factor` times longer to
    generate than a single-item call; a fixed single-item timeout silently starves it (confirmed live: a 150 s
    timeout was too short for a batch-of-5 call that fell over to a local model).
    """
    return {**config, "request_timeout_seconds": config.get("request_timeout_seconds", default) * factor}


def build_router(
    config: Mapping[str, Any],
    *,
    engine: Engine | None = None,
    metrics_engine: Engine | None = None,
    allow_paid: bool = False,
    max_usd_per_day: float = 0.0,
    tasks: dict[str, TaskBudget] | None = None,
    chunk_max_tokens: int | None = None,
    count_tokens: TokenCounter | None = None,
    batch_size: int = 1,
) -> LLMRouter:
    """Construct a router, optionally running the context-budget sizing step first.

    When `tasks`, `chunk_max_tokens` and `count_tokens` are given, every local Ollama deployment gets a single
    precomputed `num_ctx` baked into its config (so it never changes across a run) and the per-attempt request
    timeout is scaled by `batch_size`. Without them the config is used as written.
    """
    config = dict(config)
    if tasks is not None:
        if chunk_max_tokens is None or count_tokens is None:
            raise ValueError("context sizing needs tasks, chunk_max_tokens and count_tokens together")
        report = recommend_num_ctx(
            tasks=tasks, chunk_max_tokens=chunk_max_tokens, count_tokens=count_tokens, batch_size=batch_size
        )
        config = apply_global_num_ctx(config, report["global_num_ctx"])
        config = scale_request_timeout(config, batch_size)
    return LLMRouter(
        config, engine=engine, metrics_engine=metrics_engine, allow_paid=allow_paid, max_usd_per_day=max_usd_per_day
    )
