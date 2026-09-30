"""Size the local Ollama context window (`num_ctx`) once, from known configuration bounds.

Why this exists: Ollama reloads the model whenever `num_ctx` changes between consecutive requests (~6-7 s, not
a full cold start, but real cost on every call if `num_ctx` were recomputed per prompt). So the goal is *one*
static value good for every task, computed once at startup - not a per-call estimate that would trigger an
internal reload each time it moves between buckets.

The arithmetic is pure: the worst-case prompt + output size of a task is knowable in advance when the input
chunk size is capped and each call site's output budget (`max_tokens`) is a fixed constant:

    worst_case = items * (max_item_tokens + output_tokens_per_item) + template_overhead
    num_ctx    = round_up_to_bucket(worst_case * (1 + buffer))

The same "would this fit?" arithmetic (`fits_within`) is the right way to sanity-check a cloud deployment's
documented context limit, rather than discovering an overflow as a truncated response.

This module knows nothing about any particular application's prompts: the host describes its tasks as a
`{name: TaskBudget}` registry and calls `recommend_num_ctx`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

TokenCounter = Callable[[str], int]

DEFAULT_BUFFER = 0.20  # safety margin on top of the computed worst case
CONTEXT_BUCKET = 1024  # round up to a clean number; irrelevant to correctness, just tidier config/logs


@dataclass(frozen=True)
class TaskBudget:
    """One task's fixed prompt scaffolding and per-item output budget.

    `rendered_template` is the user template with every placeholder already substituted by an empty string, so
    only the literal instructions are counted (a raw template with `{...}` JSON-shape examples would otherwise
    be mistaken for format placeholders). `batchable` marks tasks whose single call bundles `batch_size` input
    items (chunk text *and* output budget scale with it); every other task is exactly one item per call.
    """

    system_prompt: str
    rendered_template: str
    output_tokens: int
    batchable: bool = False


def round_up_to_bucket(value: int, bucket: int = CONTEXT_BUCKET) -> int:
    return -(-value // bucket) * bucket


def measure_template_overhead(system_prompt: str, user_template: str, count_tokens: TokenCounter) -> int:
    """Token cost of a task's fixed prompt scaffolding alone (variable content removed)."""
    return count_tokens(system_prompt) + count_tokens(user_template)


def compute_num_ctx(
    *, chunk_max_tokens: int, template_overhead_tokens: int, output_tokens: int, buffer: float = DEFAULT_BUFFER,
    bucket: int = CONTEXT_BUCKET,
) -> int:
    """The static context window one task actually needs, worst case, with `buffer` extra headroom."""
    worst_case = chunk_max_tokens + template_overhead_tokens + output_tokens
    return round_up_to_bucket(round(worst_case * (1 + buffer)), bucket)


def global_num_ctx(
    *, chunk_max_tokens: int, template_overhead_tokens: int, output_tokens_by_task: dict[str, int],
    buffer: float = DEFAULT_BUFFER, bucket: int = CONTEXT_BUCKET,
) -> int:
    """One context window sized for the largest task, so every local Ollama deployment can share a single
    `num_ctx` across every route it serves - it then never changes between calls and never triggers a reload."""
    return compute_num_ctx(
        chunk_max_tokens=chunk_max_tokens, template_overhead_tokens=template_overhead_tokens,
        output_tokens=max(output_tokens_by_task.values()), buffer=buffer, bucket=bucket,
    )


def fits_within(context_limit: int, *, chunk_max_tokens: int, template_overhead_tokens: int, output_tokens: int) -> bool:
    """Whether a deployment's real context limit (e.g. a cloud model's documented window) comfortably covers
    the worst case for one task."""
    return context_limit >= chunk_max_tokens + template_overhead_tokens + output_tokens


def worst_case_batch_size(batch_size: int, tasks: dict[str, TaskBudget]) -> dict[str, int]:
    """Per-task item multiplier: `batch_size` for a batchable task, 1 for everything else - so
    `batch_size=1` degenerates back to plain one-item-per-call sizing."""
    return {name: batch_size if task.batchable else 1 for name, task in tasks.items()}


def recommend_num_ctx(
    *, tasks: dict[str, TaskBudget], chunk_max_tokens: int, count_tokens: TokenCounter, batch_size: int = 1
) -> dict[str, Any]:
    """The full sizing report: measured overhead and recommended `num_ctx` per task, plus the single global
    value (sized for the largest task) that every local deployment should use in practice.

    Only batchable tasks bundle `batch_size` items' text and output budget into one call; every other task is
    always exactly one item per call regardless of `batch_size`.
    """
    overheads = {
        name: measure_template_overhead(t.system_prompt, t.rendered_template, count_tokens) for name, t in tasks.items()
    }
    multipliers = worst_case_batch_size(batch_size, tasks)
    per_task = {
        name: compute_num_ctx(
            chunk_max_tokens=chunk_max_tokens * multipliers[name],
            template_overhead_tokens=overheads[name],
            output_tokens=tasks[name].output_tokens * multipliers[name],
        )
        for name in tasks
    }
    return {
        "chunk_max_tokens": chunk_max_tokens, "batch_size": batch_size, "overheads": overheads,
        "output_tokens": {name: t.output_tokens for name, t in tasks.items()},
        "per_task_num_ctx": per_task, "global_num_ctx": max(per_task.values()),
    }


def apply_global_num_ctx(config: dict[str, Any], num_ctx: int, *, model_prefix: str = "ollama_chat/") -> dict[str, Any]:
    """Return a copy of a router config with `num_ctx` set on every deployment whose model uses `model_prefix`
    (i.e. every local Ollama deployment). Call once at startup, before any deployment is used, so `num_ctx` is
    fixed for the life of the process rather than guessed in a YAML comment or recomputed per call."""
    patched = dict(config)
    patched["deployments"] = [
        {**d, "extra_body": {**d.get("extra_body", {}), "options": {**d.get("extra_body", {}).get("options", {}), "num_ctx": num_ctx}}}
        if d.get("model", "").startswith(model_prefix) else d
        for d in config["deployments"]
    ]
    return patched
