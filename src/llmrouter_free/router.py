"""Quota-aware teacher-LLM router with automatic failover.

Each task route lists deployments in preference order. For every call the router walks that list and
skips deployments that are disabled (no API key, auth failure, model not in the subscription tier), paid
while paid use is off or over budget, in an excluded model family or model (judge != generator), cooling
down after a 429/error, or over their per-minute or per-day request limits.

Deployments that share a provider budget (e.g. every model in one Mistral workspace) declare the same
`rate_group`: per-minute windows, per-day counts and rate-limit cooldowns are then shared, so a 429 on one
model is not immediately retried on a sibling that draws from the same exhausted budget. When every
candidate is only temporarily blocked, the router waits up to `max_wait_seconds` and retries.

Every attempt is recorded in `llm_calls`, which doubles as the daily quota ledger and as a response cache
keyed by prompt hash.

When only one provider family is configured, a judge call can opt into `allow_same_family_fallback`: if no
other family is usable, the router retries with a different model from the same family and marks the result
`relaxed_family=True` so the weaker independence can be recorded with the judgement.

A deployment can set `api_base` (override the provider's default endpoint) and `extra_body` (passed through
verbatim to the underlying OpenAI-compatible request, e.g. `{"chat_template_kwargs": {"thinking": true}}` for
a reasoning model). Such a model may return its chain of thought on a separate `reasoning`/`reasoning_content`
field instead of `content`; the router captures that into `LLMResult.reasoning` and persists it alongside the
cached response. If a reasoning model spends its whole token budget thinking and returns empty `content`, that
counts as `invalid_output` (fails over to the next deployment) rather than being cached as a valid empty
answer; `min_max_tokens` lets a deployment request a higher floor than the caller's default to reduce that risk.

Thread-safe for concurrent `complete()` calls from multiple worker threads (e.g. a host application's thread pool):
a single `threading.Lock` guards the in-memory selection/bookkeeping state (`_cooldown_until`, `_disabled`,
`_recent`'s per-minute windows) that `_next_deployment`/`_attempt` read and mutate, so two threads can't race
selecting the same rpm slot or missing each other's cooldown update. The lock is released before the actual
network request (`_completion()(**kwargs)`) and before any DB write, so real work still happens in parallel -
it only serializes the brief in-memory decisions around it. DB writes (`_log`/`_record_metric`, via
`get_session`) need no extra locking: each opens its own SQLAlchemy session/connection, and SQLite
engines should be configured WAL + a 30s busy_timeout (see `store.register_sqlite_pragmas`) so concurrent
writers queue briefly instead of failing.
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, func, select

from llmrouter_free.store import (
    LLMCall,
    get_session,
    init_ledger,
    init_metrics,
    make_ephemeral_engine,
    record_llm_call_metric,
)

logger = logging.getLogger(__name__)

Message = dict[str, Any]
Validator = Callable[[str], Any]

DISABLING_OUTCOMES = {"auth", "unavailable"}
GROUP_COOLDOWN_OUTCOMES = {"rate_limited", "quota"}


class AllDeploymentsExhausted(RuntimeError):
    """No deployment on the route could produce a valid response."""


@dataclass
class Deployment:
    name: str
    model: str
    api_key_env: str
    tier: str = "free"
    family: str = ""
    rate_group: str | None = None
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tags: list[str] = field(default_factory=list)
    cost_per_mtok_in: float = 0.0
    cost_per_mtok_out: float = 0.0
    api_base: str | None = None  # override the provider's default base URL
    extra_body: dict[str, Any] = field(default_factory=dict)  # passed through verbatim, e.g. chat_template_kwargs
    min_max_tokens: int | None = None  # floor on the requested max_tokens, for models that spend budget thinking
    temperature: float | None = None  # overrides the call's temperature for this deployment specifically -
                                       # e.g. Nemotron confirmed live (2026-09-16) to fabricate a comparison's
                                       # direction and misattribute a fact's material more often at high
                                       # temperature; 0.4 fixed both in repeated testing with no loss of
                                       # completeness. None means "use whatever the caller passed".

    @property
    def is_local(self) -> bool:
        """True for a local Ollama deployment - used to cap cloud retry attempts before falling back (see
        LLMRouter.max_cloud_attempts), not something a deployment declares explicitly in config."""
        return self.model.startswith("ollama_chat/")

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    @property
    def group(self) -> str:
        return self.rate_group or self.name

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.cost_per_mtok_in + tokens_out * self.cost_per_mtok_out) / 1_000_000


@dataclass
class LLMResult:
    text: str
    parsed: Any
    deployment: str
    model: str
    family: str
    reasoning: str | None = None  # the model's thinking/reasoning_content, when the deployment requests it
    cached: bool = False
    relaxed_family: bool = False  # judge came from the generator's family (different model) as a fallback
    tokens_in: int = 0  # prompt tokens - 0 for a cache hit whose original LLMCall row predates this field
    tokens_out: int = 0  # completion tokens


def _utc_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _classify_error(exc: Exception) -> str:
    """Map provider exceptions (LiteLLM wraps them with status codes) to router outcomes."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "tier_not_allowed" in text or "not available in your subscription" in text:
        return "unavailable"
    if status in (401, 403) or "authentication" in name or "permissiondenied" in name:
        return "auth"
    if status == 429 or "ratelimit" in name or '"raw_status_code":429' in text:
        daily_markers = ("per day", "daily", "quota", "requests per day", "rpd", "exceeded your current")
        return "quota" if any(m in text for m in daily_markers) else "rate_limited"
    return "error"


class LLMRouter:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        engine: Engine | None = None,
        metrics_engine: Engine | None = None,
        allow_paid: bool = False,
        max_usd_per_day: float = 0.0,
        completion_fn: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.deployments = {d["name"]: Deployment(**d) for d in config["deployments"]}
        self.routes: dict[str, list[str]] = config["routes"]
        unknown = {n for chain in self.routes.values() for n in chain} - self.deployments.keys()
        if unknown:
            raise ValueError(f"Routes reference unknown deployments: {sorted(unknown)}")
        cooldown = config.get("cooldown", {})
        self.cooldown_rate = cooldown.get("rate_limit_seconds", 60)
        self.cooldown_quota = cooldown.get("daily_quota_seconds", 86400)
        self.cooldown_error = cooldown.get("error_seconds", 30)
        self.max_attempts = config.get("max_attempts_per_call", 6)
        self.max_cloud_attempts = config.get("max_cloud_attempts")  # None = no cap (try cloud as usual)
        # A transient failure (error/invalid_output) on a deployment retries that SAME deployment this many
        # times (each retry pays its cooldown_error wait, so it's not a tight loop) before moving on to the
        # next deployment in the route - useful when the first deployment is meaningfully more reliable than
        # the fallbacks (confirmed live: nemotron-super mostly succeeds while glm-5.2/qwen3.8-27b are
        # frequently 429'd on OpenRouter's free tier, so falling back to them on nemotron's first transient
        # error wastes an attempt on a deployment less likely to succeed anyway). 1 = today's behavior
        # (move on immediately). A rate-limited deployment already gets this for free via its own cooldown
        # wait - this setting is specifically for "error"/"invalid_output" outcomes.
        self.max_attempts_per_deployment = config.get("max_attempts_per_deployment", 1)
        self.request_timeout_seconds = config.get("request_timeout_seconds", 60)
        # The ledger/cache. Without an engine the router keeps a private temp-file one: ledger + cache still work
        # within this process, nothing persists. Tables are created idempotently, so an existing database that
        # already has `llm_calls` (e.g. one written by an older host application) is reused as-is.
        self.engine = engine if engine is not None else make_ephemeral_engine()
        init_ledger(self.engine)
        # Optional analytics sink; None disables per-call metric recording entirely.
        self.metrics_engine = metrics_engine
        if metrics_engine is not None:
            init_metrics(metrics_engine)
        self.allow_paid = allow_paid
        self.max_usd_per_day = max_usd_per_day
        self._completion_fn = completion_fn
        self._clock = clock
        self._sleep = sleep
        self._cooldown_until: dict[str, float] = {}  # keyed by deployment name or rate group
        self._disabled: set[str] = set()
        self._recent: dict[str, deque[float]] = {}  # per rate group request timestamps
        self._group_members: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()  # guards the three dicts/set above - see module docstring
        for d in self.deployments.values():
            self._group_members[d.group].append(d.name)

    # --- public API ------------------------------------------------------------------------------

    def complete(
        self,
        route: str,
        messages: Sequence[Message],
        *,
        exclude_families: Sequence[str] = (),
        exclude_models: Sequence[str] = (),
        allow_same_family_fallback: bool = False,
        validate: Validator | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        response_format: dict | None = None,
        use_cache: bool = True,
        max_wait_seconds: float = 120.0,
    ) -> LLMResult:
        """Return the first valid response along the route, failing over as needed.

        `validate` receives the raw text and returns the parsed value or raises ValueError, in which
        case the attempt is logged as invalid_output and the next deployment is tried. When every candidate
        is only temporarily blocked (per-minute window or rate-limit cooldown), the call waits up to
        `max_wait_seconds` in total before giving up.

        `response_format` (see `llm.schemas.json_schema_response_format`) requests schema-constrained
        decoding instead of the generic `json_mode=True` object mode, when the caller has verified the
        route's deployments support it. Takes precedence over `json_mode` when given.
        """
        if route not in self.routes:
            raise KeyError(f"Unknown route: {route}")
        params = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode,
                 "response_format": response_format}
        prompt_hash = self._hash(route, messages, params)

        phases: list[tuple[tuple[str, ...], bool]] = [(tuple(exclude_families), False)]
        if allow_same_family_fallback and exclude_families:
            phases.append(((), True))

        attempts = 0
        waited = 0.0
        for families, relaxed in phases:
            if use_cache and (cached := self._cached(prompt_hash, families, exclude_models, validate)):
                cached.relaxed_family = relaxed
                return cached
            tried: set[str] = set()
            deployment_attempts: dict[str, int] = {}
            phase_attempts = 0
            cloud_attempts = 0
            switched_to_local = False
            while phase_attempts < self.max_attempts:
                deployment, wait = self._next_deployment(route, families, exclude_models, tried)
                if deployment is None:
                    if wait is None or waited + wait > max_wait_seconds:
                        break
                    self._sleep(wait)
                    waited += wait
                    continue
                phase_attempts += 1
                attempts += 1
                result, outcome = self._attempt(route, deployment, messages, params, prompt_hash, validate)
                if result is not None:
                    result.relaxed_family = relaxed
                    return result
                if outcome == "rate_limited":
                    pass  # rate-limited deployments may be retried once they cool down - never counts here
                elif outcome in GROUP_COOLDOWN_OUTCOMES or outcome in DISABLING_OUTCOMES:
                    # "quota" (daily limit) and auth/unavailable never benefit from an immediate same-
                    # deployment retry - exclude for the rest of this call right away, same as before.
                    tried.add(deployment.name)
                else:
                    # "error"/"invalid_output": worth retrying the SAME deployment - up to
                    # max_attempts_per_deployment times - before falling back to a weaker one. `_attempt`
                    # already set a cooldown_error wait on this deployment; clear it so `_next_deployment`
                    # can pick it again immediately rather than skipping ahead to the next deployment in
                    # the route while this one "cools down" (confirmed live: without this, one transient
                    # nemotron-super error jumped straight to a less-reliable free-tier fallback instead of
                    # giving nemotron another try first).
                    deployment_attempts[deployment.name] = deployment_attempts.get(deployment.name, 0) + 1
                    if deployment_attempts[deployment.name] >= self.max_attempts_per_deployment:
                        tried.add(deployment.name)
                    else:
                        with self._lock:
                            self._cooldown_until.pop(deployment.name, None)
                if not deployment.is_local:
                    cloud_attempts += 1
                    route_has_local = any(self.deployments[n].is_local for n in self.routes[route])
                    if (
                        self.max_cloud_attempts is not None and cloud_attempts >= self.max_cloud_attempts
                        and not switched_to_local and route_has_local
                    ):
                        # Only worth cutting cloud short if there's an actual local fallback waiting - a
                        # route with no local deployment at all (e.g. extract, cloud-only as of 2026-09-21)
                        # must keep trying every remaining cloud deployment instead: marking them all
                        # "tried" here would exhaust the route after just `max_cloud_attempts` attempts and
                        # never reach a 3rd/4th all-cloud fallback (confirmed live - openrouter-qwen3.8-27b
                        # was never attempted after nemotron-super and glm-5.2 both failed).
                        cloud_names = [d.name for d in self.deployments.values()
                                      if not d.is_local and d.name in self.routes[route]]
                        tried.update(cloud_names)  # stop retrying/waiting on cloud, jump straight to local
                        switched_to_local = True
                        logger.warning(
                            "route %r: %d cloud attempt(s) exhausted, switching to local fallback",
                            route, cloud_attempts, extra={"context": {"route": route, "cloud_attempts": cloud_attempts}},
                        )
        message = (
            f"Route '{route}' exhausted after {attempts} attempts and {waited:.0f}s waiting "
            f"(excluded families: {list(exclude_families)}, excluded models: {list(exclude_models)})"
        )
        logger.error(message, extra={"context": {"route": route, "attempts": attempts, "waited_s": waited}})
        raise AllDeploymentsExhausted(message)

    def status(self) -> list[dict[str, Any]]:
        used = self._requests_today()
        now = self._clock()
        rows = []
        for d in self.deployments.values():
            rows.append(
                {
                    "name": d.name,
                    "model": d.model,
                    "tier": d.tier,
                    "family": d.family,
                    "rate_group": d.group,
                    "enabled": d.enabled and d.name not in self._disabled,
                    "used_today": used.get(d.name, 0),
                    "rpd": d.rpd,
                    "cooldown_s": max(0, int(self._cooldown_remaining(d, now))),
                }
            )
        return rows

    def spent_today_usd(self) -> float:
        with get_session(self.engine) as s:
            total = s.scalar(
                select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                    LLMCall.tier == "paid", LLMCall.created_at >= _utc_midnight()
                )
            )
        return float(total or 0.0)

    # --- selection -------------------------------------------------------------------------------

    def _cooldown_remaining(self, d: Deployment, now: float) -> float:
        until = max(self._cooldown_until.get(d.name, 0.0), self._cooldown_until.get(d.group, 0.0))
        return until - now

    def _next_deployment(
        self, route: str, exclude_families: Sequence[str], exclude_models: Sequence[str], tried: set[str]
    ) -> tuple[Deployment | None, float | None]:
        """Pick the first usable deployment, or return how long until a temporarily blocked one frees up.

        Reserves the picked deployment's rpm slot (appends to `_recent`) before returning, under `_lock` -
        selection and reservation must be one atomic step, or two concurrent threads can both see room in
        the same rpm window and both proceed, over-running the real per-minute limit.
        """
        now = self._clock()
        used_today = self._requests_today()
        min_wait: float | None = None

        def consider_wait(seconds: float) -> None:
            nonlocal min_wait
            min_wait = seconds if min_wait is None else min(min_wait, seconds)

        with self._lock:
            for name in self.routes[route]:
                d = self.deployments[name]
                if name in tried or name in self._disabled or not d.enabled:
                    continue
                if (d.family and d.family in exclude_families) or d.model in exclude_models:
                    continue
                if d.tier == "paid" and (not self.allow_paid or self.spent_today_usd() >= self.max_usd_per_day):
                    continue
                if d.rpd is not None and sum(used_today.get(m, 0) for m in self._group_members[d.group]) >= d.rpd:
                    continue
                if (remaining := self._cooldown_remaining(d, now)) > 0:
                    consider_wait(remaining)
                    continue
                if d.rpm is not None:
                    window = self._recent.setdefault(d.group, deque())
                    while window and now - window[0] >= 60:
                        window.popleft()
                    if len(window) >= d.rpm:
                        consider_wait(60 - (now - window[0]))
                        continue
                    window.append(now)  # reserve the slot now, atomically with the selection above
                return d, None
            return None, min_wait

    def _requests_today(self) -> dict[str, int]:
        with get_session(self.engine) as s:
            rows = s.execute(
                select(LLMCall.deployment, func.count())
                .where(LLMCall.created_at >= _utc_midnight(), LLMCall.status != "cache_hit")
                .group_by(LLMCall.deployment)
            ).all()
        return {name: count for name, count in rows}

    # --- execution -------------------------------------------------------------------------------

    def _attempt(
        self,
        route: str,
        d: Deployment,
        messages: Sequence[Message],
        params: dict[str, Any],
        prompt_hash: str,
        validate: Validator | None,
    ) -> tuple[LLMResult | None, str]:
        # rpm-window reservation now happens atomically in `_next_deployment` (see its docstring) - doing
        # it again here would double-count every call against a deployment's per-minute limit.
        max_tokens = max(params["max_tokens"], d.min_max_tokens or 0)
        kwargs: dict[str, Any] = {
            "model": d.model,
            "messages": list(messages),
            "temperature": d.temperature if d.temperature is not None else params["temperature"],
            "max_tokens": max_tokens,
            "api_key": os.environ.get(d.api_key_env),
            "timeout": self.request_timeout_seconds,
        }
        if d.api_base:
            kwargs["api_base"] = d.api_base
        if d.extra_body:
            kwargs["extra_body"] = d.extra_body
        if params["response_format"] is not None:
            kwargs["response_format"] = params["response_format"]
        elif params["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}

        started_wall = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            response = self._completion()(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            outcome = _classify_error(exc)
            latency = int((time.perf_counter() - started) * 1000)
            with self._lock:
                if outcome in DISABLING_OUTCOMES:
                    self._disabled.add(d.name)
                else:
                    seconds = {"quota": self.cooldown_quota, "rate_limited": self.cooldown_rate}.get(
                        outcome, self.cooldown_error
                    )
                    key = d.group if outcome in GROUP_COOLDOWN_OUTCOMES else d.name
                    self._cooldown_until[key] = self._clock() + seconds
            self._log(route, d, prompt_hash, status=outcome, error=str(exc)[:2000], latency_ms=latency)
            self._record_metric(started_wall, route, d, outcome, latency, error=str(exc)[:2000])
            level = logging.ERROR if outcome in DISABLING_OUTCOMES else logging.WARNING
            logger.log(level, "%s on route %r via %s: %s", outcome, route, d.name, str(exc)[:300],
                      extra={"context": {"route": route, "deployment": d.name, "model": d.model,
                             "outcome": outcome, "latency_ms": latency}})
            return None, outcome

        latency = int((time.perf_counter() - started) * 1000)
        message = response.choices[0].message
        text = message.content or ""
        # Reasoning models (e.g. NVIDIA-hosted DeepSeek with chat_template_kwargs.thinking) return their
        # chain of thought on a separate field rather than in `content`; the field name varies by provider.
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or None
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        cost = d.cost(tokens_in, tokens_out) if d.tier == "paid" else 0.0

        if not text.strip():
            # A reasoning model can spend its whole max_tokens budget on `reasoning` and return no answer;
            # that is not a usable response, so fail over rather than caching an empty string as "ok".
            detail = "empty content (reasoning present)" if reasoning else "empty content"
            self._log(
                route, d, prompt_hash, status="invalid_output", error=detail, latency_ms=latency,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, reasoning_text=reasoning,
            )
            self._record_metric(started_wall, route, d, "invalid_output", latency, response=response,
                               cost=cost, error=detail)
            return None, "invalid_output"

        try:
            parsed = validate(text) if validate else text
        except ValueError as exc:
            self._log(
                route, d, prompt_hash, status="invalid_output", error=str(exc)[:2000], latency_ms=latency,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost, reasoning_text=reasoning,
            )
            self._record_metric(started_wall, route, d, "invalid_output", latency, response=response,
                               cost=cost, error=str(exc)[:2000])
            return None, "invalid_output"

        self._log(
            route, d, prompt_hash, status="ok", latency_ms=latency, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=cost, response_text=text, reasoning_text=reasoning,
        )
        self._record_metric(started_wall, route, d, "ok", latency, response=response, cost=cost)
        return (
            LLMResult(text=text, parsed=parsed, deployment=d.name, model=d.model, family=d.family,
                     reasoning=reasoning, tokens_in=tokens_in, tokens_out=tokens_out),
            "ok",
        )

    def _cached(
        self,
        prompt_hash: str,
        exclude_families: Sequence[str],
        exclude_models: Sequence[str],
        validate: Validator | None,
    ) -> LLMResult | None:
        with get_session(self.engine) as s:
            rows = s.scalars(
                select(LLMCall)
                .where(LLMCall.prompt_hash == prompt_hash, LLMCall.status == "ok")
                .order_by(LLMCall.created_at.desc())
            ).all()
        for row in rows:
            d = self.deployments.get(row.deployment)
            family = d.family if d else ""
            if (family and family in exclude_families) or row.model in exclude_models:
                continue
            try:
                parsed = validate(row.response_text or "") if validate else row.response_text
            except ValueError:
                continue
            return LLMResult(
                text=row.response_text or "", parsed=parsed, deployment=row.deployment, model=row.model,
                family=family, reasoning=row.reasoning_text, cached=True,
                tokens_in=row.tokens_in, tokens_out=row.tokens_out,
            )
        return None

    def _completion(self) -> Callable[..., Any]:
        if self._completion_fn is None:
            import litellm  # imported lazily: slow import, not needed for tests or status

            litellm.suppress_debug_info = True
            self._completion_fn = litellm.completion
        return self._completion_fn

    def _log(self, route: str, d: Deployment, prompt_hash: str, *, status: str, **fields: Any) -> None:
        with get_session(self.engine) as s:
            s.add(LLMCall(route=route, deployment=d.name, model=d.model, tier=d.tier, prompt_hash=prompt_hash,
                          status=status, **fields))

    def _record_metric(
        self, started_wall: datetime, route: str, d: Deployment, status: str, latency_ms: int, *,
        response: Any = None, cost: float | None = None, error: str | None = None,
    ) -> None:
        """Best-effort: analytics capture must never break a real pipeline call, so any failure here (e.g.
        the metrics DB being momentarily locked) is swallowed, not raised."""
        if self.metrics_engine is None:
            return
        try:
            record_llm_call_metric(
                self.metrics_engine, started_at=started_wall, route=route, deployment=d.name, requested_model=d.model,
                is_local=d.is_local, status=status, latency_ms=latency_ms, response=response,
                cost_usd=cost, error=error,
            )
        except Exception:  # noqa: BLE001
            logger.warning("failed to record LLM call metric for %s/%s", route, d.name, exc_info=True)

    @staticmethod
    def _hash(route: str, messages: Sequence[Message], params: dict[str, Any]) -> str:
        payload = json.dumps({"route": route, "messages": list(messages), "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
