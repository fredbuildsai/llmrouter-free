import json
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from llmrouter_free.router import AllDeploymentsExhausted, LLMRouter
from llmrouter_free.store import LLMCall, get_session


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


def response(text, tokens_in=10, tokens_out=5, reasoning=None):
    message = SimpleNamespace(content=text)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


CONFIG = {
    "deployments": [
        {"name": "a", "model": "p/a", "api_key_env": "KEY_A", "family": "fam1", "rpd": 100},
        {"name": "b", "model": "p/b", "api_key_env": "KEY_B", "family": "fam2", "rpd": 1},
        {"name": "c", "model": "p/c", "api_key_env": "KEY_C", "family": "fam3", "tier": "paid",
         "cost_per_mtok_in": 1_000_000.0, "cost_per_mtok_out": 0.0},
    ],
    "routes": {"qa": ["a", "b", "c"], "judge": ["a", "b"]},
    "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 30},
    "max_attempts_per_call": 5,
}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for key in ("KEY_A", "KEY_B", "KEY_C"):
        monkeypatch.setenv(key, "x")


def make_router(engine, behaviour, **kwargs):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        outcome = behaviour(kw["model"], len(calls))
        if isinstance(outcome, Exception):
            raise outcome
        return response(outcome)

    return LLMRouter(CONFIG, engine=engine, completion_fn=completion, sleep=lambda _: None, **kwargs), calls


def statuses(engine):
    with get_session(engine) as s:
        return [(c.deployment, c.status) for c in s.scalars(select(LLMCall).order_by(LLMCall.id))]


def test_rate_limit_fails_over_to_next_deployment(engine):
    router, calls = make_router(engine, lambda model, n: RateLimitError("slow down") if model == "p/a" else "ok")
    result = router.complete("qa", [{"role": "user", "content": "hi"}])
    assert result.deployment == "b"
    assert calls == ["p/a", "p/b"]
    assert statuses(engine) == [("a", "rate_limited"), ("b", "ok")]
    # 'a' is cooling down and 'b' used its single daily request, so a new prompt has nowhere to go
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "again"}], use_cache=False)
    assert calls == ["p/a", "p/b"]


def test_daily_quota_marker_gets_long_cooldown(engine):
    router, _ = make_router(
        engine, lambda model, n: RateLimitError("Quota exceeded: requests per day") if model == "p/a" else "ok"
    )
    router.complete("qa", [{"role": "user", "content": "hi"}])
    assert statuses(engine)[0] == ("a", "quota")
    assert router.status()[0]["cooldown_s"] > 80000


def test_rpd_limit_skips_exhausted_deployment_and_paid_is_off_by_default(engine):
    router, calls = make_router(engine, lambda model, n: RateLimitError("x") if model == "p/a" else "ok")
    router.complete("qa", [{"role": "user", "content": "first"}])  # uses b's only daily request
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "second"}], use_cache=False)
    assert "p/c" not in calls  # paid deployment never used without allow_paid


def test_paid_budget_is_enforced(engine):
    router, calls = make_router(
        engine, lambda model, n: "ok" if model == "p/c" else RateLimitError("x"), allow_paid=True, max_usd_per_day=5
    )
    router.complete("qa", [{"role": "user", "content": "one"}])  # costs $10 (10 tokens x $1/token)
    assert calls[-1] == "p/c"
    router._cooldown_until.clear()
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "two"}])
    assert router.spent_today_usd() == pytest.approx(10.0)


def test_judge_excludes_generator_family(engine):
    router, calls = make_router(engine, lambda model, n: "ok")
    result = router.complete("judge", [{"role": "user", "content": "grade"}], exclude_families=["fam1"])
    assert result.deployment == "b" and calls == ["p/b"]


def test_invalid_output_retries_on_next_model_and_cache_hits(engine):
    def validate(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("not json") from exc

    router, calls = make_router(engine, lambda model, n: "not json" if model == "p/a" else '{"q": 1}')
    messages = [{"role": "user", "content": "give json"}]
    first = router.complete("qa", messages, validate=validate)
    assert first.parsed == {"q": 1} and first.deployment == "b"
    second = router.complete("qa", messages, validate=validate)
    assert second.cached and second.parsed == {"q": 1}
    assert calls == ["p/a", "p/b"]  # no new provider call for the cached prompt


SINGLE_PROVIDER = {
    "deployments": [
        {"name": "gen", "model": "mistral/mistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
        {"name": "judge", "model": "mistral/magistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
    ],
    "routes": {"judge": ["gen", "judge"]},
}


def test_same_family_fallback_uses_a_different_model_and_flags_the_result(engine):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        return response("5")

    router = LLMRouter(SINGLE_PROVIDER, engine=engine, completion_fn=completion)
    messages = [{"role": "user", "content": "score this"}]
    generator = {"exclude_families": ["mistral"], "exclude_models": ["mistral/mistral-medium-latest"]}

    with pytest.raises(AllDeploymentsExhausted):
        router.complete("judge", messages, **generator)

    result = router.complete("judge", messages, allow_same_family_fallback=True, **generator)
    assert result.model == "mistral/magistral-medium-latest"
    assert result.relaxed_family
    assert calls == ["mistral/magistral-medium-latest"]


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


WORKSPACE = {
    "deployments": [
        {"name": "small", "model": "mistral/small", "api_key_env": "KEY_A", "family": "mistral", "rate_group": "ws"},
        {"name": "medium", "model": "mistral/medium", "api_key_env": "KEY_A", "family": "mistral", "rate_group": "ws"},
    ],
    "routes": {"qa": ["small", "medium"]},
    "cooldown": {"rate_limit_seconds": 60},
}


def test_rate_group_shares_cooldown_then_waits_and_retries_same_model(engine):
    clock = FakeClock()
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if len(calls) == 1:
            raise RateLimitError('{"message":"Rate limit exceeded","raw_status_code":429}')
        return response("ok")

    router = LLMRouter(WORKSPACE, engine=engine, completion_fn=completion, clock=clock, sleep=clock.sleep)
    result = router.complete("qa", [{"role": "user", "content": "hi"}], max_wait_seconds=120)

    # 'medium' shares the workspace budget, so it is not hammered after the 429; the router waits out the
    # shared cooldown and retries 'small'.
    assert calls == ["mistral/small", "mistral/small"]
    assert result.deployment == "small"
    assert clock.now == pytest.approx(1_060.0)


def test_rate_group_gives_up_when_wait_budget_is_too_small(engine):
    calls = []

    def always_rate_limited(**kw):
        calls.append(kw["model"])
        raise RateLimitError("slow down")

    router = LLMRouter(WORKSPACE, engine=engine, completion_fn=always_rate_limited, sleep=lambda _: None)
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "hi"}], max_wait_seconds=10)
    assert calls == ["mistral/small"]  # the sibling in the same rate group is never tried


def test_tier_not_allowed_disables_model_and_fails_over(engine):
    class APIError(Exception):
        status_code = 403

    tier_error = APIError('{"message":"This model is not available in your subscription tier","type":"tier_not_allowed"}')
    router, calls = make_router(engine, lambda model, n: tier_error if model == "p/a" else "ok")
    result = router.complete("qa", [{"role": "user", "content": "hi"}])
    assert result.deployment == "b"
    assert statuses(engine)[0] == ("a", "unavailable")
    assert not router.status()[0]["enabled"]


def test_auth_error_disables_deployment(engine):
    router, calls = make_router(engine, lambda model, n: AuthenticationError("bad key") if model == "p/a" else "ok")
    router.complete("qa", [{"role": "user", "content": "hi"}])
    assert not router.status()[0]["enabled"]


REASONING_MODEL = {
    "deployments": [
        {"name": "nvidia-deepseek", "model": "nvidia_nim/deepseek-ai/deepseek-v4-flash-0731",
         "api_key_env": "KEY_A", "family": "deepseek", "api_base": "https://integrate.api.nvidia.com/v1",
         "min_max_tokens": 8192, "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}},
        {"name": "fallback", "model": "p/fallback", "api_key_env": "KEY_B", "family": "llama"},
    ],
    "routes": {"qa": ["nvidia-deepseek", "fallback"]},
}


def test_api_base_and_extra_body_and_min_max_tokens_are_forwarded(engine):
    captured = {}

    def completion(**kw):
        captured.update(kw)
        return response("the answer", reasoning="thinking it through")

    router = LLMRouter(REASONING_MODEL, engine=engine, completion_fn=completion)
    result = router.complete("qa", [{"role": "user", "content": "hi"}], max_tokens=100)

    assert captured["api_base"] == "https://integrate.api.nvidia.com/v1"
    assert captured["extra_body"] == {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
    assert captured["max_tokens"] == 8192  # bumped from the caller's 100 up to the deployment's floor
    assert result.reasoning == "thinking it through"
    assert result.text == "the answer"


def test_response_format_takes_precedence_over_json_mode(engine):
    captured = {}

    def completion(**kw):
        captured.update(kw)
        return response('{"ok": true}')

    router = LLMRouter(REASONING_MODEL, engine=engine, completion_fn=completion)
    schema_format = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}, "strict": True}}
    router.complete("qa", [{"role": "user", "content": "hi"}], json_mode=True, response_format=schema_format,
                    use_cache=False)

    assert captured["response_format"] == schema_format  # not {"type": "json_object"}


def test_json_mode_still_works_when_response_format_is_not_given(engine):
    captured = {}

    def completion(**kw):
        captured.update(kw)
        return response('{"ok": true}')

    router = LLMRouter(REASONING_MODEL, engine=engine, completion_fn=completion)
    router.complete("qa", [{"role": "user", "content": "hi"}], json_mode=True, use_cache=False)

    assert captured["response_format"] == {"type": "json_object"}


DEPLOYMENT_TEMPERATURE = {
    "deployments": [
        {"name": "cool-model", "model": "p/cool", "api_key_env": "KEY_A", "family": "fam1", "temperature": 0.4},
        {"name": "default-model", "model": "p/default", "api_key_env": "KEY_B", "family": "fam2"},
    ],
    "routes": {"qa": ["cool-model", "default-model"]},
}


def test_deployment_temperature_overrides_the_calls_default(engine):
    captured = []

    def completion(**kw):
        captured.append(kw["temperature"])
        return response('{"ok": true}')

    router = LLMRouter(DEPLOYMENT_TEMPERATURE, engine=engine, completion_fn=completion)
    router.complete("qa", [{"role": "user", "content": "hi"}], temperature=0.9, use_cache=False)

    assert captured == [0.4]  # cool-model's own temperature, not the caller's 0.9


def test_deployment_without_a_temperature_falls_back_to_the_calls_value(engine, monkeypatch):
    monkeypatch.setenv("KEY_A", "")  # disable cool-model so default-model is picked
    captured = []

    def completion(**kw):
        captured.append(kw["temperature"])
        return response('{"ok": true}')

    router = LLMRouter(DEPLOYMENT_TEMPERATURE, engine=engine, completion_fn=completion)
    router.complete("qa", [{"role": "user", "content": "hi"}], temperature=0.9, use_cache=False)

    assert captured == [0.9]


CLOUD_LOCAL_CONFIG = {
    "deployments": [
        {"name": "cloud1", "model": "p/cloud1", "api_key_env": "KEY_A", "family": "fam1"},
        {"name": "cloud2", "model": "p/cloud2", "api_key_env": "KEY_B", "family": "fam2"},
        {"name": "local", "model": "ollama_chat/gemma4:e4b", "api_key_env": "KEY_A", "family": "gemma4"},
    ],
    "routes": {"extract": ["cloud1", "cloud2", "local"]},
    "max_attempts_per_call": 6,
    "max_cloud_attempts": 2,
}


def test_deployment_is_local_detects_ollama_chat_models():
    from llmrouter_free.router import Deployment

    assert Deployment(name="x", model="ollama_chat/gemma4:e4b", api_key_env="K").is_local is True
    assert Deployment(name="y", model="nvidia_nim/deepseek", api_key_env="K").is_local is False


def test_switches_to_local_after_max_cloud_attempts_even_though_local_is_last_in_chain(engine):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "ollama_chat/gemma4:e4b":
            return response("ok")
        raise RateLimitError("nope")

    router = LLMRouter(CLOUD_LOCAL_CONFIG, engine=engine, completion_fn=completion)
    result = router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert result.deployment == "local"
    # both cloud deployments were tried once each (max_cloud_attempts=2), then local - not endless cloud retries
    assert calls == ["p/cloud1", "p/cloud2", "ollama_chat/gemma4:e4b"]


ALL_CLOUD_CONFIG = {
    "deployments": [
        {"name": "cloud1", "model": "p/cloud1", "api_key_env": "KEY_A", "family": "fam1"},
        {"name": "cloud2", "model": "p/cloud2", "api_key_env": "KEY_B", "family": "fam2"},
        {"name": "cloud3", "model": "p/cloud3", "api_key_env": "KEY_C", "family": "fam3"},
    ],
    "routes": {"extract": ["cloud1", "cloud2", "cloud3"]},
    "max_attempts_per_call": 6,
    "max_cloud_attempts": 2,
}


class ServiceUnavailableError(Exception):
    """A generic transient failure - `_classify_error` falls through to \"error\" for this (no status_code,
    no rate-limit-shaped name/message)."""


def test_retries_the_same_deployment_before_moving_to_the_next_on_a_transient_error(engine):
    """max_attempts_per_deployment=3: a deployment that keeps erroring gets 3 tries before the router gives
    up on it and moves to the next one in the route - not 1, which would waste attempts falling back to a
    weaker/less-reliable deployment on the very first transient hiccup of a normally-reliable one."""
    config = {
        "deployments": [
            {"name": "strong", "model": "p/strong", "api_key_env": "KEY_A", "family": "fam1"},
            {"name": "weak", "model": "p/weak", "api_key_env": "KEY_B", "family": "fam2"},
        ],
        "routes": {"extract": ["strong", "weak"]},
        "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 0},
        "max_attempts_per_call": 10,
        "max_attempts_per_deployment": 3,
    }
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "p/strong" and calls.count("p/strong") <= 2:
            raise ServiceUnavailableError("transient")
        if kw["model"] == "p/strong":
            return response("ok")
        raise ServiceUnavailableError("should never reach weak")

    router = LLMRouter(config, engine=engine, completion_fn=completion)
    result = router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert result.deployment == "strong"
    # 2 failed attempts on "strong", then a 3rd that succeeds - "weak" is never touched
    assert calls == ["p/strong", "p/strong", "p/strong"]


def test_retry_on_the_same_deployment_does_not_wait_out_its_own_error_cooldown(engine):
    """Regression test for a real bug: `_attempt` sets a real cooldown_error wait (e.g. 30s) on a
    deployment after a transient error, which would otherwise make `_next_deployment` skip straight past
    it to the next deployment in the route (since it looks "cooling down") instead of retrying it - even
    though max_attempts_per_deployment said it should get another immediate try first. Confirmed live: a
    single nemotron-super error jumped straight to a less-reliable fallback instead of retrying nemotron."""
    config = {
        "deployments": [
            {"name": "strong", "model": "p/strong", "api_key_env": "KEY_A", "family": "fam1"},
            {"name": "weak", "model": "p/weak", "api_key_env": "KEY_B", "family": "fam2"},
        ],
        "routes": {"extract": ["strong", "weak"]},
        "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 30},
        "max_attempts_per_call": 10,
        "max_attempts_per_deployment": 3,
    }
    calls = []
    slept: list[float] = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "p/strong" and calls.count("p/strong") <= 2:
            raise ServiceUnavailableError("transient")
        if kw["model"] == "p/strong":
            return response("ok")
        raise ServiceUnavailableError("should never reach weak")

    router = LLMRouter(config, engine=engine, completion_fn=completion, sleep=slept.append)
    result = router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert result.deployment == "strong"
    assert calls == ["p/strong", "p/strong", "p/strong"]
    assert slept == []  # no waiting out the 30s error cooldown between same-deployment retries


def test_gives_up_on_a_deployment_after_max_attempts_per_deployment_and_moves_on(engine):
    config = {
        "deployments": [
            {"name": "strong", "model": "p/strong", "api_key_env": "KEY_A", "family": "fam1"},
            {"name": "weak", "model": "p/weak", "api_key_env": "KEY_B", "family": "fam2"},
        ],
        "routes": {"extract": ["strong", "weak"]},
        "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 0},
        "max_attempts_per_call": 10,
        "max_attempts_per_deployment": 3,
    }
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "p/strong":
            raise ServiceUnavailableError("always fails")
        return response("ok")

    router = LLMRouter(config, engine=engine, completion_fn=completion)
    result = router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert result.deployment == "weak"
    assert calls == ["p/strong", "p/strong", "p/strong", "p/weak"]


def test_concurrent_calls_from_multiple_threads_do_not_crash_or_double_count_rpm(engine):
    """`bg annotate --concurrency N` runs several batches through one shared LLMRouter from worker threads -
    this is a smoke test for the locking added around `_next_deployment`/`_attempt`'s shared in-memory state
    (`_cooldown_until`, `_disabled`, `_recent`). Each thread uses a distinct prompt (via a per-thread nonce
    in the message) so the response cache doesn't short-circuit most of them down to one real call - the
    point is exercising concurrent selection/reservation, not caching."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    config = {
        "deployments": [{"name": "a", "model": "p/a", "api_key_env": "KEY_A", "family": "fam1", "rpm": 100}],
        "routes": {"extract": ["a"]},
        "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 30},
        "max_attempts_per_call": 5,
    }
    seen_models = []
    lock = threading.Lock()

    def completion(**kw):
        time.sleep(0.01)  # encourage thread interleaving around the router's locked sections
        with lock:
            seen_models.append(kw["model"])
        return response("ok")

    router = LLMRouter(config, engine=engine, completion_fn=completion)

    def call(i: int):
        return router.complete("extract", [{"role": "user", "content": f"hi {i}"}], use_cache=False)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(call, range(20)))

    assert len(results) == 20
    assert all(r.deployment == "a" for r in results)
    assert len(seen_models) == 20  # every call actually reached the provider, none lost/duplicated


def test_all_cloud_route_still_tries_every_deployment_past_max_cloud_attempts(engine, monkeypatch):
    """A route with no local deployment at all must not be cut short after `max_cloud_attempts` - that cap
    exists to jump to a *local* fallback sooner, not to give up on remaining cloud deployments. Regression
    test for a real bug: with extract=[nemotron-super, glm-5.2, qwen3.8-27b] (no local fallback) and
    max_cloud_attempts=2, qwen3.8-27b was never attempted after the first two failed."""
    monkeypatch.setenv("KEY_C", "x")
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "p/cloud3":
            return response("ok")
        raise RateLimitError("nope")

    router = LLMRouter(ALL_CLOUD_CONFIG, engine=engine, completion_fn=completion)
    result = router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert result.deployment == "cloud3"
    assert calls == ["p/cloud1", "p/cloud2", "p/cloud3"]


def test_logs_a_warning_when_switching_to_local(engine, caplog):
    import logging

    def completion(**kw):
        if kw["model"] == "ollama_chat/gemma4:e4b":
            return response("ok")
        raise RateLimitError("nope")

    router = LLMRouter(CLOUD_LOCAL_CONFIG, engine=engine, completion_fn=completion)
    with caplog.at_level(logging.WARNING, logger="llmrouter_free.router"):
        router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)

    assert any("switching to local fallback" in r.message for r in caplog.records)


def test_request_timeout_defaults_to_60s_and_is_configurable(engine):
    captured = {}

    def completion(**kw):
        captured.update(kw)
        return response("ok")

    router = LLMRouter(CLOUD_LOCAL_CONFIG, engine=engine, completion_fn=completion)
    router.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)
    assert captured["timeout"] == 60

    custom_config = {**CLOUD_LOCAL_CONFIG, "request_timeout_seconds": 15}
    router2 = LLMRouter(custom_config, engine=engine, completion_fn=completion)
    router2.complete("extract", [{"role": "user", "content": "hi"}], use_cache=False)
    assert captured["timeout"] == 15


def test_reasoning_is_persisted_and_survives_the_cache(engine):
    router = LLMRouter(REASONING_MODEL, engine=engine, completion_fn=lambda **kw: response("42", reasoning="steps..."))
    messages = [{"role": "user", "content": "compute"}]

    first = router.complete("qa", messages)
    assert first.reasoning == "steps..." and not first.cached

    with get_session(engine) as s:
        row = s.scalars(select(LLMCall).where(LLMCall.status == "ok")).one()
    assert row.reasoning_text == "steps..."

    second = router.complete("qa", messages)
    assert second.cached and second.reasoning == "steps..."


def test_empty_content_with_reasoning_fails_over_instead_of_caching_blank_answer(engine):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if kw["model"].startswith("nvidia_nim"):
            return response("", reasoning="spent the whole budget thinking")
        return response("a real answer")

    router = LLMRouter(REASONING_MODEL, engine=engine, completion_fn=completion)
    result = router.complete("qa", [{"role": "user", "content": "hi"}])

    assert result.deployment == "fallback" and result.text == "a real answer"
    assert statuses(engine) == [("nvidia-deepseek", "invalid_output"), ("fallback", "ok")]
    with get_session(engine) as s:
        bad = s.scalars(select(LLMCall).where(LLMCall.status == "invalid_output")).one()
    assert bad.reasoning_text == "spent the whole budget thinking"


def test_successful_call_records_an_analytics_metric_row(engine, metrics_engine):
    from llmrouter_free.store import LLMCallMetric

    router, _ = make_router(engine, lambda model, n: "the answer", metrics_engine=metrics_engine)
    router.complete("qa", [{"role": "user", "content": "hi"}], use_cache=False)

    with get_session(metrics_engine) as s:
        row = s.scalars(select(LLMCallMetric)).one()
    assert row.route == "qa" and row.deployment == "a" and row.status == "ok"
    assert row.tokens_in == 10 and row.tokens_out == 5 and row.latency_ms is not None
    assert row.hostname and row.cpu_count and row.python_version  # machine info populated


def test_failed_call_still_records_a_metric_row_with_the_error(engine, metrics_engine):
    from llmrouter_free.store import LLMCallMetric

    router, _ = make_router(engine, lambda model, n: RateLimitError("nope"), metrics_engine=metrics_engine)
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "hi"}], use_cache=False, max_wait_seconds=0)

    with get_session(metrics_engine) as s:
        rows = s.scalars(select(LLMCallMetric)).all()
    assert rows and all(r.status == "rate_limited" and r.error for r in rows)
