"""Contract tests for the persistence layer: what a host application can rely on."""
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine, select, text

from llmrouter_free.router import LLMRouter
from llmrouter_free.store import (
    LLMCall,
    LLMCallMetric,
    get_session,
    init_ledger,
    make_ephemeral_engine,
    record_llm_call_metric,
    register_sqlite_pragmas,
)

CONFIG = {
    "deployments": [{"name": "a", "model": "p/a", "api_key_env": "KEY_A"}],
    "routes": {"qa": ["a"]},
}


def ok_response(text_="answer"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text_), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="resp-1", model="p/a", system_fingerprint="fp",
    )


def test_router_without_an_engine_keeps_a_private_ledger_and_cache(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")
    calls = []

    def completion(**kw):
        calls.append(kw)
        return ok_response()

    router = LLMRouter(CONFIG, completion_fn=completion)
    msgs = [{"role": "user", "content": "hi"}]
    assert router.complete("qa", msgs).cached is False
    assert router.complete("qa", msgs).cached is True  # replayed from the private ledger
    assert len(calls) == 1


def test_an_existing_llm_calls_table_is_reused_not_recreated(tmp_path, monkeypatch):
    """Contract: pointing the router at a database that already has `llm_calls` (written by an earlier
    host application version) keeps its rows - the cache and the daily quota ledger survive."""
    monkeypatch.setenv("KEY_A", "x")
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True))
    init_ledger(engine)
    with get_session(engine) as s:
        s.add(LLMCall(route="qa", deployment="a", model="p/a", tier="free", prompt_hash="h", status="ok",
                      response_text="old"))

    LLMRouter(CONFIG, engine=engine, completion_fn=lambda **kw: ok_response())  # constructing re-runs init

    with get_session(engine) as s:
        assert s.scalar(select(LLMCall.response_text)) == "old"


def test_ledger_columns_are_the_ones_older_databases_have(tmp_path):
    """The column set is a compatibility surface: a host application's existing table must match it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}", future=True)
    init_ledger(engine)
    with engine.connect() as c:
        cols = {row[1] for row in c.execute(text("PRAGMA table_info(llm_calls)"))}
    assert cols == {"id", "route", "deployment", "model", "tier", "prompt_hash", "tokens_in", "tokens_out",
                    "cost_usd", "latency_ms", "status", "error", "response_text", "reasoning_text", "created_at"}


def test_no_metrics_are_recorded_without_a_metrics_engine(engine, monkeypatch):
    monkeypatch.setenv("KEY_A", "x")
    router = LLMRouter(CONFIG, engine=engine, completion_fn=lambda **kw: ok_response())
    router.complete("qa", [{"role": "user", "content": "hi"}])
    assert router.metrics_engine is None


def test_a_broken_metrics_sink_never_breaks_a_real_call(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_A", "x")
    bad = create_engine(f"sqlite:///{tmp_path}/does/not/exist/m.db", future=True)  # cannot be opened
    router = LLMRouter(CONFIG, engine=engine, completion_fn=lambda **kw: ok_response())
    router.metrics_engine = bad  # bypass constructor-time init; failure must be swallowed at record time
    assert router.complete("qa", [{"role": "user", "content": "hi"}]).text == "answer"


def test_record_llm_call_metric_captures_usage_and_machine_info(metrics_engine):
    record_llm_call_metric(
        metrics_engine, started_at=datetime.now(timezone.utc), route="qa", deployment="a", requested_model="p/a",
        is_local=False, status="ok", latency_ms=120, response=ok_response(), cost_usd=0.0,
    )
    with get_session(metrics_engine) as s:
        row = s.scalars(select(LLMCallMetric)).one()
    assert (row.tokens_in, row.tokens_out, row.tokens_total) == (10, 5, 15)
    assert row.finish_reason == "stop" and row.response_id == "resp-1" and row.hostname


def test_ephemeral_engine_gives_concurrent_checkouts_distinct_connections():
    """Deterministic guard for the root cause of a real bug. The first version used an in-memory SQLite with
    `StaticPool`, which hands the SAME DBAPI connection to every caller: concurrent sessions then interleave
    their transactions on one connection ("cannot start a transaction within a transaction", and silently lost
    rows - ~1 in 60 trials of 8 threads x 40 inserts). That failure is probabilistic, so this asserts the
    invariant that causes it instead of hoping a stress run trips it."""
    engine = make_ephemeral_engine()
    with engine.connect() as first, engine.connect() as second:
        assert first.connection.dbapi_connection is not second.connection.dbapi_connection


def test_ephemeral_engine_keeps_every_row_written_by_concurrent_threads():
    """Behavioral smoke check on top of the deterministic guard above (a few bursts, not a stress run)."""
    from concurrent.futures import ThreadPoolExecutor

    for _ in range(8):
        engine = make_ephemeral_engine()
        init_ledger(engine)

        def write(i, engine=engine):
            with get_session(engine) as s:
                s.add(LLMCall(route="r", deployment="d", model="m", tier="free", prompt_hash=str(i), status="ok"))

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(write, range(40)))
        with get_session(engine) as s:
            assert len(s.scalars(select(LLMCall)).all()) == 40
