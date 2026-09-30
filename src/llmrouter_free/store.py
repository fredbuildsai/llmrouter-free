"""Persistence for the router: the call ledger/cache and the per-call analytics table.

Two independent SQLAlchemy declarative bases, because they normally live in two different databases:

- `Base` -> `llm_calls`: the operational ledger. Every attempt (success or failure) is a row. It doubles as
  the *daily quota ledger* (how many requests a deployment made today) and as the *response cache*
  (a prior `status == "ok"` row with the same prompt hash is replayed instead of calling the provider again).
  Lives in whatever database the host application passes as `engine=` - typically its main database.
- `MetricsBase` -> `llm_call_metrics`: a wide analytics record per attempt (latency, token usage, provider
  response id, machine info, ...), meant for offline analysis and deliberately kept out of the hot ledger.
  Lives in the optional `metrics_engine=`; without one, no metrics are recorded.

Both use only portable column types, so the same models run on SQLite and PostgreSQL. Table creation is
idempotent (`create_all`): pointing the router at a database that already has these tables reuses them.
"""

import atexit
import logging
import os
import platform
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, Index, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class MetricsBase(DeclarativeBase):
    """Separate metadata from `Base`: the metrics table is a different database file in practice."""

    type_annotation_map = {dict[str, Any]: JSON}


class LLMCall(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_deployment_ts", "deployment", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route: Mapped[str] = mapped_column(String(32), index=True)
    deployment: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tier: Mapped[str] = mapped_column(String(8))
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | rate_limited | quota | error | invalid_output
    error: Mapped[str | None] = mapped_column(Text)
    response_text: Mapped[str | None] = mapped_column(Text)  # cached output for status == ok
    reasoning_text: Mapped[str | None] = mapped_column(Text)  # cached thinking/reasoning_content, if returned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LLMCallMetric(MetricsBase):
    """One row per LLM call attempt (cloud or local), for offline analytics - not the operational
    cache/routing ledger (`LLMCall`)."""

    __tablename__ = "llm_call_metrics"
    __table_args__ = (Index("ix_llm_call_metrics_deployment_started", "deployment", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    route: Mapped[str] = mapped_column(String(32), index=True)
    deployment: Mapped[str] = mapped_column(String(64), index=True)
    requested_model: Mapped[str] = mapped_column(String(128))  # the model string we asked for
    is_local: Mapped[bool] = mapped_column(default=False, index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # ok | rate_limited | quota | error | invalid_output

    # Populated on both success and failure, where measurable:
    latency_ms: Mapped[int | None] = mapped_column(Integer)  # our own wall-clock measurement around the call

    # Populated on success only (response.usage / response fields - see `record_llm_call_metric`):
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    tokens_total: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    finish_reason: Mapped[str | None] = mapped_column(String(32))
    response_id: Mapped[str | None] = mapped_column(String(128))
    response_model: Mapped[str | None] = mapped_column(String(128))  # the model the provider says it used -
                                                                      # can differ from requested_model
    system_fingerprint: Mapped[str | None] = mapped_column(String(128))
    litellm_response_ms: Mapped[float | None] = mapped_column(Float)  # LiteLLM's own timing, as a cross-check
    cost_usd: Mapped[float | None] = mapped_column(Float)
    api_base: Mapped[str | None] = mapped_column(String(256))

    error: Mapped[str | None] = mapped_column(Text)

    # Machine info, denormalized onto every row so each row is independently analyzable with no join:
    hostname: Mapped[str | None] = mapped_column(String(128))
    hw_model: Mapped[str | None] = mapped_column(String(64))
    ram_bytes: Mapped[int | None] = mapped_column(Integer)
    cpu_count: Mapped[int | None] = mapped_column(Integer)
    machine_arch: Mapped[str | None] = mapped_column(String(32))
    platform_str: Mapped[str | None] = mapped_column(String(256))
    python_version: Mapped[str | None] = mapped_column(String(32))


def register_sqlite_pragmas(engine: Engine) -> Engine:
    """WAL journal + foreign keys + a 30 s busy timeout on every SQLite connection.

    The router writes its ledger row from whichever thread made the call, and a host application may hold its
    own open write transaction on the same file: without a busy timeout, two overlapping SQLite writers fail
    immediately with "database is locked" instead of waiting their turn. A no-op for non-SQLite engines.
    """
    if engine.dialect.name != "sqlite":
        return engine

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


def make_ephemeral_engine() -> Engine:
    """A private, process-lifetime SQLite database - what the router uses when it is not given an `engine=`.

    Ledger + cache work within this process; nothing survives it (the temp directory is removed at exit).

    Deliberately a real temp *file*, not `sqlite://` (in-memory): an in-memory SQLite needs `StaticPool`, which
    hands the SAME connection to every thread, so concurrent sessions interleave their transactions on one
    connection. Measured with 8 threads x 40 inserts: "cannot start a transaction within a transaction" errors
    and, worse, silently lost rows (35-39 of 40). A file with WAL + a busy timeout gives every thread its own
    connection and lets the database serialize writers correctly.
    """
    directory = tempfile.mkdtemp(prefix="llmrouter-free-")
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return register_sqlite_pragmas(create_engine(f"sqlite:///{directory}/ledger.db", future=True))


def init_ledger(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def init_metrics(engine: Engine) -> None:
    MetricsBase.metadata.create_all(engine)


@contextmanager
def get_session(engine: Engine) -> Iterator[Session]:
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@lru_cache
def machine_info() -> dict[str, Any]:
    """Static facts about the machine this process runs on, cached for the process lifetime.

    `hw.model`/`hw.memsize` are macOS-specific (`sysctl`; Apple Silicon has no CPU brand string via the stdlib
    `platform` module); on any other OS those two fields stay None rather than being guessed.
    """
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "machine_arch": platform.machine(),
        "platform_str": platform.platform(),
        "python_version": platform.python_version(),
        "hw_model": None,
        "ram_bytes": None,
    }
    if platform.system() == "Darwin":
        for key, field in (("hw.model", "hw_model"), ("hw.memsize", "ram_bytes")):
            try:
                value = subprocess.run(
                    ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
                ).stdout.strip()
                info[field] = int(value) if field == "ram_bytes" else value
            except (subprocess.SubprocessError, OSError, ValueError):
                pass  # best-effort - never let hardware introspection break a real call
    return info


def record_llm_call_metric(
    engine: Engine,
    *,
    started_at: datetime,
    route: str,
    deployment: str,
    requested_model: str,
    is_local: bool,
    status: str,
    latency_ms: int | None = None,
    response: Any = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> None:
    """Record one call attempt into `llm_call_metrics`.

    `response` is the raw LiteLLM `ModelResponse` on success, None on failure. LiteLLM normalizes every provider
    (including local Ollama) to the same OpenAI-style `usage` block and hidden params; what it does NOT surface
    is Ollama's richer native timing breakdown, so for local calls `litellm_response_ms` is the best available
    internal-timing cross-check.
    """
    usage = getattr(response, "usage", None)
    hidden = getattr(response, "_hidden_params", None) or {}
    reasoning_tokens = None
    if usage is not None and getattr(usage, "completion_tokens_details", None):
        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", None)

    with get_session(engine) as s:
        s.add(LLMCallMetric(
            started_at=started_at, route=route, deployment=deployment, requested_model=requested_model,
            is_local=is_local, status=status, latency_ms=latency_ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            tokens_total=getattr(usage, "total_tokens", None) if usage else None,
            reasoning_tokens=reasoning_tokens,
            finish_reason=getattr(response.choices[0], "finish_reason", None) if response else None,
            response_id=getattr(response, "id", None) if response else None,
            response_model=getattr(response, "model", None) if response else None,
            system_fingerprint=getattr(response, "system_fingerprint", None) if response else None,
            litellm_response_ms=hidden.get("_response_ms"),
            cost_usd=cost_usd if cost_usd is not None else hidden.get("response_cost"),
            api_base=hidden.get("api_base"),
            error=error,
            **machine_info(),
        ))
