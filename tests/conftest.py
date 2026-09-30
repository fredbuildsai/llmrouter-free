import pytest
from sqlalchemy import create_engine

from llmrouter_free.store import init_ledger, init_metrics, register_sqlite_pragmas


@pytest.fixture
def engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'ledger.db'}", future=True))
    init_ledger(engine)
    return engine


@pytest.fixture
def metrics_engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'metrics.db'}", future=True))
    init_metrics(engine)
    return engine
