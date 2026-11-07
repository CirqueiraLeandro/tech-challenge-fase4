"""Fixtures compartilhadas: ambiente de teste isolado e dados sintéticos."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Logs da suíte em diretório temporário: nunca contaminam logs/ do projeto nem o painel.
os.environ["LOGS_DIR"] = tempfile.mkdtemp(prefix="credit_test_logs_")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.config import Settings, get_settings  # noqa: E402
from src.data.synthetic import make_synthetic_credit  # noqa: E402
from src.privacy.pseudonymize import pseudonymize_ids  # noqa: E402

DIR_KEYS = ["raw", "reference", "production", "incoming", "quarantine", "models", "logs", "reports"]


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    """Settings isolado: diretórios em tmp_path e MLflow em sqlite temporário."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    for key in DIR_KEYS:
        monkeypatch.setenv(f"{key.upper()}_DIR", str(tmp_path / key))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("PUSHGATEWAY_URL", "")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Dataset sintético bruto (~1.500 linhas, schema real)."""
    return make_synthetic_credit(1500)


@pytest.fixture
def ingested_df(raw_df, settings) -> pd.DataFrame:
    """Dataset com client_id pseudonimizado (formato aceito pelo contrato)."""
    return pseudonymize_ids(raw_df, settings)
