"""Leitura dos datasets bruto, de referência e de produção."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings, get_settings
from src.constants import RAW_COLUMNS


def raw_csv_path(settings: Settings | None = None) -> Path:
    """Caminho do CSV bruto padronizado."""
    return (settings or get_settings()).path("raw") / "credit_default.csv"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Lê o CSV bruto (UTF-8) e garante as colunas esperadas."""
    df = pd.read_csv(path or raw_csv_path(), encoding="utf-8")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes no dataset bruto: {sorted(missing)}")
    return df


def load_reference(settings: Settings | None = None) -> pd.DataFrame:
    """Lê o dataset de referência (treino/teste)."""
    return pd.read_parquet((settings or get_settings()).reference_path)


def load_pool(settings: Settings | None = None) -> pd.DataFrame:
    """Lê o pool de produção (base das janelas simuladas)."""
    return pd.read_parquet((settings or get_settings()).pool_path)
