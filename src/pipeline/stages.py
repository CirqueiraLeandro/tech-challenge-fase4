"""Estágios do pipeline: funções pequenas, chamadas pela CLI e pelos orquestradores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings, get_settings
from src.contracts.rules import ReferenceStats, compute_reference_stats
from src.contracts.validator import ingest_and_validate
from src.data.loader import load_raw, raw_csv_path
from src.data.splitter import save_split, split_reference_pool
from src.logger import get_logger

logger = get_logger(__name__)


def prepare(settings: Settings | None = None, raw: pd.DataFrame | None = None) -> dict[str, int]:
    """Split reference/pool, pseudonimiza o ID e valida ambos no contrato antes de salvar."""
    settings = settings or get_settings()
    raw = raw if raw is not None else load_raw(raw_csv_path(settings))
    ref, pool = split_reference_pool(raw, settings.random_seed)
    ref, _ = ingest_and_validate(ref, "reference", settings=settings)
    stats = compute_reference_stats(ref)
    pool, _ = ingest_and_validate(pool, "pool", stats=stats, settings=settings)
    save_split(ref, pool, settings)
    return {"reference": len(ref), "pool": len(pool)}


def reference_stats(settings: Settings | None = None) -> ReferenceStats:
    """Estatísticas do reference para as regras suaves."""
    from src.data.loader import load_reference

    return compute_reference_stats(load_reference(settings))


def validate_file(path: Path, settings: Settings | None = None) -> None:
    """Valida um arquivo CSV isolado; lança ContractViolationError se reprovado."""
    settings = settings or get_settings()
    df = pd.read_csv(path, encoding="utf-8")
    stats = reference_stats(settings) if settings.reference_path.exists() else None
    ingest_and_validate(df, path.stem, stats=stats, settings=settings)
