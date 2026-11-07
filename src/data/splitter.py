"""Split único reference x pool (seed fixa, estratificado), antes de qualquer fit."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Settings, get_settings
from src.constants import ID_COL, TARGET
from src.logger import get_logger

logger = get_logger(__name__)
REFERENCE_FRACTION = 0.60


def split_reference_pool(
    df: pd.DataFrame, seed: int = 42, reference_fraction: float = REFERENCE_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o bruto em reference (60%) e pool de produção (40%), estratificado.

    A deduplicação por ``client_id`` ocorre ANTES do split para evitar vazamento
    por duplicatas entre os dois conjuntos.
    """
    df = df.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    ref, pool = train_test_split(
        df, train_size=reference_fraction, stratify=df[TARGET], random_state=seed
    )
    return ref.reset_index(drop=True), pool.reset_index(drop=True)


def save_split(ref: pd.DataFrame, pool: pd.DataFrame, settings: Settings | None = None) -> None:
    """Persiste reference e pool em diretórios separados (parquet)."""
    settings = settings or get_settings()
    settings.reference_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pool_path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(settings.reference_path, index=False)
    pool.to_parquet(settings.pool_path, index=False)
    logger.info("split_salvo", extra={"reference_rows": len(ref), "pool_rows": len(pool)})
