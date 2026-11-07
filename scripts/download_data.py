"""Baixa o dataset Default of Credit Card Clients (UCI id=350) e padroniza as colunas.

Tenta ``ucimlrepo`` e cai para ``fetch_openml``. Saída: data/raw/credit_default.csv.
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import get_settings
from src.constants import ID_COL, RAW_COLUMNS, TARGET, UCI_RENAME, X_ORDER
from src.data.loader import raw_csv_path
from src.logger import get_logger

logger = get_logger(__name__)
EXPECTED_ROWS = 30_000
UCI_URL = "https://archive.ics.uci.edu/dataset/350"


def normalize_columns(
    features: pd.DataFrame, target: pd.Series, ids: pd.Series | None
) -> pd.DataFrame:
    """Padroniza nomes: aceita nomes UCI originais ou X1..X23 (ordem do dataset)."""
    features = features.copy()
    if all(str(c).startswith("X") for c in features.columns):
        features.columns = X_ORDER
    else:
        features = features.rename(columns=UCI_RENAME)
        features.columns = [str(c).lower() for c in features.columns]
    features[ID_COL] = ids.to_numpy() if ids is not None else range(1, len(features) + 1)
    features[TARGET] = target.to_numpy()
    return features[RAW_COLUMNS]


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=350).data
    ids = data.ids.iloc[:, 0] if data.ids is not None else None
    return normalize_columns(data.features, data.targets.iloc[:, 0], ids)


def _from_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="default-of-credit-card-clients", as_frame=True)
    return normalize_columns(bunch.data, bunch.target.astype(int), None)


def verify(df: pd.DataFrame) -> None:
    """Valida volume, colunas e domínio do alvo."""
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"esperadas {EXPECTED_ROWS} linhas, obtidas {len(df)}")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("alvo deve ser binário 0/1")


def download() -> pd.DataFrame:
    """Tenta ucimlrepo e depois OpenML."""
    for name, fetch in (("ucimlrepo", _from_ucimlrepo), ("openml", _from_openml)):
        try:
            df = fetch()
            verify(df)
            logger.info("download_ok", extra={"source": name, "rows": len(df)})
            return df
        except Exception as exc:  # noqa: BLE001 - fallback entre fontes
            logger.warning("download_falhou", extra={"source": name, "error": str(exc)})
    raise RuntimeError(f"Falha em todas as fontes. Baixe manualmente: {UCI_URL}")


def main() -> int:
    """Ponto de entrada: salva o CSV padronizado."""
    try:
        df = download()
    except RuntimeError as exc:
        logger.error("download_abortado", extra={"error": str(exc)})
        return 1
    path = raw_csv_path(get_settings())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("dataset_salvo", extra={"path": str(path)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
