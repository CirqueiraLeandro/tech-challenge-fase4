"""Limpeza documentada: remapeia códigos categóricos não documentados no dataset UCI.

education 0, 5, 6 -> 4 ("outros") e marriage 0 -> 3 ("outros"). Determinística e sem
estatísticas aprendidas (não há risco de vazamento).
"""

from __future__ import annotations

import pandas as pd

from src.constants import UNDOCUMENTED_EDUCATION, UNDOCUMENTED_MARRIAGE

OTHER_EDUCATION = 4
OTHER_MARRIAGE = 3


def undocumented_rate(df: pd.DataFrame) -> float:
    """Fração de linhas com education/marriage fora do dicionário documentado."""
    bad = df["education"].isin(UNDOCUMENTED_EDUCATION) | df["marriage"].isin(UNDOCUMENTED_MARRIAGE)
    return float(bad.mean()) if len(df) else 0.0


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna cópia com os códigos não documentados remapeados para 'outros'."""
    out = df.copy()
    out["education"] = out["education"].replace(UNDOCUMENTED_EDUCATION, OTHER_EDUCATION)
    out["marriage"] = out["marriage"].replace(UNDOCUMENTED_MARRIAGE, OTHER_MARRIAGE)
    return out
