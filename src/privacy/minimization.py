"""Minimização de dados (LGPD art. 6º, III) e área segregada para auditoria de vieses."""

from __future__ import annotations

import pandas as pd

from src.constants import EXCLUDED_FROM_MODEL, HASH_COL, MODEL_FEATURES, PROTECTED_ATTRIBUTES

AUDIT_COLUMNS = [HASH_COL, *PROTECTED_ATTRIBUTES, "age"]


def minimize(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identificadores e atributos protegidos, mantendo o que o modelo precisa."""
    return df.drop(columns=[c for c in EXCLUDED_FROM_MODEL if c in df.columns])


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona exatamente ``MODEL_FEATURES`` (falha se faltar alguma)."""
    return df[MODEL_FEATURES]


def audit_view(df: pd.DataFrame) -> pd.DataFrame:
    """Área segregada: apenas o necessário para auditar vieses (finalidade específica)."""
    return df[[c for c in AUDIT_COLUMNS if c in df.columns]].copy()
