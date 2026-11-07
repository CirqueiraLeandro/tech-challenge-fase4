"""Gerador do lote corrompido (demonstração do contrato falhando).

Injeta de forma CONTROLADA pelo menos 1 violação por regra rígida H1-H7
(H8 é demonstrada por lote truncado, nos testes).
"""

from __future__ import annotations

import pandas as pd

from src.constants import HASH_COL

INJECTIONS = {
    "H1": "coluna bill_amt6 renomeada para bill_amount_6 (schema drift)",
    "H2": "limit_bal nulo em 5 linhas",
    "H3": "age = 15 e age = 130",
    "H4": "limit_bal negativo (-5000)",
    "H5": "education = 9",
    "H6": "client_hash repetido + linha duplicada exata",
    "H7": "pay_amt1 negativo (-100)",
}


def _inject_values(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[0:5], "limit_bal"] = float("nan")
    df.loc[df.index[5], "age"] = 15
    df.loc[df.index[6], "age"] = 130
    df.loc[df.index[7], "limit_bal"] = -5000.0
    df.loc[df.index[8], "education"] = 9
    df.loc[df.index[9], "pay_amt1"] = -100.0
    return df


def _inject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.index[11], HASH_COL] = df.loc[df.index[10], HASH_COL]
    return pd.concat([df, df.iloc[[12]]], ignore_index=True)


def make_bad_batch(pool: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Gera o lote ruim a partir de uma amostra do pool já ingerido (com client_hash)."""
    df = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    df = _inject_duplicates(_inject_values(df))
    return df.rename(columns={"bill_amt6": "bill_amount_6"})
