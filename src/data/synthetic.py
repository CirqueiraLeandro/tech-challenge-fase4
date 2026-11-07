"""Gerador sintético com o MESMO schema do dataset real (CI, testes e demo)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import BILL_COLS, ID_COL, PAY_AMT_COLS, PAY_COLS, RAW_COLUMNS, TARGET


def _pay_status(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Status de atraso correlacionado entre meses (-2..8)."""
    base = rng.choice([-1, 0, 0, 0, 1, 2], size=n)
    cols = {}
    for col in PAY_COLS:
        noise = rng.choice([-1, 0, 0, 0, 1], size=n)
        cols[col] = np.clip(base + noise, -2, 8)
    return pd.DataFrame(cols)


def _money(rng: np.random.Generator, n: int, limit: np.ndarray) -> pd.DataFrame:
    """Faturas e pagamentos proporcionais ao limite."""
    data = {}
    for col in BILL_COLS:
        data[col] = np.round(limit * rng.beta(2, 5, n)).astype(float)
    for col in PAY_AMT_COLS:
        data[col] = np.round(limit * rng.beta(1, 25, n)).astype(float)
    return pd.DataFrame(data)


def _target(rng: np.random.Generator, pay: pd.DataFrame, limit: np.ndarray) -> np.ndarray:
    """Alvo (~22% de default) dependente de atraso recente e limite."""
    score = 0.9 * pay["pay_0"].to_numpy() + 0.4 * pay["pay_2"].to_numpy()
    score = score - 0.000004 * limit + rng.normal(0, 1.0, len(pay))
    cutoff = np.quantile(score, 0.78)
    return (score > cutoff).astype(int)


def make_synthetic_credit(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Gera ``n`` clientes sintéticos com as colunas brutas padronizadas."""
    rng = np.random.default_rng(seed)
    limit = np.clip(np.round(rng.lognormal(11.5, 0.8, n), -3), 10_000, 800_000)
    pay = _pay_status(rng, n)
    df = pd.DataFrame(
        {
            ID_COL: np.arange(1, n + 1),
            "limit_bal": limit.astype(float),
            "sex": rng.choice([1, 2], n, p=[0.4, 0.6]),
            "education": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02]),
            "marriage": rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02]),
            "age": np.clip(rng.gamma(9, 4, n), 21, 75).astype(int),
        }
    )
    df = pd.concat([df, pay, _money(rng, n, limit)], axis=1)
    df[TARGET] = _target(rng, pay, limit)
    return df[RAW_COLUMNS]
