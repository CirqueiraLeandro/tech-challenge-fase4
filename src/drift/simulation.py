"""Simulação causal (SCM) do tempo: janelas M1..M6 a partir do pool de produção.

Cada mecanismo é independente e pode ser ligado/desligado por config ou por
intervenção (``disabled``), o que habilita a análise causal da seção 11:

- ``inflacao``     (covariate shift): bill_amt *= 1 + 0,40 t ; pay_amt *= 1 + 0,05 t
- ``aperto``       (covariate shift): limit_bal *= 1 - 0,25 t
- ``desemprego``   (CONCEPT drift): choque latente U ~ Bernoulli(0,20 t), NÃO observado
- ``perfil_jovem`` (composição): amostragem com peso 1 + 1,5 t para age < 30

Todos os sorteios são pré-gerados POR MECANISMO e por posição do chunk, de modo que
desligar um mecanismo não altera os sorteios dos demais (comparação intervencional limpa).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DriftScenario, Settings, get_settings
from src.constants import BILL_COLS, PAY_AMT_COLS
from src.logger import get_logger

logger = get_logger(__name__)

INFLATION_BILL, INFLATION_PAY = 0.40, 0.05
TIGHTENING = 0.25
UNEMPLOYMENT_RATE, FLIP_PROB = 0.20, 0.5
YOUNG_AGE, YOUNG_WEIGHT = 30, 1.5
MECHANISM_COLUMNS = {
    "inflacao": [*BILL_COLS, *PAY_AMT_COLS],
    "aperto": ["limit_bal"],
    "desemprego": ["pay_0", "pay_2", "default"],
    "perfil_jovem": [],
}


@dataclass
class WindowResult:
    """Janela simulada e o choque latente U (não observado pelo modelo)."""

    data: pd.DataFrame
    latent_u: np.ndarray


def _draws(seed: int, k: int, n: int) -> dict[str, dict[str, np.ndarray]]:
    """Sorteios pré-gerados por mecanismo (streams independentes)."""
    rng = {m: np.random.default_rng([seed, k, i]) for i, m in enumerate(MECHANISM_COLUMNS)}
    return {
        "perfil_jovem": {"u": rng["perfil_jovem"].random(n)},
        "desemprego": {
            "u": rng["desemprego"].random(n),
            "size": rng["desemprego"].integers(1, 3, size=(n, 2)),
            "flip": rng["desemprego"].random(n),
        },
    }


def _select(chunk: pd.DataFrame, u: np.ndarray, t: float, size: int, on: bool) -> np.ndarray:
    """Amostra ``size`` linhas sem reposição (Efraimidis-Spirakis: chave = log(u)/peso)."""
    weight = np.ones(len(chunk))
    if on:
        weight = np.where(chunk["age"].to_numpy() < YOUNG_AGE, 1 + YOUNG_WEIGHT * t, 1.0)
    keys = np.log(u) / weight
    return np.sort(np.argsort(-keys)[:size])


def _apply_prices(df: pd.DataFrame, t: float, inflation: bool, tightening: bool) -> pd.DataFrame:
    if inflation:
        df[BILL_COLS] = df[BILL_COLS].astype(float) * (1 + INFLATION_BILL * t)
        df[PAY_AMT_COLS] = df[PAY_AMT_COLS].astype(float) * (1 + INFLATION_PAY * t)
    if tightening:
        df["limit_bal"] = df["limit_bal"].astype(float) * (1 - TIGHTENING * t)
    return df


def _apply_unemployment(df: pd.DataFrame, t: float, d: dict, pos: np.ndarray) -> np.ndarray:
    shock = d["u"][pos] < UNEMPLOYMENT_RATE * t
    for j, col in enumerate(["pay_0", "pay_2"]):
        bumped = (df[col].to_numpy() + d["size"][pos, j]).clip(max=8)
        df[col] = np.where(shock, bumped, df[col].to_numpy())
    flip = shock & (df["default"].to_numpy() == 0) & (d["flip"][pos] < FLIP_PROB)
    df["default"] = np.where(flip, 1, df["default"].to_numpy())
    return shock


def generate_window(
    chunk: pd.DataFrame,
    k: int,
    t: float,
    scenario: DriftScenario,
    seed: int,
    disabled: frozenset[str] = frozenset(),
) -> WindowResult:
    """Gera a janela ``k`` com intensidade ``t`` aplicando os mecanismos ativos."""
    on = {m: c.enabled and m not in disabled for m, c in scenario.mechanisms.items()}
    draws = _draws(seed, k, len(chunk))
    pos = _select(
        chunk, draws["perfil_jovem"]["u"], t, scenario.window_size, on.get("perfil_jovem", False)
    )
    df = chunk.iloc[pos].reset_index(drop=True).copy()
    df = _apply_prices(df, t, on.get("inflacao", False), on.get("aperto", False))
    shock = np.zeros(len(df), dtype=bool)
    if on.get("desemprego", False):
        shock = _apply_unemployment(df, t, draws["desemprego"], pos)
    return WindowResult(df, shock)


def split_chunks(pool: pd.DataFrame, n_windows: int, seed: int) -> list[pd.DataFrame]:
    """Particiona o pool em ``n_windows`` blocos disjuntos (embaralhados com a seed)."""
    shuffled = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    positions = np.array_split(np.arange(len(shuffled)), n_windows)
    return [shuffled.iloc[idx].reset_index(drop=True) for idx in positions]


def simulate_windows(
    pool: pd.DataFrame,
    scenario: DriftScenario,
    seed: int,
    disabled: frozenset[str] = frozenset(),
) -> dict[str, WindowResult]:
    """Gera todas as janelas M1..M6 do cenário (determinístico por seed)."""
    chunks = split_chunks(pool, len(scenario.windows), seed)
    return {
        name: generate_window(chunks[k], k, t, scenario, seed, disabled)
        for k, (name, t) in enumerate(zip(scenario.windows, scenario.intensities, strict=True))
    }


def save_windows(windows: dict[str, WindowResult], settings: Settings | None = None) -> Path:
    """Persiste as janelas em data/production/windows/Mk.parquet."""
    settings = settings or get_settings()
    out = settings.path("production") / "windows"
    out.mkdir(parents=True, exist_ok=True)
    for name, result in windows.items():
        result.data.to_parquet(out / f"{name}.parquet", index=False)
    logger.info("janelas_salvas", extra={"windows": list(windows), "dir": str(out)})
    return out


def load_window(name: str, settings: Settings | None = None) -> pd.DataFrame:
    """Lê uma janela simulada."""
    settings = settings or get_settings()
    return pd.read_parquet(settings.path("production") / "windows" / f"{name}.parquet")
