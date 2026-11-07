"""Concept drift e performance por janela (AUC, KS, calibração, Brier, PSI do score)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.constants import TARGET
from src.drift.stattests import Bins, psi_from_probs
from src.models.evaluate import calibration_gap, compute_metrics
from src.models.train import build_xy


@dataclass(frozen=True)
class WindowPerformance:
    """Métricas de performance de uma janela rotulada."""

    auc: float
    ks: float
    brier: float
    predicted_rate: float
    observed_rate: float
    calibration_gap: float
    score_psi: float
    auc_drop: float

    def to_dict(self) -> dict[str, float]:
        """Dicionário serializável."""
        return asdict(self)


def score_frame(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Pontua um DataFrame bruto/ingerido com o modelo (limpeza + features dentro)."""
    x, _ = build_xy(df)
    return np.asarray(model.predict_proba(x)[:, 1])


def window_performance(
    model: Any,
    window: pd.DataFrame,
    baseline_auc: float,
    ref_score_bins: Bins,
    threshold: float = 0.5,
) -> tuple[WindowPerformance, np.ndarray]:
    """Mede a janela: AUC/KS/Brier, gap de calibração e PSI do score (prediction drift)."""
    score = score_frame(model, window)
    y = window[TARGET].to_numpy()
    m = compute_metrics(y, score, threshold)
    perf = WindowPerformance(
        auc=m["auc"],
        ks=m["ks"],
        brier=m["brier"],
        predicted_rate=float(score.mean()),
        observed_rate=float(y.mean()),
        calibration_gap=calibration_gap(y, score),
        score_psi=psi_from_probs(ref_score_bins.ref_probs, ref_score_bins.probs(score)),
        auc_drop=baseline_auc - m["auc"],
    )
    return perf, score
