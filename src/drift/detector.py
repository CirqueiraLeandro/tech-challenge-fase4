"""Detector de drift: testes próprios (PSI/KS/Chi²/JS) + relatório HTML do Evidently 0.7.

O Evidently escolhe o teste por tamanho de amostra (Wasserstein para n > 1000), então o
método é configurado EXPLICITAMENTE (PSI) para bater com o requisito e com ``stattests``.
As estatísticas do reference são calculadas UMA vez por instância (lição 13.8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import MonitoringConfig
from src.constants import MODEL_FEATURES, PAY_COLS
from src.data.cleaning import clean_categories
from src.data.features import add_features
from src.drift.stattests import (
    Bins,
    ChiSquareTest,
    JensenShannonTest,
    KSTest,
    PSITest,
    bonferroni_alpha,
)

CATEGORICAL_FEATURES = ["education", *PAY_COLS]


def feature_view(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Vista de features do modelo (limpeza + derivadas), idêntica à usada no scoring."""
    enriched = add_features(clean_categories(df))
    return enriched[features or MODEL_FEATURES].reset_index(drop=True)


def severity_from_psi(psi: float, cfg: MonitoringConfig) -> str:
    """Classifica a magnitude do PSI: stable / moderate / severe."""
    if psi >= cfg.psi_severe:
        return "severe"
    return "moderate" if psi >= cfg.psi_moderate else "stable"


class DriftDetector:
    """Compara janelas contra o reference com estatísticas pré-computadas."""

    def __init__(
        self, ref: pd.DataFrame, cfg: MonitoringConfig, features: list[str] | None = None
    ) -> None:
        self.cfg = cfg
        self.features = features or MODEL_FEATURES
        self.ref = feature_view(ref, self.features)
        self.bins = {f: Bins(self.ref[f]) for f in self.features}
        self.psi = PSITest(cfg.psi_moderate)
        self.js = JensenShannonTest(0.1)

    def _row(self, feature: str, cur: pd.Series, alpha: float) -> dict:
        ref, bins = self.ref[feature], self.bins[feature]
        psi = self.psi.compute(ref, cur, bins).statistic
        ks = KSTest(alpha).compute(ref, cur)
        categorical = feature in CATEGORICAL_FEATURES
        chi = ChiSquareTest(alpha).compute(ref, cur) if categorical else None
        p_value = chi.p_value if chi else ks.p_value
        return {
            "feature": feature,
            "psi": psi,
            "js": self.js.compute(ref, cur, bins).statistic,
            "ks_stat": ks.statistic,
            "ks_p": ks.p_value,
            "chi2_p": chi.p_value if chi else np.nan,
            "p_value": p_value,
            "severity": severity_from_psi(psi, self.cfg),
            "drifted": bool(psi >= self.cfg.psi_moderate and p_value < alpha),
        }

    def analyze(self, cur: pd.DataFrame) -> pd.DataFrame:
        """Tabela por feature: magnitude (PSI/JS), significância (KS/Chi²) e decisão."""
        cur = feature_view(cur, self.features)
        alpha = bonferroni_alpha(self.cfg.ks_alpha, len(self.features), self.cfg.bonferroni)
        return pd.DataFrame([self._row(f, cur[f], alpha) for f in self.features])

    def evidently(self, cur: pd.DataFrame, html_path: Path) -> pd.DataFrame:
        """Gera o HTML do Evidently (PSI explícito) e devolve o resumo por feature."""
        preset = DataDriftPreset(
            columns=self.features, method="psi", threshold=self.cfg.psi_moderate
        )
        snapshot = Report([preset]).run(
            current_data=feature_view(cur, self.features), reference_data=self.ref
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
        return parse_evidently(snapshot.dict(), self.cfg.psi_moderate)


def parse_evidently(payload: dict, threshold: float) -> pd.DataFrame:
    """Converte ``snapshot.dict()`` (API 0.7) em tabela feature x PSI do Evidently."""
    rows = [
        {
            "feature": m["config"]["column"],
            "evidently_psi": float(m["value"]),
            "evidently_drifted": float(m["value"]) >= threshold,
        }
        for m in payload["metrics"]
        if m["config"].get("type", "").endswith("ValueDrift")
    ]
    return pd.DataFrame(rows)


def drift_share(table: pd.DataFrame, column: str = "drifted") -> float:
    """Fração de features sinalizadas."""
    return float(table[column].mean()) if len(table) else 0.0


def cross_check(own: pd.DataFrame, evid: pd.DataFrame) -> dict:
    """Compara as features sinalizadas pelos testes próprios e pelo Evidently."""
    mine = set(own.loc[own["drifted"], "feature"])
    theirs = set(evid.loc[evid["evidently_drifted"], "feature"])
    union = mine | theirs
    return {
        "own_share": drift_share(own),
        "evidently_share": drift_share(evid, "evidently_drifted"),
        "jaccard": len(mine & theirs) / len(union) if union else 1.0,
        "only_own": sorted(mine - theirs),
        "only_evidently": sorted(theirs - mine),
    }
