"""Auditoria de vieses por sexo e faixa etária (implementação própria em pandas).

Métricas por grupo: taxa de aprovação, Disparate Impact (regra dos 80%), Equal Opportunity
(diferença na taxa de aprovação de BONS pagadores, o "TPR" do desfecho favorável) e AUC por
grupo. Usa a área segregada: sex/age servem apenas para auditar, nunca como feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import Settings, get_settings
from src.constants import TARGET
from src.data.loader import load_reference
from src.drift.performance import score_frame
from src.drift.simulation import load_window
from src.logger import get_logger
from src.models.artifacts import load_model
from src.models.train import split_train_test

logger = get_logger(__name__)
AGE_BINS = [0, 25, 40, 60, 200]
AGE_LABELS = ["<25", "25-40", "40-60", ">60"]
DI_THRESHOLD = 0.8
MIN_GROUP = 30


def age_band(age: pd.Series) -> pd.Series:
    """Faixas etárias <25, 25-40, 40-60, >60 (limite inferior exclusivo, 25 incluso em <25)."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def group_metrics(y: np.ndarray, approved: np.ndarray, score: np.ndarray) -> dict:
    """Aprovação, TPR de bons pagadores aprovados e AUC de um grupo."""
    good = y == 0
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "n": int(len(y)),
        "approval_rate": float(approved.mean()),
        "good_payer_approval_rate": float(approved[good].mean()) if good.any() else float("nan"),
        "default_rate": float(y.mean()),
        "auc": auc,
    }


def audit_attribute(
    df: pd.DataFrame, groups: pd.Series, score: np.ndarray, threshold: float
) -> dict:
    """Métricas por grupo + DI ratio e diferença de oportunidade (bons pagadores aprovados)."""
    y, approved = df[TARGET].to_numpy(), score < threshold  # aprovado = baixo risco previsto
    stats = {}
    for name in groups.dropna().unique():
        mask = (groups == name).to_numpy()
        if mask.sum() >= MIN_GROUP:
            stats[str(name)] = group_metrics(y[mask], approved[mask], score[mask])
    rates = {g: s["approval_rate"] for g, s in stats.items()}
    good = {g: s["good_payer_approval_rate"] for g, s in stats.items()}
    di = (
        min(rates.values()) / max(rates.values())
        if rates and max(rates.values()) > 0
        else float("nan")
    )
    return {
        "groups": stats,
        "disparate_impact": di,
        "passes_80_percent_rule": bool(di >= DI_THRESHOLD),
        "equal_opportunity_diff": max(good.values()) - min(good.values()) if good else float("nan"),
    }


def audit_frame(df: pd.DataFrame, score: np.ndarray, threshold: float) -> dict:
    """Auditoria por sexo e por faixa etária."""
    sex = df["sex"].map({1: "masculino", 2: "feminino"})
    return {
        "sex": audit_attribute(df, sex, score, threshold),
        "age_band": audit_attribute(df, age_band(df["age"]), score, threshold),
    }


def run(window: str | None = None, settings: Settings | None = None) -> dict:
    """Audita o conjunto de teste do reference e a janela final (drift pode criar viés)."""
    settings = settings or get_settings()
    model = load_model(settings.path("models") / "champion")
    _, test = split_train_test(load_reference(settings), settings.random_seed)
    names = sorted((settings.path("production") / "windows").glob("M*.parquet"))
    win_name = window or (names[-1].stem if names else None)
    frames = {"reference_test": test}
    if win_name:
        frames[win_name] = load_window(win_name, settings)
    report = {
        name: audit_frame(df, score_frame(model, df), settings.decision_threshold)
        for name, df in frames.items()
    }
    write_report(report, settings.path("reports") / "fairness")
    return report


def write_report(report: dict, out: Path) -> Path:
    """Grava reports/fairness/fairness_report.json."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fairness_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
