"""Retreino contínuo (CT): reference + janelas recentes rotuladas, mesmo gate e promoção.

Fecha o ciclo da política de resposta (seção 8.6): mede a recuperação na janela de
hold-out (não usada no treino) comparando o champion ANTES e DEPOIS do retreino.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings, get_settings
from src.constants import TARGET
from src.drift.performance import score_frame
from src.drift.simulation import load_window
from src.logger import get_logger
from src.models import train
from src.models.artifacts import load_model
from src.models.evaluate import calibration_gap, compute_metrics

logger = get_logger(__name__)


def holdout_metrics(model: Any, window: pd.DataFrame, threshold: float) -> dict[str, float]:
    """AUC, Brier e gap de calibração do modelo numa janela rotulada."""
    score = score_frame(model, window)
    y = window[TARGET].to_numpy()
    m = compute_metrics(y, score, threshold)
    return {"auc": m["auc"], "brier": m["brier"], "calibration_gap": calibration_gap(y, score)}


def write_summary(summary: dict, settings: Settings) -> Path:
    """Grava reports/retrain/retrain_summary.json."""
    out = settings.path("reports") / "retrain"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "retrain_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run(
    train_windows: list[str], holdout: str | None = None, settings: Settings | None = None
) -> dict:
    """Retreina com ``train_windows`` e mede a recuperação em ``holdout`` (default: última)."""
    settings = settings or get_settings()
    holdout = holdout or _last_window(settings)
    champion_dir = settings.path("models") / "champion"
    window = load_window(holdout, settings)
    before = holdout_metrics(load_model(champion_dir), window, settings.decision_threshold)
    extra = pd.concat([load_window(w, settings) for w in train_windows], ignore_index=True)
    metrics = train.run(settings, extra_train=extra, label="retrain")
    after = holdout_metrics(load_model(champion_dir), window, settings.decision_threshold)
    summary = {
        "trained_with": train_windows,
        "holdout": holdout,
        "before": before,
        "after": after,
        "reference_test_auc": metrics["auc"],
    }
    write_summary(summary, settings)
    logger.info("retreino_concluido", extra={"holdout": holdout, "auc_after": after["auc"]})
    return summary


def _last_window(settings: Settings) -> str:
    windows = sorted((settings.path("production") / "windows").glob("M*.parquet"))
    return windows[-1].stem
