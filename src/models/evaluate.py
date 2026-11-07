"""Métricas de crédito (AUC, Gini, KS, F1, Brier...) e avaliação do modelo vigente."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import Settings, get_settings  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def ks_separation(y_true: np.ndarray, score: np.ndarray) -> float:
    """KS de separação do mercado de crédito: max(TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula as métricas de crédito no threshold escolhido."""
    pred = (score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, score))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_separation(y_true, score),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, score)),
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_gap(y_true: np.ndarray, score: np.ndarray) -> float:
    """Taxa média prevista menos taxa observada de default (gap de calibração)."""
    return float(np.mean(score) - np.mean(y_true))


def plot_confusion(confusion: dict, path: Path) -> Path:
    """Salva a matriz de confusão em PNG."""
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["prev. 0", "prev. 1"])
    ax.set_yticks([0, 1], ["real 0", "real 1"])
    ax.set_title("Matriz de confusão")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_metrics(metrics: dict, path: Path) -> None:
    """Grava metrics.json em UTF-8."""
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_champion(settings: Settings | None = None) -> dict:
    """Avalia o champion no conjunto de teste do reference e grava models/metrics.json."""
    from src.models.artifacts import load_model
    from src.models.train import load_test_split

    settings = settings or get_settings()
    model = load_model(settings.path("models") / "champion")
    x_test, y_test = load_test_split(settings)
    metrics = compute_metrics(
        y_test.to_numpy(), model.predict_proba(x_test)[:, 1], settings.decision_threshold
    )
    write_metrics(metrics, settings.path("models") / "metrics.json")
    logger.info("avaliacao_champion", extra={"auc": metrics["auc"], "ks": metrics["ks"]})
    return metrics


def main() -> int:
    """Entrypoint real: ``python -m src.models.evaluate``."""
    evaluate_champion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
