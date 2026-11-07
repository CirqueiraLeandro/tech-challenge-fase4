"""Treino do modelo de crédito: valida o reference, treina no candidate e aplica o gate."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Settings, get_settings
from src.constants import MODEL_FEATURES, PROTECTED_ATTRIBUTES, TARGET
from src.contracts.schemas import CONTRACT_VERSION
from src.contracts.validator import validate_batch
from src.data.cleaning import clean_categories
from src.data.features import add_features
from src.data.loader import load_reference
from src.exceptions import ContractViolationError, GateFailedError
from src.logger import get_logger
from src.models import registry
from src.models.artifacts import save_model
from src.models.evaluate import compute_metrics, plot_confusion, write_metrics
from src.models.factory import create_model
from src.monitoring import tracking
from src.privacy.minimization import model_matrix

logger = get_logger(__name__)
TEST_SIZE = 0.20


def validate_reference(ref: pd.DataFrame, settings: Settings) -> None:
    """Primeira task do treino: o reference precisa passar no contrato."""
    result = validate_batch(ref, settings=settings)
    if not result.passed:
        raise ContractViolationError(f"reference reprovado: {result.violated_rules}")


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Limpa categorias, deriva features e devolve (X do modelo, y)."""
    enriched = add_features(clean_categories(df))
    return model_matrix(enriched), enriched[TARGET]


def split_train_test(ref: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interno do reference: 80/20 estratificado."""
    train, test = train_test_split(
        ref, test_size=TEST_SIZE, stratify=ref[TARGET], random_state=seed
    )
    return train, test


def load_test_split(settings: Settings | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstrói deterministicamente o conjunto de teste do reference."""
    settings = settings or get_settings()
    _, test = split_train_test(load_reference(settings), settings.random_seed)
    return build_xy(test)


def file_sha256(path: Path) -> str:
    """Hash SHA-256 do arquivo (linhagem do dataset)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    """SHA curto do commit atual (ou 'unknown' fora de um repositório)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def auc_with_protected(train: pd.DataFrame, test: pd.DataFrame, settings: Settings) -> float:
    """AUC de um modelo que USA sex/marriage (mede o custo da minimização, sem promovê-lo)."""
    x_tr, y_tr = build_xy(train)
    x_te, y_te = build_xy(test)
    x_tr = x_tr.join(train[PROTECTED_ATTRIBUTES])
    x_te = x_te.join(test[PROTECTED_ATTRIBUTES])
    model = create_model(settings.model_type, settings.random_seed).fit(x_tr, y_tr)
    return compute_metrics(y_te.to_numpy(), model.predict_proba(x_te)[:, 1])["auc"]


def track_training(
    model: Any, x_te: pd.DataFrame, metrics: dict, meta: dict, plot: Path
) -> str | None:
    """Loga o run em 'credit-training' e registra a versão; devolve a versão (ou None)."""
    tags = {k: str(meta[k]) for k in ("dataset_sha256", "git_sha", "contract_version")}
    with tracking.tracked_run(tracking.EXP_TRAINING, f"train-{meta['model_type']}", tags) as run:
        tracking.log_params({"model_type": meta["model_type"], "n_features": len(MODEL_FEATURES)})
        tracking.log_metrics({k: v for k, v in metrics.items() if k != "confusion"})
        tracking.log_artifact(plot)
        if run is None:
            return None
        return registry.log_model(model, meta["model_type"], x_te, model.predict_proba(x_te)[:, 1])


def run(settings: Settings | None = None) -> dict:
    """Treina, avalia, grava o candidate e promove se passar no gate."""
    settings = settings or get_settings()
    ref = load_reference(settings)
    validate_reference(ref, settings)
    train, test = split_train_test(ref, settings.random_seed)
    (x_tr, y_tr), (x_te, y_te) = build_xy(train), build_xy(test)
    model = create_model(settings.model_type, settings.random_seed).fit(x_tr, y_tr)
    score = model.predict_proba(x_te)[:, 1]
    metrics = compute_metrics(y_te.to_numpy(), score, settings.decision_threshold)
    metrics["auc_with_protected_attrs"] = auc_with_protected(train, test, settings)
    return finalize(model, x_te, metrics, settings)


def finalize(model: Any, x_te: pd.DataFrame, metrics: dict, settings: Settings) -> dict:
    """Persiste o candidate, registra no MLflow e executa o gate de promoção."""
    models_dir = settings.path("models")
    meta = {
        "dataset_sha256": file_sha256(settings.reference_path),
        "git_sha": git_sha(),
        "contract_version": CONTRACT_VERSION,
        "metrics": metrics,
    }
    save_model(model, settings.model_type, models_dir / "candidate", meta)
    plot = plot_confusion(metrics["confusion"], models_dir / "confusion_matrix.png")
    write_metrics(metrics, models_dir / "metrics.json")
    version = track_training(model, x_te, metrics, meta | {"model_type": settings.model_type}, plot)
    registry.promote_if_passes(
        models_dir / "candidate", models_dir / "champion", metrics, version, settings
    )
    logger.info("treino_concluido", extra={"auc": metrics["auc"], "version": version})
    return metrics


def main() -> int:
    """Entrypoint: ``python -m src.models.train`` (exit 3 se o gate reprovar)."""
    try:
        run()
    except GateFailedError as exc:
        logger.error("gate_reprovado", extra={"detail": str(exc)})
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
