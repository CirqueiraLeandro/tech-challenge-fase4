"""MLflow Model Registry: registro, aliases (challenger/champion) e promoção com gate.

O treino grava SOMENTE em models/candidate/; apenas :func:`promote` escreve em
models/champion/ e move o alias (lição 13.2: o gate protege a produção).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.config import Settings
from src.exceptions import GateFailedError
from src.logger import get_logger
from src.monitoring.tracking import configure, safe

logger = get_logger(__name__)
REGISTERED_NAME = "credit-scoring"


def log_model(model: Any, model_type: str, x_sample: pd.DataFrame, score: Any) -> str | None:
    """Loga o modelo com signature e input_example e registra uma nova versão (ou None)."""
    signature = infer_signature(x_sample, score)
    kwargs = {
        "signature": signature,
        "input_example": x_sample.head(3),
        "registered_model_name": REGISTERED_NAME,
    }
    if model_type == "xgboost":
        info = safe(mlflow.xgboost.log_model, model.get_booster(), name="model", **kwargs)
    else:
        info = safe(mlflow.sklearn.log_model, model, name="model", **kwargs)
    return None if info is None else str(info.registered_model_version)


def set_alias(alias: str, version: str | None, settings: Settings | None = None) -> None:
    """Move um alias do Model Registry (ignora se o registry estiver indisponível)."""
    if version is None:
        return
    safe(_set_alias, alias, version, settings)


def _set_alias(alias: str, version: str, settings: Settings | None) -> None:
    configure("credit-training", settings)
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, alias, version)


def passes_gate(metrics: dict, min_auc: float) -> bool:
    """Gate de promoção: AUC >= MIN_AUC."""
    return metrics["auc"] >= min_auc


def promote(candidate: Path, champion: Path, version: str | None, settings: Settings) -> None:
    """Copia candidate -> champion (copyfile, sem copystat) e move o alias 'champion'."""
    champion.mkdir(parents=True, exist_ok=True)
    for item in candidate.iterdir():
        shutil.copyfile(item, champion / item.name)
    set_alias("champion", version, settings)
    logger.info("modelo_promovido", extra={"version": version})


def promote_if_passes(
    candidate: Path, champion: Path, metrics: dict, version: str | None, settings: Settings
) -> None:
    """Promove apenas se passar no gate; senão lança GateFailedError e preserva o champion."""
    if not passes_gate(metrics, settings.min_auc):
        set_alias("challenger", version, settings)
        raise GateFailedError(f"AUC {metrics['auc']:.4f} < MIN_AUC {settings.min_auc:.2f}")
    promote(candidate, champion, version, settings)
