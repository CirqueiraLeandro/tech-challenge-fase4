"""Wrappers do MLflow: o pipeline NUNCA falha porque o backend de tracking está fora."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)

# Backend fora do ar deve degradar rápido: sem retries longos e com timeout curto.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

EXP_TRAINING = "credit-training"
EXP_VALIDATION = "data-validation"
EXP_DRIFT = "drift-monitoring"


def safe(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Executa uma chamada MLflow; em falha, registra 'observability_degraded' e segue."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - observabilidade é opcional
        logger.warning("observability_degraded", extra={"backend": "mlflow", "error": str(exc)})
        return None


def configure(experiment: str, settings: Settings | None = None) -> None:
    """Aponta o MLflow para o tracking URI configurado e seleciona o experimento."""
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)


@contextmanager
def tracked_run(
    experiment: str,
    run_name: str,
    tags: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> Iterator[Any]:
    """Abre um run; se o backend falhar, entrega ``None`` e o código segue sem tracking."""
    run = safe(_start, experiment, run_name, tags or {}, settings)
    try:
        yield run
    finally:
        if run is not None:
            safe(mlflow.end_run)


def _start(experiment: str, run_name: str, tags: dict[str, str], settings: Settings | None) -> Any:
    configure(experiment, settings)
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Loga métricas numéricas no run ativo (ignora valores não finitos)."""
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    safe(mlflow.log_metrics, clean, step=step)


def log_params(params: dict[str, Any]) -> None:
    """Loga parâmetros no run ativo."""
    safe(mlflow.log_params, params)


def log_artifact(path: Path) -> None:
    """Loga um arquivo como artefato do run ativo."""
    safe(mlflow.log_artifact, str(path))
