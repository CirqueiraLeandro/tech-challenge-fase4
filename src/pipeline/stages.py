"""Estágios do pipeline: cada Task só implementa ``execute()`` e delega a funções pequenas.

A CLI e os orquestradores chamam EXATAMENTE as mesmas funções de módulo (sem duplicação).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DriftScenario, Settings, get_settings, load_drift_scenario
from src.contracts.rules import ReferenceStats, compute_reference_stats
from src.contracts.validator import ingest_and_validate
from src.data.loader import load_pool, load_raw, load_reference, raw_csv_path
from src.data.splitter import save_split, split_reference_pool
from src.drift import causal, monitor
from src.drift.simulation import save_windows, simulate_windows
from src.governance import fairness
from src.logger import get_logger
from src.models import retrain, train
from src.monitoring import report
from src.pipeline.tasks import Task

logger = get_logger(__name__)
SYNTHETIC_WINDOW_FRACTION = 0.6


def prepare(settings: Settings | None = None, raw: pd.DataFrame | None = None) -> dict[str, int]:
    """Split reference/pool, pseudonimiza o ID e valida ambos no contrato antes de salvar."""
    settings = settings or get_settings()
    raw = raw if raw is not None else load_raw(raw_csv_path(settings))
    ref, pool = split_reference_pool(raw, settings.random_seed)
    ref, _ = ingest_and_validate(ref, "reference", settings=settings)
    stats = compute_reference_stats(ref)
    pool, _ = ingest_and_validate(pool, "pool", stats=stats, settings=settings)
    save_split(ref, pool, settings)
    return {"reference": len(ref), "pool": len(pool)}


def reference_stats(settings: Settings | None = None) -> ReferenceStats:
    """Estatísticas do reference para as regras suaves."""
    return compute_reference_stats(load_reference(settings))


def validate_file(path: Path, settings: Settings | None = None) -> None:
    """Valida um CSV isolado; lança ContractViolationError se reprovado."""
    settings = settings or get_settings()
    df = pd.read_csv(path, encoding="utf-8")
    stats = reference_stats(settings) if settings.reference_path.exists() else None
    ingest_and_validate(df, path.stem, stats=stats, settings=settings)


def effective_scenario(scenario: DriftScenario, pool_rows: int) -> DriftScenario:
    """Reduz window_size a 60% do bloco quando o pool é pequeno (demo sintética)."""
    chunk = pool_rows // len(scenario.windows)
    if scenario.window_size <= chunk:
        return scenario
    size = max(int(chunk * SYNTHETIC_WINDOW_FRACTION), 1)
    logger.info("window_size_ajustado", extra={"configured": scenario.window_size, "used": size})
    return scenario.model_copy(update={"window_size": size})


def simulate(settings: Settings | None = None) -> dict[str, int]:
    """Gera e persiste as janelas M1..M6 a partir do pool."""
    settings = settings or get_settings()
    pool = load_pool(settings)
    scenario = effective_scenario(load_drift_scenario(), len(pool))
    windows = simulate_windows(pool, scenario, settings.random_seed)
    save_windows(windows, settings)
    return {name: len(w.data) for name, w in windows.items()}


class PrepareTask(Task):
    """Split + contrato do reference e do pool."""

    name = "prepare"

    def execute(self, settings: Settings | None = None, raw: pd.DataFrame | None = None) -> Any:
        result = prepare(settings, raw)
        self.rows = sum(result.values())
        return result


class ValidateTask(Task):
    """Contrato isolado sobre um arquivo."""

    name = "validate"

    def execute(self, path: Path, settings: Settings | None = None) -> Any:
        validate_file(path, settings)


class TrainTask(Task):
    """Treino, gate e promoção."""

    name = "train"

    def execute(self, settings: Settings | None = None) -> Any:
        return train.run(settings)


class SimulateTask(Task):
    """Simulação causal das janelas."""

    name = "simulate"

    def execute(self, settings: Settings | None = None) -> Any:
        result = simulate(settings)
        self.rows = sum(result.values())
        return result


class MonitorTask(Task):
    """Contrato -> score -> drift -> performance -> política -> alertas."""

    name = "monitor"

    def execute(self, settings: Settings | None = None) -> Any:
        summary = monitor.monitor_windows(settings, self.dispatcher, self.metrics)
        self.rows = sum(s["n_rows"] for s in summary.values())
        return summary


class CausalTask(Task):
    """Análise causal intervencional x observacional."""

    name = "causal"

    def execute(self, settings: Settings | None = None) -> Any:
        return causal.run(settings=settings)


class FairnessTask(Task):
    """Auditoria de vieses."""

    name = "fairness"

    def execute(self, settings: Settings | None = None) -> Any:
        return fairness.run(settings=settings)


class ReportTask(Task):
    """Painel HTML consolidado."""

    name = "report"

    def execute(self, settings: Settings | None = None) -> Any:
        return report.build_report(settings)


class RetrainTask(Task):
    """Retreino com janelas recentes rotuladas."""

    name = "retrain"

    def execute(self, windows: list[str], settings: Settings | None = None) -> Any:
        return retrain.run(windows, settings=settings)
