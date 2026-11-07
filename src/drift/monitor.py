"""Monitoramento das janelas: contrato -> score -> drift -> performance -> política -> alertas.

Para cada janela M1..M6: valida o contrato (lote reprovado NÃO é pontuado), pontua com o
champion, mede drift de dados (testes próprios + Evidently) e de conceito (AUC, calibração),
aplica a política e emite métricas, alertas e runs MLflow (um por janela, ``step=k``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config import MonitoringConfig, Settings, get_settings, load_monitoring_config
from src.constants import TARGET
from src.contracts.rules import ReferenceStats, compute_reference_stats
from src.contracts.validator import validate_batch
from src.data.loader import load_reference
from src.drift import policy
from src.drift.detector import DriftDetector, cross_check, drift_share
from src.drift.performance import score_frame, window_performance
from src.drift.simulation import load_window
from src.drift.stattests import Bins
from src.logger import get_logger
from src.models.artifacts import load_model
from src.models.evaluate import compute_metrics
from src.models.train import split_train_test
from src.monitoring import tracking
from src.monitoring.alerts import Alert, AlertDispatcher
from src.monitoring.metrics import PipelineMetrics, get_metrics

logger = get_logger(__name__)


@dataclass
class MonitorContext:
    """Objetos reutilizados em todas as janelas (calculados UMA vez por run)."""

    settings: Settings
    cfg: MonitoringConfig
    model: Any
    detector: DriftDetector
    soft_stats: ReferenceStats
    baseline_auc: float
    score_bins: Bins


def build_context(settings: Settings) -> MonitorContext:
    """Prepara reference, champion, detector e baseline de AUC/score."""
    ref = load_reference(settings)
    model = load_model(settings.path("models") / "champion")
    _, test = split_train_test(ref, settings.random_seed)
    score = score_frame(model, test)
    baseline = compute_metrics(test[TARGET].to_numpy(), score, settings.decision_threshold)["auc"]
    cfg = load_monitoring_config()
    return MonitorContext(
        settings,
        cfg,
        model,
        DriftDetector(ref, cfg),
        compute_reference_stats(ref),
        baseline,
        Bins(score),
    )


def analyze_window(ctx: MonitorContext, name: str, df: pd.DataFrame) -> dict:
    """Analisa uma janela já aprovada no contrato; devolve o relatório estruturado."""
    html = ctx.settings.path("reports") / "drift" / f"{name}_data_drift.html"
    table = ctx.detector.analyze(df)
    evid = ctx.detector.evidently(df, html)
    perf, _ = window_performance(
        ctx.model, df, ctx.baseline_auc, ctx.score_bins, ctx.settings.decision_threshold
    )
    decision = policy.decide(
        float(table["psi"].max()), perf.auc_drop, perf.calibration_gap, ctx.cfg
    )
    return {
        "window": name,
        "n_rows": len(df),
        "table": table,
        "evidently": evid,
        "performance": perf,
        "decision": decision,
        "cross_check": cross_check(table, evid),
        "html": str(html),
    }


def emit_window(
    report: dict,
    step: int,
    ctx: MonitorContext,
    dispatcher: AlertDispatcher,
    metrics: PipelineMetrics,
) -> None:
    """Métricas Prometheus, alertas e run MLflow da janela."""
    name, table, perf = report["window"], report["table"], report["performance"]
    share = drift_share(table)
    metrics.set_window(name, share, perf.auc, perf.calibration_gap)
    for row in table.itertuples():
        metrics.set_psi(row.feature, name, float(row.psi))
    metrics.count_rows("score", report["n_rows"])
    metrics.count_rows("drift", report["n_rows"])
    _alerts(report, ctx.cfg, dispatcher)
    _log_mlflow(report, step, share)


def _alerts(report: dict, cfg: MonitoringConfig, dispatcher: AlertDispatcher) -> None:
    name, table, perf, decision = (
        report["window"],
        report["table"],
        report["performance"],
        report["decision"],
    )
    top = table.sort_values("psi", ascending=False).head(3)["feature"].tolist()
    if table["psi"].max() >= cfg.psi_severe:
        dispatcher.notify(
            Alert(
                "data_drift",
                "critical",
                f"PSI >= {cfg.psi_severe} em {name}",
                name,
                {"top_features": top},
            )
        )
    if perf.auc < cfg.auc_slo or decision.severity == policy.CRITICAL:
        dispatcher.notify(
            Alert(
                "model_degradation",
                "critical",
                f"{name}: {decision.severity}",
                name,
                {"auc": round(perf.auc, 4)},
            )
        )
    elif decision.severity == policy.ATTENTION:
        dispatcher.notify(
            Alert("model_degradation", "warning", f"{name}: {'; '.join(decision.reasons)}", name)
        )


def _log_mlflow(report: dict, step: int, share: float) -> None:
    name, table, perf, decision = (
        report["window"],
        report["table"],
        report["performance"],
        report["decision"],
    )
    with tracking.tracked_run(
        tracking.EXP_DRIFT, f"drift-{name}", {"window": name, "severity": decision.severity}
    ) as run:
        if run is None:
            return
        flat = {f"psi_{r.feature}": float(r.psi) for r in table.itertuples()}
        flat |= {f"ks_p_{r.feature}": float(r.ks_p) for r in table.itertuples()}
        flat |= {"drift_share": share, "n_drifted": int(table["drifted"].sum())}
        flat |= {"auc": perf.auc, "calib_gap": perf.calibration_gap, "auc_drop": perf.auc_drop}
        tracking.log_metrics(flat, step=step)
        tracking.safe(mlflow.set_tag, "action", decision.action)
        tracking.log_artifact(Path(report["html"]))


def summarize(reports: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Tabela longa window x feature e resumo por janela."""
    long = pd.concat(
        [
            r["table"].assign(window=r["window"]).merge(r["evidently"], on="feature")
            for r in reports
        ],
        ignore_index=True,
    )
    summary = {
        r["window"]: {
            "n_rows": r["n_rows"],
            "drift_share": drift_share(r["table"]),
            "n_drifted": int(r["table"]["drifted"].sum()),
            "drifted_features": r["table"].loc[r["table"]["drifted"], "feature"].tolist(),
            "max_psi": float(r["table"]["psi"].max()),
            **r["performance"].to_dict(),
            "severity": r["decision"].severity,
            "action": r["decision"].action,
            "reasons": r["decision"].reasons,
            "cross_check": r["cross_check"],
        }
        for r in reports
    }
    return long, summary


def write_outputs(long: pd.DataFrame, summary: dict, settings: Settings) -> Path:
    """Grava reports/drift/drift_summary.csv e .json."""
    out = settings.path("reports") / "drift"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "drift_summary.csv", index=False, encoding="utf-8")
    (out / "drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def blocked_window(name: str, result: Any, dispatcher: AlertDispatcher) -> None:
    """Lote reprovado no contrato: ingestão bloqueada, alerta crítico, janela não pontuada."""
    dispatcher.notify(
        Alert("contract_violation", "critical", f"{name} reprovada: {result.violated_rules}", name)
    )
    logger.error("janela_bloqueada", extra={"window": name, "rules": result.violated_rules})


def monitor_windows(
    settings: Settings | None = None,
    dispatcher: AlertDispatcher | None = None,
    metrics: PipelineMetrics | None = None,
    windows: list[str] | None = None,
) -> dict:
    """Executa o monitoramento de todas as janelas persistidas e grava os relatórios."""
    settings = settings or get_settings()
    dispatcher, metrics = dispatcher or AlertDispatcher(), metrics or get_metrics()
    ctx = build_context(settings)
    names = windows or sorted(
        p.stem for p in (settings.path("production") / "windows").glob("M*.parquet")
    )
    reports = []
    for step, name in enumerate(names, start=1):
        df = load_window(name, settings)
        result = validate_batch(df, ctx.soft_stats, settings=settings)
        tracking.record_validation(name, result)
        if not result.passed:
            blocked_window(name, result, dispatcher)
            continue
        report = analyze_window(ctx, name, df)
        emit_window(report, step, ctx, dispatcher, metrics)
        reports.append(report)
    long, summary = summarize(reports)
    write_outputs(long, summary, settings)
    return summary
