"""Métricas Prometheus (prefixo ``credit_``) com Pushgateway e fallback.

O pipeline é BATCH, então as métricas são empurradas ao Pushgateway. Usa-se um
``CollectorRegistry`` próprio (testes isolados). Labels só de conjuntos FECHADOS
(lição 13.7): valores fora do conjunto viram ``other``.
"""

from __future__ import annotations

import time

from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

from src.constants import MODEL_FEATURES, WINDOWS
from src.contracts.rules import CATALOG
from src.logger import get_logger

logger = get_logger(__name__)

JOB = "credit_pipeline"
PUSH_TIMEOUT_S = 3
TASKS = [
    "prepare", "validate", "train", "simulate", "monitor", "causal", "fairness", "report",
    "retrain",
]  # fmt: skip
STAGES = ["ingest", "validate", "score", "drift"]
BATCHES = ["reference", "pool", "incoming", *WINDOWS]
ALERT_TYPES = [
    "data_drift", "contract_violation", "model_degradation", "task_failed", "stale_pipeline",
]  # fmt: skip
SEVERITIES = ["warning", "critical"]
OTHER = "other"


def closed(value: str, allowed: list[str]) -> str:
    """Restringe um label ao conjunto fechado (evita explosão de cardinalidade)."""
    return value if value in allowed else OTHER


# atributo -> (tipo, nome Prometheus, descrição, labels). Counter expõe o sufixo "_total".
METRIC_SPECS: dict[str, tuple[type, str, str, list[str]]] = {
    "task_status": (Gauge, "credit_task_status", "1 ok / 0 falha", ["task"]),
    "task_duration": (Gauge, "credit_task_duration_seconds", "Duração da task", ["task"]),
    "task_memory": (Gauge, "credit_task_peak_memory_bytes", "Memória de pico da task", ["task"]),
    "rows_processed": (Counter, "credit_rows_processed", "Linhas processadas", ["stage"]),
    "rows_rejected": (Counter, "credit_rows_rejected", "Linhas rejeitadas por regra", ["rule"]),
    "contract_status": (Gauge, "credit_contract_status", "1 aprovado / 0 reprovado", ["batch"]),
    "feature_psi": (Gauge, "credit_feature_psi", "PSI por feature e janela", ["feature", "window"]),
    "drift_share": (Gauge, "credit_drift_share", "Fração de features em drift", ["window"]),
    "model_auc": (Gauge, "credit_model_auc", "AUC por janela", ["window"]),
    "calibration_gap": (Gauge, "credit_calibration_gap", "Previsto - observado", ["window"]),
    "alerts": (Counter, "credit_alerts", "Alertas emitidos", ["type", "severity"]),
    "last_success": (Gauge, "credit_last_success_timestamp", "Epoch do último sucesso", ["task"]),
}  # fmt: skip


class PipelineMetrics:
    """Conjunto de métricas do pipeline em um registry próprio."""

    task_status: Gauge
    task_duration: Gauge
    task_memory: Gauge
    rows_processed: Counter
    rows_rejected: Counter
    contract_status: Gauge
    feature_psi: Gauge
    drift_share: Gauge
    model_auc: Gauge
    calibration_gap: Gauge
    alerts: Counter
    last_success: Gauge

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        for attr, (kind, name, doc, labels) in METRIC_SPECS.items():
            setattr(self, attr, kind(name, doc, labels, registry=self.registry))

    def set_task(self, task: str, ok: bool, duration: float, memory: float) -> None:
        """Registra status, duração, memória e (se ok) o instante do último sucesso."""
        task = closed(task, TASKS)
        self.task_status.labels(task=task).set(1 if ok else 0)
        self.task_duration.labels(task=task).set(duration)
        self.task_memory.labels(task=task).set(memory)
        if ok:
            self.last_success.labels(task=task).set(time.time())

    def set_window(self, window: str, drift_share: float, auc: float, gap: float) -> None:
        """Registra as métricas agregadas de uma janela."""
        window = closed(window, WINDOWS)
        self.drift_share.labels(window=window).set(drift_share)
        self.model_auc.labels(window=window).set(auc)
        self.calibration_gap.labels(window=window).set(gap)

    def set_psi(self, feature: str, window: str, value: float) -> None:
        """Registra o PSI de uma feature numa janela."""
        feature = closed(feature, MODEL_FEATURES)
        self.feature_psi.labels(feature=feature, window=closed(window, WINDOWS)).set(value)

    def count_rows(self, stage: str, n: int) -> None:
        """Incrementa linhas processadas."""
        self.rows_processed.labels(stage=closed(stage, STAGES)).inc(n)

    def count_rejected(self, rule: str, n: int) -> None:
        """Incrementa linhas rejeitadas por regra do contrato."""
        self.rows_rejected.labels(rule=closed(rule, list(CATALOG))).inc(n)

    def set_contract(self, batch: str, passed: bool) -> None:
        """Registra o status do contrato de um lote."""
        self.contract_status.labels(batch=closed(batch, BATCHES)).set(1 if passed else 0)

    def count_alert(self, alert_type: str, severity: str) -> None:
        """Incrementa o contador de alertas."""
        self.alerts.labels(
            type=closed(alert_type, ALERT_TYPES), severity=closed(severity, SEVERITIES)
        ).inc()

    def push(self, url: str, group: str) -> bool:
        """Empurra ao Pushgateway; sem URL ou com falha, degrada com aviso e segue."""
        if not url:
            return False
        try:
            push_to_gateway(
                url, job=JOB, registry=self.registry,
                grouping_key={"pipeline": group}, timeout=PUSH_TIMEOUT_S,
            )  # fmt: skip
            return True
        except OSError as exc:
            logger.warning(
                "observability_degraded", extra={"backend": "pushgateway", "error": str(exc)}
            )
            return False


_current: PipelineMetrics | None = None


def get_metrics() -> PipelineMetrics:
    """Instância corrente (criada sob demanda)."""
    global _current
    if _current is None:
        _current = PipelineMetrics()
    return _current


def reset_metrics() -> PipelineMetrics:
    """Cria um registry novo (início de execução da CLI e isolamento de testes)."""
    global _current
    _current = PipelineMetrics()
    return _current
