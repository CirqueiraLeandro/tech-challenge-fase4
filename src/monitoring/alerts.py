"""Alertas (padrão Observer): o dispatcher notifica sinks registrados.

Novo canal = nova classe ``AlertSink``. O webhook é desligado por padrão
(privacy by default) e falhas de um sink nunca derrubam o pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import requests

from src.config import Settings, get_settings
from src.logger import get_logger
from src.monitoring.metrics import PipelineMetrics, get_metrics

logger = get_logger(__name__)
WEBHOOK_TIMEOUT_S = 3


@dataclass(frozen=True)
class Alert:
    """Alerta operacional (sem dados pessoais)."""

    type: str
    severity: str
    message: str
    window: str | None = None
    details: dict = field(default_factory=dict)


class AlertSink(ABC):
    """Canal de notificação."""

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Entrega o alerta."""


class LogSink(AlertSink):
    """Grava em logs/alerts.jsonl e no log estruturado."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(UTC).isoformat(), **asdict(alert)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.warning("alerta", extra={"alert_type": alert.type, "severity": alert.severity})


class MlflowSink(AlertSink):
    """Anota o run ativo do MLflow com o alerta (se houver run)."""

    def send(self, alert: Alert) -> None:
        if mlflow.active_run() is not None:
            key = f"alert.{alert.type}.{alert.window or 'global'}"
            mlflow.set_tag(key, alert.severity)


class PrometheusSink(AlertSink):
    """Incrementa credit_alerts_total{type,severity}."""

    def __init__(self, metrics: PipelineMetrics | None = None) -> None:
        self.metrics = metrics

    def send(self, alert: Alert) -> None:
        (self.metrics or get_metrics()).count_alert(alert.type, alert.severity)


class WebhookSink(AlertSink):
    """POST JSON em ALERT_WEBHOOK_URL (desligado por padrão)."""

    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, alert: Alert) -> None:
        if self.url:
            requests.post(self.url, json=asdict(alert), timeout=WEBHOOK_TIMEOUT_S)


class AlertDispatcher:
    """Subject do Observer: notifica todos os sinks, isolando falhas."""

    def __init__(self) -> None:
        self.sinks: list[AlertSink] = []

    def register(self, sink: AlertSink) -> None:
        """Adiciona um canal."""
        self.sinks.append(sink)

    def notify(self, alert: Alert) -> None:
        """Envia o alerta a todos os canais; exceção de um canal não afeta os demais."""
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception as exc:  # noqa: BLE001 - sink não pode derrubar o pipeline
                logger.warning(
                    "observability_degraded",
                    extra={"backend": type(sink).__name__, "error": str(exc)},
                )


def default_dispatcher(settings: Settings | None = None) -> AlertDispatcher:
    """Dispatcher padrão: log + MLflow + Prometheus (+ webhook só se configurado)."""
    settings = settings or get_settings()
    dispatcher = AlertDispatcher()
    dispatcher.register(LogSink(settings.path("logs") / "alerts.jsonl"))
    dispatcher.register(MlflowSink())
    dispatcher.register(PrometheusSink())
    if settings.alert_webhook_url:
        dispatcher.register(WebhookSink(settings.alert_webhook_url))
    return dispatcher
