"""Testes de métricas Prometheus, alertas (Observer), tasks instrumentadas e tracking."""

from __future__ import annotations

import json

import pytest
from prometheus_client import generate_latest

from src.contracts.badbatch import make_bad_batch
from src.contracts.validator import validate_batch
from src.monitoring import tracking
from src.monitoring.alerts import (
    Alert,
    AlertDispatcher,
    AlertSink,
    LogSink,
    MlflowSink,
    PrometheusSink,
    WebhookSink,
    default_dispatcher,
)
from src.monitoring.metrics import PipelineMetrics, closed, reset_metrics
from src.pipeline.tasks import Task, peak_memory_bytes

pytestmark = pytest.mark.unit


def _text(metrics: PipelineMetrics) -> str:
    return generate_latest(metrics.registry).decode()


# ------------------------------------------------------------------ métricas
def test_all_named_metrics_exposed():
    m = PipelineMetrics()
    m.set_task("train", True, 1.5, 1024)
    m.count_rows("score", 10)
    m.count_rejected("H3", 2)
    m.set_contract("incoming", False)
    m.set_psi("limit_bal", "M6", 0.3)
    m.set_window("M6", 0.16, 0.75, -0.04)
    m.count_alert("data_drift", "critical")
    text = _text(m)
    for name in (
        "credit_task_status", "credit_task_duration_seconds", "credit_task_peak_memory_bytes",
        "credit_rows_processed_total", "credit_rows_rejected_total", "credit_contract_status",
        "credit_feature_psi", "credit_drift_share", "credit_model_auc",
        "credit_calibration_gap", "credit_alerts_total", "credit_last_success_timestamp",
    ):  # fmt: skip
        assert name in text, name


def test_failed_task_has_no_last_success():
    m = PipelineMetrics()
    m.set_task("monitor", False, 1.0, 1)
    assert 'credit_task_status{task="monitor"} 0.0' in _text(m)
    assert "credit_last_success_timestamp{" not in _text(m)


def test_labels_are_closed_sets():
    m = PipelineMetrics()
    for i in range(50):
        m.set_psi(f"feature_inventada_{i}", "M99", 0.1)
        m.count_rejected(f"R{i}", 1)
    text = _text(m)
    assert text.count("credit_feature_psi{") == 1 and 'feature="other"' in text
    assert text.count("credit_rows_rejected_total{") == 1
    assert closed("M3", ["M3"]) == "M3" and closed("x", ["M3"]) == "other"


def test_push_without_url_or_unreachable_degrades_gracefully():
    m = PipelineMetrics()
    assert m.push("", "train") is False
    assert m.push("http://127.0.0.1:1", "train") is False


def test_registries_are_isolated():
    a, b = PipelineMetrics(), PipelineMetrics()
    a.count_rows("score", 5)
    assert "credit_rows_processed_total{" not in _text(b)


# ------------------------------------------------------------------- alertas
class _Recorder(AlertSink):
    def __init__(self):
        self.received = []

    def send(self, alert):
        self.received.append(alert)


class _Broken(AlertSink):
    def send(self, alert):
        raise RuntimeError("canal fora do ar")


def test_observer_notifies_all_sinks_and_isolates_failures():
    dispatcher, rec = AlertDispatcher(), _Recorder()
    dispatcher.register(_Broken())
    dispatcher.register(rec)
    dispatcher.notify(Alert("data_drift", "critical", "teste"))
    assert len(rec.received) == 1


def test_log_sink_writes_jsonl(tmp_path):
    path = tmp_path / "alerts.jsonl"
    LogSink(path).send(Alert("data_drift", "critical", "PSI alto", "M6", {"top": ["a"]}))
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["type"] == "data_drift" and record["window"] == "M6"


def test_prometheus_sink_counts_alert():
    metrics = reset_metrics()
    PrometheusSink().send(Alert("contract_violation", "critical", "x"))
    assert 'credit_alerts_total{severity="critical",type="contract_violation"} 1.0' in _text(
        metrics
    )


def test_webhook_off_by_default(settings, monkeypatch):
    assert not any(isinstance(s, WebhookSink) for s in default_dispatcher(settings).sinks)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://localhost:9/hook")
    settings.alert_webhook_url = "http://localhost:9/hook"
    assert any(isinstance(s, WebhookSink) for s in default_dispatcher(settings).sinks)
    WebhookSink("").send(Alert("data_drift", "warning", "sem url: não faz nada"))


def test_mlflow_sink_without_active_run_is_noop():
    MlflowSink().send(Alert("data_drift", "warning", "x"))


# --------------------------------------------------------------------- tasks
class _OkTask(Task):
    name = "report"

    def execute(self):
        self.rows = 7
        return "ok"


class _FailTask(Task):
    name = "monitor"

    def execute(self):
        raise ValueError("boom")


def test_task_template_method_success():
    metrics = reset_metrics()
    assert _OkTask(metrics=metrics).run() == "ok"
    text = _text(metrics)
    assert 'credit_task_status{task="report"} 1.0' in text
    assert "credit_task_peak_memory_bytes" in text and "credit_last_success_timestamp" in text


def test_task_failure_sets_status_alerts_and_reraises():
    metrics, rec = reset_metrics(), _Recorder()
    dispatcher = AlertDispatcher()
    dispatcher.register(rec)
    with pytest.raises(ValueError, match="boom"):
        _FailTask(dispatcher, metrics).run()
    assert 'credit_task_status{task="monitor"} 0.0' in _text(metrics)
    assert rec.received[0].type == "task_failed"


def test_peak_memory_positive():
    assert peak_memory_bytes() > 0


# ------------------------------------------------------------------ tracking
def test_record_validation_updates_metrics_and_mlflow(ingested_df, settings):
    metrics = reset_metrics()
    result = validate_batch(make_bad_batch(ingested_df, n=300))
    tracking.record_validation("incoming", result)
    text = _text(metrics)
    assert 'credit_contract_status{batch="incoming"} 0.0' in text
    assert 'credit_rows_rejected_total{rule="H3"}' in text


def test_tracked_run_degrades_without_backend(settings):
    settings.mlflow_tracking_uri = "http://127.0.0.1:1"
    with tracking.tracked_run("x", "y", settings=settings) as run:
        assert run is None
    tracking.log_metrics({"a": 1.0})
    assert tracking.safe(lambda: 1 / 0) is None
