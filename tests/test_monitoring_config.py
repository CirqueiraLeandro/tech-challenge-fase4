"""Valida os artefatos de infraestrutura: dashboard Grafana, regras de alerta e compose."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from src.monitoring.metrics import PipelineMetrics

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "monitoring"


def _metric_names() -> set[str]:
    names = set()
    for metric in PipelineMetrics().registry.collect():
        names.add(metric.name)
        names.update(s.name for s in metric.samples)
    names |= {"credit_rows_processed_total", "credit_rows_rejected_total", "credit_alerts_total"}
    return names


def _referenced(expr: str) -> set[str]:
    return set(re.findall(r"credit_[a-z_]+", expr))


def test_dashboard_has_at_least_8_panels_with_known_metrics():
    dash = json.loads((MON / "grafana/dashboards/credit_monitoring.json").read_text("utf-8"))
    assert len(dash["panels"]) >= 8
    known = _metric_names()
    for panel in dash["panels"]:
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            refs = _referenced(target["expr"])
            assert refs and refs <= known, (panel["title"], refs - known)


def test_dashboard_ids_unique_and_datasource_matches_provisioning():
    dash = json.loads((MON / "grafana/dashboards/credit_monitoring.json").read_text("utf-8"))
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    ds = yaml.safe_load(
        (MON / "grafana/provisioning/datasources/prometheus.yml").read_text("utf-8")
    )
    assert ds["datasources"][0]["uid"] == dash["panels"][0]["datasource"]["uid"]


def test_alert_rules_parse_and_cover_required_alerts():
    rules = yaml.safe_load((MON / "prometheus/alert_rules.yml").read_text("utf-8"))
    alerts = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}
    assert set(alerts) == {
        "DataDriftCritical", "ModelAUCBelowSLO", "ContractViolation", "PipelineTaskFailed",
        "PipelineStale",
    }  # fmt: skip
    known = _metric_names()
    for rule in alerts.values():
        assert _referenced(rule["expr"]) <= known and rule["labels"]["severity"]
        assert rule["annotations"]["runbook"].startswith("docs/MONITORING.md")


def test_prometheus_scrapes_pushgateway_with_honor_labels():
    cfg = yaml.safe_load((MON / "prometheus/prometheus.yml").read_text("utf-8"))
    job = cfg["scrape_configs"][0]
    assert job["honor_labels"] is True and "alert_rules.yml" in cfg["rule_files"]


def test_compose_pins_image_versions_and_ports():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))
    services = compose["services"]
    assert set(services) == {"mlflow", "pushgateway", "prometheus", "grafana"}
    for name, svc in services.items():
        tag = svc["image"].split(":")[-1]
        assert tag != "latest" and re.search(r"\d", tag), name


def test_slo_in_rules_matches_monitoring_yaml():
    from src.config import load_monitoring_config

    rules = (MON / "prometheus/alert_rules.yml").read_text("utf-8")
    assert f"credit_model_auc < {load_monitoring_config().auc_slo}" in rules
