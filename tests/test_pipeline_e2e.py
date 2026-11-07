"""E2E: `demo --synthetic` ponta a ponta, entrypoints reais, exit codes e logs sem PII."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import get_settings
from src.pipeline import cli
from src.pipeline.demo import isolate_environment

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    """Roda o demo UMA vez (≈ minutos) e devolve o workspace + trecho novo do log."""
    out = tmp_path_factory.mktemp("demo")
    log = ROOT / "logs" / "pipeline.jsonl"
    before = log.stat().st_size if log.exists() else 0
    saved_env = dict(os.environ)
    try:
        assert cli.main(["demo", "--synthetic", "--out", str(out)]) == 0
        new_log = log.read_bytes()[before:].decode("utf-8")
        yield out, new_log
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        get_settings.cache_clear()


def test_demo_generates_all_artifacts(demo_run):
    out, _ = demo_run
    for rel in (
        "reports/index.html", "reports/drift/drift_summary.csv", "reports/drift/drift_summary.json",
        "reports/drift/M1_data_drift.html", "reports/drift/M6_data_drift.html",
        "reports/causal/attribution.csv", "reports/causal/attribution.png",
        "reports/fairness/fairness_report.json", "reports/retrain/retrain_summary.json",
        "reports/contracts/reference_violations.json", "models/champion/model.json",
        "production/pool.parquet",
    ):  # fmt: skip
        assert (out / rel).exists(), rel


def test_demo_control_window_has_no_alert_and_later_windows_drift(demo_run):
    out, _ = demo_run
    summary = json.loads((out / "reports/drift/drift_summary.json").read_text("utf-8"))
    assert summary["M1"]["severity"] == "OK" and summary["M1"]["n_drifted"] == 0
    assert summary["M6"]["n_drifted"] >= 2 and summary["M6"]["max_psi"] > 0.2


def test_demo_panel_has_expected_sections(demo_run):
    out, _ = demo_run
    page = (out / "reports/index.html").read_text("utf-8")
    for section in ("Saúde das tasks", "Data contract", "Drift de dados", "causal", "fairness"):
        assert section.lower() in page.lower(), section
    assert "data:image/png;base64" in page and "M6_data_drift.html" in page


def test_logs_contain_no_personal_data(demo_run):
    _, new_log = demo_run
    assert new_log.strip()
    for line in new_log.splitlines():
        event = json.loads(line)
        assert "client_id" not in event and "client_hash" not in event and "sex" not in event
        assert not re.search(r"\b[0-9a-f]{64}\b", line)


def test_demo_did_not_touch_project_data(demo_run):
    out, _ = demo_run
    assert ROOT not in out.parents


def test_cli_exit_codes(settings, raw_df, tmp_path):
    assert cli.main(["monitor"]) == 1  # sem dados/modelo: erro genérico
    raw_dir = settings.path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_dir / "credit_default.csv", index=False)
    assert cli.main(["prepare"]) == 0
    settings.min_auc = 0.999
    assert cli.main(["train"]) == 3
    bad = raw_df.copy()
    bad.loc[0, "age"] = 3
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    assert cli.main(["validate", str(path)]) == 2


@pytest.mark.parametrize(
    "module",
    ["src.models.evaluate", "src.models.train", "scripts.validate_env", "scripts.apply_retention"],
)
def test_modules_have_real_main_blocks(module):
    source = (ROOT / (module.replace(".", "/") + ".py")).read_text("utf-8")
    assert 'if __name__ == "__main__"' in source


def test_apply_retention_runs_as_module():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.apply_retention"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert result.returncode == 0


def test_isolate_environment_redirects_everything(tmp_path, monkeypatch):
    for key in ("MODELS_DIR", "MLFLOW_TRACKING_URI", "PUSHGATEWAY_URL", "MIN_AUC"):
        monkeypatch.delenv(key, raising=False)
    saved_env = dict(os.environ)
    try:
        isolate_environment(tmp_path)
        settings = get_settings()
        assert tmp_path in settings.path("models").parents
        assert settings.pushgateway_url == ""
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        get_settings.cache_clear()
