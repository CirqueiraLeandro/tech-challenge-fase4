"""Testes de configuração e loaders YAML."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import DriftScenario, Settings, load_drift_scenario, load_monitoring_config

pytestmark = pytest.mark.unit


def test_settings_defaults_in_test_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert Settings().random_seed == 42


def test_key_required_outside_test(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PSEUDONYMIZATION_KEY", "")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_override(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("MIN_AUC", "0.8")
    assert Settings().min_auc == 0.8


def test_scenario_loads():
    scenario = load_drift_scenario()
    assert scenario.windows[0] == "M1" and scenario.intensities[0] == 0.0


def test_scenario_rejects_bad_intensity():
    with pytest.raises(ValidationError):
        DriftScenario(windows=["M1"], intensities=[1.5], window_size=10, mechanisms={})


def test_scenario_rejects_unknown_mechanism():
    with pytest.raises(ValidationError):
        DriftScenario(
            windows=["M1"], intensities=[0.0], window_size=10, mechanisms={"foo": {"enabled": True}}
        )


def test_monitoring_config_loads():
    assert load_monitoring_config().psi_severe == 0.20
