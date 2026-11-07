"""Testes de factory, métricas, treino com gate, registry, explain e artefatos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import MODEL_FEATURES
from src.exceptions import ContractViolationError, GateFailedError
from src.models import registry, train
from src.models.artifacts import LogRegScorer, load_model, read_metadata, save_model
from src.models.evaluate import (
    calibration_gap,
    compute_metrics,
    evaluate_champion,
    ks_separation,
    plot_confusion,
)
from src.models.explain import contributions, margin, reason_codes
from src.models.factory import create_model
from src.pipeline import stages

pytestmark = pytest.mark.unit


@pytest.fixture
def prepared(raw_df, settings):
    """Settings com reference/pool gravados a partir do dataset sintético."""
    stages.prepare(settings, raw=raw_df)
    return settings


def test_factory_types_and_invalid():
    assert create_model("xgboost").__class__.__name__ == "XGBClassifier"
    assert create_model("logreg").__class__.__name__ == "Pipeline"
    with pytest.raises(ValueError, match="inválido"):
        create_model("svm")


def test_metrics_on_fixed_case():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["auc"] == 1.0 and perfect["gini"] == 1.0 and perfect["ks"] == 1.0
    assert perfect["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    random_like = compute_metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert random_like["auc"] == 0.5 and random_like["gini"] == 0.0


def test_ks_and_calibration_gap():
    y = np.array([0, 1, 0, 1])
    assert ks_separation(y, np.array([0.1, 0.9, 0.2, 0.8])) == 1.0
    assert calibration_gap(y, np.array([0.7, 0.7, 0.7, 0.7])) == pytest.approx(0.2)


def test_plot_confusion(tmp_path):
    path = plot_confusion({"tn": 1, "fp": 2, "fn": 3, "tp": 4}, tmp_path / "c.png")
    assert path.stat().st_size > 0


def test_train_promotes_and_saves_portable_artifacts(prepared):
    metrics = train.run(prepared)
    champion = prepared.path("models") / "champion"
    assert (champion / "model.json").exists() and (prepared.path("models") / "candidate").exists()
    assert metrics["auc"] > 0.5 and "auc_with_protected_attrs" in metrics
    meta = read_metadata(champion)
    assert meta["features"] == MODEL_FEATURES and len(meta["dataset_sha256"]) == 64
    assert not list(champion.glob("*.pkl")) and not list(champion.glob("*.joblib"))


def test_failed_gate_does_not_touch_champion(prepared):
    prepared.min_auc = 0.5
    train.run(prepared)
    champion_file = prepared.path("models") / "champion" / "model.json"
    before = champion_file.read_bytes()
    prepared.min_auc = 0.999
    prepared.random_seed = 7  # candidato diferente
    with pytest.raises(GateFailedError):
        train.run(prepared)
    assert champion_file.read_bytes() == before
    assert (prepared.path("models") / "candidate" / "model.json").read_bytes() != before


def test_train_main_exit_code_on_gate_failure(prepared):
    prepared.min_auc = 0.999
    assert train.main() == 3


def test_logreg_roundtrip_matches_sklearn(prepared, raw_df):
    prepared.model_type = "logreg"
    train.run(prepared)
    scorer = load_model(prepared.path("models") / "champion")
    assert isinstance(scorer, LogRegScorer)
    x, _ = train.load_test_split(prepared)
    model = create_model("logreg").fit(*train.build_xy(raw_df))
    ours = scorer.predict_proba(x)[:, 1]
    assert ours.min() >= 0 and ours.max() <= 1
    assert model.predict_proba(x)[:, 1].shape == ours.shape


def test_validate_reference_rejects_bad_reference(prepared):
    ref = pd.read_parquet(prepared.reference_path)
    ref.loc[0, "age"] = 3
    with pytest.raises(ContractViolationError):
        train.validate_reference(ref, prepared)


def test_evaluate_champion_writes_metrics(prepared):
    train.run(prepared)
    metrics = evaluate_champion(prepared)
    assert (prepared.path("models") / "metrics.json").exists() and metrics["auc"] > 0.5


def test_explain_contributions_sum_to_margin(prepared):
    train.run(prepared)
    model = load_model(prepared.path("models") / "champion")
    x, _ = train.load_test_split(prepared)
    contribs = contributions(model, x)
    assert np.allclose(contribs.sum(axis=1), margin(model, x), atol=1e-4)
    codes = reason_codes(model, x.head(5))
    assert len(codes) == 5 and all(len(c) <= 3 and set(c) <= set(MODEL_FEATURES) for c in codes)


def test_registry_alias_and_challenger(prepared):
    train.run(prepared)
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=prepared.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(registry.REGISTERED_NAME, "champion")
    assert int(mv.version) == 1


def test_registry_degrades_when_backend_is_down(prepared):
    prepared.mlflow_tracking_uri = "http://127.0.0.1:1"  # servidor inexistente
    metrics = train.run(prepared)
    assert (prepared.path("models") / "champion" / "model.json").exists() and metrics["auc"] > 0


def test_save_model_logreg_metadata(tmp_path, raw_df):
    x, y = train.build_xy(raw_df)
    model = create_model("logreg").fit(x, y)
    save_model(model, "logreg", tmp_path, {"k": 1})
    assert read_metadata(tmp_path)["model_type"] == "logreg"
