"""Testes da análise causal e da auditoria de vieses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DriftScenario, MechanismConfig, MonitoringConfig
from src.data.synthetic import make_synthetic_credit
from src.drift.causal import (
    dominant,
    interventional_attribution,
    observational_attribution,
    quantile_revert,
)
from src.drift.detector import DriftDetector
from src.drift.monitor import MonitorContext
from src.drift.simulation import split_chunks
from src.drift.stattests import Bins
from src.governance.fairness import age_band, audit_attribute, audit_frame, group_metrics
from src.models.factory import create_model
from src.models.train import build_xy
from src.privacy.pseudonymize import pseudonymize_ids

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def causal_setup(tmp_path_factory):
    from src.config import Settings
    from src.drift.performance import score_frame

    settings = Settings(environment="test")
    pool = pseudonymize_ids(make_synthetic_credit(9000, seed=5), settings)
    ref, rest = pool.iloc[:3000].reset_index(drop=True), pool.iloc[3000:].reset_index(drop=True)
    x, y = build_xy(ref)
    model = create_model("xgboost").set_params(n_estimators=50).fit(x, y)
    cfg = MonitoringConfig()
    bins = Bins(score_frame(model, ref))
    ctx = MonitorContext(settings, cfg, model, DriftDetector(ref, cfg), None, 0.8, bins)
    return ctx, ref, rest


def _scenario(only: str) -> DriftScenario:
    mechanisms = {
        m: MechanismConfig(enabled=(m == only))
        for m in ("inflacao", "aperto", "desemprego", "perfil_jovem")
    }
    return DriftScenario(
        windows=["M1", "M2", "M3", "M4", "M5", "M6"],
        intensities=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        window_size=700,
        mechanisms=mechanisms,
    )


@pytest.mark.parametrize("only", ["aperto", "inflacao"])
def test_single_mechanism_is_identified_as_dominant_in_data(causal_setup, only):
    ctx, _, rest = causal_setup
    scenario = _scenario(only)
    chunk = split_chunks(rest, 6, 42)[5]
    table = interventional_attribution(ctx, chunk, 5, 1.0, scenario, 42)
    assert dominant(table, "delta_psi_sum") == only
    others = table[table["mechanism"] != only]
    assert (others["delta_psi_sum"].abs() < 1e-9).all()


def test_unemployment_alone_is_dominant_in_calibration(causal_setup):
    ctx, _, rest = causal_setup
    chunk = split_chunks(rest, 6, 42)[5]
    table = interventional_attribution(ctx, chunk, 5, 1.0, _scenario("desemprego"), 42)
    assert dominant(table, "delta_calibration_gap") == "desemprego"
    assert table.attrs["full"]["calibration_gap"] < table.attrs["none"]["calibration_gap"]


def test_observational_table_shape(causal_setup):
    ctx, ref, rest = causal_setup
    from src.drift.simulation import generate_window

    chunk = split_chunks(rest, 6, 42)[5]
    window = generate_window(chunk, 5, 1.0, _scenario("aperto"), 42).data
    obs = observational_attribution(ctx, window, ref)
    assert set(obs["mechanism"]) == {"inflacao", "aperto", "desemprego", "perfil_jovem"}
    assert obs.set_index("mechanism").loc["aperto", "delta_score_psi"] > 0


def test_quantile_revert_matches_reference_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(3, 2, 5000))
    out = quantile_revert(cur, ref)
    assert abs(out.mean() - ref.mean()) < 0.05 and abs(out.std() - ref.std()) < 0.05
    assert np.corrcoef(out, cur)[0, 1] > 0.99  # preserva a ordem


# ------------------------------------------------------------------ fairness
def test_group_metrics_fixed_case():
    y = np.array([0, 0, 1, 1])
    approved = np.array([True, True, False, True])
    out = group_metrics(y, approved, np.array([0.1, 0.2, 0.9, 0.4]))
    assert out["approval_rate"] == 0.75 and out["good_payer_approval_rate"] == 1.0
    assert out["auc"] == 1.0 and out["default_rate"] == 0.5


def test_disparate_impact_and_equal_opportunity_fixed_cases():
    n = 100
    df = pd.DataFrame({"default": [0] * n * 2})
    groups = pd.Series(["a"] * n + ["b"] * n)
    score = np.array([0.1] * 90 + [0.9] * 10 + [0.1] * 60 + [0.9] * 40)
    out = audit_attribute(df, groups, score, 0.5)
    assert out["groups"]["a"]["approval_rate"] == 0.9 and out["groups"]["b"]["approval_rate"] == 0.6
    assert out["disparate_impact"] == pytest.approx(0.6 / 0.9)
    assert out["passes_80_percent_rule"] is False
    assert out["equal_opportunity_diff"] == pytest.approx(0.3)


def test_small_groups_are_skipped_and_nan_auc_when_single_class():
    df = pd.DataFrame({"default": [0] * 40 + [1] * 5})
    groups = pd.Series(["big"] * 40 + ["tiny"] * 5)
    out = audit_attribute(df, groups, np.full(45, 0.1), 0.5)
    assert list(out["groups"]) == ["big"] and np.isnan(out["groups"]["big"]["auc"])


def test_age_band_edges():
    bands = age_band(pd.Series([20, 24, 25, 39, 40, 59, 60, 80])).astype(str).tolist()
    assert bands == ["<25", "<25", "25-40", "25-40", "40-60", "40-60", ">60", ">60"]


def test_audit_frame_has_both_attributes(raw_df):
    out = audit_frame(raw_df, np.random.default_rng(0).random(len(raw_df)), 0.5)
    assert set(out) == {"sex", "age_band"}
    assert set(out["sex"]["groups"]) == {"masculino", "feminino"}
