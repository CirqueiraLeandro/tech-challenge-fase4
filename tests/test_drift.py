"""Testes de simulação SCM, testes estatísticos, detector, performance e política."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DriftScenario, MonitoringConfig, load_drift_scenario
from src.constants import BILL_COLS, PAY_AMT_COLS
from src.contracts.validator import validate_batch
from src.data.synthetic import make_synthetic_credit
from src.drift import policy
from src.drift.detector import (
    DriftDetector,
    cross_check,
    drift_share,
    parse_evidently,
    severity_from_psi,
)
from src.drift.performance import window_performance
from src.drift.simulation import (
    MECHANISM_COLUMNS,
    generate_window,
    load_window,
    save_windows,
    simulate_windows,
    split_chunks,
)
from src.drift.stattests import (
    Bins,
    ChiSquareTest,
    JensenShannonTest,
    KSTest,
    PSITest,
    bonferroni_alpha,
    create_stattest,
)
from src.models.factory import create_model
from src.models.train import build_xy
from src.privacy.pseudonymize import pseudonymize_ids

pytestmark = pytest.mark.unit
CFG = MonitoringConfig()


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    raw = make_synthetic_credit(6000, seed=3)
    return pseudonymize_ids(raw)


@pytest.fixture(scope="module")
def scenario() -> DriftScenario:
    sc = load_drift_scenario()
    return sc.model_copy(update={"window_size": 600})


def _only(scenario: DriftScenario, mechanism: str) -> frozenset[str]:
    return frozenset(m for m in scenario.mechanisms if m != mechanism)


# ---------------------------------------------------------------- stattests
def test_psi_same_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(size=5000))
    assert PSITest(0.1).compute(a, b).statistic < 0.02


def test_psi_known_shift_exceeds_threshold():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(1.0, 1, size=5000))
    result = PSITest(0.2).compute(a, b)
    assert result.statistic > 0.2 and result.drifted


def test_psi_discrete_and_out_of_domain():
    ref = pd.Series([1, 1, 2, 2, 3] * 100)
    cur = pd.Series([1, 2, 3, 9] * 100)  # 9 nunca visto no reference
    assert Bins(ref).discrete and PSITest(0.1).compute(ref, cur).drifted


def test_ks_chi2_js_behaviour():
    rng = np.random.default_rng(1)
    a, b = pd.Series(rng.normal(size=3000)), pd.Series(rng.normal(0.5, 1, size=3000))
    assert KSTest(0.05).compute(a, b).drifted and not KSTest(0.05).compute(a, a).drifted
    cat_a, cat_b = pd.Series([0, 1] * 500), pd.Series([0] * 900 + [1] * 100)
    assert ChiSquareTest(0.05).compute(cat_a, cat_b).drifted
    assert not ChiSquareTest(0.05).compute(cat_a, cat_a).drifted
    assert ChiSquareTest(0.05).compute(pd.Series([1] * 10), pd.Series([1] * 10)).p_value == 1.0
    js = JensenShannonTest(0.1).compute(a, b)
    assert 0 < js.statistic <= 1 and JensenShannonTest(0.1).compute(a, a).statistic < 0.05


def test_factory_stattest_and_bonferroni():
    assert create_stattest("psi", 0.1).name == "psi"
    with pytest.raises(ValueError, match="inválido"):
        create_stattest("wasserstein", 0.1)
    assert bonferroni_alpha(0.05, 10, True) == 0.005 and bonferroni_alpha(0.05, 10, False) == 0.05


def test_ks_significance_alone_is_not_drift_for_tiny_shift():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(size=200_000))
    cur = pd.Series(rng.normal(0.02, 1, size=200_000))
    assert KSTest(0.05).compute(ref, cur).drifted  # n grande acusa diferença irrelevante
    assert not PSITest(0.1).compute(ref, cur).drifted  # magnitude desmente


# --------------------------------------------------------------- simulação
def test_t0_changes_nothing(pool, scenario):
    chunk = split_chunks(pool, 6, 42)[0]
    res = generate_window(chunk, 0, 0.0, scenario, 42)
    original = chunk.set_index("client_hash").loc[res.data["client_hash"]].reset_index()
    pd.testing.assert_frame_equal(
        res.data[original.columns].astype(float, errors="ignore"),
        original.astype(float, errors="ignore"),
        check_dtype=False,
    )
    assert not res.latent_u.any()


@pytest.mark.parametrize("mechanism", list(MECHANISM_COLUMNS))
def test_each_mechanism_changes_only_its_columns(pool, scenario, mechanism):
    chunk = split_chunks(pool, 6, 42)[5]
    off = generate_window(chunk, 5, 1.0, scenario, 42, _only(scenario, "nenhum"))
    one = generate_window(chunk, 5, 1.0, scenario, 42, _only(scenario, mechanism))
    if mechanism == "perfil_jovem":
        assert (one.data["age"] < 30).mean() > (off.data["age"] < 30).mean()
        return
    cols = [c for c in off.data.columns if c != "client_hash"]
    changed = {c for c in cols if not np.allclose(off.data[c], one.data[c])}
    assert changed and changed <= set(MECHANISM_COLUMNS[mechanism])


def test_disabling_one_mechanism_keeps_others_identical(pool, scenario):
    chunk = split_chunks(pool, 6, 42)[5]
    a = generate_window(chunk, 5, 1.0, scenario, 42, frozenset({"desemprego"}))
    b = generate_window(chunk, 5, 1.0, scenario, 42, frozenset())
    assert np.allclose(a.data[BILL_COLS], b.data[BILL_COLS])
    assert np.allclose(a.data["limit_bal"], b.data["limit_bal"])
    assert a.latent_u.sum() == 0 and b.latent_u.sum() > 0


def test_inflation_and_unemployment_magnitudes(pool, scenario):
    chunk = split_chunks(pool, 6, 42)[5]
    base = generate_window(chunk, 5, 0.0, scenario, 42)
    full = generate_window(chunk, 5, 1.0, scenario, 42)
    ratio = full.data[BILL_COLS].mean().to_numpy() / base.data[BILL_COLS].mean().to_numpy()
    assert (ratio > 1.3).all() and full.data[PAY_AMT_COLS].mean().sum() > 0
    assert 0.12 < full.latent_u.mean() < 0.28
    assert full.data["default"].mean() > base.data["default"].mean()
    assert full.data["pay_0"].max() <= 8


def test_determinism_and_disjoint_windows(pool, scenario):
    w1 = simulate_windows(pool, scenario, 42)
    w2 = simulate_windows(pool, scenario, 42)
    for name in w1:
        pd.testing.assert_frame_equal(w1[name].data, w2[name].data)
    ids = [set(w.data["client_hash"]) for w in w1.values()]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(6) for j in range(i + 1, 6))
    assert all(len(w.data) == 600 for w in w1.values())


def test_windows_pass_hard_contract(pool, scenario):
    for result in simulate_windows(pool, scenario, 42).values():
        assert validate_batch(result.data, min_rows=100).passed


def test_save_and_load_windows(pool, scenario, settings):
    windows = simulate_windows(pool, scenario, 42)
    save_windows(windows, settings)
    pd.testing.assert_frame_equal(load_window("M3", settings), windows["M3"].data)


# ----------------------------------------------------------------- detector
@pytest.fixture(scope="module")
def detector(pool) -> DriftDetector:
    return DriftDetector(pool, CFG)


def test_control_window_has_no_false_positive(pool, scenario, detector):
    w = simulate_windows(pool, scenario, 42)
    assert not detector.analyze(w["M1"].data)["drifted"].any()


def test_final_window_flags_changed_features_not_negative_controls(pool, scenario, detector):
    table = detector.analyze(simulate_windows(pool, scenario, 42)["M6"].data).set_index("feature")
    assert table.loc[["limit_bal", "utilization"], "drifted"].all()
    assert not table.loc[["education", "pay_5", "pay_6"], "drifted"].any()
    assert drift_share(table.reset_index()) > 0.05


def test_evidently_html_and_cross_check(pool, scenario, detector, tmp_path):
    cur = simulate_windows(pool, scenario, 42)["M6"].data
    html = tmp_path / "m6.html"
    evid = detector.evidently(cur, html)
    assert html.stat().st_size > 10_000 and {"feature", "evidently_psi"} <= set(evid.columns)
    check = cross_check(detector.analyze(cur), evid)
    assert check["jaccard"] >= 0.5
    flagged = set(evid.loc[evid["evidently_drifted"], "feature"])
    assert {"limit_bal", "utilization"} <= flagged


def test_parse_evidently_ignores_other_metrics():
    payload = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 1}},
            {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "a"}, "value": 0.3},
        ]
    }
    out = parse_evidently(payload, 0.1)
    assert out["feature"].tolist() == ["a"] and out["evidently_drifted"].tolist() == [True]


def test_severity_and_empty_share():
    assert [severity_from_psi(v, CFG) for v in (0.05, 0.15, 0.3)] == [
        "stable",
        "moderate",
        "severe",
    ]
    assert drift_share(pd.DataFrame({"drifted": []})) == 0.0
    assert (
        cross_check(
            pd.DataFrame({"feature": [], "drifted": []}),
            pd.DataFrame({"feature": [], "evidently_drifted": []}),
        )["jaccard"]
        == 1.0
    )


# -------------------------------------------------------------- performance
def test_window_performance_fields(pool):
    x, y = build_xy(pool)
    model = create_model("logreg").fit(x, y)
    from src.drift.performance import score_frame

    bins = Bins(score_frame(model, pool))
    perf, score = window_performance(model, pool.head(800), 0.70, bins)
    assert 0.5 < perf.auc < 1 and len(score) == 800
    assert perf.auc_drop == pytest.approx(0.70 - perf.auc)
    assert perf.calibration_gap == pytest.approx(perf.predicted_rate - perf.observed_rate)
    assert set(perf.to_dict()) >= {"auc", "ks", "brier", "score_psi"}


# ------------------------------------------------------------------ política
@pytest.mark.parametrize(
    ("psi", "drop", "gap", "expected"),
    [
        (0.05, 0.01, 0.0, policy.OK),
        (0.15, 0.00, 0.0, policy.ATTENTION),
        (0.05, 0.03, 0.0, policy.ATTENTION),
        (0.50, 0.01, 0.0, policy.ATTENTION),
        (0.25, 0.06, 0.0, policy.CRITICAL),
        (0.05, 0.00, 0.08, policy.CRITICAL),
        (0.05, 0.00, -0.08, policy.CRITICAL),
    ],
)
def test_policy_table(psi, drop, gap, expected):
    decision = policy.decide(psi, drop, gap, CFG)
    assert decision.severity == expected and decision.action and decision.reasons
