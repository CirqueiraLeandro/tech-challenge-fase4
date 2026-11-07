"""Testes de privacidade: HMAC, minimização, ID cru, retenção, limpeza e features."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from src.constants import (
    EXCLUDED_FROM_MODEL,
    HASH_COL,
    ID_COL,
    MODEL_FEATURES,
    PII_COLUMNS,
    PROTECTED_ATTRIBUTES,
)
from src.data.cleaning import clean_categories, undocumented_rate
from src.data.features import add_features
from src.privacy.minimization import audit_view, minimize, model_matrix
from src.privacy.pseudonymize import pseudonymize_ids, pseudonymize_value, short_hash
from src.privacy.retention import RetentionRule, apply_retention, build_rules, find_expired

pytestmark = pytest.mark.unit


def test_hmac_is_deterministic_and_key_dependent():
    assert pseudonymize_value(1, b"k1") == pseudonymize_value(1, b"k1")
    assert pseudonymize_value(1, b"k1") != pseudonymize_value(1, b"k2")
    assert pseudonymize_value(1, b"k1") != pseudonymize_value(2, b"k1")


def test_raw_id_removed(raw_df, settings):
    out = pseudonymize_ids(raw_df, settings)
    assert ID_COL not in out.columns and out[HASH_COL].is_unique


def test_null_id_stays_null(raw_df, settings):
    df = raw_df.copy().astype({ID_COL: float})
    df.loc[0, ID_COL] = float("nan")
    assert pseudonymize_ids(df, settings)[HASH_COL].isna().sum() == 1


def test_pseudonymize_noop_without_id(ingested_df, settings):
    assert pseudonymize_ids(ingested_df, settings) is ingested_df


def test_model_features_exclude_pii_and_protected():
    assert set(MODEL_FEATURES).isdisjoint({ID_COL, HASH_COL, *PROTECTED_ATTRIBUTES})
    assert set(PII_COLUMNS) <= set(EXCLUDED_FROM_MODEL)


def test_minimize_and_audit_view(ingested_df):
    assert set(minimize(ingested_df).columns).isdisjoint(EXCLUDED_FROM_MODEL)
    assert set(audit_view(ingested_df).columns) == {HASH_COL, "sex", "marriage", "age"}
    assert list(model_matrix(add_features(ingested_df)).columns) == MODEL_FEATURES


def test_short_hash():
    assert short_hash("a" * 64) == "a" * 12


def test_cleaning_remaps_undocumented(raw_df):
    df = raw_df.copy()
    df.loc[0, "education"] = 5
    df.loc[1, "education"] = 0
    df.loc[2, "marriage"] = 0
    assert undocumented_rate(df) == pytest.approx(3 / len(df))
    out = clean_categories(df)
    assert set(out["education"]) <= {1, 2, 3, 4} and set(out["marriage"]) <= {1, 2, 3}
    assert df.loc[0, "education"] == 5


def test_features_values():
    df = pd.DataFrame(
        {
            "limit_bal": [1000.0, 1000.0],
            "bill_amt1": [500.0, 0.0],
            "pay_amt1": [250.0, 10.0],
            "pay_0": [2, -1],
            "pay_2": [0, -2],
            "pay_3": [1, 0],
            "pay_4": [0, 0],
            "pay_5": [0, 0],
            "pay_6": [-1, 0],
        }
    )
    out = add_features(df)
    assert out["utilization"].tolist() == [0.5, 0.0]
    assert out["payment_ratio"].tolist() == [0.5, 5.0]
    assert out["max_delay"].tolist() == [2, 0]
    assert out["months_delayed"].tolist() == [2, 0]


def _old_file(path, days):
    path.write_text("x", encoding="utf-8")
    ts = time.time() - days * 86_400
    os.utime(path, (ts, ts))


def test_retention_dry_run_does_not_delete(tmp_path):
    old = tmp_path / "old.jsonl"
    _old_file(old, 100)
    rule = RetentionRule("logs", tmp_path, 90, "*.jsonl")
    assert apply_retention([rule], dry_run=True) == {"logs": 1}
    assert old.exists()


def test_retention_execute_deletes_only_expired(tmp_path):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    _old_file(old, 100)
    _old_file(new, 1)
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    rule = RetentionRule("logs", tmp_path, 90)
    assert apply_retention([rule], dry_run=False) == {"logs": 1}
    assert not old.exists() and new.exists() and (tmp_path / ".gitkeep").exists()


def test_retention_removes_directories(tmp_path):
    batch = tmp_path / "run1"
    batch.mkdir()
    (batch / "f.csv").write_text("x", encoding="utf-8")
    ts = time.time() - 40 * 86_400
    os.utime(batch, (ts, ts))
    apply_retention([RetentionRule("quarantine", tmp_path, 30)], dry_run=False)
    assert not batch.exists()


def test_retention_missing_dir_and_rules(tmp_path, settings):
    assert find_expired(RetentionRule("x", tmp_path / "nao_existe", 1)) == []
    assert {r.name for r in build_rules(settings)} == {"raw", "quarantine", "logs"}


def test_apply_retention_script_default_is_dry_run(settings):
    from scripts.apply_retention import main

    assert main([]) == 0
