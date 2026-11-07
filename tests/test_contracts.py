"""Testes do contrato Pandera: cada regra rígida isolada, lazy, soft, quarentena e CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.contracts.badbatch import make_bad_batch
from src.contracts.rules import CATALOG, classify_failure, compute_reference_stats
from src.contracts.schemas import FeaturesSchema, row_hashes
from src.contracts.validator import (
    ValidationResult,
    ingest_and_validate,
    validate_batch,
    write_report,
)
from src.data.cleaning import clean_categories
from src.data.features import add_features
from src.exceptions import ContractViolationError
from src.pipeline import cli

pytestmark = pytest.mark.schema


def _mutate(df, rule):
    df = df.copy()
    if rule == "H1":
        df = df.rename(columns={"limit_bal": "credit_limit"})
    elif rule == "H2":
        df.loc[0, "limit_bal"] = float("nan")
    elif rule == "H3":
        df.loc[0, "age"] = 17
    elif rule == "H4":
        df.loc[0, "limit_bal"] = 2_000_000.0
    elif rule == "H5":
        df.loc[0, "pay_0"] = 9
    elif rule == "H6":
        df.loc[1, "client_hash"] = df.loc[0, "client_hash"]
    elif rule == "H7":
        df.loc[0, "pay_amt3"] = -1.0
    return df


@pytest.mark.parametrize("rule", ["H1", "H2", "H3", "H4", "H5", "H6", "H7"])
def test_each_hard_rule_fails_in_isolation(ingested_df, rule):
    result = validate_batch(_mutate(ingested_df, rule))
    assert not result.passed
    assert result.violated_rules == [rule]


def test_h8_min_rows(ingested_df):
    result = validate_batch(ingested_df.head(50), min_rows=100)
    assert result.violated_rules == ["H8"]


def test_exact_duplicate_row_is_h6(ingested_df):
    dup = pd.concat([ingested_df, ingested_df.iloc[[3]]], ignore_index=True)
    assert "H6" in validate_batch(dup).violated_rules


def test_clean_batch_passes(ingested_df):
    result = validate_batch(ingested_df)
    assert result.passed and result.n_rows == len(ingested_df) and result.violations == []


def test_integer_money_is_widened_not_rejected(ingested_df):
    df = ingested_df.copy()
    df["limit_bal"] = df["limit_bal"].astype(int)
    assert validate_batch(df).passed


def test_string_in_numeric_column_fails_h1(ingested_df):
    df = ingested_df.copy()
    df["age"] = df["age"].astype(str)
    assert "H1" in validate_batch(df).violated_rules


def test_lazy_collects_all_violations(ingested_df):
    result = validate_batch(make_bad_batch(ingested_df, n=300))
    assert set(result.violated_rules) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}


def test_violation_examples_have_no_raw_id(ingested_df):
    result = validate_batch(make_bad_batch(ingested_df, n=300))
    assert "client_id" not in json.dumps(result.to_dict())
    assert all(len(e["client_hash"] or "") <= 12 for v in result.violations for e in v.examples)


def test_soft_rules_warn_but_do_not_block(ingested_df):
    stats = compute_reference_stats(ingested_df)
    df = ingested_df.copy()
    df["limit_bal"] = df["limit_bal"] * 0.5
    df.loc[df.index[:100], "education"] = 5
    result = validate_batch(df, stats)
    rules = {w["rule_id"] for w in result.soft_warnings}
    assert result.passed and {"S2", "S3"} <= rules


def test_soft_s1_outliers(ingested_df):
    stats = compute_reference_stats(ingested_df)
    df = ingested_df.copy()
    df["bill_amt1"] = df["bill_amt1"] * 10
    warnings = validate_batch(df, stats).soft_warnings
    assert any(w["rule_id"] == "S1" and w["column"] == "bill_amt1" for w in warnings)


def test_soft_rules_tolerate_missing_columns(ingested_df):
    stats = compute_reference_stats(ingested_df)
    df = ingested_df.drop(columns=["bill_amt6", "limit_bal"])
    assert not validate_batch(df, stats).passed


def test_ingest_quarantines_and_raises(raw_df, settings):
    bad = raw_df.copy()
    bad.loc[0, "age"] = 5
    with pytest.raises(ContractViolationError) as exc:
        ingest_and_validate(bad, "lote", settings=settings)
    assert exc.value.quarantine_path
    assert list(settings.path("quarantine").glob("*/violations.json"))
    assert (settings.path("reports") / "contracts" / "lote_violations.html").exists()


def test_ingest_pseudonymizes_valid_batch(raw_df, settings):
    df, result = ingest_and_validate(raw_df, "ok", settings=settings)
    assert result.passed and "client_id" not in df.columns and "client_hash" in df.columns


def test_report_files(ingested_df, tmp_path):
    result = validate_batch(make_bad_batch(ingested_df, n=300))
    path = write_report(result, "x", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["passed"] is False and (tmp_path / "x_violations.html").exists()


def test_catalog_has_8_hard_and_3_soft():
    hard = [r for r in CATALOG.values() if r.severity == "hard"]
    soft = [r for r in CATALOG.values() if r.severity == "soft"]
    assert len(hard) == 8 and len(soft) == 3


@pytest.mark.parametrize(
    ("column", "check", "rule"),
    [
        ("x", "column_in_schema", "H1"),
        ("x", "dtype('int64')", "H1"),
        ("x", "not_nullable", "H2"),
        ("age", "greater_than(1)", "H3"),
        ("limit_bal", "greater_than(0)", "H4"),
        ("sex", "isin([1, 2])", "H5"),
        ("client_hash", "field_uniqueness", "H6"),
        ("pay_amt2", "greater_than_or_equal_to(0)", "H7"),
    ],
)
def test_classify_failure(column, check, rule):
    assert classify_failure(column, check) == rule


def test_row_hash_detects_equal_rows(ingested_df):
    df = pd.concat([ingested_df.head(3), ingested_df.head(1)], ignore_index=True)
    assert row_hashes(df).duplicated().tolist() == [False, False, False, True]


def test_features_schema_accepts_clean_features(raw_df):
    FeaturesSchema.validate(add_features(clean_categories(raw_df)))


def test_cli_validate_exit_codes(raw_df, ingested_df, settings, tmp_path):
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    raw_df.to_csv(good, index=False)
    make_bad_batch(ingested_df, n=300).to_csv(bad, index=False)
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(bad)]) == 2


def test_validation_result_dataclass_roundtrip():
    result = ValidationResult(True, 1, 0, [], [])
    assert result.to_dict()["violated_rules"] == []
