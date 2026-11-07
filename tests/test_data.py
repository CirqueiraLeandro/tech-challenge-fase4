"""Testes de download/normalização, loader, split e gerador sintético."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.download_data import normalize_columns, verify
from scripts.validate_env import check_packages
from src.config import Settings
from src.constants import ID_COL, RAW_COLUMNS, TARGET, X_ORDER
from src.data.loader import load_pool, load_raw, load_reference
from src.data.splitter import save_split, split_reference_pool
from src.data.synthetic import make_synthetic_credit

pytestmark = pytest.mark.unit


def test_synthetic_has_real_schema():
    df = make_synthetic_credit(1500)
    assert list(df.columns) == RAW_COLUMNS
    assert 0.18 < df[TARGET].mean() < 0.26
    assert df[ID_COL].is_unique


def test_synthetic_is_deterministic():
    pd.testing.assert_frame_equal(make_synthetic_credit(200, 1), make_synthetic_credit(200, 1))


def test_normalize_x_columns():
    n = 5
    features = pd.DataFrame(np.ones((n, 23), dtype=int), columns=[f"X{i}" for i in range(1, 24)])
    out = normalize_columns(features, pd.Series([0, 1, 0, 1, 0]), pd.Series(range(10, 15)))
    assert list(out.columns) == RAW_COLUMNS
    assert out[ID_COL].tolist() == [10, 11, 12, 13, 14]
    assert X_ORDER[0] == "limit_bal"


def test_normalize_openml_lowercase_x_columns():
    """OpenML (data_id 42477) entrega x1..x23 em minúsculas: deve mapear pela ordem."""
    features = pd.DataFrame(np.arange(46).reshape(2, 23), columns=[f"x{i}" for i in range(1, 24)])
    out = normalize_columns(features, pd.Series([0, 1]), None)
    assert list(out.columns) == RAW_COLUMNS
    assert out["limit_bal"].tolist() == [0, 23] and out["pay_amt6"].tolist() == [22, 45]


def test_normalize_uci_names():
    df = make_synthetic_credit(10)
    upper = df.drop(columns=[ID_COL, TARGET]).rename(columns=str.upper)
    out = normalize_columns(upper, df[TARGET], None)
    assert list(out.columns) == RAW_COLUMNS


def test_verify_rejects_wrong_size():
    with pytest.raises(ValueError):
        verify(make_synthetic_credit(100))


def test_split_is_disjoint_stratified_and_deterministic():
    df = make_synthetic_credit(1500)
    ref, pool = split_reference_pool(df, seed=42)
    assert set(ref[ID_COL]).isdisjoint(pool[ID_COL])
    assert len(ref) == 900 and len(pool) == 600
    assert abs(ref[TARGET].mean() - pool[TARGET].mean()) < 0.01
    ref2, _ = split_reference_pool(df, seed=42)
    pd.testing.assert_frame_equal(ref, ref2)


def test_split_dedups_before_splitting():
    df = make_synthetic_credit(300)
    dup = pd.concat([df, df.iloc[:50]], ignore_index=True)
    ref, pool = split_reference_pool(dup)
    assert len(ref) + len(pool) == 300


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    settings = Settings(reference_dir=str(tmp_path / "ref"), production_dir=str(tmp_path / "prod"))
    ref, pool = split_reference_pool(make_synthetic_credit(400))
    save_split(ref, pool, settings)
    assert len(load_reference(settings)) == len(ref)
    assert len(load_pool(settings)) == len(pool)


def test_load_raw_missing_columns(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_raw(path)


def test_validate_env_packages_importable():
    assert check_packages() == []
