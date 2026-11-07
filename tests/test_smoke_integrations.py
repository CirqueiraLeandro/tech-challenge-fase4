"""Testes de fumaça das integrações de maior risco (lição 13.5).

Fixam as APIs REAIS das versões do poetry.lock: Evidently 0.7 (``Report([...]).run``),
Pandera (``pandera.pandas``) e MLflow 3 (sqlite + aliases).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.errors import SchemaErrors
from pandera.typing import Series

pytestmark = pytest.mark.smoke


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"a": rng.normal(0.8, 1, 2000), "b": rng.normal(0, 1, 2000)})
    return ref, cur


def test_evidently_new_api_generates_html_and_psi(tmp_path):
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref, cur = _frames()
    snapshot = Report([DataDriftPreset(method="psi", threshold=0.2)]).run(
        current_data=cur, reference_data=ref
    )
    out = tmp_path / "drift.html"
    snapshot.save_html(str(out))
    assert out.stat().st_size > 10_000
    values = {m["config"].get("column"): m["value"] for m in snapshot.dict()["metrics"]}
    assert values["a"] > 0.2 > values["b"]  # método PSI explícito, não o default


def test_evidently_explicit_ks_and_chisquare():
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref, cur = _frames()
    preset = DataDriftPreset(num_method="ks", cat_method="chisquare", num_threshold=0.05)
    metrics = Report([preset]).run(current_data=cur, reference_data=ref).dict()["metrics"]
    names = " ".join(m["metric_name"] for m in metrics)
    assert "method=ks" in names


def test_pandera_pandas_lazy_collects_all_errors():
    class Schema(pa.DataFrameModel):
        age: Series[int] = pa.Field(ge=18, le=100)
        limit_bal: Series[float] = pa.Field(gt=0)

        class Config:
            strict = True

    df = pd.DataFrame({"age": [15, 30], "limit_bal": [-1.0, 5.0], "extra": [1, 2]})
    with pytest.raises(SchemaErrors) as exc:
        Schema.validate(df, lazy=True)
    cases = exc.value.failure_cases
    assert {"column", "check", "failure_case", "index"} <= set(cases.columns)
    assert len(cases) >= 3  # age, limit_bal e coluna inesperada


def test_mlflow_sqlite_registry_alias_and_xgboost_flavor(tmp_path):
    import mlflow
    import xgboost as xgb
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("smoke")
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)), columns=list("abc"))
    y = (x["a"] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5).fit(x, y)
    with mlflow.start_run():
        mlflow.log_metric("auc", 0.9, step=1)
        info = mlflow.xgboost.log_model(
            model.get_booster(),
            name="model",
            signature=infer_signature(x, model.predict_proba(x)[:, 1]),
            input_example=x.head(2),
            registered_model_name="smoke-model",
        )
    client = MlflowClient()
    client.set_registered_model_alias("smoke-model", "champion", "1")
    assert int(client.get_model_version_by_alias("smoke-model", "champion").version) == 1
    assert info.model_uri
