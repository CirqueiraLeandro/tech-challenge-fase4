"""Reason codes por cliente (LGPD art. 20) via contribuições SHAP nativas do XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.constants import MODEL_FEATURES

TOP_N = 3


def contributions(model: xgb.XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    """Matriz (n, n_features + 1): contribuições por feature + bias (última coluna)."""
    dmatrix = xgb.DMatrix(x[MODEL_FEATURES])
    return model.get_booster().predict(dmatrix, pred_contribs=True)


def margin(model: xgb.XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    """Margem (log-odds) prevista, para conferir soma das contribuições + bias."""
    return model.get_booster().predict(xgb.DMatrix(x[MODEL_FEATURES]), output_margin=True)


def reason_codes(model: xgb.XGBClassifier, x: pd.DataFrame, top: int = TOP_N) -> list[list[str]]:
    """Top ``top`` features que mais EMPURRAM o cliente para inadimplência (contrib > 0)."""
    contribs = contributions(model, x)[:, :-1]
    order = np.argsort(-contribs, axis=1)[:, :top]
    names = np.array(MODEL_FEATURES)
    return [
        [str(f) for f, c in zip(names[idx], contribs[i, idx], strict=True) if c > 0]
        for i, idx in enumerate(order)
    ]
