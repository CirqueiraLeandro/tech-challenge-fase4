"""Factory de modelos e de testes estatísticos (Strategy + Factory)."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_TYPES = ("xgboost", "logreg")


def create_model(model_type: str, seed: int = 42) -> Any:
    """Cria o estimador pelo nome ('xgboost' | 'logreg'); ValueError se inválido."""
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            random_state=seed,
            n_jobs=4,
        )
    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        )
    raise ValueError(f"model_type inválido: {model_type!r}. Opções: {MODEL_TYPES}")
