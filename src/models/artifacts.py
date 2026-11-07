"""Persistência PORTÁVEL de modelos: JSON nativo, sem pickle entre ambientes (lição 13.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.constants import MODEL_FEATURES

MODEL_FILE = "model.json"
META_FILE = "metadata.json"


@dataclass
class LogRegScorer:
    """Regressão logística reconstruída a partir de parâmetros em JSON."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Probabilidades [P(0), P(1)] como no scikit-learn."""
        z = ((x[MODEL_FEATURES].to_numpy(float) - self.mean) / self.scale) @ self.coef
        p = 1.0 / (1.0 + np.exp(-(z + self.intercept)))
        return np.column_stack([1 - p, p])


def _logreg_params(pipeline: Any) -> dict:
    scaler, clf = pipeline.named_steps["scaler"], pipeline.named_steps["clf"]
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def save_model(model: Any, model_type: str, directory: Path, metadata: dict) -> Path:
    """Salva o modelo (JSON) e o metadata.json em ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    if model_type == "xgboost":
        model.save_model(path)
    else:
        path.write_text(json.dumps(_logreg_params(model)), encoding="utf-8")
    meta = {**metadata, "model_type": model_type, "features": MODEL_FEATURES}
    (directory / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def read_metadata(directory: Path) -> dict:
    """Lê o metadata.json do diretório do modelo."""
    return json.loads((directory / META_FILE).read_text(encoding="utf-8"))


def load_model(directory: Path) -> Any:
    """Carrega o modelo (XGBClassifier ou LogRegScorer) a partir do diretório."""
    meta = read_metadata(directory)
    path = directory / MODEL_FILE
    if meta["model_type"] == "xgboost":
        model = XGBClassifier()
        model.load_model(path)
        return model
    params = json.loads(path.read_text(encoding="utf-8"))
    return LogRegScorer(
        np.array(params["mean"]),
        np.array(params["scale"]),
        np.array(params["coef"]),
        params["intercept"],
    )
