"""Features derivadas determinísticas."""

from __future__ import annotations

import pandas as pd

from src.constants import PAY_COLS

CLIP_RATIO = 5.0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona utilization, payment_ratio, max_delay e months_delayed."""
    out = df.copy()
    out["utilization"] = (out["bill_amt1"] / out["limit_bal"]).clip(0, CLIP_RATIO)
    out["payment_ratio"] = (out["pay_amt1"] / out["bill_amt1"].clip(lower=1)).clip(0, CLIP_RATIO)
    out["max_delay"] = out[PAY_COLS].max(axis=1)
    out["months_delayed"] = (out[PAY_COLS] > 0).sum(axis=1)
    return out
