"""Catálogo de regras do contrato (rígidas bloqueiam; suaves avisam) e classificação."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import BILL_COLS
from src.data.cleaning import undocumented_rate

S1_MAX_OUTSIDE = 0.05
S2_MAX_UNDOCUMENTED = 0.02
S3_MAX_MEAN_SHIFT = 0.30


@dataclass(frozen=True)
class Rule:
    """Regra do contrato com id, descrição, severidade e dono."""

    rule_id: str
    description: str
    severity: str  # "hard" | "soft"
    owner: str


CATALOG: dict[str, Rule] = {
    r.rule_id: r
    for r in [
        Rule("H1", "Schema: colunas obrigatórias, tipos corretos, sem colunas extras", "hard", "Engenharia de Dados"),
        Rule("H2", "Sem nulos em nenhuma feature nem no identificador", "hard", "Engenharia de Dados"),
        Rule("H3", "18 <= age <= 100", "hard", "Risco de Crédito"),
        Rule("H4", "0 < limit_bal <= 1.000.000", "hard", "Risco de Crédito"),
        Rule("H5", "Domínios: sex in {1,2}; education 0-6; marriage 0-3; pay_x em [-2, 8]", "hard", "Risco de Crédito"),
        Rule("H6", "Unicidade: client_hash sem duplicatas e sem linhas duplicadas exatas (SHA-256)", "hard", "Engenharia de Dados"),
        Rule("H7", "pay_amt_x >= 0", "hard", "Risco de Crédito"),
        Rule("H8", "Volume mínimo do lote (>= MIN_BATCH_ROWS)", "hard", "Engenharia de Dados"),
        Rule("S1", "bill_amt fora de [p0,5; p99,5] da referência em > 5% das linhas", "soft", "Ciência de Dados"),
        Rule("S2", "education/marriage não documentados em > 2% das linhas", "soft", "Ciência de Dados"),
        Rule("S3", "Média de limit_bal desvia > 30% da média de referência", "soft", "Risco de Crédito"),
    ]
}  # fmt: skip

SCHEMA_CHECKS = {"column_in_schema", "column_in_dataframe"}
UNIQUE_CHECKS = {"field_uniqueness", "no_exact_duplicates"}


def classify_failure(column: str | None, check: str) -> str:
    """Mapeia (coluna, check do Pandera) para o id da regra rígida."""
    if check in SCHEMA_CHECKS or check.startswith(("dtype", "coerce")):
        return "H1"
    if check == "not_nullable":
        return "H2"
    if check in UNIQUE_CHECKS:
        return "H6"
    return _classify_by_column(column or "")


def _classify_by_column(column: str) -> str:
    if column == "age":
        return "H3"
    if column == "limit_bal":
        return "H4"
    if column.startswith("pay_amt"):
        return "H7"
    return "H5"


@dataclass(frozen=True)
class ReferenceStats:
    """Estatísticas do reference para as regras suaves (calculadas UMA vez)."""

    bill_low: dict[str, float]
    bill_high: dict[str, float]
    limit_mean: float


def compute_reference_stats(ref: pd.DataFrame) -> ReferenceStats:
    """Percentis 0,5 / 99,5 das faturas e média de limit_bal."""
    return ReferenceStats(
        bill_low={c: float(np.quantile(ref[c], 0.005)) for c in BILL_COLS},
        bill_high={c: float(np.quantile(ref[c], 0.995)) for c in BILL_COLS},
        limit_mean=float(ref["limit_bal"].mean()),
    )


def soft_warnings(df: pd.DataFrame, stats: ReferenceStats | None) -> list[dict]:
    """Avalia S1-S3; retorna avisos (não bloqueiam a ingestão)."""
    warnings: list[dict] = []
    if len(df) == 0:
        return warnings
    rate = undocumented_rate(df) if {"education", "marriage"} <= set(df.columns) else 0.0
    if rate > S2_MAX_UNDOCUMENTED:
        warnings.append({"rule_id": "S2", "value": round(rate, 4)})
    if stats is not None:
        warnings += _s1(df, stats) + _s3(df, stats)
    return warnings


def _s1(df: pd.DataFrame, stats: ReferenceStats) -> list[dict]:
    out = []
    for col in (c for c in BILL_COLS if c in df.columns):
        outside = ((df[col] < stats.bill_low[col]) | (df[col] > stats.bill_high[col])).mean()
        if outside > S1_MAX_OUTSIDE:
            out.append({"rule_id": "S1", "column": col, "value": round(float(outside), 4)})
    return out


def _s3(df: pd.DataFrame, stats: ReferenceStats) -> list[dict]:
    if "limit_bal" not in df.columns:
        return []
    shift = abs(float(df["limit_bal"].mean()) - stats.limit_mean) / stats.limit_mean
    if shift > S3_MAX_MEAN_SHIFT:
        return [{"rule_id": "S3", "value": round(shift, 4)}]
    return []
