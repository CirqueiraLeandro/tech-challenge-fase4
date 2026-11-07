"""Contrato de dados em Pandera (schema-as-code).

Valida o lote JÁ INGERIDO (``client_id`` cru substituído por ``client_hash``).
Valores monetários são ``float``; os demais campos, ``int``. ``strict=True`` pega
schema drift (coluna renomeada/extra/ausente) e ``coerce=False`` evita mascarar tipos.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from src.constants import (
    AGE_MAX,
    AGE_MIN,
    LIMIT_MAX,
    PAY_MAX,
    PAY_MIN,
    VALID_EDUCATION,
    VALID_MARRIAGE,
    VALID_SEX,
)

CONTRACT_VERSION = "1.0.0"


def row_hashes(df: pd.DataFrame) -> pd.Series:
    """SHA-256 da linha serializada (detecção de duplicatas exatas)."""
    rows = df.astype(str).agg("|".join, axis=1)
    return rows.map(lambda r: hashlib.sha256(r.encode("utf-8")).hexdigest())


def _pay() -> Series[int]:
    return pa.Field(ge=PAY_MIN, le=PAY_MAX, nullable=False)


def _amount_nonneg() -> Series[float]:
    return pa.Field(ge=0, nullable=False)


def _amount() -> Series[float]:
    return pa.Field(nullable=False)


class RawCreditSchema(pa.DataFrameModel):
    """Contrato do lote de crédito (regras rígidas H1-H7)."""

    client_hash: Series[str] = pa.Field(nullable=False, unique=True)
    limit_bal: Series[float] = pa.Field(gt=0, le=LIMIT_MAX, nullable=False)
    sex: Series[int] = pa.Field(isin=VALID_SEX, nullable=False)
    education: Series[int] = pa.Field(isin=VALID_EDUCATION, nullable=False)
    marriage: Series[int] = pa.Field(isin=VALID_MARRIAGE, nullable=False)
    age: Series[int] = pa.Field(ge=AGE_MIN, le=AGE_MAX, nullable=False)
    pay_0: Series[int] = _pay()
    pay_2: Series[int] = _pay()
    pay_3: Series[int] = _pay()
    pay_4: Series[int] = _pay()
    pay_5: Series[int] = _pay()
    pay_6: Series[int] = _pay()
    bill_amt1: Series[float] = _amount()
    bill_amt2: Series[float] = _amount()
    bill_amt3: Series[float] = _amount()
    bill_amt4: Series[float] = _amount()
    bill_amt5: Series[float] = _amount()
    bill_amt6: Series[float] = _amount()
    pay_amt1: Series[float] = _amount_nonneg()
    pay_amt2: Series[float] = _amount_nonneg()
    pay_amt3: Series[float] = _amount_nonneg()
    pay_amt4: Series[float] = _amount_nonneg()
    pay_amt5: Series[float] = _amount_nonneg()
    pay_amt6: Series[float] = _amount_nonneg()
    default: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    @pa.dataframe_check(name="no_exact_duplicates")
    @classmethod
    def no_exact_duplicates(cls, df: pd.DataFrame) -> pd.Series:
        """Nenhuma linha duplicada exata (hash SHA-256 da linha)."""
        return ~row_hashes(df).duplicated(keep="first")

    class Config:
        strict = True
        coerce = False


class FeaturesSchema(pa.DataFrameModel):
    """Contrato das features derivadas após limpeza (entrada do modelo)."""

    education: Series[int] = pa.Field(ge=1, le=4)
    utilization: Series[float] = pa.Field(ge=0, le=5, nullable=False)
    payment_ratio: Series[float] = pa.Field(ge=0, le=5, nullable=False)
    max_delay: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    months_delayed: Series[int] = pa.Field(ge=0, le=6)

    class Config:
        strict = False
        coerce = False
