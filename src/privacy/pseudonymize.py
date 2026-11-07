"""Pseudonimização do identificador do cliente (HMAC-SHA256 com chave secreta).

Pseudonimização NÃO é anonimização (LGPD art. 5º, III e art. 12): o dado
pseudonimizado continua sendo dado pessoal, pois a chave permite reidentificar.
O ID cru existe apenas na etapa de ingestão e nunca é persistido adiante.
"""

from __future__ import annotations

import hashlib
import hmac

import pandas as pd

from src.config import Settings, get_settings
from src.constants import HASH_COL, ID_COL

HASH_PREFIX_LEN = 12


def pseudonymize_value(value: object, key: bytes) -> str:
    """Calcula o HMAC-SHA256 (hex) de um identificador."""
    return hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def pseudonymize_ids(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Substitui ``client_id`` por ``client_hash``; nulos permanecem nulos (o contrato os barra)."""
    if ID_COL not in df.columns:
        return df
    key = (settings or get_settings()).hmac_key()
    hashed = df[ID_COL].map(lambda v: None if pd.isna(v) else pseudonymize_value(v, key))
    out = df.drop(columns=[ID_COL])
    out.insert(0, HASH_COL, hashed.astype("object"))
    return out


def short_hash(value: object) -> str:
    """Prefixo curto de um hash, para relatórios e logs."""
    return str(value)[:HASH_PREFIX_LEN]
