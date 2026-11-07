"""Constantes do projeto: colunas, features, PII e domínios válidos (fonte única)."""

from __future__ import annotations

ID_COL = "client_id"
TARGET = "default"

PAY_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

RAW_COLUMNS = [
    ID_COL,
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    *PAY_COLS,
    *BILL_COLS,
    *PAY_AMT_COLS,
    TARGET,
]
RAW_FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c not in (ID_COL, TARGET)]

# Privacidade (LGPD): ver docs/GOVERNANCE.md
PII_COLUMNS = [ID_COL]
PROTECTED_ATTRIBUTES = ["sex", "marriage"]
EXCLUDED_FROM_MODEL = [ID_COL, *PROTECTED_ATTRIBUTES]

DERIVED_FEATURES = ["utilization", "payment_ratio", "max_delay", "months_delayed"]
MODEL_FEATURES = [
    c for c in RAW_FEATURE_COLUMNS if c not in PROTECTED_ATTRIBUTES
] + DERIVED_FEATURES

# Domínios válidos (contrato)
VALID_SEX = [1, 2]
VALID_EDUCATION = list(range(0, 7))
VALID_MARRIAGE = list(range(0, 4))
PAY_MIN, PAY_MAX = -2, 8
AGE_MIN, AGE_MAX = 18, 100
LIMIT_MAX = 1_000_000
MIN_BATCH_ROWS = 100

# Códigos não documentados no dataset original
UNDOCUMENTED_EDUCATION = [0, 5, 6]
UNDOCUMENTED_MARRIAGE = [0]

WINDOWS = ["M1", "M2", "M3", "M4", "M5", "M6"]

# Mapeamento dos nomes originais (UCI) -> padrão do projeto
UCI_RENAME = {
    "ID": ID_COL,
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    **{f"BILL_AMT{i}": f"bill_amt{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt{i}" for i in range(1, 7)},
    "default.payment.next.month": TARGET,
    "Y": TARGET,
}
# Fallback OpenML/ucimlrepo: X1..X23 + Y (ordem do dataset original)
X_ORDER = [
    "limit_bal", "sex", "education", "marriage", "age",
    *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS,
]  # fmt: skip
