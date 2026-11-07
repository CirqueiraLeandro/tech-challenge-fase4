"""Validador do contrato: lazy, atômico (all-or-nothing), com quarentena e relatório."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd
from pandera.errors import SchemaErrors

from src.config import Settings, get_settings
from src.constants import HASH_COL
from src.contracts.rules import (
    CATALOG,
    SCHEMA_CHECKS,
    ReferenceStats,
    classify_failure,
    soft_warnings,
)
from src.contracts.schemas import CONTRACT_VERSION, RawCreditSchema
from src.exceptions import ContractViolationError
from src.logger import get_logger, run_id_var
from src.monitoring.tracking import record_validation
from src.privacy.pseudonymize import pseudonymize_ids, short_hash

logger = get_logger(__name__)
MONETARY_PREFIXES = ("limit_bal", "bill_amt", "pay_amt")
MAX_EXAMPLES = 5


@dataclass
class Violation:
    """Violação agregada por (regra, coluna, check). Sem PII crua: só client_hash curto."""

    rule_id: str
    column: str | None
    check: str
    n_cases: int
    examples: list[dict] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Resultado da validação de um lote."""

    passed: bool
    n_rows: int
    n_failed_rows: int
    violations: list[Violation]
    soft_warnings: list[dict]
    contract_version: str = CONTRACT_VERSION

    @property
    def violated_rules(self) -> list[str]:
        """Ids distintos das regras rígidas violadas."""
        return sorted({v.rule_id for v in self.violations})

    def to_dict(self) -> dict:
        """Serializa para JSON."""
        return asdict(self) | {"violated_rules": self.violated_rules}


def widen_monetary(df: pd.DataFrame) -> pd.DataFrame:
    """Alarga int -> float nas colunas monetárias (conversão sem perda; texto continua falhando)."""
    out = df.copy()
    for col in out.columns:
        if col.startswith(MONETARY_PREFIXES) and pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype(float)
    return out


def _example(df: pd.DataFrame, index: object, value: object) -> dict:
    client = None
    if HASH_COL in df.columns and index in df.index and not pd.isna(index):
        client = short_hash(df.loc[index, HASH_COL])
    shown = None if pd.isna(value) else str(value)
    position = None if pd.isna(index) else int(cast(int, index))
    return {"index": position, "value": shown, "client_hash": client}


DATAFRAME_LEVEL_CHECKS = {"no_exact_duplicates"}


def _collapse_dataframe_checks(cases: pd.DataFrame) -> pd.DataFrame:
    """Checks de DataFrame vêm uma vez por coluna; mantém um registro por linha afetada."""
    is_df = cases["check"].isin(DATAFRAME_LEVEL_CHECKS)
    collapsed = cases[is_df].drop_duplicates(subset=["check", "index"]).assign(column=None)
    rest = cases[~is_df].copy()
    is_schema = rest["check"].isin(SCHEMA_CHECKS)
    rest.loc[is_schema, "column"] = rest.loc[is_schema, "failure_case"]
    return pd.concat([rest, collapsed], ignore_index=True)


def _failure_violations(df: pd.DataFrame, cases: pd.DataFrame) -> list[Violation]:
    cases = _collapse_dataframe_checks(cases)
    rules = [classify_failure(c, k) for c, k in zip(cases["column"], cases["check"], strict=True)]
    cases = cases.assign(rule_id=rules)
    result = []
    for (rule, col, check), grp in cases.groupby(["rule_id", "column", "check"], dropna=False):
        head = grp.head(MAX_EXAMPLES)
        examples = [
            _example(df, i, v) for i, v in zip(head["index"], head["failure_case"], strict=True)
        ]
        column = None if pd.isna(col) else str(col)
        result.append(Violation(rule, column, str(check), len(grp), examples))
    return result


def _min_rows_violation(df: pd.DataFrame, min_rows: int) -> list[Violation]:
    if len(df) >= min_rows:
        return []
    return [
        Violation(
            "H8",
            None,
            f"min_rows({min_rows})",
            1,
            [{"index": None, "value": str(len(df)), "client_hash": None}],
        )
    ]


def validate_batch(
    df: pd.DataFrame,
    stats: ReferenceStats | None = None,
    min_rows: int | None = None,
    settings: Settings | None = None,
) -> ValidationResult:
    """Valida o lote (lazy=True) e devolve TODAS as violações, sem lançar exceção."""
    min_rows = min_rows if min_rows is not None else (settings or get_settings()).min_batch_rows
    df = widen_monetary(df)
    violations = _min_rows_violation(df, min_rows)
    n_failed = 0
    try:
        RawCreditSchema.validate(df, lazy=True)
    except SchemaErrors as exc:
        violations += _failure_violations(df, exc.failure_cases)
        n_failed = int(exc.failure_cases["index"].dropna().nunique())
    return ValidationResult(not violations, len(df), n_failed, violations, soft_warnings(df, stats))


def write_report(result: ValidationResult, name: str, out_dir: Path) -> Path:
    """Grava violações em JSON e HTML legível; retorna o caminho do JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}_violations.json"
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{name}_violations.html").write_text(render_html(result, name), encoding="utf-8")
    return json_path


_HTML_STYLE = (
    "body{font-family:sans-serif;margin:2rem}td,th{border:1px solid #ccc;padding:4px 8px}"
    "table{border-collapse:collapse}.bad{color:#b00020}.ok{color:#1b7f3b}"
)


def _html_rows(result: ValidationResult) -> str:
    return "".join(
        f"<tr><td>{v.rule_id}</td><td>{html.escape(CATALOG[v.rule_id].description)}</td>"
        f"<td>{html.escape(str(v.column))}</td><td>{html.escape(v.check)}</td>"
        f"<td>{v.n_cases}</td></tr>"
        for v in result.violations
    )


def render_html(result: ValidationResult, name: str) -> str:
    """HTML estático com a tabela de violações por regra."""
    title = f"Contrato de dados v{result.contract_version} - {html.escape(name)}"
    css = "ok" if result.passed else "bad"
    status = "APROVADO" if result.passed else "REPROVADO"
    summary = (
        f"Linhas: {result.n_rows} | violações: {len(result.violations)} | "
        f"regras: {', '.join(result.violated_rules) or '-'} | "
        f"avisos suaves: {len(result.soft_warnings)}"
    )
    head = "<tr><th>Regra</th><th>Descrição</th><th>Coluna</th><th>Check</th><th>Casos</th></tr>"
    return (
        f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{title}</title>"
        f"<style>{_HTML_STYLE}</style></head><body><h1>{title}</h1>"
        f"<h2 class='{css}'>{status}</h2><p>{summary}</p>"
        f"<table>{head}{_html_rows(result)}</table></body></html>"
    )


def quarantine(
    df: pd.DataFrame, result: ValidationResult, settings: Settings | None = None
) -> Path:
    """Move o lote rejeitado para data/quarantine/<run_id>/ com violations.json."""
    settings = settings or get_settings()
    target = settings.path("quarantine") / run_id_var.get()
    target.mkdir(parents=True, exist_ok=True)
    df.to_csv(target / "batch.csv", index=False, encoding="utf-8")
    (target / "violations.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def ingest_and_validate(
    df: pd.DataFrame,
    name: str,
    stats: ReferenceStats | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, ValidationResult]:
    """Pseudonimiza o ID, valida; em falha rígida quarentena e lança ContractViolationError."""
    settings = settings or get_settings()
    df = pseudonymize_ids(df, settings)
    result = validate_batch(df, stats, settings=settings)
    report = write_report(result, name, settings.path("reports") / "contracts")
    record_validation(name, result, report)
    logger.info(
        "contrato_validado",
        extra={
            "batch": name,
            "rows": result.n_rows,
            "passed": result.passed,
            "rules": result.violated_rules,
            "soft_warnings": len(result.soft_warnings),
        },
    )
    if not result.passed:
        path = quarantine(df, result, settings)
        raise ContractViolationError(f"lote {name} reprovado: {result.violated_rules}", str(path))
    return df, result
