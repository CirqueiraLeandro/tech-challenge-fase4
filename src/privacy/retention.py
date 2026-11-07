"""Política de retenção executável (dry-run por padrão: privacy by default)."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger(__name__)
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class RetentionRule:
    """Regra: eliminar itens de ``path`` mais antigos que ``days``."""

    name: str
    path: Path
    days: int
    pattern: str = "*"


def build_rules(settings: Settings | None = None) -> list[RetentionRule]:
    """Monta as regras a partir dos prazos do .env (RETENTION_*_DAYS)."""
    s = settings or get_settings()
    return [
        RetentionRule("raw", s.path("raw"), s.retention_raw_days, "*.csv"),
        RetentionRule("quarantine", s.path("quarantine"), s.retention_quarantine_days),
        RetentionRule("logs", s.path("logs"), s.retention_logs_days, "*.jsonl*"),
    ]


def _is_expired(item: Path, days: int, now: float) -> bool:
    return (now - item.stat().st_mtime) > days * SECONDS_PER_DAY


def find_expired(rule: RetentionRule, now: float | None = None) -> list[Path]:
    """Lista os itens vencidos (ignora arquivos .gitkeep)."""
    now = now if now is not None else time.time()
    if not rule.path.exists():
        return []
    return [
        item
        for item in rule.path.glob(rule.pattern)
        if item.name != ".gitkeep" and _is_expired(item, rule.days, now)
    ]


def _remove(item: Path) -> None:
    shutil.rmtree(item) if item.is_dir() else item.unlink()


def apply_retention(rules: list[RetentionRule], dry_run: bool = True) -> dict[str, int]:
    """Aplica as regras; com ``dry_run`` apenas conta. Registra sem dados pessoais."""
    summary: dict[str, int] = {}
    for rule in rules:
        expired = find_expired(rule)
        if not dry_run:
            for item in expired:
                _remove(item)
        summary[rule.name] = len(expired)
        logger.info(
            "retencao_aplicada",
            extra={"rule": rule.name, "items": len(expired), "dry_run": dry_run},
        )
    return summary
