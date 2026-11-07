"""Logging estruturado JSON (console + logs/pipeline.jsonl) com ``run_id``.

Uso: ``logger = get_logger(__name__)`` e ``logger.info("evento", extra={"rows": 10})``.
Proibido ``print()`` em ``src/`` e proibido logar dados pessoais.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import get_settings

_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")
_log_file: Path | None = None


def new_run_id() -> str:
    """Gera e ativa um novo ``run_id`` para a execução corrente."""
    run_id = uuid.uuid4().hex[:12]
    run_id_var.set(run_id)
    return run_id


class JsonFormatter(logging.Formatter):
    """Formata cada evento como uma linha JSON (ISO 8601, UTF-8, sem escape ASCII)."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa o registro com os campos padrão e os ``extra`` informados."""
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "run_id": run_id_var.get(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _console_handler() -> logging.Handler:
    """Handler de console em UTF-8 (o console Windows costuma ser cp1252)."""
    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    return handler


def _file_handler(path: Path) -> logging.Handler:
    """Handler de arquivo JSONL com rotação (10MB x 5)."""
    handler = RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(JsonFormatter())
    return handler


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger JSON configurado (idempotente)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    settings = get_settings()
    logs = settings.path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    logger.setLevel(settings.log_level)
    logger.addHandler(_console_handler())
    logger.addHandler(_file_handler(logs / "pipeline.jsonl"))
    logger.propagate = False
    _remember_log_file(logs / "pipeline.jsonl")
    return logger


def _remember_log_file(path: Path) -> None:
    global _log_file
    _log_file = _log_file or path


def current_log_file() -> Path:
    """Arquivo JSONL em uso pelos handlers (definido na primeira configuração do logger)."""
    return _log_file or get_settings().path("logs") / "pipeline.jsonl"
