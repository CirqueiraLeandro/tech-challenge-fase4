"""Exceções do projeto."""

from __future__ import annotations


class ContractViolationError(Exception):
    """Lote reprovado em regra RÍGIDA do contrato de dados (exit code 2)."""

    def __init__(self, message: str, quarantine_path: str | None = None) -> None:
        super().__init__(message)
        self.quarantine_path = quarantine_path


class GateFailedError(Exception):
    """Modelo candidato reprovado no gate de promoção (exit code 3)."""
