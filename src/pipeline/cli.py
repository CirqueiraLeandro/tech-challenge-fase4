"""CLI do pipeline: ``python -m src.pipeline.cli <comando>``.

Exit codes: 0 sucesso; 1 erro; 2 contrato violado; 3 gate reprovado.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from src.exceptions import ContractViolationError, GateFailedError
from src.logger import get_logger, new_run_id
from src.pipeline import stages

logger = get_logger(__name__)
EXIT_OK, EXIT_ERROR, EXIT_CONTRACT, EXIT_GATE = 0, 1, 2, 3

COMMANDS: dict[str, Callable[[argparse.Namespace], object]] = {}


def command(name: str) -> Callable:
    """Registra uma função como subcomando."""

    def decorator(func: Callable[[argparse.Namespace], object]) -> Callable:
        COMMANDS[name] = func
        return func

    return decorator


@command("prepare")
def _prepare(args: argparse.Namespace) -> object:
    return stages.prepare()


@command("train")
def _train(args: argparse.Namespace) -> object:
    from src.models import train

    return train.run()


@command("validate")
def _validate(args: argparse.Namespace) -> object:
    return stages.validate_file(Path(args.file))


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser com os subcomandos registrados."""
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name == "validate":
            sp.add_argument("file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executa o subcomando e traduz exceções em exit codes."""
    args = build_parser().parse_args(argv)
    run_id = new_run_id()
    try:
        COMMANDS[args.command](args)
    except ContractViolationError as exc:
        logger.error("contrato_violado", extra={"run_id_cmd": run_id, "detail": str(exc)})
        return EXIT_CONTRACT
    except GateFailedError as exc:
        logger.error("gate_reprovado", extra={"detail": str(exc)})
        return EXIT_GATE
    except Exception:
        logger.exception("erro_no_pipeline")
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
