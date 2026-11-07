"""CLI do pipeline: ``python -m src.pipeline.cli <comando>``.

Exit codes: 0 sucesso; 1 erro; 2 contrato violado; 3 gate reprovado.
Cada comando é uma sequência de Tasks instrumentadas; ao fim de cada Task as métricas
são empurradas ao Pushgateway (grupo = nome da task) com fallback silencioso.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.exceptions import ContractViolationError, GateFailedError
from src.logger import get_logger, new_run_id
from src.monitoring.alerts import Alert, default_dispatcher
from src.monitoring.metrics import reset_metrics
from src.pipeline import stages
from src.pipeline.tasks import Task

logger = get_logger(__name__)
EXIT_OK, EXIT_ERROR, EXIT_CONTRACT, EXIT_GATE = 0, 1, 2, 3
DEFAULT_RETRAIN_WINDOWS = ["M4", "M5"]
COMMANDS: dict[str, Callable[[argparse.Namespace], Any]] = {}


def command(name: str) -> Callable:
    """Registra uma função como subcomando."""

    def decorator(func: Callable[[argparse.Namespace], Any]) -> Callable:
        """Adiciona ``func`` ao registro de comandos."""
        COMMANDS[name] = func
        return func

    return decorator


def run_task(task_cls: type[Task], *args: Any, **kwargs: Any) -> Any:
    """Executa uma Task com registry novo, alertas padrão e push final das métricas."""
    settings = get_settings()
    metrics = reset_metrics()
    task = task_cls(default_dispatcher(settings), metrics)
    try:
        return task.run(*args, **kwargs)
    except ContractViolationError as exc:
        task.dispatcher.notify(Alert("contract_violation", "critical", str(exc)))
        raise
    finally:
        metrics.push(settings.pushgateway_url, task.name)


@command("prepare")
def _prepare(args: argparse.Namespace) -> Any:
    return run_task(stages.PrepareTask)


@command("validate")
def _validate(args: argparse.Namespace) -> Any:
    return run_task(stages.ValidateTask, Path(args.file))


@command("train")
def _train(args: argparse.Namespace) -> Any:
    return run_task(stages.TrainTask)


@command("simulate")
def _simulate(args: argparse.Namespace) -> Any:
    return run_task(stages.SimulateTask)


@command("monitor")
def _monitor(args: argparse.Namespace) -> Any:
    return run_task(stages.MonitorTask)


@command("causal")
def _causal(args: argparse.Namespace) -> Any:
    return run_task(stages.CausalTask)


@command("fairness")
def _fairness(args: argparse.Namespace) -> Any:
    return run_task(stages.FairnessTask)


@command("report")
def _report(args: argparse.Namespace) -> Any:
    return run_task(stages.ReportTask)


@command("retrain")
def _retrain(args: argparse.Namespace) -> Any:
    return run_task(stages.RetrainTask, args.with_windows or DEFAULT_RETRAIN_WINDOWS)


@command("all")
def _all(args: argparse.Namespace) -> Any:
    for name in ("prepare", "train", "simulate", "monitor", "causal", "fairness", "report"):
        COMMANDS[name](args)


@command("demo")
def _demo(args: argparse.Namespace) -> Any:
    from src.pipeline.demo import run_demo

    return run_demo(Path(args.out), synthetic=args.synthetic)


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser com os subcomandos registrados."""
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        if name == "validate":
            sp.add_argument("file")
        if name == "retrain":
            sp.add_argument("--with-windows", nargs="+", metavar="Mk")
        if name == "demo":
            sp.add_argument("--synthetic", action="store_true", required=True)
            sp.add_argument("--out", default="demo_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executa o subcomando e traduz exceções em exit codes."""
    args = build_parser().parse_args(argv)
    new_run_id()
    try:
        COMMANDS[args.command](args)
    except ContractViolationError as exc:
        logger.error("contrato_violado", extra={"detail": str(exc)})
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
