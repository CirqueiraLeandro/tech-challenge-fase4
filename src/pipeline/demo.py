"""Demo ponta a ponta com dados SINTÉTICOS em workspace isolado (usada no CI).

Nunca toca em data/, models/ nem reports/ do projeto: dados, modelos, relatórios e o
tracking do MLflow vão para ``--out``. Dado real nunca entra no CI (privacy by design).
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config import get_settings
from src.data.synthetic import make_synthetic_credit
from src.logger import get_logger

logger = get_logger(__name__)
DEMO_ROWS = 15_000
DIR_KEYS = [
    "raw", "reference", "production", "incoming", "quarantine", "models", "logs", "reports",
]  # fmt: skip


def isolate_environment(out: Path) -> None:
    """Redireciona todos os diretórios e o MLflow para o workspace da demo."""
    out.mkdir(parents=True, exist_ok=True)
    for key in DIR_KEYS:
        os.environ[f"{key.upper()}_DIR"] = str(out / key)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(out / 'mlflow.db').as_posix()}"
    os.environ["PUSHGATEWAY_URL"] = ""
    os.environ.setdefault("MIN_AUC", "0.65")
    get_settings.cache_clear()


def run_demo(out: Path, synthetic: bool = True) -> None:
    """Executa o pipeline completo (all) mais o retreino, sobre dados sintéticos."""
    from src.pipeline import cli

    isolate_environment(out)
    settings = get_settings()
    raw = settings.path("raw")
    raw.mkdir(parents=True, exist_ok=True)
    make_synthetic_credit(DEMO_ROWS).to_csv(
        raw / "credit_default.csv", index=False, encoding="utf-8"
    )
    args = cli.build_parser().parse_args(["all"])
    cli.COMMANDS["all"](args)
    cli.COMMANDS["retrain"](cli.build_parser().parse_args(["retrain"]))
    cli.COMMANDS["report"](args)
    logger.info("demo_concluida", extra={"out": str(out), "synthetic": synthetic})
