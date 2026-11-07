"""Valida o ambiente: imports, versões, diretórios e configuração."""

from __future__ import annotations

import importlib
import sys
from importlib import metadata

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
PACKAGES = [
    "pandas", "numpy", "scipy", "scikit-learn", "xgboost", "pandera", "evidently",
    "mlflow", "prometheus-client", "pydantic", "pydantic-settings", "pyyaml",
    "matplotlib", "psutil", "pyarrow",
]  # fmt: skip
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "prometheus-client": "prometheus_client",
    "pydantic-settings": "pydantic_settings",
}
DIRS = ["raw", "reference", "production", "incoming", "quarantine", "models", "logs", "reports"]


def check_packages() -> list[str]:
    """Retorna a lista de pacotes que falharam ao importar."""
    failures = []
    for pkg in PACKAGES:
        try:
            importlib.import_module(IMPORT_NAMES.get(pkg, pkg.replace("-", "_")))
            logger.info("pacote_ok", extra={"package": pkg, "version": metadata.version(pkg)})
        except Exception as exc:  # noqa: BLE001
            failures.append(pkg)
            logger.error("pacote_falhou", extra={"package": pkg, "error": str(exc)})
    return failures


def ensure_dirs() -> None:
    """Cria os diretórios de trabalho, se ausentes."""
    settings = get_settings()
    for name in DIRS:
        settings.path(name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Retorna 0 se o ambiente está íntegro, 1 caso contrário."""
    ensure_dirs()
    failures = check_packages()
    logger.info("validate_env_fim", extra={"falhas": len(failures)})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
