"""Gera data/incoming/bad_batch.csv a partir do pool de produção (violações controladas)."""

from __future__ import annotations

import sys

from src.config import get_settings
from src.contracts.badbatch import INJECTIONS, make_bad_batch
from src.data.loader import load_pool
from src.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Grava o lote corrompido e registra as injeções."""
    settings = get_settings()
    bad = make_bad_batch(load_pool(settings), seed=settings.random_seed)
    out = settings.path("incoming") / "bad_batch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    bad.to_csv(out, index=False, encoding="utf-8")
    logger.info(
        "bad_batch_gerado", extra={"path": str(out), "rows": len(bad), "injecoes": INJECTIONS}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
