"""Gera lotes sintéticos (bom e ruim) para o job contract-gate do CI, sem dataset real."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_settings
from src.contracts.badbatch import make_bad_batch
from src.data.synthetic import make_synthetic_credit
from src.privacy.pseudonymize import pseudonymize_ids


def main(argv: list[str] | None = None) -> int:
    """Escreve good_batch.csv (bruto) e bad_batch.csv (ingerido, corrompido) em --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/incoming")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic_credit(1500)
    raw.to_csv(out / "good_batch.csv", index=False, encoding="utf-8")
    ingested = pseudonymize_ids(raw, get_settings())
    make_bad_batch(ingested).to_csv(out / "bad_batch.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
