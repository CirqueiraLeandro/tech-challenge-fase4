"""Aplica a política de retenção. Por padrão apenas simula (--execute apaga de verdade)."""

from __future__ import annotations

import argparse
import sys

from src.privacy.retention import apply_retention, build_rules


def main(argv: list[str] | None = None) -> int:
    """Executa a retenção; retorna 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="elimina de fato (padrão: dry-run)")
    args = parser.parse_args(argv)
    apply_retention(build_rules(), dry_run=not args.execute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
