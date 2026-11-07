"""Fixtures compartilhadas: ambiente de teste isolado e dados sintéticos."""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")
