"""Testes do logger JSON e da proibição de print() em src/."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from src.logger import JsonFormatter, new_run_id, run_id_var

pytestmark = pytest.mark.unit


def test_json_line_has_run_id_and_extra():
    run_id = new_run_id()
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "evento", None, None)
    record.rows = 10
    data = json.loads(JsonFormatter().format(record))
    assert data["run_id"] == run_id == run_id_var.get()
    assert data["rows"] == 10 and data["event"] == "evento"


def test_unicode_not_escaped():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "validação", None, None)
    assert "validação" in JsonFormatter().format(record)


def test_no_print_in_src():
    pattern = re.compile(r"^\s*print\(", re.MULTILINE)
    for path in Path("src").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
