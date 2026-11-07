"""Task base (Template Method): instrumentação escrita UMA vez para todos os estágios.

``run()``: log de início -> ``execute()`` -> métricas de status/duração/memória/linhas ->
log de fim. Em exceção: status 0, alerta ``task_failed`` e re-raise.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Any

import psutil

from src.logger import get_logger
from src.monitoring.alerts import Alert, AlertDispatcher
from src.monitoring.metrics import PipelineMetrics, get_metrics

logger = get_logger(__name__)


def peak_memory_bytes() -> int:
    """Memória de pico do processo (peak working set no Windows; RSS nos demais)."""
    info = psutil.Process().memory_info()
    if sys.platform == "win32":
        return int(info.peak_wset)
    return int(info.rss)


class Task(ABC):
    """Estágio instrumentado: subclasses só implementam :meth:`execute`."""

    name: str = "task"

    def __init__(
        self, dispatcher: AlertDispatcher | None = None, metrics: PipelineMetrics | None = None
    ) -> None:
        self.dispatcher = dispatcher or AlertDispatcher()
        self._metrics = metrics
        self.rows = 0

    @property
    def metrics(self) -> PipelineMetrics:
        """Métricas injetadas ou o registry corrente."""
        return self._metrics or get_metrics()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Lógica do estágio (deve definir ``self.rows`` quando aplicável)."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Executa o estágio com logging e métricas padronizados."""
        logger.info("task_inicio", extra={"task": self.name})
        start = time.perf_counter()
        try:
            result = self.execute(*args, **kwargs)
        except Exception as exc:
            self._finish(start, ok=False)
            self.dispatcher.notify(
                Alert("task_failed", "critical", f"task {self.name} falhou: {type(exc).__name__}")
            )
            raise
        self._finish(start, ok=True)
        return result

    def _finish(self, start: float, ok: bool) -> None:
        duration = time.perf_counter() - start
        self.metrics.set_task(self.name, ok, duration, peak_memory_bytes())
        logger.info(
            "task_fim",
            extra={
                "task": self.name,
                "status": "ok" if ok else "falha",
                "duration_ms": round(duration * 1000, 1),
                "rows": self.rows,
            },
        )
