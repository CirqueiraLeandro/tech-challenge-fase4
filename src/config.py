"""Configuração centralizada via Pydantic Settings (.env) e loaders YAML validados."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWN_MECHANISMS = ("inflacao", "aperto", "desemprego", "perfil_jovem")


class Settings(BaseSettings):
    """Configurações da aplicação, lidas de variáveis de ambiente/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    random_seed: int = 42
    log_level: str = "INFO"

    raw_dir: str = "data/raw"
    reference_dir: str = "data/reference"
    production_dir: str = "data/production"
    incoming_dir: str = "data/incoming"
    quarantine_dir: str = "data/quarantine"
    models_dir: str = "models"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    configs_dir: str = "configs"

    model_type: str = "xgboost"
    min_auc: float = 0.75
    decision_threshold: float = 0.5
    min_batch_rows: int = 100

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    pushgateway_url: str = ""
    alert_webhook_url: str = ""
    pseudonymization_key: str = ""

    retention_raw_days: int = 3650
    retention_quarantine_days: int = 30
    retention_logs_days: int = 90

    @model_validator(mode="after")
    def _require_key_outside_test(self) -> Settings:
        """Exige a chave de pseudonimização fora do ambiente de teste."""
        if self.environment != "test" and not self.pseudonymization_key:
            raise ValueError("PSEUDONYMIZATION_KEY é obrigatória fora de ENVIRONMENT=test")
        return self

    def path(self, name: str) -> Path:
        """Resolve o diretório configurado ``<name>_dir`` relativo à raiz do projeto."""
        relative = Path(getattr(self, f"{name}_dir"))
        return relative if relative.is_absolute() else PROJECT_ROOT / relative

    @property
    def reference_path(self) -> Path:
        """Arquivo parquet do dataset de referência."""
        return self.path("reference") / "reference.parquet"

    @property
    def pool_path(self) -> Path:
        """Arquivo parquet do pool de produção."""
        return self.path("production") / "pool.parquet"


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância singleton de :class:`Settings`."""
    return Settings()


class MechanismConfig(BaseModel):
    """Configuração de um mecanismo causal do cenário de drift."""

    enabled: bool = True


class DriftScenario(BaseModel):
    """Cenário de simulação de drift (configs/drift_scenario.yaml)."""

    windows: list[str]
    intensities: list[float]
    window_size: int = Field(gt=0)
    mechanisms: dict[str, MechanismConfig]

    @field_validator("intensities")
    @classmethod
    def _intensities_in_range(cls, value: list[float]) -> list[float]:
        if any(not 0.0 <= v <= 1.0 for v in value):
            raise ValueError("intensidades devem estar em [0, 1]")
        return value

    @field_validator("mechanisms")
    @classmethod
    def _known_mechanisms(cls, value: dict[str, MechanismConfig]) -> dict[str, MechanismConfig]:
        unknown = set(value) - set(KNOWN_MECHANISMS)
        if unknown:
            raise ValueError(f"mecanismos desconhecidos: {sorted(unknown)}")
        return value

    @model_validator(mode="after")
    def _same_length(self) -> DriftScenario:
        if len(self.windows) != len(self.intensities):
            raise ValueError("windows e intensities devem ter o mesmo tamanho")
        return self


class MonitoringConfig(BaseModel):
    """Limiares e SLOs (configs/monitoring.yaml)."""

    psi_moderate: float = 0.10
    psi_severe: float = 0.20
    ks_alpha: float = 0.05
    auc_drop_attention: float = 0.02
    auc_drop_critical: float = 0.05
    calibration_gap_critical: float = 0.05
    auc_slo: float = 0.72
    bonferroni: bool = False
    critical_features: list[str] = Field(default_factory=list)


def load_yaml(path: Path) -> dict[str, Any]:
    """Lê um YAML em UTF-8 e retorna o dicionário (vazio se o arquivo estiver vazio)."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_drift_scenario(path: Path | None = None) -> DriftScenario:
    """Carrega e valida o cenário de drift."""
    path = path or get_settings().path("configs") / "drift_scenario.yaml"
    return DriftScenario(**load_yaml(path))


def load_monitoring_config(path: Path | None = None) -> MonitoringConfig:
    """Carrega e valida os limiares de monitoramento."""
    path = path or get_settings().path("configs") / "monitoring.yaml"
    return MonitoringConfig(**load_yaml(path))
