"""Política de resposta ao drift: severidade -> ação proporcional (seção 8.6).

Mitigar drift é uma política proporcional à velocidade da mudança e ao custo do erro,
não "mandar treinar de novo" a cada oscilação. Também é um controle de governança:
um modelo degradado gera decisões automatizadas incorretas/injustas (LGPD).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MonitoringConfig

OK, ATTENTION, CRITICAL = "OK", "ATENCAO", "CRITICO"
ACTIONS = {
    OK: "observar",
    ATTENTION: "investigar e revisar thresholds",
    CRITICAL: "retreinar com janela recente rotulada + revisão humana",
}
SEVERITY_LEVEL = {OK: 0, ATTENTION: 1, CRITICAL: 2}


@dataclass(frozen=True)
class PolicyDecision:
    """Decisão da política para uma janela."""

    severity: str
    action: str
    reasons: list[str]


def decide(
    max_psi: float, auc_drop: float, calibration_gap: float, cfg: MonitoringConfig
) -> PolicyDecision:
    """Aplica a tabela de severidade.

    CRITICO: PSI >= severo E queda de AUC >= crítica, OU |gap de calibração| > limite.
    ATENCAO: algum PSI moderado OU queda de AUC de atenção. OK: caso contrário.
    """
    reasons: list[str] = []
    critical = max_psi >= cfg.psi_severe and auc_drop >= cfg.auc_drop_critical
    if critical:
        reasons.append("PSI severo com queda de AUC crítica")
    if abs(calibration_gap) > cfg.calibration_gap_critical:
        critical = True
        reasons.append("gap de calibração acima do limite (concept drift)")
    if critical:
        return PolicyDecision(CRITICAL, ACTIONS[CRITICAL], reasons)
    if max_psi >= cfg.psi_moderate:
        reasons.append("PSI moderado/severo sem queda crítica de AUC")
    if auc_drop >= cfg.auc_drop_attention:
        reasons.append("queda de AUC de atenção")
    if reasons:
        return PolicyDecision(ATTENTION, ACTIONS[ATTENTION], reasons)
    return PolicyDecision(OK, ACTIONS[OK], ["sem sinais de degradação"])
