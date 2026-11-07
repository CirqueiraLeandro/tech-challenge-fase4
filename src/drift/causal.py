"""Análise causal da degradação (seção 11).

- **Intervencional** (verdade de campo, pois controlamos o gerador): regenera a janela com a
  MESMA seed desligando um mecanismo por vez (``do(mecanismo = referência)``) e mede o
  quanto da degradação cada um explica; também o efeito isolado e a não-aditividade.
- **Observacional** (o que se faria em produção): reverte as features que driftaram para a
  distribuição de referência por mapeamento de quantis e mede o efeito no score/AUC.
  Comparar os dois mostra onde o diagnóstico observacional falha (esperado: ``desemprego``,
  cujo choque U é latente e invisível em P(X)).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DriftScenario, Settings, get_settings, load_drift_scenario  # noqa: E402
from src.constants import MODEL_FEATURES, TARGET  # noqa: E402
from src.data.loader import load_pool, load_reference  # noqa: E402
from src.drift.detector import drift_share, feature_view  # noqa: E402
from src.drift.monitor import MonitorContext, build_context  # noqa: E402
from src.drift.performance import window_performance  # noqa: E402
from src.drift.simulation import MECHANISM_COLUMNS, generate_window, split_chunks  # noqa: E402
from src.drift.stattests import psi_from_probs  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models.evaluate import calibration_gap, compute_metrics  # noqa: E402

logger = get_logger(__name__)

# Features observáveis que cada mecanismo manifesta (hipótese do diagnóstico observacional)
FEATURE_GROUPS: dict[str, list[str]] = {
    "inflacao": [
        *[f"bill_amt{i}" for i in range(1, 7)],
        *[f"pay_amt{i}" for i in range(1, 7)],
        "utilization",
        "payment_ratio",
    ],
    "aperto": ["limit_bal"],
    "desemprego": ["pay_0", "pay_2", "max_delay", "months_delayed"],
    "perfil_jovem": ["age"],
}
METRICS = ["auc", "calibration_gap", "brier", "drift_share", "psi_sum"]


def evaluate_frame(ctx: MonitorContext, df: pd.DataFrame) -> dict[str, float]:
    """AUC, gap de calibração, Brier e fração de features em drift de uma janela."""
    perf, _ = window_performance(
        ctx.model, df, ctx.baseline_auc, ctx.score_bins, ctx.settings.decision_threshold
    )
    table = ctx.detector.analyze(df)
    share = drift_share(table)
    return {
        "auc": perf.auc,
        "calibration_gap": perf.calibration_gap,
        "brier": perf.brier,
        "drift_share": share,
        "psi_sum": float(table["psi"].sum()),
    }


def _scenario_metrics(
    ctx: MonitorContext, chunk: pd.DataFrame, k: int, t: float, scenario: DriftScenario, seed: int,
    disabled: frozenset[str],
) -> dict[str, float]:  # fmt: skip
    window = generate_window(chunk, k, t, scenario, seed, disabled).data
    return evaluate_frame(ctx, window)


def interventional_attribution(
    ctx: MonitorContext, chunk: pd.DataFrame, k: int, t: float, scenario: DriftScenario, seed: int
) -> pd.DataFrame:
    """Atribuição por intervenção do(mecanismo = referência), um por vez e isolado."""
    mechanisms = list(MECHANISM_COLUMNS)

    def run(off: list[str]) -> dict[str, float]:
        """Métricas da janela com os mecanismos ``off`` desligados."""
        return _scenario_metrics(ctx, chunk, k, t, scenario, seed, frozenset(off))

    full, none = run([]), run(mechanisms)
    rows = []
    for m in mechanisms:
        without, solo = run([m]), run([x for x in mechanisms if x != m])
        row: dict[str, object] = {"method": "interventional", "mechanism": m}
        for name in METRICS:
            row[f"delta_{name}"] = full[name] - without[name]  # o que o mecanismo causa
            row[f"solo_{name}"] = solo[name] - none[name]  # efeito isolado
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs |= {"full": full, "none": none}
    return table


def quantile_revert(current: pd.Series, reference: pd.Series) -> np.ndarray:
    """Mapeia ``current`` para a distribuição de ``reference`` preservando os postos."""
    ranks = current.rank(method="average").to_numpy() - 0.5
    probs = np.clip(ranks / len(current), 0.0, 1.0)
    return np.quantile(reference.to_numpy(float), probs)


def _score_with(ctx: MonitorContext, x: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    score = ctx.model.predict_proba(x)[:, 1]
    m = compute_metrics(y, score, ctx.settings.decision_threshold)
    probs = ctx.score_bins.probs(score)
    return {
        "auc": m["auc"],
        "calibration_gap": calibration_gap(y, score),
        "brier": m["brier"],
        "score_psi": psi_from_probs(ctx.score_bins.ref_probs, probs),
    }


def observational_attribution(
    ctx: MonitorContext, window: pd.DataFrame, ref: pd.DataFrame
) -> pd.DataFrame:
    """Reverte, por grupo de features, a janela para a distribuição do reference."""
    x, ref_x = feature_view(window), feature_view(ref)
    y = window[TARGET].to_numpy()
    base = _score_with(ctx, x, y)
    rows = []
    for mechanism, features in FEATURE_GROUPS.items():
        reverted = x.copy()
        for f in (f for f in features if f in MODEL_FEATURES):
            reverted[f] = quantile_revert(x[f], ref_x[f])
        after = _score_with(ctx, reverted, y)
        rows.append(
            {
                "method": "observational",
                "mechanism": mechanism,
                "delta_auc": base["auc"] - after["auc"],
                "delta_calibration_gap": base["calibration_gap"] - after["calibration_gap"],
                "delta_brier": base["brier"] - after["brier"],
                "delta_score_psi": base["score_psi"] - after["score_psi"],
            }
        )
    return pd.DataFrame(rows)


def dominant(table: pd.DataFrame, column: str) -> str:
    """Mecanismo com maior impacto absoluto na coluna indicada."""
    return str(table.loc[table[column].abs().idxmax(), "mechanism"])


def plot_attribution(inter: pd.DataFrame, obs: pd.DataFrame, path: Path) -> Path:
    """Barras: Δ gap de calibração (interv. x observ.) e Δ soma dos PSI (intervencional)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(inter))
    axes[0].bar(x - 0.2, inter["delta_calibration_gap"] * 100, 0.4, label="intervencional")
    axes[0].bar(x + 0.2, obs["delta_calibration_gap"] * 100, 0.4, label="observacional")
    axes[0].set_title("Δ gap de calibração (p.p.)")
    axes[1].bar(x, inter["delta_psi_sum"], color="#4c72b0")
    axes[1].set_title("Δ soma dos PSI - intervencional")
    for ax in axes:
        ax.set_xticks(x, inter["mechanism"], rotation=20)
        ax.axhline(0, color="black", linewidth=0.6)
    axes[0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _agreement(inter: pd.DataFrame, obs: pd.DataFrame) -> dict[str, bool]:
    """Por mecanismo: o Δ gap observacional tem o mesmo sinal e >= 50% da magnitude do real?"""
    real = inter.set_index("mechanism")["delta_calibration_gap"]
    seen = obs.set_index("mechanism")["delta_calibration_gap"]
    return {
        m: bool(np.sign(real[m]) == np.sign(seen[m]) and abs(seen[m]) >= 0.5 * abs(real[m]))
        for m in real.index
    }


def build_verdict(inter: pd.DataFrame, obs: pd.DataFrame) -> dict:
    """Veredito: mecanismos dominantes, concordância observacional e (não-)aditividade."""
    full, none = inter.attrs["full"], inter.attrs["none"]
    return {
        "dominant_data_interventional": dominant(inter, "delta_psi_sum"),
        "dominant_performance_interventional": dominant(inter, "delta_calibration_gap"),
        "dominant_performance_observational": dominant(obs, "delta_calibration_gap"),
        "full": full,
        "none": none,
        "observational_matches_intervention": _agreement(inter, obs),
        "sum_solo_calibration_gap": float(inter["solo_calibration_gap"].sum()),
        "total_calibration_gap_effect": full["calibration_gap"] - none["calibration_gap"],
    }


def write_attribution(inter: pd.DataFrame, obs: pd.DataFrame, out: Path) -> dict:
    """Grava attribution.csv/.png e attribution.json (com o veredito da comparação)."""
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat([inter, obs], ignore_index=True)
    table.to_csv(out / "attribution.csv", index=False, encoding="utf-8")
    plot_attribution(inter, obs, out / "attribution.png")
    verdict = build_verdict(inter, obs)
    text = json.dumps(verdict, ensure_ascii=False, indent=2)
    (out / "attribution.json").write_text(text, encoding="utf-8")
    return verdict


def run(window: str | None = None, settings: Settings | None = None) -> dict:
    """Roda a análise causal na janela final (ou a indicada) e grava reports/causal/."""
    settings = settings or get_settings()
    scenario = load_drift_scenario()
    k = scenario.windows.index(window) if window else len(scenario.windows) - 1
    ctx = build_context(settings)
    chunk = split_chunks(load_pool(settings), len(scenario.windows), settings.random_seed)[k]
    t = scenario.intensities[k]
    inter = interventional_attribution(ctx, chunk, k, t, scenario, settings.random_seed)
    full = generate_window(chunk, k, t, scenario, settings.random_seed).data
    obs = observational_attribution(ctx, full, load_reference(settings))
    verdict = write_attribution(inter, obs, settings.path("reports") / "causal")
    logger.info(
        "analise_causal",
        extra={
            "window": scenario.windows[k],
            **{k2: v for k2, v in verdict.items() if isinstance(v, str)},
        },
    )
    return verdict
