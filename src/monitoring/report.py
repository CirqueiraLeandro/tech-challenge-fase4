"""Painel HTML consolidado (reports/index.html): abre direto do repositório, sem servidor.

Gráficos em matplotlib embutidos como PNG base64 (sem CDN). Lê os artefatos gerados
pelo pipeline: contratos, drift, performance, causal, fairness, retreino e logs de tasks.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Settings, get_settings, load_monitoring_config  # noqa: E402
from src.logger import current_log_file, get_logger  # noqa: E402

logger = get_logger(__name__)
STYLE = (
    "body{font-family:Segoe UI,sans-serif;margin:2rem auto;max-width:1100px;color:#222}"
    "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:4px 10px;"
    "font-size:.9rem}th{background:#f0f0f0}.ok{color:#1b7f3b;font-weight:600}"
    ".bad{color:#b00020;font-weight:600}.warn{color:#b26a00;font-weight:600}"
    "h2{border-bottom:2px solid #ddd;padding-bottom:.2rem;margin-top:2rem}img{max-width:100%}"
)


def read_json(path: Path) -> dict | None:
    """Lê um JSON em UTF-8, ou ``None`` se o arquivo não existir."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def png_base64(fig: plt.Figure) -> str:
    """Serializa a figura como <img> base64."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">'


def html_table(df: pd.DataFrame) -> str:
    """Tabela HTML simples (valores escapados)."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def task_status(path: Path) -> str:
    """Último evento ``task_fim`` por task a partir do log JSONL do pipeline."""
    if not path.exists():
        return "<p>Sem execuções registradas.</p>"
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"task_fim"' in line:
            event = json.loads(line)
            latest[event["task"]] = event
    rows = [
        {"task": t, "status": e["status"], "duração (ms)": e["duration_ms"], "linhas": e["rows"],
         "quando": e["timestamp"][:19]}
        for t, e in latest.items()
    ]  # fmt: skip
    return html_table(pd.DataFrame(rows)) if rows else "<p>Sem tasks registradas.</p>"


def contracts_section(reports_dir: Path) -> str:
    """Tabela de lotes validados (aprovados e reprovados)."""
    rows = []
    for path in sorted((reports_dir / "contracts").glob("*_violations.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "lote": path.name.replace("_violations.json", ""),
                "linhas": data["n_rows"],
                "status": "APROVADO" if data["passed"] else "REPROVADO",
                "regras violadas": ", ".join(data["violated_rules"]) or "-",
                "avisos suaves": len(data["soft_warnings"]),
            }
        )
    return html_table(pd.DataFrame(rows)) if rows else "<p>Nenhum lote validado.</p>"


def psi_heatmap(long: pd.DataFrame) -> str:
    """Heatmap PSI feature x janela (escala de severidade 0 a 0,3)."""
    pivot = long.pivot(index="feature", columns="window", values="psi")
    fig, ax = plt.subplots(figsize=(7, 8))
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0, vmax=0.3, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for (i, j), value in np.ndenumerate(pivot.to_numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, label="PSI")
    ax.set_title("PSI por feature x janela")
    return png_base64(fig)


def auc_curve(summary: dict, slo: float) -> str:
    """AUC por janela com a linha de SLO."""
    names = list(summary)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(names, [summary[n]["auc"] for n in names], marker="o", label="AUC")
    ax.axhline(slo, color="red", linestyle="--", label=f"SLO {slo}")
    ax.set_title("AUC por janela")
    ax.legend()
    return png_base64(fig)


def severity_table(summary: dict) -> pd.DataFrame:
    """Tabela de severidade por janela (drift, desempenho e ação da política)."""
    return pd.DataFrame(
        [
            {
                "janela": n, "drift_share": f"{s['drift_share']:.2f}",
                "max PSI": f"{s['max_psi']:.3f}",
                "AUC": f"{s['auc']:.4f}", "queda AUC": f"{s['auc_drop']:.4f}",
                "gap calib.": f"{s['calibration_gap']:+.4f}", "severidade": s["severity"],
                "ação": s["action"],
            }
            for n, s in summary.items()
        ]
    )  # fmt: skip


def drift_section(reports_dir: Path, slo: float) -> str:
    """Seção de drift: severidade por janela, heatmap, curva de AUC e links do Evidently."""
    summary = read_json(reports_dir / "drift" / "drift_summary.json")
    csv = reports_dir / "drift" / "drift_summary.csv"
    if summary is None or not csv.exists():
        return "<p>Execute <code>monitor</code> para gerar o drift.</p>"
    links = " | ".join(
        f'<a href="drift/{n}_data_drift.html">{n}</a>'
        for n in summary
        if (reports_dir / "drift" / f"{n}_data_drift.html").exists()
    )
    long = pd.read_csv(csv, encoding="utf-8")
    charts = psi_heatmap(long) + auc_curve(summary, slo)
    return f"{html_table(severity_table(summary))}<p>Relatórios Evidently: {links}</p>{charts}"


CAUSAL_COLUMNS = [
    "method", "mechanism", "delta_auc", "delta_calibration_gap", "delta_brier",
    "delta_psi_sum", "solo_calibration_gap", "delta_score_psi",
]  # fmt: skip


def causal_section(reports_dir: Path) -> str:
    """Atribuição causal (tabela + gráfico)."""
    csv, png = (
        reports_dir / "causal" / "attribution.csv",
        reports_dir / "causal" / "attribution.png",
    )
    if not csv.exists():
        return "<p>Execute <code>causal</code>.</p>"
    table = pd.read_csv(csv, encoding="utf-8")
    table = table[[c for c in CAUSAL_COLUMNS if c in table.columns]].round(4).fillna("-")
    image = base64.b64encode(png.read_bytes()).decode() if png.exists() else ""
    return f'{html_table(table)}<img src="data:image/png;base64,{image}">'


def fairness_section(reports_dir: Path) -> str:
    """Resumo de fairness (DI e diferença de oportunidade por atributo)."""
    data = read_json(reports_dir / "fairness" / "fairness_report.json")
    if data is None:
        return "<p>Execute <code>fairness</code>.</p>"
    rows = [
        {
            "conjunto": subset, "atributo": attr,
            "disparate impact": round(v["disparate_impact"], 3),
            "regra 80%": "ok" if v["passes_80_percent_rule"] else "VIOLADA",
            "dif. oportunidade": round(v["equal_opportunity_diff"], 3),
        }
        for subset, attrs in data.items()
        for attr, v in attrs.items()
    ]  # fmt: skip
    return html_table(pd.DataFrame(rows))


def retrain_section(reports_dir: Path) -> str:
    """Resultado do retreino (antes x depois no hold-out)."""
    data = read_json(reports_dir / "retrain" / "retrain_summary.json")
    if data is None:
        return "<p>Execute <code>retrain</code>.</p>"
    rows = [
        {"momento": k, **{m: round(v, 4) for m, v in data[k].items()}} for k in ("before", "after")
    ]
    intro = f"<p>Treinado com {', '.join(data['trained_with'])}; hold-out {data['holdout']}.</p>"
    return intro + html_table(pd.DataFrame(rows))


def render_page(sections: list[tuple[str, str]]) -> str:
    """Monta a página HTML completa a partir de (título, conteúdo) das seções."""
    body = "".join(f"<h2>{html.escape(t)}</h2>{c}" for t, c in sections)
    title = "Painel de Monitoramento - Credit Scoring"
    return (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{STYLE}</style></head>"
        f"<body><h1>{title}</h1>{body}</body></html>"
    )


def build_report(settings: Settings | None = None) -> Path:
    """Gera reports/index.html a partir dos artefatos existentes."""
    settings = settings or get_settings()
    reports = settings.path("reports")
    slo = load_monitoring_config().auc_slo
    page = render_page(
        [
            ("Saúde das tasks (último run)", task_status(current_log_file())),
            ("Data contract (lotes validados)", contracts_section(reports)),
            ("Drift de dados e de conceito por janela", drift_section(reports, slo)),
            ("Análise causal da degradação (janela final)", causal_section(reports)),
            ("Auditoria de vieses (fairness)", fairness_section(reports)),
            ("Retreino (fechamento do ciclo)", retrain_section(reports)),
        ]
    )
    path = reports / "index.html"
    path.write_text(page, encoding="utf-8")
    logger.info("painel_gerado", extra={"path": str(path)})
    return path
