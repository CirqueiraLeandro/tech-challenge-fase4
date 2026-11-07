# ML Canvas — Sustentação de um Modelo de Credit Scoring

| Bloco | Conteúdo |
|---|---|
| **Proposta de valor** | Reduzir perdas por inadimplência e por recusa indevida, **detectando a degradação silenciosa** do modelo, explicando sua causa e mantendo as decisões auditáveis e conformes à LGPD. |
| **Decisões** | Aprovar/negar crédito (score < 0,5 → aprovado); ação sobre o modelo: observar, investigar ou retreinar (política OK/ATENÇÃO/CRÍTICO); rever manualmente decisões contestadas ou de fronteira. |
| **Tarefa de ML** | Classificação binária de inadimplência (`default.payment.next.month`), XGBoost tabular com baseline de regressão logística. |
| **Fontes de dados** | UCI *Default of Credit Card Clients* (30.000 clientes); janelas de produção **simuladas** por um modelo causal. |
| **Coleta / ingestão** | Lote passa por pseudonimização do ID e pelo **contrato Pandera** (8 regras rígidas, 3 suaves); reprovado → quarentena e bloqueio total (exit 2). |
| **Features** | 21 originais + 4 derivadas; **sem** `sex`, `marriage` e identificadores. |
| **Construção do modelo** | Fit só no treino do reference; candidate → **gate (AUC ≥ 0,75)** → champion; versões e linhagem (hash do dataset, git, contrato) no MLflow. |
| **Previsões / uso** | Pontuação em lote por janela; *reason codes* por cliente (art. 20). |
| **Avaliação offline** | AUC 0,7836 · Gini 0,567 · KS 0,436 · Brier 0,132 (teste do reference); fairness por sexo e idade. |
| **Monitoramento (online)** | PSI/KS/Chi²/JS + Evidently por janela; AUC, KS e **gap de calibração**; PSI do score; SLOs: AUC ≥ 0,72, PSI < 0,20, gap de calibração absoluto ≤ 5 p.p., frescor < 7 dias; Grafana, Prometheus, MLflow, logs JSON. |
| **Sustentação / retreino** | Política proporcional; retreino contínuo com janelas rotuladas recentes (mesmo gate); runbook em `docs/MONITORING.md`. |
| **Governança / LGPD** | Base legal art. 7º, X; minimização; pseudonimização (HMAC-SHA256); retenção executável (dry-run por padrão); RIPD/ROPA; auditoria de vieses. |
| **Riscos** | Drift de dados e de conceito (desemprego latente); discriminação por proxy (`education`, `age`); atraso do rótulo; falsos alarmes por PSI sem impacto em desempenho; dependência de versões (Evidently × pandas). |
| **Métricas de sucesso** | Zero lote ruim ingerido; M1 (controle) sem alerta; degradação visível em ≤ 1 janela de atraso; recuperação de AUC/calibração após o retreino; DI ≥ 0,8 em todos os atributos auditados. |
