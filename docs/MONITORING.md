# Monitoramento, SLIs/SLOs e Runbook

Duas camadas **complementares** e **opcionais em tempo de execução** (o pipeline nunca falha porque
o backend de observabilidade está fora — log `observability_degraded` e segue):

| Camada | Ferramenta | Responde a |
|--------|-----------|-----------|
| Ciclo de vida do modelo | MLflow (`credit-training`, `data-validation`, `drift-monitoring`, Model Registry com aliases) | "que modelo, treinado com quê, como evoluiu por janela?" |
| Saúde operacional + alertas | Pushgateway → Prometheus → Grafana | "o pipeline está saudável? há degradação agora?" |
| Evidência offline | `reports/index.html` (sem servidor) + logs JSON | "o que aconteceu no último run?" |

Os logs são estruturados (`logs/pipeline.jsonl`, `logs/alerts.jsonl`): `timestamp`, `level`, `event`,
`run_id`, `task`, `status`, `rows`, `duration_ms`, `window`… **nunca** dados pessoais (há teste
automatizado — `tests/test_pipeline_e2e.py::test_logs_contain_no_personal_data`).

## SLIs / SLOs

| SLI | SLO | Fonte |
|-----|-----|-------|
| Taxa de sucesso das tasks | >= 99% | `credit_task_status` |
| Lote reprovado no contrato → linhas ingeridas | 0 (bloqueio total) | `credit_contract_status`, contrato atômico |
| AUC por janela | >= 0,72 | `credit_model_auc` |
| PSI das features | < 0,20 | `credit_feature_psi` |
| Gap de calibração (previsto − observado) | \|gap\| <= 5 p.p. | `credit_calibration_gap` |
| Frescor do último sucesso | < 7 dias | `credit_last_success_timestamp` |

## Métricas, queries e ação

Rótulos só de conjuntos fechados (features, janelas M1..M6, ids de regra, tasks): cardinalidade
controlada (valores fora do conjunto viram `other`).

| Métrica | Definição / por que importa | PromQL | Limiar | Ação (runbook) |
|---------|----------------------------|--------|--------|----------------|
| `credit_task_status{task}` | 1 ok / 0 falha da última execução | `credit_task_status == 0` | 0 = falha | [PipelineTaskFailed](#pipelinetaskfailed) |
| `credit_task_duration_seconds{task}` | duração; pega regressões de desempenho | `credit_task_duration_seconds` | > 300 s (laranja 60 s) | investigar dados/volume |
| `credit_task_peak_memory_bytes{task}` | infraestrutura: pico de memória | `credit_task_peak_memory_bytes` | > 4 GB | revisar tamanho de janela |
| `credit_rows_processed_total{stage}` | volume por estágio (queda = fonte parou) | `sum by (stage) (credit_rows_processed_total)` | — | comparar com o esperado |
| `credit_rows_rejected_total{rule}` | linhas barradas por regra do contrato | `sum by (rule) (credit_rows_rejected_total)` | > 0 | [ContractViolation](#contractviolation) |
| `credit_contract_status{batch}` | 1 aprovado / 0 reprovado | `credit_contract_status == 0` | 0 | [ContractViolation](#contractviolation) |
| `credit_feature_psi{feature,window}` | magnitude do drift por feature | `max by (window) (credit_feature_psi)` | >= 0,20 | [DataDriftCritical](#datadriftcritical) |
| `credit_drift_share{window}` | fração de features em drift | `credit_drift_share` | > 30% | investigar |
| `credit_model_auc{window}` | desempenho com rótulos | `credit_model_auc < 0.72` | < 0,72 | [ModelAUCBelowSLO](#modelaucbelowslo) |
| `credit_calibration_gap{window}` | sinal de concept drift | `abs(credit_calibration_gap) > 0.05` | > 5 p.p. | retreino + revisão humana |
| `credit_alerts_total{type,severity}` | alertas emitidos | `sum by (type, severity) (credit_alerts_total)` | — | triagem |
| `credit_last_success_timestamp{task}` | detecta pipeline parado | `time() - credit_last_success_timestamp > 604800` | > 7 d | [PipelineStale](#pipelinestale) |

O Pushgateway recebe um **grupo por task** (`pipeline=<task>`), com registry novo a cada execução —
assim o dashboard nunca soma séries duplicadas.

## Alertas (`monitoring/prometheus/alert_rules.yml`)

Validadas com `promtool check rules` (via container). Visíveis em Prometheus `/alerts` e no Grafana.

### DataDriftCritical
`max by (window) (credit_feature_psi) >= 0.20`. **Ação:** abrir `reports/drift/<Mk>_data_drift.html`,
identificar as features (heatmap do painel), checar se há queda de AUC / gap de calibração para
distinguir *covariate* de *concept drift* e aplicar a política de resposta (abaixo).

### ModelAUCBelowSLO
`credit_model_auc < 0.72`. **Ação:** confirmar o rótulo da janela (atraso de rótulo), retreinar com
janelas recentes (`retrain --with-windows`) e submeter ao gate (AUC >= `MIN_AUC`).

### ContractViolation
`credit_contract_status == 0`. **Ação:** o lote já está em `data/quarantine/<run_id>/`; ler
`violations.json`, contatar o produtor, reenviar lote corrigido. Nenhuma linha foi ingerida.

### PipelineTaskFailed
`credit_task_status == 0`. **Ação:** `logs/pipeline.jsonl` (`task_fim` com `status=falha`, `run_id`),
corrigir a causa e reexecutar a task; o alerta some quando um run seguinte empurra status 1.

### PipelineStale
`time() - credit_last_success_timestamp > 7d`. **Ação:** verificar o agendamento/gatilho do pipeline.

## Política de resposta ao drift (`configs/monitoring.yaml`)

| Severidade | Condição | Ação |
|-----------|----------|------|
| OK | PSI < 0,10 em todas e queda de AUC < 0,02 | observar |
| ATENÇÃO | algum PSI entre 0,10 e 0,20, ou queda de AUC entre 0,02 e 0,05 | investigar e revisar thresholds |
| CRÍTICO | PSI >= 0,20 **e** queda de AUC >= 0,05, ou \|gap de calibração\| > 5 p.p. | retreinar com janela recente rotulada + revisão humana |

É uma política **proporcional** à velocidade da mudança e ao custo do erro — não "retreinar a cada
oscilação" (um PSI alto sem queda de desempenho é falso alarme provável).

## Fluxo de incidente (exemplo: `DataDriftCritical`)

1. **Detecção** — o alerta dispara no Prometheus (e `logs/alerts.jsonl` registra `data_drift`).
2. **Triagem** — painel Grafana + `reports/index.html`: quais features, quais janelas, severidade da
   política; M1 (controle) deve estar limpa (descarta falso positivo de pipeline).
3. **Mitigação** — se há queda de AUC/gap de calibração, `retrain --with-windows M4 M5` (mesmo gate e
   promoção); caso contrário, apenas observar e revisar thresholds.
4. **Resolução** — nova versão `champion` no Model Registry; reexecutar `monitor` e confirmar AUC e
   gap de volta ao SLO; alerta resolvido.
5. **Post-mortem** — registrar causa (ver `docs/CAUSAL_ANALYSIS.md`: qual mecanismo explica a
   degradação), decisão de retreino e ação preventiva (ex.: avaliar proxy da variável latente sob
   a minimização da LGPD).
