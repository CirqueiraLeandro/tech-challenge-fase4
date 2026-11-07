# Guia Técnico

Módulo a módulo, decisões de projeto e gotchas encontrados. Versões fixadas no `poetry.lock`:
Python 3.11, pandas 2.3.3, numpy 2.4.6, scikit-learn 1.9.1, XGBoost 3.2.0, Pandera 0.35.0,
Evidently 0.7.23, MLflow 3.18.0, prometheus-client 0.26.0, pydantic 2.14.3.

## 1. Módulos

| Módulo | Responsabilidade |
|---|---|
| `src/config.py` | `Settings` (Pydantic Settings + `.env`), loaders YAML validados (`DriftScenario`, `MonitoringConfig`). `PSEUDONYMIZATION_KEY` obrigatória fora de `ENVIRONMENT=test` |
| `src/logger.py` | logging JSON (`logs/pipeline.jsonl`, rotação 10 MB × 5), `run_id` em `contextvar`, handler UTF-8. `print()` proibido em `src/` (teste varre o código) |
| `src/constants.py` | colunas, `MODEL_FEATURES`, `PII_COLUMNS`, `PROTECTED_ATTRIBUTES`, domínios — fonte única |
| `src/data/` | `loader`, `splitter` (60/40, seed 42, estratificado, dedup **antes** do split), `cleaning` (códigos não documentados → "outros"), `features` (4 derivadas), `synthetic` (mesmo schema do real: CI e demo) |
| `src/contracts/` | `schemas` (Pandera), `rules` (catálogo e regras suaves), `validator` (lazy, quarentena, relatório JSON/HTML), `badbatch` |
| `src/privacy/` | `pseudonymize` (HMAC-SHA256), `minimization`, `retention` (executável) |
| `src/models/` | `factory`, `artifacts` (JSON nativo), `train` (candidate → gate), `evaluate`, `explain` (reason codes), `registry` (MLflow + aliases), `retrain` |
| `src/drift/` | `simulation` (SCM), `stattests` (Strategy), `detector` (Evidently + próprio), `performance`, `policy`, `monitor` (orquestra uma janela), `causal` |
| `src/monitoring/` | `metrics` (Prometheus), `alerts` (Observer), `tracking` (MLflow), `report` (painel HTML) |
| `src/governance/fairness.py` | DI, Equal Opportunity, AUC por grupo |
| `src/pipeline/` | `tasks` (Template Method), `stages`, `cli`, `demo` |

Os orquestradores são curtos (nenhuma função > 20 linhas úteis); a CLI e as stages chamam as mesmas funções.

## 2. Decisões

**Pandera × Great Expectations (e Deequ/TFDV).** Pandera é pythônico, tipado, testável com pytest e leve
para o CI (schema-as-code, `DataFrameModel`). Great Expectations 1.x foi avaliado (Data Docs, 300+
expectations) e descartado pelo peso/complexidade para este escopo; Deequ é JVM/Spark e o TFDV acopla ao
ecossistema TFX — a comparação da Aula 7 favorece Pandera para um pipeline pandas pequeno.

**Evidently.** Versão fixada em **0.7.23 (API nova)**: `Report([DataDriftPreset(...)]).run(current_data=,
reference_data=)` + `snapshot.save_html()`. A API antiga vive em `evidently.legacy.*`. O método é
configurado **explicitamente** (`method="psi"`) porque o default escolhe o teste por tamanho de amostra
(Wasserstein para n > 1000), o que não bateria com o requisito do PDF nem com `stattests.py`. NannyML/CBPE e
MMD multivariado ficam como extensões.

**MLflow + Prometheus/Grafana.** Complementares: MLflow para o ciclo de vida (experimentos, modelos, drift
por janela), Prometheus/Grafana para a saúde operacional e os alertas. Backend SQLite (o de arquivos está
descontinuado no MLflow 3.x), aliases em vez de *stages* (deprecados). O pipeline é *batch*, logo
**Pushgateway**, com um grupo por task e um `CollectorRegistry` novo por execução (evita séries duplicadas).
Ambos opcionais em tempo de execução (fallback `observability_degraded`).

**Contrato atômico.** Linhas válidas de um lote reprovado não são "aproveitadas" (circuit breaker; um lote
parcialmente corrompido é sinal de problema no produtor). Um modo `--salvage` seria extensão documentada.

**Gate protege a produção.** O treino escreve em `models/candidate/`; só `promote_if_passes` copia para
`models/champion/` (com `shutil.copyfile`) e move o alias. Corrige o bug da Fase 3, em que o treino gravava
direto no diretório servido.

**Janelas de 1.200 linhas** (a especificação cita ~2.000): ver README, seção 6.

## 3. Gotchas encontrados (e divergências do esboço do contexto)

| # | Achado | Tratamento |
|---|---|---|
| 1 | **pandas 3.1 × Evidently 0.7.23:** o resolver aceita; o import falha (`infer_dtype_from_object`) | `pandas>=2.3.3,<3` |
| 2 | **Formato do Evidently 0.7:** `snapshot.dict()` devolve `{"metrics": [...], "tests": [...]}`, com itens `{id, metric_name, config, value}` — **não** existe `metric_id`. `ValueDrift` por coluna devolve o *score* do método (PSI) ou o p-value (KS); `DriftedColumnsCount` devolve `{count, share}` | `parse_evidently` lê `config["type"]`/`config["column"]`/`value` |
| 3 | **Pandera + `from __future__ import annotations`:** `Series` precisa ser import de módulo; import local gera `NameError` ao coletar os campos | imports no topo (`schemas.py` e teste de fumaça) |
| 4 | **Pandera `failure_cases`:** checks de DataFrame (`no_exact_duplicates`) vêm explodidos, uma linha por coluna; checks de schema (`column_in_schema`/`column_in_dataframe`) trazem o nome da coluna em `failure_case`, não em `column` | `_collapse_dataframe_checks` |
| 5 | **ucimlrepo (id=350)** devolve features como `X1..X23`, alvo `Y` e ID à parte (`data.ids`); nomes UCI originais não vêm | mapeamento por ordem em `download_data.py` (mantém `pay_0` e pula `pay_1`, como no original) |
| 6 | **MLflow 3.18:** `ModelVersion.version` é `int`; `log_model(name=...)` substitui `artifact_path`; com backend HTTP fora do ar os *retries* travam por minutos | comparação com `int`; `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` e timeout 3 s |
| 7 | **Regras suaves × schema drift:** acessavam coluna ausente (`KeyError`) | tolerantes a colunas ausentes (bug real, com teste) |
| 8 | **Evidently PSI × PSI próprio:** os binnings diferem (o próprio usa 10 quantis do reference; o do Evidently não foi inspecionado); `payment_ratio` em M6: 0,069 × 0,186 | cross-check por Jaccard (≥ 0,5 no teste; 0,75 em M6) |
| 9 | `metrics.json` sobrescrito pelo retreino | `metrics_<label>.json` |
| 10 | Windows: sem `make`; EPERM em `copystat` | tabela de comandos diretos; `shutil.copyfile`; `pathlib` e `encoding="utf-8"` em todo lugar |
| 11 | Console Windows é cp1252 e o log é JSON com `ensure_ascii=False` | `stderr.reconfigure(encoding="utf-8")` |
| 12 | Counter do `prometheus_client` expõe sufixo `_total` automaticamente | `Counter("credit_rows_processed")` → `credit_rows_processed_total` |
| 13 | `np.array_split` sobre DataFrame funciona, mas emite `FutureWarning` (`swapaxes`) e não é tipado | partição por posições (`np.array_split(np.arange(n))` + `iloc`); janelas comparadas linha a linha: idênticas |
| 14 | Logger configurado na importação: a suíte gravava em `logs/pipeline.jsonl` do projeto e o painel exibia execuções de teste | `conftest` aponta `LOGS_DIR` para um diretório temporário; o painel lê `current_log_file()` |
| 15 | `.callout strong{display:block}` (template HTML dos PDFs) quebra linha em negrito inline | negrito inline com `<b>` |

## 4. Reprodutibilidade e dados

Seed 42 em tudo (split, modelo, simulação: stream por `[seed, janela, mecanismo]`). Dados e modelos não são
versionados (`.gitignore`); `scripts/download_data.py` + `prepare` reproduzem tudo. O hash SHA-256 do
reference, o SHA do git e a versão do contrato vão como tags do run do MLflow.

## 5. Como estender

- Novo teste estatístico: subclasse de `StatTest` + entrada em `TESTS` (o detector só conhece a interface).
- Novo canal de alerta: subclasse de `AlertSink` + `dispatcher.register`.
- Nova task: subclasse de `Task` implementando `execute()` (métricas, logs e alerta de falha vêm de graça).
- Novo mecanismo causal: coluna em `MECHANISM_COLUMNS`, nome em `KNOWN_MECHANISMS` e bloco no YAML.
