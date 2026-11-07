# Visão do Projeto (STAR)

## Situation — o problema de negócio

Uma fintech de concessão de crédito mantém um modelo de Credit Scoring em produção. Mudanças recentes na
economia (inflação, aperto de crédito, desemprego, novos perfis de clientes) levantam a suspeita de que o
modelo decide mal sem que ninguém perceba: o pipeline "roda verde", mas as decisões pioram — a falha
**silenciosa** é a mais perigosa (Sculley et al., 2015).

Risco concreto: aprovar crédito a quem vai inadimplir (perda financeira) e negar a bons pagadores (perda de
receita e risco de discriminação, vedada pelo art. 6º, IX da LGPD).

## Task — o que precisava ser entregue

A **camada de sustentação e confiabilidade** em volta de um classificador binário de inadimplência:
contratos de dados → scoring → detecção de drift → observabilidade → diagnóstico causal → prescrição
(política de resposta) → governança. Requisitos do PDF: pipeline de validação, scripts de simulação e
detecção de drift, dashboards/relatórios HTML, governança no README, commits semânticos e vídeo STAR; com
≥ 3 regras rígidas, separação referência × produção, relatório PSI/KS e plano LGPD.

## Action — o que foi construído

1. **Contrato Pandera** (8 regras rígidas, 3 suaves) com lote atômico, quarentena e exit code 2.
2. **Modelo XGBoost** simples, com gate de promoção, *registry* com aliases e *reason codes* (LGPD art. 20).
3. **Simulação causal** (SCM) de 6 janelas com 4 mecanismos desligáveis; M1 é o controle sem alerta.
4. **Detecção**: PSI/KS/Chi²/Jensen-Shannon próprios + relatório Evidently, com *cross-check*; concept drift
   por AUC e gap de calibração; política de resposta proporcional; retreino contínuo.
5. **Observabilidade**: MLflow (3 experimentos), Pushgateway/Prometheus/Grafana (11 painéis, 5 alertas),
   logs JSON sem PII e painel HTML offline.
6. **Causalidade**: atribuição intervencional (verdade de campo) × observacional.
7. **Governança**: pseudonimização, minimização, auditoria de vieses, retenção executável, RIPD/ROPA.

## Result — o que se mostrou

- O contrato barra o lote corrompido (301 linhas, 10 violações cobrindo H1–H7, exit 2) e aprova reference e pool.
- Baseline: AUC 0,7836, KS 0,4355 (Gini 0,567).
- Drift detectado a partir de M2 (`limit_bal`, `utilization`), até 4 de 25 features (16%) em M6, sem falso
  positivo no controle M1; PSI ≥ 0,20 em M3–M6 dispara `DataDriftCritical`.
- A degradação de desempenho é sutil e ruidosa no AUC (M6: −0,033) mas nítida na calibração (gap de −4,3 p.p.).
- A análise causal mostra que o desemprego **latente** explica ~94% do gap, enquanto a inflação domina a mudança
  em P(X); o diagnóstico observacional falha no desemprego (como previsto).
- O retreino com M4+M5 recupera o AUC em M6 de 0,7501 para 0,7702 (e o Brier de 0,1733 para 0,1685).
- Entrega: 159 testes, 98,67% de cobertura, CI com 4 jobs; ruff e mypy sem erros.

Detalhes e evidências no [README](../README.md).
