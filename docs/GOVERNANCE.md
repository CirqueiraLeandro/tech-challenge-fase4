# Governança e LGPD — RIPD, ROPA e Fairness

Complementa a seção 9 do [README](../README.md) (inventário, base legal, Privacy by Design, vieses e retenção).
Tudo aqui está **implementado e testado** (`src/privacy/`, `src/governance/`, `tests/test_privacy.py`,
`tests/test_causal_fairness.py`).

## 1. ROPA — Registro das Operações de Tratamento (art. 37)

| Campo | Descrição |
|---|---|
| Controlador | Fintech de concessão de crédito (projeto acadêmico: dados públicos da UCI) |
| Operador / sistemas | Pipeline de scoring (este repositório); MLflow, Prometheus, Grafana (sem dados pessoais) |
| Finalidade | Análise de risco de inadimplência para **proteção do crédito** |
| Base legal | Art. 7º, X (proteção do crédito); complementar art. 7º, V (execução de contrato/procedimentos preliminares) |
| Categorias de titulares | Clientes titulares de cartão/crédito |
| Dados tratados | identificador (pseudonimizado), `age`, `education`, dados financeiros (limite, faturas, pagamentos, atrasos); `sex`/`marriage` apenas em área segregada para auditoria |
| Dados sensíveis (art. 5º, II) | Nenhum tratado |
| Compartilhamento | Nenhum; webhook de alertas e Pushgateway desligados por padrão e sem dados pessoais |
| Transferência internacional | Não |
| Retenção | Ver tabela de retenção (README 9.5) |
| Segurança | HMAC-SHA256 com chave em `.env`; dados fora do git; quarentena e logs sem PII; retenção executável |
| Decisão automatizada | Sim (score) — com *reason codes* e fluxo de revisão humana (art. 20) |

## 2. RIPD resumido (art. 38)

**Descrição do tratamento:** classificação automatizada de risco de inadimplência a partir de dados
cadastrais e comportamentais, com monitoramento contínuo de qualidade, drift e vieses.

**Necessidade e proporcionalidade:** os dados financeiros são necessários à finalidade; `sex` e `marriage`
foram **excluídos** das features porque o desempenho não depende deles (AUC 0,7836 sem × 0,7828 com);
`age` e `education` são usados com monitoramento específico (proxy).

| Risco | Probabilidade | Impacto | Medida | Residual |
|---|---|---|---|---|
| Reidentificação via pseudônimo | baixa | alto | HMAC com chave secreta; ID cru fora da ingestão; relatórios só com hash curto (12 caracteres) | baixo |
| Discriminação (direta ou por proxy) | média | alto | exclusão de atributos protegidos; auditoria de DI/EO por sexo e idade (referência e M6); revisão humana de casos de fronteira | médio |
| Vazamento de dados | baixa | alto | `.gitignore` de dados e `.env`; retenção; logs sem PII (teste automatizado) | baixo |
| Decisão errada por drift | média | alto | monitoramento + política de resposta + retreino com gate; alertas | médio |
| Falta de transparência | média | médio | Model Card, *reason codes*, ROPA | baixo |

**Risco residual aceito** condicionado a: revisão humana em decisões automatizadas contestadas; monitoramento
contínuo do Disparate Impact como métrica operacional; reavaliação do RIPD a cada retreino ou mudança de feature.

## 3. Direitos do titular (art. 18) — como seriam atendidos

| Direito | Atendimento no sistema |
|---|---|
| Confirmação e acesso | consulta por `client_hash` (o controlador, detentor da chave, mapeia `client_id → client_hash`) |
| Correção | atualização do registro de origem e reprocessamento na próxima ingestão |
| Eliminação | remoção por `client_hash` nos dados operacionais (`data/`, incluindo `data/quarantine/`); `reports/` traz apenas agregados e hashes curtos de exemplos de violação; prazos legais de guarda são respeitados (README 9.5) |
| Portabilidade | exportação dos dados do titular a partir do reference/pool filtrado por `client_hash` |
| Informação sobre decisões (art. 20) | *reason codes* (3 principais fatores de risco) + revisão por analista humano |

## 4. Fairness — metodologia

`src/governance/fairness.py` (pandas, sem fairlearn): aprovado = score < threshold (0,5). Por grupo (sexo;
faixas <25, 25–40, 40–60, >60; grupos com n < 30 omitidos): taxa de aprovação, **taxa de aprovação de bons
pagadores** (Equal Opportunity), taxa de default e AUC. **Disparate Impact** = menor/maior taxa de aprovação
(regra dos 80%). Resultados reais no README 9.4 e em `reports/fairness/fairness_report.json`.

Limitações: sem renda e sem raça/cor no dataset, não é possível auditar discriminação indireta por essas
dimensões; `education` pode atuar como proxy socioeconômico; a faixa >60 tem poucos clientes (35 no teste).
