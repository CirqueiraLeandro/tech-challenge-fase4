# Data Contract — Credit Scoring

Contrato formal entre **produtor** (ingestão/simulação de lotes) e **consumidor** (modelo de
Credit Scoring). Implementado em Pandera (`src/contracts/schemas.py`), com o catálogo de regras em
`src/contracts/rules.py` e o validador em `src/contracts/validator.py`.

- **Versão do contrato:** `1.0.0` (`CONTRACT_VERSION`, gravada como tag nos runs do MLflow).
- **Escopo:** lote já **ingerido** — o `client_id` cru é substituído por `client_hash`
  (HMAC-SHA256) antes da validação.
- **Semântica:** lote **atômico** (all-or-nothing). Falha em qualquer regra rígida bloqueia a
  ingestão inteira (circuit breaker); nada é escrito em `data/production/`.

## Regras

| Id | Regra | Severidade | Dono | Efeito |
|----|-------|-----------|------|--------|
| H1 | Schema: colunas obrigatórias, tipos corretos, sem colunas extras (`strict=True`, `coerce=False`) | rígida | Engenharia de Dados | bloqueia |
| H2 | Sem nulos em nenhuma feature nem no identificador | rígida | Engenharia de Dados | bloqueia |
| H3 | `18 <= age <= 100` | rígida | Risco de Crédito | bloqueia |
| H4 | `0 < limit_bal <= 1.000.000` | rígida | Risco de Crédito | bloqueia |
| H5 | Domínios: `sex ∈ {1,2}`; `education ∈ 0..6`; `marriage ∈ 0..3`; `pay_x ∈ [-2, 8]` | rígida | Risco de Crédito | bloqueia |
| H6 | Unicidade: `client_hash` sem duplicatas e nenhuma linha duplicada exata (SHA-256 da linha) | rígida | Engenharia de Dados | bloqueia |
| H7 | `pay_amt_x >= 0` | rígida | Risco de Crédito | bloqueia |
| H8 | Volume mínimo do lote (`>= MIN_BATCH_ROWS`, padrão 100) | rígida | Engenharia de Dados | bloqueia |
| S1 | `bill_amt_x` fora de `[p0,5; p99,5]` do reference em > 5% das linhas | suave | Ciência de Dados | aviso + log |
| S2 | `education`/`marriage` não documentados em > 2% das linhas | suave | Ciência de Dados | aviso + log |
| S3 | Média de `limit_bal` desvia > 30% da média de referência | suave | Risco de Crédito | aviso + log |

Observações de implementação:

- Valores monetários são `float`; um `int` é alargado para `float` sem perda antes da validação
  (texto numa coluna numérica continua falhando em H1).
- As regras suaves são tolerantes a colunas ausentes (o schema drift é reportado por H1, sem que a
  regra suave quebre).
- Códigos de `education` (0, 5, 6) e `marriage` (0) existem no dataset UCI sem documentação; o
  contrato os aceita (H5) e `src/data/cleaning.py` os remapeia para "outros" — S2 monitora o peso
  deles.

## Violações e quarentena

`validate_batch` roda com `lazy=True` e devolve **todas** as violações (agregadas por regra, coluna
e check), com no máximo 5 exemplos por violação: índice da linha, valor inválido e `client_hash`
**curto** (12 caracteres) — nunca o ID cru. Em falha rígida:

1. `reports/contracts/<lote>_violations.json|html` (relatório legível);
2. `data/quarantine/<run_id>/batch.csv` + `violations.json` (retenção de 30 dias);
3. métricas `credit_contract_status`/`credit_rows_rejected_total`, alerta `contract_violation` e
   exit code **2** na CLI.

## Política de evolução de schema

| Mudança | Compatibilidade | Procedimento |
|---------|----------------|--------------|
| Nova coluna opcional no produtor | *forward-compatible* para o consumidor antigo, mas **bloqueada** por `strict=True` | Atualizar o schema e subir `CONTRACT_VERSION` (minor) antes de o produtor publicar |
| Remover/renomear coluna | **quebra** (*backward-incompatible*) | Versão major, período de dupla publicação e reentrega do modelo |
| Afrouxar um domínio (ex.: `pay_x` até 9) | compatível para o consumidor | Versão minor; reavaliar drift e o impacto no modelo |
| Apertar um domínio / novo limite | pode rejeitar lotes antes válidos | Versão major; comunicar ao produtor; rodar o lote histórico no contrato novo |

Toda mudança exige: PR com o teste da regra afetada (`tests/test_contracts.py`), atualização desta
tabela e do README, e nova versão em `CONTRACT_VERSION`.
