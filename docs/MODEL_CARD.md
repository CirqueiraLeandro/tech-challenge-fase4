# Model Card — Credit Scoring (XGBoost)

> **Uso restrito:** modelo didático, treinado em dados de Taiwan de 2005 e com tempo simulado.
> **Não usar em decisão de crédito real.**

## Detalhes do modelo

| | |
|---|---|
| Nome / registro | `credit-scoring` (MLflow Model Registry, alias `champion`) |
| Tipo | `XGBClassifier` — 300 árvores, `max_depth=4`, `learning_rate=0,05`, `eval_metric=auc`, `random_state=42` |
| Baseline de comparação | Regressão logística (`StandardScaler`, `class_weight="balanced"`) — via Factory |
| Tarefa | classificação binária: P(default no próximo mês \| X) |
| Entrada | 25 features: 21 originais (`limit_bal`, `education`, `age`, `pay_0`, `pay_2..6`, `bill_amt1..6`, `pay_amt1..6`) + 4 derivadas (`utilization`, `payment_ratio`, `max_delay`, `months_delayed`) |
| **Não usa** | `client_id`/`client_hash`, **`sex`, `marriage`** (minimização e não discriminação) |
| Formato | JSON nativo do XGBoost + `metadata.json` (sem pickle); MLflow com *signature* e *input_example* |
| Versões | xgboost 3.2.0, scikit-learn 1.9.1, pandas 2.3.3 |
| Contrato de dados | v1.0.0 |

## Dados

*Default of Credit Card Clients* (UCI id=350; Yeh & Lien, 2009): 30.000 clientes, ~22% de inadimplência.
Split único e estratificado (seed 42): **reference 60% (18.000)** → treino 80% / teste 20% (3.600);
**pool de produção 40% (12.000)** → janelas simuladas. Códigos não documentados (`education` 0, 5, 6;
`marriage` 0) são remapeados para "outros" na limpeza.

## Métricas (teste do reference, 3.600 clientes, threshold 0,5 — `models/metrics_train.json`)

| AUC | Gini | KS | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| 0,7836 | 0,5671 | 0,4355 | 0,4947 | 0,7021 | 0,3819 | 0,1321 |

Confusão: TN 2675 · FP 129 · FN 492 · TP 304. O recall (0,38) é baixo no threshold 0,5: a alternativa por custo
(threshold menor) deve ser definida pela política de risco. **Gate de promoção:** AUC ≥ 0,75.
Com `sex`/`marriage` o AUC seria 0,7828 — a exclusão não custa desempenho.

## Desempenho sob drift (janelas simuladas, 1.200 linhas)

| Janela | AUC | Brier | Gap de calibração | Severidade |
|---|---|---|---|---|
| M1 (controle) | 0,7746 | 0,1394 | −0,0171 | OK |
| M3 | 0,7868 | 0,1412 | +0,0192 | ATENÇÃO |
| M6 | 0,7501 | 0,1733 | −0,0426 | ATENÇÃO |

Após o retreino com M4+M5, em M6: AUC 0,7702 · Brier 0,1685 · gap −0,0374.

## Fairness (regra dos 80%, aprovado = score < 0,5)

| Conjunto | Atributo | Disparate Impact | Dif. de oportunidade |
|---|---|---|---|
| reference (teste) | sexo | 0,980 | 0,016 |
| reference (teste) | faixa etária | 0,877 | 0,042 |
| M6 | sexo | 0,998 | 0,017 |
| M6 | faixa etária | 0,864 | 0,043 |

Sem violação da regra dos 80%, mas **menores de 25 anos** têm a menor taxa de aprovação (0,802; 0,770 em M6).
Em M6 o AUC por sexo diverge (F 0,765 × M 0,725). Monitorar continuamente.

## Explicabilidade

*Reason codes*: os 3 principais fatores que elevam o risco de cada cliente, via contribuições SHAP nativas do
XGBoost (`pred_contribs`); teste garante soma das contribuições + bias = margem prevista (LGPD art. 20).

## Limitações e uso responsável

- Dataset de Taiwan (2005), sem renda; o **tempo é simulado** e o rótulo é revelado sem atraso.
- `education` e `age` podem atuar como proxies socioeconômicos/geracionais.
- Performance degrada com drift; o AUC sozinho é sinal ruidoso — acompanhar a **calibração** e o PSI.
- Concept drift por variável latente (ex.: desemprego) não é visível em P(X); exige rótulos e, possivelmente, um
  proxy sob avaliação de minimização.
- Decisões contestadas ou na fronteira devem passar por **revisão humana** (art. 20).
- Retreino sempre via mesmo gate e promoção; versões aposentadas guardadas por 5 anos para auditoria.
