# Relatório de Drift — metodologia e resultados

Gerado a partir de `reports/drift/drift_summary.csv|json` (último run completo, champion baseline,
AUC de referência no teste = 0,7836). Relatórios Evidently por janela: `reports/drift/Mk_data_drift.html`
(commitados M1 e M6; M2–M5 são regeneráveis com `monitor`).

## 1. Metodologia

- **Separação:** `data/reference/` (18.000, treino/teste) × `data/production/` (pool de 12.000 → janelas
  M1..M6 de 1.200 linhas, disjuntas). Split único, seed 42, estratificado, **antes** de qualquer estatística.
- **Simulação:** SCM com 4 mecanismos independentes (inflação, aperto, desemprego latente, perfil jovem) e
  intensidade `t = k/5`. **M1 (t=0) é o controle** e não pode gerar alerta.
- **Contrato primeiro:** cada janela passa nas regras rígidas antes do scoring (lote reprovado não é pontuado).
- **Magnitude × significância:** PSI (bins por quantis do reference, 10 bins; categóricas por valor) e
  Jensen-Shannon medem *quanto*; KS (numéricas) e Chi² (categóricas) dizem se *não é acaso*. "Drifted" =
  PSI ≥ 0,10 **e** p < 0,05 (Bonferroni configurável). Estatísticas do reference calculadas uma vez por run.
- **Evidently 0.7.23** com `method="psi"` explícito (o default seria Wasserstein para n > 1000) e limiar 0,10.
- **Concept drift:** AUC, KS de separação, Brier, taxa prevista × observada (gap de calibração) e PSI do
  score (prediction drift, proxy sem rótulo).
- **Política:** OK / ATENÇÃO / CRÍTICO (`configs/monitoring.yaml`).

## 2. PSI por feature × janela (negrito = PSI ≥ 0,10)

| Feature | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|
| `utilization` | 0.007 | **0.175** | **0.276** | **0.429** | **0.571** | **0.620** |
| `pay_2` | 0.026 | 0.048 | **0.116** | **0.197** | **0.340** | **0.331** |
| `limit_bal` | 0.006 | **0.149** | **0.281** | **0.193** | **0.294** | **0.306** |
| `payment_ratio` | 0.012 | 0.025 | 0.038 | **0.101** | **0.111** | **0.186** |
| `months_delayed` | 0.003 | 0.007 | 0.023 | 0.059 | 0.092 | 0.083 |
| `max_delay` | 0.015 | 0.022 | 0.027 | 0.039 | 0.065 | 0.069 |
| `bill_amt2` | 0.008 | 0.027 | 0.029 | 0.042 | 0.044 | 0.062 |
| `pay_0` | 0.019 | 0.017 | 0.044 | 0.043 | 0.088 | 0.061 |
| `pay_amt2` | 0.005 | 0.004 | 0.072 | 0.078 | 0.055 | 0.060 |
| `bill_amt1` | 0.005 | 0.014 | 0.018 | 0.055 | 0.043 | 0.057 |
| `age` | 0.011 | 0.011 | 0.009 | 0.037 | 0.047 | 0.053 |
| `bill_amt6` | 0.006 | 0.011 | 0.007 | 0.031 | 0.032 | 0.048 |
| `pay_amt6` | 0.007 | 0.008 | 0.006 | 0.003 | 0.006 | 0.046 |
| `bill_amt3` | 0.005 | 0.006 | 0.018 | 0.033 | 0.037 | 0.046 |
| `bill_amt5` | 0.007 | 0.009 | 0.010 | 0.029 | 0.028 | 0.043 |
| `bill_amt4` | 0.005 | 0.013 | 0.013 | 0.027 | 0.041 | 0.035 |
| `pay_amt1` | 0.015 | 0.008 | 0.007 | 0.025 | 0.025 | 0.020 |
| `pay_amt3` | 0.010 | 0.011 | 0.007 | 0.008 | 0.008 | 0.016 |
| `pay_4` | 0.011 | 0.011 | 0.006 | 0.010 | 0.002 | 0.016 |
| `pay_3` | 0.018 | 0.016 | 0.010 | 0.008 | 0.010 | 0.012 |
| `pay_5` | 0.005 | 0.009 | 0.010 | 0.009 | 0.005 | 0.011 |
| `pay_6` | 0.009 | 0.020 | 0.006 | 0.012 | 0.006 | 0.008 |
| `pay_amt5` | 0.008 | 0.004 | 0.009 | 0.006 | 0.012 | 0.008 |
| `pay_amt4` | 0.005 | 0.008 | 0.012 | 0.006 | 0.011 | 0.004 |
| `education` | 0.001 | 0.004 | 0.002 | 0.001 | 0.002 | 0.001 |

Controles negativos (`education`, `pay_3..pay_6`) permanecem < 0,02 em todas as janelas, inclusive M6.

## 3. Desempenho e política por janela

| Janela | Linhas | AUC | KS | Brier | Taxa prevista | Taxa observada | Gap | PSI do score | Severidade | Jaccard (próprio × Evidently) |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 1200 | 0.7746 | 0.4211 | 0.1394 | 0.2179 | 0.2350 | -0.0171 | 0.007 | OK | 1.00 |
| M2 | 1200 | 0.7718 | 0.4585 | 0.1460 | 0.2378 | 0.2433 | -0.0056 | 0.020 | ATENCAO | 0.00 |
| M3 | 1200 | 0.7868 | 0.4559 | 0.1412 | 0.2600 | 0.2408 | +0.0192 | 0.049 | ATENCAO | 0.67 |
| M4 | 1200 | 0.7443 | 0.3780 | 0.1570 | 0.2459 | 0.2483 | -0.0024 | 0.047 | ATENCAO | 0.75 |
| M5 | 1200 | 0.7679 | 0.4207 | 0.1646 | 0.2659 | 0.2917 | -0.0258 | 0.075 | ATENCAO | 0.75 |
| M6 | 1200 | 0.7501 | 0.3842 | 0.1733 | 0.2515 | 0.2942 | -0.0426 | 0.054 | ATENCAO | 0.75 |

## 4. Leitura

- **Sem falso positivo no controle:** M1 tem 0 features em drift nos testes próprios e 0 no Evidently, severidade OK.
- **Quais variáveis degradaram:** `utilization` (0,620), `pay_2` (0,331), `limit_bal` (0,306) e `payment_ratio`
  (0,186) em M6. `limit_bal` e `utilization` já em M2 — o aperto de crédito e a inflação aparecem primeiro; `pay_2`
  só cruza 0,10 em M3 (0,116) e 0,20 em M5.
- **A política não chegou a CRÍTICO** (queda de AUC máxima 0,039 em M4 < 0,05; |gap| máximo 0,043 em M6 < 0,05).
  Os parâmetros do cenário foram mantidos como especificados. O alerta de PSI ≥ 0,20 dispara de M3 a M6.
- **O AUC é ruidoso** em janelas de 1.200 linhas (M3 acima do baseline; M4 pior que M5 e M6). A calibração é o sinal
  mais estável do concept drift: o gap vai de -0.0171 (M1) a -0.0426 (M6) e o Brier de
  0.1394 a 0.1733.
- **Prediction drift:** o PSI do score sobe de 0.007 (M1) para 0.075 (M5) / 0.054 (M6) — abaixo do
  limiar moderado: o modelo "não percebe" a maior parte da mudança, o que é típico de falha silenciosa.
- **Cross-check:** a divergência entre os métodos se concentra onde o PSI está perto do limiar (M2: próprio sinaliza
  `limit_bal` e `utilization` — PSI 0,149 e 0,175 — e o Evidently não; M4–M6: `payment_ratio`).

## 5. Retreino (resposta)

`retrain --with-windows M4 M5`, hold-out M6: AUC 0,7501 → 0,7702; Brier 0,1733 → 0,1685; gap −0,0426 → −0,0374;
AUC no teste do reference 0,7845 (ver `reports/retrain/retrain_summary.json`).
