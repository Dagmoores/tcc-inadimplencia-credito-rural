# -*- coding: utf-8 -*-
"""Etapa 6 — Diagnóstico dos resíduos dos modelos sazonais: Ljung-Box e Jarque-Bera (Tabela 8, Figura 6).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 06_diagnostico_residuos.py
"""
from statsmodels.stats.diagnostic import acorr_ljungbox

from comum import EXOG_AGRO, EXOG_MACRO, ajusta_sarimax, df, salvar_resultados, y

R = {}
r_sar = ajusta_sarimax(y)
# Os testes descartam as observações iniciais de "aquecimento" do filtro de Kalman
# (loglikelihood_burn = 28 para d = 2, D = 1, s = 12), cujos resíduos não são informativos.
R["residuos"] = {}
for nome, res in [("SARIMA", r_sar), ("SARIMAX", ajusta_sarimax(y, df[EXOG_MACRO])),
                  ("SARIMAX-Agro", ajusta_sarimax(y, df[EXOG_AGRO]))]:
    lb = res.test_serial_correlation("ljungbox", lags=12)[0]
    jb = res.test_normality("jarquebera")[0]
    e = res.resid.iloc[res.loglikelihood_burn:]
    lb_ate2023 = acorr_ljungbox(e.loc[:"2023-12-01"], lags=[6, 12], return_df=True)
    R["residuos"][nome] = {"burn": int(res.loglikelihood_burn),
                           "lb6": [round(float(lb[0, 5]), 4), round(float(lb[1, 5]), 4)],
                           "lb12": [round(float(lb[0, 11]), 4), round(float(lb[1, 11]), 4)],
                           "jb": [round(float(jb[0]), 4), round(float(jb[1]), 4)],
                           "lb_ate2023": [round(float(x), 4) for x in lb_ate2023["lb_pvalue"]]}
salvar_resultados(R)
print(R["residuos"])
