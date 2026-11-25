# -*- coding: utf-8 -*-
"""Etapa 4 — Estimação dos modelos (Ingênuo, ARIMA, SARIMA, SARIMAX, SARIMAX-Agro, VAR, VAR-Agro) e teste fora da amostra em 2025 (Tabela 4, Figura 4).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 04_modelos_teste_2025.py
"""
from comum import EXOG_AGRO, EXOG_MACRO, MODELOS, N, ajusta_sarimax, df, metricas, prever, salvar_resultados, y

R = {}
ate_2025 = N - 12
real_2025 = y.iloc[ate_2025:]
R["teste_2025"], prev_2025, R["lags_var"] = {}, {}, {}
for m in MODELOS:
    p = prever(m, ate_2025, 12)
    prev_2025[m] = p
    R["teste_2025"][m] = metricas(real_2025, p)
    if m.startswith("VAR"):
        R["lags_var"][m] = int(prever.ultimo_lag)
R["prev_2025"] = {m: [round(v, 4) for v in prev_2025[m].values] for m in MODELOS}
R["real_2025"] = [round(v, 4) for v in real_2025.values]

# coeficientes das exógenas (amostra completa)
coef = {}
for nome, cols in [("SARIMAX", EXOG_MACRO), ("SARIMAX-Agro", EXOG_AGRO)]:
    r = ajusta_sarimax(y, df[cols])
    coef[nome] = {c: {"coef": round(float(r.params[c]), 4), "p": round(float(r.pvalues[c]), 4)} for c in cols}
    coef[nome]["AIC"] = round(float(r.aic), 3)
r_sar = ajusta_sarimax(y)
coef["SARIMA"] = {"AIC": round(float(r_sar.aic), 3)}
R["coeficientes"] = coef
salvar_resultados(R)
for m, v in R["teste_2025"].items():
    print(f"{m:13s} RMSE={v['RMSE']:.4f} MAE={v['MAE']:.4f} MAPE={v['MAPE']:.2f}%")
