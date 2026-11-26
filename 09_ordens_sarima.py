# -*- coding: utf-8 -*-
"""Etapa 9 — Sensibilidade à ordem do SARIMA: compara a especificação adotada, SARIMA(1,2,1)(1,1,1)12,
com alternativas de menor ordem de diferenciação ou menos parâmetros (AIC, MAPE em 2025 e MAE na
validação com origem móvel), justificando a escolha descrita na seção Metodologia.

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 09_ordens_sarima.py
"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from comum import N, diebold_mariano, metricas, salvar_resultados, y

ORDENS = {"(1,2,1)(1,1,1)": ((1, 2, 1), (1, 1, 1, 12)),   # especificação adotada
          "(1,1,1)(1,1,1)": ((1, 1, 1), (1, 1, 1, 12)),
          "(0,1,1)(0,1,1)": ((0, 1, 1), (0, 1, 1, 12)),
          "(1,1,0)(0,1,1)": ((1, 1, 0), (0, 1, 1, 12))}
HORIZONTES = [1, 3, 6]


def ajusta(serie, ordem, sazonal):
    return SARIMAX(serie, order=ordem, seasonal_order=sazonal,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


R = {"ordens_sarima": {}}
erros = {}
for nome, (o, s) in ORDENS.items():
    real = y.iloc[N - 12:]
    prev = ajusta(y.iloc[:N - 12], o, s).forecast(12)
    erros[nome] = {h: [float(y.iloc[a + h - 1] - ajusta(y.iloc[:a], o, s).forecast(h).iloc[-1])
                       for a in range(N - 36, N) if a + h - 1 < N] for h in HORIZONTES}
    R["ordens_sarima"][nome] = {"AIC": round(float(ajusta(y, o, s).aic), 3),
                                "MAPE_2025": metricas(real, prev)["MAPE"],
                                **{f"MAE_h{h}": float(np.mean(np.abs(erros[nome][h]))) for h in HORIZONTES}}
base = "(1,2,1)(1,1,1)"
for nome in ORDENS:
    if nome != base:  # estatística negativa: especificação adotada mais precisa
        R["ordens_sarima"][nome]["dm"] = {h: diebold_mariano(erros[base][h], erros[nome][h], h) for h in HORIZONTES}
salvar_resultados(R)
for nome, v in R["ordens_sarima"].items():
    print(f"SARIMA{nome}: AIC={v['AIC']:.1f} MAPE 2025={v['MAPE_2025']:.2f}% "
          + " ".join(f"MAE h={h}: {v[f'MAE_h{h}']:.3f}" for h in HORIZONTES))
