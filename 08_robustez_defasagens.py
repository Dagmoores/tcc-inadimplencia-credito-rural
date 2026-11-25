# -*- coding: utf-8 -*-
"""Etapa 8 — Análise de robustez: SARIMAX-Agro com variáveis exógenas defasadas em 3, 6 e 12 meses,
comparado ao SARIMA estimado na mesma amostra (Tabela 7).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 08_robustez_defasagens.py
"""
import numpy as np
import pandas as pd

from comum import EXOG_AGRO, N, ajusta_sarimax, df, diebold_mariano, metricas, salvar_resultados, y

DEFASAGENS = [3, 6, 12]
HORIZONTES = [1, 3, 6]
R = {"robustez_defasagens": {}}
for L in DEFASAGENS:
    X = df[EXOG_AGRO].shift(L)        # valores das exógenas L meses antes
    ok = X.notna().all(axis=1)        # descarta os L primeiros meses (sem defasagem disponível)

    def prev_agro(ate, h):
        r = ajusta_sarimax(y.iloc[:ate][ok.iloc[:ate]], X.iloc[:ate][ok.iloc[:ate]])
        return r.forecast(h, exog=X.iloc[ate:ate + h]).values

    def prev_sarima(ate, h):          # SARIMA na mesma amostra, para comparação justa
        return ajusta_sarimax(y.iloc[:ate][ok.iloc[:ate]]).forecast(h).values

    real = y.iloc[N - 12:]
    out = {"teste_2025": {"SARIMAX-Agro defasado": metricas(real, pd.Series(prev_agro(N - 12, 12), index=real.index)),
                          "SARIMA": metricas(real, pd.Series(prev_sarima(N - 12, 12), index=real.index))}}
    for h in HORIZONTES:
        e_agro, e_sar = [], []
        for o in range(N - 36, N):
            if o + h - 1 >= N:
                continue
            e_agro.append(float(y.iloc[o + h - 1] - prev_agro(o, h)[-1]))
            e_sar.append(float(y.iloc[o + h - 1] - prev_sarima(o, h)[-1]))
        out[f"h{h}"] = {"MAE_agro": float(np.mean(np.abs(e_agro))), "MAE_sarima": float(np.mean(np.abs(e_sar))),
                        "dm": diebold_mariano(e_sar, e_agro, h)}  # estatística negativa: SARIMA mais preciso
    R["robustez_defasagens"][L] = out
    print(f"Defasagem {L}:", {k: (round(v["MAE_agro"], 3), round(v["MAE_sarima"], 3), round(v["dm"][1], 3))
                               for k, v in out.items() if k.startswith("h")})
salvar_resultados(R)
