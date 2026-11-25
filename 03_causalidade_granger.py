# -*- coding: utf-8 -*-
"""Etapa 3 — Testes de causalidade de Granger das variáveis explicativas para a inadimplência (Tabela 3).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 03_causalidade_granger.py
"""
import contextlib
import io

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from comum import df, salvar_resultados, y

R = {}
# Séries transformadas para estacionariedade: alvo em 2a diferença, demais em 1a diferença
gr = pd.DataFrame({"alvo": y.diff().diff()})
for c in ["selic", "ipca", "log_cambio", "log_icbr_agro"]:
    gr[c] = df[c].diff()
gr = gr.dropna()
R["granger"] = {}
for c in ["selic", "ipca", "log_cambio", "log_icbr_agro"]:
    with contextlib.redirect_stdout(io.StringIO()):
        res = grangercausalitytests(gr[["alvo", c]], maxlag=6)
    R["granger"][c] = {lag: round(res[lag][0]["ssr_ftest"][1], 4) for lag in (1, 3, 6)}
salvar_resultados(R)
print(R["granger"])
