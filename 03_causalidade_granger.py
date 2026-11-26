# -*- coding: utf-8 -*-
"""Etapa 3 — Testes de causalidade de Granger das variáveis explicativas para a inadimplência, com 1, 3, 6, 9 e 12 defasagens (Tabela 3).

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
# Séries na ordem de integração indicada pelo ADF (Tabela 2): alvo em 2a diferença;
# Selic, câmbio e IC-Br em 1a diferença; IPCA (variação mensal, já estacionária) em nível
gr = pd.DataFrame({"alvo": y.diff().diff()})
for c in ["selic", "ipca", "log_cambio", "log_icbr_agro"]:
    gr[c] = df[c] if c == "ipca" else df[c].diff()
gr = gr.dropna()
R["granger"] = {}
for c in ["selic", "ipca", "log_cambio", "log_icbr_agro"]:
    with contextlib.redirect_stdout(io.StringIO()):
        res = grangercausalitytests(gr[["alvo", c]], maxlag=12)
    R["granger"][c] = {lag: round(res[lag][0]["ssr_ftest"][1], 4) for lag in (1, 3, 6, 9, 12)}
salvar_resultados(R)
print(R["granger"])
