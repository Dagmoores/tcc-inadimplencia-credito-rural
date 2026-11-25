# -*- coding: utf-8 -*-
"""Etapa 2 — Teste aumentado de Dickey-Fuller em nível, 1ª e 2ª diferenças (Tabela 2, Figura 3).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 02_estacionariedade_adf.py
"""
from comum import adf, df, salvar_resultados

R = {"adf": {}}
for c in ["inad_rural_pf_mercado", "selic", "ipca", "log_cambio", "log_icbr_agro"]:
    R["adf"][c] = {"nivel": adf(df[c]), "dif1": adf(df[c].diff()), "dif2": adf(df[c].diff().diff())}
salvar_resultados(R)
for c, v in R["adf"].items():
    print(c, v)
