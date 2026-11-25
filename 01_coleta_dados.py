# -*- coding: utf-8 -*-
"""Etapa 1 — Coleta das séries no SGS/BCB, integração mensal e estatística descritiva (Tabela 1, Figuras 1 e 2).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 01_coleta_dados.py
"""
from comum import SERIES, df, salvar_resultados

R = {}
R["descritiva"] = df[list(SERIES.values())].describe().round(4).to_dict()
R["correlacao"] = df[list(SERIES.values())].corr().round(4).to_dict()
salvar_resultados(R)
print(df[list(SERIES.values())].describe().round(2).T)
