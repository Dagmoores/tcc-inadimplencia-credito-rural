# -*- coding: utf-8 -*-
"""Etapa 1 — Coleta das séries no SGS/BCB, integração mensal, estatística descritiva e teste de sazonalidade
(dados das Figuras 1 e 2; estatísticas citadas no texto).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 01_coleta_dados.py
"""
from scipy import stats

from comum import SERIES, df, salvar_resultados, y

R = {}
R["descritiva"] = df[list(SERIES.values())].describe().round(4).to_dict()
R["correlacao"] = df[list(SERIES.values())].corr().round(4).to_dict()
# Sazonalidade: variações mensais da inadimplência agrupadas por mês do ano (teste de Kruskal-Wallis),
# com a amostra completa e com dados apenas até 2023 (antes da ruptura de 2024–2025)
R["sazonalidade"] = {}
for nome, serie in {"completa": y.diff().dropna(), "ate_2023": y.diff().dropna()[:"2023"]}.items():
    grupos = [serie[serie.index.month == m].values for m in range(1, 13)]
    R["sazonalidade"][nome] = {"kruskal_p": float(stats.kruskal(*grupos).pvalue),
                               "mediana_por_mes": serie.groupby(serie.index.month).median().round(3).to_dict()}
salvar_resultados(R)
print(df[list(SERIES.values())].describe().round(2).T)
