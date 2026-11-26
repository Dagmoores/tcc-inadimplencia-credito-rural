# -*- coding: utf-8 -*-
"""Etapa 5 — Validação com origem móvel (janela expansiva; h = 1, 3 e 6) e teste de Diebold-Mariano (Tabelas 5 e 6, Figura 5).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 05_validacao_origem_movel_dm.py
"""
import numpy as np

from comum import MODELOS, N, diebold_mariano, prever, salvar_resultados, y

R = {}
HORIZONTES = [1, 3, 6]
origens = range(N - 36, N)  # previsões-alvo de jan/2023 a dez/2025
erros = {h: {m: [] for m in MODELOS} for h in HORIZONTES}
anos_alvo = {h: [] for h in HORIZONTES}  # ano de cada mês previsto, para separar o período de ruptura (2025)
for h in HORIZONTES:
    for o in origens:
        alvo_pos = o + h - 1  # posição prevista (h passos à frente a partir da origem o)
        if alvo_pos >= N:
            continue
        anos_alvo[h].append(int(y.index[alvo_pos].year))
        for m in MODELOS:
            p = prever(m, o, h)
            erros[h][m].append(float(y.iloc[alvo_pos] - p.iloc[-1]))
R["origem_movel"] = {h: {m: {"MAE": float(np.mean(np.abs(erros[h][m]))),
                             "RMSE": float(np.sqrt(np.mean(np.square(erros[h][m])))),
                             "n": len(erros[h][m])} for m in MODELOS} for h in HORIZONTES}

# Erros do SARIMA separados por período dos meses previstos: 2023–2024 e 2025 (ruptura).
# Erro = observado − previsto; viés médio positivo indica subestimação.
R["origem_movel_periodo"] = {}
for h in HORIZONTES:
    e, a = np.array(erros[h]["SARIMA"]), np.array(anos_alvo[h])
    R["origem_movel_periodo"][h] = {"MAE_2023_2024": float(np.mean(np.abs(e[a <= 2024]))),
                                    "MAE_2025": float(np.mean(np.abs(e[a == 2025]))),
                                    "vies_2025": float(np.mean(e[a == 2025]))}

R["dm"] = {}
for h in HORIZONTES:
    R["dm"][h] = {}
    for m in MODELOS:
        if m == "SARIMA":
            continue
        R["dm"][h][m] = diebold_mariano(erros[h]["SARIMA"], erros[h][m], h)
salvar_resultados(R)
for h, d in R["origem_movel"].items():
    print("h =", h, {m: round(v["MAE"], 3) for m, v in d.items()})
print("SARIMA por período:", R["origem_movel_periodo"])
