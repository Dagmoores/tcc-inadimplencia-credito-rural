# -*- coding: utf-8 -*-
"""Etapa 7 — Geração das Figuras 1 a 6 do TCC (requer as etapas 1 a 6 executadas).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 07_figuras.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from comum import SAIDA, ajusta_sarimax, df, ler_resultados, y

R = ler_resultados()
MODELOS = list(R["teste_2025"].keys())
HORIZONTES = [1, 3, 6]
R["origem_movel"] = {int(h): v for h, v in R["origem_movel"].items()}
idx_2025 = df.index[-12:]
prev_2025 = {m: pd.Series(v, index=idx_2025) for m, v in R["prev_2025"].items()}
r_sar = ajusta_sarimax(y)

# ------------------------------------------------- Figuras
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
AZUL, LARANJA, CINZA = "#1f4e79", "#c55a11", "#7f7f7f"


def formata_datas(ax, meses=12, anos=1):
    if meses < 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, meses)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(anos))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(df.index, df["inad_rural_pf_mercado"], color=AZUL, label="Taxas de mercado (SGS 21146)")
ax.plot(df.index, df["inad_rural_pf_regulada"], color=LARANJA, label="Taxas reguladas (SGS 21147)")
ax.set_ylabel("Inadimplência (%)"); ax.set_xlabel("Período"); ax.legend(frameon=False); formata_datas(ax)
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig1_series.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True)
for a, (c, t) in zip(axs.flat, [("selic", "Meta Selic (% a.a.)"), ("ipca", "IPCA (% ao mês)"),
                                 ("cambio", "Câmbio (R$/US$)"), ("icbr_agro", "IC-Br Agropecuária (índice)")]):
    a.plot(df.index, df[c], color=AZUL); a.set_title(t, fontsize=10); formata_datas(a, anos=2)
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig2_explicativas.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].plot(y.diff(), color=AZUL); axs[0].set_title("Primeira diferença", fontsize=10)
axs[1].plot(y.diff().diff(), color=AZUL); axs[1].set_title("Segunda diferença", fontsize=10)
for a in axs: a.set_ylabel("Pontos percentuais"); formata_datas(a, anos=2)
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig3_diferencas.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(3, 2, figsize=(9, 8), sharex=True, sharey=True)
hist = y.loc["2023-01-01":]
for a, m in zip(axs.flat, ["ARIMA", "SARIMA", "SARIMAX", "SARIMAX-Agro", "VAR", "VAR-Agro"]):
    a.plot(hist.index, hist, color=CINZA, label="Observado")
    a.plot(prev_2025[m].index, prev_2025[m], color=LARANJA, marker="o", ms=3, label="Previsto")
    a.set_title(m, fontsize=10); formata_datas(a, 6); a.tick_params(axis="x", rotation=30)
for a in axs[:, 0]: a.set_ylabel("Inadimplência (%)")
axs[0, 0].legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig4_previsoes_2025.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
largura = 0.12
for i, m in enumerate(MODELOS):
    vals = [R["origem_movel"][h][m]["MAE"] for h in HORIZONTES]
    ax.bar(np.arange(3) + (i - 3) * largura, vals, largura, label=m)
ax.set_xticks(range(3)); ax.set_xticklabels([f"h = {h}" for h in HORIZONTES])
ax.set_ylabel("MAE (pontos percentuais)"); ax.set_xlabel("Horizonte de previsão (meses)")
ax.legend(frameon=False, ncol=4, fontsize=8)
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig5_origem_movel.png", dpi=200); plt.close(fig)

res_best = r_sar.resid.iloc[r_sar.loglikelihood_burn:]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
axs[0].plot(res_best, color=AZUL); axs[0].set_title("Resíduos", fontsize=10); formata_datas(axs[0], anos=2)
axs[1].hist(res_best, bins=20, color=AZUL, density=True); axs[1].set_title("Distribuição", fontsize=10)
plot_acf(res_best, lags=24, ax=axs[2], color=AZUL); axs[2].set_title("Função de autocorrelação", fontsize=10)
fig.tight_layout(); fig.savefig(f"{SAIDA}/fig6_residuos.png", dpi=200); plt.close(fig)
print("Figuras salvas em", SAIDA)
