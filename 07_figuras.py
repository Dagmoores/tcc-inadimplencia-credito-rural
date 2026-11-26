# -*- coding: utf-8 -*-
"""Etapa 7 — Geração das Figuras 1 a 6 do TCC (requer as etapas 1 a 6 executadas).

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python 07_figuras.py
"""
import logging

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # silencia aviso quando a Arial não está instalada
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from statsmodels.graphics.tsaplots import plot_acf

from comum import SAIDA, ajusta_sarimax, df, ler_resultados, y

R = ler_resultados()
MODELOS = list(R["teste_2025"].keys())
HORIZONTES = [1, 3, 6]
idx_2025 = df.index[-12:]
prev_2025 = {m: pd.Series(v, index=idx_2025) for m, v in R["prev_2025"].items()}
r_sar = ajusta_sarimax(y)

# Estilo: figuras geradas no tamanho real de impressão (15,5 cm de largura),
# fonte equivalente à Arial (Liberation Sans, mesmas métricas) e paleta validada.
CM = 1 / 2.54
LARGURA = 15.5 * CM
AZUL, LARANJA, CINZA, CINZA_CLARO, TINTA = "#2a78d6", "#eb6834", "#6b6b6b", "#c9c9c9", "#1f1f1f"
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8.5,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    # Padrão do Manual do MBA USP/Esalq: sem linhas de grade, sem borda e sem título no gráfico;
    # eixos principais em linha sólida preta de 1,5 pt; textos em preto.
    "axes.edgecolor": "black", "axes.linewidth": 1.5, "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.4, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.formatter.use_locale": False,
})


def virgula(ax, eixo="y", casas=None):
    """Formata os números do eixo com vírgula decimal (padrão brasileiro)."""
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _):
        s = f"{v:.{casas}f}" if casas is not None else f"{v:g}"
        return s.replace(".", ",").replace("-", "\u2212")
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(FuncFormatter(fmt))


def painel(ax, letra):
    """Identifica o painel com letra maiúscula, sem pontuação, no canto superior esquerdo."""
    ax.text(-0.02, 1.04, letra, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold", color="black")


def rotulo(ax, texto):
    """Nome da série dentro da área do gráfico (não é título do gráfico)."""
    ax.text(0.02, 0.97, texto, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="black")


def formata_datas(ax, meses=12, anos=1):
    if meses < 12:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, meses)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(anos))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def salvar(fig, nome):
    fig.savefig(f"{SAIDA}/{nome}", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


# Figura 1 - séries de inadimplência
fig, ax = plt.subplots(figsize=(LARGURA, 6.2 * CM))
ax.plot(df.index, df["inad_rural_pf_mercado"], color=AZUL, label="Taxas de mercado (SGS 21146)")
ax.plot(df.index, df["inad_rural_pf_regulada"], color=LARANJA, label="Taxas reguladas (SGS 21147)")
ax.set_ylabel("Inadimplência (%)"); formata_datas(ax); virgula(ax)
ax.legend(frameon=False, loc="upper left")
ax.set_xlim(df.index[0], df.index[-1] + pd.offsets.MonthBegin(1))
salvar(fig, "fig1_series.png")

# Figura 2 - variáveis explicativas (quatro painéis)
fig, axs = plt.subplots(2, 2, figsize=(LARGURA, 8.6 * CM), sharex=True)
for a, l, (c, t) in zip(axs.flat, "ABCD", [("selic", "Selic (% a.a.)"), ("ipca", "IPCA (% ao mês)"),
                                           ("cambio", "Câmbio (R\\$/US\\$)"), ("icbr_agro", "IC-Br Agropecuária")]):
    a.plot(df.index, df[c], color=AZUL, lw=1.2); a.set_ylabel(t); painel(a, l)
    formata_datas(a, anos=2); virgula(a)
fig.tight_layout(h_pad=1.2, w_pad=1.5)
salvar(fig, "fig2_explicativas.png")

# Figura 3 - diferenças da série alvo
fig, axs = plt.subplots(1, 2, figsize=(LARGURA, 5.4 * CM), sharey=False)
for a, l, (s, t) in zip(axs, "AB", [(y.diff(), "1ª diferença (pontos percentuais)"), (y.diff().diff(), "2ª diferença (pontos percentuais)")]):
    a.axhline(0, color=CINZA_CLARO, lw=0.8)
    a.plot(s, color=AZUL, lw=1.0); a.set_ylabel(t); painel(a, l)
    formata_datas(a, anos=2); virgula(a)
fig.tight_layout(w_pad=1.5)
salvar(fig, "fig3_diferencas.png")

# Figura 4 - previsões de 2025 (seis painéis)
fig, axs = plt.subplots(3, 2, figsize=(LARGURA, 12.3 * CM), sharex=True, sharey=True)
hist = y.loc["2023-01-01":]
for a, l, m in zip(axs.flat, "ABCDEF", ["ARIMA", "SARIMA", "SARIMAX", "SARIMAX-Agro", "VAR", "VAR-Agro"]):
    a.axvline(idx_2025[0] - pd.Timedelta(days=15), color=CINZA_CLARO, lw=0.8, ls="--")
    a.plot(hist.index, hist, color=CINZA, lw=1.4, label="Observado")
    a.plot(prev_2025[m].index, prev_2025[m], color=AZUL, lw=1.4, marker="o", ms=3, label="Previsto")
    painel(a, l); rotulo(a, m)
    formata_datas(a); a.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    a.yaxis.set_major_locator(MultipleLocator(3)); a.set_ylim(0, 13.5); virgula(a)
for a in axs[:, 0]:
    a.set_ylabel("Inadimplência (%)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.0, w_pad=1.2)
salvar(fig, "fig4_previsoes_2025.png")

# Figura 5 - MAE na validação com origem móvel (gráfico de pontos por horizonte)
fig, axs = plt.subplots(1, 3, figsize=(LARGURA, 6.4 * CM), sharey=True)
ordem = sorted(MODELOS, key=lambda m: R["origem_movel"]["6"][m]["MAE"], reverse=True)
for a, h in zip(axs, HORIZONTES):
    vals = [R["origem_movel"][str(h)][m]["MAE"] for m in ordem]
    cores = [AZUL if m == "SARIMA" else CINZA for m in ordem]
    ypos = range(len(ordem))
    a.hlines(ypos, 0, vals, color=CINZA_CLARO, lw=1.0)
    a.scatter(vals, ypos, color=cores, s=26, zorder=3)
    for yy, v, m in zip(ypos, vals, ordem):
        a.text(v + max(vals) * 0.04, yy, f"{v:.3f}".replace(".", ","), va="center", fontsize=7.5,
               color=AZUL if m == "SARIMA" else TINTA)
    painel(a, "ABC"[HORIZONTES.index(h)])
    a.set_xlim(0, max(vals) * 1.38)
    virgula(a, "x"); a.set_xlabel(f"MAE, h = {h} {'mês' if h == 1 else 'meses'}\n(pontos percentuais)")
axs[0].set_yticks(range(len(ordem))); axs[0].set_yticklabels(ordem)
for lbl in axs[0].get_yticklabels():
    if lbl.get_text() == "SARIMA":
        lbl.set_color(AZUL)
fig.tight_layout(w_pad=1.0)
salvar(fig, "fig5_origem_movel.png")

# Figura 6 - diagnóstico dos resíduos do SARIMA
res_best = r_sar.resid.iloc[r_sar.loglikelihood_burn:]
fig, axs = plt.subplots(1, 3, figsize=(LARGURA, 5.2 * CM))
axs[0].axhline(0, color=CINZA_CLARO, lw=0.8)
axs[0].plot(res_best, color=AZUL, lw=1.0); axs[0].set_ylabel("Resíduo (pontos percentuais)"); painel(axs[0], "A")
formata_datas(axs[0], anos=2); virgula(axs[0])
axs[1].hist(res_best, bins=18, color=AZUL, edgecolor="white", linewidth=0.6)
axs[1].set_ylabel("Frequência"); axs[1].set_xlabel("Resíduo (pontos percentuais)"); painel(axs[1], "B"); virgula(axs[1], "x")
plot_acf(res_best, lags=24, ax=axs[2], color=AZUL, vlines_kwargs={"colors": AZUL, "lw": 1.0},
         markersize=3, title="")
for coll in axs[2].collections:
    if "Poly" in coll.__class__.__name__:
        coll.set_facecolor(CINZA_CLARO); coll.set_edgecolor("none"); coll.set_alpha(0.7)
axs[2].set_ylabel("Autocorrelação"); painel(axs[2], "C"); axs[2].set_xlabel("Defasagem (meses)")
virgula(axs[2])
fig.tight_layout(w_pad=1.2)
salvar(fig, "fig6_residuos.png")
print("Figuras salvas em", SAIDA)
