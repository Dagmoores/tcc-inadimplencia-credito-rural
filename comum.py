# -*- coding: utf-8 -*-
"""Funções e dados compartilhados por todos os scripts.

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python (importado pelos demais scripts)
"""
import contextlib
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(RAIZ, "saidas")
os.makedirs(SAIDA, exist_ok=True)
ARQ = os.path.join(RAIZ, "dados", "dados_bcb_2016_2025.csv")
ARQ_RES = os.path.join(SAIDA, "resultados.json")

SERIES = {  # código SGS: nome da coluna
    432: "selic",                    # Meta da taxa Selic (% a.a.), diária -> média mensal
    433: "ipca",                     # IPCA, variação mensal (%)
    3698: "cambio",                  # Taxa de câmbio R$/US$ (venda), média mensal
    27575: "icbr_agro",              # Índice de Commodities Brasil (IC-Br) - Agropecuária, em reais
    21146: "inad_rural_pf_mercado",  # Inadimplência PF - crédito rural com taxas de mercado (alvo)
    21147: "inad_rural_pf_regulada", # Inadimplência PF - crédito rural com taxas reguladas
    21148: "inad_rural_pf_total",    # Inadimplência PF - crédito rural total
}


def coletar_bcb(inicio="2016-01-01", fim="2025-12-31"):
    """Coleta via API do SGS. A série diária da Selic é baixada em blocos de 5 anos,
    pois a API limita consultas diárias a 10 anos."""
    from bcb import sgs
    partes = []
    for cod, nome in SERIES.items():
        if cod == 432:
            s = pd.concat([sgs.get({nome: cod}, start="2016-01-01", end="2020-12-31"),
                           sgs.get({nome: cod}, start="2021-01-01", end=fim)])
        else:
            s = sgs.get({nome: cod}, start=inicio, end=fim)
        partes.append(s.resample("MS").mean())
    df = pd.concat(partes, axis=1).dropna()
    df.index.name = "data"
    return df



def carregar_dados():
    """Lê o CSV salvo (garante reprodutibilidade) ou coleta da API do BCB."""
    if os.path.exists(ARQ):
        df = pd.read_csv(ARQ, index_col=0, parse_dates=True)
    else:
        df = coletar_bcb()
        os.makedirs(os.path.dirname(ARQ), exist_ok=True)
        df.to_csv(ARQ)
    df.index.freq = "MS"
    df["log_cambio"] = np.log(df["cambio"])
    df["log_icbr_agro"] = np.log(df["icbr_agro"])
    return df


df = carregar_dados()
ALVO = "inad_rural_pf_mercado"
y = df[ALVO]
EXOG_MACRO = ["selic", "ipca"]
EXOG_AGRO = ["selic", "ipca", "log_cambio", "log_icbr_agro"]
N = len(df)


def salvar_resultados(novos):
    """Acrescenta resultados ao arquivo saidas/resultados.json."""
    res = json.load(open(ARQ_RES, encoding="utf-8")) if os.path.exists(ARQ_RES) else {}
    res.update(json.loads(json.dumps(novos, default=str)))
    with open(ARQ_RES, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def ler_resultados():
    return json.load(open(ARQ_RES, encoding="utf-8"))


def adf(s):
    r = adfuller(s.dropna(), autolag="AIC")
    return {"estat": round(r[0], 4), "p": round(r[1], 4), "lags": int(r[2])}


def ajusta_sarimax(serie, exog=None, sazonal=True):
    return SARIMAX(serie, exog=exog, order=(1, 2, 1),
                   seasonal_order=(1, 1, 1, 12) if sazonal else (0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def prever(modelo, ate, h):
    """Treina com dados até o índice 'ate' (exclusive) e prevê h passos.
    Para modelos com variáveis exógenas, usam-se os valores observados das
    exógenas no período previsto (previsão condicional ao cenário realizado)."""
    tr = df.iloc[:ate]
    idx = df.index[ate:ate + h]
    if modelo == "Ingênuo":
        return pd.Series(tr[ALVO].iloc[-1], index=idx)
    if modelo == "ARIMA":
        return ARIMA(tr[ALVO], order=(1, 2, 1)).fit().forecast(h)
    if modelo == "SARIMA":
        return ajusta_sarimax(tr[ALVO]).forecast(h)
    if modelo in ("SARIMAX", "SARIMAX-Agro"):
        cols = EXOG_MACRO if modelo == "SARIMAX" else EXOG_AGRO
        r = ajusta_sarimax(tr[ALVO], tr[cols])
        return r.forecast(h, exog=df[cols].iloc[ate:ate + h])
    if modelo in ("VAR", "VAR-Agro"):
        cols = (["selic", "ipca", ALVO, "inad_rural_pf_regulada"] if modelo == "VAR"
                else ["selic", "ipca", "log_cambio", "log_icbr_agro", ALVO])
        maxl = 12 if modelo == "VAR" else 6
        d = tr[cols]
        d2 = d.diff().diff().dropna()
        aj = VAR(d2).fit(maxlags=maxl, ic="aic")
        lag = max(aj.k_ar, 1)
        if aj.k_ar == 0:
            aj = VAR(d2).fit(1)
        fd = pd.DataFrame(aj.forecast(d2.values[-lag:], h), index=idx, columns=cols)
        nivel = fd.cumsum().add(d.diff().iloc[-1]).cumsum().add(d.iloc[-1])
        prever.ultimo_lag = aj.k_ar
        return nivel[ALVO]
    raise ValueError(modelo)


MODELOS = ["Ingênuo", "ARIMA", "SARIMA", "SARIMAX", "SARIMAX-Agro", "VAR", "VAR-Agro"]


def metricas(real, prev):
    e = real - prev
    return {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(e.abs().mean()),
            "MAPE": float((e / real).abs().mean() * 100)}


def diebold_mariano(e1, e2, h):
    """Teste DM com correção de Harvey, Leybourne e Newbold (1997); perda quadrática.
    Estatística negativa: modelo 1 mais preciso que o modelo 2."""
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)
