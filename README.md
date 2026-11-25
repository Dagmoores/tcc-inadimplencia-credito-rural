# Previsão de inadimplência no crédito rural brasileiro com séries temporais

Código-fonte do Trabalho de Conclusão de Curso **"Modelos de séries temporais para previsão de inadimplência no crédito rural brasileiro"**, apresentado ao MBA em Engenharia de Software da USP/ESALQ (2026).

**Autor:** Daniel Augusto Moore da Silva · **Orientador:** Prof. Luiz Fernando Pereira Nunes

O código coleta séries mensais do Banco Central do Brasil (jan/2016 a dez/2025), estima e compara modelos de previsão da inadimplência do crédito rural de pessoas físicas com taxas de mercado (SGS 21146) e valida os resultados com testes estatísticos.

## Correspondência entre scripts e resultados do TCC

| Etapa | Script | Resultados no TCC |
|---|---|---|
| Coleta, integração mensal e estatística descritiva | [`01_coleta_dados.py`](01_coleta_dados.py) | Tabela 1; Figuras 1 e 2 |
| Teste aumentado de Dickey-Fuller | [`02_estacionariedade_adf.py`](02_estacionariedade_adf.py) | Tabela 2; Figura 3 |
| Causalidade de Granger | [`03_causalidade_granger.py`](03_causalidade_granger.py) | Tabela 3 |
| Modelos e teste fora da amostra (2025) | [`04_modelos_teste_2025.py`](04_modelos_teste_2025.py) | Tabela 4; Figura 4 |
| Validação com origem móvel e teste de Diebold-Mariano | [`05_validacao_origem_movel_dm.py`](05_validacao_origem_movel_dm.py) | Tabelas 5 e 6; Figura 5 |
| Diagnóstico dos resíduos (Ljung-Box e Jarque-Bera) | [`06_diagnostico_residuos.py`](06_diagnostico_residuos.py) | Tabela 7; Figura 6 |
| Geração das figuras | [`07_figuras.py`](07_figuras.py) | Figuras 1 a 6 |
| Funções e dados compartilhados | [`comum.py`](comum.py) | – |

Os resultados numéricos ficam em [`saidas/resultados.json`](saidas/resultados.json) e as figuras em [`saidas/`](saidas/).

## Como executar

```bash
git clone https://github.com/Dagmoores/tcc-inadimplencia-credito-rural.git
cd tcc-inadimplencia-credito-rural
pip install -r requirements.txt
python executar_tudo.py        # ou cada script individualmente, na ordem
```

A execução completa leva cerca de 2 minutos. O arquivo [`dados/dados_bcb_2016_2025.csv`](dados/dados_bcb_2016_2025.csv) contém os dados exatamente como coletados em 23/11/2026, o que garante a reprodução dos números do TCC mesmo que o BCB revise as séries. Para coletar os dados novamente pela API, apague o CSV antes de executar.

Testado com Python 3.13, pandas 3.0, numpy 2.5, statsmodels 0.15, scipy 1.18 e matplotlib 3.11.

## Dados

| Código SGS | Série | Uso |
|---|---|---|
| 21146 | Inadimplência PF – crédito rural com taxas de mercado (%) | Variável dependente |
| 21147 | Inadimplência PF – crédito rural com taxas reguladas (%) | Endógena no VAR |
| 21148 | Inadimplência PF – crédito rural total (%) | Contexto |
| 432 | Meta da taxa Selic (% a.a.; média mensal dos valores diários) | Exógena / endógena |
| 433 | IPCA – variação mensal (%) | Exógena / endógena |
| 3698 | Taxa de câmbio R$/US$ (venda), média mensal | Exógena (agro) |
| 27575 | Índice de Commodities Brasil (IC-Br) – Agropecuária, em reais | Exógena (agro) |

Fonte: Banco Central do Brasil, Sistema Gerenciador de Séries Temporais (SGS).

## Modelos avaliados

Ingênuo (referência), ARIMA(1,2,1), SARIMA(1,2,1)(1,1,1)₁₂, SARIMAX (Selic e IPCA), SARIMAX-Agro (+ câmbio e IC-Br Agropecuária), VAR e VAR-Agro.

## Principais resultados

- **Validação com origem móvel (2023–2025):** o SARIMA teve o menor erro absoluto médio em todos os horizontes (0,199, 0,413 e 0,867 p.p. para 1, 3 e 6 meses).
- **Teste em 2025:** o VAR teve o menor erro (MAPE de 14,11%), mas não manteve o desempenho na validação com origem móvel.
- As variáveis macroeconômicas e do agronegócio não melhoraram as previsões, e os testes de causalidade de Granger não foram significativos.

## Como citar

Moore da Silva, D.A. 2026. Modelos de séries temporais para previsão de inadimplência no crédito rural brasileiro: código-fonte. Disponível em: <https://github.com/Dagmoores/tcc-inadimplencia-credito-rural>.

## Licença

[MIT](LICENSE)
