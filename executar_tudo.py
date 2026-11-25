# -*- coding: utf-8 -*-
"""Executa todas as etapas em sequência e reproduz todas as tabelas e figuras do TCC.

Parte do código do TCC "Modelos de séries temporais para previsão de inadimplência
no crédito rural brasileiro" (MBA USP/ESALQ) — Daniel Augusto Moore da Silva.
Execute a partir da pasta raiz do repositório: python executar_tudo.py
"""
import runpy
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
for etapa in ["01_coleta_dados.py", "02_estacionariedade_adf.py", "03_causalidade_granger.py",
              "04_modelos_teste_2025.py", "05_validacao_origem_movel_dm.py",
              "06_diagnostico_residuos.py", "07_figuras.py"]:
    print("\n=== Executando", etapa)
    runpy.run_path(os.path.join(RAIZ, etapa), run_name="__main__")
