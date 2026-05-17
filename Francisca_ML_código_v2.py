# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:29:07 2026

@author: franc
"""

# -*- coding: utf-8 -*-
"""
Atividade 6 — Modelos de Machine Learning
Dataset: Online Retail (UK)
Autora: Francisca Carvalhais
"""

# ============================================================
# IMPORTAR AS BIBLIOTECAS NECESSÁRIAS
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas para Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report, confusion_matrix)

# ============================================================
# CRIAR PASTAS PARA GUARDAR OS RESULTADOS
# ============================================================
pasta_base    = r"C:\Users\franc\OneDrive - UTAD\Ambiente de Trabalho\Mestrado\2º Semestre\Business Intelligence\Atividades\Atividade 6 - ML individual\resultados_v2"
pasta_limpeza = os.path.join(pasta_base, "01_limpeza")
pasta_eda     = os.path.join(pasta_base, "02_eda")
pasta_modelos = os.path.join(pasta_base, "03_modelos")

for pasta in [pasta_limpeza, pasta_eda, pasta_modelos]:
    os.makedirs(pasta, exist_ok=True)
print("✓ Pastas criadas com sucesso!")

# ============================================================
# CARREGAR O DATASET
# ============================================================
df = pd.read_csv(r"C:\Users\franc\OneDrive - UTAD\Ambiente de Trabalho\Mestrado\2º Semestre\Business Intelligence\Atividades\Atividade 6 - ML individual\supermarket_data.csv",
                 encoding='ISO-8859-1')
print(f"✓ Dataset carregado! Dimensão: {df.shape}")

# ============================================================
# PRIMEIRA INSPEÇÃO
# ============================================================
print("\n=== PRIMEIRAS 5 LINHAS ===")
print(df.head())

print("\n=== INFORMAÇÃO GERAL ===")
print(df.info())

print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
print(df[['Quantity', 'Price']].describe().round(2))