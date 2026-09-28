# -*- coding: utf-8 -*-
"""
Análise de Agrupamento de Dados - Execução Local Interativa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Carregamento do dataset com tratamento de codificação
try:
    df = pd.read_csv("dataset_cd_portugues.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("dataset_cd_portugues.csv", encoding="latin1")

# Correção de mojibake (dupla codificação de caracteres) se houver
for col in ["texto", "categoria"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x.encode('latin1').decode('utf-8') if 'Ã' in x else x
        )

print("Estrutura do DataFrame:", df.shape)
print("\nDistribuição das categorias:")
print(df["categoria"].value_counts())

# 2. Gráfico de barras da distribuição por categoria
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="categoria")
plt.xticks(rotation=45)
plt.title("Distribuição dos textos por categoria")
plt.tight_layout()
plt.show(block=False)

# 3. Geração de Embeddings
print("\nGerando embeddings com SentenceTransformer...")
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
textos = df["texto"].tolist()
embeddings = modelo.encode(textos)

print("Formato dos embeddings:", embeddings.shape)

# 4. Padronização
scaler = StandardScaler()
embeddings_norm = scaler.fit_transform(embeddings)

# 5. PCA - Variância Explicada
pca_total = PCA()
pca_total.fit(embeddings_norm)
variancia = pca_total.explained_variance_ratio_

print("\n--- Variância Explicada ---")
print(f"PC1: {variancia[0]:.4f}")
print(f"PC2: {variancia[1]:.4f}")
print(f"PC3: {variancia[2]:.4f}")
print(f"Variância acumulada (PC1 + PC2 + PC3): {sum(variancia[:3]):.4f}")

# 6. PCA 2D e Clusterização inicial (K=5)
pca2 = PCA(n_components=2)
pca_2d = pca2.fit_transform(embeddings_norm)
df_pca2 = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])

kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
df_pca2["cluster"] = kmeans_pca.fit_predict(df_pca2[["PC1", "PC2"]])
df_pca2["texto"] = df["texto"]

# Gráfico Estático PCA 2D (Matplotlib)
plt.figure(figsize=(8, 6))
plt.scatter(df_pca2["PC1"], df_pca2["PC2"], c=df_pca2["cluster"], cmap="viridis", s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Representação dos textos após PCA (2D)")
plt.show(block=False)

# Gráfico Interativo PCA 2D (Plotly)
fig_2d = px.scatter(
    df_pca2,
    x="PC1",
    y="PC2",
    color="cluster",
    hover_name="texto",
    color_continuous_scale="Viridis",
    title="Representação dos textos após PCA (2D - Interativo)",
    width=900,
    height=600
)
fig_2d.update_traces(marker=dict(size=10))
fig_2d.show()

# 7. PCA 3D
pca3 = PCA(n_components=3)
pca_3d = pca3.fit_transform(embeddings_norm)
df_pca3 = pd.DataFrame(pca_3d, columns=["PC1", "PC2", "PC3"])

# Gráfico Interativo PCA 3D (Plotly)
fig_3d = px.scatter_3d(
    df_pca3,
    x="PC1",
    y="PC2",
    z="PC3",
    color=df_pca2["cluster"],
    hover_name=df["texto"],
    color_continuous_scale="Viridis",
    title="Embeddings de Textos Projetados com PCA (3D - Interativo)"
)
fig_3d.update_layout(width=900, height=700)
fig_3d.show()

# 8. Avaliação Silhouette Score
print("\nCalculando Silhouette Score para K-Means nos embeddings originais...")
resultados = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = kmeans.fit_predict(embeddings_norm)
    sil = silhouette_score(embeddings_norm, grupos)
    resultados.append([k, sil])

resultado_silhouette = pd.DataFrame(resultados, columns=["k", "silhouette_score"])
print("\n--- Resultados Silhouette Score ---")
print(resultado_silhouette)

# Gráfico Silhouette
plt.figure(figsize=(8, 5))
plt.plot(resultado_silhouette["k"], resultado_silhouette["silhouette_score"], marker="o")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Silhouette Score")
plt.title("Avaliação dos agrupamentos")
plt.grid(True)

# 9. KMeans Final nos Embeddings Originais
melhor_k = int(resultado_silhouette.loc[resultado_silhouette["silhouette_score"].idxmax(), "k"])
print(f"\nMelhor número de grupos (K): {melhor_k}")

kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(embeddings_norm)

print("\n--- Primeiras linhas com os clusters finais ---")
print(df.head())

plt.show()