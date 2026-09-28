# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Carregamento da base de dados
df = pd.read_csv("dataset_cd_portugues.csv")

print("Estrutura do DataFrame:", df.shape)
print("\nDistribuição das categorias:")
print(df["categoria"].value_counts())

# 2. Gráfico 1: Distribuição das categorias
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="categoria")
plt.xticks(rotation=45)
plt.title("Distribuição dos textos por categoria")
plt.tight_layout()
plt.show(block=False) # block=False permite continuar a execução sem esperar fechar a janela

# 3. Embeddings
print("\nGerando embeddings com SentenceTransformer...")
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
textos = df["texto"].tolist()
embeddings = modelo.encode(textos)

# 4. Padronização
scaler = StandardScaler()
embeddings_norm = scaler.fit_transform(embeddings)

# 5. PCA Total
pca_total = PCA()
pca_total.fit(embeddings_norm)
variancia = pca_total.explained_variance_ratio_

print("\n--- Variância Explicada ---")
print(f"PC1: {variancia[0]:.4f}")
print(f"PC2: {variancia[1]:.4f}")
print(f"PC3: {variancia[2]:.4f}")
print(f"Variância acumulada (PC1 + PC2 + PC3): {sum(variancia[:3]):.4f}")

# 6. Gráfico 2: PCA 2D
pca2 = PCA(n_components=2)
pca_2d = pca2.fit_transform(embeddings_norm)
df_pca2 = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])

plt.figure(figsize=(8, 6))
plt.scatter(df_pca2["PC1"], df_pca2["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Representação dos textos após PCA (2D)")
plt.show(block=False)

# 7. Gráfico 3: PCA 3D
pca3 = PCA(n_components=3)
pca_3d = pca3.fit_transform(embeddings_norm)
df_pca3 = pd.DataFrame(pca_3d, columns=["PC1", "PC2", "PC3"])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_pca3["PC1"], df_pca3["PC2"], df_pca3["PC3"])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("Representação dos textos após PCA (Tridimensional)")
plt.show(block=False)

# 8. Silhouette Score
print("\nCalculando Silhouette Score para K-Means...")
resultados = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = kmeans.fit_predict(embeddings_norm)
    sil = silhouette_score(embeddings_norm, grupos)
    resultados.append([k, sil])

resultado_silhouette = pd.DataFrame(resultados, columns=["k", "silhouette_score"])
print("\n--- Resultados Silhouette Score ---")
print(resultado_silhouette)

# 9. Gráfico 4: Avaliação dos agrupamentos (Último gráfico)
plt.figure(figsize=(8, 5))
plt.plot(resultado_silhouette["k"], resultado_silhouette["silhouette_score"], marker="o")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Silhouette Score")
plt.title("Avaliação dos agrupamentos")
plt.grid(True)

# 10. KMeans Final
melhor_k = int(resultado_silhouette.loc[resultado_silhouette["silhouette_score"].idxmax(), "k"])
print(f"\nMelhor número de grupos (K): {melhor_k}")

kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(embeddings_norm)

print("\n--- Primeiras linhas do resultado final ---")
print(df.head())

# O último plt.show() sem parâmetros prende todas as janelas abertas na tela até você fechar
plt.show()