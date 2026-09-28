# -*- coding: utf-8 -*-
import os
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

# ReportLab para geração de PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def executar_analise_e_gerar_pdf():
    # 1. Carregamento do dataset
    try:
        df = pd.read_csv("dataset_cd_portugues.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("dataset_cd_portugues.csv", encoding="latin1")

    # Correção de dupla codificação de caracteres
    for col in ["texto", "categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: x.encode('latin1').decode('utf-8') if 'Ã' in x else x
            )

    os.makedirs("temp_img", exist_ok=True)

    # 2. Gráfico 1: Categorias
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="categoria")
    plt.xticks(rotation=45)
    plt.title("Distribuição dos Textos por Categoria")
    plt.tight_layout()
    img_cat_path = "temp_img/distribuicao_categorias.png"
    plt.savefig(img_cat_path, dpi=200)
    plt.close()

    # 3. Embeddings
    print("Gerando embeddings...")
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

    # 6. PCA 2D com K-Means (K=5)
    pca2 = PCA(n_components=2)
    pca_2d = pca2.fit_transform(embeddings_norm)
    df_pca2 = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])

    kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_pca2["cluster"] = kmeans_pca.fit_predict(df_pca2[["PC1", "PC2"]])

    plt.figure(figsize=(7, 5))
    plt.scatter(df_pca2["PC1"], df_pca2["PC2"], c=df_pca2["cluster"], cmap="viridis", alpha=0.8, s=40)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Representação dos Textos após PCA (2D com Clusters K=5)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_pca2d_path = "temp_img/pca_2d.png"
    plt.savefig(img_pca2d_path, dpi=200)
    plt.close()

    # 7. PCA 3D
    pca3 = PCA(n_components=3)
    pca_3d = pca3.fit_transform(embeddings_norm)
    df_pca3 = pd.DataFrame(pca_3d, columns=["PC1", "PC2", "PC3"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df_pca3["PC1"], df_pca3["PC2"], df_pca3["PC3"], c=df_pca2["cluster"], cmap="viridis", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("Representação dos Textos após PCA (3D)")
    plt.tight_layout()
    img_pca3d_path = "temp_img/pca_3d.png"
    plt.savefig(img_pca3d_path, dpi=200)
    plt.close()

    # 8. Silhouette Score
    print("Calculando Silhouette Score...")
    resultados = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        grupos = kmeans.fit_predict(embeddings_norm)
        sil = silhouette_score(embeddings_norm, grupos)
        resultados.append([k, sil])

    resultado_silhouette = pd.DataFrame(resultados, columns=["k", "silhouette_score"])

    # Gráfico Silhouette
    plt.figure(figsize=(8, 4))
    plt.plot(resultado_silhouette["k"], resultado_silhouette["silhouette_score"], marker="o", color='crimson')
    plt.xlabel("Número de Grupos (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Avaliação dos Agrupamentos (Método Silhouette)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    img_sil_path = "temp_img/silhouette.png"
    plt.savefig(img_sil_path, dpi=200)
    plt.close()

    # 9. KMeans Final
    melhor_k = int(resultado_silhouette.loc[resultado_silhouette["silhouette_score"].idxmax(), "k"])
    kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
    df["cluster"] = kmeans_final.fit_predict(embeddings_norm)

    # ---------------------------------------------------------
    # CONSTRUÇÃO DO PDF COM REPORTLAB
    # ---------------------------------------------------------
    print("Gerando arquivo PDF...")
    pdf_filename = "relatorio_agrupamento.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    cell_text_full = ParagraphStyle(
        'CellTextFull',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    cell_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold'
    )

    elements = []

    # Cabeçalho
    elements.append(Paragraph("Relatório de Agrupamento de Dados e Embeddings", title_style))
    elements.append(Paragraph("Análise não supervisionada com SentenceTransformers, PCA e K-Means.", body_style))
    elements.append(Spacer(1, 12))

    # Seção 1
    elements.append(Paragraph("1. Distribuição Inicial das Categorias", h2_style))
    elements.append(
        Paragraph(f"O conjunto de dados contém {df.shape[0]} amostras e {df.shape[1]} colunas.", body_style))
    elements.append(Spacer(1, 8))
    elements.append(Image(img_cat_path, width=480, height=240))
    elements.append(Spacer(1, 14))

    # ---------------------------------------------------------
    # Seção 2: Análise de Componentes Principais (PCA)
    # ---------------------------------------------------------
    elements.append(Paragraph("2. Análise de Componentes Principais (PCA)", h2_style))
    variancia_texto = (
        f"<b>PC1:</b> {variancia[0]:.2%} | "
        f"<b>PC2:</b> {variancia[1]:.2%} | "
        f"<b>PC3:</b> {variancia[2]:.2%}<br/>"
        f"<b>Variância Acumulada (PC1+PC2+PC3):</b> {sum(variancia[:3]):.2%}"
    )
    elements.append(Paragraph(variancia_texto, body_style))
    elements.append(Spacer(1, 10))

    # Gráficos em tamanho expandido colocados um abaixo do outro
    img_pca2d = Image(img_pca2d_path, width=350, height=250)
    img_pca3d = Image(img_pca3d_path, width=450, height=320)

    elements.append(img_pca2d)
    elements.append(Spacer(1, 12))
    elements.append(img_pca3d)
    elements.append(Spacer(1, 14))

    # Seção 3 (Disposição Vertical Tabela + Gráfico)
    elements.append(Paragraph("3. Escolha do Número de Clusters (Silhouette Score)", h2_style))
    elements.append(Paragraph(f"O número ideal de clusters identificado foi <b>K = {melhor_k}</b>.", body_style))
    elements.append(Spacer(1, 10))

    table_data = [["Número de Grupos (K)", "Silhouette Score"]]
    for _, row in resultado_silhouette.iterrows():
        table_data.append([str(int(row["k"])), f"{row['silhouette_score']:.4f}"])

    t_sil_data = Table(table_data, colWidths=[200, 200])
    t_sil_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    elements.append(t_sil_data)
    elements.append(Spacer(1, 12))

    img_sil = Image(img_sil_path, width=420, height=210)
    elements.append(img_sil)
    elements.append(Spacer(1, 14))

    # Seção 4 (Texto Inteiro nas Células)
    elements.append(Paragraph("4. Amostra do Agrupamento Final (K-Means)", h2_style))
    sample_df = df[["texto", "categoria", "cluster"]].head(6)

    table_sample = [[
        Paragraph("Texto (Completo)", cell_header_style),
        Paragraph("Categoria Original", cell_header_style),
        Paragraph("Cluster", cell_header_style)
    ]]

    for _, row in sample_df.iterrows():
        table_sample.append([
            Paragraph(str(row["texto"]), cell_text_full),
            Paragraph(str(row["categoria"]), cell_style),
            Paragraph(str(row["cluster"]), cell_style)
        ])

    t_sample = Table(table_sample, colWidths=[330, 100, 50])
    t_sample.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_sample)

    doc.build(elements)
    print(f"PDF gerado com sucesso: {pdf_filename}")


if __name__ == "__main__":
    executar_analise_e_gerar_pdf()