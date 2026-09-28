# Análise de Agrupamento de Dados e Embeddings de Texto

Este projeto realiza o agrupamento não supervisionado (*Clustering*) de textos em português utilizando técnica de representação por embeddings (*Deep Learning*) com o modelo `SentenceTransformer`, redução de dimensionalidade através do **PCA** e agrupamento via **K-Means**.

## 🚀 Tecnologias e Bibliotecas Utilizadas

* **Python 3.10+**
* **Pandas** e **NumPy**: Manipulação e análise dos dados.
* **Sentence-Transformers**: Gerador de embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`).
* **Scikit-Learn**:
  * `StandardScaler`: Normalização dos vetores.
  * `PCA`: Redução de dimensionalidade (2D e 3D).
  * `KMeans` & `silhouette_score`: Agrupamento e avaliação técnica da quantidade ideal de clusters ($K$).
* **Matplotlib** e **Seaborn**: Visualização de dados e dados tridimensionais.

---

## 🛠️ Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Criar e ativar o ambiente virtual (recomendado)
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar no Linux/macOS
source .venv/bin/activate

# Ativar no Windows (Prompt de Comando)
.venv\Scripts\activate.bat
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

---

## 🏃 Como Executar

Certifique-se de que o arquivo de dados `dataset_cd_portugues.csv` está na raiz do projeto e execute o script principal:

```bash
python main.py
```

---

## 📊 Fluxo da Aplicação

1. **Análise Exploratória:** Exibe a distribuição e contagem das categorias do dataset inicial.
2. **Vetorização (Embeddings):** Transforma os textos em representações vetoriais de 384 dimensões usando Deep Learning.
3. **Análise PCA:** Aplica PCA para visualizar a dispersão dos textos em gráficos 2D e 3D, imprimindo a taxa de variância explicada.
4. **Otimização de $K$ (Silhouette Score):** Testa valores de $K$ entre 2 e 10 para identificar o melhor agrupamento.
5. **Agrupamento Final:** Executa o algoritmo **K-Means** com o valor ideal de $K$ e associa o número do cluster às instâncias originais do dataset.