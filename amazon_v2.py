import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ARQUIVO = "C:/Users/mazar/OneDrive/Área de Trabalho/Projetos/Python-Amazon/amazon.csv"
le = LabelEncoder()

def abrir_arquivo(caminho):
    try:
        return pd.read_csv(caminho)
    except:
        return None

mapa_classificacao = {
    'G': 'All', 'TV-G': 'All', 'TV-Y': 'All', 'TV-Y7': 'All', 'ALL': 'All', 'L': 'All',
    'PG': 'Teen', 'TV-PG': 'Teen', 'PG-13': 'Teen', 'TV-14': 'Teen', '12': 'Teen', '14': 'Teen',
    'R': 'Adult', 'TV-MA': 'Adult', 'NC-17': 'Adult', '18': 'Adult', '16': 'Adult',
    'NR': 'Unknown', 'UR': 'Unknown', 'UNRATED': 'Unknown', 'NOT RATED': 'Unknown'
}

AZUL_PRIME = "#00A8E1"
AZUL_ESCURO = "#0F171E"
PRETO = "#000000"
CINZA_CLARO = "#F1F1F1"
BRANCO = "#FFFFFF"

def agrupar_classificacao(rating):
    if pd.isnull(rating):
        return 'Unknown'
    return mapa_classificacao.get(rating.upper().strip(), 'Unknown')

def tratar_dados(df):
    df.drop_duplicates(inplace=True)
    df.drop(['show_id', 'date_added', 'country', 'description','duration'], axis=1, inplace=True)
    df.dropna(subset=['rating'], inplace=True)
    df['cast'] = df['cast'].fillna('Desconhecido')
    df['director'] = df['director'].fillna('Desconhecido')
    df['rating'] = df['rating'].apply(agrupar_classificacao)
    df = df.query("release_year >= 2005")
    return df

def tratar_dados_mc (df):
    df['type'] = le.fit_transform(df['type'])
    df['rating'] = le.fit_transform(df['rating']) 
    df['listed_in'] = le.fit_transform(df['listed_in'])
    df['title'] = le.fit_transform(df['title'])
    df['director'] = le.fit_transform(df['director'])
    df['cast'] = le.fit_transform(df['cast'])
    return df

def grafico_categoriaTipo (df):
    contagem = df['type'].value_counts()
    plt.pie(contagem, labels=contagem.index, autopct='%1.1f%%', colors=[AZUL_PRIME, CINZA_CLARO], textprops={'color': PRETO})
    plt.title('Distribuição de Filmes e Séries na Amazon Prime Video')
    plt.legend(contagem.index, title="Tipo")
    plt.show()

def grafico_Genero(df):
    contagem = df['listed_in'].value_counts().head(10)  # top 10
    ax = contagem.plot(kind='bar', color='purple')
    bars = ax.bar(contagem.index, contagem.values, color=AZUL_PRIME)

    plt.title("Top 10 Gêneros na Amazon Prime Video", color=AZUL_PRIME)
    plt.xlabel("Gênero", color=BRANCO)
    plt.ylabel("Quantidade", color=BRANCO)
    plt.xticks(rotation=45)

    for i, valor in enumerate(contagem.values):
        ax.text(i, valor + 10, str(valor), ha='center', va='bottom', fontsize=9, color=BRANCO)

    plt.tight_layout()
    plt.show()

def grafico_faixaEtaria (df):
    contagem = df['rating'].value_counts()
    contagem.plot(kind='bar', color=AZUL_PRIME)
    plt.title("Distribuição por Classificação Etária", color=AZUL_PRIME)
    plt.xlabel("Classificação", color=BRANCO)
    plt.ylabel("Quantidade", color=PRETO)
    plt.show()


def boxplot_geral(dados, coluna):
    fig, ax = plt.subplots()
    ax.set_ylabel('Eixo y')
    
    for n, col in enumerate(dados.columns):
        if col == coluna:
            ax.boxplot(dados[col], positions= [n+1])
    plt.title(coluna)
    plt.ylabel("Valores")
    plt.show()

def grafico_tendencia(df):
    contagem = df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
    contagem.plot(kind='line', figsize=(10,5), color=[AZUL_PRIME, CINZA_CLARO])
    plt.title("Evolução de Filmes e Séries por Ano", color=AZUL_PRIME)
    plt.xlabel("Ano", color=BRANCO)
    plt.ylabel("Quantidade", color=BRANCO)
    plt.legend(title="Tipo")
    plt.show()

dados = abrir_arquivo(ARQUIVO)
ano_lancamento = 'release_year'

if dados is not None:
    print("Arquivo carregado com sucesso")
    print(dados.head())
    dados_tratados = tratar_dados(dados.copy())
    dados_tratados_mc = tratar_dados_mc(dados_tratados.copy())

    print(dados_tratados.info())
    print(dados_tratados_mc.info())
    boxplot_geral(dados_tratados_mc, ano_lancamento)


    grafico_categoriaTipo (dados_tratados)
    grafico_Genero(dados_tratados)
    grafico_faixaEtaria(dados_tratados)
    grafico_tendencia(dados_tratados)

    #Separando 70% dos dados para treino e 30% para teste
    treino = dados_tratados_mc.sample(frac=0.7, random_state=42)
    teste = dados_tratados_mc.drop(treino.index)

    print(f'Tamanho do conjunto de treino: {len(treino)}')
    print(f'Tamanho do conjunto de teste: {len(teste)}')

    treino.to_csv('amazon_treino.csv', index=False)
    teste.to_csv('amazon_teste.csv', index=False)   

    print("Conjuntos de treino e teste salvos com sucesso.")

else:
    print("Erro ao carregar o arquivo.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Separando features e target
X_train = treino.drop('rating', axis=1)
y_train = treino['rating']

X_test = teste.drop('rating', axis=1)
y_test = teste['rating']

# Criando e treinando o modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Fazendo predição
y_pred = rf.predict(X_test)

# Avaliando desempenho
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gráfico de importância das features
import matplotlib.pyplot as plt

importances = rf.feature_importances_
features = X_train.columns
plt.barh(features, importances)
plt.title("Importância das Features")
plt.show()