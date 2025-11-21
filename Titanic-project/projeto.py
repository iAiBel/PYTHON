
# 2ª fase do crisp-dm
import pandas as pd

def carregarDados(nomeArquivo):
    data = None
    try:
        data = pd.read_csv(nomeArquivo, sep= ',')

        

    except:
        print("Erro na carga de dados")
       
    return data

def prepararDados(data):
    print("Preparação de Dados")# Aqui você pode adicionar o código para preparar os dados

    print(data.info()) # Exibe informações sobre o DataFrame

    print(data.head()) # Retorna as primeiras 5 linhas do arquivo

    print(data.tail()) # Retorna as últimas 5 linhas do arquivo
    
    print(data.describe()) # Retorna estatísticas básica sobre as colunas

    data.drop_duplicates(inplace=True) # Remove duplicatas do arquivo

    data.dropna(inplace=True) # Remove linhas com valores ausentes (nulos) do dataset

    data.drop(columns=['Cabin'], inplace=True) # Remove colunas desnecessárias

    print(data)

NOMEARQUIVO = 'Titanic-dataset.csv'
data = carregarDados(NOMEARQUIVO)

if data is not None:
    prepararDados(data)
    



