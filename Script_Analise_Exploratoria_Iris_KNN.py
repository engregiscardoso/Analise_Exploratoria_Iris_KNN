# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:26 2022

@author: REGIS CARDOSO
"""


## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

## FUNÇÕES

# FUNÇÕES PARA PLOTAR GRAFICOS

def plotar_grafico_linha(df):
    plt.figure(figsize=(20, 10))
    plt.plot(df)
    plt.show()
    

def plotar_grafico_histograma(df):
    plt.figure(figsize=(15, 10))
    plt.hist(df, 10, rwidth=0.9)
    plt.show()
    

def apresenta_estatisticas(df):
    print(df.describe())


def plotar_grafico_boxplot(df):
    sns.boxplot(df)


def filtro_quartil_amplitude(df):
    
    print("Mínimo ANTES do filtro: ", min(df['coluna_avaliada']))
    print("Máximo ANTES do filtro: ", max(df['coluna_avaliada']))
    print("Total de dados ANTES do filtro: ", (df['coluna_avaliada']).count())
    
    print("")
    
    Q1 = df['coluna_avaliada'].quantile(0.25)
    Q2 = df['coluna_avaliada'].quantile(0.5)
    Q3 = df['coluna_avaliada'].quantile(0.75)
    
    Amp_interquartil = Q3 - Q1
    
    limite_inferior = (Q1 - (1.3 * Amp_interquartil))
    
    limite_superior = (Q3 + (1.3 * Amp_interquartil))
    
    df_mask=df['coluna_avaliada']>limite_inferior
    amplitudePos = df[df_mask]
    
    df_mask=amplitudePos['coluna_avaliada']<limite_superior
    df_final = amplitudePos[df_mask]
    
    print("Mínimo DEPOIS do filtro: ", min(df_final['coluna_avaliada']))
    print("Máximo DEPOIS do filtro: ", max(df_final['coluna_avaliada']))
    print("Total de dados DEPOIS do filtro: ", (df_final['coluna_avaliada']).count())
    
    return df_final

## IMPORTAR O ARQUIVO DE DADOS ###

df = pd.read_csv('iris_modificado.csv', sep=';')



## DEFINIR AS COLUNAS DO DATAFRAME ###

df.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']



## VERIFICA A DIMENSÃO DO DATASET ###

df.shape



## VERIFICAR O TIPO DE DADO EM CADA COLUNA ###

df.dtypes



## VERIFICAR AS CLASSES DO DATAFRAME ###

df['Classe'].unique()



## TRANSFORMAR AS CLASSES EM VALORES NUMÉRICOS ###

label_encoder = preprocessing.LabelEncoder()
  
df['Classe']= label_encoder.fit_transform(df['Classe'])
  
# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2


## VERIFICAR AS CLASSES DO DATAFRAME ###

df['Classe'].unique()



## TRANSFORMAR OS DADOS EM VALORES NUMÉRICOS DO TIPO FLOAT ###

columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal']

df[columns] = df[columns].apply(lambda x: x.str.replace(',', '.').astype('float'))



## VERIFICAR ESTATISTICAS DO DATAFRAME ###

df.describe(include='all')



## VERIFICAR SE EXISTEM DADOS VAZIOS NO DATAFRAME ###

verifica_valores_vazios = df.isnull().values.any()



## ANALISE EXPLORATÓRIA VIA GRÁFICOS E DADOS ESTATÍSTICOS ###

plotar_grafico_linha(df["Comprimento_Sepal"])
apresenta_estatisticas(df["Comprimento_Sepal"])
plotar_grafico_boxplot(df["Comprimento_Sepal"])

#### Analise com histogramas também é importante, porem não podem existir valores em branco nos dados
# plotar_grafico_histograma(df["Comprimento_Sepal"])


plotar_grafico_linha(df["Largura_Sepal"])
apresenta_estatisticas(df["Largura_Sepal"])
plotar_grafico_boxplot(df["Largura_Sepal"])

#### Analise com histogramas também é importante, porem não podem existir valores em branco nos dados
# plotar_grafico_histograma(df["Largura_Sepal"])


plotar_grafico_linha(df["Comprimento_Petal"])
apresenta_estatisticas(df["Comprimento_Petal"])
plotar_grafico_boxplot(df["Comprimento_Petal"])

#### Analise com histogramas também é importante, porem não podem existir valores em branco nos dados
# plotar_grafico_histograma(df["Comprimento_Petal"])


plotar_grafico_linha(df["Largura_Petal"])
apresenta_estatisticas(df["Largura_Petal"])
# plotar_grafico_boxplot(df["Largura_Petal"])

#### Analise com histogramas também é importante, porem não podem existir valores em branco nos dados
# plotar_grafico_histograma(df["Largura_Petal"])




## TRATAR OS DADOS VAZIOS 'NAN' ###
# OPÇÃO 01: EXCLUIR AS LINHAS QUE CONTENHAM DADOS NAN

df_sem_valores_nan = df.dropna()



# OPÇÃO 02: SUBSTITUIR OS VALORES POR VALORES ZERO

df = df.replace(np.nan, 0, regex=True)


# OPÇÃO 03: SUBSTITUIR OS VALORES POR VALORES DA MÉDIA, NESSE CASO PRECISA VERIFICAR BEM CERTO QUAIS VALORES MÉDIOS USAR

df["Comprimento_Sepal"] = df["Comprimento_Sepal"].replace(np.nan, df["Comprimento_Sepal"].mean(), regex=True)




plotar_grafico_histograma(df_sem_valores_nan["Comprimento_Sepal"])

plotar_grafico_histograma(df_sem_valores_nan["Largura_Sepal"])

plotar_grafico_histograma(df_sem_valores_nan["Comprimento_Petal"])

plotar_grafico_histograma(df_sem_valores_nan["Largura_Petal"])



## SEPARANDO O DATAFRAME PARA ANALISAR CADA UMA DAS CLASSES

df_mask = df_sem_valores_nan['Classe'] == 0
df_Iris_setosa = df_sem_valores_nan[df_mask]


df_mask = df_sem_valores_nan['Classe'] == 1
df_Iris_versicolor = df_sem_valores_nan[df_mask]


df_mask = df_sem_valores_nan['Classe'] == 2
df_Iris_virginica = df_sem_valores_nan[df_mask]




## ANALISE EXPLORATÓRIA VIA GRÁFICOS E DADOS ESTATÍSTICOS PARA ANALISAR CADA UMA DAS CLASSES ###

# Iris_setosa


plotar_grafico_linha(df_Iris_setosa["Comprimento_Sepal"])
apresenta_estatisticas(df_Iris_setosa["Comprimento_Sepal"])
plotar_grafico_boxplot(df_Iris_setosa["Comprimento_Sepal"])
plotar_grafico_histograma(df_Iris_setosa["Comprimento_Sepal"])

plotar_grafico_linha(df_Iris_setosa["Largura_Sepal"])
apresenta_estatisticas(df_Iris_setosa["Largura_Sepal"])
plotar_grafico_boxplot(df_Iris_setosa["Largura_Sepal"])
plotar_grafico_histograma(df_Iris_setosa["Largura_Sepal"])


plotar_grafico_linha(df_Iris_setosa["Comprimento_Petal"])
apresenta_estatisticas(df_Iris_setosa["Comprimento_Petal"])
plotar_grafico_boxplot(df_Iris_setosa["Comprimento_Petal"])
plotar_grafico_histograma(df_Iris_setosa["Comprimento_Petal"])


plotar_grafico_linha(df_Iris_setosa["Largura_Petal"])
apresenta_estatisticas(df_Iris_setosa["Largura_Petal"])
plotar_grafico_boxplot(df_Iris_setosa["Largura_Petal"])
plotar_grafico_histograma(df_Iris_setosa["Largura_Petal"])




# Iris_versicolor

plotar_grafico_linha(df_Iris_versicolor["Comprimento_Sepal"])
apresenta_estatisticas(df_Iris_versicolor["Comprimento_Sepal"])
plotar_grafico_boxplot(df_Iris_versicolor["Comprimento_Sepal"])
plotar_grafico_histograma(df_Iris_versicolor["Comprimento_Sepal"])


plotar_grafico_linha(df_Iris_versicolor["Largura_Sepal"])
apresenta_estatisticas(df_Iris_versicolor["Largura_Sepal"])
plotar_grafico_boxplot(df_Iris_versicolor["Largura_Sepal"])
plotar_grafico_histograma(df_Iris_versicolor["Largura_Sepal"])


plotar_grafico_linha(df_Iris_versicolor["Comprimento_Petal"])
apresenta_estatisticas(df_Iris_versicolor["Comprimento_Petal"])
plotar_grafico_boxplot(df_Iris_versicolor["Comprimento_Petal"])
plotar_grafico_histograma(df_Iris_versicolor["Comprimento_Petal"])


plotar_grafico_linha(df_Iris_versicolor["Largura_Petal"])
apresenta_estatisticas(df_Iris_versicolor["Largura_Petal"])
plotar_grafico_boxplot(df_Iris_versicolor["Largura_Petal"])
plotar_grafico_histograma(df_Iris_versicolor["Largura_Petal"])


# Iris_virginica

plotar_grafico_linha(df_Iris_virginica["Comprimento_Sepal"])
apresenta_estatisticas(df_Iris_virginica["Comprimento_Sepal"])
plotar_grafico_boxplot(df_Iris_virginica["Comprimento_Sepal"])
plotar_grafico_histograma(df_Iris_virginica["Comprimento_Sepal"])


plotar_grafico_linha(df_Iris_virginica["Largura_Sepal"])
apresenta_estatisticas(df_Iris_virginica["Largura_Sepal"])
plotar_grafico_boxplot(df_Iris_virginica["Largura_Sepal"])
plotar_grafico_histograma(df_Iris_virginica["Largura_Sepal"])


plotar_grafico_linha(df_Iris_virginica["Comprimento_Petal"])
apresenta_estatisticas(df_Iris_virginica["Comprimento_Petal"])
plotar_grafico_boxplot(df_Iris_virginica["Comprimento_Petal"])
plotar_grafico_histograma(df_Iris_virginica["Comprimento_Petal"])


plotar_grafico_linha(df_Iris_virginica["Largura_Petal"])
apresenta_estatisticas(df_Iris_virginica["Largura_Petal"])
plotar_grafico_boxplot(df_Iris_virginica["Largura_Petal"])
plotar_grafico_histograma(df_Iris_virginica["Largura_Petal"])




## EXCLUINDO OUTLIERS USANDO A TÉCNICA DE QUARTIL ###

# Iris_setosa - Comprimento_Sepal 

df_Iris_setosa_pre_filtro = df_Iris_setosa

df_Iris_setosa_pre_filtro.columns = ['coluna_avaliada', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado = filtro_quartil_amplitude(df_Iris_setosa_pre_filtro)

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_setosa_filtrado["Comprimento_Sepal"])

plotar_grafico_histograma(df_Iris_setosa_filtrado["Comprimento_Sepal"])




# Iris_setosa - Largura_Sepal 

df_Iris_setosa_pre_filtro = df_Iris_setosa_filtrado

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'coluna_avaliada', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado = filtro_quartil_amplitude(df_Iris_setosa_pre_filtro)

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_setosa_filtrado["Largura_Sepal"])

plotar_grafico_histograma(df_Iris_setosa_filtrado["Largura_Sepal"])



# Iris_setosa - Comprimento_Petal 

df_Iris_setosa_pre_filtro = df_Iris_setosa_filtrado

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'coluna_avaliada', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado = filtro_quartil_amplitude(df_Iris_setosa_pre_filtro)

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_setosa_filtrado["Comprimento_Petal"])

plotar_grafico_histograma(df_Iris_setosa_filtrado["Comprimento_Petal"])



# Iris_setosa - Largura_Petal 

df_Iris_setosa_pre_filtro = df_Iris_setosa_filtrado

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'coluna_avaliada', 'Classe']

df_Iris_setosa_filtrado = filtro_quartil_amplitude(df_Iris_setosa_pre_filtro)

df_Iris_setosa_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_setosa_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_setosa_filtrado["Largura_Petal"])

plotar_grafico_histograma(df_Iris_setosa_filtrado["Largura_Petal"])



##################################################################

# Iris_versicolor - Comprimento_Sepal 

df_Iris_versicolor_pre_filtro = df_Iris_versicolor

df_Iris_versicolor_pre_filtro.columns = ['coluna_avaliada', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado = filtro_quartil_amplitude(df_Iris_versicolor_pre_filtro)

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_versicolor_filtrado["Comprimento_Sepal"])

plotar_grafico_histograma(df_Iris_versicolor_filtrado["Comprimento_Sepal"])



# Iris_versicolor - Largura_Sepal 

df_Iris_versicolor_pre_filtro = df_Iris_versicolor_filtrado

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'coluna_avaliada', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado = filtro_quartil_amplitude(df_Iris_versicolor_pre_filtro)

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_versicolor_filtrado["Largura_Sepal"])

plotar_grafico_histograma(df_Iris_versicolor_filtrado["Largura_Sepal"])




# Iris_versicolor - Comprimento_Petal 

df_Iris_versicolor_pre_filtro = df_Iris_versicolor_filtrado

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'coluna_avaliada', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado = filtro_quartil_amplitude(df_Iris_versicolor_pre_filtro)

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_versicolor_filtrado["Comprimento_Petal"])

plotar_grafico_histograma(df_Iris_versicolor_filtrado["Comprimento_Petal"])



# Iris_versicolor - Largura_Petal 

df_Iris_versicolor_pre_filtro = df_Iris_versicolor_filtrado

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'coluna_avaliada', 'Classe']

df_Iris_versicolor_filtrado = filtro_quartil_amplitude(df_Iris_versicolor_pre_filtro)

df_Iris_versicolor_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_versicolor_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_versicolor_filtrado["Largura_Petal"])

plotar_grafico_histograma(df_Iris_versicolor_filtrado["Largura_Petal"])




##################################################################

# Iris_virginica - Comprimento_Sepal 

df_Iris_virginica_pre_filtro = df_Iris_virginica

df_Iris_virginica_pre_filtro.columns = ['coluna_avaliada', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado = filtro_quartil_amplitude(df_Iris_virginica_pre_filtro)

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_virginica_filtrado["Comprimento_Sepal"])

plotar_grafico_histograma(df_Iris_virginica_filtrado["Comprimento_Sepal"])

analise = df_Iris_virginica_filtrado.describe()


# Iris_virginica - Largura_Sepal 

df_Iris_virginica_pre_filtro = df_Iris_virginica_filtrado

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'coluna_avaliada', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado = filtro_quartil_amplitude(df_Iris_virginica_pre_filtro)

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_virginica_filtrado["Largura_Sepal"])

plotar_grafico_histograma(df_Iris_virginica_filtrado["Largura_Sepal"])




# Iris_virginica - Comprimento_Petal 

df_Iris_virginica_pre_filtro = df_Iris_virginica_filtrado

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'coluna_avaliada', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado = filtro_quartil_amplitude(df_Iris_virginica_pre_filtro)

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_virginica_filtrado["Comprimento_Petal"])

plotar_grafico_histograma(df_Iris_virginica_filtrado["Comprimento_Petal"])



# Iris_virginica - Largura_Petal 

df_Iris_virginica_pre_filtro = df_Iris_virginica_filtrado

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'coluna_avaliada', 'Classe']

df_Iris_virginica_filtrado = filtro_quartil_amplitude(df_Iris_virginica_pre_filtro)

df_Iris_virginica_pre_filtro.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

df_Iris_virginica_filtrado.columns = ['Comprimento_Sepal', 'Largura_Sepal', 'Comprimento_Petal', 'Largura_Petal', 'Classe']

plotar_grafico_boxplot(df_Iris_virginica_filtrado["Largura_Petal"])

plotar_grafico_histograma(df_Iris_virginica_filtrado["Largura_Petal"])





## JUNTADO AS CLASSES NOVAMENTE EM UM MESMO DATAFRAME ###

df_final_filtrado = pd.concat([df_Iris_setosa_filtrado, df_Iris_versicolor_filtrado], ignore_index = True)

df_final_filtrado = pd.concat([df_final_filtrado, df_Iris_virginica_filtrado], ignore_index = True)








## APLICANDO SMOTE PARA BALANCEAR OS DADOS CONFORME AS CLASSES ###

# Dividir o dataset em dados de treino e teste, aqui foi utilizado 30% para teste e 70% para treinamento

df_final_filtrado_SEM_classe = df_final_filtrado.iloc[:,:4]
df_final_filtrado_SOMENTE_classe = df_final_filtrado.iloc[:,4:5]




## NORMALIZAR OS DADOS ###

from sklearn.preprocessing import MaxAbsScaler
  
scaler = MaxAbsScaler()  
scaler.fit(df_final_filtrado_SEM_classe)
df_final_filtrado_SEM_classe_NORMALIZADO = scaler.transform(df_final_filtrado_SEM_classe)




X_train, X_test, y_train, y_test = train_test_split(df_final_filtrado_SEM_classe_NORMALIZADO, df_final_filtrado_SOMENTE_classe, test_size = 0.4, random_state = 0)



# Verificando quantas classes tem o dataset atual

quantidade_classes = y_train['Classe'].value_counts()
print(quantidade_classes)



# Aplicando SMOTE

oversample = SMOTE()
X_train_Smote, y_train_Smote = oversample.fit_resample(X_train, y_train)



# Verificando quantas classes tem o novo dataset

quantidade_classes = y_train_Smote['Classe'].value_counts()
print(quantidade_classes)




## CRIANDO E TREINANDO O MODELO ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report

knn_model = KNeighborsRegressor(n_neighbors=3)


knn_model.fit(X_train_Smote, y_train_Smote)





y_pred  = knn_model.predict(X_test)



## AVALIANDO O MODELO ##

y_pred = pd.DataFrame(y_pred)

y_pred = y_pred.astype(int)

print(classification_report(y_pred, y_test))


from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_pred, y_test)

print(matriz)




# IMPRIMIR UMA MATRIZ DE CONFUSÃO BONITA


# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2


#fig = plt.figure(figsize=(10,10))
#fig.suptitle('Matriz de Confusão ', fontsize=20, fontweight='bold')
#labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#label_font = {'size':'18'}  # Adjust to fit
#sns.heatmap(matriz,annot=True, fmt="d",cmap='Blues', xticklabels=labels, yticklabels= labels)




fig = plt.figure(figsize=(12,12))
cm = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.set(font_scale=3.0) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");  

# Labels, title and ticks
label_font = {'size':'18'}  # Adjust to fit
ax.set_xlabel('Previsto', fontdict=label_font);
ax.set_ylabel('Observado', fontdict=label_font);

title_font = {'size':'21'}  # Adjust to fit
ax.set_title('Confusion Matrix', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust to fit
ax.xaxis.set_ticklabels(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']);
ax.yaxis.set_ticklabels(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']);
plt.show()


