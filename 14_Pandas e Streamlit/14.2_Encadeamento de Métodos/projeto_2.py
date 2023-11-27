import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import graphviz as graphviz

from patsy import dmatrix

from matplotlib import figure

from seaborn import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto de Previsão de Renda",
     page_icon="https://media.istockphoto.com/id/171111712/pt/foto/dinheiro-em-bola-de-cristal-com-neve.jpg?s=612x612&w=0&k=20&c=aRy5JklnD1yCSnhpmPtT-8S7C92KdvROSCZDGv1q3uI=",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')

st.write('### Gráficos das Variáveis ao longo do tempo:')

#plots
fig, ax = plt.subplots(8,1,figsize=(15,20), constrained_layout=True)
plt.rcParams['legend.fontsize'] = 10
plt.rcParams["legend.title_fontsize"] = 12
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
ax[0].tick_params(axis='x', rotation=45, labelsize=10)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].set_ylabel( "Frequência" , size = 12 )
ax[0].set_xlabel( "Renda" , size = 12 )
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45, labelsize=10)
ax[1].tick_params(axis='y', labelsize=10)
ax[1].set_ylabel( "Renda" , size = 12 )
ax[1].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45, labelsize=10)
ax[2].tick_params(axis='y', labelsize=10)
ax[2].set_ylabel( "Renda" , size = 12 )
ax[2].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45, labelsize=10)
ax[3].tick_params(axis='y', labelsize=10)
ax[3].set_ylabel( "Renda" , size = 12 )
ax[3].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45, labelsize=10)
ax[4].tick_params(axis='y', labelsize=10)
ax[4].set_ylabel( "Renda" , size = 12 )
ax[4].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45, labelsize=10)
ax[5].tick_params(axis='y', labelsize=10)
ax[5].set_ylabel( "Renda" , size = 12 )
ax[5].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45, labelsize=10)
ax[6].tick_params(axis='y', labelsize=10)
ax[6].set_ylabel( "Renda" , size = 12 )
ax[6].set_xlabel( "Data de Referência" , size = 12 )

sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45, labelsize=10)
ax[7].tick_params(axis='y', labelsize=10)
ax[7].set_ylabel( "Renda" , size = 12 )
ax[7].set_xlabel( "Data de Referência" , size = 12 )
sns.despine()
st.pyplot(plt)

st.markdown("""---""")

st.write('## Gráficos Bivariada')
st.write('### Gráficos das Variáveis em função da variável resposta Renda:')
fig, ax = plt.subplots(7,1,figsize=(15,20), constrained_layout=True)
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
ax[0].tick_params(axis='x', rotation=45, labelsize=12)
ax[0].tick_params(axis='y', labelsize=12)
ax[0].set_ylabel( "Renda" , size = 15 )
ax[0].set_xlabel( "Possui Imóvel" , size = 15 )

sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45, labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].set_ylabel( "Renda" , size = 15 )
ax[1].set_xlabel( "Possui Veículo" , size = 15 )

sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45, labelsize=12)
ax[2].tick_params(axis='y', labelsize=12)
ax[2].set_ylabel( "Renda" , size = 15 )
ax[2].set_xlabel( "Quantidade de Filhos" , size = 15 )

sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45, labelsize=12)
ax[3].tick_params(axis='y', labelsize=12)
ax[3].set_ylabel( "Renda" , size = 15 )
ax[3].set_xlabel( "Tipo de Renda" , size = 15 )

sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45, labelsize=12)
ax[4].tick_params(axis='y', labelsize=12)
ax[4].set_ylabel( "Renda" , size = 15 )
ax[4].set_xlabel( "Nível Educacional" , size = 15 )

sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45, labelsize=12)
ax[5].tick_params(axis='y', labelsize=12)
ax[5].set_ylabel( "Renda" , size = 15 )
ax[5].set_xlabel( "Estado Civil" , size = 15 )

sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45, labelsize=12)
ax[6].tick_params(axis='y', labelsize=12)
ax[6].set_ylabel( "Renda" , size = 15 )
ax[6].set_xlabel( "Tipo de Residência" , size = 15 )

sns.despine()
st.pyplot(plt)

st.markdown("""---""")

st.write('## Análise das variáveis em função da idade')

renda_resumo = renda[['qtd_filhos', 'idade', 'tempo_emprego', 'renda']]

fig = plt.subplots(figsize=(3,6), constrained_layout=True)
sns.pairplot(
    renda_resumo,
    hue='idade',
    hue_order=None,
    palette=None,
    vars=None,
    x_vars=None,
    y_vars=None,
    kind='scatter',
    diag_kind='auto',
    markers=None,
    height=2.5,
    aspect=1,
    corner=False,
    dropna=False,
    plot_kws=None,
    diag_kws=None,
    grid_kws=None,
    size=5,
)
st.pyplot(plt)

st.markdown("""---""")

st.write('## Preparação dos Dados:')
st.write('### Dataframe original:')
st.dataframe(renda)
renda = (renda.drop(['Unnamed: 0', 'id_cliente'], axis=1))
renda = renda.dropna()

renda_dummies = renda.drop(['data_ref'], axis=1)
renda_dummies = pd.get_dummies(renda_dummies)

st.write('### Dataframe com transformação das variáveis em dummies:')
st.dataframe(renda_dummies)
st.write('### O tratamento das informações do dataframe consistiu em dropar as colunas - Unnamed e Id_Cliente - e transformando as variaveis em dummies para construção da árvore de regressão.')

st.write('## Modelagem:')
X = renda_dummies.drop("renda",axis = 1)
y = renda_dummies["renda"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=2360873)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=15)
regr_1.fit(X_test, y_test)
regr_2.fit(X_test, y_test)

st.write('#### Ficaram definidas duas árvores, com duas profundidades, 2 e 15. A seguier foram calculados seus respectivos MSE e R².')

df_dtr_mse = mean_squared_error(y_test, regr_1.predict(X_test))
df_dtr_r2 = regr_1.score(X_test, y_test)
template = "Árvore teste profundidade: {0}, MSE: {1:.2f}, R2: {2:.6f}"
st.write(template.format(regr_1.get_depth(), df_dtr_mse, df_dtr_r2))

df_dtr_mse = mean_squared_error(y_test, regr_2.predict(X_test))
df_dtr_r2 = regr_2.score(X_test, y_test)
template = "Árvore teste profundidade: {0}, MSE: {1:.2f}, R2: {2:.6f}"
st.write(template.format(regr_2.get_depth(), df_dtr_mse, df_dtr_r2))

# Problemas ao gerar a plotagem da árevore de regressão com o GRAPHVIZ!!!

# graph = tree.export_graphviz(regr_2, out_file=None, 
#                                 feature_names=X.columns,
#                                 filled=True)

# st.graphviz_chart(graph, use_container_width=False)

st.write('##### ÁRVORE COM PROFUNDIDADE 2:')
plt.figure(figsize=(25, 10))
tp = tree.plot_tree(regr_1, 
                    feature_names=X.columns,  
                    filled=True) 
st.pyplot(plt)

st.write('##### ÁRVORE COM PROFUNDIDADE 15:')
plt.figure(figsize=(25, 10))
tp = tree.plot_tree(regr_2, 
                    feature_names=X.columns,  
                    filled=True) 
st.pyplot(plt)

st.markdown("""---""")

st.write('#### Para a elaboração da regularização Rigde foram selecionadas apenas as variáveis que possuiam maior representação estatística no modelo.')
st.write('#### Regularização Ridge com transformação logarítmica na variável resposta RENDA.')

modelo = 'np.log(renda) ~ C(sexo) +  C(posse_de_imovel)  + idade + tempo_emprego '
md = smf.ols(modelo, data = renda)
reg = md.fit_regularized(method = 'elastic_net' 
                         , refit = True
                         , L1_wt = 0.001
                         , alpha = 1)


rr = reg.summary()
st.write(rr)


st.markdown("""---""")
st.write('## Avaliação dos resultados:')
st.write('#### Como esperado, com a transformação logarítmica, o R² teve um aumento sigbnificativo na regressão Ridge. Entretanto, a arvore de regressão apresentou um R² muito melhor que a regressão Rigde nas bases de treino e teste. Ao analisar a árvore de regressão pode-se observar com facilidade suas quebras e com quais variáveis deseja-se trabalhar, possibilitando prever a renda seguindo as iterações entre essas determinadas variáveis.')

st.markdown("""---""")
st.write('## Implantação:')
st.write('#### Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor que toma as decisões com algum nível de automação. Nessa etapa colocamos em uso o modelo desenvolvido. A implementação do modelo desenvolvido pode auxiliar a prever a renda dos clientes mesmo que não seja fornecido nenhum tipo de olerite, possibilitando a instituição financeira a oferecer tipos de serviços diferenciados aos clientes.')