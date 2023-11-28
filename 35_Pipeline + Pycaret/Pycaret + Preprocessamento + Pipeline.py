
![image.png](attachment:image.png)
# Tarefa II

Neste projeto, estamos construindo um credit scoring para cartão de crédito, em um desenho amostral com 15 safras, e utilizando 12 meses de performance.

Carregue a base de dados ```credit_scoring.ftr```.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn import metrics
from scipy.stats import ks_2samp
from scipy.stats import t

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
df = pd.read_feather('./Dados/credit_scoring.ftr')
df.head()
df.to_csv('credit_scoring_csv.csv')
df_csv = pd.read_csv('credit_scoring_csv.csv')
df_csv
print('numero de linhas: {0} \nnúmero de colunas: {1}'.format(df.shape[0], df.shape[1]))
df.mau.value_counts(normalize=True)
df['data_ref'].unique()
## Amostragem

Separe os três últimos meses como safras de validação *out of time* (oot).

Variáveis:<br>
Considere que a variável ```data_ref``` não é uma variável explicativa, é somente uma variável indicadora da safra, e não deve ser utilizada na modelagem. A variávei ```index``` é um identificador do cliente, e também não deve ser utilizada como covariável (variável explicativa). As restantes podem ser utilizadas para prever a inadimplência, incluindo a renda.

df_3meses = df[df["data_ref"].isin(pd.date_range("2016-01-01", "2016-03-01"))]
df_3meses["data_ref"].unique()
df_3meses.mau.value_counts()
metadados = pd.DataFrame(df_3meses.dtypes, columns=['dtype'])
metadados['nmissing'] = df.isna().sum()
metadados['valores_unicos'] = df.nunique()

metadados
# Transformar a variável resposta em inteiro
df_3meses['mau'] = df_3meses.mau.astype('int64')
## Descritiva básica univariada

- Descreva a base quanto ao número de linhas, número de linhas para cada mês em ```data_ref```.
- Faça uma descritiva básica univariada de cada variável. Considere as naturezas diferentes: qualitativas e quantitativas.
df_3meses.groupby('data_ref')['mau'].count()
#### A base selecionada com os últimos 3 meses apresenta-se com cerca de 50000 linhas para cada mês (jan, fev e mar de 2016)
def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

    rótulo_evento = tab.columns[0]
    rótulo_nao_evento = tab.columns[1]

    tab['pct_evento'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]
    tab['pct_nao_evento'] = tab[rótulo_nao_evento]/tab.loc['total',rótulo_nao_evento]
    coringa = 0.00001
    tab['woe'] = np.log(tab.pct_evento+coringa/tab.pct_nao_evento+coringa)
    tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento)*tab.woe
    return tab['iv_parcial'].sum()


iv = IV(df_3meses.renda, df_3meses.mau)
iv
metadados['papel'] = 'covariavel'
metadados.loc['mau','papel'] = 'resposta'
metadados['nunique'] = df.nunique()
metadados
for var in metadados[metadados.papel=='covariavel'].index:
    metadados.loc[var, 'IV'] = IV(df_3meses[var], df_3meses.mau)
metadados
df_3meses_select = df_3meses[['sexo', 'posse_de_imovel', 'qtd_filhos', 'tipo_renda', 'estado_civil', 'tipo_residencia', 'idade', 'mau']]
sns.pairplot(df_3meses_select, hue="mau")
## Descritiva bivariada

Faça uma análise descritiva bivariada de cada variável
metadados = pd.DataFrame(df_3meses.dtypes, columns=['dtype'])
metadados['nmissing'] = df_3meses.isna().sum()
metadados['valores_unicos'] = df_3meses.nunique()
metadados['papel'] = 'covariavel'
metadados.loc['mau','papel'] = 'resposta'
metadados.loc['bom','papel'] = 'resposta'
metadados
var='idade'
IV(pd.qcut(df_3meses[var],5,duplicates='drop'), df_3meses.mau)
for var in metadados[metadados.papel=='covariavel'].index:
    if  (metadados.loc[var, 'valores_unicos']>6):
        metadados.loc[var, 'IV'] = IV(pd.qcut(df[var],5,duplicates='drop'), df.mau)
    else: 
        metadados.loc[var, 'IV'] = IV(df[var], df.mau)

    
metadados
def biv_discreta(var, df):
    df['bom'] = 1-df_3meses.mau
    g = df.groupby(var)

    biv = pd.DataFrame({'qt_bom': g['bom'].sum(),
                        'qt_mau': g['mau'].sum(),
                        'mau':g['mau'].mean(), 
                        var: g['mau'].mean().index, 
                        'cont':g[var].count()})
    
    biv['ep'] = (biv.mau*(1-biv.mau)/biv.cont)**.5
    biv['mau_sup'] = biv.mau+t.ppf([0.975], biv.cont-1)*biv.ep
    biv['mau_inf'] = biv.mau+t.ppf([0.025], biv.cont-1)*biv.ep
    
    biv['logit'] = np.log(biv.mau/(1-biv.mau))
    biv['logit_sup'] = np.log(biv.mau_sup/(1-biv.mau_sup))
    biv['logit_inf'] = np.log(biv.mau_inf/(1-biv.mau_inf))

    tx_mau_geral = df.mau.mean()
    woe_geral = np.log(df.mau.mean() / (1 - df.mau.mean()))

    biv['woe'] = biv.logit - woe_geral
    biv['woe_sup'] = biv.logit_sup - woe_geral
    biv['woe_inf'] = biv.logit_inf - woe_geral

    fig, ax = plt.subplots(2,1, figsize=(8,6))
    ax[0].plot(biv[var], biv.woe, ':bo', label='woe')
    ax[0].plot(biv[var], biv.woe_sup, 'o:r', label='limite superior')
    ax[0].plot(biv[var], biv.woe_inf, 'o:r', label='limite inferior')
    
    num_cat = biv.shape[0]
    ax[0].set_xlim([-.3, num_cat-.7])

    ax[0].set_ylabel("Weight of Evidence")
    ax[0].legend(bbox_to_anchor=(.83, 1.17), ncol=3)
    
    ax[0].set_xticks(list(range(num_cat)))
    ax[0].set_xticklabels(biv[var], rotation=15)
    
    ax[1] = biv.cont.plot.bar()
    return biv
biv_discreta('estado_civil', df_3meses);
biv_discreta('tipo_renda', df_3meses)
df2 = df_3meses.copy()
df2.tipo_renda.replace({'Bolsista': 'St. ser.pub/bols.', 'Servidor público': 'St. ser.pub/bols.'}, inplace=True)
biv_discreta('tipo_renda', df2)
IV(df2.tipo_renda, df_3meses.mau)
biv_discreta('educacao', df2)
df2.educacao.replace({'Pós graduação':'Superior incompleto', 'Superior completo': 'Sup.Comp. / Pós'}, inplace=True)
biv_discreta('educacao', df2)
df2.educacao.replace({'Superior incompleto':'Sup. Comp e Inc. / Pós',
                      'Sup.Comp. / Pós' : 'Sup. Comp e Inc. / Pós',
                      'Fundamental':'Fund / Médio',
                      'Médio': 'Fund / Médio'
                     }, inplace=True)
biv_discreta('educacao', df2)
IV(df2.educacao, df.mau)
biv_discreta('tipo_residencia', df2)
df2
## Desenvolvimento do modelo

Desenvolva um modelo de *credit scoring* através de uma regressão logística.

- Trate valores missings e outliers
- Trate 'zeros estruturais'
- Faça agrupamentos de categorias conforme vimos em aula
- Proponha uma equação preditiva para 'mau'
- Caso hajam categorias não significantes, justifique
df2.isna().sum()
df2['tempo_emprego'] = df2['tempo_emprego'].fillna(0)
# ajuda para definir a equação da regressão
' + '.join(list(df2.columns))
### Ordenando por ordem decrescente as variáveis pelo IV:
metadados.sort_values(by='IV', ascending=False)
metadados.sort_values(by='IV', ascending=False)
### Selecionando os IV com valores maiores que 1%:
formula = '''
    mau ~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + tipo_renda + educacao + estado_civil + tipo_residencia + idade + tempo_emprego + qt_pessoas_residencia + renda'''

rl = smf.glm(formula, data=df2, family=sm.families.Binomial()).fit()

rl.summary()
formula = '''
    mau ~ posse_de_imovel + estado_civil + tipo_residencia + tempo_emprego + qt_pessoas_residencia + renda + sexo + tipo_renda
'''

rl = smf.glm(formula, data=df2, family=sm.families.Binomial()).fit()

rl.summary()
formula = '''
    mau ~ posse_de_imovel + tempo_emprego + renda + sexo + tipo_renda
'''

rl = smf.glm(formula, data=df2, family=sm.families.Binomial()).fit()

rl.summary()
## Avaliação do modelo

Avalie o poder discriminante do modelo pelo menos avaliando acurácia, KS e Gini.

Avalie estas métricas nas bases de desenvolvimento e *out of time*.
df2.isnull().sum()
df2['score'] = rl.predict(df2)

# Acurácia
acc = metrics.accuracy_score(df2.mau, df2.score>.068)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(df2.mau, df2.score)
auc = metrics.auc(fpr, tpr)
#Gini
gini = 2*auc -1
ks = ks_2samp(df2.loc[df2.mau == 1, 'score'], df2.loc[df2.mau != 1, 'score']).statistic

print('Acurácia: {0:.1%} \nAUC: {1:.1%} \nGINI: {2:.1%}\nKS: {3:.1%}'
      .format(acc, auc, gini, ks))
df_3meses.tipo_renda.replace({'Bolsista': 'St. ser.pub/bols.', 'Servidor público': 'St. ser.pub/bols.'}, inplace=True)
df_3meses.isna().sum()
df_3meses['tempo_emprego'] = df_3meses['tempo_emprego'].fillna(0)
df_3meses['score'] = rl.predict(df_3meses)

# Acurácia
acc = metrics.accuracy_score(df_3meses.mau, df_3meses.score>.068)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(df_3meses.mau, df_3meses.score)
auc = metrics.auc(fpr, tpr)
#Gini
gini = 2*auc -1
ks = ks_2samp(df_3meses.loc[df_3meses.mau == 1, 'score'], df_3meses.loc[df_3meses.mau != 1, 'score']).statistic

print('Acurácia: {0:.1%} \nAUC: {1:.1%} \nGINI: {2:.1%}\nKS: {3:.1%}'
      .format(acc, auc, gini, ks))
df.tipo_renda.replace({'Bolsista': 'St. ser.pub/bols.', 'Servidor público': 'St. ser.pub/bols.'}, inplace=True)
df.isna().sum()
df['tempo_emprego'] = df['tempo_emprego'].fillna(0)
df['score'] = rl.predict(df)

# Acurácia
acc = metrics.accuracy_score(df.mau, df.score>.068)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(df.mau, df.score)
auc = metrics.auc(fpr, tpr)
#Gini
gini = 2*auc -1
ks = ks_2samp(df.loc[df.mau == 1, 'score'], df.loc[df.mau != 1, 'score']).statistic

print('Acurácia: {0:.1%} \nAUC: {1:.1%} \nGINI: {2:.1%}\nKS: {3:.1%}'
      .format(acc, auc, gini, ks))
## Relatórios de Características:
def perfil_var(df, var, ev='mau', score='score', ncat=None):
    
    _df = df.copy()
    _df['ev'] = _df[ev]
    _df['nev'] = 1 - _df[ev]
    
    if ncat == None:
        g = _df.groupby(var)
    else:
        g = _df.groupby(pd.qcut(_df[var], ncat, duplicates='drop'))

    tg = g.agg({score:'mean', 'ev':'sum', 'nev':'sum'})
    tg['total'] = tg.ev + tg.nev
    tg['distribuição'] = tg.total/tg.total.sum()

    tg['total_acum'] = tg['total'].cumsum()
    tg['ev_acum']    = tg.ev.cumsum()
    tg['nev_acum']   = tg.nev.cumsum()

    tg['tx_ev']     = tg.ev/tg.total
    tg['ep']        = (tg.tx_ev*(1-tg.tx_ev)/tg.total)**.5
    tg['tx_ev_sup'] = tg.score+t.ppf([0.025], tg.total-1)*tg.ep
    tg['tx_ev_inf'] = tg.score+t.ppf([0.975], tg.total-1)*tg.ep

    fig, ax = plt.subplots()
    
    if ncat == None:
        ax.plot(tg.reset_index()[var], tg.reset_index(drop=True).score    , 'b-' , label='esperado')
        ax.plot(tg.reset_index()[var], tg.reset_index(drop=True).tx_ev    , 'r--', label='observado')
        ax.plot(tg.reset_index()[var], tg.reset_index(drop=True).tx_ev_sup, 'r:',  label='obs-ls')
        ax.plot(tg.reset_index()[var], tg.reset_index(drop=True).tx_ev_inf, 'r:',  label='obs-li')
    else:
        tg[var+'_med'] = g[var].mean()
        ax.plot(tg[var+'_med'], tg.reset_index(drop=True).score    , 'b-' , label='esperado')
        ax.plot(tg[var+'_med'], tg.reset_index(drop=True).tx_ev    , 'r--', label='observado')
        ax.plot(tg[var+'_med'], tg.reset_index(drop=True).tx_ev_sup, 'r:',  label='obs-ls')
        ax.plot(tg[var+'_med'], tg.reset_index(drop=True).tx_ev_inf, 'r:',  label='obs-li')
    return tg[['distribuição', score, 'tx_ev']]

tg = perfil_var(df2, 'tempo_emprego', ncat=5)
tg.reset_index().style.format({'score':'{:.1%}', 'tx_ev':'{:.1%}', 'distribuição':'{:.1%}'})
tg = perfil_var(df2, 'idade', ncat=5)
tg
perfil_var(df2, 'renda', ncat=5)
perfil_var(df2, 'score', ncat=5)
### Tabela de Ganhos:
df3 = df2.sort_values(by='score').reset_index().copy()
df3['tx_mau_acum'] = df3.mau.cumsum()/df3.shape[0]

df3['pct_mau_acum'] = df3.mau.cumsum()/df3.mau.sum()
df3['red_mau_acum'] = 1-df3.pct_mau_acum

df3['pct_aprovacao'] = np.array(range(df3.shape[0]))/df3.shape[0]
df3.head()
fig = px.line(df3, x="pct_aprovacao", y="tx_mau_acum", title='Taxa de maus por %aprovação')
fig.show()
fig = px.line(df3, x="pct_aprovacao", y="red_mau_acum", title='Redução na inadimplência por %aprovação')
fig.show()
df2['idade_cat']=pd.qcut(df2['idade'], 5, duplicates='drop')
tmp = df2.sort_values(by=['idade_cat','score'], ascending=True).copy()
tmp['freq']=1

tmp['freq_ac'] = tmp.groupby(['idade_cat'])['freq'].transform(lambda x: x.cumsum())
tmp['maus_ac'] = tmp.groupby(['idade_cat'])['mau'].transform(lambda x: x.cumsum())
tmp['freq_fx_idade'] = tmp.groupby(['idade_cat'])['freq'].transform(lambda x: x.sum())

tmp['pct_aprovados'] = tmp.freq_ac/tmp['freq_fx_idade']
tmp['tx_maus_pto_corte'] = tmp['maus_ac']/tmp['freq_ac']

tmp
fig = px.line(tmp, x="score", y="pct_aprovados", color='idade_cat', title='Taxa de maus por %aprovação')
fig.show()
# a - Criar um pipeline utilizando o sklearn pipeline para o preprocessamento 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from pycaret.classification import *
df_3meses.head()
X = df_3meses[['sexo',
               'posse_de_veiculo',
               'posse_de_imovel',
               'tipo_renda',
               'educacao',
               'estado_civil',
               'tipo_residencia',
               'qtd_filhos',
               'idade',
               'tempo_emprego',
               'qt_pessoas_residencia',
               'renda']]
X.info()
y = df_3meses['mau']
y.info()
#### Vamos separar 20% da base para testes (holdout) e 20% da base para validação. Os restantes 60% vamos utilizar para a base de treino.
X_, X_test, y_, y_test = train_test_split(X, y, test_size=.2, random_state=2360873)
X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=.25, random_state=1729)
## Pré processamento
### Substituição de nulos (nans)

Existe nulos na base? é dado numérico ou categórico? qual o valor de substituição? média? valor mais frequente? etc
variaveis_categoricas = [coluna for coluna in df_3meses.columns if df_3meses[coluna].dtypes.name == 'object']
variaveis_categoricas
variaveis_numericas = [coluna for coluna in df_3meses.columns if coluna not in variaveis_categoricas]
variaveis_numericas
# Definindo o tratamento para as colunas com variáveis categóricas (imputer = tratar dados faltantes)

imputer_cat = SimpleImputer(strategy='constant')
# Tratamento para variáveis numéricas:
imputer_num = SimpleImputer(strategy='median')
### Remoção de outliers

Como identificar outlier? Substituir o outlier por algum valor? Remover a linha?
# identificando outliers com o IsolationForest
outliers_pipe = IsolationForest(contamination=0.1, max_samples=100, random_state=0)
### Seleção de variáveis

Qual tipo de técnica? Boruta? Feature importance? 
pip install Boruta
from boruta import BorutaPy
rf = RandomForestClassifier( n_jobs=-1 )
boruta = BorutaPy(rf, n_estimators=1, verbose=2, random_state=42 )
### Redução de dimensionalidade (PCA)

Aplicar PCA para reduzir a dimensionalidade para 5
pca_pipe = PCA(n_components=5)
### Criação de dummies

Aplicar o get_dummies() ou onehotencoder() para transformar colunas catégoricas do dataframe em colunas de 0 e 1. 
- sexo
- posse_de_veiculo
- posse_de_imovel
- tipo_renda
- educacao
- estado_civil
- tipo_residencia
encoder_pipe = OneHotEncoder(handle_unknown='ignore', sparse=False)
### Pipeline 

Crie um pipeline contendo essas funções.

preprocessamento()
- substituicao de nulos
- remoção outliers
- PCA
- Criação de dummy de pelo menos 1 variável (posse_de_veiculo)
preprocessamento =Pipeline(steps=[('onehotencoder', encoder_pipe),
                     ('imputer_num', imputer_num),
                     ('imputer_cat', imputer_cat),
                     ('pca', pca_pipe),           
                     ('outliers', outliers_pipe)])
preprocessamento
#### Aplicando Pipeline na base de treino:
preprocessamento.fit(X_train.sample(frac=0.1),y_train.sample(frac=0.1) )
treino = preprocessamento.predict(X_train)
treino
#### Aplicando o Pipeline na Base de teste:
teste = preprocessamento.predict(X_test)
teste
preprocessamento.score_samples(X_train)
preprocessamento.score_samples(X_test)
# b - Pycaret na base de dados 

Utilize o pycaret para pre processar os dados e rodar o modelo **lightgbm**. Faça todos os passos a passos da aula e gere os gráficos finais. E o pipeline de toda a transformação.


import pandas as pd

df_py = pd.read_feather('credit_scoring.ftr')
df_py_2 = df_py.sample(40000)
df_py_2.isna().sum()
df_py_2.drop(['data_ref','index'], axis=1, inplace=True)
data = df_py_2.sample(frac=0.95, random_state=786)
data_unseen = df_py_2.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Conjunto de dados para modelagem (treino e teste): ' + str(data.shape))
print('Conjunto de dados não usados no treino/teste, apenas como validação: ' + str(data_unseen.shape))
exp_exercicio = setup(data = data, target = 'mau',
                  normalize=True, normalize_method='zscore', 
                  transformation=True, transformation_method = 'quantile',
                  fix_imbalance=True,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                  bin_numeric_features = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'])
data.dtypes
#forçando a variável qnt de filhos como numérica
data.qtd_filhos = data.qtd_filhos.astype(float)
exp_exercicio = setup(data = data, target = 'mau',
                  normalize=True, normalize_method='zscore', 
                  transformation=True, transformation_method = 'quantile',
                  fix_imbalance=True,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                  bin_numeric_features = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'])
best_model = compare_models(fold=10)
models()
### Light Gradient Boosting Machine
lightgbm = create_model('lightgbm')
#### Model tunning (Hyperparameter Tunning) - Lightgbm
tuned_lightgbm = tune_model(lightgbm, optimize='AUC')
### Analisando os resultados:
#### AUC Plot
plot_model(tuned_lightgbm, plot = 'auc')
plot_model(lightgbm, plot = 'auc')
#### Precision-Recall Plot
plot_model(tuned_lightgbm, plot = 'pr')
#### Importância das variáveis (Feature Importance) Plot
plot_model(tuned_lightgbm, plot='feature')
#### Matriz de confusão (Confusion matrix)
plot_model(tuned_lightgbm, plot = 'confusion_matrix')
evaluate_model(tuned_lightgbm)
final_lightgbm = finalize_model(tuned_lightgbm)
#Parâmetros finais do modelo Random Forest para deploy
print(final_lightgbm)
predict_model(final_lightgbm);
unseen_predictions = predict_model(final_lightgbm, data=data_unseen)
unseen_predictions.head()
unseen_predictions.dtypes
from pycaret.utils.generic import check_metric 

metric_result = check_metric(unseen_predictions['mau'], unseen_predictions['prediction_label'], metric='Accuracy')

print(metric_result)
save_model(final_lightgbm,'Final LIGHTGBC - Date 16-11-2023')
#### Carregando modelo gerado:
saved_final_gbc = load_model('Final LIGHTGBC - Date 16-11-2023')
new_prediction = predict_model(saved_final_gbc, data=data_unseen)
from pycaret.utils.generic import check_metric
check_metric(new_prediction['mau'], new_prediction['prediction_label'], metric = 'Accuracy')
saved_final_gbc.named_steps
