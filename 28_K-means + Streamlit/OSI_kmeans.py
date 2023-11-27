
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Online_Shoppers_Intentions', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write('### Agrupamento hierárquico')

    st.write('### Neste projeto vamos usar a base online shoppers purchase intention de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). Web Link. A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.')

    st.write('|Variavel                |Descrição          |\n'
             '|------------------------|:-------------------|\n'
             '|Administrative          | Quantidade de acessos em páginas administrativas|\n' 
             '|Administrative_Duration | Tempo de acesso em páginas administrativas | \n'
             '|Informational           | Quantidade de acessos em páginas informativas  | \n'
             '|Informational_Duration  | Tempo de acesso em páginas informativas  | \n'
             '|ProductRelated          | Quantidade de acessos em páginas de produtos | \n'
             '|ProductRelated_Duration | Tempo de acesso em páginas de produtos | \n'
             '|BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | \n'
             '|ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | \n'
             '|PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | \n'
             '|SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | \n'
             '|Month                   | Mês  | \n'
             '|OperatingSystems        | Sistema operacional do visitante | \n'
             '|Browser                 | Browser do visitante | \n'
             '|Region                  | Região |\n '
             '|TrafficType             | Tipo de tráfego                  | \n'
             '|VisitorType             | Tipo de visitante: novo ou recorrente | \n'
             '|Weekend                 | Indica final de semana | \n'
             '|Revenue                 | Indica se houve compra ou não |\n'
              '* variávels calculadas pelo google analytics.')
    st.markdown("---")
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Online Shoppers Intentions', \
        page_icon = './img/OSI_02.jpeg',
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write('# Online Shoppers Intentions - Análise por Agrupamento')
    st.markdown("---")
    
    # Apresenta a imagem na barra lateral da aplicação
    image = Image.open("./img/OSI_01.jpeg")
    st.sidebar.image(image)
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Online_Shoppers_Intention", type = ['csv','xlsx'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df = load_data(data_file_1)
        
        st.write(df.head(100))
        
        df.Revenue.value_counts(dropna=False)
        
        st.write('## Análise descritiva')
        st.write('### Verificado a distribuição dessas variáveis')
        
        df.describe()
        
        st.write('### Verificado se existe valores *missing*')

        df.isnull().sum()

        nullcount = df.isnull().sum()
        st.write(print('Total de valores nulos e vazios no dataset:', nullcount.sum()))
                
        st.write('### Verificando valores únicos de cada variável')

        uniques = df.nunique(axis=0)
        st.write(print(uniques))
        
        st.markdown("---")
        
        st.write('#### Variáveis Descartadas Nessa Análise:')
        
        df_clean = df.drop(['Month','Browser','OperatingSystems','Region','TrafficType','Weekend', 'VisitorType'], axis=1)
        
        st.markdown("---")
        
        st.write('#### Transfromando as variaveis qualitativas em dummies:')
        
        df_2 = pd.get_dummies(df_clean.dropna())
        df_2.info()
        
       
        st.write('## Número de grupos')
        st.write('### Neste projeto vamos adotar uma abordagem bem pragmática e avaliar agrupamentos hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do diretor da empresa.')
       
        st.write('### 3 Grupos:')
        
        cluster = KMeans(n_clusters=3, random_state=0)
        cluster.fit(df_pad_escopo)

        cluster.labels_

        df_pad['grupos'] = pd.Categorical(cluster.labels_)
        df_pad_escopo['grupos'] = pd.Categorical(cluster.labels_)
        
        
        st.write('### Visualizando a distribuição pelo seaborn - pairplot:')

        sns.pairplot(df_pad_escopo, hue='grupos')
        
        # st.write('### 4 Grupos:')
        
        # df_2['grupo_4'] = fcluster(Z, 4, criterion='maxclust')
        # df_2.grupo_4.value_counts()
        
        # st.markdown("---")
        
        # st.write('#### Unificando os dataframes')
        
        # df_date = df.reset_index().merge(df_2.reset_index(), how='left')
        # df_2.grupo_3.value_counts()
        # df_2.grupo_4.value_counts()
        
        # st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos Durante a Semana e nos Finais de Semana:')
        
        # df_date.groupby(['Weekend','Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby([ 'Weekend','Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)
        
        # st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos nos Meses do Ano:')
        
        # df_date.groupby([ 'Month','Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby([ 'Month','Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby([ 'SpecialDay','Month', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby([ 'SpecialDay','Month', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)
        
        # st.write('### Análise Descritiva com Relação a Navegação no Site pelos Grupos:')
        
        # df_date.groupby([ 'ProductRelated', 'Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby([ 'Administrative', 'Revenue',  'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # df_date.groupby(['Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)

        # st.write('#### Com relação a escolha da quantidade de grupos, entre 3 e 4, no meu ponto de vista, a escolha pelo número de 4 grupos será mais efetiva. Pois, quando se adiciona 1 grupo a mais, além dos 3 pretendidos, percebe-se que o grupo 3 é subdividido em 2, ficando o grupo 3 condensado apenas nas pessoas que utilizaram os sites no mês de maio, com o restante sendo realocado no grupo 4. E nesse grupo, apenas em 3 ocasiões (em fevereiro) as compras foram efetivadas com sucesso.')
        # st.write('#### A relação da quantidade de acesso em páginas administrativas com a efetuação de compra não possui muita relação. Porém, poucas pessoas com poucos acessos a páginas de produtos finalizam a compra de forma efetiva. No geral, é necessários mais acessos as páginas de produtos para que seja finalizada a compra com sucesso.')
        
        st.markdown("---")
        
        st.write('## Avaliação de resultados:')
        
        df_pad_escopo.groupby(['Revenue', 'grupos'])['Revenue'].count().unstack().fillna(0).style.format(precision=0)
        
        ax = df_pad.groupby(['grupos_3', 'Revenue'])['Revenue'].count().unstack().plot.bar()

        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.5),
          ncol=3, fancybox=True, shadow=True);

        df['grupos_3'] = df_pad['grupos_3']

        pd.crosstab(df['grupos_3'], df['Revenue'])
        
        st.write('#### De acordo com as análises, o grupo 2 é mas propenso a efetuar a compra do que os outros.')
        
        
if __name__ == '__main__':
	main()
    









