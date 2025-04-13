# Databricks notebook source
# MAGIC %md
# MAGIC # ESAVI: 2024
# MAGIC
# MAGIC
# MAGIC Neste projeto vamos explorar os dados do ESAVI, que correspondem aos dados de Eventos Supostamente Atribuiveis a Vacinação.
# MAGIC
# MAGIC Virou uma triste rotina sermos bombardeados por falsas informações na internet. Grande parte dessas notícias se baseiam em teorias da conspiração onde há uma verdade supostamente escondida. Na saúde não é diferente, temos observado um panorama em que cresce o número de informações falsas sobre as vacinas circulando nas redes sociais. Ignorando que há sistemas de controle estabelecidos para a aprovação de uma vacina à população bem como de acompanhamento das reações adversas, o que torna a vacinação um processo bastante seguro e necessário à saúde humana.
# MAGIC
# MAGIC Este projeto tem por proposito analisar os dados públicos sobre as possíveis reações adversas fortalecendo o conhecimento sobre as vacinas. Reforçando (1) que não há dados escondidos sobre as vacinas e (2) que existe um monitoramento regular sobre possíveis reações adversas. Com isso espero reforçar que não há vacinas inseguras sendo disponibilizadas para a população.
# MAGIC
# MAGIC Os dados do ESAVI estão disponibilizados de forma pública em uma base de dados única. O dicionário dos dados pode ser encontrado no seguinte [link](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/ESAVI/DicionariodeDados.pdf).
# MAGIC
# MAGIC Vamos iniciar o projeto fazendo o upload dos dados via DBFS no databricks. Posteriormente vamos efetuar a modelagem desses dados transformando esses dados em tabelas distintas para facilitar a análise de dados, que será a última etapa do projeto.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

# MAGIC %pip install pydrive --quiet
# MAGIC

# COMMAND ----------

!pip install pyspark --quiet

# COMMAND ----------

!pip install requests --quiet


# COMMAND ----------

!apt-get install graphviz -y
!pip install graphviz


# COMMAND ----------

!pip install pydot


# COMMAND ----------

# imports necessarios
import requests
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import unicodedata
import re
import pandas as pd
import requests
import zipfile
import io
import graphviz
from graphviz import Digraph
from IPython.display import Image, display
from pyspark.sql.functions import col, to_date, year


# COMMAND ----------

#criando a sessao e o contexto que será usado
spark = SparkSession.builder.master("local[*]").getOrCreate()
conf = SparkConf().set('spark.ui.port', '4050').setAppName("srag").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)

# COMMAND ----------

# DEPRECADO por que fiquei com receio do professor não conseguir ler o arquivo.
## Lê o arquivo CSV com Spark
#df1 = spark.read.format("csv") \
#    .option("header", "true") \
#    .option("sep", "|") \
#    .load("dbfs:/FileStore/shared_uploads/melo.katiaselene@gmail.com/esavi-1.csv")

## Converte para DataFrame do pandas
#esavi_data = df1.toPandas()

## Exibe as primeiras linhas
#esavi_data.head()


# COMMAND ----------


# Função para normalizar e remover caracteres especiais
def remover_caracteres_especiais(texto):
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFKD', texto)
        texto = texto.encode('ASCII', 'ignore').decode('utf-8')  # Remove acentos
        texto = re.sub(r'[^\w\s]', '', texto)  # Remove caracteres especiais restantes
    return texto


# COMMAND ----------



# URL do ZIP no GitHub (exemplo público ou do seu repo)
url = "https://github.com/skmelo/MVP_Estrutura_de_Dados/raw/main/esavi.zip"

# Faz o download do ZIP
response = requests.get(url)

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print("Arquivos no ZIP:", z.namelist())  # Lista os arquivos

        # Lê o CSV com codificação correta e separador "|"
        with z.open("esavi.csv") as f:
            df = pd.read_csv(f, sep="|", encoding="latin1")
            print(df.head())
else:
    print("Erro no download:", response.status_code)



# COMMAND ----------

df.head()

# COMMAND ----------

esavi_data = df.applymap(remover_caracteres_especiais)
esavi_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Agora vamos obter a descrição da tabela de dados com o intuito de entender melhor esses dados para o processo de modelagem.

# COMMAND ----------

#Puxando o nome das colunas individualmente
esavi_data.columns

# COMMAND ----------

# Checando se há linhas com dados vazios
esavi_data["nu_notificacao"].isnull().any()

# COMMAND ----------

esavi_data["nu_notificacao"].value_counts()


# COMMAND ----------

# MAGIC %md
# MAGIC No comando acima conseguimos identificar que tem algumas linhas com valores incorretos para o campo nu_notificacao. Vamos entender com mais detalhes.

# COMMAND ----------

valores = ["BELGIUM"]

# Remove espaços extras das strings
valores = [val.strip() for val in valores]

# Filtra as linhas onde a coluna 'nu_notificacao' contém algum dos valores
resultado = esavi_data[esavi_data["nu_notificacao"].str.contains('|'.join(valores), na=False)]

# Exibe o resultado
print(resultado)


# COMMAND ----------

# MAGIC %md
# MAGIC Considerando que são apenas 19 linhas, e esse padrão também foi encontrado no csv, por simplicidade vamos fazer a remoção dessas linhas.

# COMMAND ----------

# Remove as linhas onde a coluna 'nu_notificacao' contém algum dos valores
esavi_data_filtrado = esavi_data[~esavi_data["nu_notificacao"].str.contains('|'.join(valores), na=False)]



# COMMAND ----------

# checando se deu certo a remocao
esavi_data_filtrado["nu_notificacao"].value_counts()


# COMMAND ----------

esavi_data_filtrado.describe(include='all')


# COMMAND ----------

# MAGIC %md
# MAGIC Agora os dados do dataset parecem mais coerentes. Vamos seguir para a modelagem desse banco de dados.
# MAGIC
# MAGIC

# COMMAND ----------

esavi_data_filtrado.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelagem do banco de Dados
# MAGIC
# MAGIC ### Modelo Conceitual
# MAGIC Analisando a base de dados podemos definir como entidades:
# MAGIC - Evento: aspectos relacionados à ocorrência de evento adverso, gravidade e conclusão da investigação.
# MAGIC     - nu_notificacao (primary key),dt_notificacao, dt_recebimento_notificacao,ds_situacao_notificacao, dt_desfecho, ds_atendimento_medico, dt_investigacao, ds_evolucao_caso, ds_evento_adverso, ds_casualidade, ds_conduta, ds_diagnostico, dt_encerramento,ds_encerramento_grave,ds_tipo_encerramento, ds_class_gravidade_ea, ds_reacao_ea, ds_dia_duracao_ea 
# MAGIC - Vacinas: vacina e outros medicamentos registrados como possiveis motivadores do evento adverso.
# MAGIC     - ds_nome_fabricante, ds_estrategia_imuno, dt_aplicacao_imuno, ds_relacao_imuno, ds_lote_imuno, ds_dose_imuno, ds_imuno, co_imuno
# MAGIC - Medicamentos:
# MAGIC     - ds_medicamento_uso, ds_relacao_medicamento, ds_medicamento
# MAGIC - Paciente: dados demograficos do paciente que teve o efeito adverso registrado
# MAGIC     - ds_sexo, nu_mes_gestante, ds_mulher_amamentando, ds_profissional_seguranca,ds_estrangeiro,st_comunidade_tradicional, ds_raca_cor_mae, ds_gestante, nu_idade, ds_raca_cor, ds_profissional_saude,ds_doencas_pre_existentes
# MAGIC - Localidade: dados da unidade de saude e localidade.
# MAGIC     - no_estado, no_municipio, no_mun_notificacao,no_estado_notificacao, ds_tipo_atendimento

# COMMAND ----------


# Cria o grafo
dot = Digraph(comment='Modelo Conceitual ESAVI')

# Entidades
dot.node('Evento', 'Evento')
dot.node('Vacina', 'Vacina')
dot.node('Medicamento', 'Medicamento')
dot.node('Paciente', 'Paciente')
dot.node('Localidade', 'Localidade')

# Relacionamentos
dot.edge('Evento', 'Vacina', label='1:N')
dot.edge('Evento', 'Medicamento', label='1:N')
dot.edge('Evento', 'Paciente', label='1:1')
dot.edge('Evento', 'Localidade', label='1:1')

# Caminho completo no DBFS
output_path = "/dbfs/FileStore/shared_uploads/melo.katiaselene@gmail.com/modelo_conceitual_esavi"

# Renderiza o diagrama
dot.render(output_path, format='png', cleanup=True)

print("Diagrama salvo em:", output_path + ".png")


# COMMAND ----------


display(Image("/dbfs/FileStore/shared_uploads/melo.katiaselene@gmail.com/modelo_conceitual_esavi.png"))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelo lógico
# MAGIC
# MAGIC Agora que criamos o modelo teórico vamos definir o modelo lógico com base nas colunas que definimos acima.

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS katiamelo_mvp_engenharia_dados")
spark.sql("USE katiamelo_mvp_engenharia_dados")


# COMMAND ----------

# Tabela Evento
spark.sql("""
CREATE TABLE IF NOT EXISTS evento (
    nu_notificacao STRING,
    dt_notificacao DATE,
    dt_recebimento_notificacao DATE,
    ds_situacao_notificacao STRING,
    dt_desfecho DATE,
    ds_atendimento_medico STRING,
    dt_investigacao DATE,
    ds_evolucao_caso STRING,
    ds_evento_adverso STRING,
    ds_casualidade STRING,
    ds_conduta STRING,
    ds_diagnostico STRING,
    dt_encerramento DATE,
    ds_encerramento_grave STRING,
    ds_tipo_encerramento STRING,
    ds_class_gravidade_ea STRING,
    ds_reacao_ea STRING,
    ds_dia_duracao_ea INT
)
USING DELTA
""")

# Tabela Vacina
spark.sql("""
CREATE TABLE IF NOT EXISTS vacina (
    nu_notificacao STRING,
    ds_nome_fabricante STRING,
    ds_estrategia_imuno STRING,
    dt_aplicacao_imuno DATE,
    ds_relacao_imuno STRING,
    ds_lote_imuno STRING,
    ds_dose_imuno STRING,
    ds_imuno STRING,
    co_imuno STRING
)
USING DELTA
""")

# Tabela Medicamento
spark.sql("""
CREATE TABLE IF NOT EXISTS medicamento (
    nu_notificacao STRING,
    ds_medicamento_uso STRING,
    ds_relacao_medicamento STRING,
    ds_medicamento STRING
)
USING DELTA
""")

# Tabela Paciente
spark.sql("""
CREATE TABLE IF NOT EXISTS paciente (
    nu_notificacao STRING,
    ds_sexo STRING,
    nu_mes_gestante INT,
    ds_mulher_amamentando STRING,
    ds_profissional_seguranca STRING,
    ds_estrangeiro STRING,
    st_comunidade_tradicional STRING,
    ds_raca_cor_mae STRING,
    ds_gestante STRING,
    nu_idade INT,
    ds_raca_cor STRING,
    ds_profissional_saude STRING,
    ds_doencas_pre_existentes STRING
)
USING DELTA
""")

# Tabela Localidade
spark.sql("""
CREATE TABLE IF NOT EXISTS localidade (
    nu_notificacao STRING,
    no_estado STRING,
    no_municipio STRING,
    no_mun_notificacao STRING,
    no_estado_notificacao STRING,
    ds_tipo_atendimento STRING
)
USING DELTA
""")


# COMMAND ----------

#checando se as tabelas foram criadas
spark.sql("SHOW TABLES").show(truncate=False)


# COMMAND ----------

#preparando o arquivo para fazer o carregamento nas tabelas
esavi_spark = spark.createDataFrame(esavi_data_filtrado)


# COMMAND ----------

# MAGIC %md
# MAGIC Agora vamos adicionar os dados esavi nas tabelas criadas

# COMMAND ----------

# Salvando dados na tabela evento
evento_cols = [
    "nu_notificacao", "dt_notificacao", "dt_recebimento_notificacao",
    "ds_situacao_notificacao", "dt_desfecho", "ds_atendimento_medico",
    "dt_investigacao", "ds_evolucao_caso", "ds_evento_adverso",
    "ds_casualidade", "ds_conduta", "ds_diagnostico", "dt_encerramento",
    "ds_encerramento_grave", "ds_tipo_encerramento", "ds_class_gravidade_ea",
    "ds_reacao_ea", "ds_dia_duracao_ea"
]
evento_df = esavi_spark.select(evento_cols)
evento_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("evento")


# COMMAND ----------

# checando se os dados foram armazenados corretamente
spark.sql("SELECT * FROM evento LIMIT 10").show()

# COMMAND ----------

# Salvando dados na tabela vacina
vacina_cols = [
    "nu_notificacao", "ds_nome_fabricante", "ds_estrategia_imuno",
    "dt_aplicacao_imuno", "ds_relacao_imuno", "ds_lote_imuno",
    "ds_dose_imuno", "ds_imuno", "co_imuno"
]
vacina_df = esavi_spark.select(vacina_cols)
vacina_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("vacina")


# COMMAND ----------

# checando se os dados foram armazenados corretamente
spark.sql("SELECT * FROM vacina LIMIT 10").show()

# COMMAND ----------

# Salvando dados na tabela vacina
medicamento_cols = [
    "nu_notificacao", "ds_medicamento_uso", "ds_relacao_medicamento", "ds_medicamento"
]
medicamento_df = esavi_spark.select(medicamento_cols)
medicamento_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("medicamento")


# COMMAND ----------

# checando se os dados foram armazenados corretamente
spark.sql("SELECT * FROM medicamento LIMIT 10").show()

# COMMAND ----------

paciente_cols = [
    "nu_notificacao", "ds_sexo", "nu_mes_gestante", "ds_mulher_amamentando",
    "ds_profissional_seguranca", "ds_estrangeiro", "st_comunidade_tradicional",
    "ds_raca_cor_mae", "ds_gestante", "nu_idade", "ds_raca_cor",
    "ds_profissional_saude", "ds_doencas_pre_existentes"
]
paciente_df = esavi_spark.select(paciente_cols)
paciente_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("paciente")


# COMMAND ----------

# checando se os dados foram armazenados corretamente
spark.sql("SELECT * FROM paciente LIMIT 10").show()

# COMMAND ----------

localidade_cols = [
    "nu_notificacao", "no_estado", "no_municipio", 
    "no_mun_notificacao", "no_estado_notificacao", "ds_tipo_atendimento"
]
localidade_df = esavi_spark.select(localidade_cols)
localidade_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("localidade")


# COMMAND ----------

# checando se os dados foram armazenados corretamente
spark.sql("SELECT * FROM localidade LIMIT 10").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analise de Dados

# COMMAND ----------

# MAGIC %md
# MAGIC Como há uma limitação de tempo para a execução dessa investigação responderemos apenas algumas perguntas básicas:
# MAGIC - **Qual o volume de casos de reações adversas anual?**
# MAGIC - **Qual o tipo de evento adverso**?
# MAGIC - **Qual o imuno mais teve evento adverso?** 
# MAGIC - **Do volume de eventos adversos anuais quantos foram graves?**
# MAGIC - **Os eventos graves envolviam doenças pre existentes?**
# MAGIC - **Qual a prevalência de casos entre os sexos?**
# MAGIC - **Qual a prevalência de casos entre as raças?**
# MAGIC - **Qual a conduta adotada para os casos?**
# MAGIC - **Qual foi o diagnotisco dos casos?**
# MAGIC
# MAGIC Há diversas outras perguntas que poderiam ser respondidas com os dados, vamos nos concentrar nessas mais simples por ora, considerando que o escopo do MVP não é um aprofundamento muito grande nas análises de dados.

# COMMAND ----------

# MAGIC %md
# MAGIC **Qual o volume de casos de reações adversas anual?**
# MAGIC
# MAGIC Como podemos observar abaixo os eventos adversos relacionados a vacinação são bastante raros. Aumentou consideravelmente no período pós pandemia, o que pode estar influenciado por fatores como: (1) desconfiança da população em relação a vacina da covid-19 bem como ao fato de que a vacinação foi emergencial dado as características da pandemia.
# MAGIC
# MAGIC Vamos investigar especificamente os últimos anos de 2021-2024 e verificar quais foram os tipos de eventos adversos e sua gravidade.

# COMMAND ----------


# Converte a coluna 'dt_notificacao' para o tipo DATE usando o formato 'ddMMyyyy'
evento_df = evento_df.withColumn('dt_notificacao_date', to_date(col('dt_notificacao'), 'ddMMyyyy'))

volume_anual_df = evento_df.select(
    year(col('dt_notificacao_date')).alias('ano'),
    col('nu_notificacao')
).groupBy('ano').count().withColumnRenamed('count', 'volume_casos')

# Exibir os resultados
volume_anual_df.orderBy(col('ano').desc()).toPandas()


# COMMAND ----------

# Converte 'dt_notificacao' para tipo DATE
evento_df = evento_df.withColumn(
    'dt_notificacao_date',
    to_date(col('dt_notificacao'), 'ddMMyyyy')
)

# Extrai o ano
evento_df = evento_df.withColumn('ano', year(col('dt_notificacao_date')))

# Cria coluna categórica baseada em 'ds_encerramento_grave'
evento_df = evento_df.withColumn(
    'condicao_grave',
    when(col('ds_encerramento_grave').rlike("(?i)sim"), 'grave').otherwise('nao grave')
)

# Filtra os anos de interesse
evento_filtrado = evento_df.filter((col('ano') >= 2021) & (col('ano') <= 2024))

# Pivotando com a nova coluna
volume_pivotado = evento_filtrado.groupBy('ano') \
    .pivot('condicao_grave') \
    .agg(count('nu_notificacao')) \
    .orderBy(col('ano').desc())

# Exibe como tabela
volume_pivotado.toPandas()


# COMMAND ----------


volume_percentual = volume_pivotado.withColumn(
    "percentual_grave",
    expr("grave * 100.0 / (grave + `nao grave`)")
)
volume_percentual.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC Como podemos observar acima a maior parte dos eventos adversos registrados nesses anos não eram graves.

# COMMAND ----------

# Converte a coluna 'dt_notificacao' de string para date
evento_df = evento_df.withColumn(
    'dt_notificacao_date',
    to_date(col('dt_notificacao'), 'ddMMyyyy')
)

# Cria coluna de ano
evento_df = evento_df.withColumn('ano', year(col('dt_notificacao_date')))

# Filtra apenas os anos desejados (de 2021 a 2024)
evento_filtrado = evento_df.filter((col('ano') >= 2021) & (col('ano') <= 2024))

# Pivotando os dados: colunas = valores únicos de ds_conduta, linhas = anos
volume_pivotado = evento_filtrado.groupBy('ano') \
    .pivot('ds_conduta') \
    .agg(count('nu_notificacao')) \
    .orderBy(col('ano').desc())

# Converte para Pandas para exibir como tabela
volume_pivotado.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC Ao abrir em relação a conduta a grande maioria não teve contra-indicação do esquema vacinal

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusão
# MAGIC
# MAGIC Acredito que os pre-requisitos para o projeto foram satisfeitos. Devido ao tempo não foi possível explorar mais a fundo os dados e responder mais questões. 
# MAGIC
# MAGIC Analisando os dados que obtivemos até o momento foi possível observar que a maior parte dos eventos adversos foram registrados no período posterior a pandemia de covid-19. No ano de 2021 foi o momento de maior pico de registros, o que potencialmente está relacionado ao fato de que a vacinação foi uma medida emergencial para a contenção da pandemia. O que aumentou também a vigilância em cima desses potenciais eventos. A grande maior dos eventos não foram graves, bem como não guiaram a uma recomendação de suspensão do esquema vacinal.
# MAGIC
# MAGIC Podemos observar que o volume de eventos adversos antes desse período é praticamente irrisório. O que reforça a segurança das vacinas no nosso país. Adicionalmente, conseguimos observar que desde 2021 há uma tendência consideravel de queda do volume de eventos adversos.
# MAGIC

# COMMAND ----------

