import pandas as pd
import os

# Diretório onde os arquivos CSV estão localizados
base_dir = r"C:\Users\aclarari\PYTHON\Exoplanetas\base"

# Dicionário com os nomes dos arquivos CSV e as respectivas datas
csv_files = {
    "q1q6koi.csv": "2013-02-12",
    "q1q8koi.csv": "2014-01-07",
    "q1q12koi.csv": "2014-12-04",
    "q1q16koi.csv": "2014-12-18",
    #"q1q17dr24koi.csv": "2015-09-24",
    #"q1q17dr25_koi.csv": "2017-08-31",
    #"q1q17dr25supkoi.csv": "2018-09-27",
    #"cumulative.csv": "2018-09-27"
}

# Lista para armazenar os DataFrames carregados
dfs = []

# Carregar cada arquivo CSV, adicionar a coluna de data e armazenar na lista dfs
for csv_file, date_str in csv_files.items():
    file_path = os.path.join(base_dir, csv_file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(date_str)
        dfs.append(df)
    else:
        print(f"Arquivo não encontrado: {file_path}")

# Juntar todos os DataFrames da lista dfs em um único DataFrame
result = pd.concat(dfs)

# Salvar o resultado em um novo arquivo CSV
output_path = os.path.join(base_dir, "bases.csv")
result.to_csv(output_path, index=False)
