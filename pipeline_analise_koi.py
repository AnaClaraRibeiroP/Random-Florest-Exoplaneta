# Importando bibliotecas necessárias
import pandas as pd  # Pandas para manipulação de dados
from sklearn.model_selection import train_test_split  # Para dividir o conjunto de dados em treinamento e teste
from sklearn.ensemble import RandomForestClassifier  # Modelo RandomForest para classificação
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas de avaliação
from sklearn.preprocessing import LabelEncoder  # Para codificar variáveis categóricas
from sklearn.calibration import CalibratedClassifierCV  # Para calibrar as probabilidades do modelo

# Carregando o CSV a partir do caminho fornecido
file_path = r'C:\Users\aclarari\PYTHON\teste - Copia.csv'
df = pd.read_csv(file_path)

# Removendo colunas desnecessárias (se houver)
df = df.drop(['rowid'], axis=1)

# Convertendo a coluna 'koi_disposition' para labels numéricas
label_encoder = LabelEncoder()
df['koi_disposition'] = label_encoder.fit_transform(df['koi_disposition'])

# Identificando colunas categóricas e aplicando a codificação one-hot
categorical_columns = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Separando os dados em recursos (X) e rótulos (y)
X = df_encoded.drop(['koi_disposition'], axis=1)
y = df_encoded['koi_disposition']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando o modelo
rf_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Relatório de classificação
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred))

# Matriz de confusão
print('\nMatriz de Confusão:')
print(confusion_matrix(y_test, y_pred))

# Calibrando o modelo para obter as probabilidades
calibrated_model = CalibratedClassifierCV(rf_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train, y_train)

# Obtendo probabilidades no conjunto de teste
probabilities = calibrated_model.predict_proba(X_test)

# Extraindo a probabilidade para a classe 'CANDIDATE'
candidate_probabilities = probabilities[:, label_encoder.transform(['CANDIDATE'])[0]]

# Adicionando as probabilidades ao DataFrame original
df['candidate_probability'] = calibrated_model.predict_proba(X)[:, label_encoder.transform(['CANDIDATE'])[0]]

# Defina um limiar para decidir entre "confirmed" e "false positive"
limiar = 0.95

# Adiciona uma coluna ao DataFrame indicando se é "confirmed" ou "false positive"
df['prediction'] = ['confirmed' if prob >= limiar else 'false positive' for prob in df['candidate_probability']]

# Exibindo as previsões e probabilidades para as linhas marcadas como 'CANDIDATE'
candidate_rows = df[df['koi_disposition'] == label_encoder.transform(['CANDIDATE'])[0]]
print('\nPrevisões e Probabilidades para linhas marcadas como "CANDIDATE":')
print(candidate_rows[['koi_disposition', 'candidate_probability', 'prediction']])

# Salvando os resultados em um arquivo CSV
output_file_path = r'C:\Users\aclarari\PYTHON\resultado.csv'
df.to_csv(output_file_path, index=False)
print(f'\nResultados salvos em {output_file_path}')
