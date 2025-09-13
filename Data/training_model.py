import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import re

# 1. Carregar e Pré-processar os Dados
desafio_indicium_imdb = pd.read_csv('desafio_indicium_imdb.csv')
#Limpeza de dados
desafio_indicium_imdb['Gross'] = desafio_indicium_imdb['Gross'].astype(str).str.replace(',', '', regex=True)
desafio_indicium_imdb['Gross'] = pd.to_numeric(desafio_indicium_imdb['Gross'], errors='coerce')
desafio_indicium_imdb['Runtime'] = desafio_indicium_imdb['Runtime'].astype(str).str.replace(' min', '', regex=True)
desafio_indicium_imdb['Runtime'] = pd.to_numeric(desafio_indicium_imdb['Runtime'], errors='coerce')
desafio_indicium_imdb['Released_Year'] = pd.to_numeric(desafio_indicium_imdb['Released_Year'], errors='coerce')

# Remover dados faltantes nas colunas de interesse
desafio_indicium_imdb.dropna(subset=['IMDB_Rating', 'No_of_Votes', 'Meta_score', 'Runtime', 'Gross', 'Director', 'Genre'], inplace=True)

# 2. Definir Variáveis (Features e Target)
features = ['Released_Year', 'Runtime', 'Genre', 'Director', 'No_of_Votes', 'Gross', 'Meta_score']
target = 'IMDB_Rating'

X = desafio_indicium_imdb[features]
y = desafio_indicium_imdb[target]

# 3. Engenharia de Recursos (One-Hot Encoding para variáveis categóricas)
categorical_features = ['Genre', 'Director']
numerical_features = ['Released_Year', 'Runtime', 'No_of_Votes', 'Gross', 'Meta_score']

# Criar um pré-processador para as colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', SimpleImputer(strategy='median'), numerical_features) # Imputa NaNs com a mediana
    ])

# 4. Criar o Pipeline do Modelo
# Um pipeline é usado para encadear as etapas de pré-processamento e o modelo
model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

# 5. Treinar e Avaliar o Modelo
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Treinar o modelo
model_pipeline.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model_pipeline.predict(X_test)

# Avaliar o desempenho do modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE do modelo: {rmse:.4f}")
    
# 6. Salvar o Modelo
model_filename = 'predictive_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"Modelo salvo com sucesso no arquivo '{model_filename}'")
