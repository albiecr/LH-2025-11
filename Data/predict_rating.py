import pandas as pd
import joblib
import re
import numpy as np

# Este script utiliza um modelo de Machine Learning treinado para prever a nota de um filme no IMDb.
# O modelo 'predictive_model.pkl' deve ser treinado e salvo previamente.

try:
    # 1. Carregar o modelo de Machine Learning
    # O modelo é carregado a partir do arquivo .pkl para ser usado na previsão.
    model_pipeline = joblib.load('predictive_model.pkl')
    print("Modelo de previsão carregado com sucesso.")

    # 2. Preparar os dados do novo filme para a previsão
    # As características do novo filme precisam ser formatadas exatamente como os dados de treinamento.
    new_movie_data = {
        'Series_Title': 'The Shawshank Redemption',
        'Released_Year': '1994',
        'Certificate': 'A',
        'Runtime': '142 min',
        'Genre': 'Drama',
        'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'Meta_score': 80.0,
        'Director': 'Frank Darabont',
        'Star1': 'Tim Robbins',
        'Star2': 'Morgan Freeman',
        'Star3': 'Bob Gunton',
        'Star4': 'William Sadler',
        'No_of_Votes': 2343110,
        'Gross': '28,341,469'
    }
    
    # Converter os dados de entrada para um DataFrame, que é o formato esperado pelo modelo.
    new_movie_df = pd.DataFrame([new_movie_data])

    # 3. Limpar e transformar os dados
    # Aplica as mesmas etapas de pré-processamento usadas durante o treinamento do modelo.
    new_movie_df['Gross'] = new_movie_df['Gross'].astype(str).str.replace(',', '', regex=True)
    new_movie_df['Gross'] = pd.to_numeric(new_movie_df['Gross'], errors='coerce')
    
    new_movie_df['Runtime'] = new_movie_df['Runtime'].astype(str).str.replace(' min', '', regex=True)
    new_movie_df['Runtime'] = pd.to_numeric(new_movie_df['Runtime'], errors='coerce')
    
    new_movie_df['Released_Year'] = pd.to_numeric(new_movie_df['Released_Year'], errors='coerce')

    # Selecionar as features que o modelo espera para a previsão.
    features = ['Released_Year', 'Runtime', 'Genre', 'Director', 'No_of_Votes', 'Gross', 'Meta_score']
    X_new = new_movie_df[features]

    # 4. Fazer a Previsão da Nota do IMDb
    # O método 'predict' aplica todas as transformações do pipeline e faz a previsão.
    predicted_rating = model_pipeline.predict(X_new)

    print(f"A nota do IMDb prevista para 'The Shawshank Redemption' é: {predicted_rating[0]:.2f}")

except FileNotFoundError:
    print("Erro: O arquivo 'predictive_model.pkl' não foi encontrado. Por favor, treine o modelo primeiro para gerar o arquivo.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")