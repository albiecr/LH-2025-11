<img width="1467" height="462" alt="Image" src="https://github.com/user-attachments/assets/74c294a0-0ccd-4712-9d8c-2c9afcd4594f" />

# Desafio de Ciência de Dados - PProductions

## Introdução

Este projeto consiste na resolução de um desafio de Ciência de Dados, cujo objetivo é analisar um conjunto de dados de filmes para orientar a empresa **PProductions** sobre o tipo de filme que deve ser o próximo a ser desenvolvido. O projeto inclui uma análise exploratória de dados (EDA) detalhada e o desenvolvimento de um modelo preditivo para a nota do IMDb de um filme.

## Estrutura do Projeto

A pasta `Data` contém os scripts principais do projeto:

- `training_model.py`: Script para treinar o modelo de regressão e salvá-lo no formato `.pkl`.
- `predict_rating.py`: Script para carregar o modelo treinado e fazer uma previsão de nota para um filme específico.
- `desafio_indicium_imdb.csv`: O conjunto de dados original utilizado para a análise e o treinamento do modelo.

## Pré-requisitos

Para executar este projeto, você precisa ter o **Python** (versão 3.9 ou superior) instalado em sua máquina.

## Instalação

Siga os passos abaixo para configurar o ambiente e instalar as dependências necessárias.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/albiecr/LH-2025-11.git](https://github.com/albiecr/LH-2025-11.git)
    cd LH-2025-11
    ```

2.  **Crie e ative um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python -m venv venv
    # No Windows
    .\venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Como Executar o Projeto

1.  **Treinamento do Modelo:**
    Primeiro, você deve treinar o modelo para gerar o arquivo `predictive_model.pkl`. Execute o seguinte comando a partir da pasta `LH-2025-11/Data`:
    ```bash
    python training_model.py
    ```
    Isso irá treinar o modelo e salvar o arquivo `predictive_model.pkl` na mesma pasta.

2.  **Previsão de Nota:**
    Em seguida, você pode usar o modelo salvo para prever a nota de um filme. Execute o seguinte comando a partir da pasta `LH-2025-11/Data`:
    ```bash
    python predict_rating.py
    ```
    O script irá carregar o modelo, processar os dados do filme "The Shawshank Redemption" e imprimir a nota prevista no terminal.
