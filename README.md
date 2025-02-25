# Predição de Compras com Machine Learning

Este projeto utiliza algoritmos de **Machine Learning** para prever se um cliente concluirá uma compra online com base em dados de navegação e comportamento. O código foi desenvolvido como parte do curso **CS50 - Introduction to Artificial Intelligence with Python**, oferecido por **Harvard University**.

## Tecnologias Utilizadas  
- **Linguagem**: Python 3  
- **Bibliotecas**:  
  - `scikit-learn` (KNN, divisão de dados treino/teste)  
  - `csv` (manipulação de arquivos CSV)  
  - `calendar` (processamento de datas)  
- **Conceitos de IA/ML**:  
  - Classificação com **K-vizinhos mais próximos (k=1)**  
  - Métricas de avaliação: Sensibilidade e Especificidade  
  - Pré-processamento de dados (codificação categórica)  
  
## Estrutura do Projeto

A pasta contém os seguintes arquivos:

- **`shopping.py`**: Implementação do modelo de aprendizado de máquina para previsão de compras.
- **`shopping.csv`**: Conjunto de dados contendo informações sobre sessões de usuários em um site de compras, fornicido pelo CS50.

## Minha Contribuição

A implementação do modelo de **Machine Learning** para prever compras foi desenvolvida no arquivo `shopping.py`, incluindo o processamento dos dados e a avaliação do modelo.

## Features  
- **Análise de comportamento de compra**: Processa 17 características de navegação (tempo em páginas, taxas de rejeição, tipo de visitante, etc).  
- **Conversão inteligente de dados**:  
  - Transforma meses abreviados (Jan, Feb...) em índices numéricos  
  - Codifica variáveis categóricas (`VisitorType`, `Weekend`) em valores binários  
- **Modelo preditivo**: Classifica se uma sessão resultará em receita (1) ou não (0).  
- **Avaliação detalhada**: Calcula:  
  - Taxa de Verdadeiros Positivos (Sensibilidade)  
  - Taxa de Verdadeiros Negativos (Especificidade)  
- **Interface CLI**: Execução via linha de comando com parâmetros personalizáveis.  

## Como Funciona a Predição

O modelo utiliza um **classificador k-Nearest Neighbors (k-NN)** para determinar a probabilidade de um usuário efetuar uma compra. Os dados analisados incluem:

- Quantidade de páginas visitadas
- Tempo gasto em cada categoria
- Taxa de rejeição (bounce rate)
- Período do ano (mês da visita)
- Fonte de tráfego

O modelo é treinado com exemplos rotulados e avaliado com base na precisão das previsões.

## Instalação e execução:

1. Clone o repositório:
```bash
git clone https://github.com/Alvaro-Sena/compra.git  
```
2. Navegue até a pasta do repositório:
```bash
cd compra/
```
3. Instale as dependências:
```bash
   pip install -r requirements.txt
```
4. Execute o projeto:
```bash
python shopping.py shopping.csv
```

Certifique-se de que possui **Python 3** instalado no seu ambiente. Além disso, é necessário instalaçao das bibliotecas de requirements.txt.

## Contato
Caso tenha dúvidas ou sugestões, entre em contato através do meu [LinkedIn](www.linkedin.com/in/alvaro-sena), [GitHub](https://github.com/Alvaro-Sena) ou [WhatsApp](https://wa.me/447356040385).
