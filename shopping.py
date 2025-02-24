import csv
import calendar
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Verificar argumentos da linha de comando
    if len(sys.argv) != 2:
        sys.exit("Uso: python shopping.py dados")

    # Carregar dados da planilha e dividir em conjuntos de treino e teste
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Treinar o modelo e fazer previsões
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Exibir resultados
    print(f"Corretos: {(y_test == predictions).sum()}")
    print(f"Incorretos: {(y_test != predictions).sum()}")
    print(f"Taxa de Verdadeiro Positivo: {100 * sensitivity:.2f}%")
    print(f"Taxa de Verdadeiro Negativo: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Carregar dados de compras de um arquivo CSV `filename` e converter em uma lista de
    listas de evidências e uma lista de rótulos. Retornar uma tupla (evidence, labels).

    evidence deve ser uma lista de listas, onde cada lista contém os
    seguintes valores, nesta ordem:
        - Administrative, um número inteiro
        - Administrative_Duration, um número de ponto flutuante
        - Informational, um número inteiro
        - Informational_Duration, um número de ponto flutuante
        - ProductRelated, um número inteiro
        - ProductRelated_Duration, um número de ponto flutuante
        - BounceRates, um número de ponto flutuante
        - ExitRates, um número de ponto flutuante
        - PageValues, um número de ponto flutuante
        - SpecialDay, um número de ponto flutuante
        - Month, um índice de 0 (Janeiro) a 11 (Dezembro)
        - OperatingSystems, um número inteiro
        - Browser, um número inteiro
        - Region, um número inteiro
        - TrafficType, um número inteiro
        - VisitorType, um número inteiro 0 (não retornando) ou 1 (retornando)
        - Weekend, um número inteiro 0 (se falso) ou 1 (se verdadeiro)

    labels deve ser a lista correspondente de rótulos, onde cada rótulo
    é 1 se a Receita for verdadeira, e 0 caso contrário.
    """
    mes = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    mes['June'] = mes.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                mes[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Dada uma lista de listas de evidências e uma lista de rótulos, retorna um
    modelo de k-vizinhos mais próximos ajustado (k=1) treinado com os dados.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Dada uma lista de rótulos reais e uma lista de rótulos previstos,
    retorna uma tupla (sensibilidade, especificidade).

    Assume-se que cada rótulo seja 1 (positivo) ou 0 (negativo).

    `sensibilidade` deve ser um valor de ponto flutuante de 0 a 1
    representando a "taxa de verdadeiro positivo": a proporção de
    rótulos positivos reais que foram identificados corretamente.

    `especificidade` deve ser um valor de ponto flutuante de 0 a 1
    representando a "taxa de verdadeiro negativo": a proporção de
    rótulos negativos reais que foram identificados corretamente.
    """
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1

    sensitivity /= total_positive
    specificity /= total_negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()
