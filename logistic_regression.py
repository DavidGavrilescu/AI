import numpy as np

from config import (
    EPOCHS,
    LEARNING_RATE,
    NORMALIZED_FEATURE_COLUMNS,
    THRESHOLD_PREDICTIE,
)


def sigmoid(z):
    # transforma scorul in probabilitate intre 0 si 1
    return 1 / (1 + np.exp(-z))


def get_model_data(data):
    # X are feature-urile, y are raspunsul corect
    X = data[NORMALIZED_FEATURE_COLUMNS].values
    y = data["ml_label"].values

    return X, y


def train_logistic_regression(X, y):
    # w sunt ponderile, b este bias-ul
    numar_exemple = X.shape[0]
    numar_featureuri = X.shape[1]

    w = np.zeros(numar_featureuri) # ponderi
    b = 0 # bias

    for _ in range(EPOCHS):
        # z = combinatia liniara dintre feature-uri si ponderi
        z = X @ w + b

        # probabilitatea prezisa de model
        g_z = sigmoid(z)

        # cat de departe e predictia de raspunsul real
        error = g_z - y

        # gradientii spun in ce directie schimbam w si b
        dw = X.T @ error / numar_exemple
        db = np.mean(error)

        # w si b se actualizeaza cu gradient descent
        w = w - LEARNING_RATE * dw
        b = b - LEARNING_RATE * db

    return w, b


def predict_probability(X, w, b):
    # probabilitatea ca randamentul pe 5 zile sa fie peste mediana
    z = X @ w + b
    return sigmoid(z)


def calculate_accuracy(probabilities, y):
    # peste THRESHOLD_PREDICTIE inseamna ca modelul prezice 1
    predictii = probabilities >= THRESHOLD_PREDICTIE
    corecte = predictii == y

    return corecte.mean() * 100

# functia principala
def ruleaza_model_lr(train_data, test_data):
    X_train, y_train = get_model_data(train_data)
    X_test, y_test = get_model_data(test_data)

    w, b = train_logistic_regression(X_train, y_train)

    train_probabilities = predict_probability(X_train, w, b)
    test_probabilities = predict_probability(X_test, w, b)

    train_accuracy = calculate_accuracy(train_probabilities, y_train)
    test_accuracy = calculate_accuracy(test_probabilities, y_test)

    return (
        w,
        b,
        train_probabilities,
        test_probabilities,
        train_accuracy,
        test_accuracy,
    )
