import numpy as np

from config import EPOCHS, LEARNING_RATE, NORMALIZED_FEATURE_COLUMNS, THRESHOLD_PREDICTIE


def sigmoid(z):
    # g(z) = sigmoid(z)
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

    w = np.zeros(numar_featureuri) # initializam ponderile cu 0
    b = 0 # initializam bias-ul cu 0

    for _ in range(EPOCHS):
        # z = combinatia liniara dintre feature-uri si ponderi
        # @ = operatorul de produs matriceal
        z = X @ w + b

        # g_z = g(z)
        g_z = sigmoid(z)

        # error = diferenta dintre predictie si raspunsul real
        error = g_z - y

        # gradientii ne spun cum schimbam w si b
        dw = X.T @ error / numar_exemple
        db = np.mean(error)

        # w si b se actualizeaza cu gradient descent
        w = w - LEARNING_RATE * dw
        b = b - LEARNING_RATE * db

    return w, b


def predict_probability(X, w, b):
    # calculam probabilitatea ca 
    # randamentul pe 5 zile sa fie peste mediana din train
    z = X @ w + b
    return sigmoid(z)


def calculate_accuracy(probabilities, y):
    # peste THRESHOLD_PREDICTIE inseamna ca modelul prezice 1
    prediction = probabilities >= THRESHOLD_PREDICTIE
    correct = prediction == y

    return correct.mean() * 100 # returnam acuratetea in procente
