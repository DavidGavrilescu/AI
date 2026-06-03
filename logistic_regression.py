import numpy as np

from config import (
    EPOCHS,
    LEARNING_RATE,
    NORMALIZED_FEATURE_COLUMNS,
    THRESHOLD_PREDICTIE,
)

# transforma scorul z in probabilitate intre 0 si 1
def g(z):
    return 1 / (1 + np.exp(-z))


def get_date_model(data):
    X = data[NORMALIZED_FEATURE_COLUMNS].values
    y = data["ml_label"].values

    return X, y


def train_logistic_regression(X, y):
    numar_exemple, numar_featureuri = X.shape

    w = np.zeros(numar_featureuri) # ponderile sunt 0 la inceput
    b = 0
    alpha = LEARNING_RATE

    for _ in range(EPOCHS):
        # h_w(x) = w^T * f(x) + b
        h_w_x = X @ w + b

        # g(z) = sigmoid(h_w(x))
        g_z = g(h_w_x)

        # eroarea folosita in gradient descent
        delta_E = g_z - y

        # X.T intoarce matricea ca sa putem calcula gradientul pentru fiecare pondere
        transpusa_X = X.T

        dw = transpusa_X @ delta_E / numar_exemple
        db = np.mean(delta_E)

        # gradient descent
        w = w - alpha * dw
        b = b - alpha * db

    return w, b

# X este matricea de features, o inmultim cu ponderile w si adaugam bias-ul
def prezicere_probabilitati(X, w, b):
    return g(X @ w + b)


# returneaza procentul de predictii corecte
def calculeaza_acuratetea(probabilitati, y):
    """
    y = [1, 0, 1, 0]
    predictie_y = [1, 0, 0, 1]
    corecte = [True, True, False, False]
    corecte.mean() -> (2/4) * 100 = 50.0
    """

    # clasa prezisa dupa pragul de decizie
    predictie_y = probabilitati >= THRESHOLD_PREDICTIE
    corecte = predictie_y == y

    return corecte.mean() * 100


def ruleaza_logistic_regression(train_data, test_data):
    X_train, y_train = get_date_model(train_data)
    X_test, y_test = get_date_model(test_data)

    w, b = train_logistic_regression(X_train, y_train)

    probabilitati_training = prezicere_probabilitati(X_train, w, b)
    probabilitati_test = prezicere_probabilitati(X_test, w, b)

    acuratete_training = calculeaza_acuratetea(probabilitati_training, y_train)
    acuratete_test = calculeaza_acuratetea(probabilitati_test, y_test)

    return (
        w,
        b,
        probabilitati_training,
        probabilitati_test,
        acuratete_training,
        acuratete_test,
    )
