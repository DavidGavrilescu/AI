import pandas as pd

from afisare import (
    afiseaza_exemplu_normalizare,
    afiseaza_featureuri,
    afiseaza_logistic_regression,
    afiseaza_predictii_test,
    afiseaza_praguri_ml,
    afiseaza_rezumat,
    afiseaza_semnale_ml,
)
from data_processing import pregateste_date
from logistic_regression import (
    calculate_accuracy,
    get_model_data,
    predict_probability,
    train_logistic_regression,
)
from ml_signal import adauga_ml_signal, calculeaza_praguri_ml
from train_test import imparte_train_test, normalizeaza_featureuri


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # 1. pregatim datele
    data = pregateste_date()

    # 2. impartim datele in train/test
    train_data, test_data = imparte_train_test(data)

    # 3. normalizam feature-urile
    train_data, test_data = normalizeaza_featureuri(train_data, test_data)

    # 4. pregatim datele pentru Logistic Regression
    X_train, y_train = get_model_data(train_data)
    X_test, y_test = get_model_data(test_data)

    # 5. antrenam modelul
    w, b = train_logistic_regression(X_train, y_train)

    # 6. calculam probabilitatile
    train_probabilities = predict_probability(X_train, w, b)
    test_probabilities = predict_probability(X_test, w, b)

    train_data["ml_probability"] = train_probabilities
    test_data["ml_probability"] = test_probabilities

    # 7. transformam probabilitatea in semnal discret
    lower_threshold, upper_threshold = calculeaza_praguri_ml(train_data)
    train_data = adauga_ml_signal(train_data, lower_threshold, upper_threshold)
    test_data = adauga_ml_signal(test_data, lower_threshold, upper_threshold)

    # 8. verificam acuratetea
    train_accuracy = calculate_accuracy(train_probabilities, y_train)
    test_accuracy = calculate_accuracy(test_probabilities, y_test)

    # 9. afisam ce am obtinut
    afiseaza_rezumat(data, train_data, test_data)
    afiseaza_featureuri()
    afiseaza_exemplu_normalizare(train_data)
    afiseaza_logistic_regression(w, b, train_accuracy, test_accuracy)
    afiseaza_predictii_test(test_data)
    afiseaza_praguri_ml(lower_threshold, upper_threshold)
    afiseaza_semnale_ml(test_data)


if __name__ == "__main__":
    main()
