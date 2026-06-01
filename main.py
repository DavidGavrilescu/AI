import pandas as pd

from afisare import (
    afiseaza_exemplu_normalizare,
    afiseaza_featureuri,
    afiseaza_logistic_regression,
    afiseaza_predictii_test,
    afiseaza_rezumat,
)
from data_processing import pregateste_date
from logistic_regression import (
    calculate_accuracy,
    get_model_data,
    predict_probability,
    train_logistic_regression,
)
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

    # 7. verificam acuratetea
    train_accuracy = calculate_accuracy(train_probabilities, y_train)
    test_accuracy = calculate_accuracy(test_probabilities, y_test)

    # 8. afisam ce am obtinut
    afiseaza_rezumat(data, train_data, test_data)
    afiseaza_featureuri()
    afiseaza_exemplu_normalizare(train_data)
    afiseaza_logistic_regression(w, b, train_accuracy, test_accuracy)
    afiseaza_predictii_test(test_data)


if __name__ == "__main__":
    main()
