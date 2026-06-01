import pandas as pd

from afisare import afiseaza_rezultate
from data_processing import pregateste_date
from logistic_regression import (
    calculate_accuracy,
    get_model_data,
    predict_probability,
    train_logistic_regression,
)
from ml_signal import adauga_ml_signal
from train_test import (
    adauga_ml_label_dupa_mediana_train,
    imparte_train_test,
    normalizeaza_featureuri,
)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # 1. pregatim datele
    data = pregateste_date()

    # 2. impartim datele in train/test
    train_data, test_data = imparte_train_test(data)

    # 3. facem target-ul ML dupa mediana din train
    train_data, test_data, ml_threshold = adauga_ml_label_dupa_mediana_train(
        train_data,
        test_data,
    )

    # 4. normalizam feature-urile
    train_data, test_data = normalizeaza_featureuri(train_data, test_data)

    # 5. pregatim datele pentru Logistic Regression
    X_train, y_train = get_model_data(train_data)
    X_test, y_test = get_model_data(test_data)

    # 6. antrenam modelul
    w, b = train_logistic_regression(X_train, y_train)

    # 7. calculam probabilitatile
    train_probabilities = predict_probability(X_train, w, b)
    test_probabilities = predict_probability(X_test, w, b)

    train_data["ml_probability"] = train_probabilities
    test_data["ml_probability"] = test_probabilities

    # 8. facem semnalul discret pentru pasul urmator
    train_data = adauga_ml_signal(train_data)
    test_data = adauga_ml_signal(test_data)

    # 9. verificam acuratetea
    train_accuracy = calculate_accuracy(train_probabilities, y_train)
    test_accuracy = calculate_accuracy(test_probabilities, y_test)

    # 10. afisam ce am obtinut
    afiseaza_rezultate(
        data,
        train_data,
        test_data,
        w,
        b,
        ml_threshold,
        train_accuracy,
        test_accuracy,
    )


if __name__ == "__main__":
    main()
