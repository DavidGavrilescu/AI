import pandas as pd

from afisare import afiseaza_rezultate
from benchmark import compara_benchmark
from data_processing import pregateste_date
from logistic_regression import ruleaza_model_lr
from ml_signal import adauga_ml_signal
from q_learning import train_q_learning
from train_test import pregateste_train_test


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # pregatim datele
    data = pregateste_date()

    # train/test + target pentru LR
    train_data, test_data, ml_threshold = pregateste_train_test(data)

    # antrenam Logistic Regression
    (
        w,
        b,
        train_probabilities,
        test_probabilities,
        train_accuracy,
        test_accuracy,
    ) = ruleaza_model_lr(train_data, test_data)

    train_data["ml_probability"] = train_probabilities
    test_data["ml_probability"] = test_probabilities

    # transformam probabilitatea in semnal discret
    train_data = adauga_ml_signal(train_data)
    test_data = adauga_ml_signal(test_data)

    # q-learning foloseste semnalul LR + trendul
    q_table = train_q_learning(train_data)

    # output scurt pentru model
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

    # comparam doar strategiile finale
    benchmark_results = compara_benchmark(test_data, q_table)
    print("\nbenchmark:")
    print(benchmark_results.to_string(index=False))


if __name__ == "__main__":
    main()
