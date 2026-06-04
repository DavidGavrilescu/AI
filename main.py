import pandas as pd

from afisare import afiseaza_rezultate
from benchmark import compara_strategii
from data_processing import pregateste_date
from logistic_regression import ruleaza_logistic_regression
from ml_signal import adauga_ml_signal
from q_learning import train_q_learning
from train_test import pregateste_train_test


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # pregatim datele
    date = pregateste_date()

    # train/test + target pentru LR
    train_data, test_data, prag_mediana_train = pregateste_train_test(date)

    # antrenam Logistic Regression
    (
        w,
        b,
        probabilitati_training,
        probabilitati_test,
        acuratete_training,
        acuratete_test,
    ) = ruleaza_logistic_regression(train_data, test_data)

    train_data["ml_probability"] = probabilitati_training
    test_data["ml_probability"] = probabilitati_test

    # transformam probabilitatea in semnal discret
    train_data = adauga_ml_signal(train_data)
    test_data = adauga_ml_signal(test_data)

    # q-learning foloseste semnalul LR + trendul
    q_table = train_q_learning(train_data)

    # q-learning fara ML signal foloseste doar pozitia + trendul
    train_data_fara_ml = train_data.copy()
    train_data_fara_ml["ml_signal"] = 1
    q_table_fara_ml = train_q_learning(train_data_fara_ml)

    # output scurt pentru model
    afiseaza_rezultate(
        date,
        train_data,
        test_data,
        w,
        b,
        prag_mediana_train,
        acuratete_training,
        acuratete_test,
    )

    # comparam strategiile pe test
    rezultate_benchmark = compara_strategii(test_data, q_table, q_table_fara_ml)
    print("\nbenchmark:")
    print(rezultate_benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
