import pandas as pd

from afisare import (
    afiseaza_exemplu_normalizare,
    afiseaza_featureuri,
    afiseaza_rezumat,
)
from data_processing import pregateste_date
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

    # 4. afisam ce am obtinut
    afiseaza_rezumat(data, train_data, test_data)
    afiseaza_featureuri()
    afiseaza_exemplu_normalizare(train_data)


if __name__ == "__main__":
    main()
