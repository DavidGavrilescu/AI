from config import (
    FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    TRAIN_RATIO,
)


def split_date(data):
    # primele 70% train, ultimele 30% test
    index_split = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:index_split].copy().reset_index(drop=True)
    test_data = data.iloc[index_split:].copy().reset_index(drop=True)

    return train_data, test_data


def adauga_ml_label(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    # pragul este mediana randamentului viitor din train
    prag_mediana_train = train_data["future_return_5d"].median()

    # ml_label este y, adica eticheta reala pentru Logistic Regression
    train_data["ml_label"] = (
        train_data["future_return_5d"] > prag_mediana_train
    ).astype(int)
    test_data["ml_label"] = (
        test_data["future_return_5d"] > prag_mediana_train
    ).astype(int)

    return train_data, test_data, prag_mediana_train

"""
normalizeaza coloanele de features folosind min-max normalization
- Datele originale (Open, High, Low, Close, Volume)
- future_return_5d (randamentul viitor pe 5 zile)
- ml_label (1 daca future_return_5d > prag_mediana_train, altfel 0)
"""
def normalizeaza_features(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    # min si max din train ca sa nu influenteze testul
    for coloana, coloana_norm in zip(FEATURE_COLUMNS, NORMALIZED_FEATURE_COLUMNS):
        minim_train = train_data[coloana].min()
        maxim_train = train_data[coloana].max()
        diferenta = maxim_train - minim_train

        # x_norm = (x - min_train) / (max_train - min_train)
        train_data[coloana_norm] = (
            train_data[coloana] - minim_train
        ) / diferenta
        test_data[coloana_norm] = (
            test_data[coloana] - minim_train
        ) / diferenta

    return train_data, test_data

def pregateste_train_test(data):
    train_data, test_data = split_date(data)
    train_data, test_data, prag_mediana_train = adauga_ml_label(
        train_data,
        test_data,
    )
    train_data, test_data = normalizeaza_features(train_data, test_data)

    return train_data, test_data, prag_mediana_train
