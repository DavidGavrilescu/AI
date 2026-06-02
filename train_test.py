from config import (
    FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    TRAIN_RATIO,
)


def imparte_train_test(data):
    # primele 70% train, ultimele 30% test
    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy().reset_index(drop=True)
    test_data = data.iloc[split_index:].copy().reset_index(drop=True)

    return train_data, test_data


def adauga_ml_label(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    # SPY are bias de crestere, deci folosim mediana din train ca prag
    # pragul se calculeaza doar pe train ca sa evitam data leakage
    ml_threshold = train_data["future_return_5d"].median()

    train_data["ml_label"] = (
        train_data["future_return_5d"] > ml_threshold
    ).astype(int)
    test_data["ml_label"] = (
        test_data["future_return_5d"] > ml_threshold
    ).astype(int)

    return train_data, test_data, ml_threshold


def normalizeaza_features(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    # normalizarea se calculeaza doar pe train
    for column, norm_column in zip(FEATURE_COLUMNS, NORMALIZED_FEATURE_COLUMNS):
        train_min = train_data[column].min()
        train_max = train_data[column].max()
        diferenta = train_max - train_min

        train_data[norm_column] = (train_data[column] - train_min) / diferenta
        test_data[norm_column] = (test_data[column] - train_min) / diferenta

    return train_data, test_data


def pregateste_train_test(data):
    train_data, test_data = imparte_train_test(data)
    train_data, test_data, ml_threshold = adauga_ml_label(
        train_data,
        test_data,
    )
    train_data, test_data = normalizeaza_features(train_data, test_data)

    return train_data, test_data, ml_threshold
