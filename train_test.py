from config import FEATURE_COLUMNS, TRAIN_RATIO


def imparte_train_test(data):
    # primele 70% train, ultimele 30% test
    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data.iloc[:split_index].copy().reset_index(drop=True)
    test_data = data.iloc[split_index:].copy().reset_index(drop=True)

    return train_data, test_data


def normalizeaza_featureuri(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    # normalizarea se calculeaza doar pe train
    for column in FEATURE_COLUMNS:
        train_min = train_data[column].min()
        train_max = train_data[column].max()
        norm_column = column + "_norm"

        train_data[norm_column] = (train_data[column] - train_min) / (train_max - train_min)
        test_data[norm_column] = (test_data[column] - train_min) / (train_max - train_min)

    return train_data, test_data
