from config import (
    FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    SIGNAL_MARGIN,
    THRESHOLD_PREDICTIE,
    TICKER,
)


def calculeaza_baseline(data):
    # baseline = cat obtinem daca prezicem mereu clasa majoritara
    return data["ml_label"].value_counts(normalize=True).max() * 100


def afiseaza_rezultate(
    data,
    train_data,
    test_data,
    w,
    b,
    ml_threshold,
    train_accuracy,
    test_accuracy,
):
    print(f"Ticker analizat: {TICKER}")
    print(f"Randuri totale dupa curatare: {len(data)}")
    print(
        f"Train: {train_data['Date'].min().date()} -> "
        f"{train_data['Date'].max().date()} ({len(train_data)} randuri)"
    )
    print(
        f"Test:  {test_data['Date'].min().date()} -> "
        f"{test_data['Date'].max().date()} ({len(test_data)} randuri)"
    )

    print("\nFeature-uri folosite pentru model:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print("\nTarget Logistic Regression:")
    print("ml_label = 1 daca future_return_5d este peste mediana din train")
    print("ml_label = 0 daca future_return_5d este sub sau egal cu mediana din train")
    print("modelul invata daca randamentul pe 5 zile este peste normalul din train")

    print("\nExemplu de feature original + feature normalizat:")
    exemplu = train_data[
        [
            "Date",
            "daily_return",
            "daily_return_norm",
            "volume_change",
            "volume_change_norm",
            "ma_ratio",
            "ma_ratio_norm",
            "ml_label",
        ]
    ].head(5).copy()
    exemplu = exemplu.round({
        "daily_return": 2,
        "daily_return_norm": 2,
        "volume_change": 2,
        "volume_change_norm": 2,
        "ma_ratio": 2,
        "ma_ratio_norm": 2,
    })
    print(exemplu.to_string(index=False))

    print("\nPonderi Logistic Regression:")
    for column, weight in zip(NORMALIZED_FEATURE_COLUMNS, w):
        print(f"{column}: {weight:.6f}")

    train_counts = train_data["ml_label"].value_counts().to_dict()
    test_counts = test_data["ml_label"].value_counts().to_dict()

    print(f"\nBias: {b:.6f}")
    print(f"Prag ml_label, calculat pe train: {ml_threshold:.6f}")
    print(f"Train ml_label 0 / 1: {train_counts.get(0, 0)} / {train_counts.get(1, 0)}")
    print(f"Test ml_label 0 / 1:  {test_counts.get(0, 0)} / {test_counts.get(1, 0)}")
    print(f"Train baseline accuracy: {calculeaza_baseline(train_data):.2f}%")
    print(f"Test baseline accuracy:  {calculeaza_baseline(test_data):.2f}%")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy:  {test_accuracy:.2f}%")

    lower_threshold = THRESHOLD_PREDICTIE - SIGNAL_MARGIN
    upper_threshold = THRESHOLD_PREDICTIE + SIGNAL_MARGIN

    print("\nRegula pentru ml_signal:")
    print(f"ml_signal = 0 daca ml_probability < {lower_threshold:.2f}")
    print(f"ml_signal = 1 daca ml_probability este intre {lower_threshold:.2f} si {upper_threshold:.2f}")
    print(f"ml_signal = 2 daca ml_probability > {upper_threshold:.2f}")

    print("\nPrimele predictii si semnale pe test:")
    predictii = test_data[
        ["Date", "future_return_5d", "ml_label", "ml_probability", "ml_signal"]
    ].head(10).copy()
    predictii["future_return_5d"] = predictii["future_return_5d"].round(4)
    predictii["ml_probability"] = predictii["ml_probability"].round(4)
    print(predictii.to_string(index=False))
