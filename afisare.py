from config import FEATURE_COLUMNS, NORMALIZED_FEATURE_COLUMNS, TICKER


def afiseaza_rezumat(data, train_data, test_data):
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


def afiseaza_featureuri():
    print("\nFeature-uri folosite pentru model:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


def afiseaza_exemplu_normalizare(train_data):
    print("\nExemplu de feature original + feature normalizat:")

    ROTUNJIRE = 2
    
    columns_to_show = [
        "Date",
        "daily_return",
        "daily_return_norm",
        "volume_change",
        "volume_change_norm",
        "ma_ratio",
        "ma_ratio_norm",
        "label",
    ]

    example = train_data[columns_to_show].head(5).copy()
    example = example.round({
        "daily_return": ROTUNJIRE,
        "daily_return_norm": ROTUNJIRE,
        "volume_change": ROTUNJIRE,
        "volume_change_norm": ROTUNJIRE,
        "ma_ratio": ROTUNJIRE,
        "ma_ratio_norm": ROTUNJIRE,
    })

    print(example.to_string(index=False))


def afiseaza_logistic_regression(w, b, train_accuracy, test_accuracy):
    print("\nPonderi Logistic Regression:")
    for column, weight in zip(NORMALIZED_FEATURE_COLUMNS, w):
        print(f"{column}: {weight:.6f}")

    print(f"\nBias: {b:.6f}")
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy:  {test_accuracy:.2f}%")


def afiseaza_predictii_test(test_data):
    print("\nPrimele predictii pe test:")

    predictii = test_data[["Date", "ml_probability", "label"]].head(10).copy()
    predictii["ml_probability"] = predictii["ml_probability"].round(4)

    print(predictii.to_string(index=False))
