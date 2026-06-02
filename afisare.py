from config import (
    FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    SIGNAL_MARGIN,
    THRESHOLD_PREDICTIE,
    TICKER,
)


def calculeaza_baseline(data):
    # baseline = cat luam daca prezicem mereu clasa majoritara
    return data["ml_label"].value_counts(normalize=True).max() * 100


def afiseaza_interval(nume, data):
    print(
        f"{nume}: {data['Date'].min().date()} -> "
        f"{data['Date'].max().date()} ({len(data)} randuri)"
    )


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
    print(f"ticker: {TICKER}")
    print(f"randuri dupa curatare: {len(data)}")
    afiseaza_interval("train", train_data)
    afiseaza_interval("test", test_data)

    print("\nfeature-uri LR:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print("\nLR:")
    print(f"target: future_return_5d > mediana train ({ml_threshold:.6f})")
    print(
        f"baseline train/test: "
        f"{calculeaza_baseline(train_data):.2f}% / "
        f"{calculeaza_baseline(test_data):.2f}%"
    )
    print(f"accuracy train/test: {train_accuracy:.2f}% / {test_accuracy:.2f}%")
    print(f"bias: {b:.6f}")

    print("\nponderi LR:")
    for column, weight in zip(NORMALIZED_FEATURE_COLUMNS, w):
        print(f"{column}: {weight:.6f}")

    lower_threshold = THRESHOLD_PREDICTIE - SIGNAL_MARGIN
    upper_threshold = THRESHOLD_PREDICTIE + SIGNAL_MARGIN

    print("\nml_signal:")
    print(f"0 daca prob < {lower_threshold:.2f}")
    print(f"1 daca prob e intre {lower_threshold:.2f} si {upper_threshold:.2f}")
    print(f"2 daca prob > {upper_threshold:.2f}")
