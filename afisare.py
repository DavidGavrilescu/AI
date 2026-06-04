from config import (
    FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    SIGNAL_MARGIN,
    THRESHOLD_PREDICTIE,
    TICKER,
)


def calculeaza_baseline(date):
    # baseline: acuratetea daca prezicem mereu clasa majoritara
    return date["ml_label"].value_counts(normalize=True).max() * 100

def afiseaza_interval(nume, date):
    print(
        f"{nume}: {date['Date'].min().date()} -> "
        f"{date['Date'].max().date()} ({len(date)} randuri)"
    )

def afiseaza_rezultate(
    date,
    train_data,
    test_data,
    w,
    b,
    prag_mediana_train,
    acuratete_training,
    acuratete_test,
):
    print(f"ticker: {TICKER}")
    print(f"randuri dupa curatare: {len(date)}")
    afiseaza_interval("train", train_data)
    afiseaza_interval("test", test_data)

    print("\nfeature-uri LR:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print("\nLR:")
    print(f"target: future_return_5d > mediana train ({prag_mediana_train:.6f})")
    print(
        f"baseline train/test: "
        f"{calculeaza_baseline(train_data):.2f}% / "
        f"{calculeaza_baseline(test_data):.2f}%"
    )
    print(f"accuracy train/test: {acuratete_training:.2f}% / {acuratete_test:.2f}%")
    print(f"bias: {b:.6f}")

    print("\nponderi LR:")
    for coloana, pondere in zip(NORMALIZED_FEATURE_COLUMNS, w):
        print(f"{coloana}: {pondere:.6f}")

    prag_jos = THRESHOLD_PREDICTIE - SIGNAL_MARGIN
    prag_sus = THRESHOLD_PREDICTIE + SIGNAL_MARGIN

    print("\nml_signal:")
    print(f"0 daca prob < {prag_jos:.2f}")
    print(f"1 daca prob e intre {prag_jos:.2f} si {prag_sus:.2f}")
    print(f"2 daca prob > {prag_sus:.2f}")
