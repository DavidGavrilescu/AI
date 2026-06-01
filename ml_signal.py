import numpy as np


def calculeaza_praguri_ml(train_data):
    # pragurile se calculeaza doar pe train
    # 0-33% -> bearish, 33-66% -> neutral, 66-100% -> bullish
    lower_threshold = np.percentile(train_data["ml_probability"], 33)
    upper_threshold = np.percentile(train_data["ml_probability"], 66)

    return lower_threshold, upper_threshold


# ia probabilitatea de la model si o transforma in semnal de trading
def probability_to_signal(probability, lower_threshold, upper_threshold):
    # 0 = bearish, 1 = neutral, 2 = bullish
    if probability < lower_threshold:
        return 0

    if probability > upper_threshold:
        return 2

    return 1


def adauga_ml_signal(data, lower_threshold, upper_threshold):
    data = data.copy()

    # adauga o noua coloana cu semnalul de trading calculat pe baza probabilitatii
    data["ml_signal"] = data["ml_probability"].apply(
        lambda probability: probability_to_signal(
            probability,
            lower_threshold,
            upper_threshold,
        )
    )

    return data
