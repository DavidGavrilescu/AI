from config import SIGNAL_MARGIN, THRESHOLD_PREDICTIE


def adauga_ml_signal(data):
    data = data.copy()

    lower_threshold = THRESHOLD_PREDICTIE - SIGNAL_MARGIN
    upper_threshold = THRESHOLD_PREDICTIE + SIGNAL_MARGIN

    # 0 = sub normal, 1 = incert, 2 = peste normal
    data["ml_signal"] = 1
    data.loc[data["ml_probability"] < lower_threshold, "ml_signal"] = 0
    data.loc[data["ml_probability"] > upper_threshold, "ml_signal"] = 2

    return data
