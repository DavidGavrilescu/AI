from config import SIGNAL_MARGIN, THRESHOLD_PREDICTIE


def adauga_ml_signal(date):
    date = date.copy()

    prag_jos = THRESHOLD_PREDICTIE - SIGNAL_MARGIN
    prag_sus = THRESHOLD_PREDICTIE + SIGNAL_MARGIN

    # 0 = sub normal, 1 = incert, 2 = peste normal
    date["ml_signal"] = 1
    date.loc[date["ml_probability"] < prag_jos, "ml_signal"] = 0
    date.loc[date["ml_probability"] > prag_sus, "ml_signal"] = 2

    return date
