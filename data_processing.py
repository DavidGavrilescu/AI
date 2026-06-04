import pandas as pd

from config import DATA_FILE, END_DATE, START_DATE, TICKER, TREND_THRESHOLD


def incarca_date_spy():
    # folosim CSV-ul local Kaggle, ca rezultatele sa fie reproductibile
    date = pd.read_csv(DATA_FILE)
    date.columns = [coloana.strip().lower() for coloana in date.columns]

    coloane_necesare = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    }
    coloane_lipsa = coloane_necesare - set(date.columns)
    if coloane_lipsa:
        raise ValueError(
            f"{DATA_FILE} nu are coloanele necesare: {sorted(coloane_lipsa)}"
        )

    date["Date"] = pd.to_datetime(date["date"])
    data_inceput = pd.Timestamp(START_DATE)
    data_final = pd.Timestamp(END_DATE)
    date = date[(date["Date"] >= data_inceput) & (date["Date"] < data_final)].copy()
    date = date.sort_values("Date").reset_index(drop=True)

    factor_ajustare = date["adjusted_close"] / date["close"]
    date_pregatite = pd.DataFrame({
        "Date": date["Date"],
        "Open": date["open"] * factor_ajustare,
        "High": date["high"] * factor_ajustare,
        "Low": date["low"] * factor_ajustare,
        "Close": date["adjusted_close"],
        "Volume": date["volume"],
    })

    return date_pregatite


def pregateste_date():
    date = incarca_date_spy()

    """
        Construim feature-uri din pret si volum.

        Feature-uri folosite de Logistic Regression:
        - daily_return: randamentul fata de ziua precedenta
        - open_close_return: miscarea din aceeasi zi, de la Open la Close
        - high_low_range: intervalul High-Low raportat la Close, ca masura simpla de volatilitate
        - volume_change: schimbarea volumului fata de ziua precedenta
        - ma_ratio: diferenta relativa dintre media mobila pe 5 zile si cea pe 20 de zile

        Coloane folosite pentru ML si RL:
        - trend: 0 = descendent, 1 = neutru, 2 = ascendent, folosit in starea RL
        - future_return_5d: randamentul peste 5 zile, folosit pentru label-ul ML

        future_return_5d este folosit doar ca target, nu ca feature de intrare pentru model.
        Reward-ul RL se calculeaza separat din randamentul zilnic, ca in benchmark.
    """
    date["daily_return"] = date["Close"].pct_change()
    date["open_close_return"] = (date["Close"] - date["Open"]) / date["Open"]
    date["high_low_range"] = (date["High"] - date["Low"]) / date["Close"]
    date["volume_change"] = date["Volume"].pct_change()

    # moving average pe 5 si 20 de zile, folosita pentru trend
    date["ma_5"] = date["Close"].rolling(window=5).mean()
    date["ma_20"] = date["Close"].rolling(window=20).mean()
    date["ma_ratio"] = date["ma_5"] / date["ma_20"] - 1

    # randamentul viitor pe 5 zile, folosit la label si reward
    date["future_return_5d"] = date["Close"].shift(-5) / date["Close"] - 1

    # 0 = descendent, 1 = neutru, 2 = ascendent
    date["trend"] = 1
    date.loc[date["ma_ratio"] > TREND_THRESHOLD, "trend"] = 2
    date.loc[date["ma_ratio"] < -TREND_THRESHOLD, "trend"] = 0

    # primele randuri nu au medii mobile, ultimele nu au randamente viitoare
    date = date.dropna().reset_index(drop=True)

    return date
