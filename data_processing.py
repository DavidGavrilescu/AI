import yfinance as yf

from config import END_DATE, START_DATE, TICKER, TREND_THRESHOLD


def descarca_date():
    # luam datele din yahoo finance
    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    return data.reset_index()


def adauga_featureuri(data):
    data = data.copy()

    # feature-uri din pret si volum
    data["daily_return"] = data["Close"].pct_change()
    data["open_close_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volume_change"] = data["Volume"].pct_change()

    # moving average pentru trends
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()
    data["ma_ratio"] = data["ma_5"] / data["ma_20"] - 1

    return data


def adauga_randamente_viitoare(data):
    data = data.copy()

    # nu vor fi date ca input modelului, sunt doar pentru label/analiza
    data["future_return"] = data["Close"].shift(-1) / data["Close"] - 1
    data["future_return_5d"] = data["Close"].shift(-5) / data["Close"] - 1

    return data


def adauga_label(data):
    data = data.copy()

    # 1 inseamna ca pretul creste maine, 0 inseamna ca nu creste
    data["label"] = (data["future_return"] > 0).astype(int)

    return data


def adauga_trend(data):
    data = data.copy()

    # 0 = descendent, 1 = neutru, 2 = ascendent
    data["trend"] = 1
    data.loc[data["ma_ratio"] > TREND_THRESHOLD, "trend"] = 2
    data.loc[data["ma_ratio"] < -TREND_THRESHOLD, "trend"] = 0

    return data


def curata_date(data):
    # primele randuri nu au moving average, iar ultimele nu au randamente viitoare.
    return data.dropna().reset_index(drop=True)


def pregateste_date():
    data = descarca_date()
    data = adauga_featureuri(data)
    data = adauga_randamente_viitoare(data)
    data = adauga_label(data)
    data = adauga_trend(data)
    data = curata_date(data)

    return data
