import yfinance as yf

from config import END_DATE, START_DATE, TICKER, TREND_THRESHOLD


def pregateste_date():
    # luam datele din yahoo finance
    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    data = data.reset_index()

    # feature-uri din pret si volum
    data["daily_return"] = data["Close"].pct_change()
    data["open_close_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volume_change"] = data["Volume"].pct_change()

    # moving average pentru trend
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()
    data["ma_ratio"] = data["ma_5"] / data["ma_20"] - 1

    # randamente viitoare, folosite doar pentru target/analiza
    data["future_return"] = data["Close"].shift(-1) / data["Close"] - 1
    data["future_return_5d"] = data["Close"].shift(-5) / data["Close"] - 1

    # 0 = descendent, 1 = neutru, 2 = ascendent
    data["trend"] = 1
    data.loc[data["ma_ratio"] > TREND_THRESHOLD, "trend"] = 2
    data.loc[data["ma_ratio"] < -TREND_THRESHOLD, "trend"] = 0

    # primele randuri nu au moving average, iar ultimele nu au randamente viitoare
    data = data.dropna().reset_index(drop=True)

    return data
