import pandas as pd
import yfinance as yf


# proiectul foloseste SPY, un ETF care urmareste indicele S&P500
# perioada 2010-2025 ca sa avem destule date pentru analiza
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # data de final din yfinance este exclusiva - adica nu include ziua asta

# pragul de 0.5% este folosit ca sa nu consideram orice diferenta mica drept trend.
TREND_THRESHOLD = 0.005

DISPLAY_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "daily_return",
    "open_close_return",
    "high_low_range",
    "volume_change",
    "ma_5",
    "ma_20",
    "ma_ratio",
    "future_return",
    "future_return_5d",
    "label",
    "trend",
]


def download_stock_data():
    """Descarca datele istorice pentru tickerul ales"""

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    return data.reset_index()


def add_price_features(data):
    """Construieste indicatori simpli plecand de la pret si volum"""

    data = data.copy()

    # randamentul zilnic arata cu cat s-a modificat pretul fata de ziua anterioara
    data["daily_return"] = data["Close"].pct_change()

    # descriu miscarea din interiorul aceleiasi zile
    data["open_close_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]

    # volumul poate da indicii despre interesul pietei
    data["volume_change"] = data["Volume"].pct_change() # procent de schimbare al volumului fata de ziua anterioara

    # moving average ajuta la identificarea directiei generale a pretului
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()
    data["ma_ratio"] = data["ma_5"] / data["ma_20"] - 1

    return data


def add_future_returns(data):
    """Adauga randamente viitoare, folosite pentru etichete si analiza"""

    data = data.copy()

    # shift(-1) pune pretul de maine pe linia de azi
    # asa putem calcula ce s-a intamplat dupa ziua curenta
    data["future_return"] = data["Close"].shift(-1) / data["Close"] - 1

    # randamentul pe 5 zile
    data["future_return_5d"] = data["Close"].shift(-5) / data["Close"] - 1

    return data


def add_label(data):
    """Creeaza tinta de predictie: 1 daca pretul creste maine, 0 altfel"""

    data = data.copy()
    data["label"] = (data["future_return"] > 0).astype(int)
    return data


def add_trend(data):
    """Clasifica trendul in descendent, neutru sau ascendent"""

    data = data.copy()

    # 0 = trend descendent, 1 = neutru, 2 = trend ascendent
    data["trend"] = 1
    data.loc[data["ma_ratio"] > TREND_THRESHOLD, "trend"] = 2
    data.loc[data["ma_ratio"] < -TREND_THRESHOLD, "trend"] = 0

    return data


def prepare_dataset():
    """Ruleaza toti pasii de pregatire a datelor"""

    data = download_stock_data()
    data = add_price_features(data)
    data = add_future_returns(data)
    data = add_label(data)
    data = add_trend(data)

    # primele randuri nu au medii mobile, iar ultimele nu au randamente viitoare
    # le eliminam ca sa ramanem doar cu randuri complete
    return data.dropna().reset_index(drop=True)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    data = prepare_dataset()

    print(f"Ticker analizat: {TICKER}")
    print(f"Numar de randuri dupa curatare: {len(data)}")
    print("\nPrimele 3 randuri cu feature-urile construite:")
    print(data[DISPLAY_COLUMNS].head(3).round(2).to_string(index=False))

if __name__ == "__main__":
    main()
