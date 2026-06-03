import os

import pandas as pd
import yfinance as yf

from config import DATA_FILE, END_DATE, START_DATE, TICKER, TREND_THRESHOLD


def incarca_date_spy():
    # folosim aceleasi date la fiecare rulare, ca rezultatele sa fie stabile
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, parse_dates=["Date"])

    date = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    date = date.reset_index()
    date.to_csv(DATA_FILE, index=False)

    return date


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

        Coloane folosite pentru ML label si RL reward:
        - trend: 0 = descendent, 1 = neutru, 2 = ascendent, calculat din ma_ratio
        - future_return: randamentul zilei urmatoare
        - future_return_5d: randamentul peste 5 zile, folosit pentru label-ul ML si reward-ul RL

        Valorile future_return si future_return_5d sunt folosite doar ca target/reward, nu ca feature-uri de intrare pentru model.
    """
    date["daily_return"] = date["Close"].pct_change()
    date["open_close_return"] = (date["Close"] - date["Open"]) / date["Open"]
    date["high_low_range"] = (date["High"] - date["Low"]) / date["Close"]
    date["volume_change"] = date["Volume"].pct_change()

    # moving average pe 5 si 20 de zile, folosita pentru trend
    date["ma_5"] = date["Close"].rolling(window=5).mean()
    date["ma_20"] = date["Close"].rolling(window=20).mean()
    date["ma_ratio"] = date["ma_5"] / date["ma_20"] - 1

    # randamente viitoare, folosite la label si reward
    date["future_return"] = date["Close"].shift(-1) / date["Close"] - 1
    date["future_return_5d"] = date["Close"].shift(-5) / date["Close"] - 1

    # 0 = descendent, 1 = neutru, 2 = ascendent
    date["trend"] = 1
    date.loc[date["ma_ratio"] > TREND_THRESHOLD, "trend"] = 2
    date.loc[date["ma_ratio"] < -TREND_THRESHOLD, "trend"] = 0

    # primele randuri nu au medii mobile, ultimele nu au randamente viitoare
    date = date.dropna().reset_index(drop=True)

    return date
