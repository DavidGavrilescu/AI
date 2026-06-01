# date generale ale proiectului
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # yfinance nu include data de final

TRAIN_RATIO = 0.70
TREND_THRESHOLD = 0.005

# o sa fie folosite la Logistuic regression nu includem future_return, future_return_5d sau label, pentru ca tin de viitor
FEATURE_COLUMNS = [
    "daily_return",
    "open_close_return",
    "high_low_range",
    "volume_change",
    "ma_ratio",
]
